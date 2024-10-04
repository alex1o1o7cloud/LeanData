import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Divisibility.Basic
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Function
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Pow
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.LinearAlgebra.Basic
import Mathlib.LinearAlgebra.Matrix
import Mathlib.Logarithm
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith

namespace primes_x_y_eq_l283_283302

theorem primes_x_y_eq 
  {p q x y : ℕ} (hp : Nat.Prime p) (hq : Nat.Prime q)
  (hx : 0 < x) (hy : 0 < y)
  (hp_lt_x : x < p) (hq_lt_y : y < q)
  (h : (p : ℚ) / x + (q : ℚ) / y = (p * y + q * x) / (x * y)) :
  x = y :=
sorry

end primes_x_y_eq_l283_283302


namespace suitable_b_values_l283_283232

theorem suitable_b_values (b : ℤ) :
  (∃ (c d e f : ℤ), 35 * c * d + (c * f + d * e) * b + 35 = 0 ∧
    c * e = 35 ∧ d * f = 35) →
  (∃ (k : ℤ), b = 2 * k) :=
by
  intro h
  sorry

end suitable_b_values_l283_283232


namespace largest_prime_factor_3136_l283_283531

theorem largest_prime_factor_3136 : ∀ (n : ℕ), n = 3136 → ∃ p : ℕ, Prime p ∧ (p ∣ n) ∧ ∀ q : ℕ, (Prime q ∧ q ∣ n) → p ≥ q :=
by {
  sorry
}

end largest_prime_factor_3136_l283_283531


namespace find_a_if_even_function_l283_283951

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x * exp x / (exp (a * x) - 1)

theorem find_a_if_even_function :
  (∀ x : ℝ, x ≠ 0 → f a (-x) = f a x) → a = 2 :=
by
  sorry

end find_a_if_even_function_l283_283951


namespace amount_paid_l283_283385

-- Define conditions
def dozen := 12
def roses_bought := 5 * dozen
def cost_per_rose := 6
def discount := 0.80

-- Define theorem
theorem amount_paid : 
  roses_bought * cost_per_rose * discount = 288 :=
by sorry

end amount_paid_l283_283385


namespace radius_excircle_ABC_l283_283043

variables (A B C P Q : Point)
variables (r_ABP r_APQ r_AQC : ℝ) (re_ABP re_APQ re_AQC : ℝ)
variable (r_ABC : ℝ)

-- Conditions
-- Radii of the incircles of triangles ABP, APQ, and AQC are all equal to 1
axiom incircle_ABP : r_ABP = 1
axiom incircle_APQ : r_APQ = 1
axiom incircle_AQC : r_AQC = 1

-- Radii of the corresponding excircles opposite A for ABP, APQ, and AQC are 3, 6, and 5 respectively
axiom excircle_ABP : re_ABP = 3
axiom excircle_APQ : re_APQ = 6
axiom excircle_AQC : re_AQC = 5

-- Radius of the incircle of triangle ABC is 3/2
axiom incircle_ABC : r_ABC = 3 / 2

-- Theorem stating the radius of the excircle of triangle ABC opposite A is 135
theorem radius_excircle_ABC (r_ABC : ℝ) : r_ABC = 3 / 2 → ∀ (re_ABC : ℝ), re_ABC = 135 := 
by
  intros 
  sorry

end radius_excircle_ABC_l283_283043


namespace Dawn_hourly_earnings_l283_283379

theorem Dawn_hourly_earnings :
  let t_per_painting := 2 
  let num_paintings := 12
  let total_earnings := 3600
  let total_time := t_per_painting * num_paintings
  let hourly_wage := total_earnings / total_time
  hourly_wage = 150 := by
  sorry

end Dawn_hourly_earnings_l283_283379


namespace solve_inequality_l283_283701

theorem solve_inequality (x : ℝ) : 2 * x^2 + 8 * x ≤ -6 ↔ -3 ≤ x ∧ x ≤ -1 :=
by
  sorry

end solve_inequality_l283_283701


namespace smallest_absolute_value_rational_is_zero_l283_283110

theorem smallest_absolute_value_rational_is_zero : ∃ (r : ℚ), (∀ (q : ℚ), |q| ≥ |r|) ∧ r = 0 := 
begin
  use 0,
  split,
  { intro q,
    exact abs_nonneg q },
  refl,
end

end smallest_absolute_value_rational_is_zero_l283_283110


namespace ralph_lost_peanuts_l283_283059

theorem ralph_lost_peanuts :
  ∀ (a b l : ℕ), a = 74 → b = 15 → l = a - b → l = 59 :=
by
  intros a b l ha hb hl
  rw [ha, hb, hl]
  rfl
  
sorry

end ralph_lost_peanuts_l283_283059


namespace find_a_if_f_even_l283_283838

def f (a x : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem find_a_if_f_even (a : ℝ) (h : ∀ x : ℝ, x ≠ 0 → f a (-x) = f a x) : a = 2 :=
by 
  sorry

end find_a_if_f_even_l283_283838


namespace no_other_way_to_construct_five_triangles_l283_283734

-- Setting up the problem conditions
def five_triangles_from_fifteen_sticks (sticks : List ℕ) : Prop :=
  sticks.length = 15 ∧
  let split_sticks := sticks.chunk (3) in
  split_sticks.length = 5 ∧
  ∀ triplet ∈ split_sticks, 
    let a := triplet[0]
    let b := triplet[1]
    let c := triplet[2] in
    a + b > c ∧ a + c > b ∧ b + c > a

-- The proof problem statement
theorem no_other_way_to_construct_five_triangles (sticks : List ℕ) :
  five_triangles_from_fifteen_sticks sticks → 
  ∀ (alternative : List (List ℕ)), 
    (alternative.length = 5 ∧ 
    ∀ triplet ∈ alternative, triplet.length = 3 ∧
      let a := triplet[0]
      let b := triplet[1]
      let c := triplet[2] in
      a + b > c ∧ a + c > b ∧ b + c > a) →
    alternative = sticks.chunk (3) :=
by
  sorry

end no_other_way_to_construct_five_triangles_l283_283734


namespace even_function_implies_a_eq_2_l283_283901

def f (a x : ℝ) : ℝ := (x * exp x) / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2
  (h : ∀ x : ℝ, f a x = f a (-x)) : a = 2 := 
sorry

end even_function_implies_a_eq_2_l283_283901


namespace reporters_covering_local_politics_l283_283176

theorem reporters_covering_local_politics
  (total_reporters : ℝ)
  (politics_not_local_x_percentage : ℝ)
  (reporters_non_politics_percentage : ℝ)
  (h1 : politics_not_local_x_percentage = 0.30)
  (h2 : (100 - reporters_non_politics_percentage) = 14.28571428571428) :
  let covering_politics := 100 - reporters_non_politics_percentage in
  let local_politics_percentage := 0.7 * covering_politics in
  local_politics_percentage = 10 := sorry

end reporters_covering_local_politics_l283_283176


namespace tetrahedron_volume_l283_283082

noncomputable def volume_of_tetrahedron (P Q R S : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  (1 / 6) * real.abs (euclidean_inner_product_space.volume {P, Q, R, S} - {P, Q, R})

theorem tetrahedron_volume
  (P Q R S : EuclideanSpace ℝ (Fin 3))
  (hPQ : dist P Q = 6)
  (hPR : dist P R = 4)
  (hPS : dist P S = 5)
  (hQR : dist Q R = 5)
  (hQS : dist Q S = 3)
  (hRS : dist R S = (15 / 4) * real.sqrt 2) :
  volume_of_tetrahedron P Q R S = (15 * real.sqrt 2) / 2 :=
sorry

end tetrahedron_volume_l283_283082


namespace miscalculation_amount_l283_283350

theorem miscalculation_amount (a b : ℕ) (h : |(9 * (a - b))| = 9) : 9 * |a - b| + 8 = 17 := by 
  sorry

end miscalculation_amount_l283_283350


namespace simplify_fraction_l283_283065

theorem simplify_fraction (a b c d k : ℕ) (h₁ : a = 123) (h₂ : b = 9999) (h₃ : k = 41)
                           (h₄ : c = a / 3) (h₅ : d = b / 3)
                           (h₆ : c = k) (h₇ : d = 3333) :
  (a * k) / b = (k^2) / d :=
by
  sorry

end simplify_fraction_l283_283065


namespace factorize_x_cube_minus_4x_l283_283243

theorem factorize_x_cube_minus_4x (x : ℝ) : x^3 - 4 * x = x * (x + 2) * (x - 2) := 
by
  -- Continue the proof from here
  sorry

end factorize_x_cube_minus_4x_l283_283243


namespace dawns_earnings_per_hour_l283_283380

variable (hours_per_painting : ℕ) (num_paintings : ℕ) (total_earnings : ℕ)

def total_hours (hours_per_painting num_paintings : ℕ) : ℕ :=
  hours_per_painting * num_paintings

def earnings_per_hour (total_earnings total_hours : ℕ) : ℕ :=
  total_earnings / total_hours

theorem dawns_earnings_per_hour :
  hours_per_painting = 2 →
  num_paintings = 12 →
  total_earnings = 3600 →
  earnings_per_hour total_earnings (total_hours hours_per_painting num_paintings) = 150 :=
by
  intros h1 h2 h3
  sorry

end dawns_earnings_per_hour_l283_283380


namespace min_value_eval_l283_283410

noncomputable def min_value_expr (x y : ℝ) := 
  (x + 1/y) * (x + 1/y - 100) + (y + 1/x) * (y + 1/x - 100)

theorem min_value_eval (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  x = y → min_value_expr x y = -2500 :=
by
  intros hxy
  -- Insert proof steps here
  sorry

end min_value_eval_l283_283410


namespace triangle_angles_inequality_l283_283373

theorem triangle_angles_inequality (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C) (h_sum : A + B + C = Real.pi) :
  1 / Real.sin (A / 2) + 1 / Real.sin (B / 2) + 1 / Real.sin (C / 2) ≥ 6 := 
sorry

end triangle_angles_inequality_l283_283373


namespace even_function_implies_a_eq_2_l283_283861

theorem even_function_implies_a_eq_2 (a : ℝ) 
  (h : ∀ x : ℝ, f x = f (-x)) 
  (f : ℝ → ℝ := λ x, (x * exp x) / (exp (a * x) - 1)) : a = 2 :=
sorry

end even_function_implies_a_eq_2_l283_283861


namespace regular_polygon_sides_l283_283609

theorem regular_polygon_sides (theta : ℝ) (h : theta = 18) : 
  ∃ n : ℕ, 360 / theta = n ∧ n = 20 :=
by
  use 20
  constructor
  apply eq_of_div_eq_mul
  rw h
  norm_num
  sorry

end regular_polygon_sides_l283_283609


namespace even_function_implies_a_eq_2_l283_283867

def f (x a : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2 (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → a = 2 :=
by
  sorry

end even_function_implies_a_eq_2_l283_283867


namespace no_other_way_five_triangles_l283_283706

def is_triangle {α : Type*} [linear_ordered_field α] (a b c : α) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem no_other_way_five_triangles (sticks : list ℝ)
  (h_len : sticks.length = 15)
  (h_sets : ∃ (sets : list (ℝ × ℝ × ℝ)), 
    sets.length = 5 ∧ ∀ {a b c}, (a, b, c) ∈ sets → is_triangle a b c) :
  ∀ sets, sets.length = 5 ∧ ∀ {a b c}, (a, b, c) ∈ sets → is_triangle a b c → sets = h_sets :=
begin
  sorry
end

end no_other_way_five_triangles_l283_283706


namespace even_function_implies_a_eq_2_l283_283896

def f (a x : ℝ) : ℝ := (x * exp x) / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2
  (h : ∀ x : ℝ, f a x = f a (-x)) : a = 2 := 
sorry

end even_function_implies_a_eq_2_l283_283896


namespace find_a_if_even_function_l283_283945

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x * exp x / (exp (a * x) - 1)

theorem find_a_if_even_function :
  (∀ x : ℝ, x ≠ 0 → f a (-x) = f a x) → a = 2 :=
by
  sorry

end find_a_if_even_function_l283_283945


namespace min_value_inequality_l283_283325

theorem min_value_inequality (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 3 * a + 2 * b = 1) : 
  ∃ (m : ℝ), m = 25 ∧ (∀ x y, (x > 0) → (y > 0) → (3 * x + 2 * y = 1) → (3 / x + 2 / y) ≥ m) :=
sorry

end min_value_inequality_l283_283325


namespace total_time_correct_l283_283425

-- Define the individual times
def driving_time_one_way : ℕ := 20
def attending_time : ℕ := 70

-- Define the total driving time as twice the one-way driving time
def total_driving_time : ℕ := driving_time_one_way * 2

-- Define the total time as the sum of total driving time and attending time
def total_time : ℕ := total_driving_time + attending_time

-- Prove that the total time is 110 minutes
theorem total_time_correct : total_time = 110 := by
  -- The proof is omitted, we're only interested in the statement format.
  sorry

end total_time_correct_l283_283425


namespace no_other_way_to_construct_five_triangles_l283_283757

open Classical

theorem no_other_way_to_construct_five_triangles {
  sticks : List ℤ 
} (h_stick_lengths : 
  sticks = [2, 3, 4, 20, 30, 40, 200, 300, 400, 2000, 3000, 4000, 20000, 30000, 40000]) 
  (h_five_triplets : 
  ∀ t ∈ (sticks.nthLe 0 15).combinations 3, 
  (t[0] + t[1] > t[2]) ∧ (t[1] + t[2] > t[3]) ∧ (t[0] + t[2] > t[3])) :
  ¬ ∃ other_triplets, 
  (∀ t ∈ other_triplets, t.length = 3) ∧ 
  (∀ t ∈ other_triplets, (t[0] + t[1] > t[2]) ∧ (t[1] + t[2] > t[0]) ∧ (t[0] + t[2] > t[1])) ∧ 
  other_triplets ≠ (sticks.nthLe 0 15).combinations 3 := 
by
  sorry

end no_other_way_to_construct_five_triangles_l283_283757


namespace paths_from_A_to_D_l283_283664

-- Defining points A, B, C, D
inductive Point
| A
| B
| C
| D
open Point

-- Defining the conditions of the problem
def ways (p1 p2 : Point) : ℕ :=
  match (p1, p2) with
  | (A, B) => 2
  | (B, C) => 2
  | (C, D) => 2
  | (A, D) => 1
  | _ => 0

-- Proving the number of different paths from A to D
theorem paths_from_A_to_D : 
  (ways A B) * (ways B C) * (ways C D) + (ways A D) = 9 := by
  -- Noncomputable section to acknowledge possibility of symbolic computation
  noncomputable theory
  -- Returning the correct answer
  sorry

end paths_from_A_to_D_l283_283664


namespace find_a_even_function_l283_283981

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def given_function (a : ℝ) (x : ℝ) : ℝ :=
  x * Real.exp x / (Real.exp (a * x) - 1)

theorem find_a_even_function :
  (is_even_function (given_function 2)) → true :=
by
  intro h
  sorry

end find_a_even_function_l283_283981


namespace solve_system_a_l283_283072

theorem solve_system_a (x y : ℝ) (h1 : x^2 - 3 * x * y - 4 * y^2 = 0) (h2 : x^3 + y^3 = 65) : 
    x = 4 ∧ y = 1 :=
sorry

end solve_system_a_l283_283072


namespace amount_paid_l283_283384

-- Define conditions
def dozen := 12
def roses_bought := 5 * dozen
def cost_per_rose := 6
def discount := 0.80

-- Define theorem
theorem amount_paid : 
  roses_bought * cost_per_rose * discount = 288 :=
by sorry

end amount_paid_l283_283384


namespace intersection_M_N_l283_283047

theorem intersection_M_N (x y : ℝ) :
  (x^2 + y^2 = 2 ∧ y = x^2) ↔ (0 ≤ x ∧ x <= √2) :=
by
  sorry

end intersection_M_N_l283_283047


namespace algebra_inequality_l283_283454

variable {x y z : ℝ}

theorem algebra_inequality
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  x^3 * (y^2 + z^2)^2 + y^3 * (z^2 + x^2)^2 + z^3 * (x^2 + y^2)^2
  ≥ x * y * z * (x * y * (x + y)^2 + y * z * (y + z)^2 + z * x * (z + x)^2) :=
sorry

end algebra_inequality_l283_283454


namespace shelby_gold_stars_l283_283063

theorem shelby_gold_stars (stars_yesterday stars_today : ℕ) (h1 : stars_yesterday = 4) (h2 : stars_today = 3) :
  stars_yesterday + stars_today = 7 := 
by
  sorry

end shelby_gold_stars_l283_283063


namespace factorize_x_l283_283259

theorem factorize_x^3_minus_4x (x : ℝ) : x^3 - 4 * x = x * (x + 2) * (x - 2) := 
sorry

end factorize_x_l283_283259


namespace largest_prime_factor_3136_l283_283524

theorem largest_prime_factor_3136 : ∃ p, nat.prime p ∧ p ∣ 3136 ∧ (∀ q, nat.prime q ∧ q ∣ 3136 → q ≤ p) :=
by {
  sorry
}

end largest_prime_factor_3136_l283_283524


namespace no_alternative_way_to_construct_triangles_l283_283720

-- Define the triangle inequality condition
def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the set of stick lengths for each required triangle
def valid_triangle_sets (stick_set : Set (Set ℕ)) : Prop :=
  stick_set = { {2, 3, 4}, {20, 30, 40}, {200, 300, 400}, {2000, 3000, 4000}, {20000, 30000, 40000} }

-- Define the problem statement
theorem no_alternative_way_to_construct_triangles (s : Set ℕ) (h : s.card = 15) :
  (∀ t ∈ (s.powerset.filter (λ t, t.card = 3)), triangle_inequality t.some (finset_some_spec t) ∧ (∃ sets : Set (Set ℕ), sets.card = 5 ∧ ∀ u ∈ sets, u.card = 3 ∧ triangle_inequality u.some (finset_some_spec u)))
  → valid_triangle_sets (s.powerset.filter (λ t, t.card = 3)) := 
sorry

end no_alternative_way_to_construct_triangles_l283_283720


namespace factorize_x_cube_minus_4x_l283_283245

theorem factorize_x_cube_minus_4x (x : ℝ) : x^3 - 4 * x = x * (x + 2) * (x - 2) := 
by
  -- Continue the proof from here
  sorry

end factorize_x_cube_minus_4x_l283_283245


namespace meeting_time_final_time_statement_l283_283216

-- Define the speeds and distance as given conditions
def brodie_speed : ℝ := 50
def ryan_speed : ℝ := 40
def initial_distance : ℝ := 120

-- Define what we know about their meeting time and validate it mathematically
theorem meeting_time :
  (initial_distance / (brodie_speed + ryan_speed)) = 4 / 3 := sorry

-- Assert the time in minutes for completeness
noncomputable def time_in_minutes : ℝ := ((4 / 3) * 60)

-- Assert final statement matching the answer in hours and minutes
theorem final_time_statement :
  time_in_minutes = 80 := sorry

end meeting_time_final_time_statement_l283_283216


namespace common_sales_days_in_july_l283_283173

/-- Prove that the number of common sales days in July for both stores is 1. --/
theorem common_sales_days_in_july :
  (finset.range (31)).filter (λ d, (d + 1) % 7 = 0 ∧ (d - 3) % 5 = 0).card = 1 :=
begin
  -- This is where the proof would be constructed.
  sorry,
end

end common_sales_days_in_july_l283_283173


namespace inequality_solution_l283_283070

noncomputable def u (x : ℝ) : ℝ := 1 + x^2
noncomputable def v (x : ℝ) : ℝ := 1 - 3 * x^2 + 16 * x^4
noncomputable def w (x : ℝ) : ℝ := 1 + 8 * x^5

theorem inequality_solution (x : ℝ) : 
  log (u x) (w x) + log (v x) (u x) ≤ 1 + log (v x) (w x) ↔ 
  x ∈ set.Icc (-1/ (8 : ℝ)^ (1/5)) (-1/2) ∪ 
  set.Ioo (- (3 : ℝ) ^ (1/2)/ 4) 0 ∪ 
  set.Ioo 0 ((3 : ℝ)^ (1/2) / 4) ∪ 
  set.Icc (1 / 2) (1 / 2) :=
sorry

end inequality_solution_l283_283070


namespace max_digit_product_l283_283396

theorem max_digit_product (N : ℕ) (digits : List ℕ) (h1 : 0 < N) (h2 : digits.sum = 23) (h3 : digits.prod < 433) : 
  digits.prod ≤ 432 :=
sorry

end max_digit_product_l283_283396


namespace first_player_wins_l283_283184

theorem first_player_wins (n : ℕ) (h : n ≥ 3) : 
  (∃ k, n = 2 * k) ∨ (n = 3) ↔ first_player_wins :=
sorry

end first_player_wins_l283_283184


namespace boys_and_girls_arrangement_l283_283004

theorem boys_and_girls_arrangement (boys girls : ℕ) (h_boys : boys = 4) (h_girls : girls = 3) :
  (∃ arrangements : ℕ, arrangements = (fact girls) * (fact boys) ∧ arrangements = 144) :=
by
  sorry

end boys_and_girls_arrangement_l283_283004


namespace chips_recoloring_impossible_l283_283125

theorem chips_recoloring_impossible :
  (∀ a b c : ℕ, a = 2008 ∧ b = 2009 ∧ c = 2010 →
   ¬(∃ k : ℕ, a + b + c = k ∧ (a = k ∨ b = k ∨ c = k))) :=
by sorry

end chips_recoloring_impossible_l283_283125


namespace probability_of_square_root_less_than_seven_is_13_over_30_l283_283537

-- Definition of two-digit range and condition for square root check
def two_digit_numbers := Finset.range 100 \ Finset.range 10
def sqrt_condition (n : ℕ) : Prop := n < 49

-- The required probability calculation
def probability_square_root_less_than_seven : ℚ :=
  (↑(two_digit_numbers.filter sqrt_condition).card) / (↑two_digit_numbers.card)

-- The theorem stating the required probability
theorem probability_of_square_root_less_than_seven_is_13_over_30 :
  probability_square_root_less_than_seven = 13 / 30 := by
  sorry

end probability_of_square_root_less_than_seven_is_13_over_30_l283_283537


namespace no_alternative_way_to_construct_triangles_l283_283713

theorem no_alternative_way_to_construct_triangles (sticks : list ℕ) (h_len : sticks.length = 15)
  (h_set : ∃ (sets : list (list ℕ)), sets.length = 5 ∧ ∀ set ∈ sets, set.length = 3 
           ∧ (∀ (a b c : ℕ) (ha : a ∈ set) (hb : b ∈ set) (hc : c ∈ set), 
             (a + b > c ∧ a + c > b ∧ b + c > a))):
  ¬∃ (alt_sets : list (list ℕ)), alt_sets ≠ sets ∧ alt_sets.length = 5 
      ∧ ∀ set ∈ alt_sets, set.length = 3 
      ∧ (∀ (a b c : ℕ) (ha : a ∈ set) (hb : b ∈ set) (hc : c ∈ set),
        (a + b > c ∧ a + c > b ∧ b + c > a)) :=
sorry

end no_alternative_way_to_construct_triangles_l283_283713


namespace even_function_implies_a_eq_2_l283_283925

def f (x : ℝ) (a : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2 (a : ℝ) (h_even : ∀ x, f (-x) a = f x a) : a = 2 :=
by
  intro x
  sorry

end even_function_implies_a_eq_2_l283_283925


namespace random_two_digit_sqrt_prob_lt_seven_l283_283546

theorem random_two_digit_sqrt_prob_lt_seven :
  let total_count := 90 in
  let count_lt_sqrt7 := 48 - 10 + 1 in
  (count_lt_sqrt7 : ℚ) / total_count = 13 / 30 :=
by
  let total_count := 90
  let count_lt_sqrt7 := 48 - 10 + 1
  have h1 : count_lt_sqrt7 = 39 := by linarith
  have h2 : (count_lt_sqrt7 : ℚ) / total_count = (39 : ℚ) / 90 := by rw h1
  have h3 : (39 : ℚ) / 90 = 13 / 30 := by norm_num
  rw [h2, h3]
  refl

end random_two_digit_sqrt_prob_lt_seven_l283_283546


namespace tangent_line_eqn_possible_theta_range_of_m_l283_283315

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * x - (m - 1) / x - Real.log x
noncomputable def g (theta : ℝ) (x : ℝ) : ℝ := 1 / (x * Real.cos theta) + Real.log x
noncomputable def h (m : ℝ) (theta : ℝ) (x : ℝ) : ℝ := f m x - g theta x

-- 1. Prove the equation of the tangent line to f(x) at P(1, f(1)) when m = 3 is 4x - y - 3 = 0.
theorem tangent_line_eqn (x y : ℝ) (hx : x = 1) (hy : y = f 3 1) : 4 * x - y - 3 = 0 := 
sorry

-- 2. Prove the possible value of theta is 0.
theorem possible_theta (theta : ℝ) (h_increasing : ∀ (x : ℝ), 1 ≤ x → 0 ≤ Real.cos theta - x⁻¹ + 1 / x):
  theta = 0 := 
sorry

-- 3. Prove the range of possible values for m if h(x) = f(x) - g(x) is monotonic on its domain.
-- The range is (-∞, 0] ∪ [1, +∞).
theorem range_of_m (m : ℝ) (theta : ℝ):
  (∀ x : ℝ, 0 < x → 1 / x < m * x^2 - 2 * x + m) ∨ (∀ x : ℝ, 0 < x → m * x^2 - 2 * x + m < 1 / x) →
  m ∈ (-∞ : ℝ, 0] ∪ [1, ∞) := 
sorry

end tangent_line_eqn_possible_theta_range_of_m_l283_283315


namespace range_of_a_symmetric_points_l283_283319

theorem range_of_a_symmetric_points :
  ∀ (a : ℝ), (∃ x : ℝ, (1 / Real.exp 1 ≤ x ∧ x ≤ Real.exp 1) ∧ a = x^2 - 2 * Real.log x ↔ 1 ≤ a ∧ a ≤ Real.exp 2 - 2) :=
begin
  sorry
end

end range_of_a_symmetric_points_l283_283319


namespace find_a_of_even_function_l283_283771

noncomputable def f (a x : ℝ) : ℝ := (x * exp x) / (exp (a * x) - 1)

theorem find_a_of_even_function (hf : ∀ x : ℝ, f a (-x) = f a x) : a = 2 :=
by {
    sorry
}

end find_a_of_even_function_l283_283771


namespace sum_stacked_cards_l283_283464

-- Define the sets of green and orange cards
def green_cards : Finset ℕ := {1, 2, 3, 4, 5}
def orange_cards : Finset ℕ := {2, 3, 4, 5}

-- Define the condition for placing orange cards on green cards
def valid_stack (g o : ℕ) : Prop := g ≤ o

-- Define the specific green cards
def G3 : ℕ := 3
def G4 : ℕ := 4

theorem sum_stacked_cards : 
  ∃ (O3 O4 : ℕ), O3 ∈ orange_cards ∧ O4 ∈ orange_cards ∧ valid_stack G3 O3 ∧ valid_stack G4 O4 ∧ O3 + O4 = 14 :=
by {
  use [3, 4],
  simp [green_cards, orange_cards, valid_stack],
  sorry -- skipping the proof since it is not needed
}

end sum_stacked_cards_l283_283464


namespace total_computers_needed_l283_283155

theorem total_computers_needed (initial_students : ℕ) (students_per_computer : ℕ) (additional_students : ℕ) :
  initial_students = 82 →
  students_per_computer = 2 →
  additional_students = 16 →
  (initial_students + additional_students) / students_per_computer = 49 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end total_computers_needed_l283_283155


namespace flag_height_l283_283132

-- Definitions based on conditions
def flag_width : ℝ := 5
def paint_cost_per_quart : ℝ := 2
def sqft_per_quart : ℝ := 4
def total_spent : ℝ := 20

-- The theorem to prove the height h of the flag
theorem flag_height (h : ℝ) (paint_needed : ℝ -> ℝ) :
  paint_needed h = 4 := sorry

end flag_height_l283_283132


namespace simplified_expr_a_plus_b_l283_283553

open Real

noncomputable def a : ℕ := 2
noncomputable def b : ℕ := 250
def expr : ℝ := (2 ^ 5 * 5 ^ 3) ^ (1 / 4)

theorem simplified_expr : expr = (2 : ℝ) * (250 : ℝ) ^ (1 / 4) := sorry
theorem a_plus_b : a + b = 252 := by
  rw [a, b]
  norm_num

end simplified_expr_a_plus_b_l283_283553


namespace factorize_expression_l283_283241

theorem factorize_expression (a b : ℝ) : 2 * a^2 - 8 * b^2 = 2 * (a + 2 * b) * (a - 2 * b) :=
by
  sorry

end factorize_expression_l283_283241


namespace _l283_283324

noncomputable def concentric_tangent_theorem 
    (O : Point) -- Center of concentric circles
    (r_outer r_inner : ℝ) (r_outer_pos : 0 < r_outer) (r_inner_pos : 0 < r_inner) (r_inner_lt_outer : r_inner < r_outer)
    (A B P1 P2 Q1 Q2 : Point)
    (on_outer_circle_A : dist O A = r_outer)
    (on_outer_circle_B : dist O B = r_outer)
    (on_inner_circle_P1 : dist O P1 = r_inner) 
    (on_inner_circle_P2 : dist O P2 = r_inner) 
    (on_inner_circle_Q1 : dist O Q1 = r_inner) 
    (on_inner_circle_Q2 : dist O Q2 = r_inner)
    (tangent_AP1 : is_tangent A P1 O)
    (tangent_AP2 : is_tangent A P2 O)
    (tangent_BQ1 : is_tangent B Q1 O)
    (tangent_BQ2 : is_tangent B Q2 O) :
  ∃ M : Point, ∃ N : Point, 
  ((is_midpoint M A B ∧ line_perpendicular M (line_through P1 Q1)) ∨ (is_bisector N A B ∧ is_on_line N (line_through P1 Q1)))
:= sorry

end _l283_283324


namespace plot_diameter_l283_283187

theorem plot_diameter (x y : ℝ) :
  x^2 + y^2 + 8 * x - 14 * y + 15 = 0 → ∃ D : ℝ, D = 10 * real.sqrt 2 :=
sorry

end plot_diameter_l283_283187


namespace harmonic_series_induction_l283_283517

theorem harmonic_series_induction (n : ℕ) (h : 2 ≤ n) : 
  (∑ i in (finset.range (2 * n + 1)).filter (λ i, n ≤ i) (λ i, (1 : ℝ) / (i + 1))) < 1 :=
sorry

end harmonic_series_induction_l283_283517


namespace length_of_bridge_l283_283567

noncomputable def speed_kmhr_to_ms (v : ℕ) : ℝ := (v : ℝ) * (1000 / 3600)

noncomputable def distance_traveled (v : ℝ) (t : ℕ) : ℝ := v * (t : ℝ)

theorem length_of_bridge 
  (length_train : ℕ) -- 90 meters
  (speed_train_kmhr : ℕ) -- 45 km/hr
  (time_cross_bridge : ℕ) -- 30 seconds
  (conversion_factor : ℝ := 1000 / 3600) 
  : ℝ := 
  let speed_train_ms := speed_kmhr_to_ms speed_train_kmhr
  let total_distance := distance_traveled speed_train_ms time_cross_bridge
  total_distance - (length_train : ℝ)

example : length_of_bridge 90 45 30 = 285 := by
  sorry

end length_of_bridge_l283_283567


namespace find_a_for_even_function_l283_283818

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def given_function (a : ℝ) (x : ℝ) : ℝ :=
  x * real.exp x / (real.exp (a * x) - 1)

theorem find_a_for_even_function :
  ∀ (a : ℝ), is_even_function (given_function a) ↔ a = 2 :=
by 
  intro a
  sorry

end find_a_for_even_function_l283_283818


namespace cube_root_expression_is_integer_l283_283653

theorem cube_root_expression_is_integer :
  (∛(2^9 * 5^3 * 7^3) : ℝ) = 280 := by
  sorry

end cube_root_expression_is_integer_l283_283653


namespace rain_probability_simulation_l283_283124

/-
Given:
- The probability of rain on any given day is the same.
- Rolling a 1 or 2 on a die indicates rain.
- Rolling the die three times represents predictions for three consecutive days.
- Examined groups of random numbers are: 613, 265, 114, 236, 561, 435, 443, 251, 154, and 353.

Prove:
1. The probability of rain on any given day is 1/3.
2. The probability of rain on two of the three days is 1/5.
-/

theorem rain_probability_simulation : 
  (probability_rain_any_day = 1 / 3) ∧ (probability_rain_two_days = 1 / 5) :=
begin
  -- Definitions
  let groups := [613, 265, 114, 236, 561, 435, 443, 251, 154, 353],
  let rain_days (n : ℕ) := (n % 10 = 1 ∨ n % 10 = 2) + ((n / 10) % 10 = 1 ∨ (n / 10) % 10 = 2) + (n / 100 = 1 ∨ n / 100 = 2),
  
  -- Proof placeholder
  sorry
end

end rain_probability_simulation_l283_283124


namespace general_term_a_sum_Tn_l283_283402

-- Definitions and Conditions
def a (n : ℕ) : ℝ := n + 1
def sum_a (n : ℕ) : ℝ := ∑ i in range n, a i -- S_n = sum of first n terms of a_n
def b (n : ℕ) : ℝ := 3 / (a (2 * n) * a (2 * n + 2))
def sum_b (n : ℕ) : ℝ := ∑ i in range n, b i -- T_n = sum of first n terms of b_n

-- Proof Statements
theorem general_term_a (n : ℕ) (hn : n > 0):
    (a n)^2 - 2 * sum_a n = 2 - a n :=
sorry

theorem sum_Tn (n : ℕ):
    sum_b n = n / (2 * n + 3) :=
sorry

end general_term_a_sum_Tn_l283_283402


namespace correct_props_l283_283641

-- The definitions
variables (m n : Line) (a b γ : Plane)

-- Defining the propositions
def Prop1 : Prop := m ⟂ a ∧ n ‖ a → m ⟂ n
def Prop2 : Prop := a ⟂ γ ∧ b ⟂ γ → a ‖ b
def Prop3 : Prop := m ‖ a ∧ n ‖ a → m ‖ n
def Prop4 : Prop := a ‖ b ∧ b ‖ γ ∧ m ⟂ a → m ⟂ γ

-- The proof statement to be done
theorem correct_props :  Prop1 ∧ ¬Prop2 ∧ ¬Prop3 ∧ Prop4 := 
  by
    sorry

end correct_props_l283_283641


namespace find_a_of_even_function_l283_283773

noncomputable def f (a x : ℝ) : ℝ := (x * exp x) / (exp (a * x) - 1)

theorem find_a_of_even_function (hf : ∀ x : ℝ, f a (-x) = f a x) : a = 2 :=
by {
    sorry
}

end find_a_of_even_function_l283_283773


namespace percentage_slump_in_business_l283_283099

theorem percentage_slump_in_business (X : ℝ) (Y : ℝ) :
  0.05 * Y = 0.04 * X → Y = 0.8 * X → 20 := 
by
  sorry

end percentage_slump_in_business_l283_283099


namespace random_two_digit_sqrt_prob_lt_seven_l283_283547

theorem random_two_digit_sqrt_prob_lt_seven :
  let total_count := 90 in
  let count_lt_sqrt7 := 48 - 10 + 1 in
  (count_lt_sqrt7 : ℚ) / total_count = 13 / 30 :=
by
  let total_count := 90
  let count_lt_sqrt7 := 48 - 10 + 1
  have h1 : count_lt_sqrt7 = 39 := by linarith
  have h2 : (count_lt_sqrt7 : ℚ) / total_count = (39 : ℚ) / 90 := by rw h1
  have h3 : (39 : ℚ) / 90 = 13 / 30 := by norm_num
  rw [h2, h3]
  refl

end random_two_digit_sqrt_prob_lt_seven_l283_283547


namespace tennis_tournament_l283_283353

noncomputable def tennis_tournament_n (k : ℕ) : ℕ := 8 * k + 1

theorem tennis_tournament (n : ℕ) :
  (∃ k : ℕ, n = tennis_tournament_n k) ↔
  (∃ k : ℕ, n = 8 * k + 1) :=
by sorry

end tennis_tournament_l283_283353


namespace even_function_implies_a_eq_2_l283_283919

def f (x : ℝ) (a : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2 (a : ℝ) (h_even : ∀ x, f (-x) a = f x a) : a = 2 :=
by
  intro x
  sorry

end even_function_implies_a_eq_2_l283_283919


namespace cartesian_eq_curve_C2_max_value_diff_sq_l283_283104

-- Definitions based on the conditions
def curve_C1_x (θ : ℝ) : ℝ := Real.cos θ
def curve_C1_y (θ : ℝ) : ℝ := Real.sin θ

-- Scaling to obtain curve C2
def curve_C2_x (θ : ℝ) : ℝ := 2 * curve_C1_x θ
def curve_C2_y (θ : ℝ) : ℝ := Real.sqrt 3 * curve_C1_y θ

-- Part I: Cartesian equation of curve C2
theorem cartesian_eq_curve_C2 : ∀ θ, (curve_C2_x θ)^2 / 4 + (curve_C2_y θ)^2 / 3 = 1 := by
  sorry

-- Part II: Maximum value of |PA|^2 - |PB|^2
def point_A : ℝ × ℝ := (-2, 0)
def point_B : ℝ × ℝ := (1, 1)
def dist_sq (P Q : ℝ × ℝ) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2

theorem max_value_diff_sq (θ : ℝ) :
  let P := (curve_C2_x θ, curve_C2_y θ)
  |dist_sq P point_A - dist_sq P point_B| ≤ 2 + 2 * Real.sqrt 39 := by
  sorry

end cartesian_eq_curve_C2_max_value_diff_sq_l283_283104


namespace f_not_injective_l283_283017

-- Define the function f
def f (x : ℝ) : ℝ := x + (1 / x)

-- State that f is not injective
theorem f_not_injective : ¬(function.injective (λ x : {x // x ≠ 0}, f x)) :=
by
  sorry

end f_not_injective_l283_283017


namespace even_function_implies_a_eq_2_l283_283936

theorem even_function_implies_a_eq_2 (a : ℝ) : 
  (∀ x, (f : ℝ → ℝ) x = (λ x : ℝ, x * exp x / (exp (a * x) - 1)) x -> (f (-x) = f x)) -> a = 2 :=
by
  sorry

end even_function_implies_a_eq_2_l283_283936


namespace option_transformations_incorrect_l283_283336

variable {a b x : ℝ}

theorem option_transformations_incorrect (h : a < b) :
  ¬ (3 - a < 3 - b) := by
  -- Here, we would show the incorrectness of the transformation in Option B
  sorry

end option_transformations_incorrect_l283_283336


namespace length_of_median_AD_l283_283122

def Point := (ℝ × ℝ)

def distance (p1 p2 : Point) : ℝ :=
  ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)^(1/2)

def midpoint (p1 p2 : Point) : Point :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def A : Point := (2, -1)
def B : Point := (3, 2)
def C : Point := (-5, 4)

def D : Point := midpoint B C  -- Midpoint of B and C

theorem length_of_median_AD : distance A D = 5 := 
  sorry

end length_of_median_AD_l283_283122


namespace number_of_sides_of_regular_polygon_l283_283625

variable {α : Type*}

noncomputable def exterior_angle_of_regular_polygon (n : ℕ) : ℝ := 360 / n

theorem number_of_sides_of_regular_polygon (exterior_angle : ℝ) (n : ℕ) (h₁ : exterior_angle = 18) (h₂ : exterior_angle_of_regular_polygon n = exterior_angle) : n = 20 :=
by {
  -- Translation of given conditions
  have sum_exterior_angles : 360 = n * exterior_angle_of_regular_polygon n,
  -- Based on h₂ and h₁ provided
  rw [h₂, h₁] at sum_exterior_angles,
  -- Perform necessary simplifications
  have : 20 = (360 / 18 : ℕ),
  simp,
}

end number_of_sides_of_regular_polygon_l283_283625


namespace even_function_implies_a_eq_2_l283_283931

theorem even_function_implies_a_eq_2 (a : ℝ) : 
  (∀ x, (f : ℝ → ℝ) x = (λ x : ℝ, x * exp x / (exp (a * x) - 1)) x -> (f (-x) = f x)) -> a = 2 :=
by
  sorry

end even_function_implies_a_eq_2_l283_283931


namespace median_of_set_is_b_l283_283036

noncomputable def problem_statement (a : ℤ) (b : ℝ) (h1 : a ≠ 0) (h2 : 0 < b) (h3 : a * b ^ 2 = Real.log b / Real.log 2) : Prop :=
  let s := [0, 1, a, b, b ^ 2].sort (<)
  let median := s.nth (s.length / 2)
  median = b

theorem median_of_set_is_b (a : ℤ) (b : ℝ) (h1 : a ≠ 0) (h2 : 0 < b) (h3 : a * b ^ 2 = Real.log b / Real.log 2) :
  problem_statement a b h1 h2 h3 :=
by
  sorry

end median_of_set_is_b_l283_283036


namespace even_function_implies_a_eq_2_l283_283866

def f (x a : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2 (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → a = 2 :=
by
  sorry

end even_function_implies_a_eq_2_l283_283866


namespace factorize_cubic_l283_283248

theorem factorize_cubic : ∀ x : ℝ, x^3 - 4 * x = x * (x + 2) * (x - 2) :=
by
  sorry

end factorize_cubic_l283_283248


namespace geometric_sequence_S6_div_S3_l283_283288

theorem geometric_sequence_S6_div_S3 (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (h1 : a 1 + a 3 = 5 / 4)
  (h2 : a 2 + a 4 = 5 / 2)
  (hS : ∀ n, S n = a 1 * (1 - (2:ℝ) ^ n) / (1 - 2)) :
  S 6 / S 3 = 9 :=
by
  sorry

end geometric_sequence_S6_div_S3_l283_283288


namespace minimum_candies_l283_283502

theorem minimum_candies (students : ℕ) (N : ℕ) (k : ℕ) : 
  students = 25 → 
  N = 25 * k → 
  (∀ n, 1 ≤ n → n ≤ students → ∃ m, n * k + m ≤ N) → 
  600 ≤ N := 
by
  intros hs hn hd
  sorry

end minimum_candies_l283_283502


namespace bus_stops_per_hour_l283_283562

theorem bus_stops_per_hour 
  (bus_speed_without_stoppages : Float)
  (bus_speed_with_stoppages : Float)
  (bus_stops_per_hour_in_minutes : Float) :
  bus_speed_without_stoppages = 60 ∧ 
  bus_speed_with_stoppages = 45 → 
  bus_stops_per_hour_in_minutes = 15 := by
  sorry

end bus_stops_per_hour_l283_283562


namespace even_function_implies_a_eq_2_l283_283941

theorem even_function_implies_a_eq_2 (a : ℝ) : 
  (∀ x, (f : ℝ → ℝ) x = (λ x : ℝ, x * exp x / (exp (a * x) - 1)) x -> (f (-x) = f x)) -> a = 2 :=
by
  sorry

end even_function_implies_a_eq_2_l283_283941


namespace find_a_if_even_function_l283_283943

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x * exp x / (exp (a * x) - 1)

theorem find_a_if_even_function :
  (∀ x : ℝ, x ≠ 0 → f a (-x) = f a x) → a = 2 :=
by
  sorry

end find_a_if_even_function_l283_283943


namespace integer_solution_l283_283145

theorem integer_solution (x : ℕ) (h : (4 * x)^2 - 2 * x = 3178) : x = 226 :=
by
  sorry

end integer_solution_l283_283145


namespace hunter_3_proposal_l283_283590

theorem hunter_3_proposal {hunter1_coins hunter2_coins hunter3_coins : ℕ} :
  hunter3_coins = 99 ∧ hunter1_coins = 1 ∧ (hunter1_coins + hunter3_coins + hunter2_coins = 100) :=
  sorry

end hunter_3_proposal_l283_283590


namespace max_principals_l283_283235

theorem max_principals (n_years term_length max_principals: ℕ) 
  (h1 : n_years = 12) 
  (h2 : term_length = 4)
  (h3 : max_principals = 4): 
  (∃ p : ℕ, p = max_principals) :=
by
  sorry

end max_principals_l283_283235


namespace solve_for_x_l283_283473

theorem solve_for_x (x : ℚ) (h : x = -3 / 5) : 
  (3 - 2 * x) / (x + 2) + (3 * x - 6) / (3 - 2 * x) = 2 :=
by 
  rw h
  sorry

end solve_for_x_l283_283473


namespace points_are_concyclic_l283_283367

open EuclideanGeometry

variables {A B C H D E : Point}
variables {AD : Line}
variables {BE : Line}

-- Given that H is the orthocenter of triangle ABC
def is_orthocenter (H A B C : Point) : Prop :=
  cond1

-- Given that D is the midpoint of CH
def is_midpoint (D H C : Point) : Prop :=
  cond2

-- Given that BE is perpendicular to AD at point E
def is_perpendicular_at (BE AD : Line) (E : Point) : Prop :=
  cond3

-- Prove that B, C, E, and H are concyclic
theorem points_are_concyclic
  (h1: is_orthocenter H A B C)
  (h2: is_midpoint D H C)
  (h3: is_perpendicular_at BE AD E) :
  Concyclic B C E H :=
sorry

end points_are_concyclic_l283_283367


namespace largest_prime_factor_3136_l283_283526

theorem largest_prime_factor_3136 : ∃ p, nat.prime p ∧ p ∣ 3136 ∧ (∀ q, nat.prime q ∧ q ∣ 3136 → q ≤ p) :=
by {
  sorry
}

end largest_prime_factor_3136_l283_283526


namespace insufficient_info_for_pumpkins_l283_283019

variable (jason_watermelons : ℕ) (sandy_watermelons : ℕ) (total_watermelons : ℕ)

theorem insufficient_info_for_pumpkins (h1 : jason_watermelons = 37)
  (h2 : sandy_watermelons = 11)
  (h3 : jason_watermelons + sandy_watermelons = total_watermelons)
  (h4 : total_watermelons = 48) : 
  ¬∃ (jason_pumpkins : ℕ), true
:= by
  sorry

end insufficient_info_for_pumpkins_l283_283019


namespace percentage_increase_is_150_l283_283008

-- Define the parameters, including original and new number of vaccinated children
def original_children : ℕ := 60
def new_children : ℕ := 150

-- Defining the percentage increase calculation
def percentage_increase (O N : ℕ) : ℝ :=
  ((N - O) / O) * 100

-- The goal is to prove that the percentage increase is 150%
theorem percentage_increase_is_150 :
  percentage_increase original_children new_children = 150 := by
  sorry

end percentage_increase_is_150_l283_283008


namespace repeating_decimal_to_fraction_l283_283678

theorem repeating_decimal_to_fraction :
  (7.036).repeat == 781 / 111 :=
by
  sorry

end repeating_decimal_to_fraction_l283_283678


namespace even_function_implies_a_is_2_l283_283791

noncomputable def f (a x : ℝ) : ℝ := (x * Real.exp x) / (Real.exp (a * x) - 1)

theorem even_function_implies_a_is_2 (a : ℝ) 
  (h : ∀ x : ℝ, f a x = f a (-x)) : a = 2 := 
by
  sorry

end even_function_implies_a_is_2_l283_283791


namespace find_a_for_even_function_l283_283824

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def given_function (a : ℝ) (x : ℝ) : ℝ :=
  x * real.exp x / (real.exp (a * x) - 1)

theorem find_a_for_even_function :
  ∀ (a : ℝ), is_even_function (given_function a) ↔ a = 2 :=
by 
  intro a
  sorry

end find_a_for_even_function_l283_283824


namespace trip_is_400_miles_l283_283220

def fuel_per_mile_empty_plane := 20
def fuel_increase_per_person := 3
def fuel_increase_per_bag := 2
def number_of_passengers := 30
def number_of_crew := 5
def bags_per_person := 2
def total_fuel_needed := 106000

def fuel_consumption_per_mile :=
  fuel_per_mile_empty_plane +
  (number_of_passengers + number_of_crew) * fuel_increase_per_person +
  (number_of_passengers + number_of_crew) * bags_per_person * fuel_increase_per_bag

def trip_length := total_fuel_needed / fuel_consumption_per_mile

theorem trip_is_400_miles : trip_length = 400 := 
by sorry

end trip_is_400_miles_l283_283220


namespace dinosaur_dolls_distribution_l283_283129

-- Defining the conditions
def num_dolls : ℕ := 5
def num_friends : ℕ := 2

-- Lean theorem statement
theorem dinosaur_dolls_distribution :
  (num_dolls * (num_dolls - 1) = 20) :=
by
  -- Sorry placeholder for the proof
  sorry

end dinosaur_dolls_distribution_l283_283129


namespace even_function_implies_a_eq_2_l283_283940

theorem even_function_implies_a_eq_2 (a : ℝ) : 
  (∀ x, (f : ℝ → ℝ) x = (λ x : ℝ, x * exp x / (exp (a * x) - 1)) x -> (f (-x) = f x)) -> a = 2 :=
by
  sorry

end even_function_implies_a_eq_2_l283_283940


namespace game_rounds_l283_283183

noncomputable def play_game (A B C D : ℕ) : ℕ := sorry

theorem game_rounds : play_game 16 15 14 13 = 49 :=
by
  sorry

end game_rounds_l283_283183


namespace percentage_increase_l283_283024

theorem percentage_increase (original new : ℝ) (h_original : original = 50) (h_new : new = 75) : 
  (new - original) / original * 100 = 50 :=
by
  sorry

end percentage_increase_l283_283024


namespace largest_prime_factor_3136_l283_283525

theorem largest_prime_factor_3136 : ∃ p, nat.prime p ∧ p ∣ 3136 ∧ (∀ q, nat.prime q ∧ q ∣ 3136 → q ≤ p) :=
by {
  sorry
}

end largest_prime_factor_3136_l283_283525


namespace find_a_if_even_function_l283_283952

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x * exp x / (exp (a * x) - 1)

theorem find_a_if_even_function :
  (∀ x : ℝ, x ≠ 0 → f a (-x) = f a x) → a = 2 :=
by
  sorry

end find_a_if_even_function_l283_283952


namespace even_function_implies_a_eq_2_l283_283913

def f (x : ℝ) (a : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2 (a : ℝ) (h_even : ∀ x, f (-x) a = f x a) : a = 2 :=
by
  intro x
  sorry

end even_function_implies_a_eq_2_l283_283913


namespace find_a_l283_283992

noncomputable def f (x a : ℝ) : ℝ := (x * (Real.exp x)) / (Real.exp (a * x) - 1)

theorem find_a (a : ℝ) (h : ∀ x, f x a = f (-x) a) : a = 2 :=
begin
  sorry
end

end find_a_l283_283992


namespace prom_night_cost_l283_283382

def ticket_cost : ℕ := 100
def dinner_cost : ℕ := 120
def tip_percent : ℚ := 0.30
def limo_cost_per_hour : ℕ := 80
def limo_hours : ℕ := 8
def tuxedo_rental_cost : ℕ := 150

theorem prom_night_cost : 
  let 
    tickets := 2 * ticket_cost,
    dinner := dinner_cost + (tip_percent * dinner_cost),
    limo := limo_hours * limo_cost_per_hour,
    total_cost := tickets + dinner + limo + tuxedo_rental_cost
  in
    total_cost = 1146 := by
  sorry

end prom_night_cost_l283_283382


namespace even_function_implies_a_eq_2_l283_283864

def f (x a : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2 (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → a = 2 :=
by
  sorry

end even_function_implies_a_eq_2_l283_283864


namespace polynomial_divisibility_l283_283451

theorem polynomial_divisibility (m : ℕ) (hm : 0 < m) :
  ∀ x : ℝ, x * (x + 1) * (2 * x + 1) ∣ (x + 1) ^ (2 * m) - x ^ (2 * m) - 2 * x - 1 :=
by
  intro x
  sorry

end polynomial_divisibility_l283_283451


namespace even_function_implies_a_is_2_l283_283789

noncomputable def f (a x : ℝ) : ℝ := (x * Real.exp x) / (Real.exp (a * x) - 1)

theorem even_function_implies_a_is_2 (a : ℝ) 
  (h : ∀ x : ℝ, f a x = f a (-x)) : a = 2 := 
by
  sorry

end even_function_implies_a_is_2_l283_283789


namespace find_a_if_even_function_l283_283949

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x * exp x / (exp (a * x) - 1)

theorem find_a_if_even_function :
  (∀ x : ℝ, x ≠ 0 → f a (-x) = f a x) → a = 2 :=
by
  sorry

end find_a_if_even_function_l283_283949


namespace limit_S_n_l283_283323

noncomputable def a_n (n : ℕ) : ℝ :=
  if 1 ≤ n ∧ n ≤ 2 then 2 ^ (n + 1) else 1 / 3^n

noncomputable def S_n (n : ℕ) : ℝ :=
  ∑ i in Finset.range (n + 1), a_n i

theorem limit_S_n : 
  tendsto (λ n, S_n n) atTop (𝓝 (12 + 1 / 18)) :=
by
  sorry

end limit_S_n_l283_283323


namespace unique_triangle_assembly_l283_283742

def triangle_inequality (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

def is_valid_triangle_set (sticks : List (ℕ × ℕ × ℕ)) : Prop :=
  sticks.all (λ stick_lengths, triangle_inequality stick_lengths.1 stick_lengths.2 stick_lengths.3)

theorem unique_triangle_assembly (sticks : List ℕ) (h_len : sticks.length = 15)
  (h_sticks : sticks = [2, 3, 4, 20, 30, 40, 200, 300, 400, 2000, 3000, 4000, 20000, 30000, 40000]) :
  ¬ ∃ (other_config : List (ℕ × ℕ × ℕ)),
    is_valid_triangle_set other_config ∧
    other_config ≠ [(2, 3, 4), (20, 30, 40), (200, 300, 400), (2000, 3000, 4000), (20000, 30000, 40000)] :=
begin
  sorry
end

end unique_triangle_assembly_l283_283742


namespace find_a_for_even_function_l283_283827

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def given_function (a : ℝ) (x : ℝ) : ℝ :=
  x * real.exp x / (real.exp (a * x) - 1)

theorem find_a_for_even_function :
  ∀ (a : ℝ), is_even_function (given_function a) ↔ a = 2 :=
by 
  intro a
  sorry

end find_a_for_even_function_l283_283827


namespace no_other_way_to_construct_five_triangles_l283_283735

-- Setting up the problem conditions
def five_triangles_from_fifteen_sticks (sticks : List ℕ) : Prop :=
  sticks.length = 15 ∧
  let split_sticks := sticks.chunk (3) in
  split_sticks.length = 5 ∧
  ∀ triplet ∈ split_sticks, 
    let a := triplet[0]
    let b := triplet[1]
    let c := triplet[2] in
    a + b > c ∧ a + c > b ∧ b + c > a

-- The proof problem statement
theorem no_other_way_to_construct_five_triangles (sticks : List ℕ) :
  five_triangles_from_fifteen_sticks sticks → 
  ∀ (alternative : List (List ℕ)), 
    (alternative.length = 5 ∧ 
    ∀ triplet ∈ alternative, triplet.length = 3 ∧
      let a := triplet[0]
      let b := triplet[1]
      let c := triplet[2] in
      a + b > c ∧ a + c > b ∧ b + c > a) →
    alternative = sticks.chunk (3) :=
by
  sorry

end no_other_way_to_construct_five_triangles_l283_283735


namespace prove_union_sets_l283_283401

universe u

variable {α : Type u}
variable {M N : Set ℕ}
variable (a b : ℕ)

theorem prove_union_sets (h1 : M = {3, 4^a}) (h2 : N = {a, b}) (h3 : M ∩ N = {1}) : M ∪ N = {0, 1, 3} := sorry

end prove_union_sets_l283_283401


namespace find_a_if_even_function_l283_283955

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x * exp x / (exp (a * x) - 1)

theorem find_a_if_even_function :
  (∀ x : ℝ, x ≠ 0 → f a (-x) = f a x) → a = 2 :=
by
  sorry

end find_a_if_even_function_l283_283955


namespace eq_a_2_l283_283972

def f (a x : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

-- Definition of even function: ∀ x, f(-x) = f(x)
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem eq_a_2 (a : ℝ) : (even_function (f a) → a = 2) ∧ (a = 2 → even_function (f a)) :=
by
  sorry

end eq_a_2_l283_283972


namespace jordon_machine_number_l283_283026

theorem jordon_machine_number : 
  ∃ x : ℝ, (2 * x + 3 = 27) ∧ x = 12 :=
by
  sorry

end jordon_machine_number_l283_283026


namespace find_a_l283_283998

noncomputable def f (x a : ℝ) : ℝ := (x * (Real.exp x)) / (Real.exp (a * x) - 1)

theorem find_a (a : ℝ) (h : ∀ x, f x a = f (-x) a) : a = 2 :=
begin
  sorry
end

end find_a_l283_283998


namespace sum_of_shaded_cells_l283_283074

theorem sum_of_shaded_cells (a b c d e f : ℕ) 
  (h1: (a = 1 ∨ a = 2 ∨ a = 3) ∧ (b = 1 ∨ b = 2 ∨ b = 3) ∧ (c = 1 ∨ c = 2 ∨ c = 3) ∧ 
       (d = 1 ∨ d = 2 ∨ d = 3) ∧ (e = 1 ∨ e = 2 ∨ e = 3) ∧ (f = 1 ∨ f = 2 ∨ f = 3))
  (h2: (a ≠ b ∧ a ≠ c ∧ b ≠ c) ∧ 
       (d ≠ e ∧ d ≠ f ∧ e ≠ f) ∧ 
       (a ≠ d ∧ a ≠ f ∧ d ≠ f ∧ 
        b ≠ e ∧ b ≠ f ∧ c ≠ e ∧ c ≠ f))
  (h3: c = 3 ∧ d = 3 ∧ b = 2 ∧ e = 2)
  : b + e = 4 := 
sorry

end sum_of_shaded_cells_l283_283074


namespace land_area_decreases_l283_283053

theorem land_area_decreases (a : ℕ) (h : a > 4) : (a * a) > ((a + 4) * (a - 4)) :=
by
  sorry

end land_area_decreases_l283_283053


namespace cube_root_of_product_l283_283656

theorem cube_root_of_product : (∛(2^9 * 5^3 * 7^3) : ℝ) = 280 := by
  -- Conditions given
  let expr := (2^9 * 5^3 * 7^3 : ℝ)
  
  -- Statement of the problem equivalent to the correct answer
  have h : (∛expr : ℝ) = ∛(2^9 * 5^3 * 7^3) := by rfl

  -- Calculating the actual result
  have result :  ( (2^3) * 5 * 7 : ℝ) = 280 := by
    calc 
      ( (2^3) * 5 * 7 : ℝ) = (8 * 5 * 7 : ℝ) : by rfl
      ... = (40 * 7 : ℝ) : by rfl
      ... = 280 : by rfl

  -- Combining these results to finish the proof
  show (∛expr : ℝ) = 280 from
    calc 
      (∛expr : ℝ) = (2^3 * 5 * 7 : ℝ) : by sorry
      ... = 280 : by exact result

end cube_root_of_product_l283_283656


namespace omega_value_center_of_symmetry_l283_283333

theorem omega_value_center_of_symmetry (ω : ℝ) :
  (∃ k : ℤ, (∀ x : ℝ, sin (ω * x) + cos (ω * x) = sin ((2 * k + 1) * π * (x + 1 / 8) - ω / 4)) ∧ (k : ℝ) * π = (π / 8) * ω + π / 4)
  → ω = 6 :=
begin
  sorry
end

end omega_value_center_of_symmetry_l283_283333


namespace solve_f_prime_1_l283_283765

noncomputable def f (f_prime_1 : ℝ) (x : ℝ) : ℝ :=
  x^2 + 2 * x * f_prime_1 - 6

theorem solve_f_prime_1 (f_prime_1 : ℝ) :
  (∃ f : ℝ → ℝ, (∀ x, deriv (f x) = deriv (x^2 + 2 * x * f_prime_1 - 6)) ∧ (deriv (f 1) = 2 * 1 + 2 * f_prime_1) ∧ f_prime_1 = -2) :=
sorry

end solve_f_prime_1_l283_283765


namespace distance_midpoint_AD_to_BC_l283_283349

theorem distance_midpoint_AD_to_BC
  (A B C D N M : Type)
  (h_perpendicular : ∀ (A B C D N : Type), true) -- mutually perpendicular chords AB and CD
  (h_AC : AC = 6)
  (h_BC : BC = 5)
  (h_BD : BD = 3)
  : distance M.line_to BC = Real.sqrt 5 + 2 := by
  sorry

end distance_midpoint_AD_to_BC_l283_283349


namespace probability_sum_of_three_dice_eq_10_is_1_over_8_l283_283146

noncomputable def probability_sum_of_three_dice_eq_10 : ℚ :=
  let total_outcomes := 6 * 6 * 6 in
  let favorable_outcomes := 27 in
  favorable_outcomes / total_outcomes

theorem probability_sum_of_three_dice_eq_10_is_1_over_8 : 
  probability_sum_of_three_dice_eq_10 = 1 / 8 :=
by
  sorry

end probability_sum_of_three_dice_eq_10_is_1_over_8_l283_283146


namespace monotonic_decreasing_interval_l283_283489

/-- The monotonic decreasing interval of the function log_{1/2} (sin(2x + π/4)) -/
theorem monotonic_decreasing_interval (k : ℤ) :
  ∀ x : ℝ, (k * π - π / 8) < x ∧ x ≤ (k * π + π / 8) 
  ↔ ∃ y : ℝ, y = log (1/2) (sin (2 * x + π / 4)) ∧ sin (2 * x + π / 4) > 0 := 
sorry

end monotonic_decreasing_interval_l283_283489


namespace no_alternate_way_to_form_triangles_l283_283728

noncomputable def canFormTriangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def validSetOfTriples : List (ℕ × ℕ × ℕ) :=
  [(2, 3, 4), (20, 30, 40), (200, 300, 400), (2000, 3000, 4000), (20000, 30000, 40000)]

theorem no_alternate_way_to_form_triangles (stickSet : List (ℕ × ℕ × ℕ)) :
  (∀ (t : ℕ × ℕ × ℕ), t ∈ stickSet → canFormTriangle t.1 t.2 t.3) →
  length stickSet = 5 →
  stickSet = validSetOfTriples ->
  (∀ alternateSet : List (ℕ × ℕ × ℕ),
    ∀ (t : ℕ × ℕ × ℕ), t ∈ alternateSet → canFormTriangle t.1 t.2 t.3 →
    length alternateSet = 5 →
    duplicate_stick_alternateSet = duplicate_stick_stickSet) :=
sorry

end no_alternate_way_to_form_triangles_l283_283728


namespace count_elements_starting_with_4_l283_283403

def setT : set ℤ := {k | 0 ≤ k ∧ k ≤ 3000}
def T : set ℤ := {n | ∃ k ∈ setT, n = 3 ^ k}

theorem count_elements_starting_with_4
  (h_digits : (3 ^ 3000).toString.length = 1801) :
  (∃ n, count (λ x, (x.toString.front = '4')) T = n ∧ n = 1199) :=
sorry

end count_elements_starting_with_4_l283_283403


namespace even_function_implies_a_eq_2_l283_283917

def f (x : ℝ) (a : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2 (a : ℝ) (h_even : ∀ x, f (-x) a = f x a) : a = 2 :=
by
  intro x
  sorry

end even_function_implies_a_eq_2_l283_283917


namespace remainder_mod_1000_l283_283039

def q (x : ℝ) : ℝ := ∑ i in finset.range 2008 , x ^ i

noncomputable def s (x : ℝ) : ℝ :=
  let q_mod := q(x) % (x^3 + x^2 + 3*x + 1)
  in q_mod

theorem remainder_mod_1000 : abs (s 2023) % 1000 = 1 := 
by
  sorry

end remainder_mod_1000_l283_283039


namespace range_of_a_l283_283284

theorem range_of_a (a : ℝ) : (log a (1/2) < 1) ∧ (a^(1/2) < 1) → 0 < a ∧ a < 1/2 :=
by
  sorry

end range_of_a_l283_283284


namespace find_a_if_f_even_l283_283832

def f (a x : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem find_a_if_f_even (a : ℝ) (h : ∀ x : ℝ, x ≠ 0 → f a (-x) = f a x) : a = 2 :=
by 
  sorry

end find_a_if_f_even_l283_283832


namespace tan_X_of_right_triangle_l283_283684

-- Define the given lengths.
def leg1 : ℝ := 40
def hypotenuse : ℝ := 41

-- Calculate the other leg using the Pythagorean Theorem.
def leg2 : ℝ := Real.sqrt (hypotenuse^2 - leg1^2)

-- State the theorem to be proven.
theorem tan_X_of_right_triangle : Real.tan (Real.arctan (leg2 / leg1)) = leg2 / leg1 :=
by
  -- Proof would go here.
  sorry

end tan_X_of_right_triangle_l283_283684


namespace find_a_of_even_function_l283_283777

noncomputable def f (a x : ℝ) : ℝ := (x * exp x) / (exp (a * x) - 1)

theorem find_a_of_even_function (hf : ∀ x : ℝ, f a (-x) = f a x) : a = 2 :=
by {
    sorry
}

end find_a_of_even_function_l283_283777


namespace even_function_implies_a_eq_2_l283_283874

def f (x a : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2 (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → a = 2 :=
by
  sorry

end even_function_implies_a_eq_2_l283_283874


namespace number_of_integer_solutions_l283_283332

theorem number_of_integer_solutions : ∃ (n : ℕ), n = 120 ∧ ∀ (x y z : ℤ), x * y * z = 2008 → n = 120 :=
by
  sorry

end number_of_integer_solutions_l283_283332


namespace geometric_series_first_term_l283_283642

theorem geometric_series_first_term
  (r : ℝ) (S : ℝ) (h_ratio : r = -1 / 3) (h_sum : S = 9) :
  let a := S * (1 - r) in a = 12 :=
by
  sorry

end geometric_series_first_term_l283_283642


namespace parallel_necessary_but_not_sufficient_l283_283700

variables {V : Type*} [AddCommGroup V] [VectorSpace ℝ V]

def parallel (a b : V) : Prop := ∃ c : ℝ, c ≠ 0 ∧ a = c • b

theorem parallel_necessary_but_not_sufficient (a b : V) :
  parallel a b → a = b → (parallel a b ∧ ¬ (parallel b a → a = b)) :=
by sorry

end parallel_necessary_but_not_sufficient_l283_283700


namespace arithmetic_seq_problem_l283_283010

noncomputable def a_n (n : ℕ) (a1 : ℕ) (d : ℕ) : ℕ :=
  a1 + (n - 1) * d

theorem arithmetic_seq_problem :
  ∃ d : ℕ, a_n 1 2 d = 2 ∧ a_n 2 2 d + a_n 3 2 d = 13 ∧ (a_n 4 2 d + a_n 5 2 d + a_n 6 2 d = 42) :=
by
  sorry

end arithmetic_seq_problem_l283_283010


namespace telephone_partition_l283_283128

open Classical

-- Let's define our conditions.
variables {n : ℕ} (hn : n ≥ 6)
variable (G : SimpleGraph (Fin n))
variable (Hcond1 : ∀ v : Fin n, (G.degree v) = Nat.floor ((n - 1 : ℚ) / 2))
variable (Hcond2 : ∀ (a b c : Fin n), G.adj a b ∨ G.adj b c ∨ G.adj c a)

-- Our goal: to show there exist two disjoint groups of vertices with the required properties.
theorem telephone_partition (hn : n ≥ 6) : 
  ∃ (S1 S2 : Finset (Fin n)), 
    S1 ∪ S2 = Finset.univ ∧ 
    S1 ∩ S2 = ∅ ∧ 
    (∀ (x y : Fin n), x ∈ S1 → y ∈ S1 → G.adj x y) ∧ 
    (∀ (x y : Fin n), x ∈ S2 → y ∈ S2 → G.adj x y) :=
begin
  sorry
end

end telephone_partition_l283_283128


namespace pizza_combination_count_l283_283190

theorem pizza_combination_count : nat.choose 8 5 = 56 := by
  sorry

end pizza_combination_count_l283_283190


namespace standing_arrangements_of_leaders_l283_283203

noncomputable def totalStandingArrangements (n : ℕ) : ℕ :=
  if h : n = 21 then (A 2 2) * (A 18 18) else 0

theorem standing_arrangements_of_leaders :
  totalStandingArrangements 21 = (A 2 2) * (A 18 18) :=
by sorry

end standing_arrangements_of_leaders_l283_283203


namespace unique_triangle_assembly_l283_283744

def triangle_inequality (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

def is_valid_triangle_set (sticks : List (ℕ × ℕ × ℕ)) : Prop :=
  sticks.all (λ stick_lengths, triangle_inequality stick_lengths.1 stick_lengths.2 stick_lengths.3)

theorem unique_triangle_assembly (sticks : List ℕ) (h_len : sticks.length = 15)
  (h_sticks : sticks = [2, 3, 4, 20, 30, 40, 200, 300, 400, 2000, 3000, 4000, 20000, 30000, 40000]) :
  ¬ ∃ (other_config : List (ℕ × ℕ × ℕ)),
    is_valid_triangle_set other_config ∧
    other_config ≠ [(2, 3, 4), (20, 30, 40), (200, 300, 400), (2000, 3000, 4000), (20000, 30000, 40000)] :=
begin
  sorry
end

end unique_triangle_assembly_l283_283744


namespace regular_polygon_sides_l283_283631

theorem regular_polygon_sides (angle : ℝ) (h_angle : angle = 18) : ∃ n : ℕ, n = 20 :=
by
  have sum_exterior_angles : ℝ := 360
  have num_sides := sum_exterior_angles / angle
  have h_num_sides : num_sides = 20 := by
    calc
      num_sides = 360 / 18 : by rw [h_angle]
      ... = 20 : by norm_num
  use 20
  rw ← h_num_sides
  sorry

end regular_polygon_sides_l283_283631


namespace no_alternative_way_to_construct_triangles_l283_283710

theorem no_alternative_way_to_construct_triangles (sticks : list ℕ) (h_len : sticks.length = 15)
  (h_set : ∃ (sets : list (list ℕ)), sets.length = 5 ∧ ∀ set ∈ sets, set.length = 3 
           ∧ (∀ (a b c : ℕ) (ha : a ∈ set) (hb : b ∈ set) (hc : c ∈ set), 
             (a + b > c ∧ a + c > b ∧ b + c > a))):
  ¬∃ (alt_sets : list (list ℕ)), alt_sets ≠ sets ∧ alt_sets.length = 5 
      ∧ ∀ set ∈ alt_sets, set.length = 3 
      ∧ (∀ (a b c : ℕ) (ha : a ∈ set) (hb : b ∈ set) (hc : c ∈ set),
        (a + b > c ∧ a + c > b ∧ b + c > a)) :=
sorry

end no_alternative_way_to_construct_triangles_l283_283710


namespace factorize_x_l283_283257

theorem factorize_x^3_minus_4x (x : ℝ) : x^3 - 4 * x = x * (x + 2) * (x - 2) := 
sorry

end factorize_x_l283_283257


namespace expected_area_swept_by_rotation_l283_283032

theorem expected_area_swept_by_rotation :
  let Ω := circle 1
  let A : Point := some_point_on_Ω_circumference Ω
  let θ := Uniform 0 180
  expected_area_swept Ω A θ = 2 * π :=
sorry

end expected_area_swept_by_rotation_l283_283032


namespace determine_sum_of_roots_S_l283_283569

namespace QuadraticEquations

variables {a a' b b' c c' : ℝ}
variable [a_nonzero : NonZero a]
variable [a'_nonzero : NonZero a']

def sum_of_roots (a b : ℝ) [NonZero a] : ℝ := -b / a
def product_of_roots (a c : ℝ) [NonZero a] : ℝ := c / a

def λ_value (a a' c c' : ℝ) [NonZero a] [NonZero a'] : ℝ := -a^2 * c' / (a'^2 * c)

def S_value (s s' p p' : ℝ) [NonZero p] [NonZero p'] : ℝ := (p * s - p' * s') / (p - p')

theorem determine_sum_of_roots_S (a a' b b' c c' : ℝ) 
  [nonzero_a : NonZero a] [nonzero_a' : NonZero a'] 
  (s := sum_of_roots a b) (s' := sum_of_roots a' b')
  (p := product_of_roots a c) (p' := product_of_roots a' c')
  (λ := λ_value a a' c c') :
  S_value s s' p p' = (p * s - p' * s') / (p - p') :=
sorry

end QuadraticEquations

end determine_sum_of_roots_S_l283_283569


namespace height_of_barbed_wire_l283_283084

def square (x : ℝ) := x * x

theorem height_of_barbed_wire
  (area : ℝ)
  (cost_per_meter : ℝ)
  (gate_width : ℝ)
  (num_gates : ℕ)
  (total_cost : ℝ)
  (h : ℝ) :
  area = 3136 ∧ cost_per_meter = 1 ∧ gate_width = 1 ∧ num_gates = 2 ∧ total_cost = 666 →
  h = 2 :=
by
  intros h_conditions
  have A : area = 3136 := h_conditions.1
  have B : cost_per_meter = 1 := h_conditions.2
  have C : gate_width = 1 := h_conditions.3
  have D : num_gates = 2 := h_conditions.4
  have E : total_cost = 666 := h_conditions.5
  sorry

end height_of_barbed_wire_l283_283084


namespace regular_polygon_sides_l283_283616

theorem regular_polygon_sides (n : ℕ) (h : 360 = 18 * n) : n = 20 := 
by 
  sorry

end regular_polygon_sides_l283_283616


namespace no_alternative_way_to_construct_triangles_l283_283711

theorem no_alternative_way_to_construct_triangles (sticks : list ℕ) (h_len : sticks.length = 15)
  (h_set : ∃ (sets : list (list ℕ)), sets.length = 5 ∧ ∀ set ∈ sets, set.length = 3 
           ∧ (∀ (a b c : ℕ) (ha : a ∈ set) (hb : b ∈ set) (hc : c ∈ set), 
             (a + b > c ∧ a + c > b ∧ b + c > a))):
  ¬∃ (alt_sets : list (list ℕ)), alt_sets ≠ sets ∧ alt_sets.length = 5 
      ∧ ∀ set ∈ alt_sets, set.length = 3 
      ∧ (∀ (a b c : ℕ) (ha : a ∈ set) (hb : b ∈ set) (hc : c ∈ set),
        (a + b > c ∧ a + c > b ∧ b + c > a)) :=
sorry

end no_alternative_way_to_construct_triangles_l283_283711


namespace simplify_expression_l283_283659

theorem simplify_expression: 3 * Real.sqrt 48 - 6 * Real.sqrt (1 / 3) + (Real.sqrt 3 - 1) ^ 2 = 8 * Real.sqrt 3 + 4 := by
  sorry

end simplify_expression_l283_283659


namespace greatest_k_rook_free_square_l283_283397

noncomputable def greatest_k (n : ℕ) (h : n ≥ 2) : ℕ :=
  ⌈real.sqrt n⌉ - 1

theorem greatest_k_rook_free_square (n : ℕ) (h : n ≥ 2) :
  ∀ config : fin n → fin n, (∀ i j, i ≠ j → config i ≠ config j) → 
  ∃ k : ℕ, k = greatest_k n h ∧
  ∀ i j, (∀ p q, (p ≠ j ∧ q ≠ i) → config p ≠ ⟨i, sorry⟩) → sorry :=
begin
  sorry
end

end greatest_k_rook_free_square_l283_283397


namespace find_a_even_function_l283_283989

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def given_function (a : ℝ) (x : ℝ) : ℝ :=
  x * Real.exp x / (Real.exp (a * x) - 1)

theorem find_a_even_function :
  (is_even_function (given_function 2)) → true :=
by
  intro h
  sorry

end find_a_even_function_l283_283989


namespace even_function_implies_a_eq_2_l283_283894

def f (a x : ℝ) : ℝ := (x * exp x) / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2
  (h : ∀ x : ℝ, f a x = f a (-x)) : a = 2 := 
sorry

end even_function_implies_a_eq_2_l283_283894


namespace f_is_even_iff_a_is_2_l283_283804

def f (a : ℝ) (x : ℝ) : ℝ := x * Real.exp x / (Real.exp (a * x) - 1)

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem f_is_even_iff_a_is_2 : (∀ x, f 2 (-x) = f 2 x) ↔ ∀ a, a = 2 := 
sorry

end f_is_even_iff_a_is_2_l283_283804


namespace imaginary_part_of_z_l283_283483

-- Define the imaginary unit
def i : ℂ := complex.i

-- Define the complex number z
def z : ℂ := (i^5 / (1 - i)) - i

-- Theorem to prove the imaginary part of z
theorem imaginary_part_of_z : z.im = -1/2 :=
by sorry

end imaginary_part_of_z_l283_283483


namespace no_alternative_way_to_construct_triangles_l283_283718

-- Define the triangle inequality condition
def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the set of stick lengths for each required triangle
def valid_triangle_sets (stick_set : Set (Set ℕ)) : Prop :=
  stick_set = { {2, 3, 4}, {20, 30, 40}, {200, 300, 400}, {2000, 3000, 4000}, {20000, 30000, 40000} }

-- Define the problem statement
theorem no_alternative_way_to_construct_triangles (s : Set ℕ) (h : s.card = 15) :
  (∀ t ∈ (s.powerset.filter (λ t, t.card = 3)), triangle_inequality t.some (finset_some_spec t) ∧ (∃ sets : Set (Set ℕ), sets.card = 5 ∧ ∀ u ∈ sets, u.card = 3 ∧ triangle_inequality u.some (finset_some_spec u)))
  → valid_triangle_sets (s.powerset.filter (λ t, t.card = 3)) := 
sorry

end no_alternative_way_to_construct_triangles_l283_283718


namespace can_form_square_from_tiles_l283_283199

def right_triangle_tile (a b : ℕ) := ∃ (A B C : ℝ), right_triangle A B C ∧
  A = a ∧ B = b

theorem can_form_square_from_tiles :
  ∀ (n : ℕ) (a b : ℕ), n = 20 ∧ a = 1 ∧ b = 2 →
  ∃ (s : ℝ), 
    let area := (n * ((a * b) / 2 : ℝ)) in
    s^2 = area ∧
    ∃ (arrangement : list (nat × nat)), True := sorry

end can_form_square_from_tiles_l283_283199


namespace find_a_l283_283991

noncomputable def f (x a : ℝ) : ℝ := (x * (Real.exp x)) / (Real.exp (a * x) - 1)

theorem find_a (a : ℝ) (h : ∀ x, f x a = f (-x) a) : a = 2 :=
begin
  sorry
end

end find_a_l283_283991


namespace unique_solution_l283_283048

def S : Set ℕ := {0, 1, 2, 3, 4}

def op (i j : ℕ) : ℕ := (i + j) % 5

theorem unique_solution :
  ∃! x ∈ S, op (op x x) 2 = 1 := sorry

end unique_solution_l283_283048


namespace determine_a_for_even_function_l283_283888

theorem determine_a_for_even_function :
  ∃ a : ℝ, (∀ x : ℝ, x ≠ 0 → (∃ f : ℝ → ℝ, 
  f x = x * exp x / (exp (a * x) - 1) ∧
  (∀ x : ℝ, f (-x) = f x))) → a = 2 :=
by
  sorry

end determine_a_for_even_function_l283_283888


namespace sample_size_is_100_l283_283513

-- Define the number of students selected for the sample.
def num_students_sampled : ℕ := 100

-- The statement that the sample size is equal to the number of students sampled.
theorem sample_size_is_100 : num_students_sampled = 100 := 
by {
  -- Proof goes here
  sorry
}

end sample_size_is_100_l283_283513


namespace exterior_angle_regular_polygon_l283_283599

theorem exterior_angle_regular_polygon (exterior_angle : ℝ) (sides : ℕ) (h : exterior_angle = 18) : sides = 20 :=
by
  -- Use the condition that the sum of exterior angles of any polygon is 360 degrees.
  have sum_exterior_angles : ℕ := 360
  -- Set up the equation 18 * sides = 360
  have equation : 18 * sides = sum_exterior_angles := sorry
  -- Therefore, sides = 20
  sorry

end exterior_angle_regular_polygon_l283_283599


namespace work_combined_days_l283_283161

def a_time := 66
def b_time := 33
def c_time := 99
def combined_work_rate := 1 / a_time + 1 / b_time + 1 / c_time
def days_needed := 1 / combined_work_rate

theorem work_combined_days : days_needed = 18 :=
by 
  have h: combined_work_rate = 1 / 18 := by sorry
  show 1 / combined_work_rate = 18 by 
    rw [h]
    exact rfl
  rfl

end work_combined_days_l283_283161


namespace no_other_way_five_triangles_l283_283704

def is_triangle {α : Type*} [linear_ordered_field α] (a b c : α) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem no_other_way_five_triangles (sticks : list ℝ)
  (h_len : sticks.length = 15)
  (h_sets : ∃ (sets : list (ℝ × ℝ × ℝ)), 
    sets.length = 5 ∧ ∀ {a b c}, (a, b, c) ∈ sets → is_triangle a b c) :
  ∀ sets, sets.length = 5 ∧ ∀ {a b c}, (a, b, c) ∈ sets → is_triangle a b c → sets = h_sets :=
begin
  sorry
end

end no_other_way_five_triangles_l283_283704


namespace sin_squared_angle_val_l283_283589

noncomputable def problem_statement : ℝ :=
  let radius_smaller := 1
  let radius_larger := 5
  let point_A := (0, 0)
  let center_small_disk := (1, 0)
  let center_large_disk := (5, 0)
  
  -- The condition that smaller disk rolls around the larger once
  let circumference_large_disk := 2 * Real.pi * radius_larger
  let arc_moved_small_center := 2 * Real.pi * radius_smaller
  let angle_center_moved := arc_moved_small_center / radius_larger

  -- After rolling, center moved to D, point A moves to B maintaining tangency
  let final_position_B := (0, 1)
  let angle_BEA := Real.asin(1 / 5)
  let sin_squared_angle := (Real.sin angle_BEA) ^ 2

  -- Result that needs to be proven
  sin_squared_angle

theorem sin_squared_angle_val : problem_statement = 1 / 25 := 
by {
  sorry
}

end sin_squared_angle_val_l283_283589


namespace sin_theta_in_terms_of_x_l283_283044

-- Defining the conditions in Lean 4

variables {x θ : ℝ}
-- Assume θ is an acute angle
variable (hθ : 0 < θ ∧ θ < π / 2)
-- Assume the given cosine half-angle expression
variable (hcos_half : cos (θ / 2) = real.sqrt ((x + 1) / (2 * x)))

-- The proposition to prove
theorem sin_theta_in_terms_of_x :
  sin θ = real.sqrt (x^2 - 1) / x :=
sorry

end sin_theta_in_terms_of_x_l283_283044


namespace find_a_of_even_function_l283_283767

noncomputable def f (a x : ℝ) : ℝ := (x * exp x) / (exp (a * x) - 1)

theorem find_a_of_even_function (hf : ∀ x : ℝ, f a (-x) = f a x) : a = 2 :=
by {
    sorry
}

end find_a_of_even_function_l283_283767


namespace line_hyperbola_intersection_l283_283673

noncomputable def intersects_at_two_points (L : Line) (H : Hyperbola) : Prop := 
   ∃ A B : Point, A ≠ B ∧ on_line A L ∧ on_line B L ∧ on_hyperbola A H ∧ on_hyperbola B H

noncomputable def parallel_to_asymptote (L : Line) (H : Hyperbola) : Prop := 
   ∃ A : Point, on_line A L ∧ parallel L (asymptote H A)

theorem line_hyperbola_intersection : ∀ (L : Line) (H : Hyperbola),
  (¬(parallel_to_asymptote L H) ∧ intersects_at_two_points L H) ∨
  (parallel_to_asymptote L H ∧ ¬(intersects_at_two_points L H)) :=
sorry

end line_hyperbola_intersection_l283_283673


namespace portfolio_value_after_two_years_l283_283392

def initial_portfolio := 80

def first_year_growth_rate := 0.15
def add_after_6_months := 28
def withdraw_after_9_months := 10

def second_year_growth_first_6_months := 0.10
def second_year_decline_last_6_months := 0.04

def final_portfolio_value := 115.59

theorem portfolio_value_after_two_years 
  (initial_portfolio : ℝ)
  (first_year_growth_rate : ℝ)
  (add_after_6_months : ℕ)
  (withdraw_after_9_months : ℕ)
  (second_year_growth_first_6_months : ℝ)
  (second_year_decline_last_6_months : ℝ)
  (final_portfolio_value : ℝ) :
  (initial_portfolio = 80) →
  (first_year_growth_rate = 0.15) →
  (add_after_6_months = 28) →
  (withdraw_after_9_months = 10) →
  (second_year_growth_first_6_months = 0.10) →
  (second_year_decline_last_6_months = 0.04) →
  (final_portfolio_value = 115.59) :=
by
  sorry

end portfolio_value_after_two_years_l283_283392


namespace find_a_even_function_l283_283980

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def given_function (a : ℝ) (x : ℝ) : ℝ :=
  x * Real.exp x / (Real.exp (a * x) - 1)

theorem find_a_even_function :
  (is_even_function (given_function 2)) → true :=
by
  intro h
  sorry

end find_a_even_function_l283_283980


namespace quadratic_root_condition_l283_283091

theorem quadratic_root_condition (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 > 1 ∧ x2 < 1 ∧ x1^2 + 2*a*x1 + 1 = 0 ∧ x2^2 + 2*a*x2 + 1 = 0) →
  a < -1 :=
by
  sorry

end quadratic_root_condition_l283_283091


namespace even_function_implies_a_is_2_l283_283786

noncomputable def f (a x : ℝ) : ℝ := (x * Real.exp x) / (Real.exp (a * x) - 1)

theorem even_function_implies_a_is_2 (a : ℝ) 
  (h : ∀ x : ℝ, f a x = f a (-x)) : a = 2 := 
by
  sorry

end even_function_implies_a_is_2_l283_283786


namespace collinearity_of_points_concurrency_of_lines_l283_283568

-- Definitions
variables {A B C A1 B1 C1 : Type*}
variables (k : ℕ) [NonzeroType B1] (R : ℝ)
def B1_on_line_AB : Prop := ∃ t : ℝ, A+B1=t*(A+B)
def A1_on_line_BC : Prop := ∃ t : ℝ, B+A1=t*(B+C)
def C1_on_line_CA : Prop := ∃ t : ℝ, C+C1=t*(C+A)

-- Menelaus' Theorem
theorem collinearity_of_points
  (h1 : B1_on_line_AB)
  (h2 : A1_on_line_BC)
  (h3 : C1_on_line_CA)
  (hR : R = 1) :
  (k % 2 = 0) ↔ (A1 + B1 + C1 = 0) :=
sorry

-- Ceva's Theorem
theorem concurrency_of_lines
  (h1 : B1_on_line_AB)
  (h2 : A1_on_line_BC)
  (h3 : C1_on_line_CA)
  (hR : R = 1) :
  (k % 2 = 1) ↔ (A1 ∧ B1 ∧ C1 = ∅) :=
sorry

end collinearity_of_points_concurrency_of_lines_l283_283568


namespace minimum_value_function_l283_283488

-- Defining the function y
def y (x : ℝ) : ℝ := x^2 + 2 / x

-- Statement to prove the minimum value of the function
theorem minimum_value_function : ∀ (x : ℝ), x > 0 → (∃ k, (∀ x, x > 0 → y x ≥ k) ∧ (∀ x, x > 0 → y x = k → x = 1) ∧ k = 3) :=
by
  intro x hx
  use 3
  split
  {
    intro x hx
    -- Proof that y x ≥ 3 for all x > 0 goes here
    sorry
  }
  split
  {
    intro x hx h
    -- Proof that y x = 3 if and only if x = 1 goes here
    sorry
  }
  -- Proof that k = 3 is the minimum value
  rfl

end minimum_value_function_l283_283488


namespace regular_polygon_sides_l283_283610

theorem regular_polygon_sides (theta : ℝ) (h : theta = 18) : 
  ∃ n : ℕ, 360 / theta = n ∧ n = 20 :=
by
  use 20
  constructor
  apply eq_of_div_eq_mul
  rw h
  norm_num
  sorry

end regular_polygon_sides_l283_283610


namespace centers_coincide_l283_283520

open Point

-- Assume Point, Parallelogram, midpoint, center are defined in Mathlib.

variables {A B C D A₁ B₁ C₁ D₁ : Point}
variable (P : Parallelogram)
variables (A_on_AB : A₁ ∈ lineSegment A B)
          (B_on_BC : B₁ ∈ lineSegment B C)
          (C_on_CD : C₁ ∈ lineSegment C D)
          (D_on_DA : D₁ ∈ lineSegment D A)

-- Define the two parallelograms
def parallelogram1 := Parallelogram.mk A B C D
def parallelogram2 := Parallelogram.mk A₁ B₁ C₁ D₁

-- Prove that the centers coincide
theorem centers_coincide :
  center parallelogram1 = center parallelogram2 :=
sorry

end centers_coincide_l283_283520


namespace regular_polygon_sides_l283_283606

theorem regular_polygon_sides (n : ℕ) (h : 1 < n) (exterior_angle : ℝ) (h_ext : exterior_angle = 18) :
  n * exterior_angle = 360 → n = 20 :=
by 
  sorry

end regular_polygon_sides_l283_283606


namespace determinant_identity_l283_283661

def matrix := 
  ![
    ![Real.sin (a + b), Real.cos (a - b), Real.cos a],
    ![Real.cos (a - b), 1, Real.sin b],
    ![Real.cos a, Real.sin b, 1]
  ]

theorem determinant_identity (a b : ℝ) :
  matrix.determinant = Real.sin (a + b) * (1 - Real.sin b ^ 2) - Real.cos (a - b) * (Real.cos (a - b) - Real.sin b * Real.cos a) + Real.cos a * (Real.sin b * Real.cos (a - b) - Real.cos a) := 
  sorry

end determinant_identity_l283_283661


namespace total_ingredient_cups_l283_283493

def butter_flour_sugar_ratio_butter := 2
def butter_flour_sugar_ratio_flour := 5
def butter_flour_sugar_ratio_sugar := 3
def flour_used := 15

theorem total_ingredient_cups :
  butter_flour_sugar_ratio_butter + 
  butter_flour_sugar_ratio_flour + 
  butter_flour_sugar_ratio_sugar = 10 →
  flour_used / butter_flour_sugar_ratio_flour = 3 →
  6 + 15 + 9 = 30 := by
  intros
  sorry

end total_ingredient_cups_l283_283493


namespace problem1_l283_283573

theorem problem1 (f : ℚ → ℚ) (a : Fin 7 → ℚ) (h₁ : ∀ x, f x = (1 - 3 * x) * (1 + x) ^ 5)
  (h₂ : ∀ x, f x = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + a 6 * x^6) :
  a 0 + (1/3) * a 1 + (1/3^2) * a 2 + (1/3^3) * a 3 + (1/3^4) * a 4 + (1/3^5) * a 5 + (1/3^6) * a 6 = 
  (1 - 3 * (1/3)) * (1 + (1/3))^5 :=
by sorry

end problem1_l283_283573


namespace find_c_minus_d_l283_283037

noncomputable def f (c d x : ℝ) := c * x + d
noncomputable def g (x : ℝ) := -4 * x + 6
noncomputable def h (c d x : ℝ) := f c d (g x)
noncomputable def h_inv (x : ℝ) := x + 8

theorem find_c_minus_d (c d : ℝ)
  (Hh : ∀ x, h c d x = x - 8)
  (Hinv : ∀ x, h_inv (h c d x) = x)
  : c - d = 25 / 4 := 
begin
  sorry
end

end find_c_minus_d_l283_283037


namespace even_function_implies_a_is_2_l283_283784

noncomputable def f (a x : ℝ) : ℝ := (x * Real.exp x) / (Real.exp (a * x) - 1)

theorem even_function_implies_a_is_2 (a : ℝ) 
  (h : ∀ x : ℝ, f a x = f a (-x)) : a = 2 := 
by
  sorry

end even_function_implies_a_is_2_l283_283784


namespace determine_a_for_even_function_l283_283893

theorem determine_a_for_even_function :
  ∃ a : ℝ, (∀ x : ℝ, x ≠ 0 → (∃ f : ℝ → ℝ, 
  f x = x * exp x / (exp (a * x) - 1) ∧
  (∀ x : ℝ, f (-x) = f x))) → a = 2 :=
by
  sorry

end determine_a_for_even_function_l283_283893


namespace probability_sqrt_less_than_seven_l283_283544

-- Definitions and conditions from part a)
def is_two_digit_number (n : ℕ) : Prop := (10 ≤ n) ∧ (n ≤ 99)
def sqrt_less_than_seven (n : ℕ) : Prop := real.sqrt n < 7

-- Lean 4 statement for the actual proof problem
theorem probability_sqrt_less_than_seven : 
  (∃ n, is_two_digit_number n ∧ sqrt_less_than_seven n) → ∑ i in (finset.range 100).filter is_two_digit_number, if sqrt_less_than_seven i then 1 else 0 = 39 :=
sorry

end probability_sqrt_less_than_seven_l283_283544


namespace unique_triangle_assembly_l283_283749

def triangle_inequality (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

def is_valid_triangle_set (sticks : List (ℕ × ℕ × ℕ)) : Prop :=
  sticks.all (λ stick_lengths, triangle_inequality stick_lengths.1 stick_lengths.2 stick_lengths.3)

theorem unique_triangle_assembly (sticks : List ℕ) (h_len : sticks.length = 15)
  (h_sticks : sticks = [2, 3, 4, 20, 30, 40, 200, 300, 400, 2000, 3000, 4000, 20000, 30000, 40000]) :
  ¬ ∃ (other_config : List (ℕ × ℕ × ℕ)),
    is_valid_triangle_set other_config ∧
    other_config ≠ [(2, 3, 4), (20, 30, 40), (200, 300, 400), (2000, 3000, 4000), (20000, 30000, 40000)] :=
begin
  sorry
end

end unique_triangle_assembly_l283_283749


namespace change_in_ratio_of_flour_to_water_l283_283109

theorem change_in_ratio_of_flour_to_water :
  (original_ratio_flour_to_water : ℝ) (new_ratio_flour_to_water : ℝ) (change_in_ratio : ℝ) :
  let original_ratio_flour_to_water := 5 / 3   -- simplified from 10:6
  let new_ratio_flour_to_water := (5 * 4 / 3) / 2  -- derived from the new recipe
  let change_in_ratio := new_ratio_flour_to_water - original_ratio_flour_to_water
  in change_in_ratio = 1.665 := 
sorry

end change_in_ratio_of_flour_to_water_l283_283109


namespace even_function_implies_a_eq_2_l283_283899

def f (a x : ℝ) : ℝ := (x * exp x) / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2
  (h : ∀ x : ℝ, f a x = f a (-x)) : a = 2 := 
sorry

end even_function_implies_a_eq_2_l283_283899


namespace no_alternate_way_to_form_triangles_l283_283733

noncomputable def canFormTriangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def validSetOfTriples : List (ℕ × ℕ × ℕ) :=
  [(2, 3, 4), (20, 30, 40), (200, 300, 400), (2000, 3000, 4000), (20000, 30000, 40000)]

theorem no_alternate_way_to_form_triangles (stickSet : List (ℕ × ℕ × ℕ)) :
  (∀ (t : ℕ × ℕ × ℕ), t ∈ stickSet → canFormTriangle t.1 t.2 t.3) →
  length stickSet = 5 →
  stickSet = validSetOfTriples ->
  (∀ alternateSet : List (ℕ × ℕ × ℕ),
    ∀ (t : ℕ × ℕ × ℕ), t ∈ alternateSet → canFormTriangle t.1 t.2 t.3 →
    length alternateSet = 5 →
    duplicate_stick_alternateSet = duplicate_stick_stickSet) :=
sorry

end no_alternate_way_to_form_triangles_l283_283733


namespace area_difference_36_l283_283496

def square_area (d : ℝ) : ℝ := (d / Real.sqrt 2) ^ 2

theorem area_difference_36 :
  square_area 19 - square_area 17 = 36 :=
by
  unfold square_area
  calc
    (19 / Real.sqrt 2) ^ 2 - (17 / Real.sqrt 2) ^ 2 = (19^2 / 2) - (17^2 / 2) : by { sorry }
    ... = (361 / 2) - (289 / 2) : by { norm_num }
    ... = 36 : by { norm_num }

end area_difference_36_l283_283496


namespace probability_sqrt_less_than_seven_l283_283542

-- Definitions and conditions from part a)
def is_two_digit_number (n : ℕ) : Prop := (10 ≤ n) ∧ (n ≤ 99)
def sqrt_less_than_seven (n : ℕ) : Prop := real.sqrt n < 7

-- Lean 4 statement for the actual proof problem
theorem probability_sqrt_less_than_seven : 
  (∃ n, is_two_digit_number n ∧ sqrt_less_than_seven n) → ∑ i in (finset.range 100).filter is_two_digit_number, if sqrt_less_than_seven i then 1 else 0 = 39 :=
sorry

end probability_sqrt_less_than_seven_l283_283542


namespace owen_turtles_l283_283444

theorem owen_turtles (o_initial : ℕ) (j_initial : ℕ) (o_after_month : ℕ) (j_remaining : ℕ) (o_final : ℕ) 
  (h1 : o_initial = 21)
  (h2 : j_initial = o_initial - 5)
  (h3 : o_after_month = 2 * o_initial)
  (h4 : j_remaining = j_initial / 2)
  (h5 : o_final = o_after_month + j_remaining) :
  o_final = 50 :=
sorry

end owen_turtles_l283_283444


namespace suma_work_rate_l283_283168

theorem suma_work_rate (Renu_rate Combined_rate Suma_rate : ℚ) :
  (Renu_rate = 1/6) →
  (Combined_rate = 1/3) →
  (Renu_rate + Suma_rate = Combined_rate) →
  (Suma_rate = 1/6) :=
by
  intros hRenu hCombined hSum
  calc
    Suma_rate = Combined_rate - Renu_rate : by linarith
          ... = 1/3 - 1/6        : by rw [hRenu, hCombined]
          ... = 1/6              : by norm_num
  done

end suma_work_rate_l283_283168


namespace no_other_way_to_construct_five_triangles_l283_283736

-- Setting up the problem conditions
def five_triangles_from_fifteen_sticks (sticks : List ℕ) : Prop :=
  sticks.length = 15 ∧
  let split_sticks := sticks.chunk (3) in
  split_sticks.length = 5 ∧
  ∀ triplet ∈ split_sticks, 
    let a := triplet[0]
    let b := triplet[1]
    let c := triplet[2] in
    a + b > c ∧ a + c > b ∧ b + c > a

-- The proof problem statement
theorem no_other_way_to_construct_five_triangles (sticks : List ℕ) :
  five_triangles_from_fifteen_sticks sticks → 
  ∀ (alternative : List (List ℕ)), 
    (alternative.length = 5 ∧ 
    ∀ triplet ∈ alternative, triplet.length = 3 ∧
      let a := triplet[0]
      let b := triplet[1]
      let c := triplet[2] in
      a + b > c ∧ a + c > b ∧ b + c > a) →
    alternative = sticks.chunk (3) :=
by
  sorry

end no_other_way_to_construct_five_triangles_l283_283736


namespace even_function_implies_a_eq_2_l283_283870

def f (x a : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2 (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → a = 2 :=
by
  sorry

end even_function_implies_a_eq_2_l283_283870


namespace f_is_even_iff_a_is_2_l283_283809

def f (a : ℝ) (x : ℝ) : ℝ := x * Real.exp x / (Real.exp (a * x) - 1)

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem f_is_even_iff_a_is_2 : (∀ x, f 2 (-x) = f 2 x) ↔ ∀ a, a = 2 := 
sorry

end f_is_even_iff_a_is_2_l283_283809


namespace determine_a_for_even_function_l283_283884

theorem determine_a_for_even_function :
  ∃ a : ℝ, (∀ x : ℝ, x ≠ 0 → (∃ f : ℝ → ℝ, 
  f x = x * exp x / (exp (a * x) - 1) ∧
  (∀ x : ℝ, f (-x) = f x))) → a = 2 :=
by
  sorry

end determine_a_for_even_function_l283_283884


namespace find_a_if_even_function_l283_283954

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x * exp x / (exp (a * x) - 1)

theorem find_a_if_even_function :
  (∀ x : ℝ, x ≠ 0 → f a (-x) = f a x) → a = 2 :=
by
  sorry

end find_a_if_even_function_l283_283954


namespace determine_a_for_even_function_l283_283878

theorem determine_a_for_even_function :
  ∃ a : ℝ, (∀ x : ℝ, x ≠ 0 → (∃ f : ℝ → ℝ, 
  f x = x * exp x / (exp (a * x) - 1) ∧
  (∀ x : ℝ, f (-x) = f x))) → a = 2 :=
by
  sorry

end determine_a_for_even_function_l283_283878


namespace find_complex_z_l283_283692

noncomputable def conjugate (z : Complex) : Complex := Complex.conj z

theorem find_complex_z :
  ∃ (z : Complex), (3 * z + 4 * conjugate z = 5 + 20 * Complex.I) ∧ (z = (5 / 7) - 20 * Complex.I) :=
by
  sorry

end find_complex_z_l283_283692


namespace even_function_implies_a_eq_2_l283_283850

theorem even_function_implies_a_eq_2 (a : ℝ) 
  (h : ∀ x : ℝ, f x = f (-x)) 
  (f : ℝ → ℝ := λ x, (x * exp x) / (exp (a * x) - 1)) : a = 2 :=
sorry

end even_function_implies_a_eq_2_l283_283850


namespace even_function_implies_a_eq_2_l283_283927

theorem even_function_implies_a_eq_2 (a : ℝ) : 
  (∀ x, (f : ℝ → ℝ) x = (λ x : ℝ, x * exp x / (exp (a * x) - 1)) x -> (f (-x) = f x)) -> a = 2 :=
by
  sorry

end even_function_implies_a_eq_2_l283_283927


namespace f_is_even_iff_a_is_2_l283_283811

def f (a : ℝ) (x : ℝ) : ℝ := x * Real.exp x / (Real.exp (a * x) - 1)

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem f_is_even_iff_a_is_2 : (∀ x, f 2 (-x) = f 2 x) ↔ ∀ a, a = 2 := 
sorry

end f_is_even_iff_a_is_2_l283_283811


namespace cannot_transform_l283_283130

def table : Type := matrix (fin 4) (fin 4) ℕ

def k_invariant (t : table) : ℤ :=
  let a := t 0 0
  let b := t 0 1
  let c := t 1 0
  let d := t 1 1
  (a + d) - (b + c)

theorem cannot_transform (A B : table) (transformation_rule : ∀ (t : table) (r c : fin 4), 
  table) :
  k_invariant A ≠ k_invariant B →
  ¬∃ (n : ℕ) (r_cols : (fin 4) × (fin 4)), B = transformation_rule A r_cols.fst r_cols.snd := 
sorry

end cannot_transform_l283_283130


namespace option_B_correct_option_C_correct_option_D_correct_l283_283149

theorem option_B_correct (a b m : ℝ) (h1: b > a) (h2: a > 0) (h3: m > 0) : 
    (a + m) / (b + m) > a / b := 
sorry

theorem option_C_correct (a b : ℝ) (h1: a > 0) (h2: b > 0) (h3: 2 * a + b = 1) : 
    (1 / (2 * a)) + (1 / b) ≥ 4 := 
sorry

theorem option_D_correct (a b c : ℝ) :
    a = 2.1^0.3 → b = 2.1^0.4 → c = 0.6^3.1 → c < a ∧ a < b :=
sorry

end option_B_correct_option_C_correct_option_D_correct_l283_283149


namespace smallest_angle_of_ABC_l283_283357

/-- In an acute-angled triangle ABC, the point O is the center of the circumcircle, 
    and the point H is the orthocenter. If the lines OH and BC are parallel and 
    BC = 4 * OH, then the smallest angle of triangle ABC is 30 degrees. -/
theorem smallest_angle_of_ABC (A B C O H : Point) (h_triangle: AcuteTriangle A B C)
  (h_circumcenter: IsCircumcenter O A B C) (h_orthocenter: IsOrthocenter H A B C)
  (h_parallel: Parallel (line_through O H) (line_through B C))
  (h_length: dist B C = 4 * dist O H) : 
  ∃ θ : ℝ, (θ > 0 ∧ θ < 90 ∧ θ = 30) := 
sorry

end smallest_angle_of_ABC_l283_283357


namespace smallest_k_for_no_real_roots_l283_283114

noncomputable def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem smallest_k_for_no_real_roots :
  ∃ (k : ℤ), (∀ (x : ℝ), (x * x + 6 * x + 2 * k : ℝ) ≠ 0 ∧ k ≥ 5) :=
by
  sorry

end smallest_k_for_no_real_roots_l283_283114


namespace triangle_perimeter_l283_283696

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

noncomputable def perimeter (A B C : ℝ × ℝ) : ℝ :=
  distance A B + distance B C + distance C A

theorem triangle_perimeter :
  let A := (1, 2)
  let B := (1, 9)
  let C := (6, 5)
  perimeter A B C = (7 : ℝ) + real.sqrt 41 + real.sqrt 34 := by
  sorry

end triangle_perimeter_l283_283696


namespace joe_purchased_360_gallons_l283_283023

def joe_initial_paint (P : ℝ) : Prop :=
  let first_week_paint := (1/4) * P
  let remaining_paint := (3/4) * P
  let second_week_paint := (1/2) * remaining_paint
  let total_used_paint := first_week_paint + second_week_paint
  total_used_paint = 225

theorem joe_purchased_360_gallons : ∃ P : ℝ, joe_initial_paint P ∧ P = 360 :=
by
  sorry

end joe_purchased_360_gallons_l283_283023


namespace number_of_sides_of_regular_polygon_l283_283620

variable {α : Type*}

noncomputable def exterior_angle_of_regular_polygon (n : ℕ) : ℝ := 360 / n

theorem number_of_sides_of_regular_polygon (exterior_angle : ℝ) (n : ℕ) (h₁ : exterior_angle = 18) (h₂ : exterior_angle_of_regular_polygon n = exterior_angle) : n = 20 :=
by {
  -- Translation of given conditions
  have sum_exterior_angles : 360 = n * exterior_angle_of_regular_polygon n,
  -- Based on h₂ and h₁ provided
  rw [h₂, h₁] at sum_exterior_angles,
  -- Perform necessary simplifications
  have : 20 = (360 / 18 : ℕ),
  simp,
}

end number_of_sides_of_regular_polygon_l283_283620


namespace simplify_complex_number_l283_283471

theorem simplify_complex_number (i : ℂ) (h : i^2 = -1) : i * (1 - i)^2 = 2 := by
  sorry

end simplify_complex_number_l283_283471


namespace problem_solution_l283_283278

theorem problem_solution : 
  ∃ x : ℕ, (x > 0 ∧ 7 * nat.choose 6 x = 20 * nat.choose 7 (x - 1) ∧ 
  (nat.choose 20 (20 - x) + nat.choose (17 + x) (x - 1) = 1330)) :=
by
  use 3
  split
  { sorry } -- proof that x > 0
  split
  { sorry } -- proof that 7 * nat.choose 6 3 = 20 * nat.choose 7 2
  { sorry } -- proof that nat.choose 20 17 + nat.choose 20 2 = 1330

end problem_solution_l283_283278


namespace f_decreasing_on_positive_f_expression_for_negative_l283_283481

-- Define the function and the conditions
noncomputable def f : ℝ → ℝ :=
  λ x, if x > 0 then (2 / x - 1) else if x < 0 then (-2 / x - 1) else 0

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

axiom f_even : even_function f

-- Part 1: Prove f(x) is decreasing on (0, ∞)
theorem f_decreasing_on_positive : ∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → f x1 > f x2 := by
  sorry

-- Part 2: The analytical expression of f(x) when x < 0
theorem f_expression_for_negative : ∀ x : ℝ, x < 0 → f x = -2 / x - 1 := by
  sorry

end f_decreasing_on_positive_f_expression_for_negative_l283_283481


namespace trajectory_equation_l283_283011

open Real

variables {x y : ℝ}

def is_symmetric (A B : ℝ × ℝ) (O : ℝ × ℝ) :=
  B = (2 * O.1 - A.1, 2 * O.2 - A.2)

def slope (P1 P2 : ℝ × ℝ) :=
  (P2.2 - P1.2) / (P2.1 - P1.1)

def trajectory_condition (A B P : ℝ × ℝ) :=
  slope A P * slope B P = -1 / 3

theorem trajectory_equation (A B P : ℝ × ℝ) 
  (h1: is_symmetric A B (0, 0))
  (h2: A = (-1, 1)) 
  (h3: B = (1, -1)) 
  (cond: trajectory_condition A B P):
  x^2 + 3 * y^2 = 4 ∧ x ≠ 1 ∧ x ≠ -1 :=
sorry

end trajectory_equation_l283_283011


namespace find_integer_in_range_divisible_by_18_l283_283262

theorem find_integer_in_range_divisible_by_18 
  (n : ℕ) (h1 : 900 ≤ n) (h2 : n ≤ 912) (h3 : n % 18 = 0) : n = 900 :=
sorry

end find_integer_in_range_divisible_by_18_l283_283262


namespace consecutive_integers_eq_l283_283078

theorem consecutive_integers_eq (a b c d e: ℕ) (h1: b = a + 1) (h2: c = a + 2) (h3: d = a + 3) (h4: e = a + 4) (h5: a^2 + b^2 + c^2 = d^2 + e^2) : a = 10 :=
by
  sorry

end consecutive_integers_eq_l283_283078


namespace find_x_l283_283414

theorem find_x :
  let N : ℕ := 2 ^ (2 ^ 2)
  ∃ (x : ℝ), N * N ^ (N ^ N) = 2 ^ (2 ^ x) → x = 66 :=
by
  let N := 2 ^ (2 ^ 2)
  use 66
  intro h
  sorry

end find_x_l283_283414


namespace determine_a_for_even_function_l283_283886

theorem determine_a_for_even_function :
  ∃ a : ℝ, (∀ x : ℝ, x ≠ 0 → (∃ f : ℝ → ℝ, 
  f x = x * exp x / (exp (a * x) - 1) ∧
  (∀ x : ℝ, f (-x) = f x))) → a = 2 :=
by
  sorry

end determine_a_for_even_function_l283_283886


namespace repeating_decimal_equiv_fraction_l283_283681

def repeating_decimal_to_fraction (x : ℚ) : ℚ := 781 / 111

theorem repeating_decimal_equiv_fraction : 7.036036036.... = 781 / 111 :=
  by
  -- lean does not support floating point literals with repeating decimals directly,
  -- need to construct the repeating decimal manually
  let x : ℚ := 7 + 36 / 999 + 36 / 999^2 + 36 / 999^3 + ...
  exactly repeating_decimal_to_fraction (x)

end repeating_decimal_equiv_fraction_l283_283681


namespace alex_is_15_l283_283374

variable (Inez_age : ℕ)
variable (Zack_age : ℕ)
variable (Jose_age : ℕ)
variable (Alex_age : ℕ)

axiom inez_is_18 : Inez_age = 18
axiom zack_is_5_years_older_than_inez : Zack_age = Inez_age + 5
axiom jose_is_6_years_younger_than_zack : Jose_age = Zack_age - 6
axiom alex_is_2_years_younger_than_jose : Alex_age = Jose_age - 2

theorem alex_is_15 : Alex_age = 15 :=
by
  rw [inez_is_18, zack_is_5_years_older_than_inez, jose_is_6_years_younger_than_zack, alex_is_2_years_younger_than_jose]
  sorry

end alex_is_15_l283_283374


namespace centroid_distance_relation_l283_283376

theorem centroid_distance_relation 
  (ABC : Triangle)
  (C_right : ABC.is_right_triangle C)
  (S_centroid : S = ABC.centroid) :
  SA^2 + SB^2 = 5 * SC^2 := 
  sorry

end centroid_distance_relation_l283_283376


namespace expr_is_square_l283_283057

theorem expr_is_square (n : ℕ) : ∃ k : ℕ, (5 + 10^(n+1)) * (∑ i in Finset.range (n + 1), 10^i) + 1 = k^2 :=
by
  have h1 : 1 + 10 + 10^2 + ... + 10^n = ∑ i in Finset.range (n + 1), 10^i, by sorry
  have h2 : ∑ i in Finset.range (n + 1), 10^i = (10^(n+1) - 1) / 9, by sorry
  have h3 : (5 + 10^(n+1)) * ((10^(n+1) - 1) / 9) + 1 = (10^(n+1) + 2)^2 / 9^2, by sorry
  existsi (10^(n+1) + 2) / 3
  sorry

end expr_is_square_l283_283057


namespace smallest_prime_factor_1953_l283_283141

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k, k ∣ n → k = 1 ∨ k = n

def smallest_prime_factor (n : ℕ) : ℕ :=
  Nat.find (λ p, is_prime p ∧ p ∣ n)

theorem smallest_prime_factor_1953 : smallest_prime_factor 1953 = 3 := by
  have h_odd : 1953 % 2 ≠ 0 := by norm_num
  have h_sum_div_3 : sum_of_digits 1953 % 3 = 0 := by norm_num
  sorry

end smallest_prime_factor_1953_l283_283141


namespace find_a_if_f_even_l283_283841

def f (a x : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem find_a_if_f_even (a : ℝ) (h : ∀ x : ℝ, x ≠ 0 → f a (-x) = f a x) : a = 2 :=
by 
  sorry

end find_a_if_f_even_l283_283841


namespace simplify_trig_expression_l283_283470

theorem simplify_trig_expression (x y : ℝ) : 
    cos (x + y) * sin y - sin (x + y) * cos y = - sin (x + y) :=
by
  sorry

end simplify_trig_expression_l283_283470


namespace part_a_part_b_l283_283415

variables {O I I_a : Point}
variables {A B C : Triangle}

-- Definitions of the circumcenter, incenter, and excenter
def circumcenter (t : Triangle) : Point := O
def incenter (t : Triangle) : Point := I
def excenter (t : Triangle) : Point := I_a

-- Distances and radii
variables {d R r d_a r_a : ℝ}

def OI (t : Triangle) := (circumcenter t).distance (incenter t)
def OI_a (t : Triangle) := (circumcenter t).distance (excenter t)
def circumradius (t : Triangle) := R
def inradius (t : Triangle) := r
def exradius (t : Triangle) := r_a

theorem part_a (t : Triangle) : 
  OI t = d → circumradius t = R → inradius t = r → d^2 = R^2 - 2 * R * r := sorry

theorem part_b (t : Triangle) : 
  OI_a t = d_a → circumradius t = R → exradius t = r_a → d_a^2 = R^2 + 2 * R * r_a := sorry

end part_a_part_b_l283_283415


namespace range_of_a_l283_283311

theorem range_of_a (a : ℝ) (h1 : a ≤ 1)
(h2 : ∃ n₁ n₂ n₃ : ℤ, a ≤ n₁ ∧ n₁ < n₂ ∧ n₂ < n₃ ∧ n₃ ≤ 2 - a
  ∧ (∀ x : ℤ, a ≤ x ∧ x ≤ 2 - a → x = n₁ ∨ x = n₂ ∨ x = n₃)) :
  -1 < a ∧ a ≤ 0 :=
by
  sorry

end range_of_a_l283_283311


namespace Jermaine_more_than_Terrence_l283_283021

theorem Jermaine_more_than_Terrence :
  ∀ (total_earnings Terrence_earnings Emilee_earnings : ℕ),
    total_earnings = 90 →
    Terrence_earnings = 30 →
    Emilee_earnings = 25 →
    (total_earnings - Terrence_earnings - Emilee_earnings) - Terrence_earnings = 5 := by
  sorry

end Jermaine_more_than_Terrence_l283_283021


namespace sum_inequality_l283_283303

theorem sum_inequality (n : ℕ) (h_n : (0.2 * n : ℝ).denom = 1) (x : ℕ → ℝ)
  (h_pos : ∀ i, 1 ≤ i ∧ i ≤ n → 0 < x i)
  (h_eq : (∑ i in finset.range(n), 1 / (i + 1) * (x (i + 1))^2) 
          + 2 * ∑ i in finset.range(n), ∑ j in finset.Ico(i + 1, n + 1), 1 / j * x (i + 1) * x j = 1) :
  ∑ i in finset.range(n), (i + 1) * x (i + 1) ≤ real.sqrt (4 / 3 * n ^ 3 - 1 / 3 * n) :=
sorry

end sum_inequality_l283_283303


namespace find_a_of_even_function_l283_283770

noncomputable def f (a x : ℝ) : ℝ := (x * exp x) / (exp (a * x) - 1)

theorem find_a_of_even_function (hf : ∀ x : ℝ, f a (-x) = f a x) : a = 2 :=
by {
    sorry
}

end find_a_of_even_function_l283_283770


namespace eq_a_2_l283_283958

def f (a x : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

-- Definition of even function: ∀ x, f(-x) = f(x)
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem eq_a_2 (a : ℝ) : (even_function (f a) → a = 2) ∧ (a = 2 → even_function (f a)) :=
by
  sorry

end eq_a_2_l283_283958


namespace find_a_l283_283994

noncomputable def f (x a : ℝ) : ℝ := (x * (Real.exp x)) / (Real.exp (a * x) - 1)

theorem find_a (a : ℝ) (h : ∀ x, f x a = f (-x) a) : a = 2 :=
begin
  sorry
end

end find_a_l283_283994


namespace bottles_produced_by_10_machines_in_4_minutes_l283_283061

variable (rate_per_machine : ℕ)
variable (total_bottles_per_minute_six_machines : ℕ := 240)
variable (number_of_machines : ℕ := 6)
variable (new_number_of_machines : ℕ := 10)
variable (time_in_minutes : ℕ := 4)

theorem bottles_produced_by_10_machines_in_4_minutes :
  rate_per_machine = total_bottles_per_minute_six_machines / number_of_machines →
  (new_number_of_machines * rate_per_machine * time_in_minutes) = 1600 := 
sorry

end bottles_produced_by_10_machines_in_4_minutes_l283_283061


namespace james_total_spent_l283_283178

theorem james_total_spent:
  let entry_fee := 20
  let friends_drinks := 2 * 5 * 6
  let self_drinks := 6 * 6
  let food := 14
  let total_before_tip := entry_fee + friends_drinks + self_drinks + food
  let tip := (friends_drinks + self_drinks + food) * 0.30
  total_before_tip + tip = 163 :=
by
  let entry_fee := 20
  let friends_drinks := 2 * 5 * 6
  let self_drinks := 6 * 6
  let food := 14
  let total_before_tip := entry_fee + friends_drinks + self_drinks + food
  let tip := (friends_drinks + self_drinks + food) * 0.30
  have entry_fee_val : entry_fee = 20 := by rfl
  have friends_drinks_val : friends_drinks = 60 := by rfl
  have self_drinks_val : self_drinks = 36 := by rfl
  have food_val : food = 14 := by rfl
  have total_before_tip_val : total_before_tip = 130 := by rfl
  have tip_val : tip = 33 := by rfl
  show total_before_tip + tip = 163, by
    calc total_before_tip + tip
          = 130 + 33 : by rw [total_before_tip_val, tip_val]
      ... = 163     : by norm_num

end james_total_spent_l283_283178


namespace no_alternative_way_to_construct_triangles_l283_283725

-- Define the triangle inequality condition
def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the set of stick lengths for each required triangle
def valid_triangle_sets (stick_set : Set (Set ℕ)) : Prop :=
  stick_set = { {2, 3, 4}, {20, 30, 40}, {200, 300, 400}, {2000, 3000, 4000}, {20000, 30000, 40000} }

-- Define the problem statement
theorem no_alternative_way_to_construct_triangles (s : Set ℕ) (h : s.card = 15) :
  (∀ t ∈ (s.powerset.filter (λ t, t.card = 3)), triangle_inequality t.some (finset_some_spec t) ∧ (∃ sets : Set (Set ℕ), sets.card = 5 ∧ ∀ u ∈ sets, u.card = 3 ∧ triangle_inequality u.some (finset_some_spec u)))
  → valid_triangle_sets (s.powerset.filter (λ t, t.card = 3)) := 
sorry

end no_alternative_way_to_construct_triangles_l283_283725


namespace joy_tape_deficit_l283_283027

noncomputable def tape_needed_field (width length : ℕ) : ℕ :=
2 * (length + width)

noncomputable def tape_needed_trees (num_trees circumference : ℕ) : ℕ :=
num_trees * circumference

def tape_total_needed (tape_field tape_trees : ℕ) : ℕ :=
tape_field + tape_trees

theorem joy_tape_deficit (tape_has : ℕ) (tape_field tape_trees: ℕ) : ℤ :=
tape_has - (tape_field + tape_trees)

example : joy_tape_deficit 180 (tape_needed_field 35 80) (tape_needed_trees 3 5) = -65 := by
sorry

end joy_tape_deficit_l283_283027


namespace num_of_appliances_l283_283635

theorem num_of_appliances (purchase_price sell_price total_profit : ℤ) 
    (h_purchase : purchase_price = 230) 
    (h_sell : sell_price = 250) 
    (h_profit : total_profit = 680) : 
    ∃ (n : ℤ), n = total_profit / (sell_price - purchase_price) ∧ n = 34 := 
by
  have profit_per_item : ℤ := sell_price - purchase_price
  rw [h_purchase, h_sell] at profit_per_item
  have num_items : ℤ := total_profit / profit_per_item
  rw [h_profit, sub_eq_add_neg, add_comm] at num_items
  exact ⟨num_items, ⟨rfl, rfl⟩⟩

end num_of_appliances_l283_283635


namespace paperclips_capacity_l283_283581

theorem paperclips_capacity {v1 v2 : ℝ} (c1 : ℕ) (h1 : v1 = 25) (h2 : c1 = 50) (h3 : v2 = 100) (h4 : ∀ (v1 v2 : ℝ) (c1 : ℕ), c1 * √(v2 / v1) = c2): 
  ∃ (c2 : ℕ), c2 = 100 :=
by
  sorry

end paperclips_capacity_l283_283581


namespace find_a_if_even_function_l283_283948

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x * exp x / (exp (a * x) - 1)

theorem find_a_if_even_function :
  (∀ x : ℝ, x ≠ 0 → f a (-x) = f a x) → a = 2 :=
by
  sorry

end find_a_if_even_function_l283_283948


namespace difference_of_squares_l283_283418

/-- Lean 4 statement of the proof problem -/
theorem difference_of_squares (x y : ℝ) 
    (h1 : x = 3001^500 - 3001^(-500))
    (h2 : y = 3001^500 + 3001^(-500)) : 
    x^2 - y^2 = -4 :=
by
  sorry

end difference_of_squares_l283_283418


namespace no_alternate_way_to_form_triangles_l283_283732

noncomputable def canFormTriangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def validSetOfTriples : List (ℕ × ℕ × ℕ) :=
  [(2, 3, 4), (20, 30, 40), (200, 300, 400), (2000, 3000, 4000), (20000, 30000, 40000)]

theorem no_alternate_way_to_form_triangles (stickSet : List (ℕ × ℕ × ℕ)) :
  (∀ (t : ℕ × ℕ × ℕ), t ∈ stickSet → canFormTriangle t.1 t.2 t.3) →
  length stickSet = 5 →
  stickSet = validSetOfTriples ->
  (∀ alternateSet : List (ℕ × ℕ × ℕ),
    ∀ (t : ℕ × ℕ × ℕ), t ∈ alternateSet → canFormTriangle t.1 t.2 t.3 →
    length alternateSet = 5 →
    duplicate_stick_alternateSet = duplicate_stick_stickSet) :=
sorry

end no_alternate_way_to_form_triangles_l283_283732


namespace eq_a_2_l283_283971

def f (a x : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

-- Definition of even function: ∀ x, f(-x) = f(x)
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem eq_a_2 (a : ℝ) : (even_function (f a) → a = 2) ∧ (a = 2 → even_function (f a)) :=
by
  sorry

end eq_a_2_l283_283971


namespace cevian_lines_meet_at_one_point_l283_283056

-- Define the structure of a triangle with an incircle
structure Triangle :=
  (A B C : Point)
  (incircle : Circle)
  (D E F : Point) -- Points where incircle touches the sides BC, CA, AB respectively
  (tangent_D : IsTangent incircle ⟨B, C⟩ D)
  (tangent_E : IsTangent incircle ⟨C, A⟩ E)
  (tangent_F : IsTangent incircle ⟨A, B⟩ F)

-- Prove the desired assertion that the lines intersect at a single point
theorem cevian_lines_meet_at_one_point (T : Triangle) : 
  ∃ P : Point, Line_through T.A T.D = Line_through T.B T.E ∧ Line_through T.B T.E = Line_through T.C T.F := 
by
  sorry

end cevian_lines_meet_at_one_point_l283_283056


namespace angle_BAC_measure_l283_283371

variable (A B C X Y : Type)
variables (angle_ABC angle_BAC : ℝ)
variables (len_AX len_XY len_YB len_BC : ℝ)

theorem angle_BAC_measure 
  (h1 : AX = XY) 
  (h2 : XY = YB) 
  (h3 : XY = 2 * AX) 
  (h4 : angle_ABC = 150) :
  angle_BAC = 26.25 :=
by
  -- The proof would be required here.
  -- Following the statement as per instructions.
  sorry

end angle_BAC_measure_l283_283371


namespace total_surface_area_eq_l283_283586

-- Definitions for the given conditions
def height (cylinder : Type) : ℝ := 12
def radius (cylinder : Type) : ℝ := 5

-- Theorem statement
theorem total_surface_area_eq (cylinder : Type) :
  let h := height cylinder
  let r := radius cylinder
  2 * Real.pi * r^2 + 2 * Real.pi * r * h = 170 * Real.pi := sorry

end total_surface_area_eq_l283_283586


namespace find_a_for_even_function_l283_283828

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def given_function (a : ℝ) (x : ℝ) : ℝ :=
  x * real.exp x / (real.exp (a * x) - 1)

theorem find_a_for_even_function :
  ∀ (a : ℝ), is_even_function (given_function a) ↔ a = 2 :=
by 
  intro a
  sorry

end find_a_for_even_function_l283_283828


namespace Joey_return_speed_l283_283018

/-
Conditions:
1. Joey runs a 6-mile long route to deliver packages and returns along the same path.
2. It takes Joey 1 hour to run the 6-mile route (to deliver packages).
3. The average speed of the round trip is 8 miles/hour.
-/
lemma return_speed (distance : ℝ) (delivery_time : ℝ) (average_speed : ℝ) (return_speed : ℝ) :
  distance = 6 → delivery_time = 1 → average_speed = 8 → 
  return_speed = (2 * average_speed * delivery_time) - delivery_time * average_speed :=
by
  sorry

theorem Joey_return_speed : 
  (return_speed 6 1 8) = 12 :=
by
  sorry

end Joey_return_speed_l283_283018


namespace mass_percentage_ba_in_bao_l283_283269

-- Define the constants needed in the problem
def molarMassBa : ℝ := 137.33
def molarMassO : ℝ := 16.00

-- Calculate the molar mass of BaO
def molarMassBaO : ℝ := molarMassBa + molarMassO

-- Express the problem as a Lean theorem for proof
theorem mass_percentage_ba_in_bao : 
  (molarMassBa / molarMassBaO) * 100 = 89.55 := by
  sorry

end mass_percentage_ba_in_bao_l283_283269


namespace hyperbola_k_range_l283_283312

theorem hyperbola_k_range (k : ℝ) : 
  (∀ x y : ℝ, (x^2 / (k + 2) - y^2 / (5 - k) = 1)) → (-2 < k ∧ k < 5) :=
by
  sorry

end hyperbola_k_range_l283_283312


namespace probability_of_red_greater_than_white_l283_283504

variables (num_red_balls num_white_balls : ℕ) (prob : ℚ)

axiom box_contains : num_red_balls = 3 ∧ num_white_balls = 3

axiom fair_die : ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 6 → prob = 1 / 6

noncomputable def probability (num_red num_white : ℕ) : ℚ :=
  -- Here would be the probability calculation logic, skipped for brevity

theorem probability_of_red_greater_than_white :
  (box_contains num_red_balls num_white_balls) →
  (fair_die 6) →
  (probability 3 3 * (
    probability 1 1 * (1/6) * (3 * 3/15) +
    0 + 
    probability 2 2 * (1/6) * (3 * 3/15) +
    0 +
    probability 3 3 * (1/6) *
    (1 - (1/10 + 1/10 + 1/6) / 2)
  ) = 19 / 60 :=
by 
  intros h1 h2
  sorry

end probability_of_red_greater_than_white_l283_283504


namespace miriam_pushups_friday_l283_283430

-- Definitions and conditions
def pushups_monday : ℕ := 5

def pushups_tuesday : ℕ := 
  (1.4 : ℚ) * pushups_monday

def pushups_wednesday : ℕ := 
  2 * pushups_monday

def total_pushups_mon_to_wed : ℕ :=
  pushups_monday + pushups_tuesday + pushups_wednesday

def pushups_thursday : ℕ :=
  (total_pushups_mon_to_wed : ℚ) / 2

def total_pushups_mon_to_thu : ℕ :=
  pushups_monday + pushups_tuesday + pushups_wednesday + pushups_thursday

def pushups_friday : ℕ :=
  total_pushups_mon_to_thu

-- Theorem to be proven
theorem miriam_pushups_friday : pushups_friday = 33 := 
by
  sorry

end miriam_pushups_friday_l283_283430


namespace rational_powers_implies_rational_a_rational_powers_implies_rational_b_l283_283406

open Real

theorem rational_powers_implies_rational_a (x : ℝ) :
  (∃ r₁ r₂ : ℚ, x^7 = r₁ ∧ x^12 = r₂) → (∃ q : ℚ, x = q) :=
by
  sorry

theorem rational_powers_implies_rational_b (x : ℝ) :
  (∃ r₁ r₂ : ℚ, x^9 = r₁ ∧ x^12 = r₂) → (∃ q : ℚ, x = q) :=
by
  sorry

end rational_powers_implies_rational_a_rational_powers_implies_rational_b_l283_283406


namespace jan_total_amount_paid_l283_283389

-- Define the conditions
def dozen := 12
def roses_in_dozen := 5 * dozen
def cost_per_rose := 6
def discount := 0.8

-- Define the expected result
def total_amount_paid := 288

-- Theorem statement to prove the total amount paid
theorem jan_total_amount_paid :
  roses_in_dozen * cost_per_rose * discount = total_amount_paid := 
by
  sorry

end jan_total_amount_paid_l283_283389


namespace min_value_frac_l283_283298

theorem min_value_frac (x y : ℝ) (h : x^2 + y^2 = 1) : 
  ∃ (θ : ℝ) (H : 0 ≤ θ ∧ θ < 2 * Real.pi), 
  (∃ (t : ℝ) (ht : -Real.sqrt 2 ≤ t ∧ t ≤ Real.sqrt 2), 
  ∃ (u : ℝ) (u = t^2 - 1), 
  (t = Real.sin θ + Real.cos θ ∧ u = 2 * Real.sin θ * Real.cos θ) ∧ 
  ∀ (val : ℝ), (val = t - 1) → val = -1 - Real.sqrt 2).  

end min_value_frac_l283_283298


namespace even_function_implies_a_eq_2_l283_283926

theorem even_function_implies_a_eq_2 (a : ℝ) : 
  (∀ x, (f : ℝ → ℝ) x = (λ x : ℝ, x * exp x / (exp (a * x) - 1)) x -> (f (-x) = f x)) -> a = 2 :=
by
  sorry

end even_function_implies_a_eq_2_l283_283926


namespace determine_a_for_even_function_l283_283882

theorem determine_a_for_even_function :
  ∃ a : ℝ, (∀ x : ℝ, x ≠ 0 → (∃ f : ℝ → ℝ, 
  f x = x * exp x / (exp (a * x) - 1) ∧
  (∀ x : ℝ, f (-x) = f x))) → a = 2 :=
by
  sorry

end determine_a_for_even_function_l283_283882


namespace avg_speed_climbing_proof_l283_283051

-- Definitions of the conditions
variables (D : ℝ) (t_up t_down : ℝ) (v_avg_whole_journey : ℝ)

-- Given conditions
def conditions : Prop :=
  (t_up = 4) ∧
  (t_down = 2) ∧
  (v_avg_whole_journey = 1.5)

-- The total distance for the whole journey
def total_distance : ℝ := 2 * D

-- The total time for the whole journey
def total_time : ℝ := t_up + t_down

-- The average speed for the whole journey
def avg_speed_whole_journey : ℝ := total_distance / total_time

-- The average speed while climbing to the top
def avg_speed_climbing : ℝ := D / t_up

-- The theorem to prove
theorem avg_speed_climbing_proof (h : conditions) : avg_speed_climbing D t_up = 1.125 :=
by
  -- Insert proof here 
  sorry

end avg_speed_climbing_proof_l283_283051


namespace find_a_l283_283997

noncomputable def f (x a : ℝ) : ℝ := (x * (Real.exp x)) / (Real.exp (a * x) - 1)

theorem find_a (a : ℝ) (h : ∀ x, f x a = f (-x) a) : a = 2 :=
begin
  sorry
end

end find_a_l283_283997


namespace Jeff_total_laps_l283_283020

theorem Jeff_total_laps (laps_saturday : ℕ) (laps_sunday_morning : ℕ) (laps_remaining : ℕ)
  (h1 : laps_saturday = 27) (h2 : laps_sunday_morning = 15) (h3 : laps_remaining = 56) :
  (laps_saturday + laps_sunday_morning + laps_remaining) = 98 := 
by
  sorry

end Jeff_total_laps_l283_283020


namespace mowing_difference_l283_283029

-- Define the number of times mowed in spring and summer
def mowedSpring : ℕ := 8
def mowedSummer : ℕ := 5

-- Prove the difference between spring and summer mowing is 3
theorem mowing_difference : mowedSpring - mowedSummer = 3 := by
  sorry

end mowing_difference_l283_283029


namespace number_of_sides_of_regular_polygon_l283_283621

variable {α : Type*}

noncomputable def exterior_angle_of_regular_polygon (n : ℕ) : ℝ := 360 / n

theorem number_of_sides_of_regular_polygon (exterior_angle : ℝ) (n : ℕ) (h₁ : exterior_angle = 18) (h₂ : exterior_angle_of_regular_polygon n = exterior_angle) : n = 20 :=
by {
  -- Translation of given conditions
  have sum_exterior_angles : 360 = n * exterior_angle_of_regular_polygon n,
  -- Based on h₂ and h₁ provided
  rw [h₂, h₁] at sum_exterior_angles,
  -- Perform necessary simplifications
  have : 20 = (360 / 18 : ℕ),
  simp,
}

end number_of_sides_of_regular_polygon_l283_283621


namespace length_FJ_is_35_l283_283646

noncomputable def length_of_FJ (h : ℝ) : ℝ :=
  let FG := 50
  let HI := 20
  let trapezium_area := (1 / 2) * (FG + HI) * h
  let half_trapezium_area := trapezium_area / 2
  let JI_area := (1 / 2) * 35 * h
  35

theorem length_FJ_is_35 (h : ℝ) : length_of_FJ h = 35 :=
  sorry

end length_FJ_is_35_l283_283646


namespace area_ratio_of_hexagon_l283_283007

theorem area_ratio_of_hexagon (ABCDEF : Hexagon)
  (P Q R : Point)
  (hP : midpoint (ABCDEF.side AB) P)
  (hQ : midpoint (ABCDEF.side DE) Q)
  (hR : midpoint (ABCDEF.diagonal AC) R) :
  (area (triangle B P R))/(area (triangle D R Q)) = 1 :=
sorry

end area_ratio_of_hexagon_l283_283007


namespace find_a_l283_283995

noncomputable def f (x a : ℝ) : ℝ := (x * (Real.exp x)) / (Real.exp (a * x) - 1)

theorem find_a (a : ℝ) (h : ∀ x, f x a = f (-x) a) : a = 2 :=
begin
  sorry
end

end find_a_l283_283995


namespace balloon_height_after_movements_l283_283234

def sequence_of_movements (initial_height : ℕ) : ℕ :=
  let after_first_ascend := initial_height + 6
  let after_first_descend := after_first_ascend - 2
  let after_second_ascend := after_first_descend + 3
  after_second_ascend - 2

theorem balloon_height_after_movements
  (initial_height : ℕ) :
  sequence_of_movements initial_height = initial_height + 5 :=
by
  simp [sequence_of_movements]
  sorry

end balloon_height_after_movements_l283_283234


namespace repeating_decimal_to_fraction_l283_283679

theorem repeating_decimal_to_fraction :
  (7.036).repeat == 781 / 111 :=
by
  sorry

end repeating_decimal_to_fraction_l283_283679


namespace find_a_even_function_l283_283983

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def given_function (a : ℝ) (x : ℝ) : ℝ :=
  x * Real.exp x / (Real.exp (a * x) - 1)

theorem find_a_even_function :
  (is_even_function (given_function 2)) → true :=
by
  intro h
  sorry

end find_a_even_function_l283_283983


namespace even_function_implies_a_eq_2_l283_283910

def f (x : ℝ) (a : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2 (a : ℝ) (h_even : ∀ x, f (-x) a = f x a) : a = 2 :=
by
  intro x
  sorry

end even_function_implies_a_eq_2_l283_283910


namespace determine_a_for_even_function_l283_283881

theorem determine_a_for_even_function :
  ∃ a : ℝ, (∀ x : ℝ, x ≠ 0 → (∃ f : ℝ → ℝ, 
  f x = x * exp x / (exp (a * x) - 1) ∧
  (∀ x : ℝ, f (-x) = f x))) → a = 2 :=
by
  sorry

end determine_a_for_even_function_l283_283881


namespace findNumberOfIntegers_l283_283695

def arithmeticSeq (a d n : ℕ) : ℕ :=
  a + d * n

def isInSeq (n : ℕ) : Prop :=
  ∃ k : ℕ, k ≤ 33 ∧ n = arithmeticSeq 1 3 k

def validInterval (n : ℕ) : Bool :=
  (n + 1) / 3 % 2 = 1

theorem findNumberOfIntegers :
  ∃ count : ℕ, count = 66 ∧ (∀ n : ℕ, 1 ≤ n ∧ n ≤ 100 ∧ ¬isInSeq n → validInterval n = true) :=
sorry

end findNumberOfIntegers_l283_283695


namespace alice_marble_groups_l283_283639

-- Define the number of each colored marble Alice has
def pink_marble := 1
def blue_marble := 1
def white_marble := 1
def black_marbles := 4

-- The function to count the number of different groups of two marbles Alice can choose
noncomputable def count_groups : Nat :=
  let total_colors := 4  -- Pink, Blue, White, and one representative black
  1 + (total_colors.choose 2)

-- The theorem statement 
theorem alice_marble_groups : count_groups = 7 := by 
  sorry

end alice_marble_groups_l283_283639


namespace remainder_div_5_l283_283563

theorem remainder_div_5 (n : ℕ): (∃ k : ℤ, n = 10 * k + 7) → (∃ m : ℤ, n = 5 * m + 2) :=
by
  sorry

end remainder_div_5_l283_283563


namespace even_function_implies_a_eq_2_l283_283907

def f (a x : ℝ) : ℝ := (x * exp x) / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2
  (h : ∀ x : ℝ, f a x = f a (-x)) : a = 2 := 
sorry

end even_function_implies_a_eq_2_l283_283907


namespace chocolates_remaining_l283_283330

def chocolates := 24
def chocolates_first_day := 4
def chocolates_eaten_second_day := (2 * chocolates_first_day) - 3
def chocolates_eaten_third_day := chocolates_first_day - 2
def chocolates_eaten_fourth_day := chocolates_eaten_third_day - 1

theorem chocolates_remaining :
  chocolates - (chocolates_first_day + chocolates_eaten_second_day + chocolates_eaten_third_day + chocolates_eaten_fourth_day) = 12 := by
  sorry

end chocolates_remaining_l283_283330


namespace b_more_than_c_l283_283162

variable (a b c : ℝ)

-- Given conditions
def total_subscription : Prop := a + b + c = 50000
def a_more_than_b : Prop := a = b + 4000
def total_profit : Prop := 36000
def a_profit : Prop := 15120

-- Required to prove
theorem b_more_than_c (h1 : total_subscription a b c) (h2 : a_more_than_b a b) (h3 : (15120 / 36000) = ((b + 4000) / 50000)) :
  b = c + 5000 :=
sorry

end b_more_than_c_l283_283162


namespace even_function_implies_a_is_2_l283_283785

noncomputable def f (a x : ℝ) : ℝ := (x * Real.exp x) / (Real.exp (a * x) - 1)

theorem even_function_implies_a_is_2 (a : ℝ) 
  (h : ∀ x : ℝ, f a x = f a (-x)) : a = 2 := 
by
  sorry

end even_function_implies_a_is_2_l283_283785


namespace cube_root_expression_is_integer_l283_283654

theorem cube_root_expression_is_integer :
  (∛(2^9 * 5^3 * 7^3) : ℝ) = 280 := by
  sorry

end cube_root_expression_is_integer_l283_283654


namespace determine_a_for_even_function_l283_283885

theorem determine_a_for_even_function :
  ∃ a : ℝ, (∀ x : ℝ, x ≠ 0 → (∃ f : ℝ → ℝ, 
  f x = x * exp x / (exp (a * x) - 1) ∧
  (∀ x : ℝ, f (-x) = f x))) → a = 2 :=
by
  sorry

end determine_a_for_even_function_l283_283885


namespace sixty_five_percent_of_forty_minus_four_fifths_of_twenty_five_l283_283163

theorem sixty_five_percent_of_forty_minus_four_fifths_of_twenty_five :
  (65/100) * 40 - (4/5) * 25 = 6 :=
by
  calc 
    (65 / 100) * 40 
      : rfl
    ... - (4 / 5) * 25 
      : rfl
    ... = 6 
      : sorry

end sixty_five_percent_of_forty_minus_four_fifths_of_twenty_five_l283_283163


namespace no_alternative_way_to_construct_triangles_l283_283722

-- Define the triangle inequality condition
def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the set of stick lengths for each required triangle
def valid_triangle_sets (stick_set : Set (Set ℕ)) : Prop :=
  stick_set = { {2, 3, 4}, {20, 30, 40}, {200, 300, 400}, {2000, 3000, 4000}, {20000, 30000, 40000} }

-- Define the problem statement
theorem no_alternative_way_to_construct_triangles (s : Set ℕ) (h : s.card = 15) :
  (∀ t ∈ (s.powerset.filter (λ t, t.card = 3)), triangle_inequality t.some (finset_some_spec t) ∧ (∃ sets : Set (Set ℕ), sets.card = 5 ∧ ∀ u ∈ sets, u.card = 3 ∧ triangle_inequality u.some (finset_some_spec u)))
  → valid_triangle_sets (s.powerset.filter (λ t, t.card = 3)) := 
sorry

end no_alternative_way_to_construct_triangles_l283_283722


namespace surface_area_SABC_l283_283369

def isosceles_right_triangle (a b c : ℝ) : Prop :=
a^2 + b^2 = c^2

def equilateral_triangle (x y z : ℝ) : Prop :=
x = y ∧ y = z

variables (S A B C : Type) [metric_space S] {d SA SB SC AB BC CA : ℝ}

def tetrahedron_faces : Prop :=
isosceles_right_triangle SA SB d ∧
isosceles_right_triangle SB SC d ∧
isosceles_right_triangle SC SA d ∧
equilateral_triangle AB BC CA

def side_length_eq_two : Prop :=
AB = 2 ∧ BC = 2 ∧ CA = 2

def surface_area_tetrahedron : ℝ :=
3 + real.sqrt 3

theorem surface_area_SABC
  (h1 : tetrahedron_faces S A B C)
  (h2 : side_length_eq_two S A B C) :
  surface_area_tetrahedron S A B C = 3 + real.sqrt 3 :=
sorry

end surface_area_SABC_l283_283369


namespace constant_term_expansion_l283_283087

def binomial_term (a b : ℂ) (n k : ℕ) : ℂ := complex.C (binomial n k) * a ^ (n - k) * b ^ k

def expansion (x : ℂ) : ℂ := 
  (x + 3/x) * (binomial_term x (-2/x) 5 0 +
               binomial_term x (-2/x) 5 1 +
               binomial_term x (-2/x) 5 2 +
               binomial_term x (-2/x) 5 3 +
               binomial_term x (-2/x) 5 4 +
               binomial_term x (-2/x) 5 5)

theorem constant_term_expansion : 
  ∀ x : ℂ, x ≠ 0 → ∃ c : ℂ, expansion x = c * x^0 ∧ c = 40 := 
begin
  intros,
  sorry
end

end constant_term_expansion_l283_283087


namespace coeff_x4_in_poly_expansion_is_neg160_l283_283478

-- Define the given polynomial.
def poly := x * (2 * x - 1) ^ 6

-- State that the coefficient of x^4 in this expansion is -160.
theorem coeff_x4_in_poly_expansion_is_neg160 :
  coefficient (poly.expand) 4 = -160 :=
sorry

end coeff_x4_in_poly_expansion_is_neg160_l283_283478


namespace base_b_cube_l283_283103

theorem base_b_cube (b : ℕ) : (b > 4) → (∃ n : ℕ, (b^2 + 4 * b + 4 = n^3)) ↔ (b = 5 ∨ b = 6) :=
by
  sorry

end base_b_cube_l283_283103


namespace find_a_of_even_function_l283_283774

noncomputable def f (a x : ℝ) : ℝ := (x * exp x) / (exp (a * x) - 1)

theorem find_a_of_even_function (hf : ∀ x : ℝ, f a (-x) = f a x) : a = 2 :=
by {
    sorry
}

end find_a_of_even_function_l283_283774


namespace find_x_converges_to_l283_283685

noncomputable def series_sum (x : ℝ) : ℝ := ∑' n : ℕ, (4 * (n + 1) - 2) * x^n

theorem find_x_converges_to (x : ℝ) (h : |x| < 1) :
  series_sum x = 60 → x = 29 / 30 :=
by
  sorry

end find_x_converges_to_l283_283685


namespace part1_part2_part2_maximum_l283_283421

-- Define the conditions and the function
def cond (x y : ℝ) := x^2 + x * y + y^2 = 1
def F (x y : ℝ) := x^3 * y + x * y^3

-- Part (1): Prove that F(x, y) ≥ -2
theorem part1 (x y : ℝ) (h : cond x y) : F x y ≥ -2 := sorry

-- Part (2): Prove that the maximum value of F(x, y) is 1 / 4
theorem part2 (x y : ℝ) (h : cond x y) : F x y ≤ 1 / 4 := sorry

-- Additional theorem to state the exact maximum value
theorem part2_maximum (x y : ℝ) (h : cond x y) : ∃ x y : ℝ, cond x y ∧ F x y = 1 / 4 := sorry

end part1_part2_part2_maximum_l283_283421


namespace sin_2x_vs_2sinx_l283_283148

theorem sin_2x_vs_2sinx (x : ℝ) (n : ℤ) :
  2 * sin x * cos x ≥ 2 * sin x ↔ 2 * π * n - π ≤ x ∧ x ≤ 2 * π * n ∨
  2 * sin x * cos x ≤ 2 * sin x ↔ 2 * π * n ≤ x ∧ x ≤ π + 2 * π * n :=
by {
  sorry
}

end sin_2x_vs_2sinx_l283_283148


namespace find_ab_and_m_l283_283764

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + b * x^2

theorem find_ab_and_m (a b m : ℝ) (P : ℝ × ℝ)
  (h1 : P = (-1, -2))
  (h2 : ∀ (x : ℝ), (3 * a * x^2 + 2 * b * x) = -1/3 ↔ x = -1)
  (h3 : ∀ (x : ℝ), f a b x = a * x ^ 3 + b * x ^ 2)
  : (a = -13/3 ∧ b = -19/3) ∧ (0 < m ∧ m < 38/39) :=
sorry

end find_ab_and_m_l283_283764


namespace average_of_two_integers_l283_283085

theorem average_of_two_integers {A B C D : ℕ} (h1 : A + B + C + D = 200) (h2 : C ≤ 130) : (A + B) / 2 = 35 :=
by
  sorry

end average_of_two_integers_l283_283085


namespace f_is_even_iff_a_is_2_l283_283806

def f (a : ℝ) (x : ℝ) : ℝ := x * Real.exp x / (Real.exp (a * x) - 1)

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem f_is_even_iff_a_is_2 : (∀ x, f 2 (-x) = f 2 x) ↔ ∀ a, a = 2 := 
sorry

end f_is_even_iff_a_is_2_l283_283806


namespace circle_radius_l283_283165

theorem circle_radius (D : ℝ) (h : D = 14) : (D / 2) = 7 :=
by
  sorry

end circle_radius_l283_283165


namespace term_count_expansion_l283_283094

theorem term_count_expansion (x y z : ℕ) :
  (∑ a in finset.range 1006, (2009 - 2 * a)) = 1005^2 :=
by sorry

end term_count_expansion_l283_283094


namespace eq_a_2_l283_283965

def f (a x : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

-- Definition of even function: ∀ x, f(-x) = f(x)
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem eq_a_2 (a : ℝ) : (even_function (f a) → a = 2) ∧ (a = 2 → even_function (f a)) :=
by
  sorry

end eq_a_2_l283_283965


namespace num_men_in_first_group_l283_283073

-- Define conditions
def work_rate_first_group (x : ℕ) : ℚ := (1 : ℚ) / (20 * x)
def work_rate_second_group : ℚ := (1 : ℚ) / (15 * 24)

-- State the theorem
theorem num_men_in_first_group : ∀ x : ℕ, work_rate_first_group x = work_rate_second_group → x = 18 :=
by
  intro x h
  -- You can leave the proof for the implementation
  sorry

end num_men_in_first_group_l283_283073


namespace find_a_if_even_function_l283_283946

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x * exp x / (exp (a * x) - 1)

theorem find_a_if_even_function :
  (∀ x : ℝ, x ≠ 0 → f a (-x) = f a x) → a = 2 :=
by
  sorry

end find_a_if_even_function_l283_283946


namespace Beth_wins_on_721_l283_283644

def nim_value (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 1
  | 5 => 4
  | 6 => 3
  | 7 => 5
  | _ => 0 -- Assuming only values up to 7 matter for this problem.

def winning_strategy (heights : List ℕ) : Prop :=
  nim_value heights.head ⊕ nim_value heights.nth 1.getD 0 ⊕ nim_value heights.nth 2.getD 0 = 0

theorem Beth_wins_on_721 : winning_strategy [7, 2, 1] :=
  sorry

end Beth_wins_on_721_l283_283644


namespace jan_total_payment_l283_283387

theorem jan_total_payment (roses_per_dozen : ℕ) (dozens : ℕ) (cost_per_rose : ℕ) (discount_rate : ℝ) : ℕ :=
  let total_roses := dozens * roses_per_dozen in
  let total_cost := total_roses * cost_per_rose in
  let discounted_cost := (total_cost : ℝ) * discount_rate in
  total_cost * (discount_rate.to_real : ℝ).to_nat

example : jan_total_payment 12 5 6 0.8 = 288 := by
  sorry

end jan_total_payment_l283_283387


namespace sort_descending_l283_283331

theorem sort_descending (a b c : ℕ) : 
  let (a, b, c) := if a < b then (b, a, c) else (a, b, c) in
  let (a, b, c) := if a < c then (c, b, a) else (a, b, c) in
  let (a, b, c) := if b < c then (a, c, b) else (a, b, c) in
  a >= b ∧ b >= c :=
by
  sorry

end sort_descending_l283_283331


namespace subtraction_divisible_l283_283552

theorem subtraction_divisible (n m d : ℕ) (h1 : n = 13603) (h2 : m = 31) (h3 : d = 13572) : 
  (n - m) % d = 0 := by
  sorry

end subtraction_divisible_l283_283552


namespace factorize_x_cube_minus_4x_l283_283242

theorem factorize_x_cube_minus_4x (x : ℝ) : x^3 - 4 * x = x * (x + 2) * (x - 2) := 
by
  -- Continue the proof from here
  sorry

end factorize_x_cube_minus_4x_l283_283242


namespace arithmetic_sequence_theorem_l283_283295

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- Definition of an arithmetic sequence
def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

-- Given conditions
def arithmetic_sequence_condition (a : ℕ → ℝ) (d : ℝ) : Prop :=
  a 1 + a 3 + a 9 = 20

-- Hypothesis that 'a' is in arithmetic sequence form
axiom arithmetic_sequence_form (a : ℕ → ℝ) (d : ℝ) :
  (∀ n : ℕ, a n = arithmetic_sequence a d n)

-- Theorem to prove
theorem arithmetic_sequence_theorem
  (a : ℕ → ℝ) (d : ℝ)
  [arithmetic_sequence_form a d] :
  (arithmetic_sequence_condition a d) →
  4 * a 5 - a 7 = 20 :=
by
  assume h,
  sorry

end arithmetic_sequence_theorem_l283_283295


namespace sin_double_alpha_l283_283339

theorem sin_double_alpha (α : ℝ) (h1 : α ∈ Ioo 0 (π/2))
  (h2 : cos (π / 4 - α) = 2 * sqrt 2 * cos (2 * α)) : 
  sin (2 * α) = 15 / 16 :=
sorry

end sin_double_alpha_l283_283339


namespace Xiao_Ming_min_steps_l283_283170

-- Problem statement: Prove that the minimum number of steps Xiao Ming needs to move from point A to point B is 5,
-- given his movement pattern and the fact that he can reach eight different positions from point C.

def min_steps_from_A_to_B : ℕ :=
  5

theorem Xiao_Ming_min_steps (A B C : Type) (f : A → B → C) : 
  (min_steps_from_A_to_B = 5) :=
by
  sorry

end Xiao_Ming_min_steps_l283_283170


namespace determine_a_for_even_function_l283_283889

theorem determine_a_for_even_function :
  ∃ a : ℝ, (∀ x : ℝ, x ≠ 0 → (∃ f : ℝ → ℝ, 
  f x = x * exp x / (exp (a * x) - 1) ∧
  (∀ x : ℝ, f (-x) = f x))) → a = 2 :=
by
  sorry

end determine_a_for_even_function_l283_283889


namespace even_function_implies_a_eq_2_l283_283869

def f (x a : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2 (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → a = 2 :=
by
  sorry

end even_function_implies_a_eq_2_l283_283869


namespace find_a_if_f_even_l283_283843

def f (a x : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem find_a_if_f_even (a : ℝ) (h : ∀ x : ℝ, x ≠ 0 → f a (-x) = f a x) : a = 2 :=
by 
  sorry

end find_a_if_f_even_l283_283843


namespace no_other_way_to_construct_five_triangles_l283_283756

open Classical

theorem no_other_way_to_construct_five_triangles {
  sticks : List ℤ 
} (h_stick_lengths : 
  sticks = [2, 3, 4, 20, 30, 40, 200, 300, 400, 2000, 3000, 4000, 20000, 30000, 40000]) 
  (h_five_triplets : 
  ∀ t ∈ (sticks.nthLe 0 15).combinations 3, 
  (t[0] + t[1] > t[2]) ∧ (t[1] + t[2] > t[3]) ∧ (t[0] + t[2] > t[3])) :
  ¬ ∃ other_triplets, 
  (∀ t ∈ other_triplets, t.length = 3) ∧ 
  (∀ t ∈ other_triplets, (t[0] + t[1] > t[2]) ∧ (t[1] + t[2] > t[0]) ∧ (t[0] + t[2] > t[1])) ∧ 
  other_triplets ≠ (sticks.nthLe 0 15).combinations 3 := 
by
  sorry

end no_other_way_to_construct_five_triangles_l283_283756


namespace f_is_even_iff_a_is_2_l283_283813

def f (a : ℝ) (x : ℝ) : ℝ := x * Real.exp x / (Real.exp (a * x) - 1)

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem f_is_even_iff_a_is_2 : (∀ x, f 2 (-x) = f 2 x) ↔ ∀ a, a = 2 := 
sorry

end f_is_even_iff_a_is_2_l283_283813


namespace solve_quadratic_1_solve_quadratic_2_l283_283068

theorem solve_quadratic_1 :
  ∃ x1 x2 : ℝ, (2*x^2 - 4*x - 1 = 0) ↔ (x = (2 + real.sqrt 6) / 2) ∨ (x = (2 - real.sqrt 6) / 2) :=
by sorry

theorem solve_quadratic_2 :
  ∃ x1 x2 : ℝ, (x - 3)^2 = 3*x*(x - 3) ↔ (x = 3) ∨ (x = -3/2) :=
by sorry

end solve_quadratic_1_solve_quadratic_2_l283_283068


namespace even_function_implies_a_eq_2_l283_283900

def f (a x : ℝ) : ℝ := (x * exp x) / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2
  (h : ∀ x : ℝ, f a x = f a (-x)) : a = 2 := 
sorry

end even_function_implies_a_eq_2_l283_283900


namespace power_mod_zero_problem_solution_l283_283169

theorem power_mod_zero (n : ℕ) (h : n ≥ 2) : 2 ^ n % 4 = 0 :=
  sorry

theorem problem_solution : 2 ^ 300 % 4 = 0 :=
  power_mod_zero 300 (by norm_num)

end power_mod_zero_problem_solution_l283_283169


namespace possible_values_l283_283762

theorem possible_values (a : ℝ) (h : a > 1) : ∃ (v : ℝ), (v = 5 ∨ v = 6 ∨ v = 7) ∧ (a + 4 / (a - 1) = v) :=
sorry

end possible_values_l283_283762


namespace even_coeff_sum_l283_283399

theorem even_coeff_sum (n : ℕ) (a : ℕ → ℤ) (s : ℤ) (f : ℤ → ℤ) :
  (∀ x, f(x) = (1 - x + x^2) ^ n) →
  (f(1) = 1) →
  (f(-1) = 3 ^ n) →
  (∀ k, a k = f(k)) →
  2 * s = 1 + 3 ^ n →
  s = (1 + 3 ^ n) / 2 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end even_coeff_sum_l283_283399


namespace find_a_of_even_function_l283_283781

noncomputable def f (a x : ℝ) : ℝ := (x * exp x) / (exp (a * x) - 1)

theorem find_a_of_even_function (hf : ∀ x : ℝ, f a (-x) = f a x) : a = 2 :=
by {
    sorry
}

end find_a_of_even_function_l283_283781


namespace even_function_implies_a_eq_2_l283_283895

def f (a x : ℝ) : ℝ := (x * exp x) / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2
  (h : ∀ x : ℝ, f a x = f a (-x)) : a = 2 := 
sorry

end even_function_implies_a_eq_2_l283_283895


namespace no_alternative_way_to_construct_triangles_l283_283717

theorem no_alternative_way_to_construct_triangles (sticks : list ℕ) (h_len : sticks.length = 15)
  (h_set : ∃ (sets : list (list ℕ)), sets.length = 5 ∧ ∀ set ∈ sets, set.length = 3 
           ∧ (∀ (a b c : ℕ) (ha : a ∈ set) (hb : b ∈ set) (hc : c ∈ set), 
             (a + b > c ∧ a + c > b ∧ b + c > a))):
  ¬∃ (alt_sets : list (list ℕ)), alt_sets ≠ sets ∧ alt_sets.length = 5 
      ∧ ∀ set ∈ alt_sets, set.length = 3 
      ∧ (∀ (a b c : ℕ) (ha : a ∈ set) (hb : b ∈ set) (hc : c ∈ set),
        (a + b > c ∧ a + c > b ∧ b + c > a)) :=
sorry

end no_alternative_way_to_construct_triangles_l283_283717


namespace find_a_for_even_function_l283_283825

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def given_function (a : ℝ) (x : ℝ) : ℝ :=
  x * real.exp x / (real.exp (a * x) - 1)

theorem find_a_for_even_function :
  ∀ (a : ℝ), is_even_function (given_function a) ↔ a = 2 :=
by 
  intro a
  sorry

end find_a_for_even_function_l283_283825


namespace find_a_for_even_function_l283_283815

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def given_function (a : ℝ) (x : ℝ) : ℝ :=
  x * real.exp x / (real.exp (a * x) - 1)

theorem find_a_for_even_function :
  ∀ (a : ℝ), is_even_function (given_function a) ↔ a = 2 :=
by 
  intro a
  sorry

end find_a_for_even_function_l283_283815


namespace even_function_implies_a_eq_2_l283_283868

def f (x a : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2 (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → a = 2 :=
by
  sorry

end even_function_implies_a_eq_2_l283_283868


namespace edwards_hourly_rate_l283_283674

theorem edwards_hourly_rate 
  (R : ℝ) 
  (H_worked : 45) 
  (H_earned : 210) 
  (H_regular_hours : 40) 
  (H_overtime_hours : H_worked - H_regular_hours) 
  (H_total_earning : R * H_regular_hours + 2 * R * H_overtime_hours = H_earned) :
  R = 4.2 := 
sorry

end edwards_hourly_rate_l283_283674


namespace distance_D_from_line_l283_283197

variables (A B C D O : Point)
variables (a b : ℝ)
variables (line : Line)
variables (is_parallelogram : Parallelogram ABCD)
variable (H : Intersects_single_point line B)
variable (H1 : Distance_from_line A line = a)
variable (H2 : Distance_from_line C line = b)
variable (Center_O : Center ABCD = O)
variable (Intersection_O : Intersects AC BD O)

theorem distance_D_from_line :
  Distance_from_line D line = a + b :=
sorry

end distance_D_from_line_l283_283197


namespace find_a_if_f_even_l283_283845

def f (a x : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem find_a_if_f_even (a : ℝ) (h : ∀ x : ℝ, x ≠ 0 → f a (-x) = f a x) : a = 2 :=
by 
  sorry

end find_a_if_f_even_l283_283845


namespace range_of_a_l283_283283

theorem range_of_a (a : ℝ) : (log a (1/2) < 1) ∧ (a^(1/2) < 1) → 0 < a ∧ a < 1/2 :=
by
  sorry

end range_of_a_l283_283283


namespace acute_angle_half_angle_l283_283413

theorem acute_angle_half_angle
  (A X Y Z B : Point)
  (h_convex : convex_pentagon A X Y Z B)
  (h_inscribed : inscribed_in_semicircle A X Y Z B (diameter A B))
  (P Q R S : Point)
  (h_perp_P : is_foot_of_perpendicular P Y (line A X))
  (h_perp_Q : is_foot_of_perpendicular Q Y (line B X))
  (h_perp_R : is_foot_of_perpendicular R Y (line A Z))
  (h_perp_S : is_foot_of_perpendicular S Y (line B Z))
  (O : Point)
  (h_midpoint : midpoint O A B) :
  angle (line P Q) (line R S) = (1/2) * angle X O Z :=
sorry

end acute_angle_half_angle_l283_283413


namespace jan_total_payment_l283_283386

theorem jan_total_payment (roses_per_dozen : ℕ) (dozens : ℕ) (cost_per_rose : ℕ) (discount_rate : ℝ) : ℕ :=
  let total_roses := dozens * roses_per_dozen in
  let total_cost := total_roses * cost_per_rose in
  let discounted_cost := (total_cost : ℝ) * discount_rate in
  total_cost * (discount_rate.to_real : ℝ).to_nat

example : jan_total_payment 12 5 6 0.8 = 288 := by
  sorry

end jan_total_payment_l283_283386


namespace find_a_l283_283996

noncomputable def f (x a : ℝ) : ℝ := (x * (Real.exp x)) / (Real.exp (a * x) - 1)

theorem find_a (a : ℝ) (h : ∀ x, f x a = f (-x) a) : a = 2 :=
begin
  sorry
end

end find_a_l283_283996


namespace find_a_even_function_l283_283978

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def given_function (a : ℝ) (x : ℝ) : ℝ :=
  x * Real.exp x / (Real.exp (a * x) - 1)

theorem find_a_even_function :
  (is_even_function (given_function 2)) → true :=
by
  intro h
  sorry

end find_a_even_function_l283_283978


namespace probability_sum_of_nine_l283_283134

theorem probability_sum_of_nine (a b : set ℕ) 
  (H₁ : a = {2, 3, 4, 5})
  (H₂ : b = {4, 5, 6, 7, 8}) :
  (∃ p : ℚ, p = 3 / 20 ∧ probability (λ x, x ∈ a) (λ y, y ∈ b) (λ x y, x + y = 9) = p) := 
sorry

end probability_sum_of_nine_l283_283134


namespace find_a_of_even_function_l283_283778

noncomputable def f (a x : ℝ) : ℝ := (x * exp x) / (exp (a * x) - 1)

theorem find_a_of_even_function (hf : ∀ x : ℝ, f a (-x) = f a x) : a = 2 :=
by {
    sorry
}

end find_a_of_even_function_l283_283778


namespace mutually_exclusive_event_l283_283233

-- Define the events
def hits_first_shot : Prop := sorry  -- Placeholder for "hits the target on the first shot"
def hits_second_shot : Prop := sorry  -- Placeholder for "hits the target on the second shot"
def misses_first_shot : Prop := ¬ hits_first_shot
def misses_second_shot : Prop := ¬ hits_second_shot

-- Define the main events in the problem
def hitting_at_least_once : Prop := hits_first_shot ∨ hits_second_shot
def missing_both_times : Prop := misses_first_shot ∧ misses_second_shot

-- Statement of the theorem
theorem mutually_exclusive_event :
  missing_both_times ↔ ¬ hitting_at_least_once :=
by sorry

end mutually_exclusive_event_l283_283233


namespace repeating_decimal_equiv_fraction_l283_283682

def repeating_decimal_to_fraction (x : ℚ) : ℚ := 781 / 111

theorem repeating_decimal_equiv_fraction : 7.036036036.... = 781 / 111 :=
  by
  -- lean does not support floating point literals with repeating decimals directly,
  -- need to construct the repeating decimal manually
  let x : ℚ := 7 + 36 / 999 + 36 / 999^2 + 36 / 999^3 + ...
  exactly repeating_decimal_to_fraction (x)

end repeating_decimal_equiv_fraction_l283_283682


namespace shaded_total_area_l283_283365

theorem shaded_total_area:
  ∀ (r₁ r₂ r₃ : ℝ),
  π * r₁ ^ 2 = 100 * π →
  r₂ = r₁ / 2 →
  r₃ = r₂ / 2 →
  (1 / 2) * (π * r₁ ^ 2) + (1 / 2) * (π * r₂ ^ 2) + (1 / 2) * (π * r₃ ^ 2) = 65.625 * π :=
by
  intro r₁ r₂ r₃ h₁ h₂ h₃
  sorry

end shaded_total_area_l283_283365


namespace visitors_not_enjoyed_not_understood_l283_283123

theorem visitors_not_enjoyed_not_understood (V E U : ℕ) (hv_v : V = 520)
  (hu_e : E = U) (he : E = 3 * V / 4) : (V / 4) = 130 :=
by
  rw [hv_v] at he
  sorry

end visitors_not_enjoyed_not_understood_l283_283123


namespace even_function_implies_a_eq_2_l283_283939

theorem even_function_implies_a_eq_2 (a : ℝ) : 
  (∀ x, (f : ℝ → ℝ) x = (λ x : ℝ, x * exp x / (exp (a * x) - 1)) x -> (f (-x) = f x)) -> a = 2 :=
by
  sorry

end even_function_implies_a_eq_2_l283_283939


namespace factorize_expr_l283_283253

theorem factorize_expr (x : ℝ) : x^3 - 4 * x = x * (x + 2) * (x - 2) :=
  sorry

end factorize_expr_l283_283253


namespace NY_Mets_fans_count_l283_283348

noncomputable def NY_Yankees_fans (M: ℝ) : ℝ := (3/2) * M
noncomputable def Boston_Red_Sox_fans (M: ℝ) : ℝ := (5/4) * M
noncomputable def LA_Dodgers_fans (R: ℝ) : ℝ := (2/7) * R

theorem NY_Mets_fans_count :
  ∃ M : ℕ, let Y := NY_Yankees_fans M
           let R := Boston_Red_Sox_fans M
           let D := LA_Dodgers_fans R
           Y + M + R + D = 780 ∧ M = 178 :=
by
  sorry

end NY_Mets_fans_count_l283_283348


namespace find_n_l283_283006

variable {a : ℕ → ℝ}
variable {q : ℝ}
variable (n : ℕ)

-- Condition 1: Given a positive geometric sequence {a_n} with common ratio q > 0
axiom geo_sequence (h_q_pos : q > 0) (a_pos : ∀ n, a n > 0) (geo_seq : ∀ n, a (n + 1) = a n * q)
-- Condition 2: a₁a₂a₃ = 4
axiom cond1 : a 1 * a 2 * a 3 = 4
-- Condition 3: a₄a₅a₆ = 8
axiom cond2 : a 4 * a 5 * a 6 = 8
-- Condition 4: aₙa₍ₙ₊₁₎a₍ₙ₊₂₎ = 128
axiom cond3 : a n * a (n + 1) * a (n + 2) = 128

theorem find_n : n = 6 :=
by
  sorry

end find_n_l283_283006


namespace men_count_eq_eight_l283_283582

theorem men_count_eq_eight (M W B : ℕ) (total_earnings : ℝ) (men_wages : ℝ)
  (H1 : M = W) (H2 : W = B) (H3 : B = 8)
  (H4 : total_earnings = 105) (H5 : men_wages = 7) :
  M = 8 := 
by 
  -- We need to show M = 8 given conditions
  sorry

end men_count_eq_eight_l283_283582


namespace sqrt_2m_minus_n_l283_283300

variable (m n x y : ℝ)

theorem sqrt_2m_minus_n :
  x = 2 ∧ y = 1 ∧ (m * x + n * y = 8) ∧ (n * x - m * y = 1) → (Real.sqrt (2 * m - n) = 2) :=
begin
  intros h,
  rcases h with ⟨hx, hy, h1, h2⟩,
  -- Further proof would go here
  sorry
end

end sqrt_2m_minus_n_l283_283300


namespace eq_a_2_l283_283962

def f (a x : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

-- Definition of even function: ∀ x, f(-x) = f(x)
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem eq_a_2 (a : ℝ) : (even_function (f a) → a = 2) ∧ (a = 2 → even_function (f a)) :=
by
  sorry

end eq_a_2_l283_283962


namespace f_is_even_iff_a_is_2_l283_283803

def f (a : ℝ) (x : ℝ) : ℝ := x * Real.exp x / (Real.exp (a * x) - 1)

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem f_is_even_iff_a_is_2 : (∀ x, f 2 (-x) = f 2 x) ↔ ∀ a, a = 2 := 
sorry

end f_is_even_iff_a_is_2_l283_283803


namespace train_speed_is_87_kmh_l283_283200

noncomputable def lengthOfTrain : ℝ := 180
noncomputable def lengthOfBridge : ℝ := 400
noncomputable def timeToCrossBridge : ℝ := 24
noncomputable def expectedSpeedKmH : ℝ := 87

theorem train_speed_is_87_kmh :
  let totalDistance := lengthOfTrain + lengthOfBridge in
  let speedMs := totalDistance / timeToCrossBridge in
  let speedKmh := speedMs * 3.6 in
  speedKmh = expectedSpeedKmH :=
by
  sorry

end train_speed_is_87_kmh_l283_283200


namespace even_function_implies_a_eq_2_l283_283862

def f (x a : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2 (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → a = 2 :=
by
  sorry

end even_function_implies_a_eq_2_l283_283862


namespace p1_eq_p2_iff_ad_eq_bc_l283_283419

-- Define the variables and expressions
variables {a b c d : ℝ}

-- Root definitions for two polynomials
def x1 := b + c + d
def x2 := -(a + b + c)
def x3 := a - d

def y1 := a + c + d
def y2 := -(a + b + d)
def y3 := b - c

def p1 := x1 * x2 + x2 * x3 + x3 * x1
def p2 := y1 * y2 + y2 * y3 + y3 * y1

-- The theorem statement to prove
theorem p1_eq_p2_iff_ad_eq_bc : p1 = p2 ↔ a * d = b * c :=
by sorry

end p1_eq_p2_iff_ad_eq_bc_l283_283419


namespace even_function_implies_a_eq_2_l283_283872

def f (x a : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2 (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → a = 2 :=
by
  sorry

end even_function_implies_a_eq_2_l283_283872


namespace sum_q_t_12_l283_283035

open Polynomial

noncomputable def T : Set (Fin 12 → Bool) := Set.univ

noncomputable def q_t (t : Fin 12 → Bool) : (Polynomial ℚ) := 
  ∑ (i : Fin 12), if t i then X^i else 0

noncomputable def q := ∑ t in T, q_t t

theorem sum_q_t_12 : 
  (∑ t in T, q_t t) 12 = 2048 := 
  sorry

end sum_q_t_12_l283_283035


namespace solve_m_value_l283_283361

-- Define the polar equation of the curve
def polar_equation_ρ (θ : ℝ) : ℝ := 2 / (1 - Mathlib.cos θ)

-- Define the conversion from polar to Cartesian coordinates
def polar_to_cartesian (ρ θ : ℝ) : (ℝ × ℝ) :=
  let x := ρ * Mathlib.cos θ
  let y := ρ * Mathlib.sin θ
  (x, y)

-- Define the Cartesian equation of the curve
def cartesian_equation (x y : ℝ) : Prop := y^2 = 4*x + 4

-- Define the properties of the line l passing through the point M(m, 0)
def line_l (m t α : ℝ) : ℝ × ℝ :=
  let x := m + t * Mathlib.cos α
  let y := t * Mathlib.sin α
  (x, y)

-- Prove that if the polar equation converts to the Cartesian equation, then m is 1/4
theorem solve_m_value (m : ℝ) :
  (∀ θ, ∃ (x y : ℝ), polar_to_cartesian (polar_equation_ρ θ) θ = (x, y) ∧ cartesian_equation x y) →
  ((∀ t α, let (x, y) := line_l m t α in cartesian_equation x y) → 
  (∀ t1 t2 α, (t1 + t2 = 4 * Mathlib.cos α / Mathlib.sin α^2) → 
  (t1 * t2 = -(4 * m + 4) / Mathlib.sin α^2) →
  ((1 / t1^2 + 1 / t2^2 = 1 / 64) → (m = 1 / 4)))) :=
by
  sorry

end solve_m_value_l283_283361


namespace find_a_if_f_even_l283_283836

def f (a x : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem find_a_if_f_even (a : ℝ) (h : ∀ x : ℝ, x ≠ 0 → f a (-x) = f a x) : a = 2 :=
by 
  sorry

end find_a_if_f_even_l283_283836


namespace no_other_way_to_construct_five_triangles_l283_283750

open Classical

theorem no_other_way_to_construct_five_triangles {
  sticks : List ℤ 
} (h_stick_lengths : 
  sticks = [2, 3, 4, 20, 30, 40, 200, 300, 400, 2000, 3000, 4000, 20000, 30000, 40000]) 
  (h_five_triplets : 
  ∀ t ∈ (sticks.nthLe 0 15).combinations 3, 
  (t[0] + t[1] > t[2]) ∧ (t[1] + t[2] > t[3]) ∧ (t[0] + t[2] > t[3])) :
  ¬ ∃ other_triplets, 
  (∀ t ∈ other_triplets, t.length = 3) ∧ 
  (∀ t ∈ other_triplets, (t[0] + t[1] > t[2]) ∧ (t[1] + t[2] > t[0]) ∧ (t[0] + t[2] > t[1])) ∧ 
  other_triplets ≠ (sticks.nthLe 0 15).combinations 3 := 
by
  sorry

end no_other_way_to_construct_five_triangles_l283_283750


namespace find_integer_for_combination_of_square_l283_283093

theorem find_integer_for_combination_of_square (y : ℝ) :
  ∃ (k : ℝ), (y^2 + 14*y + 60) = (y + 7)^2 + k ∧ k = 11 :=
by
  use 11
  sorry

end find_integer_for_combination_of_square_l283_283093


namespace cost_jam_l283_283676

noncomputable def cost_of_jam (N B J : ℕ) : ℝ :=
  N * J * 5 / 100

theorem cost_jam (N B J : ℕ) (h₁ : N > 1) (h₂ : 4 * N + 20 = 414) :
  cost_of_jam N B J = 2.25 := by
  sorry

end cost_jam_l283_283676


namespace solve_fractional_eq_l283_283069

theorem solve_fractional_eq (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ -3) : (1 / x = 6 / (x + 3)) → (x = 0.6) :=
by
  sorry

end solve_fractional_eq_l283_283069


namespace part1_geometric_sequence_part2_sum_T_n_l283_283291

noncomputable def a (n : ℕ) : ℝ := 3 + 2 * n + 3 * 2 ^ n
noncomputable def b (n : ℕ) : ℝ := 1 / (a n + 3 * 2 ^ n)
noncomputable def S (n : ℕ) : ℝ := 2 * a n + n^2
noncomputable def T (n : ℕ) : ℝ := (2 * S n - n^2) / 2

-- Prove that the sequence {a_n - 2n - 3} is geometric with a common ratio of 2.
theorem part1_geometric_sequence : ∃ q : ℝ, ∀ n : ℕ, n > 0 → a n - 2 * n - 3 = q * (a (n - 1) - 2 * (n - 1) - 3) := sorry

-- Prove that the sum of the first n terms T_n of the sequence {b_n b_{n+1}} is T_n = n / (5 * (2n + 5)).
theorem part2_sum_T_n (n : ℕ) : (∑ k in Finset.range n, b k * b (k + 1)) = n / (5 * (2 * n + 5)) := sorry

end part1_geometric_sequence_part2_sum_T_n_l283_283291


namespace probability_of_event_A_l283_283212

-- Define variables within the given interval
variables (x y : ℝ)
-- Define the intervals for the variables
axiom h_intervals : 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1

-- Define the event A
def event_A : Prop := (x / 2 + y) ≥ 1

-- Define the probability function P
noncomputable def P (A : Prop) : ℝ := 
  if A then 1 - (1 / 4) else 0  -- Placeholder for actual probability calculation function

-- Prove that the probability of the event A is 0.25
theorem probability_of_event_A : P event_A = 0.25 :=
sorry

end probability_of_event_A_l283_283212


namespace no_other_way_five_triangles_l283_283707

def is_triangle {α : Type*} [linear_ordered_field α] (a b c : α) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem no_other_way_five_triangles (sticks : list ℝ)
  (h_len : sticks.length = 15)
  (h_sets : ∃ (sets : list (ℝ × ℝ × ℝ)), 
    sets.length = 5 ∧ ∀ {a b c}, (a, b, c) ∈ sets → is_triangle a b c) :
  ∀ sets, sets.length = 5 ∧ ∀ {a b c}, (a, b, c) ∈ sets → is_triangle a b c → sets = h_sets :=
begin
  sorry
end

end no_other_way_five_triangles_l283_283707


namespace fit_data_with_model_l283_283273

noncomputable def original_model (c k x : ℝ) : ℝ := c * exp (k * x)

theorem fit_data_with_model (c k : ℝ) (h_c_pos : c > 0) :
  (∀ x, log (original_model c k x) = 2 * x - 1) -> k = 2 ∧ c = 1 / exp 1 :=
by
  intro h
  sorry

end fit_data_with_model_l283_283273


namespace smallest_good_number_correct_l283_283290

noncomputable
def is_good_number (k : ℝ) (n : ℕ) : Prop :=
  ∃ (x : ℕ → ℝ), (∀ i, 0 < x i) ∧ (∑ i in finset.range n, x i = k * ∑ i in finset.range n, (1 / x i)) 
  ∧ (∑ i in finset.range n, (x i ^ 2 / (x i ^ 2 + 1)) ≤ k * ∑ i in finset.range n, (1 / (x i ^ 2 + 1)))

noncomputable
def smallest_good_number (n : ℕ) : ℝ :=
  if h : n ≥ 3 then Inf {k : ℝ | k > 1 ∧ is_good_number k n} else 1

theorem smallest_good_number_correct (n : ℕ) (h : n ≥ 3) :
  smallest_good_number n = n - 1 :=
by
  sorry

end smallest_good_number_correct_l283_283290


namespace find_a_even_function_l283_283986

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def given_function (a : ℝ) (x : ℝ) : ℝ :=
  x * Real.exp x / (Real.exp (a * x) - 1)

theorem find_a_even_function :
  (is_even_function (given_function 2)) → true :=
by
  intro h
  sorry

end find_a_even_function_l283_283986


namespace find_a_solutions_l283_283688

theorem find_a_solutions (a : ℝ) :
  (a ∈ set.Ioo 2 3 ∧ ∃ x1 x2 : ℤ, x1 = -1 ∧ x2 = 1) ∨ 
  (∃ n : ℤ, n ≥ 2 ∧ a = 1 + 3 * n / 2 ∧ ∃ x1 x2 : ℤ, x1 = -n ∧ x2 = n) ↔ 
  ∃! (x : ℤ), |2 + |(x : ℝ)| - a| - |a - |(x + 1 : ℝ)| - |(x - 1 : ℝ)|| = 2 + |(x : ℝ)| + |(x + 1 : ℝ)| + |(x - 1 : ℝ)| - 2 * a := 
by 
  sorry

end find_a_solutions_l283_283688


namespace find_a_of_even_function_l283_283766

noncomputable def f (a x : ℝ) : ℝ := (x * exp x) / (exp (a * x) - 1)

theorem find_a_of_even_function (hf : ∀ x : ℝ, f a (-x) = f a x) : a = 2 :=
by {
    sorry
}

end find_a_of_even_function_l283_283766


namespace solution_l283_283077

-- Define the positive integers a and b.
variables (a b : ℤ)
-- Define the condition that they are positive.
variables (ha : a > 0) (hb : b > 0)
-- Define the initial condition of the complex equation.
variable (h : (a - b * complex.I)^2 = 8 - 6 * complex.I)

theorem solution : a - b * complex.I = 3 - complex.I :=
by
  sorry

end solution_l283_283077


namespace find_a_even_function_l283_283987

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def given_function (a : ℝ) (x : ℝ) : ℝ :=
  x * Real.exp x / (Real.exp (a * x) - 1)

theorem find_a_even_function :
  (is_even_function (given_function 2)) → true :=
by
  intro h
  sorry

end find_a_even_function_l283_283987


namespace determine_a_for_even_function_l283_283887

theorem determine_a_for_even_function :
  ∃ a : ℝ, (∀ x : ℝ, x ≠ 0 → (∃ f : ℝ → ℝ, 
  f x = x * exp x / (exp (a * x) - 1) ∧
  (∀ x : ℝ, f (-x) = f x))) → a = 2 :=
by
  sorry

end determine_a_for_even_function_l283_283887


namespace probability_of_all_three_types_l283_283213

noncomputable def probability_dolls_4_boxes : ℝ := 0.216

variable {P_A P_B P_C : ℝ}

-- Given probabilities
axiom prob_A: P_A = 0.6
axiom prob_B: P_B = 0.3
axiom prob_C: P_C = 0.1

-- Main statement to prove
theorem probability_of_all_three_types :  
  (4.choose 1 * P_C * 3.choose 1 * P_B * 2.choose 2 * P_A^2 +
   4.choose 1 * P_C * 3.choose 2 * P_B^2 * P_A +
   4.choose 2 * P_C^2 * 2.choose 1 * P_B * P_A) = probability_dolls_4_boxes :=
by
  sorry

end probability_of_all_three_types_l283_283213


namespace find_a_even_function_l283_283984

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def given_function (a : ℝ) (x : ℝ) : ℝ :=
  x * Real.exp x / (Real.exp (a * x) - 1)

theorem find_a_even_function :
  (is_even_function (given_function 2)) → true :=
by
  intro h
  sorry

end find_a_even_function_l283_283984


namespace no_other_way_to_construct_five_triangles_l283_283740

-- Setting up the problem conditions
def five_triangles_from_fifteen_sticks (sticks : List ℕ) : Prop :=
  sticks.length = 15 ∧
  let split_sticks := sticks.chunk (3) in
  split_sticks.length = 5 ∧
  ∀ triplet ∈ split_sticks, 
    let a := triplet[0]
    let b := triplet[1]
    let c := triplet[2] in
    a + b > c ∧ a + c > b ∧ b + c > a

-- The proof problem statement
theorem no_other_way_to_construct_five_triangles (sticks : List ℕ) :
  five_triangles_from_fifteen_sticks sticks → 
  ∀ (alternative : List (List ℕ)), 
    (alternative.length = 5 ∧ 
    ∀ triplet ∈ alternative, triplet.length = 3 ∧
      let a := triplet[0]
      let b := triplet[1]
      let c := triplet[2] in
      a + b > c ∧ a + c > b ∧ b + c > a) →
    alternative = sticks.chunk (3) :=
by
  sorry

end no_other_way_to_construct_five_triangles_l283_283740


namespace train_crossing_time_l283_283515

noncomputable def length_train1 := 210 -- meters
noncomputable def length_train2 := 260 -- meters
noncomputable def speed_train1 := 60 * (5/18) -- converting km/hr to m/s
noncomputable def speed_train2 := 40 * (5/18) -- converting km/hr to m/s
noncomputable def total_distance := length_train1 + length_train2 -- combined length of both trains

noncomputable def relative_speed := speed_train1 + speed_train2 -- m/s

noncomputable def time_to_cross := total_distance / relative_speed

theorem train_crossing_time :
  time_to_cross ≈ 16.92 := sorry

end train_crossing_time_l283_283515


namespace right_angle_triangle_l283_283466

theorem right_angle_triangle
  (α β γ : ℝ)
  (h_triangle : α + β + γ = π)
  (h_condition : (sin α + sin β) / (cos α + cos β) = sin γ) :
  γ = π / 2 :=
  sorry

end right_angle_triangle_l283_283466


namespace collaptz_sequence_sum_l283_283474

def Collaptz : ℕ → ℕ
| n :=
  if n % 2 = 0 then
    n / 2
  else
    3 * n - 1

def collaptz_seq_does_not_contain_1 (n : ℕ) : Prop :=
  ¬ (∃ m, nth_collaptz n (m+1) = 1)

def nth_collaptz (n : ℕ) : ℕ → ℕ 
| 0 := n
| (m+1) := Collaptz (nth_collaptz m)

theorem collaptz_sequence_sum :
  let ns := [5, 7, 9] in
  ∀ n ∈ ns, collaptz_seq_does_not_contain_1 n ∧ (ns.sum = 21) :=
by sorry

end collaptz_sequence_sum_l283_283474


namespace no_alternate_way_to_form_triangles_l283_283729

noncomputable def canFormTriangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def validSetOfTriples : List (ℕ × ℕ × ℕ) :=
  [(2, 3, 4), (20, 30, 40), (200, 300, 400), (2000, 3000, 4000), (20000, 30000, 40000)]

theorem no_alternate_way_to_form_triangles (stickSet : List (ℕ × ℕ × ℕ)) :
  (∀ (t : ℕ × ℕ × ℕ), t ∈ stickSet → canFormTriangle t.1 t.2 t.3) →
  length stickSet = 5 →
  stickSet = validSetOfTriples ->
  (∀ alternateSet : List (ℕ × ℕ × ℕ),
    ∀ (t : ℕ × ℕ × ℕ), t ∈ alternateSet → canFormTriangle t.1 t.2 t.3 →
    length alternateSet = 5 →
    duplicate_stick_alternateSet = duplicate_stick_stickSet) :=
sorry

end no_alternate_way_to_form_triangles_l283_283729


namespace even_function_implies_a_eq_2_l283_283897

def f (a x : ℝ) : ℝ := (x * exp x) / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2
  (h : ∀ x : ℝ, f a x = f a (-x)) : a = 2 := 
sorry

end even_function_implies_a_eq_2_l283_283897


namespace even_function_implies_a_eq_2_l283_283873

def f (x a : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2 (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → a = 2 :=
by
  sorry

end even_function_implies_a_eq_2_l283_283873


namespace solution_set_f_lt_2_range_a_inequality_l283_283318

noncomputable def f (x : ℝ) : ℝ := |2 * x + 1| - |x - 1|

theorem solution_set_f_lt_2 : { x : ℝ | f x < 2 } = set.Ioo (-4) (2 / 3) :=
by
  sorry

theorem range_a_inequality : (∃ x : ℝ, f x ≤ a - (a^2 / 2)) ↔ -1 ≤ a ∧ a ≤ 3 :=
by
  sorry

end solution_set_f_lt_2_range_a_inequality_l283_283318


namespace find_m_n_l283_283164

theorem find_m_n (m n : ℕ) (h : (1/5 : ℝ)^m * (1/4 : ℝ)^n = 1 / (10 : ℝ)^4) : m = 4 ∧ n = 2 :=
sorry

end find_m_n_l283_283164


namespace probability_of_square_root_less_than_seven_is_13_over_30_l283_283534

-- Definition of two-digit range and condition for square root check
def two_digit_numbers := Finset.range 100 \ Finset.range 10
def sqrt_condition (n : ℕ) : Prop := n < 49

-- The required probability calculation
def probability_square_root_less_than_seven : ℚ :=
  (↑(two_digit_numbers.filter sqrt_condition).card) / (↑two_digit_numbers.card)

-- The theorem stating the required probability
theorem probability_of_square_root_less_than_seven_is_13_over_30 :
  probability_square_root_less_than_seven = 13 / 30 := by
  sorry

end probability_of_square_root_less_than_seven_is_13_over_30_l283_283534


namespace t_range_l283_283322

noncomputable def a (n : ℕ) : ℝ :=
  if n = 1 then 1 else ∑ i in finset.range n, i + 1

noncomputable def b (n : ℕ) : ℝ :=
  ∑ i in finset.range ((2 * n) - n), 1 / a (n + 1 + i)

theorem t_range (t : ℝ) (h : ∀ (n : ℕ), n > 0 → ∀ (m : ℝ), 1 ≤ m ∧ m ≤ 2 → m^2 - m * t + 1/3 > b n) : t < 1 :=
sorry

end t_range_l283_283322


namespace concurrency_and_euler_line_l283_283400

variables {A B C : Type}

/-- The points C₁, A₁, and B₁ are the feet of the altitudes from A, B, and C respectively in the acute-angled triangle ABC.
    The points C₂, A₂, and B₂ are the perpendicular projections of the orthocenter of triangle ABC onto the lines A₁B₁, B₁C₁, and C₁A₁ respectively.
    We need to prove that the lines AA₂, BB₂, and CC₂ are concurrent, and the point of concurrency lies on the Euler line of triangle ABC. -/
theorem concurrency_and_euler_line (ABC : Triangle A B C)
  (C₁ A₁ B₁ : Type) -- Feet of the altitudes
  (C₂ A₂ B₂ : Type) -- Projections of orthocenter
  [foot_of_altitude ABC C₁ A₁ B₁]
  [projection_of_orthocenter ABC C₂ A₂ B₂] :
  ∃ H, is_concurrent (line_through A A₂) (line_through B B₂) (line_through C C₂) H ∧
       lies_on_euler_line H ABC :=
sorry

end concurrency_and_euler_line_l283_283400


namespace chi_squared_test_expectation_correct_distribution_table_correct_l283_283198

-- Given data for the contingency table
def male_good := 52
def male_poor := 8
def female_good := 28
def female_poor := 12
def total := 100

-- Define the $\chi^2$ calculation
def chi_squared_value : ℚ :=
  (total * (male_good * female_poor - male_poor * female_good)^2) / 
  ((male_good + male_poor) * (female_good + female_poor) * (male_good + female_good) * (male_poor + female_poor))

-- The $\chi^2$ value to compare against for 99% confidence
def critical_value_99 : ℚ := 6.635

-- Prove that $\chi^2$ value is less than the critical value for 99% confidence
theorem chi_squared_test :
  chi_squared_value < critical_value_99 :=
by
  -- Sorry to skip the proof as instructed
  sorry

-- Probability data and expectations for successful shots
def prob_male_success : ℚ := 2 / 3
def prob_female_success : ℚ := 1 / 2

-- Probabilities of the number of successful shots
def prob_X_0 : ℚ := (1 - prob_male_success) ^ 2 * (1 - prob_female_success)
def prob_X_1 : ℚ := 2 * prob_male_success * (1 - prob_male_success) * (1 - prob_female_success) +
                    (1 - prob_male_success) ^ 2 * prob_female_success
def prob_X_2 : ℚ := prob_male_success ^ 2 * (1 - prob_female_success) +
                    2 * prob_male_success * (1 - prob_male_success) * prob_female_success
def prob_X_3 : ℚ := prob_male_success ^ 2 * prob_female_success

def expectation_X : ℚ :=
  0 * prob_X_0 + 
  1 * prob_X_1 + 
  2 * prob_X_2 + 
  3 * prob_X_3

-- The expected value of X
def expected_value_X : ℚ := 11 / 6

-- Prove the expected value is as calculated
theorem expectation_correct :
  expectation_X = expected_value_X :=
by
  -- Sorry to skip the proof as instructed
  sorry

-- Define the distribution table based on calculated probabilities
def distribution_table : List (ℚ × ℚ) :=
  [(0, prob_X_0), (1, prob_X_1), (2, prob_X_2), (3, prob_X_3)]

-- The correct distribution table
def correct_distribution_table : List (ℚ × ℚ) :=
  [(0, 1 / 18), (1, 5 / 18), (2, 4 / 9), (3, 2 / 9)]

-- Prove the distribution table is as calculated
theorem distribution_table_correct :
  distribution_table = correct_distribution_table :=
by
  -- Sorry to skip the proof as instructed
  sorry

end chi_squared_test_expectation_correct_distribution_table_correct_l283_283198


namespace even_function_implies_a_eq_2_l283_283918

def f (x : ℝ) (a : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2 (a : ℝ) (h_even : ∀ x, f (-x) a = f x a) : a = 2 :=
by
  intro x
  sorry

end even_function_implies_a_eq_2_l283_283918


namespace f_is_even_iff_a_is_2_l283_283800

def f (a : ℝ) (x : ℝ) : ℝ := x * Real.exp x / (Real.exp (a * x) - 1)

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem f_is_even_iff_a_is_2 : (∀ x, f 2 (-x) = f 2 x) ↔ ∀ a, a = 2 := 
sorry

end f_is_even_iff_a_is_2_l283_283800


namespace even_function_implies_a_is_2_l283_283790

noncomputable def f (a x : ℝ) : ℝ := (x * Real.exp x) / (Real.exp (a * x) - 1)

theorem even_function_implies_a_is_2 (a : ℝ) 
  (h : ∀ x : ℝ, f a x = f a (-x)) : a = 2 := 
by
  sorry

end even_function_implies_a_is_2_l283_283790


namespace units_digit_expression_mod_10_l283_283272

theorem units_digit_expression_mod_10 : ((2 ^ 2023) * (5 ^ 2024) * (11 ^ 2025)) % 10 = 0 := 
by 
  -- Proof steps would go here
  sorry

end units_digit_expression_mod_10_l283_283272


namespace range_of_A_l283_283416

def f (A x : ℝ) : ℝ := A * (x^2 - 2*x) * Real.exp x - Real.exp x + 1

theorem range_of_A (A : ℝ) : (∀ x : ℝ, x ≤ 0 → f A x ≥ 0) → A ≥ -1/2 :=
begin
  intro h,
  sorry
end

end range_of_A_l283_283416


namespace even_function_implies_a_eq_2_l283_283857

theorem even_function_implies_a_eq_2 (a : ℝ) 
  (h : ∀ x : ℝ, f x = f (-x)) 
  (f : ℝ → ℝ := λ x, (x * exp x) / (exp (a * x) - 1)) : a = 2 :=
sorry

end even_function_implies_a_eq_2_l283_283857


namespace mary_spent_total_amount_l283_283049

def shirt_cost : ℝ := 13.04
def jacket_cost : ℝ := 12.27
def total_cost : ℝ := 25.31

theorem mary_spent_total_amount :
  shirt_cost + jacket_cost = total_cost := sorry

end mary_spent_total_amount_l283_283049


namespace eq_a_2_l283_283959

def f (a x : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

-- Definition of even function: ∀ x, f(-x) = f(x)
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem eq_a_2 (a : ℝ) : (even_function (f a) → a = 2) ∧ (a = 2 → even_function (f a)) :=
by
  sorry

end eq_a_2_l283_283959


namespace no_other_way_to_construct_five_triangles_l283_283739

-- Setting up the problem conditions
def five_triangles_from_fifteen_sticks (sticks : List ℕ) : Prop :=
  sticks.length = 15 ∧
  let split_sticks := sticks.chunk (3) in
  split_sticks.length = 5 ∧
  ∀ triplet ∈ split_sticks, 
    let a := triplet[0]
    let b := triplet[1]
    let c := triplet[2] in
    a + b > c ∧ a + c > b ∧ b + c > a

-- The proof problem statement
theorem no_other_way_to_construct_five_triangles (sticks : List ℕ) :
  five_triangles_from_fifteen_sticks sticks → 
  ∀ (alternative : List (List ℕ)), 
    (alternative.length = 5 ∧ 
    ∀ triplet ∈ alternative, triplet.length = 3 ∧
      let a := triplet[0]
      let b := triplet[1]
      let c := triplet[2] in
      a + b > c ∧ a + c > b ∧ b + c > a) →
    alternative = sticks.chunk (3) :=
by
  sorry

end no_other_way_to_construct_five_triangles_l283_283739


namespace factor_expression_l283_283240

-- Define the expression to simplify
def expr : ℝ → ℝ := λ x, 81 - 16 * (x - 1) ^ 2

-- Define the factored expression
def factored_expr : ℝ → ℝ := λ x, (13 - 4 * x) * (5 + 4 * x)

-- State the theorem
theorem factor_expression (x : ℝ) : expr x = factored_expr x := 
by {
  -- leaving the proof as an exercise
  sorry
}

-- The statement above can be compiled purely based on conditions and question

end factor_expression_l283_283240


namespace even_function_implies_a_eq_2_l283_283849

theorem even_function_implies_a_eq_2 (a : ℝ) 
  (h : ∀ x : ℝ, f x = f (-x)) 
  (f : ℝ → ℝ := λ x, (x * exp x) / (exp (a * x) - 1)) : a = 2 :=
sorry

end even_function_implies_a_eq_2_l283_283849


namespace no_other_way_five_triangles_l283_283709

def is_triangle {α : Type*} [linear_ordered_field α] (a b c : α) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem no_other_way_five_triangles (sticks : list ℝ)
  (h_len : sticks.length = 15)
  (h_sets : ∃ (sets : list (ℝ × ℝ × ℝ)), 
    sets.length = 5 ∧ ∀ {a b c}, (a, b, c) ∈ sets → is_triangle a b c) :
  ∀ sets, sets.length = 5 ∧ ∀ {a b c}, (a, b, c) ∈ sets → is_triangle a b c → sets = h_sets :=
begin
  sorry
end

end no_other_way_five_triangles_l283_283709


namespace find_a_if_even_function_l283_283947

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x * exp x / (exp (a * x) - 1)

theorem find_a_if_even_function :
  (∀ x : ℝ, x ≠ 0 → f a (-x) = f a x) → a = 2 :=
by
  sorry

end find_a_if_even_function_l283_283947


namespace no_alternative_way_to_construct_triangles_l283_283715

theorem no_alternative_way_to_construct_triangles (sticks : list ℕ) (h_len : sticks.length = 15)
  (h_set : ∃ (sets : list (list ℕ)), sets.length = 5 ∧ ∀ set ∈ sets, set.length = 3 
           ∧ (∀ (a b c : ℕ) (ha : a ∈ set) (hb : b ∈ set) (hc : c ∈ set), 
             (a + b > c ∧ a + c > b ∧ b + c > a))):
  ¬∃ (alt_sets : list (list ℕ)), alt_sets ≠ sets ∧ alt_sets.length = 5 
      ∧ ∀ set ∈ alt_sets, set.length = 3 
      ∧ (∀ (a b c : ℕ) (ha : a ∈ set) (hb : b ∈ set) (hc : c ∈ set),
        (a + b > c ∧ a + c > b ∧ b + c > a)) :=
sorry

end no_alternative_way_to_construct_triangles_l283_283715


namespace find_a_of_even_function_l283_283776

noncomputable def f (a x : ℝ) : ℝ := (x * exp x) / (exp (a * x) - 1)

theorem find_a_of_even_function (hf : ∀ x : ℝ, f a (-x) = f a x) : a = 2 :=
by {
    sorry
}

end find_a_of_even_function_l283_283776


namespace fractional_part_water_after_replacements_l283_283576

theorem fractional_part_water_after_replacements : 
  ∀ (initial_volume water_replaced antifreeze_added repetitions : ℕ), 
    initial_volume = 20 → 
    water_replaced = 5 → 
    antifreeze_added = 5 → 
    repetitions = 4 → 
    (let remaining_fraction := (1 - (water_replaced.toRat / initial_volume.toRat)) in
    remaining_fraction ^ repetitions.toRat) = (81 / 256) := 
by
  intros initial_volume water_replaced antifreeze_added repetitions 
  assume h_initial_volume h_water_replaced h_antifreeze_added h_repetitions
  let remaining_fraction := (1 - (water_replaced.toRat / initial_volume.toRat))
  have frac_4_times := remaining_fraction ^ repetitions.toRat
  show frac_4_times = (81 / 256)
  sorry

end fractional_part_water_after_replacements_l283_283576


namespace regular_polygon_sides_l283_283612

theorem regular_polygon_sides (theta : ℝ) (h : theta = 18) : 
  ∃ n : ℕ, 360 / theta = n ∧ n = 20 :=
by
  use 20
  constructor
  apply eq_of_div_eq_mul
  rw h
  norm_num
  sorry

end regular_polygon_sides_l283_283612


namespace first_car_distance_l283_283133

-- Definitions for conditions
variable (x : ℝ) -- distance the first car ran before taking the right turn
def distance_apart_initial := 150 -- initial distance between the cars
def distance_first_car_main_road := 2 * x -- total distance first car ran on the main road
def distance_second_car := 62 -- distance the second car ran due to breakdown
def distance_between_cars := 38 -- distance between the cars after running 

-- Proof (statement only, no solution steps)
theorem first_car_distance (hx : distance_apart_initial = distance_first_car_main_road + distance_second_car + distance_between_cars) : 
  x = 25 :=
by
  unfold distance_apart_initial distance_first_car_main_road distance_second_car distance_between_cars at hx
  -- Implementation placeholder
  sorry

end first_car_distance_l283_283133


namespace regular_polygon_sides_l283_283611

theorem regular_polygon_sides (theta : ℝ) (h : theta = 18) : 
  ∃ n : ℕ, 360 / theta = n ∧ n = 20 :=
by
  use 20
  constructor
  apply eq_of_div_eq_mul
  rw h
  norm_num
  sorry

end regular_polygon_sides_l283_283611


namespace Sally_cards_l283_283460

theorem Sally_cards (initial_cards : ℕ) (cards_from_dan : ℕ) (cards_bought : ℕ) :
  initial_cards = 27 →
  cards_from_dan = 41 →
  cards_bought = 20 →
  initial_cards + cards_from_dan + cards_bought = 88 :=
by {
  intros,
  sorry
}

end Sally_cards_l283_283460


namespace probability_sqrt_less_than_seven_l283_283543

-- Definitions and conditions from part a)
def is_two_digit_number (n : ℕ) : Prop := (10 ≤ n) ∧ (n ≤ 99)
def sqrt_less_than_seven (n : ℕ) : Prop := real.sqrt n < 7

-- Lean 4 statement for the actual proof problem
theorem probability_sqrt_less_than_seven : 
  (∃ n, is_two_digit_number n ∧ sqrt_less_than_seven n) → ∑ i in (finset.range 100).filter is_two_digit_number, if sqrt_less_than_seven i then 1 else 0 = 39 :=
sorry

end probability_sqrt_less_than_seven_l283_283543


namespace fill_cistern_time_l283_283135

open BigOperators

theorem fill_cistern_time : 
  let rate_a := 1 / 60.0
  let rate_b := 1 / 120.0
  let rate_c := -1 / 120.0
  let combined_rate := rate_a + rate_b + rate_c
  combined_rate = rate_a → 
  let fill_time := 60.0
  fill_time = 60.0 :=
by 
  sorry

end fill_cistern_time_l283_283135


namespace exterior_angle_regular_polygon_l283_283596

theorem exterior_angle_regular_polygon (exterior_angle : ℝ) (sides : ℕ) (h : exterior_angle = 18) : sides = 20 :=
by
  -- Use the condition that the sum of exterior angles of any polygon is 360 degrees.
  have sum_exterior_angles : ℕ := 360
  -- Set up the equation 18 * sides = 360
  have equation : 18 * sides = sum_exterior_angles := sorry
  -- Therefore, sides = 20
  sorry

end exterior_angle_regular_polygon_l283_283596


namespace TimTotalRunHoursPerWeek_l283_283508

def TimUsedToRunTimesPerWeek : ℕ := 3
def TimAddedExtraDaysPerWeek : ℕ := 2
def MorningRunHours : ℕ := 1
def EveningRunHours : ℕ := 1

theorem TimTotalRunHoursPerWeek :
  (TimUsedToRunTimesPerWeek + TimAddedExtraDaysPerWeek) * (MorningRunHours + EveningRunHours) = 10 :=
by
  sorry

end TimTotalRunHoursPerWeek_l283_283508


namespace sin_angle_ACB_in_terms_of_u_v_l283_283359

theorem sin_angle_ACB_in_terms_of_u_v
  (A B C D : Type) [EuclideanSpace ℝ A] [EuclideanSpace ℝ B] [EuclideanSpace ℝ C] [EuclideanSpace ℝ D]
  (angle_ADB angle_ADC angle_BDC : ℝ)
  (u v : ℝ)
  (h_angle_ADB : angle_ADB = π / 2)
  (h_angle_ADC : angle_ADC = π / 2)
  (h_angle_BDC : angle_BDC = π / 2)
  (h_cos_CAD : u = cos angle (A, C, D))
  (h_cos_CBD : v = cos angle (C, B, D)) :
  sin (angle (A, C, B)) = sqrt (1 - u^2 * v^2) := 
sorry

end sin_angle_ACB_in_terms_of_u_v_l283_283359


namespace mike_books_l283_283429

variable (t b l : Nat)

theorem mike_books (h₁ : t = 91) (h₂ : b = 56) (h₃ : t - b = l) : l = 35 := 
by
  rwa [h₁, h₂]
  simp only [Nat.sub_eq_iff_eq_add.mpr (Nat.le.intro rfl)]
  exact @rfl 35

end mike_books_l283_283429


namespace no_possible_k_l283_283214
open Classical

noncomputable def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem no_possible_k : 
  ∀ (k : ℕ), 
    (∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ (p + q = 74) ∧ (x^2 - 74*x + k = 0)) -> False :=
by sorry

end no_possible_k_l283_283214


namespace find_a_for_even_function_l283_283823

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def given_function (a : ℝ) (x : ℝ) : ℝ :=
  x * real.exp x / (real.exp (a * x) - 1)

theorem find_a_for_even_function :
  ∀ (a : ℝ), is_even_function (given_function a) ↔ a = 2 :=
by 
  intro a
  sorry

end find_a_for_even_function_l283_283823


namespace find_a_of_even_function_l283_283768

noncomputable def f (a x : ℝ) : ℝ := (x * exp x) / (exp (a * x) - 1)

theorem find_a_of_even_function (hf : ∀ x : ℝ, f a (-x) = f a x) : a = 2 :=
by {
    sorry
}

end find_a_of_even_function_l283_283768


namespace f_is_even_iff_a_is_2_l283_283799

def f (a : ℝ) (x : ℝ) : ℝ := x * Real.exp x / (Real.exp (a * x) - 1)

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem f_is_even_iff_a_is_2 : (∀ x, f 2 (-x) = f 2 x) ↔ ∀ a, a = 2 := 
sorry

end f_is_even_iff_a_is_2_l283_283799


namespace least_amount_needed_to_buy_rope_l283_283081

theorem least_amount_needed_to_buy_rope : 
  (∀ (Tanesha_needs : ℕ) (rope_length : ℕ) (rope_cost : ℝ), 
    let pieces := 10 in
    let piece_length := 6 in
    let total_length := pieces * piece_length in
    let feet_needed := total_length / 12 in
    let cost_per_foot := 1.25 in
    (total_length = 60 → feet_needed = 5 → rope_cost == cost_per_foot) → 
    rope_cost * feet_needed = 6.25) :=
begin
  sorry
end

end least_amount_needed_to_buy_rope_l283_283081


namespace owen_final_turtle_count_l283_283447

theorem owen_final_turtle_count (owen_initial johanna_initial : ℕ)
  (h1: owen_initial = 21)
  (h2: johanna_initial = owen_initial - 5) :
  let owen_after_1_month := 2 * owen_initial,
      johanna_after_1_month := johanna_initial / 2,
      owen_final := owen_after_1_month + johanna_after_1_month
  in
  owen_final = 50 :=
by
  -- Solution steps go here.
  sorry

end owen_final_turtle_count_l283_283447


namespace log_expression_problem_l283_283760

noncomputable def log_expressions (p q : ℝ) : Prop :=
  (log 10 7 = q / (1/2 + 1/p + q))

theorem log_expression_problem 
  (p q : ℝ)
  (h₁ : log 9 4 = p)
  (h₂ : log 4 7 = q) :
  log_expressions p q :=
by
  sorry

end log_expression_problem_l283_283760


namespace simplify_trig_expression_l283_283469

theorem simplify_trig_expression (x y : ℝ) : 
    cos (x + y) * sin y - sin (x + y) * cos y = - sin (x + y) :=
by
  sorry

end simplify_trig_expression_l283_283469


namespace even_function_implies_a_is_2_l283_283796

noncomputable def f (a x : ℝ) : ℝ := (x * Real.exp x) / (Real.exp (a * x) - 1)

theorem even_function_implies_a_is_2 (a : ℝ) 
  (h : ∀ x : ℝ, f a x = f a (-x)) : a = 2 := 
by
  sorry

end even_function_implies_a_is_2_l283_283796


namespace f_is_even_iff_a_is_2_l283_283798

def f (a : ℝ) (x : ℝ) : ℝ := x * Real.exp x / (Real.exp (a * x) - 1)

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem f_is_even_iff_a_is_2 : (∀ x, f 2 (-x) = f 2 x) ↔ ∀ a, a = 2 := 
sorry

end f_is_even_iff_a_is_2_l283_283798


namespace even_function_implies_a_eq_2_l283_283905

def f (a x : ℝ) : ℝ := (x * exp x) / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2
  (h : ∀ x : ℝ, f a x = f a (-x)) : a = 2 := 
sorry

end even_function_implies_a_eq_2_l283_283905


namespace contrapositive_lemma_l283_283088

theorem contrapositive_lemma (a : ℝ) (h : a^2 ≤ 9) : a < 4 := 
sorry

end contrapositive_lemma_l283_283088


namespace number_in_B_hand_is_three_l283_283449

theorem number_in_B_hand_is_three :
  ∃ a b : ℕ,
     (a ∈ ({1, 2, 3, 4, 5} : set ℕ) ∧ b ∈ ({1, 2, 3, 4, 5} : set ℕ)) ∧
     a ≠ b ∧
     -- Condition 1: Player A cannot determine the larger number
     (∀ x ∈ {1, 5}, a ≠ x) ∧
     -- Condition 2: Player B hears A's statement and also cannot determine the larger number
     (∀ x ∈ {1, 2, 4, 5}, b ≠ x) ∧
     -- Conclusion: The number in player B's hand is 3
     b = 3 :=
begin
  sorry
end

end number_in_B_hand_is_three_l283_283449


namespace factorize_cubic_l283_283251

theorem factorize_cubic : ∀ x : ℝ, x^3 - 4 * x = x * (x + 2) * (x - 2) :=
by
  sorry

end factorize_cubic_l283_283251


namespace regular_polygon_sides_l283_283626

theorem regular_polygon_sides (angle : ℝ) (h_angle : angle = 18) : ∃ n : ℕ, n = 20 :=
by
  have sum_exterior_angles : ℝ := 360
  have num_sides := sum_exterior_angles / angle
  have h_num_sides : num_sides = 20 := by
    calc
      num_sides = 360 / 18 : by rw [h_angle]
      ... = 20 : by norm_num
  use 20
  rw ← h_num_sides
  sorry

end regular_polygon_sides_l283_283626


namespace find_a_l283_283999

noncomputable def f (x a : ℝ) : ℝ := (x * (Real.exp x)) / (Real.exp (a * x) - 1)

theorem find_a (a : ℝ) (h : ∀ x, f x a = f (-x) a) : a = 2 :=
begin
  sorry
end

end find_a_l283_283999


namespace factorize_expr_l283_283256

theorem factorize_expr (x : ℝ) : x^3 - 4 * x = x * (x + 2) * (x - 2) :=
  sorry

end factorize_expr_l283_283256


namespace number_of_equilateral_triangles_in_lattice_l283_283663

-- Definitions
def isOctagonalLattice (V : Type) (E : V → V → Prop) : Prop :=
  ∃ octagon_points : list V,
  ∃ extended_points : list V,
  -- Condition 1: Vertices of the octagon are evenly spaced and each is one unit from its nearest neighbor.
  (ζ octagon_points.length = 8 ∧
   (∀ i, E (octagon_points.nthLe i sorry) (octagon_points.nthLe ((i + 1) % 8) sorry) = 1)) ∧
  -- Condition 2: Each side of the octagon is extended one unit outward.
  (ζ extended_points.length = 8 ∧
   (∀ i, E (octagon_points.nthLe i sorry) (extended_points.nthLe i sorry) = 1))

-- Problem statement
theorem number_of_equilateral_triangles_in_lattice (V : Type) (E : V → V → Prop) 
  (h : isOctagonalLattice V E) : 
  number_of_equilateral_triangles V E = 24 :=
sorry

end number_of_equilateral_triangles_in_lattice_l283_283663


namespace no_alternative_way_to_construct_triangles_l283_283719

-- Define the triangle inequality condition
def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the set of stick lengths for each required triangle
def valid_triangle_sets (stick_set : Set (Set ℕ)) : Prop :=
  stick_set = { {2, 3, 4}, {20, 30, 40}, {200, 300, 400}, {2000, 3000, 4000}, {20000, 30000, 40000} }

-- Define the problem statement
theorem no_alternative_way_to_construct_triangles (s : Set ℕ) (h : s.card = 15) :
  (∀ t ∈ (s.powerset.filter (λ t, t.card = 3)), triangle_inequality t.some (finset_some_spec t) ∧ (∃ sets : Set (Set ℕ), sets.card = 5 ∧ ∀ u ∈ sets, u.card = 3 ∧ triangle_inequality u.some (finset_some_spec u)))
  → valid_triangle_sets (s.powerset.filter (λ t, t.card = 3)) := 
sorry

end no_alternative_way_to_construct_triangles_l283_283719


namespace unique_triangle_assembly_l283_283746

def triangle_inequality (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

def is_valid_triangle_set (sticks : List (ℕ × ℕ × ℕ)) : Prop :=
  sticks.all (λ stick_lengths, triangle_inequality stick_lengths.1 stick_lengths.2 stick_lengths.3)

theorem unique_triangle_assembly (sticks : List ℕ) (h_len : sticks.length = 15)
  (h_sticks : sticks = [2, 3, 4, 20, 30, 40, 200, 300, 400, 2000, 3000, 4000, 20000, 30000, 40000]) :
  ¬ ∃ (other_config : List (ℕ × ℕ × ℕ)),
    is_valid_triangle_set other_config ∧
    other_config ≠ [(2, 3, 4), (20, 30, 40), (200, 300, 400), (2000, 3000, 4000), (20000, 30000, 40000)] :=
begin
  sorry
end

end unique_triangle_assembly_l283_283746


namespace geometric_sequence_fourth_term_l283_283095

theorem geometric_sequence_fourth_term (a : ℝ) (r : ℝ) (h : a = 512) (h1 : a * r^5 = 125) :
  a * r^3 = 1536 :=
by
  sorry

end geometric_sequence_fourth_term_l283_283095


namespace even_function_implies_a_eq_2_l283_283871

def f (x a : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2 (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → a = 2 :=
by
  sorry

end even_function_implies_a_eq_2_l283_283871


namespace even_function_implies_a_eq_2_l283_283877

def f (x a : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2 (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → a = 2 :=
by
  sorry

end even_function_implies_a_eq_2_l283_283877


namespace first_reduction_percentage_l283_283105

theorem first_reduction_percentage (P : ℝ) (x : ℝ) (h : 0.30 * (1 - x / 100) * P = 0.225 * P) : x = 25 :=
by
  sorry

end first_reduction_percentage_l283_283105


namespace winning_candidate_received_percentage_l283_283209

theorem winning_candidate_received_percentage :
  ∀ (total_members votes_casting : ℕ) (percentage_total_members votes_received: ℝ),
    total_members = 1600 →
    votes_casting = 525 →
    percentage_total_members = 19.6875 →
    votes_received = (percentage_total_members / 100) * total_members →
    (votes_received / votes_casting) * 100 = 60 :=
by
  intros total_members votes_casting percentage_total_members votes_received
  intro h_total_members h_votes_casting h_percentage_total_members h_votes_received
  rw [h_total_members, h_votes_casting, h_percentage_total_members, h_votes_received]
  sorry

end winning_candidate_received_percentage_l283_283209


namespace b_joined_after_x_months_l283_283634

-- Establish the given conditions as hypotheses
theorem b_joined_after_x_months
  (a_start_capital : ℝ)
  (b_start_capital : ℝ)
  (profit_ratio : ℝ)
  (months_in_year : ℝ)
  (a_capital_time : ℝ)
  (b_capital_time : ℝ)
  (a_profit_ratio : ℝ)
  (b_profit_ratio : ℝ)
  (x : ℝ)
  (h1 : a_start_capital = 3500)
  (h2 : b_start_capital = 9000)
  (h3 : profit_ratio = 2 / 3)
  (h4 : months_in_year = 12)
  (h5 : a_capital_time = 12)
  (h6 : b_capital_time = 12 - x)
  (h7 : a_profit_ratio = 2)
  (h8 : b_profit_ratio = 3)
  (h_ratio : (a_start_capital * a_capital_time) / (b_start_capital * b_capital_time) = profit_ratio) :
  x = 5 :=
by
  sorry

end b_joined_after_x_months_l283_283634


namespace distance_from_pointA_to_line_l283_283368

-- Define the polar coordinates of point A
def pointA : ℝ × ℝ := (1, Real.pi)

-- Define the line equation in polar coordinates
def line_eq_in_polar (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 2

-- Define a function to convert from polar to cartesian coordinates
def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ := (ρ * Real.cos θ, ρ * Real.sin θ)

-- Point A in Cartesian coordinates
def pointA_cartesian : ℝ × ℝ := polar_to_cartesian 1 Real.pi

-- Define the line equation in Cartesian coordinates (x = 2)
def line_eq_in_cartesian (p : ℝ × ℝ) : Prop := p.1 = 2

-- Function to calculate the distance from a point to a vertical line in Cartesian coordinates
def distance_to_vertical_line (p : ℝ × ℝ) (l : ℝ) : ℝ := abs (l - p.1)

-- Theorem statement
theorem distance_from_pointA_to_line :
  distance_to_vertical_line pointA_cartesian 2 = 3 := by
  sorry

end distance_from_pointA_to_line_l283_283368


namespace angle_AMN_60deg_l283_283648

open EuclideanGeometry

-- Definition of the problem
variables {O1 O2 A B C D E F M N : Point}
variables {ω1 ω2 : Circle}

-- Conditions
axiom intersect_circles : ω1 ∩ ω2 = {A, B}
axiom opposite_sides : ¬ same_side O1 O2 A B
axiom diameter_AC : diameter ω1 A C
axiom extend_BC_int_ω2 : extends_to B C D ω2
axiom arc_E_on_ω2 : on_arc E (arc_BC ω2)
axiom extend_EB_int_ω2 : extends_to E B F ω2
axiom midpoint_CD : midpoint M C D
axiom midpoint_EF : midpoint N E F
axiom AC2CE : AC = 2 * CE

-- Statement to prove
theorem angle_AMN_60deg : angle AMN = 60 :=
sorry

end angle_AMN_60deg_l283_283648


namespace caps_percentage_l283_283206

open Real

-- Define the conditions as given in part (a)
def total_caps : ℝ := 575
def red_caps : ℝ := 150
def green_caps : ℝ := 120
def blue_caps : ℝ := 175
def yellow_caps : ℝ := total_caps - (red_caps + green_caps + blue_caps)

-- Define the problem asking for the percentages of each color and proving the answer
theorem caps_percentage :
  (red_caps / total_caps) * 100 = 26.09 ∧
  (green_caps / total_caps) * 100 = 20.87 ∧
  (blue_caps / total_caps) * 100 = 30.43 ∧
  (yellow_caps / total_caps) * 100 = 22.61 :=
by
  -- proof steps would go here
  sorry

end caps_percentage_l283_283206


namespace f_is_even_iff_a_is_2_l283_283812

def f (a : ℝ) (x : ℝ) : ℝ := x * Real.exp x / (Real.exp (a * x) - 1)

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem f_is_even_iff_a_is_2 : (∀ x, f 2 (-x) = f 2 x) ↔ ∀ a, a = 2 := 
sorry

end f_is_even_iff_a_is_2_l283_283812


namespace percentage_slump_in_business_l283_283100

theorem percentage_slump_in_business (X : ℝ) (Y : ℝ) :
  0.05 * Y = 0.04 * X → Y = 0.8 * X → 20 := 
by
  sorry

end percentage_slump_in_business_l283_283100


namespace no_other_way_five_triangles_l283_283708

def is_triangle {α : Type*} [linear_ordered_field α] (a b c : α) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem no_other_way_five_triangles (sticks : list ℝ)
  (h_len : sticks.length = 15)
  (h_sets : ∃ (sets : list (ℝ × ℝ × ℝ)), 
    sets.length = 5 ∧ ∀ {a b c}, (a, b, c) ∈ sets → is_triangle a b c) :
  ∀ sets, sets.length = 5 ∧ ∀ {a b c}, (a, b, c) ∈ sets → is_triangle a b c → sets = h_sets :=
begin
  sorry
end

end no_other_way_five_triangles_l283_283708


namespace sum_of_squares_of_geometric_sequence_l283_283115

noncomputable def geometric_sequence (b₁ r : ℝ) (n : ℕ) : ℝ :=
  b₁ * r^(n-1)

theorem sum_of_squares_of_geometric_sequence (b₁ r : ℝ) :
  (∑ n in finset.range 2013, (geometric_sequence b₁ r (n + 6))) = 6 →
  (∑ n in finset.range 2013, (-1)^n * (geometric_sequence b₁ r (n + 6))) = 3 →
  (∑ n in finset.range 2013, (geometric_sequence b₁ r (n + 6))^2) = 18 :=
by
  intros h1 h2
  sorry

end sum_of_squares_of_geometric_sequence_l283_283115


namespace problem_statement_l283_283405

theorem problem_statement :
  let n := (minimal (λ x: ℕ, x > 0 ∧ (105 ∣ x) ∧ (num_divisors x = 120))) in
  n / 105 = 5952640 :=
by
  sorry

end problem_statement_l283_283405


namespace solve_exponential_eq_l283_283687

theorem solve_exponential_eq (x : ℝ) : 
  2^x + 3^x + 4^x + 5^x = 7^x ↔ x = 2.5 :=
by {
  -- Proof goes here
  sorry
}

end solve_exponential_eq_l283_283687


namespace intersection_and_vector_sum_l283_283485

-- Define the conditions and the statement
theorem intersection_and_vector_sum (k : ℝ) :
  (∀ x : ℝ, -π/2 < x ∧ x < π/2 → y = tan x ∧ y = k * x) →
  (let A := (-π/2 : ℝ, 0 : ℝ);
       O := (0 : ℝ, 0 : ℝ);
       AM := (-(π/2) : ℝ, tan (-(π/2)) : ℝ);
       AN := (π/2 : ℝ, tan (π/2) : ℝ) in
     (AM + AN) • A = π^2 / 2) :=
by sorry

end intersection_and_vector_sum_l283_283485


namespace a_plus_b_l283_283041

-- Definitions and conditions
def f (x : ℝ) (a b : ℝ) := a * x + b
def g (x : ℝ) := 3 * x - 7

theorem a_plus_b (a b : ℝ) (h : ∀ x : ℝ, g (f x a b) = 4 * x + 5) : a + b = 16 / 3 :=
by
  sorry

end a_plus_b_l283_283041


namespace num_remaining_integers_l283_283112

def T := {n : ℕ | 1 ≤ n ∧ n ≤ 100}

def is_multiple (a b : ℕ) : Prop := b ∣ a

def multiples (m : ℕ) : finset ℕ := (finset.range 101).filter (λ n, is_multiple m n)

def total_multiples := 
  (multiples 2).card + (multiples 3).card + (multiples 5).card 
  - (multiples 6).card - (multiples 10).card - (multiples 15).card 
  + (multiples 30).card

def remaining_integers := T.card - total_multiples

theorem num_remaining_integers : remaining_integers = 26 := 
by
  ref ⟨T, remaining_integers⟩ sorry

end num_remaining_integers_l283_283112


namespace Robins_hair_length_l283_283459

theorem Robins_hair_length :
  ∀ (initial_growth_cut : ℕ × ℕ × ℕ),
  initial_growth_cut = (14, 8, 20) → initial_growth_cut.1 + initial_growth_cut.2 - initial_growth_cut.3 = 2 :=
by
  intro initial_growth_cut
  intro h
  rw [h]
  calc 14 + 8 - 20 = 22 - 20 : by rw [Nat.add_sub_assoc, add_comm] -- first perform the addition
               ... = 2 : by rw [Nat.sub_self, zero_add]
  sorry

end Robins_hair_length_l283_283459


namespace find_a_for_even_function_l283_283816

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def given_function (a : ℝ) (x : ℝ) : ℝ :=
  x * real.exp x / (real.exp (a * x) - 1)

theorem find_a_for_even_function :
  ∀ (a : ℝ), is_even_function (given_function a) ↔ a = 2 :=
by 
  intro a
  sorry

end find_a_for_even_function_l283_283816


namespace largest_prime_factor_of_3136_l283_283529

theorem largest_prime_factor_of_3136 : ∃ p, p.prime ∧ p ∣ 3136 ∧ ∀ q, q.prime ∧ q ∣ 3136 → q ≤ p :=
sorry

end largest_prime_factor_of_3136_l283_283529


namespace regular_polygon_sides_l283_283613

theorem regular_polygon_sides (theta : ℝ) (h : theta = 18) : 
  ∃ n : ℕ, 360 / theta = n ∧ n = 20 :=
by
  use 20
  constructor
  apply eq_of_div_eq_mul
  rw h
  norm_num
  sorry

end regular_polygon_sides_l283_283613


namespace factorize_expr_l283_283254

theorem factorize_expr (x : ℝ) : x^3 - 4 * x = x * (x + 2) * (x - 2) :=
  sorry

end factorize_expr_l283_283254


namespace weekly_sessions_difference_l283_283219

def avg_sessions (sessions : ℕ) (weeks : ℕ) : ℝ := sessions / weeks

def total_average (session_counts : List ℕ) (weeks : ℕ) : ℝ :=
  let total_sessions := session_counts.sum
  let num_individuals := session_counts.length
  total_sessions / (num_individuals * weeks)

def absolute_difference (a b : ℝ) : ℝ := abs (a - b)

def total_difference (session_counts : List ℕ) (weeks : ℕ) : ℝ :=
  let overall_avg := total_average session_counts weeks
  session_counts.map (λs => absolute_difference (avg_sessions s weeks) overall_avg).sum

theorem weekly_sessions_difference :
  total_difference [16, 24, 30, 20, 36, 40] 8 = 5.75 :=
by
  sorry

end weekly_sessions_difference_l283_283219


namespace max_value_k_l283_283316

def f (x : ℝ) : ℝ := x * Real.log x
def l (k : ℤ) (x : ℝ) : ℝ := (k - 2) * x - k + 1

theorem max_value_k :
  ∀ x > 1, f x > l 4 x :=
by
  sorry

end max_value_k_l283_283316


namespace probability_both_classes_from_grade_one_l283_283591

theorem probability_both_classes_from_grade_one:
  let total_classes := 21 + 14 + 7 in
  let n1 := 3 in -- 3 classes selected from grade one.
  let n2 := 2 in -- 2 classes selected from grade two.
  let n3 := 1 in -- 1 class selected from grade three.
  let selected_classes := n1 + n2 + n3 in
  let total_pairs := (selected_classes * (selected_classes - 1)) / 2 in
  let grade_one_pairs := (n1 * (n1 - 1)) / 2 in
  (grade_one_pairs.to_rat / total_pairs.to_rat) = (1 / 5 : ℚ) :=
by
  sorry

end probability_both_classes_from_grade_one_l283_283591


namespace even_function_implies_a_is_2_l283_283792

noncomputable def f (a x : ℝ) : ℝ := (x * Real.exp x) / (Real.exp (a * x) - 1)

theorem even_function_implies_a_is_2 (a : ℝ) 
  (h : ∀ x : ℝ, f a x = f a (-x)) : a = 2 := 
by
  sorry

end even_function_implies_a_is_2_l283_283792


namespace no_alternative_way_to_construct_triangles_l283_283716

theorem no_alternative_way_to_construct_triangles (sticks : list ℕ) (h_len : sticks.length = 15)
  (h_set : ∃ (sets : list (list ℕ)), sets.length = 5 ∧ ∀ set ∈ sets, set.length = 3 
           ∧ (∀ (a b c : ℕ) (ha : a ∈ set) (hb : b ∈ set) (hc : c ∈ set), 
             (a + b > c ∧ a + c > b ∧ b + c > a))):
  ¬∃ (alt_sets : list (list ℕ)), alt_sets ≠ sets ∧ alt_sets.length = 5 
      ∧ ∀ set ∈ alt_sets, set.length = 3 
      ∧ (∀ (a b c : ℕ) (ha : a ∈ set) (hb : b ∈ set) (hc : c ∈ set),
        (a + b > c ∧ a + c > b ∧ b + c > a)) :=
sorry

end no_alternative_way_to_construct_triangles_l283_283716


namespace regular_hexagon_construction_l283_283287

noncomputable def construct_regular_hexagon (O : Point) (r : ℝ) : Prop :=
  ∃ (A B C D E F : Point), 
    IsRegularHexagon (Hexagon.mk A B C D E F) ∧
    InsideCircumscription (Hexagon.mk A B C D E F) (circle O r)

theorem regular_hexagon_construction (O : Point) (r : ℝ) : 
  construct_regular_hexagon O r :=
sorry

end regular_hexagon_construction_l283_283287


namespace factorize_x_cube_minus_4x_l283_283244

theorem factorize_x_cube_minus_4x (x : ℝ) : x^3 - 4 * x = x * (x + 2) * (x - 2) := 
by
  -- Continue the proof from here
  sorry

end factorize_x_cube_minus_4x_l283_283244


namespace no_other_way_to_construct_five_triangles_l283_283753

open Classical

theorem no_other_way_to_construct_five_triangles {
  sticks : List ℤ 
} (h_stick_lengths : 
  sticks = [2, 3, 4, 20, 30, 40, 200, 300, 400, 2000, 3000, 4000, 20000, 30000, 40000]) 
  (h_five_triplets : 
  ∀ t ∈ (sticks.nthLe 0 15).combinations 3, 
  (t[0] + t[1] > t[2]) ∧ (t[1] + t[2] > t[3]) ∧ (t[0] + t[2] > t[3])) :
  ¬ ∃ other_triplets, 
  (∀ t ∈ other_triplets, t.length = 3) ∧ 
  (∀ t ∈ other_triplets, (t[0] + t[1] > t[2]) ∧ (t[1] + t[2] > t[0]) ∧ (t[0] + t[2] > t[1])) ∧ 
  other_triplets ≠ (sticks.nthLe 0 15).combinations 3 := 
by
  sorry

end no_other_way_to_construct_five_triangles_l283_283753


namespace no_other_way_to_construct_five_triangles_l283_283751

open Classical

theorem no_other_way_to_construct_five_triangles {
  sticks : List ℤ 
} (h_stick_lengths : 
  sticks = [2, 3, 4, 20, 30, 40, 200, 300, 400, 2000, 3000, 4000, 20000, 30000, 40000]) 
  (h_five_triplets : 
  ∀ t ∈ (sticks.nthLe 0 15).combinations 3, 
  (t[0] + t[1] > t[2]) ∧ (t[1] + t[2] > t[3]) ∧ (t[0] + t[2] > t[3])) :
  ¬ ∃ other_triplets, 
  (∀ t ∈ other_triplets, t.length = 3) ∧ 
  (∀ t ∈ other_triplets, (t[0] + t[1] > t[2]) ∧ (t[1] + t[2] > t[0]) ∧ (t[0] + t[2] > t[1])) ∧ 
  other_triplets ≠ (sticks.nthLe 0 15).combinations 3 := 
by
  sorry

end no_other_way_to_construct_five_triangles_l283_283751


namespace find_a_even_function_l283_283988

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def given_function (a : ℝ) (x : ℝ) : ℝ :=
  x * Real.exp x / (Real.exp (a * x) - 1)

theorem find_a_even_function :
  (is_even_function (given_function 2)) → true :=
by
  intro h
  sorry

end find_a_even_function_l283_283988


namespace maximize_value_l283_283411

noncomputable def maximum_value (x y : ℝ) : ℝ :=
  3 * x - 2 * y

theorem maximize_value (x y : ℝ) (h : x^2 + y^2 + x * y = 1) : maximum_value x y ≤ 5 :=
sorry

end maximize_value_l283_283411


namespace third_smallest_1_tens_l283_283147

noncomputable def third_smallest_number_with_tens_1 (numbers : List ℕ) : ℕ :=
  let tens_1_numbers :=
    numbers.filter (λ n, n / 10 % 10 = 1)
  (tens_1_numbers.nth 2).getOrElse 0

theorem third_smallest_1_tens : third_smallest_number_with_tens_1 [1, 4, 6, 7] = 614 := 
by sorry

end third_smallest_1_tens_l283_283147


namespace even_function_implies_a_eq_2_l283_283856

theorem even_function_implies_a_eq_2 (a : ℝ) 
  (h : ∀ x : ℝ, f x = f (-x)) 
  (f : ℝ → ℝ := λ x, (x * exp x) / (exp (a * x) - 1)) : a = 2 :=
sorry

end even_function_implies_a_eq_2_l283_283856


namespace average_first_150_l283_283223

def sequence (n : ℕ) : ℤ :=
  (-1)^n * (3*n + 1) / 2

noncomputable def average_sequence (N : ℕ) (f : ℕ → ℤ) : ℚ :=
  (∑ i in finset.range N, f i) / N

theorem average_first_150 :
  average_sequence 150 sequence = -1/6 :=
sorry

end average_first_150_l283_283223


namespace square_digit_end_2_or_not_l283_283172

theorem square_digit_end_2_or_not
  (n : ℕ) :
  (∀ k : ℕ, k^2 % 10 ∈ {0, 1, 4, 5, 6, 9}) →
  (¬ ∃ m : ℕ, (m^2 % 10 = 2) ∧ (m ∈ {2, 3, 7, 8})) :=
by {
  intro h,
  sorry
}

end square_digit_end_2_or_not_l283_283172


namespace approximate_perimeter_of_semi_circle_l283_283195

def radius : ℝ := 10
def pi_approx : ℝ := 3.14159

noncomputable def perimeter_semi_circle := (10 * pi_approx) + (2 * radius)

theorem approximate_perimeter_of_semi_circle : perimeter_semi_circle ≈ 51.42 :=
by
  sorry

end approximate_perimeter_of_semi_circle_l283_283195


namespace parametric_plane_equation_l283_283289

theorem parametric_plane_equation :
  ∃ (A B C D : ℤ), (A > 0) ∧ (Int.gcd A (Int.gcd B (Int.gcd C D)) = 1) ∧
  (∀ (x y z : ℝ), (2 + 2 * x - 3 * y = 2 + 2 * x - 3 * y) ∧
                   (4 + x = 4 + x) ∧
                   (1 - 3 * x + 4 * y = 1 - 3 * x + 4 * y) ↔
                   4 * x + 5 * y + 3 * z - 31 = 0) :=
begin
  use [4, 5, 3, -31],
  split,
  { -- Proof that A > 0
    norm_num },
  split,
  { -- Proof that gcd condition
    norm_num },
  { -- Proof of the plane equation
    intros x y z,
    split,
    { intro h,
      simp at h,
      sorry },
    { intro h,
      simp at h,
      sorry }
  }
end

end parametric_plane_equation_l283_283289


namespace even_function_implies_a_eq_2_l283_283865

def f (x a : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2 (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → a = 2 :=
by
  sorry

end even_function_implies_a_eq_2_l283_283865


namespace even_function_implies_a_eq_2_l283_283922

def f (x : ℝ) (a : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2 (a : ℝ) (h_even : ∀ x, f (-x) a = f x a) : a = 2 :=
by
  intro x
  sorry

end even_function_implies_a_eq_2_l283_283922


namespace volume_of_cone_from_sector_is_correct_l283_283637

/-
The problem is to show that the volume of a cone formed by rolling a two-third sector of a circle
with a radius of 6 inches is equal to (32 / 3) * pi * sqrt 5 cubic inches.
-/

theorem volume_of_cone_from_sector_is_correct :
  let r := 6 in
  let sector_fraction := 2 / 3 in
  let arc_length := sector_fraction * (2 * Real.pi * r) in
  let base_radius := arc_length / (2 * Real.pi) in
  let slant_height := r in
  let cone_height := Real.sqrt (slant_height^2 - base_radius^2) in
  (1 / 3) * Real.pi * base_radius^2 * cone_height = (32 / 3) * Real.pi * Real.sqrt 5 :=
by
  sorry

end volume_of_cone_from_sector_is_correct_l283_283637


namespace even_function_implies_a_eq_2_l283_283858

theorem even_function_implies_a_eq_2 (a : ℝ) 
  (h : ∀ x : ℝ, f x = f (-x)) 
  (f : ℝ → ℝ := λ x, (x * exp x) / (exp (a * x) - 1)) : a = 2 :=
sorry

end even_function_implies_a_eq_2_l283_283858


namespace find_a_if_f_even_l283_283834

def f (a x : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem find_a_if_f_even (a : ℝ) (h : ∀ x : ℝ, x ≠ 0 → f a (-x) = f a x) : a = 2 :=
by 
  sorry

end find_a_if_f_even_l283_283834


namespace find_a_for_even_function_l283_283829

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def given_function (a : ℝ) (x : ℝ) : ℝ :=
  x * real.exp x / (real.exp (a * x) - 1)

theorem find_a_for_even_function :
  ∀ (a : ℝ), is_even_function (given_function a) ↔ a = 2 :=
by 
  intro a
  sorry

end find_a_for_even_function_l283_283829


namespace function_d_is_odd_and_increasing_l283_283208

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

noncomputable def is_increasing_function (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f (x) < f (y)

def interval_01 := {x : ℝ | 0 < x ∧ x < 1}

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

theorem function_d_is_odd_and_increasing :
  is_odd_function f ∧ is_increasing_function f interval_01 :=
by
  sorry

end function_d_is_odd_and_increasing_l283_283208


namespace length_of_c_l283_283347

open Real

theorem length_of_c (a b c : ℝ) (h1 : a^2 - 5*a + 2 = 0) (h2 : b^2 - 5*b + 2 = 0) (h3 : ∠C = π / 3) :
  c = sqrt 19 :=
sorry

end length_of_c_l283_283347


namespace roots_calc_l283_283514

theorem roots_calc {a b c d : ℝ} (h1: a ≠ 0) (h2 : 125 * a + 25 * b + 5 * c + d = 0) (h3 : -27 * a + 9 * b - 3 * c + d = 0) :
  (b + c) / a = -19 :=
by
  sorry

end roots_calc_l283_283514


namespace find_a_even_function_l283_283976

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def given_function (a : ℝ) (x : ℝ) : ℝ :=
  x * Real.exp x / (Real.exp (a * x) - 1)

theorem find_a_even_function :
  (is_even_function (given_function 2)) → true :=
by
  intro h
  sorry

end find_a_even_function_l283_283976


namespace simplify_fraction_l283_283067

theorem simplify_fraction :
  ∀ (a b : ℕ), a = 180 → b = 16200 → a / nat.gcd a b = 1 ∧ b / nat.gcd a b = 90 :=
by
  intros a b ha hb
  sorry

end simplify_fraction_l283_283067


namespace number_of_sides_of_regular_polygon_l283_283623

variable {α : Type*}

noncomputable def exterior_angle_of_regular_polygon (n : ℕ) : ℝ := 360 / n

theorem number_of_sides_of_regular_polygon (exterior_angle : ℝ) (n : ℕ) (h₁ : exterior_angle = 18) (h₂ : exterior_angle_of_regular_polygon n = exterior_angle) : n = 20 :=
by {
  -- Translation of given conditions
  have sum_exterior_angles : 360 = n * exterior_angle_of_regular_polygon n,
  -- Based on h₂ and h₁ provided
  rw [h₂, h₁] at sum_exterior_angles,
  -- Perform necessary simplifications
  have : 20 = (360 / 18 : ℕ),
  simp,
}

end number_of_sides_of_regular_polygon_l283_283623


namespace angle_ABD_obtuse_l283_283436

theorem angle_ABD_obtuse 
  (A B C D : Type)
  [ac : LinearOrderedField A]
  [Point : A → Type]
  [has_A : has_insert A A Point]
  (AC BC : A)
  (h1 : AC > BC)
  (h2 : ∀ CD, CD = BC)
  (h3: OnExtension A C D) :
  ∃ ABD, ABD > 90 :=
by sorry

end angle_ABD_obtuse_l283_283436


namespace regular_polygon_sides_l283_283602

theorem regular_polygon_sides (n : ℕ) (h : 1 < n) (exterior_angle : ℝ) (h_ext : exterior_angle = 18) :
  n * exterior_angle = 360 → n = 20 :=
by 
  sorry

end regular_polygon_sides_l283_283602


namespace minimize_cost_regular_minimize_cost_discount_l283_283111

-- Define the relevant conditions provided in the problem
def price_per_ton : ℕ := 1500
def transportation_fee : ℕ := 100
def daily_rice_need : ℕ := 1
def storage_cost_per_ton : ℕ := 2
def discount_threshold : ℕ := 20
def discount_rate : ℝ := 0.95

-- Define the cost functions for regular and discounted purchasing
def avg_daily_cost_regular (n : ℕ) : ℝ :=
  (2 * ((n * (n + 1)) / 2) + transportation_fee + price_per_ton * n) / n

def avg_daily_cost_discounted (m : ℕ) (h : m ≥ discount_threshold) : ℝ :=
  (2 * ((m * (m + 1)) / 2) + transportation_fee + price_per_ton * discount_rate * m) / m

-- The mathematical proof problem statement:
theorem minimize_cost_regular :
  let n := 10 in
  avg_daily_cost_regular n = 1521 := by
  sorry

theorem minimize_cost_discount :
  ∃ m ≥ discount_threshold, avg_daily_cost_discounted m (by assumption) < 1521 := by
  existsi 20
  apply nat.le_refl
  sorry

end minimize_cost_regular_minimize_cost_discount_l283_283111


namespace factorize_expr_l283_283252

theorem factorize_expr (x : ℝ) : x^3 - 4 * x = x * (x + 2) * (x - 2) :=
  sorry

end factorize_expr_l283_283252


namespace even_function_implies_a_eq_2_l283_283916

def f (x : ℝ) (a : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2 (a : ℝ) (h_even : ∀ x, f (-x) a = f x a) : a = 2 :=
by
  intro x
  sorry

end even_function_implies_a_eq_2_l283_283916


namespace red_ball_probability_correct_l283_283652

-- Definition of the problem conditions
def BoxA := {red := 5, white := 2}
def BoxB := {red := 4, white := 3}

-- A function to calculate the probability of the final event
def final_probability : ℚ :=
  let prob_red_from_A := (5 / 7 : ℚ)
  let prob_white_from_A := (2 / 7 : ℚ)
  let prob_red_after_red := (5 / 8 : ℚ)
  let prob_red_after_white := (4 / 8 : ℚ)
  prob_red_from_A * prob_red_after_red + prob_white_from_A * prob_red_after_white

-- The statement to prove
theorem red_ball_probability_correct :
  final_probability = 33 / 56 := by
  sorry

end red_ball_probability_correct_l283_283652


namespace factorize_x_l283_283260

theorem factorize_x^3_minus_4x (x : ℝ) : x^3 - 4 * x = x * (x + 2) * (x - 2) := 
sorry

end factorize_x_l283_283260


namespace kitty_cleaning_time_l283_283437

def weekly_cleaning_time (pick_up: ℕ) (vacuum: ℕ) (clean_windows: ℕ) (dust: ℕ) : ℕ :=
  pick_up + vacuum + clean_windows + dust

def total_cleaning_time (weeks: ℕ) (pick_up: ℕ) (vacuum: ℕ) (clean_windows: ℕ) (dust: ℕ) : ℕ :=
  weeks * weekly_cleaning_time pick_up vacuum clean_windows dust

theorem kitty_cleaning_time :
  total_cleaning_time 4 5 20 15 10 = 200 := by
  sorry

end kitty_cleaning_time_l283_283437


namespace coefficient_x3_expansion_l283_283691

theorem coefficient_x3_expansion : 
  (∃ (f : ℕ → ℕ → ℕ), (f 5 3 = nat.choose 5 3) ∧ (f 6 3 = nat.choose 6 3) ∧ 
  (f 5 3 - f 6 3 = -10)) :=
by
  existsi nat.choose
  simp
  sorry

end coefficient_x3_expansion_l283_283691


namespace even_function_implies_a_eq_2_l283_283933

theorem even_function_implies_a_eq_2 (a : ℝ) : 
  (∀ x, (f : ℝ → ℝ) x = (λ x : ℝ, x * exp x / (exp (a * x) - 1)) x -> (f (-x) = f x)) -> a = 2 :=
by
  sorry

end even_function_implies_a_eq_2_l283_283933


namespace minimum_value_l283_283407

open Real

theorem minimum_value {x y : ℝ} (hx : 0 < x) (hy : 0 < y) :
  ∃ u, (u = x + 1 / y + y + 1 / x) ∧ ((x + 1 / y) * (x + 1 / y - 100) + (y + 1 / x) * (y + 1 / x - 100) = 1 / 2 * (u - 100) ^ 2 - 2500) ∧ -2500 ≤ 1 / 2 * (u - 100) ^ 2 - 2500 :=
begin
  sorry
end

end minimum_value_l283_283407


namespace geometric_sequence_product_l283_283366

theorem geometric_sequence_product 
    (a : ℕ → ℝ)
    (h_geom : ∀ n m, a (n + m) = a n * a m)
    (h_roots : ∀ x, x^2 - 3*x + 2 = 0 → (x = a 7 ∨ x = a 13)) :
  a 2 * a 18 = 2 := 
sorry

end geometric_sequence_product_l283_283366


namespace sin_sq_theta_is_27_over_32_l283_283030

theorem sin_sq_theta_is_27_over_32 :
  ∀ (A B C D : Point) (θ : ℝ),
  IsIsoscelesTrapezoid A B C D →
  length A B = 5 →
  length C D = 8 →
  length B C = 6 →
  ∃ X, ∠ A X D = 180 - ∠ B X C → θ →

  sin θ ^ 2 = 27 / 32 :=
by
  intro A B C D θ H_trapezoid H_ab H_cd H_bc H_exists_angle
  sorry

end sin_sq_theta_is_27_over_32_l283_283030


namespace even_function_implies_a_eq_2_l283_283860

theorem even_function_implies_a_eq_2 (a : ℝ) 
  (h : ∀ x : ℝ, f x = f (-x)) 
  (f : ℝ → ℝ := λ x, (x * exp x) / (exp (a * x) - 1)) : a = 2 :=
sorry

end even_function_implies_a_eq_2_l283_283860


namespace triangle_angles_from_circle_division_l283_283177

theorem triangle_angles_from_circle_division (r1 r2 r3 : ℕ) (total_circumference : ℝ)
  (h1 : r1 = 7) (h2 : r2 = 11) (h3 : r3 = 6) (h4 : total_circumference = 360) :
  let part_measure := total_circumference / (r1 + r2 + r3),
      arc1 := r1 * part_measure,
      arc2 := r2 * part_measure,
      arc3 := r3 * part_measure,
      angle1 := arc1 / 2,
      angle2 := arc2 / 2,
      angle3 := arc3 / 2
  in angle1 = 52.5 ∧ angle2 = 82.5 ∧ angle3 = 45 :=
by
  sorry

end triangle_angles_from_circle_division_l283_283177


namespace Owen_final_turtle_count_l283_283442

variable (Owen_turtles : ℕ) (Johanna_turtles : ℕ)

def final_turtles (Owen_turtles Johanna_turtles : ℕ) : ℕ :=
  let initial_Owen_turtles := Owen_turtles
  let initial_Johanna_turtles := Owen_turtles - 5
  let Owen_after_month := initial_Owen_turtles * 2
  let Johanna_after_losing_half := initial_Johanna_turtles / 2
  let Owen_after_donation := Owen_after_month + Johanna_after_losing_half
  Owen_after_donation

theorem Owen_final_turtle_count : final_turtles 21 (21 - 5) = 50 :=
by
  sorry

end Owen_final_turtle_count_l283_283442


namespace simplify_expr_l283_283472

theorem simplify_expr (x : ℝ) :
  2 * x^2 * (4 * x^3 - 3 * x + 5) - 4 * (x^3 - x^2 + 3 * x - 8) =
    8 * x^5 - 10 * x^3 + 14 * x^2 - 12 * x + 32 :=
by
  sorry

end simplify_expr_l283_283472


namespace largest_divisor_of_product_of_three_consecutive_odd_integers_l283_283034

theorem largest_divisor_of_product_of_three_consecutive_odd_integers :
  ∀ n : ℕ, n > 0 → ∃ d : ℕ, d = 3 ∧ ∀ m : ℕ, m ∣ ((2*n-1)*(2*n+1)*(2*n+3)) → m ≤ d :=
by
  sorry

end largest_divisor_of_product_of_three_consecutive_odd_integers_l283_283034


namespace no_other_way_five_triangles_l283_283705

def is_triangle {α : Type*} [linear_ordered_field α] (a b c : α) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem no_other_way_five_triangles (sticks : list ℝ)
  (h_len : sticks.length = 15)
  (h_sets : ∃ (sets : list (ℝ × ℝ × ℝ)), 
    sets.length = 5 ∧ ∀ {a b c}, (a, b, c) ∈ sets → is_triangle a b c) :
  ∀ sets, sets.length = 5 ∧ ∀ {a b c}, (a, b, c) ∈ sets → is_triangle a b c → sets = h_sets :=
begin
  sorry
end

end no_other_way_five_triangles_l283_283705


namespace min_value_of_expression_l283_283140

theorem min_value_of_expression : ∀ s : ℝ, ∃ m : ℝ, m = 148 ∧ m ≤ -8 * s^2 + 64 * s + 20 := 
by
  intro s
  use 148
  split
  case left
    rfl
  case right
    calc
      -8 * s^2 + 64 * s + 20 = -8 * (s^2 - 8 * s + 16) + 148 - 128 + 20 : by sorry
                            ... = -8 * (s - 4)^2 + 148                       : by sorry
                            ... ≤ 148                                        : by sorry

end min_value_of_expression_l283_283140


namespace gnome_seating_possible_for_odd_n_l283_283171

-- Problem Statement in Lean 4
theorem gnome_seating_possible_for_odd_n (n : ℕ) (h : n % 2 = 1) (h_gt1 : n > 1) : 
  ∃ (seating : List ℕ), (∀ i j : ℕ, (i < seating.length ∧ j < seating.length ∧ seating.nth i ≠ seating.nth j) → 
    (seating.nth i = seating.nth (j + 1 % seating.length) ∨ 
     seating.nth i = seating.nth (j - 1 % seating.length))) 
  := by
  sorry

end gnome_seating_possible_for_odd_n_l283_283171


namespace value_added_to_number_l283_283505

theorem value_added_to_number (n v : ℤ) (h1 : n = 9)
  (h2 : 3 * (n + 2) = v + n) : v = 24 :=
by
  sorry

end value_added_to_number_l283_283505


namespace even_function_implies_a_eq_2_l283_283932

theorem even_function_implies_a_eq_2 (a : ℝ) : 
  (∀ x, (f : ℝ → ℝ) x = (λ x : ℝ, x * exp x / (exp (a * x) - 1)) x -> (f (-x) = f x)) -> a = 2 :=
by
  sorry

end even_function_implies_a_eq_2_l283_283932


namespace vector_representation_l283_283640

variables (a e₁ e₂ : ℝ × ℝ)
variables (λ μ : ℝ)

def vec_add (v₁ v₂ : ℝ × ℝ) : ℝ × ℝ :=
  (v₁.1 + v₂.1, v₁.2 + v₂.2)

def scalar_mul (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (c * v.1, c * v.2)

def is_linear_combination (a e₁ e₂ : ℝ × ℝ) : Prop :=
  ∃ λ μ : ℝ, a = vec_add (scalar_mul λ e₁) (scalar_mul μ e₂)

theorem vector_representation :
  let a := (3, -2) in
  let e₁ := (-1, 2) in
  let e₂ := (5, -2) in
  is_linear_combination a e₁ e₂ :=
by
  sorry

end vector_representation_l283_283640


namespace even_function_implies_a_eq_2_l283_283855

theorem even_function_implies_a_eq_2 (a : ℝ) 
  (h : ∀ x : ℝ, f x = f (-x)) 
  (f : ℝ → ℝ := λ x, (x * exp x) / (exp (a * x) - 1)) : a = 2 :=
sorry

end even_function_implies_a_eq_2_l283_283855


namespace fathers_age_multiple_l283_283497

theorem fathers_age_multiple 
  (Johns_age : ℕ)
  (sum_of_ages : ℕ)
  (additional_years : ℕ)
  (m : ℕ)
  (h1 : Johns_age = 15)
  (h2 : sum_of_ages = 77)
  (h3 : additional_years = 32)
  (h4 : sum_of_ages = Johns_age + (Johns_age * m + additional_years)) :
  m = 2 := 
by 
  sorry

end fathers_age_multiple_l283_283497


namespace number_of_sides_of_regular_polygon_l283_283622

variable {α : Type*}

noncomputable def exterior_angle_of_regular_polygon (n : ℕ) : ℝ := 360 / n

theorem number_of_sides_of_regular_polygon (exterior_angle : ℝ) (n : ℕ) (h₁ : exterior_angle = 18) (h₂ : exterior_angle_of_regular_polygon n = exterior_angle) : n = 20 :=
by {
  -- Translation of given conditions
  have sum_exterior_angles : 360 = n * exterior_angle_of_regular_polygon n,
  -- Based on h₂ and h₁ provided
  rw [h₂, h₁] at sum_exterior_angles,
  -- Perform necessary simplifications
  have : 20 = (360 / 18 : ℕ),
  simp,
}

end number_of_sides_of_regular_polygon_l283_283622


namespace domain_of_f_l283_283670

noncomputable def f (x : ℝ) : ℝ := real.sqrt (1 - real.log x / real.log 10)

theorem domain_of_f :
  { x : ℝ | 0 < x ∧ x ≤ 10 } = { x : ℝ | f x = real.sqrt (1 - real.log x / real.log 10) ∧ 1 - real.log x / real.log 10 ≥ 0 } := 
sorry

end domain_of_f_l283_283670


namespace cos_double_alpha_proof_l283_283335

theorem cos_double_alpha_proof (α : ℝ) (h1 : Real.sin (π / 3 - α) = 1 / 3) : 
  Real.cos (π / 3 + 2 * α) = - 7 / 9 :=
by
  sorry

end cos_double_alpha_proof_l283_283335


namespace Tim_running_hours_l283_283509

theorem Tim_running_hours
  (initial_days : ℕ)
  (additional_days : ℕ)
  (hours_per_session : ℕ)
  (sessions_per_day : ℕ)
  (total_days : ℕ)
  (total_hours_per_week : ℕ) :
  initial_days = 3 →
  additional_days = 2 →
  hours_per_session = 1 →
  sessions_per_day = 2 →
  total_days = initial_days + additional_days →
  total_hours_per_week = total_days * (hours_per_session * sessions_per_day) →
  total_hours_per_week = 10 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4] at h5
  rw [nat.add_comm (initial_days * (hours_per_session * sessions_per_day)), nat.add_assoc, nat.mul_comm hours_per_session sessions_per_day] at h5
  rw h5 at h6
  exact h6

end Tim_running_hours_l283_283509


namespace even_function_implies_a_is_2_l283_283794

noncomputable def f (a x : ℝ) : ℝ := (x * Real.exp x) / (Real.exp (a * x) - 1)

theorem even_function_implies_a_is_2 (a : ℝ) 
  (h : ∀ x : ℝ, f a x = f a (-x)) : a = 2 := 
by
  sorry

end even_function_implies_a_is_2_l283_283794


namespace eq_a_2_l283_283966

def f (a x : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

-- Definition of even function: ∀ x, f(-x) = f(x)
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem eq_a_2 (a : ℝ) : (even_function (f a) → a = 2) ∧ (a = 2 → even_function (f a)) :=
by
  sorry

end eq_a_2_l283_283966


namespace no_alternative_way_to_construct_triangles_l283_283724

-- Define the triangle inequality condition
def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the set of stick lengths for each required triangle
def valid_triangle_sets (stick_set : Set (Set ℕ)) : Prop :=
  stick_set = { {2, 3, 4}, {20, 30, 40}, {200, 300, 400}, {2000, 3000, 4000}, {20000, 30000, 40000} }

-- Define the problem statement
theorem no_alternative_way_to_construct_triangles (s : Set ℕ) (h : s.card = 15) :
  (∀ t ∈ (s.powerset.filter (λ t, t.card = 3)), triangle_inequality t.some (finset_some_spec t) ∧ (∃ sets : Set (Set ℕ), sets.card = 5 ∧ ∀ u ∈ sets, u.card = 3 ∧ triangle_inequality u.some (finset_some_spec u)))
  → valid_triangle_sets (s.powerset.filter (λ t, t.card = 3)) := 
sorry

end no_alternative_way_to_construct_triangles_l283_283724


namespace angle_VOC_l283_283521

theorem angle_VOC : 
  let V_lat := (0 : Real) -- Vikram's latitude
  let V_long := (78 : Real) -- Vikram's longitude
  let C_lat := (30 : Real) -- Charles' latitude
  let C_long := (91 : Real) -- Charles' longitude
  ∀ (O : Type) [IsSphere O],
  -- Given spherical coordinates for V and C
  let V := (V_lat, V_long)
  let C := (C_lat, C_long)
  -- Calculate the angle VOC
  angle_VOC = Real.arccos (-Real.cos (C_lat) * Real.cos (169))


end angle_VOC_l283_283521


namespace no_alternate_way_to_form_triangles_l283_283730

noncomputable def canFormTriangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def validSetOfTriples : List (ℕ × ℕ × ℕ) :=
  [(2, 3, 4), (20, 30, 40), (200, 300, 400), (2000, 3000, 4000), (20000, 30000, 40000)]

theorem no_alternate_way_to_form_triangles (stickSet : List (ℕ × ℕ × ℕ)) :
  (∀ (t : ℕ × ℕ × ℕ), t ∈ stickSet → canFormTriangle t.1 t.2 t.3) →
  length stickSet = 5 →
  stickSet = validSetOfTriples ->
  (∀ alternateSet : List (ℕ × ℕ × ℕ),
    ∀ (t : ℕ × ℕ × ℕ), t ∈ alternateSet → canFormTriangle t.1 t.2 t.3 →
    length alternateSet = 5 →
    duplicate_stick_alternateSet = duplicate_stick_stickSet) :=
sorry

end no_alternate_way_to_form_triangles_l283_283730


namespace even_function_implies_a_is_2_l283_283795

noncomputable def f (a x : ℝ) : ℝ := (x * Real.exp x) / (Real.exp (a * x) - 1)

theorem even_function_implies_a_is_2 (a : ℝ) 
  (h : ∀ x : ℝ, f a x = f a (-x)) : a = 2 := 
by
  sorry

end even_function_implies_a_is_2_l283_283795


namespace regular_polygon_sides_l283_283619

theorem regular_polygon_sides (n : ℕ) (h : 360 = 18 * n) : n = 20 := 
by 
  sorry

end regular_polygon_sides_l283_283619


namespace sum_of_number_and_reverse_l283_283090

def digit_representation (n m : ℕ) (a b : ℕ) :=
  n = 10 * a + b ∧
  m = 10 * b + a ∧
  n - m = 9 * (a * b) + 3

theorem sum_of_number_and_reverse :
  ∃ a b n m : ℕ, digit_representation n m a b ∧ n + m = 22 :=
by
  sorry

end sum_of_number_and_reverse_l283_283090


namespace license_plate_palindrome_probability_l283_283423

def isPalindrome (s : String) : Prop :=
  s = s.reverse

def four_digit_palindrome_prob : ℚ :=
  (10 * 10 : ℚ) / (10 ^ 4)

def four_letter_palindrome_prob : ℚ :=
  (26 * 26 : ℚ) / (26 ^ 4)

def combined_palindrome_prob : ℚ :=
  four_digit_palindrome_prob + four_letter_palindrome_prob - four_digit_palindrome_prob * four_letter_palindrome_prob

def simplified_frac (r : ℚ) : ℚ :=
  (97 : ℚ) / 8450

theorem license_plate_palindrome_probability :
  ∑ (p q : ℕ) (h₁ : p = 97) (h₂ : q = 8450), p + q = 8547 :=
sorry

end license_plate_palindrome_probability_l283_283423


namespace find_a_for_even_function_l283_283820

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def given_function (a : ℝ) (x : ℝ) : ℝ :=
  x * real.exp x / (real.exp (a * x) - 1)

theorem find_a_for_even_function :
  ∀ (a : ℝ), is_even_function (given_function a) ↔ a = 2 :=
by 
  intro a
  sorry

end find_a_for_even_function_l283_283820


namespace four_digit_perfect_square_l283_283686

theorem four_digit_perfect_square (N : ℕ) : 
  (1000 ≤ N ∧ N < 10000) ∧ ∃ x y : ℕ, 
  (N = 1100 * x + 11 * y) ∧ 
  (N = 7744) ∧ 
  (N % 11 = 0) ∧ 
  (int.sqrt N) * (int.sqrt N) = N :=
by 
  sorry

end four_digit_perfect_square_l283_283686


namespace probability_sqrt_less_than_seven_l283_283545

-- Definitions and conditions from part a)
def is_two_digit_number (n : ℕ) : Prop := (10 ≤ n) ∧ (n ≤ 99)
def sqrt_less_than_seven (n : ℕ) : Prop := real.sqrt n < 7

-- Lean 4 statement for the actual proof problem
theorem probability_sqrt_less_than_seven : 
  (∃ n, is_two_digit_number n ∧ sqrt_less_than_seven n) → ∑ i in (finset.range 100).filter is_two_digit_number, if sqrt_less_than_seven i then 1 else 0 = 39 :=
sorry

end probability_sqrt_less_than_seven_l283_283545


namespace no_other_way_to_construct_five_triangles_l283_283754

open Classical

theorem no_other_way_to_construct_five_triangles {
  sticks : List ℤ 
} (h_stick_lengths : 
  sticks = [2, 3, 4, 20, 30, 40, 200, 300, 400, 2000, 3000, 4000, 20000, 30000, 40000]) 
  (h_five_triplets : 
  ∀ t ∈ (sticks.nthLe 0 15).combinations 3, 
  (t[0] + t[1] > t[2]) ∧ (t[1] + t[2] > t[3]) ∧ (t[0] + t[2] > t[3])) :
  ¬ ∃ other_triplets, 
  (∀ t ∈ other_triplets, t.length = 3) ∧ 
  (∀ t ∈ other_triplets, (t[0] + t[1] > t[2]) ∧ (t[1] + t[2] > t[0]) ∧ (t[0] + t[2] > t[1])) ∧ 
  other_triplets ≠ (sticks.nthLe 0 15).combinations 3 := 
by
  sorry

end no_other_way_to_construct_five_triangles_l283_283754


namespace unique_triangle_assembly_l283_283743

def triangle_inequality (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

def is_valid_triangle_set (sticks : List (ℕ × ℕ × ℕ)) : Prop :=
  sticks.all (λ stick_lengths, triangle_inequality stick_lengths.1 stick_lengths.2 stick_lengths.3)

theorem unique_triangle_assembly (sticks : List ℕ) (h_len : sticks.length = 15)
  (h_sticks : sticks = [2, 3, 4, 20, 30, 40, 200, 300, 400, 2000, 3000, 4000, 20000, 30000, 40000]) :
  ¬ ∃ (other_config : List (ℕ × ℕ × ℕ)),
    is_valid_triangle_set other_config ∧
    other_config ≠ [(2, 3, 4), (20, 30, 40), (200, 300, 400), (2000, 3000, 4000), (20000, 30000, 40000)] :=
begin
  sorry
end

end unique_triangle_assembly_l283_283743


namespace maximum_value_of_sqrt_expr_le_6_l283_283045

noncomputable def maximum_value_of_sqrt_expr (a b c : ℝ) (h₀ : a ≥ 0) (h₁ : b ≥ 0) (h₂ : c ≥ 0) (h : a + b + c = 7) : ℝ :=
    sqrt (3 * a + 1) + sqrt (3 * b + 1) + sqrt (3 * c + 1)

theorem maximum_value_of_sqrt_expr_le_6 :
  ∃ a b c : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 7 ∧ (maximum_value_of_sqrt_expr a b c H0 H1 H2 H) ≤ 6 :=
sorry

end maximum_value_of_sqrt_expr_le_6_l283_283045


namespace infinitude_of_composite_z_l283_283455

theorem infinitude_of_composite_z (a : ℕ) (h : ∃ k : ℕ, k > 1 ∧ a = 4 * k^4) : 
  ∀ n : ℕ, ¬ Prime (n^4 + a) :=
by sorry

end infinitude_of_composite_z_l283_283455


namespace length_CP_equals_incircle_radius_l283_283372

theorem length_CP_equals_incircle_radius {A B C M N P: Type*} 
  [IncircleRadius ABC r] [RightAngleTriangleABC ABC A B C]
  [AngleBisector M A B AC] [AngleBisector N B A BC]
  [IntersectionPoint P MN AltitudeFromC] 
  : length (CP) = incircle_radius ABC := 
sorry

end length_CP_equals_incircle_radius_l283_283372


namespace build_wall_time_l283_283215

theorem build_wall_time :
  let brenda_rate := 720 / 8
  let brandon_rate := 720 / 12
  let combined_rate := brenda_rate + brandon_rate - 20
  let total_bricks := 720
  total_bricks / combined_rate ≈ 5.54 :=
by
  sorry

end build_wall_time_l283_283215


namespace no_alternate_way_to_form_triangles_l283_283726

noncomputable def canFormTriangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def validSetOfTriples : List (ℕ × ℕ × ℕ) :=
  [(2, 3, 4), (20, 30, 40), (200, 300, 400), (2000, 3000, 4000), (20000, 30000, 40000)]

theorem no_alternate_way_to_form_triangles (stickSet : List (ℕ × ℕ × ℕ)) :
  (∀ (t : ℕ × ℕ × ℕ), t ∈ stickSet → canFormTriangle t.1 t.2 t.3) →
  length stickSet = 5 →
  stickSet = validSetOfTriples ->
  (∀ alternateSet : List (ℕ × ℕ × ℕ),
    ∀ (t : ℕ × ℕ × ℕ), t ∈ alternateSet → canFormTriangle t.1 t.2 t.3 →
    length alternateSet = 5 →
    duplicate_stick_alternateSet = duplicate_stick_stickSet) :=
sorry

end no_alternate_way_to_form_triangles_l283_283726


namespace speed_of_man_in_still_water_l283_283160

-- Define the conditions as constants
constant v_s : ℝ
constant v_m : ℝ

-- The given conditions translated to Lean statements
axiom condition_1 : 30 = (v_m + v_s) * 3
axiom condition_2 : 18 = (v_m - v_s) * 3

-- The theorem that states the speed of the man in still water is 8 km/h
theorem speed_of_man_in_still_water : v_m = 8 := 
by {
  -- formal proof should go here
  sorry
}

end speed_of_man_in_still_water_l283_283160


namespace coefficient_x4_term_l283_283690

def binomial_coeff (n k : ℕ) : ℤ := (Nat.choose n k : ℤ)

def expand_binomial (coeffs : List ℤ) (n : ℕ) (x : ℤ) : ℤ :=
coeffs.foldrWithIndex (λ i a acc, acc + a * (x^i)) 0

def binomial_expansion (n : ℕ) (x : ℤ) : List ℤ := 
List.range (n + 1).map (λ r, (-1)^(n-r) * binomial_coeff n r * x^r)

theorem coefficient_x4_term :
  expand_binomial (binomial_expansion 5 1) 2 (-1) =
     -5 := by
  sorry

end coefficient_x4_term_l283_283690


namespace exists_n_harmonic_gt_10_exists_n_harmonic_gt_1000_l283_283377

theorem exists_n_harmonic_gt_10 : ∃ n : ℕ, ∑ k in Finset.range (n + 1), (1 / (k + 1 : ℝ)) > 10 := 
sorry

theorem exists_n_harmonic_gt_1000 : ∃ n : ℕ, ∑ k in Finset.range (n + 1), (1 / (k + 1 : ℝ)) > 1000 :=
sorry

end exists_n_harmonic_gt_10_exists_n_harmonic_gt_1000_l283_283377


namespace problem_statement_l283_283277

noncomputable def logBase (n m : ℝ) := Real.log m / Real.log n

theorem problem_statement (m n : ℝ) (h1 : 1 < m) (h2 : m < n) :
  let a := (logBase n m)^2,
      b := logBase n m^2,
      c := logBase n (logBase n m)
  in c < a ∧ a < b :=
by
  let log_n_m := logBase n m
  let a := log_n_m ^ 2
  let b := logBase n (m ^ 2)
  let c := logBase n log_n_m
  sorry

end problem_statement_l283_283277


namespace juliet_remainder_l283_283028

theorem juliet_remainder (c d : ℤ) : (c ≡ 86 [MOD 100]) → (d ≡ 144 [MOD 150]) → (c + d ≡ 30 [MOD 50]) :=
by
  intros h_c h_d
  sorry

end juliet_remainder_l283_283028


namespace system1_solution_system2_solution_l283_283071

theorem system1_solution :
  ∃ x y : ℝ, 3 * x + 4 * y = 16 ∧ 5 * x - 8 * y = 34 ∧ x = 6 ∧ y = -1/2 :=
by
  sorry

theorem system2_solution :
  ∃ x y : ℝ, (x - 1) / 2 + (y + 1) / 3 = 1 ∧ x + y = 4 ∧ x = -1 ∧ y = 5 :=
by
  sorry

end system1_solution_system2_solution_l283_283071


namespace even_function_implies_a_eq_2_l283_283906

def f (a x : ℝ) : ℝ := (x * exp x) / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2
  (h : ∀ x : ℝ, f a x = f a (-x)) : a = 2 := 
sorry

end even_function_implies_a_eq_2_l283_283906


namespace sum_u_n_l283_283522

-- Initial conditions
def u0 : ℝ × ℝ := (2, 1)
def z0 : ℝ × ℝ := (3, -1)

-- Projection function
def proj (a b : ℝ × ℝ) : ℝ × ℝ :=
  let scale := (a.1 * b.1 + a.2 * b.2)/(b.1 * b.1 + b.2 * b.2)
  (scale * b.1, scale * b.2)

-- Recursive definitions
def u : ℕ → ℝ × ℝ
| 0 => u0
| n+1 => proj (z n) u0

def z : ℕ → ℝ × ℝ
| 0 => z0
| n+1 => proj (u n) z0

-- Sum u_n starting from 1
def sum_u := (2*u0.1, 2*u0.2)

theorem sum_u_n (s : ℝ × ℝ) :
  s = (u 1).1 + (u 2).1 + (u 3).1 + ... :=
  s = sum_u := by sorry

end sum_u_n_l283_283522


namespace total_computers_needed_l283_283156

theorem total_computers_needed (initial_students : ℕ) (students_per_computer : ℕ) (additional_students : ℕ) :
  initial_students = 82 →
  students_per_computer = 2 →
  additional_students = 16 →
  (initial_students + additional_students) / students_per_computer = 49 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end total_computers_needed_l283_283156


namespace marla_errand_total_time_l283_283426

theorem marla_errand_total_time :
  let drive_time := 20
  let school_time := 70
  let total_time := 2 * drive_time + school_time
  total_time = 110 :=
by
  let drive_time := 20
  let school_time := 70
  let total_time := 2 * drive_time + school_time
  show total_time = 110
  sorry

end marla_errand_total_time_l283_283426


namespace number_of_sides_of_regular_polygon_l283_283624

variable {α : Type*}

noncomputable def exterior_angle_of_regular_polygon (n : ℕ) : ℝ := 360 / n

theorem number_of_sides_of_regular_polygon (exterior_angle : ℝ) (n : ℕ) (h₁ : exterior_angle = 18) (h₂ : exterior_angle_of_regular_polygon n = exterior_angle) : n = 20 :=
by {
  -- Translation of given conditions
  have sum_exterior_angles : 360 = n * exterior_angle_of_regular_polygon n,
  -- Based on h₂ and h₁ provided
  rw [h₂, h₁] at sum_exterior_angles,
  -- Perform necessary simplifications
  have : 20 = (360 / 18 : ℕ),
  simp,
}

end number_of_sides_of_regular_polygon_l283_283624


namespace degree_polynomial_expression_l283_283237

noncomputable def P (x : ℝ) : ℝ := 4 * x^6 + 2 * x^5 - 3 * x + 6
noncomputable def Q (x : ℝ) : ℝ := 5 * x^12 - 2 * x^9 + 7 * x^6 - 15
noncomputable def R (x : ℝ) : ℝ := 3 * x^3 + 4

theorem degree_polynomial_expression : 
  ∀ x : ℝ, degree (P x * Q x - (R x)^6) = 18 := 
by
  sorry

end degree_polynomial_expression_l283_283237


namespace oscar_leap_longer_l283_283650

-- Definitions based on conditions
def number_poles : ℕ := 51
def total_distance_feet : ℝ := 5280
def elmer_strides_per_gap : ℕ := 38
def oscar_leaps_per_gap : ℕ := 15

-- Number of gaps between the poles 
def number_gaps : ℕ := number_poles - 1

-- Total number of strides and leaps
def total_elmer_strides : ℕ := elmer_strides_per_gap * number_gaps
def total_oscar_leaps : ℕ := oscar_leaps_per_gap * number_gaps

-- Length of each stride and leap
def length_elmer_stride : ℝ := total_distance_feet / total_elmer_strides
def length_oscar_leap : ℝ := total_distance_feet / total_oscar_leaps

-- Difference in length between Oscar's leap and Elmer's stride
def difference_in_length : ℝ := length_oscar_leap - length_elmer_stride

theorem oscar_leap_longer : difference_in_length = 4.26 := by
  sorry

end oscar_leap_longer_l283_283650


namespace jeffreys_total_steps_l283_283144

-- Define the conditions
def effective_steps_per_pattern : ℕ := 1
def total_effective_distance : ℕ := 66
def steps_per_pattern : ℕ := 5

-- Define the proof problem
theorem jeffreys_total_steps : ∀ (N : ℕ), 
  N = (total_effective_distance * steps_per_pattern) := 
sorry

end jeffreys_total_steps_l283_283144


namespace tan_alpha_half_sin_2alpha_plus_pi4_l283_283761

section

variables {α : ℝ} (ha : ∀ θ ∈ Ioo 0 (π / 4), (2 * sin θ, 1) → (cos θ, 1))

/-- Problem (1): If the vectors are parallel, then the tangent of the angle is 1/2 -/
theorem tan_alpha_half (h : ∃ θ ∈ Ioo 0 (π / 4), (2 * sin θ / cos θ = 1/2)) : 
  ∃ θ ∈ Ioo 0 (π / 4), tan θ = 1 / 2 := 
sorry

/-- Problem (2): If the dot product equals 9/5, then sin(2α + π/4) equals 7√2/10 -/
theorem sin_2alpha_plus_pi4 (h : ∃ θ ∈ Ioo 0 (π / 4), 2 * sin θ * cos θ + 1 = 9 / 5) : 
  ∃ θ ∈ Ioo 0 (π / 4), sin (2 * θ + π / 4) = 7 * sqrt 2 / 10 :=
sorry

end

end tan_alpha_half_sin_2alpha_plus_pi4_l283_283761


namespace expression_takes_many_values_l283_283665

theorem expression_takes_many_values (x : ℝ) (h₁ : x ≠ 3) (h₂ : x ≠ -2) :
  (∃ y : ℝ, y ≠ 0 ∧ y ≠ (y + 1) ∧ 
    (3 * x ^ 2 + 2 * x - 5) / ((x - 3) * (x + 2)) - (5 * x - 7) / ((x - 3) * (x + 2)) = y) :=
by
  sorry

end expression_takes_many_values_l283_283665


namespace range_of_a_l283_283281

theorem range_of_a (a : ℝ) (h_cond1 : log a (1/2) < 1) (h_cond2 : a^(1/2) < 1) : 0 < a ∧ a < 1/2 :=
by
  sorry

end range_of_a_l283_283281


namespace eq_a_2_l283_283969

def f (a x : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

-- Definition of even function: ∀ x, f(-x) = f(x)
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem eq_a_2 (a : ℝ) : (even_function (f a) → a = 2) ∧ (a = 2 → even_function (f a)) :=
by
  sorry

end eq_a_2_l283_283969


namespace positive_n_modulus_comparison_l283_283672

theorem positive_n_modulus_comparison (n : ℕ) (h : |5 + n * I| = 5 * Real.sqrt 26) : n = 25 := 
  sorry

end positive_n_modulus_comparison_l283_283672


namespace regular_polygon_sides_l283_283617

theorem regular_polygon_sides (n : ℕ) (h : 360 = 18 * n) : n = 20 := 
by 
  sorry

end regular_polygon_sides_l283_283617


namespace coin_landing_heads_prob_l283_283175

theorem coin_landing_heads_prob (p : ℝ) (h : p^2 * (1 - p)^3 = 0.03125) : p = 0.5 :=
by
sorry

end coin_landing_heads_prob_l283_283175


namespace find_a_if_f_even_l283_283831

def f (a x : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem find_a_if_f_even (a : ℝ) (h : ∀ x : ℝ, x ≠ 0 → f a (-x) = f a x) : a = 2 :=
by 
  sorry

end find_a_if_f_even_l283_283831


namespace find_directrix_l283_283321

/-- Statement of the problem -/
def parabola_directrix (p : ℝ) (y_midpoint : ℝ) : Prop :=
  (0 < p) ∧
  (y_midpoint = 2) ∧
  (∃ (A B : ℝ × ℝ), 
    (A.1 > 0 ∧ A.2 ^ 2 = 2 * p * A.1) ∧
    (B.1 > 0 ∧ B.2 ^ 2 = 2 * p * B.1) ∧
    (((A.2 + B.2) / 2 = 2) ∧
    (∃ (m : ℝ), (m = 1) ∧ ((2 / (A.2 + B.2)) = 1)))) →
  (directrix_eq : ℝ) (directrix_eq = -1)

theorem find_directrix (p : ℝ) (y_midpoint : ℝ) :
  parabola_directrix p y_midpoint :=
sorry

end find_directrix_l283_283321


namespace q_investment_l283_283565

variable {p q : ℝ}
variable {investment_p : ℝ} (investment_p = 500000)
variable {profit_ratio_p profit_ratio_q : ℝ} (profit_ratio_p = 2) (profit_ratio_q = 4)
variable {investment_q : ℝ}

theorem q_investment (h : investment_p / profit_ratio_p = investment_q / profit_ratio_q) : investment_q = 1000000 := by
  sorry

end q_investment_l283_283565


namespace area_DBC_l283_283364

open Real

def point := ℝ × ℝ

def midpoint (p1 p2: point) : point := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def dist (p1 p2: point) : ℝ := sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def point_A : point := (0, 8)
def point_B : point := (0, 0)
def point_C : point := (10, 2)

def point_D : point := midpoint point_A point_B -- (0, 4)
def point_E : point := midpoint point_B point_C -- (5, 1)

def area_triangle (p1 p2 p3: point) : ℝ :=
  abs ((p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2)) / 2)

theorem area_DBC :
  area_triangle point_D point_B point_C = 20 :=
sorry

end area_DBC_l283_283364


namespace fish_population_estimation_l283_283166

theorem fish_population_estimation (N : ℕ) (h1 : 80 ≤ N)
  (h_tagged_returned : true)
  (h_second_catch : 80 ≤ N)
  (h_tagged_in_second_catch : 2 = 80 * 80 / N) :
  N = 3200 :=
by
  sorry

end fish_population_estimation_l283_283166


namespace no_other_way_five_triangles_l283_283703

def is_triangle {α : Type*} [linear_ordered_field α] (a b c : α) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem no_other_way_five_triangles (sticks : list ℝ)
  (h_len : sticks.length = 15)
  (h_sets : ∃ (sets : list (ℝ × ℝ × ℝ)), 
    sets.length = 5 ∧ ∀ {a b c}, (a, b, c) ∈ sets → is_triangle a b c) :
  ∀ sets, sets.length = 5 ∧ ∀ {a b c}, (a, b, c) ∈ sets → is_triangle a b c → sets = h_sets :=
begin
  sorry
end

end no_other_way_five_triangles_l283_283703


namespace regular_polygon_sides_l283_283608

theorem regular_polygon_sides (theta : ℝ) (h : theta = 18) : 
  ∃ n : ℕ, 360 / theta = n ∧ n = 20 :=
by
  use 20
  constructor
  apply eq_of_div_eq_mul
  rw h
  norm_num
  sorry

end regular_polygon_sides_l283_283608


namespace find_a_even_function_l283_283985

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def given_function (a : ℝ) (x : ℝ) : ℝ :=
  x * Real.exp x / (Real.exp (a * x) - 1)

theorem find_a_even_function :
  (is_even_function (given_function 2)) → true :=
by
  intro h
  sorry

end find_a_even_function_l283_283985


namespace bases_with_final_digit_one_in_780_l283_283698

theorem bases_with_final_digit_one_in_780 : 
  let countBases : ℕ := (Finset.range 8).filter (λ b, 779 % (b + 2) = 1).card in
  countBases = 2 := 
by 
  sorry

end bases_with_final_digit_one_in_780_l283_283698


namespace sum_of_cubes_l283_283432

theorem sum_of_cubes (n : ℕ) (hn : n > 0) : 
  (∑ k in Finset.range n.succ, k^3) = (n * (n + 1) / 2) ^ 2 := 
sorry

end sum_of_cubes_l283_283432


namespace find_m_l283_283181

variable (m : ℝ)
variable (area : ℝ)
variable (dim1 : ℝ) := 2 * m + 9
variable (dim2 : ℝ) := m - 3

theorem find_m (h : dim1 * dim2 = area) (area = 55) : 
  m = (-3 + Real.sqrt 665) / 4 := 
sorry

end find_m_l283_283181


namespace even_function_implies_a_eq_2_l283_283851

theorem even_function_implies_a_eq_2 (a : ℝ) 
  (h : ∀ x : ℝ, f x = f (-x)) 
  (f : ℝ → ℝ := λ x, (x * exp x) / (exp (a * x) - 1)) : a = 2 :=
sorry

end even_function_implies_a_eq_2_l283_283851


namespace ratio_AD_AB_l283_283370

section TriangleProblem

variables {A B C D E : Type} [EuclideanGeometry A B C D E]

-- Assume angular conditions in triangle ABC
def angle_A : ℝ := 60
def angle_B : ℝ := 45

-- Line DE conditions within the triangle
def angle_ADE : ℝ := 75

-- Assuming the area division condition
axiom equal_area_division : 
  ∀ sABC sADE : ℝ, sABC = 2 * sADE → True

-- Prove the desired ratio
theorem ratio_AD_AB (AD AB : ℝ) : 
  angle_A = 60 → 
  angle_B = 45 → 
  angle_ADE = 75 → 
  equal_area_division (area_triangle ABC) (area_triangle ADE) → 
  AD / AB = 1 / (sqrt 6) :=
sorry

end TriangleProblem

end ratio_AD_AB_l283_283370


namespace find_a_if_even_function_l283_283950

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x * exp x / (exp (a * x) - 1)

theorem find_a_if_even_function :
  (∀ x : ℝ, x ≠ 0 → f a (-x) = f a x) → a = 2 :=
by
  sorry

end find_a_if_even_function_l283_283950


namespace factorize_cubic_l283_283250

theorem factorize_cubic : ∀ x : ℝ, x^3 - 4 * x = x * (x + 2) * (x - 2) :=
by
  sorry

end factorize_cubic_l283_283250


namespace distinct_values_of_expression_l283_283228

theorem distinct_values_of_expression : 
  let s := {1, 3, 5, 7, 9, 11, 13, 15}
  in 
  ∃ (μ : Nat), 
  (∀  (p q : ℕ), p ∈ s → q ∈ s →  
    ∃ x ∈ s, 
    x = pq + p + q
    ) → 
  μ = (number_of_distinct_outcomes_of_the_expression pq + p + q) sorry,
where number_of_distinct_outcomes is a hypothetical function (intended for demonstration purposes).

end distinct_values_of_expression_l283_283228


namespace even_function_implies_a_is_2_l283_283793

noncomputable def f (a x : ℝ) : ℝ := (x * Real.exp x) / (Real.exp (a * x) - 1)

theorem even_function_implies_a_is_2 (a : ℝ) 
  (h : ∀ x : ℝ, f a x = f a (-x)) : a = 2 := 
by
  sorry

end even_function_implies_a_is_2_l283_283793


namespace Owen_final_turtle_count_l283_283441

variable (Owen_turtles : ℕ) (Johanna_turtles : ℕ)

def final_turtles (Owen_turtles Johanna_turtles : ℕ) : ℕ :=
  let initial_Owen_turtles := Owen_turtles
  let initial_Johanna_turtles := Owen_turtles - 5
  let Owen_after_month := initial_Owen_turtles * 2
  let Johanna_after_losing_half := initial_Johanna_turtles / 2
  let Owen_after_donation := Owen_after_month + Johanna_after_losing_half
  Owen_after_donation

theorem Owen_final_turtle_count : final_turtles 21 (21 - 5) = 50 :=
by
  sorry

end Owen_final_turtle_count_l283_283441


namespace Sally_cards_l283_283461

theorem Sally_cards (initial_cards : ℕ) (cards_from_dan : ℕ) (cards_bought : ℕ) :
  initial_cards = 27 →
  cards_from_dan = 41 →
  cards_bought = 20 →
  initial_cards + cards_from_dan + cards_bought = 88 :=
by {
  intros,
  sorry
}

end Sally_cards_l283_283461


namespace cosine_of_largest_angle_l283_283345

theorem cosine_of_largest_angle 
  (a b c : ℝ) (k : ℝ) (h : k > 0)
  (h_ratio : a = 2 * k ∧ b = 4 * k ∧ c = 3 * k) :
  let cos_angle_B := (a^2 + c^2 - b^2) / (2 * a * c)
  in cos_angle_B = -1/4 :=
by 
  sorry

end cosine_of_largest_angle_l283_283345


namespace lambda_range_l283_283292

def seq_a (n : ℕ) : ℕ → ℕ
| 0       := 1
| (n + 1) := 3 * (seq_a n) + 1

def b_n (n : ℕ) : ℝ :=
  (n : ℝ) / ((3^n - 1) * 2^(n-2)) * (seq_a n / 2)

def T_n (n : ℕ) : ℝ :=
  (finset.range n).sum b_n

theorem lambda_range (λ : ℝ) :
  (∀ n : ℕ, n > 0 → 2^n * λ < 2^(n + 1) - 2) ↔ λ < 1 := by
  sorry

end lambda_range_l283_283292


namespace determine_number_of_men_l283_283158

noncomputable def number_of_men (M : ℕ) : Prop :=
  let W : ℝ := 1 in  -- Consider the amount of work as a constant
  W / (M * 20) = W / ((M - 4) * 25)

theorem determine_number_of_men : number_of_men 20 :=
by
  intros M
  unfold number_of_men
  sorry  -- Proof is to be constructed

end determine_number_of_men_l283_283158


namespace yard_length_l283_283009

-- Definition of the problem conditions
def num_trees : Nat := 11
def distance_between_trees : Nat := 15

-- Length of the yard is given by the product of (num_trees - 1) and distance_between_trees
theorem yard_length :
  (num_trees - 1) * distance_between_trees = 150 :=
by
  sorry

end yard_length_l283_283009


namespace solve_tangent_problem_l283_283268

noncomputable def problem_statement : Prop :=
  ∃ (n : ℤ), (-90 < n ∧ n < 90) ∧ (Real.tan (n * Real.pi / 180) = Real.tan (255 * Real.pi / 180)) ∧ (n = 75)

-- This is the statement of the problem we are proving.
theorem solve_tangent_problem : problem_statement :=
by
  sorry

end solve_tangent_problem_l283_283268


namespace part_I_part_II_1_part_II_2_l283_283758

theorem part_I (n : ℕ) (h : (3/(n * (n-1)/2) = 3/28)) : n = 8 := sorry

theorem part_II_1 (a : ℕ → ℤ) (h : (1 - 2 * (Polynomial.X : Polynomial ℤ))^8 = Polynomial.sum (a : ℕ → ℤ)) :
  a 3 = -448 := sorry

theorem part_II_2 (a : ℕ → ℤ) (h : (1 - 2 * (Polynomial.X : Polynomial ℤ))^8 = Polynomial.sum (a : ℕ → ℤ)) :
  (5.choose 2 * 4.choose 1 + 4.choose 3)/(9.choose 3) = 11/21 := sorry

end part_I_part_II_1_part_II_2_l283_283758


namespace compound_interest_correct_l283_283638

variable (a r x : ℕ) (y : ℝ)

noncomputable def compoundInterest (a : ℝ) (r : ℝ) (x : ℕ) : ℝ :=
  a * (1 + r) ^ x

theorem compound_interest_correct :
  compoundInterest 1000 0.0225 4 ≈ 1093.08 :=
by
  sorry

end compound_interest_correct_l283_283638


namespace no_alternative_way_to_construct_triangles_l283_283714

theorem no_alternative_way_to_construct_triangles (sticks : list ℕ) (h_len : sticks.length = 15)
  (h_set : ∃ (sets : list (list ℕ)), sets.length = 5 ∧ ∀ set ∈ sets, set.length = 3 
           ∧ (∀ (a b c : ℕ) (ha : a ∈ set) (hb : b ∈ set) (hc : c ∈ set), 
             (a + b > c ∧ a + c > b ∧ b + c > a))):
  ¬∃ (alt_sets : list (list ℕ)), alt_sets ≠ sets ∧ alt_sets.length = 5 
      ∧ ∀ set ∈ alt_sets, set.length = 3 
      ∧ (∀ (a b c : ℕ) (ha : a ∈ set) (hb : b ∈ set) (hc : c ∈ set),
        (a + b > c ∧ a + c > b ∧ b + c > a)) :=
sorry

end no_alternative_way_to_construct_triangles_l283_283714


namespace total_computers_needed_l283_283153

theorem total_computers_needed
    (initial_students : ℕ)
    (students_per_computer : ℕ)
    (additional_students : ℕ)
    (initial_computers : ℕ := initial_students / students_per_computer)
    (total_computers : ℕ := initial_computers + (additional_students / students_per_computer))
    (h1 : initial_students = 82)
    (h2 : students_per_computer = 2)
    (h3 : additional_students = 16) :
    total_computers = 49 :=
by
  -- The proof would normally go here
  sorry

end total_computers_needed_l283_283153


namespace locus_of_right_angle_vertex_l283_283296

variables {x y : ℝ}

/-- Given points M(-2,0) and N(2,0), if P(x,y) is the right-angled vertex of
  a right-angled triangle with MN as its hypotenuse, then the locus equation
  of P is given by x^2 + y^2 = 4 with the condition x ≠ ±2. -/
theorem locus_of_right_angle_vertex (h : x ≠ 2 ∧ x ≠ -2) :
  x^2 + y^2 = 4 :=
sorry

end locus_of_right_angle_vertex_l283_283296


namespace probability_sqrt_two_digit_less_than_seven_l283_283538

noncomputable def prob_sqrt_less_than_seven : ℚ := 
  let favorable := 39
  let total := 90
  favorable / total

theorem probability_sqrt_two_digit_less_than_seven : 
  prob_sqrt_less_than_seven = 13 / 30 := by
  sorry

end probability_sqrt_two_digit_less_than_seven_l283_283538


namespace limit_of_sequence_l283_283658

noncomputable def sequence (n : ℕ) : ℝ :=
  ( (13 * n + 3) / (13 * n - 10) ) ^ (n - 3)

theorem limit_of_sequence :
  ∃ l : ℝ, filter.tendsto (λ n : ℕ, sequence n) filter.at_top (nhds l) ∧ l = real.exp (16 / 13) :=
begin
  sorry
end

end limit_of_sequence_l283_283658


namespace factorize_cubic_l283_283247

theorem factorize_cubic : ∀ x : ℝ, x^3 - 4 * x = x * (x + 2) * (x - 2) :=
by
  sorry

end factorize_cubic_l283_283247


namespace probability_sqrt_two_digit_less_than_seven_l283_283541

noncomputable def prob_sqrt_less_than_seven : ℚ := 
  let favorable := 39
  let total := 90
  favorable / total

theorem probability_sqrt_two_digit_less_than_seven : 
  prob_sqrt_less_than_seven = 13 / 30 := by
  sorry

end probability_sqrt_two_digit_less_than_seven_l283_283541


namespace max_students_distribution_l283_283487

-- Define the four quantities
def pens : ℕ := 4261
def pencils : ℕ := 2677
def erasers : ℕ := 1759
def notebooks : ℕ := 1423

-- Prove that the greatest common divisor (GCD) of these four quantities is 1
theorem max_students_distribution : Nat.gcd (Nat.gcd (Nat.gcd pens pencils) erasers) notebooks = 1 :=
by
  sorry

end max_students_distribution_l283_283487


namespace AP_cannot_be_three_l283_283645

theorem AP_cannot_be_three (A B C P : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited P]
  (is_right_triangle_ABC : ∃ (angle_C : ℝ), angle_C = 90 ∧ is_triangle A B C)
  (AC_eq_3 : ∀ {AC}, AC = 3)
  (P_on_BC : ∀ {P B C}, is_point_on_line P B C) :
  ∀ (AP : ℝ), AP ≠ 3 :=
by
  sorry

end AP_cannot_be_three_l283_283645


namespace weight_of_replaced_girl_l283_283477

theorem weight_of_replaced_girl 
  (avg_weight_increase : 3)  -- The increase in avg weight of 8 girls 
  (weight_new_girl : 94)     -- The weight of the new girl 
  : ∃ W : ℝ, W = 70 :=       -- The weight of the replaced girl

by
  -- Definitions based on the condition
  let total_weight_increase := 8 * avg_weight_increase
  have h1 : total_weight_increase = 24 := by rfl  -- Calculate total weight increase
  let weight_replaced_girl := weight_new_girl - total_weight_increase
  have h2 : weight_replaced_girl = 70 := by rfl  -- Substitute and solve for W
  use weight_replaced_girl
  exact h2

end weight_of_replaced_girl_l283_283477


namespace joanna_has_8_dollars_l283_283022

def joanna_money (J B S : ℝ) :=
  B = 3 * J ∧ S = J / 2 ∧ J + B + S = 36 → J = 8

theorem joanna_has_8_dollars : ∃ J B S : ℝ, joanna_money J B S :=
by
  exists 8, 24, 4
  unfold joanna_money
  simp
  intros h1 h2 h3
  sorry

end joanna_has_8_dollars_l283_283022


namespace total_time_correct_l283_283424

-- Define the individual times
def driving_time_one_way : ℕ := 20
def attending_time : ℕ := 70

-- Define the total driving time as twice the one-way driving time
def total_driving_time : ℕ := driving_time_one_way * 2

-- Define the total time as the sum of total driving time and attending time
def total_time : ℕ := total_driving_time + attending_time

-- Prove that the total time is 110 minutes
theorem total_time_correct : total_time = 110 := by
  -- The proof is omitted, we're only interested in the statement format.
  sorry

end total_time_correct_l283_283424


namespace sally_cards_l283_283463

theorem sally_cards (initial_cards dan_cards bought_cards : ℕ) (h1 : initial_cards = 27) (h2 : dan_cards = 41) (h3 : bought_cards = 20) :
  initial_cards + dan_cards + bought_cards = 88 := by
  sorry

end sally_cards_l283_283463


namespace find_a_if_f_even_l283_283844

def f (a x : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem find_a_if_f_even (a : ℝ) (h : ∀ x : ℝ, x ≠ 0 → f a (-x) = f a x) : a = 2 :=
by 
  sorry

end find_a_if_f_even_l283_283844


namespace find_a1_to_a5_sum_l283_283759

theorem find_a1_to_a5_sum :
  let f (x : ℝ) := (1 - 2 * x) ^ 5
  ∃ (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ), 
  (f 0 = a_0) ∧ 
  (a_0 = 1) ∧ 
  (f 1 = a_0 + a_1 + a_2 + a_3 + a_4 + a_5) ∧ 
  (f 1 = (a_0 + a_1 + a_2 + a_3 + a_4 + a_5) = -1) ∧
  (a_1 + a_2 + a_3 + a_4 + a_5 = -2) :=
by
  sorry

end find_a1_to_a5_sum_l283_283759


namespace bug_lands_on_2_after_2023_jumps_l283_283062

-- Definitions of prime and composite
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ d, d ∣ n → d = 1 ∨ d = n)

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬ is_prime n

-- Mapping state transitions based on rules
def transition (p : ℕ) : ℕ :=
  if is_prime p then (p + 1) % 7
  else if is_composite p then (p + 3) % 7
  else (p + 1) % 7 -- Handle the case for 1 which is neither prime nor composite

-- Prove that after 2023 jumps starting from 7, the bug lands on 2
theorem bug_lands_on_2_after_2023_jumps :
  (Nat.iterate transition 2023 7) = 2 :=
begin
  sorry -- Proof to be filled in
end

end bug_lands_on_2_after_2023_jumps_l283_283062


namespace min_sum_value_l283_283286

open Nat

theorem min_sum_value (n : ℕ) (h_n : n ≥ 3) (x : Fin (n-1) → ℕ)
    (h1 : ∑ i in Finset.range (n-1), x i = n)
    (h2 : ∑ i in Finset.range (n-1), (i + 1) * x i = 2n - 2) :
    ∑ i in Finset.range (n-1), (i + 1) * (2n - (i + 1)) * x i = 3n^2 - 3n :=
by
  sorry

end min_sum_value_l283_283286


namespace train_speed_l283_283201

theorem train_speed
  (length_train : ℝ)
  (length_bridge : ℝ)
  (time_seconds : ℝ)
  (total_distance : length_train + length_bridge = 550)
  (speed_mps : total_distance / time_seconds = 12.5)
  (conversion_factor : 3.6)
  (speed_kph : speed_mps * conversion_factor = 45) :
  length_train = 410 →
  length_bridge = 140 →
  time_seconds = 44 →
  speed_kph = 45 :=
by
  intros h_train h_bridge h_time
  sorry

end train_speed_l283_283201


namespace determine_a_for_even_function_l283_283890

theorem determine_a_for_even_function :
  ∃ a : ℝ, (∀ x : ℝ, x ≠ 0 → (∃ f : ℝ → ℝ, 
  f x = x * exp x / (exp (a * x) - 1) ∧
  (∀ x : ℝ, f (-x) = f x))) → a = 2 :=
by
  sorry

end determine_a_for_even_function_l283_283890


namespace new_unemployment_rate_is_66_percent_l283_283475

theorem new_unemployment_rate_is_66_percent
  (initial_unemployment_rate : ℝ)
  (initial_employment_rate : ℝ)
  (u_increases_by_10_percent : initial_unemployment_rate * 1.1 = new_unemployment_rate)
  (e_decreases_by_15_percent : initial_employment_rate * 0.85 = new_employment_rate)
  (sum_is_100_percent : initial_unemployment_rate + initial_employment_rate = 100) :
  new_unemployment_rate = 66 :=
by
  sorry

end new_unemployment_rate_is_66_percent_l283_283475


namespace reflected_parabola_equation_l283_283492

-- Define the given parabola equation
def parabola (x : ℝ) : ℝ := x^2

-- Define the line of reflection
def reflection_line (x : ℝ) : ℝ := x + 2

-- The reflected equation statement to be proved
theorem reflected_parabola_equation (x y : ℝ) :
  (parabola x = y) ∧ (reflection_line x = y) →
  (∃ y' x', x = y'^2 - 4 * y' + 2 ∧ y = x' + 2 ∧ x' = y - 2) :=
sorry

end reflected_parabola_equation_l283_283492


namespace necessary_and_sufficient_condition_l283_283404

-- Definitions based on the conditions
def purely_imaginary (z : ℂ) : Prop := z.re = 0

-- Lean statement for the problem
theorem necessary_and_sufficient_condition (m : ℝ) (i : ℂ) (h_i : i = complex.I) : 
  (purely_imaginary (complex.of_real (m^2 - m) + m * i)) ↔ (m = 1) :=
by { sorry }

end necessary_and_sufficient_condition_l283_283404


namespace spencer_total_distance_l283_283394

def distances : ℝ := 0.3 + 0.1 + 0.4

theorem spencer_total_distance :
  distances = 0.8 :=
sorry

end spencer_total_distance_l283_283394


namespace alice_distance_correct_l283_283207

noncomputable def alice_distance_from_start (north_meters east1_feet south_meters_and_feet east2_feet : ℕ) : ℝ :=
  let north_feet := 3.281 * north_meters
  let south_feet := 3.281 * (south_meters_and_feet - north_meters)
  real.sqrt (south_feet ^ 2 + (east1_feet + east2_feet) ^ 2)

theorem alice_distance_correct :
  alice_distance_from_start 12 40 (12 + 40 / 3.281) 16 = 68.836 :=
by
  -- Calculation steps skipped as instructed
  sorry

end alice_distance_correct_l283_283207


namespace fred_change_l283_283433

theorem fred_change : 
  let ticket_price := 8.25
  let number_of_tickets := 4
  let movie_borrowed_cost := 9.50
  let paid_amount := 50
  let total_cost := (number_of_tickets * ticket_price) + movie_borrowed_cost
  let change := paid_amount - total_cost
  change = 7.50 :=
by
  let ticket_price := 8.25
  let number_of_tickets := 4
  let movie_borrowed_cost := 9.50
  let paid_amount := 50
  let total_cost := (number_of_tickets * ticket_price) + movie_borrowed_cost
  let change := paid_amount - total_cost
  show change = 7.50 from sorry

end fred_change_l283_283433


namespace repeating_decimal_product_l283_283218

theorem repeating_decimal_product :
  let x := 2 / 3 in
  x * (7 / 3) = 14 / 9 := 
by
  let x := 2 / 3
  have fraction_eq : x = 2 / 3 := rfl
  have product_eq : x * (7 / 3) = (2 * 7) / (3 * 3) := by
    rw fraction_eq
    norm_num
  exact product_eq

end repeating_decimal_product_l283_283218


namespace smallest_of_three_consecutive_odd_numbers_l283_283119

theorem smallest_of_three_consecutive_odd_numbers (x : ℤ) (h : x + (x + 2) + (x + 4) = 69) : x = 21 :=
sorry

end smallest_of_three_consecutive_odd_numbers_l283_283119


namespace installation_cost_per_hour_is_correct_l283_283131

-- Definitions based on the conditions
def land_cost_per_acre : ℝ := 20
def house_cost : ℝ := 120000
def cow_cost_per_cow : ℝ := 1000
def chicken_cost_per_chicken : ℝ := 5
def solar_panel_equipment_cost : ℝ := 6000
def installation_hours : ℝ := 6
def total_expenditure : ℝ := 147700

-- Additional definitions to compute total cost before solar panels
def land_cost (acres : ℝ) : ℝ := acres * land_cost_per_acre
def total_land_cost := land_cost 30
def cow_cost (cows : ℝ) : ℝ := cows * cow_cost_per_cow
def total_cow_cost := cow_cost 20
def chicken_cost (chickens : ℝ) : ℝ := chickens * chicken_cost_per_chicken
def total_chicken_cost := chicken_cost 100

def total_cost_before_solar_panels : ℝ :=
  total_land_cost + house_cost + total_cow_cost + total_chicken_cost

def cost_of_solar_panels : ℝ :=
  total_expenditure - total_cost_before_solar_panels

def installation_cost : ℝ :=
  cost_of_solar_panels - solar_panel_equipment_cost

def installation_cost_per_hour : ℝ :=
  installation_cost / installation_hours

-- Proof that the installation cost per hour is $3433.33
theorem installation_cost_per_hour_is_correct : 
  installation_cost_per_hour = 3433.33 := 
by 
  sorry

end installation_cost_per_hour_is_correct_l283_283131


namespace no_divisors_between_2_and_100_l283_283417

theorem no_divisors_between_2_and_100 (n : ℕ) (h : ∀ k : ℕ, k ∈ finset.range 100 \ {0} → (∑ i in finset.range (n+1), i^k  % n = 0)) :
  ∀ p, 2 ≤ p ∧ p ≤ 100 → ¬ p ∣ n :=
begin
  sorry
end

end no_divisors_between_2_and_100_l283_283417


namespace domain_of_f_zeros_of_f_l283_283314

def log_a (a : ℝ) (x : ℝ) : ℝ := sorry -- Assume definition of logarithm base 'a'.

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log_a a (2 - x)

theorem domain_of_f (a : ℝ) : ∀ x : ℝ, 2 - x > 0 ↔ x < 2 :=
by
  sorry

theorem zeros_of_f (a : ℝ) : f a 1 = 0 :=
by
  sorry

end domain_of_f_zeros_of_f_l283_283314


namespace factorize_x_l283_283258

theorem factorize_x^3_minus_4x (x : ℝ) : x^3 - 4 * x = x * (x + 2) * (x - 2) := 
sorry

end factorize_x_l283_283258


namespace f_is_even_iff_a_is_2_l283_283807

def f (a : ℝ) (x : ℝ) : ℝ := x * Real.exp x / (Real.exp (a * x) - 1)

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem f_is_even_iff_a_is_2 : (∀ x, f 2 (-x) = f 2 x) ↔ ∀ a, a = 2 := 
sorry

end f_is_even_iff_a_is_2_l283_283807


namespace derivative_poly_1_derivative_poly_2_derivative_frac_3_l283_283693

-- Proof Problem 1: Proving the derivative of 2x^3 - 3x^2 - 4 is 6x^2 - 6x
theorem derivative_poly_1 (x : ℝ) : 
  (deriv (λ x : ℝ, 2*x^3 - 3*x^2 - 4) x) = 6*x^2 - 6*x := 
by
  sorry

-- Proof Problem 2: Proving the derivative of x * ln x is ln x + 1
theorem derivative_poly_2 (x : ℝ) (h : x > 0) : 
  (deriv (λ x : ℝ, x * log x) x) = log x + 1 := 
by
  sorry

-- Proof Problem 3: Proving the derivative of cos x / x is - (x * sin x + cos x) / x^2
theorem derivative_frac_3 (x : ℝ) (h : x ≠ 0) : 
  (deriv (λ x : ℝ, cos x / x) x) = - (x * sin x + cos x) / x^2 := 
by
  sorry

end derivative_poly_1_derivative_poly_2_derivative_frac_3_l283_283693


namespace am_plus_kl_plus_cn_eq_ac_l283_283572

variables {A B C I M N K L : Type*}
variables [Incenter A B C I] [Line_through_point I A B I M] [Line_through_point I B C I N]
variables (acute_triangle : Acute_triangle B M N)
variables (angle_eq_IL_IA : ∠ I L A = ∠ I M B) (angle_eq_IK_IN : ∠ I K C = ∠ I N B)

theorem am_plus_kl_plus_cn_eq_ac (h1 : Incenter A B C I) 
                                  (h2 : Line_through_point I A B I M) 
                                  (h3 : Line_through_point I B C I N) 
                                  (h4 : Acute_triangle B M N) 
                                  (h5 : ∠ I L A = ∠ I M B) 
                                  (h6 : ∠ I K C = ∠ I N B) :
  AM + KL + CN = AC := 
sorry

end am_plus_kl_plus_cn_eq_ac_l283_283572


namespace unique_triangle_assembly_l283_283745

def triangle_inequality (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

def is_valid_triangle_set (sticks : List (ℕ × ℕ × ℕ)) : Prop :=
  sticks.all (λ stick_lengths, triangle_inequality stick_lengths.1 stick_lengths.2 stick_lengths.3)

theorem unique_triangle_assembly (sticks : List ℕ) (h_len : sticks.length = 15)
  (h_sticks : sticks = [2, 3, 4, 20, 30, 40, 200, 300, 400, 2000, 3000, 4000, 20000, 30000, 40000]) :
  ¬ ∃ (other_config : List (ℕ × ℕ × ℕ)),
    is_valid_triangle_set other_config ∧
    other_config ≠ [(2, 3, 4), (20, 30, 40), (200, 300, 400), (2000, 3000, 4000), (20000, 30000, 40000)] :=
begin
  sorry
end

end unique_triangle_assembly_l283_283745


namespace probability_blue_higher_than_yellow_l283_283580

theorem probability_blue_higher_than_yellow :
  (∀ (n : ℕ) (prob_event : ℝ → ℝ),
     let blue_probability := λ k, 3^(-k : ℝ),
         yellow_probability := λ k, if prob_event = 1 then 2 * 3^(-k : ℝ) else 3^(-k : ℝ),
         event_probability := λ k, 1 / 2 in
     (2 * ∑ k, 9^(-k) + 3 * ∑ k, 9^(-k) =
     2 * (1 / 1 - (1 / 9)) + 3 * (1 / 1 - (1 / 9)) / 2 - ∑ (k : ℕ), blue_probability k * yellow_probability k) ≤ 1 / 8) :=
sorry

end probability_blue_higher_than_yellow_l283_283580


namespace dislikes_TV_and_books_l283_283002

-- The problem conditions
def total_people : ℕ := 800
def percent_dislikes_TV : ℚ := 25 / 100
def percent_dislikes_both : ℚ := 15 / 100

-- The expected answer
def expected_dislikes_TV_and_books : ℕ := 30

-- The proof problem statement
theorem dislikes_TV_and_books : 
  (total_people * percent_dislikes_TV) * percent_dislikes_both = expected_dislikes_TV_and_books := by 
  sorry

end dislikes_TV_and_books_l283_283002


namespace gcd_prime_factorization_lcm_prime_factorization_l283_283060

open Nat

theorem gcd_prime_factorization (n m : ℕ) (p : ℕ → ℕ) (alpha beta : ℕ → ℕ) (k : ℕ) 
  (hn : n = ∏ i in range k, p i ^ alpha i) (hm : m = ∏ i in range k, p i ^ beta i):
  gcd n m = ∏ i in range k, p i ^ min (alpha i) (beta i) :=
sorry

theorem lcm_prime_factorization (n m : ℕ) (p : ℕ → ℕ) (alpha beta : ℕ → ℕ) (k : ℕ) 
  (hn : n = ∏ i in range k, p i ^ alpha i) (hm : m = ∏ i in range k, p i ^ beta i):
  lcm n m = ∏ i in range k, p i ^ max (alpha i) (beta i) :=
sorry

end gcd_prime_factorization_lcm_prime_factorization_l283_283060


namespace largest_c_real_positive_l283_283299

theorem largest_c_real_positive (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ c : ℝ, (c = 2) ∧ 
    ∀ x > 0, 
    c ≤ max (a * x + 1 / (a * x)) (b * x + 1 / (b * x)) :=
by
  existsi (2 : ℝ)
  split
  . rfl
  . intros
    sorry

end largest_c_real_positive_l283_283299


namespace count_remaining_integers_l283_283226

def setT : Set ℕ := { x | x ≤ 100 ∧ x > 0 }

def isMultipleOf (m n : ℕ) : Prop := n % m = 0

noncomputable def remainingIntegersInT : ℕ :=
  setT.card - 
  (setT.filter (isMultipleOf 4)).card - 
  (setT.filter (isMultipleOf 5)).card + 
  (setT.filter (λ x, isMultipleOf 4 x ∧ isMultipleOf 5 x)).card

theorem count_remaining_integers :
  remainingIntegersInT = 60 :=
by
  sorry

end count_remaining_integers_l283_283226


namespace modulus_of_z_l283_283338

-- Definitions and theorems
theorem modulus_of_z (z : ℂ) (h : z * (1 + I) = I) : complex.abs z = real.sqrt 2 / 2 :=
by
  sorry

end modulus_of_z_l283_283338


namespace determine_a_for_even_function_l283_283879

theorem determine_a_for_even_function :
  ∃ a : ℝ, (∀ x : ℝ, x ≠ 0 → (∃ f : ℝ → ℝ, 
  f x = x * exp x / (exp (a * x) - 1) ∧
  (∀ x : ℝ, f (-x) = f x))) → a = 2 :=
by
  sorry

end determine_a_for_even_function_l283_283879


namespace parity_sum_matches_parity_of_M_l283_283571

theorem parity_sum_matches_parity_of_M (N M : ℕ) (even_numbers odd_numbers : ℕ → ℤ)
  (hn : ∀ i, i < N → even_numbers i % 2 = 0)
  (hm : ∀ i, i < M → odd_numbers i % 2 ≠ 0) : 
  (N + M) % 2 = M % 2 := 
sorry

end parity_sum_matches_parity_of_M_l283_283571


namespace jan_total_amount_paid_l283_283391

-- Define the conditions
def dozen := 12
def roses_in_dozen := 5 * dozen
def cost_per_rose := 6
def discount := 0.8

-- Define the expected result
def total_amount_paid := 288

-- Theorem statement to prove the total amount paid
theorem jan_total_amount_paid :
  roses_in_dozen * cost_per_rose * discount = total_amount_paid := 
by
  sorry

end jan_total_amount_paid_l283_283391


namespace question_1_question_2_l283_283191

variable (a : ℝ) (x : ℝ)

theorem question_1 (h1 : 2 * a - 3 + 5 - a = 0) : a = -2 ∧ x = 49 :=
by
  sorry
  
theorem question_2 (h1 : a = -2) (h2 : x = 49) : sqrt (x + 12 * a) = 5 ∨ sqrt (x + 12 * a) = -5 :=
by
  sorry

end question_1_question_2_l283_283191


namespace trade_in_value_l283_283393

theorem trade_in_value (x : ℝ) (movies : ℕ) (dvd_cost total_cost : ℝ) (h1 : movies = 100) (h2 : dvd_cost = 10) (h3 : total_cost = 800) :
  100 * x + total_cost = movies * dvd_cost → x = 2 :=
by
  intros h
  rw [h1, h2, h3] at h
  linarith

end trade_in_value_l283_283393


namespace unique_triangle_assembly_l283_283748

def triangle_inequality (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

def is_valid_triangle_set (sticks : List (ℕ × ℕ × ℕ)) : Prop :=
  sticks.all (λ stick_lengths, triangle_inequality stick_lengths.1 stick_lengths.2 stick_lengths.3)

theorem unique_triangle_assembly (sticks : List ℕ) (h_len : sticks.length = 15)
  (h_sticks : sticks = [2, 3, 4, 20, 30, 40, 200, 300, 400, 2000, 3000, 4000, 20000, 30000, 40000]) :
  ¬ ∃ (other_config : List (ℕ × ℕ × ℕ)),
    is_valid_triangle_set other_config ∧
    other_config ≠ [(2, 3, 4), (20, 30, 40), (200, 300, 400), (2000, 3000, 4000), (20000, 30000, 40000)] :=
begin
  sorry
end

end unique_triangle_assembly_l283_283748


namespace final_score_eq_l283_283438

variable (initial_score : ℝ)
def deduction_lost_answer : ℝ := 1
def deduction_error : ℝ := 0.5
def deduction_checks : ℝ := 0

def total_deduction : ℝ := deduction_lost_answer + deduction_error + deduction_checks

theorem final_score_eq : final_score = initial_score - total_deduction := by
  sorry

end final_score_eq_l283_283438


namespace find_a_if_f_even_l283_283833

def f (a x : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem find_a_if_f_even (a : ℝ) (h : ∀ x : ℝ, x ≠ 0 → f a (-x) = f a x) : a = 2 :=
by 
  sorry

end find_a_if_f_even_l283_283833


namespace even_function_implies_a_eq_2_l283_283912

def f (x : ℝ) (a : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2 (a : ℝ) (h_even : ∀ x, f (-x) a = f x a) : a = 2 :=
by
  intro x
  sorry

end even_function_implies_a_eq_2_l283_283912


namespace snail_wins_the_race_l283_283595

noncomputable def rabbit_and_snail_race : Prop :=
  ∃ (s r : ℝ) (t₁ t₂ t₃ t₄ T : ℝ), 
    T = 200 ∧
    s * T = 200 ∧
    r * (t₁ + t₃) = 200 ∧
    t₁ + t₂ + t₃ + t₄ = T ∧
    s = 1 ∧
    r = 10

theorem snail_wins_the_race : rabbit_and_snail_race → True := 
  begin
    sorry,
  end

end snail_wins_the_race_l283_283595


namespace determine_a_for_even_function_l283_283880

theorem determine_a_for_even_function :
  ∃ a : ℝ, (∀ x : ℝ, x ≠ 0 → (∃ f : ℝ → ℝ, 
  f x = x * exp x / (exp (a * x) - 1) ∧
  (∀ x : ℝ, f (-x) = f x))) → a = 2 :=
by
  sorry

end determine_a_for_even_function_l283_283880


namespace no_alternative_way_to_construct_triangles_l283_283721

-- Define the triangle inequality condition
def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the set of stick lengths for each required triangle
def valid_triangle_sets (stick_set : Set (Set ℕ)) : Prop :=
  stick_set = { {2, 3, 4}, {20, 30, 40}, {200, 300, 400}, {2000, 3000, 4000}, {20000, 30000, 40000} }

-- Define the problem statement
theorem no_alternative_way_to_construct_triangles (s : Set ℕ) (h : s.card = 15) :
  (∀ t ∈ (s.powerset.filter (λ t, t.card = 3)), triangle_inequality t.some (finset_some_spec t) ∧ (∃ sets : Set (Set ℕ), sets.card = 5 ∧ ∀ u ∈ sets, u.card = 3 ∧ triangle_inequality u.some (finset_some_spec u)))
  → valid_triangle_sets (s.powerset.filter (λ t, t.card = 3)) := 
sorry

end no_alternative_way_to_construct_triangles_l283_283721


namespace find_a_of_even_function_l283_283775

noncomputable def f (a x : ℝ) : ℝ := (x * exp x) / (exp (a * x) - 1)

theorem find_a_of_even_function (hf : ∀ x : ℝ, f a (-x) = f a x) : a = 2 :=
by {
    sorry
}

end find_a_of_even_function_l283_283775


namespace domain_of_g_is_infinite_l283_283667

def g : ℤ → ℤ
| c => if c % 2 = 1 then 3 * c + 2 else c / 2

theorem domain_of_g_is_infinite :
  (∀ c : ℤ, ∃ d : ℤ, g c = d) →
  ∀ n : ℕ, ∃ m : ℕ, m > n ∧ (∃ c : ℤ, nat_abs c = m ∧ c ∈ domain_of g) :=
by
  sorry

end domain_of_g_is_infinite_l283_283667


namespace Q_lies_on_AD_l283_283031

variables {A B C D P Q : Type*}
variables [is_midpoint D B C]
variables [lies_on P A D]
variables [is_angle_bisector (⟨A, B, Q, P⟩) (⟨A, C, Q, P⟩)]
variables [perpendicular B Q C Q]

theorem Q_lies_on_AD :
  lies_on Q A D :=
sorry

end Q_lies_on_AD_l283_283031


namespace yangtze_in_scientific_notation_l283_283102

theorem yangtze_in_scientific_notation (length : ℤ) (h : length = 6300000) : 
  length = 6.3 * 10^6 :=
by
  sorry

end yangtze_in_scientific_notation_l283_283102


namespace find_m_l283_283310

theorem find_m 
  (x1 x2 : ℝ) 
  (m : ℝ)
  (h1 : x1 + x2 = m)
  (h2 : x1 * x2 = 2 * m - 1)
  (h3 : x1^2 + x2^2 = 7) : 
  m = 5 :=
by
  sorry

end find_m_l283_283310


namespace unique_point_P_l283_283005

noncomputable def isParallelogram {P Q R S : Type*} [EuclideanGeometry.Polygon A B C D] 
(point P : Point) : Prop :=
  P = center(Point A, Point B, Point C, Point D) /\
  area(Triangle A B P) = area(Triangle B C P) = area(Triangle C D P) = area(Triangle D A P)

theorem unique_point_P (A B C D : Point) (h : Quadrilateral A B C D) :
  (∀ P : Point, isParallelogram A B C D P → P = midpoint(Diagonal A C) ∧ midpoint(Diagonal B D) ∧ area(Triangle A B P) = area(Triangle B C P) = area(Triangle C D P) = area(Triangle D A P) -> (A B || C D)) -> ∃ ! (P : Point), isParallelogram A B C D P :=
  sorry

end unique_point_P_l283_283005


namespace distance_difference_l283_283205

-- Definition of speeds and time
def speed_alberto : ℕ := 16
def speed_clara : ℕ := 12
def time_hours : ℕ := 5

-- Distance calculation functions
def distance (speed time : ℕ) : ℕ := speed * time

-- Main theorem statement
theorem distance_difference : 
  distance speed_alberto time_hours - distance speed_clara time_hours = 20 :=
by
  sorry

end distance_difference_l283_283205


namespace find_f_2012_l283_283101

variable (f : ℕ → ℝ)

axiom f_one : f 1 = 3997
axiom recurrence : ∀ x, f x - f (x + 1) = 1

theorem find_f_2012 : f 2012 = 1986 :=
by
  -- Skipping proof
  sorry

end find_f_2012_l283_283101


namespace reseating_ways_l283_283675

/-- Eleven women are seated in 11 chairs in a line with one standing initially.
They reseat themselves such that:
1. The woman who was standing sits in a spot that is occupied or adjacent to someone who was previously seated.
2. The others sit in their original seat or an adjacent one.
Prove that the number of ways they can be reseated is 610. -/
theorem reseating_ways : 
  ∃ (T : ℕ → ℕ), 
  (T 2 = 2) ∧ (∀ n, n ≥ 3 → T n = 2 * (S (n - 1) + T (n - 1))) 
  ∧ (T 11 = 610) := 
sorry

end reseating_ways_l283_283675


namespace sin_half_angle_inequality_l283_283014

theorem sin_half_angle_inequality
  (A B C : ℝ)
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (h_sum : A + B + C = π)
  : sin (A / 2) * sin (B / 2) * sin (C / 2) ≤ 1 / 8 := 
sorry

end sin_half_angle_inequality_l283_283014


namespace gradient_in_cylindrical_coords_l283_283657

variable {ρ ϕ z : ℝ}

def scalar_field (ρ ϕ z : ℝ) : ℝ := ρ + z * cos ϕ

def cylindrical_gradient (ρ ϕ z : ℝ) : (ℝ × ℝ × ℝ) :=
  (1, -z * sin ϕ / ρ, cos ϕ)

theorem gradient_in_cylindrical_coords :
  ∀ (ρ ϕ z : ℝ), (∇ (scalar_field ρ ϕ z)) = (cylindrical_gradient ρ ϕ z) :=
by
  sorry

end gradient_in_cylindrical_coords_l283_283657


namespace g_bounded_by_one_l283_283560

open Real

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom ax1 : ∀ x y : ℝ, f(x + y) + f(x - y) = 2 * f(x) * g(y)
axiom ax2 : ∀ x : ℝ, |f x| ≤ 1
axiom ax3 : ∃ x : ℝ, f x ≠ 0

theorem g_bounded_by_one : ∀ x : ℝ, |g x| ≤ 1 :=
by
  sorry

end g_bounded_by_one_l283_283560


namespace no_alternate_way_to_form_triangles_l283_283727

noncomputable def canFormTriangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def validSetOfTriples : List (ℕ × ℕ × ℕ) :=
  [(2, 3, 4), (20, 30, 40), (200, 300, 400), (2000, 3000, 4000), (20000, 30000, 40000)]

theorem no_alternate_way_to_form_triangles (stickSet : List (ℕ × ℕ × ℕ)) :
  (∀ (t : ℕ × ℕ × ℕ), t ∈ stickSet → canFormTriangle t.1 t.2 t.3) →
  length stickSet = 5 →
  stickSet = validSetOfTriples ->
  (∀ alternateSet : List (ℕ × ℕ × ℕ),
    ∀ (t : ℕ × ℕ × ℕ), t ∈ alternateSet → canFormTriangle t.1 t.2 t.3 →
    length alternateSet = 5 →
    duplicate_stick_alternateSet = duplicate_stick_stickSet) :=
sorry

end no_alternate_way_to_form_triangles_l283_283727


namespace factorize_cubic_l283_283249

theorem factorize_cubic : ∀ x : ℝ, x^3 - 4 * x = x * (x + 2) * (x - 2) :=
by
  sorry

end factorize_cubic_l283_283249


namespace c_is_perfect_square_l283_283297

theorem c_is_perfect_square (a b c : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c) (h : c = a + b / a - 1 / b) : ∃ m : ℕ, c = m * m :=
by
  sorry

end c_is_perfect_square_l283_283297


namespace prob_at_least_one_hit_correct_prob_exactly_two_hits_correct_l283_283079

-- Define the probabilities that A, B, and C hit the target
def prob_A := 0.7
def prob_B := 0.6
def prob_C := 0.5

-- Define the probabilities that A, B, and C miss the target
def miss_A := 1 - prob_A
def miss_B := 1 - prob_B
def miss_C := 1 - prob_C

-- Probability that no one hits the target
def prob_no_hits := miss_A * miss_B * miss_C

-- Probability that at least one person hits the target
def prob_at_least_one_hit := 1 - prob_no_hits

-- Probabilities for the cases where exactly two people hit the target:
def prob_A_B_hits := prob_A * prob_B * miss_C
def prob_A_C_hits := prob_A * miss_B * prob_C
def prob_B_C_hits := miss_A * prob_B * prob_C

-- Probability that exactly two people hit the target
def prob_exactly_two_hits := prob_A_B_hits + prob_A_C_hits + prob_B_C_hits

-- Theorem statement to prove the probabilities match given conditions
theorem prob_at_least_one_hit_correct : prob_at_least_one_hit = 0.94 := by
  sorry

theorem prob_exactly_two_hits_correct : prob_exactly_two_hits = 0.44 := by
  sorry

end prob_at_least_one_hit_correct_prob_exactly_two_hits_correct_l283_283079


namespace find_a_if_even_function_l283_283944

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x * exp x / (exp (a * x) - 1)

theorem find_a_if_even_function :
  (∀ x : ℝ, x ≠ 0 → f a (-x) = f a x) → a = 2 :=
by
  sorry

end find_a_if_even_function_l283_283944


namespace base_representation_final_digit_final_digit_base_625_2_l283_283669

   theorem base_representation_final_digit (b : ℕ) (h₁ : 3 ≤ b ∧ b ≤ 10) (h₂ : 625 % b = 2) : b = 7 :=
   by {
       sorry
   }

   theorem final_digit_base_625_2 : ∃! b, (3 ≤ b ∧ b ≤ 10) ∧ 625 % b = 2 :=
   by {
       use 7,
       split,
       { constructor,
         { exact ⟨le_of_eq rfl, le_of_eq rfl⟩, sorry },
         { intro h, exact 7 }
       },
       { intro b, intro h,
         have h₁ := base_representation_final_digit b,
         exact h₁ h.1 h.2
       },
       sorry
   }
   
end base_representation_final_digit_final_digit_base_625_2_l283_283669


namespace regular_polygon_sides_l283_283607

theorem regular_polygon_sides (n : ℕ) (h : 1 < n) (exterior_angle : ℝ) (h_ext : exterior_angle = 18) :
  n * exterior_angle = 360 → n = 20 :=
by 
  sorry

end regular_polygon_sides_l283_283607


namespace find_a_l283_283990

noncomputable def f (x a : ℝ) : ℝ := (x * (Real.exp x)) / (Real.exp (a * x) - 1)

theorem find_a (a : ℝ) (h : ∀ x, f x a = f (-x) a) : a = 2 :=
begin
  sorry
end

end find_a_l283_283990


namespace solve_for_z_l283_283263

theorem solve_for_z (z : ℂ) : 
  z * z = -121 - 110 * complex.i ↔ 
  z = 3 * real.sqrt 2 - (55 * real.sqrt 2 / 6) * complex.i ∨ 
  z = -3 * real.sqrt 2 + (55 * real.sqrt 2 / 6) * complex.i :=
by
  sorry

end solve_for_z_l283_283263


namespace students_count_l283_283559

theorem students_count :
  ∀ (sets marbles_per_set marbles_per_student total_students : ℕ),
    sets = 3 →
    marbles_per_set = 32 →
    marbles_per_student = 4 →
    total_students = (sets * marbles_per_set) / marbles_per_student →
    total_students = 24 :=
by
  intros sets marbles_per_set marbles_per_student total_students
  intros h_sets h_marbles_per_set h_marbles_per_student h_total_students
  rw [h_sets, h_marbles_per_set, h_marbles_per_student] at h_total_students
  exact h_total_students

end students_count_l283_283559


namespace abs_diff_of_two_numbers_l283_283120

theorem abs_diff_of_two_numbers (x y : ℝ) (h1 : x + y = 36) (h2 : x * y = 320) : |x - y| = 4 := 
by
  sorry

end abs_diff_of_two_numbers_l283_283120


namespace g_7_sub_g_3_div_g_4_eq_2_5_l283_283096

variable (g : ℝ → ℝ)
variable (h : ∀ c d : ℝ, c^2 * g(d) = d^2 * g(c))
variable (hg4 : g(4) ≠ 0)

theorem g_7_sub_g_3_div_g_4_eq_2_5 : (g 7 - g 3) / g 4 = 2.5 :=
by
  sorry

end g_7_sub_g_3_div_g_4_eq_2_5_l283_283096


namespace workshop_manager_assignment_l283_283354

variables {X Y : Type} [fintype X] [fintype Y] (k : ℕ)

-- Conditions
axiom workers_operate_k_machines (x : X) : fintype.card {y : Y // (x, y) ∈ relation} = k
axiom machines_operated_by_k_workers (y : Y) : fintype.card {x : X // (x, y) ∈ relation} = k

-- Problem statement
theorem workshop_manager_assignment :
  (∃ f : X → Y, function.bijective f ∧ (∀ x, (x, f x) ∈ relation)) :=
sorry

end workshop_manager_assignment_l283_283354


namespace case_a_case_b_l283_283519

noncomputable def sequence (x₁ : ℝ) : ℕ → ℝ
| 0       := x₁
| (n + 1) := (sequence n) ^ 2 - (sequence n) + 1

def sum_series (x₁ : ℝ) := ∑' n, 1 / (sequence x₁ n)

theorem case_a (x₁ : ℝ) (h : x₁ = 0.5) : ¬convergent (sum_series x₁) := sorry

theorem case_b (x₁ : ℝ) (h : x₁ = 2) : sum_series x₁ = 1 := sorry

end case_a_case_b_l283_283519


namespace tickets_won_whack_a_mole_l283_283152

variable (t : ℕ)

def tickets_from_skee_ball : ℕ := 9
def cost_per_candy : ℕ := 6
def number_of_candies : ℕ := 7
def total_tickets_needed : ℕ := cost_per_candy * number_of_candies

theorem tickets_won_whack_a_mole : t + tickets_from_skee_ball = total_tickets_needed → t = 33 :=
by
  intro h
  have h1 : total_tickets_needed = 42 := by sorry
  have h2 : tickets_from_skee_ball = 9 := by rfl
  rw [h2, h1] at h
  sorry

end tickets_won_whack_a_mole_l283_283152


namespace point_further_from_cheese_l283_283189

def coordinates_cheese : ℝ × ℝ := (15, 12)
def coordinates_mouse : ℝ × ℝ := (3, -3)
def mouse_path (x : ℝ) : ℝ := -4 * x + 9

def perpendicular_slope (m : ℝ) : ℝ := -1 / m
def slope_path : ℝ := -4
def perp_slope_path : ℝ := perpendicular_slope slope_path

noncomputable def line_through_cheese (x : ℝ) : ℝ := (1 / 4) * (x - 15) + 12

def point_of_interest : ℝ × ℝ := (-3 / 5, 153 / 20)
def sum_point_of_interest : ℝ := -3 / 5 + 153 / 20

theorem point_further_from_cheese :
  let a := -3 / 5 in
  let b := 153 / 20 in
  a + b = 7.05 :=
by
  sorry

end point_further_from_cheese_l283_283189


namespace range_of_a_for_extrema_l283_283337

noncomputable def f (a x : ℝ) : ℝ := x^3 + 3 * a * x^2 + 3 * (a + 2) * x + 1

theorem range_of_a_for_extrema (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ is_max (f a) x₁ ∧ is_min (f a) x₂) ↔ (a > 2 ∨ a < -1) :=
sorry

end range_of_a_for_extrema_l283_283337


namespace repeating_decimal_to_fraction_l283_283677

theorem repeating_decimal_to_fraction :
  (7.036).repeat == 781 / 111 :=
by
  sorry

end repeating_decimal_to_fraction_l283_283677


namespace number_of_valid_x_l283_283193

theorem number_of_valid_x (x : ℕ) : 
  ((x + 3) * (x - 3) * (x ^ 2 + 9) < 500) ∧ (x - 3 > 0) ↔ x = 4 :=
sorry

end number_of_valid_x_l283_283193


namespace exists_odd_Hx_l283_283293

variable {M : Type} [Finite M] {H : M → Set M}

-- Define the conditions
axiom odd_cardinality_of_M : ↑(card M) % 2 = 1
axiom unique_correspondence : ∀ x : M, H x ⊆ M
axiom self_containment : ∀ x : M, x ∈ H x
axiom mutual_inclusion : ∀ x y : M, y ∈ H x ↔ x ∈ H y

theorem exists_odd_Hx : ∃ x : M, ↑(card (H x)) % 2 = 1 := sorry

end exists_odd_Hx_l283_283293


namespace common_ratio_geometric_sequence_l283_283117

theorem common_ratio_geometric_sequence (a1 q : ℝ) 
  (h1 : a1 * (1 - q^4) / (1 - q) = 1)
  (h2 : a1 * (1 - q^8) / (1 - q) = 17) :
  q = 2 ∨ q = -2 :=
by
  sorry

end common_ratio_geometric_sequence_l283_283117


namespace days_without_calls_l283_283431

theorem days_without_calls (year_days : ℕ) (d4 d6 d8 : ℕ)
  (init_call : nat.gcd 4 (nat.gcd 6 8) = 1)
  (year_days_eq : year_days = 365)
  (d4_eq : d4 = 4)
  (d6_eq : d6 = 6)
  (d8_eq : d8 = 8) :
  let total_calls := (year_days / d4 + year_days / d6 + year_days / d8)
  let lcm_46 := nat.lcm d4 d6
  let lcm_48 := nat.lcm d4 d8
  let lcm_68 := nat.lcm d6 d8
  let lcm_468 := nat.lcm lcm_46 d8
  let overlap_calls := (year_days / lcm_46 + year_days / lcm_68 + year_days / lcm_468)
  let intersect_calls := year_days / lcm_468 in
  year_days - (total_calls - overlap_calls + intersect_calls) = 244 :=
sorry

end days_without_calls_l283_283431


namespace find_a_for_even_function_l283_283822

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def given_function (a : ℝ) (x : ℝ) : ℝ :=
  x * real.exp x / (real.exp (a * x) - 1)

theorem find_a_for_even_function :
  ∀ (a : ℝ), is_even_function (given_function a) ↔ a = 2 :=
by 
  intro a
  sorry

end find_a_for_even_function_l283_283822


namespace complex_num_question_l283_283412

noncomputable def z : ℂ := sorry

theorem complex_num_question (hz : 10 * complex.abs z ^ 2 = 3 * complex.abs (z + 3) ^ 2 + complex.abs (z ^ 2 - 1) ^ 2 + 40) :
  z + 9 / z = (9 + real.sqrt 61) / 2 ∨ z + 9 / z = (9 - real.sqrt 61) / 2 := sorry

end complex_num_question_l283_283412


namespace exists_three_distinct_shortest_paths_l283_283453

-- Define the type for the surface of a tetrahedron
structure TetrahedronSurface :=
  (vertices : fin 4 → ℝ × ℝ × ℝ)
  (edges : fin 6 → fin 2 → fin 4)
  (faces : fin 4 → fin 3 → fin 4)

-- Definition of a regular tetrahedron
def regular_tetrahedron : TetrahedronSurface := {
  vertices := 
    λ i => 
      match i with
      | 0 => (0, 0, 1)
      | 1 => (√(8/9), 0, -1/3)
      | 2 => (−√(2/9), √(2/3), -1/3)
      | 3 => (−√(2/9), −√(2/3), -1/3),
  edges := 
    λ i =>
      match i with
      | 0 => (0, 1)
      | 1 => (0, 2)
      | 2 => (0, 3)
      | 3 => (1, 2)
      | 4 => (1, 3)
      | 5 => (2, 3),
  faces := 
    λ i =>
      match i with
      | 0 => (0, 1, 2)
      | 1 => (0, 1, 3)
      | 2 => (0, 2, 3)
      | 3 => (1, 2, 3)
}

-- Define a point on the surface of the tetrahedron
structure PointOnSurface (T : TetrahedronSurface) :=
  (coords : ℝ × ℝ × ℝ)

-- Main theorem statement
theorem exists_three_distinct_shortest_paths (T : TetrahedronSurface) (M : PointOnSurface T) :
  ∃ M' : PointOnSurface T, ∃ (paths : fin 3 → list (PointOnSurface T)),
    (∀ i, length (paths i) = shortest_path_length M M') ∧ distinct paths :=
sorry

end exists_three_distinct_shortest_paths_l283_283453


namespace correct_formula_l283_283633

def seq : ℕ → ℤ
| 1 := 1
| 2 := -2
| 3 := 4
| 4 := -8
| 5 := 16
| 6 := -32
| _ := 0  -- handle other cases for completeness

def formula (n : ℕ) : ℤ := (-1) ^ (n + 1) * 2 ^ (n - 1)

theorem correct_formula : ∀ n, seq n = formula n :=
by
  sorry

end correct_formula_l283_283633


namespace distance_between_A_and_C_l283_283192

theorem distance_between_A_and_C :
  ∀ (AB BC CD AD AC : ℝ),
  AB = 3 → BC = 2 → CD = 5 → AD = 6 → AC = 1 := 
by
  intros AB BC CD AD AC hAB hBC hCD hAD
  have h1 : AD = AB + BC + CD := by sorry
  have h2 : 6 = 3 + 2 + AC := by sorry
  have h3 : 6 = 5 + AC := by sorry
  have h4 : AC = 1 := by sorry
  exact h4

end distance_between_A_and_C_l283_283192


namespace table_tennis_basketball_teams_l283_283054

theorem table_tennis_basketball_teams (X Y : ℕ)
  (h1 : X + Y = 50) 
  (h2 : 7 * Y = 3 * X)
  (h3 : 2 * (X - 8) = 3 * (Y + 8)) :
  X = 35 ∧ Y = 15 :=
by
  sorry

end table_tennis_basketball_teams_l283_283054


namespace number_of_non_empty_proper_subsets_l283_283556

theorem number_of_non_empty_proper_subsets {α : Type} (s : Set α) (h : s = {2, 3, 4}) : 
  (Set.toFinset (s.subsets \ {∅, s})).card = 6 :=
by
  sorry

end number_of_non_empty_proper_subsets_l283_283556


namespace find_x_l283_283042

variable {a b x : ℝ}
variable (h₀ : b ≠ 0)
variable (h₁ : (3 * a)^(2 * b) = a^b * x^b)

theorem find_x (h₀ : b ≠ 0) (h₁ : (3 * a)^(2 * b) = a^b * x^b) : x = 9 * a :=
sorry

end find_x_l283_283042


namespace range_of_a_l283_283282

theorem range_of_a (a : ℝ) (h_cond1 : log a (1/2) < 1) (h_cond2 : a^(1/2) < 1) : 0 < a ∧ a < 1/2 :=
by
  sorry

end range_of_a_l283_283282


namespace exterior_angle_regular_polygon_l283_283600

theorem exterior_angle_regular_polygon (exterior_angle : ℝ) (sides : ℕ) (h : exterior_angle = 18) : sides = 20 :=
by
  -- Use the condition that the sum of exterior angles of any polygon is 360 degrees.
  have sum_exterior_angles : ℕ := 360
  -- Set up the equation 18 * sides = 360
  have equation : 18 * sides = sum_exterior_angles := sorry
  -- Therefore, sides = 20
  sorry

end exterior_angle_regular_polygon_l283_283600


namespace age_problem_contradiction_l283_283182

theorem age_problem_contradiction (C1 C2 : ℕ) (k : ℕ)
  (h1 : 15 = k * (C1 + C2))
  (h2 : 20 = 2 * (C1 + 5 + C2 + 5)) : false :=
by
  sorry

end age_problem_contradiction_l283_283182


namespace no_other_way_to_construct_five_triangles_l283_283737

-- Setting up the problem conditions
def five_triangles_from_fifteen_sticks (sticks : List ℕ) : Prop :=
  sticks.length = 15 ∧
  let split_sticks := sticks.chunk (3) in
  split_sticks.length = 5 ∧
  ∀ triplet ∈ split_sticks, 
    let a := triplet[0]
    let b := triplet[1]
    let c := triplet[2] in
    a + b > c ∧ a + c > b ∧ b + c > a

-- The proof problem statement
theorem no_other_way_to_construct_five_triangles (sticks : List ℕ) :
  five_triangles_from_fifteen_sticks sticks → 
  ∀ (alternative : List (List ℕ)), 
    (alternative.length = 5 ∧ 
    ∀ triplet ∈ alternative, triplet.length = 3 ∧
      let a := triplet[0]
      let b := triplet[1]
      let c := triplet[2] in
      a + b > c ∧ a + c > b ∧ b + c > a) →
    alternative = sticks.chunk (3) :=
by
  sorry

end no_other_way_to_construct_five_triangles_l283_283737


namespace find_a_even_function_l283_283974

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def given_function (a : ℝ) (x : ℝ) : ℝ :=
  x * Real.exp x / (Real.exp (a * x) - 1)

theorem find_a_even_function :
  (is_even_function (given_function 2)) → true :=
by
  intro h
  sorry

end find_a_even_function_l283_283974


namespace even_function_implies_a_eq_2_l283_283929

theorem even_function_implies_a_eq_2 (a : ℝ) : 
  (∀ x, (f : ℝ → ℝ) x = (λ x : ℝ, x * exp x / (exp (a * x) - 1)) x -> (f (-x) = f x)) -> a = 2 :=
by
  sorry

end even_function_implies_a_eq_2_l283_283929


namespace even_function_implies_a_eq_2_l283_283875

def f (x a : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2 (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → a = 2 :=
by
  sorry

end even_function_implies_a_eq_2_l283_283875


namespace find_a_if_f_even_l283_283839

def f (a x : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem find_a_if_f_even (a : ℝ) (h : ∀ x : ℝ, x ≠ 0 → f a (-x) = f a x) : a = 2 :=
by 
  sorry

end find_a_if_f_even_l283_283839


namespace even_function_implies_a_eq_2_l283_283853

theorem even_function_implies_a_eq_2 (a : ℝ) 
  (h : ∀ x : ℝ, f x = f (-x)) 
  (f : ℝ → ℝ := λ x, (x * exp x) / (exp (a * x) - 1)) : a = 2 :=
sorry

end even_function_implies_a_eq_2_l283_283853


namespace inscribed_circle_radius_l283_283550

theorem inscribed_circle_radius (A B C : Type) [euclidean_space A] [has_dist A]
  (AB AC BC : ℝ) (hAB : dist A B = 6) (hAC : dist A C = 8) (hBC : dist B C = 10) :
  ∃ r : ℝ, r = 2 :=
begin
  sorry
end

end inscribed_circle_radius_l283_283550


namespace minimum_value_l283_283408

open Real

theorem minimum_value {x y : ℝ} (hx : 0 < x) (hy : 0 < y) :
  ∃ u, (u = x + 1 / y + y + 1 / x) ∧ ((x + 1 / y) * (x + 1 / y - 100) + (y + 1 / x) * (y + 1 / x - 100) = 1 / 2 * (u - 100) ^ 2 - 2500) ∧ -2500 ≤ 1 / 2 * (u - 100) ^ 2 - 2500 :=
begin
  sorry
end

end minimum_value_l283_283408


namespace determine_a_for_even_function_l283_283883

theorem determine_a_for_even_function :
  ∃ a : ℝ, (∀ x : ℝ, x ≠ 0 → (∃ f : ℝ → ℝ, 
  f x = x * exp x / (exp (a * x) - 1) ∧
  (∀ x : ℝ, f (-x) = f x))) → a = 2 :=
by
  sorry

end determine_a_for_even_function_l283_283883


namespace hyperbola_equation_intersection_point_l283_283306

-- Define the conditions
def isAsymptote (a b : ℝ) (P : ℝ × ℝ) : Prop := (a * P.1 + b * P.2 = 0)
def passesThrough (C : ℝ → ℝ → ℝ) (P : ℝ × ℝ) : Prop := C P.1 P.2 = 0
def hyperbola (C : ℝ → ℝ → ℝ) (x y : ℝ) : Prop := C x y = 3 * x^2 - y^2

-- Given constants
noncomputable def sqrt2 : ℝ := Real.sqrt 2
noncomputable def sqrt3 : ℝ := Real.sqrt 3

-- Define points and lines
def P : ℝ × ℝ := (sqrt2, sqrt3)
def line_eq (k x y: ℝ) : Prop := y = k * x + 2
def discriminant_eq (k : ℝ) : Prop := (3 - k^2) * x^2 - 4 * k * x - 7 = 0

-- Main statements
theorem hyperbola_equation :
  (isAsymptote √3 1 P) ∧ (passesThrough (λ x y => 3 * x^2 - y^2) P) →
  ∃ λ, ∀ x y, 3 * x^2 - y^2 = λ ↔ x^2 - y^2 / 3 = 1 := by
  sorry

theorem intersection_point :
  ∀ k x y,
  (line_eq k x y) ∧ (hyperbola (λ x y => 3 * x^2 - y^2) x y) →
  (∃ x, discriminant_eq k x = 0) ↔ (k = √3 ∨ k = -√3 ∨ k = Real.sqrt 7 ∨ k = -Real.sqrt 7) := by
  sorry

end hyperbola_equation_intersection_point_l283_283306


namespace sum_abscissas_eq_4_point_5_l283_283265

theorem sum_abscissas_eq_4_point_5 :
  (∑ x in ({0, 1/7, 2/7, 3/7, 4/7, 5/7, 6/7, 1/6, 1/2, 5/6} : Finset ℝ), x) = 4.5 := by
  sorry

end sum_abscissas_eq_4_point_5_l283_283265


namespace largest_k_divides_3n_plus_1_l283_283697

theorem largest_k_divides_3n_plus_1 (n : ℕ) (hn : 0 < n) : ∃ k : ℕ, k = 2 ∧ n % 2 = 1 ∧ 2^k ∣ 3^n + 1 ∨ k = 1 ∧ n % 2 = 0 ∧ 2^k ∣ 3^n + 1 :=
sorry

end largest_k_divides_3n_plus_1_l283_283697


namespace power_sum_greater_than_n_infinitely_many_l283_283274

noncomputable def power_sum (n : ℕ) : ℕ :=
  let prime_factors := Multiset.filter Nat.Prime (Multiset.range (n + 1))
  prime_factors.foldl (λ acc p =>
    let k := Nat.log p n
    acc + p ^ k
  ) 0

theorem power_sum_greater_than_n_infinitely_many :
  ∃ᶠ n in at_top, power_sum n > n :=
by
  sorry

end power_sum_greater_than_n_infinitely_many_l283_283274


namespace initial_markup_percentage_l283_283188

theorem initial_markup_percentage
  (cost_price : ℝ := 100)
  (profit_percentage : ℝ := 14)
  (discount_percentage : ℝ := 5)
  (selling_price : ℝ := cost_price * (1 + profit_percentage / 100))
  (x : ℝ := 20) :
  (cost_price + cost_price * x / 100) * (1 - discount_percentage / 100) = selling_price := by
  sorry

end initial_markup_percentage_l283_283188


namespace tan_of_right_triangle_l283_283683

theorem tan_of_right_triangle (A B C : ℝ) (h : A^2 + B^2 = C^2) (hA : A = 30) (hC : C = 37) : 
  (37^2 - 30^2).sqrt / 30 = (469).sqrt / 30 := by
  sorry

end tan_of_right_triangle_l283_283683


namespace tan_alpha_mul_cot_beta_l283_283304

open Real

noncomputable theory

theorem tan_alpha_mul_cot_beta 
  (α β : ℝ) 
  (h1 : sin (α + β) = 1 / 2) 
  (h2 : sin (α - β) = 1 / 3) 
  : tan α * cot β = 5 := 
by 
  sorry

end tan_alpha_mul_cot_beta_l283_283304


namespace even_function_implies_a_eq_2_l283_283908

def f (a x : ℝ) : ℝ := (x * exp x) / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2
  (h : ∀ x : ℝ, f a x = f a (-x)) : a = 2 := 
sorry

end even_function_implies_a_eq_2_l283_283908


namespace graph_in_quadrants_l283_283362

theorem graph_in_quadrants (x y : ℝ) : 
  (y = -6 / x) → 
  ((x > 0 → y < 0 ∧ x < 0 → y > 0) → 
  (y = -6 / x → (x > 0 ∨ x < 0) ∧ ((x > 0 ∧ y < 0) ∨ (x < 0 ∧ y > 0)) → 
  ((x > 0 ∧ y < 0) → (x, y) ∈ { (x, y) | x > 0 ∧ y < 0 }) ∧ 
  ((x < 0 ∧ y > 0) → (x, y) ∈ { (x, y) | x < 0 ∧ y > 0 }))) :=
begin
  sorry
end

end graph_in_quadrants_l283_283362


namespace f_is_even_iff_a_is_2_l283_283810

def f (a : ℝ) (x : ℝ) : ℝ := x * Real.exp x / (Real.exp (a * x) - 1)

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem f_is_even_iff_a_is_2 : (∀ x, f 2 (-x) = f 2 x) ↔ ∀ a, a = 2 := 
sorry

end f_is_even_iff_a_is_2_l283_283810


namespace five_fourths_of_twelve_fifths_eq_three_l283_283266

theorem five_fourths_of_twelve_fifths_eq_three : (5 : ℝ) / 4 * (12 / 5) = 3 := 
by 
  sorry

end five_fourths_of_twelve_fifths_eq_three_l283_283266


namespace no_alternate_way_to_form_triangles_l283_283731

noncomputable def canFormTriangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def validSetOfTriples : List (ℕ × ℕ × ℕ) :=
  [(2, 3, 4), (20, 30, 40), (200, 300, 400), (2000, 3000, 4000), (20000, 30000, 40000)]

theorem no_alternate_way_to_form_triangles (stickSet : List (ℕ × ℕ × ℕ)) :
  (∀ (t : ℕ × ℕ × ℕ), t ∈ stickSet → canFormTriangle t.1 t.2 t.3) →
  length stickSet = 5 →
  stickSet = validSetOfTriples ->
  (∀ alternateSet : List (ℕ × ℕ × ℕ),
    ∀ (t : ℕ × ℕ × ℕ), t ∈ alternateSet → canFormTriangle t.1 t.2 t.3 →
    length alternateSet = 5 →
    duplicate_stick_alternateSet = duplicate_stick_stickSet) :=
sorry

end no_alternate_way_to_form_triangles_l283_283731


namespace volume_of_sphere_l283_283307

theorem volume_of_sphere
    (area1 : ℝ) (area2 : ℝ) (distance : ℝ)
    (h1 : area1 = 9 * π)
    (h2 : area2 = 16 * π)
    (h3 : distance = 1) :
    ∃ R : ℝ, (4 / 3) * π * R ^ 3 = 500 * π / 3 :=
by
  sorry

end volume_of_sphere_l283_283307


namespace owen_turtles_l283_283445

theorem owen_turtles (o_initial : ℕ) (j_initial : ℕ) (o_after_month : ℕ) (j_remaining : ℕ) (o_final : ℕ) 
  (h1 : o_initial = 21)
  (h2 : j_initial = o_initial - 5)
  (h3 : o_after_month = 2 * o_initial)
  (h4 : j_remaining = j_initial / 2)
  (h5 : o_final = o_after_month + j_remaining) :
  o_final = 50 :=
sorry

end owen_turtles_l283_283445


namespace find_a_if_f_even_l283_283837

def f (a x : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem find_a_if_f_even (a : ℝ) (h : ∀ x : ℝ, x ≠ 0 → f a (-x) = f a x) : a = 2 :=
by 
  sorry

end find_a_if_f_even_l283_283837


namespace milk_production_l283_283075

variables (m n p x q r : ℝ)
variables (h_vals : m > 0 ∧ n > 0 ∧ p > 0 ∧ q > 0 ∧ r > 0 ∧ x > 0 ∧ x ≤ m)

theorem milk_production :
  (q * r * (m + 0.2 * x) * n) / (m * p) = (q * r * (m + 0.2 * x) * n) / (m * p) :=
by
suffices : (∀ (m n p x q r : ℝ), 
  m > 0 ∧ n > 0 ∧ p > 0 ∧ q > 0 ∧ r > 0 ∧ x > 0 ∧ x ≤ m → 
  (q * r * (m + 0.2 * x) * n) / (m * p) = (q * r * (m + 0.2 * x) * n) / (m * p)), from
sorry

end milk_production_l283_283075


namespace W_3_7_eq_13_l283_283343

-- Define the operation W
def W (x y : ℤ) : ℤ := y + 5 * x - x^2

-- State the theorem
theorem W_3_7_eq_13 : W 3 7 = 13 := by
  sorry

end W_3_7_eq_13_l283_283343


namespace exist_arithmetic_geometric_sequences_l283_283107

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n + r

noncomputable def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, b (n + 1) = b n * q

theorem exist_arithmetic_geometric_sequences : 
  ∃ (a b : ℕ → ℝ), arithmetic_sequence a ∧ geometric_sequence a ∧ arithmetic_sequence b ∧ geometric_sequence b :=
by 
suffices h : ∃ c : ℝ, c ≠ 0 → (∃ a b : ℕ → ℝ, (∀ n, a (n + 1) = a n) ∧ (∀ n, b (n + 1) = b n) ∧ (∀ n, a (n + 1) = a n + c) ∧ (∀ n, b (n + 1) = b n * 1)), by {
      obtain ⟨c, hc, ha, hb⟩ := h,
      use ha,
      use hb,
      exact ⟨_,_,_,_⟩,
    },
sorry

end exist_arithmetic_geometric_sequences_l283_283107


namespace factorize_expr_l283_283255

theorem factorize_expr (x : ℝ) : x^3 - 4 * x = x * (x + 2) * (x - 2) :=
  sorry

end factorize_expr_l283_283255


namespace largest_prime_factor_of_3136_l283_283527

theorem largest_prime_factor_of_3136 : ∃ p, p.prime ∧ p ∣ 3136 ∧ ∀ q, q.prime ∧ q ∣ 3136 → q ≤ p :=
sorry

end largest_prime_factor_of_3136_l283_283527


namespace no_alternative_way_to_construct_triangles_l283_283723

-- Define the triangle inequality condition
def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the set of stick lengths for each required triangle
def valid_triangle_sets (stick_set : Set (Set ℕ)) : Prop :=
  stick_set = { {2, 3, 4}, {20, 30, 40}, {200, 300, 400}, {2000, 3000, 4000}, {20000, 30000, 40000} }

-- Define the problem statement
theorem no_alternative_way_to_construct_triangles (s : Set ℕ) (h : s.card = 15) :
  (∀ t ∈ (s.powerset.filter (λ t, t.card = 3)), triangle_inequality t.some (finset_some_spec t) ∧ (∃ sets : Set (Set ℕ), sets.card = 5 ∧ ∀ u ∈ sets, u.card = 3 ∧ triangle_inequality u.some (finset_some_spec u)))
  → valid_triangle_sets (s.powerset.filter (λ t, t.card = 3)) := 
sorry

end no_alternative_way_to_construct_triangles_l283_283723


namespace percentage_wood_wasted_is_25_l283_283561

open Real -- Import the real number namespace

-- Define the radius of the sphere
def sphere_radius : ℝ := 9

-- Define the height of the cone
def cone_height : ℝ := 9

-- Define the diameter of the base of the cone
def cone_diameter : ℝ := 18

-- Define the radius of the base of the cone
def cone_radius : ℝ := cone_diameter / 2

-- Define the volume of the cone
def volume_cone : ℝ := (1 / 3) * (π) * (cone_radius ^ 2) * (cone_height)

-- Define the volume of the sphere
def volume_sphere : ℝ := (4 / 3) * (π) * (sphere_radius ^ 3)

-- Define the percentage of wood wasted
def percentage_wood_wasted : ℝ := (volume_cone / volume_sphere) * 100

-- Lean 4 statement to prove the percentage of wood wasted is 25%
theorem percentage_wood_wasted_is_25 :
  percentage_wood_wasted = 25 :=
  sorry

end percentage_wood_wasted_is_25_l283_283561


namespace total_races_needed_to_determine_champion_l283_283356

-- Defining the initial conditions
def num_sprinters : ℕ := 256
def lanes : ℕ := 8
def sprinters_per_race := lanes
def eliminated_per_race := sprinters_per_race - 1

-- The statement to be proved: The number of races required to determine the champion
theorem total_races_needed_to_determine_champion :
  ∃ (races : ℕ), races = 37 ∧
  ∀ s : ℕ, s = num_sprinters → 
  ∀ l : ℕ, l = lanes → 
  ∃ e : ℕ, e = eliminated_per_race →
  s - (races * e) = 1 :=
by sorry

end total_races_needed_to_determine_champion_l283_283356


namespace max_marks_l283_283566

theorem max_marks (M : ℝ) (h1 : 0.25 * M = 185 + 25) : M = 840 :=
by
  sorry

end max_marks_l283_283566


namespace even_function_implies_a_is_2_l283_283787

noncomputable def f (a x : ℝ) : ℝ := (x * Real.exp x) / (Real.exp (a * x) - 1)

theorem even_function_implies_a_is_2 (a : ℝ) 
  (h : ∀ x : ℝ, f a x = f a (-x)) : a = 2 := 
by
  sorry

end even_function_implies_a_is_2_l283_283787


namespace probability_P_conditions_l283_283224

/-- Consider a right triangle ABC with AB = 10, BC = 6 and ∠C = 90°. A point P is randomly placed within the triangle. -/
theorem probability_P_conditions :
  let ABC_area := 1/2 * 10 * 6
  let PBC_area_condition := ∀ P, (1/2 * BC * h (P proj_BC)) < 10
  let AB_distance_condition := ∀ P, distance (P proj_AB) < 3
  probability (PBC_area_condition ∧ AB_distance_condition) = 1/2 :=
sorry

end probability_P_conditions_l283_283224


namespace coefficient_expansion_l283_283313

noncomputable def coef_term (x y : ℕ) : ℕ :=
  (Nat.choose 5 2) * (2 * Nat.choose 3 1) * 45

theorem coefficient_expansion (x y : ℕ) :
  coef_term x y = 540 :=
by {
  sorry
}

end coefficient_expansion_l283_283313


namespace annie_has_12_brownies_left_l283_283210

noncomputable def initial_brownies := 100
noncomputable def portion_for_admin := (3 / 5 : ℚ) * initial_brownies
noncomputable def leftover_after_admin := initial_brownies - portion_for_admin
noncomputable def portion_for_carl := (1 / 4 : ℚ) * leftover_after_admin
noncomputable def leftover_after_carl := leftover_after_admin - portion_for_carl
noncomputable def portion_for_simon := 3
noncomputable def leftover_after_simon := leftover_after_carl - portion_for_simon
noncomputable def portion_for_friends := (2 / 3 : ℚ) * leftover_after_simon
noncomputable def each_friend_get := portion_for_friends / 5
noncomputable def total_given_to_friends := each_friend_get * 5
noncomputable def final_brownies := leftover_after_simon - total_given_to_friends

theorem annie_has_12_brownies_left : final_brownies = 12 := by
  sorry

end annie_has_12_brownies_left_l283_283210


namespace suff_but_not_nec_l283_283285

noncomputable def p (x1 x2 : ℝ) : Prop := x1^2 + 5 * x1 - 6 = 0 ∧ x2^2 + 5 * x2 - 6 = 0
def q (x1 x2 : ℝ) : Prop := x1 + x2 = -5

theorem suff_but_not_nec (x1 x2 : ℝ) : p x1 x2 → q x1 x2 ∧ ¬ (q x1 x2 → p x1 x2) :=
by
  intros h₁
  split
  { -- Prove p → q
    sorry },
  { -- Prove ¬(q → p)
    sorry }

end suff_but_not_nec_l283_283285


namespace random_two_digit_sqrt_prob_lt_seven_l283_283548

theorem random_two_digit_sqrt_prob_lt_seven :
  let total_count := 90 in
  let count_lt_sqrt7 := 48 - 10 + 1 in
  (count_lt_sqrt7 : ℚ) / total_count = 13 / 30 :=
by
  let total_count := 90
  let count_lt_sqrt7 := 48 - 10 + 1
  have h1 : count_lt_sqrt7 = 39 := by linarith
  have h2 : (count_lt_sqrt7 : ℚ) / total_count = (39 : ℚ) / 90 := by rw h1
  have h3 : (39 : ℚ) / 90 = 13 / 30 := by norm_num
  rw [h2, h3]
  refl

end random_two_digit_sqrt_prob_lt_seven_l283_283548


namespace area_of_largest_circumscribed_equilateral_triangle_area_of_smallest_inscribed_equilateral_triangle_l283_283294

open Real

variables (a b c S : ℝ)

def triangle_area_of_largest_circumscribed_equilateral_triangle : ℝ :=
  (sqrt 3 * (a^2 + b^2 + c^2)) / 6 + 2 * S

def triangle_area_of_smallest_inscribed_equilateral_triangle (S_0 : ℝ) : ℝ :=
  S^2 / S_0

theorem area_of_largest_circumscribed_equilateral_triangle :
  triangle_area_of_largest_circumscribed_equilateral_triangle a b c S =
  (sqrt 3 * (a^2 + b^2 + c^2)) / 6 + 2 * S :=
by
  sorry

theorem area_of_smallest_inscribed_equilateral_triangle :
  let S_0 := triangle_area_of_largest_circumscribed_equilateral_triangle a b c S in
  triangle_area_of_smallest_inscribed_equilateral_triangle a b c S S_0 =
  S^2 / S_0 :=
by
  sorry

end area_of_largest_circumscribed_equilateral_triangle_area_of_smallest_inscribed_equilateral_triangle_l283_283294


namespace total_shaded_area_l283_283222

theorem total_shaded_area (S T U : ℝ) 
  (h1 : 12 / S = 4)
  (h2 : S / T = 2)
  (h3 : T / U = 2) :
  1 * (S * S) + 4 * (T * T) + 8 * (U * U) = 22.5 := by
sorry

end total_shaded_area_l283_283222


namespace circle_to_rectangle_l283_283452

theorem circle_to_rectangle :
  ∃ (parts : list (set ℝ × set ℝ)), 
    (∀ p ∈ parts, is_rotatable_and_flippable p) ∧
    (union_of_parts_is_circle_with_radius_one parts) ∧ 
    (rearranged_parts_form_rectangle parts (1, 2.4)) :=
by
  sorry

end circle_to_rectangle_l283_283452


namespace octagon_area_proof_l283_283113

noncomputable theory

def side_length_of_octagon := 10

def diagonal_of_octagon : ℝ := 10 * (1 + Real.sqrt 2)

def area_of_shaded_region : ℝ := 300 + 200 * Real.sqrt 2

theorem octagon_area_proof (a : ℝ) (d : ℝ) (area : ℝ) 
  (h1 : a = side_length_of_octagon) 
  (h2 : d = diagonal_of_octagon) 
  (h3 : area = area_of_shaded_region) :
  (area = d^2) :=
by
  sorry

end octagon_area_proof_l283_283113


namespace f_is_even_iff_a_is_2_l283_283805

def f (a : ℝ) (x : ℝ) : ℝ := x * Real.exp x / (Real.exp (a * x) - 1)

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem f_is_even_iff_a_is_2 : (∀ x, f 2 (-x) = f 2 x) ↔ ∀ a, a = 2 := 
sorry

end f_is_even_iff_a_is_2_l283_283805


namespace FindDotsOnFaces_l283_283503

-- Define the structure of a die with specific dot distribution
structure Die where
  three_dots_face : ℕ
  two_dots_faces : ℕ
  one_dot_faces : ℕ

-- Define the problem scenario of 7 identical dice forming 'П' shape
noncomputable def SevenIdenticalDiceFormP (A B C : ℕ) : Prop :=
  ∃ (d : Die), 
    d.three_dots_face = 3 ∧
    d.two_dots_faces = 2 ∧
    d.one_dot_faces = 1 ∧
    (d.three_dots_face + d.two_dots_faces + d.one_dot_faces = 6) ∧
    (A = 2) ∧
    (B = 2) ∧
    (C = 3) 

-- State the theorem to prove A = 2, B = 2, C = 3 given the conditions
theorem FindDotsOnFaces (A B C : ℕ) (h : SevenIdenticalDiceFormP A B C) : A = 2 ∧ B = 2 ∧ C = 3 :=
  by sorry

end FindDotsOnFaces_l283_283503


namespace find_a_of_even_function_l283_283780

noncomputable def f (a x : ℝ) : ℝ := (x * exp x) / (exp (a * x) - 1)

theorem find_a_of_even_function (hf : ∀ x : ℝ, f a (-x) = f a x) : a = 2 :=
by {
    sorry
}

end find_a_of_even_function_l283_283780


namespace triangle_area_correct_l283_283344

noncomputable def triangle_area_given_conditions (a b c : ℝ) (A : ℝ) : ℝ :=
  if h : a = c + 4 ∧ b = c + 2 ∧ Real.cos A = -1/2 then
  1/2 * b * c * Real.sin A
  else 0

theorem triangle_area_correct :
  ∀ (a b c : ℝ), ∀ A : ℝ, a = c + 4 → b = c + 2 → Real.cos A = -1/2 → 
  triangle_area_given_conditions a b c A = 15 * Real.sqrt 3 / 4 :=
by
  intros a b c A ha hb hc
  simp [triangle_area_given_conditions, ha, hb, hc]
  sorry

end triangle_area_correct_l283_283344


namespace probability_contains_black_and_white_l283_283276

-- Define the conditions and proof statement
theorem probability_contains_black_and_white :
  let total_balls := 16 in
  let black_balls := 10 in
  let white_balls := 6 in
  let total_ways := Nat.choose total_balls 3 in
  let ways_all_black := Nat.choose black_balls 3 in
  let ways_all_white := Nat.choose white_balls 3 in
  let p_all_black_or_white := (ways_all_black + ways_all_white) / total_ways in
  (1 - p_all_black_or_white) = 3 / 4 :=
by
  let total_balls := 16
  let black_balls := 10
  let white_balls := 6
  let total_ways := Nat.choose total_balls 3
  let ways_all_black := Nat.choose black_balls 3
  let ways_all_white := Nat.choose white_balls 3
  let p_all_black_or_white := (ways_all_black + ways_all_white) / total_ways
  have p_ := (1 - p_all_black_or_white) = 3 / 4
  sorry

end probability_contains_black_and_white_l283_283276


namespace eq_a_2_l283_283964

def f (a x : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

-- Definition of even function: ∀ x, f(-x) = f(x)
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem eq_a_2 (a : ℝ) : (even_function (f a) → a = 2) ∧ (a = 2 → even_function (f a)) :=
by
  sorry

end eq_a_2_l283_283964


namespace master_efficiency_comparison_l283_283428

theorem master_efficiency_comparison (z_parts : ℕ) (z_hours : ℕ) (l_parts : ℕ) (l_hours : ℕ)
    (hz : z_parts = 5) (hz_time : z_hours = 8)
    (hl : l_parts = 3) (hl_time : l_hours = 4) :
    (z_parts / z_hours : ℚ) < (l_parts / l_hours : ℚ) → false :=
by
  -- This is a placeholder for the proof, which is not needed as per the instructions.
  sorry

end master_efficiency_comparison_l283_283428


namespace probability_sqrt_two_digit_less_than_seven_l283_283540

noncomputable def prob_sqrt_less_than_seven : ℚ := 
  let favorable := 39
  let total := 90
  favorable / total

theorem probability_sqrt_two_digit_less_than_seven : 
  prob_sqrt_less_than_seven = 13 / 30 := by
  sorry

end probability_sqrt_two_digit_less_than_seven_l283_283540


namespace find_a_for_even_function_l283_283817

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def given_function (a : ℝ) (x : ℝ) : ℝ :=
  x * real.exp x / (real.exp (a * x) - 1)

theorem find_a_for_even_function :
  ∀ (a : ℝ), is_even_function (given_function a) ↔ a = 2 :=
by 
  intro a
  sorry

end find_a_for_even_function_l283_283817


namespace max_number_of_ways_to_take_candies_is_13_l283_283647

-- Definitions for the candies and plates.
def plates := {A, B, C, D}

-- Allowed moves: 
def move_one_plate (p : plates) : ℕ := sorry
def move_three_plates (p : plates) : ℕ := sorry
def move_four_plates : ℕ := sorry 
def move_two_adjacent_plates (p1 p2 : plates) : ℕ := sorry

-- Calculation of the number of ways for each type of move and their summation.
theorem max_number_of_ways_to_take_candies_is_13 : 
  (⟦{ p : plates | move_one_plate p }⟧.size + ⟦{ p : plates | move_three_plates p }⟧.size + 
  ⟦{ () | move_four_plates }⟧.size + ⟦{ (p1, p2) : plates × plates | move_two_adjacent_plates p1 p2 }⟧.size) = 13 :=
by
  sorry

end max_number_of_ways_to_take_candies_is_13_l283_283647


namespace no_other_way_to_construct_five_triangles_l283_283738

-- Setting up the problem conditions
def five_triangles_from_fifteen_sticks (sticks : List ℕ) : Prop :=
  sticks.length = 15 ∧
  let split_sticks := sticks.chunk (3) in
  split_sticks.length = 5 ∧
  ∀ triplet ∈ split_sticks, 
    let a := triplet[0]
    let b := triplet[1]
    let c := triplet[2] in
    a + b > c ∧ a + c > b ∧ b + c > a

-- The proof problem statement
theorem no_other_way_to_construct_five_triangles (sticks : List ℕ) :
  five_triangles_from_fifteen_sticks sticks → 
  ∀ (alternative : List (List ℕ)), 
    (alternative.length = 5 ∧ 
    ∀ triplet ∈ alternative, triplet.length = 3 ∧
      let a := triplet[0]
      let b := triplet[1]
      let c := triplet[2] in
      a + b > c ∧ a + c > b ∧ b + c > a) →
    alternative = sticks.chunk (3) :=
by
  sorry

end no_other_way_to_construct_five_triangles_l283_283738


namespace bruce_age_multiple_of_son_l283_283217

structure Person :=
  (age : ℕ)

def bruce := Person.mk 36
def son := Person.mk 8
def multiple := 3

theorem bruce_age_multiple_of_son :
  ∃ (x : ℕ), bruce.age + x = multiple * (son.age + x) ∧ x = 6 :=
by
  use 6
  sorry

end bruce_age_multiple_of_son_l283_283217


namespace Owen_final_turtle_count_l283_283440

variable (Owen_turtles : ℕ) (Johanna_turtles : ℕ)

def final_turtles (Owen_turtles Johanna_turtles : ℕ) : ℕ :=
  let initial_Owen_turtles := Owen_turtles
  let initial_Johanna_turtles := Owen_turtles - 5
  let Owen_after_month := initial_Owen_turtles * 2
  let Johanna_after_losing_half := initial_Johanna_turtles / 2
  let Owen_after_donation := Owen_after_month + Johanna_after_losing_half
  Owen_after_donation

theorem Owen_final_turtle_count : final_turtles 21 (21 - 5) = 50 :=
by
  sorry

end Owen_final_turtle_count_l283_283440


namespace even_function_implies_a_eq_2_l283_283904

def f (a x : ℝ) : ℝ := (x * exp x) / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2
  (h : ∀ x : ℝ, f a x = f a (-x)) : a = 2 := 
sorry

end even_function_implies_a_eq_2_l283_283904


namespace pascal_20_fifth_element_equals_fifth_from_end_l283_283231

theorem pascal_20_fifth_element_equals_fifth_from_end :
  (nat.choose 20 4 = 4845) ∧ (nat.choose 20 16 = 4845) :=
by
  sorry

end pascal_20_fifth_element_equals_fifth_from_end_l283_283231


namespace even_function_implies_a_eq_2_l283_283938

theorem even_function_implies_a_eq_2 (a : ℝ) : 
  (∀ x, (f : ℝ → ℝ) x = (λ x : ℝ, x * exp x / (exp (a * x) - 1)) x -> (f (-x) = f x)) -> a = 2 :=
by
  sorry

end even_function_implies_a_eq_2_l283_283938


namespace trains_cross_time_l283_283137

theorem trains_cross_time
  (length : ℝ)
  (time1 time2 : ℝ)
  (h_length : length = 120)
  (h_time1 : time1 = 5)
  (h_time2 : time2 = 15)
  : (2 * length) / ((length / time1) + (length / time2)) = 7.5 := by
  sorry

end trains_cross_time_l283_283137


namespace even_function_implies_a_eq_2_l283_283859

theorem even_function_implies_a_eq_2 (a : ℝ) 
  (h : ∀ x : ℝ, f x = f (-x)) 
  (f : ℝ → ℝ := λ x, (x * exp x) / (exp (a * x) - 1)) : a = 2 :=
sorry

end even_function_implies_a_eq_2_l283_283859


namespace current_speed_is_correct_l283_283186

noncomputable def speed_of_current (row_speed_kmph : ℝ) (distance_m : ℝ) (time_s : ℝ) : ℝ :=
  let row_speed_ms := row_speed_kmph * 1000 / 3600
  let speed_downstream := distance_m / time_s
  speed_downstream - row_speed_ms

theorem current_speed_is_correct :
  speed_of_current 15 15 2.9997600191984644 ≈ 0.83333333 := by
  sorry

end current_speed_is_correct_l283_283186


namespace even_function_implies_a_is_2_l283_283783

noncomputable def f (a x : ℝ) : ℝ := (x * Real.exp x) / (Real.exp (a * x) - 1)

theorem even_function_implies_a_is_2 (a : ℝ) 
  (h : ∀ x : ℝ, f a x = f a (-x)) : a = 2 := 
by
  sorry

end even_function_implies_a_is_2_l283_283783


namespace regular_polygon_sides_l283_283614

theorem regular_polygon_sides (n : ℕ) (h : 360 = 18 * n) : n = 20 := 
by 
  sorry

end regular_polygon_sides_l283_283614


namespace exterior_angle_regular_polygon_l283_283597

theorem exterior_angle_regular_polygon (exterior_angle : ℝ) (sides : ℕ) (h : exterior_angle = 18) : sides = 20 :=
by
  -- Use the condition that the sum of exterior angles of any polygon is 360 degrees.
  have sum_exterior_angles : ℕ := 360
  -- Set up the equation 18 * sides = 360
  have equation : 18 * sides = sum_exterior_angles := sorry
  -- Therefore, sides = 20
  sorry

end exterior_angle_regular_polygon_l283_283597


namespace repeating_decimal_equiv_fraction_l283_283680

def repeating_decimal_to_fraction (x : ℚ) : ℚ := 781 / 111

theorem repeating_decimal_equiv_fraction : 7.036036036.... = 781 / 111 :=
  by
  -- lean does not support floating point literals with repeating decimals directly,
  -- need to construct the repeating decimal manually
  let x : ℚ := 7 + 36 / 999 + 36 / 999^2 + 36 / 999^3 + ...
  exactly repeating_decimal_to_fraction (x)

end repeating_decimal_equiv_fraction_l283_283680


namespace eq_a_2_l283_283967

def f (a x : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

-- Definition of even function: ∀ x, f(-x) = f(x)
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem eq_a_2 (a : ℝ) : (even_function (f a) → a = 2) ∧ (a = 2 → even_function (f a)) :=
by
  sorry

end eq_a_2_l283_283967


namespace find_a_if_f_even_l283_283830

def f (a x : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem find_a_if_f_even (a : ℝ) (h : ∀ x : ℝ, x ≠ 0 → f a (-x) = f a x) : a = 2 :=
by 
  sorry

end find_a_if_f_even_l283_283830


namespace term_with_x_first_power_rational_terms_in_expansion_largest_coefficient_terms_l283_283308

noncomputable def expansion_term_r (a b : ℕ) (x : ℝ) (r : ℕ) : ℝ := 
  (Nat.choose a r) * (2^(-r)) * (x^(4 - (3 / 4) * r))

theorem term_with_x_first_power (x : ℝ) :
  expansion_term_r 8 (1 / 24) x 4 = (35 / 8) * x :=
sorry

theorem rational_terms_in_expansion (x : ℝ) :
  ∃ t1 t2 t3, expansion_term_r 8 (1 / 24) x 0 = t1 ∧
              expansion_term_r 8 (1 / 24) x 4 = t2 ∧
              expansion_term_r 8 (1 / 24) x 8 = t3 ∧
              t1 = x^4 ∧ t2 = (35 / 8) * x ∧ t3 = (1 / 256) / x^2 :=
sorry

theorem largest_coefficient_terms (x : ℝ) :
  ∃ t3 t4, expansion_term_r 8 (1 / 24) x 2 = t3 ∧
           expansion_term_r 8 (1 / 24) x 3 = t4 ∧
           t3 = 7 * x^(5 / 2) ∧ t4 = 7 * x^(7 / 4) :=
sorry

end term_with_x_first_power_rational_terms_in_expansion_largest_coefficient_terms_l283_283308


namespace proof_problem_l283_283360

variables (a b c : ℝ)
noncomputable def circle_equation : Prop :=
  let C := {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 3)^2 = 13} in
  ∃ A B : ℝ × ℝ, A ∈ C ∧ B ∈ C ∧
    (∃ x y : ℝ, y = x^2 - 6*x + 5 ∧ 
      (x = 0 ∧ y = 5 ∧ A = (0, 5)) ∨ 
      (x = 1 ∧ y = 0 ∧ B = (1, 0)) ∨ 
      (x = 5 ∧ y = 0 ∧ B = (5, 0))) ∧
    (∃ a : ℝ, (a = sqrt 13 ∨ a = -sqrt 13) ∧
      ∃ A B : ℝ × ℝ, 
        ((x - y + a = 0 ∧ y = x^2 - 6*x + 5) ∧ (A.1 = B.1 ∧ CA ⊥ CB)))

theorem proof_problem :
  circle_equation
  sorry

end proof_problem_l283_283360


namespace right_side_of_equation_l283_283341

theorem right_side_of_equation (m : ℤ) (h : m = 8) : (-2) ^ (2 * m) = 65536 := by
  rw [h]
  norm_num
  -- (This line will compute the exponentiation and prove the equality)
  sorry

end right_side_of_equation_l283_283341


namespace regular_polygon_sides_l283_283629

theorem regular_polygon_sides (angle : ℝ) (h_angle : angle = 18) : ∃ n : ℕ, n = 20 :=
by
  have sum_exterior_angles : ℝ := 360
  have num_sides := sum_exterior_angles / angle
  have h_num_sides : num_sides = 20 := by
    calc
      num_sides = 360 / 18 : by rw [h_angle]
      ... = 20 : by norm_num
  use 20
  rw ← h_num_sides
  sorry

end regular_polygon_sides_l283_283629


namespace diminished_value_is_seven_l283_283593

theorem diminished_value_is_seven (x y : ℕ) (hx : x = 280)
  (h_eq : x / 5 + 7 = x / 4 - y) : y = 7 :=
by {
  sorry
}

end diminished_value_is_seven_l283_283593


namespace even_function_phi_l283_283317

noncomputable def phi (f : ℝ → ℝ) := ∃ φ : ℝ, (0 < φ ∧ φ < π ∧ ∀ x : ℝ, f x = 2 * sin (2 * x + φ - π / 6)) → f (-x) = f x

theorem even_function_phi :
  (∃ φ : ℝ, (0 < φ ∧ φ < π) ∧ ∀ x : ℝ, 2 * sin (2 * x + φ - π / 6) = 2 * sin (2 * (-x) + φ - π / 6)) →
  φ = 2 * π / 3 :=
sorry

end even_function_phi_l283_283317


namespace octagon_perimeter_l283_283194

def side_length_meters : ℝ := 2.3
def number_of_sides : ℕ := 8
def meter_to_cm (meters : ℝ) : ℝ := meters * 100

def perimeter_cm (side_length_meters : ℝ) (number_of_sides : ℕ) : ℝ :=
  meter_to_cm side_length_meters * number_of_sides

theorem octagon_perimeter :
  perimeter_cm side_length_meters number_of_sides = 1840 :=
by
  sorry

end octagon_perimeter_l283_283194


namespace complementary_angles_equal_l283_283555

-- Define complementary angles
def complementary (a b : ℝ) : Prop := a + b = 90

-- Assuming two angles θ₁ and θ₂ are complementary to θ.
variable (θ θ₁ θ₂ : ℝ)
hypothesis h1 : complementary θ θ₁
hypothesis h2 : complementary θ θ₂

-- Statement: Complementary angles of the same angle are equal.
theorem complementary_angles_equal : θ₁ = θ₂ :=
by
  sorry

end complementary_angles_equal_l283_283555


namespace sum_of_two_integers_l283_283106
open Nat

theorem sum_of_two_integers :
  ∃ (x y : ℕ), x * y - x - y = 87 ∧ gcd x y = 1 ∧ x < 30 ∧ y < 30 ∧ x + y = 28 :=
by
  sorry

end sum_of_two_integers_l283_283106


namespace find_a_even_function_l283_283979

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def given_function (a : ℝ) (x : ℝ) : ℝ :=
  x * Real.exp x / (Real.exp (a * x) - 1)

theorem find_a_even_function :
  (is_even_function (given_function 2)) → true :=
by
  intro h
  sorry

end find_a_even_function_l283_283979


namespace characteristic_function_inequality_l283_283465

noncomputable def characteristic_function (ϕ : ℝ → ℂ) : Prop :=
  ∀ (s t : ℝ), let M := Matrix.ofVecList 3 3 [[1, ϕ(-t), ϕ(-s)],
                                              [ϕ(t), 1, ϕ(t-s)],
                                              [ϕ(s), ϕ(s-t), 1]] in
  M.det ≥ 0

theorem characteristic_function_inequality (ϕ : ℝ → ℂ) (h : characteristic_function ϕ) (s t : ℝ) :
  |ϕ(t - s)| ≥ |ϕ(s) * ϕ(t)| - sqrt(1 - |ϕ(s)|^2) * sqrt(1 - |ϕ(t)|^2) :=
by
  sorry

end characteristic_function_inequality_l283_283465


namespace lily_pads_half_lake_l283_283167

noncomputable def size (n : ℕ) : ℝ := sorry

theorem lily_pads_half_lake {n : ℕ} (h : size 48 = size 0 * 2^48) : size 47 = (size 48) / 2 :=
by 
  sorry

end lily_pads_half_lake_l283_283167


namespace arithmetic_seq_property_l283_283046

theorem arithmetic_seq_property (a : ℕ → ℝ) (d : ℝ) (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_pos : 0 < a 1) (h_ineq : a 1 < a 2) : a 2 > real.sqrt (a 1 * a 3) :=
by
  -- (Proof not required, so use sorry)
  sorry

end arithmetic_seq_property_l283_283046


namespace regular_polygon_sides_l283_283630

theorem regular_polygon_sides (angle : ℝ) (h_angle : angle = 18) : ∃ n : ℕ, n = 20 :=
by
  have sum_exterior_angles : ℝ := 360
  have num_sides := sum_exterior_angles / angle
  have h_num_sides : num_sides = 20 := by
    calc
      num_sides = 360 / 18 : by rw [h_angle]
      ... = 20 : by norm_num
  use 20
  rw ← h_num_sides
  sorry

end regular_polygon_sides_l283_283630


namespace no_other_way_to_construct_five_triangles_l283_283755

open Classical

theorem no_other_way_to_construct_five_triangles {
  sticks : List ℤ 
} (h_stick_lengths : 
  sticks = [2, 3, 4, 20, 30, 40, 200, 300, 400, 2000, 3000, 4000, 20000, 30000, 40000]) 
  (h_five_triplets : 
  ∀ t ∈ (sticks.nthLe 0 15).combinations 3, 
  (t[0] + t[1] > t[2]) ∧ (t[1] + t[2] > t[3]) ∧ (t[0] + t[2] > t[3])) :
  ¬ ∃ other_triplets, 
  (∀ t ∈ other_triplets, t.length = 3) ∧ 
  (∀ t ∈ other_triplets, (t[0] + t[1] > t[2]) ∧ (t[1] + t[2] > t[0]) ∧ (t[0] + t[2] > t[1])) ∧ 
  other_triplets ≠ (sticks.nthLe 0 15).combinations 3 := 
by
  sorry

end no_other_way_to_construct_five_triangles_l283_283755


namespace find_a_for_even_function_l283_283826

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def given_function (a : ℝ) (x : ℝ) : ℝ :=
  x * real.exp x / (real.exp (a * x) - 1)

theorem find_a_for_even_function :
  ∀ (a : ℝ), is_even_function (given_function a) ↔ a = 2 :=
by 
  intro a
  sorry

end find_a_for_even_function_l283_283826


namespace rectangle_area_l283_283352

-- Definition of a rectangle and its properties
structure Rectangle (PQRS : Type) :=
  (P Q R S : PQRS)
  (PQ RS PS RQ : ℝ)
  (rect : PQRS)

-- The problem statement in Lean
theorem rectangle_area (PQRS : Type) (P Q R S : PQRS) (PQ RS PS RQ : ℝ)
  [rectPQRS : Rectangle PQRS]
  (trisected_angle_R : ∃ I J, ∠PRJ = 30 ∧ ∠QRJ = 60 ∧ I ∈ PQ ∧ J ∈ PS)
  (QI : ℝ) (PJ : ℝ) (hQI : QI = 8) (hPJ : PJ = 3) :
  PQ * PS = 216 := 
sorry

end rectangle_area_l283_283352


namespace cubic_polynomial_sum_l283_283089

theorem cubic_polynomial_sum (q : ℚ → ℚ)
  (hq : ∀ x, ∃ a b c d, q x = a * x^3 + b * x^2 + c * x + d)
  (h1 : q 3 = 2)
  (h2 : q 9 = 22)
  (h3 : q 18 = 14)
  (h4 : q 24 = 34) :
  (Finset.sum (Finset.filter (λ x, 2 ≤ x ∧ x ≤ 25) (Finset.range 26)) q) = 432 :=
by
  sorry

end cubic_polynomial_sum_l283_283089


namespace total_profit_of_business_l283_283159

theorem total_profit_of_business (a_investment b_investment managing_fee_percent total_received_by_a remaining_profit_ratio a_received_ratio : ℝ)
  (H_a_investment : a_investment = 2000)
  (H_b_investment : b_investment = 3000)
  (H_managing_fee_percent : managing_fee_percent = 0.10)
  (H_total_received_by_a : total_received_by_a = 4416)
  (H_remaining_profit_ratio : remaining_profit_ratio = 1 - managing_fee_percent)
  (H_a_received_ratio : a_received_ratio = a_investment / (a_investment + b_investment))
  (H_total_capital : a_investment + b_investment = 5000)
  (H_profit_ratio_identity : a_received_ratio + (1 - a_received_ratio) = 1) :
  let total_profit := total_received_by_a / (managing_fee_percent + a_received_ratio * remaining_profit_ratio)
  in total_profit = 9600 :=
by {
  sorry
}

end total_profit_of_business_l283_283159


namespace amount_paid_l283_283383

-- Define conditions
def dozen := 12
def roses_bought := 5 * dozen
def cost_per_rose := 6
def discount := 0.80

-- Define theorem
theorem amount_paid : 
  roses_bought * cost_per_rose * discount = 288 :=
by sorry

end amount_paid_l283_283383


namespace required_sticks_l283_283211

variables (x y : ℕ)
variables (h1 : 2 * x + 3 * y = 96)
variables (h2 : x + y = 40)

theorem required_sticks (x y : ℕ) (h1 : 2 * x + 3 * y = 96) (h2 : x + y = 40) : 
  x = 24 ∧ y = 16 ∧ (96 - (x * 2 + y * 3) / 2) = 116 :=
by
  sorry

end required_sticks_l283_283211


namespace weight_of_abc_l283_283086

-- Definitions based on the conditions in (a)
variables {a b c d e f i j g h k : ℝ}
def sum_group1 : ℝ := a + b + c + f + i
def sum_group2 : ℝ := a + b + c + d + e + f + i + j
def sum_group3 : ℝ := d + e + f + (d + 6) + (e - 8) + i + (j + 5)

-- Lean statement using the above definitions to prove the desired weight
theorem weight_of_abc :
  sum_group1 = 395 ∧
  sum_group2 = 664 ∧
  sum_group3 = 588 + 3 → 
  a + b + c = 237 :=
begin
  intros,
  sorry
end

end weight_of_abc_l283_283086


namespace Tim_weekly_water_intake_l283_283506

variable (daily_bottle_intake : ℚ)
variable (additional_intake : ℚ)
variable (quart_to_ounces : ℚ)
variable (days_in_week : ℕ := 7)

theorem Tim_weekly_water_intake (H1 : daily_bottle_intake = 2 * 1.5)
                              (H2 : additional_intake = 20)
                              (H3 : quart_to_ounces = 32) :
  (daily_bottle_intake * quart_to_ounces + additional_intake) * days_in_week = 812 := by
  sorry

end Tim_weekly_water_intake_l283_283506


namespace total_children_in_school_l283_283434

theorem total_children_in_school (B : ℕ) (C : ℕ) 
  (h1 : B = 2 * C)
  (h2 : B = 4 * (C - 350)) :
  C = 700 :=
by sorry

end total_children_in_school_l283_283434


namespace even_function_implies_a_eq_2_l283_283923

def f (x : ℝ) (a : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2 (a : ℝ) (h_even : ∀ x, f (-x) a = f x a) : a = 2 :=
by
  intro x
  sorry

end even_function_implies_a_eq_2_l283_283923


namespace regular_polygon_sides_l283_283618

theorem regular_polygon_sides (n : ℕ) (h : 360 = 18 * n) : n = 20 := 
by 
  sorry

end regular_polygon_sides_l283_283618


namespace regular_polygon_sides_l283_283603

theorem regular_polygon_sides (n : ℕ) (h : 1 < n) (exterior_angle : ℝ) (h_ext : exterior_angle = 18) :
  n * exterior_angle = 360 → n = 20 :=
by 
  sorry

end regular_polygon_sides_l283_283603


namespace circle_construct_l283_283518

noncomputable theory
open_locale classical

structure Point where
  x : ℝ
  y : ℝ

structure Line where
  m : ℝ
  c : ℝ

def on_same_side (A B : Point) (l : Line) : Prop :=
  (l.m * A.x + l.c - A.y) * (l.m * B.x + l.c - B.y) > 0

def distance_to_line (p : Point) (l : Line) : ℝ :=
  abs (l.m * p.x - p.y + l.c) / sqrt (l.m^2 + 1)

theorem circle_construct {A B : Point} {l : Line} :
  on_same_side A B l →
  (distance_to_line A l ≠ distance_to_line B l ∨
  distance_to_line A l = distance_to_line B l) →
  ∃ C : Point, ∃ r : ℝ, 
    r > 0 ∧ (dist C A = r) ∧ (dist C B = r) ∧ (distance_to_line C l = r) := 
begin
  intros h_same_side h_dist_relation,
  by_cases hA_on_l : l.m * A.x - A.y + l.c = 0,
  { sorry }, -- Case where A is on the line l: No solution
  by_cases hB_on_l : l.m * B.x - B.y + l.c = 0,
  { sorry }, -- Case where B is on the line l: No solution
  cases h_dist_relation,
  { sorry }, -- Case where distances to l are different: Two solutions
  { sorry }  -- Case where distances to l are same: One solution
end

end circle_construct_l283_283518


namespace even_function_implies_a_eq_2_l283_283903

def f (a x : ℝ) : ℝ := (x * exp x) / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2
  (h : ∀ x : ℝ, f a x = f a (-x)) : a = 2 := 
sorry

end even_function_implies_a_eq_2_l283_283903


namespace f_is_even_iff_a_is_2_l283_283801

def f (a : ℝ) (x : ℝ) : ℝ := x * Real.exp x / (Real.exp (a * x) - 1)

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem f_is_even_iff_a_is_2 : (∀ x, f 2 (-x) = f 2 x) ↔ ∀ a, a = 2 := 
sorry

end f_is_even_iff_a_is_2_l283_283801


namespace find_a_if_f_even_l283_283842

def f (a x : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem find_a_if_f_even (a : ℝ) (h : ∀ x : ℝ, x ≠ 0 → f a (-x) = f a x) : a = 2 :=
by 
  sorry

end find_a_if_f_even_l283_283842


namespace dot_product_is_correct_l283_283328

variables (α : ℝ)
def a := (Real.sin (2 * α), Real.cos α)
def b := (1, Real.cos α)
def tan_α := 1 / 2

theorem dot_product_is_correct (h : Real.tan α = tan_α) :
  (a α).1 * (b α).1 + (a α).2 * (b α).2 = 8 / 5 :=
by
  sorry

end dot_product_is_correct_l283_283328


namespace even_function_implies_a_eq_2_l283_283937

theorem even_function_implies_a_eq_2 (a : ℝ) : 
  (∀ x, (f : ℝ → ℝ) x = (λ x : ℝ, x * exp x / (exp (a * x) - 1)) x -> (f (-x) = f x)) -> a = 2 :=
by
  sorry

end even_function_implies_a_eq_2_l283_283937


namespace find_a_l283_283993

noncomputable def f (x a : ℝ) : ℝ := (x * (Real.exp x)) / (Real.exp (a * x) - 1)

theorem find_a (a : ℝ) (h : ∀ x, f x a = f (-x) a) : a = 2 :=
begin
  sorry
end

end find_a_l283_283993


namespace even_function_implies_a_is_2_l283_283782

noncomputable def f (a x : ℝ) : ℝ := (x * Real.exp x) / (Real.exp (a * x) - 1)

theorem even_function_implies_a_is_2 (a : ℝ) 
  (h : ∀ x : ℝ, f a x = f a (-x)) : a = 2 := 
by
  sorry

end even_function_implies_a_is_2_l283_283782


namespace part_II_part_III_l283_283340

def seq_has_property_P (a : ℕ → ℤ) (k : ℕ) : Prop :=
  k ≥ 3 ∧
  (∀ i j, 1 ≤ i ∧ i ≤ j ∧ j ≤ k → a j + a i ∈ set.range a ∨ a j - a i ∈ set.range a) ∧
  a 1 = 0 →

def part_I : Prop := 
  seq_has_property_P (λ n, match n with 
                            | 1 := 0 
                            | 2 := 1 
                            | 3 := 2 
                            | _ := 0 end) 3

theorem part_II {a : ℕ → ℤ} {k : ℕ} (hP : seq_has_property_P a k) (i : ℕ) : 
  1 ≤ i ∧ i ≤ k → a k - a i ∈ set.range a := sorry

theorem part_III {a : ℕ → ℤ} {k : ℕ} (hP : seq_has_property_P a k) (hk : k ≥ 5) : 
  ∃ d : ℤ, ∀ n : ℕ, 1 ≤ n ∧ n ≤ k - 1 → a (n + 1) - a n = d := sorry

end part_II_part_III_l283_283340


namespace complete_square_l283_283092

theorem complete_square (x m : ℝ) : x^2 + 2 * x - 2 = 0 → (x + m)^2 = 3 → m = 1 := sorry

end complete_square_l283_283092


namespace factorize_x_l283_283261

theorem factorize_x^3_minus_4x (x : ℝ) : x^3 - 4 * x = x * (x + 2) * (x - 2) := 
sorry

end factorize_x_l283_283261


namespace find_m_l283_283116

theorem find_m (m : ℕ) :
  let p := (m + x) * (1 + x)^4,
      even_coeff_sum := m + 6 * m + m + 4 + 4
  in even_coeff_sum = 24 → m = 2 :=
begin
  assume h : m + 6 * m + m + 4 + 4 = 24,
  sorry
end

end find_m_l283_283116


namespace find_real_nut_without_sacrificing_l283_283439

-- Declare the nuts with their properties
inductive Nut : Type
| real : Nut
| artificial : Nut

-- Define the set of nuts
def nuts : List Nut := [Nut.real, Nut.real, Nut.artificial, Nut.artificial, Nut.real, Nut.real]

-- Hypothesis about the properties of the weights
axiom same_weight_real (r1 r2 : Nut) : r1 = Nut.real → r2 = Nut.real → r1 = r2
axiom same_weight_artificial (a1 a2 : Nut) : a1 = Nut.artificial → a2 = Nut.artificial → a1 = a2
axiom artificial_lighter (r : Nut) (a : Nut) : r = Nut.real → a = Nut.artificial → r > a

-- Hypothesis about the balance scale with the addition that sacrificing makes it invalid 
axiom balance_scale (n1 n2 n3 n4 : Nut) (sacrificed_nut : Nut) : 
  (sacrificed_nut = Nut.real ∨ sacrificed_nut = Nut.artificial) →
  (n1 = Nut.real ∨ n1 = Nut.artificial) →
  (n2 = Nut.real ∨ n2 = Nut.artificial) →
  (n3 = Nut.real ∨ n3 = Nut.artificial) →
  (n4 = Nut.real ∨ n4 = Nut.artificial) →
  sorry

-- The main theorem to find at least one real nut without sacrificing it
theorem find_real_nut_without_sacrificing (nuts : List Nut) (w1 w2 w3 w4 : Nut) (sacrifice : Nut) : 
  (sacrifice ≠ Nut.real) → extract_real_nut (w1 :: w2 :: w3 :: w4 :: xs) :=
by
  sorry

end find_real_nut_without_sacrificing_l283_283439


namespace even_function_implies_a_eq_2_l283_283915

def f (x : ℝ) (a : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2 (a : ℝ) (h_even : ∀ x, f (-x) a = f x a) : a = 2 :=
by
  intro x
  sorry

end even_function_implies_a_eq_2_l283_283915


namespace problem_statement_l283_283395

variables (A B C P O_a O_b O_c O H : Type*) [AffineSpace A]
variable [MetricSpace X]
variables (circumcenter : Triangle → Point) (power_of_point : Point → Circle → ℝ)
variables (ABC_tri : Δ A B C) (PBC_tri PAC_tri PAB_tri: Δ P B C P A C P A B)
variables (circumcircle_ABC : Circle)

-- Definitions
def is_orthocenter (H : Point) (ABC : Δ A B C) : Prop :=
  ∃ P, ∀ (Pa : Point), 
  (Pa = circumcenter PBC_tri ∧ Pa = circumcenter PAC_tri ∧ Pa = circumcenter PAB_tri) ∧
  (P lies on the circumcircle of Δ ABC_p, where A, B, C are the vertices), 
  ∀ (Orthocenter : Point), H = resultant of (where H is the orthocenter of the given vertices).

def are_concurrent_lines (A B C : Point) (Pbc Pac Pam : Δ P B C P A C P A B → Line) : Prop :=
  ∃ X, lines pass through X, 
  (Io_a : circumcenter) (Iob : circumcenter) 

-- The problem statement
theorem problem_statement :
  (∀ (ABC : Δ A B C), acute_triangle ABC ∧
      (let PBC_tri := triangle P B C in
      let PAC_tri := triangle P A C in
      let PAB_tri := triangle P A B in
      let O_a := circumcenter PBC_tri in
      let O_b := circumcenter PAC_tri in
      let O_c := circumcenter PAB_tri in
      is_orthocenter H ABC ∧
      are_concurrent_lines AO_a BO_b CO_c ∧
      power_of_point X circumcircle_ABC = -((a^2 + b^2 + c^2 - 5*R^2) / 4)) ∧
  sorry -- as we skip the proof.

end problem_statement_l283_283395


namespace chord_length_and_line_equation_l283_283305

theorem chord_length_and_line_equation (M : ℝ × ℝ) (d : ℝ) :
  M = (-3, -3) ∧ d = sqrt 5 ∧
  (∃ l : ℝ → ℝ → Prop, (l M.1 M.2) ∧ 
  (∀ (x y : ℝ), (x, y) ∈ l → x^2 + (y + 2)^2 = 25 ∧ abs (2 + 3 * (x / y) - 3) / sqrt (1 + (x / y)^2) = sqrt 5)) →
  (∃ k : ℝ, k = 4 * sqrt 5) ∧ 
  (∃ a b c : ℝ, (a, b, c = 1, 2, 9) ∨ (a, b, c = 2, -1, 3) ∧ ∀x y, a*x + b*y + c = 0)) :=
by
  sorry

end chord_length_and_line_equation_l283_283305


namespace negation_of_not_both_are_not_even_l283_283490

variables {a b : ℕ}

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem negation_of_not_both_are_not_even :
  ¬ (¬ is_even a ∧ ¬ is_even b) ↔ (is_even a ∨ is_even b) :=
by
  sorry

end negation_of_not_both_are_not_even_l283_283490


namespace proof_problem_l283_283151

open Classical

-- Definition for statement ②
def negation_sine : Prop := 
  ¬(∀ x ∈ ℝ, sin x ≤ 1) = ∃ x₀ ∈ ℝ, sin x₀ > 1

-- Definition for statement ③
def sufficient_condition (p q : Prop) : Prop := 
  (p ∧ q) → (p ∨ q)

-- Definition for statement ④
variables {a b : ℝ} {α : set ℝ}

def perpendicular_parallel (a b : ℝ) (α : set ℝ) : Prop :=
  (a ⊥ α) ∧ (b ∥ α) → (a ⊥ b)

-- Final theorem that encapsulates the conditions and the correct statements
theorem proof_problem :
  (negation_sine ∧ sufficient_condition p q ∧ perpendicular_parallel a b α) :=
begin
  sorry
end

end proof_problem_l283_283151


namespace area_of_square_opposite_vertices_l283_283435

open Real

def distance (p q : ℝ × ℝ) : ℝ :=
  sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

def side_length_of_square_from_diagonal (d : ℝ) : ℝ :=
  d / sqrt 2

def area_of_square_from_side_length (s : ℝ) : ℝ :=
  s^2

theorem area_of_square_opposite_vertices 
  (p q : ℝ × ℝ) (h₁ : p = (1, 2)) (h₂ : q = (5, 6)) : 
  area_of_square_from_side_length (side_length_of_square_from_diagonal (distance p q)) = 16 :=
by
  sorry

end area_of_square_opposite_vertices_l283_283435


namespace number_of_soccer_campers_l283_283636

-- Conditions as definitions in Lean
def total_campers : ℕ := 88
def basketball_campers : ℕ := 24
def football_campers : ℕ := 32
def soccer_campers : ℕ := total_campers - (basketball_campers + football_campers)

-- Theorem statement to prove
theorem number_of_soccer_campers : soccer_campers = 32 := by
  sorry

end number_of_soccer_campers_l283_283636


namespace probability_of_square_root_less_than_seven_is_13_over_30_l283_283535

-- Definition of two-digit range and condition for square root check
def two_digit_numbers := Finset.range 100 \ Finset.range 10
def sqrt_condition (n : ℕ) : Prop := n < 49

-- The required probability calculation
def probability_square_root_less_than_seven : ℚ :=
  (↑(two_digit_numbers.filter sqrt_condition).card) / (↑two_digit_numbers.card)

-- The theorem stating the required probability
theorem probability_of_square_root_less_than_seven_is_13_over_30 :
  probability_square_root_less_than_seven = 13 / 30 := by
  sorry

end probability_of_square_root_less_than_seven_is_13_over_30_l283_283535


namespace even_function_implies_a_is_2_l283_283797

noncomputable def f (a x : ℝ) : ℝ := (x * Real.exp x) / (Real.exp (a * x) - 1)

theorem even_function_implies_a_is_2 (a : ℝ) 
  (h : ∀ x : ℝ, f a x = f a (-x)) : a = 2 := 
by
  sorry

end even_function_implies_a_is_2_l283_283797


namespace man_l283_283592

theorem man's_age_twice_brother's_age_in_2_years (B M : ℕ) (h1 : B = 10) (h2 : M = B + 12) : ∃ Y : ℕ, M + Y = 2 * (B + Y) ∧ Y = 2 :=
by 
  use 2
  split
  · rw [h1, h2]
    linarith
  · rfl

end man_l283_283592


namespace determine_a_for_even_function_l283_283892

theorem determine_a_for_even_function :
  ∃ a : ℝ, (∀ x : ℝ, x ≠ 0 → (∃ f : ℝ → ℝ, 
  f x = x * exp x / (exp (a * x) - 1) ∧
  (∀ x : ℝ, f (-x) = f x))) → a = 2 :=
by
  sorry

end determine_a_for_even_function_l283_283892


namespace greg_sarah_apples_l283_283329

-- Definitions and Conditions
variable {G : ℕ}
variable (H0 : 2 * G + 2 * G + (2 * G - 5) = 49)

-- Statement of the problem
theorem greg_sarah_apples : 
  2 * G = 18 :=
by
  sorry

end greg_sarah_apples_l283_283329


namespace decimal_to_base7_l283_283666

-- Define the decimal number
def decimal_number : ℕ := 2011

-- Define the base-7 conversion function
def to_base7 (n : ℕ) : List ℕ :=
  if n < 7 then [n]
  else to_base7 (n / 7) ++ [n % 7]

-- Calculate the base-7 representation of 2011
def base7_representation : List ℕ := to_base7 decimal_number

-- Prove that the base-7 representation of 2011 is [5, 6, 0, 2]
theorem decimal_to_base7 : base7_representation = [5, 6, 0, 2] :=
  by sorry

end decimal_to_base7_l283_283666


namespace cube_volume_and_surface_area_l283_283118

theorem cube_volume_and_surface_area (e : ℕ) (h : 12 * e = 72) :
  (e^3 = 216) ∧ (6 * e^2 = 216) := by
  sorry

end cube_volume_and_surface_area_l283_283118


namespace cost_of_coffee_A_per_kg_l283_283196

theorem cost_of_coffee_A_per_kg (x : ℝ) :
  (240 * x + 240 * 12 = 480 * 11) → x = 10 :=
by
  intros h
  sorry

end cost_of_coffee_A_per_kg_l283_283196


namespace quadrilateral_area_ln_l283_283500

theorem quadrilateral_area_ln (n : ℕ) (h_pos : 0 < n)
  (h_area : real.log ( (n+1) * (n+2) / (n * (n+3))) = real.log (91 / 90)) : n = 12 :=
sorry

end quadrilateral_area_ln_l283_283500


namespace sum_parallel_segments_l283_283458

theorem sum_parallel_segments :
  let n := 200
  let length_AB := 6
  let length_CB := 8
  let diagonal := Real.sqrt (length_AB^2 + length_CB^2)
  let segment_length (k : ℕ) := (10 * (n - k) / n)
  let total_length := 2 * (finset.sum (finset.range n) (λ k, segment_length k)) - 10
  total_length = 1990 :=
sorry

end sum_parallel_segments_l283_283458


namespace parabola_intersection_probability_l283_283227

theorem parabola_intersection_probability :
  let domain := {1, 2, 3, 4, 5, 6}
  ∃ (a b c d e f : ℤ), a ∈ domain ∧ b ∈ domain ∧ c ∈ domain ∧ d ∈ domain ∧ e ∈ domain ∧ f ∈ domain →
  (Prob (\exists x : ℝ, x^2 + a*x + b*x + c = x^2 + d*x + e*x + f) = 31/36) :=
begin
  sorry
end

end parabola_intersection_probability_l283_283227


namespace even_function_implies_a_eq_2_l283_283924

def f (x : ℝ) (a : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2 (a : ℝ) (h_even : ∀ x, f (-x) a = f x a) : a = 2 :=
by
  intro x
  sorry

end even_function_implies_a_eq_2_l283_283924


namespace find_a_for_even_function_l283_283819

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def given_function (a : ℝ) (x : ℝ) : ℝ :=
  x * real.exp x / (real.exp (a * x) - 1)

theorem find_a_for_even_function :
  ∀ (a : ℝ), is_even_function (given_function a) ↔ a = 2 :=
by 
  intro a
  sorry

end find_a_for_even_function_l283_283819


namespace no_other_way_to_construct_five_triangles_l283_283741

-- Setting up the problem conditions
def five_triangles_from_fifteen_sticks (sticks : List ℕ) : Prop :=
  sticks.length = 15 ∧
  let split_sticks := sticks.chunk (3) in
  split_sticks.length = 5 ∧
  ∀ triplet ∈ split_sticks, 
    let a := triplet[0]
    let b := triplet[1]
    let c := triplet[2] in
    a + b > c ∧ a + c > b ∧ b + c > a

-- The proof problem statement
theorem no_other_way_to_construct_five_triangles (sticks : List ℕ) :
  five_triangles_from_fifteen_sticks sticks → 
  ∀ (alternative : List (List ℕ)), 
    (alternative.length = 5 ∧ 
    ∀ triplet ∈ alternative, triplet.length = 3 ∧
      let a := triplet[0]
      let b := triplet[1]
      let c := triplet[2] in
      a + b > c ∧ a + c > b ∧ b + c > a) →
    alternative = sticks.chunk (3) :=
by
  sorry

end no_other_way_to_construct_five_triangles_l283_283741


namespace repeating_decimal_and_rounded_value_of_quotient_l283_283495

theorem repeating_decimal_and_rounded_value_of_quotient :
  let q := 2.2 / 6
  in (q = 0.36666...) ∧ (q.round 2 = 0.37) :=
by
  sorry

end repeating_decimal_and_rounded_value_of_quotient_l283_283495


namespace apple_juice_fraction_l283_283136

theorem apple_juice_fraction 
    (capacity_p1 : ℕ) (capacity_p2 : ℕ) 
    (full_p1: ℚ) (full_p2: ℚ) 
    (p1_fraction : ℚ) (p2_fraction : ℚ)
    (h1 : capacity_p1 = 800)
    (h2 : capacity_p2 = 700) 
    (h3 : p1_fraction = 1/4)
    (h4 : p2_fraction = 3/7) 
    (ij_p1_volume : full_p1 = capacity_p1 * p1_fraction)
    (ij_p2_volume : full_p2 = capacity_p2 * p2_fraction) :
    full_p1 + full_p2 = (capacity_p1 * p1_fraction) + (capacity_p2 * p2_fraction) -> 
    (full_p1 + full_p2) / (capacity_p1 + capacity_p2) = 1/3 :=
begin
sorry
end
 
end apple_juice_fraction_l283_283136


namespace even_function_implies_a_eq_2_l283_283854

theorem even_function_implies_a_eq_2 (a : ℝ) 
  (h : ∀ x : ℝ, f x = f (-x)) 
  (f : ℝ → ℝ := λ x, (x * exp x) / (exp (a * x) - 1)) : a = 2 :=
sorry

end even_function_implies_a_eq_2_l283_283854


namespace factorize_x_cube_minus_4x_l283_283246

theorem factorize_x_cube_minus_4x (x : ℝ) : x^3 - 4 * x = x * (x + 2) * (x - 2) := 
by
  -- Continue the proof from here
  sorry

end factorize_x_cube_minus_4x_l283_283246


namespace owen_final_turtle_count_l283_283448

theorem owen_final_turtle_count (owen_initial johanna_initial : ℕ)
  (h1: owen_initial = 21)
  (h2: johanna_initial = owen_initial - 5) :
  let owen_after_1_month := 2 * owen_initial,
      johanna_after_1_month := johanna_initial / 2,
      owen_final := owen_after_1_month + johanna_after_1_month
  in
  owen_final = 50 :=
by
  -- Solution steps go here.
  sorry

end owen_final_turtle_count_l283_283448


namespace isosceles_triangle_perimeter_l283_283643

variable (a b : ℕ) 

theorem isosceles_triangle_perimeter (h1 : a = 3) (h2 : b = 6) : 
  ∃ P, (a = 3 ∧ b = 6 ∧ P = 15 ∨ b = 3 ∧ a = 6 ∧ P = 15) := by
  use 15
  sorry

end isosceles_triangle_perimeter_l283_283643


namespace no_other_way_to_construct_five_triangles_l283_283752

open Classical

theorem no_other_way_to_construct_five_triangles {
  sticks : List ℤ 
} (h_stick_lengths : 
  sticks = [2, 3, 4, 20, 30, 40, 200, 300, 400, 2000, 3000, 4000, 20000, 30000, 40000]) 
  (h_five_triplets : 
  ∀ t ∈ (sticks.nthLe 0 15).combinations 3, 
  (t[0] + t[1] > t[2]) ∧ (t[1] + t[2] > t[3]) ∧ (t[0] + t[2] > t[3])) :
  ¬ ∃ other_triplets, 
  (∀ t ∈ other_triplets, t.length = 3) ∧ 
  (∀ t ∈ other_triplets, (t[0] + t[1] > t[2]) ∧ (t[1] + t[2] > t[0]) ∧ (t[0] + t[2] > t[1])) ∧ 
  other_triplets ≠ (sticks.nthLe 0 15).combinations 3 := 
by
  sorry

end no_other_way_to_construct_five_triangles_l283_283752


namespace degree_f_plus_g_is_3_l283_283076

-- Define the polynomials f and g
def f (z : ℂ) : ℂ := a₃*z^3 + a₂*z^2 + a₁*z + a₀
def g (z : ℂ) : ℂ := b₁*z + b₀

-- Assume the required conditions
axiom a₃_ne_zero : a₃ ≠ 0

-- Degree of a polynomial
noncomputable def degree (h : ℂ → ℂ) : ℕ := sorry

-- Statement: Proving the degree of f(z) + g(z) is 3
theorem degree_f_plus_g_is_3 : degree (λ z : ℂ, f z + g z) = 3 :=
begin
  sorry
end

end degree_f_plus_g_is_3_l283_283076


namespace train_crosses_platform_in_30_seconds_l283_283578

noncomputable def train_speed (train_length time_to_cross_pole : ℕ) : ℚ :=
  train_length / time_to_cross_pole

noncomputable def time_to_cross_platform (train_length platform_length speed : ℚ) : ℚ :=
  (train_length + platform_length) / speed

theorem train_crosses_platform_in_30_seconds :
  ∀ (train_length platform_length time_to_cross_pole : ℕ),
  time_to_cross_pole = 18 →
  train_length = 300 →
  platform_length = 200 →
  time_to_cross_platform train_length platform_length (train_speed train_length time_to_cross_pole) = 30 := 
by
  intros train_length platform_length time_to_cross_pole h_pole h_train h_platform
  rw [h_train, h_platform, h_pole]
  simp [train_speed, time_to_cross_platform]
  rw [mul_div_cancel_left (500 : ℚ) (ne_of_gt (zero_lt_bit0 (zero_lt_succ 3)))] -- 500 is non-zero
  norm_num
  sorry

end train_crosses_platform_in_30_seconds_l283_283578


namespace dealer_profit_percentage_l283_283180

-- Define the conditions
def cost_price_kg : ℕ := 1000
def given_weight_kg : ℕ := 575

-- Define the weight saved by the dealer
def weight_saved : ℕ := cost_price_kg - given_weight_kg

-- Define the profit percentage formula
def profit_percentage : ℕ → ℕ → ℚ := λ saved total_weight => (saved : ℚ) / (total_weight : ℚ) * 100

-- The main theorem statement
theorem dealer_profit_percentage : profit_percentage weight_saved cost_price_kg = 42.5 :=
by
  sorry

end dealer_profit_percentage_l283_283180


namespace eq_a_2_l283_283961

def f (a x : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

-- Definition of even function: ∀ x, f(-x) = f(x)
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem eq_a_2 (a : ℝ) : (even_function (f a) → a = 2) ∧ (a = 2 → even_function (f a)) :=
by
  sorry

end eq_a_2_l283_283961


namespace problem_statement_l283_283038

variable (f g : ℝ → ℝ)

noncomputable theory

-- Conditions
def condition1 : Prop := ∀ x, g (x + 1) - f (2 - x) = 2
def condition2 : Prop := ∀ x, f' x = g' (x - 1)
def condition3 : Prop := odd (λ x, g (x + 2))

-- Proving objectives
theorem problem_statement
  (h1 : condition1 f g)
  (h2 : condition2 f g)
  (h3 : condition3 g) :
  g 2 = 0 ∧
  (∃ p, ∀ x, f (x + p) = f x) ∧
  ∑ k in finset.range 2023 + 1, g k = 0 :=
by
  sorry

end problem_statement_l283_283038


namespace probability_sqrt_two_digit_less_than_seven_l283_283539

noncomputable def prob_sqrt_less_than_seven : ℚ := 
  let favorable := 39
  let total := 90
  favorable / total

theorem probability_sqrt_two_digit_less_than_seven : 
  prob_sqrt_less_than_seven = 13 / 30 := by
  sorry

end probability_sqrt_two_digit_less_than_seven_l283_283539


namespace equation_of_line_through_A_l283_283267

-- Definitions of givens
def point_A : Point := ⟨-Real.sqrt 3, 3⟩
def line_eq := λ x y, Real.sqrt 3 * x + y + 1 = 0
def line_through_point (p : Point) (m : Real) (x y : Real) := y - p.y = m * (x - p.x)
def slope_half_angle := Real.sqrt 3 -- The slope corresponding to 60 degrees

-- The equation to be proven
def target_line_eq := λ x y, Real.sqrt 3 * x - y + 6 = 0

-- The proof problem
theorem equation_of_line_through_A (x y : Real) :
  line_through_point point_A slope_half_angle x y → target_line_eq x y :=
by
  sorry

end equation_of_line_through_A_l283_283267


namespace even_function_implies_a_eq_2_l283_283920

def f (x : ℝ) (a : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2 (a : ℝ) (h_even : ∀ x, f (-x) a = f x a) : a = 2 :=
by
  intro x
  sorry

end even_function_implies_a_eq_2_l283_283920


namespace correct_statement_about_cell_division_l283_283150

theorem correct_statement_about_cell_division
  (metaphase_mitosis : ∀ (chromatids : ℕ) (dna_molecules : ℕ), chromatids = dna_molecules)
  (anaphase_mitosis : ∀ (chromatids : ℕ), chromatids = 0)
  (homologous_meiosis : ∀ (chromosomes : Type), ∃ (pair : chromosomes × chromosomes), true)
  (y_from_sperm : ∀ (sperm : Type), true)
  (chromatids_not_separate_sperm : ∀ (chromatid: Type), (∃ (sperm_X : chromatid × chromatid), true) ∨ (∃ (sperm_Y : chromatid × chromatid), true))
  (offspring_chromosomal_composition : ∀ (chromosome : Type), ∃ (offspring: chromosome), true)
  (secondary_spermatocyte_stage: ∀ (cell: Type), ∃ (yy_chromosomes : cell × cell), true) :
  "If chromatids do not separate during the formation of sperm, offspring with XXY may occur." := by
  sorry

end correct_statement_about_cell_division_l283_283150


namespace f_is_even_iff_a_is_2_l283_283802

def f (a : ℝ) (x : ℝ) : ℝ := x * Real.exp x / (Real.exp (a * x) - 1)

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem f_is_even_iff_a_is_2 : (∀ x, f 2 (-x) = f 2 x) ↔ ∀ a, a = 2 := 
sorry

end f_is_even_iff_a_is_2_l283_283802


namespace det_AB2_l283_283334

variables {A B : Matrix _ _ ℝ}

def det_A : ℝ := -3
def det_B : ℝ := 5

theorem det_AB2 : det (A * B * B) = -75 :=
  by sorry

end det_AB2_l283_283334


namespace unique_triangle_assembly_l283_283747

def triangle_inequality (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

def is_valid_triangle_set (sticks : List (ℕ × ℕ × ℕ)) : Prop :=
  sticks.all (λ stick_lengths, triangle_inequality stick_lengths.1 stick_lengths.2 stick_lengths.3)

theorem unique_triangle_assembly (sticks : List ℕ) (h_len : sticks.length = 15)
  (h_sticks : sticks = [2, 3, 4, 20, 30, 40, 200, 300, 400, 2000, 3000, 4000, 20000, 30000, 40000]) :
  ¬ ∃ (other_config : List (ℕ × ℕ × ℕ)),
    is_valid_triangle_set other_config ∧
    other_config ≠ [(2, 3, 4), (20, 30, 40), (200, 300, 400), (2000, 3000, 4000), (20000, 30000, 40000)] :=
begin
  sorry
end

end unique_triangle_assembly_l283_283747


namespace min_value_eval_l283_283409

noncomputable def min_value_expr (x y : ℝ) := 
  (x + 1/y) * (x + 1/y - 100) + (y + 1/x) * (y + 1/x - 100)

theorem min_value_eval (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  x = y → min_value_expr x y = -2500 :=
by
  intros hxy
  -- Insert proof steps here
  sorry

end min_value_eval_l283_283409


namespace intersect_point_sum_l283_283486

theorem intersect_point_sum (a' b' : ℝ) (x y : ℝ) 
    (h1 : x = (1 / 3) * y + a')
    (h2 : y = (1 / 3) * x + b')
    (h3 : x = 2)
    (h4 : y = 4) : 
    a' + b' = 4 :=
by
  sorry

end intersect_point_sum_l283_283486


namespace even_function_implies_a_eq_2_l283_283934

theorem even_function_implies_a_eq_2 (a : ℝ) : 
  (∀ x, (f : ℝ → ℝ) x = (λ x : ℝ, x * exp x / (exp (a * x) - 1)) x -> (f (-x) = f x)) -> a = 2 :=
by
  sorry

end even_function_implies_a_eq_2_l283_283934


namespace rationalize_denominator_l283_283457

theorem rationalize_denominator :
  ∃ (P Q R S : ℤ), S > 0 ∧ ¬ ∃ (p : ℕ), prime p ∧ p^2 ∣ Q ∧ gcd P R S = 1 ∧ 
  let expr := (21 - 7 * Real.sqrt 8) in
  P = -7 ∧ Q = 8 ∧ R = 21 ∧ S = 1 ∧ 
  P + Q + R + S = 23 ∧ (7 * (3 - Real.sqrt 8)) / (3 + Real.sqrt 8) = expr := 
begin
  sorry
end

end rationalize_denominator_l283_283457


namespace find_AE_l283_283229

-- Definitions of the conditions
variable (A B C D E : Type) [linear_ordered_field A]
variable (AB CD AC AE EC : A)
variable (Area_AED Area_BEC : A)

-- Setting the values based on the problem's conditions
def convex_quadrilateral (A B C D : Type) : Prop := sorry

-- Problem conditions in Lean
axiom AB_eq : AB = 10
axiom CD_eq : CD = 15
axiom AC_eq : AC = 18
axiom intersection : AE + EC = AC
axiom area_ratio : Area_AED = 2 * Area_BEC

-- The proof statement we want to verify
theorem find_AE : AE = 18 - 18 * real.sqrt 2 :=
by
  sorry

end find_AE_l283_283229


namespace dawns_earnings_per_hour_l283_283381

variable (hours_per_painting : ℕ) (num_paintings : ℕ) (total_earnings : ℕ)

def total_hours (hours_per_painting num_paintings : ℕ) : ℕ :=
  hours_per_painting * num_paintings

def earnings_per_hour (total_earnings total_hours : ℕ) : ℕ :=
  total_earnings / total_hours

theorem dawns_earnings_per_hour :
  hours_per_painting = 2 →
  num_paintings = 12 →
  total_earnings = 3600 →
  earnings_per_hour total_earnings (total_hours hours_per_painting num_paintings) = 150 :=
by
  intros h1 h2 h3
  sorry

end dawns_earnings_per_hour_l283_283381


namespace find_a_if_even_function_l283_283953

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x * exp x / (exp (a * x) - 1)

theorem find_a_if_even_function :
  (∀ x : ℝ, x ≠ 0 → f a (-x) = f a x) → a = 2 :=
by
  sorry

end find_a_if_even_function_l283_283953


namespace regular_polygon_sides_l283_283628

theorem regular_polygon_sides (angle : ℝ) (h_angle : angle = 18) : ∃ n : ℕ, n = 20 :=
by
  have sum_exterior_angles : ℝ := 360
  have num_sides := sum_exterior_angles / angle
  have h_num_sides : num_sides = 20 := by
    calc
      num_sides = 360 / 18 : by rw [h_angle]
      ... = 20 : by norm_num
  use 20
  rw ← h_num_sides
  sorry

end regular_polygon_sides_l283_283628


namespace maximize_profit_l283_283583

section StoreProfit

variables {x y W : ℝ}

/-- Given conditions:
  Cost price = $100
  Selling price must not exceed $1.4$ times the cost price (i.e. $140)
  Sales quantity y is a linear function of the selling price x.
  Data points: (130, 140) and (140, 120) -/
noncomputable def sales_function : ℝ → ℝ :=
  fun x => -2 * x + 400

def profit_function (x : ℝ) : ℝ :=
  (x - 100) * (sales_function x)

theorem maximize_profit (hx : x ≤ 140) : profit_function 140 = 4800 :=
by
  sorry

end StoreProfit

end maximize_profit_l283_283583


namespace stratified_sampling_correct_l283_283575

def elderly := 27
def middle_aged := 54
def young := 81
def elderly_sampled := 6
def middle_aged_sampled := 12
def young_sampled := 18
def total_sampled := 36

theorem stratified_sampling_correct :
  let ratio := (elderly_sampled : ℕ) / elderly = (middle_aged_sampled : ℕ) / middle_aged ∧
               (elderly_sampled : ℕ) / elderly = (young_sampled : ℕ) / young ∧
               total_sampled = elderly_sampled + middle_aged_sampled + young_sampled
  in ratio → total_sampled = 36 :=
by
  intros
  sorry

end stratified_sampling_correct_l283_283575


namespace find_a_even_function_l283_283975

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def given_function (a : ℝ) (x : ℝ) : ℝ :=
  x * Real.exp x / (Real.exp (a * x) - 1)

theorem find_a_even_function :
  (is_even_function (given_function 2)) → true :=
by
  intro h
  sorry

end find_a_even_function_l283_283975


namespace eq_a_2_l283_283970

def f (a x : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

-- Definition of even function: ∀ x, f(-x) = f(x)
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem eq_a_2 (a : ℝ) : (even_function (f a) → a = 2) ∧ (a = 2 → even_function (f a)) :=
by
  sorry

end eq_a_2_l283_283970


namespace find_a_even_function_l283_283977

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def given_function (a : ℝ) (x : ℝ) : ℝ :=
  x * Real.exp x / (Real.exp (a * x) - 1)

theorem find_a_even_function :
  (is_even_function (given_function 2)) → true :=
by
  intro h
  sorry

end find_a_even_function_l283_283977


namespace total_dog_legs_l283_283516

theorem total_dog_legs (total_animals cats dogs: ℕ) (h1: total_animals = 300) 
  (h2: cats = 2 / 3 * total_animals) 
  (h3: dogs = 1 / 3 * total_animals): (dogs * 4) = 400 :=
by
  sorry

end total_dog_legs_l283_283516


namespace centers_of_squares_form_square_l283_283064

variables {A B C D P Q R S O : Type}
variables [AffineSpace ℝ A] [AffineSpace ℝ B] [AffineSpace ℝ C] [AffineSpace ℝ D]
variables [AffineSpace ℝ P] [AffineSpace ℝ Q] [AffineSpace ℝ R] [AffineSpace ℝ S]
variables [AffineSpace ℝ O]

-- Define the rhombus and its properties
def is_rhombus (A B C D : Type) [AffineSpace ℝ A] [AffineSpace ℝ B] [AffineSpace ℝ C] [AffineSpace ℝ D] :=
  ∃ (O : Type) [AffineSpace ℝ O], -- Exists a center O where diagonals intersect
  true -- We just need the existence for the context, we don't specify the internal property in this lean code

-- Define the external squares on each side of the rhombus and the property of their centers
def external_square_center (A B : Type) [AffineSpace ℝ A] [AffineSpace ℝ B] (P : Type) [AffineSpace ℝ P] :=
  true -- Centers P, Q, R, S defined similarly for each side of the rhombus, simplified for Lean code

-- Finally, the statement of the proof problem
theorem centers_of_squares_form_square (h_rhombus : is_rhombus A B C D)
    (h_sq1 : external_square_center A B P)
    (h_sq2 : external_square_center B C Q)
    (h_sq3 : external_square_center C D R)
    (h_sq4 : external_square_center D A S) :
    is_square P Q R S :=
sorry

end centers_of_squares_form_square_l283_283064


namespace regular_polygon_sides_l283_283615

theorem regular_polygon_sides (n : ℕ) (h : 360 = 18 * n) : n = 20 := 
by 
  sorry

end regular_polygon_sides_l283_283615


namespace jan_total_amount_paid_l283_283390

-- Define the conditions
def dozen := 12
def roses_in_dozen := 5 * dozen
def cost_per_rose := 6
def discount := 0.8

-- Define the expected result
def total_amount_paid := 288

-- Theorem statement to prove the total amount paid
theorem jan_total_amount_paid :
  roses_in_dozen * cost_per_rose * discount = total_amount_paid := 
by
  sorry

end jan_total_amount_paid_l283_283390


namespace integral_coefficient_expansion_l283_283479

theorem integral_coefficient_expansion (a : ℝ) (h : -(35 * a^3) = -280) : ∫ x in a..(2 * Real.exp 1), 1 / x = 1 :=
by
  have ha : a = 2 := 
    sorry -- Proof that a = 2 based on the algebraic condition provided
  rw ha
  simp
  -- Further proof steps to show the integral result are skipped
  sorry

end integral_coefficient_expansion_l283_283479


namespace total_surface_area_eq_l283_283585

-- Definitions for the given conditions
def height (cylinder : Type) : ℝ := 12
def radius (cylinder : Type) : ℝ := 5

-- Theorem statement
theorem total_surface_area_eq (cylinder : Type) :
  let h := height cylinder
  let r := radius cylinder
  2 * Real.pi * r^2 + 2 * Real.pi * r * h = 170 * Real.pi := sorry

end total_surface_area_eq_l283_283585


namespace spinner_final_direction_is_south_l283_283375

/-- Represents possible directions a spinner can point to -/
inductive Direction
| north
| east
| south
| west

open Direction

/-- Calculates the final direction after rotating a given amount of revolutions clockwise from north -/
def finalDirection (cw1 ccw2 cw3 : ℚ) : Direction :=
  let net_rot := cw1 - ccw2 + cw3    -- Calculate net rotations
  let net_revs := (net_rot % 1)     -- Reduce to within one full revolution
  match (net_revs * 4 : ℚ).toNat with
  | 0 => north
  | 1 => east
  | 2 => south
  | 3 => west
  | _ => sorry                     -- Should not happen, placeholder for completeness

def initial_revolutions_cw := 3 + 1 / 2     -- 3 1/2 revolutions clockwise
def revolutions_ccw := 5 + 1 / 3            -- 5 1/3 revolutions counterclockwise
def final_revolutions_cw := 2 + 1 / 6       -- 2 1/6 revolutions clockwise

theorem spinner_final_direction_is_south :
  finalDirection initial_revolutions_cw revolutions_ccw final_revolutions_cw = south := by
  sorry

end spinner_final_direction_is_south_l283_283375


namespace midpoint_triangle_similar_to_original_l283_283523

-- Define the structure for triangles
structure Triangle (P : Type) :=
(a b c : P)

-- Define the notion of similarity for triangles
def similar {P : Type} [Field P] [NormedAddCommGroup P] [NormedSpace ℝ P] (T1 T2 : Triangle P) : Prop :=
∃ (k : ℝ), k > 0 ∧
  dist T1.a T1.b = k * dist T2.a T2.b ∧
  dist T1.b T1.c = k * dist T2.b T2.c ∧
  dist T1.c T1.a = k * dist T2.c T2.a ∧
  angle T1.a T1.b T1.c = angle T2.a T2.b T2.c ∧
  angle T1.b T1.c T1.a = angle T2.b T2.c T2.a ∧
  angle T1.c T1.a T1.b = angle T2.c T2.a T2.b

-- Define the midpoint of two points
def midpoint {P : Type} [AddCommGroup P] [Module ℝ P] (p1 p2 : P) : P := (p1 + p2) / 2

-- Define similar triangles
def triangle_similar_midpoints {P : Type} [Field P] [NormedAddCommGroup P] [NormedSpace ℝ P]
  (T1 T2 : Triangle P) : Prop :=
let A1 := midpoint T1.a T2.a,
    B1 := midpoint T1.b T2.b,
    C1 := midpoint T1.c T2.c in
similar ⟨A1, B1, C1⟩ T1

-- The final theorem statement
theorem midpoint_triangle_similar_to_original {P : Type} [Field P] [NormedAddCommGroup P] [NormedSpace ℝ P]
  (T1 T2 : Triangle P) (h : similar T1 T2) : triangle_similar_midpoints T1 T2 :=
sorry

end midpoint_triangle_similar_to_original_l283_283523


namespace cake_radius_l283_283083

theorem cake_radius (x y : ℝ) : 
  (x^2 + y^2 + 1 = 2*x + 6*y) → ∃ r : ℝ, r = 3 :=
by
  intro h
  use 3
  sorry

end cake_radius_l283_283083


namespace even_function_implies_a_eq_2_l283_283846

theorem even_function_implies_a_eq_2 (a : ℝ) 
  (h : ∀ x : ℝ, f x = f (-x)) 
  (f : ℝ → ℝ := λ x, (x * exp x) / (exp (a * x) - 1)) : a = 2 :=
sorry

end even_function_implies_a_eq_2_l283_283846


namespace gcd_decreased_by_15_eq_zero_l283_283342

def a : ℕ := 7350
def b : ℕ := 165

theorem gcd_decreased_by_15_eq_zero (a b : ℕ) : a = 7350 → b = 165 → (Nat.gcd a b) - 15 = 0 :=
by
  intros ha hb
  rw [ha, hb]
  have hgcd : Nat.gcd 7350 165 = 15 := by sorry
  rw [hgcd]
  norm_num

end gcd_decreased_by_15_eq_zero_l283_283342


namespace cylinder_surface_area_l283_283587

-- Define the necessary parameters based on the conditions
def height : ℝ := 12
def radius : ℝ := 5

-- The formula to calculate the total surface area of a cylinder
def total_surface_area (h r : ℝ) : ℝ := 2 * Real.pi * r * (r + h)

-- The goal is to prove that this total surface area is 170π for given h and r
theorem cylinder_surface_area : total_surface_area height radius = 170 * Real.pi := 
by 
  -- Proof is left as an exercise.
  sorry

end cylinder_surface_area_l283_283587


namespace sally_cards_l283_283462

theorem sally_cards (initial_cards dan_cards bought_cards : ℕ) (h1 : initial_cards = 27) (h2 : dan_cards = 41) (h3 : bought_cards = 20) :
  initial_cards + dan_cards + bought_cards = 88 := by
  sorry

end sally_cards_l283_283462


namespace exterior_angle_regular_polygon_l283_283598

theorem exterior_angle_regular_polygon (exterior_angle : ℝ) (sides : ℕ) (h : exterior_angle = 18) : sides = 20 :=
by
  -- Use the condition that the sum of exterior angles of any polygon is 360 degrees.
  have sum_exterior_angles : ℕ := 360
  -- Set up the equation 18 * sides = 360
  have equation : 18 * sides = sum_exterior_angles := sorry
  -- Therefore, sides = 20
  sorry

end exterior_angle_regular_polygon_l283_283598


namespace smallest_k_for_power_of_3_l283_283230

def seq_a : ℕ → ℝ
| 0       := 1
| 1       := real.sqrt 3
| (n + 2) := (seq_a (n + 1))^3 * (seq_a n)

-- Define the product up to k
def product_a_up_to : ℕ → ℝ
| 0     := seq_a 0
| (k+1) := product_a_up_to k * seq_a (k+1)

-- The goal is to find the smallest k such that product is a power of 3
-- Prove this equivalence
theorem smallest_k_for_power_of_3 : ∃ (k : ℕ), (∀ m : ℕ, product_a_up_to k = 3^m) → k = 3 :=
sorry

end smallest_k_for_power_of_3_l283_283230


namespace find_april_decrease_l283_283001

open Real

noncomputable def initial_price : ℝ := 100.0
noncomputable def january_increase (S₀ : ℝ) := S₀ * 1.15
noncomputable def february_decrease (S₁ : ℝ) := S₁ * 0.9
noncomputable def march_increase (S₂ : ℝ) := S₂ * 1.2
noncomputable def april_decrease (S₃ : ℝ) (y : ℝ) := S₃ * (1 - y / 100)

theorem find_april_decrease :
  let S₀ := initial_price in
  let S₁ := january_increase S₀ in
  let S₂ := february_decrease S₁ in
  let S₃ := march_increase S₂ in
  S₄ = initial_price →
  S₄ = april_decrease S₃ 19 :=
by
  sorry

end find_april_decrease_l283_283001


namespace number_of_students_in_Diligence_before_transfer_l283_283000

-- Define the total number of students and the transfer information
def total_students : ℕ := 50
def transferred_students : ℕ := 2

-- Define the number of students in Diligence before the transfer
def students_in_Diligence_before : ℕ := 23

-- Let's prove that the number of students in Diligence before the transfer is 23
theorem number_of_students_in_Diligence_before_transfer :
  (total_students / 2) - transferred_students = students_in_Diligence_before :=
by {
  -- The proof is omitted as instructed
  sorry
}

end number_of_students_in_Diligence_before_transfer_l283_283000


namespace percentage_slump_in_business_l283_283097

theorem percentage_slump_in_business (X Y : ℝ) (h1 : 0.04 * X = 0.05 * Y) : 
  (1 - Y / X) * 100 = 20 :=
by
  sorry

end percentage_slump_in_business_l283_283097


namespace no_alternative_way_to_construct_triangles_l283_283712

theorem no_alternative_way_to_construct_triangles (sticks : list ℕ) (h_len : sticks.length = 15)
  (h_set : ∃ (sets : list (list ℕ)), sets.length = 5 ∧ ∀ set ∈ sets, set.length = 3 
           ∧ (∀ (a b c : ℕ) (ha : a ∈ set) (hb : b ∈ set) (hc : c ∈ set), 
             (a + b > c ∧ a + c > b ∧ b + c > a))):
  ¬∃ (alt_sets : list (list ℕ)), alt_sets ≠ sets ∧ alt_sets.length = 5 
      ∧ ∀ set ∈ alt_sets, set.length = 3 
      ∧ (∀ (a b c : ℕ) (ha : a ∈ set) (hb : b ∈ set) (hc : c ∈ set),
        (a + b > c ∧ a + c > b ∧ b + c > a)) :=
sorry

end no_alternative_way_to_construct_triangles_l283_283712


namespace even_function_implies_a_eq_2_l283_283863

def f (x a : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2 (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → a = 2 :=
by
  sorry

end even_function_implies_a_eq_2_l283_283863


namespace min_value_of_function_l283_283694

theorem min_value_of_function (x : ℝ) (hx : x > 0) :
  ∃ y, y = (3 + x + x^2) / (1 + x) ∧ y = -1 + 2 * Real.sqrt 3 :=
sorry

end min_value_of_function_l283_283694


namespace required_interest_rate_l283_283202

theorem required_interest_rate :
  let total_available := 12000 in
  let invested1 := 5000 in
  let rate1 := 0.06 in
  let invested2 := 4000 in
  let rate2 := 0.035 in
  let desired_income := 600 in
  let total_income := invested1 * rate1 + invested2 * rate2 in
  let remaining_to_invest := total_available - (invested1 + invested2) in
  let additional_income_needed := desired_income - total_income in
  let required_rate_percent := (additional_income_needed * 100) / remaining_to_invest in
  required_rate_percent = 5.33 :=
sorry

end required_interest_rate_l283_283202


namespace distinct_paintings_of_square_l283_283662

theorem distinct_paintings_of_square : 
  let disks := ({0, 1, 2, 3} : Finset ℕ),
      colorings := { l // (l.count 0 = 2) ∧ (l.count 1 = 1) ∧ (l.count 2 = 1) } in
  let symmetries := { rot0 := disks, rot90 := disks, rot180 := disks,
                       rot270 := disks, refh := disks, refv := disks, refd1 := disks, refd2 := disks } in 
  colorings.card / symmetries.card = 3 :=
by sorry

end distinct_paintings_of_square_l283_283662


namespace find_a_if_even_function_l283_283956

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x * exp x / (exp (a * x) - 1)

theorem find_a_if_even_function :
  (∀ x : ℝ, x ≠ 0 → f a (-x) = f a x) → a = 2 :=
by
  sorry

end find_a_if_even_function_l283_283956


namespace min_max_ratio_of_areas_l283_283570

theorem min_max_ratio_of_areas
  (a b c r : ℝ)
  (h : 6 * (a + b + c) * r^2 = a * b * c)
  (S S1 : ℝ)
  (hS : S = (1 / 2) * r * (a + b + c))
  (R : ℝ)
  (hR : R = 4 * S / (a * b * c))
  (h2 : S1 = (| R^2 - r^2 |) * (1 / 2) * sin A * sin B * sin C / (4 * R^2)) :
  (5 - 2 * real.sqrt 3) / 36 ≤ S1 / S ∧ S1 / S ≤ (5 + 2 * real.sqrt 3) / 36 :=
by
  sorry

end min_max_ratio_of_areas_l283_283570


namespace regular_polygon_sides_l283_283627

theorem regular_polygon_sides (angle : ℝ) (h_angle : angle = 18) : ∃ n : ℕ, n = 20 :=
by
  have sum_exterior_angles : ℝ := 360
  have num_sides := sum_exterior_angles / angle
  have h_num_sides : num_sides = 20 := by
    calc
      num_sides = 360 / 18 : by rw [h_angle]
      ... = 20 : by norm_num
  use 20
  rw ← h_num_sides
  sorry

end regular_polygon_sides_l283_283627


namespace Tim_running_hours_l283_283510

theorem Tim_running_hours
  (initial_days : ℕ)
  (additional_days : ℕ)
  (hours_per_session : ℕ)
  (sessions_per_day : ℕ)
  (total_days : ℕ)
  (total_hours_per_week : ℕ) :
  initial_days = 3 →
  additional_days = 2 →
  hours_per_session = 1 →
  sessions_per_day = 2 →
  total_days = initial_days + additional_days →
  total_hours_per_week = total_days * (hours_per_session * sessions_per_day) →
  total_hours_per_week = 10 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4] at h5
  rw [nat.add_comm (initial_days * (hours_per_session * sessions_per_day)), nat.add_assoc, nat.mul_comm hours_per_session sessions_per_day] at h5
  rw h5 at h6
  exact h6

end Tim_running_hours_l283_283510


namespace exactly_three_primes_probability_l283_283080

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11

noncomputable def prob_prime (n : ℕ) : ℚ := 
  if is_prime n then 5 / 12 else 7 / 12

noncomputable def prob_exactly_three_primes : ℚ := 
  10 * (5 / 12) ^ 3 * (7 / 12) ^ 2

theorem exactly_three_primes_probability :
  prob_exactly_three_primes = 30625 / 124416 :=
by 
  sorry

end exactly_three_primes_probability_l283_283080


namespace angle_between_vectors_is_150_degrees_l283_283327

variables (a b : EuclideanSpace ℝ (Fin 3))
variables (theta: ℝ)

-- Conditions
def cond1 := ‖a‖ = Real.sqrt 3
def cond2 := ‖b‖ = 1
def cond3 := inner a b = -3 / 2

-- Problem Statement
theorem angle_between_vectors_is_150_degrees (h1 : cond1 a) (h2 : cond2 b) (h3 : cond3 a b) : 
  theta = 150 :=
sorry

end angle_between_vectors_is_150_degrees_l283_283327


namespace graph_shift_left_l283_283512

theorem graph_shift_left (x : ℝ) :
  (∀ x, (sin (2 * x) + cos (2 * x)) = sqrt 2 * cos (2 * (x + (π / 8)) - (π / 4))) → 
  (∀ x, sqrt 2 * cos (2 * x) = sqrt 2 * cos (2 * (x + (π / 8)) - (π / 4))) := 
by
  sorry

end graph_shift_left_l283_283512


namespace owen_final_turtle_count_l283_283446

theorem owen_final_turtle_count (owen_initial johanna_initial : ℕ)
  (h1: owen_initial = 21)
  (h2: johanna_initial = owen_initial - 5) :
  let owen_after_1_month := 2 * owen_initial,
      johanna_after_1_month := johanna_initial / 2,
      owen_final := owen_after_1_month + johanna_after_1_month
  in
  owen_final = 50 :=
by
  -- Solution steps go here.
  sorry

end owen_final_turtle_count_l283_283446


namespace max_y_coordinate_difference_l283_283482

theorem max_y_coordinate_difference :
  ∀ x y : ℝ,
    (y = 3 - x^2 + x^3 ∧ y = 1 + x^2 + x^3) →
    (exists max_diff : ℝ, max_diff = 2) :=
begin
  sorry
end

end max_y_coordinate_difference_l283_283482


namespace determine_a_for_even_function_l283_283891

theorem determine_a_for_even_function :
  ∃ a : ℝ, (∀ x : ℝ, x ≠ 0 → (∃ f : ℝ → ℝ, 
  f x = x * exp x / (exp (a * x) - 1) ∧
  (∀ x : ℝ, f (-x) = f x))) → a = 2 :=
by
  sorry

end determine_a_for_even_function_l283_283891


namespace regular_polygon_sides_l283_283605

theorem regular_polygon_sides (n : ℕ) (h : 1 < n) (exterior_angle : ℝ) (h_ext : exterior_angle = 18) :
  n * exterior_angle = 360 → n = 20 :=
by 
  sorry

end regular_polygon_sides_l283_283605


namespace probability_of_square_root_less_than_seven_is_13_over_30_l283_283536

-- Definition of two-digit range and condition for square root check
def two_digit_numbers := Finset.range 100 \ Finset.range 10
def sqrt_condition (n : ℕ) : Prop := n < 49

-- The required probability calculation
def probability_square_root_less_than_seven : ℚ :=
  (↑(two_digit_numbers.filter sqrt_condition).card) / (↑two_digit_numbers.card)

-- The theorem stating the required probability
theorem probability_of_square_root_less_than_seven_is_13_over_30 :
  probability_square_root_less_than_seven = 13 / 30 := by
  sorry

end probability_of_square_root_less_than_seven_is_13_over_30_l283_283536


namespace TimTotalRunHoursPerWeek_l283_283507

def TimUsedToRunTimesPerWeek : ℕ := 3
def TimAddedExtraDaysPerWeek : ℕ := 2
def MorningRunHours : ℕ := 1
def EveningRunHours : ℕ := 1

theorem TimTotalRunHoursPerWeek :
  (TimUsedToRunTimesPerWeek + TimAddedExtraDaysPerWeek) * (MorningRunHours + EveningRunHours) = 10 :=
by
  sorry

end TimTotalRunHoursPerWeek_l283_283507


namespace production_rate_problem_l283_283351

theorem production_rate_problem :
  ∀ (G T : ℕ), 
  (∀ w t, w * 3 * t = 450 * t / 150) ∧
  (∀ w t, w * 2 * t = 300 * t / 150) ∧
  (∀ w t, w * 2 * t = 360 * t / 90) ∧
  (∀ w t, w * (5/2) * t = 450 * t / 90) ∧
  (75 * 2 * 4 = 300) →
  (75 * 2 * 4 = 600) := sorry

end production_rate_problem_l283_283351


namespace find_angle_C_find_area_triangle_l283_283015

noncomputable def C_value (A B C : ℝ) (h : (sin A)^2 + (sin B)^2 - (sin C)^2 = 2 * sin A * sin B * (sqrt 3 - cos C)) : Prop :=
  C = π / 6

noncomputable def area_of_triangle (A BC : ℝ) (C : ℝ) (hA : A = π / 4) (hBC : BC = 2) (hC : C = π / 6) : Prop :=
  let B := π - A - C in
  let sinB := sin A * cos C + cos A * sin C in
  (1/2) * BC * sqrt 2 * sinB = (1 + sqrt 3) / 2

-- Statements to be proven:
-- Proof that C = π / 6 given the trigonometric condition
theorem find_angle_C (A B C : ℝ) 
  (h : (sin A)^2 + (sin B)^2 - (sin C)^2 = 2 * sin A * sin B * (sqrt 3 - cos C)) : 
  C_value A B C h :=
by sorry

-- Proof that area of triangle is (1 + sqrt 3) / 2 given A = 45°, BC = 2, and C = 30°
theorem find_area_triangle (A BC : ℝ) (C : ℝ) 
  (hA : A = π / 4) (hBC : BC = 2) (hC : C = π / 6) : 
  area_of_triangle A BC C hA hBC hC :=
by sorry

end find_angle_C_find_area_triangle_l283_283015


namespace find_a_of_even_function_l283_283779

noncomputable def f (a x : ℝ) : ℝ := (x * exp x) / (exp (a * x) - 1)

theorem find_a_of_even_function (hf : ∀ x : ℝ, f a (-x) = f a x) : a = 2 :=
by {
    sorry
}

end find_a_of_even_function_l283_283779


namespace find_a_given_f_l283_283574

def f (x a : ℝ) : ℝ := Real.logBase 2 (x^2 + a)
theorem find_a_given_f (h : f 3 a = 1) : a = -7 :=
sorry

end find_a_given_f_l283_283574


namespace find_a_if_f_even_l283_283840

def f (a x : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem find_a_if_f_even (a : ℝ) (h : ∀ x : ℝ, x ≠ 0 → f a (-x) = f a x) : a = 2 :=
by 
  sorry

end find_a_if_f_even_l283_283840


namespace probability_exactly_two_primes_correct_l283_283651

open BigOperators

-- Define the set of primes between 1 and 20
def primes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19}

-- Define the probability calculation
noncomputable def probability_of_prime : ℚ := 8 / 20
noncomputable def probability_of_non_prime : ℚ := 1 - probability_of_prime

-- Define the combined probability of exactly two primes and one non-prime
noncomputable def probability_two_primes (n : ℕ) (k : ℕ) : ℚ :=
(probability_of_prime ^ 2) * probability_of_non_prime * (k.choose 2) 

-- Specific case for n = 3, k = 3 (three dice)
noncomputable def probability_exactly_two_primes : ℚ := probability_two_primes 3 3

-- The target probability should match
theorem probability_exactly_two_primes_correct : probability_exactly_two_primes = 36 / 125 := by
  sorry

end probability_exactly_two_primes_correct_l283_283651


namespace exterior_angle_regular_polygon_l283_283601

theorem exterior_angle_regular_polygon (exterior_angle : ℝ) (sides : ℕ) (h : exterior_angle = 18) : sides = 20 :=
by
  -- Use the condition that the sum of exterior angles of any polygon is 360 degrees.
  have sum_exterior_angles : ℕ := 360
  -- Set up the equation 18 * sides = 360
  have equation : 18 * sides = sum_exterior_angles := sorry
  -- Therefore, sides = 20
  sorry

end exterior_angle_regular_polygon_l283_283601


namespace lower_side_length_l283_283476

noncomputable def lower_side (l : ℝ) (A : ℝ) (h : ℝ) (upper_side : ℝ) : Prop :=
  A = (1 / 2) * (l + upper_side) * h

theorem lower_side_length :
  ∃ l : ℝ, 
    let upper_side := l + 3.4 in
    let area := 100.62 in
    let height := 5.2 in
    lower_side l area height upper_side
    ∧ l = 17.65 :=
by
  exists 17.65
  let upper_side := 17.65 + 3.4
  let area := 100.62
  let height := 5.2
  show lower_side 17.65 area height upper_side
  sorry

end lower_side_length_l283_283476


namespace find_a_if_even_function_l283_283957

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x * exp x / (exp (a * x) - 1)

theorem find_a_if_even_function :
  (∀ x : ℝ, x ≠ 0 → f a (-x) = f a x) → a = 2 :=
by
  sorry

end find_a_if_even_function_l283_283957


namespace even_function_implies_a_eq_2_l283_283914

def f (x : ℝ) (a : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2 (a : ℝ) (h_even : ∀ x, f (-x) a = f x a) : a = 2 :=
by
  intro x
  sorry

end even_function_implies_a_eq_2_l283_283914


namespace random_two_digit_sqrt_prob_lt_seven_l283_283549

theorem random_two_digit_sqrt_prob_lt_seven :
  let total_count := 90 in
  let count_lt_sqrt7 := 48 - 10 + 1 in
  (count_lt_sqrt7 : ℚ) / total_count = 13 / 30 :=
by
  let total_count := 90
  let count_lt_sqrt7 := 48 - 10 + 1
  have h1 : count_lt_sqrt7 = 39 := by linarith
  have h2 : (count_lt_sqrt7 : ℚ) / total_count = (39 : ℚ) / 90 := by rw h1
  have h3 : (39 : ℚ) / 90 = 13 / 30 := by norm_num
  rw [h2, h3]
  refl

end random_two_digit_sqrt_prob_lt_seven_l283_283549


namespace q_minus_p_value_l283_283499

theorem q_minus_p_value :
  let a := 5.457
  let b := 2.951
  let c := 3.746
  let d := 4.398
  let p := (Float.round 16.552 * 100).toNat / 100
  let q := 5.457 + 2.951 + 3.746 + 4.398
  q - p = 0.002 :=
by
  -- Placeholder proof
  sorry

end q_minus_p_value_l283_283499


namespace parabola_equation_l283_283594

-- Given conditions:
def focus : ℝ × ℝ := (5, -2)
def directrix (x y : ℝ) := 4 * x - 5 * y - 20 = 0

-- Theorem statement:
theorem parabola_equation :
  ∀ (x y : ℝ),
    let lhs := (x - 5) ^ 2 + (y + 2) ^ 2 in
    let rhs := (4 * x - 5 * y - 20) ^ 2 / 41 in
    lhs = rhs →
    25 * x ^ 2 - 40 * x * y + 16 * y ^ 2 - 250 * x - 36 * y + 789 = 0 := by
    sorry

end parabola_equation_l283_283594


namespace find_a_of_even_function_l283_283769

noncomputable def f (a x : ℝ) : ℝ := (x * exp x) / (exp (a * x) - 1)

theorem find_a_of_even_function (hf : ∀ x : ℝ, f a (-x) = f a x) : a = 2 :=
by {
    sorry
}

end find_a_of_even_function_l283_283769


namespace find_a_if_even_function_l283_283942

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x * exp x / (exp (a * x) - 1)

theorem find_a_if_even_function :
  (∀ x : ℝ, x ≠ 0 → f a (-x) = f a x) → a = 2 :=
by
  sorry

end find_a_if_even_function_l283_283942


namespace cylinder_surface_area_l283_283632

theorem cylinder_surface_area (r h : ℝ) (r_value : r = 3) (h_value : h = 8) : 
  let lateral_surface_area := 2 * Real.pi * r * h,
      base_area := Real.pi * r^2,
      total_surface_area := lateral_surface_area + 2 * base_area 
  in total_surface_area = 66 * Real.pi := 
by
  sorry

end cylinder_surface_area_l283_283632


namespace sum_of_divisors_91_l283_283142

def divisors (n : ℕ) : Set ℕ := { d | ∃ k, d * k = n }

theorem sum_of_divisors_91 : (∑ x in divisors 91, x) = 112 :=
by
  sorry

end sum_of_divisors_91_l283_283142


namespace larger_root_minus_smaller_root_eq_l283_283484

theorem larger_root_minus_smaller_root_eq :
  let a := 7 + 4 * real.sqrt 3,
      b := 2 + real.sqrt 3,
      c := -2,
      larger_root := (-b + real.sqrt (b^2 - 4 * a * c)) / (2 * a),
      smaller_root := (-b - real.sqrt (b^2 - 4 * a * c)) / (2 * a) in
  (larger_root - smaller_root) = 6 - 3 * real.sqrt 3 :=
by
  sorry

end larger_root_minus_smaller_root_eq_l283_283484


namespace cube_root_of_product_l283_283655

theorem cube_root_of_product : (∛(2^9 * 5^3 * 7^3) : ℝ) = 280 := by
  -- Conditions given
  let expr := (2^9 * 5^3 * 7^3 : ℝ)
  
  -- Statement of the problem equivalent to the correct answer
  have h : (∛expr : ℝ) = ∛(2^9 * 5^3 * 7^3) := by rfl

  -- Calculating the actual result
  have result :  ( (2^3) * 5 * 7 : ℝ) = 280 := by
    calc 
      ( (2^3) * 5 * 7 : ℝ) = (8 * 5 * 7 : ℝ) : by rfl
      ... = (40 * 7 : ℝ) : by rfl
      ... = 280 : by rfl

  -- Combining these results to finish the proof
  show (∛expr : ℝ) = 280 from
    calc 
      (∛expr : ℝ) = (2^3 * 5 * 7 : ℝ) : by sorry
      ... = 280 : by exact result

end cube_root_of_product_l283_283655


namespace geometry_problem_l283_283420

-- Definitions of the essential points and lines in the problem
variables {A B C D E F : Point}
variable {α : Triangle}

-- Assumptions for the problem
axiom ABC_isosceles : is_isosceles_triangle α A B C
axiom D_on_BC : lies_on D (segment B C)
axiom F_on_arc_ADC : lies_on_arc F (circumcircle A D C)
axiom E_on_AB : lies_on E (segment A B) ∧ lies_on E (circumcircle B D F)

-- To Prove
theorem geometry_problem :
  CD * EF + DF * AE = BD * AF :=
begin
  sorry,
end

end geometry_problem_l283_283420


namespace boa_constrictor_is_70_inches_l283_283204

-- Definitions based on given problem conditions
def garden_snake_length : ℕ := 10
def boa_constrictor_length : ℕ := 7 * garden_snake_length

-- Statement to prove
theorem boa_constrictor_is_70_inches : boa_constrictor_length = 70 :=
by
  sorry

end boa_constrictor_is_70_inches_l283_283204


namespace relationship_among_f_l283_283280

theorem relationship_among_f (
  f : ℝ → ℝ
) (h_even : ∀ x, f x = f (-x))
  (h_periodic : ∀ x, f (x - 1) = f (x + 1))
  (h_increasing : ∀ a b, (0 ≤ a ∧ a < b ∧ b ≤ 1) → f a < f b) :
  f 2 < f (-5.5) ∧ f (-5.5) < f (-1) :=
by
  sorry

end relationship_among_f_l283_283280


namespace integer_coordinates_for_platonic_solids_l283_283554

theorem integer_coordinates_for_platonic_solids :
  ∀ (solid : PlatonicSolid), 
    (∃ vertices : List (ℝ × ℝ × ℝ), 
       (∀ v ∈ vertices, ∃ a b c : ℤ, v = (a, b, c)) 
       → (solid = PlatonicSolid.Cube 
           ∨ solid = PlatonicSolid.Tetrahedron 
           ∨ solid = PlatonicSolid.Octahedron)) :=
sorry

end integer_coordinates_for_platonic_solids_l283_283554


namespace imaginary_part_of_z_l283_283422

def complex_number_z (z : ℂ) : Prop :=
  z * (1 + complex.I) = 4

theorem imaginary_part_of_z :
  ∃ z : ℂ, complex_number_z z ∧ z.im = -2 :=
by
  sorry

end imaginary_part_of_z_l283_283422


namespace probability_intersects_circle_l283_283456

def line (k : ℝ) := ∀ x y : ℝ, y = k * (x + 3)
def circle := ∀ x y : ℝ, x^2 + y^2 = 1

theorem probability_intersects_circle :
  let P := (set.Icc (-(real.sqrt 2) / 4) (real.sqrt 2 / 4)).measure / (real.measure (set.Icc (-1) 1))
  P = real.sqrt 2 / 4 :=
sorry

end probability_intersects_circle_l283_283456


namespace fraction_between_l283_283139

theorem fraction_between (a b : ℚ) (ha : a = 3/4) (hb : b = 5/7) (h1 : b > 1/2) :
  let c := (1/2) * (a + b),
  ∃ (c : ℚ), c = 41/56 ∧ c > 1/2 :=
by {
  -- Stack the actual proof here later.
  sorry
}

end fraction_between_l283_283139


namespace even_function_implies_a_eq_2_l283_283935

theorem even_function_implies_a_eq_2 (a : ℝ) : 
  (∀ x, (f : ℝ → ℝ) x = (λ x : ℝ, x * exp x / (exp (a * x) - 1)) x -> (f (-x) = f x)) -> a = 2 :=
by
  sorry

end even_function_implies_a_eq_2_l283_283935


namespace fraction_sum_l283_283564

variable {w x y : ℚ}  -- assuming w, x, and y are rational numbers

theorem fraction_sum (h1 : w / x = 1 / 3) (h2 : w / y = 2 / 3) : (x + y) / y = 3 :=
sorry

end fraction_sum_l283_283564


namespace intersection_of_median_and_altitude_l283_283346

noncomputable def intersection_point (A B C : ℝ × ℝ) : ℝ × ℝ :=
  let M := mid_point A B
  let CM := line_through C M
  let AC := line_through A C
  let BN := perpendicular_bisector AC B
  find_intersection CM BN

theorem intersection_of_median_and_altitude (A B C : ℝ × ℝ) (hA : A = (5, 1)) (hB : B = (-1, -3)) (hC : C = (4, 3)) :
  intersection_point A B C = (5 / 3, -5 / 3) :=
by
  -- Proof will go here
  sorry

end intersection_of_median_and_altitude_l283_283346


namespace remainder_2_pow_224_plus_104_l283_283551

theorem remainder_2_pow_224_plus_104 (x : ℕ) (h1 : x = 2 ^ 56) : 
  (2 ^ 224 + 104) % (2 ^ 112 + 2 ^ 56 + 1) = 103 := 
by
  sorry

end remainder_2_pow_224_plus_104_l283_283551


namespace f_is_even_iff_a_is_2_l283_283808

def f (a : ℝ) (x : ℝ) : ℝ := x * Real.exp x / (Real.exp (a * x) - 1)

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem f_is_even_iff_a_is_2 : (∀ x, f 2 (-x) = f 2 x) ↔ ∀ a, a = 2 := 
sorry

end f_is_even_iff_a_is_2_l283_283808


namespace travel_time_difference_in_minutes_l283_283174

/-
A bus travels at an average speed of 40 miles per hour.
We need to prove that the difference in travel time between a 360-mile trip and a 400-mile trip equals 60 minutes.
-/

theorem travel_time_difference_in_minutes 
  (speed : ℝ) (distance1 distance2 : ℝ) 
  (h1 : speed = 40) 
  (h2 : distance1 = 360) 
  (h3 : distance2 = 400) :
  (distance2 / speed - distance1 / speed) * 60 = 60 := by
  sorry

end travel_time_difference_in_minutes_l283_283174


namespace total_students_stratified_sampling_l283_283511

namespace HighSchool

theorem total_students_stratified_sampling 
  (sample_size : ℕ)
  (sample_grade10 : ℕ)
  (sample_grade11 : ℕ)
  (students_grade12 : ℕ) 
  (n : ℕ)
  (H1 : sample_size = 100)
  (H2 : sample_grade10 = 24)
  (H3 : sample_grade11 = 26)
  (H4 : students_grade12 = 600)
  (H5 : ∀ n, (students_grade12 / n * sample_size = sample_size - sample_grade10 - sample_grade11) → n = 1200) :
  n = 1200 :=
sorry

end HighSchool

end total_students_stratified_sampling_l283_283511


namespace angle_B_value_range_a_plus_c_l283_283355

-- Defining the given conditions
variables {A B C a b c : ℝ}
variables (h₀ : 0 < A ∧ A < π/2)
variables (h₁ : 0 < B ∧ B < π/2)
variables (h₂ : 0 < C ∧ C < π/2)
variables (h3 : a > 0 ∧ b > 0 ∧ c > 0)
variables (h4 : a = 4 * sin A)
variables (h5 : b = 2 * sqrt 3)
variables (h6 : c = 4 * sin C)
variables (h7 : (cos A) / a + (cos B) / b = 2 * sqrt 3 * (sin C) / (3 * a))

-- Define the two proofs we need to address
theorem angle_B_value :
  B = π / 3 :=
sorry

theorem range_a_plus_c : 6 < a + c ∧ a + c ≤ 4 * sqrt 3 :=
sorry

end angle_B_value_range_a_plus_c_l283_283355


namespace number_of_correct_propositions_is_zero_l283_283763

variables (a b : Line) (α β γ : Plane)

-- Define the propositions
def prop1 : Prop := (a ∥ α) ∧ (b ∥ α) → (a ∥ b)
def prop2 : Prop := (α ⊥ β) ∧ (β ⊥ γ) → (α ∥ γ)
def prop3 : Prop := (a ∥ α) ∧ (a ∥ β) → (α ∥ β)
def prop4 : Prop := (a ∥ b) ∧ (b ⊂ α) → (a ∥ α)

-- Define the number of correct propositions
def number_of_correct_propositions : ℕ :=
  if prop1 then 1 else 0 +
  if prop2 then 1 else 0 +
  if prop3 then 1 else 0 +
  if prop4 then 1 else 0

-- The statement to prove
theorem number_of_correct_propositions_is_zero :
  number_of_correct_propositions a b α β γ = 0 :=
sorry

end number_of_correct_propositions_is_zero_l283_283763


namespace ratio_of_speeds_l283_283450

theorem ratio_of_speeds (P R : ℝ) (total_time : ℝ) (time_rickey : ℝ)
  (h1 : total_time = 70)
  (h2 : time_rickey = 40)
  (h3 : total_time - time_rickey = 30) :
  P / R = 3 / 4 :=
by
  sorry

end ratio_of_speeds_l283_283450


namespace even_function_implies_a_eq_2_l283_283928

theorem even_function_implies_a_eq_2 (a : ℝ) : 
  (∀ x, (f : ℝ → ℝ) x = (λ x : ℝ, x * exp x / (exp (a * x) - 1)) x -> (f (-x) = f x)) -> a = 2 :=
by
  sorry

end even_function_implies_a_eq_2_l283_283928


namespace polynomial_identity_l283_283494

noncomputable def p (x : ℝ) : ℝ := x 

theorem polynomial_identity (p : ℝ → ℝ) (h : ∀ q : ℝ → ℝ, ∀ x : ℝ, p (q x) = q (p x)) : 
  (∀ x : ℝ, p x = x) :=
by
  sorry

end polynomial_identity_l283_283494


namespace even_function_implies_a_eq_2_l283_283847

theorem even_function_implies_a_eq_2 (a : ℝ) 
  (h : ∀ x : ℝ, f x = f (-x)) 
  (f : ℝ → ℝ := λ x, (x * exp x) / (exp (a * x) - 1)) : a = 2 :=
sorry

end even_function_implies_a_eq_2_l283_283847


namespace pencils_in_each_group_l283_283055

/-- Peter has 154 pencils and he created 14 groups. Prove that the number of pencils in each group is 11. -/
theorem pencils_in_each_group (pencils : ℕ) (groups : ℕ) (h_pencils : pencils = 154) (h_groups : groups = 14) :
  pencils / groups = 11 :=
by
  rw [h_pencils, h_groups]
  norm_num
  sorry

end pencils_in_each_group_l283_283055


namespace find_prime_number_p_l283_283491

theorem find_prime_number_p (p : ℕ) (h1 : p.prime) (h2 : 100 ≤ p ∧ p ≤ 110) (h3 : p % 9 = 7) : p = 106 :=
sorry

end find_prime_number_p_l283_283491


namespace even_function_implies_a_eq_2_l283_283911

def f (x : ℝ) (a : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2 (a : ℝ) (h_even : ∀ x, f (-x) a = f x a) : a = 2 :=
by
  intro x
  sorry

end even_function_implies_a_eq_2_l283_283911


namespace decomposition_correct_l283_283557

def x : ℝ^3 := ![8, 9, 4]
def p : ℝ^3 := ![1, 0, 1]
def q : ℝ^3 := ![0, -2, 1]
def r : ℝ^3 := ![1, 3, 0]

theorem decomposition_correct :
  ∃ α β γ : ℝ, x = α • p + β • q + γ • r ∧ α = 7 ∧ β = -3 ∧ γ = 1 := by
  sorry

end decomposition_correct_l283_283557


namespace eq_a_2_l283_283973

def f (a x : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

-- Definition of even function: ∀ x, f(-x) = f(x)
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem eq_a_2 (a : ℝ) : (even_function (f a) → a = 2) ∧ (a = 2 → even_function (f a)) :=
by
  sorry

end eq_a_2_l283_283973


namespace eq_a_2_l283_283963

def f (a x : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

-- Definition of even function: ∀ x, f(-x) = f(x)
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem eq_a_2 (a : ℝ) : (even_function (f a) → a = 2) ∧ (a = 2 → even_function (f a)) :=
by
  sorry

end eq_a_2_l283_283963


namespace nth_derivative_at_1_l283_283033

noncomputable def f (x : ℝ) : ℝ := if h : x > 0 then (Real.log x) / x else 0

theorem nth_derivative_at_1 (n : ℕ) :
  (deriv^[n] (λ x, if h : x > 0 then (Real.log x) / x else 0)) 1 = (-1)^(n+1) * Nat.factorial n * (∑ i in Finset.range n.succ, 1 / (i + 1)) :=
by
  sorry

end nth_derivative_at_1_l283_283033


namespace even_function_implies_a_eq_2_l283_283921

def f (x : ℝ) (a : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2 (a : ℝ) (h_even : ∀ x, f (-x) a = f x a) : a = 2 :=
by
  intro x
  sorry

end even_function_implies_a_eq_2_l283_283921


namespace fourier_series_x_plus_one_l283_283239

noncomputable def fourier_series_expansion (x : ℝ) : ℝ :=
1 + 2 * ∑' (n : ℕ) in {n : ℕ | n > 0}, ((-1)^(n-1)) / (n) * (Real.sin (n*x))

theorem fourier_series_x_plus_one (x : ℝ) (h : -π < x ∧ x < π) :
  (x + 1 = fourier_series_expansion x) :=
sorry

end fourier_series_x_plus_one_l283_283239


namespace even_function_implies_a_eq_2_l283_283909

def f (a x : ℝ) : ℝ := (x * exp x) / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2
  (h : ∀ x : ℝ, f a x = f a (-x)) : a = 2 := 
sorry

end even_function_implies_a_eq_2_l283_283909


namespace probability_alex_jamie_paired_l283_283003

theorem probability_alex_jamie_paired :
  let total_students := 50
  let paired_students := 10
  let remaining_students := total_students - 2 * paired_students
  let possible_partners := remaining_students - 1
  in possible_partners = 29 ∧ (1:ℚ) / possible_partners = 1 / 29 :=
by
  -- Definitions from Conditions
  let total_students := 50
  let paired_students := 10
  let remaining_students := total_students - 2 * paired_students
  let possible_partners := remaining_students - 1
  -- Checking the proof statements
  simp [remaining_students, possible_partners]
  exact sorry

end probability_alex_jamie_paired_l283_283003


namespace jordan_running_time_l283_283025

-- Define the conditions given in the problem
variables (time_steve : ℕ) (distance_steve distance_jordan_1 distance_jordan_2 distance_jordan_3 : ℕ)

-- Assign the known values
axiom time_steve_def : time_steve = 24
axiom distance_steve_def : distance_steve = 3
axiom distance_jordan_1_def : distance_jordan_1 = 2
axiom distance_jordan_2_def : distance_jordan_2 = 1
axiom distance_jordan_3_def : distance_jordan_3 = 5

axiom half_time_condition : ∀ t_2, t_2 = time_steve / 2

-- The proof problem
theorem jordan_running_time : ∀ t_j1 t_j2 t_j3, 
  (t_j1 = time_steve / 2 ∧ 
   t_j2 = t_j1 / 2 ∧ 
   t_j3 = t_j2 * 5) →
  t_j3 = 30 := 
by
  intros t_j1 t_j2 t_j3 h
  sorry

end jordan_running_time_l283_283025


namespace grasshoppers_never_meet_l283_283358

theorem grasshoppers_never_meet :
  (∀ i j : ℕ, (i = 0 ∨ i = 2019) ∧ (j = 0 ∨ j = 2020)) →
  (∀ (k : ℕ) (x1 y1 x2 y2 : ℕ), 
     (x1, y1) = if even (k % 2) then (0, 0) else (2019, 2020) ∧ 
     (x2, y2) = if even (k % 2) then (2019, 2020) else (0, 0) ∧
     (abs (x2 - x1) + abs (y2 - y1) = k) →
     (x1 ≠ x2 ∨ y1 ≠ y2)) :=
sorry

end grasshoppers_never_meet_l283_283358


namespace max_good_corners_l283_283012

-- Define the notion of a "good corner" in a 10 x 10 grid.
def is_good_corner (grid : ℕ → ℕ → ℕ) (i j : ℕ) : Prop :=
  (grid i j > grid (i+1) j ∧ grid i j > grid i (j+1))

-- Define the 10 x 10 grid condition.
def grid_condition (grid : ℕ → ℕ → ℕ) : Prop :=
  ∀ i j, 1 ≤ grid i j ∧ grid i j ≤ 100

-- Prove the maximum number of good corners.
theorem max_good_corners : 
  ∀ (grid : ℕ → ℕ → ℕ), grid_condition grid → 
  ∀ (i j : ℕ), (0 ≤ i ∧ i < 9) ∧ (0 ≤ j ∧ j < 9) → 
  (∃ n, 2*n = 162 ∧ ∀ (x y : ℕ), (0 ≤ x ∧ x < 9) ∧ (0 ≤ y ∧ y < 9) → 
  n = (∑ x y in finset.range 9, if is_good_corner grid x y then 1 else 0)). 
Proof :=
by
-- Omitted for brevity, just a theorem statement
sorry

end max_good_corners_l283_283012


namespace general_formula_an_general_formula_bn_exists_arithmetic_sequence_bn_l283_283309

variable (a_n : ℕ → ℝ)
variable (b_n : ℕ → ℝ)
variable (S_n : ℕ → ℝ)
variable (d : ℝ)

-- Define the initial conditions
axiom a2_a3_condition : a_n 2 * a_n 3 = 15
axiom S4_condition : S_n 4 = 16
axiom b_recursion : ∀ (n : ℕ), b_n (n + 1) - b_n n = 1 / (a_n n * a_n (n + 1))

-- Define the proofs
theorem general_formula_an : ∀ (n : ℕ), a_n n = 2 * n - 1 :=
sorry

theorem general_formula_bn : ∀ (n : ℕ), b_n n = (3 * n - 2) / (2 * n - 1) :=
sorry

theorem exists_arithmetic_sequence_bn : ∃ (m n : ℕ), m ≠ n ∧ b_n 2 + b_n n = 2 * b_n m ∧ b_n 2 = 4 / 3 ∧ (n = 8 ∧ m = 3) :=
sorry

end general_formula_an_general_formula_bn_exists_arithmetic_sequence_bn_l283_283309


namespace eq_a_2_l283_283960

def f (a x : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

-- Definition of even function: ∀ x, f(-x) = f(x)
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem eq_a_2 (a : ℝ) : (even_function (f a) → a = 2) ∧ (a = 2 → even_function (f a)) :=
by
  sorry

end eq_a_2_l283_283960


namespace greatest_integer_le_100y_l283_283040

noncomputable def y : ℝ :=
  (∑ n in finset.range 90, real.cos ((n+1 : ℝ) * real.pi / 180)) / 
  (∑ n in finset.range 90, real.sin ((n+1 : ℝ) * real.pi / 180))

theorem greatest_integer_le_100y : ∀ y : ℝ, y = 
  (∑ n in finset.range 90, real.cos ((n+1 : ℝ) * real.pi / 180)) / 
  (∑ n in finset.range 90, real.sin ((n+1 : ℝ) * real.pi / 180)) →
  ⌊100 * y⌋ = 145 :=
by
  sorry

end greatest_integer_le_100y_l283_283040


namespace marla_errand_total_time_l283_283427

theorem marla_errand_total_time :
  let drive_time := 20
  let school_time := 70
  let total_time := 2 * drive_time + school_time
  total_time = 110 :=
by
  let drive_time := 20
  let school_time := 70
  let total_time := 2 * drive_time + school_time
  show total_time = 110
  sorry

end marla_errand_total_time_l283_283427


namespace area_difference_l283_283363

theorem area_difference (T_area : ℝ) (omega_area : ℝ) (H1 : T_area = (25 * Real.sqrt 3) / 4) 
  (H2 : omega_area = 4 * Real.pi) (H3 : 3 * (X - Y) = T_area - omega_area) :
  X - Y = (25 * Real.sqrt 3) / 12 - (4 * Real.pi) / 3 :=
by 
  sorry

end area_difference_l283_283363


namespace Dawn_hourly_earnings_l283_283378

theorem Dawn_hourly_earnings :
  let t_per_painting := 2 
  let num_paintings := 12
  let total_earnings := 3600
  let total_time := t_per_painting * num_paintings
  let hourly_wage := total_earnings / total_time
  hourly_wage = 150 := by
  sorry

end Dawn_hourly_earnings_l283_283378


namespace find_a_of_even_function_l283_283772

noncomputable def f (a x : ℝ) : ℝ := (x * exp x) / (exp (a * x) - 1)

theorem find_a_of_even_function (hf : ∀ x : ℝ, f a (-x) = f a x) : a = 2 :=
by {
    sorry
}

end find_a_of_even_function_l283_283772


namespace marina_max_socks_l283_283126

theorem marina_max_socks (white black : ℕ) (hw : white = 8) (hb : black = 15) :
  ∃ n, n = 17 ∧ ∀ w b, w + b = n → 0 ≤ w ∧ 0 ≤ b ∧ w ≤ black ∧ b ≤ black ∧ w ≤ white ∧ b ≤ black → b > w :=
sorry

end marina_max_socks_l283_283126


namespace second_place_score_is_six_l283_283127

-- Define the conditions
def roundRobinTournament (teams : List ℕ) : Prop :=
  teams.length = 8 ∧
  ∀ i < teams.length, teams.get! i ≤ 7 ∧
  ∀ i j < teams.length, i ≠ j → teams.get! i ≠ teams.get! j ∧
  ∃ S₁ S₂ S₃ S₄ S₅ S₆ S₇ S₈,
    teams = [S₁, S₂, S₃, S₄, S₅, S₆, S₇, S₈] ∧
    (∀ i, 0 ≤ teams.get! i ≤ 7) ∧
    teams.indexOf (teams.max!) = 0 ∧
    S₂ = S₅ + S₆ + S₇ + S₈

-- Prove the statement
theorem second_place_score_is_six (teams : List ℕ) (h : roundRobinTournament teams) : 
  ∃ S, S = 6 ∧ teams.indexOf S = 1 :=
by
  sorry

end second_place_score_is_six_l283_283127


namespace trig_identity_simplify_l283_283467

theorem trig_identity_simplify (x y : ℝ) :
  cos (x + y) * sin y - sin (x + y) * cos y = sin x :=
by
  sorry

end trig_identity_simplify_l283_283467


namespace find_a_for_even_function_l283_283821

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def given_function (a : ℝ) (x : ℝ) : ℝ :=
  x * real.exp x / (real.exp (a * x) - 1)

theorem find_a_for_even_function :
  ∀ (a : ℝ), is_even_function (given_function a) ↔ a = 2 :=
by 
  intro a
  sorry

end find_a_for_even_function_l283_283821


namespace largest_prime_factor_of_3136_l283_283528

theorem largest_prime_factor_of_3136 : ∃ p, p.prime ∧ p ∣ 3136 ∧ ∀ q, q.prime ∧ q ∣ 3136 → q ≤ p :=
sorry

end largest_prime_factor_of_3136_l283_283528


namespace eq_a_2_l283_283968

def f (a x : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

-- Definition of even function: ∀ x, f(-x) = f(x)
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem eq_a_2 (a : ℝ) : (even_function (f a) → a = 2) ∧ (a = 2 → even_function (f a)) :=
by
  sorry

end eq_a_2_l283_283968


namespace FerrisWheelCostIsTwo_l283_283157

noncomputable def costFerrisWheel (rollerCoasterCost multipleRideDiscount coupon totalTicketsBought : ℝ) : ℝ :=
  totalTicketsBought + multipleRideDiscount + coupon - rollerCoasterCost

theorem FerrisWheelCostIsTwo :
  let rollerCoasterCost := 7.0
  let multipleRideDiscount := 1.0
  let coupon := 1.0
  let totalTicketsBought := 7.0
  costFerrisWheel rollerCoasterCost multipleRideDiscount coupon totalTicketsBought = 2.0 :=
by
  sorry

end FerrisWheelCostIsTwo_l283_283157


namespace largest_prime_factor_3136_l283_283532

theorem largest_prime_factor_3136 : ∀ (n : ℕ), n = 3136 → ∃ p : ℕ, Prime p ∧ (p ∣ n) ∧ ∀ q : ℕ, (Prime q ∧ q ∣ n) → p ≥ q :=
by {
  sorry
}

end largest_prime_factor_3136_l283_283532


namespace even_function_implies_a_eq_2_l283_283852

theorem even_function_implies_a_eq_2 (a : ℝ) 
  (h : ∀ x : ℝ, f x = f (-x)) 
  (f : ℝ → ℝ := λ x, (x * exp x) / (exp (a * x) - 1)) : a = 2 :=
sorry

end even_function_implies_a_eq_2_l283_283852


namespace trig_identity_simplify_l283_283468

theorem trig_identity_simplify (x y : ℝ) :
  cos (x + y) * sin y - sin (x + y) * cos y = sin x :=
by
  sorry

end trig_identity_simplify_l283_283468


namespace min_height_of_prism_l283_283577

/-- 
Given a rectangular prism of dimensions 4 × 4 × h that fits 8 small spheres each 
with radius 1 and 1 large sphere with radius 2, prove that the minimum value of h 
is 2 + 2 * sqrt 7.
-/
theorem min_height_of_prism (h : ℝ) : 
  (∃ h, 0 < h ∧ 
    (∀ (s : ℝ) (r1 r2 : ℝ), s = 1 ∧ r2 = 2 → 
    let n : ℕ := 8 in 
    n * r1 * r1 * r1 < 4 * 4 * h ∧  r2 * r2 * r2 < 4 * 4 * h)) → 
  h = 2 + 2 * Real.sqrt 7 := sorry

end min_height_of_prism_l283_283577


namespace no_sum_of_cubes_eq_2002_l283_283016

theorem no_sum_of_cubes_eq_2002 :
  ¬ ∃ (a b c : ℕ), (a ^ 3 + b ^ 3 + c ^ 3 = 2002) :=
sorry

end no_sum_of_cubes_eq_2002_l283_283016


namespace smallest_b_such_that_x4_plus_b2_composite_l283_283270

theorem smallest_b_such_that_x4_plus_b2_composite :
  ∃ (b : ℕ), (b > 0) ∧ (∀ x : ℤ, ¬ Nat.Prime (x^4 + b^2)) ∧ (∀ b' : ℕ, (b' > 0) → (b' < b) → (∃ x : ℤ, Nat.Prime (x^4 + b'^2))) :=
by
  sorry

end smallest_b_such_that_x4_plus_b2_composite_l283_283270


namespace Phone_Bill_October_Phone_Bill_November_December_Extra_Cost_November_December_l283_283558

/-- Definitions for phone plans A and B and phone call durations -/
def fixed_cost_A : ℕ := 18
def free_minutes_A : ℕ := 1500
def price_per_minute_A : ℕ → ℚ := λ t => 0.1 * t

def fixed_cost_B : ℕ := 38
def free_minutes_B : ℕ := 4000
def price_per_minute_B : ℕ → ℚ := λ t => 0.07 * t

def call_duration_October : ℕ := 2600
def total_bill_November_December : ℚ := 176
def total_call_duration_November_December : ℕ := 5200

/-- Problem statements to be proven -/

theorem Phone_Bill_October : 
  fixed_cost_A + price_per_minute_A (call_duration_October - free_minutes_A) = 128 :=
  sorry

theorem Phone_Bill_November_December (x : ℕ) (h1 : 0 ≤ x) (h2 : x ≤ total_call_duration_November_December) : 
  let bill_November := fixed_cost_A + price_per_minute_A (x - free_minutes_A)
  let bill_December := fixed_cost_B + price_per_minute_B (total_call_duration_November_December - x - free_minutes_B)
  bill_November + bill_December = total_bill_November_December :=
  sorry
  
theorem Extra_Cost_November_December :
  let actual_cost := 138 + 38
  let hypothetical_cost := fixed_cost_A + price_per_minute_A (total_call_duration_November_December - free_minutes_A)
  hypothetical_cost - actual_cost = 80 :=
  sorry

end Phone_Bill_October_Phone_Bill_November_December_Extra_Cost_November_December_l283_283558


namespace jan_total_payment_l283_283388

theorem jan_total_payment (roses_per_dozen : ℕ) (dozens : ℕ) (cost_per_rose : ℕ) (discount_rate : ℝ) : ℕ :=
  let total_roses := dozens * roses_per_dozen in
  let total_cost := total_roses * cost_per_rose in
  let discounted_cost := (total_cost : ℝ) * discount_rate in
  total_cost * (discount_rate.to_real : ℝ).to_nat

example : jan_total_payment 12 5 6 0.8 = 288 := by
  sorry

end jan_total_payment_l283_283388


namespace geometric_sequence_sum_l283_283225

noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 0 then 1
  else if n = 1 then 1 / 2
  else (sequence (n - 1) ^ 2) / (sequence (n - 2))

theorem geometric_sequence_sum (n : ℕ) :
  let a := sequence in
  a 1 * a 3 + a 2 * a 4 + ∑ i in ((fin.range (n + 1)) : finset ℕ), a (i + 2) * a (i + 4) = 
  (1 / 3) * (1 - 1 / (4^n)) :=
sorry

end geometric_sequence_sum_l283_283225


namespace sin_vertex_angle_isosceles_triangle_l283_283498

theorem sin_vertex_angle_isosceles_triangle (α β : ℝ) (h_isosceles : β = 2 * α) (tan_base_angle : Real.tan α = 2 / 3) :
  Real.sin β = 12 / 13 := 
sorry

end sin_vertex_angle_isosceles_triangle_l283_283498


namespace even_function_implies_a_eq_2_l283_283848

theorem even_function_implies_a_eq_2 (a : ℝ) 
  (h : ∀ x : ℝ, f x = f (-x)) 
  (f : ℝ → ℝ := λ x, (x * exp x) / (exp (a * x) - 1)) : a = 2 :=
sorry

end even_function_implies_a_eq_2_l283_283848


namespace percentage_slump_in_business_l283_283098

theorem percentage_slump_in_business (X Y : ℝ) (h1 : 0.04 * X = 0.05 * Y) : 
  (1 - Y / X) * 100 = 20 :=
by
  sorry

end percentage_slump_in_business_l283_283098


namespace find_a_for_even_function_l283_283814

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def given_function (a : ℝ) (x : ℝ) : ℝ :=
  x * real.exp x / (real.exp (a * x) - 1)

theorem find_a_for_even_function :
  ∀ (a : ℝ), is_even_function (given_function a) ↔ a = 2 :=
by 
  intro a
  sorry

end find_a_for_even_function_l283_283814


namespace even_function_implies_a_eq_2_l283_283902

def f (a x : ℝ) : ℝ := (x * exp x) / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2
  (h : ∀ x : ℝ, f a x = f a (-x)) : a = 2 := 
sorry

end even_function_implies_a_eq_2_l283_283902


namespace no_other_way_five_triangles_l283_283702

def is_triangle {α : Type*} [linear_ordered_field α] (a b c : α) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem no_other_way_five_triangles (sticks : list ℝ)
  (h_len : sticks.length = 15)
  (h_sets : ∃ (sets : list (ℝ × ℝ × ℝ)), 
    sets.length = 5 ∧ ∀ {a b c}, (a, b, c) ∈ sets → is_triangle a b c) :
  ∀ sets, sets.length = 5 ∧ ∀ {a b c}, (a, b, c) ∈ sets → is_triangle a b c → sets = h_sets :=
begin
  sorry
end

end no_other_way_five_triangles_l283_283702


namespace operation_result_l283_283699

def star (a b c : ℝ) : ℝ := (a + b + c) ^ 2

theorem operation_result (x : ℝ) : star (x - 1) (1 - x) 1 = 1 := 
by
  sorry

end operation_result_l283_283699


namespace min_bounces_for_height_less_than_two_l283_283579

theorem min_bounces_for_height_less_than_two (k : ℕ) :
  (∀ n < k, 800 * (3 / 4 : ℝ)^n ≥ 2) ∧ (800 * (3 / 4 : ℝ)^k < 2) ↔ k = 21 :=
begin
  sorry
end

end min_bounces_for_height_less_than_two_l283_283579


namespace even_function_implies_a_eq_2_l283_283876

def f (x a : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2 (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → a = 2 :=
by
  sorry

end even_function_implies_a_eq_2_l283_283876


namespace even_function_implies_a_eq_2_l283_283898

def f (a x : ℝ) : ℝ := (x * exp x) / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2
  (h : ∀ x : ℝ, f a x = f a (-x)) : a = 2 := 
sorry

end even_function_implies_a_eq_2_l283_283898


namespace compare_abc_l283_283279

noncomputable theory

def a : ℝ := 0.5 ^ 0.6
def b : ℝ := 0.6 ^ 0.5
def c : ℝ := Real.log 5 / Real.log 6  -- log_6 5 can be written using natural logs

theorem compare_abc : a < b ∧ b < c := by
  sorry

end compare_abc_l283_283279


namespace fans_per_bleacher_l283_283050

theorem fans_per_bleacher 
  (total_fans : ℕ) 
  (sets_of_bleachers : ℕ) 
  (h_total : total_fans = 2436) 
  (h_sets : sets_of_bleachers = 3) : 
  total_fans / sets_of_bleachers = 812 := 
by 
  sorry

end fans_per_bleacher_l283_283050


namespace expression_evaluation_l283_283238

theorem expression_evaluation : 
  (81 ^ (1 / 4 - 1 / (log 4 / log 9)) + 25 ^ (log 8 / log 125)) * 49 ^ (log 2 / log 7) = 19 := 
by
  sorry

end expression_evaluation_l283_283238


namespace cylinder_surface_area_l283_283588

-- Define the necessary parameters based on the conditions
def height : ℝ := 12
def radius : ℝ := 5

-- The formula to calculate the total surface area of a cylinder
def total_surface_area (h r : ℝ) : ℝ := 2 * Real.pi * r * (r + h)

-- The goal is to prove that this total surface area is 170π for given h and r
theorem cylinder_surface_area : total_surface_area height radius = 170 * Real.pi := 
by 
  -- Proof is left as an exercise.
  sorry

end cylinder_surface_area_l283_283588


namespace summation_is_ratio_of_relatively_prime_integers_final_answer_is_37_l283_283121

theorem summation_is_ratio_of_relatively_prime_integers : 
  (∑ k in Finset.range 361 \ Finset.singleton 0, (1 : ℝ) / (k * real.sqrt (k + 1) + (k + 1) * real.sqrt k)) = 18/19 := sorry

theorem final_answer_is_37 : 
  let m := 18 in
  let n := 19 in
  m + n = 37 := by
  rfl

end summation_is_ratio_of_relatively_prime_integers_final_answer_is_37_l283_283121


namespace find_a_even_function_l283_283982

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def given_function (a : ℝ) (x : ℝ) : ℝ :=
  x * Real.exp x / (Real.exp (a * x) - 1)

theorem find_a_even_function :
  (is_even_function (given_function 2)) → true :=
by
  intro h
  sorry

end find_a_even_function_l283_283982


namespace elements_not_in_A_or_B_l283_283501

def U := Finset ℕ
def A := Finset ℕ
def B := Finset ℕ

variables (u : U) (a : A) (b : B)

-- Conditions
axiom hU : u.card = 193
axiom hA : a.card = 116
axiom hB : b.card = 41
axiom hAB : (a ∩ b).card = 23

-- Question and answer proof statement
theorem elements_not_in_A_or_B : (u \ (a ∪ b)).card = 59 :=
by
  -- Proof omitted, using sorry as a placeholder
  sorry

end elements_not_in_A_or_B_l283_283501


namespace twentieth_common_number_l283_283326

theorem twentieth_common_number : 
  (∃ (m n : ℤ), (4 * m - 1) = (3 * n + 2) ∧ 20 * 12 - 1 = 239) := 
by
  sorry

end twentieth_common_number_l283_283326


namespace distance_center_to_chord_half_length_other_chord_l283_283058

theorem distance_center_to_chord_half_length_other_chord 
  (A B C D O : Point)
  (circle : IsInscribed (Quadrilateral.mk A B C D) O)
  (perpendicular_diagonals : Perpendicular (Segment.mk A C) (Segment.mk B D)) :
  distance_to_chord O (Segment.mk A D) = 0.5 * length (Segment.mk B C) :=
by
  sorry

end distance_center_to_chord_half_length_other_chord_l283_283058


namespace num_students_in_scientific_notation_l283_283660

-- Define the number of students in scientific notation problem
def num_students : ℕ := 77431000

-- The scientific notation form
def scientific_notation := 7.7431 * 10^7

-- Statement to prove the equivalence
theorem num_students_in_scientific_notation : (num_students : ℝ) = scientific_notation :=
sorry

end num_students_in_scientific_notation_l283_283660


namespace log_b_2024_l283_283668

noncomputable def op_plus (a b : ℝ) : ℝ := a^(Real.log b / Real.log 5)
noncomputable def op_times (a b : ℝ) : ℝ := a^(1 / (Real.log b / Real.log 5))

def b_sequence : ℕ → ℝ
| 3 := op_times 4 3
| (n+1) := if h : n+1 ≥ 4 then
  op_plus (op_times (n+1) n) (b_sequence n) 
else 
  0

theorem log_b_2024 : abs (Real.log (b_sequence 2024) / Real.log 5 - 4) < 1 :=
sorry

end log_b_2024_l283_283668


namespace train_length_theorem_l283_283185

noncomputable def train_length (speed_kmph : ℕ) (platform_length : ℕ) (crossing_time : ℕ) : ℕ :=
  let speed_mps := speed_kmph * 5 / 18
  let distance := speed_mps * crossing_time
  distance - platform_length

theorem train_length_theorem : train_length 72 210 26 = 310 :=
by 
  unfold train_length
  norm_num
  sorry  -- Detailed proof steps omitted

end train_length_theorem_l283_283185


namespace tournament_participants_l283_283584

theorem tournament_participants (x : ℕ) (h1 : ∀ g b : ℕ, g = 2 * b)
  (h2 : ∀ p : ℕ, p = 3 * x) 
  (h3 : ∀ G B : ℕ, G + B = (3 * x * (3 * x - 1)) / 2)
  (h4 : ∀ G B : ℕ, G / B = 7 / 9) 
  (h5 : x = 11) :
  3 * x = 33 :=
by
  sorry

end tournament_participants_l283_283584


namespace parallelogram_base_length_l283_283689

theorem parallelogram_base_length
  (height : ℝ) (area : ℝ) (base : ℝ) 
  (h1 : height = 18) 
  (h2 : area = 576) 
  (h3 : area = base * height) : 
  base = 32 :=
by
  rw [h1, h2] at h3
  sorry

end parallelogram_base_length_l283_283689


namespace domain_of_sqrt_tan_eq_1_l283_283480

theorem domain_of_sqrt_tan_eq_1 (x : ℝ) : 
  (∃ k ∈ ℤ, (π / 4) + k * π ≤ x ∧ x < (π / 2) + k * π) ↔ (tan x - 1 ≥ 0) :=
sorry

end domain_of_sqrt_tan_eq_1_l283_283480


namespace find_n_with_10_digit_sum_l283_283264

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem find_n_with_10_digit_sum : ∃! n, (1 ≤ n) ∧ (n ≤ 27) ∧ (∃ finset.filter (λ x, digit_sum x = n) (finset.range 1001)).card = 10 := by
sorry

end find_n_with_10_digit_sum_l283_283264


namespace regular_polygon_sides_l283_283604

theorem regular_polygon_sides (n : ℕ) (h : 1 < n) (exterior_angle : ℝ) (h_ext : exterior_angle = 18) :
  n * exterior_angle = 360 → n = 20 :=
by 
  sorry

end regular_polygon_sides_l283_283604


namespace owen_turtles_l283_283443

theorem owen_turtles (o_initial : ℕ) (j_initial : ℕ) (o_after_month : ℕ) (j_remaining : ℕ) (o_final : ℕ) 
  (h1 : o_initial = 21)
  (h2 : j_initial = o_initial - 5)
  (h3 : o_after_month = 2 * o_initial)
  (h4 : j_remaining = j_initial / 2)
  (h5 : o_final = o_after_month + j_remaining) :
  o_final = 50 :=
sorry

end owen_turtles_l283_283443


namespace rotated_triangle_surface_area_volume_l283_283013

theorem rotated_triangle_surface_area_volume :
  ∀ (A B C : Type) [Triangle A B C],
  ∠ACB = 15 * (π / 180) ∧ ∠CBA = 120 * (π / 180) ∧ AB = 1 →
  surface_area (rotate_triangle_around_side AB A B C) ≈ 45.17 ∧
  volume (rotate_triangle_around_side AB A B C) ≈ 5.86 := 
by
  intro A B C,
  intro htriangle hangle_cond hab_cond,
  sorry

end rotated_triangle_surface_area_volume_l283_283013


namespace inequality1_solution_inequality2_solution_l283_283271

variables (x a : ℝ)

theorem inequality1_solution : (∀ x : ℝ, (2 * x) / (x + 1) < 1 ↔ -1 < x ∧ x < 1) :=
by
  sorry

theorem inequality2_solution (a : ℝ) : 
  (∀ x : ℝ, x^2 + (2 - a) * x - 2 * a ≥ 0 ↔ 
    (a = -2 → true) ∧ 
    (a > -2 → (x ≤ -2 ∨ x ≥ a)) ∧ 
    (a < -2 → (x ≤ a ∨ x ≥ -2))) :=
by
  sorry

end inequality1_solution_inequality2_solution_l283_283271


namespace bijective_iff_gcd_condition_divisibility_theorem_l283_283398

namespace BijectiveFunctionProof

variables {n a : ℕ} (M : fin n) (f_a : M → M)

-- let n be a positive integer, n >= 2
def valid_n : Prop := n ≥ 2

-- definition of f_a
def f_a_def (x : M) : M := ⟨a * x.val % n, sorry⟩

-- gcd(a, n) = 1
def gcd_condition : Prop := Nat.gcd a n = 1

-- f_a is bijective if and only if gcd(a, n) = 1
theorem bijective_iff_gcd_condition : (Function.Bijective f_a) ↔ gcd_condition := sorry

-- if f_a is bijective and n is a prime number, then a^(n(n-1)) - 1 is divisible by n^2
theorem divisibility_theorem (hn_prime : Nat.Prime n) (h_bij : Function.Bijective f_a) :
  n^2 ∣ a^(n * (n - 1)) - 1 := sorry

end BijectiveFunctionProof

end bijective_iff_gcd_condition_divisibility_theorem_l283_283398


namespace arctan_sum_l283_283221

theorem arctan_sum : 
  ∃ (θ φ: ℝ), θ = Real.arctan (2 / 5) ∧ φ = Real.arctan (5 / 2) ∧ (θ + φ = π / 2) :=
begin
  sorry
end

end arctan_sum_l283_283221


namespace transformation_matrix_l283_283533

open Real
open Matrix

def R_60 : Matrix (Fin 2) (Fin 2) ℝ :=
  !![(1/2 : ℝ), -(sqrt 3 / 2); (sqrt 3 / 2), (1/2)]

def S_2 : Matrix (Fin 2) (Fin 2) ℝ :=
  !![2, 0; 0, 2]

theorem transformation_matrix :
  (S_2 ⬝ R_60) = !![1, -sqrt 3; sqrt 3, 1] :=
by
  sorry

end transformation_matrix_l283_283533


namespace value_of_k_l283_283275

theorem value_of_k (k : ℝ) : (2 - k * 2 = -4 * (-1)) → k = -1 :=
by
  intro h
  sorry

end value_of_k_l283_283275


namespace cube_root_simplification_l283_283143

theorem cube_root_simplification {a b : ℕ} (h : (a * b^(1/3) : ℝ) = (2450 : ℝ)^(1/3)) 
  (a_pos : 0 < a) (b_pos : 0 < b) (h_smallest : ∀ b', 0 < b' → (∃ a', (a' * b'^(1/3) : ℝ) = (2450 : ℝ)^(1/3) → b ≤ b')) :
  a + b = 37 := 
sorry

end cube_root_simplification_l283_283143


namespace even_function_implies_a_eq_2_l283_283930

theorem even_function_implies_a_eq_2 (a : ℝ) : 
  (∀ x, (f : ℝ → ℝ) x = (λ x : ℝ, x * exp x / (exp (a * x) - 1)) x -> (f (-x) = f x)) -> a = 2 :=
by
  sorry

end even_function_implies_a_eq_2_l283_283930


namespace range_of_f_l283_283108

-- Define the function f
def f (x : ℝ) : ℝ := 1 - (sin x)^2 - 2 * sin x

-- State the theorem about the range of the function f
theorem range_of_f : ∀ y, (∃ x, f x = y) ↔ -2 ≤ y ∧ y ≤ 2 :=
by sorry

end range_of_f_l283_283108


namespace units_digit_F500_is_7_l283_283052

def F (n : ℕ) : ℕ := 2 ^ (2 ^ (2 * n)) + 1

theorem units_digit_F500_is_7 : (F 500) % 10 = 7 := 
  sorry

end units_digit_F500_is_7_l283_283052


namespace magnitude_of_complex_l283_283236

open Complex

theorem magnitude_of_complex : abs (Complex.mk (3/4) (-5/6)) = Real.sqrt (181) / 12 :=
by
  sorry

end magnitude_of_complex_l283_283236


namespace find_a_if_f_even_l283_283835

def f (a x : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem find_a_if_f_even (a : ℝ) (h : ∀ x : ℝ, x ≠ 0 → f a (-x) = f a x) : a = 2 :=
by 
  sorry

end find_a_if_f_even_l283_283835


namespace simplify_ellipse_eq_l283_283066

theorem simplify_ellipse_eq (x y : ℝ) :
  sqrt ((x - 4)^2 + y^2) + sqrt ((x + 4)^2 + y^2) = 10 → (x^2 / 25) + (y^2 / 9) = 1 :=
by
  sorry

end simplify_ellipse_eq_l283_283066


namespace encoded_integer_one_less_l283_283179

theorem encoded_integer_one_less (BDF BEA BFB EAB : ℕ)
  (hBDF : BDF = 1 * 7^2 + 3 * 7 + 6)
  (hBEA : BEA = 1 * 7^2 + 5 * 7 + 0)
  (hBFB : BFB = 1 * 7^2 + 5 * 7 + 1)
  (hEAB : EAB = 5 * 7^2 + 0 * 7 + 1)
  : EAB - 1 = 245 :=
by
  sorry

end encoded_integer_one_less_l283_283179


namespace largest_prime_factor_3136_l283_283530

theorem largest_prime_factor_3136 : ∀ (n : ℕ), n = 3136 → ∃ p : ℕ, Prime p ∧ (p ∣ n) ∧ ∀ q : ℕ, (Prime q ∧ q ∣ n) → p ≥ q :=
by {
  sorry
}

end largest_prime_factor_3136_l283_283530


namespace quadrilateral_exists_l283_283320

noncomputable theory

variables (a b c d f : ℝ)

-- Given the conditions 
def condition_1 : Prop := (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ f > 0)
def condition_2 : Prop := (b + d ≥ 2 * f)

-- Prove that there exists a quadrilateral with consecutive sides a, b, c, d where
-- the length of the line segment connecting midpoints of a and c is f
theorem quadrilateral_exists (h1 : condition_1 a b c d f) (h2 : condition_2 b d f) :
  ∃ A B C D : ℝ × ℝ, 
    dist A B = a ∧ 
    dist B C = b ∧ 
    dist C D = c ∧ 
    dist D A = d ∧ 
    dist ((A + B) / 2) ((C + D) / 2) = f := 
sorry

end quadrilateral_exists_l283_283320


namespace total_computers_needed_l283_283154

theorem total_computers_needed
    (initial_students : ℕ)
    (students_per_computer : ℕ)
    (additional_students : ℕ)
    (initial_computers : ℕ := initial_students / students_per_computer)
    (total_computers : ℕ := initial_computers + (additional_students / students_per_computer))
    (h1 : initial_students = 82)
    (h2 : students_per_computer = 2)
    (h3 : additional_students = 16) :
    total_computers = 49 :=
by
  -- The proof would normally go here
  sorry

end total_computers_needed_l283_283154


namespace even_function_implies_a_is_2_l283_283788

noncomputable def f (a x : ℝ) : ℝ := (x * Real.exp x) / (Real.exp (a * x) - 1)

theorem even_function_implies_a_is_2 (a : ℝ) 
  (h : ∀ x : ℝ, f a x = f a (-x)) : a = 2 := 
by
  sorry

end even_function_implies_a_is_2_l283_283788


namespace factorial_fraction_is_integer_l283_283301

open Nat

theorem factorial_fraction_is_integer (m n : ℕ) : 
  ↑((factorial (2 * m)) * (factorial (2 * n))) % (factorial m * factorial n * factorial (m + n)) = 0 := sorry

end factorial_fraction_is_integer_l283_283301


namespace chef_wage_increase_percentage_l283_283649

theorem chef_wage_increase_percentage :
  ∀ (manager_wage chef_wage dishwasher_wage : ℝ),
    manager_wage = 7.50 →
    dishwasher_wage = manager_wage / 2 →
    chef_wage = manager_wage - 3 →
    ((chef_wage - dishwasher_wage) / dishwasher_wage) * 100 = 20 :=
by
  intros manager_wage chef_wage dishwasher_wage
  intros h_manager_wage h_dishwasher_wage h_chef_wage
  have h1 : chef_wage - dishwasher_wage = 4.50 - 3.75, from sorry
  have h2 : (chef_wage - dishwasher_wage) / dishwasher_wage = (0.75 / 3.75), from sorry
  have h3 : (0.75 / 3.75) * 100 = 20, from sorry
  exact h3

end chef_wage_increase_percentage_l283_283649


namespace odd_cardinality_subsets_l283_283138

theorem odd_cardinality_subsets (n : ℕ) : 
  (∃ I_n P_n : ℕ, I_n + P_n = 2^n ∧ P_n - I_n = 0 ∧ I_n = 2^(n-1)) :=
by 
  use 2 ^ (n - 1), 2 ^ (n - 1),
  split;
  {
    exact (2 ^ n = 2 ^ (n - 1) + 2 ^ (n - 1)),
    sorry,
    exact (2 ^ (n - 1) = 2 ^ (n - 1)),
  }

end odd_cardinality_subsets_l283_283138


namespace interval_of_decrease_l283_283671

noncomputable def f (x : ℝ) : ℝ := (x^2 - 3) * Real.exp x

theorem interval_of_decrease : Ioc -3 1 ⊆ { x : ℝ | (deriv f x) < 0 } :=
sorry

end interval_of_decrease_l283_283671
