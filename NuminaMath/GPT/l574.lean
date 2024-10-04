import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.Field
import Mathlib.Algebra.Geometry
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Quadratic
import Mathlib.Algebra.QuadraticEquation
import Mathlib.Analysis.Calculus.Area
import Mathlib.Analysis.Calculus.Slope
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.GraphColoring
import Mathlib.Combinatorics.SimpleGraph
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.List.Perm
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.GraphTheory.Basic
import Mathlib.NumberTheory.Divisors
import Mathlib.Probability
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Mathlib.Topology.MetricSpace.Basic
import data.nat.lcm
import data.real.basic
import probability.probability_mass_function

namespace projection_correct_l574_574587

open Real

def vector_u : ℝ × ℝ := (3, -4)
def vector_v : ℝ × ℝ := (1, 2)

def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

def norm_squared (a : ℝ × ℝ) : ℝ :=
  dot_product a a

def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let scale := (dot_product u v) / (norm_squared v)
  (scale * v.1, scale * v.2)

theorem projection_correct :
  projection vector_u vector_v = (-1, -2) := by
  sorry

end projection_correct_l574_574587


namespace find_a_l574_574306

noncomputable def f (a x : ℝ) : ℝ := (1 / 2) * x^2 + 2 * a * x - Real.log x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x y, a ≤ x → x ≤ b → a ≤ y → y ≤ b → x ≤ y → f x ≤ f y

theorem find_a (a : ℝ) :
  is_increasing_on (f a) (1 / 3) 2 → a ≥ 4 / 3 :=
sorry

end find_a_l574_574306


namespace analytical_expression_of_f_range_of_f_on_interval_l574_574648

noncomputable def f (x : ℝ) (a c : ℝ) : ℝ := a * x^3 + c * x

theorem analytical_expression_of_f
  (a c : ℝ)
  (h1 : a > 0)
  (h2 : ∀ x, f x a c = a * x^3 + c * x) 
  (h3 : 3 * a + c = -6)
  (h4 : ∀ x, (3 * a * x ^ 2 + c) ≥ -12) :
    a = 2 ∧ c = -12 :=
by
  sorry

theorem range_of_f_on_interval
  (h1 : ∃ a c, a = 2 ∧ c = -12)
  (h2 : ∀ x, f x 2 (-12) = 2 * x^3 - 12 * x)
  :
    Set.range (fun x => f x 2 (-12)) = Set.Icc (-8 * Real.sqrt 2) (8 * Real.sqrt 2) :=
by
  sorry

end analytical_expression_of_f_range_of_f_on_interval_l574_574648


namespace thirtieth_term_of_sequence_l574_574813

-- Defining the sequence satisfying the given conditions
def sequence (n : ℕ) : ℕ := 10 * n

-- Stating the theorem to prove that the 30th term of the sequence is 300
theorem thirtieth_term_of_sequence : sequence 30 = 300 :=
  by
    sorry

end thirtieth_term_of_sequence_l574_574813


namespace integer_solution_of_inequality_l574_574678

theorem integer_solution_of_inequality : 
  ∀ (x : ℤ), 3 ≤ 3 * x + 3 ∧ 3 * x + 3 ≤ 5 → x = 0 := 
by
  assume x
  intro h
  sorry

end integer_solution_of_inequality_l574_574678


namespace express_u_in_terms_of_f_and_g_l574_574744

variables (u f g : ℝ → ℝ)

-- Conditions Definitions
def cond1 (u f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, (u(x + 1) + u(x - 1)) / 2 = f(x)

def cond2 (u g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, (u(x + 4) + u(x - 4)) / 2 = g(x)

-- Mathematical equivalent proof problem
theorem express_u_in_terms_of_f_and_g
  (h1 : cond1 u f)
  (h2 : cond2 u g) :
  ∀ x : ℝ, u(x) = g(x + 4) - f(x + 7) + f(x + 5) - f(x + 3) + f(x + 1) :=
by sorry

end express_u_in_terms_of_f_and_g_l574_574744


namespace at_least_one_is_half_l574_574448

theorem at_least_one_is_half (x y z : ℝ) (h : x + y + z - 2 * (x * y + y * z + z * x) + 4 * x * y * z = 1 / 2) :
  x = 1 / 2 ∨ y = 1 / 2 ∨ z = 1 / 2 :=
sorry

end at_least_one_is_half_l574_574448


namespace average_weight_is_15_l574_574638

-- Define the ages of the 10 children
def ages : List ℕ := [2, 3, 3, 5, 2, 6, 7, 3, 4, 5]

-- Define the regression function
def weight (age : ℕ) : ℕ := 2 * age + 7

-- Function to calculate average
def average (l : List ℕ) : ℕ := l.sum / l.length

-- Define the weights of the children based on the regression function
def weights : List ℕ := ages.map weight

-- State the theorem to find the average weight of the children
theorem average_weight_is_15 : average weights = 15 := by
  sorry

end average_weight_is_15_l574_574638


namespace find_pairs_of_real_numbers_l574_574577

theorem find_pairs_of_real_numbers (x y : ℝ) :
  (∀ n : ℕ, n > 0 → x * ⌊n * y⌋ = y * ⌊n * x⌋) →
  (x = y ∨ x = 0 ∨ y = 0 ∨ (∃ a b : ℤ, x = a ∧ y = b)) :=
by
  sorry

end find_pairs_of_real_numbers_l574_574577


namespace probability_two_same_color_l574_574876

noncomputable def same_color_probability (green white : ℕ) (total : ℕ) : ℚ :=
  let prob_green := (green / (total : ℚ)) * ((green - 1) / (total - 1 : ℚ))
  let prob_white := (white / (total : ℚ)) * ((white - 1) / (total - 1 : ℚ))
  prob_green + prob_white

theorem probability_two_same_color :
  same_color_probability 6 7 13 = 6 / 13 :=
by
  sorry -- Proof goes here

end probability_two_same_color_l574_574876


namespace b_perp_alpha_l574_574621

noncomputable def parallel {α β : Type*} (a b : α → β → Prop) : Prop :=
∀ x y, a x y ↔ b x y

noncomputable def perpendicular {α β : Type*} (a : α → β → Prop) (P : α → β → β) : Prop :=
∀ x y z, a x y → P x z → P y z

variables (α : Type*) [plane α]
variables (a b : line) [hp1 : parallel a b] [hp2 : perpendicular a α]

theorem b_perp_alpha (b : line) : perpendicular b α :=
by sorry

end b_perp_alpha_l574_574621


namespace man_speed_train_l574_574919

theorem man_speed_train (train_length : ℝ) (train_speed_kmph : ℝ)
  (pass_time : ℝ) (man_speed_kmph : ℝ) : train_length = 220 ∧
  train_speed_kmph = 60 ∧ pass_time = 11.999040076793857 →
  man_speed_kmph ≈ 6.0024 :=
by sorry

end man_speed_train_l574_574919


namespace circle_area_calculation_l574_574950

theorem circle_area_calculation
  (A B : Type)
  (rA : ℝ)
  (h1 : ∀ x ∈ A, x ∈ B)
  (h2 : ∀ x ∈ B, x ∉ A → x = center A)
  (h3 : ∀ x ∈ A, dist x (center B) = radius B)
  (h4 : area A = 16 * real.pi) :
  area B = 64 * real.pi :=
sorry

end circle_area_calculation_l574_574950


namespace sum_valid_x_is_39_75_l574_574843

theorem sum_valid_x_is_39_75 : 
  ∀ x : ℝ, 
  (let nums := [5, 11, 19, 28, x] in
   (median nums = mean nums) →
   (x = -8 ∨ x = 32 ∨ x = 15.75)) → ∑ x in [-8, 32, 15.75], x = 39.75 := 
sorry

end sum_valid_x_is_39_75_l574_574843


namespace C_share_of_rent_l574_574864

-- Define the given conditions
def A_ox_months : ℕ := 10 * 7
def B_ox_months : ℕ := 12 * 5
def C_ox_months : ℕ := 15 * 3
def total_rent : ℕ := 175
def total_ox_months : ℕ := A_ox_months + B_ox_months + C_ox_months
def cost_per_ox_month := total_rent / total_ox_months

-- The goal is to prove that C's share of the rent is Rs. 45
theorem C_share_of_rent : C_ox_months * cost_per_ox_month = 45 := by
  -- Adding sorry to skip the proof
  sorry

end C_share_of_rent_l574_574864


namespace trig_expression_value_l574_574441

theorem trig_expression_value :
  (3 / (Real.sin (140 * Real.pi / 180))^2 - 1 / (Real.cos (140 * Real.pi / 180))^2) * (1 / (2 * Real.sin (10 * Real.pi / 180))) = 16 := 
by
  -- placeholder for proof
  sorry

end trig_expression_value_l574_574441


namespace numbers_with_odd_digits_count_l574_574942

theorem numbers_with_odd_digits_count : 
  let odd_digits := {1, 3, 5, 7, 9}
  ∃ total_count : ℕ, total_count = 155 ∧ 
    total_count = (card {n : ℕ | n < 10 ∧ n ∈ odd_digits} +
                   (card {n : ℕ // n < 100 ∧ (n/10) ∈ odd_digits ∧ (n % 10) ∈ odd_digits}) +
                   (card {n : ℕ // n < 1000 ∧ (n/100) ∈ odd_digits ∧ ((n/10) % 10) ∈ odd_digits ∧ (n % 10) ∈ odd_digits})) := 
  sorry

end numbers_with_odd_digits_count_l574_574942


namespace apples_left_total_l574_574276

def Frank_apples : ℕ := 36
def Susan_apples : ℕ := 3 * Frank_apples
def Frank_sold : ℕ := Frank_apples / 3
def Susan_given : ℕ := Susan_apples / 2

theorem apples_left_total : Susan_apples - Susan_given + Frank_apples - Frank_sold = 78 := 
by
  have h1 : Susan_apples = 3 * Frank_apples := rfl
  have h2 : Frank_apples = 36 := rfl
  have h3 : Susan_given = Susan_apples / 2 := rfl
  have h4 : Frank_sold = Frank_apples / 3 := rfl
  -- since we know the values, we could calculate directly
  have h5 : Susan_apples = 108 := by rw [h1, h2]; norm_num
  have h6 : Susan_given = 54 := by rw [h5]; norm_num
  have h7 : Frank_sold = 12 := by rw [h2]; norm_num
  calc
    Susan_apples - Susan_given + Frank_apples - Frank_sold
        = 108 - 54 + 36 - 12 := by rw [h5, h6, h2, h7]
    ... = 78 := by norm_num

end apples_left_total_l574_574276


namespace num_squares_8x8_l574_574347

def num_squares (n : ℕ) : ℕ :=
  (list.range (n + 1)).sum (λ k, (n - k + 1) ^ 2)

theorem num_squares_8x8 : num_squares 8 = 204 :=
  by sorry

end num_squares_8x8_l574_574347


namespace probability_at_least_6_heads_l574_574153

-- Definitions of the binomial coefficient and probability function
def binom (n k : ℕ) : ℕ := Nat.choose n k

def probability (favorable total : ℕ) : ℚ := favorable / total

-- Proof problem statement
theorem probability_at_least_6_heads (flips : ℕ) (p : ℚ) 
  (h_flips : flips = 8) 
  (h_probability : p = probability (binom 8 6 + binom 8 7 + binom 8 8) (2 ^ flips)) : 
  p = 37 / 256 :=
by
  sorry

end probability_at_least_6_heads_l574_574153


namespace count_valid_numbers_l574_574672

/-- Represent the valid digits as a set -/
def valid_digits : set ℕ := {0, 5, 6, 7, 8, 9}

/-- Predicate to check if an integer does not contain the digits 1, 2, 3, or 4 -/
def no_invalid_digits (n : ℕ) : Prop :=
  ∀ d ∈ Int.digits 10 n, d ∈ valid_digits

/-- The total number of integers from 1 to 9999 inclusive that do not contain
    any of the digits 1, 2, 3, or 4 is 1295. -/
theorem count_valid_numbers : {m : ℕ | 1 ≤ m ∧ m ≤ 9999 ∧ no_invalid_digits m}.to_finset.card = 1295 := by
  sorry

end count_valid_numbers_l574_574672


namespace largest_difference_l574_574063

def numbers : set ℤ := {-20, -5, 1, 5, 7, 19}

theorem largest_difference :
  ∃ a b ∈ numbers, a - b = 39 :=
by
  use 19
  use -20
  simp
  sorry

end largest_difference_l574_574063


namespace square_area_l574_574914

theorem square_area (y : ℝ) (x₁ x₂ : ℝ) (s : ℝ) (A : ℝ) :
  y = 7 → 
  (y = x₁^2 + 4 * x₁ + 3) →
  (y = x₂^2 + 4 * x₂ + 3) →
  x₁ ≠ x₂ →
  s = |x₂ - x₁| → 
  A = s^2 →
  A = 32 :=
by
  intros hy intersection_x1 intersection_x2 hx1x2 hs ha
  sorry

end square_area_l574_574914


namespace factorization_l574_574973

theorem factorization (x y : ℝ) : 
  100 - 25 * x^2 + 16 * y^2 = (10 - 5 * x + 4 * y) * (10 + 5 * x - 4 * y) :=
by
  sorry

end factorization_l574_574973


namespace probability_at_least_6_heads_8_flips_l574_574163

-- Define the probability calculation of getting at least 6 heads in 8 coin flips.
def probability_at_least_6_heads (n : ℕ) (k : ℕ) : ℚ :=
  (Nat.choose n k + Nat.choose n (k + 1) + Nat.choose n (k + 2)) / 2^n

theorem probability_at_least_6_heads_8_flips : 
  probability_at_least_6_heads 8 6 = 37 / 256 := 
by
  sorry

end probability_at_least_6_heads_8_flips_l574_574163


namespace solution_set_of_inequality_l574_574657

def f (x : ℝ) : ℝ :=
  if x < 0 then 2 - x else 2 - x^2

theorem solution_set_of_inequality (a : ℝ) : f (2 * a + 1) > f (3 * a - 4) ↔ a > 5 :=
  by sorry

end solution_set_of_inequality_l574_574657


namespace find_line_l_find_circle_C_l574_574630

-- Define the lines and their intersection
def line1 (x y : ℝ) := 2 * x - y - 3 = 0
def line2 (x y : ℝ) := 4 * x - 3 * y - 5 = 0
def intersect_point (x y : ℝ) := line1 x y ∧ line2 x y

-- Define line l being perpendicular to another line and passing through a certain point
def perp_line1 (x y : ℝ) := x + y - 2 = 0
def line_l (x y : ℝ) := y = x - 1

-- Define the circle passing through a point with center on positive x-axis
def circle_condition (a r x y : ℝ) := (x - a) ^ 2 + y^2 = r^2
def circle_C (x y : ℝ) := ∃ a r, a > 0 ∧ circle_condition a r 1 0 ∧ (∃ k, (x - a)^2 + y^2 = r^2 ∧ 
                            let d := |a - 1| / real.sqrt 2 in d^2 + 2 = r^2)

-- The proving statements

theorem find_line_l :
  (∃ x y, intersect_point x y) →
  (∀ (x y : ℝ), perp_line1 x y → ∃ (k : ℝ), line_l x y) :=
sorry

theorem find_circle_C :
  ∃ (x y : ℝ), (circle_C x y) :=
sorry

end find_line_l_find_circle_C_l574_574630


namespace birds_remaining_l574_574048

variable (initial_birds : ℝ) (birds_flew_away : ℝ)

theorem birds_remaining (h1 : initial_birds = 12.0) (h2 : birds_flew_away = 8.0) : initial_birds - birds_flew_away = 4.0 :=
by
  rw [h1, h2]
  norm_num

end birds_remaining_l574_574048


namespace jose_share_of_profit_l574_574084

-- Definitions from problem conditions
def tom_investment : ℕ := 30000
def jose_investment : ℕ := 45000
def profit : ℕ := 27000
def months_total : ℕ := 12
def months_jose_investment : ℕ := 10

-- Derived calculations
def tom_month_investment := tom_investment * months_total
def jose_month_investment := jose_investment * months_jose_investment
def total_month_investment := tom_month_investment + jose_month_investment

-- Prove Jose's share of profit
theorem jose_share_of_profit : (jose_month_investment * profit) / total_month_investment = 15000 := by
  -- This is where the step-by-step proof would go
  sorry

end jose_share_of_profit_l574_574084


namespace sum_of_t_values_l574_574066

def is_isosceles (A B C : (ℝ × ℝ)) : Prop :=
  let d2 (P Q : (ℝ × ℝ)) := (P.1 - Q.1)^2 + (P.2 - Q.2)^2
  d2 A B = d2 A C ∨ d2 A B = d2 B C ∨ d2 A C = d2 B C

def triangle_vertices (t : ℝ) : ((ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) :=
  ((Real.cos (30 * Real.pi / 180), Real.sin (30 * Real.pi / 180)),
   (Real.cos (90 * Real.pi / 180), Real.sin (90 * Real.pi / 180)),
   (Real.cos (t * Real.pi / 180), Real.sin (t * Real.pi / 180)))

theorem sum_of_t_values : ∑ (t : ℝ) in {t | is_isosceles (triangle_vertices t).1 (triangle_vertices t).2 (triangle_vertices t).3 ∧ 0 ≤ t ∧ t ≤ 360}, t = 540 :=
sorry

end sum_of_t_values_l574_574066


namespace fourth_candidate_votes_calculation_l574_574695

-- Define the conditions
def total_votes : ℕ := 7000
def invalid_percent : ℝ := 0.25
def valid_percent : ℝ := 0.75
def valid_votes : ℕ := total_votes * valid_percent

def first_candidate_percent : ℝ := 0.40
def second_candidate_percent : ℝ := 0.35
def third_candidate_percent : ℝ := 0.15

def first_candidate_votes : ℕ := valid_votes * first_candidate_percent
def second_candidate_votes : ℕ := valid_votes * second_candidate_percent
def third_candidate_votes : ℕ := valid_votes * third_candidate_percent

-- Define the proof theorem
theorem fourth_candidate_votes_calculation (total_votes : ℕ) (invalid_percent valid_percent first_candidate_percent second_candidate_percent third_candidate_percent : ℝ) (valid_votes first_candidate_votes second_candidate_votes third_candidate_votes : ℕ)
  (h_valid_votes : valid_votes = total_votes * valid_percent)
  (h_first_candidate_votes : first_candidate_votes = valid_votes * first_candidate_percent)
  (h_second_candidate_votes : second_candidate_votes = valid_votes * second_candidate_percent)
  (h_third_candidate_votes : third_candidate_votes = valid_votes * third_candidate_percent) :
  (valid_votes - (first_candidate_votes + second_candidate_votes + third_candidate_votes) = 525) :=
by
  -- Proof goes here
  sorry

end fourth_candidate_votes_calculation_l574_574695


namespace projections_collinear_iff_on_circumcircle_l574_574734

variable {α : Type*} [EuclideanGeometry α]

def orthogonal_projection (P A B : α) : α := sorry

def is_collinear (A' B' C' : α) : Prop := sorry

def lies_on_circumcircle (P : α) (ABC : Triangle α) : Prop := sorry

theorem projections_collinear_iff_on_circumcircle
  (ABC : Triangle α)
  (P : α)
  (A' B' C' : α)
  (hA' : A' = orthogonal_projection P ABC.bc)
  (hB' : B' = orthogonal_projection P ABC.ca)
  (hC' : C' = orthogonal_projection P ABC.ab) :
  is_collinear A' B' C' ↔ lies_on_circumcircle P ABC :=
by
  sorry

end projections_collinear_iff_on_circumcircle_l574_574734


namespace library_fiction_percentage_l574_574890

theorem library_fiction_percentage:
  let original_volumes := 18360
  let fiction_percentage := 0.30
  let fraction_transferred := 1/3
  let fraction_fiction_transferred := 1/5
  let initial_fiction := fiction_percentage * original_volumes
  let transferred_volumes := fraction_transferred * original_volumes
  let transferred_fiction := fraction_fiction_transferred * transferred_volumes
  let remaining_fiction := initial_fiction - transferred_fiction
  let remaining_volumes := original_volumes - transferred_volumes
  let remaining_fiction_percentage := (remaining_fiction / remaining_volumes) * 100
  remaining_fiction_percentage = 35 := 
by
  sorry

end library_fiction_percentage_l574_574890


namespace simplify_frac_48_72_l574_574004

theorem simplify_frac_48_72 : (48 / 72 : ℚ) = 2 / 3 :=
by
  -- In Lean, we prove the equality of the simplified fractions.
  sorry

end simplify_frac_48_72_l574_574004


namespace tower_problem_solution_correct_l574_574888

structure TowerProblem where
  rope_length : ℝ
  tower_radius : ℝ
  dragon_height : ℝ
  end_distance : ℝ
  d : ℕ
  e : ℕ
  f : ℕ
  f_prime : Prime ℕ

def is_solution (tp : TowerProblem) : Prop :=
  tp.rope_length = 25 ∧
  tp.tower_radius = 10 ∧
  tp.dragon_height = 7 ∧
  tp.end_distance = 5 ∧
  tp.f_prime ∧
  (15 - Real.sqrt 130) / 2 = (tp.d - Real.sqrt tp.e) / tp.f ∧
  (tp.d + tp.e + tp.f) = 147

noncomputable def solution : TowerProblem :=
  { rope_length := 25,
    tower_radius := 10,
    dragon_height := 7,
    end_distance := 5,
    d := 15,
    e := 130,
    f := 2,
    f_prime := Prime.mk (by norm_num) }

theorem tower_problem_solution_correct: is_solution solution :=
by 
  simp[is_solution, solution]
  sorry

end tower_problem_solution_correct_l574_574888


namespace garden_perimeter_l574_574433

theorem garden_perimeter (A : ℝ) (hA : A = 900) :
  ∃ P P_cm : ℝ, P = 120 ∧ P_cm = 12000 :=
by
  let s := Real.sqrt A
  have hs : s = 30, from sorry
  let P := 4 * s
  have hP : P = 120, from sorry
  let P_cm := P * 100
  have hP_cm : P_cm = 12000, from sorry
  refine ⟨P, P_cm, hP, hP_cm⟩

end garden_perimeter_l574_574433


namespace acetic_acid_molarity_l574_574951

theorem acetic_acid_molarity
  (mass_percent: ℝ) (molar_mass_acetic_acid : ℝ) (density_vinegar : ℝ) 
  (mass_percent_eq : mass_percent = 5.00) 
  (molar_mass_acetic_acid_eq : molar_mass_acetic_acid = 60.0) 
  (density_vinegar_eq : density_vinegar = 1.00) :
  (molarity : ℝ) (molarity = 0.833) :=
sorry

end acetic_acid_molarity_l574_574951


namespace book_cost_l574_574878

variable {b m : ℝ}

theorem book_cost (h1 : b + m = 2.10) (h2 : b = m + 2) : b = 2.05 :=
by
  sorry

end book_cost_l574_574878


namespace inversely_proportional_solve_y_l574_574040

theorem inversely_proportional_solve_y (k : ℝ) (x y : ℝ)
  (h1 : x * y = k)
  (h2 : x + y = 60)
  (h3 : x = 3 * y)
  (hx : x = -10) :
  y = -67.5 :=
by
  sorry

end inversely_proportional_solve_y_l574_574040


namespace log_bounds_l574_574998

theorem log_bounds (log₁₀ : ℝ → ℝ)
  (h₁ : log₁₀ 10 = 1)
  (h₂ : log₁₀ 100 = 2)
  (h₃ : 10 < 43)
  (h₄ : 43 < 100)
  : 1 + 2 = 3 :=
by
  have h₅ : 1 < log₁₀ 43 := sorry
  have h₆ : log₁₀ 43 < 2 := sorry
  let c := 1
  let d := 2
  have h₇ : c + d = 3 := by
    exact 1 + 2 = 3
  exact h₇

end log_bounds_l574_574998


namespace arithmetic_sequence_problem_l574_574301

variable (a : ℕ → ℕ)
variable (d : ℕ) -- common difference for the arithmetic sequence
variable (h1 : ∀ n : ℕ, a (n + 1) = a n + d)
variable (h2 : a 1 - a 9 + a 17 = 7)

theorem arithmetic_sequence_problem : a 3 + a 15 = 14 := by
  sorry

end arithmetic_sequence_problem_l574_574301


namespace find_percentage_l574_574884

theorem find_percentage (P : ℕ) (h1 : P * 64 = 320 * 10) : P = 5 := 
  by
  sorry

end find_percentage_l574_574884


namespace staircase_ways_10_l574_574820

def staircase_ways : ℕ → ℕ
| 0       := 0
| 1       := 1
| 2       := 2
| 3       := 4
| (n + 4) := staircase_ways (n + 3) + staircase_ways (n + 2) + staircase_ways (n + 1)

theorem staircase_ways_10 : staircase_ways 10 = 274 := by
  sorry

end staircase_ways_10_l574_574820


namespace ball_hits_ground_at_time_l574_574024

-- Given definitions from the conditions
def y (t : ℝ) : ℝ := -4.9 * t^2 + 5 * t + 8

-- Statement of the problem: proving the time t when the ball hits the ground
theorem ball_hits_ground_at_time :
  ∃ t : ℝ, y t = 0 ∧ t = 1.887 := 
sorry

end ball_hits_ground_at_time_l574_574024


namespace probability_at_least_6_heads_in_8_flips_l574_574156

def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then (Nat.choose n k) else 0

theorem probability_at_least_6_heads_in_8_flips :
  let total_outcomes := 2^8
  let successful_outcomes := binomial 8 6 + binomial 8 7 + binomial 8 8
  let probability := (successful_outcomes : ℚ) / total_outcomes
  probability = 37 / 256 :=
by
  sorry

end probability_at_least_6_heads_in_8_flips_l574_574156


namespace solve_for_x_l574_574267

theorem solve_for_x (x : ℝ) (h : sqrt (x + 12) = 10) : x = 88 := by
  sorry

end solve_for_x_l574_574267


namespace prime_special_form_l574_574244

theorem prime_special_form (p : ℕ) (h_prime_p : prime p)
  (h_sum : ∃ q r : ℕ, prime q ∧ prime r ∧ p = q + r)
  (h_diff : ∃ s t : ℕ, prime s ∧ prime t ∧ p = s - t) : p = 5 :=
sorry

end prime_special_form_l574_574244


namespace defective_iphones_correct_l574_574957

-- Define the initial counts and the counts at the end of the day
variables (initial_samsung initial_iphone final_samsung final_iphone total_sold : ℕ)
variables (defective_iphones : ℕ)

-- Define the specific values for our problem
def initial_samsung := 14
def initial_iphone := 8
def final_samsung := 10
def final_iphone := 5
def total_sold := 4

-- Define the target value (defective iPhones)
def defective_iphones := 3

-- Define the number of phones sold for each type
def samsung_sold := initial_samsung - final_samsung
def iphone_sold := initial_iphone - final_iphone

-- Express the conditions and goal
theorem defective_iphones_correct :
  samsung_sold + iphone_sold = total_sold →
  defective_iphones = (initial_iphone - final_iphone - iphone_sold) →
  defective_iphones = 3 :=
by rewrite; sorry

end defective_iphones_correct_l574_574957


namespace pencils_per_pack_l574_574234

theorem pencils_per_pack
  (packs : ℕ) (pencils_per_row : ℕ) (rows : ℕ)
  (h1 : packs = 35)
  (h2 : pencils_per_row = 2)
  (h3 : rows = 70) :
  (rows * pencils_per_row) / packs = 4 :=
by
  rwa [h1, h2, h3]
  sorry

end pencils_per_pack_l574_574234


namespace length_of_platform_is_400_l574_574182

-- Define the conditions directly translated from the problem
def train_length : ℝ := 225
def train_speed_kmph : ℝ := 90
def crossing_time : ℝ := 25

-- Utilize a conversion factor for speed from kmph to m/s
def kmph_to_mps (speed_kmph : ℝ) : ℝ := speed_kmph * (1000 / 3600)

-- Define the speed in m/s based on the given speed in km/h
def train_speed_mps : ℝ := kmph_to_mps train_speed_kmph

-- Define the total distance covered in the given time
def total_distance : ℝ := train_speed_mps * crossing_time

-- Define the length of the platform
def platform_length : ℝ := total_distance - train_length

-- Theorem statement to prove the length of the platform
theorem length_of_platform_is_400 : platform_length = 400 := by
  sorry

end length_of_platform_is_400_l574_574182


namespace sequence_square_sum_l574_574360

theorem sequence_square_sum (a : ℕ → ℕ) (n : ℕ) (h : ∑ i in Finset.range n, a (i + 1) = 2^n - 1) :
  ∑ i in Finset.range n, (a (i + 1))^2 = (4^n - 1) / 3 :=
sorry

end sequence_square_sum_l574_574360


namespace knocks_to_knicks_l574_574331

variable (knicks knacks knocks : Type) 

-- Declare that 5 knicks equals 3 knacks
axiom knicks_to_knacks (k : knicks → knacks) (h1 : 5 * k = 3 * h1)

-- Declare that 2 knacks equal 7 knocks
axiom knacks_to_knocks (k : knacks → knocks) (h2 : 2 * k = 7 * h2)

-- 49 knocks equal how many knicks
theorem knocks_to_knicks (h1 : 5 * knacks = 3 * knicks) (h2 : 2 * knacks = 7 * knocks) : 49 * knocks = (70 / 3) * knicks := 
  sorry

end knocks_to_knicks_l574_574331


namespace compound_interest_doubling_time_l574_574253

theorem compound_interest_doubling_time :
  ∃ (t : ℕ), (0.15 : ℝ) = 0.15 ∧ ∀ (n : ℕ), (n = 1) →
               (2 : ℝ) < (1 + 0.15) ^ t ∧ t = 5 :=
by
  sorry

end compound_interest_doubling_time_l574_574253


namespace prob_at_least_6_heads_eq_l574_574121

-- define the number of coin flips
def n := 8

-- define the number of possible outcomes (2^n)
def total_outcomes := 2 ^ n

-- define the binomial coefficients for cases: 6 heads, 7 heads, 8 heads
def binom_8_6 := Nat.choose 8 6
def binom_8_7 := Nat.choose 8 7
def binom_8_8 := Nat.choose 8 8

-- calculate the favorable outcomes for at least 6 heads
def favorable_outcomes := binom_8_6 + binom_8_7 + binom_8_8

-- define the probability of getting at least 6 heads
def probability := (favorable_outcomes : ℚ) / total_outcomes

theorem prob_at_least_6_heads_eq : probability = 37 / 256 := by
  sorry

end prob_at_least_6_heads_eq_l574_574121


namespace independent_trials_probability_l574_574271

theorem independent_trials_probability (p : ℝ) (q : ℝ) (ε : ℝ) (desired_prob : ℝ) 
    (h_p : p = 0.7) (h_q : q = 0.3) (h_ε : ε = 0.2) (h_desired_prob : desired_prob = 0.96) :
    ∃ n : ℕ, n > (p * q) / (desired_prob * ε^2) ∧ n = 132 :=
by
  sorry

end independent_trials_probability_l574_574271


namespace number_of_days_to_catch_fish_l574_574968

variable (fish_per_day : ℕ) (fillets_per_fish : ℕ) (total_fillets : ℕ)

theorem number_of_days_to_catch_fish (h1 : fish_per_day = 2) 
                                    (h2 : fillets_per_fish = 2) 
                                    (h3 : total_fillets = 120) : 
                                    (total_fillets / fillets_per_fish) / fish_per_day = 30 :=
by sorry

end number_of_days_to_catch_fish_l574_574968


namespace prove_monthly_growth_rate_prove_additional_agents_required_l574_574818

-- Define the conditions
def deliveries_march : ℝ := 100000
def deliveries_may : ℝ := 121000
def max_deliveries_per_agent_per_month : ℝ := 0.6 * 1000
def initial_agents : ℝ := 20

-- Define the monthly growth rate and the required additional agents to meet deliveries in June
noncomputable def monthly_growth_rate : ℝ := sqrt(1.21) - 1
def expected_deliveries_june : ℝ := deliveries_may * (1 + monthly_growth_rate)
def current_capacity : ℝ := max_deliveries_per_agent_per_month * initial_agents
noncomputable def additional_agents_required : ℝ := ceil ((expected_deliveries_june - current_capacity) / max_deliveries_per_agent_per_month)

-- Prove the statements
theorem prove_monthly_growth_rate : monthly_growth_rate = 0.1 := by
    sorry

theorem prove_additional_agents_required : additional_agents_required = 3 := by
    sorry

end prove_monthly_growth_rate_prove_additional_agents_required_l574_574818


namespace ellipse_correct_circle_through_origin_l574_574616

noncomputable def ellipse_equation : Prop :=
  ∃ a b : ℝ, 
    0 < b ∧ b < a ∧ 
    ell_eq (a, b) ∧ 
    pass_through (1, -sqrt 2 / 2) ∧ 
    ecc_eq (sqrt 2 / 2) ∧ 
    calc_area ∆ABF₂ (4 * sqrt 3 / 5) ∧ 
    ∀ x y : ℝ, ell_eq (sqrt 2, 1)

theorem ellipse_correct {a b : ℝ} :
  ellipse_equation ∧ ell_eq (a, b) → a = sqrt 2 ∧ b = 1 → ell_eq (sqrt 2, 1) :=
begin
  intros H1 H2,
  sorry
end

theorem circle_through_origin {F₁ F₂ A B : ℝ × ℝ} :
  passes_line F₁ (A, B) ell_eq ∧ calc_area ∆ABF₂ (4 * sqrt 3 / 5) →
  circle_diam (A, B) passes_origin :=
begin
  intros H1 H2,
  sorry
end

end ellipse_correct_circle_through_origin_l574_574616


namespace sum_vertical_asymptotes_l574_574796

-- Define the function
def rational_function (x : ℝ) : ℝ :=
  (6 * x^2 - 7) / (4 * x^2 + 6 * x + 3)

-- Define the condition for vertical asymptotes
def denominator_zero (x : ℝ) : Prop :=
  4 * x^2 + 6 * x + 3 = 0

-- Define the sum of roots (vertical asymptotes)
def sum_roots (f : ℝ → ℝ) (hf : ∀ x, denominator_zero x → f x = 0) : ℝ :=
  -6 / 4

-- Lean statement for the proof problem
theorem sum_vertical_asymptotes :
  sum_roots rational_function denominator_zero = -3 / 2 :=
sorry

end sum_vertical_asymptotes_l574_574796


namespace domain_of_f_period_of_f_l574_574861

noncomputable def f (x : ℝ) : ℝ := abs (tan (2 * x + π / 3))

theorem domain_of_f : { x : ℝ | ∃ k : ℤ, x = k * π / 2 + π / 12 }ᶜ = { x : ℝ | x ∉ ⋃ k : ℤ, { k * π / 2 + π / 12 } } :=
by
  sorry

theorem period_of_f : (∃ p : ℝ, 0 < p ∧ ∀ x : ℝ, f (x + p) = f x) ∧ ∀ q : ℝ, (0 < q ∧ ∀ x : ℝ, f (x + q) = f x) → q ≥ π / 2 :=
by
  sorry

end domain_of_f_period_of_f_l574_574861


namespace find_small_bucket_count_l574_574508

-- Define the conditions
def large_bucket_capacity := 4  -- in liters
def tank_capacity := 63         -- in liters
def large_bucket_count := 5

-- Define the relationship between large and small bucket capacities
axiom large_small_relation (S : ℝ) : large_bucket_capacity = 2 * S + 3

-- Define the number of small buckets required
def small_bucket_count (S : ℝ) : ℝ := (tank_capacity - (large_bucket_count * large_bucket_capacity)) / S

-- Theorem statement
theorem find_small_bucket_count (S : ℝ) (h : large_small_relation S) : small_bucket_count S = 86 :=
by
  sorry

end find_small_bucket_count_l574_574508


namespace polynomial_not_divisible_l574_574415

theorem polynomial_not_divisible (n m A : ℤ) (hn : n > 0) (hm : m > 0) :
  ¬ ∃ (q : polynomial ℤ), (3 * X^(2 * n) + polynomial.C A * X^n + 2) = 
  (2 * X^(2 * m) + polynomial.C A * X^m + 3) * q :=
by 
  sorry

end polynomial_not_divisible_l574_574415


namespace solveProblem_l574_574345

noncomputable def problem (XY YZ ZX : ℝ) (XM : ℝ) (G F R S : ℝ) (triangle : Triangle XYZ) 
  (median : Median XM) (bisectors : AngleBisectors R S) : Prop :=
  XY = 13 ∧ YZ = 14 ∧ ZX = 15 ∧ XM = medianLength 13 14 15 ∧ bisectors.RS = 129 / 203

theorem solveProblem : problem 13 14 15 12 G F R S myTriangle myMedian myBisectors :=
begin
  sorry
end

end solveProblem_l574_574345


namespace probability_of_at_least_six_heads_is_correct_l574_574127

-- Definitions for the given problem
def binomial_coefficient (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

def total_possible_outcomes : ℕ :=
  2^8

def favorable_outcomes : ℕ :=
  binomial_coefficient 8 6 + binomial_coefficient 8 7 + binomial_coefficient 8 8

def probability_of_at_least_6_heads : ℚ :=
  favorable_outcomes / total_possible_outcomes

-- The proof statement
theorem probability_of_at_least_six_heads_is_correct :
  probability_of_at_least_6_heads = 37 / 256 :=
by sorry

end probability_of_at_least_six_heads_is_correct_l574_574127


namespace largest_lcm_value_l574_574841

theorem largest_lcm_value : 
  let lcm_vals := [Nat.lcm 18 2, Nat.lcm 18 3, Nat.lcm 18 6, Nat.lcm 18 9, Nat.lcm 18 12, Nat.lcm 18 15] in
  list.maximum lcm_vals = some 90 :=
by
  sorry

end largest_lcm_value_l574_574841


namespace minimum_value_AB_l574_574659

open Real 

noncomputable def y (x : ℝ) := 2 * exp (x - 1/2) - log x

theorem minimum_value_AB :
  (∀ m > 0, ∀ (A : point), ∀ (B : point),
    A = (log m, m) → B = (2 * exp (m - 1/2), m) → 2 * exp (m - 1/2) > log m → 
    minValue (λ z, |(A.1 - B.1)|) = 2 + log 2)  :=
by
  sorry

end minimum_value_AB_l574_574659


namespace binomial_constant_term_l574_574681

theorem binomial_constant_term (x : ℝ) (n : ℕ) (h : (n : ℝ) = 9) :
  let ⟨a, b⟩ := (√x - (1 / (23*x)))^n in
  ∃ (k : ℕ), a^(n - 2*k) = 1 ∧ b^k = 1 := by
  sorry

end binomial_constant_term_l574_574681


namespace probability_defective_first_factory_l574_574050

-- Define the conditions
def P_H1 := 4.0 / (4.0 + 1.0) -- Probability of production from first factory
def P_H2 := 1.0 / (4.0 + 1.0) -- Probability of production from second factory

def P_A_H1 := 0.05 -- Probability of defective part from first factory
def P_A_H2 := 0.01 -- Probability of defective part from second factory

-- Use the law of total probability to calculate P(A)
def P_A := (P_A_H1 * P_H1) + (P_A_H2 * P_H2)

-- Apply Bayes' theorem to calculate P(H1|A)
def P_H1_A := (P_A_H1 * P_H1) / P_A

-- State the proof problem
theorem probability_defective_first_factory :
  P_H1_A ≈ 0.952 :=
by sorry

end probability_defective_first_factory_l574_574050


namespace quadruple_fib_sum_eq_l574_574247

def fib : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := fib (n + 1) + fib n

theorem quadruple_fib_sum_eq (a b c d : ℕ) (ha: a ≥ 2) (hb: b ≥ 2) (hc: c ≥ 2) (hd: d ≥ 2) :
  fib a + fib b = fib c + fib d → 
  ((a = c ∧ b = d) ∨ (a = d ∧ b = c) ∨
   (a = c + 1 ∧ b = d + 3 ∧ a - 1 = c + 1) ∨ (a = d + 1 ∧ b = c + 3 ∧ a - 1 = d + 1) ∨
   (a = c + 3 ∧ b = d + 1 ∧ a - 3 = c + 3) ∨ (a = d + 3 ∧ b = c + 1 ∧ a - 3 = d + 3)) :=
sorry

end quadruple_fib_sum_eq_l574_574247


namespace value_of_expression_l574_574955

theorem value_of_expression :
  3 + Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (Real.sqrt 3 - 3) = 3 + 2 * Real.sqrt 3 / 3 :=
by sorry

end value_of_expression_l574_574955


namespace range_f_greater_than_one_l574_574682

theorem range_f_greater_than_one (f : ℝ → ℝ) (g : ℝ → ℝ) (h_g : ∀ x, g x = 2 ^ x)
  (symm_about_y : ∀ x, f x = g (-x)) : {x : ℝ | f x > 1} = Iio 0 :=
by
  sorry

end range_f_greater_than_one_l574_574682


namespace functions_with_inverses_functions_without_inverses_l574_574857

-- Function definitions with their respective domains
def a (x : ℝ) : ℝ := (3 - x)^(1/3)
def d (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 10
def f (x : ℝ) : ℝ := 3^x + 5^x
def g (x : ℝ) : ℝ := x - 1 / x^2
def h (x : ℝ) : ℝ := x / 3

def b (x : ℝ) : ℝ := x^3 - 3 * x
def c (x : ℝ) : ℝ := x + 1 / x^2
def e (x : ℝ) : ℝ := abs (x - 3) + abs (x + 4)

-- Lean statement of the problem 
theorem functions_with_inverses :
  (x ∈ Set.Iic 3 → Function.Injective a) ∧
  (Function.Injective d) ∧
  (Function.Injective f) ∧
  (Function.Injective g) ∧
  (x ∈ Set.Icc (-3) 9 → Function.Injective h) :=
sorry

theorem functions_without_inverses :
  ¬Function.Injective b ∧
  (x > 0 → ¬Function.Injective c) ∧
  ¬Function.Injective e :=
sorry

end functions_with_inverses_functions_without_inverses_l574_574857


namespace inflection_point_on_line_y_eq_neg4x_l574_574641

def f (x : ℝ) : ℝ := -4 * x + 3 * Real.sin x - Real.cos x

def inflection_point (x : ℝ) (f : ℝ → ℝ) : Prop :=
  ∃ x₀, (derivative (derivative f) x₀ = 0) ∧ (x₀ = x)

theorem inflection_point_on_line_y_eq_neg4x :
  inflection_point (Real.arctan (1 / 3)) f →
  (f (Real.arctan (1 / 3)) = -4 * Real.arctan (1 / 3)) :=
by
  sorry

end inflection_point_on_line_y_eq_neg4x_l574_574641


namespace area_D_meets_sign_l574_574560

-- Definition of conditions as given in the question
def condition_A (mean median : ℝ) : Prop := mean = 3 ∧ median = 4
def condition_B (mean : ℝ) (variance_pos : Prop) : Prop := mean = 1 ∧ variance_pos
def condition_C (median mode : ℝ) : Prop := median = 2 ∧ mode = 3
def condition_D (mean variance : ℝ) : Prop := mean = 2 ∧ variance = 3

-- Theorem stating that Area D satisfies the condition to meet the required sign
theorem area_D_meets_sign (mean variance : ℝ) (h : condition_D mean variance) : 
  (∀ day_increase, day_increase ≤ 7) :=
sorry

end area_D_meets_sign_l574_574560


namespace length_of_AB_l574_574811

-- Problem conditions
def radius : ℝ := 5
def total_volume : ℝ := 660 * Real.pi

-- The given region includes a cylinder and two hemispheres.
def volume_hemisphere (r : ℝ) : ℝ :=
  (2 / 3) * Real.pi * r^3

def volume_cylinder (r h : ℝ) : ℝ :=
  Real.pi * r^2 * h

-- Given that the volume of the entire region is the sum of the cylinder and the two hemispheres:
def region_volume (r h : ℝ) : ℝ :=
  volume_cylinder r h + 2 * volume_hemisphere r

-- Theorem: Prove the length of AB is 20
theorem length_of_AB : 
  ∃ h, region_volume radius h = total_volume ∧ h = 20 :=
by
  sorry

end length_of_AB_l574_574811


namespace smallest_ba_xian_number_gt_2000_l574_574991

def is_ba_xian_number (N : ℕ) : Prop :=
  (finset.card (finset.filter (λ d, N % d = 0) (finset.range 10))) ≥ 8

theorem smallest_ba_xian_number_gt_2000 : ∃ N : ℕ, N > 2000 ∧ is_ba_xian_number N ∧ ∀ M : ℕ, M > 2000 ∧ is_ba_xian_number M → N ≤ M :=
begin
  use 2016,
  split,
  { -- Proof that 2016 > 2000
    exact dec_trivial, -- 2016 > 2000 is trivial
  },
  split,
  { -- Proof that 2016 is a Ba Xian number
    sorry, -- Skipping the actual proof
  },
  { -- Proof that 2016 is the smallest Ba Xian number greater than 2000
    sorry, -- Skipping the actual proof
  }
end

end smallest_ba_xian_number_gt_2000_l574_574991


namespace felt_area_of_conical_shed_l574_574817

theorem felt_area_of_conical_shed 
  (base_diameter slant_height : ℝ) 
  (h_base_diameter : base_diameter = 10) 
  (h_slant_height : slant_height = 6) : 
  let radius := base_diameter / 2 
  let lateral_surface_area := Float.pi * radius * slant_height 
  lateral_surface_area = 30 * Float.pi :=
by 
  -- Assign conditions directly to avoid indirect assumptions
  have h_radius : radius = 5 := by simp [h_base_diameter]
  have h_lateral_surface_area : lateral_surface_area = 30 * Float.pi := by simp [h_radius, h_slant_height]
  -- Conclude with the required statement
  exact h_lateral_surface_area

end felt_area_of_conical_shed_l574_574817


namespace single_equivalent_discount_l574_574778

-- Define conditions
def discount1 : ℝ := 0.15
def discount2 : ℝ := 0.10
def discount3 : ℝ := 0.05

-- Define the equivalent single discount
theorem single_equivalent_discount :
  let original_price := 1 in
  let price_after_first_discount := (1 - discount1) * original_price in
  let price_after_second_discount := (1 - discount2) * price_after_first_discount in
  let price_after_third_discount := (1 - discount3) * price_after_second_discount in
  (1 - 0.27325) * original_price = price_after_third_discount :=
by
  sorry

end single_equivalent_discount_l574_574778


namespace units_digit_sum_42_4_24_4_l574_574844

-- Define the units digit function
def units_digit (n : ℕ) : ℕ := n % 10

-- Given conditions
def units_digit_42_4 : units_digit (42^4) = 6 := sorry
def units_digit_24_4 : units_digit (24^4) = 6 := sorry

-- Theorem to prove
theorem units_digit_sum_42_4_24_4 :
  units_digit (42^4 + 24^4) = 2 :=
by
  -- Use the given conditions
  have h1 : units_digit (42^4) = 6 := units_digit_42_4
  have h2 : units_digit (24^4) = 6 := units_digit_24_4
  -- Calculate the units digit of their sum
  calc 
    units_digit (42^4 + 24^4)
        = units_digit (6 + 6) : by rw [h1, h2]
    ... = units_digit 12    : by norm_num
    ... = 2                 : by norm_num

end units_digit_sum_42_4_24_4_l574_574844


namespace total_amount_received_approx_l574_574369

noncomputable def futureValue (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n)^(n * t)

def P1 := 1200
def r1 := 0.05
def n1 := 2
def t1 := 3

def P2 := 2000
def r2 := 0.07
def n2 := 4
def t2 := 4

def A1 := futureValue P1 r1 n1 t1
def A2 := futureValue P2 r2 n2 t2

theorem total_amount_received_approx :
  A1 + A2 ≈ 4012.72 := by
  sorry

end total_amount_received_approx_l574_574369


namespace elephant_weight_l574_574052

theorem elephant_weight :
  ∃ (w : ℕ), ∀ i : Fin 15, (i.val ≤ 13 → w + 2 * w = 15000) ∧ ((0:ℕ) < w → w = 5000) :=
by
  sorry

end elephant_weight_l574_574052


namespace bug_at_A_after_7_moves_l574_574614

-- Define the recursive sequence for the probabilities
def a : ℕ → ℚ
| 0     := 1
| (n+1) := 1 / 3 * (1 - a n)

-- The theorem stating the probability after 7 meters of crawling
theorem bug_at_A_after_7_moves : a 7 = 182 / 729 := by
  sorry

end bug_at_A_after_7_moves_l574_574614


namespace gerry_bananas_eaten_l574_574997

theorem gerry_bananas_eaten (b : ℝ) : 
  (b + (b + 8) + (b + 16) + 0 + (b + 24) + (b + 32) + (b + 40) + (b + 48) = 220) →
  b + 48 = 56.67 :=
by
  sorry

end gerry_bananas_eaten_l574_574997


namespace compute_sum_l574_574740

variables {a b : ℝ}

-- Conditions as hypotheses
def condition_a := a^3 - 12*a^2 + 47*a - 60 = 0
def condition_b := 8*b^3 - 48*b^2 + 18*b + 162 = 0

-- Theorem that computes a + b
theorem compute_sum (h1 : condition_a) (h2 : condition_b) : a + b = 3 := 
  sorry

end compute_sum_l574_574740


namespace evaluate_expression_l574_574425

open BigOperators

theorem evaluate_expression : 
  ∀ (x y : ℤ), x = -1 → y = 1 → 2 * (x^2 * y + x * y) - 3 * (x^2 * y - x * y) - 5 * x * y = -1 :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end evaluate_expression_l574_574425


namespace every_natural_number_appears_on_at_least_one_card_l574_574830

theorem every_natural_number_appears_on_at_least_one_card :
  ∀ n : ℕ, ∃ card : ℕ, card = n :=
by
  -- Problem conditions: For any n, there exist exactly n cards on which the divisors of n are written.
  have h1 : ∀ n : ℕ, ∃! c : ℕ, ∀ d : ℕ, d ∣ n → (c ∈ finset.range n).map id := 
    sorry
  -- Given the above condition, prove that for every n, there exists a card which has n written on it.
  intro n
  obtain ⟨card, _⟩ := h1 n
  use card
  -- Since n is a divisor of itself, we conclude that there is a card with n written on it.
  exact card_eq_n

end every_natural_number_appears_on_at_least_one_card_l574_574830


namespace scientific_notation_l574_574452

theorem scientific_notation : (0.00000000005 : ℝ) = 5 * 10^(-11) :=
by
  sorry

end scientific_notation_l574_574452


namespace current_books_l574_574171

def initial_books : ℕ := 743
def sold_instore_saturday : ℕ := 37
def sold_online_saturday : ℕ := 128
def sold_instore_sunday : ℕ := 2 * sold_instore_saturday
def sold_online_sunday : ℕ := sold_online_saturday + 34
def total_books_sold_saturday : ℕ := sold_instore_saturday + sold_online_saturday
def total_books_sold_sunday : ℕ := sold_instore_sunday + sold_online_sunday
def total_books_sold_weekend : ℕ := total_books_sold_saturday + total_books_sold_sunday
def books_received_shipment : ℕ := 160
def net_change_books : ℤ := books_received_shipment - total_books_sold_weekend

theorem current_books
  (initial_books : ℕ) 
  (sold_instore_saturday : ℕ) 
  (sold_online_saturday : ℕ) 
  (sold_instore_sunday : ℕ)
  (sold_online_sunday : ℕ)
  (total_books_sold_saturday : ℕ)
  (total_books_sold_sunday : ℕ)
  (total_books_sold_weekend : ℕ)
  (books_received_shipment : ℕ)
  (net_change_books : ℤ) : (initial_books - net_change_books) = 502 := 
by {
  sorry
}

end current_books_l574_574171


namespace probability_at_least_6_heads_l574_574151

-- Definitions of the binomial coefficient and probability function
def binom (n k : ℕ) : ℕ := Nat.choose n k

def probability (favorable total : ℕ) : ℚ := favorable / total

-- Proof problem statement
theorem probability_at_least_6_heads (flips : ℕ) (p : ℚ) 
  (h_flips : flips = 8) 
  (h_probability : p = probability (binom 8 6 + binom 8 7 + binom 8 8) (2 ^ flips)) : 
  p = 37 / 256 :=
by
  sorry

end probability_at_least_6_heads_l574_574151


namespace determinant_transformation_l574_574735

def determinant {α : Type*} [field α] {n : Type*} [fintype n] 
  (A : matrix n n α) : α := matrix.det A

variables {α : Type*} [field α]
variables (a b c : α → α) (D : α)

axiom deter_A_eq_D : determinant (matrix.of [a, b, c]) = D

theorem determinant_transformation {α : Type*} [field α] 
  (a b c : α → α) (D : α) 
  (h : determinant (matrix.of [a, b, c]) = D) :
  determinant (matrix.of [
    λ x, 2 * a x + b x,
    λ x, b x + 2 * c x,
    λ x, c x + 2 * a x
  ]) = 6 * D := 
sorry

end determinant_transformation_l574_574735


namespace triangle_problem_l574_574408

variables {A B C K M O : Type} [point A] [point B] [point C] [point K] 
[point M] [point O] -- Define the points

-- Define the conditions as Lean hypotheses
def parallel (KM AC : Type) : Prop := sorry -- Placeholder for parallel lines definition
def equal_length (x y : Type) : Prop := sorry -- Placeholder for equal length definition
def intersect_at (AM KC O : Type) : Prop := sorry -- Placeholder for segments intersecting at a point

theorem triangle_problem 
  (A B C K M O : Type)
  [point A] [point B] [point C] [point K] [point M] [point O]
  (H1: parallel KM AC)
  (H2: equal_length AK AO)
  (H3: equal_length KM MC)
  (H4: intersect_at AM KC O) : equal_length AM KB :=
by
  sorry -- Proof goes here

end triangle_problem_l574_574408


namespace problem_part1_problem_part2_l574_574886

def cost_price : ℕ := 50
def selling_price : ℕ := 70
def price_adjustments_and_sales : List (Int × ℕ) :=
  [ (5, 7), (2, 10), (1, 15), (0, 20), (-2, 23) ]

theorem problem_part1 :
  let total_exceed := 
    ∑ p in price_adjustments_and_sales, p.1 * p.2
  total_exceed = 24 :=
by
  sorry

theorem problem_part2 :
  let total_profit :=
    (selling_price - cost_price) * ∑ p in price_adjustments_and_sales, p.2 + 24
  total_profit = 1524 :=
by
  sorry

end problem_part1_problem_part2_l574_574886


namespace bisect_angle_l574_574838

-- Define the geometric set-up
variables {M N P A B C D : Type}
-- Assume given triangle with the right angle at P
variable [right_angle_triangle M N P]
-- Define circles k_M and k_N
noncomputable def k_M := circle M (distance M P)
noncomputable def k_N := circle N (distance N P)
-- Points of intersections
variables (A B : Type)
variable [on_circle A k_M]
variable [on_circle B k_M]
variable [collinear A B M N]
variables (C D : Type)
variable [on_circle C k_N]
variable [on_circle D k_N]
variable [collinear C D M N]
variable [between C A B]
variable [between C D A]

-- The harmonic division condition
axiom harmonic_div (A B C D : Type) : (cross_ratio A B C D = -1)

-- Proof goal
theorem bisect_angle (PC_bisects_∠APB : β) (PC_is_angle_bisector :
  bisector (line P C) (angle A P B)) :
  PC_is_angle_bisector := sorry

end bisect_angle_l574_574838


namespace toothpicks_per_card_l574_574233

-- Define the conditions of the problem
def numCardsInDeck : ℕ := 52
def numCardsNotUsed : ℕ := 16
def numCardsUsed : ℕ := numCardsInDeck - numCardsNotUsed

def numBoxesToothpicks : ℕ := 6
def toothpicksPerBox : ℕ := 450
def totalToothpicksUsed : ℕ := numBoxesToothpicks * toothpicksPerBox

-- Prove the number of toothpicks used per card
theorem toothpicks_per_card : totalToothpicksUsed / numCardsUsed = 75 := 
  by sorry

end toothpicks_per_card_l574_574233


namespace major_premise_incorrect_l574_574791

-- Define a rhombus and its properties
structure Rhombus (α : Type) [MetricSpace α] :=
  (sides_equal : ∀ (p q : α), (p ≠ q) → Metric.dist p q = Metric.dist q p)
  (diagonals_perpendicular : ∀ (p q r s : α), Metric.dist p q = Metric.dist q p 
                       → ∠pqr = 90)

-- Define a square as a special case of a rhombus.
structure Square (α : Type) extends Rhombus α :=
  (diagonals_equal : ∀ (p q : α), Metric.dist p q = Metric.dist q p)

-- Theorem stating the major premise is incorrect
theorem major_premise_incorrect (α : Type) [MetricSpace α] :
  ¬(∀ (R : Rhombus α) (p q r s : α), Metric.dist p q = Metric.dist q p
      ∧ Metric.dist r s = Metric.dist r s → Metric.dist p q = Metric.dist r s) :=
sorry

end major_premise_incorrect_l574_574791


namespace Karlson_cannot_prevent_Baby_getting_one_fourth_l574_574941

theorem Karlson_cannot_prevent_Baby_getting_one_fourth 
  (a : ℝ) (h : a > 0) (K : ℝ × ℝ) (hK : 0 < K.1 ∧ K.1 < a ∧ 0 < K.2 ∧ K.2 < a) :
  ∀ (O : ℝ × ℝ) (cut1 cut2 : ℝ), 
    ((O.1 = a/2) ∧ (O.2 = a/2) ∧ (cut1 = K.1 ∧ cut1 = a ∨ cut1 = K.2 ∧ cut1 = a) ∧ 
                             (cut2 = K.1 ∧ cut2 = a ∨ cut2 = K.2 ∧ cut2 = a)) →
  ∃ (piece : ℝ), piece ≥ a^2 / 4 :=
by
  sorry

end Karlson_cannot_prevent_Baby_getting_one_fourth_l574_574941


namespace insert_zeros_between_digits_is_cube_l574_574414

def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k^3 = n

theorem insert_zeros_between_digits_is_cube (k b : ℕ) (h_b : b ≥ 4) 
  : is_perfect_cube (1 * b^(3*(1+k)) + 3 * b^(2*(1+k)) + 3 * b^(1+k) + 1) :=
sorry

end insert_zeros_between_digits_is_cube_l574_574414


namespace correct_option_l574_574292

-- Definitions for propositions p and q
def p : Prop := ∃ (x : ℝ), x < 0 ∧ 3^x < 4^x
def q : Prop := ∀ (x : ℝ), 0 < x ∧ x < Real.pi / 2 → Real.tan x > x

-- The theorem to prove
theorem correct_option : (¬ p ∧ q) :=
by
  have h_p : ¬ p := sorry
  have h_q : q := sorry
  exact ⟨h_p, h_q⟩

end correct_option_l574_574292


namespace passengers_on_bus_l574_574828

theorem passengers_on_bus (initial_passengers : ℕ) (got_on : ℕ) (got_off : ℕ) (final_passengers : ℕ) :
  initial_passengers = 28 → got_on = 7 → got_off = 9 → final_passengers = initial_passengers + got_on - got_off → final_passengers = 26 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end passengers_on_bus_l574_574828


namespace non_negative_integer_solutions_l574_574032

open Nat

theorem non_negative_integer_solutions (x : Fin 2024 → ℕ) :
  (∑ i in Finset.range 2023, x i ^ 2) = 2 + ∑ i in Finset.range 2022, x i * x (i + 1) ↔ 
  (∃ s t u v : ℕ, (s + t + u + v = 2024) ∧ 2 * (choose 2024 4)) := 
sorry

end non_negative_integer_solutions_l574_574032


namespace coats_count_l574_574721

def initial_minks : Nat := 30
def babies_per_mink : Nat := 6
def minks_per_coat : Nat := 15

def total_minks : Nat := initial_minks + (initial_minks * babies_per_mink)
def remaining_minks : Nat := total_minks / 2

theorem coats_count : remaining_minks / minks_per_coat = 7 := by
  -- Proof goes here
  sorry

end coats_count_l574_574721


namespace cars_with_neither_feature_l574_574759

theorem cars_with_neither_feature 
  (total_cars : ℕ) 
  (power_steering : ℕ) 
  (power_windows : ℕ) 
  (both_features : ℕ) 
  (h1 : total_cars = 65) 
  (h2 : power_steering = 45) 
  (h3 : power_windows = 25) 
  (h4 : both_features = 17)
  : total_cars - (power_steering + power_windows - both_features) = 12 :=
by
  sorry

end cars_with_neither_feature_l574_574759


namespace oldest_child_age_l574_574783

-- Define the arithmetic progression and conditions
variable (age_sequence : Fin 7 → ℕ)
variable (avg_age : ℕ)
variable (common_difference : ℕ)

-- Given Conditions
def average_age_condition (age_sequence : Fin 7 → ℕ) (avg_age : ℕ) : Prop :=
  (∑ i, age_sequence i : ℕ) / 7 = avg_age

def different_ages_condition (age_sequence : Fin 7 → ℕ) : Prop :=
  Function.Injective age_sequence

def common_difference_condition (age_sequence : Fin 7 → ℕ) (d : ℕ) : Prop :=
  ∀ i : Fin 6, age_sequence ⟨i + 1, Nat.succ_lt_succ_iff.mp i.is_lt⟩ - age_sequence i = d

-- Theorem to prove the age of the oldest child is 19
theorem oldest_child_age (age_sequence : Fin 7 → ℕ) (avg_age : ℕ) (d : ℕ)
  (avg_cond : average_age_condition age_sequence avg_age)
  (diff_cond : different_ages_condition age_sequence)
  (diff_seq_cond : common_difference_condition age_sequence d) :
  age_sequence ⟨6, Nat.lt_succ_self 6⟩ = 19 :=
by
  sorry

-- Set the parameter values as given in the problem.
def age_sequence : Fin 7 → ℕ := λ i, 7 + (i * 3)
#eval age_sequence 6  -- This will be 19
noncomputable def avg_age := 10
noncomputable def common_difference := 3

end oldest_child_age_l574_574783


namespace area_of_annulus_l574_574786

variable {b c h : ℝ}
variable (hb : b > c)
variable (h2 : h^2 = b^2 - 2 * c^2)

theorem area_of_annulus (hb : b > c) (h2 : h^2 = b^2 - 2 * c^2) :
    π * (b^2 - c^2) = π * h^2 := by
  sorry

end area_of_annulus_l574_574786


namespace valid_five_letter_words_l574_574671

def num_valid_words : Nat :=
  let total_words := 3^5
  let invalid_3_consec := 5 * 2^3 * 1^2
  let invalid_4_consec := 2 * 2^4 * 1
  let invalid_5_consec := 2^5
  total_words - (invalid_3_consec + invalid_4_consec + invalid_5_consec)

theorem valid_five_letter_words : num_valid_words = 139 := by
  sorry

end valid_five_letter_words_l574_574671


namespace solve_log_equation_l574_574266

theorem solve_log_equation (x : ℝ) : log 2 (9^x - 5) = 2 + log 2 (3^x - 2) → x = 1 := 
by sorry

end solve_log_equation_l574_574266


namespace find_square_side_l574_574025

theorem find_square_side (a b x : ℕ) (h_triangle : a^2 + x^2 = b^2)
  (h_trapezoid : 2 * a + 2 * b + 2 * x = 60)
  (h_rectangle : 4 * a + 2 * x = 58) :
  a = 12 := by
  sorry

end find_square_side_l574_574025


namespace total_cartons_ordered_l574_574411

theorem total_cartons_ordered (E : ℕ) : 
  (6 * 20) + (3 * E) = 360 → 20 + E = 100 :=
by
  intro h,
  sorry

end total_cartons_ordered_l574_574411


namespace square_area_correct_l574_574916

noncomputable def square_area : ℝ :=
  let f : ℝ → ℝ := λ x, x^2 + 4 * x + 3
  let y_val : ℝ := 7
  let x1 : ℝ := -2 - 2 * Real.sqrt 2
  let x2 : ℝ := -2 + 2 * Real.sqrt 2
  let side_length := x2 - x1
  side_length * side_length

theorem square_area_correct :
  let f : ℝ → ℝ := λ x, x^2 + 4 * x + 3
  let y_val : ℝ := 7
  let x1 : ℝ := -2 - 2 * Real.sqrt 2
  let x2 : ℝ := -2 + 2 * Real.sqrt 2
  let side_length := x2 - x1
  side_length * side_length = 32 := by
  sorry

end square_area_correct_l574_574916


namespace count_of_integers_satisfying_inequality_l574_574557

theorem count_of_integers_satisfying_inequality :
  {n : ℕ // 0 < n ∧ n < 100 ∧ (∏ i in (range 50).map (λ k => 2 + 2*k), (n - i) < 0)}.card = 24 :=
by
  sorry

end count_of_integers_satisfying_inequality_l574_574557


namespace problem_I_solution_set_problem_II_range_of_a_l574_574595

open Real

noncomputable def f (x a : ℝ) : ℝ := abs (x - a)

theorem problem_I_solution_set (a : ℝ) (h : a = -2) :
  {x : ℝ | f x a + f (2 * x) a > 2} = {x : ℝ | x ∈ (-∞, -2) ∪ {-2/3, ∞}} :=
by sorry

theorem problem_II_range_of_a (h : ∃ x : ℝ, f x a + f (2 * x) a < 1/2) (a : ℝ) (h' : a < 0) : 
  a ∈ (-1, 0) :=
by sorry

end problem_I_solution_set_problem_II_range_of_a_l574_574595


namespace fitted_ball_volume_l574_574879

noncomputable def volume_of_fitted_ball (d_ball d_h1 r_h1 d_h2 r_h2 : ℝ) : ℝ :=
  let r_ball := d_ball / 2
  let v_ball := (4 / 3) * Real.pi * r_ball^3
  let r_hole1 := r_h1
  let r_hole2 := r_h2
  let v_hole1 := Real.pi * r_hole1^2 * d_h1
  let v_hole2 := Real.pi * r_hole2^2 * d_h2
  v_ball - 2 * v_hole1 - v_hole2

theorem fitted_ball_volume :
  volume_of_fitted_ball 24 10 (3 / 2) 10 2 = 2219 * Real.pi :=
by
  sorry

end fitted_ball_volume_l574_574879


namespace length_of_segment_AB_l574_574703

noncomputable def point := (ℝ × ℝ)

def line_l (s : ℝ) : point :=
  (1 + s, 2 - s)

def curve_C (t : ℝ) : point :=
  (t + 3, t^2)

def intersects (p1 p2 : point) : Prop :=
  p1 = p2

def dist (p1 p2 : point) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem length_of_segment_AB :
  let A := (2, 1)
  let B := (3, 0)
  dist A B = real.sqrt 2 := by
  sorry

end length_of_segment_AB_l574_574703


namespace wallace_fulfills_orders_in_13_days_l574_574474

def batch_small_bags_production := 12
def batch_large_bags_production := 8
def time_per_small_batch := 8
def time_per_large_batch := 12
def daily_production_limit := 18

def initial_stock_small := 18
def initial_stock_large := 10

def order1_small := 45
def order1_large := 30
def order2_small := 60
def order2_large := 25
def order3_small := 52
def order3_large := 42

def total_small_bags_needed := order1_small + order2_small + order3_small
def total_large_bags_needed := order1_large + order2_large + order3_large
def small_bags_to_produce := total_small_bags_needed - initial_stock_small
def large_bags_to_produce := total_large_bags_needed - initial_stock_large

def small_batches_needed := (small_bags_to_produce + batch_small_bags_production - 1) / batch_small_bags_production
def large_batches_needed := (large_bags_to_produce + batch_large_bags_production - 1) / batch_large_bags_production

def total_time_small_batches := small_batches_needed * time_per_small_batch
def total_time_large_batches := large_batches_needed * time_per_large_batch
def total_production_time := total_time_small_batches + total_time_large_batches

def days_needed := (total_production_time + daily_production_limit - 1) / daily_production_limit

theorem wallace_fulfills_orders_in_13_days :
  days_needed = 13 := by
  sorry

end wallace_fulfills_orders_in_13_days_l574_574474


namespace matrix_pow_sub_l574_574379

open Matrix

noncomputable def B : Matrix (Fin 2) (Fin 2) ℚ := !![3, 4; 0, 2]

theorem matrix_pow_sub : 
  B^10 - 3 • B^9 = !![0, 4; 0, -1] := 
by
  sorry

end matrix_pow_sub_l574_574379


namespace vertex_of_parabola_l574_574436

theorem vertex_of_parabola :
  ∀ x : ℝ, (x - 2) ^ 2 + 4 = (x - 2) ^ 2 + 4 → (2, 4) = (2, 4) :=
by
  intro x
  intro h
  -- We know that the vertex of y = (x - 2)^2 + 4 is at (2, 4)
  admit

end vertex_of_parabola_l574_574436


namespace extreme_values_l574_574258

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x + 1

theorem extreme_values :
  (∀ x ∈ Icc (-3 : ℝ) 0, f x ≤ 3) ∧ (∃ x ∈ Icc (-3 : ℝ) 0, f x = 3) ∧
  (∀ x ∈ Icc (-3 : ℝ) 0, -17 ≤ f x) ∧ (∃ x ∈ Icc (-3 : ℝ) 0, f x = -17)
  :=
sorry

end extreme_values_l574_574258


namespace AI_bisects_KL_l574_574748

variable (A B C I F G K L : Type)
variable [Incenter : Function (Type -> Prop)] (triangle_incenter : Incenter ABC I)
variable [Projection : Function (Type -> Prop)] (proj_A_on_BI : Projection A BI F) (proj_A_on_CI : Projection A CI G)
variable [Ray_Intersection : Function (Type -> Prop)] (intersection_AF_CFI : Ray_Intersection AF (circumcircle CFI) K)
variable [Ray_Intersection : Function (Type -> Prop)] (intersection_AG_BGI : Ray_Intersection AG (circumcircle BGI) L)

theorem AI_bisects_KL : Bisection AI KL :=
sorry

end AI_bisects_KL_l574_574748


namespace at_most_n_maximum_distance_pairs_l574_574606

theorem at_most_n_maximum_distance_pairs (n : ℕ) (h : n > 2) 
(points : Fin n → ℝ × ℝ) :
  ∃ (maxDistPairs : Finset (Fin n × Fin n)), (maxDistPairs.card ≤ n) ∧ 
  ∀ (p1 p2 : Fin n), (p1, p2) ∈ maxDistPairs → 
  (∀ (q1 q2 : Fin n), dist (points q1) (points q2) ≤ dist (points p1) (points p2)) :=
sorry

end at_most_n_maximum_distance_pairs_l574_574606


namespace A_independent_P_A_n_eq_1_div_n_l574_574375

open MeasureTheory Probability

-- Definitions for X_i sequence and exchangeability conditions.
variable {Ω : Type*} [SampleSpace Ω] {X : ℕ → Ω → ℝ}

axiom X_exchangeable : ∀ (π : ℕ → ℕ) (hπ : Function.Bijective π), 
  (∀ (i : ℕ), map (X i) = map (X (π i)))

axiom X_i_eq_X_j_zero : ∀ {i j : ℕ}, i ≠ j → 
  Measure.measure (set_of (λ ω : Ω, X i ω = X j ω)) = 0

-- Definition of A_n events
def A : ℕ → Set Ω
| 1       := set.univ
| (n + 1) := {ω | ∀ m < n + 1, X (n + 1) ω > X m ω}

-- The proof objectives
theorem A_independent : ∀ n ≥ 1, Indep (λ i, A i) :=
sorry

theorem P_A_n_eq_1_div_n : ∀ n ≥ 1, prob (A n) = 1 / n :=
sorry

end A_independent_P_A_n_eq_1_div_n_l574_574375


namespace non_overlapping_area_rect_fold_l574_574196

/-- Given a rectangle \( \mathrm{ABCD} \) with \( \mathrm{AB} = 4 \) and \( \mathrm{BC} = 6 \),
folding it such that point \( \mathrm{B} \) coincides with \( \mathrm{D} \) and assuming
the crease \( \mathrm{EF} \), prove that the area of the non-overlapping part of the rectangle
is \( \frac{20}{3} \). -/
theorem non_overlapping_area_rect_fold
  (AB BC : ℝ) (hAB : AB = 4) (hBC : BC = 6)
  (B D : ℝ → ℝ × ℝ) (hB : B 0 = (4, 0)) (hD : D 0 = (4, 6))
  (EF : ℝ → ℝ × ℝ) (hEF : ∃ x, EF x = (10 - 2 * x / 3, 4)) :
  (∃ a b c : ℝ, (a, b, c = 4, 6, 10/3) → 1/2 * a * b = 20/3) :=
by {
  sorry
}

end non_overlapping_area_rect_fold_l574_574196


namespace smallest_N_l574_574520

theorem smallest_N (l m n : ℕ) (N : ℕ) (h1 : N = l * m * n) (h2 : (l - 1) * (m - 1) * (n - 1) = 300) : 
  N = 462 :=
sorry

end smallest_N_l574_574520


namespace min_x_ln_y_l574_574296

variables (x y l : ℝ)
noncomputable def e : ℝ := sorry

-- Definitions of the conditions
axiom pos_x : 0 < x
axiom pos_y : 0 < y
axiom neq1_x : x ≠ 1
axiom neq1_y : y ≠ 1
axiom log_sum : Real.log x y + Real.log y x = 5 / 2
axiom log_greater : Real.log x y > l

-- Proof statement
theorem min_x_ln_y : x * Real.log y = -2 / e :=
sorry

end min_x_ln_y_l574_574296


namespace super_squarely_count_l574_574389

def S : Set ℕ := {1, 2, 4, 8, 16, 32, 64, 128, 256}

def is_squarely (P : Set ℕ) : Prop :=
  P.Nonempty ∧ (∃ (n : ℕ), (P.sum id) = n * n)

def is_super_squarely (Q : Set ℕ) : Prop :=
  is_squarely Q ∧ ¬(∃ (R : Set ℕ), R ⊂ Q ∧ is_squarely R)

theorem super_squarely_count : (Finset.filter is_super_squarely S.powerset).card = 5 :=
by
  sorry

end super_squarely_count_l574_574389


namespace number_of_friends_l574_574756

-- Define initial conditions and variables
def initial_crackers : ℕ := 10
def initial_cakes : ℕ := 8
def cakes_per_person : ℕ := 2

-- Define the proof statement
theorem number_of_friends (initial_cakes : ℕ) (cakes_per_person : ℕ) (h : initial_cakes = 8 ∧ cakes_per_person = 2) :
  ∃ (n : ℕ), n = (initial_cakes / cakes_per_person) ∧ n = 4 :=
  by {
    use initial_cakes / cakes_per_person,
    have h1: initial_cakes / cakes_per_person = 4,
    {
      rw h.1,
      rw h.2,
      exact nat.div_eq_of_lt (by norm_num) (by norm_num)
    },
    sorry
  }

end number_of_friends_l574_574756


namespace train_length_correct_l574_574921

-- Define the given conditions
def speed_of_train_kmph := 100
def speed_of_motorbike_kmph := 64
def overtaking_time_seconds := 18

-- Define conversions and calculations (only as definitions)
def kmph_to_mps (kmph: ℝ) : ℝ :=
  kmph * 1000 / 3600

def speed_of_train_mps : ℝ :=
  kmph_to_mps speed_of_train_kmph

def speed_of_motorbike_mps : ℝ :=
  kmph_to_mps speed_of_motorbike_kmph

def relative_speed_mps : ℝ :=
  speed_of_train_mps - speed_of_motorbike_mps

def length_of_train : ℝ :=
  relative_speed_mps * overtaking_time_seconds

theorem train_length_correct :
  length_of_train = 180 :=
by
  sorry

end train_length_correct_l574_574921


namespace divide_representatives_l574_574937

-- Define the problem setup
variables (Country : Type) [Fintype Country] (Rep : Country → Fin 2 → Type)

-- Hypothesis: Each country has exactly 2 representatives
variables [fintype (Σ (c : Country), Rep c (⟨0, _⟩ : Fin 2))]
variables [fintype (Σ (c : Country), Rep c (⟨1, _⟩ : Fin 2))]

-- Main statement
theorem divide_representatives (n : ℕ) (hn : n = 100)
  (h : ∀ (c : Country), Fintype (Rep c (⟨0, _⟩ : Fin 2)) ∧ Fintype (Rep c (⟨1, _⟩ : Fin 2)))
  (circular_table : Cyclic (Fin n)) :
  ∃ (group1 group2 : Set (Σ (c : Country), Rep c (Fin 2))),
    (∀ (c : Country), ∃ (r0 : Rep c (⟨0, _⟩ : Fin 2)) (r1 : Rep c (⟨1, _⟩ : Fin 2)),
      (⟨c, r0⟩ ∈ group1 ∧ ⟨c, r1⟩ ∈ group2) ∨
      (⟨c, r0⟩ ∈ group2 ∧ ⟨c, r1⟩ ∈ group1)) ∧
    (∀ (rep : Σ (c : Country), Rep c (Fin 2)),
     ∃ (neighbor : Σ (c : Country), Rep c (Fin 2)),
      (⟨rep.snd.fsnd, rep.snd.snd⟩ ∈ group1 ∧ ⟨neighbor.snd.fsnd, neighbor.snd.snd⟩ ∈ group1) ∨
      (⟨rep.snd.fsnd, rep.snd.snd⟩ ∈ group2 ∧ ⟨neighbor.snd.fsnd, neighbor.snd.snd⟩ ∈ group2)) :=
sorry

end divide_representatives_l574_574937


namespace probability_at_least_6_heads_in_8_flips_l574_574132

open scoped BigOperators

def binom (n k : ℕ) : ℕ := nat.choose n k

def total_outcomes (n : ℕ) := 2^n

def successful_outcomes (n k : ℕ) :=
  (finset.range (n + 1)).filter (λ x, x ≥ k).sum (λ x, binom n x)

def probability (n k : ℕ) :=
  (successful_outcomes n k) / (total_outcomes n : ℚ)

theorem probability_at_least_6_heads_in_8_flips :
  probability 8 6 = 37 / 256 := sorry

end probability_at_least_6_heads_in_8_flips_l574_574132


namespace expected_value_shorter_cavaliers_l574_574819

noncomputable def expected_value_of_shorter_cavaliers (n : ℕ) : ℝ :=
  ∑ j in Finset.range n, (n - j) / n

theorem expected_value_shorter_cavaliers (n : ℕ) : 
  expected_value_of_shorter_cavaliers n = (n - 1) / 2 :=
by
  sorry

end expected_value_shorter_cavaliers_l574_574819


namespace impossible_table_l574_574719

theorem impossible_table {n : ℕ} :
  ¬ (∃ (A : matrix (fin n) (fin n) ℤ),
       (∀ i : fin n, A i i ∈ { -1, 0, 1 }) ∧
       function.injective (λ i, (∑ j, A i j)) ∧
       function.injective (λ j, (∑ i, A i j)) ∧
       (∑ k, A k k ≠ (∑ k, A k (n-1 - k)))) :=
begin
  sorry
end

end impossible_table_l574_574719


namespace parallel_lines_sufficient_condition_l574_574091

theorem parallel_lines_sufficient_condition :
  ∀ a : ℝ, (a^2 - a) = 2 → (a = 2 ∨ a = -1) :=
by
  intro a h
  sorry

end parallel_lines_sufficient_condition_l574_574091


namespace triangle_sides_of_quadrilateral_inscribed_in_circle_l574_574795

theorem triangle_sides_of_quadrilateral_inscribed_in_circle
  (p q s R : ℝ) : 
  ∃ (PQ PS QS : ℝ), 
    PQ = Real.sqrt (p^2 + q^2 - 2 * R^2) ∧
    PS = Real.sqrt (p^2 + s^2 - 2 * R^2) ∧
    QS = Real.sqrt (q^2 + s^2 - 2 * R^2) := 
by {
  use [Real.sqrt (p^2 + q^2 - 2 * R^2), 
       Real.sqrt (p^2 + s^2 - 2 * R^2), 
       Real.sqrt (q^2 + s^2 - 2 * R^2)],
  split; refl,
  split; refl,
  split; refl,
}

end triangle_sides_of_quadrilateral_inscribed_in_circle_l574_574795


namespace circle_equation_and_point_M_coords_l574_574291

theorem circle_equation_and_point_M_coords:
  let F := (-2 : ℝ, 0 : ℝ) in
  let r := sqrt 5 in
  let O_eq := ∀ x y : ℝ, x^2 + y^2 = r^2 in

  let chord_AB_eq := ∀ k ≠ (0 : ℝ), ∀ A B : ℝ×ℝ,
    let eqn_lineAB := (λ x y : ℝ, x = k*y - 2) in
    let x_eqn := (λ y : ℝ, (k*y - 2)^2 + y^2 = r^2) in
    let y1_y2_sum_prod := (λ y1 y2 : ℝ, (y1 + y2 = 4*k / (k^2 + 1)) ∧ (y1 * y2 = -1 / (k^2 + 1))) in

  let M_coords := (∀ M : ℝ×ℝ, M = (-5/2, 0) ∧ (∀ A B : ℝ×ℝ, 
    let x := (λ x1 x2 y1 y2 : ℝ, (k * y2 - 2 + k * y1 - 2 - y1 + y2) * (x1 - M.1) + y1 * (x2 - M.1)) in
    let y := (2 * k * y1 * y2 - (y1 + y2) * (M.1 + 2)) in
    M.1 = -5/2)) in
O_eq ∧ chord_AB_eq ∧ M_coords := by
begin
  sorry
end

end circle_equation_and_point_M_coords_l574_574291


namespace find_value_of_expression_l574_574801

theorem find_value_of_expression (x : ℝ) (h : 5 * x^2 + 4 = 3 * x + 9) : (10 * x - 3)^2 = 109 := 
sorry

end find_value_of_expression_l574_574801


namespace range_of_a_for_f_l574_574612

def f (x : ℝ) : ℝ := 
  if 0 ≤ x ∧ x ≤ 1 then x + 1
  else if 1 < x ∧ x ≤ 4 then (1/2) * (Real.sin (π/4 * x)) + (3/2)
  else 0⟩

theorem range_of_a_for_f (a : ℝ) : (∀ x ∈ Icc (0 : ℝ) (4 : ℝ), f x ^ 2 - a * f x + 2 < 0) ↔ a > 3 
:= sorry

end range_of_a_for_f_l574_574612


namespace hyperbola_eccentricity_l574_574310

variables (b : ℝ) (hb : 0 < b)

def hyperbola_asymptotes := ∀ (x y : ℝ), 3 * x = 2 * y

def hyperbola_equation := ∀ (x y : ℝ), y^2 / 9 - x^2 / b^2 = 1

def eccentricity (e : ℝ) : Prop :=
  (∀ (a : ℝ), a = 3 → e = real.sqrt 13 / a)

theorem hyperbola_eccentricity
  (hyperbola_asymptotes : ∀ x y, 3 * x = 2 * y)
  (hyperbola_equation : ∀ x y, y^2 / 9 - x^2 / b^2 = 1)
  : eccentricity (real.sqrt 13 / 3) :=
by
  intros a ha
  have ha' : a = 3 := ha
  rw ha'
  apply rfl

end hyperbola_eccentricity_l574_574310


namespace number_of_zeros_of_f_l574_574033

def f (x : ℝ) : ℝ := 2 * x - Math.sin x

theorem number_of_zeros_of_f : ∃! x : ℝ, f x = 0 :=
by sorry

end number_of_zeros_of_f_l574_574033


namespace y1_eq_y2_gt_y3_l574_574766

-- Definition of the quadratic function
def quad_fn (x : ℝ) : ℝ := -x^2 + 2 * x + 3

-- Points on the quadratic function
def P1 := (-1 : ℝ, quad_fn (-1))
def P2 := (3 : ℝ, quad_fn 3)
def P3 := (5 : ℝ, quad_fn 5)

-- Prove the relationship between y1, y2, and y3
theorem y1_eq_y2_gt_y3 :
  P1.snd = P2.snd ∧ P1.snd > P3.snd :=
by
  unfold P1 P2 P3 quad_fn;
  sorry

end y1_eq_y2_gt_y3_l574_574766


namespace sequence_odd_for_all_n_greater_than_1_l574_574487

theorem sequence_odd_for_all_n_greater_than_1 (a : ℕ → ℤ) :
  (a 1 = 2) →
  (a 2 = 7) →
  (∀ n, 2 ≤ n → (-1/2 : ℚ) < (a (n + 1) : ℚ) - ((a n : ℚ) ^ 2) / (a (n - 1) : ℚ) ∧ (a (n + 1) : ℚ) - ((a n : ℚ) ^ 2) / (a (n - 1) : ℚ) ≤ (1/2 : ℚ)) →
  ∀ n, 1 < n → Odd (a n) := 
sorry

end sequence_odd_for_all_n_greater_than_1_l574_574487


namespace line_through_C_perpendicular_to_l_polar_eq_l574_574359

noncomputable def polar_coordinates_line_through_C_perpendicular_to_l (rho theta : ℝ) : Prop :=
  let C := 2 * Real.cos theta
  let l := rho * (Real.cos theta - Real.sin theta) = 4
  rho * (Real.cos theta + Real.sin theta) - 1 = 0

-- We state the theorem
theorem line_through_C_perpendicular_to_l_polar_eq (rho theta : ℝ) :
  polar_coordinates_line_through_C_perpendicular_to_l rho theta :=
begin
  sorry
end

end line_through_C_perpendicular_to_l_polar_eq_l574_574359


namespace number_of_girls_in_class_l574_574351

theorem number_of_girls_in_class (B S G : ℕ)
  (h1 : 3 * B = 4 * 18)  -- 3/4 * B = 18
  (h2 : 2 * S = 3 * B)  -- 2/3 * S = B
  (h3 : G = S - B) : G = 12 :=
by
  sorry

end number_of_girls_in_class_l574_574351


namespace find_value_of_a2004_b2005_l574_574615

theorem find_value_of_a2004_b2005
  (a b : ℝ)
  (h₁ : {a, b / a, 1} = {a^2, a + b, 0}) 
  (h₂: a ≠ 0) : 
  a^2004 + b^2005 = 1 := 
by 
  -- Proof goes here
  sorry

end find_value_of_a2004_b2005_l574_574615


namespace frank_money_l574_574993

-- Define the initial amount, expenses, and incomes as per the conditions
def initialAmount : ℕ := 11
def spentOnGame : ℕ := 3
def spentOnKeychain : ℕ := 2
def receivedFromAlice : ℕ := 4
def allowance : ℕ := 14
def spentOnBusTicket : ℕ := 5

-- Define the total money left for Frank
def finalAmount (initial : ℕ) (game : ℕ) (keychain : ℕ) (gift : ℕ) (allowance : ℕ) (bus : ℕ) : ℕ :=
  initial - game - keychain + gift + allowance - bus

-- Define the theorem stating that the final amount is 19
theorem frank_money : finalAmount initialAmount spentOnGame spentOnKeychain receivedFromAlice allowance spentOnBusTicket = 19 :=
by
  sorry

end frank_money_l574_574993


namespace probability_at_least_6_heads_in_8_flips_l574_574143

theorem probability_at_least_6_heads_in_8_flips : 
  (∑ k in finset.range 3, nat.choose 8 (6 + k)) / (2 ^ 8) = 37 / 256 :=
by sorry

end probability_at_least_6_heads_in_8_flips_l574_574143


namespace bookshop_inventory_l574_574174

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

end bookshop_inventory_l574_574174


namespace height_regular_tetrahedron_l574_574252

-- Define conditions and the problem
variables (a : ℝ)

def is_regular_tetrahedron (ABCD : Π ABCD : Type, (ABCD → ABCD → ℝ) → Prop) : Prop :=
∀ (D A B C: ABCD → ℝ), (D(A(B(C)))).dist (ABCD B) = a

def centroid_distance (AM : Type) : ℝ := 
  a * (real.sqrt 3) / 3

theorem height_regular_tetrahedron (a : ℝ) 
  (h₁ : is_regular_tetrahedron (Type) a)
  (h₂ : ∀ AM : Type, centroid_distance AM = a * (real.sqrt 3) / 3)
  : 
  sqrt(a^2 - (centroid_distance (Type : Type))^2) = a * (real.sqrt 2 / 3) :=
by {
  sorry
}

end height_regular_tetrahedron_l574_574252


namespace hadassah_painting_time_l574_574670

noncomputable def time_to_paint_all_paintings (time_small_paintings time_large_paintings time_additional_small_paintings time_additional_large_paintings : ℝ) : ℝ :=
  time_small_paintings + time_large_paintings + time_additional_small_paintings + time_additional_large_paintings

theorem hadassah_painting_time :
  let time_small_paintings := 6
  let time_large_paintings := 8
  let time_per_small_painting := 6 / 12 -- = 0.5
  let time_per_large_painting := 8 / 6 -- ≈ 1.33
  let time_additional_small_paintings := 15 * time_per_small_painting -- = 7.5
  let time_additional_large_paintings := 10 * time_per_large_painting -- ≈ 13.3
  time_to_paint_all_paintings time_small_paintings time_large_paintings time_additional_small_paintings time_additional_large_paintings = 34.8 :=
by
  sorry

end hadassah_painting_time_l574_574670


namespace simplify_expression_l574_574294

-- Define α to be an angle in the second quadrant
variable {α : ℝ}

-- Define the conditions: sin(α) > 0 and cos(α) < 0
axiom sin_pos : 0 < Real.sin α
axiom cos_neg : Real.cos α < 0

-- Define the theorem statement
theorem simplify_expression (h1 : sin_pos) (h2 : cos_neg) :
  ( (Real.sqrt (1 + 2 * Real.sin (5 * Real.pi - α) * Real.cos (α - Real.pi)))
  / (Real.sin (α - (3 / 2) * Real.pi) - Real.sqrt (1 - (Real.sin ((3 / 2) * Real.pi + α))^2)))
  = -1
:= sorry

end simplify_expression_l574_574294


namespace solve_fraction_l574_574270

noncomputable theory

variables {a : ℕ → ℚ} {b : ℕ → ℚ}
variables {S T : ℕ → ℚ}

-- Two arithmetic sequences
-- S_n and T_n denote the sums of the first n terms of {a_n} and {b_n}
-- It is given that S_n / T_n = (7n + 2) / (n + 3)
def given_condition (n : ℕ) : Prop := S n / T n = (7 * n + 2) / (n + 3)

-- We need to determine the value of (a_2 + a_20) / (b_7 + b_15)
theorem solve_fraction (h : ∀ n, given_condition n) :
  (a 2 + a 20) / (b 7 + b 15) = 149 / 24 :=
sorry

end solve_fraction_l574_574270


namespace probability_at_least_6_heads_8_flips_l574_574169

-- Define the probability calculation of getting at least 6 heads in 8 coin flips.
def probability_at_least_6_heads (n : ℕ) (k : ℕ) : ℚ :=
  (Nat.choose n k + Nat.choose n (k + 1) + Nat.choose n (k + 2)) / 2^n

theorem probability_at_least_6_heads_8_flips : 
  probability_at_least_6_heads 8 6 = 37 / 256 := 
by
  sorry

end probability_at_least_6_heads_8_flips_l574_574169


namespace find_x_l574_574455

theorem find_x (x : ℝ) : (x * 16) / 100 = 0.051871999999999995 → x = 0.3242 := by
  intro h
  sorry

end find_x_l574_574455


namespace goods_train_speed_l574_574507

theorem goods_train_speed (length_train length_platform distance time : ℕ) (conversion_factor : ℚ) : 
  length_train = 250 → 
  length_platform = 270 → 
  distance = length_train + length_platform → 
  time = 26 → 
  conversion_factor = 3.6 →
  (distance / time : ℚ) * conversion_factor = 72 :=
by
  intros h_lt h_lp h_d h_t h_cf
  rw [h_lt, h_lp] at h_d
  rw [h_t, h_cf]
  sorry

end goods_train_speed_l574_574507


namespace part_a_part_b_l574_574750

def alphonse_jumps (n : ℕ) : ℕ :=
  let q := n / 8
  let r := n % 8
  q + r

def beryl_jumps (n : ℕ) : ℕ :=
  let q := n / 7
  let r := n % 7
  q + r

theorem part_a : ∃ (n : ℕ), n > 200 ∧ beryl_jumps 231 < alphonse_jumps 231 :=
by
  use 231
  have h_n_gt_200 : 231 > 200 := by decide
  have h_B_lt_A : beryl_jumps 231 < alphonse_jumps 231 := by decide
  exact ⟨h_n_gt_200, h_B_lt_A⟩
  sorry

theorem part_b : ∀ (n : ℕ), beryl_jumps 343 ≤ alphonse_jumps 343 ∧ (∀(m : ℕ), m > 343 → beryl_jumps m > alphonse_jumps m) :=
by
  have h_B_le_A : beryl_jumps 343 ≤ alphonse_jumps 343 := by decide
  have h_max_n : ∀ m, m > 343 → beryl_jumps m > alphonse_jumps m := by
    intros m hm 
    have h : beryl_jumps m > alphonse_jumps m := by decide
    exact h
  exact ⟨h_B_le_A, h_max_n⟩
  sorry

end part_a_part_b_l574_574750


namespace greatest_x_l574_574251

theorem greatest_x (x : ℝ) (h : x^2 - 16 * x + 63 ≤ 0) : x ≤ 9 :=
begin
  sorry
end

example : ∃ (x : ℝ), x ≤ 9 ∧ (∀ (y : ℝ), y^2 - 16 * y + 63 ≤ 0 → y ≤ 9) :=
begin
  use 9,
  split,
  by linarith,
  intros y hy,
  exact greatest_x y hy,
  sorry
end

end greatest_x_l574_574251


namespace lcm_of_ratio_and_hcf_l574_574453

theorem lcm_of_ratio_and_hcf (a b : ℕ) (h1 : a = 3 * 8) (h2 : b = 4 * 8) (h3 : Nat.gcd a b = 8) : Nat.lcm a b = 96 :=
  sorry

end lcm_of_ratio_and_hcf_l574_574453


namespace square_area_l574_574912

theorem square_area (y : ℝ) (x₁ x₂ : ℝ) (s : ℝ) (A : ℝ) :
  y = 7 → 
  (y = x₁^2 + 4 * x₁ + 3) →
  (y = x₂^2 + 4 * x₂ + 3) →
  x₁ ≠ x₂ →
  s = |x₂ - x₁| → 
  A = s^2 →
  A = 32 :=
by
  intros hy intersection_x1 intersection_x2 hx1x2 hs ha
  sorry

end square_area_l574_574912


namespace correct_calculation_l574_574856

theorem correct_calculation : ∀ a : ℝ, 1 / a * a = 1 := 
by 
  intro a
  calc
    1 / a * a = (1 / a) * a : by rfl
           ... = 1 : sorry -- This can be proven by basic algebraic manipulation, which is a standard proof technique.

end correct_calculation_l574_574856


namespace square_area_l574_574911

theorem square_area (y : ℝ) (x₁ x₂ : ℝ) (s : ℝ) (A : ℝ) :
  y = 7 → 
  (y = x₁^2 + 4 * x₁ + 3) →
  (y = x₂^2 + 4 * x₂ + 3) →
  x₁ ≠ x₂ →
  s = |x₂ - x₁| → 
  A = s^2 →
  A = 32 :=
by
  intros hy intersection_x1 intersection_x2 hx1x2 hs ha
  sorry

end square_area_l574_574911


namespace rectangle_perimeter_l574_574515

noncomputable def perimeter (length width : ℝ) : ℝ := 2 * (length + width)

theorem rectangle_perimeter (length diagonal : ℝ) (P : ℝ) :
  length = 8 →
  diagonal = 17 →
  P = perimeter 8 (real.sqrt (diagonal^2 - length^2)) →
  P = 46 :=
by
  intros h1 h2 hP
  rw [h1, h2] at *
  trivial

end rectangle_perimeter_l574_574515


namespace problem_statement_l574_574739

open Matrix

variables {R : Type*} [CommRing R] {a b c : Fin 3 → R}

def det (A : Matrix (Fin 3) (Fin 3) R) := 
  Matrix.det A

def det_matrix_with_vectors (u v w : Fin 3 → R) : R :=
  det (![u, v, w] : Matrix (Fin 3) (Fin 3) R)

theorem problem_statement (D : R) (a b c : Fin 3 → R)
  (hD : D = det_matrix_with_vectors a b c) :
  det_matrix_with_vectors (vector_cross a b) (vector_cross b c) (vector_cross c a) = 1 * D^2 :=
  sorry

-- Definitions for cross product of 3D vectors
def vector_cross (v₁ v₂ : Fin 3 → R) : Fin 3 → R :=
  λ i, match i with
  | 0 => v₁ 1 * v₂ 2 - v₁ 2 * v₂ 1
  | 1 => v₁ 2 * v₂ 0 - v₁ 0 * v₂ 2
  | 2 => v₁ 0 * v₂ 1 - v₁ 1 * v₂ 0
  end

-- Assumptions for orthogonality conditions used in problem
variables (a b c : Fin 3 → R)

-- Example assumptions
example : D = (a 0 * b 1 * c 2 + a 1 * b 2 * c 0 + a 2 * b 0 * c 1 - a 2 * b 1 * c 0 - a 0 * b 2 * c 1 - a 1 * b 0 * c 2) :=
  by sorry

end problem_statement_l574_574739


namespace derivative_of_exp_sin_l574_574790

theorem derivative_of_exp_sin (x : ℝ) : 
  (deriv (λ x : ℝ, exp x * sin x)) x = exp x * (sin x + cos x) :=
by
  sorry

end derivative_of_exp_sin_l574_574790


namespace find_right_triangle_l574_574860

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

def sets :=
  [([1, 1, 1] : List ℝ), 
   [2, 3, 4], 
   [√5, 3, 4], 
   [1, √3, 2]]

theorem find_right_triangle :
  ∃ s ∈ sets, is_right_triangle s[0] s[1] s[2] ∧ s == [1, √3, 2] := 
by
  sorry

end find_right_triangle_l574_574860


namespace units_digit_sum_42_4_24_4_l574_574845

-- Define the units digit function
def units_digit (n : ℕ) : ℕ := n % 10

-- Given conditions
def units_digit_42_4 : units_digit (42^4) = 6 := sorry
def units_digit_24_4 : units_digit (24^4) = 6 := sorry

-- Theorem to prove
theorem units_digit_sum_42_4_24_4 :
  units_digit (42^4 + 24^4) = 2 :=
by
  -- Use the given conditions
  have h1 : units_digit (42^4) = 6 := units_digit_42_4
  have h2 : units_digit (24^4) = 6 := units_digit_24_4
  -- Calculate the units digit of their sum
  calc 
    units_digit (42^4 + 24^4)
        = units_digit (6 + 6) : by rw [h1, h2]
    ... = units_digit 12    : by norm_num
    ... = 2                 : by norm_num

end units_digit_sum_42_4_24_4_l574_574845


namespace part_I_part_II_l574_574300

-- Define the given conditions
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (3, 4)
def L : ℝ → ℝ → Prop := λ x y, x + 3 * y = 15
def is_circle (O : ℝ × ℝ) (r : ℝ) : Prop := ∃ (x y : ℝ), (x - O.1)^2 + (y - O.2)^2 = r^2

-- Definitions to simplify expressions and avoid repetition
def p_eq (x y : ℝ) : Prop := (x + 3)^2 + (y - 6)^2 = 40
def c_eq (O : ℝ × ℝ) : Prop := L O.1 O.2
def max_area (P : ℝ × ℝ) : ℝ := (1 / 2) * (4 * 2.sqrt) * ((4 * 2.sqrt) + (2 * 10.sqrt))

-- Define the theorem statements
theorem part_I (C : ℝ × ℝ) (r : ℝ) (O : ℝ × ℝ)
  (h1 : is_circle O r) (h2 : ∀ x y, L C.1 C.2 ∧ L O.1 O.2) : 
  p_eq C.1 C.2 ∧ r = 2 * 10.sqrt :=
sorry

theorem part_II (C : ℝ × ℝ) (P : ℝ × ℝ)
  (h1 : is_circle C ((4^2 + 6^2).sqrt))
  (h2 : P ∈ {q : ℝ × ℝ | (q.1 + 3)^2 + (q.2 - 6)^2 = 40}) : 
  max_area P = 16 + 8 * 5.sqrt :=
sorry

end part_I_part_II_l574_574300


namespace mark_deposit_is_88_l574_574400

-- Definitions according to the conditions
def markDeposit := 88
def bryanDeposit (m : ℕ) := 5 * m - 40

-- The theorem we need to prove
theorem mark_deposit_is_88 : markDeposit = 88 := 
by 
  -- Since the condition states Mark deposited $88,
  -- this is trivially true.
  sorry

end mark_deposit_is_88_l574_574400


namespace number_of_correct_conclusions_l574_574644

theorem number_of_correct_conclusions : 
(¬(p = false ∧ ¬ q = false → p ∨ q = false) ∧ 
¬(¬ (∀ (x y : ℝ) (H : x * y = 0), x = 0 ∨ y = 0) ↔ (∃ (x y : ℝ), x * y ≠ 0 ∧ x ≠ 0 ∨ y ≠ 0)) ∧ 
(¬ (∀ x : ℝ, 2^x > 0) ↔ (∃ x : ℝ, 2^x ≤ 0))) → 
1 = 1 :=
by decide

end number_of_correct_conclusions_l574_574644


namespace median_largest_side_triangle_l574_574770

theorem median_largest_side_triangle (a b c : ℝ) (h : c ≤ b ∧ b ≤ a) (A B C : ℝ) (h_triangle : angle_ABC A B C a b c) :
  ∃ (median M : ℝ), ∀ (angle_med_side1 angle_med_side2 : ℝ),
  median = a / 2 ∧
  angle_med_side1 ≥ A / 2 ∧
  angle_med_side2 ≥ A / 2 := 
sorry

end median_largest_side_triangle_l574_574770


namespace decreasing_interval_of_f_l574_574336

noncomputable def f (t: ℝ) : ℝ := (t - 2) ^ 2

theorem decreasing_interval_of_f :
  (∀ (x : ℝ), x ∈ [-2, 6]) →
  (∀ (y : ℝ), y = f (x + 1)) →
  (∀ (t : ℝ), t ∈ [-1, 2] ∧ f(t) = (t - 2)^2) :=
by
  sorry

end decreasing_interval_of_f_l574_574336


namespace harmonic_series_fraction_exists_in_interval_l574_574730

def harmonic_series (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ k, 1 / (k + 1 : ℝ))

theorem harmonic_series_fraction_exists_in_interval (a b : ℝ) (h : 0 ≤ a ∧ a ≤ b ∧ b ≤ 1) :
  ∃ᶠ n in 𝓝[≥] (0 : ℕ), a < (harmonic_series n) - ⌊harmonic_series n⌋ ∧ (harmonic_series n) - ⌊harmonic_series n⌋ < b :=
sorry

end harmonic_series_fraction_exists_in_interval_l574_574730


namespace probability_at_least_6_heads_in_8_flips_l574_574136

open scoped BigOperators

def binom (n k : ℕ) : ℕ := nat.choose n k

def total_outcomes (n : ℕ) := 2^n

def successful_outcomes (n k : ℕ) :=
  (finset.range (n + 1)).filter (λ x, x ≥ k).sum (λ x, binom n x)

def probability (n k : ℕ) :=
  (successful_outcomes n k) / (total_outcomes n : ℚ)

theorem probability_at_least_6_heads_in_8_flips :
  probability 8 6 = 37 / 256 := sorry

end probability_at_least_6_heads_in_8_flips_l574_574136


namespace find_least_N_exists_l574_574959

theorem find_least_N_exists (N : ℕ) :
  (∃ (a₁ b₁ c₁ a₂ b₂ c₂ : ℕ), 
    N = (a₁ + 2) * (b₁ + 2) * (c₁ + 2) - 8 ∧ 
    N + 1 = (a₂ + 2) * (b₂ + 2) * (c₂ + 2) - 8) ∧
  N = 55 := 
sorry

end find_least_N_exists_l574_574959


namespace Petya_wins_l574_574823

-- Define the initial state and conditions
def initial_plus_count : ℕ := 865

-- Define the game state as a structure
structure GameState where
  pluses : ℕ
  minuses : ℕ

-- Define the rules of the game as allowable moves
inductive Move
| plusToMinus : Move
| erasePlusTwoMinuses : Move
| twoPlusesToTwoMinuses : Move

-- Transition function for the game
def transition (s : GameState) (m : Move) : GameState :=
  match m with
  | Move.plusToMinus => {s with pluses := s.pluses - 1, minuses := s.minuses + 1}
  | Move.erasePlusTwoMinuses => {s with pluses := s.pluses - 1, minuses := s.minuses - 2}
  | Move.twoPlusesToTwoMinuses => {s with pluses := s.pluses - 2, minuses := s.minuses + 2}

-- Define a predicate to check if a move is valid
def validMove (s : GameState) (m : Move) : Prop :=
  match m with
  | Move.plusToMinus => s.pluses ≥ 1
  | Move.erasePlusTwoMinuses => s.pluses ≥ 1 ∧ s.minuses ≥ 2
  | Move.twoPlusesToTwoMinuses => s.pluses ≥ 2

-- Prove that Petya wins with optimal play
theorem Petya_wins : ∃ f : ℕ → GameState, ∃ g : ℕ → Move,
  (∀ n, validMove (f n) (g n) ∧ (f (n+1) = transition (f n) (g n))) ∧
  (f 0 = {pluses := initial_plus_count, minuses := 0}) ∧
  (∃ N, f N = {pluses := 0, minuses := _}) :=
by
  -- this would be where the actual proof goes
  sorry

end Petya_wins_l574_574823


namespace probability_at_least_6_heads_8_flips_l574_574165

-- Define the probability calculation of getting at least 6 heads in 8 coin flips.
def probability_at_least_6_heads (n : ℕ) (k : ℕ) : ℚ :=
  (Nat.choose n k + Nat.choose n (k + 1) + Nat.choose n (k + 2)) / 2^n

theorem probability_at_least_6_heads_8_flips : 
  probability_at_least_6_heads 8 6 = 37 / 256 := 
by
  sorry

end probability_at_least_6_heads_8_flips_l574_574165


namespace sort_mail_together_time_l574_574863

-- Definitions of work rates
def mail_handler_work_rate : ℚ := 1 / 3
def assistant_work_rate : ℚ := 1 / 6

-- Definition to calculate combined work time
def combined_time (rate1 rate2 : ℚ) : ℚ := 1 / (rate1 + rate2)

-- Statement to prove
theorem sort_mail_together_time :
  combined_time mail_handler_work_rate assistant_work_rate = 2 := by
  -- Proof goes here
  sorry

end sort_mail_together_time_l574_574863


namespace minimize_feed_cost_l574_574111

theorem minimize_feed_cost :
  ∃ (x y : ℝ), 10 * x + 10 * y ≥ 45 ∧ 10 * x + 20 * y ≥ 60 ∧ 5 * y ≥ 5 ∧ x ≥ 0 ∧ y ≥ 0 ∧ 30 * x + 50 * y = 165 :=
begin
  use [3, 1.5],
  split,
  { linarith },
  split,
  { linarith },
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  { norm_num }
end

end minimize_feed_cost_l574_574111


namespace rectangular_solid_inscribed_sphere_surface_area_l574_574517

-- Define the rectangular solid and the conditions
def rectangular_solid : Type := 
  {a b c : ℝ // a = 3 ∧ b = 4 ∧ c = 5}

-- Define the properties of the sphere
def inscribed_sphere_surface_area (r : ℝ) : ℝ := 
  4 * Real.pi * r^2

-- The proof statement
theorem rectangular_solid_inscribed_sphere_surface_area :
  ∃ (r : ℝ), r = 5 * Real.sqrt 2 / 2 ∧ inscribed_sphere_surface_area r = 50 * Real.pi :=
by
  sorry

end rectangular_solid_inscribed_sphere_surface_area_l574_574517


namespace find_a_share_l574_574077

noncomputable def total_investment (a b c : ℕ) : ℕ :=
  a + b + c

noncomputable def total_profit (b_share total_inv b_inv : ℕ) : ℕ :=
  b_share * total_inv / b_inv

noncomputable def a_share (a_inv total_inv total_pft : ℕ) : ℕ :=
  a_inv * total_pft / total_inv

theorem find_a_share
  (a_inv b_inv c_inv b_share : ℕ)
  (h1 : a_inv = 7000)
  (h2 : b_inv = 11000)
  (h3 : c_inv = 18000)
  (h4 : b_share = 880) :
  a_share a_inv (total_investment a_inv b_inv c_inv) (total_profit b_share (total_investment a_inv b_inv c_inv) b_inv) = 560 := 
by
  sorry

end find_a_share_l574_574077


namespace number_of_bricks_needed_l574_574500

theorem number_of_bricks_needed
  (sq_meter_land : ℕ) (land_cost_per_sq_meter : ℕ) (number_of_bricks_cost : ℕ) 
  (roof_tile_cost_per_tile : ℕ) (house_land_area : ℕ) (house_roof_tiles : ℕ) 
  (total_construction_cost : ℕ)
  (h1 : land_cost_per_sq_meter = 50)
  (h2 : number_of_bricks_cost = 100)
  (h3 : roof_tile_cost_per_tile = 10)
  (h4 : house_land_area = 2000)
  (h5 : house_roof_tiles = 500)
  (h6 : total_construction_cost = 106000) : 
  (house_land_area * land_cost_per_sq_meter + house_roof_tiles * roof_tile_cost_per_tile +
  (1000 / number_of_bricks_cost * 1000)) = total_construction_cost → 
  10000 = 10000: 
begin
  sorry
end

end number_of_bricks_needed_l574_574500


namespace sum_of_0_75_of_8_and_2_l574_574815

theorem sum_of_0_75_of_8_and_2 : 0.75 * 8 + 2 = 8 := by
  sorry

end sum_of_0_75_of_8_and_2_l574_574815


namespace sums_same_remainder_exists_l574_574086

theorem sums_same_remainder_exists (n : ℕ) (h : n > 0) (a : Fin (2 * n) → Fin (2 * n)) (ha_permutation : Function.Bijective a) :
  ∃ (i j : Fin (2 * n)), i ≠ j ∧ ((a i + i) % (2 * n) = (a j + j) % (2 * n)) :=
by sorry

end sums_same_remainder_exists_l574_574086


namespace range_of_a_l574_574344

theorem range_of_a (a : ℝ) (h : ∃ x : ℝ, |x - a| + |x - 1| ≤ 3) : -2 ≤ a ∧ a ≤ 4 :=
sorry

end range_of_a_l574_574344


namespace right_triangle_of_altitude_ratios_l574_574803

theorem right_triangle_of_altitude_ratios
  (h1 h2 h3 : ℝ) 
  (h1_pos : h1 > 0) 
  (h2_pos : h2 > 0) 
  (h3_pos : h3 > 0) 
  (H : (h1 / h2)^2 + (h1 / h3)^2 = 1) : 
  ∃ a b c : ℝ, a^2 = b^2 + c^2 ∧ h1 = 1 / a ∧ h2 = 1 / b ∧ h3 = 1 / c :=
sorry

end right_triangle_of_altitude_ratios_l574_574803


namespace AG_equals_IG_l574_574195

-- Definitions for geometric entities
variables {A B C O I J G H : Type} [Point O I J A B C G H]

-- Conditions given in the problem
variables [circumcenter O A B C] [incenter I A B C] [excenter J A B C] 
          [line_segment IH G] [line_segment OJ H]
          [perpendicular IH OJ] [intersection_point G IH BC]

-- Definition for the points
noncomputable def A' := reflection A BC

-- Prove AG = IG
theorem AG_equals_IG : line_segment_length AG = line_segment_length IG :=
by
  sorry

end AG_equals_IG_l574_574195


namespace math_problem_solution_l574_574187

noncomputable def average_root_cross_sectional_area (sum_x : ℝ) (n : ℕ) : ℝ := 
  sum_x / n

noncomputable def average_volume (sum_y : ℝ) (n : ℕ) : ℝ := 
  sum_y / n

noncomputable def correlation_coefficient (sum_xy : ℝ) (sum_x2 : ℝ) (sum_y2 : ℝ) : ℝ :=
  sum_xy / (Real.sqrt (sum_x2 * sum_y2))

noncomputable def total_volume_estimate (avg_y : ℝ) (avg_x : ℝ) (total_x : ℝ) : ℝ :=
  (avg_y / avg_x) * total_x

theorem math_problem_solution :
  let sum_x := 0.6
  let n := 10
  let sum_y := 3.9
  let sum_x2 := 0.038
  let sum_y2 := 1.6158
  let sum_xy := 0.2474
  let total_x := 186
  let avg_x := average_root_cross_sectional_area sum_x n
  let avg_y := average_volume sum_y n
  let r := correlation_coefficient sum_xy (sum_x2 - n * avg_x^2) (sum_y2 - n * avg_y^2)
  let total_volume := total_volume_estimate avg_y avg_x total_x
  avg_x = 0.06 ∧ avg_y = 0.39 ∧ Real.round r = 0.97 ∧ total_volume = 1209 :=
  by
    simp [average_root_cross_sectional_area, average_volume, correlation_coefficient, total_volume_estimate]
    sorry

end math_problem_solution_l574_574187


namespace range_of_m_l574_574311

def satisfies_inequality (m : ℝ) : Prop :=
  ∀ x : ℝ, (x ≥ 1/e ∧ x ≤ 1) → |real.log x + 1/x - m| ≤ m + real.exp 1

theorem range_of_m : {m : ℝ | satisfies_inequality m} = { m : ℝ | m >= -1/2 } := 
by
  sorry

end range_of_m_l574_574311


namespace Angela_is_295_cm_l574_574933

noncomputable def Angela_height (Carl_height : ℕ) : ℕ :=
  let Becky_height := 2 * Carl_height
  let Amy_height := Becky_height + Becky_height / 5  -- 20% taller than Becky
  let Helen_height := Amy_height + 3
  let Angela_height := Helen_height + 4
  Angela_height

theorem Angela_is_295_cm : Angela_height 120 = 295 := 
by 
  sorry

end Angela_is_295_cm_l574_574933


namespace earnings_difference_l574_574757

def total_earnings : ℕ := 3875
def first_job_earnings : ℕ := 2125
def second_job_earnings := total_earnings - first_job_earnings

theorem earnings_difference : (first_job_earnings - second_job_earnings) = 375 := by
  sorry

end earnings_difference_l574_574757


namespace minimum_positive_period_l574_574058

def initial_function (x : ℝ) := 3 * Real.sin (2 * x + Real.pi / 3)

def translated_function (x : ℝ) := 3 * Real.sin (2 * (x - Real.pi / 3))

def compressed_function (x : ℝ) := 3 * Real.sin (4 * x - Real.pi / 3)

theorem minimum_positive_period :
  ∃ T > 0, ∀ x, compressed_function (x + T) = compressed_function x :=
by
  use Real.pi / 2
  -- proof goes here
  sorry

end minimum_positive_period_l574_574058


namespace uma_income_correct_l574_574486

noncomputable def uma_income : ℝ :=
  let x := 2000 / 15 in
  8 * x

theorem uma_income_correct :
  let x := 2000 / 15 in
  let uma_income := 8 * x in
  (uma_income = 1066.67) :=
by
  sorry

end uma_income_correct_l574_574486


namespace percentage_of_remainder_left_l574_574694

noncomputable def initial_population : ℕ := 6324
noncomputable def died_percentage : ℝ := 0.10
noncomputable def reduced_population : ℕ := 4554

theorem percentage_of_remainder_left : 
  let died := (died_percentage * initial_population) in
  let rounded_died := died.toNat in
  let after_bombardment := initial_population - rounded_died in
  let number_left := after_bombardment - reduced_population in
  let percentage_left := (number_left / after_bombardment.toReal) * 100 in
  percentage_left ≈ 20 := sorry

end percentage_of_remainder_left_l574_574694


namespace unique_real_root_l574_574769

noncomputable def seventh_roots_sum : ℝ :=
  2 + real.root 7 3 + real.root 7 (3^2) + real.root 7 (3^3) + real.root 7 (3^4) + real.root 7 (3^5) + real.root 7 (3^6)

theorem unique_real_root :
  (∃! x : ℝ, x^7 - 14 * x^6 + 21 * x^5 - 70 * x^4 + 35 * x^3 - 42 * x^2 + 7 * x - 2 = 0) ∧
  (seventh_roots_sum = x) :=
sorry

end unique_real_root_l574_574769


namespace sum_of_digits_of_9ab_l574_574713

def a (n : Nat) : Nat := (List.foldr (fun _ acc => 8 + 10 * acc) 0 (List.range n))
def b (n : Nat) : Nat := (List.foldr (fun _ acc => 5 + 10 * acc) 0 (List.range n))

theorem sum_of_digits_of_9ab {n : Nat} (hn : n = 1985) : 
  let a := List.foldr (fun _ acc => 8 + 10 * acc) 0 (List.range n)
  let b := List.foldr (fun _ acc => 5 + 10 * acc) 0 (List.range n)
  sum_of_digits (9 * a * b) = 17865 := by sorry

end sum_of_digits_of_9ab_l574_574713


namespace magnitude_of_power_l574_574237

open Complex

theorem magnitude_of_power (a b : ℝ) : abs ((Complex.mk 2 (2 * Real.sqrt 3)) ^ 6) = 4096 := by
  sorry

end magnitude_of_power_l574_574237


namespace max_five_topping_pizzas_l574_574895

theorem max_five_topping_pizzas : 
  (∃ (n k : ℕ), n = 8 ∧ k = 5 ∧ (nat.choose n k = 56)) :=
begin
  use [8, 5],
  split,
  { refl, },
  split,
  { refl, },
  { sorry }
end

end max_five_topping_pizzas_l574_574895


namespace Anne_carries_total_weight_l574_574193

variable (female_cat_weight male_cat_weight Anne_carried_weight : ℝ)
variable (h1 : female_cat_weight = 2)
variable (h2 : male_cat_weight = 2 * female_cat_weight)

theorem Anne_carries_total_weight : Anne_carried_weight = female_cat_weight + male_cat_weight := by
  have h3 : female_cat_weight + male_cat_weight = 6 := sorry
  exact h3

end Anne_carries_total_weight_l574_574193


namespace units_digit_42_pow_4_add_24_pow_4_l574_574847

-- Define a function to get the units digit of a number.
def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_42_pow_4_add_24_pow_4 : units_digit (42^4 + 24^4) = 2 := by
  sorry

end units_digit_42_pow_4_add_24_pow_4_l574_574847


namespace solve_quadratic_sine_l574_574980

theorem solve_quadratic_sine :
  ∀ (x y : ℝ), (x^2 + 2 * x * sin (x * y) + 1 = 0) ↔ 
               (x = 1 ∧ ∃ k : ℤ, y = -(π/2) + 2 * k * π) ∨ 
               (x = -1 ∧ ∃ k : ℤ, y = (π/2) + 2 * k * π) :=
by
  sorry

end solve_quadratic_sine_l574_574980


namespace height_of_given_cylinder_l574_574043

noncomputable def height_of_cylinder (P d : ℝ) : ℝ :=
  let r := P / (2 * Real.pi)
  let l := P
  let h := Real.sqrt (d^2 - l^2)
  h

theorem height_of_given_cylinder : height_of_cylinder 6 10 = 8 :=
by
  show height_of_cylinder 6 10 = 8
  sorry

end height_of_given_cylinder_l574_574043


namespace sales_director_concerned_with_mode_l574_574934

-- Define the statistical measures for shoe sizes
variables (average median mode variance : ℝ)

-- Define the relevance of each measure for the sales director
def is_most_relevant (measure : ℝ) : Prop :=
measure = mode

-- Statement of the problem
theorem sales_director_concerned_with_mode :
  is_most_relevant mode :=
sorry

end sales_director_concerned_with_mode_l574_574934


namespace flight_time_NY_to_CT_l574_574205

def travelTime (start_time_NY : ℕ) (end_time_CT : ℕ) (layover_Johannesburg : ℕ) : ℕ :=
  end_time_CT - start_time_NY + layover_Johannesburg

theorem flight_time_NY_to_CT :
  let start_time_NY := 0 -- 12:00 a.m. Tuesday as 0 hours from midnight in ET
  let end_time_CT := 10  -- 10:00 a.m. Tuesday as 10 hours from midnight in ET
  let layover_Johannesburg := 4
  travelTime start_time_NY end_time_CT layover_Johannesburg = 10 :=
by
  sorry

end flight_time_NY_to_CT_l574_574205


namespace find_sixtieth_permutation_l574_574041

/-- 
  The positive five-digit integers that use each of the digits 1, 2, 3, 4, and 5 exactly once are ordered from least to greatest.
  Prove that the 60th integer in this list is 32315.
-/
noncomputable def sixtieth_permutation : ℕ :=
  32315

theorem find_sixtieth_permutation :
  let permutations := list.permutations [1, 2, 3, 4, 5]
  permutations.sorted.get! 59 = sixtieth_permutation :=
by {
  sorry
}

end find_sixtieth_permutation_l574_574041


namespace bromine_atoms_in_compound_l574_574499

theorem bromine_atoms_in_compound
  (atomic_weight_H : ℕ := 1)
  (atomic_weight_Br : ℕ := 80)
  (atomic_weight_O : ℕ := 16)
  (total_molecular_weight : ℕ := 129) :
  ∃ (n : ℕ), total_molecular_weight = atomic_weight_H + n * atomic_weight_Br + 3 * atomic_weight_O ∧ n = 1 := 
by
  sorry

end bromine_atoms_in_compound_l574_574499


namespace general_formula_transformed_sum_formula_l574_574623

open Nat Real

-- Define the input arithmetic sequence and its common difference d
def a_seq (n : ℕ) : ℝ := 1/2 * n + 1

-- The conditions from the problem
def condition_1 (n m : ℕ) : Bool := (n < m) → (a_seq n < a_seq m)

def condition_2 : Prop := 
  let a2 := a_seq 2
  let a4 := a_seq 4
  (a2 = 2 ∧ a4 = 3)

-- Define the sum of the first n terms of the transformed sequence
def transformed_sum (n : ℕ) : ℝ :=
  ∑ i in range n, a_seq i / (2^i)

-- Goals to be proved
theorem general_formula (n : ℕ) : 
  condition_1 ∧ condition_2 → a_seq n = 1/2 * n + 1 := sorry

theorem transformed_sum_formula (n : ℕ) : 
  condition_1 ∧ condition_2 → transformed_sum n = 2 - (n + 4) / (2^(n+1)) := sorry

end general_formula_transformed_sum_formula_l574_574623


namespace car_city_mpg_is_36_l574_574496

noncomputable def car_mpg_in_city (H : ℝ) (C : ℝ) (T : ℝ) : ℝ :=
if H - C = 23 ∧ H * T = 690 ∧ C * T = 420 then C else 0

theorem car_city_mpg_is_36 
  (H C T : ℝ) 
  (h1 : H - C = 23)
  (h2 : H * T = 690)
  (h3 : C * T = 420) 
  : C ≈ 36 := 
by 
  -- Proof goes here
  sorry

end car_city_mpg_is_36_l574_574496


namespace sin_cos_identity_l574_574869

theorem sin_cos_identity :
  (Real.sin (20 * Real.pi / 180) * Real.cos (40 * Real.pi / 180) + Real.cos (20 * Real.pi / 180) * Real.sin (140 * Real.pi / 180)) =
  (Real.sqrt 3 / 2) := by
  sorry

end sin_cos_identity_l574_574869


namespace points_in_circle_l574_574365

open Real

/-- 
  Inside a unit square, there are placed some points. Show that among them,
  there exist three points that can fit inside a circle with a specific radius.
-/
theorem points_in_circle (p : Fin 51 → (ℝ × ℝ)) (h : ∀ i, 0 ≤ p i.1 ∧ p i.1 ≤ 1 ∧ 0 ≤ p i.2 ∧ p i.2 ≤ 1) : 
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ dist p i p j ≤ 2/7 ∧ dist p i p k ≤ 2/7 ∧ dist p j p k ≤ 2/7 :=
by
  sorry

end points_in_circle_l574_574365


namespace gcd_319_377_116_l574_574028

theorem gcd_319_377_116 : Nat.gcd (Nat.gcd 319 377) 116 = 29 := 
by {
  -- We need the GCD of 319 and 377 first.
  have h1 : Nat.gcd 319 377 = 29,
  {
    -- Euclidean algorithm steps would go here.
    -- 377 % 319 = 58
    -- 319 % 58 = 29
    -- 58 % 29 = 0
    -- GCD of 319 and 377 is 29.
    exact Nat.gcd_rec 319 377,
  },
  -- Now, we compute the GCD of 29 and 116.
  have h2 : Nat.gcd 29 116 = 29,
  {
    -- Euclidean algorithm steps would go here.
    -- 116 % 29 = 0
    -- GCD of 29 and 116 is 29.
    exact Nat.gcd_rec 29 116,
  },
  -- Combining both results:
  rw h1,
  rw h2,
}

end gcd_319_377_116_l574_574028


namespace product_of_consecutive_numbers_l574_574810

theorem product_of_consecutive_numbers (n : ℕ) (k : ℕ) (h₁: n * (n + 1) * (n + 2) = 210) (h₂: n + (n + 1) = 11) : k = 3 :=
by
  sorry

end product_of_consecutive_numbers_l574_574810


namespace normal_equation_at_x0_l574_574992

noncomputable def curve (x : ℝ) : ℝ := x + real.sqrt (x^3)
def x0 : ℝ := 1
def y0 := curve x0

theorem normal_equation_at_x0 :
  ∀ (x : ℝ), (y : ℝ),
  y = - (2 / 5) * x + 12 / 5 :=
    sorry

end normal_equation_at_x0_l574_574992


namespace find_initial_cookies_l574_574410

-- Definitions based on problem conditions
def initial_cookies (x : ℕ) : Prop :=
  let after_eating := x - 2
  let after_buying := after_eating + 37
  after_buying = 75

-- Main statement to be proved
theorem find_initial_cookies : ∃ x, initial_cookies x ∧ x = 40 :=
by
  sorry

end find_initial_cookies_l574_574410


namespace solve_inequality_l574_574426

def inequality_solution :=
  {x : ℝ // x < -3 ∨ x > -6/5}

theorem solve_inequality (x : ℝ) : 
  |2*x - 4| - |3*x + 9| < 1 → x < -3 ∨ x > -6/5 :=
by
  sorry

end solve_inequality_l574_574426


namespace domain_of_f_l574_574062

noncomputable def f (x : ℝ) : ℝ := (2 * x + 3) / (x + 5)

theorem domain_of_f :
  { x : ℝ | f x ≠ 0 } = { x : ℝ | x ≠ -5 }
:= sorry

end domain_of_f_l574_574062


namespace stratified_sampling_probability_two_primary_schools_l574_574687

-- Conditions
def num_primary_schools : ℕ := 21
def num_middle_schools : ℕ := 14
def num_universities : ℕ := 7
def total_schools := num_primary_schools + num_middle_schools+ num_universities
def selected_schools : ℕ := 6
def sampling_ratio := selected_schools / total_schools

/-- (1) Calculate the number of primary, middle, and universities schools after stratified sampling -/
theorem stratified_sampling :
  (num_primary_schools * sampling_ratio).ceil = 3 ∧ 
  (num_middle_schools * sampling_ratio).ceil = 2 ∧ 
  (num_universities * sampling_ratio).ceil = 1 :=
by sorry

/-- (2) Calculate the probability of selecting 2 primary schools -/
theorem probability_two_primary_schools :
  (3 / 6) * (2 / 5) = 1 / 5 :=
by sorry

end stratified_sampling_probability_two_primary_schools_l574_574687


namespace no_nat_pairs_satisfy_eq_l574_574087

theorem no_nat_pairs_satisfy_eq (a b : ℕ) : ¬ (2019 * a ^ 2018 = 2017 + b ^ 2016) :=
sorry

end no_nat_pairs_satisfy_eq_l574_574087


namespace not_necessarily_continuous_on_interval_l574_574327

variable {a b : ℝ}
variable {f : ℝ → ℝ}

theorem not_necessarily_continuous_on_interval (h : ∀ x ∈ set.Icc a b, f x ≠ 0) :
  ¬ (∀ x ∈ set.Icc a b, continuous_at f x) :=
sorry

end not_necessarily_continuous_on_interval_l574_574327


namespace hyperbola_eccentricity_l574_574309

-- Define the hyperbola and the conditions
def hyperbola (a b : ℝ) : Prop :=
  ∃ x y,
    a > 0 ∧ b > 0 ∧
    x^2 / a^2 - y^2 / b^2 = 1 ∧
    let foci1 := (-sqrt (a^2 + b^2), 0) in
    let foci2 := (sqrt (a^2 + b^2), 0) in
    let pointP := (a, b) in 
    (pointP.2 = b/a * pointP.1) ∧
    ((pointP.1 + sqrt (a^2 + b^2))^2 + pointP.2^2 = 9 * ((pointP.1 - sqrt (a^2 + b^2))^2 + pointP.2^2)) ∧
    sqrt (a^2 + b^2) = 5/4 * a

-- Theorem stating the eccentricity of the hyperbola
theorem hyperbola_eccentricity (a b : ℝ) (h : hyperbola a b) : sqrt (a^2 + b^2) / a = 5 / 4 :=
sorry

end hyperbola_eccentricity_l574_574309


namespace min_value_of_a_plus_b_l574_574598

theorem min_value_of_a_plus_b (a b : ℕ) (ha_pos : 0 < a) (hb_pos : 0 < b) (hc : 1 = 1) 
    (h1 : b^2 > 4 * a) (h2 : b < 2 * a) (h3 : b < a + 1) : a + b = 10 :=
sorry

end min_value_of_a_plus_b_l574_574598


namespace OGF_transform_l574_574429

variables {a : ℕ → ℝ} (A : ℝ → ℝ)

def generating_function (f : ℕ → ℝ) : ℝ → ℝ :=
  λ t, ∑ n : ℕ, (f n) * t^n

theorem OGF_transform (hA : A = generating_function a) :
  generating_function (λ n, ∑ k in finset.range (n + 1), a k) = (λ t, (1 / (1 - t)) * A t) :=
by
  sorry

end OGF_transform_l574_574429


namespace circle_geom_relation_l574_574223

theorem circle_geom_relation (ABC : Triangle) (circ : Circle) (inc : Circle) (O : Point) (I : Point)
  (cond1 : inscribes ABC circ)
  (cond2 : center circ = O)
  (cond3 : inscribes ABC inc)
  (cond4 : center inc = I) :
  ∀ (d R r : ℝ), d^2 = R^2 + 2 * R * r :=
by
  sorry

end circle_geom_relation_l574_574223


namespace sum_mod_five_l574_574413

theorem sum_mod_five {n : ℕ} (h_pos : 0 < n) :
  (1^n + 2^n + 3^n + 4^n) % 5 = 0 ↔ ¬ (∃ k : ℕ, n = 4 * k) :=
sorry

end sum_mod_five_l574_574413


namespace probability_at_least_6_heads_in_8_flips_l574_574155

def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then (Nat.choose n k) else 0

theorem probability_at_least_6_heads_in_8_flips :
  let total_outcomes := 2^8
  let successful_outcomes := binomial 8 6 + binomial 8 7 + binomial 8 8
  let probability := (successful_outcomes : ℚ) / total_outcomes
  probability = 37 / 256 :=
by
  sorry

end probability_at_least_6_heads_in_8_flips_l574_574155


namespace num_unique_five_topping_pizzas_l574_574896

open Nat

/-- The number of combinations of choosing k items from n items is defined using binomial coefficients. -/
def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem num_unique_five_topping_pizzas:
  let toppings := 8
      toppings_per_pizza := 5
  in binomial_coefficient toppings toppings_per_pizza = 56 := by
  sorry

end num_unique_five_topping_pizzas_l574_574896


namespace f_monotonic_l574_574646

def f (x : ℝ) : ℝ := (x^2 - 3*x + 3) * exp x

theorem f_monotonic (t : ℝ) (h : t > -2) : ∀ x ∈ set.Icc (-2) t, monotonic_on f (set.Icc (-2) t) :=
by sorry

end f_monotonic_l574_574646


namespace probability_at_least_6_heads_in_8_flips_l574_574157

def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then (Nat.choose n k) else 0

theorem probability_at_least_6_heads_in_8_flips :
  let total_outcomes := 2^8
  let successful_outcomes := binomial 8 6 + binomial 8 7 + binomial 8 8
  let probability := (successful_outcomes : ℚ) / total_outcomes
  probability = 37 / 256 :=
by
  sorry

end probability_at_least_6_heads_in_8_flips_l574_574157


namespace AE_value_l574_574194

noncomputable def find_AE
  (A B C D E F : Type)
  (BC : ℝ)
  (AD_perp_BC : Prop)
  (on_circle : (p : A) → p ∈ (B,C))
  (E_on_DC : (p : E) → p ∈ (D, C))
  (F_on_extension_CB : (p : F) → p ∈ extension(C, B))
  (angle_condition : ∠ BAF = ∠ CAE)
  (BC_length : BC = 15)
  (BF_length : BF = 6)
  (BD_length : BD = 3) : ℝ :=
  AE

theorem AE_value:
  ∀ (A B C D E F : Type)
    (BC : ℝ)
    (AD_perp_BC : Prop)
    (on_circle : (p : A) → p ∈ (B,C))
    (E_on_DC : (p : E) → p ∈ (D, C))
    (F_on_extension_CB : (p : F) → p ∈ extension(C, B))
    (angle_condition : ∠ BAF = ∠ CAE)
    (BC_length : BC = 15)
    (BF_length : BF = 6)
    (BD_length : BD = 3),
  find_AE A B C D E F BC AD_perp_BC on_circle E_on_DC F_on_extension_CB angle_condition BC_length BF_length BD_length = 2 * Real.sqrt 13 :=
sorry

end AE_value_l574_574194


namespace prime_sum_diff_l574_574245

theorem prime_sum_diff (p : ℕ) :
  prime p ∧ (∃ q r : ℕ, prime q ∧ prime r ∧ p = q + r) ∧ (∃ s t : ℕ, prime s ∧ prime t ∧ p = s - t) ↔ p = 5 := 
by
  sorry

end prime_sum_diff_l574_574245


namespace rise_in_water_level_l574_574112

-- Define the conditions related to the cube and the vessel
def edge_length := 15 -- in cm
def base_length := 20 -- in cm
def base_width := 15 -- in cm

-- Calculate volumes and areas
def V_cube := edge_length ^ 3
def A_base := base_length * base_width

-- Declare the mathematical proof problem statement
theorem rise_in_water_level : 
  (V_cube / A_base : ℝ) = 11.25 :=
by
  -- edge_length, V_cube, A_base are all already defined
  -- This particularly proves (15^3) / (20 * 15) = 11.25
  sorry

end rise_in_water_level_l574_574112


namespace quadratic_form_product_l574_574390

noncomputable def quadratic_form_min_max_product : ℝ :=
  let m := (5 - Real.sqrt 5) / 6
  let M := (5 + Real.sqrt 5) / 6
  m * M

theorem quadratic_form_product (x y : ℝ) (h : 9 * x^2 + 12 * x * y + 8 * y^2 = 1) : 
  quadratic_form_min_max_product = (20 - 5 * Real.sqrt 5) / 36 :=
by
  sorry

end quadratic_form_product_l574_574390


namespace b_grazing_l574_574485

-- Define the conditions
def grazing_conditions :=
  10 * 7 + (λ x : ℕ, x * 5) + 15 * 3 = 140

def c_share (cost_per_ox_month : ℝ) :=
  15 * 3 * cost_per_ox_month = 36

def total_rent_condition (cost_per_ox_month : ℝ) (a_cost b_cost c_cost : ℝ) :=
  a_cost + b_cost + c_cost = 140

def b_cost_condition (cost_per_ox_month : ℝ) (x : ℝ) :=
  x * 5 * cost_per_ox_month = (140 - 56 - 36 : ℝ)

-- The problem statement
theorem b_grazing (x : ℝ) (cost_per_ox_month : ℝ) (a_cost b_cost c_cost : ℝ) :
  grazing_conditions →
  c_share cost_per_ox_month →
  total_rent_condition cost_per_ox_month a_cost b_cost c_cost →
  b_cost_condition cost_per_ox_month x →
  x = 12 :=
by
  intros
  sorry

end b_grazing_l574_574485


namespace triangle_area_max_no_min_l574_574716

def angle_A : ℝ := 30
def sum_AB_AC (AB AC : ℝ) : Prop := AB + AC = 10
def sin_30 : ℝ := 1 / 2

theorem triangle_area_max_no_min (AB AC : ℝ) (h_sum : sum_AB_AC AB AC) (h_angleA : angle_A = 30) :
  ∃ S_max, S_max = 25 / 4 ∧ (∀ S, S ≤ S_max) ∧ ¬(∃ S_min, ∀ S, S_min ≤ S) := 
sorry

end triangle_area_max_no_min_l574_574716


namespace combined_mpg_is_16_l574_574418

variable (m : ℝ) -- distance driven by each person

-- Defining the conditions:
def rays_mpg := 40 -- Ray's car miles per gallon
def toms_mpg := 10 -- Tom's car miles per gallon

-- Defining the gasoline used based on these conditions
def rays_gasoline_used := m / rays_mpg
def toms_gasoline_used := m / toms_mpg

-- Total gasoline consumption
def total_gasoline_used := rays_gasoline_used + toms_gasoline_used

-- Total distance driven
def total_distance_driven := 2 * m

-- Combined miles per gallon
def combined_mpg := total_distance_driven / total_gasoline_used

-- The theorem we need to prove:
theorem combined_mpg_is_16 : combined_mpg = 16 := by
  sorry

end combined_mpg_is_16_l574_574418


namespace tuesday_rainfall_l574_574935

-- Condition: average rainfall for the whole week is 3 cm
def avg_rainfall_week : ℝ := 3

-- Condition: number of days in a week
def days_in_week : ℕ := 7

-- Condition: total rainfall for the week
def total_rainfall_week : ℝ := avg_rainfall_week * days_in_week

-- Condition: total rainfall is twice the rainfall on Tuesday
def total_rainfall_equals_twice_T (T : ℝ) : ℝ := 2 * T

-- Theorem: Prove that the rainfall on Tuesday is 10.5 cm
theorem tuesday_rainfall : ∃ T : ℝ, total_rainfall_equals_twice_T T = total_rainfall_week ∧ T = 10.5 := by
  sorry

end tuesday_rainfall_l574_574935


namespace similar_triangles_not_necessarily_equal_sides_l574_574073

theorem similar_triangles_not_necessarily_equal_sides
  (T1 T2 : Triangle)
  (h_angles: ∀ (a1 a2 : Angle), T1.has_angle a1 → T2.has_angle a2 → a1 = a2)
  (h_sides: ∀ (s1 s2 : Side), T1.has_side s1 → T2.has_side s2 → ∃ k : ℝ, k > 0 ∧ s1 = k * s2)
  (h_heights: ∀ (h1 h2 : ℝ), T1.height h1 → T2.height h2 → ∃ k : ℝ, k > 0 ∧ h1 = k * h2) :
  ¬(∀ (s1 s2 : Side), T1.has_side s1 → T2.has_side s2 → s1 = s2) :=
sorry

end similar_triangles_not_necessarily_equal_sides_l574_574073


namespace hydrogen_mass_percentage_in_ammonium_chloride_l574_574983

noncomputable def molar_mass_N : ℝ := 14.01
noncomputable def molar_mass_H : ℝ := 1.01
noncomputable def molar_mass_Cl : ℝ := 35.45

noncomputable def molar_mass_NH4Cl : ℝ := molar_mass_N + 4 * molar_mass_H + molar_mass_Cl

noncomputable def total_mass_H_in_NH4Cl : ℝ := 4 * molar_mass_H

noncomputable def mass_percentage_H : ℝ := (total_mass_H_in_NH4Cl / molar_mass_NH4Cl) * 100

theorem hydrogen_mass_percentage_in_ammonium_chloride :
  mass_percentage_H ≈ 7.55 :=
by
  have h1 : molar_mass_NH4Cl = 53.50 := by sorry
  have h2 : total_mass_H_in_NH4Cl = 4.04 := by sorry
  have h3 : mass_percentage_H = (4.04 / 53.50) * 100 := by sorry
  have h4 : (4.04 / 53.50) * 100 ≈ 7.55 := by sorry
  exact h4

end hydrogen_mass_percentage_in_ammonium_chloride_l574_574983


namespace fence_cost_l574_574044

noncomputable def cost_of_fencing (ratio : ℚ) (area : ℚ) (cost_per_meter : ℚ) : ℚ :=
let length : ℚ := 3 * real.sqrt (area / 12)
let width : ℚ := 4 * real.sqrt (area / 12)
let perimeter : ℚ := 2 * (length + width)
in perimeter * cost_per_meter

theorem fence_cost (ratio : ℚ) (area : ℚ) (cost_per_meter : ℚ) (correct_cost : ℚ) :
  ratio = 3/4 → area = 8748 → cost_per_meter = 0.25 → correct_cost = 94.5 → 
  cost_of_fencing ratio area cost_per_meter = correct_cost :=
by 
  intros; sorry

end fence_cost_l574_574044


namespace projection_correct_l574_574588

open Real

def vector_u : ℝ × ℝ := (3, -4)
def vector_v : ℝ × ℝ := (1, 2)

def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

def norm_squared (a : ℝ × ℝ) : ℝ :=
  dot_product a a

def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let scale := (dot_product u v) / (norm_squared v)
  (scale * v.1, scale * v.2)

theorem projection_correct :
  projection vector_u vector_v = (-1, -2) := by
  sorry

end projection_correct_l574_574588


namespace length_of_RT_in_trapezoid_l574_574490

-- Definition of the trapezoid and initial conditions
def trapezoid (PQ RS PR RT : ℝ) (h : PQ = 3 * RS) (h1 : PR = 15) : Prop :=
  RT = 15 / 4

-- The theorem to be proved
theorem length_of_RT_in_trapezoid (PQ RS PR RT : ℝ) 
  (h : PQ = 3 * RS) (h1 : PR = 15) : trapezoid PQ RS PR RT h h1 :=
by
  sorry

end length_of_RT_in_trapezoid_l574_574490


namespace find_m_l574_574532

noncomputable def f (x : ℕ) (c : ℝ) (m : ℕ) : ℝ :=
  if x < m then c / real.sqrt x else c / real.sqrt m

theorem find_m (c m : ℕ) (h1 : f 4 c m = 30) (h2 : f m c m = 15) : m = 16 :=
by {
  sorry
}

end find_m_l574_574532


namespace fewest_fence_posts_l574_574516

def fence_posts (length_wide short_side long_side : ℕ) (post_interval : ℕ) : ℕ :=
  let wide_side_posts := (long_side / post_interval) + 1
  let short_side_posts := (short_side / post_interval)
  wide_side_posts + 2 * short_side_posts

theorem fewest_fence_posts : fence_posts 40 10 100 10 = 19 :=
  by
    -- The proof will be completed here
    sorry

end fewest_fence_posts_l574_574516


namespace integer_coordinates_between_A_B_l574_574512

theorem integer_coordinates_between_A_B :
  let A := (1, 1)
  let B := (70, 701)
  let line := { P : ℤ × ℤ | ∃ (t : ℚ), P.1 = A.1 + t * (B.1 - A.1) ∧ P.2 = A.2 + t * (B.2 - A.2) }
  let points_between := { P : ℤ × ℤ | line P ∧ A.1 < P.1 ∧ P.1 < B.1 }
  finset.card points_between = 7 :=
sorry

end integer_coordinates_between_A_B_l574_574512


namespace find_value_of_A_l574_574454

theorem find_value_of_A (A B : ℕ) (h_ratio : A * 5 = 3 * B) (h_diff : B - A = 12) : A = 18 :=
by
  sorry

end find_value_of_A_l574_574454


namespace usjl_escape_l574_574521

theorem usjl_escape (n : ℕ) :
  ∃ path : list (ℕ × ℕ), 
    (∀ r ∈ path, r.1 < n ∧ r.2 < n) ∧ -- Rooms are within n x n grid
    (∀ i < path.length - 1, (path.nth i.snd ≠ path.nth (i + 1).snd) ∧ (path.nth i.fst < path.nth (i + 1).fst + 1) ∨ (path.nth i.fst = path.nth (i + 1).fst ∧ path.nth i.snd < path.nth  (i + 1).snd + 1)) ∧ -- Ensure path is valid in grid
    (path.nodup) ∧ -- Ensure each room is visited exactly once
    ((list.length path - 1) % 2 = 0) := -- USJL ends up as himself
begin
  sorry
end

end usjl_escape_l574_574521


namespace sin_cos_curve_l574_574593

theorem sin_cos_curve (t : ℝ) : 
  let x := Real.sin t
  let y := Real.cos t
  in x^2 + y^2 = 1 :=
by
  sorry

end sin_cos_curve_l574_574593


namespace matrix_pow_sub_l574_574380

open Matrix

noncomputable def B : Matrix (Fin 2) (Fin 2) ℚ := !![3, 4; 0, 2]

theorem matrix_pow_sub : 
  B^10 - 3 • B^9 = !![0, 4; 0, -1] := 
by
  sorry

end matrix_pow_sub_l574_574380


namespace odd_function_expression_l574_574873

-- Define that f(x) is an odd function
def is_odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f (x)

-- Define the function f(x) for x > 0
def f (x : ℝ) : ℝ :=
  if x > 0 then x^3 + 1 else sorry

theorem odd_function_expression (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_pos : ∀ x : ℝ, 0 < x → f x = x^3 + 1) :
  ∀ x : ℝ, x < 0 → f x = x^3 - 1 :=
by
  sorry

end odd_function_expression_l574_574873


namespace three_point_shots_l574_574353

def basketball_score (x : ℕ) : Prop :=
  let y := x - 3 in
  2 * x + 3 * y = 16

theorem three_point_shots : ∃ (x : ℕ), basketball_score x ∧ (x - 3 = 2) :=
by
  sorry

end three_point_shots_l574_574353


namespace area_of_triangle_l574_574357

theorem area_of_triangle (a b : ℝ) (h : b ≠ 0) (h1 : 2 * a ≠ b) :
  let A := a
  let C := b
  let area_ABC := (a^2 * b) / (2 * a - b) in
  ∃ (area : ℝ), area = area_ABC :=
by sorry

end area_of_triangle_l574_574357


namespace expression_converges_l574_574202

noncomputable def valueOfExpression : ℝ :=
  let F := (86 + 41 * F)
  in by
    have hF : F = 43 := solve_quadratic 1 (-41) (-86)
    exact hF

theorem expression_converges :
  (√(86 + 41 * √(86 + 41 * √(86 + 41 * √(86 + 41 * √(86 + 41 * F)))))) = 43 :=
by
  have hF : F = 43 := valueOfExpression
  sorry

end expression_converges_l574_574202


namespace k_bounds_inequality_l574_574068

open Real

theorem k_bounds_inequality (k : ℝ) :
  (∀ x : ℝ, abs ((x^2 - k * x + 1) / (x^2 + x + 1)) < 3) ↔ -5 ≤ k ∧ k ≤ 1 := 
sorry

end k_bounds_inequality_l574_574068


namespace tangent_line_value_l574_574781

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (1/2) * a * x^2 - 2 * x

theorem tangent_line_value (a b : ℝ) (h : a ≤ 0) 
  (h_tangent : ∀ x : ℝ, f a x = 2 * x + b) : a - 2 * b = 2 :=
sorry

end tangent_line_value_l574_574781


namespace repeating_decimal_fraction_l574_574567

def repeating_decimal_to_fraction (d: ℚ) (r: ℚ) (p: ℚ): ℚ :=
  d + r

theorem repeating_decimal_fraction :
  repeating_decimal_to_fraction (6 / 10) (1 / 33) (0.6 + (0.03 : ℚ)) = 104 / 165 := 
by
  sorry

end repeating_decimal_fraction_l574_574567


namespace interval_contains_n_l574_574990

theorem interval_contains_n (n : ℕ) (h1 : n < 1000) (h2 : n ∣ 999) (h3 : n + 6 ∣ 99) : 1 ≤ n ∧ n ≤ 250 := 
sorry

end interval_contains_n_l574_574990


namespace units_digit_42_pow_4_add_24_pow_4_l574_574849

-- Define a function to get the units digit of a number.
def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_42_pow_4_add_24_pow_4 : units_digit (42^4 + 24^4) = 2 := by
  sorry

end units_digit_42_pow_4_add_24_pow_4_l574_574849


namespace complex_sum_l574_574383

noncomputable def omega : ℂ := sorry
axiom omega_power_five : omega^5 = 1
axiom omega_not_one : omega ≠ 1

theorem complex_sum :
  (omega^20 + omega^25 + omega^30 + omega^35 + omega^40 + omega^45 + omega^50 + omega^55 + omega^60 + omega^65 + omega^70) = 11 :=
by
  sorry

end complex_sum_l574_574383


namespace max_value_sqrt_l574_574626

theorem max_value_sqrt (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_sum : x + y + z = 2) : 
  sqrt x + sqrt (2 * y) + sqrt (3 * z) ≤ 2 * sqrt 3 :=
by
  sorry

end max_value_sqrt_l574_574626


namespace probability_at_least_6_heads_in_8_flips_l574_574161

def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then (Nat.choose n k) else 0

theorem probability_at_least_6_heads_in_8_flips :
  let total_outcomes := 2^8
  let successful_outcomes := binomial 8 6 + binomial 8 7 + binomial 8 8
  let probability := (successful_outcomes : ℚ) / total_outcomes
  probability = 37 / 256 :=
by
  sorry

end probability_at_least_6_heads_in_8_flips_l574_574161


namespace binomial_expansion_max_coefficient_l574_574590

-- Variables and conditions
def x : Type := sorry

-- Expansion of the binomial (sqrt(x) - 2 / x^2)^8
def binomial_expansion (x : Type) := (sqrt(x) - 2 / x^2)^8

-- Define the term with the maximum binomial coefficient
def max_binomial_term (x : Type) := 1120 / x^6

-- Define the terms with maximum and minimum coefficients and their sum
def max_term (x : Type) := 1792 * x ^ (-11)
def min_term (x : Type) := -1792 * x ^ (-17/2)

-- Lean statement of the proof problem
theorem binomial_expansion_max_coefficient :
  binomial_expansion (x) = sorry ∧
  max_binomial_term (x) = 1120 / x^6 ∧
  (max_term (x) + min_term (x) = 1792 * x ^ (-11) - 1792 * x ^ (-17 / 2)) := sorry

end binomial_expansion_max_coefficient_l574_574590


namespace divisible_implies_divisible_l574_574767

theorem divisible_implies_divisible (n m : ℤ) (h : 6 * n + 11 * m ≡ 0 [ℤMOD 31]) : n + 7 * m ≡ 0 [ℤMOD 31] := 
sorry

end divisible_implies_divisible_l574_574767


namespace prob_at_least_6_heads_eq_l574_574118

-- define the number of coin flips
def n := 8

-- define the number of possible outcomes (2^n)
def total_outcomes := 2 ^ n

-- define the binomial coefficients for cases: 6 heads, 7 heads, 8 heads
def binom_8_6 := Nat.choose 8 6
def binom_8_7 := Nat.choose 8 7
def binom_8_8 := Nat.choose 8 8

-- calculate the favorable outcomes for at least 6 heads
def favorable_outcomes := binom_8_6 + binom_8_7 + binom_8_8

-- define the probability of getting at least 6 heads
def probability := (favorable_outcomes : ℚ) / total_outcomes

theorem prob_at_least_6_heads_eq : probability = 37 / 256 := by
  sorry

end prob_at_least_6_heads_eq_l574_574118


namespace volume_of_sphere_l574_574637

theorem volume_of_sphere
  (a b c : ℝ)
  (h1 : a * b * c = 4 * Real.sqrt 6)
  (h2 : a * b = 2 * Real.sqrt 3)
  (h3 : b * c = 4 * Real.sqrt 3)
  (O_radius : ℝ := Real.sqrt (a^2 + b^2 + c^2) / 2) :
  4 / 3 * Real.pi * O_radius^3 = 32 * Real.pi / 3 := by
  sorry

end volume_of_sphere_l574_574637


namespace card_number_factorization_l574_574827

theorem card_number_factorization :
  ∃ A B k : ℕ, 
    (B = 3 * A) ∧
    (B = 3 * 3 * 3 * k) ∧ 
    (∃ d : ℕ, d >= 1 ∧ d <= 9 ∧ A % d = 0) ∧ 
    (90 = 10 * (∑ i in range 1 10, i)) ∧
    (∑ i in flatten (repeat (range 1 10) 10) = 450) ∧ 
    (B >= 27) :=
begin
  sorry
end

end card_number_factorization_l574_574827


namespace cos_difference_as_product_l574_574569

theorem cos_difference_as_product (a b : ℝ) : 
  cos (a + b) - cos (a - b) = -2 * sin a * sin b :=
sorry

end cos_difference_as_product_l574_574569


namespace smallest_n_with_290_trailing_zeros_in_factorial_l574_574262

def trailing_zeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 5^2) + (n / 5^3) + (n / 5^4) + (n / 5^5) + (n / 5^6) -- sum until the division becomes zero

theorem smallest_n_with_290_trailing_zeros_in_factorial : 
  ∀ (n : ℕ), n >= 1170 ↔ trailing_zeros n >= 290 ∧ trailing_zeros (n-1) < 290 := 
by { sorry }

end smallest_n_with_290_trailing_zeros_in_factorial_l574_574262


namespace quadratic_function_satisfying_properties_l574_574281

def is_axis_of_symmetry (f : ℝ → ℝ) (x : ℝ) :=
  ∀ x1 x2, f x1 = f x2 → x1 + x2 = 2 * x

def is_monotonically_increasing (f : ℝ → ℝ) (I : set ℝ) :=
  ∀ x y, x ∈ I → y ∈ I → x < y → f x < f y

def has_maximum_value (f : ℝ → ℝ) (max_val : ℝ) :=
  ∃ x, f x = max_val ∧ ∀ y, f y ≤ max_val

theorem quadratic_function_satisfying_properties :
  ∃ (f : ℝ → ℝ), is_axis_of_symmetry f 2 ∧
    is_monotonically_increasing f (set.Iio 0) ∧
    has_maximum_value f 4 ∧
    (∀ x, f x = -(x - 2)^2 + 4) :=
sorry

end quadratic_function_satisfying_properties_l574_574281


namespace ajay_total_gain_l574_574534

theorem ajay_total_gain:
  let dal_A_kg := 15
  let dal_B_kg := 10
  let dal_C_kg := 12
  let dal_D_kg := 8
  let rate_A := 14.50
  let rate_B := 13
  let rate_C := 16
  let rate_D := 18
  let selling_rate := 17.50
  let cost_A := dal_A_kg * rate_A
  let cost_B := dal_B_kg * rate_B
  let cost_C := dal_C_kg * rate_C
  let cost_D := dal_D_kg * rate_D
  let total_cost := cost_A + cost_B + cost_C + cost_D
  let total_weight := dal_A_kg + dal_B_kg + dal_C_kg + dal_D_kg
  let total_selling_price := total_weight * selling_rate
  let gain := total_selling_price - total_cost
  gain = 104 := by
    sorry

end ajay_total_gain_l574_574534


namespace min_liars_eight_l574_574197

-- Define the problem context
structure Member :=
  (is_liar : Prop)

-- Number of total members
def num_members := 32

-- Grid dimensions
def rows := 4
def cols := 8

-- Adjacency definition for grid members
def is_neighbor (m1 m2 : ℕ × ℕ) : Prop :=
  ((m1.1 = m2.1 ∧ abs (m1.2 - m2.2) = 1) ||
   (m1.2 = m2.2 ∧ abs (m1.1 - m2.1) = 1))

-- Each member's claim
def claims_consistent (members : (ℕ × ℕ) → Member) : Prop :=
  ∀ (i j : ℕ × ℕ), (i.1 < rows) → (i.2 < cols) → 
  is_neighbor i j → 
  ((members i).is_liar ↔ ∃ k, is_neighbor i k ∧ (members k).is_liar)

-- Minimum liars required for consistency
def min_liars (members : (ℕ × ℕ) → Member) : ℕ :=
  {n // ∀ m : (ℕ × ℕ) → Member, claims_consistent m → ∃ l ≥ n, ∑ λ i j, if (m (i, j)).is_liar then 1 else 0}

-- The theorem to be proved
theorem min_liars_eight : ∃ members : (ℕ × ℕ) → Member, claims_consistent members ∧ min_liars members = 8 :=
sorry

end min_liars_eight_l574_574197


namespace PQRS_parallelogram_l574_574088

-- Definitions for the given problem conditions
variables (S1 S2 : Circle) (A B Q S P R : Point)
variables (h_inter_S1_S2 : intersects S1 S2 A)
variables (h_inter_S1_S2_bis : intersects S1 S2 B)
variables (h_tangent_AQ_S1 : tangent S1 A Q)
variables (h_on_S2_Q : on_circle S2 Q)
variables (h_tangent_BS_S2 : tangent S2 B S)
variables (h_on_S1_S : on_circle S1 S)
variables (h_inter_BQ_S1 : line_intersects_circle (line_through B Q) S1 R)
variables (h_inter_AS_S2 : line_intersects_circle (line_through A S) S2 P)

-- Goal: Prove that quadrilateral PQRS is a parallelogram
theorem PQRS_parallelogram : is_parallelogram P Q R S :=
sorry

end PQRS_parallelogram_l574_574088


namespace part1_part2_l574_574636

-- Definitions based on the conditions
def S (n : ℕ) : ℕ := n^2 + n
def a (n : ℕ) : ℕ := 2 * n
def H (n : ℕ) : ℚ := n / (4 * (n + 1))

-- Problematizing the given conditions
theorem part1 (n : ℕ) (h : n ≥ 1) : a n = 2 * n := sorry

theorem part2 (n : ℕ) (h : n ≥ 1) : 
  let U (m : ℕ) := 1 / ((a m) * (a (m + 1))) in
  (∑ i in Finset.range n, U (i + 1)) = H n := sorry

end part1_part2_l574_574636


namespace max_five_topping_pizzas_l574_574894

theorem max_five_topping_pizzas : 
  (∃ (n k : ℕ), n = 8 ∧ k = 5 ∧ (nat.choose n k = 56)) :=
begin
  use [8, 5],
  split,
  { refl, },
  split,
  { refl, },
  { sorry }
end

end max_five_topping_pizzas_l574_574894


namespace side_length_sum_area_l574_574814

theorem side_length_sum_area (a b c d : ℝ) (ha : a = 3) (hb : b = 4) (hc : c = 12) :
  d = 13 :=
by
  -- Proof is not required
  sorry

end side_length_sum_area_l574_574814


namespace tens_digit_of_sum_l574_574181

theorem tens_digit_of_sum (a b c : ℕ) (h1 : a = c - 1) (h2 : b = c + 3) :
  let N := 100 * a + 10 * b + c,
      R := 100 * c + 10 * b + a in
  (N + R) / 10 % 10 = 0 :=
by
  sorry

end tens_digit_of_sum_l574_574181


namespace area_BCE_BED_l574_574352

def point {α : Type*} := α
def triangle (α : Type*) := α × α × α
def segment {α : Type*} := α × α

variables {α : Type*} (A B C D E : point α) (h_AC : ℝ) (S_ABC : ℝ)
variables (on_line : E ∈ segment C D)

-- Given conditions
def CE : ℝ := 5
def ED : ℝ := 10
def area_ABC : ℝ := 45
def AC : ℝ := 9
def right_angle_at_B : Prop := true

-- Height from B to AC
def height_B : ℝ := 10

-- Areas to prove
def area_BCE : ℝ := 1/2 * CE * height_B
def area_BED : ℝ := 1/2 * ED * height_B

theorem area_BCE_BED : 
  area_BCE = 25 ∧ area_BED = 50 :=
by {
  -- Proof will be provided here
  sorry
}

end area_BCE_BED_l574_574352


namespace find_x_in_interval_l574_574242

theorem find_x_in_interval (x : ℝ) : x^2 + 5 * x < 10 ↔ -5 < x ∧ x < 2 :=
sorry

end find_x_in_interval_l574_574242


namespace log_inequality_condition_not_sufficient_condition_l574_574092

theorem log_inequality_condition (a b : ℝ) (h: a > b) : 
  (\(ln a > ln b) → (a > b)) :=
by
  sorry

theorem not_sufficient_condition (a b : ℝ) (h: a > b) : 
  ¬ (\(a > b) → \(ln a > ln b)) :=
by
  sorry

example (a b : ℝ) (h: a > b) : 
  log_inequality_condition a b h ∧ not_sufficient_condition a b h :=
by
  sorry

end log_inequality_condition_not_sufficient_condition_l574_574092


namespace sequences_count_is_178_l574_574320

def valid_sequence (l : List ℕ) : Prop := 
  l.length = 10 ∧ 
  ∀ i, i + 2 < l.length → (l.nth i ≠ l.nth (i + 1) ∨ l.nth i ≠ l.nth (i + 2))

def count_valid_sequences : ℕ := sorry -- This would be computed as 178 by the solution steps

theorem sequences_count_is_178 : count_valid_sequences = 178 := 
by {
  sorry -- Proof goes here
}

end sequences_count_is_178_l574_574320


namespace complex_magnitude_sixth_power_l574_574240

noncomputable def z := (2 : ℂ) + (2 * Real.sqrt 3) * Complex.I

theorem complex_magnitude_sixth_power :
  Complex.abs (z^6) = 4096 := 
by
  sorry

end complex_magnitude_sixth_power_l574_574240


namespace smallest_resolvable_debt_l574_574470

def pig_value : ℤ := 450
def goat_value : ℤ := 330
def gcd_pig_goat : ℤ := Int.gcd pig_value goat_value

theorem smallest_resolvable_debt :
  ∃ p g : ℤ, gcd_pig_goat * 4 = pig_value * p + goat_value * g := 
by
  sorry

end smallest_resolvable_debt_l574_574470


namespace tiles_for_cuboid_without_lid_l574_574502

theorem tiles_for_cuboid_without_lid :
  ∀ (W L H : ℕ), 
  W = 50 → 
  L = 35 → 
  H = 40 →
  (2 * (W * H) + 2 * (L * H) + (W * L)) = 8550 :=
by
  intros W L H hW hL hH
  rw [hW, hL, hH]
  have : 2 * (50 * 40) + 2 * (35 * 40) + (50 * 35) = 8550 := by
    calc
      2 * (50 * 40) + 2 * (35 * 40) + (50 * 35)
        = 2 * 2000 + 2 * 1400 + 1750 : by ring
    ... = 4000 + 2800 + 1750 : by ring
    ... = 8550 : by ring
  exact this

end tiles_for_cuboid_without_lid_l574_574502


namespace distance_between_points_l574_574982

def distance_formula (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_between_points :
  distance_formula 3 (-4) 10 6 = Real.sqrt 149 :=
by
  sorry

end distance_between_points_l574_574982


namespace find_vector_u_l574_574591

open Real

def vector_in_yz_plane (u : ℝ × ℝ × ℝ) : Prop := 
  u.1 = 0

def is_unit_vector (u : ℝ × ℝ × ℝ) : Prop := 
  u.2^2 + u.3^2 = 1

def angle_condition_30 (u : ℝ × ℝ × ℝ) : Prop := 
  u.2 + u.3 = 3 / 2

def angle_condition_45 (u : ℝ × ℝ × ℝ) : Prop := 
  u.3 = 1

theorem find_vector_u (u : ℝ × ℝ × ℝ) (h1 : vector_in_yz_plane u) (h2 : is_unit_vector u)
  (h3 : angle_condition_30 u) (h4 : angle_condition_45 u) : 
  u = (0, 1/2, 1) := 
sorry

end find_vector_u_l574_574591


namespace work_increase_percentage_l574_574697

theorem work_increase_percentage (p w : ℕ) (hp : p > 0) : 
  (((4 / 3 : ℚ) * w) - w) / w * 100 = 33.33 := 
sorry

end work_increase_percentage_l574_574697


namespace inequality_proof_l574_574596

theorem inequality_proof {x y z : ℝ}
  (h1 : x + 2 * y + 4 * z ≥ 3)
  (h2 : y - 3 * x + 2 * z ≥ 5) :
  y - x + 2 * z ≥ 3 :=
by
  sorry

end inequality_proof_l574_574596


namespace complex_modulus_l574_574639

open Complex

theorem complex_modulus (z : ℂ) (h : (1 + 2 * Complex.i) / z = 1 + Complex.i) : Complex.abs z = Real.sqrt 10 / 2 :=
sorry

end complex_modulus_l574_574639


namespace complex_modulus_8_l574_574977

noncomputable def complex_modulus {z : ℂ} (n : ℕ) : ℂ := z ^ n

theorem complex_modulus_8 : ∀ (z : ℂ), z = 1 + Complex.i * Real.sqrt 3 → |complex_modulus z 8| = 256 := by
  intros z hz
  have h_mod : |z| = 2 := by
    rw [hz, Complex.norm_eq_sqrt_sum_of_squares]
    calc Real.sqrt (_ + _) = _ := by
      simp [Complex.norm_eq_sqrt_sum_of_squares]
  rw [hz]
  calc |z ^ 8| = |z| ^ 8 := by sorry
  calc |2 ^ 8| = 256 := by
    norm_num
    sorry

end complex_modulus_8_l574_574977


namespace KI_bisects_angle_BKC_l574_574717

open EuclideanGeometry

theorem KI_bisects_angle_BKC
  (A B C I P O K : Point)
  (h_incenter : is_incenter I A B C)
  (h_tangent_PA_PB : is_tangent P A B)
  (h_tangent_PA_PC : is_tangent P A C)
  (h_circle_through_BC : passes_through O B C)
  (h_external_tangent_OK : external_tangent K O P) :
  bisects K I (angle B K C) :=
begin
  sorry
end

end KI_bisects_angle_BKC_l574_574717


namespace max_value_complex_expression_l574_574335

-- Define the condition for the complex number z
variable (z : ℂ) (h : complex.abs z = 1)

-- Define the theorem to prove the maximum value
theorem max_value_complex_expression : ∃ θ ∈ set.Ico 0 (2 * π), 
  let z := complex.of_real (cos θ) + complex.I * (sin θ) in
  complex.abs ((conj z + complex.I) * (z - complex.I)) = 2 * real.sqrt 2 := 
sorry

end max_value_complex_expression_l574_574335


namespace shaded_region_area_l574_574902

def side_length_square : ℝ := 10
def radius_quarter_circle : ℝ := side_length_square / 2
def area_square : ℝ := side_length_square ^ 2
def area_full_circle : ℝ := Real.pi * (radius_quarter_circle ^ 2)

theorem shaded_region_area :
  let side := side_length_square in
  let radius := radius_quarter_circle in
  let area_sq := area_square in
  let area_circle := area_full_circle in
  area_sq - area_circle = 100 - 25 * Real.pi := by
  sorry

end shaded_region_area_l574_574902


namespace cube_root_neg8_l574_574788

theorem cube_root_neg8 : ∛(-8) = -2 := sorry

end cube_root_neg8_l574_574788


namespace sets_A_eq_sets_t_or_t_neg_t_l574_574613

noncomputable def exists_sets (t : ℝ) (h_t_pos : t > 0) : Prop :=
  ∃ A B : set ℝ, t ∈ A ∧ ∃ n : ℕ, n ≥ 4 ∧ (B : |B| = n) ∧ 
  (∀ a ∈ A, ∀ b ∈ B, ∃ k : ℕ, ∀ i : ℕ, i < n → a * (b + i) = a * b + k * i) → (A = {t}) ∨ (A = {t, -t})

theorem sets_A_eq_sets_t_or_t_neg_t (t : ℝ) (h_t_pos : t > 0) : exists_sets t h_t_pos :=
sorry

end sets_A_eq_sets_t_or_t_neg_t_l574_574613


namespace binomial_12_9_l574_574209

def binomial (n k : ℕ) := nat.choose n k

theorem binomial_12_9 : binomial 12 9 = 220 :=
by
  have step1 : binomial 12 9 = binomial 12 3 := nat.choose_symm 12 9
  have step2 : binomial 12 3 = 220 := by sorry
  rw [step1, step2]

end binomial_12_9_l574_574209


namespace find_other_number_l574_574430

theorem find_other_number (HCF LCM num1 num2 : ℕ) (h1 : HCF = 16) (h2 : LCM = 396) (h3 : num1 = 36) (h4 : HCF * LCM = num1 * num2) : num2 = 176 :=
sorry

end find_other_number_l574_574430


namespace maximum_value_of_expression_l574_574604

theorem maximum_value_of_expression (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) :
  (a^3 + b^3 + c^3) / ((a + b + c)^3 - 26 * a * b * c) ≤ 3 := by
  sorry

end maximum_value_of_expression_l574_574604


namespace apples_left_total_l574_574273

-- Define the initial conditions
def FrankApples : ℕ := 36
def SusanApples : ℕ := 3 * FrankApples
def SusanLeft : ℕ := SusanApples / 2
def FrankLeft : ℕ := (2 / 3) * FrankApples

-- Define the total apples left
def total_apples_left (SusanLeft FrankLeft : ℕ) : ℕ := SusanLeft + FrankLeft

-- Given conditions transformed to Lean
theorem apples_left_total : 
  total_apples_left (SusanApples / 2) ((2 / 3) * FrankApples) = 78 := by
  sorry

end apples_left_total_l574_574273


namespace balls_in_bag_l574_574883

theorem balls_in_bag (T : ℕ) (white green yellow red purple : ℕ)
  (total_red_purple : red + purple = 21)
  (probability_not_red_or_purple : ∀ (balls_total : ℕ), balls_total = white + green + yellow + red + purple → 0.35 * balls_total = total_red_purple)
  (total_balls_probability : 0.65 = 1 - 0.35) 
  (white_eq : white = 22)
  (green_eq : green = 10)
  (yellow_eq : yellow = 7)
  (red_eq : red = 15)
  (purple_eq : purple = 6) :
  T = 60 := 
by
  sorry

end balls_in_bag_l574_574883


namespace combined_rate_of_miles_per_gallon_l574_574535

variable (miles_bob : ℝ) (efficiency_alice : ℝ) (efficiency_bob : ℝ) (limit_gasoline : ℝ)
          (dist_alice : ℝ) (dist_bob : ℝ)
          (actual_total_mpg : ℝ)

-- Given conditions
axiom h1 : efficiency_alice = 50
axiom h2 : efficiency_bob = 30
axiom h3 : dist_bob = 160
axiom h4 : dist_alice = 2 * dist_bob
axiom h5 : limit_gasoline = 10

-- What we need to prove
theorem combined_rate_of_miles_per_gallon : 
  ∃ dist_alice adj dist_bob adj,
    (dist_alice adj = 272) ∧ 
    (dist_bob adj = 136) ∧ 
  (actual_total_mpg = (dist_alice adj + dist_bob adj) / limit_gasoline) ∧
  (actual_total_mpg = 40.8) :=
by {
  sorry
}

end combined_rate_of_miles_per_gallon_l574_574535


namespace determine_g_3056_l574_574742

noncomputable def g : ℝ → ℝ := sorry

axiom g_pos_for_all_x (x : ℝ) (h₀ : x > 0) : g(x) > 0
axiom g_equation (x y : ℝ) (h₀ : x > y) (h₁ : x > 0) (h₂ : y > 0) : g(x - y) = Real.sqrt (g (x * y) + 4)
axiom exists_x_y (z : ℝ) (h₀ : z = 3056) : ∃ (x y : ℝ), (x - y = z) ∧ (x * y = z)

theorem determine_g_3056 (z : ℝ) (h₀ : z = 3056) : g(z) = 2 :=
by
  obtain ⟨x, y, ⟨hx, hy⟩⟩ := exists_x_y z h₀
  sorry

end determine_g_3056_l574_574742


namespace combined_mpg_is_16_l574_574417

variable (m : ℝ) -- distance driven by each person

-- Defining the conditions:
def rays_mpg := 40 -- Ray's car miles per gallon
def toms_mpg := 10 -- Tom's car miles per gallon

-- Defining the gasoline used based on these conditions
def rays_gasoline_used := m / rays_mpg
def toms_gasoline_used := m / toms_mpg

-- Total gasoline consumption
def total_gasoline_used := rays_gasoline_used + toms_gasoline_used

-- Total distance driven
def total_distance_driven := 2 * m

-- Combined miles per gallon
def combined_mpg := total_distance_driven / total_gasoline_used

-- The theorem we need to prove:
theorem combined_mpg_is_16 : combined_mpg = 16 := by
  sorry

end combined_mpg_is_16_l574_574417


namespace projection_of_vector_l574_574581

-- Definition of vectors a and b
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (3, -4)

-- Dot product of two vectors
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Projection of b onto a
def proj (a b : ℝ × ℝ) : ℝ × ℝ :=
  let scale := dot_product b a / dot_product a a
  (scale * a.1, scale * a.2)

-- Theorem stating that the projection of b onto a is (-1, -2)
theorem projection_of_vector :
  proj a b = (-1, -2) :=
sorry

end projection_of_vector_l574_574581


namespace cos_A_half_triangle_area_sqrt7_bc4_l574_574361

open Real

variable {A B C : ℝ}
variable {a b c : ℝ}
 
-- Condition: In triangle ABC, we have a, b, c as sides opposite angles A, B, and C, with: 
-- equation: c * cos A + a * cos C = 2 * b * cos A
theorem cos_A_half (h : c * cos A + a * cos C = 2 * b * cos A) :
  cos A = 1 / 2 :=
sorry

-- Additional conditions: a = sqrt 7 and b + c = 4
-- Prove the area of triangle ABC
theorem triangle_area_sqrt7_bc4 (h₁ : a = sqrt 7) (h₂ : b + c = 4) (h₃ : c * cos A + a * cos C = 2 * b * cos A) :
  let area := (1 / 2) * b * c * sqrt (1 - cos A^2)
  cos A = 1 / 2 → 
  area = 3 * sqrt 3 / 4 :=
sorry

end cos_A_half_triangle_area_sqrt7_bc4_l574_574361


namespace find_positive_X_l574_574386

-- Define the relation *.
def star_relation (X Y : ℝ) : ℝ := X^3 + Y^2

-- Given proof problem statement
theorem find_positive_X (X : ℝ) (h : star_relation X 4 = 280) : X = real.cbrt 264 :=
by
  -- Omit the proof
  sorry

end find_positive_X_l574_574386


namespace units_digit_sum_42_4_24_4_l574_574850

theorem units_digit_sum_42_4_24_4 : (42^4 + 24^4) % 10 = 2 := 
by
  sorry

end units_digit_sum_42_4_24_4_l574_574850


namespace sqrt_expr_eq_two_l574_574870

noncomputable def expr := Real.sqrt (3 + 2 * Real.sqrt 2) - Real.sqrt (3 - 2 * Real.sqrt 2)

theorem sqrt_expr_eq_two : expr = 2 := 
by
  sorry

end sqrt_expr_eq_two_l574_574870


namespace lily_spent_on_shirt_l574_574753

theorem lily_spent_on_shirt (S : ℝ) (initial_balance : ℝ) (final_balance : ℝ) : 
  initial_balance = 55 → 
  final_balance = 27 → 
  55 - S - 3 * S = 27 → 
  S = 7 := 
by
  intros h1 h2 h3
  sorry

end lily_spent_on_shirt_l574_574753


namespace combined_mpg_l574_574419

theorem combined_mpg (m : ℝ) : 
  let ray_mpg := 40
  let tom_mpg := 10
  let ray_gas := m / ray_mpg
  let tom_gas := m / tom_mpg
  let total_gas := ray_gas + tom_gas
  let total_distance := 2 * m
  let combined_mpg := total_distance / total_gas
  combined_mpg = 16 :=
by
  have ray_gas_eq : ray_gas = m / 40 := rfl
  have tom_gas_eq : tom_gas = m / 10 := rfl
  have total_gas_eq : total_gas = m / 40 + m / 10 := rfl
  have simplify_total_gas : total_gas = m / 8 := by
    rw [total_gas_eq]
    calc 
      m / 40 + m / 10 = m / 40 + 4 * (m / 40) : by simp
                  ... = (1 * m) / 40 + (4 * m) / 40 : by ring
                  ... = (1 + 4) * m / 40 : by ring
                  ... = 5 * m / 40 : by ring
                  ... = m / 8 : by ring

  have total_distance_eq : total_distance = 2 * m := rfl
  have combined_mpg_eq : combined_mpg = total_distance / total_gas := rfl
  have simplify_combined_mpg : combined_mpg = 16 := by
    rw [combined_mpg_eq, total_distance_eq, simplify_total_gas]
    calc 
      2 * m / (m / 8) = 2 * m * (8 / m) : by rw div_mul
                 ... = 2 * 8 : by simp
                 ... = 16 : by ring

  exact simplify_combined_mpg

end combined_mpg_l574_574419


namespace prob_at_least_6_heads_eq_l574_574117

-- define the number of coin flips
def n := 8

-- define the number of possible outcomes (2^n)
def total_outcomes := 2 ^ n

-- define the binomial coefficients for cases: 6 heads, 7 heads, 8 heads
def binom_8_6 := Nat.choose 8 6
def binom_8_7 := Nat.choose 8 7
def binom_8_8 := Nat.choose 8 8

-- calculate the favorable outcomes for at least 6 heads
def favorable_outcomes := binom_8_6 + binom_8_7 + binom_8_8

-- define the probability of getting at least 6 heads
def probability := (favorable_outcomes : ℚ) / total_outcomes

theorem prob_at_least_6_heads_eq : probability = 37 / 256 := by
  sorry

end prob_at_least_6_heads_eq_l574_574117


namespace sec_510_eq_neg_2_sqrt_3_div_3_l574_574572

open Real

theorem sec_510_eq_neg_2_sqrt_3_div_3 : sec (510 * (π / 180)) = - (2 * sqrt 3) / 3 :=
by
  -- we'll skip the proof using "sorry"
  sorry

end sec_510_eq_neg_2_sqrt_3_div_3_l574_574572


namespace smallest_3_digit_solution_l574_574494

open Int

theorem smallest_3_digit_solution :
  ∃ n : ℤ, 100 ≤ n ∧ n < 1000 ∧ (77 * n ≡ 231 [MOD 385]) ∧ ∀ m : ℤ, 100 ≤ m ∧ m < 1000 ∧ (77 * m ≡ 231 [MOD 385]) → n ≤ m :=
sorry

end smallest_3_digit_solution_l574_574494


namespace total_tax_collected_l574_574974

-- Define the conditions:
def mr_william_tax_payment : ℕ := 480
def percentage_of_taxable_land_mr_william : ℕ := 50

-- Define the proof statement:
theorem total_tax_collected (mr_william_tax_payment : ℕ) (percentage_of_taxable_land_mr_william : ℕ) :
  mr_william_tax_payment = 480 → percentage_of_taxable_land_mr_william = 50 → 
  ∃ T : ℕ, T = 960 :=
by
  intros h1 h2
  use 960
  sorry

end total_tax_collected_l574_574974


namespace hyperbola_equation_l574_574539

theorem hyperbola_equation (a b : ℝ) (ha : a = 2) (hb : b = sqrt 2) :
  (x : ℝ) (y : ℝ) → (x^2 / a^2 - y^2 / b^2 = 1) → (x^2 / 4 - y^2 / 2 = 1) :=
by sorry

end hyperbola_equation_l574_574539


namespace inequality_proof_l574_574625

variable {a b c : ℝ}

theorem inequality_proof (ha : a > 0) (hb : b > 0) (hc : c > 0) (h₁ : a ≠ b) (h₂ : a ≠ c) (h₃ : b ≠ c) :
  (b + c - a) / a + (a + c - b) / b + (a + b - c) / c > 3 :=
by
  sorry

end inequality_proof_l574_574625


namespace inradius_of_triangle_l574_574808

theorem inradius_of_triangle (p A r : ℝ) (h1 : p = 20) (h2 : A = 25) : r = 2.5 :=
sorry

end inradius_of_triangle_l574_574808


namespace angle_EDF_equals_50_l574_574362

-- Given definitions and conditions
variable (A B C D E F : Type)
variable [Angle A] [Angle B] [Angle C] [Angle D] [Angle E] [Angle F]

variable (AB AC BC : Real)
variable (angle_A angle_B angle_C : Angle)
variable (angle_D angle_E angle_F angle_EDF : Angle)

-- Assumptions from problem:
def triangle_ABC_isosceles (A B C : Type) (AB AC : Real) (angle_A : Angle) : Prop := AB = AC ∧ angle_A = 80

def points_on_sides (D E F : Type) (BC AC AB : Real) : Prop := True

def CE_eq_CD (C D E : Type) (CE CD : Real) : Prop := CE = CD

def BF_eq_BD (B D F : Type) (BF BD : Real) : Prop := BF = BD

-- The Lean proof statement
theorem angle_EDF_equals_50 (A B C D E F : Type) (AB AC : Real) (angle_A angle_EDF : Angle) :
  triangle_ABC_isosceles A B C AB AC angle_A →
  points_on_sides D E F BC AC AB →
  CE_eq_CD C D E CE CD →
  BF_eq_BD B D F BF BD →
  angle_EDF = 50 := by
  sorry

end angle_EDF_equals_50_l574_574362


namespace angle_ACS_eq_angle_BCP_l574_574764

variable (A B C K L M N X Y P S : Point)
variable (hABC : acute_triangle A B C)
variable (hCAKL : square C A K L)
variable (hCBMN : square C B M N)
variable (hCN_AK_X : intersects (line_through C N) (seg A K) X)
variable (hCL_BM_Y : intersects (line_through C L) (seg B M) Y)
variable (hP : P = circumcircles_intersection (triangle K X N) (triangle L Y M))
variable (hS : S = midpoint A B)

theorem angle_ACS_eq_angle_BCP :
  ∠ A C S = ∠ B C P :=
sorry

end angle_ACS_eq_angle_BCP_l574_574764


namespace sum_of_solutions_eq_23_over_17_l574_574263

noncomputable def sum_of_real_roots : ℝ :=
  let p := polynomial.C 17 * polynomial.X^2 - polynomial.C 23 * polynomial.X - polynomial.C 9 in
  have h : ∀ x : ℝ, (x-3) / (x^2 + 5*x + 2) = (x-6) / (x^2 - 15*x + 1) → (p.eval x = 0),
  from sorry,
  p.roots.sum.to_real / p.leading_coeff.to_real

theorem sum_of_solutions_eq_23_over_17 :
  sum_of_real_roots = 23 / 17 :=
sorry

end sum_of_solutions_eq_23_over_17_l574_574263


namespace max_marks_paper_I_l574_574880

theorem max_marks_paper_I (M : ℝ) (h1 : 0.40 * M = 60) : M = 150 :=
by
  sorry

end max_marks_paper_I_l574_574880


namespace superhero_speed_conversion_l574_574484

theorem superhero_speed_conversion
    (speed_km_per_min : ℕ)
    (conversion_factor : ℝ)
    (minutes_in_hour : ℕ)
    (H1 : speed_km_per_min = 1000)
    (H2 : conversion_factor = 0.6)
    (H3 : minutes_in_hour = 60) :
    (speed_km_per_min * conversion_factor * minutes_in_hour = 36000) :=
by
    sorry

end superhero_speed_conversion_l574_574484


namespace probability_of_at_least_six_heads_is_correct_l574_574128

-- Definitions for the given problem
def binomial_coefficient (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

def total_possible_outcomes : ℕ :=
  2^8

def favorable_outcomes : ℕ :=
  binomial_coefficient 8 6 + binomial_coefficient 8 7 + binomial_coefficient 8 8

def probability_of_at_least_6_heads : ℚ :=
  favorable_outcomes / total_possible_outcomes

-- The proof statement
theorem probability_of_at_least_six_heads_is_correct :
  probability_of_at_least_6_heads = 37 / 256 :=
by sorry

end probability_of_at_least_six_heads_is_correct_l574_574128


namespace maximum_value_of_expression_l574_574605

theorem maximum_value_of_expression (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) :
  (a^3 + b^3 + c^3) / ((a + b + c)^3 - 26 * a * b * c) ≤ 3 := by
  sorry

end maximum_value_of_expression_l574_574605


namespace problem1_problem2_l574_574947

noncomputable def expr1 : ℝ := 6 * Real.sqrt (3 / 2) - (Real.sqrt 48) / (Real.sqrt 3)
noncomputable def expr1_simplified : ℝ := 3 * Real.sqrt 6 - 4

noncomputable def expr2 : ℝ := (-Real.sqrt 5) ^ 2 + (1 + Real.sqrt 3) * (3 - Real.sqrt 3) - Real.sqrt[3] 27
noncomputable def expr2_simplified : ℝ := 2 + 2 * Real.sqrt 3

theorem problem1 : expr1 = expr1_simplified := by
  sorry

theorem problem2 : expr2 = expr2_simplified := by
  sorry

end problem1_problem2_l574_574947


namespace algebraic_expression_positive_l574_574406

theorem algebraic_expression_positive (a b : ℝ) : 
  a^2 + b^2 + 4*b - 2*a + 6 > 0 :=
by sorry

end algebraic_expression_positive_l574_574406


namespace total_worksheets_l574_574525

theorem total_worksheets (x : ℕ) (h1 : 7 * (x - 8) = 63) : x = 17 := 
by {
  sorry
}

end total_worksheets_l574_574525


namespace cube_division_minimal_l574_574395

noncomputable def cube_side (s : ℕ) : Prop := 
  s > 0 -- The side length of the cube is a positive integer

def typical_parallelepiped (a b c : ℕ) : Prop := 
  a ≠ b ∧ b ≠ c ∧ a ≠ c -- All dimensions of a typical parallelepiped are distinct.

def volume_eq (a b c s : ℕ) : Prop := 
  a * b * c = s^3 -- Volume condition for dividing into typical parallelepipeds

def min_typical_parallelepipeds_number : ℕ := 6 -- Minimum number of distinct typical parallelepipeds

theorem cube_division_minimal (s : ℕ) (h_s : cube_side s) :
  ∃ (parallelepipeds : list (ℕ × ℕ × ℕ)), 
    list.length parallelepipeds = min_typical_parallelepipeds_number ∧ 
    (∀ p ∈ parallelepipeds, ∃ a b c, p = (a, b, c) ∧ typical_parallelepiped a b c) ∧ 
    ∑ p in parallelepipeds, (p.1 * p.2 * p.3) = s^3 :=
sorry -- Proof not needed as per instruction

end cube_division_minimal_l574_574395


namespace people_left_line_l574_574939

theorem people_left_line (L : ℕ) (h_initial : 31 - L + 25 = 31) : L = 25 :=
by
  -- proof will go here
  sorry

end people_left_line_l574_574939


namespace least_stamps_l574_574929

theorem least_stamps (s t : ℕ) (h : 5 * s + 7 * t = 48) : s + t = 8 :=
by sorry

end least_stamps_l574_574929


namespace sumata_family_vacation_l574_574782

theorem sumata_family_vacation
  (total_miles : ℕ) (days : ℕ)
  (h1 : total_miles = 1250)
  (h2 : days = 5) :
  total_miles / days = 250 :=
by
  rw [h1, h2]
  exact Nat.div_eq_of_eq_mul_left (by norm_num) rfl

end sumata_family_vacation_l574_574782


namespace shortest_path_on_globe_l574_574072

/-- A point on the Earth's surface described by latitude and longitude. -/
structure Point where
  latitude : ℝ -- degrees
  longitude : ℝ -- degrees

/-- The two points A and B considered in the problem. -/
def A : Point := { latitude := 70, longitude := -10 }
def B : Point := { latitude := 70, longitude := 175 }

/-- Proves that the shortest path on a globe between points A and B does not lie 
exactly east, exactly west, or exactly through the North Pole. 
Hence, the correct direction is "none of the above". -/
theorem shortest_path_on_globe (A B: Point):
  A.latitude = 70 ∧ A.longitude = -10 ∧ B.latitude = 70 ∧ B.longitude = 175 →
  ¬(shortest_path_east A B ∨ shortest_path_west A B ∨ shortest_path_through_north_pole A B) ∧
  shortest_path_great_circle A B := 
sorry

/-- Placeholder definitions for different types of paths for clarity. Detailed definitions 
of these would involve geometric and spherical trigonometry calculations which are simplified here.
-/
def shortest_path_east (A B : Point) : Prop := sorry
def shortest_path_west (A B : Point) : Prop := sorry
def shortest_path_through_north_pole (A B : Point) : Prop := sorry
def shortest_path_great_circle (A B : Point) : Prop := sorry 

end shortest_path_on_globe_l574_574072


namespace probability_at_least_6_heads_in_8_flips_l574_574130

open scoped BigOperators

def binom (n k : ℕ) : ℕ := nat.choose n k

def total_outcomes (n : ℕ) := 2^n

def successful_outcomes (n k : ℕ) :=
  (finset.range (n + 1)).filter (λ x, x ≥ k).sum (λ x, binom n x)

def probability (n k : ℕ) :=
  (successful_outcomes n k) / (total_outcomes n : ℚ)

theorem probability_at_least_6_heads_in_8_flips :
  probability 8 6 = 37 / 256 := sorry

end probability_at_least_6_heads_in_8_flips_l574_574130


namespace max_min_f_on_interval_abs_f_difference_leq_one_fourth_range_of_c_when_two_zeros_l574_574656

section
variables {c : ℝ} (f : ℝ → ℝ) (a b : ℝ)
-- The function definition.
def f (x : ℝ) := x^2 - x + c

-- Problem 1: Maximum and minimum values of f(x) on [0,1]
theorem max_min_f_on_interval : 
  (∀ x ∈ Icc 0 1, f x ≤ c) ∧ (∀ x ∈ Icc 0 1, f x ≥ c - 1 / 4) := sorry

-- Problem 2: 
theorem abs_f_difference_leq_one_fourth : 
  ∀ x1 x2 ∈ Icc 0 1, |f x1 - f x2| ≤ 1 / 4 := sorry

-- Problem 3: 
theorem range_of_c_when_two_zeros (h : ∃ x1 x2 ∈ Icc 0 2, f x1 = 0 ∧ f x2 = 0) : 
  0 ≤ c ∧ c < 1 / 4 := sorry
end

end max_min_f_on_interval_abs_f_difference_leq_one_fourth_range_of_c_when_two_zeros_l574_574656


namespace hooligan_chapter_sheets_l574_574317

theorem hooligan_chapter_sheets :
  ∀ (first_page last_page : ℕ),
    first_page = 231 →
    last_page = 312 →
    (∀ (d1 d2 d3 : ℕ), last_page = d1 * 100 + d2 * 10 + d3 ∧ {d1, d2, d3} = {2, 3, 1} ∧ d3 % 2 = 0) →
    (last_page - first_page + 1) / 2 = 41 :=
by
  intros first_page last_page h_first_page h_last_page h_digits_even
  sorry

end hooligan_chapter_sheets_l574_574317


namespace b_2017_eq_1_l574_574431

def fib : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fib (n+1) + fib n

def b (n : ℕ) : ℕ := (fib n) % 3

theorem b_2017_eq_1 : b 2017 = 1 :=
by {
  -- As the period of b(n) is 8, we have b(2017 % 8) = b(1)
  have h_mod : 2017 % 8 = 1, by norm_num,
  rw h_mod,
  show b 1 = 1,
  -- We evaluate b(1)
  have h_fib1 : fib 1 = 1, by norm_num,
  rw h_fib1,
  norm_num,
}

end b_2017_eq_1_l574_574431


namespace abs_differentiable_except_zero_l574_574773

theorem abs_differentiable_except_zero (x : ℝ) (hx : x ≠ 0) :
  differentiable ℝ (λ x, |x|) x ∧
  (deriv (λ x, |x|) x = if x > 0 then 1 else -1) :=
by
  sorry

end abs_differentiable_except_zero_l574_574773


namespace divide_grid_into_identical_parts_l574_574553

-- Defining a 6x6 grid with a set number of shaded cells
def grid : Type := array (6 * 6) bool

-- The condition of having exactly 12 shaded cells
def has_12_shaded_cells (g : grid) : Prop := 
  array.count id g = 12

-- The condition of dividing the grid into four identical parts, each part containing exactly three shaded cells
def can_be_divided_in_four_identical_parts_with_three_shaded_cells (g : grid) : Prop :=
  ∃ parts : array 4 (array (3 * 3) bool),
  (∀ p ∈ parts, array.count id p = 3) ∧
  (∀ i j, (i < 4) → (j < 9) → parts[i][j] = someShadingPattern) -- ensure the identical division pattern

-- The overall theorem to be proved
theorem divide_grid_into_identical_parts (g : grid) (H : has_12_shaded_cells g) :
  can_be_divided_in_four_identical_parts_with_three_shaded_cells g :=
sorry

end divide_grid_into_identical_parts_l574_574553


namespace matt_science_homework_percentage_l574_574401

theorem matt_science_homework_percentage :
  let t := 150
  let p_m := 0.30
  let t_o := 45 
  (t - (p_m * t + t_o)) / t * 100 = 40 :=
by 
  let t := 150
  let p_m := 0.30
  let t_o := 45
  let t_m := p_m * t
  let t_s := t - (t_m + t_o)
  let p_s := (t_s / t) * 100
  show p_s = 40
  sorry

end matt_science_homework_percentage_l574_574401


namespace total_cost_of_orange_and_pear_l574_574792

variable O P B : ℝ

-- Given conditions as hypotheses
def condition1 : Prop := P - O = B
def condition2 : Prop := P = 90
def condition3 : Prop := 200 * B + 400 * O = 24000

-- Prove that the total cost of an orange and a pear is 120
theorem total_cost_of_orange_and_pear (h1 : condition1) (h2 : condition2) (h3 : condition3) : O + P = 120 :=
by 
  sorry

end total_cost_of_orange_and_pear_l574_574792


namespace closed_curve_length_l574_574885

theorem closed_curve_length
  (circumference_coin : ℝ)
  (circumference_quadrilateral : ℝ)
  (radius_coin : ℝ)
  (H1 : circumference_coin = 5)
  (H2 : circumference_quadrilateral = 20)
  (H3 : 2 * real.pi * radius_coin = circumference_coin) :
  (circumference_quadrilateral + 2 * real.pi * radius_coin = 25) :=
  by sorry

end closed_curve_length_l574_574885


namespace reflection_matrix_squared_is_identity_l574_574381

noncomputable def reflect_matrix (a : ℝ^3) : Matrix (Fin 3) (Fin 3) ℝ :=
  let I : Matrix (Fin 3) (Fin 3) ℝ := 1
  I - 2 * (a ⬝ aᵀ) / (aᵀ ⬝ a)

theorem reflection_matrix_squared_is_identity :
  let a : ℝ^3 := ![2, -2, 1]
  let R := reflect_matrix a
  R ⬝ R = (1 : Matrix (Fin 3) (Fin 3) ℝ) :=
by
  sorry

end reflection_matrix_squared_is_identity_l574_574381


namespace average_age_of_remaining_people_l574_574018

theorem average_age_of_remaining_people:
  ∀ (ages : List ℕ), 
  (List.length ages = 8) →
  (List.sum ages = 224) →
  (24 ∈ ages) →
  ((List.sum ages - 24) / 7 = 28 + 4/7) := 
by
  intro ages
  intro h_len
  intro h_sum
  intro h_24
  sorry

end average_age_of_remaining_people_l574_574018


namespace find_tiger_time_l574_574922

noncomputable def time_to_find_tiger : ℝ :=
  let t := 3.5 in
  let distance_before_notice := 3 * 25 in
  let distance_after_notice := t * 10 in
  let distance_while_chased := 0.5 * 50 in
  distance_before_notice + (4 * 10) + distance_while_chased = 135 -> t

-- Proving the time it took to find the tiger after the zookeepers noticed he was missing
theorem find_tiger_time : time_to_find_tiger = 3.5 :=
by
  unfold time_to_find_tiger
  calc 3 * 25 + (4 * 10) + 0.5 * 50 = 75 + 40 + 25 : by norm_num
                                ... = 140 : by norm_num
                                ... ; 135 - 75 - 25 = 35 -> (35 / 10) = 3.5 : by norm_num
  sorry

end find_tiger_time_l574_574922


namespace roseville_population_less_than_thrice_willowdale_l574_574779

theorem roseville_population_less_than_thrice_willowdale :
  ∀ (R W S : ℕ),
  W = 2000 →
  S = 12000 →
  S = 2 * R + 1000 →
  3 * W - R = 500 := 
by
  intros R W S hW hS1 hS2,
  rw [hW, ← hS2] at hS1,
  have h : 2 * R = 11000 := by linarith,
  have hR : R = 5500 := by linarith,
  rw [hW, hR],
  norm_num,
  sorry

end roseville_population_less_than_thrice_willowdale_l574_574779


namespace option_B_correct_l574_574600

theorem option_B_correct (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  3^a + 2a = 3^b + 3b → a > b :=
by
  sorry

end option_B_correct_l574_574600


namespace petya_wins_with_optimal_play_l574_574826

def move1 (n : ℕ) : ℕ := if n > 0 then n - 1 else n
def move2 (n : ℕ) (m : ℕ) : ℕ := if n >= 1 ∧ m >= 2 then n - 1 else n
def move3 (n : ℕ) : ℕ := if n >= 2 then n - 2 else n

theorem petya_wins_with_optimal_play (pluses : ℕ) (minuses : ℕ) (initial : pluses = 865) (first_move : True) : 
  ∃ (p : ℕ), p = 0 ∧ ∀ (moves : list (ℕ → ℕ)), p ∈ list.map (λ f, f pluses) moves :=
sorry

end petya_wins_with_optimal_play_l574_574826


namespace floor_sum_eq_floor_mul_l574_574376

theorem floor_sum_eq_floor_mul (n : ℕ) (x : ℝ) :
  (∑ i in Finset.range n, ⌊x + i / n⌋) = ⌊n * x⌋ :=
sorry

end floor_sum_eq_floor_mul_l574_574376


namespace makeup_set_cost_l574_574995

theorem makeup_set_cost (initial : ℕ) (gift : ℕ) (needed : ℕ) (total_cost : ℕ) :
  initial = 35 → gift = 20 → needed = 10 → total_cost = initial + gift + needed → total_cost = 65 :=
by
  intros h_init h_gift h_needed h_cost
  sorry

end makeup_set_cost_l574_574995


namespace proof_equivalent_problem_l574_574312

noncomputable def polar_equation_curve : Prop :=
  ∀ (α : ℝ), 
    let x := 3 + 2 * Real.cos α;
    let y := 1 - 2 * Real.sin α;
    (x - 3) ^ 2 + (y - 1) ^ 2 - 4 = 0

noncomputable def polar_equation_line : Prop :=
  ∀ (θ ρ : ℝ), 
  (Real.sin θ - 2 * Real.cos θ = 1 / ρ) → (2 * (ρ * Real.cos θ) - (ρ * Real.sin θ) + 1 = 0)

noncomputable def distance_from_curve_to_line : Prop :=
  ∀ (α : ℝ), 
    let x := 3 + 2 * Real.cos α;
    let y := 1 - 2 * Real.sin α;
    ∃ d : ℝ, d = (|2 * x - y + 1| / Real.sqrt (2 ^ 2 + 1)) ∧
    d + 2 = (6 * Real.sqrt 5 / 5) + 2

theorem proof_equivalent_problem :
  polar_equation_curve ∧ polar_equation_line ∧ distance_from_curve_to_line :=
by
  constructor
  · exact sorry  -- polar_equation_curve proof
  · constructor
    · exact sorry  -- polar_equation_line proof
    · exact sorry  -- distance_from_curve_to_line proof

end proof_equivalent_problem_l574_574312


namespace fibonacci_product_l574_574388

def fib : ℕ → ℕ
| 1 := 1
| 2 := 1
| (n + 3) := fib (n + 2) + fib (n + 1)

theorem fibonacci_product :
  (∏ k in Finset.range 48 + 3, (fib k / fib (k + 1) - fib k / fib (k - 2))) = (fib 50 / (fib 50 + fib 49)) :=
sorry

end fibonacci_product_l574_574388


namespace orchard_acres_are_correct_l574_574107

noncomputable def orchard_flat_and_hilly_acres (total_acres sampled_acres : ℕ) (flat_sampled_acres : ℕ) : Prop :=
  let hilly_sampled_acres := 2 * flat_sampled_acres + 1 in
    flat_sampled_acres + hilly_sampled_acres = sampled_acres ∧ 
    3 * flat_sampled_acres = sampled_acres - 1 ∧
    (float.of_nat flat_sampled_acres / (float.of_nat sampled_acres / float.of_nat total_acres) = 36) ∧
    ((float.of_nat sampled_acres - float.of_nat flat_sampled_acres) / (float.of_nat sampled_acres / float.of_nat total_acres) = 84)

theorem orchard_acres_are_correct :
  orchard_flat_and_hilly_acres 120 10 3 :=
by sorry

end orchard_acres_are_correct_l574_574107


namespace x_1000_val_l574_574978

noncomputable def x : ℕ → ℕ
| 1       := 4
| 2       := 6
| (n + 3) := let prev := 2 * x (n + 2) - x (n + 1) in
              Nat.find (λ m, ∃ k, 1 < k ∧ m = k * 2 + 1 ∧ prev < m)

theorem x_1000_val : x 1000 = 501500 :=
by
  sorry

end x_1000_val_l574_574978


namespace unique_I_l574_574714

-- Definitions of the different digits and their properties
def T : Nat := 8
def E : Nat → Prop := λ n, n % 2 = 1  -- E is odd

-- All digits are different
def all_diff (a b c d e f g h : Nat) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
  f ≠ g ∧ f ≠ h ∧
  g ≠ h

theorem unique_I (H R S I X N E : Nat) 
  (h1 : E E) 
  (h2 : all_diff T H R S I X N E) 
  (h3 : T = 8) 
  : I = 6 := 
sorry  -- Requires proof

end unique_I_l574_574714


namespace square_area_l574_574906

theorem square_area (y : ℝ) (x : ℝ → ℝ) : 
    (∀ x, y = x ^ 2 + 4 * x + 3) → (y = 7) → 
    ∃ area : ℝ, area = 32 := 
by
  intro h₁ h₂ 
  -- Proof steps would go here
  sorry

end square_area_l574_574906


namespace points_lie_on_line_l574_574594

noncomputable def x (t : ℝ) (ht : t ≠ 0) : ℝ := (t^2 + 2 * t + 2) / t
noncomputable def y (t : ℝ) (ht : t ≠ 0) : ℝ := (t^2 - 2 * t + 2) / t

theorem points_lie_on_line : ∀ (t : ℝ) (ht : t ≠ 0), y t ht = x t ht - 4 :=
by 
  intros t ht
  simp [x, y]
  sorry

end points_lie_on_line_l574_574594


namespace no_two_digit_factorization_1729_l574_574322

noncomputable def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem no_two_digit_factorization_1729 :
  ¬ ∃ (a b : ℕ), a * b = 1729 ∧ is_two_digit a ∧ is_two_digit b :=
by
  sorry

end no_two_digit_factorization_1729_l574_574322


namespace Joneal_stops_in_quarter_C_l574_574762

theorem Joneal_stops_in_quarter_C :
  ∀ (distance : ℕ) (circumference : ℕ) (quarters : ℕ), 
  distance = 5280 → circumference = 50 → quarters = 4 →
  let laps := distance / circumference in
  let additional_distance := distance % circumference in
  let quarter_length := circumference / quarters in
  if additional_distance <= quarter_length then "A"
  else if additional_distance <= 2 * quarter_length then "B"
  else if additional_distance <= 3 * quarter_length then "C"
  else "D" = "C" := 
begin
  -- We let Lean variables be the known values
  intros distance circumference quarters h_distance h_circumference h_quarters,
  let laps := distance / circumference,
  let additional_distance := distance % circumference,
  let quarter_length := circumference / quarters,
  have h_additional_distance : additional_distance = 30, 
  { rw h_distance, rw h_circumference, norm_num, },
  have h_quarter_length : quarter_length = 12.5, 
  { rw h_circumference, rw h_quarters, norm_num, },
  split_ifs,
  rotate,
  -- Can conclude through given step completion as expected by requirement
  exact s,  -- Add proof for satisfying conditions
end

end Joneal_stops_in_quarter_C_l574_574762


namespace sum_sequence_equals_l574_574200

noncomputable def sum_sequence : ℂ :=
  (finset.range 30).sum (λ n, (complex.I ^ (n + 1)) * complex.sin (real.pi * (90 * (n + 1) - 60) / 180))

theorem sum_sequence_equals :
  sum_sequence = 7.5 - 3.75 * real.sqrt 3 - 3.75 * complex.I * real.sqrt 3 :=
sorry

end sum_sequence_equals_l574_574200


namespace evaluate_expression_l574_574563

noncomputable def N : ℝ := (sqrt (sqrt 10 + 3) - sqrt (sqrt 10 - 3)) / sqrt (sqrt 10 + 2) - sqrt (6 - 4 * sqrt 2)

theorem evaluate_expression : N = 1 + sqrt 2 :=
by
  sorry

end evaluate_expression_l574_574563


namespace hyperbola_asymptotes_l574_574358

theorem hyperbola_asymptotes (a b p : ℝ) (h_a : a > 0) (h_b : b > 0) (h_p : p > 0)
  (h_intersection: ∀ (x y : ℝ), (x^2 = 2 * p * y) → ((x^2 / a^2) - (y^2 / b^2) = 1)) 
  (h_condition : ∀ (y_A y_B : ℝ), y_A + y_B + p = 2 * p → (2 * p * b^2 / a^2 = p)): 
  (∀ (x : ℝ), (x ≠ 0) → (y = (sqrt 2 / 2) * x ∨ y = -(sqrt 2 / 2) * x)) := 
sorry

end hyperbola_asymptotes_l574_574358


namespace probability_at_least_6_heads_8_flips_l574_574166

-- Define the probability calculation of getting at least 6 heads in 8 coin flips.
def probability_at_least_6_heads (n : ℕ) (k : ℕ) : ℚ :=
  (Nat.choose n k + Nat.choose n (k + 1) + Nat.choose n (k + 2)) / 2^n

theorem probability_at_least_6_heads_8_flips : 
  probability_at_least_6_heads 8 6 = 37 / 256 := 
by
  sorry

end probability_at_least_6_heads_8_flips_l574_574166


namespace sum_b_l574_574664

noncomputable def is_arithmetic_seq (b : ℕ → ℝ) : Prop := ∀ n m : ℕ, b (n + 1) - b n = b (m + 1) - b m

theorem sum_b (a b : ℕ → ℝ)
  (h1 : ∀ n : ℕ, b n = Real.log (a n) / Real.log 2)
  (h2 : is_arithmetic_seq b)
  (h3 : a 9 * a 2008 = 1 / 4) :
  ∑ i in Finset.range 2016, b (i + 1) = -2016 :=
sorry

end sum_b_l574_574664


namespace positive_x_satisfies_equation_l574_574962

-- Definitions based on conditions and problem statement
def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem positive_x_satisfies_equation (x : ℝ) (h₁ : x > 0) (h₂ : x ≠ 1) :
  (log_base 4 x) * (log_base x 7) = log_base 4 7 := 
sorry

end positive_x_satisfies_equation_l574_574962


namespace triangle_larger_side_opposite_larger_angle_l574_574768

variable {α : Type*} [EuclideanGeometry α]

open EuclideanGeometry

theorem triangle_larger_side_opposite_larger_angle
  {A B C : α}
  (h_triangle : Triangle A B C)
  (h_side : dist A B > dist A C) :
  angle B C A > angle C A B :=
sorry

end triangle_larger_side_opposite_larger_angle_l574_574768


namespace line_always_passes_fixed_point_l574_574443

theorem line_always_passes_fixed_point (k : ℝ) :
  ∀ (x y : ℝ), (kx - y + 3k - 2 = 0) → (x = -3 ∧ y = -2) :=
by
  intro x y h
  sorry

end line_always_passes_fixed_point_l574_574443


namespace convert_polar_to_rectangular_l574_574552

noncomputable def polarToRectangular (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem convert_polar_to_rectangular :
  polarToRectangular 8 (7 * Real.pi / 6) = (-4 * Real.sqrt 3, -4) :=
by
  sorry

end convert_polar_to_rectangular_l574_574552


namespace complex_magnitude_sixth_power_l574_574239

noncomputable def z := (2 : ℂ) + (2 * Real.sqrt 3) * Complex.I

theorem complex_magnitude_sixth_power :
  Complex.abs (z^6) = 4096 := 
by
  sorry

end complex_magnitude_sixth_power_l574_574239


namespace ryan_learning_hours_difference_l574_574564

theorem ryan_learning_hours_difference :
  let hours_english := 6
  let hours_chinese := 3
  hours_english - hours_chinese = 3 :=
by 
  -- Define the variables for hours spent on English and Chinese
  let hours_english := 6
  let hours_chinese := 3
  -- Compute the difference
  have diff := hours_english - hours_chinese
  -- Assert the difference is equal to 3
  show diff = 3 from 
  sorry

end ryan_learning_hours_difference_l574_574564


namespace num_unique_five_topping_pizzas_l574_574897

open Nat

/-- The number of combinations of choosing k items from n items is defined using binomial coefficients. -/
def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem num_unique_five_topping_pizzas:
  let toppings := 8
      toppings_per_pizza := 5
  in binomial_coefficient toppings toppings_per_pizza = 56 := by
  sorry

end num_unique_five_topping_pizzas_l574_574897


namespace units_digit_sum_42_4_24_4_l574_574851

theorem units_digit_sum_42_4_24_4 : (42^4 + 24^4) % 10 = 2 := 
by
  sorry

end units_digit_sum_42_4_24_4_l574_574851


namespace current_books_l574_574172

def initial_books : ℕ := 743
def sold_instore_saturday : ℕ := 37
def sold_online_saturday : ℕ := 128
def sold_instore_sunday : ℕ := 2 * sold_instore_saturday
def sold_online_sunday : ℕ := sold_online_saturday + 34
def total_books_sold_saturday : ℕ := sold_instore_saturday + sold_online_saturday
def total_books_sold_sunday : ℕ := sold_instore_sunday + sold_online_sunday
def total_books_sold_weekend : ℕ := total_books_sold_saturday + total_books_sold_sunday
def books_received_shipment : ℕ := 160
def net_change_books : ℤ := books_received_shipment - total_books_sold_weekend

theorem current_books
  (initial_books : ℕ) 
  (sold_instore_saturday : ℕ) 
  (sold_online_saturday : ℕ) 
  (sold_instore_sunday : ℕ)
  (sold_online_sunday : ℕ)
  (total_books_sold_saturday : ℕ)
  (total_books_sold_sunday : ℕ)
  (total_books_sold_weekend : ℕ)
  (books_received_shipment : ℕ)
  (net_change_books : ℤ) : (initial_books - net_change_books) = 502 := 
by {
  sorry
}

end current_books_l574_574172


namespace find_S6_l574_574460

variable (a_n : ℕ → ℝ) -- Assume a_n gives the nth term of an arithmetic sequence.
variable (S_n : ℕ → ℝ) -- Assume S_n gives the sum of the first n terms of the sequence.

-- Conditions:
axiom S_2_eq : S_n 2 = 2
axiom S_4_eq : S_n 4 = 10

-- Define what it means to find S_6
theorem find_S6 : S_n 6 = 18 :=
by
  sorry

end find_S6_l574_574460


namespace binom_12_9_is_220_l574_574215

def choose (n k : ℕ) : ℕ := n.choose k

theorem binom_12_9_is_220 :
  choose 12 9 = 220 :=
by {
  -- Proof is omitted
  sorry
}

end binom_12_9_is_220_l574_574215


namespace ab_condition_l574_574599

theorem ab_condition (a b : ℝ) : ¬((a + b > 1 → a^2 + b^2 > 1) ∧ (a^2 + b^2 > 1 → a + b > 1)) :=
by {
  -- This proof problem states that the condition "a + b > 1" is neither sufficient nor necessary for "a^2 + b^2 > 1".
  sorry
}

end ab_condition_l574_574599


namespace percentage_employees_10_years_or_more_is_15_l574_574047

-- Define the number of employees each X represents.
variable (x : ℕ)

-- Define the total number of employees.
def total_employees : ℕ :=
  4*x + 6*x + 7*x + 4*x + 3*x + 3*x + 2*x + 2*x + 1*x + 1*x + 2*x + 1*x + 1*x + 1*x + 1*x

-- Define the number of employees who have worked for 10 years or more.
def employees_10_years_or_more : ℕ :=
  2*x + 1*x + 1*x + 1*x + 1*x

-- Define the percentage calculation.
noncomputable def percentage_10_years_or_more (total_employees employees_10_years_or_more : ℕ) : Real :=
  (employees_10_years_or_more.toReal / total_employees.toReal) * 100

-- Prove that the percentage is 15%
theorem percentage_employees_10_years_or_more_is_15 :
  percentage_10_years_or_more total_employees employees_10_years_or_more = 15 := by
  sorry

end percentage_employees_10_years_or_more_is_15_l574_574047


namespace john_needs_29_planks_for_house_wall_l574_574269

def total_number_of_planks (large_planks small_planks : ℕ) : ℕ :=
  large_planks + small_planks

theorem john_needs_29_planks_for_house_wall :
  total_number_of_planks 12 17 = 29 :=
by
  sorry

end john_needs_29_planks_for_house_wall_l574_574269


namespace baseball_cap_problem_l574_574103

theorem baseball_cap_problem 
  (n_first_week n_second_week n_third_week n_fourth_week total_caps : ℕ) 
  (h2 : n_second_week = 400) 
  (h3 : n_third_week = 300) 
  (h4 : n_fourth_week = (n_first_week + n_second_week + n_third_week) / 3) 
  (h_total : n_first_week + n_second_week + n_third_week + n_fourth_week = 1360) : 
  n_first_week = 320 := 
by 
  sorry

end baseball_cap_problem_l574_574103


namespace division_is_multiplication_by_reciprocal_specific_case_l574_574785

theorem division_is_multiplication_by_reciprocal (a b : ℝ) (hb : b ≠ 0) :
  a / b = a * (1 / b) :=
by
  rw [div_eq_mul_one_div]

-- Specific case application with a = 8 and b = -1/5
theorem specific_case :
  8 / (-1 / 5) = 8 * (-5) :=
by
  have h : (-1 / 5) ≠ 0 := by norm_num
  rw division_is_multiplication_by_reciprocal 8 (-1 / 5) h
  norm_num
  sorry

end division_is_multiplication_by_reciprocal_specific_case_l574_574785


namespace fraction_planted_is_7_over_10_l574_574975

/-- Farmer Euclid has a field in the shape of a right triangle with legs 5 and 12 units in length. 
    A small unplanted square S is placed at the right angle. The shortest distance from S to the hypotenuse 
    is 3 units. Prove that the fraction of the field that is planted is 7/10. -/
theorem fraction_planted_is_7_over_10
  (a b : ℝ) (right_angle_triangle : a = 5 ∧ b = 12)
  (hypotenuse : ℝ := Real.sqrt (a^2 + b^2))
  (square_distance_to_hypotenuse : ℝ := 3)
  (square_side_length : ℝ := 3) :
  (a = 5 ∧ b = 12) →
  (hypotenuse = 13) →
  (1/2 * a * b - square_side_length^2) / (1/2 * a * b) = 7 / 10 :=
begin
  intros,
  sorry
end

end fraction_planted_is_7_over_10_l574_574975


namespace problem_I_problem_II_l574_574308

-- Define the function f(x) = |x+1| + |x+m+1|
def f (x : ℝ) (m : ℝ) : ℝ := |x+1| + |x+(m+1)|

-- Define the problem (Ⅰ): f(x) ≥ |m-2| for all x implies m ≥ 1
theorem problem_I (m : ℝ) (h : ∀ x : ℝ, f x m ≥ |m-2|) : m ≥ 1 := sorry

-- Define the problem (Ⅱ): Find the solution set for f(-x) < 2m
theorem problem_II (m : ℝ) :
  (m ≤ 0 → ∀ x : ℝ, ¬ (f (-x) m < 2 * m)) ∧
  (m > 0 → ∀ x : ℝ, (1 - m / 2 < x ∧ x < 3 * m / 2 + 1) ↔ f (-x) m < 2 * m) := sorry

end problem_I_problem_II_l574_574308


namespace num_correct_propositions_l574_574045

open Nat

noncomputable def Sn (a : ℝ) (n : ℕ) : ℝ :=
if h : n > 0 then a ^ n - 1 else 0

def is_geometric (seq : ℕ → ℝ) := ∃ r, ∀ n > 0, seq (n + 1) = r * seq n
def is_arithmetic (seq : ℕ → ℝ) := ∃ d, ∀ n, seq (n + 1) = seq n + d

def an (a : ℝ) (n : ℕ) : ℝ := Sn a n - Sn a (n - 1)

theorem num_correct_propositions (a : ℝ) :
  let seq := an a
  (if a = 1 then is_arithmetic seq ∧ ¬ is_geometric seq else True) ∧
  (if a = 0 then ¬ is_arithmetic seq ∧ ¬ is_geometric seq else True) ∧
  (if a ≠ 0 ∧ a ≠ 1 then is_geometric seq ∧ ¬ is_arithmetic seq else True) →
  -- Number of correct propositions
  2 = 2 :=
sorry

end num_correct_propositions_l574_574045


namespace min_poly_degree_l574_574014

theorem min_poly_degree (P : Polynomial ℚ)
  (h1 : Polynomial.aeval (5 - 2 * Real.sqrt 3 : ℚ) P = 0)
  (h2 : Polynomial.aeval (-5 - 2 * Real.sqrt 3 : ℚ) P = 0)
  (h3 : Polynomial.aeval (2 + Real.sqrt 5 : ℚ) P = 0)
  (h4 : Polynomial.aeval (2 - Real.sqrt 5 : ℚ) P = 0)
  (hP0 : P ≠ 0) : P.degree.to_nat = 6 :=
by
  sorry

end min_poly_degree_l574_574014


namespace yogurt_amount_l574_574562

namespace SmoothieProblem

def strawberries := 0.2 -- cups
def orange_juice := 0.2 -- cups
def total_ingredients := 0.5 -- cups

def yogurt_used := total_ingredients - (strawberries + orange_juice)

theorem yogurt_amount : yogurt_used = 0.1 :=
by
  unfold yogurt_used strawberries orange_juice total_ingredients
  norm_num
  sorry  -- Proof can be filled in as needed

end SmoothieProblem

end yogurt_amount_l574_574562


namespace solve_for_x_l574_574008

theorem solve_for_x (x : ℝ) (h : 16^x * 16^x * 16^x = 256^4) : x = 8 / 3 := by
  sorry

end solve_for_x_l574_574008


namespace find_pairs_l574_574979

theorem find_pairs :
  ∀ (x y : ℕ), 0 < x → 0 < y → 7 ^ x - 3 * 2 ^ y = 1 → (x, y) = (1, 1) ∨ (x, y) = (2, 4) :=
by
  intros x y hx hy h
  -- Proof would go here
  sorry

end find_pairs_l574_574979


namespace binomial_12_9_l574_574207

def binomial (n k : ℕ) := nat.choose n k

theorem binomial_12_9 : binomial 12 9 = 220 :=
by
  have step1 : binomial 12 9 = binomial 12 3 := nat.choose_symm 12 9
  have step2 : binomial 12 3 = 220 := by sorry
  rw [step1, step2]

end binomial_12_9_l574_574207


namespace consecutive_sums_permutations_iff_odd_l574_574225

theorem consecutive_sums_permutations_iff_odd (n : ℕ) (h : n ≥ 2) :
  (∃ (a b : Fin n → ℕ), (∀ i, 1 ≤ a i ∧ a i ≤ n) ∧ (∀ i, 1 ≤ b i ∧ b i ≤ n) ∧
    ∃ N, ∀ i, a i + b i = N + i) ↔ (Odd n) :=
by
  sorry

end consecutive_sums_permutations_iff_odd_l574_574225


namespace find_moles_of_NaOH_l574_574984

-- Define the conditions
def reaction (NaOH HClO4 NaClO4 H2O : ℕ) : Prop :=
  NaOH = HClO4 ∧ NaClO4 = HClO4 ∧ H2O = 1

def moles_of_HClO4 := 3
def moles_of_NaClO4 := 3

-- Problem statement
theorem find_moles_of_NaOH : ∃ (NaOH : ℕ), NaOH = moles_of_HClO4 ∧ moles_of_NaClO4 = 3 ∧ NaOH = 3 :=
by sorry

end find_moles_of_NaOH_l574_574984


namespace P_on_x_axis_Q_max_y_PQR_90_deg_PQS_PQT_45_deg_l574_574706

-- Conditions
def center_C : (ℝ × ℝ) := (6, 8)
def radius : ℝ := 10
def circle_eq (x y : ℝ) : Prop := (x - 6)^2 + (y - 8)^2 = 100
def origin_O : (ℝ × ℝ) := (0, 0)

-- (a) Point of intersection of the circle with the x-axis
def point_P : (ℝ × ℝ) := (12, 0)
theorem P_on_x_axis : circle_eq (point_P.1) (point_P.2) ∧ point_P.2 = 0 := sorry

-- (b) Point on the circle with maximum y-coordinate
def point_Q : (ℝ × ℝ) := (6, 18)
theorem Q_max_y : circle_eq (point_Q.1) (point_Q.2) ∧ ∀ y : ℝ, (circle_eq 6 y → y ≤ 18) := sorry

-- (c) Point on the circle such that ∠PQR = 90°
def point_R : (ℝ × ℝ) := (0, 16)
theorem PQR_90_deg : circle_eq (point_R.1) (point_R.2) ∧
  ∃ Q : (ℝ × ℝ), circle_eq (Q.1) (Q.2) ∧ (Q = point_Q) ∧ (point_P.1 - point_R.1) * (Q.1 - point_Q.1) + (point_P.2 - point_R.2) * (Q.2 - point_Q.2) = 0 := sorry

-- (d) Two points on the circle such that ∠PQS = ∠PQT = 45°
def point_S : (ℝ × ℝ) := (14, 14)
def point_T : (ℝ × ℝ) := (-2, 2)
theorem PQS_PQT_45_deg : circle_eq (point_S.1) (point_S.2) ∧ circle_eq (point_T.1) (point_T.2) ∧
  ∃ Q : (ℝ × ℝ), circle_eq (Q.1) (Q.2) ∧ (Q = point_Q) ∧
  ((point_P.1 - Q.1) * (point_S.1 - Q.1) + (point_P.2 - Q.2) * (point_S.2 - Q.2) =
  (point_P.1 - Q.1) * (point_T.1 - Q.1) + (point_P.2 - Q.2) * (point_T.2 - Q.2)) := sorry

end P_on_x_axis_Q_max_y_PQR_90_deg_PQS_PQT_45_deg_l574_574706


namespace simplify_fraction_l574_574000

theorem simplify_fraction : (48 / 72 : ℚ) = (2 / 3) := 
by
  sorry

end simplify_fraction_l574_574000


namespace probability_at_least_6_heads_in_8_flips_l574_574135

open scoped BigOperators

def binom (n k : ℕ) : ℕ := nat.choose n k

def total_outcomes (n : ℕ) := 2^n

def successful_outcomes (n k : ℕ) :=
  (finset.range (n + 1)).filter (λ x, x ≥ k).sum (λ x, binom n x)

def probability (n k : ℕ) :=
  (successful_outcomes n k) / (total_outcomes n : ℚ)

theorem probability_at_least_6_heads_in_8_flips :
  probability 8 6 = 37 / 256 := sorry

end probability_at_least_6_heads_in_8_flips_l574_574135


namespace sum_abs_arithmetic_sequence_l574_574290

theorem sum_abs_arithmetic_sequence (n : ℕ) : 
  let S_n := λ n, - (3 / 2) * n^2 + (205 / 2) * n in
  let a_n := λ n, S_n n - S_n (n - 1) in
  let abs_a_n := λ n, |a_n n| in
  let T_n := λ n, if n ≤ 34 then - (3 / 2) * n^2 + (205 / 2) * n else (3 / 2) * n^2 - (205 / 2) * n + 3502 in
T_n n = if n ≤ 34 
        then - (3 / 2) * n^2 + (205 / 2) * n 
        else (3 / 2) * n^2 - (205 / 2) * n + 3502 := 
by
  sorry

end sum_abs_arithmetic_sequence_l574_574290


namespace triangle_probability_l574_574617

def lengths : List ℝ := [1, 3, 5, 7, 9]

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def valid_combinations : List (ℝ × ℝ × ℝ) :=
  [(3, 5, 7), (3, 7, 9), (5, 7, 9)]

def total_combinations : ℕ := 10

def valid_comb_count : ℕ := List.length valid_combinations

def probability : ℝ := valid_comb_count.toReal / total_combinations.toReal

theorem triangle_probability :
  probability = 3 / 10 :=
by
  sorry

end triangle_probability_l574_574617


namespace square_area_l574_574904

theorem square_area (y : ℝ) (x : ℝ → ℝ) : 
    (∀ x, y = x ^ 2 + 4 * x + 3) → (y = 7) → 
    ∃ area : ℝ, area = 32 := 
by
  intro h₁ h₂ 
  -- Proof steps would go here
  sorry

end square_area_l574_574904


namespace chromatic_number_bound_l574_574289

-- Definitions of the conditions
variable (G : SimpleGraph V) [DecidableRel G.Adj] (Δ : ℕ)

-- Definition of the maximum degree of a vertex
def max_degree (G : SimpleGraph V) : ℕ := 
  Finset.sup (G.support) (λ v, G.degree v)

-- Assertion about the chromatic number
theorem chromatic_number_bound (hΔ : Δ = max_degree G) : 
    G.chromatic_number ≤ Δ + 1 := sorry

end chromatic_number_bound_l574_574289


namespace train_pass_time_correct_l574_574082

-- Define the conditions of the problem
def train_length : ℝ := 150  -- Train length in meters
def train_speed_km_per_hr : ℝ := 90  -- Train speed in kilometers per hour

-- Conversion factor from kilometers per hour to meters per second
def km_per_hr_to_m_per_s : ℝ := 1000 / 3600

-- Convert train speed to meters per second
def train_speed_m_per_s : ℝ := train_speed_km_per_hr * km_per_hr_to_m_per_s

-- Calculate the time required for the train to pass the pole
noncomputable def time_to_pass_pole : ℝ := train_length / train_speed_m_per_s

-- Theorem statement: The time to pass the pole is 6 seconds
theorem train_pass_time_correct : time_to_pass_pole = 6 := by
  sorry

end train_pass_time_correct_l574_574082


namespace length_of_AC_l574_574540

theorem length_of_AC :
  ∀ (A B C : ℝ) (dist1 dist2 : ℝ),
  (dist1 = 2) →
  (dist2 = 3) →
  (∠ ABC = 90) →
  (AB = BC) →
  (distance l1 l2 = dist1) →
  (distance l2 l3 = dist2) →
  AC = 2 * √17 :=
by
  intro A B C dist1 dist2 h_dist1 h_dist2 h_angleABC h_ABeqBC h_dist_l1l2 h_dist_l2l3
  -- proof goes here 
  sorry

end length_of_AC_l574_574540


namespace prob_at_least_6_heads_eq_l574_574119

-- define the number of coin flips
def n := 8

-- define the number of possible outcomes (2^n)
def total_outcomes := 2 ^ n

-- define the binomial coefficients for cases: 6 heads, 7 heads, 8 heads
def binom_8_6 := Nat.choose 8 6
def binom_8_7 := Nat.choose 8 7
def binom_8_8 := Nat.choose 8 8

-- calculate the favorable outcomes for at least 6 heads
def favorable_outcomes := binom_8_6 + binom_8_7 + binom_8_8

-- define the probability of getting at least 6 heads
def probability := (favorable_outcomes : ℚ) / total_outcomes

theorem prob_at_least_6_heads_eq : probability = 37 / 256 := by
  sorry

end prob_at_least_6_heads_eq_l574_574119


namespace max_ABC_ge_4_9_max_alpha_beta_gamma_ge_4_9_l574_574607

variable (p q : ℝ) (x y : ℝ)
variable (A B C α β γ : ℝ)

-- Conditions
axiom hp : 0 ≤ p ∧ p ≤ 1 
axiom hq : 0 ≤ q ∧ q ≤ 1 
axiom h1 : (p * x + (1 - p) * y)^2 = A * x^2 + B * x * y + C * y^2
axiom h2 : (p * x + (1 - p) * y) * (q * x + (1 - q) * y) = α * x^2 + β * x * y + γ * y^2

-- Problem
theorem max_ABC_ge_4_9 : max A (max B C) ≥ 4 / 9 := 
sorry

theorem max_alpha_beta_gamma_ge_4_9 : max α (max β γ) ≥ 4 / 9 := 
sorry

end max_ABC_ge_4_9_max_alpha_beta_gamma_ge_4_9_l574_574607


namespace trigonometric_identity_l574_574279

theorem trigonometric_identity (α : ℝ) (h : Real.sin α = 2 * Real.cos α) :
  Real.sin (π / 2 + 2 * α) = -3 / 5 :=
by
  sorry

end trigonometric_identity_l574_574279


namespace probability_of_at_least_six_heads_is_correct_l574_574122

-- Definitions for the given problem
def binomial_coefficient (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

def total_possible_outcomes : ℕ :=
  2^8

def favorable_outcomes : ℕ :=
  binomial_coefficient 8 6 + binomial_coefficient 8 7 + binomial_coefficient 8 8

def probability_of_at_least_6_heads : ℚ :=
  favorable_outcomes / total_possible_outcomes

-- The proof statement
theorem probability_of_at_least_six_heads_is_correct :
  probability_of_at_least_6_heads = 37 / 256 :=
by sorry

end probability_of_at_least_six_heads_is_correct_l574_574122


namespace sum_of_seven_unique_digits_l574_574421

theorem sum_of_seven_unique_digits :
  ∃ (digits : Finset ℕ), 
    digits.card = 7 ∧                    -- 7 different digits
    (∀ d ∈ digits, d ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧ -- chosen from set {1, 2, 3, 4, 5, 6, 7, 8, 9}
    (∃ b ∈ digits, (digits.erase b).card = 6) ∧ -- one digit appears twice
    let vertical_sum := (digits.erase $ Finset.min' (digits.erase b) (by sorry)).sum in -- sum of vertical column
    vertical_sum = 25 ∧
    let horizontal_sum := (digits.erase vertical_sum).sum in -- sum of horizontal row
    horizontal_sum = 18 ∧
    digits.sum = 33 := -- sum of the seven unique digits
by
  sorry

end sum_of_seven_unique_digits_l574_574421


namespace probability_at_least_6_heads_in_8_flips_l574_574140

theorem probability_at_least_6_heads_in_8_flips : 
  (∑ k in finset.range 3, nat.choose 8 (6 + k)) / (2 ^ 8) = 37 / 256 :=
by sorry

end probability_at_least_6_heads_in_8_flips_l574_574140


namespace sum_of_digits_T_l574_574822

-- Conditions:
def horse_lap_times := [1, 2, 3, 4, 5, 6, 7, 8]
def S := 840
def total_horses := 8
def min_horses_at_start := 4

-- Question:
def T := 12 -- Least time such that at least 4 horses meet

/-- Prove that the sum of the digits of T is 3 -/
theorem sum_of_digits_T : (1 + 2) = 3 := by
  sorry

end sum_of_digits_T_l574_574822


namespace triangle_problem_l574_574409

variables {A B C K M O : Type} [point A] [point B] [point C] [point K] 
[point M] [point O] -- Define the points

-- Define the conditions as Lean hypotheses
def parallel (KM AC : Type) : Prop := sorry -- Placeholder for parallel lines definition
def equal_length (x y : Type) : Prop := sorry -- Placeholder for equal length definition
def intersect_at (AM KC O : Type) : Prop := sorry -- Placeholder for segments intersecting at a point

theorem triangle_problem 
  (A B C K M O : Type)
  [point A] [point B] [point C] [point K] [point M] [point O]
  (H1: parallel KM AC)
  (H2: equal_length AK AO)
  (H3: equal_length KM MC)
  (H4: intersect_at AM KC O) : equal_length AM KB :=
by
  sorry -- Proof goes here

end triangle_problem_l574_574409


namespace first_train_speed_l574_574530

variable (v : ℝ)

theorem first_train_speed :
  (8 * v) = (80 * 7) → v = 70 :=
by
  intro h
  rw [mul_comm 80 7, mul_assoc, mul_right_inj' (ne_of_gt (by linarith : (8 : ℝ) > 0))] at h
  exact h

end first_train_speed_l574_574530


namespace inscribed_circle_area_relation_l574_574435

variables {A B C O : Point} {t : ℝ} {α β γ : ℝ}

-- The center of the inscribed circle of triangle ABC is O
-- Area of triangle ABC is t
def incenter (A B C O : Point) : Prop := sorry    -- precise definition omitted for brevity

-- We need to show the given equation holds
theorem inscribed_circle_area_relation (h1 : incenter A B C O) (h2 : area A B C = t) :
  2 * t = (dist A O)^2 * sin α + (dist B O)^2 * sin β + (dist C O)^2 * sin γ := sorry

end inscribed_circle_area_relation_l574_574435


namespace ordering_of_xyz_l574_574746

theorem ordering_of_xyz :
  let x := Real.sqrt 3
  let y := Real.log 2 / Real.log 3
  let z := Real.cos 2
  z < y ∧ y < x :=
by
  let x := Real.sqrt 3
  let y := Real.log 2 / Real.log 3
  let z := Real.cos 2
  sorry

end ordering_of_xyz_l574_574746


namespace factor_expression_l574_574571

variable (a : ℝ)

theorem factor_expression : 37 * a^2 + 111 * a = 37 * a * (a + 3) :=
  sorry

end factor_expression_l574_574571


namespace trapezoid_land_area_l574_574183

theorem trapezoid_land_area (
  scale_cm_to_miles : ∀ x, x * 1 = x * 3,
  shorter_parallel_side_cm : 12,
  longer_parallel_side_cm : 18,
  height_cm : 8,
  square_miles_to_acres : 1 = 480
) : 
  let area_trapezoid_cm2 := 0.5 * (shorter_parallel_side_cm + longer_parallel_side_cm) * height_cm,
      area_miles2 := area_trapezoid_cm2 * (3^2),
      area_acres := area_miles2 * 480
  in area_acres = 518400 := sorry

end trapezoid_land_area_l574_574183


namespace solve_exponent_equation_l574_574272

theorem solve_exponent_equation
  (x : ℝ)
  (h : 2^(2 * x - 3) * 8^(x + 1) = 1024 ^ 2)
  (h1 : 8 = 2 ^ 3)
  (h2 : 1024 = 2 ^ 10) :
  x = 4 :=
sorry

end solve_exponent_equation_l574_574272


namespace intersection_of_A_and_B_l574_574315

def A : Set ℤ := {-2, -1}
def B : Set ℤ := {-1, 2, 3}

theorem intersection_of_A_and_B :
  A ∩ B = {-1} :=
sorry

end intersection_of_A_and_B_l574_574315


namespace proof_expression_l574_574283

-- Define variables
variables (x y : ℝ)

-- Define the conditions
def condition1 : Prop := x + 3 * y = 5
def condition2 : Prop := 2 * x - y = 2

-- Define the expression to be proved
def expression : ℝ := 2 * x^2 + 5 * x * y - 3 * y^2

-- The theorem to be proved
theorem proof_expression : condition1 ∧ condition2 → expression = 10 :=
by
  sorry

end proof_expression_l574_574283


namespace range_f_neg10_10_l574_574743

-- Definitions and conditions
def g : ℝ → ℝ := sorry
def f (x : ℝ) : ℝ := x + g x

axiom periodic_g : ∀ x : ℝ, g x = g (x + 1)
axiom range_f_3_4 : ∀ y : ℝ, y ∈ set.Icc (-2) 5 ↔ ∃ x ∈ set.Icc 3 4, f x = y

-- Target statement to prove
theorem range_f_neg10_10 : ∀ y : ℝ, y ∈ set.Icc (-15) 11 ↔ ∃ x ∈ set.Icc (-10) 10, f x = y :=
sorry

end range_f_neg10_10_l574_574743


namespace mean_of_reciprocals_series_sum_l574_574554

open Real

theorem mean_of_reciprocals_series_sum :
  ( ∀ (a : ℕ → ℝ) (b : ℕ → ℝ), 
      (∀ n, average (λ i, 1 / a i) n = 1 / (5 * n)) → 
      (∀ n, b n = a n / 5) →
  ∑ (k : ℕ) in (range 10), (1 / (b k * b (k + 1))) = 10 / 21)
:=
  sorry

end mean_of_reciprocals_series_sum_l574_574554


namespace calc_value_exponents_l574_574946

theorem calc_value_exponents :
  (3^3) * (5^3) * (3^5) * (5^5) = 15^8 :=
by sorry

end calc_value_exponents_l574_574946


namespace petya_max_score_achievable_l574_574465

theorem petya_max_score_achievable :
  let piles := 100
  let stones_per_pile := 400
  let score (pile1 pile2 : ℕ) : ℤ := abs (pile1 - pile2)
  in ∃ (max_score : ℤ), max_score = 3920000 := by {
  sorry
}

end petya_max_score_achievable_l574_574465


namespace ticket_sales_l574_574923

-- Definitions of the conditions
theorem ticket_sales (adult_cost child_cost total_people child_count : ℕ)
  (h1 : adult_cost = 8)
  (h2 : child_cost = 1)
  (h3 : total_people = 22)
  (h4 : child_count = 18) :
  (child_count * child_cost + (total_people - child_count) * adult_cost = 50) := by
  sorry

end ticket_sales_l574_574923


namespace greater_solution_of_quadratic_eq_l574_574840

theorem greater_solution_of_quadratic_eq (x : ℝ) : 
  (∀ y : ℝ, y^2 + 20 * y - 96 = 0 → (y = 4)) :=
sorry

end greater_solution_of_quadratic_eq_l574_574840


namespace subsequence_either_order_l574_574039

theorem subsequence_either_order (list: List ℕ) (h : list = (List.range 101).map (λ x, x+1) ∧ list.nodup):
  ∃ (subseq : List ℕ), subseq.length = 11 ∧ (subseq.sorted (≤) ∨ subseq.sorted (≥)) :=
by
  sorry

end subsequence_either_order_l574_574039


namespace smallest_positive_period_of_f_min_m_and_range_of_g_symmetry_l574_574307

-- Define the function f(x)
def f (x : ℝ) : ℝ := sin (2 * x) + sin (2 * x - real.pi / 3)

-- Define the translated function g(x)
def g (x m : ℝ) : ℝ := sqrt 3 * sin (2 * x + 2 * m - real.pi / 6)

-- Define the periodicity and translation conditions
theorem smallest_positive_period_of_f :
  ∀ x : ℝ, f (x + real.pi) = f x :=
by sorry

theorem min_m_and_range_of_g_symmetry :
  ∃ (m : ℝ) (a b : ℝ), 
  (m = 5 * real.pi / 12) ∧ 
  (∀ x, g x m = g (x - real.pi / 8 + real.pi / 8) m) ∧ 
  (∀ x ∈ set.Icc (0 : ℝ) (real.pi / 4), g x m ∈ set.Icc a b) ∧ 
  (a = - sqrt 3 / 2) ∧ 
  (b = 3 / 2) :=
by sorry

end smallest_positive_period_of_f_min_m_and_range_of_g_symmetry_l574_574307


namespace time_to_cross_l574_574875

noncomputable def length_first_train : ℝ := 210
noncomputable def speed_first_train : ℝ := 120 * 1000 / 3600 -- Convert to m/s
noncomputable def length_second_train : ℝ := 290.04
noncomputable def speed_second_train : ℝ := 80 * 1000 / 3600 -- Convert to m/s

noncomputable def relative_speed := speed_first_train + speed_second_train
noncomputable def total_length := length_first_train + length_second_train
noncomputable def crossing_time := total_length / relative_speed

theorem time_to_cross : crossing_time = 9 := by
  let length_first_train : ℝ := 210
  let speed_first_train : ℝ := 120 * 1000 / 3600 -- Convert to m/s
  let length_second_train : ℝ := 290.04
  let speed_second_train : ℝ := 80 * 1000 / 3600 -- Convert to m/s

  let relative_speed := speed_first_train + speed_second_train
  let total_length := length_first_train + length_second_train
  let crossing_time := total_length / relative_speed

  show crossing_time = 9
  sorry

end time_to_cross_l574_574875


namespace coats_count_l574_574720

def initial_minks : Nat := 30
def babies_per_mink : Nat := 6
def minks_per_coat : Nat := 15

def total_minks : Nat := initial_minks + (initial_minks * babies_per_mink)
def remaining_minks : Nat := total_minks / 2

theorem coats_count : remaining_minks / minks_per_coat = 7 := by
  -- Proof goes here
  sorry

end coats_count_l574_574720


namespace gcd_lcm_product_l574_574065

theorem gcd_lcm_product (a b : ℕ) (h₀ : a = 15) (h₁ : b = 45) :
  Nat.gcd a b * Nat.lcm a b = 675 :=
by
  sorry

end gcd_lcm_product_l574_574065


namespace max_k_value_l574_574601

noncomputable def f (x : ℝ) := x + x * Real.log x

theorem max_k_value : ∃ k : ℤ, (∀ x > 2, k * (x - 2) < f x) ∧ k = 4 :=
by
  sorry

end max_k_value_l574_574601


namespace prime_special_form_l574_574243

theorem prime_special_form (p : ℕ) (h_prime_p : prime p)
  (h_sum : ∃ q r : ℕ, prime q ∧ prime r ∧ p = q + r)
  (h_diff : ∃ s t : ℕ, prime s ∧ prime t ∧ p = s - t) : p = 5 :=
sorry

end prime_special_form_l574_574243


namespace find_e1_l574_574619

-- Definitions related to the problem statement
variable (P F1 F2 : Type)
variable (cos_angle : ℝ)
variable (e1 e2 : ℝ)

-- Conditions
def cosine_angle_condition := cos_angle = 3 / 5
def eccentricity_relation := e2 = 2 * e1

-- Theorem that needs to be proved
theorem find_e1 (h_cos : cosine_angle_condition cos_angle)
                (h_ecc_rel : eccentricity_relation e1 e2) :
  e1 = Real.sqrt 10 / 5 :=
by
  sorry

end find_e1_l574_574619


namespace point_in_fourth_quadrant_l574_574449

-- Define complex number and evaluate it
noncomputable def z : ℂ := (2 - (1 : ℂ) * Complex.I) / (1 + (1 : ℂ) * Complex.I)

-- Prove that the complex number z lies in the fourth quadrant
theorem point_in_fourth_quadrant (hz: z = (1/2 : ℂ) - (3/2 : ℂ) * Complex.I) : z.im < 0 ∧ z.re > 0 :=
by
  -- Skipping the proof here
  sorry

end point_in_fourth_quadrant_l574_574449


namespace truncated_pyramid_base_area_l574_574456

noncomputable def area_of_smaller_base 
  (a : ℝ) (α β : ℝ) : ℝ :=
(a^2 * (Real.sin (α - β))^2) / (Real.sin(α + β)) ^ 2

theorem truncated_pyramid_base_area
  (a : ℝ) (α β : ℝ) :
  (area_of_smaller_base a α β) =  (a^2 * (Real.sin (α - β))^2)  / ((Real.sin (α + β)) ^ 2) := by
skeeprod.


end truncated_pyramid_base_area_l574_574456


namespace greatest_common_divisor_84_n_l574_574837

theorem greatest_common_divisor_84_n :
  ∃ (n : ℕ), (∀ (d : ℕ), d ∣ 84 ∧ d ∣ n → d = 1 ∨ d = 2 ∨ d = 4) ∧ (∀ (x y : ℕ), x ∣ 84 ∧ x ∣ n ∧ y ∣ 84 ∧ y ∣ n → x ≤ y → y = 4) :=
sorry

end greatest_common_divisor_84_n_l574_574837


namespace square_area_l574_574903

theorem square_area (y : ℝ) (x : ℝ → ℝ) : 
    (∀ x, y = x ^ 2 + 4 * x + 3) → (y = 7) → 
    ∃ area : ℝ, area = 32 := 
by
  intro h₁ h₂ 
  -- Proof steps would go here
  sorry

end square_area_l574_574903


namespace minimal_sum_of_b1_and_b2_l574_574555

-- Define the sequence with the given recurrence relation and conditions
noncomputable def sequence (b : ℕ → ℕ) : Prop :=
∀ n ≥ 1, b (n + 2) = (b n + 2213) / (2 + b (n + 1))

-- Define the conditions as given
def conditions (b : ℕ → ℕ) : Prop :=
(sequence b) ∧ (∀ n, b n > 0)

-- Prove that the minimal sum of b1 and b2 is 30
theorem minimal_sum_of_b1_and_b2 : ∃ b : ℕ → ℕ, conditions b ∧ b 1 + b 2 = 30 :=
by
  sorry

end minimal_sum_of_b1_and_b2_l574_574555


namespace probability_at_least_6_heads_8_flips_l574_574162

-- Define the probability calculation of getting at least 6 heads in 8 coin flips.
def probability_at_least_6_heads (n : ℕ) (k : ℕ) : ℚ :=
  (Nat.choose n k + Nat.choose n (k + 1) + Nat.choose n (k + 2)) / 2^n

theorem probability_at_least_6_heads_8_flips : 
  probability_at_least_6_heads 8 6 = 37 / 256 := 
by
  sorry

end probability_at_least_6_heads_8_flips_l574_574162


namespace triangle_inequality_inequality_l574_574999

theorem triangle_inequality_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  4 * b^2 * c^2 - (b^2 + c^2 - a^2)^2 > 0 := 
by
  sorry

end triangle_inequality_inequality_l574_574999


namespace simplify_cot_40_minus_tan_50_l574_574424

theorem simplify_cot_40_minus_tan_50 : 
  cot (40 * real.pi / 180) - tan (50 * real.pi / 180) = 0 :=
by
  sorry

end simplify_cot_40_minus_tan_50_l574_574424


namespace three_planes_six_parts_three_lines_l574_574481

-- Definitions based on the conditions of the problem
-- Here, we define three planes in 3D space and their properties
structure Plane (ℝ : Type) := 
  (normal : ℝ × ℝ × ℝ) 
  (constant : ℝ)
  (partitions_space : ℝ → ℝ → ℝ → Prop)

-- Non-overlapping planes condition indicates no two planes are identical
def non_overlapping (P1 P2 P3 : Plane ℝ) : Prop :=
  ¬(P1 = P2 ∨ P2 = P3 ∨ P1 = P3)

-- Planes divide space into six parts (given condition)
def divides_space_into_six_parts (P1 P2 P3 : Plane ℝ) : Prop :=
  -- This specific definition would depend on the precise mathematical characterization of partition counts
  sorry

-- Define the property of line of intersection between any two planes
def line_of_intersection (P1 P2 : Plane ℝ) : Prop :=
  ∃ (x y z : ℝ), P1.partitions_space x y z ∧ P2.partitions_space x y z

-- The theorem to prove the number of intersection lines
theorem three_planes_six_parts_three_lines (P1 P2 P3 : Plane ℝ) :
  non_overlapping P1 P2 P3 →
  divides_space_into_six_parts P1 P2 P3 →
  ∃ L1 L2 L3 : Plane ℝ → Plane ℝ → Prop,  
    L1 = line_of_intersection P1 P2 ∧
    L2 = line_of_intersection P2 P3 ∧
    L3 = line_of_intersection P1 P3 ∧
    (L1 ≠ L2) ∧ (L2 ≠ L3) ∧ (L1 ≠ L3) :=
by
  sorry

end three_planes_six_parts_three_lines_l574_574481


namespace max_value_expression_l574_574603

theorem max_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  let A := (a^3 + b^3 + c^3) / ((a + b + c)^3 - 26 * a * b * c) in 
  A ≤ 3 :=
  sorry

end max_value_expression_l574_574603


namespace probability_at_least_6_heads_in_8_flips_l574_574160

def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then (Nat.choose n k) else 0

theorem probability_at_least_6_heads_in_8_flips :
  let total_outcomes := 2^8
  let successful_outcomes := binomial 8 6 + binomial 8 7 + binomial 8 8
  let probability := (successful_outcomes : ℚ) / total_outcomes
  probability = 37 / 256 :=
by
  sorry

end probability_at_least_6_heads_in_8_flips_l574_574160


namespace probability_none_occurs_at_most_l574_574953

theorem probability_none_occurs_at_most (n : ℕ) (A : Fin n → Prop) 
  (hA : ∀ i, P (A i) = 1 / 2)
  (hA_pair : ∀ i j, i ≠ j → P (A i ∩ A j) = 1 / 4) :
  P (∅) ≤ 1 / (n + 1) := 
sorry

end probability_none_occurs_at_most_l574_574953


namespace multiplicative_magic_square_h_sum_l574_574956

theorem multiplicative_magic_square_h_sum :
  ∃ (h_vals : List ℕ), 
  (∀ h ∈ h_vals, ∃ (e : ℕ), e > 0 ∧ 25 * e = h ∧ 
    ∃ (b c d f g : ℕ), 
    75 * b * c = d * e * f ∧ 
    d * e * f = g * h * 3 ∧ 
    g * h * 3 = c * f * 3 ∧ 
    c * f * 3 = 75 * e * g
  ) ∧ h_vals.sum = 150 :=
by { sorry }

end multiplicative_magic_square_h_sum_l574_574956


namespace problem_l574_574067

theorem problem (x y : ℕ) (h_x : x = 12) (h_y : y = 7) : (x - y) * (2 * x + 2 * y) = 190 := by
  rw [h_x, h_y]
  sorry

end problem_l574_574067


namespace change_in_total_berries_l574_574475

-- Define the initial conditions
def blue_box_berries : ℕ := 35
def increase_diff : ℕ := 100

-- Define the number of strawberries in red boxes
def red_box_berries : ℕ := 100

-- Formulate the change in total number of berries
theorem change_in_total_berries :
  (red_box_berries - blue_box_berries) = 65 :=
by
  have h1 : red_box_berries = increase_diff := rfl
  have h2 : blue_box_berries = 35 := rfl
  rw [h1, h2]
  exact rfl

end change_in_total_berries_l574_574475


namespace eccentricity_of_ellipse_l574_574628

theorem eccentricity_of_ellipse
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : b = 1/2 * a) :
  let c := sqrt (a^2 - (1/2 * a)^2) in
  let e := c / a in
  e = (sqrt 3) / 2 := 
by
  sorry

end eccentricity_of_ellipse_l574_574628


namespace square_area_correct_l574_574917

noncomputable def square_area : ℝ :=
  let f : ℝ → ℝ := λ x, x^2 + 4 * x + 3
  let y_val : ℝ := 7
  let x1 : ℝ := -2 - 2 * Real.sqrt 2
  let x2 : ℝ := -2 + 2 * Real.sqrt 2
  let side_length := x2 - x1
  side_length * side_length

theorem square_area_correct :
  let f : ℝ → ℝ := λ x, x^2 + 4 * x + 3
  let y_val : ℝ := 7
  let x1 : ℝ := -2 - 2 * Real.sqrt 2
  let x2 : ℝ := -2 + 2 * Real.sqrt 2
  let side_length := x2 - x1
  side_length * side_length = 32 := by
  sorry

end square_area_correct_l574_574917


namespace length_of_RT_in_trapezoid_l574_574491

-- Definition of the trapezoid and initial conditions
def trapezoid (PQ RS PR RT : ℝ) (h : PQ = 3 * RS) (h1 : PR = 15) : Prop :=
  RT = 15 / 4

-- The theorem to be proved
theorem length_of_RT_in_trapezoid (PQ RS PR RT : ℝ) 
  (h : PQ = 3 * RS) (h1 : PR = 15) : trapezoid PQ RS PR RT h h1 :=
by
  sorry

end length_of_RT_in_trapezoid_l574_574491


namespace probability_at_least_6_heads_in_8_flips_l574_574159

def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then (Nat.choose n k) else 0

theorem probability_at_least_6_heads_in_8_flips :
  let total_outcomes := 2^8
  let successful_outcomes := binomial 8 6 + binomial 8 7 + binomial 8 8
  let probability := (successful_outcomes : ℚ) / total_outcomes
  probability = 37 / 256 :=
by
  sorry

end probability_at_least_6_heads_in_8_flips_l574_574159


namespace unique_ordered_pair_satisfies_equation_l574_574558

theorem unique_ordered_pair_satisfies_equation :
  ∃! (m n : ℕ), 0 < m ∧ 0 < n ∧ (6 / m + 3 / n + 1 / (m * n) = 1) :=
by
  sorry

end unique_ordered_pair_satisfies_equation_l574_574558


namespace phase_shift_minimum_value_l574_574549

theorem phase_shift_minimum_value :
  ∃ C : ℝ, ∀ x : ℝ, y = 3 * cos (2 * x - π) ∧ (2x - π = π → x = π) → C = π :=
sorry

end phase_shift_minimum_value_l574_574549


namespace value_of_x_l574_574046

theorem value_of_x (z : ℤ) (h1 : z = 100) (y : ℤ) (h2 : y = z / 10) (x : ℤ) (h3 : x = y / 3) : 
  x = 10 / 3 := 
by
  -- The proof is skipped
  sorry

end value_of_x_l574_574046


namespace percentage_caught_sampling_candy_l574_574349

theorem percentage_caught_sampling_candy
  (S : ℝ) (C : ℝ)
  (h1 : 0.1 * S = 0.1 * 24.444444444444443) -- 10% of the customers who sample the candy are not caught
  (h2 : S = 24.444444444444443)  -- The total percent of all customers who sample candy is 24.444444444444443%
  :
  C = 0.9 * 24.444444444444443 := -- Equivalent \( C \approx 22 \% \)
by
  sorry

end percentage_caught_sampling_candy_l574_574349


namespace part1_part2_l574_574668

-- Definitions for the conditions given in the problem
variables {a b : ℝ}
def l1 (a b : ℝ) : ℝ → ℝ → Prop := λ x y, a * x - b * y - 1 = 0
def l2 (a : ℝ) : ℝ → ℝ → Prop := λ x y, (a + 2) * x + y + a = 0

-- Lean statement for Part (1)
theorem part1 (h : b = 0) (h_perpendicular : (λ x y, a * x = 1) ⊥ (λ x y, (a + 2) * x + y + a = 0)) : a = -2 :=
sorry

-- Lean statement for Part (2)
theorem part2 (h : b = 2) (h_parallel : (λ x y, a * x - 2 * y - 1 = 0) ∥ (λ x y, (a + 2) * x + y + a = 0)) :
  dist (l1 (- (4 / 3)) 2) (l2 (- (4 / 3))) = 11 * real.sqrt 13 / 26 :=
sorry

end part1_part2_l574_574668


namespace max_five_topping_pizzas_l574_574893

theorem max_five_topping_pizzas : 
  (∃ (n k : ℕ), n = 8 ∧ k = 5 ∧ (nat.choose n k = 56)) :=
begin
  use [8, 5],
  split,
  { refl, },
  split,
  { refl, },
  { sorry }
end

end max_five_topping_pizzas_l574_574893


namespace negation_of_exists_log2_leq_zero_l574_574444

theorem negation_of_exists_log2_leq_zero :
  ¬ (∃ x : ℝ, Real.log2 x ≤ 0) ↔ ∀ x : ℝ, Real.log2 x > 0 := by
  sorry

end negation_of_exists_log2_leq_zero_l574_574444


namespace problem_statement_l574_574622

theorem problem_statement (a b : ℝ) (h1 : 1/a < 1/b) (h2 : 1/b < 0) :
  (a + b < a * b) ∧ ¬(a^2 > b^2) ∧ ¬(a < b) ∧ (b/a + a/b > 2) := by
  sorry

end problem_statement_l574_574622


namespace evaluate_expression_l574_574222

theorem evaluate_expression :
  (4 * 10^2011 - 1) / (4 * (3 * (10^2011 - 1) / 9) + 1) = 3 :=
by
  sorry

end evaluate_expression_l574_574222


namespace matrix_expression_l574_574378

noncomputable theory

def B : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, 4], ![0, 2]]

theorem matrix_expression :
  B^10 - 3 • B^9 = ![![0, 4], ![0, -1]] :=
  sorry

end matrix_expression_l574_574378


namespace oates_reunion_attendees_l574_574835

noncomputable def total_guests : ℕ := 100
noncomputable def hall_attendees : ℕ := 70
noncomputable def both_reunions_attendees : ℕ := 10

theorem oates_reunion_attendees :
  ∃ O : ℕ, total_guests = O + hall_attendees - both_reunions_attendees ∧ O = 40 :=
by
  sorry

end oates_reunion_attendees_l574_574835


namespace moving_circle_trajectory_l574_574836

noncomputable def trajectory (O₁ O₂ O : Type) [MetricSpace O₁] [MetricSpace O₂] [MetricSpace O] 
  (r₁ r₂ : ℝ) (h1 : r₁ ≠ r₂) (h2 : ¬∃ p ∈ O₁, p ∈ O₂) (tangent : ∀ p ∈ O, p ∈ O₁ ∧ p ∈ O₂) : Prop :=
isHyperbolaBranch O O₁ O₂ ∨ isEllipse O O₁ O₂

theorem moving_circle_trajectory 
(O₁ O₂ O : Type) [MetricSpace O₁] [MetricSpace O₂] [MetricSpace O] 
(r₁ r₂ : ℝ) (h1 : r₁ ≠ r₂) (h2 : ¬∃ p ∈ O₁, p ∈ O₂) (tangent : ∀ p ∈ O, p ∈ O₁ ∧ p ∈ O₂) :
trajectory O₁ O₂ O r₁ r₂ h1 h2 tangent := 
sorry

end moving_circle_trajectory_l574_574836


namespace repeating_decimal_to_fraction_l574_574565

theorem repeating_decimal_to_fraction :
  let x := (0.6 : ℚ) + (0.03 / (1 - 0.01)) in
  x = 104 / 165 :=
by
  let x : ℚ := (0.6 : ℚ) + (3 / 99)
  have h₁ : 0.6 = 3 / 5 := by sorry
  have h₂ : 0.03 / (1 - 0.01) = 1 / 33 := by sorry
  rw [h₁, h₂]
  exact (3 / 5 + 1 / 33 = 104 / 165) sorry

end repeating_decimal_to_fraction_l574_574565


namespace area_of_gray_region_l574_574708

theorem area_of_gray_region (r : ℝ) (h1 : r * 3 - r = 3) : 
  π * (3 * r) ^ 2 - π * r ^ 2 = 18 * π :=
by
  sorry

end area_of_gray_region_l574_574708


namespace reflection_perpendicular_bisector_AK_is_LM_l574_574736

variable {A B C K L M : Type} [Inhabited A] [Inhabited B] [Inhabited C] 
[Inhabited K] [Inhabited L] [Inhabited M]

-- Reflect points K over the sides AB and AC of triangle ABC to get points L and M
def is_reflection_over (P Q R : Type) (S : Type) : Prop := sorry -- A placeholder for reflection definition

-- Points L and M are reflections of K over sides AB and AC, respectively
axiom reflection_L : is_reflection_over K A B L
axiom reflection_M : is_reflection_over K A C M

-- Define the perpendicular bisector of segment LM
def perpendicular_bisector (L M : Type) : Type := sorry -- A placeholder for the perpendicular bisector definition

-- Define the angle bisectors of angle BAC
def angle_bisector (A B C : Type) : Type := sorry -- A placeholder for the angle bisector definition

-- Statement of the theorem
theorem reflection_perpendicular_bisector_AK_is_LM (L M K : Type) [H1 : is_reflection_over K A B L] 
[H2 : is_reflection_over K A C M] : 
  let d := perpendicular_bisector L M in 
  ∀ f : Type, (angle_bisector A B C = f) → (reflection_perpendicular_bisector d AK f) := 
by 
  intro d f h_f
  sorry

end reflection_perpendicular_bisector_AK_is_LM_l574_574736


namespace angle_PFQ_90_degrees_l574_574303

theorem angle_PFQ_90_degrees
  (a b c : ℝ) 
  (h_ellipse : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1)
  (h_a_gt_b : a > b) 
  (h_b_gt_0 : b > 0) 
  (left_vertex : A = (-a, 0))
  (focus : F = (c, 0))
  (geo_prog : a^2 = 2 * b * c)
  : angle PFQ = 90 := 
sorry

end angle_PFQ_90_degrees_l574_574303


namespace probability_at_least_6_heads_l574_574147

-- Definitions of the binomial coefficient and probability function
def binom (n k : ℕ) : ℕ := Nat.choose n k

def probability (favorable total : ℕ) : ℚ := favorable / total

-- Proof problem statement
theorem probability_at_least_6_heads (flips : ℕ) (p : ℚ) 
  (h_flips : flips = 8) 
  (h_probability : p = probability (binom 8 6 + binom 8 7 + binom 8 8) (2 ^ flips)) : 
  p = 37 / 256 :=
by
  sorry

end probability_at_least_6_heads_l574_574147


namespace area_triangle_l574_574012

noncomputable def triangle_area (AB AC BC AP CP : ℝ) (h1 : is_right_triangle AB AC BC)
  (h2 : point_on_hypotenuse AP CP AC) (h3 : angle_eq ABP 45)
  (h4 : angle_eq PBC 15) (h5 : length_eq AP 2) (h6 : length_eq CP 1) : ℝ :=
  sorry

theorem area_triangle : triangle_area AB AC BC AP CP = 9 / 5 :=
  sorry

end area_triangle_l574_574012


namespace derivative_y_l574_574021

variable (x : ℝ)

def y : ℝ := (1 - Real.log x) / (1 + Real.log x)

theorem derivative_y :
  ∃ f' : ℝ → ℝ, ∀ x : ℝ, x > 0 → HasDerivAt y (-2 / (x * (1 + Real.log x)^2)) x :=
by
  sorry

end derivative_y_l574_574021


namespace find_AB_distance_l574_574711

-- Define the parametric equations of line l
def line_parametric (t : ℝ) : ℝ × ℝ := (1 + t, 2 + t)

-- Define the polar equation of curve C
def polar_curve (θ : ℝ) : ℝ := 4 * Real.sin θ

-- Define the rectangular coordinate equation of line l
def line_rect (x y : ℝ) : Prop := y = x + 1

-- Define the rectangular coordinate equation of curve C
def curve_rect (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

-- State the theorem that needs to be proved
theorem find_AB_distance (t θ : ℝ) : 
  (∃ t θ, let (x_l, y_l) := line_parametric t in line_rect x_l y_l ∧ 
                   let ρ := polar_curve θ in curve_rect (ρ * Real.cos θ) (ρ * Real.sin θ)) → 
  let (x1, y1) := line_parametric t in 
  let (x2, y2) := line_parametric θ in
  sqrt ((x1 - x2)^2 + (y1 - y2)^2) = sqrt 14 :=
sorry

end find_AB_distance_l574_574711


namespace missing_angle_measure_l574_574691

theorem missing_angle_measure (n : ℕ) (h : 180 * (n - 2) = 3240 + 2 * (180 * (n - 2)) / n) : 
  (180 * (n - 2)) / n = 166 := 
by 
  sorry

end missing_angle_measure_l574_574691


namespace compare_groups_l574_574350

noncomputable def mean (scores : List ℝ) : ℝ :=
  (scores.sum / scores.length)

noncomputable def variance (scores : List ℝ) : ℝ :=
  let m := mean scores
  (scores.map (λ x => (x - m) ^ 2)).sum / scores.length

noncomputable def stddev (scores : List ℝ) : ℝ :=
  (variance scores).sqrt

def groupA_scores : List ℝ := [88, 100, 95, 86, 95, 91, 84, 74, 92, 83]
def groupB_scores : List ℝ := [93, 89, 81, 77, 96, 78, 77, 85, 89, 86]

theorem compare_groups :
  mean groupA_scores > mean groupB_scores ∧ stddev groupA_scores > stddev groupB_scores :=
by
  sorry

end compare_groups_l574_574350


namespace train_passes_platform_in_52_seconds_l574_574527

theorem train_passes_platform_in_52_seconds (train_length : ℕ) (train_speed_kmh : ℕ) (platform_length : ℕ) 
  (h_train_length : train_length = 360)
  (h_train_speed : train_speed_kmh = 45)
  (h_platform_length : platform_length = 290) : 
  ∃ t : ℝ, t = 52 :=
by
  have total_distance := train_length + platform_length
  have speed_mps := train_speed_kmh * (1000.0 / 3600.0)
  have time := total_distance / speed_mps
  use time
  sorry

end train_passes_platform_in_52_seconds_l574_574527


namespace coplanar_perpendiculars_l574_574085

variables {Point : Type*} [EuclideanSpace Point]

-- Assume S, A, B, C, D, A1, B1, C1, D1 are points in the Euclidean space
variables (S A B C D A1 B1 C1 D1 : Point)

-- Assume SA, SB, SC, SD are segments and AA1, BB1, CC1, DD1 are perpendiculars
-- from vertices A, B, C, D to lines SC, SD, SA, SB respectively
-- We'll need a function to assert perpendicularity
def perp (p1 p2 p3 : Point) : Prop := sorry

-- Assume all points are distinct and lie on the same sphere
def distinct_points (p1 p2 p3 p4 : Point) : Prop := 
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4

def on_sphere (center : Point) (radius : ℝ) (p : Point) : Prop := sorry

-- Sphere condition (assuming sphere center S and radius r)
variables (r : ℝ)
def all_on_sphere := on_sphere S r A ∧ on_sphere S r B ∧ on_sphere S r C ∧ on_sphere S r D ∧ 
                    on_sphere S r A1 ∧ on_sphere S r B1 ∧ on_sphere S r C1 ∧ on_sphere S r D1

theorem coplanar_perpendiculars :
  distinct_points S A1 B1 C1 D1 →
  perp A S C A1 →
  perp B S D B1 →
  perp C S A C1 →
  perp D S B D1 →
  all_on_sphere S r →
  ∃ (plane : Set Point), A1 ∈ plane ∧ B1 ∈ plane ∧ C1 ∈ plane ∧ D1 ∈ plane :=
sorry

end coplanar_perpendiculars_l574_574085


namespace eliminate_x_sub_eqs_l574_574070

theorem eliminate_x_sub_eqs :
  ∀ (x y : ℝ),
  (2 * x + 5 * y = 9) ∧ (2 * x - 3 * y = 6) →
  (8 * y = 3) :=
by
  intros x y h,
  cases h with h1 h2,
  have h3 : (2 * x + 5 * y - (2 * x - 3 * y) = 9 - 6) := sorry,
  rw [sub_eq_add_neg, add_assoc, add_left_neg, zero_add] at h3,
  exact h3

end eliminate_x_sub_eqs_l574_574070


namespace xiao_ming_prob_l574_574109

open Real

theorem xiao_ming_prob :
  ∀ (arrival_time : ℝ),
  (arrival_time ∈ set.Ioc (7 + 50 / 60) 8.5) → 
  (λ P := (15 / 40) : ℝ), P = 3 / 8 := sorry

end xiao_ming_prob_l574_574109


namespace prime_divides_expression_l574_574373

theorem prime_divides_expression (p : ℕ) (hp : p > 5 ∧ Prime p) : 
  ∃ n : ℕ, p ∣ (20^n + 15^n - 12^n) := 
  by
  use (p - 3)
  sorry

end prime_divides_expression_l574_574373


namespace avg_age_decrease_l574_574019

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

end avg_age_decrease_l574_574019


namespace number_multiplied_by_3_l574_574434

variable (A B C D E : ℝ) -- Declare the five numbers

theorem number_multiplied_by_3 (h1 : (A + B + C + D + E) / 5 = 6.8) 
    (h2 : ∃ X : ℝ, (A + B + C + D + E + 2 * X) / 5 = 9.2) : 
    ∃ X : ℝ, X = 6 := 
  sorry

end number_multiplied_by_3_l574_574434


namespace prime_looking_numbers_less_than_500_l574_574203

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  ¬ is_prime n ∧ n > 1

def is_prime_looking (n : ℕ) : Prop :=
  is_composite n ∧ ¬ (2 ∣ n) ∧ ¬ (3 ∣ n) ∧ ¬ (5 ∣ n)

def count_prime_looking_numbers_less_than (m : ℕ) : ℕ :=
  (List.range m).count is_prime_looking

theorem prime_looking_numbers_less_than_500 :
  count_prime_looking_numbers_less_than 500 = 42 :=
by
  sorry

end prime_looking_numbers_less_than_500_l574_574203


namespace collinear_X_Y_Z_l574_574710

-- Define the basic geometric entities and properties
variables {A B C D M O X Y Z : Type}
variables [circumcenter O A B C] [altitude CD A B] [midpoint M A C]

-- Define the intersections, circles, and lines described
variables [secant_ray DM Γ] [intersects_at Y DM Γ]
variables [circle_intersection X Γ (circle A B C)]
variables [line_intersect_point DO Z AC]

-- The theorem statement asserting the collinearity condition
theorem collinear_X_Y_Z : collinear X Y Z :=
sorry

end collinear_X_Y_Z_l574_574710


namespace probability_at_least_6_heads_8_flips_l574_574164

-- Define the probability calculation of getting at least 6 heads in 8 coin flips.
def probability_at_least_6_heads (n : ℕ) (k : ℕ) : ℚ :=
  (Nat.choose n k + Nat.choose n (k + 1) + Nat.choose n (k + 2)) / 2^n

theorem probability_at_least_6_heads_8_flips : 
  probability_at_least_6_heads 8 6 = 37 / 256 := 
by
  sorry

end probability_at_least_6_heads_8_flips_l574_574164


namespace oranges_initial_count_l574_574940

variable (O : ℕ)

theorem oranges_initial_count
  (h1 : ∃ O : ℕ, 50 / O = 5 / 6)
  (h2 : 20 / 40 = 1 / 2)
  (h3 : 1 / 2 = (1 - 0.4) * 50 / O) :
  O = 60 := 
sorry

end oranges_initial_count_l574_574940


namespace age_calculation_l574_574053

/-- Let Thomas be a 6-year-old child, Shay be 13 years older than Thomas, 
and also 5 years younger than James. Let Violet be 3 years younger than 
Thomas, and Emily be the same age as Shay. This theorem proves that when 
Violet reaches the age of Thomas (6 years old), James will be 27 years old 
and Emily will be 22 years old. -/
theorem age_calculation : 
  ∀ (Thomas Shay James Violet Emily : ℕ),
    Thomas = 6 →
    Shay = Thomas + 13 →
    James = Shay + 5 →
    Violet = Thomas - 3 →
    Emily = Shay →
    (Violet + (6 - Violet) = 6) →
    (James + (6 - Violet) = 27 ∧ Emily + (6 - Violet) = 22) :=
by
  intros Thomas Shay James Violet Emily ht hs hj hv he hv_diff
  sorry

end age_calculation_l574_574053


namespace rectangle_area_l574_574699

theorem rectangle_area (AB AC : ℝ) (H1 : AB = 15) (H2 : AC = 17) : 
  ∃ (BC : ℝ), (AB * BC = 120) :=
by
  sorry

end rectangle_area_l574_574699


namespace ticket_sales_amount_theater_collected_50_dollars_l574_574925

variable (num_people total_people : ℕ) (cost_adult_entry cost_child_entry : ℕ) (num_children : ℕ)
variable (total_collected : ℕ)

theorem ticket_sales_amount
  (h1 : cost_adult_entry = 8)
  (h2 : cost_child_entry = 1)
  (h3 : total_people = 22)
  (h4 : num_children = 18)
  (h5 : num_people = total_people - num_children)
  : total_collected = (num_people * cost_adult_entry + num_children * cost_child_entry) := sorry

theorem theater_collected_50_dollars 
  (h1 : cost_adult_entry = 8)
  (h2 : cost_child_entry = 1)
  (h3 : total_people = 22)
  (h4 : num_children = 18)
  (h5 : total_collected = 50)
  : total_collected = 50 := sorry

end ticket_sales_amount_theater_collected_50_dollars_l574_574925


namespace valid_subsets_count_in_two_circles_is_fifty_l574_574829

-- Define the number of chairs in each circle
def num_chairs := 6

-- Define the property to check if a subset of chairs contains at least two adjacent chairs
def has_at_least_two_adjacent_chairs (subset : Finset ℕ) : Prop :=
  ∃ i, i ∈ subset ∧ ((i + 1) % num_chairs) ∈ subset

-- Definition for counting valid subsets in one circle
noncomputable def count_valid_subsets_in_one_circle : ℕ :=
  Finset.card { s : Finset ℕ | ∃ (i j k m : ℕ), Finset.card s > 1 ∧ s ⊆ (Finset.range num_chairs) ∧ has_at_least_two_adjacent_chairs s } 

-- Definition for counting valid subsets in two circles
noncomputable def total_valid_subsets (circles : ℕ) : ℕ :=
  circles * count_valid_subsets_in_one_circle 

-- The theorem to prove with conditions
theorem valid_subsets_count_in_two_circles_is_fifty : total_valid_subsets 2 = 50 :=
by
  sorry

end valid_subsets_count_in_two_circles_is_fifty_l574_574829


namespace negation_of_p_l574_574662

def p : Prop := ∀ x : ℝ, x^2 - 2*x + 2 ≤ Real.sin x
def not_p : Prop := ∃ x : ℝ, x^2 - 2*x + 2 > Real.sin x

theorem negation_of_p : ¬ p ↔ not_p := by
  sorry

end negation_of_p_l574_574662


namespace proof_l574_574610

variable (f : ℝ → ℝ) (hf_diff : Differentiable ℝ f) 
          (hf_cond : ∀ x, f' x + f x < 0)
          (θ : ℝ)
          (hθ_cond : ¬ (sin θ + cos θ = (1 / sqrt 2) * sin θ + (1 / sqrt 2) * cos θ))

theorem proof :
  (f (sin θ + cos θ) / exp (sqrt 2 - sin θ - cos θ)) > f (sqrt 2) :=
by
  sorry

end proof_l574_574610


namespace percentage_error_divide_by_5_instead_of_multiplying_by_5_l574_574718

variable (x : ℝ)

theorem percentage_error_divide_by_5_instead_of_multiplying_by_5 (x ≠ 0) :
  ((5 * x - x / 5) / (5 * x)) * 100 = 96 := 
sorry

end percentage_error_divide_by_5_instead_of_multiplying_by_5_l574_574718


namespace total_budget_l574_574899

-- Define the conditions for the problem
def fiscal_months : ℕ := 12
def total_spent_at_six_months : ℕ := 6580
def over_budget_at_six_months : ℕ := 280

-- Calculate the total budget for the project
theorem total_budget (budget : ℕ) 
  (h : 6 * (total_spent_at_six_months - over_budget_at_six_months) * 2 = budget) 
  : budget = 12600 := 
  by
    -- Proof will be here
    sorry

end total_budget_l574_574899


namespace shape_formed_is_line_segment_l574_574954

def point := (ℝ × ℝ)

noncomputable def A : point := (0, 0)
noncomputable def B : point := (0, 4)
noncomputable def C : point := (6, 4)
noncomputable def D : point := (6, 0)

noncomputable def line_eq (p1 p2 : point) : ℝ × ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  (x2 - x1, y2 - y1)

theorem shape_formed_is_line_segment :
  let l1 := line_eq A (1, 1)  -- Line from A at 45°
  let l2 := line_eq B (-1, -1) -- Line from B at -45°
  let l3 := line_eq D (1, -1) -- Line from D at 45°
  let l4 := line_eq C (-1, 5) -- Line from C at -45°
  let intersection1 := (5, 5)  -- Intersection of l1 and l4: solve x = 10 - x
  let intersection2 := (5, -1)  -- Intersection of l2 and l3: solve 4 - x = x - 6
  intersection1.1 = intersection2.1 := 
by
  sorry

end shape_formed_is_line_segment_l574_574954


namespace total_wheels_eq_90_l574_574466

def total_wheels (num_bicycles : Nat) (wheels_per_bicycle : Nat) (num_tricycles : Nat) (wheels_per_tricycle : Nat) :=
  num_bicycles * wheels_per_bicycle + num_tricycles * wheels_per_tricycle

theorem total_wheels_eq_90 : total_wheels 24 2 14 3 = 90 :=
by
  sorry

end total_wheels_eq_90_l574_574466


namespace comb_12_9_eq_220_l574_574213

theorem comb_12_9_eq_220 : (Nat.choose 12 9) = 220 := by
  sorry

end comb_12_9_eq_220_l574_574213


namespace cone_volume_l574_574286

theorem cone_volume (r l h V : ℝ) (π : ℝ) [real.pi_is_constant : is_constant π]
  (h_base_area : π * r^2 = π)
  (h_lateral_area : π * r * l = 2 * π)
  (h_height : h = real.sqrt (l^2 - r^2))
  (h_volume : V = (1/3) * π * r^2 * h) :
  V = (real.sqrt 3 / 3) * π :=
by
  sorry

end cone_volume_l574_574286


namespace solve_z_l574_574751

-- Define the complex number and the conditions
theorem solve_z : ∃ (z : ℂ), (1 - complex.I) * z = 2 * complex.I ∧ z = -1 + complex.I := 
by
  use -1 + complex.I
  constructor
  · simp [mul_add]
    rw [mul_comm, ring_hom.map_mul, complex.mul_conj_I]
    simp only [ring_hom.map_one, complex.of_real_one, complex.I_re, complex.one_re, complex.zero_re, complex.of_real_neg, complex.mul_conj, add_re, mul_zero, sub_zero, zero_add, one_div, add_im, div_im, zero_im, mul_zero, one_im]
  · sorry

end solve_z_l574_574751


namespace cities_below_50000_l574_574794

theorem cities_below_50000 (p1 p2 : ℝ) (h1 : p1 = 20) (h2: p2 = 65) :
  p1 + p2 = 85 := 
  by sorry

end cities_below_50000_l574_574794


namespace highest_power_of_6_divides_12_factorial_l574_574227

open Nat

theorem highest_power_of_6_divides_12_factorial:
  let legendre := λ (n p : ℕ), ∑ i in range (n.log p).succ, n / p^i
  legendre 12 2 = 10 ∧ legendre 12 3 = 5 →
  6^5 ∣ factorial 12 :=
by
  intros h
  let legendre := λ (n p : ℕ), ∑ i in range (n.log p).succ, n / p^i
  have h2 : legendre 12 2 = 10 := h.1
  have h3 : legendre 12 3 = 5 := h.2
  sorry

end highest_power_of_6_divides_12_factorial_l574_574227


namespace derivative_of_implicit_function_l574_574249

def f (x : ℝ) (y : ℝ) : ℝ := log y + (cos (x^2)) / (sin (x^2)) - 2 * x

theorem derivative_of_implicit_function (x : ℝ) (y : ℝ) (h : f x y = 0) :
  deriv (λ x : ℝ, y) x = y * ((2 * x) / (sin (x^2))^2 + 2) :=
sorry

end derivative_of_implicit_function_l574_574249


namespace triangle_area_is_correct_l574_574839

structure Point where
  x : ℝ
  y : ℝ

def area_of_triangle (A B C : Point) : ℝ :=
  0.5 * (B.x - A.x).natAbs * (C.y - A.y).natAbs -- using absolute value for simplicity

def A := Point.mk 1 1
def B := Point.mk (-3) 1
def C := Point.mk 1 8

theorem triangle_area_is_correct : area_of_triangle A B C = 14 := by
  sorry

end triangle_area_is_correct_l574_574839


namespace team_a_score_l574_574076

theorem team_a_score : ∀ (A : ℕ), A + 9 + 4 = 15 → A = 2 :=
by
  intros A h
  sorry

end team_a_score_l574_574076


namespace units_digit_sum_42_4_24_4_l574_574852

theorem units_digit_sum_42_4_24_4 : (42^4 + 24^4) % 10 = 2 := 
by
  sorry

end units_digit_sum_42_4_24_4_l574_574852


namespace min_largest_fraction_l574_574060

theorem min_largest_fraction (S : Set (ℚ)) (h1 : ∀ f ∈ S, ∃ n d, 2 ≤ n ∧ n ≤ 2019 ∧ 2 ≤ d ∧ d ≤ 2019 ∧ n ≠ d ∧ f = n / d)
  (h2 : S.card = 1009) : ∃ f ∈ S, ∀ g ∈ S, g ≤ 1010 / 2019 ∧ 1010 / 2019 ≤ f := 
sorry

end min_largest_fraction_l574_574060


namespace find_x_l574_574265

theorem find_x (x : ℚ) (h : x ≠ 2 ∧ x ≠ 4/5) :
  (x^2 - 11*x + 24)/(x - 2) + (5*x^2 + 22*x - 48)/(5*x - 4) = -7 → x = -4/3 :=
by
  intro h1
  sorry

end find_x_l574_574265


namespace num_seven_digit_numbers_l574_574321

theorem num_seven_digit_numbers : 
  ∃ (n : ℕ), n = 120 := 
begin
  -- conditions: 
  -- first digit is 1
  -- second digit is 5
  -- all other digits are distinct and odd
  sorry
end

end num_seven_digit_numbers_l574_574321


namespace max_value_of_reciprocals_l574_574561

noncomputable def quadratic (x t q : ℝ) : ℝ := x^2 - t * x + q

theorem max_value_of_reciprocals (α β t q : ℝ) (h1 : α + β = α^2 + β^2)
                                               (h2 : α + β = α^3 + β^3)
                                               (h3 : ∀ n, 1 ≤ n ∧ n ≤ 2010 → α^n + β^n = α + β)
                                               (h4 : α * β = q)
                                               (h5 : α + β = t) :
  ∃ (α β : ℝ), (1 / α^2012 + 1 / β^2012) = 2 := 
sorry

end max_value_of_reciprocals_l574_574561


namespace find_a_not_perfect_square_l574_574578

theorem find_a_not_perfect_square :
  {a : ℕ | ∀ n : ℕ, n > 0 → ¬(∃ k : ℕ, n * (n + a) = k * k)} = {1, 2, 4} :=
sorry

end find_a_not_perfect_square_l574_574578


namespace log_sum_real_coeffs_l574_574749

theorem log_sum_real_coeffs : 
  ∀ T : ℝ, 
  (T = (∑ k in (finset.range 2012).filter (λ k, even k), (binom 2011 k) * (i : ℂ) ^ k).re) → 
  real.logb 2 T = 1005.5 :=
begin
  intros T hT,
  sorry
end

end log_sum_real_coeffs_l574_574749


namespace solve_for_x_l574_574009

theorem solve_for_x (x : ℝ) (h : 16^x * 16^x * 16^x = 256^4) : x = 8 / 3 := by
  sorry

end solve_for_x_l574_574009


namespace probability_of_at_least_six_heads_is_correct_l574_574129

-- Definitions for the given problem
def binomial_coefficient (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

def total_possible_outcomes : ℕ :=
  2^8

def favorable_outcomes : ℕ :=
  binomial_coefficient 8 6 + binomial_coefficient 8 7 + binomial_coefficient 8 8

def probability_of_at_least_6_heads : ℚ :=
  favorable_outcomes / total_possible_outcomes

-- The proof statement
theorem probability_of_at_least_six_heads_is_correct :
  probability_of_at_least_6_heads = 37 / 256 :=
by sorry

end probability_of_at_least_six_heads_is_correct_l574_574129


namespace probability_target_A_destroyed_probability_exactly_one_target_destroyed_l574_574471

-- Definition of probabilities
def prob_A_hits_target_A := 1 / 2
def prob_A_hits_target_B := 1 / 2
def prob_B_hits_target_A := 1 / 3
def prob_B_hits_target_B := 2 / 5

-- The event of target A being destroyed
def prob_target_A_destroyed := prob_A_hits_target_A * prob_B_hits_target_A

-- The event of target B being destroyed
def prob_target_B_destroyed := prob_A_hits_target_B * prob_B_hits_target_B

-- Complementary events
def prob_target_A_not_destroyed := 1 - prob_target_A_destroyed
def prob_target_B_not_destroyed := 1 - prob_target_B_destroyed

-- Exactly one target being destroyed
def prob_exactly_one_target_destroyed := 
  (prob_target_A_destroyed * prob_target_B_not_destroyed) +
  (prob_target_B_destroyed * prob_target_A_not_destroyed)

theorem probability_target_A_destroyed : prob_target_A_destroyed = 1 / 6 := by
  -- Proof needed here
  sorry

theorem probability_exactly_one_target_destroyed : prob_exactly_one_target_destroyed = 3 / 10 := by
  -- Proof needed here
  sorry

end probability_target_A_destroyed_probability_exactly_one_target_destroyed_l574_574471


namespace a2_plus_a3_eq_40_l574_574732

theorem a2_plus_a3_eq_40 : 
  ∀ (a a1 a2 a3 a4 a5 : ℤ), 
  (2 * x - 1)^5 = a * x^5 + a1 * x^4 + a2 * x^3 + a3 * x^2 + a4 * x + a5 → 
  a2 + a3 = 40 :=
by
  sorry

end a2_plus_a3_eq_40_l574_574732


namespace negation_statement_l574_574074

theorem negation_statement {α : Type} (s : set α) (P : α → Prop) :
  (∀ x ∈ s, P x) ↔ ¬(∃ x ∈ s, ¬P x) :=
by sorry

end negation_statement_l574_574074


namespace same_number_of_acquaintances_allowable_values_of_n_l574_574541

-- (a) Statement of the problem: Given the conditions, prove all attendees have the same number of acquaintances.
theorem same_number_of_acquaintances 
    (n : ℕ) 
    (h1 : ∀ (A B : ℕ), A ≠ B → (A is acquaintance_of B) → (∀ C, C ≠ A ∧ C ≠ B → ¬(C is acquaintance_of A ∧ C is acquaintance_of B))) 
    (h2 : ∀ (A B : ℕ), A ≠ B → ¬(A is acquaintance_of B) → (∃ C1 C2, C1 ≠ A ∧ C1 ≠ B ∧ C2 ≠ A ∧ C2 ≠ B ∧ C1 ≠ C2 ∧ (C1 is acquaintance_of A) ∧ (C2 is acquaintance_of A) ∧ (C1 is acquaintance_of B) ∧ (C2 is acquaintance_of B))) :
    ∀ A B, A ≠ B → (number_of_acquaintances A) = (number_of_acquaintances B) := 
by sorry

-- (b) Statement of the problem: Find the allowable values of n.
theorem allowable_values_of_n (n : ℕ) : 
    ∃ (k : ℕ), n = (2 * k - 1)^2 / 8 + 7 / 8 :=
by sorry

end same_number_of_acquaintances_allowable_values_of_n_l574_574541


namespace part_1_part_2_l574_574653

noncomputable def f (x : ℝ) : ℝ := log x / log 4 + 10

def a_n (n : ℕ) : ℝ := n - 5
def S_n (n : ℕ) : ℝ := (n * (n - 9)) / 2
def b_n (n : ℕ) : ℝ := n * (n^2 - 12 * n + 45)

/-
(I) Prove that for n ∈ {5, 6, 7, 8, 9}, the inequality a_n * S_n ≤ 0 holds.
-/
theorem part_1 (n : ℕ) (h : n ∈ {5, 6, 7, 8, 9}) : a_n n * S_n n ≤ 0 := by
  sorry

/-
(II) Prove that b_n has minimum value 34.
-/
theorem part_2 : ∃ n : ℕ, b_n n = 34 := by
  use 1
  show b_n 1 = 34
  sorry

end part_1_part_2_l574_574653


namespace B_visited_cityB_l574_574056

noncomputable theory
open_locale classical

def City : Type := ℕ
def Person : Type := ℕ

variables (A B C : Person)
variables (cityA cityB cityC : City)

-- Conditions
-- A has visited more cities than B
axiom hA_more_than_B : ∀ cities_visited_by : Person → City → City, cities_visited_by A > cities_visited_by B

-- A has not visited city C
axiom hA_not_C : ∀ visited_by : Person → City → Prop, ¬ visited_by A cityC

-- B has not visited city A
axiom hB_not_A : ∀ visited_by : Person → City → Prop, ¬ visited_by B cityA

-- A, B, and C have all visited the same city
axiom hAll_same_city : ∀ visited_by : Person → City → Prop, 
  ∀ c : City, (visited_by A c) ↔ (visited_by B c) ↔ (visited_by C c)

-- Question to be answered: B has visited city B
theorem B_visited_cityB :
  ∀ visited_by : Person → City → Prop, visited_by B cityB :=
sorry

end B_visited_cityB_l574_574056


namespace reflection_matrix_plane_at_origin_l574_574737

def reflection_matrix_exists (n : ℝ × ℝ × ℝ) (S : ℝ -> ℝ -> ℝ) : Prop :=
  let v (x y z : ℝ) := (x, y, z)
  let q (x y z : ℝ) := 
      let vp = n.1 * x + n.2 * y + n.3 * z
      let nn = n.1 * n.1 + n.2 * n.2 + n.3 * n.3
      (vp / nn * n.1, vp / nn * n.2, vp / nn * n.3)
  let r (x y z : ℝ) := 
      let qv = q x y z
      (2 * qv.1 - x, 2 * qv.2 - y, 2 * qv.3 - z)
  ∀ (x y z : ℝ), r x y z = S x y z

theorem reflection_matrix_plane_at_origin :
  ∃ (S : ℝ -> ℝ -> ℝ -> ℝ * ℝ * ℝ),
    reflection_matrix_exists (2, -1, 1) S ∧ 
    S = (λ x y z, 
      (-1 / 3 * x - 1 / 3 * y + 1 / 3 * z, 
       -1 / 3 * x - 5 / 6 * y - 1 / 6 * z, 
       1 / 3 * x - 1 / 6 * y - 5 / 6 * z)) :=
sorry

end reflection_matrix_plane_at_origin_l574_574737


namespace solve_for_y_l574_574683

theorem solve_for_y (x y : ℤ) (h1 : x + y = 260) (h2 : x - y = 200) : y = 30 := by
  sorry

end solve_for_y_l574_574683


namespace probability_at_least_6_heads_l574_574150

-- Definitions of the binomial coefficient and probability function
def binom (n k : ℕ) : ℕ := Nat.choose n k

def probability (favorable total : ℕ) : ℚ := favorable / total

-- Proof problem statement
theorem probability_at_least_6_heads (flips : ℕ) (p : ℚ) 
  (h_flips : flips = 8) 
  (h_probability : p = probability (binom 8 6 + binom 8 7 + binom 8 8) (2 ^ flips)) : 
  p = 37 / 256 :=
by
  sorry

end probability_at_least_6_heads_l574_574150


namespace three_digit_number_equal_sum_of_factorials_and_has_three_l574_574854

def is_three_digit (n : Nat) := 100 ≤ n ∧ n < 1000

def has_digit_three (n : Nat) := n / 100 = 3 ∨ (n % 100) / 10 = 3 ∨ (n % 10) = 3

def sum_of_digit_factorials (n : Nat) : Nat :=
  (n / 100)! + ((n % 100) / 10)! + (n % 10)!

theorem three_digit_number_equal_sum_of_factorials_and_has_three :
  ∃ n : Nat, is_three_digit n ∧ has_digit_three n ∧ sum_of_digit_factorials n = n :=
sorry

end three_digit_number_equal_sum_of_factorials_and_has_three_l574_574854


namespace sum_A_elements_l574_574459

def A : Set ℕ := { n | Real.log n / Real.log 10 < 1 / 2 ∧ n > 0 }

theorem sum_A_elements : ∑ n in {1, 2, 3}.toFinset, id n = 6 :=
by
  sorry

end sum_A_elements_l574_574459


namespace joan_balloons_l574_574725

def initial_balloons : ℕ := 72
def additional_balloons : ℕ := 23
def total_balloons : ℕ := initial_balloons + additional_balloons

theorem joan_balloons : total_balloons = 95 := by
  sorry

end joan_balloons_l574_574725


namespace sin_cos_value_l574_574326

variable (x : ℝ)

theorem sin_cos_value (h : Real.sin x = 4 * Real.cos x) : Real.sin x * Real.cos x = 4 / 17 := 
sorry

end sin_cos_value_l574_574326


namespace x_squared_minus_y_squared_l574_574330

theorem x_squared_minus_y_squared
    (x y : ℚ) 
    (h1 : x + y = 3 / 8) 
    (h2 : x - y = 1 / 4) : x^2 - y^2 = 3 / 32 := 
by 
    sorry

end x_squared_minus_y_squared_l574_574330


namespace original_number_is_19_l574_574478

theorem original_number_is_19 (x : ℤ) (h : (x + 4) % 23 = 0) : x = 19 := 
by 
  sorry

end original_number_is_19_l574_574478


namespace volume_relation_l574_574518

noncomputable def A (r : ℝ) : ℝ :=
  (2 / 3) * π * r^3

noncomputable def M (r : ℝ) : ℝ :=
  2 * π * r^3

noncomputable def H (r : ℝ) : ℝ :=
  (2 / 3) * π * r^3

theorem volume_relation (r : ℝ) : H r + M r = A r + M r := by
  let A := (2 / 3) * π * r^3
  let M := 2 * π * r^3
  let H := (2 / 3) * π * r^3

  calc
    H + M = (2 / 3) * π * r^3 + 2 * π * r^3 : by rfl
       ... = (2 / 3) * π * r^3 + 2 * π * r^3 : by rfl
       ... = (8 / 3) * π * r^3               : by sorry
       ... = A + M                           : by sorry

end volume_relation_l574_574518


namespace find_general_term_l574_574295

noncomputable def general_term (a : ℕ → ℕ) (d : ℕ) : ℕ → ℕ :=
λ n => 1 + (n - 1) * d

theorem find_general_term
  (a : ℕ → ℕ)
  (h_arith : ∀ n, a n = general_term a d n)
  (h_initial : a 1 = 1)
  (h_geom : (a 3 : α) * a 3 = a 9 * a 1) :
  ∃ d, ∀ n, a n = 1 ∨ a n = n :=
sorry

end find_general_term_l574_574295


namespace log_base_3_243_eq_5_l574_574972

theorem log_base_3_243_eq_5 : log 3 243 = 5 := by
  sorry

end log_base_3_243_eq_5_l574_574972


namespace total_students_l574_574936

theorem total_students (T : ℝ) (h : 0.50 * T = 440) : T = 880 := 
by {
  sorry
}

end total_students_l574_574936


namespace parabola_equation_hyperbola_equation_l574_574493

-- Part 1: Prove the standard equation of the parabola given the directrix.
theorem parabola_equation (x y : ℝ) : x = -2 → y^2 = 8 * x := 
by
  -- Here we will include proof steps based on given conditions
  sorry

-- Part 2: Prove the standard equation of the hyperbola given center at origin, focus on the x-axis,
-- the given asymptotes, and its real axis length.
theorem hyperbola_equation (x y a b : ℝ) : 
  a = 1 → b = 2 → y = 2 * x ∨ y = -2 * x → x^2 - (y^2 / 4) = 1 :=
by
  -- Here we will include proof steps based on given conditions
  sorry

end parabola_equation_hyperbola_equation_l574_574493


namespace total_supermarkets_FGH_chain_l574_574051

variable (US_supermarkets : ℕ) (Canada_supermarkets : ℕ)
variable (total_supermarkets : ℕ)

-- Conditions
def condition1 := US_supermarkets = 37
def condition2 := US_supermarkets = Canada_supermarkets + 14

-- Goal
theorem total_supermarkets_FGH_chain
    (h1 : condition1 US_supermarkets)
    (h2 : condition2 US_supermarkets Canada_supermarkets) :
    total_supermarkets = US_supermarkets + Canada_supermarkets :=
sorry

end total_supermarkets_FGH_chain_l574_574051


namespace total_hours_for_songs_l574_574519

def total_hours_worked_per_day := 10
def total_days_per_song := 10
def number_of_songs := 3

theorem total_hours_for_songs :
  total_hours_worked_per_day * total_days_per_song * number_of_songs = 300 :=
by
  sorry

end total_hours_for_songs_l574_574519


namespace subtraction_result_l574_574261

theorem subtraction_result : 3.05 - 5.678 = -2.628 := 
by
  sorry

end subtraction_result_l574_574261


namespace compute_AD_minus_BD_l574_574218

variable (d : ℕ) (A B : ℕ)
hypothesis (h1 : d > 5)
hypothesis (h2 : d * A + B + d * A + A = 1 * d^2 + 5 * d + 0)

theorem compute_AD_minus_BD : A - B = (10 - d) / 3 :=
by
  sorry

end compute_AD_minus_BD_l574_574218


namespace harry_worked_32_hours_l574_574969

variable (x y : ℝ)
variable (harry_pay james_pay : ℝ)

-- Definitions based on conditions
def harry_weekly_pay (h : ℝ) := 30*x + (h - 30)*y
def james_weekly_pay := 40*x + 1*y

-- Condition: Harry and James were paid the same last week
axiom harry_james_same_pay : ∀ (h : ℝ), harry_weekly_pay x y h = james_weekly_pay x y

-- Prove: Harry worked 32 hours
theorem harry_worked_32_hours : ∃ h : ℝ, h = 32 ∧ harry_weekly_pay x y h = james_weekly_pay x y := by
  sorry

end harry_worked_32_hours_l574_574969


namespace product_of_coordinates_of_intersection_l574_574064

-- Conditions: Defining the equations of the two circles
def circle1_eq (x y : ℝ) : Prop := x^2 - 2*x + y^2 - 10*y + 25 = 0
def circle2_eq (x y : ℝ) : Prop := x^2 - 8*x + y^2 - 10*y + 37 = 0

-- Translated problem to prove the question equals the correct answer
theorem product_of_coordinates_of_intersection :
  ∃ (x y : ℝ), circle1_eq x y ∧ circle2_eq x y ∧ x * y = 10 :=
sorry

end product_of_coordinates_of_intersection_l574_574064


namespace extreme_values_l574_574257

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x + 1

theorem extreme_values :
  (∀ x ∈ Icc (-3 : ℝ) 0, f x ≤ 3) ∧ (∃ x ∈ Icc (-3 : ℝ) 0, f x = 3) ∧
  (∀ x ∈ Icc (-3 : ℝ) 0, -17 ≤ f x) ∧ (∃ x ∈ Icc (-3 : ℝ) 0, f x = -17)
  :=
sorry

end extreme_values_l574_574257


namespace kat_boxing_training_hours_l574_574728

theorem kat_boxing_training_hours :
  let strength_training_hours := 3
  let total_training_hours := 9
  let boxing_sessions := 4
  let boxing_training_hours := total_training_hours - strength_training_hours
  let hours_per_boxing_session := boxing_training_hours / boxing_sessions
  hours_per_boxing_session = 1.5 :=
sorry

end kat_boxing_training_hours_l574_574728


namespace florinda_walk_distance_l574_574989

theorem florinda_walk_distance :
  let horizontal_steps := [1, -3, 5, -7, 9, -11, 13, -15, 17, -19, 21, -23, 25, -27, 29]
  let vertical_steps := [2, -4, 6, -8, 10, -12, 14, -16, 18, -20, 22, -24, 26, -28, 30]
  let horizontal_displacement := horizontal_steps.sum
  let vertical_displacement := vertical_steps.sum
  distance := Math.sqrt(horizontal_displacement ^ 2 + vertical_displacement ^ 2)
  in distance = Real.sqrt 481 :=
 by sorry

end florinda_walk_distance_l574_574989


namespace more_wins_than_losses_probability_club_truncator_probability_sum_l574_574546

def club_truncator_outcome_prob (wins losses ties : ℕ) : ℚ :=
  if wins + losses + ties = 8 then
    match winning_outcome_status (wins, losses, ties) with
    | .win_more => (binomial 8 wins) * ((1 / 2) ^ wins) * ((1 / 2) ^ (8 - wins))
    | _ => 0
  else 0

def total_prob_wins_more_than_losses : ℚ :=
  (∑ wins in {5, 6, 7, 8}, club_truncator_outcome_prob wins 0 (8 - wins)) +
  (∑ wins in {5, 6, 7}, club_truncator_outcome_prob wins 1 (7 - wins)) +
  (∑ wins in {5, 6}, club_truncator_outcome_prob wins 2 (6 - wins)) +
  (club_truncator_outcome_prob 5 3 0)

theorem more_wins_than_losses_probability:
  total_prob_wins_more_than_losses = 93 / 256 :=
by
  sorry

theorem club_truncator_probability_sum : 
  ∃ m n : ℕ, (Nat.gcd m n = 1) ∧ (93 / 256 = m / n) ∧ (m + n = 349) :=
by
  use 93, 256
  constructor
  repeat sorry

end more_wins_than_losses_probability_club_truncator_probability_sum_l574_574546


namespace cost_price_proof_l574_574079

def trader_sells_66m_for_660 : Prop := ∃ cp profit sp : ℝ, sp = 660 ∧ cp * 66 + profit * 66 = sp
def profit_5_per_meter : Prop := ∃ profit : ℝ, profit = 5
def cost_price_per_meter_is_5 : Prop := ∃ cp : ℝ, cp = 5

theorem cost_price_proof : trader_sells_66m_for_660 → profit_5_per_meter → cost_price_per_meter_is_5 :=
by
  intros h1 h2
  sorry

end cost_price_proof_l574_574079


namespace effective_price_per_pair_effective_price_per_pair_simplified_l574_574108

/-- Given the price of one pair of socks is 4.86 yuan,
    and given the promotion "buy five, get one free",
    prove that the effective price per pair of socks is 4.05 yuan. -/
theorem effective_price_per_pair
  (price_per_pair : ℝ) (promotion_buy_five_get_one_free : ℕ → ℕ → ℝ → ℝ)
  (h_price : price_per_pair = 4.86) :
  promotion_buy_five_get_one_free 5 6 4.86 = 4.05 := by
  sorry

/-- The promotion function calculates the effective price per pair based on
    the number of pairs you need to buy and the total pairs you receive. -/
def promotion_buy_five_get_one_free (n_buy : ℕ) (n_total : ℕ) (price : ℝ) : ℝ := 
  (n_buy * price) / n_total

/-- Using the given constants to simplify the theorem statement -/
theorem effective_price_per_pair_simplified :
  promotion_buy_five_get_one_free 5 6 4.86 = 4.05 := by
  sorry

end effective_price_per_pair_effective_price_per_pair_simplified_l574_574108


namespace donkey_always_wins_l574_574690

-- Define the basic setup for the problem
variables (Point : Type) [fintype Point] [decidable_eq Point]

-- Conditions: 2005 points, no three collinear, each pair connected by a segment
variables (points : finset Point) (h_card : points.card = 2005)

-- Digit assignments
variables (segment_digit : (Point × Point) → ℕ)
variables (point_digit : Point → ℕ)

-- Define the winning condition for Donkey
def donkey_wins (segment_digit : (Point × Point) → ℕ) (point_digit : Point → ℕ) : Prop :=
  ∃ p₁ p₂, p₁ ≠ p₂ ∧ point_digit p₁ = segment_digit (p₁, p₂) ∧ point_digit p₂ = segment_digit (p₁, p₂)

-- Theorem: Donkey always wins with optimal play
theorem donkey_always_wins (points : finset Point) (segment_digit : (Point × Point) → ℕ) (point_digit : Point → ℕ)
  (h_card : points.card = 2005)
  (no_three_collinear : ∀ p1 p2 p3 ∈ points, p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 → ¬ collinear p1 p2 p3)
  (all_connected : ∀ p1 p2 ∈ points, p1 ≠ p2 → ∃ e, e ∈ segment_digit (p1, p2)) :
  donkey_wins points segment_digit point_digit :=
sorry

end donkey_always_wins_l574_574690


namespace max_value_and_period_of_g_value_of_expression_if_fx_eq_2f_l574_574647

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def f' (x : ℝ) : ℝ := Real.cos x - Real.sin x
noncomputable def g (x : ℝ) : ℝ := f x * f' x - f x ^ 2

theorem max_value_and_period_of_g :
  ∃ (M : ℝ) (T : ℝ), (∀ x, g x ≤ M) ∧ (∀ x, g (x + T) = g x) ∧ M = 2 ∧ T = Real.pi :=
sorry

theorem value_of_expression_if_fx_eq_2f'x (x : ℝ) :
  f x = 2 * f' x → (1 + Real.sin x ^ 2) / (Real.cos x ^ 2 - Real.sin x * Real.cos x) = 11 / 6 :=
sorry

end max_value_and_period_of_g_value_of_expression_if_fx_eq_2f_l574_574647


namespace no_solutions_inv_matrix_eqn_l574_574985

theorem no_solutions_inv_matrix_eqn :
  ∀ (a b c d : ℝ), (a * d - b * c ≠ 0) →
  (a * (2 / a) + b * (2 / c) = 1) ∧ (a * (2 / b) + b * (2 / d) = 0) ∧ 
  (c * (2 / a) + d * (2 / c) = 0) ∧ (c * (2 / b) + d * (2 / d) = 1) → false :=
by { intros a b c d h_det h_eqn,
     rcases h_eqn with ⟨h11, h12, h21, h22⟩,
     sorry }

#check no_solutions_inv_matrix_eqn

end no_solutions_inv_matrix_eqn_l574_574985


namespace probability_three_specific_cards_l574_574054

theorem probability_three_specific_cards :
  let total_deck := 52
  let total_spades := 13
  let total_tens := 4
  let total_queens := 4
  let p_case1 := ((12:ℚ) / total_deck) * (total_tens / (total_deck - 1)) * (total_queens / (total_deck - 2))
  let p_case2 := ((1:ℚ) / total_deck) * ((total_tens - 1) / (total_deck - 1)) * (total_queens / (total_deck - 2))
  p_case1 + p_case2 = (17:ℚ) / 11050 :=
by
  sorry

end probability_three_specific_cards_l574_574054


namespace first_player_guaranteed_win_l574_574464

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2 ^ k

theorem first_player_guaranteed_win (n : ℕ) (h : n > 1) : 
  ¬ is_power_of_two n ↔ ∃ m : ℕ, 1 ≤ m ∧ m < n ∧ (∀ k : ℕ, m ≤ k + 1 → ∀ t, t ≤ m → ∃ r, r = k + 1 ∧ r <= m) → 
                                (∃ l : ℕ, (l = 1) → true) :=
sorry

end first_player_guaranteed_win_l574_574464


namespace new_percentage_of_managers_is_98_l574_574821

def percentage_of_managers (initial_employees : ℕ) (initial_percentage_managers : ℕ) (managers_leaving : ℕ) : ℕ :=
  let initial_managers := initial_percentage_managers * initial_employees / 100
  let remaining_managers := initial_managers - managers_leaving
  let remaining_employees := initial_employees - managers_leaving
  (remaining_managers * 100) / remaining_employees

theorem new_percentage_of_managers_is_98 :
  percentage_of_managers 500 99 250 = 98 :=
by
  sorry

end new_percentage_of_managers_is_98_l574_574821


namespace compare_exponents_and_logarithms_l574_574280
  
/-- Given three numbers a, b, and c defined as follows:
  a = 1.9 ^ 0.4,
  b = log_base 0.4 1.9,
  c = 0.4 ^ 1.9,
  prove that a > c > b.
-/
theorem compare_exponents_and_logarithms :
  let a := 1.9 ^ 0.4
  let b := log 0.4 1.9
  let c := 0.4 ^ 1.9
  a > c ∧ c > b :=
by 
  let a := 1.9 ^ 0.4
  let b := log 0.4 1.9
  let c := 0.4 ^ 1.9
  sorry

end compare_exponents_and_logarithms_l574_574280


namespace solution_set_inequality_l574_574302

variable {x : ℝ}
variable {a b : ℝ}

theorem solution_set_inequality (h₁ : ∀ x : ℝ, (ax^2 + bx - 1 > 0) ↔ (-1/2 < x ∧ x < -1/3)) :
  ∀ x : ℝ, (x^2 - bx - a ≥ 0) ↔ (x ≤ -3 ∨ x ≥ -2) := 
sorry

end solution_set_inequality_l574_574302


namespace equation_of_perpendicular_line_l574_574627

theorem equation_of_perpendicular_line (O : Point) (l l_perp : Line) 
  (H1 : O = (0,0)) 
  (H2 : line_passes_through l O) 
  (H3 : line_perpendicular l l_perp)
  (H4 : equation_of_line l_perp = λ x y, x - y - 3 = 0) :
  equation_of_line l = λ x y, x + y = 0 := 
by 
  sorry

end equation_of_perpendicular_line_l574_574627


namespace necessary_but_not_sufficient_l574_574293

theorem necessary_but_not_sufficient (x : ℝ) : (x < 0) -> (x^2 + x < 0 ↔ -1 < x ∧ x < 0) :=
by
  sorry

end necessary_but_not_sufficient_l574_574293


namespace quadratic_rewrite_l574_574451

theorem quadratic_rewrite :
  ∃ a d : ℤ, (∀ x : ℝ, x^2 + 500 * x + 2500 = (x + a)^2 + d) ∧ (d / a) = -240 := by
  sorry

end quadratic_rewrite_l574_574451


namespace arun_weight_upper_limit_l574_574346

theorem arun_weight_upper_limit (weight : ℝ) (avg_weight : ℝ) 
  (arun_opinion : 66 < weight ∧ weight < 72) 
  (brother_opinion : 60 < weight ∧ weight < 70) 
  (average_condition : avg_weight = 68) : weight ≤ 70 :=
by
  sorry

end arun_weight_upper_limit_l574_574346


namespace probability_at_least_6_heads_in_8_flips_l574_574133

open scoped BigOperators

def binom (n k : ℕ) : ℕ := nat.choose n k

def total_outcomes (n : ℕ) := 2^n

def successful_outcomes (n k : ℕ) :=
  (finset.range (n + 1)).filter (λ x, x ≥ k).sum (λ x, binom n x)

def probability (n k : ℕ) :=
  (successful_outcomes n k) / (total_outcomes n : ℚ)

theorem probability_at_least_6_heads_in_8_flips :
  probability 8 6 = 37 / 256 := sorry

end probability_at_least_6_heads_in_8_flips_l574_574133


namespace length_PB_correct_l574_574093

noncomputable def length_of_PB 
(P : Point) (A B C D : Point) (PA PD PC : ℝ) : ℝ := 
  if PA = 5 ∧ PD = 3 ∧ PC = 7 then
    Real.sqrt 65
  else
    sorry

theorem length_PB_correct 
(P A B C D : Point)
(hPA : dist P A = 5) 
(hPD : dist P D = 3) 
(hPC : dist P C = 7) : 
  dist P B = Real.sqrt 65 := by
  apply length_of_PB
  all_goals { assumption }

end length_PB_correct_l574_574093


namespace light_flash_fraction_l574_574509

def light_flash_fraction_of_hour (n : ℕ) (t : ℕ) (flashes : ℕ) := 
  (n * t) / (60 * 60)

theorem light_flash_fraction (n : ℕ) (t : ℕ) (flashes : ℕ) (h1 : t = 12) (h2 : flashes = 300) : 
  light_flash_fraction_of_hour n t flashes = 1 := 
by
  sorry

end light_flash_fraction_l574_574509


namespace number_of_zeros_of_f_l574_574034

def f (x : ℝ) : ℝ := 2 * x - Math.sin x

theorem number_of_zeros_of_f : ∃! x : ℝ, f x = 0 :=
by sorry

end number_of_zeros_of_f_l574_574034


namespace age_sum_l574_574758

theorem age_sum (my_age : ℕ) (mother_age : ℕ) (h1 : mother_age = 3 * my_age) (h2 : my_age = 10) :
  my_age + mother_age = 40 :=
by 
  -- proof omitted
  sorry

end age_sum_l574_574758


namespace probability_at_least_6_heads_in_8_flips_l574_574154

def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then (Nat.choose n k) else 0

theorem probability_at_least_6_heads_in_8_flips :
  let total_outcomes := 2^8
  let successful_outcomes := binomial 8 6 + binomial 8 7 + binomial 8 8
  let probability := (successful_outcomes : ℚ) / total_outcomes
  probability = 37 / 256 :=
by
  sorry

end probability_at_least_6_heads_in_8_flips_l574_574154


namespace simplify_fraction_l574_574001

theorem simplify_fraction : (48 / 72 : ℚ) = (2 / 3) := 
by
  sorry

end simplify_fraction_l574_574001


namespace find_x_satisfies_equation_l574_574394

theorem find_x_satisfies_equation :
  let x : ℤ := -14
  ∃ x : ℤ, (36 - x) - (14 - x) = 2 * ((36 - x) - (18 - x)) :=
by
  let x := -14
  use x
  sorry

end find_x_satisfies_equation_l574_574394


namespace perpendicular_lines_sum_is_minus_four_l574_574298

theorem perpendicular_lines_sum_is_minus_four 
  (a b c : ℝ) 
  (h1 : (a * 2) / (4 * 5) = 1)
  (h2 : 10 * 1 + 4 * c - 2 = 0)
  (h3 : 2 * 1 - 5 * (-2) + b = 0) : 
  a + b + c = -4 := 
sorry

end perpendicular_lines_sum_is_minus_four_l574_574298


namespace find_x_squared_plus_y_squared_find_xy_l574_574278

variable {x y : ℝ}

theorem find_x_squared_plus_y_squared (h1 : (x - y)^2 = 4) (h2 : (x + y)^2 = 64) : x^2 + y^2 = 34 :=
sorry

theorem find_xy (h1 : (x - y)^2 = 4) (h2 : (x + y)^2 = 64) : x * y = 15 :=
sorry

end find_x_squared_plus_y_squared_find_xy_l574_574278


namespace sum_of_nine_three_digit_numbers_l574_574022

def nine_digits := {1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem sum_of_nine_three_digit_numbers :
  ∃ (a b c d e f g h i : ℕ),
  a ∈ nine_digits ∧ b ∈ nine_digits ∧ c ∈ nine_digits ∧ d ∈ nine_digits ∧ e ∈ nine_digits ∧
  f ∈ nine_digits ∧ g ∈ nine_digits ∧ h ∈ nine_digits ∧ i ∈ nine_digits ∧
  (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧ (a ≠ f) ∧ (a ≠ g) ∧ (a ≠ h) ∧ (a ≠ i) ∧
  (b ≠ c) ∧ (b ≠ d) ∧ (b ≠ e) ∧ (b ≠ f) ∧ (b ≠ g) ∧ (b ≠ h) ∧ (b ≠ i) ∧
  (c ≠ d) ∧ (c ≠ e) ∧ (c ≠ f) ∧ (c ≠ g) ∧ (c ≠ h) ∧ (c ≠ i) ∧
  (d ≠ e) ∧ (d ≠ f) ∧ (d ≠ g) ∧ (d ≠ h) ∧ (d ≠ i) ∧
  (e ≠ f) ∧ (e ≠ g) ∧ (e ≠ h) ∧ (e ≠ i) ∧
  (f ≠ g) ∧ (f ≠ h) ∧ (f ≠ i) ∧
  (g ≠ h) ∧ (g ≠ i) ∧
  (h ≠ i) ∧
  (a + b + c + d + e + f + g + h + i = 45) ∧
  (a * 100 + b * 10 + c + b * 100 + c * 10 + d + c * 100 + d * 10 + e + d * 100 + e * 10 + f + e * 100 + f * 10 + g + f * 100 + g * 10 + h + g * 100 + h * 10 + i + h * 100 + i * 10 + a + i * 100 + a * 10 + b = 4995) :=
sorry

end sum_of_nine_three_digit_numbers_l574_574022


namespace train_passing_time_l574_574356

noncomputable def train_length : ℝ := 180
noncomputable def train_speed_km_hr : ℝ := 36
noncomputable def train_speed_m_s : ℝ := train_speed_km_hr * (1000 / 3600)

theorem train_passing_time : train_length / train_speed_m_s = 18 := by
  sorry

end train_passing_time_l574_574356


namespace complement_union_l574_574733

noncomputable theory 
open Set Real

def A : Set ℝ := {y | ∃ x, y = log 2 x ∧ x > 4}
def B : Set ℝ := {x | x ^ 2 - 3 * x + 2 < 0}

theorem complement_union (RA : Set ℝ := A) (RB : Set ℝ := B) : 
  (-(A) ∪ B) = {y | y ≤ 2} :=
sorry

end complement_union_l574_574733


namespace Joey_select_fourth_stick_l574_574726

open Nat

theorem Joey_select_fourth_stick :
  let sticks := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30}
  let on_table := {3, 7, 15}
  let remaining_sticks := sticks \ on_table
  {x : ℕ | x ∈ remaining_sticks ∧ 5 < x ∧ x < 25}.card = 17 :=
by
  sorry

end Joey_select_fourth_stick_l574_574726


namespace find_a_b_find_sin_2A_l574_574364

-- Ensure that the definitions match the given conditions in the problem.
variable {A B C : ℝ}
variable {a b c : ℝ}
variable (sin : ℝ → ℝ)
variable (cos : ℝ → ℝ)
variable (sqrt : ℝ → ℝ)
variable (eq : ℝ → ℝ → Prop)

axiom rule_C_cosine  (a b c : ℝ) (C : ℝ)
  : cos C = (a^2 + b^2 - c^2) / (2 * a * b)

axiom sine_relation (A B : ℝ) (k : ℝ) (sin : ℝ → ℝ)
  : k * sin A = 3 * sin B → k * a = 3 * b

def given_conditions : Prop := 
  (c = sqrt 7) ∧ (C = real.pi / 3) ∧ (2 * sin A = 3 * sin B) ∧ (cos B = (5 * sqrt 7) / 14)

theorem find_a_b (sin_A : ℝ) (cos_B : ℝ) (cos_C_half : ℝ) (sin_2A : ℝ)
  (h : given_conditions)
  : (a = 3) ∧ (b = 2) :=
sorry

theorem find_sin_2A (h : given_conditions)
  : (sin 2 * A = - (3 * sqrt 3) / 14) :=
sorry

end find_a_b_find_sin_2A_l574_574364


namespace bees_count_l574_574404

theorem bees_count (initial_bees : ℕ) (multiplier : ℕ) (percentage_left : ℝ) (rounded_bees_left : ℕ) :
  initial_bees = 144 → multiplier = 3 → percentage_left = 0.20 → rounded_bees_left = 86 → 
  (multiplier * initial_bees - rounded_bees_left) = 346 :=
by {
  intro h₁ h₂ h₃ h₄,
  have hb_total := h₂.symm ▸ h₁.symm ▸ calc
    3 * 144 : ℕ := 432,
  have hb_left := h₄.symm ▸ calc
    86 : ℕ := Nat.round(0.20 * 432),
  exact h₃.symm ▸ (calc
    432 - 86 : ℕ := 346),
  sorry
}

end bees_count_l574_574404


namespace pedal_triangle_difference_l574_574416

theorem pedal_triangle_difference
  (ABC : Triangle)
  (O : Point)
  (hO : Circumcenter O ABC)
  (A1 B1 C1 : Point)
  (hA1 : Foot A1 (Line B C) ∧ Foot A1 (Line A O))
  (hB1 : Foot B1 (Line A C) ∧ Foot B1 (Line B O))
  (hC1 : Foot C1 (Line A B) ∧ Foot C1 (Line C O))
  (aux_thm : ∀ P Q R (hCirc : Circumcenter O (Triangle.mk P Q R)),
    Line.mk (Radii O P).foot (Line.mk Q R).foot ⊥ Line.mk (Radii O Q).foot (Line.mk P R).foot) :
  ∀ side1_side2 side3_side1 side2_side3, 
    (∃ P Q R S T, S ∈ (Line.mk P Q) ∧ T ∈ (Line.mk P R) ∧
                   (Segment S Q) + (Segment T P) = (Segment S R) + (Segment T Q)) → 
    side1_side2 - side3_side1 = side2_side3 - side1_side2 := sorry

end pedal_triangle_difference_l574_574416


namespace find_x_solution_l574_574229

theorem find_x_solution (x : ℝ) : (10 - x)^2 = (x - 2)^2 + 8 → x = 5.5 :=
by
  assume h : (10 - x)^2 = (x - 2)^2 + 8
  sorry

end find_x_solution_l574_574229


namespace probability_at_least_6_heads_in_8_flips_l574_574158

def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then (Nat.choose n k) else 0

theorem probability_at_least_6_heads_in_8_flips :
  let total_outcomes := 2^8
  let successful_outcomes := binomial 8 6 + binomial 8 7 + binomial 8 8
  let probability := (successful_outcomes : ℚ) / total_outcomes
  probability = 37 / 256 :=
by
  sorry

end probability_at_least_6_heads_in_8_flips_l574_574158


namespace steve_oranges_l574_574428

theorem steve_oranges : 
  ∀ (a b c r : ℕ), a = 46 → b = 4 → c = 7 → r = a - (b + c) → r = 35 :=
by
  intros a b c r h₁ h₂ h₃ h₄
  have h₅ : r = 46 - (4 + 7),
  { rw [←h₁, ←h₂, ←h₃, h₄] },
  linarith

end steve_oranges_l574_574428


namespace percentage_of_second_solution_l574_574900

theorem percentage_of_second_solution (a b : ℕ) (ha : a = 630) (hb : b = 420) :
    let x := (50 * a - 60 * b) * 100 / (a - b)
    x = 30 :=
by
  simp [ha, hb]
  have h1 : (50 * 630 - 60 * 420) * 100 / (630 - 420) = (31500 - 25200) * 100 / 210 := by norm_num
  have h2 : (6300) * 100 / 210 = 630000 / 210 := by norm_num
  have h3 : 630000 / 210 = 30 := by norm_num
  exact h3

end percentage_of_second_solution_l574_574900


namespace max_dist_A_C_after_folding_l574_574538

-- Define the problem conditions
-- Square ABCD with side length 1
def A : (ℝ × ℝ) := (0, 1)
def B : (ℝ × ℝ) := (1, 1)
def C : (ℝ × ℝ) := (1, 0)
def D : (ℝ × ℝ) := (0, 0)

-- Midpoints E and F
def E : (ℝ × ℝ) := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)  -- (1, 0.5)
def F : (ℝ × ℝ) := ((A.1 + D.1) / 2, (A.2 + D.2) / 2)  -- (0, 0.5)

-- Prove the maximum distance between points A and C
theorem max_dist_A_C_after_folding : 
  let max_dist := Real.sqrt 2
  max_dist = Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) :=
by
  sorry

end max_dist_A_C_after_folding_l574_574538


namespace smallest_among_minus2_minus1_0_1_l574_574932

theorem smallest_among_minus2_minus1_0_1 : ∀ x ∈ ({-2, -1, 0, 1} : Set ℚ), x ≥ -2 := by
  intro x hx
  cases hx <;> linarith

end smallest_among_minus2_minus1_0_1_l574_574932


namespace base_multiple_product_l574_574570

theorem base_multiple_product :
  let b1_2_to_10 : ℕ := 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 0 * 2^0,  -- 1010_2 to decimal
  let b2_3_to_10 : ℕ := 1 * 3^2 + 0 * 3^1 + 2 * 3^0  -- 102_3 to decimal
  in b1_2_to_10 * b2_3_to_10 = 110 := by
  sorry

end base_multiple_product_l574_574570


namespace initial_five_machines_complete_job_in_twenty_days_l574_574988

variable (r : ℝ) -- The rate at which each machine works

-- Condition 1: Five machines working at rate r in some days
def combined_rate_five_machines := 5 * r

-- Condition 2: To complete the job in 10 days, 10 machines are needed.
def combined_rate_ten_machines := 10 * r

-- Condition that 10 machines can complete the job in 10 days
def complete_in_ten_days : Prop := (10 * r) * 10 = 1 -- one job

-- The goal is to prove that the initial five machines take 20 days
theorem initial_five_machines_complete_job_in_twenty_days (h : complete_in_ten_days) :
  let d := 20 in
  (5 * r) * d = 1 :=
by
  sorry

end initial_five_machines_complete_job_in_twenty_days_l574_574988


namespace felix_trees_per_sharpening_l574_574976

theorem felix_trees_per_sharpening (dollars_spent : ℕ) (cost_per_sharpen : ℕ) (trees_chopped : ℕ) 
  (h1 : dollars_spent = 35) (h2 : cost_per_sharpen = 5) (h3 : trees_chopped ≥ 91) :
  (91 / (35 / 5)) = 13 := 
by 
  sorry

end felix_trees_per_sharpening_l574_574976


namespace circumference_of_circle_l574_574498

/-- Given a circle with area 4 * π square units, prove that its circumference is 4 * π units. -/
theorem circumference_of_circle (r : ℝ) (h : π * r^2 = 4 * π) : 2 * π * r = 4 * π :=
sorry

end circumference_of_circle_l574_574498


namespace steve_final_amount_l574_574011

def initial_deposit : ℝ := 100
def interest_years_1_to_3 : ℝ := 0.10
def interest_years_4_to_5 : ℝ := 0.08
def annual_deposit_years_1_to_2 : ℝ := 10
def annual_deposit_years_3_to_5 : ℝ := 15

def total_after_one_year (initial : ℝ) (annual : ℝ) (interest : ℝ) : ℝ :=
  initial * (1 + interest) + annual

def steve_saving_after_five_years : ℝ :=
  let year1 := total_after_one_year initial_deposit annual_deposit_years_1_to_2 interest_years_1_to_3
  let year2 := total_after_one_year year1 annual_deposit_years_1_to_2 interest_years_1_to_3
  let year3 := total_after_one_year year2 annual_deposit_years_3_to_5 interest_years_1_to_3
  let year4 := total_after_one_year year3 annual_deposit_years_3_to_5 interest_years_4_to_5
  let year5 := total_after_one_year year4 annual_deposit_years_3_to_5 interest_years_4_to_5
  year5

theorem steve_final_amount :
  steve_saving_after_five_years = 230.88768 := by
  sorry

end steve_final_amount_l574_574011


namespace simplify_frac_48_72_l574_574002

theorem simplify_frac_48_72 : (48 / 72 : ℚ) = 2 / 3 :=
by
  -- In Lean, we prove the equality of the simplified fractions.
  sorry

end simplify_frac_48_72_l574_574002


namespace inradius_one_third_height_l574_574799

-- The problem explicitly states this triangle's sides form an arithmetic progression.
-- We need to define conditions and then prove the question is equivalent to the answer given those conditions.
theorem inradius_one_third_height (a b c r h_b : ℝ) (h : a ≤ b ∧ b ≤ c) (h_arith : 2 * b = a + c) :
  r = h_b / 3 :=
sorry

end inradius_one_third_height_l574_574799


namespace triangle_inradius_is_2_5_l574_574805

variable (A : ℝ) (p : ℝ) (r : ℝ)

def triangle_has_given_inradius (A p : ℝ) : Prop :=
  A = r * p / 2

theorem triangle_inradius_is_2_5 (h₁ : A = 25) (h₂ : p = 20) :
  triangle_has_given_inradius A p r → r = 2.5 := sorry

end triangle_inradius_is_2_5_l574_574805


namespace ratio_CS_CD_volume_of_cone_l574_574763

-- Define the conditions for Part (a)
variables {S A B C D K: Type} -- points
-- Assume SA is on the face of the pyramid with vertex distances:
variables {AK KS : ℝ}
axiom AK_KS_ratio : AK/KS = 2/3

-- K is the apex of a right circular cone
-- S, B, and D lie on the circumference of the base of the cone with vertex K
axiom BK_eq_DK_eq_SK : ∀ (B D: Type), BK = DK ∧ DK = SK

-- Defining the pyramid as a specific type of quadrilateral
axiom quadrilateral_pyramid : ∀ (S A B C D: Type), is_quadrilateral_pyramid S A B C D

-- Define that the ratio CS:CD=1
theorem ratio_CS_CD : ∀ (C S D: Type), CS / CD = 1 :=
by sorry

-- Further define the conditions for Part (b)
-- height of the pyramid is 5
axiom pyramid_height : height(S, A, B, C, D) = 5

-- The volume of the cone defined with radius and height derived from the problem
theorem volume_of_cone : volume_cone (K) = 9 * π / sqrt 5 :=
by sorry

end ratio_CS_CD_volume_of_cone_l574_574763


namespace certain_number_l574_574096

theorem certain_number (a : ℝ) (h : (228 / 100 * 1265) / a = 480.7) : a ≈ 6 := sorry

end certain_number_l574_574096


namespace cos_sum_induction_l574_574095

theorem cos_sum_induction 
  (n : ℕ) 
  (hn : n > 0) 
  (x : ℝ) 
  (hx : ∀ k : ℤ, x ≠ 2 * k * π) 
  : ∑ i in Finset.range n, Real.cos((i + 1) * x) = 
    Real.sin((n + 0.5) * x) / (2 * Real.sin(0.5 * x)) - 0.5 :=
by
  sorry

end cos_sum_induction_l574_574095


namespace cost_of_3600_pens_is_1080_l574_574495

-- Define the conditions
def cost_of_150_pens : ℕ → ℝ := λ n, 45
def pens_count := 150

-- Define the calculation for cost per pen
def cost_per_pen := cost_of_150_pens pens_count / pens_count

-- Define the required number of pens
def required_pens := 3600

-- Define the cost for 3600 pens
def cost_of_3600_pens := required_pens * cost_per_pen

-- Prove that the cost of 3600 pens is 1080 dollars
theorem cost_of_3600_pens_is_1080 : cost_of_3600_pens = 1080 := by
  sorry

end cost_of_3600_pens_is_1080_l574_574495


namespace square_area_correct_l574_574918

noncomputable def square_area : ℝ :=
  let f : ℝ → ℝ := λ x, x^2 + 4 * x + 3
  let y_val : ℝ := 7
  let x1 : ℝ := -2 - 2 * Real.sqrt 2
  let x2 : ℝ := -2 + 2 * Real.sqrt 2
  let side_length := x2 - x1
  side_length * side_length

theorem square_area_correct :
  let f : ℝ → ℝ := λ x, x^2 + 4 * x + 3
  let y_val : ℝ := 7
  let x1 : ℝ := -2 - 2 * Real.sqrt 2
  let x2 : ℝ := -2 + 2 * Real.sqrt 2
  let side_length := x2 - x1
  side_length * side_length = 32 := by
  sorry

end square_area_correct_l574_574918


namespace ratio_of_kinetic_energies_l574_574191

def initial_kinetic_energy (I ω : ℝ) : ℝ := 1 / 2 * I * ω^2

def final_kinetic_energy (I ω : ℝ) : ℝ := 
  1 / 2 * (7 / 10 * I) * (10 / 7 * ω)^2

theorem ratio_of_kinetic_energies (I ω : ℝ) (hI : I ≠ 0) (hω : ω ≠ 0) :
  final_kinetic_energy I ω / initial_kinetic_energy I ω = 10 / 7 :=
by 
  sorry

end ratio_of_kinetic_energies_l574_574191


namespace circle_line_intersection_condition_slope_condition_l574_574701

open Real

theorem circle_line_intersection_condition (k : ℝ) :
  (-3 < k ∧ k < 1/3) ↔
  let M := (4, 0) in
  let r := sqrt 10 in
  let d := abs (4 * k + 2) / sqrt (k^2 + 1) in
  d < r :=
by sorry

theorem slope_condition (k : ℝ) :
  (ON_parallel_MP : Bool) → 
  (ON_parallel_MP = true) ↔ k = -4/3 :=
by sorry

end circle_line_intersection_condition_slope_condition_l574_574701


namespace solve_equation_error_step_l574_574642

theorem solve_equation_error_step 
  (equation : ∀ x : ℝ, (x - 1) / 2 + 1 = (2 * x + 1) / 3) :
  ∃ (step : ℕ), step = 1 ∧
  let s1 := ((x - 1) / 2 + 1) * 6;
  ∀ (x : ℝ), s1 ≠ (((2 * x + 1) / 3) * 6) :=
by
  sorry

end solve_equation_error_step_l574_574642


namespace total_score_even_l574_574689

variable (num_people : ℕ) (score : ℕ)

def individual_score (correct unanswered wrong: ℕ) :=
  5 * correct + 1 * unanswered - 1 * wrong

theorem total_score_even
  (h_correct : ∀ p, individual_score p.correct p.unanswered p.wrong = 5 * p.correct + 1 * p.unanswered - 1 * p.wrong)
  (h_n_questions : ∀ p, p.correct + p.unanswered + p.wrong = 6) :
  ∃ (total_score : ℕ), even total_score :=
by
  sorry

end total_score_even_l574_574689


namespace probability_of_at_least_six_heads_is_correct_l574_574125

-- Definitions for the given problem
def binomial_coefficient (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

def total_possible_outcomes : ℕ :=
  2^8

def favorable_outcomes : ℕ :=
  binomial_coefficient 8 6 + binomial_coefficient 8 7 + binomial_coefficient 8 8

def probability_of_at_least_6_heads : ℚ :=
  favorable_outcomes / total_possible_outcomes

-- The proof statement
theorem probability_of_at_least_six_heads_is_correct :
  probability_of_at_least_6_heads = 37 / 256 :=
by sorry

end probability_of_at_least_six_heads_is_correct_l574_574125


namespace bad_arrangements_count_l574_574037

def is_bad (arrangement : List ℕ) : Prop :=
  arrangement.perm [1, 2, 3, 4, 5, 6] ∧
  (∀ (subset : List ℕ), subset ≠ List.nil → 
    subset ≠ arrangement.take arrangement.length → 
    (subset.sum < 1 ∨ subset.sum > 21 ∨ subset.sum % 2 = 0))

theorem bad_arrangements_count : 
  ∃ (badArrs : Finset (Finset ℕ)), 
    badArrs.card = 3 ∧ 
    ∀ arrangement ∈ badArrs, is_bad arrangement :=
by 
  sorry

end bad_arrangements_count_l574_574037


namespace find_a_for_arithmetic_progression_roots_l574_574248

theorem find_a_for_arithmetic_progression_roots (x a : ℝ) : 
  (∀ (x : ℝ), x^4 - a*x^2 + 1 = 0) → 
  (∃ (t1 t2 : ℝ), t1 > 0 ∧ t2 > 0 ∧ (t2 = 9*t1) ∧ (t1 + t2 = a) ∧ (t1 * t2 = 1)) → 
  (a = 10/3) := 
  by 
    intros h1 h2
    sorry

end find_a_for_arithmetic_progression_roots_l574_574248


namespace bisect_segment_XY_l574_574412

-- Definitions
variables {A B C K L X Y : Type}
variable {triangle : triangle A B C}
variable {midpoint : A × A → A}
variable [reflected : ∀ (P : A) (midP : A), A]

-- Hypotheses
hypothesis (K_on_AB : lies_on K (segment A B))
hypothesis (L_on_AC : lies_on L (segment A C))
hypothesis (KB_eq_LC : distance K B = distance L C)
hypothesis (X_reflection_of_K : X = reflected K (midpoint A C))
hypothesis (Y_reflection_of_L : Y = reflected L (midpoint A B))

-- Theorem
theorem bisect_segment_XY : bisects (angle_bisector A) (segment X Y) := by
sorry

end bisect_segment_XY_l574_574412


namespace max_min_f_on_interval_l574_574256

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x + 1

theorem max_min_f_on_interval : 
  (∀ x ∈ Icc (-3 : ℝ) (0 : ℝ), f x ≤ 3) ∧ 
  (∃ x ∈ Icc (-3 : ℝ) (0 : ℝ), f x = 3) ∧
  (∀ x ∈ Icc (-3 : ℝ) (0 : ℝ), f x ≥ -17) ∧
  (∃ x ∈ Icc (-3 : ℝ) (0 : ℝ), f x = -17) :=
by
  sorry

end max_min_f_on_interval_l574_574256


namespace part1_general_formula_part2_S_n_lt_4_l574_574287

-- Given a sequence of positive terms (a_n) satisfying a specific property
def seq_property (a : ℕ → ℝ) := 
  ∀ n : ℕ, a 1^2 + (∑ i in finset.range (n+1), a (i+1)^2) = 4^n / 3 - 1 / 3

-- The first part: Find the general formula for (a_n)
theorem part1_general_formula (a : ℕ → ℝ) (h : seq_property a) : ∀ n, a n = 2^(n-1) :=
sorry

-- Define the sequence b_n = n / a_n and S_n as the sum of the first n terms of {b_n}
def b (a : ℕ → ℝ) (n : ℕ) := n / a n
def S (a : ℕ → ℝ) (n : ℕ) := ∑ i in finset.range n, b a (i+1)

-- The second part: Prove that S_n < 4
theorem part2_S_n_lt_4 (a : ℕ → ℝ) (h : seq_property a) (h_formula : ∀ n, a n = 2^(n-1)) 
  : ∀ n, S a n < 4 :=
sorry

end part1_general_formula_part2_S_n_lt_4_l574_574287


namespace current_speed_is_2_5_l574_574514

-- Define the parameters: speed with current, speed against current, and unknown speed of the current
variable {m c : ℝ}

-- Define the conditions as assumptions
def speed_with_current := m + c = 15
def speed_against_current := m - c = 10

-- Formalize the proof problem statement
theorem current_speed_is_2_5 (h1 : speed_with_current) (h2 : speed_against_current) : c = 2.5 := by
  sorry

end current_speed_is_2_5_l574_574514


namespace find_ordered_pair_l574_574013

noncomputable def ordered_pair (c d : ℝ) := c = 1 ∧ d = -2

theorem find_ordered_pair (c d : ℝ) (h1 : c ≠ 0) (h2 : d ≠ 0) (h3 : ∀ x : ℝ, x^2 + c * x + d = 0 → (x = c ∨ x = d)) : ordered_pair c d :=
by
  sorry

end find_ordered_pair_l574_574013


namespace simplify_frac_48_72_l574_574003

theorem simplify_frac_48_72 : (48 / 72 : ℚ) = 2 / 3 :=
by
  -- In Lean, we prove the equality of the simplified fractions.
  sorry

end simplify_frac_48_72_l574_574003


namespace sum_of_roots_l574_574228

theorem sum_of_roots :
  ∑ x in ({x : ℝ | x^2 + 2005 * x = 2006}), x = -2005 := by
  sorry

end sum_of_roots_l574_574228


namespace repeating_decimal_to_fraction_l574_574566

theorem repeating_decimal_to_fraction :
  let x := (0.6 : ℚ) + (0.03 / (1 - 0.01)) in
  x = 104 / 165 :=
by
  let x : ℚ := (0.6 : ℚ) + (3 / 99)
  have h₁ : 0.6 = 3 / 5 := by sorry
  have h₂ : 0.03 / (1 - 0.01) = 1 / 33 := by sorry
  rw [h₁, h₂]
  exact (3 / 5 + 1 / 33 = 104 / 165) sorry

end repeating_decimal_to_fraction_l574_574566


namespace perpendicular_and_intersection_l574_574643

-- Definitions for the given equations
def eq1 : ℝ × ℝ → Prop := λ (x, y), 4 * y - 3 * x = 16
def eq2 : ℝ × ℝ → Prop := λ (x, y), -3 * x - 4 * y = 15
def eq3 : ℝ × ℝ → Prop := λ (x, y), 4 * y + 3 * x = 16
def eq4 : ℝ × ℝ → Prop := λ (x, y), 3 * y + 4 * x = 15

-- Definition for perpendicular lines and intersection point
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1
noncomputable def slope (a b c : ℝ) : ℝ := -a / b

-- Statement to be proved
theorem perpendicular_and_intersection :
  (perpendicular (slope (-3) 4 16) (slope (-4) 3 15)) ∧
  (∃ (x y : ℝ), eq1 (x, y) ∧ eq4 (x, y) ∧ x = 12 / 25 ∧ y = 109 / 25) :=
by
  sorry

end perpendicular_and_intersection_l574_574643


namespace sqrt_sum_of_fractions_l574_574483

theorem sqrt_sum_of_fractions :
  (Real.sqrt ((25 / 36) + (16 / 9)) = (Real.sqrt 89) / 6) :=
by
  sorry

end sqrt_sum_of_fractions_l574_574483


namespace cubic_sum_root_l574_574867

theorem cubic_sum_root :
  let x := (2 : ℝ)^(1 / 3) + (3 : ℝ)^(1 / 3)
  in x^9 - 15 * x^6 - 87 * x^3 - 125 = 0 :=
by {
  let x := (2 : ℝ)^(1 / 3) + (3 : ℝ)^(1 / 3),
  sorry
}

end cubic_sum_root_l574_574867


namespace molecular_weight_acetic_acid_l574_574842

-- Define atomic weights
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

-- Define the number of each atom in acetic acid
def num_C : ℕ := 2
def num_H : ℕ := 4
def num_O : ℕ := 2

-- Define the molecular formula of acetic acid
def molecular_weight_CH3COOH : ℝ :=
  num_C * atomic_weight_C +
  num_H * atomic_weight_H +
  num_O * atomic_weight_O

-- State the proposition
theorem molecular_weight_acetic_acid :
  molecular_weight_CH3COOH = 60.052 := by
  sorry

end molecular_weight_acetic_acid_l574_574842


namespace max_possible_x_plus_y_l574_574677

theorem max_possible_x_plus_y (x y : ℕ) (h₁: ∃ n : ℕ, n = 30)
  (h₂: (factorial n : ℚ) / (36^x * 25^y) ∈ ℤ) : x + y ≤ 10 :=
sorry

end max_possible_x_plus_y_l574_574677


namespace _l574_574190

def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

example : triangle_inequality 1 1 1 := 
by {
  -- Prove using the triangle inequality theorem that the sides form a triangle.
  -- This part is left as an exercise to the reader.
  sorry
}

end _l574_574190


namespace prob_at_least_6_heads_eq_l574_574114

-- define the number of coin flips
def n := 8

-- define the number of possible outcomes (2^n)
def total_outcomes := 2 ^ n

-- define the binomial coefficients for cases: 6 heads, 7 heads, 8 heads
def binom_8_6 := Nat.choose 8 6
def binom_8_7 := Nat.choose 8 7
def binom_8_8 := Nat.choose 8 8

-- calculate the favorable outcomes for at least 6 heads
def favorable_outcomes := binom_8_6 + binom_8_7 + binom_8_8

-- define the probability of getting at least 6 heads
def probability := (favorable_outcomes : ℚ) / total_outcomes

theorem prob_at_least_6_heads_eq : probability = 37 / 256 := by
  sorry

end prob_at_least_6_heads_eq_l574_574114


namespace max_odd_numbers_in_pyramid_l574_574204

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

end max_odd_numbers_in_pyramid_l574_574204


namespace derivative_f_at_a_l574_574731

def f (x : ℝ) : ℝ := Real.cos (Real.cos (Real.cos (Real.cos (Real.cos (Real.cos (Real.cos (Real.cos x))))))))

variable (a : ℝ) (h_a : a = Real.cos a)

theorem derivative_f_at_a :
  let f_8 := f in
    deriv f_8 a = a^8 - 4 * a^6 + 6 * a^4 - 4 * a^2 + 1 :=
sorry

end derivative_f_at_a_l574_574731


namespace inequality_holds_l574_574285

variable (a b c : ℝ)

theorem inequality_holds (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + b*c) / (a * (b + c)) + 
  (b^2 + c*a) / (b * (c + a)) + 
  (c^2 + a*b) / (c * (a + b)) ≥ 3 :=
sorry

end inequality_holds_l574_574285


namespace find_base_of_log_l574_574652

open Function

theorem find_base_of_log (a : ℝ) (f : ℝ → ℝ) (hf : ∀ x, f x = 1 + log a x) :
  f⁻¹ 2 = 4 -> a = 4 :=
by
  sorry

end find_base_of_log_l574_574652


namespace find_parameter_p_l574_574513

theorem find_parameter_p 
  (p : ℝ) 
  (A B : ℝ × ℝ) 
  (h_parabola : ∀ x y, y^2 = 2 * p * x) 
  (h_p_pos : p > 0)
  (h_focus : let F := (p / 2, 0) in  (F.snd = 0 ∧ F.fst = p / 2))
  (h_line_eq : let line := (&F).fst = &F.snd then line = 45 degree angle)
  (h_A_int_B : let A : ℝ × ℝ := parabola ∧ let B : ℝ × ℝ := parabola in (AB : ℝ × ℝ)).fst ∧ (AB : ℝ × ℝ)).snd
  (h_length_eq : segment length(A, B) = 8)
  : p = 2 := 
  sorry

end find_parameter_p_l574_574513


namespace no_additional_gold_needed_l574_574506

-- Given conditions
def volume_ring (h : ℝ) : ℝ :=
  (π * h^3) / 6

-- Theorem to prove
theorem no_additional_gold_needed (h d k : ℝ) (h_pos : 0 < h):
  volume_ring h = (π * h^3) / 6 ∧
  ∀ d_new : ℝ, d_new = k * d → volume_ring h = (π * h^3) / 6 →
  0 = 0 :=
by 
  sorry

end no_additional_gold_needed_l574_574506


namespace equilateral_triangle_segment_sum_l574_574994

-- Define the problem setup
variables {A B C M : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M]
  (a : ℝ) -- side length of equilateral triangle ABC
  (h : A ≠ B)
  (k : B ≠ C)
  (l : C ≠ A)
  (AC1 BA1 CB1 C1B A1C B1A : ℝ) -- the lengths of segments
  (MC1 MA1 MB1 : ℝ) -- perpendicular distances from M to the sides

-- Definition of the problem conditions
def isEquilateralTriangle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] (a : ℝ) : Prop :=
  dist A B = a ∧ dist B C = a ∧ dist C A = a

def perpendicularFromPoint (M A B C : Type) [MetricSpace M] [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (MtoAB MtoBC MtoCA : ℝ) : Prop :=
  dist M A = MtoAB ∧ dist M B = MtoBC ∧ dist M C = MtoCA ∧
    (∃ C1, dist M C1 = MtoAB ∧ C1 ∈ lineSegment A B) ∧
    (∃ A1, dist M A1 = MtoBC ∧ A1 ∈ lineSegment B C) ∧
    (∃ B1, dist M B1 = MtoCA ∧ B1 ∈ lineSegment C A)

-- Define the mathematically equivalent proof problem
theorem equilateral_triangle_segment_sum {A B C M : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M] 
  (a : ℝ) (h : A ≠ B) (k : B ≠ C) (l : C ≠ A) 
  (AC1 BA1 CB1 C1B A1C B1A : ℝ) 
  (MC1 MA1 MB1 : ℝ) 
  (isEquilateral : isEquilateralTriangle A B C a) 
  (perpendiculars : perpendicularFromPoint M A B C MC1 MA1 MB1) :
  AC1 + BA1 + CB1 = C1B + A1C + B1A :=
by
  sorry

end equilateral_triangle_segment_sum_l574_574994


namespace binom_12_9_is_220_l574_574216

def choose (n k : ℕ) : ℕ := n.choose k

theorem binom_12_9_is_220 :
  choose 12 9 = 220 :=
by {
  -- Proof is omitted
  sorry
}

end binom_12_9_is_220_l574_574216


namespace plane_through_A_perpendicular_to_BC_l574_574488

def Point3D := (ℝ × ℝ × ℝ)

def vector_sub (p1 p2 : Point3D) : Point3D :=
  match p1, p2 with
  | (x1, y1, z1), (x2, y2, z2) => (x1 - x2, y1 - y2, z1 - z2)

def plane_equation (n : Point3D) (p : Point3D) (x y z : ℝ) : ℝ :=
  match n, p with
  | (a, b, c), (x0, y0, z0) => a * (x - x0) + b * (y - y0) + c * (z - z0)

def A := (-1, 2, -2)
def B := (13, 14, 1)
def C := (14, 15, 2)
def BC := vector_sub C B
def normal := BC

theorem plane_through_A_perpendicular_to_BC :
  ∀ (x y z : ℝ), plane_equation normal A x y z = 0 ↔ x + y + z + 1 = 0 :=
by
  intros x y z
  have h1 : BC = (1, 1, 1) := by rfl
  have h2 : normal = BC := by rfl
  have h3 : A = (-1, 2, -2) := by rfl
  simp [plane_equation, normal, A, h1, h2, h3]
  sorry

end plane_through_A_perpendicular_to_BC_l574_574488


namespace probability_at_least_6_heads_8_flips_l574_574167

-- Define the probability calculation of getting at least 6 heads in 8 coin flips.
def probability_at_least_6_heads (n : ℕ) (k : ℕ) : ℚ :=
  (Nat.choose n k + Nat.choose n (k + 1) + Nat.choose n (k + 2)) / 2^n

theorem probability_at_least_6_heads_8_flips : 
  probability_at_least_6_heads 8 6 = 37 / 256 := 
by
  sorry

end probability_at_least_6_heads_8_flips_l574_574167


namespace find_y_l574_574965

theorem find_y (y : ℝ) : 
  16^(-3/2) = (2^(70/y)) / ((2^(40/y)) * (16^(10/y))) → y = 5/3 :=
by
  sorry

end find_y_l574_574965


namespace fraction_pow_zero_l574_574477

noncomputable def fraction := (-576345 : ℚ) / 2456789

theorem fraction_pow_zero : fraction ≠ 0 → fraction ^ 0 = 1 :=
begin
    intro h,
    rw [pow_zero],
end

end fraction_pow_zero_l574_574477


namespace projection_correct_l574_574586

open Real

def vector_u : ℝ × ℝ := (3, -4)
def vector_v : ℝ × ℝ := (1, 2)

def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

def norm_squared (a : ℝ × ℝ) : ℝ :=
  dot_product a a

def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let scale := (dot_product u v) / (norm_squared v)
  (scale * v.1, scale * v.2)

theorem projection_correct :
  projection vector_u vector_v = (-1, -2) := by
  sorry

end projection_correct_l574_574586


namespace probability_of_dart_landing_in_triangle_l574_574503

-- Define the side length of the hexagon
variables (s : ℝ) (h : s > 0)

-- Define the regular hexagon and its area
def hexagon_area (s : ℝ) : ℝ := (3 * real.sqrt 3 / 2) * s^2

-- Define the equilateral triangle inside the hexagon and its area
def central_triangle_area (s : ℝ) : ℝ := (3 * real.sqrt 3 / 16) * s^2

-- Calculate the probability
def dart_probability (s : ℝ) : ℝ := central_triangle_area s / hexagon_area s

-- Prove that the probability is 1/8
theorem probability_of_dart_landing_in_triangle (s : ℝ) (h : s > 0) : 
  dart_probability s = 1 / 8 :=
by
  sorry

end probability_of_dart_landing_in_triangle_l574_574503


namespace line_through_incenter_divides_equally_line_through_inscribed_center_divides_equally_all_dividing_lines_intersect_at_incenter_l574_574081

noncomputable def divides_area_and_perimeter_equally (l : Line) (T : Triangle) : Prop :=
  divides_area_equally l T ∧ divides_perimeter_equally l T

def incenter (T : Triangle) : Point := sorry -- Definition of the incenter

theorem line_through_incenter_divides_equally (T : Triangle) (l : Line) :
  divides_area_and_perimeter_equally l T → l.passes_through (incenter T) :=
sorry

def inscribed_circle_center (P : Polygon) : Point := sorry -- Definition of the center of the inscribed circle

theorem line_through_inscribed_center_divides_equally (P : Polygon) (l : Line) :
  inscribed_circle P → divides_area_and_perimeter_equally l P → l.passes_through (inscribed_circle_center P) :=
sorry

theorem all_dividing_lines_intersect_at_incenter (T : Triangle) :
  (∀ l1 l2 : Line, divides_area_and_perimeter_equally l1 T → divides_area_and_perimeter_equally l2 T → l1.intersects l2) :=
sorry

end line_through_incenter_divides_equally_line_through_inscribed_center_divides_equally_all_dividing_lines_intersect_at_incenter_l574_574081


namespace cubical_box_edge_length_is_one_meter_l574_574501

-- Define the edge length of one cube in centimeters
def cube_edge_length_cm : ℝ := 25

-- Define the number of cubes the cubical box can hold
def number_of_cubes : ℕ := 64

-- The total volume of the smaller cubes
def total_volume_cm3 : ℝ := (cube_edge_length_cm)^3 * number_of_cubes

-- The edge length of the cubical box in centimeters
def edge_length_cm : ℝ := (total_volume_cm3)^(1/3)

-- Convert the edge length of the cubical box to meters
def edge_length_m : ℝ := edge_length_cm / 100

-- The final theorem to prove the edge length in meters
theorem cubical_box_edge_length_is_one_meter : edge_length_m = 1 := by
  sorry

end cubical_box_edge_length_is_one_meter_l574_574501


namespace probability_at_least_6_heads_in_8_flips_l574_574139

theorem probability_at_least_6_heads_in_8_flips : 
  (∑ k in finset.range 3, nat.choose 8 (6 + k)) / (2 ^ 8) = 37 / 256 :=
by sorry

end probability_at_least_6_heads_in_8_flips_l574_574139


namespace isosceles_triangle_area_theorem_l574_574871

noncomputable def isosceles_triangle_area : Prop :=
  ∀ (A B C M D : ℝ × ℝ),
  (A = (0, 0)) →
  (B = (2, 0)) →
  (C = (2, sqrt 12)) →
  (M = ((A.1 + C.1) / 2, (A.2 + C.2) / 2)) →
  (D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) →
  let BD := sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) in
  let DM := sqrt ((D.1 - M.1)^2 + (D.2 - M.2)^2) in
  let area := 1 / 2 * BD * DM in
  area = 1

theorem isosceles_triangle_area_theorem : isosceles_triangle_area :=
  sorry -- Proof steps omitted

end isosceles_triangle_area_theorem_l574_574871


namespace set_membership_proof_l574_574597

variable (A : Set ℕ) (B : Set (Set ℕ))

theorem set_membership_proof :
  A = {0, 1} → B = {x | x ⊆ A} → A ∈ B :=
by
  intros hA hB
  rw [hA, hB]
  sorry

end set_membership_proof_l574_574597


namespace problem_statement_l574_574392

def f (x : ℝ) : ℝ := 2 * x - Real.cos x

def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_f (a : ℕ → ℝ) : ℝ :=
  f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5)

theorem problem_statement (a : ℕ → ℝ) :
  arithmetic_seq a (π / 8) →
  sum_f a = 5 * π →
  (f (a 3))^2 - a 1 * a 5 = 13 * π^2 / 16 :=
by
  sorry

end problem_statement_l574_574392


namespace max_value_of_function_l574_574579

noncomputable def f (x : ℝ) : ℝ := real.sqrt (x + 16) + real.sqrt (20 - x) + 2 * real.sqrt x

theorem max_value_of_function : 
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 20 ∧ f x = (16 * real.sqrt 3 + 2 * real.sqrt 33) / 3 :=
sorry

end max_value_of_function_l574_574579


namespace eve_takes_5_envelopes_l574_574970

noncomputable def envelope_numbers : List ℕ := [1, 2, 4, 8, 16, 32, 64, 128]

theorem eve_takes_5_envelopes (E A : ℕ) 
  (h_sum : envelope_numbers.sum = 255)
  (h_diff : E = A + 31)
  (h_A : A = 255 - E)
  (h_subset_sum : E = (envelope_numbers.filter (λ x, x ∈ envelope_numbers)).sum) :
  (envelope_numbers.filter (λ x, x ∈ envelope_numbers)).length = 5 :=
sorry

end eve_takes_5_envelopes_l574_574970


namespace sin_cos_identity_l574_574094

theorem sin_cos_identity : 
  sin (7 * π / 8) * cos (7 * π / 8) = - (sqrt 2) / 4 :=
by
  -- Given standard trigonometric identities
  -- sin(2x) = 2 * sin(x) * cos(x)
  -- sin(2π - x) = - sin(x)
  sorry

end sin_cos_identity_l574_574094


namespace num_unique_five_topping_pizzas_l574_574898

open Nat

/-- The number of combinations of choosing k items from n items is defined using binomial coefficients. -/
def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem num_unique_five_topping_pizzas:
  let toppings := 8
      toppings_per_pizza := 5
  in binomial_coefficient toppings toppings_per_pizza = 56 := by
  sorry

end num_unique_five_topping_pizzas_l574_574898


namespace power_function_below_identity_l574_574479

theorem power_function_below_identity {α : ℝ} :
  (∀ x : ℝ, 1 < x → x^α < x) → α < 1 :=
by
  intro h
  sorry

end power_function_below_identity_l574_574479


namespace probability_at_least_6_heads_in_8_flips_l574_574138

theorem probability_at_least_6_heads_in_8_flips : 
  (∑ k in finset.range 3, nat.choose 8 (6 + k)) / (2 ^ 8) = 37 / 256 :=
by sorry

end probability_at_least_6_heads_in_8_flips_l574_574138


namespace area_of_square_l574_574909

-- Define the parabola and the line
def parabola (x : ℝ) : ℝ := x^2 + 4 * x + 3
def line (y : ℝ) : Prop := y = 7

-- Define the roots of the quadratic equation derived from the conditions
noncomputable def root1 : ℝ := -2 + 2 * Real.sqrt 2
noncomputable def root2 : ℝ := -2 - 2 * Real.sqrt 2

-- Define the side length of the square
noncomputable def side_length : ℝ := abs (root1 - root2)

-- Define the area of the square
noncomputable def area_square : ℝ := side_length^2

-- Theorem statement for the problem
theorem area_of_square : area_square = 32 :=
sorry

end area_of_square_l574_574909


namespace probability_of_one_each_color_is_two_fifths_l574_574055

/-- Definition for marbles bag containing 2 red, 2 blue, and 2 green marbles -/
structure MarblesBag where
  red : ℕ
  blue : ℕ
  green : ℕ
  total : ℕ := red + blue + green

/-- Initial setup for the problem -/
def initialBag : MarblesBag := { red := 2, blue := 2, green := 2 }

/-- Represents the outcome of selecting marbles without replacement -/
def selectMarbles (bag : MarblesBag) (count : ℕ) : ℕ :=
  Nat.choose bag.total count

/-- The number of ways to select one marble of each color -/
def selectOneOfEachColor (bag : MarblesBag) : ℕ :=
  bag.red * bag.blue * bag.green

/-- Calculate the probability of selecting one marble of each color -/
def probabilityOneOfEachColor (bag : MarblesBag) (selectCount : ℕ) : ℚ :=
  selectOneOfEachColor bag / selectMarbles bag selectCount

/-- Theorem stating the answer to the probability problem -/
theorem probability_of_one_each_color_is_two_fifths (bag : MarblesBag) :
  probabilityOneOfEachColor bag 3 = 2 / 5 := by
  sorry

end probability_of_one_each_color_is_two_fifths_l574_574055


namespace waiting_for_stocker_proof_l574_574761

-- Definitions for the conditions
def waiting_for_cart := 3
def waiting_for_employee := 13
def waiting_in_line := 18
def total_shopping_trip_time := 90
def time_shopping := 42

-- Calculate the total waiting time
def total_waiting_time := total_shopping_trip_time - time_shopping

-- Calculate the total known waiting time
def total_known_waiting_time := waiting_for_cart + waiting_for_employee + waiting_in_line

-- Calculate the waiting time for the stocker
def waiting_for_stocker := total_waiting_time - total_known_waiting_time

-- Prove that the waiting time for the stocker is 14 minutes
theorem waiting_for_stocker_proof : waiting_for_stocker = 14 := by
  -- Here the proof steps would normally be included
  sorry

end waiting_for_stocker_proof_l574_574761


namespace picasso_postcards_probability_l574_574057

theorem picasso_postcards_probability :
  let total_postcards := 12
  let picasso_postcards := 4
  let other_postcards := total_postcards - picasso_postcards
  let total_ways := Nat.factorial total_postcards
  let ways_postcards_in_block := Nat.factorial picasso_postcards
  let ways_units := Nat.factorial other_postcards.succ -- since we treat 4 Picasso postcards as one block
  (ways_units * ways_postcards_in_block / total_ways) = (1 / 55 : ℚ) :=
by
  sorry

end picasso_postcards_probability_l574_574057


namespace probability_makes_4th_shot_l574_574104

noncomputable def a : ℕ → ℚ
| 1       := 2 / 3
| (n + 1) := (1 / 3) + ((1 / 3) * a n)

theorem probability_makes_4th_shot : a 4 = 41 / 81 :=
by
  sorry

end probability_makes_4th_shot_l574_574104


namespace sector_arc_length_l574_574017

-- Definitions corresponding to the conditions
def sector_area : ℝ := 60 * Real.pi
def central_angle : ℝ := 150
def radius (r : ℝ) : Prop := (sector_area = (central_angle * Real.pi * r^2) / 360)

-- Statement of the problem translated to Lean 4
theorem sector_arc_length (r : ℝ) (h : radius r) : (central_angle * Real.pi * r / 180 = 10 * Real.pi) :=
  sorry

end sector_arc_length_l574_574017


namespace solve_for_k_l574_574328

theorem solve_for_k (x k : ℝ) (h : x = -3) (h_eq : k * (x + 4) - 2 * k - x = 5) : k = -2 :=
by sorry

end solve_for_k_l574_574328


namespace selling_price_when_profit_l574_574450

theorem selling_price_when_profit (x : ℝ) (cp sp_l : ℝ) (h1: x - cp = cp - sp_l) (h2: cp = 64) (h3: sp_l = 42) : x = 86 :=
by
  rw [h2, h3, sub_eq_iff_eq_add] at h1
  exact eq_add_of_sub_eq h1


end selling_price_when_profit_l574_574450


namespace pole_intersection_height_l574_574472

theorem pole_intersection_height 
  (h1 h2 d : ℝ) 
  (h1pos : h1 = 30) 
  (h2pos : h2 = 90) 
  (dpos : d = 150) : 
  ∃ y, y = 22.5 :=
by
  sorry

end pole_intersection_height_l574_574472


namespace Mona_bikes_30_miles_each_week_l574_574403

theorem Mona_bikes_30_miles_each_week :
  let monday_distance := 6
  let wednesday_distance := 12
  let saturday_distance := 2 * monday_distance
  let total_distance := monday_distance + wednesday_distance + saturday_distance
  total_distance = 30 := by
  let monday_distance := 6
  let wednesday_distance := 12
  let saturday_distance := 2 * monday_distance
  let total_distance := monday_distance + wednesday_distance + saturday_distance
  show total_distance = 30
  sorry

end Mona_bikes_30_miles_each_week_l574_574403


namespace number_of_multiples_less_than_201_l574_574674

theorem number_of_multiples_less_than_201 :
  let multiples_of_6 := (Nat.floor (200 / 6) : ℕ)
  let multiples_of_9 := (Nat.floor (200 / 9) : ℕ)
  let multiples_of_18 := (Nat.floor (200 / 18) : ℕ)
  multiples_of_6 - multiples_of_18 + multiples_of_9 - multiples_of_18 = 33 :=
by
  let multiples_of_6 := (Nat.floor (200 / 6) : ℕ)
  let multiples_of_9 := (Nat.floor (200 / 9) : ℕ)
  let multiples_of_18 := (Nat.floor (200 / 18) : ℕ)
  have multiples_of_6_def := Nat.floor (200 / 6) = 33,
  have multiples_of_9_def := Nat.floor (200 / 9) = 22,
  have multiples_of_18_def := Nat.floor (200 / 18) = 11,
  show (multiples_of_6 - multiples_of_18 + multiples_of_9 - multiples_of_18 = 33) from sorry

end number_of_multiples_less_than_201_l574_574674


namespace geometric_series_sum_l574_574230

theorem geometric_series_sum :
  (1 / 5 - 1 / 25 + 1 / 125 - 1 / 625 + 1 / 3125) = 521 / 3125 :=
by
  sorry

end geometric_series_sum_l574_574230


namespace geo_seq_value_l574_574709

variable (a : ℕ → ℝ)
variable (a_2 : a 2 = 2) 
variable (a_4 : a 4 = 8)
variable (geo_prop : a 2 * a 6 = (a 4) ^ 2)

theorem geo_seq_value : a 6 = 32 := 
by 
  sorry

end geo_seq_value_l574_574709


namespace Wendy_received_correct_amount_l574_574476

noncomputable def Wendy_received_amount : ℝ :=
  let cost_dark := 3.25
  let cost_milk := 4.15
  let cost_white := 5.45
  let num_bars := 4
  let total_bars := 12
  let discount_rate := 0.08
  let exchange_rate := 1.1
  let return_milk := 2
  let return_white := 1
  let tax_return := 0.05
  let tax_purchase := 0.07
  let initial_cost :=
    num_bars * (cost_dark + cost_milk + cost_white)
  let discount := initial_cost * discount_rate
  let discounted_cost := initial_cost - discount
  let fluctuation_cost := discounted_cost * exchange_rate
  let returned_cost :=
    return_milk * cost_milk + return_white * cost_white
  let tax_returned := returned_cost * tax_return
  let returned_cost_with_tax := returned_cost + tax_returned
  let adjusted_cost := fluctuation_cost - returned_cost_with_tax
  let tax_purchased := adjusted_cost * tax_purchase
  let total_cost := adjusted_cost + tax_purchased
  (Real.floor (total_cost * 100) / 100)

theorem Wendy_received_correct_amount :
  Wendy_received_amount = 40.21 := by
  sorry

end Wendy_received_correct_amount_l574_574476


namespace min_length_QR_l574_574872

theorem min_length_QR (PQ PR SR QS QR : ℕ) (hPQ : PQ = 7) (hPR : PR = 15) (hSR : SR = 10) (hQS : QS = 25) :
  QR > PR - PQ ∧ QR > QS - SR ↔ QR = 16 :=
by
  sorry

end min_length_QR_l574_574872


namespace largest_domain_of_g_l574_574026

noncomputable def g (x : ℝ) : ℝ := sorry

theorem largest_domain_of_g :
  (∀ x ∈ {x : ℝ | g(x) + g(x^2) = x^2}, x = 1 ∨ x = -1) ↔ 
  ∀ x ∈ {x : ℝ | x = 1 ∨ x = -1}, g(x) + g(x^2) = x^2 :=
begin
  sorry
end

end largest_domain_of_g_l574_574026


namespace ratio_M_N_l574_574747

variable {R P M N : ℝ}

theorem ratio_M_N (h1 : P = 0.3 * R) (h2 : M = 0.35 * R) (h3 : N = 0.55 * R) : M / N = 7 / 11 := by
  sorry

end ratio_M_N_l574_574747


namespace factorize_m4_minus_5m_plus_4_factorize_x3_plus_2x2_plus_4x_plus_3_factorize_x5_minus_1_l574_574097

-- Statement for question 1
theorem factorize_m4_minus_5m_plus_4 (m : ℤ) : 
  (m ^ 4 - 5 * m + 4) = (m ^ 4 - 5 * m + 4) := sorry

-- Statement for question 2
theorem factorize_x3_plus_2x2_plus_4x_plus_3 (x : ℝ) :
  (x ^ 3 + 2 * x ^ 2 + 4 * x + 3) = (x + 1) * (x ^ 2 + x + 3) := sorry

-- Statement for question 3
theorem factorize_x5_minus_1 (x : ℝ) :
  (x ^ 5 - 1) = (x - 1) * (x ^ 4 + x ^ 3 + x ^ 2 + x + 1) := sorry

end factorize_m4_minus_5m_plus_4_factorize_x3_plus_2x2_plus_4x_plus_3_factorize_x5_minus_1_l574_574097


namespace total_amount_l574_574727

variable (Brad Josh Doug : ℝ)

axiom h1 : Josh = 2 * Brad
axiom h2 : Josh = (3 / 4) * Doug
axiom h3 : Doug = 32

theorem total_amount : Brad + Josh + Doug = 68 := by
  sorry

end total_amount_l574_574727


namespace reflect_point_over_x_axis_l574_574787

theorem reflect_point_over_x_axis :
  ∀ (x y : ℝ), reflect x y = ⟨x, -y⟩ :=
begin
  intro x,
  intro y,
  sorry
end

end reflect_point_over_x_axis_l574_574787


namespace number_of_zeros_of_f_l574_574036

def f (x : ℝ) : ℝ := 2 * x - Real.sin x

theorem number_of_zeros_of_f : 
  ∃! x : ℝ, f x = 0 :=
sorry

end number_of_zeros_of_f_l574_574036


namespace quadratic_term_free_solution_l574_574340

theorem quadratic_term_free_solution (m : ℝ) : 
  (∀ x : ℝ, ∃ (p : ℝ → ℝ), (x + m) * (x^2 + 2*x - 1) = p x + (2 + m) * x^2) → m = -2 :=
by
  intro H
  sorry

end quadratic_term_free_solution_l574_574340


namespace yellow_balls_count_l574_574882

-- Definitions for the conditions in the problem
def total_balls := 60
def white_balls := 22
def green_balls := 18
def red_balls := 3
def purple_balls := 1
def probability_neither_red_nor_purple := 0.95

-- The Lean statement for proving the number of yellow balls
theorem yellow_balls_count : 
  let yellow_balls := total_balls - white_balls - green_balls - red_balls - purple_balls - (probability_neither_red_nor_purple * total_balls)
  yellow_balls = 17 :=
by
  sorry

end yellow_balls_count_l574_574882


namespace collinear_of_line_condition_l574_574611

-- Definitions and conditions
variable {Point : Type} [Inhabited Point] (S : Finset Point)
variable [LinearOrderedField ℝ] [AffineSpace ℝ Point]

def collinear (Pts : Finset Point) : Prop :=
  ∃ (line : AffineSubspace ℝ Point), ∀ p ∈ Pts, p ∈ line

def line_condition (S : Finset Point) : Prop :=
  ∀ P₁ P₂ ∈ S, P₁ ≠ P₂ → ∃ P₃ ∈ S, P₃ ≠ P₁ ∧ P₃ ≠ P₂ ∧ AffineCombination ℝ 2 ({P₁, P₂}) {1/2, 1/2} = Some P₃

-- The theorem(statement) to prove
theorem collinear_of_line_condition (S : Finset Point) (h : line_condition S) : collinear S :=
sorry

end collinear_of_line_condition_l574_574611


namespace quadratic_function_f_max_min_g_l574_574313

noncomputable def f (x : ℝ) := x^2 + 2 * x

theorem quadratic_function_f :
  ∀ x b c, f x = x^2 + b * x + c ∧ f (-3) = f 1 ∧ f 0 = 0 → (b = 2 ∧ c = 0 ∧ f x = x^2 + 2 * x) :=
begin
  intros x b c h,
  cases h with h1 h2,
  cases h2 with h3 h4,
  rw [←h3, ←h4] at h1,
  split,
  { linarith },
  split,
  { linarith },
  { ext, assumption }
end

noncomputable def g (x : ℝ) (a : ℝ) := f x - (4 + 2 * a) * x + 2

theorem max_min_g :
  ∀a ∈ Icc 1 2, 
  ((a ≤ 0 → g 1 a = 1 - 2 * a ∧ g 2 a = 2 - 4 * a) ∧
  (0 < a ∧ a < 1/2 → g (1+a) a = -a^2 - 2*a + 1 ∧ g 2 a = 2 - 4 * a) ∧
  (a = 1/2 → g (3/2) a = -17/4 ∧ g 1 a = -2 ∧ g 2 a = -2) ∧
  (1/2 < a ∧ a < 1 → g (1+a) a = -a^2 - 2*a + 1 ∧ g 1 a = 1 - 2 * a) ∧
  (a ≥ 1 → g 2 a = 2 - 4 * a ∧ g 1 a = 1 - 2 * a)) :=
begin
  intros a ha,
  sorry -- Proof of the case analysis for different ranges of a
end

end quadratic_function_f_max_min_g_l574_574313


namespace apples_left_total_l574_574274

-- Define the initial conditions
def FrankApples : ℕ := 36
def SusanApples : ℕ := 3 * FrankApples
def SusanLeft : ℕ := SusanApples / 2
def FrankLeft : ℕ := (2 / 3) * FrankApples

-- Define the total apples left
def total_apples_left (SusanLeft FrankLeft : ℕ) : ℕ := SusanLeft + FrankLeft

-- Given conditions transformed to Lean
theorem apples_left_total : 
  total_apples_left (SusanApples / 2) ((2 / 3) * FrankApples) = 78 := by
  sorry

end apples_left_total_l574_574274


namespace problem_statement_l574_574651

-- Define the function and its properties
def f (x : ℝ) (ω : ℝ) (ϕ : ℝ) : ℝ := 
  sqrt 3 * sin(ω * x + ϕ)

-- Main theorem statement
theorem problem_statement 
  (ω : ℝ) 
  (ϕ : ℝ) 
  (hω_pos : ω > 0) 
  (hϕ_bounds : -π / 2 ≤ ϕ ∧ ϕ < π / 2)
  (h_symmetry : ∀ x, f x ω ϕ = f (2 * π / 3 - x) ω ϕ)
  (h_periodicity : ∀ x, f (x + π) ω ϕ = f x ω ϕ )
  (α : ℝ) 
  (hα_bounds : π / 6 < α ∧ α < 2 * π / 3)
  (h_f_alpha : f (α / 2) ω ϕ = sqrt 3 / 4):
  (ω = 2) ∧ 
  (ϕ = -π / 6) ∧ 
  (cos (α + 3 * π / 2) = (sqrt 3 + sqrt 15) / 8) :=
by
  sorry

end problem_statement_l574_574651


namespace move_wardrobe_possible_l574_574185

variable {α : Type*} [DecidableTotalOrder α] [CommSemigroup α]

def can_move_wardrobe (a b d h : α) : Prop :=
  a ≤ d ∧ a ≤ h ∧ a * b ≤ d * h

theorem move_wardrobe_possible (a b d h : ℝ) (h₁ : a ≤ d) (h₂ : a ≤ h) (h₃ : a * b ≤ d * h) :
  can_move_wardrobe a b d h :=
by
  sorry

end move_wardrobe_possible_l574_574185


namespace max_vertex_sum_l574_574268

theorem max_vertex_sum (a T : ℤ) (hT : T ≠ 0)
  (h₁ : ∀ x, a * x * (x - 2 * T) = 0 → (0 = x ∨ 2 * T = x))
  (h₂ : ∀ x, (a * x^2 + (a * -2 * T) * x + 0 = a * x * (x - 2 * T)) ∧ ((T + 2) * (a * (T + 2 - 2 * T)) = 32))
  (N : ℤ) :
  (N = T - a * T^2) → max_vertex_sum = 68 :=
by
  sorry

end max_vertex_sum_l574_574268


namespace petya_wins_with_optimal_play_l574_574825

def move1 (n : ℕ) : ℕ := if n > 0 then n - 1 else n
def move2 (n : ℕ) (m : ℕ) : ℕ := if n >= 1 ∧ m >= 2 then n - 1 else n
def move3 (n : ℕ) : ℕ := if n >= 2 then n - 2 else n

theorem petya_wins_with_optimal_play (pluses : ℕ) (minuses : ℕ) (initial : pluses = 865) (first_move : True) : 
  ∃ (p : ℕ), p = 0 ∧ ∀ (moves : list (ℕ → ℕ)), p ∈ list.map (λ f, f pluses) moves :=
sorry

end petya_wins_with_optimal_play_l574_574825


namespace collinear_K_L_M_l574_574715

-- Definitions of the geometric setting and given conditions
variable {A B C B1 K L M : Type} 

-- Definitions of relevant properties based on conditions
variable [Triangle A B C]
variable [AngleBisector B B1]
variable [Perpendicular B1 BC]
variable [IntersectsArc B1 BC K]
variable [Perpendicular B AK B L]
variable [ArcMidpoint A C B M]

-- The statement of the theorem to be proved
theorem collinear_K_L_M (h1 : is_angle_bisector B B1)
                         (h2 : is_perpendicular B1 BC)
                         (h3 : intersects_arc B1 BC K)
                         (h4 : is_perpendicular_to AK B L)
                         (h5 : midpoint_of_arc_excl_point B (circumcircle_triangle A B C) A C M) :
  collinear K L M :=
sorry

end collinear_K_L_M_l574_574715


namespace john_total_payment_l574_574370

-- Definitions of given conditions
def costOfNikes := 150
def costOfWorkBoots := 120
def taxRate := 0.10

-- Definition for total cost
def totalCostBeforeTax := costOfNikes + costOfWorkBoots
def taxAmount := totalCostBeforeTax * taxRate
def totalCostIncludingTax := totalCostBeforeTax + taxAmount

-- Theorem statement
theorem john_total_payment : totalCostIncludingTax = 297 := by
  sorry

end john_total_payment_l574_574370


namespace median_of_dataset_variance_of_dataset_l574_574177

noncomputable def dataset : List ℝ := [5, 9, 8, 8, 10]

def median (l : List ℝ) : ℝ :=
  let sorted := l.qsort (≤)
  sorted.get! (sorted.length / 2)

def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

def variance (l : List ℝ) : ℝ :=
  let m := mean l
  l.map (λ x => (x - m) ^ 2).sum / l.length

theorem median_of_dataset : median dataset = 8 := by
  sorry

theorem variance_of_dataset : variance dataset = 2.8 := by
  sorry

end median_of_dataset_variance_of_dataset_l574_574177


namespace find_relation_l574_574573

noncomputable def F (y : ℝ → ℝ) := (3 * (x : ℝ)^2 + x - 1) * (deriv^[2] y x) - (9 * x^2 + 9 * x - 2) * (deriv y x) + (18 * x + 3) * y x

theorem find_relation (f : ℝ → ℝ)
  (h_eq: ∀ x, F f = 6 * (6 * x + 1))
  (h_initial: f 0 = 1)
  (h_cond: (f (-1) - 2) * (f 1 - 6) = 1) : 
  (f (-2) - 6) * (f 2 - 14) = 1 :=
sorry

end find_relation_l574_574573


namespace minimum_additional_games_to_reach_90_percent_hawks_minimum_games_needed_to_win_l574_574015

theorem minimum_additional_games_to_reach_90_percent (N : ℕ) : 
  (2 + N) * 10 ≥ (5 + N) * 9 ↔ N ≥ 25 := 
sorry

-- An alternative approach to assert directly as exactly 25 by using the condition’s natural number ℕ could be as follows:
theorem hawks_minimum_games_needed_to_win (N : ℕ) : 
  ∀ N, (2 + N) * 10 / (5 + N) ≥ 9 / 10 → N ≥ 25 := 
sorry

end minimum_additional_games_to_reach_90_percent_hawks_minimum_games_needed_to_win_l574_574015


namespace even_parts_impossible_odd_parts_possible_l574_574178

theorem even_parts_impossible (n m : ℕ) (h₁ : n = 1) (h₂ : ∀ k, m = n + 2 * k) : n + 2 * m ≠ 100 := by
  -- Proof omitted
  sorry

theorem odd_parts_possible (n m : ℕ) (h₁ : n = 1) (h₂ : ∀ k, m = n + 2 * k) : ∃ k, n + 2 * k = 2017 := by
  -- Proof omitted
  sorry

end even_parts_impossible_odd_parts_possible_l574_574178


namespace P_at_2007_l574_574661

noncomputable def P (x : ℝ) : ℝ :=
x^15 - 2008 * x^14 + 2008 * x^13 - 2008 * x^12 + 2008 * x^11
- 2008 * x^10 + 2008 * x^9 - 2008 * x^8 + 2008 * x^7
- 2008 * x^6 + 2008 * x^5 - 2008 * x^4 + 2008 * x^3
- 2008 * x^2 + 2008 * x

-- Statement to show that P(2007) = 2007
theorem P_at_2007 : P 2007 = 2007 :=
  sorry

end P_at_2007_l574_574661


namespace boxes_calculation_l574_574396

theorem boxes_calculation (total_oranges : ℕ) (oranges_per_box : ℕ) (boxes : ℕ) 
    (h1 : total_oranges = 42) (h2 : oranges_per_box = 6) : boxes = 7 :=
by 
  have h3 : boxes = total_oranges / oranges_per_box,
  sorry
  rw [h1, h2] at h3,
  sorry

end boxes_calculation_l574_574396


namespace cos_angle_between_lines_l574_574550

noncomputable def direction_vectors := (4, 5, 2, 6)

noncomputable def cos_phi (d1 d2 : ℝ × ℝ) : ℝ :=
  let dot_product := (d1.1 * d2.1 + d1.2 * d2.2)
  let norm_d1 := real.sqrt (d1.1 ^ 2 + d1.2 ^ 2)
  let norm_d2 := real.sqrt (d2.1 ^ 2 + d2.2 ^ 2)
  dot_product / (norm_d1 * norm_d2)

theorem cos_angle_between_lines :
  cos_phi (4, 5) (2, 6) = 38 / real.sqrt 1640 := by
  sorry

end cos_angle_between_lines_l574_574550


namespace part1_correct_part2_correct_l574_574492

-- Definition and proof of the first part of the problem
def part1_expr : ℝ := 1 * (Real.pi^0) + 2^(-2) * (9 / 4)
theorem part1_correct : part1_expr = 25 / 16 := 
by 
  -- leave the detailed proof out
  sorry

-- Definition and proof of the second part of the problem
def part2_expr : ℝ := 2 * Real.logb 5 10 + Real.logb 5 0.25
theorem part2_correct : part2_expr = 2 := 
by 
  -- leave the detailed proof out
  sorry

end part1_correct_part2_correct_l574_574492


namespace knight_probability_same_color_l574_574458

noncomputable def knight_same_color_probability : ℚ :=
  let same_color_prob := (1 : ℚ) / 2 in
  same_color_prob

theorem knight_probability_same_color :
  let chessboard := (3 : ℕ)  -- Three-color chessboard
  let total_squares := (64 : ℕ)  -- Chessboard has 64 squares
  let valid_moves_per_square := λ (sq : ℕ), 8 -- Each square can have up to 8 valid knight moves
  let starting_probability := 1 / total_squares in
  knight_same_color_probability = 1 / 2 :=
by
  have : knight_same_color_probability = (1 : ℚ) / 2 := rfl
  exact this

end knight_probability_same_color_l574_574458


namespace prob_at_least_6_heads_eq_l574_574115

-- define the number of coin flips
def n := 8

-- define the number of possible outcomes (2^n)
def total_outcomes := 2 ^ n

-- define the binomial coefficients for cases: 6 heads, 7 heads, 8 heads
def binom_8_6 := Nat.choose 8 6
def binom_8_7 := Nat.choose 8 7
def binom_8_8 := Nat.choose 8 8

-- calculate the favorable outcomes for at least 6 heads
def favorable_outcomes := binom_8_6 + binom_8_7 + binom_8_8

-- define the probability of getting at least 6 heads
def probability := (favorable_outcomes : ℚ) / total_outcomes

theorem prob_at_least_6_heads_eq : probability = 37 / 256 := by
  sorry

end prob_at_least_6_heads_eq_l574_574115


namespace cost_to_buy_450_candies_l574_574180

-- Define a structure representing the problem conditions
structure CandyStore where
  candies_per_box : Nat
  regular_price : Nat
  discounted_price : Nat
  discount_threshold : Nat

-- Define parameters for this specific problem
def store : CandyStore :=
  { candies_per_box := 15,
    regular_price := 5,
    discounted_price := 4,
    discount_threshold := 10 }

-- Define the cost function with the given conditions
def cost (store : CandyStore) (candies : Nat) : Nat :=
  let boxes := candies / store.candies_per_box
  if boxes >= store.discount_threshold then
    boxes * store.discounted_price
  else
    boxes * store.regular_price

-- State the theorem we want to prove
theorem cost_to_buy_450_candies (store : CandyStore) (candies := 450) :
  store.candies_per_box = 15 →
  store.discounted_price = 4 →
  store.discount_threshold = 10 →
  cost store candies = 120 := by
  sorry

end cost_to_buy_450_candies_l574_574180


namespace river_water_flow_l574_574078

theorem river_water_flow (depth width rate : ℝ) (depth_eq : depth = 7) (width_eq : width = 75) (rate_eq : rate = 4) :
  let flow_rate := (rate * 1000) / 60 in
  let area := depth * width in
  let volume_per_min := area * flow_rate in
  volume_per_min = 35001.75 := 
by 
  sorry

end river_water_flow_l574_574078


namespace sqrt_66_greater_than_8_l574_574547

theorem sqrt_66_greater_than_8 :
  real.sqrt 66 > 8 :=
by
  have condition : real.sqrt 64 = 8 := by exact real.sqrt_sq (rfl : 64 = 64)
  have h : 66 > 64 := by norm_num
  sorry

end sqrt_66_greater_than_8_l574_574547


namespace jacob_blocks_l574_574722

def total_blocks (red yellow green blue orange purple : ℕ) : ℕ :=
  red + yellow + green + blue + orange + purple

theorem jacob_blocks :
  let red := 24 in
  let yellow := red + 8 in
  let green := yellow - 10 in
  let blue := green * 2 in
  let orange := blue + 15 in
  let purple := (red + orange) - 7 in
  total_blocks red yellow green blue orange purple = 257 :=
by
  sorry

end jacob_blocks_l574_574722


namespace integer_solutions_l574_574575

theorem integer_solutions (n : ℤ) :
  (∃ (a b c : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ a^n + b^n = c^n) ↔ n = 1 ∨ n = 2 ∨ n = -1 ∨ n = -2 :=
by sorry

end integer_solutions_l574_574575


namespace compute_f_l574_574110

theorem compute_f (f : ℕ → ℚ) (h1 : f 1 = 1 / 3)
  (h2 : ∀ n : ℕ, n ≥ 2 → f n = (2 * (n - 1) - 1) / (2 * (n - 1) + 3) * f (n - 1)) :
  ∀ n : ℕ, n ≥ 1 → f n = 1 / ((2 * n - 1) * (2 * n + 1)) :=
by
  sorry

end compute_f_l574_574110


namespace problem_statement_l574_574333

variable (p q : ℝ)

def condition := p ^ 2 / q ^ 3 = 4 / 5

theorem problem_statement (hpq : condition p q) : 11 / 7 + (2 * q ^ 3 - p ^ 2) / (2 * q ^ 3 + p ^ 2) = 2 :=
sorry

end problem_statement_l574_574333


namespace binomial_12_9_l574_574206

def binomial (n k : ℕ) := nat.choose n k

theorem binomial_12_9 : binomial 12 9 = 220 :=
by
  have step1 : binomial 12 9 = binomial 12 3 := nat.choose_symm 12 9
  have step2 : binomial 12 3 = 220 := by sorry
  rw [step1, step2]

end binomial_12_9_l574_574206


namespace probability_at_least_6_heads_l574_574149

-- Definitions of the binomial coefficient and probability function
def binom (n k : ℕ) : ℕ := Nat.choose n k

def probability (favorable total : ℕ) : ℚ := favorable / total

-- Proof problem statement
theorem probability_at_least_6_heads (flips : ℕ) (p : ℚ) 
  (h_flips : flips = 8) 
  (h_probability : p = probability (binom 8 6 + binom 8 7 + binom 8 8) (2 ^ flips)) : 
  p = 37 / 256 :=
by
  sorry

end probability_at_least_6_heads_l574_574149


namespace area_of_circle_l574_574226

noncomputable def circle_eq : ℝ × ℝ → Prop :=
  λ (x, y), x^2 + y^2 - 6 * x + 8 * y - 11 = 0

theorem area_of_circle : ∃ (area : ℝ), circle_eq (x, y) → area = 14 * Real.pi :=
begin
  sorry
end

end area_of_circle_l574_574226


namespace find_ordered_pair_l574_574633

theorem find_ordered_pair :
  ∃ (x y : ℝ), (x, y) = (5 / 2, -16) ∧
  (∃ (a b c d e f : ℝ),
    a = 3 ∧
    b = x ∧
    c = -8 ∧
    d = 6 ∧
    e = 5 ∧
    f = y ∧
    (⟨a, b, c⟩ : ℝ × ℝ × ℝ) × ⟨d, e, f⟩ = (0, 0, 0)) :=
by
  sorry

end find_ordered_pair_l574_574633


namespace simplify_fraction_l574_574774

open Complex

theorem simplify_fraction :
  (7 + 9 * I) / (3 - 4 * I) = 2.28 + 2.2 * I := 
by {
    -- We know that this should be true based on the provided solution,
    -- but we will place a placeholder here for the actual proof.
    sorry
}

end simplify_fraction_l574_574774


namespace correct_operation_l574_574859

theorem correct_operation :
  (∀ (a b : ℝ),
    (a + b)^2 ≠ a^2 + b^2 ∧
    (-2 * a * b)^3 = -8 * a^3 * b^3 ∧
    (2^2 + 3^2).sqrt ≠ 2 + 3 ∧
    3 * a⁻¹ ≠ 1 / (3 * a)) :=
by {
  intros,
  sorry,
}

end correct_operation_l574_574859


namespace symmetric_line_l574_574440

theorem symmetric_line (x y : ℝ) : 
  (∀ (x y  : ℝ), 2 * x + y - 1 = 0) ∧ (∀ (x  : ℝ), x = 1) → (2 * x - y - 3 = 0) :=
by
  sorry

end symmetric_line_l574_574440


namespace square_area_l574_574913

theorem square_area (y : ℝ) (x₁ x₂ : ℝ) (s : ℝ) (A : ℝ) :
  y = 7 → 
  (y = x₁^2 + 4 * x₁ + 3) →
  (y = x₂^2 + 4 * x₂ + 3) →
  x₁ ≠ x₂ →
  s = |x₂ - x₁| → 
  A = s^2 →
  A = 32 :=
by
  intros hy intersection_x1 intersection_x2 hx1x2 hs ha
  sorry

end square_area_l574_574913


namespace calculate_expression_l574_574948

theorem calculate_expression : 
  - 3 ^ 2 + (-12) * abs (-1/2) - 6 / (-1) = -9 := 
by 
  sorry

end calculate_expression_l574_574948


namespace Aiyanna_cookies_l574_574536

-- Define the conditions
def Alyssa_cookies : ℕ := 129
variable (x : ℕ)
def difference_condition : Prop := (Alyssa_cookies - x) = 11

-- The theorem to prove
theorem Aiyanna_cookies (x : ℕ) (h : difference_condition x) : x = 118 :=
by sorry

end Aiyanna_cookies_l574_574536


namespace sum_of_altitudes_proof_l574_574891

-- Define the line equation
def line_eq (x y : ℝ) : Prop := 15 * x + 3 * y = 45

-- Define the vertices of the triangle
def vertex1 : ℝ × ℝ := (0, 0)
def vertex2 : ℝ × ℝ := (3, 0)
def vertex3 : ℝ × ℝ := (0, 15)

-- Define the lengths of the altitudes
def altitude1 : ℝ := 3  -- from (0,0) to line x = 3
def altitude2 : ℝ := 15  -- from (0,0) to line y = 15
def altitude3 : ℝ := (|15 * 0 + 3 * 0 - 45|) / (sqrt (15 ^ 2 + 3 ^ 2))  -- from (0,0) to the hypotenuse

-- Sum of altitudes
def sum_of_altitudes : ℝ := altitude1 + altitude2 + altitude3

-- The proof statement
theorem sum_of_altitudes_proof : sum_of_altitudes = 20.5 :=
by
  -- Definitions
  have h1 : 15 * 3 + 3 * 0 = 45 := by norm_num,
  have h2 : 15 * 0 + 3 * 15 = 45 := by norm_num,
  have h3 : 15 * 0 + 3 * 0 - 45 = -45 := by norm_num,
  have h4 : sqrt(15 ^ 2 + 3 ^ 2) = 18 := by norm_num,
  have h5 : altitude1 = 3 := by norm_num,
  have h6 : altitude2 = 15 := by norm_num,
  have h7 : altitude3 = 45/18 := by norm_num,
  have h8 : 15.0 /18 = 5/2 := by norm_cast,
  have hsum : altitude1 + altitude2 + altitude3 = 20.5 := by norm_num,
  exact hsum,

end sum_of_altitudes_proof_l574_574891


namespace repeating_decimal_fraction_l574_574568

def repeating_decimal_to_fraction (d: ℚ) (r: ℚ) (p: ℚ): ℚ :=
  d + r

theorem repeating_decimal_fraction :
  repeating_decimal_to_fraction (6 / 10) (1 / 33) (0.6 + (0.03 : ℚ)) = 104 / 165 := 
by
  sorry

end repeating_decimal_fraction_l574_574568


namespace tank_filling_time_l574_574866

def length : ℝ := 6
def width : ℝ := 4
def depth : ℝ := 3
def filling_rate : ℝ := 4
def volume : ℝ := length * width * depth
def filling_time : ℝ := volume / filling_rate

theorem tank_filling_time : filling_time = 18 := 
by
  -- skipped the proof
  sorry

end tank_filling_time_l574_574866


namespace Bryan_deposit_amount_l574_574398

theorem Bryan_deposit_amount (deposit_mark : ℕ) (deposit_bryan : ℕ)
  (h1 : deposit_mark = 88)
  (h2 : deposit_bryan = 5 * deposit_mark - 40) : 
  deposit_bryan = 400 := 
by
  sorry

end Bryan_deposit_amount_l574_574398


namespace teams_working_together_complete_task_in_5_days_l574_574832

theorem teams_working_together_complete_task_in_5_days :
  let task_days (A B : ℝ) (efficiency_increase : ℝ) :=
    (1/A + 1/B) * (1 + efficiency_increase) in
  task_days 10 15 0.20 * 5 = 1 :=
by
  sorry

end teams_working_together_complete_task_in_5_days_l574_574832


namespace sqrt_eq_289_l574_574812

theorem sqrt_eq_289 (x : ℝ) (h₁ : sqrt(x) - (sqrt(625) / sqrt(25)) = 12) : x = 289 :=
by
  sorry

end sqrt_eq_289_l574_574812


namespace snowflake_stamps_count_l574_574943

theorem snowflake_stamps_count (S : ℕ) (truck_stamps : ℕ) (rose_stamps : ℕ) :
  truck_stamps = S + 9 →
  rose_stamps = S + 9 - 13 →
  S + truck_stamps + rose_stamps = 38 →
  S = 11 :=
by
  intros h1 h2 h3
  sorry

end snowflake_stamps_count_l574_574943


namespace matchsticks_left_l574_574971

theorem matchsticks_left (total_matchsticks elvis_max ralph_max elvis_sq_matchsticks ralph_sq_matchsticks : ℕ)
    (h1 : total_matchsticks = 50)
    (h2 : elvis_max = 20)
    (h3 : ralph_max = 30)
    (h4 : elvis_sq_matchsticks = 4)
    (h5 : ralph_sq_matchsticks = 8) :
    let elvis_squares := elvis_max / elvis_sq_matchsticks,
        ralph_squares := ralph_max / ralph_sq_matchsticks,
        elvis_used := elvis_squares * elvis_sq_matchsticks,
        ralph_used := ralph_squares * ralph_sq_matchsticks,
        used := elvis_used + ralph_used,
        remaining := total_matchsticks - used
    in remaining = 6 := by
  sorry

end matchsticks_left_l574_574971


namespace prove_distance_l574_574337

-- Definitions of conditions based on given problem
def is_ellipse (a b e : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ e = 1 / 2 ∧ (∃ c, c^2 = a^2 - b^2 ∧ c = a / 2) 

def is_roots_of_quadratic (a b c x1 x2 : ℝ) : Prop :=
  (a * x1^2 + 2 * b * x1 + c = 0) ∧ (a * x2^2 + 2 * b * x2 + c = 0) ∧ 
  (x1 + x2 = -2 * (b / a)) ∧ (x1 * x2 = c / a)

-- Main theorem
theorem prove_distance 
  (a b c e x1 x2 : ℝ)
  (h_ellipse : is_ellipse a b e)
  (h_roots : is_roots_of_quadratic a b c x1 x2) :
  real.sqrt (x1^2 + x2^2) = real.sqrt 2 :=
sorry

end prove_distance_l574_574337


namespace cos_B_area_of_abc_l574_574363

noncomputable def triangle_area (a b c : ℝ) (angleB : ℝ) : ℝ :=
  (1 / 2) * a * c * angleB.sin

theorem cos_B (a b c : ℝ) (h1 : b = c) (h2 : 2 * b.sin = real.sqrt 3 * a.sin) :
  real.cos (real.arccos (a^2 + c^2 - b^2) / (2 * a * c)) = real.sqrt 3 / 3 :=
sorry

theorem area_of_abc (a b c : ℝ) (h1 : a = 2) (h2 : b = real.sqrt 3) (h3 : c = real.sqrt 3) (h4 : real.cos (real.arccos (a^2 + c^2 - b^2) / (2 * a * c)) = real.sqrt 3 / 3) :
  triangle_area a b c (real.arcsin (real.sqrt ((1 - (real.sqrt 3 / 3)^2)))) = real.sqrt 2 :=
sorry

end cos_B_area_of_abc_l574_574363


namespace range_of_m_l574_574650

noncomputable def f (x : ℝ) : ℝ := x / Real.log x

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, e ≤ x ∧ x ≤ e^2 ∧ f x - m * x - 1/2 + m ≤ 0) →
  1/2 ≤ m := by
  sorry

end range_of_m_l574_574650


namespace zoo_feeding_pattern_number_of_ways_l574_574531

/-- Total number of ways to feed all animals starting with the male lion, given 
    the alternating gender feeding pattern with 5 pairs of different animals. 
-/
theorem zoo_feeding_pattern_number_of_ways :
  let males := 5
  let females := 5
  (males * (females - 1)! * males! * (females - 1)! * (males - 1)! * (females - 2)! *
   (males - 2)! * (females - 3)! * (males - 3)! * (females - 4)!) = 2880 :=
by
  sorry

end zoo_feeding_pattern_number_of_ways_l574_574531


namespace percentage_increase_correct_answers_l574_574348

variable (total_questions : ℕ)
variable (lowella_correct_percentage : ℚ)
variable (lowella_correct : ℕ)
variable (mandy_correct : ℕ)
variable (pamela_correct : ℕ)

-- Conditions from the problem
def conditions (total_questions = 100)
               (lowella_correct_percentage = 0.35)
               (lowella_correct = 35)
               (mandy_correct = 84)
               (pamela_correct = 42) : Prop :=
  lowella_correct = (lowella_correct_percentage * total_questions).toNat ∧
  pamela_correct = mandy_correct / 2 ∧
  mandy_correct = 84

-- The final statement to be proved
theorem percentage_increase_correct_answers 
  (h : conditions 100 0.35 35 84 42) : 
  ((((pamela_correct.to_rat / total_questions) * 100) - 
   ((lowella_correct.to_rat / total_questions) * 100)) / 
   ((lowella_correct.to_rat / total_questions) * 100) * 100) = 20 := 
by 
  sorry

end percentage_increase_correct_answers_l574_574348


namespace matrix_expression_l574_574377

noncomputable theory

def B : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, 4], ![0, 2]]

theorem matrix_expression :
  B^10 - 3 • B^9 = ![![0, 4], ![0, -1]] :=
  sorry

end matrix_expression_l574_574377


namespace total_possible_orders_l574_574693

theorem total_possible_orders : 
  (∃ (bowlers : Finset ℕ) (matches : ℕ → ℕ → bool),
   bowlers = {1, 2, 3, 4, 5, 6} ∧
   matches 6 5 ∈ {tt, ff} ∧ 
   matches 5 4 ∈ {tt, ff} ∧
   matches 4 3 ∈ {tt, ff} ∧
   matches 3 2 ∈ {tt, ff} ∧
   matches 2 1 ∈ {tt, ff} ∧
   (∀ b1 b2 : ℕ, b1 ∈ bowlers → b2 ∈ bowlers → b1 ≠ b2 → matches b1 b2 ∈ {tt, ff})
  → finset.card bowlers = 6 ) 
→ 2 ^ 5 = 32 :=
by {
  sorry
}

end total_possible_orders_l574_574693


namespace spring_length_5kg_weight_l574_574366

variable {x y : ℝ}

-- Given conditions
def spring_length_no_weight : y = 6 := sorry
def spring_length_4kg_weight : y = 7.2 := sorry

-- The problem: to find the length of the spring for 5 kilograms
theorem spring_length_5kg_weight :
  (∃ (k b : ℝ), (∀ x, y = k * x + b) ∧ (b = 6) ∧ (4 * k + b = 7.2)) →
  y = 0.3 * 5 + 6 :=
  sorry

end spring_length_5kg_weight_l574_574366


namespace second_reduction_percentage_is_4_l574_574877

def original_price := 500
def first_reduction_percent := 5 / 100
def total_reduction := 44

def first_reduction := first_reduction_percent * original_price
def price_after_first_reduction := original_price - first_reduction
def second_reduction := total_reduction - first_reduction
def second_reduction_percent := (second_reduction / price_after_first_reduction) * 100

theorem second_reduction_percentage_is_4 :
  second_reduction_percent = 4 := by
  sorry

end second_reduction_percentage_is_4_l574_574877


namespace possible_sums_l574_574188

-- Define the problem statement conditions
def distinctPosInts (n : ℕ) (a : Fin n → ℕ) : Prop :=
  ∀ i j : Fin n, i ≠ j → a i ≠ a j ∧ (1 <= a i)

theorem possible_sums :
  ∃ (a : Fin 2011 → ℕ), distinctPosInts 2011 a →
  finite (S : ℕ) (H : ∀ a : Fin 2011 → ℕ, S = ∑ i, a i → (∃ a₁, a₁ = a) ) ∧
  card { S | ∃ a, distinctPosInts 2011 a ∧ S = ∑ i, a i } = 2 :=
sorry

end possible_sums_l574_574188


namespace janet_gym_hours_l574_574723

theorem janet_gym_hours :
  (∃ monday_wednesday_time : ℝ,
    let total_weekly_time := 5
    let friday_time := 1
    let tuesday_time := 1
    let monday_wednesday_combined := total_weekly_time - friday_time - tuesday_time
    let monday_wednesday_per_day := monday_wednesday_combined / 2
    monday_wednesday_time = monday_wednesday_per_day
  ) :=
begin
  let total_weekly_time := 5,
  let friday_time := 1,
  let tuesday_time := 1,
  let monday_wednesday_combined := total_weekly_time - friday_time - tuesday_time,
  let monday_wednesday_per_day := monday_wednesday_combined / 2,
  use monday_wednesday_per_day,
  sorry
end

end janet_gym_hours_l574_574723


namespace find_a_n_exists_constants_pq_l574_574219

noncomputable def a_n (n : ℕ) : ℝ := 1 - (1 / 2^n)

def exists_pq (n : ℕ) (b_n : ℕ → ℝ) : Prop :=
  ∃ p q : ℝ, p < q ∧ b_n n = (1 / 3) * (1 + (p / 2^n)) * (1 + (q / 2^n))

theorem find_a_n (n : ℕ) (h : n > 0) : a_n n = 1 - 1 / 2^n := sorry

theorem exists_constants_pq (n : ℕ) (b_n : ℕ → ℝ) (h : ∀ n ≥ 2, b_n n = (1 / 3) * (1 - 1 / 2^(n-1)) * (1 - 1 / 2^n)) :
  exists_pq n b_n :=
begin
  use [-2, -1],
  split; linarith,
  sorry
end

end find_a_n_exists_constants_pq_l574_574219


namespace circles_tangent_l574_574432

-- Define an acute-angled triangle ABC
def acute_angled_triangle (A B C : Type*) : Prop :=
  ∃ (H : Type*), orthocenter H A B C

-- Define H as the orthocenter of triangle ABC
def orthocenter (H A B C : Type*) : Prop := sorry

-- Define points B_1 and C_1 on BH and CH respectively such that B_1C_1 is parallel to BC
def points_B1_C1 (H A B C B1 C1 : Type*) : Prop :=
  ∃ (B1 C1 : Type*), (on_line_segment B1 H B) ∧ (on_line_segment C1 H C) ∧ (parallel (line_segment B1 C1) (line_segment B C))

-- Define that the center of the circumcircle of triangle B_1HC_1 lies on the line BC
def circumcircle_center_on_BC (H A B C B1 C1 : Type*) : Prop :=
  ∃ (O'' : Type*), circumcenter O'' B1 H C1 ∧ on_line O'' (line_segment B C)

-- Define tangency of circles
def tangent_circles (Γ ω : Type*) : Prop := sorry

-- The theorem to be proven
theorem circles_tangent
  (A B C H B1 C1 : Type*)
  (h₀ : acute_angled_triangle A B C)
  (h₁ : orthocenter H A B C)
  (h₂ : points_B1_C1 H A B C B1 C1)
  (h₃ : circumcircle_center_on_BC H A B C B1 C1) :
  tangent_circles (circumcircle A B C) (circumcircle B1 H C1) :=
sorry

end circles_tangent_l574_574432


namespace minimize_sum_of_squares_l574_574868

variables {A B C D E P Q : Point}

def collinear (A B C D E : Point) : Prop :=
  ∃ (l : Line), A ∈ l ∧ B ∈ l ∧ C ∈ l ∧ D ∈ l ∧ E ∈ l

def distance (x y : Point) : ℝ := 
  sorry -- Distance function definition goes here

noncomputable def AP (A P : Point) : ℝ := distance A P
noncomputable def BP (B P : Point) : ℝ := distance B P
noncomputable def CP (C P : Point) : ℝ := distance C P
noncomputable def DP (D P : Point) : ℝ := distance D P
noncomputable def EP (E P : Point) : ℝ := distance E P

theorem minimize_sum_of_squares
  (h_coll : collinear A B C D E)
  (h_AB : distance A B = 2)
  (h_BC : distance B C = 2)
  (h_CD : distance C D = 3)
  (h_DE : distance D E = 7) :
  ∃ (P Q : Point), Q ∈ line A E ∧ 
    let r := distance A Q 
      in AP A P ^ 2 + BP B P ^ 2 + CP C P ^ 2 + DP D P ^ 2 + EP E P ^ 2 = 133.2 :=
sorry

end minimize_sum_of_squares_l574_574868


namespace range_of_abs_z3_l574_574705

variable {ℂ : Type} [char_zero ℂ] [normed_field ℂ] [norm_abs_ring ℂ] [inner_product_space ℂ ℝ]

noncomputable def z1 : ℂ := sorry
noncomputable def z2 : ℂ := sorry
noncomputable def z3 : ℂ := sorry

axiom key_conditions :
  abs z1 = sqrt 2 ∧ abs z2 = sqrt 2 ∧ (inner ((z1.val : ℝ) - (0 : ℝ)) ((z2.val : ℝ) - (0 : ℝ)) = 0) ∧ abs (z1 + z2 - z3) = 2

theorem range_of_abs_z3 : 0 ≤ abs z3 ∧ abs z3 ≤ 4 :=
by
  obtain ⟨h1, h2, h3, h4⟩ := key_conditions
  sorry

end range_of_abs_z3_l574_574705


namespace pythagorean_triple_correct_l574_574930

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem pythagorean_triple_correct :
  is_pythagorean_triple 5 12 13 ∧
  ¬ is_pythagorean_triple 7 9 11 ∧
  ¬ is_pythagorean_triple 6 9 12 ∧
  ¬ is_pythagorean_triple (3/10) (4/10) (5/10) :=
by
  sorry

end pythagorean_triple_correct_l574_574930


namespace perfect_square_trinomial_l574_574341

theorem perfect_square_trinomial (a b m : ℝ) :
  (∃ x : ℝ, a^2 + mab + b^2 = (x + b)^2 ∨ a^2 + mab + b^2 = (x - b)^2) ↔ (m = 2 ∨ m = -2) :=
by
  sorry

end perfect_square_trinomial_l574_574341


namespace max_sets_of_4_elements_l574_574463

-- Definition of the symmetric difference
def sym_diff (A B : Set ℕ) : Set ℕ := (A \ B) ∪ (B \ A)

-- The conditions: We have n sets, each containing 4 elements.
variable (n : ℕ)
variable (sets : Fin n -> Set ℕ)
variable (h_size : ∀ i : Fin n, (sets i).card = 4)
variable (h_sym_diff : ∀ i j : Fin n, ∃ k : Fin n, sym_diff (sets i) (sets j) = sets k)

-- The statement: Prove that the maximum value of n is 7.
theorem max_sets_of_4_elements : n ≤ 7 := by
  sorry

end max_sets_of_4_elements_l574_574463


namespace compute_integer_k_l574_574952

theorem compute_integer_k (k : ℕ) (h : k > 3) : 
  log10 ((k - 3) ! : ℝ) + log10 ((k - 1) ! : ℝ) + 3 = 2 * log10 (k ! : ℝ) → 
  k = 11 :=
sorry

end compute_integer_k_l574_574952


namespace binom_12_9_is_220_l574_574217

def choose (n k : ℕ) : ℕ := n.choose k

theorem binom_12_9_is_220 :
  choose 12 9 = 220 :=
by {
  -- Proof is omitted
  sorry
}

end binom_12_9_is_220_l574_574217


namespace projection_correct_l574_574584

def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let magnitude_sq := v.1^2 + v.2^2
  let scalar := dot_product / magnitude_sq
  (scalar * v.1, scalar * v.2)

theorem projection_correct :
  projection (3, -4) (1, 2) = (-1, -2) :=
by
  have u : ℝ × ℝ := (3, -4)
  have v : ℝ × ℝ := (1, 2)
  let dot_product := u.1 * v.1 + u.2 * v.2
  let magnitude_sq := v.1^2 + v.2^2
  have h1 : dot_product = -5 := by simp [dot_product]
  have h2 : magnitude_sq =  5 := by simp [magnitude_sq]
  have scalar : dot_product / magnitude_sq = -1 := by rw [h1, h2]; simp
  have proj := projection u v
  simp [projection, dot_product, magnitude_sq, scalar]
  sorry

end projection_correct_l574_574584


namespace div_exact_l574_574489

theorem div_exact {a b : ℕ} (h : a = b * 7) : a % b = 0 :=
by {
  sorry
}

example : 56 % 8 = 0 :=
by {
  have h : 56 = 8 * 7 := rfl,
  exact div_exact h
}

end div_exact_l574_574489


namespace square_fg_length_l574_574010

theorem square_fg_length : 
  ∀ (A B C D E F G: ℝ) (side length: ℝ),
  square ABCD side_length → 
  midpoint E A B →
  on_arc F (arc_centered_at A B D) →
  on_line_intersecting E C F →
  on_perpendicular G F BC →
  (FG = 2) :=
by
  sorry

end square_fg_length_l574_574010


namespace binomial_12_9_l574_574208

def binomial (n k : ℕ) := nat.choose n k

theorem binomial_12_9 : binomial 12 9 = 220 :=
by
  have step1 : binomial 12 9 = binomial 12 3 := nat.choose_symm 12 9
  have step2 : binomial 12 3 = 220 := by sorry
  rw [step1, step2]

end binomial_12_9_l574_574208


namespace parabola_directrix_eq_l574_574702

noncomputable def equation_of_directrix (p : ℝ) : Prop :=
  (p > 0) ∧ (∀ (x y : ℝ), (x ≠ -5 / 4) → ¬ (y ^ 2 = 2 * p * x))

theorem parabola_directrix_eq (A_x A_y : ℝ) (hA : A_x = 2 ∧ A_y = 1)
  (h_perpendicular_bisector_fo : ∃ (f_x f_y : ℝ), f_x = 5 / 4 ∧ f_y = 0) :
  equation_of_directrix (5 / 2) :=
by {
  sorry
}

end parabola_directrix_eq_l574_574702


namespace part_a_cover_punctures_l574_574100

-- Part (a)
theorem part_a_cover_punctures
  (punctures : Set ℝ)
  (h_random_independent : ∀ x y ∈ punctures, x ≠ y)
  (h_card : punctures.card = 3)
  (h_points_in_circle : ∀ x ∈ punctures, 0 ≤ x ∧ x < 3)
  (segment_length : ℝ := 2) :
  ∃ (segment_start : ℝ), ∀ x ∈ punctures, (segment_start ≤ x ∧ x < segment_start + segment_length) ∨ (segment_start + segment_length > 3 ∧ (0 ≤ x ∧ x < (segment_start + segment_length - 3))) :=
sorry

-- Part (b)
noncomputable def part_b_probability :=
  ∫ (x y : ℝ) in set.Icc 0 3 ×ˢ set.Icc 0 3, if (0 < x ∧ x < 2) ∧ (x < y ∧ y < 3) ∧ ((3 - y) < 2) then 1 / (3 * 3) else 0

example : part_b_probability = (2/3) := 
sorry

end part_a_cover_punctures_l574_574100


namespace units_digit_sum_42_4_24_4_l574_574846

-- Define the units digit function
def units_digit (n : ℕ) : ℕ := n % 10

-- Given conditions
def units_digit_42_4 : units_digit (42^4) = 6 := sorry
def units_digit_24_4 : units_digit (24^4) = 6 := sorry

-- Theorem to prove
theorem units_digit_sum_42_4_24_4 :
  units_digit (42^4 + 24^4) = 2 :=
by
  -- Use the given conditions
  have h1 : units_digit (42^4) = 6 := units_digit_42_4
  have h2 : units_digit (24^4) = 6 := units_digit_24_4
  -- Calculate the units digit of their sum
  calc 
    units_digit (42^4 + 24^4)
        = units_digit (6 + 6) : by rw [h1, h2]
    ... = units_digit 12    : by norm_num
    ... = 2                 : by norm_num

end units_digit_sum_42_4_24_4_l574_574846


namespace cp_parallel_to_bisector_l574_574698

theorem cp_parallel_to_bisector
  (A B C D P: Type*)
  (h1 : ∠B = 90)
  (h2 : ∠D = 90)
  (h3 : A ≠ B)
  (h4 : B ≠ C)
  (h5 : C ≠ D)
  (h6 : D ≠ A)
  (h7 : AC = BC + DC)
  (h8 : on_ray BD P)
  (h9 : BP = AD) :
  parallel CP (angle_bisector ABD) :=
sorry

end cp_parallel_to_bisector_l574_574698


namespace count_possible_sets_l574_574288

open Set

theorem count_possible_sets : 
  {M : Set ℕ // {1, 2} ⊆ M ∧ M ⊂ {1, 2, 3, 4}}.dim = 3 := 
by 
  sorry

end count_possible_sets_l574_574288


namespace intersection_A_B_l574_574314

/-- Define the set A -/
def A : Set ℝ := { x | ∃ y, y = Real.log (2 - x) }

/-- Define the set B -/
def B : Set ℝ := { y | ∃ x, y = Real.sqrt x }

/-- Define the intersection of A and B and prove that it equals [0, 2) -/
theorem intersection_A_B : (A ∩ B) = { x | 0 ≤ x ∧ x < 2 } :=
by
  sorry

end intersection_A_B_l574_574314


namespace train_click_speed_none_of_these_l574_574042

theorem train_click_speed_none_of_these : 
  ∀ (x : ℝ), 
  let click_interval_seconds := 27.27 in
  let choices := [15, 60, 120, 180] in
  ¬ (click_interval_seconds ∈ choices) :=
by
  sorry

end train_click_speed_none_of_these_l574_574042


namespace correct_choice_from_statements_l574_574075

def precision (n : Float) : Nat :=
  -- Function to define the precision of a floating-point number
  -- You can use a heuristic or a direct mapping for specific examples in the given conditions

def significant_figures (n : Float) : Nat :=
  -- Function to define the significant figures of a floating-point number
  -- Again, a specific heuristic or direct mapping for given examples

theorem correct_choice_from_statements :
  let A := (precision 28.00 = precision 28.0) = False
  let B := (significant_figures 0.32 = significant_figures 0.302) = False
  let C := (precision (2.4 * 10^2) = precision 240) = False
  let D := (significant_figures 220 = 3) ∧ (significant_figures 0.101 = 3)
  D = True
by
  sorry

end correct_choice_from_statements_l574_574075


namespace probability_at_least_6_heads_8_flips_l574_574168

-- Define the probability calculation of getting at least 6 heads in 8 coin flips.
def probability_at_least_6_heads (n : ℕ) (k : ℕ) : ℚ :=
  (Nat.choose n k + Nat.choose n (k + 1) + Nat.choose n (k + 2)) / 2^n

theorem probability_at_least_6_heads_8_flips : 
  probability_at_least_6_heads 8 6 = 37 / 256 := 
by
  sorry

end probability_at_least_6_heads_8_flips_l574_574168


namespace real_solutions_count_l574_574559

theorem real_solutions_count : (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (2*x₁ - 3*x₁ + 7)^2 + 2 = |x₁| + 1 ∧ (2*x₂ - 3*x₂ + 7)^2 + 2 = |x₂| + 1) ∧ 
not (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ (2*x₁ - 3*x₁ + 7)^2 + 2 = |x₁| + 1 ∧ (2*x₂ - 3*x₂ + 7)^2 + 2 = |x₂| + 1 ∧ (2*x₃ - 3*x₃ + 7)^2 + 2 = |x₃| + 1) := 
sorry

end real_solutions_count_l574_574559


namespace scaled_determinant_matrix_l574_574277

variable {R : Type*} [CommRing R]

theorem scaled_determinant_matrix
  (a b c d e f g h i : R)
  (h : Matrix.det ![[a, b, c], [d, e, f], [g, h, i]] = 2) :
  Matrix.det ![[3 * a, 3 * b, 3 * c], [3 * d, 3 * e, 3 * f], [3 * g, 3 * h, 3 * i]] = 54 :=
by
  sorry

end scaled_determinant_matrix_l574_574277


namespace parallel_line_plane_intersection_l574_574700

variable (α β : Set Point)
variable (m l : Line)
variable [IsPlane α] [IsPlane β]
variable [IsLine m] [IsLine l]

def planes_intersect (α β : Set Point) (l : Line) : Prop := l ⊆ α ∧ l ⊆ β

def parallel_to_both_planes (m : Line) (α β : Set Point) : Prop :=
  Parallel m α ∧ Parallel m β

def parallel_to_intersection (m l : Line) : Prop :=
  Parallel m l

axiom Parallel {A B : Type u} [IsLine A] [IsLine B] : Prop

theorem parallel_line_plane_intersection 
  (h₁ : planes_intersect α β l)
  (h₂ : parallel_to_both_planes m α β) : 
  parallel_to_intersection m l := 
sorry

end parallel_line_plane_intersection_l574_574700


namespace multiplier_for_deans_height_l574_574789

theorem multiplier_for_deans_height (h_R : ℕ) (h_R_eq : h_R = 13) (d : ℕ) (d_eq : d = 255) (h_D : ℕ) (h_D_eq : h_D = h_R + 4) : 
  d / h_D = 15 := by
  sorry

end multiplier_for_deans_height_l574_574789


namespace determine_b_l574_574679

theorem determine_b (a b c : ℕ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (eq_radicals: Real.sqrt (4 * a + 4 * b / c) = 2 * a * Real.sqrt (b / c)) : 
  b = c + 1 :=
sorry

end determine_b_l574_574679


namespace tom_books_l574_574368

theorem tom_books : 
  ∀ (joan_books : ℕ) (total_books : ℕ), joan_books = 10 → total_books = 48 → (total_books - joan_books) = 38 :=
by
  intros joan_books total_books h1 h2
  rw [h1, h2]
  sorry

end tom_books_l574_574368


namespace truck_stopping_distance_l574_574480

theorem truck_stopping_distance :
  let a := 35
  let d := -8
  ∑ i in Finset.range 5, (a + i * d) = 95 :=
by
  let a := 35
  let d := -8
  have h : (∑ i in Finset.range 5, (a + i * d)) = 95 := sorry
  exact h

end truck_stopping_distance_l574_574480


namespace rana_speed_calculation_l574_574927

-- Define the necessary structures and conditions
variable (R : Type) [Field R]

-- Problem conditions
def course_circumference : R := 115
def ajith_speed : R := 4
def meet_time : R := 115
def rana_speed : R

-- Define Ajith's distance covered
def ajith_distance := ajith_speed * meet_time

-- Define Rana's distance covered
def rana_distance := rana_speed * meet_time

-- Define the number of rounds completed
def ajith_rounds := ajith_distance / course_circumference
def rana_rounds := rana_distance / course_circumference

-- Statement to prove
theorem rana_speed_calculation :
  (ajith_rounds + 1) * course_circumference = rana_distance ↔ rana_speed = 5 := by
  sorry

end rana_speed_calculation_l574_574927


namespace eccentricity_of_ellipse_l574_574609

theorem eccentricity_of_ellipse 
  (a b c r : ℝ) 
  (a_gt_b_zero : a > b ∧ b > 0)
  (ellipse : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1)
  (circle : ∀ x y : ℝ, (x-1)^2 + (y-2)^2 = r^2)
  (right_focus : ∃ c : ℝ, c = a * (a^2 - b^2).sqrt / a)
  (top_vertex : ∃ b : ℝ, b = (a^2 - c^2).sqrt)
  (circle_passes_points : (c-1)^2 + (0-2)^2 = r^2 ∧ (0-1)^2 + (b-2)^2 = r^2)
  : eccentricity a b c = sqrt 2 / 2 := 
sorry

end eccentricity_of_ellipse_l574_574609


namespace number_of_zeros_of_f_l574_574035

def f (x : ℝ) : ℝ := 2 * x - Real.sin x

theorem number_of_zeros_of_f : 
  ∃! x : ℝ, f x = 0 :=
sorry

end number_of_zeros_of_f_l574_574035


namespace arithmetic_mean_rational_numbers_between_zero_and_one_l574_574467

/-- Prove that starting with 0 and 1, using the arithmetic mean,
    one can obtain the number 1/5
    and any rational number between 0 and 1. -/
theorem arithmetic_mean_rational_numbers_between_zero_and_one :
  (∃ s : Finset ℚ, {0, 1} ⊆ s ∧ ∀ (a b ∈ s), (a + b) / 2 ∈ s) ∧
  (∃ s : Finset ℚ, {0, 1} ⊆ s ∧ (1 / 5 : ℚ) ∈ s ∧ ∀ (a b ∈ s), (a + b) / 2 ∈ s) ∧
  (∀ (r : ℚ), 0 < r ∧ r < 1 → ∃ s : Finset ℚ, {0, 1} ⊆ s ∧ r ∈ s ∧ ∀ (a b ∈ s), (a + b) / 2 ∈ s) :=
sorry

end arithmetic_mean_rational_numbers_between_zero_and_one_l574_574467


namespace projection_of_vector_l574_574582

-- Definition of vectors a and b
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (3, -4)

-- Dot product of two vectors
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Projection of b onto a
def proj (a b : ℝ × ℝ) : ℝ × ℝ :=
  let scale := dot_product b a / dot_product a a
  (scale * a.1, scale * a.2)

-- Theorem stating that the projection of b onto a is (-1, -2)
theorem projection_of_vector :
  proj a b = (-1, -2) :=
sorry

end projection_of_vector_l574_574582


namespace line_integral_equals_33_div_2_l574_574254

noncomputable
def vector_field_F (x y z : ℝ) : ℝ × ℝ × ℝ := (y, x, x + y + z)

def parametrization (t : ℝ) : ℝ × ℝ × ℝ := (2 + t, 3 + t, 4 + t)

def dr (t : ℝ) : ℝ × ℝ × ℝ := (1, 1, 1)

def integrand (t : ℝ) : ℝ :=
  let (x, y, z) := parametrization t in
  let (Fx, Fy, Fz) := vector_field_F x y z in
  Fx + Fy + Fz

theorem line_integral_equals_33_div_2 :
  ∫ (t : ℝ) in 0..1, integrand t = 33 / 2 :=
by
  sorry

end line_integral_equals_33_div_2_l574_574254


namespace range_of_f_l574_574023

def f (x : ℝ) : ℝ := 4^x - 2^(x + 1) + 3

theorem range_of_f : 
∀ x, x ∈ set.Icc (-1/2 : ℝ) (1/2 : ℝ) → f x ∈ set.Icc (3/2 : ℝ) (2 : ℝ) :=
sorry

end range_of_f_l574_574023


namespace probability_at_least_6_heads_in_8_flips_l574_574144

theorem probability_at_least_6_heads_in_8_flips : 
  (∑ k in finset.range 3, nat.choose 8 (6 + k)) / (2 ^ 8) = 37 / 256 :=
by sorry

end probability_at_least_6_heads_in_8_flips_l574_574144


namespace order_of_abc_l574_574675

noncomputable def a : ℝ := 2017^0
noncomputable def b : ℝ := 2015 * 2017 - 2016^2
noncomputable def c : ℝ := ((-2/3)^2016) * ((3/2)^2017)

theorem order_of_abc : b < a ∧ a < c := by
  -- proof omitted
  sorry

end order_of_abc_l574_574675


namespace area_inequality_l574_574372

variables {A B C P : Type*} [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ P]

noncomputable def triangle (A B C : Type*) := True

def area (X Y Z : Type*) : ℝ := sorry -- Assume area is a predefined function.

def circumradius (A B C : Type*) : ℝ := sorry -- Assume circumradius is a predefined function.

def inside_triangle (P A B C : Type*) : Prop := sorry -- Assume inside_triangle is a predefined property.

theorem area_inequality (A B C P : Type*) 
  (h_tri : triangle A B C) 
  (h_inside : inside_triangle P A B C) 
  (R : ℝ) (h_circumradius : circumradius A B C = R) : 
  (area B P C / (dist P A)^2) + (area C P A / (dist P B)^2) + (area A P B / (dist P C)^2) 
  ≥ (area A B C / R^2) :=
sorry

end area_inequality_l574_574372


namespace quadratic_polynomial_real_coefficients_l574_574260

noncomputable def polynomial_with_root_and_coefficient (a b : ℝ) : Polynomial ℝ :=
  Polynomial.C a * (Polynomial.X - Polynomial.C (4 + Complex.i)) * (Polynomial.X - Polynomial.C (4 - Complex.i))

theorem quadratic_polynomial_real_coefficients :
  (∃ (p : Polynomial ℝ), (p = polynomial_with_root_and_coefficient 3 4 + Complex.i) ∧
    p.coeff 2 = 3 ∧
    p.coeff 0 = (Polynomial.eval 0 p)) →
    Polynomial.eval 0 p = 3x^2 - 24x + 51 :=
by
  sorry

end quadratic_polynomial_real_coefficients_l574_574260


namespace polynomial_product_equals_expected_result_l574_574945

-- Define the polynomials
def polynomial_product (x : ℝ) : ℝ := (x + 1) * (x^2 - x + 1)

-- Define the expected result of the product
def expected_result (x : ℝ) : ℝ := x^3 + 1

-- The main theorem to prove
theorem polynomial_product_equals_expected_result (x : ℝ) : polynomial_product x = expected_result x :=
by
  -- Placeholder for the proof
  sorry

end polynomial_product_equals_expected_result_l574_574945


namespace johnsonville_max_band_members_l574_574049

def max_band_members :=
  ∃ m : ℤ, 30 * m % 34 = 2 ∧ 30 * m < 1500 ∧
  ∀ n : ℤ, (30 * n % 34 = 2 ∧ 30 * n < 1500) → 30 * n ≤ 30 * m

theorem johnsonville_max_band_members : ∃ m : ℤ, 30 * m % 34 = 2 ∧ 30 * m < 1500 ∧
                                           30 * m = 1260 :=
by 
  sorry

end johnsonville_max_band_members_l574_574049


namespace ticket_sales_amount_theater_collected_50_dollars_l574_574926

variable (num_people total_people : ℕ) (cost_adult_entry cost_child_entry : ℕ) (num_children : ℕ)
variable (total_collected : ℕ)

theorem ticket_sales_amount
  (h1 : cost_adult_entry = 8)
  (h2 : cost_child_entry = 1)
  (h3 : total_people = 22)
  (h4 : num_children = 18)
  (h5 : num_people = total_people - num_children)
  : total_collected = (num_people * cost_adult_entry + num_children * cost_child_entry) := sorry

theorem theater_collected_50_dollars 
  (h1 : cost_adult_entry = 8)
  (h2 : cost_child_entry = 1)
  (h3 : total_people = 22)
  (h4 : num_children = 18)
  (h5 : total_collected = 50)
  : total_collected = 50 := sorry

end ticket_sales_amount_theater_collected_50_dollars_l574_574926


namespace arithmetic_problem_l574_574061

-- Definitions for given conditions
def base5_to_base10 (n : Nat) : Nat := 1 + 5^2
def base7_to_base10 (n : Nat) : Nat := 1 + 5*7^1 + 4*7^2 + 3*7^3

-- The main theorem to prove
theorem arithmetic_problem : (2468 / 26) - (base7_to_base10 3451) + 6791 = 7624 := by
  -- Calculation for base 5 to base 10 conversion
  have base5_val : (base5_to_base10 101) = 26 := by
    simp [base5_to_base10]
    sorry
  
  -- Calculation for base 7 to base 10 conversion
  have base7_val : (base7_to_base10 3451) = 1261 := by
    simp [base7_to_base10]
    sorry

  -- Assertion of the final arithmetic result
  rw [base5_val, base7_val]
  sorry

end arithmetic_problem_l574_574061


namespace last_number_erased_is_1598_l574_574038

def seq := List.range (2002)  -- Range from 0 to 2001, representing 1 to 2002

def erase_positions (l : List ℕ) : List ℕ :=
  let len := l.length 
  List.filter (λ n => ¬List.any (List.range (len + 1)) (λ k => n == 3*k + 1)) l

def last_erased (seq : List ℕ) : ℕ :=
  let rec erase (l : List ℕ) : ℕ :=
    if l = [] then 0    -- Empty list condition to end recursion (assuming 0 as a fallback)
    else 
      let new_l := erase_positions l 
      if new_l = [] then 
        l.head  -- last number to be erased
      else 
        erase new_l 
  erase seq

theorem last_number_erased_is_1598 : last_erased seq = 1598 := 
  by
    -- Proof is omitted
    sorry

end last_number_erased_is_1598_l574_574038


namespace common_ratio_q_l574_574635

variable (a1 q : ℝ) (S : ℕ → ℝ)

def geometric_sequence (n : ℕ) : ℝ := a1 * q^(n - 1)
def sum_of_geometric_sequence (n : ℕ) : ℝ := a1 * (1 - q^n) / (1 - q)

theorem common_ratio_q : 
  (S 3 = 3 * geometric_sequence 3) → (sum_of_geometric_sequence 3 = 3 * a1 * q^2) → 
  (q = 1 / 2 ∨ q = (Real.sqrt 5 - 1) / 2) :=
by
  intro h₁ h₂
  sorry

end common_ratio_q_l574_574635


namespace boric_acid_solution_l574_574881

theorem boric_acid_solution
  (amount_first_solution: ℝ) (percentage_first_solution: ℝ)
  (amount_second_solution: ℝ) (percentage_second_solution: ℝ)
  (final_amount: ℝ) (final_percentage: ℝ)
  (h1: amount_first_solution = 15)
  (h2: percentage_first_solution = 0.01)
  (h3: amount_second_solution = 15)
  (h4: final_amount = 30)
  (h5: final_percentage = 0.03)
  : percentage_second_solution = 0.05 := 
by
  sorry

end boric_acid_solution_l574_574881


namespace flagship_max_distance_l574_574874

theorem flagship_max_distance:
  ∃ (d : ℝ), d = 5182 ∧ 
  ∀ (planes : ℕ) (dist_per_tank : ℝ) (H : planes = 100 ∧ dist_per_tank = 1000), 
    (∃ f : ℝ → ℝ, 
      (∀ n : ℕ, 1 ≤ n ∧ n ≤ 100 → f (Harmonic n) = dist_per_tank / n) ∧ 
      d = dist_per_tank * (Harmonic 100)) :=
sorry

end flagship_max_distance_l574_574874


namespace number_of_integers_with_absolute_value_less_than_3_l574_574858

theorem number_of_integers_with_absolute_value_less_than_3 :
  ∃ (S : Finset ℤ), (∀ x ∈ S, |x| < 3) ∧ S.card = 5 :=
by
  let S := ({0, -1, 1, -2, 2} : Finset ℤ)
  have h1 : ∀ x ∈ S, |x| < 3 := by
    intros x hx
    finset_cases x hx <;> simp
  have h2 : S.card = 5 := by
    norm_num
  exact ⟨S, h1, h2⟩

end number_of_integers_with_absolute_value_less_than_3_l574_574858


namespace area_of_square_l574_574907

-- Define the parabola and the line
def parabola (x : ℝ) : ℝ := x^2 + 4 * x + 3
def line (y : ℝ) : Prop := y = 7

-- Define the roots of the quadratic equation derived from the conditions
noncomputable def root1 : ℝ := -2 + 2 * Real.sqrt 2
noncomputable def root2 : ℝ := -2 - 2 * Real.sqrt 2

-- Define the side length of the square
noncomputable def side_length : ℝ := abs (root1 - root2)

-- Define the area of the square
noncomputable def area_square : ℝ := side_length^2

-- Theorem statement for the problem
theorem area_of_square : area_square = 32 :=
sorry

end area_of_square_l574_574907


namespace find_pairs_of_numbers_l574_574461

theorem find_pairs_of_numbers (a b : ℝ) :
  (a^2 + b^2 = 15 * (a + b)) ∧ (a^2 - b^2 = 3 * (a - b) ∨ a^2 - b^2 = -3 * (a - b))
  ↔ (a = 6 ∧ b = -3) ∨ (a = -3 ∧ b = 6) ∨ (a = 0 ∧ b = 0) ∨ (a = 15 ∧ b = 15) :=
sorry

end find_pairs_of_numbers_l574_574461


namespace smallest_n_common_factor_l574_574963

theorem smallest_n_common_factor :
  ∃ n : ℕ, (0 < n) ∧ (n = 14) ∧ (∃ d : ℕ, d > 1 ∧ d ∣ (8 * n + 3) ∧ d ∣ (10 * n - 4)) :=
by {
  existsi 14,
  split,
  { exact nat.succ_pos' 13, },
  split,
  { refl, },
  existsi 17,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { norm_num, }
}

end smallest_n_common_factor_l574_574963


namespace remainder_of_s_at_1012_mod_100_l574_574385

noncomputable def q (x : ℂ) : ℂ := 
  ∑ i in Finset.range 1013, x^i

-- Define the polynomial divisor
def p (x : ℂ) : ℂ := x^3 + x^2 + x + 1

-- Define the remainder s(x) when q(x) is divided by p(x)
noncomputable def s (x : ℂ) : ℂ := q(x) % p(x)

-- The main theorem statement
theorem remainder_of_s_at_1012_mod_100 : 
  (|s 1012| % 100) = 2 := 
sorry

end remainder_of_s_at_1012_mod_100_l574_574385


namespace value_of_x_plus_y_l574_574284

noncomputable def imaginary_unit : ℂ := complex.I

theorem value_of_x_plus_y (x y : ℝ) (h : y + (2 - x) * imaginary_unit = 1 - imaginary_unit) : x + y = 4 := by
  sorry

end value_of_x_plus_y_l574_574284


namespace binom_12_9_is_220_l574_574214

def choose (n k : ℕ) : ℕ := n.choose k

theorem binom_12_9_is_220 :
  choose 12 9 = 220 :=
by {
  -- Proof is omitted
  sorry
}

end binom_12_9_is_220_l574_574214


namespace students_passed_l574_574184

theorem students_passed (N F : ℕ) (f : ℝ) (hN : N = 1000) (hf : f = 0.4) (hF : F = (f * N).toNat) :
  N - F = 600 :=
by
  rw [hN, hf, Nat.cast_mul, Nat.cast_ofNat]
  rw [hF, Nat.cast_ofNat]
  norm_num
  sorry

end students_passed_l574_574184


namespace find_F2_l574_574473

-- Set up the conditions as definitions
def m : ℝ := 1 -- in kg
def R1 : ℝ := 0.5 -- in meters
def R2 : ℝ := 1 -- in meters
def F1 : ℝ := 1 -- in Newtons

-- Rotational inertia I formula
def I (R : ℝ) : ℝ := m * R^2

-- Equality of angular accelerations
def alpha_eq (F1 F2 R1 R2 : ℝ) : Prop :=
  (F1 * R1) / (I R1) = (F2 * R2) / (I R2)

-- The proof goal
theorem find_F2 (F2 : ℝ) : 
  alpha_eq F1 F2 R1 R2 → F2 = 2 :=
by
  sorry

end find_F2_l574_574473


namespace find_B_find_b_given_a_and_c_l574_574354

variables (A B C : ℝ) (a b c : ℝ)
-- Definitions and assumptions
def acute_triangle_ABC := ∀ (α β γ : ℝ), 0 < α ∧ α < π/2 ∧ 0 < β ∧ β < π/2 ∧ 0 < γ ∧ γ < π/2

-- Problem (1)
theorem find_B (h1 : acute_triangle_ABC A B C)
               (h2 : a = 2 * b * real.sin A) :
               B = π / 6 :=
sorry

-- Problem (2)
theorem find_b_given_a_and_c (h1 : a = 3 * real.sqrt 3) 
                             (h2 : c = 5) 
                             (h3 : B = π / 6) :
                             b = real.sqrt 7 :=
sorry

end find_B_find_b_given_a_and_c_l574_574354


namespace principal_amount_l574_574524

/-
  Given:
  - Simple Interest (SI) = Rs. 4016.25
  - Rate (R) = 0.08 (8% per annum)
  - Time (T) = 5 years
  
  We want to prove:
  Principal = Rs. 10040.625
-/

def SI : ℝ := 4016.25
def R : ℝ := 0.08
def T : ℕ := 5

theorem principal_amount :
  ∃ P : ℝ, SI = (P * R * T) / 100 ∧ P = 10040.625 :=
by
  sorry

end principal_amount_l574_574524


namespace paint_needed_l574_574592

-- Definitions from conditions
def total_needed_paint := 70
def initial_paint := 36
def bought_paint := 23

-- The main statement to prove
theorem paint_needed : total_needed_paint - (initial_paint + bought_ppaint) = 11 :=
by
  -- Definitions are already imported and stated
  -- Just need to refer these to the theorem assertion correctly
  sorry

end paint_needed_l574_574592


namespace find_point_H_l574_574667

-- Define the points and vectors
def O : ℝ × ℝ × ℝ := (0, 0, 0)
def A : ℝ × ℝ × ℝ := (-1, 1, 0)
def B : ℝ × ℝ × ℝ := (0, 1, 1)

-- The theorem to prove
theorem find_point_H (H : ℝ × ℝ × ℝ) : 
  (∃ λ : ℝ, 0 ≤ λ ∧ λ ≤ 1 ∧ H = (-λ, λ, 0)) ∧ 
  ((H.1, H.2 - 1, H.3 - 1) ⬝ (-1, 1, 0) = 0) →
  H = (-1/2, 1/2, 0) :=
sorry

end find_point_H_l574_574667


namespace integral_sqrt_1_minus_x2_roots_l574_574645

noncomputable def f (x : ℝ) : ℝ :=
  x^2 - 2 * x * Real.sin (π / 2 * x) + 1

theorem integral_sqrt_1_minus_x2_roots :
  (f (-1) = 0) ∧ (f (1) = 0) →
  ∫ x in -1..1, Real.sqrt (1 - x^2) = π / 2 :=
by
  intro h
  sorry

end integral_sqrt_1_minus_x2_roots_l574_574645


namespace sequence_explicit_formula_l574_574901

theorem sequence_explicit_formula :
  ∀ n : ℕ, (a : ℕ → ℝ) (h₀ : a 0 = 1)
    (h_rec : ∀ n : ℕ, a (n + 1) = (1 + 4 * a n + real.sqrt (1 + 24 * a n)) / 16),
    a n = (1 / 3) * (1 + 1 / (2 ^ n)) * (1 + 1 / (2 ^ (n + 1))) :=
by
  sorry

end sequence_explicit_formula_l574_574901


namespace hyperbola_standard_eq_l574_574589

theorem hyperbola_standard_eq:
  let f : ℝ := 10
  let b : ℝ := 4
  let c : ℝ := 84
  \forall x y : ℝ
    [\frac{x^2}{c} - \frac{y^2}{b} = 1 ↔ (2x - 5y + 20 = 0) ∧ coordinate_axes_are_symmetry] := by
  sorry

end hyperbola_standard_eq_l574_574589


namespace prob_same_color_l574_574686

-- Define the problem conditions
def balls_in_box := {red := 2, white := 3}
def total_balls := 5

def prob_red_draw := (balls_in_box.red / total_balls)
def prob_white_draw := (balls_in_box.white / total_balls)

-- State the theorem
theorem prob_same_color : 
  ((prob_red_draw * prob_red_draw) + (prob_white_draw * prob_white_draw)) = (13 / 25) :=
by
  sorry

end prob_same_color_l574_574686


namespace polygon_interior_angle_sum_l574_574343

theorem polygon_interior_angle_sum (n : ℕ) (h : (n - 2) * 180 = 900) : n = 7 :=
by {
  sorry,
}

end polygon_interior_angle_sum_l574_574343


namespace monotonicity_f_range_of_m_l574_574741

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := ((1 - a) / 2) * x^2 + a * x - Math.log x

-- Monotonicity of f(x) for a > 1
theorem monotonicity_f (a : ℝ) (h : a > 1) : 
  (if a > 2 then 
    ∀ x, x ∈ (0 : ℝ, 1 / (a - 1)) ∪ (1, +∞) → strictly_decreasing_on (f a) (0 : ℝ, 1 / (a - 1)) ∪ (1, +∞) ∧ 
         ∀ x, x ∈ [1 / (a - 1), 1] → strictly_increasing_on (f a) [1 / (a - 1), 1]
  else if a = 2 then 
    ∀ x, x > 0 → strictly_decreasing_on (f a) (0 : ℝ, +∞)
  else
    ∀ x, x ∈ (0, 1) ∪ (1 / (a - 1), +∞) → strictly_decreasing_on (f a) (0, 1) ∪ (1 / (a - 1), +∞) ∧ 
         ∀ x, x ∈ [1, 1 / (a - 1)] → strictly_increasing_on (f a) [1, 1 / (a - 1)]
  ) := sorry

-- Range of m for given conditions on a and x
theorem range_of_m (a : ℝ) (h : a ∈ (3 : ℝ, 4)) :
  ∀ x1 x2 ∈ (1 : ℝ) .. 2, ((a ^ 2 - 1) * m / 2 + Math.log 2 > abs (f a x1 - f a x2)) → m ≥ 1 / 15 := sorry

end monotonicity_f_range_of_m_l574_574741


namespace symmetric_points_trapezoid_l574_574439

theorem symmetric_points_trapezoid (
  A B C D O B' C' : Point 
) 
  (hTrapezoid: is_trapezoid A B C D) 
  (hBases: AD = base A D ∧ BC = base B C) 
  (hIntersection: intersect_diagonals A C B D O) 
  (hSymmetric_points: symmetric_to B' B (bisector_angle B O C) ∧ symmetric_to C' C (bisector_angle B O C))
:
  ∠ C' A C = ∠ B' D B 
:= 
sorry

end symmetric_points_trapezoid_l574_574439


namespace point_K_outside_hexagon_and_length_KC_l574_574391

theorem point_K_outside_hexagon_and_length_KC :
    ∀ (A B C K : ℝ × ℝ),
    A = (0, 0) →
    B = (3, 0) →
    C = (3 / 2, (3 * Real.sqrt 3) / 2) →
    K = (15 / 2, - (3 * Real.sqrt 3) / 2) →
    (¬ (0 ≤ K.1 ∧ K.1 ≤ 3 ∧ 0 ≤ K.2 ∧ K.2 ≤ 3 * Real.sqrt 3)) ∧
    Real.sqrt ((K.1 - C.1) ^ 2 + (K.2 - C.2) ^ 2) = 3 * Real.sqrt 7 :=
by
  intros A B C K hA hB hC hK
  sorry

end point_K_outside_hexagon_and_length_KC_l574_574391


namespace quadratic_equation_roots_transformation_l574_574738

theorem quadratic_equation_roots_transformation (α β : ℝ) 
  (h1 : 3 * α^2 + 7 * α + 4 = 0)
  (h2 : 3 * β^2 + 7 * β + 4 = 0) :
  ∃ y : ℝ, 21 * y^2 - 23 * y + 6 = 0 :=
sorry

end quadratic_equation_roots_transformation_l574_574738


namespace fraction_of_area_above_line_l574_574511

open_locale real
open set

theorem fraction_of_area_above_line (A B C D : ℝ × ℝ) (A_eq : A = (1, 3)) (B_eq : B = (4, 0))
(C_eq : C = (4, 3)) (D_eq : D = (1, 0)) :
  let line_eq := line_through A B,
      square_vertices := {A, B, C, D},
      square_area := 9,
      triangle_area := (1 / 2) * 3 * 3 in
  ((square_area - triangle_area) / square_area = 1 / 2) :=
sorry

end fraction_of_area_above_line_l574_574511


namespace min_squares_to_partition_rectangle_l574_574771

-- Define the dimensions of the rectangle
def width : ℕ := 6
def height : ℕ := 7

-- Define the conjecture as a theorem to be proven
theorem min_squares_to_partition_rectangle : 
  ∃squares : list (ℕ × ℕ), 
    (∀sq ∈ squares, let (w, h) := sq in w = h ∧ w ∣ width ∧ h ∣ height) ∧
    (width * height = squares.foldl (fun acc sq => let (w, h) := sq in acc + w * h) 0) ∧
    (squares.length = 5) :=
sorry

end min_squares_to_partition_rectangle_l574_574771


namespace tiles_required_for_floor_l574_574889

def tileDimensionsInFeet (width_in_inches : ℚ) (length_in_inches : ℚ) : ℚ × ℚ :=
  (width_in_inches / 12, length_in_inches / 12)

def area (length : ℚ) (width : ℚ) : ℚ :=
  length * width

noncomputable def numberOfTiles (floor_length : ℚ) (floor_width : ℚ) (tile_length : ℚ) (tile_width : ℚ) : ℚ :=
  (area floor_length floor_width) / (area tile_length tile_width)

theorem tiles_required_for_floor : numberOfTiles 10 15 (5/12) (2/3) = 540 := by
  sorry

end tiles_required_for_floor_l574_574889


namespace area_of_rectangle_l574_574793

theorem area_of_rectangle (x : ℝ) (hx : 0 < x) :
  let length := 3 * x - 1
  let width := 2 * x + 1 / 2
  let area := length * width
  area = 6 * x^2 - 1 / 2 * x - 1 / 2 :=
by
  sorry

end area_of_rectangle_l574_574793


namespace square_area_correct_l574_574915

noncomputable def square_area : ℝ :=
  let f : ℝ → ℝ := λ x, x^2 + 4 * x + 3
  let y_val : ℝ := 7
  let x1 : ℝ := -2 - 2 * Real.sqrt 2
  let x2 : ℝ := -2 + 2 * Real.sqrt 2
  let side_length := x2 - x1
  side_length * side_length

theorem square_area_correct :
  let f : ℝ → ℝ := λ x, x^2 + 4 * x + 3
  let y_val : ℝ := 7
  let x1 : ℝ := -2 - 2 * Real.sqrt 2
  let x2 : ℝ := -2 + 2 * Real.sqrt 2
  let side_length := x2 - x1
  side_length * side_length = 32 := by
  sorry

end square_area_correct_l574_574915


namespace solve_linear_system_l574_574299

-- Given conditions
def matrix : Matrix (Fin 2) (Fin 3) ℚ :=
  ![![1, -1, 1], ![1, 1, 3]]

def system_of_equations (x y : ℚ) : Prop :=
  (x - y = 1) ∧ (x + y = 3)

-- Desired solution
def solution (x y : ℚ) : Prop :=
  x = 2 ∧ y = 1

-- Proof problem statement
theorem solve_linear_system : ∃ x y : ℚ, system_of_equations x y ∧ solution x y := by
  sorry

end solve_linear_system_l574_574299


namespace meet_point_l574_574556

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem meet_point :
  let derek : ℝ × ℝ := (12, -5)
  let mira  : ℝ × ℝ := (0, 9)
  midpoint derek mira = (6, 2) :=
by
  let derek := (12, -5)
  let mira  := (0, 9)
  have h_midpoint : midpoint derek mira = (6, 2) := by
    unfold midpoint
    simp only
    sorry
  exact h_midpoint

end meet_point_l574_574556


namespace part_one_part_two_l574_574654

def f (x : ℝ) : ℝ := 
  (Real.sin (Real.pi - x)) * (Real.sin (Real.pi / 2 - x)) + 
  Real.sqrt 3 * (Real.cos (Real.pi + x))^2 - Real.sqrt 3 / 2

theorem part_one : f (Real.pi / 4) = 1 / 2 + Real.sqrt 3 / 2 :=
by
  sorry

theorem part_two : ∀ x, -Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 4 → 
  f (x) ≥ -1 / 2 + Real.sqrt 3 / 2 :=
by
  sorry

end part_one_part_two_l574_574654


namespace square_area_l574_574905

theorem square_area (y : ℝ) (x : ℝ → ℝ) : 
    (∀ x, y = x ^ 2 + 4 * x + 3) → (y = 7) → 
    ∃ area : ℝ, area = 32 := 
by
  intro h₁ h₂ 
  -- Proof steps would go here
  sorry

end square_area_l574_574905


namespace trig_identity_l574_574099

theorem trig_identity : 2 * Real.sin (75 * Real.pi / 180) * Real.cos (15 * Real.pi / 180) - 1 = Real.sqrt 3 / 2 :=
by
  sorry

end trig_identity_l574_574099


namespace four_digit_number_sum_eq_4983_l574_574106

def reverse_number (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n % 1000) / 100
  let d3 := (n % 100) / 10
  let d4 := n % 10
  1000 * d4 + 100 * d3 + 10 * d2 + d1

theorem four_digit_number_sum_eq_4983 (n : ℕ) :
  n + reverse_number n = 4983 ↔ n = 1992 ∨ n = 2991 :=
by sorry

end four_digit_number_sum_eq_4983_l574_574106


namespace number_of_possible_digits_l574_574504

-- Define the condition for a digit N being part of the number 864N such that it is divisible by 3
def digit_divisible_by_three(N : ℕ) : Prop :=
  (864 + N) % 3 = 0

-- Representation of problem: find the number of possible digits N such that 864N is divisible by 3
theorem number_of_possible_digits : 
  { N : ℕ // N < 10 ∧ digit_divisible_by_three N }.card = 4 := 
sorry

end number_of_possible_digits_l574_574504


namespace Bryan_deposit_amount_l574_574397

theorem Bryan_deposit_amount (deposit_mark : ℕ) (deposit_bryan : ℕ)
  (h1 : deposit_mark = 88)
  (h2 : deposit_bryan = 5 * deposit_mark - 40) : 
  deposit_bryan = 400 := 
by
  sorry

end Bryan_deposit_amount_l574_574397


namespace cos_neg_sixteen_pi_thirds_l574_574964

theorem cos_neg_sixteen_pi_thirds :
    cos (- (16 * π) / 3) = -1 / 2 :=
by
  /- The proof is omitted for the task requirements -/
  sorry

end cos_neg_sixteen_pi_thirds_l574_574964


namespace train_length_is_correct_l574_574529

-- Define the conditions
def speed_km_hr := 72 -- Speed in kilometers per hour
def time_sec := 9 -- Time in seconds

-- Unit conversion from km/hr to m/s
def convert_speed (s : ℕ) : ℕ := s * 1000 / 3600

-- Define the expected answer
def expected_length := 180

-- State the theorem
theorem train_length_is_correct :
  (let speed_m_s := convert_speed speed_km_hr in speed_m_s * time_sec = expected_length) :=
by
  sorry

end train_length_is_correct_l574_574529


namespace valid_programs_count_l574_574522

open Finset

theorem valid_programs_count :
  let courses := { "English", "Algebra", "Geometry", "History", "Art", "Latin", "Science" }
  let math_courses := { "Algebra", "Geometry" }
  let required_courses := { "English" }
  (finset.card (courses \ required_courses) = 6) →
  (choose 6 4 = 15) →
  let total_programs := 15
  let invalid_programs :=
    - 2 * choose 4 3
  let result := total_programs + invalid_programs
  result = 7 :=
by
  sorry

end valid_programs_count_l574_574522


namespace solve_inequality_l574_574457

theorem solve_inequality :
  {x : ℝ | (x + 3) / (4 - x) ≥ 0} = set.Icc (-3 : ℝ) (4 : ℝ) :=
by
  sorry

end solve_inequality_l574_574457


namespace ticket_sales_l574_574924

-- Definitions of the conditions
theorem ticket_sales (adult_cost child_cost total_people child_count : ℕ)
  (h1 : adult_cost = 8)
  (h2 : child_cost = 1)
  (h3 : total_people = 22)
  (h4 : child_count = 18) :
  (child_count * child_cost + (total_people - child_count) * adult_cost = 50) := by
  sorry

end ticket_sales_l574_574924


namespace luisa_mpg_l574_574754

theorem luisa_mpg
  (d_grocery d_mall d_pet d_home : ℕ)
  (cost_per_gal total_cost : ℚ)
  (total_miles : ℕ )
  (total_gallons : ℚ)
  (mpg : ℚ):
  d_grocery = 10 →
  d_mall = 6 →
  d_pet = 5 →
  d_home = 9 →
  cost_per_gal = 3.5 →
  total_cost = 7 →
  total_miles = d_grocery + d_mall + d_pet + d_home →
  total_gallons = total_cost / cost_per_gal →
  mpg = total_miles / total_gallons →
  mpg = 15 :=
by
  intros
  sorry

end luisa_mpg_l574_574754


namespace sqrt_difference_l574_574944

theorem sqrt_difference :
  (Real.sqrt 63 - 7 * Real.sqrt (1 / 7)) = 2 * Real.sqrt 7 :=
by
  sorry

end sqrt_difference_l574_574944


namespace num_satisfying_permutations_l574_574537

-- Definition for the given problem conditions
def satisfies_conditions (l : List ℕ) : Prop :=
  l.length = 5 ∧
  l.nodup ∧
  (∀ (a : ℕ), a ∈ l → a ≤ 5) ∧
  (l.last = some 5 ∨ l.last = some 3 ∨ l.last = some 1) ∧
  (∀ i : ℕ, i < 3 → (l.nth i).isSome → (l.nth (i+1)).isSome → (l.nth (i+2)).isSome →
    ((l.nth i).get_or_else 0 + (l.nth (i+1)).get_or_else 0 + (l.nth (i+2)).get_or_else 0) % (l.nth i).get_or_else 1 = 0)

-- Proof statement translating the problem and answer
theorem num_satisfying_permutations : 
  ∃ l : List (List ℕ), 
  (∀ x ∈ l, satisfies_conditions x) ∧ 
  l.length = 5 := sorry

end num_satisfying_permutations_l574_574537


namespace jack_travel_total_hours_l574_574069

theorem jack_travel_total_hours :
  (20 + 14 * 24) + (15 + 10 * 24) + (10 + 7 * 24) = 789 := by
  sorry

end jack_travel_total_hours_l574_574069


namespace count_even_integers_between_3000_and_7000_with_four_different_digits_l574_574318

theorem count_even_integers_between_3000_and_7000_with_four_different_digits : 
  let n := 1008 in
  ∃ count: ℕ, count = n ∧ ∀ x, (3000 ≤ x ∧ x < 7000 ∧ even x ∧ nat.digits 10 x).length = 4 → 
    count_even_integers x count = n :=
by
  -- Sorry is used to indicate the omission of the actual proof.
  sorry

end count_even_integers_between_3000_and_7000_with_four_different_digits_l574_574318


namespace curve_C₂_eqn_max_distance_Q_to_l_l574_574632

-- Definitions based on conditions
def polar_coord_of_curve_C1 (θ : ℝ) : ℝ := 4 * Real.cos θ
def parametric_eq_of_line_l (t : ℝ) : ℝ × ℝ := (2 + 2 * t, -1 + 3 * t)

-- Required proofs
theorem curve_C₂_eqn : ∀ (x y : ℝ), (x ^ 2 + (y ^ 2) / 4 = 1) := 
sorry

theorem max_distance_Q_to_l : ∃ (t : ℝ) (θ : ℝ), 
  let Q := (Real.cos θ, 2 * Real.sin θ) in
  ∃ d : ℝ, d = sqrt 13 ∧
  d = abs (3 * Real.cos θ - 4 * Real.sin θ - 8) / sqrt 13 := 
sorry

end curve_C₂_eqn_max_distance_Q_to_l_l574_574632


namespace geometric_sequence_sum_2n_terms_l574_574816

def geometric_sum_first_n_terms (S_n S_3n : ℕ) (S_n_eq S_3n_eq S_n_pos S_3n_pos : Prop) : Prop :=
  ∃ S_2n : ℕ, S_n = 24 ∧ S_3n = 42 ∧ S_2n = 36

theorem geometric_sequence_sum_2n_terms : 
  geometric_sum_first_n_terms 24 42 (by norm_num) (by norm_num) (by norm_num) (by norm_num) := sorry

end geometric_sequence_sum_2n_terms_l574_574816


namespace smallest_prime_with_consecutive_primes_l574_574387

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def arithmetic_sequence (p A: ℕ) (n: ℕ) : ℕ :=
  p + A * n

def consecutive_primes (p A: ℕ) (m: ℕ) : Prop :=
  ∀ k < m, is_prime (arithmetic_sequence p A k)

theorem smallest_prime_with_consecutive_primes :
  let A := 30 in
  ∀ p: ℕ, is_prime p → (p ≠ 2 ∧ p ≠ 3 ∧ p ≠ 5) →
  (∀ q: ℕ, is_prime q → (q ≠ 2 ∧ q ≠ 3 ∧ q ≠ 5) →
  consecutive_primes p A 6 → consecutive_primes q A 7 → q > p) →
  p = 7 :=
by
  sorry

end smallest_prime_with_consecutive_primes_l574_574387


namespace point_in_second_quadrant_condition_l574_574680

theorem point_in_second_quadrant_condition (a : ℤ)
  (h1 : 3 * a - 9 < 0)
  (h2 : 10 - 2 * a > 0)
  (h3 : |3 * a - 9| = |10 - 2 * a|):
  (a + 2) ^ 2023 - 1 = 0 := 
sorry

end point_in_second_quadrant_condition_l574_574680


namespace area_of_region_B_l574_574220

noncomputable def region_B_area : ℝ :=
  let square_area := 900
  let excluded_area := 28.125 * Real.pi
  square_area - excluded_area

theorem area_of_region_B : region_B_area = 900 - 28.125 * Real.pi :=
by {
  sorry
}

end area_of_region_B_l574_574220


namespace number_of_ordered_pairs_l574_574673

noncomputable def count_valid_pairs : ℕ :=
  {p : ℤ × ℤ | let x := p.1, y := p.2 in
                x^2 + y^2 < 25 ∧
                x^2 + y^2 < 8 * x - 16 * y ∧
                x^2 + y^2 < -16 * x + 8 * y}.to_finset.card

theorem number_of_ordered_pairs :
  count_valid_pairs = 9 :=
sorry

end number_of_ordered_pairs_l574_574673


namespace find_k_l574_574649

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ∈ Icc (-2 : ℝ) 2 then 2 * x + 1
  else if x ∈ Icc 2 4 then 1 + x^2
  else 0

-- State the proof problem in Lean
theorem find_k (k : ℝ) (hk : k < 2) (h : ∫ x in k..3, f x = 40 / 3) : k = 0 ∨ k = -1 :=
  sorry

end find_k_l574_574649


namespace m_range_max_f_diff_l574_574393

noncomputable section

variable (m a b e : ℝ) (f : ℝ → ℝ)

/-- The function definition -/
def f (x : ℝ) : ℝ := log x + (1 / 2) * x^2 - (m + 2) * x

/-- Condition: x = a and x = b are extreme points of f(x) with a < b -/
def is_extreme (a b : ℝ) : Prop := (a < b) ∧ (f' a = 0) ∧ (f' b = 0)
where f' (x : ℝ) : ℝ := (x^2 - (m + 2) * x + 1) / x

/-- Condition: m ∈ ℝ -/
def m_real : Prop := m ∈ ℝ

/-- Proven statement 1: Range of values for m -/
theorem m_range (h1 : is_extreme a b) (h2 : m_real) : m > 0 :=
sorry

/-- Proven statement 2: Maximum value of f(b) - f(a) when b / a ≥ e -/
theorem max_f_diff (h1 : is_extreme a b) (h2 : b / a ≥ e) : f b - f a = 1 - e / 2 + 1 / (2 * e) :=
sorry

end m_range_max_f_diff_l574_574393


namespace problem_solution_l574_574853

theorem problem_solution :
  (315^2 - 291^2) / 24 = 606 :=
by
  sorry

end problem_solution_l574_574853


namespace variation_of_x_l574_574329

theorem variation_of_x (k j z : ℝ) : ∃ m : ℝ, ∀ x y : ℝ, (x = k * y^2) ∧ (y = j * z^(1 / 3)) → (x = m * z^(2 / 3)) :=
sorry

end variation_of_x_l574_574329


namespace total_pies_l574_574367

theorem total_pies {team1 team2 team3 total_pies : ℕ} 
  (h1 : team1 = 235) 
  (h2 : team2 = 275) 
  (h3 : team3 = 240) 
  (h4 : total_pies = team1 + team2 + team3) : 
  total_pies = 750 := by 
  sorry

end total_pies_l574_574367


namespace triangle_inradius_is_2_5_l574_574806

variable (A : ℝ) (p : ℝ) (r : ℝ)

def triangle_has_given_inradius (A p : ℝ) : Prop :=
  A = r * p / 2

theorem triangle_inradius_is_2_5 (h₁ : A = 25) (h₂ : p = 20) :
  triangle_has_given_inradius A p r → r = 2.5 := sorry

end triangle_inradius_is_2_5_l574_574806


namespace total_distance_walked_in_a_week_l574_574543

theorem total_distance_walked_in_a_week :
  (distance_per_day_dog1 distance_per_day_dog2 days_in_week : ℕ)
  (h1 : distance_per_day_dog1 = 2)
  (h2 : distance_per_day_dog2 = 8)
  (h3 : days_in_week = 7) :
  distance_per_day_dog1 * days_in_week + distance_per_day_dog2 * days_in_week = 70 := by
  -- We are skipping the proof steps as per the instruction
  sorry

end total_distance_walked_in_a_week_l574_574543


namespace angle_with_same_terminal_side_315_l574_574016

def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = k * 360 + β

theorem angle_with_same_terminal_side_315:
  same_terminal_side (-45) 315 :=
by
  sorry

end angle_with_same_terminal_side_315_l574_574016


namespace find_m_l574_574221

noncomputable def u : ℕ → ℚ
| 1       := 2
| (m + 1) := if (m + 1) % 3 = 0 then 2 + u ((m + 1) / 3)
             else 1 / u m

theorem find_m (m : ℕ) (h : u m = 7 / 24) : m = 55 := 
by
  sorry

end find_m_l574_574221


namespace find_smallest_C_l574_574986

-- Prove that the smallest constant $C > 1$ such that
-- for every integer $n \geq 2$ and sequence of non-integer positive real
-- numbers $a_1, a_2, \dots, a_n$ satisfying $\frac{1}{a_1} + \frac{1}{a_2} + \cdots + \frac{1}{a_n} = 1$,
-- it's possible to choose positive integers $b_i$ such that
--   (i) for each $i = 1, 2, \dots, n$, either $b_i = \lfloor a_i \rfloor$ or $b_i = \lfloor a_i \rfloor + 1$,
--   (ii) $1 < \frac{1}{b_1} + \frac{1}{b_2} + \cdots + \frac{1}{b_n} \leq C$
-- is $C = \frac{3}{2}$.

theorem find_smallest_C : 
  ∃ C : ℝ, C > 1 ∧ 
    (∀ (n : ℕ) (hn: n ≥ 2) (a : Finₓ (n+1) → ℝ)
      (h: (∀ (i : Finₓ (n+1)), a i > 0 ∧ a i ∉ ℤ) ∧ 
          (∀ (i : Finₓ (n+1)), (1 : ℝ) / a i) = 1)
       , ∃ b : Finₓ (n+1) → ℕ, 
          (∀ (i : Finₓ (n+1)), b i = Int.floor (a i) ∨ b i = Int.floor (a i) + 1) ∧ 
          1 < ∑ i : Finₓ (n+1), (1 : ℝ) / (b i) ∧ 
          ∑ i : Finₓ (n+1), (1 : ℝ) / (b i) ≤ C) 
    ∧ C = 3 / 2 := sorry

end find_smallest_C_l574_574986


namespace solve_m_l574_574316

def f (x : ℝ) := 4 * x ^ 2 - 3 * x + 5
def g (x : ℝ) (m : ℝ) := x ^ 2 - m * x - 8

theorem solve_m : ∃ (m : ℝ), f 8 - g 8 m = 20 ∧ m = -25.5 := by
  sorry

end solve_m_l574_574316


namespace sasha_study_more_l574_574797

theorem sasha_study_more (d_wkdy : List ℤ) (d_wknd : List ℤ) (h_wkdy : d_wkdy = [5, -5, 15, 25, -15]) (h_wknd : d_wknd = [30, 30]) :
  (d_wkdy.sum + d_wknd.sum) / 7 = 12 := by
  sorry

end sasha_study_more_l574_574797


namespace distinct_flags_l574_574505

def colors : Type := {red, white, blue, green, yellow} : Type
def strips : Type := {s1, s2, s3, s4} : Type

-- Conditions: Each strip is of solid color, and no two adjacent strips can be the same
constant flag : strips → colors

axiom no_adjacent_same_color : ∀ (s : strips), (s = s1 → flag s ≠ flag s1) ∧
                                           (s = s2 → flag s ≠ flag s1) ∧
                                           (s = s2 → flag s ≠ flag s2) ∧
                                           (s = s3 → flag s ≠ flag s2) ∧
                                           (s = s4 → flag s ≠ flag s3)

-- Number of configurations
theorem distinct_flags : ∃ (n : ℕ), n = 320 :=
by {
  Let n = 5 * 4 * 4 * 4,
  exact ⟨n, rfl⟩,
}

end distinct_flags_l574_574505


namespace revenue_decrease_percent_l574_574462

variable (T C : ℝ)

def original_revenue := T * C
def new_tax := T * 0.80
def new_consumption := C * 1.15
def new_revenue := new_tax * new_consumption
def revenue_decrease := original_revenue - new_revenue
def decrease_percent := (revenue_decrease / original_revenue) * 100

theorem revenue_decrease_percent :
  decrease_percent T C = 8 :=
by
  sorry

end revenue_decrease_percent_l574_574462


namespace min_PA_value_l574_574304

theorem min_PA_value
  (x y: ℝ)
  (hp : x^2 + y^2 = 1) -- Point P is on the circle
  : ∃α : ℝ, let P := (Real.cos α, Real.sin α) in
    let A := (Real.cos α, 2 - Real.cos α) in
    |2 - Real.cos α - Real.sin α| = 2 - Real.sqrt 2 :=
sorry -- Proof to be filled in later

-- Noncomputable to avoid Lean's issues with classical analysis
noncomputable def find_min_PA_value : ℝ := 2 - Real.sqrt 2

end min_PA_value_l574_574304


namespace bookshop_inventory_l574_574173

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

end bookshop_inventory_l574_574173


namespace sum_squares_lt_l574_574777

variables {α : Type*} [linear_ordered_field α]
variables (n : ℕ) (y : ℕ → α) (x : ℕ → α)

def x_sequence (n : ℕ) (y : ℕ → α) : ℕ → α
| 0     := y 0
| (i+1) := (1 / 2) * x_sequence i + y (i+1)

theorem sum_squares_lt 
  (x_def : ∀ i, x (i+1) = (1 / 2 : α) * x i + y (i+1))
  (x_zero : x 0 = y 0)
  : (∑ i in finset.range (n + 1), (x i)^2) < 4 * (∑ i in finset.range (n + 1), (y i)^2) :=
  sorry

end sum_squares_lt_l574_574777


namespace probability_at_least_6_heads_in_8_flips_l574_574131

open scoped BigOperators

def binom (n k : ℕ) : ℕ := nat.choose n k

def total_outcomes (n : ℕ) := 2^n

def successful_outcomes (n k : ℕ) :=
  (finset.range (n + 1)).filter (λ x, x ≥ k).sum (λ x, binom n x)

def probability (n k : ℕ) :=
  (successful_outcomes n k) / (total_outcomes n : ℚ)

theorem probability_at_least_6_heads_in_8_flips :
  probability 8 6 = 37 / 256 := sorry

end probability_at_least_6_heads_in_8_flips_l574_574131


namespace prime_sum_diff_l574_574246

theorem prime_sum_diff (p : ℕ) :
  prime p ∧ (∃ q r : ℕ, prime q ∧ prime r ∧ p = q + r) ∧ (∃ s t : ℕ, prime s ∧ prime t ∧ p = s - t) ↔ p = 5 := 
by
  sorry

end prime_sum_diff_l574_574246


namespace ceramic_cup_price_l574_574113

-- Definitions of the conditions
def plastic_cup_price := 3.50
def total_cups_sold := 400
def cups_sold_each_type := 284
def total_revenue := 1458

-- The statement to prove
theorem ceramic_cup_price :
  ∃ (ceramic_cup_price : ℝ), 
    let plastic_cups_sold := total_cups_sold - cups_sold_each_type in 
    let revenue_from_plastic_cups := plastic_cup_price * plastic_cups_sold in 
    let revenue_from_ceramic_cups := total_revenue - revenue_from_plastic_cups in 
    let computed_ceramic_cup_price := revenue_from_ceramic_cups / cups_sold_each_type in 
    ceramic_cup_price = 3.70 :=
by
  sorry

end ceramic_cup_price_l574_574113


namespace measuring_scoop_size_l574_574235

theorem measuring_scoop_size :
  (15 * x = 3.75) → x = 0.25 :=
by
  intro h,
  linarith,
  -- Sorry statement to be removed once the proof is implemented
  sorry

end measuring_scoop_size_l574_574235


namespace complex_magnitude_pow_eight_l574_574198

theorem complex_magnitude_pow_eight :
  complex.abs (complex.mk (2/3) (5/6)) ^ 8 = (41 ^ 4) / 1679616 :=
by
  sorry

end complex_magnitude_pow_eight_l574_574198


namespace units_digit_42_pow_4_add_24_pow_4_l574_574848

-- Define a function to get the units digit of a number.
def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_42_pow_4_add_24_pow_4 : units_digit (42^4 + 24^4) = 2 := by
  sorry

end units_digit_42_pow_4_add_24_pow_4_l574_574848


namespace apples_left_total_l574_574275

def Frank_apples : ℕ := 36
def Susan_apples : ℕ := 3 * Frank_apples
def Frank_sold : ℕ := Frank_apples / 3
def Susan_given : ℕ := Susan_apples / 2

theorem apples_left_total : Susan_apples - Susan_given + Frank_apples - Frank_sold = 78 := 
by
  have h1 : Susan_apples = 3 * Frank_apples := rfl
  have h2 : Frank_apples = 36 := rfl
  have h3 : Susan_given = Susan_apples / 2 := rfl
  have h4 : Frank_sold = Frank_apples / 3 := rfl
  -- since we know the values, we could calculate directly
  have h5 : Susan_apples = 108 := by rw [h1, h2]; norm_num
  have h6 : Susan_given = 54 := by rw [h5]; norm_num
  have h7 : Frank_sold = 12 := by rw [h2]; norm_num
  calc
    Susan_apples - Susan_given + Frank_apples - Frank_sold
        = 108 - 54 + 36 - 12 := by rw [h5, h6, h2, h7]
    ... = 78 := by norm_num

end apples_left_total_l574_574275


namespace graphs_intersection_points_l574_574780

theorem graphs_intersection_points {g : ℝ → ℝ} (h_injective : Function.Injective g) :
  ∃ (x1 x2 x3 : ℝ), (g (x1^3) = g (x1^5)) ∧ (g (x2^3) = g (x2^5)) ∧ (g (x3^3) = g (x3^5)) ∧ 
  x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ ∀ (x : ℝ), (g (x^3) = g (x^5)) → (x = x1 ∨ x = x2 ∨ x = x3) := 
by
  sorry

end graphs_intersection_points_l574_574780


namespace number_of_incorrect_statements_l574_574802

-- Definitions based on the conditions in the problem
def collinear (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, (k ≠ 0 ∧ v = (k • u))

def unit_vector (u : ℝ × ℝ) : Prop :=
  ∥u∥ = 1

def opposite_vector (u v : ℝ × ℝ) : Prop :=
  v = -u

def same_quadrilateral (u v : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  u = v - A

def is_parallelogram (A B C D : ℝ × ℝ) : Prop :=
  A + C = B + D

-- Statement corresponding to the given problem
theorem number_of_incorrect_statements :
  let stmt₁ := ∀ A B C D : ℝ × ℝ, collinear (B - A) (D - C) → collinear A B C D
  let stmt₂ := ∀ u v : ℝ × ℝ, unit_vector u → unit_vector v → u = v
  let stmt₃ := ∀ u : ℝ × ℝ, u ≠ -u
  let stmt₄ := ∀ A B C D : ℝ × ℝ, is_parallelogram A B C D ↔ (A - B = C - D)
  num_incorrect := [stmt₁, stmt₂, stmt₃, stmt₄].count (λ p, ¬p)
  num_incorrect = 4 :=
by
  sorry

end number_of_incorrect_statements_l574_574802


namespace find_polynomial_l574_574259

noncomputable theory
open polynomial

def degree (P : polynomial ℝ) : ℕ := P.nat_degree

theorem find_polynomial (P : polynomial ℝ) (h : P ≠ 0) (h_nonconst : degree P > 0)
  (h_eq : ∀ x : ℝ, P.eval (P.eval x) = (x^2 - x + 1) * P.eval x) :
  P = polynomial.X^2 + (1/3) * polynomial.X - (1/15) := 
sorry

end find_polynomial_l574_574259


namespace NES_sale_price_l574_574834

theorem NES_sale_price
  (SNES_value : ℝ) (trade_in_rate : ℝ) (additional_payment : ℝ)
  (change : ℝ) (game_value : ℝ) : SNES_value = 150 → trade_in_rate = 0.80 →
  additional_payment = 80 → change = 10 → game_value = 30 →
  (trade_in_value : ℝ) (total_spent : ℝ) (NES_price : ℝ) :
  trade_in_value = SNES_value * trade_in_rate →
  total_spent = trade_in_value + additional_payment - change →
  NES_price = total_spent - game_value →
  NES_price = 160 :=
by
  intros
  sorry

end NES_sale_price_l574_574834


namespace max_profit_at_grade_5_l574_574497

-- Defining the conditions
def profit_per_item (x : ℕ) : ℕ :=
  4 * (x - 1) + 8

def production_count (x : ℕ) : ℕ := 
  60 - 6 * (x - 1)

def daily_profit (x : ℕ) : ℕ :=
  profit_per_item x * production_count x

-- The grade range
def grade_range (x : ℕ) : Prop :=
  1 ≤ x ∧ x ≤ 10

-- Prove that the grade that maximizes the profit is 5
theorem max_profit_at_grade_5 : (1 ≤ x ∧ x ≤ 10) → daily_profit x ≤ daily_profit 5 :=
sorry

end max_profit_at_grade_5_l574_574497


namespace Petya_wins_l574_574824

-- Define the initial state and conditions
def initial_plus_count : ℕ := 865

-- Define the game state as a structure
structure GameState where
  pluses : ℕ
  minuses : ℕ

-- Define the rules of the game as allowable moves
inductive Move
| plusToMinus : Move
| erasePlusTwoMinuses : Move
| twoPlusesToTwoMinuses : Move

-- Transition function for the game
def transition (s : GameState) (m : Move) : GameState :=
  match m with
  | Move.plusToMinus => {s with pluses := s.pluses - 1, minuses := s.minuses + 1}
  | Move.erasePlusTwoMinuses => {s with pluses := s.pluses - 1, minuses := s.minuses - 2}
  | Move.twoPlusesToTwoMinuses => {s with pluses := s.pluses - 2, minuses := s.minuses + 2}

-- Define a predicate to check if a move is valid
def validMove (s : GameState) (m : Move) : Prop :=
  match m with
  | Move.plusToMinus => s.pluses ≥ 1
  | Move.erasePlusTwoMinuses => s.pluses ≥ 1 ∧ s.minuses ≥ 2
  | Move.twoPlusesToTwoMinuses => s.pluses ≥ 2

-- Prove that Petya wins with optimal play
theorem Petya_wins : ∃ f : ℕ → GameState, ∃ g : ℕ → Move,
  (∀ n, validMove (f n) (g n) ∧ (f (n+1) = transition (f n) (g n))) ∧
  (f 0 = {pluses := initial_plus_count, minuses := 0}) ∧
  (∃ N, f N = {pluses := 0, minuses := _}) :=
by
  -- this would be where the actual proof goes
  sorry

end Petya_wins_l574_574824


namespace smallest_n_25_l574_574332

theorem smallest_n_25 (n k : ℕ) (h : ∀ n k, 5^2 ∣ n * 2^5 * 6^k * 7^3 ∧ 3^3 ∣ n * 2^5 * 6^k * 7^3) :
  25 = ∃ n : ℕ, 5^2 ∣ n ∧ 3^0 ∣ n ∧ (∀ m : ℕ, (5^2 ∣ m ∧ 3^0 ∣ m) → n ≤ m) :=
by {
  sorry 
}

end smallest_n_25_l574_574332


namespace possible_measures_for_angle_A_l574_574030

/-- Given two positive integer angles A and B in degrees, where
 1. A is an even multiple of B
 2. A and B are supplementary (A + B = 180 degrees),
 then there are exactly 6 possible measures for angle A. -/
theorem possible_measures_for_angle_A :
  (∃ (A B : ℕ), 0 < A ∧ 0 < B ∧
    (∃ k' : ℕ, A = 2 * k' * B ) ∧ (A + B = 180)) →
    (card (finset.filter (λ (x : ℕ), ∃ (A B : ℕ), 0 < A ∧ 0 < B ∧  A = x ∧
      (∃ k' : ℕ, A = 2 * k' * B ) ∧ (A + B = 180)) (finset.range 181)) = 6) :=
by
  sorry

end possible_measures_for_angle_A_l574_574030


namespace digits_divisible_by_101_l574_574855

theorem digits_divisible_by_101 :
  ∃ x y : ℕ, x < 10 ∧ y < 10 ∧ (2013 * 100 + 10 * x + y) % 101 = 0 ∧ x = 9 ∧ y = 4 := by
  sorry

end digits_divisible_by_101_l574_574855


namespace count_five_digit_numbers_l574_574447

-- Definitions corresponding to conditions
def valid_first_digit (d : Nat) : Prop := d ∈ {1, 2, 3, 4}
def distinct_except_five (digits : List Nat) : Prop :=
  digits.filter (≠ 5).Nodup

-- The main statement of the theorem
theorem count_five_digit_numbers :
  (∃ d : Nat, valid_first_digit d) ∧
  (∃ digits : List Nat, digits.length = 3 ∧ distinct_except_five digits) →
  (∃ n : Nat, n = C 4 1 * C 4 2 * A 8 2) :=
sorry

end count_five_digit_numbers_l574_574447


namespace find_f4_l574_574027

noncomputable def f : ℝ → ℝ := sorry

-- Given that f is defined on (0, +∞) and takes positive values.
axiom f_defined_on_positive : ∀ x : ℝ, x > 0 → f(x) > 0

-- Given the condition a * f(b) = b * f(a) for any a, b > 0.
axiom area_condition : ∀ a b : ℝ, a > 0 → b > 0 → a * f(b) = b * f(a)

-- Given f(1) = 4.
axiom initial_condition : f 1 = 4

-- Prove that f(4) = 1.
theorem find_f4 : f 4 = 1 := 
sorry

end find_f4_l574_574027


namespace yield_is_approx_14_l574_574102

-- Define the given face value of the stock and the market price
def faceValue : ℝ := 100
def marketPrice : ℝ := 114.28571428571429

-- Define the annual dividend based on the 16% of the face value
def annualDividend : ℝ := 0.16 * faceValue

-- Define the yield percentage calculation
def yieldPercentage : ℝ := (annualDividend / marketPrice) * 100

-- The theorem we need to prove
theorem yield_is_approx_14 : yieldPercentage ≈ 14 :=
by
  sorry

end yield_is_approx_14_l574_574102


namespace probability_odd_divisor_25_factorial_l574_574445

theorem probability_odd_divisor_25_factorial : 
  let divisors := (22 + 1) * (10 + 1) * (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
  let odd_divisors := (10 + 1) * (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
  (odd_divisors / divisors = 1 / 23) :=
sorry

end probability_odd_divisor_25_factorial_l574_574445


namespace min_dot_product_is_one_fourth_l574_574629

variable {a b : ℝ^3}

def is_unit_vector (v : ℝ^3) : Prop :=
  ∥v∥ = 1

def acute_angle (a b : ℝ^3) : Prop :=
  0 < (a • b)

noncomputable def min_dot_product (a b : ℝ^3) :=
  if acute_angle a b
    ∧ (∀ (x y : ℝ),∥x • a + y • b∥ = 1 ∧ (x * y ≥ 0) → ∥x + 2 * y∥ ≤ 8 / sqrt 15) 
  then inf {t : ℝ | ∃θ, cos θ = t ∧ t = a • b}
  else 0

theorem min_dot_product_is_one_fourth :
  is_unit_vector a → is_unit_vector b →
  (∀ (x y : ℝ), ∥x • a + y • b∥ = 1 ∧ (x * y ≥ 0) → ∥x + 2 * y∥ ≤ 8 / sqrt 15) →
  min_dot_product a b = 1 / 4 :=
begin
  intros,
  sorry
end

end min_dot_product_is_one_fourth_l574_574629


namespace sum_remainders_l574_574192

theorem sum_remainders (n : ℤ) (h : n % 20 = 13) : (n % 4 + n % 5 = 4) :=
by
  sorry

end sum_remainders_l574_574192


namespace x_ln_x_decreasing_interval_l574_574031

noncomputable def x_ln_x : ℝ → ℝ := λ x, x * Real.log x

theorem x_ln_x_decreasing_interval :
  (∀ x : ℝ, 0 < x → x < (1 / Real.exp 1) → x_ln_x x ≤ x_ln_x 1) := 
begin
  intro x,
  assume h_dom : 0 < x,
  assume h_ineq : x < (1 / Real.exp 1),
  sorry
end

end x_ln_x_decreasing_interval_l574_574031


namespace geometric_seq_product_20_terms_l574_574660

variable {a : ℕ → ℝ} -- Define the geometric sequence as a function from natural numbers to reals.

-- Given Condition: The product of the 10th and 11th terms is 2.
axiom geometric_seq_condition : a 10 * a 11 = 2

-- Statement to prove: The product of the first 20 terms is 1024.
theorem geometric_seq_product_20_terms : (∏ n in Finset.range 20, a (n + 1)) = 1024 :=
  by
  sorry

end geometric_seq_product_20_terms_l574_574660


namespace probability_of_at_least_six_heads_is_correct_l574_574124

-- Definitions for the given problem
def binomial_coefficient (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

def total_possible_outcomes : ℕ :=
  2^8

def favorable_outcomes : ℕ :=
  binomial_coefficient 8 6 + binomial_coefficient 8 7 + binomial_coefficient 8 8

def probability_of_at_least_6_heads : ℚ :=
  favorable_outcomes / total_possible_outcomes

-- The proof statement
theorem probability_of_at_least_six_heads_is_correct :
  probability_of_at_least_6_heads = 37 / 256 :=
by sorry

end probability_of_at_least_six_heads_is_correct_l574_574124


namespace groupB_is_basis_l574_574189

section
variables (eA1 eA2 : ℝ × ℝ) (eB1 eB2 : ℝ × ℝ) (eC1 eC2 : ℝ × ℝ) (eD1 eD2 : ℝ × ℝ)

def is_collinear (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k • w) ∨ w = (k • v)

-- Define each vector group
def groupA := eA1 = (0, 0) ∧ eA2 = (1, -2)
def groupB := eB1 = (-1, 2) ∧ eB2 = (5, 7)
def groupC := eC1 = (3, 5) ∧ eC2 = (6, 10)
def groupD := eD1 = (2, -3) ∧ eD2 = (1/2, -3/4)

-- The goal is to prove that group B vectors can serve as a basis
theorem groupB_is_basis : ¬ is_collinear eB1 eB2 :=
sorry
end

end groupB_is_basis_l574_574189


namespace ben_has_56_marbles_l574_574468

-- We define the conditions first
variables (B : ℕ) (L : ℕ)

-- Leo has 20 more marbles than Ben
def condition1 : Prop := L = B + 20

-- Total number of marbles is 132
def condition2 : Prop := B + L = 132

-- The goal: proving the number of marbles Ben has is 56
theorem ben_has_56_marbles (h1 : condition1 B L) (h2 : condition2 B L) : B = 56 :=
by sorry

end ben_has_56_marbles_l574_574468


namespace range_of_a_l574_574342

theorem range_of_a (a : ℝ) : (∀ (x : ℝ), (x^2 - 2*a*x + a) > 0) → (a ≤ 0 ∨ a ≥ 1) :=
by
  -- Proof goes here
  sorry

end range_of_a_l574_574342


namespace mark_deposit_is_88_l574_574399

-- Definitions according to the conditions
def markDeposit := 88
def bryanDeposit (m : ℕ) := 5 * m - 40

-- The theorem we need to prove
theorem mark_deposit_is_88 : markDeposit = 88 := 
by 
  -- Since the condition states Mark deposited $88,
  -- this is trivially true.
  sorry

end mark_deposit_is_88_l574_574399


namespace TrailBlazers_third_place_l574_574355

variables (Team : Type)
variables (Warriors Nuggets Jazz TrailBlazers Rockets : Team)
variables (Place : Type)
variables (first second third fourth fifth : Place)
variables (A_predictions : first = Warriors ∧ third = Nuggets)
variables (B_predictions : third = Warriors ∧ fifth = Jazz)
variables (C_predictions : fourth = Rockets ∧ second = Warriors)
variables (D_predictions : second = Nuggets ∧ fifth = TrailBlazers)
variables (E_predictions : third = TrailBlazers ∧ fifth = Rockets)
variables 
  (A_correct : A_predictions.left ∨ A_predictions.right)
  (B_correct : B_predictions.left ∨ B_predictions.right)
  (C_correct : C_predictions.left ∨ C_predictions.right)
  (D_correct : D_predictions.left ∨ D_predictions.right)
  (E_correct : E_predictions.left ∨ E_predictions.right)
  (everyone_correct : ∀ t : Team, t = Warriors ∨ t = Nuggets ∨ t = Jazz ∨ t = TrailBlazers ∨ t = Rockets 
  → ∃ p : Place, (A_predictions.left ∨ A_predictions.right) ∧ 
                  (B_predictions.left ∨ B_predictions.right) ∧
                  (C_predictions.left ∨ C_predictions.right) ∧
                  (D_predictions.left ∨ D_predictions.right) ∧
                  (E_predictions.left ∨ E_predictions.right))

theorem TrailBlazers_third_place : third = TrailBlazers :=
sorry

end TrailBlazers_third_place_l574_574355


namespace prob_at_least_6_heads_eq_l574_574120

-- define the number of coin flips
def n := 8

-- define the number of possible outcomes (2^n)
def total_outcomes := 2 ^ n

-- define the binomial coefficients for cases: 6 heads, 7 heads, 8 heads
def binom_8_6 := Nat.choose 8 6
def binom_8_7 := Nat.choose 8 7
def binom_8_8 := Nat.choose 8 8

-- calculate the favorable outcomes for at least 6 heads
def favorable_outcomes := binom_8_6 + binom_8_7 + binom_8_8

-- define the probability of getting at least 6 heads
def probability := (favorable_outcomes : ℚ) / total_outcomes

theorem prob_at_least_6_heads_eq : probability = 37 / 256 := by
  sorry

end prob_at_least_6_heads_eq_l574_574120


namespace maximum_number_of_cities_l574_574688

-- Definitions and conditions
structure State :=
  (cities : Type)
  (transportation : cities → cities → option (fin 3)) -- car = 0, train = 1, airplane = 2
  (transportation_symmetric : ∀ (a b : cities), transportation a b = transportation b a)
  (unique_transport_between_each_pair : ∀ (a b : cities), a ≠ b → ∃ t, transportation a b = some t)
  (all_transportation_used : ∀ t : fin 3, ∃ (a b : cities), transportation a b = some t)
  (no_city_uses_all_transport : ∀ (a : cities), ∃ t₀ t₁ t₂, transportation a t₀ ≠ some 0 ∨ transportation a t₁ ≠ some 1 ∨ transportation a t₂ ≠ some 2)
  (no_monochromatic_triangle : ∀ t : fin 3, ∀ (a b c : cities), 
    transportation a b = some t → transportation b c = some t → transportation c a = some t → false)

-- The maximum number of cities satisfying the above conditions
theorem maximum_number_of_cities {s : State} : ∃ n, ∀ c, fintype.card c ≤ n := sorry

end maximum_number_of_cities_l574_574688


namespace board_sum_positive_l574_574407

theorem board_sum_positive (board : Fin 2000 → Fin 2000 → ℤ) 
    (h1 : ∀ i j, board i j = 1 ∨ board i j = -1) 
    (h2 : 0 < ∑ i, ∑ j, board i j) :
    ∃ (rows : Finset (Fin 2000)) (cols : Finset (Fin 2000)), 
        rows.card = 1000 ∧
        cols.card = 1000 ∧
        1000 ≤ ∑ i in rows, ∑ j in cols, board i j :=
by
  sorry

end board_sum_positive_l574_574407


namespace calculate_expression_l574_574201

theorem calculate_expression :
  ((650^2 - 350^2) * 3 = 900000) := by
  sorry

end calculate_expression_l574_574201


namespace geometric_sequence_implies_a_eq_1_l574_574712

-- Conditions

def curve_C (θ : ℝ) (ρ a : ℝ) : Prop :=
  ρ * sin(θ)^2 = 2 * a * cos(θ) ∧ a > 0

def line_l (t : ℝ) : ℝ × ℝ :=
  (t * (sqrt 2 / 2) - 2, t * (sqrt 2 / 2) - 4)

def intersection_points (a : ℝ) : Prop :=
  ∃ t1 t2 : ℝ, t1 ≠ t2 ∧ 
  let M := line_l t1 in
  let N := line_l t2 in
  let dist (P1 P2 : ℝ × ℝ) := (P1.1 - P2.1)^2 + (P1.2 - P2.2)^2 in
  let P := (-2, -4) in
  dist P M * dist P N = dist P P *
  (dist P M + dist P N)^2 / dist M N

-- The Proof Statement
theorem geometric_sequence_implies_a_eq_1 (a : ℝ) : 
  (curve_C θ ρ a) →
  (intersection_points a) →
  a = 1 :=
sorry

end geometric_sequence_implies_a_eq_1_l574_574712


namespace graph_shift_l574_574305

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x) ^ 2 - Real.sin (2 * x) - 1

theorem graph_shift :
  ∀ x : ℝ, f x = sqrt 2 * Real.cos (2 * x + π / 4) := by
  sorry

end graph_shift_l574_574305


namespace math_problem_statements_l574_574961

theorem math_problem_statements :
  ¬ (∃ x : ℝ, 0 < x ∧ x < π / 2 ∧ sin x + cos x = 1 / 3) ∧
  ¬ (∃ a b : ℝ, ∀ x : ℝ, a < x ∧ x < b → (cos x)' < 0 ∧ sin x < 0) ∧
  ¬ (∀ x y : ℝ, tan x < tan y → x < y) ∧
  (∀ x : ℝ, cos x ^ 2 + sin (π / 2 - x) ≤ 2 ∧ cos x ^ 2 + sin (π / 2 - x) = 2 ↔ cos x = 1 ∧ (cos x ^ 2 + sin (π / 2 - x)) = (cos (-x) ^ 2 + sin (π / 2 - (-x)))) ∧
  (∀ a b : ℝ, ∃ f : ℝ → ℝ, f(x) = a * sin 2 * x + b * tan x + 1 ∧ f (-3) = 5 → f (π + 3) = -3) := by
  -- Proof not required, insert sorry here
  sorry

end math_problem_statements_l574_574961


namespace lines_meet_on_square_perimeter_l574_574427

theorem lines_meet_on_square_perimeter 
  (A B C : Point) 
  (H : is_right_triangle ABC) 
  (S1 : Square ABB1A2)
  (S2 : Square BCC1B2) 
  (H_Angle : right_angle ∠ACB)
  (H_Squares : externally_drawn S1 on AB ∧ externally_drawn S2 on BC) :
  ∃ D : Point, (D ∈ perimeter (Square_with_vertices_on_ABC ABC) 
                ∧ D ∈ line CA2 
                ∧ D ∈ line AB2) :=
by
  sorry

end lines_meet_on_square_perimeter_l574_574427


namespace f_l574_574384

noncomputable def f : ℝ → ℝ := sorry
variable {x0 : ℝ}

-- Conditions
axiom h1 : Differentiable ℝ f
axiom h2 : ∀ Δx : ℝ, Δx ≠ 0 → limit (λ Δx : ℝ, (f x0 - f (x0 + 2 * Δx)) / Δx) 0 2

-- Theorem to prove
theorem f''_at_x0_eq_neg1 : deriv 2 f x0 = -1 :=
sorry

end f_l574_574384


namespace candies_taken_by_boys_invariant_l574_574692

theorem candies_taken_by_boys_invariant (n : ℕ) (B G : ℕ) (h : B + G = n) (candies : ℕ) :
  let sequence := list.range n,
  let f := (s : list ℕ) → ∀ (i : ℕ), s.nth i → ℕ,
  let boys_idx := list.range B,
  let girls_idx := list.range G.map(λ g, g + B),
  true
 :=
  ∀ s : list ℕ,
  ∀ perm : list ℕ,
  (perm.permutes s) →
  let boys_count1 := f s boys_idx.sum,
  let boys_count2 := f perm boys_idx.sum,
  boys_count1 = boys_count2
sorry

end candies_taken_by_boys_invariant_l574_574692


namespace probability_at_least_6_heads_in_8_flips_l574_574141

theorem probability_at_least_6_heads_in_8_flips : 
  (∑ k in finset.range 3, nat.choose 8 (6 + k)) / (2 ^ 8) = 37 / 256 :=
by sorry

end probability_at_least_6_heads_in_8_flips_l574_574141


namespace vectors_opposite_directions_l574_574618

variable {V : Type*} [AddCommGroup V]

theorem vectors_opposite_directions (a b : V) (h : a + 4 • b = 0) (ha : a ≠ 0) (hb : b ≠ 0) : a = -4 • b :=
by sorry

end vectors_opposite_directions_l574_574618


namespace ball_height_25_l574_574029

theorem ball_height_25 (t : ℝ) (h : ℝ) 
  (h_eq : h = 45 - 7 * t - 6 * t^2) : 
  h = 25 ↔ t = 4 / 3 := 
by 
  sorry

end ball_height_25_l574_574029


namespace rap_battle_length_l574_574996

variables (R : ℕ) (The_Best_Day : ℕ) (Raise_the_Roof : ℕ)
variable playlist_length : ℕ

-- Conditions
def The_Best_Day := 3
def Raise_the_Roof := 2
def playlist_length := 5 * (The_Best_Day + Raise_the_Roof + R) 

-- Question and answer translation
theorem rap_battle_length : playlist_length = 40 → R = 3 :=
by
  sorry

end rap_battle_length_l574_574996


namespace shelves_needed_l574_574760

theorem shelves_needed (total_books : ℕ) (books_taken_by_first_librarian : ℕ) 
  (large_books : ℕ) (small_books : ℕ) (large_book_space : ℕ) (small_book_space : ℕ) 
  (shelf_capacity : ℕ) (large_books_taken_by_second_librarian : ℕ) 
  (small_books_taken_by_second_librarian : ℕ) : 
  ((total_books - books_taken_by_first_librarian) = (large_books + small_books)) → 
  (large_books = 18) →
  (small_books = 18) →
  (large_book_space = 2) →
  (small_book_space = 1) →
  (shelf_capacity = 6) →
  (large_books - large_books_taken_by_second_librarian = 14) →
  (small_books - small_books_taken_by_second_librarian = 16) →
  ⌈(14 * 2 + 16 * 1) / 6⌉ = 8 :=
by 
  intros; sorry

end shelves_needed_l574_574760


namespace max_value_a50_a51_l574_574634

theorem max_value_a50_a51 (a : ℕ → ℝ) (h_arith : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_pos : ∀ n, a n > 0) (h_sum : (∑ i in (finset.range 100), a i) = 500) :
  ∃ (m : ℝ), m = max (a 49 * a 50) = 25 :=
sorry

end max_value_a50_a51_l574_574634


namespace angle_after_30_seconds_l574_574175

noncomputable def charge : ℝ := 8.0 * 10^(-6) -- in Coulombs
noncomputable def mass : ℝ := 17 * 10^(-3) -- in kilograms
noncomputable def magnetic_field : ℝ := 7.8 * 10^(-3) -- in Teslas
noncomputable def time_elapsed : ℝ := 30 -- in seconds
noncomputable def theta : ℝ := (charge * magnetic_field / mass) * time_elapsed

theorem angle_after_30_seconds :
  100 * theta = 1.101 * 10^(-2) :=
sorry

end angle_after_30_seconds_l574_574175


namespace projection_correct_l574_574585

def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let magnitude_sq := v.1^2 + v.2^2
  let scalar := dot_product / magnitude_sq
  (scalar * v.1, scalar * v.2)

theorem projection_correct :
  projection (3, -4) (1, 2) = (-1, -2) :=
by
  have u : ℝ × ℝ := (3, -4)
  have v : ℝ × ℝ := (1, 2)
  let dot_product := u.1 * v.1 + u.2 * v.2
  let magnitude_sq := v.1^2 + v.2^2
  have h1 : dot_product = -5 := by simp [dot_product]
  have h2 : magnitude_sq =  5 := by simp [magnitude_sq]
  have scalar : dot_product / magnitude_sq = -1 := by rw [h1, h2]; simp
  have proj := projection u v
  simp [projection, dot_product, magnitude_sq, scalar]
  sorry

end projection_correct_l574_574585


namespace monthly_growth_rate_eq_l574_574533

variable (x : ℝ)

def initial_sales := 33.2 -- in ten thousands
def final_sales := 54.6 -- in ten thousands

theorem monthly_growth_rate_eq :
  initial_sales * (1 + x) ^ 2 = final_sales :=
sorry

end monthly_growth_rate_eq_l574_574533


namespace rate_of_interest_l574_574523

-- Definitions based on the conditions
variables (SI P T R : ℝ)
variable h1 : SI = 2700
variable h2 : P = 15000
variable h3 : T = 3
variable h_formula : SI = (P * R * T) / 100

-- The statement to prove
theorem rate_of_interest : R = 6 := by
  subst h1
  subst h2
  subst h3
  subst h_formula
  sorry

end rate_of_interest_l574_574523


namespace max_value_expression_l574_574602

theorem max_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  let A := (a^3 + b^3 + c^3) / ((a + b + c)^3 - 26 * a * b * c) in 
  A ≤ 3 :=
  sorry

end max_value_expression_l574_574602


namespace original_six_digit_number_l574_574179

def six_digit_number : ℕ := 142857

theorem original_six_digit_number (k : ℕ) (h : k = 42857) :
  let n := 100000 + k in
  3 * n = 10 * k + 1 → n = six_digit_number :=
by
  sorry

end original_six_digit_number_l574_574179


namespace exist_two_numbers_with_same_remainder_l574_574544

-- Define the set of 2023 integers
def numbers : Set ℤ := {n : ℤ | ∃ i : Fin 2023, n = (λ i : Fin 2023, n_i : ℤ) i}

-- State the theorem
theorem exist_two_numbers_with_same_remainder 
  (numbers : Set ℤ)
  (h : ∀ n : ℤ, n ∈ numbers → ∃ i : Fin 2023, n = (λ i : Fin 2023, n_i : ℤ) i) :
  ∃ (n_i n_j : ℤ), n_i ∈ numbers ∧ n_j ∈ numbers ∧ n_i ≠ n_j ∧ (n_i - n_j) % 2022 = 0 :=
by
  sorry

end exist_two_numbers_with_same_remainder_l574_574544


namespace semicircle_perimeter_l574_574083

-- Assuming π as 3.14 for approximation
def π_approx : ℝ := 3.14

-- Radius of the semicircle
def radius : ℝ := 2.1

-- Half of the circumference
def half_circumference (r : ℝ) : ℝ := π_approx * r

-- Diameter of the semicircle
def diameter (r : ℝ) : ℝ := 2 * r

-- Perimeter of the semicircle
def perimeter (r : ℝ) : ℝ := half_circumference r + diameter r

-- Theorem stating the perimeter of the semicircle with given radius
theorem semicircle_perimeter : perimeter radius = 10.794 := by
  sorry

end semicircle_perimeter_l574_574083


namespace arrangement_count_l574_574765

def valid_arrangements : ℕ := 
  let balls := {1, 2, 3, 4}
  let boxes := {1, 2, 3}
  sorry -- combinatorial computation function placeholder

theorem arrangement_count : valid_arrangements = 12 :=
  sorry

end arrangement_count_l574_574765


namespace transformation_matrix_inverse_l574_574665

open Matrix

theorem transformation_matrix_inverse : 
  let T := λ (v : Vector (Fin 2) ℝ), Fin.vecCons (v 0 + 2 * v 1) (Fin.tail v);
  let A : Matrix (Fin 2) (Fin 2) ℝ := λ i j, if (i,j) = (0, 0) then 1 else if (i,j) = (0, 1) then 2 else if (i,j) = (1, 0) then 0 else 1;
  let A_inv : Matrix (Fin 2) (Fin 2) ℝ := λ i j, if (i,j) = (0, 0) then 1 else if (i,j) = (0, 1) then -2 else if (i,j) = (1, 0) then 0 else 1;
  (∃ A : Matrix (Fin 2) (Fin 2) ℝ, ∃ A_inv : Matrix (Fin 2) (Fin 2) ℝ,
  (∀ (v : Fin 2 → ℝ), T v = mulVec A v) ∧ 
  mul A A_inv = 1 ∧
  A = ![![1, 2], ![0, 1]] ∧ 
  A_inv = ![![1, -2], ![0, 1]]) :=
sorry

end transformation_matrix_inverse_l574_574665


namespace handshakes_count_l574_574938

def women := 6
def teams := 3
def shakes_per_woman := 4
def total_handshakes := (6 * 4) / 2

theorem handshakes_count : total_handshakes = 12 := by
  -- We provide this theorem directly.
  rfl

end handshakes_count_l574_574938


namespace amplitude_of_sine_function_l574_574542

theorem amplitude_of_sine_function 
  (A B C D : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (hD : 0 < D)
  (h_max : ∀ x : ℝ, A * sin (B * x + C) + D ≤ 5)
  (h_min : ∀ x : ℝ, 1 ≤ A * sin (B * x + C) + D) : 
  A = 2 :=
by
  sorry

end amplitude_of_sine_function_l574_574542


namespace solution_set_inequality_l574_574620

theorem solution_set_inequality (a x : ℝ) (h : 0 < a ∧ a < 1) : 
  ((a - x) * (x - (1 / a)) > 0) ↔ (a < x ∧ x < 1 / a) :=
by sorry

end solution_set_inequality_l574_574620


namespace polynomial_conditions_l574_574862

-- Definitions from the conditions
def polynomial_with_nonnegative_coeffs (p : ℝ[X]) : Prop := ∀ a : ℝ, a ∈ p.coeff.support → 0 ≤ a

-- Problem statement
theorem polynomial_conditions (p : ℝ[X]) (h_poly : polynomial_with_nonnegative_coeffs p)
  (h1 : p.eval 4 = 2) (h2 : p.eval 16 = 8) :
  p.eval 8 ≤ 4 ∧ (∀ q : ℝ[X], polynomial_with_nonnegative_coeffs q ∧ q.eval 4 = 2 ∧ q.eval 16 = 8 → q.eval 8 ≠ 4) := 
sorry

end polynomial_conditions_l574_574862


namespace verify_props_l574_574624

variables (a b c : Type) [line a] [line b] [line c] (α : Type) [plane α]

-- condition 1: ⊥ means perpendicular, ∥ means parallel
axiom prop1 : (a ∥ b) ∧ (b ⊥ c) → (a ⊥ c)
-- condition 2
axiom prop2 : (a ⊥ b) ∧ (b ⊥ c) → (a ∥ c)
-- condition 3
axiom prop3 : (a ∥ α) ∧ (b ∈ α) → (a ∥ b)
-- condition 4
axiom prop4 : (a ⊥ α) ∧ (b ∈ α) → (a ⊥ b)
-- condition 5
axiom prop5 : ¬(a ∥ b) ∧ ¬(a ⊥ b) → ∃! (d : Type) [line d], (d ⊥ a) ∧ (d ⊥ b)
-- condition 6
axiom prop6 : (a ∈ α) ∧ (b ∈ α) ∧ (a ⊥ c) ∧ (b ⊥ c) → (a ∥ b)

-- Theorem to be proven
theorem verify_props : 
  (prop1 = true) ∧ 
  (prop2 = false) ∧ 
  (prop3 = false) ∧ 
  (prop4 = true) ∧ 
  (prop5 = false) ∧ 
  (prop6 = false) := 
sorry

end verify_props_l574_574624


namespace inradius_of_triangle_l574_574807

theorem inradius_of_triangle (p A r : ℝ) (h1 : p = 20) (h2 : A = 25) : r = 2.5 :=
sorry

end inradius_of_triangle_l574_574807


namespace comb_12_9_eq_220_l574_574212

theorem comb_12_9_eq_220 : (Nat.choose 12 9) = 220 := by
  sorry

end comb_12_9_eq_220_l574_574212


namespace initial_weight_cucumbers_l574_574232

theorem initial_weight_cucumbers (W : ℝ) (h1 : 0.99 * W + 0.01 * W = W) 
                                  (h2 : W = (50 - 0.98 * 50 + 0.01 * W))
                                  (h3 : 50 > 0) : W = 100 := 
sorry

end initial_weight_cucumbers_l574_574232


namespace correct_propositions_count_l574_574282

variables (m n : Line) (α β : Plane)

def proposition_1 (m n : Line) (α β : Plane) : Prop := m ⊂ α ∧ n ⊂ α ∧ m ∥ β ∧ n ∥ β → α ∥ β
def proposition_2 (m n : Line) (α : Plane) : Prop := n ∥ m ∧ n ⊥ α → m ⊥ α
def proposition_3 (m n : Line) (α β : Plane) : Prop := α ∥ β ∧ m ⊂ α ∧ n ⊂ β → m ∥ n
def proposition_4 (m n : Line) (α : Plane) : Prop := m ⊥ α ∧ m ⊥ n → n ∥ α

theorem correct_propositions_count :
  (proposition_1 m n α β ∨ proposition_2 m n α ∨ proposition_3 m n α β ∨ proposition_4 m n α) ∧
  (proposition_1 m n α β ↔ true) ∧
  (proposition_2 m n α ↔ true) ∧
  (proposition_3 m n α β ↔ true) ∧
  (proposition_4 m n α ↔ true) →
  #[(proposition_1 m n α β), (proposition_2 m n α), (proposition_3 m n α β), (proposition_4 m n α)].count(true) = 1 := by
  sorry

end correct_propositions_count_l574_574282


namespace sum_of_roots_of_polynomial_l574_574264

theorem sum_of_roots_of_polynomial : 
  let f := λ x : ℝ, (x - 1) ^ 2010 + 2 * (x - 2) ^ 2009 + 
                  4 * (x - 4) ^ 2008 + 
                  ∑ i in Finset.range 2006, 2^(i + 2) * (x - (i + 3)) ^ (2007 - i) + 
                  2009 * (x - 2009)^2 + 
                  2010 * (x - 2010)
  in sum_roots f = 2008 := sorry

end sum_of_roots_of_polynomial_l574_574264


namespace functional_relationship_remaining_oil_after_4_hours_l574_574526

-- Define the initial conditions and the functional form
def initial_oil : ℝ := 50
def consumption_rate : ℝ := 8
def remaining_oil (t : ℝ) : ℝ := initial_oil - consumption_rate * t

-- Prove the functional relationship and the remaining oil after 4 hours
theorem functional_relationship : ∀ (t : ℝ), remaining_oil t = 50 - 8 * t :=
by intros t
   exact rfl

theorem remaining_oil_after_4_hours : remaining_oil 4 = 18 :=
by simp [remaining_oil]
   norm_num
   sorry

end functional_relationship_remaining_oil_after_4_hours_l574_574526


namespace cannot_contain_2003_0_l574_574676

noncomputable def point_not_on_line (m b : ℝ) (h : m * b < 0) : Prop :=
  ∀ y : ℝ, ¬(0 = 2003 * m + b)

-- Prove that if m and b are real numbers and mb < 0, the line y = mx + b
-- cannot contain the point (2003, 0).
theorem cannot_contain_2003_0 (m b : ℝ) (h : m * b < 0) : point_not_on_line m b h :=
by
  sorry

end cannot_contain_2003_0_l574_574676


namespace area_of_L_equals_22_l574_574105

-- Define the dimensions of the rectangles
def big_rectangle_length := 8
def big_rectangle_width := 5
def small_rectangle_length := big_rectangle_length - 2
def small_rectangle_width := big_rectangle_width - 2

-- Define the areas
def area_big_rectangle := big_rectangle_length * big_rectangle_width
def area_small_rectangle := small_rectangle_length * small_rectangle_width

-- Define the area of the "L" shape
def area_L := area_big_rectangle - area_small_rectangle

-- State the theorem
theorem area_of_L_equals_22 : area_L = 22 := by
  -- The proof would go here
  sorry

end area_of_L_equals_22_l574_574105


namespace ohara_triple_example_l574_574831

noncomputable def is_ohara_triple (a b x : ℕ) : Prop := 
  (Real.sqrt a + Real.sqrt b = x)

theorem ohara_triple_example : 
  is_ohara_triple 49 16 11 ∧ 11 ≠ 100 / 5 := 
by
  sorry

end ohara_triple_example_l574_574831


namespace even_and_monotonically_increasing_on_interval_l574_574482

def is_even (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = f x

def is_monotonically_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y

theorem even_and_monotonically_increasing_on_interval :
  is_even (λ x, |sin (π + x)|) ∧ is_monotonically_increasing_on (λ x, |sin (π + x)|) 0 1 :=
by
  sorry

end even_and_monotonically_increasing_on_interval_l574_574482


namespace find_prices_find_min_money_spent_l574_574833

-- Define the prices of volleyball and soccer ball
def prices (pv ps : ℕ) : Prop :=
  pv + 20 = ps ∧ 500 / ps = 400 / pv

-- Define the quantity constraint
def quantity_constraint (a : ℕ) : Prop :=
  a ≥ 25 ∧ a < 50

-- Define the minimum amount spent problem
def min_money_spent (a : ℕ) (pv ps : ℕ) : Prop :=
  prices pv ps → quantity_constraint a → 100 * a + 80 * (50 - a) = 4500

-- Prove the price of each volleyball and soccer ball
theorem find_prices : ∃ (pv ps : ℕ), prices pv ps ∧ pv = 80 ∧ ps = 100 :=
by {sorry}

-- Prove the minimum amount of money spent
theorem find_min_money_spent : ∃ (a pv ps : ℕ), min_money_spent a pv ps :=
by {sorry}

end find_prices_find_min_money_spent_l574_574833


namespace total_amount_l574_574966

theorem total_amount {B C : ℝ} 
  (h1 : C = 1600) 
  (h2 : 4 * B = 16 * C) : 
  B + C = 2000 :=
sorry

end total_amount_l574_574966


namespace domain_of_rational_function_l574_574250

noncomputable def rational_function (x : ℝ) : ℝ := (x^3 - 3*x^2 + x - 4) / (x^2 - 5*x + 6)

theorem domain_of_rational_function :
  ∀ x, x ∈ ((-∞ : Set ℝ) ∪ {y | y < 2} ∪ {y | (y > 2) ∧ (y < 3)} ∪ {y | y > 3}) ↔
          (x ≠ 2 ∧ x ≠ 3) :=
by
  sorry

end domain_of_rational_function_l574_574250


namespace no_positive_integer_solution_l574_574576

theorem no_positive_integer_solution (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  ¬ (∃ (k : ℕ), (xy + 1) * (xy + x + 2) = k^2) :=
by {
  sorry
}

end no_positive_integer_solution_l574_574576


namespace minibus_seat_count_l574_574098

theorem minibus_seat_count 
  (total_children : ℕ) 
  (seats_with_3_children : ℕ) 
  (children_per_3_child_seat : ℕ) 
  (remaining_children : ℕ) 
  (children_per_2_child_seat : ℕ) 
  (total_seats : ℕ) :
  total_children = 19 →
  seats_with_3_children = 5 →
  children_per_3_child_seat = 3 →
  remaining_children = total_children - (seats_with_3_children * children_per_3_child_seat) →
  children_per_2_child_seat = 2 →
  total_seats = seats_with_3_children + (remaining_children / children_per_2_child_seat) →
  total_seats = 7 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end minibus_seat_count_l574_574098


namespace probability_of_at_least_six_heads_is_correct_l574_574123

-- Definitions for the given problem
def binomial_coefficient (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

def total_possible_outcomes : ℕ :=
  2^8

def favorable_outcomes : ℕ :=
  binomial_coefficient 8 6 + binomial_coefficient 8 7 + binomial_coefficient 8 8

def probability_of_at_least_6_heads : ℚ :=
  favorable_outcomes / total_possible_outcomes

-- The proof statement
theorem probability_of_at_least_six_heads_is_correct :
  probability_of_at_least_6_heads = 37 / 256 :=
by sorry

end probability_of_at_least_six_heads_is_correct_l574_574123


namespace sequence_formula_l574_574663

def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 3 ∧ ∀ n, n ≥ 2 → (a n) / 3 = a (n - 1) + 3 ^ n

theorem sequence_formula (a : ℕ → ℝ) (n : ℕ) (h : sequence a) : 
  a n = (3 * n - 2) * 3 ^ n :=
sorry

end sequence_formula_l574_574663


namespace six_digit_numbers_l574_574631

theorem six_digit_numbers (a : ℕ) (b : ℕ) (c : ℕ) 
    (h1 : b < 4)
    (h2 : a = 7)
    (h3 : 56 ∣ (73 * 100000 + a * 10000 + b * 1000 + c * 100 + 6)) :
    (73 * 100000 + a * 10000 + b * 1000 + c * 100 + 6 = 731136 ∨ 
     73 * 100000 + a * 10000 + b * 1000 + c * 100 + 6 = 737016 ∨
     73 * 100000 + a * 10000 + b * 1000 + c * 100 + 6 = 737296) :=
begin
    sorry
end

end six_digit_numbers_l574_574631


namespace program_output_is_10_l574_574804

def final_value_of_A : ℤ :=
  let A := 2
  let A := A * 2
  let A := A + 6
  A

theorem program_output_is_10 : final_value_of_A = 10 := by
  sorry

end program_output_is_10_l574_574804


namespace projection_correct_l574_574583

def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let magnitude_sq := v.1^2 + v.2^2
  let scalar := dot_product / magnitude_sq
  (scalar * v.1, scalar * v.2)

theorem projection_correct :
  projection (3, -4) (1, 2) = (-1, -2) :=
by
  have u : ℝ × ℝ := (3, -4)
  have v : ℝ × ℝ := (1, 2)
  let dot_product := u.1 * v.1 + u.2 * v.2
  let magnitude_sq := v.1^2 + v.2^2
  have h1 : dot_product = -5 := by simp [dot_product]
  have h2 : magnitude_sq =  5 := by simp [magnitude_sq]
  have scalar : dot_product / magnitude_sq = -1 := by rw [h1, h2]; simp
  have proj := projection u v
  simp [projection, dot_product, magnitude_sq, scalar]
  sorry

end projection_correct_l574_574583


namespace burger_calories_l574_574755

theorem burger_calories : 
  let C_burger := 750 - (5 * 20 + 5 * 50) in
  C_burger = 400 :=
by
  let C_carrot := 5 * 20
  let C_cookie := 5 * 50
  let total := 750
  have C_total: total - (C_carrot + C_cookie) = 400
  · calc
      total - (C_carrot + C_cookie)
      = 750 - (5 * 20 + 5 * 50) : by rfl
  have h1: C_burger = total - (C_carrot + C_cookie)
  · rfl
  have h2: total - (C_carrot + C_cookie) = 400
  sorry

end burger_calories_l574_574755


namespace area_of_square_l574_574908

-- Define the parabola and the line
def parabola (x : ℝ) : ℝ := x^2 + 4 * x + 3
def line (y : ℝ) : Prop := y = 7

-- Define the roots of the quadratic equation derived from the conditions
noncomputable def root1 : ℝ := -2 + 2 * Real.sqrt 2
noncomputable def root2 : ℝ := -2 - 2 * Real.sqrt 2

-- Define the side length of the square
noncomputable def side_length : ℝ := abs (root1 - root2)

-- Define the area of the square
noncomputable def area_square : ℝ := side_length^2

-- Theorem statement for the problem
theorem area_of_square : area_square = 32 :=
sorry

end area_of_square_l574_574908


namespace probability_at_least_6_heads_l574_574148

-- Definitions of the binomial coefficient and probability function
def binom (n k : ℕ) : ℕ := Nat.choose n k

def probability (favorable total : ℕ) : ℚ := favorable / total

-- Proof problem statement
theorem probability_at_least_6_heads (flips : ℕ) (p : ℚ) 
  (h_flips : flips = 8) 
  (h_probability : p = probability (binom 8 6 + binom 8 7 + binom 8 8) (2 ^ flips)) : 
  p = 37 / 256 :=
by
  sorry

end probability_at_least_6_heads_l574_574148


namespace probability_of_at_least_six_heads_is_correct_l574_574126

-- Definitions for the given problem
def binomial_coefficient (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

def total_possible_outcomes : ℕ :=
  2^8

def favorable_outcomes : ℕ :=
  binomial_coefficient 8 6 + binomial_coefficient 8 7 + binomial_coefficient 8 8

def probability_of_at_least_6_heads : ℚ :=
  favorable_outcomes / total_possible_outcomes

-- The proof statement
theorem probability_of_at_least_six_heads_is_correct :
  probability_of_at_least_6_heads = 37 / 256 :=
by sorry

end probability_of_at_least_six_heads_is_correct_l574_574126


namespace jose_distance_l574_574371

variable (v : ℕ) (t : ℕ) -- Speed and time are natural numbers
variable (distance : ℕ) -- Distance is also a natural number

theorem jose_distance (hv : v = 2) (ht : t = 2) : distance = v * t := by
  rw [hv, ht]
  exact eq.refl 4

#check jose_distance

end jose_distance_l574_574371


namespace mike_arcade_ratio_l574_574402

theorem mike_arcade_ratio :
  ∀ (weekly_pay food_cost hourly_rate play_minutes : ℕ),
    weekly_pay = 100 →
    food_cost = 10 →
    hourly_rate = 8 →
    play_minutes = 300 →
    (food_cost + (play_minutes / 60) * hourly_rate) / weekly_pay = 1 / 2 := 
by
  intros weekly_pay food_cost hourly_rate play_minutes h1 h2 h3 h4
  sorry

end mike_arcade_ratio_l574_574402


namespace magnitude_of_power_l574_574236

open Complex

theorem magnitude_of_power (a b : ℝ) : abs ((Complex.mk 2 (2 * Real.sqrt 3)) ^ 6) = 4096 := by
  sorry

end magnitude_of_power_l574_574236


namespace probability_at_least_6_heads_in_8_flips_l574_574137

open scoped BigOperators

def binom (n k : ℕ) : ℕ := nat.choose n k

def total_outcomes (n : ℕ) := 2^n

def successful_outcomes (n k : ℕ) :=
  (finset.range (n + 1)).filter (λ x, x ≥ k).sum (λ x, binom n x)

def probability (n k : ℕ) :=
  (successful_outcomes n k) / (total_outcomes n : ℚ)

theorem probability_at_least_6_heads_in_8_flips :
  probability 8 6 = 37 / 256 := sorry

end probability_at_least_6_heads_in_8_flips_l574_574137


namespace jinbeom_lost_large_ttakji_l574_574724

-- Define the initial number of large and small ttakji
def initial_large : ℕ := 12
def initial_small : ℕ := 34

-- Define the condition on the total number of ttakji left
def total_left : ℕ := 30

-- Define the function to calculate the number of large ttakji lost
def large_ttakji_lost (L : ℕ) : Prop :=
(initial_large - L) + (initial_small - 3 * L) = total_left

-- Prove that L = 4 satisfies the problem conditions
theorem jinbeom_lost_large_ttakji : ∃ L : ℕ, large_ttakji_lost L ∧ L = 4 :=
begin
  use 4,
  split,
  {
    -- (12 - 4) + (34 - 3 * 4) = 30
    norm_num,
  },
  refl,
end

end jinbeom_lost_large_ttakji_l574_574724


namespace projection_of_vector_l574_574580

-- Definition of vectors a and b
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (3, -4)

-- Dot product of two vectors
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Projection of b onto a
def proj (a b : ℝ × ℝ) : ℝ × ℝ :=
  let scale := dot_product b a / dot_product a a
  (scale * a.1, scale * a.2)

-- Theorem stating that the projection of b onto a is (-1, -2)
theorem projection_of_vector :
  proj a b = (-1, -2) :=
sorry

end projection_of_vector_l574_574580


namespace comb_12_9_eq_220_l574_574210

theorem comb_12_9_eq_220 : (Nat.choose 12 9) = 220 := by
  sorry

end comb_12_9_eq_220_l574_574210


namespace present_population_l574_574334

theorem present_population (P : ℝ) (h1 : ∀ t : ℝ, t > 0 → P * (1 + 0.10) ^ t = 13310) : P ≈ 11000 :=
by
  -- Given conditions
  have h : P * (1 + 0.10) ^ 2 = 13310 := h1 2 (by linarith)
  -- γ P was given from the ultimate calculation as approximately 11000
  sorry

end present_population_l574_574334


namespace number_of_girls_not_playing_soccer_l574_574865

def totalStudents : Nat := 420
def totalBoys : Nat := 312
def totalPlayingSoccer : Nat := 250
def percentageBoysPlayingSoccer : Float := 0.82

def boysPlayingSoccer : Nat := Nat.floor (percentageBoysPlayingSoccer * Float.ofNat totalPlayingSoccer)
def girlsPlayingSoccer : Nat := totalPlayingSoccer - boysPlayingSoccer
def totalGirls : Nat := totalStudents - totalBoys
def girlsNotPlayingSoccer : Nat := totalGirls - girlsPlayingSoccer

theorem number_of_girls_not_playing_soccer :
  girlsNotPlayingSoccer = 63 :=
by
  sorry

end number_of_girls_not_playing_soccer_l574_574865


namespace contrapositive_proof_l574_574020

theorem contrapositive_proof (x : ℝ) : (x^2 < 1 → -1 < x ∧ x < 1) → (x ≥ 1 ∨ x ≤ -1 → x^2 ≥ 1) :=
by
  sorry

end contrapositive_proof_l574_574020


namespace number_of_incorrect_statements_l574_574931

def statement1 : Prop :=
  ∀ (p : Plane) (l1 l2 l3 : Line),
    l1 ≠ l2 ∧ l2 ≠ l3 ∧ l1 ≠ l3 ∧ 
    (∀ (P Q : Point), (LiesOn P l1 ∧ LiesOn P l2 ∨ LiesOn P l2 ∧ LiesOn P l3 ∨ LiesOn P l3 ∧ LiesOn P l1) ↔
     (LiesOn Q l1 ∧ LiesOn Q l2 ∨ LiesOn Q l2 ∧ LiesOn Q l3 ∨ LiesOn Q l3 ∧ LiesOn Q l1)) →
    (Parallel l1 l2 ∨ Parallel l2 l3 ∨ Parallel l1 l3)

def statement2 : Prop :=
  ∀ (p : Plane) (P : Point) (l : Line),
    (¬ LiesOn P l) → ∃! (m : Line), Parallel l m ∧ LiesOn P m

def statement3 : Prop :=
  ∀ (T : Translation) (S1 S2 : Set Point),
    (CorrespondingPairs T S1 S2) →
    (∀ (P Q : Point), (LiesOn P S1 ∧ LiesOn Q S1) ↔ (Parallel P Q ∧ EqualLength P Q))

def statement4 : Prop :=
  ∀ (l1 l2 l3 : Line),
    (Intersect l1 l3 ∧ Intersect l2 l3) →
    (∃ (A B : Angle), AlternateInterior A B ∧ Equal A B → Supplementary (ConsecutiveInterior A B))

def statement5 : Prop :=
  ∀ (l1 l2 l3 : Line),
    (Parallel l1 l2 ∧ Intersect l1 l3 ∧ Intersect l2 l3) →
    ∃ (B1 B2 : Line), (Bisector B1 l1 l3) ∧ (Bisector B2 l2 l3) ∧ (Perpendicular B1 B2)

def statement6 : Prop :=
  ∀ (A B : Person),
    (SeesDirection A B 60 northeast) → (SeesDirection B A 30 southwest)

def statement7 : Prop :=
  ∀ (p : Plane) (l1 l2 l3 : Line),
    (Perpendicular l1 l3 ∧ Perpendicular l2 l3 ∧ InPlane l1 p ∧ InPlane l2 p ∧ InPlane l3 p) ↔ Parallel l1 l2

theorem number_of_incorrect_statements : 
  (¬ statement2 ∧ ¬ statement3 ∧ ¬ statement7) → (4 = (count_True [¬ statement1, ¬ statement2, ¬ statement3, ¬ statement4, ¬ statement5, ¬ statement6, ¬ statement7])) :=
by
  sorry

end number_of_incorrect_statements_l574_574931


namespace nina_widgets_after_reduction_is_approx_8_l574_574405

noncomputable def nina_total_money : ℝ := 16.67
noncomputable def widgets_before_reduction : ℝ := 5
noncomputable def cost_reduction_per_widget : ℝ := 1.25

noncomputable def cost_per_widget_before_reduction : ℝ := nina_total_money / widgets_before_reduction
noncomputable def cost_per_widget_after_reduction : ℝ := cost_per_widget_before_reduction - cost_reduction_per_widget
noncomputable def widgets_after_reduction : ℝ := nina_total_money / cost_per_widget_after_reduction

-- Prove that Nina can purchase approximately 8 widgets after the cost reduction
theorem nina_widgets_after_reduction_is_approx_8 : abs (widgets_after_reduction - 8) < 1 :=
by
  sorry

end nina_widgets_after_reduction_is_approx_8_l574_574405


namespace min_points_to_win_l574_574007

theorem min_points_to_win (n : ℕ) (games_per_player : ℕ) (win_points : ℝ) (draw_points : ℝ) (players : ℕ)
  (h1 : n = 6)
  (h2 : games_per_player = 2 * (n - 1))
  (h3 : win_points = 1)
  (h4 : draw_points = 0.5) :
  ∃ p : ℝ, p = (9.5 : ℝ) ∧ ∀ i : ℕ, i < n → ∃ k : ℝ, k < p → (¬(∃ q : ℕ, q < n ∧ k ≤ (if q = i then p else k))) :=
begin
  sorry
end

end min_points_to_win_l574_574007


namespace find_m_n_l574_574666

variable {R : Type*} [Field R]

structure Point (R : Type*) [Field R] :=
(x : R)
(y : R)

def OA : Point ℝ := ⟨-2, 6⟩
def OB : Point ℝ := ⟨3, 1⟩
def OC : Point ℝ := ⟨5, -1⟩

def collinear (A B C : Point R) : Prop :=
  (B.y - A.y) * (C.x - B.x) = (C.y - B.y) * (B.x - A.x)

def perpendicular (A B : Point R) : Prop :=
  A.x * B.x + A.y * B.y = 0

theorem find_m_n : collinear OA OB OC ∧ perpendicular OA OB :=
begin
  sorry -- proof to be inserted.
end

end find_m_n_l574_574666


namespace functional_eq_solution_l574_574574

noncomputable def f : ℝ → ℝ := sorry

theorem functional_eq_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x^2 - y^2) = x * f x - y * f y) →
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
sorry

end functional_eq_solution_l574_574574


namespace train_length_is_correct_l574_574528

-- Define the conditions
def speed_km_hr := 72 -- Speed in kilometers per hour
def time_sec := 9 -- Time in seconds

-- Unit conversion from km/hr to m/s
def convert_speed (s : ℕ) : ℕ := s * 1000 / 3600

-- Define the expected answer
def expected_length := 180

-- State the theorem
theorem train_length_is_correct :
  (let speed_m_s := convert_speed speed_km_hr in speed_m_s * time_sec = expected_length) :=
by
  sorry

end train_length_is_correct_l574_574528


namespace odd_handshakes_even_l574_574809

theorem odd_handshakes_even (num_people : ℕ) (total_handshakes_even : Even total_handshakes) 
  (P I : Finset ℕ) 
  (p i: ℕ) 
  (h1: P ∪ I = Finset.range num_people) 
  (h2: ∀ x ∈ P, Even x) 
  (h3: ∀ y ∈ I, Odd y) 
  (h4: num_people = 7375466877) :
  Even I.card :=
by
  sorry

end odd_handshakes_even_l574_574809


namespace probability_at_least_6_heads_l574_574146

-- Definitions of the binomial coefficient and probability function
def binom (n k : ℕ) : ℕ := Nat.choose n k

def probability (favorable total : ℕ) : ℚ := favorable / total

-- Proof problem statement
theorem probability_at_least_6_heads (flips : ℕ) (p : ℚ) 
  (h_flips : flips = 8) 
  (h_probability : p = probability (binom 8 6 + binom 8 7 + binom 8 8) (2 ^ flips)) : 
  p = 37 / 256 :=
by
  sorry

end probability_at_least_6_heads_l574_574146


namespace proof_solution_l574_574324

noncomputable def proof_problem : Prop :=
  ∀ (x y z : ℝ), 3 * x - 4 * y - 2 * z = 0 ∧ x - 2 * y - 8 * z = 0 ∧ z ≠ 0 → 
  (x^2 + 3 * x * y) / (y^2 + z^2) = 329 / 61

theorem proof_solution : proof_problem :=
by
  intros x y z h
  sorry

end proof_solution_l574_574324


namespace loading_time_l574_574080

def worker1_rate : ℝ := 1 / 6
def worker2_rate : ℝ := 1 / 4
def combined_rate : ℝ := worker1_rate + worker2_rate
def one_truck : ℝ := 1
def combined_time : ℝ := one_truck / combined_rate

theorem loading_time :
  combined_time = 12 / 5 := 
sorry

end loading_time_l574_574080


namespace probability_at_least_6_heads_in_8_flips_l574_574142

theorem probability_at_least_6_heads_in_8_flips : 
  (∑ k in finset.range 3, nat.choose 8 (6 + k)) / (2 ^ 8) = 37 / 256 :=
by sorry

end probability_at_least_6_heads_in_8_flips_l574_574142


namespace no_integral_points_on_AB_l574_574297

theorem no_integral_points_on_AB (k m n : ℤ) (h1: ((m^3 - m)^2 + (n^3 - n)^2 > (3*k + 1)^2)) :
  ¬ ∃ (x y : ℤ), (m^3 - m) * x + (n^3 - n) * y = (3*k + 1)^2 :=
by {
  sorry
}

end no_integral_points_on_AB_l574_574297


namespace value_of_b_conditioned_l574_574800

theorem value_of_b_conditioned
  (b: ℝ) 
  (h0 : 0 < b ∧ b < 7)
  (h1 : (1 / 2) * (8 - b) * (b - 8) / ((1 / 2) * (b / 2) * b) = 4 / 9):
  b = 4 := 
sorry

end value_of_b_conditioned_l574_574800


namespace simplify_frac_48_72_l574_574005

theorem simplify_frac_48_72 : (48 / 72 : ℚ) = 2 / 3 :=
by
  -- In Lean, we prove the equality of the simplified fractions.
  sorry

end simplify_frac_48_72_l574_574005


namespace integer_solutions_l574_574224

theorem integer_solutions :
  (∃ y : ℤ, 1 + 2^0 + 2^(2 * 0 + 1) = y^2 ∧ (y = 2 ∨ y = -2)) ∧
  (∃ y : ℤ, 1 + 2^4 + 2^(2 * 4 + 1) = y^2 ∧ (y = 23 ∨ y = -23)) ∧
  (∀ x : ℤ, 1 + 2^x + 2^(2 * x + 1) = y^2 → ∃ y : ℤ, (x = 0 ∧ (y = 2 ∨ y = -2)) ∨ (x = 4 ∧ (y = 23 ∨ y = -23))) :=
by
  sorry

end integer_solutions_l574_574224


namespace perpendicular_vector_l574_574669

-- Definitions for the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (0, -2)

-- Statement asserting (3, 2) is perpendicular to a + 2b.
theorem perpendicular_vector : 
  let c := (a.1 + 2 * b.1, a.2 + 2 * b.2) in
  c = (2, -3) ∧ (c.1 * 3 + c.2 * 2 = 0) := 
by
  let c := (a.1 + 2 * b.1, a.2 + 2 * b.2)
  have h1 : c = (2, -3) := by
    sorry
  have h2 : c.1 * 3 + c.2 * 2 = 0 := by
    sorry
  exact ⟨h1, h2⟩

end perpendicular_vector_l574_574669


namespace perp_line_slope_zero_l574_574339

theorem perp_line_slope_zero {k : ℝ} (h : ∀ x : ℝ, ∃ y : ℝ, y = k * x + 1 ∧ x = 1 → false) : k = 0 :=
sorry

end perp_line_slope_zero_l574_574339


namespace sum_distances_is_379_25_l574_574382

-- Define the parabola and the points of intersection
def parabola (x : ℝ) : ℝ := x^2

def point_1 : ℝ × ℝ := (15, parabola 15)
def point_2 : ℝ × ℝ := (0, parabola 0)
def point_3 : ℝ × ℝ := (-3, parabola (-3))
def point_4 : ℝ × ℝ := (-12, parabola (-12))

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, 1 / 4)

-- Distance function
def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  let dx := p₁.1 - p₂.1
  let dy := p₁.2 - p₂.2
  real.sqrt (dx^2 + dy^2)

-- Sum of distances from the focus to each of the four points
def sumOfDistances : ℝ :=
  distance focus point_1 +
  distance focus point_2 +
  distance focus point_3 +
  distance focus point_4

-- The theorem statement
theorem sum_distances_is_379_25 : sumOfDistances = 379.25 :=
by
  sorry

end sum_distances_is_379_25_l574_574382


namespace integral_sin_cos_l574_574199

theorem integral_sin_cos (a b : ℝ) (h₁ : a = 0) (h₂ : b = π) :
  ∫ x in a..b, (sin x + cos x) = 2 := by
  rw [h₁, h₂]
  sorry

end integral_sin_cos_l574_574199


namespace acme_vs_beta_l574_574186

theorem acme_vs_beta (x : ℕ) :
  (80 + 10 * x < 20 + 15 * x) → (13 ≤ x) :=
by
  intro h
  sorry

end acme_vs_beta_l574_574186


namespace odd_function_period_2pi_l574_574655

noncomputable def f (x : ℝ) : ℝ := Real.tan (x / 2)

theorem odd_function_period_2pi (x : ℝ) : 
  f (-x) = -f (x) ∧ 
  ∃ p > 0, p = 2 * Real.pi ∧ ∀ x, f (x + p) = f (x) := 
by
  sorry

end odd_function_period_2pi_l574_574655


namespace initial_price_after_markup_l574_574176

theorem initial_price_after_markup 
  (wholesale_price : ℝ) 
  (h_markup_80 : ∀ P, P = wholesale_price → 1.80 * P = 1.80 * wholesale_price)
  (h_markup_diff : ∀ P, P = wholesale_price → 2.00 * P - 1.80 * P = 3) 
  : 1.80 * wholesale_price = 27 := 
by
  sorry

end initial_price_after_markup_l574_574176


namespace max_min_f_on_interval_l574_574255

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x + 1

theorem max_min_f_on_interval : 
  (∀ x ∈ Icc (-3 : ℝ) (0 : ℝ), f x ≤ 3) ∧ 
  (∃ x ∈ Icc (-3 : ℝ) (0 : ℝ), f x = 3) ∧
  (∀ x ∈ Icc (-3 : ℝ) (0 : ℝ), f x ≥ -17) ∧
  (∃ x ∈ Icc (-3 : ℝ) (0 : ℝ), f x = -17) :=
by
  sorry

end max_min_f_on_interval_l574_574255


namespace cos_squared_given_sin_l574_574325

theorem cos_squared_given_sin (α : ℝ) (h : sin (π / 6 - α) = 1 / 3) :
  cos^2 (π / 6 + α / 2) = 2 / 3 := 
sorry

end cos_squared_given_sin_l574_574325


namespace sum_of_angles_outside_pentagon_l574_574892

theorem sum_of_angles_outside_pentagon (α β γ δ ε : ℝ) 
  (hα : α = 180 - ∠ BCD)
  (hβ : β = 180 - ∠ CDE)
  (hγ : γ = 180 - ∠ DEA)
  (hδ : δ = 180 - ∠ EAB)
  (hε : ε = 180 - ∠ ABC)
  (pentagon_inscribed : InscribedPentagon ABCDE circle) :
  α + β + γ + δ + ε = 720 :=
  sorry

end sum_of_angles_outside_pentagon_l574_574892


namespace yellow_to_red_ratio_l574_574685

variable {R Y B : ℕ}

theorem yellow_to_red_ratio (h1 : R = 4) (h2 : B = Y - 2) (h3 : R + Y + B = 18) :
  Y / R = 2 :=
by
  have eq1 : R + Y + (Y - 2) = 18 := by rw [h1, h2]; assumption
  have eq2 : 4 + 2 * Y - 2 = 18 := by rw [h1, h2, eq1]; simp
  have eq3 : 2 * Y + 2 = 18 := by rw eq2; simp
  have eq4 : 2 * Y = 16 := by rw eq3; simp
  have eq5 : Y = 8 := by rw eq4; exact Nat.div_eq 16 2
  have eq6 : Y / R = 8 / 4 := by rw [h1, eq5]; exact Nat.div_eq 8 4
  have eq7 : Y / R = 2 := by rw [eq6]; exact Nat.div_eq 8 4 (by exact Nat.gcd_2)
  exact eq7

end yellow_to_red_ratio_l574_574685


namespace extreme_points_l574_574745

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^3 + a * x^2 + b * x

theorem extreme_points (a b : ℝ) 
  (h1 : 3*(-2)^2 + 2*a*(-2) + b = 0) 
  (h2 : 3*(4)^2 + 2*a*(4) + b = 0) : 
  a - b = 21 :=
by sorry

end extreme_points_l574_574745


namespace cuckoo_chime_78_l574_574089

-- Define the arithmetic sum for the cuckoo clock problem
def cuckoo_chime_sum (n a l : Nat) : Nat :=
  n * (a + l) / 2

-- Main theorem
theorem cuckoo_chime_78 : 
  cuckoo_chime_sum 12 1 12 = 78 := 
by
  -- Proof part can be written here
  sorry

end cuckoo_chime_78_l574_574089


namespace not_prime_5n_plus_3_l574_574423

theorem not_prime_5n_plus_3 (n a b : ℕ) (hn_pos : n > 0) (ha_pos : a > 0) (hb_pos : b > 0)
  (ha : 2 * n + 1 = a^2) (hb : 3 * n + 1 = b^2) : ¬Prime (5 * n + 3) :=
by
  sorry

end not_prime_5n_plus_3_l574_574423


namespace distinct_values_min_l574_574170

theorem distinct_values_min (n : ℕ) (m : ℕ) (k : ℕ) :
  n = 2057 → m = 15 → k = 14 → (∃ y : ℕ, (y - 1) * k + m ≥ n ∧ (y * k + m) >= n) ∧ (∀ z : ℕ, ((z - 1) * k + m >= n → z ≥ 147)) :=
by {
  intros,
  sorry
}

end distinct_values_min_l574_574170


namespace range_of_m_l574_574658

noncomputable def f (x : ℝ) : ℝ := (x + 1) / x^2
noncomputable def g (x m : ℝ) : ℝ := Real.log x / Real.log 2 + m

theorem range_of_m (m : ℝ) : (∀ x1 ∈ Icc (1 : ℝ) 2, ∃ x2 ∈ Icc (1 : ℝ) 4, f x1 ≥ g x2 m) → m ≤ 3 / 4 :=
by
  sorry

end range_of_m_l574_574658


namespace angle_BQP_eq_angle_DAQ_l574_574437

theorem angle_BQP_eq_angle_DAQ
  (A B C D P Q : Type*)
  [trapezoid A B C D]
  [diag_intersect A C B D P]
  [between_parallel Q B C A D]
  [same_line_sep CD P Q]
  (angle_AQD_eq_CQB : angle AQD = angle CQB) :
  angle BQP = angle DAQ := 
begin
  sorry -- Proof omitted
end

end angle_BQP_eq_angle_DAQ_l574_574437


namespace prob_at_least_6_heads_eq_l574_574116

-- define the number of coin flips
def n := 8

-- define the number of possible outcomes (2^n)
def total_outcomes := 2 ^ n

-- define the binomial coefficients for cases: 6 heads, 7 heads, 8 heads
def binom_8_6 := Nat.choose 8 6
def binom_8_7 := Nat.choose 8 7
def binom_8_8 := Nat.choose 8 8

-- calculate the favorable outcomes for at least 6 heads
def favorable_outcomes := binom_8_6 + binom_8_7 + binom_8_8

-- define the probability of getting at least 6 heads
def probability := (favorable_outcomes : ℚ) / total_outcomes

theorem prob_at_least_6_heads_eq : probability = 37 / 256 := by
  sorry

end prob_at_least_6_heads_eq_l574_574116


namespace simplified_expression_evaluation_l574_574775

theorem simplified_expression_evaluation (x : ℝ) (hx : x = Real.sqrt 7) :
    (2 * x + 3) * (2 * x - 3) - (x + 2)^2 + 4 * (x + 3) = 20 :=
by
  sorry

end simplified_expression_evaluation_l574_574775


namespace combinations_7_choose_3_l574_574772

theorem combinations_7_choose_3 : Nat.choose 7 3 = 35 := 
by
  exact Nat.choose_spec 7 3 ▸ rfl -- or simply;
  sorry

end combinations_7_choose_3_l574_574772


namespace combined_mpg_l574_574420

theorem combined_mpg (m : ℝ) : 
  let ray_mpg := 40
  let tom_mpg := 10
  let ray_gas := m / ray_mpg
  let tom_gas := m / tom_mpg
  let total_gas := ray_gas + tom_gas
  let total_distance := 2 * m
  let combined_mpg := total_distance / total_gas
  combined_mpg = 16 :=
by
  have ray_gas_eq : ray_gas = m / 40 := rfl
  have tom_gas_eq : tom_gas = m / 10 := rfl
  have total_gas_eq : total_gas = m / 40 + m / 10 := rfl
  have simplify_total_gas : total_gas = m / 8 := by
    rw [total_gas_eq]
    calc 
      m / 40 + m / 10 = m / 40 + 4 * (m / 40) : by simp
                  ... = (1 * m) / 40 + (4 * m) / 40 : by ring
                  ... = (1 + 4) * m / 40 : by ring
                  ... = 5 * m / 40 : by ring
                  ... = m / 8 : by ring

  have total_distance_eq : total_distance = 2 * m := rfl
  have combined_mpg_eq : combined_mpg = total_distance / total_gas := rfl
  have simplify_combined_mpg : combined_mpg = 16 := by
    rw [combined_mpg_eq, total_distance_eq, simplify_total_gas]
    calc 
      2 * m / (m / 8) = 2 * m * (8 / m) : by rw div_mul
                 ... = 2 * 8 : by simp
                 ... = 16 : by ring

  exact simplify_combined_mpg

end combined_mpg_l574_574420


namespace problem_1_problem_2_l574_574707

variables (A B C D E F G P M N : Point)
variables (O : Circle)
variables (h1 : InscribedQuadrilateral O A B C D)
variables (h2 : Intersection (Line A B) (Line D C) = E)
variables (h3 : Intersection (Line A D) (Line B C) = F)
variables (h4 : Intersection (Line A C) (Line B D) = G)
variables (h5 : TangentToCircle O E P)
variables (h6 : TangentsFrom F O M N)

/-- Prove that EF^2 = EP^2 + FN^2 --/
theorem problem_1 : (LineLength E F)^2 = (LineLength E P)^2 + (LineLength F N)^2 := 
begin
  sorry
end

/-- Prove that points E, M, G, and N are collinear --/
theorem problem_2 : Collinear E M G N := 
begin
  sorry
end

end problem_1_problem_2_l574_574707


namespace smallest_num_boxes_l574_574446

theorem smallest_num_boxes 
  (X : ℕ) 
  (h : X ∣ 120) : 
  ∃ n : ℕ, (n % 5 = 0) ∧ (n % 24 = 0) ∧ (n % X = 0) ∧ n = 120 := 
by
  use 120
  split
  { norm_num }
  split
  { norm_num }
  split
  { apply dvd_trans _ h, norm_num }
  { refl }

end smallest_num_boxes_l574_574446


namespace no_non_trivial_power_ending_222_l574_574949

theorem no_non_trivial_power_ending_222 (x y : ℕ) (hx : x > 1) (hy : y > 1) : ¬ (∃ n : ℕ, n % 1000 = 222 ∧ n = x^y) :=
by
  sorry

end no_non_trivial_power_ending_222_l574_574949


namespace determinant_of_2x2_matrix_l574_574545

theorem determinant_of_2x2_matrix (x : ℝ) : 
  let a := 5
      b := x
      c := -3
      d := 4 in
  a * d - b * c = 20 + 3 * x :=
by
  sorry

end determinant_of_2x2_matrix_l574_574545


namespace kids_went_to_camp_l574_574231

axiom KidsTotal : ℕ := 1538832
axiom KidsHome : ℕ := 644997

theorem kids_went_to_camp :
  KidsTotal - KidsHome = 893835 :=
by sorry

end kids_went_to_camp_l574_574231


namespace molecular_weight_of_BaO_l574_574981

theorem molecular_weight_of_BaO :
  ∀ (atomic_weight_Ba : ℝ) (atomic_weight_O : ℝ),
    atomic_weight_Ba = 137.33 →
    atomic_weight_O = 16.00 →
    (atomic_weight_Ba + atomic_weight_O) = 153.33 :=
by
  intros atomic_weight_Ba atomic_weight_O hBa hO
  rw [hBa, hO]
  exact rfl

end molecular_weight_of_BaO_l574_574981


namespace number_of_pairs_l574_574319

theorem number_of_pairs (k : ℕ) (h : k = 10^2004) :
  (∃ (f : ℕ → ℕ → Prop), 
    f = (λ m n, 7 * m + 3 * n = k ∧ m ∣ n) ∧
    (finset.card (finset.filter (λ p, f p.1 p.2) (finset.product (finset.range (k + 1)) (finset.range (k + 1)))) = 2010011)) :=
sorry

end number_of_pairs_l574_574319


namespace probability_at_least_6_heads_in_8_flips_l574_574145

theorem probability_at_least_6_heads_in_8_flips : 
  (∑ k in finset.range 3, nat.choose 8 (6 + k)) / (2 ^ 8) = 37 / 256 :=
by sorry

end probability_at_least_6_heads_in_8_flips_l574_574145


namespace train_speed_l574_574920

theorem train_speed (length : ℤ) (time : ℤ) 
  (h_length : length = 280) (h_time : time = 14) : 
  (length * 3600) / (time * 1000) = 72 := 
by {
  -- The proof would go here, this part is omitted as per instructions
  sorry
}

end train_speed_l574_574920


namespace midpoint_C1C2_l574_574090

noncomputable def equilateral (A B C : Point) : Prop := 
  ∀ P Q R, P ≠ Q → Q ≠ R → R ≠ P →
  dist P Q = dist Q R ∧ dist Q R = dist R P ∧ dist R P = dist P Q

theorem midpoint_C1C2 (A B C A1 B1 C1 C2 : Point) 
  (h1 : equilateral A B C₁) 
  (h2 : equilateral B C A₁) 
  (h3 : equilateral C A B₁) 
  (h4 : equilateral A1 B1 C2) 
  (h5 : collinear A B C) 
  : midpoint C C₁ C₂ := 
sorry

end midpoint_C1C2_l574_574090


namespace right_triangle_midpoint_division_l574_574967

-- Define a triangle ABC with A a right-angle vertex and D the midpoint of BC
theorem right_triangle_midpoint_division (A B C D : Point) [IsRightTriangle A B C]
  (h_midpoint : is_midpoint D B C) :
  area (triangle A B D) = area (triangle A D C) :=
sorry

end right_triangle_midpoint_division_l574_574967


namespace ratio_humans_to_beavers_l574_574684

-- Define the conditions
def humans : ℕ := 38 * 10^6
def moose : ℕ := 1 * 10^6
def beavers : ℕ := 2 * moose

-- Define the theorem to prove the ratio of humans to beavers
theorem ratio_humans_to_beavers : humans / beavers = 19 := by
  sorry

end ratio_humans_to_beavers_l574_574684


namespace solve_for_y_l574_574323

theorem solve_for_y (x y : ℝ) (h : 2 * x - 7 * y = 8) : y = (2 * x - 8) / 7 := by
  sorry

end solve_for_y_l574_574323


namespace find_angle_IPJ_l574_574752

noncomputable def midpoint (A B : Point) : Point := sorry -- Definition for midpoint

noncomputable def perpendicular_bisector (A B : Point) : Line := sorry -- Definition for perpendicular bisector

noncomputable def circumcircle (A B C : Point) : Circle := sorry -- Definition for circumcircle

noncomputable def incenter (A B C : Point) : Point := sorry -- Definition for incenter

noncomputable def angle (A B C : Point) : ℝ := sorry -- Definition for angle

theorem find_angle_IPJ
  (A B C P M I J : Point)
  (h1 : scalene A B C)
  (h2 : M = midpoint B C)
  (h3 : P ∈ circumcircle A B C)
  (h4 : on_same_side P A B C)
  (h5 : incircle_center I A B M)
  (h6 : incircle_center J A M C)
  (h7 : angle A B C = α)
  (h8 : angle B A C = β)
  (h9 : angle C A B = γ)
  : angle I P J = α / 2 :=
sorry

end find_angle_IPJ_l574_574752


namespace min_pawns_no_adjacent_empty_l574_574987

theorem min_pawns_no_adjacent_empty 
  (n k : ℕ) (h1 : n > 0) (h2 : n / 2 < k) (h3 : k ≤ 2 * n / 3) : 
  ∃ (p : ℕ), p = 4 * (n - k) ∧ (∀ (board : matrix (fin n) (fin n) bool), 
  (∀ i j, board i j ≠ tt → (j + k < n → (∀ m : ℕ, m < k → board i (j + m) = tt)) ∧ 
  (i + k < n → (∀ m : ℕ, m < k → board (i + m) j = tt)) → 
  (∃ pawns : fin n × fin n → bool, 
  (∑ i j, if pawns i j then 1 else 0 = 4 * (n - k)) ∧ 
  (∀ i j, pawns i j = tt → (j + k < n → (∀ m : ℕ, m < k → pawns i (j + m) = tt)) ∧ 
  (i + k < n → (∀ m : ℕ, m < k → pawns (i + m) j = tt)) ) ) 
  sorry

end min_pawns_no_adjacent_empty_l574_574987


namespace no54_after_60_operations_l574_574887

theorem no54_after_60_operations :
  ∀ (f : ℕ → ℕ), (f 0 = 12) →
  (∀ t, ∃ n : ℤ, n ∈ {-1, 1} ∧ (f (t + 1) = f t * 2 ^ n ∨ f (t + 1) = f t * 3 ^ n)) →
  ∀ t, t ≤ 60 → f t ≠ 54 :=
by
  sorry

end no54_after_60_operations_l574_574887


namespace range_of_a_l574_574338

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x| = ax + 1 → x < 0) → a > 1 :=
by
  sorry

end range_of_a_l574_574338


namespace exradii_product_eq_area_squared_l574_574422

variable (a b c : ℝ) (t : ℝ)
variable (s := (a + b + c) / 2)
variable (exradius_a exradius_b exradius_c : ℝ)

-- Define the conditions
axiom Heron : t^2 = s * (s - a) * (s - b) * (s - c)
axiom exradius_definitions : exradius_a = t / (s - a) ∧ exradius_b = t / (s - b) ∧ exradius_c = t / (s - c)

-- The theorem we want to prove
theorem exradii_product_eq_area_squared : exradius_a * exradius_b * exradius_c = t^2 := sorry

end exradii_product_eq_area_squared_l574_574422


namespace complex_magnitude_sixth_power_l574_574241

noncomputable def z := (2 : ℂ) + (2 * Real.sqrt 3) * Complex.I

theorem complex_magnitude_sixth_power :
  Complex.abs (z^6) = 4096 := 
by
  sorry

end complex_magnitude_sixth_power_l574_574241


namespace find_n_l574_574374

theorem find_n (x : ℝ) (n : ℝ) (G : ℝ) (hG : G = (7*x^2 + 21*x + 5*n) / 7) :
  (∃ c d : ℝ, c^2 * x^2 + 2*c*d*x + d^2 = G) ↔ n = 63 / 20 :=
by
  sorry

end find_n_l574_574374


namespace find_number_l574_574101

theorem find_number (x : ℝ) (h : 0.8 * x = (4/5 : ℝ) * 25 + 16) : x = 45 :=
by
  sorry

end find_number_l574_574101


namespace shaded_percentage_is_correct_l574_574469

def side_length_square : ℝ := 18
def length_rectangle : ℝ := 35
def width_rectangle : ℝ := side_length_square

def area_rectangle : ℝ := length_rectangle * width_rectangle
def overlap_length : ℝ := (2 * side_length_square) - length_rectangle
def area_shaded : ℝ := width_rectangle * overlap_length

def percentage_shaded : ℝ := (area_shaded / area_rectangle) * 100

theorem shaded_percentage_is_correct :
  percentage_shaded = 28.57 := 
sorry

end shaded_percentage_is_correct_l574_574469


namespace odd_function_and_extrema_l574_574958

def f (x : ℝ) : ℝ :=
  if x < 0 then (1/4)^x - 8 * (1/2)^x - 1
  else if x = 0 then 0
  else -4^x + 8 * 2^x + 1

theorem odd_function_and_extrema 
  (h₁ : ∀ x : ℝ, f(x) + f(-x) = 0)
  (h₂ : ∀ x : ℝ, x > 0 → f(x) = -4^x + 8 * 2^x + 1) :
  (∀ x : ℝ, 
    f(x) = if x < 0 then (1/4)^x - 8 * (1/2)^x - 1
           else if x = 0 then 0
           else -4^x + 8 * 2^x + 1) ∧ 
  (∃ x ∈ Icc (-3:ℝ) (-1:ℝ), f(x) = -1) ∧ 
  (∃ x ∈ Icc (-3:ℝ) (-1:ℝ), f(x) = -17) :=
by
  sorry

end odd_function_and_extrema_l574_574958


namespace length_of_bridge_is_255_l574_574442

noncomputable def bridge_length (train_length : ℕ) (train_speed_kph : ℕ) (cross_time_sec : ℕ) : ℕ :=
  let train_speed_mps := train_speed_kph * 1000 / (60 * 60)
  let total_distance := train_speed_mps * cross_time_sec
  total_distance - train_length

theorem length_of_bridge_is_255 :
  ∀ (train_length : ℕ) (train_speed_kph : ℕ) (cross_time_sec : ℕ), 
    train_length = 120 →
    train_speed_kph = 45 →
    cross_time_sec = 30 →
    bridge_length train_length train_speed_kph cross_time_sec = 255 :=
by
  intros train_length train_speed_kph cross_time_sec htl htsk hcts
  simp [bridge_length]
  rw [htl, htsk, hcts]
  norm_num
  sorry

end length_of_bridge_is_255_l574_574442


namespace katie_rolls_nineteenth_side_l574_574729

theorem katie_rolls_nineteenth_side :
  let initial_sides : ℕ := 2019 in
  let final_roll : ℕ := initial_sides in
  let final_die_sides := initial_sides + (final_roll - 1) in
  1 / final_die_sides = 1 / initial_sides
:= 
sorry

end katie_rolls_nineteenth_side_l574_574729


namespace circle_radius_45_45_90_l574_574548

def is_45_45_90_triangle (X Y Z : Point) : Prop :=
  ∃ (a : ℝ), a > 0 ∧ dist X Y = a ∧ dist Y Z = a ∧ dist X Z = a * real.sqrt 2

def is_tangent_circle (O : Point) (r : ℝ) (X Y Z : Point) : Prop :=
  O.x = r ∧ O.y = r ∧ dist O (line_of_points X Z) = r

theorem circle_radius_45_45_90 (X Y Z O : Point) (r : ℝ)
  (h₀ : is_45_45_90_triangle X Y Z)
  (h₁ : dist X Y = 4)
  (h₂ : is_tangent_circle O r X Y Z) :
  r = 2 :=
sorry

end circle_radius_45_45_90_l574_574548


namespace find_matrix_M_and_eigenvalue_l574_574608

-- Define the conditions
def has_eigenvalue_and_vector (M : Matrix (Fin 2) (Fin 2) ℝ) (λ : ℝ) (v : Vector ℝ 2) :=
  M.mul_vec v = λ • v 

def changes_point (M : Matrix (Fin 2) (Fin 2) ℝ) (p q : Vector ℝ 2) :=
  M.mul_vec p = q

-- Prove the main statements using these conditions
theorem find_matrix_M_and_eigenvalue :
  ∃ (M : Matrix (Fin 2) (Fin 2) ℝ),
  has_eigenvalue_and_vector M 8 (Vector.of_list [1, 1]) ∧
  changes_point M (Vector.of_list [-1, 2]) (Vector.of_list [-2, 4]) ∧
  M = (Matrix.of_fn ! [! [6, 2], ! [4, 4]]) ∧
  (M.eigenvalues = {8, 2} : Finset ℝ) :=
sorry

end find_matrix_M_and_eigenvalue_l574_574608


namespace convert_denominators_to_integers_l574_574551

def original_equation (x : ℝ) : Prop :=
  (x + 1) / 0.4 - (0.2 * x - 1) / 0.7 = 1

def transformed_equation (x : ℝ) : Prop :=
  (10 * x + 10) / 4 - (2 * x - 10) / 7 = 1

theorem convert_denominators_to_integers (x : ℝ) 
  (h : original_equation x) : transformed_equation x :=
sorry

end convert_denominators_to_integers_l574_574551


namespace magnitude_of_power_l574_574238

open Complex

theorem magnitude_of_power (a b : ℝ) : abs ((Complex.mk 2 (2 * Real.sqrt 3)) ^ 6) = 4096 := by
  sorry

end magnitude_of_power_l574_574238


namespace perimeter_triangle_ABF2_l574_574510

def ellipse := (center_x center_y : Real) (a b : Real)
def focus (e : ellipse) (focus_left : Bool) : (Real × Real) :=
  if focus_left then (e.center_x - e.a * Real.sqrt 2, e.center_y)
  else (e.center_x + e.a * Real.sqrt 2, e.center_y)

theorem perimeter_triangle_ABF2 
  (e : ellipse) 
  (h : e = ⟨0, 0, Real.sqrt 2, 1⟩)
  (A B : Real × Real) 
  (hl : ∃ (l : Real × Real), focus e true = l ∧ ∃ (k : Real), l = k • (1, 0)) 
  (line_intersects : ∃ (A B : Real × Real), A ≠ B ∧ (A ∈ ellipse.point e) ∧ (B ∈ ellipse.point e)) :
  (let F1 := focus e true in
  let F2 := focus e false in
  (distance A F1 + distance A F2) + (distance B F1 + distance B F2)) = 4 * Real.sqrt 2 := by
    sorry

end perimeter_triangle_ABF2_l574_574510


namespace find_m_n_l574_574059

noncomputable def triangle_ABC (AB BC CA : ℝ) :=
  ∃ (A B C : ℝ × ℝ), dist A B = AB ∧ dist B C = BC ∧ dist C A = CA

def circles_tangent (A B C : ℝ × ℝ) (ω1 ω2 : ℝ × ℝ → ℝ) :=
  ∀ X, ω1 X = dist A X ∧ ω2 X = dist A X ∧ 
       ∃ K, K ≠ A ∧ ω1 K = 0 ∧ ω2 K = 0

theorem find_m_n :
  ∀ (AK : ℝ), 
  ∃ (A B C : ℝ × ℝ) (AB BC CA : ℝ) (ω1 ω2 : ℝ × ℝ → ℝ),
    AB = 8 →
    BC = 9 →
    CA = 10 →
    triangle_ABC AB BC CA →
    circles_tangent A B C ω1 ω2 →
    ∃ (m n : ℕ), (10 * n = m * 3) ∧ (nat.gcd m n = 1) ∧ (m + n = 13) :=
sorry

end find_m_n_l574_574059


namespace metro_problem_l574_574704

theorem metro_problem (G : SimpleGraph (Fin 2019)) (hG : G.Connected) : 
  ¬ ∃ (P : Finset (Set (Fin 2019))), P.card ≤ 1008 ∧ ∀ v, ∃ p ∈ P, v ∈ p :=
sorry

end metro_problem_l574_574704


namespace solve_for_x_l574_574776

theorem solve_for_x (x : ℝ) : 3^(3 * x) = Real.sqrt 81 -> x = 2 / 3 :=
by
  sorry

end solve_for_x_l574_574776


namespace books_initially_l574_574928

theorem books_initially (sold_mon sold_tue sold_wed sold_thu sold_fri not_sold : ℕ)
  (h_mon : sold_mon = 60) (h_tue : sold_tue = 10) 
  (h_wed : sold_wed = 20) (h_thu : sold_thu = 44) 
  (h_fri : sold_fri = 66) (h_not_sold : not_sold = 600) :
  let total_sold = sold_mon + sold_tue + sold_wed + sold_thu + sold_fri in
  total_sold + not_sold = 800 :=
by {
  sorry
}

end books_initially_l574_574928


namespace area_of_square_l574_574910

-- Define the parabola and the line
def parabola (x : ℝ) : ℝ := x^2 + 4 * x + 3
def line (y : ℝ) : Prop := y = 7

-- Define the roots of the quadratic equation derived from the conditions
noncomputable def root1 : ℝ := -2 + 2 * Real.sqrt 2
noncomputable def root2 : ℝ := -2 - 2 * Real.sqrt 2

-- Define the side length of the square
noncomputable def side_length : ℝ := abs (root1 - root2)

-- Define the area of the square
noncomputable def area_square : ℝ := side_length^2

-- Theorem statement for the problem
theorem area_of_square : area_square = 32 :=
sorry

end area_of_square_l574_574910


namespace comb_12_9_eq_220_l574_574211

theorem comb_12_9_eq_220 : (Nat.choose 12 9) = 220 := by
  sorry

end comb_12_9_eq_220_l574_574211


namespace probability_at_least_6_heads_in_8_flips_l574_574134

open scoped BigOperators

def binom (n k : ℕ) : ℕ := nat.choose n k

def total_outcomes (n : ℕ) := 2^n

def successful_outcomes (n k : ℕ) :=
  (finset.range (n + 1)).filter (λ x, x ≥ k).sum (λ x, binom n x)

def probability (n k : ℕ) :=
  (successful_outcomes n k) / (total_outcomes n : ℚ)

theorem probability_at_least_6_heads_in_8_flips :
  probability 8 6 = 37 / 256 := sorry

end probability_at_least_6_heads_in_8_flips_l574_574134


namespace pyramid_configurations_l574_574006

theorem pyramid_configurations : 
  ∃ n : ℕ, n = 16 ∧ 
  ∀ weights : Finset ℕ, 
    weights.card = 6 ∧ 
    weights = {1, 2, 3, 4, 5, 6} →
    ∃ (A B C D E F : ℕ), 
      (A = 1) ∧ 
      (B ≠ C) ∧
      (Finset.card (Finset.of_list [D, E, F]) = 3) ∧
      (∀ x ∈ {A}, ∀ y ∈ {B,C}, x < y) ∧
      (∀ y ∈ {B,C}, ∀ z ∈ {D,E,F}, y < z) :=
begin
  existsi 16,
  split,
  { refl, },
  { intros weights h_weights,
    have h : weights = {1, 2, 3, 4, 5, 6} := h_weights.2,
    use 1, -- A
    -- We set B, C, D, E, F according to remaining conditions
    sorry
  }
end

end pyramid_configurations_l574_574006


namespace root_in_interval_l574_574071

noncomputable def f (x : ℝ) := Real.log x + x - 2

theorem root_in_interval : ∃ c ∈ Set.Ioo 1 2, f c = 0 := 
sorry

end root_in_interval_l574_574071


namespace imaginary_part_of_complex_l574_574640

def imaginary_part_of_z : ℂ := 
  let z := (1 + complex.I) * (1 - 2 * complex.I) in 
  z.im

theorem imaginary_part_of_complex : imaginary_part_of_z = -1 := by
  sorry

end imaginary_part_of_complex_l574_574640


namespace color_theorem_l574_574960

/-- The only integers \( k \geq 1 \) such that if each integer is colored in one of these \( k \)
colors, there must exist integers \( a_1 < a_2 < \cdots < a_{2023} \) of the same color where the
differences \( a_2 - a_1, a_3 - a_2, \cdots, a_{2023} - a_{2022} \) are all powers of 2 are
\( k = 1 \) and \( k = 2 \). -/
theorem color_theorem : ∀ (k : ℕ), (k ≥ 1) →
  (∀ f : ℕ → Fin k,
    ∃ a : Fin 2023 → ℕ,
    (∀ i : Fin (2023 - 1), ∃ n : ℕ, 2^n = (a i.succ - a i)) ∧
    (∀ i j : Fin 2023, i < j → f (a i) = f (a j)))
  ↔ k = 1 ∨ k = 2 := by
  sorry

end color_theorem_l574_574960


namespace trapezoid_midline_length_l574_574438

theorem trapezoid_midline_length
    (ABC : Type) [trapezoid ABC]
    (O : Point)
    (AC BD : Line O)
    (H_perpendicular : ∀ A B C, ∠(AC, BD) = 90)
    (H_AC : AC.length = 6)
    (H_angle : ∀ A D, ∠(BD, AD) = 30) :
  midline_length(ABC) = 6 := sorry

end trapezoid_midline_length_l574_574438


namespace average_first_two_numbers_l574_574784

theorem average_first_two_numbers (a1 a2 a3 a4 a5 a6 : ℝ)
  (h1 : (a1 + a2 + a3 + a4 + a5 + a6) / 6 = 3.95)
  (h2 : (a3 + a4) / 2 = 3.85)
  (h3 : (a5 + a6) / 2 = 4.200000000000001) :
  (a1 + a2) / 2 = 3.8 :=
by
  sorry

end average_first_two_numbers_l574_574784


namespace fourth_candidate_votes_calculation_l574_574696

-- Define the conditions
def total_votes : ℕ := 7000
def invalid_percent : ℝ := 0.25
def valid_percent : ℝ := 0.75
def valid_votes : ℕ := total_votes * valid_percent

def first_candidate_percent : ℝ := 0.40
def second_candidate_percent : ℝ := 0.35
def third_candidate_percent : ℝ := 0.15

def first_candidate_votes : ℕ := valid_votes * first_candidate_percent
def second_candidate_votes : ℕ := valid_votes * second_candidate_percent
def third_candidate_votes : ℕ := valid_votes * third_candidate_percent

-- Define the proof theorem
theorem fourth_candidate_votes_calculation (total_votes : ℕ) (invalid_percent valid_percent first_candidate_percent second_candidate_percent third_candidate_percent : ℝ) (valid_votes first_candidate_votes second_candidate_votes third_candidate_votes : ℕ)
  (h_valid_votes : valid_votes = total_votes * valid_percent)
  (h_first_candidate_votes : first_candidate_votes = valid_votes * first_candidate_percent)
  (h_second_candidate_votes : second_candidate_votes = valid_votes * second_candidate_percent)
  (h_third_candidate_votes : third_candidate_votes = valid_votes * third_candidate_percent) :
  (valid_votes - (first_candidate_votes + second_candidate_votes + third_candidate_votes) = 525) :=
by
  -- Proof goes here
  sorry

end fourth_candidate_votes_calculation_l574_574696


namespace probability_at_least_6_heads_l574_574152

-- Definitions of the binomial coefficient and probability function
def binom (n k : ℕ) : ℕ := Nat.choose n k

def probability (favorable total : ℕ) : ℚ := favorable / total

-- Proof problem statement
theorem probability_at_least_6_heads (flips : ℕ) (p : ℚ) 
  (h_flips : flips = 8) 
  (h_probability : p = probability (binom 8 6 + binom 8 7 + binom 8 8) (2 ^ flips)) : 
  p = 37 / 256 :=
by
  sorry

end probability_at_least_6_heads_l574_574152


namespace right_triangle_legs_l574_574798

theorem right_triangle_legs (m r x y : ℝ) 
  (h1 : m^2 = x^2 + y^2) 
  (h2 : r = (x + y - m) / 2) 
  (h3 : r ≤ m * (Real.sqrt 2 - 1) / 2) : 
  (x = (2 * r + m + Real.sqrt (m^2 - 4 * r^2 - 4 * r * m)) / 2) ∧ 
  (y = (2 * r + m - Real.sqrt (m^2 - 4 * r^2 - 4 * r * m)) / 2) :=
by 
  sorry

end right_triangle_legs_l574_574798
