import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Functional.Iterate
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.Calculus.Basic
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Sort
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Digits
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Log
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Logic.Basic
import Mathlib.NumberTheory.ArithmeticFunction
import Mathlib.NumberTheory.Divisors
import Mathlib.NumberTheory.EuclideanDomain
import Mathlib.Probability.Statistics
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith

namespace min_positive_period_and_intervals_mono_increase_cos_2x0_value_l147_147748

-- Define the function f(x)
def f (x : ℝ) : ℝ := 
  sin (5 * Real.pi / 6 - 2 * x) - 
  2 * sin (x - Real.pi / 4) * cos (x + 3 * Real.pi / 4)

-- State the first part of the problem
theorem min_positive_period_and_intervals_mono_increase :
  (∃ T > 0, ∀ x, f(x + T) = f x) ∧ 
  (∀ k : ℤ, 
    ∃ a b : ℝ, 
    a = k * Real.pi - Real.pi / 6 ∧ 
    b = k * Real.pi + Real.pi / 3 ∧ 
    ∀ x, (a ≤ x ∧ x ≤ b) → f'(x) ≥ 0) := sorry

-- State the second part of the problem
theorem cos_2x0_value (x0 : ℝ) (h₀ : x0 ∈ Icc (Real.pi / 3) (7 * Real.pi / 12)) 
  (hx0 : f x0 = 1 / 3) : 
  cos (2 * x0) = - (2 * Real.sqrt 6 + 1) / 6 := sorry

end min_positive_period_and_intervals_mono_increase_cos_2x0_value_l147_147748


namespace eleventh_grade_sample_size_l147_147412

-- Definitions based on the conditions
def ratio : ℕ × ℕ × ℕ := (3, 3, 4)
def sample_size : ℕ := 50
def proportion_eleventh := (ratio.2 : ℚ) / (ratio.1 + ratio.2 + ratio.3) -- 3 / (3 + 3 + 4)

-- Theorem statement based on the mathematically equivalent proof problem
theorem eleventh_grade_sample_size : 
  proportion_eleventh * sample_size = 15 := sorry

end eleventh_grade_sample_size_l147_147412


namespace smallest_5_digit_palindrome_in_base_3_is_3_digit_palindrome_in_other_base_l147_147200

def is_palindrome (n : ℕ) (b : ℕ) : Prop :=
  let digits := nat.digits b n in digits = digits.reverse

theorem smallest_5_digit_palindrome_in_base_3_is_3_digit_palindrome_in_other_base :
  ∃ (n b : ℕ), is_palindrome n 3 ∧ digits 3 n = [1, 0, 0, 0, 1] ∧ is_palindrome n b ∧ b ≠ 3 ∧ 
  (digits b n).length = 3 :=
by
  exists 82 -- decimal representation of 10001_3
  exists 5 -- the base 5
  split
  { unfold is_palindrome
    rw digits,
    refl, } -- Proof that 82 is a palindrome in base 3
  split
  { rw nat.digits_eq_of_dvd,
    exact nat.digits_fst_eq_one_of_5_digits_palindrome (show 82 % 3 = _, by norm_num) } -- showing [1,0,0,0,1] = 10001_3
  split
  { unfold is_palindrome,
    rw digits,
    refl, } -- Proof that 322_5
  split
  { exact ne_of_gt nat.succ_le_succ (nat.succ_le (show 82 > 5, by norm_num)) }
  rw digits,
  exact len_pos.nth_drop _ -- length of 3 in base 5


end smallest_5_digit_palindrome_in_base_3_is_3_digit_palindrome_in_other_base_l147_147200


namespace real_root_probability_approx_l147_147984

theorem real_root_probability_approx (b : ℝ)
  (h1 : b ∈ Set.Icc (-15 : ℝ) (20 : ℝ)) :
  let p := Polynomial.X^4 + C b * Polynomial.X^3 + C (b + 3) * Polynomial.X^2 + C (-3 * b + 4) * Polynomial.X - C 6 in
  let d := b - 1 in
  [p.eval (1 : ℝ) = 0, p.eval (-2 : ℝ) = 0, d^2 - 12 ≥ 0] →
  ∃ (prob : ℝ), prob ≈ 0.8596 :=
sorry

end real_root_probability_approx_l147_147984


namespace probability_product_lt_50_l147_147858

-- Definitions based on the problem conditions.
def numbersPaco : Finset ℕ := Finset.range 5  -- Represents {1, 2, 3, 4, 5}
def numbersManu : Finset ℕ := Finset.range 15 -- Represents {1, 2, 3, ..., 15}

def countFavorableOutcomes : ℕ :=
  let outcomes := [(p, m) | p ∈ numbersPaco, m ∈ numbersManu, p * m < 50]
  outcomes.length

def totalOutcomes : ℕ :=
  numbersPaco.card * numbersManu.card

def probabilityProductLessThan50 : ℚ :=
  (countFavorableOutcomes : ℚ) / (totalOutcomes : ℚ)

theorem probability_product_lt_50 :
  probabilityProductLessThan50 = 22 / 25 :=
by
  sorry

end probability_product_lt_50_l147_147858


namespace no_triangle_lines_l147_147382

theorem no_triangle_lines (m : ℝ) : 
  let L1 := (4 * x + y = 4)
  let L2 := (m * x + y = 0)
  let L3 := (2 * x - 3 * m * y = 4)
  ∃ m : ℝ, ((m = 4) ∨ (m = -1/6) ∨ (m = -1) ∨ (m = 2/3)) → 
  (¬ triangle (L1, L2, L3)) := 
by 
  sorry

end no_triangle_lines_l147_147382


namespace impossible_to_place_50_pieces_on_torus_grid_l147_147621

theorem impossible_to_place_50_pieces_on_torus_grid :
  ¬ (∃ (a b c x y z : ℕ),
    a + b + c = 50 ∧
    2 * a ≤ x ∧ x ≤ 2 * b ∧
    2 * b ≤ y ∧ y ≤ 2 * c ∧
    2 * c ≤ z ∧ z ≤ 2 * a) :=
by
  sorry

end impossible_to_place_50_pieces_on_torus_grid_l147_147621


namespace largest_multiple_of_15_less_than_500_l147_147097

theorem largest_multiple_of_15_less_than_500 : ∃ (x : ℕ), (x < 500) ∧ (x % 15 = 0) ∧ ∀ (y : ℕ), (y < 500) ∧ (y % 15 = 0) → y ≤ x :=
begin
  use 495,
  split,
  { norm_num },
  split,
  { norm_num },
  intros y hy1 hy2,
  sorry,
end

end largest_multiple_of_15_less_than_500_l147_147097


namespace solve_inequality_l147_147492

theorem solve_inequality (x : ℝ) :
  x * Real.log (x^2 + x + 1) / Real.log 10 < 0 ↔ x < -1 :=
sorry

end solve_inequality_l147_147492


namespace largest_multiple_of_15_less_than_500_l147_147108

theorem largest_multiple_of_15_less_than_500 :
∀ x : ℕ, (∃ k : ℕ, x = 15 * k ∧ 0 < x ∧ x < 500) → x ≤ 495 :=
begin
  sorry
end

end largest_multiple_of_15_less_than_500_l147_147108


namespace intersection_primes_evens_l147_147298

open Set

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def evens : Set ℕ := {n | n % 2 = 0}
def primes : Set ℕ := {n | is_prime n}

theorem intersection_primes_evens :
  primes ∩ evens = {2} :=
by sorry

end intersection_primes_evens_l147_147298


namespace sum_of_roots_of_quadratic_eq_l147_147931

theorem sum_of_roots_of_quadratic_eq : 
  (∑ x in ({x : ℝ | x * x = 16 * x - 5}.to_finset), x) = 16 :=
sorry

end sum_of_roots_of_quadratic_eq_l147_147931


namespace lowest_painting_cost_l147_147531

variable (x y z a b c : ℝ)
variable (h1 : x < y)
variable (h2 : y < z)
variable (h3 : a < b)
variable (h4 : b < c)

theorem lowest_painting_cost : az + by + cx < ay + bx + cz :=
  sorry

end lowest_painting_cost_l147_147531


namespace largest_integer_satisfying_l147_147626

theorem largest_integer_satisfying (x : ℤ) : 
  (∃ x, (2/7 : ℝ) < (x / 6 : ℝ) ∧ (x / 6 : ℝ) < 3/4) → x = 4 := 
by 
  sorry

end largest_integer_satisfying_l147_147626


namespace triangle_tangent_circle_angle_l147_147539

/-- 
  Given triangle PAB is formed by three tangents to circle O,
  and ∠ APB = 40 degrees, then angle ∠ AOB = 80 degrees.
-/
theorem triangle_tangent_circle_angle
  (O P A B : Type)
  [InnerProductSpace ℝ P]
  [Circle O] -- assuming some definition for circle
  (h_triangle_tangents : TangentsToCircle O {P, A, B}) -- PAB tangents to O
  (h_angle_APB : ∠ P A B = 40) : ∠ A O B = 80 :=
sorry

end triangle_tangent_circle_angle_l147_147539


namespace eval_expression_l147_147677

noncomputable def ceil_sqrt_16_div_9 : ℕ := ⌈Real.sqrt (16 / 9 : ℚ)⌉
noncomputable def ceil_16_div_9 : ℕ := ⌈(16 / 9 : ℚ)⌉
noncomputable def ceil_square_16_div_9 : ℕ := ⌈(16 / 9 : ℚ)^2⌉

theorem eval_expression : ceil_sqrt_16_div_9 + ceil_16_div_9 + ceil_square_16_div_9 = 8 :=
by
  -- The following sorry is a placeholder, indicating that the proof is skipped.
  sorry

end eval_expression_l147_147677


namespace final_result_invariant_l147_147547

/-- We define the invariant function that calculates the product of a list of numbers. -/
def product (l : List ℤ) : ℤ :=
  l.foldr (· * ·) 1

/-- Define a function that performs the operation on two integers. -/
def operation (x y : ℤ) : ℤ :=
  if x = y then 1 else -1

/-- Define a function that performs one step of the process on a list. -/
def step (l : List ℤ) : List ℤ :=
  match l with
  | []      => []
  | [a]     => [a]
  | a::b::t => operation a b :: t

/-- Define a function that repeatedly applies the step function until one element remains. -/
def reduce (l : List ℤ) : ℤ :=
  match l with
  | [] => 1
  | [a] => a
  | _ =>
    let rec loop (l' : List ℤ) : ℤ :=
      match l' with
      | [a] => a
      | _ => loop (step l')
    loop l

theorem final_result_invariant {n : ℕ} (l : List ℤ) (hl : l.length = n) (hl : ∀ x ∈ l, x = 1 ∨ x = -1) :
  reduce l = if product l = 1 then 1 else -1 :=
  -- Proof omitted
  sorry

end final_result_invariant_l147_147547


namespace cos_C_in_right_triangle_l147_147400

theorem cos_C_in_right_triangle (A B C : Type) [EuclideanGeometry A B C] : 
  ∀ (BC k : ℝ) (tan_C : ℝ), (∠ A B C = 90) → (tan_C = 2) → (cos C = sqrt(5) / 5) :=
by
  -- definition of triangle and right angle
  sorry

end cos_C_in_right_triangle_l147_147400


namespace probability_is_correct_l147_147185

-- Define the coins and their counts
structure Coins :=
  (pennies : ℕ)
  (nickels : ℕ)
  (dimes : ℕ)
  (quarters : ℕ)

-- Given coins setup
def coinBox : Coins := ⟨3, 3, 5, 4⟩

-- Total number of coins
def totalCoins (box : Coins) : ℕ := box.pennies + box.nickels + box.dimes + box.quarters

-- Number of ways to draw 8 coins from 15
def totalOutcomes : ℕ := nat.choose 15 8

-- Number of successful outcomes
def successfulOutcomes : ℕ :=
  let case1 := (nat.choose 4 4) * (nat.choose 11 4)
  let case2 := (nat.choose 4 3) * (nat.choose 5 2) * (nat.choose 6 3)
  let case3 := (nat.choose 4 2) * (nat.choose 5 3) * (nat.choose 6 3)
  case1 + case2 + case3

-- Probability calculation defined
def probability : ℚ := successfulOutcomes / totalOutcomes

-- Statement to prove
theorem probability_is_correct :
  totalCoins coinBox = 15 →
  totalOutcomes = 6435 →
  successfulOutcomes = 2330 →
  probability = 2330 / 6435 := sorry

end probability_is_correct_l147_147185


namespace largest_multiple_of_15_less_than_500_l147_147053

theorem largest_multiple_of_15_less_than_500 : 
  ∃ n : ℕ, n * 15 < 500 ∧ ∀ m : ℕ, m * 15 < 500 → m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l147_147053


namespace line_slope_is_negative_three_halves_l147_147930

theorem line_slope_is_negative_three_halves : 
  ∀ (x y : ℝ), (4 * y = -6 * x + 12) → (∀ x y, y = -((3/2) * x) + 3) :=
begin
  sorry
end

end line_slope_is_negative_three_halves_l147_147930


namespace monic_quadratic_with_root_l147_147709

theorem monic_quadratic_with_root (x : ℂ) (hf : x = 3 - 4 * complex.I) :
    ∃ (P : polynomial ℂ), polynomial.monic P ∧ P.coeff 0 = 25 ∧ P.coeff 1 = -6 ∧ P.coeff 2 = 1 ∧ polynomial.aeval (3 - 4 * complex.I) P = 0 :=
sorry

end monic_quadratic_with_root_l147_147709


namespace simplify_and_evaluate_expr_evaluate_at_zero_l147_147488

theorem simplify_and_evaluate_expr (x : ℝ) (hx1 : x ≠ 1) (hx2 : x ≠ 2) :
  (3 / (x - 1) - x - 1) / ((x^2 - 4 * x + 4) / (x - 1)) = (2 + x) / (2 - x) :=
by
  sorry

theorem evaluate_at_zero :
  (2 + 0 : ℝ) / (2 - 0) = 1 :=
by
  norm_num

end simplify_and_evaluate_expr_evaluate_at_zero_l147_147488


namespace largest_multiple_of_15_less_than_500_is_495_l147_147124

-- Define the necessary conditions
def is_multiple_of_15 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 15 * k

def is_less_than_500 (n : ℕ) : Prop :=
  n < 500

-- Problem statement: Prove that the largest positive multiple of 15 less than 500 is 495
theorem largest_multiple_of_15_less_than_500_is_495 : 
  ∀ n : ℕ, is_multiple_of_15 n → is_less_than_500 n → n ≤ 495 := 
by
  sorry

end largest_multiple_of_15_less_than_500_is_495_l147_147124


namespace log_product_identity_l147_147258

theorem log_product_identity : log 5 2 * log 4 25 = 1 :=
by
  -- sorry is a placeholder for the proof
  sorry

end log_product_identity_l147_147258


namespace ratio_of_larger_to_smaller_l147_147524

variable {x y : ℝ}

-- Condition for x and y being positive and x > y
axiom x_pos : 0 < x
axiom y_pos : 0 < y
axiom x_gt_y : x > y

-- Condition for sum and difference relationship
axiom sum_diff_relation : x + y = 7 * (x - y)

-- Theorem: Ratio of the larger number to the smaller number is 2
theorem ratio_of_larger_to_smaller : x / y = 2 :=
by
  sorry

end ratio_of_larger_to_smaller_l147_147524


namespace original_salary_l147_147895

theorem original_salary (final_salary : ℝ) (h : final_salary = 5225) :
  ∃ (S : ℝ), S = 5225 / (1.10 * 0.95 * 1.08 * 0.97) :=
by
  use 5225 / (1.10 * 0.95 * 1.08 * 0.97)
  rw h
  sorry

end original_salary_l147_147895


namespace find_number_l147_147179

theorem find_number :
  ∃ x : ℝ, 5 * x - (2 * 1.4 / 1.3) = 4 ∧ x ≈ 1.23076923077 :=
by
  -- Supposed conditions from problem statement and mathematical reasoning.
  let c := (2 * 1.4) / 1.3
  have h: c ≈ 2.15384615385 := by sorry -- Approximate calculation step
  let res := (4 + c) / 5
  have res_approx: res ≈ 1.23076923077 := by sorry -- Final step to conclude res is approximately 1.23076923077
  exact ⟨res, by
    split
    · suffices 5 * res - c = 4, by
        rwa [h] -- use approximation
      rw [← h]
      ring
    · exact res_approx⟩

end find_number_l147_147179


namespace size_of_angle_B_length_of_side_b_and_area_l147_147407

-- Given problem conditions
variables (A B C : ℝ) (a b c : ℝ)
variables (h1 : a < b) (h2 : b < c) (h3 : a / Real.sin A = 2 * b / Real.sqrt 3)

-- Prove that B = π / 3
theorem size_of_angle_B : B = Real.pi / 3 := 
sorry

-- Additional conditions for part (2)
variables (h4 : a = 2) (h5 : c = 3) (h6 : Real.cos B = 1 / 2)

-- Prove b = √7 and the area of triangle ABC
theorem length_of_side_b_and_area :
  b = Real.sqrt 7 ∧ 1/2 * a * c * Real.sin B = 3 * Real.sqrt 3 / 2 :=
sorry

end size_of_angle_B_length_of_side_b_and_area_l147_147407


namespace largest_multiple_of_15_less_than_500_l147_147073

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), (n < 500) ∧ (15 ∣ n) ∧ (∀ m : ℕ, (m < 500) ∧ (15 ∣ m) → m ≤ n) ∧ n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l147_147073


namespace find_d_l147_147828

theorem find_d (c d : ℝ) (f g : ℝ → ℝ)
  (hf : ∀ x, f x = 5 * x + c)
  (hg : ∀ x, g x = c * x + 3)
  (hfg : ∀ x, f (g x) = 15 * x + d) :
  d = 18 :=
sorry

end find_d_l147_147828


namespace angle_bfg_is_72_5_degrees_l147_147401

open Real

theorem angle_bfg_is_72_5_degrees
  (A B C F G : Type)
  [P : PlaneGeometry A B C F G]
  (h_angle_A : ∠A = 40)
  (h_angle_B : ∠B = 70)
  (h_AF_eq_FC : AF = FC)
  (h_BF_eq_BG : BF = BG) :
  ∠BFG = 72.5 :=
by
  sorry

end angle_bfg_is_72_5_degrees_l147_147401


namespace kit_price_correct_l147_147188

def filter_price_1 : ℝ := 12.45
def filter_price_2 : ℝ := 14.05
def filter_price_3 : ℝ := 11.50
def percentage_saved : ℝ := 11.03448275862069

noncomputable def price_individual : ℝ := 
  (2 * filter_price_1) + (2 * filter_price_2) + filter_price_3

noncomputable def amount_saved : ℝ := 
  (percentage_saved / 100) * price_individual

noncomputable def kit_price : ℝ := 
  price_individual - amount_saved

theorem kit_price_correct :
  round (kit_price * 100) / 100 = 57.38 := 
by sorry

end kit_price_correct_l147_147188


namespace proof_problem_l147_147675

def sqrt_frac : ℚ := real.sqrt (16 / 9)
def frac : ℚ := 16 / 9
def square_frac : ℚ := frac * frac

def ceil_sqrt_frac : ℤ := ⌈sqrt_frac⌉.to_int
def ceil_frac : ℤ := ⌈frac⌉.to_int
def ceil_square_frac : ℤ := ⌈square_frac⌉.to_int

theorem proof_problem :
  ceil_sqrt_frac + ceil_frac + ceil_square_frac = 8 :=
by
  -- Placeholder for the actual proof.
  sorry

end proof_problem_l147_147675


namespace lucky_N_l147_147314
-- Import the necessary library

-- Define the problem statement in Lean
theorem lucky_N (N : ℕ) : 
  (∀ cubes : Fin N → ℤ, -- given N cubes labeled with integers (representing the colors)
    -- All arrangements are good
    (∀ robot_start_pos : Fin N,
      determine_final_color N cubes robot_start_pos = determine_final_color N cubes 0))
  ↔ ∃ k : ℕ, N = 2^k := 
sorry

-- define necessary functions used in theorem but leave them as sorry for now
def determine_final_color (N : ℕ) (cubes : Fin N → ℤ) (start_pos : Fin N) : ℤ :=
sorry

end lucky_N_l147_147314


namespace clock_angle_at_9_20_l147_147555

theorem clock_angle_at_9_20 :
  let hour_angle := 90.0 -- At 9:00, the hour and minute hand form 90 degrees
  let minute_hand_movement := 20 * 6.0 -- In 20 minutes, the minute hand moves in degrees
  let hour_hand_movement := 20 * 0.5 -- In 20 minutes, the hour hand moves in degrees
  let total_angle := hour_angle + (minute_hand_movement - hour_hand_movement)
in total_angle = 200.0 :=
by {
  let hour_angle := 90.0
  let minute_hand_movement := 20.0 * 6.0
  let hour_hand_movement := 20.0 * 0.5
  let total_angle := hour_angle + (minute_hand_movement - hour_hand_movement)
  show total_angle = 200.0, from sorry
}

end clock_angle_at_9_20_l147_147555


namespace largest_multiple_of_15_less_than_500_is_495_l147_147125

-- Define the necessary conditions
def is_multiple_of_15 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 15 * k

def is_less_than_500 (n : ℕ) : Prop :=
  n < 500

-- Problem statement: Prove that the largest positive multiple of 15 less than 500 is 495
theorem largest_multiple_of_15_less_than_500_is_495 : 
  ∀ n : ℕ, is_multiple_of_15 n → is_less_than_500 n → n ≤ 495 := 
by
  sorry

end largest_multiple_of_15_less_than_500_is_495_l147_147125


namespace minor_premise_is_exponential_l147_147157

-- Define the conditions
def exponential_function_increasing (a : ℝ) (h : a > 1) : Prop := ∀ x y : ℝ, x < y → a^x < a^y 
def function_is_exponential : Prop := ∃ f : ℝ → ℝ, (∀x y : ℝ, f(x) * f(y) = f(x + y)) ∧ (∀x : ℝ, f(x) = 2^x) 

-- Statement to be proved
theorem minor_premise_is_exponential : exponential_function_increasing 2 (by norm_num) → function_is_exponential :=
sorry

end minor_premise_is_exponential_l147_147157


namespace largest_multiple_15_under_500_l147_147090

-- Define the problem statement
theorem largest_multiple_15_under_500 : ∃ (n : ℕ), n < 500 ∧ n % 15 = 0 ∧ (∀ m, m < 500 ∧ m % 15 = 0 → m ≤ n) :=
by {
  use 495,
  split,
  {
    exact lt_of_le_of_ne (le_refl _) (ne_of_lt (nat.lt_succ_self 499))
  },
  split,
  {
    exact nat.mod_eq_zero_of_dvd ⟨33, rfl⟩,
  },
  {
    intros m hm h,
    have hmn : m ≤ 495,
    {
      have div15 := hm/15,
      suffices : div15 < 33+1, from nat.le_of_lt_add_one this,
      exact this, sorry,
    },
    exact hmn,
  },
}

end largest_multiple_15_under_500_l147_147090


namespace sequence_geometric_solution_l147_147519

theorem sequence_geometric_solution {a : ℕ → ℝ} 
  (hlog : ∀ n, real.log2 (a (n + 1)) = 1 + real.log2 (a n))
  (hinit : a 3 = 10) : a 8 = 320 :=
sorry

end sequence_geometric_solution_l147_147519


namespace games_in_tournament_l147_147214

def single_elimination_games (n : Nat) : Nat :=
  n - 1

theorem games_in_tournament : single_elimination_games 24 = 23 := by
  sorry

end games_in_tournament_l147_147214


namespace triangle_height_l147_147500

theorem triangle_height (x y : ℝ) :
  let area := (x^3 * y)^2
  let base := (2 * x * y)^2
  base ≠ 0 →
  (2 * area) / base = x^4 / 2 :=
by
  sorry

end triangle_height_l147_147500


namespace ellipse_equation_l147_147421

-- Define the conditions for the ellipse
variable (a b : ℝ)
variable (h1 : a > b ∧ b ≥ 1)
variable (h2 : (a^2 - b^2) / a^2 = 3/4)
variable (N : ℝ × ℝ)
variable (h3 : ∃ N : ℝ × ℝ, sqrt ((N.1)^2 + (N.2 - 3)^2) = 4)

-- Define the problem and its statement
theorem ellipse_equation : 
  (∃ a b : ℝ, a > b ∧ b ≥ 1 ∧ (a^2 - b^2) / a^2 = 3/4 ∧ ∀ N : ℝ × ℝ, sqrt((N.1)^2 + (N.2 - 3)^2) = 4 → (4b^2 = 4)) →
  ∃ a b : ℝ, a = 2 ∧ b = 1 ∧ (∀ x y, (x^2) / a^2 + (y^2) / b^2 = 1 ↔ (x^2) / 4 + (y^2) = 1) :=
sorry

end ellipse_equation_l147_147421


namespace find_sum_of_x_coordinates_A_l147_147919

-- Define the given points and areas
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (240, 0)
def D : ℝ × ℝ := (720, 450)
def E : ℝ × ℝ := (730, 461)

def area_ABC := 2019
def area_ADE := 8010

-- Distance calculation function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Define the function to compute the x-coordinate sum
noncomputable def compute_x_coordinate_sum (b : ℝ) : ℝ :=
  let a1 := 270 + b - 1456
  let a2 := 270 + b + 1456
  a1 + a2

-- Define the height h from A to BC in triangle ABC
noncomputable def h : ℝ := 2 * area_ABC / distance B C

-- Define the absolute height adjustment
noncomputable def abs_height : ℝ := 1456

-- Create the final statement for the summation of x coordinates
theorem find_sum_of_x_coordinates_A :
  let b1 := h in
  let b2 := -h in
  compute_x_coordinate_sum b1 + compute_x_coordinate_sum b2 = 573.65 :=
begin
  sorry,
end

end find_sum_of_x_coordinates_A_l147_147919


namespace largest_prime_factor_expr_l147_147271

noncomputable def expr : ℤ := 20^3 + 15^4 - 10^5

theorem largest_prime_factor_expr : ∃ p : ℕ, prime p ∧ p = 41 ∧ (∀ q : ℕ, prime q ∧ q ∣ expr → q ≤ 41) :=
by {
  sorry
}

end largest_prime_factor_expr_l147_147271


namespace plates_are_multiple_of_eleven_l147_147624

theorem plates_are_multiple_of_eleven
    (P : ℕ)    -- Number of plates
    (S : ℕ := 33)    -- Number of spoons
    (g : ℕ := 11)    -- Greatest number of groups
    (hS : S % g = 0)    -- Condition: All spoons can be divided into these groups evenly
    (hP : ∀ (k : ℕ), P = k * g) : ∃ x : ℕ, P = 11 * x :=
by
  sorry

end plates_are_multiple_of_eleven_l147_147624


namespace conditional_probability_l147_147535

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_event_A (x y : ℕ) : Prop := is_even (x + y)
def is_event_B (x y : ℕ) : Prop := x + y < 7

noncomputable def P (s : Set (ℕ × ℕ)) : ℝ :=
  (s.to_finset.card : ℝ) / (Finset.univ.card : ℝ)

theorem conditional_probability (x y : ℕ) :
  (P {p ∈ (Finset.univ : Finset (ℕ × ℕ)) | is_event_A p.1 p.2 ∧ is_event_B p.1 p.2}) /
  (P {p ∈ (Finset.univ : Finset (ℕ × ℕ)) | is_event_A p.1 p.2}) = 1 / 2 := by
  sorry

end conditional_probability_l147_147535


namespace set_intersection_set_union_set_complement_l147_147453

open Set

variable (U : Set ℝ) (A B : Set ℝ)
noncomputable def setA : Set ℝ := {x | x^2 - 3*x - 4 ≥ 0}
noncomputable def setB : Set ℝ := {x | x < 5}

theorem set_intersection : (U = univ) -> (A = setA) -> (B = setB) -> A ∩ B = Ico 4 5 := by
  intros
  sorry

theorem set_union : (U = univ) -> (A = setA) -> (B = setB) -> A ∪ B = univ := by
  intros
  sorry

theorem set_complement : (U = univ) -> (A = setA) -> U \ A = Ioo (-1 : ℝ) 4 := by
  intros
  sorry

end set_intersection_set_union_set_complement_l147_147453


namespace max_acute_triangles_l147_147733

theorem max_acute_triangles (P : set (EuclideanSpace.Real 2)) (h : P.card = 4) (h_no_collinear : ∀ {A B C : EuclideanSpace.Real 2}, A ∈ P → B ∈ P → C ∈ P → 
  ¬Collinear ℝ (λ (x : fin 3), if x = 0 then A else if x = 1 then B else C)) : ∃ T : set (set (EuclideanSpace.Real 2)), T.card = 3 ∧ (∀ t ∈ T, is_acute_triangle t) :=
sorry

end max_acute_triangles_l147_147733


namespace sum_of_solutions_eq_320_l147_147914

theorem sum_of_solutions_eq_320 :
  ∃ (S : Finset ℝ), 
  (∀ x ∈ S, 0 < x ∧ x < 180 ∧ (1 + (Real.sin x / Real.sin (4 * x)) = (Real.sin (3 * x) / Real.sin (2 * x)))) 
  ∧ S.sum id = 320 :=
by {
  sorry
}

end sum_of_solutions_eq_320_l147_147914


namespace sequence_sum_l147_147730

open BigOperators

noncomputable def a_n (n : ℕ) : ℕ :=
if n = 1 then 1 else (n^2) - (n-1)^2

noncomputable def b_n (n : ℕ) : ℝ :=
(1 / 2) ^ a_n n

noncomputable def T_n (n : ℕ) : ℝ :=
(2 / 3) * (1 - (1 / 4) ^ n)

theorem sequence_sum {a_n : ℕ → ℕ} {b_n : ℕ → ℝ}
  (h1 : ∀ n, a_n n = if n = 1 then 1 else n^2 - (n - 1)^2)
  (h2 : ∀ n, b_n n = (1 / 2) ^ a_n n) :
  ∀ n, ∑ x in finset.range n, b_n (x + 1) = T_n n :=
begin
  sorry
end

end sequence_sum_l147_147730


namespace jerry_action_figures_l147_147808

theorem jerry_action_figures :
  ∃ (figures_before : ℕ) (books : ℕ) (figures_added : ℕ),
    figures_before = 5 ∧ books = 9 ∧ figures_added = 7 ∧
    (figures_before + figures_added - books = 3) :=
by {
  use 5,
  use 9,
  use 7,
  simp,
  sorry
}

end jerry_action_figures_l147_147808


namespace cevian_concurrent_or_parallel_l147_147446

-- Define points on a triangle and conditions of the problem
variables {A B C A1 B1 C1 : Type*}

-- Define the segments on the sides of the given triangle
variables (AC1 C1B BA1 A1C CB1 B1A : Real)

-- Condition: cevians intersect at a single point or are all parallel
def cevian_ratio (AC1 C1B BA1 A1C CB1 B1A : Real) : Prop :=
  (AC1 / C1B) * (BA1 / A1C) * (CB1 / B1A) = 1

-- Main statement to prove Ceva's Theorem
theorem cevian_concurrent_or_parallel :
  cevian_ratio AC1 C1B BA1 A1C CB1 B1A → 
  (concurrent_or_parallel AA1 BB1 CC1) := sorry

end cevian_concurrent_or_parallel_l147_147446


namespace inequality_holds_if_and_only_if_a_gt_3_l147_147627

theorem inequality_holds_if_and_only_if_a_gt_3 (a : ℝ) :
  (∀ θ ∈ Icc (0:ℝ) (Real.pi/2), 
    sin (2 * θ)
    - (2 * Real.sqrt 2 + Real.sqrt 2 * a) * sin (θ + Real.pi / 4)
    - (2 * Real.sqrt 2) / cos (θ - Real.pi / 4)
    > -3 - 2 * a)
    ↔ (a > 3) :=
sorry

end inequality_holds_if_and_only_if_a_gt_3_l147_147627


namespace child_in_central_111_l147_147417

theorem child_in_central_111
  (families : Fin 111 → (ℕ × ℕ × ℕ)) -- indices representing father, mother, and child
  (row : Fin 333 → ℕ) -- arrangement of 333 people
  (h : ∀ (i : Fin 111), let (f, m, c) := families i in (row f < row c ∧ row c < row m) ∨ (row m < row c ∧ row c < row f)) :
  ∃ (i : Fin 333), 111 ≤ i.val ∧ i.val < 222 ∧ (∃ j : Fin 111, row i = (families j).2.2) := -- 2.2 for the child
sorry

end child_in_central_111_l147_147417


namespace rationalize_expression_l147_147386

theorem rationalize_expression :
  let expr := (Real.sqrt 6 + 1 / Real.sqrt 6 + Real.sqrt 8 + 1 / Real.sqrt 8) in
  ∃ (a b c : ℕ), c > 0 ∧
    c * expr = ↑a * Real.sqrt 6 + ↑b * Real.sqrt 8 ∧
    a + b + c = 106 :=
by
  sorry

end rationalize_expression_l147_147386


namespace roots_sum_l147_147444

theorem roots_sum :
  ∀ (u v w : ℂ), (u, v, w).pairwise (≠) → (u * v * w = 6) →
  (u + v + w = 6) → (u * v + v * w + w * u = 11) →
  (∑ s in (@finset.univ (sym3 ℂ) _) (λ t, t.1)) = (49 / 6) := by
sorry

end roots_sum_l147_147444


namespace geometric_sequence_properties_l147_147787

-- Define the first term and common ratio
def first_term : ℕ := 12
def common_ratio : ℚ := 1/2

-- Define the formula for the n-th term of the geometric sequence
def nth_term (a : ℕ) (r : ℚ) (n : ℕ) := a * r^(n-1)

-- The 8th term in the sequence
def term_8 := nth_term first_term common_ratio 8

-- Half of the 8th term
def half_term_8 := (1/2) * term_8

-- Prove that the 8th term is 3/32 and half of the 8th term is 3/64
theorem geometric_sequence_properties : 
  (term_8 = (3/32)) ∧ (half_term_8 = (3/64)) := 
by 
  sorry

end geometric_sequence_properties_l147_147787


namespace factorial_quotient_l147_147932

-- Define the factorial function
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- State the theorem
theorem factorial_quotient : factorial 20 / (factorial 7 * factorial 13) = 77520 := by
  sorry

end factorial_quotient_l147_147932


namespace julian_owes_jenny_l147_147570

-- Define the initial debt and the additional borrowed amount
def initial_debt : ℕ := 20
def additional_borrowed : ℕ := 8

-- Define the total debt
def total_debt : ℕ := initial_debt + additional_borrowed

-- Statement of the problem: Prove that total_debt equals 28
theorem julian_owes_jenny : total_debt = 28 :=
by
  sorry

end julian_owes_jenny_l147_147570


namespace median_of_set_l147_147738

variable (a : ℤ) (c : ℝ)

def conditions : Prop :=
  a ≠ 0 ∧ c > 0 ∧ a * c^3 = Real.log10 c

theorem median_of_set :
  conditions a c → median {0, 1, a, c, 1 / c} = c :=
by
  sorry

end median_of_set_l147_147738


namespace edward_made_in_summer_l147_147254

theorem edward_made_in_summer
  (spring_earnings : ℤ)
  (spent_on_supplies : ℤ)
  (final_amount : ℤ)
  (S : ℤ)
  (h1 : spring_earnings = 2)
  (h2 : spent_on_supplies = 5)
  (h3 : final_amount = 24)
  (h4 : spring_earnings + S - spent_on_supplies = final_amount) :
  S = 27 := 
by
  sorry

end edward_made_in_summer_l147_147254


namespace largest_multiple_of_15_less_than_500_l147_147019

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), n < 500 ∧ 15 ∣ n ∧ ∀ (m : ℕ), m < 500 ∧ 15 ∣ m -> m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l147_147019


namespace brick_surface_area_l147_147187

variable (X Y Z : ℝ)

#check 4 * X + 4 * Y + 2 * Z = 72 → 
       4 * X + 2 * Y + 4 * Z = 96 → 
       2 * X + 4 * Y + 4 * Z = 102 →
       2 * (X + Y + Z) = 54

theorem brick_surface_area (h1 : 4 * X + 4 * Y + 2 * Z = 72)
                           (h2 : 4 * X + 2 * Y + 4 * Z = 96)
                           (h3 : 2 * X + 4 * Y + 4 * Z = 102) :
                           2 * (X + Y + Z) = 54 := by
  sorry

end brick_surface_area_l147_147187


namespace dot_product_proof_l147_147737

noncomputable def dot_product := 
  let e₁ e₂ : ℝ := 1  -- since they are unit vectors, their magnitudes are 1
  let cos_angle := Math.cos(Real.pi / 3)
  let e₁_dot_e₂ := e₁ * e₂ * cos_angle
  
  let a := 2 * e₁ + e₂
  let b := -3 * e₁ + 2 * e₂
  a * b = -7 / 2

theorem dot_product_proof 
  (unit_vecs : ∀ i, norm (e i) = 1)
  (angle_e1_e2 : ∠e 1 2 = Real.pi / 3)
  (a_def : a = 2 * e 1 + e 2)
  (b_def : b = -3 * e 1 + 2 * e 2) : 
  dot_product := 
begin
  sorry
end

end dot_product_proof_l147_147737


namespace largest_multiple_of_15_less_than_500_l147_147002

theorem largest_multiple_of_15_less_than_500 : ∃ n : ℕ, (n > 0 ∧ 15 * n < 500) ∧ (∀ m : ℕ, m > n → 15 * m >= 500) ∧ 15 * n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l147_147002


namespace find_Δ_l147_147741

-- Define the constants and conditions
variables (Δ p : ℕ)
axiom condition1 : Δ + p = 84
axiom condition2 : (Δ + p) + p = 153

-- State the theorem
theorem find_Δ : Δ = 15 :=
by
  sorry

end find_Δ_l147_147741


namespace last_remaining_digit_is_four_l147_147158

-- Define a function that creates the sequence by repeating "12345" 403 times
def make_sequence : String := 
  let repeated_seq := List.repeat "12345" 403
  String.join repeated_seq

-- Define a function that deletes digits at odd positions
def delete_odds (s : String) : String :=
  s.toList.enum.filter (λ (pair : Nat × Char), pair.1 % 2 = 1).map (λ pair, pair.2) |> String.mk

-- Define a function that repeats the deletion until one character remains
noncomputable def last_digit (s : String) : Char :=
  if s.length = 1 then s.get ⟨0, rfl⟩
  else last_digit (delete_odds s)

-- Main theorem: starting from the sequence and repeating the deletion process
theorem last_remaining_digit_is_four : last_digit make_sequence = '4' :=
  sorry

end last_remaining_digit_is_four_l147_147158


namespace quadrilateral_condition_equiv_l147_147506

noncomputable def point := ℝ × ℝ
noncomputable def length (p1 p2 : point) : ℝ := (p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2

variables (A E C D K L B : point)

-- Given conditions
def length_AD_DC_eq_AB_BC (A D C B : point) : Prop := length A D + length D C = length A B + length B C
def length_AK_CK_eq_AL_CL (A K L C : point) : Prop := length A K + length C K = length A L + length C L
def length_BK_DK_eq_BL_DL (B K L D : point) : Prop := length B K + length D K = length B L + length D L

theorem quadrilateral_condition_equiv : 
  (length_AD_DC_eq_AB_BC A D C B) ↔ (length_AK_CK_eq_AL_CL A K L C) ↔ (length_BK_DK_eq_BL_DL B K L D) := 
sorry

end quadrilateral_condition_equiv_l147_147506


namespace boat_breadth_is_two_l147_147966

noncomputable def breadth_of_boat (L h m g ρ : ℝ) : ℝ :=
  let W := m * g
  let V := W / (ρ * g)
  V / (L * h)

theorem boat_breadth_is_two :
  breadth_of_boat 7 0.01 140 9.81 1000 = 2 := 
by
  unfold breadth_of_boat
  simp
  sorry

end boat_breadth_is_two_l147_147966


namespace edge_c_eq_3_or_5_l147_147403

noncomputable def a := 7
noncomputable def b := 8
noncomputable def A := Real.pi / 3

theorem edge_c_eq_3_or_5 (c : ℝ) (h : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A) : c = 3 ∨ c = 5 :=
by
  sorry

end edge_c_eq_3_or_5_l147_147403


namespace parabolas_intersect_at_point_l147_147863

theorem parabolas_intersect_at_point :
  ∀ (p q : ℝ), p + q = 2019 → (1 : ℝ)^2 + (p : ℝ) * 1 + q = 2020 :=
by
  intros p q h
  sorry

end parabolas_intersect_at_point_l147_147863


namespace tan_390_correct_l147_147956

-- We assume basic trigonometric functions and their properties
noncomputable def tan_390_equals_sqrt3_div3 : Prop :=
  Real.tan (390 * Real.pi / 180) = Real.sqrt 3 / 3

theorem tan_390_correct : tan_390_equals_sqrt3_div3 :=
  by
  -- Proof is omitted
  sorry

end tan_390_correct_l147_147956


namespace evaluate_ceiling_sum_l147_147638

theorem evaluate_ceiling_sum :
  (⌈Real.sqrt (16 / 9)⌉ : ℤ) + (⌈(16 / 9: ℝ)⌉ : ℤ) + (⌈(16 / 9: ℝ)^2⌉ : ℤ) = 8 := 
by
  -- Placeholder for proof
  sorry

end evaluate_ceiling_sum_l147_147638


namespace towels_to_wash_l147_147614

theorem towels_to_wash :
  let g1 := 50 in
  let g2 := g1 + 0.2 * g1 in
  let g3 := g2 + 0.25 * g2 in
  let g4 := g3 + (1 / 3) * g3 in
  g1 + g2 + g3 + g4 = 285 :=
by
  sorry

end towels_to_wash_l147_147614


namespace evaluate_expression_l147_147657

theorem evaluate_expression : 
  (⌈Real.sqrt (16 / 9)⌉ + ⌈ (16 / 9 : ℝ ) ⌉ + ⌈Real.pow (16 / 9 : ℝ ) 2⌉) = 8 := 
by 
  sorry

end evaluate_expression_l147_147657


namespace percent_not_covering_politics_l147_147262

-- Definitions based on the conditions
def total_reporters : ℕ := 100
def local_politics_reporters : ℕ := 28
def percent_cover_local_politics : ℚ := 0.7

-- To be proved
theorem percent_not_covering_politics :
  let politics_reporters := local_politics_reporters / percent_cover_local_politics 
  (total_reporters - politics_reporters) / total_reporters = 0.6 := 
by
  sorry

end percent_not_covering_politics_l147_147262


namespace binary_addition_and_subtraction_correct_l147_147221

def add_binary_and_subtract : ℕ :=
  let n1 := 0b1101  -- binary for 1101_2
  let n2 := 0b0010  -- binary for 10_2
  let n3 := 0b0101  -- binary for 101_2
  let n4 := 0b1011  -- expected result 1011_2
  n1 + n2 + n3 - 0b0011  -- subtract binary for 11_2

theorem binary_addition_and_subtraction_correct : add_binary_and_subtract = 0b1011 := 
by 
  sorry

end binary_addition_and_subtraction_correct_l147_147221


namespace bowls_total_marbles_l147_147540

theorem bowls_total_marbles :
  let C2 := 600
  let C1 := (3 / 4 : ℝ) * C2
  let C3 := (1 / 2 : ℝ) * C1
  C1 = 450 ∧ C3 = 225 ∧ (C1 + C2 + C3 = 1275) := 
by
  let C2 := 600
  let C1 := (3 / 4 : ℝ) * C2
  let C3 := (1 / 2 : ℝ) * C1
  have hC1 : C1 = 450 := by norm_num
  have hC3 : C3 = 225 := by norm_num
  have hTotal : C1 + C2 + C3 = 1275 := by norm_num
  exact ⟨hC1, hC3, hTotal⟩

end bowls_total_marbles_l147_147540


namespace prob_dot_product_gt_half_l147_147599

theorem prob_dot_product_gt_half :
  let vertices := {i ∈ Finset.range 2017 | i ≠ 0}
  let pairs := { (i, j) ∈ vertices × vertices | i ≠ j }
  let count_pairs := pairs.card
  let dot_prod_condition := { (i, j) ∈ pairs | (Real.cos (2 * Real.pi * (i:Nnreal) / 2017 - 2 * Real.pi * (j:Nnreal) / 2017) > 1/2) }
  let count_condition := dot_prod_condition.card
  (count_condition / count_pairs : ℚ) = 1/3 :=
by
  let vertices := {i ∈ Finset.range 2017 | i ≠ 0}
  let pairs := { (i, j) ∈ vertices × vertices | i ≠ j }
  let count_pairs := pairs.card
  let dot_prod_condition := { (i, j) ∈ pairs | (Real.cos (2 * Real.pi * (i:Nnreal) / 2017 - 2 * Real.pi * (j:Nnreal) / 2017) > 1/2) }
  let count_condition := dot_prod_condition.card
  have h : (count_condition / count_pairs : ℚ) = 1/3, sorry
  exact h

end prob_dot_product_gt_half_l147_147599


namespace inconsistent_linear_system_l147_147495

theorem inconsistent_linear_system :
  ¬ ∃ (x1 x2 x3 : ℝ), 
    (2 * x1 + 5 * x2 - 4 * x3 = 8) ∧
    (3 * x1 + 15 * x2 - 9 * x3 = 5) ∧
    (5 * x1 + 5 * x2 - 7 * x3 = 1) :=
by
  -- Proof of inconsistency
  sorry

end inconsistent_linear_system_l147_147495


namespace min_positive_period_of_f_l147_147277

-- Define the function f
def f (x : ℝ) : ℝ := sqrt 3 * tan (x / 2 - π / 4)

-- State the theorem about the minimum positive period of f
theorem min_positive_period_of_f : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = 2 * π :=
by
  sorry

end min_positive_period_of_f_l147_147277


namespace largest_multiple_of_15_less_than_500_l147_147066

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), (n < 500) ∧ (15 ∣ n) ∧ (∀ m : ℕ, (m < 500) ∧ (15 ∣ m) → m ≤ n) ∧ n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l147_147066


namespace train_length_l147_147943

theorem train_length (speed_km_hr : ℝ) (time_seconds : ℝ) (speed_ms : ℝ) (distance_m : ℝ)
  (h1 : speed_km_hr = 90)
  (h2 : time_seconds = 9)
  (h3 : speed_ms = speed_km_hr * (1000 / 3600))
  (h4 : distance_m = speed_ms * time_seconds) :
  distance_m = 225 :=
by
  sorry

end train_length_l147_147943


namespace remaining_kibble_l147_147455

def starting_kibble : ℕ := 12
def mary_kibble_morning : ℕ := 1
def mary_kibble_evening : ℕ := 1
def frank_kibble_afternoon : ℕ := 1
def frank_kibble_late_evening : ℕ := 2 * frank_kibble_afternoon

theorem remaining_kibble : starting_kibble - (mary_kibble_morning + mary_kibble_evening + frank_kibble_afternoon + frank_kibble_late_evening) = 7 := by
  sorry

end remaining_kibble_l147_147455


namespace length_of_train_l147_147994

-- We state the problem as a theorem in Lean
theorem length_of_train (bridge_length : ℝ) (crossing_time : ℝ) (train_speed_kmh : ℝ)
  (h_bridge_length : bridge_length = 150)
  (h_crossing_time : crossing_time = 32)
  (h_train_speed_kmh : train_speed_kmh = 45) :
  ∃ (train_length : ℝ), train_length = 250 := 
by
  -- We assume the necessary conditions as given
  have train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)
  have total_distance : ℝ := train_speed_ms * crossing_time
  have train_length : ℝ := total_distance - bridge_length
  -- Conclude the length of the train is 250
  use train_length
  -- The proof steps are skipped using 'sorry'
  sorry

end length_of_train_l147_147994


namespace func_g_neither_even_nor_odd_l147_147803

noncomputable def func_g (x : ℝ) : ℝ := (⌈x⌉ : ℝ) - (1 / 3)

theorem func_g_neither_even_nor_odd :
  (¬ ∀ x, func_g (-x) = func_g x) ∧ (¬ ∀ x, func_g (-x) = -func_g x) :=
by
  sorry

end func_g_neither_even_nor_odd_l147_147803


namespace exists_hamiltonian_cycle_with_one_transport_change_l147_147527

-- Define the problem parameters and the conditions
variables {N : ℕ} (cities : Finset ℕ) (roads airways : Finset (ℕ × ℕ))

-- Problem constraints
def complete_graph (N : ℕ) (roads airways : Finset (ℕ × ℕ)) : Prop :=
  ∀ (i j : ℕ), i ≠ j → ((i, j) ∈ roads ∨ (i, j) ∈ airways)

-- Define what a Hamiltonian cycle is
def is_hamiltonian_cycle (cycle : List ℕ) (roads airways : Finset (ℕ × ℕ)) : Prop :=
  (List.Nodup cycle) ∧ (cycle.length = N + 1) ∧
  ∀ (k : ℕ), k < N → (((cycle.nth_le k _) , (cycle.nth_le (k+1) _)) ∈ roads ∨ 
                                    ((cycle.nth_le k _) , (cycle.nth_le (k+1) _)) ∈ airways)

-- Define the changing transportation condition
def changes_transportation_at_most_once (cycle : List ℕ) (roads airways : Finset (ℕ × ℕ)) : Prop :=
  let grouped_edges := List.groupBy (λ e, if e ∈ roads then "road" else "airway")
                                    (List.zip cycle (cycle.tail ++ [cycle.head]))
  in grouped_edges.length ≤ 2

-- The main theorem
theorem exists_hamiltonian_cycle_with_one_transport_change 
  (hc : complete_graph N roads airways) : 
  ∃ (cycle : List ℕ), is_hamiltonian_cycle cycle roads airways ∧ 
                      changes_transportation_at_most_once cycle roads airways :=
sorry

end exists_hamiltonian_cycle_with_one_transport_change_l147_147527


namespace negation_of_existence_l147_147888

theorem negation_of_existence (T : Type) (triangle : T → Prop) (sum_interior_angles : T → ℝ) :
  (¬ ∃ t : T, sum_interior_angles t ≠ 180) ↔ (∀ t : T, sum_interior_angles t = 180) :=
by 
  sorry

end negation_of_existence_l147_147888


namespace largest_multiple_of_15_less_than_500_l147_147052

theorem largest_multiple_of_15_less_than_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ (∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x) := sorry

end largest_multiple_of_15_less_than_500_l147_147052


namespace largest_multiple_of_15_less_than_500_l147_147136

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), 15 * n < 500 ∧ 15 * n ≥ 15 * (33 - 1) + 1 := by
  sorry

end largest_multiple_of_15_less_than_500_l147_147136


namespace book_distribution_l147_147561

theorem book_distribution (x : ℕ) (h1 : 9 * x + 7 < 11 * x) : 
  9 * x + 7 = totalBooks - 9 * x ∧ totalBooks - 9 * x = 7 :=
by
  sorry

end book_distribution_l147_147561


namespace log_x_y_eq_4pi_l147_147971

-- Definitions of the conditions
def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

def radius (x : ℝ) : ℝ := log_base 2 (x^3)
def circumference (y : ℝ) : ℝ := log_base 16 (y^6)

-- Statement of the proof problem
theorem log_x_y_eq_4pi {x y : ℝ} (hx : 0 < x) (hy : 0 < y) :
  let r := radius x,
      C := circumference y
  in log_base x y = 4 * Real.pi :=
begin
  sorry
end

end log_x_y_eq_4pi_l147_147971


namespace calculate_f_one_l147_147345

def f (x : ℝ) : ℝ := x^2 + |x - 2|

theorem calculate_f_one : f 1 = 2 := by
  sorry

end calculate_f_one_l147_147345


namespace largest_multiple_of_15_less_than_500_l147_147017

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), n < 500 ∧ 15 ∣ n ∧ ∀ (m : ℕ), m < 500 ∧ 15 ∣ m -> m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l147_147017


namespace largest_multiple_of_15_less_than_500_l147_147075

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), (n < 500) ∧ (15 ∣ n) ∧ (∀ m : ℕ, (m < 500) ∧ (15 ∣ m) → m ≤ n) ∧ n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l147_147075


namespace find_odd_and_increasing_function_l147_147612

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_monotonically_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop := 
  ∀ ⦃a b⦄, a ∈ I → b ∈ I → a < b → f a < f b

theorem find_odd_and_increasing_function :
  (is_odd_function (λ x : ℝ, x ^ 3)) ∧ (is_monotonically_increasing (λ x : ℝ, x ^ 3) {x | x > 0}) ∧ 
  ¬(is_odd_function (λ x : ℝ, |x|)) ∧ 
  ¬(is_odd_function (λ x : ℝ, real.log x)) ∧ 
  ¬(is_monotonically_increasing (λ x : ℝ, x⁻¹) {x | x > 0}) :=
by sorry

end find_odd_and_increasing_function_l147_147612


namespace correct_statement_is_D_l147_147939

/-
Given the following statements and their conditions:
A: Conducting a comprehensive survey is not an accurate approach to understand the sleep situation of middle school students in Changsha.
B: The mode of the dataset \(-1\), \(2\), \(5\), \(5\), \(7\), \(7\), \(4\) is not \(7\) only, because both \(5\) and \(7\) are modes.
C: A probability of precipitation of \(90\%\) does not guarantee it will rain tomorrow.
D: If two datasets, A and B, have the same mean, and the variances \(s_{A}^{2} = 0.3\) and \(s_{B}^{2} = 0.02\), then set B with a lower variance \(s_{B}^{2}\) is more stable.

Prove that the correct statement based on these conditions is D.
-/
theorem correct_statement_is_D
  (dataset_A dataset_B : Type)
  (mean_A mean_B : ℝ)
  (sA2 sB2 : ℝ)
  (h_same_mean: mean_A = mean_B)
  (h_variances: sA2 = 0.3 ∧ sB2 = 0.02)
  (h_stability: sA2 > sB2) :
  (if sA2 = 0.3 ∧ sB2 = 0.02 ∧ sA2 > sB2 then "D" else "not D") = "D" := by
  sorry

end correct_statement_is_D_l147_147939


namespace no_positive_integer_satisfies_condition_l147_147442

def sum_of_divisors (n : ℕ) : ℕ := 
  Finset.sum (Finset.filter (λ d, n % d = 0) (Finset.range (n + 1))) id

theorem no_positive_integer_satisfies_condition : 
  (∀ n : ℕ, n > 0 → sum_of_divisors (sum_of_divisors n) ≠ n + 3) :=
by
  intros n hn
  sorry -- Proof needed

end no_positive_integer_satisfies_condition_l147_147442


namespace maximize_car_travel_l147_147296

theorem maximize_car_travel (front_tires : ℕ) (rear_tires : ℕ) (x : ℕ)
  (h1 : front_tires = 42000) (h2 : rear_tires = 56000) : 
  let total_distance := min (x + (56000 - x)) (x + (42000 - x))
  in total_distance = 42000 :=
by
  sorry

end maximize_car_travel_l147_147296


namespace largest_whole_number_satisfying_inequality_l147_147145

theorem largest_whole_number_satisfying_inequality : ∃ n : ℤ, (1 / 3 + n / 7 < 1) ∧ (∀ m : ℤ, (1 / 3 + m / 7 < 1) → m ≤ n) ∧ n = 4 :=
sorry

end largest_whole_number_satisfying_inequality_l147_147145


namespace triangle_area_l147_147703

theorem triangle_area (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
                      {a b c : Real} (h_right_triangle : a * a + b * b = c * c)
                      (angle_BAC : ∠ BAC = 45)
                      (len_BC : c = 6) :
                      1/2 * a * b = 9 :=
by
  sorry

end triangle_area_l147_147703


namespace evaluate_ceiling_sum_l147_147641

theorem evaluate_ceiling_sum :
  (⌈Real.sqrt (16 / 9)⌉ : ℤ) + (⌈(16 / 9: ℝ)⌉ : ℤ) + (⌈(16 / 9: ℝ)^2⌉ : ℤ) = 8 := 
by
  -- Placeholder for proof
  sorry

end evaluate_ceiling_sum_l147_147641


namespace find_min_value_of_g_l147_147334

noncomputable def f : ℝ → ℝ := sorry

def strictly_decreasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f y < f x

theorem find_min_value_of_g :
  (∀ x : ℝ, 0 < x → f x * f (f x + 2 / x) = 1 / 3) →
  strictly_decreasing f (set.Ioi 0) →
  ∃ x : ℝ, 0 < x ∧ ∀ y : ℝ, 0 < y → (g y = f y + 4 * y^2) → g y ≥ 3 :=
  sorry

end find_min_value_of_g_l147_147334


namespace min_distance_from_curve_to_line_l147_147892

noncomputable def point_on_curve (x : ℝ) : ℝ := Real.exp x

def minimum_distance_to_line (P Q : ℝ × ℝ) : ℝ := 
  abs (Q.fst * P.fst - Q.snd * P.snd + 0) / Real.sqrt (1^2 + (-1)^2)

theorem min_distance_from_curve_to_line :
  ∃ (x : ℝ), minimum_distance_to_line (0, 1) (x, point_on_curve x) = Real.sqrt 2 / 2 :=
sorry

end min_distance_from_curve_to_line_l147_147892


namespace area_bounded_by_equation_and_line_l147_147551

-- Definition of the equation and the line
def equation (x y : ℝ) : Prop := (x^2 + y^2 = 2 * (|x| + |y|))

def line (x y : ℝ) : Prop := (x + y = 2)

-- Statement about the area bounded by the given equation and line
theorem area_bounded_by_equation_and_line :
  ∀ (x y : ℝ), equation x y ∧ line x y → (bounded_area (equation x y) (line x y) = π) :=
sorry

end area_bounded_by_equation_and_line_l147_147551


namespace largest_multiple_of_15_less_than_500_l147_147142

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), 15 * n < 500 ∧ 15 * n ≥ 15 * (33 - 1) + 1 := by
  sorry

end largest_multiple_of_15_less_than_500_l147_147142


namespace distance_A_B_solve_for_x_possible_values_of_x_motion_time_distance_E_P_distance_E_P2_distance_E_P3_distance_E_P4_l147_147176

-- Definition of distance on the number line
def distance (x₁ x₂ : ℤ) : ℤ := |x₁ - x₂|

-- Preliminary Application
def pointA : ℤ := -1
def pointB : ℤ := 2

theorem distance_A_B : distance pointA pointB = 3 :=
by
  rw [distance, pointA, pointB]
  norm_num
  sorry

theorem solve_for_x (x : ℤ) : distance pointA x = 2 ↔ x = 1 ∨ x = -3 :=
by
  rw [distance, pointA]
  norm_num
  sorry

theorem possible_values_of_x (x : ℤ) : (distance pointA x + distance pointB x = distance pointA pointB) ↔ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2 :=
by
  rw [distance, pointA, pointB]
  norm_num
  sorry

-- Comprehensive Application
def pointD : ℤ := -2
def pointF : ℤ := 6
def speed : ℤ := 2

theorem motion_time (t : ℝ) : (distance pointD (2 * t - 2) = 3) ↔ t = 1.5 ∨ t =... 6.5 :=
by
  rw [distance, pointD]
  sorry

def pointE : ℤ := 4

theorem distance_E_P (t : ℝ) : 
0 < t ∧ t ≤ 3 → distance pointE (2 * t - 2) = 6 - 2 * t :=
by
  rw [distance, pointE]
  norm_num
  sorry

theorem distance_E_P2 (t : ℝ) : 
3 < t ∧ t ≤ 4 → distance pointE (2 * t - 2) = 2 * t - 6 :=
by
  rw [distance, pointE]
  norm_num
  sorry

theorem distance_E_P3 (t : ℝ) : 
4 < t ∧ t ≤ 5 → distance pointE (10 - 2 * t) = 10 - 2 * t :=
by
  rw [distance, pointE]
  norm_num
  sorry

theorem distance_E_P4 (t : ℝ) : 
5 < t ∧ t ≤ 8 → distance pointE (2 * t - 10) = 2 * t - 10 :=
by
  rw [distance, pointE]
  norm_num
  sorry

end distance_A_B_solve_for_x_possible_values_of_x_motion_time_distance_E_P_distance_E_P2_distance_E_P3_distance_E_P4_l147_147176


namespace eval_expression_l147_147679

noncomputable def ceil_sqrt_16_div_9 : ℕ := ⌈Real.sqrt (16 / 9 : ℚ)⌉
noncomputable def ceil_16_div_9 : ℕ := ⌈(16 / 9 : ℚ)⌉
noncomputable def ceil_square_16_div_9 : ℕ := ⌈(16 / 9 : ℚ)^2⌉

theorem eval_expression : ceil_sqrt_16_div_9 + ceil_16_div_9 + ceil_square_16_div_9 = 8 :=
by
  -- The following sorry is a placeholder, indicating that the proof is skipped.
  sorry

end eval_expression_l147_147679


namespace hyperbola_eccentricity_l147_147323

theorem hyperbola_eccentricity (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) :
  let F1 := (-c, 0)
      F2 := (c, 0)
      hyp_eq := (x : ℝ) (y : ℝ), x^2 / a^2 - y^2 / b^2 = 1
      line_parallel := (x - c) * b / a
      other_asymptote := - x * b / a
      M := (c / 2, - b * c / (2 * a))
  in ((c / 2)^2 + (b * c / (2 * a))^2 = c^2) → e = 2 :=
begin
  sorry
end

end hyperbola_eccentricity_l147_147323


namespace sqrt_expr_sum_equality_l147_147398

theorem sqrt_expr_sum_equality :
  let a := 14
  let b := 3
  let c := 2
  (\sqrt(6) + 1/\sqrt(6) + \sqrt(8) + 1/\sqrt(8) = (a * \sqrt(6) + b * \sqrt(8)) / c) :=
begin
  sorry
end

lemma abc_sum :
  let a := 14
  let b := 3
  let c := 2
  (a + b + c = 19) :=
begin
  sorry
end

end sqrt_expr_sum_equality_l147_147398


namespace largest_multiple_of_15_less_than_500_is_495_l147_147120

-- Define the necessary conditions
def is_multiple_of_15 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 15 * k

def is_less_than_500 (n : ℕ) : Prop :=
  n < 500

-- Problem statement: Prove that the largest positive multiple of 15 less than 500 is 495
theorem largest_multiple_of_15_less_than_500_is_495 : 
  ∀ n : ℕ, is_multiple_of_15 n → is_less_than_500 n → n ≤ 495 := 
by
  sorry

end largest_multiple_of_15_less_than_500_is_495_l147_147120


namespace traffic_light_probability_change_l147_147215

theorem traffic_light_probability_change :
  let cycle_length := 100
  let green_duration := 45
  let yellow_duration := 5
  let red_duration := cycle_length - green_duration - yellow_duration
  let observation_interval := 5
  let critical_intervals := 3
  let total_intervals := cycle_length / observation_interval
  in (critical_intervals / total_intervals : ℚ) = 3 / 20 := by
  sorry

end traffic_light_probability_change_l147_147215


namespace real_part_of_z_l147_147341

open Complex

theorem real_part_of_z :
  let z : ℂ := (⟨1, 2⟩ * ⟨3, -1⟩) in
  re z = 5 :=
by
  let z : ℂ := (⟨1, 2⟩ * ⟨3, -1⟩)
  have h : re z = 5, skip
  exact h

end real_part_of_z_l147_147341


namespace evaluate_expression_l147_147663

theorem evaluate_expression :
  let x := (16 : ℚ) / 9
  in ⌈(√x)⌉ + ⌈x⌉ + ⌈x^2⌉ = 8 :=
by
  let x := (16 : ℚ) / 9
  sorry

end evaluate_expression_l147_147663


namespace hermia_elected_probability_l147_147833

def probability_hermia_elected (n : ℕ) (h1 : Odd n) (h2 : n > 0) : ℝ :=
  (2 ^ n - 1 : ℝ) / (n * 2 ^ (n - 1))

theorem hermia_elected_probability (n : ℕ) (h1 : Odd n) (h2 : 0 < n) :
  probability_hermia_elected n h1 h2 = (2 ^ n - 1 : ℝ) / (n * 2 ^ (n - 1)) :=
sorry

end hermia_elected_probability_l147_147833


namespace evaluate_expression_l147_147660

theorem evaluate_expression : 
  (⌈Real.sqrt (16 / 9)⌉ + ⌈ (16 / 9 : ℝ ) ⌉ + ⌈Real.pow (16 / 9 : ℝ ) 2⌉) = 8 := 
by 
  sorry

end evaluate_expression_l147_147660


namespace largest_multiple_of_15_below_500_l147_147030

theorem largest_multiple_of_15_below_500 : ∃ (k : ℕ), 15 * k < 500 ∧ ∀ (m : ℕ), 15 * m < 500 → 15 * m ≤ 15 * k := 
by
  existsi 33
  split
  · norm_num
  · intro m h
    have h1 : m ≤ 33 := Nat.le_of_lt_succ (Nat.div_lt_succ 500 15 m)
    exact (mul_le_mul_left (by norm_num : 0 < 15)).mpr h1
  sorry

end largest_multiple_of_15_below_500_l147_147030


namespace repeating_decimal_to_fraction_l147_147695

theorem repeating_decimal_to_fraction : ∃ (x : ℚ), x = 3 + 145 / 999 ∧ x = 3142 / 999 := 
begin
    sorry
end

end repeating_decimal_to_fraction_l147_147695


namespace center_square_is_8_l147_147941

-- Define the problem conditions in Lean
def is_adjacency_valid (grid : ℕ → ℕ → ℕ) : Prop :=
  ∀ n, 1 ≤ n ∧ n < 9 →
    (∃ (i j i' j' : ℕ),
      (grid i j = n ∧ grid i' j' = n + 1) ∧ (abs (i - i') = 1 ∧ j = j' ∨ abs (j - j') = 1 ∧ i = i'))

def is_sum_corners_equal_20 (grid : ℕ → ℕ → ℕ) : Prop :=
  grid 0 0 + grid 0 2 + grid 2 0 + grid 2 2 = 20

def is_sum_middle_column_even (grid : ℕ → ℕ → ℕ) : Prop :=
  (grid 0 1 + grid 1 1 + grid 2 1) % 2 = 0

-- The main theorem stating the number in the center square is 8
theorem center_square_is_8
  (grid : ℕ → ℕ → ℕ)
  (h_nums : ∀ i j, 0 ≤ grid i j ∧ grid i j ≤ 9 ∧ (∀ i' j', grid i' j' ≠ grid i j → (i, j) ≠ (i', j')))
  (h_adjacency : is_adjacency_valid grid)
  (h_corners_sum : is_sum_corners_equal_20 grid)
  (h_middle_even : is_sum_middle_column_even grid) :
  grid 1 1 = 8 :=
sorry

end center_square_is_8_l147_147941


namespace eval_expression_l147_147678

noncomputable def ceil_sqrt_16_div_9 : ℕ := ⌈Real.sqrt (16 / 9 : ℚ)⌉
noncomputable def ceil_16_div_9 : ℕ := ⌈(16 / 9 : ℚ)⌉
noncomputable def ceil_square_16_div_9 : ℕ := ⌈(16 / 9 : ℚ)^2⌉

theorem eval_expression : ceil_sqrt_16_div_9 + ceil_16_div_9 + ceil_square_16_div_9 = 8 :=
by
  -- The following sorry is a placeholder, indicating that the proof is skipped.
  sorry

end eval_expression_l147_147678


namespace expansion_contains_no_x2_l147_147333

theorem expansion_contains_no_x2 (n : ℕ) (h1 : 5 ≤ n ∧ n ≤ 8) :
  ¬ (∃ k, (x + 1)^2 * (x + 1 / x^3)^n = k * x^2) → n = 7 :=
sorry

end expansion_contains_no_x2_l147_147333


namespace geom_sequence_relation_l147_147525

variable (a1 : ℝ) (q : ℝ) (n : ℕ)

def sum_geom (a1 q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then a1 * n else a1 * (1 - q^n) / (1 - q)

theorem geom_sequence_relation (A B C : ℝ) (hA : A = sum_geom a1 q n) 
    (hB : B = sum_geom a1 q (2 * n)) (hC : C = sum_geom a1 q (3 * n)) :
  B * (B - A) = A * (C - A) := 
by 
  -- Proof goes here
  sorry

end geom_sequence_relation_l147_147525


namespace simplify_and_evaluate_expr_evaluate_at_zero_l147_147489

theorem simplify_and_evaluate_expr (x : ℝ) (hx1 : x ≠ 1) (hx2 : x ≠ 2) :
  (3 / (x - 1) - x - 1) / ((x^2 - 4 * x + 4) / (x - 1)) = (2 + x) / (2 - x) :=
by
  sorry

theorem evaluate_at_zero :
  (2 + 0 : ℝ) / (2 - 0) = 1 :=
by
  norm_num

end simplify_and_evaluate_expr_evaluate_at_zero_l147_147489


namespace radius_of_incircle_eq_l147_147497

variables {A B C D E F G : Type} [triangle A B C] [incircle ω A B C D E]
  (BF FG GA : ℝ) (r : ℝ)

def proper_conditions (ω : incircle A B C D E) (BF FG GA : ℝ) : Prop := 
  tangent ω BC D ∧ 
  tangent ω CA E ∧ 
  bisects (angle_bisector A) DE F ∧ 
  bisects (angle_bisector B) DE G ∧
  BF = 1 ∧ 
  FG = 6 ∧ 
  GA = 6

theorem radius_of_incircle_eq : proper_conditions ω BF FG GA →
  radius_of_incircle ω = r :=
by sorry

end radius_of_incircle_eq_l147_147497


namespace largest_multiple_of_15_below_500_l147_147037

theorem largest_multiple_of_15_below_500 : ∃ (k : ℕ), 15 * k < 500 ∧ ∀ (m : ℕ), 15 * m < 500 → 15 * m ≤ 15 * k := 
by
  existsi 33
  split
  · norm_num
  · intro m h
    have h1 : m ≤ 33 := Nat.le_of_lt_succ (Nat.div_lt_succ 500 15 m)
    exact (mul_le_mul_left (by norm_num : 0 < 15)).mpr h1
  sorry

end largest_multiple_of_15_below_500_l147_147037


namespace initial_outlay_l147_147485

-- Definition of given conditions
def manufacturing_cost (I : ℝ) (sets : ℕ) (cost_per_set : ℝ) : ℝ := I + sets * cost_per_set
def revenue (sets : ℕ) (price_per_set : ℝ) : ℝ := sets * price_per_set
def profit (revenue manufacturing_cost : ℝ) : ℝ := revenue - manufacturing_cost

-- Given data
def sets : ℕ := 500
def cost_per_set : ℝ := 20
def price_per_set : ℝ := 50
def given_profit : ℝ := 5000

-- The statement to prove
theorem initial_outlay (I : ℝ) : 
  profit (revenue sets price_per_set) (manufacturing_cost I sets cost_per_set) = given_profit → 
  I = 10000 := by
  sorry

end initial_outlay_l147_147485


namespace number_of_elements_begin_with_1_l147_147820

-- Define the set T
def T : set ℕ := {n | ∃ k : ℤ, 0 ≤ k ∧ k ≤ 1500 ∧ n = 3^k}

-- Given conditions
def cond_1 : 3^1500 = 3691...2743 := sorry
def cond_2 : nat.digits 10 (3^1500) = 717 := sorry

-- The theorem to prove
theorem number_of_elements_begin_with_1 : 
  (∃ count : ℕ, count = 784 ∧ (∀ n ∈ T, nat.digits 10 n = 717 ∨ n = 1)) :=
sorry

end number_of_elements_begin_with_1_l147_147820


namespace quadratic_inequality_solution_l147_147722

-- Definitions based on given conditions
def is_quadratic (f : ℝ → ℝ) : Prop := ∃ a b c, a ≠ 0 ∧ f = λ x, a * x^2 + b * x + c

def solution_set (f : ℝ → ℝ) (a : ℝ) : Set ℝ :=
  if -1 < a ∧ a < 0 then {x : ℝ | x < 0 ∨ (5 < x ∧ x < -5/a)}
  else if a = -1 then {x : ℝ | x < 0}
  else if a < -1 then {x : ℝ | x < 0 ∨ (-5/a < x ∧ x < 5)}
  else ∅

-- Problem statement in Lean
theorem quadratic_inequality_solution :
  ∃ (f : ℝ → ℝ), is_quadratic f
  ∧ (∀ x, f x < 0 ↔ 0 < x ∧ x < 5)
  ∧ (∀ a, a < 0 → (∀ x, (2 * x^2 + (a - 10) * x + 5) / f x > 1 ↔ x ∈ solution_set f a)) :=
by
  sorry

end quadratic_inequality_solution_l147_147722


namespace y_values_l147_147818

def conditions (x : ℝ) : Prop :=
  x^2 + 6 * (x / (x - 3)) ^ 2 = 81

def y (x : ℝ) : ℝ :=
  ((x - 3) ^ 2 * (x + 4)) / (3 * x - 4)

theorem y_values :
  {y | ∃ x : ℝ, conditions x ∧ y = ((x - 3) ^ 2 * (x + 4)) / (3 * x - 4)}
  = {-9, 225 / 176} :=
sorry

end y_values_l147_147818


namespace cube_volume_skew_diagonals_l147_147915

theorem cube_volume_skew_diagonals (a : ℝ) :
  (∃ d1 d2 : ℝ, d1 ≠ d2 ∧ distance d1 d2 = 1) →
  ∃ V : ℝ, V = a^3 ∧ (V = 1 ∨ V = 3 * sqrt 3) :=
by sorry

end cube_volume_skew_diagonals_l147_147915


namespace min_price_per_car_to_avoid_losses_min_order_volume_to_compete_l147_147162

section TeslaModelS

variables (purchase_price : ℝ) (customs_duties : ℝ)
          (monthly_rent : ℝ) (monthly_salary : ℝ)
          (other_expenses : ℝ) (num_cars : ℝ)
          (competitor_price : ℝ) (charging_station_price : ℝ)

def total_custom_expenses (num_cars : ℝ) (purchase_price : ℝ) (customs_duties : ℝ) : ℝ :=
  num_cars * (purchase_price + customs_duties)

def total_fixed_expenses (monthly_rent : ℝ) (monthly_salary : ℝ) (other_expenses : ℝ) : ℝ :=
  monthly_rent + monthly_salary + other_expenses

def total_expenses (num_cars : ℝ) (purchase_price : ℝ) (customs_duties : ℝ) 
                   (monthly_rent : ℝ) (monthly_salary : ℝ) (other_expenses : ℝ) : ℝ :=
  total_custom_expenses num_cars purchase_price customs_duties + 
    total_fixed_expenses monthly_rent monthly_salary other_expenses

def average_cost_per_car (num_cars : ℝ) (purchase_price : ℝ) (customs_duties : ℝ)
                          (monthly_rent : ℝ) (monthly_salary : ℝ) (other_expenses : ℝ) : ℝ :=
  total_expenses num_cars purchase_price customs_duties monthly_rent monthly_salary other_expenses / num_cars

theorem min_price_per_car_to_avoid_losses : 
  average_cost_per_car 30 2.5 2 0.05 0.37 0.18 = 4.52 := 
  sorry

theorem min_order_volume_to_compete : 
  ∃ (x : ℝ), x ≥ 2 ∧ (charging_station_price = 0.4 ∧ competitor_price = 5.3 ∧
  (average_cost_per_car x 2.5 2 0.05 0.37 0.18 ≤ competitor_price - charging_station_price)) :=
  sorry

end TeslaModelS

end min_price_per_car_to_avoid_losses_min_order_volume_to_compete_l147_147162


namespace elixir_combinations_l147_147220

theorem elixir_combinations : 
  (∃ H G : ℕ, H = 4 ∧ G = 5 ∧ (∃ incompatible_combinations : ℕ, incompatible_combinations = 3 ∧ ((H * G) - incompatible_combinations = 17))) :=
by
  let H := 4
  let G := 5
  let incompatible_combinations := 3
  have initial_combinations : ℕ := H * G
  have valid_combinations : ℕ := initial_combinations - incompatible_combinations
  exact Exists.intro H (Exists.intro G (And.intro rfl (And.intro rfl (Exists.intro incompatible_combinations (And.intro rfl (by rw [valid_combinations]; exact rfl))))))

end elixir_combinations_l147_147220


namespace dave_age_l147_147177

theorem dave_age (C D E : ℝ) (h1 : C = 4 * D) (h2 : E = D + 5) (h3 : C = E) : D = 5 / 3 :=
by
  sorry

end dave_age_l147_147177


namespace find_value_of_z_l147_147327

theorem find_value_of_z (z : ℂ) (h1 : ∀ a : ℝ, z = a * I) (h2 : ((z + 2) / (1 - I)).im = 0) : z = -2 * I :=
sorry

end find_value_of_z_l147_147327


namespace polynomial_is_x_plus_one_l147_147174

theorem polynomial_is_x_plus_one (p : ℤ → ℤ) (h1 : ∀ n : ℕ, p n > n)
  (h2 : ∀ N : ℕ, ∃ i : ℕ, (λ s : ℕ → ℕ, s 1 = 1 ∧ ∀ i : ℕ, s (i + 1) = p (s i)) i ∣ N) :
  p = λ x, x + 1 := 
sorry

end polynomial_is_x_plus_one_l147_147174


namespace largest_multiple_of_15_below_500_l147_147027

theorem largest_multiple_of_15_below_500 : ∃ (k : ℕ), 15 * k < 500 ∧ ∀ (m : ℕ), 15 * m < 500 → 15 * m ≤ 15 * k := 
by
  existsi 33
  split
  · norm_num
  · intro m h
    have h1 : m ≤ 33 := Nat.le_of_lt_succ (Nat.div_lt_succ 500 15 m)
    exact (mul_le_mul_left (by norm_num : 0 < 15)).mpr h1
  sorry

end largest_multiple_of_15_below_500_l147_147027


namespace height_difference_percentage_l147_147947

theorem height_difference_percentage (q p : ℝ) (h : p = 0.6 * q) : (q - p) / p * 100 = 66.67 := 
by
  sorry

end height_difference_percentage_l147_147947


namespace count_numbers_divisible_by_square_factors_l147_147366

theorem count_numbers_divisible_by_square_factors :
  {n : ℕ | 1 ≤ n ∧ n ≤ 100 ∧ (∃ k : ℕ, k > 1 ∧ k^2 ∣ n)}.toFinset.card = 42 :=
by 
  sorry

end count_numbers_divisible_by_square_factors_l147_147366


namespace critical_points_l147_147246

def g (x : ℝ) : ℝ :=
  if (-3 < x ∧ x ≤ 0) then -x - 3 else
  if (0 < x ∧ x ≤ 2) then x - 3 else
  if (2 < x ∧ x ≤ 3) then x^2 - 4*x + 6 else 0

theorem critical_points :
  (∀ x ∈ Icc (-3 : ℝ) 3, 
    (g x = - x - 3 ∧ -3 < x ∧ x ≤ 0) ∨
    (g x = x - 3 ∧ 0 < x ∧ x ≤ 2) ∨
    (g x = x^2 - 4 * x + 6 ∧ 2 < x ∧ x ≤ 3)) →
  (∀ x, g' x = 0 → 
    (x = 0 → g x = -3) ∨ 
    (x = 2 → g x = 2) ∧ 
    g'' x > 0) :=
sorry

end critical_points_l147_147246


namespace tom_read_chapters_l147_147537

theorem tom_read_chapters (chapters pages: ℕ) (h1: pages = 8 * chapters) (h2: pages = 24):
  chapters = 3 :=
by
  sorry

end tom_read_chapters_l147_147537


namespace women_science_majors_l147_147784

theorem women_science_majors :
  ∀ (total_class_percentage men_percentage non_science_percentage men_science_percentage : ℝ),
  total_class_percentage = 100 →
  men_percentage = 40 → 
  non_science_percentage = 60 →
  men_science_percentage = 70 →
  (100 - non_science_percentage) = (0.7 * men_percentage + (total_class_percentage - men_percentage - non_science_percentage) * (100 / (total_class_percentage - men_percentage))) :=
  
begin
  intros total_class_percentage men_percentage non_science_percentage men_science_percentage h1 h2 h3 h4,
  -- Initialize values based on conditions
  let total_science_percentage := 100 - non_science_percentage,
  let men_science_class_percentage := men_science_percentage * men_percentage / 100,
  let women_science_class_percentage := total_science_percentage - men_science_class_percentage,
  let women_percentage := total_class_percentage - men_percentage,
  -- Calculate the percentage of women who are science majors
  have h6 : women_science_percentage = women_science_class_percentage / women_percentage * 100,
  exact h5,
end

end women_science_majors_l147_147784


namespace dice_product_probability_l147_147920

def dice_count := 8
def dice_faces := 6

noncomputable def probability_divisible_by_4_and_3 : ℚ := 
  1554975 / 1679616

theorem dice_product_probability :
  let p_div_4 : ℚ := (1 - (1 / 2)^dice_count - dice_count * ((1 / 2)^dice_count)) in
  let p_div_3 : ℚ := (1 - (2 / 3)^dice_count) in
  let p_combined : ℚ := p_div_4 * p_div_3 in
  (1 - p_combined) = probability_divisible_by_4_and_3 := 
by sorry

end dice_product_probability_l147_147920


namespace sqrt_expr_to_rational_form_l147_147393

theorem sqrt_expr_to_rational_form :
  ∃ (a b c : ℕ), 0 < c ∧ (∑ i in [28, 27, 12], i) = 28 + 27 + 12 ∧
  (sqrt 6 + 1 / sqrt 6 + sqrt 8 + 1 / sqrt 8 = (a * sqrt 6 + b * sqrt 8) / c) ∧
  a + b + c = 67 := 
by
  use 28, 27, 12
  split
  { exact nat.succ_pos' _ }
  split
  { norm_num }
  split
  { -- omitted proof
    sorry }
  { norm_num }

end sqrt_expr_to_rational_form_l147_147393


namespace rice_on_8th_day_l147_147566

variable (a1 : ℕ) (d : ℕ) (n : ℕ)
variable (rice_per_laborer : ℕ)

def is_arithmetic_sequence (a1 d : ℕ) (n : ℕ) : ℕ :=
  a1 + (n - 1) * d

theorem rice_on_8th_day (ha1 : a1 = 64) (hd : d = 7) (hr : rice_per_laborer = 3) :
  let a8 := is_arithmetic_sequence a1 d 8
  (a8 * rice_per_laborer = 339) :=
by
  sorry

end rice_on_8th_day_l147_147566


namespace find_d_l147_147825

def f (x : ℝ) (c : ℝ) : ℝ := 5 * x + c
def g (x : ℝ) (c : ℝ) : ℝ := c * x + 3

theorem find_d (c d : ℝ) (h : ∀ x : ℝ, f (g x c) c = 15 * x + d) : d = 18 :=
by
  sorry

end find_d_l147_147825


namespace angle_C_value_l147_147795

-- Define the type for representing angles
def angle := ℝ

-- Define the condition of the problem: Given quadrilateral ABCD is a parallelogram
-- and the sum of angles A and C is 140 degrees.
variable (a c : angle)
variable (parallelogram : True)  -- A placeholder for parallelogram property
variable (angle_sum : a + c = 140)  -- Given condition sum of angles

-- The theorem stating the desired conclusion that ∠C = 70 degrees
theorem angle_C_value (h : parallelogram) (h' : a = c) (h'' : angle_sum) : c = 70 := 
sorry  -- Proof is omitted


end angle_C_value_l147_147795


namespace hermia_elected_probability_l147_147834

def probability_hermia_elected (n : ℕ) (h1 : Odd n) (h2 : n > 0) : ℝ :=
  (2 ^ n - 1 : ℝ) / (n * 2 ^ (n - 1))

theorem hermia_elected_probability (n : ℕ) (h1 : Odd n) (h2 : 0 < n) :
  probability_hermia_elected n h1 h2 = (2 ^ n - 1 : ℝ) / (n * 2 ^ (n - 1)) :=
sorry

end hermia_elected_probability_l147_147834


namespace mn_minus_one_not_prime_l147_147873

theorem mn_minus_one_not_prime {m n : ℤ} (hm : m ≥ 2) (hn : n ≥ 2) (hdiv : m + n - 1 ∣ m^2 + n^2 - 1) : ¬ prime (m + n - 1) :=
sorry

end mn_minus_one_not_prime_l147_147873


namespace solve_exponential_eq_l147_147899

theorem solve_exponential_eq (x : ℝ) : 4^x - 6 * 2^x + 8 = 0 ↔ x = 1 ∨ x = 2 :=
by sorry

end solve_exponential_eq_l147_147899


namespace sqrt_expr_to_rational_form_l147_147394

theorem sqrt_expr_to_rational_form :
  ∃ (a b c : ℕ), 0 < c ∧ (∑ i in [28, 27, 12], i) = 28 + 27 + 12 ∧
  (sqrt 6 + 1 / sqrt 6 + sqrt 8 + 1 / sqrt 8 = (a * sqrt 6 + b * sqrt 8) / c) ∧
  a + b + c = 67 := 
by
  use 28, 27, 12
  split
  { exact nat.succ_pos' _ }
  split
  { norm_num }
  split
  { -- omitted proof
    sorry }
  { norm_num }

end sqrt_expr_to_rational_form_l147_147394


namespace solve_for_b_l147_147405

theorem solve_for_b (a b c : ℝ) (cosC : ℝ) (h_a : a = 3) (h_c : c = 4) (h_cosC : cosC = -1/4) :
    c^2 = a^2 + b^2 - 2 * a * b * cosC → b = 7 / 2 :=
by 
  intro h_cosine_theorem
  sorry

end solve_for_b_l147_147405


namespace evaluate_expression_l147_147655

theorem evaluate_expression : 
  (⌈Real.sqrt (16 / 9)⌉ + ⌈ (16 / 9 : ℝ ) ⌉ + ⌈Real.pow (16 / 9 : ℝ ) 2⌉) = 8 := 
by 
  sorry

end evaluate_expression_l147_147655


namespace rotation_90_degrees_y_axis_l147_147607

def rotation_matrix_y_90 := 
  !![![0, 0, 1], 
    ![0, 1, 0], 
    ![-1, 0, 0]]

def input_vector := !![3, -2, 1]

def expected_output_vector := !![1, -2, -3]

theorem rotation_90_degrees_y_axis:
  rotation_matrix_y_90.mul_vec input_vector = expected_output_vector :=
by sorry

end rotation_90_degrees_y_axis_l147_147607


namespace ceiling_sum_evaluation_l147_147686

noncomputable def evaluateCeilingSum : ℝ := 
  ⌈Real.sqrt (16 / 9)⌉ + ⌈(16 / 9)⌉ + ⌈((16 / 9) ^ 2)⌉ 

theorem ceiling_sum_evaluation : evaluateCeilingSum = 8 := by
  sorry

end ceiling_sum_evaluation_l147_147686


namespace solve_eq_nonzero_solve_eq_zero_zero_solve_eq_zero_nonzero_l147_147564

-- Case 1: a ≠ 0
theorem solve_eq_nonzero (a b : ℝ) (h : a ≠ 0) : ∃ x : ℝ, x = -b / a ∧ a * x + b = 0 :=
by
  sorry

-- Case 2: a = 0 and b = 0
theorem solve_eq_zero_zero (a b : ℝ) (h1 : a = 0) (h2 : b = 0) : ∀ x : ℝ, a * x + b = 0 :=
by
  sorry

-- Case 3: a = 0 and b ≠ 0
theorem solve_eq_zero_nonzero (a b : ℝ) (h1 : a = 0) (h2 : b ≠ 0) : ¬ ∃ x : ℝ, a * x + b = 0 :=
by
  sorry

end solve_eq_nonzero_solve_eq_zero_zero_solve_eq_zero_nonzero_l147_147564


namespace max_sum_abc_l147_147316

theorem max_sum_abc (a b c : ℝ) (h1 : 1 ≤ a) (h2 : 1 ≤ b) (h3 : 1 ≤ c) 
  (h4 : a * b * c + 2 * a^2 + 2 * b^2 + 2 * c^2 + c * a - c * b - 4 * a + 4 * b - c = 28) :
  a + b + c ≤ 6 :=
sorry

end max_sum_abc_l147_147316


namespace rectangle_dim_int_l147_147203

theorem rectangle_dim_int (a b : ℝ) (H : ∃ n m : ℕ, a * n = b * m) : a ∈ ℤ ∨ b ∈ ℤ :=
sorry

end rectangle_dim_int_l147_147203


namespace total_mixture_price_l147_147244

theorem total_mixture_price : 
  ∀ (cashew_price peanut_price mix_pounds cashew_pounds peanut_pounds total_price : ℝ),
  cashew_price = 5.00 ∧ 
  peanut_price = 2.00 ∧ 
  mix_pounds = 25 ∧ 
  cashew_pounds = 11 ∧ 
  peanut_pounds = mix_pounds - cashew_pounds ∧ 
  total_price = cashew_pounds * cashew_price + peanut_pounds * peanut_price ->
  total_price = 83.00 :=
by
  intros cashew_price peanut_price mix_pounds cashew_pounds peanut_pounds total_price h
  
  -- extracting conditions
  cases h with hc h,
  cases h with hp h,
  cases h with hm h,
  cases h with hcs h,
  cases h with hpq h,
  cases h with ht,
  
  -- substituting values according to conditions
  simp [hc, hp, hm, hcs, ht, hpq]
  sorry

end total_mixture_price_l147_147244


namespace min_draws_to_ensure_all_colors_l147_147782

-- Define the number of balls of each color
def white_balls : ℕ := 8
def black_balls : ℕ := 9
def yellow_balls : ℕ := 7

-- Total number of draws needed to ensure at least one ball of each color is drawn
def min_draws_to_get_all_colors (total_white total_black total_yellow : ℕ) : ℕ :=
total_white + total_black + 1

theorem min_draws_to_ensure_all_colors :
  ∀ (total_white total_black total_yellow : ℕ), total_white = 8 → total_black = 9 → total_yellow = 7 →
  min_draws_to_get_all_colors total_white total_black total_yellow = 18 :=
by
  intros total_white total_black total_yellow h_white h_black h_yellow
  rw [h_white, h_black, h_yellow]
  unfold min_draws_to_get_all_colors
  norm_num
  sorry

end min_draws_to_ensure_all_colors_l147_147782


namespace marbles_per_box_l147_147910

-- Define the total number of marbles
def total_marbles : Nat := 18

-- Define the number of boxes
def number_of_boxes : Nat := 3

-- Prove there are 6 marbles in each box
theorem marbles_per_box : total_marbles / number_of_boxes = 6 := by
  sorry

end marbles_per_box_l147_147910


namespace A_beats_B_by_20_meters_l147_147790

theorem A_beats_B_by_20_meters :
  (A_distance : ℕ) → (A_time : ℕ) → (B_time : ℕ) → (B_distance : ℕ) →
  A_distance = 100 → A_time = 20 → B_time = 25 →
  B_distance = (100 / 25) * 20 →
  A_distance - B_distance = 20 :=
by
  intros A_distance A_time B_time B_distance h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end A_beats_B_by_20_meters_l147_147790


namespace growth_pattern_equation_l147_147205

theorem growth_pattern_equation (x : ℕ) :
  1 + x + x^2 = 73 :=
sorry

end growth_pattern_equation_l147_147205


namespace prob_A_inter_B_l147_147165

variables {α : Type*} [measurable_space α]
variables (μ : measure_theory.measure α)

def prob_event (p : ℝ) : measure_theory.measure α := p • μ

-- Event definitions
variables (A B : set α)

-- Given conditions
def P_A : measure_theory.measure α := prob_event μ 0.8
def P_B : measure_theory.measure α := prob_event μ 0.55
def P_A'_inter_B' : measure_theory.measure α := prob_event μ 0.2

-- The proof we need to show
theorem prob_A_inter_B :
  μ (A ∩ B) = 0.55 :=
sorry

end prob_A_inter_B_l147_147165


namespace distinct_digit_values_possible_l147_147799

noncomputable def distinct_digits (A B C D : ℕ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧ A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10

theorem distinct_digit_values_possible (A B C D : ℕ) (h : distinct_digits A B C D) (h_eq : A + B = D) : 
  ∃ D, D ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) :=
sorry

end distinct_digit_values_possible_l147_147799


namespace area_of_triangle_ABC_is_250_l147_147800

-- Definitions for the sides and the right angle
def AB : ℝ := 25
def AC : ℝ := 20
def angleA : ℝ := 90

-- Definition of the area of triangle ABC
def area_triangle_ABC : ℝ :=
  (1 / 2) * AB * AC

-- The theorem stating the area of triangle ABC is 250 cm^2
theorem area_of_triangle_ABC_is_250 :
  area_triangle_ABC = 250 := by
  sorry

end area_of_triangle_ABC_is_250_l147_147800


namespace inequality_solution_l147_147335

theorem inequality_solution (m : ℝ) (h : m = 2) : ∀ x : ℝ, 2 * x + m ≤ 0 ↔ x ≤ -1 :=
by
  intro x
  rw [h]
  simp
  exact ⟨λ h => (le_add_iff_nonpos_left 2).mp h, λ h => (le_add_iff_nonpos_left 2).mpr h⟩

end inequality_solution_l147_147335


namespace triangle_area_is_120_l147_147217

-- Define the triangle sides
def a : ℕ := 10
def b : ℕ := 24
def c : ℕ := 26

-- Define a function to calculate the area of a right-angled triangle
noncomputable def right_triangle_area (a b : ℕ) : ℕ := (a * b) / 2

-- Statement to prove the area of the triangle
theorem triangle_area_is_120 : right_triangle_area 10 24 = 120 :=
by
  sorry

end triangle_area_is_120_l147_147217


namespace cycle_of_eleven_l147_147499

-- Define the problem conditions
variables (Spies : Type) [Fintype Spies] [DecidableEq Spies]

def watches (A B : Spies) : Prop := sorry -- Anti-symmetric relation

-- Given there are 16 spies
axiom spies_card : Fintype.card Spies = 16

-- Anti-symmetric relationship formalized
axiom anti_symmetric_watching : ∀ A B : Spies, watches A B → ¬ watches B A

-- Any set of 10 spies can form a directed cycle
axiom cycle_of_ten : ∀ (s : Finset Spies), s.card = 10 → 
  ∃ (f : Fin s → Fin s), Function.Bijective f ∧ ∀ i : Fin s, watches (s.toList.nthLe i.1 i.2) (s.toList.nthLe (f i).1 (f i).2)

-- The main theorem to prove
theorem cycle_of_eleven : ∀ (s : Finset Spies), s.card = 11 → 
  ∃ (f : Fin s → Fin s), Function.Bijective f ∧ ∀ i : Fin s, watches (s.toList.nthLe i.1 i.2) (s.toList.nthLe (f i).1 (f i).2) :=
begin
  sorry
end

end cycle_of_eleven_l147_147499


namespace find_d_l147_147829

theorem find_d (c d : ℝ) (f g : ℝ → ℝ)
  (hf : ∀ x, f x = 5 * x + c)
  (hg : ∀ x, g x = c * x + 3)
  (hfg : ∀ x, f (g x) = 15 * x + d) :
  d = 18 :=
sorry

end find_d_l147_147829


namespace log_expression_simplification_l147_147619

theorem log_expression_simplification :
  (log 96 / log 48 - log 192 / log 24 = 2) :=
by sorry

end log_expression_simplification_l147_147619


namespace bicycle_trip_distance_l147_147186

theorem bicycle_trip_distance
  (D : ℝ) -- Distance traveled in the first part of the trip
  (T1 : ℝ := D / 12) -- Time taken for the first part
  (T2 : ℝ := 12 / 10) -- Time taken for the second part
  (Dtotal : ℝ := D + 12) -- Total distance traveled
  (Ttotal : ℝ := T1 + T2) -- Total time taken
  (Savg : ℝ := 10.82) -- Average speed for the entire trip
  (h : 10.82 = Dtotal / Ttotal) -- Given condition: average speed for the entire trip
  : D ≈ 10 := sorry

end bicycle_trip_distance_l147_147186


namespace cone_base_radius_l147_147338

theorem cone_base_radius (S : ℝ) (hs : S = 15 * real.pi) (unfolds_into_semicircle : ∀ r l, π * l = 2 * π * r → l = 2 * r) : ∃ r : ℝ, r = real.sqrt 5 :=
by
  let r := real.sqrt 5
  have h1 : ∃ r l, l = 2 * r ∧ π * r^2 + π * r * l = 15 * real.pi,
  { use [r, 2 * r],
    split,
    { exact unfolds_into_semicircle r (2 * r) (by ring) },
    { simp only [mul_add, add_mul, pow_two],
      ring_nf,
      exact hs }
    },
  use r,
  exact sqrt_sq (show (0 : ℝ) <= 5, by norm_num),
  sorry

-- Add more details if required

end cone_base_radius_l147_147338


namespace at_least_one_positive_l147_147821

theorem at_least_one_positive (a b c : ℝ) (h : ¬ (a = b ∧ b = c) ) :
  let x := a^2 - b * c,
      y := b^2 - c * a,
      z := c^2 - a * b 
  in x + y + z > 0 → x > 0 ∨ y > 0 ∨ z > 0 :=
by
  sorry

end at_least_one_positive_l147_147821


namespace sequence_constant_l147_147496

theorem sequence_constant
  (a : ℕ → ℤ)
  (d : ℤ)
  (h1 : ∀ n, Nat.Prime (Int.natAbs (a n)))
  (h2 : ∀ n, a (n + 2) = a (n + 1) + a n + d) :
  ∃ c : ℤ, ∀ n, a n = c :=
by
  sorry

end sequence_constant_l147_147496


namespace number_of_valid_pairs_l147_147251

def is_valid_pair (i j : ℕ) : Prop :=
  0 ≤ i ∧ i < j ∧ j ≤ 150 ∧ 6 ∣ (j - i)

def count_valid_pairs : ℕ :=
  (Finset.Icc 0 150).sum (λ j, (Finset.Icc 0 j).count (λ i, is_valid_pair i j))

theorem number_of_valid_pairs : count_valid_pairs = 1825 := by
  -- Omitted proof steps
  sorry

end number_of_valid_pairs_l147_147251


namespace no_convex_pentagon_without_interior_lattice_points_l147_147630

-- Definition of a convex pentagon with lattice points
structure ConvexPentagon :=
(vertices : Fin 5 → (ℤ × ℤ))
(convex : ∀ (i j : Fin 5), ∃ k l, k ≠ l ∧ ¬ is_collinear (vertices i) (vertices j) (vertices k) (vertices l))
(lattice_points : ∀ i, vertices i ∈ set_of (λ p, ∃ x y : ℤ, p = (x, y)))

-- The goal theorem statement
theorem no_convex_pentagon_without_interior_lattice_points (P : ConvexPentagon) :
  ∃ q : ℤ × ℤ, q ∈ convex_hull (set_of (λ p, ∃ i j : Fin 5, q = midpoint (P.vertices i) (P.vertices j))) :=
sorry

end no_convex_pentagon_without_interior_lattice_points_l147_147630


namespace seating_arrangement_l147_147913

theorem seating_arrangement : 
    ∃ x : ℕ, (8 * x + 7 * (13 - x) = 97) ∧ (x = 6) := 
by
    use 6
    split
    · calc
        8 * 6 + 7 * (13 - 6) = 8 * 6 + 7 * 7 : by rw Nat.sub_self_add
                      ... = 48 + 49 : by rw [Nat.mul_comm 8 6, Nat.mul_comm 7 7]
                      ... = 97 : by rfl
    · rfl

end seating_arrangement_l147_147913


namespace stone_width_l147_147195

theorem stone_width (length_hall breadth_hall : ℝ) (num_stones length_stone : ℝ) (total_area_hall total_area_stones area_stone : ℝ)
  (h1 : length_hall = 36) (h2 : breadth_hall = 15) (h3 : num_stones = 5400) (h4 : length_stone = 2) 
  (h5 : total_area_hall = length_hall * breadth_hall * (10 * 10))
  (h6 : total_area_stones = num_stones * area_stone) 
  (h7 : area_stone = length_stone * (5 : ℝ)) 
  (h8 : total_area_stones = total_area_hall) : 
  (5 : ℝ) = 5 :=  
by sorry

end stone_width_l147_147195


namespace simplify_fraction_rationalize_denominator_l147_147490

theorem simplify_fraction_rationalize_denominator :
  (∀ (a b : ℝ), sqrt (a * b) = sqrt a * sqrt b) →
  (sqrt 200 = 10 * sqrt 2) →
  (3 * sqrt 50 = 15 * sqrt 2) →
  (5 / (sqrt 200 + 3 * sqrt 50 + 5) = (5 * sqrt 2 - 1) / 49) :=
by
  intros _ h1 h2
  rw [h1, h2]
  sorry

end simplify_fraction_rationalize_denominator_l147_147490


namespace quadratic_inequality_solution_set_l147_147493

theorem quadratic_inequality_solution_set {x : ℝ} : 
  x^2 < x + 6 ↔ (-2 < x ∧ x < 3) :=
by
  sorry

end quadratic_inequality_solution_set_l147_147493


namespace equal_papers_per_cousin_l147_147562

-- Given conditions
def haley_origami_papers : Float := 48.0
def cousins_count : Float := 6.0

-- Question and expected answer
def papers_per_cousin (total_papers : Float) (cousins : Float) : Float :=
  total_papers / cousins

-- Proof statement asserting the correct answer
theorem equal_papers_per_cousin :
  papers_per_cousin haley_origami_papers cousins_count = 8.0 :=
sorry

end equal_papers_per_cousin_l147_147562


namespace length_of_MQ_of_triangle_MPQ_l147_147429

theorem length_of_MQ_of_triangle_MPQ 
  (DE EF : ℝ)
  (hDE : DE = 5)
  (hEF : EF = 7)
  : ∃ (MQ : ℝ), MQ = 17 :=
by 
  use 17
  sorry

end length_of_MQ_of_triangle_MPQ_l147_147429


namespace area_of_entire_shaded_region_l147_147986

noncomputable def area_of_shaded_region : ℝ :=
  let num_squares := 24
  let num_squares_in_rectangle := 12
  let diagonal_length := 8
  let area_of_rectangle := (diagonal_length ^ 2) / (num_squares_in_rectangle.toReal / 2)
  let area_per_square := area_of_rectangle / num_squares_in_rectangle.toReal
  let total_area := area_per_square * num_squares.toReal
  total_area

theorem area_of_entire_shaded_region :
  area_of_shaded_region = 64 := sorry

end area_of_entire_shaded_region_l147_147986


namespace total_photos_l147_147473

def initial_photos : ℕ := 100
def photos_first_week : ℕ := 50
def photos_second_week : ℕ := 2 * photos_first_week
def photos_third_and_fourth_weeks : ℕ := 80

theorem total_photos (initial_photos photos_first_week photos_second_week photos_third_and_fourth_weeks : ℕ) :
  initial_photos = 100 ∧
  photos_first_week = 50 ∧
  photos_second_week = 2 * photos_first_week ∧
  photos_third_and_fourth_weeks = 80 →
  initial_photos + photos_first_week + photos_second_week + photos_third_and_fourth_weeks = 330 :=
by
  sorry

end total_photos_l147_147473


namespace taco_selling_price_correct_l147_147993

noncomputable def taco_selling_price
  (total_beef : ℝ) -- the total pounds of beef the taco truck buys
  (beef_per_taco : ℝ) -- the amount of beef used per taco in pounds
  (cost_per_taco : ℝ) -- the cost to make each taco in dollars
  (profit : ℝ) -- the total profit in dollars
  : ℝ :=
  let num_tacos := total_beef / beef_per_taco in
  let total_cost := num_tacos * cost_per_taco in
  let total_revenue := total_cost + profit in
  total_revenue / num_tacos

theorem taco_selling_price_correct
  (h1 : total_beef = 100) -- the taco truck buys 100 pounds of beef
  (h2 : beef_per_taco = 0.25) -- they use 0.25 pounds of beef per taco
  (h3 : cost_per_taco = 1.5) -- each taco takes $1.5 to make
  (h4 : profit = 200) -- they made a profit of $200 when they used all the beef
  : taco_selling_price 100 0.25 1.5 200 = 2 :=
by
  sorry

end taco_selling_price_correct_l147_147993


namespace perpendicular_bisector_locus_l147_147843

-- Definitions

def is_center_and_radius (C : Type) (O : C) (r : ℝ) : Prop := sorry -- Circle specific details
def is_on_circle {C : Type} (C : C) (O : C) (r : ℝ) (P : C) : Prop := sorry

def is_perpendicular_bisector {C : Type} (A B D : C) : Prop := 
  ∀ (X : C), dist X B = dist X D ↔ ∃ (M : C), (dist M B = dist M D ∧ dist X B = dist X D)

-- Problem Statement in Lean 4

theorem perpendicular_bisector_locus (C : Type) (O : C) (r : ℝ) (B D : C) :
  is_center_and_radius C O r → is_on_circle C O r B → is_on_circle C O r D →
  ∀ (A : C), (dist A B = dist A D) ↔ is_perpendicular_bisector A B D := sorry

end perpendicular_bisector_locus_l147_147843


namespace students_taking_both_music_and_art_l147_147191

theorem students_taking_both_music_and_art 
  (total_students music_students art_students neither_music_nor_art students_taking_both : ℕ)
  (h1 : total_students = 500)
  (h2 : music_students = 50)
  (h3 : art_students = 20)
  (h4 : neither_music_nor_art = 440)
  (h5 : total_students - neither_music_nor_art = music_students + art_students - students_taking_both) :
  students_taking_both = 10 :=
by
  unfold total_students at h1 h5
  unfold music_students at h2 h5
  unfold art_students at h3 h5
  unfold neither_music_nor_art at h4 h5
  have h6 := h1 - h4
  have h7 := h2 + h3
  have h8 := h7 - h6
  show students_taking_both = 10 from sorry

end students_taking_both_music_and_art_l147_147191


namespace cos_660_degrees_is_one_half_l147_147576

noncomputable def cos_660_eq_one_half : Prop :=
  (Real.cos (660 * Real.pi / 180) = 1 / 2)

theorem cos_660_degrees_is_one_half : cos_660_eq_one_half :=
by
  sorry

end cos_660_degrees_is_one_half_l147_147576


namespace largest_multiple_of_15_less_than_500_l147_147022

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), n < 500 ∧ 15 ∣ n ∧ ∀ (m : ℕ), m < 500 ∧ 15 ∣ m -> m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l147_147022


namespace trapezium_area_l147_147702

theorem trapezium_area (a b h : ℝ) (ha : a = 24) (hb : b = 18) (hh : h = 15) : 
  1/2 * (a + b) * h = 315 ∧ h = 15 :=
by 
  -- The proof steps would go here
  sorry

end trapezium_area_l147_147702


namespace largest_multiple_of_15_less_than_500_l147_147102

theorem largest_multiple_of_15_less_than_500 : ∃ (x : ℕ), (x < 500) ∧ (x % 15 = 0) ∧ ∀ (y : ℕ), (y < 500) ∧ (y % 15 = 0) → y ≤ x :=
begin
  use 495,
  split,
  { norm_num },
  split,
  { norm_num },
  intros y hy1 hy2,
  sorry,
end

end largest_multiple_of_15_less_than_500_l147_147102


namespace largest_multiple_of_15_less_than_500_l147_147011

theorem largest_multiple_of_15_less_than_500 : ∃ n : ℕ, (n > 0 ∧ 15 * n < 500) ∧ (∀ m : ℕ, m > n → 15 * m >= 500) ∧ 15 * n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l147_147011


namespace correct_relation_l147_147156

theorem correct_relation 
  (h1 : ¬ (0 ∈ ∅))
  (h2 : 0 ∈ {0})
  (h3 : ¬ (0 ⊆ {0}))
  (h4 : ¬ ({0} ⊂ ∅)) : 0 ∈ {0} :=
by 
  exact h2

end correct_relation_l147_147156


namespace brownies_per_person_l147_147290

-- Define the conditions as constants
def columns : ℕ := 6
def rows : ℕ := 3
def people : ℕ := 6

-- Define the total number of brownies
def total_brownies : ℕ := columns * rows

-- Define the theorem to be proved
theorem brownies_per_person : total_brownies / people = 3 :=
by sorry

end brownies_per_person_l147_147290


namespace f_prime_0_does_not_exist_l147_147615

noncomputable def f (x : ℝ) : ℝ :=
if x = 0 then 0 else sin (x * sin (3 / x))

theorem f_prime_0_does_not_exist : ∀ (h : ∀ ε > 0, ∃ δ > 0, ∀ Δx (hΔx : 0 < |Δx| ∧ |Δx| < δ),
    |(f (0 + Δx) - f 0) / Δx| < ε), False :=
begin
  sorry
end

end f_prime_0_does_not_exist_l147_147615


namespace sample_size_stratified_sampling_l147_147970

theorem sample_size_stratified_sampling :
  let N_business := 120
  let N_management := 24
  let N_logistics := 16
  let N_total := N_business + N_management + N_logistics
  let n_management_chosen := 3
  let sampling_fraction := n_management_chosen / N_management
  let sample_size := N_total * sampling_fraction
  sample_size = 20 :=
by
  -- Definitions:
  let N_business := 120
  let N_management := 24
  let N_logistics := 16
  let N_total := N_business + N_management + N_logistics
  let n_management_chosen := 3
  let sampling_fraction := n_management_chosen / N_management
  let sample_size := N_total * sampling_fraction
  
  -- Proof:
  sorry

end sample_size_stratified_sampling_l147_147970


namespace ratio_w_y_l147_147894

theorem ratio_w_y 
  (w x y z : ℚ) 
  (h1 : w / x = 4 / 3)
  (h2 : y / z = 3 / 2)
  (h3 : z / x = 1 / 6) : 
  w / y = 16 / 3 :=
sorry

end ratio_w_y_l147_147894


namespace analytical_expression_corrected_range_of_a_plus_2b_l147_147746

open Real

def function_f (x : ℝ) := 2 * sin (4 * x - π / 6)

theorem analytical_expression_corrected {x : ℝ} (h1 : 0 < 2) (h2 : 0 < 4) 
  (h3 : abs (-π / 6) < π / 2) (h4 : x ∈ ℝ) :
  function_f x = 2 * sin (4 * x - π / 6) :=
sorry

theorem range_of_a_plus_2b (A B C a b c : ℝ) (h5 : f (C / 4) = 2)
  (h6 : c = sqrt (3) / 2) (h7 : C = 2 * π / 3) :
  a + 2 * b ∈ Ioo (sqrt (3) / 2) sqrt (3) :=
sorry

end analytical_expression_corrected_range_of_a_plus_2b_l147_147746


namespace color_points_l147_147479

def is_white (p : ℤ × ℤ) : Prop := (p.1 % 2 = 1) ∧ (p.2 % 2 = 1)
def is_black (p : ℤ × ℤ) : Prop := (p.1 % 2 = 0) ∧ (p.2 % 2 = 0)
def is_red (p : ℤ × ℤ) : Prop := (p.1 % 2 = 1 ∧ p.2 % 2 = 0) ∨ (p.1 % 2 = 0 ∧ p.2 % 2 = 1)

theorem color_points :
  (∀ n : ℤ, ∃ (p : ℤ × ℤ), (p.2 = n) ∧ is_white p ∧
                             is_black ⟨p.1, n * 2⟩ ∧
                             is_red ⟨p.1, n * 2 + 1⟩) ∧ 
  (∀ (A B C : ℤ × ℤ), 
    is_white A → is_red B → is_black C → 
    ∃ D : ℤ × ℤ, is_red D ∧ 
    (A.1 + C.1 - B.1 = D.1 ∧
     A.2 + C.2 - B.2 = D.2)) := sorry

end color_points_l147_147479


namespace EllipseHyperbolaSameFociEquationAndArea_l147_147731

theorem EllipseHyperbolaSameFociEquationAndArea
  (h1: ∃ (E: ℝ→ℝ→Prop), ∃ (a b: ℝ), a > b ∧ b > 0 ∧ E = (λ x y, x^2/a^2 + y^2/b^2 = 1) ∧ a^2 - b^2 = 4)
  (h2: ∀ (x y: ℝ), (x^2/3) - y^2 = 1 → ∃ E: ℝ→ℝ→Prop, (E x y))
  (h3: (2, 5/3) ∈ {(x, y) | (x^2/3) - y^2 = 1})
  (h4: ∀ A B C D: ℝ×ℝ, (0, 0) ∈ line_through_origin A B ∧ line_through_origin C D ∧ perpendicular A B C D ∧ (A ∈ E ∧ B ∈ E ∧ C ∈ E ∧ D ∈ E)) :
  ((∃ a b: ℝ, a > b ∧ b > 0 ∧ E = (λ x y, x^2/9 + y^2/5 = 1)) ∧
   (∃ S: set ℝ, S = left_open_interval (90/7) (6*sqrt 5))) := by
  sorry

end EllipseHyperbolaSameFociEquationAndArea_l147_147731


namespace eval_expression_l147_147682

noncomputable def ceil_sqrt_16_div_9 : ℕ := ⌈Real.sqrt (16 / 9 : ℚ)⌉
noncomputable def ceil_16_div_9 : ℕ := ⌈(16 / 9 : ℚ)⌉
noncomputable def ceil_square_16_div_9 : ℕ := ⌈(16 / 9 : ℚ)^2⌉

theorem eval_expression : ceil_sqrt_16_div_9 + ceil_16_div_9 + ceil_square_16_div_9 = 8 :=
by
  -- The following sorry is a placeholder, indicating that the proof is skipped.
  sorry

end eval_expression_l147_147682


namespace largest_multiple_of_15_less_than_500_l147_147132

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), 15 * n < 500 ∧ 15 * n ≥ 15 * (33 - 1) + 1 := by
  sorry

end largest_multiple_of_15_less_than_500_l147_147132


namespace acute_angle_at_3_16_l147_147554

def angle_between_clock_hands (hour minute : ℕ) : ℝ :=
  let minute_angle := (minute / 60) * 360
  let hour_angle := (hour % 12) * 30 + (minute / 60) * 30
  |hour_angle - minute_angle|

theorem acute_angle_at_3_16 : angle_between_clock_hands 3 16 = 2 := 
sorry

end acute_angle_at_3_16_l147_147554


namespace days_before_realization_l147_147973

-- Defining the total work, work rate per person per day, and the days before realization condition
variables (W w : ℝ) (d : ℝ)
# Axiom: Total work done by 10 people before realization
axiom work_by_10_before_realization : 10 * w * d = W / 4
# Axiom: Remaining work done by 8 people after realization in 75 days
axiom work_by_8_after_realization : 8 * w * 75 = 3 * W / 4

-- The Lean theorem we wish to prove:
theorem days_before_realization (W w : ℝ) (d : ℝ) 
  (h1 : 10 * w * d = W / 4) 
  (h2 : 8 * w * 75 = 3 * W / 4) 
: d = 20 := sorry

end days_before_realization_l147_147973


namespace largest_square_with_five_interior_lattice_points_l147_147982

theorem largest_square_with_five_interior_lattice_points :
  ∃ (s : ℝ), (∀ (x y : ℤ), 1 ≤ x ∧ x < s ∧ 1 ≤ y ∧ y < s) → ((⌊s⌋ - 1)^2 = 5) ∧ s^2 = 18 := sorry

end largest_square_with_five_interior_lattice_points_l147_147982


namespace distance_between_points_midpoint_between_points_l147_147163

-- Define the points chosen by Liam and Zara
def liam_point : ℂ := 3 + 4 * Complex.i
def zara_point : ℂ := -2 - 3 * Complex.i

-- Define a theorem to prove the distance between the points
theorem distance_between_points : Complex.abs (liam_point - zara_point) = Real.sqrt 74 := by
  sorry

-- Define a theorem to prove the midpoint between the points
theorem midpoint_between_points : (liam_point + zara_point) / 2 = 0.5 + 0.5 * Complex.i := by
  sorry

end distance_between_points_midpoint_between_points_l147_147163


namespace slope_of_line_l147_147927

theorem slope_of_line : ∀ (x y : ℝ), 4 * y = -6 * x + 12 → ∃ m b : ℝ, y = m * x + b ∧ m = -3 / 2 :=
by 
sorry

end slope_of_line_l147_147927


namespace angles_equality_l147_147802

open EuclideanGeometry

variables {A B C M P N Q : Point}
variables {circumcircle : Circle}
variables (triangle_ABC : Triangle A B C)
variables (midpoint_M : Midpoint B C M)
variables (tangent_P : TangentPoint circumcircle B P)
variables (tangent_P : TangentPoint circumcircle C P)
variables (midpoint_N : Midpoint M P N)
variables (intersection_Q : Intersection (Line A N) circumcircle Q)

theorem angles_equality :
  ∠ P M Q = ∠ M A Q :=
sorry

end angles_equality_l147_147802


namespace find_phi_l147_147774

noncomputable def phi_value (f : ℝ → ℝ) (phi : ℝ) : ℝ :=
  if 0 < phi ∧ phi < π ∧ (∀ x, f (2 * x + phi) = sin (2 * x + phi))
  ∧ (∀ x, f (2 * (π / 3 - x) + phi) = f (2 * (π / 3 + x) + phi))
  then phi else 0

theorem find_phi :
  let f := λ x, sin (2 * x + (5 * π / 6))
  phi := 5 * π / 6
  0 < phi ∧ phi < π ∧ (∀ x, f x = sin (2 * x + phi)) ∧
  (∀ x, f (π / 3 - x) = f (π / 3 + x)) →
  phi_value f phi = 5 * π / 6 :=
by
  intros
  split_ifs
  · rfl
  · exfalso; exact h
  sorry

end find_phi_l147_147774


namespace reggies_brother_long_shots_l147_147409

-- Define the number of points per type of shot
def layup_points : ℕ := 1
def free_throw_points : ℕ := 2
def long_shot_points : ℕ := 3

-- Define the number of shots made by Reggie
def reggie_layups : ℕ := 3
def reggie_free_throws : ℕ := 2
def reggie_long_shots : ℕ := 1

-- Define the total number of points made by Reggie
def reggie_points : ℕ :=
  reggie_layups * layup_points + reggie_free_throws * free_throw_points + reggie_long_shots * long_shot_points

-- Define the total points by which Reggie loses
def points_lost_by : ℕ := 2

-- Prove the number of long shots made by Reggie's brother
theorem reggies_brother_long_shots : 
  (reggie_points + points_lost_by) / long_shot_points = 4 := by
  sorry

end reggies_brother_long_shots_l147_147409


namespace eval_product_eq_1093_l147_147256

noncomputable def z : ℂ := Complex.exp (2 * Real.pi * Complex.I / 7)

theorem eval_product_eq_1093 : (3 - z) * (3 - z^2) * (3 - z^3) * (3 - z^4) * (3 - z^5) * (3 - z^6) = 1093 := by
  sorry

end eval_product_eq_1093_l147_147256


namespace fourth_number_tenth_row_l147_147974

def grid_extend {α : Type} [DecidableLinearOrder α] [AddGroup α] (n m a : α) : α :=
  a + (m - 1) * (n - 1)

theorem fourth_number_tenth_row :
  let first_number_row1 := 1
  let increment := 4
  let numbers_per_row := 6
  let last_number_row9 := grid_extend increment numbers_per_row 321 in
  let first_number_row10 := last_number_row9 + 1 in
  first_number_row10 + 3 * increment = 338 :=
by
  sorry

end fourth_number_tenth_row_l147_147974


namespace convert_to_rectangular_l147_147623

def cylindrical_to_rectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

theorem convert_to_rectangular :
  cylindrical_to_rectangular 10 (Real.pi / 4) 8 = (5 * Real.sqrt 2, 5 * Real.sqrt 2, 8) :=
by
  sorry

end convert_to_rectangular_l147_147623


namespace find_k_and_p_l147_147749

def f (x : ℝ) (k : ℤ) : ℝ := x ^ (-(k^2 : ℝ) + k + 2)
def g (x : ℝ) (p : ℝ) (f : ℝ → ℝ) : ℝ := 1 - p * (f x) + (2*p - 1) * x

theorem find_k_and_p :
  ∀ k : ℤ, (f 2 k < f 3 k) →
    (k = 0 ∨ k = 1) ∧
    ∃ p > 0, (∀ x ∈ Icc (-1 : ℝ) 2, g x p (f x) k ∈ Icc (-4 : ℝ) (17/8)) →
    p = 2 :=
by
  sorry

end find_k_and_p_l147_147749


namespace ratio_AM_MC_l147_147450

noncomputable theory

variables (Γ1 Γ2 : Circle)
variables (O : Point)
variables (A : Point)
variables (B : Point)
variables (AB : Line)
variables (C : Point)
variables (D : Point)
variables (E : Point)
variables (F : Point)
variables (M : Point)

-- Definitions of conditions
def circles_concentric (Γ1 Γ2 : Circle) (O : Point) : Prop :=
  (Γ1.center = O) ∧ (Γ2.center = O)

def point_on_circle (P : Point) (Γ : Circle) : Prop :=
  Γ.contains P

def line_tangent (L : Line) (Γ : Circle) (P : Point) : Prop :=
  Line.tangent_at L Γ P

def second_intersection_point (A B : Point) (Γ : Circle) : Point :=
  Classical.choose (exists_second_intersection_point A B Γ)

def midpoint (A B : Point) : Point :=
  Classical.choose (exists_midpoint A B)

def perpendicular_bisectors_intersection (DE CF : Line) : Point :=
  Classical.choose (exists_perpendicular_bisectors_intersection DE CF)

-- Hypotheses as conditions in Lean
axiom h1 : circles_concentric Γ1 Γ2 O
axiom h2 : point_on_circle A Γ1
axiom h3 : point_on_circle B Γ2
axiom h4 : line_tangent AB Γ2 B
axiom h5 : C = second_intersection_point A B Γ1
axiom h6 : D = midpoint A B
axiom h7 : ∃ line_A : Line, (Line.contains line_A A) ∧
  ∀ (P : Point), (point_on_circle P Γ2 → Line.contains line_A P → P = E ∨ P = F)
axiom h8 : M = perpendicular_bisectors_intersection (line D E) (line C F)
axiom h9 : Line.contains AB M

theorem ratio_AM_MC : (distance A M / distance M C) = 3 / 2 :=
sorry

end ratio_AM_MC_l147_147450


namespace intervals_and_extreme_values_a_eq_1_range_of_a_for_two_zeroes_l147_147308

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  (1 / 3) * x^3 + (1 - a) / 2 * x^2 - a^2 * log x + a^2 * log a

theorem intervals_and_extreme_values_a_eq_1 :
  ∃ (x : ℝ) (f : ℝ → ℝ),
    (f = λ x, (1 / 3) * x^3 - log x) ∧
    (∀ x, 0 < x ∧ x < 1 → f'(x) < 0) ∧
    (∀ x, x > 1 → f'(x) > 0) ∧
    (f 1 = 1 / 3) := sorry

theorem range_of_a_for_two_zeroes :
  ∀ a > 3, ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ a = 0 ∧ f x₂ a = 0 := sorry

end intervals_and_extreme_values_a_eq_1_range_of_a_for_two_zeroes_l147_147308


namespace c_share_l147_147192

theorem c_share (S : ℝ) (b_share_per_rs c_share_per_rs : ℝ)
  (h1 : S = 246)
  (h2 : b_share_per_rs = 0.65)
  (h3 : c_share_per_rs = 0.40) :
  (c_share_per_rs * S) = 98.40 :=
by sorry

end c_share_l147_147192


namespace cube_volume_in_cubic_feet_l147_147950

-- Definitions and Conditions
def length_in_inches : ℝ := 120
def width_in_inches : ℝ := 108
def square_inches_to_square_feet : ℝ := (12 * 12 : ℝ)

-- Equivalent mathematical proof problem
theorem cube_volume_in_cubic_feet
  (length : ℝ) (width : ℝ) (ft_to_inch : ℝ)
  (volume_correct : volume : ℝ := ( (((sqrt ((length * width)/6)) / ft_to_inch ) ^3 ) )
  (h1 : length = 120)
  (h2 : width = 108)
  (h3 : ft_to_inch = 12):
  volume = 58.181818 :=
sorry -- Proof not needed

end cube_volume_in_cubic_feet_l147_147950


namespace largest_multiple_of_15_less_than_500_l147_147074

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), (n < 500) ∧ (15 ∣ n) ∧ (∀ m : ℕ, (m < 500) ∧ (15 ∣ m) → m ≤ n) ∧ n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l147_147074


namespace prob1_prob2_l147_147753

-- Define the polynomial function
def polynomial (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Problem 1: Prove |b| ≤ 1, given conditions
theorem prob1 (a b c : ℝ) (h : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |polynomial a b c x| ≤ 1) : |b| ≤ 1 :=
sorry

-- Problem 2: Find a = 2, given conditions
theorem prob2 (a b c : ℝ) 
  (h1 : polynomial a b c 0 = -1) 
  (h2 : polynomial a b c 1 = 1) 
  (h3 : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |polynomial a b c x| ≤ 1) : 
  a = 2 :=
sorry

end prob1_prob2_l147_147753


namespace xy_equation_solution_l147_147766

theorem xy_equation_solution (x y : ℝ) (h1 : x * y = 10) (h2 : x^2 * y + x * y^2 + x + y = 120) :
  x^2 + y^2 = 11980 / 121 :=
by
  sorry

end xy_equation_solution_l147_147766


namespace find_x_values_l147_147322

noncomputable def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem find_x_values :
  { x : ℕ | combination 10 x = combination 10 (3 * x - 2) } = {1, 3} :=
by
  sorry

end find_x_values_l147_147322


namespace largest_multiple_of_15_less_than_500_l147_147012

theorem largest_multiple_of_15_less_than_500 : ∃ n : ℕ, (n > 0 ∧ 15 * n < 500) ∧ (∀ m : ℕ, m > n → 15 * m >= 500) ∧ 15 * n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l147_147012


namespace area_of_circle_l147_147148

def diameter := 10
def radius := diameter / 2
def area (r : ℝ) := real.pi * r^2
def expected_area := 25 * real.pi

theorem area_of_circle :
    area radius = expected_area := by
  sorry

end area_of_circle_l147_147148


namespace evaluate_ceiling_sum_l147_147640

theorem evaluate_ceiling_sum :
  (⌈Real.sqrt (16 / 9)⌉ : ℤ) + (⌈(16 / 9: ℝ)⌉ : ℤ) + (⌈(16 / 9: ℝ)^2⌉ : ℤ) = 8 := 
by
  -- Placeholder for proof
  sorry

end evaluate_ceiling_sum_l147_147640


namespace evaluate_expression_l147_147666

theorem evaluate_expression :
  let x := (16 : ℚ) / 9
  in ⌈(√x)⌉ + ⌈x⌉ + ⌈x^2⌉ = 8 :=
by
  let x := (16 : ℚ) / 9
  sorry

end evaluate_expression_l147_147666


namespace largest_multiple_of_15_less_than_500_l147_147050

theorem largest_multiple_of_15_less_than_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ (∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x) := sorry

end largest_multiple_of_15_less_than_500_l147_147050


namespace find_d_l147_147826

def f (x : ℝ) (c : ℝ) : ℝ := 5 * x + c
def g (x : ℝ) (c : ℝ) : ℝ := c * x + 3

theorem find_d (c d : ℝ) (h : ∀ x : ℝ, f (g x c) c = 15 * x + d) : d = 18 :=
by
  sorry

end find_d_l147_147826


namespace seating_arrangements_l147_147874

theorem seating_arrangements (family : Fin 5)
  (driver : family → Bool)
  (eldest_child_can_drive : family → Prop)
  (is_eldest_child : family → Prop)
  (is_adult : family → Prop)
  (is_grandmother : family → Prop)
  (H1 : ∃ f, is_eldest_child f ∧ eldest_child_can_drive f)
  (H2 : ∃ f, is_grandmother f ∧ ¬ eldest_child_can_drive f)
  (H3 : ∃ f1 f2, is_adult f1 ∧ is_adult f2 ∧ f1 ≠ f2)
  (H4 : ∀ f, ¬ is_grandmother f → (is_eldest_child f ∨ is_adult f))
  (H5 : (∃ f, driver f ∧ (is_adult f ∨ is_eldest_child f) ∧ ¬ is_grandmother f))
  : 72 = nat.fst (3 * 4 * Real.factorial 3)
:= sorry

end seating_arrangements_l147_147874


namespace largest_multiple_of_15_less_than_500_l147_147061

theorem largest_multiple_of_15_less_than_500 : 
  ∃ n : ℕ, n * 15 < 500 ∧ ∀ m : ℕ, m * 15 < 500 → m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l147_147061


namespace probability_product_less_50_l147_147860

/--
 Paco uses a spinner to select a number from 1 through 5 with equal probability.
 Manu uses a spinner to select a number from 1 through 15 with equal probability.
 Prove that the probability that the product of Manu's number and Paco's number is less than 50
 is equal to 22/25.
-/
theorem probability_product_less_50 :
  let paco_outcomes := Finset.range 5 + 1
  let manu_outcomes := Finset.range 15 + 1
  let num_valid_combinations := paco_outcomes.filter (λ p,
    manu_outcomes.filter (λ m, p * m < 50).card
  ).sum
  let total_combinations := paco_outcomes.card * manu_outcomes.card
  num_valid_combinations / total_combinations = 22 / 25 :=
by
  sorry

end probability_product_less_50_l147_147860


namespace customers_who_left_tip_l147_147219

-- Define the initial number of customers
def initial_customers : ℕ := 39

-- Define the additional number of customers during lunch rush
def additional_customers : ℕ := 12

-- Define the number of customers who didn't leave a tip
def no_tip_customers : ℕ := 49

-- Prove the number of customers who did leave a tip
theorem customers_who_left_tip : (initial_customers + additional_customers) - no_tip_customers = 2 := by
  sorry

end customers_who_left_tip_l147_147219


namespace largest_multiple_of_15_less_than_500_l147_147111

theorem largest_multiple_of_15_less_than_500 :
∀ x : ℕ, (∃ k : ℕ, x = 15 * k ∧ 0 < x ∧ x < 500) → x ≤ 495 :=
begin
  sorry
end

end largest_multiple_of_15_less_than_500_l147_147111


namespace composite_number_of_valid_sequence_l147_147902

-- Define the properties of the sequence and multiples
def is_multiple_of_seventeen_or_twenty_three (n : ℕ) : Prop :=
  n % 17 = 0 ∨ n % 23 = 0

-- Define the condition for the sequence of digits
def valid_sequence (digits : List ℕ) : Prop :=
  digits.length = 2000 ∧
  (∀ i, 1 ≤ i ∧ i < 2000 → is_multiple_of_seventeen_or_twenty_three (10 * digits.nth_le (i - 1) sorry + digits.nth_le i sorry)) ∧
  (1 ∈ digits ∧ 9 ∈ digits ∧ 8 ∈ digits ∧ 7 ∈ digits)

-- Statement to prove that a sequence satisfying the conditions must represent a composite number
theorem composite_number_of_valid_sequence (digits : List ℕ) (h : valid_sequence digits) : ∃ d, d > 1 ∧ (10 ^ 1999 * digits.head!! - some sorry + ... + digits.nth!! 1999) % d = 0 := sorry

end composite_number_of_valid_sequence_l147_147902


namespace probability_of_B_l147_147578

variable (P_A P_A_or_B P_B : ℝ)

noncomputable def probability_B (P_A : ℝ) (P_A_or_B : ℝ) : ℝ :=
  1 - (1 - P_A_or_B) / (1 - P_A)

theorem probability_of_B (h1 : P_A = 0.6) (h2 : P_A_or_B = 0.92) : P_B = 0.8 :=
by
  rw [probability_B]
  have h3 : 0.4 = 1 - P_A := by linarith
  have h4 : 0.08 = 1 - P_A_or_B := by linarith
  rw [← h3, ← h4]
  linarith


end probability_of_B_l147_147578


namespace percentage_error_in_area_l147_147944

theorem percentage_error_in_area (s : ℝ) (h : s > 0) :
  let s' := s * (1 + 0.03)
  let A := s * s
  let A' := s' * s'
  ((A' - A) / A) * 100 = 6.09 :=
by
  sorry

end percentage_error_in_area_l147_147944


namespace profit_percentage_is_25_l147_147613

-- Define the profit calculation and profit percentage calculation
def sp : ℝ := 500 -- Selling price
def cp : ℝ := 400 -- Cost price
def profit (sp cp : ℝ) : ℝ := sp - cp
def profit_percentage (sp cp : ℝ) : ℝ := (profit sp cp / cp) * 100

-- The theorem to prove that the profit percentage is 25%
theorem profit_percentage_is_25 (sp cp : ℝ) (h1 : sp = 500) (h2 : cp = 400) : 
  profit_percentage sp cp = 25 :=
by 
  sorry

end profit_percentage_is_25_l147_147613


namespace red_ball_prob_gt_black_ball_prob_l147_147908

theorem red_ball_prob_gt_black_ball_prob (m : ℕ) (h : 8 > m) : m ≠ 10 :=
by
  sorry

end red_ball_prob_gt_black_ball_prob_l147_147908


namespace polynomial_divisibility_l147_147264

theorem polynomial_divisibility (a b : ℤ) :
  (∀ x : ℤ, x^2 - 1 ∣ x^5 - 3 * x^4 + a * x^3 + b * x^2 - 5 * x - 5) ↔ (a = 4 ∧ b = 8) :=
sorry

end polynomial_divisibility_l147_147264


namespace largest_multiple_of_15_less_than_500_is_495_l147_147122

-- Define the necessary conditions
def is_multiple_of_15 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 15 * k

def is_less_than_500 (n : ℕ) : Prop :=
  n < 500

-- Problem statement: Prove that the largest positive multiple of 15 less than 500 is 495
theorem largest_multiple_of_15_less_than_500_is_495 : 
  ∀ n : ℕ, is_multiple_of_15 n → is_less_than_500 n → n ≤ 495 := 
by
  sorry

end largest_multiple_of_15_less_than_500_is_495_l147_147122


namespace width_calculation_l147_147168

noncomputable def width_of_wall (h l w : ℝ) : ℝ :=
  if (h = 6 * w) ∧ (l = 7 * h) ∧ (l * w * h = 16128)
  then w
  else 0

theorem width_calculation :
  ∃ w : ℝ, (h = 6 * w) ∧ (l = 7 * h) ∧ (l * w * h = 16128) → w = real.cbrt 384 :=
begin
  sorry
end

end width_calculation_l147_147168


namespace find_circle_radius_l147_147292

theorem find_circle_radius (r : ℝ) :
  (∃ M C : ℝ, C = √7 ∧ 
    (∃ internal external : ℝ, internal = r ∧ external = 2 * internal ∧ 
      external * (external + internal) = C^2 - r^2)) → 
  r = 1 :=
by
  sorry

end find_circle_radius_l147_147292


namespace largest_multiple_of_15_less_than_500_l147_147096

theorem largest_multiple_of_15_less_than_500 : ∃ (x : ℕ), (x < 500) ∧ (x % 15 = 0) ∧ ∀ (y : ℕ), (y < 500) ∧ (y % 15 = 0) → y ≤ x :=
begin
  use 495,
  split,
  { norm_num },
  split,
  { norm_num },
  intros y hy1 hy2,
  sorry,
end

end largest_multiple_of_15_less_than_500_l147_147096


namespace intersection_of_A_and_B_l147_147317

def A := {x : ℝ | x^2 - 4 * x - 5 ≤ 0}
def B := {x : ℝ | log 2 x < 2}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 4} :=
sorry

end intersection_of_A_and_B_l147_147317


namespace arithmetic_geometric_sequence_l147_147325

theorem arithmetic_geometric_sequence (a : ℕ → ℤ) (d : ℤ) (S : ℕ → ℤ)
  (h1 : d ≠ 0)
  (h2 : ∀ n, a (n + 1) = a n + d)
  (h3 : a 3 * 2 = a 1 + 2 * d)
  (h4 : a 4 = a 1 + 3 * d)
  (h5 : a 8 = a 1 + 7 * d)
  (h_geo : (a 1 + 3 * d) ^ 2 = (a 1 + 2 * d) * (a 1 + 7 * d))
  (h_sum : S 4 = (a 1 * 4) + (d * (4 * 3 / 2))) :
  a 1 * d < 0 ∧ d * S 4 < 0 :=
by sorry

end arithmetic_geometric_sequence_l147_147325


namespace largest_multiple_of_15_less_than_500_l147_147101

theorem largest_multiple_of_15_less_than_500 : ∃ (x : ℕ), (x < 500) ∧ (x % 15 = 0) ∧ ∀ (y : ℕ), (y < 500) ∧ (y % 15 = 0) → y ≤ x :=
begin
  use 495,
  split,
  { norm_num },
  split,
  { norm_num },
  intros y hy1 hy2,
  sorry,
end

end largest_multiple_of_15_less_than_500_l147_147101


namespace no_fixed_points_implies_no_double_fixed_points_l147_147448

theorem no_fixed_points_implies_no_double_fixed_points (f : ℝ → ℝ) (hf : ∀ x, f x ≠ x) :
  ∀ x, f (f x) ≠ x :=
sorry

end no_fixed_points_implies_no_double_fixed_points_l147_147448


namespace complex_root_circle_radius_l147_147227

noncomputable def radius_of_circle : ℂ → ℂ := sorry

theorem complex_root_circle_radius (z : ℂ) :
  (z + 2)^6 = 64 * z^6 → abs z = 2 / 3 :=
begin
  sorry
end

end complex_root_circle_radius_l147_147227


namespace li_ming_estimated_weight_is_correct_l147_147712

-- Define the regression equation as a function
def regression_equation (x : ℝ) : ℝ := 0.7 * x - 52

-- Define the height of Li Ming
def li_ming_height : ℝ := 180

-- The estimated weight according to the regression equation
def estimated_weight : ℝ := regression_equation li_ming_height

-- Theorem statement: Given the height, the weight should be 74
theorem li_ming_estimated_weight_is_correct : estimated_weight = 74 :=
by
  sorry

end li_ming_estimated_weight_is_correct_l147_147712


namespace fineness_solution_l147_147548

def fineness_problem (x y z : ℝ) :=
(5 * x + 3 * y = 6.08) ∧ 
(5 * x + 2 * z = 5.46) ∧ 
(2 * z + 3 * y = 4.26)

theorem fineness_solution :
  ∃ x y z : ℝ, fineness_problem x y z ∧ x = 0.728 ∧ y = 0.813 ∧ z = 0.91 :=
by {
  use 0.728,
  use 0.813,
  use 0.91,
  show fineness_problem 0.728 0.813 0.91,
  all_goals {
    split,
    trivial,
  },
  sorry
}

end fineness_solution_l147_147548


namespace largest_multiple_of_15_less_than_500_l147_147103

theorem largest_multiple_of_15_less_than_500 : ∃ (x : ℕ), (x < 500) ∧ (x % 15 = 0) ∧ ∀ (y : ℕ), (y < 500) ∧ (y % 15 = 0) → y ≤ x :=
begin
  use 495,
  split,
  { norm_num },
  split,
  { norm_num },
  intros y hy1 hy2,
  sorry,
end

end largest_multiple_of_15_less_than_500_l147_147103


namespace min_price_per_car_to_avoid_losses_min_order_volume_to_compete_l147_147159

-- Conditions
def purchase_price_per_car := 2.5 -- in million rubles
def customs_duties_per_car := 2 -- in million rubles
def monthly_office_rent := 0.05 -- in million rubles
def monthly_salary := 0.37 -- in million rubles
def other_monthly_expenses := 0.18 -- in million rubles
def order_quantity := 30 -- number of cars
def competitor_price_per_car := 5.3 -- in million rubles
def charging_station_value := 0.4 -- in million rubles

-- The resulting problems as Lean 4 statements:

-- Part (a) - Minimum price per car to avoid losses.
theorem min_price_per_car_to_avoid_losses :
  (order_quantity * (purchase_price_per_car + customs_duties_per_car) + 
  monthly_office_rent + monthly_salary + other_monthly_expenses) / order_quantity = 4.52 :=
  sorry

-- Part (b) - Minimum order volume to compete.
theorem min_order_volume_to_compete :
  ∃ (X : ℕ), X >= 2 ∧ 
  (((purchase_price_per_car + customs_duties_per_car) * X + 
  monthly_office_rent + monthly_salary + other_monthly_expenses) / X) ≤ 
  (competitor_price_per_car - charging_station_value) :=
  sorry

end min_price_per_car_to_avoid_losses_min_order_volume_to_compete_l147_147159


namespace largest_multiple_of_15_less_than_500_l147_147138

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), 15 * n < 500 ∧ 15 * n ≥ 15 * (33 - 1) + 1 := by
  sorry

end largest_multiple_of_15_less_than_500_l147_147138


namespace largest_multiple_of_15_less_than_500_l147_147043

theorem largest_multiple_of_15_less_than_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ (∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x) := sorry

end largest_multiple_of_15_less_than_500_l147_147043


namespace largest_prime_factor_of_expr_l147_147269

open Nat

theorem largest_prime_factor_of_expr : 
  let term1 := 20 ^ 3
  let term2 := 15 ^ 4
  let term3 := 10 ^ 5
  let expression := term1 + term2 - term3
  prime 37 ∧ (∀ p : ℕ, prime p → p ∣ expression → p ≤ 37) := 
by
  let term1 := 20 ^ 3
  let term2 := 15 ^ 4
  let term3 := 10 ^ 5
  let expression := term1 + term2 - term3
  have h : expression = 2 * 37 * 5 ^ 3 := sorry
  have prime_37 : prime 37 := sorry
  have largest_prime_factor : (∀ p : ℕ, prime p → p ∣ expression → p ≤ 37) := sorry
  exact ⟨prime_37, largest_prime_factor⟩

end largest_prime_factor_of_expr_l147_147269


namespace aunt_gave_each_l147_147805

def Jade_initial_money : ℕ := 38
def Julia_initial_money : ℕ := Jade_initial_money / 2
def Jack_initial_money : ℕ := 12
def Total_after_gift : ℕ := 132

theorem aunt_gave_each :
  let total_initial := Jade_initial_money + Julia_initial_money + Jack_initial_money in
  let total_gift := Total_after_gift - total_initial in
  total_gift / 3 = 21 :=
by
  let total_initial := Jade_initial_money + Julia_initial_money + Jack_initial_money
  let total_gift := Total_after_gift - total_initial
  have h1 : Julia_initial_money = 19 := by sorry
  have h2 : total_initial = 69 := by sorry
  have h3 : total_gift = 63 := by sorry
  show total_gift / 3 = 21 from by sorry

end aunt_gave_each_l147_147805


namespace otimes_subtraction_l147_147890

def otimes (a b : ℝ) : ℝ := (a ^ 3) / (b ^ 2)

theorem otimes_subtraction :
  (((otimes 2 3) |> (λ x, otimes x 4)) - (otimes 2 ((otimes 3 4)))) = -2016 / 729 :=
by
  sorry

end otimes_subtraction_l147_147890


namespace max_route_length_l147_147588

-- Definitions based on conditions
def intersections : Finset ℕ := (Finset.range 36)
def streets : Finset (ℕ × ℕ) := sorry  -- assuming some predefined set of streets as pairs of intersections
def A : ℕ := 0
def B : ℕ := 35

-- Definition of a valid route from A to B
def valid_route (route:List (ℕ × ℕ)) : Prop :=
  (∀ e ∈ route, e ∈ streets) ∧
  (route.head = (A, _)) ∧
  (route.last = (_, B)) ∧
  (List.nodup (route.map Prod.fst)) ∧ -- No intersection is visited more than once
  (List.nodup (route.map Prod.snd))

-- Definition to count the number of streets in a route
def route_length (route:List (ℕ × ℕ)) : ℕ := route.length

-- Statement to prove
theorem max_route_length (route : List (ℕ × ℕ)) (h : valid_route route) : 
  route_length route ≤ 34 := sorry

end max_route_length_l147_147588


namespace rationalize_expression_l147_147385

theorem rationalize_expression :
  let expr := (Real.sqrt 6 + 1 / Real.sqrt 6 + Real.sqrt 8 + 1 / Real.sqrt 8) in
  ∃ (a b c : ℕ), c > 0 ∧
    c * expr = ↑a * Real.sqrt 6 + ↑b * Real.sqrt 8 ∧
    a + b + c = 106 :=
by
  sorry

end rationalize_expression_l147_147385


namespace largest_multiple_of_15_less_than_500_l147_147078

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), (n < 500) ∧ (15 ∣ n) ∧ (∀ m : ℕ, (m < 500) ∧ (15 ∣ m) → m ≤ n) ∧ n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l147_147078


namespace average_greater_median_l147_147763

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

end average_greater_median_l147_147763


namespace rogers_spending_l147_147486

theorem rogers_spending (B m p : ℝ) (H1 : m = 0.25 * (B - p)) (H2 : p = 0.10 * (B - m)) : 
  m + p = (4 / 13) * B :=
sorry

end rogers_spending_l147_147486


namespace altitude_locus_l147_147708

def in_equilateral_triangle (A B C M : Point) : Prop :=
  -- Define that point M is inside an equilateral triangle ABC
  sorry

def lies_on_altitude (A B C M : Point) : Prop :=
  -- Define that point M lies on one of the altitudes of the equilateral triangle ABC
  sorry

def angles_condition (A B C M : Point) : Prop :=
  -- Define the angles condition: ∠MAB + ∠MBC + ∠MCA = 90°
  sorry

theorem altitude_locus (A B C M : Point) (h_eq : equilateral_triangle A B C) :
  in_equilateral_triangle A B C M →
  angles_condition A B C M →
  lies_on_altitude A B C M :=
sorry

end altitude_locus_l147_147708


namespace rogers_cookie_price_l147_147289

/-- Given:
1. Art's cookies are rectangles.
2. Art bakes exactly 15 cookies.
3. Roger's cookies are squares.
4. Roger bakes exactly 20 cookies.
5. Each friend uses the same amount of dough.
6. Art’s cookies sell for 75 cents each.

Prove: To earn the same amount from a single batch, one of Roger’s cookies should cost 56.25 cents. -/
theorem rogers_cookie_price
  (art_num_cookies : ℕ := 15)
  (roger_num_cookies : ℕ := 20)
  (art_cookie_price : ℕ := 75)
  (equal_dough_usage : ∀ (art_dough roger_dough : ℕ), art_dough = roger_dough) :
  (roger_cookie_price : ℚ) :=
by
  -- The necessary proof steps and details would go here
  sorry

end rogers_cookie_price_l147_147289


namespace car_speed_when_not_serviced_l147_147968

theorem car_speed_when_not_serviced 
  (v_serviced : ℕ) 
  (t_serviced : ℕ) 
  (d_serviced : ℕ) 
  (t_not_serviced : ℕ) 
  (v_not_serviced : ℕ) :
  v_serviced = 90 ∧ t_serviced = 3 ∧ d_serviced = 270 ∧ t_not_serviced = 6 ∧ 
  (d_serviced = v_not_serviced * t_not_serviced) → 
  v_not_serviced = 45 :=
by {
  intros h,
  have hs := h.1,
  have ht := h.2.1,
  have hd := h.2.2.1,
  have ht_not := h.2.2.2.1,
  have heq := h.2.2.2.2,
  rw heq,
  rw ht_not,
  rw ht,
  rw hs,
  exact sorry
}

end car_speed_when_not_serviced_l147_147968


namespace coprime_exponents_iff_l147_147889

theorem coprime_exponents_iff (p q : ℕ) : 
  Nat.gcd (2^p - 1) (2^q - 1) = 1 ↔ Nat.gcd p q = 1 :=
by 
  sorry

end coprime_exponents_iff_l147_147889


namespace largest_multiple_of_15_less_than_500_is_495_l147_147130

-- Define the necessary conditions
def is_multiple_of_15 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 15 * k

def is_less_than_500 (n : ℕ) : Prop :=
  n < 500

-- Problem statement: Prove that the largest positive multiple of 15 less than 500 is 495
theorem largest_multiple_of_15_less_than_500_is_495 : 
  ∀ n : ℕ, is_multiple_of_15 n → is_less_than_500 n → n ≤ 495 := 
by
  sorry

end largest_multiple_of_15_less_than_500_is_495_l147_147130


namespace crayons_count_after_operations_l147_147462

theorem crayons_count_after_operations : 
  let initial_crayons := 18
  let loss_percentage := 50 / 100
  let crayons_lost := initial_crayons * loss_percentage
  let after_loss := initial_crayons - crayons_lost
  let first_purchase := 20
  let after_first_purchase := after_loss + first_purchase
  let contest_win := 15
  let after_contest_win := after_first_purchase + contest_win
  let second_purchase := 25
  let after_second_purchase := after_contest_win + second_purchase
  let brother_take_percentage := 30 / 100
  let brother_takes := Real.floor (after_second_purchase * brother_take_percentage)
  let final_crayons := after_second_purchase - brother_takes
  final_crayons = 49 := by
  -- Definitions instantiation
  let initial_crayons := 18
  let loss_percentage := 50 / 100
  let crayons_lost := initial_crayons * loss_percentage
  let after_loss := initial_crayons - crayons_lost
  let first_purchase := 20
  let after_first_purchase := after_loss + first_purchase
  let contest_win := 15
  let after_contest_win := after_first_purchase + contest_win
  let second_purchase := 25
  let after_second_purchase := after_contest_win + second_purchase
  let brother_take_percentage := 30 / 100
  let brother_takes := Real.floor (after_second_purchase * brother_take_percentage)
  let final_crayons := after_second_purchase - brother_takes
  -- Goal proof skipped
  -- (initial_crayons - initial_crayons * loss_percentage + first_purchase + contest_win + second_purchase - Real.floor ((initial_crayons - initial_crayons * loss_percentage + first_purchase + contest_win + second_purchase) * brother_take_percentage)) = 49
  sorry

end crayons_count_after_operations_l147_147462


namespace total_time_correct_l147_147948

open_locale classical

noncomputable def speed_of_boat : ℝ := 16
noncomputable def speed_of_stream : ℝ := 2
noncomputable def distance_to_place : ℝ := 7200

noncomputable def downstream_speed : ℝ := speed_of_boat + speed_of_stream
noncomputable def upstream_speed : ℝ := speed_of_boat - speed_of_stream

noncomputable def time_downstream : ℝ := distance_to_place / downstream_speed
noncomputable def time_upstream : ℝ := distance_to_place / upstream_speed
noncomputable def total_time : ℝ := time_downstream + time_upstream

theorem total_time_correct : total_time = 914.2857 := 
  by
  -- Here one would provide the proof that total_time equals 914.2857 based on the given conditions.
  sorry

end total_time_correct_l147_147948


namespace largest_multiple_of_15_less_than_500_l147_147107

theorem largest_multiple_of_15_less_than_500 :
∀ x : ℕ, (∃ k : ℕ, x = 15 * k ∧ 0 < x ∧ x < 500) → x ≤ 495 :=
begin
  sorry
end

end largest_multiple_of_15_less_than_500_l147_147107


namespace min_value_f_range_of_a_log_inequality_l147_147723

-- Definition of f(x) = x * ln x
def f (x : ℝ) : ℝ := x * Real.log x

-- Definition of g(x) = -x^2 + a*x - 3
def g (x a : ℝ) : ℝ := -x^2 + a*x - 3

-- Prove the minimum value problem
theorem min_value_f (t : ℝ) (ht : t > 0) : 
  (if 0 < t ∧ t < 1/e then f(1/e) = -1/e else f t = t * Real.log t) := sorry

-- Prove that if 2f(x) >= g(x) for all x > 0, then a <= 4
theorem range_of_a (x a : ℝ) (hx : x > 0) (h : 2 * f x ≥ g x a) : 
  a ≤ 4 := sorry

-- Prove that ln x > 1/(e^x) - 2/(ex) for all x > 0
theorem log_inequality (x : ℝ) (hx : x > 0) : 
  Real.log x > 1/(Real.exp x) - 2/(e * x) := sorry

end min_value_f_range_of_a_log_inequality_l147_147723


namespace volume_of_circumscribed_sphere_l147_147957

theorem volume_of_circumscribed_sphere (vol_cube : ℝ) (h : vol_cube = 8) :
  ∃ (vol_sphere : ℝ), vol_sphere = 4 * Real.sqrt 3 * Real.pi := 
sorry

end volume_of_circumscribed_sphere_l147_147957


namespace geometric_seq_roots_product_l147_147801

theorem geometric_seq_roots_product (α : Type*) [Preorder α] [Mul α] [Pow α ℕ] [Add α]
  (a : ℕ → α) (h_geom : ∃ r > 0, ∀ n, a (n+1) = a n * r)
  (h_roots : ∃ a1 a19 : α, a1 * a19 = 16 ∧ a1 + a19 = 10 ∧ a 1 = a1 ∧ a 19 = a19) :
  a 8 * a 10 * a 12 = 64 :=
by
  sorry

end geometric_seq_roots_product_l147_147801


namespace length_of_LD_l147_147311

theorem length_of_LD 
  (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ)
  (K L : ℝ × ℝ)
  (A B C D : ℝ × ℝ)
  (h_square : A.1 = D.1 ∧ A.2 = B.2 ∧ B.1 = C.1 ∧ C.2 = D.2 ∧ (C.1 - A.1) = a ∧ (B.2 - D.2) = a)
  (h_onCD : L.2 = D.2 ∧ C.1 ≤ L.1 ∧ L.1 ≤ D.1)
  (h_onextDA : K.2 = A.2 ∧ K.1 < A.1)
  (h_angle : (K.1 - B.1) * (L.1 - B.1) + (K.2 - B.2) * (L.2 - B.2) = 0)
  (h_KD : real.sqrt ((K.1 - D.1) ^ 2 + (K.2 - D.2) ^ 2) = 19)
  (h_CL : real.sqrt ((C.1 - L.1) ^ 2 + (C.2 - L.2) ^ 2) = 6) :
  real.sqrt ((L.1 - D.1) ^ 2 + (L.2 - D.2) ^ 2) = 7 :=
begin
  sorry
end

end length_of_LD_l147_147311


namespace friends_games_l147_147435

theorem friends_games (F : ℕ) :
  (81 = F + 22) → (F = 59) :=
by
  intro h
  rw [← h]
  norm_num
  exact eq.refl 59

end friends_games_l147_147435


namespace parabolas_are_equal_l147_147173

-- Definitions from conditions
variables {p₁ p₂ q₁ q₂ t₁ t₂ : ℝ} 
variables {f₁ f₂ : ℝ → ℝ}

-- Definitions of the parabolas
def f1 (x : ℝ) : ℝ := x^2 + p₁ * x + q₁
def f2 (x : ℝ) : ℝ := x^2 + p₂ * x + q₂

-- Non-parallel lines
def l1 (x : ℝ) : ℝ := t₁ * x
def l2 (x : ℝ) : ℝ := t₂ * x

-- The segments cut by these parabolas on these lines are equal
axiom segments_equal_on_lines : 
  (∀ x, t₁ * x = f1 x → ∀ x', t₁ * x = f2 x → x = x') ∧ 
  (∀ x, t₂ * x = f1 x → ∀ x', t₂ * x = f2 x → x = x')

-- Proof goal
theorem parabolas_are_equal : p₁ = p₂ ∧ q₁ = q₂ :=
sorry

end parabolas_are_equal_l147_147173


namespace largest_multiple_of_15_less_than_500_l147_147021

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), n < 500 ∧ 15 ∣ n ∧ ∀ (m : ℕ), m < 500 ∧ 15 ∣ m -> m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l147_147021


namespace book_distribution_l147_147558

theorem book_distribution (x : ℤ) (h : 9 * x + 7 < 11 * x) : 
  ∀ (b : ℤ), (b = 9 * x + 7) → b mod 9 = 7 :=
by
  intro b
  intro hb
  have : b = 9 * x + 7, from hb
  rw [←this]
  sorry

end book_distribution_l147_147558


namespace radius_of_complex_roots_l147_147222

noncomputable def complex_radius (z : ℂ) : ℝ :=
  if (z + 2)^6 = 64 * z^6 then |z + 2| / |z| else 0

theorem radius_of_complex_roots :
  ∀ z : ℂ, (z + 2)^6 = 64 * z^6 → complex_radius z = 2 / Real.sqrt 3 :=
by sorry

end radius_of_complex_roots_l147_147222


namespace find_angle_KMP_l147_147897

noncomputable def triangle_with_incircle (α : ℝ) :=
  ∃ (A B C K P M : ℝ) (incircle_center : ℝ),
    (K ∈ segment B C) ∧
    (P ∈ segment A C) ∧
    (M ∈ segment B A) ∧
    (angle A = 2 * α) ∧
    (∃ (r : ℝ), -- radius of incircle
       tangent A (circle incircle_center r) K ∧
       tangent A (circle incircle_center r) P ∧
       tangent B (circle incircle_center r) K ∧
       tangent B (circle incircle_center r) M ∧
       tangent C (circle incircle_center r) M ∧
       tangent C (circle incircle_center r) P)

theorem find_angle_KMP (α : ℝ) : 
  ∀ A B C K P M incircle_center, 
    triangle_with_incircle α →
    angle K M P = 90 - α :=
by
  intros
  sorry

end find_angle_KMP_l147_147897


namespace derivative_signs_for_negative_x_l147_147330

variables {f g : ℝ → ℝ}

-- Conditions
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def even_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x
def positive_derivative_for_positive_x (f g : ℝ → ℝ) : Prop := 
  ∀ x, x > 0 → (has_deriv_at f x (f x)) ∧ (has_deriv_at g x (g x))

-- The theorem to prove
theorem derivative_signs_for_negative_x (hodd_f : odd_function f) (heven_g : even_function g) (hpos_deriv : positive_derivative_for_positive_x f g) :
  ∀ x, x < 0 → (has_deriv_at f x (f x)) ∧ (has_deriv_at g x (g x)) :=
by
  sorry 

end derivative_signs_for_negative_x_l147_147330


namespace range_of_a_l147_147348

noncomputable def f (a x : ℝ) : ℝ := x^2 - a * x + a + 3
noncomputable def g (a x : ℝ) : ℝ := a * x - 2 * a

theorem range_of_a (a : ℝ) : (∃ x₀ : ℝ, f a x₀ < 0 ∧ g a x₀ < 0) → 7 < a :=
by
  intro h
  sorry

end range_of_a_l147_147348


namespace largest_multiple_of_15_below_500_l147_147034

theorem largest_multiple_of_15_below_500 : ∃ (k : ℕ), 15 * k < 500 ∧ ∀ (m : ℕ), 15 * m < 500 → 15 * m ≤ 15 * k := 
by
  existsi 33
  split
  · norm_num
  · intro m h
    have h1 : m ≤ 33 := Nat.le_of_lt_succ (Nat.div_lt_succ 500 15 m)
    exact (mul_le_mul_left (by norm_num : 0 < 15)).mpr h1
  sorry

end largest_multiple_of_15_below_500_l147_147034


namespace largest_multiple_of_15_less_than_500_l147_147009

theorem largest_multiple_of_15_less_than_500 : ∃ n : ℕ, (n > 0 ∧ 15 * n < 500) ∧ (∀ m : ℕ, m > n → 15 * m >= 500) ∧ 15 * n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l147_147009


namespace budget_percentage_for_genetically_modified_organisms_l147_147584

theorem budget_percentage_for_genetically_modified_organisms
  (microphotonics : ℝ)
  (home_electronics : ℝ)
  (food_additives : ℝ)
  (industrial_lubricants : ℝ)
  (astrophysics_degrees : ℝ) :
  microphotonics = 14 →
  home_electronics = 24 →
  food_additives = 15 →
  industrial_lubricants = 8 →
  astrophysics_degrees = 72 →
  (72 / 360) * 100 = 20 →
  100 - (14 + 24 + 15 + 8 + 20) = 19 :=
  sorry

end budget_percentage_for_genetically_modified_organisms_l147_147584


namespace number_of_people_born_in_country_l147_147436

theorem number_of_people_born_in_country (immigrants new_people : ℕ) (h1 : immigrants = 16320) (h2 : new_people = 106491) : 
  new_people - immigrants = 90171 := 
by 
  rw [h1, h2]
  sorry

end number_of_people_born_in_country_l147_147436


namespace ab_root_of_x6x4x3x2_l147_147372

theorem ab_root_of_x6x4x3x2 (a b : ℂ) (h : a^4 + a^3 - 1 = 0 ∧ b^4 + b^3 - 1 = 0): 
(ab: ℂ) := 
ab^6 + ab^4 + ab^3 - ab^2 - 1 = 0 :=
by
  sorry

end ab_root_of_x6x4x3x2_l147_147372


namespace total_photos_l147_147472

def initial_photos : ℕ := 100
def photos_first_week : ℕ := 50
def photos_second_week : ℕ := 2 * photos_first_week
def photos_third_and_fourth_weeks : ℕ := 80

theorem total_photos (initial_photos photos_first_week photos_second_week photos_third_and_fourth_weeks : ℕ) :
  initial_photos = 100 ∧
  photos_first_week = 50 ∧
  photos_second_week = 2 * photos_first_week ∧
  photos_third_and_fourth_weeks = 80 →
  initial_photos + photos_first_week + photos_second_week + photos_third_and_fourth_weeks = 330 :=
by
  sorry

end total_photos_l147_147472


namespace interest_difference_l147_147951

noncomputable def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * r * t

noncomputable def compound_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ := 
  P * (1 + r)^t - P

theorem interest_difference : 
  simple_interest 500 0.20 2 - (500 * (1 + 0.20)^2 - 500) = 20 := by
  sorry

end interest_difference_l147_147951


namespace proof_problem_l147_147674

def sqrt_frac : ℚ := real.sqrt (16 / 9)
def frac : ℚ := 16 / 9
def square_frac : ℚ := frac * frac

def ceil_sqrt_frac : ℤ := ⌈sqrt_frac⌉.to_int
def ceil_frac : ℤ := ⌈frac⌉.to_int
def ceil_square_frac : ℤ := ⌈square_frac⌉.to_int

theorem proof_problem :
  ceil_sqrt_frac + ceil_frac + ceil_square_frac = 8 :=
by
  -- Placeholder for the actual proof.
  sorry

end proof_problem_l147_147674


namespace larger_solid_volume_l147_147724

theorem larger_solid_volume (s : ℝ) (h1 : s > 0) :
  let D := (0, 0, 0)
  let M := (s / 2, s, 0)
  let N := (s, 0, s / 2)
  -- Assume plane cuts through D, M, and N
  -- The volume of the larger solid section after division by the plane is given by:
  (s^3 - (7 * s^3 / 48)) = (41 * s^3 / 48) :=
begin
  -- Let the rest be deferred.
  sorry
end

end larger_solid_volume_l147_147724


namespace problem_l147_147716

-- Defining the given polynomial
def polynomial (x : ℂ) : ℂ := (1 - 2*x)^7

-- Defining the coefficients
def a (n : ℕ) : ℂ := (polynomial n).coeff n

-- Main theorem statement
theorem problem (f : ℂ → ℂ) : 
  ∃ (a : ℕ → ℂ), (f = polynomial) ∧
  (a 0 = 1) ∧
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = -2) ∧
  (a 1 + a 3 + a 5 + a 7 = -1094) ∧
  (|a 0| + |a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| + |a 7| = 2187) :=
begin
  sorry
end

end problem_l147_147716


namespace ARML_Super_Relay_Final_Sum_l147_147533

-- Definitions based on the problem conditions
def T : ℕ := sorry -- Define T, which is TNYWR
def A : ℕ := sorry -- Answer to problem 8
def B : ℕ := sorry -- Answer to problem 9
def C : ℕ := sorry -- Answer to problem 10

-- Final proof statement
theorem ARML_Super_Relay_Final_Sum :
  -- Sum of all answers = 6286
  let sum := 
    (binom T 6) +         -- Problem 2
    (if T % 2 = 0 then T/2 else T/factorial.min_factor) / 2 +  -- Problem 3
    (binomial_probability_T_heads 4) +     -- Problem 4
    (last_digit_of_power T T) +            -- Problem 5
    (binomial_probability_T_heads 6) +     -- Problem 6
    (smallest_prime_p_for_Fermat T) +      -- Problem 7
    (quadruple_MN_int_solutions M N) +     -- Problem 8
    (unique_coprime_n T) +                 -- Problem 9
    (binomial_probability_T_heads 10) +    -- Problem 10
    (last_digit_of_power T T) +            -- Problem 11
    (binomial_probability_T_heads 12) +    -- Problem 12
    (if T % 2 = 0 then T/2 else T/factorial.min_factor) / 2 +  -- Problem 13
    (binom T 6) +         -- Problem 14
    (gcd A C * B)         -- Problem 15
  in
  sum = 6286 := sorry

end ARML_Super_Relay_Final_Sum_l147_147533


namespace largest_multiple_of_15_less_than_500_is_495_l147_147123

-- Define the necessary conditions
def is_multiple_of_15 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 15 * k

def is_less_than_500 (n : ℕ) : Prop :=
  n < 500

-- Problem statement: Prove that the largest positive multiple of 15 less than 500 is 495
theorem largest_multiple_of_15_less_than_500_is_495 : 
  ∀ n : ℕ, is_multiple_of_15 n → is_less_than_500 n → n ≤ 495 := 
by
  sorry

end largest_multiple_of_15_less_than_500_is_495_l147_147123


namespace sequence_is_positive_integer_l147_147760

noncomputable def a_seq : ℕ → ℕ
| 0     := 0   -- Unused, to adhere to 1-based indexing in sequence
| 1     := 1
| 2     := 1
| (n+3) := (a_seq (n+2)) ^ 2 + (-1) ^ n / (a_seq (n+1))

theorem sequence_is_positive_integer (n : ℕ) (hn : 0 < n):
  ∃ k : ℕ , k = a_seq n :=
by sorry

end sequence_is_positive_integer_l147_147760


namespace unobstructed_line_of_sight_l147_147342

theorem unobstructed_line_of_sight (a : ℝ) :
  let C (x : ℝ) := 2 * x ^ 2 in
  let A := (0, -2) in
  let B := (3, a) in
  a < 10 :=
sorry

end unobstructed_line_of_sight_l147_147342


namespace set_star_assoc_l147_147287

variables {α : Type*} (A B : set α)

def set_star (X Y : set α) : set α := {x | x ∈ X ∧ x ∉ Y}

theorem set_star_assoc : set_star A (set_star A B) = A ∩ B :=
by {
  sorry
}

end set_star_assoc_l147_147287


namespace two_digit_number_as_expression_l147_147218

-- Define the conditions of the problem
variables (a : ℕ)

-- Statement to be proved
theorem two_digit_number_as_expression (h : 0 ≤ a ∧ a ≤ 9) : 10 * a + 1 = 10 * a + 1 := by
  sorry

end two_digit_number_as_expression_l147_147218


namespace roots_in_interval_l147_147767

theorem roots_in_interval (a : ℝ) (h : a > 3) :
  ∃! x : ℝ, x ∈ Ioo 0 2 ∧ x^3 - a * x^2 + 1 = 0 :=
sorry

end roots_in_interval_l147_147767


namespace gcd_102_238_l147_147511

theorem gcd_102_238 : Int.gcd 102 238 = 34 :=
by
  sorry

end gcd_102_238_l147_147511


namespace evaluate_expression_l147_147656

theorem evaluate_expression : 
  (⌈Real.sqrt (16 / 9)⌉ + ⌈ (16 / 9 : ℝ ) ⌉ + ⌈Real.pow (16 / 9 : ℝ ) 2⌉) = 8 := 
by 
  sorry

end evaluate_expression_l147_147656


namespace radius_of_circle_complex_roots_l147_147230

noncomputable def radius_of_complex_roots_circle : ℝ :=
  let z := ℂ in 
  if h : (∀ z : ℂ, (z + 2)^6 = 64 * (z)^6 → ∥z + 2∥ = 2 * ∥z∥) then
    (4 / 3)
  else
    0  -- placeholder, actual proof logic not required

theorem radius_of_circle_complex_roots :
  (∀ z : ℂ, (z + 2)^6 = 64 * (z)^6 → ∥z + 2∥ = 2 * ∥z∥) → radius_of_complex_roots_circle = (4 / 3) :=
by sorry

end radius_of_circle_complex_roots_l147_147230


namespace boat_width_l147_147601

-- Definitions: river width, number of boats, and space between/banks
def river_width : ℝ := 42
def num_boats : ℕ := 8
def space_between : ℝ := 2

-- Prove the width of each boat given the conditions
theorem boat_width : 
  ∃ w : ℝ, 
    8 * w + 7 * space_between + 2 * space_between = river_width ∧
    w = 3 :=
by
  sorry

end boat_width_l147_147601


namespace largest_multiple_of_15_less_than_500_l147_147104

theorem largest_multiple_of_15_less_than_500 : ∃ (x : ℕ), (x < 500) ∧ (x % 15 = 0) ∧ ∀ (y : ℕ), (y < 500) ∧ (y % 15 = 0) → y ≤ x :=
begin
  use 495,
  split,
  { norm_num },
  split,
  { norm_num },
  intros y hy1 hy2,
  sorry,
end

end largest_multiple_of_15_less_than_500_l147_147104


namespace largest_multiple_of_15_less_than_500_l147_147064

theorem largest_multiple_of_15_less_than_500 : 
  ∃ n : ℕ, n * 15 < 500 ∧ ∀ m : ℕ, m * 15 < 500 → m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l147_147064


namespace largest_multiple_15_under_500_l147_147086

-- Define the problem statement
theorem largest_multiple_15_under_500 : ∃ (n : ℕ), n < 500 ∧ n % 15 = 0 ∧ (∀ m, m < 500 ∧ m % 15 = 0 → m ≤ n) :=
by {
  use 495,
  split,
  {
    exact lt_of_le_of_ne (le_refl _) (ne_of_lt (nat.lt_succ_self 499))
  },
  split,
  {
    exact nat.mod_eq_zero_of_dvd ⟨33, rfl⟩,
  },
  {
    intros m hm h,
    have hmn : m ≤ 495,
    {
      have div15 := hm/15,
      suffices : div15 < 33+1, from nat.le_of_lt_add_one this,
      exact this, sorry,
    },
    exact hmn,
  },
}

end largest_multiple_15_under_500_l147_147086


namespace isosceles_triangle_area_l147_147416

noncomputable def point := ℝ × ℝ

noncomputable def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

variable {P Q R : point}

theorem isosceles_triangle_area (h_iso : distance P Q = distance P R)
  (h_PQ : distance P Q = 13) (h_QR : distance Q R = 10) : 
  (1 / 2) * 10 * real.sqrt (13 ^ 2 - (10 / 2) ^ 2) = 60 :=
by
  sorry

end isosceles_triangle_area_l147_147416


namespace intersection_of_A_and_B_l147_147318

def A := {x : ℝ | x^2 - 4 * x - 5 ≤ 0}
def B := {x : ℝ | log 2 x < 2}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 4} :=
sorry

end intersection_of_A_and_B_l147_147318


namespace largest_multiple_of_15_less_than_500_l147_147006

theorem largest_multiple_of_15_less_than_500 : ∃ n : ℕ, (n > 0 ∧ 15 * n < 500) ∧ (∀ m : ℕ, m > n → 15 * m >= 500) ∧ 15 * n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l147_147006


namespace palmer_total_photos_l147_147477

theorem palmer_total_photos (initial_photos : ℕ) (first_week_photos : ℕ) (third_fourth_weeks_photos : ℕ) :
  (initial_photos = 100) →
  (first_week_photos = 50) →
  (third_fourth_weeks_photos = 80) →
  let second_week_photos := 2 * first_week_photos in
  let total_bali_photos := first_week_photos + second_week_photos + third_fourth_weeks_photos in
  let total_photos := initial_photos + total_bali_photos in
  total_photos = 330 :=
by
  intros h_initial h_first_week h_third_fourth_weeks
  let second_week_photos := 2 * first_week_photos
  let total_bali_photos := first_week_photos + second_week_photos + third_fourth_weeks_photos
  let total_photos := initial_photos + total_bali_photos
  sorry

end palmer_total_photos_l147_147477


namespace geometric_sequence_a3_value_l147_147337

theorem geometric_sequence_a3_value :
  ∀ (x : ℝ), let S : ℕ → ℝ := λ n, (x^2 + 3 * x) * 2^n - x + 1 in
  let a : ℕ → ℝ := λ n, match n with
    | 0 => S 0
    | 1 => S 1
    | n + 1 => S n.succ - S n 
  in a 3 = -8 :=
by
  intros
  let S := λ n, (x^2 + 3 * x) * 2^n - x + 1
  let a := λ n, match n with
    | 0 => S 0
    | 1 => S 1
    | n + 1 => S (n + 1) - S n 
  show a 3 = -8
  sorry

end geometric_sequence_a3_value_l147_147337


namespace product_of_consecutive_nat_is_divisible_by_2_l147_147864

theorem product_of_consecutive_nat_is_divisible_by_2 (n : ℕ) : 2 ∣ n * (n + 1) :=
sorry

end product_of_consecutive_nat_is_divisible_by_2_l147_147864


namespace terminal_side_point_sin_value_l147_147734

theorem terminal_side_point_sin_value (y : ℝ) (β : ℝ)
  (hP : ∃ β, P = (-√3, y) ∧ sin β = (√13)/13) :
  y = 1/2 :=
by
  sorry

end terminal_side_point_sin_value_l147_147734


namespace largest_multiple_of_15_less_than_500_l147_147004

theorem largest_multiple_of_15_less_than_500 : ∃ n : ℕ, (n > 0 ∧ 15 * n < 500) ∧ (∀ m : ℕ, m > n → 15 * m >= 500) ∧ 15 * n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l147_147004


namespace derivative_of_y_l147_147707

noncomputable def y (x : ℝ) : ℝ :=
  3 * Real.arcsin (3 / (4 * x + 1)) + 2 * Real.sqrt (4 * x^2 + 2 * x - 2)

def deriv_y (x : ℝ) : ℝ :=
  (7 * (4 * x + 1)) / (2 * Real.sqrt (4 * x^2 + 2 * x - 2))

theorem derivative_of_y (x : ℝ) (h : 4 * x + 1 > 0) : 
  (derivative y x) = deriv_y x := 
sorry

end derivative_of_y_l147_147707


namespace sin_values_l147_147324

theorem sin_values (x : ℝ) (h : Real.sec x + Real.tan x = 5/3) : 
  Real.sin x = 8/17 ∨ Real.sin x = -1/17 := 
by
  sorry

end sin_values_l147_147324


namespace curve_is_line_l147_147705

-- Let theta be the angle in polar coordinates
variable {θ : ℝ}

-- Define the condition given in the problem
def condition : Prop := θ = π / 4

-- Prove that the curve defined by this condition is a line
theorem curve_is_line (h : condition) : ∃ (m b : ℝ), ∀ x y : ℝ, y = m * x + b :=
by
  sorry

end curve_is_line_l147_147705


namespace range_of_m_for_single_root_l147_147380

theorem range_of_m_for_single_root (f : ℝ → ℝ) (m : ℝ) :
  (∀ x, f x = m * x ^ 2 + 3 * x - m - 2) →
  (∃! x ∈ set.Ioo (0 : ℝ) 1, f x = 0) ↔ m > -2 :=
by
  sorry

end range_of_m_for_single_root_l147_147380


namespace cakes_difference_l147_147237

-- Definitions of the given conditions
def cakes_sold : ℕ := 78
def cakes_bought : ℕ := 31

-- The theorem to prove
theorem cakes_difference : cakes_sold - cakes_bought = 47 :=
by sorry

end cakes_difference_l147_147237


namespace total_pieces_four_row_triangle_l147_147793

def rods_in_row (n : ℕ) : ℕ :=
  3 * n

def connectors_in_row (n : ℕ) : ℕ :=
  2 * n

def total_rods (n : ℕ) : ℕ :=
  ∑ i in range n, rods_in_row (i + 1)

def total_connectors (n : ℕ) : ℕ :=
  ∑ i in range (n + 1), connectors_in_row (i + 1)

theorem total_pieces_four_row_triangle : total_rods 4 + total_connectors 4 = 60 :=
by
  sorry

end total_pieces_four_row_triangle_l147_147793


namespace displacement_after_moves_l147_147789

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem displacement_after_moves :
  let steps_for_primes := 2
  let steps_for_composites := 3
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
  let composites := (List.range 29).tail.filter (λ n, ¬is_prime (n + 2))
  let total_steps_forward := (primes.length * steps_for_primes)
  let total_steps_backward := (composites.length * steps_for_composites)
  let net_displacement := total_steps_forward - total_steps_backward
  in 
  abs(net_displacement) = 37 :=
by
  sorry

end displacement_after_moves_l147_147789


namespace largest_prime_factor_l147_147275

theorem largest_prime_factor (a b c : ℕ) (h1 : a = 20) (h2 : b = 15) (h3 : c = 10) :
  Nat.gcd (a * a * a) (b * b * b * b) = 1 ∧ Nat.gcd (a * a * a) (c * c * c * c * c) = 1 ∧ Nat.gcd (b * b * b * b) (c * c * c * c * c) = 1 →
  Nat.largest_prime_factor (a ^ 3 + b ^ 4 - c ^ 5) = 13 :=
by
  sorry

end largest_prime_factor_l147_147275


namespace find_A_plus_B_l147_147629

theorem find_A_plus_B (A B : ℚ) (h : ∀ x : ℚ, (Bx - 17) / (x^2 - 7x + 12) = A / (x - 3) + 4 / (x - 4)) :
  A = 5/4 ∧ B = 21/4 → A + B = 13/2 :=
by
  intro h
  sorry

end find_A_plus_B_l147_147629


namespace cube_cross_section_min_area_l147_147587

noncomputable def minimum_cross_section_area (a : ℝ) : ℝ :=
  (sqrt 6 * a^2) / 2

theorem cube_cross_section_min_area (a : ℝ) :
  ∃ (S : ℝ), S = minimum_cross_section_area a :=
by
  use (sqrt 6 * a^2) / 2
  sorry

end cube_cross_section_min_area_l147_147587


namespace largest_multiple_of_15_less_than_500_l147_147065

theorem largest_multiple_of_15_less_than_500 : 
  ∃ n : ℕ, n * 15 < 500 ∧ ∀ m : ℕ, m * 15 < 500 → m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l147_147065


namespace lisa_hotdog_record_l147_147852

theorem lisa_hotdog_record
  (hotdogs_eaten : ℕ)
  (eaten_in_first_half : ℕ)
  (rate_per_minute : ℕ)
  (time_in_minutes : ℕ)
  (first_half_duration : ℕ)
  (remaining_time : ℕ) :
  eaten_in_first_half = 20 →
  rate_per_minute = 11 →
  first_half_duration = 5 →
  remaining_time = 5 →
  time_in_minutes = first_half_duration + remaining_time →
  hotdogs_eaten = eaten_in_first_half + rate_per_minute * remaining_time →
  hotdogs_eaten = 75 := by
  intros
  sorry

end lisa_hotdog_record_l147_147852


namespace largest_multiple_of_15_less_than_500_l147_147051

theorem largest_multiple_of_15_less_than_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ (∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x) := sorry

end largest_multiple_of_15_less_than_500_l147_147051


namespace correct_money_calculation_l147_147953

structure BootSale :=
(initial_money : ℕ)
(price_per_boot : ℕ)
(total_taken : ℕ)
(total_returned : ℕ)
(money_spent : ℕ)
(remaining_money_to_return : ℕ)

theorem correct_money_calculation (bs : BootSale) :
  bs.initial_money = 25 →
  bs.price_per_boot = 12 →
  bs.total_taken = 25 →
  bs.total_returned = 5 →
  bs.money_spent = 3 →
  bs.remaining_money_to_return = 2 →
  bs.total_taken - bs.total_returned + bs.money_spent = 23 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end correct_money_calculation_l147_147953


namespace problem_F_distinct_mod_p_l147_147813

open BigOperators

-- Defining the problem
theorem problem_F_distinct_mod_p (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∀ (a b : ℕ), a ∈ finset.range p → b ∈ finset.range p → a ≠ b →
  (∑ k in finset.range (p-1), (k+1) * a^k) % p ≠ (∑ k in finset.range (p-1), (k+1) * b^k) % p :=
by
  intros a b ha hb hab
  sorry

end problem_F_distinct_mod_p_l147_147813


namespace largest_multiple_of_15_less_than_500_l147_147133

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), 15 * n < 500 ∧ 15 * n ≥ 15 * (33 - 1) + 1 := by
  sorry

end largest_multiple_of_15_less_than_500_l147_147133


namespace edward_toy_cars_l147_147633

def initial_amount : ℝ := 17.80
def cost_per_car : ℝ := 0.95
def cost_of_race_track : ℝ := 6.00
def remaining_amount : ℝ := 8.00

theorem edward_toy_cars : ∃ (n : ℕ), initial_amount - remaining_amount = n * cost_per_car + cost_of_race_track ∧ n = 4 := by
  sorry

end edward_toy_cars_l147_147633


namespace find_added_water_l147_147788

theorem find_added_water 
  (initial_vol : ℝ)
  (milk_ratio : ℝ)
  (water_ratio : ℝ)
  (evaporated_water : ℝ)
  (final_milk_ratio : ℝ)
  (final_water_ratio : ℝ)
  : (initial_vol = 80) → 
    (milk_ratio = 7) → 
    (water_ratio = 3) → 
    (evaporated_water = 8) →
    (final_milk_ratio = 5) →
    (final_water_ratio = 4) →
    let initial_milk := (milk_ratio / (milk_ratio + water_ratio)) * initial_vol in
    let initial_water := (water_ratio / (milk_ratio + water_ratio)) * initial_vol in
    let remaining_water := initial_water - evaporated_water in
    let x := ((final_water_ratio / final_milk_ratio) * initial_milk) - remaining_water
    in x = 28.8 :=
by
  sorry

end find_added_water_l147_147788


namespace largest_multiple_of_15_less_than_500_l147_147059

theorem largest_multiple_of_15_less_than_500 : 
  ∃ n : ℕ, n * 15 < 500 ∧ ∀ m : ℕ, m * 15 < 500 → m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l147_147059


namespace eval_expression_l147_147680

noncomputable def ceil_sqrt_16_div_9 : ℕ := ⌈Real.sqrt (16 / 9 : ℚ)⌉
noncomputable def ceil_16_div_9 : ℕ := ⌈(16 / 9 : ℚ)⌉
noncomputable def ceil_square_16_div_9 : ℕ := ⌈(16 / 9 : ℚ)^2⌉

theorem eval_expression : ceil_sqrt_16_div_9 + ceil_16_div_9 + ceil_square_16_div_9 = 8 :=
by
  -- The following sorry is a placeholder, indicating that the proof is skipped.
  sorry

end eval_expression_l147_147680


namespace fraction_simplification_l147_147620

-- Define the numerator and denominator based on given conditions
def numerator : ℤ := 1 - 2 + 4 - 8 + 16 - 32 + 64 - 128 + 256
def denominator : ℤ := 2 - 4 + 8 - 16 + 32 - 64 + 128 - 256 + 512

-- Lean theorem that encapsulates the problem
theorem fraction_simplification : (numerator : ℚ) / (denominator : ℚ) = 1 / 2 :=
by
  sorry

end fraction_simplification_l147_147620


namespace maximize_car_travel_l147_147297

theorem maximize_car_travel (front_tires : ℕ) (rear_tires : ℕ) (x : ℕ)
  (h1 : front_tires = 42000) (h2 : rear_tires = 56000) : 
  let total_distance := min (x + (56000 - x)) (x + (42000 - x))
  in total_distance = 42000 :=
by
  sorry

end maximize_car_travel_l147_147297


namespace ceiling_sum_evaluation_l147_147690

noncomputable def evaluateCeilingSum : ℝ := 
  ⌈Real.sqrt (16 / 9)⌉ + ⌈(16 / 9)⌉ + ⌈((16 / 9) ^ 2)⌉ 

theorem ceiling_sum_evaluation : evaluateCeilingSum = 8 := by
  sorry

end ceiling_sum_evaluation_l147_147690


namespace ceiling_sum_l147_147648

theorem ceiling_sum :
  let a := 4 / 3
  let b := 16 / 9
  let c := 256 / 81
  ⌈a⌉ + ⌈b⌉ + ⌈c⌉ = 8 := by
  sorry

end ceiling_sum_l147_147648


namespace ratio_of_squares_l147_147791

noncomputable def right_triangle : Type := sorry -- Placeholder for the right triangle type

variables (a b c : ℕ)

-- Given lengths of the triangle sides
def triangle_sides (a b c : ℕ) : Prop :=
  a = 5 ∧ b = 12 ∧ c = 13 ∧ a^2 + b^2 = c^2

-- Define x and y based on the conditions in the problem
def side_length_square_x (x : ℝ) : Prop :=
  0 < x ∧ x < 5 ∧ x < 12

def side_length_square_y (y : ℝ) : Prop :=
  0 < y ∧ y < 13

-- The main theorem to prove
theorem ratio_of_squares (x y : ℝ) :
  ∀ a b c, triangle_sides a b c →
  side_length_square_x x →
  side_length_square_y y →
  x / y = 1 :=
sorry

end ratio_of_squares_l147_147791


namespace production_time_l147_147911

-- Define the rates for machines A, B, and C
def rateA := 1 / 6
def rateB := 1 / 7
def rateC := 1 / 9

-- Define the combined rate
def combined_rate := rateA + rateB + rateC

-- Define the time it takes for all machines to complete the job together
def time := 1 / combined_rate

-- Prove that the time is approximately 2.377 hours
theorem production_time : time ≈  2.377 := by
  unfold time combined_rate rateA rateB rateC
  norm_num
  sorry

end production_time_l147_147911


namespace maximum_value_l147_147437

theorem maximum_value (R P K : ℝ) (h₁ : 3 * Real.sqrt 3 * R ≥ P) (h₂ : K = P * R / 4) : 
  (K * P) / (R^3) ≤ 27 / 4 :=
by
  sorry

end maximum_value_l147_147437


namespace number_of_irreducible_fractions_l147_147265

open BigOperators

/-- Definition of the product from 2 to 10 as a natural number. -/
def N : ℕ := ∏ i in range (2, 11), i

/-- Prime factorization of the product from 2 to 10. -/
def prime_factors : List (ℕ × ℕ) :=
  [(2, 8), (3, 4), (5, 2), (7, 1)]

/-- Statement of the problem: The number of irreducible fractions such that the product of the
numerator and denominator is equal to the product of the numbers from 2 to 10 is 18. -/
theorem number_of_irreducible_fractions : 
  ∃ n : ℕ, n = 18 ∧ 
  ∀ (p q : ℕ), p * q = N → Nat.gcd p q = 1 → n = 18 :=
sorry

end number_of_irreducible_fractions_l147_147265


namespace find_d_l147_147822

-- Definitions of the functions f and g and condition on f(g(x))
def f (x : ℝ) (c : ℝ) : ℝ := 5 * x + c
def g (x : ℝ) (c : ℝ) : ℝ := c * x + 3

theorem find_d (c d x : ℝ) (h : f (g x c) c = 15 * x + d) : d = 18 :=
sorry

end find_d_l147_147822


namespace distance_fall_free_fall_l147_147761

noncomputable def distance_travelled (g t : ℝ) : ℝ := ∫ τ in 0..t, g * τ

theorem distance_fall_free_fall (g t₀ : ℝ) :
  distance_travelled g t₀ = (1 / 2) * g * t₀^2 :=
by
  sorry

end distance_fall_free_fall_l147_147761


namespace binary_to_decimal_conversion_l147_147622

def binary_to_decimal (binary : List Nat) : Nat :=
  binary.foldr (λ (bit power acc), acc + bit * 2 ^ power) 0 ∘ List.zip (List.range (List.length binary))

-- Given conditions
def binary_number : List Nat := [1, 1, 0, 0, 1, 1]
def possible_answer : Nat := 51

theorem binary_to_decimal_conversion :
  binary_to_decimal binary_number = possible_answer :=
by
  sorry

end binary_to_decimal_conversion_l147_147622


namespace reciprocal_key_problem_l147_147459

theorem reciprocal_key_problem :
  let f : ℝ → ℝ := λ x, 1 / x
  in (f^[2]) 50 = 50 ∧ (∀ n : ℕ, 0 < n → (n < 2 → (f^[n]) 50 ≠ 50)) :=
begin
  sorry,
end

end reciprocal_key_problem_l147_147459


namespace distinct_lines_l147_147530

-- Define the conditions as points and lines
axiom points : Type
axiom A B C D : points
axiom line : points → points → Type
axiom outside (p : points) (l : line A B) : Prop

-- Define the properties given in conditions
axiom Cd_outside_AB : outside C (line A B)
axiom Dd_outside_AB : outside D (line A B)

-- The statement to be proved
theorem distinct_lines (A B C D : points) (h₁: outside C (line A B)) (h₂: outside D (line A B)) :
  ∃ n, n = 4 ∨ n = 6 :=
sorry

end distinct_lines_l147_147530


namespace largest_multiple_of_15_less_than_500_l147_147076

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), (n < 500) ∧ (15 ∣ n) ∧ (∀ m : ℕ, (m < 500) ∧ (15 ∣ m) → m ≤ n) ∧ n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l147_147076


namespace least_area_triangle_DEF_l147_147900

noncomputable def area_of_triangle (r : ℝ) (θ1 θ2 θ3 : ℝ) : ℝ :=
1 / 2 * r * r * abs (sin (θ2 - θ1) + sin (θ3 - θ2) + sin (θ1 - θ3))

theorem least_area_triangle_DEF :
  ∀ (r : ℝ) (θ1 θ2 θ3 : ℝ), 
  r = real.sqrt 2 →
  ∃ (θD θE θF : ℝ),
  θD ≠ θE ∧ θE ≠ θF ∧ θF ≠ θD ∧ 
  θD = 0 ∧ θE = 2 * real.pi / 10 ∧ θF = real.pi →
  area_of_triangle r θD θE θF = 1 :=
begin
  intros r θ1 θ2 θ3 hr hex,
  sorry
end

end least_area_triangle_DEF_l147_147900


namespace largest_multiple_of_15_less_than_500_l147_147114

theorem largest_multiple_of_15_less_than_500 :
∀ x : ℕ, (∃ k : ℕ, x = 15 * k ∧ 0 < x ∧ x < 500) → x ≤ 495 :=
begin
  sorry
end

end largest_multiple_of_15_less_than_500_l147_147114


namespace largest_multiple_of_15_less_than_500_l147_147044

theorem largest_multiple_of_15_less_than_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ (∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x) := sorry

end largest_multiple_of_15_less_than_500_l147_147044


namespace total_hours_worked_l147_147999

theorem total_hours_worked (amber_hours : ℕ) (armand_hours : ℕ) (ella_hours : ℕ) 
(h_amber : amber_hours = 12) 
(h_armand : armand_hours = (1 / 3) * amber_hours) 
(h_ella : ella_hours = 2 * amber_hours) :
amber_hours + armand_hours + ella_hours = 40 :=
sorry

end total_hours_worked_l147_147999


namespace train_length_l147_147605

theorem train_length (speed_kmh : ℕ) (cross_time_s : ℕ) (bridge_length_m : ℕ) 
  (conversion_factor : ℝ) (converted_speed_ms : ℝ) (distance_traveled_m : ℝ) : 
  speed_kmh = 45 → 
  cross_time_s = 30 → 
  bridge_length_m = 265 → 
  conversion_factor = (1000:ℝ) / 3600 → 
  converted_speed_ms = (speed_kmh:ℝ) * conversion_factor → 
  distance_traveled_m = converted_speed_ms * (cross_time_s:ℝ) → 
  (distance_traveled_m - (bridge_length_m:ℝ) = 110) :=
begin
  intros h_speed h_time h_blength h_cfactor h_speed_conv h_distance,
  sorry
end

end train_length_l147_147605


namespace radius_of_complex_roots_l147_147224

noncomputable def complex_radius (z : ℂ) : ℝ :=
  if (z + 2)^6 = 64 * z^6 then |z + 2| / |z| else 0

theorem radius_of_complex_roots :
  ∀ z : ℂ, (z + 2)^6 = 64 * z^6 → complex_radius z = 2 / Real.sqrt 3 :=
by sorry

end radius_of_complex_roots_l147_147224


namespace evaluate_expression_l147_147668

theorem evaluate_expression :
  let x := (16 : ℚ) / 9
  in ⌈(√x)⌉ + ⌈x⌉ + ⌈x^2⌉ = 8 :=
by
  let x := (16 : ℚ) / 9
  sorry

end evaluate_expression_l147_147668


namespace base_of_exponent_in_prime_factorization_eq_two_l147_147884

def f (n : ℕ) : ℕ := (∏ i in finset.range (n^2 - 3) + 4, i)
def g (n : ℕ) : ℕ := (∏ i in finset.range n + 1, i^2)

theorem base_of_exponent_in_prime_factorization_eq_two :
  prime_factor_exponent (f 3 / g 3) 4 = 2 :=
by
  sorry

end base_of_exponent_in_prime_factorization_eq_two_l147_147884


namespace complex_root_circle_radius_l147_147226

noncomputable def radius_of_circle : ℂ → ℂ := sorry

theorem complex_root_circle_radius (z : ℂ) :
  (z + 2)^6 = 64 * z^6 → abs z = 2 / 3 :=
begin
  sorry
end

end complex_root_circle_radius_l147_147226


namespace determine_a_binomial_square_l147_147250

theorem determine_a_binomial_square
  (a : ℝ) :
  (∃ r s : ℝ, ax^2 + 21x + 9 = (r * x + s) ^ 2 ∧ s = 3 ∧ r = 7 / 2) →
  a = 49 / 4 := 
sorry

end determine_a_binomial_square_l147_147250


namespace palmer_total_photos_l147_147476

theorem palmer_total_photos (initial_photos : ℕ) (first_week_photos : ℕ) (third_fourth_weeks_photos : ℕ) :
  (initial_photos = 100) →
  (first_week_photos = 50) →
  (third_fourth_weeks_photos = 80) →
  let second_week_photos := 2 * first_week_photos in
  let total_bali_photos := first_week_photos + second_week_photos + third_fourth_weeks_photos in
  let total_photos := initial_photos + total_bali_photos in
  total_photos = 330 :=
by
  intros h_initial h_first_week h_third_fourth_weeks
  let second_week_photos := 2 * first_week_photos
  let total_bali_photos := first_week_photos + second_week_photos + third_fourth_weeks_photos
  let total_photos := initial_photos + total_bali_photos
  sorry

end palmer_total_photos_l147_147476


namespace angle_is_pi_over_3_l147_147370

open Real

def vec_a : ℝ × ℝ := (1, sqrt 3)
def vec_b : ℝ × ℝ := (3, 0)

noncomputable def angle_between_vectors (a b : ℝ × ℝ) : ℝ :=
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (sqrt (a.1 ^ 2 + a.2 ^ 2) * sqrt (b.1 ^ 2 + b.2 ^ 2)))

theorem angle_is_pi_over_3 : angle_between_vectors vec_a vec_b = π / 3 := 
  sorry

end angle_is_pi_over_3_l147_147370


namespace hermia_elected_probability_l147_147840

-- Define the problem statement and conditions in Lean 4
noncomputable def probability_hermia_elected (n : ℕ) (h_odd : (n % 2 = 1)) (h_pos : n > 0) : ℚ :=
  if n = 1 then 1 else (2^n - 1) / (n * 2^(n-1))

-- Lean theorem statement
theorem hermia_elected_probability (n : ℕ) (h_odd : (n % 2 = 1)) (h_pos : n > 0) : 
  probability_hermia_elected n h_odd h_pos = (2^n - 1) / (n * 2^(n-1)) :=
by
  sorry

end hermia_elected_probability_l147_147840


namespace show_AT_eq_RC_l147_147996

-- Define necessary points and properties
variables {A B C D M P R Q S T : Type*}

-- Assume ABCD is a cyclic quadrilateral
axiom cyclic_quadrilateral (A B C D : Type*) : Prop

-- Assume M is the midpoint of CD
axiom midpoint_CD (M C D : Type*) : Prop

-- Assume the intersection of diagonals AC and BD is P
axiom intersection_diag (A B C D P : Type*) : Prop

-- Assume circle through P touching CD at M meets AC again at R and BD again at Q
axiom circle_properties (P C D M A R B Q : Type*) : Prop

-- Assume S on BD such that BS = DQ
axiom point_S_BD (S B D Q : Type*) : Prop

-- Assume line through S parallel to AB meets AC at T
axiom line_parallel (S A B T C : Type*) : Prop

-- The goal is to show AT = RC
theorem show_AT_eq_RC (A B C D M P R Q S T : Type*)
  [cyclic_quadrilateral A B C D]
  [midpoint_CD M C D]
  [intersection_diag A B C D P]
  [circle_properties P C D M A R B Q]
  [point_S_BD S B D Q]
  [line_parallel S A B T C] :
  T = R :=
sorry

end show_AT_eq_RC_l147_147996


namespace relationships_with_correlation_l147_147882

/-- The relationships given in the problem -/
inductive Relationship
| Great_Teachers_To_Outstanding_Students : Relationship
| Volume_To_Radius_Of_Sphere : Relationship
| Apple_Yield_To_Climate : Relationship
| Diameter_To_Height_Of_Tree : Relationship
| Student_To_Student_ID : Relationship
| Crow_Caws_Not_Good_Omen : Relationship

/-- Determining if a relationship has a correlation -/
def hasCorrelation : Relationship → Prop
| Relationship.Great_Teachers_To_Outstanding_Students := true
| Relationship.Volume_To_Radius_Of_Sphere := false
| Relationship.Apple_Yield_To_Climate := true
| Relationship.Diameter_To_Height_Of_Tree := true
| Relationship.Student_To_Student_ID := false
| Relationship.Crow_Caws_Not_Good_Omen := false

theorem relationships_with_correlation :
  (hasCorrelation Relationship.Great_Teachers_To_Outstanding_Students) ∧
  (hasCorrelation Relationship.Apple_Yield_To_Climate) ∧
  (hasCorrelation Relationship.Diameter_To_Height_Of_Tree) :=
by {
  repeat { split <|> trivial }
}

end relationships_with_correlation_l147_147882


namespace divisors_of_g_l147_147713

/-- For positive integers n, let g(n) be the smallest positive integer k such that:
  1. The decimal representation of 1/k ends with exactly n trailing zeros.
  2. The sum of digits of 1/k equals n.
    Determine the number of positive integer divisors of g(2010). -/
def g (n : ℕ) : ℕ := 2 ^ n * 5 ^ n * 3

theorem divisors_of_g (n : ℕ) (h : n = 2010) :
  ∃ d : ℕ, d = 8084442 ∧ Nat.divisors_count (g n) = d :=
by
  use 8084442
  split
  · rfl
  · sorry

end divisors_of_g_l147_147713


namespace find_mode_of_scores_l147_147896

def scores : List ℕ :=
  [64, 65, 73, 81, 85, 85, 92, 96, 97, 98, 98, 102, 102, 102, 104, 106, 106, 106, 106, 110, 110]

def mode (l : List ℕ) : ℕ :=
  l.foldr (λ x acc, if l.count x > l.count acc then x else acc) 0

theorem find_mode_of_scores : mode scores = 106 :=
by
  sorry

end find_mode_of_scores_l147_147896


namespace proposition_p_not_negation_of_p_l147_147757

noncomputable def f (x : ℝ) : ℝ := 2^(x - 1) - 1

theorem proposition_p (x : ℝ) (h : x > 1) : 2^(x - 1) - 1 > 0 :=
begin
  sorry
end

theorem not_negation_of_p : ¬ (∃ x : ℝ, x > 1 ∧ 2^(x - 1) - 1 ≤ 0) :=
begin
  sorry
end

end proposition_p_not_negation_of_p_l147_147757


namespace count_valid_three_digit_numbers_l147_147764

def is_prime_digit (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def valid_three_digit_numbers : List (ℕ × ℕ × ℕ) :=
  List.filter (λ (x : ℕ × ℕ × ℕ), 
    let (a, b, c) := x in
    is_prime_digit a ∧ is_prime_digit b ∧ is_prime_digit c ∧ (a + b + c) % 2 = 0)
  ((List.Product (List.Product [2, 3, 5, 7] [2, 3, 5, 7]) [2, 3, 5, 7]))

theorem count_valid_three_digit_numbers : valid_three_digit_numbers.length = 18 := by
  sorry

end count_valid_three_digit_numbers_l147_147764


namespace derivative_y_l147_147502

noncomputable def y (x : ℝ) : ℝ := x * cos x

theorem derivative_y (x : ℝ) : deriv y x = cos x - x * sin x :=
by
  sorry

end derivative_y_l147_147502


namespace evaluate_expression_l147_147667

theorem evaluate_expression :
  let x := (16 : ℚ) / 9
  in ⌈(√x)⌉ + ⌈x⌉ + ⌈x^2⌉ = 8 :=
by
  let x := (16 : ℚ) / 9
  sorry

end evaluate_expression_l147_147667


namespace hexagon_square_side_length_l147_147636

theorem hexagon_square_side_length
  (ABCDEF : Hexagon) (PQRS : Square)
  (h1 : is_inscribed ABCDEF PQRS)
  (h2_PQ : PQRS.PQ.on_line ABCDEF.BC)
  (h2_QR : PQRS.QR.on_line ABCDEF.DE)
  (h2_RS : PQRS.RS.on_line ABCDEF.EF)
  (h3 : ABCDEF.AB = 50)
  (h4 : ABCDEF.EF = 51 * (real.sqrt 3 - 2)) :
  PQRS.side_length = 34 * real.sqrt 3 - 34 := 
sorry

end hexagon_square_side_length_l147_147636


namespace james_ladder_wood_l147_147806

theorem james_ladder_wood (wood_length : ℝ) 
  (rung_length : ℝ) (rung_spacing : ℝ)
  (support_length : ℝ) (extra_support_length : ℝ)
  (tree_height : ℝ)
  (hwood : wood_length = 300) 
  (hrung_len : rung_length = 18 / 12) 
  (hrung_spacing : rung_spacing = 6 / 12)
  (hsupport_len : support_length = 50)
  (hextra_support_len : extra_support_length = 3)
  (htree_height : tree_height = 50) :
  let total_support_length := 2 * (support_length + 2 * extra_support_length),
      num_rungs := tree_height / rung_spacing + 1,
      total_rung_length := num_rungs * rung_length
  in wood_length - (total_support_length + total_rung_length) = 36.5 := 
by
  sorry

end james_ladder_wood_l147_147806


namespace circle_center_radius_l147_147878

theorem circle_center_radius : 
  ∃ (center : ℝ × ℝ) (radius : ℝ), 
  center = (2, 0) ∧ radius = 2 ∧ ∀ (x y : ℝ), x^2 + y^2 - 4 * x = 0 ↔ (x - 2)^2 + y^2 = 4 :=
by
  sorry

end circle_center_radius_l147_147878


namespace marble_problem_l147_147184

theorem marble_problem (total_marbles : ℕ)
    (h_total : total_marbles = 50 / 0.35)
    (red_marbles : ℕ)
    (h_red : red_marbles = 0.25 * total_marbles)
    (blue_marbles : ℕ)
    (h_blue : blue_marbles = 0.15 * total_marbles)
    (yellow_marbles : ℕ)
    (h_yellow : yellow_marbles = 0.20 * total_marbles)
    (purple_marbles : ℕ)
    (h_purple : purple_marbles = 0.05 * total_marbles)
    (white_marbles : ℕ)
    (h_white : white_marbles = 50)
    (blue_marbles_after : ℕ)
    (h_replace : blue_marbles_after = blue_marbles + (red_marbles / 3)) :
  blue_marbles_after = 33 :=
sorry

end marble_problem_l147_147184


namespace count_valid_numbers_l147_147942

-- Definition for the modified sequence rules
def next_term (n : Int) : Int :=
  if n % 2 = 0 then n + 10 else n - 5

-- Check if the sequence starting from 'start' ever hits -1
def hits_neg (start : Int) : Bool :=
  let rec go (n : Int) : Bool :=
    if n < 0 then true
    else if n = -1 then true
    else go (next_term n)
  go start

def is_valid (n : Int) : Bool :=
  decide $ 1 <= n ∧ n <= 25 ∧ ¬hits_neg n

theorem count_valid_numbers : 
  (Finset.filter (λ n => is_valid n) (Finset.range (25 + 1))).card = 20 := 
sorry

end count_valid_numbers_l147_147942


namespace subset_equality_l147_147954

theorem subset_equality :
  { p : ℝ × ℝ × ℝ × ℝ × ℝ | let a := p.1, b := p.2.1, c := p.2.2.1, d := p.2.2.2.1, e := p.2.2.2.2 in 
    (0 < a) ∧ (0 < b) ∧ (0 < c) ∧ (0 < d) ∧ (0 < e) ∧ 
    (Real.logb (1/6) (a^6 + b^6 + c^6 + d^6 + e^6 + 1) ≥ 
     Real.logb (1/6) a + Real.logb (1/6) b + Real.logb (1/6) c + Real.logb (1/6) d + Real.logb (1/6) e - 1)} = 
  { (1, 1, 1, 1, 1) } :=
by
  sorry

end subset_equality_l147_147954


namespace cone_surface_area_l147_147306

theorem cone_surface_area (r : ℝ) (theta : ℝ) (l : ℝ) (arc_length : ℝ) :
  r = 1 ∧ theta = π / 3 ∧ arc_length = 2 * π * r ∧ (π * l / 3 = arc_length) ∧ l = 6 →
  let base_area := π * r^2 in
  let lateral_area := π * r * l in
  let S := base_area + lateral_area in
  S = 7 * π :=
begin
  sorry
end

end cone_surface_area_l147_147306


namespace num_ints_in_range_with_same_remainder_l147_147365

theorem num_ints_in_range_with_same_remainder (n r : ℤ) (h1 : 200 < n) (h2 : n < 300)
  (h3 : n % 7 = r) (h4 : n % 9 = r) : finset.card ((finset.Icc 201 299).filter (λ n, (n % 7 = r) ∧ (n % 9 = r))) = 7 := 
begin
  sorry
end

end num_ints_in_range_with_same_remainder_l147_147365


namespace rationalize_expression_l147_147384

theorem rationalize_expression :
  let expr := (Real.sqrt 6 + 1 / Real.sqrt 6 + Real.sqrt 8 + 1 / Real.sqrt 8) in
  ∃ (a b c : ℕ), c > 0 ∧
    c * expr = ↑a * Real.sqrt 6 + ↑b * Real.sqrt 8 ∧
    a + b + c = 106 :=
by
  sorry

end rationalize_expression_l147_147384


namespace largest_multiple_of_15_less_than_500_l147_147105

theorem largest_multiple_of_15_less_than_500 :
∀ x : ℕ, (∃ k : ℕ, x = 15 * k ∧ 0 < x ∧ x < 500) → x ≤ 495 :=
begin
  sorry
end

end largest_multiple_of_15_less_than_500_l147_147105


namespace permutation_exists_l147_147832

theorem permutation_exists (n : ℕ) (h_even : n % 2 = 0) :
  ∃ (x : Fin n → Fin n), (∀ i : Fin n, (x (⟨(i : ℕ) + 1 % n, Nat.mod_lt (i + 1 % n) (Nat.pos_of_ne_zero (λ h, h_even (zero_mul 2).symm)))⟩) =
    2 * x i ∨ x (⟨(i : ℕ) + 1 % n, Nat.mod_lt (i + 1 % n) (Nat.pos_of_ne_zero (λ h, h_even (zero_mul 2).symm)))⟩ =
    2 * x i - 1 ∨ x (⟨(i : ℕ) + 1 % n, Nat.mod_lt (i + 1 % n) (Nat.pos_of_ne_zero (λ h, h_even (zero_mul 2).symm)))⟩ =
    2 * x i - n ∨ x (⟨(i : ℕ) + 1 % n, Nat.mod_lt (i + 1 % n) (Nat.pos_of_ne_zero (λ h, h_even (zero_mul 2).symm)))⟩ =
    2 * x i - n - 1) ∧ x ⟨0, Nat.zero_lt_of_ne_zero (λ h, h_even (zero_mul 2).symm)⟩ = x ⟨n, Nat.pos_of_ne_zero (λ h, h_even (zero_mul 2).symm)⟩ := 
  sorry

end permutation_exists_l147_147832


namespace find_n_l147_147901

-- Define the divisors and their natural logarithms
def sum_log_of_divisors (n : ℕ) : ℝ :=
  let ln2 := Real.log 2
  let ln3 := Real.log 3
  (\sum a in Finset.range (n + 1), a * ln2 + \sum b in Finset.range (n + 1), b * ln3) * (n + 1)
  + (\sum b in Finset.range (n + 1), b * ln3 + \sum a in Finset.range (n + 1), a * ln2) * (n + 1)
  * (1 / 2) * (n * (n + 1) * (n + 1)) * (ln2 + ln3)

-- The problem statement expressed in Lean
theorem find_n (n : ℕ) (h : sum_log_of_divisors n = 315) : n = 6 :=
begin
  sorry -- Proof will go here
end

end find_n_l147_147901


namespace division_of_powers_l147_147938

theorem division_of_powers (a : ℝ) (h : a ≠ 0) : a^10 / a^9 = a := 
by sorry

end division_of_powers_l147_147938


namespace avg_abc_l147_147501

variable (A B C : ℕ)

-- Conditions
def avg_ac : Prop := (A + C) / 2 = 29
def age_b : Prop := B = 26

-- Theorem stating the average age of a, b, and c
theorem avg_abc (h1 : avg_ac A C) (h2 : age_b B) : (A + B + C) / 3 = 28 := by
  sorry

end avg_abc_l147_147501


namespace area_of_triangle_AOB_l147_147309

theorem area_of_triangle_AOB 
  (S_sector : ℝ)
  (radius : ℝ)
  (central_angle : ℝ)
  (h1 : S_sector = 2 * real.pi / 3)
  (h2 : radius = 2)
  (h3 : central_angle = real.pi / 3) :
  (1 / 2 * radius * radius * real.sin(central_angle) = real.sqrt 3) :=
by sorry

end area_of_triangle_AOB_l147_147309


namespace largest_prime_factor_l147_147274

theorem largest_prime_factor (a b c : ℕ) (h1 : a = 20) (h2 : b = 15) (h3 : c = 10) :
  Nat.gcd (a * a * a) (b * b * b * b) = 1 ∧ Nat.gcd (a * a * a) (c * c * c * c * c) = 1 ∧ Nat.gcd (b * b * b * b) (c * c * c * c * c) = 1 →
  Nat.largest_prime_factor (a ^ 3 + b ^ 4 - c ^ 5) = 13 :=
by
  sorry

end largest_prime_factor_l147_147274


namespace probability_product_lt_50_l147_147859

-- Definitions based on the problem conditions.
def numbersPaco : Finset ℕ := Finset.range 5  -- Represents {1, 2, 3, 4, 5}
def numbersManu : Finset ℕ := Finset.range 15 -- Represents {1, 2, 3, ..., 15}

def countFavorableOutcomes : ℕ :=
  let outcomes := [(p, m) | p ∈ numbersPaco, m ∈ numbersManu, p * m < 50]
  outcomes.length

def totalOutcomes : ℕ :=
  numbersPaco.card * numbersManu.card

def probabilityProductLessThan50 : ℚ :=
  (countFavorableOutcomes : ℚ) / (totalOutcomes : ℚ)

theorem probability_product_lt_50 :
  probabilityProductLessThan50 = 22 / 25 :=
by
  sorry

end probability_product_lt_50_l147_147859


namespace evaluate_expression_l147_147653

theorem evaluate_expression : 
  (⌈Real.sqrt (16 / 9)⌉ + ⌈ (16 / 9 : ℝ ) ⌉ + ⌈Real.pow (16 / 9 : ℝ ) 2⌉) = 8 := 
by 
  sorry

end evaluate_expression_l147_147653


namespace fraction_of_buttons_l147_147460

variable (K S M : ℕ)  -- Kendra's buttons, Sue's buttons, Mari's buttons

theorem fraction_of_buttons (H1 : M = 5 * K + 4) 
                            (H2 : S = 6)
                            (H3 : M = 64) :
  S / K = 1 / 2 := by
  sorry

end fraction_of_buttons_l147_147460


namespace tangent_slope_l147_147197

noncomputable def Point := (ℝ × ℝ)

def Parabola (x : ℝ) : ℝ := x^2

def Circle (x y : ℝ) : Prop := (x-2)^2 + y^2 = 5

def tangent_to_parabola (l : ℝ → ℝ) (p : Point) : Prop :=
  ∃ (a : ℝ), p = (a, Parabola a) ∧ l = λ x, 2 * a * x - a^2

def tangent_to_circle (l : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), Circle a b ∧ l = (λ x : ℝ, -2 * x - 1)

theorem tangent_slope : 
  ∀ (l : ℝ → ℝ), (∃ (P : Point), P = (1, -3) ∧ tangent_to_parabola l P ∧ tangent_to_circle l)
  → (∃ (k : ℝ), k = -2) :=
by
  sorry

end tangent_slope_l147_147197


namespace largest_multiple_of_15_under_500_l147_147000

theorem largest_multiple_of_15_under_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ ∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x :=
by 
  sorry

end largest_multiple_of_15_under_500_l147_147000


namespace sum_x_y_z_l147_147441

noncomputable def a : ℝ := -Real.sqrt (9 / 25)
noncomputable def b : ℝ := Real.sqrt ((3 + Real.sqrt 2)^2 / 14)

theorem sum_x_y_z :
  a^2 = 9 / 25 →
  b^2 = (3 + Real.sqrt 2)^2 / 14 →
  a < 0 →
  0 < b →
  ∃ x y z : ℕ, (a - b)^3 = (x * Real.sqrt y) / z ∧ x + y + z = 34584 :=
begin
  intros h1 h2 h3 h4,
  use [270, 14, 34300],
  split,
  { sorry },
  { norm_num }
end

end sum_x_y_z_l147_147441


namespace selected_numbers_in_range_l147_147989

noncomputable def systematic_sampling (n_students selected_students interval_num start_num n : ℕ) : ℕ :=
  start_num + interval_num * (n - 1)

theorem selected_numbers_in_range (x : ℕ) :
  (500 = 500) ∧ (50 = 50) ∧ (10 = 500 / 50) ∧ (6 ∈ {y : ℕ | 1 ≤ y ∧ y ≤ 10}) ∧ (125 ≤ x ∧ x ≤ 140) → 
  (x = systematic_sampling 500 50 10 6 13 ∨ x = systematic_sampling 500 50 10 6 14) :=
by
  sorry

end selected_numbers_in_range_l147_147989


namespace sumata_family_miles_driven_per_day_l147_147875

theorem sumata_family_miles_driven_per_day :
  let total_miles := 1837.5
  let number_of_days := 13.5
  let miles_per_day := total_miles / number_of_days
  (miles_per_day : Real) = 136.1111 :=
by
  sorry

end sumata_family_miles_driven_per_day_l147_147875


namespace pyramid_cube_volume_ratio_l147_147586

noncomputable def ratio_of_volumes (α : ℝ) : ℝ :=
  (3 * Real.sqrt 2 * Real.cot α) / ((1 + Real.sqrt 2 * Real.cot α)^3)

theorem pyramid_cube_volume_ratio (α : ℝ) :
  let V_cube := (x: ℝ) → x^3,
      V_pyramid := (x: ℝ) → (Real.sqrt 2 * x^3 * (1 + Real.sqrt 2 * Real.cot α)^3) / (6 * Real.tan α)
  in ratio_of_volumes α = (V_cube x / V_pyramid x) :=
by
  sorry

end pyramid_cube_volume_ratio_l147_147586


namespace proof_problem_l147_147672

def sqrt_frac : ℚ := real.sqrt (16 / 9)
def frac : ℚ := 16 / 9
def square_frac : ℚ := frac * frac

def ceil_sqrt_frac : ℤ := ⌈sqrt_frac⌉.to_int
def ceil_frac : ℤ := ⌈frac⌉.to_int
def ceil_square_frac : ℤ := ⌈square_frac⌉.to_int

theorem proof_problem :
  ceil_sqrt_frac + ceil_frac + ceil_square_frac = 8 :=
by
  -- Placeholder for the actual proof.
  sorry

end proof_problem_l147_147672


namespace inequality_one_inequality_two_l147_147848

noncomputable def primes : Set ℕ := {p | Nat.prime p}

theorem inequality_one (n : ℕ) (h : 3 ≤ n) : 
  (∑ p in primes.filter (λ p, p ≤ n), (1 : ℝ) / p) ≥ Real.log (Real.log n) + O(1) :=
sorry

theorem inequality_two (n k : ℕ) (h1 : 3 ≤ n) (h2 : 0 < k) : 
  (∑ p in primes.filter (λ p, p ≤ n), (1 : ℝ) / p) ≤ (Real.ofNat k ! * Real.ofNat k * Real.log n)^(1 / Real.ofNat k) :=
sorry

end inequality_one_inequality_two_l147_147848


namespace ceiling_sum_l147_147652

theorem ceiling_sum :
  let a := 4 / 3
  let b := 16 / 9
  let c := 256 / 81
  ⌈a⌉ + ⌈b⌉ + ⌈c⌉ = 8 := by
  sorry

end ceiling_sum_l147_147652


namespace slope_of_line_l147_147928

theorem slope_of_line : ∀ (x y : ℝ), 4 * y = -6 * x + 12 → ∃ m b : ℝ, y = m * x + b ∧ m = -3 / 2 :=
by 
sorry

end slope_of_line_l147_147928


namespace minimize_M_l147_147439

noncomputable def M (a b : ℝ) : ℝ := max (3 * a^2 + 2 * b) (3 * b^2 + 2 * a)

theorem minimize_M :
  ∃ a b : ℝ, (a = -1/3 ∧ b = -1/3) ∧ M a b = -1/3 :=
by
  use [-1/3, -1/3]
  split
  · simp
  · sorry

end minimize_M_l147_147439


namespace total_votes_cast_l147_147189

theorem total_votes_cast (V : ℝ) (h1 : V > 0) (h2 : 0.35 * V = candidate_votes) (h3 : candidate_votes + 2400 = rival_votes) (h4 : candidate_votes + rival_votes = V) : V = 8000 := 
by
  sorry

end total_votes_cast_l147_147189


namespace intersecting_segments_l147_147432

theorem intersecting_segments {r : ℝ} {n : ℕ}
  (h_radius : r > 0)
  (h_segments : ∀ i, i < 4 * n → ∃ (x_i y_i x_f y_f : ℝ), (x_i - x_f)^2 + (y_i - y_f)^2 = 1) :
  ∃ l, (∃ (x y : ℝ), x^2 + y^2 < r^2) ∧
           (l.slope = 0 ∨ l.slope = ∞) ∧
           (∃ i j, i ≠ j ∧ ( (l.is_parallel_to_segment (x_i, y_i) (x_f, y_f)) ∨ 
                               (l.is_perpendicular_to_segment (x_i, y_i) (x_f, y_f)) ) ) :=
sorry

end intersecting_segments_l147_147432


namespace hope_number_2010_l147_147376

def is_hope_number (n : ℕ) : Prop :=
  (nat.digits 10 n).count (λ d, d % 2 = 0) % 2 = 0

def hope_number_seq (k : ℕ) (n : ℕ) : ℕ :=
  Nat.findₓ (λ m, ¬∃ m' < m, is_hope_number m') k

theorem hope_number_2010 : hope_number_seq 2010 = 4019 := 
sorry

end hope_number_2010_l147_147376


namespace find_first_term_l147_147520

noncomputable def factorial : ℕ → ℕ 
| 0       := 1
| (n + 1) := (n + 1) * factorial n

def geo_seq_first_term (a r : ℝ) (n : ℕ) : ℝ := a * r ^ n

theorem find_first_term (a r : ℝ) (h6 : geo_seq_first_term a r 5 = factorial 8)
  (h9 : geo_seq_first_term a r 8 = factorial 9) : a = 166 := by
  sorry

end find_first_term_l147_147520


namespace sum_fraction_sqrt_bound_l147_147172

theorem sum_fraction_sqrt_bound (n : ℕ)
  (h₀ : 0 < n)
  (x : Fin n → ℝ)
  (hx : ∀ i, 0 < x i)
  (hsum : ∑ i in Finset.range n, x i = 1) :
  1 ≤ ∑ i in Finset.range n, 
        let s₁ := ∑ j in Finset.range i, x j
        let s₂ := ∑ j in Finset.range (n-i), x (i + j)
        (x i) / (Real.sqrt (1 + s₁) * Real.sqrt (s₂ + x i)) 
    ∧ ∑ i in Finset.range n, 
        let s₁ := ∑ j in Finset.range i, x j
        let s₂ := ∑ j in Finset.range (n-i), x (i + j)
        (x i) / (Real.sqrt (1 + s₁) * Real.sqrt (s₂ + x i)) < Real.pi / 2 := 
sorry

end sum_fraction_sqrt_bound_l147_147172


namespace apples_in_basket_l147_147569

noncomputable def total_apples (good_cond: ℕ) (good_ratio: ℝ) := (good_cond : ℝ) / good_ratio

theorem apples_in_basket : total_apples 66 0.88 = 75 :=
by
  sorry

end apples_in_basket_l147_147569


namespace ceiling_sum_l147_147650

theorem ceiling_sum :
  let a := 4 / 3
  let b := 16 / 9
  let c := 256 / 81
  ⌈a⌉ + ⌈b⌉ + ⌈c⌉ = 8 := by
  sorry

end ceiling_sum_l147_147650


namespace largest_multiple_of_15_less_than_500_l147_147054

theorem largest_multiple_of_15_less_than_500 : 
  ∃ n : ℕ, n * 15 < 500 ∧ ∀ m : ℕ, m * 15 < 500 → m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l147_147054


namespace exponent_calculation_l147_147240

theorem exponent_calculation : (-1 : ℤ) ^ 53 + (2 : ℤ) ^ (5 ^ 3 - 2 ^ 3 + 3 ^ 2) = 2 ^ 126 - 1 :=
by 
  sorry

end exponent_calculation_l147_147240


namespace gcd_lcm_sum_l147_147556

-- You can define the GCD and LCM using Lean's built-in functions
def gcd (a b : ℕ) : ℕ := Nat.gcd a b
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

-- Creating the theorem we need to prove
theorem gcd_lcm_sum : gcd 45 75 + lcm 48 18 = 159 :=
by
  -- Proof is skipped by using sorry
  sorry

end gcd_lcm_sum_l147_147556


namespace problem_statement_eq_l147_147310

noncomputable def given_sequence (a : ℝ) (n : ℕ) : ℝ :=
  a^n

noncomputable def Sn (a : ℝ) (n : ℕ) (an : ℝ) : ℝ :=
  (a / (a - 1)) * (an - 1)

noncomputable def bn (a : ℝ) (n : ℕ) : ℝ :=
  2 * (Sn a n (given_sequence a n)) / (given_sequence a n) + 1

noncomputable def cn (a : ℝ) (n : ℕ) : ℝ :=
  (n - 1) * (bn a n)

noncomputable def Tn (a : ℝ) (n : ℕ) : ℝ :=
  (List.range n).foldl (λ acc k => acc + cn a (k + 1)) 0

theorem problem_statement_eq :
  ∀ (a : ℝ) (n : ℕ), a ≠ 0 → a ≠ 1 →
  (bn a n = (3:ℝ)^n) →
  Tn (1 / 3) n = 3^(n+1) * (2 * n - 3) / 4 + 9 / 4 :=
by
  intros
  sorry

end problem_statement_eq_l147_147310


namespace integral_of_quadratic_function_l147_147344

theorem integral_of_quadratic_function :
  ∃ m : ℝ, (∀ x : ℝ, f x = x^2 + 2*x + m) →
  (∀ x : ℝ, (x^2 + 2*x + m) ≥ -1) →
  (∫ x in 1..2, x^2 + 2*x + m) = 16 / 3 := 
sorry

end integral_of_quadratic_function_l147_147344


namespace Marty_paint_combinations_l147_147463

theorem Marty_paint_combinations :
  let colors := 5 -- blue, green, yellow, black, white
  let styles := 3 -- brush, roller, sponge
  let invalid_combinations := 1 * 1 -- white paint with roller
  let total_combinations := (4 * styles) + (1 * (styles - 1))
  total_combinations = 14 :=
by
  -- Define the total number of combinations excluding the invalid one
  let colors := 5
  let styles := 3
  let invalid_combinations := 1 -- number of invalid combinations (white with roller)
  let valid_combinations := (4 * styles) + (1 * (styles - 1))
  show valid_combinations = 14
  {
    exact rfl -- This will assert that the valid_combinations indeed equals 14
  }

end Marty_paint_combinations_l147_147463


namespace largest_multiple_of_15_less_than_500_l147_147134

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), 15 * n < 500 ∧ 15 * n ≥ 15 * (33 - 1) + 1 := by
  sorry

end largest_multiple_of_15_less_than_500_l147_147134


namespace weight_difference_l147_147877

-- Defining the weights of the individuals
variables (a b c d e : ℝ)

-- Given conditions as hypotheses
def conditions :=
  (a = 75) ∧
  ((a + b + c) / 3 = 84) ∧
  ((a + b + c + d) / 4 = 80) ∧
  ((b + c + d + e) / 4 = 79)

-- Theorem statement to prove the desired result
theorem weight_difference (h : conditions a b c d e) : e - d = 3 :=
by
  sorry

end weight_difference_l147_147877


namespace cone_volume_proof_l147_147776

-- The radius and central angle of the sector (lateral surface of the cone)
def lateral_radius : ℝ := 2
def central_angle : ℝ := 3 * π / 2 -- 270 degrees in radians

-- Volume of the cone using the given conditions
def cone_volume (r h : ℝ) : ℝ := (1 / 3) * π * r^2 * h

theorem cone_volume_proof
    (hr : 2 * π * (3 / 4 * lateral_radius) = 2 * π * r)
    (hh : h = sqrt (lateral_radius^2 - (r^2))) :
    cone_volume r h = (3 * sqrt 7 / 8) * π := by
  sorry

end cone_volume_proof_l147_147776


namespace imo_is_perfect_square_l147_147575

theorem imo_is_perfect_square (CMO : ℕ) (h₁ : 100 ≤ CMO ∧ CMO ≤ 999) (h₂ : ∃ CM O, CM * 10 + O = CMO ∧ ∃ m, sqrt CMO = m ∧ m = CM - sqrt O) : 
∃ k : ℕ, let IMO := match CMO with
  | 100 => (10 ^ 2022)^2
  | 121 => (10 ^ 2022 + 1)^2
  | 144 => (10 ^ 2022 + 2)^2
  | 169 => (10 ^ 2022 + 3)^2
  | _ => 0 -- for other cases, we assume IMO = 0 which cannot happen due to the constraints of the question.
in IMO = k^2 :=
by
  sorry

end imo_is_perfect_square_l147_147575


namespace largest_multiple_of_15_less_than_500_l147_147095

theorem largest_multiple_of_15_less_than_500 : ∃ (x : ℕ), (x < 500) ∧ (x % 15 = 0) ∧ ∀ (y : ℕ), (y < 500) ∧ (y % 15 = 0) → y ≤ x :=
begin
  use 495,
  split,
  { norm_num },
  split,
  { norm_num },
  intros y hy1 hy2,
  sorry,
end

end largest_multiple_of_15_less_than_500_l147_147095


namespace part1_part2_part3_l147_147304

-- Part 1
theorem part1 (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_f : ∀ x : ℝ, a * x - b * x^2 ≤ 1) :
  a ≤ 2 * real.sqrt b :=
sorry

-- Part 2
theorem part2 (a b : ℝ) (h_a : 0 < a) (h_b_gt1 : 1 < b) (h_f : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → a * x - b * x^2 ≤ 1) :
  b - 1 ≤ a ∧ a ≤ 2 * real.sqrt b :=
sorry

-- Part 3
theorem part3 (a b : ℝ) (h_a : 0 < a) (h_b_le1 : 0 < b ∧ b ≤ 1) (h_f : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → a * x - b * x^2 ≤ 1) :
  a ≤ b + 1 :=
sorry

end part1_part2_part3_l147_147304


namespace gravitational_potential_energy_doubling_distance_l147_147937

theorem gravitational_potential_energy_doubling_distance
  (G m1 m2 d U : ℝ)
  (hU : U = -G * m1 * m2 / d) : 
  (let U' := -G * m1 * m2 / (2 * d) in U' = U / 2) :=
by
  sorry

end gravitational_potential_energy_doubling_distance_l147_147937


namespace largest_multiple_of_15_less_than_500_l147_147013

theorem largest_multiple_of_15_less_than_500 : ∃ n : ℕ, (n > 0 ∧ 15 * n < 500) ∧ (∀ m : ℕ, m > n → 15 * m >= 500) ∧ 15 * n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l147_147013


namespace speed_ratio_l147_147602

theorem speed_ratio (v_A v_B : ℝ) (h : 71 / v_B = 142 / v_A) : v_A / v_B = 2 :=
by
  sorry

end speed_ratio_l147_147602


namespace frame_covered_area_l147_147604

theorem frame_covered_area :
  let side_length := 10
  let frame_width := 1
  let hole_side_length := side_length - 2 * frame_width
  let frame_area := side_length ^ 2 - hole_side_length ^ 2
  let num_frames := 5
  let total_overlapping_area := 8
  frame_area * num_frames - total_overlapping_area = 172 := by
  let side_length := 10
  let frame_width := 1
  let hole_side_length := side_length - 2 * frame_width
  let frame_area := side_length ^ 2 - hole_side_length ^ 2
  let num_frames := 5
  let total_overlapping_area := 8
  show frame_area * num_frames - total_overlapping_area = 172, by sorry

end frame_covered_area_l147_147604


namespace particle_prob_at_point_l147_147597

-- Definition of the problem conditions
def moves (n : ℕ) (k : ℕ) := (finset.powerset_len k (finset.range n))
def prob_move (r : ℚ) (u : ℚ) (k1 k2 n : ℕ) := (∑ _ in moves n k1, r^k1 * u^k2)

-- Statement of the theorem
theorem particle_prob_at_point :
  prob_move (1/2 : ℚ) (1/2 : ℚ) 2 3 5 = (5 / 16 : ℚ) :=
sorry

end particle_prob_at_point_l147_147597


namespace maximize_AP_l147_147798

-- Define the parametric equation of the line l
def line_l (α t : ℝ) : ℝ × ℝ :=
  (1 + t * Real.cos α, t * Real.sin α)

-- Define the polar equation of the curve C and its Cartesian form
def curve_C (θ : ℝ) : ℝ := 4 * Real.cos θ
def curve_C_cartesian (x y : ℝ) : Prop := x^2 + y^2 - 4 * x = 0

-- Define the point A
def point_A : ℝ × ℝ := (1, 0)

-- Prove the coordinates of point P that maximizes |AP|
theorem maximize_AP (α : ℝ) (hα : 0 ≤ α ∧ α < Real.pi) :
  ∃ P : ℝ × ℝ, 
    (P = (1, Real.sqrt 3) ∨ P = (1, -Real.sqrt 3)) ∧ 
    ∀ Q, curve_C_cartesian (fst Q) (snd Q) → 
         (1 / (Real.sqrt ((fst point_A - fst Q)^2 + (snd point_A - snd Q)^2))) ≤ 
         (1 / (Real.sqrt ((fst point_A - 1)^2 + (snd point_A - (if snd Q ≥ 0 then Real.sqrt 3 else -Real.sqrt 3))^2)))
:= sorry

end maximize_AP_l147_147798


namespace sqrt_expr_to_rational_form_l147_147391

theorem sqrt_expr_to_rational_form :
  ∃ (a b c : ℕ), 0 < c ∧ (∑ i in [28, 27, 12], i) = 28 + 27 + 12 ∧
  (sqrt 6 + 1 / sqrt 6 + sqrt 8 + 1 / sqrt 8 = (a * sqrt 6 + b * sqrt 8) / c) ∧
  a + b + c = 67 := 
by
  use 28, 27, 12
  split
  { exact nat.succ_pos' _ }
  split
  { norm_num }
  split
  { -- omitted proof
    sorry }
  { norm_num }

end sqrt_expr_to_rational_form_l147_147391


namespace land_sale_value_correct_l147_147234

theorem land_sale_value_correct :
  let acres_to_sq_yards := 4840
  let yard_to_meters := 0.9144
  let hectare_to_sq_meters := 10000
  let plot_area_acres := 3
  let cost_per_hectare := 250000
  
  let plot_area_sq_yards := plot_area_acres * acres_to_sq_yards
  let plot_area_sq_meters := plot_area_sq_yards * (yard_to_meters ^ 2)
  let plot_area_hectares := plot_area_sq_meters / hectare_to_sq_meters
  let total_value_rubles := plot_area_hectares * cost_per_hectare

  total_value_rubles = 303514 := 
begin
  -- Here we have predefined conditions and the goal equation
  let acres_to_sq_yards := 4840
  let yard_to_meters := 0.9144
  let hectare_to_sq_meters := 10000
  let plot_area_acres := 3
  let cost_per_hectare := 250000
  
  let plot_area_sq_yards := plot_area_acres * acres_to_sq_yards
  let plot_area_sq_meters := plot_area_sq_yards * (yard_to_meters ^ 2)
  let plot_area_hectares := plot_area_sq_meters / hectare_to_sq_meters
  let total_value_rubles := plot_area_hectares * cost_per_hectare
  
  have h : total_value_rubles = 303514, 
  sorry  -- Proof would go here
end

end land_sale_value_correct_l147_147234


namespace find_a_b_of_solution_set_l147_147522

theorem find_a_b_of_solution_set :
  ∃ a b : ℝ, (∀ x : ℝ, x^2 + (a + 1) * x + a * b = 0 ↔ x = -1 ∨ x = 4) → a + b = -3 :=
by
  sorry

end find_a_b_of_solution_set_l147_147522


namespace salary_january_l147_147876

variable (J F M A May : ℝ)

theorem salary_january 
  (h1 : J + F + M + A = 32000) 
  (h2 : F + M + A + May = 33600) 
  (h3 : May = 6500) : 
  J = 4900 := 
by {
 sorry 
}

end salary_january_l147_147876


namespace max_f_value_l147_147815

def f (S : List ℕ) : ℕ :=
  (List.map (λ ⟨a,b⟩, Nat.abs (a - b)) (List.zip S (List.tail S))).minimum

def max_f (n : ℕ) : ℕ :=
  Int.toNat (Int.floor (n / 2 : ℤ))

theorem max_f_value (n : ℕ) (S : List ℕ) (h₁ : ∀ i, i ∈ S → 1 ≤ i ∧ i ≤ n) (h₂ : S.nodup) : f S ≤ max_f n := 
sorry

end max_f_value_l147_147815


namespace largest_multiple_of_15_below_500_l147_147035

theorem largest_multiple_of_15_below_500 : ∃ (k : ℕ), 15 * k < 500 ∧ ∀ (m : ℕ), 15 * m < 500 → 15 * m ≤ 15 * k := 
by
  existsi 33
  split
  · norm_num
  · intro m h
    have h1 : m ≤ 33 := Nat.le_of_lt_succ (Nat.div_lt_succ 500 15 m)
    exact (mul_le_mul_left (by norm_num : 0 < 15)).mpr h1
  sorry

end largest_multiple_of_15_below_500_l147_147035


namespace largest_multiple_of_15_less_than_500_l147_147062

theorem largest_multiple_of_15_less_than_500 : 
  ∃ n : ℕ, n * 15 < 500 ∧ ∀ m : ℕ, m * 15 < 500 → m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l147_147062


namespace square_area_l147_147992

theorem square_area (perimeter : ℝ) (h : perimeter = 24) : ∃ (area : ℝ), area = 36 :=
by
  let side_length := perimeter / 4
  have side_length_eq : side_length = 6 := by
    rw [h]
    simp [side_length]
  let area := side_length ^ 2
  use area
  have area_eq : area = 36 := by
    rw [side_length_eq]
    simp [area]
  exact area_eq

end square_area_l147_147992


namespace infinite_solutions_l147_147483

theorem infinite_solutions (n : ℤ) : ∃ᶠ (x y z : ℕ) in (at_top : filter (ℕ × ℕ × ℕ)), (x^2 + y^2 - z^2 : ℤ) = n :=
by sorry

end infinite_solutions_l147_147483


namespace cake_flour_amount_l147_147853

theorem cake_flour_amount (sugar_cups : ℕ) (flour_already_in : ℕ) (extra_flour_needed : ℕ) (total_flour : ℕ) 
  (h1 : sugar_cups = 7) 
  (h2 : flour_already_in = 2)
  (h3 : extra_flour_needed = 2)
  (h4 : total_flour = sugar_cups + extra_flour_needed) : 
  total_flour = 9 := 
sorry

end cake_flour_amount_l147_147853


namespace palmer_total_photos_l147_147475

theorem palmer_total_photos (initial_photos : ℕ) (first_week_photos : ℕ) (third_fourth_weeks_photos : ℕ) :
  (initial_photos = 100) →
  (first_week_photos = 50) →
  (third_fourth_weeks_photos = 80) →
  let second_week_photos := 2 * first_week_photos in
  let total_bali_photos := first_week_photos + second_week_photos + third_fourth_weeks_photos in
  let total_photos := initial_photos + total_bali_photos in
  total_photos = 330 :=
by
  intros h_initial h_first_week h_third_fourth_weeks
  let second_week_photos := 2 * first_week_photos
  let total_bali_photos := first_week_photos + second_week_photos + third_fourth_weeks_photos
  let total_photos := initial_photos + total_bali_photos
  sorry

end palmer_total_photos_l147_147475


namespace prism_inscribed_in_sphere_iff_right_prism_with_congruent_bases_l147_147549

-- Let P be a prism and S be a sphere such that all vertices of P lie on S
variable (P : Type) [prism P]
variable (S : Type) [sphere S]
variable (vertices : set (point S))
variable (prism_vertices : set (point P))
variable (all_vertices_on_sphere : ∀ v ∈ prism_vertices, v ∈ vertices)

-- The proof goal:
theorem prism_inscribed_in_sphere_iff_right_prism_with_congruent_bases (P : Type) [prism P]
  (S : Type) [sphere S]
  (all_vertices_on_sphere : ∀ v ∈ prism_vertices, v ∈ vertices) :
  (exists (right_prism : Type) [prism right_prism],
    right_prism = P ∧ congruent_bases P ∧
    (∀ base ∈ prism_bases P, ∃ circle, base ⊆ circle ∧ congruent_circles S circle)) ↔
    (∀ base ∈ prism_bases P, exists (circ : circle (point S)), 
      base ⊆ circ ∧ congruent_circles S circ) := sorry

end prism_inscribed_in_sphere_iff_right_prism_with_congruent_bases_l147_147549


namespace percent_both_correctly_answered_l147_147946

def percent (n : ℝ) : ℝ := n / 100

variables {total_test_takers : ℝ} (answered_first_correctly answered_second_correctly answered_neither_correctly : ℝ)

-- Conditions given in the problem
def condition1 := answered_first_correctly = percent 85 * total_test_takers
def condition2 := answered_second_correctly = percent 70 * total_test_takers
def condition3 := answered_neither_correctly = percent 5 * total_test_takers

-- Question : percentage of test takers who answered both questions correctly
theorem percent_both_correctly_answered :
  (total_test_takers - answered_first_correctly + answered_second_correctly - answered_neither_correctly = percent 60 * total_test_takers) :=
by
  sorry

end percent_both_correctly_answered_l147_147946


namespace ratio_of_cylinder_volumes_l147_147962

noncomputable def ratio_of_volumes (height1 circum1 height2 circum2 : ℝ) : ℝ :=
  let r1 := circum1 / (2 * Real.pi) in
  let V1 := Real.pi * r1^2 * height1 in
  let r2 := circum2 / (2 * Real.pi) in
  let V2 := Real.pi * r2^2 * height2 in
  V2 / V1

theorem ratio_of_cylinder_volumes :
  ratio_of_volumes 10 7 7 10 = 14 / 5 :=
by
  sorry

end ratio_of_cylinder_volumes_l147_147962


namespace platform_length_is_correct_l147_147594

def speed_kmph : ℝ := 72
def seconds_to_cross_platform : ℝ := 26
def train_length_m : ℝ := 270.0416

noncomputable def length_of_platform : ℝ :=
  let speed_mps := speed_kmph * (1000 / 3600)
  let total_distance := speed_mps * seconds_to_cross_platform
  total_distance - train_length_m

theorem platform_length_is_correct : 
  length_of_platform = 249.9584 := 
by
  sorry

end platform_length_is_correct_l147_147594


namespace amount_per_referral_correct_l147_147810

def amount_per_referral (total_friends: ℕ) (total_money: ℝ) : ℝ :=
total_money / total_friends

theorem amount_per_referral_correct :
  amount_per_referral 24 125 = 5.21 := 
by {
  sorry -- the detailed proof will go here 
}

end amount_per_referral_correct_l147_147810


namespace largest_multiple_of_15_less_than_500_l147_147113

theorem largest_multiple_of_15_less_than_500 :
∀ x : ℕ, (∃ k : ℕ, x = 15 * k ∧ 0 < x ∧ x < 500) → x ≤ 495 :=
begin
  sorry
end

end largest_multiple_of_15_less_than_500_l147_147113


namespace liars_correctness_l147_147909

theorem liars_correctness (Alan Bob Casey Dan Eric : Prop) 
(Alan_statement : ¬Alan)
(Bob_statement : Alan ∧ Bob = false)
(Casey_statement : ¬Alan ∧ ¬Bob = true)
(Dan_statement : Casey → Eric)
(Eric_statement : (¬Alan ∧ ¬Bob ∧ ¬Dan ∧ ¬Eric) ∨ (Alan ∧ Bob ∧ Dan ∧ Eric)) : 

(¬Alan ∧ ¬Bob ∧ ¬Dan ∧ ¬Eric) ∧ Casey :=
by {
  -- Since Alan statements negate and Alan as liar
  have h₁ := Alan_statement,
  -- Basing Bob statement negates truth also as liar
  have h₂ := (Bob_statement h₁),
  -- Consistent Casey to assert truth-teller
  have h₃ := (Casey_statement h₁ h₂),
  -- Evaluate Dan per Casey:
  have hd := (Dan_statement h₃),
  -- Validate if erring Eric's require odd via combinations
  have he := (Eric_statement h₁ h₂ false.intro hd),
  
  -- Conclusion thus:
  split,
  repeat {exact sorry}, -- proving requirement here 
}

end liars_correctness_l147_147909


namespace games_in_tournament_l147_147213

def single_elimination_games (n : Nat) : Nat :=
  n - 1

theorem games_in_tournament : single_elimination_games 24 = 23 := by
  sorry

end games_in_tournament_l147_147213


namespace largest_multiple_of_15_less_than_500_l147_147109

theorem largest_multiple_of_15_less_than_500 :
∀ x : ℕ, (∃ k : ℕ, x = 15 * k ∧ 0 < x ∧ x < 500) → x ≤ 495 :=
begin
  sorry
end

end largest_multiple_of_15_less_than_500_l147_147109


namespace exists_even_in_sequence_l147_147415

theorem exists_even_in_sequence 
  (a : ℕ → ℕ)
  (h₀ : ∀ n : ℕ, a (n+1) = a n + (a n % 10)) :
  ∃ n : ℕ, a n % 2 = 0 :=
sorry

end exists_even_in_sequence_l147_147415


namespace num_whole_numbers_between_sqrts_l147_147368

theorem num_whole_numbers_between_sqrts :
  ∃ n : ℕ, n = 5 ∧ 
  ∀ x : ℕ, (⌊sqrt 18⌋.val + 1 ≤ x) ∧ (x ≤ ⌊sqrt 98⌋.val - 1) ↔ (5 ≤ x ∧ x ≤ 9) :=
begin
  have lt_18 : (⌊sqrt 18⌋.val + 1) = 5 := by sorry,
  have gt_98 : (⌊sqrt 98⌋.val - 1) = 9 := by sorry,
  use 5,
  split,
  exact rfl,
  intro,
  split,
  intro hx,
  rw ←lt_18 at hx,
  rw ←gt_98 at hx,
  exact hx,
  intro h5to9,
  rw lt_18,
  rw gt_98,
  exact h5to9,
end

end num_whole_numbers_between_sqrts_l147_147368


namespace median_is_2073_l147_147553

def median_of_list (l : List ℕ) : ℕ :=
  let sorted_l := l.sorted (≤)
  let n := sorted_l.length
  if n % 2 = 0 then
    (sorted_l.get (n / 2 - 1) + sorted_l.get (n / 2)) / 2
  else
    (sorted_l.get (n / 2))

def list_1_to_2030_and_squares : List ℕ :=
  (List.range 2030).map (λ n => n + 1) ++ (List.range 2030).map (λ n => (n + 1) * (n + 1))

theorem median_is_2073 :
  median_of_list list_1_to_2030_and_squares = 2073 :=
sorry

end median_is_2073_l147_147553


namespace largest_multiple_of_15_less_than_500_l147_147016

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), n < 500 ∧ 15 ∣ n ∧ ∀ (m : ℕ), m < 500 ∧ 15 ∣ m -> m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l147_147016


namespace number_multiplied_by_any_integer_results_in_itself_l147_147934

theorem number_multiplied_by_any_integer_results_in_itself (N : ℤ) (h : ∀ (x : ℤ), N * x = N) : N = 0 :=
  sorry

end number_multiplied_by_any_integer_results_in_itself_l147_147934


namespace constant_term_expansion_l147_147923

theorem constant_term_expansion : 
  let expr : ℕ → ℝ := λ k, (Nat.choose 8 k) * (2 : ℝ) ^ k * (2 : ℝ) ^ (8 - k)
  in expr 4 = 17920 :=
by
  let expr : ℕ → ℝ := λ k, (Nat.choose 8 k) * (2 : ℝ) ^ k * (2 : ℝ) ^ (8 - k)
  have h : 8.choose 4 = 70 := sorry -- This step can be computed but proved here easily
  show expr 4 = 17920 from sorry

end constant_term_expansion_l147_147923


namespace fish_distance_l147_147573

theorem fish_distance (cube_side : ℝ) (num_fish : ℕ) (fish_positions : fin num_fish → (ℝ × ℝ × ℝ))
  (h_cube : cube_side = 2) (h_num_fish : num_fish = 9)
  (h_fish_in_cube : ∀ i, let (x, y, z) := fish_positions i in 0 ≤ x ∧ x ≤ cube_side ∧ 0 ≤ y ∧ y ≤ cube_side ∧ 0 ≤ z ∧ z ≤ cube_side) :
  ∃ i j, i ≠ j ∧ dist (fish_positions i) (fish_positions j) < real.sqrt 3 :=
sorry

end fish_distance_l147_147573


namespace amount_brought_by_sisters_l147_147544

-- Definitions based on conditions
def cost_per_ticket : ℕ := 8
def number_of_tickets : ℕ := 2
def change_received : ℕ := 9

-- Statement to prove
theorem amount_brought_by_sisters :
  (cost_per_ticket * number_of_tickets + change_received) = 25 :=
by
  -- Using assumptions directly
  let total_cost := cost_per_ticket * number_of_tickets
  have total_cost_eq : total_cost = 16 := by sorry
  let amount_brought := total_cost + change_received
  have amount_brought_eq : amount_brought = 25 := by sorry
  exact amount_brought_eq

end amount_brought_by_sisters_l147_147544


namespace hermia_elected_probability_l147_147835

def probability_hermia_elected (n : ℕ) (h1 : Odd n) (h2 : n > 0) : ℝ :=
  (2 ^ n - 1 : ℝ) / (n * 2 ^ (n - 1))

theorem hermia_elected_probability (n : ℕ) (h1 : Odd n) (h2 : 0 < n) :
  probability_hermia_elected n h1 h2 = (2 ^ n - 1 : ℝ) / (n * 2 ^ (n - 1)) :=
sorry

end hermia_elected_probability_l147_147835


namespace jaya_amitabh_number_of_digits_l147_147807

-- Definitions
def is_two_digit_number (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def digit_sum (n1 n2 : ℕ) : ℕ :=
  let (d1, d2) := (n1 % 10, n1 / 10)
  let (d3, d4) := (n2 % 10, n2 / 10)
  d1 + d2 + d3 + d4
def append_ages (j a : ℕ) : ℕ := 1000 * (j / 10) + 100 * (j % 10) + 10 * (a / 10) + (a % 10)
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- Main theorem
theorem jaya_amitabh_number_of_digits 
  (j a : ℕ) 
  (hj : is_two_digit_number j)
  (ha : is_two_digit_number a)
  (h_sum : digit_sum j a = 7)
  (h_square : is_perfect_square (append_ages j a)) : 
  ∃ n : ℕ, String.length (toString (append_ages j a)) = 4 :=
by
  sorry

end jaya_amitabh_number_of_digits_l147_147807


namespace largest_multiple_of_15_less_than_500_l147_147045

theorem largest_multiple_of_15_less_than_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ (∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x) := sorry

end largest_multiple_of_15_less_than_500_l147_147045


namespace find_single_digit_A_l147_147977

theorem find_single_digit_A (A : ℕ) (h1 : A < 10) (h2 : (11 * A)^2 = 5929) : A = 7 := 
sorry

end find_single_digit_A_l147_147977


namespace tiles_removal_operations_l147_147603

theorem tiles_removal_operations:
  ∀ (initial_tiles : Finset ℕ) (operation : Finset ℕ → Finset ℕ) (number_of_operations : ℕ),
    initial_tiles = Finset.range 122 →
    operation = (λ s, Finset.image (λ x, x - Finset.count x (Finset.filter (λ n, ∃ m, m*m = n) s)) s) →
    ∃ n : ℕ, n = 11 ∧ 
    (Finset.card (Nat.iterate operation n initial_tiles) < 
    Finset.card (Finset.filter (λ n, ∃ m, m*m = n) (Nat.iterate operation n initial_tiles))).card.required_tiles :=
begin
  sorry
end

end tiles_removal_operations_l147_147603


namespace proof_problem_l147_147673

def sqrt_frac : ℚ := real.sqrt (16 / 9)
def frac : ℚ := 16 / 9
def square_frac : ℚ := frac * frac

def ceil_sqrt_frac : ℤ := ⌈sqrt_frac⌉.to_int
def ceil_frac : ℤ := ⌈frac⌉.to_int
def ceil_square_frac : ℤ := ⌈square_frac⌉.to_int

theorem proof_problem :
  ceil_sqrt_frac + ceil_frac + ceil_square_frac = 8 :=
by
  -- Placeholder for the actual proof.
  sorry

end proof_problem_l147_147673


namespace max_distance_proof_l147_147295

-- Given conditions
constant front_tire_lifespan : ℕ := 42000
constant rear_tire_lifespan : ℕ := 56000
constant maximum_distance_traveled : ℕ := 42000

theorem max_distance_proof : maximum_distance_traveled = 42000 := 
begin
  sorry,
end

end max_distance_proof_l147_147295


namespace largest_multiple_15_under_500_l147_147091

-- Define the problem statement
theorem largest_multiple_15_under_500 : ∃ (n : ℕ), n < 500 ∧ n % 15 = 0 ∧ (∀ m, m < 500 ∧ m % 15 = 0 → m ≤ n) :=
by {
  use 495,
  split,
  {
    exact lt_of_le_of_ne (le_refl _) (ne_of_lt (nat.lt_succ_self 499))
  },
  split,
  {
    exact nat.mod_eq_zero_of_dvd ⟨33, rfl⟩,
  },
  {
    intros m hm h,
    have hmn : m ≤ 495,
    {
      have div15 := hm/15,
      suffices : div15 < 33+1, from nat.le_of_lt_add_one this,
      exact this, sorry,
    },
    exact hmn,
  },
}

end largest_multiple_15_under_500_l147_147091


namespace largest_multiple_of_15_below_500_l147_147031

theorem largest_multiple_of_15_below_500 : ∃ (k : ℕ), 15 * k < 500 ∧ ∀ (m : ℕ), 15 * m < 500 → 15 * m ≤ 15 * k := 
by
  existsi 33
  split
  · norm_num
  · intro m h
    have h1 : m ≤ 33 := Nat.le_of_lt_succ (Nat.div_lt_succ 500 15 m)
    exact (mul_le_mul_left (by norm_num : 0 < 15)).mpr h1
  sorry

end largest_multiple_of_15_below_500_l147_147031


namespace largest_prime_factor_expr_l147_147273

noncomputable def expr : ℤ := 20^3 + 15^4 - 10^5

theorem largest_prime_factor_expr : ∃ p : ℕ, prime p ∧ p = 41 ∧ (∀ q : ℕ, prime q ∧ q ∣ expr → q ≤ 41) :=
by {
  sorry
}

end largest_prime_factor_expr_l147_147273


namespace hermia_elected_probability_l147_147836

noncomputable def probability_h ispected_president (n : ℕ) (h : n % 2 = 1) : ℚ :=
  (2^n - 1) / (n * 2^(n-1))

theorem hermia_elected_probability (n : ℕ) (h : n % 2 = 1) :
  let P := probability_h ispected_president n h in 
  hermia_elected_probability = P := 
  sorry

end hermia_elected_probability_l147_147836


namespace largest_multiple_15_under_500_l147_147083

-- Define the problem statement
theorem largest_multiple_15_under_500 : ∃ (n : ℕ), n < 500 ∧ n % 15 = 0 ∧ (∀ m, m < 500 ∧ m % 15 = 0 → m ≤ n) :=
by {
  use 495,
  split,
  {
    exact lt_of_le_of_ne (le_refl _) (ne_of_lt (nat.lt_succ_self 499))
  },
  split,
  {
    exact nat.mod_eq_zero_of_dvd ⟨33, rfl⟩,
  },
  {
    intros m hm h,
    have hmn : m ≤ 495,
    {
      have div15 := hm/15,
      suffices : div15 < 33+1, from nat.le_of_lt_add_one this,
      exact this, sorry,
    },
    exact hmn,
  },
}

end largest_multiple_15_under_500_l147_147083


namespace true_proposition_l147_147517

-- Define propositions p and q
def p (a b : Vector) : Prop := (a ≠ 0 ∧ b ≠ 0) → (a.dot b < 0 → angle a b = π / 2)
def q (f : ℝ → ℝ) (x0 : ℝ) : Prop := (f' x0 = 0) → (is_extreme_point f x0)

-- Given that p and q are incorrect
axiom p_incorrect : ¬p
axiom q_incorrect : ¬q

-- Prove (¬p) ∧ (¬q)
theorem true_proposition : (¬p) ∧ (¬q) :=
by
  exact ⟨p_incorrect, q_incorrect⟩

end true_proposition_l147_147517


namespace first_duck_fraction_l147_147529

-- Definitions based on the conditions
variable (total_bread : ℕ) (left_bread : ℕ) (second_duck_bread : ℕ) (third_duck_bread : ℕ)

-- Given values
def given_values : Prop :=
  total_bread = 100 ∧ left_bread = 30 ∧ second_duck_bread = 13 ∧ third_duck_bread = 7

-- Proof statement
theorem first_duck_fraction (h : given_values total_bread left_bread second_duck_bread third_duck_bread) :
  (total_bread - left_bread) - (second_duck_bread + third_duck_bread) = 1/2 * total_bread := by 
  sorry

end first_duck_fraction_l147_147529


namespace calculate_expression_l147_147617

theorem calculate_expression : (35 / (5 * 2 + 5)) * 6 = 14 :=
by
  sorry

end calculate_expression_l147_147617


namespace pizzas_served_during_lunch_l147_147987

theorem pizzas_served_during_lunch {total_pizzas dinner_pizzas lunch_pizzas: ℕ} 
(h_total: total_pizzas = 15) (h_dinner: dinner_pizzas = 6) (h_eq: total_pizzas = dinner_pizzas + lunch_pizzas) : 
lunch_pizzas = 9 := by
  sorry

end pizzas_served_during_lunch_l147_147987


namespace intersection_A_B_l147_147354

-- Step d: translate to Lean definitions and theorem statement.
def set_A : set ℝ := { x | x + 1 > 0 }
def set_B : set ℝ := { x | 2x - 3 < 0 }

theorem intersection_A_B :
  set_A ∩ set_B = { x | -1 < x ∧ x < (3 / 2) } :=
by
  ext x
  split
  { intro hx
    rcases hx with ⟨hA, hB⟩
    unfold set_A at hA
    unfold set_B at hB
    split
    { linarith }
    { linarith } }
  { intro hx
    cases hx with h1 h2
    split
    { unfold set_A
      linarith }
    { unfold set_B
      linarith } }

end intersection_A_B_l147_147354


namespace find_d_l147_147823

-- Definitions of the functions f and g and condition on f(g(x))
def f (x : ℝ) (c : ℝ) : ℝ := 5 * x + c
def g (x : ℝ) (c : ℝ) : ℝ := c * x + 3

theorem find_d (c d x : ℝ) (h : f (g x c) c = 15 * x + d) : d = 18 :=
sorry

end find_d_l147_147823


namespace brownies_per_person_l147_147291

-- Define the conditions as constants
def columns : ℕ := 6
def rows : ℕ := 3
def people : ℕ := 6

-- Define the total number of brownies
def total_brownies : ℕ := columns * rows

-- Define the theorem to be proved
theorem brownies_per_person : total_brownies / people = 3 :=
by sorry

end brownies_per_person_l147_147291


namespace line_slope_is_negative_three_halves_l147_147929

theorem line_slope_is_negative_three_halves : 
  ∀ (x y : ℝ), (4 * y = -6 * x + 12) → (∀ x y, y = -((3/2) * x) + 3) :=
begin
  sorry
end

end line_slope_is_negative_three_halves_l147_147929


namespace max_pieces_l147_147571

noncomputable def volume (l w t : ℚ) : ℚ :=
  l * w * t

theorem max_pieces (H1 : volume 16 8 (15 / 2) = 960) (H2 : volume 5 3 (5 / 2) = 37.5) :
  ∃ n : ℕ, n = Nat.floor (960 / 37.5) ∧ n = 25 :=
by
  sorry

end max_pieces_l147_147571


namespace total_photos_l147_147470

def initial_photos : ℕ := 100
def photos_first_week : ℕ := 50
def photos_second_week : ℕ := 2 * photos_first_week
def photos_third_fourth_week : ℕ := 80
def photos_from_bali : ℕ := photos_first_week + photos_second_week + photos_third_fourth_week

theorem total_photos (initial_photos photos_from_bali : ℕ) : initial_photos + photos_from_bali = 330 :=
by
  have h1 : initial_photos = 100 := rfl
  have h2 : photos_from_bali = 50 + (2 * 50) + 80 := rfl
  show 100 + (50 + 100 + 80) = 330
  sorry

end total_photos_l147_147470


namespace sum_lucky_numbers_divisible_by_13_l147_147967

theorem sum_lucky_numbers_divisible_by_13 : 
  (∑ n in finset.range (10^6), (n / 1000).digits.sum = (n % 1000).digits.sum -> n) % 13 = 0 :=
sorry

end sum_lucky_numbers_divisible_by_13_l147_147967


namespace count_special_numbers_l147_147364

theorem count_special_numbers : 
  let n := {x : ℕ // 1000 ≤ x ∧ x < 3000 ∧ (∃ i : ℕ, 0 ≤ i ∧ i ≤ 2 ∧ (x / 10^(i+1) % 10) = (x / 10^i % 10))}
  in fintype.card n = 5400 := by
  sorry

end count_special_numbers_l147_147364


namespace carsProducedInEurope_l147_147582

-- Definitions of the conditions
def carsProducedInNorthAmerica : ℕ := 3884
def totalCarsProduced : ℕ := 6755

-- Theorem statement
theorem carsProducedInEurope : ∃ (carsProducedInEurope : ℕ), totalCarsProduced = carsProducedInNorthAmerica + carsProducedInEurope ∧ carsProducedInEurope = 2871 := by
  sorry

end carsProducedInEurope_l147_147582


namespace correct_statements_l147_147426

variables (x y z : ℝ)

def symmetryPointX : ℝ × ℝ × ℝ := (x, -y, z)
def symmetryPointYOz : ℝ × ℝ × ℝ := (x, -y, -z)
def symmetryPointY : ℝ × ℝ × ℝ := (x, -y, z)
def symmetryPointOrigin : ℝ × ℝ × ℝ := (-x, -y, -z)

def correctStatementsCount : ℕ := 
  (if symmetryPointX = (x, -y, -z) then 1 else 0) +
  (if symmetryPointYOz = (-x, y, z) then 1 else 0) +
  (if symmetryPointY = (-x, y, -z) then 1 else 0) +
  (if symmetryPointOrigin = (-x, -y, -z) then 1 else 0)

theorem correct_statements :
  correctStatementsCount x y z = 1 :=
sorry

end correct_statements_l147_147426


namespace total_photos_l147_147474

def initial_photos : ℕ := 100
def photos_first_week : ℕ := 50
def photos_second_week : ℕ := 2 * photos_first_week
def photos_third_and_fourth_weeks : ℕ := 80

theorem total_photos (initial_photos photos_first_week photos_second_week photos_third_and_fourth_weeks : ℕ) :
  initial_photos = 100 ∧
  photos_first_week = 50 ∧
  photos_second_week = 2 * photos_first_week ∧
  photos_third_and_fourth_weeks = 80 →
  initial_photos + photos_first_week + photos_second_week + photos_third_and_fourth_weeks = 330 :=
by
  sorry

end total_photos_l147_147474


namespace equal_chords_l147_147431

variable {ABC : Triangle}
variables {A B C : Point} (k_A k_B : Circle) (D E : Point)

-- Define that circles k_A and k_B are inscribed
def inscribedCircle (k : Circle) (A B : Point) (T : Point) : Prop :=
  k.tangent A T ∧ k.tangent B T ∧ T ∈ k

-- Define the conditions from the problem
def conditions : Prop :=
  inscribedCircle k_A A C E ∧
  inscribedCircle k_B B C D

-- Define the proof problem
theorem equal_chords 
  (h : conditions k_A k_B A B C D E) :
  ∃ S R : Point, 
    (S ∈ k_A) ∧ (R ∈ k_B) ∧ (line_through S E D) ∧ (line_through R E D) ∧
    chord_length k_A E S = chord_length k_B D R :=
sorry

end equal_chords_l147_147431


namespace coloring_ways_l147_147433

-- Define a factorial function
def factorial : Nat → Nat
| 0       => 1
| (n + 1) => (n + 1) * factorial n

-- Define a derangement function
def derangement : Nat → Nat
| 0       => 1
| 1       => 0
| (n + 1) => n * (derangement n + derangement (n - 1))

-- Prove the main theorem
theorem coloring_ways : 
  let six_factorial := factorial 6
  let derangement_6 := derangement 6
  let derangement_5 := derangement 5
  720 * (derangement_6 + derangement_5) = 222480 := by
    let six_factorial := 720
    let derangement_6 := derangement 6
    let derangement_5 := derangement 5
    show six_factorial * (derangement_6 + derangement_5) = 222480
    sorry

end coloring_ways_l147_147433


namespace part_a_part_b_l147_147831

-- Definitions for part (a)
def catalan_seq (c : ℕ → ℕ) : Prop :=
  c 1 = 1 ∧
  c 2 = c 1 * c 1 ∧
  (∀ k ≥ 3, c k = (∑ i in finset.range (k - 1), c (i + 1) * c (k - i - 1)))

def generating_function (f : ℕ → ℕ) (x : ℝ) : ℝ :=
  ∑ k in finset.range (100), (f k) * x^k -- approximation for readability; ideally this should be an infinite series.

-- Part (a): f(x)^2 = f(x) - x
theorem part_a (f : ℕ → ℕ → ℝ) (x : ℝ) (c : ℕ → ℕ) :
  catalan_seq c →
  (generating_function f x)^2 = generating_function f x - x :=
sorry

-- Part (b): c_n = (2n-2)! / (n! * (n-1)!)
theorem part_b (c : ℕ → ℕ) (n : ℕ) :
  c 1 = 1 →
  ∀ k ≥ 3, c k = (∑ i in finset.range (k - 1), c (i + 1) * c (k - i - 1)) →
  c n = (factorial (2 * n - 2)) / (factorial n * factorial (n - 1)) :=
sorry

end part_a_part_b_l147_147831


namespace max_value_S_n_l147_147307

noncomputable def arith_seq (a b r: ℕ → ℝ) := ∀ n: ℕ, a n = r + n * b
noncomputable def decreasing_arith_seq (a: ℕ → ℝ) := ∃ (r b: ℕ → ℝ), arith_seq a b r ∧ b < 0
noncomputable def S (a: ℕ → ℝ) (n: ℕ) := ∑ i in finset.range (n + 1), a i

theorem max_value_S_n (a: ℕ → ℝ)
  (h : decreasing_arith_seq a)
  (h₁ : S a 5 = S a 10) :
  ∃ n, n = 7 ∨ n = 8 ∧ S a n = max (S a n) :=
begin
  sorry
end

end max_value_S_n_l147_147307


namespace golden_ratio_isosceles_triangle_golden_ratio_l147_147567

-- Part (a): The Golden Ratio Problem

theorem golden_ratio (y z : ℝ) (h1 : 0 < y) (h2 : y < z) (h3 : (y + z) / z = z / y) : 
  (y / z) = (Real.sqrt 5 - 1) / 2 :=
by sorry

-- Part (b): The Isosceles Triangle and Golden Ratio Problem

theorem isosceles_triangle_golden_ratio (a b : ℝ) (h1 : ∠A = 72°) (h2 : AC = AB) : 
  (a + b) / b = b / a :=
by sorry

end golden_ratio_isosceles_triangle_golden_ratio_l147_147567


namespace max_value_of_y_exists_greatest_y_l147_147374

theorem max_value_of_y (a b y : ℝ) (h1 : a + b + y = 5) (h2 : a * b + b * y + a * y = 3) : y ≤ 13 / 3 :=
begin
  sorry
end

theorem exists_greatest_y (a b y : ℝ) (h1 : a + b + y = 5) (h2 : a * b + b * y + a * y = 3) : 
  ∃ y, y ≤ 13 / 3 ∧ ∀ z, z ≤ 13 / 3 → y = z :=
begin
  sorry
end

end max_value_of_y_exists_greatest_y_l147_147374


namespace base_five_equals_base_b_l147_147281

theorem base_five_equals_base_b : ∃ (b : ℕ), b > 0 ∧ (2 * 5^1 + 4 * 5^0) = (1 * b^2 + 0 * b^1 + 1 * b^0) := by
  sorry

end base_five_equals_base_b_l147_147281


namespace arc_area_sum_l147_147814

theorem arc_area_sum (s : unit_circle_arc) (A B : ℝ) (θ : ℝ) (h1 : arc_in_first_quadrant s)
  (hA : A = area_below s) (hB : B = area_left s) :
  A + B = θ - sin θ :=
sorry

end arc_area_sum_l147_147814


namespace magnitude_proj_u_proj_w_v_correct_l147_147850

variable (v w : ℝ × ℝ × ℝ)
variable (u : ℝ × ℝ × ℝ := (1, 2, 2))
variable (v_dot_w : ℝ := 6) -- v ⋅ w = 6
variable (w_norm : ℝ := 4) -- ||w|| = 4

noncomputable def magnitude_proj_u_proj_w_v : ℝ :=
  let w1 := w.1
  let w2 := w.2.1
  let w3 := w.2.2
  let u1 := u.1
  let u2 := u.2.1
  let u3 := u.2.2
  let u_norm := Math.sqrt (u1*u1 + u2*u2 + u3*u3)
  let proj_v_w := (v_dot_w / (w_norm * w_norm)) * w
  let dot_proj_v_w_u := (proj_v_w.1 * u1 + proj_v_w.2.1 * u2 + proj_v_w.2.2 * u3)
  let proj_u_proj_v_w := (dot_proj_v_w_u / (u_norm * u_norm)) * u
  Math.abs (proj_u_proj_v_w.1*u1 + proj_u_proj_v_w.2.1*u2 + proj_u_proj_v_w.2.2*u3) / 72 * u_norm

theorem magnitude_proj_u_proj_w_v_correct :
  magnitude_proj_u_proj_w_v v w = (Math.abs (3 * (w.1 + 2*w.2.1 + 2*w.2.2)) / 72) := 
sorry

end magnitude_proj_u_proj_w_v_correct_l147_147850


namespace largest_multiple_of_15_less_than_500_l147_147003

theorem largest_multiple_of_15_less_than_500 : ∃ n : ℕ, (n > 0 ∧ 15 * n < 500) ∧ (∀ m : ℕ, m > n → 15 * m >= 500) ∧ 15 * n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l147_147003


namespace value_of_f_15_l147_147507

theorem value_of_f_15 (f : ℝ → ℝ) (h : ∀ x y : ℝ, f(x) + f(3 * x + y) + 6 * x * y = f(4 * x - y) + 3 * x ^ 2 + 2) : f 15 = -110.5 := sorry

end value_of_f_15_l147_147507


namespace tagged_fish_in_second_catch_l147_147783

theorem tagged_fish_in_second_catch 
  (total_fish : ℕ := 3200) 
  (initial_tagged : ℕ := 80) 
  (second_catch : ℕ := 80) 
  (T : ℕ) 
  (h : (T : ℚ) / second_catch = initial_tagged / total_fish) :
  T = 2 :=
by 
  sorry

end tagged_fish_in_second_catch_l147_147783


namespace continued_fraction_value_l147_147253

noncomputable def continued_fraction : ℝ :=
  let rec cf : ℕ → ℝ
  | 0 => 3
  | 1 => 2
  | n => if n % 2 = 0 then 
          2 + (5 / cf (n - 1))
        else 
          3 + (5 / cf (n - 1))
  cf 0

theorem continued_fraction_value : continued_fraction = 5 :=
  sorry

end continued_fraction_value_l147_147253


namespace largest_whole_number_n_l147_147147

theorem largest_whole_number_n : ∃ (n : ℕ), (frac (n / 7) + 1/3 < 1) ∧ ∀ (m : ℕ), (frac (m / 7) + 1/3 < 1) → m ≤ n :=
begin
  use 4,
  split,
  { norm_num },
  { intros m h,
    norm_num at h,
    sorry
  }
end

end largest_whole_number_n_l147_147147


namespace partial_fraction_decomposition_sum_zero_l147_147616

noncomputable theory

open_locale classical

theorem partial_fraction_decomposition_sum_zero
  (A B C D E F : ℝ) :
  (∀ x : ℝ, x ≠ 0 → x ≠ -1 → x ≠ -2 → x ≠ -3 → x ≠ -4 → x ≠ -5 →
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5)) →
  A + B + C + D + E + F = 0 :=
begin
  sorry
end

end partial_fraction_decomposition_sum_zero_l147_147616


namespace solution_set_f_l147_147302

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2^x + x^(1/2) - 1

theorem solution_set_f (x : ℝ) (hx_pos : x > 0) : 
  f x > f (2 * x - 4) ↔ 2 < x ∧ x < 4 :=
sorry

end solution_set_f_l147_147302


namespace smallest_top_block_number_l147_147210

-- Define the pyramid structure and number assignment problem
def block_pyramid : Type := sorry

-- Given conditions:
-- 4 layers, specific numberings, and block support structure.
structure Pyramid :=
  (Layer1 : Fin 16 → ℕ)
  (Layer2 : Fin 9 → ℕ)
  (Layer3 : Fin 4 → ℕ)
  (TopBlock : ℕ)

-- Constraints on block numbers
def is_valid (P : Pyramid) : Prop :=
  -- base layer numbers are from 1 to 16
  (∀ i, 1 ≤ P.Layer1 i ∧ P.Layer1 i ≤ 16) ∧
  -- each above block is the sum of directly underlying neighboring blocks
  (∀ i, P.Layer2 i = P.Layer1 (i * 3) + P.Layer1 (i * 3 + 1) + P.Layer1 (i * 3 + 2)) ∧
  (∀ i, P.Layer3 i = P.Layer2 (i * 3) + P.Layer2 (i * 3 + 1) + P.Layer2 (i * 3 + 2)) ∧
  P.TopBlock = P.Layer3 0 + P.Layer3 1 + P.Layer3 2 + P.Layer3 3

-- Statement of the theorem
theorem smallest_top_block_number : ∃ P : Pyramid, is_valid P ∧ P.TopBlock = ComputedValue := sorry

end smallest_top_block_number_l147_147210


namespace poly_identity_l147_147717

theorem poly_identity (x : ℂ) (h : 1 + x + x^2 + x^3 + x^4 = 0) : 
  (1 + x + x^2 + x^3 + ... + x^1989) = 0 :=
by 
  sorry

end poly_identity_l147_147717


namespace find_a_l147_147425

structure Point2D where
  x : ℝ
  y : ℝ

structure PolarPoint where
  r : ℝ
  theta : ℝ

def toCartesian (p : PolarPoint) : Point2D :=
  ⟨p.r * Real.cos p.theta, p.r * Real.sin p.theta⟩

def lineThroughPoints (A B : Point2D) : ℝ → ℝ := 
  let m := (B.y - A.y) / (B.x - A.x)
  let b := A.y - m * A.x
  λ x, m * x + b

def curve (a : ℝ) (p : Point2D) : Prop :=
  (p.x - a / 2) ^ 2 + p.y ^ 2 = (a / 2) ^ 2

def intersectsExactlyOnce (line : ℝ → ℝ) (a : ℝ) : Prop :=
  let eq := λ x, (x - a / 2)^2 + (line x)^2 - (a / 2)^2
  let Δ := (λ p : ℝ × ℝ × ℝ, p.2.2^2 - 4*p.1*p.2.1) (1, (1 + 1, 1)) -- generic quadratic formula’s discriminant
  Δ = 0

theorem find_a :
  ∃ a : ℝ, a > 0 ∧ let A := toCartesian ⟨√3, π / 6⟩ in
                  let B := toCartesian ⟨3, 0⟩ in
                  let l := lineThroughPoints A B in
                  curve a A ∧ curve a B ∧ intersectsExactlyOnce l a :=
begin
  use 6,
  -- omitted proof, sorry to skip actual proof steps
  sorry,
end

end find_a_l147_147425


namespace partial_fraction_decomposition_l147_147504

-- Define the given rational function
def f (x : ℝ) : ℝ := (306 * x ^ 2 - 450 * x + 162) / ((8 * x - 7) * (5 * x - 4) * (2 * x - 1))

-- Define the partial fractions
def p1 (x : ℝ) : ℝ := 9 / (8 * x - 7)
def p2 (x : ℝ) : ℝ := 6 / (5 * x - 4)
def p3 (x : ℝ) : ℝ := 3 / (2 * x - 1)

-- Define the total partial fraction sum
def partial_sum (x : ℝ) : ℝ := p1 x + p2 x + p3 x

-- The statement that proves the equivalence
theorem partial_fraction_decomposition (x : ℝ) : f x = partial_sum x := 
sorry

end partial_fraction_decomposition_l147_147504


namespace hermia_elected_probability_l147_147837

noncomputable def probability_h ispected_president (n : ℕ) (h : n % 2 = 1) : ℚ :=
  (2^n - 1) / (n * 2^(n-1))

theorem hermia_elected_probability (n : ℕ) (h : n % 2 = 1) :
  let P := probability_h ispected_president n h in 
  hermia_elected_probability = P := 
  sorry

end hermia_elected_probability_l147_147837


namespace obtuse_triangle_has_two_acute_angles_l147_147361

-- Definitions based on conditions
def is_triangle (angles : list ℝ) : Prop :=
  angles.length = 3 ∧ angles.sum = 180

def is_obtuse_triangle (angles : list ℝ) : Prop :=
  is_triangle angles ∧ angles.any (> 90)

def acute_angles_count (angles : list ℝ) : ℝ :=
  angles.count (< 90)

-- Theorem based on conditions and the correct answer
theorem obtuse_triangle_has_two_acute_angles (angles : list ℝ) :
  is_obtuse_triangle angles → acute_angles_count angles = 2 := by
  sorry

end obtuse_triangle_has_two_acute_angles_l147_147361


namespace worker_usual_time_l147_147949

theorem worker_usual_time (S T : ℝ) (D : ℝ) (h1 : D = S * T)
    (h2 : D = (3/4) * S * (T + 8)) : T = 24 :=
by
  sorry

end worker_usual_time_l147_147949


namespace prob_greater_than_2_l147_147893

noncomputable section

open Probability

-- Let's define the random variable ξ following the normal distribution N(0, σ^2).
def ξ (σ : ℝ) : MeasureTheory.ProbabilityMeasure ℝ := MeasureTheory.ProbabilityMeasure.ofReal le_real_of_is_finite measure_space volume

-- Given conditions in the problem
variable {σ : ℝ}
axiom h₁ : MeasureTheory.Measure.map (λ x : ℝ, x) (MeasureTheory.ProbabilityMeasure.toMeasure (ξ σ)) = 
                MeasureTheory.ProbabilityMeasure.toMeasure (MeasureTheory.ProbabilityMeasure.normal 0 σ)
axiom h₂ : ∫ x in -2..2, MeasureTheory.density (MeasureTheory.ProbabilityMeasure.toMeasure (ξ σ)) (λ x, 1) = 0.6

-- The question to prove
theorem prob_greater_than_2 : 
  ∫ x in 2..∞, MeasureTheory.density (MeasureTheory.ProbabilityMeasure.toMeasure (ξ σ)) (λ x, 1) = 0.2 :=
sorry

end prob_greater_than_2_l147_147893


namespace largest_multiple_of_15_less_than_500_l147_147058

theorem largest_multiple_of_15_less_than_500 : 
  ∃ n : ℕ, n * 15 < 500 ∧ ∀ m : ℕ, m * 15 < 500 → m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l147_147058


namespace intersection_problem_l147_147321

noncomputable def A (x : ℝ) : ℝ := x^2 - 4 * x - 5

noncomputable def B (x : ℝ) : ℝ := if x^2 - 1 > 0 then real.log (x^2 - 1) else 0

theorem intersection_problem :
  (∀ x, ∃ y, y = A x) ∧ (∃ x, B x ≠ 0) → 
  ∀ y, y ∈ range A ∩ range B ↔ y ∈ (Icc (-9 : ℝ) (1 : ℝ) ∪ Ioi (1 : ℝ)) :=
sorry

end intersection_problem_l147_147321


namespace largest_multiple_of_15_less_than_500_is_495_l147_147126

-- Define the necessary conditions
def is_multiple_of_15 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 15 * k

def is_less_than_500 (n : ℕ) : Prop :=
  n < 500

-- Problem statement: Prove that the largest positive multiple of 15 less than 500 is 495
theorem largest_multiple_of_15_less_than_500_is_495 : 
  ∀ n : ℕ, is_multiple_of_15 n → is_less_than_500 n → n ≤ 495 := 
by
  sorry

end largest_multiple_of_15_less_than_500_is_495_l147_147126


namespace eval_expression_l147_147681

noncomputable def ceil_sqrt_16_div_9 : ℕ := ⌈Real.sqrt (16 / 9 : ℚ)⌉
noncomputable def ceil_16_div_9 : ℕ := ⌈(16 / 9 : ℚ)⌉
noncomputable def ceil_square_16_div_9 : ℕ := ⌈(16 / 9 : ℚ)^2⌉

theorem eval_expression : ceil_sqrt_16_div_9 + ceil_16_div_9 + ceil_square_16_div_9 = 8 :=
by
  -- The following sorry is a placeholder, indicating that the proof is skipped.
  sorry

end eval_expression_l147_147681


namespace circle_intersects_y_axis_with_constraints_l147_147379

theorem circle_intersects_y_axis_with_constraints {m n : ℝ} 
    (H1 : n = m ^ 2 + 2 * m + 2) 
    (H2 : abs m <= 2) : 
    1 ≤ n ∧ n < 10 :=
sorry

end circle_intersects_y_axis_with_constraints_l147_147379


namespace circle_partition_arcs_l147_147869

theorem circle_partition_arcs (numbers : list ℝ) (h1 : ∀ x ∈ numbers, 0 < x ∧ x ≤ 1) :
  ∃ (arcs : list (list ℝ)), (length arcs = 3) ∧ 
  (∃ sums : list ℝ, (length sums = 3) ∧ (∀ x ∈ sums, 0 ≤ x) ∧ 
    (∀ i j : ℕ, i < 3 → j < 3 → |(sums.nth i).get_or_else 0 - (sums.nth j).get_or_else 0| ≤ 1)) :=
sorry

end circle_partition_arcs_l147_147869


namespace total_surface_area_of_resulting_structure_l147_147577

-- Definitions for the conditions
def bigCube := 12 * 12 * 12
def smallCube := 2 * 2 * 2
def totalSmallCubes := 64
def removedCubes := 7
def remainingCubes := totalSmallCubes - removedCubes
def surfaceAreaPerSmallCube := 24
def extraExposedSurfaceArea := 6
def effectiveSurfaceAreaPerSmallCube := surfaceAreaPerSmallCube + extraExposedSurfaceArea

-- Definition and the main statement of the proof problem.
def totalSurfaceArea := remainingCubes * effectiveSurfaceAreaPerSmallCube

theorem total_surface_area_of_resulting_structure : totalSurfaceArea = 1710 :=
by
  sorry

end total_surface_area_of_resulting_structure_l147_147577


namespace second_alloy_weight_l147_147418

-- Define the problem conditions
def alloy1_chromium_percentage := 0.12
def alloy2_chromium_percentage := 0.08
def weight_alloy1 := 20
def chromium_percentage_new_alloy := 0.09454545454545453
def weight_alloy2 (x : ℝ) := x
def total_weight (x : ℝ) := weight_alloy1 + weight_alloy2 x

-- Define the equation representing the balance of chromium
def chromium_balance_equation (x : ℝ) :=
  (alloy1_chromium_percentage * weight_alloy1) + (alloy2_chromium_percentage * weight_alloy2 x) =
  chromium_percentage_new_alloy * total_weight x

-- The theorem we aim to prove
theorem second_alloy_weight : ∃ x : ℝ, chromium_balance_equation x ∧ x = 35 := sorry

end second_alloy_weight_l147_147418


namespace circle_radius_l147_147541

theorem circle_radius (r₂ : ℝ) : 
  (∃ r₁ : ℝ, r₁ = 5 ∧ (∀ d : ℝ, d = 7 → (d = r₁ + r₂ ∨ d = abs (r₁ - r₂)))) → (r₂ = 2 ∨ r₂ = 12) :=
by
  sorry

end circle_radius_l147_147541


namespace largest_multiple_of_15_less_than_500_l147_147115

theorem largest_multiple_of_15_less_than_500 :
∀ x : ℕ, (∃ k : ℕ, x = 15 * k ∧ 0 < x ∧ x < 500) → x ≤ 495 :=
begin
  sorry
end

end largest_multiple_of_15_less_than_500_l147_147115


namespace sqrt_sum_form_l147_147389

theorem sqrt_sum_form (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : sqrt 6 + 1 / sqrt 6 + sqrt 8 + 1 / sqrt 8 = (a * sqrt 6 + b * sqrt 8) / c)
  (hc_min : ∀ d, ∃ a' b', d > 0 → sqrt 6 + 1 / sqrt 6 + sqrt 8 + 1 / sqrt 8 = (a' * sqrt 6 + b' * sqrt 8) / d  → d ≥ c) :
  a + b + c = 192 :=
by
  sorry

end sqrt_sum_form_l147_147389


namespace min_value_of_f_for_x_gt_1_l147_147887

-- Definition of the function
def f (x : ℝ) : ℝ := (4 * x^2 + 2 * x + 5) / (x^2 + x + 1)

-- The target minimum value
def target_min := (16 - 2 * Real.sqrt 7) / 3

-- Theorem stating the minimum value of the function for x > 1
theorem min_value_of_f_for_x_gt_1 :
  ∀ x : ℝ, x > 1 → f x ≥ target_min :=
sorry

end min_value_of_f_for_x_gt_1_l147_147887


namespace meet_time_l147_147515

theorem meet_time 
  (circumference : ℝ) 
  (deepak_speed_kmph : ℝ) 
  (wife_speed_kmph : ℝ) 
  (deepak_speed_mpm : ℝ := deepak_speed_kmph * 1000 / 60) 
  (wife_speed_mpm : ℝ := wife_speed_kmph * 1000 / 60) 
  (relative_speed : ℝ := deepak_speed_mpm + wife_speed_mpm)
  (time_to_meet : ℝ := circumference / relative_speed) :
  circumference = 660 → 
  deepak_speed_kmph = 4.5 → 
  wife_speed_kmph = 3.75 → 
  time_to_meet = 4.8 :=
by 
  intros h1 h2 h3 
  sorry

end meet_time_l147_147515


namespace flood_damage_in_usd_l147_147591

theorem flood_damage_in_usd (damage_in_cad : ℕ) (exchange_rate : ℚ) (conversion_factor : exchange_rate = 1.25) : 
  damage_in_cad = 50000000 -> 
  (damage_in_cad * (4/5)) = 40000000 :=
by
  intros h
  rw h
  simp
  sorry

end flood_damage_in_usd_l147_147591


namespace ceiling_sum_l147_147649

theorem ceiling_sum :
  let a := 4 / 3
  let b := 16 / 9
  let c := 256 / 81
  ⌈a⌉ + ⌈b⌉ + ⌈c⌉ = 8 := by
  sorry

end ceiling_sum_l147_147649


namespace solve_sqrt_eq_l147_147178

theorem solve_sqrt_eq (x : ℝ) (h1 : 9 + sqrt (15 + 5 * x) ≥ 0) (h2 : 3 + sqrt (3 + x) ≥ 0) :
  sqrt (9 + sqrt (15 + 5 * x)) + sqrt (3 + sqrt (3 + x)) = 5 + sqrt 15 ↔ x = -2 :=
by sorry

end solve_sqrt_eq_l147_147178


namespace function_range_l147_147732

noncomputable def f (x : ℝ) : ℝ := sorry
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem function_range (f : ℝ → ℝ)
    (domain_def : ∀ x, (x ∈ (-1,0) ∪ (0,1)) → f x ∈ ℝ)
    (h_even : is_even f)
    (h_feq0 : f (1 / Real.exp 1) = 0)
    (h_derivative: ∀ x, 0 < x ∧ x < 1 → (1 - x^2) * Real.log (1 - x^2) * (deriv f x) > 2 * x * f x)
    : ∀ x, x ∈ (-1, -1 / Real.exp 1) ∪ (1 / Real.exp 1, 1) → f x < 0 :=
sorry

end function_range_l147_147732


namespace a_plus_d_eq_zero_l147_147443

noncomputable def f (a b c d x : ℝ) : ℝ := (2 * a * x + b) / (c * x + 2 * d)

theorem a_plus_d_eq_zero (a b c d : ℝ) (h : a * b * c * d ≠ 0) (hff : ∀ x, f a b c d (f a b c d x) = 3 * x - 4) : a + d = 0 :=
by
  sorry

end a_plus_d_eq_zero_l147_147443


namespace min_fraction_l147_147313

variables (a_n : ℕ → ℝ) (S_n : ℕ → ℝ)
noncomputable def is_arithmetic_sequence (a_n : ℕ → ℝ) : Prop :=
∀ n, a_n n > 0

noncomputable def sum_first_n_terms (a_n : ℕ → ℝ) (n : ℕ) : ℝ :=
n / 2 * (2 * a_n 1 + (n - 1) * (a_n 2 - a_n 1))

noncomputable def forms_geometric_sequence (x y z : ℝ) : Prop :=
y^2 = x * z

theorem min_fraction (a_n S_n : ℕ → ℝ) (h_arith : is_arithmetic_sequence a_n)
  (h_sum : ∀ n, S_n n = sum_first_n_terms a_n n)
  (h_geom : forms_geometric_sequence (1 / 3) (S_n 3 + 1) (S_n 9)) :
  ∃ S3 S6 : ℝ, minimum (S6 / S3) = 5 :=
sorry

end min_fraction_l147_147313


namespace solve_cubic_inequality_l147_147699

theorem solve_cubic_inequality :
  { x : ℝ | x^3 + x^2 - 7 * x + 6 < 0 } = { x : ℝ | -3 < x ∧ x < 1 ∨ 1 < x ∧ x < 2 } :=
by
  sorry

end solve_cubic_inequality_l147_147699


namespace necessary_condition_of_perpendicular_l147_147959

variables (α β : set ℝ → Prop) (m : ℝ → ℝ → Prop)

-- α represents plane α
-- β represents plane β
-- m represents the line m

-- conditions
variable (h1 : ∀ x y z, α x y z → ¬ β x y z)
variable (h2 : ∀ x y, m x y → α x y 0)

-- question translated to Lean statement
theorem necessary_condition_of_perpendicular
  (h3 : ∀ x y, m x y → β x y 0): (∀ x y z, α x y z → β x y z) :=
sorry

end necessary_condition_of_perpendicular_l147_147959


namespace sqrt_expr_sum_equality_l147_147397

theorem sqrt_expr_sum_equality :
  let a := 14
  let b := 3
  let c := 2
  (\sqrt(6) + 1/\sqrt(6) + \sqrt(8) + 1/\sqrt(8) = (a * \sqrt(6) + b * \sqrt(8)) / c) :=
begin
  sorry
end

lemma abc_sum :
  let a := 14
  let b := 3
  let c := 2
  (a + b + c = 19) :=
begin
  sorry
end

end sqrt_expr_sum_equality_l147_147397


namespace line_circle_shortest_chord_l147_147936

noncomputable def shortest_chord_slope (x y : ℝ) (C : x = y - 1 ∧ (x - 2)^2 + (y - 1)^2 = 5) : Prop :=
  let k := -((x - 2)/(y - 1)) in ∃ l : Line, l.slope = k ∧ l.y_intercept = 2 − k

theorem line_circle_shortest_chord (k : ℝ) (l : Line) (C : Circle) (h1 : l.equation = y = k * (x - 1) + 2)
  (h2 : C.equation = (x - 2)^2 + (y - 1)^2 = 5) :
  shortest_chord_slope x y C :=
sorry

end line_circle_shortest_chord_l147_147936


namespace cube_root_of_b_minus_a_is_neg_one_l147_147378

theorem cube_root_of_b_minus_a_is_neg_one {a b : ℝ} (h : b = a - 1) : real.cbrt (b - a) = -1 := by
  sorry

end cube_root_of_b_minus_a_is_neg_one_l147_147378


namespace find_fraction_l147_147190

theorem find_fraction (x : ℝ) (H_pos : x > 0) (H_eq : x = 1) : 
  ∃ f : ℝ, f * x = (144 / 216) * (1 / x) ∧ f = 2 / 3 :=
by
  use 2 / 3
  split
  · rw [H_eq, mul_one, div_self]
  · exact rfl
  done

end find_fraction_l147_147190


namespace first_sequence_correct_second_sequence_correct_l147_147710

theorem first_sequence_correct (a1 a2 a3 a4 a5 : ℕ) (h1 : a1 = 12) (h2 : a2 = a1 + 4) (h3 : a3 = a2 + 4) (h4 : a4 = a3 + 4) (h5 : a5 = a4 + 4) :
  a4 = 24 ∧ a5 = 28 :=
by sorry

theorem second_sequence_correct (b1 b2 b3 b4 b5 : ℕ) (h1 : b1 = 2) (h2 : b2 = b1 * 2) (h3 : b3 = b2 * 2) (h4 : b4 = b3 * 2) (h5 : b5 = b4 * 2) :
  b4 = 16 ∧ b5 = 32 :=
by sorry

end first_sequence_correct_second_sequence_correct_l147_147710


namespace largest_multiple_of_15_less_than_500_is_495_l147_147128

-- Define the necessary conditions
def is_multiple_of_15 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 15 * k

def is_less_than_500 (n : ℕ) : Prop :=
  n < 500

-- Problem statement: Prove that the largest positive multiple of 15 less than 500 is 495
theorem largest_multiple_of_15_less_than_500_is_495 : 
  ∀ n : ℕ, is_multiple_of_15 n → is_less_than_500 n → n ≤ 495 := 
by
  sorry

end largest_multiple_of_15_less_than_500_is_495_l147_147128


namespace circle_standard_eq_l147_147305

noncomputable def M : ℝ × ℝ := (-2, 2)
noncomputable def N : ℝ × ℝ := (-1, -1)

def on_line (C : ℝ × ℝ) : Prop := C.1 - C.2 - 1 = 0

def circle_eq (C : ℝ × ℝ) (r : ℝ) : ℝ × ℝ → Prop :=
  λ P, (P.1 - C.1)^2 + (P.2 - C.2)^2 = r^2

theorem circle_standard_eq :
  ∃ C r, on_line C ∧ circle_eq C r M ∧ circle_eq C r N ∧ circle_eq C 5 = λ P, (P.1 - 3)^2 + (P.2 - 2)^2 = 25 :=
sorry

end circle_standard_eq_l147_147305


namespace cyclist_wait_time_is_20_minutes_l147_147169

noncomputable def calculate_cyclist_wait_time
  (hiker_speed : ℝ)
  (cyclist_speed : ℝ)
  (cyclist_time_minutes : ℝ) : ℝ :=
let cyclist_time_hours := cyclist_time_minutes / 60 in
let distance_by_cyclist := cyclist_speed * cyclist_time_hours in
let time_hiker_hours := distance_by_cyclist / hiker_speed in
time_hiker_hours * 60

-- The given parameters from the problem
def hiker_speed := 5   -- miles per hour
def cyclist_speed := 20 -- miles per hour
def cyclist_time := 5   -- minutes

theorem cyclist_wait_time_is_20_minutes :
  calculate_cyclist_wait_time hiker_speed cyclist_speed cyclist_time = 20 := 
by 
  sorry

end cyclist_wait_time_is_20_minutes_l147_147169


namespace sin_A_of_triangle_l147_147780

-- Definitions corresponding to conditions
def angle_B := Real.pi / 4
def height_to_side_BC (a : ℝ) (b c : ℝ) : Prop := c = (Real.sqrt 2 / 3) * a

-- The main theorem statement
theorem sin_A_of_triangle 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (h₁ : B = angle_B)
  (h₂ : height_to_side_BC a b c) :
  Real.sin A = 3 * Real.sqrt 10 / 10 :=
by
  sorry

end sin_A_of_triangle_l147_147780


namespace unique_n_l147_147286

def sum_condition (n : ℕ) (a : Fin n → ℝ) : Prop :=
  ∑ i, a i = 17

def S (n : ℕ) (a : Fin n → ℝ) : ℝ :=
  ∑ i in Finset.range n + 1, Real.sqrt ((2 * (i : ℕ) - 1)^2 + (a ⟨i - 1, Nat.sub_lt i.pos⟩)^2)

theorem unique_n (n : ℕ) (a : Fin n → ℝ) (h : sum_condition n a) (hs : S n a ∈ ℤ) : n = 12 :=
sorry

end unique_n_l147_147286


namespace radius_of_circle_complex_roots_l147_147228

noncomputable def radius_of_complex_roots_circle : ℝ :=
  let z := ℂ in 
  if h : (∀ z : ℂ, (z + 2)^6 = 64 * (z)^6 → ∥z + 2∥ = 2 * ∥z∥) then
    (4 / 3)
  else
    0  -- placeholder, actual proof logic not required

theorem radius_of_circle_complex_roots :
  (∀ z : ℂ, (z + 2)^6 = 64 * (z)^6 → ∥z + 2∥ = 2 * ∥z∥) → radius_of_complex_roots_circle = (4 / 3) :=
by sorry

end radius_of_circle_complex_roots_l147_147228


namespace largest_multiple_of_15_less_than_500_l147_147099

theorem largest_multiple_of_15_less_than_500 : ∃ (x : ℕ), (x < 500) ∧ (x % 15 = 0) ∧ ∀ (y : ℕ), (y < 500) ∧ (y % 15 = 0) → y ≤ x :=
begin
  use 495,
  split,
  { norm_num },
  split,
  { norm_num },
  intros y hy1 hy2,
  sorry,
end

end largest_multiple_of_15_less_than_500_l147_147099


namespace largest_multiple_of_15_less_than_500_l147_147023

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), n < 500 ∧ 15 ∣ n ∧ ∀ (m : ℕ), m < 500 ∧ 15 ∣ m -> m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l147_147023


namespace isosceles_right_triangle_area_l147_147514

theorem isosceles_right_triangle_area (h : ℝ) (area : ℝ) (hypotenuse_condition : h = 6 * Real.sqrt 2) : 
  area = 18 :=
  sorry

end isosceles_right_triangle_area_l147_147514


namespace evaluate_expression_l147_147664

theorem evaluate_expression :
  let x := (16 : ℚ) / 9
  in ⌈(√x)⌉ + ⌈x⌉ + ⌈x^2⌉ = 8 :=
by
  let x := (16 : ℚ) / 9
  sorry

end evaluate_expression_l147_147664


namespace intersection_of_sets_l147_147319

-- Definition of sets A and B as conditions in the mathematical problem.
def setA : Set ℝ := {x | x^2 - 4 * x - 5 ≤ 0}
def setB : Set ℝ := {x | real.log x / real.log 2 < 2}

-- The theorem stating that the intersection of sets A and B equals (0, 4).
theorem intersection_of_sets : (setA ∩ setB) = {x | 0 < x ∧ x < 4} :=
by sorry

end intersection_of_sets_l147_147319


namespace range_of_m_l147_147347

theorem range_of_m {m : ℝ} (f : ℝ → ℝ) (h₀ : ∀ x ∈ Icc m (m + 1), f x < 0) :
  (f = λ x, x^2 + m * x - 1) → (- real.sqrt 2 / 2 < m ∧ m < 0) :=
by
  intros hf
  sorry

end range_of_m_l147_147347


namespace ceiling_sum_evaluation_l147_147692

noncomputable def evaluateCeilingSum : ℝ := 
  ⌈Real.sqrt (16 / 9)⌉ + ⌈(16 / 9)⌉ + ⌈((16 / 9) ^ 2)⌉ 

theorem ceiling_sum_evaluation : evaluateCeilingSum = 8 := by
  sorry

end ceiling_sum_evaluation_l147_147692


namespace min_value_of_x_l147_147960

-- Definitions for the conditions given in the problem
def men := 4
def women (x : ℕ) := x
def min_x := 594

-- Definition of the probability p
def C (n k : ℕ) : ℕ := sorry -- Define the binomial coefficient properly

def probability (x : ℕ) : ℚ :=
  (2 * (C (x+1) 2) + (x + 1)) /
  (C (x + 1) 3 + 3 * (C (x + 1) 2) + (x + 1))

-- The theorem statement to prove
theorem min_value_of_x (x : ℕ) : probability x ≤ 1 / 100 →  x = min_x := 
by
  sorry

end min_value_of_x_l147_147960


namespace period_of_tangent_transform_l147_147149

noncomputable def period_of_transformed_tangent :=
  let f := λ x : ℝ, tan (x / 3 + π / 4)
  f.period = 3 * π

theorem period_of_tangent_transform : period_of_transformed_tangent :=
sorry

end period_of_tangent_transform_l147_147149


namespace find_m_l147_147589

noncomputable def cylinder_wedge_volume (diameter : ℝ) (angle : ℝ) : ℝ :=
  let r := diameter / 2
  let height := diameter
  let cylinder_volume := π * r^2 * height
  (1 / 2) * (angle / 360) * cylinder_volume

theorem find_m : ∃ (m : ℤ), cylinder_wedge_volume 20 60 = m * π ∧ m = 333 :=
by
  use 333
  simp [cylinder_wedge_volume, Real.pi, show 20 / 2 = 10, by norm_num, show (60 : ℝ) / 360 = 1 / 6, by norm_num]
  ring_nf
  split
  · sorry
  · refl

end find_m_l147_147589


namespace rhombus_unique_diagonal_property_rectangle_not_diagonal_property_l147_147516

def is_rhombus (a b c d : Point) : Prop :=
  (a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a) ∧
  (dist a b = dist b c ∧ dist b c = dist c d ∧ dist c d = dist d_a) ∧
  dist a c = dist b d

def is_rectangle (a b c d : Point) : Prop :=
  (a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a) ∧
  (dist a b = dist c d ∧ dist a d = dist b c)

def diagonals_perpendicular (a b c d : Point) : Prop :=
  ((lineToSeg a c).IsPerp (lineToSeg b d))


theorem rhombus_unique_diagonal_property (a b c d : Point) :
  is_rhombus a b c d → diagonals_perpendicular a b c d := sorry

theorem rectangle_not_diagonal_property (a b c d : Point) :
  is_rectangle a b c d → ¬diagonals_perpendicular a b c d := sorry

end rhombus_unique_diagonal_property_rectangle_not_diagonal_property_l147_147516


namespace hemisphere_surface_area_l147_147526

theorem hemisphere_surface_area (r : ℝ) (h_r : r = 5) :
  let sphere_surface_area := 4 * Real.pi * r^2
  let hemisphere_curved_area := sphere_surface_area / 2
  let base_area := Real.pi * r^2
  (hemisphere_curved_area + base_area) = 75 * Real.pi :=
by
  -- Here we are setting r to 5 to use the given condition
  have r_eq_5 : r = 5 := h_r
  -- The surface area of sphere with radius r
  let sphere_surface_area := 4 * Real.pi * r^2
  -- Curved surface area of the hemisphere
  let hemisphere_curved_area := sphere_surface_area / 2
  -- Base area of the hemisphere
  let base_area := Real.pi * r^2
  -- Total surface area of the hemisphere
  let total_surface_area := hemisphere_curved_area + base_area
  calc
    total_surface_area
    = (4 * Real.pi * 5^2 / 2) + (Real.pi * 5^2) : by rw [h_r, r_eq_5]
    ... = 75 * Real.pi : sorry

end hemisphere_surface_area_l147_147526


namespace smallest_possible_N_l147_147972

theorem smallest_possible_N (table_size N : ℕ) (h_table_size : table_size = 72) :
  (∀ seating : Finset ℕ, (seating.card = N) → (seating ⊆ Finset.range table_size) →
    ∃ i ∈ Finset.range table_size, (seating = ∅ ∨ ∃ j, (j ∈ seating) ∧ (i = (j + 1) % table_size ∨ i = (j - 1) % table_size)))
  → N = 18 :=
by sorry

end smallest_possible_N_l147_147972


namespace cone_height_ratio_l147_147600

theorem cone_height_ratio (r h : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) 
  (rolls_19_times : 19 * 2 * Real.pi * r = 2 * Real.pi * Real.sqrt (r^2 + h^2)) :
  h / r = 6 * Real.sqrt 10 :=
by
  -- problem setup and mathematical manipulations
  sorry

end cone_height_ratio_l147_147600


namespace fewest_colored_paper_l147_147809
   
   /-- Jungkook, Hoseok, and Seokjin shared colored paper. 
       Jungkook took 10 cards, Hoseok took 7, and Seokjin took 2 less than Jungkook. 
       Prove that Hoseok took the fewest pieces of colored paper. -/
   theorem fewest_colored_paper 
       (Jungkook Hoseok Seokjin : ℕ)
       (hj : Jungkook = 10)
       (hh : Hoseok = 7)
       (hs : Seokjin = Jungkook - 2) :
       Hoseok < Jungkook ∧ Hoseok < Seokjin :=
   by
     sorry
   
end fewest_colored_paper_l147_147809


namespace profit_share_b_l147_147568

theorem profit_share_b (P : ℝ) (Pa : ℝ) (Pb : ℝ) (Pc : ℝ)
  (h_cap_a : 8000) (h_cap_b : 10000) (h_cap_c : 12000)
  (h_ratio : Pa / Pb / Pc = 4 / 5 / 6)
  (h_diff : Pc - Pa = 600) :
  Pb = 1500 :=
by 
  sorry

end profit_share_b_l147_147568


namespace valid_triangle_sides_l147_147995

theorem valid_triangle_sides (x : ℕ) (h1 : 3 < x * x) (h2 : x * x < 19) : x = 2 ∨ x = 3 ∨ x = 4 :=
by sorry

end valid_triangle_sides_l147_147995


namespace largest_multiple_of_15_less_than_500_l147_147010

theorem largest_multiple_of_15_less_than_500 : ∃ n : ℕ, (n > 0 ∧ 15 * n < 500) ∧ (∀ m : ℕ, m > n → 15 * m >= 500) ∧ 15 * n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l147_147010


namespace total_amount_received_l147_147581

variable (A B : Type) [LinearOrder A] [LinearOrder B]

def A_rate : ℝ := 1 / 30
def B_rate : ℝ := 1 / 20
def B_share : ℝ := 600

theorem total_amount_received (total_amount : ℝ) : 
  (3 / 5) * total_amount = B_share → 
  total_amount = 1000 :=
by
  intro h
  sorry

end total_amount_received_l147_147581


namespace largest_multiple_of_15_less_than_500_l147_147116

theorem largest_multiple_of_15_less_than_500 :
∀ x : ℕ, (∃ k : ℕ, x = 15 * k ∧ 0 < x ∧ x < 500) → x ≤ 495 :=
begin
  sorry
end

end largest_multiple_of_15_less_than_500_l147_147116


namespace largest_multiple_of_15_less_than_500_l147_147112

theorem largest_multiple_of_15_less_than_500 :
∀ x : ℕ, (∃ k : ℕ, x = 15 * k ∧ 0 < x ∧ x < 500) → x ≤ 495 :=
begin
  sorry
end

end largest_multiple_of_15_less_than_500_l147_147112


namespace hockey_league_games_l147_147906

theorem hockey_league_games :
  let teams := 30 in
  let matches_per_team := 20 in
  let total_games := (teams * (teams - 1) * matches_per_team) / 2 in
  let games_with_scoring_system_A := 0.60 * total_games in
  let games_with_scoring_system_B := 0.40 * total_games in
  total_games = 17400 ∧ games_with_scoring_system_A = 10440 ∧ games_with_scoring_system_B = 6960 :=
by
  sorry

end hockey_league_games_l147_147906


namespace tangent_line_angle_APB_vector_sum_max_line_AB_through_fixed_point_final_result_l147_147719

-- Definitions for points A, B on circle O and line l
def O (x y : ℝ) := x^2 + y^2 = 1
def l (x y : ℝ) := x + y - 2 = 0

-- Tangent Line
theorem tangent_line (x y : ℝ) (hx : O x y) : l x y → |1| / (real.sqrt (1^2 + 1^2)) = 1 :=
by sorry

-- Right Angle Condition
theorem angle_APB (A B : ℝ × ℝ) (hA : O A.1 A.2) (hB : O B.1 B.2) (P : ℝ × ℝ) : ¬ (l P.1 P.2 ∧ angle A P B = π / 2) :=
by sorry

-- Vector Sum Condition
theorem vector_sum_max (A B : ℝ × ℝ) (hA : O A.1 A.2) (hB : O B.1 B.2) (P : ℝ × ℝ) : |(A.1 - P.1 + B.1 - P.1)| = (|√3| + 1) → max_sum :=
by sorry

-- Line AB Passing Through Fixed Point
theorem line_AB_through_fixed_point (A B : ℝ × ℝ) (hA : O A.1 A.2) (hB : O B.1 B.2) (P : ℝ × ℝ) (tangentPA : O P.1 P.2) (tangentPB : O P.1 P.2) :
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = (1/2, 1/2) :=
by sorry

-- Final result combining correct options
theorem final_result : (tangent_line ∧ line_AB_through_fixed_point) :=
by sorry

end tangent_line_angle_APB_vector_sum_max_line_AB_through_fixed_point_final_result_l147_147719


namespace evaluate_x_squared_plus_y_squared_l147_147328

theorem evaluate_x_squared_plus_y_squared (x y : ℝ) (h₁ : 3 * x + y = 20) (h₂ : 4 * x + y = 25) :
  x^2 + y^2 = 50 :=
sorry

end evaluate_x_squared_plus_y_squared_l147_147328


namespace largest_multiple_of_15_less_than_500_l147_147100

theorem largest_multiple_of_15_less_than_500 : ∃ (x : ℕ), (x < 500) ∧ (x % 15 = 0) ∧ ∀ (y : ℕ), (y < 500) ∧ (y % 15 = 0) → y ≤ x :=
begin
  use 495,
  split,
  { norm_num },
  split,
  { norm_num },
  intros y hy1 hy2,
  sorry,
end

end largest_multiple_of_15_less_than_500_l147_147100


namespace increase_80_by_50_percent_l147_147961

theorem increase_80_by_50_percent : 
  let original_number := 80
  let percentage_increase := 0.5
  let increase := original_number * percentage_increase
  let final_number := original_number + increase
  final_number = 120 := 
by 
  sorry

end increase_80_by_50_percent_l147_147961


namespace largest_multiple_of_15_less_than_500_l147_147018

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), n < 500 ∧ 15 ∣ n ∧ ∀ (m : ℕ), m < 500 ∧ 15 ∣ m -> m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l147_147018


namespace sum_of_coordinates_eq_60_l147_147714

theorem sum_of_coordinates_eq_60 :
  ∀ (P : ℕ) (Q : ℕ) 
    (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ),
    (P = 4 ∧ Q = 12 ∧ 
    (y1 = 10 + P ∨ y1 = 10 - P) ∧
    (y2 = 10 + P ∨ y2 = 10 - P) ∧
    (y3 = 10 + P ∨ y3 = 10 - P) ∧
    (y4 = 10 + P ∨ y4 = 10 - P) ∧
    (real.sqrt ((x1 - 5)^2 + (y1 - 10)^2) = Q) ∧ 
    (real.sqrt ((x2 - 5)^2 + (y2 - 10)^2) = Q) ∧
    (real.sqrt ((x3 - 5)^2 + (y3 - 10)^2) = Q) ∧
    (real.sqrt ((x4 - 5)^2 + (y4 - 10)^2) = Q)
    ) → (x1 + x2 + x3 + x4 + y1 + y2 + y3 + y4 = 60) :=
  sorry

end sum_of_coordinates_eq_60_l147_147714


namespace geometric_sequence_log_sum_l147_147285

theorem geometric_sequence_log_sum (a : ℕ → ℝ) (r : ℝ)
  (h_geo : ∀ n, a (n + 1) = a n * r)
  (h_pos : ∀ n, a n > 0)
  (h_condition : a 5 * a 6 + a 4 * a 7 = 18) :
  log 3 (a 1) + log 3 (a 2) + log 3 (a 3) + log 3 (a 4) +
  log 3 (a 5) + log 3 (a 6) + log 3 (a 7) + log 3 (a 8) +
  log 3 (a 9) + log 3 (a 10) = 10 :=
by 
  sorry

end geometric_sequence_log_sum_l147_147285


namespace man_speed_in_still_water_l147_147978

-- Definitions of the conditions in a)
def speed_current : ℝ := 3  -- 3 kmph
def distance_downstream : ℝ := 0.08  -- in kilometers
def time_downstream : ℝ := 15.99872010239181 / 3600  -- in hours

-- Target statement to prove the man's speed in still water
theorem man_speed_in_still_water : 
  let speed_downstream := distance_downstream / time_downstream in
  let speed_still_water := speed_downstream - speed_current in
  speed_still_water = 15 :=
by
  sorry  -- Proof to be completed

end man_speed_in_still_water_l147_147978


namespace proof_problem_l147_147676

def sqrt_frac : ℚ := real.sqrt (16 / 9)
def frac : ℚ := 16 / 9
def square_frac : ℚ := frac * frac

def ceil_sqrt_frac : ℤ := ⌈sqrt_frac⌉.to_int
def ceil_frac : ℤ := ⌈frac⌉.to_int
def ceil_square_frac : ℤ := ⌈square_frac⌉.to_int

theorem proof_problem :
  ceil_sqrt_frac + ceil_frac + ceil_square_frac = 8 :=
by
  -- Placeholder for the actual proof.
  sorry

end proof_problem_l147_147676


namespace find_xyz_values_l147_147355

theorem find_xyz_values (x y z : ℝ) (h₁ : x + y + z = Real.pi) (h₂ : x ≥ 0) (h₃ : y ≥ 0) (h₄ : z ≥ 0) :
    (x = Real.pi ∧ y = 0 ∧ z = 0) ∨
    (x = 0 ∧ y = Real.pi ∧ z = 0) ∨
    (x = 0 ∧ y = 0 ∧ z = Real.pi) ∨
    (x = Real.pi / 6 ∧ y = Real.pi / 3 ∧ z = Real.pi / 2) :=
sorry

end find_xyz_values_l147_147355


namespace fifteen_pretty_sum_l147_147625

theorem fifteen_pretty_sum (T : ℕ) :
  (∀ (n : ℕ), (15 ∣ n ∧ nat.totient n = 15) → n < 2023 → False) → 
  T = 0 → T / 15 = 0 :=
by
  intro fifteen_pretty_cond T_def
  rw T_def
  exact nat.zero_div 15

end fifteen_pretty_sum_l147_147625


namespace f_zero_solution_set_monotonic_intervals_and_extremes_l147_147359

noncomputable def vector_a := (1 : ℝ, real.sqrt 3)
noncomputable def vector_b (x : ℝ) := (real.cos x, real.sin x)
noncomputable def f (x : ℝ) := (vector_a.1 * real.cos x + vector_a.2 * real.sin x) - 1

-- Part 1
theorem f_zero_solution_set :
  {x | f x = 0} = {x | ∃ (k : ℤ), x = 2 * k * real.pi ∨ x = (2 / 3) * real.pi + 2 * k * real.pi} :=
sorry

-- Part 2
theorem monotonic_intervals_and_extremes (x : ℝ) (h : 0 ≤ x ∧ x ≤ real.pi / 2) :
  (∀ y, 0 ≤ y ∧ y ≤ real.pi / 3 → f y ≤ f (y + real.pi / 3)) ∧
  (∀ y, real.pi / 3 ≤ y ∧ y ≤ real.pi / 2 → f y ≥ f (y + (real.pi / 6))) ∧
  (f 0 = 0) ∧ (f (real.pi / 3) = 1) :=
sorry

end f_zero_solution_set_monotonic_intervals_and_extremes_l147_147359


namespace number_of_ordered_pairs_l147_147438

noncomputable def phi (n : ℕ) : ℕ := 
  if h : n > 0 then (Fintype.card { x : ℕ // 0 < x ∧ x ≤ n ∧ GCD x n = 1 }) else 0

def number_of_divisors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d, d > 0 ∧ n % d = 0).card

def N (p : ℕ) : ℕ :=
   ((Finset.range (p * (p-1) + 1)).filter (λ ⟨x, y⟩, 
     1 ≤ x ∧ x ≤ p * (p-1) ∧ 
     1 ≤ y ∧ y ≤ p * (p-1) ∧ 
     x ^ y % p = 1 ∧ y ^ x % p = 1)).card

theorem number_of_ordered_pairs (p : ℕ) [Fact p.Prime] :
  ((phi(p-1) * number_of_divisors(p-1))^2 ≤ N(p) ∧ N(p) ≤ ((p-1) * number_of_divisors(p-1))^2) :=
by sorry

end number_of_ordered_pairs_l147_147438


namespace sqrt_expr_sum_equality_l147_147395

theorem sqrt_expr_sum_equality :
  let a := 14
  let b := 3
  let c := 2
  (\sqrt(6) + 1/\sqrt(6) + \sqrt(8) + 1/\sqrt(8) = (a * \sqrt(6) + b * \sqrt(8)) / c) :=
begin
  sorry
end

lemma abc_sum :
  let a := 14
  let b := 3
  let c := 2
  (a + b + c = 19) :=
begin
  sorry
end

end sqrt_expr_sum_equality_l147_147395


namespace largest_multiple_of_15_less_than_500_l147_147047

theorem largest_multiple_of_15_less_than_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ (∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x) := sorry

end largest_multiple_of_15_less_than_500_l147_147047


namespace find_k_find_t_l147_147422

def vector := (ℝ × ℝ)
def a : vector := (1, -2)
def b : vector := (3, 4)

def parallel (v1 v2 : vector) : Prop :=
  ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

def orthogonal (v1 v2 : vector) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem find_k (k : ℝ) :
  parallel (3 • a - b) (a + k • b) → k = -1/3 :=
by
  sorry

theorem find_t (t : ℝ) :
  orthogonal (a - t • b) b → t = -1/5 :=
by
  sorry


end find_k_find_t_l147_147422


namespace find_ordered_triple_l147_147428

variables {D E F J : Type} [AddCommGroup D] [AddCommGroup E] [AddCommGroup F] [AddCommGroup J]
variables [Module ℚ D] [Module ℚ E] [Module ℚ F] [Module ℚ J]
variables (d e f : ℚ) (p q r : ℚ)
variables (over_D : D) (over_E : E) (over_F : F) (over_J : J)

-- Definitions based on the given conditions.
def lengths := (d = 8) ∧ (e = 10) ∧ (f = 6)
def incenter_linearity := over_J = p • over_D + q • over_E + r • over_F
def sum_coeff := p + q + r = 1

-- The target theorem to be proved:
theorem find_ordered_triple (h_lengths : lengths d e f) (h_sum : sum_coeff p q r) : 
  incenter_linearity over_D over_E over_F over_J (1/3) (5/12) (1/4) :=
by
  sorry

end find_ordered_triple_l147_147428


namespace problem_part1_problem_part2_l147_147742

noncomputable def f1 : ℝ × ℝ := (-(2 : ℝ).sqrt, 0)
noncomputable def f2 : ℝ × ℝ := ((2 : ℝ).sqrt, 0)
noncomputable def pQ : ℝ × ℝ := (4*(2:ℝ).sqrt / 3, 1 / 3)
noncomputable def p := (2: ℝ).sqrt / 24

theorem problem_part1 :
  let Q := pQ in
  angle f1 Q f2 = real.pi / 3 →
  |Q.1 - f1.1| + |Q.1 - f2.1| = 4 →
  1/2 * (|Q.1 - f1.1| * |Q.1 - f2.1| * (3: ℝ).sqrt / 2) = (3: ℝ).sqrt / 3 :=
sorry

theorem problem_part2 :
  let Q := pQ in
  let p := (2 : ℝ).sqrt / 24 in
  (Q.2^2 = p * Q.1) →
  y^2 = p * x :=
sorry

end problem_part1_problem_part2_l147_147742


namespace range_of_c_l147_147797

-- Define the circle and the line in Lean
def circle : ℝ → ℝ → Prop := λ x y, x^2 + y^2 = 4
def line (c : ℝ) : ℝ → ℝ → Prop := λ x y, 4 * x - 3 * y + c = 0

-- Define conditions for the distance
def distance_from_origin_to_line (c : ℝ) : ℝ := |c| / 5

-- Define the property to test the distance constraint
def distance_constraint (c : ℝ) : Prop := distance_from_origin_to_line c < 1

-- Range of c values to be proven
theorem range_of_c (c : ℝ) (h : ∀ x y, circle x y → distance_constraint c) : -5 < c ∧ c < 5 := by
  sorry

end range_of_c_l147_147797


namespace find_x_l147_147299

-- Definitions based on given conditions
def vector_a : ℝ × ℝ × ℝ := (2, -1, 3)
def vector_b (x : ℝ) : ℝ × ℝ × ℝ := (-4, 2, 3 - x)
def perpendicular (v w : ℝ × ℝ × ℝ) := v.1 * w.1 + v.2 * w.2 + v.3 * w.3 = 0

-- The proof statement
theorem find_x (x : ℝ) (h : perpendicular vector_a (vector_b x)) : x = -1/3 :=
by
  unfold perpendicular at h
  unfold vector_a at h
  unfold vector_b at h
  sorry -- Proof skipped

end find_x_l147_147299


namespace find_k_l147_147196

-- Defining the points
def point1 : ℝ × ℝ := (6, 8)
def point3 : ℝ × ℝ := (-10, 4)

-- Defining the variable point
variables (k : ℝ)
def point2 : ℝ × ℝ := (-2, k)

-- Definition of the slope calculation based on the points being collinear
def slope (p1 p2 : ℝ × ℝ) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)

-- The theorem stating the condition that the slopes must be equal
theorem find_k (k : ℝ) : slope point1 point2 = slope point2 point3 → k = 6 :=
by
  sorry

end find_k_l147_147196


namespace largest_multiple_of_15_less_than_500_is_495_l147_147118

-- Define the necessary conditions
def is_multiple_of_15 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 15 * k

def is_less_than_500 (n : ℕ) : Prop :=
  n < 500

-- Problem statement: Prove that the largest positive multiple of 15 less than 500 is 495
theorem largest_multiple_of_15_less_than_500_is_495 : 
  ∀ n : ℕ, is_multiple_of_15 n → is_less_than_500 n → n ≤ 495 := 
by
  sorry

end largest_multiple_of_15_less_than_500_is_495_l147_147118


namespace curve_is_line_l147_147706

-- Let theta be the angle in polar coordinates
variable {θ : ℝ}

-- Define the condition given in the problem
def condition : Prop := θ = π / 4

-- Prove that the curve defined by this condition is a line
theorem curve_is_line (h : condition) : ∃ (m b : ℝ), ∀ x y : ℝ, y = m * x + b :=
by
  sorry

end curve_is_line_l147_147706


namespace valid_votes_correct_l147_147794

def total_votes : ℝ := 12000
def invalid_votes_perc : ℝ := 0.20
def blank_votes_perc : ℝ := 0.05
def double_votes_perc : ℝ := 0.03

def invalid_votes : ℝ := invalid_votes_perc * total_votes
def blank_votes : ℝ := blank_votes_perc * total_votes
def double_votes : ℝ := double_votes_perc * total_votes

def non_valid_votes : ℝ := invalid_votes + blank_votes + double_votes
def valid_votes : ℝ := total_votes - non_valid_votes

def cand1_perc_valid : ℝ := 0.45
def cand2_perc_valid : ℝ := 0.35
def cand3_perc_valid : ℝ := 1 - (cand1_perc_valid + cand2_perc_valid)

def cand1_votes : ℝ := cand1_perc_valid * valid_votes
def cand2_votes : ℝ := cand2_perc_valid * valid_votes
def cand3_votes : ℝ := cand3_perc_valid * valid_votes

theorem valid_votes_correct :
  cand1_votes = 3888 ∧ cand2_votes = 3024 ∧ cand3_votes = 1728 :=
by
  unfold total_votes invalid_votes_perc blank_votes_perc double_votes_perc
         invalid_votes blank_votes double_votes non_valid_votes valid_votes 
         cand1_perc_valid cand2_perc_valid cand3_perc_valid
         cand1_votes cand2_votes cand3_votes
  simp
  norm_num
  exact ⟨rfl, rfl, rfl⟩

end valid_votes_correct_l147_147794


namespace max_a4a7_value_l147_147326

variable {a : ℕ → ℝ}
variable {d : ℝ}

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop := ∀ n m : ℕ, a (n + 1) = a n + d

-- Given conditions
def given_conditions (a : ℕ → ℝ) (d : ℝ) : Prop := 
  arithmetic_sequence a d ∧ a 5 = 4 -- a6 = 4 so we use index 5 since Lean is 0-indexed

-- Define the product a4 * a7
def a4a7_product (a : ℕ → ℝ) (d : ℝ) : ℝ := (a 5 - 2 * d) * (a 5 + d)

-- The maximum value of a4 * a7
def max_a4a7 (a : ℕ → ℝ) (d : ℝ) : ℝ := 18

-- The proof problem statement
theorem max_a4a7_value (a : ℕ → ℝ) (d : ℝ) :
  given_conditions a d → a4a7_product a d = max_a4a7 a d :=
by
  sorry

end max_a4a7_value_l147_147326


namespace largest_prime_factor_l147_147276

theorem largest_prime_factor (a b c : ℕ) (h1 : a = 20) (h2 : b = 15) (h3 : c = 10) :
  Nat.gcd (a * a * a) (b * b * b * b) = 1 ∧ Nat.gcd (a * a * a) (c * c * c * c * c) = 1 ∧ Nat.gcd (b * b * b * b) (c * c * c * c * c) = 1 →
  Nat.largest_prime_factor (a ^ 3 + b ^ 4 - c ^ 5) = 13 :=
by
  sorry

end largest_prime_factor_l147_147276


namespace matchstick_equality_l147_147293

theorem matchstick_equality :
  abs ((22 : ℝ) / 7 - Real.pi) < 0.1 := 
sorry

end matchstick_equality_l147_147293


namespace probability_of_x_gt_8y_l147_147480

-- Define the rectangular region
structure RectangularRegion where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ

-- Define the specific rectangular region given in the problem
def rectangle : RectangularRegion :=
  { x_min := 0, x_max := 401, y_min := 0, y_max := 402 }

-- Define the function that calculates the probability
noncomputable def probability_x_gt_8y (r : RectangularRegion) : ℚ :=
  -- Calculate area of triangle formed by y = (1/8) * x under the rectangle
  let x_intersect := r.x_max
  let y_intersect := (1 / 8) * x_intersect
  let triangle_area := (1 / 2) * x_intersect * y_intersect
  -- Calculate area of rectangle
  let rectangle_area := (r.x_max - r.x_min) * (r.y_max - r.y_min)
  -- Probability
  triangle_area / rectangle_area

-- The proof statement
theorem probability_of_x_gt_8y : 
  probability_x_gt_8y rectangle = 201 / 3216 :=
by
  sorry

end probability_of_x_gt_8y_l147_147480


namespace max_area_proof_l147_147315

-- Define the original curve
def original_curve (x : ℝ) : ℝ := x^2 + x - 2

-- Reflective symmetry curve about point (p, 2p)
def transformed_curve (p x : ℝ) : ℝ := -x^2 + (4 * p + 1) * x - 4 * p^2 + 2 * p + 2

-- Intersection conditions
def intersecting_curves (p x : ℝ) : Prop :=
original_curve x = transformed_curve p x

-- Range for valid p values
def valid_p (p : ℝ) : Prop := -1 ≤ p ∧ p ≤ 2

-- Prove the problem statement which involves ensuring the curves intersect in the range
theorem max_area_proof :
  ∀ (p : ℝ), valid_p p → ∀ (x : ℝ), intersecting_curves p x →
  ∃ (A : ℝ), A = abs (original_curve x - transformed_curve p x) :=
by
  intros p hp x hx
  sorry

end max_area_proof_l147_147315


namespace dice_roll_probability_l147_147981

theorem dice_roll_probability : 
  ∃ (m n : ℕ), (1 ≤ m ∧ m ≤ 6) ∧ (1 ≤ n ∧ n ≤ 6) ∧ (m - n > 0) ∧ 
  ( (15 : ℚ) / 36 = (5 : ℚ) / 12 ) :=
by {
  sorry
}

end dice_roll_probability_l147_147981


namespace total_students_in_line_l147_147940

-- Define the conditions
def students_in_front : Nat := 15
def students_behind : Nat := 12

-- Define the statement to prove: total number of students in line is 28
theorem total_students_in_line : students_in_front + 1 + students_behind = 28 := 
by 
  -- Placeholder for the proof
  sorry

end total_students_in_line_l147_147940


namespace radius_of_circle_complex_roots_l147_147229

noncomputable def radius_of_complex_roots_circle : ℝ :=
  let z := ℂ in 
  if h : (∀ z : ℂ, (z + 2)^6 = 64 * (z)^6 → ∥z + 2∥ = 2 * ∥z∥) then
    (4 / 3)
  else
    0  -- placeholder, actual proof logic not required

theorem radius_of_circle_complex_roots :
  (∀ z : ℂ, (z + 2)^6 = 64 * (z)^6 → ∥z + 2∥ = 2 * ∥z∥) → radius_of_complex_roots_circle = (4 / 3) :=
by sorry

end radius_of_circle_complex_roots_l147_147229


namespace min_visible_faces_sum_l147_147963

def opposite_sides_sum (d: ℕ) := d + (7 - d) = 7

def visible_faces_min_sum (corner_sum edge_sum face_sum: ℕ) (corner_count edge_count face_count: ℕ): ℕ :=
  corner_count * corner_sum + edge_count * edge_sum + face_count * face_sum

theorem min_visible_faces_sum:
  ∀ (corner_count edge_count face_count: ℕ),
  corner_count = 8 →
  edge_count = 24 →
  face_count = 24 →
  visible_faces_min_sum 6 3 1 8 24 24 = 144 :=
by
  intros corner_count edge_count face_count h1 h2 h3
  unfold visible_faces_min_sum
  rw [h1, h2, h3]
  norm_num
  exact 144

end min_visible_faces_sum_l147_147963


namespace expr1_eval_expr2_eval_l147_147259

theorem expr1_eval : (3 * Real.sqrt 27 - 2 * Real.sqrt 12) * (2 * Real.sqrt (16 / 3) + 3 * Real.sqrt (25 / 3)) = 115 := 
by
  -- Sorry serves as a placeholder for the proof.
  sorry

theorem expr2_eval : (5 * Real.sqrt 21 - 3 * Real.sqrt 15) / (5 * Real.sqrt (8 / 3) - 3 * Real.sqrt (5 / 3)) = 3 := 
by
  -- Sorry serves as a placeholder for the proof.
  sorry

end expr1_eval_expr2_eval_l147_147259


namespace largest_whole_number_satisfying_inequality_l147_147144

theorem largest_whole_number_satisfying_inequality : ∃ n : ℤ, (1 / 3 + n / 7 < 1) ∧ (∀ m : ℤ, (1 / 3 + m / 7 < 1) → m ≤ n) ∧ n = 4 :=
sorry

end largest_whole_number_satisfying_inequality_l147_147144


namespace find_rate_of_current_l147_147988

-- Parameters and definitions
variables (r w : Real)

-- Conditions of the problem
def original_journey := 3 * r^2 - 23 * w^2 = 0
def modified_journey := 6 * r^2 - 2 * w^2 + 40 * w = 0

-- Main theorem to prove
theorem find_rate_of_current (h1 : original_journey r w) (h2 : modified_journey r w) :
  w = 10 / 11 :=
sorry

end find_rate_of_current_l147_147988


namespace arithmetic_sequence_ratio_l147_147819

def S (n : ℕ) : ℝ := (n / 2) * (a 1 + a n)

theorem arithmetic_sequence_ratio (h : S 9 = 9 * S 5) :
  (a 5) / (a 3) = 5 :=
sorry

end arithmetic_sequence_ratio_l147_147819


namespace largest_multiple_of_15_less_than_500_is_495_l147_147121

-- Define the necessary conditions
def is_multiple_of_15 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 15 * k

def is_less_than_500 (n : ℕ) : Prop :=
  n < 500

-- Problem statement: Prove that the largest positive multiple of 15 less than 500 is 495
theorem largest_multiple_of_15_less_than_500_is_495 : 
  ∀ n : ℕ, is_multiple_of_15 n → is_less_than_500 n → n ≤ 495 := 
by
  sorry

end largest_multiple_of_15_less_than_500_is_495_l147_147121


namespace value_of_a4_l147_147736

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d
def sum_of_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) := ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

theorem value_of_a4 {a : ℕ → ℝ} {S : ℕ → ℝ} (h1 : arithmetic_sequence a)
  (h2 : sum_of_arithmetic_sequence S a) (h3 : S 7 = 28) :
  a 4 = 4 := 
  sorry

end value_of_a4_l147_147736


namespace maximum_sum_of_composites_l147_147357

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ n = a * b

def pairwise_coprime (A B C : ℕ) : Prop :=
  Nat.gcd A B = 1 ∧ Nat.gcd A C = 1 ∧ Nat.gcd B C = 1

theorem maximum_sum_of_composites (A B C : ℕ)
  (hA : is_composite A) (hB : is_composite B) (hC : is_composite C)
  (h_pairwise : pairwise_coprime A B C)
  (h_prod_eq : A * B * C = 11011 * 28) :
  A + B + C = 1626 := 
sorry

end maximum_sum_of_composites_l147_147357


namespace largest_multiple_of_15_less_than_500_l147_147056

theorem largest_multiple_of_15_less_than_500 : 
  ∃ n : ℕ, n * 15 < 500 ∧ ∀ m : ℕ, m * 15 < 500 → m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l147_147056


namespace concentration_is_20_percent_l147_147583

def concentration_of_mixture (pure_water_volume solution_volume solution_concentration : ℝ) : ℝ :=
  (solution_volume * solution_concentration) / (pure_water_volume + solution_volume)

theorem concentration_is_20_percent :
  concentration_of_mixture 1 0.5 0.6 = 0.2 :=
by
  sorry

end concentration_is_20_percent_l147_147583


namespace largest_multiple_of_15_less_than_500_l147_147024

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), n < 500 ∧ 15 ∣ n ∧ ∀ (m : ℕ), m < 500 ∧ 15 ∣ m -> m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l147_147024


namespace largest_multiple_15_under_500_l147_147085

-- Define the problem statement
theorem largest_multiple_15_under_500 : ∃ (n : ℕ), n < 500 ∧ n % 15 = 0 ∧ (∀ m, m < 500 ∧ m % 15 = 0 → m ≤ n) :=
by {
  use 495,
  split,
  {
    exact lt_of_le_of_ne (le_refl _) (ne_of_lt (nat.lt_succ_self 499))
  },
  split,
  {
    exact nat.mod_eq_zero_of_dvd ⟨33, rfl⟩,
  },
  {
    intros m hm h,
    have hmn : m ≤ 495,
    {
      have div15 := hm/15,
      suffices : div15 < 33+1, from nat.le_of_lt_add_one this,
      exact this, sorry,
    },
    exact hmn,
  },
}

end largest_multiple_15_under_500_l147_147085


namespace line_tangent_to_parabola_j_eq_98_l147_147628

theorem line_tangent_to_parabola_j_eq_98 (j : ℝ) :
  (∀ x y : ℝ, y^2 = 32 * x → 4 * x + 7 * y + j = 0 → x ≠ 0) →
  j = 98 :=
by
  sorry

end line_tangent_to_parabola_j_eq_98_l147_147628


namespace sum_in_base_9_correct_l147_147997

theorem sum_in_base_9_correct :
  let n1 := 2 * 9^2 + 7 * 9^1 + 6 * 9^0,
      n2 := 8 * 9^2 + 0 * 9^1 + 3 * 9^0,
      n3 := 7 * 9^1 + 2 * 9^0,
      sum := 1 * 9^3 + 2 * 9^2 + 1 * 9^1 + 6 * 9^0
  in n1 + n2 + n3 = sum :=
by {
  let n1 := 2 * 9^2 + 7 * 9^1 + 6 * 9^0,
  let n2 := 8 * 9^2 + 0 * 9^1 + 3 * 9^0,
  let n3 := 7 * 9^1 + 2 * 9^0,
  let sum := 1 * 9^3 + 2 * 9^2 + 1 * 9^1 + 6 * 9^0,
  sorry
}

end sum_in_base_9_correct_l147_147997


namespace injective_statement_not_injective_f1_not_injective_f2_monotonic_not_impl_injective_correct_statements_l147_147503

-- Definitions
def is_injective (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, f x₁ = f x₂ → x₁ = x₂

def f1 (x : ℝ) : ℝ := x^2 - 2*x
def f2 (x : ℝ) : ℝ :=
  if x >= 2 then log x / log 2 else 2 - x

-- Statements
theorem injective_statement (x1 x2 : ℝ) (f : ℝ → ℝ) (h_inj : is_injective f) :
  x1 ≠ x2 → f x1 ≠ f x2 :=
by
  intro h
  intro h_eq
  apply h
  apply h_inj
  apply h_eq

theorem not_injective_f1 : ¬ is_injective f1 :=
by
  intro h
  have h1 : f1 0 = f1 2 := by simp [f1]
  have h2 : 0 ≠ 2 := by norm_num
  have h3 : 0 = 2 := h 0 2 h1
  contradiction

theorem not_injective_f2 : ¬ is_injective f2 :=
by
  intro h
  have h1 : f2 0 = f2 4 := by simp [f2]; split_ifs; linarith
  have h2 : 0 ≠ 4 := by norm_num
  have h3 : 0 = 4 := h 0 4 h1
  contradiction

theorem monotonic_not_impl_injective (f : ℝ → ℝ) (D : set ℝ) (h_mono : monotone_on f D) :
  ¬ is_injective f :=
by
  sorry

-- Correct Answer
theorem correct_statements : (¬ is_injective f1) ∧ (¬ is_injective f2) ∧ (injective_statement) ∧ (¬ monotonic_not_impl_injective) :=
by
  split;
  [ apply not_injective_f1,
    split;
    [ apply not_injective_f2,
      split;
      [ apply injective_statement, 
        apply monotonic_not_impl_injective ]]]


end injective_statement_not_injective_f1_not_injective_f2_monotonic_not_impl_injective_correct_statements_l147_147503


namespace solution_set_of_inequality_l147_147521

theorem solution_set_of_inequality (x : ℝ) : (x - 1 ≤ (1 + x) / 3) → (x ≤ 2) :=
by
  sorry

end solution_set_of_inequality_l147_147521


namespace largest_number_is_number2_l147_147155

def number1 := 8.23456
def number2 := 8.234 + (5 / 9)   -- 8.234\overline{5}
def number3 := 8.23 + (45 / 99)  -- 8.23\overline{45}
def number4 := 8.2 + (345 / 999) -- 8.2\overline{345}
def number5 := 8 + (2345 / 9999) -- 8.\overline{2345}

theorem largest_number_is_number2 :
  number2 > number1 ∧ number2 > number3 ∧ number2 > number4 ∧ number2 > number5 :=
sorry

end largest_number_is_number2_l147_147155


namespace prob_both_persons_two_shots_prob_at_least_three_shots_l147_147336

variable (P_A_shot : ℝ)
variable (P_B_shot : ℝ)
variable (N_shots_A : ℕ)
variable (N_shots_B : ℕ)

-- Definitions of the shooting probabilities and number of shots
def P_A := 0.4
def P_B := 0.6
def N := 2

-- Theorem 1: Probability that both persons make two shots each is 0.0576
theorem prob_both_persons_two_shots : 
  P_A_shot = 0.4 → P_B_shot = 0.6 → N_shots_A = 2 → N_shots_B = 2 → 
  (P_A_shot * P_A_shot * P_B_shot * P_B_shot) = 0.0576 :=
by
  intros
  rw [P_A, P_B, N]
  sorry

-- Theorem 2: Probability that at least three shots are made by the two persons is 0.1824
theorem prob_at_least_three_shots : 
  P_A_shot = 0.4 → P_B_shot = 0.6 → N_shots_A = 2 → N_shots_B = 2 → 
  ((P_A_shot * P_A_shot * P_B_shot * P_B_shot) + (P_A_shot * P_A_shot * P_B_shot * (1 - P_B_shot)) + (P_A_shot * P_B_shot * (1 - P_A_shot) * P_B_shot)) = 0.1824 :=
by
  intros
  rw [P_A, P_B, N]
  sorry

end prob_both_persons_two_shots_prob_at_least_three_shots_l147_147336


namespace not_exclusively_square_l147_147413

-- Define the conditions of the quadrilateral
variables (ABCD : Type) [quadrilateral ABCD]
variables (AC BD : line_segment ABCD)
variables (inscribed_circle circumscribed_circle : circle ABCD)

-- State that AC and BD are diagonals and are perpendicular
axiom diagonals_perpendicular : AC.perpendicular BD

-- State that it is possible to inscribe and circumscribe circles around the quadrilateral
axiom inscribable : inscribed_circle
axiom circumscribable : circumscribed_circle

-- Define the theorem to be proven
theorem not_exclusively_square : ¬(∀ (ABCD : Type) [quadrilateral ABCD] (AC BD : line_segment ABCD)
  (inscribed_circle circumscribed_circle : circle ABCD),
  AC.perpendicular BD →
  inscribable →
  circumscribable →
  is_square ABCD) :=
sorry

end not_exclusively_square_l147_147413


namespace factorize_eq_l147_147696

theorem factorize_eq (x : ℝ) : 2 * x^3 - 8 * x = 2 * x * (x + 2) * (x - 2) := 
by
  sorry

end factorize_eq_l147_147696


namespace find_y_l147_147542

def initial_radius : ℝ := 5
def initial_height : ℝ := 4

def volume_with_increased_radius (y : ℝ) : ℝ :=
  Real.pi * (initial_radius + y)^2 * initial_height

def volume_with_increased_height (y : ℝ) : ℝ :=
  Real.pi * initial_radius^2 * (initial_height + y)

theorem find_y (y : ℝ) : (volume_with_increased_radius y = volume_with_increased_height y) → y = 5 / 4 :=
by
  sorry

end find_y_l147_147542


namespace sqrt_expr_sum_equality_l147_147396

theorem sqrt_expr_sum_equality :
  let a := 14
  let b := 3
  let c := 2
  (\sqrt(6) + 1/\sqrt(6) + \sqrt(8) + 1/\sqrt(8) = (a * \sqrt(6) + b * \sqrt(8)) / c) :=
begin
  sorry
end

lemma abc_sum :
  let a := 14
  let b := 3
  let c := 2
  (a + b + c = 19) :=
begin
  sorry
end

end sqrt_expr_sum_equality_l147_147396


namespace angle_AMD_eq_45_l147_147867

theorem angle_AMD_eq_45
  (ABCD : Type)
  [rectangle ABCD]
  (A B C D M : ABCD)
  (h_AB : length A B = 6)
  (h_BC : length B C = 3)
  (M_on_AB : M ∈ segment A B)
  (h_angles : ∠AMD = ∠CMD) :
  ∠AMD = 45 := sorry

end angle_AMD_eq_45_l147_147867


namespace intersection_of_sets_l147_147353

def setP : Set ℝ := { x | x ≤ 3 }
def setQ : Set ℝ := { x | x > 1 }

theorem intersection_of_sets : setP ∩ setQ = { x | 1 < x ∧ x ≤ 3 } :=
by
  sorry

end intersection_of_sets_l147_147353


namespace ceiling_sum_l147_147646

theorem ceiling_sum :
  let a := 4 / 3
  let b := 16 / 9
  let c := 256 / 81
  ⌈a⌉ + ⌈b⌉ + ⌈c⌉ = 8 := by
  sorry

end ceiling_sum_l147_147646


namespace ceiling_sum_evaluation_l147_147688

noncomputable def evaluateCeilingSum : ℝ := 
  ⌈Real.sqrt (16 / 9)⌉ + ⌈(16 / 9)⌉ + ⌈((16 / 9) ^ 2)⌉ 

theorem ceiling_sum_evaluation : evaluateCeilingSum = 8 := by
  sorry

end ceiling_sum_evaluation_l147_147688


namespace number_of_shelves_in_library_l147_147905

theorem number_of_shelves_in_library (total_books : ℕ) (books_per_shelf : ℕ) (h1 : total_books = 14240) (h2 : books_per_shelf = 8) : total_books / books_per_shelf = 1780 :=
by {
  rw [h1, h2],
  norm_num,
  sorry
}

end number_of_shelves_in_library_l147_147905


namespace equation_of_circle_C_l147_147206

theorem equation_of_circle_C :
  ∀ (a b : ℕ),
    2500 = 1000 + 900 + 600 →
    100 = a + 900 + b →
    (∃ (B C : Point) (A : Point), A = (1, -1) ∧ angle A B C = 120 ° ∧ B ≠ C ∧ (C.x - 1)^2 + (C.y + 1)^2 = (origin_distance A 5 3 1) * √((3/√34)) → 
    (C.x - 1)^2 + (C.y + 1)^2 = (3 / sqrt 34 )^2 * 18 / 17 :=
sorry

end equation_of_circle_C_l147_147206


namespace curve_is_line_l147_147704

-- Let theta be the angle in polar coordinates
variable {θ : ℝ}

-- Define the condition given in the problem
def condition : Prop := θ = π / 4

-- Prove that the curve defined by this condition is a line
theorem curve_is_line (h : condition) : ∃ (m b : ℝ), ∀ x y : ℝ, y = m * x + b :=
by
  sorry

end curve_is_line_l147_147704


namespace circumcenter_area_comparison_circumcenter_concurrence_condition_l147_147811

variables {A B C X Y Z A' B' C' : Type*}

-- Define triangle and points on segments
variables [IsTriangle A B C]
variables (X : PointOnSegment B C)
variables (Y : PointOnSegment A C)
variables (Z : PointOnSegment A B)
variables (A' : Circumcenter A Z Y)
variables (B' : Circumcenter B X Z)
variables (C' : Circumcenter C Y X)

-- Define areas of triangles
noncomputable def area_ABC : ℝ := area A B C
noncomputable def area_A'B'C' : ℝ := area A' B' C'

-- State the theorem
theorem circumcenter_area_comparison : 4 * area_A'B'C' ≥ area_ABC :=
begin
  sorry,
end

-- State condition for equality
theorem circumcenter_concurrence_condition : AA' ∧ BB' ∧ CC' are_concurrent ↔ 4 * area_A'B'C' = area_ABC :=
begin
  sorry,
end

end circumcenter_area_comparison_circumcenter_concurrence_condition_l147_147811


namespace largest_multiple_of_15_less_than_500_is_495_l147_147127

-- Define the necessary conditions
def is_multiple_of_15 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 15 * k

def is_less_than_500 (n : ℕ) : Prop :=
  n < 500

-- Problem statement: Prove that the largest positive multiple of 15 less than 500 is 495
theorem largest_multiple_of_15_less_than_500_is_495 : 
  ∀ n : ℕ, is_multiple_of_15 n → is_less_than_500 n → n ≤ 495 := 
by
  sorry

end largest_multiple_of_15_less_than_500_is_495_l147_147127


namespace num_edges_remaining_solid_l147_147208

def num_edges_original_cube : ℕ := 12
def num_corners : ℕ := 8
def small_cube_side_length : ℕ := 3 -- 1.5*2 to avoid dealing with floating point in Lean

theorem num_edges_remaining_solid : Prop :=
  ∃ remaining_edges : ℕ, remaining_edges = 36 ∧ 
  (let
    side_length_original_cube := 4,
    num_edges := num_edges_original_cube,
    edges_per_corner := 3,
    shared_edges := edges_per_corner / 3 -- each small cube contributes these shared edges
  in num_edges + (num_corners * edges_per_corner) / shared_edges = remaining_edges)

end num_edges_remaining_solid_l147_147208


namespace eq_of_operation_l147_147151

theorem eq_of_operation {x : ℝ} (h : 60 + 5 * 12 / (x / 3) = 61) : x = 180 :=
by
  sorry

end eq_of_operation_l147_147151


namespace evaluate_expression_l147_147257

theorem evaluate_expression (x : ℝ) (hx : x < -1) : sqrt (x / (1 - (2 * x - 3) / (x + 1))) = complex.I * sqrt (x^2 - 3 * x - 4) :=
by
  sorry

end evaluate_expression_l147_147257


namespace find_fraction_value_l147_147300

noncomputable section

open Real

theorem find_fraction_value (α : ℝ) (h : sin (α / 2) - 2 * cos (α / 2) = 1) :
  (1 + sin α + cos α) / (1 + sin α - cos α) = 1 :=
sorry

end find_fraction_value_l147_147300


namespace evaluate_expression_l147_147659

theorem evaluate_expression : 
  (⌈Real.sqrt (16 / 9)⌉ + ⌈ (16 / 9 : ℝ ) ⌉ + ⌈Real.pow (16 / 9 : ℝ ) 2⌉) = 8 := 
by 
  sorry

end evaluate_expression_l147_147659


namespace smallest_missing_digit_units_place_cube_l147_147152

theorem smallest_missing_digit_units_place_cube :
  ∀ d : Fin 10, ∃ n : ℕ, (n ^ 3) % 10 = d :=
by
  sorry

end smallest_missing_digit_units_place_cube_l147_147152


namespace eccentricity_range_l147_147726

-- Definition of the hyperbola
def hyperbola (a b : ℝ) := {p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1}

-- Definitions for the parameters and points
variables {a b : ℝ} (ha : a > 0) (hb : b > 0)
def e := sqrt (1 + b^2 / a^2)

-- Main statement

theorem eccentricity_range : 1 < e ha hb ∧ e ha hb < sqrt 2 :=
  sorry

end eccentricity_range_l147_147726


namespace sale_in_fourth_month_l147_147194

variables (sale1 sale2 sale3 sale5 sale6 averageSale : ℝ)

theorem sale_in_fourth_month (h1 : sale1 = 5420) (h2 : sale2 = 5660) (h3 : sale3 = 6200)
                            (h5 : sale5 = 6500) (h6 : sale6 = 8270) (h_avg : averageSale = 6400) :
                            ∃ sale4, sale4 = 6350 :=
begin
  let total_sales := averageSale * 6,
  have h_total_sales : total_sales = 38400, by simp [h_avg, total_sales],
  let known_sales := sale1 + sale2 + sale3 + sale5 + sale6,
  have h_known_sales : known_sales = 32050, by simp [h1, h2, h3, h5, h6, known_sales],
  let sale4 := total_sales - known_sales,
  have h_sale4 : sale4 = 6350, by simp [h_total_sales, h_known_sales, sale4],
  use sale4,
  exact h_sale4
end

end sale_in_fourth_month_l147_147194


namespace not_quadratic_eq3_l147_147154

-- Define the equations as functions or premises
def eq1 (x : ℝ) := 9 * x^2 = 7 * x
def eq2 (y : ℝ) := abs (y^2) = 8
def eq3 (y : ℝ) := 3 * y * (y - 1) = y * (3 * y + 1)
def eq4 (x : ℝ) := abs 2 * (x^2 + 1) = abs 10

-- Define what it means to be a quadratic equation
def is_quadratic (eq : ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, eq x = (a * x^2 + b * x + c = 0)

-- Prove that eq3 is not a quadratic equation
theorem not_quadratic_eq3 : ¬ is_quadratic eq3 :=
sorry

end not_quadratic_eq3_l147_147154


namespace trebled_resultant_l147_147199

theorem trebled_resultant (n : ℕ) (h : n = 20) : 3 * ((2 * n) + 5) = 135 := 
by
  sorry

end trebled_resultant_l147_147199


namespace find_tangent_point_l147_147280

theorem find_tangent_point (x : ℝ) (y : ℝ) (h_curve : y = x^2) (h_slope : 2 * x = 1) : 
    (x, y) = (1/2, 1/4) :=
sorry

end find_tangent_point_l147_147280


namespace harmonic_leq_sum_of_distinct_natural_numbers_l147_147865

open BigOperators

-- Define the problem
theorem harmonic_leq_sum_of_distinct_natural_numbers (n : ℕ) (a : Fin n → ℕ)
  (h_distinct : Function.Injective a) :
  ∑ i in Finset.range n, (1 : ℚ) / (i + 1) ≤
  ∑ i in Finset.range n, (a i : ℚ) / ((i + 1)^2) := 
sorry

end harmonic_leq_sum_of_distinct_natural_numbers_l147_147865


namespace perimeter_pentagon_l147_147924

noncomputable def AB : ℝ := 1
noncomputable def BC : ℝ := Real.sqrt 2
noncomputable def CD : ℝ := Real.sqrt 3
noncomputable def DE : ℝ := 2

noncomputable def AC : ℝ := Real.sqrt (AB^2 + BC^2)
noncomputable def AD : ℝ := Real.sqrt (AC^2 + CD^2)
noncomputable def AE : ℝ := Real.sqrt (AD^2 + DE^2)

theorem perimeter_pentagon (ABCDE : List ℝ) (H : ABCDE = [AB, BC, CD, DE, AE]) :
  List.sum ABCDE = 3 + Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 10 :=
by
  sorry -- Proof skipped as instructed

end perimeter_pentagon_l147_147924


namespace number_exceeds_fraction_l147_147170

theorem number_exceeds_fraction (x : ℝ) (hx : x = 0.45 * x + 1000) : x = 1818.18 := 
by
  sorry

end number_exceeds_fraction_l147_147170


namespace largest_multiple_of_15_less_than_500_l147_147001

theorem largest_multiple_of_15_less_than_500 : ∃ n : ℕ, (n > 0 ∧ 15 * n < 500) ∧ (∀ m : ℕ, m > n → 15 * m >= 500) ∧ 15 * n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l147_147001


namespace evaluate_expression_l147_147662

theorem evaluate_expression :
  let x := (16 : ℚ) / 9
  in ⌈(√x)⌉ + ⌈x⌉ + ⌈x^2⌉ = 8 :=
by
  let x := (16 : ℚ) / 9
  sorry

end evaluate_expression_l147_147662


namespace conic_section_properties_l147_147743

theorem conic_section_properties :
  ∀ (ρ θ : ℝ),
  (∀ θ, ρ = 8 * sin θ / (1 + cos (2 * θ))) →
  let x := ρ * cos θ,
      y := ρ * sin θ in
  (x^2 = 4 * y) ∧ 2 = 2 :=
by
  sorry

end conic_section_properties_l147_147743


namespace candidates_appeared_in_each_state_l147_147785

theorem candidates_appeared_in_each_state (X : ℝ) (h1 : 0.07 * X - 0.06 * X = 81) : X = 8100 :=
by
  unfold
  sorry

end candidates_appeared_in_each_state_l147_147785


namespace angle_BFO_is_right_l147_147585

-- Definitions for the various points and circles
variables {A B C D E F O : Type}
variables [is_point A] [is_point B] [is_point C] [is_point D] [is_point E] [is_point F] [is_point O]
variables [is_circle (circle_through A B D)] [is_circle (circle_through B C E)]
variables (circle_AEC : is_circle (circle_through A E D C O))

-- Given conditions are translated into definitions
def condition1 : Prop := (circle_through A B D).intersects (line_segment B C D)
def condition2 : Prop := (circle_through B C E).intersects (line_segment A B E) 
                         ∧ (circle_through B C E).intersects (circle_through A B D) at F
def condition3 : Prop := A ∈ circle_AEC ∧ E ∈ circle_AEC ∧ D ∈ circle_AEC ∧ C ∈ circle_AEC ∧ center circle_AEC = O

-- Main statement to prove
theorem angle_BFO_is_right : condition1 ∧ condition2 ∧ condition3 → angle B F O = π / 2 :=
sorry

end angle_BFO_is_right_l147_147585


namespace range_of_a_l147_147343

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ∈ Icc 0 (2 * Real.pi) →
    (2 * a - 1) * Real.sin x + (2 - a) * Real.sin (2 * x) = Real.sin (3 * x)) →
  ∃ (c : ℝ), ∀ (x n : ℕ), x = c + n * (Real.pi) →
    (a ∈ Set.Iic (-2) ∨ a = 0 ∨ a ∈ Set.Ici 2) :=
by
  sorry

end range_of_a_l147_147343


namespace hermia_elected_probability_l147_147841

-- Define the problem statement and conditions in Lean 4
noncomputable def probability_hermia_elected (n : ℕ) (h_odd : (n % 2 = 1)) (h_pos : n > 0) : ℚ :=
  if n = 1 then 1 else (2^n - 1) / (n * 2^(n-1))

-- Lean theorem statement
theorem hermia_elected_probability (n : ℕ) (h_odd : (n % 2 = 1)) (h_pos : n > 0) : 
  probability_hermia_elected n h_odd h_pos = (2^n - 1) / (n * 2^(n-1)) :=
by
  sorry

end hermia_elected_probability_l147_147841


namespace wall_building_problem_l147_147181

theorem wall_building_problem 
    (num_workers_1 : ℕ) (length_wall_1 : ℕ) (days_1 : ℕ)
    (num_workers_2 : ℕ) (length_wall_2 : ℕ) (days_2 : ℕ) :
    num_workers_1 = 8 → length_wall_1 = 140 → days_1 = 42 →
    num_workers_2 = 30 → length_wall_2 = 100 →
    (work_done : ℕ → ℕ → ℕ) → 
    (work_done length_wall_1 days_1 = num_workers_1 * days_1 * length_wall_1) →
    (work_done length_wall_2 days_2 = num_workers_2 * days_2 * length_wall_2) →
    (days_2 = 8) :=
by
  intros h1 h2 h3 h4 h5 wf wlen1 wlen2
  sorry

end wall_building_problem_l147_147181


namespace chips_probability_l147_147183

theorem chips_probability : 
  let total_chips := 12,
      green_chips := 4,
      blue_chips := 3,
      yellow_chips := 5,
      total_arrangements := Nat.factorial total_chips,
      group_arrangements := Nat.factorial green_chips * Nat.factorial blue_chips * Nat.factorial yellow_chips * 2 in
  (group_arrangements / total_arrangements : ℚ) = 1 / 13860 := 
by sorry

end chips_probability_l147_147183


namespace find_t_of_odd_function_l147_147375

theorem find_t_of_odd_function (f : ℝ → ℝ) (t : ℝ) 
  (h_odd : ∀ x ∈ set.Icc t (t^2 - 3*t - 3), f (-x) = -f x):
  t = -1 :=
sorry

end find_t_of_odd_function_l147_147375


namespace simplify_expression_l147_147740

theorem simplify_expression {x y : ℝ}
  (h : y = sqrt (x - 2) + sqrt (2 - x) + 2) :
  |y - sqrt 3| - (x - 2 + sqrt 2) ^ 2 = -sqrt 3 :=
by
  sorry

end simplify_expression_l147_147740


namespace basketball_lineup_l147_147964

theorem basketball_lineup : 
  let total_players := 12 in
  let twins := 2 in
  let triplets := 3 in
  let lineup_size := 5 in
  let players_left := total_players - twins - triplets in
  (nat.choose triplets 2 + nat.choose triplets 3) * nat.choose players_left 1 = 28 :=
by sorry

end basketball_lineup_l147_147964


namespace sports_day_results_l147_147592

-- Conditions and questions
variables (a b c : ℕ)
variables (class1_score class2_score class3_score class4_score : ℕ)

-- Conditions given in the problem
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom a_gt_b_gt_c : a > b ∧ b > c
axiom no_ties : (class1_score ≠ class2_score) ∧ (class2_score ≠ class3_score) ∧ (class3_score ≠ class4_score) ∧ (class1_score ≠ class3_score) ∧ (class1_score ≠ class4_score) ∧ (class2_score ≠ class4_score)
axiom class_scores : class1_score + class2_score + class3_score + class4_score = 40

-- To prove
theorem sports_day_results : a + b + c = 8 ∧ a = 5 :=
by
  sorry

end sports_day_results_l147_147592


namespace exists_visible_point_l147_147598

open Nat -- to use natural numbers and their operations

def is_visible (x y : ℤ) : Prop :=
  Int.gcd x y = 1

theorem exists_visible_point (n : ℕ) (hn : n > 0) :
  ∃ a b : ℤ, is_visible a b ∧
  ∀ (P : ℤ × ℤ), (P ≠ (a, b) → (Int.sqrt ((P.fst - a) * (P.fst - a) + (P.snd - b) * (P.snd - b)) > n)) :=
sorry

end exists_visible_point_l147_147598


namespace ceiling_sum_evaluation_l147_147691

noncomputable def evaluateCeilingSum : ℝ := 
  ⌈Real.sqrt (16 / 9)⌉ + ⌈(16 / 9)⌉ + ⌈((16 / 9) ^ 2)⌉ 

theorem ceiling_sum_evaluation : evaluateCeilingSum = 8 := by
  sorry

end ceiling_sum_evaluation_l147_147691


namespace basis_service_B_l147_147233

def vector := ℤ × ℤ

def not_collinear (v1 v2 : vector) : Prop :=
  v1.1 * v2.2 ≠ v1.2 * v2.1

def A : vector × vector := ((0, 0), (2, 3))
def B : vector × vector := ((-1, 3), (5, -2))
def C : vector × vector := ((3, 4), (6, 8))
def D : vector × vector := ((2, -3), (-2, 3))

theorem basis_service_B : not_collinear B.1 B.2 := by
  sorry

end basis_service_B_l147_147233


namespace book_distribution_l147_147560

theorem book_distribution (x : ℕ) (h1 : 9 * x + 7 < 11 * x) : 
  9 * x + 7 = totalBooks - 9 * x ∧ totalBooks - 9 * x = 7 :=
by
  sorry

end book_distribution_l147_147560


namespace gcd_102_238_l147_147512

theorem gcd_102_238 : Int.gcd 102 238 = 34 :=
by
  sorry

end gcd_102_238_l147_147512


namespace find_m_l147_147166

theorem find_m (m : ℤ) (h : (-2)^(2 * m) = 2^(24 - m)) : 
  m = 8 := 
sorry

end find_m_l147_147166


namespace solve_trigonometric_inequality_l147_147284

noncomputable def trigonometric_inequality (x : ℝ) : Prop :=
  x ∈ Set.Ioo 0 (2 * Real.pi) ∧ 2^x * (2 * Real.sin x - Real.sqrt 3) ≥ 0

theorem solve_trigonometric_inequality :
  ∀ x, x ∈ Set.Ioo 0 (2 * Real.pi) → (2^x * (2 * Real.sin x - Real.sqrt 3) ≥ 0 ↔ x ∈ Set.Icc (Real.pi / 3) (2 * Real.pi / 3)) :=
by
  intros x hx
  sorry

end solve_trigonometric_inequality_l147_147284


namespace largest_multiple_of_15_less_than_500_l147_147110

theorem largest_multiple_of_15_less_than_500 :
∀ x : ℕ, (∃ k : ℕ, x = 15 * k ∧ 0 < x ∧ x < 500) → x ≤ 495 :=
begin
  sorry
end

end largest_multiple_of_15_less_than_500_l147_147110


namespace largest_multiple_of_15_less_than_500_l147_147139

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), 15 * n < 500 ∧ 15 * n ≥ 15 * (33 - 1) + 1 := by
  sorry

end largest_multiple_of_15_less_than_500_l147_147139


namespace largest_multiple_of_15_below_500_l147_147039

theorem largest_multiple_of_15_below_500 : ∃ (k : ℕ), 15 * k < 500 ∧ ∀ (m : ℕ), 15 * m < 500 → 15 * m ≤ 15 * k := 
by
  existsi 33
  split
  · norm_num
  · intro m h
    have h1 : m ≤ 33 := Nat.le_of_lt_succ (Nat.div_lt_succ 500 15 m)
    exact (mul_le_mul_left (by norm_num : 0 < 15)).mpr h1
  sorry

end largest_multiple_of_15_below_500_l147_147039


namespace total_votes_l147_147414

theorem total_votes (emma_votes : ℕ) (vote_fraction : ℚ) (h_emma : emma_votes = 45) (h_fraction : vote_fraction = 3/7) :
  emma_votes = vote_fraction * 105 :=
by {
  sorry
}

end total_votes_l147_147414


namespace largest_multiple_of_15_less_than_500_l147_147070

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), (n < 500) ∧ (15 ∣ n) ∧ (∀ m : ℕ, (m < 500) ∧ (15 ∣ m) → m ≤ n) ∧ n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l147_147070


namespace solve_for_y_l147_147245

def dataset : List ℕ := [70, 110, y, 45, 55, 210, 95, y, 180]

def dataset_mean := (765 + 2 * y) / 9

def is_median (l : List ℕ) (x : ℕ) : Prop :=
  let sorted := l.sort
  let n := sorted.length
  sorted.nth! (n / 2) = x

def is_mode (l : List ℕ) (x : ℕ) : Prop :=
  l.count x > 1 ∧ ∀ y, y ≠ x → l.count x ≥ l.count y

theorem solve_for_y (y : ℕ) : y = 95 :=
  sorry

end solve_for_y_l147_147245


namespace range_of_a_l147_147593

variable (f : ℝ → ℝ)

-- Condition 1: ∀ x ∈ ℝ, f(x) + f(-x) = x^2 / 2
axiom condition1 : ∀ x : ℝ, f(x) + f(-x) = x^2 / 2

-- Condition 2: ∀ x ∈ [0, +∞), 2f'(x) > x
axiom condition2 : ∀ x : ℝ, 0 ≤ x → 2 * (deriv f x) > x

-- Condition 3: f(a) - f(2 - a) ≥ a - 1
axiom condition3 : ∀ a : ℝ, f(a) - f(2 - a) ≥ a - 1

theorem range_of_a (a : ℝ) : 1 ≤ a :=
by
  sorry

end range_of_a_l147_147593


namespace find_C_range_a_minus_half_b_l147_147358

variable (A B C a b c : Real)

-- Conditions as an example
-- Triangle ABC with sides a, b, c opposite to angles A, B, C
-- and c * tan(C) = sqrt(3) * (a * cos(B) + b * cos(A))
axiom triangle_ABC : c * Real.tan(C) = Real.sqrt 3 * (a * Real.cos(B) + b * Real.cos(A))

-- Prove angle C equals π / 3
theorem find_C (h : triangle_ABC) : C = Real.pi / 3 := by
  sorry

-- Conditions for the second part of the problem
-- Triangle ABC is acute and c = sqrt(3)
axiom acute_triangle_and_c : (0 < A < Real.pi / 2) ∧ (0 < B < Real.pi / 2) ∧ (0 < C < Real.pi / 2) ∧ c = Real.sqrt 3

-- Λet a and b be defined from the sine law
def a_sine_law := 2 * Real.sin(A)
def b_sine_law := 2 * Real.sin(B)
def a_minus_half_b := a_sine_law - b_sine_law / 2

-- Prove the range of values for a - b / 2
theorem range_a_minus_half_b (h : acute_triangle_and_c) : 0 < a_minus_half_b ∧ a_minus_half_b < 3 / 2 := by
  sorry

end find_C_range_a_minus_half_b_l147_147358


namespace largest_multiple_of_15_less_than_500_l147_147008

theorem largest_multiple_of_15_less_than_500 : ∃ n : ℕ, (n > 0 ∧ 15 * n < 500) ∧ (∀ m : ℕ, m > n → 15 * m >= 500) ∧ 15 * n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l147_147008


namespace square_side_length_s2_l147_147868

theorem square_side_length_s2 (s1 s2 s3 : ℕ)
  (h1 : s1 + s2 + s3 = 3322)
  (h2 : s1 - s2 + s3 = 2020) :
  s2 = 651 :=
by sorry

end square_side_length_s2_l147_147868


namespace bus_speed_including_stoppages_l147_147263

theorem bus_speed_including_stoppages (speed_excluding_stoppages : ℝ) 
(stop_time_per_hour : ℝ) :
  (speed_excluding_stoppages = 54) →
  (stop_time_per_hour = 10 / 60) →
  let total_time_per_hour := 1 in
  let running_time_per_hour := 1 - stop_time_per_hour in
  let distance_covered := speed_excluding_stoppages * running_time_per_hour in
  let speed_including_stoppages := distance_covered / total_time_per_hour in
  speed_including_stoppages = 45 :=
begin
  intros h1 h2,
  have running_time_eq : running_time_per_hour = 50 / 60,
  { rw [h2, sub_eq_add_neg, (by norm_num : 1 = 60 / 60)], ring },
  have distance_eq : distance_covered = 54 * (50 / 60),
  { rwa [h1, running_time_eq] },
  rw [h1] at distance_eq, simp at distance_eq,
  have speed_eq : speed_including_stoppages = (54 * (50 / 60)) / 1,
  { rwa distance_eq },
  norm_num1 at speed_eq,
  exact speed_eq,
end

end bus_speed_including_stoppages_l147_147263


namespace largest_multiple_15_under_500_l147_147081

-- Define the problem statement
theorem largest_multiple_15_under_500 : ∃ (n : ℕ), n < 500 ∧ n % 15 = 0 ∧ (∀ m, m < 500 ∧ m % 15 = 0 → m ≤ n) :=
by {
  use 495,
  split,
  {
    exact lt_of_le_of_ne (le_refl _) (ne_of_lt (nat.lt_succ_self 499))
  },
  split,
  {
    exact nat.mod_eq_zero_of_dvd ⟨33, rfl⟩,
  },
  {
    intros m hm h,
    have hmn : m ≤ 495,
    {
      have div15 := hm/15,
      suffices : div15 < 33+1, from nat.le_of_lt_add_one this,
      exact this, sorry,
    },
    exact hmn,
  },
}

end largest_multiple_15_under_500_l147_147081


namespace solid_volume_formula_l147_147209

noncomputable def volume_of_solid (A : ℝ → ℝ) (h B M T : ℝ) : ℝ :=
  (h * (B + 4 * M + T)) / 6

theorem solid_volume_formula (a b c d h : ℝ) :
  let A := λ z : ℝ, a * z^3 + b * z^2 + c * z + d
  let B := A (-h / 2)
  let M := A 0
  let T := A (h / 2)
  volume_of_solid A h B M T = ∫ z in -h / 2..h / 2, A z := 
sorry

end solid_volume_formula_l147_147209


namespace f_monotone_intervals_find_a_b_l147_147346

noncomputable def f (a b x: ℝ) := a * sin (2 * x + π / 3) + b

-- Define the conditions
def cond1 (a : ℝ) : Prop := a > 0
def cond2 (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ π / 4
def cond3 (f : ℝ → ℝ) : Prop := ∀ x ∈ Icc (0 : ℝ) (π / 4), 1 ≤ f x ∧ f x ≤ 3

-- Define the ranges
def range1 (x : ℝ) (k : ℤ) : Prop := k * π - 5 * π / 12 ≤ x ∧ x ≤ k * π + π / 12

-- Prove monotonically increasing intervals
theorem f_monotone_intervals (a b : ℝ) (k : ℤ) :
  cond1 a →
  ∀ x, range1 x k → monotone (λ x, f a b x)
:= sorry

-- Prove the values of a and b
theorem find_a_b (a b : ℝ) :
  cond1 a →
  cond3 (f a b) →
  a = 4 ∧ b = -1
:= sorry

end f_monotone_intervals_find_a_b_l147_147346


namespace largest_multiple_of_15_less_than_500_l147_147014

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), n < 500 ∧ 15 ∣ n ∧ ∀ (m : ℕ), m < 500 ∧ 15 ∣ m -> m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l147_147014


namespace minimum_possible_value_l147_147451

noncomputable def minimum_value (α γ : ℂ) : ℝ := 
  let f : ℂ → ℂ := λ z, (2 + 3 * complex.I) * z^2 + α * z + γ
  real.abs α + real.abs γ

theorem minimum_possible_value (α γ : ℂ) 
  (h1 : (2 + 3 * complex.I) + α + γ ∈ ℝ) -- f(1) is real
  (h2 : (2 * -1 + 3 * complex.I * -1) + α * complex.I + γ ∈ ℝ) -- f(i) is real 
  : minimum_value α γ = 3 :=
sorry

end minimum_possible_value_l147_147451


namespace entry_order_arrangement_l147_147631

theorem entry_order_arrangement :
  let foreign_guests : ℕ := 4 in
  let security_personnel : ℕ := 2 in
  (perm security_personnel security_personnel) * (perm foreign_guests foreign_guests) = 48 :=
by
  sorry

end entry_order_arrangement_l147_147631


namespace S_2018_eq_4732_l147_147729

-- Define the sequence
def sequence (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  if n = 0 then 5
  else if a n % 2 = 0 then a n / 2
       else 3 * a n + 1

-- Define the sum of the first n terms
def sum_first_n_terms (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  Nat.sum (List.range n) a

-- The main theorem we need to prove
theorem S_2018_eq_4732 : ∃ a : ℕ → ℕ, 
  a 0 = 5 ∧ 
  (∀ n, a (n+1) = if a n % 2 = 0 then a n / 2 else 3 * a n + 1) ∧
  sum_first_n_terms a 2018 = 4732 :=
sorry

end S_2018_eq_4732_l147_147729


namespace problem_statement_l147_147817

theorem problem_statement (n : ℕ) (hn : 0 < n) (h : ∃ k : ℤ, (1 / 2 : ℚ) + (1 / 3) + (1 / 5) + (1 / n) = k) : ¬(2 ∣ n) := 
begin
  have h_n_15 : n = 15, from sorry,
  have h2_dvd_15 : ¬(2 ∣ 15), from sorry,
  exact h2_dvd_15,
end

end problem_statement_l147_147817


namespace sufficient_but_not_necessary_condition_l147_147574

theorem sufficient_but_not_necessary_condition 
  (a b : ℝ) (h : a > b ∧ b > 0) : (a^2 > b^2) ∧ (¬ ∀ (a' b' : ℝ), a'^2 > b'^2 → a' > b' ∧ b' > 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_l147_147574


namespace train_must_pass_through_marked_segment_l147_147523

-- Definitions for the problem setup
def station := string -- a station is represented by its name as string
def line := list station -- a line is represented by a list of stations

-- Given conditions
noncomputable def condition1 : list line := [
  ["A", "B", "C", "D", "E"], -- line 1
  ["F", "G", "H", "I", "J"], -- line 2
  ["K", "L", "M", "N", "O"] -- line 3
]
noncomputable def condition2 := (∀ (t : ℕ) (s : station), t = t + 1)  -- Train travels one section per minute
noncomputable def condition3 := (∀ (s : station), s ∈ ["E", "J", "O"] → s = "E") -- Train turns back at terminal
noncomputable def condition4 := (∀ (s : station), s ∈ (["B", "D", "G", "I", "L", "N"])) -- Train must switch lines at transfer stations
noncomputable def condition5 := (A : station) → (B : station) → (time : ℕ) := 
  (A = "A") ∧ (B = "B") ∧ (time = 2016)

-- Proof obligation
theorem train_must_pass_through_marked_segment (A B : station) (time : ℕ) :
  condition5 A B time → 
  (∃ segment : (station × station), (segment = ("marked_start", "marked_end"))) :=
by
  intro h
  sorry

end train_must_pass_through_marked_segment_l147_147523


namespace largest_multiple_15_under_500_l147_147088

-- Define the problem statement
theorem largest_multiple_15_under_500 : ∃ (n : ℕ), n < 500 ∧ n % 15 = 0 ∧ (∀ m, m < 500 ∧ m % 15 = 0 → m ≤ n) :=
by {
  use 495,
  split,
  {
    exact lt_of_le_of_ne (le_refl _) (ne_of_lt (nat.lt_succ_self 499))
  },
  split,
  {
    exact nat.mod_eq_zero_of_dvd ⟨33, rfl⟩,
  },
  {
    intros m hm h,
    have hmn : m ≤ 495,
    {
      have div15 := hm/15,
      suffices : div15 < 33+1, from nat.le_of_lt_add_one this,
      exact this, sorry,
    },
    exact hmn,
  },
}

end largest_multiple_15_under_500_l147_147088


namespace radius_of_complex_roots_l147_147223

noncomputable def complex_radius (z : ℂ) : ℝ :=
  if (z + 2)^6 = 64 * z^6 then |z + 2| / |z| else 0

theorem radius_of_complex_roots :
  ∀ z : ℂ, (z + 2)^6 = 64 * z^6 → complex_radius z = 2 / Real.sqrt 3 :=
by sorry

end radius_of_complex_roots_l147_147223


namespace proof_problem_l147_147669

def sqrt_frac : ℚ := real.sqrt (16 / 9)
def frac : ℚ := 16 / 9
def square_frac : ℚ := frac * frac

def ceil_sqrt_frac : ℤ := ⌈sqrt_frac⌉.to_int
def ceil_frac : ℤ := ⌈frac⌉.to_int
def ceil_square_frac : ℤ := ⌈square_frac⌉.to_int

theorem proof_problem :
  ceil_sqrt_frac + ceil_frac + ceil_square_frac = 8 :=
by
  -- Placeholder for the actual proof.
  sorry

end proof_problem_l147_147669


namespace true_propositions_l147_147232

-- Definition of each condition
def proposition1 : Prop := ∀ {x y : ℝ}, (x * y = 1) → (y = 1 / x)
def proposition2 : Prop := ∀ {A B : Triangle}, (area(A) ≠ area(B)) → (¬congruent(A, B))
def proposition3 : Prop := ∀ {x y : ℝ}, (x + y ≠ 3) → (x ≠ 1 ∨ y ≠ 2)
def proposition4 : Prop := ∀ x : ℝ, 4 * x^2 - 4 * x + 1 > 0

-- The theorem we want to prove
theorem true_propositions : 
  proposition1 ∧ 
  proposition2 ∧ 
  ¬proposition3 ∧ 
  proposition4 :=
by
  sorry

end true_propositions_l147_147232


namespace range_of_x_when_a_eq_1_p_and_q_range_of_a_when_not_p_sufficient_for_not_q_l147_147844

-- Define the propositions
def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := -x^2 + 5 * x - 6 ≥ 0

-- Question 1: Prove that for a = 1 and p ∧ q is true, the range of x is [2, 3)
theorem range_of_x_when_a_eq_1_p_and_q : 
  ∀ x : ℝ, p 1 x ∧ q x → 2 ≤ x ∧ x < 3 := 
by sorry

-- Question 2: Prove that if ¬p is a sufficient but not necessary condition for ¬q, 
-- then the range of a is (1, 2)
theorem range_of_a_when_not_p_sufficient_for_not_q :
  ∀ a : ℝ, (∀ x : ℝ, ¬p a x → ¬q x) ∧ (∃ x : ℝ, ¬(¬p a x → ¬q x)) → 1 < a ∧ a < 2 := 
by sorry

end range_of_x_when_a_eq_1_p_and_q_range_of_a_when_not_p_sufficient_for_not_q_l147_147844


namespace sequence_bounds_l147_147759

theorem sequence_bounds (a : ℕ → ℝ) (H1 : a 1 = 1) (H2 : ∀ n : ℕ, 0 < n → a (n+1) = a n + (a n)^(1/n.to_real)) 
  (n : ℕ) (h : 0 < n) :
  n.to_real ≤ a n ∧ a n < 2 * (n.to_real) ∧ a n ≤ n.to_real + 4 * (n.to_real - 1).sqrt := 
sorry

end sequence_bounds_l147_147759


namespace largest_multiple_of_15_less_than_500_l147_147131

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), 15 * n < 500 ∧ 15 * n ≥ 15 * (33 - 1) + 1 := by
  sorry

end largest_multiple_of_15_less_than_500_l147_147131


namespace triangle_ABC_area_l147_147879

def point : Type := ℝ × ℝ

def A : point := (2, -1)
def B : point := (3, 1)
def C : point := (2^1999, 2^2000)

def triangle_area (A B C : point) : ℝ :=
  0.5 * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2) |

theorem triangle_ABC_area :
  triangle_area A B C = 2.5 :=
by
  sorry

end triangle_ABC_area_l147_147879


namespace sqrt_sum_form_l147_147388

theorem sqrt_sum_form (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : sqrt 6 + 1 / sqrt 6 + sqrt 8 + 1 / sqrt 8 = (a * sqrt 6 + b * sqrt 8) / c)
  (hc_min : ∀ d, ∃ a' b', d > 0 → sqrt 6 + 1 / sqrt 6 + sqrt 8 + 1 / sqrt 8 = (a' * sqrt 6 + b' * sqrt 8) / d  → d ≥ c) :
  a + b + c = 192 :=
by
  sorry

end sqrt_sum_form_l147_147388


namespace largest_multiple_of_15_less_than_500_l147_147025

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), n < 500 ∧ 15 ∣ n ∧ ∀ (m : ℕ), m < 500 ∧ 15 ∣ m -> m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l147_147025


namespace ceiling_sum_evaluation_l147_147687

noncomputable def evaluateCeilingSum : ℝ := 
  ⌈Real.sqrt (16 / 9)⌉ + ⌈(16 / 9)⌉ + ⌈((16 / 9) ^ 2)⌉ 

theorem ceiling_sum_evaluation : evaluateCeilingSum = 8 := by
  sorry

end ceiling_sum_evaluation_l147_147687


namespace kibble_remaining_l147_147457

theorem kibble_remaining 
  (initial_amount : ℕ) (morning_mary : ℕ) (evening_mary : ℕ) 
  (afternoon_frank : ℕ) (evening_frank : ℕ) :
  initial_amount = 12 →
  morning_mary = 1 →
  evening_mary = 1 →
  afternoon_frank = 1 →
  evening_frank = 2 * afternoon_frank →
  initial_amount - (morning_mary + evening_mary + afternoon_frank + evening_frank) = 7 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  done

end kibble_remaining_l147_147457


namespace remainder_of_777_pow_444_mod_13_l147_147926

theorem remainder_of_777_pow_444_mod_13 :
  (777 ^ 444) % 13 = 1 :=
by 
  have h1 : 777 % 13 = 10 := by sorry,
  have h2 : 10 ^ 6 % 13 = 1 := by sorry,
  sorry

end remainder_of_777_pow_444_mod_13_l147_147926


namespace find_real_numbers_l147_147282

noncomputable def satisfies_equation (x y z : ℝ) : Prop := 
    x + y + z + 3 / (x - 1) + 3 / (y - 1) + 3 / (z - 1) = 
    2 * (Real.sqrt (x + 2) + Real.sqrt (y + 2) + Real.sqrt (z + 2))

theorem find_real_numbers (x y z : ℝ) (hx : x > 1) (hy : y > 1) (hz : z > 1) :
    satisfies_equation x y z ↔ x = (3 + Real.sqrt 13) / 2 ∧ y = (3 + Real.sqrt 13) / 2 ∧ z = (3 + Real.sqrt 13) / 2 :=
sorry

end find_real_numbers_l147_147282


namespace probability_different_colors_l147_147907

theorem probability_different_colors (red_chips green_chips : ℕ) (total_chips : ℕ)
  (h_red : red_chips = 7) (h_green : green_chips = 5) (h_total : total_chips = 7 + 5) :
  let p_diff_colors := (red_chips / total_chips) * (green_chips / total_chips) +
                       (green_chips / total_chips) * (red_chips / total_chips)
  in p_diff_colors = 35 / 72 :=
by {
  -- Proof goes here
  sorry
}

end probability_different_colors_l147_147907


namespace f_f_neg2_eq_5_l147_147772

def f (x : ℝ) : ℝ :=
  if x > 0 then x + 2 else x^2 - 1

theorem f_f_neg2_eq_5 : f (f (-2)) = 5 := by
  sorry

end f_f_neg2_eq_5_l147_147772


namespace not_in_range_l147_147288

noncomputable def f (b x : ℝ) := x^2 + b * x + 5

theorem not_in_range (b : ℝ) : (¬∃ x : ℝ, f b x = -2) ↔ b ∈ Ioo (-Real.sqrt 28) (Real.sqrt 28) := by
  sorry

end not_in_range_l147_147288


namespace percentage_reduction_in_side_length_l147_147550

theorem percentage_reduction_in_side_length (a : ℝ) (h : a = 1) (area_loss : (1 * 1) - (3 / 4) = 1 / 4) :
  let new_side_length := real.sqrt (3 / 4)
  (1 - new_side_length) * 100 ≈ 13 :=
sorry

end percentage_reduction_in_side_length_l147_147550


namespace partial_fraction_constant_sum_zero_l147_147261

noncomputable def partial_fraction_sum (A B C D E F : ℚ) : Prop :=
  1 = A * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5) +
      B * x * (x + 2) * (x + 3) * (x + 4) * (x + 5) +
      C * x * (x + 1) * (x + 3) * (x + 4) * (x + 5) +
      D * x * (x + 1) * (x + 2) * (x + 4) * (x + 5) +
      E * x * (x + 1) * (x + 2) * (x + 3) * (x + 5) +
      F * x * (x + 1) * (x + 2) * (x + 3) * (x + 4)

theorem partial_fraction_constant_sum_zero (A B C D E F : ℚ) (h : partial_fraction_sum A B C D E F) : 
  A + B + C + D + E + F = 0 :=
sorry

end partial_fraction_constant_sum_zero_l147_147261


namespace alex_natural_growth_per_month_l147_147610

-- Define the given conditions
def height_required : ℝ := 54
def current_height : ℝ := 48
def growth_per_hour_upside_down : ℝ := 1 / 12
def hours_per_month_upside_down : ℝ := 2
def months_per_year : ℝ := 12

-- Define the question to be proven: Alex grows 1/3 inch per month naturally
theorem alex_natural_growth_per_month :
  let total_growth_needed := height_required - current_height in
  let growth_per_month_upside_down := hours_per_month_upside_down * growth_per_hour_upside_down in
  let total_growth_per_year_upside_down := growth_per_month_upside_down * months_per_year in
  let natural_growth_per_year := total_growth_needed - total_growth_per_year_upside_down in
  let natural_growth_per_month := natural_growth_per_year / months_per_year in
  natural_growth_per_month = 1 / 3 :=
by
  sorry

end alex_natural_growth_per_month_l147_147610


namespace problem_solution_l147_147351

theorem problem_solution {f : ℝ → ℝ} (m n a : ℝ) :
  (f = λ x, m * x^2 - 2 * x - 3) →
  ∀ x, f x < 0 ↔ (-1 < x ∧ x < n) →
  (m = 1 ∧ n = 3) →
  (∀ x, 2 * x^2 - 4 * x + n > (m + 1) * x - 1 ↔ x ∈ Icc (-∞) 1 ∪ Icc 2 ∞) ∧
  (∃ a, a ∈ Ioo 0 1 ∧ ∀ x, x ∈ Icc 1 2 → (let t := a^x in f (a^x) - 4 * a^(x + 1) ≥ -4))
  :=
by 
  sorry

end problem_solution_l147_147351


namespace largest_number_is_C_l147_147563

theorem largest_number_is_C (A B C D E : ℝ) 
  (hA : A = 0.989) 
  (hB : B = 0.9098) 
  (hC : C = 0.9899) 
  (hD : D = 0.9009) 
  (hE : E = 0.9809) : 
  C > A ∧ C > B ∧ C > D ∧ C > E := 
by 
  sorry

end largest_number_is_C_l147_147563


namespace largest_multiple_15_under_500_l147_147089

-- Define the problem statement
theorem largest_multiple_15_under_500 : ∃ (n : ℕ), n < 500 ∧ n % 15 = 0 ∧ (∀ m, m < 500 ∧ m % 15 = 0 → m ≤ n) :=
by {
  use 495,
  split,
  {
    exact lt_of_le_of_ne (le_refl _) (ne_of_lt (nat.lt_succ_self 499))
  },
  split,
  {
    exact nat.mod_eq_zero_of_dvd ⟨33, rfl⟩,
  },
  {
    intros m hm h,
    have hmn : m ≤ 495,
    {
      have div15 := hm/15,
      suffices : div15 < 33+1, from nat.le_of_lt_add_one this,
      exact this, sorry,
    },
    exact hmn,
  },
}

end largest_multiple_15_under_500_l147_147089


namespace largest_multiple_of_15_less_than_500_l147_147007

theorem largest_multiple_of_15_less_than_500 : ∃ n : ℕ, (n > 0 ∧ 15 * n < 500) ∧ (∀ m : ℕ, m > n → 15 * m >= 500) ∧ 15 * n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l147_147007


namespace tangent_line_inv_g_at_0_l147_147752

noncomputable def g (x : ℝ) := Real.log x

theorem tangent_line_inv_g_at_0 
  (h₁ : ∀ x, g x = Real.log x) 
  (h₂ : ∀ x, x > 0): 
  ∃ m b, (∀ x y, y = g⁻¹ x → y - m * x = b) ∧ 
         (m = 1) ∧ 
         (b = 1) ∧ 
         (∀ x y, x - y + 1 = 0) := 
by
  sorry

end tangent_line_inv_g_at_0_l147_147752


namespace find_initial_quantities_l147_147904

/-- 
Given:
- x + y = 92
- (2/5) * x + (1/4) * y = 26

Prove:
- x = 20
- y = 72
-/
theorem find_initial_quantities (x y : ℝ) (h1 : x + y = 92) (h2 : (2/5) * x + (1/4) * y = 26) :
  x = 20 ∧ y = 72 :=
sorry

end find_initial_quantities_l147_147904


namespace Q_n_large_l147_147248

noncomputable def T (n : ℕ) : ℕ := n * (n + 1) / 2

noncomputable def Q (n : ℕ) : ℝ :=
  let prod_expr : ℝ := ∏ k in Finset.range (n - 2) + 3, (Nat.factorial (T k)) / (Nat.factorial (T k - 1) : ℝ)
  prod_expr

theorem Q_n_large (n : ℕ) (hn : n = 50) : ∃ M : ℝ, Q n > M :=
by
  use 10^10 -- some large number
  sorry

end Q_n_large_l147_147248


namespace intersection_of_sets_l147_147320

-- Definition of sets A and B as conditions in the mathematical problem.
def setA : Set ℝ := {x | x^2 - 4 * x - 5 ≤ 0}
def setB : Set ℝ := {x | real.log x / real.log 2 < 2}

-- The theorem stating that the intersection of sets A and B equals (0, 4).
theorem intersection_of_sets : (setA ∩ setB) = {x | 0 < x ∧ x < 4} :=
by sorry

end intersection_of_sets_l147_147320


namespace sum_solutions_eq_10000_l147_147754

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 2 * x else f (x - 1) + 1

-- The main theorem
theorem sum_solutions_eq_10000 :
  let solutions := {x : ℝ | 0 ≤ x ∧ x ≤ 100 ∧ f x = x - 1 / 5}
  ∑ x in solutions, x = 10000 := sorry

end sum_solutions_eq_10000_l147_147754


namespace integer_solutions_of_quadratic_eq_l147_147711

theorem integer_solutions_of_quadratic_eq (b : ℤ) :
  ∃ p q : ℤ, (p+9) * (q+9) = 81 ∧ p + q = -b ∧ p * q = 9*b :=
sorry

end integer_solutions_of_quadratic_eq_l147_147711


namespace gcd_102_238_l147_147510

-- Define the two numbers involved
def a : ℕ := 102
def b : ℕ := 238

-- State the theorem
theorem gcd_102_238 : Int.gcd a b = 34 :=
by
  sorry

end gcd_102_238_l147_147510


namespace largest_multiple_of_15_less_than_500_l147_147020

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), n < 500 ∧ 15 ∣ n ∧ ∀ (m : ℕ), m < 500 ∧ 15 ∣ m -> m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l147_147020


namespace hermia_elected_probability_l147_147838

noncomputable def probability_h ispected_president (n : ℕ) (h : n % 2 = 1) : ℚ :=
  (2^n - 1) / (n * 2^(n-1))

theorem hermia_elected_probability (n : ℕ) (h : n % 2 = 1) :
  let P := probability_h ispected_president n h in 
  hermia_elected_probability = P := 
  sorry

end hermia_elected_probability_l147_147838


namespace largest_multiple_of_15_below_500_l147_147028

theorem largest_multiple_of_15_below_500 : ∃ (k : ℕ), 15 * k < 500 ∧ ∀ (m : ℕ), 15 * m < 500 → 15 * m ≤ 15 * k := 
by
  existsi 33
  split
  · norm_num
  · intro m h
    have h1 : m ≤ 33 := Nat.le_of_lt_succ (Nat.div_lt_succ 500 15 m)
    exact (mul_le_mul_left (by norm_num : 0 < 15)).mpr h1
  sorry

end largest_multiple_of_15_below_500_l147_147028


namespace smallest_positive_period_of_f_max_min_value_of_f_l147_147747

def f (x : ℝ) := Real.cos x * (Real.sin x - Real.cos x)

theorem smallest_positive_period_of_f : ∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ π :=
begin
  sorry
end

theorem max_min_value_of_f : 
  let I := Set.Icc (-(Real.pi / 4)) (Real.pi / 4)
  in ∃ (M m : ℝ), (∀ x ∈ I, f x ≤ M) ∧ (∀ x ∈ I, f x ≥ m) ∧ M = 0 ∧ m = (- (Real.sqrt 2 + 1) / 2) :=
begin
  sorry
end

end smallest_positive_period_of_f_max_min_value_of_f_l147_147747


namespace max_min_difference_l147_147445

variable (x y z : ℝ)

theorem max_min_difference :
  x + y + z = 3 →
  x^2 + y^2 + z^2 = 18 →
  (max z (-z)) - ((min z (-z))) = 6 :=
  by
    intros h1 h2
    sorry

end max_min_difference_l147_147445


namespace total_investment_is_3000_interest_rate_is_8percent_l147_147862

-- Define the given conditions as assumptions
variable (total_investment : ℝ := 3000)
variable (yearly_interest : ℝ := 256)
variable (invested_800 : ℝ := 800)
variable (rate_10percent : ℝ := 0.10)

-- Define the interest calculation for the $800 investment
def interest_from_800 := invested_800 * rate_10percent

-- Calculate the remaining interest
def remaining_interest := yearly_interest - interest_from_800

-- Calculate the remaining investment
def remaining_investment := total_investment - invested_800

-- Define the interest rate for the remaining investment
def interest_rate_remaining := remaining_interest / remaining_investment

-- Prove the questions using the given conditions
theorem total_investment_is_3000 : total_investment = 3000 := by
  sorry

theorem interest_rate_is_8percent : interest_rate_remaining = 0.08 := by
  sorry

end total_investment_is_3000_interest_rate_is_8percent_l147_147862


namespace hermia_elected_probability_l147_147839

-- Define the problem statement and conditions in Lean 4
noncomputable def probability_hermia_elected (n : ℕ) (h_odd : (n % 2 = 1)) (h_pos : n > 0) : ℚ :=
  if n = 1 then 1 else (2^n - 1) / (n * 2^(n-1))

-- Lean theorem statement
theorem hermia_elected_probability (n : ℕ) (h_odd : (n % 2 = 1)) (h_pos : n > 0) : 
  probability_hermia_elected n h_odd h_pos = (2^n - 1) / (n * 2^(n-1)) :=
by
  sorry

end hermia_elected_probability_l147_147839


namespace largest_multiple_of_15_less_than_500_l147_147143

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), 15 * n < 500 ∧ 15 * n ≥ 15 * (33 - 1) + 1 := by
  sorry

end largest_multiple_of_15_less_than_500_l147_147143


namespace obtuse_triangle_two_acute_angles_l147_147362

-- Define the angle type (could be Real between 0 and 180 in degrees).
def is_acute (θ : ℝ) : Prop := 0 < θ ∧ θ < 90

def is_obtuse (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

-- Define an obtuse triangle using three angles α, β, γ
structure obtuse_triangle :=
(angle1 angle2 angle3 : ℝ)
(sum_angles_eq : angle1 + angle2 + angle3 = 180)
(obtuse_condition : is_obtuse angle1 ∨ is_obtuse angle2 ∨ is_obtuse angle3)

-- The theorem to prove the number of acute angles in an obtuse triangle is 2.
theorem obtuse_triangle_two_acute_angles (T : obtuse_triangle) : 
  (is_acute T.angle1 ∧ is_acute T.angle2 ∧ ¬ is_acute T.angle3) ∨ 
  (is_acute T.angle1 ∧ ¬ is_acute T.angle2 ∧ is_acute T.angle3) ∨ 
  (¬ is_acute T.angle1 ∧ is_acute T.angle2 ∧ is_acute T.angle3) :=
by sorry

end obtuse_triangle_two_acute_angles_l147_147362


namespace line_through_point_parallel_l147_147886

theorem line_through_point_parallel (c : ℝ) :
  (∀ x y : ℝ, (2 * x - 3 * y + 4 = 0) ↔ (2 * x - 3 * y + c = 0))
  ∧ (2 * (-1) - 3 * 2 + c = 0) → (c = 8) :=
begin
  intro h,
  cases h with H1 H2,
  have h_c : c = 8,
  {
    linarith,
  },
  exact h_c
end

end line_through_point_parallel_l147_147886


namespace probability_is_correct_l147_147792

noncomputable def probability_total_more_than_7 : ℚ :=
  let total_outcomes := 36
  let favorable_outcomes := 15
  favorable_outcomes / total_outcomes

theorem probability_is_correct :
  probability_total_more_than_7 = 5 / 12 :=
by
  sorry

end probability_is_correct_l147_147792


namespace alex_age_is_15_l147_147781

-- Defining the variables and equations based on the conditions
variables (A P : ℤ)

-- Problem conditions
def condition1 := A + 3 = 3 * (P + 3)
def condition2 := A - 1 = 7 * (P - 1)

-- Statement that needs to be proven
theorem alex_age_is_15 (h1 : condition1) (h2 : condition2) : A = 15 := sorry

end alex_age_is_15_l147_147781


namespace largest_multiple_of_15_less_than_500_l147_147049

theorem largest_multiple_of_15_less_than_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ (∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x) := sorry

end largest_multiple_of_15_less_than_500_l147_147049


namespace range_of_a_l147_147745

-- Define the piecewise function f(x)
def f (x a : ℝ) : ℝ :=
  if x ≤ 0 then 2^x + a else x + 4 / x

-- Define the theorem statement asserting the range of a for which f(x) has a minimum value
theorem range_of_a (a : ℝ) : (∃ x : ℝ, ∀ y : ℝ, f y a ≥ f x a) ↔ 4 ≤ a :=
by
  apply Iff.intro
  sorry

end range_of_a_l147_147745


namespace queens_probability_l147_147922

theorem queens_probability :
  let l := 64 * 63,
      k := 28 * 21 + 20 * 23 + 12 * 25 + 4 * 27,
      v₁ := k / l,
      v₃ := v₁^3,
      v := 1 - v₃
  in v = 0.953 :=
by
  sorry

end queens_probability_l147_147922


namespace quadratic_points_order_l147_147735

theorem quadratic_points_order (y1 y2 y3 : ℝ) :
  (y1 = -2 * (1:ℝ) ^ 2 + 4) →
  (y2 = -2 * (2:ℝ) ^ 2 + 4) →
  (y3 = -2 * (-3:ℝ) ^ 2 + 4) →
  y1 > y2 ∧ y2 > y3 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end quadratic_points_order_l147_147735


namespace simplify_fraction_expr_l147_147871

theorem simplify_fraction_expr (a : ℝ) (h : a ≠ 1) : (a / (a - 1) + 1 / (1 - a)) = 1 := by
  sorry

end simplify_fraction_expr_l147_147871


namespace number_of_unique_patterns_l147_147545

/-
Problem: Given a 4x4 grid and using eight 2x1 tiles, we want to prove that the number of unique and vertically symmetrical configurations with two adjacent blank squares in the first row is 5.
-/

def two_by_one_tile : Type := {xy : ℕ × ℕ // xy.1 < 4 ∧ xy.2 < 4}

def is_valid_tile_placement (placements : list two_by_one_tile) : Prop :=
  (∀ xy ∈ placements, xy.2 = (xy.1 + 1, xy.2)) ∧ 
  (list.length placements = 8) ∧ 
  (∀ i j ∈ placements, i ≠ j → i.1 ≠ j.1)

def has_adjacent_blanks (row : list ℕ) : Prop :=
  (row = [0, 0, 1, 1] ∨ row = [1, 0, 0, 1] ∨ row = [1, 1, 0, 0]) 

def is_vertically_symmetrical (placements : list two_by_one_tile) : Prop :=
  ∀ xy ∈ placements, (3 - xy.1, xy.2) ∈ placements

theorem number_of_unique_patterns :
  ∃ placements : list (list ℕ),
    (∀ row ∈ placements, is_valid_tile_placement row) ∧ 
    (has_adjacent_blanks (placements.head)) ∧ 
    (is_vertically_symmetrical placements) ∧
    (list.length (quotient.mk' (placements.map is_valid_tile_placement)) = 5) :=
sorry

end number_of_unique_patterns_l147_147545


namespace largest_no_grey_faces_l147_147468

theorem largest_no_grey_faces (total_cubes : ℕ) (interior_cubes : ℕ) : 
  total_cubes = 27 → interior_cubes = 1 → 
  ∃ max_no_grey_cubes: ℕ, max_no_grey_cubes = 15 := 
by
  intros _ _
  exists_exactly 15
  sorry

end largest_no_grey_faces_l147_147468


namespace find_initial_men_l147_147804

noncomputable def initial_men_planned (M : ℕ) : Prop :=
  let initial_days := 10
  let additional_days := 20
  let total_days := initial_days + additional_days
  let men_sent := 25
  let initial_work := M * initial_days
  let remaining_men := M - men_sent
  let remaining_work := remaining_men * total_days
  initial_work = remaining_work 

theorem find_initial_men :
  ∃ M : ℕ, initial_men_planned M ∧ M = 38 :=
by
  have h : initial_men_planned 38 :=
    by
      sorry
  exact ⟨38, h, rfl⟩

end find_initial_men_l147_147804


namespace class_funds_l147_147419

theorem class_funds (total_contribution : ℕ) (students : ℕ) (contribution_per_student : ℕ) (remaining_amount : ℕ) 
    (h1 : total_contribution = 90) 
    (h2 : students = 19) 
    (h3 : contribution_per_student = 4) 
    (h4 : remaining_amount = total_contribution - (students * contribution_per_student)) : 
    remaining_amount = 14 :=
sorry

end class_funds_l147_147419


namespace largest_multiple_of_15_less_than_500_l147_147140

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), 15 * n < 500 ∧ 15 * n ≥ 15 * (33 - 1) + 1 := by
  sorry

end largest_multiple_of_15_less_than_500_l147_147140


namespace neither_sufficient_nor_necessary_l147_147371

variable (a b : ℝ)

theorem neither_sufficient_nor_necessary (h1 : 0 < a * b ∧ a * b < 1) : ¬ (b < 1 / a) ∨ ¬ (1 / a < b) := by
  sorry

end neither_sufficient_nor_necessary_l147_147371


namespace largest_multiple_of_15_less_than_500_l147_147042

theorem largest_multiple_of_15_less_than_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ (∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x) := sorry

end largest_multiple_of_15_less_than_500_l147_147042


namespace magnitude_of_vector_addition_linear_combination_solution_collinear_points_l147_147849

open Real

-- Definitions of the vectors
def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (1, -1)
def c : ℝ × ℝ := (4, -5)

-- Problem statement part (Ⅰ)
theorem magnitude_of_vector_addition :
  |((a.1 + 2 * b.1), (a.2 + 2 * b.2))| = 1 := sorry

-- Problem statement part (Ⅱ)
theorem linear_combination_solution (λ μ : ℝ) :
  (c = (λ * a.1 + μ * b.1, λ * a.2 + μ * b.2)) →
  (λ + μ = 2) := sorry

-- Problem statement part (Ⅲ)
theorem collinear_points (u v w : ℝ × ℝ) :
  (u = (-1, 2) + (1, -1)) → 
  (v = (-1, 2) - 2 * (1, -1)) →
  (w = 4 * (-1, 2) - 2 * (1, -1)) →
  (∃ k : ℝ, w = k • v) :=
  sorry

end magnitude_of_vector_addition_linear_combination_solution_collinear_points_l147_147849


namespace evaluate_expression_l147_147661

theorem evaluate_expression :
  let x := (16 : ℚ) / 9
  in ⌈(√x)⌉ + ⌈x⌉ + ⌈x^2⌉ = 8 :=
by
  let x := (16 : ℚ) / 9
  sorry

end evaluate_expression_l147_147661


namespace question_1_question_2_l147_147751

open Real

noncomputable def f (x a : ℝ) := abs (x - a) + 3 * x

theorem question_1 :
  {x : ℝ | f x 1 > 3 * x + 2} = {x : ℝ | x > 3 ∨ x < -1} :=
by 
  sorry
  
theorem question_2 (h : {x : ℝ | f x a ≤ 0} = {x : ℝ | x ≤ -1}) :
  a = 2 :=
by 
  sorry

end question_1_question_2_l147_147751


namespace sum_of_perimeters_l147_147207

theorem sum_of_perimeters (n : ℕ) (h : n > 4) : 
  let P1 := 120 in 
  Σ i in Finset.range n, P1 / 2 ^ i = 240 * (1 - (1 / 2) ^ n) :=
by
  sorry

end sum_of_perimeters_l147_147207


namespace card_arrangements_limit_l147_147312

variables (n : ℕ)
def deck_length := 2 * n + 1

-- Define possible operations
inductive operation
| moveTopToBottom : ℕ → operation -- move a portion of cards from top to bottom
| insertTopNIntoBottom : operation -- insert the top n cards into n gaps in the bottom n+1 cards

-- Define the condition for the problem
def is_valid_arrangement (arrangement : list (fin (deck_length n))) : Prop :=
  ∀ (i j k : ℕ) (h₀ : i < j) (h₁ : j < k) (h₂ : k < deck_length n),
    (arrangement.nth_le i h₀ - 2 * arrangement.nth_le j h₁ + arrangement.nth_le k h₂) % (deck_length n) = 0

-- Proof problem
theorem card_arrangements_limit (n : ℕ) :
  numArrangements (deck_length n) ≤ 2 * n * (2 * n + 1) :=
sorry

end card_arrangements_limit_l147_147312


namespace largest_prime_factor_expr_l147_147272

noncomputable def expr : ℤ := 20^3 + 15^4 - 10^5

theorem largest_prime_factor_expr : ∃ p : ℕ, prime p ∧ p = 41 ∧ (∀ q : ℕ, prime q ∧ q ∣ expr → q ≤ 41) :=
by {
  sorry
}

end largest_prime_factor_expr_l147_147272


namespace largest_multiple_of_15_less_than_500_l147_147137

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), 15 * n < 500 ∧ 15 * n ≥ 15 * (33 - 1) + 1 := by
  sorry

end largest_multiple_of_15_less_than_500_l147_147137


namespace union_of_sets_l147_147352

def setA := { x : ℝ | -1 ≤ 2 * x + 1 ∧ 2 * x + 1 ≤ 3 }
def setB := { x : ℝ | (x - 2) / x ≤ 0 }

theorem union_of_sets :
  { x : ℝ | -1 ≤ x ∧ x ≤ 2 } = setA ∪ setB :=
by
  sorry

end union_of_sets_l147_147352


namespace enrique_sold_6_shirts_l147_147255

noncomputable def calculate_commission (sales : ℕ → ℝ) (percent : ℝ) : ℝ := sales * percent

theorem enrique_sold_6_shirts (earnings : ℝ)
  (commission_rate : ℝ)
  (suit_price : ℝ)
  (loafers_price : ℝ)
  (shirt_price : ℝ)
  (suit_quantity : ℕ)
  (loafers_quantity : ℕ)
  (total_commission : ℝ)
  (shirt_sold_commission : ℝ)
  (shirts_quantity_calculated : ℕ):
  earnings = 300 →
  commission_rate = 0.15 →
  suit_price = 700 →
  loafers_price = 150 →
  shirt_price = 50 →
  suit_quantity = 2 →
  loafers_quantity = 2 →
  shirts_quantity_calculated = (total_commission - (calculate_commission (suit_quantity * suit_price) commission_rate + calculate_commission (loafers_quantity * loafers_price) commission_rate)) / calculate_commission shirt_price commission_rate →
  total_commission = earnings →
  shirts_quantity_calculated = 6 :=
by
  intros
  sorry

end enrique_sold_6_shirts_l147_147255


namespace range_of_a_l147_147775

noncomputable def f (x : ℝ) : ℝ := 6 / x - x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x → x^2 + a * x - 6 > 0) ↔ 5 ≤ a :=
by
  sorry

end range_of_a_l147_147775


namespace proposition_not_hold_for_4_l147_147969

variable (P : ℕ → Prop)

axiom induction_step (k : ℕ) (hk : k > 0) : P k → P (k + 1)
axiom base_case : ¬ P 5

theorem proposition_not_hold_for_4 : ¬ P 4 :=
sorry

end proposition_not_hold_for_4_l147_147969


namespace cos_B_given_conditions_l147_147402

noncomputable theory

-- Define the variables for the triangle
variables {A B C : Type} [real A] [real B] [real C]

-- Condition definitions
def a : ℝ := 15
def b : ℝ := 10
def angle_A : ℝ := 60 * real.pi / 180 -- Converting to radians

-- Define the Law of Sines (only necessary part for the given proof)
def law_of_sines (a b : ℝ) (A B : ℝ) : ℝ := (15 / (real.sqrt 3 / 2)) = (10 / (real.sin B))

-- Problem statement equivalent in Lean
theorem cos_B_given_conditions (h1 : a = 15) (h2 : b = 10) (h3 : angle_A = 60 * real.pi / 180)
: ∃ (B : ℝ), real.cos B = real.sqrt 6 / 3 :=
sorry

end cos_B_given_conditions_l147_147402


namespace number_line_move_l147_147377

theorem number_line_move (A B: ℤ):  A = -3 → B = A + 4 → B = 1 := by
  intros hA hB
  rw [hA] at hB
  rw [hB]
  sorry

end number_line_move_l147_147377


namespace largest_multiple_of_15_below_500_l147_147033

theorem largest_multiple_of_15_below_500 : ∃ (k : ℕ), 15 * k < 500 ∧ ∀ (m : ℕ), 15 * m < 500 → 15 * m ≤ 15 * k := 
by
  existsi 33
  split
  · norm_num
  · intro m h
    have h1 : m ≤ 33 := Nat.le_of_lt_succ (Nat.div_lt_succ 500 15 m)
    exact (mul_le_mul_left (by norm_num : 0 < 15)).mpr h1
  sorry

end largest_multiple_of_15_below_500_l147_147033


namespace theta_calculation_l147_147618

open Complex

noncomputable def calculate_theta : ℝ :=
  ∑ k in ({3, 11, 19, 27, 35, 43, 51, 59} : Finset ℕ), exp (↑k * π * I / 60)

theorem theta_calculation :
  (calculate_theta == (8 * exp (31 * π * I / 60))) :=
by
  sorry

end theta_calculation_l147_147618


namespace parabola_directrix_p_l147_147756

/-- Given a parabola with equation y^2 = 2px and directrix x = -2, prove that p = 4 -/
theorem parabola_directrix_p (p : ℝ) :
  (∀ y x : ℝ, y^2 = 2 * p * x) ∧ (∀ x : ℝ, x = -2 → True) → p = 4 :=
by
  sorry

end parabola_directrix_p_l147_147756


namespace tetrahedron_medians_midpoints_intersect_single_point_l147_147481

structure Tetrahedron :=
(A B C D : Point)

def is_median (A B C D : Point) (M : Point) :=
    -- (Assumption of median logic for faces)
    M is the intersection point of the medians of face ABC

def is_midpoint (A B : Point) (K : Point) :=
    -- (Assumption of midpoint definition)
    K is the midpoint of AB

theorem tetrahedron_medians_midpoints_intersect_single_point
    (tet : Tetrahedron)
    (M1 : Point) (M2 : Point) (M3 : Point)
    (K1 : Point) (K2 : Point) (K3 : Point)
    (O : Point) :
    is_median tet.A tet.B tet.C tet.D M1 ∧
    is_median tet.A tet.C tet.B tet.D M2 ∧
    is_median tet.A tet.D tet.B tet.C M3 ∧
    is_midpoint tet.A tet.B K1 ∧
    is_midpoint tet.C tet.D K2 ∧
    is_midpoint tet.B tet.D K3 →
    intersect_at_one_point {M1, M2, M3, K1, K2, K3} O :=
sorry

end tetrahedron_medians_midpoints_intersect_single_point_l147_147481


namespace ceiling_sum_evaluation_l147_147685

noncomputable def evaluateCeilingSum : ℝ := 
  ⌈Real.sqrt (16 / 9)⌉ + ⌈(16 / 9)⌉ + ⌈((16 / 9) ^ 2)⌉ 

theorem ceiling_sum_evaluation : evaluateCeilingSum = 8 := by
  sorry

end ceiling_sum_evaluation_l147_147685


namespace evaluate_ceiling_sum_l147_147642

theorem evaluate_ceiling_sum :
  (⌈Real.sqrt (16 / 9)⌉ : ℤ) + (⌈(16 / 9: ℝ)⌉ : ℤ) + (⌈(16 / 9: ℝ)^2⌉ : ℤ) = 8 := 
by
  -- Placeholder for proof
  sorry

end evaluate_ceiling_sum_l147_147642


namespace largest_multiple_of_15_less_than_500_l147_147041

theorem largest_multiple_of_15_less_than_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ (∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x) := sorry

end largest_multiple_of_15_less_than_500_l147_147041


namespace polynomial_coefficients_l147_147484

noncomputable def a : ℝ := 15
noncomputable def b : ℝ := -198
noncomputable def c : ℝ := 1

theorem polynomial_coefficients :
  (∀ x₁ x₂ x₃ : ℝ, 
    (x₁ + x₂ + x₃ = 0) ∧ 
    (x₁ * x₂ + x₂ * x₃ + x₃ * x₁ = -3) ∧ 
    (x₁ * x₂ * x₃ = -1) → 
    (a = 15) ∧ 
    (b = -198) ∧ 
    (c = 1)) := 
by sorry

end polynomial_coefficients_l147_147484


namespace percentage_reduction_approx_15_l147_147991

theorem percentage_reduction_approx_15 :
  let original_employees := 229.41
  let new_employees := 195
  let percentage_reduction := ((original_employees - new_employees) / original_employees) * 100
  percentage_reduction ≈ 15 :=
by sorry

end percentage_reduction_approx_15_l147_147991


namespace a_value_l147_147331

noncomputable def find_a (a : ℝ) : Prop :=
  let term_coeff := (2018 * (-a) + 1)
  in term_coeff = 2019

theorem a_value : find_a (-1) :=
by sorry

end a_value_l147_147331


namespace largest_multiple_of_15_less_than_500_l147_147093

theorem largest_multiple_of_15_less_than_500 : ∃ (x : ℕ), (x < 500) ∧ (x % 15 = 0) ∧ ∀ (y : ℕ), (y < 500) ∧ (y % 15 = 0) → y ≤ x :=
begin
  use 495,
  split,
  { norm_num },
  split,
  { norm_num },
  intros y hy1 hy2,
  sorry,
end

end largest_multiple_of_15_less_than_500_l147_147093


namespace arithmetic_geometric_seq_problem_l147_147329

def is_arithmetic_sequence (seq : List ℝ) : Prop :=
  ∀ (i j k : ℕ), i < j ∧ j < k → (seq.nth i).get_or_else 0 + (seq.nth k).get_or_else 0 = 2 * (seq.nth j).get_or_else 0

def is_geometric_sequence (seq : List ℝ) : Prop :=
  ∀ (i j k : ℕ), i < j ∧ j < k → (seq.nth i).get_or_else 0 * (seq.nth k).get_or_else 0 = (seq.nth j).get_or_else 0 ^ 2

theorem arithmetic_geometric_seq_problem
  (a_seq : List ℝ)
  (g_seq : List ℝ)
  (h1 : a_seq = [-7, a_seq.nth 1, a_seq.nth 2, -1])
  (h2 : g_seq = [-4, g_seq.nth 1, g_seq.nth 2, g_seq.nth 3, -1])
  (h_arith : is_arithmetic_sequence a_seq)
  (h_geom : is_geometric_sequence g_seq)
  (a1 a2 : ℝ)
  (h_a1 : a_seq.nth 1 = some a1)
  (h_a2 : a_seq.nth 2 = some a2)
  (b2 : ℝ)
  (h_b2 : g_seq.nth 2 = some b2) :
  (a2 - a1) / b2 = -1 := 
sorry

end arithmetic_geometric_seq_problem_l147_147329


namespace largest_multiple_of_15_less_than_500_l147_147040

theorem largest_multiple_of_15_less_than_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ (∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x) := sorry

end largest_multiple_of_15_less_than_500_l147_147040


namespace point_P_divides_ratios_l147_147856

variable (A B C M1 M2 N1 N2 P : Point)
variables (k1 k2 : ℝ)

axiom AM1_C : AM1 / M1C = k1
axiom CN1_B : CN1 / N1B = k1
axiom AM2_C : AM2 / M2C = k2
axiom CN2_B : CN2 / N2B = k2
axiom P_intersection : P ∈ Line(M1, N1) ∧ P ∈ Line(M2, N2)

theorem point_P_divides_ratios :
  P = intersection_point (Line(M1, N1)) (Line(M2, N2)) →
  M1P / PN1 = k2 ∧ M2P / PN2 = k1 :=
sorry

end point_P_divides_ratios_l147_147856


namespace mitch_earns_correctly_l147_147464

noncomputable def mitch_weekly_earnings : ℝ :=
  let earnings_mw := 3 * (3 * 5 : ℝ) -- Monday to Wednesday
  let earnings_tf := 2 * (6 * 4 : ℝ) -- Thursday and Friday
  let earnings_sat := 4 * 6         -- Saturday
  let earnings_sun := 5 * 8         -- Sunday
  let total_earnings := earnings_mw + earnings_tf + earnings_sat + earnings_sun
  let after_expenses := total_earnings - 25
  let after_tax := after_expenses - 0.10 * after_expenses
  after_tax

theorem mitch_earns_correctly : mitch_weekly_earnings = 118.80 := by
  sorry

end mitch_earns_correctly_l147_147464


namespace time_to_pass_l147_147596

noncomputable def speed_to_mps (v : ℝ) : ℝ := v * 1000 / 3600

theorem time_to_pass (v_passenger v_goods : ℝ) (L_goods : ℝ) :
  v_passenger = 60 → v_goods = 52 → L_goods = 280 →
  let v_relative := speed_to_mps (v_passenger + v_goods) in
  let time := L_goods / v_relative in
  time ≈ 9 := 
by sorry

end time_to_pass_l147_147596


namespace largest_multiple_of_15_below_500_l147_147036

theorem largest_multiple_of_15_below_500 : ∃ (k : ℕ), 15 * k < 500 ∧ ∀ (m : ℕ), 15 * m < 500 → 15 * m ≤ 15 * k := 
by
  existsi 33
  split
  · norm_num
  · intro m h
    have h1 : m ≤ 33 := Nat.le_of_lt_succ (Nat.div_lt_succ 500 15 m)
    exact (mul_le_mul_left (by norm_num : 0 < 15)).mpr h1
  sorry

end largest_multiple_of_15_below_500_l147_147036


namespace passengers_heads_l147_147198

theorem passengers_heads (X : ℕ) (H_cats : 7) (H_legs : 41) : 7 + X + 1 = 14 :=
by
  have cats_legs := 7 * 4
  have tot_legs := cats_legs + 2 * X + 1
  have legs_eq := tot_legs = 41
  have legs_simp := 28 + 2 * X + 1 = 41
  have eq_X := 2 * X + 29 = 41
  have simplify := 2 * X = 12
  have X_value := X = 6
  have tot_heads := 7 + X + 1
  have heads_value := 7 + 6 + 1 = 14
  exact heads_value

end passengers_heads_l147_147198


namespace geometric_sequence_value_of_a_l147_147420

theorem geometric_sequence_value_of_a (a : ℝ) (h : a > 0) :
  (∀ t : ℝ, (-2 + (real.sqrt 2 / 2) * t, -1 + (real.sqrt 2 / 2) * t) ∈
    {p : ℝ × ℝ | ∃ ρ θ : ℝ, p = (ρ * real.cos θ, ρ * real.sin θ) ∧ ρ * (real.cos θ)^2 = 4 * a * real.sin θ}) →
  (∃ (PM PN MN: ℝ), PM * PM = 4 * PN * MN ∧ PM * PM = PN * MN) →
  a = 1 / 4 :=
by
  sorry

end geometric_sequence_value_of_a_l147_147420


namespace largest_whole_number_n_l147_147146

theorem largest_whole_number_n : ∃ (n : ℕ), (frac (n / 7) + 1/3 < 1) ∧ ∀ (m : ℕ), (frac (m / 7) + 1/3 < 1) → m ≤ n :=
begin
  use 4,
  split,
  { norm_num },
  { intros m h,
    norm_num at h,
    sorry
  }
end

end largest_whole_number_n_l147_147146


namespace parabola_directrix_l147_147880

theorem parabola_directrix (p : ℝ) :
  (y y : ℝ, y^2 = 6 * p → x = -3 / 2) :=
  by sorry

end parabola_directrix_l147_147880


namespace triangle_even_side_possibilities_l147_147349

theorem triangle_even_side_possibilities {x : ℕ} (hx1 : x % 2 = 0) (hx2 : 5 < x) (hx3 : x < 11) : 
  {x : ℕ | x % 2 = 0 ∧ 5 < x ∧ x < 11}.card = 3 :=
by
  sorry

end triangle_even_side_possibilities_l147_147349


namespace max_distance_proof_l147_147294

-- Given conditions
constant front_tire_lifespan : ℕ := 42000
constant rear_tire_lifespan : ℕ := 56000
constant maximum_distance_traveled : ℕ := 42000

theorem max_distance_proof : maximum_distance_traveled = 42000 := 
begin
  sorry,
end

end max_distance_proof_l147_147294


namespace projective_transformation_l147_147727

variables (l : Line) (circle : Circle) (M : Point) 
           (R : PlaneMotion) [preserves_circle R circle]

def P_M (M : Point) [on_circle M circle] : Projection l circle := sorry
def P_{N} (N : Point) [on_circle N circle] : Projection l circle := sorry

theorem projective_transformation (M : Point) [on_circle M circle] 
  (hM : ¬ lies_on_line M l)
  (R : PlaneMotion) [preserves_circle R circle] :
  is_projective_transformation (P_M M)⁻¹ ∘ R ∘ P_M M := 
sorry

end projective_transformation_l147_147727


namespace original_number_l147_147980

theorem original_number (n : ℕ) : 
  (∃ n, let dsum := (n.digits 10).sum in 2010 ≤ n ∧ n ≤ 2019 ∧ n - dsum = 2007) := 
sorry

end original_number_l147_147980


namespace original_fraction_l147_147381

variable (numerator denominator : ℚ)

def new_numerator := 1.15 * numerator
def new_denominator := 0.92 * denominator
def resulting_fraction := (new_numerator / new_denominator : ℚ)

theorem original_fraction :
  resulting_fraction = 15 / 16 →
  numerator / denominator = 4 / 3 :=
by
  sorry

end original_fraction_l147_147381


namespace matrix_determinant_l147_147693

def matrix_det {α : Type*} [CommRing α] (M : Matrix (Fin 3) (Fin 3) α) : α :=
  M.det

theorem matrix_determinant (θ φ : ℝ) : 
  matrix_det ![
                 ![ sin θ * sin φ, sin θ * cos φ, cos θ],
                 ![            cos φ,        -sin φ,      0],
                 ![-cos θ * sin φ, -cos θ * cos φ, sin θ]
               ] = -1 :=
by sorry

end matrix_determinant_l147_147693


namespace remaining_kibble_l147_147456

def starting_kibble : ℕ := 12
def mary_kibble_morning : ℕ := 1
def mary_kibble_evening : ℕ := 1
def frank_kibble_afternoon : ℕ := 1
def frank_kibble_late_evening : ℕ := 2 * frank_kibble_afternoon

theorem remaining_kibble : starting_kibble - (mary_kibble_morning + mary_kibble_evening + frank_kibble_afternoon + frank_kibble_late_evening) = 7 := by
  sorry

end remaining_kibble_l147_147456


namespace product_zero_probability_l147_147543

noncomputable def probability_product_zero : ℚ :=
  let set := {-3, -2, -1,  0,  0, 6}  -- Note: -0 is automatically equivalent to 0 in Lean definition
  let total_combinations := Nat.choose 6 2
  let favorable_combinations := 5 + 4 -- 5 combinations for 0 and 4 for -0 considering equivalency
  favorable_combinations / total_combinations

theorem product_zero_probability : 
  probability_product_zero = 3 / 5 :=
begin
  sorry
end

end product_zero_probability_l147_147543


namespace largest_multiple_of_15_less_than_500_l147_147067

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), (n < 500) ∧ (15 ∣ n) ∧ (∀ m : ℕ, (m < 500) ∧ (15 ∣ m) → m ≤ n) ∧ n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l147_147067


namespace batsman_total_score_l147_147965

def runs_from_boundaries : ℕ := 6 * 4
def runs_from_sixes : ℕ := 4 * 6
def runs_from_boundaries_and_sixes : ℕ := runs_from_boundaries + runs_from_sixes

theorem batsman_total_score (T : ℕ) 
  (h1 : runs_from_boundaries = 24)
  (h2 : runs_from_sixes = 24)
  (h3 : runs_from_boundaries_and_sixes = 48)
  (h4 : 0.60 * T = 0.6 * T)
  (h5 : runs_from_boundaries_and_sixes + 0.6 * T = T) :
  T = 120 :=
by
  sorry

end batsman_total_score_l147_147965


namespace special_four_digit_numbers_l147_147193

noncomputable def count_special_four_digit_numbers : Nat :=
  -- The task is to define the number of four-digit numbers formed using the digits {0, 1, 2, 3, 4}
  -- that contain the digit 0 and have exactly two digits repeating
  144

theorem special_four_digit_numbers : count_special_four_digit_numbers = 144 := by
  sorry

end special_four_digit_numbers_l147_147193


namespace translation_correct_l147_147777

-- Define the first line l1
def l1 (x : ℝ) : ℝ := 2 * x - 2

-- Define the second line l2
def l2 (x : ℝ) : ℝ := 2 * x

-- State the theorem
theorem translation_correct :
  ∀ x : ℝ, l2 x = l1 x + 2 :=
by
  intro x
  unfold l1 l2
  sorry

end translation_correct_l147_147777


namespace wheel_distance_covered_l147_147608

theorem wheel_distance_covered :
  let π : ℝ := 3.14159
  let d : ℝ := 14
  let n : ℝ := 19.017288444040037
  let circumference : ℝ := π * d
  let distance : ℝ := circumference * n
  distance = 836.103 :=
by
  trivial -- placeholder for proof
  sorry

end wheel_distance_covered_l147_147608


namespace necessarily_negative_l147_147866

theorem necessarily_negative (a b c : ℝ) (h1 : 0 < a ∧ a < 2) (h2 : -2 < b ∧ b < 0) (h3 : 0 < c ∧ c < 1) : b + c < 0 :=
sorry

end necessarily_negative_l147_147866


namespace sum_of_roots_l147_147765

theorem sum_of_roots (x : ℝ) (h : (x + 3) * (x - 2) = 15) : x = -1 :=
sorry

end sum_of_roots_l147_147765


namespace ceiling_sum_evaluation_l147_147689

noncomputable def evaluateCeilingSum : ℝ := 
  ⌈Real.sqrt (16 / 9)⌉ + ⌈(16 / 9)⌉ + ⌈((16 / 9) ^ 2)⌉ 

theorem ceiling_sum_evaluation : evaluateCeilingSum = 8 := by
  sorry

end ceiling_sum_evaluation_l147_147689


namespace wayne_fathers_gift_l147_147546

theorem wayne_fathers_gift (initial_blocks : ℕ) (final_blocks : ℕ) (blocks_given : ℕ) :
  initial_blocks = 9 → final_blocks = 15 → final_blocks = initial_blocks + blocks_given → blocks_given = 6 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  linarith
sorry

end wayne_fathers_gift_l147_147546


namespace sqrt_expr_to_rational_form_l147_147392

theorem sqrt_expr_to_rational_form :
  ∃ (a b c : ℕ), 0 < c ∧ (∑ i in [28, 27, 12], i) = 28 + 27 + 12 ∧
  (sqrt 6 + 1 / sqrt 6 + sqrt 8 + 1 / sqrt 8 = (a * sqrt 6 + b * sqrt 8) / c) ∧
  a + b + c = 67 := 
by
  use 28, 27, 12
  split
  { exact nat.succ_pos' _ }
  split
  { norm_num }
  split
  { -- omitted proof
    sorry }
  { norm_num }

end sqrt_expr_to_rational_form_l147_147392


namespace find_d_l147_147830

theorem find_d (c d : ℝ) (f g : ℝ → ℝ)
  (hf : ∀ x, f x = 5 * x + c)
  (hg : ∀ x, g x = c * x + 3)
  (hfg : ∀ x, f (g x) = 15 * x + d) :
  d = 18 :=
sorry

end find_d_l147_147830


namespace minimum_value_of_expression_l147_147816

variable (a b c : ℝ)

noncomputable def expression (a b c : ℝ) := (a + b) / c + (a + c) / b + (b + c) / a

theorem minimum_value_of_expression (hp1 : 0 < a) (hp2 : 0 < b) (hp3 : 0 < c) (h1 : a = 2 * b) (h2 : a = 2 * c) :
  expression a b c = 9.25 := 
sorry

end minimum_value_of_expression_l147_147816


namespace find_phi_l147_147773

noncomputable def function_even (f : ℝ → ℝ) : Prop :=
∀ x, f(x) = f(-x)

def problem_condition (x : ℝ) : ℝ :=
sin (2 * x) + sqrt 3 * cos (2 * x)

theorem find_phi : 
  function_even (λ x, problem_condition (x + π / 12)) :=
by
  sorry

end find_phi_l147_147773


namespace largest_piece_perimeter_near_hundredth_l147_147434

def isosceles_triangle (b h : ℝ) : Prop := 
  b = 10 ∧ h = 12

def equal_area_division (n : ℕ) : Prop := 
  n = 10

noncomputable def perimeter_of_largest_piece (b h : ℝ) (n : ℕ) : ℝ :=
  let P (k : ℕ) : ℝ := 1 + (Real.sqrt (h^2 + k^2)) + (Real.sqrt (h^2 + (k + 1)^2)) in
  P (n - 1)

theorem largest_piece_perimeter_near_hundredth (b h : ℝ) (n : ℕ) 
  (Htriangle : isosceles_triangle b h) (Hdivision : equal_area_division n) :
  Real.floor ((perimeter_of_largest_piece b h n) * 100 + 0.5) / 100 = 31.62 :=
by 
  unfold isosceles_triangle at Htriangle
  cases Htriangle with Hb Hh
  have Hbk : b = 10 := Hb
  have Hhk : h = 12 := Hh
  unfold equal_area_division at Hdivision
  have Hnk : n = 10 := Hdivision
  have Plargest : perimeter_of_largest_piece b h n = (1 + Real.sqrt 225 + Real.sqrt 244) :=
    by {
      rw [Hbk, Hhk, Hnk],
      norm_num,
    }
  sorry

end largest_piece_perimeter_near_hundredth_l147_147434


namespace find_a_values_l147_147700

noncomputable def unique_solution_param (a : ℝ) : Prop :=
  ∀ x : ℝ, a * |x + 2| + (x^2 + x - 12) / (x + 4) = 0

theorem find_a_values :
  {a : ℝ | unique_solution_param a} = {a : ℝ | a ∈ set.Icc (-1 : ℝ) 1 ∪ {7 / 2}} :=
sorry

end find_a_values_l147_147700


namespace smallest_b_distance_l147_147201

theorem smallest_b_distance {rect : set (ℝ × ℝ)} (h₁ : ∀ r ∈ rect, r.1 < 2 ∧ r.2 < 1)
  (points : fin 7 → ℝ × ℝ) (h₂ : ∀ p, points p ∈ rect) :
  ∃ (p1 p2 : fin 7), p1 ≠ p2 ∧ dist (points p1) (points p2) ≤ √10 / 3 := 
sorry

end smallest_b_distance_l147_147201


namespace area_quadrilateral_eq_sum_of_triangles_l147_147424

variables {P : Type*} [EuclideanGeometry P] {A B C D E F : P}
open EuclideanGeometry

-- We are given that E and F are midpoints and ABCD is convex
def midpoint (A B M : P) : Prop := dist A M = dist B M ∧ ∃ (I : P), I ∈ line_through A B ∧ segment A I = segment I B ∧ seg_eq I M M B

-- Conditions in Lean:
variable (AB_midpoint : midpoint A B E)
variable (CD_midpoint : midpoint C D F)
variable (convex_ABCD : convex_hull_insert _ _ _ _ ∉ segment A C)

-- Now, we aim to prove the desired statement:
theorem area_quadrilateral_eq_sum_of_triangles :
  area_2d_quadrilateral A B C D = area_2d_triangle A B F + area_2d_triangle C D E :=
sorry

end area_quadrilateral_eq_sum_of_triangles_l147_147424


namespace three_digit_number_divisible_by_nine_l147_147153

/-- Proving that a three-digit number with a units digit of 2 and a 
    hundreds digit of 4, which is divisible by 9, is equal to 432. -/
theorem three_digit_number_divisible_by_nine : 
  ∃ (n : ℕ), n < 1000 ∧ 
             (n % 10 = 2) ∧ 
             (n / 100 = 4) ∧ 
             (∑ d in (n.digits 10), d) % 9 = 0 ∧ 
             n = 432 :=
sorry

end three_digit_number_divisible_by_nine_l147_147153


namespace range_of_m_l147_147755

theorem range_of_m (m : ℝ) :
  (∀ θ : ℝ, m^2 + (cos θ^2 - 5) * m + 4 * sin θ^2 ≥ 0) ↔ (m ≤ 0 ∨ m ≥ 4) :=
by
  sorry

end range_of_m_l147_147755


namespace base_prime_representation_of_360_l147_147851

theorem base_prime_representation_of_360 :
  let n := 360
  ∃ a b c d : ℕ, (2^a * 3^b * 5^c * 7^d = n) ∧
                 (a = 3) ∧ (b = 2) ∧ (c = 1) ∧ (d = 0) ∧
                 (1000 * a + 100 * b + 10 * c + d = 3201) :=
by
  let n := 360
  use 3, 2, 1, 0
  have h1 : 2^3 * 3^2 * 5^1 * 7^0 = n := by norm_num
  have h2 : 3 = 3 := rfl
  have h3 : 2 = 2 := rfl
  have h4 : 1 = 1 := rfl
  have h5 : 0 = 0 := rfl
  have h6 : 1000 * 3 + 100 * 2 + 10 * 1 + 0 = 3201 := by norm_num
  exact ⟨h1, h2, h3, h4, h5, h6⟩

end base_prime_representation_of_360_l147_147851


namespace largest_multiple_of_15_less_than_500_l147_147005

theorem largest_multiple_of_15_less_than_500 : ∃ n : ℕ, (n > 0 ∧ 15 * n < 500) ∧ (∀ m : ℕ, m > n → 15 * m >= 500) ∧ 15 * n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l147_147005


namespace find_interest_rate_l147_147235

noncomputable def annual_interest_rate (P A : ℝ) (n : ℕ) (t r : ℝ) : Prop :=
  A = P * (1 + r / n)^(n * t)

theorem find_interest_rate :
  annual_interest_rate 5000 5100.50 4 0.5 0.04 :=
by
  sorry

end find_interest_rate_l147_147235


namespace eval_expression_l147_147694

theorem eval_expression : (1 / 8 : ℝ) ^ (- 1 / 3) = 2 := 
by sorry

end eval_expression_l147_147694


namespace num_lineups_l147_147467

-- Define the given conditions
def num_players : ℕ := 12
def num_lineman : ℕ := 4
def num_qb_among_lineman : ℕ := 2
def num_running_backs : ℕ := 3

-- State the problem and the result as a theorem
theorem num_lineups : 
  (num_lineman * (num_qb_among_lineman) * (num_running_backs) * (num_players - num_lineman - num_qb_among_lineman - num_running_backs + 3) = 216) := 
by
  -- The proof will go here
  sorry

end num_lineups_l147_147467


namespace cards_net_cost_equivalence_l147_147399

-- Define the purchase amount
def purchase_amount : ℝ := 10000

-- Define cashback percentages
def debit_card_cashback : ℝ := 0.01
def credit_card_cashback : ℝ := 0.005

-- Define interest rate for keeping money in the debit account
def interest_rate : ℝ := 0.005

-- A function to calculate the net cost after 1 month using the debit card
def net_cost_debit_card (purchase_amount : ℝ) (cashback_percentage : ℝ) : ℝ :=
  purchase_amount - purchase_amount * cashback_percentage

-- A function to calculate the net cost after 1 month using the credit card
def net_cost_credit_card (purchase_amount : ℝ) (cashback_percentage : ℝ) (interest_rate : ℝ) : ℝ :=
  purchase_amount - purchase_amount * cashback_percentage - purchase_amount * interest_rate

-- Final theorem stating that the net cost using both cards is the same
theorem cards_net_cost_equivalence : 
  net_cost_debit_card purchase_amount debit_card_cashback = 
  net_cost_credit_card purchase_amount credit_card_cashback interest_rate :=
by
  sorry

end cards_net_cost_equivalence_l147_147399


namespace largest_multiple_of_15_less_than_500_l147_147026

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), n < 500 ∧ 15 ∣ n ∧ ∀ (m : ℕ), m < 500 ∧ 15 ∣ m -> m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l147_147026


namespace problem_solution_l147_147739

noncomputable def f : ℕ+ → ℝ := sorry

theorem problem_solution (f_prop : ∀ a b : ℕ+, f (a + b) = f a * f b) (f_init : f 1 = 2) :
  (∑ k in (finset.range 1009).map (λ n, 2 * n + 1), f (k + 1) / f k) = 2018 := 
  sorry

end problem_solution_l147_147739


namespace sum_of_values_l147_147508

def f (x : Int) : Int := Int.natAbs x - 3
def g (x : Int) : Int := -x

def fogof (x : Int) : Int := f (g (f x))

theorem sum_of_values :
  (fogof (-5)) + (fogof (-4)) + (fogof (-3)) + (fogof (-2)) + (fogof (-1)) + (fogof 0) + (fogof 1) + (fogof 2) + (fogof 3) + (fogof 4) + (fogof 5) = -17 :=
by
  sorry

end sum_of_values_l147_147508


namespace find_rate_percent_l147_147925

-- Definitions of conditions
variables (SI P T : ℝ)
-- Setting the known values:
def si_value := (SI = 500)
def principal_value := (P = 2000)
def time_value := (T = 2)

-- Definition of the formula for Simple Interest
def simple_interest (P R T : ℝ) : ℝ := P * R * T / 100

-- The proof statement
theorem find_rate_percent (h1 : si_value) (h2 : principal_value) (h3 : time_value) : 
  ∃ R : ℝ, simple_interest P R T = SI ∧ R = 12.5 :=
sorry

end find_rate_percent_l147_147925


namespace volume_of_cylinder_correct_l147_147278

noncomputable def volume_of_cylinder (side_length : ℝ) : ℝ :=
  let r := side_length / 2 in
  let h := side_length in
  π * r * r * h

theorem volume_of_cylinder_correct : volume_of_cylinder 16 = 1024 * π :=
by
  sorry

end volume_of_cylinder_correct_l147_147278


namespace kibble_remaining_l147_147458

theorem kibble_remaining 
  (initial_amount : ℕ) (morning_mary : ℕ) (evening_mary : ℕ) 
  (afternoon_frank : ℕ) (evening_frank : ℕ) :
  initial_amount = 12 →
  morning_mary = 1 →
  evening_mary = 1 →
  afternoon_frank = 1 →
  evening_frank = 2 * afternoon_frank →
  initial_amount - (morning_mary + evening_mary + afternoon_frank + evening_frank) = 7 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  done

end kibble_remaining_l147_147458


namespace sum_of_21_terms_is_231_l147_147728

-- Definition of the sequence and conditions
def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n > 0 → a n + a (n + 1) = 2 * n + 1

-- Definition of the sum of the first n terms
def seq_sum (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  if n = 0 then 0 else ∑ k in Finset.range n, a (k + 1)

-- The theorem we need to prove
theorem sum_of_21_terms_is_231 (a : ℕ → ℕ) (h : sequence a) : seq_sum a 21 = 231 :=
by
  sorry

end sum_of_21_terms_is_231_l147_147728


namespace sum_of_b_is_real_l147_147842

theorem sum_of_b_is_real (n : ℕ) (p g : ℂ[X])
  (hp : p = ∑ i in finset.range (n + 1), (a i) * X ^ (n - i))
  (hg : g = ∑ i in finset.range (n + 1), (b i) * X ^ (n - i))
  (roots_g_are_squares : ∀ x, p.is_root x → g.is_root (x^2))
  (sum_a1_a3_real : real (∑ i in finset.range (n) | even i, a i))
  (sum_a2_a4_real : real (∑ i in finset.range (n) | odd i, a i)):
  real (∑ i in finset.range (n + 1), b i) := sorry

end sum_of_b_is_real_l147_147842


namespace product_of_roots_l147_147440

theorem product_of_roots : 
  (∀ (a b c : ℂ), (root (3 * X^3 - 9 * X^2 + 5 * X - 10) a ∧
                   root (3 * X^3 - 9 * X^2 + 5 * X - 10) b ∧
                   root (3 * X^3 - 9 * X^2 + 5 * X - 10) c) →
                  (a * b * c = (10 / 3 : ℂ))) :=
begin
  sorry
end

end product_of_roots_l147_147440


namespace gcd_102_238_l147_147509

-- Define the two numbers involved
def a : ℕ := 102
def b : ℕ := 238

-- State the theorem
theorem gcd_102_238 : Int.gcd a b = 34 :=
by
  sorry

end gcd_102_238_l147_147509


namespace number_of_zeros_of_f_monotonicity_of_h_range_of_a_l147_147845

noncomputable def f (x : ℝ) : ℝ := Real.log x - Real.exp (1 - x)
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := a * (x^2 - 1) - 1 / x
noncomputable def h (x : ℝ) (a : ℝ) : ℝ := g x a - f x + (Real.exp x - x * Real.exp x) / (x * Real.exp x)

theorem number_of_zeros_of_f :
  ∃! x ∈ Ioo 1 Real.e, f x = 0 :=
sorry

theorem monotonicity_of_h (a : ℝ):
  (a ≤ 0 → ∀ x > 0, h x a ≤ h (x + 1) a) ∧ (a > 0 → 
    ∀ x ∈ Ioo 0 (1 / Real.sqrt (2*a)), h x a ≤ h (x + 1) a ∧ 
    ∀ x ∈ Ioo (1 / Real.sqrt (2 * a)) Real.Infty, 
    h (x + 1) a ≤ h x a) :=
sorry

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Ioo 1 Real.Infty, f x < g x a) ↔ a ∈ Icc (1 / 2) Real.Infty :=
sorry

end number_of_zeros_of_f_monotonicity_of_h_range_of_a_l147_147845


namespace sn_expression_tn_sum_l147_147427

section problem

def sequence_a (a : ℕ → ℝ) : Prop := 
  a 1 = 1 ∧ ∀ n ≥ 2, (∑ i in finset.range n, a (i+1))^2 = a n * ((∑ i in finset.range n, a (i+1)) - 1 / 2)

theorem sn_expression (a : ℕ → ℝ) (h : sequence_a a) (n : ℕ) (hn : n ≥ 1) : 
  (∑ i in finset.range n, a (i+1)) = 1 / (2 * n - 1) :=
sorry

def b_n (a : ℕ → ℝ) (n : ℕ) : ℝ := (∑ i in finset.range n.succ, a (i+1)) / (2 * n + 1)

theorem tn_sum (a : ℕ → ℝ) (h : sequence_a a) (n : ℕ) (hn : n ≥ 1) :
  (∑ i in finset.range n, b_n a (i + 1)) = n / (2 * n + 1) :=
sorry

end problem

end sn_expression_tn_sum_l147_147427


namespace largest_multiple_of_15_less_than_500_l147_147094

theorem largest_multiple_of_15_less_than_500 : ∃ (x : ℕ), (x < 500) ∧ (x % 15 = 0) ∧ ∀ (y : ℕ), (y < 500) ∧ (y % 15 = 0) → y ≤ x :=
begin
  use 495,
  split,
  { norm_num },
  split,
  { norm_num },
  intros y hy1 hy2,
  sorry,
end

end largest_multiple_of_15_less_than_500_l147_147094


namespace integral_of_x_squared_l147_147260

noncomputable def f : ℝ → ℝ := λ x, x^2

theorem integral_of_x_squared :
  ∫ x in -1..1, f x = 2 / 3 :=
by sorry

end integral_of_x_squared_l147_147260


namespace min_price_per_car_to_avoid_losses_min_order_volume_to_compete_l147_147161

section TeslaModelS

variables (purchase_price : ℝ) (customs_duties : ℝ)
          (monthly_rent : ℝ) (monthly_salary : ℝ)
          (other_expenses : ℝ) (num_cars : ℝ)
          (competitor_price : ℝ) (charging_station_price : ℝ)

def total_custom_expenses (num_cars : ℝ) (purchase_price : ℝ) (customs_duties : ℝ) : ℝ :=
  num_cars * (purchase_price + customs_duties)

def total_fixed_expenses (monthly_rent : ℝ) (monthly_salary : ℝ) (other_expenses : ℝ) : ℝ :=
  monthly_rent + monthly_salary + other_expenses

def total_expenses (num_cars : ℝ) (purchase_price : ℝ) (customs_duties : ℝ) 
                   (monthly_rent : ℝ) (monthly_salary : ℝ) (other_expenses : ℝ) : ℝ :=
  total_custom_expenses num_cars purchase_price customs_duties + 
    total_fixed_expenses monthly_rent monthly_salary other_expenses

def average_cost_per_car (num_cars : ℝ) (purchase_price : ℝ) (customs_duties : ℝ)
                          (monthly_rent : ℝ) (monthly_salary : ℝ) (other_expenses : ℝ) : ℝ :=
  total_expenses num_cars purchase_price customs_duties monthly_rent monthly_salary other_expenses / num_cars

theorem min_price_per_car_to_avoid_losses : 
  average_cost_per_car 30 2.5 2 0.05 0.37 0.18 = 4.52 := 
  sorry

theorem min_order_volume_to_compete : 
  ∃ (x : ℝ), x ≥ 2 ∧ (charging_station_price = 0.4 ∧ competitor_price = 5.3 ∧
  (average_cost_per_car x 2.5 2 0.05 0.37 0.18 ≤ competitor_price - charging_station_price)) :=
  sorry

end TeslaModelS

end min_price_per_car_to_avoid_losses_min_order_volume_to_compete_l147_147161


namespace least_number_subtracted_l147_147933

theorem least_number_subtracted (n : ℕ) (k : ℕ) (m : ℕ) (h : k = 384729) (h₁ : m = 331) (h₂ : ∃ r : ℕ, h % h₁ = r -> n = r) : n = 135 :=
by
  sorry

end least_number_subtracted_l147_147933


namespace max_sum_max_product_l147_147847

noncomputable def problem1 (a : ℕ → ℝ) : Prop :=
  (∀ i, ∑ i in finset.range 40, a i = 0) ∧ 
  (∀ i, |a i - a ((i + 1) % 40)| ≤ 1)

theorem max_sum (a : ℕ → ℝ) (h : problem1 a) : 
  a 10 + a 20 + a 30 + a 40 ≤ 10 :=
sorry

theorem max_product (a : ℕ → ℝ) (h : problem1 a) : 
  a 10 * a 20 + a 30 * a 40 ≤ 425 / 8 :=
sorry

end max_sum_max_product_l147_147847


namespace coordinates_of_B_l147_147423

open Real

def transform (ratio : ℝ) (P : ℝ × ℝ) : ℝ × ℝ :=
  (ratio * P.1, ratio * P.2)

def point_A : ℝ × ℝ := (-4, 2)
def point_B : ℝ × ℝ := (-6, -4)
def similarity_ratio : ℝ := 1 / 2
def point_B' : ℝ × ℝ := transform similarity_ratio point_B

theorem coordinates_of_B' : point_B' = (-3, -2) :=
by 
  sorry

end coordinates_of_B_l147_147423


namespace find_d_l147_147824

-- Definitions of the functions f and g and condition on f(g(x))
def f (x : ℝ) (c : ℝ) : ℝ := 5 * x + c
def g (x : ℝ) (c : ℝ) : ℝ := c * x + 3

theorem find_d (c d x : ℝ) (h : f (g x c) c = 15 * x + d) : d = 18 :=
sorry

end find_d_l147_147824


namespace ben_subtracts_79_from_40_sq_to_get_39_sq_l147_147536

theorem ben_subtracts_79_from_40_sq_to_get_39_sq :
  ∀ (n : ℤ), n = 40 → 39^2 = 40^2 - 79 :=
by
  intros n hn
  rw hn
  sorry

end ben_subtracts_79_from_40_sq_to_get_39_sq_l147_147536


namespace cars_travel_same_distance_l147_147243

-- Definition of the problem conditions
def carX_speed : ℝ := 35
def carY_speed : ℝ := 46
def carZ_speed : ℝ := 65
def carY_start_time : ℝ := 1.2
def carZ_start_time : ℝ := 3.6

-- The distance each car travels (to be proven)
theorem cars_travel_same_distance :
  ∀ (t : ℝ), 35 * (t + 3.6) = 46 * (t + 1.2) → 35 * (t + 3.6) = 65 * t → 
  35 * (t + 3.6) = 273 := by {
  intros t h1 h2,
  sorry
}

end cars_travel_same_distance_l147_147243


namespace expression_simplification_l147_147505

theorem expression_simplification : (sqrt 100 + sqrt 9) * (sqrt 100 - sqrt 9) = 91 :=
by 
  have h1 : sqrt 100 = 10 := sorry
  have h2 : sqrt 9 = 3 := sorry
  -- Simplifying using the values proved above
  calc
    (sqrt 100 + sqrt 9) * (sqrt 100 - sqrt 9)
        = (10 + 3) * (10 - 3) : by rw [h1, h2]
    ... = 13 * 7            : by norm_num
    ... = 91                : by norm_num

end expression_simplification_l147_147505


namespace largest_multiple_of_15_less_than_500_is_495_l147_147119

-- Define the necessary conditions
def is_multiple_of_15 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 15 * k

def is_less_than_500 (n : ℕ) : Prop :=
  n < 500

-- Problem statement: Prove that the largest positive multiple of 15 less than 500 is 495
theorem largest_multiple_of_15_less_than_500_is_495 : 
  ∀ n : ℕ, is_multiple_of_15 n → is_less_than_500 n → n ≤ 495 := 
by
  sorry

end largest_multiple_of_15_less_than_500_is_495_l147_147119


namespace obtuse_triangle_two_acute_angles_l147_147363

-- Define the angle type (could be Real between 0 and 180 in degrees).
def is_acute (θ : ℝ) : Prop := 0 < θ ∧ θ < 90

def is_obtuse (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

-- Define an obtuse triangle using three angles α, β, γ
structure obtuse_triangle :=
(angle1 angle2 angle3 : ℝ)
(sum_angles_eq : angle1 + angle2 + angle3 = 180)
(obtuse_condition : is_obtuse angle1 ∨ is_obtuse angle2 ∨ is_obtuse angle3)

-- The theorem to prove the number of acute angles in an obtuse triangle is 2.
theorem obtuse_triangle_two_acute_angles (T : obtuse_triangle) : 
  (is_acute T.angle1 ∧ is_acute T.angle2 ∧ ¬ is_acute T.angle3) ∨ 
  (is_acute T.angle1 ∧ ¬ is_acute T.angle2 ∧ is_acute T.angle3) ∨ 
  (¬ is_acute T.angle1 ∧ is_acute T.angle2 ∧ is_acute T.angle3) :=
by sorry

end obtuse_triangle_two_acute_angles_l147_147363


namespace largest_multiple_of_15_less_than_500_l147_147068

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), (n < 500) ∧ (15 ∣ n) ∧ (∀ m : ℕ, (m < 500) ∧ (15 ∣ m) → m ≤ n) ∧ n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l147_147068


namespace sqrt_area_inequality_l147_147175

theorem sqrt_area_inequality
  (A B C D O K L M N : Point)
  (S S1 S2 : ℝ)
  (hO_inside: inside_quadrilateral O A B C D)
  (hK: on_segment K A B)
  (hL: on_segment L B C)
  (hM: on_segment M C D)
  (hN: on_segment N D A)
  (hOKBL_parallelogram: parallelogram O K B L)
  (hOMDN_parallelogram: parallelogram O M D N)
  (hS: area_quadrilateral A B C D = S)
  (hS1: area_quadrilateral O N A K = S1)
  (hS2: area_quadrilateral O L C M = S2):
  sqrt S ≥ sqrt S1 + sqrt S2 :=
  sorry

end sqrt_area_inequality_l147_147175


namespace common_point_of_geometric_prog_lines_l147_147998

theorem common_point_of_geometric_prog_lines (a b c : ℝ)
  (h1 : ∃ r : ℝ, b = a * r ∧ c = a * r^2) :
  ∃ p : ℝ × ℝ, p = (0, 0) ∧ (a ≠ 0) → ∀ (x y : ℝ), a * x + b * y = c → (x, y) = (0, 0) :=
by
  let p := (0, 0)
  use p
  split
  · exact rfl
  · intros ha x y h2
    have h3 : b = a * (c / b),
    { sorry },
    -- Detailed steps of the proof would come here.
    sorry

end common_point_of_geometric_prog_lines_l147_147998


namespace evaluate_expression_l147_147658

theorem evaluate_expression : 
  (⌈Real.sqrt (16 / 9)⌉ + ⌈ (16 / 9 : ℝ ) ⌉ + ⌈Real.pow (16 / 9 : ℝ ) 2⌉) = 8 := 
by 
  sorry

end evaluate_expression_l147_147658


namespace largest_multiple_of_15_less_than_500_l147_147071

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), (n < 500) ∧ (15 ∣ n) ∧ (∀ m : ℕ, (m < 500) ∧ (15 ∣ m) → m ≤ n) ∧ n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l147_147071


namespace book_distribution_l147_147559

theorem book_distribution (x : ℤ) (h : 9 * x + 7 < 11 * x) : 
  ∀ (b : ℤ), (b = 9 * x + 7) → b mod 9 = 7 :=
by
  intro b
  intro hb
  have : b = 9 * x + 7, from hb
  rw [←this]
  sorry

end book_distribution_l147_147559


namespace find_triangle_area_l147_147778

noncomputable def triangle_area (a b c: ℝ) : ℝ :=
  real.sqrt ((1 / 4) * (c^2 * a^2 - ( (c^2 + a^2 - b^2) / 2 )^2))

theorem find_triangle_area :
  let a := (real.sqrt 2 - 1),
      b := (real.sqrt 5),
      c := (real.sqrt 2 + 1)
  in
  let S := triangle_area a b c
  in S = (real.sqrt 3 / 4) :=
by
  let a := (real.sqrt 2 - 1)
  let b := (real.sqrt 5)
  let c := (real.sqrt 2 + 1)
  let S := triangle_area a b c
  have h_area : S = real.sqrt 3 / 4 := sorry
  exact h_area

end find_triangle_area_l147_147778


namespace prop1_converse_prop1_inverse_prop1_contrapositive_prop2_converse_prop2_inverse_prop2_contrapositive_l147_147565

-- Proposition 1
theorem prop1_converse (a b c : ℝ) (h : c < 0) (h' : a < b) : ac > bc := sorry
theorem prop1_inverse (a b c : ℝ) (h : c < 0) (h' : ac ≤ bc) : a ≥ b := sorry
theorem prop1_contrapositive (a b c : ℝ) (h : c < 0) (h' : a ≥ b) : ac ≤ bc := sorry

-- Proposition 2
theorem prop2_converse (a b : ℝ) (h : a = 0 ∨ b = 0) : ab = 0 := sorry
theorem prop2_inverse (a b : ℝ) (h : ab ≠ 0) : a ≠ 0 ∧ b ≠ 0 := sorry
theorem prop2_contrapositive (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) : ab ≠ 0 := sorry

end prop1_converse_prop1_inverse_prop1_contrapositive_prop2_converse_prop2_inverse_prop2_contrapositive_l147_147565


namespace complement_U_A_l147_147356

open Set

def U : Set ℤ := {-2, -1, 0, 1, 2}
def A : Set ℤ := {x | ∃ n ∈ ℤ, x = 2 / (n - 1) ∧ x ∈ U}

theorem complement_U_A : (U \ A) = {0} :=
by sorry

end complement_U_A_l147_147356


namespace find_m_l147_147247

def f (x m : ℝ) : ℝ := x ^ 2 - 3 * x + m
def g (x m : ℝ) : ℝ := 2 * x ^ 2 - 6 * x + 5 * m

theorem find_m (m : ℝ) (h : 3 * f 3 m = 2 * g 3 m) : m = 0 :=
by sorry

end find_m_l147_147247


namespace min_fence_length_l147_147252

theorem min_fence_length (x : ℝ) (h : x > 0) (A : x * (64 / x) = 64) : 2 * (x + 64 / x) ≥ 32 :=
by
  have t := (2 * (x + 64 / x)) 
  sorry -- Proof omitted, only statement provided as per instructions

end min_fence_length_l147_147252


namespace largest_multiple_of_15_less_than_500_l147_147057

theorem largest_multiple_of_15_less_than_500 : 
  ∃ n : ℕ, n * 15 < 500 ∧ ∀ m : ℕ, m * 15 < 500 → m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l147_147057


namespace tangent_fixed_point_l147_147720

open Classical

noncomputable def circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

noncomputable def tangent_point (A B O : ℝ × ℝ) :=
∃ P : ℝ × ℝ, (P ∈ line {p : ℝ × ℝ | p.1 + p.2 - 2 = 0}) ∧
  dist P A = dist P B ∧
  dist P O = 1

theorem tangent_fixed_point (A B : ℝ × ℝ) (hA : A ∈ circle) (hB : B ∈ circle) :
  (∀ P, (P ∈ line {p : ℝ × ℝ | p.1 + p.2 - 2 = 0}) → 
    dist P A = dist P B → 
    dist P (0,0) = 1) → 
  ∃ C : ℝ × ℝ, C = (1/2, 1/2) ∧ 
  (∃ P, tangent_point A B (0,0) → 
  (P = C)) := 
sorry

end tangent_fixed_point_l147_147720


namespace evaluate_ceiling_sum_l147_147639

theorem evaluate_ceiling_sum :
  (⌈Real.sqrt (16 / 9)⌉ : ℤ) + (⌈(16 / 9: ℝ)⌉ : ℤ) + (⌈(16 / 9: ℝ)^2⌉ : ℤ) = 8 := 
by
  -- Placeholder for proof
  sorry

end evaluate_ceiling_sum_l147_147639


namespace midpoint_AD_quadrilateral_parallelogram_l147_147983

section cyclic_quadrilateral_proof

variables {A B C D O P X M Y N : Type}

-- Conditions
axiom cyclic_quadrilateral (ABCD : quadrilateral) : is_cyclic ABCD
axiom diagonals_intersect_perpendicular (ABCD : quadrilateral) (AC BD : line) (P : point)
  (h1 : is_diagonal AC ABCD) (h2 : is_diagonal BD ABCD) (h3 : intersects AC BD P) : is_perpendicular AC BD
axiom center_of_circle (O ABCD : point) (h1 : is_center_of_circumcircle O ABCD) : true
axiom line_r_properties (P X M : point) (r : line) (BC AD : line)
  (h1 : originates_from P r) (h2 : is_perpendicular r BC) (h3 : intersects r BC X) (h4 : intersects r AD M) : true
axiom line_s_properties (P Y N : point) (s : line) (AD BC : line)
  (h1 : originates_from P s) (h2 : is_perpendicular s AD) (h3 : intersects s AD Y) (h4 : intersects s BC N) : true

-- To Prove
theorem midpoint_AD (ABCD : quadrilateral) (AC BD : line) (P O M : point)
  (hyp1 : is_cyclic ABCD) 
  (hyp2 : is_diagonal AC ABCD) 
  (hyp3 : is_diagonal BD ABCD) 
  (hyp4 : intersects AC BD P) 
  (hyp5 : is_perpendicular AC BD) 
  (hyp6 : is_center_of_circumcircle O ABCD) 
  (hyp7 : originates_from P r) 
  (hyp8 : is_perpendicular r BC) 
  (hyp9 : intersects r BC X) 
  (hyp10 : intersects r AD M) : 
  is_midpoint M AD :=
sorry

theorem quadrilateral_parallelogram (ABCD : quadrilateral) (AC BD : line) (P O X M Y N : point)
  (hyp1 : is_cyclic ABCD) 
  (hyp2 : is_diagonal AC ABCD) 
  (hyp3 : is_diagonal BD ABCD) 
  (hyp4 : intersects AC BD P) 
  (hyp5 : is_perpendicular AC BD) 
  (hyp6 : is_center_of_circumcircle O ABCD) 
  (hyp7 : originates_from P r) 
  (hyp8 : is_perpendicular r BC) 
  (hyp9 : intersects r BC X) 
  (hyp10 : intersects r AD M) 
  (hyp11 : originates_from P s) 
  (hyp12 : is_perpendicular s AD) 
  (hyp13 : intersects s AD Y) 
  (hyp14 : intersects s BC N) : 
  is_parallelogram (OMPN OM PN) :=
sorry

end cyclic_quadrilateral_proof

end midpoint_AD_quadrilateral_parallelogram_l147_147983


namespace eval_expression_l147_147684

noncomputable def ceil_sqrt_16_div_9 : ℕ := ⌈Real.sqrt (16 / 9 : ℚ)⌉
noncomputable def ceil_16_div_9 : ℕ := ⌈(16 / 9 : ℚ)⌉
noncomputable def ceil_square_16_div_9 : ℕ := ⌈(16 / 9 : ℚ)^2⌉

theorem eval_expression : ceil_sqrt_16_div_9 + ceil_16_div_9 + ceil_square_16_div_9 = 8 :=
by
  -- The following sorry is a placeholder, indicating that the proof is skipped.
  sorry

end eval_expression_l147_147684


namespace num_three_digit_multiples_of_105_l147_147367

theorem num_three_digit_multiples_of_105 : 
  (∃ n : ℕ, (∀ k : ℕ, (100 ≤ n * 105 ∧ n * 105 ≤ 999) → k = 9)) :=
begin
  sorry
end

end num_three_digit_multiples_of_105_l147_147367


namespace quadratic_root_relationship_l147_147758

theorem quadratic_root_relationship
  (m1 m2 : ℝ)
  (x1 x2 x3 x4 : ℝ)
  (h_eq1 : m1 * x1^2 + (1 / 3) * x1 + 1 = 0)
  (h_eq2 : m1 * x2^2 + (1 / 3) * x2 + 1 = 0)
  (h_eq3 : m2 * x3^2 + (1 / 3) * x3 + 1 = 0)
  (h_eq4 : m2 * x4^2 + (1 / 3) * x4 + 1 = 0)
  (h_order : x1 < x3 ∧ x3 < x4 ∧ x4 < x2 ∧ x2 < 0) :
  m2 > m1 ∧ m1 > 0 :=
sorry

end quadratic_root_relationship_l147_147758


namespace largest_multiple_15_under_500_l147_147079

-- Define the problem statement
theorem largest_multiple_15_under_500 : ∃ (n : ℕ), n < 500 ∧ n % 15 = 0 ∧ (∀ m, m < 500 ∧ m % 15 = 0 → m ≤ n) :=
by {
  use 495,
  split,
  {
    exact lt_of_le_of_ne (le_refl _) (ne_of_lt (nat.lt_succ_self 499))
  },
  split,
  {
    exact nat.mod_eq_zero_of_dvd ⟨33, rfl⟩,
  },
  {
    intros m hm h,
    have hmn : m ≤ 495,
    {
      have div15 := hm/15,
      suffices : div15 < 33+1, from nat.le_of_lt_add_one this,
      exact this, sorry,
    },
    exact hmn,
  },
}

end largest_multiple_15_under_500_l147_147079


namespace number_of_sophomores_l147_147975

theorem number_of_sophomores (n x : ℕ) (freshmen seniors selected freshmen_selected : ℕ)
  (h_freshmen : freshmen = 450)
  (h_seniors : seniors = 250)
  (h_selected : selected = 60)
  (h_freshmen_selected : freshmen_selected = 27)
  (h_eq : selected / (freshmen + seniors + x) = freshmen_selected / freshmen) :
  x = 300 := by
  sorry

end number_of_sophomores_l147_147975


namespace ceiling_sum_l147_147647

theorem ceiling_sum :
  let a := 4 / 3
  let b := 16 / 9
  let c := 256 / 81
  ⌈a⌉ + ⌈b⌉ + ⌈c⌉ = 8 := by
  sorry

end ceiling_sum_l147_147647


namespace simplify_and_evaluate_expression_l147_147870

variable (x y : ℚ)

theorem simplify_and_evaluate_expression :
    x = 2 / 15 → y = 3 / 2 → 
    (2 * x + y)^2 - (3 * x - y)^2 + 5 * x * (x - y) = 1 :=
by 
  intros h1 h2
  subst h1
  subst h2
  sorry

end simplify_and_evaluate_expression_l147_147870


namespace time_to_cross_approximately_19_seconds_l147_147216

-- Define the necessary variables
def length_of_train : ℝ := 124
def length_of_platform : ℝ := 234.9176
def speed_of_train_kmph : ℝ := 68
def speed_conversion_factor : ℝ := 1000 / 3600

-- Calculate the speed of the train in m/s
def speed_of_train_mps : ℝ := speed_of_train_kmph * speed_conversion_factor

-- Calculate the total distance to be covered
def total_distance : ℝ := length_of_train + length_of_platform

-- Calculate the time to cross the platform
def time_to_cross : ℝ := total_distance / speed_of_train_mps

-- State the main theorem
theorem time_to_cross_approximately_19_seconds :
  abs (time_to_cross - 19) < 1 :=
by
  -- Placeholder for actual proof
  sorry

end time_to_cross_approximately_19_seconds_l147_147216


namespace P_neg2_lt_ξ_lt_0_l147_147846

-- Let ξ be a random variable following a normal distribution N(μ, ε^2)
axiom ξ : ℝ → ℝ
axiom μ ε : ℝ

-- Assume ξ follows N(μ, ε^2)
axiom ξ_normal : ∀ x, ξ x ~ Normal μ (ε^2)

-- Given conditions
axiom P_ξ_lt_neg2 : Prob(ξ < -2) = 0.3
axiom P_ξ_gt_2 : Prob(ξ > 2) = 0.3

theorem P_neg2_lt_ξ_lt_0 : Prob(-2 < ξ < 0) = 0.2 := by
    sorry

end P_neg2_lt_ξ_lt_0_l147_147846


namespace simplify_2M_minus_N_value_when_x_neg2_y_neg4_y_must_be_one_fourth_for_independence_of_x_l147_147762

def M (x y : ℝ) : ℝ := x^2 + xy + 2y - 2
def N (x y : ℝ) : ℝ := 2x^2 - 2xy + x - 4

theorem simplify_2M_minus_N (x y : ℝ) : 2 * M x y - N x y = 4 * x * y + 4 * y - x := 
by sorry

theorem value_when_x_neg2_y_neg4 : 2 * M (-2) (-4) - N (-2) (-4) = 18 := 
by sorry

theorem y_must_be_one_fourth_for_independence_of_x (x y : ℝ) (h : ((4 * y - 1) * x + 4 * y = 0)) : y = 1 / 4 :=
by sorry

end simplify_2M_minus_N_value_when_x_neg2_y_neg4_y_must_be_one_fourth_for_independence_of_x_l147_147762


namespace rectangle_perimeter_inequality_l147_147985

-- Define rectilinear perimeters
def perimeter (length : ℝ) (width : ℝ) : ℝ := 2 * (length + width)

-- Definitions for rectangles contained within each other
def rectangle_contained (len1 wid1 len2 wid2 : ℝ) : Prop :=
  len1 ≤ len2 ∧ wid1 ≤ wid2

-- Statement of the problem
theorem rectangle_perimeter_inequality (l1 w1 l2 w2 : ℝ) (h : rectangle_contained l1 w1 l2 w2) :
  perimeter l1 w1 ≤ perimeter l2 w2 :=
sorry

end rectangle_perimeter_inequality_l147_147985


namespace appropriate_sampling_methods_l147_147411

-- Definitions of the conditions
def num_families : ℕ := 400
def num_high_income : ℕ := 120
def num_middle_income : ℕ := 180
def num_low_income : ℕ := 100
def sample_size_families : ℕ := 100

def num_volleyball_players : ℕ := 12
def sample_size_volleyball : ℕ := 3

-- The propositions representing the sampling methods for different surveys
def sampling_method_families : Prop := (sample_size_families <= num_families) ∧ (num_high_income + num_middle_income + num_low_income = num_families)
def appropriate_sampling_families : Prop := sampling_method_families → stratified_sampling

def sampling_method_volleyball : Prop := (sample_size_volleyball <= num_volleyball_players)
def appropriate_sampling_volleyball : Prop := sampling_method_volleyball → random_sampling

-- Proof problem: Prove that given the conditions, the appropriate sampling methods are as described
theorem appropriate_sampling_methods :
  appropriate_sampling_families ∧ appropriate_sampling_volleyball :=
by
  sorry

end appropriate_sampling_methods_l147_147411


namespace length_of_symmetric_closed_line_le_200_l147_147855

-- Definition of the grid size and symmetry conditions
def gridSize : Nat := 15

-- We assume the following conditions
variables 
  (is_closed_non_intersecting : Prop) -- The line does not intersect itself and is closed
  (connects_centers_adjacent_cells : Prop) -- The line connects centers of adjacent cells
  (is_symmetric_to_diagonal : Prop) -- The line is symmetric with respect to one of the main diagonals

-- Final proof problem
theorem length_of_symmetric_closed_line_le_200
  (h1 : is_closed_non_intersecting)
  (h2 : connects_centers_adjacent_cells)
  (h3 : is_symmetric_to_diagonal) :
  ∃ length ≤ 200, length := sorry

end length_of_symmetric_closed_line_le_200_l147_147855


namespace evaluate_expression_l147_147654

theorem evaluate_expression : 
  (⌈Real.sqrt (16 / 9)⌉ + ⌈ (16 / 9 : ℝ ) ⌉ + ⌈Real.pow (16 / 9 : ℝ ) 2⌉) = 8 := 
by 
  sorry

end evaluate_expression_l147_147654


namespace odd_periodic_function_l147_147768

theorem odd_periodic_function (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_period : ∀ x : ℝ, f (x + 5) = f x)
  (h_f1 : f 1 = 1)
  (h_f2 : f 2 = 2) :
  f 3 - f 4 = -1 :=
sorry

end odd_periodic_function_l147_147768


namespace camping_trip_percentage_l147_147373

theorem camping_trip_percentage (T : ℝ)
  (h1 : 16 / 100 ≤ 1)
  (h2 : T - 16 / 100 ≤ 1)
  (h3 : T = 64 / 100) :
  T = 64 / 100 := by
  sorry

end camping_trip_percentage_l147_147373


namespace proof_problem_l147_147671

def sqrt_frac : ℚ := real.sqrt (16 / 9)
def frac : ℚ := 16 / 9
def square_frac : ℚ := frac * frac

def ceil_sqrt_frac : ℤ := ⌈sqrt_frac⌉.to_int
def ceil_frac : ℤ := ⌈frac⌉.to_int
def ceil_square_frac : ℤ := ⌈square_frac⌉.to_int

theorem proof_problem :
  ceil_sqrt_frac + ceil_frac + ceil_square_frac = 8 :=
by
  -- Placeholder for the actual proof.
  sorry

end proof_problem_l147_147671


namespace product_eq_1280_l147_147164

axiom eq1 (a b c d : ℝ) : 2 * a + 4 * b + 6 * c + 8 * d = 48
axiom eq2 (a b c d : ℝ) : 4 * d + 2 * c = 2 * b
axiom eq3 (a b c d : ℝ) : 4 * b + 2 * c = 2 * a
axiom eq4 (a b c d : ℝ) : c - 2 = d
axiom eq5 (a b c d : ℝ) : d + b = 10

theorem product_eq_1280 (a b c d : ℝ) : 2 * a + 4 * b + 6 * c + 8 * d = 48 → 4 * d + 2 * c = 2 * b → 4 * b + 2 * c = 2 * a → c - 2 = d → d + b = 10 → a * b * c * d = 1280 :=
by 
  intro h1 h2 h3 h4 h5
  -- we put the proof here
  sorry

end product_eq_1280_l147_147164


namespace circle_equation_l147_147267

-- Define the circle's equation as a predicate
def is_circle (x y a b r : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

-- Given conditions, defining the known center and passing point
def center_x : ℝ := 2
def center_y : ℝ := -3
def point_M_x : ℝ := -1
def point_M_y : ℝ := 1

-- Prove that the circle with the given conditions has the correct equation
theorem circle_equation :
  is_circle x y center_x center_y 5 ↔ 
  ∀ x y : ℝ, (x - center_x)^2 + (y + center_y)^2 = 25 := sorry

end circle_equation_l147_147267


namespace largest_prime_factor_of_expr_l147_147270

open Nat

theorem largest_prime_factor_of_expr : 
  let term1 := 20 ^ 3
  let term2 := 15 ^ 4
  let term3 := 10 ^ 5
  let expression := term1 + term2 - term3
  prime 37 ∧ (∀ p : ℕ, prime p → p ∣ expression → p ≤ 37) := 
by
  let term1 := 20 ^ 3
  let term2 := 15 ^ 4
  let term3 := 10 ^ 5
  let expression := term1 + term2 - term3
  have h : expression = 2 * 37 * 5 ^ 3 := sorry
  have prime_37 : prime 37 := sorry
  have largest_prime_factor : (∀ p : ℕ, prime p → p ∣ expression → p ≤ 37) := sorry
  exact ⟨prime_37, largest_prime_factor⟩

end largest_prime_factor_of_expr_l147_147270


namespace mrs_hilt_has_more_money_l147_147465

/-- Mrs. Hilt has two pennies, two dimes, and two nickels. 
    Jacob has four pennies, one nickel, and one dime. 
    Prove that Mrs. Hilt has $0.13 more than Jacob. -/
theorem mrs_hilt_has_more_money 
  (hilt_pennies hilt_dimes hilt_nickels : ℕ)
  (jacob_pennies jacob_dimes jacob_nickels : ℕ)
  (value_penny value_nickel value_dime : ℝ)
  (H1 : hilt_pennies = 2) (H2 : hilt_dimes = 2) (H3 : hilt_nickels = 2)
  (H4 : jacob_pennies = 4) (H5 : jacob_dimes = 1) (H6 : jacob_nickels = 1)
  (H7 : value_penny = 0.01) (H8 : value_nickel = 0.05) (H9 : value_dime = 0.10) :
  ((hilt_pennies * value_penny + hilt_dimes * value_dime + hilt_nickels * value_nickel) 
   - (jacob_pennies * value_penny + jacob_dimes * value_dime + jacob_nickels * value_nickel) 
   = 0.13) :=
by sorry

end mrs_hilt_has_more_money_l147_147465


namespace bob_percentage_improvement_correct_l147_147945

def seconds_in_minute : Int := 60

def bob_time_minutes : Int := 10
def bob_time_seconds : Int := 40
def bob_total_seconds : Int :=
  (bob_time_minutes * seconds_in_minute) + bob_time_seconds

def sister_time_minutes : Int := 9
def sister_time_seconds : Int := 17
def sister_total_seconds : Int :=
  (sister_time_minutes * seconds_in_minute) + sister_time_seconds

def time_difference : Int :=
  bob_total_seconds - sister_total_seconds

def percentage_improvement : Real :=
  (time_difference.toReal / bob_total_seconds.toReal) * 100

theorem bob_percentage_improvement_correct :
  percentage_improvement ≈ 12.97 :=
sorry

end bob_percentage_improvement_correct_l147_147945


namespace light_glow_time_l147_147885

def elapsed_time (start_hour start_min start_sec end_hour end_min end_sec : ℕ) : ℕ :=
  let start_total_seconds := start_hour * 3600 + start_min * 60 + start_sec
  let end_total_seconds := end_hour * 3600 + end_min * 60 + end_sec
  end_total_seconds - start_total_seconds

theorem light_glow_time :
  let total_seconds := elapsed_time 1 57 58 3 20 47
  let max_glows := 155.28125
  (total_seconds / max_glows) = 32 :=
by
  sorry

end light_glow_time_l147_147885


namespace largest_prime_factor_of_expr_l147_147268

open Nat

theorem largest_prime_factor_of_expr : 
  let term1 := 20 ^ 3
  let term2 := 15 ^ 4
  let term3 := 10 ^ 5
  let expression := term1 + term2 - term3
  prime 37 ∧ (∀ p : ℕ, prime p → p ∣ expression → p ≤ 37) := 
by
  let term1 := 20 ^ 3
  let term2 := 15 ^ 4
  let term3 := 10 ^ 5
  let expression := term1 + term2 - term3
  have h : expression = 2 * 37 * 5 ^ 3 := sorry
  have prime_37 : prime 37 := sorry
  have largest_prime_factor : (∀ p : ℕ, prime p → p ∣ expression → p ≤ 37) := sorry
  exact ⟨prime_37, largest_prime_factor⟩

end largest_prime_factor_of_expr_l147_147268


namespace factorize_eq_l147_147697

theorem factorize_eq (x : ℝ) : 2 * x^3 - 8 * x = 2 * x * (x + 2) * (x - 2) := 
by
  sorry

end factorize_eq_l147_147697


namespace consecutive_nums_sum_as_product_l147_147952

theorem consecutive_nums_sum_as_product {n : ℕ} (h : 100 < n) :
  ∃ (a b c : ℕ), (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c) ∧ (2 ≤ a) ∧ (2 ≤ b) ∧ (2 ≤ c) ∧ 
  ((n + (n+1) + (n+2) = a * b * c) ∨ ((n+1) + (n+2) + (n+3) = a * b * c)) :=
by
  sorry

end consecutive_nums_sum_as_product_l147_147952


namespace dried_mushroom_mass_l147_147715

-- Define the conditions
def fresh_mushroom_water_content : ℚ := 0.90
def dried_mushroom_water_content : ℚ := 0.12
def mass_fresh_mushrooms : ℚ := 44

-- Define the problem statement
theorem dried_mushroom_mass :
  let dried_mushroom_dry_content := 1 - dried_mushroom_water_content in
  let dry_substance := mass_fresh_mushrooms * (1 - fresh_mushroom_water_content) in
  let mass_dried_mushrooms := dry_substance / dried_mushroom_dry_content in
  mass_dried_mushrooms = 5 :=
by
  let dried_mushroom_dry_content := 1 - dried_mushroom_water_content
  let dry_substance := mass_fresh_mushrooms * (1 - fresh_mushroom_water_content)
  let mass_dried_mushrooms := dry_substance / dried_mushroom_dry_content
  show mass_dried_mushrooms = 5
  sorry

end dried_mushroom_mass_l147_147715


namespace triangle_AE_eq_DE_l147_147430

-- Define the triangle and the relevant points
variables {A B C F D E : Point}
variables {AF BC AB : Line}

-- Given conditions
variables (H1 : is_triangle A B C)
variables (H2 : is_median AF (triangle_point A B C))
variables (H3 : is_midpoint F B C)
variables (H4 : is_midpoint D A F)
variables (H5 : intersect_at E (line_through C D) AB)
variables (H6 : dist B D = dist B F)

-- The theorem to prove
theorem triangle_AE_eq_DE (A B C F D E : Point)
  (AF : Line) (BC : Line) (AB : Line)
  (H1 : is_triangle A B C)
  (H2 : is_median AF (triangle_point A B C))
  (H3 : is_midpoint F B C)
  (H4 : is_midpoint D A F)
  (H5 : intersect_at E (line_through C D) AB)
  (H6 : dist B D = dist B F) 
  : dist A E = dist D E :=
  sorry

end triangle_AE_eq_DE_l147_147430


namespace horse_revolutions_l147_147976

theorem horse_revolutions (r1 r2 r3 : ℝ) (rev1 : ℕ) 
  (h1 : r1 = 30) (h2 : r2 = 15) (h3 : r3 = 10) (h4 : rev1 = 40) :
  (r2 / r1 = 1 / 2 ∧ 2 * rev1 = 80) ∧ (r3 / r1 = 1 / 3 ∧ 3 * rev1 = 120) :=
by
  sorry

end horse_revolutions_l147_147976


namespace goshawk_nature_reserve_l147_147408

-- Define the problem statement and conditions
def percent_hawks (H W K : ℝ) : Prop :=
  ∃ H W K : ℝ,
    -- Condition 1: 35% of the birds are neither hawks, paddyfield-warblers, nor kingfishers
    1 - (H + W + K) = 0.35 ∧
    -- Condition 2: 40% of the non-hawks are paddyfield-warblers
    W = 0.40 * (1 - H) ∧
    -- Condition 3: There are 25% as many kingfishers as paddyfield-warblers
    K = 0.25 * W ∧
    -- Given all conditions, calculate the percentage of hawks
    H = 0.65

theorem goshawk_nature_reserve :
  ∃ H W K : ℝ,
    1 - (H + W + K) = 0.35 ∧
    W = 0.40 * (1 - H) ∧
    K = 0.25 * W ∧
    H = 0.65 := by
    -- Proof is omitted
    sorry

end goshawk_nature_reserve_l147_147408


namespace proposition_R_is_converse_negation_of_P_l147_147771

variables (x y : ℝ)

def P : Prop := x + y = 0 → x = -y
def Q : Prop := ¬(x + y = 0) → x ≠ -y
def R : Prop := x ≠ -y → ¬(x + y = 0)

theorem proposition_R_is_converse_negation_of_P : R x y ↔ ¬P x y :=
by sorry

end proposition_R_is_converse_negation_of_P_l147_147771


namespace total_photos_l147_147471

def initial_photos : ℕ := 100
def photos_first_week : ℕ := 50
def photos_second_week : ℕ := 2 * photos_first_week
def photos_third_fourth_week : ℕ := 80
def photos_from_bali : ℕ := photos_first_week + photos_second_week + photos_third_fourth_week

theorem total_photos (initial_photos photos_from_bali : ℕ) : initial_photos + photos_from_bali = 330 :=
by
  have h1 : initial_photos = 100 := rfl
  have h2 : photos_from_bali = 50 + (2 * 50) + 80 := rfl
  show 100 + (50 + 100 + 80) = 330
  sorry

end total_photos_l147_147471


namespace system_solutions_l147_147494

-- Define the system of equations
def system_equations (x y z : ℂ) : Prop :=
  (x^2 + 2 * y * z = x) ∧
  (y^2 + 2 * z * x = z) ∧
  (z^2 + 2 * x * y = y)

-- List all solutions
def solutions : set (ℂ × ℂ × ℂ) :=
{
  (0, 0, 0),
  (2/3, -(1/3), -(1/3)),
  (1/3, (-(1) + complex.sqrt 3 * complex.I)/6, (-(1) - complex.sqrt 3 * complex.I)/6),
  (1/3, (-(1) - complex.sqrt 3 * complex.I)/6, (-(1) + complex.sqrt 3 * complex.I)/6),
  (1, 0, 0),
  (1/3, 1/3, 1/3),
  (2/3, (1 + complex.sqrt 3 * complex.I)/6, (1 - complex.sqrt 3 * complex.I)/6),
  (2/3, (1 - complex.sqrt 3 * complex.I)/6, (1 + complex.sqrt 3 * complex.I)/6)
}

-- Prove that the solutions set is the solution to the system of equations
theorem system_solutions : ∀ x y z : ℂ, system_equations x y z ↔ (x, y, z) ∈ solutions := 
by sorry

end system_solutions_l147_147494


namespace single_elimination_games_l147_147211

theorem single_elimination_games (n : ℕ) (h : n = 24) :
  let games_played := n - 1 in
  games_played = 23 :=
by
  have h2 := h ▸ rfl
  rw [h2]
  sorry

end single_elimination_games_l147_147211


namespace sub_fixed_points_sum_zero_l147_147769

-- Define a sub-fixed point for function f
def is_sub_fixed_point (f : ℝ → ℝ) (t : ℝ) : Prop := f t = -t

-- Define the functions f and g
def f (x : ℝ) : ℝ := Real.log x
def g (x : ℝ) : ℝ := Real.exp x

-- Define the sum of all sub-fixed points of the given functions
def sub_fixed_points_sum (m : ℝ) : Prop :=
  (∃ t : ℝ, is_sub_fixed_point f t) ∧ 
  (∃ u : ℝ, is_sub_fixed_point g u) ∧
  (∃ t u : ℝ, is_sub_fixed_point f t ∧ is_sub_fixed_point g u ∧ m = t + u)

theorem sub_fixed_points_sum_zero :
  sub_fixed_points_sum 0 :=
sorry

end sub_fixed_points_sum_zero_l147_147769


namespace emily_strawberry_harvest_l147_147635

-- Define the dimensions of the garden
def garden_length : ℕ := 10
def garden_width : ℕ := 7

-- Define the planting density
def plants_per_sqft : ℕ := 3

-- Define the yield per plant
def strawberries_per_plant : ℕ := 12

-- Define the expected number of strawberries
def expected_strawberries : ℕ := 2520

-- Theorem statement to prove the total number of strawberries
theorem emily_strawberry_harvest :
  garden_length * garden_width * plants_per_sqft * strawberries_per_plant = expected_strawberries :=
by
  -- Proof goes here (for now, we use sorry to indicate the proof is omitted)
  sorry

end emily_strawberry_harvest_l147_147635


namespace beaker_difference_l147_147632

variable (A B : ℝ)

theorem beaker_difference :
  A + B = 9.28 ∧ A = 2.95 → B - A = 3.38 :=
by
  intro h
  cases h with h1 h2
  have h3 : B = 9.28 - A := by linarith
  rw [h2, h3]
  linarith
  sorry

end beaker_difference_l147_147632


namespace max_f_value_inequality_m_n_l147_147750

section
variable (x : ℝ)

def f (x : ℝ) := abs (x - 1) - 2 * abs (x + 1)

theorem max_f_value : ∃ k, (∀ x : ℝ, f x ≤ k) ∧ (∃ x₀ : ℝ, f x₀ = k) ∧ k = 2 := 
by sorry

theorem inequality_m_n (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : 1 / m + 1 / (2 * n) = 2) :
  m + 2 * n ≥ 2 :=
by sorry

end

end max_f_value_inequality_m_n_l147_147750


namespace min_reciprocal_sum_max_sum_l147_147303

theorem min_reciprocal_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x^2 + y^2 = x + y) :
    2 ≤ 1/x + 1/y :=
begin
  sorry
end

theorem max_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x^2 + y^2 = x + y) :
    x + y ≤ 2 :=
begin
  sorry
end

end min_reciprocal_sum_max_sum_l147_147303


namespace exponential_function_decreasing_l147_147881

theorem exponential_function_decreasing (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  (0 < a ∧ a < 1) → ¬ (∀ x : ℝ, x > 0 → a ^ x > 0) :=
by
  sorry

end exponential_function_decreasing_l147_147881


namespace evaluate_ceiling_sum_l147_147644

theorem evaluate_ceiling_sum :
  (⌈Real.sqrt (16 / 9)⌉ : ℤ) + (⌈(16 / 9: ℝ)⌉ : ℤ) + (⌈(16 / 9: ℝ)^2⌉ : ℤ) = 8 := 
by
  -- Placeholder for proof
  sorry

end evaluate_ceiling_sum_l147_147644


namespace largest_multiple_of_15_below_500_l147_147038

theorem largest_multiple_of_15_below_500 : ∃ (k : ℕ), 15 * k < 500 ∧ ∀ (m : ℕ), 15 * m < 500 → 15 * m ≤ 15 * k := 
by
  existsi 33
  split
  · norm_num
  · intro m h
    have h1 : m ≤ 33 := Nat.le_of_lt_succ (Nat.div_lt_succ 500 15 m)
    exact (mul_le_mul_left (by norm_num : 0 < 15)).mpr h1
  sorry

end largest_multiple_of_15_below_500_l147_147038


namespace sqrt2_sufficient_for_2x_3_l147_147955

theorem sqrt2_sufficient_for_2x_3 : 
  (∀ x : ℝ, x < sqrt 2 → 2 * x < 3) ∧ ¬ (∀ x : ℝ, 2 * x < 3 → x < sqrt 2) :=
begin
  split,
  { intros x hx,
    calc 2 * x < 2 * sqrt 2 : mul_lt_mul_of_pos_left hx (by norm_num : (0 : ℝ) < 2)
         ... < 3 : by norm_num [sqrt_lt, real.sqrt_lt] },
  { intro h,
    have contra : 2 * (3 / 2) < 3 := by calc 2 * (3 / 2) = 3 : by norm_num,
    specialize h (3 / 2) contra,
    have x_non_lt : ¬ (3 / 2 < sqrt 2) := by norm_num [sqrt_lt, real.sqrt_lt],
    contradiction }
end

end sqrt2_sufficient_for_2x_3_l147_147955


namespace largest_multiple_15_under_500_l147_147084

-- Define the problem statement
theorem largest_multiple_15_under_500 : ∃ (n : ℕ), n < 500 ∧ n % 15 = 0 ∧ (∀ m, m < 500 ∧ m % 15 = 0 → m ≤ n) :=
by {
  use 495,
  split,
  {
    exact lt_of_le_of_ne (le_refl _) (ne_of_lt (nat.lt_succ_self 499))
  },
  split,
  {
    exact nat.mod_eq_zero_of_dvd ⟨33, rfl⟩,
  },
  {
    intros m hm h,
    have hmn : m ≤ 495,
    {
      have div15 := hm/15,
      suffices : div15 < 33+1, from nat.le_of_lt_add_one this,
      exact this, sorry,
    },
    exact hmn,
  },
}

end largest_multiple_15_under_500_l147_147084


namespace solve_triangle_problem_l147_147404

noncomputable def triangle_B_angle (a b c : ℝ) (C : ℝ) (h1 : b * Real.cos C = (2 * a - c) * Real.cos B) : ℝ :=
  B

noncomputable def triangle_area (a b c : ℝ) (h_b : b = Real.sqrt 7) (h_ac : a + c = 4) : ℝ :=
  let B := Real.pi / 3
  let ac := 3
  (1 / 2) * ac * (Real.sqrt 3 / 2)

theorem solve_triangle_problem (a b c : ℝ) (C : ℝ) (h1 : b * Real.cos C = (2 * a - c) * Real.cos (Real.pi / 3))
  (hb : b = Real.sqrt 7) (hac : a + c = 4) :
  triangle_B_angle a b c C h1 = Real.pi / 3 ∧
  triangle_area a b c hb hac = (3 * Real.sqrt 3) / 4 :=
by {
  sorry,
}

end solve_triangle_problem_l147_147404


namespace trig_ratios_l147_147339

noncomputable def r (x y : ℕ) : ℕ := Real.sqrt (x^2 + y^2)

theorem trig_ratios (x y : ℕ) (h : x = 3) (k : y = 4) :
  sin (α) = y / r x y ∧ 
  cos (α) = x / r x y ∧ 
  tan (α) = y / x := 
  begin
    have hr : r 3 4 = 5 := by norm_num, -- Using norm_num to provide the calculated r
    rw [h, k, hr],
    split,
    {
      calc sin (α) = 4 / 5 : sorry,
    },
    split,
    {
      calc cos (α) = 3 / 5 : sorry,
    },
    {
      calc tan (α) = 4 / 3 : sorry,
    }
  end

end trig_ratios_l147_147339


namespace small_pizza_price_l147_147891

variable (S : ℝ)

theorem small_pizza_price :
  (∃ (S : ℝ),
    let revenue_large := 3 * 8 in
    let revenue_total := 40 in
    let revenue_small := revenue_total - revenue_large in
    8 * S = revenue_small) → S = 2 :=
by
  intro h
  obtain ⟨S, h⟩ := h
  simp only [mul_comm] at h
  have h1 : 3 * 8 = 24 := by norm_num
  have h2 : 40 - 24 = 16 := by norm_num
  rw [h1, h2] at h
  linarith
  sorry

end small_pizza_price_l147_147891


namespace find_p_plus_q_l147_147449

noncomputable def circle_eq (c : ℝ × ℝ) (r : ℝ) (x y : ℝ) := (x - c.1)^2 + (y - c.2)^2 = r^2

theorem find_p_plus_q :
  let u1 := circle_eq (-4, 10) 12 in
  let u2 := circle_eq (4, 10) 8 in
  let y_eq_bx (b x : ℝ) := b * x in
  ∃ (b : ℝ) (p q : ℕ), (Nat.gcd p q = 1) ∧ n : ℝ, u3 := circle_eq (c.1, y_eq_bx b c.1) r x y → 
  (r + 12 = sqrt((c.1 + 4)^2 + (y_eq_bx b c.1 - 10)^2)) ∧ 
  (8 - r = sqrt((c.1 - 4)^2 + (y_eq_bx b c.1 - 10)^2)) → 
  (n^2 = (p : ℝ) / (q : ℝ)) → 
  p + q = 94 :=
by
  sorry

end find_p_plus_q_l147_147449


namespace total_cars_is_65_l147_147454

theorem total_cars_is_65
    (Cathy_cars : ℕ) 
    (H1 : Cathy_cars = 5)
    (Lindsey_cars : ℕ) 
    (H2 : Lindsey_cars = Cathy_cars + 4)
    (Carol_cars : ℕ) 
    (H3 : Carol_cars = 2 * Cathy_cars)
    (Susan_cars : ℕ) 
    (H4 : Susan_cars = Carol_cars - 2)
    (Erica_cars : ℕ) 
    (H5 : Erica_cars = Lindsey_cars + Nat.round (0.25 * Lindsey_cars))
    (Jack_cars : ℕ) 
    (H6 : Jack_cars = (Susan_cars + Carol_cars) / 2)
    (Kevin_cars : ℕ) 
    (H7 : Kevin_cars = Nat.round (0.9 * (Lindsey_cars + Cathy_cars))) :
    Cathy_cars + Lindsey_cars + Carol_cars + Susan_cars + Erica_cars + Jack_cars + Kevin_cars = 65 :=
    sorry

end total_cars_is_65_l147_147454


namespace billy_video_count_l147_147239

theorem billy_video_count 
  (generate_suggestions : ℕ) 
  (rounds : ℕ) 
  (videos_in_total : ℕ)
  (H1 : generate_suggestions = 15)
  (H2 : rounds = 5)
  (H3 : videos_in_total = generate_suggestions * rounds + 1) : 
  videos_in_total = 76 := 
by
  sorry

end billy_video_count_l147_147239


namespace min_price_per_car_to_avoid_losses_min_order_volume_to_compete_l147_147160

-- Conditions
def purchase_price_per_car := 2.5 -- in million rubles
def customs_duties_per_car := 2 -- in million rubles
def monthly_office_rent := 0.05 -- in million rubles
def monthly_salary := 0.37 -- in million rubles
def other_monthly_expenses := 0.18 -- in million rubles
def order_quantity := 30 -- number of cars
def competitor_price_per_car := 5.3 -- in million rubles
def charging_station_value := 0.4 -- in million rubles

-- The resulting problems as Lean 4 statements:

-- Part (a) - Minimum price per car to avoid losses.
theorem min_price_per_car_to_avoid_losses :
  (order_quantity * (purchase_price_per_car + customs_duties_per_car) + 
  monthly_office_rent + monthly_salary + other_monthly_expenses) / order_quantity = 4.52 :=
  sorry

-- Part (b) - Minimum order volume to compete.
theorem min_order_volume_to_compete :
  ∃ (X : ℕ), X >= 2 ∧ 
  (((purchase_price_per_car + customs_duties_per_car) * X + 
  monthly_office_rent + monthly_salary + other_monthly_expenses) / X) ≤ 
  (competitor_price_per_car - charging_station_value) :=
  sorry

end min_price_per_car_to_avoid_losses_min_order_volume_to_compete_l147_147160


namespace sum_AM_BM_CM_leq_sum_A1M_B1M_C1M_l147_147447

theorem sum_AM_BM_CM_leq_sum_A1M_B1M_C1M
  (O G : Point)
  (A B C M A1 B1 C1 : Point)
  (hcircumcenter : IsCircumcenter O △ABC)
  (hcentroid : IsCentroid G △ABC)
  (hangle : 90 < ∠OMG ∧ ∠OMG < 180)
  (hAM : LineThrough A M)
  (hBM : LineThrough B M)
  (hCM : LineThrough C M)
  (hA1 : LineThroughIntersectCircumcircle A M A1 △ABC)
  (hB1 : LineThroughIntersectCircumcircle B M B1 △ABC)
  (hC1 : LineThroughIntersectCircumcircle C M C1 △ABC) :
  AM + BM + CM ≤ A1M + B1M + C1M := 
sorry

end sum_AM_BM_CM_leq_sum_A1M_B1M_C1M_l147_147447


namespace simplest_common_denominator_l147_147898

theorem simplest_common_denominator (x : ℝ) (h1 : x^2 - 1 ≠ 0) (h2 : x^2 + x ≠ 0) : 
  (∀ a b : ℝ, (a ≠ 0) ∧ (b ≠ 0) ∧ (∃ c d : ℝ, a = x^2 - 1 ∧ b = x^2 + x ∧ c = x ∧ d = (x+1)) → 
  ∃ e : ℝ, e = x * (x + 1) * (x - 1)) :=
begin
  intros a b hb hc,
  cases hb with ha hb,
  cases hb with hb hd,
  rcases hd with ⟨c, d, rfl, rfl, rfl, rfl⟩,
  use x * (x + 1) * (x - 1),
  dsimp,
  assumption
end

end simplest_common_denominator_l147_147898


namespace cone_height_from_sector_l147_147990

theorem cone_height_from_sector (r θ : ℝ) (h : ℝ)
  (hr : r = 5) (hθ : θ = 144) : h = sqrt 21 := 
by
  sorry

end cone_height_from_sector_l147_147990


namespace largest_multiple_of_15_less_than_500_l147_147069

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), (n < 500) ∧ (15 ∣ n) ∧ (∀ m : ℕ, (m < 500) ∧ (15 ∣ m) → m ≤ n) ∧ n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l147_147069


namespace regular_octagon_angle_PTV_l147_147552

-- Definition: regular octagon
structure regular_octagon (P Q R S T U V W : Type) :=
  (edges : list (P × P))
  (unique_vertices : ∀ v, list.count (v:=(P ∪ Q ∪ R ∪ S ∪ T ∪ U ∪ V ∪ W)) = 1)
  (equal_edges : ∀ (e1 e2 : P × P), e1 ∈ edges ∧ e2 ∈ edges → length e1 = length e2)
  (equal_angles : ∀ (v : P ∪ Q ∪ R ∪ S ∪ T ∪ U ∪ V ∪ W), list.count (adj_vertex v edges) = 2)

-- Definition: angle PTV in the regular octagon
noncomputable def angle_PTV := 22.5

-- Proof statement
theorem regular_octagon_angle_PTV (P Q R S T U V W : Type) (h : regular_octagon P Q R S T U V W) : 
  angle_PTV = 22.5 :=
sorry

end regular_octagon_angle_PTV_l147_147552


namespace tangent_line_through_P_line_through_P_chord_length_8_l147_147340

open Set

def circle (x y : ℝ) : Prop := x^2 + y^2 = 25

def point_P : ℝ × ℝ := (3, 4)

def tangent_line (x y : ℝ) : Prop := 3 * x + 4 * y - 25 = 0

def line_m_case1 (x : ℝ) : Prop := x = 3

def line_m_case2 (x y : ℝ) : Prop := 7 * x - 24 * y + 75 = 0

theorem tangent_line_through_P :
  tangent_line point_P.1 point_P.2 :=
sorry

theorem line_through_P_chord_length_8 :
  (∀ x y, circle x y → line_m_case1 x ∨ line_m_case2 x y) :=
sorry

end tangent_line_through_P_line_through_P_chord_length_8_l147_147340


namespace segment_length_parallel_l147_147779

noncomputable def d_length (AB BC AC: ℝ) : ℝ :=
  if h1: AB = BC ∧ BC = 400 ∧ AC = 480 then
    2400 / 11
  else
    0

theorem segment_length_parallel (P: Point) (AB BC AC: ℝ) (d: ℝ) : 
  AB = 400 ∧ BC = 400 ∧ AC = 480 ∧ segment_through_parallel (P ABC) (d) →
  d = 2400 / 11 := 
by 
  sorry

end segment_length_parallel_l147_147779


namespace count_parallelograms_with_conditions_l147_147478

theorem count_parallelograms_with_conditions : 
  let Area := 1000000
  let lattice_points (p : ℝ × ℝ) := ∃ (m n : ℤ), (p.1 = m) ∧ (p.2 = n)
  let B_conditions (B : ℝ × ℝ) := lattice_points B ∧ (B.1 > 0) ∧ (B.1 = B.2) -- B is on y = x
  let D_conditions (D : ℝ × ℝ) := ∃ (k : ℤ), (k > 1) ∧ (lattice_points D) ∧ (D.1 > 0) ∧ (D.2 = k * D.1) -- D is on y = kx
  let parallelogram_area (B D : ℝ × ℝ) := (B.1 * ((B.2) * (D.1) - B.1 * D.2)).nat_abs
  ∃ (B D : ℝ × ℝ), B_conditions B ∧ D_conditions D ∧ parallelogram_area B D = Area ∧ 
  (∃ (n : ℕ), n = 784) :=
sorry

end count_parallelograms_with_conditions_l147_147478


namespace evaluate_ceiling_sum_l147_147637

theorem evaluate_ceiling_sum :
  (⌈Real.sqrt (16 / 9)⌉ : ℤ) + (⌈(16 / 9: ℝ)⌉ : ℤ) + (⌈(16 / 9: ℝ)^2⌉ : ℤ) = 8 := 
by
  -- Placeholder for proof
  sorry

end evaluate_ceiling_sum_l147_147637


namespace largest_multiple_15_under_500_l147_147082

-- Define the problem statement
theorem largest_multiple_15_under_500 : ∃ (n : ℕ), n < 500 ∧ n % 15 = 0 ∧ (∀ m, m < 500 ∧ m % 15 = 0 → m ≤ n) :=
by {
  use 495,
  split,
  {
    exact lt_of_le_of_ne (le_refl _) (ne_of_lt (nat.lt_succ_self 499))
  },
  split,
  {
    exact nat.mod_eq_zero_of_dvd ⟨33, rfl⟩,
  },
  {
    intros m hm h,
    have hmn : m ≤ 495,
    {
      have div15 := hm/15,
      suffices : div15 < 33+1, from nat.le_of_lt_add_one this,
      exact this, sorry,
    },
    exact hmn,
  },
}

end largest_multiple_15_under_500_l147_147082


namespace largest_multiple_of_15_below_500_l147_147029

theorem largest_multiple_of_15_below_500 : ∃ (k : ℕ), 15 * k < 500 ∧ ∀ (m : ℕ), 15 * m < 500 → 15 * m ≤ 15 * k := 
by
  existsi 33
  split
  · norm_num
  · intro m h
    have h1 : m ≤ 33 := Nat.le_of_lt_succ (Nat.div_lt_succ 500 15 m)
    exact (mul_le_mul_left (by norm_num : 0 < 15)).mpr h1
  sorry

end largest_multiple_of_15_below_500_l147_147029


namespace evaluate_expression_l147_147665

theorem evaluate_expression :
  let x := (16 : ℚ) / 9
  in ⌈(√x)⌉ + ⌈x⌉ + ⌈x^2⌉ = 8 :=
by
  let x := (16 : ℚ) / 9
  sorry

end evaluate_expression_l147_147665


namespace sqrt_sum_form_l147_147390

theorem sqrt_sum_form (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : sqrt 6 + 1 / sqrt 6 + sqrt 8 + 1 / sqrt 8 = (a * sqrt 6 + b * sqrt 8) / c)
  (hc_min : ∀ d, ∃ a' b', d > 0 → sqrt 6 + 1 / sqrt 6 + sqrt 8 + 1 / sqrt 8 = (a' * sqrt 6 + b' * sqrt 8) / d  → d ≥ c) :
  a + b + c = 192 :=
by
  sorry

end sqrt_sum_form_l147_147390


namespace find_x_in_interval_8_9_l147_147725

noncomputable def f (x : ℝ) : ℝ :=
  if h : x ∈ Ioc 0 1 then log 2 x else
  if x = 0 then 0 else
  sorry -- The actual formula of f(x) for all x is sophisticated

axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom f_odd_shifted : ∀ x : ℝ, f (x + 1) = -f (-x + 1)
axiom f_at_zero : f 0 = 0

theorem find_x_in_interval_8_9 (x : ℝ) :
  x ∈ Ioc 8 9 → f x + 2 = f (1 / 2) → x = 65 / 8 :=
begin
  sorry -- Proof omitted
end

end find_x_in_interval_8_9_l147_147725


namespace obtuse_triangle_has_two_acute_angles_l147_147360

-- Definitions based on conditions
def is_triangle (angles : list ℝ) : Prop :=
  angles.length = 3 ∧ angles.sum = 180

def is_obtuse_triangle (angles : list ℝ) : Prop :=
  is_triangle angles ∧ angles.any (> 90)

def acute_angles_count (angles : list ℝ) : ℝ :=
  angles.count (< 90)

-- Theorem based on conditions and the correct answer
theorem obtuse_triangle_has_two_acute_angles (angles : list ℝ) :
  is_obtuse_triangle angles → acute_angles_count angles = 2 := by
  sorry

end obtuse_triangle_has_two_acute_angles_l147_147360


namespace min_p1q1_l147_147498

-- Define p(x) and q(x) as monic polynomials with nonnegative integer coefficients
def p (x : ℕ) : polynomial ℕ := sorry
def q (x : ℕ) : polynomial ℕ := sorry

-- The given condition for the polynomials
def condition (x : ℕ) (hx : x ≥ 2) : Prop :=
  (1 : ℚ) / (5 * x : ℚ) ≥ (1 : ℚ) / (q x).eval ↑x - (1 : ℚ) / (p x).eval ↑x ∧
  (1 : ℚ) / (q x).eval ↑x - (1 : ℚ) / (p x).eval ↑x ≥ (1 : ℚ) / (3 * x^2 : ℚ)

-- The proof statement
theorem min_p1q1 : 
  (∀ x : ℕ, x ≥ 2 → condition x (le_of_lt (by decide)))
  → p 1 * q 1 = 3 :=
sorry

end min_p1q1_l147_147498


namespace seq_solution_l147_147518

def seq (n : ℕ) : ℚ

axiom init_cond : seq 1 = 1 / 2

axiom sum_cond : ∀ n : ℕ, 1 ≤ n → (finset.sum (finset.range (n + 1)) seq) = n^2 * (seq n)

theorem seq_solution : ∀ n : ℕ, 1 ≤ n → seq n = 1 / (n * (n + 1)) := sorry

end seq_solution_l147_147518


namespace probability_inner_hexagon_l147_147590

-- Define the setup for the problem
structure HexagonBoard where
  side_length : ℝ  -- The side length for the outer hexagon
  area_outer : ℝ := (3 * Real.sqrt 3 / 2) * (side_length ^ 2)
  area_inner : ℝ := (3 * Real.sqrt 3 / 8) * (side_length ^ 2)

-- Prove the probability is 1/4
theorem probability_inner_hexagon (hb : HexagonBoard) :
  (hb.area_inner / hb.area_outer) = 1 / 4 := 
by 
  sorry

end probability_inner_hexagon_l147_147590


namespace sequence_combination_l147_147249

noncomputable def sequence (t : ℝ) (n : ℕ) : ℝ :=
  match n with
  | 1     => Real.exp t
  | 2     => t * Real.exp t
  | 3     => t^2 * Real.exp t
  | (n+1) => (t^n) * Real.exp t

theorem sequence_combination (t : ℝ) : 
  ∃ n₁ n₂, sequence t n₁ + sequence t n₂ = t^3 * Real.exp t + Real.exp t :=
by
  sorry

end sequence_combination_l147_147249


namespace substract_repeating_decimal_l147_147698

noncomputable def repeating_decimal : ℝ := 1 / 3

theorem substract_repeating_decimal (x : ℝ) (h : x = repeating_decimal) : 
  1 - x = 2 / 3 :=
by
  sorry

end substract_repeating_decimal_l147_147698


namespace group_A_percentage_forgot_homework_l147_147912

theorem group_A_percentage_forgot_homework :
  let group_A_students := 20
  let group_B_students := 80
  let group_B_forgot_percent := 15 / 100
  let total_forgot_percent := 16 / 100
  (group_B_students * group_B_forgot_percent + group_A_students * ?x / 100) / 
  (group_A_students + group_B_students) = total_forgot_percent → 
  ?x = 20 :=
by 
  let group_A_students := 20
  let group_B_students := 80
  let group_B_forgot_percent := 15 / 100
  let total_forgot_percent := 16 / 100
  sorry

end group_A_percentage_forgot_homework_l147_147912


namespace eval_expression_l147_147683

noncomputable def ceil_sqrt_16_div_9 : ℕ := ⌈Real.sqrt (16 / 9 : ℚ)⌉
noncomputable def ceil_16_div_9 : ℕ := ⌈(16 / 9 : ℚ)⌉
noncomputable def ceil_square_16_div_9 : ℕ := ⌈(16 / 9 : ℚ)^2⌉

theorem eval_expression : ceil_sqrt_16_div_9 + ceil_16_div_9 + ceil_square_16_div_9 = 8 :=
by
  -- The following sorry is a placeholder, indicating that the proof is skipped.
  sorry

end eval_expression_l147_147683


namespace ellipse_properties_l147_147744

noncomputable def a : ℚ := 2
noncomputable def b : ℚ := sqrt 3

-- Conditions
def ellipse (a b : ℚ) : set (ℚ × ℚ) :=
{ p | let ⟨x, y⟩ := p in (x / a)^2 + (y / b)^2 = 1 }

def focus (p : ℚ × ℚ) : Prop := p = (1, 0)

def trisection_points (p1 p2 : ℚ × ℚ) (b : ℚ) : Prop :=
let y := b / 3 in p1 = (0, y) ∧ p2 = (0, 2 * y)

def equilateral_triangle (A B C : ℚ × ℚ) : Prop :=
let (x1, y1) := A in
let (x2, y2) := B in
let (x3, y3 := C) in
(y2 - y1)^2 + (x2 - x1)^2 = (y3 - y2)^2 + (x3 - x2)^2 ∧
(y3 - y2)^2 + (x3 - x2)^2 = (y1 - y3)^2 + (x1 - x3)^2

def H : ℚ × ℚ := (3, 0)

def line_through (C : set (ℚ × ℚ)) (H : ℚ × ℚ) :=
{ P | ∃ k : ℚ, P ∈ C ∧ (∃ (x1 x2 : ℚ), (x1 + x2) / 2 = 3) }

-- Problem Statement
theorem ellipse_properties :
  (∃ (a b : ℚ), focus (1, 0) ∧ trisection_points (0, b / 3) (0, 2 * b / 3) b ∧
   equilateral_triangle (0, b / 3) (0, 2 * b / 3) (1, 0) ∧ ellipse a b = (λ ⟨x, y⟩, (x / a)^2 + (y / b)^2 = 1)) ∧
  (∃ t : ℚ, line_through (ellipse a b) H →
   ∀ P ∈ (ellipse a b), (|over ⟨xa, ya⟩ - over ⟨xb, yb⟩| < sqrt 3 →
   t^2 ∈ (20 - sqrt 283, 4)) sorry

end ellipse_properties_l147_147744


namespace proof_problem_l147_147670

def sqrt_frac : ℚ := real.sqrt (16 / 9)
def frac : ℚ := 16 / 9
def square_frac : ℚ := frac * frac

def ceil_sqrt_frac : ℤ := ⌈sqrt_frac⌉.to_int
def ceil_frac : ℤ := ⌈frac⌉.to_int
def ceil_square_frac : ℤ := ⌈square_frac⌉.to_int

theorem proof_problem :
  ceil_sqrt_frac + ceil_frac + ceil_square_frac = 8 :=
by
  -- Placeholder for the actual proof.
  sorry

end proof_problem_l147_147670


namespace largest_multiple_of_15_less_than_500_is_495_l147_147129

-- Define the necessary conditions
def is_multiple_of_15 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 15 * k

def is_less_than_500 (n : ℕ) : Prop :=
  n < 500

-- Problem statement: Prove that the largest positive multiple of 15 less than 500 is 495
theorem largest_multiple_of_15_less_than_500_is_495 : 
  ∀ n : ℕ, is_multiple_of_15 n → is_less_than_500 n → n ≤ 495 := 
by
  sorry

end largest_multiple_of_15_less_than_500_is_495_l147_147129


namespace single_elimination_games_l147_147212

theorem single_elimination_games (n : ℕ) (h : n = 24) :
  let games_played := n - 1 in
  games_played = 23 :=
by
  have h2 := h ▸ rfl
  rw [h2]
  sorry

end single_elimination_games_l147_147212


namespace largest_multiple_of_15_less_than_500_l147_147106

theorem largest_multiple_of_15_less_than_500 :
∀ x : ℕ, (∃ k : ℕ, x = 15 * k ∧ 0 < x ∧ x < 500) → x ≤ 495 :=
begin
  sorry
end

end largest_multiple_of_15_less_than_500_l147_147106


namespace car_R_speed_l147_147171

theorem car_R_speed (v : ℝ) (h1 : ∀ t_R t_P : ℝ, t_R * v = 800 ∧ t_P * (v + 10) = 800) (h2 : ∀ t_R t_P : ℝ, t_P + 2 = t_R) :
  v = 50 := by
  sorry

end car_R_speed_l147_147171


namespace tangent_fixed_point_l147_147721

open Classical

noncomputable def circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

noncomputable def tangent_point (A B O : ℝ × ℝ) :=
∃ P : ℝ × ℝ, (P ∈ line {p : ℝ × ℝ | p.1 + p.2 - 2 = 0}) ∧
  dist P A = dist P B ∧
  dist P O = 1

theorem tangent_fixed_point (A B : ℝ × ℝ) (hA : A ∈ circle) (hB : B ∈ circle) :
  (∀ P, (P ∈ line {p : ℝ × ℝ | p.1 + p.2 - 2 = 0}) → 
    dist P A = dist P B → 
    dist P (0,0) = 1) → 
  ∃ C : ℝ × ℝ, C = (1/2, 1/2) ∧ 
  (∃ P, tangent_point A B (0,0) → 
  (P = C)) := 
sorry

end tangent_fixed_point_l147_147721


namespace combined_mean_score_l147_147854

-- Definitions based on the conditions
def mean_score_class1 : ℕ := 90
def mean_score_class2 : ℕ := 80
def ratio_students (n1 n2 : ℕ) : Prop := n1 / n2 = 2 / 3

-- Proof statement
theorem combined_mean_score (n1 n2 : ℕ) 
  (h1 : ratio_students n1 n2) 
  (h2 : mean_score_class1 = 90) 
  (h3 : mean_score_class2 = 80) : 
  ((mean_score_class1 * n1) + (mean_score_class2 * n2)) / (n1 + n2) = 84 := 
by
  sorry

end combined_mean_score_l147_147854


namespace tangent_line_ST_tangent_to_circumcircle_ABC_l147_147167

-- Definitions of the conditions
variables {A B C O X Y T S : Type*}
variables [circumcenter : O = circumcenter A B C]
variables [circle_w : circle O tangent BC]
variables [X_Y_intersection : X Y = intersection tangentLine A (circle O) BC]
variables [X_side : same_side AO X B]
variables [T_intersection : T = intersection tangentLine (circumcircle A B C) B line_from_X_parallel_to AC]
variables [S_intersection : S = intersection tangentLine (circumcircle A B C) C line_from_Y_parallel_to AB]

-- Theorem statement
theorem tangent_line_ST_tangent_to_circumcircle_ABC :
  tangent ST (circumcircle A B C) :=
sorry

end tangent_line_ST_tangent_to_circumcircle_ABC_l147_147167


namespace floor_diff_bounds_l147_147487

theorem floor_diff_bounds (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  0 ≤ Int.floor (a + b) - (Int.floor a + Int.floor b) ∧ 
  Int.floor (a + b) - (Int.floor a + Int.floor b) ≤ 1 :=
by
  sorry

end floor_diff_bounds_l147_147487


namespace probability_product_less_50_l147_147861

/--
 Paco uses a spinner to select a number from 1 through 5 with equal probability.
 Manu uses a spinner to select a number from 1 through 15 with equal probability.
 Prove that the probability that the product of Manu's number and Paco's number is less than 50
 is equal to 22/25.
-/
theorem probability_product_less_50 :
  let paco_outcomes := Finset.range 5 + 1
  let manu_outcomes := Finset.range 15 + 1
  let num_valid_combinations := paco_outcomes.filter (λ p,
    manu_outcomes.filter (λ m, p * m < 50).card
  ).sum
  let total_combinations := paco_outcomes.card * manu_outcomes.card
  num_valid_combinations / total_combinations = 22 / 25 :=
by
  sorry

end probability_product_less_50_l147_147861


namespace area_rhombus_GHIJ_l147_147204

open EuclideanGeometry

/-- Define the side length of the regular hexagon -/
def side_length : ℝ := 12

/-- Define the vertices of the regular hexagon as points A, B, C, D, E, F -/
variables (A B C D E F : Point)

/-- Define the regular hexagon with side length -/
def regular_hexagon (A B C D E F : Point) : Prop :=
  regular_polygon (A::B::C::D::E::F::[]) side_length

/-- Define the midpoints of sides AB, CD, EF, and BC forming the rhombus GHIJ -/
variables (G H I J : Point)
def rhombus_GHIJ (G H I J : Point) : Prop :=
  is_midpoint G A B ∧ is_midpoint H C D ∧ is_midpoint I E F ∧ is_midpoint J B C ∧ 
  rhombus G H I J

/-- The theorem that the area of the rhombus GHIJ is 72√3 cm² -/
theorem area_rhombus_GHIJ (A B C D E F G H I J : Point) 
  (h_hex : regular_hexagon A B C D E F)
  (h_mid : rhombus_GHIJ G H I J) : 
  area G H I J = 72 * Real.sqrt 3 := 
sorry

end area_rhombus_GHIJ_l147_147204


namespace largest_multiple_of_15_less_than_500_l147_147135

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), 15 * n < 500 ∧ 15 * n ≥ 15 * (33 - 1) + 1 := by
  sorry

end largest_multiple_of_15_less_than_500_l147_147135


namespace largest_multiple_of_15_less_than_500_l147_147141

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), 15 * n < 500 ∧ 15 * n ≥ 15 * (33 - 1) + 1 := by
  sorry

end largest_multiple_of_15_less_than_500_l147_147141


namespace largest_multiple_of_15_less_than_500_l147_147092

theorem largest_multiple_of_15_less_than_500 : ∃ (x : ℕ), (x < 500) ∧ (x % 15 = 0) ∧ ∀ (y : ℕ), (y < 500) ∧ (y % 15 = 0) → y ≤ x :=
begin
  use 495,
  split,
  { norm_num },
  split,
  { norm_num },
  intros y hy1 hy2,
  sorry,
end

end largest_multiple_of_15_less_than_500_l147_147092


namespace find_roots_l147_147283

noncomputable def P (x : ℝ) : ℝ := 4 * x^4 + 17 * x^3 - 37 * x^2 + 6 * x

theorem find_roots : 
  ∀ x, P x = 0 ↔ x = 0 ∨ x = 1/2 ∨ x = (-9 + real.sqrt 129) / 4 ∨ x = (-9 - real.sqrt 129) / 4 :=
by
  sorry

end find_roots_l147_147283


namespace NinaCalculationCorrectAnswer_l147_147466

variable (y : ℝ)

noncomputable def NinaMistakenCalculation (y : ℝ) : ℝ :=
(y + 25) * 5

noncomputable def NinaCorrectCalculation (y : ℝ) : ℝ :=
(y - 25) / 5

theorem NinaCalculationCorrectAnswer (hy : (NinaMistakenCalculation y) = 200) :
  (NinaCorrectCalculation y) = -2 := by
  sorry

end NinaCalculationCorrectAnswer_l147_147466


namespace largest_multiple_of_15_below_500_l147_147032

theorem largest_multiple_of_15_below_500 : ∃ (k : ℕ), 15 * k < 500 ∧ ∀ (m : ℕ), 15 * m < 500 → 15 * m ≤ 15 * k := 
by
  existsi 33
  split
  · norm_num
  · intro m h
    have h1 : m ≤ 33 := Nat.le_of_lt_succ (Nat.div_lt_succ 500 15 m)
    exact (mul_le_mul_left (by norm_num : 0 < 15)).mpr h1
  sorry

end largest_multiple_of_15_below_500_l147_147032


namespace find_price_of_pencil_l147_147872

-- Define the variables and constants given in the problem
variables (n p r : ℝ)

-- Given conditions as equations
def condition1 := 6 * n + 4 * p = 7.44
def condition2 := 3 * n + 7 * p = 6.73
def condition3 := n + 2 * p + r = 3.36

-- The statement to prove
theorem find_price_of_pencil (h1 : condition1) (h2 : condition2) (h3 : condition3) : p = 0.602 :=
sorry

end find_price_of_pencil_l147_147872


namespace QF_length_l147_147350

open Real

noncomputable def focus : Point := (2, 0)
noncomputable def parabola (x y : Real) := y^2 = 8 * x

def directrix (p : Point) := p.fst = -2 -- Assume point P lies on directrix l
def lies_on_parabola (q : Point) := parabola q.fst q.snd

def vector_eq (p q f : Point) := 
  let v_fp := (-4, f.snd - p.snd)
  let v_fq := (q.fst - f.fst, q.snd - f.snd) 
  v_fq = -4 • v_fp

theorem QF_length (Q F : Point) (hP : Point) (h1 : directrix hP)
  (h2 : lies_on_parabola Q) (h3 : vector_eq hP Q F) : |Q - F| = 20 := 
by
  sorry

end QF_length_l147_147350


namespace largest_multiple_15_under_500_l147_147080

-- Define the problem statement
theorem largest_multiple_15_under_500 : ∃ (n : ℕ), n < 500 ∧ n % 15 = 0 ∧ (∀ m, m < 500 ∧ m % 15 = 0 → m ≤ n) :=
by {
  use 495,
  split,
  {
    exact lt_of_le_of_ne (le_refl _) (ne_of_lt (nat.lt_succ_self 499))
  },
  split,
  {
    exact nat.mod_eq_zero_of_dvd ⟨33, rfl⟩,
  },
  {
    intros m hm h,
    have hmn : m ≤ 495,
    {
      have div15 := hm/15,
      suffices : div15 < 33+1, from nat.le_of_lt_add_one this,
      exact this, sorry,
    },
    exact hmn,
  },
}

end largest_multiple_15_under_500_l147_147080


namespace point_O_is_circumcenter_l147_147301

variable {V : Type*} [InnerProductSpace ℝ V]

variables (A B C H O : V)

-- Conditions
def is_in_the_same_plane (A B C H O : V) : Prop := ∃ n, 
  n = (A - B) ∧ n = (B - H) ∧ n = (H - O) -- Simplified to ensure all points are in the same plane

def condition_1 : Prop := (A - H) • (A - B) = (A - H) • (A - C)
def condition_2 : Prop := (B - H) • (B - A) = (B - H) • (B - C)
def condition_3 : Prop := A + B + C - 3 * O = H

-- Conclusion
def is_circumcenter (O A B C : V) : Prop := 
  dist O A = dist O B ∧ dist O B = dist O C

-- Lean theorem statement
theorem point_O_is_circumcenter (A B C H O : V) 
  (h_plane : is_in_the_same_plane A B C H O)
  (h1 : condition_1 A H B C)
  (h2 : condition_2 B H A C)
  (h3 : condition_3 O A B C H) : is_circumcenter O A B C :=
by sorry

end point_O_is_circumcenter_l147_147301


namespace triangle_ratio_l147_147857

theorem triangle_ratio (a d : ℝ) :
  (a ≠ 0 ∧ d ≠ 0) ∧ (∃ θ : ℝ, θ = 120) ∧ (∃ (cos_θ : ℝ), cos_θ = -1/2) →
  ∀ (a + d) + 2d = (∃ r : ℝ, r = 3 / 2) → 
  (a : (a + d) : (a + 2d) = 3 : 5 : 7) :=
by
  sorry

end triangle_ratio_l147_147857


namespace largest_multiple_15_under_500_l147_147087

-- Define the problem statement
theorem largest_multiple_15_under_500 : ∃ (n : ℕ), n < 500 ∧ n % 15 = 0 ∧ (∀ m, m < 500 ∧ m % 15 = 0 → m ≤ n) :=
by {
  use 495,
  split,
  {
    exact lt_of_le_of_ne (le_refl _) (ne_of_lt (nat.lt_succ_self 499))
  },
  split,
  {
    exact nat.mod_eq_zero_of_dvd ⟨33, rfl⟩,
  },
  {
    intros m hm h,
    have hmn : m ≤ 495,
    {
      have div15 := hm/15,
      suffices : div15 < 33+1, from nat.le_of_lt_add_one this,
      exact this, sorry,
    },
    exact hmn,
  },
}

end largest_multiple_15_under_500_l147_147087


namespace minimum_value_of_translated_function_l147_147538

noncomputable def original_function (x : ℝ) : ℝ :=
  2 * sin (π / 3 - x) - cos (π / 6 + x)

noncomputable def translated_function (x : ℝ) : ℝ :=
  2 * sin (π / 3 - (x - π / 4)) - cos (π / 6 + (x - π / 4))

theorem minimum_value_of_translated_function : ∃ x : ℝ, translated_function x = -1 :=
sorry

end minimum_value_of_translated_function_l147_147538


namespace triangle_BC_length_l147_147406

noncomputable def length_BC (α β γ : ℝ) (A B C : ℝ) : ℝ :=
  let AB := 1
  let AC := Real.sqrt 2
  let B := Real.pi / 4 -- 45 degrees in radians
  if AB = 1 ∧ AC = Real.sqrt 2 ∧ B = Real.pi / 4 then
    let C := Real.asin (AB * Real.sin B / AC)
    let A := Real.pi - B - C
    let BC := AB * Real.sin A / Real.sin C
    BC
  else 0

theorem triangle_BC_length :
  length_BC (1 : ℝ) (Real.sqrt 2) (Real.pi / 4) = (Real.sqrt 2 + Real.sqrt 6) / 2 :=
by
  sorry

end triangle_BC_length_l147_147406


namespace complex_root_circle_radius_l147_147225

noncomputable def radius_of_circle : ℂ → ℂ := sorry

theorem complex_root_circle_radius (z : ℂ) :
  (z + 2)^6 = 64 * z^6 → abs z = 2 / 3 :=
begin
  sorry
end

end complex_root_circle_radius_l147_147225


namespace union_example_l147_147452

theorem union_example (P Q : Set ℕ) (hP : P = {1, 2, 3, 4}) (hQ : Q = {2, 4}) :
  P ∪ Q = {1, 2, 3, 4} :=
by
  sorry

end union_example_l147_147452


namespace find_k_l147_147279

theorem find_k (x y z k : ℝ) 
  (h1 : 9 / (x + y) = k / (x + 2 * z)) 
  (h2 : 9 / (x + y) = 14 / (z - y)) 
  (h3 : y = 2 * x) 
  (h4 : x + z = 10) :
  k = 46 :=
by
  sorry

end find_k_l147_147279


namespace largest_multiple_of_15_less_than_500_l147_147117

theorem largest_multiple_of_15_less_than_500 :
∀ x : ℕ, (∃ k : ℕ, x = 15 * k ∧ 0 < x ∧ x < 500) → x ≤ 495 :=
begin
  sorry
end

end largest_multiple_of_15_less_than_500_l147_147117


namespace quadratic_function_choice_l147_147611

-- Define what it means to be a quadratic function
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, f x = a * x^2 + b * x + c

-- Define the given equations as functions
def f_A (x : ℝ) : ℝ := 3 * x
def f_B (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c
def f_C (x : ℝ) : ℝ := (x - 1)^2
def f_D (x : ℝ) : ℝ := 2

-- State the Lean theorem statement
theorem quadratic_function_choice : is_quadratic f_C := sorry

end quadratic_function_choice_l147_147611


namespace sum_of_b_for_one_solution_l147_147369

theorem sum_of_b_for_one_solution :
  let A := 3
  let C := 12
  ∀ b : ℝ, ((b + 5)^2 - 4 * A * C = 0) → (b = 7 ∨ b = -17) → (7 + (-17)) = -10 :=
by
  intro A C b
  sorry

end sum_of_b_for_one_solution_l147_147369


namespace second_batch_students_l147_147528

theorem second_batch_students :
  ∃ x : ℕ,
    (40 * 45 + x * 55 + 60 * 65 : ℝ) / (40 + x + 60) = 56.333333333333336 ∧
    x = 50 :=
by
  use 50
  sorry

end second_batch_students_l147_147528


namespace evaluate_ceiling_sum_l147_147643

theorem evaluate_ceiling_sum :
  (⌈Real.sqrt (16 / 9)⌉ : ℤ) + (⌈(16 / 9: ℝ)⌉ : ℤ) + (⌈(16 / 9: ℝ)^2⌉ : ℤ) = 8 := 
by
  -- Placeholder for proof
  sorry

end evaluate_ceiling_sum_l147_147643


namespace total_photos_l147_147469

def initial_photos : ℕ := 100
def photos_first_week : ℕ := 50
def photos_second_week : ℕ := 2 * photos_first_week
def photos_third_fourth_week : ℕ := 80
def photos_from_bali : ℕ := photos_first_week + photos_second_week + photos_third_fourth_week

theorem total_photos (initial_photos photos_from_bali : ℕ) : initial_photos + photos_from_bali = 330 :=
by
  have h1 : initial_photos = 100 := rfl
  have h2 : photos_from_bali = 50 + (2 * 50) + 80 := rfl
  show 100 + (50 + 100 + 80) = 330
  sorry

end total_photos_l147_147469


namespace max_balloons_with_24_dollars_l147_147461

-- Define the conditions
def cost_of_bag_small : ℝ := 4
def cost_of_bag_medium : ℝ := 6
def cost_of_bag_extralarge : ℝ := 12
def balloons_in_bag_small : ℕ := 50
def balloons_in_bag_medium : ℕ := 75
def balloons_in_bag_extralarge : ℕ := 200
def total_money : ℝ := 24

-- Define the main proof problem statement
theorem max_balloons_with_24_dollars : 
  let cost_per_balloon_small := cost_of_bag_small / balloons_in_bag_small
      cost_per_balloon_medium := cost_of_bag_medium / balloons_in_bag_medium
      cost_per_balloon_extralarge := cost_of_bag_extralarge / balloons_in_bag_extralarge
      num_bags := total_money / cost_of_bag_extralarge
      total_balloons := num_bags.to_nat * balloons_in_bag_extralarge
  in total_balloons = 400 :=
by
  sorry

end max_balloons_with_24_dollars_l147_147461


namespace total_votes_l147_147918

theorem total_votes (x : ℕ) :
  (0.75 * ↑x - 0.25 * ↑x = 150) → (x = 300) :=
by
  sorry

end total_votes_l147_147918


namespace rate_calculation_l147_147202

noncomputable def rate_per_sq_meter
  (lawn_length : ℝ) (lawn_breadth : ℝ)
  (road_width : ℝ) (total_cost : ℝ) : ℝ :=
  let area_road_1 := road_width * lawn_breadth
  let area_road_2 := road_width * lawn_length
  let area_intersection := road_width * road_width
  let total_area_roads := (area_road_1 + area_road_2) - area_intersection
  total_cost / total_area_roads

theorem rate_calculation :
  rate_per_sq_meter 100 60 10 4500 = 3 := by
  sorry

end rate_calculation_l147_147202


namespace train_length_proof_l147_147606

variable (time_crosses_bridge : ℝ := 14.284571519992687)
variable (length_bridge : ℝ := 150)
variable (speed_train_kmph : ℝ := 63)

def kmph_to_mps (speed_kmph : ℝ) : ℝ := speed_kmph * (1000 / 3600)

theorem train_length_proof :
  let speed_train := kmph_to_mps speed_train_kmph in
  let total_distance := speed_train * time_crosses_bridge in
  total_distance - length_bridge = 99.98 :=
by
  let speed_train := kmph_to_mps speed_train_kmph
  let total_distance := speed_train * time_crosses_bridge
  have h : total_distance = 249.980001599894 := sorry -- Calculation placeholder
  rw [h]
  norm_num [length_bridge]

end train_length_proof_l147_147606


namespace construct_triangle_l147_147921

-- Define the sides and the included angle as given conditions
variables (a b C : ℝ)

-- The proof statement: we can construct a triangle with given sides and included angle
theorem construct_triangle (a b C : ℝ) : ∃ (A B C : Point), 
      distance A B = a ∧ 
      distance A C = b ∧
      angle A B C = C :=
by
  sorry

end construct_triangle_l147_147921


namespace largest_multiple_of_15_less_than_500_l147_147046

theorem largest_multiple_of_15_less_than_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ (∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x) := sorry

end largest_multiple_of_15_less_than_500_l147_147046


namespace correct_serial_numbers_l147_147609

def random_table : list (list ℕ) :=
  [ [8442, 1753, 3157, 2455, 0688, 7704, 7447, 6721, 7633, 5026, 8392],
    [6301, 5316, 5916, 9275, 3862, 9821, 5071, 7512, 8673, 5807, 4439],
    [1326, 3321, 1342, 7864, 1607, 8252, 0744, 3815, 0324, 4299, 7931] ]

def extract_serial_numbers (table : list (list ℕ)) (start_row start_col : ℕ) : list ℕ :=
  -- Implementation to extract the first three valid serial numbers
  -- Note: The actual extraction logic is omitted for brevity.
  [165, 538, 629] -- This should be the result according to the sample provided.

theorem correct_serial_numbers :
  extract_serial_numbers random_table 8 7 = [165, 538, 629] :=
by
  sorry -- Proof to be constructed.

end correct_serial_numbers_l147_147609


namespace infinite_n_exists_l147_147572

def prime_factors (n : ℕ) : List ℕ :=
  -- Function to return the list of prime factors of n (implementation left as an exercise)
  sorry

def omega (n : ℕ) : ℕ :=
  prime_factors(n).prod

def f (n : ℕ) : ℤ :=
  (-1)^(omega n)

theorem infinite_n_exists :
  ∃^∞ (n : ℕ), f (n-1) = 1 ∧ f n = 1 ∧ f (n+1) = 1 :=
  sorry

end infinite_n_exists_l147_147572


namespace find_d_l147_147827

def f (x : ℝ) (c : ℝ) : ℝ := 5 * x + c
def g (x : ℝ) (c : ℝ) : ℝ := c * x + 3

theorem find_d (c d : ℝ) (h : ∀ x : ℝ, f (g x c) c = 15 * x + d) : d = 18 :=
by
  sorry

end find_d_l147_147827


namespace max_of_three_diff_pos_int_with_mean_7_l147_147917

theorem max_of_three_diff_pos_int_with_mean_7 (a b c : ℕ) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_mean : (a + b + c) / 3 = 7) :
  max a (max b c) = 18 := 
sorry

end max_of_three_diff_pos_int_with_mean_7_l147_147917


namespace min_distance_parabola_circle_l147_147770

theorem min_distance_parabola_circle :
  let P := (x₁, y₁) in
  let Q := (x₂, y₂) in
  (y₁^2 = x₁) ∧ ((x₂ - 3)^2 + y₂^2 = 1) → 
  ∃ d, d = |((x₁ - x₂)^2 + (y₁ - y₂)^ 2)^(1/2)| 
         ∧ d = (sqrt 11) / 2 - 1 :=
sorry

end min_distance_parabola_circle_l147_147770


namespace largest_multiple_of_15_less_than_500_l147_147048

theorem largest_multiple_of_15_less_than_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ (∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x) := sorry

end largest_multiple_of_15_less_than_500_l147_147048


namespace area_of_right_triangle_with_hypotenuse_and_angle_l147_147513

theorem area_of_right_triangle_with_hypotenuse_and_angle 
  (hypotenuse : ℝ) (angle : ℝ) 
  (hyp_eq : hypotenuse = 10) (angle_eq : angle = 30) :
  ∃ area : ℝ, area = 25 * real.sqrt 3 / 2 :=
by
  sorry

end area_of_right_triangle_with_hypotenuse_and_angle_l147_147513


namespace domain_of_f_2x_plus_1_l147_147332

theorem domain_of_f_2x_plus_1 {f : ℝ → ℝ} :
  (∀ x, (-2 : ℝ) ≤ x ∧ x ≤ 3 → (-3 : ℝ) ≤ x - 1 ∧ x - 1 ≤ 2) →
  (∀ x, (-3 : ℝ) ≤ x ∧ x ≤ 2 → (-2 : ℝ) ≤ (x : ℝ) ∧ x ≤ 1/2) →
  ∀ x, (-2 : ℝ) ≤ x ∧ x ≤ 1 / 2 → ∀ y, y = 2 * x + 1 → (-3 : ℝ) ≤ y ∧ y ≤ 2 :=
by
  sorry

end domain_of_f_2x_plus_1_l147_147332


namespace houses_built_during_boom_l147_147532

theorem houses_built_during_boom :
  let original_houses := 20817
  let current_houses := 118558
  let houses_built := current_houses - original_houses
  houses_built = 97741 := by
  sorry

end houses_built_during_boom_l147_147532


namespace ExpandedOHaraTripleValue_l147_147534

/-- Define an Expanded O'Hara triple -/
def isExpandedOHaraTriple (a b x : ℕ) : Prop :=
  2 * (Nat.sqrt a + Nat.sqrt b) = x

/-- Prove that for given a=64 and b=49, x is equal to 30 if (a, b, x) is an Expanded O'Hara triple -/
theorem ExpandedOHaraTripleValue (a b x : ℕ) (ha : a = 64) (hb : b = 49) (h : isExpandedOHaraTriple a b x) : x = 30 :=
by
  sorry

end ExpandedOHaraTripleValue_l147_147534


namespace largest_multiple_of_15_less_than_500_l147_147055

theorem largest_multiple_of_15_less_than_500 : 
  ∃ n : ℕ, n * 15 < 500 ∧ ∀ m : ℕ, m * 15 < 500 → m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l147_147055


namespace cost_per_chair_l147_147634

theorem cost_per_chair (total_spent : ℕ) (chairs_bought : ℕ) (cost : ℕ) 
  (h1 : total_spent = 180) 
  (h2 : chairs_bought = 12) 
  (h3 : cost = total_spent / chairs_bought) : 
  cost = 15 :=
by
  -- Proof steps go here (skipped with sorry)
  sorry

end cost_per_chair_l147_147634


namespace remainder_of_poly_l147_147150

-- Define the function f
def f (x : ℝ) : ℝ := x^4 - 6 * x^3 + 12 * x^2 + 18 * x - 22

-- State the theorem about the remainder using the Remainder Theorem
theorem remainder_of_poly : (eval 4 (f four)) = 114 :=
by
sorority

end remainder_of_poly_l147_147150


namespace quadratic_trinomial_with_integral_roots_l147_147238

theorem quadratic_trinomial_with_integral_roots (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
  (∃ x : ℤ, a * x^2 + b * x + c = 0) ∧ 
  (∃ x : ℤ, (a + 1) * x^2 + (b + 1) * x + (c + 1) = 0) ∧ 
  (∃ x : ℤ, (a + 2) * x^2 + (b + 2) * x + (c + 2) = 0) :=
sorry

end quadratic_trinomial_with_integral_roots_l147_147238


namespace fourth_power_of_cube_root_l147_147883

theorem fourth_power_of_cube_root :
  ((3 + real.sqrt (1 + real.sqrt 5)) ^ (1 / 3)) ^ 4 = 9 + 12 * real.sqrt 6 :=
by
  sorry

end fourth_power_of_cube_root_l147_147883


namespace remaining_wire_length_l147_147180

theorem remaining_wire_length (total_wire_length : ℝ) (square_side_length : ℝ) 
  (h₀ : total_wire_length = 60) (h₁ : square_side_length = 9) : 
  total_wire_length - 4 * square_side_length = 24 :=
by
  sorry

end remaining_wire_length_l147_147180


namespace ceiling_sum_l147_147645

theorem ceiling_sum :
  let a := 4 / 3
  let b := 16 / 9
  let c := 256 / 81
  ⌈a⌉ + ⌈b⌉ + ⌈c⌉ = 8 := by
  sorry

end ceiling_sum_l147_147645


namespace units_digit_fraction_l147_147557

theorem units_digit_fraction :
  (30 * 31 * 32 * 33 * 34 * 35) % 5000 % 10 = 6 := by
  -- Factoring the terms to their prime components
  have h30 : 30 = 2 * 3 * 5 := rfl
  have h31 : 31 = 31 := rfl
  have h32 : 32 = 2^5 := rfl
  have h33 : 33 = 3 * 11 := rfl
  have h34 : 34 = 2 * 17 := rfl
  have h35 : 35 = 5 * 7 := rfl
  have h5000 : 5000 = 2^3 * 5^3 := rfl

  -- Starting the proof process using the given conditions
  sorry

end units_digit_fraction_l147_147557


namespace square_area_l147_147935

-- Definitions
def price_per_foot : ℝ := 59
def total_cost : ℝ := 4012
def side_length_of_square (total_cost price_per_foot : ℝ) : ℝ := total_cost / (4 * price_per_foot)
def area_of_square (side_length : ℝ) : ℝ := side_length^2

-- Theorem statement
theorem square_area (total_cost price_per_foot : ℝ) (H_cost : total_cost = 4012) (H_price : price_per_foot = 59) :
  area_of_square (side_length_of_square total_cost price_per_foot) = 289 :=
by
  sorry

end square_area_l147_147935


namespace curve_C_equation_minimum_distance_l147_147916

noncomputable def equation_of_curve_C : Prop :=
  ∀ M : ℝ × ℝ, (∃ (x y : ℝ), M = (x, y) ∧ (let A : ℝ × ℝ := (0, -1)
                                               B : ℝ × ℝ := (x, -3)
                                               MA : ℝ × ℝ := (0 - x, -1 - y)
                                               MB : ℝ × ℝ := (0 - x, -1 - -3)
                                               MM : ℝ × ℝ := (x, - y)
                                           in (MA.1 * MB.1 + MA.2 * MB.2 + MA.1 * MM.1 + MA.2 * MM.2 = 0)))
    → y = x^2 - 2

noncomputable def minimum_distance_to_tangent : Prop :=
  ∀ P : ℝ × ℝ, (∃ (x0 y0 : ℝ), P = (x0, y0) ∧ y0 = x0^2 - 2) 
    → (let l := λ (x : ℝ), 2 * x0 * x - 2 * x0^2 + x0 in
       ∀ O : ℝ × ℝ, let O := (0, 0) in 
       O.dist l = 2)

-- Prove the propositions
theorem curve_C_equation : equation_of_curve_C := sorry
theorem minimum_distance : minimum_distance_to_tangent := sorry

end curve_C_equation_minimum_distance_l147_147916


namespace quadrilateral_is_rhombus_l147_147796

-- Define the points A, B, C, and D.
variables (A B C D : Type) [Add] [Zero A] [Zero B] [Zero C] [Zero D]

-- Define the segments AC and BD.
def AC := A + C
def BD := B + D

-- Introduce the condition that AC is perpendicular to BD.
axiom AC_perpendicular_BD : ⊥ AC BD

-- Introduce the condition that AC and BD bisect each other.
axiom bisect_each_other : AC /= BD

-- The theorem statement.
theorem quadrilateral_is_rhombus : is_rhombus A B C D :=
sorry

end quadrilateral_is_rhombus_l147_147796


namespace with_replacement_distribution_with_replacement_expectation_without_replacement_distribution_without_replacement_expectation_l147_147595

-- Define the conditions
def balls : ℕ := 8
def red_balls : ℕ := 4
def black_balls : ℕ := 4

-- Define the events for winning and losing - This is only a conceptualization, not complete implementation
def winning_draw (with_replacement : Bool) : ProbabilitySpace → bool := sorry
def losing_draw (with_replacement : Bool) : ProbabilitySpace → bool := sorry

-- Part 1: With Replacement
theorem with_replacement_distribution :
  ∀ X : ℕ, X ∈ {0, 1, 2} →
  P(X = 0) = 16/49 ∧ P(X = 1) = 24/49 ∧ P(X = 2) = 9/49 := sorry

theorem with_replacement_expectation :
  E(X) = 6/7 := sorry

-- Part 2: Without Replacement
theorem without_replacement_distribution :
  ∀ Y : ℕ, Y ∈ {0, 1, 2} →
  P(Y = 0) = 12/35 ∧ P(Y = 1) = 16/35 ∧ P(Y = 2) = 7/35 := sorry

theorem without_replacement_expectation :
  E(Y) = 6/7 := sorry

end with_replacement_distribution_with_replacement_expectation_without_replacement_distribution_without_replacement_expectation_l147_147595


namespace largest_multiple_of_15_less_than_500_l147_147077

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), (n < 500) ∧ (15 ∣ n) ∧ (∀ m : ℕ, (m < 500) ∧ (15 ∣ m) → m ≤ n) ∧ n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l147_147077


namespace batsman_average_increase_l147_147579

theorem batsman_average_increase 
  (A : ℕ)
  (h1 : ∀ n ≤ 11, (1 / (n : ℝ)) * (A * n + 60) = 38) 
  (h2 : 1 / 12 * (A * 11 + 60) = 38)
  (h3 : ∀ n ≤ 12, (A * n : ℝ) ≤ (A * (n + 1) : ℝ)) :
  38 - A = 2 := 
sorry

end batsman_average_increase_l147_147579


namespace speed_downstream_l147_147979

def speed_in_still_water := 12 -- man in still water
def speed_of_stream := 6  -- speed of stream
def speed_upstream := 6  -- rowing upstream

theorem speed_downstream : 
  speed_in_still_water + speed_of_stream = 18 := 
by 
  sorry

end speed_downstream_l147_147979


namespace trailingZeros_310_fact_l147_147241

-- Define the function to compute trailing zeros in factorials
def trailingZeros (n : ℕ) : ℕ := 
  if n = 0 then 0 else n / 5 + trailingZeros (n / 5)

-- Define the specific case for 310!
theorem trailingZeros_310_fact : trailingZeros 310 = 76 := 
by 
  sorry

end trailingZeros_310_fact_l147_147241


namespace number_of_chapters_l147_147580

theorem number_of_chapters (total_pages : ℕ) (pages_per_chapter : ℕ) (h_total : total_pages = 1891) (h_per_chapter : pages_per_chapter = 61) :
  total_pages / pages_per_chapter = 31 :=
by
  rw [h_total, h_per_chapter]
  norm_num

end number_of_chapters_l147_147580


namespace problem_statement_l147_147482

noncomputable def binomial_coefficient (n m : ℕ) : ℚ :=
(n.factorial) / ((m.factorial) * ((n - m).factorial))

def S (n : ℕ) : ℤ :=
∑ m in finset.range (n + 1), (-1 : ℤ) ^ m * (binomial_coefficient n m).num

theorem problem_statement :
  1990 * (∑ m in finset.range (996), (-1 : ℤ) ^ m * (binomial_coefficient 1990 m).num) + 1 = 0 :=
sorry

end problem_statement_l147_147482


namespace prob_one_white_ball_eq_prob_three_white_balls_eq_l147_147410

noncomputable def prob_one_white_ball (m n : ℕ) : ℝ :=
  if h : n ≠ 0 then (m: ℝ) / (n: ℝ) else 0

noncomputable def prob_three_white_balls (m n : ℕ) : ℝ :=
  if h : n ≠ 0 then (m: ℝ) / (n: ℝ) else 0

theorem prob_one_white_ball_eq (m n : ℕ) (h₀ : 0 < n) :
  prob_one_white_ball m n = (m: ℝ) / (n: ℝ) :=
by {
  rw [prob_one_white_ball],
  split_ifs with h,
  { refl },
  { contradiction },
  sorry
}

theorem prob_three_white_balls_eq (m n : ℕ) (h₀ : 0 < n) :
  prob_three_white_balls m n = (m: ℝ) / (n: ℝ) :=
by {
  rw [prob_three_white_balls],
  split_ifs with h,
  { refl },
  { contradiction },
  sorry
}

end prob_one_white_ball_eq_prob_three_white_balls_eq_l147_147410


namespace rationalize_expression_l147_147383

theorem rationalize_expression :
  let expr := (Real.sqrt 6 + 1 / Real.sqrt 6 + Real.sqrt 8 + 1 / Real.sqrt 8) in
  ∃ (a b c : ℕ), c > 0 ∧
    c * expr = ↑a * Real.sqrt 6 + ↑b * Real.sqrt 8 ∧
    a + b + c = 106 :=
by
  sorry

end rationalize_expression_l147_147383


namespace A_work_rate_l147_147182

theorem A_work_rate {
  let A_rate : ℚ := 1 / x -- rate of A.
  let B_rate : ℚ := 1 / 8 -- rate of B.
  let C_rate : ℚ := C_rate -- rate of C.

  -- Combined rate for A, B, and C.
  assume combined_rate : (1 / x + B_rate + C_rate = 1 / 3),

  -- Rs. 300 for A; Rs. 225 for B; Rs. 75 for C,
  -- implying their work rate ratio.
  assume ratio_A_B : A_rate = 4 / 3 * B_rate,
  assume ratio_A_C : A_rate = 4 * C_rate,

  -- From the problem solution assertions:
  have A_work_days : x = 6 := by sorry
} -- the theorem for A's work rate proving x = 6

end A_work_rate_l147_147182


namespace anna_rearrangements_time_l147_147236

theorem anna_rearrangements_time :
  let name := ["A", "N", "N", "A"]
  let rearrangements_per_minute := 8
  let total_rearrangements := Nat.factorial 4 / (Nat.factorial 2 * Nat.factorial 2)
  let time_in_minutes := total_rearrangements / rearrangements_per_minute
  let time_in_hours := time_in_minutes / 60
  time_in_hours = 0.0125 :=
by
  let name := ["A", "N", "N", "A"]
  let rearrangements_per_minute := 8
  let total_rearrangements := Nat.factorial 4 / (Nat.factorial 2 * Nat.factorial 2)
  let time_in_minutes := total_rearrangements / rearrangements_per_minute
  let time_in_hours := time_in_minutes / 60
  have h : time_in_hours = (6 / 8) / 60 := by
    simp [total_rearrangements, Nat.factorial]
    norm_num
  rw h
  norm_num
  exact rfl

end anna_rearrangements_time_l147_147236


namespace number_not_palindrome_l147_147242

theorem number_not_palindrome (n : ℕ) (h : 1 < n) : 
  ¬(let number := (list.range n).bind (λ i, to_string (i + 1)) in 
    number = number.reverse) :=
sorry

end number_not_palindrome_l147_147242


namespace pieces_from_friend_l147_147231

theorem pieces_from_friend (F : ℕ) (total_needed : ℕ := 376) (brother_given : ℕ := 136) (still_needed : ℕ := 117) 
  (total_have_now : ℕ := total_needed - still_needed) 
  (friend_given : ℕ := total_have_now - brother_given) :
  F = friend_given ↔ F = 123 :=
by
  have total_have_now : ℕ := total_needed - still_needed,
  have friend_given : ℕ := total_have_now - brother_given,
  simp_all,
  sorry

end pieces_from_friend_l147_147231


namespace midpoint_3d_l147_147266

/-- Midpoint calculation in 3D space -/
theorem midpoint_3d (x1 y1 z1 x2 y2 z2 : ℝ) : 
  (x1, y1, z1) = (2, -3, 6) → 
  (x2, y2, z2) = (8, 5, -4) → 
  ((x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2) = (5, 1, 1) := 
by
  intros
  sorry

end midpoint_3d_l147_147266


namespace largest_multiple_of_15_less_than_500_l147_147063

theorem largest_multiple_of_15_less_than_500 : 
  ∃ n : ℕ, n * 15 < 500 ∧ ∀ m : ℕ, m * 15 < 500 → m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l147_147063


namespace ceiling_sum_l147_147651

theorem ceiling_sum :
  let a := 4 / 3
  let b := 16 / 9
  let c := 256 / 81
  ⌈a⌉ + ⌈b⌉ + ⌈c⌉ = 8 := by
  sorry

end ceiling_sum_l147_147651


namespace solve_for_x_l147_147491

-- Define the given condition
def condition (x : ℝ) : Prop := (x - 5) ^ 3 = -((1 / 27)⁻¹)

-- State the problem as a Lean theorem
theorem solve_for_x : ∃ x : ℝ, condition x ∧ x = 2 := by
  sorry

end solve_for_x_l147_147491


namespace median_of_six_students_l147_147958

noncomputable def median_of_heights : ℝ :=
  let heights := [1.72, 1.78, 1.75, 1.80, 1.69, 1.77]
  in ((heights.sorted!!2) + (heights.sorted!!3)) / 2

theorem median_of_six_students :
  median_of_heights = 1.76 :=
by
  sorry

end median_of_six_students_l147_147958


namespace largest_multiple_of_15_less_than_500_l147_147015

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), n < 500 ∧ 15 ∣ n ∧ ∀ (m : ℕ), m < 500 ∧ 15 ∣ m -> m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l147_147015


namespace largest_multiple_of_15_less_than_500_l147_147098

theorem largest_multiple_of_15_less_than_500 : ∃ (x : ℕ), (x < 500) ∧ (x % 15 = 0) ∧ ∀ (y : ℕ), (y < 500) ∧ (y % 15 = 0) → y ≤ x :=
begin
  use 495,
  split,
  { norm_num },
  split,
  { norm_num },
  intros y hy1 hy2,
  sorry,
end

end largest_multiple_of_15_less_than_500_l147_147098


namespace tangent_line_angle_APB_vector_sum_max_line_AB_through_fixed_point_final_result_l147_147718

-- Definitions for points A, B on circle O and line l
def O (x y : ℝ) := x^2 + y^2 = 1
def l (x y : ℝ) := x + y - 2 = 0

-- Tangent Line
theorem tangent_line (x y : ℝ) (hx : O x y) : l x y → |1| / (real.sqrt (1^2 + 1^2)) = 1 :=
by sorry

-- Right Angle Condition
theorem angle_APB (A B : ℝ × ℝ) (hA : O A.1 A.2) (hB : O B.1 B.2) (P : ℝ × ℝ) : ¬ (l P.1 P.2 ∧ angle A P B = π / 2) :=
by sorry

-- Vector Sum Condition
theorem vector_sum_max (A B : ℝ × ℝ) (hA : O A.1 A.2) (hB : O B.1 B.2) (P : ℝ × ℝ) : |(A.1 - P.1 + B.1 - P.1)| = (|√3| + 1) → max_sum :=
by sorry

-- Line AB Passing Through Fixed Point
theorem line_AB_through_fixed_point (A B : ℝ × ℝ) (hA : O A.1 A.2) (hB : O B.1 B.2) (P : ℝ × ℝ) (tangentPA : O P.1 P.2) (tangentPB : O P.1 P.2) :
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = (1/2, 1/2) :=
by sorry

-- Final result combining correct options
theorem final_result : (tangent_line ∧ line_AB_through_fixed_point) :=
by sorry

end tangent_line_angle_APB_vector_sum_max_line_AB_through_fixed_point_final_result_l147_147718


namespace positive_real_solution_eq_l147_147701

theorem positive_real_solution_eq :
  ∃ x : ℝ, 0 < x ∧ ( (1/4) * (5 * x^2 - 4) = (x^2 - 40 * x - 5) * (x^2 + 20 * x + 2) ) ∧ x = 20 + 10 * Real.sqrt 41 :=
by
  sorry

end positive_real_solution_eq_l147_147701


namespace largest_multiple_of_15_less_than_500_l147_147072

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), (n < 500) ∧ (15 ∣ n) ∧ (∀ m : ℕ, (m < 500) ∧ (15 ∣ m) → m ≤ n) ∧ n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l147_147072


namespace sqrt_sum_form_l147_147387

theorem sqrt_sum_form (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : sqrt 6 + 1 / sqrt 6 + sqrt 8 + 1 / sqrt 8 = (a * sqrt 6 + b * sqrt 8) / c)
  (hc_min : ∀ d, ∃ a' b', d > 0 → sqrt 6 + 1 / sqrt 6 + sqrt 8 + 1 / sqrt 8 = (a' * sqrt 6 + b' * sqrt 8) / d  → d ≥ c) :
  a + b + c = 192 :=
by
  sorry

end sqrt_sum_form_l147_147387


namespace largest_multiple_of_15_less_than_500_l147_147060

theorem largest_multiple_of_15_less_than_500 : 
  ∃ n : ℕ, n * 15 < 500 ∧ ∀ m : ℕ, m * 15 < 500 → m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l147_147060


namespace geometric_sequence_sum_l147_147786

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ)
  (h_geometric : ∀ n, a (n + 1) = r * a n)
  (h_sum1 : a 1 + a 2 = 40)
  (h_sum2 : a 3 + a 4 = 60) :
  a 5 + a 6 = 90 :=
sorry

end geometric_sequence_sum_l147_147786


namespace female_listeners_count_female_listeners_percentage_l147_147903

-- Definitions based on the given conditions
def total_listeners : ℕ := 240
def male_listeners : ℕ := 85
def female_non_listeners : ℕ := 125

-- Definitions implied by the table
def total_non_listeners : ℕ := 190 := 125 + 105

-- Statements to prove
theorem female_listeners_count (total_listeners male_listeners : ℕ) : 
  total_listeners = 240 → 
  male_listeners = 85 → 
  females_listen = total_listeners - male_listeners :=
begin
  intros t m,
  have f_listen := t - m,
  exact f_listen
end

theorem female_listeners_percentage (total_listeners female_listeners : ℕ) :
  total_listeners = 240 → 
  (female_listeners / total_listeners) * 100 = 64.58 := 
begin
  intros t f,
  have p := (f.to_float / t.to_float) * 100,
  exact p
end

end female_listeners_count_female_listeners_percentage_l147_147903


namespace image_of_f_l147_147812

noncomputable def f (a b c : ℝ) : ℝ :=
  let h := (a * b) / c
  let r := (a + b - c) / 2
  h / r

theorem image_of_f :
  ∀ (a b : ℝ), a > 0 ∧ b > 0 ∧ a^2 + b^2 = c^2 →
  ∃ (y : ℝ), y ∈ Im f ∧ y > 2 ∧ y <= 1 + Real.sqrt 2 :=
sorry

end image_of_f_l147_147812
