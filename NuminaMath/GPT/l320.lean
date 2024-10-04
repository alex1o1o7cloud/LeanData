import Mathlib

namespace goods_train_speed_l320_320197

theorem goods_train_speed
  (length_train : ℝ) (length_platform : ℝ) (time_seconds : ℝ)
  (h_train_length : length_train = 250.0416)
  (h_platform_length : length_platform = 270)
  (h_time : time_seconds = 26) :
  (length_train + length_platform) / time_seconds * 3.6 = 72 := by
    sorry

end goods_train_speed_l320_320197


namespace count_nat_divisible_by_sqrt_floor_l320_320393

def is_nat_divisible_by_sqrt_floor (N : ℕ) : Prop :=
  N % (floor (real.sqrt N)) = 0

theorem count_nat_divisible_by_sqrt_floor :
  (finset.range 1000001).filter (λ N, is_nat_divisible_by_sqrt_floor N).card = 1000 :=
by
  sorry

end count_nat_divisible_by_sqrt_floor_l320_320393


namespace determine_a_l320_320358

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def f (a : ℝ) (x : ℝ) : ℝ :=
  x^3 * (a * 2^x - 2^(-x))

theorem determine_a : ∃ a : ℝ, is_even_function (f a) ∧ a = 1 :=
by
  use 1
  sorry

end determine_a_l320_320358


namespace option_d_is_correct_l320_320182

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def is_monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a < x → x < y → y < b → f x ≤ f y

theorem option_d_is_correct : 
  is_even_function (λ x, 10 * |x|) ∧ is_monotonically_increasing (λ x, 10 * |x|) 0 (real.to_nnreal (real.sqrt real.inf)) :=
by
  sorry

end option_d_is_correct_l320_320182


namespace fractional_unit_addition_l320_320571

theorem fractional_unit_addition :
  let fractional_unit : ℚ := 1/6,
      num_fractions : ℕ := 7,
      base_fraction : ℚ := 5/6,
      smallest_prime : ℕ := 2 in
  base_fraction + fractional_unit * num_fractions = smallest_prime := 
by
  sorry

end fractional_unit_addition_l320_320571


namespace tan_double_angle_l320_320347

theorem tan_double_angle (α : Real) (h1 : α > π ∧ α < 3 * π / 2) (h2 : Real.sin (π - α) = -3/5) :
  Real.tan (2 * α) = 24/7 := 
by
  sorry

end tan_double_angle_l320_320347


namespace max_min_norm_l320_320817

/-- Given vectors a and b, find the maximum and minimum values of |2a - b| -/
theorem max_min_norm (θ : ℝ) :
    let a := (Real.cos θ, Real.sin θ)
    let b := (Real.sqrt 3, 1)
    ∃ max min, (|2 • a - b| = 4 → max) ∧ (|2 • a - b| = 0 → min) :=
by
    sorry

end max_min_norm_l320_320817


namespace digits_around_decimal_point_l320_320703

open Real

noncomputable def a : ℝ := (sqrt 2 + sqrt 3) ^ 2
noncomputable def b : ℝ := (sqrt 2 - sqrt 3) ^ 2

theorem digits_around_decimal_point :
    (a ^ 1006) ≈ 7.9 :=
sorry

end digits_around_decimal_point_l320_320703


namespace zero_piles_possible_l320_320974

-- Define the types for our problem
def state := (ℕ × ℕ × ℕ)

inductive move
| take_marble : move
| double_pile : ℕ → move -- pile index to double

-- Function to apply a move to a state
def apply_move : move → state → state
| move.take_marble, (a, b, c) => (a - 1, b - 1, c - 1)
| move.double_pile 0, (a, b, c) => (2 * a, b, c)
| move.double_pile 1, (a, b, c) => (a, 2 * b, c)
| move.double_pile 2, (a, b, c) => (a, b, 2 * c)
| _, s => s -- default case to handle invalid moves, although should not be used

-- Define the possible moves the boy can make
inductive reachable : state → state → Prop
| refl : ∀ (s : state), reachable s s
| step : ∀ (s t : state) (m : move), reachable s t →
         reachable s (apply_move m t)

-- The main theorem
theorem zero_piles_possible (s : state) : reachable s (0, 0, 0) :=
Sorry

end zero_piles_possible_l320_320974


namespace minimum_c_l320_320055

axiom natural_numbers (a b c : ℕ) : Prop 

theorem minimum_c (a b c : ℕ) (h1 : a < b) (h2 : b < c)
  (h3 : ∀ x y, 2*x + y = 2021 ∧ y = |x - a| + |x - b| + |x - c| → ∃! x, y) : 
  c ≥ 1011 :=
sorry

end minimum_c_l320_320055


namespace marcia_total_time_proof_l320_320917

-- Define the minutes spent on different activities
def science_project_time : ℕ := 300
def research_time : ℕ := 45
def presentation_time : ℕ := 75
def discussion_time : ℕ := 2 * 60
def rehearsal_time : ℕ := 1.5 * 60

-- Define the total time spent in minutes
def total_minutes : ℕ :=
  science_project_time + research_time + presentation_time + discussion_time + rehearsal_time

-- Convert total minutes to hours (total_minutes / 60)
def total_time_in_hours : ℝ := total_minutes / 60

-- The theorem statement to prove
theorem marcia_total_time_proof : total_time_in_hours = 10.5 :=
by
  sorry

end marcia_total_time_proof_l320_320917


namespace shortest_distance_proof_l320_320623

noncomputable def shortest_distance_on_parallelepiped (a b c : ℝ) (h : a > b) (h2 : b > c) : ℝ :=
  sqrt (a^2 + (b + c)^2)

theorem shortest_distance_proof (a b c : ℝ) (hab : a > b) (hbc : b > c) :
  shortest_distance_on_parallelepiped a b c hab hbc = sqrt (a^2 + (b + c)^2) :=
sorry

end shortest_distance_proof_l320_320623


namespace sum_of_sin_squared_angles_l320_320247

theorem sum_of_sin_squared_angles :
  (∑ i in Finset.range 30, Real.sin (6 * (i + 1)) * Real.sin (6 * (i + 1))) = 15.5 :=
sorry

end sum_of_sin_squared_angles_l320_320247


namespace zero_point_interval_l320_320848

noncomputable def f (x : ℝ) : ℝ := -x^3 - 3 * x + 5

theorem zero_point_interval: 
  ∃ x₀ : ℝ, f x₀ = 0 → 1 < x₀ ∧ x₀ < 2 :=
sorry

end zero_point_interval_l320_320848


namespace closest_point_to_origin_l320_320939

def y (x : ℝ) := x + 1 / x

theorem closest_point_to_origin : ∃ x : ℝ, x > 0 ∧ (x, y x) = (1 / 2^(1/4 : ℝ), (1 + real.sqrt 2) / 2^(1/4 : ℝ)) :=
by
  sorry

end closest_point_to_origin_l320_320939


namespace line_equation_l320_320548

-- Given conditions
variables (k x x0 y y0 : ℝ)
variable (line_passes_through : ∀ x0 y0, y0 = k * x0 + l)
variable (M0 : (ℝ × ℝ))

-- Main statement we need to prove
theorem line_equation (k x x0 y y0 : ℝ) (M0 : (ℝ × ℝ)) (line_passes_through : ∀ x0 y0, y0 = k * x0 + l) :
  y - y0 = k * (x - x0) :=
sorry

end line_equation_l320_320548


namespace positive_integer_solutions_inequality_l320_320101

theorem positive_integer_solutions_inequality :
  {x : ℕ | 2 * x + 9 ≥ 3 * (x + 2)} = {1, 2, 3} :=
by
  sorry

end positive_integer_solutions_inequality_l320_320101


namespace alice_bob_meet_l320_320221

theorem alice_bob_meet :
  ∃ k : ℕ, (4 * k - 4 * (k / 5) ≡ 8 * k [MOD 15]) ∧ (k = 5) :=
by
  sorry

end alice_bob_meet_l320_320221


namespace closest_point_to_origin_l320_320942

def y (x : ℝ) := x + 1 / x

theorem closest_point_to_origin : ∃ x : ℝ, x > 0 ∧ (x, y x) = (1 / 2^(1/4 : ℝ), (1 + real.sqrt 2) / 2^(1/4 : ℝ)) :=
by
  sorry

end closest_point_to_origin_l320_320942


namespace part1_part2_l320_320648

open Set

noncomputable def A : Set ℝ := { x | 1 ≤ x ∧ x ≤ 4 }
def B := λ (a : ℝ), { x : ℝ | x < a }

theorem part1 (x : ℝ) :
  x ∈ (A ∩ (compl (B 3))) ↔ 3 ≤ x ∧ x ≤ 4 := by
  sorry

theorem part2 (a : ℝ) :
  (A ⊆ B a) ↔ 4 < a := by
  sorry

end part1_part2_l320_320648


namespace tan_Y_l320_320883

variable (X Y Z : Type) [MetricSpace X] [MetricSpace Y] [MetricSpace Z]

variable (angle : X → Y → Z → ℝ)
variable (distance : X → X → ℝ)
variable (θ : ℝ)
variable (a b : ℝ)

noncomputable def triangle_data : Prop := 
  ∃ (X Y Z : ℝ),
    (angle X Y Z = θ ∧ θ = real.pi / 2) ∧
    (distance X Z = a ∧ a = 4) ∧ 
    (distance Y Z = b ∧ b = real.sqrt 17) ∧ 
    arctan_rat a (sqrt(1)) = 4

theorem tan_Y {X Y Z : Type} [MetricSpace X] [MetricSpace Y] [MetricSpace Z] : ∀ {X Y Z} (angle : X → Y → Z → ℝ) (distance : X → X → ℝ) (θ : ℝ) (a b : ℝ), 
  (triangle_data X Y Z angle distance θ a b) → 
  ( tan (angle Y Z) = 4 ) := sorry

end tan_Y_l320_320883


namespace minExpression_is_sqrt15_l320_320906

noncomputable def minExpression (a b : ℝ) : ℝ :=
  a^2 + b^2 + 1 / a^2 + 1 / b^2 + b / a + a / b

theorem minExpression_is_sqrt15 {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
  minExpression a b = sqrt 15 :=
sorry

end minExpression_is_sqrt15_l320_320906


namespace probability_at_least_one_multiple_of_4_l320_320229

theorem probability_at_least_one_multiple_of_4 :
  let bound := 50
  let multiples_of_4 := 12
  let probability_no_multiple_of_4 := (38 / 50) * (38 / 50)
  let probability_at_least_one_multiple_of_4 := 1 - probability_no_multiple_of_4
  (probability_at_least_one_multiple_of_4 = 528 / 1250) := 
by
  -- Define the conditions
  let bound := 50
  let multiples_of_4 := 12
  let probability_no_multiple_of_4 := (38 / 50) * (38 / 50)
  let probability_at_least_one_multiple_of_4 := 1 - probability_no_multiple_of_4
  sorry

end probability_at_least_one_multiple_of_4_l320_320229


namespace hyperbola_eccentricity_range_l320_320803

theorem hyperbola_eccentricity_range (a b e : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_upper : b / a < 2) :
  e = Real.sqrt (1 + (b / a) ^ 2) → 1 < e ∧ e < Real.sqrt 5 :=
by
  sorry

end hyperbola_eccentricity_range_l320_320803


namespace sin_pi_minus_alpha_cos_2pi_minus_alpha_sin_minus_cos_l320_320180

-- Problem 1: Given that tan(α) = 3, prove that sin(π - α) * cos(2π - α) = 3 / 10.
theorem sin_pi_minus_alpha_cos_2pi_minus_alpha (α : ℝ) (h : Real.tan α = 3) : 
  Real.sin (Real.pi - α) * Real.cos (2 * Real.pi - α) = 3 / 10 :=
by
  sorry

-- Problem 2: Given that sin(α) * cos(α) = 1/4 and 0 < α < π/4, prove that sin(α) - cos(α) = - sqrt(2) / 2.
theorem sin_minus_cos (α : ℝ) (h₁ : Real.sin α * Real.cos α = 1 / 4) (h₂ : 0 < α) (h₃ : α < Real.pi / 4) :
  Real.sin α - Real.cos α = - (Real.sqrt 2) / 2 :=
by
  sorry

end sin_pi_minus_alpha_cos_2pi_minus_alpha_sin_minus_cos_l320_320180


namespace find_a9_l320_320032

variable (S : ℕ → ℤ) (a : ℕ → ℤ)
variable (d a1 : ℤ)

def arithmetic_seq (n : ℕ) : ℤ :=
  a1 + ↑n * d

def sum_arithmetic_seq (n : ℕ) : ℤ :=
  n * a1 + (n * (n - 1) / 2) * d

axiom h1 : sum_arithmetic_seq 8 = 4 * arithmetic_seq 3
axiom h2 : arithmetic_seq 7 = -2

theorem find_a9 : arithmetic_seq 9 = -6 :=
by
  sorry

end find_a9_l320_320032


namespace trapezoid_area_l320_320279

theorem trapezoid_area (x : ℝ) : 
  (∃ (base₁ base₂ height : ℝ), base₁ = 4 * x ∧ base₂ = 3 * x ∧ height = x) →
  ∃ (area : ℝ), area = (base₁ + base₂) / 2 * height := 
sorry

end trapezoid_area_l320_320279


namespace regular_tetrahedron_properties_l320_320700

structure RegularTetrahedron :=
  (all_edges_equal : ∀ e₁ e₂, e₁ = e₂)
  (angle_between_edges_equal : ∀ v e₁ e₂, angle e₁ e₂ = angle e₂ e₁)
  (all_faces_congruent_equilateral : ∀ f₁ f₂, congruent f₁ f₂)
  (dihedral_angle_between_faces_equal : ∀ f₁ f₂, dihedral_angle f₁ f₂ = dihedral_angle f₂ f₁)

theorem regular_tetrahedron_properties :
  RegularTetrahedron :=
begin
  sorry,
end

end regular_tetrahedron_properties_l320_320700


namespace sin_squared_sum_l320_320274

theorem sin_squared_sum : 
  ∑ k in finset.range 30 \ {0}, (sin (6 * k + 6) * (real.pi / 180))^2 = 15 :=
sorry

end sin_squared_sum_l320_320274


namespace find_a_for_even_function_l320_320363

theorem find_a_for_even_function (a : ℝ) (h : ∀ x : ℝ, x^3 * (a * 2^x - 2^(-x)) = (-x)^3 * (a * 2^(-x) - 2^x)) : a = 1 :=
by
  -- Placeholder for proof
  sorry

end find_a_for_even_function_l320_320363


namespace range_of_g_l320_320696

noncomputable def g (x : ℝ) : ℝ := (sin x) ^ 6 + (cos x) ^ 4

theorem range_of_g :
  {y : ℝ | ∃ x : ℝ, g x = y} = Set.Icc (29 / 27) 1 := 
sorry

end range_of_g_l320_320696


namespace find_investment_period_l320_320219

-- Definitions for the principal amount, interest rate, maturity amount, and compounding frequency
def P : ℝ := 8000
def r : ℝ := 0.05
def A : ℝ := 8820
def n : ℕ := 1

-- The goal to prove:
theorem find_investment_period : ∃ t : ℝ, A = P * (1 + r / n) ^ (n * t) ∧ t ≈ 2 :=
by
  sorry

end find_investment_period_l320_320219


namespace zeroes_of_function_l320_320408

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := |x^2 - 4x| - a

theorem zeroes_of_function (a : ℝ) : (∃ x1 x2 x3 : ℝ, f x1 a = 0 ∧ f x2 a = 0 ∧ f x3 a = 0) → a = 4 := 
sorry

end zeroes_of_function_l320_320408


namespace number_divisible_by_11_l320_320852

theorem number_divisible_by_11 (N Q : ℕ) (h1 : N = 11 * Q) (h2 : Q + N + 11 = 71) : N = 55 :=
by
  sorry

end number_divisible_by_11_l320_320852


namespace repeating_decimal_sum_l320_320735

noncomputable def repeating_decimal_0_3 : ℚ := 1 / 3
noncomputable def repeating_decimal_0_6 : ℚ := 2 / 3
noncomputable def repeating_decimal_0_2 : ℚ := 2 / 9

theorem repeating_decimal_sum :
  repeating_decimal_0_3 + repeating_decimal_0_6 - repeating_decimal_0_2 = 7 / 9 :=
by
  sorry

end repeating_decimal_sum_l320_320735


namespace cookies_in_jar_l320_320123

theorem cookies_in_jar (C : ℕ) (h : C - 1 = (C + 5) / 2) : C = 7 :=
by
  -- Proof goes here
  sorry

end cookies_in_jar_l320_320123


namespace percent_students_in_70_79_range_l320_320862

theorem percent_students_in_70_79_range (students_90_100 students_80_89 students_70_79 students_60_69 students_below_60 : ℕ)
  (h1 : students_90_100 = 7)
  (h2 : students_80_89 = 6)
  (h3 : students_70_79 = 9)
  (h4 : students_60_69 = 4)
  (h5 : students_below_60 = 3) : 
  (students_70_79 : ℝ) / (students_90_100 + students_80_89 + students_70_79 + students_60_69 + students_below_60 : ℝ) * 100 ≈ 31.03 :=
sorry

end percent_students_in_70_79_range_l320_320862


namespace B_work_rate_l320_320187

theorem B_work_rate (A_rate C_rate combined_rate : ℝ) (B_days : ℝ) (hA : A_rate = 1 / 4) (hC : C_rate = 1 / 8) (hCombined : A_rate + 1 / B_days + C_rate = 1 / 2) : B_days = 8 :=
by
  sorry

end B_work_rate_l320_320187


namespace IMO_2001_P1_l320_320909

open BigOperators

noncomputable def S (n : ℕ) (k : Fin n → ℤ) (a : Perm (Fin n)) : ℤ :=
∑ i, k i * ↑(a i)

theorem IMO_2001_P1 (n : ℕ) (k : Fin n → ℤ) (h1 : Odd n) (h2 : 1 < n) :
  ∃ (b c : Perm (Fin n)), b ≠ c ∧ (S n k b - S n k c) % nat.factorial n = 0 := 
sorry

end IMO_2001_P1_l320_320909


namespace find_a_for_even_function_l320_320364

theorem find_a_for_even_function (a : ℝ) (h : ∀ x : ℝ, x^3 * (a * 2^x - 2^(-x)) = (-x)^3 * (a * 2^(-x) - 2^x)) : a = 1 :=
by
  -- Placeholder for proof
  sorry

end find_a_for_even_function_l320_320364


namespace find_p5_l320_320657

-- Conditions defined
def is_cubic_polynomial (p : ℝ → ℝ) : Prop :=
  ∃ a b c d, p = λ x, a * x^3 + b * x^2 + c * x + d

def condition (p : ℝ → ℝ) : Prop :=
  p 1 = 2 / 1^3 ∧ p 2 = 2 / 2^3 ∧ p 3 = 2 / 3^3 ∧ p 4 = 2 / 4^3

-- The mathematical proof problem to be stated in Lean
theorem find_p5 (p : ℝ → ℝ) (h1 : is_cubic_polynomial p) (h2 : condition p) : p 5 = -53 / 300 :=
  sorry

end find_p5_l320_320657


namespace partition_infinite_grid_l320_320439

/-- A grid cell representation -/
structure Cell := (x : Int) (y : Int)

/-- Representation of a 1x2 domino in a grid, specifying two adjacent cells -/
structure Domino := (c1 c2 : Cell) (adj : (c1.x = c2.x ∧ (c1.y = c2.y + 1 ∨ c1.y = c2.y - 1)) ∨ (c1.y = c2.y ∧ (c1.x = c2.x + 1 ∨ c1.x = c2.x - 1)))

/-- Condition stating that any straight line following the grid lines intersects only a finite number of dominoes -/
def finite_intersections (dominoes : Set Domino) :=
  ∀ m : Int, (∃ M : Int, ∀ d ∈ dominoes, d.c1.x ≠ m ∨ d.c2.x ≠ m → abs(d.c1.y - d.c2.y) < M) ∧
             (∃ N : Int, ∀ d ∈ dominoes, d.c1.y ≠ m ∨ d.c2.y ≠ m → abs(d.c1.x - d.c2.x) < N)

/-- The main theorem stating the possibility of partitioning the infinite grid into dominoes with the given condition -/
theorem partition_infinite_grid :
  ∃ (dominoes : Set Domino), finite_intersections dominoes := 
sorry

end partition_infinite_grid_l320_320439


namespace trig_equiv_l320_320067

-- Define the variables and the conditions
variables (α β γ : ℝ)

-- State the theorem
theorem trig_equiv (h : sin (β + γ) * sin (β - γ) = sin α ^ 2) : cos (α + γ) * cos (α - γ) = cos β ^ 2 := 
sorry

end trig_equiv_l320_320067


namespace numbers_masha_thought_l320_320525

noncomputable def distinct_natural_numbers_greater_than_eleven_and_sum_to_satisfy_conditions : ℕ → ℕ → Prop :=
λ a b, a ≠ b ∧ a > 11 ∧ b > 11 ∧ (a + b = 28) ∧ (¬ (∃ x y, x ≠ y ∧ x > 11 ∧ y > 11 ∧ x + y = a + b ∧ (x ≠ a ∧ y ≠ b)))

theorem numbers_masha_thought (a b : ℕ) (h : distinct_natural_numbers_greater_than_eleven_and_sum_to_satisfy_conditions a b) : 
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) :=
by sorry

end numbers_masha_thought_l320_320525


namespace age_difference_is_58_l320_320533

def Milena_age : ℕ := 7
def Grandmother_age : ℕ := 9 * Milena_age
def Grandfather_age : ℕ := Grandmother_age + 2
def Age_difference : ℕ := Grandfather_age - Milena_age

theorem age_difference_is_58 : Age_difference = 58 := by
  sorry

end age_difference_is_58_l320_320533


namespace evaluate_expression_l320_320702

theorem evaluate_expression : 
  (-4)^4 / 4^2 + 2^5 - 7^2 = -1 := by
  have h1 : (-4)^4 = 256 := rfl
  have h2 : 4^2 = 16 := rfl
  have h3 : 2^5 = 32 := rfl
  have h4 : 7^2 = 49 := rfl
  sorry

end evaluate_expression_l320_320702


namespace determine_numbers_l320_320472

theorem determine_numbers (a b : ℕ) (S : ℕ) (h1 : a ≠ b) (h2 : a > 11) (h3 : b > 11)
  (h4 : S = a + b) (h5 : (∀ (x y : ℕ), x + y = S → x ≠ y → (x = a ∧ y = b) ∨ (x = b ∧ y = a)) = false)
  (h6 : a % 2 = 0 ∨ b % 2 = 0) : 
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) := 
  sorry

end determine_numbers_l320_320472


namespace find_b_for_monotonicity_l320_320784

variable (b : ℝ)

def f (x : ℝ) : ℝ := (1/3) * x^3 + b * x^2 + (b + 2) * x + 3

theorem find_b_for_monotonicity :
  (∀ x : ℝ, x^2 + 2 * b * x + (b + 2) ≥ 0) ↔ (-1 ≤ b ∧ b ≤ 2) := by
  sorry

end find_b_for_monotonicity_l320_320784


namespace proof_problem_l320_320660

def is_monotone_incr_on {α β : Type*} [LinearOrder α] [Preorder β] (f : α → β) (s : Set α) :=
  ∀ ⦃a b⦄, a ∈ s → b ∈ s → a < b → f a < f b

noncomputable def f (x : ℝ) : ℝ :=
if x ∈ Set.Icc 0 1 then 2^x - cos x else
if x + 2 ∈ Set.Icc 0 1 then -(2^(x + 2) - cos (x + 2)) else
   sorry -- rest of piecewise definition through periodicity and oddness

theorem proof_problem :
  let x1 := 2018
  let x2 := 2019 / 2
  let x3 := 2020 / 3
in f(x1) < f(x2) ∧ f(x2) < f(x3) :=
by
  have h1 : f x1 = f 0 := sorry
  have h2 : f x2 = f (1/2) := sorry
  have h3 : f x3 = f (7/12) := sorry
  have h_mono : is_monotone_incr_on f (Set.Icc 0 1) := sorry
  have h_order : (0 : ℝ) < (1/2 : ℝ) ∧ (1/2 : ℝ) < (7/12 : ℝ) := by norm_num
  exact ⟨h_mono h_order.left, h_mono h_order.right⟩

end proof_problem_l320_320660


namespace even_divisors_8_factorial_l320_320834

theorem even_divisors_8_factorial : 
  let n := (2^7) * (3^2) * 5 * 7 in
  ∃ (count : ℕ), even_divisors_count n = 84 := 
sorry

end even_divisors_8_factorial_l320_320834


namespace probability_even_sum_l320_320133

open Nat

def balls : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def even_sum_probability : ℚ :=
  let total_outcomes := 12 * 11
  let even_balls := balls.filter (λ n => n % 2 = 0)
  let odd_balls := balls.filter (λ n => n % 2 = 1)
  let even_outcomes := even_balls.length * (even_balls.length - 1)
  let odd_outcomes := odd_balls.length * (odd_balls.length - 1)
  let favorable_outcomes := even_outcomes + odd_outcomes
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ)

theorem probability_even_sum :
  even_sum_probability = 5 / 11 := by
  sorry

end probability_even_sum_l320_320133


namespace prob_two_out_of_three_successes_l320_320209

open ProbabilityTheory

noncomputable def prob_two_successes (p : ℝ) (h0 : 0 ≤ p) (h1 : p ≤ 1) : ℝ :=
  3 * p^2 * (1 - p)

theorem prob_two_out_of_three_successes (p : ℝ) (h0 : 0 ≤ p) (h1 : p ≤ 1) :
  (∑ A in ({ [ tt, tt, ff ], [ tt, ff, tt ], [ ff, tt, tt ] } : finset (vector bool 3)),
    probs_of_independent_events A p) = 3 * p^2 * (1 - p) :=
by
  sorry

-- Helper function to calculate the probability of specific outcomes of independent events.
def probs_of_independent_events (outcomes : vector bool 3) (p : ℝ) : ℝ :=
  (outcomes.to_list.map (λ b, if b then p else 1 - p)).prod

end prob_two_out_of_three_successes_l320_320209


namespace triangle_angle_A_is_80_degrees_l320_320435

theorem triangle_angle_A_is_80_degrees
  (A B C M N : Type)
  [is_triangle A B C]
  (angle_A_is_largest : ∀ x, x ∈ [B, C] → |angle A x| < |angle A|)
  (M_symmetric_wrt_B : is_symmetric_to_angle_bisector A B M)
  (N_symmetric_wrt_C : is_symmetric_to_angle_bisector A C N)
  (h : ∠ M A N = 50) : ∠ A = 80 := 
sorry

end triangle_angle_A_is_80_degrees_l320_320435


namespace number_of_equilateral_triangle_curves_l320_320775

def is_equilateral_triangle (A B C : Point) : Prop := 
  dist A B = dist B C ∧ dist B C = dist C A

def is_equilateral_triangle_curve (A : Point) (T : set Point) : Prop :=
  ∃ B C, B ∈ T ∧ C ∈ T ∧ is_equilateral_triangle A B C

-- Definitions of the given point and curves:
def A : Point := ⟨-1, -1⟩

def curve1 : set Point := {P | let (x, y) := P in x + y - 3 = 0 ∧ 0 ≤ x ∧ x ≤ 3}
def curve2 : set Point := {P | let (x, y) := P in x^2 + y^2 = 2 ∧ - real.sqrt 2 ≤ x ∧ x ≤ 0}
def curve3 : set Point := {P | let (x, y) := P in y = 1/x ∧ 0 < x}

-- Problem statement to prove:
theorem number_of_equilateral_triangle_curves : 
  (1 : ℕ) + (1 : ℕ) = (2 : ℕ) :=
  have h1 : is_equilateral_triangle_curve A curve1, from sorry,
  have h2 : ¬ is_equilateral_triangle_curve A curve2, from sorry,
  have h3 : is_equilateral_triangle_curve A curve3, from sorry,
  show (1 : ℕ) + (1 : ℕ) = (2 : ℕ), from rfl

end number_of_equilateral_triangle_curves_l320_320775


namespace point_B_representation_l320_320060

theorem point_B_representation : 
  ∀ A B : ℤ, A = -3 → B = A + 7 → B = 4 := 
by
  intros A B hA hB
  rw hA at hB
  rw hB
  exact rfl

end point_B_representation_l320_320060


namespace amount_spent_on_marbles_l320_320687

-- Definitions of conditions
def cost_of_football : ℝ := 5.71
def total_spent_on_toys : ℝ := 12.30

-- Theorem statement
theorem amount_spent_on_marbles : (total_spent_on_toys - cost_of_football) = 6.59 :=
by
  sorry

end amount_spent_on_marbles_l320_320687


namespace find_a_for_even_function_l320_320361

theorem find_a_for_even_function (a : ℝ) (h : ∀ x : ℝ, x^3 * (a * 2^x - 2^(-x)) = (-x)^3 * (a * 2^(-x) - 2^x)) : a = 1 :=
by
  -- Placeholder for proof
  sorry

end find_a_for_even_function_l320_320361


namespace inverse_matrix_l320_320607

-- Define the transformation matrix M
def M : Matrix (Fin 2) (Fin 2) ℝ :=
  ![\[⟨(1 / 4 : ℝ), 0⟩, ⟨0, 2⟩]]

-- Define the expected inverse of M
theorem inverse_matrix :
  inverse M = ![\[⟨(4 : ℝ), 0⟩, ⟨0, (1 / 2 : ℝ)⟩]] :=
sorry

end inverse_matrix_l320_320607


namespace root_of_cubic_equation_l320_320317

theorem root_of_cubic_equation : 
  let a := 3 + (11 / 9) * real.sqrt 6
  let b := 3 - (11 / 9) * real.sqrt 6
  let x₀ := real.cbrt a + real.cbrt b
  (x₀ = 2) ∧ (x₀^3 - x₀ - 6 = 0) :=
by
  -- Define constants
  let a := 3 + (11 / 9) * real.sqrt 6
  let b := 3 - (11 / 9) * real.sqrt 6
  let x₀ := real.cbrt a + real.cbrt b
  -- Define the expected conditions
  have h1 : x₀ = 2 := sorry
  have h2 : x₀^3 - x₀ - 6 = 0 := by
    rw h1
    -- Calculation to verify the root
    norm_num
  exact ⟨h1, h2⟩

end root_of_cubic_equation_l320_320317


namespace max_value_of_a_plus_b_l320_320872

noncomputable theory

variables {A B C : ℝ} {a b c : ℝ}

theorem max_value_of_a_plus_b (h1: 0 < A ∧ A < π/2) 
  (h2: 0 < B ∧ B < π/2) 
  (h3: 0 < C ∧ C < π/2) 
  (h4: A + B + C = π) 
  (h5: c = 2) 
  (h6: sqrt 3 * a - 2 * c * sin A = 0) 
  : a + b ≤ 4 :=
begin
  sorry,
end

end max_value_of_a_plus_b_l320_320872


namespace second_train_speed_l320_320215

/-- A train leaves Delhi at 9 a.m. at a speed of 30 kmph. 
    Another train leaves at 2 p.m. on the same day and in the same direction. 
    The two trains meet 600 km away from Delhi. What is the speed of the second train? -/
theorem second_train_speed (distance : ℕ) (first_train_speed : ℕ) (first_train_time : ℕ) 
  (second_train_time : ℕ) (meeting_time_first_train : ℕ) (meeting_time_second_train : ℕ) 
  (h1 : distance = 600) (h2 : first_train_speed = 30) 
  (h3 : first_train_time = 20) (h4 : second_train_time = 15) 
  (h5 : meeting_time_first_train = 20) (h6 : meeting_time_second_train = 15) : 
  (distance / second_train_time = 40) :=
begin
  sorry
end

end second_train_speed_l320_320215


namespace calculate_fg2_calculate_gf2_calculate_ggg_neg2_explicit_g_f_x_explicit_f_g_x_l320_320383

def f (x : ℝ) : ℝ := x^2 - 1

def g (x : ℝ) : ℝ := if x > 0 then x - 1 else 2 - x

theorem calculate_fg2 : f (g 2) = 0 := by
  sorry

theorem calculate_gf2 : g (f 2) = 2 := by
  sorry
  
theorem calculate_ggg_neg2 : g (g (g (-2))) = 2 := by
  sorry

theorem explicit_g_f_x (x : ℝ) : g (f x) = 
  if x < -1 ∨ x > 1 then x^2 - 2 else 4 - x^2 := by
  sorry

theorem explicit_f_g_x (x : ℝ) : f (g x) = 
  if x > 0 then x^2 - 2x else x^2 - 4x + 3 := by
  sorry

end calculate_fg2_calculate_gf2_calculate_ggg_neg2_explicit_g_f_x_explicit_f_g_x_l320_320383


namespace athlete_speed_l320_320694

theorem athlete_speed (d t : ℝ) (H_d : d = 200) (H_t : t = 40) : (d / t) = 5 := by
  sorry

end athlete_speed_l320_320694


namespace domain_of_f_l320_320995

noncomputable def f (x : ℝ) : ℝ := real.sqrt (real.log x) + real.log (5 - 3 * x)

theorem domain_of_f : ∀ x : ℝ, (1 ≤ x ∧ x < 5 / 3) ↔ (0 < x ∧ real.log x ≥ 0 ∧ 5 - 3 * x > 0) := 
by {
  sorry
}

end domain_of_f_l320_320995


namespace horizontal_asymptote_of_f_l320_320731

noncomputable def horizontal_asymptote (f : ℝ → ℝ) (L : ℝ) : Prop :=
  ∀ ε > 0, ∃ M > 0, ∀ x > M, |f x - L| < ε

def f (x : ℝ) : ℝ := (7 * x^2 + 4) / (4 * x^2 + 3 * x - 1)

theorem horizontal_asymptote_of_f :
  horizontal_asymptote f (7 / 4) :=
sorry

end horizontal_asymptote_of_f_l320_320731


namespace num_statements_imply_impl_l320_320714

variable (p q r : Prop)

def cond1 := p ∧ q ∧ ¬r
def cond2 := ¬p ∧ q ∧ r
def cond3 := p ∧ q ∧ r
def cond4 := ¬p ∧ ¬q ∧ ¬r

def impl := ((p → ¬q) → ¬r)

theorem num_statements_imply_impl : 
  (cond1 p q r → impl p q r) ∧ 
  (cond3 p q r → impl p q r) ∧ 
  (cond4 p q r → impl p q r) ∧ 
  ¬(cond2 p q r → impl p q r) :=
by {
  sorry
}

end num_statements_imply_impl_l320_320714


namespace bead_count_l320_320919

variable (total_beads : ℕ) (blue_beads : ℕ) (red_beads : ℕ) (white_beads : ℕ) (silver_beads : ℕ)

theorem bead_count : total_beads = 40 ∧ blue_beads = 5 ∧ red_beads = 2 * blue_beads ∧ white_beads = blue_beads + red_beads ∧ silver_beads = total_beads - (blue_beads + red_beads + white_beads) → silver_beads = 10 :=
by
  intro h
  sorry

end bead_count_l320_320919


namespace ball_travel_distance_fifth_hit_l320_320677

-- Define the distance function for the super ball after n bounces
def ball_distance (n : ℕ) : ℝ :=
  let initial_drop := 200
  let rebound_ratio := 2 / 3
  (range n).sum (λ i => initial_drop * (rebound_ratio ^ i) * 2) + initial_drop

-- Prove that the total distance after the ball hits the ground the fifth time is 4200 feet
theorem ball_travel_distance_fifth_hit :
  ball_distance 5 = 4200 :=
sorry

end ball_travel_distance_fifth_hit_l320_320677


namespace find_a_for_even_function_l320_320367

-- conditions
def f (a : ℝ) (x : ℝ) : ℝ := x^3 * (a * 2^x - 2^(-x))

-- definition of an even function
def is_even_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g x = g (-x)

-- the proof problem statement
theorem find_a_for_even_function (a : ℝ) : is_even_function (f a) ↔ a = 1 :=
by
  sorry

end find_a_for_even_function_l320_320367


namespace even_divisors_of_8fac_l320_320823

theorem even_divisors_of_8fac : 
  let num_even_divisors := ∏ x in {a | 1 ≤ a ∧ a ≤ 7}.card * 
                                      {b | 0 ≤ b ∧ b ≤ 2}.card *
                                      {c | 0 ≤ c ∧ c ≤ 1}.card *
                                      {d | 0 ≤ d ∧ d ≤ 1}.card
  in num_even_divisors = 84 := by
  sorry

end even_divisors_of_8fac_l320_320823


namespace triangle_angle_bisectors_l320_320429

open EuclideanGeometry

theorem triangle_angle_bisectors
  {A B C M P K : Point}
  (hABC: ∠ABC = 120°)
  (hBM: angle_bisector B [A, C] M) 
  (hBCA_ext: external_angle_bisector (∠BCA) [A, B] P)
  (hK: line_intersection (segment MP) (segment BC) K):
  ∠AKM = ∠KPC :=
by
  sorry

end triangle_angle_bisectors_l320_320429


namespace triangle_ABC_AC_l320_320013

-- Defining the relevant points and lengths in the triangle
variables {A B C D : Type} 
variables (AB CD : ℝ)
variables (AD BC AC : ℝ)

-- Given constants
axiom hAB : AB = 3
axiom hCD : CD = Real.sqrt 3
axiom hAD_BC : AD = BC

-- The final theorem statement that needs to be proved
theorem triangle_ABC_AC :
  (AD = BC) ∧ (CD = Real.sqrt 3) ∧ (AB = 3) → AC = Real.sqrt 7 :=
by
  intros h
  sorry

end triangle_ABC_AC_l320_320013


namespace sum_sin_squared_angles_l320_320261

theorem sum_sin_squared_angles : 
  ∑ k in finset.range 30, (sin (6 * (k + 1) * (real.pi / 180)))^2 = 31 / 2 := 
sorry

end sum_sin_squared_angles_l320_320261


namespace mike_hours_per_week_new_job_l320_320052

-- Definitions corresponding to the conditions
variable (h : ℕ) (wage_per_hour : ℕ := 200 / h) -- hourly wage
variable (earnings_seasonal : ℕ := 7200)
variable (weeks_seasonal : ℕ := 36)
variable (earnings_new : ℕ := 3600)
variable (weeks_new : ℕ := 18)

-- Statement to prove
theorem mike_hours_per_week_new_job : 
  (7200 = wage_per_hour * h * 36) → 
  (3600 = wage_per_hour * h * 18) →
  h = 10 := 
by
  intros h_seasonal h_new
  have wage_calc : wage_per_hour = 200 / h := rfl
  sorry

end mike_hours_per_week_new_job_l320_320052


namespace cost_of_paving_theorem_l320_320092

-- Definitions based on problem conditions
def length_of_room := 6 -- in meters
def width_of_room := 4.75 -- in meters
def rate_per_sq_meter := 900 -- in Rs. per sq. meter

-- Calculate the area
def area_of_floor := length_of_room * width_of_room

-- Calculate the cost
def cost_of_paving := area_of_floor * rate_per_sq_meter

-- Theorem stating the final result
theorem cost_of_paving_theorem : cost_of_paving = 25650 := by
  -- Assertion of the expected result
  sorry

end cost_of_paving_theorem_l320_320092


namespace number_of_numbers_is_six_l320_320991

noncomputable def total_numbers (avg_all sum1 sum2 sum3 : ℝ) (cond : avg_all = 2.80 ∧ sum1 = 4.8 ∧ sum2 = 4.6 ∧ sum3 = 7.4) : ℝ :=
  let total_sum := sum1 + sum2 + sum3 in
  total_sum / avg_all

theorem number_of_numbers_is_six
  (avg_all : ℝ) (sum1 : ℝ) (sum2 : ℝ) (sum3 : ℝ)
  (cond : avg_all = 2.80 ∧ sum1 = 4.8 ∧ sum2 = 4.6 ∧ sum3 = 7.4) :
  total_numbers avg_all sum1 sum2 sum3 cond = 6 := 
  sorry

end number_of_numbers_is_six_l320_320991


namespace log_sum_eval_l320_320300

theorem log_sum_eval :
  (Real.logb 5 625 + Real.logb 5 5 - Real.logb 5 (1 / 25)) = 7 :=
by
  have h1 : Real.logb 5 625 = 4 := by sorry
  have h2 : Real.logb 5 5 = 1 := by sorry
  have h3 : Real.logb 5 (1 / 25) = -2 := by sorry
  rw [h1, h2, h3]
  norm_num

end log_sum_eval_l320_320300


namespace sin_squared_sum_l320_320270

theorem sin_squared_sum : 
  ∑ k in finset.range 30 \ {0}, (sin (6 * k + 6) * (real.pi / 180))^2 = 15 :=
sorry

end sin_squared_sum_l320_320270


namespace range_of_a_l320_320769

noncomputable def f (x : ℝ) : ℝ :=
if x ∈ Icc (-1 : ℝ) 0 then x^2
else if x ∈ Icc (-1 : ℝ) (-1 + 2) then f (x - 2)
else if x ∈ Icc 0 1 then f (-x)
else if x ∈ Icc 1 3 then (x - 2)^2
else f (x % 2)

def g (x : ℝ) (a : ℝ) : ℝ :=
f x - Real.log (x + 2) / Real.log a

theorem range_of_a {a : ℝ} (h1 : ∀ x : ℝ, f (x + 2) = f x)
  (h2 : ∀ x : ℝ, f (-x) = f x)
  (h3 : ∀ x ∈ Icc (-1 : ℝ) 0, f x = x^2)
  (h4 : ∃ x ∈ Icc (-1 : ℝ) 3, g x a = 0 ∧ (∃ y ∈ Icc (-1 : ℝ) 3, g y a = 0)) :
  a ∈ Set.Ici 5 :=
sorry

end range_of_a_l320_320769


namespace train_passing_time_correct_l320_320145

def train_length_speed : ℝ × ℝ :=
  (80, 36 * (5 / 18)) -- Train A: 80 meters, 36 km/h converted to m/s

def train_length_speed' : ℝ × ℝ :=
  (120, 45 * (5 / 18)) -- Train B: 120 meters, 45 km/h converted to m/s

def total_passing_time (length_speed1 length_speed2 : ℝ × ℝ) : ℝ :=
  let length1 := length_speed1.1
  let speed1 := length_speed1.2
  let length2 := length_speed2.1
  let speed2 := length_speed2.2
  let relative_speed := speed1 + speed2
  let total_distance := length1 + length2
  total_distance / relative_speed

theorem train_passing_time_correct :
  total_passing_time train_length_speed train_length_speed' ≈ 8.89 := by
  sorry

end train_passing_time_correct_l320_320145


namespace females_in_coach_class_l320_320057

theorem females_in_coach_class (total_passengers : ℕ) (female_percentage : ℝ) 
(sit_in_first_class : ℝ) (male_in_first_class : ℝ)
(h1 : total_passengers = 120)
(h2 : female_percentage = 0.45)
(h3 : sit_in_first_class = 0.10)
(h4 : male_in_first_class = 1/3) : 
  let females_total := (female_percentage * total_passengers : ℕ),
  females_first_class := ((2/3) * (sit_in_first_class * total_passengers) : ℕ) in
  females_total - females_first_class = 46 :=
by
  sorry

end females_in_coach_class_l320_320057


namespace maximize_expected_benefit_l320_320172

-- Definitions
variables (k n : ℕ) (w : ℝ)

-- Conditions
axiom probability_not_detained : ∀ k : ℕ, k > 0 → (1 - (1 / k : ℝ) > 0)

-- Expected benefit function
def expected_benefit (n k : ℕ) (w : ℝ) :=
  (w * n : ℝ) * (1 - (1 / k : ℝ)) ^ n

-- Theorem statement
theorem maximize_expected_benefit (k : ℕ) (w : ℝ) (hkw : w > 0) (hk : k > 0) :
  ∃ n : ℕ, (n = k - 1) ∧ (∀ m : ℕ, expected_benefit m k w ≤ expected_benefit (k - 1) k w) :=
sorry

end maximize_expected_benefit_l320_320172


namespace geometry_problem_l320_320175

theorem geometry_problem
  {α : Type} [EuclideanGeometry α]
  {A B C D E F I R S T : α}
  (triangle_ABC : triangle ABC)
  (circumcircle : is_circumcircle_of triangle_ABC _)
  (incenter : is_incenter_of_triangle I triangle_ABC)
  (AID : ∃ D, (reflect_point_through_line A I triangle_ABC = D) ∧ (D ≠ A) ∧ (is_circumcircle_of (triangle AID) circumcircle))
  (BIE : ∃ E, (reflect_point_through_line B I triangle_ABC = E) ∧ (E ≠ B) ∧ (is_circumcircle_of (triangle BIE) circumcircle))
  (CIF : ∃ F, (reflect_point_through_line C I triangle_ABC = F) ∧ (F ≠ C) ∧ (is_circumcircle_of (triangle CIF) circumcircle))
  (tangent_tangent : ∀ P: α, tangent_to_circle P circumcircle -> 
                    ∃ Q, (intersect_lines_at P circumcircle = Q) ∧ (reflect_point_through_line P I triangle_ABC = Q))
  (tangent_F := tangent_to_circle F circumcircle)
  (tangent_D := tangent_to_circle D circumcircle)
  (tangent_E := tangent_to_circle E circumcircle):

  length_segment A R * length_segment B S * length_segment C T = length_segment I D * length_segment I E * length_segment I F :=
sorry

end geometry_problem_l320_320175


namespace symmetry_center_of_f_l320_320115

noncomputable def f (x : ℝ) : ℝ := (x + 1) / x

theorem symmetry_center_of_f :
  ∃ c : ℝ × ℝ, c = (0, 1) ∧
    (∀ x : ℝ, x ≠ 0 → f (-x) = 2 * (c.2 - f(x)) - f(x)) :=
by
  sorry

end symmetry_center_of_f_l320_320115


namespace min_value_f_l320_320311

-- Define the function f(x)
def f (x : ℝ) : ℝ := (15 - x) * (13 - x) * (15 + x) * (13 + x) + 200 * x^2

-- State the theorem to be proved
theorem min_value_f : ∃ (x : ℝ), (∀ y : ℝ, f y ≥ 33) ∧ f x = 33 := by
  sorry

end min_value_f_l320_320311


namespace masha_numbers_l320_320521

theorem masha_numbers (a b : ℕ) (S : ℕ) (h1 : a ≠ b) (h2 : a > 11) (h3 : b > 11) (h4 : S = a + b) 
    (h5 : (∀ x y : ℕ, x + y = S → x = a ∨ y = a → abs x - y = a) ∧ (even a ∨ even b)) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) := 
by sorry

end masha_numbers_l320_320521


namespace product_of_roots_eq_100_l320_320622

noncomputable def polynomial : Polynomial ℝ := Polynomial.Coeff 3 * X^3 - Polynomial.Coeff 14 * X^2 + Polynomial.Coeff 67 * X - Polynomial.Coeff 100

theorem product_of_roots_eq_100 :
  let roots := Polynomial.roots polynomial in
  let prod := roots.prod in
  prod = (100 : ℝ) :=
by
  sorry

end product_of_roots_eq_100_l320_320622


namespace avg_primes_between_30_and_50_l320_320745

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_30_and_50 : List ℕ := [31, 37, 41, 43, 47]

def sum_primes : ℕ := primes_between_30_and_50.sum

def count_primes : ℕ := primes_between_30_and_50.length

def average_primes : ℚ := (sum_primes : ℚ) / (count_primes : ℚ)

theorem avg_primes_between_30_and_50 : average_primes = 39.8 := by
  sorry

end avg_primes_between_30_and_50_l320_320745


namespace determine_w_arithmetic_seq_l320_320645

theorem determine_w_arithmetic_seq (w : ℝ) (h : (w ≠ 0) ∧ 
  (1 / w - 1 / 2 = 1 / 2 - 1 / 3) ∧ (1 / 2 - 1 / 3 = 1 / 3 - 1 / 6)) :
  w = 3 / 2 := 
sorry

end determine_w_arithmetic_seq_l320_320645


namespace black_cars_count_l320_320106

theorem black_cars_count
    (r b : ℕ)
    (r_ratio : r = 33)
    (ratio_condition : r / b = 3 / 8) :
    b = 88 :=
by 
  sorry

end black_cars_count_l320_320106


namespace find_numbers_l320_320506

theorem find_numbers (a b : ℕ) (h1 : a > 11) (h2 : b > 11) (h3 : a ≠ b)
  (h4 : (∃ S, S = a + b) ∧ (∀ (x y : ℕ), x ≠ y → x + y = a + b → (x > 11) → (y > 11) → ¬(x = a ∨ y = a) → ¬(x = b ∨ y = b)))
  (h5 : even a ∨ even b) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) :=
by
  sorry

end find_numbers_l320_320506


namespace count_polynomials_l320_320452

-- Defining the statement of the proof problem
theorem count_polynomials (n : ℕ) (hn : 0 < n) : 
  let P := {P : ℕ → ℕ // ∃ (k : ℕ), P(2) = n ∧ ∀ (i : ℕ), P i ∈ {0, 1, 2, 3}} in
  P.card = (n / 2) + 1 := 
sorry

end count_polynomials_l320_320452


namespace arithmetic_sequence_sum_l320_320044

theorem arithmetic_sequence_sum (a_n : ℕ → ℤ) (S_n : ℕ → ℤ) (m : ℕ) 
  (h1 : S_n m = 0) (h2 : S_n (m - 1) = -2) (h3 : S_n (m + 1) = 3) :
  m = 5 :=
sorry

end arithmetic_sequence_sum_l320_320044


namespace new_class_mean_l320_320861

theorem new_class_mean (n1 n2 : ℕ) (mean1 mean2 : ℝ) (h1 : n1 = 45) (h2 : n2 = 5) (h3 : mean1 = 0.85) (h4 : mean2 = 0.90) : 
(n1 + n2 = 50) → 
((n1 * mean1 + n2 * mean2) / (n1 + n2) = 0.855) := 
by
  intro total_students
  sorry

end new_class_mean_l320_320861


namespace average_gas_mileage_round_trip_l320_320193

theorem average_gas_mileage_round_trip :
  (300 / ((150 / 28) + (150 / 18))) = 22 := by
sorry

end average_gas_mileage_round_trip_l320_320193


namespace derangement_count_l320_320868

noncomputable def D (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 0
  else n! * ∑ k in Finset.range (n + 1), (-1)^k / k!

theorem derangement_count (n : ℕ) : D n = n! * ∑ k in Finset.range (n + 1), (-1)^k / k! := sorry

end derangement_count_l320_320868


namespace simplify_log_expression_l320_320980

theorem simplify_log_expression : 
  (2 * log 4 3 + log 8 3) * (log 3 2 + log 9 2) = 2 := 
by 
  sorry

end simplify_log_expression_l320_320980


namespace combined_money_l320_320022

variables (S P : ℝ)

noncomputable def Kim_money := 1.12
noncomputable def Kim_to_Sal_relation := 1.4 * S
noncomputable def Sal_to_Phil_relation := 0.8 * P

theorem combined_money (h1 : Kim_money = Kim_to_Sal_relation) (h2 : S = Sal_to_Phil_relation) :
  S + P = 1.80 := 
sorry

end combined_money_l320_320022


namespace part1_part2_l320_320109

noncomputable def seq_a (n : ℕ) : ℝ :=
  if n = 1 then 1 else (1 / 2) ^ (n - 1)

theorem part1 (n : ℕ) (h₀ : n ≥ 1) :
  ∀ n : ℕ, n ≥ 1 → (let S_n := (2^n - 1) * seq_a n in S_n = (2^n - 1) * seq_a n) := 
sorry

noncomputable def seq_b (n : ℕ) : ℝ := n * seq_a n

theorem part2 (n : ℕ) (h₀ : n ≥ 1) :
  let T_n := (finset.range n).sum (λ k, seq_b (k + 1))
  in T_n = 4 - (n + 2) / (2 ^  (n - 1)) :=
sorry

end part1_part2_l320_320109


namespace original_radius_of_cylinder_l320_320728

theorem original_radius_of_cylinder : 
  ∀ (r : ℝ), ∀ (h : ℝ), ∀ (increase : ℝ),
    h = 3 → increase = 7 →
    (π * (r + increase) ^ 2 * h = π * r ^ 2 * (h + increase)) →
    r = 7 := 
by
  intros r h increase h_eq increase_eq volume_eq.
  sorry

end original_radius_of_cylinder_l320_320728


namespace num_even_divisors_of_8_l320_320826

def factorial (n : Nat) : Nat :=
  match n with
  | 0     => 1
  | Nat.succ n' => Nat.succ n' * factorial n'

-- Define the prime factorization of 8!
def prime_factors_eight_factorial : Nat := 2^7 * 3^2 * 5 * 7

-- Definition of an even divisor of 8!
def is_even_divisor (d : Nat) : Prop :=
  d ∣ prime_factors_eight_factorial ∧ 2 ∣ d

-- Calculation of number of even divisors of 8!
def num_even_divisors_8! : Nat :=
  7 * 3 * 2 * 2

theorem num_even_divisors_of_8! :
  num_even_divisors_8! = 84 :=
sorry

end num_even_divisors_of_8_l320_320826


namespace triangle_area_is_3_l320_320010

noncomputable def triangle_area (AB AC : ℝ) (sinA : ℝ) : ℝ :=
  0.5 * AB * AC * sinA
  
theorem triangle_area_is_3 (A B C : Type) [NormedGroup A] [NormedSpace ℝ A]
  (AB AC : A) (cosA : ℝ) (dot_product_AB_AC : ℝ) (sinA : ℝ)
  (h1 : cosA = 4 / 5) (h2 : dot_product_AB_AC = 8) (h3 : sinA = 3 / 5) :
  triangle_area (norm AB) (norm AC) sinA = 3 :=
by
  unfold triangle_area
  sorry

end triangle_area_is_3_l320_320010


namespace solve_speed_l320_320162

noncomputable def speed_of_man_in_still_water (v_m v_s : ℝ) : Prop :=
v_m = 11 ∧ 
  (v_m + v_s) * 3 = 48 ∧
  (v_m - v_s) * 3 = 18

theorem solve_speed : ∃ v_m v_s : ℝ, speed_of_man_in_still_water v_m v_s :=
begin
  use 11,
  have v_s := (48 - 3 * 11) / 3,
  use v_s,
  simp [speed_of_man_in_still_water],
  split; norm_num,
  split;
  ring,
end

end solve_speed_l320_320162


namespace student_score_l320_320212

/-- 
Given that a student's test score is directly proportional to the product of the time spent studying and the difficulty factor,
and given that the student scored 90 points on a test with a difficulty factor of 1.5 after studying for 2 hours,
prove that the student would receive 300 points on a second test of the same format if they studied for 5 hours and the test has a difficulty factor of 2.
-/
theorem student_score 
  (proportional : ∀ (score time difficulty : ℝ), score = time * difficulty * 30)
  (score1 : 90)
  (time1 : 2)
  (difficulty1 : 1.5)
  (score2 : ℝ)
  (time2 : 5)
  (difficulty2 : 2) :
  score2 = 300 :=
by
  have h1 : 90 = 2 * 1.5 * 30, from proportional 90 2 1.5
  have h2 : score2 = 5 * 2 * 30, from proportional score2 5 2
  rw [h2, ← mul_assoc]
  linarith

end student_score_l320_320212


namespace sum_sin_squares_deg_6_to_174_l320_320233

theorem sum_sin_squares_deg_6_to_174 : 
  ∑ k in Finset.range 30, Real.sin (6 * k * (Real.pi / 180)) ^ 2 = 15.5 := by
    sorry

end sum_sin_squares_deg_6_to_174_l320_320233


namespace area_to_paint_l320_320606

def wall_height : ℕ := 10
def wall_length : ℕ := 15
def door_height : ℕ := 3
def door_length : ℕ := 5
def window_height : ℕ := 2
def window_length : ℕ := 3

theorem area_to_paint : (wall_height * wall_length) - (door_height * door_length + window_height * window_length) = 129 := by
  sorry

end area_to_paint_l320_320606


namespace cubic_polynomial_properties_l320_320565

theorem cubic_polynomial_properties
  (a b c r s t : ℝ)
  (h1 : a = -(r + s + t))
  (h2 : b = r * s + s * t + t * r)
  (h3 : r ≥ s)
  (h4 : s ≥ t) :
  let k := a^2 - 3 * b in
  k ≥ 0 ∧ sqrt k ≤ r - t :=
by
  let k := a^2 - 3 * b
  have h5 : k = 1 / 2 * ((r - s)^2 + (s - t)^2 + (r - t)^2),
    sorry
  have h6 : k ≥ 0,
    sorry
  have h7 : sqrt k ≤ r - t,
    sorry
  exact ⟨h6, h7⟩

end cubic_polynomial_properties_l320_320565


namespace solve_equation_l320_320076

theorem solve_equation (x : ℝ) : ((x-3)^2 + 4*x*(x-3) = 0) → (x = 3 ∨ x = 3/5) :=
by
  sorry

end solve_equation_l320_320076


namespace initial_percentage_of_alcohol_solution_l320_320537
open Real

theorem initial_percentage_of_alcohol_solution :
  ∃ P : ℝ, 
    (let initial_volume := 1 in
     let drained_volume := 0.4 in
     let remaining_volume := initial_volume - drained_volume in
     let added_solution_alcohol_percentage := 0.5 in
     let final_solution_alcohol_percentage := 0.65 in
     let total_volume := initial_volume in
     remaining_volume * P + drained_volume * added_solution_alcohol_percentage = total_volume * final_solution_alcohol_percentage) 
    ∧ P = 0.75 :=
by 
  sorry

end initial_percentage_of_alcohol_solution_l320_320537


namespace repeating_sixths_denominator_l320_320566

theorem repeating_sixths_denominator :
  let S := (0.succ_div 1 succ : ℚ) in -- Define S to handle repeating decimals
  S.denom = 3 := 
begin
  let S : ℚ := 2 / 3,
  sorry
end

end repeating_sixths_denominator_l320_320566


namespace sum_sin_squares_deg_6_to_174_l320_320239

theorem sum_sin_squares_deg_6_to_174 : 
  ∑ k in Finset.range 30, Real.sin (6 * k * (Real.pi / 180)) ^ 2 = 15.5 := by
    sorry

end sum_sin_squares_deg_6_to_174_l320_320239


namespace positional_relationship_l320_320404

-- Definitions based on the problem conditions
def is_skew (l1 l2 : Line) : Prop :=
  ¬ (l1 ∥ l2) ∧ ¬ (l1 ∩ l2).nonempty
  
def is_parallel (l1 l2 : Line) : Prop :=
  l1 ∥ l2
  
-- The theorem statement
theorem positional_relationship (l1 l2 l3 : Line) 
  (h1 : is_skew l1 l2) 
  (h2 : is_parallel l3 l1) : 
  (¬ is_parallel l3 l2) ∧ (l3 ∩ l2).nonempty ∨ is_skew l3 l2 :=
sorry

end positional_relationship_l320_320404


namespace exists_minimized_coefficients_l320_320104

-- Definition of f(x) with the assumption it has a quadratic term coefficient of 1 and integer coefficients for linear and constant terms.
def quadratic_function (b c : ℤ) (x : ℚ) := x^2 + b*x + c 

-- Existence of coefficients such that f(f(x)) = 0 has four distinct real roots in arithmetic progression.
theorem exists_minimized_coefficients :
  ∃ b c : ℤ, (∃ roots : list ℚ, roots.length = 4 ∧ ∃ a d : ℚ, 
     roots = [a - d, a, a + d, a + 2*d] ∧
     ∀ x : ℚ, (quadratic_function b c (quadratic_function b c x)) = 0 ↔ x ∈ roots) ∧
     (1 + b + c) = 128 :=
sorry

end exists_minimized_coefficients_l320_320104


namespace tunnel_length_correct_l320_320581

noncomputable def tunnel_length (truck_length : ℝ) (time_to_exit : ℝ) (speed_mph : ℝ) (mile_to_feet : ℝ) : ℝ :=
let speed_fps := (speed_mph * mile_to_feet) / 3600
let total_distance := speed_fps * time_to_exit
total_distance - truck_length

theorem tunnel_length_correct :
  tunnel_length 66 6 45 5280 = 330 :=
by
  sorry

end tunnel_length_correct_l320_320581


namespace floor_25_x_625_l320_320450

theorem floor_25_x_625 : 
  (∀ n ≥ 1, ∃ x : ℕ → ℝ, x 1 = 1 ∧ ∀ n ≥ 1, x (n + 1) = x n + 1 / (2 * x n)) → 
  (floor (25 * (x 625)) = 625) :=
by
  intro h
  sorry

end floor_25_x_625_l320_320450


namespace AQ_to_AC_ratio_l320_320536

variable (A B C D P Q : Type)
variable [Add A] [Add B] [Add C] [Add D] [Add P] [Add Q]
variable [Linear A] [Linear B] [Linear C] [Linear D] [Linear P] [Linear Q]

variable {n : ℕ}
variable {AP AD BP AQ AC : ℝ}
variable {parallelogram : Type}
variable [A_eq : parallelogram (A, B, C, D)]

/-- Given a parallelogram ABCD, a point P on AD such that AP : AD = 1 : n, 
and point Q is the intersection of lines AC and BP, 
prove that AQ : AC = 1 : (n+1) -/
theorem AQ_to_AC_ratio (h1 : ∃ (ABCD : parallelogram) (AP AD BP AQ AC : ℝ), 
  AP / AD = 1 / n) (h2 : ∃ Q, Q = intersection (AC) (BP)) :
  AQ / AC = 1 / (n + 1) :=
sorry

end AQ_to_AC_ratio_l320_320536


namespace michelle_silver_beads_l320_320925

theorem michelle_silver_beads :
  ∀ (total_beads blue_beads red_beads white_beads silver_beads : ℕ),
    total_beads = 40 →
    blue_beads = 5 →
    red_beads = 2 * blue_beads →
    white_beads = blue_beads + red_beads →
    silver_beads = total_beads - (blue_beads + red_beads + white_beads) →
    silver_beads = 10 :=
by {
  intros total_beads blue_beads red_beads white_beads silver_beads,
  assume h1 h2 h3 h4 h5,
  sorry
}

end michelle_silver_beads_l320_320925


namespace max_f_on_interval_l320_320750

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 + Real.sin x

theorem max_f_on_interval : 
  ∃ x ∈ Set.Icc (2 * Real.pi / 5) (3 * Real.pi / 4), f x = (1 + Real.sqrt 2) / 2 :=
by
  sorry

end max_f_on_interval_l320_320750


namespace cookies_in_jar_l320_320127

noncomputable def number_of_cookies_in_jar : ℕ := sorry

theorem cookies_in_jar :
  (number_of_cookies_in_jar - 1) = (1 / 2 : ℝ) * (number_of_cookies_in_jar + 5) →
  number_of_cookies_in_jar = 7 :=
by
  sorry

end cookies_in_jar_l320_320127


namespace closest_point_to_origin_l320_320961

theorem closest_point_to_origin : 
  ∃ x y : ℝ, x > 0 ∧ y = x + 1/x ∧ (x, y) = (1/(2^(1/4)), (1 + 2^(1/2))/(2^(1/4))) :=
by
  sorry

end closest_point_to_origin_l320_320961


namespace sum_of_sin_squared_angles_l320_320253

theorem sum_of_sin_squared_angles :
  (∑ i in Finset.range 30, Real.sin (6 * (i + 1)) * Real.sin (6 * (i + 1))) = 15.5 :=
sorry

end sum_of_sin_squared_angles_l320_320253


namespace product_divisible_by_3_product_residue_1_mod_3_product_residue_2_mod_3_l320_320614

noncomputable def residue_probability_zero (a b : ℕ) : ℚ :=
  if (a % 3 = 0 ∧ b % 3 = 0) ∨ (a % 3 = 0 ∧ b % 3 = 1) ∨ (a % 3 = 0 ∧ b % 3 = 2) ∨ (a % 3 = 1 ∧ b % 3 = 0) ∨ (a % 3 = 2 ∧ b % 3 = 0) then
    5 / 9 else 0

noncomputable def residue_probability_one (a b : ℕ) : ℚ :=
  if (a % 3 = 1 ∧ b % 3 = 1) ∨ (a % 3 = 2 ∧ b % 3 = 2) then
    2 / 9 else 0

noncomputable def residue_probability_two (a b : ℕ) : ℚ :=
  if (a % 3 = 1 ∧ b % 3 = 2) ∨ (a % 3 = 2 ∧ b % 3 = 1) then
    2 / 9 else 0

-- Proof statements for each probability
theorem product_divisible_by_3 (a b : ℕ) :
  (a % 3) * (b % 3) % 3 = 0 → residue_probability_zero a b = 5 / 9 := by
  sorry

theorem product_residue_1_mod_3 (a b : ℕ) :
  (a % 3) * (b % 3) % 3 = 1 → residue_probability_one a b = 2 / 9 := by
  sorry

theorem product_residue_2_mod_3 (a b : ℕ) :
  (a % 3) * (b % 3) % 3 = 2 → residue_probability_two a b = 2 / 9 := by
  sorry

end product_divisible_by_3_product_residue_1_mod_3_product_residue_2_mod_3_l320_320614


namespace masha_numbers_unique_l320_320483

def natural_numbers : Set ℕ := {n | n > 11}

theorem masha_numbers_unique (a b : ℕ) (ha : a ∈ natural_numbers) (hb : b ∈ natural_numbers) (hne : a ≠ b)
  (hs_equals : ∃ S, S = a + b)
  (sasha_initially_uncertain : ∀ (a b : ℕ), a ∈ natural_numbers → b ∈ natural_numbers → a ≠ b → (a + b = S) → ¬ (Sasha_can_determine_initially a b S))
  (masha_hint : ∃ (a_even : ℕ), a_even ∈ natural_numbers ∧ (a_even % 2 = 0) ∧ (a_even = a ∨ a_even = b))
  (sasha_then_confident : ∀ (a b : ℕ), a ∈ natural_numbers → b ∈ natural_numbers → a ≠ b → (a + b = S) → (a_even = a ∨ a_even = b) → Sasha_can_determine_confidently a b S) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) := by
  sorry

end masha_numbers_unique_l320_320483


namespace line_through_point_with_inclination_l320_320746

theorem line_through_point_with_inclination :
  ∃ (l : ℝ → ℝ → Prop), (l (-sqrt 3) 1) ∧ (∀ x y, l x y ↔ sqrt 3 * x + y + 2 = 0) :=
sorry

end line_through_point_with_inclination_l320_320746


namespace probability_even_sum_two_draws_l320_320761

def set_five : Set ℕ := {1, 2, 3, 4, 5}

theorem probability_even_sum_two_draws (s : Set ℕ) (h : s = set_five) :
  (∃ a b ∈ s, a ≠ b ∧ (a + b) % 2 = 0) →
  (∃ p, p = 0.4) :=
by
  -- We will prove this using counting and combinatorics directly.
  sorry

end probability_even_sum_two_draws_l320_320761


namespace numbers_masha_thought_l320_320531

noncomputable def distinct_natural_numbers_greater_than_eleven_and_sum_to_satisfy_conditions : ℕ → ℕ → Prop :=
λ a b, a ≠ b ∧ a > 11 ∧ b > 11 ∧ (a + b = 28) ∧ (¬ (∃ x y, x ≠ y ∧ x > 11 ∧ y > 11 ∧ x + y = a + b ∧ (x ≠ a ∧ y ≠ b)))

theorem numbers_masha_thought (a b : ℕ) (h : distinct_natural_numbers_greater_than_eleven_and_sum_to_satisfy_conditions a b) : 
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) :=
by sorry

end numbers_masha_thought_l320_320531


namespace num_solutions_eq_40_l320_320281

theorem num_solutions_eq_40 : 
  ∀ (n : ℕ), 
  (∃ seq : ℕ → ℕ, seq 1 = 4 ∧ (∀ k : ℕ, 1 ≤ k → seq (k + 1) = seq k + 4) ∧ seq 10 = 40) :=
by
  sorry

end num_solutions_eq_40_l320_320281


namespace sum_of_sin_squared_angles_l320_320248

theorem sum_of_sin_squared_angles :
  (∑ i in Finset.range 30, Real.sin (6 * (i + 1)) * Real.sin (6 * (i + 1))) = 15.5 :=
sorry

end sum_of_sin_squared_angles_l320_320248


namespace product_factors_l320_320705

theorem product_factors : (∏ n in finset.range 11, (1 - (1 : ℝ) / (n + 2))) = 1 / 12 := 
sorry

end product_factors_l320_320705


namespace sum_first_and_seventh_l320_320793

variable {a : ℕ → ℝ}

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d for some d

theorem sum_first_and_seventh (h : arithmetic_sequence a) (h_sum : a 2 + a 3 + a 7 = 6): 
  a 1 + a 7 = 4 :=
sorry

end sum_first_and_seventh_l320_320793


namespace unique_prime_sum_59_l320_320001

theorem unique_prime_sum_59 : ∃! (p q : ℕ), p ≠ q ∧ (prime p ∧ prime q) ∧ p + q = 59 := by
  sorry

end unique_prime_sum_59_l320_320001


namespace closest_point_to_origin_on_graph_l320_320936

theorem closest_point_to_origin_on_graph :
  ∃ x : ℝ, x > 0 ∧ (y = x + 1/x ∧ (x, y) = (1/real.root 4 2, (1 + real.sqrt 2)/real.root 4 2)) := sorry

end closest_point_to_origin_on_graph_l320_320936


namespace smallest_q_p_l320_320902

noncomputable def q_p_difference : ℕ := 3

theorem smallest_q_p (p q : ℕ) (hp : 0 < p) (hq : 0 < q) (h1 : 5 * q < 9 * p) (h2 : 9 * p < 5 * q) : q - p = q_p_difference → q = 7 :=
by
  sorry

end smallest_q_p_l320_320902


namespace parabola_intersection_l320_320577

theorem parabola_intersection (m : ℝ) :
  (∃ p₁ p₂ : ℝ × ℝ, (p₁ = (x, 0) ∧ p₂ = (0, y)) ∧ 
          (∀ t ∈ {p₁, p₂}, y = 2 * x ^ 2 + 8 * x + m) ∧ 
          (p₁ ≠ p₂ ∧ p₁ ≠ p₂)) →
  m = 0 ∨ m = 8 :=
sorry

end parabola_intersection_l320_320577


namespace sum_sn_eq_sum_inv_seq_lt_l320_320043

noncomputable def a_seq : ℕ → ℕ
| 0     := 1
| (n+1) := 2 * a_seq n + 1

def sum_seq (n : ℕ) : ℕ :=
∑ i in Finset.range n, a_seq i

def S_n (n : ℕ) : ℕ := 2^(n+1) - 2 - n

theorem sum_sn_eq : ∀ n : ℕ, sum_seq n = S_n n := by
  intros
  sorry

theorem sum_inv_seq_lt (n : ℕ) (hn : 2 ≤ n) : 
  (∑ i in Finset.range n, (1 / a_seq i : ℝ)) < n := by
  intros
  sorry

end sum_sn_eq_sum_inv_seq_lt_l320_320043


namespace lisa_boxes_sold_l320_320023

noncomputable def boxes_sold_by_lisa (k j l : ℕ) : ℕ :=
  floor((j : ℝ) / 2)

theorem lisa_boxes_sold (k : ℕ) (h1 : k = 54) (j : ℕ) (h2 : j = k + 17) (l : ℕ) (h3 : l = boxes_sold_by_lisa k j l) : l = 35 :=
by
  sorry

end lisa_boxes_sold_l320_320023


namespace circle_intersection_points_l320_320542

theorem circle_intersection_points {Q D : Type} [metric_space Q] [metric_space D]
  (r1 r2 : ℝ) (d : ℝ) (h1 : r1 = 5) (h2 : r2 = 8) (h3 : 3 < d) (h4 : d < 13) :
  ∃ p1 p2 : Q, 2 ≤ set.finite.card {p : Q | dist p (circle_center Q) = r1 ∧ dist p (circle_center D) = r2} :=
by
  sorry

end circle_intersection_points_l320_320542


namespace jim_remaining_miles_l320_320017

theorem jim_remaining_miles:
  (total_miles : ℕ) (miles_driven : ℕ) (remaining_miles : ℕ) 
  (h1 : total_miles = 1200) 
  (h2 : miles_driven = 215) 
  (h3 : remaining_miles = total_miles - miles_driven) :
  remaining_miles = 985 :=
by {
  rw [h1, h2, h3],
  norm_num,
  sorry,  -- since we are just writing the statement and not the proof
}

end jim_remaining_miles_l320_320017


namespace triangle_locus_l320_320712

theorem triangle_locus (A B C M P I : Point) (hIso : distance A B = distance A C)
  (hMid : midpoint M B C) (hInterior : inside P (triangle A B C))
  (hSumAngles : angle B P M + angle C P A = π) :
  P ∈ line A M ∨ (arc (circumcircle (triangle B I C)) (interior (triangle A B C))) :=
sorry

end triangle_locus_l320_320712


namespace conjugate_complex_quadrant_l320_320006

theorem conjugate_complex_quadrant :
  let z := 1 - 1 / (1 - Complex.i)^2
  let z_conj := Complex.conj z
  0 < Complex.re z_conj ∧ 0 < Complex.im z_conj :=
by 
  sorry

end conjugate_complex_quadrant_l320_320006


namespace hawks_score_l320_320297

theorem hawks_score (x y : ℕ) (h1 : x + y = 82) (h2 : x - y = 18) : y = 32 :=
sorry

end hawks_score_l320_320297


namespace sum_of_roots_l320_320579

def quadratic_polynomial (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

def condition (a b c : ℝ) (x : ℝ) :=
  quadratic_polynomial a b c (x^3 + x) ≥ quadratic_polynomial a b c (x^2 + 1)

theorem sum_of_roots (a b c : ℝ) (h : ∀ x : ℝ, condition a b c x) :
  b = -4 * a → -(b / a) = 4 :=
by
  sorry

end sum_of_roots_l320_320579


namespace max_large_numbers_l320_320437

-- Defining the properties of large-right and large-left numbers
def large_right (lst : List ℕ) (n : ℕ) : Prop :=
  ∃ i, lst.get? i = some n ∧ ∀ j, j > i → lst.get? j < some n

def large_left (lst : List ℕ) (n : ℕ) : Prop :=
  ∃ i, lst.get? i = some n ∧ ∀ j, j < i → lst.get? j < some n

-- Main theorem statement
theorem max_large_numbers :
  ∃ (k : ℕ), (∀ lst : List ℕ, lst.length = 100 →
              (k = lst.countp (large_right lst)) ∧
              (k = lst.countp (large_left lst)))
              ∧ k = 50 :=
by
  sorry

end max_large_numbers_l320_320437


namespace probability_product_odd_and_less_than_20_l320_320071

theorem probability_product_odd_and_less_than_20 (balls : Finset ℕ) (h : balls = {1, 2, 3, 4, 5, 6, 7}) :
  (∃ (p : ℚ), p = 15 / 49 ∧
  (∃ b1 b2 ∈ balls, Odd b1 ∧ Odd b2 ∧ (b1 * b2 < 20))) :=
by
  -- Remaining proof here
  sorry

end probability_product_odd_and_less_than_20_l320_320071


namespace age_of_participant_who_left_l320_320893

theorem age_of_participant_who_left
  (avg_age_first_room : ℕ)
  (num_people_first_room : ℕ)
  (avg_age_second_room : ℕ)
  (num_people_second_room : ℕ)
  (increase_in_avg_age : ℕ)
  (total_num_people : ℕ)
  (final_avg_age : ℕ)
  (initial_avg_age : ℕ)
  (sum_ages : ℕ)
  (person_left : ℕ) :
  avg_age_first_room = 20 ∧ 
  num_people_first_room = 8 ∧
  avg_age_second_room = 45 ∧
  num_people_second_room = 12 ∧
  increase_in_avg_age = 1 ∧
  total_num_people = num_people_first_room + num_people_second_room ∧
  final_avg_age = initial_avg_age + increase_in_avg_age ∧
  initial_avg_age = (sum_ages) / total_num_people ∧
  sum_ages = (avg_age_first_room * num_people_first_room + avg_age_second_room * num_people_second_room) ∧
  19 * final_avg_age = sum_ages - person_left
  → person_left = 16 :=
by sorry

end age_of_participant_who_left_l320_320893


namespace findPerimeterRange_l320_320384

noncomputable section

open Real

-- Conditions
def curveEqParametric (α : ℝ) : ℝ × ℝ :=
  (2 * cos α, sqrt 3 * sin α)

def pointC := (3 : ℝ, 2 : ℝ)

-- Definitions
def rectCoordEq (x y : ℝ) : Prop :=
  (x ^ 2 / 4 + y ^ 2 / 3 = 1)

-- Perimeter range of rectangle ABCD
def perimeterRange (α θ : ℝ) : ℝ :=
  let AB : ℝ := 3 - 2 * cos α
  let AD : ℝ := 2 - sqrt 3 * sin α
  2 * (AB + AD)

def angleTheta : ℝ :=
  Real.arctan (sqrt 3 / 2)

-- Theorem
theorem findPerimeterRange (α : ℝ) (h : α = θ + π / 2) :
  10 - 2 * sqrt 7 ≤ perimeterRange α angleTheta ∧
  perimeterRange α angleTheta ≤ 10 + 2 * sqrt 7 := sorry

end findPerimeterRange_l320_320384


namespace positive_integer_solutions_l320_320100

theorem positive_integer_solutions (x : ℕ) (h : 2 * x + 9 ≥ 3 * (x + 2)) : x = 1 ∨ x = 2 ∨ x = 3 :=
by
  sorry

end positive_integer_solutions_l320_320100


namespace sum_infinite_series_l320_320709

noncomputable def series_expression : ℕ → ℚ :=
  λ n, (3 * n + 2) / (n * (n + 1) * (n + 3))

theorem sum_infinite_series :
  ∑' n, series_expression n = -5 / 3 :=
by
  sorry

end sum_infinite_series_l320_320709


namespace minimize_total_cost_l320_320223

noncomputable def event_probability_without_measures : ℚ := 0.3
noncomputable def loss_if_event_occurs : ℚ := 4000000
noncomputable def cost_measure_A : ℚ := 450000
noncomputable def prob_event_not_occurs_measure_A : ℚ := 0.9
noncomputable def cost_measure_B : ℚ := 300000
noncomputable def prob_event_not_occurs_measure_B : ℚ := 0.85

noncomputable def total_cost_no_measures : ℚ :=
  event_probability_without_measures * loss_if_event_occurs

noncomputable def total_cost_measure_A : ℚ :=
  cost_measure_A + (1 - prob_event_not_occurs_measure_A) * loss_if_event_occurs

noncomputable def total_cost_measure_B : ℚ :=
  cost_measure_B + (1 - prob_event_not_occurs_measure_B) * loss_if_event_occurs

noncomputable def total_cost_measures_A_and_B : ℚ :=
  cost_measure_A + cost_measure_B + (1 - prob_event_not_occurs_measure_A) * (1 - prob_event_not_occurs_measure_B) * loss_if_event_occurs

theorem minimize_total_cost :
  min (min total_cost_no_measures total_cost_measure_A) (min total_cost_measure_B total_cost_measures_A_and_B) = total_cost_measures_A_and_B :=
by sorry

end minimize_total_cost_l320_320223


namespace cookies_in_jar_l320_320125

noncomputable def C : ℕ := sorry

theorem cookies_in_jar (h : C - 1 = (C + 5) / 2) : C = 7 := by
  sorry

end cookies_in_jar_l320_320125


namespace sum_of_squares_of_sines_l320_320243

theorem sum_of_squares_of_sines : 
  (\sum n in finset.range 1 30, real.sin (6 * n : ℝ) ^ 2) = 31 / 2 :=
begin
  sorry
end

end sum_of_squares_of_sines_l320_320243


namespace opening_night_ticket_price_l320_320665

theorem opening_night_ticket_price :
  let matinee_customers := 32
  let evening_customers := 40
  let opening_night_customers := 58
  let matinee_price := 5
  let evening_price := 7
  let popcorn_price := 10
  let total_revenue := 1670
  let total_customers := matinee_customers + evening_customers + opening_night_customers
  let popcorn_customers := total_customers / 2
  let total_matinee_revenue := matinee_customers * matinee_price
  let total_evening_revenue := evening_customers * evening_price
  let total_popcorn_revenue := popcorn_customers * popcorn_price
  let known_revenue := total_matinee_revenue + total_evening_revenue + total_popcorn_revenue
  let opening_night_revenue := total_revenue - known_revenue
  let opening_night_price := opening_night_revenue / opening_night_customers
  opening_night_price = 10 := by
  sorry

end opening_night_ticket_price_l320_320665


namespace walking_time_l320_320157

noncomputable def time_to_reach_destination (mr_harris_speed : ℝ) (mr_harris_time_to_store : ℝ) (your_speed : ℝ) (distance_factor : ℝ) : ℝ :=
  let store_distance := mr_harris_speed * mr_harris_time_to_store
  let your_destination_distance := distance_factor * store_distance
  your_destination_distance / your_speed

theorem walking_time (mr_harris_speed your_speed : ℝ) (mr_harris_time_to_store : ℝ) (distance_factor : ℝ) (h_speed : your_speed = 2 * mr_harris_speed) (h_time : mr_harris_time_to_store = 2) (h_factor : distance_factor = 3) :
  time_to_reach_destination mr_harris_speed mr_harris_time_to_store your_speed distance_factor = 3 :=
by
  rw [h_time, h_speed, h_factor]
  -- calculations based on given conditions
  sorry

end walking_time_l320_320157


namespace tan_C_value_l320_320416

theorem tan_C_value (A B C : ℝ)
  (h_cos_A : Real.cos A = 4/5)
  (h_tan_A_minus_B : Real.tan (A - B) = -1/2) :
  Real.tan C = 11/2 :=
sorry

end tan_C_value_l320_320416


namespace cannot_determine_congruence_with_two_acute_angles_l320_320626

/-
  Define right-angled triangles and criteria for congruence.
-/
structure RightAngledTriangle :=
  (a b c : ℝ)
  (h_c : ∠ c = 90)

-- Definitions of conditions A, B, C, D
def condition_A (T₁ T₂ : RightAngledTriangle) := 
  ∠ a T₁ = ∠ a T₂ ∧ ∠ b T₁ = ∠ b T₂

def condition_B (T₁ T₂ : RightAngledTriangle) := 
  (∃s: ℝ, ∃A: ℝ, s ∈ {T₁.a, T₁.b, T₁.c} ∧ s ∈ {T₂.a, T₂.b, T₂.c} ∧
          (A ∈ {∠ a T₁, ∠ b T₁, ∠ c T₁}) ∧ (A ∈ {∠ a T₂, ∠ b T₂, ∠ c T₂}))

def condition_C (T₁ T₂ : RightAngledTriangle) :=
  T₁.a = T₂.a ∧ T₁.b = T₂.b

def condition_D (T₁ T₂ : RightAngledTriangle) :=
  (T₁.a = T₂.a ∨ T₁.b = T₂.b) ∧ T₁.c = T₂.c

-- The theorem to be proven
theorem cannot_determine_congruence_with_two_acute_angles (T₁ T₂ : RightAngledTriangle) :
  ¬ (condition_A T₁ T₂ → congruent T₁ T₂) :=
sorry

end cannot_determine_congruence_with_two_acute_angles_l320_320626


namespace find_a_for_even_function_l320_320368

-- conditions
def f (a : ℝ) (x : ℝ) : ℝ := x^3 * (a * 2^x - 2^(-x))

-- definition of an even function
def is_even_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g x = g (-x)

-- the proof problem statement
theorem find_a_for_even_function (a : ℝ) : is_even_function (f a) ↔ a = 1 :=
by
  sorry

end find_a_for_even_function_l320_320368


namespace abc_sum_l320_320983

theorem abc_sum (a b c : ℤ)
  (h1 : Polynomial.gcd (Polynomial.X^2 + Polynomial.C a * Polynomial.X + Polynomial.C b)
                       (Polynomial.X^2 + Polynomial.C b * Polynomial.X + Polynomial.C c)
      = Polynomial.X - 1)
  (h2 : Polynomial.lcm (Polynomial.X^2 + Polynomial.C a * Polynomial.X + Polynomial.C b)
                       (Polynomial.X^2 + Polynomial.C b * Polynomial.X + Polynomial.C c)
      = Polynomial.X^3 - 6 * Polynomial.X^2 + 11 * Polynomial.X - 6) :
  a + b + c = 8 := 
sorry

end abc_sum_l320_320983


namespace min_value_expression_l320_320912

theorem min_value_expression {x y z w : ℝ} 
  (hx : 0 ≤ x ∧ x ≤ 1) 
  (hy : 0 ≤ y ∧ y ≤ 1) 
  (hz : 0 ≤ z ∧ z ≤ 1) 
  (hw : 0 ≤ w ∧ w ≤ 1) : 
  ∃ m, m = 2 ∧ ∀ x y z w, (0 ≤ x ∧ x ≤ 1) ∧ (0 ≤ y ∧ y ≤ 1) ∧ (0 ≤ z ∧ z ≤ 1) ∧ (0 ≤ w ∧ w ≤ 1) →
  m ≤ (1 / ((1 - x) * (1 - y) * (1 - z) * (1 - w)) + 1 / ((1 + x) * (1 + y) * (1 + z) * (1 + w))) :=
by
  sorry

end min_value_expression_l320_320912


namespace problem_one_problem_two_l320_320232

-- Problem 1
theorem problem_one : -9 + 5 * (-6) - 18 / (-3) = -33 :=
by
  sorry

-- Problem 2
theorem problem_two : ((-3/4) - (5/8) + (9/12)) * (-24) + (-8) / (2/3) = -6 :=
by
  sorry

end problem_one_problem_two_l320_320232


namespace cn_zero_implies_n_zero_l320_320039

theorem cn_zero_implies_n_zero (n : ℕ)
  (a_n b_n c_n : ℤ)
  (H1 : (1 + 4 * real.cbrt 2 - 4 * real.cbrt 4)^n
    = a_n + b_n * real.cbrt 2 + c_n * real.cbrt 4)
  (H2 : c_n = 0) : n = 0 :=
sorry

end cn_zero_implies_n_zero_l320_320039


namespace tangent_line_at_e_l320_320569

noncomputable def f (x : ℝ) : ℝ := Real.log x

theorem tangent_line_at_e :
  ∃ (m b : ℝ), (m = (1 / e)) ∧ (b = 1) ∧ (∀ x y : ℝ, y = m * x + b -> x - e * y = 0) :=
begin
  let m := 1 / e,
  let b := 1,
  use [m, b],
  split,
  { refl, },
  split,
  { refl, },
  intros x y h,
  have h1 : y = (1 / e) * x + 1 := h,
  rw h1,
  ring,
end

end tangent_line_at_e_l320_320569


namespace oplus_value_l320_320907

variable (a b : ℝ)

def tensor (a b : ℝ) : ℝ := (a + b) / (a - b)

def oplus (a b : ℝ) : ℝ := 2 * (tensor a b)

theorem oplus_value : ((oplus (oplus 8 6) 2) = 8 / 3) :=
by 
  sorry

end oplus_value_l320_320907


namespace sam_distance_when_meeting_l320_320760

theorem sam_distance_when_meeting :
  ∃ t : ℝ, (35 = 2 * t + 5 * t) ∧ (5 * t = 25) :=
by
  sorry

end sam_distance_when_meeting_l320_320760


namespace percent_decrease_approx_57_l320_320562

-- Define the costs in the given years
def cost_1995 : ℝ := 35
def cost_2015 : ℝ := 15

-- Define the amount of decrease and percent decrease
def amount_decrease (c_1995 c_2015 : ℝ) : ℝ := c_1995 - c_2015
def percent_decrease (amount_decrease original_amount : ℝ) : ℝ :=
  (amount_decrease / original_amount) * 100

-- Theorem statement proving that the percent decrease is approximately 57%
theorem percent_decrease_approx_57 :
  percent_decrease (amount_decrease cost_1995 cost_2015) cost_1995 ≈ 57 :=
by
  sorry -- Skipping proof as instructed

end percent_decrease_approx_57_l320_320562


namespace find_c_l320_320072

-- Definitions for the problem conditions
def total_area (squares : ℕ) : ℝ := squares * 1
def divided_area (total : ℝ) : ℝ := total / 2
def triangle_area (base height : ℝ) : ℝ := (base * height) / 2

-- Conditions based on the problem
def condition_area_eq_seven : total_area 7 = 7 := rfl
def condition_half_area : divided_area 7 = 3.5 := rfl
def condition_line_eq (c : ℝ) : (4 - c) * 4 / 2 = 3.5 := sorry

-- The statement to prove
theorem find_c (c : ℝ) (h : (4 - c) * 4 / 2 = 3.5) : c = 2.25 := sorry

end find_c_l320_320072


namespace even_divisors_8_factorial_l320_320838

theorem even_divisors_8_factorial : 
  let n := (2^7) * (3^2) * 5 * 7 in
  ∃ (count : ℕ), even_divisors_count n = 84 := 
sorry

end even_divisors_8_factorial_l320_320838


namespace reciprocal_of_one_twentieth_l320_320107

theorem reciprocal_of_one_twentieth : (1 / (1 / 20 : ℝ)) = 20 := 
by
  sorry

end reciprocal_of_one_twentieth_l320_320107


namespace positive_integer_solutions_inequality_l320_320102

theorem positive_integer_solutions_inequality :
  {x : ℕ | 2 * x + 9 ≥ 3 * (x + 2)} = {1, 2, 3} :=
by
  sorry

end positive_integer_solutions_inequality_l320_320102


namespace twenty_percent_less_than_sixty_equals_one_third_more_than_what_number_l320_320136

theorem twenty_percent_less_than_sixty_equals_one_third_more_than_what_number :
  (4 / 3) * n = 48 → n = 36 :=
by
  intro h
  sorry

end twenty_percent_less_than_sixty_equals_one_third_more_than_what_number_l320_320136


namespace f_of_sqrt_e_l320_320374

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then 2 * x - 1 else Real.log x

theorem f_of_sqrt_e : f (f (Real.sqrt Real.exp)) = 0 := by
  sorry

end f_of_sqrt_e_l320_320374


namespace find_n_l320_320005

variable {a_n : ℕ → ℤ} (h1 : a_n 4 = 7) (h2 : a_n 3 + a_n 6 = 16) (h3 : ∃ a d, ∀ n, a_n n = a + (n - 1) * d)

theorem find_n (h : a_n 16 = 31) : ∃ n, a_n n = 31 ∧ n = 16 :=
by
  use 16
  split
  · exact h
  · rfl

end find_n_l320_320005


namespace parity_equivalence_l320_320460

def p_q_parity_condition (p q : ℕ) : Prop :=
  (p^3 - q^3) % 2 = 0 ↔ (p + q) % 2 = 0

theorem parity_equivalence (p q : ℕ) : p_q_parity_condition p q :=
by sorry

end parity_equivalence_l320_320460


namespace maximal_sum_S_l320_320910

theorem maximal_sum_S (x : ℕ → ℝ) (nonneg : ∀ i, 0 ≤ x i)
  (h : ∀ i ∈ Finset.range 100, x i + x (i+1) + x (i+2) ≤ 1) :
  ∑ i in Finset.range 100, x i * x (i+2) ≤ 25 / 2 :=
sorry

end maximal_sum_S_l320_320910


namespace lower_amount_rent_l320_320534

theorem lower_amount_rent (L : ℚ) (total_rent : ℚ) (reduction : ℚ)
  (h1 : total_rent = 2000)
  (h2 : reduction = 200)
  (h3 : 10 * (60 - L) = reduction) :
  L = 40 := by
  sorry

end lower_amount_rent_l320_320534


namespace axis_of_symmetry_l320_320306

theorem axis_of_symmetry (a : ℝ) (h : a ≠ 0) : y = - 1 / (4 * a) :=
sorry

end axis_of_symmetry_l320_320306


namespace correct_statement_l320_320629

theorem correct_statement :
  (¬(x + 2 = 5 ∧ ¬(isExpression x + 2 = 5)) ∧
   ¬((x + yz) / (2*x) ∧ ¬(isMonomial ((x + yz) / (2*x)))) ∧
   (4 * x^2 - 3 * x - 2 ∧ isSum (4 * x^2 - 3 * x - 2)) ∧
   ¬(2 ∧ ¬(isMonomial 2))) → (C) :=
by
  sorry

end correct_statement_l320_320629


namespace p_is_correct_l320_320080

noncomputable def p (x : ℝ) : ℝ := -2 * x^6 + 4 * x^4 + 27 * x^3 + 27 * x^2 + 15 * x + 5

theorem p_is_correct (x : ℝ) :
  p(x) + (2 * x^6 + 4 * x^4 + 6 * x^2) = (8 * x^4 + 27 * x^3 + 33 * x^2 + 15 * x + 5) :=
by
  sorry

end p_is_correct_l320_320080


namespace graph_of_rational_function_quadrants_proof_given_conditions_l320_320405

theorem graph_of_rational_function_quadrants (k : ℝ) (h₀ : k ≠ 0) (h₁ : (1 : ℝ) ≠ 0) :
  (∀ x : ℝ, (x > 0 → k > 0 → 0 < k / x) ∧ (x < 0 → k > 0 → k / x < 0)) :=
by 
  sorry -- Proof omitted.

noncomputable def k_value : ℝ := 4 / 3

theorem proof_given_conditions : k_value ≠ 0 ∧ (∀ x : ℝ, (x > 0 → 0 < k_value / x) ∧ (x < 0 → k_value / x < 0)) := 
by
  split
  · -- proof that k_value ≠ 0
    sorry 
  · -- proof that the graph is in first and third quadrants
    sorry

end graph_of_rational_function_quadrants_proof_given_conditions_l320_320405


namespace find_a_given_factor_condition_l320_320302

theorem find_a_given_factor_condition (a b : ℤ)
  (h : ∃ P : Polynomial ℤ, P * (Polynomial.X ^ 2 - Polynomial.X - 1) = a * Polynomial.X ^ 17 + b * Polynomial.X ^ 16 + 1) :
  a = 987 := by
  sorry

end find_a_given_factor_condition_l320_320302


namespace find_principal_amount_l320_320744

def compound_interest (P r : ℝ) (n t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem find_principal_amount :
  ∃ P : ℝ, 
  let r := 0.04 in
  let n := 2 in
  let t := 1.5 in
  let CI := 302.98 in
  let A := CI + P in
  A = compound_interest P r n t :=
begin
  use 4949.67,
  sorry
end

end find_principal_amount_l320_320744


namespace determine_a_l320_320360

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def f (a : ℝ) (x : ℝ) : ℝ :=
  x^3 * (a * 2^x - 2^(-x))

theorem determine_a : ∃ a : ℝ, is_even_function (f a) ∧ a = 1 :=
by
  use 1
  sorry

end determine_a_l320_320360


namespace solve_system_of_equations_l320_320078

theorem solve_system_of_equations : ∃ (x y : ℝ), 4 * x + y = 6 ∧ 3 * x - y = 1 ∧ x = 1 ∧ y = 2 :=
by
  existsi (1 : ℝ)
  existsi (2 : ℝ)
  sorry

end solve_system_of_equations_l320_320078


namespace closest_point_to_origin_l320_320957

theorem closest_point_to_origin : 
  ∃ x y : ℝ, x > 0 ∧ y = x + 1/x ∧ (x, y) = (1/(2^(1/4)), (1 + 2^(1/2))/(2^(1/4))) :=
by
  sorry

end closest_point_to_origin_l320_320957


namespace light_flashes_in_three_quarters_hour_l320_320200

theorem light_flashes_in_three_quarters_hour (flash_interval seconds_in_three_quarters_hour : ℕ) 
  (h1 : flash_interval = 15) (h2 : seconds_in_three_quarters_hour = 2700) : 
  (seconds_in_three_quarters_hour / flash_interval = 180) :=
by
  sorry

end light_flashes_in_three_quarters_hour_l320_320200


namespace sum_cndn_equals_one_tenth_l320_320900

noncomputable def sequence_satisfies (c d : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, (3 + complex.I)^n = c n + d n * complex.I

noncomputable def sum_expression (c d : ℕ → ℝ) :=
  ∑' n, (c n * d n) / (7 : ℝ)^n

theorem sum_cndn_equals_one_tenth 
  (c d : ℕ → ℝ)  
  (h : sequence_satisfies c d) : 
  sum_expression c d = 1 / 10 :=
  sorry

end sum_cndn_equals_one_tenth_l320_320900


namespace oatmeal_raisin_cookies_dozen_l320_320698

/-- Betty & Paige are raising money for their kids' little league team by hosting a bake sale. -/
def betty_dozen_chocolate_chip : ℕ := 4

/-- Betty baked some dozens of oatmeal raisin cookies. -/
def betty_dozen_oatmeal_raisin := sorry -- Placeholder for the variable we are to solve

/-- Betty baked 2 dozen regular brownies. -/
def betty_dozen_regular_brownies : ℕ := 2

/-- Paige baked 6 dozen sugar cookies. -/
def paige_dozen_sugar_cookies : ℕ := 6

/-- Paige baked 3 dozen blondies. -/
def paige_dozen_blondies : ℕ := 3

/-- Paige baked 5 dozen cream cheese swirled brownies. -/
def paige_dozen_cream_cheese_swirled_brownies : ℕ := 5

/-- Total amount of money raised by selling the baked goods. -/
def total_money_raised : ℕ := 432

/-- Assume that cookies are sold at $1.00 each and blondies/brownies are sold at $2.00 each. 
    Calculate the number of dozens of oatmeal raisin cookies baked by Betty. -/
theorem oatmeal_raisin_cookies_dozen : betty_dozen_oatmeal_raisin = 6 := by
  sorry

end oatmeal_raisin_cookies_dozen_l320_320698


namespace A_infinite_l320_320351

noncomputable def f : ℝ → ℝ := sorry

def A : Set ℝ := { a : ℝ | f a > a ^ 2 }

theorem A_infinite
  (h_f_def : ∀ x : ℝ, ∃ y : ℝ, y = f x)
  (h_inequality: ∀ x : ℝ, (f x) ^ 2 ≤ 2 * x ^ 2 * f (x / 2))
  (h_A_nonempty : A ≠ ∅) :
  Set.Infinite A := 
sorry

end A_infinite_l320_320351


namespace numbers_masha_thought_l320_320527

noncomputable def distinct_natural_numbers_greater_than_eleven_and_sum_to_satisfy_conditions : ℕ → ℕ → Prop :=
λ a b, a ≠ b ∧ a > 11 ∧ b > 11 ∧ (a + b = 28) ∧ (¬ (∃ x y, x ≠ y ∧ x > 11 ∧ y > 11 ∧ x + y = a + b ∧ (x ≠ a ∧ y ≠ b)))

theorem numbers_masha_thought (a b : ℕ) (h : distinct_natural_numbers_greater_than_eleven_and_sum_to_satisfy_conditions a b) : 
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) :=
by sorry

end numbers_masha_thought_l320_320527


namespace count_sequences_divisible_by_5_l320_320028

theorem count_sequences_divisible_by_5 (n : ℕ) (h : n > 0) :
  (∃ (s : vector (fin 5) (2 * n)), (∑ i in s, i.val ^ 2) % 5 = 0) = true :=
by
  sorry

end count_sequences_divisible_by_5_l320_320028


namespace probability_above_line_l320_320971

def is_above_line (x y : ℕ) : Prop :=
  x + y > 5

def valid_points : List (ℕ × ℕ) :=
  [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)]

def valid_pairs_above_line : List (ℕ × ℕ) :=
  valid_points.filter (λ p, is_above_line p.1 p.2)

theorem probability_above_line : 
  (valid_pairs_above_line.length : ℚ) / (valid_points.length : ℚ) = 3 / 5 :=
by
  sorry

end probability_above_line_l320_320971


namespace average_age_is_81_l320_320120

noncomputable def average_age_grandmothers (M : ℕ) (x : ℕ) : Prop :=
  let grandfathers := 2 * M,
      combined_age := M * x + grandfathers * (x - 5),
      total_pensioners := M + grandfathers,
      average_age := combined_age / total_pensioners in
  77 < (combined_age : ℝ) / total_pensioners ∧ 
  (combined_age : ℝ) / total_pensioners < 78 ∧
  x ∈ ℤ

theorem average_age_is_81 {M x : ℕ} 
  (h : average_age_grandmothers M x) :
  x = 81 :=
sorry

end average_age_is_81_l320_320120


namespace bisects_AB_and_AC_l320_320090

variables {A B C K L M P Q : Type*}
variables [Incircle α : Geometry.Circle] [Triangle α : Geometry.Triangle]
variables [Touches α : Incircle.touches α K L M] 
variables [Parallel α A L K P : Geometry.Line] [Parallel α A M K Q : Geometry.Line]

theorem bisects_AB_and_AC (A B C K L M P Q : Type*) 
  [Incircle α : Geometry.Circle]
  [Triangle α : Geometry.Triangle]
  [Touches α : Incircle.touches α K L M]
  [Parallel α A L K P : Geometry.Line]
  [Parallel α A M K Q : Geometry.Line]
  : bisects P Q A B ∧ bisects P Q A C :=
sorry

end bisects_AB_and_AC_l320_320090


namespace even_function_a_value_l320_320354

theorem even_function_a_value {f : ℝ → ℝ} (a : ℝ) :
  (∀ x : ℝ, f x = x^3 * (a * 2^x - 2^(-x)) ∧ f x = f (-x)) → a = 1 :=
by
  intros h,
  sorry

end even_function_a_value_l320_320354


namespace certain_number_is_approx_l320_320188

theorem certain_number_is_approx (x : ℝ) (h : 5.4 * x - (0.6 * 10) / 1.2 = 31.000000000000004) :
  x ≈ 6.666666666666667 :=
sorry

end certain_number_is_approx_l320_320188


namespace two_times_left_movements_l320_320617

-- Define conditions
def movement_to_the_right_is_positive : ℤ := 3
def movement_to_the_left_is_negative : ℤ := -3
def times_moved_left : ℕ := 2

-- Define the expected result
def correct_expression := times_moved_left * movement_to_the_left_is_negative

-- Theorem stating the correct representation of the movement
theorem two_times_left_movements (movement_to_the_right_is_positive = 3) 
   (movement_to_the_left_is_negative = -3) 
   (times_moved_left = 2) : 
   correct_expression = -6 := by
   sorry

end two_times_left_movements_l320_320617


namespace hyperbola_eccentricity_l320_320087

theorem hyperbola_eccentricity :
  ∀ (x y : ℝ), (x^2 / 4) - (y^2 / 5) = 1 →
    let a := 2 in
    let b := sqrt 5 in
    let c := sqrt (a^2 + b^2) in
    c / a = 3 / 2 := by
  intros x y h
  let a := 2
  let b := sqrt 5
  let c := sqrt (a^2 + b^2)
  have ha : a^2 = 4 := by sorry  -- Note: proof step skipped here
  have hb : b^2 = 5 := by sorry  -- Note: proof step skipped here
  have hc : c = 3 := by sorry    -- Note: proof step skipped here
  have he : c / a = 3 / 2 := by sorry  -- Note: proof step skipped here
  exact he

end hyperbola_eccentricity_l320_320087


namespace rate_of_interest_is_5_percent_l320_320676

-- Defining the conditions as constants
def simple_interest : ℝ := 4016.25
def principal : ℝ := 16065
def time_period : ℝ := 5

-- Proving that the rate of interest is 5%
theorem rate_of_interest_is_5_percent (R : ℝ) : 
  simple_interest = (principal * R * time_period) / 100 → 
  R = 5 :=
by
  intro h
  sorry

end rate_of_interest_is_5_percent_l320_320676


namespace root_probability_l320_320465

theorem root_probability (a : ℝ) (h : 0 < a ∧ a < 1) : 
  let f (x : ℝ) := x^2 - (2*a + 1)*x + a^2 + 1 in
  ∃ t : ℝ, (∃ x : ℝ, f x = 0) ↔ t = 1 / 4 :=
begin
  -- proof goes here
  sorry
end

end root_probability_l320_320465


namespace range_of_m_l320_320914

theorem range_of_m (f : ℝ → ℝ) (h_decreasing : ∀ x y, x < y → f y ≤ f x) (m : ℝ) (h : f (m-1) > f (2*m-1)) : 0 < m :=
by
  sorry

end range_of_m_l320_320914


namespace williams_probability_at_least_one_correct_l320_320633

theorem williams_probability_at_least_one_correct :
  let p_wrong := (1 / 2 : ℝ)
  let p_all_wrong := p_wrong ^ 3
  let p_at_least_one_right := 1 - p_all_wrong
  p_at_least_one_right = 7 / 8 :=
by
  sorry

end williams_probability_at_least_one_correct_l320_320633


namespace boxes_filled_l320_320656

noncomputable def bags_per_box := 6
noncomputable def balls_per_bag := 8
noncomputable def total_balls := 720

theorem boxes_filled (h1 : balls_per_bag = 8) (h2 : bags_per_box = 6) (h3 : total_balls = 720) :
  (total_balls / balls_per_bag) / bags_per_box = 15 :=
by
  sorry

end boxes_filled_l320_320656


namespace incorrect_statement_l320_320800

def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 2)

theorem incorrect_statement 
  (hA : ∀ x, f (x + Real.pi) = f x)
  (hB : ∀ x ∈ Set.Icc (0:ℝ) (Real.pi / 2), f x < f (x + 0.1))  -- Simplified increasing function
  (hC : f (-3 * Real.pi / 4) = 0)
  (hD : ¬∀ y, f (- (y - 5 * Real.pi / 4)) = f (y + 5 * Real.pi / 4)) :
  ¬(∀ y, f y = f (2 * (5 * Real.pi / 4) - y)) :=
sorry

end incorrect_statement_l320_320800


namespace absented_student_one_day_out_of_thirty_l320_320863

theorem absented_student_one_day_out_of_thirty
  (p_absent : ℝ := 1/30)
  (p_present : ℝ := 29/30) :
  let prob_one_absent_other_present := (p_present * p_absent) + (p_absent * p_present) in
  (prob_one_absent_other_present * 100).round = 6.4 :=
by
  -- The proof would go here
  sorry

end absented_student_one_day_out_of_thirty_l320_320863


namespace B_share_correct_l320_320682

noncomputable def total_profit : ℝ := 19800

noncomputable def investment_ratios (a b c d : ℝ) : Prop :=
  a = 3 * b ∧ b = (2 / 3) * c ∧ c = (1 / 2) * d

noncomputable def durations (dur_A dur_B dur_C dur_D : ℝ) : Prop :=
  dur_A = 6 ∧ dur_B = 9 ∧ dur_C = 12 ∧ dur_D = 4

noncomputable def profit_shares (a b c d dur_A dur_B dur_C dur_D total_profit : ℝ) : Prop :=
  let total_shares := (a * dur_A + b * dur_B + c * dur_C + d * dur_D)
  in total_shares = total_profit

noncomputable def b_total_share (a b c d dur_A dur_B dur_C dur_D : ℝ) :=
  let b_share := b * dur_B
  let b_bonus := 0.10 * b_share
  b_share + b_bonus

theorem B_share_correct (a b c d : ℝ) 
  (h1 : investment_ratios a b c d) 
  (h2 : durations 6 9 12 4) 
  (h3 : profit_shares a b c d 6 9 12 4 total_profit):
  b_total_share a b c d 6 9 12 4 ≈ 3438.95 :=
sorry

end B_share_correct_l320_320682


namespace sin_beta_value_l320_320781

theorem sin_beta_value (α β : ℝ) (h1 : Real.tan α = 1 / 3) (h2 : Real.tan (α + β) = 1 / 2) : 
    Real.sin β = sqrt 2 / 10 ∨ Real.sin β = - sqrt 2 / 10 := 
by sorry

end sin_beta_value_l320_320781


namespace abc_value_l320_320038

theorem abc_value (a b c : ℂ) 
  (h1 : a * b + 5 * b = -20)
  (h2 : b * c + 5 * c = -20)
  (h3 : c * a + 5 * a = -20) : 
  a * b * c = -100 := 
by {
  sorry
}

end abc_value_l320_320038


namespace maximize_f_at_1_5_l320_320152

noncomputable def f (x: ℝ) : ℝ := -3 * x^2 + 9 * x + 5

theorem maximize_f_at_1_5 : ∀ x: ℝ, f 1.5 ≥ f x := by
  sorry

end maximize_f_at_1_5_l320_320152


namespace closest_point_l320_320956

noncomputable def closest_point_to_origin : ℝ × ℝ :=
  let x := (1 : ℝ) / Real.root 2 4 in
  let y := x + 1 / x in
  (x, y)

theorem closest_point (x y : ℝ) (h : y = x + 1 / x) (hx : x > 0) :
  (x, y) = closest_point_to_origin :=
begin
  sorry
end

end closest_point_l320_320956


namespace rational_function_horizontal_asymptote_l320_320282

theorem rational_function_horizontal_asymptote :
  ∃ p : Polynomial ℝ, (∀ (x : ℝ), p.degree ≤ 6) ∧ 
                      (Polynomial.leadingCoeff p = 3) ∧ 
                      (∀ (x : ℝ), (limit (λ x, (p.eval x) / (3 * x^6 - 2 * x^5 + 4 * x^3 - x + 2)) → 1)) :=
by
  sorry

end rational_function_horizontal_asymptote_l320_320282


namespace sin_alpha_trig_expression_l320_320790

theorem sin_alpha {α : ℝ} (hα : ∃ P : ℝ × ℝ, P = (4/5, -3/5)) :
  Real.sin α = -3/5 :=
sorry

theorem trig_expression {α : ℝ} 
  (hα : Real.sin α = -3/5) : 
  (Real.sin (π / 2 - α) / Real.sin (α + π)) - 
  (Real.tan (α - π) / Real.cos (3 * π - α)) = 19 / 48 :=
sorry

end sin_alpha_trig_expression_l320_320790


namespace distribution_methods_l320_320876

theorem distribution_methods (n m k : Nat) (h : n = 23) (h1 : m = 10) (h2 : k = 2) :
  (∃ d : Nat, d = Nat.choose m 1 + 2 * Nat.choose m 2 + Nat.choose m 3) →
  ∃ x : Nat, x = 220 :=
by
  sorry

end distribution_methods_l320_320876


namespace sum_of_possible_students_l320_320207

-- Definitions based on the conditions
def num_students (s : ℕ) : Prop :=
  210 ≤ s ∧ s ≤ 300 ∧ (s - 3) % 8 = 0

-- The Lean statement
theorem sum_of_possible_students : 
  (Finset.sum (Finset.filter num_students (Finset.range 301))) = 3000 :=
by
  sorry

end sum_of_possible_students_l320_320207


namespace race_completion_time_l320_320420

-- Definitions used in the problem
variable (T_A T_B : ℝ) -- Time taken by A and B
variable (V_A V_B : ℝ) -- Speeds of A and B

-- Conditions
variable cond1 : T_B = T_A + 10
variable cond2 : 1000 / T_A = 800 / T_B

-- The theorem to prove
theorem race_completion_time : T_A = 50 :=
by
  -- sorry for the placeholder proof; actual proof omitted.
  sorry

end race_completion_time_l320_320420


namespace prime_p_perfect_cube_l320_320751

theorem prime_p_perfect_cube (p : ℕ) (hp : Nat.Prime p) (h : ∃ n : ℕ, 13 * p + 1 = n^3) :
  p = 2 ∨ p = 211 :=
by
  sorry

end prime_p_perfect_cube_l320_320751


namespace max_three_statements_true_l320_320911

theorem max_three_statements_true (c d : ℝ) :
  ¬(∃ (c d : ℝ) (h1 : c > 0) (h2 : d > 0), (h3 : c > d) ∧ 
    (¬((1/c) > (1/d)) ∧ ¬(c^2 < d^2) ∧ (c > d) ∧ (c > 0) ∧ (d > 0))) :=
sorry

end max_three_statements_true_l320_320911


namespace sum_of_sin_squared_angles_l320_320252

theorem sum_of_sin_squared_angles :
  (∑ i in Finset.range 30, Real.sin (6 * (i + 1)) * Real.sin (6 * (i + 1))) = 15.5 :=
sorry

end sum_of_sin_squared_angles_l320_320252


namespace jo_bob_pulled_chain_first_time_l320_320442

/-- Given the conditions of the balloon ride, prove that Jo-Bob pulled the chain
    for the first time for 15 minutes. --/
theorem jo_bob_pulled_chain_first_time (x : ℕ) : 
  (50 * x - 100 + 750 = 1400) → (x = 15) :=
by
  intro h
  sorry

end jo_bob_pulled_chain_first_time_l320_320442


namespace union_sets_M_N_l320_320809

-- Define the sets M and N
def M : Set ℝ := {x | x > 1}
def N : Set ℝ := {x | -3 < x ∧ x < 2}

-- The proof statement: the union of M and N should be x > -3
theorem union_sets_M_N : (M ∪ N) = {x | x > -3} :=
sorry

end union_sets_M_N_l320_320809


namespace sin_squared_sum_l320_320268

theorem sin_squared_sum : 
  ∑ k in finset.range 30 \ {0}, (sin (6 * k + 6) * (real.pi / 180))^2 = 15 :=
sorry

end sin_squared_sum_l320_320268


namespace angle_A_range_cos_expr_l320_320418

variables (A B C a b c : ℝ) (triangle_ABC : a > 0 ∧ b > 0 ∧ c > 0 ∧ A + B + C = π)
variable (h : sqrt 3 * a * cos C = (2 * b - sqrt 3 * c) * cos A)

theorem angle_A (h : sqrt 3 * a * cos C = (2 * b - sqrt 3 * c) * cos A) : A = π/6 :=
sorry

theorem range_cos_expr (hA : A = π/6) : -((sqrt 3 + 2) / 2) < cos (5 * π / 2 - B) - 2 * sin (C / 2) ^ 2 ∧ cos (5 * π / 2 - B) - 2 * sin (C / 2) ^ 2 ≤ sqrt 3 - 1 :=
sorry

end angle_A_range_cos_expr_l320_320418


namespace numbers_masha_thought_l320_320532

noncomputable def distinct_natural_numbers_greater_than_eleven_and_sum_to_satisfy_conditions : ℕ → ℕ → Prop :=
λ a b, a ≠ b ∧ a > 11 ∧ b > 11 ∧ (a + b = 28) ∧ (¬ (∃ x y, x ≠ y ∧ x > 11 ∧ y > 11 ∧ x + y = a + b ∧ (x ≠ a ∧ y ≠ b)))

theorem numbers_masha_thought (a b : ℕ) (h : distinct_natural_numbers_greater_than_eleven_and_sum_to_satisfy_conditions a b) : 
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) :=
by sorry

end numbers_masha_thought_l320_320532


namespace sum_of_sin_squared_angles_l320_320251

theorem sum_of_sin_squared_angles :
  (∑ i in Finset.range 30, Real.sin (6 * (i + 1)) * Real.sin (6 * (i + 1))) = 15.5 :=
sorry

end sum_of_sin_squared_angles_l320_320251


namespace Masha_thought_of_numbers_l320_320515

theorem Masha_thought_of_numbers : ∃ a b : ℕ, a ≠ b ∧ a > 11 ∧ b > 11 ∧ (a + b = 28) ∧ (a % 2 = 0 ∨ b % 2 = 0) ∧ (a = 12 ∧ b = 16 ∨ a = 16 ∧ b = 12) :=
by
  sorry

end Masha_thought_of_numbers_l320_320515


namespace max_pencils_l320_320856

def pencil_price : ℕ := 25
def total_cents : ℕ := 5000

theorem max_pencils : ∃ n, n = 200 ∧ pencil_price * n ≤ total_cents :=
by
  existsi 200
  split
  . rfl -- This states that n = 200
  . sorry -- This is the actual inequality proof that 25 * 200 ≤ 5000

end max_pencils_l320_320856


namespace scalene_triangle_area_l320_320711

theorem scalene_triangle_area (outer_triangle_area : ℝ) (hexagon_area : ℝ) (num_scalene_triangles : ℕ)
  (h1 : outer_triangle_area = 25) (h2 : hexagon_area = 4) (h3 : num_scalene_triangles = 6) : 
  (outer_triangle_area - hexagon_area) / num_scalene_triangles = 3.5 :=
by
  sorry

end scalene_triangle_area_l320_320711


namespace angle_AB_BC_l320_320814

noncomputable def point := ℝ × ℝ

def A : point := (3, 0)
def B : point := (2, 2)
def C : point := (5, -4)

def vector (p1 p2 : point) : point := (p2.1 - p1.1, p2.2 - p1.2)

def angle_between_vectors (v1 v2 : point) : ℝ := 
  let dot_product := v1.1 * v2.1 + v1.2 * v2.2
  let norm_v1 := real.sqrt (v1.1 ^ 2 + v1.2 ^ 2)
  let norm_v2 := real.sqrt (v2.1 ^ 2 + v2.2 ^ 2)
  real.acos (dot_product / (norm_v1 * norm_v2))

theorem angle_AB_BC : 
  angle_between_vectors (vector A B) (vector B C) = real.pi :=
by
  unfold vector angle_between_vectors
  simp
  sorry

end angle_AB_BC_l320_320814


namespace jane_crayon_count_l320_320699

def billy_crayons : ℝ := 62.0
def total_crayons : ℝ := 114
def jane_crayons : ℝ := total_crayons - billy_crayons

theorem jane_crayon_count : jane_crayons = 52 := by
  unfold jane_crayons
  show total_crayons - billy_crayons = 52
  sorry

end jane_crayon_count_l320_320699


namespace bowling_ball_weight_l320_320930

-- Definitions for the conditions
def kayak_weight : ℕ := 36
def total_weight_of_two_kayaks := 2 * kayak_weight
def total_weight_of_nine_bowling_balls (ball_weight : ℕ) := 9 * ball_weight  

theorem bowling_ball_weight (w : ℕ) (h1 : total_weight_of_two_kayaks = total_weight_of_nine_bowling_balls w) : w = 8 :=
by
  -- Proof goes here
  sorry

end bowling_ball_weight_l320_320930


namespace masha_numbers_l320_320490

theorem masha_numbers {a b : ℕ} (h1 : a ≠ b) (h2 : 11 < a) (h3 : 11 < b) 
  (h4 : ∃ S, S = a + b ∧ (∀ x y, x + y = S → x ≠ y → 11 < x ∧ 11 < y → 
       (¬(x = a ∧ y = b) ∧ ¬(x = b ∧ y = a)))) 
  (h5 : even a ∨ even b)
  (h6 : ∀ x y, (even x ∨ even y) → x ≠ y → 11 < x ∧ 11 < y ∧ x + y = a + b → 
       x = a ∧ y = b ∨ x = b ∧ y = a) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) :=
by
  sorry

end masha_numbers_l320_320490


namespace mod_equiv_pow_five_l320_320982

theorem mod_equiv_pow_five (m : ℤ) (hm : 0 ≤ m ∧ m < 11) (h : 12^5 ≡ m [ZMOD 11]) : m = 1 :=
by
  sorry

end mod_equiv_pow_five_l320_320982


namespace range_of_a_l320_320380

noncomputable def is_monotonically_increasing (f : ℝ → ℝ) (I : set ℝ) :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x ≤ y → f x ≤ f y

theorem range_of_a (a : ℝ) :
  is_monotonically_increasing (λ x : ℝ, x^2 + a * real.log x) {x : ℝ | 1 < x} ↔ a >= -2 :=
by {
  sorry
}

end range_of_a_l320_320380


namespace find_s_l320_320611

open Real

theorem find_s (P Q R V W : Point) (s : ℝ)
  (hP : P = ⟨0, 10⟩)
  (hQ : Q = ⟨3, 0⟩)
  (hR : R = ⟨10, 0⟩)
  (hline_PQ : ∀ x, y = - 10 / 3 * x + 10)
  (hline_PR : ∀ x, y = - x + 10)
  (hV : V = ⟨3 - 3 * s / 10, s⟩)
  (hW : W = ⟨10 - s, s⟩)
  (h_area : 1 / 2 * (7 / 10 * (10 - s)) * (10 - s) = 18) :
  s = 10 - sqrt 51.429 :=
sorry

end find_s_l320_320611


namespace smallest_value_of_y_square_l320_320041

-- Let's define the conditions
variable (EF GH y : ℝ)

-- The given conditions of the problem
def is_isosceles_trapezoid (EF GH y : ℝ) : Prop :=
  EF = 100 ∧ GH = 25 ∧ y > 0

def has_tangent_circle (EF GH y : ℝ) : Prop :=
  is_isosceles_trapezoid EF GH y ∧ 
  ∃ P : ℝ, P = EF / 2

-- Main proof statement
theorem smallest_value_of_y_square (EF GH y : ℝ)
  (h1 : is_isosceles_trapezoid EF GH y)
  (h2 : has_tangent_circle EF GH y) :
  y^2 = 1875 :=
  sorry

end smallest_value_of_y_square_l320_320041


namespace number_of_even_divisors_of_factorial_eight_l320_320830

-- Definition of 8! and its prime factorization
def factorial_eight : ℕ := 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1
def prime_factorization_factorial_eight : Prop :=
  factorial_eight = 2^7 * 3^2 * 5 * 7

-- The main theorem statement
theorem number_of_even_divisors_of_factorial_eight :
  prime_factorization_factorial_eight →
  ∃ n, n = 7 * 3 * 2 * 2 ∧
  (∀ d, d ∣ factorial_eight → (∃ a b c d, 1 ≤ a ∧ a ≤ 7 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 1 ∧ 0 ≤ d ∧ d ≤ 1 ∧ d = 2^a * 3^b * 5^c * 7^d) →
  (7 * 3 * 2 * 2 = n)) :=
by
  intro h
  use 84
  sorry

end number_of_even_divisors_of_factorial_eight_l320_320830


namespace max_area_triangle_l320_320377

open Real

-- Definitions and hypotheses
def f (x : ℝ) : ℝ := 2 * sqrt 3 * sin x * cos x - 2 * (cos x)^2

example : ∀ x, -3 ≤ f x ∧ f x ≤ 1 :=
by
  sorry

theorem max_area_triangle (A : ℝ) (a : ℝ) (fA : f A = -2) (h_a : a = sqrt 3) :
  ∃ (b c : ℝ), let S := (b * c * sin A) / 2 in S = sqrt 3 / 4 :=
by
  sorry

end max_area_triangle_l320_320377


namespace friends_room_assignment_l320_320654

theorem friends_room_assignment :
  (∑ n in (finset.range 7), if n == 0 then 0 else if n == 1 then 0 else if n == 2 then 15 * 24 / 2 else if n == 3 then 15 * 24 / 2 else if n == 4 then 20 * 6 * 120 * 6 else 15 * 1 * 360) = 49320 := 
by sorry

end friends_room_assignment_l320_320654


namespace necessary_and_sufficient_condition_l320_320884

theorem necessary_and_sufficient_condition (A B : ℝ) (h : A ≠ B) :
  cos (2 * A) ≠ cos (2 * B) ↔ A ≠ B :=
by {
  sorry
}

end necessary_and_sufficient_condition_l320_320884


namespace vector_dot_product_l320_320417

variables (A B C : Type) [InnerProductSpace ℝ A] 

theorem vector_dot_product :
  let AB := (0 : A) - (1 : A)
  let AC := (0 : A) - (2 : A)
  (abs (AB) = 4) ∧ (abs (AC) = 1) ∧ (∥AB ∥ ∥ × ∥AC∥) / 2 = sqrt 3 →
  (innerProductSpace.dot AB AC = 2) ∨ (innerProductSpace.dot AB AC = -2) :=
sorry

end vector_dot_product_l320_320417


namespace right_triangle_sides_l320_320216

theorem right_triangle_sides (a d : ℝ) (k : ℕ) (h_pos_a : 0 < a) (h_pos_d : 0 < d) (h_pos_k : 0 < k) :
  (a = 3) ∧ (d = 1) ∧ (k = 2) ↔ (a^2 + (a + d)^2 = (a + k * d)^2) :=
by 
  sorry

end right_triangle_sides_l320_320216


namespace max_length_sequence_l320_320461

-- Define the sequence and its properties
variables {n k : ℕ}
variables (a : ℕ → ℕ)

-- The sequence must consist of positive integers not exceeding n
def valid_sequence (a : ℕ → ℕ) (n : ℕ) :=
  ∀ i, a i ≤ n ∧ 1 ≤ a i

-- Adjacent terms of the sequence are distinct
def no_adjacent_terms_equal (a : ℕ → ℕ) :=
  ∀ i, a i ≠ a (i + 1)

-- There do not exist indices p < q < r < s such that a_p = a_r ≠ a_q = a_s
def no_four_indices (a : ℕ → ℕ) :=
  ∀ p q r s, p < q < r < s → a p = a r → a q = a s → a p ≠ a q

-- State the problem: Prove the maximum value of k
theorem max_length_sequence {a : ℕ → ℕ} (valid : valid_sequence a n)
    (no_adj : no_adjacent_terms_equal a) (no_four : no_four_indices a) :
    k = 4 * n - 2 :=
sorry

end max_length_sequence_l320_320461


namespace marble_cost_l320_320690

def AlyssaSpentOnMarbles (totalSpent onToys footballCost : ℝ) : ℝ :=
 totalSpent - footballCost

theorem marble_cost:
  AlyssaSpentOnMarbles 12.30 5.71 = 6.59 :=
by 
  unfold AlyssaSpentOnMarbles 
  sorry

end marble_cost_l320_320690


namespace apple_distribution_l320_320685

theorem apple_distribution : 
  ∃ (a b c : ℕ), 
    a + b + c = 30 ∧ a ≥ 3 ∧ b ≥ 3 ∧ c ≥ 3 ∧ 
    (∑ x in {a' | ∃ (a b c: ℕ), a' = a - 3 ∧ b = b' + 3 ∧ c = c' + 3 ∧ a' + b' + c' = 21}, 1) = 253 := 
sorry

end apple_distribution_l320_320685


namespace sarah_numbers_sum_l320_320969

-- Definition of x and y being integers with their respective ranges
def isTwoDigit (x : ℕ) : Prop := 10 ≤ x ∧ x ≤ 99
def isThreeDigit (y : ℕ) : Prop := 100 ≤ y ∧ y ≤ 999

-- The condition relating x and y
def formedNumber (x y : ℕ) : Prop := 1000 * x + y = 7 * x * y

-- The Lean 4 statement for the proof problem
theorem sarah_numbers_sum (x y : ℕ) (H1 : isTwoDigit x) (H2 : isThreeDigit y) (H3 : formedNumber x y) : x + y = 1074 :=
  sorry

end sarah_numbers_sum_l320_320969


namespace bridge_length_l320_320202

   noncomputable def walking_speed_km_per_hr : ℝ := 6
   noncomputable def walking_time_minutes : ℝ := 15

   noncomputable def length_of_bridge (speed_km_per_hr : ℝ) (time_min : ℝ) : ℝ :=
     (speed_km_per_hr * 1000 / 60) * time_min

   theorem bridge_length :
     length_of_bridge walking_speed_km_per_hr walking_time_minutes = 1500 := 
   by
     sorry
   
end bridge_length_l320_320202


namespace masha_numbers_unique_l320_320479

def natural_numbers : Set ℕ := {n | n > 11}

theorem masha_numbers_unique (a b : ℕ) (ha : a ∈ natural_numbers) (hb : b ∈ natural_numbers) (hne : a ≠ b)
  (hs_equals : ∃ S, S = a + b)
  (sasha_initially_uncertain : ∀ (a b : ℕ), a ∈ natural_numbers → b ∈ natural_numbers → a ≠ b → (a + b = S) → ¬ (Sasha_can_determine_initially a b S))
  (masha_hint : ∃ (a_even : ℕ), a_even ∈ natural_numbers ∧ (a_even % 2 = 0) ∧ (a_even = a ∨ a_even = b))
  (sasha_then_confident : ∀ (a b : ℕ), a ∈ natural_numbers → b ∈ natural_numbers → a ≠ b → (a + b = S) → (a_even = a ∨ a_even = b) → Sasha_can_determine_confidently a b S) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) := by
  sorry

end masha_numbers_unique_l320_320479


namespace fraction_area_outside_circle_l320_320873

-- Use noncomputable theory as necessary for constructs like real numbers and π
noncomputable theory

-- Define the equilateral triangle and its properties
def area_outside_circle (r : ℝ) : ℝ :=
  let s := 2 * r in 
  let area_triangle := (s * s * real.sqrt 3) / 4 in
  let area_sector := (real.pi * r * r) / 6 in
  let area_triangle_segment := (r * r * real.sqrt 3) / 2 in
  let area_segment := area_sector - area_triangle_segment in
  let total_area_outside := area_triangle - 2 * area_segment in
  total_area_outside / area_triangle

-- The theorem to be proved with the correct fraction value
theorem fraction_area_outside_circle (r : ℝ) : area_outside_circle r = (4 / 3 - real.pi / (3 * real.sqrt 3)) := 
by 
  sorry -- Proof to be provided

end fraction_area_outside_circle_l320_320873


namespace chess_tournament_l320_320596

def number_of_players := 30

def total_games (n : ℕ) : ℕ := n * (n - 1) / 2

theorem chess_tournament : total_games number_of_players = 435 := by
  sorry

end chess_tournament_l320_320596


namespace range_of_a_for_decreasing_log_function_l320_320381

noncomputable def function_decreasing_on_interval (a : ℝ) (f : ℝ → ℝ) (interval : set ℝ) : Prop :=
  ∀ x y ∈ interval, x < y → f y < f x

theorem range_of_a_for_decreasing_log_function :
  ∀ a : ℝ, (a > 0) ∧ (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → (2 - a * x) > 0) →
    function_decreasing_on_interval a (λ x, log a (2 - a * x)) (set.Icc 0 1) →
    1 < a ∧ a < 2 :=
begin
  sorry
end

end range_of_a_for_decreasing_log_function_l320_320381


namespace hexagon_shaded_area_fraction_l320_320541

/-- A regular hexagon ABCDEF with center O and midpoint Y of side AB
     What fraction of the area of the hexagon is shaded? -/
theorem hexagon_shaded_area_fraction (O A B C D E F Y : Point) 
  (eq_side : ∀ (X1 X2 : Point), X1 ∈ {A, B, C, D, E, F} → X2 ∈ {A, B, C, D, E, F} → dist X1 O = dist X2 O)
  (mid_Y_AB : dist A Y = dist Y B ∧ collinear {A, Y, B}) :
  shadedRegionFraction O A B C D E F Y = 5 / 12 :=
sorry

end hexagon_shaded_area_fraction_l320_320541


namespace cookies_in_jar_l320_320126

noncomputable def C : ℕ := sorry

theorem cookies_in_jar (h : C - 1 = (C + 5) / 2) : C = 7 := by
  sorry

end cookies_in_jar_l320_320126


namespace not_exactly_1000_good_numbers_in_first_2000_l320_320222

def good_number (n : ℕ) : Prop := sorry
def bad_number (n : ℕ) : Prop := sorry

axiom good_if_six_add (m : ℕ) (hm : good_number m) : good_number (m + 6)
axiom bad_if_fifteen_add (n : ℕ) (hn : bad_number n) : bad_number (n + 15)
axiom classify_number (n : ℕ) : good_number n ∨ bad_number n

theorem not_exactly_1000_good_numbers_in_first_2000 : 
    ¬ (finset.card ((finset.range 2000).filter good_number) = 1000) := by
  sorry

end not_exactly_1000_good_numbers_in_first_2000_l320_320222


namespace customers_who_didnt_tip_l320_320697

theorem customers_who_didnt_tip:
  ∀ (total_customers tips_per_customer total_tips : ℕ),
  total_customers = 10 →
  tips_per_customer = 3 →
  total_tips = 15 →
  (total_customers - total_tips / tips_per_customer) = 5 :=
by
  intros
  sorry

end customers_who_didnt_tip_l320_320697


namespace deal_cost_l320_320666

theorem deal_cost : 
  ∀ (ticket popcorn drink candy deal_savings : ℝ), 
    ticket = 8 ∧ 
    popcorn = ticket - 3 ∧
    drink = popcorn + 1 ∧
    candy = drink / 2 ∧
    deal_savings = 2 →
    (ticket + popcorn + drink + candy) - deal_savings = 20 :=
by
  intros ticket popcorn drink candy deal_savings h
  cases h with ht hpdc
  have hp : popcorn = 5 := calc
    popcorn = ticket - 3   : hpdc.1
          ... = 8 - 3      : by rw ht
          ... = 5          : by norm_num
  have hd : drink = 6 := calc
    drink = popcorn + 1    : hpdc.2.1
         ... = 5 + 1       : by rw hp
         ... = 6           : by norm_num
  have hc : candy = 3 := calc
    candy = drink / 2      : hpdc.2.2.1
         ... = 6 / 2       : by rw hd
         ... = 3           : by norm_num
  have hs : deal_savings = 2 := hpdc.2.2.2
  calc
    (ticket + popcorn + drink + candy) - deal_savings
    = (8 + 5 + 6 + 3) - 2 : by rw [ht, hp, hd, hc, hs]
    ... = 20              : by norm_num

end deal_cost_l320_320666


namespace octagon_shaded_area_l320_320111

open Real

theorem octagon_shaded_area (side_length : ℝ) (h1 : side_length = 8)
  (A B C D E F G H : Point) (mid_AB mid_CD mid_EF mid_GH : Point)
  (h2 : midpoint A B = mid_AB) (h3 : midpoint C D = mid_CD)
  (h4 : midpoint E F = mid_EF) (h5 : midpoint G H = mid_GH) :
  let diagonal_length := side_length * sqrt 2
  in shaded_area mid_AB mid_CD mid_EF mid_GH = 128 :=
by
  sorry

end octagon_shaded_area_l320_320111


namespace abs_sum_of_factors_of_quadratic_l320_320850

variable (h b c d : ℤ)

theorem abs_sum_of_factors_of_quadratic :
  (∀ x : ℤ, 6 * x * x + x - 12 = (h * x + b) * (c * x + d)) →
  (|h| + |b| + |c| + |d| = 12) :=
by
  sorry

end abs_sum_of_factors_of_quadratic_l320_320850


namespace volume_P3_l320_320336

noncomputable def volumeSeq (n : ℕ) : ℚ :=
  if n = 0 then 1
  else let recVolume (i) := 
        if i = 0 then 1 
        else recVolume (i - 1) + (6/8) * (recVolume (i - 1) - recVolume (i - 2)) in
         recVolume (n)

theorem volume_P3 :
  let P3_Volume_rat := volumeSeq 3
  let m := 69
  let n := 32
  ∃ m n, rel_prime m n ∧ P3_Volume_rat = m / n ∧ m + n = 101 
  := by {
    let P3_Volume_rat := volumeSeq 3
    let m := 69
    let n := 32
    existsi m, n
    sorry
  }

end volume_P3_l320_320336


namespace true_statements_l320_320449

-- Define the variables and conditions
variables {m : ℝ}
def quadratic_eq (m : ℝ) (x : ℝ) := ((m + 1) * x^2 + 4 * m * x + (m - 3))

-- Lean 4 statement for the problem
theorem true_statements 
  (h1 : m ≠ -1)
  (h2 : discrim : (quadratic_eq m x).discrim (m + 1) (4 * m) (m - 3) > 0) :
  ∀ x, 
    ((∀ x, m ≠ -1) → 
    (quadratic_eq m x = 0 → (real_root_1 < -1 ∨ real_root_2 < -1)) ∧
    (quadratic_eq m x = 0 → ¬(real_root_1 > 1 ∨ real_root_2 > 1))) :=
sorry

end true_statements_l320_320449


namespace projection_of_v_onto_plane_P_l320_320752

-- Definitions and given conditions
noncomputable def v : ℝ^3 := ![2, 3, 1]
noncomputable def n : ℝ^3 := ![2, -4, 5]
noncomputable def p : ℝ^3 := ![32/15, 41/15, 18/15]

def plane_eq (x y z : ℝ) : Prop := 2 * x - 4 * y + 5 * z = 0

-- Projection problem statement
theorem projection_of_v_onto_plane_P :
    let u := v - p in
    plane_eq (v.1) (v.2) (v.3) →
    p = v - (u.1 * n / (n.1 * n.1 + n.2 * n.2 + n.3 * n.3)) :=
sorry

end projection_of_v_onto_plane_P_l320_320752


namespace largest_unexpressible_integer_l320_320458

-- Definitions based on conditions
variables (a b c : ℕ) (coprime_ab : Nat.coprime a b) (coprime_bc : Nat.coprime b c) (coprime_ca : Nat.coprime c a)
variables (x y z : ℕ)

-- Main Theorem Statement
theorem largest_unexpressible_integer :
  (2 * a * b * c - a * b - b * c - c * a) = k →
  ¬ ∃ (x y z : ℕ), k = x * b * c + y * c * a + z * a * b :=
by
  sorry

end largest_unexpressible_integer_l320_320458


namespace range_of_a_l320_320858

theorem range_of_a (a : ℝ) (A : Set ℝ) (h : A = {x | a * x^2 - 3 * x + 1 = 0} ∧ ∃ (n : ℕ), 2 ^ n - 1 = 3) :
  a ∈ Set.Ioo (-(1:ℝ)/0) 0 ∪ Set.Ioo 0 (9 / 4) :=
sorry

end range_of_a_l320_320858


namespace complex_imaginary_unit_theorem_l320_320564

def complex_imaginary_unit_equality : Prop :=
  let i := Complex.I
  i * (i + 1) = -1 + i

theorem complex_imaginary_unit_theorem : complex_imaginary_unit_equality :=
by
  sorry

end complex_imaginary_unit_theorem_l320_320564


namespace sum_of_squares_of_sines_l320_320242

theorem sum_of_squares_of_sines : 
  (\sum n in finset.range 1 30, real.sin (6 * n : ℝ) ^ 2) = 31 / 2 :=
begin
  sorry
end

end sum_of_squares_of_sines_l320_320242


namespace length_of_CD_l320_320009

theorem length_of_CD (A B C D : \Type)
  (AB BC AC : \Type)
  (h_triangle : AB = 9 ∧ BC = 12 ∧ AC = 15)
  (angle_bisector : CD → B → A → C → \Type)
  : CD = 4 * \sqrt 10 :=
sorry

end length_of_CD_l320_320009


namespace masha_numbers_l320_320520

theorem masha_numbers (a b : ℕ) (S : ℕ) (h1 : a ≠ b) (h2 : a > 11) (h3 : b > 11) (h4 : S = a + b) 
    (h5 : (∀ x y : ℕ, x + y = S → x = a ∨ y = a → abs x - y = a) ∧ (even a ∨ even b)) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) := 
by sorry

end masha_numbers_l320_320520


namespace distance_between_parallel_lines_l320_320389

-- Definitions of the lines in Lean terms
def line1 (x y : ℝ) := 2 * x - 4 * y - 4 = 0
def line2 (x y : ℝ) := 2 * x - 4 * y + 1 = 0

-- Function to calculate the distance between the two parallel lines
def distance_between_lines (a b c1 c2 : ℝ) : ℝ :=
  abs (c2 - c1) / real.sqrt (a^2 + b^2)

-- The theorem statement for the distance between line1 and line2
theorem distance_between_parallel_lines :
  distance_between_lines 2 (-4) (-4) 1 = (real.sqrt 5) / 2 :=
by
  sorry

end distance_between_parallel_lines_l320_320389


namespace slope_of_CD_is_neg1_l320_320563

theorem slope_of_CD_is_neg1 :
  let f(x y : ℝ) := x^2 + y^2 - 6 * x + 8 * y - 20 = 0
      g(x y : ℝ) := x^2 + y^2 - 8 * x + 6 * y + 12 = 0
  in ∀ (C D : ℝ × ℝ), 
      (f C.1 C.2) ∧ (f D.1 D.2) ∧ (g C.1 C.2) ∧ (g D.1 D.2) →
      ∃ m : ℝ, m = -1 :=
sorry

end slope_of_CD_is_neg1_l320_320563


namespace complement_intersection_l320_320345

open Set

theorem complement_intersection (M N : Set ℝ) (hM : M = {x | 0 < x ∧ x < 1}) (hN : N = {y | 2 ≤ y}) :
  (@compl ℝ _ M) ∩ N = {y | 2 ≤ y} :=
by
  -- assume definitions of M and N
  rw [hM, hN]
  -- simplifications
  rw [compl_set_of]
  -- reformulate the intersection in terms of set properties
  congr
  -- endpoints and intervals
  rw [← Set.union_comm, uIcc_compl (le_refl (0:ℝ))]

  sorry -- proof steps to be filled in.

end complement_intersection_l320_320345


namespace even_divisors_of_8fac_l320_320820

theorem even_divisors_of_8fac : 
  let num_even_divisors := ∏ x in {a | 1 ≤ a ∧ a ≤ 7}.card * 
                                      {b | 0 ≤ b ∧ b ≤ 2}.card *
                                      {c | 0 ≤ c ∧ c ≤ 1}.card *
                                      {d | 0 ≤ d ∧ d ≤ 1}.card
  in num_even_divisors = 84 := by
  sorry

end even_divisors_of_8fac_l320_320820


namespace Masha_thought_of_numbers_l320_320512

theorem Masha_thought_of_numbers : ∃ a b : ℕ, a ≠ b ∧ a > 11 ∧ b > 11 ∧ (a + b = 28) ∧ (a % 2 = 0 ∨ b % 2 = 0) ∧ (a = 12 ∧ b = 16 ∨ a = 16 ∧ b = 12) :=
by
  sorry

end Masha_thought_of_numbers_l320_320512


namespace sixth_equation_l320_320931

theorem sixth_equation :
  (6 + 7 + 8 + 9 + 10 + 11 + 12 + 13 + 14 + 15 + 16 = 121) :=
by
  sorry

end sixth_equation_l320_320931


namespace dihedral_angle_cosine_l320_320110

theorem dihedral_angle_cosine (S A B C D E : ℝ)
  (h_tetrahedron : regular_tetrahedron S A B C)
  (base_triangle : equilateral_triangle A B C)
  (edge_length_SA : dist S A = 2) (edge_length_SB : dist S B = 2) (edge_length_SC : dist S C = 2)
  (base_edge_length : dist A B = 1)
  (plane_divides_volume : divides_volume S A B C D E)
  (D_midpoint_SC : midpoint D S C)
  (E_midpoint_AB : midpoint E A B) :
  cos_dihedral_angle (S A B C) (plane (S D E)) = 2 / sqrt 15 := sorry

end dihedral_angle_cosine_l320_320110


namespace minimum_triangles_cover_l320_320148

theorem minimum_triangles_cover (
    side_length_large : ℝ,
    side_length_small1 : ℝ,
    side_length_small2 : ℝ,
    area_large : ℝ,
    area_small1 : ℝ,
    area_small2 : ℝ) 
    (h_large : side_length_large = 9)
    (h_small1 : side_length_small1 = 1)
    (h_small2 : side_length_small2 = 2)
    (h_area_large : area_large = (sqrt 3 / 4) * side_length_large^2)
    (h_area_small1 : area_small1 = (sqrt 3 / 4) * side_length_small1^2)
    (h_area_small2 : area_small2 = (sqrt 3 / 4) * side_length_small2^2)
    : minimum_number_of_triangles_needed area_large area_small1 area_small2 = 21 :=
sorry

noncomputable def minimum_number_of_triangles_needed (area_large area_small1 area_small2 : ℝ) : ℕ :=
  let num_small2 := (ceil (area_large / area_small2)).toNat in
  if area_large - num_small2 * area_small2 = 0 then num_small2
  else num_small2 + (ceil ((area_large - num_small2 * area_small2) / area_small1)).toNat


end minimum_triangles_cover_l320_320148


namespace prob_B_win_correct_l320_320146

-- Define the probabilities for player A winning and a draw
def prob_A_win : ℝ := 0.3
def prob_draw : ℝ := 0.4

-- Define the total probability of all outcomes
def total_prob : ℝ := 1

-- Define the probability of player B winning
def prob_B_win : ℝ := total_prob - prob_A_win - prob_draw

-- Proof problem: Prove that the probability of player B winning is 0.3
theorem prob_B_win_correct : prob_B_win = 0.3 :=
by
  -- The proof would go here, but we use sorry to skip it for now.
  sorry

end prob_B_win_correct_l320_320146


namespace expression_value_l320_320296

theorem expression_value :
  (2^1006 + 5^1007)^2 - (2^1006 - 5^1007)^2 = 40 * 10^1006 :=
by sorry

end expression_value_l320_320296


namespace vacuum_cleaner_cost_l320_320020

-- Variables
variables (V : ℝ)

-- Conditions
def cost_of_dishwasher := 450
def coupon := 75
def total_spent := 625

-- The main theorem to prove
theorem vacuum_cleaner_cost : V + cost_of_dishwasher - coupon = total_spent → V = 250 :=
by
  -- Proof logic goes here
  sorry

end vacuum_cleaner_cost_l320_320020


namespace number_of_possible_values_l320_320913

noncomputable def A (a : ℝ) : Set ℝ := {1, 2, a}
noncomputable def B (a : ℝ) : Set ℝ := {1, a^2}

theorem number_of_possible_values (a : ℝ) :
  (A a ∪ B a = A a) →
  ∃! x : Set ℝ, (x = 0 ∨ x = -Real.sqrt 2 ∨ x = Real.sqrt 2) :=
by
  sorry

end number_of_possible_values_l320_320913


namespace arctan_sum_eq_pi_div_six_l320_320034

theorem arctan_sum_eq_pi_div_six
  (a : ℝ) (b : ℝ)
  (ha : a = 1 / 3)
  (h_eq : (a + 1) * (b + 1) = 3 / 2) :
  arctan a + arctan b = π / 6 :=
begin
  sorry
end

end arctan_sum_eq_pi_div_six_l320_320034


namespace nat_solutions_not_perfect_squares_l320_320447

theorem nat_solutions_not_perfect_squares 
  (a b : ℕ) 
  (h1 : a > b) 
  (h2 : (a % 2) = (b % 2)) :
  ∀ x ∈ { b^2 + 1, a^2 - a - b^2 },
  (x ∈ ℕ) ∧ ¬ ∃ k : ℕ, k^2 = x := 
sorry

end nat_solutions_not_perfect_squares_l320_320447


namespace sin_eq_given_conditions_l320_320310

theorem sin_eq_given_conditions : 
  ∃ n : ℤ, -180 ≤ n ∧ n ≤ 180 ∧ (sin (n : ℝ) * π / 180) = sin (830 * π / 180) ↔ (n = 70 ∨ n = 110) :=
by
  sorry

end sin_eq_given_conditions_l320_320310


namespace x_eq_1_iff_quadratic_eq_zero_l320_320994

theorem x_eq_1_iff_quadratic_eq_zero :
  ∀ x : ℝ, (x = 1) ↔ (x^2 - 2 * x + 1 = 0) := by
  sorry

end x_eq_1_iff_quadratic_eq_zero_l320_320994


namespace M_intersect_N_l320_320388

def M : Set ℤ := {x | Real.log (x^2) = 0}
def N : Set ℤ := {x | (1 / 2) < 2^(x + 1) ∧ 2^(x + 1) < 4}

theorem M_intersect_N : M ∩ N = {-1} := by
  sorry

end M_intersect_N_l320_320388


namespace range_of_t_express_f_as_g_max_value_f_range_of_m_l320_320325

namespace ProofProblem

open Real

noncomputable def f (a x : ℝ) : ℝ := a * cos x + sqrt (1 + sin x) + sqrt (1 - sin x)

theorem range_of_t (a : ℝ) (x : ℝ) (ha : a < 0) (hx : x ∈ Icc (-π / 2) (π / 2)) :
  let t := sqrt (1 + sin x) + sqrt (1 - sin x) in t ∈ Icc (sqrt 2) 2 :=
sorry

theorem express_f_as_g (a : ℝ) (x : ℝ) (ha : a < 0) (hx : x ∈ Icc (-π / 2) (π / 2)) :
  let t := sqrt (1 + sin x) + sqrt (1 - sin x) in f a x = (a / 2) * t^2 + t - a :=
sorry

theorem max_value_f (a : ℝ) (h_neg_a : a < 0) :
  let t_min := (sqrt 2) in let t_max := (2) in 
  (t_min ≤ -1/a ∧ -1/a ≤ t_max → ∃ t ∈ Icc t_min t_max, f a (-acos ((t^2 - 2) / 2)) = (-1/a) + t - a) ∧
  (-1/a < t_min ∧ t_min < t_max → ∃ t ∈ Icc t_min t_max, f a (-acos ((t^2 - 2) / 2)) = (a + 2)) ∧
  (t_min < -1/a ∧ -1/a < pi_max → ∃ t ∈ Icc t_min t_max, f a (-acos ((t^2 - 2) / 2)) = (sqrt 2)) :=
sorry

theorem range_of_m (x1 x2 : ℝ) (a : ℝ) (ha : a = -1) (hx1 : x1 ∈ Icc (-π / 2) (π / 2)) (hx2 : x2 ∈ Icc (-π / 2) (π / 2)):
  ∃ m, m ≥ 1/2 ∧ (abs (f a x1 - f a x2) ≤ m) :=
sorry

end ProofProblem

end range_of_t_express_f_as_g_max_value_f_range_of_m_l320_320325


namespace even_divisors_8_factorial_l320_320836

theorem even_divisors_8_factorial : 
  let n := (2^7) * (3^2) * 5 * 7 in
  ∃ (count : ℕ), even_divisors_count n = 84 := 
sorry

end even_divisors_8_factorial_l320_320836


namespace parallel_line_through_focus_l320_320568

theorem parallel_line_through_focus :
  ∀ (l : ℝ) (x y : ℝ), 
    (l = 3 * x - 2 * y - 3) → -- The line l is of the form 3x - 2y - c = 0
    (y ^ 2 = 2 * x) →         -- The parabola equation is y^2 = 2x
    ((x, y) = (1 / 2, 0)) →   -- Focus of the parabola (y^2 = 2x) is (1/2, 0)
    l = 6 * x - 4 * y - 3 :=  -- Therefore, the required line's equation is 6x - 4y - 3
by
  intros l x y Hl Hparabola Hfocus,
  sorry

end parallel_line_through_focus_l320_320568


namespace find_numbers_l320_320502

theorem find_numbers (a b : ℕ) (h1 : a > 11) (h2 : b > 11) (h3 : a ≠ b)
  (h4 : (∃ S, S = a + b) ∧ (∀ (x y : ℕ), x ≠ y → x + y = a + b → (x > 11) → (y > 11) → ¬(x = a ∨ y = a) → ¬(x = b ∨ y = b)))
  (h5 : even a ∨ even b) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) :=
by
  sorry

end find_numbers_l320_320502


namespace Sergey_full_years_l320_320553

def full_years (years months weeks days hours : ℕ) : ℕ :=
  years + months / 12 + (weeks * 7 + days) / 365

theorem Sergey_full_years 
  (years : ℕ)
  (months : ℕ)
  (weeks : ℕ)
  (days : ℕ)
  (hours : ℕ) :
  years = 36 →
  months = 36 →
  weeks = 36 →
  days = 36 →
  hours = 36 →
  full_years years months weeks days hours = 39 :=
by
  intros
  sorry

end Sergey_full_years_l320_320553


namespace valid_divisors_of_196_l320_320118

theorem valid_divisors_of_196 : 
  ∃ d : Finset Nat, (∀ x ∈ d, 1 < x ∧ x < 196 ∧ 196 % x = 0) ∧ d.card = 7 := by
  sorry

end valid_divisors_of_196_l320_320118


namespace solution_is_3_l320_320304

theorem solution_is_3 (x : ℝ) (h : log x 243 = log 2 32) : x = 3 := sorry

end solution_is_3_l320_320304


namespace walking_time_l320_320156

noncomputable def time_to_reach_destination (mr_harris_speed : ℝ) (mr_harris_time_to_store : ℝ) (your_speed : ℝ) (distance_factor : ℝ) : ℝ :=
  let store_distance := mr_harris_speed * mr_harris_time_to_store
  let your_destination_distance := distance_factor * store_distance
  your_destination_distance / your_speed

theorem walking_time (mr_harris_speed your_speed : ℝ) (mr_harris_time_to_store : ℝ) (distance_factor : ℝ) (h_speed : your_speed = 2 * mr_harris_speed) (h_time : mr_harris_time_to_store = 2) (h_factor : distance_factor = 3) :
  time_to_reach_destination mr_harris_speed mr_harris_time_to_store your_speed distance_factor = 3 :=
by
  rw [h_time, h_speed, h_factor]
  -- calculations based on given conditions
  sorry

end walking_time_l320_320156


namespace num_even_divisors_of_8_l320_320827

def factorial (n : Nat) : Nat :=
  match n with
  | 0     => 1
  | Nat.succ n' => Nat.succ n' * factorial n'

-- Define the prime factorization of 8!
def prime_factors_eight_factorial : Nat := 2^7 * 3^2 * 5 * 7

-- Definition of an even divisor of 8!
def is_even_divisor (d : Nat) : Prop :=
  d ∣ prime_factors_eight_factorial ∧ 2 ∣ d

-- Calculation of number of even divisors of 8!
def num_even_divisors_8! : Nat :=
  7 * 3 * 2 * 2

theorem num_even_divisors_of_8! :
  num_even_divisors_8! = 84 :=
sorry

end num_even_divisors_of_8_l320_320827


namespace probability_at_least_one_pen_l320_320103

noncomputable def PAs  := 3/5
noncomputable def PBs  := 2/3
noncomputable def PABs := PAs * PBs

theorem probability_at_least_one_pen : PAs + PBs - PABs = 13 / 15 := by
  sorry

end probability_at_least_one_pen_l320_320103


namespace sin_squared_sum_l320_320269

theorem sin_squared_sum : 
  ∑ k in finset.range 30 \ {0}, (sin (6 * k + 6) * (real.pi / 180))^2 = 15 :=
sorry

end sin_squared_sum_l320_320269


namespace numbers_masha_thought_l320_320528

noncomputable def distinct_natural_numbers_greater_than_eleven_and_sum_to_satisfy_conditions : ℕ → ℕ → Prop :=
λ a b, a ≠ b ∧ a > 11 ∧ b > 11 ∧ (a + b = 28) ∧ (¬ (∃ x y, x ≠ y ∧ x > 11 ∧ y > 11 ∧ x + y = a + b ∧ (x ≠ a ∧ y ≠ b)))

theorem numbers_masha_thought (a b : ℕ) (h : distinct_natural_numbers_greater_than_eleven_and_sum_to_satisfy_conditions a b) : 
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) :=
by sorry

end numbers_masha_thought_l320_320528


namespace angle_liquid_surface_cube_is_34_18_l320_320662

noncomputable def angle_between_liquid_surface_and_cube (γ_l γ_c : ℝ) : ℝ :=
  let tan_phi := 0.6823 in -- Given computed tangent approximation
  Real.atan tan_phi

theorem angle_liquid_surface_cube_is_34_18 :
  ∀ (a : ℝ), (γ_c : ℝ), γ_l : ℝ,
  (γ_c = γ_l / 3) →
  angle_between_liquid_surface_and_cube γ_l γ_c ≈ 34.308 :
  sorry

end angle_liquid_surface_cube_is_34_18_l320_320662


namespace closest_point_to_origin_l320_320958

theorem closest_point_to_origin : 
  ∃ x y : ℝ, x > 0 ∧ y = x + 1/x ∧ (x, y) = (1/(2^(1/4)), (1 + 2^(1/2))/(2^(1/4))) :=
by
  sorry

end closest_point_to_origin_l320_320958


namespace minimizing_plane_through_G_l320_320898

-- The vectors X, Y, Z, and G are defined as vectors in the space
variables {X Y Z G : ℝ × ℝ × ℝ}
-- P is any point in the plane
variables {P : ℝ × ℝ × ℝ}

-- The condition that the plane contains point G
def contains_G (V : ℝ × ℝ × ℝ) : Prop :=
  V.1 * G.1 + V.2 * G.2 + V.3 * G.3 = 1

-- The equation of the plane through point G intersecting rays OX, OY, OZ in points A, B, C minimizing the volume of tetrahedron OABC
theorem minimizing_plane_through_G (V : ℝ × ℝ × ℝ) (h_V : contains_G V) :
  G.1 * P.1 + G.2 * P.2 + G.3 * P.3 = G.1 * G.1 + G.2 * G.2 + G.3 * G.3 :=
sorry

end minimizing_plane_through_G_l320_320898


namespace sum_sin_squared_angles_l320_320263

theorem sum_sin_squared_angles : 
  ∑ k in finset.range 30, (sin (6 * (k + 1) * (real.pi / 180)))^2 = 31 / 2 := 
sorry

end sum_sin_squared_angles_l320_320263


namespace number_of_gorgeous_grids_l320_320438

theorem number_of_gorgeous_grids : ∃ (n : ℕ), n = 9^8 ∧
  (∀ grid : (Fin 9 → Fin 9 → Fin 9), 
    (∀ i : Fin 9, multiset.of_fn (λ j, grid i j) = {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
     multiset.of_fn (λ j, grid j i) = {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
     (multiset.of_fn (λ (k : Fin 9), grid k k)).sum % 9 = 0 ∧
     (multiset.of_fn (λ (k : Fin 9), grid k (8 - k))).sum % 9 = 0)) :=
begin
  sorry
end

end number_of_gorgeous_grids_l320_320438


namespace masha_numbers_l320_320517

theorem masha_numbers (a b : ℕ) (S : ℕ) (h1 : a ≠ b) (h2 : a > 11) (h3 : b > 11) (h4 : S = a + b) 
    (h5 : (∀ x y : ℕ, x + y = S → x = a ∨ y = a → abs x - y = a) ∧ (even a ∨ even b)) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) := 
by sorry

end masha_numbers_l320_320517


namespace rectangle_becomes_square_l320_320877

variable {A B C D O : Type} [normed_add_comm_group A] [normed_add_comm_group B] [normed_add_comm_group C] [normed_add_comm_group D] [normed_add_comm_group O]

variable {AB CD : ℝ}

theorem rectangle_becomes_square (hAB : AB > 0) (hBC : CD > 0)
  (h_diagonals_intersect : ∃ O : Type, true)
  (h_sides_equal : AB = CD) :
  ∃ s : ℝ, AB = s ∧ CD = s :=
by {
  sorry
}

end rectangle_becomes_square_l320_320877


namespace find_numbers_l320_320508

theorem find_numbers (a b : ℕ) (h1 : a > 11) (h2 : b > 11) (h3 : a ≠ b)
  (h4 : (∃ S, S = a + b) ∧ (∀ (x y : ℕ), x ≠ y → x + y = a + b → (x > 11) → (y > 11) → ¬(x = a ∨ y = a) → ¬(x = b ∨ y = b)))
  (h5 : even a ∨ even b) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) :=
by
  sorry

end find_numbers_l320_320508


namespace sides_sqrt_triangle_l320_320545

theorem sides_sqrt_triangle (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) 
  (habc : a + b > c) (hbca : b + c > a) (hcab : c + a > b) :
  (sqrt a + sqrt b > sqrt c) ∧ (sqrt b + sqrt c > sqrt a) ∧ (sqrt c + sqrt a > sqrt b) := 
by
  sorry

end sides_sqrt_triangle_l320_320545


namespace correct_finance_specialization_l320_320419

-- Variables representing percentages of students specializing in different subjects
variables (students : Type) -- Type of students
           (is_specializing_finance : students → Prop) -- Predicate for finance specialization
           (is_specializing_marketing : students → Prop) -- Predicate for marketing specialization

-- Given conditions
def finance_specialization_percentage : ℝ := 0.88 -- 88% of students are taking finance specialization
def marketing_specialization_percentage : ℝ := 0.76 -- 76% of students are taking marketing specialization

-- The proof statement
theorem correct_finance_specialization (h_finance : finance_specialization_percentage = 0.88) :
  finance_specialization_percentage = 0.88 :=
by
  sorry

end correct_finance_specialization_l320_320419


namespace shortest_altitude_l320_320582

theorem shortest_altitude (a b c : ℝ) (ha : a = 13) (hb : b = 14) (hc : c = 15) : 
  ∃ h, h = 11.2 ∧ h = 2 * (sqrt (( (a+b+c)/2) * ( (a+b+c)/2 - a) * ( (a+b+c)/2 - b) * ( (a+b+c)/2 - c))) / c :=
by sorry

end shortest_altitude_l320_320582


namespace radius_of_sector_l320_320559

theorem radius_of_sector (θ : ℝ) (A : ℝ) (r : ℝ) : θ = 38 ∧ A = 47.77142857142857 → r ≈ 12 :=
by
  sorry

end radius_of_sector_l320_320559


namespace value_of_x_l320_320593

theorem value_of_x (x : ℝ) (h : x = -x) : x = 0 := 
by 
  sorry

end value_of_x_l320_320593


namespace transformed_quadratic_fn_equation_l320_320132

theorem transformed_quadratic_fn_equation :
  ∀ x : ℝ,
  let f := λ x : ℝ, -2 * x^2 + 1 in
  let g := λ x : ℝ, -2 * (x - 3)^2 - 1 in
  g x = -2 * (x - 3)^2 - 1 :=
by sorry

end transformed_quadratic_fn_equation_l320_320132


namespace walking_problem_statement_l320_320540

-- Definitions of the conditions
variables
  (v_A v_B : ℕ)  -- speeds of person A and B respectively in meters per minute
  (t_A t_B t_C t_AB : ℕ) -- time variables t_A, t_B, t_C, t_AB in minutes
  (d_AB : ℕ) -- distance between points A and B in meters

-- Condition 1
def personA_pre_walked_5_5_minutes_before_B_starts := t_A = 5.5
-- Condition 2
def personB_walks_30_meters_per_minute_faster_than_personA := v_B = v_A + 30
-- Condition 3
def A_meets_B_at_C := t_A + t_AB = t_B
-- Condition 4
def time_A_to_C_is_4_minutes_longer_than_time_C_to_B := t_A = t_B + 4
-- Condition 5
def time_B_to_C_to_A_is_3_minutes_longer_than_time_B_to_C := t_A + t_B + t_C = t_B + 3

-- The proof problem statement
theorem walking_problem_statement
  (h1: personA_pre_walked_5_5_minutes_before_B_starts)
  (h2: personB_walks_30_meters_per_minute_faster_than_personA)
  (h3: A_meets_B_at_C)
  (h4: time_A_to_C_is_4_minutes_longer_than_time_C_to_B)
  (h5: time_B_to_C_to_A_is_3_minutes_longer_than_time_B_to_C) :
  (t_A = 10) ∧ (d_AB = 1440) :=
by
  sorry

end walking_problem_statement_l320_320540


namespace find_a_for_even_function_l320_320366

-- conditions
def f (a : ℝ) (x : ℝ) : ℝ := x^3 * (a * 2^x - 2^(-x))

-- definition of an even function
def is_even_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g x = g (-x)

-- the proof problem statement
theorem find_a_for_even_function (a : ℝ) : is_even_function (f a) ↔ a = 1 :=
by
  sorry

end find_a_for_even_function_l320_320366


namespace bakery_production_l320_320988

-- Definitions of baking capacities and loaves baked each day
def oven_capacity : ℕ := 20

def sourdough_Wednesday : ℕ := 5
def rye_Thursday : ℕ := 7
def sourdough_Friday : ℕ := 10
def rye_Saturday : ℕ := 14
def sourdough_Sunday : ℕ := 19

-- Function to calculate loaves baked given previous day's production and type
def baked_loaves (prev_loaves : ℕ) (bread_type : String) : ℕ :=
  min (truncate (1.5 * prev_loaves)) oven_capacity

-- Proof problem statement
theorem bakery_production :
  baked_loaves rye_Saturday "rye" = 20 ∧
  baked_loaves sourdough_Sunday "sourdough" = 20 :=
by {
  sorry
}

end bakery_production_l320_320988


namespace max_a_squared_b_l320_320777

theorem max_a_squared_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a * (a + b) = 27) : a^2 * b ≤ 54 :=
sorry

end max_a_squared_b_l320_320777


namespace evaluate_fraction_l320_320301

theorem evaluate_fraction : (25 * 5 + 5^2) / (5^2 - 15) = 15 := 
by
  sorry

end evaluate_fraction_l320_320301


namespace percentage_reduction_l320_320670

theorem percentage_reduction (original reduced : ℕ) (h₁ : original = 260) (h₂ : reduced = 195) :
  (original - reduced) / original * 100 = 25 := by
  sorry

end percentage_reduction_l320_320670


namespace sum_of_sin_squared_angles_l320_320249

theorem sum_of_sin_squared_angles :
  (∑ i in Finset.range 30, Real.sin (6 * (i + 1)) * Real.sin (6 * (i + 1))) = 15.5 :=
sorry

end sum_of_sin_squared_angles_l320_320249


namespace a5_value_S8_value_l320_320771

-- Definitions based on the conditions
def seq (n : ℕ) : ℕ :=
if n = 0 then 0
else if n = 1 then 1
else 2 * seq (n - 1)

noncomputable def S (n : ℕ) : ℕ :=
(1 - 2^n) / (1 - 2)

-- Proof statements
theorem a5_value : seq 5 = 16 := sorry

theorem S8_value : S 8 = 255 := sorry

end a5_value_S8_value_l320_320771


namespace path_length_correct_l320_320967

noncomputable def length_of_path_traveled_by_point_A
    (AB CD BC DA : ℝ)
    (AB_eq_CD : AB = 4) (BC_eq_DA: BC = 8)
    (first_rotation_angle second_rotation_angle : ℝ)
    (first_rotation_angle_eq : first_rotation_angle = 90)
    (second_rotation_angle_eq : second_rotation_angle = 90):
    ℝ :=
  let r1 := (AB^2 + DA^2).sqrt,
      arc1 := (2 * r1 * Real.pi) / 4,
      arc2 := (2 * BC * Real.pi) / 4
  in arc1 + arc2

theorem path_length_correct :
  ∀ (AB CD BC DA : ℝ)
    (AB_eq_CD : AB = 4) (BC_eq_DA: BC = 8)
    (first_rotation_angle second_rotation_angle : ℝ)
    (first_rotation_angle_eq : first_rotation_angle = 90)
    (second_rotation_angle_eq : second_rotation_angle = 90),
  length_of_path_traveled_by_point_A AB CD BC DA AB_eq_CD BC_eq_DA first_rotation_angle second_rotation_angle first_rotation_angle_eq second_rotation_angle_eq
  = (4 + 2 * Real.sqrt 5) * Real.pi := 
by
  intros,
  sorry

end path_length_correct_l320_320967


namespace sabrina_herbs_l320_320068

theorem sabrina_herbs (S V : ℕ) 
  (h1 : 2 * S = 12)
  (h2 : 12 + S + V = 29) :
  V - S = 5 := by
  sorry

end sabrina_herbs_l320_320068


namespace value_of_expression_l320_320151

theorem value_of_expression :
  (3^2 - 3) - (4^2 - 4) + (5^2 - 5) - (6^2 - 6) = -16 :=
by
  sorry

end value_of_expression_l320_320151


namespace unique_B_for_A47B_divisible_by_7_l320_320292

-- Define the conditions
def A : ℕ := 4

-- Define the main proof problem statement
theorem unique_B_for_A47B_divisible_by_7 : 
  ∃! B : ℕ, B ≤ 9 ∧ (100 * A + 70 + B) % 7 = 0 :=
        sorry

end unique_B_for_A47B_divisible_by_7_l320_320292


namespace cookies_in_jar_l320_320128

noncomputable def number_of_cookies_in_jar : ℕ := sorry

theorem cookies_in_jar :
  (number_of_cookies_in_jar - 1) = (1 / 2 : ℝ) * (number_of_cookies_in_jar + 5) →
  number_of_cookies_in_jar = 7 :=
by
  sorry

end cookies_in_jar_l320_320128


namespace number_of_even_divisors_of_factorial_eight_l320_320832

-- Definition of 8! and its prime factorization
def factorial_eight : ℕ := 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1
def prime_factorization_factorial_eight : Prop :=
  factorial_eight = 2^7 * 3^2 * 5 * 7

-- The main theorem statement
theorem number_of_even_divisors_of_factorial_eight :
  prime_factorization_factorial_eight →
  ∃ n, n = 7 * 3 * 2 * 2 ∧
  (∀ d, d ∣ factorial_eight → (∃ a b c d, 1 ≤ a ∧ a ≤ 7 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 1 ∧ 0 ≤ d ∧ d ≤ 1 ∧ d = 2^a * 3^b * 5^c * 7^d) →
  (7 * 3 * 2 * 2 = n)) :=
by
  intro h
  use 84
  sorry

end number_of_even_divisors_of_factorial_eight_l320_320832


namespace integer_sequence_solution_l320_320738

theorem integer_sequence_solution :
  ∀ (x : Fin 1997 → ℤ),
    (∑ k, 2^(k : ℕ) * x k ^ 1997) = 1996 * ∏ k, x k →
    ∀ i, x i = 0 :=
by sorry

end integer_sequence_solution_l320_320738


namespace closest_point_on_graph_l320_320947

theorem closest_point_on_graph (x y : ℝ) (h1 : x > 0) (h2 : y = x + 1/x) :
  (x = 1/real.root 4 2) ∧ (y = (1 + real.sqrt 2) / real.root 4 2) :=
sorry

end closest_point_on_graph_l320_320947


namespace min_positive_period_of_f_interval_monotonically_decreasing_of_f_min_value_of_f_in_interval_l320_320376

noncomputable def f (x : ℝ) : ℝ := 2 * cos x * (sin x - (real.sqrt 3) * cos x) + (real.sqrt 3)

theorem min_positive_period_of_f :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ ∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T :=
  sorry

theorem interval_monotonically_decreasing_of_f :
  ∀ k : ℤ, (∀ x, (k * real.pi + 5 * real.pi / 12 ≤ x) ∧ (x ≤ k * real.pi + 11 * real.pi / 12) 
  → f' x < 0) :=
  sorry

theorem min_value_of_f_in_interval :
  ∃ x, x ∈ (set.Icc (real.pi / 2) real.pi) ∧ f x = -2 ∧ (∀ y ∈ (set.Icc (real.pi / 2) real.pi), f y ≥ -2) :=
  sorry

end min_positive_period_of_f_interval_monotonically_decreasing_of_f_min_value_of_f_in_interval_l320_320376


namespace num_valid_row_lengths_l320_320652

-- Define the problem parameters.
def number_of_members : Nat := 90

-- Define the range for the number of students per row.
def valid_row_length (x : Nat) : Prop := 4 ≤ x ∧ x ≤ 15

-- Define a predicate that checks if a number is a valid row length that divides 90.
def valid_divisor (x : Nat) : Prop := x ∣ number_of_members ∧ valid_row_length x

-- Prove that there are exactly 5 valid row lengths.
theorem num_valid_row_lengths : (Finset.card (Finset.filter valid_divisor (Finset.range (number_of_members + 1)))) = 5 := 
  sorry

end num_valid_row_lengths_l320_320652


namespace closest_point_to_origin_l320_320941

def y (x : ℝ) := x + 1 / x

theorem closest_point_to_origin : ∃ x : ℝ, x > 0 ∧ (x, y x) = (1 / 2^(1/4 : ℝ), (1 + real.sqrt 2) / 2^(1/4 : ℝ)) :=
by
  sorry

end closest_point_to_origin_l320_320941


namespace value_OA_OB_l320_320786

-- Define the parametric and polar equations of the curves C1 and C2.
def parametric_C1 (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, Real.sin θ)
def polar_C2 (θ : ℝ) : ℝ := 2 * Real.sin θ

-- Define the points M1 and M2 in polar coordinates and their conversions.
def M1 := (1 : ℝ, Real.pi / 2)
def M2 := (2 : ℝ, 0)

-- Define the line equation passing through M1 and M2 and its intersection properties.
def line_M1M2 : ℝ × ℝ → Prop := λ (x : ℝ, y : ℝ), x + 2 * y = 2

-- Define the theorem statement.
theorem value_OA_OB :
  ∀ (O A B : ℝ × ℝ)
    (P Q : ℝ × ℝ)
    (h1 : line_M1M2 P)
    (h2 : line_M1M2 Q)
    (hC2 : ∀ θ : ℝ, P = polar_C2 θ ∧ Q = polar_C2 (θ + Real.pi / 2))
    (h_angles : ∀ θ, P = polar_C2 θ ∧ Q = polar_C2 (θ + Real.pi / 2)),
    (1 / ((Real.sqrt ((fst A)^2 + (snd A)^2))^2) + 1 / ((Real.sqrt ((fst B)^2 + (snd B)^2))^2)) = 5 / 4 :=
sorry

end value_OA_OB_l320_320786


namespace area_HIK_l320_320085

theorem area_HIK :
  ∀ (A C D B C E F D E J L J F G H H I K : ℝ),
    (A = 22 ∧ B = 500 ∧ D = 482 ∧ F = 22) → 
    (H I K = 26) :=
by
  sorry

end area_HIK_l320_320085


namespace interest_rate_difference_l320_320214

-- Definitions for the given conditions
def principal : ℝ := 900
def time : ℝ := 3
def interest_diff : ℝ := 81

-- Definitions for the rates
noncomputable def original_rate : ℝ := sorry
noncomputable def higher_rate : ℝ := sorry

-- Simple interest formula
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (rate / 100) * time

-- Assertion of the problem statement
theorem interest_rate_difference :
  simple_interest principal higher_rate time = simple_interest principal original_rate time + interest_diff →
  higher_rate - original_rate = 1 / 3 := 
sorry

end interest_rate_difference_l320_320214


namespace mashas_numbers_l320_320498

def is_even (n : ℕ) : Prop := n % 2 = 0

def problem_statement (a b : ℕ) : Prop :=
  a ≠ b ∧ a > 11 ∧ b > 11 ∧ is_even a ∧ a + b = 28
  
theorem mashas_numbers : ∃ (a b : ℕ), problem_statement a b :=
by
  use 12
  use 16
  unfold problem_statement
  split
  -- a ≠ b
  exact dec_trivial
  split
  -- a > 11
  exact dec_trivial
  split
  -- b > 11
  exact dec_trivial
  split
  -- is_even a
  exact dec_trivial
  -- a + b = 28
  exact dec_trivial

end mashas_numbers_l320_320498


namespace number_of_second_shift_members_l320_320724

-- Definitions of conditions
def first_shift_members := 60
def first_shift_pension_percentage := 0.20
def first_shift_pension_members := first_shift_members * first_shift_pension_percentage

def third_shift_members := 40
def third_shift_pension_percentage := 0.10
def third_shift_pension_members := third_shift_members * third_shift_pension_percentage

def total_pension_percentage := 0.24

-- Definition of unknown variable
variable (S : ℕ)

-- Total number of members and pension participating members
def total_members := first_shift_members + S + third_shift_members
def total_pension_members := 0.24 * total_members

def total_pension_from_shifts := first_shift_pension_members + (0.40 * S) + third_shift_pension_members

-- The theorem to prove
theorem number_of_second_shift_members : S = 50 :=
by {
  have eq1 : total_pension_members = total_pension_from_shifts := by sorry,
  sorry
}

end number_of_second_shift_members_l320_320724


namespace even_divisors_8_factorial_l320_320837

theorem even_divisors_8_factorial : 
  let n := (2^7) * (3^2) * 5 * 7 in
  ∃ (count : ℕ), even_divisors_count n = 84 := 
sorry

end even_divisors_8_factorial_l320_320837


namespace min_value_of_a_l320_320343

theorem min_value_of_a (a b c d : ℕ) (h1 : a > b) (h2 : b > c) (h3 : c > d) (h4 : a + b + c + d = 2004) (h5 : a^2 - b^2 + c^2 - d^2 = 2004) : a = 503 :=
sorry

end min_value_of_a_l320_320343


namespace problem_statement_l320_320035

noncomputable def a := 3 ^ 0.3
noncomputable def b := 3 ^ 0.4
noncomputable def c := Real.logb 4 0.3

def is_even_function (f : ℝ → ℝ) := ∀ x, f x = f (-x)
def is_increasing_in_negative (f : ℝ → ℝ) := ∀ x y, x < y ∧ y < 0 → f x < f y

theorem problem_statement (f : ℝ → ℝ) 
  (h_even : is_even_function f) 
  (h_increasing : is_increasing_in_negative f) :
  f c > f a ∧ f a > f b :=
by
  sorry

end problem_statement_l320_320035


namespace part_i_part_ii_l320_320899

def f : ℕ → ℤ 
| n := if ∃ j ≥ 0, n = 2^j - 1 then 0 else f (n-1) - 1

theorem part_i (n : ℕ) : ∃ k ≥ 0, f n + n = 2^k - 1 := by
  sorry

theorem part_ii : f (2^1990) = -1 := by
  sorry

end part_i_part_ii_l320_320899


namespace pastries_more_than_cakes_l320_320228

theorem pastries_more_than_cakes (pastries_sold cakes_sold: ℕ): 
  pastries_sold = 154 → cakes_sold = 78 → (pastries_sold - cakes_sold) = 76 := 
by 
  -- given conditions
  intros h1 h2
  -- shape of the goal
  rw [h1, h2]
  -- solve the goal
  exact rfl

end pastries_more_than_cakes_l320_320228


namespace distinct_product_sum_l320_320294

theorem distinct_product_sum : 
  (∃ (X Y : ℕ), X ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ Y ∈ {0, 5} ∧
   (X = 8 ∧ Y = 0) ∨ (X = 3 ∧ Y = 5) ∧ 
   (59000000 + 10000 * X + 10 * Y + 3452) % 45 = 0 ∧ 
   (∑ d in [5, 9, 0, 3, 4, 5, 2, X, Y], d) % 9 = 0) → 
  ∑ xy, (if xy = 0 * 8 then 0 else if xy = 5 * 3 then 15 else 0) = 15 :=
by sorry

end distinct_product_sum_l320_320294


namespace product_of_permutations_even_l320_320895

open Nat

variables {n : ℕ}
variables (p : Fin n → Fin n)

theorem product_of_permutations_even 
  (h_perm : Function.Bijective p)
  (h_odd : Odd n) : Even (∏ i in Finset.finRange n, (p i).val - i.val) :=
sorry

end product_of_permutations_even_l320_320895


namespace sin_pi_minus_alpha_l320_320780

theorem sin_pi_minus_alpha (α : ℝ) (h : Real.sin α = 5/13) : Real.sin (π - α) = 5/13 :=
by
  sorry

end sin_pi_minus_alpha_l320_320780


namespace negation_of_proposition_is_false_l320_320575

theorem negation_of_proposition_is_false :
  (¬ ∀ (x : ℝ), x < 0 → x^2 > 0) = true :=
by
  sorry

end negation_of_proposition_is_false_l320_320575


namespace economical_refuel_l320_320053

theorem economical_refuel (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) : 
  (x + y) / 2 > (2 * x * y) / (x + y) :=
sorry -- Proof omitted

end economical_refuel_l320_320053


namespace Yuna_drank_most_l320_320018

noncomputable def Jimin_juice : ℝ := 0.7
noncomputable def Eunji_juice : ℝ := Jimin_juice - 1/10
noncomputable def Yoongi_juice : ℝ := 4/5
noncomputable def Yuna_juice : ℝ := Jimin_juice + 0.2

theorem Yuna_drank_most :
  Yuna_juice = max (max Jimin_juice Eunji_juice) (max Yoongi_juice Yuna_juice) :=
by
  sorry

end Yuna_drank_most_l320_320018


namespace closest_point_to_origin_on_graph_l320_320937

theorem closest_point_to_origin_on_graph :
  ∃ x : ℝ, x > 0 ∧ (y = x + 1/x ∧ (x, y) = (1/real.root 4 2, (1 + real.sqrt 2)/real.root 4 2)) := sorry

end closest_point_to_origin_on_graph_l320_320937


namespace sum_of_squared_sines_equals_31_over_2_l320_320254

noncomputable def sum_of_squared_sines : ℝ :=
  (∑ n in finset.range 30, real.sin ((n + 1 : ℕ) * 6 * real.pi / 180) ^ 2)

theorem sum_of_squared_sines_equals_31_over_2 :
  sum_of_squared_sines = 31 / 2 :=
by sorry

end sum_of_squared_sines_equals_31_over_2_l320_320254


namespace number_of_even_divisors_of_factorial_eight_l320_320833

-- Definition of 8! and its prime factorization
def factorial_eight : ℕ := 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1
def prime_factorization_factorial_eight : Prop :=
  factorial_eight = 2^7 * 3^2 * 5 * 7

-- The main theorem statement
theorem number_of_even_divisors_of_factorial_eight :
  prime_factorization_factorial_eight →
  ∃ n, n = 7 * 3 * 2 * 2 ∧
  (∀ d, d ∣ factorial_eight → (∃ a b c d, 1 ≤ a ∧ a ≤ 7 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 1 ∧ 0 ≤ d ∧ d ≤ 1 ∧ d = 2^a * 3^b * 5^c * 7^d) →
  (7 * 3 * 2 * 2 = n)) :=
by
  intro h
  use 84
  sorry

end number_of_even_divisors_of_factorial_eight_l320_320833


namespace transport_cost_l320_320591

theorem transport_cost (weight_g : ℕ) (cost_per_kg : ℕ) (weight_kg : ℕ) (total_cost : ℕ)
  (h1 : weight_g = 2000)
  (h2 : cost_per_kg = 15000)
  (h3 : weight_kg = weight_g / 1000)
  (h4 : total_cost = weight_kg * cost_per_kg) :
  total_cost = 30000 :=
by
  sorry

end transport_cost_l320_320591


namespace find_B_find_ac_l320_320860

noncomputable theory

variables {A B C : ℝ} {a b c S : ℝ}

-- Given conditions
axiom sides_opposite : a = opposite(A) ∧ b = opposite(B) ∧ c = opposite(C)
axiom given1 : a + c / b = 2 * real.sin (C + real.pi / 6)
axiom given2 : b = 2 * real.sqrt 7
axiom given3 : S = 3 * real.sqrt 3

-- Prove B = π/3
theorem find_B (h : a + c / b = 2 * real.sin (C + real.pi / 6)) : B = real.pi / 3 :=
sorry

-- Prove a + c = 8 given additional conditions
theorem find_ac (h1 : b = 2 * real.sqrt 7) (h2 : S = 3 * real.sqrt 3) (h3 : B = real.pi / 3) : a + c = 8 :=
sorry

end find_B_find_ac_l320_320860


namespace max_tiles_on_floor_l320_320642

theorem max_tiles_on_floor
  (tile_w tile_h floor_w floor_h : ℕ)
  (h_tile_w : tile_w = 25)
  (h_tile_h : tile_h = 65)
  (h_floor_w : floor_w = 150)
  (h_floor_h : floor_h = 390) :
  max ((floor_h / tile_h) * (floor_w / tile_w))
      ((floor_h / tile_w) * (floor_w / tile_h)) = 36 :=
by
  -- Given conditions and calculations will be proved in the proof.
  sorry

end max_tiles_on_floor_l320_320642


namespace b_seq_arithmetic_m_inequality_l320_320808

noncomputable def a_seq (n : ℕ) : ℝ := sorry
def b_seq (n : ℕ) : ℝ := 2^(n-1) * a_seq n
def c_seq (n : ℕ) : ℝ := 1 / (b_seq n * b_seq (n + 1))
def S_seq (n : ℕ) : ℝ := (1 - (1 / (n + 1 : ℝ)))

theorem b_seq_arithmetic : ∀ n : ℕ, b_seq (n + 1) = b_seq n + 1 := sorry

theorem m_inequality (m : ℝ) : (∀ n : ℕ, m * S_seq n < n + 4 * (-1)^n) → m < -6 := sorry

end b_seq_arithmetic_m_inequality_l320_320808


namespace investment_P_l320_320058

-- Define the investments and profit sharing ratio
def investment_Q : ℕ := 15000
def profit_ratio_P_Q : ℕ × ℕ := (4, 1)

-- Prove that P's investment equals Rs 60,000
theorem investment_P :
  ∃ X : ℕ, X / investment_Q = (profit_ratio_P_Q.1 / profit_ratio_P_Q.2) ∧ X = 60000 :=
by
  use 60000
  split
  calc
    60000 / 15000 = 4 : by norm_num
  sorry

end investment_P_l320_320058


namespace min_positive_period_f_min_value_f_l320_320095

-- Define the function y = 2sin(2x + π/6) + 1
def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6) + 1

-- Statement of the problems
theorem min_positive_period_f : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi := sorry

theorem min_value_f : ∃ x, f x = -1 := sorry

end min_positive_period_f_min_value_f_l320_320095


namespace integral_eval_l320_320732

theorem integral_eval : ∫ x in (0..3), (2 * x - 1) = 6 := by
  sorry

end integral_eval_l320_320732


namespace mashas_numbers_l320_320495

def is_even (n : ℕ) : Prop := n % 2 = 0

def problem_statement (a b : ℕ) : Prop :=
  a ≠ b ∧ a > 11 ∧ b > 11 ∧ is_even a ∧ a + b = 28
  
theorem mashas_numbers : ∃ (a b : ℕ), problem_statement a b :=
by
  use 12
  use 16
  unfold problem_statement
  split
  -- a ≠ b
  exact dec_trivial
  split
  -- a > 11
  exact dec_trivial
  split
  -- b > 11
  exact dec_trivial
  split
  -- is_even a
  exact dec_trivial
  -- a + b = 28
  exact dec_trivial

end mashas_numbers_l320_320495


namespace value_of_a_l320_320561

theorem value_of_a (a b c : ℤ) (h1 : a < b) (h2 : b < c) (h3 : (a + b + c) / 3 = 4 * b) (h4 : c / b = 11) : a = 0 :=
by
  sorry

end value_of_a_l320_320561


namespace sum_sin_squared_angles_l320_320264

theorem sum_sin_squared_angles : 
  ∑ k in finset.range 30, (sin (6 * (k + 1) * (real.pi / 180)))^2 = 31 / 2 := 
sorry

end sum_sin_squared_angles_l320_320264


namespace area_of_isosceles_triangle_l320_320012

-- Define the isosceles triangle DEF with DE = DF and medians DX and FY that are perpendicular.
variables {D E F X Y : Type*}

-- Definitions for points and medians
def isosceles_triangle (D E F : Type*) : Prop :=
  dist D E = dist D F

def perpendicular (DX FY : Type*) : Prop :=
  is_perpendicular DX FY

def median (D X : Type*) : Prop :=
  dist D X = 15

def centroid_ratio (D E F G : Type*) : Prop :=
  divides_in_ratio D G 3 1

-- The main Lean 4 statement for the proof problem
theorem area_of_isosceles_triangle {D E F X Y : Type*}
  (h1 : isosceles_triangle D E F)
  (h2 : perpendicular DX FY)
  (h3 : median D X)
  (h4 : median F Y)
  (h5 : centroid_ratio D E F X Y) :
  area D E F = 1012.5 :=
sorry

end area_of_isosceles_triangle_l320_320012


namespace mashas_numbers_l320_320497

def is_even (n : ℕ) : Prop := n % 2 = 0

def problem_statement (a b : ℕ) : Prop :=
  a ≠ b ∧ a > 11 ∧ b > 11 ∧ is_even a ∧ a + b = 28
  
theorem mashas_numbers : ∃ (a b : ℕ), problem_statement a b :=
by
  use 12
  use 16
  unfold problem_statement
  split
  -- a ≠ b
  exact dec_trivial
  split
  -- a > 11
  exact dec_trivial
  split
  -- b > 11
  exact dec_trivial
  split
  -- is_even a
  exact dec_trivial
  -- a + b = 28
  exact dec_trivial

end mashas_numbers_l320_320497


namespace circles_intersection_area_l320_320143
open Real

noncomputable def area_of_intersection (theta : ℝ) (area_triangle : ℝ) : ℝ :=
  9 * theta - 2 * area_triangle

theorem circles_intersection_area (theta area_triangle : ℝ) :
  let circle1 := { center := (-1 : ℝ, 0 : ℝ), radius := 3 : ℝ }
  let circle2 := { center := (0 : ℝ, -1 : ℝ), radius := 3 : ℝ }
  area_of_intersection theta area_triangle = 9 * theta - 2 * area_triangle := 
  by
  let circle1 := { center := (-1 : ℝ, 0 : ℝ), radius := 3 : ℝ }
  let circle2 := { center := (0 : ℝ, -1 : ℝ), radius := 3 : ℝ }
  exact EQ.refl _, sorry

end circles_intersection_area_l320_320143


namespace point_B_representation_l320_320061

theorem point_B_representation : 
  ∀ A B : ℤ, A = -3 → B = A + 7 → B = 4 := 
by
  intros A B hA hB
  rw hA at hB
  rw hB
  exact rfl

end point_B_representation_l320_320061


namespace sum_of_squares_of_sines_l320_320244

theorem sum_of_squares_of_sines : 
  (\sum n in finset.range 1 30, real.sin (6 * n : ℝ) ^ 2) = 31 / 2 :=
begin
  sorry
end

end sum_of_squares_of_sines_l320_320244


namespace percentage_equivalence_l320_320183

-- Defining the constants and the question in Lean
def x := 200
def y := 100 -- This is the correct answer based on the solution
def percentage (p : ℝ) (n : ℝ) := (p / 100) * n

theorem percentage_equivalence : percentage 10 x = percentage 20 y := 
by
  -- Proof goes here
  sorry

end percentage_equivalence_l320_320183


namespace part1_part2_l320_320799

-- Define the function f
def f (x : ℝ) : ℝ := sin (3 * x) * cos (x) - cos (3 * x) * sin (x) + cos (2 * x)

-- First part: Prove that f(π/4) = 1
theorem part1 : f (π / 4) = 1 := 
by 
  sorry

-- Second part: Prove the intervals of monotonic increase
theorem part2 (k : ℤ) : k * π - 3 * π / 8 ≤ 2 * π / 8 + π / 4 ≤ k * π + π / 8 :=
by 
  sorry

end part1_part2_l320_320799


namespace aerith_successful_pair_l320_320683

theorem aerith_successful_pair :
  ∃ (l1 l2 : Char), l1 = 'o' ∧ l2 = 'e' ∧
  (∀ (n : ℕ), ∃ (a b : ℕ), a * b = n ∧ (('o' ∈ (repr a).to_list ∨ 'e' ∈ (repr a).to_list) ∧ ('o' ∈ (repr b).to_list ∨ 'e' ∈ (repr b).to_list))) := by
  sorry

end aerith_successful_pair_l320_320683


namespace find_k_values_l320_320535

theorem find_k_values (k : ℝ) :
  (let L1 := {p : ℝ × ℝ | p.1 - 2 * p.2 + 1 = 0},
       L2 := {p : ℝ × ℝ | p.1 - 1 = 0},
       L3 := {p : ℝ × ℝ | p.1 + k * p.2 = 0} in
    (∃ P, P ∈ L1 ∧ P ∈ L2 ∧ P ∈ L3) ∨
    (∀ p1 p2 ∈ ℝ, (p1 - 2 * p2 + 1 = 0) → (p1 + k * p2 = 0)) ∨
    (∀ p1 p2 ∈ ℝ, (p1 - 1 = 0) → (p1 + k * p2 = 0))
   ) →
  k = 0 ∨ k = -1 ∨ k = -2 :=
begin
  sorry
end

end find_k_values_l320_320535


namespace exists_coloring_no_ap_10_l320_320976

def no_monochromatic_ap_10 {α : Type*} (s : finset α) (coloring : α → fin 4) : Prop :=
  ∀ (a d : α), (a ∈ s) → (∀ i, (a + i * d ∈ s)) → ∃ i j, (1 ≤ i) → (i < j) → (j ≤ 9) → (coloring (a + i * d) ≠ coloring (a + j * d))

theorem exists_coloring_no_ap_10 :
  ∃ (coloring : fin 2008 → fin 4), no_monochromatic_ap_10 (finset.range 2008) coloring :=
sorry

end exists_coloring_no_ap_10_l320_320976


namespace question1_question2_question3_general_term_question3_sum_of_terms_l320_320385

noncomputable def seq (a : ℕ → ℕ) (h₁ : a 1 = 5)
  (h₂ : ∀ n : ℕ, 2 ≤ n → a n = 2 * a (n - 1) + 2^n - 1) : Prop :=
  a 2 = 13 ∧ a 3 = 33

noncomputable def lambda_check (a : ℕ → ℕ) (λ : ℝ) (h₁ : a 1 = 5)
  (h₂ : ∀ n : ℕ, 2 ≤ n → a n = 2 * a (n - 1) + 2^n - 1)
  (h₃ : ∀ n : ℕ, 2 ≤ n →
    ((λ + a (n + 1)) / 2^(n + 1)) - ((λ + a n) / 2^n) = ((λ + a 2) / 2^2) - ((λ + a 1) / 2)) :
  ℝ := λ = -1

noncomputable def general_term_formula (a : ℕ → ℕ) (h₁ : a 1 = 5)
  (h₂ : ∀ n : ℕ, 2 ≤ n → a n = 2 * a (n - 1) + 2^n - 1) (n : ℕ) : Prop :=
  a n = (n + 1) * 2^n + 1

noncomputable def sum_of_terms (a : ℕ → ℕ) (h₁ : a 1 = 5)
  (h₂ : ∀ n : ℕ, 2 ≤ n → a n = 2 * a (n - 1) + 2^n - 1) (n : ℕ) : ℕ :=
  ∑ i in range n.succ, a i = n * 2^(n + 1) + n

-- Statements to be proved:

theorem question1 : ∀ (a : ℕ → ℕ) (h₁ : a 1 = 5) (h₂ : ∀ n : ℕ, 2 ≤ n → a n = 2 * a (n - 1) + 2^n - 1),
  seq a h₁ h₂ :=
sorry

theorem question2 : ∀ (a : ℕ → ℕ) (λ : ℝ) (h₁ : a 1 = 5)
  (h₂ : ∀ n : ℕ, 2 ≤ n → a n = 2 * a (n - 1) + 2^n - 1)
  (h₃ : ∀ n : ℕ, 2 ≤ n →
    ((λ + a (n + 1)) / 2^(n + 1)) - ((λ + a n) / 2^n) = ((λ + a 2) / 2^2) - ((λ + a 1) / 2)),
  λ = -1 :=
sorry

theorem question3_general_term : ∀ (a : ℕ → ℕ) (h₁ : a 1 = 5)
  (h₂ : ∀ n : ℕ, 2 ≤ n → a n = 2 * a (n - 1) + 2^n - 1) (n : ℕ),
  a n = (n + 1) * 2^n + 1 :=
sorry

theorem question3_sum_of_terms : ∀ (a : ℕ → ℕ) (h₁ : a 1 = 5)
  (h₂ : ∀ n : ℕ, 2 ≤ n → a n = 2 * a (n - 1) + 2^n - 1) (n : ℕ),
  ∑ i in range n.succ, a i = n * 2^(n + 1) + n :=
sorry

end question1_question2_question3_general_term_question3_sum_of_terms_l320_320385


namespace case_a_second_player_wins_case_b_first_player_wins_case_c_winner_based_on_cell_color_case_d_examples_l320_320144

-- Conditions for Case (a)
def corner_cell (board : Type) (cell : board) : Prop :=
  -- definition to determine if a cell is a corner cell
  sorry

theorem case_a_second_player_wins (board : Type) (starting_cell : board) (player : ℕ) :
  corner_cell board starting_cell → 
  player = 2 :=
by
  sorry
  
-- Conditions for Case (b)
def initial_setup_according_to_figure (board : Type) (starting_cell : board) : Prop :=
  -- definition to determine if a cell setup matches the figure
  sorry

theorem case_b_first_player_wins (board : Type) (starting_cell : board) (player : ℕ) :
  initial_setup_according_to_figure board starting_cell → 
  player = 1 :=
by
  sorry

-- Conditions for Case (c)
def black_cell (board : Type) (cell : board) : Prop :=
  -- definition to determine if a cell is black
  sorry

theorem case_c_winner_based_on_cell_color (board : Type) (starting_cell : board) (player : ℕ) :
  (black_cell board starting_cell → player = 1) ∧ (¬ black_cell board starting_cell → player = 2) :=
by
  sorry
  
-- Conditions for Case (d)
def same_starting_cell_two_games (board : Type) (starting_cell : board) : Prop :=
  -- definition for same starting cell but different outcomes in games
  sorry

theorem case_d_examples (board : Type) (starting_cell : board) (player1 player2 : ℕ) :
  (same_starting_cell_two_games board starting_cell → (player1 = 1 ∧ player2 = 2)) ∨ 
  (same_starting_cell_two_games board starting_cell → (player1 = 2 ∧ player2 = 1)) :=
by
  sorry

end case_a_second_player_wins_case_b_first_player_wins_case_c_winner_based_on_cell_color_case_d_examples_l320_320144


namespace intersection_in_first_quadrant_l320_320804

theorem intersection_in_first_quadrant (k : ℝ) :
  (let x := (2 - 4 * k) / (2 * k + 1) in
   let y := (6 * k + 1) / (2 * k + 1) in
   (2 * k + 1) ≠ 0 → x > 0 ∧ y > 0) ↔ -1/6 < k ∧ k < 1/2 :=
by sorry

end intersection_in_first_quadrant_l320_320804


namespace polar_to_rectangular_intersection_PA_PB_l320_320333

-- Curve C in polar coordinates
def polar_curve (ρ θ : ℝ) : Prop :=
  ρ * (sin θ)^2 = 2 * cos θ

-- Curve C in rectangular coordinates
def rectangular_curve (x y : ℝ) : Prop :=
  y^2 = 2 * x

-- Parametric equations of line l when alpha = π/3
def line_l (t : ℝ) : ℝ × ℝ :=
  (1/2 + 1/2 * t, (sqrt 3 / 2) * t)

-- Prove the equivalence of the polar curve to the rectangular curve
theorem polar_to_rectangular (x y : ℝ) (ρ θ : ℝ) (h : polar_curve ρ θ) (hx : ρ * cos θ = x) (hy : ρ * sin θ = y) : rectangular_curve x y := by
  sorry

theorem intersection_PA_PB (α : ℝ) (A B : ℝ × ℝ) (t1 t2 : ℝ)
  (alpha_eq : α = π / 3)
  (intersects : line_l t1 = A ∧ line_l t2 = B)
  (curve_A : rectangular_curve A.1 A.2)
  (curve_B : rectangular_curve B.1 B.2)
  (P : ℝ × ℝ := (1/2, 0)) :
  |(P.1 - A.1) ^ 2 + (P.2 - A.2) ^ 2 ^ (1/2)| + |(P.1 - B.1) ^ 2 + (P.2 - B.2) ^ 2 ^ (1/2)| = 8 / 3 := by
  sorry

end polar_to_rectangular_intersection_PA_PB_l320_320333


namespace closest_point_l320_320952

noncomputable def closest_point_to_origin : ℝ × ℝ :=
  let x := (1 : ℝ) / Real.root 2 4 in
  let y := x + 1 / x in
  (x, y)

theorem closest_point (x y : ℝ) (h : y = x + 1 / x) (hx : x > 0) :
  (x, y) = closest_point_to_origin :=
begin
  sorry
end

end closest_point_l320_320952


namespace birds_in_tree_l320_320184

theorem birds_in_tree (initial_birds : ℝ) (birds_flew_away : ℝ) (h : initial_birds = 21.0) (h_flew : birds_flew_away = 14.0) : 
initial_birds - birds_flew_away = 7.0 :=
by
  -- proof goes here
  sorry

end birds_in_tree_l320_320184


namespace problem1_problem2_l320_320454

variable {a b : ℝ}

theorem problem1 (h : a > b) : a - 3 > b - 3 :=
by sorry

theorem problem2 (h : a > b) : -4 * a < -4 * b :=
by sorry

end problem1_problem2_l320_320454


namespace sum_sin_squares_deg_6_to_174_l320_320238

theorem sum_sin_squares_deg_6_to_174 : 
  ∑ k in Finset.range 30, Real.sin (6 * k * (Real.pi / 180)) ^ 2 = 15.5 := by
    sorry

end sum_sin_squares_deg_6_to_174_l320_320238


namespace division_improper_application_l320_320307

theorem division_improper_application (a b c d : ℝ) (h1 : 12.5 / 12.5 = 2.4 / 2.4)
  (h2 : 25 * (0.5 / 0.5) = 4 * (0.6 / 0.6)) :
  25 ≠ 4 :=
by
  have : 0.5 / 0.5 = 1 := by norm_num
  have : 0.6 / 0.6 = 1 := by norm_num
  simp at h2
  contradiction
  sorry

end division_improper_application_l320_320307


namespace sequence_even_indices_l320_320770

theorem sequence_even_indices (r : ℕ) (h : r > 0) :
  ∀ n : ℕ, let a : ℕ → ℕ := λ n, Nat.recOn n 1 (λ n a_n, (n * a_n + 2 * (n + 1) ^ (2 * r)) / (n + 2)) in
  (a n) % 2 = 0 ↔ n % 4 = 0 ∨ n % 4 = 3 := sorry

end sequence_even_indices_l320_320770


namespace smallest_positive_period_of_f_cos_value_when_f_neg_one_l320_320379

noncomputable def f (x : ℝ) : ℝ := (sqrt 3) * (Real.sin x) * (Real.cos x) - (Real.cos x)^2

theorem smallest_positive_period_of_f :
  ∀ (T : ℝ), (T > 0) → (∀ x : ℝ, f (x + T) = f x) → T = π :=
sorry

theorem cos_value_when_f_neg_one (x : ℝ) :
  f x = -1 → Real.cos ( (2 * Real.pi) / 3 - 2 * x ) = -1 / 2 :=
sorry

end smallest_positive_period_of_f_cos_value_when_f_neg_one_l320_320379


namespace car_time_ratio_l320_320140

theorem car_time_ratio (S V0 : ℝ) (hV0 : 0 < V0) :
  let V1 := V0 / 3 in
  let t1 := S / V1 in
  let t2 := (sum (finset.range 8) (λ i, (S / (2 ^ i) / (V0 / (2 ^ (i - 1))))) + S / V0) in
  t2 / t1 = 5 / 3 :=
by
  sorry

end car_time_ratio_l320_320140


namespace independence_iff_expectation_condition_l320_320977

open MeasureTheory

variable {α : Type*} {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) (ξ : Ω → α) (𝒢 : MeasurableSpace Ω)
variable (g : α → ℝ)

def is_independent (ξ : Ω → α) (𝒢 : MeasurableSpace Ω) : Prop :=
∀ A ∈ 𝒢.sets, ∀ B ∈ borel α, P (A ∩ ξ ⁻¹' B) = P A * P (ξ ⁻¹' B)

def expectation_condition (ξ : Ω → α) (𝒢 : MeasurableSpace Ω) (g : α → ℝ) : Prop :=
integrable g (pmap ξ 𝒢) ∧ 
∀ B ∈ borel α, ∀ A ∈ 𝒢.sets, 
integral A (g ∘ ξ) = P A * integral (g ∘ ξ)

theorem independence_iff_expectation_condition (ξ : Ω → α) (𝒢 : MeasurableSpace Ω) :
is_independent ξ 𝒢 ↔ expectation_condition ξ 𝒢 g :=
sorry

end independence_iff_expectation_condition_l320_320977


namespace non_zero_digits_of_fraction_l320_320843

def fraction : ℚ := 80 / (2^4 * 5^9)

def decimal_expansion (x : ℚ) : String :=
  -- some function to compute the decimal expansion of a fraction as a string
  "0.00000256" -- placeholder

def non_zero_digits_to_right (s : String) : ℕ :=
  -- some function to count the number of non-zero digits to the right of the decimal point in the string
  3 -- placeholder

theorem non_zero_digits_of_fraction : non_zero_digits_to_right (decimal_expansion fraction) = 3 := by
  sorry

end non_zero_digits_of_fraction_l320_320843


namespace mashas_numbers_l320_320493

def is_even (n : ℕ) : Prop := n % 2 = 0

def problem_statement (a b : ℕ) : Prop :=
  a ≠ b ∧ a > 11 ∧ b > 11 ∧ is_even a ∧ a + b = 28
  
theorem mashas_numbers : ∃ (a b : ℕ), problem_statement a b :=
by
  use 12
  use 16
  unfold problem_statement
  split
  -- a ≠ b
  exact dec_trivial
  split
  -- a > 11
  exact dec_trivial
  split
  -- b > 11
  exact dec_trivial
  split
  -- is_even a
  exact dec_trivial
  -- a + b = 28
  exact dec_trivial

end mashas_numbers_l320_320493


namespace determine_numbers_l320_320471

theorem determine_numbers (a b : ℕ) (S : ℕ) (h1 : a ≠ b) (h2 : a > 11) (h3 : b > 11)
  (h4 : S = a + b) (h5 : (∀ (x y : ℕ), x + y = S → x ≠ y → (x = a ∧ y = b) ∨ (x = b ∧ y = a)) = false)
  (h6 : a % 2 = 0 ∨ b % 2 = 0) : 
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) := 
  sorry

end determine_numbers_l320_320471


namespace max_value_approx_l320_320050

-- Define the function to be maximized
def f (x y : ℝ) : ℝ := x^2 * y / (x^4 + y^2)

-- Define the domain conditions
def domain_condition_x (x : ℝ) : Prop := 0 < x ∧ x ≤ 3/4
def domain_condition_y (y : ℝ) : Prop := 1/4 ≤ y ∧ y ≤ 2/3

-- Define the maximum value we're seeking
def maximum_value_within_domain : ℝ := 0.371

-- Statement of the problem rewritten in Lean 4
theorem max_value_approx (x y : ℝ)
  (hx : domain_condition_x x) (hy : domain_condition_y y) :
  f x y ≤ maximum_value_within_domain := sorry

end max_value_approx_l320_320050


namespace cookies_in_jar_l320_320122

theorem cookies_in_jar (C : ℕ) (h : C - 1 = (C + 5) / 2) : C = 7 :=
by
  -- Proof goes here
  sorry

end cookies_in_jar_l320_320122


namespace expand_product_l320_320733

theorem expand_product (x : ℝ) : 2 * (x - 3) * (x + 6) = 2 * x^2 + 6 * x - 36 :=
by sorry

end expand_product_l320_320733


namespace closest_point_on_graph_l320_320946

theorem closest_point_on_graph (x y : ℝ) (h1 : x > 0) (h2 : y = x + 1/x) :
  (x = 1/real.root 4 2) ∧ (y = (1 + real.sqrt 2) / real.root 4 2) :=
sorry

end closest_point_on_graph_l320_320946


namespace distance_between_parallel_lines_l320_320704

theorem distance_between_parallel_lines : 
  ∀ (A B C₁ C₂ : ℝ), (A = 1) → (B = 1) → (C₁ = -1) → (C₂ = 1) →
  (x y : ℝ), (x + y - C₁ = 0) → (x + y + C₂ = 0) →
  real.abs (C₂ - C₁) / real.sqrt (A^2 + B^2) = real.sqrt 2 :=
begin
  intros A B C₁ C₂ hA hB hC₁ hC₂ x y hxy₁ hxy₂,
  sorry
end

end distance_between_parallel_lines_l320_320704


namespace Verdi_birth_day_l320_320997

/--
Determine the day of the week on which Giuseppe Verdi was born.
Given that:
1. He was born on October 10, 1813.
2. His 170th birthday was celebrated on October 10, 1983, which was a Monday.
3. A year is a leap year if it is divisible by 400, or divisible by 4 but not by 100.

Prove that Giuseppe Verdi was born on a Sunday.
-/
theorem Verdi_birth_day
  (dob : Day := { year := 1813, month := 10, day := 10 })
  (birthday_1983 : Day := { year := 1983, month := 10, day := 10 })
  (birthday_1983_is_monday : dayOfWeek birthday_1983 = DayOfWeek.monday)
  (leap_year : ∀ y : ℕ, (y % 400 = 0 ∨ (y % 4 = 0 ∧ y % 100 ≠ 0)) ↔ is_leap_year y)
  : dayOfWeek dob = DayOfWeek.sunday :=
by 
  sorry

end Verdi_birth_day_l320_320997


namespace quadratic_inequality_solution_l320_320409

theorem quadratic_inequality_solution (a : ℝ) :
  (∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) ↔ (a < -1 ∨ a > 3) :=
by
  sorry

end quadratic_inequality_solution_l320_320409


namespace ellipse_equation_and_max_AB_l320_320339

-- Definitions for the ellipse and the point conditions
def ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

def point_p := (-Real.sqrt 3, 1/2 : ℝ)
def focus := (Real.sqrt 3, 0 : ℝ)
def a := 2
def b := 1

-- Maximum value of |AB|
def max_AB := (5 * Real.sqrt 6) / 6

-- The statement to prove the given question
theorem ellipse_equation_and_max_AB :
  (ellipse a b = {p : ℝ × ℝ | (p.1^2 / 4) + (p.2^2) = 1}) ∧
  ∀ (l : ℝ → (ℝ × ℝ) → Prop),
    (∀ M ∈ {M | M = (0, Real.sqrt 2)}, l M) → -- Line passes through M(0, sqrt(2))
    -- Maximum value of |AB| as stated in the problem
    ∃ A B : ℝ × ℝ, 
       A ≠ B ∧ A ∈ ellipse a b ∧ B ∈ ellipse a b ∧  
       (line l A B M) →
       |AB| ≤  max_AB :=
by
  sorry

end ellipse_equation_and_max_AB_l320_320339


namespace sin_pi_plus_alpha_l320_320324

theorem sin_pi_plus_alpha (α : ℝ) (h₁ : 0 < α ∧ α < π / 2)
    (h₂ : cos (π / 3 + α) = 1 / 3) :
    sin (π + α) = (sqrt 3 - 2 * sqrt 2) / 6 := by
  sorry

end sin_pi_plus_alpha_l320_320324


namespace area_of_circle_with_given_circumference_l320_320621

-- Defining the given problem's conditions as variables
variables (C : ℝ) (r : ℝ) (A : ℝ)
  
-- The condition that circumference is 12π meters
def circumference_condition : Prop := C = 12 * Real.pi
  
-- The relationship between circumference and radius
def radius_relationship : Prop := C = 2 * Real.pi * r
  
-- The formula to calculate the area of the circle
def area_formula : Prop := A = Real.pi * r^2
  
-- The proof goal that we need to establish
theorem area_of_circle_with_given_circumference :
  circumference_condition C ∧ radius_relationship C r ∧ area_formula A r → A = 36 * Real.pi :=
by
  intros
  sorry -- Skipping the proof, to be done later

end area_of_circle_with_given_circumference_l320_320621


namespace bead_count_l320_320920

variable (total_beads : ℕ) (blue_beads : ℕ) (red_beads : ℕ) (white_beads : ℕ) (silver_beads : ℕ)

theorem bead_count : total_beads = 40 ∧ blue_beads = 5 ∧ red_beads = 2 * blue_beads ∧ white_beads = blue_beads + red_beads ∧ silver_beads = total_beads - (blue_beads + red_beads + white_beads) → silver_beads = 10 :=
by
  intro h
  sorry

end bead_count_l320_320920


namespace mutual_choice_exists_l320_320538

theorem mutual_choice_exists (n : ℕ) (hn : n = 100) 
  (f g : fin n → fin n) 
  (hf : ∀ i j : fin n, i ≤ j → f i ≤ f j)
  (hg : ∀ i j : fin n, i ≤ j → g i ≤ g j) : 
  ∃ k : fin n, f (g k) = k := 
by
  sorry

end mutual_choice_exists_l320_320538


namespace sum_first_five_even_numbers_l320_320603

theorem sum_first_five_even_numbers : (2 + 4 + 6 + 8 + 10) = 30 :=
by
  sorry

end sum_first_five_even_numbers_l320_320603


namespace find_numbers_l320_320501

theorem find_numbers (a b : ℕ) (h1 : a > 11) (h2 : b > 11) (h3 : a ≠ b)
  (h4 : (∃ S, S = a + b) ∧ (∀ (x y : ℕ), x ≠ y → x + y = a + b → (x > 11) → (y > 11) → ¬(x = a ∨ y = a) → ¬(x = b ∨ y = b)))
  (h5 : even a ∨ even b) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) :=
by
  sorry

end find_numbers_l320_320501


namespace a5_plus_a6_l320_320334

def sequence_sum (n : ℕ) : ℕ := n^2

theorem a5_plus_a6 :
  let S := sequence_sum in
  S 6 - S 4 = 20 :=
by
  sorry

end a5_plus_a6_l320_320334


namespace sufficient_but_not_necessary_condition_not_necessary_condition_sufficient_condition_l320_320462

theorem sufficient_but_not_necessary_condition {x y : ℝ} (h : x > y > 0) : x / y > 1 :=
by
  sorry

theorem not_necessary_condition {x y : ℝ} (h : x / y > 1) : ¬(x > y > 0) :=
by
  sorry

theorem sufficient_condition :
  (∀ x y : ℝ, x > y > 0 → x / y > 1) ∧ ¬(∀ x y : ℝ, x / y > 1 → x > y > 0) :=
by
  exact ⟨sufficient_but_not_necessary_condition, not_necessary_condition⟩

end sufficient_but_not_necessary_condition_not_necessary_condition_sufficient_condition_l320_320462


namespace reachable_points_infinite_grid_l320_320278

theorem reachable_points_infinite_grid :
  ∀ (x y : ℤ), (∃ steps : ℕ, (x, y) = steps.foldl (λ (pos : ℤ × ℤ) _, match pos with
    | (a, b) => if even (a + b) then
      [(a + 1, b + 1), (a + 1, b - 1), (a - 1, b + 1), (a - 1, b - 1)].head!
    else pos end) (0, 0)) → even (x + y) :=
by
  sorry

end reachable_points_infinite_grid_l320_320278


namespace part1_part2_l320_320320

-- Define what it means to be a value-preserving interval
def is_value_preserving_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  a < b ∧ (∀ x y, a ≤ x ∧ x ≤ b ∧ a ≤ y ∧ y ≤ b → f(x) = y → (x = a ∨ x = b ∨ (a < x ∧ x < b)))

-- Prove that the interval [0, 1/2] is a value-preserving interval for y = 2x^2
theorem part1 : is_value_preserving_interval (λ x, 2 * x^2) 0 (1/2) := sorry

-- Prove that the function y = x^2 - 2x + m has a value-preserving interval if m ∈ [1, 5/4) ∪ [2, 9/4)
theorem part2 (m : ℝ) : (m ∈ set.Ico 1 (5/4) ∨ m ∈ set.Ico 2 (9/4)) →
  ∃ a b, is_value_preserving_interval (λ x, x^2 - 2 * x + m) a b := sorry

end part1_part2_l320_320320


namespace length_AF_l320_320004

open Real

theorem length_AF
  (AB BC: ℝ) 
  (H_AB: AB = 24) 
  (H_BC: BC = 7)
  (F: ℝ × ℝ) 
  (CF: ℝ) 
  (H_CF: angle (complex.I) (exp (-22.5 * complex.I * PI / 180)) = 22.5) :
  norm (complex.ofReal (24 - F.1) + complex.I * (7 - F.2)) = 7 * sqrt (2 + sqrt 2) :=
sorry

end length_AF_l320_320004


namespace sum_of_squared_sines_equals_31_over_2_l320_320258

noncomputable def sum_of_squared_sines : ℝ :=
  (∑ n in finset.range 30, real.sin ((n + 1 : ℕ) * 6 * real.pi / 180) ^ 2)

theorem sum_of_squared_sines_equals_31_over_2 :
  sum_of_squared_sines = 31 / 2 :=
by sorry

end sum_of_squared_sines_equals_31_over_2_l320_320258


namespace difference_between_numbers_l320_320114

theorem difference_between_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 190) : x - y = 19 :=
by sorry

end difference_between_numbers_l320_320114


namespace curve_not_conic_section_l320_320904

theorem curve_not_conic_section (z : ℂ) (h : abs (z - 1/z) = 1) :
  ¬(is_ellipse z ∨ is_parabola z ∨ is_hyperbola z) := 
sorry

def is_ellipse (z : ℂ) : Prop := sorry
def is_parabola (z : ℂ) : Prop := sorry
def is_hyperbola (z : ℂ) : Prop := sorry

end curve_not_conic_section_l320_320904


namespace masha_numbers_l320_320522

theorem masha_numbers (a b : ℕ) (S : ℕ) (h1 : a ≠ b) (h2 : a > 11) (h3 : b > 11) (h4 : S = a + b) 
    (h5 : (∀ x y : ℕ, x + y = S → x = a ∨ y = a → abs x - y = a) ∧ (even a ∨ even b)) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) := 
by sorry

end masha_numbers_l320_320522


namespace hexagon_sides_equality_proof_l320_320549

noncomputable
def hexagon_opposite_sides_equal (A B C D E F : Point) (inscribed_circle_center : Point) : Prop :=
  let opposite_sides := 
    (distance A B = distance D E) ∧ 
    (distance B C = distance E F) ∧ 
    (distance C D = distance F A)
  ∧ symmetric_center inscribed_circle_center A D
  ∧ symmetric_center inscribed_circle_center B E
  ∧ symmetric_center inscribed_circle_center C F
  ∧ symmetric_center inscribed_circle_center D A
  ∧ symmetric_center inscribed_circle_center E B
  ∧ symmetric_center inscribed_circle_center F C
  in opposite_sides

-- Proof statement
theorem hexagon_sides_equality_proof (A B C D E F : Point) (inscribed_circle_center : Point) :
  hexagon_opposite_sides_equal A B C D E F inscribed_circle_center :=
by sorry

end hexagon_sides_equality_proof_l320_320549


namespace root_in_interval_l320_320082

open Real

def f (x : ℝ) : ℝ := 2^x + 3 * x - 4

theorem root_in_interval :
  ∃ x, 0 < x ∧ x < 1 ∧ f x = 0 :=
by
  -- Use the given conditions to state the problem
  have h₀ : f 0 < 0 := by norm_num
  have h₁ : f 1 > 0 := by norm_num
  -- Intermediate Value Theorem
  exact exists_root_intermediate_value 0 1 h₀ h₁

# The proof steps are not required, hence adding sorry.

end root_in_interval_l320_320082


namespace michelle_silver_beads_l320_320926

theorem michelle_silver_beads :
  ∀ (total_beads blue_beads red_beads white_beads silver_beads : ℕ),
    total_beads = 40 →
    blue_beads = 5 →
    red_beads = 2 * blue_beads →
    white_beads = blue_beads + red_beads →
    silver_beads = total_beads - (blue_beads + red_beads + white_beads) →
    silver_beads = 10 :=
by {
  intros total_beads blue_beads red_beads white_beads silver_beads,
  assume h1 h2 h3 h4 h5,
  sorry
}

end michelle_silver_beads_l320_320926


namespace find_numbers_l320_320505

theorem find_numbers (a b : ℕ) (h1 : a > 11) (h2 : b > 11) (h3 : a ≠ b)
  (h4 : (∃ S, S = a + b) ∧ (∀ (x y : ℕ), x ≠ y → x + y = a + b → (x > 11) → (y > 11) → ¬(x = a ∨ y = a) → ¬(x = b ∨ y = b)))
  (h5 : even a ∨ even b) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) :=
by
  sorry

end find_numbers_l320_320505


namespace find_largest_number_l320_320401

theorem find_largest_number (a : ℤ) (h : a = -3) : 
  (9 : ℤ) ∈ { -3 * a, 5 * a, 18 / a, a * a, 1 } ∧
  ∀ x ∈ { -3 * a, 5 * a, 18 / a, a * a, 1 }, x ≤ 9 := 
by
  sorry

end find_largest_number_l320_320401


namespace closest_point_to_origin_l320_320943

def y (x : ℝ) := x + 1 / x

theorem closest_point_to_origin : ∃ x : ℝ, x > 0 ∧ (x, y x) = (1 / 2^(1/4 : ℝ), (1 + real.sqrt 2) / 2^(1/4 : ℝ)) :=
by
  sorry

end closest_point_to_origin_l320_320943


namespace find_inverse_of_CD_mod_1000000_l320_320451

theorem find_inverse_of_CD_mod_1000000 :
  ∃ M : ℕ, (M < 1000000) ∧ (M * (123456 * 142857) % 1000000 = 1) ∧ (M = 814815) :=
by
  -- noncomputable theory
  -- Let C = 123456 and D = 142857
  let C : ℕ := 123456
  let D : ℕ := 142857

  -- Calculation to find the multiplicative inverse of CD mod 1000000
  have H1 : 123456 * D % 1000000 = 17172832 % 1000000 := by sorry
  have H2 : 17172832 % 1000000 = 814815 := by sorry

  use 814815
  split
  · -- Prove that M < 1000000
    exact Nat.lt_of_sub_eq 814815 rfl

  · split
    · -- Prove that M * (C * D) % 1000000 = 1
      calc
        814815 * (123456 * 142857) % 1000000 
        = 814815 * 17172832 % 1000000 := by rw H1
        = 814815 * C * D % 1000000 := by sorry  -- some simplification step
        = 1 := by sorry

    · -- Prove equality to the correct answer
      rfl

end find_inverse_of_CD_mod_1000000_l320_320451


namespace closest_point_l320_320954

noncomputable def closest_point_to_origin : ℝ × ℝ :=
  let x := (1 : ℝ) / Real.root 2 4 in
  let y := x + 1 / x in
  (x, y)

theorem closest_point (x y : ℝ) (h : y = x + 1 / x) (hx : x > 0) :
  (x, y) = closest_point_to_origin :=
begin
  sorry
end

end closest_point_l320_320954


namespace max_non_overlapping_cross_shapes_l320_320186

-- Definition of cross shape and board size.
def cross_shape : Fin 5 × Fin 5 → Prop :=
λ ⟨x, y⟩, (x = 0 ∧ y ≠ 0 ∧ y ≠ 4) ∨ (x = 4 ∧ y ≠ 0 ∧ y ≠ 4) ∨ (y = 0 ∧ x ≠ 0 ∧ x ≠ 4) ∨ (y = 4 ∧ x ≠ 0 ∧ x ≠ 4) ∨ (x > 0 ∧ x < 4 ∧ y > 0 ∧ y < 4)

def chessboard : Fin 10 × Fin 11 → Prop := λ _, True

-- Maximum number of cross shapes on the board.
theorem max_non_overlapping_cross_shapes : 
  ∃ max_shapes : ℕ, (∀ placement : (Fin 10 × Fin 11 → Prop), 
  (∀ p, placement p → cross_shape p) →  (∃ n ≤ max_shapes, ∀ i < n, ∃ c, placement c)) ∧ max_shapes = 15 :=
sorry

end max_non_overlapping_cross_shapes_l320_320186


namespace masha_numbers_l320_320524

theorem masha_numbers (a b : ℕ) (S : ℕ) (h1 : a ≠ b) (h2 : a > 11) (h3 : b > 11) (h4 : S = a + b) 
    (h5 : (∀ x y : ℕ, x + y = S → x = a ∨ y = a → abs x - y = a) ∧ (even a ∨ even b)) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) := 
by sorry

end masha_numbers_l320_320524


namespace ratio_of_hexagons_l320_320190

theorem ratio_of_hexagons (r : ℝ) (h_inscribed : hexagon_inscribed r) (h_circumscribed : hexagon_circumscribed r) :
  area h_circumscribed / area h_inscribed = 3 := 
sorry

end ratio_of_hexagons_l320_320190


namespace odd_divisors_lt_100_l320_320394

theorem odd_divisors_lt_100 : 
  let n := 100
  let perfect_squares := { k : ℕ // k < n ∧ ∃ m : ℕ, k = m * m }
  ∃ (count : ℕ), (count = perfect_squares.card) ∧ (count = 9) :=
by
  sorry

end odd_divisors_lt_100_l320_320394


namespace closest_point_to_origin_on_graph_l320_320934

theorem closest_point_to_origin_on_graph :
  ∃ x : ℝ, x > 0 ∧ (y = x + 1/x ∧ (x, y) = (1/real.root 4 2, (1 + real.sqrt 2)/real.root 4 2)) := sorry

end closest_point_to_origin_on_graph_l320_320934


namespace second_car_time_ratio_l320_320138

-- Definitions for the problem
variables {V_0 V_1 S : ℝ}

-- Conditions
def speed_relation (V_1 V_0 : ℝ) := V_1 = V_0 / 3
def distance_relation (S : ℝ) : Prop := S > 0
def time_first_car (V_1 S : ℝ) := S / V_1
def time_second_car (V_0 S : ℝ) := 5 * (S / V_0)

-- Statement of the proof problem
theorem second_car_time_ratio (h1 : speed_relation V_1 V_0) (h2 : distance_relation S) :
  (time_second_car V_0 S) / (time_first_car V_1 S) = 5 / 3 := 
sorry

end second_car_time_ratio_l320_320138


namespace determine_b_div_a_l320_320764

noncomputable def f (a b x : ℝ) : ℝ := x^3 + a * x^2 + b * x - a^2 - 7 * a

theorem determine_b_div_a
  (a b : ℝ)
  (hf_deriv : ∀ x : ℝ, (deriv (f a b)) x = 3 * x^2 + 2 * a * x + b)
  (hf_max : f a b 1 = 10)
  (hf_deriv_at_1 : (deriv (f a b)) 1 = 0) :
  b / a = -3 / 2 :=
sorry

end determine_b_div_a_l320_320764


namespace compute_LM_length_l320_320866

-- Definitions of lengths and equidistant property
variables (GH JK LM : ℝ) 
variables (equidistant : GH * 2 = 120 ∧ JK * 2 = 80)
variables (parallel : GH = 120 ∧ JK = 80 ∧ GH = JK)

-- State the theorem to prove lengths
theorem compute_LM_length (GH JD LM : ℝ) (equidistant : GH * 2 = 120 ∧ JK * 2 = 80)
  (parallel : GH = 120 ∧ JK = 80 ∧ GH = JK) :
  LM = (2 / 3) * 80 := 
sorry

end compute_LM_length_l320_320866


namespace lindy_distance_traveled_l320_320166

/-- Jack and Christina are standing 240 feet apart on a level surface. 
Jack walks in a straight line toward Christina at a constant speed of 5 feet per second. 
Christina walks in a straight line toward Jack at a constant speed of 3 feet per second. 
Lindy runs at a constant speed of 9 feet per second from Christina to Jack, back to Christina, back to Jack, and so forth. 
The total distance Lindy travels when the three meet at one place is 270 feet. -/
theorem lindy_distance_traveled
    (initial_distance : ℝ)
    (jack_speed : ℝ)
    (christina_speed : ℝ)
    (lindy_speed : ℝ)
    (time_to_meet : ℝ)
    (total_distance_lindy : ℝ) :
    initial_distance = 240 ∧
    jack_speed = 5 ∧
    christina_speed = 3 ∧
    lindy_speed = 9 ∧
    time_to_meet = (initial_distance / (jack_speed + christina_speed)) ∧
    total_distance_lindy = lindy_speed * time_to_meet →
    total_distance_lindy = 270 :=
by
  sorry

end lindy_distance_traveled_l320_320166


namespace odd_expression_is_odd_l320_320984

theorem odd_expression_is_odd (p q : ℤ) (hp : p % 2 = 1) (hq : q % 2 = 1) : (4 * p * q + 1) % 2 = 1 :=
sorry

end odd_expression_is_odd_l320_320984


namespace true_propositions_b_l320_320998

theorem true_propositions_b :
  (¬ (∀ a b : ℝ, (a ≤ b) → (a < b)) ∧
   (∀ a : ℝ, (a = 1) → (∀ x : ℝ, (a*x^2 - x + 3 ≥ 0)) ∧
   (∀ C₁ C₂ : ℝ, (C₁ = C₂) → (C₁ = C₂)) ∧
   ¬ ∀ x : ℝ, (rational (sqrt 2 * x)) → (irrational x))) :=
begin
  sorry
end

end true_propositions_b_l320_320998


namespace no_adjacent_A_B_l320_320973

theorem no_adjacent_A_B (A B : Type) (other_people : Fin 5 → Type) :
  let total_people := other_people.to_list ++ [A, B] in
  let total_arrangements := (∏ x in (Finset.univ : Finset (Fin 5)), fintype.card (other_people x)) * 
                            (∏ x in (Finset.univ : Finset (Fin 6)), if x = 0 then fintype.card A else if x = 1 then fintype.card B else 1) in
  total_arrangements = 3600 :=
by sorry

end no_adjacent_A_B_l320_320973


namespace find_functions_l320_320740

noncomputable def pair_of_functions_condition (f g : ℝ → ℝ) : Prop :=
∀ x y : ℝ, g (f (x + y)) = f x + (2 * x + y) * g y

theorem find_functions (f g : ℝ → ℝ) :
  pair_of_functions_condition f g →
  (∃ c d : ℝ, ∀ x : ℝ, f x = c * (x + d)) :=
sorry

end find_functions_l320_320740


namespace missing_digit_B_l320_320130

theorem missing_digit_B :
  ∃ B : ℕ, 0 ≤ B ∧ B ≤ 9 ∧ (200 + 10 * B + 5) % 13 = 0 := 
sorry

end missing_digit_B_l320_320130


namespace expected_value_best_of_seven_games_correct_l320_320987

noncomputable def expected_value_best_of_seven_games : ℚ :=
  4 * (1 / 8) + 5 * (1 / 4) + 6 * (5 / 16) + 7 * (5 / 16)

theorem expected_value_best_of_seven_games_correct :
  expected_value_best_of_seven_games = 93 / 16 :=
by
  sorry

end expected_value_best_of_seven_games_correct_l320_320987


namespace moles_of_HC2H3O2_needed_l320_320313

theorem moles_of_HC2H3O2_needed :
  (∀ (HC2H3O2 NaHCO3 H2O : ℕ), 
    (HC2H3O2 + NaHCO3 = NaC2H3O2 + H2O + CO2) → 
    (H2O = 3) → 
    (NaHCO3 = 3) → 
    HC2H3O2 = 3) :=
by
  intros HC2H3O2 NaHCO3 H2O h_eq h_H2O h_NaHCO3
  -- Hint: You can use the balanced chemical equation to derive that HC2H3O2 must be 3
  sorry

end moles_of_HC2H3O2_needed_l320_320313


namespace sum_sin_squared_angles_l320_320267

theorem sum_sin_squared_angles : 
  ∑ k in finset.range 30, (sin (6 * (k + 1) * (real.pi / 180)))^2 = 31 / 2 := 
sorry

end sum_sin_squared_angles_l320_320267


namespace solve_quadratic_inequality_l320_320077

theorem solve_quadratic_inequality (a x : ℝ) :
  (x ^ 2 - (2 + a) * x + 2 * a < 0) ↔ 
  ((a < 2 ∧ a < x ∧ x < 2) ∨ (a = 2 ∧ false) ∨ 
   (a > 2 ∧ 2 < x ∧ x < a)) :=
by sorry

end solve_quadratic_inequality_l320_320077


namespace right_triangle_cotangent_l320_320423

theorem right_triangle_cotangent
  (A B C : Point)
  (h : ∠ A B C = 90°)
  (AC BC : ℝ)
  (hAC : AC = 3)
  (hBC : BC = 4) :
  Real.cot (∠ A B C) = 3 / 4 := by
  sorry

end right_triangle_cotangent_l320_320423


namespace initial_money_was_600_l320_320284

variable (M : ℝ)
variable (initial_money : ℝ)

-- Conditions
def spent_on_gas (init_money : ℝ) := init_money * (1 / 3)
def remaining_after_gas (init_money : ℝ) := init_money * (2 / 3)
def spent_on_food (remaining_money : ℝ) := remaining_money * (1 / 4)
def remaining_after_food (init_money : ℝ) := (remaining_after_gas init_money) * (3 / 4)

-- Given condition
def remaining_money_condition := remaining_after_food initial_money = 300

-- Goal to prove
theorem initial_money_was_600 (h : remaining_money_condition) : initial_money = 600 :=
by
  sorry

end initial_money_was_600_l320_320284


namespace range_of_f_ge_one_l320_320796

def f (x : ℝ) : ℝ :=
  if x < 1 then abs (x + 1) else -x + 3

theorem range_of_f_ge_one (x : ℝ) :
  f x ≥ 1 ↔ (x ≤ -2 ∨ (0 ≤ x ∧ x ≤ 2)) := by
  sorry

end range_of_f_ge_one_l320_320796


namespace min_value_cos_sin_cos_sin_minimum_l320_320312

theorem min_value_cos_sin (θ : ℝ) (hθ : 0 < θ ∧ θ < π) : 
  ∃ x, x = cos (θ / 2) * (2 - sin θ) ∧ ∀ y, y = cos (θ / 2) * (2 - sin θ) → x ≤ y := 
begin
  sorry
end

theorem cos_sin_minimum (θ : ℝ) (hθ : 0 < θ ∧ θ < π) :
  ∃ x, x = cos (θ / 2) * (2 - sin θ) ∧ x = 0 := 
by 
  have := min_value_cos_sin θ hθ,
  sorry

end min_value_cos_sin_cos_sin_minimum_l320_320312


namespace no_primes_in_Q_plus_m_l320_320031

def Q : ℕ := 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem no_primes_in_Q_plus_m (m : ℕ) (hm : 2 ≤ m ∧ m ≤ 32) : ¬is_prime (Q + m) :=
by
  sorry  -- Proof would be provided here

end no_primes_in_Q_plus_m_l320_320031


namespace eq_of_nonzero_real_x_l320_320291

theorem eq_of_nonzero_real_x (x : ℝ) (hx : x ≠ 0) (a b : ℝ) (ha : a = 9) (hb : b = 18) :
  ((a * x) ^ 10 = (b * x) ^ 5) → x = 2 / 9 :=
by
  sorry

end eq_of_nonzero_real_x_l320_320291


namespace circle_inside_triangle_l320_320066

-- Define the problem conditions
def triangle_sides : ℕ × ℕ × ℕ := (3, 4, 5)
def circle_area : ℚ := 25 / 8

-- Define the problem statement
theorem circle_inside_triangle (a b c : ℕ) (area : ℚ)
    (h1 : (a, b, c) = triangle_sides)
    (h2 : area = circle_area) :
    ∃ r R : ℚ, R < r ∧ 2 * r = a + b - c ∧ R^2 = area / π := sorry

end circle_inside_triangle_l320_320066


namespace mike_spent_on_mower_blades_l320_320051

theorem mike_spent_on_mower_blades (x : ℝ) 
  (initial_money : ℝ := 101) 
  (cost_of_games : ℝ := 54) 
  (games : ℝ := 9) 
  (price_per_game : ℝ := 6) 
  (h1 : 101 - x = 54) :
  x = 47 := 
by
  sorry

end mike_spent_on_mower_blades_l320_320051


namespace arithmetic_seq_a12_l320_320338

theorem arithmetic_seq_a12 (a : ℕ → ℝ) (d : ℝ)
  (h1 : a 4 = 1)
  (h2 : a 7 + a 9 = 16)
  (h3 : ∀ n, a n = a 1 + (n - 1) * d) :
  a 12 = 15 :=
by sorry

end arithmetic_seq_a12_l320_320338


namespace smallest_b_for_six_points_in_rectangle_l320_320985

theorem smallest_b_for_six_points_in_rectangle :
  ∃ b : ℝ, (∀ (points : fin 6 → ℝ × ℝ), (∀ i, 0 ≤ (points i).1 ∧ (points i).1 ≤ 2 
  ∧ 0 ≤ (points i).2 ∧ (points i).2 ≤ 1)
  → (∃ i j : fin 6, i ≠ j ∧ dist (points i) (points j) ≤ b)) ∧ b = (real.sqrt 5) / 2 :=
by
  sorry

end smallest_b_for_six_points_in_rectangle_l320_320985


namespace tangent_y_intercept_l320_320755

def curve (x : ℝ) := x^2 + 11

def point_of_tangency : ℝ × ℝ := (1, 12)

theorem tangent_y_intercept :
  let m := (deriv curve) 1 in
  let tangent_line (x : ℝ) := m * (x - (point_of_tangency.1)) + (point_of_tangency.2) in
  tangent_line 0 = 10 :=
by
  sorry

end tangent_y_intercept_l320_320755


namespace payment_required_l320_320192

-- Definitions of the conditions
def price_suit : ℕ := 200
def price_tie : ℕ := 40
def num_suits : ℕ := 20
def discount_option_1 (x : ℕ) (hx : x > 20) : ℕ := price_suit * num_suits + (x - num_suits) * price_tie
def discount_option_2 (x : ℕ) (hx : x > 20) : ℕ := (price_suit * num_suits + x * price_tie) * 9 / 10

-- Theorem that needs to be proved
theorem payment_required (x : ℕ) (hx : x > 20) :
  discount_option_1 x hx = 40 * x + 3200 ∧ discount_option_2 x hx = 3600 + 36 * x :=
by sorry

end payment_required_l320_320192


namespace directrix_of_parabola_l320_320098

noncomputable def m : ℝ := -3 / 2

def point : ℝ × ℝ := (-2, real.sqrt 3)

def parabola (m : ℝ) := ∀ (x y : ℝ), y^2 = m * x → (x, y) = point

theorem directrix_of_parabola (m : ℝ) (h : parabola m) : (m = -3/2) → ∃ x : ℝ, x = 3 / 8 :=
by
  intro h_m
  use 3 / 8
  exact sorry

end directrix_of_parabola_l320_320098


namespace determine_q_l320_320999

-- Lean 4 statement
theorem determine_q (a : ℝ) (q : ℝ → ℝ) :
  (∀ x, q x = a * (x + 2) * (x - 3)) ∧ q 1 = 8 →
  q x = - (4 / 3) * x ^ 2 + (4 / 3) * x + 8 := 
sorry

end determine_q_l320_320999


namespace cosine_product_value_l320_320634

theorem cosine_product_value :
  (cos (Real.pi / 15) * cos (2 * Real.pi / 15) * cos (3 * Real.pi / 15) *
   cos (4 * Real.pi / 15) * cos (5 * Real.pi / 15) * cos (6 * Real.pi / 15) *
   cos (7 * Real.pi / 15)) = -1 / 128 :=
by sorry

end cosine_product_value_l320_320634


namespace radius_of_tangent_sphere_l320_320768

-- Define the vertices A and B of the unit cube
def A : ℝ × ℝ × ℝ := (0, 0, 0)
def B : ℝ × ℝ × ℝ := (1, 1, 1)

-- Define the set of conditions for the cube and the sphere
def unit_cube : set (ℝ × ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1 ∧ 0 ≤ p.3 ∧ p.3 ≤ 1}

def is_center_inside_cube (c : ℝ × ℝ × ℝ) :=
  c ∈ unit_cube

def is_tangent_to_faces_at_A (c : ℝ × ℝ × ℝ) (r : ℝ) : Prop :=
  c = (r, r, r)

def is_tangent_to_edges_at_B (c : ℝ × ℝ × ℝ) (r : ℝ) : Prop :=
  abs (1 - c.1) = r ∧ abs (1 - c.2) = r ∧ abs (1 - c.3) = r

-- Proposition to prove
theorem radius_of_tangent_sphere :
  ∃ (c : ℝ × ℝ × ℝ) (r : ℝ),
  is_center_inside_cube c ∧ 
  is_tangent_to_faces_at_A c r ∧
  is_tangent_to_edges_at_B c r ∧
  r = 1 / 2 :=
by
  apply Exists.intro (1 / 2, 1 / 2, 1 / 2),
  apply Exists.intro (1 / 2),
  split,
  {
    dsimp [unit_cube],
    repeat {split},
    all_goals {norm_num},
  },
  split,
  {
    dsimp [is_tangent_to_faces_at_A],
    norm_num,
  },
  split,
  {
    dsimp [is_tangent_to_edges_at_B],
    repeat {split},
    all_goals {norm_num},
  },
  {
    norm_num
  }

  sorry

end radius_of_tangent_sphere_l320_320768


namespace dad_vacuum_time_l320_320016

theorem dad_vacuum_time (x : ℕ) (h1 : 2 * x + 5 = 27) (h2 : x + (2 * x + 5) = 38) :
  (2 * x + 5) = 27 := by
  sorry

end dad_vacuum_time_l320_320016


namespace ratio_of_spiders_l320_320210

theorem ratio_of_spiders (legs_per_spider : ℕ) (total_legs : ℕ) (S : ℕ) (H1 : legs_per_spider = 8) 
(H2 : total_legs = 112) (H3 : 8 * S = total_legs) :
  S : (legs_per_spider / 2) = 7 : 2 :=
by
  sorry

end ratio_of_spiders_l320_320210


namespace seq_transformation_l320_320759

variable (a : ℕ → ℝ)
variable (M : ℝ)

def seq_gt (a : ℕ → ℝ) (M : ℝ) : Prop :=
  ∀ n, a n ≥ M ∨ a (n + 1) ≥ M

theorem seq_transformation (h : seq_gt a M) : seq_gt (λ n, 2 * (a n) + 1) (2 * M + 1) :=
  sorry

end seq_transformation_l320_320759


namespace find_vector_at_zero_l320_320201

def vector_at_t (a d : ℝ × ℝ × ℝ) (t : ℝ) : ℝ × ℝ × ℝ :=
  (a.1 + t * d.1, a.2 + t * d.2, a.3 + t * d.3)

theorem find_vector_at_zero :
  ∃(a d : ℝ × ℝ × ℝ),
    vector_at_t a d 1 = (2, 4, 9) ∧
    vector_at_t a d (-1) = (-1, 1, 2) ∧
    vector_at_t a d 0 = (1/2, 5/2, 11/2) :=
by
  -- Proof omitted
  sorry

end find_vector_at_zero_l320_320201


namespace express_n_as_sum_of_squares_l320_320897

theorem express_n_as_sum_of_squares (N : ℕ) (a b c : ℤ) (ha: a % 3 = 0) (hb : b % 3 = 0) (hc : c % 3 = 0) (hN : N = a^2 + b^2 + c^2) :
  ∃ x y z : ℤ, ¬ (x % 3 = 0) ∧ ¬ (y % 3 = 0) ∧ ¬ (z % 3 = 0) ∧ N = x^2 + y^2 + z^2 :=
begin
  sorry
end

end express_n_as_sum_of_squares_l320_320897


namespace sequence_periodic_l320_320386

def sequence (a : ℕ → ℚ) : Prop :=
a 1 = 2 ∧ ∀ n, a (n + 1) = -1 / (a n + 1)

-- The theorem we want to prove
theorem sequence_periodic (a : ℕ → ℚ) (h : sequence a) : a 2001 = -3/2 :=
by
  sorry

end sequence_periodic_l320_320386


namespace depth_of_water_in_second_cylinder_l320_320195

-- Define the radii and heights of both cylinders
def r1 : ℝ := 2
def h1 : ℝ := 8
def r2 : ℝ := 4

-- Define the volume formula for a cylinder
def volume_cylinder (r h : ℝ) : ℝ := π * r^2 * h

-- Calculate the volume of the first cylinder
def V1 : ℝ := volume_cylinder r1 h1

-- Define the volume of water after pouring into the second cylinder
def poured_water_volume : ℝ := V1

-- Given the radius of the second cylinder, solve for the height of water in the second cylinder
def height_of_water_in_second_cylinder : ℝ := poured_water_volume / (π * r2^2)

theorem depth_of_water_in_second_cylinder :
  height_of_water_in_second_cylinder = 2 :=
by
  sorry

end depth_of_water_in_second_cylinder_l320_320195


namespace even_divisors_of_8fac_l320_320821

theorem even_divisors_of_8fac : 
  let num_even_divisors := ∏ x in {a | 1 ≤ a ∧ a ≤ 7}.card * 
                                      {b | 0 ≤ b ∧ b ≤ 2}.card *
                                      {c | 0 ≤ c ∧ c ≤ 1}.card *
                                      {d | 0 ≤ d ∧ d ≤ 1}.card
  in num_even_divisors = 84 := by
  sorry

end even_divisors_of_8fac_l320_320821


namespace closest_point_on_graph_l320_320949

theorem closest_point_on_graph (x y : ℝ) (h1 : x > 0) (h2 : y = x + 1/x) :
  (x = 1/real.root 4 2) ∧ (y = (1 + real.sqrt 2) / real.root 4 2) :=
sorry

end closest_point_on_graph_l320_320949


namespace range_a_l320_320372

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then (1 / 2) * x - (1 / 2)
  else Real.log x

theorem range_a (a : ℝ) : 
  (f (f a) = Real.log (f a)) ↔ (a ∈ Set.Ici Real.exp 1) :=
sorry

end range_a_l320_320372


namespace angle_sum_proof_l320_320425

theorem angle_sum_proof (A B C x y : ℝ) 
  (hA : A = 35) 
  (hB : B = 65) 
  (hC : C = 40) 
  (hx : x = 130 - C)
  (hy : y = 90 - A) :
  x + y = 140 := by
  sorry

end angle_sum_proof_l320_320425


namespace find_numbers_l320_320400

noncomputable def first_number_x : ℝ :=
  let x := 160 in x

noncomputable def second_number_y : ℝ :=
  let y := 520 / 3 in y

theorem find_numbers (x y : ℝ) 
  (h1 : 0.35 * x = 0.50 * x - 24)
  (h2 : 0.30 * y = 0.55 * 160 - 36) :
  (x = first_number_x) ∧ (y = second_number_y) :=
by
  sorry

end find_numbers_l320_320400


namespace geometry_correct_choice_l320_320701

theorem geometry_correct_choice :
  ∀ (P Q R S : Type) [line P] [plane Q] [line R] [plane S],
  (∀ (l1 l2 : line P), (l1 ⊥ R → l2 ⊥ R → l1 || l2) = false) ∧
  (∀ (l1 l2 : line Q), (l1 ⊥ Q → l2 ⊥ Q → l1 || l2) = true) ∧
  (∀ (p1 p2 : plane R), (p1 ⊥ R → p2 ⊥ R → p1 || p2) = true) ∧
  (∀ (p1 p2 : plane S), (p1 ⊥ S → p2 ⊥ S → p1 || p2) = false) →
  (correct_choice = B) :=
by sorry

end geometry_correct_choice_l320_320701


namespace distance_from_dormitory_to_city_l320_320641

theorem distance_from_dormitory_to_city (D : ℝ)
  (h1 : (1 / 5) * D + (2 / 3) * D + 4 = D) : D = 30 := by
  sorry

end distance_from_dormitory_to_city_l320_320641


namespace shortest_altitude_l320_320583

theorem shortest_altitude (a b c : ℝ) (ha : a = 13) (hb : b = 14) (hc : c = 15) : 
  ∃ h, h = 11.2 ∧ h = 2 * (sqrt (( (a+b+c)/2) * ( (a+b+c)/2 - a) * ( (a+b+c)/2 - b) * ( (a+b+c)/2 - c))) / c :=
by sorry

end shortest_altitude_l320_320583


namespace count_ordered_pairs_l320_320710

def is_real (z: ℂ) : Prop :=
  z.im = 0

noncomputable def count_pairs (x y : ℕ) (hx : 1 ≤ x) (hy : x < y) (hz : y ≤ 200) : Prop :=
  is_real (complex.abs (complex.I ^ x + complex.I ^ y)) ∧ 
  int.cast (complex.abs (complex.I ^ x + complex.I ^ y)) % 5 = 0

theorem count_ordered_pairs : (∑ x in finset.Icc 1 199, ∑ y in finset.Icc (x+1) 200, if count_pairs x y (by linarith) (by linarith) (by linarith) then 1 else 0) = 2352 :=
sorry

end count_ordered_pairs_l320_320710


namespace right_triangle_cos_l320_320421

theorem right_triangle_cos (A B C : Type) [right_triangle A B C]
  (AB AC : ℝ) (h1 : AB = 3) (h2 : AC = 5) :
  cosine_angle A B = 3 / 5 :=
by
  sorry

end right_triangle_cos_l320_320421


namespace range_of_m_l320_320346

theorem range_of_m (m : ℝ) (a : ℝ) :
  (∃ A B : ℝ × ℝ, (A.1 - 2)^2 + (A.2 - m)^2 = 4 ∧ (B.1 - 2)^2 + (B.2 - m)^2 = 4 ∧ |A.1 - B.1, A.2 - B.2| = 2 * sqrt 3) →
  (∃ P : ℝ × ℝ, P = (((a * P.2) : ℝ), (P.1 - 2) * a - 2 * a + 4 = 0) ∧ 
                   (P.1 - 2)^2 + (P.2 + 1)^2 = 5 ∧ 
                   P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) →
  (sqrt 5 - 2) ≤ m ∧ m ≤ sqrt 5 :=
begin
  sorry
end

end range_of_m_l320_320346


namespace sequence_value_l320_320806

theorem sequence_value {a : ℕ → ℝ} 
  (h1 : a 1 = 2)
  (h2 : ∀ n, a (n + 1) = a n + real.log (↑(n + 1) / ↑n) / real.log 2) :
  a 8 = 5 := 
sorry

end sequence_value_l320_320806


namespace exists_positive_int_for_nonneg_poly_l320_320455

theorem exists_positive_int_for_nonneg_poly (p : Polynomial ℝ) (h : ∀ x : ℝ, 0 ≤ x → 0 < p.eval x) :
  ∃ n : ℕ, ∀ x : ℝ, Polynomial.eval (Polynomial.mul (Polynomial.pow (Polynomial.X + 1) n) p) x ≥ 0 := 
by 
  sorry

end exists_positive_int_for_nonneg_poly_l320_320455


namespace second_car_time_ratio_l320_320139

-- Definitions for the problem
variables {V_0 V_1 S : ℝ}

-- Conditions
def speed_relation (V_1 V_0 : ℝ) := V_1 = V_0 / 3
def distance_relation (S : ℝ) : Prop := S > 0
def time_first_car (V_1 S : ℝ) := S / V_1
def time_second_car (V_0 S : ℝ) := 5 * (S / V_0)

-- Statement of the proof problem
theorem second_car_time_ratio (h1 : speed_relation V_1 V_0) (h2 : distance_relation S) :
  (time_second_car V_0 S) / (time_first_car V_1 S) = 5 / 3 := 
sorry

end second_car_time_ratio_l320_320139


namespace minimum_employees_same_zodiac_minimum_employees_same_birthday_l320_320637

-- Statement for the first part
theorem minimum_employees_same_zodiac (num_signs : ℕ) (num_employees : ℕ) : 
  num_signs = 12 → 
  num_employees = 13 → 
  ∃ (signs : Finset ℕ) (employees : Finset ℕ), 
    signs.card = num_signs ∧ 
    employees.card = num_employees ∧ 
    ∃ (sign : ℕ), sign ∈ signs ∧ 2 ≤ (employees.filter (λ e, e = sign)).card :=
by
  intros hsigns hemployees
  sorry

-- Statement for the second part
theorem minimum_employees_same_birthday (num_days : ℕ) (num_employees : ℕ) : 
  num_days = 7 → 
  num_employees = 22 → 
  ∃ (days : Finset ℕ) (employees : Finset (ℕ × ℕ)), 
    days.card = num_days ∧ 
    employees.card = num_employees ∧ 
    ∃ (day : ℕ), day ∈ days ∧ 4 ≤ (employees.filter (λ e, e.2 = day)).card :=
by
  intros hdays hemployees
  sorry

end minimum_employees_same_zodiac_minimum_employees_same_birthday_l320_320637


namespace sum_of_squares_of_sines_l320_320240

theorem sum_of_squares_of_sines : 
  (\sum n in finset.range 1 30, real.sin (6 * n : ℝ) ^ 2) = 31 / 2 :=
begin
  sorry
end

end sum_of_squares_of_sines_l320_320240


namespace find_a_for_power_function_l320_320572

theorem find_a_for_power_function : 
  ∃ a : ℝ, ∀ x : ℝ, (f : ℝ → ℝ := λ x, x^a), f 4 = 1 / 2 :=
sorry

end find_a_for_power_function_l320_320572


namespace shortest_altitude_13_14_15_l320_320586

noncomputable def shortest_altitude_in_triangle (a b c : ℕ) : ℝ :=
  let s := (a + b + c) / 2 in
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c)) in
  min (2 * area / a) (min (2 * area / b) (2 * area / c))

theorem shortest_altitude_13_14_15 : 
  shortest_altitude_in_triangle 13 14 15 = 11.2 := 
sorry

end shortest_altitude_13_14_15_l320_320586


namespace calculate_womans_haircut_cost_l320_320081

-- Define the necessary constants and conditions
def W : ℝ := sorry
def child_haircut_cost : ℝ := 36
def tip_percentage : ℝ := 0.20
def total_tip : ℝ := 24
def number_of_children : ℕ := 2

-- Helper function to calculate total cost before the tip
def total_cost_before_tip (W : ℝ) (number_of_children : ℕ) (child_haircut_cost : ℝ) : ℝ :=
  W + number_of_children * child_haircut_cost

-- Lean statement for the main theorem
theorem calculate_womans_haircut_cost (W : ℝ) (child_haircut_cost : ℝ) (tip_percentage : ℝ)
  (total_tip : ℝ) (number_of_children : ℕ) :
  (tip_percentage * total_cost_before_tip W number_of_children child_haircut_cost) = total_tip →
  W = 48 :=
by
  sorry

end calculate_womans_haircut_cost_l320_320081


namespace sum_of_squared_sines_equals_31_over_2_l320_320257

noncomputable def sum_of_squared_sines : ℝ :=
  (∑ n in finset.range 30, real.sin ((n + 1 : ℕ) * 6 * real.pi / 180) ^ 2)

theorem sum_of_squared_sines_equals_31_over_2 :
  sum_of_squared_sines = 31 / 2 :=
by sorry

end sum_of_squared_sines_equals_31_over_2_l320_320257


namespace mashas_numbers_l320_320494

def is_even (n : ℕ) : Prop := n % 2 = 0

def problem_statement (a b : ℕ) : Prop :=
  a ≠ b ∧ a > 11 ∧ b > 11 ∧ is_even a ∧ a + b = 28
  
theorem mashas_numbers : ∃ (a b : ℕ), problem_statement a b :=
by
  use 12
  use 16
  unfold problem_statement
  split
  -- a ≠ b
  exact dec_trivial
  split
  -- a > 11
  exact dec_trivial
  split
  -- b > 11
  exact dec_trivial
  split
  -- is_even a
  exact dec_trivial
  -- a + b = 28
  exact dec_trivial

end mashas_numbers_l320_320494


namespace problem1_problem2_l320_320708

theorem problem1 : (-2)^2 - Real.sqrt 4 + Real.cbrt 27 = 5 :=
by sorry

theorem problem2 : (-Real.sqrt 5)^2 - Real.cbrt ((-8)^2) + Real.sqrt ((-3)^2) = 4 :=
by sorry

end problem1_problem2_l320_320708


namespace Masha_thought_of_numbers_l320_320516

theorem Masha_thought_of_numbers : ∃ a b : ℕ, a ≠ b ∧ a > 11 ∧ b > 11 ∧ (a + b = 28) ∧ (a % 2 = 0 ∨ b % 2 = 0) ∧ (a = 12 ∧ b = 16 ∨ a = 16 ∧ b = 12) :=
by
  sorry

end Masha_thought_of_numbers_l320_320516


namespace sum_sin_squares_deg_6_to_174_l320_320236

theorem sum_sin_squares_deg_6_to_174 : 
  ∑ k in Finset.range 30, Real.sin (6 * k * (Real.pi / 180)) ^ 2 = 15.5 := by
    sorry

end sum_sin_squares_deg_6_to_174_l320_320236


namespace repeating_decimal_as_fraction_l320_320734

theorem repeating_decimal_as_fraction : (3 + 167 / 999 : ℚ) = 3164 / 999 := 
by sorry

end repeating_decimal_as_fraction_l320_320734


namespace time_to_destination_l320_320159

-- Variables and Conditions
variable (Harris_speed : ℝ)  -- Speed of Mr. Harris (in units of distance per hour)
variable (your_speed : ℝ)  -- Your speed (in units of distance per hour)
variable (time_Harris : ℝ) -- Time taken by Mr. Harris to reach the store (in hours)
variable (distance_ratio : ℝ) -- Ratio of your destination distance to the store distance

-- Hypotheses based on given conditions
hypothesis h1 : your_speed = 2 * Harris_speed
hypothesis h2 : time_Harris = 2
hypothesis h3 : distance_ratio = 3

-- Statement to prove
theorem time_to_destination : (3 / 2 : ℝ) = 3 := by 
    sorry

end time_to_destination_l320_320159


namespace kiera_total_envelopes_l320_320444

theorem kiera_total_envelopes (B Y G T : ℕ) (hB : B = 120) (hY : Y = B - 25) (hG : G = 5 * Y) (hT : T = B + Y + G) : T = 690 :=
by {
  rw hB at hY,
  rw hB at hT,
  rw hY at hG,
  rw hY at hT,
  rw hG at hT,
  norm_num at hT,
  exact hT,
}

end kiera_total_envelopes_l320_320444


namespace marcy_votes_correct_l320_320466

-- Definition of variables based on the conditions
def joey_votes : ℕ := 8
def barry_votes : ℕ := 2 * (joey_votes + 3)
def marcy_votes : ℕ := 3 * barry_votes

-- The main statement to prove
theorem marcy_votes_correct : marcy_votes = 66 := 
by 
  sorry

end marcy_votes_correct_l320_320466


namespace monotonic_intervals_of_f_maximum_k_exists_x0_for_a_l320_320378

-- Definition of the function f
def f (x : ℝ) : ℝ := Real.log (x + 1) - x

-- Proof Problem 1: Monotonic intervals of f(x)
theorem monotonic_intervals_of_f :
  (∀ x ∈ Ioo (-1 : ℝ) 0, f' x > 0) ∧ (∀ x ∈ Ioo (0 : ℝ) ∞, f' x < 0) :=
sorry

-- Proof Problem 2: Maximum value of k given the condition
theorem maximum_k :
  ∀ k : ℤ, (∀ x > 1, f (x - 1) + x > k * (1 - 3 / x)) → k ≤ 4 :=
sorry

-- Proof Problem 3: Existence of x0 for given a
theorem exists_x0_for_a (a : ℝ) (ha : 0 < a ∧ a < 1) :
  ∃ x0 > 0, Real.exp (f x0) < 1 - a / 2 * x0 ^ 2 :=
sorry

end monotonic_intervals_of_f_maximum_k_exists_x0_for_a_l320_320378


namespace part1_tangent_line_eq_part2_monotonicity_part3_max_value_l320_320798

noncomputable def f (x a : ℝ) : ℝ := log (x + 1) - (a * x ^ 2 + x) / (1 + x) ^ 2
noncomputable def g (x : ℝ) : ℝ := (1 + 1 / x) ^ x + (1 + x) ^ (1 / x)

-- Part 1: Equation of the tangent line at x = e - 1 when a = 1.
theorem part1_tangent_line_eq (e : ℝ) (h_e_pos : 0 < e) : 
  (∀ (x : ℝ), f x 1 = log (x + 1) - x / (1 + x)) →
  ∀ (x : ℝ), x = e - 1 → 
  (∀ y, y = f (e - 1) 1 + (f' (e - 1) 1) * (x - (e - 1))
  )
:= sorry

-- Part 2: Monotonicity of f(x) when 2/3 < a ≤ 2.
theorem part2_monotonicity (a : ℝ) (h_a_range : 2 / 3 < a ∧ a ≤ 2) :
  ∀ x, 
  (f' x a) > 0 ↔ (-1 < x ∧ x < 0) ∨ (2 * a - 3 < x) ∧ 
  (f' x a < 0 ↔ 0 < x ∧ x < 2 * a - 3
  )
:= sorry

-- Part 3: Maximum value of g(x) for x > 0.
theorem part3_max_value (x : ℝ) (h_x_pos : 0 < x) :
  (∀ y, g x = e^y) →
  g x ≤ 4 :=
sorry

end part1_tangent_line_eq_part2_monotonicity_part3_max_value_l320_320798


namespace even_divisors_of_8fac_l320_320819

theorem even_divisors_of_8fac : 
  let num_even_divisors := ∏ x in {a | 1 ≤ a ∧ a ≤ 7}.card * 
                                      {b | 0 ≤ b ∧ b ≤ 2}.card *
                                      {c | 0 ≤ c ∧ c ≤ 1}.card *
                                      {d | 0 ≤ d ∧ d ≤ 1}.card
  in num_even_divisors = 84 := by
  sorry

end even_divisors_of_8fac_l320_320819


namespace simplify_log_expression_l320_320979

theorem simplify_log_expression : 
  (2 * log 4 3 + log 8 3) * (log 3 2 + log 9 2) = 2 := 
by 
  sorry

end simplify_log_expression_l320_320979


namespace thousandth_term_of_coprime_sequence_l320_320305

theorem thousandth_term_of_coprime_sequence:
  let a := 105
  let seq_nth_coprime (n : Nat) := (@List.filter Nat (λ x => Nat.gcd x a = 1) (List.range ((n+1)*a)))
    nth n
  in seq_nth_coprime 999 = 2186 := sorry

end thousandth_term_of_coprime_sequence_l320_320305


namespace extremum_at_three_l320_320330

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  (x^2 + a*x + b) * Real.exp (3 - x)

theorem extremum_at_three (a b : ℝ) (h : 3 = 3) :
  -- Part (Ⅰ.1): Prove relationship between a and b
  (∃ b, b = -3 - 2 * a) ∧ 
  -- Part (Ⅰ.2): Prove intervals of monotonicity
  ((a < -4 → ∀ x, (3 < x ∧ x < -a-1 → f x a b > 0) ∧ (x < 3 ∨ x > -a-1 → f x a b < 0)) ∧
   (a = -4 → ∀ x, f x a b ≤ 0) ∧
   (a > -4 → ∀ x, (-a-1 < x ∧ x < 3 → f x a b > 0) ∧ (x < -a-1 ∨ x > 3 → f x a b < 0))) ∧ 
  -- Part (Ⅱ): Prove range of f(x) on [0, 4] when a > 0
  (a > 0 → ∀ x, 0 ≤ x ∧ x ≤ 4 → f x a b ∈ Icc (-(2 * a + 3) * Real.exp 3) (a + 6)) :=
  sorry

end extremum_at_three_l320_320330


namespace masha_numbers_l320_320491

theorem masha_numbers {a b : ℕ} (h1 : a ≠ b) (h2 : 11 < a) (h3 : 11 < b) 
  (h4 : ∃ S, S = a + b ∧ (∀ x y, x + y = S → x ≠ y → 11 < x ∧ 11 < y → 
       (¬(x = a ∧ y = b) ∧ ¬(x = b ∧ y = a)))) 
  (h5 : even a ∨ even b)
  (h6 : ∀ x y, (even x ∨ even y) → x ≠ y → 11 < x ∧ 11 < y ∧ x + y = a + b → 
       x = a ∧ y = b ∨ x = b ∧ y = a) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) :=
by
  sorry

end masha_numbers_l320_320491


namespace min_value_reciprocal_l320_320328

variable {a b : ℝ}

theorem min_value_reciprocal (h1 : a * b > 0) (h2 : a + 4 * b = 1) : 
  ∃ (a b : ℝ), (a > 0) ∧ (b > 0) ∧ ((1/a) + (1/b) = 9) := 
by
  sorry

end min_value_reciprocal_l320_320328


namespace exist_equidistant_plane_l320_320342

variable (A B C D E F : Point) -- assuming Point is a type representing points in space

theorem exist_equidistant_plane :
  ∃ (P : Plane), equidistant P (A, B, C, D, E, F) ∧ separates P {A, B, C} {D, E, F} :=
by
  sorry

end exist_equidistant_plane_l320_320342


namespace soap_last_duration_l320_320467

-- Definitions of the given conditions
def cost_per_bar := 8 -- cost in dollars
def total_spent := 48 -- total spent in dollars
def months_in_year := 12

-- Definition of the query statement/proof goal
theorem soap_last_duration (h₁ : total_spent = 48) (h₂ : cost_per_bar = 8) (h₃ : months_in_year = 12) : months_in_year / (total_spent / cost_per_bar) = 2 :=
by 
  -- Placeholder for the proof
  sorry

end soap_last_duration_l320_320467


namespace closest_point_to_origin_on_graph_l320_320933

theorem closest_point_to_origin_on_graph :
  ∃ x : ℝ, x > 0 ∧ (y = x + 1/x ∧ (x, y) = (1/real.root 4 2, (1 + real.sqrt 2)/real.root 4 2)) := sorry

end closest_point_to_origin_on_graph_l320_320933


namespace shortest_altitude_13_14_15_l320_320585

noncomputable def shortest_altitude_in_triangle (a b c : ℕ) : ℝ :=
  let s := (a + b + c) / 2 in
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c)) in
  min (2 * area / a) (min (2 * area / b) (2 * area / c))

theorem shortest_altitude_13_14_15 : 
  shortest_altitude_in_triangle 13 14 15 = 11.2 := 
sorry

end shortest_altitude_13_14_15_l320_320585


namespace find_original_radius_l320_320725

-- Definitions based on the conditions in the problem
def volume_with_increased_radius (r : ℝ) : ℝ := π * (r + 7)^2 * 3
def volume_with_increased_height (r : ℝ) : ℝ := π * r^2 * 10
def original_height : ℝ := 3

theorem find_original_radius (r : ℝ) (h1 : volume_with_increased_radius r = volume_with_increased_height r) : r = 7 :=
by
  sorry

end find_original_radius_l320_320725


namespace salary_percentage_change_l320_320443

theorem salary_percentage_change (S : ℝ) : 
  let S1 := S * 0.5,
      S2 := S1 * 1.3,
      S3 := S2 * 0.8,
      S4 := S3 * 0.9,
      S5 := S4 * 1.25 in
  ((S5 - S) / S) * 100 = -41.5 :=
by
  let S1 := S * 0.5
  let S2 := S1 * 1.3
  let S3 := S2 * 0.8
  let S4 := S3 * 0.9
  let S5 := S4 * 1.25
  sorry

end salary_percentage_change_l320_320443


namespace Michelle_silver_beads_count_l320_320922

theorem Michelle_silver_beads_count 
  (total_beads : ℕ) (blue_beads : ℕ) (red_beads : ℕ) (white_beads : ℕ) (silver_beads : ℕ)
  (h1 : total_beads = 40)
  (h2 : blue_beads = 5)
  (h3 : red_beads = 2 * blue_beads)
  (h4 : white_beads = blue_beads + red_beads)
  (h5 : silver_beads = total_beads - (blue_beads + red_beads + white_beads)) :
  silver_beads = 10 :=
by
  sorry

end Michelle_silver_beads_count_l320_320922


namespace sum_of_sin_squared_angles_l320_320250

theorem sum_of_sin_squared_angles :
  (∑ i in Finset.range 30, Real.sin (6 * (i + 1)) * Real.sin (6 * (i + 1))) = 15.5 :=
sorry

end sum_of_sin_squared_angles_l320_320250


namespace arithmetic_sequence_a_value_l320_320369

theorem arithmetic_sequence_a_value :
  ∀ (a : ℤ), (-7) - a = a - 1 → a = -3 :=
by
  intro a
  intro h
  sorry

end arithmetic_sequence_a_value_l320_320369


namespace instantaneous_rate_of_change_at_t0_plus_1_l320_320788

noncomputable def h (t : ℝ) : ℝ := (2 / 3) * t^3 + 3 * t^2

def height_rate_of_change (t : ℝ) : ℝ := (deriv h) t

variable (t0 : ℝ)
variable (h'_at_t0 : height_rate_of_change t0 = 8)

theorem instantaneous_rate_of_change_at_t0_plus_1 :
  height_rate_of_change (t0 + 1) = 20 := by
  sorry

end instantaneous_rate_of_change_at_t0_plus_1_l320_320788


namespace identified_rectangle_perimeter_l320_320672

-- Define the side length of the square
def side_length_mm : ℕ := 75

-- Define the heights of the rectangles
variables (x y z : ℕ)

-- Define conditions
def rectangles_cut_condition (x y z : ℕ) : Prop := x + y + z = side_length_mm
def perimeter_relation_condition (x y z : ℕ) : Prop := 2 * (x + side_length_mm) = (y + side_length_mm) + (z + side_length_mm)

-- Define the perimeter of the identified rectangle
def identified_perimeter_mm (x : ℕ) := 2 * (x + side_length_mm)

-- Define conversion from mm to cm
def mm_to_cm (mm : ℕ) : ℕ := mm / 10

-- Final proof statement
theorem identified_rectangle_perimeter :
  ∃ x y z : ℕ, rectangles_cut_condition x y z ∧ perimeter_relation_condition x y z ∧ mm_to_cm (identified_perimeter_mm x) = 20 := 
sorry

end identified_rectangle_perimeter_l320_320672


namespace fifty_percent_greater_than_88_l320_320415

theorem fifty_percent_greater_than_88 :
  ∃ x : ℝ, x = 88 + 0.5 * 88 ∧ x = 132 :=
begin
  use (88 + 0.5 * 88),
  split,
  { refl, },
  { norm_num, },
end

end fifty_percent_greater_than_88_l320_320415


namespace fraction_of_married_men_l320_320865

theorem fraction_of_married_men (total_women : ℕ) (prob_single : ℝ) (total_people : ℕ) (married_men : ℕ) :
  total_women = 7 ∧ prob_single = 3 / 7 ∧ total_people = 11 ∧ married_men = 4 →
  (married_men : ℝ) / (total_people : ℝ) = 4 / 11 :=
by
  intros h,
  cases h with hw hrest,
  cases hrest with hps hrest,
  cases hrest with htp hm,
  rw [hm, htp],
  exact rfl

end fraction_of_married_men_l320_320865


namespace largest_reciprocal_l320_320628

theorem largest_reciprocal :
  ∀ (x ∈ {1 / 4, 3 / 7, 2, 7, 1000}), 1 / x ≤ 4 → x = 1 / 4 :=
by {
  intros x hx h,
  sorry
}

end largest_reciprocal_l320_320628


namespace no_sum_of_three_squares_2015_l320_320975

theorem no_sum_of_three_squares_2015 :
  ¬ ∃ (x y z : ℤ), x^2 + y^2 + z^2 = 2015 := by
  -- We'll need to show that 2015 is congruent to 7 modulo 8.
  have h1 : 2015 % 8 = 7 := by norm_num,

  -- We need to note the possible values of squares modulo 8.
  have h2 : ∀ (n : ℤ), n^2 % 8 = 0 ∨ n^2 % 8 = 1 ∨ n^2 % 8 = 4 := by
    intro n,
    fin_cases n % 8,
    norm_num,
    norm_num,
    norm_num,
    norm_num,
    norm_num,
    norm_num,
    norm_num,
    norm_num,
 
  -- Now we need to use these possible values to demonstrate that the sum cannot be 7.
  have h3 : ∀ x y z : ℤ, (x^2 % 8 + y^2 % 8 + z^2 % 8) ∈ {0, 1, 2, 3, 4, 5, 6} := by
    intros,
    rcases h2 x with ⟨hx0|hx1|hx4⟩,
    rcases h2 y with ⟨hy0|hy1|hy4⟩,
    rcases h2 z with ⟨hz0|hz1|hz4⟩,
    all_goals {
      norm_num at *,
      exact hx0,
      exact hx1,
      exact hx4,
    },

  -- Finally, we conclude that since 7 ∉ {0, 1, 2, 3, 4, 5, 6}, the sum of three squares cannot be 2015.
  intro h,
  rcases h with ⟨x, y, z, hsum⟩,
  have hmod := congr_arg (λ t : ℤ, t % 8) hsum,
  rw [← int.add_mod, ← int.add_mod, h2 x, h2 y, h2 z, h3, h1] at hmod,
  norm_num at hmod,
  exact hmod

end no_sum_of_three_squares_2015_l320_320975


namespace smallest_card_A_union_B_l320_320972

theorem smallest_card_A_union_B (A B : Type) (hA : Fintype.card A = 30) (hB : Fintype.card B = 25) (h_disjoint : ∀ a ∈ A, ∀ b ∈ B, a ≠ b)  : Fintype.card (A ∪ B) = 55 :=
by
  sorry

end smallest_card_A_union_B_l320_320972


namespace time_to_empty_tank_l320_320679

def cubic_feet_to_cubic_inches(feet : ℕ) : ℕ :=
  feet * 1728

def inlet_flow_rate : ℕ := 5
def outlet_A_flow_rate : ℕ := 9
def outlet_B_flow_rate : ℕ := 8
def outlet_C_flow_rate : ℕ := 7
def outlet_D_flow_rate : ℕ := 6

def net_flow_rate_minute1 : ℕ := inlet_flow_rate - outlet_A_flow_rate - outlet_B_flow_rate
def net_flow_rate_minute2 : ℕ := inlet_flow_rate - outlet_C_flow_rate - outlet_D_flow_rate

def total_over_two_minutes : ℤ := net_flow_rate_minute1 + net_flow_rate_minute2
def tank_volume_cubic_feet : ℕ := 20

/-- Given the conditions, prove that the tank will empty in 3456 minutes. -/
theorem time_to_empty_tank :
  let tank_volume_cubic_inches := cubic_feet_to_cubic_inches tank_volume_cubic_feet in
  tank_volume_cubic_inches = 34560 ∧
  total_over_two_minutes = -20 →
  34560 / (-total_over_two_minutes) * 2 = 3456 :=
  by
  intros tank_volume_cubic_inches
  intros tank_volume_cubic_inches_eq_and_total_over_two_minutes_eq
  have h1 := tank_volume_cubic_inches_eq_and_total_over_two_minutes_eq.1
  have h2 := tank_volume_cubic_inches_eq_and_total_over_two_minutes_eq.2
  rw [h1, h2]
  norm_num
  sorry

end time_to_empty_tank_l320_320679


namespace seven_pow_eight_mod_100_l320_320170

theorem seven_pow_eight_mod_100 :
  (7 ^ 8) % 100 = 1 := 
by {
  -- here can be the steps of the proof, but for now we use sorry
  sorry
}

end seven_pow_eight_mod_100_l320_320170


namespace value_of_e_is_91_l320_320164

noncomputable def value_of_e (a b c d e : ℤ) (k : ℤ) : Prop :=
  a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧ d % 2 = 1 ∧ e % 2 = 1 ∧
  b = a + 2 * k ∧ c = a + 4 * k ∧ d = a + 6 * k ∧ e = a + 8 * k ∧
  a + c = 146 ∧ k > 0 ∧ 2 * k ≥ 4 ∧ k ≠ 2

theorem value_of_e_is_91 (a b c d e k : ℤ) (h : value_of_e a b c d e k) : e = 91 :=
  sorry

end value_of_e_is_91_l320_320164


namespace calculate_length_of_bridge_l320_320680

/-- Define the conditions based on given problem -/
def length_of_bridge (speed1 speed2 : ℕ) (length1 length2 : ℕ) (time : ℕ) : ℕ :=
    let distance_covered_train1 := speed1 * time
    let bridge_length_train1 := distance_covered_train1 - length1
    let distance_covered_train2 := speed2 * time
    let bridge_length_train2 := distance_covered_train2 - length2
    max bridge_length_train1 bridge_length_train2

/-- Given conditions -/
def speed_train1 := 15 -- in m/s
def length_train1 := 130 -- in meters
def speed_train2 := 20 -- in m/s
def length_train2 := 90 -- in meters
def crossing_time := 30 -- in seconds

theorem calculate_length_of_bridge : length_of_bridge speed_train1 speed_train2 length_train1 length_train2 crossing_time = 510 :=
by
  -- omitted proof
  sorry

end calculate_length_of_bridge_l320_320680


namespace min_value_inv_sum_l320_320783

theorem min_value_inv_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 1 / b) ≥ 4 :=
begin
  sorry, -- proof omitted
end

end min_value_inv_sum_l320_320783


namespace petri_dishes_count_l320_320878

theorem petri_dishes_count (germs_total : ℕ) (germs_per_dish : ℕ) (condition1 : germs_total = 0.037 * 100000) (condition2 : germs_per_dish = 50) :
  (germs_total / germs_per_dish) = 74 :=
by {
  sorry
}

end petri_dishes_count_l320_320878


namespace masha_numbers_l320_320485

theorem masha_numbers {a b : ℕ} (h1 : a ≠ b) (h2 : 11 < a) (h3 : 11 < b) 
  (h4 : ∃ S, S = a + b ∧ (∀ x y, x + y = S → x ≠ y → 11 < x ∧ 11 < y → 
       (¬(x = a ∧ y = b) ∧ ¬(x = b ∧ y = a)))) 
  (h5 : even a ∨ even b)
  (h6 : ∀ x y, (even x ∨ even y) → x ≠ y → 11 < x ∧ 11 < y ∧ x + y = a + b → 
       x = a ∧ y = b ∨ x = b ∧ y = a) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) :=
by
  sorry

end masha_numbers_l320_320485


namespace surface_area_of_double_cone_volume_of_double_cone_l320_320989

-- Since we are dealing with non-computable operations
noncomputable theory

def = 3.14159

def area_right_triangle : ℝ := 121.5

-- Convert angle from degrees, minutes, seconds to degrees
def angle_deg : ℝ := 36 + (52 / 60) + (10.7 / 3600)

-- Expected surface area of the double cone
def expected_surface_area : ℝ := 1068.75

-- Expected volume of the double cone
def expected_volume : ℝ := 2748.2

-- The sin and cos values for the given angle
def sin_alpha : ℝ := 0.6 -- approximation of sin(36.8696°)
def cos_alpha : ℝ := 0.8 -- approximation of cos(36.8696°)

theorem surface_area_of_double_cone
  (t : ℝ := area_right_triangle)
  (α : ℝ := angle_deg)
  (π : ℝ := 3.14159) :
  2 * t * π * (sin_alpha + cos_alpha) = expected_surface_area :=
by
  sorry

theorem volume_of_double_cone
  (t : ℝ := area_right_triangle)
  (α : ℝ := angle_deg)
  (π : ℝ := 3.14159) :
  (2 * π * t / 3) * real.sqrt (t * real.sin (2 * α)) = expected_volume :=
by
  sorry

-- Use the Lean mathlib library for trigonometric functions

end surface_area_of_double_cone_volume_of_double_cone_l320_320989


namespace incorrect_multiplicative_inverse_product_l320_320630

theorem incorrect_multiplicative_inverse_product:
  ∃ (a b : ℝ), a + b = 0 ∧ a * b ≠ 1 :=
by
  sorry

end incorrect_multiplicative_inverse_product_l320_320630


namespace hexagon_sqrt7_distance_l320_320205

-- Definition of a point
structure Point where
  x : ℝ
  y : ℝ

-- Distance between two points in the plane
def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Definition of a regular hexagon with side length 1
structure RegularHexagon where
  vertices : fin 6 → Point
  side_length : ∀ i, distance (vertices i) (vertices ((i + 1) % 6)) = 1

-- The theorem we want to prove
theorem hexagon_sqrt7_distance :
  ∀ (hex : RegularHexagon),
    ∃ (p1 p2 : Point),
      (distance p1 p2 = real.sqrt 7) ∧
      (∃ (lines : list (Point × Point)), (∀ (p : Point), p ∈ [p1, p2] → p ∈ hex.vertices.to_list ∨
        ∃ (line ∈ lines), p ∈ line)
      ) : sorry

end hexagon_sqrt7_distance_l320_320205


namespace snakes_in_breeding_ball_l320_320048

theorem snakes_in_breeding_ball (x : ℕ) (h : 3 * x + 12 = 36) : x = 8 :=
by sorry

end snakes_in_breeding_ball_l320_320048


namespace lambda_range_l320_320322

noncomputable def vector1 := (1 : ℝ, 3 : ℝ)
noncomputable def vector2 (λ : ℝ) := (2 + λ, 1 : ℝ)

theorem lambda_range : 
  (∃ λ : ℝ, λ ∈ (-5 : ℝ, ∞) ∧ λ ≠ -5 / 3) →
     angle_is_acute vector1 (vector2 λ) →
     λ ∈ (-5 : ℝ, ∞) \ {-5 / 3} :=
by
  sorry

end lambda_range_l320_320322


namespace trivet_height_after_breakage_l320_320217

theorem trivet_height_after_breakage:
  (∀ (ABC : Triangle), 
      -- original lengths of the legs
      (ABC.legA = ABC.legB) ∧ 
      (ABC.legA = 7) ∧ 
      (ABC.legC = 6) ∧ 
      -- original angles between the legs
      (ABC.angle_A_B = 60) ∧ 
      (ABC.angle_A_C = 75) ∧ 
      (ABC.angle_B_C = 75) ∧ 
      -- lengths after breakage
      (ABC.legC_after_break = 5) →
      -- height after breakage
      ABC.height = 1.17) := 
sorry

end trivet_height_after_breakage_l320_320217


namespace inequality_solution_l320_320286

noncomputable theory

def f (x : ℝ) : ℝ := Real.log (x^2 + 1) + |x|

theorem inequality_solution :
  { x : ℝ | f (2 * x - 1) > f (x + 1) } = { x | x < 0 ∨ x > 2 } :=
sorry

end inequality_solution_l320_320286


namespace find_initial_sum_l320_320213

-- Define the conditions as constants
def A1 : ℝ := 590
def A2 : ℝ := 815
def t1 : ℝ := 2
def t2 : ℝ := 7

-- Define the variables
variable (P r : ℝ)

-- First condition after 2 years
def condition1 : Prop := A1 = P + P * r * t1

-- Second condition after 7 years
def condition2 : Prop := A2 = P + P * r * t2

-- The statement we need to prove: the initial sum of money P is 500
theorem find_initial_sum (h1 : condition1 P r) (h2 : condition2 P r) : P = 500 :=
sorry

end find_initial_sum_l320_320213


namespace function_domain_l320_320105

theorem function_domain (x : ℝ) :
  (x - 3 > 0) ∧ (5 - x ≥ 0) ↔ (3 < x ∧ x ≤ 5) :=
by
  sorry

end function_domain_l320_320105


namespace angle_in_third_quadrant_l320_320841

theorem angle_in_third_quadrant (α : ℝ) (h1 : Real.cos α < 0) (h2 : Real.tan α > 0) : 
  (π < α ∧ α < 3 * π / 2) :=
sorry

end angle_in_third_quadrant_l320_320841


namespace min_perimeter_of_triangle_l320_320859

theorem min_perimeter_of_triangle
  (a b : ℕ)
  (h_eq : a = b)
  (h_angle : ∠BAC = 120)
  (h_incenter_tangent : ∀ {r r_a : ℝ}, r = (∂ABC)/s → r_a = (∂ABC)/(s-a) → r + r_a = ω.radius)
  (h_excircle_tangent : ∀ {r r_a : ℝ}, r_a ⊥ ω → r + r_a = ω.radius) :
  min_perimeter ∆ABC = 18 := sorry

end min_perimeter_of_triangle_l320_320859


namespace parity_D_n_l320_320108

def D : ℕ → ℤ
| 0 := 0
| 1 := 0
| 2 := 1
| n := if h : 3 ≤ n then D (n - 1) + D (n - 3) else 0

theorem parity_D_n : 
  (D 2021 % 2, D 2022 % 2, D 2023 % 2) = (0, 1, 0) := 
by
  sorry

end parity_D_n_l320_320108


namespace estimated_germination_probability_l320_320659

section Germination

-- Define the germination rates for the given number of seeds
def germination_rate (n : ℕ) : ℝ :=
  if n = 500 then 0.974 else
  if n = 1000 then 0.983 else
  if n = 2000 then 0.971 else
  if n = 10000 then 0.973 else
  if n = 20000 then 0.971 else
  0

-- The theorem stating the estimated probability of germination for one seed
theorem estimated_germination_probability : 
  (choose n, n > 0 ∧ n ∈ {500, 1000, 2000, 10000, 20000}) → (germination_rate n).round = 0.97 :=
sorry

end Germination

end estimated_germination_probability_l320_320659


namespace calculate_height_l320_320993

def base_length : ℝ := 2 -- in cm
def base_width : ℝ := 5 -- in cm
def volume : ℝ := 30 -- in cm^3

theorem calculate_height: base_length * base_width * 3 = volume :=
by
  -- base_length * base_width = 10
  -- 10 * 3 = 30
  sorry

end calculate_height_l320_320993


namespace sqrt_inequality_l320_320785

variable {x y : ℝ}

theorem sqrt_inequality (x_gt_zero : x > 0) (y_gt_zero : y > 0) : 
    sqrt (x^2 / y) + sqrt (y^2 / x) ≥ sqrt x + sqrt y :=
sorry

end sqrt_inequality_l320_320785


namespace area_of_T_l320_320453

-- Define the conditions
def ω : ℂ := -1/2 + (1 / 2) * complex.I * real.sqrt 3

def T (a b c : ℝ) : set ℂ :=
  {z : ℂ | ∃ a b c, (0 ≤ a ∧ a ≤ 1) ∧ (0 ≤ b ∧ b ≤ 2) ∧ (0 ≤ c ∧ c ≤ 2) ∧ z = a + b * ω + c * (ω^2)}

-- Define our target theorem
theorem area_of_T : complex.abs ((2 * ω) * (-2 * complex.I * ω)) = 6 * real.sqrt 3 :=
begin
  sorry
end

end area_of_T_l320_320453


namespace pizza_volume_l320_320206

theorem pizza_volume (h : ℝ) (d : ℝ) (n : ℕ) 
  (h_cond : h = 1/2) 
  (d_cond : d = 16) 
  (n_cond : n = 8) 
  : (π * (d / 2) ^ 2 * h / n = 4 * π) :=
by
  sorry

end pizza_volume_l320_320206


namespace xy_value_l320_320550

theorem xy_value (x y : ℝ) (h1 : 2^x = 64^(y + 1)) (h2 : 27^y = 3^(x - 2)) : x * y = 8 / 3 := by  
  sorry

end xy_value_l320_320550


namespace sale_price_as_percentage_l320_320664

def original_price : ℝ := 500.00
def first_discount : ℝ := 0.10
def second_discount : ℝ := 0.20
def third_discount : ℝ := 0.05

def discounted_price (price : ℝ) (discount : ℝ) : ℝ := price * (1 - discount)

theorem sale_price_as_percentage :
  let final_price := discounted_price (discounted_price (discounted_price original_price first_discount) second_discount) third_discount in
  final_price / original_price * 100 = 68.4 := by
  sorry

end sale_price_as_percentage_l320_320664


namespace probability_number_greater_than_3_from_0_5_l320_320414

noncomputable def probability_number_greater_than_3_in_0_5 : ℝ :=
  let total_interval_length := 5 - 0
  let event_interval_length := 5 - 3
  event_interval_length / total_interval_length

theorem probability_number_greater_than_3_from_0_5 :
  probability_number_greater_than_3_in_0_5 = 2 / 5 :=
by
  sorry

end probability_number_greater_than_3_from_0_5_l320_320414


namespace find_a_for_even_function_l320_320362

theorem find_a_for_even_function (a : ℝ) (h : ∀ x : ℝ, x^3 * (a * 2^x - 2^(-x)) = (-x)^3 * (a * 2^(-x) - 2^x)) : a = 1 :=
by
  -- Placeholder for proof
  sorry

end find_a_for_even_function_l320_320362


namespace identified_rectangle_perimeter_l320_320671

-- Define the side length of the square
def side_length_mm : ℕ := 75

-- Define the heights of the rectangles
variables (x y z : ℕ)

-- Define conditions
def rectangles_cut_condition (x y z : ℕ) : Prop := x + y + z = side_length_mm
def perimeter_relation_condition (x y z : ℕ) : Prop := 2 * (x + side_length_mm) = (y + side_length_mm) + (z + side_length_mm)

-- Define the perimeter of the identified rectangle
def identified_perimeter_mm (x : ℕ) := 2 * (x + side_length_mm)

-- Define conversion from mm to cm
def mm_to_cm (mm : ℕ) : ℕ := mm / 10

-- Final proof statement
theorem identified_rectangle_perimeter :
  ∃ x y z : ℕ, rectangles_cut_condition x y z ∧ perimeter_relation_condition x y z ∧ mm_to_cm (identified_perimeter_mm x) = 20 := 
sorry

end identified_rectangle_perimeter_l320_320671


namespace range_of_f_l320_320456

open Real

noncomputable def f (x y z w : ℝ) : ℝ :=
  x / (x + y) + y / (y + z) + z / (z + x) + w / (w + x)

theorem range_of_f (x y z w : ℝ) (h1x : 0 < x) (h1y : 0 < y) (h1z : 0 < z) (h1w : 0 < w) :
  1 < f x y z w ∧ f x y z w < 2 :=
  sorry

end range_of_f_l320_320456


namespace solution_n_value_l320_320097

open BigOperators

noncomputable def problem_statement (a b n : ℝ) : Prop :=
  ∃ (A B : ℝ), A = Real.log a ∧ B = Real.log b ∧
    (7 * A + 15 * B) - (4 * A + 9 * B) = (11 * A + 20 * B) - (7 * A + 15 * B) ∧
    (4 + 135) * B = Real.log (b^n)

theorem solution_n_value (a b : ℝ) (h_pos : a > 0) (h_pos_b : b > 0) :
  problem_statement a b 139 :=
by
  sorry

end solution_n_value_l320_320097


namespace michelle_silver_beads_l320_320927

theorem michelle_silver_beads :
  ∀ (total_beads blue_beads red_beads white_beads silver_beads : ℕ),
    total_beads = 40 →
    blue_beads = 5 →
    red_beads = 2 * blue_beads →
    white_beads = blue_beads + red_beads →
    silver_beads = total_beads - (blue_beads + red_beads + white_beads) →
    silver_beads = 10 :=
by {
  intros total_beads blue_beads red_beads white_beads silver_beads,
  assume h1 h2 h3 h4 h5,
  sorry
}

end michelle_silver_beads_l320_320927


namespace first_term_of_geometric_series_l320_320113

-- Define the conditions and the question
theorem first_term_of_geometric_series (a r : ℝ) (h1 : a / (1 - r) = 30) (h2 : a^2 / (1 - r^2) = 180) : a = 10 :=
by sorry

end first_term_of_geometric_series_l320_320113


namespace false_proposition_D_is_false_l320_320691

-- Define the propositions
def proposition_A (a b : ℝ) : Prop := ¬(a ≤ b → 2 ^ a ≤ 2 ^ b - 1)
def proposition_B (a : ℝ) : Prop := ¬∀ x : ℝ, a ∈ set.Ioi 0 → (monotone (λ x:ℝ, a ^ x))
def proposition_C : Prop := (∀ x : ℝ, (real.sin (x + real.pi) = real.sin x)) ∨ (∀ x : ℝ, (real.sin (2 * x + 2 * real.pi) = real.sin (2 * x)))
def proposition_D (x y : ℝ) : Prop := (x ^ 2 + y ^ 2 = 0) → (x * y = 0)

-- Define the proof problem statement
theorem false_proposition_D_is_false : ¬(proposition_D x y) :=
by sorry

end false_proposition_D_is_false_l320_320691


namespace proof_problem_example_l320_320406

noncomputable def integer_solution_pairs_count : ℕ :=
  let a_vals := {a : ℤ | 1 ≤ a ∧ a ≤ 9}
  let b_vals := {b : ℤ | 25 ≤ b ∧ b ≤ 32}
  set.size a_vals * set.size b_vals

theorem proof_problem_example : integer_solution_pairs_count = 72 :=
by
  have a_vals_card : a_vals.card = 9 := sorry
  have b_vals_card : b_vals.card = 8 := sorry
  calc integer_solution_pairs_count
      = a_vals.card * b_vals.card : sorry
  ... = 9 * 8                  : by rw [a_vals_card, b_vals_card]
  ... = 72                      : by norm_num

end proof_problem_example_l320_320406


namespace length_to_width_ratio_l320_320573

-- Define the conditions
def width := 6
def area := 108

-- The length of the rectangle, derived from the conditions
def length := area / width

-- The main statement: The ratio of the length to the width is 3:1
theorem length_to_width_ratio : length / width = 3 :=
by
  unfold length width area
  -- the following is auto calculated based on the values from the condition
  simp
  sorry


end length_to_width_ratio_l320_320573


namespace masha_numbers_unique_l320_320478

def natural_numbers : Set ℕ := {n | n > 11}

theorem masha_numbers_unique (a b : ℕ) (ha : a ∈ natural_numbers) (hb : b ∈ natural_numbers) (hne : a ≠ b)
  (hs_equals : ∃ S, S = a + b)
  (sasha_initially_uncertain : ∀ (a b : ℕ), a ∈ natural_numbers → b ∈ natural_numbers → a ≠ b → (a + b = S) → ¬ (Sasha_can_determine_initially a b S))
  (masha_hint : ∃ (a_even : ℕ), a_even ∈ natural_numbers ∧ (a_even % 2 = 0) ∧ (a_even = a ∨ a_even = b))
  (sasha_then_confident : ∀ (a b : ℕ), a ∈ natural_numbers → b ∈ natural_numbers → a ≠ b → (a + b = S) → (a_even = a ∨ a_even = b) → Sasha_can_determine_confidently a b S) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) := by
  sorry

end masha_numbers_unique_l320_320478


namespace average_of_four_given_conditions_l320_320990

noncomputable def average_of_four_integers : ℕ × ℕ × ℕ × ℕ → ℚ :=
  λ ⟨a, b, c, d⟩ => (a + b + c + d : ℚ) / 4

theorem average_of_four_given_conditions :
  ∀ (A B C D : ℕ), 
    (A + B) / 2 = 35 → 
    C = 130 → 
    D = 1 → 
    average_of_four_integers (A, B, C, D) = 50.25 := 
by
  intros A B C D hAB hC hD
  unfold average_of_four_integers
  sorry

end average_of_four_given_conditions_l320_320990


namespace marble_cost_l320_320689

def AlyssaSpentOnMarbles (totalSpent onToys footballCost : ℝ) : ℝ :=
 totalSpent - footballCost

theorem marble_cost:
  AlyssaSpentOnMarbles 12.30 5.71 = 6.59 :=
by 
  unfold AlyssaSpentOnMarbles 
  sorry

end marble_cost_l320_320689


namespace min_value_of_quadratic_l320_320290

theorem min_value_of_quadratic (x : ℝ) : 
  ∃ m : ℝ, (∀ z : ℝ, z = 5 * x ^ 2 + 20 * x + 25 → z ≥ m) ∧ m = 5 :=
by
  sorry

end min_value_of_quadratic_l320_320290


namespace solution_set_f_x_leq_5_range_of_a_l320_320802

-- Problem 1: Prove that f(x) ≤ 5 has a solution set [0, 2].
theorem solution_set_f_x_leq_5 :
  ( ∀ x : ℝ, f x = abs (2 * x - 3) + abs (x + 2) ) →
  ( set_of (λ x, f x ≤ 5) = set.Icc 0 2 ) :=
sorry

-- Problem 2: Prove the range of values for a is [7, +∞)
theorem range_of_a (a : ℝ) :
  ( ∀ x : ℝ, f x = abs (2 * x - 3) + abs (x + 2) ) →
  ( ∀ x : ℝ, (-1 ≤ x ∧ x ≤ 2) → ( f x ≤ a - abs x) ) →
  ( a ≥ 7 ) :=
sorry

end solution_set_f_x_leq_5_range_of_a_l320_320802


namespace find_values_x_y_z_of_N_squared_eq_2_I_and_xyz_eq_2_l320_320894

noncomputable theory
open Complex Matrix

variables {x y z : ℂ}

def N : Matrix (Fin 3) (Fin 3) ℂ := 
  ![![x, y, z], 
    ![z, x, y], 
    ![y, z, x]]

theorem find_values_x_y_z_of_N_squared_eq_2_I_and_xyz_eq_2 
  (hN2I : N ⬝ N = 2 • (1 : Matrix (Fin 3) (Fin 3) ℂ))
  (hxyz : x * y * z = 2) :
  ∃ (a b : ℂ), (a = 2 * Complex.sqrt 2 + 6 ∨ a = 6 - 2 * Complex.sqrt 2) ∧ (x^3 + y^3 + z^3 = a) := 
sorry

end find_values_x_y_z_of_N_squared_eq_2_I_and_xyz_eq_2_l320_320894


namespace sin_squared_sum_l320_320273

theorem sin_squared_sum : 
  ∑ k in finset.range 30 \ {0}, (sin (6 * k + 6) * (real.pi / 180))^2 = 15 :=
sorry

end sin_squared_sum_l320_320273


namespace main_problem_l320_320743

variable {a b x : ℝ}

theorem main_problem (h₁: -1 ≤ a ∧ a ≤ 1) (h₂: -5 ≤ x ∧ x ≤ -1) :
    ¬(x^2 + 6*x + 2*(a + b + 1)*sqrt(-x^2 - 6*x - 5) + 8 < a^2 + b^2 + 2*a) →
    b ∈ Set.Icc (-1 : ℝ) 4 := 
sorry

end main_problem_l320_320743


namespace distance_from_origin_to_line_l320_320567

-- Define the line equation coefficients
def A : ℝ := 1
def B : ℝ := -2
def C : ℝ := 3

-- Define the point (the origin)
def x0 : ℝ := 0
def y0 : ℝ := 0

-- The goal is to prove the distance from the origin to the line is 3 * sqrt(5) / 5
theorem distance_from_origin_to_line : 
  (abs (A * x0 + B * y0 + C) / real.sqrt (A ^ 2 + B ^ 2) = 3 * real.sqrt 5 / 5) :=
by
  sorry

end distance_from_origin_to_line_l320_320567


namespace polynomial_independent_of_m_l320_320805

theorem polynomial_independent_of_m (m : ℝ) (x : ℝ) (h : 6 * x^2 + (1 - 2 * m) * x + 7 * m = 6 * x^2 + x) : 
  x = 7 / 2 :=
by
  sorry

end polynomial_independent_of_m_l320_320805


namespace binomial_expansion_identity_l320_320029

theorem binomial_expansion_identity 
    (a : ℕ → ℝ)
    (h : ∀ x : ℝ, (ℝ.sqrt 2 - x) ^ 10 = ∑ i in finset.range 11, a i * x ^ i) : 
    (∑ i in finset.range 6, a (2 * i)) ^ 2 - (∑ i in finset.range 5, a (2 * i + 1)) ^ 2 = 1 := 
sorry

end binomial_expansion_identity_l320_320029


namespace find_c_sum_bounds_l320_320427

noncomputable def a_n (c : ℝ) (n : ℕ) : ℝ :=
if n = 0 then 1 else 1 + (n - 1) * c

theorem find_c : 
  (∃ c : ℝ, 
    let a1 := 1, a2 := 1 + c, a5 := 1 + 4 * c in
    c ≠ 0 ∧ 
    a1 * a5 = a2 * a2) → c = 2 :=
sorry

theorem sum_bounds (c : ℝ) (h : c = 2) : 
  (∀ n : ℕ, 
    let a := fun n => 2 * n - 1,
    let b := fun n => 1 / (a n * a (n + 1)),
    let S_n := finset.sum (finset.range n) b in 
    1/3 ≤ S_n ∧ S_n < 1/2) :=
sorry

end find_c_sum_bounds_l320_320427


namespace remainder_a52_div_52_l320_320901

def a_n (n : ℕ) : ℕ := 
  (List.range (n + 1)).foldl (λ acc x => acc * 10 ^ (Nat.digits 10 x).length + x) 0

theorem remainder_a52_div_52 : (a_n 52) % 52 = 28 := 
  by
  sorry

end remainder_a52_div_52_l320_320901


namespace cost_per_toy_initially_l320_320441

-- defining conditions
def num_toys : ℕ := 200
def percent_sold : ℝ := 0.8
def price_per_toy : ℝ := 30
def profit : ℝ := 800

-- defining the problem
theorem cost_per_toy_initially :
  ((num_toys * percent_sold) * price_per_toy - profit) / (num_toys * percent_sold) = 25 :=
by
  sorry

end cost_per_toy_initially_l320_320441


namespace numbers_masha_thought_l320_320530

noncomputable def distinct_natural_numbers_greater_than_eleven_and_sum_to_satisfy_conditions : ℕ → ℕ → Prop :=
λ a b, a ≠ b ∧ a > 11 ∧ b > 11 ∧ (a + b = 28) ∧ (¬ (∃ x y, x ≠ y ∧ x > 11 ∧ y > 11 ∧ x + y = a + b ∧ (x ≠ a ∧ y ≠ b)))

theorem numbers_masha_thought (a b : ℕ) (h : distinct_natural_numbers_greater_than_eleven_and_sum_to_satisfy_conditions a b) : 
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) :=
by sorry

end numbers_masha_thought_l320_320530


namespace am_gm_inequality_l320_320908

theorem am_gm_inequality (n : ℕ) (a : Fin n → ℝ) (h_pos : ∀ i, 0 < a i) (h_prod : ∏ i, a i = 1) :
    (∏ i, (1 + a i)) ≥ 2 ^ n ∧ (∀ i, a i = 1 → ∏ i, (1 + a i) = 2 ^ n) := by
  sorry

end am_gm_inequality_l320_320908


namespace sum_of_squared_sines_equals_31_over_2_l320_320259

noncomputable def sum_of_squared_sines : ℝ :=
  (∑ n in finset.range 30, real.sin ((n + 1 : ℕ) * 6 * real.pi / 180) ^ 2)

theorem sum_of_squared_sines_equals_31_over_2 :
  sum_of_squared_sines = 31 / 2 :=
by sorry

end sum_of_squared_sines_equals_31_over_2_l320_320259


namespace Masha_thought_of_numbers_l320_320513

theorem Masha_thought_of_numbers : ∃ a b : ℕ, a ≠ b ∧ a > 11 ∧ b > 11 ∧ (a + b = 28) ∧ (a % 2 = 0 ∨ b % 2 = 0) ∧ (a = 12 ∧ b = 16 ∨ a = 16 ∧ b = 12) :=
by
  sorry

end Masha_thought_of_numbers_l320_320513


namespace radius_of_inscribed_semicircle_in_isosceles_triangle_l320_320280

theorem radius_of_inscribed_semicircle_in_isosceles_triangle
    (BC : ℝ) (h : ℝ) (r : ℝ)
    (H_eq : BC = 24)
    (H_height : h = 18)
    (H_area : 0.5 * BC * h = 0.5 * 24 * 18) :
    r = 18 / π := by
    sorry

end radius_of_inscribed_semicircle_in_isosceles_triangle_l320_320280


namespace masha_numbers_l320_320488

theorem masha_numbers {a b : ℕ} (h1 : a ≠ b) (h2 : 11 < a) (h3 : 11 < b) 
  (h4 : ∃ S, S = a + b ∧ (∀ x y, x + y = S → x ≠ y → 11 < x ∧ 11 < y → 
       (¬(x = a ∧ y = b) ∧ ¬(x = b ∧ y = a)))) 
  (h5 : even a ∨ even b)
  (h6 : ∀ x y, (even x ∨ even y) → x ≠ y → 11 < x ∧ 11 < y ∧ x + y = a + b → 
       x = a ∧ y = b ∨ x = b ∧ y = a) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) :=
by
  sorry

end masha_numbers_l320_320488


namespace set_intersection_l320_320811

theorem set_intersection (M N : Set ℝ) (hM : M = {x | x < 3}) (hN : N = {x | x > 2}) :
  M ∩ N = {x | 2 < x ∧ x < 3} :=
sorry

end set_intersection_l320_320811


namespace average_weight_proof_l320_320992

variables (W_A W_B W_C W_D W_E : ℝ)

noncomputable def final_average_weight (W_A W_B W_C W_D W_E : ℝ) : ℝ := (W_B + W_C + W_D + W_E) / 4

theorem average_weight_proof
  (h1 : (W_A + W_B + W_C) / 3 = 84)
  (h2 : W_A = 77)
  (h3 : (W_A + W_B + W_C + W_D) / 4 = 80)
  (h4 : W_E = W_D + 5) :
  final_average_weight W_A W_B W_C W_D W_E = 97.25 :=
by
  sorry

end average_weight_proof_l320_320992


namespace sin_squared_sum_l320_320271

theorem sin_squared_sum : 
  ∑ k in finset.range 30 \ {0}, (sin (6 * k + 6) * (real.pi / 180))^2 = 15 :=
sorry

end sin_squared_sum_l320_320271


namespace product_of_integers_l320_320321

theorem product_of_integers (X Y Z W : ℚ) (h_sum : X + Y + Z + W = 100)
  (h_relation : X + 5 = Y - 5 ∧ Y - 5 = 3 * Z ∧ 3 * Z = W / 3) :
  X * Y * Z * W = 29390625 / 256 := by
  sorry

end product_of_integers_l320_320321


namespace Masha_thought_of_numbers_l320_320510

theorem Masha_thought_of_numbers : ∃ a b : ℕ, a ≠ b ∧ a > 11 ∧ b > 11 ∧ (a + b = 28) ∧ (a % 2 = 0 ∨ b % 2 = 0) ∧ (a = 12 ∧ b = 16 ∨ a = 16 ∧ b = 12) :=
by
  sorry

end Masha_thought_of_numbers_l320_320510


namespace triangle_ineq_l320_320073

theorem triangle_ineq
  (a b c : ℝ)
  (triangle_sides : 0 < a ∧ 0 < b ∧ 0 < c)
  (triangle_ineq : a + b + c = 1) :
  a^2 + b^2 + c^2 + 4 * a * b * c < 1 / 2 :=
by
  sorry

end triangle_ineq_l320_320073


namespace complement_A_in_U_l320_320813

def U := {2, 4, 5, 7, 8}
def A := {4, 8}
def complement (U A : Set ℕ) := {x ∈ U | x ∉ A}

theorem complement_A_in_U :
  complement U A = {2, 5, 7} :=
by
  sorry

end complement_A_in_U_l320_320813


namespace domain_F_range_F_l320_320791

noncomputable def f (x : ℝ) : ℝ :=
  -2 * (x - 1)^2 + 2

noncomputable def g (x : ℝ) : ℝ :=
  Real.log(x) / Real.log(2)

noncomputable def F (x : ℝ) : ℝ :=
  g (f x)

-- Let's state the domain and range of F(x)
theorem domain_F :
  ∀ x, 0 < x ∧ x < 2 → 0 < f x ∧ f x ≤ 2 :=
sorry

theorem range_F :
  ∀ y, y ≤ 1 → ∃ x, F x = y :=
sorry

end domain_F_range_F_l320_320791


namespace a_n_not_periodic_beta_n_not_periodic_l320_320632

noncomputable def a_n (n : ℕ) : ℕ :=
  let int_part := (Real.sqrt 10) ^ n
  int_part.floorDigits % 10

noncomputable def beta_n (n : ℕ) : ℕ :=
  let int_part := (Real.sqrt 2) ^ n
  int_part.floorDigits % 10

theorem a_n_not_periodic : ¬ ∃ p : ℕ, ∀ n : ℕ, a_n (n + p) = a_n n :=
sorry

theorem beta_n_not_periodic : ¬ ∃ p : ℕ, ∀ n : ℕ, beta_n (n + p) = beta_n n :=
sorry

end a_n_not_periodic_beta_n_not_periodic_l320_320632


namespace combined_work_rate_l320_320635

theorem combined_work_rate (W : ℝ) 
  (A_rate : ℝ := W / 10) 
  (B_rate : ℝ := W / 5) : 
  A_rate + B_rate = 3 * W / 10 := 
by
  sorry

end combined_work_rate_l320_320635


namespace range_of_k_l320_320413

theorem range_of_k (k : ℝ) (h₁ : ∃ x y : ℝ, k*x - y + 1 = 0 ∧ x - k*y = 0 ∧ x < 0 ∧ y > 0) : k ∈ Ioo (-1 : ℝ) (0 : ℝ) :=
sorry

end range_of_k_l320_320413


namespace rectangular_coord_equation_C2_min_PQ_distance_l320_320007

/-- Problem (I): Prove the rectangular coordinate equation for a polar curve -/
theorem rectangular_coord_equation_C2 (θ : ℝ) (ρ := cos θ) : ∀ (x y : ℝ),
  ρ = (x^2 + y^2).sqrt → (x, y) = (ρ * cos θ, ρ * sin θ) → x^2 + y^2 = x :=
begin
  intros x y Hρ Hxy,
  sorry
end

/-- Problem (II): Calculate the minimum distance between points on two specific curves -/
theorem min_PQ_distance (α : ℝ) (x1 y1 x2 y2: ℝ) :
  x1 = 2 * cos α → y1 = (sqrt 2) * sin α 
  → x2^2 + y2^2 = x2 
  → ∀ P Q : ℝ, (P = (x1, y1) ∧ Q = (x2, y2)) 
  → min_val_PQ P Q = (sqrt 7 - 1) / 2 :=
begin
  intros Hx1 Hy1 Hx2 H_PQ,
  sorry
end

end rectangular_coord_equation_C2_min_PQ_distance_l320_320007


namespace monotonic_increasing_interval_l320_320096

-- Define the function y = ln(x^2 - 2x)
def f (x : ℝ) : ℝ := Real.log (x^2 - 2 * x)

-- State the main theorem
theorem monotonic_increasing_interval (x : ℝ) (h : x^2 - 2 * x > 0) : (2 < x) → MonotonicIncreasingOn f (Set.Ioi 2) :=
by
  sorry

end monotonic_increasing_interval_l320_320096


namespace sum_of_squared_sines_equals_31_over_2_l320_320256

noncomputable def sum_of_squared_sines : ℝ :=
  (∑ n in finset.range 30, real.sin ((n + 1 : ℕ) * 6 * real.pi / 180) ^ 2)

theorem sum_of_squared_sines_equals_31_over_2 :
  sum_of_squared_sines = 31 / 2 :=
by sorry

end sum_of_squared_sines_equals_31_over_2_l320_320256


namespace closest_point_to_origin_l320_320960

theorem closest_point_to_origin : 
  ∃ x y : ℝ, x > 0 ∧ y = x + 1/x ∧ (x, y) = (1/(2^(1/4)), (1 + 2^(1/2))/(2^(1/4))) :=
by
  sorry

end closest_point_to_origin_l320_320960


namespace masha_numbers_l320_320486

theorem masha_numbers {a b : ℕ} (h1 : a ≠ b) (h2 : 11 < a) (h3 : 11 < b) 
  (h4 : ∃ S, S = a + b ∧ (∀ x y, x + y = S → x ≠ y → 11 < x ∧ 11 < y → 
       (¬(x = a ∧ y = b) ∧ ¬(x = b ∧ y = a)))) 
  (h5 : even a ∨ even b)
  (h6 : ∀ x y, (even x ∨ even y) → x ≠ y → 11 < x ∧ 11 < y ∧ x + y = a + b → 
       x = a ∧ y = b ∨ x = b ∧ y = a) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) :=
by
  sorry

end masha_numbers_l320_320486


namespace problem_1_problem_2_l320_320179

-- Problem (1)
theorem problem_1 (a : ℝ) (h : a^(1/2) + a^(-1/2) = 3) : 
  (a^2 + a^(-2) + 1) / (a + a^(-1) - 1) = 8 := 
sorry

-- Problem (2)
theorem problem_2 :
  sqrt ((1 - sqrt 2)^2) + 2^(-2) * (9 / 16)^(-0.5) + 2^(log 2 3) - (log 10 8 + log 10 125) = 
  sqrt 2 - (2 / 3) := 
sorry

end problem_1_problem_2_l320_320179


namespace third_side_length_l320_320093

theorem third_side_length
  (a c : ℝ)
  (h : let ma := (a * (1 / 2)) in let mc := (c * (1 / 2)) in let midpt33 := (ma * a)^2 + (mc * c)^2; 
       (ma ≥ 0) ∧ (mc ≥ 0) ∧ (midpt33 = c^2/4) ∧ (ma^2 + c^2/4 = 0)) :
  (b : ℝ) :=
b = sqrt ((a^2 + c^2) / 5) 
  
skip'

end third_side_length_l320_320093


namespace closest_point_on_graph_l320_320950

theorem closest_point_on_graph (x y : ℝ) (h1 : x > 0) (h2 : y = x + 1/x) :
  (x = 1/real.root 4 2) ∧ (y = (1 + real.sqrt 2) / real.root 4 2) :=
sorry

end closest_point_on_graph_l320_320950


namespace tan_eq_123_deg_l320_320308

theorem tan_eq_123_deg (n : ℤ) (h : -180 < n ∧ n < 180) : 
  real.tan (n * real.pi / 180) = real.tan (123 * real.pi / 180) → n = 123 ∨ n = -57 :=
by
  -- to do the proof
  sorry

end tan_eq_123_deg_l320_320308


namespace single_appetizer_cost_l320_320604

def total_cost : ℝ := 50
def entree_percent : ℝ := 0.80
def appetizer_percent : ℝ := 0.20
def number_of_appetizers : ℕ := 2

theorem single_appetizer_cost :
  let entree_cost := entree_percent * total_cost in
  let appetizer_cost := appetizer_percent * total_cost in
  let single_appetizer_cost := appetizer_cost / number_of_appetizers in
  single_appetizer_cost = 5 := by
  sorry

end single_appetizer_cost_l320_320604


namespace sum_of_altitudes_less_than_sum_of_sides_l320_320547

-- Define a triangle with sides and altitudes properties
structure Triangle :=
(A B C : Point)
(a b c : ℝ)
(m_a m_b m_c : ℝ)
(sides : a + b > c ∧ b + c > a ∧ c + a > b) -- Triangle Inequality

axiom altitude_property (T : Triangle) :
  T.m_a < T.b ∧ T.m_b < T.c ∧ T.m_c < T.a

-- The theorem to prove
theorem sum_of_altitudes_less_than_sum_of_sides (T : Triangle) :
  T.m_a + T.m_b + T.m_c < T.a + T.b + T.c :=
sorry

end sum_of_altitudes_less_than_sum_of_sides_l320_320547


namespace probability_divisible_by_5_l320_320667

def is_three_digit_number (n : ℕ) : Prop := n ≥ 100 ∧ n ≤ 999
def is_divisible_by_5 (n : ℕ) : Prop := n % 5 = 0
def count_3d_numbers := 999 - 100 + 1 -- Number of three-digit numbers 

theorem probability_divisible_by_5 : 
  let total := count_3d_numbers in
  let divisible_by_5_set := (100 + (nat.succ 179) * 5 - 5 - 1) / 5 + 1 in
  divisible_by_5_set / total = 1 / 5 :=
by
  sorry

end probability_divisible_by_5_l320_320667


namespace prime_p_divisors_l320_320742

theorem prime_p_divisors (p : ℕ) : 
  (Nat.Prime p) → (Nat.numDivisors (p^2 + 11) < 11) → p = 5 :=
by
  intros hp h_divisors
  sorry

end prime_p_divisors_l320_320742


namespace find_side_a_l320_320882

theorem find_side_a (a b c : ℝ) (B : ℝ)
  (h1 : c = Real.sqrt 2)
  (h2 : b = Real.sqrt 6)
  (h3 : B = 120) :
  a = Real.sqrt 2 :=
sorry

end find_side_a_l320_320882


namespace number_of_ways_to_stand_with_one_person_between_l320_320597

theorem number_of_ways_to_stand_with_one_person_between (A B : Type) (people : Fin 5) :
  ∃ n, n = 36 ∧ (number_of_ways A B people n) :=
sorry

end number_of_ways_to_stand_with_one_person_between_l320_320597


namespace effect_of_dimension_changes_on_area_l320_320165

variable {L B : ℝ}  -- Original length and breadth

def original_area (L B : ℝ) : ℝ := L * B

def new_length (L : ℝ) : ℝ := 1.15 * L

def new_breadth (B : ℝ) : ℝ := 0.90 * B

def new_area (L B : ℝ) : ℝ := new_length L * new_breadth B

theorem effect_of_dimension_changes_on_area (L B : ℝ) :
  new_area L B = 1.035 * original_area L B :=
by
  sorry

end effect_of_dimension_changes_on_area_l320_320165


namespace exists_xyz_t_l320_320576

theorem exists_xyz_t (x y z t : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : t > 0) (h5 : x + y + z + t = 15) : ∃ y, y = 12 :=
by
  sorry

end exists_xyz_t_l320_320576


namespace tan_alpha_sub_pi_quarter_eq_three_l320_320762

theorem tan_alpha_sub_pi_quarter_eq_three (α : ℝ) 
  (h1 : sin α + 2 * cos α = 0) : 
  Real.tan (α - Real.pi / 4) = 3 := by
  sorry

end tan_alpha_sub_pi_quarter_eq_three_l320_320762


namespace tetrahedron_volume_le_one_eight_l320_320574

theorem tetrahedron_volume_le_one_eight {A B C D : Type} 
  (e₁_AB e₂_AC e₃_AD e₄_BC e₅_BD : ℝ) (h₁ : e₁_AB ≤ 1) (h₂ : e₂_AC ≤ 1) (h₃ : e₃_AD ≤ 1)
  (h₄ : e₄_BC ≤ 1) (h₅ : e₅_BD ≤ 1) : 
  ∃ (vol : ℝ), vol ≤ 1 / 8 :=
sorry

end tetrahedron_volume_le_one_eight_l320_320574


namespace distance_from_P_to_AB_l320_320432

theorem distance_from_P_to_AB
  (A B C P : Type)
  [InnerProductSpace ℝ A]
  [InnerProductSpace ℝ B]
  [InnerProductSpace ℝ C]
  [InnerProductSpace ℝ P]
  (AB : Real := 6)
  (height_C_to_AB : Real := 4)
  (ratio_areas : Real := 1/3)
  (h : P → A ≠ A → C)
  (parallel_line : ∀ {x y : A}, x ∥ y → x ∥ AB): 
     distance from P to AB = 2*sqrt(3) - 2 := 
by
  sorry

end distance_from_P_to_AB_l320_320432


namespace sum_of_squares_of_sines_l320_320241

theorem sum_of_squares_of_sines : 
  (\sum n in finset.range 1 30, real.sin (6 * n : ℝ) ^ 2) = 31 / 2 :=
begin
  sorry
end

end sum_of_squares_of_sines_l320_320241


namespace even_function_a_value_l320_320355

theorem even_function_a_value {f : ℝ → ℝ} (a : ℝ) :
  (∀ x : ℝ, f x = x^3 * (a * 2^x - 2^(-x)) ∧ f x = f (-x)) → a = 1 :=
by
  intros h,
  sorry

end even_function_a_value_l320_320355


namespace slope_range_l320_320765

noncomputable def slope (m : ℝ) : ℝ := m / (m^2 + 1)

theorem slope_range (m : ℝ) : slope m ∈ Icc (-1/2) (1/2) :=
by
  sorry

end slope_range_l320_320765


namespace sin_cos_inequality_l320_320403

theorem sin_cos_inequality (n : ℕ) (h : ∀ x : ℝ, sin x ^ n + cos x ^ n ≥ 1 / n) : n ≤ 8 :=
by
  sorry

end sin_cos_inequality_l320_320403


namespace justin_gathering_hours_l320_320021

noncomputable def gathering_time_in_hours
  (num_classmates : ℕ) (avg_time_per_flower : ℝ) (flowers_lost : ℕ)
  (remaining_time : ℝ) : ℝ :=
  let total_flowers := num_classmates in
  let flowers_still_needed := remaining_time / avg_time_per_flower in
  let total_needed_flowers := flowers_still_needed + flowers_lost in
  let flowers_already_gathered := total_flowers - total_needed_flowers in
  let gathering_time := flowers_already_gathered * avg_time_per_flower in
  let gathering_time_hours := gathering_time / 60 in
  gathering_time_hours

theorem justin_gathering_hours
  (num_classmates : ℕ) (avg_time_per_flower : ℝ) (flowers_lost : ℕ)
  (remaining_time : ℝ) (h1 : num_classmates = 30) (h2 : avg_time_per_flower = 10)
  (h3 : flowers_lost = 3) (h4 : remaining_time = 210) :
  gathering_time_in_hours num_classmates avg_time_per_flower flowers_lost remaining_time = 1 :=
by
  unfold gathering_time_in_hours
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end justin_gathering_hours_l320_320021


namespace total_cost_898_8_l320_320643

theorem total_cost_898_8 :
  ∀ (M R F : ℕ → ℝ), 
    (10 * M 1 = 24 * R 1) →
    (6 * F 1 = 2 * R 1) →
    (F 1 = 21) →
    (4 * M 1 + 3 * R 1 + 5 * F 1 = 898.8) :=
by
  intros M R F h1 h2 h3
  sorry

end total_cost_898_8_l320_320643


namespace series_sum_eq_l320_320706

theorem series_sum_eq :
  (∑ n in Finset.range 30, complex.I ^ (n + 1) * real.cos ((30 + 90 * n) * real.pi / 180)) 
  = -13 / 2 + real.sqrt 3 / 2 - complex.I / 2 :=
by
  sorry

end series_sum_eq_l320_320706


namespace updated_sum_l320_320208

theorem updated_sum (n : ℕ) (s : ℤ) (x : Fin n → ℤ) (hx : ∑ i, x i = s) :
  ∑ i, (3 * x i + 33) = 3 * s + 33 * n :=
by
  sorry

end updated_sum_l320_320208


namespace original_purchase_price_l320_320663

-- Define the conditions and question
theorem original_purchase_price (P S : ℝ) (h1 : S = P + 0.25 * S) (h2 : 16 = 0.80 * S - P) : P = 240 :=
by
  -- Proof steps would go here
  sorry

end original_purchase_price_l320_320663


namespace value_of_a_in_terms_of_b_l320_320638

noncomputable def value_of_a (b : ℝ) : ℝ :=
  b * (38.1966 / 61.8034)

theorem value_of_a_in_terms_of_b (b a : ℝ) :
  (∀ x : ℝ, (b / x = 61.80339887498949 / 100) ∧ (x = (a + b) * (61.80339887498949 / 100)))
  → a = value_of_a b :=
by
  sorry

end value_of_a_in_terms_of_b_l320_320638


namespace symmetric_point_coordinates_l320_320853

theorem symmetric_point_coordinates (a b : ℝ) (hp : (3, 4) = (a + 3, b + 4)) :
  (a, b) = (5, 2) :=
  sorry

end symmetric_point_coordinates_l320_320853


namespace max_b_a_l320_320915

theorem max_b_a (a b : ℝ) (h1 : (∀ x ∈ set.Icc a b, sin x ∈ set.Icc (-1 : ℝ) (-1/2)))
                (h2 : ∀ x : ℝ, (x < a ∨ x > b) → sin x < -1/2 ∨ sin x > -1) :
                b - a ≤ 4 * π / 3 :=
begin
  sorry
end

end max_b_a_l320_320915


namespace max_sum_product_l320_320079

noncomputable def largest_sum_of_products (f g h j : ℕ) : ℕ :=
  if f < g ∧ f ∈ {4, 5, 9, 10} ∧ g ∈ {4, 5, 9, 10} ∧ h ∈ {4, 5, 9, 10} ∧ j ∈ {4, 5, 9, 10} then
    let s := f + g + h + j
    let sum_sq := s * s
    let ind_sum_sq := f*f + g*g + h*h + j*j
    let min_fh_gj := if f = 4 ∧ g = 5 ∧ h = 9 ∧ j = 10 then 85 else 0 -- simplification for the example
    let total_sum := (sum_sq - ind_sum_sq) / 2 - min_fh_gj
    total_sum
  else
    0

theorem max_sum_product : ∃ (f g h j : ℕ), f < g ∧ f ∈ {4, 5, 9, 10} ∧ g ∈ {4, 5, 9, 10} ∧ h ∈ {4, 5, 9, 10} ∧ j ∈ {4, 5, 9, 10} ∧ largest_sum_of_products f g h j = 196 :=
by
  -- placeholders for the actual proof
  sorry

end max_sum_product_l320_320079


namespace figure_150_nonoverlapping_units_l320_320737

   def f (n : ℕ) : ℕ := 3 * n^2 + 3 * n + 1

   theorem figure_150_nonoverlapping_units :
     f(150) = 67951 :=
   by
     sorry
   
end figure_150_nonoverlapping_units_l320_320737


namespace sum_cosines_eq_zero_l320_320767

theorem sum_cosines_eq_zero (x : ℝ) :
  let a : ℕ → ℝ := λ n, Real.cos (x + (2 / 7) * n * Real.pi) in
  a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 = 0 :=
by
  let a : ℕ → ℝ := λ n, Real.cos (x + (2 / 7) * n * Real.pi)
  sorry

end sum_cosines_eq_zero_l320_320767


namespace closest_distance_to_l_l320_320426

-- Definitions based on identified conditions
def line_l (x y : ℝ) : Prop := y = x + 2
def curve_C (x y : ℝ) : Prop := y^2 = x

-- The point P that the parallel line passes through
variables (x0 y0 : ℝ)

-- Condition for the point P
def point_P_condition (x0 y0 : ℝ) : Prop := y0^2 < x0

-- Definition of the closest distance
def closest_distance (x0 y0 : ℝ) : ℝ := (abs (y0^2 - y0 + 3)) / (sqrt 2)

-- Main theorem statement
theorem closest_distance_to_l (x0 y0 : ℝ) (h: y0^2 < x0) (intersections : |PA| * |PB| = 2) : 
  closest_distance x0 y0 = 11 * (sqrt 2) / 8 := by
  sorry

end closest_distance_to_l_l320_320426


namespace closest_point_to_origin_on_graph_l320_320938

theorem closest_point_to_origin_on_graph :
  ∃ x : ℝ, x > 0 ∧ (y = x + 1/x ∧ (x, y) = (1/real.root 4 2, (1 + real.sqrt 2)/real.root 4 2)) := sorry

end closest_point_to_origin_on_graph_l320_320938


namespace f_7_eq_neg3_l320_320329

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 4) = f x
axiom f_interval  : ∀ x : ℝ, 0 < x ∧ x < 2 → f x = -x + 4

theorem f_7_eq_neg3 : f 7 = -3 :=
  sorry

end f_7_eq_neg3_l320_320329


namespace seating_arrangement_count_l320_320875

def ways_to_seat_around_table (n : ℕ) (fixed_block : ℕ) : ℕ :=
  (n - 1)! * fixed_block!

theorem seating_arrangement_count :
  ways_to_seat_around_table 9 2 = 80640 :=
by sorry -- Proof is skipped as instructed

end seating_arrangement_count_l320_320875


namespace ThreePowerTowerIsLarger_l320_320155

-- original power tower definitions
def A : ℕ := 3^(3^(3^3))
def B : ℕ := 2^(2^(2^(2^2)))

-- reduced forms given from the conditions
def reducedA : ℕ := 3^(3^27)
def reducedB : ℕ := 2^(2^16)

theorem ThreePowerTowerIsLarger : reducedA > reducedB := by
  sorry

end ThreePowerTowerIsLarger_l320_320155


namespace trader_gain_percentage_l320_320639

def pen_gain_percentage (C : ℝ) : ℝ := 
  let cost_price := 100 * C
  let gain := 30 * C
  let gain_percentage := (gain / cost_price) * 100
  gain_percentage

theorem trader_gain_percentage (C : ℝ) (hC : C > 0) : pen_gain_percentage C = 30 :=
by
  sorry

end trader_gain_percentage_l320_320639


namespace skew_lines_no_common_point_sufficient_l320_320412

-- Definitions
def two_lines_in_space (L1 L2 : Type) := (L1 ≠ L2)

def no_common_point (L1 L2 : Type) : Prop := ∀ p, ¬ (p ∈ L1 ∧ p ∈ L2)

def are_skew_lines (L1 L2 : Type) : Prop := no_common_point L1 L2 ∧ ¬ parallel L1 L2

-- Theorem statement
theorem skew_lines_no_common_point_sufficient (L1 L2 : Type) (h : two_lines_in_space L1 L2) :
  no_common_point L1 L2 → are_skew_lines L1 L2 :=
by
  sorry

end skew_lines_no_common_point_sufficient_l320_320412


namespace total_spent_on_pens_l320_320723

/-- Dorothy, Julia, and Robert go to the store to buy school supplies.
    Dorothy buys half as many pens as Julia.
    Julia buys three times as many pens as Robert.
    Robert buys 4 pens.
    The cost of one pen is $1.50.
    Prove that the total amount of money spent on pens by the three friends is $33. 
-/
theorem total_spent_on_pens :
  let cost_per_pen := 1.50
  let robert_pens := 4
  let julia_pens := 3 * robert_pens
  let dorothy_pens := julia_pens / 2
  let total_pens := robert_pens + julia_pens + dorothy_pens
  total_pens * cost_per_pen = 33 := 
by
  let cost_per_pen := 1.50
  let robert_pens := 4
  let julia_pens := 3 * robert_pens
  let dorothy_pens := julia_pens / 2
  let total_pens := robert_pens + julia_pens + dorothy_pens
  sorry

end total_spent_on_pens_l320_320723


namespace percent_of_number_l320_320169

theorem percent_of_number (x : ℝ) (p : ℝ) (d : ℝ) (h1 : p = 0.01) (h2 : d = 12356) : 
  Float.round (d * (p / 100)) = 1.24 := 
by
  sorry

end percent_of_number_l320_320169


namespace masha_numbers_unique_l320_320484

def natural_numbers : Set ℕ := {n | n > 11}

theorem masha_numbers_unique (a b : ℕ) (ha : a ∈ natural_numbers) (hb : b ∈ natural_numbers) (hne : a ≠ b)
  (hs_equals : ∃ S, S = a + b)
  (sasha_initially_uncertain : ∀ (a b : ℕ), a ∈ natural_numbers → b ∈ natural_numbers → a ≠ b → (a + b = S) → ¬ (Sasha_can_determine_initially a b S))
  (masha_hint : ∃ (a_even : ℕ), a_even ∈ natural_numbers ∧ (a_even % 2 = 0) ∧ (a_even = a ∨ a_even = b))
  (sasha_then_confident : ∀ (a b : ℕ), a ∈ natural_numbers → b ∈ natural_numbers → a ≠ b → (a + b = S) → (a_even = a ∨ a_even = b) → Sasha_can_determine_confidently a b S) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) := by
  sorry

end masha_numbers_unique_l320_320484


namespace distinct_numbers_on_board_l320_320928

def count_distinct_numbers (Mila_divisors : ℕ) (Zhenya_divisors : ℕ) (common : ℕ) : ℕ :=
  Mila_divisors + Zhenya_divisors - (common - 1)

theorem distinct_numbers_on_board :
  count_distinct_numbers 10 9 2 = 13 := by
  sorry

end distinct_numbers_on_board_l320_320928


namespace geometric_product_logarithm_l320_320886

theorem geometric_product_logarithm (n : ℕ) (hn : 0 < n) : 
  let T_n := 100 ^ ((n + 2) / 2 : ℝ) in a_n = log T_n → a_n = n + 2 :=
by
  sorry

end geometric_product_logarithm_l320_320886


namespace multiply_by_11_l320_320966

theorem multiply_by_11 (A B k : ℕ) (h1 : 10 * A + B < 100) (h2 : A + B = 10 + k) :
  (10 * A + B) * 11 = 100 * (A + 1) + 10 * k + B :=
by 
  sorry

end multiply_by_11_l320_320966


namespace ellipse_equation_slope_range_l320_320340

open Real

-- Conditions
def conditions (c a b : Real) (h : a > b) :=
  (a > 0) ∧ (b > 0) ∧
  (1 / a^2 + (9 / 4) / b^2 = 1) ∧
  (a = 2 * c) ∧
  (b^2 = (3 / 4) * a^2) ∧
  (1 - ((1 / (3 * a^2))) * ((4 / (3 * b^2 + 4)) - 2) = 0)

-- Problem 1: Prove equation of the ellipse C
theorem ellipse_equation (c a b : Real) (h : a > b) (hc : conditions c a b h) :
  (a = 2) ∧ (b = sqrt 3) ∧ (∀ (x y : Real), (x^2 / 4 + y^2 / 3 = 1 ↔ ∀ x y, x^2 ≤ 4 ∧ y^2 ≤ 3)) :=
begin
  sorry
end

-- Problem 2: Range of slope k of line MA
theorem slope_range (c a b m : Real) (h : a > b) (hc : conditions c a b h) :
  -1 / 8 ≤ (m / (4 * m^2 + 4)) ∧ (m = 0 → -1 / 8 ≤ 0 ∧ 0 ≤ 1 / 8) ∧ (m ≠ 0 → -1 / 8 ≤ 1 / (4 * m + 4 / m) ∧ 1 / (4 * m + 4 / m) ≤ 1 / 8) :=
begin
  sorry
end

end ellipse_equation_slope_range_l320_320340


namespace total_leftover_tarts_l320_320681

variable (cherry_tart blueberry_tart peach_tart : ℝ)
variable (h1 : cherry_tart = 0.08)
variable (h2 : blueberry_tart = 0.75)
variable (h3 : peach_tart = 0.08)

theorem total_leftover_tarts : 
  cherry_tart + blueberry_tart + peach_tart = 0.91 := 
by 
  sorry

end total_leftover_tarts_l320_320681


namespace perpendicular_slopes_b_l320_320089

theorem perpendicular_slopes_b (b : ℚ) : 
  (∀ x y : ℚ, 3 * x + 4 * y - 12 = 0 → (b * x + 4 * y + 5 = 0 → 
  ( (-3 / 4) * (-b / 4) = -1 ) ) ) :=
begin
  sorry
end

end perpendicular_slopes_b_l320_320089


namespace find_f_4500_l320_320756

noncomputable def f : ℕ → ℕ
| 0 => 1
| (n + 3) => f n + 2 * n + 3
| n => sorry  -- This handles all other cases, but should not be called.

theorem find_f_4500 : f 4500 = 6750001 :=
by
  sorry

end find_f_4500_l320_320756


namespace coin_flip_probability_heads_l320_320986

def flip (c : ℕ) : Prop := 
  c = 0 ∨ c = 1 -- Each coin can be either heads (1) or tails (0)

def probability_heads_penny_dime_50 : ℚ :=
  let total_outcomes := 2^5
  let favorable_outcomes := 2 * 2 -- there are 4 ways for the penny, dime, and 50-cent coin to be heads
  favorable_outcomes / total_outcomes

theorem coin_flip_probability_heads :
  probability_heads_penny_dime_50 = 1 / 8 :=
by
  sorry

end coin_flip_probability_heads_l320_320986


namespace measure_of_angle_C_l320_320433

theorem measure_of_angle_C (a b c : ℝ) (h : a^2 + b^2 - c^2 = ab) : 
  ∃ C : ℝ, 0 < C ∧ C < π ∧ cos C = 1 / 2 ∧ C = π / 3 :=
by
  sorry

end measure_of_angle_C_l320_320433


namespace schools_are_harmonious_l320_320546

-- Define the context of schools and routes as a graph
variables {V : Type} [fintype V] {E : Type} [fintype E]

-- Define a tree as a connected acyclic graph with exactly one path between any two vertices
def is_tree (G : simple_graph V) : Prop :=
  G.connected ∧ ∀ (u v : V), ∃! (p : path G u v), true

-- Define harmonious - for this context harmonious means the country forms a tree
def harmonious (G : simple_graph V) : Prop := is_tree G

-- Given condition that any two schools (vertices) are connected by exactly one direct route (path)
axiom unique_path_between_any_two_schools (G : simple_graph V) :
  ∀ (u v : V), ∃! (p : path G u v), true

-- The main theorem to prove
theorem schools_are_harmonious (G : simple_graph V) :
  harmonious G :=
begin
  apply unique_path_between_any_two_schools,
end

end schools_are_harmonious_l320_320546


namespace smallest_perimeter_of_divided_rectangle_l320_320204

theorem smallest_perimeter_of_divided_rectangle :
  ∃ (a b : ℕ), (b = 3 * a ∧ a > 0 ∧
  (2 * (2 * a + b) + 2 * (12 * a - 2 * b)) = 52) :=
begin
  sorry
end

end smallest_perimeter_of_divided_rectangle_l320_320204


namespace positive_integer_solutions_l320_320099

theorem positive_integer_solutions (x : ℕ) (h : 2 * x + 9 ≥ 3 * (x + 2)) : x = 1 ∨ x = 2 ∨ x = 3 :=
by
  sorry

end positive_integer_solutions_l320_320099


namespace sqrt_div_13_eq_4_l320_320112

theorem sqrt_div_13_eq_4 : (sqrt 2704 / 13 = 4) :=
by
  sorry

end sqrt_div_13_eq_4_l320_320112


namespace closest_point_to_origin_l320_320940

def y (x : ℝ) := x + 1 / x

theorem closest_point_to_origin : ∃ x : ℝ, x > 0 ∧ (x, y x) = (1 / 2^(1/4 : ℝ), (1 + real.sqrt 2) / 2^(1/4 : ℝ)) :=
by
  sorry

end closest_point_to_origin_l320_320940


namespace find_m_l320_320181

theorem find_m (m : ℝ) : (mx + 2y + 2 = 0) is parallel to (3x - y - 2 = 0) ⟹ m = -6 :=
begin
  -- Conditions
  let slope1 := -m / 2,
  let slope2 := 3,

  -- Question to prove
  have h : slope1 = slope2,
  cc,
  have h_m : m = -6,
  -- solving for m
  sorry,
end

end find_m_l320_320181


namespace M_on_circle_O_l320_320331

structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2)

noncomputable def A : Point := { x := 2, y := -3 }
noncomputable def M : Point := { x := 5, y := -7 }
noncomputable def O : Circle := { center := A, radius := 5 }

theorem M_on_circle_O : distance A M = O.radius :=
  sorry

end M_on_circle_O_l320_320331


namespace number_of_boys_initial_l320_320227

-- Define the initial conditions and the problem statement in Lean
def initial_boys_girls_ratio (b g : ℕ) : Prop :=
  b = 3 * some positive integer ∧ g = 4 * some positive integer

def boys_girls_transferred (b_initial g_initial b_final g_final : ℕ) : Prop :=
  b_final = b_initial - 10 ∧ g_final = g_initial - 20

def final_boys_girls_ratio (b_final g_final : ℕ) : Prop :=
  5 * b_final = 4 * g_final

-- Primary statement to verify the number of boys at the beginning of the year
theorem number_of_boys_initial (b g b_final g_final : ℕ) 
  (h1 : initial_boys_girls_ratio b g)
  (h2 : boys_girls_transferred b g b_final g_final)
  (h3 : final_boys_girls_ratio b_final g_final) : b = 90 :=
by
  sorry

end number_of_boys_initial_l320_320227


namespace compute_logarithmic_x_l320_320844

theorem compute_logarithmic_x (x : ℝ) (h : real.logb 2 (x^2) + real.logb (1/2) x = 7) : x = 128 :=
sorry

end compute_logarithmic_x_l320_320844


namespace michael_has_12_birds_l320_320918

theorem michael_has_12_birds 
  (total_pets : ℕ) 
  (dogs_percent : ℝ) 
  (cats_percent : ℝ) 
  (bunnies_percent : ℝ) 
  (total_pets_eq : total_pets = 120) 
  (dogs_percent_eq : dogs_percent = 0.30) 
  (cats_percent_eq : cats_percent = 0.40) 
  (bunnies_percent_eq : bunnies_percent = 0.20) : 
  ∃ (birds : ℕ), birds = 12 :=
begin
  sorry
end

end michael_has_12_birds_l320_320918


namespace determine_a_l320_320357

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def f (a : ℝ) (x : ℝ) : ℝ :=
  x^3 * (a * 2^x - 2^(-x))

theorem determine_a : ∃ a : ℝ, is_even_function (f a) ∧ a = 1 :=
by
  use 1
  sorry

end determine_a_l320_320357


namespace sum_of_squared_sines_equals_31_over_2_l320_320260

noncomputable def sum_of_squared_sines : ℝ :=
  (∑ n in finset.range 30, real.sin ((n + 1 : ℕ) * 6 * real.pi / 180) ^ 2)

theorem sum_of_squared_sines_equals_31_over_2 :
  sum_of_squared_sines = 31 / 2 :=
by sorry

end sum_of_squared_sines_equals_31_over_2_l320_320260


namespace zoe_total_earnings_l320_320428

theorem zoe_total_earnings
  (weeks : ℕ → ℝ)
  (weekly_hours : ℕ → ℝ)
  (wage_per_hour : ℝ)
  (h1 : weekly_hours 3 = 28)
  (h2 : weekly_hours 2 = 18)
  (h3 : weeks 3 - weeks 2 = 64.40)
  (h_same_wage : ∀ n, weeks n = weekly_hours n * wage_per_hour) :
  weeks 3 + weeks 2 = 296.24 :=
sorry

end zoe_total_earnings_l320_320428


namespace sqrt_inequality_l320_320719

theorem sqrt_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  sqrt (x / (y + 2 * z)) + sqrt (y / (2 * x + z)) + sqrt (z / (x + 2 * y)) > sqrt 3 :=
  sorry

end sqrt_inequality_l320_320719


namespace sum_of_squares_of_roots_l320_320578

theorem sum_of_squares_of_roots (α β : ℝ)
  (h_root1 : 10 * α^2 - 14 * α - 24 = 0)
  (h_root2 : 10 * β^2 - 14 * β - 24 = 0)
  (h_distinct : α ≠ β) :
  α^2 + β^2 = 169 / 25 :=
sorry

end sum_of_squares_of_roots_l320_320578


namespace placement_possible_l320_320815

structure Triangle (A B C : Type) :=
  (congruent : Triangle A B C → Triangle A B C → Prop)

def placeTriangles (H H1 : Triangle) (same_orientation : Prop) : Prop :=
  sorry

theorem placement_possible 
  {ABC A1B1C1 : Triangle}
  (congruent : Triangle.congruent ABC A1B1C1)
  (parallel_lines : ∀ (A A1 B B1 C C1 : Type), are_parallel (A B1) (A1 B) ∧ are_parallel (B C1) (B1 C) ∧ are_parallel (C A1) (C1 A)) :
  ∃ (same_orientation : Prop), placeTriangles ABC A1B1C1 same_orientation ∧ placeTriangles ABC A1B1C1 (not same_orientation) :=
sorry

end placement_possible_l320_320815


namespace determine_t_l320_320851

theorem determine_t (t : ℝ) (h : t = 1 / (3 - real.cbrt 8)) : t = 1 :=
by
  sorry

end determine_t_l320_320851


namespace evaluate_nested_function_l320_320375

def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 else 2^x

theorem evaluate_nested_function : f (f (-1)) = 1 / 4 :=
by {
  have h1 : f (-1) = 1 / 2, 
  { 
    -- here we will provide the steps or sorry for now
    sorry 
  },
  have h2 : f (1 / 2) = 1 / 4,
  { 
    -- here we will provide the steps or sorry for now
    sorry 
  },
  rw [h1, h2], 
  exact rfl
}

end evaluate_nested_function_l320_320375


namespace num_even_divisors_of_8_l320_320825

def factorial (n : Nat) : Nat :=
  match n with
  | 0     => 1
  | Nat.succ n' => Nat.succ n' * factorial n'

-- Define the prime factorization of 8!
def prime_factors_eight_factorial : Nat := 2^7 * 3^2 * 5 * 7

-- Definition of an even divisor of 8!
def is_even_divisor (d : Nat) : Prop :=
  d ∣ prime_factors_eight_factorial ∧ 2 ∣ d

-- Calculation of number of even divisors of 8!
def num_even_divisors_8! : Nat :=
  7 * 3 * 2 * 2

theorem num_even_divisors_of_8! :
  num_even_divisors_8! = 84 :=
sorry

end num_even_divisors_of_8_l320_320825


namespace circles_internally_tangent_l320_320293

-- Define the equations of the circles
def C1_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y + 1 = 0
def C2_eq (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 6*y - 39 = 0

-- Define the centers and radii derived from the equations
def C1_center : ℝ × ℝ := (2, 1)
def C2_center : ℝ × ℝ := (-1, -3)
def C1_radius : ℝ := 2
def C2_radius : ℝ := 7

-- Calculate the distance between the centers
def distance_between_centers : ℝ := Real.sqrt ((2 - (-1)) ^ 2 + (1 - (-3)) ^ 2)

-- Proof statement that the circles are internally tangent
theorem circles_internally_tangent : distance_between_centers = C2_radius - C1_radius :=
  by sorry

end circles_internally_tangent_l320_320293


namespace tangent_line_at_minus2_l320_320349

open Real

-- Define that f is an even function and differentiable
variables {f : ℝ → ℝ}
hypothesis h_even : ∀ x, f (-x) = f x
hypothesis h_diff : Differentiable ℝ f

-- Given limit condition
noncomputable 
def limit_condition : Prop := 
  ∀ L : ℝ, (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < abs x ∧ abs x < δ → abs ((f (2 + x) - f 2) / (2 * x) - L) < ε) ∧ -1 = L

-- Required proof that the tangent line at (-2,1) is y = 2x + 5
theorem tangent_line_at_minus2 : 
  limit_condition f → ∀ x : ℝ, f' (-2) = 2 → ∃ m b, f (-2) = 1 ∧ m = 2 ∧ b = 5 := by
  sorry

end tangent_line_at_minus2_l320_320349


namespace masha_numbers_l320_320487

theorem masha_numbers {a b : ℕ} (h1 : a ≠ b) (h2 : 11 < a) (h3 : 11 < b) 
  (h4 : ∃ S, S = a + b ∧ (∀ x y, x + y = S → x ≠ y → 11 < x ∧ 11 < y → 
       (¬(x = a ∧ y = b) ∧ ¬(x = b ∧ y = a)))) 
  (h5 : even a ∨ even b)
  (h6 : ∀ x y, (even x ∨ even y) → x ≠ y → 11 < x ∧ 11 < y ∧ x + y = a + b → 
       x = a ∧ y = b ∨ x = b ∧ y = a) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) :=
by
  sorry

end masha_numbers_l320_320487


namespace expression_eval_neg_sqrt_l320_320570

variable (a : ℝ)

theorem expression_eval_neg_sqrt (ha : a < 0) : a * Real.sqrt (-1 / a) = -Real.sqrt (-a) :=
by
  sorry

end expression_eval_neg_sqrt_l320_320570


namespace keun_bae_jumps_fourth_day_l320_320892

def jumps (n : ℕ) : ℕ :=
  match n with
  | 0 => 15
  | n + 1 => 2 * jumps n

theorem keun_bae_jumps_fourth_day : jumps 3 = 120 :=
by
  sorry

end keun_bae_jumps_fourth_day_l320_320892


namespace inequality1_inequality2_l320_320344

variable {a b c : ℝ}

theorem inequality1 (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) : 
  sqrt (3 * a + 1) + sqrt (3 * b + 1) + sqrt (3 * c + 1) ≤ 3 * sqrt 2 := 
sorry

theorem inequality2 (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) : 
  2 * (a^3 + b^3 + c^3) ≥ a * b + b * c + c * a - 3 * a * b * c := 
sorry

end inequality1_inequality2_l320_320344


namespace trig_identity_proof_l320_320707

theorem trig_identity_proof :
  sin (135 * Real.pi / 180) * cos (-15 * Real.pi / 180) + cos (225 * Real.pi / 180) * sin (15 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end trig_identity_proof_l320_320707


namespace coeff_x2_y7_expansion_l320_320649

-- Define the problem
theorem coeff_x2_y7_expansion (x y : ℕ) :
    (coeff ((x-y)*(x+y)^8) (x^2) (y^7) = -20) :=
sorry

end coeff_x2_y7_expansion_l320_320649


namespace length_of_field_l320_320644

variable (l w : ℝ)

theorem length_of_field : 
  (l = 2 * w) ∧ (8 * 8 = 64) ∧ ((8 * 8) = (1 / 50) * l * w) → l = 80 :=
by
  sorry

end length_of_field_l320_320644


namespace rice_weight_kg_to_pounds_l320_320599

theorem rice_weight_kg_to_pounds 
  (k : ℝ) (m : ℝ) 
  (h₁ : k = 0.454)
  (h₂ : m = 150) : 
  Int.nearest (m / k) = 330 := 
by
  sorry

end rice_weight_kg_to_pounds_l320_320599


namespace Masha_thought_of_numbers_l320_320514

theorem Masha_thought_of_numbers : ∃ a b : ℕ, a ≠ b ∧ a > 11 ∧ b > 11 ∧ (a + b = 28) ∧ (a % 2 = 0 ∨ b % 2 = 0) ∧ (a = 12 ∧ b = 16 ∨ a = 16 ∧ b = 12) :=
by
  sorry

end Masha_thought_of_numbers_l320_320514


namespace fraction_equality_l320_320088

theorem fraction_equality : (2 + 4) / (1 + 2) = 2 := by
  sorry

end fraction_equality_l320_320088


namespace projection_is_correct_l320_320370

noncomputable def projection_of_b_onto_a
  (e1 e2 : ℝ × ℝ)
  (h1 : ‖e1‖ = 1)
  (h2 : ‖e2‖ = 1)
  (h_angle : real.angle e1 e2 = real.pi * 2 / 3) 
  (a : ℝ × ℝ := (3 * e1.1, 3 * e1.2)) 
  (b : ℝ × ℝ := (e1.1 - e2.1, e1.2 - e2.2)) : ℝ := 
(a.1 * b.1 + a.2 * b.2) / 3.0

theorem projection_is_correct
  (e1 e2 : ℝ × ℝ)
  (h1 : ‖e1‖ = 1)
  (h2 : ‖e2‖ = 1)
  (h_angle : real.angle e1 e2 = real.pi * 2 / 3)
  (a : ℝ × ℝ := (3 * e1.1, 3 * e1.2)) 
  (b : ℝ × ℝ := (e1.1 - e2.1, e1.2 - e2.2)) :
  projection_of_b_onto_a e1 e2 h1 h2 h_angle = 3 / 2 := 
sorry

end projection_is_correct_l320_320370


namespace whale_tongue_weight_difference_l320_320091

noncomputable def tongue_weight_blue_whale_kg : ℝ := 2700
noncomputable def tongue_weight_fin_whale_kg : ℝ := 1800
noncomputable def kg_to_pounds : ℝ := 2.20462
noncomputable def ton_to_pounds : ℝ := 2000

noncomputable def tongue_weight_blue_whale_tons := (tongue_weight_blue_whale_kg * kg_to_pounds) / ton_to_pounds
noncomputable def tongue_weight_fin_whale_tons := (tongue_weight_fin_whale_kg * kg_to_pounds) / ton_to_pounds
noncomputable def weight_difference_tons := tongue_weight_blue_whale_tons - tongue_weight_fin_whale_tons

theorem whale_tongue_weight_difference :
  weight_difference_tons = 0.992079 :=
by
  sorry

end whale_tongue_weight_difference_l320_320091


namespace probability_of_all_genuine_given_equal_weight_l320_320602

noncomputable theory
open ProbabilityTheory MeasureTheory

-- Definitions and assumptions based on the given conditions
def genuine_coins : ℕ := 9 -- Number of genuine coins
def counterfeit_coins : ℕ := 3 -- Number of counterfeit coins
def total_coins : ℕ := genuine_coins + counterfeit_coins -- Total number of coins
def select_size : ℕ := 3 -- Number of coins to select

-- Event A: All six selected coins are genuine
def event_A : Event := 
  { ω | ∀ i < 2 * select_size, is_genuine (select ω i) }

-- Event B: The combined weight of the first set equals the combined weight of the second set
def event_B : Event := 
  { ω | combined_weight (select ω 0 3) = combined_weight (select ω 3 6) }

-- The probability that all six selected coins are genuine 
def problem : Prop :=
  P(event_A | event_B) = 1

-- The main theorem statement
theorem probability_of_all_genuine_given_equal_weight : problem := 
by sorry

end probability_of_all_genuine_given_equal_weight_l320_320602


namespace tan_double_angle_complicated_expression_l320_320424

noncomputable theory

open Real

-- Define the conditions as Lean definitions
def initial_side (α : ℝ) : Prop :=
  ∀ x : ℝ, x ≥ 0 → cos α = 1 → sin α = 0

def terminal_side (α : ℝ) : Prop :=
  ∀ x : ℝ, x ≤ 0 → x ≠ 0 → tan α = 2

-- Define the proof problems
theorem tan_double_angle (α : ℝ) (h_init : initial_side α) (h_term : terminal_side α) : 
  tan (2 * α) = -4 / 3 :=
sorry

theorem complicated_expression (α : ℝ) (h_init : initial_side α) (h_term : terminal_side α) : 
  2 * (cos (α / 2)) ^ 2 - 2 * sin (α - π) - 1) / (sqrt 2 * cos (α - 7 * π / 4)) = -5 :=
sorry

end tan_double_angle_complicated_expression_l320_320424


namespace calories_in_300g_lemonade_l320_320049

def lemonade_calories (lemon_juice_in_g : Nat) (sugar_in_g : Nat) (water_in_g : Nat) (lemon_juice_cal : Nat) (sugar_cal : Nat) : Nat :=
  (lemon_juice_in_g * lemon_juice_cal / 100) + (sugar_in_g * sugar_cal / 100)

def total_weight (lemon_juice_in_g : Nat) (sugar_in_g : Nat) (water_in_g : Nat) : Nat :=
  lemon_juice_in_g + sugar_in_g + water_in_g

theorem calories_in_300g_lemonade :
  (lemonade_calories 500 200 1000 30 400) * 300 / (total_weight 500 200 1000) = 168 := 
  by
    sorry

end calories_in_300g_lemonade_l320_320049


namespace length_of_third_side_in_triangle_l320_320871

noncomputable def length_of_third_side (a b : ℝ) (theta : ℝ) : ℝ :=
  real.sqrt (a^2 + b^2 - 2 * a * b * real.cos theta)

theorem length_of_third_side_in_triangle : 
  length_of_third_side 11 15 (150 * real.pi / 180) = real.sqrt (346 + 165 * real.sqrt 3) :=
by
  sorry

end length_of_third_side_in_triangle_l320_320871


namespace A_has_winning_strategy_l320_320601

noncomputable def winning_strategy_for_A : Prop :=
  ∃ (strategy : ∀ (turn : ℕ) (pos : ℕ), ℕ),
    (∀ (turn : ℕ) (pos : ℕ), 
      (turn = 0 ∧ strategy turn pos = 53 ∧ pos + 53 ≤ 2017) ∨
      (turn > 0 ∧ ((turn % 2 = 1 → (strategy turn pos = 53 ∧ pos + 53 ≤ 2017 ∨ strategy turn pos = 2 ∧ pos ≥ 2)) ∧
      (turn % 2 = 0 → (strategy turn pos = 53 ∧ pos + 53 ≤ 2017 ∨ strategy turn pos = 2 ∧ pos ≥ 2)))) ∧
    (∀ (pos : ℕ), strategy (40 * 2) pos + 19 * 53 - 14 = 2017)

theorem A_has_winning_strategy : winning_strategy_for_A := sorry

end A_has_winning_strategy_l320_320601


namespace find_a5_l320_320371

-- Definition of the arithmetic sequence and conditions.
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  a 1 + a 2 = 7 ∧ a 1 - a 3 = -6

-- Main theorem to be proved.
theorem find_a5 (a : ℕ → ℕ) (h : arithmetic_sequence a) : a 5 = 14 :=
by
  sorry -- This is where the proof would go.

end find_a5_l320_320371


namespace even_divisors_8_factorial_l320_320835

theorem even_divisors_8_factorial : 
  let n := (2^7) * (3^2) * 5 * 7 in
  ∃ (count : ℕ), even_divisors_count n = 84 := 
sorry

end even_divisors_8_factorial_l320_320835


namespace masha_numbers_l320_320489

theorem masha_numbers {a b : ℕ} (h1 : a ≠ b) (h2 : 11 < a) (h3 : 11 < b) 
  (h4 : ∃ S, S = a + b ∧ (∀ x y, x + y = S → x ≠ y → 11 < x ∧ 11 < y → 
       (¬(x = a ∧ y = b) ∧ ¬(x = b ∧ y = a)))) 
  (h5 : even a ∨ even b)
  (h6 : ∀ x y, (even x ∨ even y) → x ≠ y → 11 < x ∧ 11 < y ∧ x + y = a + b → 
       x = a ∧ y = b ∨ x = b ∧ y = a) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) :=
by
  sorry

end masha_numbers_l320_320489


namespace find_numbers_l320_320504

theorem find_numbers (a b : ℕ) (h1 : a > 11) (h2 : b > 11) (h3 : a ≠ b)
  (h4 : (∃ S, S = a + b) ∧ (∀ (x y : ℕ), x ≠ y → x + y = a + b → (x > 11) → (y > 11) → ¬(x = a ∨ y = a) → ¬(x = b ∨ y = b)))
  (h5 : even a ∨ even b) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) :=
by
  sorry

end find_numbers_l320_320504


namespace range_of_r_l320_320411

theorem range_of_r (r : ℝ) (h_r : r > 0) :
  let M := {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ 4}
  let N := {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 ≤ r^2}
  (∀ p, p ∈ N → p ∈ M) → 0 < r ∧ r ≤ 2 - Real.sqrt 2 :=
by
  sorry

end range_of_r_l320_320411


namespace surface_area_of_cube_l320_320772

theorem surface_area_of_cube :
  let r := 3
  let h := 3 * Real.sqrt 3
  let V_cone := (1/3) * π * r^2 * h
  let V_cube := 3 * V_cone
  let a := Real.cbrt (3 * Real.sqrt 3 * π)
  let S := 6 * a^2
  S = 5433 * π^2 := by
  have h_def : h = 3 * Real.sqrt 3 := rfl
  have V_cone_def : V_cone = (1/3) * π * r^2 * h := rfl
  have V_cube_def : V_cube = 3 * V_cone := rfl
  have a_def : a = Real.cbrt (3 * Real.sqrt 3 * π) := rfl
  have S_def : S = 6 * a^2 := rfl
  sorry

end surface_area_of_cube_l320_320772


namespace mashas_numbers_l320_320500

def is_even (n : ℕ) : Prop := n % 2 = 0

def problem_statement (a b : ℕ) : Prop :=
  a ≠ b ∧ a > 11 ∧ b > 11 ∧ is_even a ∧ a + b = 28
  
theorem mashas_numbers : ∃ (a b : ℕ), problem_statement a b :=
by
  use 12
  use 16
  unfold problem_statement
  split
  -- a ≠ b
  exact dec_trivial
  split
  -- a > 11
  exact dec_trivial
  split
  -- b > 11
  exact dec_trivial
  split
  -- is_even a
  exact dec_trivial
  -- a + b = 28
  exact dec_trivial

end mashas_numbers_l320_320500


namespace solve_equation_l320_320075

theorem solve_equation :
  ∀ x : ℝ, (101 * x ^ 2 - 18 * x + 1) ^ 2 - 121 * x ^ 2 * (101 * x ^ 2 - 18 * x + 1) + 2020 * x ^ 4 = 0 ↔ 
    x = 1 / 18 ∨ x = 1 / 9 :=
by
  intro x
  sorry

end solve_equation_l320_320075


namespace simplify_sum_identity_l320_320797

def f (x : ℝ) : ℝ := x / (1 + x)

theorem simplify_sum_identity : f (1 / 9) + f (1 / 7) + f (1 / 5) + f (1 / 3) + f 1 + f 3 + f 5 + f 7 + f 9 = 9 / 2 :=
by
  -- Conditions to be used
  have fx_identity : ∀ (x : ℝ), x ≠ 0 → f x + f (1 / x) = 1 := λ x x_ne_zero, 
    by
      have h₁ : f x = x / (1 + x), from rfl
      have h₂ : f (1 / x) = (1 / x) / ((1 / x) + 1), from rfl
      rewrite [h₁, h₂]
      field_simp
  have f1_eq : f 1 = 1 / 2 := by 
    unfold f 
    norm_num

  sorry

end simplify_sum_identity_l320_320797


namespace greatest_partition_l320_320718

-- Define the condition on the partitions of the positive integers
def satisfies_condition (A : ℕ → Prop) (n : ℕ) : Prop :=
∃ a b : ℕ, a ≠ b ∧ A a ∧ A b ∧ a + b = n

-- Define what it means for k subsets to meet the requirements
def partition_satisfies (k : ℕ) : Prop :=
∃ A : ℕ → ℕ → Prop,
  (∀ i : ℕ, i < k → ∀ n ≥ 15, satisfies_condition (A i) n)

-- Our conjecture is that k can be at most 3 for the given condition
theorem greatest_partition (k : ℕ) : k ≤ 3 :=
sorry

end greatest_partition_l320_320718


namespace number_of_white_balls_l320_320874

theorem number_of_white_balls (total_balls : ℕ) (red_prob black_prob : ℝ)
  (h_total : total_balls = 50)
  (h_red_prob : red_prob = 0.15)
  (h_black_prob : black_prob = 0.45) :
  ∃ (white_balls : ℕ), white_balls = 20 :=
by
  sorry

end number_of_white_balls_l320_320874


namespace exists_coprime_with_210_l320_320964

theorem exists_coprime_with_210 (a : ℕ) : ∃ k ∈ (list.range 11).map (λ n, a + n), Nat.gcd k 210 = 1 :=
by
  sorry

end exists_coprime_with_210_l320_320964


namespace joe_first_lift_weight_l320_320640

variable (x y : ℕ)

def joe_lift_conditions (x y : ℕ) : Prop :=
  x + y = 600 ∧ 2 * x = y + 300

theorem joe_first_lift_weight (x y : ℕ) (h : joe_lift_conditions x y) : x = 300 :=
by
  sorry

end joe_first_lift_weight_l320_320640


namespace part1_l320_320647

variable (f : ℝ → ℝ)
variable (x1 x2 : ℝ)
variable (h0 : ∀ x, 0 ≤ x → f x = Real.sqrt x)
variable (h1 : 0 ≤ x1)
variable (h2 : 0 ≤ x2)
variable (h3 : x1 ≠ x2)

theorem part1 : (1/2) * (f x1 + f x2) < f ((x1 + x2) / 2) :=
  sorry

end part1_l320_320647


namespace sum_of_squared_sines_equals_31_over_2_l320_320255

noncomputable def sum_of_squared_sines : ℝ :=
  (∑ n in finset.range 30, real.sin ((n + 1 : ℕ) * 6 * real.pi / 180) ^ 2)

theorem sum_of_squared_sines_equals_31_over_2 :
  sum_of_squared_sines = 31 / 2 :=
by sorry

end sum_of_squared_sines_equals_31_over_2_l320_320255


namespace vertex_angle_is_130_8_l320_320870

-- Define the given conditions
variables {a b h : ℝ}

def is_isosceles_triangle (a b h : ℝ) : Prop :=
  a^2 = b * 3 * h ∧ b = 2 * h

-- Define the obtuse condition on the vertex angle
def vertex_angle_obtuse (a b h : ℝ) : Prop :=
  ∃ θ : ℝ, 120 < θ ∧ θ < 180 ∧ θ = (130.8 : ℝ)

-- The formal proof statement using Lean 4
theorem vertex_angle_is_130_8 (a b h : ℝ) 
  (h1 : is_isosceles_triangle a b h)
  (h2 : vertex_angle_obtuse a b h) : 
  ∃ (φ : ℝ), φ = 130.8 :=
sorry

end vertex_angle_is_130_8_l320_320870


namespace tan_sum_identity_l320_320348

theorem tan_sum_identity (α : ℝ) (h : Real.tan α = 1 / 2) : Real.tan (α + π / 4) = 3 := 
by 
  sorry

end tan_sum_identity_l320_320348


namespace even_divisors_of_8fac_l320_320822

theorem even_divisors_of_8fac : 
  let num_even_divisors := ∏ x in {a | 1 ≤ a ∧ a ≤ 7}.card * 
                                      {b | 0 ≤ b ∧ b ≤ 2}.card *
                                      {c | 0 ≤ c ∧ c ≤ 1}.card *
                                      {d | 0 ≤ d ∧ d ≤ 1}.card
  in num_even_divisors = 84 := by
  sorry

end even_divisors_of_8fac_l320_320822


namespace prove_m_range_l320_320387

-- Definitions of sets A and B
def SetA (m : ℝ) : set (ℝ × ℝ) := {p | ∃ x, p = (x, x^2 + m * x + 2)}
def SetB : set (ℝ × ℝ) := {p | ∃ x, p = (x, x + 1)}

-- The intersection condition for sets A and B
def intersection_non_empty (m : ℝ) : Prop :=
  ∃ x : ℝ, (x, x^2 + m * x + 2) ∈ SetB

-- Proof problem statement
theorem prove_m_range (m : ℝ) : intersection_non_empty m ↔ m ∈ {x : ℝ | x ≤ -1} ∪ {x : ℝ | x ≥ 3} := sorry

end prove_m_range_l320_320387


namespace probability_correct_l320_320025

noncomputable def probability_at_least_one_ge_2500 (a b c : ℕ) : ℚ :=
  if (a + b + c = 3000 ∧ a > 0 ∧ b > 0 ∧ c > 0) then (374250 : ℚ) / 4498501 else 0

theorem probability_correct :
  ∀ (a b c : ℕ), (a + b + c = 3000 ∧ a > 0 ∧ b > 0 ∧ c > 0) → 
                  probability_at_least_one_ge_2500 a b c = (374250 : ℚ) / 4498501 :=
begin
  intros a b c h,
  simp [probability_at_least_one_ge_2500, h]
end

end probability_correct_l320_320025


namespace masha_numbers_l320_320518

theorem masha_numbers (a b : ℕ) (S : ℕ) (h1 : a ≠ b) (h2 : a > 11) (h3 : b > 11) (h4 : S = a + b) 
    (h5 : (∀ x y : ℕ, x + y = S → x = a ∨ y = a → abs x - y = a) ∧ (even a ∨ even b)) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) := 
by sorry

end masha_numbers_l320_320518


namespace cows_grazed_by_B_l320_320185

theorem cows_grazed_by_B
    (A_cows : ℕ := 24)
    (A_months : ℕ := 3)
    (B_months : ℕ := 5)
    (C_cows : ℕ := 35)
    (C_months : ℕ := 4)
    (D_cows : ℕ := 21)
    (D_months : ℕ := 3)
    (A_share_rent : ℕ := 720)
    (total_rent : ℕ := 3250) :
    let A_cow_months := A_cows * A_months,
        B_cows := 10,
        C_cow_months := C_cows * C_months,
        D_cow_months := D_cows * D_months,
        total_cow_months := A_cow_months + B_cows * B_months + C_cow_months + D_cow_months
    in
    (A_share_rent * total_cow_months = A_cow_months * total_rent) :=
by sorry

end cows_grazed_by_B_l320_320185


namespace tile_difference_l320_320885

theorem tile_difference (Initial_blue Initial_green Additional_green_tiles_per_border Number_of_borders : ℕ) 
    (h1 : Initial_blue = 15)
    (h2 : Initial_green = 6)
    (h3 : Additional_green_tiles_per_border = 18)
    (h4 : Number_of_borders = 2) :
    (Initial_green + (Additional_green_tiles_per_border * Number_of_borders)) - Initial_blue = 27 :=
by 
    rw [h1, h2, h3, h4]
    sorry

end tile_difference_l320_320885


namespace binary_sequence_fibonacci_l320_320713

def fibonacci : ℕ → ℕ 
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n+1) + fibonacci n

def binary_no_two_consecutive_ones : ℕ → ℕ 
| 0     := 1
| 1     := 2
| (n+2) := binary_no_two_consecutive_ones (n+1) + binary_no_two_consecutive_ones n

theorem binary_sequence_fibonacci (n : ℕ) : 
  binary_no_two_consecutive_ones n = fibonacci (n + 2) :=
sorry

end binary_sequence_fibonacci_l320_320713


namespace alicia_average_speed_correct_l320_320686

/-
Alicia drove 320 miles in 6 hours.
Alicia drove another 420 miles in 7 hours.
Prove Alicia's average speed for the entire journey is 56.92 miles per hour.
-/

def alicia_total_distance : ℕ := 320 + 420
def alicia_total_time : ℕ := 6 + 7
def alicia_average_speed : ℚ := alicia_total_distance / alicia_total_time

theorem alicia_average_speed_correct : alicia_average_speed = 56.92 :=
by
  -- Proof goes here
  sorry

end alicia_average_speed_correct_l320_320686


namespace masha_numbers_l320_320492

theorem masha_numbers {a b : ℕ} (h1 : a ≠ b) (h2 : 11 < a) (h3 : 11 < b) 
  (h4 : ∃ S, S = a + b ∧ (∀ x y, x + y = S → x ≠ y → 11 < x ∧ 11 < y → 
       (¬(x = a ∧ y = b) ∧ ¬(x = b ∧ y = a)))) 
  (h5 : even a ∨ even b)
  (h6 : ∀ x y, (even x ∨ even y) → x ≠ y → 11 < x ∧ 11 < y ∧ x + y = a + b → 
       x = a ∧ y = b ∨ x = b ∧ y = a) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) :=
by
  sorry

end masha_numbers_l320_320492


namespace ant_sequence_count_l320_320692

-- Define the sequences of centers passed through as a sequence of visits for the ant on a cube.
variable (C : Type) [Fintype C] [DecidableEq C]

-- Define the condition of moving to adjacent faces and visiting each exactly once, returning to start.
def valid_sequences (s : List C) : Prop :=
  s.head = s.getLast (ne_of_gt (by simp)) ∧
  (List.nodup s) ∧
  (∀ i < s.length - 1, (s.nth i).some ≠ (s.nth (i + 1)).some) ∧
  (∀ x, List.count x s = 1)

-- Given a cube with 6 faces, calculate the total number of valid sequences given the conditions.
theorem ant_sequence_count : ∑ s in (Fintype.piFinset (λ _, Fin 6)), If valid_sequences s.length = 7 then 1 else 0 = 32 :=
  by
    sorry

end ant_sequence_count_l320_320692


namespace real_solutions_of_equation_l320_320314

theorem real_solutions_of_equation :
  (∃! x : ℝ, (5 * x) / (x^2 + 2 * x + 4) + (6 * x) / (x^2 - 6 * x + 4) = -4 / 3) :=
sorry

end real_solutions_of_equation_l320_320314


namespace find_original_radius_l320_320726

-- Definitions based on the conditions in the problem
def volume_with_increased_radius (r : ℝ) : ℝ := π * (r + 7)^2 * 3
def volume_with_increased_height (r : ℝ) : ℝ := π * r^2 * 10
def original_height : ℝ := 3

theorem find_original_radius (r : ℝ) (h1 : volume_with_increased_radius r = volume_with_increased_height r) : r = 7 :=
by
  sorry

end find_original_radius_l320_320726


namespace calculate_t_l320_320609

theorem calculate_t :
  let A := (0, 8)
  let B := (2, 0)
  let C := (10, 0)
  ∃ t : ℝ, 
    let T := (1 / 4 * (8 - t), t)
    let U := (5 / 4 * (8 - t), t)
    1 / 2 * (8 - t)^2 = 20 ∧ t = 8 - 2 * Real.sqrt 10 :=
by
  let A := (0, 8)
  let B := (2, 0)
  let C := (10, 0)
  let T := (1 / 4 * (8 - t), t)
  let U := (5 / 4 * (8 - t), t)
  let area_ATU := 1 / 2 * (8 - t)^2
  use t
  split
  sorry
  sorry

end calculate_t_l320_320609


namespace prove_problem_conditions_l320_320026

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def problem_conditions (S : Set ℕ) (n : ℕ) :=
  ∃ (subset : Finset ℕ), 
    subset ⊆ (S : Set ℕ) ∧
    subset.card = 10 ∧
    ∀ (partition : Finset.div2 subset) (A B : Finset ℕ),
      A.card = 5 ∧ B.card = 5 ∧ 
      (A ∪ B = subset) ∧ (A ∩ B = ∅) →
      (∃ a ∈ A, ∀ b ∈ A, a ≠ b → is_coprime a b) ∧ 
      (∃ b ∈ B, ∀ a ∈ B, b ≠ a → ¬ is_coprime b a)

theorem prove_problem_conditions : 
  ∀ (S : Set ℕ), S = {1, 2, ..., 98} → 
  ∃ (n : ℕ), 
    n = 50 ∧
    problem_conditions S n := sorry

end prove_problem_conditions_l320_320026


namespace complement_A_inter_B_range_of_a_l320_320390

open Set

-- Define sets A and B based on the conditions
def A : Set ℝ := {x | -4 ≤ x - 6 ∧ x - 6 ≤ 0}
def B : Set ℝ := {x | 2 * x - 6 ≥ 3 - x}

-- Define set C based on the conditions
def C (a : ℝ) : Set ℝ := {x | x ≤ a}

-- Problem 1: Prove the complement of (A ∩ B) in ℝ is the set of x where (x < 3 or x > 6)
theorem complement_A_inter_B :
  compl (A ∩ B) = {x | x < 3} ∪ {x | x > 6} :=
sorry

-- Problem 2: Prove that A ∩ C = A implies a ∈ [6, ∞)
theorem range_of_a {a : ℝ} (hC : A ∩ C a = A) :
  6 ≤ a :=
sorry

end complement_A_inter_B_range_of_a_l320_320390


namespace statement_A_true_l320_320631

-- Define the conditions
def cofunctions_equal (θ : ℝ) : Prop :=
  sin θ = cos (90 - θ)

-- Define Statement A
theorem statement_A_true (θ : ℝ) : cofunctions_equal θ :=
  by sorry

end statement_A_true_l320_320631


namespace space_divided_by_five_spheres_l320_320014

-- Define a_n as the maximum number of regions a sphere can be divided by n circles
def a : ℕ → ℕ
| 2 := 2
| (n + 1) := a n + 2 * n

-- Define b_n as the maximum number of regions the space can be divided by n spheres
def b : ℕ → ℕ
| 2 := 2
| (n + 1) := b n + a n

theorem space_divided_by_five_spheres : b 5 = 22 :=
by {
    -- In practice, we would complete this proof using the definitions and calculations provided.
    sorry
}

end space_divided_by_five_spheres_l320_320014


namespace determine_numbers_l320_320475

theorem determine_numbers (a b : ℕ) (S : ℕ) (h1 : a ≠ b) (h2 : a > 11) (h3 : b > 11)
  (h4 : S = a + b) (h5 : (∀ (x y : ℕ), x + y = S → x ≠ y → (x = a ∧ y = b) ∨ (x = b ∧ y = a)) = false)
  (h6 : a % 2 = 0 ∨ b % 2 = 0) : 
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) := 
  sorry

end determine_numbers_l320_320475


namespace cooper_pies_days_l320_320283

theorem cooper_pies_days :
  ∃ d : ℕ, 7 * d - 50 = 34 ∧ d = 12 :=
by
  sorry

end cooper_pies_days_l320_320283


namespace sheets_in_backpack_l320_320046

-- Definitions for the conditions
def total_sheets := 91
def desk_sheets := 50

-- Theorem statement with the goal
theorem sheets_in_backpack (total_sheets : ℕ) (desk_sheets : ℕ) (h1 : total_sheets = 91) (h2 : desk_sheets = 50) : 
  ∃ backpack_sheets : ℕ, backpack_sheets = total_sheets - desk_sheets ∧ backpack_sheets = 41 :=
by
  -- The proof is omitted here
  sorry

end sheets_in_backpack_l320_320046


namespace proper_subsets_count_is_seven_l320_320592

def set_123 := {1, 2, 3}
def is_proper_subset (s t : Set ℕ) : Prop := s ⊆ t ∧ s ≠ t
noncomputable def proper_subsets_count : ℕ := {
  count := (Finset.powerset (Finset.insert 1 (Finset.insert 2 (Finset.singleton 3)))).filter (λ s, s.card < 3)
}.card - 1

theorem proper_subsets_count_is_seven : proper_subsets_count = 7 := by
  sorry

end proper_subsets_count_is_seven_l320_320592


namespace find_number_l320_320436

theorem find_number 
  (a b c d : ℤ)
  (h1 : 0 ≤ a ∧ a ≤ 9)
  (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : 0 ≤ c ∧ c ≤ 9)
  (h4 : 0 ≤ d ∧ d ≤ 9)
  (h5 : 6 * a + 9 * b + 3 * c + d = 88)
  (h6 : a - b + c - d = -6)
  (h7 : a - 9 * b + 3 * c - d = -46) : 
  1000 * a + 100 * b + 10 * c + d = 6507 := 
sorry

end find_number_l320_320436


namespace even_function_a_value_l320_320353

theorem even_function_a_value {f : ℝ → ℝ} (a : ℝ) :
  (∀ x : ℝ, f x = x^3 * (a * 2^x - 2^(-x)) ∧ f x = f (-x)) → a = 1 :=
by
  intros h,
  sorry

end even_function_a_value_l320_320353


namespace proposition1_proposition2_proposition3_proposition4_correct_propositions_l320_320716

theorem proposition1 (x : ℝ) : ¬periodic (λ x, sin (abs x)) :=
sorry

theorem proposition2 (x : ℝ) : ¬monotone (λ x, tan x) :=
sorry

theorem proposition3 : ∃ T : ℝ, T = π / 2 ∧ ∀ x : ℝ, abs (cos (2 * x)) = abs (cos (2 * (x + T))) :=
sorry

theorem proposition4 : ∃ x : ℝ, x = -π / 6 ∧ (2 * sin (2 * x + π / 3)) = 0 :=
sorry

theorem correct_propositions : (proposition1, proposition3, proposition4) := 
sorry

end proposition1_proposition2_proposition3_proposition4_correct_propositions_l320_320716


namespace higher_degree_polynomial_solutions_do_not_continue_indefinitely_l320_320590

theorem higher_degree_polynomial_solutions_do_not_continue_indefinitely:
  (∀ a b c : ℝ, (∃ x y : ℝ, x = (-b + real.sqrt(b^2 - 4*a*c)) / (2*a) 
    ∨ x = (-b - real.sqrt(b^2 - 4*a*c)) / (2*a) ∨ y = x)) → 
  (∀ a b c d : ℝ, (∃ t u v : ℝ, -- conditions related to cubic equations are complex (roots involve cube roots) 
    ∧ ∃ x y : ℝ, -- solutions to cubic equations can be written in terms of cube roots 
    x = (stok:={}) → _) →
  (∀ a b c d e : ℝ, (∃ u v w t : ℝ, -- solutions to quartic equations involve fourth degree roots 
    ∧ ∃ x y : ℝ, -- solutions to quartic equations can be written in terms of fourth roots
    x = (stok:={}) → _) →
  ¬(∀ n ≥ 5 (zero_le_n : 0 ≤ n) (degree_eq_n : degree f = n), 
  ∃ a₀ a₁ ... aₙ : ℝ, ∃ x : ℝ, -- there exist polynomials of any higher degree n ≥ 5 
  x = (stok:={})) :=
by {
  sorry
}

end higher_degree_polynomial_solutions_do_not_continue_indefinitely_l320_320590


namespace solve_for_y_l320_320153

theorem solve_for_y 
  (a b c d y : ℚ) 
  (h₀ : a ≠ b) 
  (h₁ : a ≠ 0) 
  (h₂ : c ≠ d) 
  (h₃ : (b + y) / (a + y) = d / c) : 
  y = (a * d - b * c) / (c - d) :=
by
  sorry

end solve_for_y_l320_320153


namespace maximize_profit_l320_320655

noncomputable def I (x : ℝ) : ℝ :=
  if h : 0 < x ∧ x ≤ 2 then 2 * (x - 1) * Real.exp (x - 2) + 2
  else if h' : 2 < x ∧ x ≤ 50 then 440 + 3050 / x - 9000 / x^2
  else 0 -- default case for Lean to satisfy definition

noncomputable def P (x : ℝ) : ℝ :=
  if h : 0 < x ∧ x ≤ 2 then 2 * x * (x - 1) * Real.exp (x - 2) - 448 * x - 180
  else if h' : 2 < x ∧ x ≤ 50 then -10 * x - 9000 / x + 2870
  else 0 -- default case for Lean to satisfy definition

theorem maximize_profit :
  (∀ x : ℝ, 0 < x ∧ x ≤ 50 → P x ≤ 2270) ∧ P 30 = 2270 :=
by
  sorry

end maximize_profit_l320_320655


namespace smaller_hemisphere_radius_l320_320661

noncomputable def volume_of_hemisphere (r : ℝ) : ℝ := (2 / 3) * real.pi * r^3

theorem smaller_hemisphere_radius :
  ∀ (r : ℝ),
  let large_volume := volume_of_hemisphere 2 in
  let small_volume := 64 * volume_of_hemisphere r in
  large_volume = small_volume → r = 1 / 2 :=
by
  intros r large_volume small_volume H
  have volume_eq : volume_of_hemisphere 2 = 64 * volume_of_hemisphere r,
    from H
  sorry

end smaller_hemisphere_radius_l320_320661


namespace closest_point_to_origin_l320_320959

theorem closest_point_to_origin : 
  ∃ x y : ℝ, x > 0 ∧ y = x + 1/x ∧ (x, y) = (1/(2^(1/4)), (1 + 2^(1/2))/(2^(1/4))) :=
by
  sorry

end closest_point_to_origin_l320_320959


namespace probability_exactly_two_heads_l320_320658

theorem probability_exactly_two_heads :
  (nat.choose 6 2) / (2^6) = (15 : ℚ) / 64 :=
by {
  sorry
}

end probability_exactly_two_heads_l320_320658


namespace probability_green_or_yellow_l320_320199

noncomputable def probability_red : ℝ := 0.25
noncomputable def probability_orange : ℝ := 0.35
noncomputable def probability_sum_is_one (P_Y P_G : ℝ) : Prop :=
  probability_red + probability_orange + P_Y + P_G = 1

theorem probability_green_or_yellow {P_Y P_G : ℝ} (h : probability_sum_is_one P_Y P_G) :
  P_Y + P_G = 0.40 :=
by
  simp [probability_sum_is_one] at h
  linarith

end probability_green_or_yellow_l320_320199


namespace closest_point_l320_320953

noncomputable def closest_point_to_origin : ℝ × ℝ :=
  let x := (1 : ℝ) / Real.root 2 4 in
  let y := x + 1 / x in
  (x, y)

theorem closest_point (x y : ℝ) (h : y = x + 1 / x) (hx : x > 0) :
  (x, y) = closest_point_to_origin :=
begin
  sorry
end

end closest_point_l320_320953


namespace point_B_is_4_l320_320063

def point_A : ℤ := -3
def units_to_move : ℤ := 7
def point_B : ℤ := point_A + units_to_move

theorem point_B_is_4 : point_B = 4 :=
by
  sorry

end point_B_is_4_l320_320063


namespace prime_p_squared_plus_71_divisors_l320_320040

open Nat

def is_prime (n : ℕ) : Prop := Nat.Prime n

def num_distinct_divisors (n : ℕ) : ℕ :=
  (factors n).toFinset.card

theorem prime_p_squared_plus_71_divisors (p : ℕ) (hp : is_prime p) 
  (hdiv : num_distinct_divisors (p ^ 2 + 71) ≤ 10) : p = 2 ∨ p = 3 :=
sorry

end prime_p_squared_plus_71_divisors_l320_320040


namespace marla_errand_time_l320_320047

theorem marla_errand_time :
  let drive_time_one_way := 20
  let parent_teacher_night := 70
  let drive_time_total := drive_time_one_way * 2
  let total_time := drive_time_total + parent_teacher_night
  total_time = 110 := by
  let drive_time_one_way := 20
  let parent_teacher_night := 70
  let drive_time_total := drive_time_one_way * 2
  let total_time := drive_time_total + parent_teacher_night
  sorry

end marla_errand_time_l320_320047


namespace cookies_in_jar_l320_320129

noncomputable def number_of_cookies_in_jar : ℕ := sorry

theorem cookies_in_jar :
  (number_of_cookies_in_jar - 1) = (1 / 2 : ℝ) * (number_of_cookies_in_jar + 5) →
  number_of_cookies_in_jar = 7 :=
by
  sorry

end cookies_in_jar_l320_320129


namespace trajectory_eq_of_M_l320_320776

-- Define the coordinates of points A and B
def A := (-1 : ℝ, 0 : ℝ)
def B := (1 : ℝ, 0 : ℝ)

-- Deduces the equation of the trajectory of point M given the conditions
theorem trajectory_eq_of_M (x y : ℝ) (h : (y / (x + 1) - y / (x - 1) = 2)) :
    y = 1 - x^2 ∧ x ≠ 1 ∧ x ≠ -1 :=
by
  sorry

end trajectory_eq_of_M_l320_320776


namespace cookies_in_jar_l320_320124

noncomputable def C : ℕ := sorry

theorem cookies_in_jar (h : C - 1 = (C + 5) / 2) : C = 7 := by
  sorry

end cookies_in_jar_l320_320124


namespace number_of_even_divisors_of_factorial_eight_l320_320829

-- Definition of 8! and its prime factorization
def factorial_eight : ℕ := 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1
def prime_factorization_factorial_eight : Prop :=
  factorial_eight = 2^7 * 3^2 * 5 * 7

-- The main theorem statement
theorem number_of_even_divisors_of_factorial_eight :
  prime_factorization_factorial_eight →
  ∃ n, n = 7 * 3 * 2 * 2 ∧
  (∀ d, d ∣ factorial_eight → (∃ a b c d, 1 ≤ a ∧ a ≤ 7 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 1 ∧ 0 ≤ d ∧ d ≤ 1 ∧ d = 2^a * 3^b * 5^c * 7^d) →
  (7 * 3 * 2 * 2 = n)) :=
by
  intro h
  use 84
  sorry

end number_of_even_divisors_of_factorial_eight_l320_320829


namespace second_car_speed_correct_l320_320612

noncomputable def first_car_speed : ℝ := 90

noncomputable def time_elapsed (h : ℕ) (m : ℕ) : ℝ := h + m / 60

noncomputable def distance_travelled (speed : ℝ) (time : ℝ) : ℝ := speed * time

def distance_ratio_at_832 (dist1 dist2 : ℝ) : Prop := dist1 = 1.2 * dist2
def distance_ratio_at_920 (dist1 dist2 : ℝ) : Prop := dist1 = 2 * dist2

noncomputable def time_first_car_832 : ℝ := time_elapsed 0 24
noncomputable def dist_first_car_832 : ℝ := distance_travelled first_car_speed time_first_car_832

noncomputable def dist_second_car_832 : ℝ := dist_first_car_832 / 1.2

noncomputable def time_first_car_920 : ℝ := time_elapsed 1 12
noncomputable def dist_first_car_920 : ℝ := distance_travelled first_car_speed time_first_car_920

noncomputable def dist_second_car_920 : ℝ := dist_first_car_920 / 2

noncomputable def time_second_car_travel : ℝ := time_elapsed 0 42

noncomputable def second_car_speed : ℝ := (dist_second_car_920 - dist_second_car_832) / time_second_car_travel

theorem second_car_speed_correct :
  second_car_speed = 34.2857 := by
  sorry

end second_car_speed_correct_l320_320612


namespace triangle_incircle_distance_l320_320011

open real

variables {A B C P : Point} (r : ℝ) (incircle_radius : ∀ Δ : Triangle, r Δ)
variables (inside_triangle : ∀ Δ : Triangle, Point → Prop)

-- Theorem statement
theorem triangle_incircle_distance (Δ : Triangle) (hΔ : inside_triangle Δ P)
  (hΔr : incircle_radius Δ = r) :
  distance P A + distance P B + distance P C ≥ 6 * r :=
sorry

end triangle_incircle_distance_l320_320011


namespace sum_sin_squares_deg_6_to_174_l320_320235

theorem sum_sin_squares_deg_6_to_174 : 
  ∑ k in Finset.range 30, Real.sin (6 * k * (Real.pi / 180)) ^ 2 = 15.5 := by
    sorry

end sum_sin_squares_deg_6_to_174_l320_320235


namespace angle_BHC_l320_320434

-- Define the triangle and its angles.
variables (A B C D E F H : Type) [angle_structure : has_angle A B C D E F H]

-- Given angles
axiom angle_ABC : angle_structure.angle ABC = 58
axiom angle_ACB : angle_structure.angle ACB = 20

-- Triangle ABC with orthocenter H
axiom orthocenter_condition : angle_structure.orthocenter A B C = H

-- Problem statement to prove:
theorem angle_BHC (ABC_angle_58 : angle_structure.angle ABC = 58)
                  (ACB_angle_20 : angle_structure.angle ACB = 20)
                  (H_orthocenter : angle_structure.orthocenter A B C = H) :
              angle_structure.angle BHC = 78 :=
by sorry

end angle_BHC_l320_320434


namespace vector_magnitude_l320_320818

variables (x : ℝ)
def a := (x, -1 : ℝ × ℝ)
def b := (1, real.sqrt 3 : ℝ × ℝ)

theorem vector_magnitude : (a.1 * b.1 + a.2 * b.2 = 0) → real.sqrt (a.1^2 + a.2^2) = 2 :=
by
  sorry

end vector_magnitude_l320_320818


namespace find_mn_solutions_l320_320739

theorem find_mn_solutions :
  ∀ (m n : ℤ), m^5 - n^5 = 16 * m * n →
  (m = 0 ∧ n = 0) ∨ (m = -2 ∧ n = 2) :=
by
  sorry

end find_mn_solutions_l320_320739


namespace johnson_family_seating_l320_320558

theorem johnson_family_seating : 
  let boys : Nat := 5,
      girls : Nat := 4,
      chairs : Nat := 9 in
  let total_seatings := Nat.factorial chairs in
  ∀ seating: Finset (Fin chairs → FiniteType (Sum (Fin boys) (Fin girls))),
    total_seatings - 0 = 362880 := by 
  sorry

end johnson_family_seating_l320_320558


namespace find_numbers_l320_320503

theorem find_numbers (a b : ℕ) (h1 : a > 11) (h2 : b > 11) (h3 : a ≠ b)
  (h4 : (∃ S, S = a + b) ∧ (∀ (x y : ℕ), x ≠ y → x + y = a + b → (x > 11) → (y > 11) → ¬(x = a ∨ y = a) → ¬(x = b ∨ y = b)))
  (h5 : even a ∨ even b) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) :=
by
  sorry

end find_numbers_l320_320503


namespace original_radius_of_cylinder_l320_320727

theorem original_radius_of_cylinder : 
  ∀ (r : ℝ), ∀ (h : ℝ), ∀ (increase : ℝ),
    h = 3 → increase = 7 →
    (π * (r + increase) ^ 2 * h = π * r ^ 2 * (h + increase)) →
    r = 7 := 
by
  intros r h increase h_eq increase_eq volume_eq.
  sorry

end original_radius_of_cylinder_l320_320727


namespace number_of_valid_x_l320_320392

theorem number_of_valid_x :
  (∃ x : ℕ, 3 * x < 100 ∧ 4 * x ≥ 100) →
  (finset.Icc 25 33).card = 9 :=
by
  intro h
  sorry

end number_of_valid_x_l320_320392


namespace age_of_new_person_l320_320167

theorem age_of_new_person (avg_age : ℝ) (x : ℝ) 
  (h1 : 10 * avg_age - (10 * (avg_age - 3)) = 42 - x) : 
  x = 12 := 
by
  sorry

end age_of_new_person_l320_320167


namespace not_approximately_equal_exp_l320_320299

noncomputable def multinomial_approximation (n k₁ k₂ k₃ k₄ k₅ : ℕ) : ℝ :=
  (n.factorial : ℝ) / ((k₁.factorial : ℝ) * (k₂.factorial : ℝ) * (k₃.factorial : ℝ) * (k₄.factorial : ℝ) * (k₅.factorial : ℝ))

theorem not_approximately_equal_exp (e : ℝ) (h1 : e > 0) :
  e ^ 2737 ≠ multinomial_approximation 1000 70 270 300 220 140 :=
by 
  sorry  

end not_approximately_equal_exp_l320_320299


namespace harmonic_quadrilateral_l320_320422

theorem harmonic_quadrilateral
  (O : Circle)
  (A B C H D M E F : Point)
  (h1 : ∠A < 90 ∧ ∠B < 90 ∧ ∠C < 90)
  (h2 : O.inscribed_triangle A B C)
  (h3 : ∀ P : Point, is_orthocenter H A B C)
  (h4 : line_intersect AH BC D)
  (h5 : midpoint M B C)
  (h6 : line_extend_intersect MH O E)
  (h7 : line_extend_intersect ED O F) :
  harmonic_quadrilateral A B F C :=
sorry

end harmonic_quadrilateral_l320_320422


namespace Masha_thought_of_numbers_l320_320511

theorem Masha_thought_of_numbers : ∃ a b : ℕ, a ≠ b ∧ a > 11 ∧ b > 11 ∧ (a + b = 28) ∧ (a % 2 = 0 ∨ b % 2 = 0) ∧ (a = 12 ∧ b = 16 ∨ a = 16 ∧ b = 12) :=
by
  sorry

end Masha_thought_of_numbers_l320_320511


namespace Masha_thought_of_numbers_l320_320509

theorem Masha_thought_of_numbers : ∃ a b : ℕ, a ≠ b ∧ a > 11 ∧ b > 11 ∧ (a + b = 28) ∧ (a % 2 = 0 ∨ b % 2 = 0) ∧ (a = 12 ∧ b = 16 ∨ a = 16 ∧ b = 12) :=
by
  sorry

end Masha_thought_of_numbers_l320_320509


namespace solve_system_eqs_l320_320981

theorem solve_system_eqs : 
    ∃ (x y z : ℚ), 
    4 * x - 3 * y + z = -10 ∧
    3 * x + 5 * y - 2 * z = 8 ∧
    x - 2 * y + 7 * z = 5 ∧
    x = -51 / 61 ∧ 
    y = 378 / 61 ∧ 
    z = 728 / 61 := by
  sorry

end solve_system_eqs_l320_320981


namespace tension_in_string_force_on_board_l320_320653

variables (M m g μ a T F : ℝ)

-- Given conditions
def board_mass := M = 4
def block_mass := m = 1
def gravity := g = 10
def acceleration := a = g / 5
def friction_coefficient := μ = 0.2

-- Prove the tension in the string
theorem tension_in_string :
  board_mass M →
  block_mass m →
  gravity g →
  acceleration a →
  friction_coefficient μ →
  T = m * (a + μ * g) →
  T = 4 :=
by
  sorry

-- Prove the force acting on the board
theorem force_on_board :
  board_mass M →
  block_mass m →
  gravity g →
  acceleration a →
  friction_coefficient μ →
  T = 4 →
  F - μ * g * (M + 2 * m) - T = M * a →
  F = 24 :=
by
  sorry

end tension_in_string_force_on_board_l320_320653


namespace problem_part_1_problem_part_2_l320_320391

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
noncomputable def vector_b : ℝ × ℝ := (3, -Real.sqrt 3)
noncomputable def f (x : ℝ) : ℝ := (vector_a x).1 * vector_b.1 + (vector_a x).2 * vector_b.2

theorem problem_part_1 (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi) : 
  (vector_a x).1 * vector_b.2 = (vector_a x).2 * vector_b.1 → 
  x = 5 * Real.pi / 6 :=
by
  sorry

theorem problem_part_2 (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi) :
  (∀ t, 0 ≤ t ∧ t ≤ Real.pi → f x ≤ f t) → x = 0 ∧ f 0 = 3 ∧ 
  (∀ t, 0 ≤ t ∧ t ≤ Real.pi → f x ≥ f t) → x = 5 * Real.pi / 6 ∧ f (5 * Real.pi / 6) = -2 * Real.sqrt 3 :=
by
  sorry

end problem_part_1_problem_part_2_l320_320391


namespace find_numbers_l320_320507

theorem find_numbers (a b : ℕ) (h1 : a > 11) (h2 : b > 11) (h3 : a ≠ b)
  (h4 : (∃ S, S = a + b) ∧ (∀ (x y : ℕ), x ≠ y → x + y = a + b → (x > 11) → (y > 11) → ¬(x = a ∨ y = a) → ¬(x = b ∨ y = b)))
  (h5 : even a ∨ even b) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) :=
by
  sorry

end find_numbers_l320_320507


namespace percent_of_carnations_l320_320196

variable (totalFlowers : ℚ)
variable (pinkFlowers : ℚ := 3/5 * totalFlowers)
variable (redFlowers : ℚ := 2/5 * totalFlowers)
variable (pinkRoses : ℚ := 1/3 * pinkFlowers)
variable (pinkCarnations : ℚ := 2/3 * pinkFlowers)
variable (redCarnations : ℚ := 3/4 * redFlowers)

theorem percent_of_carnations
  (h1 : pinkFlowers = 3/5 * totalFlowers)
  (h2 : redFlowers = 2/5 * totalFlowers)
  (h3 : pinkRoses = 1/3 * pinkFlowers)
  (h4 : pinkCarnations = 2/3 * pinkFlowers)
  (h5 : redCarnations = 3/4 * redFlowers) :
  (pinkCarnations + redCarnations) / totalFlowers * 100 = 70 := by
  sorry

end percent_of_carnations_l320_320196


namespace Cheerful_snakes_not_Green_l320_320131

variables {Snake : Type} (snakes : Finset Snake)
variable (Cheerful Green CanSing CanMultiply : Snake → Prop)

-- Conditions
axiom Cheerful_impl_CanSing : ∀ s, Cheerful s → CanSing s
axiom Green_impl_not_CanMultiply : ∀ s, Green s → ¬ CanMultiply s
axiom not_CanMultiply_impl_not_CanSing : ∀ s, ¬ CanMultiply s → ¬ CanSing s

-- Question
theorem Cheerful_snakes_not_Green : ∀ s, Cheerful s → ¬ Green s :=
by sorry

end Cheerful_snakes_not_Green_l320_320131


namespace masha_numbers_unique_l320_320477

def natural_numbers : Set ℕ := {n | n > 11}

theorem masha_numbers_unique (a b : ℕ) (ha : a ∈ natural_numbers) (hb : b ∈ natural_numbers) (hne : a ≠ b)
  (hs_equals : ∃ S, S = a + b)
  (sasha_initially_uncertain : ∀ (a b : ℕ), a ∈ natural_numbers → b ∈ natural_numbers → a ≠ b → (a + b = S) → ¬ (Sasha_can_determine_initially a b S))
  (masha_hint : ∃ (a_even : ℕ), a_even ∈ natural_numbers ∧ (a_even % 2 = 0) ∧ (a_even = a ∨ a_even = b))
  (sasha_then_confident : ∀ (a b : ℕ), a ∈ natural_numbers → b ∈ natural_numbers → a ≠ b → (a + b = S) → (a_even = a ∨ a_even = b) → Sasha_can_determine_confidently a b S) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) := by
  sorry

end masha_numbers_unique_l320_320477


namespace f_one_l320_320037

def g'_k (k x : ℤ) : ℤ := (3 * x^2) / (10^(k + 1))

def smallest_perfect_square_with_ndigits (n : ℤ) : ℤ := 
  let m := 10^(n - 1) 
  m * m

def f (k : ℤ) : ℤ := 
  let start := smallest_perfect_square_with_ndigits (k + 2)
  let squares := List.range (10 ^ (k + 2)) |>.map (λ x, g'_k k (start + x))
  List.foldl min 0 (List.drop 1 squares) - 1

theorem f_one : f 1 = 301 := by
  sorry

end f_one_l320_320037


namespace least_positive_integer_x_l320_320748

theorem least_positive_integer_x (x : ℕ) (h : x + 5683 ≡ 420 [MOD 17]) : x = 7 :=
sorry

end least_positive_integer_x_l320_320748


namespace sum_of_divisor_and_quotient_is_correct_l320_320154

theorem sum_of_divisor_and_quotient_is_correct (divisor quotient : ℕ)
  (h1 : 1000 ≤ divisor ∧ divisor < 10000) -- Divisor is a four-digit number.
  (h2 : quotient * divisor + remainder = original_number) -- Division condition (could be more specific)
  (h3 : remainder < divisor) -- Remainder condition
  (h4 : original_number = 82502) -- Given original number
  : divisor + quotient = 723 := 
sorry

end sum_of_divisor_and_quotient_is_correct_l320_320154


namespace initial_distance_proof_l320_320440

-- Define the speeds of Jack, Christina, and Lindy
def v_J : ℝ := 4
def v_C : ℝ := 5
def v_L : ℝ := 8

-- Define the total distance Lindy has traveled
def d_L : ℝ := 240

-- Define the initial distance between Jack and Christina
def initial_distance : ℝ := 270

-- Prove that the initial distance between Jack and Christina is 270 feet
theorem initial_distance_proof :
  let combined_speed := v_J + v_C
  let time := d_L / v_L
  let D := time * combined_speed
  D = initial_distance :=
by
  -- We define the combined speed, the time, and calculate the distance D
  let combined_speed := v_J + v_C
  let time := d_L / v_L
  let D := time * combined_speed

  -- Since combined_speed = 4 + 5 = 9, time = 240 / 8 = 30, and D = time * combined_speed = 30 * 9 = 270
  show D = initial_distance from by
    simp [combined_speed, time, initial_distance, D]
    sorry

end initial_distance_proof_l320_320440


namespace num_odd_divisors_lt_100_l320_320397

theorem num_odd_divisors_lt_100 : (∃! n : ℕ, n = 9 ∧ ∀ m : ℕ, m < 100 → (odd_divisors m ↔ m = 1^2 ∨ m = 2^2 ∨ m = 3^2 ∨ m = 4^2 ∨ m = 5^2 ∨ m = 6^2 ∨ m = 7^2 ∨ m = 8^2 ∨ m = 9^2)) := sorry

-- Definitions for clarity
def odd_divisors (m : ℕ) : Prop := (finset.range (m + 1)).filter (λ d, m % d = 0).card % 2 = 1

end num_odd_divisors_lt_100_l320_320397


namespace hexagon_coloring_l320_320298

noncomputable def num_colorings_hexagon : Nat :=
  7 * 6 * 5 * 4 * 3 * 2

theorem hexagon_coloring :
  let A B C D E F : Nat := 0
  ∃ (colors : Fin 7 → Nat),
    (∀ i j, i ≠ j → colors i ≠ colors j) ∧
    num_colorings_hexagon = 5040 :=
by
  sorry

end hexagon_coloring_l320_320298


namespace hyperbola_eccentricity_range_l320_320463

-- Define the hyperbola and conditions
def hyperbola_eq (x y : ℝ) (m : ℝ) : Prop :=
  x^2 / (m + 1) - y^2 / (3 - m) = 1

def is_focus_left (x y : ℝ) (focus : ℝ) : Prop :=
  x = -focus ∧ y = 0

def slope_ge_sqrt3 (k : ℝ) : Prop :=
  k ≥ Real.sqrt 3

def is_midpoint (x1 y1 x2 y2 mx my : ℝ) : Prop :=
  mx = (x1 + x2) / 2 ∧ my = (y1 + y2) / 2

def orthogonal (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 * x2 + y1 * y2 = 0

-- Main theorem stating the range of eccentricity for the given hyperbola
theorem hyperbola_eccentricity_range (m c k x y a b p q : ℝ) (h_hyperbola : hyperbola_eq x y m)
    (h_focus : is_focus_left x y c) (h_slope : slope_ge_sqrt3 k) 
    (h_intersects_A : hyperbola_eq a b m) (h_intersects_B : hyperbola_eq p q m)
    (h_midpoint_P : is_midpoint a b x y p q) (h_midpoint_Q : is_midpoint p q x y p q)
    (h_orthogonal : orthogonal x y p q) :
    ∃ e : Set.Set ℝ, e = Set.Ici (Real.sqrt 3 + 1) :=
by
  sorry

end hyperbola_eccentricity_range_l320_320463


namespace Michelle_silver_beads_count_l320_320924

theorem Michelle_silver_beads_count 
  (total_beads : ℕ) (blue_beads : ℕ) (red_beads : ℕ) (white_beads : ℕ) (silver_beads : ℕ)
  (h1 : total_beads = 40)
  (h2 : blue_beads = 5)
  (h3 : red_beads = 2 * blue_beads)
  (h4 : white_beads = blue_beads + red_beads)
  (h5 : silver_beads = total_beads - (blue_beads + red_beads + white_beads)) :
  silver_beads = 10 :=
by
  sorry

end Michelle_silver_beads_count_l320_320924


namespace recalculated_average_ratio_l320_320867

theorem recalculated_average_ratio (x : Fin 50 → ℝ) :
  let A := (∑ i, x i) / 50 in
  let A' := ((∑ i, x i) + 2 * A) / 51 in
  A' / A = 2 / 51 := 
sorry

end recalculated_average_ratio_l320_320867


namespace smallest_n_satisfying_sqrt_inequality_l320_320150

theorem smallest_n_satisfying_sqrt_inequality :
  ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 ∧ (sqrt m - sqrt (m-1 : ℕ) < 0.02) → n ≤ m) ∧ 
  sqrt 626 - sqrt 625 < 0.02 := sorry

end smallest_n_satisfying_sqrt_inequality_l320_320150


namespace max_value_of_even_terms_sum_l320_320916

noncomputable def max_sum_even_a_seq (a : ℕ → ℕ) : ℕ :=
  ∑ i in Finset.range 1011, a (2 * (i + 1))

theorem max_value_of_even_terms_sum :
  ∃ a : ℕ → ℕ,
    (∀ k : ℕ, 0 < k → a (2 * k - 1) + a (2 * k) = 2^k) ∧
    (∀ m : ℕ, 0 < m → ∃ l : ℕ, l ≤ m ∧ a (m + 1) = ∑ i in Finset.range (m + 1), if l ≤ i then a i else 0) ∧
    max_sum_even_a_seq a = (2^1013 - 5) / 3 :=
begin
  sorry
end

end max_value_of_even_terms_sum_l320_320916


namespace greatest_difference_is_124_l320_320161

-- Define the variables a, b, c, and x
variables (a b c x : ℕ)

-- Define the conditions of the problem
def conditions (a b c : ℕ) := 
  (4 * a = 2 * b) ∧ 
  (4 * a = c) ∧ 
  (a > 0) ∧ 
  (a < 10) ∧ 
  (b < 10) ∧ 
  (c < 10)

-- Define the value of a number given its digits
def number (a b c : ℕ) := 100 * a + 10 * b + c

-- Define the maximum and minimum values of x
def max_val (a : ℕ) := number a (2 * a) (4 * a)
def min_val (a : ℕ) := number a (2 * a) (4 * a)

-- Define the greatest difference
def greatest_difference := max_val 2 - min_val 1

-- Prove that the greatest difference is 124
theorem greatest_difference_is_124 : greatest_difference = 124 :=
by 
  unfold greatest_difference 
  unfold max_val 
  unfold min_val 
  unfold number 
  sorry

end greatest_difference_is_124_l320_320161


namespace max_degree_l320_320448

noncomputable def isPolynomial (f : ℤ × ℤ → ℝ) : Prop :=
  ∀ C : ℤ, 
    ∃ P1 P2 P3 P4 : ℤ → ℝ, 
    (degree P1 ≤ 100 ∧ degree P2 ≤ 100 ∧ 
     degree P3 ≤ 100 ∧ degree P4 ≤ 100) ∧ 
    (∀ x : ℤ, f (x, C) = P1 x) ∧
    (∀ x : ℤ, f (C, x) = P2 x) ∧
    (∀ x : ℤ, f (x, x + C) = P3 x) ∧
    (∀ x : ℤ, f (x, C - x) = P4 x)

theorem max_degree (f : ℤ × ℤ → ℝ) (h : isPolynomial f) :
  ∃ P : (ℤ × ℤ) → ℝ, is_polynomial_bivar P ∧ degree_bivar P = 133 :=
sorry

end max_degree_l320_320448


namespace perimeter_of_original_square_l320_320211

-- Definitions
variables {x : ℝ}
def rect_width := x
def rect_length := 4 * x
def rect_perimeter := 56
def original_square_perimeter := 32

-- Statement
theorem perimeter_of_original_square (x : ℝ) (h : 28 * x = 56) : 4 * (4 * x) = 32 :=
by
  -- Since the proof is not required, we apply sorry to end the theorem.
  sorry

end perimeter_of_original_square_l320_320211


namespace find_integer_mod_l320_320309

theorem find_integer_mod (n : ℕ) (h1 : 0 ≤ n) (h2 : n ≤ 9) : n ≡ -1212 [MOD 10] → n = 8 := 
by 
  sorry

end find_integer_mod_l320_320309


namespace rectangle_diagonal_length_proof_parallel_l320_320171

-- Definition of a rectangle whose sides are parallel to the coordinate axes
structure RectangleParallel :=
  (a b : ℕ)
  (area_eq : a * b = 2018)
  (diagonal_length : ℕ)

-- Prove that the length of the diagonal of the given rectangle is sqrt(1018085)
def rectangle_diagonal_length_parallel : RectangleParallel → Prop :=
  fun r => r.diagonal_length = Int.sqrt (r.a * r.a + r.b * r.b)

theorem rectangle_diagonal_length_proof_parallel (r : RectangleParallel)
  (h1 : r.a * r.b = 2018)
  (h2 : r.a ≠ r.b)
  (h3 : r.diagonal_length = Int.sqrt (r.a * r.a + r.b * r.b)) :
  r.diagonal_length = Int.sqrt 1018085 := 
  sorry

end rectangle_diagonal_length_proof_parallel_l320_320171


namespace first_discount_is_10_percentage_l320_320693

-- Declare the conditions given in the problem statement
def normal_price : ℝ := 174.99999999999997
def first_discount_percentage : ℝ := 10
def second_discount_percentage : ℝ := 20
def first_discount (price : ℝ) : ℝ := price * (first_discount_percentage / 100)
def second_discount (price : ℝ) : ℝ := price * (second_discount_percentage / 100)

-- State the theorem to prove that the first discount percentage is indeed 10%
theorem first_discount_is_10_percentage :
  first_discount_percentage = 10 := 
by
  sorry

end first_discount_is_10_percentage_l320_320693


namespace lines_parallel_l320_320142

theorem lines_parallel
  (circle1 circle2 : Circle)
  (A B : Point)
  (h_intersect : A ∈ circle1 ∧ B ∈ circle1 ∧ A ∈ circle2 ∧ B ∈ circle2)
  (C D : Point)
  (h_AC : LineSegment A C)
  (h_BD : LineSegment B D)
  (E F : Point)
  (h_extension : extends_to C E circle2 ∧ extends_to D F circle2) :
  parallel (line_through C D) (line_through E F) :=
sorry

end lines_parallel_l320_320142


namespace distance_light_travels_500_years_l320_320086

-- Define the given conditions
def distance_in_one_year_miles : ℝ := 5.87e12
def years_traveling : ℝ := 500
def miles_to_kilometers : ℝ := 1.60934

-- Define the expected distance in kilometers after 500 years
def expected_distance_in_kilometers : ℝ  := 4.723e15

-- State the theorem: the distance light travels in 500 years in kilometers
theorem distance_light_travels_500_years :
  (distance_in_one_year_miles * years_traveling * miles_to_kilometers) 
    = expected_distance_in_kilometers := 
by
  sorry

end distance_light_travels_500_years_l320_320086


namespace no_seating_in_four_consecutive_seats_l320_320002

theorem no_seating_in_four_consecutive_seats :
  let total_arrangements := Nat.factorial 10
  let grouped_arrangements := Nat.factorial 7 * Nat.factorial 4
  let acceptable_arrangements := total_arrangements - grouped_arrangements
  acceptable_arrangements = 3507840 :=
by
  sorry

end no_seating_in_four_consecutive_seats_l320_320002


namespace largest_common_term_l320_320083

/-- The arithmetic progression sequence1 --/
def sequence1 (n : ℕ) : ℤ := 4 + 5 * n

/-- The arithmetic progression sequence2 --/
def sequence2 (n : ℕ) : ℤ := 5 + 8 * n

/-- The common term condition for sequence1 --/
def common_term_condition1 (a : ℤ) : Prop := ∃ n : ℕ, a = sequence1 n

/-- The common term condition for sequence2 --/
def common_term_condition2 (a : ℤ) : Prop := ∃ n : ℕ, a = sequence2 n

/-- The largest common term less than 1000 --/
def is_largest_common_term (a : ℤ) : Prop :=
  common_term_condition1 a ∧ common_term_condition2 a ∧ a < 1000 ∧
  ∀ b : ℤ, common_term_condition1 b ∧ common_term_condition2 b ∧ b < 1000 → b ≤ a

/-- Lean theorem statement --/
theorem largest_common_term :
  ∃ a : ℤ, is_largest_common_term a ∧ a = 989 :=
sorry

end largest_common_term_l320_320083


namespace numbers_masha_thought_l320_320526

noncomputable def distinct_natural_numbers_greater_than_eleven_and_sum_to_satisfy_conditions : ℕ → ℕ → Prop :=
λ a b, a ≠ b ∧ a > 11 ∧ b > 11 ∧ (a + b = 28) ∧ (¬ (∃ x y, x ≠ y ∧ x > 11 ∧ y > 11 ∧ x + y = a + b ∧ (x ≠ a ∧ y ≠ b)))

theorem numbers_masha_thought (a b : ℕ) (h : distinct_natural_numbers_greater_than_eleven_and_sum_to_satisfy_conditions a b) : 
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) :=
by sorry

end numbers_masha_thought_l320_320526


namespace product_mod5_is_zero_l320_320275

theorem product_mod5_is_zero :
  (2023 * 2024 * 2025 * 2026) % 5 = 0 :=
by
  sorry

end product_mod5_is_zero_l320_320275


namespace simplify_expression_l320_320978

theorem simplify_expression :
  18 * (14 / 15) * (1 / 12) - (1 / 5) = 1 / 2 :=
by
  sorry

end simplify_expression_l320_320978


namespace instantaneous_velocity_at_t1_l320_320402

theorem instantaneous_velocity_at_t1 :
  (∀ t : ℝ, deriv (λ t, 2 * t^2) t = 4 * t) →
  deriv (λ t, 2 * t^2) 1 = 4 :=
begin
  intros h_deriv,
  exact h_deriv 1,
end

end instantaneous_velocity_at_t1_l320_320402


namespace min_value_of_m_plus_3n_l320_320795

theorem min_value_of_m_plus_3n (m n : ℝ) (h1 : log m n = -1) (h2 : m > 0) (h3 : n > 0) (h4 : m ≠ 1) : 
    m + 3 * n = 2 * Real.sqrt 3 :=
sorry

end min_value_of_m_plus_3n_l320_320795


namespace probability_all_same_color_l320_320651

theorem probability_all_same_color :
  let total_marbles := 20
  let red_marbles := 5
  let white_marbles := 7
  let blue_marbles := 8
  let total_ways_to_draw_3 := (total_marbles * (total_marbles - 1) * (total_marbles - 2)) / 6
  let ways_to_draw_3_red := (red_marbles * (red_marbles - 1) * (red_marbles - 2)) / 6
  let ways_to_draw_3_white := (white_marbles * (white_marbles - 1) * (white_marbles - 2)) / 6
  let ways_to_draw_3_blue := (blue_marbles * (blue_marbles - 1) * (blue_marbles - 2)) / 6
  let probability := (ways_to_draw_3_red + ways_to_draw_3_white + ways_to_draw_3_blue) / total_ways_to_draw_3
  probability = 101/1140 :=
by
  sorry

end probability_all_same_color_l320_320651


namespace imaginary_part_div_z1_z2_l320_320766

noncomputable def z1 := 1 - 3 * Complex.I
noncomputable def z2 := 3 + Complex.I

theorem imaginary_part_div_z1_z2 : 
  Complex.im ((1 + 3 * Complex.I) / (3 + Complex.I)) = 4 / 5 := 
by 
  sorry

end imaginary_part_div_z1_z2_l320_320766


namespace closest_point_on_graph_l320_320948

theorem closest_point_on_graph (x y : ℝ) (h1 : x > 0) (h2 : y = x + 1/x) :
  (x = 1/real.root 4 2) ∧ (y = (1 + real.sqrt 2) / real.root 4 2) :=
sorry

end closest_point_on_graph_l320_320948


namespace max_ab_extremum_at_one_l320_320846

theorem max_ab_extremum_at_one (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
(f : ℝ → ℝ) (h_f : f = λ x, 4 * x^3 - a * x^2 - 2 * b * x + 2) 
(h_ext : ∃ x₀, x₀ = 1 ∧ has_deriv_at f (0 : ℝ) x₀) :
  ab ≤ 9 :=
by
  have h_deriv : deriv f = λ x, 12 * x^2 - 2 * a * x - 2 * b,
  from funext (λ x, by simp [h_f, deriv]); 
  have h_ext_one : deriv f (1 : ℝ) = 0,
  from (h_ext.2) (h_ext.1.symm) ;
  have : 12 - 2 * a - 2 * b = 0,  
  by rw [h_deriv, pi.sub_apply, h_ext_one];
  have h_sum : a + b = 6, 
  from eq_of_sub_eq_zero this;
  have h_am_gm : a + b ≥ 2 * (sqrt (a * b)),
  from (add_le_add_iff_right (- a)).mpr (am_gm a b);
  have : sqrt (a * b) ≤ 3,
  by simp [h_sum, sqrt_le_sqrt_iff, h_am_gm];
  have : a * b ≤ 9, 
  by field_simp [mul_le_mul_iff_nonneg_right, mul_sqrt_lt_sqrt_mul h1 h2, h1, mul_eq_zero];
  exact this
end

end max_ab_extremum_at_one_l320_320846


namespace power_sum_l320_320295

theorem power_sum :
  (-3)^3 + (-3)^2 + (-3) + 3 + 3^2 + 3^3 = 18 :=
by
  sorry

end power_sum_l320_320295


namespace masha_numbers_l320_320523

theorem masha_numbers (a b : ℕ) (S : ℕ) (h1 : a ≠ b) (h2 : a > 11) (h3 : b > 11) (h4 : S = a + b) 
    (h5 : (∀ x y : ℕ, x + y = S → x = a ∨ y = a → abs x - y = a) ∧ (even a ∨ even b)) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) := 
by sorry

end masha_numbers_l320_320523


namespace largest_sum_of_products_l320_320594

theorem largest_sum_of_products (a b c d : ℕ) (ha : a = 2) (hb : b = 3) (hc : c = 5) (hd : d = 6) :
  a * b + b * c + c * d + d * a = 63 :=
by 
  rw [ha, hb, hc, hd]
  -- Perform the necessary calculations
  sorry

end largest_sum_of_products_l320_320594


namespace num_even_divisors_of_8_l320_320828

def factorial (n : Nat) : Nat :=
  match n with
  | 0     => 1
  | Nat.succ n' => Nat.succ n' * factorial n'

-- Define the prime factorization of 8!
def prime_factors_eight_factorial : Nat := 2^7 * 3^2 * 5 * 7

-- Definition of an even divisor of 8!
def is_even_divisor (d : Nat) : Prop :=
  d ∣ prime_factors_eight_factorial ∧ 2 ∣ d

-- Calculation of number of even divisors of 8!
def num_even_divisors_8! : Nat :=
  7 * 3 * 2 * 2

theorem num_even_divisors_of_8! :
  num_even_divisors_8! = 84 :=
sorry

end num_even_divisors_of_8_l320_320828


namespace sum_card_eq_min_n_l320_320173

variable {A : ℕ → Set ℕ}
variable {n : ℕ}

noncomputable def satisfies_conditions (A : ℕ → Set ℕ) (n : ℕ) : Prop :=
  (∀ i, i < n → (i ∉ A i) ∧ (3 ≤ card (A i))) ∧
  (∀ i j, i ≠ j → i < n → j < n → (i ∈ A j ↔ j ∉ A i))

theorem sum_card_eq (A : ℕ → Set ℕ) (n : ℕ) (h : satisfies_conditions A n) : 
  ∑ i in finset.range n, card (A i) = n * (n - 1) / 2 :=
sorry

theorem min_n (n : ℕ) : 
  (∀ A, satisfies_conditions A n → ∑ i in finset.range n, card (A i) ≥ 3 * n) ↔ n ≥ 7 :=
sorry

end sum_card_eq_min_n_l320_320173


namespace exists_prime_not_in_list_l320_320544

open Nat

theorem exists_prime_not_in_list (l : List ℕ) (h : ∀ p ∈ l, Prime p) : 
  ∃ q, Prime q ∧ q ∉ l := 
sorry

end exists_prime_not_in_list_l320_320544


namespace odd_divisors_lt_100_l320_320395

theorem odd_divisors_lt_100 : 
  let n := 100
  let perfect_squares := { k : ℕ // k < n ∧ ∃ m : ℕ, k = m * m }
  ∃ (count : ℕ), (count = perfect_squares.card) ∧ (count = 9) :=
by
  sorry

end odd_divisors_lt_100_l320_320395


namespace find_a_for_even_function_l320_320365

-- conditions
def f (a : ℝ) (x : ℝ) : ℝ := x^3 * (a * 2^x - 2^(-x))

-- definition of an even function
def is_even_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g x = g (-x)

-- the proof problem statement
theorem find_a_for_even_function (a : ℝ) : is_even_function (f a) ↔ a = 1 :=
by
  sorry

end find_a_for_even_function_l320_320365


namespace find_number_of_students_l320_320198

-- Parameters
variable (n : ℕ) (C : ℕ)
def first_and_last_picked_by_sam (n : ℕ) (C : ℕ) : Prop := 
  C + 1 = 2 * n

-- Conditions: number of candies is 120, the bag completes 2 full rounds at the table.
theorem find_number_of_students
  (C : ℕ) (h_C: C = 120) (h_rounds: 2 * n = C):
  n = 60 :=
by
  sorry

end find_number_of_students_l320_320198


namespace floor_e_minus_3_eq_negative_one_l320_320730

theorem floor_e_minus_3_eq_negative_one 
  (e : ℝ) 
  (h : 2 < e ∧ e < 3) : 
  (⌊e - 3⌋ = -1) :=
by
  sorry

end floor_e_minus_3_eq_negative_one_l320_320730


namespace find_certain_number_l320_320398

theorem find_certain_number (x : ℤ) (h : x - 5 = 4) : x = 9 :=
sorry

end find_certain_number_l320_320398


namespace candy_eaten_l320_320757

/--
Given:
- Faye initially had 47 pieces of candy
- Faye ate x pieces the first night
- Faye's sister gave her 40 more pieces
- Now Faye has 62 pieces of candy

We need to prove:
- Faye ate 25 pieces of candy the first night.
-/
theorem candy_eaten (x : ℕ) (h1 : 47 - x + 40 = 62) : x = 25 :=
by
  sorry

end candy_eaten_l320_320757


namespace alice_outfits_l320_320684

theorem alice_outfits :
  let trousers := 5
  let shirts := 8
  let jackets := 4
  let shoes := 2
  trousers * shirts * jackets * shoes = 320 :=
by
  sorry

end alice_outfits_l320_320684


namespace jackie_phil_probability_l320_320889

noncomputable def probability_same_heads : ℚ :=
  let fair_coin := (1 + 1: ℚ)
  let p3_coin := (2 + 3: ℚ)
  let p2_coin := (1 + 2: ℚ)
  let generating_function := fair_coin * p3_coin * p2_coin
  let sum_of_coefficients := 30
  let sum_of_squares_of_coefficients := 290
  sum_of_squares_of_coefficients / (sum_of_coefficients ^ 2)

theorem jackie_phil_probability : probability_same_heads = 29 / 90 := by
  sorry

end jackie_phil_probability_l320_320889


namespace arithmetic_sequence_general_term_geometric_sequence_sum_l320_320337

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

end arithmetic_sequence_general_term_geometric_sequence_sum_l320_320337


namespace bead_count_l320_320921

variable (total_beads : ℕ) (blue_beads : ℕ) (red_beads : ℕ) (white_beads : ℕ) (silver_beads : ℕ)

theorem bead_count : total_beads = 40 ∧ blue_beads = 5 ∧ red_beads = 2 * blue_beads ∧ white_beads = blue_beads + red_beads ∧ silver_beads = total_beads - (blue_beads + red_beads + white_beads) → silver_beads = 10 :=
by
  intro h
  sorry

end bead_count_l320_320921


namespace roots_of_equation_l320_320968

theorem roots_of_equation (x : ℝ) : 
    x^4 + 4 * x - 1 = 0 ↔ 
    ∃ y, y = (−1 + real.sqrt (2 * real.sqrt 2 - 1)) / real.sqrt 2 ∨ y = (−1 - real.sqrt (2 * real.sqrt 2 - 1)) / real.sqrt 2 :=
by 
    sorry

end roots_of_equation_l320_320968


namespace john_took_11_more_chickens_than_ray_l320_320557

noncomputable def chickens_taken_by_john (mary_chickens : ℕ) : ℕ := mary_chickens + 5
noncomputable def chickens_taken_by_ray (mary_chickens : ℕ) : ℕ := mary_chickens - 6
def ray_chickens : ℕ := 10

-- The theorem to prove:
theorem john_took_11_more_chickens_than_ray :
  ∃ (mary_chickens : ℕ), chickens_taken_by_john mary_chickens - ray_chickens = 11 :=
by
  -- Initial assumptions and derivation steps should be provided here.
  sorry

end john_took_11_more_chickens_than_ray_l320_320557


namespace incorrect_inference_l320_320543

variables {Point : Type} [EuclideanSpace Point] 
variables {a b : Line Point} {α : Plane Point}

-- Hypotheses
def perpendicular_to_plane (l : Line Point) (p : Plane Point) := ∀ P ∈ l, P ∈ p

variables (H₁ : perpendicular_to_plane a α) 
variables (H₂ : perpendicular_to_plane b α)
variables (H : ∀ A B : Point, A ∈ a → B ∈ b → A ≠ B → lies_in_plane A α ∧ lies_in_plane B α)

-- Proof goal
theorem incorrect_inference (H₁ : perpendicular_to_plane a α) 
                           (H₂ : perpendicular_to_plane b α) 
                           (H : ∀ A B : Point, A ∈ a → B ∈ b → lies_in_plane A α ∧ lies_in_plane B α)
                           : ¬ (∀ A B : Point, A ∈ a → B ∈ b → perpendicular_to_plane a (Line_through A B) 
                                                                     → perpendicular_to_plane b (Line_through A B) 
                                                                     → parallel_lines a b) :=
sorry

end incorrect_inference_l320_320543


namespace part1_daily_sales_52_part2_max_profit_l320_320189

variables (purchase_price : ℝ) (base_price : ℝ) (base_sales : ℝ) (price_increase : ℝ) (sales_decrease : ℝ)

-- Conditions
def conditions := 
  purchase_price = 40 ∧ 
  base_price = 50 ∧ 
  base_sales = 200 ∧ 
  price_increase = 1 ∧ 
  sales_decrease = 10

-- Part 1: Prove daily sales quantity at 52 yuan is 180 items
theorem part1_daily_sales_52 (h : conditions) : 
  let selling_price := 52 in 
  base_sales - (selling_price - base_price) * sales_decrease = 180 := 
by
  intros
  cases h with _ h1
  cases h1 with _ h2
  cases h2 with _ h3
  cases h3 with _ h4
  rw [← h2, ← h3, ← h4]
  simp
  sorry

-- Part 2: Prove the price that maximizes profit and the maximum profit
theorem part2_max_profit (h : conditions) : 
  let x := 55 in 
  let y := -10 * (x - 55)^2 + 2250 in
  y = 2250 :=
by
  intros
  cases h with _ h1
  cases h1 with _ h2
  cases h2 with _ h3
  cases h3 with _ h4
  rw [← h2, ← h4]
  simp
  sorry

end part1_daily_sales_52_part2_max_profit_l320_320189


namespace min_moves_to_equalize_boxes_l320_320600

def initialCoins : List ℕ := [5, 8, 11, 17, 20, 15, 10]

def targetCoins (boxes : List ℕ) : ℕ := boxes.sum / boxes.length

def movesRequiredToBalance : List ℕ → ℕ
| [5, 8, 11, 17, 20, 15, 10] => 22
| _ => sorry

theorem min_moves_to_equalize_boxes :
  movesRequiredToBalance initialCoins = 22 :=
by
  sorry

end min_moves_to_equalize_boxes_l320_320600


namespace Michelle_silver_beads_count_l320_320923

theorem Michelle_silver_beads_count 
  (total_beads : ℕ) (blue_beads : ℕ) (red_beads : ℕ) (white_beads : ℕ) (silver_beads : ℕ)
  (h1 : total_beads = 40)
  (h2 : blue_beads = 5)
  (h3 : red_beads = 2 * blue_beads)
  (h4 : white_beads = blue_beads + red_beads)
  (h5 : silver_beads = total_beads - (blue_beads + red_beads + white_beads)) :
  silver_beads = 10 :=
by
  sorry

end Michelle_silver_beads_count_l320_320923


namespace ratio_of_initial_to_doubled_l320_320203

theorem ratio_of_initial_to_doubled (x : ℕ) (h : 3 * (2 * x + 13) = 93) : x : (2 * x) = 1 : 2 := by
  have h1 : 3 * (2 * x + 13) = 93 := h
  have h2 : 6 * x + 39 = 93 := by simp [h1]
  have h3 : 6 * x = 54 := by linarith
  have h4 : x = 54 / 6 := by linarith
  have h5 : x = 9 := by simp [h4]
  have h6 : 2 * x = 2 * 9 := by simp [h5]
  have h7 : 2 * x = 18 := by simp [h6]
  have h8 : x / x = 1 := by sorry
  have h9 : (2 * x) / x = 2 := by sorry
  show x : (2 * x) = 1 : 2 from by simp [h8, h9]

end ratio_of_initial_to_doubled_l320_320203


namespace three_digit_integers_count_l320_320839

theorem three_digit_integers_count (N : ℕ) :
  (∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 
            n % 7 = 4 ∧ 
            n % 8 = 3 ∧ 
            n % 10 = 2) → N = 3 :=
by
  sorry

end three_digit_integers_count_l320_320839


namespace find_D_coordinates_l320_320070

theorem find_D_coordinates :
  ∀ (A B C : ℝ × ℝ),
    A = (-2, 5) →
    C = (3, 8) →
    B = (-3, 0) →
    ∃ D : ℝ × ℝ, D = (2, 3) :=
by
  intros A B C hA hC hB
  use (2, 3)
  intros hD
  sorry

end find_D_coordinates_l320_320070


namespace numbers_masha_thought_l320_320529

noncomputable def distinct_natural_numbers_greater_than_eleven_and_sum_to_satisfy_conditions : ℕ → ℕ → Prop :=
λ a b, a ≠ b ∧ a > 11 ∧ b > 11 ∧ (a + b = 28) ∧ (¬ (∃ x y, x ≠ y ∧ x > 11 ∧ y > 11 ∧ x + y = a + b ∧ (x ≠ a ∧ y ≠ b)))

theorem numbers_masha_thought (a b : ℕ) (h : distinct_natural_numbers_greater_than_eleven_and_sum_to_satisfy_conditions a b) : 
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) :=
by sorry

end numbers_masha_thought_l320_320529


namespace geometric_series_proof_l320_320027

-- Definitions used in conditions
def a (n : ℕ) (a : ℤ) : ℕ → ℤ := λ n, (∑ i in finset.range n, a ^ i)

-- Theorem statement
theorem geometric_series_proof (a : ℤ) (n1 n2 : ℕ) (h1 : n2 > n1) (h2 : ∀ (p : ℕ), p ∣ (n2 - n1) → prime p → p ∣ (a ^ p - 1)) : 
  ∃ k : ℤ, (a n2 a - a n1 a) / (n2 - n1) = k := 
sorry

end geometric_series_proof_l320_320027


namespace sum_of_roots_l320_320763

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi / 4)

theorem sum_of_roots (m : ℝ) (x1 x2 : ℝ) (h1 : 0 ≤ x1) (h2 : x1 < 2 * Real.pi)
  (h3 : 0 ≤ x2) (h4 : x2 < 2 * Real.pi) (h_distinct : x1 ≠ x2)
  (h_eq1 : f x1 = m) (h_eq2 : f x2 = m) : x1 + x2 = Real.pi / 2 ∨ x1 + x2 = 5 * Real.pi / 2 :=
by
  sorry

end sum_of_roots_l320_320763


namespace probability_both_genders_among_selected_l320_320191

open Finset

theorem probability_both_genders_among_selected :
  (choose 7 3).toRat ≠ 0 →
  (5.choose 3).toRat / (7.choose 3).toRat + (2.choose 3).toRat / (7.choose 3).toRat ≠ 1 →
  ∀ (n : ℕ), n = (7.choose 3).toRat →
  ((n - (5.choose 3).toRat - (2.choose 3).toRat) / n = (3 / 5 : ℝ)) :=
by
  sorry

end probability_both_genders_among_selected_l320_320191


namespace determine_numbers_l320_320469

theorem determine_numbers (a b : ℕ) (S : ℕ) (h1 : a ≠ b) (h2 : a > 11) (h3 : b > 11)
  (h4 : S = a + b) (h5 : (∀ (x y : ℕ), x + y = S → x ≠ y → (x = a ∧ y = b) ∨ (x = b ∧ y = a)) = false)
  (h6 : a % 2 = 0 ∨ b % 2 = 0) : 
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) := 
  sorry

end determine_numbers_l320_320469


namespace n_not_six_n_is_seven_l320_320595

open Nat

-- Define the conditions for the problem
def nails_and_strings (n : ℕ) : Prop :=
  n > 1 ∧ ∑ i in finset.range (n - 1), (i + 1) = (n - 1) * n / 2

-- The property that states for every triplet of colors, there is a triangle with exactly those three colors
def triplet_color_triangle_property (n : ℕ) : Prop :=
  ∀ (colors : finset (finset ℕ)), colors.card = 3 →
  ∃ triangle : finset (finset ℕ), triangle.card = 3 ∧
  triangle ⊆ (finset.powerset (finset.range n)).filter (λ t, t.card = 2) ∧
  ∀ t ∈ triangle, t.values ∈ colors

-- The proof statements for n = 6 and n = 7
theorem n_not_six (n : ℕ) (h : n = 6) :
  ¬ nails_and_strings n ∧ triplet_color_triangle_property n :=
sorry

theorem n_is_seven (n : ℕ) (h : n = 7) :
  nails_and_strings n ∧ triplet_color_triangle_property n :=
sorry

end n_not_six_n_is_seven_l320_320595


namespace projection_vectors_l320_320625

-- Define the given vectors
def vec1 : ℝ × ℝ := (-3, 2)
def vec2 : ℝ × ℝ := (4, -1)
def vec_p : ℝ × ℝ := (5 / 8, 7 / 8)

-- Define the projection condition
def projection_condition (v : ℝ × ℝ) (p : ℝ × ℝ) : Prop :=
  let ⟨vx, vy⟩ := v
  let ⟨px, py⟩ := p
  (vx * px + vy * py = 0)

-- Problem statement: Prove that there exists a vector v such that both vec1 and vec2, 
-- when projected onto v, result in the same vector vec_p.
theorem projection_vectors :
  ∃ v : ℝ × ℝ, projection_condition vec1 vec_p ∧ projection_condition vec2 vec_p := 
sorry

end projection_vectors_l320_320625


namespace quotient_base_6_l320_320736

noncomputable def base_6_to_base_10 (n : ℕ) : ℕ := 
  match n with
  | 2314 => 2 * 6^3 + 3 * 6^2 + 1 * 6^1 + 4
  | 14 => 1 * 6^1 + 4
  | _ => 0

noncomputable def base_10_to_base_6 (n : ℕ) : ℕ := 
  match n with
  | 55 => 1 * 6^2 + 3 * 6^1 + 5
  | _ => 0

theorem quotient_base_6 :
  base_10_to_base_6 ((base_6_to_base_10 2314) / (base_6_to_base_10 14)) = 135 :=
by
  sorry

end quotient_base_6_l320_320736


namespace closest_point_to_origin_on_graph_l320_320935

theorem closest_point_to_origin_on_graph :
  ∃ x : ℝ, x > 0 ∧ (y = x + 1/x ∧ (x, y) = (1/real.root 4 2, (1 + real.sqrt 2)/real.root 4 2)) := sorry

end closest_point_to_origin_on_graph_l320_320935


namespace line_through_p_with_equal_intercepts_l320_320996

/-- Prove that the equations of the line passing through the point P(1,2) and having equal intercepts on both coordinate axes are x + y - 3 = 0 or 2x - y = 0. -/
theorem line_through_p_with_equal_intercepts (P : ℝ × ℝ) (line1 line2 : ℝ → ℝ → Prop) :
  P = (1, 2) →
  (∀ x y : ℝ, line1 x y ↔ x + y - 3 = 0) →
  (∀ x y : ℝ, line2 x y ↔ 2x - y = 0) →
  (∃ line : ℝ → ℝ → Prop, (∀ x y : ℝ, line x y ↔ line1 x y) ∨ (∀ x y : ℝ, line x y ↔ line2 x y)) :=
by { sorry }

end line_through_p_with_equal_intercepts_l320_320996


namespace max_sum_of_squares_eq_l320_320619

theorem max_sum_of_squares_eq (x y : ℝ) (h1 : x^2 + y^2 = 100) (h2 : x * y = 40) : 
  x + y ≤ 6 * Real.sqrt 5 := 
by
  sorry

end max_sum_of_squares_eq_l320_320619


namespace decreasing_power_function_l320_320855

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x ^ k

theorem decreasing_power_function (k : ℝ) : 
  (∀ x : ℝ, 0 < x → (f k x) ≤ 0) ↔ k < 0 ∧ k ≠ 0 := sorry

end decreasing_power_function_l320_320855


namespace determine_a_l320_320359

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def f (a : ℝ) (x : ℝ) : ℝ :=
  x^3 * (a * 2^x - 2^(-x))

theorem determine_a : ∃ a : ℝ, is_even_function (f a) ∧ a = 1 :=
by
  use 1
  sorry

end determine_a_l320_320359


namespace volume_of_tetrahedron_l320_320319

theorem volume_of_tetrahedron (r γ : ℝ) : 
  ∃ V: ℝ, V = (2 * r^3 * real.sqrt 3 * (real.cos (γ / 2))^2) / 
   ((real.sqrt 3 - real.sqrt (1 - real.cos γ))^3 * real.sqrt (1 - real.cos γ)) :=
sorry

end volume_of_tetrahedron_l320_320319


namespace rhombus_area_l320_320084

theorem rhombus_area (d1 d2 : ℕ) (h1 : d1 = 12) (h2 : d2 = 15) : 
  (d1 * d2) / 2 = 90 := 
by 
  rw [h1, h2]
  norm_num

end rhombus_area_l320_320084


namespace intersection_A_B_eq_l320_320810

def A : set ℝ := {x | x^2 - 5 * x + 6 <= 0}
def B : set ℝ := {x | abs (2 * x - 1) > 3}

theorem intersection_A_B_eq :
  (A ∩ B) = {x | 2 < x ∧ x <= 3} :=
by sorry

end intersection_A_B_eq_l320_320810


namespace four_digit_number_with_divisors_l320_320588

def is_four_digit (n : Nat) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_minimal_divisor (n p : Nat) : Prop :=
  p > 1 ∧ n % p = 0
  
def is_maximal_divisor (n q : Nat) : Prop :=
  q < n ∧ n % q = 0
  
theorem four_digit_number_with_divisors :
  ∃ (n p : Nat), is_four_digit n ∧ is_minimal_divisor n p ∧ n = 49 * p * p :=
by
  sorry

end four_digit_number_with_divisors_l320_320588


namespace ratio_H_R_l320_320332

noncomputable def minimize_surface_area (V : ℝ) (r h : ℝ) : Prop := 
  π * r^2 * h = 2 * V ∧ 2 * π * r^2 + 2 * π * r * h ≤ (2 * π * (V / (π * r))^(2 / 3))

theorem ratio_H_R (V : ℝ) (r h : ℝ) (H R : ℝ) 
  (hv : π * r^2 * h = 2 * V) 
  (hm : minimize_surface_area V r h)
  (hR : r = R) (hH : h = H): 
  H / R = 2 :=
begin
  sorry -- Proof is not required per the instructions
end

end ratio_H_R_l320_320332


namespace eccentricity_of_ellipse_l320_320789

theorem eccentricity_of_ellipse (a b : ℝ) (h_ab : a > b) (h_b : b > 0) :
  (∀ x y : ℝ, (y = -2 * x + 1 → ∃ x₁ y₁ x₂ y₂ : ℝ, (y₁ = -2 * x₁ + 1 ∧ y₂ = -2 * x₂ + 1) ∧ 
    (x₁ / a * x₁ / a + y₁ / b * y₁ / b = 1) ∧ (x₂ / a * x₂ / a + y₂ / b * y₂ / b = 1) ∧ 
    ((x₁ + x₂) / 2 = 4 * (y₁ + y₂) / 2)) → (x / a)^2 + (y / b)^2 = 1) →
  ∃ e : ℝ, e = Real.sqrt (1 - (b / a) ^ 2) ∧ e = (Real.sqrt 2) / 2 :=
sorry

end eccentricity_of_ellipse_l320_320789


namespace determine_numbers_l320_320474

theorem determine_numbers (a b : ℕ) (S : ℕ) (h1 : a ≠ b) (h2 : a > 11) (h3 : b > 11)
  (h4 : S = a + b) (h5 : (∀ (x y : ℕ), x + y = S → x ≠ y → (x = a ∧ y = b) ∨ (x = b ∧ y = a)) = false)
  (h6 : a % 2 = 0 ∨ b % 2 = 0) : 
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) := 
  sorry

end determine_numbers_l320_320474


namespace problem1_problem2_l320_320231

-- Problem 1
theorem problem1 (x : ℝ) : (1 : ℝ) * (-2 * x^2)^3 + x^2 * x^4 - (-3 * x^3)^2 = -16 * x^6 := 
sorry

-- Problem 2
theorem problem2 (a b : ℝ) : (a - b)^2 * (b - a)^4 + (b - a)^3 * (a - b)^3 = 0 := 
sorry

end problem1_problem2_l320_320231


namespace bugs_meet_at_point_p_l320_320613

noncomputable def radius_large := 6
noncomputable def radius_small := 3
noncomputable def speed_large := 4 * Real.pi
noncomputable def speed_small := 3 * Real.pi

-- Circumference calculations based on given radii
noncomputable def circumference_large := 2 * Real.pi * radius_large
noncomputable def circumference_small := 2 * Real.pi * radius_small

-- Time to complete one journey around the circles
noncomputable def time_large := circumference_large / speed_large
noncomputable def time_small := circumference_small / speed_small

-- Proof that the bugs will next meet at point P in 6 minutes
theorem bugs_meet_at_point_p :
  ∃ t : ℕ, t = Nat.lcm (time_large.natAbs) (time_small.natAbs) ∧ t = 6 :=
by
  sorry

end bugs_meet_at_point_p_l320_320613


namespace trajectory_and_line_properties_l320_320787

def circle (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4
def line (λ x y : ℝ) : Prop := (1 + λ) * x + y - sqrt 3 * (1 + λ) - 1 = 0

theorem trajectory_and_line_properties :
  (∀ (λ : ℝ), ∃ x y : ℝ, line λ x y ∧ circle x y) ∧ -- Prove $A$ is true.
  ¬(∃ λ : ℝ, ∀ x y : ℝ, ¬(line λ x y ∧ circle x y)) ∧ -- Prove $B$ is false.
  (∃ λ : ℝ, ∃ x₁ y₁ x₂ y₂ : ℝ, (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ line λ x₁ y₁ ∧ circle x₁ y₁ ∧ line λ x₂ y₂ ∧ circle x₂ y₂) ∧ -- Prove $C$ is true.
  (∃ P : ℝ × ℝ, ∃ λ : ℝ, circle P.1 P.2 ∧ (∀ Q : ℝ × ℝ, line λ Q.1 Q.2 → dist P Q ≤ 4) ∧ dist P (sqrt 3, 1) = 4) := -- Prove $D$ is true.
sorry

end trajectory_and_line_properties_l320_320787


namespace sum_sin_squared_angles_l320_320265

theorem sum_sin_squared_angles : 
  ∑ k in finset.range 30, (sin (6 * (k + 1) * (real.pi / 180)))^2 = 31 / 2 := 
sorry

end sum_sin_squared_angles_l320_320265


namespace total_interest_at_tenth_year_l320_320168

-- Define the conditions for the simple interest problem
variables (P R T : ℝ)

-- Given conditions in the problem
def initial_condition : Prop := (P * R * 10) / 100 = 800
def trebled_principal_condition : Prop := (3 * P * R * 5) / 100 = 1200

-- Statement to prove
theorem total_interest_at_tenth_year (h1 : initial_condition P R) (h2 : trebled_principal_condition P R) :
  (800 + 1200) = 2000 := by
  sorry

end total_interest_at_tenth_year_l320_320168


namespace distance_inequality_l320_320446

theorem distance_inequality
  (A B C P : Point)
  (a b c : ℝ)
  (hP_interior : interior_of_triangle A B C P)
  (ha : a = dist B C)
  (hb : b = dist C A)
  (hc : c = dist A B) :
  (dist P A) / a + (dist P B) / b + (dist P C) / c ≥  sqrt 3 := sorry

end distance_inequality_l320_320446


namespace max_leap_years_in_180_period_l320_320225

def is_leap_year (year : ℕ) : Prop :=
  (year % 4 = 0 ∧ ¬ (year % 100 = 0 ∧ year % 400 ≠ 0))

def max_leap_years_in_180_years := 43

theorem max_leap_years_in_180_period : 
  ∀ (startYear : ℕ),
  (∑ y in (Finset.range 180).map (λ n, startYear + n), if is_leap_year y then 1 else 0) 
  ≤ max_leap_years_in_180_years :=
sorry

end max_leap_years_in_180_period_l320_320225


namespace solution_set_of_inequality_l320_320847

theorem solution_set_of_inequality (a x : ℝ) (h : 1 < a) :
  (x - a) * (x - (1 / a)) > 0 ↔ x < 1 / a ∨ x > a :=
by
  sorry

end solution_set_of_inequality_l320_320847


namespace amount_spent_on_marbles_l320_320688

-- Definitions of conditions
def cost_of_football : ℝ := 5.71
def total_spent_on_toys : ℝ := 12.30

-- Theorem statement
theorem amount_spent_on_marbles : (total_spent_on_toys - cost_of_football) = 6.59 :=
by
  sorry

end amount_spent_on_marbles_l320_320688


namespace time_to_destination_l320_320158

-- Variables and Conditions
variable (Harris_speed : ℝ)  -- Speed of Mr. Harris (in units of distance per hour)
variable (your_speed : ℝ)  -- Your speed (in units of distance per hour)
variable (time_Harris : ℝ) -- Time taken by Mr. Harris to reach the store (in hours)
variable (distance_ratio : ℝ) -- Ratio of your destination distance to the store distance

-- Hypotheses based on given conditions
hypothesis h1 : your_speed = 2 * Harris_speed
hypothesis h2 : time_Harris = 2
hypothesis h3 : distance_ratio = 3

-- Statement to prove
theorem time_to_destination : (3 / 2 : ℝ) = 3 := by 
    sorry

end time_to_destination_l320_320158


namespace induction_factor_l320_320616

theorem induction_factor (k : ℕ) (h : k > 0) :
  let LHS_nk := (∏ i in (finset.range k).map (λ n, n + 1), (k + i))
  let LHS_nkp1 := (∏ i in (finset.range (k + 1)).map (λ n, n + 1), (k + 1 + i))
  LHS_nkp1 = LHS_nk * (frac (2k + 1) (k + 1)) * (2k + 2) :=
begin
  sorry
end

end induction_factor_l320_320616


namespace find_positive_integer_n_l320_320741

theorem find_positive_integer_n :
  ∃ (n : ℕ), (∃ (k : ℕ) (m : ℕ → ℕ), (∀ i, 1 ≤ i ∧ i ≤ k → m i ≥ 4) ∧ (n = ∏ i in finset.range k, m i) ∧ 
    n = 2^( (∏ i in finset.range k, (m i - 1)) / 2^k ) - 1) ∧ 
    n = 7 :=
by
  sorry

end find_positive_integer_n_l320_320741


namespace find_PR_plus_PS_l320_320177

open EuclideanGeometry

-- Square with side length s
variables {s : ℝ} (A B C D P S R F Q : Point)

-- Conditions
variable (H1 : Square A B C D)
variable (H2 : OnLine P A B)
variable (H3 : Perpendicular (P -- S) (B -- D))
variable (H4 : Perpendicular (P -- R) (A -- C))
variable (H5 : Perpendicular (A -- F) (B -- D))
variable (H6 : Midpoint F B D)
variable (H7 : Perpendicular (P -- Q) (A -- F))

-- Conclusion
theorem find_PR_plus_PS : PR + PS = s * Real.sqrt 2 := by
  sorry

end find_PR_plus_PS_l320_320177


namespace red_balloon_is_one_l320_320598

open Nat

theorem red_balloon_is_one (R B : Nat) (h1 : R + B = 85) (h2 : R ≥ 1) (h3 : ∀ i j, i < R → j < R → i ≠ j → (i < B ∨ j < B)) : R = 1 :=
by
  sorry

end red_balloon_is_one_l320_320598


namespace multiplication_distributive_example_l320_320176

theorem multiplication_distributive_example : 23 * 4 = 20 * 4 + 3 * 4 := by
  sorry

end multiplication_distributive_example_l320_320176


namespace sum_of_roots_l320_320753

theorem sum_of_roots : ∀ x : ℝ, x^2 - 2004 * x + 2021 = 0 → x = 2004 := by
  sorry

end sum_of_roots_l320_320753


namespace minimum_product_l320_320464

-- Definitions based on the given conditions
variables {x1 x2 x3 : ℝ}

-- Given conditions
def conditions : Prop :=
  (0 < x1) ∧ (0 < x2) ∧ (0 < x3) ∧ 
  (x1 + x2 + x3 = 4) ∧ 
  (∀ {i j : ℕ}, i ≠ j → ∃ (xi xj : ℝ), xi = x1 ∨ xi = x2 ∨ xi = x3 ∧ xi = x1 ∨ xi = x2 ∨ xi = x3 ∧ 2 * xi^2 + 2 * xj^2 - 5 * xi * xj ≤ 0)

-- The objective to prove
theorem minimum_product : conditions → ∃ (m : ℝ), (m = x1 * x2 * x3) ∧ (m = 2) :=
by
  intro h
  sorry

end minimum_product_l320_320464


namespace not_perfect_square_l320_320555

theorem not_perfect_square (a : ℤ) : ¬ (∃ x : ℤ, a^2 + 4 = x^2) := 
sorry

end not_perfect_square_l320_320555


namespace cost_price_correct_l320_320891

def total_cost : ℝ := 407
def total_length : ℝ := 9.25
def cost_price_per_meter (total_cost : ℝ) (total_length : ℝ) : ℝ := total_cost / total_length

theorem cost_price_correct : cost_price_per_meter total_cost total_length = 44 :=
by
  -- the proof will go here
  sorry

end cost_price_correct_l320_320891


namespace rectangle_perimeter_l320_320674

-- Definitions of the conditions
def side_length_square : ℕ := 75  -- side length of the square in mm
def height_sum (x y z : ℕ) : Prop := x + y + z = side_length_square  -- sum of heights of the rectangles

-- Perimeter definition
def perimeter (h : ℕ) (w : ℕ) : ℕ := 2 * (h + w)

-- Statement of the problem
theorem rectangle_perimeter (x y z : ℕ) (h_sum : height_sum x y z)
  (h1 : perimeter x side_length_square = (perimeter y side_length_square + perimeter z side_length_square) / 2)
  : perimeter x side_length_square = 200 := by
  sorry

end rectangle_perimeter_l320_320674


namespace prob_C_equals_prob_D_l320_320879

-- Defining the points of the teams
def points_A := 22
def points_B := 22
def points_C := 21
def points_D := 20

-- Defining the probabilities of winning for each team are equal
axiom equal_probabilities : ∀(X Y : Type), (winning_probability X = winning_probability Y)

-- Define the theorem to compare probabilities of C and D advancing
theorem prob_C_equals_prob_D : probability_of_advancing C = probability_of_advancing D :=
by
  -- The proof goes here, which we are skipping.
  sorry

end prob_C_equals_prob_D_l320_320879


namespace angle_between_vectors_l320_320042

open Real

variables {V : Type*} [inner_product_space ℝ V]

theorem angle_between_vectors 
  (a b : V) 
  (ha : ∥a∥ ≠ 0) 
  (hb : ∥b∥ ≠ 0) 
  (hab : ∥a∥ = ∥b∥) 
  (dot_product_cond : (2 • a + b) ⬝ b = 0) 
  : real.angle a b = real.angle_of_degrees 120 :=
sorry

end angle_between_vectors_l320_320042


namespace masha_numbers_unique_l320_320482

def natural_numbers : Set ℕ := {n | n > 11}

theorem masha_numbers_unique (a b : ℕ) (ha : a ∈ natural_numbers) (hb : b ∈ natural_numbers) (hne : a ≠ b)
  (hs_equals : ∃ S, S = a + b)
  (sasha_initially_uncertain : ∀ (a b : ℕ), a ∈ natural_numbers → b ∈ natural_numbers → a ≠ b → (a + b = S) → ¬ (Sasha_can_determine_initially a b S))
  (masha_hint : ∃ (a_even : ℕ), a_even ∈ natural_numbers ∧ (a_even % 2 = 0) ∧ (a_even = a ∨ a_even = b))
  (sasha_then_confident : ∀ (a b : ℕ), a ∈ natural_numbers → b ∈ natural_numbers → a ≠ b → (a + b = S) → (a_even = a ∨ a_even = b) → Sasha_can_determine_confidently a b S) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) := by
  sorry

end masha_numbers_unique_l320_320482


namespace reciprocals_of_each_other_l320_320399

theorem reciprocals_of_each_other (a b : ℝ) (h : (a + b)^2 - (a - b)^2 = 4) : a * b = 1 :=
by 
  sorry

end reciprocals_of_each_other_l320_320399


namespace most_likely_red_balls_l320_320000

-- Define the conditions as variables
variable (total_balls : ℕ)
variable (prob_red : ℝ)

-- Define the expected number of red balls
def num_red_balls (total_balls : ℕ) (prob_red : ℝ) : ℝ :=
  total_balls * prob_red

-- Prove the main statement
theorem most_likely_red_balls (h1 : total_balls = 20) (h2 : prob_red = 0.4) :
  num_red_balls total_balls prob_red = 8 :=
by
  -- Substituting the values given in the hypotheses
  rw [h1, h2]
  -- Simplify the expression
  norm_num

-- A placeholder for the full implementation
sorry

end most_likely_red_balls_l320_320000


namespace cartons_loaded_l320_320678

def total_cartons : Nat := 50
def cans_per_carton : Nat := 20
def cans_left_to_load : Nat := 200

theorem cartons_loaded (C : Nat) (h : cans_per_carton ≠ 0) : 
  C = total_cartons - (cans_left_to_load / cans_per_carton) := by
  sorry

end cartons_loaded_l320_320678


namespace op_proof_l320_320285

-- Definition of the operation \(\oplus\)
def op (x y : ℝ) : ℝ := x^2 + y

-- Theorem statement for the given proof problem
theorem op_proof (h : ℝ) : op h (op h h) = 2 * h^2 + h :=
by 
  sorry

end op_proof_l320_320285


namespace quadrilateral_is_parallelogram_l320_320064

-- Given a quadrilateral, prove it is a parallelogram under given conditions:
theorem quadrilateral_is_parallelogram
  (A B C D P Q : Point)
  (O : Point)
  (r1 r2 : ℝ)
  (h1 : projections_of_point_lies_on_circle P A B C D O r1)
  (h2 : projections_of_point_lies_on_circle Q A B C D O r2)
  (h3 : r1 ≠ r2)
  : parallelogram A B C D :=
by sorry

end quadrilateral_is_parallelogram_l320_320064


namespace region_area_l320_320618

theorem region_area :
  let region := {p : ℝ × ℝ | |4 * p.1 - 24| + |3 * p.2 + 10| ≤ 4} in
  ∃ area : ℝ, area = 4 / 3 ∧ (measure.theory.measure {} region).to_value = area :=
begin
  sorry
end

end region_area_l320_320618


namespace min_range_six_observations_l320_320669

theorem min_range_six_observations (x : Fin 6 → ℝ) 
  (h_mean : (∑ i, x i) / 6 = 8) 
  (h_median : (x 2 + x 3) / 2 = 9) : 
  ∃ (x_min x_max : ℝ), (x_max - x_min) = 3 :=
by
  sorry

end min_range_six_observations_l320_320669


namespace rainy_days_last_week_l320_320932

-- All conditions in Lean definitions
def even_integer (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k
def cups_of_tea_n (n : ℤ) : ℤ := 3
def total_drinks (R NR : ℤ) (m : ℤ) : Prop := 2 * m * R + 3 * NR = 36
def more_tea_than_hot_chocolate (R NR : ℤ) (m : ℤ) : Prop := 3 * NR - 2 * m * R = 12
def odd_number_of_rainy_days (R : ℤ) : Prop := R % 2 = 1
def total_days_in_week (R NR : ℤ) : Prop := R + NR = 7

-- Main statement
theorem rainy_days_last_week : ∃ R m NR : ℤ, 
  odd_number_of_rainy_days R ∧ 
  total_days_in_week R NR ∧ 
  total_drinks R NR m ∧ 
  more_tea_than_hot_chocolate R NR m ∧
  R = 3 :=
by
  sorry

end rainy_days_last_week_l320_320932


namespace arrange_7_people_l320_320695

open Finset

theorem arrange_7_people (A B C D : ℕ) (others : Finset ℕ) 
  (h : others.card = 3) 
  (hA_front : ∀ perm : Perm (A::B::C::D::others), perm 0 ≠ A) 
  (hB_end : ∀ perm : Perm (A::B::C::D::others), perm 6 ≠ B) 
  (hC_D_adj : ∀ perm : Perm (A::B::C::D::others), (∀ i, (C::D::[]) ⊂ (perm i))) :
  card ((A :: B :: C :: D :: (others.toList)).permutations.filter (λ perm, ¬perm.head? == some A ∧ ¬perm.reverse.head? == some B ∧
    ((perm.indexOf C + 1 = perm.indexOf D) ∨ (perm.indexOf D + 1 = perm.indexOf C)))) = 1008 := 
sorry

end arrange_7_people_l320_320695


namespace all_radii_equal_l320_320580
-- Lean 4 statement

theorem all_radii_equal (r : ℝ) (h : r = 2) : r = 2 :=
by
  sorry

end all_radii_equal_l320_320580


namespace range_of_a_l320_320065

def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0
def q (a : ℝ) : Prop := ∀ x : ℝ, (3 - 2 * a) ^ x > 0 -- using our characterization for 'increasing'

theorem range_of_a (a : ℝ) : (p a ∨ q a) ∧ ¬ (p a ∧ q a) ↔ (a ≤ -2 ∨ (1 ≤ a ∧ a < 2)) :=
by
  sorry

end range_of_a_l320_320065


namespace cot_ratio_l320_320033

theorem cot_ratio (a b c : ℝ) (α β γ : ℝ) (h1 : a^2 + b^2 = 2020 * c^2)
  (h2 : α + β + γ = π)
  (h3 : LawOfSines a b c α β γ)
  (h4 : ∀ x, cot x = cos x / sin x) :
  (cot γ) / (cot α + cot β) = 1009.5 :=
by
  sorry

end cot_ratio_l320_320033


namespace uki_total_profit_five_days_l320_320615

theorem uki_total_profit_five_days
  (selling_price_cupcake : ℚ := 1.50)
  (selling_price_cookies : ℚ := 2.00)
  (selling_price_biscuits : ℚ := 1.00)
  (average_cupcakes_per_day : ℕ := 20)
  (average_cookies_per_day : ℕ := 10)
  (average_biscuits_per_day : ℕ := 20)
  (cost_cupcake : ℚ := 0.75)
  (cost_cookies : ℚ := 1.00)
  (cost_biscuits : ℚ := 0.50)
  (days : ℕ := 5) :
  let profit_per_cupcake := selling_price_cupcake - cost_cupcake,
      daily_profit_cupcake := profit_per_cupcake * average_cupcakes_per_day,
      profit_per_cookies := selling_price_cookies - cost_cookies,
      daily_profit_cookies := profit_per_cookies * average_cookies_per_day,
      profit_per_biscuits := selling_price_biscuits - cost_biscuits,
      daily_profit_biscuits := profit_per_biscuits * average_biscuits_per_day,
      total_daily_profit := daily_profit_cupcake + daily_profit_cookies + daily_profit_biscuits,
      total_profit := total_daily_profit * days
  in
  total_profit = 175 :=
by sorry

end uki_total_profit_five_days_l320_320615


namespace number_of_even_divisors_of_factorial_eight_l320_320831

-- Definition of 8! and its prime factorization
def factorial_eight : ℕ := 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1
def prime_factorization_factorial_eight : Prop :=
  factorial_eight = 2^7 * 3^2 * 5 * 7

-- The main theorem statement
theorem number_of_even_divisors_of_factorial_eight :
  prime_factorization_factorial_eight →
  ∃ n, n = 7 * 3 * 2 * 2 ∧
  (∀ d, d ∣ factorial_eight → (∃ a b c d, 1 ≤ a ∧ a ≤ 7 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 1 ∧ 0 ≤ d ∧ d ≤ 1 ∧ d = 2^a * 3^b * 5^c * 7^d) →
  (7 * 3 * 2 * 2 = n)) :=
by
  intro h
  use 84
  sorry

end number_of_even_divisors_of_factorial_eight_l320_320831


namespace units_digit_of_n_l320_320754

def units_digit (x : ℕ) : ℕ := x % 10

theorem units_digit_of_n 
  (m n : ℕ) 
  (h1 : m * n = 21 ^ 6) 
  (h2 : units_digit m = 7) : 
  units_digit n = 3 := 
sorry

end units_digit_of_n_l320_320754


namespace point_B_is_4_l320_320062

def point_A : ℤ := -3
def units_to_move : ℤ := 7
def point_B : ℤ := point_A + units_to_move

theorem point_B_is_4 : point_B = 4 :=
by
  sorry

end point_B_is_4_l320_320062


namespace smallest_omega_l320_320608

theorem smallest_omega (ω : ℝ) (hω : ω > 0)
  (f : ℝ → ℝ) (h₁ : ∀ x, f x = 2 * sin (ω * x - π / 4)) :
  (∀ x, 2 * sin (ω * (x + π / 4) - π / 4) = 2 * sin (ω * (x - π / 4) - π / 4)) →
  (∃ k : ℤ, ω = 2 * k ∧ k > 0) → 
  ω = 2 := 
by
  sorry

end smallest_omega_l320_320608


namespace magnitude_of_AB_l320_320792

open Real EuclideanGeometry

variable (OA OB AB : ℝ × ℝ)

-- Definition of vector OA and conditions
def vector_OA : ℝ × ℝ := (1, -3)

def orthogonal_vectors (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

def same_magnitude (u v : ℝ × ℝ) : Prop :=
  u.1 ^ 2 + u.2 ^ 2 = v.1 ^ 2 + v.2 ^ 2

def magnitude (v : ℝ × ℝ) : ℝ :=
  sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Proving the final magnitude of vector AB
theorem magnitude_of_AB
  (h1 : same_magnitude vector_OA OB)
  (h2 : orthogonal_vectors vector_OA OB)
  (h3 : AB = (OB.1 - 1, OB.2 + 3)) :
  magnitude AB = 2 * sqrt 5 := 
sorry

end magnitude_of_AB_l320_320792


namespace new_efficiency_ratio_l320_320116

theorem new_efficiency_ratio (M : ℝ) (x : ℕ) (r1 r2 : ℝ)
  (h1 : r1 = 322.5) (h2 : r2 = 187.5)
  (h3 : M = r1 * x)
  (h4 : M = r2 * (x + 18)) :
  M / 50 = 161.25 := 
  by sorry

noncomputable def main :=
  new_efficiency_ratio 8062.5 25 322.5 187.5 rfl rfl 
    (by unfold r1; linarith)
    (by unfold r2; linarith)

#eval main

end new_efficiency_ratio_l320_320116


namespace domino_tiling_impossible_l320_320887

theorem domino_tiling_impossible :
  ¬ ∃ (tiles : list (ℕ × ℕ × bool)), 
    (∀ t ∈ tiles, t.snd.fst < 6 ∧ t.snd.fst + (if t.snd.snd then 1 else 0) < 6 ∧ t.fst.fst < 6 ∧ t.fst.fst + (if ¬t.snd.snd then 1 else 0) < 6 ∧ (t.fst.fst + t.snd.fst) % 2 = 1) ∧  
    (length tiles = 18) ∧ 
    (∀ seam, (seam < 6 → (∀ t ∈ tiles, (t.fst.fst = seam ∨ t.snd.fst = seam) ∧ (t.fst.snd = seam ∨ t.snd.snd = seam) → faliste)) :=
by
  sorry

end domino_tiling_impossible_l320_320887


namespace inequality_proof_l320_320352

variable (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
variable (h_ab_bc_ca : a * b + b * c + c * a = 1)

theorem inequality_proof :
  (3 / Real.sqrt (a^2 + 1)) + (4 / Real.sqrt (b^2 + 1)) + (12 / Real.sqrt (c^2 + 1)) < (39 / 2) :=
by
  sorry

end inequality_proof_l320_320352


namespace fraction_of_cookies_with_nuts_l320_320468

theorem fraction_of_cookies_with_nuts
  (nuts_per_cookie : ℤ)
  (total_cookies : ℤ)
  (total_nuts : ℤ)
  (h1 : nuts_per_cookie = 2)
  (h2 : total_cookies = 60)
  (h3 : total_nuts = 72) :
  (total_nuts / nuts_per_cookie) / total_cookies = 3 / 5 := by
  sorry

end fraction_of_cookies_with_nuts_l320_320468


namespace square_properties_l320_320620

/-- Define the side length of the square -/
def side_length : ℝ := 30 * Real.sqrt 3

/-- Define the diagonal of the square -/
def diagonal : ℝ := side_length * Real.sqrt 2

/-- Define the perimeter of the square -/
def perimeter : ℝ := 4 * side_length

/-- Main theorem to verify the diagonal and the perimeter -/
theorem square_properties :
  diagonal = 30 * Real.sqrt 6 ∧ perimeter = 120 * Real.sqrt 3 :=
by
  sorry

end square_properties_l320_320620


namespace decreasing_power_function_unique_m_l320_320720

theorem decreasing_power_function_unique_m (m : ℝ) :
  (m^2 - m - 1 > 0) ∧ (m^2 - 2m - 3 < 0) → m = 2 :=
by
  sorry

end decreasing_power_function_unique_m_l320_320720


namespace sum_sin_squares_deg_6_to_174_l320_320237

theorem sum_sin_squares_deg_6_to_174 : 
  ∑ k in Finset.range 30, Real.sin (6 * k * (Real.pi / 180)) ^ 2 = 15.5 := by
    sorry

end sum_sin_squares_deg_6_to_174_l320_320237


namespace largest_possible_b_l320_320117

theorem largest_possible_b (a b c : ℕ) (h1 : 1 < c) (h2 : c < b) (h3 : b < a) (h4 : a * b * c = 360) : b = 10 :=
sorry

end largest_possible_b_l320_320117


namespace no_infinite_sequence_of_positive_integers_l320_320722

theorem no_infinite_sequence_of_positive_integers (a : ℕ → ℕ) (h: ∀ n, 0 < a n) : 
  ¬ (∀ n, a (n+2) = nat.sqrt(a (n+1)) + a n) :=
by sorry

end no_infinite_sequence_of_positive_integers_l320_320722


namespace closest_point_to_origin_l320_320944

def y (x : ℝ) := x + 1 / x

theorem closest_point_to_origin : ∃ x : ℝ, x > 0 ∧ (x, y x) = (1 / 2^(1/4 : ℝ), (1 + real.sqrt 2) / 2^(1/4 : ℝ)) :=
by
  sorry

end closest_point_to_origin_l320_320944


namespace even_function_a_value_l320_320356

theorem even_function_a_value {f : ℝ → ℝ} (a : ℝ) :
  (∀ x : ℝ, f x = x^3 * (a * 2^x - 2^(-x)) ∧ f x = f (-x)) → a = 1 :=
by
  intros h,
  sorry

end even_function_a_value_l320_320356


namespace problem_solution_l320_320323

theorem problem_solution (a b : ℝ) (h0 : 0 < a) (h1 : a < b) (h2 : a + b = 1) :
  (Real.log a / Real.log 2) + (Real.log b / Real.log 2) < -2 :=
by
  have h3 : a + b = 1 := h2
  calc 
    Real.log a / Real.log 2 + Real.log b / Real.log 2 
        = (Real.log (a * b) / Real.log 2) : by sorry
    ... < -2 : by sorry

end problem_solution_l320_320323


namespace closest_point_l320_320951

noncomputable def closest_point_to_origin : ℝ × ℝ :=
  let x := (1 : ℝ) / Real.root 2 4 in
  let y := x + 1 / x in
  (x, y)

theorem closest_point (x y : ℝ) (h : y = x + 1 / x) (hx : x > 0) :
  (x, y) = closest_point_to_origin :=
begin
  sorry
end

end closest_point_l320_320951


namespace single_appetizer_cost_l320_320605

def total_cost : ℝ := 50
def entree_percent : ℝ := 0.80
def appetizer_percent : ℝ := 0.20
def number_of_appetizers : ℕ := 2

theorem single_appetizer_cost :
  let entree_cost := entree_percent * total_cost in
  let appetizer_cost := appetizer_percent * total_cost in
  let single_appetizer_cost := appetizer_cost / number_of_appetizers in
  single_appetizer_cost = 5 := by
  sorry

end single_appetizer_cost_l320_320605


namespace z_real_iff_z_complex_iff_z_pure_imag_iff_l320_320624

noncomputable def z (m : ℝ) : ℂ := complex.mk (m + 1) (m - 1)

theorem z_real_iff (m : ℝ) : (∃ (re : ℝ), z m = re) ↔ m = 1 :=
by sorry

theorem z_complex_iff (m : ℝ) : (∃ (re im : ℝ), z m = complex.mk re im) ↔ m ≠ 1 :=
by sorry

theorem z_pure_imag_iff (m : ℝ) : (∃ (im : ℝ), z m = complex.mk 0 im) ↔ m = -1 :=
by sorry

end z_real_iff_z_complex_iff_z_pure_imag_iff_l320_320624


namespace adam_and_bettie_same_score_l320_320218

def score (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n % 2 = 1 then score (n - 1) + 1
  else score (n / 2)

def probability_same_score (p q : ℕ) : Prop :=
  ∃ (p q : ℕ), Nat.gcd p q = 1 ∧ (∑ k in Finset.range (8), (Nat.choose 7 k)^2) = p ∧ (2^14) = q ∧ p = 429
  
theorem adam_and_bettie_same_score : probability_same_score 429 2048 :=
  sorry

end adam_and_bettie_same_score_l320_320218


namespace units_digit_k_squared_plus_2_k_l320_320459

noncomputable def k : ℕ := 2017^2 + 2^2017

theorem units_digit_k_squared_plus_2_k : (k^2 + 2^k) % 10 = 3 := 
  sorry

end units_digit_k_squared_plus_2_k_l320_320459


namespace determinant_of_matrix_is_zero_l320_320277

theorem determinant_of_matrix_is_zero :
  let A := Matrix.ofVecs 3 3 [![sin 2, sin 3, sin 4], ![sin 5, sin 6, sin 7], ![sin 8, sin 9, sin 10]] in
  A.det = 0 :=
by {
  sorry
}

end determinant_of_matrix_is_zero_l320_320277


namespace car_time_ratio_l320_320141

theorem car_time_ratio (S V0 : ℝ) (hV0 : 0 < V0) :
  let V1 := V0 / 3 in
  let t1 := S / V1 in
  let t2 := (sum (finset.range 8) (λ i, (S / (2 ^ i) / (V0 / (2 ^ (i - 1))))) + S / V0) in
  t2 / t1 = 5 / 3 :=
by
  sorry

end car_time_ratio_l320_320141


namespace valid_params_l320_320094

-- Definition for the parameterization of a line
def parameterization (base: ℝ × ℝ) (dir: ℝ × ℝ) (t : ℝ) : ℝ × ℝ :=
  (base.1 + t * dir.1, base.2 + t * dir.2)

-- Definition for the line equation y = (5/3)x - 5
def line_eq (x y : ℝ) : Prop := y = (5/3) * x - 5

-- Definitions for the bases and directions in the given options
def base_A : ℝ × ℝ := (0, -5)
def direction_A : ℝ × ℝ := (3, 5)
def base_B : ℝ × ℝ := (3, 0)
def direction_B : ℝ × ℝ := (5, 3)
def base_C : ℝ × ℝ := (-3, -10)
def direction_C : ℝ × ℝ := (9, 15)
def base_D : ℝ × ℝ := (5, 10 / 3)
def direction_D : ℝ × ℝ := (15, 25)
def base_E : ℝ × ℝ := (1, -10 / 3)
def direction_E : ℝ × ℝ := (5, 9)

-- Correct parameterizations
def valid_parameterizations : list (ℝ × ℝ × ℝ × ℝ) :=
  [(base_A, direction_A), (base_C, direction_C), (base_D, direction_D)]

-- Statement to prove the valid parameterizations are exactly A, C, D
theorem valid_params : 
  ∀ (t : ℝ), 
  (parameterization base_A direction_A t = (base_A.1 + t * direction_A.1, base_A.2 + t * direction_A.2) ∧ line_eq (parameterization base_A direction_A t).1 (parameterization base_A direction_A t).2) ∧
  (parameterization base_C direction_C t = (base_C.1 + t * direction_C.1, base_C.2 + t * direction_C.2) ∧ line_eq (parameterization base_C direction_C t).1 (parameterization base_C direction_C t).2) ∧
  (parameterization base_D direction_D t = (base_D.1 + t * direction_D.1, base_D.2 + t * direction_D.2) ∧ line_eq (parameterization base_D direction_D t).1 (parameterization base_D direction_D t).2) :=
by sorry

end valid_params_l320_320094


namespace twenty_percent_less_than_sixty_equals_one_third_more_than_what_number_l320_320137

theorem twenty_percent_less_than_sixty_equals_one_third_more_than_what_number :
  (4 / 3) * n = 48 → n = 36 :=
by
  intro h
  sorry

end twenty_percent_less_than_sixty_equals_one_third_more_than_what_number_l320_320137


namespace average_gas_mileage_round_trip_l320_320675

theorem average_gas_mileage_round_trip :
  let distance_to_city := 150
  let mpg_sedan := 25
  let mpg_rental := 15
  let total_distance := 2 * distance_to_city
  let gas_used_outbound := distance_to_city / mpg_sedan
  let gas_used_return := distance_to_city / mpg_rental
  let total_gas_used := gas_used_outbound + gas_used_return
  let avg_gas_mileage := total_distance / total_gas_used
  avg_gas_mileage = 18.75 := by
{
  sorry
}

end average_gas_mileage_round_trip_l320_320675


namespace determine_numbers_l320_320473

theorem determine_numbers (a b : ℕ) (S : ℕ) (h1 : a ≠ b) (h2 : a > 11) (h3 : b > 11)
  (h4 : S = a + b) (h5 : (∀ (x y : ℕ), x + y = S → x ≠ y → (x = a ∧ y = b) ∨ (x = b ∧ y = a)) = false)
  (h6 : a % 2 = 0 ∨ b % 2 = 0) : 
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) := 
  sorry

end determine_numbers_l320_320473


namespace ratio_new_circumference_new_diameter_l320_320857

theorem ratio_new_circumference_new_diameter (r : ℝ) : 
  let new_radius := r + 2 in
  let new_diameter := 2 * new_radius in
  let new_circumference := 2 * Real.pi * new_radius in
  new_circumference / new_diameter = Real.pi :=
by
  -- proof omitted
  sorry

end ratio_new_circumference_new_diameter_l320_320857


namespace determine_numbers_l320_320476

theorem determine_numbers (a b : ℕ) (S : ℕ) (h1 : a ≠ b) (h2 : a > 11) (h3 : b > 11)
  (h4 : S = a + b) (h5 : (∀ (x y : ℕ), x + y = S → x ≠ y → (x = a ∧ y = b) ∨ (x = b ∧ y = a)) = false)
  (h6 : a % 2 = 0 ∨ b % 2 = 0) : 
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) := 
  sorry

end determine_numbers_l320_320476


namespace height_of_triangle_l320_320560

-- Definition of the base and area of the triangle
def base : ℝ := 18
def area : ℝ := 54

-- The formula for the area of a triangle
def area_of_triangle (b h : ℝ) : ℝ := (b * h) / 2

-- The height of the triangle to be proven
theorem height_of_triangle : ∀ (h : ℝ), area_of_triangle base h = area → h = 6 :=
by
  intro h
  intro h_area
  sorry

end height_of_triangle_l320_320560


namespace solve_for_a_l320_320801

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 1 else 2 * x

theorem solve_for_a (a : ℝ) (h : f a = 10) : a = 5 ∨ a = -3 :=
by
  sorry

end solve_for_a_l320_320801


namespace min_value_product_l320_320149

theorem min_value_product (a : Fin 6 → ℕ) 
  (hperm : ∃ σ : Equiv.Perm (Fin 6), ∀ i, a i = σ i) :
  (∏ i, (a i - a ((i + 1) % 6)) / (a ((i + 2) % 6) - a ((i + 3) % 6))) = 1 := by  
  sorry

end min_value_product_l320_320149


namespace distance_between_points_l320_320816

theorem distance_between_points (A B : ℝ) (hA : |A| = 2) (hB : |B| = 7) :
  |A - B| = 5 ∨ |A - B| = 9 := 
sorry

end distance_between_points_l320_320816


namespace rupert_candles_l320_320059

theorem rupert_candles (peter_candles : ℕ) (rupert_times_older : ℝ) (h1 : peter_candles = 10) (h2 : rupert_times_older = 3.5) :
    ∃ rupert_candles : ℕ, rupert_candles = peter_candles * rupert_times_older := 
by
  sorry

end rupert_candles_l320_320059


namespace vector_subtract_scalar_mult_l320_320303

def v1 : ℝ × ℝ := (2, -5)
def v2 : ℝ × ℝ := (-1, 7)
def k : ℝ := 4

theorem vector_subtract_scalar_mult :
  (v1.1 - k * v2.1, v1.2 - k * v2.2) = (6, -33) :=
by
  -- vector components
  let x1 := v1.1
  let y1 := v1.2
  let x2 := v2.1
  let y2 := v2.2
  -- proofs
  calc
    (x1 - k * x2) = 6 := sorry
    (y1 - k * y2) = -33 := sorry

end vector_subtract_scalar_mult_l320_320303


namespace sum_sin_squared_angles_l320_320262

theorem sum_sin_squared_angles : 
  ∑ k in finset.range 30, (sin (6 * (k + 1) * (real.pi / 180)))^2 = 31 / 2 := 
sorry

end sum_sin_squared_angles_l320_320262


namespace triangle_angle_cosine_identity_l320_320431
noncomputable theory
open Real

theorem triangle_angle_cosine_identity (A B C : ℝ) (a b c : ℝ)
  (hA : A > 0) (hB : 0 < B ∧ B < π) (hC : C > 0)
  (h_cosine : 2 * b * cos B = a * cos C + c * cos A) :
  B = π / 3 :=
sorry

end triangle_angle_cosine_identity_l320_320431


namespace line_repr_exists_same_line_iff_scalar_multiple_l320_320174

-- Given that D is a line in 3D space, there exist a, b, c not all zero
theorem line_repr_exists
  (D : Set (ℝ × ℝ × ℝ)) :
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) ∧ 
  (D = {p | ∃ (u v w : ℝ), p = (u, v, w) ∧ a * u + b * v + c * w = 0}) :=
sorry

-- Given two lines represented by different coefficients being the same
-- Prove that the coefficients are scalar multiples of each other
theorem same_line_iff_scalar_multiple
  (α1 β1 γ1 α2 β2 γ2 : ℝ) :
  (∀ (u v w : ℝ), α1 * u + β1 * v + γ1 * w = 0 ↔ α2 * u + β2 * v + γ2 * w = 0) ↔
  (∃ k : ℝ, k ≠ 0 ∧ α2 = k * α1 ∧ β2 = k * β1 ∧ γ2 = k * γ1) :=
sorry

end line_repr_exists_same_line_iff_scalar_multiple_l320_320174


namespace find_number_l320_320134

theorem find_number (n : ℝ) : (4 / 3) * n = 48 → n = 36 :=
by
  intro h
  have h1 : n = 48 * (3 / 4) := by
    sorry
  exact h1

end find_number_l320_320134


namespace intersection_and_union_of_A_and_B_l320_320045

section

variable (U : Set ℕ)
variable (A : Set ℕ := {0, 2, 4, 6})
variable (CUA : Set ℕ := {-1, -3, 1, 3})
variable (CUB : Set ℕ := {-1, 0, 2})

def set_B : Set ℕ := U \ CUB

theorem intersection_and_union_of_A_and_B :
  let B := set_B U CUB in
  (U = {-3, -1, 0, 1, 2, 3, 4, 6}) →
  (B = {-3, 1, 3, 4, 6}) →
  (A ∩ B = {4, 6}) ∧ (A ∪ B = {-3, 0, 1, 2, 3, 4, 6}) :=
begin
  intros B hU hB,
  split,
  sorry, -- Prove A ∩ B = {4, 6}
  sorry  -- Prove A ∪ B = {-3, 0, 1, 2, 3, 4, 6}
end

end

end intersection_and_union_of_A_and_B_l320_320045


namespace problem_solution_l320_320551

theorem problem_solution (x y z : ℝ) (h1 : x * y + y * z + z * x = 4) (h2 : x * y * z = 6) :
  (x * y - (3 / 2) * (x + y)) * (y * z - (3 / 2) * (y + z)) * (z * x - (3 / 2) * (z + x)) = 81 / 4 :=
by
  sorry

end problem_solution_l320_320551


namespace sum_of_squares_of_sines_l320_320245

theorem sum_of_squares_of_sines : 
  (\sum n in finset.range 1 30, real.sin (6 * n : ℝ) ^ 2) = 31 / 2 :=
begin
  sorry
end

end sum_of_squares_of_sines_l320_320245


namespace paula_cans_used_l320_320963

/-- 
  Paula originally had enough paint to cover 42 rooms. 
  Unfortunately, she lost 4 cans of paint on her way, 
  and now she can only paint 34 rooms. 
  Prove the number of cans she used for these 34 rooms is 17.
-/
theorem paula_cans_used (R L P C : ℕ) (hR : R = 42) (hL : L = 4) (hP : P = 34)
    (hRooms : R - ((R - P) / L) * L = P) :
  C = 17 :=
by
  sorry

end paula_cans_used_l320_320963


namespace difference_max_min_abs_diff_l320_320160

theorem difference_max_min_abs_diff (a b : ℕ) (h_lcm : Nat.lcm a b = 20) (h_gcd : Nat.gcd a b = 1) : 
  (∃ a b : ℕ, Nat.lcm a b = 20 ∧ Nat.gcd a b = 1) → 
  (let diffs := {abs (a - b) | ∃ (a b : ℕ), Nat.lcm a b = 20 ∧ Nat.gcd a b = 1} in 
    (Set.max diffs - Set.min diffs) = 38) :=
begin
  sorry
end

end difference_max_min_abs_diff_l320_320160


namespace find_Minchos_chocolate_l320_320556

variable (M : ℕ)  -- Define M as a natural number

-- Define the conditions as Lean hypotheses
def TaeminChocolate := 5 * M
def KibumChocolate := 3 * M
def TotalChocolate := TaeminChocolate M + KibumChocolate M

theorem find_Minchos_chocolate (h : TotalChocolate M = 160) : M = 20 :=
by
  sorry

end find_Minchos_chocolate_l320_320556


namespace find_number_l320_320178

theorem find_number (n : ℤ) 
  (h : (69842 * 69842 - n * n) / (69842 - n) = 100000) : 
  n = 30158 :=
sorry

end find_number_l320_320178


namespace isos_trapezium_coloring_l320_320024

theorem isos_trapezium_coloring {n m k : ℕ} 
  (h_n_pos : n > 0) 
  (h_m_cond : m ≥ n^2 - n + 1) 
  (h_color : ∀ (coloring : Fin (n * m) → Fin k), 
            ∃ (A B C D : Fin (n * m)), 
            A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧ 
            same_color coloring A B ∧ same_color coloring B C ∧ 
            same_color coloring C D ∧ same_color coloring D A ∧ 
            is_isosceles_trapezium A B C D) : 
  k ≤ n - 1 :=
sorry

/-- Helper functions and definitions (e.g., same_color, is_isosceles_trapezium) need to be defined for the theorem to be complete. --/

end isos_trapezium_coloring_l320_320024


namespace initial_average_weight_l320_320119

theorem initial_average_weight (A : ℝ) : 
  (6 * A + 145) / 7 = 151 → 
  A = 152 := 
by
  intro h,
  have eq1 : 6 * A + 145 = 7 * 151 := eq_of_div_eq_div (by simp at h; exact h),
  rw mul_comm at eq1,
  linarith,
  sorry

end initial_average_weight_l320_320119


namespace sum_sin_squares_deg_6_to_174_l320_320234

theorem sum_sin_squares_deg_6_to_174 : 
  ∑ k in Finset.range 30, Real.sin (6 * k * (Real.pi / 180)) ^ 2 = 15.5 := by
    sorry

end sum_sin_squares_deg_6_to_174_l320_320234


namespace floor_div_of_M_l320_320779

open BigOperators

theorem floor_div_of_M {M : ℕ} 
  (h : ∑ k in finset.range(8) (3 + k).fact * (22 - (3 + k)).fact = 21! * M) : 
  M = 95290 → floor (M / 100) = 952 :=
by
  sorry

end floor_div_of_M_l320_320779


namespace no_primes_in_sequence_l320_320646

open Nat

def p : ℕ := 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47 * 53 * 59 * 61

theorem no_primes_in_sequence :
  ∀ (n : ℕ), 2 ≤ n ∧ n ≤ 59 → ¬ Prime (p + n) :=
by
  intros n hn
  have h_prime_divisors : ∀ k ≤ 61, Prime k → k ∣ p :=
  -- Proof that each prime number k ≤ 61 divides p
    sorry
  intro h_prime
  have h_divisor : ∃ k, Prime k ∧ k ≤ 61 ∧ k ∣ (p + n),
  -- Proof by divisibility: Showing there exists a prime k ≤ 61 dividing (p + n)
    sorry
  cases h_divisor with k hk
  have := h_prime_divisors k hk.2.2 hk.1
  have h_contradiction : k ∣ n,
  -- Using properties of divisors, show k must divide n leading to contradiction
    sorry
  exact nat.not_dvd_of_pos_of_lt (lt_of_le_of_lt hk.2.2 (le_of_lt (lt_by_cases n hk.2.1)).resolve_left h_prime)

end no_primes_in_sequence_l320_320646


namespace sheila_weekly_earnings_l320_320554

-- Definitions for conditions
def hours_per_day_on_MWF : ℕ := 8
def days_worked_on_MWF : ℕ := 3
def hours_per_day_on_TT : ℕ := 6
def days_worked_on_TT : ℕ := 2
def hourly_rate : ℕ := 10

-- Total weekly hours worked
def total_weekly_hours : ℕ :=
  (hours_per_day_on_MWF * days_worked_on_MWF) + (hours_per_day_on_TT * days_worked_on_TT)

-- Total weekly earnings
def weekly_earnings : ℕ :=
  total_weekly_hours * hourly_rate

-- Lean statement for the proof
theorem sheila_weekly_earnings : weekly_earnings = 360 :=
  sorry

end sheila_weekly_earnings_l320_320554


namespace hypotenuse_length_l320_320869

theorem hypotenuse_length (ABC : Triangle) (right_angle_C : isRightTriangle ABC C)
  (CD_height : DroppedHeight C D AB) (projection_BD : Projection BD BC l)
  (projection_AD : Projection AD AC m) 
  (AB_hypotenuse : length_hypotenuse ABC = hypotenuse_AB) :
  let AB := (l^(2/3) + m^(2/3)) * (sqrt (l^(2/3) + m^(2/3))) in
  length AB = AB_hypotenuse :=
by
  sorry

end hypotenuse_length_l320_320869


namespace mashas_numbers_l320_320499

def is_even (n : ℕ) : Prop := n % 2 = 0

def problem_statement (a b : ℕ) : Prop :=
  a ≠ b ∧ a > 11 ∧ b > 11 ∧ is_even a ∧ a + b = 28
  
theorem mashas_numbers : ∃ (a b : ℕ), problem_statement a b :=
by
  use 12
  use 16
  unfold problem_statement
  split
  -- a ≠ b
  exact dec_trivial
  split
  -- a > 11
  exact dec_trivial
  split
  -- b > 11
  exact dec_trivial
  split
  -- is_even a
  exact dec_trivial
  -- a + b = 28
  exact dec_trivial

end mashas_numbers_l320_320499


namespace triangle_interior_angle_l320_320610

-- Define the given values and equations
variables (x : ℝ) 
def arc_DE := x + 80
def arc_EF := 2 * x + 30
def arc_FD := 3 * x - 25

-- The main proof statement
theorem triangle_interior_angle :
  arc_DE x + arc_EF x + arc_FD x = 360 →
  0.5 * (arc_EF x) = 60.83 :=
by sorry

end triangle_interior_angle_l320_320610


namespace find_number_l320_320135

theorem find_number (n : ℝ) : (4 / 3) * n = 48 → n = 36 :=
by
  intro h
  have h1 : n = 48 * (3 / 4) := by
    sorry
  exact h1

end find_number_l320_320135


namespace shortest_altitude_l320_320584

theorem shortest_altitude (a b c : ℝ) (ha : a = 13) (hb : b = 14) (hc : c = 15) : 
  ∃ h, h = 11.2 ∧ h = 2 * (sqrt (( (a+b+c)/2) * ( (a+b+c)/2 - a) * ( (a+b+c)/2 - b) * ( (a+b+c)/2 - c))) / c :=
by sorry

end shortest_altitude_l320_320584


namespace reflection_on_midpoint_line_l320_320445

theorem reflection_on_midpoint_line
    (ABC : Triangle)
    (acute_ABC : Acute ABC)
    (K : Point)
    (H : Point)
    (H_is_orthocenter : is_orthocenter H ABC)
    (O : Point)
    (O_is_circumcenter : is_circumcenter O ABC)
    (KOH_is_acute : Acute (Triangle.mk K O H))
    (P : Point)
    (P_is_circumcenter : is_circumcenter P (Triangle.mk K O H))
    (Q : Point)
    (Q_is_reflection_of_P : Q = reflection_over_line P HO)
    (M_AB : Point)
    (M_AC : Point)
    (M_AB_is_midpoint : is_midpoint M_AB (line.mk A B))
    (M_AC_is_midpoint : is_midpoint M_AC (line.mk A C)) :
    lies_on Q (line.mk M_AB M_AC) := 
sorry

end reflection_on_midpoint_line_l320_320445


namespace red_black_separation_l320_320341

-- Define the finite sets of red and black points in a plane
variable (Red : Set Point) (Black : Set Point)
variable [Fintype Red] [Fintype Black]

-- Define the condition: for any 4 points, there exists a line separating the red and black points
def separable_four_points_condition (R B : Set Point) : Prop :=
  ∀ (p1 p2 ∈ R) (q1 q2 ∈ B), ∃ l : Line, (∀ p ∈ {p1, p2}, l ∥ p) ∧ (∀ q ∈ {q1,q2}, ¬ l ∥ q)

-- Define the problem: prove there exists a line that separates all red points from all black points
theorem red_black_separation
  (h : separable_four_points_condition Red Black) :
  ∃ l : Line, ∀ p ∈ Red, ∀ q ∈ Black, l ∥ p ∨ ¬ l ∥ q :=
sorry

end red_black_separation_l320_320341


namespace valid_odd_and_increasing_functions_l320_320627

   def is_odd_function (f : ℝ → ℝ) : Prop :=
     ∀ x, f (-x) = -f (x)

   def is_increasing_function (f : ℝ → ℝ) : Prop :=
     ∀ x y, x < y → f (x) < f (y)

   noncomputable def f1 (x : ℝ) : ℝ := 3 * x^2
   noncomputable def f2 (x : ℝ) : ℝ := 6 * x
   noncomputable def f3 (x : ℝ) : ℝ := x * abs x
   noncomputable def f4 (x : ℝ) : ℝ := x + 1 / x

   theorem valid_odd_and_increasing_functions :
     (is_odd_function f2 ∧ is_increasing_function f2) ∧
     (is_odd_function f3 ∧ is_increasing_function f3) :=
   by
     sorry -- Proof goes here
   
end valid_odd_and_increasing_functions_l320_320627


namespace remaining_value_subtract_70_percent_from_4500_l320_320849

theorem remaining_value_subtract_70_percent_from_4500 (num : ℝ) 
  (h : 0.36 * num = 2376) : 4500 - 0.70 * num = -120 :=
by
  sorry

end remaining_value_subtract_70_percent_from_4500_l320_320849


namespace complement_set_l320_320812

open Set

theorem complement_set (U M : Set ℕ) (hU : U = {1, 2, 3, 4, 5, 6}) (hM : M = {1, 2, 4}) :
  compl M ∩ U = {3, 5, 6} := 
by
  rw [compl, hU, hM]
  sorry

end complement_set_l320_320812


namespace num_odd_divisors_lt_100_l320_320396

theorem num_odd_divisors_lt_100 : (∃! n : ℕ, n = 9 ∧ ∀ m : ℕ, m < 100 → (odd_divisors m ↔ m = 1^2 ∨ m = 2^2 ∨ m = 3^2 ∨ m = 4^2 ∨ m = 5^2 ∨ m = 6^2 ∨ m = 7^2 ∨ m = 8^2 ∨ m = 9^2)) := sorry

-- Definitions for clarity
def odd_divisors (m : ℕ) : Prop := (finset.range (m + 1)).filter (λ d, m % d = 0).card % 2 = 1

end num_odd_divisors_lt_100_l320_320396


namespace determine_numbers_l320_320470

theorem determine_numbers (a b : ℕ) (S : ℕ) (h1 : a ≠ b) (h2 : a > 11) (h3 : b > 11)
  (h4 : S = a + b) (h5 : (∀ (x y : ℕ), x + y = S → x ≠ y → (x = a ∧ y = b) ∨ (x = b ∧ y = a)) = false)
  (h6 : a % 2 = 0 ∨ b % 2 = 0) : 
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) := 
  sorry

end determine_numbers_l320_320470


namespace exists_point_D_l320_320896

theorem exists_point_D
  {A B C D X Y : Type*}
  [triangle A B C] 
  [acute_triangle A B C]
  (hAB_AC : length B C > length A C)
  (h_interior_XY : interior_XY A B C X Y)
  (h_cyclic_BCXY : cyclic_quad B C X Y)
  (h_angle_cond : ∠ AXB - ∠ ACB = ∠ CYA - ∠ CBA) :
  ∃ D : Type*, ∀ X Y, (interior_XY A B C X Y) →
  [cyclic_quad B C X Y] →
  (∠ AXB - ∠ ACB = ∠ CYA - ∠ CBA) →
  (line_passing_through D X Y) := 
begin
  sorry
end

end exists_point_D_l320_320896


namespace find_f_neg_a_l320_320774

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

variable (f : ℝ → ℝ)
variable (a : ℝ)

-- Conditions
hypothesis odd_f : is_odd_function f
hypothesis f_a_eq_11 : f a = 11

-- Proof statement
theorem find_f_neg_a : f (-a) = -11 :=
by
  sorry

end find_f_neg_a_l320_320774


namespace cos_OA_BC_is_zero_l320_320881

-- Define the variables and conditions for the problem
variable (O A B C : ℝ)
variable (OB OC : ℝ)
variable (OA_angle_OB : ℝ)
variable (OA_angle_OC : ℝ)
variable [normed_space ℝ (euclidean_space ℝ (fin 3))]
variable [OB_eq_OC : OB = OC] 
variable [angle_OA_OB : OA_angle_OB = π / 3]
variable [angle_OA_OC : OA_angle_OC = π / 3]

-- State the theorem to prove the cosine value is zero
theorem cos_OA_BC_is_zero (O A B C : ℝ) (OB OC : ℝ) (OA_angle_OB : ℝ) (OA_angle_OC : ℝ) 
  [normed_space ℝ (euclidean_space ℝ (fin 3))] [OB_eq_OC : OB = OC] 
  [angle_OA_OB : OA_angle_OB = π / 3] [angle_OA_OC : OA_angle_OC = π / 3] :
  cos (inner (O - A) (B - C)) = 0 :=
sorry

end cos_OA_BC_is_zero_l320_320881


namespace tangent_line_with_minimum_slope_l320_320747

def f (x : ℝ) : ℝ := (1/3) * x^3 - x^2 + 3 * x - (1/3)

theorem tangent_line_with_minimum_slope :
  ∃ t : ℝ, tangent_line_with_slope_at f t 1 2 ∧ 2 * (x : ℝ) = y : ℝ := 0 :=
by
  sorry

end tangent_line_with_minimum_slope_l320_320747


namespace cube_tetrahedron_height_l320_320194

/--
Given a unit cube where a corner is chopped off by a plane running through the three adjacent vertices,
prove that the height of the resulting tetrahedron, when the freshly-cut face is placed flat on a table, is 
\( \frac{2\sqrt{3}}{3} \).
-/
theorem cube_tetrahedron_height :
  let side_length : ℝ := 1
  let chopping_distance : ℝ := 1
  let cut_face_length : ℝ := √2
  let cut_face_area : ℝ := (√3 / 4) * (cut_face_length ^ 2)
  ∀ h : ℝ, (1 / 3) * cut_face_area * h = (1 / 6) →
  h = 2 * (√3 / 3) := by
  sorry

end cube_tetrahedron_height_l320_320194


namespace distinct_real_roots_form_geometric_progression_eq_170_l320_320289

theorem distinct_real_roots_form_geometric_progression_eq_170 
  (a : ℝ) :
  (∃ (u : ℝ) (v : ℝ) (hu : u ≠ 0) (hv : v ≠ 0) (hv1 : |v| ≠ 1), 
  (16 * u^12 + (2 * a + 17) * u^6 * v^3 - a * u^9 * v - a * u^3 * v^9 + 16 = 0)) 
  → a = 170 :=
by sorry

end distinct_real_roots_form_geometric_progression_eq_170_l320_320289


namespace perpendicular_lines_l320_320407

theorem perpendicular_lines (a : ℝ) : 
  (∀ x y : ℝ, (x + a * y - a = 0) → (a * x - (2 * a - 3) * y - 1 = 0) → 
    (∀ x y : ℝ, ( -1 / a ) * ( -a / (2 * a - 3)) = 1 )) → a = 3 := 
by
  sorry

end perpendicular_lines_l320_320407


namespace valid_distribution_count_l320_320721

def student : Type := {A, B, C, D}
def class : Type := {class1, class2}

def valid_distribution (d : student → class) : Prop :=
  d A ≠ d B ∧ -- A and B cannot be in the same class
  (∃ s : class, ∃ t : class, s ≠ t ∧ -- Each class has at least one student
    (∃ st1, ∃ st2, d st1 = s ∧ d st2 = t))

def number_of_valid_distributions : ℕ := 
  if h : ∃ (d : student → class), valid_distribution d then
    6 -- Using the solved correct number of distributions from the problem
  else
    0
  
theorem valid_distribution_count : number_of_valid_distributions = 6 :=
  sorry

end valid_distribution_count_l320_320721


namespace titu_andreescu_l320_320650

theorem titu_andreescu (n : ℕ) (h : 0 < n) :
  ∃ (k : Fin (n + 1) → ℕ), (∀ i, 1 < k i) ∧
  (∀ i j, i ≠ j → Nat.gcd (k i) (k j) = 1) ∧
  (∃ a b : ℕ, b = a + 1 ∧ ∏ i in Finset.range (n + 1), k i - 1 = a * b) :=
sorry

end titu_andreescu_l320_320650


namespace simplify_and_evaluate_expr_l320_320074

theorem simplify_and_evaluate_expr (a b : ℕ) (h₁ : a = 2) (h₂ : b = 2023) : 
  (a + b)^2 + b * (a - b) - 3 * a * b = 4 := by
  sorry

end simplify_and_evaluate_expr_l320_320074


namespace remainder_123456789012_l320_320316

theorem remainder_123456789012 :
  let N := 123456789012 in
  let M := 180 in
  (M = 4 * 9 * 5) →
  (N ≡ 0 [MOD 4]) →
  (N ≡ 2 [MOD 5]) →
  (N ≡ 3 [MOD 9]) →
  N ≡ 12 [MOD 180] := by
  sorry

end remainder_123456789012_l320_320316


namespace equal_segments_BE_CF_l320_320905

-- Define triangle points and conditions
variables {A B C D E F : Point}
variable [triangle: Triangle A B C]

-- Given conditions
variables (AB_less_AC : length B A < length C A)
variables (angle_bisector_AD : AngleBisector (Angle A B C) D)
variables (perpendicular_BC_bisector : PerpendicularBisector B C D)

-- Define points E and F with given perpendicular conditions
variables (E_on_AB : OnLineSegment A B E)
variables (F_on_AC : OnLineSegment A C F)
variables (DE_perpendicular_AB : Perpendicular D E A B)
variables (DF_perpendicular_AC : Perpendicular D F A C)

-- Required to prove BE == CF
theorem equal_segments_BE_CF : length B E = length C F :=
sorry

end equal_segments_BE_CF_l320_320905


namespace eighth_prime_is_19_l320_320717

def prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

noncomputable def nth_prime (n : ℕ) : ℕ :=
  (0, List.filter prime [1..]).2.nth (n - 1) |>.get_or_else 0

theorem eighth_prime_is_19 : nth_prime 8 = 19 := by
  sorry

end eighth_prime_is_19_l320_320717


namespace log_constant_l320_320226

theorem log_constant (y : ℝ) (h1 : log 8 5 = y) : 
  ∃ (m : ℝ), log 2 125 = m * y ∧ m = 9 :=
by
  sorry

end log_constant_l320_320226


namespace area_of_ellipse_irrational_l320_320410

theorem area_of_ellipse_irrational
  (a b : ℚ) : 
  (∃ (p q r s : ℤ), q ≠ 0 ∧ s ≠ 0 ∧ a = p / q ∧ b = r / s) → 
  ¬ (∃ (k : ℚ), ∃ (π' : ℝ), π' = real.pi ∧ π' * (p.to_real * r.to_real) / (q.to_real * s.to_real) = k.to_real) :=
sorry

end area_of_ellipse_irrational_l320_320410


namespace star_five_three_l320_320287

def star (a b : ℤ) : ℤ := 4 * a - 2 * b

theorem star_five_three : star 5 3 = 14 := by
  sorry

end star_five_three_l320_320287


namespace closest_point_on_graph_l320_320945

theorem closest_point_on_graph (x y : ℝ) (h1 : x > 0) (h2 : y = x + 1/x) :
  (x = 1/real.root 4 2) ∧ (y = (1 + real.sqrt 2) / real.root 4 2) :=
sorry

end closest_point_on_graph_l320_320945


namespace central_angle_of_sector_l320_320864

theorem central_angle_of_sector (r A θ : ℝ) (hr : r = 2) (hA : A = 4) :
  θ = 2 :=
by
  sorry

end central_angle_of_sector_l320_320864


namespace lines_parallel_under_parallel_lighting_l320_320965

open_locale affine

noncomputable theory

variables {V : Type*} [inner_product_space ℝ V]
variables {P : Type*} [metric_space P] [normed_add_torsor V P]

/-- This theorem states that under parallel lighting, the lines that connect the shadows of all points
on the first and second projection planes are parallel to each other. -/
theorem lines_parallel_under_parallel_lighting
  (lighting_parallel : ∀ (p q : P), p ≠ q → (∃ (u : V), u ≠ 0 ∧ ∀ (r : ℝ), ∃ (x y : P), x ≠ y ∧ 
    line_map x y r = line_map p q r ∧ q -ᵥ p = (y -ᵥ x)) )
  (A B C : P) 
  (A1 A2 B1 B2 C1 C2 : P)
  (shadowA1 : shadow_map A A1)
  (shadowA2 : shadow_map A A2)
  (shadowB1 : shadow_map B B1)
  (shadowB2 : shadow_map B B2)
  (shadowC1 : shadow_map C C1)
  (shadowC2 : shadow_map C C2) :
  (co_line (A1 -ᵥ A2) (B1 -ᵥ B2) ∧ co_line (B1 -ᵥ B2) (C1 -ᵥ C2)) :=
sorry

end lines_parallel_under_parallel_lighting_l320_320965


namespace int_power_sum_is_integer_l320_320888

def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n

theorem int_power_sum_is_integer {x : ℝ} (h : is_integer (x + 1/x)) (n : ℤ) : is_integer (x^n + 1/x^n) :=
by
  sorry

end int_power_sum_is_integer_l320_320888


namespace sum_of_squares_of_sines_l320_320246

theorem sum_of_squares_of_sines : 
  (\sum n in finset.range 1 30, real.sin (6 * n : ℝ) ^ 2) = 31 / 2 :=
begin
  sorry
end

end sum_of_squares_of_sines_l320_320246


namespace fourth_person_after_switch_l320_320929

universe u

def person: Type u := String

def original_order: List person :=
  ["Naeun", "Leejun", "Yoonseo", "Seohyun", "Soyul"]

def switch (l: List person) (i j : ℕ): List person :=
  if h₁ : i < l.length ∧ j < l.length then
    (List.mapWithIndex (λ k x, if k = i then l.get ⟨j, h₁.right⟩ else if k = j then l.get ⟨i, h₁.left⟩ else x) l)
  else l

theorem fourth_person_after_switch :
  switch original_order 2 3 !! 3 = "Yoonseo" :=
by
  sorry

end fourth_person_after_switch_l320_320929


namespace xy_value_l320_320030

variable (x y : ℝ)

-- Definitions for points A, B, and C and the condition that C is the midpoint of A and B
def A : ℝ × ℝ := (1, 8)
def B : ℝ × ℝ := (x, y)
def C : ℝ × ℝ := (3, 5)

-- Condition that C is the midpoint of A and B
def isMidpoint (A B C : ℝ × ℝ) : Prop :=
  C = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Statement of the problem: Prove that xy = 10 given the conditions
theorem xy_value : isMidpoint A B C → x * y = 10 := by
  sorry

end xy_value_l320_320030


namespace proof_PA_eq_PK_l320_320430

noncomputable def angle_greater_than_90 (ABC : Triangle) : Prop :=
  ABC.angleB > 90

noncomputable def tangents_meet (ABC : Triangle) (A B P : Point) (circle_ABC : Circle)
  (tangent_A : TangentLine circle_ABC A)
  (tangent_B : TangentLine circle_ABC B) : Prop :=
  tangent_A.meet tangent_B = P

noncomputable def perpendicular_meet (B C K : Point) (line_BC : Line) (line_AC : Line) : Prop :=
  line_BC.perpendicularThrough B ∧ meets line_AC K

theorem proof_PA_eq_PK (ABC : Triangle) (A B C P K : Point) (circle_ABC : Circle)
  (tangent_A : TangentLine circle_ABC A) (tangent_B : TangentLine circle_ABC B)
  (line_BC : Line) (line_AC : Line) :
  angle_greater_than_90 ABC →
  tangents_meet ABC A B P circle_ABC tangent_A tangent_B →
  perpendicular_meet B C K line_BC line_AC →
  distance P A = distance P K :=
by
  sorry

end proof_PA_eq_PK_l320_320430


namespace region_area_ratio_l320_320552

/-
Regions I, II, III, and IV are each bounded by squares.

The perimeter of region I is 16 units,
the perimeter of region II is 32 units,
the perimeter of region III is 48 units,
the perimeter of region IV is twice that of region III.

What is the ratio of the area of region I to the area of region IV?
-/

theorem region_area_ratio :
  (let side_length_I := 16 / 4,
       side_length_IV := 2 * (48 / 4)
   in (side_length_I * side_length_I) / (side_length_IV * side_length_IV) = 1 / 36) := 
by
  sorry

end region_area_ratio_l320_320552


namespace geometric_sequence_a3_l320_320880

theorem geometric_sequence_a3 (a : ℕ → ℝ) (q : ℝ) (h1 : a 4 = a 1 * q ^ 3) (h2 : a 2 = a 1 * q) (h3 : a 5 = a 1 * q ^ 4) 
    (h4 : a 4 - a 2 = 6) (h5 : a 5 - a 1 = 15) : a 3 = 4 ∨ a 3 = -4 :=
by
  sorry

end geometric_sequence_a3_l320_320880


namespace evaluate_expression_l320_320729

variable (b : ℝ)

theorem evaluate_expression : ( ( (b^(16/8))^(1/4) )^3 * ( (b^(16/4))^(1/8) )^3 ) = b^3 := by
  sorry

end evaluate_expression_l320_320729


namespace masha_numbers_l320_320519

theorem masha_numbers (a b : ℕ) (S : ℕ) (h1 : a ≠ b) (h2 : a > 11) (h3 : b > 11) (h4 : S = a + b) 
    (h5 : (∀ x y : ℕ, x + y = S → x = a ∨ y = a → abs x - y = a) ∧ (even a ∨ even b)) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) := 
by sorry

end masha_numbers_l320_320519


namespace ellipse_properties_l320_320773

noncomputable def ellipse_eq : Prop :=
  let a : ℝ := 2
  let b : ℝ := sqrt 3
  ∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1 ↔ (4 * x^2) + (3 * y^2) = 12

noncomputable def slope_MN : Prop :=
  ∀ (k₁ : ℝ) (r : ℝ), 0 < r ∧ r < 3/2 ∧ (8 * k₁ - 12) ≠ 0 → 
    ∀ (x₁ y₁ x₂ y₂ l₁ l₂ : ℝ),
    let k₂ := -k₁ in
    let equation := 4 * (x₁^2) * (k₁^2 + 3/4) + 3 * (y₁^2) = 12 in
    let equation := 4 * (x₂^2) * (k₂^2 + 3/4) + 3 * (y₂^2) = 12 in
    let line_eq1 := y₁ = k₁ * x₁ - k₁ + 3/2 in
    let line_eq2 := y₂ = k₂ * x₂ - k₂ + 3/2 in
    (y₁ - y₂) / (x₁ - x₂) = 1/2

theorem ellipse_properties : ellipse_eq ∧ slope_MN := by
  constructor
  { -- Proof for ellipse_eq
    sorry, 
  },
  { -- Proof for slope_MN
    sorry
  }

end ellipse_properties_l320_320773


namespace paper_needed_l320_320970

theorem paper_needed : 26 + 26 + 10 = 62 := by
  sorry

end paper_needed_l320_320970


namespace masha_numbers_unique_l320_320480

def natural_numbers : Set ℕ := {n | n > 11}

theorem masha_numbers_unique (a b : ℕ) (ha : a ∈ natural_numbers) (hb : b ∈ natural_numbers) (hne : a ≠ b)
  (hs_equals : ∃ S, S = a + b)
  (sasha_initially_uncertain : ∀ (a b : ℕ), a ∈ natural_numbers → b ∈ natural_numbers → a ≠ b → (a + b = S) → ¬ (Sasha_can_determine_initially a b S))
  (masha_hint : ∃ (a_even : ℕ), a_even ∈ natural_numbers ∧ (a_even % 2 = 0) ∧ (a_even = a ∨ a_even = b))
  (sasha_then_confident : ∀ (a b : ℕ), a ∈ natural_numbers → b ∈ natural_numbers → a ≠ b → (a + b = S) → (a_even = a ∨ a_even = b) → Sasha_can_determine_confidently a b S) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) := by
  sorry

end masha_numbers_unique_l320_320480


namespace sin_squared_sum_l320_320272

theorem sin_squared_sum : 
  ∑ k in finset.range 30 \ {0}, (sin (6 * k + 6) * (real.pi / 180))^2 = 15 :=
sorry

end sin_squared_sum_l320_320272


namespace cookies_in_jar_l320_320121

theorem cookies_in_jar (C : ℕ) (h : C - 1 = (C + 5) / 2) : C = 7 :=
by
  -- Proof goes here
  sorry

end cookies_in_jar_l320_320121


namespace parabola_standard_equation_l320_320318

theorem parabola_standard_equation :
  ∃ p1 p2 : ℝ, p1 > 0 ∧ p2 > 0 ∧ (y^2 = 2 * p1 * x ∨ x^2 = 2 * p2 * y) ∧ ((6, 4) ∈ {(x, y) | y^2 = 2 * p1 * x} ∨ (6, 4) ∈ {(x, y) | x^2 = 2 * p2 * y}) := 
  sorry

end parabola_standard_equation_l320_320318


namespace difference_AB_l320_320230

def A : ℕ :=
  (List.range' 1 40).filterMap (λ n => if n % 2 = 0 then some (n * (n + 1)) else none)
  .sum + 41

def B : ℕ :=
  1 + (List.range' 2 40).filterMap (λ i => if i % 2 = 1 then some (i * (i + 1)) else none)
  .sum + 40 * 41

theorem difference_AB : |A - B| = 380 := by
  sorry

end difference_AB_l320_320230


namespace king_not_in_right_mind_l320_320015

variable (RightMind : Prop → Prop)
variable (Q K : Prop)

-- Given condition: The Queen of Spades thinks that the King of Spades thinks she is not in her right mind.
axiom (H : Q → (K → ¬ RightMind Q))

-- Prove: The King of Spades is not in his right mind.
theorem king_not_in_right_mind : ¬ RightMind K := by
  sorry

end king_not_in_right_mind_l320_320015


namespace fifth_term_constant_l320_320840

theorem fifth_term_constant (n : ℕ) (h : 0 < n) :
  let T : ℕ → ℕ → ℤ := λ n r, ↑(Nat.choose n r * (2 * n - 3 * r).choose 0)
  let fifth_term_is_constant := T n 4
  fifth_term_is_constant = 1 → n = 6 :=
sorry

end fifth_term_constant_l320_320840


namespace mashas_numbers_l320_320496

def is_even (n : ℕ) : Prop := n % 2 = 0

def problem_statement (a b : ℕ) : Prop :=
  a ≠ b ∧ a > 11 ∧ b > 11 ∧ is_even a ∧ a + b = 28
  
theorem mashas_numbers : ∃ (a b : ℕ), problem_statement a b :=
by
  use 12
  use 16
  unfold problem_statement
  split
  -- a ≠ b
  exact dec_trivial
  split
  -- a > 11
  exact dec_trivial
  split
  -- b > 11
  exact dec_trivial
  split
  -- is_even a
  exact dec_trivial
  -- a + b = 28
  exact dec_trivial

end mashas_numbers_l320_320496


namespace g_720_l320_320036

-- Define the function and conditions
def g : ℕ → ℕ 
axiom g_mul {x y : ℕ} (hx : x > 0) (hy : y > 0) : g(x * y) = g x + g y
axiom g_12 : g 12 = 18
axiom g_48 : g 48 = 26

-- State the theorem to be proven
theorem g_720 : g 720 = 46 :=
by
  -- Proof placeholder
  sorry

end g_720_l320_320036


namespace masha_numbers_unique_l320_320481

def natural_numbers : Set ℕ := {n | n > 11}

theorem masha_numbers_unique (a b : ℕ) (ha : a ∈ natural_numbers) (hb : b ∈ natural_numbers) (hne : a ≠ b)
  (hs_equals : ∃ S, S = a + b)
  (sasha_initially_uncertain : ∀ (a b : ℕ), a ∈ natural_numbers → b ∈ natural_numbers → a ≠ b → (a + b = S) → ¬ (Sasha_can_determine_initially a b S))
  (masha_hint : ∃ (a_even : ℕ), a_even ∈ natural_numbers ∧ (a_even % 2 = 0) ∧ (a_even = a ∨ a_even = b))
  (sasha_then_confident : ∀ (a b : ℕ), a ∈ natural_numbers → b ∈ natural_numbers → a ≠ b → (a + b = S) → (a_even = a ∨ a_even = b) → Sasha_can_determine_confidently a b S) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) := by
  sorry

end masha_numbers_unique_l320_320481


namespace value_of_f_neg_5_l320_320373

theorem value_of_f_neg_5 (a b : ℝ) (h : f 5 = 7) : f (-5) = -3 :=
by
  let f := λ x : ℝ, a * Real.sin x + b * Real.tan x + 2
  have h1 : f 5 = 7 := h
  sorry

end value_of_f_neg_5_l320_320373


namespace eccentricity_of_ellipse_l320_320854
-- Import the Mathlib library for mathematical tools and structures

-- Define the condition for the ellipse and the arithmetic sequence
variables {a b c : ℝ} (h1 : a > b) (h2 : b > 0) (h3 : 2 * b = a + c) (h4 : b^2 = a^2 - c^2)

-- State the theorem to prove
theorem eccentricity_of_ellipse : ∃ e : ℝ, e = 3 / 5 :=
by
  -- Proof would go here
  sorry

end eccentricity_of_ellipse_l320_320854


namespace probability_decreasing_function_l320_320326

-- Definitions of the conditions and the function

def a_values := [-2, 0, 1, 2, 3]
def b_values := [3, 5]

def f(a : ℤ) (b : ℤ) (x : ℝ) : ℝ := (a^2 - 2) * Real.exp x + b

def is_decreasing (a : ℤ) : Prop := a^2 - 2 < 0

-- Lean proof statement
theorem probability_decreasing_function : 
  (let favorable_a := (a_values.filter (λ a, is_decreasing a)).length,
       total_a := a_values.length,
       total_b := b_values.length in
   (favorable_a * total_b) / (total_a * total_b) = 2 / 5) := sorry

end probability_decreasing_function_l320_320326


namespace seashells_total_l320_320054

theorem seashells_total :
  let monday := 5
  let tuesday := 7 - 3
  let wednesday := (2 * monday) / 2
  let thursday := 3 * 7
  monday + tuesday + wednesday + thursday = 35 :=
by
  sorry

end seashells_total_l320_320054


namespace problem_I_number_of_zeros_problem_II_inequality_l320_320382

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x * Real.exp 1 - 1

theorem problem_I_number_of_zeros : 
  ∃! (x1 x2 : ℝ), f x1 = 0 ∧ f x2 = 0 ∧ x1 ≠ x2 := 
sorry

theorem problem_II_inequality (a : ℝ) (h_a : a ≤ 0) (x : ℝ) (h_x : x ≥ 1) : 
  f x ≥ a * Real.log x - 1 := 
sorry

end problem_I_number_of_zeros_problem_II_inequality_l320_320382


namespace solution_set_l320_320288

-- Define determinant operation on 2x2 matrices
def determinant (a b c d : ℝ) := a * d - b * c

-- Define the condition inequality
def condition (x : ℝ) : Prop :=
  determinant x 3 (-x) x < determinant 2 0 1 2

-- Prove that the solution to the condition is -4 < x < 1
theorem solution_set : {x : ℝ | condition x} = {x : ℝ | -4 < x ∧ x < 1} :=
by
  sorry

end solution_set_l320_320288


namespace joe_investment_interest_rate_l320_320890

noncomputable def compound_interest_rate (P : ℝ) (r : ℝ) (years : ℕ) : ℝ :=
  P * (1 + r) ^ years

theorem joe_investment_interest_rate :
  ∃ r : ℝ, ∃ P : ℝ, 
    (compound_interest_rate P r 3 = 310) ∧
    (compound_interest_rate P r 8 = 410) ∧
    (r ≈ 0.05868) :=
begin
  use (310 / (1 + 0.05868) ^ 3), -- P
  use 0.05868,
  split,
  { -- First condition 310 = P * (1 + 0.05868) ^ 3
    sorry },
  split,
  { -- Second condition 410 = P * (1 + 0.05868) ^ 8
    sorry },
  { -- Rate r ≈ 0.05868
    sorry }
end

end joe_investment_interest_rate_l320_320890


namespace seed_mixture_Y_fescue_l320_320069

-- Define the conditions
def seed_mixture_X_ryegrass := 0.4
def seed_mixture_X_bluegrass := 0.6

def seed_mixture_Y_ryegrass := 0.25

def final_mixture_ryegrass := 0.3

def weight_X_in_mixture := 1 / 3
def weight_Y_in_mixture := 2 / 3

-- Define the question
def fescue_percentage_in_Y (F : ℝ) :=
  seed_mixture_Y_ryegrass + F = 1

-- Define the main theorem
theorem seed_mixture_Y_fescue :
  ∀ F : ℝ,
  seed_mixture_X_ryegrass * weight_X_in_mixture +
    seed_mixture_Y_ryegrass * weight_Y_in_mixture = final_mixture_ryegrass →
  fescue_percentage_in_Y F → F = 0.75 :=
begin
  intros F h1 h2,
  calc F = 0.75 : sorry
end

end seed_mixture_Y_fescue_l320_320069


namespace Debby_bottles_l320_320715

theorem Debby_bottles (bottles_per_day : ℕ) (days : ℕ) (total_bottles : ℕ) : 
  bottles_per_day = 109 → 
  days = 74 →
  total_bottles = bottles_per_day * days →
  total_bottles = 8066 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

#eval Debby_bottles 109 74 8066 rfl rfl rfl  -- Example evaluation.

end Debby_bottles_l320_320715


namespace coordinates_of_OZ_l320_320794

noncomputable def z : ℂ := 2 * complex.i / (1 + complex.i)
def Z := (z.re, z.im)
def O := (0, 0)
def OZ := (Z.1 - O.1, Z.2 - O.2)

theorem coordinates_of_OZ : OZ = (1, 1) :=
by
  -- Simplification steps can be added here for the actual proof
  sorry

end coordinates_of_OZ_l320_320794


namespace simple_interest_rate_l320_320163

theorem simple_interest_rate (P : ℝ) (T : ℝ) (R : ℝ) (H : T = 4 ∧ (7 / 6) * P = P + (P * R * T / 100)) :
  R = 4.17 :=
by
  sorry

end simple_interest_rate_l320_320163


namespace find_number_l320_320539

theorem find_number (x : ℝ) (h : x / 3 = 1.005 * 400) : x = 1206 := 
by 
sorry

end find_number_l320_320539


namespace tan_sq_diff_l320_320842

theorem tan_sq_diff (x : ℝ) (h : cos x * cos x * cot x = sin x * sin x) : tan x ^ 6 - tan x ^ 2 = 1 :=
by sorry

end tan_sq_diff_l320_320842


namespace sequence_formula_l320_320807

theorem sequence_formula (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n ≥ 2, a n = 2 * a (n - 1) + 1) :
  ∀ n : ℕ, a n = 2 ^ n - 1 :=
sorry

end sequence_formula_l320_320807


namespace arithmetic_sequence_diff_l320_320782

-- Define the arithmetic sequence properties
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
variable (a : ℕ → ℤ)
variable (h1 : is_arithmetic_sequence a 2)

-- Prove that a_5 - a_2 = 6
theorem arithmetic_sequence_diff : a 5 - a 2 = 6 :=
by sorry

end arithmetic_sequence_diff_l320_320782


namespace gcd_of_105_1001_2436_l320_320147

noncomputable def gcd_problem : ℕ :=
  Nat.gcd (Nat.gcd 105 1001) 2436

theorem gcd_of_105_1001_2436 : gcd_problem = 7 :=
by {
  sorry
}

end gcd_of_105_1001_2436_l320_320147


namespace surface_area_of_given_cylinder_l320_320668

noncomputable def surface_area_of_cylinder (length width : ℝ) : ℝ :=
  let r := (length / (2 * Real.pi))
  let h := width
  2 * Real.pi * r^2 + 2 * Real.pi * r * h

theorem surface_area_of_given_cylinder : 
  surface_area_of_cylinder (4 * Real.pi) 2 = 16 * Real.pi :=
by
  -- Proof will be filled here
  sorry

end surface_area_of_given_cylinder_l320_320668


namespace num_even_divisors_of_8_l320_320824

def factorial (n : Nat) : Nat :=
  match n with
  | 0     => 1
  | Nat.succ n' => Nat.succ n' * factorial n'

-- Define the prime factorization of 8!
def prime_factors_eight_factorial : Nat := 2^7 * 3^2 * 5 * 7

-- Definition of an even divisor of 8!
def is_even_divisor (d : Nat) : Prop :=
  d ∣ prime_factors_eight_factorial ∧ 2 ∣ d

-- Calculation of number of even divisors of 8!
def num_even_divisors_8! : Nat :=
  7 * 3 * 2 * 2

theorem num_even_divisors_of_8! :
  num_even_divisors_8! = 84 :=
sorry

end num_even_divisors_of_8_l320_320824


namespace minimum_value_of_expression_l320_320903

theorem minimum_value_of_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : (x / y + y / z + z / x) + (y / x + z / y + x / z) = 10) :
  ∃ P, (P = (x / y + y / z + z / x) * (y / x + z / y + x / z)) ∧ P = 25 := 
by sorry

end minimum_value_of_expression_l320_320903


namespace shortest_altitude_13_14_15_l320_320587

noncomputable def shortest_altitude_in_triangle (a b c : ℕ) : ℝ :=
  let s := (a + b + c) / 2 in
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c)) in
  min (2 * area / a) (min (2 * area / b) (2 * area / c))

theorem shortest_altitude_13_14_15 : 
  shortest_altitude_in_triangle 13 14 15 = 11.2 := 
sorry

end shortest_altitude_13_14_15_l320_320587


namespace silk_distribution_l320_320220

theorem silk_distribution :
  ∀ (total_silk : ℕ) (required_per_dress : ℕ) (total_dresses : ℕ) (num_friends : ℕ),
  total_silk = 600 →
  required_per_dress = 5 →
  total_dresses = 100 →
  num_friends = 5 →
  let silk_for_dresses := total_dresses * required_per_dress in
  let remaining_silk := total_silk - silk_for_dresses in
  let silk_per_friend := remaining_silk / num_friends in
  silk_per_friend = 20 := 
by
  intros total_silk required_per_dress total_dresses num_friends
         h_total_silk h_required_per_dress h_total_dresses h_num_friends
  let silk_for_dresses := total_dresses * required_per_dress
  let remaining_silk := total_silk - silk_for_dresses
  let silk_per_friend := remaining_silk / num_friends
  sorry

end silk_distribution_l320_320220


namespace probability_of_forming_CHORAL_is_correct_l320_320019

-- Definitions for selecting letters with given probabilities
def probability_select_C_A_L_from_CAMEL : ℚ :=
  1 / 10

def probability_select_H_O_R_from_SHRUB : ℚ :=
  1 / 10

def probability_select_G_from_GLOW : ℚ :=
  1 / 2

-- Calculating the total probability of selecting letters to form "CHORAL"
def probability_form_CHORAL : ℚ :=
  probability_select_C_A_L_from_CAMEL * 
  probability_select_H_O_R_from_SHRUB * 
  probability_select_G_from_GLOW

theorem probability_of_forming_CHORAL_is_correct :
  probability_form_CHORAL = 1 / 200 :=
by
  -- Statement to be proven here
  sorry

end probability_of_forming_CHORAL_is_correct_l320_320019


namespace solve_inequality_l320_320589

theorem solve_inequality (x : ℝ) : -1/3 * x + 1 ≤ -5 → x ≥ 18 := 
  sorry

end solve_inequality_l320_320589


namespace DE_eq_DF_l320_320335

-- Define the given triangle and equilateral triangles
variables {A B C E F : Type}

-- Conditions for triangles
axiom triangle_ABC : Triangle A B C
axiom equilateral_triangle_AFB_on_AB : Equilateral △A F B ∧ IsOutward B A F
axiom equilateral_triangle_ACE_on_AC : Equilateral △A C E ∧ IsOutward C A E

-- Goal to prove DE = DF
theorem DE_eq_DF (A B C D E F : Point)
                 (triangle_ABC : Triangle A B C)
                 (equilateral_triangle_AFB_on_AB : Equilateral △A F B ∧ IsOutward B A F)
                 (equilateral_triangle_ACE_on_AC : Equilateral △A C E ∧ IsOutward C A E) :
  distance D E = distance D F :=
by
  sorry

end DE_eq_DF_l320_320335


namespace closest_point_to_origin_l320_320962

theorem closest_point_to_origin : 
  ∃ x y : ℝ, x > 0 ∧ y = x + 1/x ∧ (x, y) = (1/(2^(1/4)), (1 + 2^(1/2))/(2^(1/4))) :=
by
  sorry

end closest_point_to_origin_l320_320962


namespace triangle_base_length_l320_320749

theorem triangle_base_length
  (area : ℝ)
  (height : ℝ)
  (h_area : area = 36)
  (h_height : height = 8) :
  ∃ (base : ℝ), (area = (base * height) / 2) ∧ base = 9 :=
by
  use 9
  split
  {
    calc
      area 
        = 36 : by rw [h_area]
      ... = (9 * 8) / 2 : by norm_num
  }
  {
    norm_num
  }

end triangle_base_length_l320_320749


namespace rectangle_perimeter_l320_320673

-- Definitions of the conditions
def side_length_square : ℕ := 75  -- side length of the square in mm
def height_sum (x y z : ℕ) : Prop := x + y + z = side_length_square  -- sum of heights of the rectangles

-- Perimeter definition
def perimeter (h : ℕ) (w : ℕ) : ℕ := 2 * (h + w)

-- Statement of the problem
theorem rectangle_perimeter (x y z : ℕ) (h_sum : height_sum x y z)
  (h1 : perimeter x side_length_square = (perimeter y side_length_square + perimeter z side_length_square) / 2)
  : perimeter x side_length_square = 200 := by
  sorry

end rectangle_perimeter_l320_320673


namespace find_odd_prime_satisfying_condition_l320_320315

def permuted_residue_system (p : ℕ) (b : Fin (p-1) → Fin (p-1)) : Prop :=
  ∀ i, (Nat.mod ((i+1)^b i) p) ≠ 0 ∧ Nat.coprime ((i+1)^b i) p

theorem find_odd_prime_satisfying_condition (p : ℕ) (hp_prime : Nat.Prime p) (hp_odd : p % 2 = 1) :
  (∃ b : Fin (p-1) → Fin (p-1), bijective b ∧ permuted_residue_system p b) → p = 3 :=
sorry

end find_odd_prime_satisfying_condition_l320_320315


namespace closest_point_l320_320955

noncomputable def closest_point_to_origin : ℝ × ℝ :=
  let x := (1 : ℝ) / Real.root 2 4 in
  let y := x + 1 / x in
  (x, y)

theorem closest_point (x y : ℝ) (h : y = x + 1 / x) (hx : x > 0) :
  (x, y) = closest_point_to_origin :=
begin
  sorry
end

end closest_point_l320_320955


namespace product_mod5_is_zero_l320_320276

theorem product_mod5_is_zero :
  (2023 * 2024 * 2025 * 2026) % 5 = 0 :=
by
  sorry

end product_mod5_is_zero_l320_320276


namespace three_mathematicians_speak_same_language_l320_320056

theorem three_mathematicians_speak_same_language :
  (∃ (speak : Fin 9 → Fin 9 → Prop),
  (∀ (i j k : Fin 9), i ≠ j → j ≠ k → i ≠ k → (speak i j ∨ speak i k ∨ speak j k)) ∧
  (∀ i : Fin 9, ∃ (langs : Fin 3 → Prop), ∀ j : Fin 3, speak i j)) →
  ∃ i j k : Fin 9, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ (∀ l : Fin 9, speak i l = speak j l ∧ speak j l = speak k l):=
begin
  sorry
end

end three_mathematicians_speak_same_language_l320_320056


namespace smallest_n_is_50_l320_320457

noncomputable def smallest_n : ℕ :=
  Inf {n | ∃ (x : ℕ → ℝ) (h1 : ∀ i, 0 ≤ x i) (h2 : ∑ i in finset.range n, x i = 1) (h3 : ∑ i in finset.range n, x i ^ 2 ≤ 1 / 50)}

theorem smallest_n_is_50 : smallest_n = 50 :=
by
  sorry

end smallest_n_is_50_l320_320457


namespace sqrt_a_b_c_l320_320758

noncomputable def complex_triangle : ℂ := sorry

def lambda_value : ℝ := sorry

def a : ℕ := 1
def b : ℕ := 4 * 2016^2 - 3
def c : ℕ := 2

def omega_abs : ℝ := 2016

-- We state the Lean theorem corresponding to the problem conditions and the final answer.

theorem sqrt_a_b_c :
  (|complex_triangle| = omega_abs) →
  (∃ (λ : ℝ), λ > 1 ∧ ∀ (ω : ℂ), (complex_triangle = ω ∧ ∃ (a b c : ℕ), a = 1 ∧ b = 4 * 2016^2 - 3 ∧ c = 2)) →
  (λ ω -> (λ * ω = complex_triangle) ∧ (λ = (a + real.sqrt b) / c)) →
  real.sqrt(a + b + c) = 4032 :=
begin
  intros homega hlambda heq,
  sorry
end

end sqrt_a_b_c_l320_320758


namespace compare_abc_l320_320327

def a : ℝ := 5 ^ 0.2
def b : ℝ := Real.log 3 / Real.log π  -- using change of base formula for logarithms
def c : ℝ := Real.log 0.2 / Real.log 5  -- likewise, using change of base formula

theorem compare_abc : a > b ∧ b > c :=
by
  have ha : a = 5 ^ 0.2 := by rfl
  have hb : b = Real.log 3 / Real.log π := by rfl
  have hc : c = Real.log 0.2 / Real.log 5 := by rfl
  sorry

end compare_abc_l320_320327


namespace cube_faces_coloring_l320_320003

noncomputable def number_of_distinct_cube_colorings : ℕ :=
  let total_colorings := nat.factorial 6
  let rotational_symmetries := 24
  total_colorings / rotational_symmetries

theorem cube_faces_coloring : number_of_distinct_cube_colorings = 30 := by
  sorry

end cube_faces_coloring_l320_320003


namespace x_intercept_is_4_l320_320008

/-- Define the points through which the line passes --/
def point1 : ℝ × ℝ := (10, 3)
def point2 : ℝ × ℝ := (-10, -7)

/-- Define the slope calculation for the line passing through the points --/
def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

/-- Define the equation of the line using point-slope form --/
def line_eq (m : ℝ) (p : ℝ × ℝ) (x : ℝ) : ℝ :=
  p.2 + m * (x - p.1)

/-- Define the x-intercept calculation by setting y = 0 in the line equation --/
def x_intercept (m : ℝ) (p : ℝ × ℝ) : ℝ :=
  -(p.2) / m + p.1

/-- Prove that the x-intercept is 4 for the given points --/
theorem x_intercept_is_4 : x_intercept (slope point1 point2) point1 = 4 :=
by
  sorry

end x_intercept_is_4_l320_320008


namespace part_I_intersection_part_I_union_complements_part_II_range_l320_320778

namespace MathProof

-- Definitions of the sets A, B, and C
def A : Set ℝ := {x | 3 < x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2 * a - 1}

-- Prove that the intersection of A and B is {x | 3 < x ∧ x < 6}
theorem part_I_intersection : A ∩ B = {x | 3 < x ∧ x < 6} := sorry

-- Prove that the union of the complements of A and B is {x | x ≤ 3 ∨ x ≥ 6}
theorem part_I_union_complements : (Aᶜ ∪ Bᶜ) = {x | x ≤ 3 ∨ x ≥ 6} := sorry

-- Prove the range of a such that C is a subset of B and B union C equals B
theorem part_II_range (a : ℝ) : B ∪ C a = B → (a ≤ 1 ∨ 2 ≤ a ∧ a ≤ 5) := sorry

end MathProof

end part_I_intersection_part_I_union_complements_part_II_range_l320_320778


namespace sum_sin_squared_angles_l320_320266

theorem sum_sin_squared_angles : 
  ∑ k in finset.range 30, (sin (6 * (k + 1) * (real.pi / 180)))^2 = 31 / 2 := 
sorry

end sum_sin_squared_angles_l320_320266


namespace votes_cast_l320_320636

theorem votes_cast (V : ℝ) (h1 : ∃ (x : ℝ), x = 0.35 * V) (h2 : ∃ (y : ℝ), y = x + 2100) : V = 7000 :=
by sorry

end votes_cast_l320_320636


namespace trig_cos_identity_l320_320845

theorem trig_cos_identity (α : ℝ) (h₁ : α ∈ Ioo 0 (π / 4)) 
  (h₂ : tan α + 1 / tan α = 5 / 2) : 
  cos (2 * α - π / 4) = 7 * sqrt 2 / 10 := 
sorry

end trig_cos_identity_l320_320845


namespace chord_sum_equal_l320_320224

theorem chord_sum_equal 
  (O P A B C D E F : Point) 
  (h_circle : Circle O)
  (h_intersect : intersects_at_point O A B C D E F P)
  (h_angles : ∀ X Y Z, ∠(X, P, Y) = 60 ∧ ∠(Y, P, Z) = 60 ∧ ∠(Z, P, X) = 60) :
  distance P A + distance P E + distance P D = distance P C + distance P B + distance P F :=
  sorry

end chord_sum_equal_l320_320224


namespace find_x_plus_one_over_x_l320_320350

open Real

theorem find_x_plus_one_over_x (x : ℝ) (h : x ^ 3 + 1 / x ^ 3 = 110) : x + 1 / x = 5 :=
sorry

end find_x_plus_one_over_x_l320_320350
