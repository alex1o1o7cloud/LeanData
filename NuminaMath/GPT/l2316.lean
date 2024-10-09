import Mathlib

namespace rayden_spent_more_l2316_231698

-- Define the conditions
def lily_ducks := 20
def lily_geese := 10
def lily_chickens := 5
def lily_pigeons := 30

def rayden_ducks := 3 * lily_ducks
def rayden_geese := 4 * lily_geese
def rayden_chickens := 5 * lily_chickens
def rayden_pigeons := lily_pigeons / 2

def duck_price := 15
def geese_price := 20
def chicken_price := 10
def pigeon_price := 5

def lily_total := lily_ducks * duck_price +
                  lily_geese * geese_price +
                  lily_chickens * chicken_price +
                  lily_pigeons * pigeon_price

def rayden_total := rayden_ducks * duck_price +
                    rayden_geese * geese_price +
                    rayden_chickens * chicken_price +
                    rayden_pigeons * pigeon_price

def spending_difference := rayden_total - lily_total

theorem rayden_spent_more : spending_difference = 1325 := 
by 
  unfold spending_difference rayden_total lily_total -- to simplify the definitions
  sorry -- Proof is omitted

end rayden_spent_more_l2316_231698


namespace find_f_neg_2017_l2316_231613

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f x = f (-x)
axiom periodic_function : ∀ x : ℝ, x ≥ 0 → f (x + 2) = f x
axiom log_function : ∀ x : ℝ, 0 ≤ x ∧ x < 2 → f x = Real.log (x + 1) / Real.log 2

theorem find_f_neg_2017 : f (-2017) = 1 := by
  sorry

end find_f_neg_2017_l2316_231613


namespace sphere_surface_area_l2316_231611

theorem sphere_surface_area 
  (a b c : ℝ) 
  (h1 : a = 1)
  (h2 : b = 2)
  (h3 : c = 2)
  (h_spherical_condition : ∃ R : ℝ, ∀ (x y z : ℝ), x^2 + y^2 + z^2 = (2 * R)^2) :
  4 * Real.pi * ((3 / 2)^2) = 9 * Real.pi :=
by
  sorry

end sphere_surface_area_l2316_231611


namespace max_value_a_l2316_231681

-- Define the variables and the constraint on the circle
def circular_arrangement_condition (x: ℕ → ℕ) : Prop :=
  ∀ i: ℕ, 1 ≤ x i ∧ x i ≤ 10 ∧ x i ≠ x (i + 1)

-- Define the existence of three consecutive numbers summing to at least 18
def three_consecutive_sum_ge_18 (x: ℕ → ℕ) : Prop :=
  ∃ i: ℕ, x i + x (i + 1) + x (i + 2) ≥ 18

-- The main theorem we aim to prove
theorem max_value_a : ∀ (x: ℕ → ℕ), circular_arrangement_condition x → three_consecutive_sum_ge_18 x :=
  by sorry

end max_value_a_l2316_231681


namespace sin_cos_identity_l2316_231650

theorem sin_cos_identity (x : ℝ) (h : Real.cos x - 5 * Real.sin x = 2) : Real.sin x + 5 * Real.cos x = -28 / 13 := 
  sorry

end sin_cos_identity_l2316_231650


namespace find_f_2_l2316_231623

theorem find_f_2 (f : ℝ → ℝ) (h₁ : f 1 = 0)
  (h₂ : ∀ x y : ℝ, f (x^2 + y^2) = (x + y) * (f x + f y)) :
  f 2 = 0 :=
sorry

end find_f_2_l2316_231623


namespace numerical_value_expression_l2316_231654

theorem numerical_value_expression (a b : ℝ) (h1 : a ≠ b) 
  (h2 : 1 / (a^2 + 1) + 1 / (b^2 + 1) = 2 / (ab + 1)) : 
  1 / (a^2 + 1) + 1 / (b^2 + 1) + 2 / (ab + 1) = 2 := 
by 
  -- Proof outline provided in the solution section, but actual proof is omitted
  sorry

end numerical_value_expression_l2316_231654


namespace expression_eval_neg_sqrt_l2316_231647

variable (a : ℝ)

theorem expression_eval_neg_sqrt (ha : a < 0) : a * Real.sqrt (-1 / a) = -Real.sqrt (-a) :=
by
  sorry

end expression_eval_neg_sqrt_l2316_231647


namespace find_digits_l2316_231630

/-- 
  Find distinct digits A, B, C, and D such that 9 * (100 * A + 10 * B + C) = B * (1000 * B + 100 * C + 10 * D + B).
 -/
theorem find_digits
  (A B C D : ℕ)
  (hA : A ≠ B) (hA : A ≠ C) (hA : A ≠ D)
  (hB : B ≠ C) (hB : B ≠ D)
  (hC : C ≠ D)
  (hNonZeroB : B ≠ 0) :
  9 * (100 * A + 10 * B + C) = B * (1000 * B + 100 * C + 10 * D + B) ↔ (A = 2 ∧ B = 1 ∧ C = 9 ∧ D = 7) := by
  sorry

end find_digits_l2316_231630


namespace min_value_of_quadratic_l2316_231683

theorem min_value_of_quadratic (a b : ℝ) (h1 : a * b ≠ 0) (h2 : a^2 ≠ b^2) : 
  ∃ (x : ℝ), (∃ (y_min : ℝ), y_min = -( (abs (a - b)/2)^2 ) 
  ∧ ∀ (x : ℝ), (x - a)*(x - b) ≥ y_min) :=
sorry

end min_value_of_quadratic_l2316_231683


namespace sum_of_first_five_primes_with_units_digit_3_l2316_231696

open Nat

-- Predicate to check if a number has a units digit of 3
def hasUnitsDigit3 (n : ℕ) : Prop :=
n % 10 = 3

-- List of the first five prime numbers that have a units digit of 3
def firstFivePrimesUnitsDigit3 : List ℕ :=
[3, 13, 23, 43, 53]

-- Definition for sum of the first five primes with units digit 3
def sumFirstFivePrimesUnitsDigit3 : ℕ :=
(firstFivePrimesUnitsDigit3).sum

-- Theorem statement
theorem sum_of_first_five_primes_with_units_digit_3 :
  sumFirstFivePrimesUnitsDigit3 = 135 := by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l2316_231696


namespace number_of_moles_of_H2O_l2316_231624

def reaction_stoichiometry (n_NaOH m_Cl2 : ℕ) : ℕ :=
  1  -- Moles of H2O produced according to the balanced equation with the given reactants

theorem number_of_moles_of_H2O 
  (n_NaOH : ℕ) (m_Cl2 : ℕ) 
  (h_NaOH : n_NaOH = 2) 
  (h_Cl2 : m_Cl2 = 1) :
  reaction_stoichiometry n_NaOH m_Cl2 = 1 :=
by
  rw [h_NaOH, h_Cl2]
  -- Would typically follow with the proof using the conditions and stoichiometric relation
  sorry  -- Proof step omitted

end number_of_moles_of_H2O_l2316_231624


namespace exponential_function_value_l2316_231666

noncomputable def f (x : ℝ) : ℝ := 2^x

theorem exponential_function_value :
  f (f 2) = 16 := by
  simp only [f]
  sorry

end exponential_function_value_l2316_231666


namespace largest_angle_in_triangle_l2316_231667

open Real

theorem largest_angle_in_triangle
  (A B C : ℝ)
  (h : sin A / sin B / sin C = 1 / sqrt 2 / sqrt 5) :
  A ≤ B ∧ B ≤ C → C = 3 * π / 4 :=
by
  sorry

end largest_angle_in_triangle_l2316_231667


namespace smallest_number_divisible_by_set_l2316_231616

theorem smallest_number_divisible_by_set : ∃ x : ℕ, (∀ d ∈ [12, 24, 36, 48, 56, 72, 84], (x - 24) % d = 0) ∧ x = 1032 := 
by {
  sorry
}

end smallest_number_divisible_by_set_l2316_231616


namespace fire_alarms_and_passengers_discrete_l2316_231600

-- Definitions of the random variables
def xi₁ : ℕ := sorry  -- number of fire alarms in a city within one day
def xi₂ : ℝ := sorry  -- temperature in a city within one day
def xi₃ : ℕ := sorry  -- number of passengers at a train station in a city within a month

-- Defining the concept of discrete random variable
def is_discrete (X : Type) : Prop := 
  ∃ f : X → ℕ, ∀ x : X, ∃ n : ℕ, f x = n

-- Statement of the proof problem
theorem fire_alarms_and_passengers_discrete :
  is_discrete ℕ ∧ is_discrete ℕ ∧ ¬ is_discrete ℝ :=
by
  have xi₁_discrete : is_discrete ℕ := sorry
  have xi₃_discrete : is_discrete ℕ := sorry
  have xi₂_not_discrete : ¬ is_discrete ℝ := sorry
  exact ⟨xi₁_discrete, xi₃_discrete, xi₂_not_discrete⟩

end fire_alarms_and_passengers_discrete_l2316_231600


namespace g_at_5_l2316_231665

noncomputable def g : ℝ → ℝ := sorry

axiom functional_equation :
  ∀ (x : ℝ), g x + 3 * g (2 - x) = 4 * x ^ 2 - 5 * x + 1

theorem g_at_5 : g 5 = -5 / 4 :=
by
  let h := functional_equation
  sorry

end g_at_5_l2316_231665


namespace linear_relation_is_correct_maximum_profit_l2316_231674

-- Define the given data points
structure DataPoints where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ

-- Define the given conditions
def conditions : DataPoints := ⟨50, 100, 60, 90⟩

-- Define the cost and sell price range conditions
def cost_per_kg : ℝ := 20
def max_selling_price : ℝ := 90

-- Define the linear relationship function
def linear_relationship (k b x : ℝ) : ℝ := k * x + b

-- Define the profit function
def profit_function (x : ℝ) : ℝ := (x - cost_per_kg) * (linear_relationship (-1) 150 x)

-- Statements to Prove
theorem linear_relation_is_correct (k b : ℝ) :
  linear_relationship k b 50 = 100 ∧
  linear_relationship k b 60 = 90 →
  (b = 150 ∧ k = -1) := by
  intros h
  sorry

theorem maximum_profit :
  ∃ x : ℝ, 20 ≤ x ∧ x ≤ max_selling_price ∧ profit_function x = 4225 := by
  use 85
  sorry

end linear_relation_is_correct_maximum_profit_l2316_231674


namespace sum_of_digits_joey_age_l2316_231685

def int.multiple (a b : ℕ) := ∃ k : ℕ, a = k * b

theorem sum_of_digits_joey_age (J C M n : ℕ) (h1 : J = C + 2) (h2 : M = 2) (h3 : ∃ k, C = k * M) (h4 : C = 12) (h5 : J + n = 26) : 
  (2 + 6 = 8) :=
by
  sorry

end sum_of_digits_joey_age_l2316_231685


namespace problem_statement_l2316_231619

theorem problem_statement (x : ℝ) (h : x ≠ 2) :
  (x * (x + 1)) / ((x - 2)^2) ≥ 8 ↔ (1 ≤ x ∧ x < 2) ∨ (32/7 < x) :=
by 
  sorry

end problem_statement_l2316_231619


namespace odd_function_at_zero_l2316_231607

theorem odd_function_at_zero
  (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x) :
  f 0 = 0 :=
by
  sorry

end odd_function_at_zero_l2316_231607


namespace triangle_perimeter_l2316_231679

-- Definitions and given conditions
def side_length_a (a : ℝ) : Prop := a = 6
def inradius (r : ℝ) : Prop := r = 2
def circumradius (R : ℝ) : Prop := R = 5

-- The final proof statement to be proven
theorem triangle_perimeter (a r R : ℝ) (b c P : ℝ) 
  (h1 : side_length_a a)
  (h2 : inradius r)
  (h3 : circumradius R)
  (h4 : P = 2 * ((a + b + c) / 2)) :
  P = 24 :=
sorry

end triangle_perimeter_l2316_231679


namespace find_x_range_l2316_231612

-- Define the condition for the expression to be meaningful
def meaningful_expr (x : ℝ) : Prop := x - 3 ≥ 0

-- The range of values for x is equivalent to x being at least 3
theorem find_x_range (x : ℝ) : meaningful_expr x ↔ x ≥ 3 := by
  sorry

end find_x_range_l2316_231612


namespace birds_on_fence_l2316_231658

def number_of_birds_on_fence : ℕ := 20

theorem birds_on_fence (x : ℕ) (h : 2 * x + 10 = 50) : x = number_of_birds_on_fence :=
by
  sorry

end birds_on_fence_l2316_231658


namespace arithmetic_mean_of_remaining_numbers_l2316_231655

-- Definitions and conditions
def initial_set_size : ℕ := 60
def initial_arithmetic_mean : ℕ := 45
def numbers_to_remove : List ℕ := [50, 55, 60]

-- Calculation of the total sum
def total_sum : ℕ := initial_arithmetic_mean * initial_set_size

-- Calculation of the sum of the numbers to remove
def sum_of_removed_numbers : ℕ := numbers_to_remove.sum

-- Sum of the remaining numbers
def new_sum : ℕ := total_sum - sum_of_removed_numbers

-- Size of the remaining set
def remaining_set_size : ℕ := initial_set_size - numbers_to_remove.length

-- The arithmetic mean of the remaining numbers
def new_arithmetic_mean : ℚ := new_sum / remaining_set_size

-- The proof statement
theorem arithmetic_mean_of_remaining_numbers :
  new_arithmetic_mean = 2535 / 57 :=
by
  sorry

end arithmetic_mean_of_remaining_numbers_l2316_231655


namespace simplify_fraction_l2316_231694

theorem simplify_fraction :
  (1 / (3 / (Real.sqrt 5 + 2) + 4 / (Real.sqrt 7 - 2))) = (3 / (9 * Real.sqrt 5 + 4 * Real.sqrt 7 - 10)) :=
sorry

end simplify_fraction_l2316_231694


namespace problem_l2316_231690

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q

theorem problem
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_pos : ∀ n, 0 < a n)
  (h_geom : geometric_sequence a q)
  (h1 : a 0 + a 1 = 4 / 9)
  (h2 : a 2 + a 3 + a 4 + a 5 = 40) :
  (a 6 + a 7 + a 8) / 9 = 117 :=
sorry

end problem_l2316_231690


namespace fizz_preference_count_l2316_231664

-- Definitions from conditions
def total_people : ℕ := 500
def fizz_angle : ℕ := 270
def total_angle : ℕ := 360
def fizz_fraction : ℚ := fizz_angle / total_angle

-- The target proof statement
theorem fizz_preference_count (hp : total_people = 500) 
                              (ha : fizz_angle = 270) 
                              (ht : total_angle = 360)
                              (hf : fizz_fraction = 3 / 4) : 
    total_people * fizz_fraction = 375 := by
    sorry

end fizz_preference_count_l2316_231664


namespace curvilinear_quadrilateral_area_l2316_231606

-- Conditions: Define radius R, and plane angles of the tetrahedral angle.
noncomputable def radius (R : Real) : Prop :=
  R > 0

noncomputable def angle (theta : Real) : Prop :=
  theta = 60

-- Establishing the final goal based on the given conditions and solution's correct answer.
theorem curvilinear_quadrilateral_area
  (R : Real)     -- given radius of the sphere
  (hR : radius R) -- the radius of the sphere touching all edges
  (theta : Real)  -- given angle in degrees
  (hθ : angle theta) -- the plane angle of 60 degrees
  :
  ∃ A : Real, 
    A = π * R^2 * (16/3 * (Real.sqrt (2/3)) - 2) := 
  sorry

end curvilinear_quadrilateral_area_l2316_231606


namespace percentage_calculation_l2316_231660

theorem percentage_calculation (percentage : ℝ) (h : percentage * 50 = 0.15) : percentage = 0.003 :=
by
  sorry

end percentage_calculation_l2316_231660


namespace james_pitbull_count_l2316_231672

-- Defining the conditions
def husky_count : ℕ := 5
def retriever_count : ℕ := 4
def retriever_pups_per_retriever (husky_pups_per_husky : ℕ) : ℕ := husky_pups_per_husky + 2
def husky_pups := husky_count * 3
def retriever_pups := retriever_count * (retriever_pups_per_retriever 3)
def pitbull_pups (P : ℕ) : ℕ := P * 3
def total_pups (P : ℕ) : ℕ := husky_pups + retriever_pups + pitbull_pups P
def total_adults (P : ℕ) : ℕ := husky_count + retriever_count + P
def condition (P : ℕ) : Prop := total_pups P = total_adults P + 30

-- The proof objective
theorem james_pitbull_count : ∃ P : ℕ, condition P → P = 2 := by
  sorry

end james_pitbull_count_l2316_231672


namespace number_of_articles_l2316_231615

-- Conditions
variables (C S : ℚ)
-- Given that the cost price of 50 articles is equal to the selling price of some number of articles N.
variables (N : ℚ) (h1 : 50 * C = N * S)
-- Given that the gain is 11.11111111111111 percent.
variables (gain : ℚ := 1/9) (h2 : S = C * (1 + gain))

-- Prove that N = 45
theorem number_of_articles (C S : ℚ) (N : ℚ) (h1 : 50 * C = N * S)
    (gain : ℚ := 1/9) (h2 : S = C * (1 + gain)) : N = 45 :=
by
  sorry

end number_of_articles_l2316_231615


namespace cos_pi_minus_alpha_l2316_231620

open Real

variable (α : ℝ)

theorem cos_pi_minus_alpha (h1 : 0 < α ∧ α < π / 2) (h2 : sin α = 4 / 5) : cos (π - α) = -3 / 5 := by
  sorry

end cos_pi_minus_alpha_l2316_231620


namespace all_stones_weigh_the_same_l2316_231634

theorem all_stones_weigh_the_same (x : Fin 13 → ℕ)
  (h : ∀ (i : Fin 13), ∃ (A B : Finset (Fin 13)), A.card = 6 ∧ B.card = 6 ∧
    i ∉ A ∧ i ∉ B ∧ ∀ (j k : Fin 13), j ∈ A → k ∈ B → x j = x k): 
  ∀ i j : Fin 13, x i = x j := 
sorry

end all_stones_weigh_the_same_l2316_231634


namespace same_solution_implies_value_of_m_l2316_231618

theorem same_solution_implies_value_of_m (x m : ℤ) (h₁ : -5 * x - 6 = 3 * x + 10) (h₂ : -2 * m - 3 * x = 10) : m = -2 :=
by
  sorry

end same_solution_implies_value_of_m_l2316_231618


namespace y_intercept_of_line_with_slope_3_and_x_intercept_7_0_l2316_231657

def line_equation (m x1 y1 x y : ℝ) : Prop :=
  y - y1 = m * (x - x1)

theorem y_intercept_of_line_with_slope_3_and_x_intercept_7_0 :
  ∃ b : ℝ, line_equation 3 7 0 0 b ∧ b = -21 :=
by
  sorry

end y_intercept_of_line_with_slope_3_and_x_intercept_7_0_l2316_231657


namespace general_formula_a_n_sum_T_n_l2316_231684

-- Definitions of the sequences
def a (n : ℕ) : ℕ := 4 + (n - 1) * 1
def S (n : ℕ) : ℕ := n / 2 * (2 * 4 + (n - 1) * 1)
def b (n : ℕ) : ℕ := 2 ^ (a n - 3)
def T (n : ℕ) : ℕ := 2 * (2 ^ n - 1)

-- Given conditions
axiom a4_eq_7 : a 4 = 7
axiom S2_eq_9 : S 2 = 9

-- Theorems to prove
theorem general_formula_a_n : ∀ n, a n = n + 3 := 
by sorry

theorem sum_T_n : ∀ n, T n = 2 ^ (n + 1) - 2 := 
by sorry

end general_formula_a_n_sum_T_n_l2316_231684


namespace metal_sheets_per_panel_l2316_231692

-- Define the given conditions
def num_panels : ℕ := 10
def rods_per_sheet : ℕ := 10
def rods_per_beam : ℕ := 4
def beams_per_panel : ℕ := 2
def total_rods_needed : ℕ := 380

-- Question translated to Lean statement
theorem metal_sheets_per_panel (S : ℕ) (h : 10 * (10 * S + 8) = 380) : S = 3 := 
  sorry

end metal_sheets_per_panel_l2316_231692


namespace expression_even_l2316_231627

theorem expression_even (a b c : ℕ) (ha : a % 2 = 0) (hb : b % 2 = 1) :
  ∃ k : ℕ, 2^a * (b+1) ^ 2 * c = 2 * k :=
by
sorry

end expression_even_l2316_231627


namespace max_sum_of_four_distinct_with_lcm_165_l2316_231697

theorem max_sum_of_four_distinct_with_lcm_165 (a b c d : ℕ)
  (h1 : Nat.lcm a b = 165)
  (h2 : Nat.lcm a c = 165)
  (h3 : Nat.lcm a d = 165)
  (h4 : Nat.lcm b c = 165)
  (h5 : Nat.lcm b d = 165)
  (h6 : Nat.lcm c d = 165)
  (h7 : a ≠ b) (h8 : a ≠ c) (h9 : a ≠ d)
  (h10 : b ≠ c) (h11 : b ≠ d) (h12 : c ≠ d) :
  a + b + c + d ≤ 268 := sorry

end max_sum_of_four_distinct_with_lcm_165_l2316_231697


namespace solve_for_y_l2316_231614

def solution (y : ℝ) : Prop :=
  2 * Real.arctan (1/3) - Real.arctan (1/5) + Real.arctan (1/y) = Real.pi / 4

theorem solve_for_y (y : ℝ) : solution y → y = 31 / 9 :=
by
  intro h
  sorry

end solve_for_y_l2316_231614


namespace total_number_of_fish_l2316_231675

def number_of_tuna : Nat := 5
def number_of_spearfish : Nat := 2

theorem total_number_of_fish : number_of_tuna + number_of_spearfish = 7 := by
  sorry

end total_number_of_fish_l2316_231675


namespace explicit_formula_of_odd_function_monotonicity_in_interval_l2316_231638

-- Using Noncomputable because divisions are involved.
noncomputable def f (x : ℝ) (p q : ℝ) : ℝ := (p * x^2 + 2) / (q - 3 * x)

theorem explicit_formula_of_odd_function (p q : ℝ) 
  (h_odd : ∀ x : ℝ, f x p q = - f (-x) p q) 
  (h_value : f 2 p q = -5/3) : 
  f x 2 0 = -2/3 * (x + 1/x) :=
by sorry

theorem monotonicity_in_interval {x : ℝ} (h_domain : 0 < x ∧ x < 1) : 
  ∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ x2 < 1 -> f x1 2 0 < f x2 2 0 :=
by sorry

end explicit_formula_of_odd_function_monotonicity_in_interval_l2316_231638


namespace B_pow_150_eq_I_l2316_231645

def B : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![0, 1, 0],
  ![0, 0, 1],
  ![1, 0, 0]
]

theorem B_pow_150_eq_I : B^(150 : ℕ) = (1 : Matrix (Fin 3) (Fin 3) ℝ) :=
by {
  sorry
}

end B_pow_150_eq_I_l2316_231645


namespace fraction_sum_l2316_231676

theorem fraction_sum : (3 / 8) + (9 / 14) = (57 / 56) := by
  sorry

end fraction_sum_l2316_231676


namespace function_is_zero_l2316_231641

variable (n : ℕ) (a : Fin n → ℤ) (f : ℤ → ℝ)

axiom condition : ∀ (k l : ℤ), l ≠ 0 → (Finset.univ.sum (λ i => f (k + a i * l)) = 0)

theorem function_is_zero : ∀ x : ℤ, f x = 0 := by
  sorry

end function_is_zero_l2316_231641


namespace walter_age_1999_l2316_231688

variable (w g : ℕ) -- represents Walter's age (w) and his grandmother's age (g) in 1994
variable (birth_sum : ℕ) (w_age_1994 : ℕ) (g_age_1994 : ℕ)

axiom h1 : g = 2 * w
axiom h2 : (1994 - w) + (1994 - g) = 3838

theorem walter_age_1999 (w g : ℕ) (h1 : g = 2 * w) (h2 : (1994 - w) + (1994 - g) = 3838) : w + 5 = 55 :=
by
  sorry

end walter_age_1999_l2316_231688


namespace sum_contains_even_digit_l2316_231626

-- Define the five-digit integer and its reversed form
def reversed_digits (n : ℕ) : ℕ := 
  let a := n % 10
  let b := (n / 10) % 10
  let c := (n / 100) % 10
  let d := (n / 1000) % 10
  let e := (n / 10000) % 10
  a * 10000 + b * 1000 + c * 100 + d * 10 + e

theorem sum_contains_even_digit (n m : ℕ) (h1 : n >= 10000) (h2 : n < 100000) (h3 : m = reversed_digits n) : 
  ∃ d : ℕ, d < 10 ∧ d % 2 = 0 ∧ (n + m) % 10 = d ∨ (n + m) / 10 % 10 = d ∨ (n + m) / 100 % 10 = d ∨ (n + m) / 1000 % 10 = d ∨ (n + m) / 10000 % 10 = d := 
sorry

end sum_contains_even_digit_l2316_231626


namespace minimum_value_frac_abc_l2316_231633

variable (a b c : ℝ)

theorem minimum_value_frac_abc
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a + b + 2 * c = 2) :
  (a + b) / (a * b * c) ≥ 8 :=
sorry

end minimum_value_frac_abc_l2316_231633


namespace actual_cost_l2316_231686

theorem actual_cost (x : ℝ) (h : 0.80 * x = 200) : x = 250 :=
sorry

end actual_cost_l2316_231686


namespace avg_first_six_results_l2316_231678

theorem avg_first_six_results (A : ℝ) :
  (∀ (results : Fin 12 → ℝ), 
    (results 0 + results 1 + results 2 + results 3 + results 4 + results 5 + 
     results 6 + results 7 + results 8 + results 9 + results 10 + results 11) / 11 = 60 → 
    (results 0 + results 1 + results 2 + results 3 + results 4 + results 5) / 6 = A → 
    (results 5 + results 6 + results 7 + results 8 + results 9 + results 10) / 6 = 63 → 
    results 5 = 66) → 
  A = 58 :=
by
  sorry

end avg_first_six_results_l2316_231678


namespace acid_solution_mix_l2316_231689

theorem acid_solution_mix (x : ℝ) (h₁ : 0.2 * x + 50 = 0.35 * (100 + x)) : x = 100 :=
by
  sorry

end acid_solution_mix_l2316_231689


namespace candle_problem_l2316_231677

-- Define the initial heights and burn rates of the candles
def heightA (t : ℝ) : ℝ := 12 - 2 * t
def heightB (t : ℝ) : ℝ := 15 - 3 * t

-- Lean theorem statement for the given problem
theorem candle_problem : ∃ t : ℝ, (heightA t = (1/3) * heightB t) ∧ t = 7 :=
by
  -- This is to keep the theorem statement valid without the proof
  sorry

end candle_problem_l2316_231677


namespace evaluate_expression_at_x_eq_3_l2316_231635

theorem evaluate_expression_at_x_eq_3 : (3 ^ 3) ^ (3 ^ 3) = 27 ^ 27 := by
  sorry

end evaluate_expression_at_x_eq_3_l2316_231635


namespace difference_between_two_numbers_l2316_231604

theorem difference_between_two_numbers 
  (x y : ℝ) 
  (h1 : x + y = 20) 
  (h2 : x - y = 10) 
  (h3 : x^2 - y^2 = 200) : 
  x - y = 10 :=
by 
  sorry

end difference_between_two_numbers_l2316_231604


namespace nonzero_fraction_exponent_zero_l2316_231693

theorem nonzero_fraction_exponent_zero (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) : (a / b : ℚ)^0 = 1 := 
by 
  sorry

end nonzero_fraction_exponent_zero_l2316_231693


namespace problem_l2316_231636

theorem problem (x y z : ℝ) (h1 : x = y + z) (h2 : x = 2) : 
  x^3 + 3 * y^2 + 3 * z^2 + 3 * x * y * z = 20 := by
sorry

end problem_l2316_231636


namespace cylinder_intersection_in_sphere_l2316_231602

theorem cylinder_intersection_in_sphere
  (a b c d e f : ℝ)
  (x y z : ℝ)
  (h1 : (x - a)^2 + (y - b)^2 < 1)
  (h2 : (y - c)^2 + (z - d)^2 < 1)
  (h3 : (z - e)^2 + (x - f)^2 < 1) :
  (x - (a + f) / 2)^2 + (y - (b + c) / 2)^2 + (z - (d + e) / 2)^2 < 3 / 2 := 
sorry

end cylinder_intersection_in_sphere_l2316_231602


namespace third_height_less_than_30_l2316_231642

theorem third_height_less_than_30 (h_a h_b : ℝ) (h_a_pos : h_a = 12) (h_b_pos : h_b = 20) : 
    ∃ (h_c : ℝ), h_c < 30 :=
by
  sorry

end third_height_less_than_30_l2316_231642


namespace increasing_range_of_a_l2316_231653

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x + 1 / x

theorem increasing_range_of_a (a : ℝ) : (∀ x > (1/2), (3 * x^2 + a - 1 / x^2) ≥ 0) ↔ a ≥ (13 / 4) :=
by sorry

end increasing_range_of_a_l2316_231653


namespace intercepts_equal_lines_parallel_l2316_231639

-- Definition of the conditions: line equations
def line_l (a : ℝ) : Prop := ∀ x y : ℝ, a * x + 3 * y + 1 = 0

-- Problem (1) : The intercepts of the line on the two coordinate axes are equal
theorem intercepts_equal (a : ℝ) (h : line_l a) : a = 3 := by
  sorry

-- Problem (2): The line is parallel to x + (a-2)y + a = 0
theorem lines_parallel (a : ℝ) (h : line_l a) : (∀ x y : ℝ, x + (a-2) * y + a = 0) → a = 3 := by
  sorry

end intercepts_equal_lines_parallel_l2316_231639


namespace minimum_workers_needed_l2316_231610

-- Definitions
def job_completion_time : ℕ := 45
def days_worked : ℕ := 9
def portion_job_done : ℚ := 1 / 5
def team_size : ℕ := 10
def job_remaining : ℚ := (1 - portion_job_done)
def days_remaining : ℕ := job_completion_time - days_worked
def daily_completion_rate_by_team : ℚ := portion_job_done / days_worked
def daily_completion_rate_per_person : ℚ := daily_completion_rate_by_team / team_size
def required_daily_rate : ℚ := job_remaining / days_remaining

-- Statement to be proven
theorem minimum_workers_needed :
  (required_daily_rate / daily_completion_rate_per_person) = 10 :=
sorry

end minimum_workers_needed_l2316_231610


namespace Jim_catches_Bob_in_20_minutes_l2316_231637

theorem Jim_catches_Bob_in_20_minutes
  (Bob_Speed : ℕ := 6)
  (Jim_Speed : ℕ := 9)
  (Head_Start : ℕ := 1) :
  (Head_Start / (Jim_Speed - Bob_Speed) * 60 = 20) :=
by
  sorry

end Jim_catches_Bob_in_20_minutes_l2316_231637


namespace boundary_length_of_divided_rectangle_l2316_231628

/-- Suppose a rectangle is divided into three equal parts along its length and two equal parts along its width, 
creating semicircle arcs connecting points on adjacent sides. Given the rectangle has an area of 72 square units, 
we aim to prove that the total length of the boundary of the resulting figure is 36.0. -/
theorem boundary_length_of_divided_rectangle 
(area_of_rectangle : ℝ)
(length_divisions : ℕ)
(width_divisions : ℕ)
(semicircle_arcs_length : ℝ)
(straight_segments_length : ℝ) :
  area_of_rectangle = 72 →
  length_divisions = 3 →
  width_divisions = 2 →
  semicircle_arcs_length = 7 * Real.pi →
  straight_segments_length = 14 →
  semicircle_arcs_length + straight_segments_length = 36 :=
by
  intros h_area h_length_div h_width_div h_arc_length h_straight_length
  sorry

end boundary_length_of_divided_rectangle_l2316_231628


namespace unused_types_l2316_231621

theorem unused_types (total_resources : ℕ) (used_types : ℕ) (valid_types : ℕ) :
  total_resources = 6 → used_types = 23 → valid_types = 2^total_resources - 1 - used_types → valid_types = 40 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3


end unused_types_l2316_231621


namespace simplify_expression_eval_at_2_l2316_231625

theorem simplify_expression (a b c x : ℝ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c) :
    (x^2 + a)^2 / ((a - b) * (a - c)) + (x^2 + b)^2 / ((b - a) * (b - c)) + (x^2 + c)^2 / ((c - a) * (c - b)) =
    x^4 + x^2 * (a + b + c) + (a^2 + b^2 + c^2) :=
sorry

theorem eval_at_2 (a b c : ℝ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c) :
    (2^2 + a)^2 / ((a - b) * (a - c)) + (2^2 + b)^2 / ((b - a) * (b - c)) + (2^2 + c)^2 / ((c - a) * (c - b)) =
    16 + 4 * (a + b + c) + (a^2 + b^2 + c^2) :=
sorry

end simplify_expression_eval_at_2_l2316_231625


namespace increasing_intervals_decreasing_interval_l2316_231648

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 - x

theorem increasing_intervals : 
  (∀ x, x < -1/3 → deriv f x > 0) ∧ 
  (∀ x, x > 1 → deriv f x > 0) :=
sorry

theorem decreasing_interval : 
  ∀ x, -1/3 < x ∧ x < 1 → deriv f x < 0 :=
sorry

end increasing_intervals_decreasing_interval_l2316_231648


namespace Yoongi_stack_taller_than_Taehyung_l2316_231601

theorem Yoongi_stack_taller_than_Taehyung :
  let height_A := 3
  let height_B := 3.5
  let count_A := 16
  let count_B := 14
  let total_height_A := height_A * count_A
  let total_height_B := height_B * count_B
  total_height_B > total_height_A ∧ (total_height_B - total_height_A = 1) :=
by
  sorry

end Yoongi_stack_taller_than_Taehyung_l2316_231601


namespace total_tires_mike_changed_l2316_231663

theorem total_tires_mike_changed (num_motorcycles : ℕ) (tires_per_motorcycle : ℕ)
                                (num_cars : ℕ) (tires_per_car : ℕ)
                                (total_tires : ℕ) :
  num_motorcycles = 12 →
  tires_per_motorcycle = 2 →
  num_cars = 10 →
  tires_per_car = 4 →
  total_tires = num_motorcycles * tires_per_motorcycle + num_cars * tires_per_car →
  total_tires = 64 := by
  intros h1 h2 h3 h4 h5
  sorry

end total_tires_mike_changed_l2316_231663


namespace mason_savings_fraction_l2316_231670

theorem mason_savings_fraction (M p b : ℝ) (h : (1 / 4) * M = (2 / 5) * b * p) : 
  (M - b * p) / M = 3 / 8 :=
by 
  sorry

end mason_savings_fraction_l2316_231670


namespace expenditure_ratio_l2316_231661

/-- A man saves 35% of his income in the first year. -/
def saving_rate_first_year : ℝ := 0.35

/-- His income increases by 35% in the second year. -/
def income_increase_rate : ℝ := 0.35

/-- His savings increase by 100% in the second year. -/
def savings_increase_rate : ℝ := 1.0

theorem expenditure_ratio
  (I : ℝ)  -- first year income
  (S1 : ℝ := saving_rate_first_year * I)  -- first year saving
  (E1 : ℝ := I - S1)  -- first year expenditure
  (I2 : ℝ := I + income_increase_rate * I)  -- second year income
  (S2 : ℝ := 2 * S1)  -- second year saving (increases by 100%)
  (E2 : ℝ := I2 - S2)  -- second year expenditure
  :
  (E1 + E2) / E1 = 2
  :=
  sorry

end expenditure_ratio_l2316_231661


namespace relationship_abc_l2316_231643

theorem relationship_abc (a b c : ℝ) (ha : a = Real.exp 0.1 - 1) (hb : b = 0.1) (hc : c = Real.log 1.1) :
  c < b ∧ b < a :=
by
  sorry

end relationship_abc_l2316_231643


namespace necessary_but_not_sufficient_condition_l2316_231617

theorem necessary_but_not_sufficient_condition (x : ℝ) : 
  (∃ x, x > 2 ∧ ¬ (x > 3)) ∧ 
  (∀ x, x > 3 → x > 2) := by sorry

end necessary_but_not_sufficient_condition_l2316_231617


namespace proof_problem_l2316_231682

noncomputable def a : ℝ := 3.54
noncomputable def b : ℝ := 1.32
noncomputable def result : ℝ := (a - b) * 2

theorem proof_problem : result = 4.44 := by
  sorry

end proof_problem_l2316_231682


namespace find_special_four_digit_square_l2316_231651

def is_four_digit (n : ℕ) : Prop := n >= 1000 ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop := ∃ (x : ℕ), x * x = n

def same_first_two_digits (n : ℕ) : Prop := (n / 1000) = (n / 100 % 10)

def same_last_two_digits (n : ℕ) : Prop := (n % 100 / 10) = (n % 10)

theorem find_special_four_digit_square :
  ∃ (n : ℕ), is_four_digit n ∧ is_perfect_square n ∧ same_first_two_digits n ∧ same_last_two_digits n ∧ n = 7744 := 
sorry

end find_special_four_digit_square_l2316_231651


namespace Petya_wins_optimally_l2316_231605

-- Defining the game state and rules
inductive GameState
| PetyaWin
| VasyaWin

-- Rules of the game
def game_rule (n : ℕ) : Prop :=
  n > 0 ∧ (n % 3 = 0 ∨ n % 3 = 1 ∨ n % 3 = 2)

-- Determine the winner given the initial number of minuses
def determine_winner (n : ℕ) : GameState :=
  if n % 3 = 0 then GameState.PetyaWin else GameState.VasyaWin

-- Theorem: Petya will win the game if both play optimally
theorem Petya_wins_optimally (n : ℕ) (h1 : n = 2021) (h2 : game_rule n) : determine_winner n = GameState.PetyaWin :=
by {
  sorry
}

end Petya_wins_optimally_l2316_231605


namespace combination_8_5_l2316_231659

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def combination (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem combination_8_5 : combination 8 5 = 56 := by
  sorry

end combination_8_5_l2316_231659


namespace smallest_m_l2316_231608

noncomputable def f (x : ℝ) : ℝ := sorry

theorem smallest_m (f : ℝ → ℝ) (x y : ℝ) (hx : 0 ≤ x) (hy : y ≤ 1) (h_eq : f 0 = f 1) 
(h_lt : forall x y : ℝ, 0 ≤ x → x ≤ 1 → 0 ≤ y → y ≤ 1 → |f x - f y| < |x - y|): 
|f x - f y| < 1 / 2 := 
sorry

end smallest_m_l2316_231608


namespace increasing_interval_f_l2316_231662

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x - 3)

theorem increasing_interval_f :
  (∀ x, x ∈ Set.Ioi 3 → f x ∈ Set.Ioi 3) := sorry

end increasing_interval_f_l2316_231662


namespace inverse_g_167_is_2_l2316_231687

def g (x : ℝ) := 5 * x^5 + 7

theorem inverse_g_167_is_2 : g⁻¹' {167} = {2} := by
  sorry

end inverse_g_167_is_2_l2316_231687


namespace sum_last_two_digits_of_x2012_l2316_231673

def sequence_defined (x : ℕ → ℕ) : Prop :=
  (x 1 = 5 ∨ x 1 = 7) ∧ ∀ k ≥ 1, (x (k+1) = 5^(x k) ∨ x (k+1) = 7^(x k))

def last_two_digits (n : ℕ) : ℕ :=
  n % 100

def possible_values : List ℕ :=
  [25, 7, 43]

theorem sum_last_two_digits_of_x2012 {x : ℕ → ℕ} (h : sequence_defined x) :
  List.sum (List.map last_two_digits [25, 7, 43]) = 75 :=
  by
    sorry

end sum_last_two_digits_of_x2012_l2316_231673


namespace shortest_path_from_A_to_D_not_inside_circle_l2316_231644

noncomputable def shortest_path_length : ℝ :=
  let A : ℝ × ℝ := (0, 0)
  let D : ℝ × ℝ := (18, 24)
  let O : ℝ × ℝ := (9, 12)
  let r : ℝ := 15
  15 * Real.pi

theorem shortest_path_from_A_to_D_not_inside_circle :
  let A := (0, 0)
  let D := (18, 24)
  let O := (9, 12)
  let r := 15
  shortest_path_length = 15 * Real.pi := 
by
  sorry

end shortest_path_from_A_to_D_not_inside_circle_l2316_231644


namespace room_length_l2316_231656

theorem room_length (L : ℕ) (h : 72 * L + 918 = 2718) : L = 25 := by
  sorry

end room_length_l2316_231656


namespace inverse_function_property_l2316_231649

noncomputable def f (a x : ℝ) : ℝ := (x - a) * |x|

theorem inverse_function_property (a : ℝ) :
  (∃ g : ℝ → ℝ, ∀ x : ℝ, f a (g x) = x) ↔ a = 0 :=
by sorry

end inverse_function_property_l2316_231649


namespace find_doodads_produced_in_four_hours_l2316_231668

theorem find_doodads_produced_in_four_hours :
  ∃ (n : ℕ),
    (∀ (workers hours widgets doodads : ℕ),
      (workers = 150 ∧ hours = 2 ∧ widgets = 800 ∧ doodads = 500) ∨
      (workers = 100 ∧ hours = 3 ∧ widgets = 750 ∧ doodads = 600) ∨
      (workers = 80  ∧ hours = 4 ∧ widgets = 480 ∧ doodads = n)
    ) → n = 640 :=
sorry

end find_doodads_produced_in_four_hours_l2316_231668


namespace percentage_difference_l2316_231622

theorem percentage_difference :
  let x := 50
  let y := 30
  let p1 := 60
  let p2 := 30
  (p1 / 100 * x) - (p2 / 100 * y) = 21 :=
by
  sorry

end percentage_difference_l2316_231622


namespace area_of_rectangle_with_diagonal_length_l2316_231603

variable (x : ℝ)

def rectangle_area_given_diagonal_length (x : ℝ) : Prop :=
  ∃ (w l : ℝ), l = 3 * w ∧ w^2 + l^2 = x^2 ∧ (w * l = (3 / 10) * x^2)

theorem area_of_rectangle_with_diagonal_length (x : ℝ) :
  rectangle_area_given_diagonal_length x :=
sorry

end area_of_rectangle_with_diagonal_length_l2316_231603


namespace range_of_k_l2316_231629

noncomputable def f (k x : ℝ) := (k * x + 7) / (k * x^2 + 4 * k * x + 3)

theorem range_of_k (k : ℝ) : (∀ x : ℝ, k * x^2 + 4 * k * x + 3 ≠ 0) ↔ 0 ≤ k ∧ k < 3 / 4 :=
by
  sorry

end range_of_k_l2316_231629


namespace find_d_l2316_231680

variables {x y z k d : ℝ}
variables {a : ℝ} (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0)
variables (h_ap : x * (y - z) + y * (z - x) + z * (x - y) = 0)
variables (h_sum : x * (y - z) + (y * (z - x) + d) + (z * (x - y) + 2 * d) = k)

theorem find_d : d = k / 3 :=
sorry

end find_d_l2316_231680


namespace one_kid_six_whiteboards_l2316_231632

theorem one_kid_six_whiteboards (k: ℝ) (b1 b2: ℝ) (t1 t2: ℝ) 
  (hk: k = 1) (hb1: b1 = 3) (hb2: b2 = 6) 
  (ht1: t1 = 20) 
  (H: 4 * t1 / b1 = t2 / b2) : 
  t2 = 160 := 
by
  -- provide the proof here
  sorry

end one_kid_six_whiteboards_l2316_231632


namespace students_behind_yoongi_l2316_231640

theorem students_behind_yoongi (n k : ℕ) (hn : n = 30) (hk : k = 20) : n - (k + 1) = 9 := by
  sorry

end students_behind_yoongi_l2316_231640


namespace number_of_cars_in_train_l2316_231652

theorem number_of_cars_in_train
  (constant_speed : Prop)
  (cars_in_12_seconds : ℕ)
  (time_to_clear : ℕ)
  (cars_per_second : ℕ → ℕ → ℚ)
  (total_time_seconds : ℕ) :
  cars_in_12_seconds = 8 →
  time_to_clear = 180 →
  cars_per_second cars_in_12_seconds 12 = 2 / 3 →
  total_time_seconds = 180 →
  cars_per_second cars_in_12_seconds 12 * total_time_seconds = 120 :=
by
  sorry

end number_of_cars_in_train_l2316_231652


namespace minimum_value_func1_minimum_value_func2_l2316_231646

-- Problem (1): 
theorem minimum_value_func1 (x : ℝ) (h : x > -1) : 
  (x + 4 / (x + 1) + 6) ≥ 9 :=
sorry

-- Problem (2): 
theorem minimum_value_func2 (x : ℝ) (h : x > 1) : 
  (x^2 + 8) / (x - 1) ≥ 8 :=
sorry

end minimum_value_func1_minimum_value_func2_l2316_231646


namespace maximum_xyz_l2316_231669

-- Given conditions
variables {x y z : ℝ}

-- Lean 4 statement with the conditions
theorem maximum_xyz (h₁ : x * y + 2 * z = (x + z) * (y + z))
  (h₂ : x + y + 2 * z = 2)
  (h₃ : 0 < x) (h₄ : 0 < y) (h₅ : 0 < z) :
  xyz = 0 :=
sorry

end maximum_xyz_l2316_231669


namespace prob_A_championship_win_is_correct_expectation_X_is_correct_distribution_X_is_correct_l2316_231691

/-- Let us define the probabilities for school A winning the events -/
def prob_A_wins_event_1 : ℝ := 0.5
def prob_A_wins_event_2 : ℝ := 0.4
def prob_A_wins_event_3 : ℝ := 0.8

/-- The total probability of school A winning the championship -/
noncomputable def prob_A_championship_wins : ℝ :=
  prob_A_wins_event_1 * prob_A_wins_event_2 * prob_A_wins_event_3 +   -- All three events
  (prob_A_wins_event_1 * prob_A_wins_event_2 * (1 - prob_A_wins_event_3) + -- First two events
   prob_A_wins_event_1 * (1 - prob_A_wins_event_2) * prob_A_wins_event_3 + -- First and third event
   (1 - prob_A_wins_event_1) * prob_A_wins_event_2 * prob_A_wins_event_3)  -- Second and third events

/-- The distribution for school B's scores -/
def score_dist_B : List (ℕ × ℝ) :=
  [(0, 0.16), (10, 0.44), (20, 0.34), (30, 0.06)]

/-- The expectation of X (total score of school B) -/
noncomputable def expectation_X : ℝ :=
  0 * 0.16 + 10 * 0.44 + 20 * 0.34 + 30 * 0.06

/-- The proofs for the derived results -/
theorem prob_A_championship_win_is_correct : prob_A_championship_wins = 0.6 := sorry

theorem expectation_X_is_correct : expectation_X = 13 := sorry

theorem distribution_X_is_correct :
  score_dist_B = [(0, 0.16), (10, 0.44), (20, 0.34), (30, 0.06)] := sorry

end prob_A_championship_win_is_correct_expectation_X_is_correct_distribution_X_is_correct_l2316_231691


namespace arithmetic_mean_of_multiples_of_6_l2316_231671

/-- The smallest three-digit multiple of 6 is 102. -/
def smallest_multiple_of_6 : ℕ := 102

/-- The largest three-digit multiple of 6 is 996. -/
def largest_multiple_of_6 : ℕ := 996

/-- The common difference in the arithmetic sequence of multiples of 6 is 6. -/
def common_difference_of_sequence : ℕ := 6

/-- The number of terms in the arithmetic sequence of three-digit multiples of 6. -/
def number_of_terms : ℕ := (largest_multiple_of_6 - smallest_multiple_of_6) / common_difference_of_sequence + 1

/-- The sum of the arithmetic sequence of three-digit multiples of 6. -/
def sum_of_sequence : ℕ := number_of_terms * (smallest_multiple_of_6 + largest_multiple_of_6) / 2

/-- The arithmetic mean of all positive three-digit multiples of 6 is 549. -/
theorem arithmetic_mean_of_multiples_of_6 : 
  let mean := sum_of_sequence / number_of_terms
  mean = 549 :=
by
  sorry

end arithmetic_mean_of_multiples_of_6_l2316_231671


namespace original_price_of_shoes_l2316_231631

theorem original_price_of_shoes (P : ℝ) (h : 0.08 * P = 16) : P = 200 :=
sorry

end original_price_of_shoes_l2316_231631


namespace find_sum_of_x_and_y_l2316_231699

theorem find_sum_of_x_and_y (x y : ℝ) 
  (h1 : (x-1)^3 + 1997*(x-1) = -1)
  (h2 : (y-1)^3 + 1997*(y-1) = 1) :
  x + y = 2 :=
sorry

end find_sum_of_x_and_y_l2316_231699


namespace min_value_3x_4y_l2316_231609

open Real

theorem min_value_3x_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 5 * x * y) : 3 * x + 4 * y = 5 :=
by
  sorry

end min_value_3x_4y_l2316_231609


namespace vincent_back_to_A_after_5_min_p_plus_q_computation_l2316_231695

def probability (n : ℕ) : ℚ :=
  if n = 0 then 1
  else 1 / 4 * (1 - probability (n - 1))

theorem vincent_back_to_A_after_5_min : 
  probability 5 = 51 / 256 :=
by sorry

theorem p_plus_q_computation :
  51 + 256 = 307 :=
by linarith

end vincent_back_to_A_after_5_min_p_plus_q_computation_l2316_231695
