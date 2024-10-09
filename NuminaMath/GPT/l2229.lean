import Mathlib

namespace find_positive_product_l2229_222973

variable (a b c d e f : ℝ)

-- Define the condition that exactly one of the products is positive
def exactly_one_positive (p1 p2 p3 p4 p5 : ℝ) : Prop :=
  (p1 > 0 ∧ p2 < 0 ∧ p3 < 0 ∧ p4 < 0 ∧ p5 < 0) ∨
  (p1 < 0 ∧ p2 > 0 ∧ p3 < 0 ∧ p4 < 0 ∧ p5 < 0) ∨
  (p1 < 0 ∧ p2 < 0 ∧ p3 > 0 ∧ p4 < 0 ∧ p5 < 0) ∨
  (p1 < 0 ∧ p2 < 0 ∧ p3 < 0 ∧ p4 > 0 ∧ p5 < 0) ∨
  (p1 < 0 ∧ p2 < 0 ∧ p3 < 0 ∧ p4 < 0 ∧ p5 > 0)

theorem find_positive_product (h : a ≠ 0) (h' : b ≠ 0) (h'' : c ≠ 0) (h''' : d ≠ 0) (h'''' : e ≠ 0) (h''''' : f ≠ 0) 
  (exactly_one : exactly_one_positive (a * c * d) (a * c * e) (b * d * e) (b * d * f) (b * e * f)) :
  b * d * e > 0 :=
sorry

end find_positive_product_l2229_222973


namespace problem1_correct_problem2_correct_l2229_222953

-- Definition for Problem 1
def problem1 (a b c d : ℚ) : ℚ :=
  (a - b + c) * d

-- Statement for Problem 1
theorem problem1_correct : problem1 (1/6) (5/7) (2/3) (-42) = -5 :=
by
  sorry

-- Definitions for Problem 2
def problem2 (a b c d : ℚ) : ℚ :=
  (-a^2 + b^2 * c - d^2 / |d|)

-- Statement for Problem 2
theorem problem2_correct : problem2 (-2) (-3) (-2/3) 4 = -14 :=
by
  sorry

end problem1_correct_problem2_correct_l2229_222953


namespace least_integer_gt_sqrt_450_l2229_222930

theorem least_integer_gt_sqrt_450 : 
  ∀ n : ℕ, (∃ m : ℕ, m * m ≤ 450 ∧ 450 < (m + 1) * (m + 1)) → n = 22 :=
by
  sorry

end least_integer_gt_sqrt_450_l2229_222930


namespace angle_AOC_is_45_or_15_l2229_222976

theorem angle_AOC_is_45_or_15 (A O B C : Type) (α β γ : ℝ) 
  (h1 : α = 30) (h2 : β = 15) : γ = 45 ∨ γ = 15 :=
sorry

end angle_AOC_is_45_or_15_l2229_222976


namespace problem_equivalent_to_l2229_222966

theorem problem_equivalent_to (x : ℝ)
  (A : x^2 = 5*x - 6 ↔ x = 2 ∨ x = 3)
  (B : x^2 - 5*x + 6 = 0 ↔ x = 2 ∨ x = 3)
  (C : x = x + 1 ↔ false)
  (D : x^2 - 5*x + 7 = 1 ↔ x = 2 ∨ x = 3)
  (E : x^2 - 1 = 5*x - 7 ↔ x = 2 ∨ x = 3) :
  ¬ (x = x + 1) :=
by sorry

end problem_equivalent_to_l2229_222966


namespace total_percentage_increase_l2229_222965

noncomputable def initialSalary : ℝ := 60
noncomputable def firstRaisePercent : ℝ := 10
noncomputable def secondRaisePercent : ℝ := 15
noncomputable def promotionRaisePercent : ℝ := 20

theorem total_percentage_increase :
  let finalSalary := initialSalary * (1 + firstRaisePercent / 100) * (1 + secondRaisePercent / 100) * (1 + promotionRaisePercent / 100)
  let increase := finalSalary - initialSalary
  let percentageIncrease := (increase / initialSalary) * 100
  percentageIncrease = 51.8 := by
  sorry

end total_percentage_increase_l2229_222965


namespace hundredth_odd_positive_integer_equals_199_even_integer_following_199_equals_200_l2229_222949

theorem hundredth_odd_positive_integer_equals_199 : (2 * 100 - 1 = 199) :=
by {
  sorry
}

theorem even_integer_following_199_equals_200 : (199 + 1 = 200) :=
by {
  sorry
}

end hundredth_odd_positive_integer_equals_199_even_integer_following_199_equals_200_l2229_222949


namespace scrabble_score_l2229_222999

-- Definitions derived from conditions
def value_first_and_third : ℕ := 1
def value_middle : ℕ := 8
def multiplier : ℕ := 3

-- Prove the total points earned by Jeremy
theorem scrabble_score : (value_first_and_third * 2 + value_middle) * multiplier = 30 :=
by
  sorry

end scrabble_score_l2229_222999


namespace fraction_to_terminating_decimal_l2229_222990

theorem fraction_to_terminating_decimal : (49 : ℚ) / 160 = 0.30625 := 
sorry

end fraction_to_terminating_decimal_l2229_222990


namespace unique_sequence_l2229_222944

theorem unique_sequence (n : ℕ) (h : 1 < n)
  (x : Fin (n-1) → ℕ)
  (h_pos : ∀ i, 0 < x i)
  (h_incr : ∀ i j, i < j → x i < x j)
  (h_symm : ∀ i : Fin (n-1), x i + x ⟨n - 2 - i.val, sorry⟩ = 2 * n)
  (h_sum : ∀ i j : Fin (n-1), x i + x j < 2 * n → ∃ k : Fin (n-1), x i + x j = x k) :
  ∀ i : Fin (n-1), x i = 2 * (i + 1) :=
by
  sorry

end unique_sequence_l2229_222944


namespace calculate_3_diamond_4_l2229_222962

-- Define the operations
def op (a b : ℝ) : ℝ := a^2 + 2 * a * b
def diamond (a b : ℝ) : ℝ := 4 * a + 6 * b - op a b

-- State the theorem
theorem calculate_3_diamond_4 : diamond 3 4 = 3 := by
  sorry

end calculate_3_diamond_4_l2229_222962


namespace solve_f_eq_x_l2229_222902

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f_inv : ℝ → ℝ := sorry

axiom f_inv_domain : ∀ (x : ℝ), 0 ≤ x ∧ x < 1 → 1 ≤ f_inv x ∧ f_inv x < 2
axiom f_inv_range : ∀ (x : ℝ), 2 < x ∧ x ≤ 4 → 0 ≤ f_inv x ∧ f_inv x < 1
-- Assumption that f is invertible on [0, 3]
axiom f_inv_exists : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 3 → ∃ y : ℝ, f y = x

theorem solve_f_eq_x : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ f x = x → x = 2 :=
by
  sorry

end solve_f_eq_x_l2229_222902


namespace sum_of_digits_is_15_l2229_222987

theorem sum_of_digits_is_15
  (A B C D E : ℕ) 
  (h_distinct: A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ C ≠ D ∧ C ≠ E ∧ D ≠ E)
  (h_digits: A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10)
  (h_divisible_by_9: (A * 10000 + B * 1000 + C * 100 + D * 10 + E) % 9 = 0) 
  : A + B + C + D + E = 15 := 
sorry

end sum_of_digits_is_15_l2229_222987


namespace harmonic_mean_2_3_6_l2229_222936

def harmonic_mean (a b c : ℕ) : ℚ := 3 / ((1 / a) + (1 / b) + (1 / c))

theorem harmonic_mean_2_3_6 : harmonic_mean 2 3 6 = 3 := 
by
  sorry

end harmonic_mean_2_3_6_l2229_222936


namespace ch4_contains_most_atoms_l2229_222972

def molecule_atoms (molecule : String) : Nat :=
  match molecule with
  | "O₂"   => 2
  | "NH₃"  => 4
  | "CO"   => 2
  | "CH₄"  => 5
  | _      => 0

theorem ch4_contains_most_atoms :
  ∀ (a b c d : Nat), 
  a = molecule_atoms "O₂" →
  b = molecule_atoms "NH₃" →
  c = molecule_atoms "CO" →
  d = molecule_atoms "CH₄" →
  d > a ∧ d > b ∧ d > c :=
by
  intros
  sorry

end ch4_contains_most_atoms_l2229_222972


namespace smallest_possible_z_l2229_222992

theorem smallest_possible_z :
  ∃ (z : ℕ), (z = 6) ∧ 
  ∃ (u w x y : ℕ), u < w ∧ w < x ∧ x < y ∧ y < z ∧ 
  u.succ = w ∧ w.succ = x ∧ x.succ = y ∧ y.succ = z ∧ 
  u^3 + w^3 + x^3 + y^3 = z^3 :=
by
  use 6
  sorry

end smallest_possible_z_l2229_222992


namespace multiply_exp_result_l2229_222977

theorem multiply_exp_result : 121 * (5 ^ 4) = 75625 :=
by
  sorry

end multiply_exp_result_l2229_222977


namespace fraction_of_students_older_than_4_years_l2229_222998

-- Definitions based on conditions
def total_students := 50
def students_younger_than_3 := 20
def students_not_between_3_and_4 := 25
def students_older_than_4 := students_not_between_3_and_4 - students_younger_than_3
def fraction_older_than_4 := students_older_than_4 / total_students

-- Theorem to prove the desired fraction
theorem fraction_of_students_older_than_4_years : fraction_older_than_4 = 1/10 :=
by
  sorry

end fraction_of_students_older_than_4_years_l2229_222998


namespace an_squared_diff_consec_cubes_l2229_222985

theorem an_squared_diff_consec_cubes (a b : ℕ → ℤ) (n : ℕ) :
  a 1 = 1 → b 1 = 0 →
  (∀ n ≥ 1, a (n + 1) = 7 * (a n) + 12 * (b n) + 6) →
  (∀ n ≥ 1, b (n + 1) = 4 * (a n) + 7 * (b n) + 3) →
  a n ^ 2 = (b n + 1) ^ 3 - (b n) ^ 3 :=
by
  sorry

end an_squared_diff_consec_cubes_l2229_222985


namespace parity_of_f_min_value_of_f_min_value_of_f_l2229_222929

open Real

def f (a x : ℝ) := x^2 + abs (x - a) + 1

theorem parity_of_f (a : ℝ) :
  (∀ x : ℝ, f 0 x = f 0 (-x)) ∧ (∀ x : ℝ, f a x ≠ f a (-x) ∧ f a x ≠ -f a x) ↔ a = 0 :=
by sorry

theorem min_value_of_f (a : ℝ) (h : a ≤ -1/2) : 
  ∀ x : ℝ, x ≥ a → f a x ≥ f a (-1/2) :=
by sorry

theorem min_value_of_f' (a : ℝ) (h : -1/2 < a) : 
  ∀ x : ℝ, x ≥ a → f a x ≥ f a a :=
by sorry

end parity_of_f_min_value_of_f_min_value_of_f_l2229_222929


namespace original_amount_of_cooking_oil_l2229_222983

theorem original_amount_of_cooking_oil (X : ℝ) (H : (2 / 5 * X + 300) + (1 / 2 * (X - (2 / 5 * X + 300)) - 200) + 800 = X) : X = 2500 :=
by simp at H; linarith

end original_amount_of_cooking_oil_l2229_222983


namespace product_of_two_numbers_l2229_222958

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 + y^2 = 120) :
  x * y = -20 :=
sorry

end product_of_two_numbers_l2229_222958


namespace quadratic_inequality_solution_set_l2229_222961

variable (a b c : ℝ)

theorem quadratic_inequality_solution_set (h1 : ∀ x : ℝ, ax^2 + bx + c > 0 → (-1 / 3 < x ∧ x < 2)) :
  ∀ x : ℝ, cx^2 + bx + a < 0 → (-3 < x ∧ x < 1 / 2) :=
by
  sorry

end quadratic_inequality_solution_set_l2229_222961


namespace minimum_value_expr_l2229_222959

noncomputable def expr (x y z : ℝ) : ℝ := 
  3 * x^2 + 2 * x * y + 3 * y^2 + 2 * y * z + 3 * z^2 - 3 * x + 3 * y - 3 * z + 9

theorem minimum_value_expr : 
  ∃ (x y z : ℝ), ∀ (a b c : ℝ), expr a b c ≥ expr x y z ∧ expr x y z = 3/2 :=
sorry

end minimum_value_expr_l2229_222959


namespace non_congruent_triangles_perimeter_18_l2229_222993

theorem non_congruent_triangles_perimeter_18 :
  ∃ (triangles : Finset (Finset ℕ)), triangles.card = 11 ∧
  (∀ t ∈ triangles, t.card = 3 ∧ (∃ a b c : ℕ, t = {a, b, c} ∧ a + b + c = 18 ∧ a + b > c ∧ a + c > b ∧ b + c > a)) :=
sorry

end non_congruent_triangles_perimeter_18_l2229_222993


namespace surface_area_of_interior_of_box_l2229_222933

-- Definitions from conditions in a)
def length : ℕ := 25
def width : ℕ := 40
def cut_side : ℕ := 4

-- The proof statement we need to prove, using the correct answer from b)
theorem surface_area_of_interior_of_box : 
  (length - 2 * cut_side) * (width - 2 * cut_side) + 2 * (cut_side * (length + width - 2 * cut_side)) = 936 :=
by
  sorry

end surface_area_of_interior_of_box_l2229_222933


namespace market_value_of_stock_l2229_222940

def face_value : ℝ := 100
def dividend_per_share : ℝ := 0.10 * face_value
def yield : ℝ := 0.08

theorem market_value_of_stock : (dividend_per_share / yield) = 125 := by
  -- Proof not required
  sorry

end market_value_of_stock_l2229_222940


namespace math_problem_solution_l2229_222922

theorem math_problem_solution : ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ x^y - 1 = y^x ∧ 2*x^y = y^x + 5 ∧ x = 2 ∧ y = 2 :=
by {
  sorry
}

end math_problem_solution_l2229_222922


namespace barbara_wins_l2229_222939

theorem barbara_wins (n : ℕ) (h : n = 15) (num_winning_sequences : ℕ) :
  num_winning_sequences = 8320 :=
sorry

end barbara_wins_l2229_222939


namespace find_incorrect_value_l2229_222942

variable (k b : ℝ)

-- Linear function definition
def linear_function (x : ℝ) : ℝ := k * x + b

-- Given points
theorem find_incorrect_value (h₁ : linear_function k b (-1) = 3)
                             (h₂ : linear_function k b 0 = 2)
                             (h₃ : linear_function k b 1 = 1)
                             (h₄ : linear_function k b 2 = 0)
                             (h₅ : linear_function k b 3 = -2) :
                             (∃ x y, linear_function k b x ≠ y) := by
  sorry

end find_incorrect_value_l2229_222942


namespace point_A_on_x_axis_l2229_222945

def point_A : ℝ × ℝ := (-2, 0)

theorem point_A_on_x_axis : point_A.snd = 0 :=
by
  unfold point_A
  sorry

end point_A_on_x_axis_l2229_222945


namespace no_nonconstant_poly_prime_for_all_l2229_222907

open Polynomial

theorem no_nonconstant_poly_prime_for_all (f : Polynomial ℤ) (h : ∀ n : ℕ, Prime (f.eval (n : ℤ))) :
  ∃ c : ℤ, f = Polynomial.C c :=
sorry

end no_nonconstant_poly_prime_for_all_l2229_222907


namespace baked_goods_not_eaten_l2229_222904

theorem baked_goods_not_eaten : 
  let cookies_initial := 200
  let brownies_initial := 150
  let cupcakes_initial := 100
  
  let cookies_after_wife := cookies_initial - 0.30 * cookies_initial
  let brownies_after_wife := brownies_initial - 0.20 * brownies_initial
  let cupcakes_after_wife := cupcakes_initial / 2
  
  let cookies_after_daughter := cookies_after_wife - 40
  let brownies_after_daughter := brownies_after_wife - 0.15 * brownies_after_wife
  
  let cookies_after_friend := cookies_after_daughter - (cookies_after_daughter / 4)
  let brownies_after_friend := brownies_after_daughter - 0.10 * brownies_after_daughter
  let cupcakes_after_friend := cupcakes_after_wife - 10
  
  let cookies_after_other_friend := cookies_after_friend - 0.05 * cookies_after_friend
  let brownies_after_other_friend := brownies_after_friend - 0.05 * brownies_after_friend
  let cupcakes_after_other_friend := cupcakes_after_friend - 5
  
  let cookies_after_javier := cookies_after_other_friend / 2
  let brownies_after_javier := brownies_after_other_friend / 2
  let cupcakes_after_javier := cupcakes_after_other_friend / 2
  
  let total_remaining := cookies_after_javier + brownies_after_javier + cupcakes_after_javier
  total_remaining = 98 := by
{
  sorry
}

end baked_goods_not_eaten_l2229_222904


namespace simple_interest_rate_l2229_222989

theorem simple_interest_rate (P : ℝ) (r : ℝ) (T : ℝ) (SI : ℝ)
  (h1 : SI = P / 5)
  (h2 : T = 10)
  (h3 : SI = (P * r * T) / 100) :
  r = 2 :=
by
  sorry

end simple_interest_rate_l2229_222989


namespace mary_prevents_pat_l2229_222916

noncomputable def smallest_initial_integer (N: ℕ) : Prop :=
  N > 2017 ∧ 
  ∀ x, ∃ n: ℕ, 
  (x = N + n * 2018 → x % 2018 ≠ 0 ∧
   (2017 * x + 2) % 2018 ≠ 0 ∧
   (2017 * x + 2021) % 2018 ≠ 0)

theorem mary_prevents_pat (N : ℕ) : smallest_initial_integer N → N = 2022 :=
sorry

end mary_prevents_pat_l2229_222916


namespace polynomial_function_correct_l2229_222947

theorem polynomial_function_correct :
  ∀ (f : ℝ → ℝ),
  (∀ (x : ℝ), f (x^2 + 1) = x^4 + 5 * x^2 + 3) →
  ∀ (x : ℝ), f (x^2 - 1) = x^4 + x^2 - 3 :=
by
  sorry

end polynomial_function_correct_l2229_222947


namespace henry_earnings_correct_l2229_222920

-- Define constants for the amounts earned per task
def earn_per_lawn : Nat := 5
def earn_per_leaves : Nat := 10
def earn_per_driveway : Nat := 15

-- Define constants for the number of tasks he actually managed to do
def lawns_mowed : Nat := 5
def leaves_raked : Nat := 3
def driveways_shoveled : Nat := 2

-- Define the expected total earnings calculation
def expected_earnings : Nat :=
  (lawns_mowed * earn_per_lawn) +
  (leaves_raked * earn_per_leaves) +
  (driveways_shoveled * earn_per_driveway)

-- State the theorem that the total earnings are 85 dollars.
theorem henry_earnings_correct : expected_earnings = 85 :=
by
  sorry

end henry_earnings_correct_l2229_222920


namespace inletRate_is_3_l2229_222971

def volumeTank (v_cubic_feet : ℕ) : ℕ :=
  1728 * v_cubic_feet

def outletRate1 : ℕ := 9 -- rate of first outlet in cubic inches/min
def outletRate2 : ℕ := 6 -- rate of second outlet in cubic inches/min
def tankVolume : ℕ := volumeTank 30 -- tank volume in cubic inches
def minutesToEmpty : ℕ := 4320 -- time to empty the tank in minutes

def effectiveRate (inletRate : ℕ) : ℕ :=
  outletRate1 + outletRate2 - inletRate

theorem inletRate_is_3 : (15 - 3) * minutesToEmpty = tankVolume :=
  by simp [outletRate1, outletRate2, tankVolume, minutesToEmpty]; sorry

end inletRate_is_3_l2229_222971


namespace grassy_pathway_area_correct_l2229_222974

-- Define the dimensions of the plot and the pathway width
def length_plot : ℝ := 15
def width_plot : ℝ := 10
def width_pathway : ℝ := 2

-- Define the required areas
def total_area : ℝ := (length_plot + 2 * width_pathway) * (width_plot + 2 * width_pathway)
def plot_area : ℝ := length_plot * width_plot
def grassy_pathway_area : ℝ := total_area - plot_area

-- Prove that the area of the grassy pathway is 116 m²
theorem grassy_pathway_area_correct : grassy_pathway_area = 116 := by
  sorry

end grassy_pathway_area_correct_l2229_222974


namespace integer_values_between_fractions_l2229_222984

theorem integer_values_between_fractions :
  let a := 4 / (Real.sqrt 3 + Real.sqrt 2)
  let b := 4 / (Real.sqrt 5 - Real.sqrt 3)
  ((⌊b⌋ - ⌈a⌉) + 1) = 6 :=
by sorry

end integer_values_between_fractions_l2229_222984


namespace find_tangent_parallel_to_x_axis_l2229_222950

theorem find_tangent_parallel_to_x_axis :
  ∃ (x y : ℝ), y = x^2 - 3 * x ∧ (2 * x - 3 = 0) ∧ (x = 3 / 2) ∧ (y = -9 / 4) := 
by
  sorry

end find_tangent_parallel_to_x_axis_l2229_222950


namespace paths_mat8_l2229_222903

-- Define variables
def grid := [
  ["M", "A", "M", "A", "M"],
  ["A", "T", "A", "T", "A"],
  ["M", "A", "M", "A", "M"],
  ["A", "T", "A", "T", "A"],
  ["M", "A", "M", "A", "M"]
]

def is_adjacent (x1 y1 x2 y2 : Nat): Bool :=
  (x1 = x2 ∧ (y1 = y2 + 1 ∨ y1 = y2 - 1)) ∨ (y1 = y2 ∧ (x1 = x2 + 1 ∨ x1 = x2 - 1))

def count_paths (grid: List (List String)): Nat :=
  -- implementation to count number of paths
  4 * 4 * 2

theorem paths_mat8 (grid: List (List String)): count_paths grid = 32 := by
  sorry

end paths_mat8_l2229_222903


namespace find_multiple_l2229_222982

theorem find_multiple (n : ℕ) (h₁ : n = 5) (m : ℕ) (h₂ : 7 * n - 15 > m * n) : m = 3 :=
by
  sorry

end find_multiple_l2229_222982


namespace smallest_positive_integer_l2229_222901

theorem smallest_positive_integer (n : ℕ) :
  (n % 45 = 0 ∧ n % 60 = 0 ∧ n % 25 ≠ 0 ↔ n = 180) :=
sorry

end smallest_positive_integer_l2229_222901


namespace incorrect_statement_C_l2229_222911

theorem incorrect_statement_C :
  (∀ (b h : ℝ), b > 0 → h > 0 → 2 * (b * h) = (2 * b) * h) ∧
  (∀ (r h : ℝ), r > 0 → h > 0 → 2 * (π * r^2 * h) = π * r^2 * (2 * h)) ∧
  (∀ (a : ℝ), a > 0 → 4 * (a^3) ≠ (2 * a)^3) ∧
  (∀ (a b : ℚ), b ≠ 0 → a / (2 * b) ≠ (a / 2) / b) ∧
  (∀ (x : ℝ), x < 0 → 2 * x < x) :=
by
  sorry

end incorrect_statement_C_l2229_222911


namespace train_crossing_time_correct_l2229_222946

noncomputable def train_crossing_time (speed_kmph : ℕ) (length_m : ℕ) (train_dir_opposite : Bool) : ℕ :=
  if train_dir_opposite then
    let speed_mps := speed_kmph * 1000 / 3600
    let relative_speed := speed_mps + speed_mps
    let total_distance := length_m + length_m
    total_distance / relative_speed
  else 0

theorem train_crossing_time_correct :
  train_crossing_time 54 120 true = 8 :=
by
  sorry

end train_crossing_time_correct_l2229_222946


namespace polygon_sides_l2229_222969

theorem polygon_sides (n : ℕ) 
  (h1 : sum_interior_angles = 180 * (n - 2))
  (h2 : sum_exterior_angles = 360)
  (h3 : sum_interior_angles = 3 * sum_exterior_angles) : 
  n = 8 :=
by
  sorry

end polygon_sides_l2229_222969


namespace find_x_l2229_222938

theorem find_x (x : ℤ) (A : Set ℤ) (B : Set ℤ) (hA : A = {1, 4, x}) (hB : B = {1, 2 * x, x ^ 2}) (hinter : A ∩ B = {4, 1}) : x = -2 :=
sorry

end find_x_l2229_222938


namespace fencing_required_l2229_222960

theorem fencing_required
  (L : ℝ) (W : ℝ) (A : ℝ) (hL : L = 20) (hA : A = 80) (hW : A = L * W) :
  (L + 2 * W) = 28 :=
by {
  sorry
}

end fencing_required_l2229_222960


namespace evaluate_expression_l2229_222921

theorem evaluate_expression : 40 + 5 * 12 / (180 / 3) = 41 :=
by
  -- Proof goes here
  sorry

end evaluate_expression_l2229_222921


namespace remainder_mul_mod_l2229_222937

theorem remainder_mul_mod (a b n : ℕ) (h₁ : a ≡ 3 [MOD n]) (h₂ : b ≡ 150 [MOD n]) (n_eq : n = 400) : 
  (a * b) % n = 50 :=
by 
  sorry

end remainder_mul_mod_l2229_222937


namespace total_bones_in_graveyard_l2229_222924

def total_skeletons : ℕ := 20

def adult_women : ℕ := total_skeletons / 2
def adult_men : ℕ := (total_skeletons - adult_women) / 2
def children : ℕ := (total_skeletons - adult_women) / 2

def bones_adult_woman : ℕ := 20
def bones_adult_man : ℕ := bones_adult_woman + 5
def bones_child : ℕ := bones_adult_woman / 2

def bones_graveyard : ℕ :=
  (adult_women * bones_adult_woman) +
  (adult_men * bones_adult_man) +
  (children * bones_child)

theorem total_bones_in_graveyard :
  bones_graveyard = 375 :=
sorry

end total_bones_in_graveyard_l2229_222924


namespace johns_pace_l2229_222964

variable {J : ℝ} -- John's pace during his final push

theorem johns_pace
  (steve_speed : ℝ := 3.8)
  (initial_gap : ℝ := 15)
  (finish_gap : ℝ := 2)
  (time : ℝ := 42.5)
  (steve_covered : ℝ := steve_speed * time)
  (john_covered : ℝ := steve_covered + initial_gap + finish_gap)
  (johns_pace_equation : J * time = john_covered) :
  J = 4.188 :=
by
  sorry

end johns_pace_l2229_222964


namespace a_runs_4_times_faster_than_b_l2229_222951

theorem a_runs_4_times_faster_than_b (v_A v_B : ℝ) (k : ℝ) 
    (h1 : v_A = k * v_B) 
    (h2 : 92 / v_A = 23 / v_B) : 
    k = 4 := 
sorry

end a_runs_4_times_faster_than_b_l2229_222951


namespace complement_P_relative_to_U_l2229_222941

variable (U : Set ℝ) (P : Set ℝ)

theorem complement_P_relative_to_U (hU : U = Set.univ) (hP : P = {x : ℝ | x < 1}) : 
  U \ P = {x : ℝ | x ≥ 1} := by
  sorry

end complement_P_relative_to_U_l2229_222941


namespace percy_swimming_hours_l2229_222963

theorem percy_swimming_hours :
  let weekday_hours_per_day := 2
  let weekdays := 5
  let weekend_hours := 3
  let weeks := 4
  let total_weekday_hours_per_week := weekday_hours_per_day * weekdays
  let total_weekend_hours_per_week := weekend_hours
  let total_hours_per_week := total_weekday_hours_per_week + total_weekend_hours_per_week
  let total_hours_over_weeks := total_hours_per_week * weeks
  total_hours_over_weeks = 64 :=
by
  sorry

end percy_swimming_hours_l2229_222963


namespace any_positive_integer_can_be_expressed_l2229_222927

theorem any_positive_integer_can_be_expressed 
  (N : ℕ) (hN : 0 < N) : 
  ∃ (p q u v : ℤ), N = p * q + u * v ∧ (u - v = 2 * (p - q)) := 
sorry

end any_positive_integer_can_be_expressed_l2229_222927


namespace no_prime_solutions_l2229_222967

theorem no_prime_solutions (p q : ℕ) (hp : p > 5) (hq : q > 5) (pp : Nat.Prime p) (pq : Nat.Prime q)
  (h : p * q ∣ (5^p - 2^p) * (5^q - 2^q)) : False :=
sorry

end no_prime_solutions_l2229_222967


namespace symmetric_inverse_sum_l2229_222917

theorem symmetric_inverse_sum {f g : ℝ → ℝ} (h₁ : ∀ x, f (-x - 2) = -f (x)) (h₂ : ∀ y, g (f y) = y) (h₃ : ∀ y, f (g y) = y) (x₁ x₂ : ℝ) (h₄ : x₁ + x₂ = 0) : 
  g x₁ + g x₂ = -2 :=
by
  sorry

end symmetric_inverse_sum_l2229_222917


namespace min_square_side_length_l2229_222918

theorem min_square_side_length 
  (table_length : ℕ) (table_breadth : ℕ) (cube_side : ℕ) (num_tables : ℕ)
  (cond1 : table_length = 12)
  (cond2 : table_breadth = 16)
  (cond3 : cube_side = 4)
  (cond4 : num_tables = 4) :
  (2 * table_length + 2 * table_breadth) = 56 := 
by
  sorry

end min_square_side_length_l2229_222918


namespace orchard_trees_l2229_222913

theorem orchard_trees (x p : ℕ) (h : x + p = 480) (h2 : p = 3 * x) : x = 120 ∧ p = 360 :=
by
  sorry

end orchard_trees_l2229_222913


namespace quadratic_points_relation_l2229_222926

theorem quadratic_points_relation (h y1 y2 y3 : ℝ) :
  (∀ x, x = -1/2 → y1 = -(x-2) ^ 2 + h) ∧
  (∀ x, x = 1 → y2 = -(x-2) ^ 2 + h) ∧
  (∀ x, x = 2 → y3 = -(x-2) ^ 2 + h) →
  y1 < y2 ∧ y2 < y3 :=
by
  -- The required proof is omitted
  sorry

end quadratic_points_relation_l2229_222926


namespace cathy_wins_probability_l2229_222932

theorem cathy_wins_probability : 
  -- Definitions of the problem conditions
  let p_win := (1 : ℚ) / 6
  let p_not_win := (5 : ℚ) / 6
  -- The probability that Cathy wins
  (p_not_win ^ 2 * p_win) / (1 - p_not_win ^ 3) = 25 / 91 :=
by
  sorry

end cathy_wins_probability_l2229_222932


namespace difference_of_fractions_l2229_222931

theorem difference_of_fractions (a : ℝ) (b : ℝ) (h₁ : a = 7000) (h₂ : b = 1/10) :
  (a * b - a * (0.1 / 100)) = 693 :=
by 
  sorry

end difference_of_fractions_l2229_222931


namespace fraction_comparison_l2229_222957

theorem fraction_comparison
  (a b c d : ℝ)
  (h1 : a / b < c / d)
  (h2 : b > 0)
  (h3 : d > 0)
  (h4 : b > d) :
  (a + c) / (b + d) < (1 / 2) * (a / b + c / d) :=
by
  sorry

end fraction_comparison_l2229_222957


namespace leak_emptying_time_l2229_222910

theorem leak_emptying_time (fill_rate_no_leak : ℝ) (combined_rate_with_leak : ℝ) (L : ℝ) :
  fill_rate_no_leak = 1/10 →
  combined_rate_with_leak = 1/12 →
  fill_rate_no_leak - L = combined_rate_with_leak →
  1 / L = 60 :=
by
  intros h1 h2 h3
  sorry

end leak_emptying_time_l2229_222910


namespace seashells_given_to_Joan_l2229_222991

def S_original : ℕ := 35
def S_now : ℕ := 17

theorem seashells_given_to_Joan :
  (S_original - S_now) = 18 := by
  sorry

end seashells_given_to_Joan_l2229_222991


namespace find_number_l2229_222979

noncomputable def calc1 : Float := 0.47 * 1442
noncomputable def calc2 : Float := 0.36 * 1412
noncomputable def diff : Float := calc1 - calc2

theorem find_number :
  ∃ (n : Float), (diff + n = 6) :=
sorry

end find_number_l2229_222979


namespace percentage_increase_B_more_than_C_l2229_222981

noncomputable def percentage_increase :=
  let C_m := 14000
  let A_annual := 470400
  let A_m := A_annual / 12
  let B_m := (2 / 5) * A_m
  ((B_m - C_m) / C_m) * 100

theorem percentage_increase_B_more_than_C : percentage_increase = 12 :=
  sorry

end percentage_increase_B_more_than_C_l2229_222981


namespace average_first_six_numbers_l2229_222968

theorem average_first_six_numbers (A : ℝ) (h1 : (11 : ℝ) * 9.9 = (6 * A + 6 * 11.4 - 22.5)) : A = 10.5 :=
by sorry

end average_first_six_numbers_l2229_222968


namespace base_9_perfect_square_b_l2229_222905

theorem base_9_perfect_square_b (b : ℕ) (a : ℕ) 
  (h0 : 0 < b) (h1 : b < 9) (h2 : a < 9) : 
  ∃ n, n^2 ≡ 729 * b + 81 * a + 54 [MOD 81] :=
sorry

end base_9_perfect_square_b_l2229_222905


namespace remainder_when_three_times_number_minus_seven_divided_by_seven_l2229_222986

theorem remainder_when_three_times_number_minus_seven_divided_by_seven (n : ℤ) (h : n % 7 = 2) : (3 * n - 7) % 7 = 6 := by
  sorry

end remainder_when_three_times_number_minus_seven_divided_by_seven_l2229_222986


namespace sum_of_first_3030_terms_l2229_222954

-- Define geometric sequence sum for n terms
noncomputable def geom_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r ^ n) / (1 - r)

-- Given conditions
axiom geom_sum_1010 (a r : ℝ) (hr : r ≠ 1) : geom_sum a r 1010 = 100
axiom geom_sum_2020 (a r : ℝ) (hr : r ≠ 1) : geom_sum a r 2020 = 190

-- Prove that the sum of the first 3030 terms is 271
theorem sum_of_first_3030_terms (a r : ℝ) (hr : r ≠ 1) :
  geom_sum a r 3030 = 271 :=
by
  sorry

end sum_of_first_3030_terms_l2229_222954


namespace point_on_angle_bisector_l2229_222909

theorem point_on_angle_bisector (a : ℝ) 
  (h : (2 : ℝ) * a + (3 : ℝ) = a) : a = -3 :=
sorry

end point_on_angle_bisector_l2229_222909


namespace larger_of_two_numbers_l2229_222915

theorem larger_of_two_numbers (x y : ℝ) (h1 : x + y = 50) (h2 : x - y = 8) : max x y = 29 :=
by
  sorry

end larger_of_two_numbers_l2229_222915


namespace geometric_sequence_tenth_term_l2229_222955

theorem geometric_sequence_tenth_term :
  let a := 4
  let r := (4 / 3 : ℚ)
  a * r ^ 9 = (1048576 / 19683 : ℚ) :=
by
  sorry

end geometric_sequence_tenth_term_l2229_222955


namespace stratified_sampling_expected_females_l2229_222935

noncomputable def sample_size := 14
noncomputable def total_athletes := 44 + 33
noncomputable def female_athletes := 33
noncomputable def stratified_sample := (female_athletes * sample_size) / total_athletes

theorem stratified_sampling_expected_females :
  stratified_sample = 6 :=
by
  sorry

end stratified_sampling_expected_females_l2229_222935


namespace seashells_initial_count_l2229_222996

theorem seashells_initial_count (S : ℝ) (h : S + 4.0 = 10) : S = 6.0 :=
by
  sorry

end seashells_initial_count_l2229_222996


namespace simplify_fraction_l2229_222995

theorem simplify_fraction : 
  (5 / (2 * Real.sqrt 27 + 3 * Real.sqrt 12 + Real.sqrt 108)) = (5 * Real.sqrt 3 / 54) :=
by
  -- Proof will be provided here
  sorry

end simplify_fraction_l2229_222995


namespace find_dividend_l2229_222970

theorem find_dividend (x D : ℕ) (q r : ℕ) (h_q : q = 4) (h_r : r = 3)
  (h_div : D = x * q + r) (h_sum : D + x + q + r = 100) : D = 75 :=
by
  sorry

end find_dividend_l2229_222970


namespace problem_1_part_1_problem_1_part_2_l2229_222952

-- Define the function f
def f (x a : ℝ) := |x - a| + 3 * x

-- The first problem statement - Part (Ⅰ)
theorem problem_1_part_1 (x : ℝ) : { x | x ≥ 3 ∨ x ≤ -1 } = { x | f x 1 ≥ 3 * x + 2 } :=
by {
  sorry
}

-- The second problem statement - Part (Ⅱ)
theorem problem_1_part_2 : { x | x ≤ -1 } = { x | f x 2 ≤ 0 } :=
by {
  sorry
}

end problem_1_part_1_problem_1_part_2_l2229_222952


namespace minimum_value_of_expression_l2229_222994

theorem minimum_value_of_expression
  (a b c : ℝ)
  (h : 2 * a + 2 * b + c = 8) :
  ∃ x, (x = (a - 1)^2 + (b + 2)^2 + (c - 3)^2) ∧ x ≥ (49 / 9) :=
sorry

end minimum_value_of_expression_l2229_222994


namespace determine_ABC_l2229_222956

-- Define values in the new base system
def base_representation (A B C : ℕ) : ℕ :=
  A * (A+1)^7 + A * (A+1)^6 + A * (A+1)^5 + C * (A+1)^4 + B * (A+1)^3 + B * (A+1)^2 + B * (A+1) + C

-- The conditions given by the problem
def condition (A B C : ℕ) : Prop :=
  ((A+1)^8 - 2*(A+1)^4 + 1) = base_representation A B C

-- The theorem to be proved
theorem determine_ABC : ∃ (A B C : ℕ), A = 2 ∧ B = 0 ∧ C = 1 ∧ condition A B C :=
by
  existsi 2
  existsi 0
  existsi 1
  unfold condition base_representation
  sorry

end determine_ABC_l2229_222956


namespace trapezium_division_l2229_222948

theorem trapezium_division (h : ℝ) (m n : ℕ) (h_pos : 0 < h) 
  (areas_equal : 4 / (3 * ↑m) = 7 / (6 * ↑n)) :
  m + n = 15 := by
  sorry

end trapezium_division_l2229_222948


namespace find_m_probability_l2229_222914

theorem find_m_probability (m : ℝ) (ξ : ℕ → ℝ) :
  (ξ 1 = m * (2/3)) ∧ (ξ 2 = m * (2/3)^2) ∧ (ξ 3 = m * (2/3)^3) ∧ 
  (ξ 1 + ξ 2 + ξ 3 = 1) → 
  m = 27 / 38 := 
sorry

end find_m_probability_l2229_222914


namespace pictures_at_the_museum_l2229_222912

theorem pictures_at_the_museum (M : ℕ) (zoo_pics : ℕ) (deleted_pics : ℕ) (remaining_pics : ℕ)
    (h1 : zoo_pics = 15) (h2 : deleted_pics = 31) (h3 : remaining_pics = 2) (h4 : zoo_pics + M = deleted_pics + remaining_pics) :
    M = 18 := 
sorry

end pictures_at_the_museum_l2229_222912


namespace second_shift_fraction_of_total_l2229_222980

theorem second_shift_fraction_of_total (W E : ℕ) (h1 : ∀ (W : ℕ), E = (3 * W / 4))
  : let W₁ := W
    let E₁ := E
    let widgets_first_shift := W₁ * E₁
    let widgets_per_second_shift_employee := (2 * W₁) / 3
    let second_shift_employees := (4 * E₁) / 3
    let widgets_second_shift := (2 * W₁ / 3) * (4 * E₁ / 3)
    let total_widgets := widgets_first_shift + widgets_second_shift
    let fraction_second_shift := widgets_second_shift / total_widgets
    fraction_second_shift = 8 / 17 :=
sorry

end second_shift_fraction_of_total_l2229_222980


namespace no_real_solutions_l2229_222923

theorem no_real_solutions :
  ¬ ∃ x : ℝ, (4 * x^3 + 3 * x^2 + x + 2) / (x - 2) = 4 * x^2 + 5 :=
by
  sorry

end no_real_solutions_l2229_222923


namespace cars_meet_time_l2229_222925

-- Define the initial conditions as Lean definitions
def distance_car1 (t : ℝ) : ℝ := 15 * t
def distance_car2 (t : ℝ) : ℝ := 20 * t
def total_distance : ℝ := 105

-- Define the proposition we want to prove
theorem cars_meet_time : ∃ (t : ℝ), distance_car1 t + distance_car2 t = total_distance ∧ t = 3 :=
by
  sorry

end cars_meet_time_l2229_222925


namespace find_common_ratio_and_difference_l2229_222928

theorem find_common_ratio_and_difference (q d : ℤ) 
  (h1 : q^3 = 1 + 7 * d) 
  (h2 : 1 + q + q^2 + q^3 = 1 + 7 * d + 21) : 
  (q = 4 ∧ d = 9) ∨ (q = -5 ∧ d = -18) :=
by
  sorry

end find_common_ratio_and_difference_l2229_222928


namespace max_min_value_l2229_222908

noncomputable def f (x : ℝ) : ℝ := (2 * x) / (x - 2)

theorem max_min_value (M m : ℝ) (hM : M = f 3) (hm : m = f 4) : (m * m) / M = 8 / 3 := by
  sorry

end max_min_value_l2229_222908


namespace train_usual_time_l2229_222919

theorem train_usual_time (T : ℝ) (h1 : T > 0) : 
  (4 / 5 : ℝ) * (T + 1/2) = T :=
by 
  sorry

end train_usual_time_l2229_222919


namespace number_of_triangles_l2229_222997

theorem number_of_triangles (n : ℕ) (h : n = 10) : ∃ k, k = 120 ∧ n.choose 3 = k :=
by
  sorry

end number_of_triangles_l2229_222997


namespace factorial_expression_simplification_l2229_222934

theorem factorial_expression_simplification : 
  (4 * (Nat.factorial 7) + 28 * (Nat.factorial 6)) / (Nat.factorial 8) = 1 := 
by 
  sorry

end factorial_expression_simplification_l2229_222934


namespace sin_theta_value_l2229_222906

theorem sin_theta_value {θ : ℝ} (h₁ : 9 * (Real.tan θ)^2 = 4 * Real.cos θ) (h₂ : 0 < θ ∧ θ < Real.pi) : 
  Real.sin θ = 1 / 3 :=
by
  sorry

end sin_theta_value_l2229_222906


namespace mark_charged_more_hours_l2229_222975

theorem mark_charged_more_hours (P K M : ℕ) 
  (h_total : P + K + M = 144)
  (h_pat_kate : P = 2 * K)
  (h_pat_mark : P = M / 3) : M - K = 80 := 
by
  sorry

end mark_charged_more_hours_l2229_222975


namespace proof_problem_l2229_222978

open Real

noncomputable def problem_condition1 (A B : ℝ) : Prop :=
  (sin A - sin B) * (sin A + sin B) = sin (π/3 - B) * sin (π/3 + B)

noncomputable def problem_condition2 (b c : ℝ) (a : ℝ) (dot_product : ℝ) : Prop :=
  b * c * cos (π / 3) = dot_product ∧ a = 2 * sqrt 7

noncomputable def problem_condition3 (a b c : ℝ) : Prop := 
  a^2 = (b + c)^2 - 3 * b * c

noncomputable def problem_condition4 (b c : ℝ) : Prop := 
  b < c

theorem proof_problem (A B : ℝ) (a b c dot_product : ℝ)
  (h1 : problem_condition1 A B)
  (h2 : problem_condition2 b c a dot_product)
  (h3 : problem_condition3 a b c)
  (h4 : problem_condition4 b c) :
  (A = π / 3) ∧ (b = 4 ∧ c = 6) :=
by {
  sorry
}

end proof_problem_l2229_222978


namespace cone_lateral_surface_area_l2229_222900

theorem cone_lateral_surface_area (r V: ℝ) (h : ℝ) (l : ℝ) (L: ℝ):
  r = 3 →
  V = 12 * Real.pi →
  V = (1 / 3) * Real.pi * r^2 * h →
  l = Real.sqrt (r^2 + h^2) →
  L = Real.pi * r * l →
  L = 15 * Real.pi :=
by
  intros hr hv hV hl hL
  rw [hr, hv] at hV
  sorry

end cone_lateral_surface_area_l2229_222900


namespace calculate_distance_l2229_222943

def velocity (t : ℝ) : ℝ := 3 * t^2 + t

theorem calculate_distance : ∫ t in (0 : ℝ)..(4 : ℝ), velocity t = 72 := 
by
  sorry

end calculate_distance_l2229_222943


namespace nalani_net_amount_l2229_222988

-- Definitions based on the conditions
def luna_birth := 10 -- Luna gave birth to 10 puppies
def stella_birth := 14 -- Stella gave birth to 14 puppies
def luna_sold := 8 -- Nalani sold 8 puppies from Luna's litter
def stella_sold := 10 -- Nalani sold 10 puppies from Stella's litter
def luna_price := 200 -- Price per puppy for Luna's litter is $200
def stella_price := 250 -- Price per puppy for Stella's litter is $250
def luna_cost := 80 -- Cost of raising each puppy from Luna's litter is $80
def stella_cost := 90 -- Cost of raising each puppy from Stella's litter is $90

-- Theorem stating the net amount received by Nalani
theorem nalani_net_amount : 
        luna_sold * luna_price + stella_sold * stella_price - 
        (luna_birth * luna_cost + stella_birth * stella_cost) = 2040 :=
by 
  sorry

end nalani_net_amount_l2229_222988
