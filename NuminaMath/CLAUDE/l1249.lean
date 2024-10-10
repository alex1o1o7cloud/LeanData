import Mathlib

namespace complex_number_in_second_quadrant_l1249_124941

theorem complex_number_in_second_quadrant : 
  let z : ℂ := 2 * Complex.I * (Complex.I + 1) + 1
  (z.re < 0) ∧ (z.im > 0) := by sorry

end complex_number_in_second_quadrant_l1249_124941


namespace rhombus_area_rhombus_area_is_88_l1249_124943

/-- The area of a rhombus with vertices at (0, 5.5), (8, 0), (0, -5.5), and (-8, 0) is 88 square units. -/
theorem rhombus_area : ℝ → Prop :=
  fun area =>
    let v1 : ℝ × ℝ := (0, 5.5)
    let v2 : ℝ × ℝ := (8, 0)
    let v3 : ℝ × ℝ := (0, -5.5)
    let v4 : ℝ × ℝ := (-8, 0)
    let d1 : ℝ := v1.2 - v3.2
    let d2 : ℝ := v2.1 - v4.1
    area = (d1 * d2) / 2 ∧ area = 88

theorem rhombus_area_is_88 : rhombus_area 88 := by
  sorry

end rhombus_area_rhombus_area_is_88_l1249_124943


namespace geometric_sequence_minimum_l1249_124994

theorem geometric_sequence_minimum (a : ℕ → ℝ) (q : ℝ) (h_q : q ≠ 1) :
  (∃ s t : ℕ, s ≠ 0 ∧ t ≠ 0 ∧ a s * a t = (a 5)^2) →
  (∃ s t : ℕ, s ≠ 0 ∧ t ≠ 0 ∧ a s * a t = (a 5)^2 ∧
    ∀ u v : ℕ, u ≠ 0 → v ≠ 0 → a u * a v = (a 5)^2 →
      4/s + 1/(4*t) ≤ 4/u + 1/(4*v)) →
  (∃ s t : ℕ, s ≠ 0 ∧ t ≠ 0 ∧ a s * a t = (a 5)^2 ∧ 4/s + 1/(4*t) = 5/8) :=
by sorry

end geometric_sequence_minimum_l1249_124994


namespace souvenir_problem_l1249_124933

/-- Represents the number of ways to select souvenirs -/
def souvenir_selection_ways (total_types : ℕ) (expensive_types : ℕ) (cheap_types : ℕ) 
  (expensive_price : ℕ) (cheap_price : ℕ) (total_spent : ℕ) : ℕ :=
  (Nat.choose expensive_types 5) + 
  (Nat.choose expensive_types 4) * (Nat.choose cheap_types 2)

/-- The problem statement -/
theorem souvenir_problem : 
  souvenir_selection_ways 11 8 3 10 5 50 = 266 := by
  sorry

end souvenir_problem_l1249_124933


namespace triangle_properties_l1249_124932

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.b - (1/2) * t.c = t.a * Real.cos t.C)
  (h2 : 4 * (t.b + t.c) = 3 * t.b * t.c)
  (h3 : t.a = 2 * Real.sqrt 3) :
  t.A = Real.pi / 3 ∧ 
  (1/2) * t.b * t.c * Real.sin t.A = 2 * Real.sqrt 3 := by
  sorry

end triangle_properties_l1249_124932


namespace sport_preference_related_to_gender_l1249_124991

-- Define the contingency table
def contingency_table : Matrix (Fin 2) (Fin 2) ℕ :=
  ![![40, 20],
    ![20, 30]]

-- Define the calculated K^2 value
def calculated_k_squared : ℝ := 7.82

-- Define the critical values and their corresponding probabilities
def critical_values : List (ℝ × ℝ) :=
  [(2.706, 0.10), (3.841, 0.05), (6.635, 0.01), (7.879, 0.005), (10.828, 0.001)]

-- Define the confidence level we want to prove
def target_confidence : ℝ := 0.99

-- Theorem statement
theorem sport_preference_related_to_gender :
  ∃ (lower_k upper_k : ℝ) (lower_p upper_p : ℝ),
    (lower_k, lower_p) ∈ critical_values ∧
    (upper_k, upper_p) ∈ critical_values ∧
    lower_k < calculated_k_squared ∧
    calculated_k_squared < upper_k ∧
    lower_p > 1 - target_confidence ∧
    upper_p < 1 - target_confidence :=
by sorry


end sport_preference_related_to_gender_l1249_124991


namespace max_vovochka_candies_l1249_124971

/-- Represents the distribution of candies to classmates -/
def CandyDistribution := Fin 25 → ℕ

/-- The total number of candies -/
def totalCandies : ℕ := 200

/-- Checks if a candy distribution satisfies the condition that any 16 classmates have at least 100 candies -/
def isValidDistribution (d : CandyDistribution) : Prop :=
  ∀ (s : Finset (Fin 25)), s.card = 16 → (s.sum d) ≥ 100

/-- Calculates the number of candies Vovochka keeps for himself given a distribution -/
def vovochkaCandies (d : CandyDistribution) : ℕ :=
  totalCandies - (Finset.univ.sum d)

/-- Theorem stating that the maximum number of candies Vovochka can keep is 37 -/
theorem max_vovochka_candies :
  (∃ (d : CandyDistribution), isValidDistribution d ∧ vovochkaCandies d = 37) ∧
  (∀ (d : CandyDistribution), isValidDistribution d → vovochkaCandies d ≤ 37) :=
sorry

end max_vovochka_candies_l1249_124971


namespace sally_grew_six_carrots_l1249_124901

/-- The number of carrots grown by Fred -/
def fred_carrots : ℕ := 4

/-- The total number of carrots grown by Sally and Fred -/
def total_carrots : ℕ := 10

/-- The number of carrots grown by Sally -/
def sally_carrots : ℕ := total_carrots - fred_carrots

theorem sally_grew_six_carrots : sally_carrots = 6 := by
  sorry

end sally_grew_six_carrots_l1249_124901


namespace increasing_interval_of_f_l1249_124952

noncomputable def f (x : ℝ) := x - Real.exp x

theorem increasing_interval_of_f :
  {x : ℝ | ∀ y, x < y → f x < f y} = Set.Iio 0 :=
sorry

end increasing_interval_of_f_l1249_124952


namespace complex_power_six_l1249_124986

theorem complex_power_six (i : ℂ) (h : i^2 = -1) : (1 + i)^6 = -8*i := by
  sorry

end complex_power_six_l1249_124986


namespace no_solution_fractional_equation_l1249_124917

theorem no_solution_fractional_equation :
  ∀ x : ℝ, (((1 - x) / (x - 2)) + 2 ≠ 1 / (2 - x)) ∨ (x = 2) :=
by sorry

end no_solution_fractional_equation_l1249_124917


namespace yarn_crochet_length_l1249_124969

def yarn_problem (total_length : ℝ) (num_parts : ℕ) (parts_used : ℕ) : Prop :=
  total_length = 10 ∧ 
  num_parts = 5 ∧ 
  parts_used = 3 ∧ 
  (total_length / num_parts) * parts_used = 6

theorem yarn_crochet_length : 
  ∀ (total_length : ℝ) (num_parts : ℕ) (parts_used : ℕ),
  yarn_problem total_length num_parts parts_used :=
by
  sorry

end yarn_crochet_length_l1249_124969


namespace nth_root_approximation_l1249_124936

/-- Approximation of nth root of x₀ⁿ + Δx --/
theorem nth_root_approximation
  (n : ℕ) (x₀ Δx ε : ℝ) (h_x₀_pos : x₀ > 0) (h_Δx_small : |Δx| < x₀^n) :
  ∃ (ε : ℝ), ε > 0 ∧ 
  |((x₀^n + Δx)^(1/n : ℝ) : ℝ) - (x₀ + Δx / (n * x₀^(n-1)))| < ε :=
by sorry

end nth_root_approximation_l1249_124936


namespace inscribed_cube_surface_area_l1249_124906

theorem inscribed_cube_surface_area :
  ∀ (outer_cube_surface_area : ℝ) (inner_cube_surface_area : ℝ),
    outer_cube_surface_area = 54 →
    (∃ (outer_cube_side : ℝ) (sphere_diameter : ℝ) (inner_cube_diagonal : ℝ) (inner_cube_side : ℝ),
      outer_cube_surface_area = 6 * outer_cube_side^2 ∧
      sphere_diameter = outer_cube_side ∧
      inner_cube_diagonal = sphere_diameter ∧
      inner_cube_diagonal = inner_cube_side * Real.sqrt 3 ∧
      inner_cube_surface_area = 6 * inner_cube_side^2) →
    inner_cube_surface_area = 18 :=
by
  sorry


end inscribed_cube_surface_area_l1249_124906


namespace six_by_six_untileable_large_rectangle_tileable_six_by_eight_tileable_l1249_124935

/-- A domino is a 1x2 tile -/
structure Domino :=
  (length : Nat := 2)
  (width : Nat := 1)

/-- A rectangle with dimensions m and n -/
structure Rectangle (m n : Nat) where
  mk ::

/-- A tiling of a rectangle with dominoes -/
def Tiling (m n : Nat) := List Domino

/-- A seam is a straight line not cutting through any dominoes -/
def HasSeam (t : Tiling m n) : Prop := sorry

/-- Theorem: A 6x6 square cannot be tiled with dominoes without a seam -/
theorem six_by_six_untileable : 
  ∀ (t : Tiling 6 6), HasSeam t := sorry

/-- Theorem: Any m×n rectangle where m, n > 6 and mn is even can be tiled without a seam -/
theorem large_rectangle_tileable (m n : Nat) 
  (hm : m > 6) (hn : n > 6) (h_even : Even (m * n)) : 
  ∃ (t : Tiling m n), ¬HasSeam t := sorry

/-- Theorem: A 6x8 rectangle can be tiled without a seam -/
theorem six_by_eight_tileable : 
  ∃ (t : Tiling 6 8), ¬HasSeam t := sorry

end six_by_six_untileable_large_rectangle_tileable_six_by_eight_tileable_l1249_124935


namespace complex_fraction_equals_i_l1249_124988

theorem complex_fraction_equals_i :
  let i : ℂ := Complex.I
  (1 + i) / (1 - i) = i := by sorry

end complex_fraction_equals_i_l1249_124988


namespace mona_group_size_l1249_124900

/-- The number of groups Mona joined --/
def num_groups : ℕ := 9

/-- The number of unique players Mona grouped with --/
def unique_players : ℕ := 33

/-- The number of non-unique player slots --/
def non_unique_slots : ℕ := 3

/-- The number of players in each group, including Mona --/
def players_per_group : ℕ := 5

theorem mona_group_size :
  (num_groups * (players_per_group - 1)) - non_unique_slots = unique_players :=
by sorry

end mona_group_size_l1249_124900


namespace rachel_picked_apples_l1249_124920

def apples_picked (initial_apples remaining_apples : ℕ) : ℕ :=
  initial_apples - remaining_apples

theorem rachel_picked_apples (initial_apples remaining_apples : ℕ) 
  (h1 : initial_apples = 9)
  (h2 : remaining_apples = 7) :
  apples_picked initial_apples remaining_apples = 2 := by
sorry

end rachel_picked_apples_l1249_124920


namespace square_of_sum_80_5_l1249_124945

theorem square_of_sum_80_5 : (80 + 5)^2 = 7225 := by
  sorry

end square_of_sum_80_5_l1249_124945


namespace smallest_perfect_square_divisible_by_2_3_5_l1249_124934

theorem smallest_perfect_square_divisible_by_2_3_5 :
  ∃ n : ℕ, n > 0 ∧ 
  (∃ m : ℕ, n = m^2) ∧
  2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧
  (∀ k : ℕ, k > 0 → (∃ l : ℕ, k = l^2) → 2 ∣ k → 3 ∣ k → 5 ∣ k → k ≥ n) ∧
  n = 900 :=
sorry

end smallest_perfect_square_divisible_by_2_3_5_l1249_124934


namespace solution_set_a_2_range_of_a_l1249_124942

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

-- Part 1: Solution set when a = 2
theorem solution_set_a_2 :
  {x : ℝ | f 2 x ≥ 4} = {x : ℝ | x ≤ 3/2 ∨ x ≥ 11/2} := by sorry

-- Part 2: Range of a
theorem range_of_a :
  {a : ℝ | ∀ x, f a x ≥ 4} = {a : ℝ | a ≤ -1 ∨ a ≥ 3} := by sorry

end solution_set_a_2_range_of_a_l1249_124942


namespace theta_value_l1249_124972

theorem theta_value (θ : Real) (h1 : 1 / Real.sin θ + 1 / Real.cos θ = 35 / 12) 
  (h2 : θ ∈ Set.Ioo 0 (Real.pi / 2)) :
  θ = Real.arcsin (3 / 5) ∨ θ = Real.arcsin (4 / 5) := by
  sorry

end theta_value_l1249_124972


namespace line_parallel_to_plane_relation_l1249_124973

-- Define the necessary structures
structure Line3D where
  -- Add necessary fields for a 3D line

structure Plane3D where
  -- Add necessary fields for a 3D plane

-- Define the relationships
def parallel (l : Line3D) (p : Plane3D) : Prop :=
  sorry

def within (l : Line3D) (p : Plane3D) : Prop :=
  sorry

def parallel_lines (l1 l2 : Line3D) : Prop :=
  sorry

def skew_lines (l1 l2 : Line3D) : Prop :=
  sorry

-- Theorem statement
theorem line_parallel_to_plane_relation (m n : Line3D) (α : Plane3D) 
    (h1 : parallel m α) (h2 : within n α) :
  parallel_lines m n ∨ skew_lines m n :=
sorry

end line_parallel_to_plane_relation_l1249_124973


namespace popped_kernel_probability_l1249_124975

theorem popped_kernel_probability (total : ℝ) (h_total : total > 0) :
  let white := (2 / 3) * total
  let yellow := (1 / 3) * total
  let white_popped := (1 / 2) * white
  let yellow_popped := (2 / 3) * yellow
  let total_popped := white_popped + yellow_popped
  (white_popped / total_popped) = (3 / 5) := by
  sorry

end popped_kernel_probability_l1249_124975


namespace hardcover_nonfiction_count_l1249_124953

/-- Represents the number of books of each type --/
structure BookCounts where
  paperbackFiction : ℕ
  paperbackNonfiction : ℕ
  hardcoverNonfiction : ℕ
  hardcoverFiction : ℕ

/-- The total number of books --/
def totalBooks : ℕ := 10000

/-- Conditions for the book counts --/
def validBookCounts (bc : BookCounts) : Prop :=
  bc.paperbackFiction + bc.paperbackNonfiction + bc.hardcoverNonfiction + bc.hardcoverFiction = totalBooks ∧
  bc.paperbackNonfiction = bc.hardcoverNonfiction + 100 ∧
  bc.paperbackFiction * 3 = bc.hardcoverFiction * 5 ∧
  bc.hardcoverFiction = totalBooks / 100 * 12 ∧
  bc.paperbackNonfiction + bc.hardcoverNonfiction = totalBooks / 100 * 30

theorem hardcover_nonfiction_count (bc : BookCounts) (h : validBookCounts bc) : 
  bc.hardcoverNonfiction = 1450 := by
  sorry

end hardcover_nonfiction_count_l1249_124953


namespace geometric_progression_problem_l1249_124913

def geometric_progression (b₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := b₁ * q^(n - 1)

theorem geometric_progression_problem (b₁ b₅ : ℝ) (h₁ : b₁ = Real.sqrt 3) (h₅ : b₅ = Real.sqrt 243) :
  ∃ q : ℝ, (q = Real.sqrt 3 ∨ q = -Real.sqrt 3) ∧
    geometric_progression b₁ q 5 = b₅ ∧
    geometric_progression b₁ q 6 = 27 ∨ geometric_progression b₁ q 6 = -27 :=
by sorry

end geometric_progression_problem_l1249_124913


namespace power_product_equals_four_l1249_124957

theorem power_product_equals_four : 4^2020 * (1/4)^2019 = 4 := by
  sorry

end power_product_equals_four_l1249_124957


namespace last_three_digits_of_5_pow_1999_l1249_124982

def last_three_digits (n : ℕ) : ℕ := n % 1000

theorem last_three_digits_of_5_pow_1999 :
  last_three_digits (5^1999) = 125 := by
  sorry

end last_three_digits_of_5_pow_1999_l1249_124982


namespace bouquet_cost_50_l1249_124946

/-- Represents the cost function for bouquets at Bella's Blossom Shop -/
def bouquet_cost (n : ℕ) : ℚ :=
  let base_price := (36 : ℚ) / 18 * n.min 40
  let extra_price := if n > 40 then (36 : ℚ) / 18 * (n - 40) else 0
  let total_price := base_price + extra_price
  if n > 40 then total_price * (9 / 10) else total_price

theorem bouquet_cost_50 : bouquet_cost 50 = 90 := by
  sorry

end bouquet_cost_50_l1249_124946


namespace inequality_preservation_l1249_124916

theorem inequality_preservation (x y : ℝ) (h : x > y) : x / 5 > y / 5 := by
  sorry

end inequality_preservation_l1249_124916


namespace calculate_expression_l1249_124954

theorem calculate_expression : 
  50000 - ((37500 / 62.35)^2 + Real.sqrt 324) = -311752.222 := by
  sorry

end calculate_expression_l1249_124954


namespace least_marbles_ten_marbles_john_marbles_l1249_124974

theorem least_marbles (m : ℕ) : m > 0 ∧ m % 7 = 3 ∧ m % 4 = 2 → m ≥ 10 := by
  sorry

theorem ten_marbles : 10 % 7 = 3 ∧ 10 % 4 = 2 := by
  sorry

theorem john_marbles : ∃ m : ℕ, m > 0 ∧ m % 7 = 3 ∧ m % 4 = 2 ∧ ∀ n : ℕ, (n > 0 ∧ n % 7 = 3 ∧ n % 4 = 2) → m ≤ n := by
  sorry

end least_marbles_ten_marbles_john_marbles_l1249_124974


namespace last_digit_3_count_l1249_124924

/-- The number of terms in the sequence 7^1, 7^2, ..., 7^2008 that have a last digit of 3 -/
def count_last_digit_3 : ℕ := 502

/-- The length of the sequence 7^1, 7^2, ..., 7^2008 -/
def sequence_length : ℕ := 2008

theorem last_digit_3_count :
  count_last_digit_3 = sequence_length / 4 :=
sorry

end last_digit_3_count_l1249_124924


namespace xy_xz_yz_bounds_l1249_124907

open Real

theorem xy_xz_yz_bounds (x y z : ℝ) (h : 5 * (x + y + z) = x^2 + y^2 + z^2) :
  (∃ N n : ℝ, (∀ a b c : ℝ, 5 * (a + b + c) = a^2 + b^2 + c^2 → a * b + a * c + b * c ≤ N) ∧
              (∀ a b c : ℝ, 5 * (a + b + c) = a^2 + b^2 + c^2 → n ≤ a * b + a * c + b * c) ∧
              N = 75 ∧ n = 0) := by
  sorry

#check xy_xz_yz_bounds

end xy_xz_yz_bounds_l1249_124907


namespace not_concurrent_deduction_l1249_124919

/-- Represents a proof method -/
inductive ProofMethod
  | Synthetic
  | Analytic

/-- Represents the direction of reasoning -/
inductive ReasoningDirection
  | CauseToEffect
  | EffectToCause

/-- Maps a proof method to its reasoning direction -/
def methodDirection (m : ProofMethod) : ReasoningDirection :=
  match m with
  | ProofMethod.Synthetic => ReasoningDirection.CauseToEffect
  | ProofMethod.Analytic => ReasoningDirection.EffectToCause

/-- Theorem stating that synthetic and analytic methods do not concurrently deduce cause and effect -/
theorem not_concurrent_deduction :
  ∀ (m : ProofMethod), methodDirection m ≠ ReasoningDirection.CauseToEffect ∨ 
                       methodDirection m ≠ ReasoningDirection.EffectToCause :=
by
  sorry

end not_concurrent_deduction_l1249_124919


namespace negation_equivalence_l1249_124956

theorem negation_equivalence : 
  (¬ ∃ x₀ : ℝ, x₀^2 - x₀ + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 - x + 1 > 0) := by
  sorry

end negation_equivalence_l1249_124956


namespace sum_of_arithmetic_sequences_l1249_124985

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem sum_of_arithmetic_sequences
  (a b : ℕ → ℝ)
  (ha : arithmetic_sequence a)
  (hb : arithmetic_sequence b)
  (h1 : a 1 = 25)
  (h2 : b 1 = 75)
  (h3 : a 2 + b 2 = 100) :
  a 37 + b 37 = 100 := by
sorry

end sum_of_arithmetic_sequences_l1249_124985


namespace function_properties_l1249_124992

def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * f y

theorem function_properties (f : ℝ → ℝ) 
  (h1 : functional_equation f) 
  (h2 : f (1/2) = 0) 
  (h3 : f 0 ≠ 0) : 
  (f 0 = 1) ∧ (∀ x : ℝ, f (1/2 + x) = -f (1/2 - x)) := by
  sorry

end function_properties_l1249_124992


namespace number_relationship_l1249_124995

-- Define the numbers in their respective bases
def a : ℕ := 33
def b : ℕ := 5 * 6 + 2  -- 52 in base 6
def c : ℕ := 16 + 8 + 4 + 2 + 1  -- 11111 in base 2

-- Theorem statement
theorem number_relationship : a > b ∧ b > c := by
  sorry

end number_relationship_l1249_124995


namespace units_digit_of_n_squared_plus_two_to_n_l1249_124959

theorem units_digit_of_n_squared_plus_two_to_n (n : ℕ) :
  n = 2023^2 + 2^2023 →
  (n^2 + 2^n) % 10 = 7 := by
sorry

end units_digit_of_n_squared_plus_two_to_n_l1249_124959


namespace toy_distribution_l1249_124977

/-- Given a number of pens and toys distributed among students, 
    where each student receives the same number of pens and toys, 
    prove that the number of toys is a multiple of the number of students. -/
theorem toy_distribution (num_pens : ℕ) (num_toys : ℕ) (num_students : ℕ) 
  (h1 : num_pens = 451)
  (h2 : num_students = 41)
  (h3 : num_pens % num_students = 0)
  (h4 : num_toys % num_students = 0) :
  ∃ k : ℕ, num_toys = num_students * k :=
sorry

end toy_distribution_l1249_124977


namespace order_of_abc_l1249_124983

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem order_of_abc : a > c ∧ c > b := by sorry

end order_of_abc_l1249_124983


namespace factorization_proof_l1249_124926

theorem factorization_proof (x : ℝ) : 2*x^3 - 8*x^2 + 8*x = 2*x*(x - 2)^2 := by
  sorry

end factorization_proof_l1249_124926


namespace total_bones_l1249_124927

/-- The number of bones Xiao Qi has -/
def xiao_qi_bones : ℕ := sorry

/-- The number of bones Xiao Shi has -/
def xiao_shi_bones : ℕ := sorry

/-- The number of bones Xiao Ha has -/
def xiao_ha_bones : ℕ := sorry

/-- Xiao Ha has 2 more bones than twice the number of bones Xiao Shi has -/
axiom ha_shi_relation : xiao_ha_bones = 2 * xiao_shi_bones + 2

/-- Xiao Shi has 3 more bones than three times the number of bones Xiao Qi has -/
axiom shi_qi_relation : xiao_shi_bones = 3 * xiao_qi_bones + 3

/-- Xiao Ha has 5 fewer bones than seven times the number of bones Xiao Qi has -/
axiom ha_qi_relation : xiao_ha_bones = 7 * xiao_qi_bones - 5

/-- The total number of bones is 141 -/
theorem total_bones :
  xiao_qi_bones + xiao_shi_bones + xiao_ha_bones = 141 :=
sorry

end total_bones_l1249_124927


namespace violet_family_ticket_cost_l1249_124968

/-- The cost of separate tickets for Violet's family -/
def separate_ticket_cost (adult_price children_price : ℕ) (num_adults num_children : ℕ) : ℕ :=
  adult_price * num_adults + children_price * num_children

/-- Theorem: The total cost of separate tickets for Violet's family is $155 -/
theorem violet_family_ticket_cost :
  separate_ticket_cost 35 20 1 6 = 155 :=
by sorry

end violet_family_ticket_cost_l1249_124968


namespace derivative_f_l1249_124984

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.log (Real.cos (1/3)) + (Real.sin (23*x))^2 / (23 * Real.cos (46*x))

-- State the theorem
theorem derivative_f :
  ∀ x : ℝ, deriv f x = Real.tan (46*x) / Real.cos (46*x) :=
by sorry

end derivative_f_l1249_124984


namespace obtuse_angle_range_l1249_124909

/-- The angle between two 2D vectors is obtuse if and only if their dot product is negative -/
def is_obtuse_angle (a b : Fin 2 → ℝ) : Prop :=
  (a 0 * b 0 + a 1 * b 1) < 0

/-- The set of real numbers x for which the angle between (1, 3) and (x, -1) is obtuse -/
def obtuse_angle_set : Set ℝ :=
  {x : ℝ | is_obtuse_angle (![1, 3]) (![x, -1])}

theorem obtuse_angle_range :
  obtuse_angle_set = {x : ℝ | x < -1/3 ∨ (-1/3 < x ∧ x < 3)} := by sorry

end obtuse_angle_range_l1249_124909


namespace one_eighth_divided_by_one_fourth_l1249_124961

theorem one_eighth_divided_by_one_fourth (a b c : ℚ) :
  a = 1 / 8 → b = 1 / 4 → c = 1 / 2 → a / b = c := by
  sorry

end one_eighth_divided_by_one_fourth_l1249_124961


namespace sector_arc_length_and_area_l1249_124979

/-- Given a sector with radius 2 and central angle π/6, prove that the arc length is π/3 and the area is π/3 -/
theorem sector_arc_length_and_area :
  let r : ℝ := 2
  let θ : ℝ := π / 6
  let arc_length : ℝ := r * θ
  let sector_area : ℝ := (1 / 2) * r * r * θ
  arc_length = π / 3 ∧ sector_area = π / 3 := by
sorry


end sector_arc_length_and_area_l1249_124979


namespace cookie_box_duration_l1249_124929

/-- Proves that a box of cookies lasts 9 days given the specified conditions -/
theorem cookie_box_duration (oldest_son_cookies : ℕ) (youngest_son_cookies : ℕ) (total_cookies : ℕ) : 
  oldest_son_cookies = 4 → 
  youngest_son_cookies = 2 → 
  total_cookies = 54 → 
  (total_cookies / (oldest_son_cookies + youngest_son_cookies) : ℕ) = 9 := by
  sorry

#check cookie_box_duration

end cookie_box_duration_l1249_124929


namespace intersection_of_M_and_N_l1249_124938

-- Define the sets M and N
def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def N : Set ℝ := {y | ∃ x ∈ M, y = x^2}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x | 0 ≤ x ∧ x ≤ 1} := by
  sorry

end intersection_of_M_and_N_l1249_124938


namespace cookfire_logs_remaining_l1249_124904

/-- Represents the number of logs remaining in a cookfire after a given number of hours -/
def logs_remaining (initial_logs : ℕ) (burn_rate : ℕ) (add_rate : ℕ) (hours : ℕ) : ℤ :=
  initial_logs + hours * (add_rate - burn_rate)

/-- Theorem stating the number of logs remaining after x hours for the given cookfire scenario -/
theorem cookfire_logs_remaining (x : ℕ) :
  logs_remaining 8 4 3 x = 8 - x :=
sorry

end cookfire_logs_remaining_l1249_124904


namespace tan_alpha_plus_pi_fourth_l1249_124931

/-- If the terminal side of angle α passes through point (-1, 2), 
    then tan(α + π/4) = -1/3 -/
theorem tan_alpha_plus_pi_fourth (α : ℝ) :
  (∃ (t : ℝ), t > 0 ∧ t * Real.cos α = -1 ∧ t * Real.sin α = 2) →
  Real.tan (α + π/4) = -1/3 := by
sorry

end tan_alpha_plus_pi_fourth_l1249_124931


namespace largest_n_value_largest_n_achievable_l1249_124910

/-- Represents a digit in base 5 -/
def Base5Digit := Fin 5

/-- Represents a digit in base 9 -/
def Base9Digit := Fin 9

/-- Converts a number from base 5 to base 10 -/
def fromBase5 (a b c : Base5Digit) : ℕ :=
  25 * a.val + 5 * b.val + c.val

/-- Converts a number from base 9 to base 10 -/
def fromBase9 (c b a : Base9Digit) : ℕ :=
  81 * c.val + 9 * b.val + a.val

theorem largest_n_value (n : ℕ) 
  (h1 : ∃ (a b c : Base5Digit), n = fromBase5 a b c)
  (h2 : ∃ (a b c : Base9Digit), n = fromBase9 c b a) :
  n ≤ 111 := by
  sorry

theorem largest_n_achievable : 
  ∃ (n : ℕ) (a b c : Base5Digit) (x y z : Base9Digit),
    n = fromBase5 a b c ∧ 
    n = fromBase9 z y x ∧ 
    n = 111 := by
  sorry

end largest_n_value_largest_n_achievable_l1249_124910


namespace problem_part1_problem_part2_l1249_124960

-- Part 1
theorem problem_part1 : (-2)^3 + |(-3)| - Real.tan (π/4) = -6 := by sorry

-- Part 2
theorem problem_part2 (a : ℝ) : (a + 2)^2 - a*(a - 4) = 8*a + 4 := by sorry

end problem_part1_problem_part2_l1249_124960


namespace manufacturing_plant_optimization_l1249_124978

noncomputable def f (x : ℝ) : ℝ := 4 * (1 - x) * x^2

def domain (t : ℝ) (x : ℝ) : Prop := 0 < x ∧ x ≤ 2*t/(2*t+1)

theorem manufacturing_plant_optimization (t : ℝ) 
  (h1 : 0 < t) (h2 : t ≤ 2) :
  (f 0.5 = 0.5) ∧
  (∀ x, domain t x →
    (1 ≤ t → f x ≤ 16/27 ∧ (f x = 16/27 → x = 2/3)) ∧
    (t < 1 → f x ≤ 16*t^2/(2*t+1)^3 ∧ (f x = 16*t^2/(2*t+1)^3 → x = 2*t/(2*t+1)))) :=
by sorry

end manufacturing_plant_optimization_l1249_124978


namespace lukes_father_twenty_bills_l1249_124903

def mother_fifty : ℕ := 1
def mother_twenty : ℕ := 2
def mother_ten : ℕ := 3

def father_fifty : ℕ := 4
def father_ten : ℕ := 1

def school_fee : ℕ := 350

theorem lukes_father_twenty_bills :
  ∃ (father_twenty : ℕ),
    50 * mother_fifty + 20 * mother_twenty + 10 * mother_ten +
    50 * father_fifty + 20 * father_twenty + 10 * father_ten = school_fee ∧
    father_twenty = 1 :=
by sorry

end lukes_father_twenty_bills_l1249_124903


namespace lucille_paint_cans_l1249_124963

/-- Represents the dimensions of a wall -/
structure Wall where
  width : ℝ
  height : ℝ

/-- Calculates the area of a wall -/
def wallArea (w : Wall) : ℝ := w.width * w.height

/-- Represents the room to be painted -/
structure Room where
  wall1 : Wall
  wall2 : Wall
  wall3 : Wall
  wall4 : Wall

/-- Calculates the total area of all walls in the room -/
def totalArea (r : Room) : ℝ :=
  wallArea r.wall1 + wallArea r.wall2 + wallArea r.wall3 + wallArea r.wall4

/-- The coverage area of one can of paint -/
def paintCoverage : ℝ := 2

/-- Lucille's room configuration -/
def lucilleRoom : Room :=
  { wall1 := { width := 3, height := 2 }
  , wall2 := { width := 3, height := 2 }
  , wall3 := { width := 5, height := 2 }
  , wall4 := { width := 4, height := 2 } }

/-- Theorem: Lucille needs 15 cans of paint -/
theorem lucille_paint_cans : 
  ⌈(totalArea lucilleRoom) / paintCoverage⌉ = 15 := by sorry

end lucille_paint_cans_l1249_124963


namespace arithmetic_sequence_problem_l1249_124937

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum sequence
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2

/-- The main theorem -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
    (h₁ : seq.S 3 = 9) (h₂ : seq.S 6 = 36) : 
  seq.a 6 + seq.a 7 + seq.a 8 = 39 := by
  sorry

end arithmetic_sequence_problem_l1249_124937


namespace distance_from_origin_l1249_124966

/-- Given a point (x,y) satisfying certain conditions, prove that its distance from the origin is √(286 + 2√221) -/
theorem distance_from_origin (x y : ℝ) (h1 : y = 8) (h2 : x > 1) 
  (h3 : Real.sqrt ((x - 1)^2 + 2^2) = 15) : 
  Real.sqrt (x^2 + y^2) = Real.sqrt (286 + 2 * Real.sqrt 221) := by
  sorry

end distance_from_origin_l1249_124966


namespace common_solution_condition_l1249_124964

theorem common_solution_condition (a b : ℝ) : 
  (∃ x y : ℝ, 19 * x^2 + 19 * y^2 + a * x + b * y + 98 = 0 ∧ 
               98 * x^2 + 98 * y^2 + a * x + b * y + 19 = 0) ↔ 
  a^2 + b^2 ≥ 13689 := by
sorry

end common_solution_condition_l1249_124964


namespace roots_polynomial_sum_l1249_124965

theorem roots_polynomial_sum (p q : ℝ) : 
  p^2 - 6*p + 10 = 0 → q^2 - 6*q + 10 = 0 → p^4 + p^5*q^3 + p^3*q^5 + q^4 = 16056 :=
by sorry

end roots_polynomial_sum_l1249_124965


namespace game_ends_in_41_rounds_l1249_124998

/-- Represents a player in the token game -/
structure Player where
  name : String
  tokens : ℕ

/-- Represents the state of the game -/
structure GameState where
  players : List Player
  rounds : ℕ

/-- Simulates one round of the game -/
def playRound (state : GameState) : GameState :=
  sorry

/-- Checks if the game has ended (i.e., a player has run out of tokens) -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Simulates the entire game until it ends -/
def playGame (initialState : GameState) : GameState :=
  sorry

/-- Theorem stating that the game ends after exactly 41 rounds -/
theorem game_ends_in_41_rounds :
  let initialState : GameState :=
    { players := [
        { name := "D", tokens := 16 },
        { name := "E", tokens := 15 },
        { name := "F", tokens := 13 }
      ],
      rounds := 0
    }
  let finalState := playGame initialState
  finalState.rounds = 41 ∧ isGameOver finalState := by
  sorry

end game_ends_in_41_rounds_l1249_124998


namespace jane_yellow_sheets_l1249_124997

/-- The number of old, yellow sheets of drawing paper Jane has -/
def yellowSheets (totalSheets brownSheets : ℕ) : ℕ :=
  totalSheets - brownSheets

theorem jane_yellow_sheets : 
  let totalSheets : ℕ := 55
  let brownSheets : ℕ := 28
  yellowSheets totalSheets brownSheets = 27 := by
  sorry

end jane_yellow_sheets_l1249_124997


namespace problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_l1249_124955

-- Problem 1
theorem problem_1 : (1 * -5) + 9 = 4 := by sorry

-- Problem 2
theorem problem_2 : 12 - (-16) + (-2) - 1 = 25 := by sorry

-- Problem 3
theorem problem_3 : 6 / (-2) * (-1/3) = 1 := by sorry

-- Problem 4
theorem problem_4 : (-15) * (1/3 + 1/5) = -8 := by sorry

-- Problem 5
theorem problem_5 : (-2)^3 - (-8) / |-(4/3)| = -2 := by sorry

-- Problem 6
theorem problem_6 : -(1^2022) - (1/2 - 1/3) * 3 = -3/2 := by sorry

end problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_l1249_124955


namespace non_adjacent_book_selection_l1249_124939

/-- The number of books on the shelf -/
def total_books : ℕ := 12

/-- The number of books to be chosen -/
def books_to_choose : ℕ := 5

/-- The theorem stating that the number of ways to choose 5 books out of 12
    such that no two chosen books are adjacent is equal to C(8,5) -/
theorem non_adjacent_book_selection :
  (Nat.choose (total_books - books_to_choose + 1) books_to_choose) =
  (Nat.choose 8 5) := by sorry

end non_adjacent_book_selection_l1249_124939


namespace marks_score_l1249_124911

theorem marks_score (highest_score : ℕ) (score_range : ℕ) (marks_score : ℕ) :
  highest_score = 98 →
  score_range = 75 →
  marks_score = 2 * (highest_score - score_range) →
  marks_score = 46 :=
by
  sorry

end marks_score_l1249_124911


namespace science_club_membership_l1249_124912

theorem science_club_membership (total : ℕ) (chem : ℕ) (bio : ℕ) (both : ℕ) 
  (h1 : total = 80)
  (h2 : chem = 48)
  (h3 : bio = 40)
  (h4 : both = 25) :
  total - (chem + bio - both) = 17 := by
  sorry

end science_club_membership_l1249_124912


namespace boys_to_girls_ratio_l1249_124925

theorem boys_to_girls_ratio (total : ℕ) (diff : ℕ) : 
  total = 36 → 
  diff = 6 → 
  ∃ (boys girls : ℕ), 
    boys = girls + diff ∧ 
    boys + girls = total ∧ 
    boys * 5 = girls * 7 := by
sorry

end boys_to_girls_ratio_l1249_124925


namespace bridge_extension_l1249_124944

/-- The width of the river in inches -/
def river_width : ℕ := 487

/-- The length of the existing bridge in inches -/
def existing_bridge_length : ℕ := 295

/-- The additional length needed for the bridge to cross the river -/
def additional_length : ℕ := river_width - existing_bridge_length

theorem bridge_extension :
  additional_length = 192 := by sorry

end bridge_extension_l1249_124944


namespace zoo_revenue_example_l1249_124987

/-- Calculates the total money made by a zoo over two days given the number of children and adults each day and the ticket prices. -/
def zoo_revenue (child_price adult_price : ℕ) (mon_children mon_adults tues_children tues_adults : ℕ) : ℕ :=
  (mon_children * child_price + mon_adults * adult_price) +
  (tues_children * child_price + tues_adults * adult_price)

/-- Theorem stating that the zoo made $61 in total for both days. -/
theorem zoo_revenue_example : zoo_revenue 3 4 7 5 4 2 = 61 := by
  sorry

end zoo_revenue_example_l1249_124987


namespace symmetric_point_y_axis_l1249_124951

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Symmetry with respect to the y-axis -/
def symmetricYAxis (p : Point3D) : Point3D :=
  { x := -p.x, y := p.y, z := -p.z }

/-- The theorem stating that the point symmetric to (2, -3, 5) with respect to the y-axis is (-2, -3, -5) -/
theorem symmetric_point_y_axis :
  let original := Point3D.mk 2 (-3) 5
  symmetricYAxis original = Point3D.mk (-2) (-3) (-5) := by
  sorry

end symmetric_point_y_axis_l1249_124951


namespace correct_formula_l1249_124947

def f (x : ℝ) : ℝ := 5 * x^2 + x

theorem correct_formula : 
  (f 0 = 0) ∧ 
  (f 1 = 20) ∧ 
  (f 2 = 60) ∧ 
  (f 3 = 120) ∧ 
  (f 4 = 200) := by
  sorry

end correct_formula_l1249_124947


namespace system_of_equations_solution_l1249_124989

theorem system_of_equations_solution : ∀ x y : ℚ,
  (6 * x - 48 * y = 2) ∧ (3 * y - x = 4) →
  x^2 + y^2 = 442 / 25 := by
sorry

end system_of_equations_solution_l1249_124989


namespace chocolates_remaining_chocolates_remaining_day6_l1249_124958

/-- Chocolates remaining after 5 days of eating with given conditions -/
theorem chocolates_remaining (total : ℕ) (day1 : ℕ) (day2 : ℕ) : ℕ :=
  let day3 := day1 - 3
  let day4 := 2 * day3 + 1
  let day5 := day2 / 2
  total - (day1 + day2 + day3 + day4 + day5)

/-- Proof that 14 chocolates remain on Day 6 given the problem conditions -/
theorem chocolates_remaining_day6 :
  chocolates_remaining 48 6 8 = 14 := by
  sorry

end chocolates_remaining_chocolates_remaining_day6_l1249_124958


namespace fixed_point_of_exponential_function_l1249_124905

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ (x : ℝ), x = 1 ∧ (a^(x - 1) + 1 = x) := by
  sorry

end fixed_point_of_exponential_function_l1249_124905


namespace circle_and_line_properties_l1249_124999

-- Define the circles and line
def circle_M (x y : ℝ) := 2*x^2 + 2*y^2 - 8*x - 8*y - 1 = 0
def circle_N (x y : ℝ) := x^2 + y^2 + 2*x + 2*y - 6 = 0
def line_l (x y : ℝ) := x + y - 9 = 0

-- Define the angle condition
def angle_BAC : ℝ := 45

-- Theorem statement
theorem circle_and_line_properties :
  ∃ (x y : ℝ),
    -- 1. Equation of circle through intersection of M and N, and origin
    (x^2 + y^2 - (50/11)*x - (50/11)*y = 0) ∧
    -- 2a. Equations of line AC when x-coordinate of A is 4
    ((5*x + y - 25 = 0) ∨ (x - 5*y + 21 = 0)) ∧
    -- 2b. Range of possible x-coordinates for point A
    (∀ (m : ℝ), (m ∈ Set.Icc 3 6) ↔ 
      (∃ (y : ℝ), line_l m y ∧ 
        ∃ (B C : ℝ × ℝ), 
          circle_M B.1 B.2 ∧ 
          circle_M C.1 C.2 ∧
          (angle_BAC = 45))) :=
sorry

end circle_and_line_properties_l1249_124999


namespace fgh_supermarkets_in_us_l1249_124990

theorem fgh_supermarkets_in_us (total : ℕ) (difference : ℕ) 
  (h_total : total = 84)
  (h_difference : difference = 14) :
  let us := (total + difference) / 2
  us = 49 := by
sorry

end fgh_supermarkets_in_us_l1249_124990


namespace ABC_collinear_l1249_124918

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Three points in the plane -/
def A : Point := ⟨-1, 4⟩
def B : Point := ⟨-3, 2⟩
def C : Point := ⟨0, 5⟩

/-- Definition of collinearity for three points -/
def collinear (p q r : Point) : Prop :=
  (q.y - p.y) * (r.x - q.x) = (r.y - q.y) * (q.x - p.x)

/-- Theorem: Points A, B, and C are collinear -/
theorem ABC_collinear : collinear A B C := by
  sorry

end ABC_collinear_l1249_124918


namespace max_sum_with_reciprocals_l1249_124921

theorem max_sum_with_reciprocals (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + y + 1/x + 1/y = 5) : 
  ∀ a b : ℝ, a > 0 → b > 0 → a + b + 1/a + 1/b = 5 → x + y ≥ a + b ∧ x + y ≤ 4 :=
by sorry

end max_sum_with_reciprocals_l1249_124921


namespace tangent_four_implies_expression_l1249_124949

theorem tangent_four_implies_expression (α : Real) (h : Real.tan α = 4) :
  (1 + Real.cos (2 * α) + 8 * Real.sin α ^ 2) / Real.sin (2 * α) = 65 / 4 := by
  sorry

end tangent_four_implies_expression_l1249_124949


namespace arithmetic_sequence_sum_l1249_124930

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 3 + a 5 + a 12 - a 2 = 12 →
  a 7 + a 11 = 12 := by
sorry

end arithmetic_sequence_sum_l1249_124930


namespace shortest_distance_to_quadratic_curve_l1249_124996

/-- The shortest distance from a point to a quadratic curve -/
theorem shortest_distance_to_quadratic_curve
  (m k a b : ℝ) :
  let curve := fun (x : ℝ) => m * x^2 + k
  let P := (a, b)
  let Q := fun (c : ℝ) => (c, curve c)
  ∃ (c : ℝ), ∀ (x : ℝ),
    dist P (Q c) ≤ dist P (Q x) ∧
    dist P (Q c) = |m * a^2 + k - b| :=
by sorry

end shortest_distance_to_quadratic_curve_l1249_124996


namespace quadratic_real_root_condition_l1249_124940

theorem quadratic_real_root_condition (b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by
sorry

end quadratic_real_root_condition_l1249_124940


namespace binomial_10_choose_5_l1249_124923

theorem binomial_10_choose_5 : Nat.choose 10 5 = 252 := by
  sorry

end binomial_10_choose_5_l1249_124923


namespace midpoint_coordinate_sum_l1249_124962

/-- The sum of the coordinates of the midpoint of a segment with endpoints (6, 12) and (0, -6) is 6. -/
theorem midpoint_coordinate_sum : 
  let p1 : ℝ × ℝ := (6, 12)
  let p2 : ℝ × ℝ := (0, -6)
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  midpoint.1 + midpoint.2 = 6 := by
  sorry

end midpoint_coordinate_sum_l1249_124962


namespace parallel_vectors_sum_l1249_124902

def a : ℝ × ℝ × ℝ := (3, -2, 4)
def b : ℝ → ℝ → ℝ × ℝ × ℝ := λ x y ↦ (1, x, y)

theorem parallel_vectors_sum (x y : ℝ) :
  (∃ (k : ℝ), b x y = k • a) → x + y = 2/3 := by
  sorry

end parallel_vectors_sum_l1249_124902


namespace no_5_6_8_multiplier_l1249_124922

/-- Function to get the number of digits in a positive integer -/
def num_digits (n : ℕ) : ℕ :=
  if n < 10 then 1 else 1 + num_digits (n / 10)

/-- Function to get the leading digit of a positive integer -/
def leading_digit (n : ℕ) : ℕ :=
  if n < 10 then n else leading_digit (n / 10)

/-- Function to move the leading digit to the end -/
def move_leading_digit (n : ℕ) : ℕ :=
  let d := num_digits n
  let lead := leading_digit n
  (n - lead * 10^(d-1)) * 10 + lead

/-- Theorem stating that no integer becomes 5, 6, or 8 times larger when its leading digit is moved to the end -/
theorem no_5_6_8_multiplier (n : ℕ) (h : n ≥ 10) : 
  let m := move_leading_digit n
  m ≠ 5*n ∧ m ≠ 6*n ∧ m ≠ 8*n :=
sorry

end no_5_6_8_multiplier_l1249_124922


namespace middle_school_enrollment_l1249_124948

theorem middle_school_enrollment (band_percentage : Real) (sports_percentage : Real)
  (band_count : Nat) (sports_count : Nat)
  (h1 : band_percentage = 0.20)
  (h2 : sports_percentage = 0.30)
  (h3 : band_count = 168)
  (h4 : sports_count = 252) :
  ∃ (total : Nat), (band_count : Real) / band_percentage = total ∧
                   (sports_count : Real) / sports_percentage = total ∧
                   total = 840 := by
  sorry

end middle_school_enrollment_l1249_124948


namespace cos_beta_value_l1249_124976

theorem cos_beta_value (α β : Real) (P : ℝ × ℝ) :
  P = (3, 4) →
  P.1 = 3 * Real.cos α ∧ P.2 = 3 * Real.sin α →
  Real.cos (α + β) = 1/3 →
  β ∈ Set.Ioo 0 Real.pi →
  Real.cos β = (3 + 8 * Real.sqrt 2) / 15 := by
  sorry

end cos_beta_value_l1249_124976


namespace cubic_inequality_l1249_124908

theorem cubic_inequality (x : ℝ) :
  x ≥ 0 → (x^3 - 9*x^2 - 16*x > 0 ↔ x > 16) := by sorry

end cubic_inequality_l1249_124908


namespace jerry_trays_capacity_l1249_124915

def jerry_trays (trays_table1 trays_table2 num_trips : ℕ) : ℕ :=
  (trays_table1 + trays_table2) / num_trips

theorem jerry_trays_capacity :
  jerry_trays 9 7 2 = 8 := by
  sorry

end jerry_trays_capacity_l1249_124915


namespace banana_bread_theorem_l1249_124981

/-- The number of bananas needed to make one loaf of banana bread -/
def bananas_per_loaf : ℕ := 4

/-- The number of loaves made on Monday -/
def monday_loaves : ℕ := 3

/-- The number of loaves made on Tuesday -/
def tuesday_loaves : ℕ := 2 * monday_loaves

/-- The total number of bananas used over two days -/
def total_bananas : ℕ := bananas_per_loaf * (monday_loaves + tuesday_loaves)

theorem banana_bread_theorem : total_bananas = 36 := by
  sorry

end banana_bread_theorem_l1249_124981


namespace fraction_relationship_l1249_124980

theorem fraction_relationship (a b c : ℝ) 
  (h1 : a / b = 3 / 5) 
  (h2 : b / c = 2 / 7) : 
  c / a = 35 / 6 := by
  sorry

end fraction_relationship_l1249_124980


namespace expand_and_simplify_l1249_124950

theorem expand_and_simplify (x : ℝ) (hx : x ≠ 0) :
  (3 / 7) * ((14 / x^3) + 15*x - 6*x^5) = 6 / x^3 + (45*x) / 7 - (18*x^5) / 7 := by
  sorry

end expand_and_simplify_l1249_124950


namespace f_nonnegative_iff_a_eq_one_f_greater_than_x_ln_x_minus_sin_x_l1249_124967

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - x - a

-- Theorem 1: f(x) ≥ 0 if and only if a = 1
theorem f_nonnegative_iff_a_eq_one :
  (∀ x, f a x ≥ 0) ↔ a = 1 :=
sorry

-- Theorem 2: For a ≥ 1, f(x) > x ln x - sin x for all x > 0
theorem f_greater_than_x_ln_x_minus_sin_x
  (a : ℝ) (h : a ≥ 1) :
  ∀ x > 0, f a x > x * Real.log x - Real.sin x :=
sorry

end f_nonnegative_iff_a_eq_one_f_greater_than_x_ln_x_minus_sin_x_l1249_124967


namespace aubreys_garden_aubreys_garden_proof_l1249_124993

/-- Aubrey's Garden Planting Problem -/
theorem aubreys_garden (tomato_cucumber_ratio : Nat) (plants_per_row : Nat) (tomatoes_per_plant : Nat) (total_tomatoes : Nat) : Nat :=
  let tomato_rows := total_tomatoes / (plants_per_row * tomatoes_per_plant)
  let cucumber_rows := tomato_rows * tomato_cucumber_ratio
  tomato_rows + cucumber_rows

/-- Proof of Aubrey's Garden Planting Problem -/
theorem aubreys_garden_proof :
  aubreys_garden 2 8 3 120 = 15 := by
  sorry

end aubreys_garden_aubreys_garden_proof_l1249_124993


namespace lucy_bank_balance_l1249_124970

theorem lucy_bank_balance (initial_balance deposit withdrawal : ℕ) :
  initial_balance = 65 →
  deposit = 15 →
  withdrawal = 4 →
  initial_balance + deposit - withdrawal = 76 := by
sorry

end lucy_bank_balance_l1249_124970


namespace inequality_proof_l1249_124914

theorem inequality_proof (x y : ℝ) : x^2 + y^2 + 1 ≥ x*y + x + y := by
  sorry

end inequality_proof_l1249_124914


namespace container_volume_ratio_l1249_124928

theorem container_volume_ratio : 
  ∀ (A B C : ℚ),
  (4/5 : ℚ) * A = (3/5 : ℚ) * B →
  (3/5 : ℚ) * B = (3/4 : ℚ) * C →
  A / C = (15/16 : ℚ) :=
by
  sorry

end container_volume_ratio_l1249_124928
