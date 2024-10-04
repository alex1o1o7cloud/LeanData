import Mathlib

namespace taller_tree_height_l180_180985

theorem taller_tree_height
  (h : ℕ)
  (h_shorter_ratio : h - 16 = (3 * h) / 4) : h = 64 := by
  sorry

end taller_tree_height_l180_180985


namespace find_monotonically_decreasing_interval_find_range_of_g_l180_180755

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

def adjacent_axes_of_symmetry (ω : ℝ) : Prop :=
  (2 * Math.pi) / ω = Math.pi

def problem_conditions (ω ϕ : ℝ) (f : ℝ → ℝ) : Prop :=
  (0 < ϕ ∧ ϕ < Math.pi) ∧
  is_odd f ∧
  adjacent_axes_of_symmetry ω

noncomputable def f (ω ϕ x : ℝ) := Math.sqrt 3 * Real.sin(ω * x + ϕ) - Real.cos(ω * x + ϕ)

theorem find_monotonically_decreasing_interval :
  ∀ (ω ϕ : ℝ), problem_conditions ω ϕ (f ω ϕ) →
  ∀ x, x ∈ Set.Ioo (-Math.pi / 2) (Math.pi / 4) →
  f 2 (Math.pi / 6) x ∈ Set.Ioo (-Math.pi / 2) (-Math.pi / 4) := 
sorry

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin(4 * x - Math.pi / 3)

theorem find_range_of_g :
  ∀ x, x ∈ Set.Icc (-Math.pi / 12) (Math.pi / 6) →
  g x ∈ Set.Icc (-2 : ℝ) (Real.sqrt 3) := 
sorry

end find_monotonically_decreasing_interval_find_range_of_g_l180_180755


namespace min_gennadys_l180_180635

theorem min_gennadys (alexanders borises vasilies : ℕ) (x : ℕ) 
    (h1 : alexanders = 45)
    (h2 : borises = 122)
    (h3 : vasilies = 27)
    (h4 : x = 49)
    (h5 : borises - 1 = alexanders + vasilies + x) :
  x = 49 := 
begin
  sorry,
end

end min_gennadys_l180_180635


namespace cost_price_percentage_l180_180140

variables (CP MP SP : ℝ) (x : ℝ)

theorem cost_price_percentage (h1 : CP = (x / 100) * MP)
                             (h2 : SP = 0.5 * MP)
                             (h3 : SP = 2 * CP) :
                             x = 25 := by
  sorry

end cost_price_percentage_l180_180140


namespace algorithm_not_infinite_l180_180591

-- Definitions and conditions based on the problem
def algorithm (P : Type) : Prop :=
  ∃ (steps : list P), true

-- The goal is to prove that for any algorithm, it does not possess infiniteness
theorem algorithm_not_infinite :
  ∀ (P : Type) (a : algorithm P), ¬ ∃ (steps : list P), false := by
  -- Proof is omitted
  sorry

end algorithm_not_infinite_l180_180591


namespace max_sqrt_expression_exists_y_max_sqrt_l180_180315

theorem max_sqrt_expression {y : ℝ} (h : -49 ≤ y ∧ y ≤ 49) : 
  sqrt (49 + y) + sqrt (49 - y) ≤ 14 :=
by
  sorry

theorem exists_y_max_sqrt : 
  ∃ y : ℝ, -49 ≤ y ∧ y ≤ 49 ∧ (sqrt (49 + y) + sqrt (49 - y) = 14) :=
by
  use 0
  split
  { linarith }
  split
  { linarith }
  calc
    sqrt (49 + 0) + sqrt (49 - 0)
        = sqrt 49 + sqrt 49 : by congr; ring
    ... = 7 + 7 : by rw [sqrt_eq_rsqrt 49, sqrt_eq_rsqrt 49]
    ... = 14 : by ring

end max_sqrt_expression_exists_y_max_sqrt_l180_180315


namespace boundary_points_distance_probability_l180_180454

theorem boundary_points_distance_probability
  (a b c : ℕ)
  (h1 : ∀ (x y : ℝ), x ∈ [0, 4] → y ∈ [0, 4] → (|x - y| ≥ 1 / 2 → True))
  (h2 : ∀ (x y : ℝ), x ∈ [0, 4] → y ∈ [0, 4] → True)
  (h3 : ∃ a b c : ℕ, a - b * Real.pi = 2 ∧ c = 4 ∧ Int.gcd (Int.ofNat a) (Int.gcd (Int.ofNat b) (Int.ofNat c)) = 1) :
  (a + b + c = 62) := sorry

end boundary_points_distance_probability_l180_180454


namespace min_gennadies_l180_180630

theorem min_gennadies 
  (n_Alexanders : ℕ) (n_Borises : ℕ) (n_Vasilies : ℕ) 
  (x_Gennadies : ℕ) 
  (h_Alexanders : n_Alexanders = 45) 
  (h_Borises   : n_Borises = 122) 
  (h_Vasilies  : n_Vasilies = 27) 
  (h_condition : ∀ p : ℕ, p = n_Borises - 1 → p = 121) 
  (h_total     : ∀ q : ℕ, q = 45 + 27 → q = 72)
  : x_Gennadies = 49 := 
sorry

end min_gennadies_l180_180630


namespace exponent_of_five_in_30_factorial_l180_180040

theorem exponent_of_five_in_30_factorial : 
  nat.factorial_prime_exponent 30 5 = 7 := 
sorry

end exponent_of_five_in_30_factorial_l180_180040


namespace sum_first_50_natural_numbers_l180_180533

-- Define the sum of the first n natural numbers
def sum_natural (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

-- Prove that the sum of the first 50 natural numbers is 1275
theorem sum_first_50_natural_numbers : sum_natural 50 = 1275 := 
by
  -- Skipping proof details
  sorry

end sum_first_50_natural_numbers_l180_180533


namespace total_sweaters_knit_l180_180433

-- Definitions from condition a)
def monday_sweaters : ℕ := 8
def tuesday_sweaters : ℕ := monday_sweaters + 2
def wednesday_sweaters : ℕ := tuesday_sweaters - 4
def thursday_sweaters : ℕ := wednesday_sweaters
def friday_sweaters : ℕ := monday_sweaters / 2

-- Theorem statement
theorem total_sweaters_knit : 
  monday_sweaters + tuesday_sweaters + wednesday_sweaters + thursday_sweaters + friday_sweaters = 34 :=
  by
    sorry

end total_sweaters_knit_l180_180433


namespace decrease_percent_revenue_l180_180218

theorem decrease_percent_revenue (T C : ℝ) (hT : T > 0) (hC : C > 0) : 
  let original_revenue := T * C
  let new_tax := 0.80 * T
  let new_consumption := 1.10 * C
  let new_revenue := new_tax * new_consumption
  let decrease_in_revenue := original_revenue - new_revenue
  let decrease_percent := (decrease_in_revenue / original_revenue) * 100
  decrease_percent = 12 := by
  sorry

end decrease_percent_revenue_l180_180218


namespace sum_of_possible_g_values_is_168_l180_180175

-- Define a structure for the multiplicative magic square
structure MagicSquare (a b c d e f g h i : ℕ) :=
(magic_constant : ℕ)
(row1 : a * b * c = magic_constant)
(row2 : d * e * f = magic_constant)
(row3 : g * h * i = magic_constant)
(col1 : a * d * g = magic_constant)
(col2 : b * e * h = magic_constant)
(col3 : c * f * i = magic_constant)
(diag1 : a * e * i = magic_constant)
(diag2 : c * e * g = magic_constant)
(all_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f ∧ 0 < g ∧ 0 < h ∧ 0 < i)

noncomputable def sum_of_possible_g_values : ℕ :=
  ∑ g in { x : ℕ | ∃ (a b c d e f h : ℕ), 
    (MagicSquare 60 b c d e f g h 3 ∧ 
    x = g) }, id

-- The theorem we need to prove
theorem sum_of_possible_g_values_is_168 : sum_of_possible_g_values = 168 := sorry

end sum_of_possible_g_values_is_168_l180_180175


namespace exponent_of_5_in_30_fact_l180_180022

def count_powers_of_5 (n : ℕ) : ℕ :=
  if n < 5 then 0
  else n / 5 + count_powers_of_5 (n / 5)

theorem exponent_of_5_in_30_fact : count_powers_of_5 30 = 7 := 
  by
    sorry

end exponent_of_5_in_30_fact_l180_180022


namespace exponent_of_five_in_factorial_l180_180058

theorem exponent_of_five_in_factorial:
  (nat.factors 30!).count 5 = 7 :=
begin
  sorry
end

end exponent_of_five_in_factorial_l180_180058


namespace equation_of_l2_l180_180348

theorem equation_of_l2 
  (a : ℝ × ℝ) (b : ℝ × ℝ) (P : ℝ × ℝ) (l2_perpendicular_to_l1 : a.1 * b.1 + a.2 * b.2 = 0) 
  (P_on_l2 : ∃ m: ℝ, b.2 * m = -b.1 ∧ m * 0 + 5 = b.2):
  a = (1, 3) ∧ b = (-1, 1/3) ∧ P = (0, 5) → (∀ x y: ℝ, (x, y) ∈ l2_perpendicular_to_l1 → x + 3 * y = 15) :=
by
  -- Parameters and assumptions
  intros
  -- Skipping proof as per instructions
  sorry

end equation_of_l2_l180_180348


namespace ayeon_travel_time_l180_180280

-- Conditions
def distance_to_hospital_km : ℝ := 0.09
def time_for_3_meters_seconds : ℕ := 4
def distance_covered_meters : ℕ := 3
def distance_to_hospital_m := distance_to_hospital_km * 1000

-- Given that Ayeon takes 4 seconds for 3 meters, we calculate the time it takes for her to travel 90 meters.
def ayeon_time_to_hospital : ℝ :=
  (time_for_3_meters_seconds / distance_covered_meters.to_real) * distance_to_hospital_m

-- Theorem to be proved
theorem ayeon_travel_time :
  ayeon_time_to_hospital = 120 := by
  sorry

end ayeon_travel_time_l180_180280


namespace liquid_level_ratio_l180_180521

noncomputable def lower_cone_radius := 4 -- cm
noncomputable def upper_cone_radius := 8 -- cm
noncomputable def cone_height := 8 -- cm
noncomputable def marble_radius := 2 -- cm
noncomputable def initial_liquid_height := 8 -- cm

theorem liquid_level_ratio :
  ∃ ratio : ℝ,
    ratio = 3.67 ∧
    (∀ hₗ hᵤ : ℝ,
      hₗ = initial_liquid_height → 
      hᵤ = initial_liquid_height →
      let Vₗ := (1 / 3) * ℂ.pi * (lower_cone_radius ^ 2) * hₗ in
      let Vᵤ := (1 / 3) * ℂ.pi * (upper_cone_radius ^ 2) * hᵤ in
      let Vₘ := (4 / 3) * ℂ.pi * (marble_radius ^ 3) in
      let x := (Vₗ + Vₘ) / (128 * ℂ.pi) ^ (1/3) in
      let y := (Vᵤ + Vₘ) / (512 * ℂ.pi) ^ (1/3) in
      let Δhₗ := 8 * (x - 1) in
      let Δhᵤ := 8 * (y - 1) in
      Δhₗ / Δhᵤ = ratio) := sorry

end liquid_level_ratio_l180_180521


namespace pears_for_36_bananas_l180_180920

theorem pears_for_36_bananas (p : ℕ) (bananas : ℕ) (pears : ℕ) (h : 9 * pears = 6 * bananas) :
  36 * pears = 9 * 24 :=
by
  sorry

end pears_for_36_bananas_l180_180920


namespace find_k_l180_180336

theorem find_k (k : ℝ) : (∀ x y : ℝ, (x + k * y - 2 * k = 0) → (k * x - (k - 2) * y + 1 = 0) → x * k + y * (-1 / k) + y * 2 = 0) →
  (k = 0 ∨ k = 3) :=
by
  sorry

end find_k_l180_180336


namespace min_value_a_l180_180075

theorem min_value_a (a b c d : ℚ) (h₀ : a > 0)
  (h₁ : ∀ n : ℕ, (a * n^3 + b * n^2 + c * n + d).den = 1) :
  a = 1/6 := by
  -- Proof goes here
  sorry

end min_value_a_l180_180075


namespace correct_proposition_l180_180969

namespace ComplexProposition

open Complex

def proposition1 := 0 > -I -- Proposition ①: 0 is greater than -i
def proposition2 (z1 z2 : ℂ) : Prop := (conj z1 = z2) ↔ (z1 + z2).im = 0 -- Proposition ②: Two complex numbers are conjugates if and only if their sum is real
def proposition3 (x y : ℝ) : Prop := x + y * I = 1 + I ↔ x = 1 ∧ y = 1 -- Proposition ③: The necessary and sufficient condition for x + yi = 1 + i is x = y = 1
def proposition4 (a : ℝ) : Prop := (a ≠ 0) → (a ↔ a * I) -- Proposition ④: One-to-one correspondence between real numbers and pure imaginary numbers
def proposition5 (z : ℂ) : Prop := abs z ^ 2 = abs (conj z) ^ 2 ∧ abs z ^ 2 = z * conj z -- Proposition ⑤: Modulus property for complex number

theorem correct_proposition : proposition5 :=
by sorry

end ComplexProposition

end correct_proposition_l180_180969


namespace exponent_of_five_in_30_factorial_l180_180032

theorem exponent_of_five_in_30_factorial : 
  nat.factorial_prime_exponent 30 5 = 7 := 
sorry

end exponent_of_five_in_30_factorial_l180_180032


namespace inequality_proof_l180_180113

theorem inequality_proof (a b : ℝ) (h1 : a < 1) (h2 : b < 1) (h3 : a + b ≥ 1/3) : 
  (1 - a) * (1 - b) ≤ 25/36 :=
by
  sorry

end inequality_proof_l180_180113


namespace combined_weight_of_student_and_sister_l180_180386

theorem combined_weight_of_student_and_sister
  (S : ℝ) (R : ℝ)
  (h1 : S = 90)
  (h2 : S - 6 = 2 * R) :
  S + R = 132 :=
by
  sorry

end combined_weight_of_student_and_sister_l180_180386


namespace quadratic_symmetry_example_l180_180730

-- Define the quadratic function p(x)
def quadratic_func (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_symmetry_example :
  ∃ (a b c : ℝ), 
  let p := quadratic_func a b c in
  p 6 = 2 ∧ 
  (∀ k : ℝ, p (9 - k) = p (9 + k)) →
  p 12 = 2 :=
by
  sorry

end quadratic_symmetry_example_l180_180730


namespace total_value_of_gold_is_l180_180886

-- Definitions based on the conditions
def legacyBars : ℕ := 5
def aleenaBars : ℕ := legacyBars - 2
def valuePerBar : ℝ := 2200
def totalValue : ℝ := (legacyBars + aleenaBars) * valuePerBar

-- Theorem statement
theorem total_value_of_gold_is :
  totalValue = 17600 := by
  -- We add sorry here to skip the proof
  sorry

end total_value_of_gold_is_l180_180886


namespace treasure_probability_l180_180251

def probability_of_treasure_and_no_traps (num_islands : ℕ) (num_islands_with_treasure : ℕ)
    (prob_treasure_no_traps : ℚ) (prob_no_traps_no_treasure : ℚ) :=
  let ways_to_choose_treasure := nat.choose num_islands num_islands_with_treasure in
  let probability_per_scenario := (prob_treasure_no_traps ^ num_islands_with_treasure) *
                                  (prob_no_traps_no_treasure ^ (num_islands - num_islands_with_treasure)) in
  (ways_to_choose_treasure : ℚ) * probability_per_scenario

theorem treasure_probability :
  probability_of_treasure_and_no_traps 8 4 (1/3) (1/2) = 35/648 := by
  sorry

end treasure_probability_l180_180251


namespace find_b_l180_180849

-- Define the trigonometric sine values
noncomputable def sin_45 := Real.sin (Real.pi / 4)
noncomputable def sin_60 := Real.sin (Real.pi / 3)

-- Define the given conditions
def A : ℝ := Real.pi / 4  -- 45 degrees in radians
def B : ℝ := Real.pi / 3  -- 60 degrees in radians
def a : ℝ := 2

-- Define the theorem to prove
theorem find_b (A B a : ℝ) (hA : A = Real.pi / 4) (hB : B = Real.pi / 3) (ha : a = 2) : 
  let b := (a * Real.sin B / Real.sin A) in b = Real.sqrt 6 :=
by
  sorry

end find_b_l180_180849


namespace intersection_eq_l180_180769

open Set

def M : Set ℤ := {-1, 0, 1}
def N : Set ℕ := {n | n > 0}

theorem intersection_eq : M ∩ N = {0, 1} := by
  sorry

end intersection_eq_l180_180769


namespace find_cost_of_jersey_l180_180986

def cost_of_jersey (J : ℝ) : Prop := 
  let shorts_cost := 15.20
  let socks_cost := 6.80
  let total_players := 16
  let total_cost := 752
  total_players * (J + shorts_cost + socks_cost) = total_cost

theorem find_cost_of_jersey : cost_of_jersey 25 :=
  sorry

end find_cost_of_jersey_l180_180986


namespace length_of_real_axis_l180_180369

variable (a b : ℝ)

def passes_through_point (a b : ℝ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (a^2 - 2 = 8 / a^2)

def parallel_line_distance_asymptote (a b : ℝ) : Prop :=
  (b = 2 * sqrt 2 * a) ∧ (abs (2 * a) / sqrt (a^2 + b^2) = 2 / 3)

theorem length_of_real_axis (a b : ℝ) (h₁ : passes_through_point a b) (h₂ : parallel_line_distance_asymptote a b) :
  2 * a = 2 := 
sorry

end length_of_real_axis_l180_180369


namespace cube_equation_l180_180785

theorem cube_equation (w : ℝ) 
  (h : (w + 5)^3 = (w + 2) * (3w^2 + 13w + 14)) :
  w^3 = -2 * w^2 + (35 / 2) * w + (97 / 2) :=
by
  have h1 : (w + 5)^3 = w^3 + 15 * w^2 + 75 * w + 125 := by sorry
  have h2 : (w + 2) * (3 * w^2 + 13 * w + 14) = 3 * w^3 + 19 * w^2 + 40 * w + 28 := by sorry
  have h3 : w^3 + 15 * w^2 + 75 * w + 125 = 3 * w^3 + 19 * w^2 + 40 * w + 28 := by rw [h1, h2, h]
  have h4 : 0 = 2 * w^3 + 4 * w^2 - 35 * w - 97 := by sorry
  have h5 : 2 * w^3 = -4 * w^2 + 35 * w + 97 := by sorry
  have h6 : w^3 = -2 * w^2 + (35 / 2) * w + (97 / 2) := by sorry
  exact h6
  sorry

end cube_equation_l180_180785


namespace total_games_played_l180_180502

-- Define the conditions as given in the problem
def ratio_of_games : ℕ × ℕ × ℕ := (4, 3, 1)  -- the ratio of wins, losses, and ties
def losses : ℕ := 9                           -- the number of games lost

-- Prove the total number of games played
theorem total_games_played : 
  let parts_per_game := losses / ratio_of_games.2
  let total_parts := ratio_of_games.1 + ratio_of_games.2 + ratio_of_games.3
  total_parts * parts_per_game = 24 :=
by
  sorry

end total_games_played_l180_180502


namespace convert_to_cylindrical_l180_180294

-- Define the function for converting rectangular to cylindrical coordinates
def rectangular_to_cylindrical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := Real.atan2 y x
  (r, θ, z)

-- Given conditions
def point_rectangular : ℝ × ℝ × ℝ := (3, -3 * Real.sqrt 3, 2)
def expected_result : ℝ × ℝ × ℝ := (6, 5 * Real.pi / 3, 2)

-- The theorem to prove
theorem convert_to_cylindrical :
  rectangular_to_cylindrical 3 (-3 * Real.sqrt 3) 2 = expected_result := by
  sorry

end convert_to_cylindrical_l180_180294


namespace max_quotient_l180_180786

theorem max_quotient (a b : ℝ) (h1 : 300 ≤ a) (h2 : a ≤ 500) (h3 : 800 ≤ b) (h4 : b ≤ 1600) :
  ∃ (max_value : ℝ), max_value = 16 / 3 :=
by
  use 16 / 3
  sorry

end max_quotient_l180_180786


namespace teacher_total_score_l180_180261

variable (written_score : ℕ)
variable (interview_score : ℕ)
variable (weight_written : ℝ)
variable (weight_interview : ℝ)

theorem teacher_total_score :
  (written_score = 80) → (interview_score = 60) → (weight_written = 0.6) → (weight_interview = 0.4) →
  (written_score * weight_written + interview_score * weight_interview = 72) :=
by
  sorry

end teacher_total_score_l180_180261


namespace smallest_positive_period_symmetry_axis_not_even_function_decreasing_interval_l180_180364

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + (4 * Real.pi / 3))

theorem smallest_positive_period (T : ℝ) : T = Real.pi ↔ (∀ x : ℝ, f (x + T) = f x) := by
  sorry

theorem symmetry_axis (x : ℝ) : x = (7 * Real.pi / 12) ↔ (∀ y : ℝ, f (2 * x - y) = f y) := by
  sorry

theorem not_even_function : ¬ (∀ x : ℝ, f (x + (Real.pi / 3)) = f (-x - (Real.pi / 3))) := by
  sorry

theorem decreasing_interval (k : ℤ) (x : ℝ) : (k * Real.pi - (5 * Real.pi / 12) ≤ x ∧ x ≤ k * Real.pi + (Real.pi / 12)) ↔ (∀ x1 x2 : ℝ, x1 < x2 → f x1 ≥ f x2) := by
  sorry

end smallest_positive_period_symmetry_axis_not_even_function_decreasing_interval_l180_180364


namespace acute_angled_triangle_range_l180_180847

theorem acute_angled_triangle_range (x : ℝ) (h : (x^2 + 6)^2 < (x^2 + 4)^2 + (4 * x)^2) : x > (Real.sqrt 15) / 3 := sorry

end acute_angled_triangle_range_l180_180847


namespace true_propositions_count_l180_180269

-- Define the propositions as conditions
def prop1 := ∀ (α β : Type) [plane α] [plane β], 
  (∃ (A B C : α) (hAC : ¬ Collinear A B C), A ∈ β ∧ B ∈ β ∧ C ∈ β) → α = β

def prop2 := ∀ (l m : Type) [line l] [line m], 
  l ≠ m → ¬ (l ∩ m).is_plane

def prop3 := ∀ (l m n : Type) [line l] [line m] [line n], 
  (∃ (P : Point), P ∈ l ∧ P ∈ m ∧ P ∈ n) → ∃ (α : plane), l ⊆ α ∧ m ⊆ α ∧ n ⊆ α

def prop4 := ∀ (M : Point) (α β : plane) (l : line),
  M ∈ α ∧ M ∈ β ∧ α ∩ β = l → M ∈ l

-- Prove the mathematical statement
theorem true_propositions_count : 
  (prop1 ∨ prop2 ∨ prop3 ∨ prop4) ↔ (2) := by sorry

end true_propositions_count_l180_180269


namespace coefficient_x2y4_expansion_l180_180421

open_locale big_operators

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
if h : k ≤ n then nat.choose n k else 0

theorem coefficient_x2y4_expansion :
  ∀ x y : ℝ, expand (x + y) * expand (x - y) ^ 5,
  ∑ k in finset.range 5, 
    (-1) ^ k * (binomial_coefficient 5 k) * (expand (x ^ (5 - k)) * expand (y ^ k)) : 
  (coe (to_finsupp x ^ 2 * coe (to_finsupp y ^ 4)) = -5 :=
begin
  sorry
end

end coefficient_x2y4_expansion_l180_180421


namespace sec_tan_eq_l180_180830

theorem sec_tan_eq (x : ℝ) (h : Real.cos x ≠ 0) : 
  Real.sec x + Real.tan x = 7 / 3 → Real.sec x - Real.tan x = 3 / 7 :=
by
  intro h1
  sorry

end sec_tan_eq_l180_180830


namespace complex_modulus_product_l180_180352

-- Definitions of the conditions directly from the problem
def z : ℂ := (√3 + complex.I) / (1 - √3 * complex.I)^2
def z_conj : ℂ := conj z

-- The theorem to prove the desired result
theorem complex_modulus_product : z * z_conj = 1 / 4 := by
  sorry

end complex_modulus_product_l180_180352


namespace percentage_not_red_roses_l180_180407

-- Definitions for the conditions
def roses : Nat := 25
def tulips : Nat := 40
def daisies : Nat := 60
def lilies : Nat := 15
def sunflowers : Nat := 10
def totalFlowers : Nat := roses + tulips + daisies + lilies + sunflowers -- 150
def redRoses : Nat := roses / 2 -- 12 (considering integer division)

-- Statement to prove
theorem percentage_not_red_roses : 
  ((totalFlowers - redRoses) * 100 / totalFlowers) = 92 := by
  sorry

end percentage_not_red_roses_l180_180407


namespace geometry_problem_l180_180593

open Real

noncomputable def radius_of_circle_J
  (radius_Z : ℝ) (radius_G : ℝ) (radius_H : ℝ) (radius_I : ℝ) (radius_F : ℝ)
  (vertex_dist_G : ℝ) (vertex_dist_H : ℝ) (vertex_dist_I : ℝ) : ℝ :=
let r := (vertex_dist_H + radius_H) - radius_Z in
let p := 137 in
let q := 8 in
  ∃ (r : ℝ), r = p / q ∧ p + q = 145

-- Definitions based on conditions
def given_conditions (radius_Z radius_G radius_H radius_I radius_F vertex_dist_G vertex_dist_H vertex_dist_I : ℝ) : Prop :=
radius_Z = 15 ∧
radius_G = 5  ∧
radius_H = 3 ∧
radius_I = 3 ∧
radius_F = 1 ∧
vertex_dist_G = radius_Z - radius_G + sqrt 3 * radius_Z ∧
vertex_dist_H = radius_Z - radius_H + sqrt 3 * radius_Z ∧
vertex_dist_I = radius_Z - radius_I + sqrt 3 * radius_Z

theorem geometry_problem : 
∀ (radius_Z radius_G radius_H radius_I radius_F vertex_dist_G vertex_dist_H vertex_dist_I : ℝ),
  given_conditions radius_Z radius_G radius_H radius_I radius_F vertex_dist_G vertex_dist_H vertex_dist_I →
    radius_of_circle_J radius_Z radius_G radius_H radius_I radius_F vertex_dist_G vertex_dist_H vertex_dist_I := sorry

end geometry_problem_l180_180593


namespace find_a_find_m_l180_180096

noncomputable def f (a : ℝ) (x : ℝ) := x * (Real.exp x - 1) - a * x ^ 2

theorem find_a :
  let x := 1 in 
  let f₁ := f 1 1 in
  let f₁' := (Real.exp 1 + 1 * Real.exp 1 - 1 - a * 2 * 1) in
  (1, f 1 1).snd = 2 * Real.exp 1 - 2 → a = 1 / 2 :=
by
  intros x f₁ f₁' h
  simp at h
  sorry

theorem find_m (a : ℝ) :
  a = 1 / 2 → 
  (∀ x, let f' := (x + 1) * (Real.exp x - 1) in 
    ((2 * x - 3) < f' ∧ f' < (3 * x - 2)) = (x < -1 ∨ x > 0)) →
  (∀ m, (2 * m - 3 > 0 ∧ 3 * m - 2 > -1) = (-1 < m ∧ m ≤ 1 / 3 ∨ m ≥ 3 / 2)) :=
by
  intros a h1 f' m
  sorry

end find_a_find_m_l180_180096


namespace exponent_of_5_in_30_factorial_l180_180013

theorem exponent_of_5_in_30_factorial : 
  (nat.factors 30!).count 5 = 7 :=
sorry

end exponent_of_5_in_30_factorial_l180_180013


namespace binary_island_strategy_sufficient_l180_180928

noncomputable def binary_island_minimum_n (message_length damaged_length : ℕ) : ℕ :=
  damaged_length + message_length

theorem binary_island_strategy_sufficient (n : ℕ) (h : n = binary_island_minimum_n 2016 10) :
  ∃ (strategy : ℕ → ℕ → bool), ∀ message ∈ ({msg | ∃ (start end_ : ℕ), 
  (start + 2016 ≤ end_) ∧ (end_ ≤ start + n)} : set (ℕ → bool)), 
  strategy message_length damaged_length = true :=
  sorry

end binary_island_strategy_sufficient_l180_180928


namespace log_final_l180_180133

noncomputable def log_condition {x : ℝ} (h : x > 1) : 
  log 2 (log 4 x) + log 4 (log 16 x) + log 16 (log 2 x) = 0

theorem log_final {x : ℝ} (h : x > 1) (h_cond : log_condition h) :
  log 2 (log 16 x) + log 16 (log 4 x) + log 4 (log 2 x) = -1 / 4 := 
sorry

end log_final_l180_180133


namespace monotonic_intervals_f_range_f_neg1_sum_inequality_l180_180760

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x - a * x * Real.log x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := (f x a) / x - 1

open Real

-- (1) Monotonic intervals of f(x)
theorem monotonic_intervals_f (a : ℝ) (h : a ≤ 0) : 
  -- Prove the monotonic intervals of f(x) based on the value of a
  sorry

-- (2) When a = -1
namespace a_eq_neg1

noncomputable def f_neg1 : ℝ → ℝ := λ x, x + x * log x
noncomputable def g_neg1 : ℝ → ℝ := λ x, (f_neg1 x) / x - 1

-- (i) Range of f(x) on [e^{-e}, e]
theorem range_f_neg1 : 
  set.range (λ x : set.Icc (exp (-(exp(1)))) (exp(1)), f_neg1 x) 
  = set.Icc (-(1/(exp(2)))) (2 * exp(1)) :=
  sorry

-- (ii) Prove the inequality for the sum
theorem sum_inequality (n : ℕ) (hn : 2 ≤ n):
  ∑ k in finset.range (n + 1).filter (λ k, 2 ≤ k), 1 / log k > (3 * n^2 - n - 2) / (n * (n + 1)) :=
  sorry

end a_eq_neg1

end monotonic_intervals_f_range_f_neg1_sum_inequality_l180_180760


namespace sec_minus_tan_l180_180807

theorem sec_minus_tan (x : ℝ) (h : Real.sec x + Real.tan x = 7 / 3) : Real.sec x - Real.tan x = 3 / 7 :=
by
  sorry

end sec_minus_tan_l180_180807


namespace ratio_AO_OF_FC_l180_180277

open EuclideanGeometry

theorem ratio_AO_OF_FC
  (ABCD : Square)
  (O : Point)
  (E : Point)
  (F : Point)
  (H1 : AC O BD)
  (H2 : E ∈ Segment BC)
  (H3 : EC = (1/4) * BC)
  (H4 : F ∈ Line DE)
  (H5 : DE ∩ AC = F) :
  ratio (Segment AO) (Segment OF) (Segment FC) = (5, 3, 2) :=
by
  sorry

end ratio_AO_OF_FC_l180_180277


namespace symmetric_line_thm_l180_180484

noncomputable def is_symmetric_line 
  (a b c : ℝ) (p : ℝ × ℝ) (l l' : ℝ × ℝ → Prop) : Prop :=
  ∀ (x y : ℝ), (l (x, y) ↔ a * x + b * y + c = 0) →
               (l' (2 * (fst p) - x, 2 * (snd p) - y) ↔ a * x + b * y - 8 = 0)

theorem symmetric_line_thm :
  ∃ l' : ℝ × ℝ → Prop, is_symmetric_line 2 3 (-6) (1, -1) (λ p, 2 * p.1 + 3 * p.2 + (-6) = 0) l' :=
begin
  -- Proof goes here
  sorry
end

end symmetric_line_thm_l180_180484


namespace exam_questions_l180_180876

theorem exam_questions (total_time questions_answered time_used: ℕ) 
    (remaining_time : total_time = time_used + remaining_time)
    (rate: ℚ)
    (rate_def: rate = questions_answered / time_used): 
    total_time = 60 ∧ questions_answered = 16 ∧ time_used = 12 → 
    rate * (60 - time_used) + questions_answered = 80 := 
by
  intros h
  cases h with h_total_time h1
  cases h1 with h_questions_answered h_time_used
  sorry

end exam_questions_l180_180876


namespace fair_game_stakes_ratio_l180_180186

theorem fair_game_stakes_ratio (n : ℕ) (deck_size : ℕ) (player_count : ℕ)
  (L : ℕ → ℝ) : 
  deck_size = 36 → player_count = 36 → 
  (∀ k : ℕ, k < player_count - 1 → 
    (L (k + 1)) / (L k) = 35 / 36) :=
by
  intros h_deck_size h_player_count k hk
  simp [h_deck_size, h_player_count, hk]
  sorry

end fair_game_stakes_ratio_l180_180186


namespace min_gennadies_l180_180643

noncomputable section

def minGennadiesNeeded (alexanders borises vasilies : Nat) : Nat :=
  let needed_gaps := borises - 1
  let total_others := alexanders + vasilies
  if needed_gaps > total_others then needed_gaps - total_others else 0

theorem min_gennadies (alexanders borises vasilies : Nat) (h_alex: alexanders = 45) (h_boris: borises = 122) (h_vasil: vasilies = 27):
  minGennadiesNeeded alexanders borises vasilies = 49 := by
  rw [h_alex, h_boris, h_vasil]
  simp [minGennadiesNeeded]
  sorry

end min_gennadies_l180_180643


namespace find_a_l180_180452

def g (x : ℝ) : ℝ :=
  if x ≤ 0 then -x else 3 * x - 41

theorem find_a :
  ∃ (a : ℝ), a < 0 ∧ g (g (g a)) = g (g (g 8)) ∧ a = -58 / 3 :=
by
  sorry

end find_a_l180_180452


namespace sec_tan_eq_l180_180834

theorem sec_tan_eq (x : ℝ) (h : Real.cos x ≠ 0) : 
  Real.sec x + Real.tan x = 7 / 3 → Real.sec x - Real.tan x = 3 / 7 :=
by
  intro h1
  sorry

end sec_tan_eq_l180_180834


namespace min_number_of_gennadys_l180_180665

theorem min_number_of_gennadys (a b v g : ℕ) (h_a : a = 45) (h_b : b = 122) (h_v : v = 27)
    (h_needed_g : g = 49) :
    (b - 1) - (a + v) = g :=
by
  -- We include sorry because we are focusing on the statement, not the proof itself.
  sorry

end min_number_of_gennadys_l180_180665


namespace exponent_of_5_in_30_fact_l180_180017

def count_powers_of_5 (n : ℕ) : ℕ :=
  if n < 5 then 0
  else n / 5 + count_powers_of_5 (n / 5)

theorem exponent_of_5_in_30_fact : count_powers_of_5 30 = 7 := 
  by
    sorry

end exponent_of_5_in_30_fact_l180_180017


namespace angle_LKM_45_iff_square_l180_180471

theorem angle_LKM_45_iff_square (ABCD : Type) [rect : Rectangle ABCD] (O : Circle) (P : O.MinorArc CD) (proj : Projections P ABCD) :
  let K := proj.projection_on_AB
      L := proj.projection_on_AC
      M := proj.projection_on_BD in
  (∠LKM = 45°) ↔ (is_square ABCD) :=
sorry

end angle_LKM_45_iff_square_l180_180471


namespace length_of_AK_l180_180934

-- Define points, square, and side length
variables (A B C D G H J K : ℝ) -- coordinates
variable (s : ℝ) -- side length of square ABCD

-- Given conditions
axiom square_side_length : s = 18
axiom midpoint_J : J = 9 -- since J is the midpoint of side BC of length 18
axiom area_triangle_AJK : 0.5 * (20) * 18 = 180 -- Given area condition of triangle AJK

-- Using Pythagorean theorem to find AK
noncomputable def length_AK : ℝ := real.sqrt (20^2 + 9^2)

-- The theorem to be proved
theorem length_of_AK : length_AK = real.sqrt 481 := 
sorry

end length_of_AK_l180_180934


namespace part_I_part_II_l180_180172

noncomputable theory
open Real Nat

def a : ℕ → ℝ
| 0     := 1/2
| (n+1) := sqrt (a n)

-- Define sequence b_n
def b (n : ℕ) : ℝ :=
sqrt ((a (n+1)) / (a n) - (a n) / (a (n+1)))

-- Sum of first n terms of sequence b
def S (n : ℕ) : ℝ :=
∑ i in range n, b i

-- Formalize part (I)
theorem part_I (n : ℕ) (hn : n ≥ 1) : 0 < (a (n+1))^2 - (a n)^2 ∧ (a (n+1))^2 - (a n)^2 ≤ 1/4 :=
by sorry

-- Formalize part (II)
theorem part_II (n : ℕ) : S n < 3/4 :=
by sorry

end part_I_part_II_l180_180172


namespace chess_club_officer_selection_l180_180956

def number_of_officer_selections (n : ℕ) (alice bob charlie : ℕ) (accept_pos_if_both : ℕ → ℕ → Prop) (accept_pos : ℕ → ℕ → Prop) : ℕ :=
  let without_alice_bob := 23 * 22 * 21
  let with_alice_bob := 3 * 2 * 22
  without_alice_bob + with_alice_bob

theorem chess_club_officer_selection :
  number_of_officer_selections 25 1 2 3 (λ a b, a = 1 ∧ b = 2) (λ c a, a ≠ 1 → c ≠ 3) = 10758 :=
by
  unfold number_of_officer_selections
  sorry

end chess_club_officer_selection_l180_180956


namespace volume_of_tetrahedron_zero_l180_180519

theorem volume_of_tetrahedron_zero 
  (a b c : ℝ)
  (h1 : sqrt (a^2 + b^2) = 3)
  (h2 : sqrt (b^2 + c^2) = 4)
  (h3 : sqrt (c^2 + a^2) = 5)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0) :
  (1 / 6) * a * b * c = 0 := 
sorry

end volume_of_tetrahedron_zero_l180_180519


namespace midpoint_and_distance_l180_180464

def initial_midpoint (a b c d : ℝ) : ℝ × ℝ :=
  ((a + c) / 2, (b + d) / 2)

def new_midpoint (a b c d : ℝ) : ℝ × ℝ :=
  ((a + 3 + (c - 5)) / 2, (b + 5 + (d - 3)) / 2)

def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt (((fst p - fst q) ^ 2) + ((snd p - snd q) ^ 2))

theorem midpoint_and_distance (a b c d : ℝ) 
  (m n : ℝ) (h₀ : (m, n) = initial_midpoint a b c d) :
  let mid_new := new_midpoint a b c d in
  fst mid_new = m - 1 ∧ snd mid_new = n + 1 ∧ distance (m, n) mid_new = Real.sqrt 2 :=
by
  let m' := initial_midpoint a b c d
  let M' := new_midpoint a b c d
  let dist := distance (m, n) M'
  split
  { sorry }  -- Proof that fst M' = m - 1
  { sorry }  -- Proof that snd M' = n + 1
  { sorry }  -- Proof that dist = Real.sqrt 2

end midpoint_and_distance_l180_180464


namespace find_n_l180_180309

theorem find_n (n a b : ℕ) (h1 : a > b) (h2 : n = (4 * a * b) / (a - b)) : n > 4 ∧ (n % 4 ≠ 3 ∨ ¬ nat.prime n) := by
  sorry

end find_n_l180_180309


namespace exponent_of_five_in_30_factorial_l180_180044

theorem exponent_of_five_in_30_factorial : 
  nat.factorial_prime_exponent 30 5 = 7 := 
sorry

end exponent_of_five_in_30_factorial_l180_180044


namespace min_gennadys_needed_l180_180655

variables (A B V G : ℕ)

theorem min_gennadys_needed
  (hA : A = 45)
  (hB : B = 122)
  (hV : V = 27)
  (hG : ∀ i, i < 121 → A + V < 121 → G ≥ 49) :
  G = 49 :=
sorry

end min_gennadys_needed_l180_180655


namespace complex_solution_l180_180912

theorem complex_solution (z : ℂ) (h : complex.I * z = 2 - complex.I) : z = -1 - 2 * complex.I :=
sorry

end complex_solution_l180_180912


namespace AD_bisects_angle_XAY_l180_180975

-- Definitions of the geometry setup
noncomputable def triangle (A B C : Type*) := (A B C : Point)
noncomputable def circle (O : Point) (r : ℝ) := { Q : Point | (Q.x - O.x)^2 + (Q.y - O.y)^2 = r^2 }
noncomputable def bisector_internal (A B C D : Point) := ∃ L, L.point_on_line(D, A) ∧ L.point_on_line(D, middle(B, C))

-- Given points A, B, C forming a triangle
variables (A B C D M X Y : Point)

-- The circumcircle of triangle ABC
axiom circumcircle_of_triangle_ABC (A B C : Point) :
  circle(M, MB.dist)

-- Line through D intersects ω at X and Y
axiom line_through_D_intersects_omega (D X Y : Point) :
  collinear {D, X, Y} ∧ X ∈ (omega M (dist M B)) ∧ Y ∈ (omega M (dist M B))

-- Theorem to prove AD bisects ∠XAY
theorem AD_bisects_angle_XAY (A B C M D X Y : Point)
  (h1 : circumcircle_of_triangle_ABC A B C)
  (h2 : line_through_D_intersects_omega D X Y)
  (h3 : bisector_internal A B C D) :
  bisects (line_through D A) (angle X A Y) :=
sorry

end AD_bisects_angle_XAY_l180_180975


namespace travel_with_two_transfers_l180_180517

universe u

-- Assuming a set of tram stops as vertices and intersections of diagonals in a convex polygon
variable {Stop : Type u} [decidable_eq Stop] [fintype Stop]

-- Given: No three diagonals intersect at a single point
axiom no_three_diagonals_intersect : ∀ {a b c d e f : Stop},
  ¬(a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a ≠ d ∧ b ≠ e ∧ c ≠ f ∧
    (∀ {p : Stop}, p ∈ {a, b, c} → p ≠ d) ∧ 
    (∀ {q : Stop}, q ∈ {a, b, c} → q ≠ e) ∧ 
    (∀ {r : Stop}, r ∈ {a, b, c} → r ≠ f))

-- Given: Tram routes are established along some of the diagonals such that each stop is served by at least one tram route
axiom tram_routes_cover : ∀ (s : Stop), ∃ (d1 d2 : Stop), 
  s = d1 ∨ s = d2 ∨ (s = midpoint d1 d2)  -- This assumes a function midpoint:: Stop → Stop → Stop

-- To prove: From any stop to any other stop, it is possible to travel with at most two transfers
theorem travel_with_two_transfers (A B : Stop) : 
  ∃ (P Q : Stop), 
    (A = P ∨ 
    (P = midpoint (some_diagonal P Q) A)) ∧ 
    (B = Q ∨ 
    (Q = midpoint (some_diagonal P Q) B)) :=
sorry

end travel_with_two_transfers_l180_180517


namespace fair_betting_scheme_fair_game_l180_180191

noncomputable def fair_game_stakes (L L_k : ℕ → ℚ) : Prop :=
  ∀ k: ℕ, 1 < k → L_k (k+1) = (35/36) * L_k k

theorem fair_betting_scheme_fair_game :
  ∃ L_k : ℕ → ℚ, fair_game_stakes (λ k, L_k k) (λ k, L_k k) :=
begin
  let L_k : ℕ → ℚ := λ k, (35 / 36) ^ k,
  use L_k,
  unfold fair_game_stakes,
  intros k hk,
  rw [mul_assoc, ←mul_pow],
  ring,
end

end fair_betting_scheme_fair_game_l180_180191


namespace heartbeats_during_race_l180_180273

theorem heartbeats_during_race 
  (heart_rate : ℕ) -- average heartbeats per minute
  (pace : ℕ) -- minutes per mile
  (distance : ℕ) -- distance in miles
  (heart_rate_avg : heart_rate = 160) -- condition (1)
  (pace_per_mile : pace = 6) -- condition (2)
  (race_distance : distance = 20) -- condition (3)
  : heart_rate * (pace * distance) = 19200 := 
by
  rw [heart_rate_avg, pace_per_mile, race_distance]
  exact eq.refl 19200

end heartbeats_during_race_l180_180273


namespace bernoulli_poly_diff_bernoulli_poly_sum_l180_180547

-- Part (a)
theorem bernoulli_poly_diff (B : ℕ → ℂ → ℂ) (n : ℕ) (z : ℂ) (h : n ≥ 2) :
  B n (z + 1) - B n z = n * z^(n - 1) :=
sorry

-- Part (b)
theorem bernoulli_poly_sum (B : ℕ → ℂ → ℂ) (n k : ℕ) (h : n ≥ 1) :
  (∑ i in Finset.range (k + 1), i^n) = (1 / (n + 1)) * (B (n + 1) (k + 1) - B (n + 1) 0) :=
sorry

end bernoulli_poly_diff_bernoulli_poly_sum_l180_180547


namespace sec_tan_identity_l180_180826

theorem sec_tan_identity (x : ℝ) (h : Real.sec x + Real.tan x = 7 / 3) : Real.sec x - Real.tan x = 3 / 7 := 
by
  sorry

end sec_tan_identity_l180_180826


namespace base_salary_calculation_l180_180667

theorem base_salary_calculation
  (prev_salary : ℝ)
  (comm_rate : ℝ)
  (sale_price : ℝ)
  (min_sales : ℝ)
  (commission : ℝ)
  (total_commission : ℝ)
  (base_salary : ℝ) :
  prev_salary = 75000 ∧ comm_rate = 0.15 ∧ sale_price = 750 ∧ min_sales = 266.67 ∧
  commission = comm_rate * sale_price ∧ total_commission = min_sales * commission ∧
  prev_salary = base_salary + total_commission →
  base_salary = 45000 :=
begin
  sorry
end

end base_salary_calculation_l180_180667


namespace min_gennadies_l180_180647

noncomputable section

def minGennadiesNeeded (alexanders borises vasilies : Nat) : Nat :=
  let needed_gaps := borises - 1
  let total_others := alexanders + vasilies
  if needed_gaps > total_others then needed_gaps - total_others else 0

theorem min_gennadies (alexanders borises vasilies : Nat) (h_alex: alexanders = 45) (h_boris: borises = 122) (h_vasil: vasilies = 27):
  minGennadiesNeeded alexanders borises vasilies = 49 := by
  rw [h_alex, h_boris, h_vasil]
  simp [minGennadiesNeeded]
  sorry

end min_gennadies_l180_180647


namespace sum_primes_powers_mod_l180_180456

/-- Definition of prime numbers -/
def prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

/-- List of primes as a sequence of natural numbers -/
def primes : ℕ → ℕ
| 0     := 2
| (n+1) := (Nat.succ (primes n^4 - 1)).find (λ p, prime p)

/-- Main theorem statement -/
theorem sum_primes_powers_mod (n : ℕ) (h1: 2 ≤ n) (h2: n ≤ 2550) : 
  (∑ k in Finset.range'.succ' (2, 2550), primes k ^ ((primes k)^4 - 1)) % 2550 = 2548 := 
sorry

end sum_primes_powers_mod_l180_180456


namespace average_points_per_tenth_grader_l180_180402

theorem average_points_per_tenth_grader (x : ℕ) (h1 : 0 < x) :
  (let tenth_graders := 10 * x
       total_participants := x + tenth_graders
       total_points := total_participants * (total_participants - 1) / 2
       average_points_tenth_graders := total_points / tenth_graders in
   average_points_tenth_graders = (11 * x - 1) / 2) :=
by
  sorry

end average_points_per_tenth_grader_l180_180402


namespace probability_both_l180_180586

variable (Ω : Type) [ProbabilitySpace Ω]
variable (A B : Event Ω)
variable [decidable (A ∧ B)]

def probability_over_60 (P : ℙ ∋ A) : ℝ := 0.20
def probability_hypertension_given_over_60 (P : ℙ ∋ B | A) : ℝ := 0.45

theorem probability_both :
  (probability_over_60 Ω A) * (probability_hypertension_given_over_60 Ω B A) = 0.09 :=
by
  sorry

end probability_both_l180_180586


namespace dodecahedron_has_5_cubes_l180_180548

def dodecahedron : Type := sorry -- Define the type for dodecahedron vertices.

def is_cube (s : set dodecahedron) : Prop := sorry -- Define a predicate to check if a set of vertices forms a cube.

theorem dodecahedron_has_5_cubes :
  ∃ (cubes : finset (finset dodecahedron)), 
  (∀ s ∈ cubes, is_cube s ∧ finset.card s = 8) ∧
  finset.card cubes = 5 :=
  sorry

end dodecahedron_has_5_cubes_l180_180548


namespace interest_after_3_years_l180_180955

def compound_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r)^t

def interest_earned (P : ℝ) (A : ℝ) : ℝ :=
  A - P

theorem interest_after_3_years :
  ∀ (P : ℝ) (r : ℝ) (t : ℕ), P = 2000 ∧ r = 0.02 ∧ t = 3 →
  let A := compound_interest P r t in
  interest_earned P A = 122.416 :=
by 
  intros
  sorry

end interest_after_3_years_l180_180955


namespace digit_100th_is_4_digit_1000th_is_3_l180_180588

noncomputable section

def digit_100th_place : Nat :=
  4

def digit_1000th_place : Nat :=
  3

theorem digit_100th_is_4 (n : ℕ) (h1 : n ∈ {m | m = 100}) : digit_100th_place = 4 := by
  sorry

theorem digit_1000th_is_3 (n : ℕ) (h1 : n ∈ {m | m = 1000}) : digit_1000th_place = 3 := by
  sorry

end digit_100th_is_4_digit_1000th_is_3_l180_180588


namespace max_correct_answers_l180_180568

theorem max_correct_answers :
  ∃ (c w b : ℕ), c + w + b = 25 ∧ 4 * c - 3 * w = 57 ∧ c = 18 :=
by {
  sorry
}

end max_correct_answers_l180_180568


namespace alternating_draws_probability_l180_180234

theorem alternating_draws_probability :
  let num_white := 6
  let num_black := 6
  let total_balls := num_white + num_black
  let successful_sequences := 2  -- "BWBWBWBWBWBW" and "WBWBWBWBWBWB"
  let total_arrangements := Nat.choose total_balls num_white
  (successful_sequences / total_arrangements : ℚ) = 1 / 462 :=
by
  -- Definitions
  let num_white := 6
  let num_black := 6
  let total_balls := num_white + num_black
  let successful_sequences := 2  -- There are two sequences that alternate perfectly

  -- Calculate the total number of ways to arrange 6 black and 6 white balls
  let total_arrangements := Nat.choose total_balls num_white  -- This calculates binomial coefficient C(12, 6)

  -- Given the above, show the probability
  have h1 : (successful_sequences / total_arrangements : ℚ) = 1 / 462,
  sorry

  exact h1

end alternating_draws_probability_l180_180234


namespace volume_inequality_l180_180290

theorem volume_inequality {
  x : ℕ
  -- Define the conditions: the dimensions of the prism and the restriction on x
  (hx_pos : x > 3)
  (hx4 : x = 4 → (x + 3) * (x - 3) * (x ^ 3 + 27) < 5000)
  (hx5 : x = 5 → (x + 3) * (x - 3) * (x ^ 3 + 27) < 5000)
  (hx6 : x = 6 → ¬ ((x + 3) * (x - 3) * (x ^ 3 + 27) < 5000))
} : ∃ (x1 x2 : ℕ), x1 = 4 ∧ x2 = 5 :=
by {
  -- Area to insert the detailed proof steps
  sorry
}

end volume_inequality_l180_180290


namespace length_of_ac_l180_180169

def is_acute_triangle (A B C : Point) : Prop :=
  ∠ABC < π / 2 ∧ ∠BCA < π / 2 ∧ ∠CAB < π / 2
  
def circumcircle_radius (A B C : Point) : ℝ :=
  1

def center_of_circle_passing_through_A_C_and_orthocenter (A B C : Point) (H : Point) : Prop :=
  (circle_center A C H).inside_circumcircle A B C

theorem length_of_ac (A B C : Point) (H : Point) (hR : circumcircle_radius A B C = 1) :
  center_of_circle_passing_through_A_C_and_orthocenter A B C H →
  is_acute_triangle A B C →
  dist A C = sqrt 3 :=
sorry

end length_of_ac_l180_180169


namespace fair_game_condition_l180_180193

variables (n : ℕ) (L : ℝ) {p : ℕ → ℝ}

-- Define the probability p_k for the k-th player.
def probability (k : ℕ) : ℝ := (35.0 / 36.0) ^ k

-- Define the expected value of the k-th player.
def expected_value (L : ℝ) (Lk : ℝ) (k : ℕ) : ℝ := L * probability k - Lk

-- Define the conditions of the fair game.
def fair_game := ∀ k, expected_value L (L * probability k) k = 0

-- Main theorem stating that the game is fair if stakes decrease proportionally by a factor of 35/36.
theorem fair_game_condition (k : ℕ) (L : ℝ) :
  fair_game :=
by
  sorry

end fair_game_condition_l180_180193


namespace no_solution_a_solution_b_l180_180310

def f (n : ℕ) : ℕ :=
  if n = 0 then
    0
  else
    n / 7 + f (n / 7)

theorem no_solution_a :
  ¬ ∃ n : ℕ, 7 ^ 399 ∣ n! ∧ ¬ 7 ^ 400 ∣ n! := sorry

theorem solution_b :
  {n : ℕ | 7 ^ 400 ∣ n! ∧ ¬ 7 ^ 401 ∣ n!} = {2401, 2402, 2403, 2404, 2405, 2406, 2407} := sorry

end no_solution_a_solution_b_l180_180310


namespace exponent_of_5_in_30_fact_l180_180025

def count_powers_of_5 (n : ℕ) : ℕ :=
  if n < 5 then 0
  else n / 5 + count_powers_of_5 (n / 5)

theorem exponent_of_5_in_30_fact : count_powers_of_5 30 = 7 := 
  by
    sorry

end exponent_of_5_in_30_fact_l180_180025


namespace total_flowers_collected_l180_180598

/- Definitions for the given conditions -/
def maxFlowers : ℕ := 50
def arwenTulips : ℕ := 20
def arwenRoses : ℕ := 18
def arwenSunflowers : ℕ := 6

def elrondTulips : ℕ := 2 * arwenTulips
def elrondRoses : ℕ := if 3 * arwenRoses + elrondTulips > maxFlowers then maxFlowers - elrondTulips else 3 * arwenRoses

def galadrielTulips : ℕ := if 3 * elrondTulips > maxFlowers then maxFlowers else 3 * elrondTulips
def galadrielRoses : ℕ := if 2 * arwenRoses + galadrielTulips > maxFlowers then maxFlowers - galadrielTulips else 2 * arwenRoses

def galadrielSunflowers : ℕ := 0 -- she didn't pick any sunflowers
def legolasSunflowers : ℕ := arwenSunflowers + galadrielSunflowers
def legolasRemaining : ℕ := maxFlowers - legolasSunflowers
def legolasRosesAndTulips : ℕ := legolasRemaining / 2
def legolasTulips : ℕ := legolasRosesAndTulips
def legolasRoses : ℕ := legolasRosesAndTulips

def arwenTotal : ℕ := arwenTulips + arwenRoses + arwenSunflowers
def elrondTotal : ℕ := elrondTulips + elrondRoses
def galadrielTotal : ℕ := galadrielTulips + galadrielRoses + galadrielSunflowers
def legolasTotal : ℕ := legolasTulips + legolasRoses + legolasSunflowers

def totalFlowers : ℕ := arwenTotal + elrondTotal + galadrielTotal + legolasTotal

theorem total_flowers_collected : totalFlowers = 194 := by
  /- This will be where the proof goes, but we leave it as a placeholder. -/
  sorry

end total_flowers_collected_l180_180598


namespace solution_set_real_implies_conditions_l180_180174

variable {a b c : ℝ}

theorem solution_set_real_implies_conditions (h1 : a ≠ 0)
  (h2 : ∀ x : ℝ, a * x^2 + b * x + c < 0) : a < 0 ∧ (b^2 - 4 * a * c) < 0 := 
sorry

end solution_set_real_implies_conditions_l180_180174


namespace find_m_l180_180774

theorem find_m 
(x0 m : ℝ)
(h1 : m ≠ 0)
(h2 : x0^2 - x0 + m = 0)
(h3 : (2 * x0)^2 - 2 * x0 + 3 * m = 0)
: m = -2 :=
sorry

end find_m_l180_180774


namespace train_length_l180_180583

noncomputable def speed_kph := 56  -- speed in km/hr
def time_crossing := 9  -- time in seconds
noncomputable def speed_mps := speed_kph * 1000 / 3600  -- converting km/hr to m/s

theorem train_length : speed_mps * time_crossing = 140 := by
  -- conversion and result approximation
  sorry

end train_length_l180_180583


namespace fair_game_condition_l180_180194

variables (n : ℕ) (L : ℝ) {p : ℕ → ℝ}

-- Define the probability p_k for the k-th player.
def probability (k : ℕ) : ℝ := (35.0 / 36.0) ^ k

-- Define the expected value of the k-th player.
def expected_value (L : ℝ) (Lk : ℝ) (k : ℕ) : ℝ := L * probability k - Lk

-- Define the conditions of the fair game.
def fair_game := ∀ k, expected_value L (L * probability k) k = 0

-- Main theorem stating that the game is fair if stakes decrease proportionally by a factor of 35/36.
theorem fair_game_condition (k : ℕ) (L : ℝ) :
  fair_game :=
by
  sorry

end fair_game_condition_l180_180194


namespace percentage_flags_of_both_colors_l180_180213

theorem percentage_flags_of_both_colors (F C : ℕ) (h_even : F % 2 = 0) 
  (h_FC : F = 2 * C) (h_blue : 0.6 * C = (0.6 * C).to_nat) 
  (h_red : 0.7 * C = (0.7 * C).to_nat) : 
  (0.3 * C).to_nat = 30 * C / 100 :=
by sorry

end percentage_flags_of_both_colors_l180_180213


namespace possible_values_of_a_l180_180981

noncomputable def quadratic_function_range (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f x = x^2 - 2 * a * x + 2 * a + 4 ∧ f x ≥ 1

theorem possible_values_of_a (a : ℝ) :
  (quadratic_function_range (λ x, x^2 - 2 * a * x + 2 * a + 4) a) ↔ (a = -1 ∨ a = 3) :=
by
  sorry

end possible_values_of_a_l180_180981


namespace total_heartbeats_during_race_l180_180270

namespace Heartbeats

def avg_heart_beats_per_minute : ℕ := 160
def pace_minutes_per_mile : ℕ := 6
def race_distance_miles : ℕ := 20

theorem total_heartbeats_during_race :
  (race_distance_miles * pace_minutes_per_mile * avg_heart_beats_per_minute = 19200) :=
by
  sorry

end Heartbeats

end total_heartbeats_during_race_l180_180270


namespace sequence_2018_value_l180_180374

theorem sequence_2018_value (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, a (n + 1) - a n = (-1 / 2) ^ n) :
  a 2018 = (2 * (1 - (1 / 2) ^ 2018)) / 3 :=
by sorry

end sequence_2018_value_l180_180374


namespace formula_an_formula_bn_sum_cn_l180_180460

noncomputable theory

-- Given conditions
def Sn (n : ℕ) : ℕ := n^2
def an (n : ℕ) : ℕ := if n = 1 then 1 else Sn n - Sn (n-1)
def bn (n : ℕ) : ℝ := (1/2)^(n-1)
def cn (n : ℕ) : ℝ := an n * bn n

-- Prove the general formula for an
theorem formula_an (n : ℕ) : an n = 2 * n - 1 := 
sorry

-- Prove the general formula for bn
theorem formula_bn (n : ℕ) : bn n = (1/2)^(n-1) :=
sorry

-- Prove the sum of the first n terms of the sequence cn
theorem sum_cn (n : ℕ) : ∑ i in Finset.range n, cn (i + 1) = 6 - (2 * n + 3) / 2^(n-1) :=
sorry

end formula_an_formula_bn_sum_cn_l180_180460


namespace regression_line_equation_l180_180349

variable (slope : ℝ) (center_x center_y : ℝ) (a : ℝ) 

-- Conditions
def estimated_slope (slope : ℝ) : Prop := slope = 1.23
def center_of_sample_points (center_x center_y : ℝ) : Prop := center_x = 4 ∧ center_y = 5

-- Define the regression line given point and slope
def regression_line (x : ℝ) : ℝ := slope * x + a

-- Define the center point substitution into the regression line
def center_point_substitution (center_x center_y : ℝ) : Prop := 
  center_y = slope * center_x + a

-- The correct answer we need to prove
def regression_line_equation_correct (a : ℝ) : Prop := 
  regression_line a = 1.23 * a + 0.08

theorem regression_line_equation : 
  estimated_slope slope → 
  center_of_sample_points center_x center_y → 
  center_point_substitution center_x center_y → 
  regression_line_equation_correct (4) :=
by
  sorry

end regression_line_equation_l180_180349


namespace fruit_price_is_correct_l180_180567

-- Definitions of each condition
def water_price : ℝ := 0.50
def snack_price : ℝ := 1.00
variable (fruit_price : ℝ)

-- Definition of the bundle cost
def bundle_cost (fruit_price : ℝ) :=
  water_price + 3 * snack_price + 2 * fruit_price

-- Condition of selling price
def selling_price : ℝ := 4.6

--Prove that fruit_price is 0.55
theorem fruit_price_is_correct : fruit_price = 0.55 :=
  by
  have h : bundle_cost fruit_price = selling_price, from sorry,
  have eq1 : water_price + 3 * snack_price + 2 * fruit_price = 4.6, by exact h,
  -- Further solving equation can be put here
  sorry

end fruit_price_is_correct_l180_180567


namespace Jules_total_blocks_walked_l180_180879

def vacation_cost : ℝ := 1000
def family_members : ℝ := 5
def charge_per_walk_start : ℝ := 2
def charge_per_block : ℝ := 1.25
def dogs_walked : ℕ := 20

def total_contribution_per_member : ℝ := vacation_cost / family_members

def total_blocks_walked (total_contribution_per_member : ℝ) 
                         (charge_per_walk_start : ℝ) 
                         (charge_per_block : ℝ) 
                         (dogs_walked : ℕ) : ℝ :=
  let earnings_per_dog := charge_per_walk_start + charge_per_block * (real.ceil (total_contribution_per_member / (charge_per_walk_start + charge_per_block * dogs_walked))) in
  earnings_per_dog * dogs_walked

theorem Jules_total_blocks_walked : 
  total_blocks_walked total_contribution_per_member charge_per_walk_start charge_per_block dogs_walked = 140 :=
  sorry

end Jules_total_blocks_walked_l180_180879


namespace sufficient_but_not_necessary_l180_180384

theorem sufficient_but_not_necessary (x : ℝ) : (x < 1) → (x * abs x - 2 < 0) :=
by {
  assume h : x < 1,
  sorry
}

end sufficient_but_not_necessary_l180_180384


namespace sec_minus_tan_l180_180791

theorem sec_minus_tan (x : ℝ) (h : real.sec x + real.tan x = 7 / 3) :
  real.sec x - real.tan x = 3 / 7 :=
sorry

end sec_minus_tan_l180_180791


namespace fair_betting_scheme_fair_game_l180_180189

noncomputable def fair_game_stakes (L L_k : ℕ → ℚ) : Prop :=
  ∀ k: ℕ, 1 < k → L_k (k+1) = (35/36) * L_k k

theorem fair_betting_scheme_fair_game :
  ∃ L_k : ℕ → ℚ, fair_game_stakes (λ k, L_k k) (λ k, L_k k) :=
begin
  let L_k : ℕ → ℚ := λ k, (35 / 36) ^ k,
  use L_k,
  unfold fair_game_stakes,
  intros k hk,
  rw [mul_assoc, ←mul_pow],
  ring,
end

end fair_betting_scheme_fair_game_l180_180189


namespace part1_part2_part3_l180_180362

def f (x : ℝ) (a : ℝ) : ℝ := (2^x - a) / (2^x + a)

-- Part (1): a = 1 makes f(x) an odd function
theorem part1 : ∀ (x : ℝ), f (-x) 1 = -f x 1 := sorry

-- Part (2): f(x) is increasing on ℝ
theorem part2 : ∀ (x1 x2 : ℝ), x1 < x2 → f x1 1 < f x2 1 := sorry

-- Part (3): maximum value of f(x) on (-∞, 1] is 1/3
theorem part3 : ∀ x : ℝ, x ≤ 1 → f x 1 ≤ f 1 1 := sorry

end part1_part2_part3_l180_180362


namespace number_of_days_l180_180388

theorem number_of_days (a b c : ℕ) (a_pos : a > 0) (b_pos : b > 0) (c_pos : c > 0) (same_eff : (1 / (a * b)) * c = 1) : 
  let parts_per_day_per_person := c / (a * b) in
  let parts_per_day_per_b_people := b * parts_per_day_per_person in
  let days_required := a / parts_per_day_per_b_people in
  days_required = a^2 / c :=
sorry

end number_of_days_l180_180388


namespace min_gennadys_l180_180605

-- Defining the basic constants for each name type
def Alexanders : Nat := 45
def Borises : Nat := 122
def Vasilies : Nat := 27

-- Define the proof statement to check the minimum number of Gennadys needed
theorem min_gennadys (a b v : Nat) (no_adjacent: a = 45 ∧ b = 122 ∧ v = 27) : ∃ g : Nat, g = 49 :=
by
  -- Using provided conditions
  cases no_adjacent with h_a h_bv
  cases h_bv with h_b h_v
  -- Correct answer derived from the solution
  use 49
  -- skipping proof details 
  sorry

end min_gennadys_l180_180605


namespace exists_cube_sum_in_interval_l180_180470

theorem exists_cube_sum_in_interval (n : ℕ) :
  ∃ (x y : ℕ), n - 4 * nat.sqrt n ≤ x^3 + y^3 ∧ x^3 + y^3 ≤ n + 4 * nat.sqrt n := sorry

end exists_cube_sum_in_interval_l180_180470


namespace minimum_gennadys_l180_180622

theorem minimum_gennadys (alexs borises vasilies x : ℕ) (h₁ : alexs = 45) (h₂ : borises = 122) (h₃ : vasilies = 27)
    (h₄ : ∀ i, i ∈ list.range (borises-1) → alexs + vasilies + x > i) : 
    x = 49 :=
by 
    sorry

end minimum_gennadys_l180_180622


namespace Keith_spent_on_CD_player_l180_180069

theorem Keith_spent_on_CD_player 
  (spent_on_speakers : ℝ)
  (spent_on_tires : ℝ)
  (total_spent : ℝ)
  (spent_on_CD_player : ℝ) : 
  spent_on_speakers = 136.01 → 
  spent_on_tires = 112.46 → 
  total_spent = 387.85 → 
  spent_on_CD_player = total_spent - (spent_on_speakers + spent_on_tires) → 
  spent_on_CD_player = 139.38 :=
by
  intros h₁ h₂ h₃ h₄
  rw [h₁, h₂, h₃, h₄]
  sorry

end Keith_spent_on_CD_player_l180_180069


namespace solve_inequality_l180_180950

theorem solve_inequality (a x : ℝ) (h : a < 0) :
  (56 * x^2 + a * x - a^2 < 0) ↔ (a / 8 < x ∧ x < -a / 7) :=
by
  sorry

end solve_inequality_l180_180950


namespace heartbeats_during_race_l180_180272

theorem heartbeats_during_race 
  (heart_rate : ℕ) -- average heartbeats per minute
  (pace : ℕ) -- minutes per mile
  (distance : ℕ) -- distance in miles
  (heart_rate_avg : heart_rate = 160) -- condition (1)
  (pace_per_mile : pace = 6) -- condition (2)
  (race_distance : distance = 20) -- condition (3)
  : heart_rate * (pace * distance) = 19200 := 
by
  rw [heart_rate_avg, pace_per_mile, race_distance]
  exact eq.refl 19200

end heartbeats_during_race_l180_180272


namespace scientific_notation_of_508_billion_l180_180159

theorem scientific_notation_of_508_billion:
  (508 * (10:ℝ)^9) = (5.08 * (10:ℝ)^11) := 
begin
  sorry
end

end scientific_notation_of_508_billion_l180_180159


namespace cameron_paint_area_l180_180673

theorem cameron_paint_area :
  let length := 15
  let width := 12
  let height := 10
  let door_window_area := 80
  let num_bedrooms := 4
  let area_per_bedroom := 2 * (length * height) + 2 * (width * height)
  let paintable_area_per_bedroom := area_per_bedroom - door_window_area
  let total_paintable_area := num_bedrooms * paintable_area_per_bedroom
  total_paintable_area = 1840 :=
by
  unfold length width height door_window_area num_bedrooms area_per_bedroom paintable_area_per_bedroom total_paintable_area
  sorry

end cameron_paint_area_l180_180673


namespace area_of_G1G2G3_l180_180083

open Classical

variables {A B C P : Type}
variables [point : AddGroup A] (B C P : A)
variables (G1 G2 G3 : Point)

def area (triangle : set A) : ℝ := sorry

def is_point_inside (P : A) (ABC : set A) := sorry

def centroid (triangle : set A) : Point := sorry

theorem area_of_G1G2G3 (ABC : set A) (P : Point) (G1 G2 G3 : Point) (h_inside : is_point_inside P ABC):
  area ABC = 24 →
  G1 = centroid ({P, B, C}) →
  G2 = centroid ({P, C, A}) →
  G3 = centroid ({P, A, B}) →
  area ({G1, G2, G3}) = 24 / 9 :=
by
  sorry

end area_of_G1G2G3_l180_180083


namespace sum_of_coefficients_l180_180303

noncomputable def slant_asymptote (p q : ℚ[X]) (x : ℚ) : Prop :=
  ∀ x, y = q/x

lemma slant_asymptote_of_frac (p q : PolyRat) (x : ℝ) :
  p ≠ 0 →
  (p / q).asymptote x = asymptote (3 * x + 11) →
  p = 3 * x ^ 2 + 5 * x - 11 ∧ q = x - 2 :=
begin
  -- proof starts here
  sorry
end

theorem sum_of_coefficients :
  sum_of_coefficients ((3 : ℚ) + 11) = 14 :=
begin
  -- proof starts here
  sorry
end

end sum_of_coefficients_l180_180303


namespace find_QF_length_l180_180371

def parabola := λ (x y : ℝ), y^2 = 8 * x

noncomputable def point_distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem find_QF_length :
  ∀ (xQ yQ xF yF xP yP : ℝ),
    parabola xQ yQ →
    xF = 2 →
    yF = 0 →
    yP = 0 →
    let FP := point_distance xF yF xP yP in
    let FQ := point_distance xF yF xQ yQ in
    FP = 4 * FQ →
    FQ = 3 :=
  by
    intros
    sorry

end find_QF_length_l180_180371


namespace min_gennadys_needed_l180_180650

variables (A B V G : ℕ)

theorem min_gennadys_needed
  (hA : A = 45)
  (hB : B = 122)
  (hV : V = 27)
  (hG : ∀ i, i < 121 → A + V < 121 → G ≥ 49) :
  G = 49 :=
sorry

end min_gennadys_needed_l180_180650


namespace inscribed_circle_radius_of_equilateral_triangle_l180_180571

noncomputable def circumscribed_radius : ℝ :=
  3 + 2 * Real.sqrt 3

def equilateral_triangle_inscribed_circle_radius (R : ℝ) : ℝ :=
  R * Real.sqrt 3 * (2 - Real.sqrt 3) / (2 * (2 + Real.sqrt 3))

theorem inscribed_circle_radius_of_equilateral_triangle (R : ℝ) (h : R = 3 + 2 * Real.sqrt 3) :
  equilateral_triangle_inscribed_circle_radius R = 3 / 2 :=
by
  rw h
  sorry

end inscribed_circle_radius_of_equilateral_triangle_l180_180571


namespace divisor_of_23_quotient_7_remainder_2_l180_180107

-- Definitions and proof statement
theorem divisor_of_23_quotient_7_remainder_2 : ∃ d : ℕ, 23 = 7 * d + 2 ∧ d = 3 :=
by 
  -- Definitions of the problem
  let dividend := 23
  let quotient := 7
  let remainder := 2

  -- proof stating
  use 3 -- candidate for divisor
  have proof : 23 = 7 * 3 + 2 := rfl
  exact ⟨proof, rfl⟩

end divisor_of_23_quotient_7_remainder_2_l180_180107


namespace equal_sides_of_inscribed_pentagon_l180_180114

theorem equal_sides_of_inscribed_pentagon 
  (A B C D E : Type)
  [metric_space Type]
  [circle_inscribed (A B C D E : fin 5)]
  (angle_A angle_B angle_C angle_D angle_E : ℝ) 
  (h1 : angle_A = angle_B)
  (h2 : angle_B = angle_C)
  (h3 : angle_C = angle_D)
  (h4 : angle_D = angle_E) :
  side_length A B = side_length B C ∧
  side_length B C = side_length C D ∧
  side_length C D = side_length D E ∧
  side_length D E = side_length E A :=
begin
  sorry
end

end equal_sides_of_inscribed_pentagon_l180_180114


namespace exponent_of_five_in_factorial_l180_180049

theorem exponent_of_five_in_factorial:
  (nat.factors 30!).count 5 = 7 :=
begin
  sorry
end

end exponent_of_five_in_factorial_l180_180049


namespace scientific_notation_of_508_billion_yuan_l180_180161

-- Definition for a billion in the international system.
def billion : ℝ := 10^9

-- The amount of money given in the problem.
def amount_in_billion (n : ℝ) : ℝ := n * billion

-- The Lean theorem statement to prove.
theorem scientific_notation_of_508_billion_yuan :
  amount_in_billion 508 = 5.08 * 10^11 :=
by
  sorry

end scientific_notation_of_508_billion_yuan_l180_180161


namespace triangle_AED_area_l180_180942

theorem triangle_AED_area
  (A B C D E : Type)
  [geometry A B C D]
  (AB BC AC BE AE ED : ℝ)
  (AB_eq_6 : AB = 6)
  (BC_eq_8 : BC = 8)
  (E_perpendicular_to_AC : is_perpendicular E B AC)
  (area_triangle_AED : real_triangle_area AE ED = 14.4) : 
  area_triangle_AED = 14.4 :=
by sorry

end triangle_AED_area_l180_180942


namespace value_of_m_l180_180166

def is_decreasing_on_positive_real (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 < x → x < y → f y < f x

def power_function (m : ℝ) (x : ℝ) := (m ^ 2 - m - 1) * x ^ (m ^ 2 + m - 3)

theorem value_of_m (m : ℝ) :
  is_decreasing_on_positive_real (power_function m) → m = -1 :=
sorry

end value_of_m_l180_180166


namespace plane_perpendicular_to_line_l_perpendicular_to_both_planes_l180_180337

variables {α β : Type} [plane α] [plane β]
variable (l : line)
variable perpendicular : ∀ {a b : Type} [plane a] [plane b], Prop
variable intersection : ∀ {a b : Type} [plane a] [plane b], line

-- Given conditions
axiom alpha_perpendicular_beta : perpendicular α β
axiom alpha_intersects_beta_at_l : intersection α β = l

-- Statement to prove
theorem plane_perpendicular_to_line_l_perpendicular_to_both_planes
  (φ : Type) [plane φ]
  (perpendicular_to_l : perpendicular φ l) :
  perpendicular φ α ∧ perpendicular φ β := 
sorry

end plane_perpendicular_to_line_l_perpendicular_to_both_planes_l180_180337


namespace fraction_multiplication_result_l180_180536

theorem fraction_multiplication_result :
  (5 * 7) / 8 = 4 + 3 / 8 :=
by
  sorry

end fraction_multiplication_result_l180_180536


namespace sec_minus_tan_l180_180806

theorem sec_minus_tan (x : ℝ) (h : Real.sec x + Real.tan x = 7 / 3) : Real.sec x - Real.tan x = 3 / 7 :=
by
  sorry

end sec_minus_tan_l180_180806


namespace total_ears_l180_180784

theorem total_ears (total_puppies droopy_ear_puppies pointed_ear_puppies ears_per_puppy : ℕ)
  (h1 : total_puppies = 250)
  (h2 : droopy_ear_puppies = 150)
  (h3 : pointed_ear_puppies = 100)
  (h4 : ears_per_puppy = 2) :
  droopy_ear_puppies * ears_per_puppy + pointed_ear_puppies * ears_per_puppy = 500 := 
by
  rw [h2, h3, h4]
  norm_num
  sorry

end total_ears_l180_180784


namespace solution_set_of_inequality_l180_180173

theorem solution_set_of_inequality :
  {x : ℝ | (3 * x - 1) / (2 - x) ≥ 0} = {x : ℝ | 1 / 3 ≤ x ∧ x < 2} :=
by
  sorry

end solution_set_of_inequality_l180_180173


namespace exponent_of_5_in_30_fact_l180_180027

def count_powers_of_5 (n : ℕ) : ℕ :=
  if n < 5 then 0
  else n / 5 + count_powers_of_5 (n / 5)

theorem exponent_of_5_in_30_fact : count_powers_of_5 30 = 7 := 
  by
    sorry

end exponent_of_5_in_30_fact_l180_180027


namespace fair_game_stakes_ratio_l180_180187

theorem fair_game_stakes_ratio (n : ℕ) (deck_size : ℕ) (player_count : ℕ)
  (L : ℕ → ℝ) : 
  deck_size = 36 → player_count = 36 → 
  (∀ k : ℕ, k < player_count - 1 → 
    (L (k + 1)) / (L k) = 35 / 36) :=
by
  intros h_deck_size h_player_count k hk
  simp [h_deck_size, h_player_count, hk]
  sorry

end fair_game_stakes_ratio_l180_180187


namespace no_solution_for_given_eqn_l180_180494

open Real

noncomputable def no_real_solution (x : ℝ) : Prop :=
1 - log (sin x) = cos x 

theorem no_solution_for_given_eqn :
  ¬ ∃ x : ℝ, 1 - log (sin x) = cos x := 
by {
  sorry
}

end no_solution_for_given_eqn_l180_180494


namespace find_slope_intercept_l180_180246

variable (x y : ℝ)

def line_equation : Prop :=
  (λ x y : ℝ, ⟨2, -1⟩ • (⟨x, y⟩ - ⟨3, -4⟩) = 0)

theorem find_slope_intercept :
  line_equation x y →
  ∃ m b : ℝ, (∀ x : ℝ, y = m * x + b) ∧ m = 2 ∧ b = -10 :=
by
  intros
  existsi (2 : ℝ)
  existsi (-10 : ℝ)
  sorry

end find_slope_intercept_l180_180246


namespace curveG_symmetric_curveG_unique_common_point_curveG_common_points_x_y_curveG_max_distance_l180_180749

-- Define the curve G as a set of points (x, y) satisfying the equation x^3 + y^3 - 6xy = 0
def curveG (x y : ℝ) : Prop :=
  x^3 + y^3 - 6 * x * y = 0

-- Prove symmetry of curveG with respect to the line y = x
theorem curveG_symmetric (x y : ℝ) (h : curveG x y) : curveG y x :=
  sorry

-- Prove unique common point with the line x + y - 6 = 0
theorem curveG_unique_common_point : ∃! p : ℝ × ℝ, curveG p.1 p.2 ∧ p.1 + p.2 = 6 :=
  sorry

-- Prove curveG has at least one common point with the line x - y + 1 = 0
theorem curveG_common_points_x_y : ∃ p : ℝ × ℝ, curveG p.1 p.2 ∧ p.1 - p.2 + 1 = 0 :=
  sorry

-- Prove the maximum distance from any point on the curveG to the origin is 3√2
theorem curveG_max_distance : ∀ p : ℝ × ℝ, curveG p.1 p.2 → p.1 > 0 → p.2 > 0 → (p.1^2 + p.2^2 ≤ 18) :=
  sorry

end curveG_symmetric_curveG_unique_common_point_curveG_common_points_x_y_curveG_max_distance_l180_180749


namespace find_m_l180_180775

-- Define the vectors as given in the conditions
def a : ℝ × ℝ := (-1, 2)
def b (m : ℝ) : ℝ × ℝ := (m, 1)

-- Define the condition for perpendicular vectors using the dot product
def perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

-- Formalize the statement to be proved
theorem find_m (m : ℝ) (h : perpendicular a (a.1 + b m.1, a.2 + b m.2)) : m = 7 := by
  sorry

end find_m_l180_180775


namespace correct_statement_l180_180540

-- We define the conditions of the problem as individual statements
def statement1 : Prop := ¬∃ (S : Set α), (S = ∅)
def statement2 : Prop := ∀ (S : Set α), ∃ (a b : Set α), a ≠ b ∧ a ⊆ S ∧ b ⊆ S
def statement3 : Prop := ∀ (S : Set α), (∅ ⊂ S)
def statement4 : Prop := ∀ (A : Set α), (∅ ⊆ A → A ≠ ∅)

-- The problem is to verify which statement among these is correct.
-- Here, only statement 4 is correct.
theorem correct_statement : ¬ statement1 ∧ ¬ statement2 ∧ ¬ statement3 ∧ statement4 :=
begin
  sorry,  -- skipping the proof
end

end correct_statement_l180_180540


namespace fixed_point_P_l180_180324

variable (a : ℝ)
def f (x : ℝ) := 4 + a^(x-1)

theorem fixed_point_P (a : ℝ) : f a 1 = 5 :=
by
  sorry

end fixed_point_P_l180_180324


namespace smallest_y_l180_180131

theorem smallest_y (x y : ℤ) (h : x * y + 3 * x + 2 * y = 1) : ∃ y_min, y_min = -10 :=
by
  use -10
  sorry

end smallest_y_l180_180131


namespace boat_sinking_weight_range_l180_180231

theorem boat_sinking_weight_range
  (L_min L_max : ℝ)
  (B_min B_max : ℝ)
  (D_min D_max : ℝ)
  (sink_rate : ℝ)
  (down_min down_max : ℝ)
  (min_weight max_weight : ℝ)
  (condition1 : 3 ≤ L_min ∧ L_max ≤ 5)
  (condition2 : 2 ≤ B_min ∧ B_max ≤ 3)
  (condition3 : 1 ≤ D_min ∧ D_max ≤ 2)
  (condition4 : sink_rate = 0.01)
  (condition5 : 0.03 ≤ down_min ∧ down_max ≤ 0.06)
  (condition6 : ∀ D, D_min ≤ D ∧ D ≤ D_max → (D - down_max) ≥ 0.5)
  (condition7 : min_weight = down_min * (10 / 0.01))
  (condition8 : max_weight = down_max * (10 / 0.01)) :
  min_weight = 30 ∧ max_weight = 60 := 
sorry

end boat_sinking_weight_range_l180_180231


namespace factor_of_M_l180_180894

theorem factor_of_M (a b c : ℕ) (h₀ : a ≠ 0) (h₁ : a < 10) (h₂ : b < 10) (h₃ : c < 10) : 
  1 ∣ (101010 * a + 10001 * b + 100 * c) :=
sorry

end factor_of_M_l180_180894


namespace binom_512_512_eq_one_l180_180680

theorem binom_512_512_eq_one : nat.choose 512 512 = 1 :=
by sorry

end binom_512_512_eq_one_l180_180680


namespace negation_of_P_l180_180493

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ (m : ℕ), m ∣ n → m = 1 ∨ m = n)

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def P : Prop := ∀ n : ℕ, is_prime n → is_odd n

theorem negation_of_P : ¬ P ↔ ∃ n : ℕ, is_prime n ∧ ¬ is_odd n :=
by sorry

end negation_of_P_l180_180493


namespace min_gennadies_l180_180626

theorem min_gennadies 
  (n_Alexanders : ℕ) (n_Borises : ℕ) (n_Vasilies : ℕ) 
  (x_Gennadies : ℕ) 
  (h_Alexanders : n_Alexanders = 45) 
  (h_Borises   : n_Borises = 122) 
  (h_Vasilies  : n_Vasilies = 27) 
  (h_condition : ∀ p : ℕ, p = n_Borises - 1 → p = 121) 
  (h_total     : ∀ q : ℕ, q = 45 + 27 → q = 72)
  : x_Gennadies = 49 := 
sorry

end min_gennadies_l180_180626


namespace imaginary_part_of_z_l180_180973

-- Define the complex number z
def z : ℂ := 1 - complex.I * complex.I

-- Prove that the imaginary part of z is -1
theorem imaginary_part_of_z : z.im = -1 :=
by 
  sorry

end imaginary_part_of_z_l180_180973


namespace necessary_but_not_sufficient_l180_180716

-- Define non-zero vectors in a vector space
variables {V : Type*} [AddCommGroup V] [VectorSpace ℝ V]
variables (a b : V)

-- Conditions: non-zero vectors
variables (ha : a ≠ 0) (hb : b ≠ 0)

example (h : a + b = 0) : ∃ k : ℝ, b = -k • a := by
  sorry

example (h_parallel : ∃ k : ℝ, b = k • a) : ¬ (a + b = 0) ↔ k ≠ -1 := by
  sorry

-- The main statement
theorem necessary_but_not_sufficient : 
  (a + b = 0 → ∃ k : ℝ, b = -k • a) ∧ (∃ k : ℝ, b = k • a → ¬ (a + b = 0)) :=
by sorry

end necessary_but_not_sufficient_l180_180716


namespace find_XY_squared_l180_180450

/- Let ABC be an acute scalene triangle with circumcircle ω. 
The tangents to ω at B and C intersect at T. 
Let X and Y be the projections of T onto lines AB and AC, respectively. 
Suppose BT = CT = 18, BC = 24, and TX^2 + TY^2 + XY^2 = 1458. 
We need to prove XY^2 = 1386. -/

variables {A B C T X Y : Type}
variables [triangle A B C] [circumcircle ω A B C]
variables [tangent_to ω B T] [tangent_to ω C T]
variables [projection T X (line A B)] [projection T Y (line A C)]
variables {BT CT BC TX TY XY : ℝ}
variables (h1 : BT = 18) (h2 : CT = 18) (h3 : BC = 24) 
variables (h4 : TX^2 + TY^2 + XY^2 = 1458)

theorem find_XY_squared : XY^2 = 1386 := 
by 
  sorry

end find_XY_squared_l180_180450


namespace min_gennadys_l180_180604

-- Defining the basic constants for each name type
def Alexanders : Nat := 45
def Borises : Nat := 122
def Vasilies : Nat := 27

-- Define the proof statement to check the minimum number of Gennadys needed
theorem min_gennadys (a b v : Nat) (no_adjacent: a = 45 ∧ b = 122 ∧ v = 27) : ∃ g : Nat, g = 49 :=
by
  -- Using provided conditions
  cases no_adjacent with h_a h_bv
  cases h_bv with h_b h_v
  -- Correct answer derived from the solution
  use 49
  -- skipping proof details 
  sorry

end min_gennadys_l180_180604


namespace math_problem_proof_l180_180414

def ratio_area_BFD_square_ABCE (x : ℝ) (AF FE DE CD : ℝ) (h1 : AF = FE / 3) (h2 : CD = 3 * DE) : Prop :=
  let AE := (AF + FE)
  let area_square := (AE)^2
  let area_triangle_BFD := area_square - (1/2 * AF * (AE - FE) + 1/2 * (AE - FE) * FE + 1/2 * DE * CD)
  (area_triangle_BFD / area_square) = (1/16)
  
theorem math_problem_proof (x AF FE DE CD : ℝ) (h1 : AF = FE / 3) (h2 : CD = 3 * DE) (area_ratio : area_triangle_BFD / area_square = 1/16) : ratio_area_BFD_square_ABCE x AF FE DE CD h1 h2 :=
sorry

end math_problem_proof_l180_180414


namespace complex_imaginary_part_l180_180311

-- Define the complex numbers involved and their arithmetic
theorem complex_imaginary_part :
  let z1 := (1 : ℂ) / (-2 + (1 : ℂ) * I)
  let z2 := (1 : ℂ) / (1 - 2 * I)
  Im (z1 + z2) = 1 / 5 :=
by
  sorry

end complex_imaginary_part_l180_180311


namespace sum_binom_lemma_l180_180935

theorem sum_binom_lemma (n m : ℕ) :
  (∑ k in Finset.range(1 + n), (-1)^(n - k) * k^m * Nat.choose n k) =
    if 0 < m ∧ m < n then 0 else if m = n then n.factorial else 0 :=
sorry

end sum_binom_lemma_l180_180935


namespace intersection_of_A_and_B_l180_180734

-- Given sets A and B
def A : Set ℤ := { -1, 0, 1, 2 }
def B : Set ℤ := { 0, 2, 3 }

-- Prove that the intersection of A and B is {0, 2}
theorem intersection_of_A_and_B : A ∩ B = {0, 2} := 
by
  sorry

end intersection_of_A_and_B_l180_180734


namespace total_sweaters_knit_l180_180434

-- Definitions from condition a)
def monday_sweaters : ℕ := 8
def tuesday_sweaters : ℕ := monday_sweaters + 2
def wednesday_sweaters : ℕ := tuesday_sweaters - 4
def thursday_sweaters : ℕ := wednesday_sweaters
def friday_sweaters : ℕ := monday_sweaters / 2

-- Theorem statement
theorem total_sweaters_knit : 
  monday_sweaters + tuesday_sweaters + wednesday_sweaters + thursday_sweaters + friday_sweaters = 34 :=
  by
    sorry

end total_sweaters_knit_l180_180434


namespace exponent_of_5_in_30_factorial_l180_180009

theorem exponent_of_5_in_30_factorial : 
  (nat.factors 30!).count 5 = 7 :=
sorry

end exponent_of_5_in_30_factorial_l180_180009


namespace solve_quadratic_1_solve_quadratic_2_solve_cubic_no_real_roots_solve_quadratic_equivalent_solve_circle_point_l180_180948

-- Question 1
theorem solve_quadratic_1 (x : ℝ) (h : 4 * x^2 - 12 * x + 9 = 0) : x = 3 / 2 := 
sorry

-- Question 2
theorem solve_quadratic_2 (x : ℝ) (h : x^2 - real.pi^2 = 0) : x = real.pi ∨ x = -real.pi :=
sorry

-- Question 3
theorem solve_cubic (a : ℝ) (h : a^3 - a = 0) : a = 0 ∨ a = 1 ∨ a = -1 :=
sorry

-- Question 4
theorem no_real_roots (x : ℝ) (h : x^2 + 1 + real.pi = 2 * x) : false :=
sorry

-- Question 5
theorem solve_quadratic_equivalent (x : ℝ) (h : x^2 = real.pi^2 - 2 * real.pi + 1) : 
  x = real.pi - 1 ∨ x = 1 - real.pi :=
sorry

-- Question 6
theorem solve_circle_point (x y : ℝ) (h : x^2 + y^2 - 4 * x + 4 = 0) : x = 2 ∧ y = 0 :=
sorry

end solve_quadratic_1_solve_quadratic_2_solve_cubic_no_real_roots_solve_quadratic_equivalent_solve_circle_point_l180_180948


namespace find_a_for_odd_function_l180_180148

theorem find_a_for_odd_function (a : ℝ) :
  (∀ x : ℝ, (f x) + (f (-x)) = 0) 
  → a = -2 :=
by
  -- Define function f
  let f := λ x : ℝ, (3^(x + 1) - 1) / (3^x - 1) + a * (sin x + cos x) ^ 2
  sorry

end find_a_for_odd_function_l180_180148


namespace minimum_gennadys_l180_180619

theorem minimum_gennadys (alexs borises vasilies x : ℕ) (h₁ : alexs = 45) (h₂ : borises = 122) (h₃ : vasilies = 27)
    (h₄ : ∀ i, i ∈ list.range (borises-1) → alexs + vasilies + x > i) : 
    x = 49 :=
by 
    sorry

end minimum_gennadys_l180_180619


namespace count_triangles_l180_180326

def valid_triangles (n : ℕ) : ℕ :=
  (n * (n - 2) * (2 * n - 5)) / 24

theorem count_triangles (n : ℕ) (h : n ≥ 3) : 
  (some (V : ℕ), V = valid_triangles n) :=
  sorry

end count_triangles_l180_180326


namespace monotonic_increasing_interval_l180_180492

noncomputable def function := λ x: ℝ, -x^2

theorem monotonic_increasing_interval :
  {x : ℝ | ∀ y, function y ≤ function x ↔ y ≤ x} = Set.Iic 0 := 
sorry

end monotonic_increasing_interval_l180_180492


namespace find_a_l180_180488

def f (a : ℝ) (x : ℝ) : ℝ := log (10^x + 1) + a * x

theorem find_a (a : ℝ) : (∀ x : ℝ, f a x = f a (-x)) → a = -1/2 := by
  intros h
  sorry

end find_a_l180_180488


namespace concert_distance_l180_180924

/-- We have the following conditions:
1. Mrs. Hilt and her sister drove 32 miles and then stopped for gas.
2. Her sister put 28 gallons of gas in the car.
3. They were left with 46 miles to drive.

We want to prove that the total distance to the concert was 78 miles. --/
theorem concert_distance (d1 d2 : ℕ) (d3 : ℕ) (h1 : d1 = 32) (h2 : d3 = 46) : d1 + d3 = 78 :=
by { sorry }

# This states that given:
# d1 = 32 (miles before stopping for gas)
# d3 = 46 (miles left to drive after getting gas)
# the total distance (d1 + d3) equals 78.

end concert_distance_l180_180924


namespace sec_minus_tan_l180_180819

theorem sec_minus_tan
  (x : ℝ)
  (h1 : Real.sec x + Real.tan x = 7 / 3)
  (h2 : (Real.sec x + Real.tan x) * (Real.sec x - Real.tan x) = 1) :
  Real.sec x - Real.tan x = 3 / 7 :=
by
  sorry

end sec_minus_tan_l180_180819


namespace adam_total_cost_l180_180267

theorem adam_total_cost 
    (sandwiches_count : ℕ)
    (sandwiches_price : ℝ)
    (chips_count : ℕ)
    (chips_price : ℝ)
    (water_count : ℕ)
    (water_price : ℝ)
    (sandwich_discount : sandwiches_count = 4 ∧ sandwiches_price = 4 ∧ sandwiches_count = 3 + 1)
    (tax_rate : ℝ)
    (initial_tax_rate : tax_rate = 0.10)
    (chips_cost : chips_count = 3 ∧ chips_price = 3.50)
    (water_cost : water_count = 2 ∧ water_price = 2) : 
  (3 * sandwiches_price + chips_count * chips_price + water_count * water_price) * (1 + tax_rate) = 29.15 := 
by
  sorry

end adam_total_cost_l180_180267


namespace jerry_games_before_birthday_l180_180875

def num_games_before (current received : ℕ) : ℕ :=
  current - received

theorem jerry_games_before_birthday : 
  ∀ (current received before : ℕ), current = 9 → received = 2 → before = num_games_before current received → before = 7 :=
by
  intros current received before h_current h_received h_before
  rw [h_current, h_received] at h_before
  exact h_before

end jerry_games_before_birthday_l180_180875


namespace expected_cereal_difference_l180_180669

-- Define the days Bob eats sweetened or unsweetened cereal
def days_in_week := 6
def weekdays := 52 * days_in_week
def sundays := 52 + 1 -- 52 Sundays + 1 extra day in a non-leap year
def total_days := weekdays + sundays

-- Probability definitions
def weekday_prob_unsweetened := (4 / 8 : ℝ)
def weekday_prob_sweetened := (4 / 8 : ℝ)
def sunday_prob_unsweetened := (42 / 64 + 20 / 64 : ℝ)
def sunday_prob_sweetened := (2 / 64 : ℝ)

-- Expected values calculations
def expected_weekdays_unsweetened := weekday_prob_unsweetened * days_in_week * 52
def expected_weekdays_sweetened := weekday_prob_sweetened * days_in_week * 52
def expected_sundays_unsweetened := sunday_prob_unsweetened * 52
def expected_sundays_sweetened := sunday_prob_sweetened * 52

def expected_total_unsweetened := expected_weekdays_unsweetened + expected_sundays_unsweetened
def expected_total_sweetened := expected_weekdays_sweetened + expected_sundays_sweetened

noncomputable def expected_difference := expected_total_unsweetened - expected_total_sweetened

theorem expected_cereal_difference :
  expected_difference = 47.75 := by
  sorry

end expected_cereal_difference_l180_180669


namespace rectangular_prism_volume_l180_180255

theorem rectangular_prism_volume
  (l w h : ℝ)
  (face1 : l * w = 6)
  (face2 : w * h = 8)
  (face3 : l * h = 12) : l * w * h = 24 := sorry

end rectangular_prism_volume_l180_180255


namespace exponent_of_five_in_30_factorial_l180_180036

theorem exponent_of_five_in_30_factorial : 
  nat.factorial_prime_exponent 30 5 = 7 := 
sorry

end exponent_of_five_in_30_factorial_l180_180036


namespace KimSweaterTotal_l180_180437

theorem KimSweaterTotal :
  let monday := 8
  let tuesday := monday + 2
  let wednesday := tuesday - 4
  let thursday := wednesday
  let friday := monday / 2
  monday + tuesday + wednesday + thursday + friday = 34 := by
  sorry

end KimSweaterTotal_l180_180437


namespace solution_l180_180900
noncomputable def problem_statement : Prop :=
  ∀ (a : ℝ), (a + complex.i) * 2 * complex.i > 0 → a = -1

theorem solution : problem_statement := sorry

end solution_l180_180900


namespace correct_location_l180_180204

-- Define the possible options
inductive Location
| A : Location
| B : Location
| C : Location
| D : Location

-- Define the conditions
def option_A : Prop := ¬(∃ d, d ≠ "right")
def option_B : Prop := ¬(∃ d, d ≠ 900)
def option_C : Prop := ¬(∃ d, d ≠ "west")
def option_D : Prop := (∃ d₁ d₂, d₁ = "west" ∧ d₂ = 900)

-- The objective is to prove that option D is the correct description of the location
theorem correct_location : ∃ l, l = Location.D → 
  (option_A ∧ option_B ∧ option_C ∧ option_D) :=
by
  sorry

end correct_location_l180_180204


namespace Sarah_must_sell_637_burgers_to_recover_7000_l180_180472

noncomputable def min_burgers_to_recover_investment (investment : ℕ) (price_per_burger : ℕ) (cost_per_burger : ℕ) : ℕ :=
  let net_gain_per_burger := price_per_burger - cost_per_burger
  let min_burgers := (investment + net_gain_per_burger - 1) / net_gain_per_burger  -- Ceiling of investment / net_gain_per_burger
  min_burgers

theorem Sarah_must_sell_637_burgers_to_recover_7000 :
  min_burgers_to_recover_investment 7000 15 4 = 637 :=
by
  -- Define the variables
  let investment := 7000
  let price_per_burger := 15
  let cost_per_burger := 4
  let net_gain_per_burger := price_per_burger - cost_per_burger  -- 11
  let min_burgers := (investment + net_gain_per_burger - 1) / net_gain_per_burger  -- Ceiling of investment / net_gain_per_burger
  -- Show the minimum number of burgers
  show min_burgers = 637 from sorry

end Sarah_must_sell_637_burgers_to_recover_7000_l180_180472


namespace frequency_is_approximately_fifteen_l180_180155

-- Define variables for start and end times, and number of glows
def start_time : Nat := 1 * 3600 + 57 * 60 + 58  -- Convert 1:57:58 to seconds
def end_time : Nat := 3 * 3600 + 20 * 60 + 47    -- Convert 3:20:47 to seconds
def number_of_glows : ℝ := 331.27

-- Calculate the frequency in seconds per glow
def frequency : ℝ := (end_time - start_time) / number_of_glows

-- Prove that the frequency is approximately 15 seconds per glow
theorem frequency_is_approximately_fifteen : abs (frequency - 15) < 1 :=
by
  -- Statement intentionally left unfinished for illustration purposes
  sorry

end frequency_is_approximately_fifteen_l180_180155


namespace smallest_k_is_22_l180_180503

-- Define the recursive sequence (a_n)
noncomputable def a : ℕ → ℝ
| 0       := 1
| 1       := real.root (23 : ℝ) 3
| (n + 2) := a (n + 1) * (a n)^3

-- Define the sum of the sequence b_n
def b (n : ℕ) : ℤ :=
  match n with
  | 0       := 0
  | 1       := 1
  | (n + 2) := b (n + 1) + 3 * b n

def s : ℕ → ℤ
| 0       := 0
| (n + 1) := s n + b (n + 1)

-- Prove that the smallest k such that a_1 * a_2 * ... * a_k is an integer is 22
theorem smallest_k_is_22 : ∃ k : ℕ, k = 22 ∧ (s k % 23 = 0) := 
by sorry

end smallest_k_is_22_l180_180503


namespace value_of_4b_minus_a_l180_180967

theorem value_of_4b_minus_a (a b : ℕ) (h1 : a > b) (h2 : x^2 - 20*x + 96 = (x - a)*(x - b)) : 4*b - a = 20 :=
  sorry

end value_of_4b_minus_a_l180_180967


namespace problem_statement_period_property_symmetry_property_zero_property_l180_180363

noncomputable def f (x : ℝ) : ℝ := Real.cos (x + Real.pi / 3)

theorem problem_statement : ¬(∀ x : ℝ, (Real.pi / 2 < x ∧ x < Real.pi) → f x > f (x + ε))
  → ∃ x : ℝ, f (x + Real.pi) = 0 :=
by
  intro h
  use Real.pi / 6
  sorry

theorem period_property : ∀ k : ℤ, f (x + 2 * k * Real.pi) = f x :=
by
  intro k
  sorry

theorem symmetry_property : ∀ y : ℝ, f (8 * Real.pi / 3 - y) = f (8 * Real.pi / 3 + y) :=
by
  intro y
  sorry

theorem zero_property : f (Real.pi / 6 + Real.pi) = 0 :=
by
  sorry

end problem_statement_period_property_symmetry_property_zero_property_l180_180363


namespace num_senior_in_sample_l180_180576

-- Definitions based on conditions
def total_students : ℕ := 2000
def senior_students : ℕ := 700
def sample_size : ℕ := 400

-- Theorem statement for the number of senior students in the sample
theorem num_senior_in_sample : 
  (senior_students * sample_size) / total_students = 140 :=
by 
  sorry

end num_senior_in_sample_l180_180576


namespace min_silver_dollars_l180_180164

theorem min_silver_dollars : ∃ (x : ℕ), 
  -- let A be the number of dollars received by the eldest son plus maid 
  ∃ (A : ℕ), A = 1 + x / 4 ∧
  -- let B be the number of dollars received by the second son plus maid 
  ∃ (B : ℕ), B = 1 + (x - A) / 4 ∧
  -- let C be the number of dollars received by the third son plus maid 
  ∃ (C : ℕ), C = 1 + (x - A - B) / 4 ∧
  -- let D be the number of dollars received by the forth son plus maid 
  ∃ (D : ℕ), D = 1 + (x - A - B - C) / 4 ∧
  -- each remaining son receives 1/4 of the remainder plus another 1 dollar for the maid
  ∃ (R : ℕ), R = x - A - B - C - D ∧
  x = A + B + C + D + R ∧ x % 4 = 1 ∧
  x % 5 = 1 ∧
  x = 1021 :=
begin
  sorry
end

end min_silver_dollars_l180_180164


namespace initial_apples_l180_180945

theorem initial_apples (AP : ℕ) (added_by_susan : 8) (total_apples : 17) : 
  AP = 9 := 
by
  have h1 : total_apples = AP + added_by_susan by sorry
  sorry

end initial_apples_l180_180945


namespace min_gennadies_l180_180642

noncomputable section

def minGennadiesNeeded (alexanders borises vasilies : Nat) : Nat :=
  let needed_gaps := borises - 1
  let total_others := alexanders + vasilies
  if needed_gaps > total_others then needed_gaps - total_others else 0

theorem min_gennadies (alexanders borises vasilies : Nat) (h_alex: alexanders = 45) (h_boris: borises = 122) (h_vasil: vasilies = 27):
  minGennadiesNeeded alexanders borises vasilies = 49 := by
  rw [h_alex, h_boris, h_vasil]
  simp [minGennadiesNeeded]
  sorry

end min_gennadies_l180_180642


namespace beetle_speed_l180_180592

theorem beetle_speed
  (distance_ant : ℝ )
  (time_minutes : ℝ)
  (distance_beetle : ℝ) 
  (distance_percent_less : ℝ)
  (time_hours : ℝ)
  (beetle_speed_kmh : ℝ)
  (h1 : distance_ant = 600)
  (h2 : time_minutes = 10)
  (h3 : time_hours = time_minutes / 60)
  (h4 : distance_percent_less = 0.25)
  (h5 : distance_beetle = distance_ant * (1 - distance_percent_less))
  (h6 : beetle_speed_kmh = distance_beetle / time_hours) : 
  beetle_speed_kmh = 2.7 :=
by 
  sorry

end beetle_speed_l180_180592


namespace GM_parallel_KH_l180_180737

-- Define the orthocenter condition including the altitudes and their intersection
variable {A B C H D E F G K M : Point}

-- Given the orthocenter H with altitudes intersecting at H
axiom orthocenter 
  (h₁ : H = orthocenter A B C)
  (h₂ : altitude A B C H = AD)
  (h₃ : altitude B C A H = BE)
  (h₄ : altitude C A B H = CF)

-- EF intersects AD at G and AK intersects BC at M
axiom intersections 
  (h₅ : line_through EF AD G)
  (h₆ : line_through AK BC M)
  
-- AK is the diameter of the circumcircle of ΔABC
axiom diameter 
  (h₇ : diameter AK (circumcircle A B C))

-- Statement that needs to be proven
theorem GM_parallel_KH : GM ∥ KH :=
by
  -- Insert proof here.
  sorry

end GM_parallel_KH_l180_180737


namespace ratio_b_a_l180_180719

theorem ratio_b_a (a b : ℝ) (h : ∃ (points : Fin 4 → ℝ × ℝ),
  (dist (points 0) (points 1) = a ∧
   dist (points 0) (points 2) = a ∧
   dist (points 0) (points 3) = a ∧
   dist (points 1) (points 2) = b ∧
   dist (points 1) (points 3) = a ∧
   dist (points 2) (points 3) = 2a)) :
  b = a * Real.sqrt 3 := sorry

end ratio_b_a_l180_180719


namespace part_one_part_two_l180_180332

-- Part (I) Statement
theorem part_one
  (a : ℕ → ℝ) -- The sequence a_n
  (S : ℕ → ℝ) -- The sequence of sums S_n
  (h1 : ∀ n, a n ≠ 0) -- Condition: a_n ≠ 0
  (h2 : ∀ n, S n = ∑ i in finset.range n, a (i + 1)) -- Condition: S_n is the sum of first n terms
  (h3 : ∀ n, 2 ^ n * a (n + 1) = S n) -- Condition: b = 0 and 2^n * a_(n+1) = S_n
  :
  ∀ n, let T := ∑ i in finset.range n, (a (i + 2) / a (i + 1))
        in T = (1 / 2) + (1 / 2) * n - 1 / (2 ^ n) :=
sorry

-- Part (II) Statement
theorem part_two
  (a : ℕ → ℝ) -- The sequence a_n
  (S : ℕ → ℝ) -- The sequence of sums S_n
  (h1 : ∀ n, a n ≠ 0) -- Condition: a_n ≠ 0
  (h2 : ∀ n, S n = ∑ i in finset.range n, a (i + 1)) -- Condition: S_n is the sum of first n terms
  (h3 : ∀ n, 2 ^ n * a (n + 1) = n ^ 2 * S n) -- Condition: b = 2 and 2^n * a_(n+1) = n^2 * S_n
  :
  ∃ n, ∀ m, m ≠ n → (S (n + 1) / S n) > (S (m + 1) / S m) ∧ (S (n + 1) / S n) = 17 / 8 :=
sorry

end part_one_part_two_l180_180332


namespace count_ordered_quadruples_l180_180709

theorem count_ordered_quadruples :
  {q : (ℝ × ℝ × ℝ × ℝ) // (q.1.1 ^ 2 + q.1.2 ^ 2 + q.2.1 ^ 2 + q.2.2 ^ 2 = 6) 
  ∧ ((q.1.1 + q.1.2 + q.2.1 + q.2.2) * (q.1.1 ^ 3 + q.1.2 ^ 3 + q.2.1 ^ 3 + q.2.2 ^ 3) = 36)}.to_finset.card = 15 :=
by sorry

end count_ordered_quadruples_l180_180709


namespace sum_of_arithmetic_sequence_l180_180534

theorem sum_of_arithmetic_sequence :
  ∃ a1 a10 n S, a1 = -4 ∧ a10 = 37 ∧ n = 10 ∧ 
  (S = (n * (a1 + a10)) / 2) ∧ S = 165 :=
by
  use [-4, 37, 10, 165]
  sorry

end sum_of_arithmetic_sequence_l180_180534


namespace type_B_toy_cost_l180_180589

/-
Allie has 15 toys in total worth $84. There are 3 types of toys: Type A, Type B, and Type C.
One Type A toy is worth $12. There are 3 Type A toys. 
There are 6 Type B toys and they all have the same value. 
There are 6 Type C toys and they have a different value than Type B toys. 
The total value of all Type C toys is $30.

Prove that the cost of one Type B toy is $3.
-/
theorem type_B_toy_cost (total_value : ℕ) (total_toys : ℕ) (value_A : ℕ) (num_A : ℕ) (num_B : ℕ) (num_C : ℕ) (total_value_C : ℕ) :
  total_value = 84 ∧ total_toys = 15 ∧ value_A = 12 ∧ num_A = 3 ∧ num_B = 6 ∧ num_C = 6 ∧ total_value_C = 30 →
  (total_value - num_A * value_A - total_value_C) / num_B = 3 :=
by
  intro h
  cases h with h1 h234567
  cases h234567 with h2 h34567
  cases h34567 with h3 h4567
  cases h4567 with h4 h567
  cases h567 with h5 h67
  cases h67 with h6 h7
  rw [h1, h2, h3, h4, h5, h6, h7]
  norm_num
  sorry

end type_B_toy_cost_l180_180589


namespace minimum_gennadys_l180_180624

theorem minimum_gennadys (alexs borises vasilies x : ℕ) (h₁ : alexs = 45) (h₂ : borises = 122) (h₃ : vasilies = 27)
    (h₄ : ∀ i, i ∈ list.range (borises-1) → alexs + vasilies + x > i) : 
    x = 49 :=
by 
    sorry

end minimum_gennadys_l180_180624


namespace greatest_visible_unit_cubes_in_12x12x12_cube_l180_180230

theorem greatest_visible_unit_cubes_in_12x12x12_cube :
  (let face_units := 12 * 12 in
   let total_units := 3 * face_units in
   let shared_edge_units := 3 * (12 - 1) in
   let corner_unit := 1 in
   total_units - shared_edge_units + corner_unit = 400) := by
  sorry

end greatest_visible_unit_cubes_in_12x12x12_cube_l180_180230


namespace min_gennadys_l180_180636

theorem min_gennadys (alexanders borises vasilies : ℕ) (x : ℕ) 
    (h1 : alexanders = 45)
    (h2 : borises = 122)
    (h3 : vasilies = 27)
    (h4 : x = 49)
    (h5 : borises - 1 = alexanders + vasilies + x) :
  x = 49 := 
begin
  sorry,
end

end min_gennadys_l180_180636


namespace total_colored_pencils_l180_180675

-- Define Cheryl's number of colored pencils
def Cheryl := ℕ

-- Define Cyrus's number of colored pencils
def Cyrus := ℕ

-- Define Madeline's number of colored pencils
def Madeline := ℕ

-- Given conditions as Lean definitions
def cheryl_thrice_cyrus (C : Cheryl) (Y : Cyrus) : Prop := C = 3 * Y
def madeline_half_cheryl (C : Cheryl) (M : Madeline) : Prop := M = 63 ∧ 63 = C / 2

-- Total colored pencils theorem
theorem total_colored_pencils (C : Cheryl) (Y : Cyrus) (M : Madeline) 
  (h1 : cheryl_thrice_cyrus C Y) (h2 : madeline_half_cheryl C M) : C + Y + M = 231 :=
by
  sorry

end total_colored_pencils_l180_180675


namespace part1_part2_l180_180772

-- Conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := { x | (x - 2) / (x - 3) < 0 }
def B (a : ℝ) : Set ℝ := { x | (x - a) * (x - a^2 - 2) < 0 }
def p (x : ℝ) : Prop := x ∈ A
def q (x : ℝ) : Prop := x ∈ B

-- Part (1)
theorem part1 (a : ℝ) (h : a = 1/2) :
  (U \ B a) ∪ A = { x | x ≤ 1 / 2 ∨ x > 2 } :=
sorry

-- Part (2)
theorem part2 :
  { a : ℝ | ∀ x, q x → p x ∧ ∃ x, q x ∧ ¬p x } = { a | a ∈ (-∞, -1] ∪ [1, 2] } :=
sorry

end part1_part2_l180_180772


namespace sec_tan_identity_l180_180824

theorem sec_tan_identity (x : ℝ) (h : Real.sec x + Real.tan x = 7 / 3) : Real.sec x - Real.tan x = 3 / 7 := 
by
  sorry

end sec_tan_identity_l180_180824


namespace find_x_for_h_equal_12_l180_180094

def f (x : ℝ) : ℝ := 18 / (x + 2)
def h (x : ℝ) : ℝ := 2 * (f⁻¹' {x} : Set ℝ).toReal -- using inverse function image and ensuring it's converted to a real number.

theorem find_x_for_h_equal_12 : h (9 / 4) = 12 := by sorry

end find_x_for_h_equal_12_l180_180094


namespace total_fencing_l180_180499

open Real

def playground_side_length : ℝ := 27
def garden_length : ℝ := 12
def garden_width : ℝ := 9
def flower_bed_radius : ℝ := 5
def sandpit_side1 : ℝ := 7
def sandpit_side2 : ℝ := 10
def sandpit_side3 : ℝ := 13

theorem total_fencing : 
    4 * playground_side_length + 
    2 * (garden_length + garden_width) + 
    2 * Real.pi * flower_bed_radius + 
    (sandpit_side1 + sandpit_side2 + sandpit_side3) = 211.42 := 
    by sorry

end total_fencing_l180_180499


namespace sum_of_two_consecutive_negative_integers_l180_180167

theorem sum_of_two_consecutive_negative_integers (n : ℤ) (h : n * (n + 1) = 812) (h_neg : n < 0 ∧ (n + 1) < 0) : 
  n + (n + 1) = -57 :=
sorry

end sum_of_two_consecutive_negative_integers_l180_180167


namespace probability_identical_cubes_after_rotation_l180_180514

theorem probability_identical_cubes_after_rotation :
  let total_ways_to_paint_cubes := 3 * 2^4 * 1 -- Simplistic model for single cube painting per adjacency condition
  let identical_configurations := 66 -- Hypothetically derived using symmetry considerations
  let total_configurations := total_ways_to_paint_cubes ^ 3
  3 * total_configurations = total_ways_to_paint_cubes := by
  have h1 : total_configurations = 48^3 := by sorry
  have h2 : identical_configurations = 66 := by sorry
  have h3 : total_ways_to_paint_cubes = 48 := by sorry

  show (identical_configurations / total_configurations = 66 / 110592) from by sorry

end probability_identical_cubes_after_rotation_l180_180514


namespace chuck_total_play_area_l180_180286

noncomputable def chuck_play_area (leash_radius : ℝ) : ℝ :=
  let middle_arc_area := (1 / 2) * Real.pi * leash_radius^2
  let corner_arc_area := 2 * (1 / 4) * Real.pi * leash_radius^2
  middle_arc_area + corner_arc_area

theorem chuck_total_play_area (leash_radius : ℝ) (shed_width shed_length : ℝ) 
  (h_radius : leash_radius = 4) (h_width : shed_width = 4) (h_length : shed_length = 6) :
  chuck_play_area leash_radius = 16 * Real.pi :=
by
  sorry

end chuck_total_play_area_l180_180286


namespace julia_played_tag_with_4_kids_on_tuesday_l180_180432

variable (k_monday : ℕ) (k_diff : ℕ)

theorem julia_played_tag_with_4_kids_on_tuesday
  (h_monday : k_monday = 16)
  (h_diff : k_monday = k_tuesday + 12) :
  k_tuesday = 4 :=
by
  sorry

end julia_played_tag_with_4_kids_on_tuesday_l180_180432


namespace equilateral_triangle_of_orthocenter_divides_altitudes_l180_180937

variable {A B C H A1 B1 : Type}

-- Define points and their relationships
def is_orthocenter (H : Type) (A B C : Type) : Prop :=
sorry

def divides_altitudes_same_ratio (H A1 B A B1 : Type) : Prop :=
A1 * H * B = B1 * H * A

theorem equilateral_triangle_of_orthocenter_divides_altitudes
  (A B C H A1 B1 : Type)
  (h_orthocenter : is_orthocenter H A B C)
  (h_ratio : divides_altitudes_same_ratio H A1 B A B1) : 
  A = B ∧ B = C ∧ C = A :=
sorry

end equilateral_triangle_of_orthocenter_divides_altitudes_l180_180937


namespace sec_minus_tan_l180_180793

theorem sec_minus_tan (x : ℝ) (h : real.sec x + real.tan x = 7 / 3) :
  real.sec x - real.tan x = 3 / 7 :=
sorry

end sec_minus_tan_l180_180793


namespace lightest_height_is_135_l180_180972

-- Definitions based on the problem conditions
def heights_in_ratio (a b c d : ℕ) : Prop :=
  ∃ x : ℕ, a = 3 * x ∧ b = 4 * x ∧ c = 5 * x ∧ d = 6 * x

def height_condition (a c d : ℕ) : Prop :=
  d + a = c + 180

-- Lean statement describing the proof problem
theorem lightest_height_is_135 :
  ∀ (a b c d : ℕ),
  heights_in_ratio a b c d →
  height_condition a c d →
  a = 135 :=
by
  intro a b c d
  intro h_in_ratio h_condition
  sorry

end lightest_height_is_135_l180_180972


namespace equal_diagonals_of_convex_quadrilateral_l180_180404

theorem equal_diagonals_of_convex_quadrilateral 
  {A B C D M N : Type} [ConvexQuadrilateral A B C D]
  (mid_AB : midpoint M A B)
  (mid_CD : midpoint N C D)
  (equal_angles : ∀ (l : Line), l.pass_through M N ∧ 
    angle_with_diagonals l A C = angle_with_diagonals l B D) :
  distance A C = distance B D :=
sorry

end equal_diagonals_of_convex_quadrilateral_l180_180404


namespace probability_fewer_heads_than_tails_is_793_over_2048_l180_180532

noncomputable def probability_fewer_heads_than_tails (n : ℕ) : ℝ :=
(793 / 2048 : ℚ)

theorem probability_fewer_heads_than_tails_is_793_over_2048 :
  probability_fewer_heads_than_tails 12 = (793 / 2048 : ℚ) :=
sorry

end probability_fewer_heads_than_tails_is_793_over_2048_l180_180532


namespace find_population_growth_rates_l180_180165

-- Define the conditions as functions of x, y, z
def population_growth (x y z : ℕ) : ℚ :=
  (3 / 2)^x * (128 / 225)^y * (5 / 6)^z

-- Define the main theorem to prove the question with the answer
theorem find_population_growth_rates :
  ∃ (x y z : ℕ), x = 4 ∧ y = 1 ∧ z = 2 ∧ population_growth x y z = 2 :=
by
  existsi 4
  existsi 1
  existsi 2
  split; try { refl }
  sorry

end find_population_growth_rates_l180_180165


namespace sum_exp_neg_g_converges_l180_180457

def f (n : ℕ) : ℕ := 
  if n = 0 then 0 else Nat.trailingZeroes n

def g (n : ℕ) : ℕ := 
  ∑ i in Finset.range (n + 1), f i

theorem sum_exp_neg_g_converges : 
  ∃ L : ℝ, L = ∑' n, Real.exp (- (g n)) := 
sorry

end sum_exp_neg_g_converges_l180_180457


namespace distance_equality_proof_l180_180889

variables {Γ : Type} [circle Γ] (I : point Γ)
variables {A B C D : Γ}
variables {O : circle (triangle A I C)}
variables {X Y Z T : O}

-- Conditions
axiom tangent_AB : tangent_to_circle A B Γ
axiom tangent_BC : tangent_to_circle B C Γ
axiom tangent_CD : tangent_to_circle C D Γ
axiom tangent_DA : tangent_to_circle D A Γ

axiom extension_BA_X : extends_beyond_to BA A X O
axiom extension_BC_Z : extends_beyond_to BC C Z O
axiom extension_AD_Y : extends_beyond_to AD D Y O
axiom extension_CD_T : extends_beyond_to CD D T O

-- Proof goal
theorem distance_equality_proof :
  distance A D + distance D T + distance T X + distance X A =
  distance C D + distance D Y + distance Y Z + distance Z C := sorry

end distance_equality_proof_l180_180889


namespace ratio_of_triangle_areas_l180_180409

theorem ratio_of_triangle_areas (kx ky k : ℝ)
(n m : ℕ) (h1 : n > 0) (h2 : m > 0) :
  let A := (1 / 2) * (ky / m) * (kx / 2)
  let B := (1 / 2) * (kx / n) * (ky / 2)
  (A / B) = (n / m) :=
by
  sorry

end ratio_of_triangle_areas_l180_180409


namespace range_g_l180_180302

def g (x : ℝ) : ℝ := 1 / (x^2 + 1)

theorem range_g : Set.Ioo 0 1 = (set_of (fun y => 0 < y ∧ y ≤ 1)) :=
by
  sorry

end range_g_l180_180302


namespace inverse_of_r_l180_180919

def p (x : ℝ) : ℝ := 4 * x - 7
def q (x : ℝ) : ℝ := 3 * x + 2
def r (x : ℝ) : ℝ := p (q x)

noncomputable def r_inv (x : ℝ) : ℝ := (x - 1) / 12

theorem inverse_of_r :
  function.inverse r = r_inv :=
sorry

end inverse_of_r_l180_180919


namespace students_know_mothers_birthday_l180_180498

-- Defining the given conditions
def total_students : ℕ := 40
def A : ℕ := 10
def B : ℕ := 12
def C : ℕ := 22
def D : ℕ := 26

-- Statement to prove
theorem students_know_mothers_birthday : (B + C) = 22 :=
by
  sorry

end students_know_mothers_birthday_l180_180498


namespace valid_pairs_l180_180308

theorem valid_pairs (a b : ℕ) (h : a ≤ b)
  (cond : ∀ x : ℕ, Gcd.gcd x a * Gcd.gcd x b = Gcd.gcd x 20 * Gcd.gcd x 22) :
  (a = 2 ∧ b = 220) ∨ (a = 4 ∧ b = 110) ∨ (a = 10 ∧ b = 44) ∨ (a = 20 ∧ b = 22) :=
sorry

end valid_pairs_l180_180308


namespace boys_basketball_clay_maple_l180_180882

structure Attendance :=
  (total_students : ℕ)
  (boys : ℕ)
  (girls : ℕ)
  (jonas_students : ℕ)
  (jonas_boys : ℕ)
  (clay_students : ℕ)
  (maple_students : ℕ)
  (girls_swimming : ℕ)
  (clay_boys_swimming : ℕ)

def conditions : Attendance :=
  { total_students := 120,
    boys := 70,
    girls := 50,
    jonas_students := 50,
    jonas_boys := 28,
    clay_students := 40,
    maple_students := 30,
    girls_swimming := 16,
    clay_boys_swimming := 10 }

theorem boys_basketball_clay_maple (a : Attendance) : 
  a.boys - a.jonas_boys - a.clay_boys_swimming = 30 :=
by simp [conditions]; sorry

end boys_basketball_clay_maple_l180_180882


namespace square_area_perimeter_l180_180266

theorem square_area_perimeter (d : ℝ) (h : d = 12 * real.sqrt 2) :
  (∃ s : ℝ, d = s * real.sqrt 2 ∧ s^2 = 144 ∧ (4 * s) = 48) :=
by
  sorry

end square_area_perimeter_l180_180266


namespace compute_c_plus_d_l180_180316

theorem compute_c_plus_d (c d : ℕ) (hcd : 0 < c ∧ 0 < d ∧ (d - c) = 435) 
  (h_prod : (∏ i in (finset.range 435).map (λ n, c + 2 * n), log (c + 2 * n) / log (c + 2 * (n - 1))) = 3) : 
  c + d = 738 :=
sorry

end compute_c_plus_d_l180_180316


namespace bruce_bank_ratio_l180_180282

noncomputable def bruce_aunt : ℝ := 75
noncomputable def bruce_grandfather : ℝ := 150
noncomputable def bruce_bank : ℝ := 45
noncomputable def bruce_total : ℝ := bruce_aunt + bruce_grandfather
noncomputable def bruce_ratio : ℝ := bruce_bank / bruce_total

theorem bruce_bank_ratio :
  bruce_ratio = 1 / 5 :=
by
  -- proof goes here
  sorry

end bruce_bank_ratio_l180_180282


namespace smallest_number_of_eggs_l180_180211

theorem smallest_number_of_eggs (c : ℕ) :
  (∃ c : ℕ, 10 * c - 3 > 130) → 137 = (10 * 14 - 3) :=
by
  intro h
  existsi 14
  trivial

end smallest_number_of_eggs_l180_180211


namespace last_four_digits_of_m_smallest_4_9_l180_180089

theorem last_four_digits_of_m_smallest_4_9 
  (m : ℕ) 
  (h1 : ∃ n : ℕ, n > 0 ∧ ∀ k > 0, k | 4 ∧ k | 9 → m <= k)
  (h2 : ∃ l : ℕ, m % 4 = 0) 
  (h3 : ∃ p : ℕ, m % 9 = 0) 
  (h4 : ∀ d : list ℕ, d.all (λ x, x = 4 ∨ x = 9) → m = d.foldr (λ x acc, x + 10 * acc) 0 ∧ (d.count 4 ≥ 2 ∧ d.count 9 ≥ 2))
  : (m % 10000) = 4944 :=
by
  sorry

end last_four_digits_of_m_smallest_4_9_l180_180089


namespace new_person_weight_l180_180537

variable {W : ℝ} -- Total weight of the original group of 15 people
variable {N : ℝ} -- Weight of the new person

theorem new_person_weight
  (avg_increase : (W - 90 + N) / 15 = (W - 90) / 14 + 3.7)
  : N = 55.5 :=
sorry

end new_person_weight_l180_180537


namespace max_value_of_f_smallest_positive_period_of_f_values_of_x_satisfying_f_ge_1_l180_180844

/-- Define the given function f(x) -/
noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sqrt 2 * Real.cos x * Real.sin (x + Real.pi / 4) - 1

/-- The maximum value of the function f(x) is sqrt(2) -/
theorem max_value_of_f : ∃ x, f x = Real.sqrt 2 := 
sorry

/-- The smallest positive period of the function f(x) -/
theorem smallest_positive_period_of_f :
  ∃ p, p > 0 ∧ ∀ x, f (x + p) = f x ∧ p = Real.pi :=
sorry

/-- The set of values x that satisfy f(x) ≥ 1 -/
theorem values_of_x_satisfying_f_ge_1 :
  ∀ x, f x ≥ 1 ↔ ∃ k : ℤ, k * Real.pi ≤ x ∧ x ≤ k * Real.pi + Real.pi / 4 :=
sorry

end max_value_of_f_smallest_positive_period_of_f_values_of_x_satisfying_f_ge_1_l180_180844


namespace problem_l180_180861

-- Define the general term in the expansion
def general_term (n r : ℕ) (x : ℝ) : ℝ :=
  (Nat.choose n r) * x^(n - (3*r / 2))

-- Binomial theorem related facts
lemma binomial_theorem_expansion (n : ℕ) (a b : ℝ) :
  (a + b)^n = ∑ r in Finset.range (n + 1), (Nat.choose n r) * a^(n - r) * b^r := sorry

-- Sum of binomial coefficients.
lemma sum_binomial_coefficients (n : ℕ) : ∑ r in Finset.range (n + 1), (Nat.choose n r) = 2^n := sorry

-- Problem: Prove the correct options in the binomial expansion given the conditions.
theorem problem (n : ℕ) :
  let x : ℝ := n in
  let expansion := (x + 1/√x)^9 in
  -- Option B: The sum of the binomial coefficients of the odd terms is 256.
  (∑ r in Finset.range 10, if r % 2 = 1 then Nat.choose 9 r else 0) = 256 ∧
  -- Option C: The constant term is 84.
  (let r := 6 in Nat.choose 9 r) = 84 := sorry

end problem_l180_180861


namespace distinct_painted_cubes_l180_180239

theorem distinct_painted_cubes : 
  let painted_cube_valid (cube : (Fin 6 → ℕ)) := 
    cube 0 = 1 ∧ -- One face is yellow (represented by 1)
    Multiset.card (Multiset.filter (λ x => x = 2) (Multiset.of_fn cube)) = 2 ∧ -- Two faces are purple (represented by 2)
    Multiset.card (Multiset.filter (λ x => x = 3) (Multiset.of_fn cube)) = 3 -- Three faces are orange (represented by 3)
  in 
  ∃ (painted_cubes : Finset (Fin 6 → ℕ)), 
    (∀ x ∈ painted_cubes, painted_cube_valid x .T -- Valid painted cubes
    (painted_cubes.card = 6) -- Number of distinct valid painted cubes is 6
:= 
sorry

end distinct_painted_cubes_l180_180239


namespace find_radius_of_sphere_l180_180743

-- Definitions for the problem conditions
def base_edge_length : ℝ := 1
def side_edge_length : ℝ := 2

-- Statement of the proof problem
theorem find_radius_of_sphere 
  (b : ℝ := base_edge_length) 
  (s : ℝ := side_edge_length) 
  (circumscribed_sphere_radius : ℝ := real.sqrt (b^2 + b^2 + s^2) / 2)
  : circumscribed_sphere_radius = real.sqrt (6) / 2 := 
sorry

end find_radius_of_sphere_l180_180743


namespace sum_of_roots_zero_l180_180141

noncomputable theory

variables {Q : ℝ → ℝ}

-- Define Q as a cubic polynomial
def Q_cubic (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- Include the condition from the problem
def condition_ineq (a b c d : ℝ) : Prop :=
  ∀ x : ℝ, Q_cubic a b c d (x^4 + 2 * x) ≥ Q_cubic a b c d (x^3 + 2)

-- The main theorem stating the proof problem
theorem sum_of_roots_zero (a b c d : ℝ) (h : condition_ineq a b c d) :
  ∑ root : ℝ in (Multiset.map (λ x, (x - (x * x + 1))) [3, 3, 1]), true := 
sorry

end sum_of_roots_zero_l180_180141


namespace find_b_l180_180506

theorem find_b (a b c : ℝ) (h1 : a + b + c = 150) (h2 : a + 10 = c^2) (h3 : b - 5 = c^2) : 
  b = (1322 - 2 * Real.sqrt 1241) / 16 := 
by 
  sorry

end find_b_l180_180506


namespace arithmetic_progression_less_than_100_nat_l180_180333

theorem arithmetic_progression_less_than_100_nat {a1 d : ℕ} (h_nonzero : d ≠ 0) (h_no_digit_9 : ∀ n : ℕ, ¬ (a1 + (n - 1) * d).digit 9) :
  ∃ N < 100, ∀ n : ℕ, n ≤ N → ¬ (a1 + (n - 1) * d).digit 9 := 
sorry

end arithmetic_progression_less_than_100_nat_l180_180333


namespace ellipse_proof_l180_180745

def focal_distance := 2 * Real.sqrt (2)
def major_axis_length := 6
def point_H := (0, 1)
def slope_sum := 1

noncomputable def ellipse_eq := ∀ (x y : ℝ), (x^2 / 9 + y^2 = 1)

noncomputable def real_number_k := ∃ (k : ℝ), 
  (k = 2) ∧ ∀ (x y : ℝ), 
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (y₁ = k * x₁ + 3) ∧ (y₂ = k * x₂ + 3) ∧ 
    (x₁ + x₂ ≠ 0) ∧ 
    ((k * y₁ + k * y₂ + (2 * k - 3 / (x₁ * x₂)) = 1))

theorem ellipse_proof : ellipse_eq ∧ real_number_k := by
  sorry

end ellipse_proof_l180_180745


namespace minimum_gennadies_l180_180616

theorem minimum_gennadies (A B V G : ℕ) (hA : A = 45) (hB : B = 122) (hV : V = 27) (hGap : G + A + V >= B - 1) :
  G >= 49 :=
by 
  have := by linarith [hGap, hA, hB, hV]
  exact this

end minimum_gennadies_l180_180616


namespace sum_of_first_6_terms_l180_180739

variable {α : Type*} [LinearOrderedField α] 

-- Definitions based on conditions
def is_arithmetic_sequence (a : ℕ → α) : Prop :=
  ∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

def Sn (a : ℕ → α) (n : ℕ) : α :=
  (finset.range n).sum a

-- Given conditions
variables (a : ℕ → α)
variable (S3_eq_12 : Sn a 3 = 12)
variable (a2_a4_eq_4 : a 1 + a 3 = 4)

-- Prove that S6 = 6
theorem sum_of_first_6_terms (a : ℕ → α) (ha : is_arithmetic_sequence a) :
  Sn a 6 = 6 :=
by
  sorry

end sum_of_first_6_terms_l180_180739


namespace range_of_a_no_solution_l180_180845

theorem range_of_a_no_solution (a : ℝ) :
  ¬ ∃ x : ℝ, |x + 2| + |3 - x| < 2 * a + 1 → a ≤ 2 :=
begin
  intros h,
  have H : 5 ≤ 2 * a + 1,
  { specialize h (3/2),
    linarith [abs_nonneg (3/2 + 2), abs_nonneg (3 - 3/2)] },
  linarith,
end

end range_of_a_no_solution_l180_180845


namespace countMultiples23Or4ButNot5Below2011_l180_180379

def isMultipleOf (d n : ℕ) : Prop := ∃ k : ℕ, n = d * k

def countMultiplesBelow (d n : ℕ) : ℕ :=
(n / d)

def countMultiples23Or4ButNot5Below (n : ℕ) : ℕ :=
let countDiv3 := countMultiplesBelow 3 n;
let countDiv4 := countMultiplesBelow 4 n;
let countDiv3And4 := countMultiplesBelow 12 n;
let count3Or4 := countDiv3 + countDiv4 - countDiv3And4 in
let countDiv5 := countMultiplesBelow 5 n;
let countDiv15 := countMultiplesBelow 15 n;
let countDiv20 := countMultiplesBelow 20 n;
let countDiv60 := countMultiplesBelow 60 n;
count3Or4 - (countDiv15 + countDiv20 - countDiv60)

theorem countMultiples23Or4ButNot5Below2011 : countMultiples23Or4ButNot5Below 2010 = 804 :=
by sorry

end countMultiples23Or4ButNot5Below2011_l180_180379


namespace coupon_discount_percentage_correct_l180_180496

-- Define the original price, increase, and final sale price conditions
def original_price : ℝ := 200
def increase_percentage : ℝ := 0.30
def increased_price : ℝ := original_price * (1 + increase_percentage)
def sale_price : ℝ := 182
def discount_amount : ℝ := increased_price - sale_price
def discount_percentage : ℝ := (discount_amount / increased_price) * 100

-- The theorem statement to prove
theorem coupon_discount_percentage_correct :
  discount_percentage = 30 := 
by 
  sorry

end coupon_discount_percentage_correct_l180_180496


namespace polynomial_value_l180_180354

theorem polynomial_value (x y : ℝ) (h : x - 2 * y + 3 = 8) : x - 2 * y = 5 :=
by
  sorry

end polynomial_value_l180_180354


namespace advanced_ticket_cost_unique_l180_180136

noncomputable def cost_of_advanced_ticket 
    (total_tickets : ℕ) 
    (cost_door_ticket : ℝ) 
    (total_revenue : ℝ) 
    (door_tickets_sold : ℕ) 
    (cost_advanced_ticket : ℝ) : Prop := 
  total_tickets = 800 ∧ 
  cost_door_ticket = 22 ∧ 
  total_revenue = 16640 ∧ 
  door_tickets_sold = 672  →
  cost_advanced_ticket = 14.5

theorem advanced_ticket_cost_unique :
  ∀ total_tickets cost_door_ticket total_revenue door_tickets_sold cost_advanced_ticket, 
  cost_of_advanced_ticket total_tickets cost_door_ticket total_revenue door_tickets_sold cost_advanced_ticket := 
  begin
    intros,
    sorry
  end

end advanced_ticket_cost_unique_l180_180136


namespace sec_tan_identity_l180_180823

theorem sec_tan_identity (x : ℝ) (h : Real.sec x + Real.tan x = 7 / 3) : Real.sec x - Real.tan x = 3 / 7 := 
by
  sorry

end sec_tan_identity_l180_180823


namespace triangle_area_of_parabola_hyperbola_l180_180139

-- Definitions for parabola and hyperbola
def parabola_directrix (a : ℕ) (x y : ℝ) : Prop := x^2 = 16 * y
def hyperbola_asymptotes (a b : ℕ) (x y : ℝ) : Prop := x^2 / (a^2) - y^2 / (b^2) = 1

-- Theorem stating the area of the triangle formed by the intersections of the asymptotes with the directrix
theorem triangle_area_of_parabola_hyperbola (a b : ℕ) (h : a = 1) (h' : b = 1) : 
  ∃ (area : ℝ), area = 16 :=
sorry

end triangle_area_of_parabola_hyperbola_l180_180139


namespace min_gennadys_l180_180639

theorem min_gennadys (alexanders borises vasilies : ℕ) (x : ℕ) 
    (h1 : alexanders = 45)
    (h2 : borises = 122)
    (h3 : vasilies = 27)
    (h4 : x = 49)
    (h5 : borises - 1 = alexanders + vasilies + x) :
  x = 49 := 
begin
  sorry,
end

end min_gennadys_l180_180639


namespace sum_mod_7_l180_180531

theorem sum_mod_7 :
  (∑ i in Finset.range 206, i) % 7 = 0 :=
by
  sorry

end sum_mod_7_l180_180531


namespace equal_sharing_of_chicken_wings_l180_180222

theorem equal_sharing_of_chicken_wings 
  (initial_wings : ℕ) (additional_wings : ℕ) (number_of_friends : ℕ)
  (total_wings : ℕ) (wings_per_person : ℕ)
  (h_initial : initial_wings = 8)
  (h_additional : additional_wings = 10)
  (h_number : number_of_friends = 3)
  (h_total : total_wings = initial_wings + additional_wings)
  (h_division : wings_per_person = total_wings / number_of_friends) :
  wings_per_person = 6 := 
  by
  sorry

end equal_sharing_of_chicken_wings_l180_180222


namespace oranges_weight_l180_180214

theorem oranges_weight (A O : ℕ) (h1 : O = 5 * A) (h2 : A + O = 12) : O = 10 := 
by 
  sorry

end oranges_weight_l180_180214


namespace monotonicity_of_f_range_of_a_harmonic_greater_than_log_l180_180367

noncomputable def f (x a : ℝ) : ℝ := Real.exp x + a * x + a

theorem monotonicity_of_f {a : ℝ} :
  (∀ x : ℝ, deriv (f x a) > 0) ↔ a ≥ 0 ∧
  (∃ y : ℝ, y = Real.log (-a) ∧ ∀ x < y, deriv (f x a) < 0 ∧ ∀ x > y, deriv (f x a) > 0) := 
  sorry

theorem range_of_a : 
  (∀ a : ℝ, a < 0 → a ∈ Set.Ico (-1 : ℝ) 0) := 
  sorry

theorem harmonic_greater_than_log {n : ℕ} (hn : 0 < n) :
  ∑ i in Finset.range n.succ, 1 / (i + 1) > Real.log (n + 1) :=
  sorry

end monotonicity_of_f_range_of_a_harmonic_greater_than_log_l180_180367


namespace sin_alpha_eq_one_half_l180_180722

theorem sin_alpha_eq_one_half {α : ℝ} (hα1 : 0 < α) (hα2 : α < π) (h : sin α = cos (2 * α)) :
  sin α = 1 / 2 :=
sorry

end sin_alpha_eq_one_half_l180_180722


namespace sec_sub_tan_l180_180799

theorem sec_sub_tan (x : ℝ) (h : sec x + tan x = 7 / 3) : sec x - tan x = 3 / 7 := by
  sorry

end sec_sub_tan_l180_180799


namespace length_of_AC_l180_180868

variable (A B C : Type)
variable [Inhabited A] [Inhabited B] [Inhabited C]

theorem length_of_AC {AB AC : Real}
  (ABC_right : ∀ (A B C : Point), right_triangle A B C) -- Condition 1: Right triangle with right angle at A
  (tan_C : ∀ (A B C : Point), tan C = 4 / 3)            -- Condition 2: tan C = 4/3
  (AB_length : AB = 3) :                                 -- Condition 3: AB = 3
  AC = 4 :=                                              
sorry

end length_of_AC_l180_180868


namespace min_gennadys_l180_180637

theorem min_gennadys (alexanders borises vasilies : ℕ) (x : ℕ) 
    (h1 : alexanders = 45)
    (h2 : borises = 122)
    (h3 : vasilies = 27)
    (h4 : x = 49)
    (h5 : borises - 1 = alexanders + vasilies + x) :
  x = 49 := 
begin
  sorry,
end

end min_gennadys_l180_180637


namespace log_domain_eq_l180_180963

noncomputable def quadratic_expr (x : ℝ) : ℝ := x^2 - 2 * x - 3

def log_domain (x : ℝ) : Prop := quadratic_expr x > 0

theorem log_domain_eq :
  {x : ℝ | log_domain x} = 
  {x : ℝ | x < -1} ∪ {x : ℝ | x > 3} :=
by {
  sorry
}

end log_domain_eq_l180_180963


namespace parking_lot_length_l180_180063

theorem parking_lot_length (W : ℝ) (U : ℝ) (A_car : ℝ) (N_cars : ℕ) (H_w : W = 400) (H_u : U = 0.80) (H_Acar : A_car = 10) (H_Ncars : N_cars = 16000) :
  (U * (W * L) = N_cars * A_car) → (L = 500) :=
by
  sorry

end parking_lot_length_l180_180063


namespace slope_y_intercept_sum_l180_180418

def point := ℝ × ℝ

def midpoint (A B : point) : point :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def slope (A B : point) : ℝ :=
  (B.2 - A.2) / (B.1 - A.1)

def y_intercept (m : ℝ) (P : point) : ℝ :=
  P.2 - m * P.1

def line_equation (m b : ℝ) (x : ℝ) : ℝ :=
  m * x + b

theorem slope_y_intercept_sum :
  let P : point := (0, 10) in
  let Q : point := (0, 0) in
  let R : point := (10, 0) in
  let G : point := midpoint P Q in
  let m : ℝ := slope R G in
  let b : ℝ := y_intercept m G in
  m + b = 9 / 2 :=
begin
  sorry,
end

end slope_y_intercept_sum_l180_180418


namespace cost_per_box_of_cookies_l180_180698

-- Given conditions
def initial_money : ℝ := 20
def mother_gift : ℝ := 2 * initial_money
def total_money : ℝ := initial_money + mother_gift
def cupcake_price : ℝ := 1.50
def num_cupcakes : ℝ := 10
def cost_cupcakes : ℝ := num_cupcakes * cupcake_price
def money_after_cupcakes : ℝ := total_money - cost_cupcakes
def remaining_money : ℝ := 30
def num_boxes_cookies : ℝ := 5
def money_spent_on_cookies : ℝ := money_after_cupcakes - remaining_money

-- Theorem: Calculate the cost per box of cookies
theorem cost_per_box_of_cookies : (money_spent_on_cookies / num_boxes_cookies) = 3 :=
by
  sorry

end cost_per_box_of_cookies_l180_180698


namespace pure_ghee_added_l180_180859

theorem pure_ghee_added
  (Q : ℕ) (hQ : Q = 30)
  (P : ℕ)
  (original_pure_ghee : ℕ := (Q / 2))
  (original_vanaspati : ℕ := (Q / 2))
  (new_total_ghee : ℕ := Q + P)
  (new_vanaspati_fraction : ℝ := 0.3) :
  original_vanaspati = (new_vanaspati_fraction * ↑new_total_ghee : ℝ) → P = 20 := by
  sorry

end pure_ghee_added_l180_180859


namespace exponent_of_5_in_30_factorial_l180_180006

theorem exponent_of_5_in_30_factorial : 
  (nat.factors 30!).count 5 = 7 :=
sorry

end exponent_of_5_in_30_factorial_l180_180006


namespace friend_speed_l180_180874

theorem friend_speed (total_distance jenna_distance friend_distance jenna_speed total_time break_time friend_time : ℝ)
  (h1 : jenna_distance = 200)
  (h2 : friend_distance = 100)
  (h3 : total_distance = jenna_distance + friend_distance)
  (h4 : jenna_speed = 50)
  (h5 : total_time = 10)
  (h6 : break_time = 2 * 30 / 60) -- 2 breaks * 30 minutes per break converted to hours
  (h7 : friend_time = total_time - break_time - jenna_distance / jenna_speed)
  (h8 : (friend_distance / friend_time) = 20) : (friend_distance / friend_time) = 20 := 
by
  rw [←h8]
  sorry

end friend_speed_l180_180874


namespace cylindrical_coordinates_of_point_l180_180297

theorem cylindrical_coordinates_of_point :
  ∀ (x y z : ℝ), x = 3 → y = -3 * Real.sqrt 3 → z = 2 →
  ∃ r θ, r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ r = 6 ∧ θ = 5 * Real.pi / 3 ∧ z = 2 :=
by
  intros x y z hx hy hz
  use 6, (5 * Real.pi / 3)
  split
  { exact zero_lt_six },
  split
  { norm_num },
  split
  { norm_num },
  split
  { exact hx.symm ▸ hy.symm ▸ rfl },
  split
  { exact hz },
  sorry

end cylindrical_coordinates_of_point_l180_180297


namespace peaches_picked_l180_180124

theorem peaches_picked (initial final : ℕ) (h_initial : initial = 13) (h_final : final = 55) : final - initial = 42 :=
by {
  rw [h_final, h_initial],
  norm_num,
  sorry,
}

end peaches_picked_l180_180124


namespace inequality_for_positive_integer_l180_180946

theorem inequality_for_positive_integer (n : ℕ) (h : n > 0) :
  n^n ≤ (n!)^2 ∧ (n!)^2 ≤ ((n + 1) * (n + 2) / 6)^n := by
  sorry

end inequality_for_positive_integer_l180_180946


namespace ratio_of_larger_to_smaller_l180_180176

theorem ratio_of_larger_to_smaller (x y : ℝ) (h1 : x > y) (h2 : x + y = 7 * (x - y)) : x / y = 2 :=
sorry

end ratio_of_larger_to_smaller_l180_180176


namespace john_less_than_anna_l180_180067

theorem john_less_than_anna (J A L T : ℕ) (h1 : A = 50) (h2: L = 3) (h3: T = 82) (h4: T + L = A + J) : A - J = 15 :=
by
  sorry

end john_less_than_anna_l180_180067


namespace sec_minus_tan_l180_180810

-- Define the problem in Lean 4
theorem sec_minus_tan (x : ℝ) (h : Real.sec x + Real.tan x = 7 / 3) : Real.sec x - Real.tan x = 3 / 7 := by
  -- One could also include here the necessary mathematical facts and connections.
  sorry -- Proof to be provided

end sec_minus_tan_l180_180810


namespace prove_greatest_value_l180_180683

noncomputable def Q (x : ℝ) : ℝ := x^4 + 2 * x^3 - x^2 - 4 * x + 4

def Q1 := Q 1 -- Evaluated as 2
def product_of_zeros := 4 -- Product of the zeros of Q
def product_of_non_real_zeros := sorry -- Assumed to be less than 2
def sum_of_coefficients := 2 -- Sum of the coefficients of Q
def sum_of_real_zeros := 0 -- Sum of the real zeros

theorem prove_greatest_value :
  max Q1 (max product_of_zeros (max product_of_non_real_zeros (max sum_of_coefficients sum_of_real_zeros))) = 4 :=
sorry

end prove_greatest_value_l180_180683


namespace sec_tan_eq_l180_180832

theorem sec_tan_eq (x : ℝ) (h : Real.cos x ≠ 0) : 
  Real.sec x + Real.tan x = 7 / 3 → Real.sec x - Real.tan x = 3 / 7 :=
by
  intro h1
  sorry

end sec_tan_eq_l180_180832


namespace sec_tan_identity_l180_180825

theorem sec_tan_identity (x : ℝ) (h : Real.sec x + Real.tan x = 7 / 3) : Real.sec x - Real.tan x = 3 / 7 := 
by
  sorry

end sec_tan_identity_l180_180825


namespace part1_part2_l180_180765

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2 * x - 1| + |2 * x + 2|

-- The minimum value M of f(x)
def M : ℝ := 3

-- Define the conditions for a and b in part (2)
def satisfies_condition (a b : ℝ) : Prop := a^2 + 2 * b^2 = M

-- The inequalities to prove
theorem part1 (x : ℝ) (h₁ : f x < M + |2 * x + 2|) : -1 < x ∧ x < 2 :=
sorry

theorem part2 (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : satisfies_condition a b) : 2 * a + b ≤ (3 * real.sqrt 6) / 2 :=
sorry

end part1_part2_l180_180765


namespace sec_sub_tan_l180_180798

theorem sec_sub_tan (x : ℝ) (h : sec x + tan x = 7 / 3) : sec x - tan x = 3 / 7 := by
  sorry

end sec_sub_tan_l180_180798


namespace imaginary_part_of_z_l180_180717

-- Define the given complex number z
def z : ℂ := - (complex.i / (1 + complex.i))

-- Define the solution condition: the imaginary part of z
theorem imaginary_part_of_z : complex.im z = -1 / 2 :=
by
  -- Here we would provide a proof, but it's not required in this task.
  sorry

end imaginary_part_of_z_l180_180717


namespace fair_betting_scheme_fair_game_l180_180192

noncomputable def fair_game_stakes (L L_k : ℕ → ℚ) : Prop :=
  ∀ k: ℕ, 1 < k → L_k (k+1) = (35/36) * L_k k

theorem fair_betting_scheme_fair_game :
  ∃ L_k : ℕ → ℚ, fair_game_stakes (λ k, L_k k) (λ k, L_k k) :=
begin
  let L_k : ℕ → ℚ := λ k, (35 / 36) ^ k,
  use L_k,
  unfold fair_game_stakes,
  intros k hk,
  rw [mul_assoc, ←mul_pow],
  ring,
end

end fair_betting_scheme_fair_game_l180_180192


namespace exists_dividing_line_l180_180443

/-- A regular polygon with colored vertices and certain properties -/
structure RegularPolygon (α : Type) :=
  (vertices : set α) -- Set of vertices
  (coloring : α → Fin 3) -- Coloring function (3 colors: red, white, blue)
  (is_regular : is_regular_polygon vertices) -- Regular polygon property
  (is_patriotic : is_patriotic_set vertices coloring) -- Patriotic property: equal number of each color
  (dazzling_edges_even : is_even (dazzling_edges_count vertices coloring)) -- even number of dazzling edges

/-- A definition of dazzling edge -/
def is_dazzling_edge {α : Type} (v1 v2 : α) (coloring : α → Fin 3) :=
  coloring v1 ≠ coloring v2

/-- The definition of patriotic set -/
def is_patriotic_set (α : Type) (s : set α) (coloring : α → Fin 3) :=
  ∀ c : Fin 3, (s.filter (λ v, coloring v = c)).size = s.card / 3

/-- The alternating number of dazzling edges -/
def dazzling_edges_count {α : Type} (vertices : set α) (coloring : α → Fin 3) : ℕ :=
  (vertices.to_list.zip vertices.to_list.rotate 1).count (λ ⟨v1, v2⟩, is_dazzling_edge v1 v2 coloring)

/-- The theorem statement -/
theorem exists_dividing_line (α : Type) [fintype α] [decidable_eq α]
  (P : RegularPolygon α) :
  ∃ line : set α → Prop, 
    (∀ v ∈ P.vertices, ¬ line v) ∧ 
    ∃ (S1 S2 : set α), 
    S1 ≠ ∅ ∧ S2 ≠ ∅ ∧ is_patriotic_set α S1 P.coloring ∧ is_patriotic_set α S2 P.coloring ∧ 
    (∀ v, v ∈ S1 → ¬ v ∈ S2) := 
sorry

end exists_dividing_line_l180_180443


namespace quadrant_z_l180_180748

noncomputable def z : ℂ := 2 + 2 * I

theorem quadrant_z (h : (z - I) * (2 - I) = 5) : 
  z.re > 0 ∧ z.im > 0 := sorry

end quadrant_z_l180_180748


namespace train_length_approx_140_l180_180585

noncomputable def km_per_hr_to_m_per_s (speed_km_hr : ℝ) : ℝ :=
  speed_km_hr * 1000 / 3600

def train_length (speed_km_hr : ℝ) (time_sec : ℝ) : ℝ :=
  let speed_m_s := km_per_hr_to_m_per_s speed_km_hr
  speed_m_s * time_sec

theorem train_length_approx_140 :
  train_length 56 9 ≈ 140 :=
by
  -- The proof can be filled in here
  sorry

end train_length_approx_140_l180_180585


namespace methane_required_for_chlorination_l180_180501

-- Definitions based on the conditions
def chlorine_molecule : Type := ℕ
def methane : Type := ℕ
def chloromethane : Type := ℕ
def hydrochloric_acid : Type := ℕ

-- Mechanism conditions
axiom initiation_step (Cl₂ hν : chlorine_molecule) : chlorine_molecule <-> chlorine_molecule
axiom propagation_step1 :
  methane → chlorine_molecule → (chloromethane × hydrochloric_acid)
axiom propagation_step2 :
  chloromethane → chlorine_molecule → chloromethane
axiom termination_step (Cl : chlorine_molecule) : chlorine_molecule

-- Proof statement
theorem methane_required_for_chlorination :
  ∀ (Cl₂_initial CH4_needed : ℕ),
  Cl₂_initial = 2 →
  let CH3Cl_production := 2 in
  let HCl_production := 2 in
  propagation_step1 CH4_needed Cl₂_initial = (CH3Cl_production, HCl_production) →
  CH4_needed = 2 :=
sorry

end methane_required_for_chlorination_l180_180501


namespace tomato_price_per_kilo_l180_180108

theorem tomato_price_per_kilo 
  (initial_money: ℝ) (money_left: ℝ)
  (potato_price_per_kilo: ℝ) (potato_kilos: ℝ)
  (cucumber_price_per_kilo: ℝ) (cucumber_kilos: ℝ)
  (banana_price_per_kilo: ℝ) (banana_kilos: ℝ)
  (tomato_kilos: ℝ)
  (spent_on_potatoes: initial_money - money_left = potato_price_per_kilo * potato_kilos)
  (spent_on_cucumbers: initial_money - money_left = cucumber_price_per_kilo * cucumber_kilos)
  (spent_on_bananas: initial_money - money_left = banana_price_per_kilo * banana_kilos)
  (total_spent: initial_money - money_left = 74)
  : (74 - (potato_price_per_kilo * potato_kilos + cucumber_price_per_kilo * cucumber_kilos + banana_price_per_kilo * banana_kilos)) / tomato_kilos = 3 := 
sorry

end tomato_price_per_kilo_l180_180108


namespace part1_part2_l180_180764

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2 * x - 1| + |2 * x + 2|

-- The minimum value M of f(x)
def M : ℝ := 3

-- Define the conditions for a and b in part (2)
def satisfies_condition (a b : ℝ) : Prop := a^2 + 2 * b^2 = M

-- The inequalities to prove
theorem part1 (x : ℝ) (h₁ : f x < M + |2 * x + 2|) : -1 < x ∧ x < 2 :=
sorry

theorem part2 (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : satisfies_condition a b) : 2 * a + b ≤ (3 * real.sqrt 6) / 2 :=
sorry

end part1_part2_l180_180764


namespace number_of_female_students_l180_180142

noncomputable def total_students : ℕ := 1600
noncomputable def sample_size : ℕ := 200
noncomputable def sampled_males : ℕ := 110
noncomputable def sampled_females := sample_size - sampled_males
noncomputable def total_males := (sampled_males * total_students) / sample_size
noncomputable def total_females := total_students - total_males

theorem number_of_female_students : total_females = 720 := 
sorry

end number_of_female_students_l180_180142


namespace find_circle_center_l180_180355

theorem find_circle_center {x y : ℝ} :
  (x^2 + y^2 - 2 * x = 0) → (∃ c : ℝ × ℝ, c = (1, 0)) :=
by 
  intros h
  use (1, 0)
  sorry

end find_circle_center_l180_180355


namespace horse_speed_l180_180138

-- Definition of the conditions
def area_of_square_field := 576 -- in square kilometers
def time_to_complete_run := 8 -- in hours
noncomputable def side_length := Real.sqrt area_of_square_field -- side length of the square field
def perimeter := 4 * side_length -- perimeter of the square field

-- Statement to prove
theorem horse_speed : perimeter / time_to_complete_run = 12 := by
  sorry

end horse_speed_l180_180138


namespace geometric_number_difference_is_403_l180_180284

def is_geometric_sequence (a b c : ℕ) : Prop :=
  b * b = a * c

def is_3_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def has_distinct_digits (n : ℕ) : Prop :=
  let digits := [n / 100, (n / 10) % 10, n % 10] in
  list.nodup digits

def starts_with_even_digit (n : ℕ) : Prop :=
  let first_digit := n / 100 in
  first_digit % 2 = 0

theorem geometric_number_difference_is_403 :
  ∃ a b c d e f : ℕ, 
    is_3_digit_number a ∧ is_3_digit_number b ∧
    has_distinct_digits a ∧ has_distinct_digits b ∧
    starts_with_even_digit a ∧ starts_with_even_digit b ∧
    is_geometric_sequence (a / 100) ((a / 10) % 10) (a % 10) ∧
    is_geometric_sequence (b / 100) ((b / 10) % 10) (b % 10) ∧
    (d = a / 100) ∧ (e = b / 100) ∧ (d % 2 = 0) ∧ (e % 2 = 0) ∧
    (max a b = f) ∧ (min a b = c) ∧ 
    f - c = 403 := 
sorry

end geometric_number_difference_is_403_l180_180284


namespace asymptotes_of_C2_l180_180341

noncomputable section

variables {a b : ℝ} (h₀ : a > b) (h₁ : b > 0)
-- Definition of ellipse C1
def ellipse_C1 (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

-- Definition of hyperbola C2
def hyperbola_C2 (x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

-- Definition of product of eccentricities
def product_of_eccentricities : Prop :=
(e_1 : ℝ) (e_2 : ℝ), e_1 = √((a^2 - b^2) / a^2) ∧ e_2 = √((a^2 + b^2) / a^2) →
e_1 * e_2 = √3 / 2

-- Proof statement to determine the asymptotes of C2
theorem asymptotes_of_C2 (h : product_of_eccentricities) :
  ∀ x y : ℝ, (hyperbola_C2 x y ↔ x ± √2 * y = 0) :=
sorry

end asymptotes_of_C2_l180_180341


namespace rationalize_denominator_l180_180941

theorem rationalize_denominator (a b c d : ℝ) (h1 : a / (b * (c * d)) = a / (b * (c * d))) :
  (7 / (2 * Real.sqrt 50)) = (7 * Real.sqrt 2) / 20 :=
by
  rw [Real.sqrt_eq_rpow] at h1
  sorry

end rationalize_denominator_l180_180941


namespace value_of_4b_minus_a_l180_180966

theorem value_of_4b_minus_a (a b : ℕ) (h1 : a > b) (h2 : x^2 - 20*x + 96 = (x - a)*(x - b)) : 4*b - a = 20 :=
  sorry

end value_of_4b_minus_a_l180_180966


namespace balloon_ratio_l180_180684

/-- Janice has 6 water balloons. --/
def Janice_balloons : Nat := 6

/-- Randy has half as many water balloons as Janice. --/
def Randy_balloons : Nat := Janice_balloons / 2

/-- Cynthia has 12 water balloons. --/
def Cynthia_balloons : Nat := 12

/-- The ratio of Cynthia's water balloons to Randy's water balloons is 4:1. --/
theorem balloon_ratio : Cynthia_balloons / Randy_balloons = 4 := by
  sorry

end balloon_ratio_l180_180684


namespace find_function_l180_180301

theorem find_function (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y * f x) + f (x * y) = f x + f (2019 * y)) →
  (f = λ x, c ∨ f = λ x, 2019 - x ∨ f = (λ x, if x = 0 then c else 0)) := by
  sorry

end find_function_l180_180301


namespace intersection_not_always_polyhedron_l180_180842

-- Define a polyhedral angle as a structure with apex and faces
structure PolyhedralAngle where
  apex : ℝ × ℝ × ℝ
  faces : Set (Set (ℝ × ℝ × ℝ))

-- Define the property of a plane intersecting all faces of a polyhedral angle
def plane_intersects_faces (P : Set (ℝ × ℝ × ℝ)) (angle : PolyhedralAngle) :=
  ∀ face ∈ angle.faces, (P ∩ face).Nonempty

-- The theorem statement
theorem intersection_not_always_polyhedron (angle : PolyhedralAngle) (P : Set (ℝ × ℝ × ℝ)) :
  plane_intersects_faces P angle →
  ¬∃ polyhedron : Set (ℝ × ℝ × ℝ), (∀ face ∈ angle.faces, (P ∩ face).Nonempty ∧ is_polyhedron(polyhedron)) :=
by
  sorry

-- Auxiliary definitions (placeholders) and properties needed
def is_polyhedron (S : Set (ℝ × ℝ × ℝ)) : Prop := sorry

end intersection_not_always_polyhedron_l180_180842


namespace expected_tosses_l180_180512

theorem expected_tosses :
  let E : ℕ → ℝ :=
    λ n, if n = 1 then 1
         else if n = 2 then 2
         else 3 / 2 + 1 / 2 * E (n - 1) + 1 / 4 * E (n - 2)
  in E 10 = 4 + 135 / 256 :=
by
  sorry

end expected_tosses_l180_180512


namespace measure_of_angle_Q_l180_180428

theorem measure_of_angle_Q (P Q R X Y Z : Type) (triangle_PQR : Triangle P Q R)
    (right_angle_Q : angle Q = 90)
    (PQ_length : length (segment P Q) = 2)
    (QR_length : length (segment Q R) = 8)
    (X_on_PQ : X ∈ segment P Q)
    (line_through_X_parallel_to_QR_intersects_PR_at_Y : parallel (line_through X Q) (line_through P R))
    (line_through_Y_parallel_to_PQ_intersects_QR_at_Z : parallel (line_through Y P) (line_through Q R))
    (least_XZ : length (segment X Z) = 1.6) :
  angle Q = 90 :=
begin
  sorry
end

end measure_of_angle_Q_l180_180428


namespace expected_steps_proof_l180_180465

-- Definitions for the states and transitions in our problem
def initialState : ℕ := 0b1010
def finalState : ℕ := 0b0101

-- Expected steps from state A (initial) to state C (final)
noncomputable def expected_steps_A_to_C : ℕ := 6

-- State transition conditions and probabilities
axiom state_transition_A_to_B : true
axiom state_transition_B_to_A : ℕ := 1 / 4
axiom state_transition_B_to_C : ℕ := 1 / 4
axiom state_transition_B_to_B : ℕ := 1 / 2

-- Expected values for states B and A
noncomputable def E_Y : ℕ := 4
noncomputable def E_X : ℕ := 1 + E_Y

-- Proof goal: Expected number of steps to reach the final state from the initial state
theorem expected_steps_proof : expected_steps_A_to_C = 6 := by
  -- We state the problem given the conditions from the solution.
  sorry

end expected_steps_proof_l180_180465


namespace statement_C_l180_180343

variables (m n : Line) (α β : Plane)

-- Conditions
axiom different_lines : m ≠ n
axiom different_planes : α ≠ β
axiom n_in_beta : n ⊆ β

-- The correct statement to prove
theorem statement_C : (m ∥ n) ∧ (m ⊥ α) → (α ⊥ β) :=
by
  sorry

end statement_C_l180_180343


namespace scientific_notation_of_508_billion_yuan_l180_180162

-- Definition for a billion in the international system.
def billion : ℝ := 10^9

-- The amount of money given in the problem.
def amount_in_billion (n : ℝ) : ℝ := n * billion

-- The Lean theorem statement to prove.
theorem scientific_notation_of_508_billion_yuan :
  amount_in_billion 508 = 5.08 * 10^11 :=
by
  sorry

end scientific_notation_of_508_billion_yuan_l180_180162


namespace odd_number_representation_l180_180725

theorem odd_number_representation (n : ℤ) : 
  (∃ m : ℤ, 2 * m + 1 = 2 * n + 3) ∧ (¬ ∃ m : ℤ, 2 * m + 1 = 4 * n - 1) :=
by
  -- Proof steps would go here
  sorry

end odd_number_representation_l180_180725


namespace midpoints_and_equal_angles_implies_equal_diagonals_l180_180405

variables {A B C D M N K : Type} [EuclideanGeometry.cm A] [EuclideanGeometry.cm B] [EuclideanGeometry.cm C] [EuclideanGeometry.cm D] [EuclideanGeometry.cm M] [EuclideanGeometry.cm N] [EuclideanGeometry.cm K]

-- Suppose ABCD is a convex quadrilateral
-- M and N are the midpoints of AB and CD respectively
-- and a line through M and N forms equal angles with the diagonals AC and BD.
-- We need to show that AC = BD.

theorem midpoints_and_equal_angles_implies_equal_diagonals 
  (ABCD_convex : convex_quadrilateral A B C D)
  (M_mid_AB : midpoint M A B)
  (N_mid_CD : midpoint N C D)
  (MN_equal_angles : ∃ K, is_midpoint K A D ∧ 
    (angle_eq (line_through M N) (line_through A C) (line_through M N) (line_through B D))) :
  (length A C) = (length B D) := 
sorry

end midpoints_and_equal_angles_implies_equal_diagonals_l180_180405


namespace seq_geometric_sum_b_sum_c_l180_180768

noncomputable def a_sequence : ℕ → ℚ
| 0     := 1/4
| (n+1) := a_sequence n / ((-1)^(n+1) * a_sequence n - 2)

def b_sequence (a_sequence: ℕ → ℚ) (n: ℕ) := (1 / (a_sequence n)^2)

def c_sequence (a_sequence: ℕ → ℚ) (n: ℕ) :=
  a_sequence n * Float.sin ((2 * n + 1) * (Float.pi / 2))

def S_n (a_sequence: ℕ → ℚ) (n: ℕ) :=
  (3 * 4^n + 6 * 2^n + n - 9 : ℚ)

theorem seq_geometric (n : ℕ) :
  ∃ (r : ℚ) (g0 : ℚ), (∀ n: ℕ, (((1 / a_sequence (n+1)) + (-1)^(n+1)) = ((1 / a_sequence n) + (-1)^n) * r) ∧ 
  (((1 / a_sequence 0) + (-1)^0) = g0)) :=
sorry

theorem sum_b (n : ℕ) : 
S_n a_sequence n = (3 * 4^n + 6 * 2^n + n - 9 : ℚ) :=
sorry

theorem sum_c (n : ℕ) :
∑ i in range n,  (c_sequence a_sequence i) < (4/7 : ℚ) :=
sorry

end seq_geometric_sum_b_sum_c_l180_180768


namespace decreasing_function_range_of_a_l180_180358

noncomputable def f (a x : ℝ) : ℝ :=
  if x >= 3 then 2 * a * x + 4 else (a * x + 2) / (x - 2)

theorem decreasing_function_range_of_a :
  (∀ x y, 2 < x < y → (f a x > f a y)) ↔ (-1 < a ∧ a ≤ -2 / 3) :=
by
  intro a
  split
  · -- Suppose the function is decreasing
    assume h
    have ha1 : a < 0 := by sorry
    have ha2 : a > -1 := by sorry
    have ha3 : 6 * a + 4 ≤ 3 * a + 2 := by sorry
    exact ⟨ha2, (by linarith)⟩
  · -- Suppose the range of a
    assume h_a
    intro x y
    assume hx
    cases h_a with ha2 ha3
    cases hx with hx42 hy
    exact sorry

end decreasing_function_range_of_a_l180_180358


namespace smallest_cube_dividing_pq2r4_l180_180907

-- Definitions of conditions
variables {p q r : ℕ} [Fact (Nat.Prime p)] [Fact (Nat.Prime q)] [Fact (Nat.Prime r)]
variables (h_distinct : p ≠ q ∧ p ≠ r ∧ q ≠ r)

-- Definitions used in the proof
def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, m^3 = n

def smallest_perfect_cube_dividing (n k : ℕ) : Prop :=
  is_perfect_cube k ∧ n ∣ k ∧ ∀ k', is_perfect_cube k' ∧ n ∣ k' → k ≤ k'

-- The proof problem
theorem smallest_cube_dividing_pq2r4 (h_distinct : p ≠ q ∧ p ≠ r ∧ q ≠ r) :
  smallest_perfect_cube_dividing (p * q^2 * r^4) ((p * q * r^2)^3) :=
sorry

end smallest_cube_dividing_pq2r4_l180_180907


namespace min_gennadys_needed_l180_180651

variables (A B V G : ℕ)

theorem min_gennadys_needed
  (hA : A = 45)
  (hB : B = 122)
  (hV : V = 27)
  (hG : ∀ i, i < 121 → A + V < 121 → G ≥ 49) :
  G = 49 :=
sorry

end min_gennadys_needed_l180_180651


namespace find_marked_price_of_blouse_l180_180943

-- Define constants
def discount_rate := 0.18
def discounted_price := 147.60

-- Calculate the marked price
def marked_price (P : ℝ) : Prop :=
  (1 - discount_rate) * P = discounted_price

-- The Lean theorem statement
theorem find_marked_price_of_blouse (P : ℝ) : marked_price P → P = 180 :=
by
  intro h
  sorry

end find_marked_price_of_blouse_l180_180943


namespace fair_betting_scheme_fair_game_l180_180190

noncomputable def fair_game_stakes (L L_k : ℕ → ℚ) : Prop :=
  ∀ k: ℕ, 1 < k → L_k (k+1) = (35/36) * L_k k

theorem fair_betting_scheme_fair_game :
  ∃ L_k : ℕ → ℚ, fair_game_stakes (λ k, L_k k) (λ k, L_k k) :=
begin
  let L_k : ℕ → ℚ := λ k, (35 / 36) ^ k,
  use L_k,
  unfold fair_game_stakes,
  intros k hk,
  rw [mul_assoc, ←mul_pow],
  ring,
end

end fair_betting_scheme_fair_game_l180_180190


namespace robot_toy_movement_condition_l180_180581

theorem robot_toy_movement_condition :
  let toy := (15 : ℝ, 12 : ℝ)
  let line_robot := λ x : ℝ, -4 * x + 9
  let perp_slope (m : ℝ) := -1/m
  let line_perp := λ (x : ℝ), 1/4 * x + 33/4

  ∃ (c d : ℝ),
    line_robot c = d ∧
    line_perp c = d ∧
    (c, d) = (3/17, 135/17) ∧
    c + d = 138/17 := sorry

end robot_toy_movement_condition_l180_180581


namespace tan_phi_l180_180897
-- We use a broad import to ensure all required libraries are included.

-- Translate the given conditions into definitions.
variable (y : ℝ) (φ : ℝ)
variable (hc₁ : 0 < φ ∧ φ < π / 2) -- φ is an acute angle
variable (hc₂ : cos (φ / 2) = sqrt ((y + 2) / (3 * y))) -- given condition

-- The theorem to be proven.
theorem tan_phi (h : 0 < y) : 
    tan φ = sqrt (8 * (y - 2) * (y + 1) / (y + 4) ^ 2) := sorry

end tan_phi_l180_180897


namespace problem_l180_180740

noncomputable def f : ℝ → ℝ :=
  sorry

theorem problem :
  (∀ x : ℝ, f x = f (-x)) ∧ 
  (∀ x ≥ 0, f (x + 2) = f x) ∧ 
  (∀ x ∈ set.Ico (0 : ℝ) (2 : ℝ), f x = real.log (x + 1) / real.log 2) →
  f (-2008) + f 2009 = 1 :=
by
  sorry

end problem_l180_180740


namespace range_a_and_inequality_l180_180459

noncomputable def f (x a : ℝ) : ℝ := x^2 - a * Real.log (x + 2)
noncomputable def f' (x a : ℝ) : ℝ := 2 * x - a / (x + 2)

theorem range_a_and_inequality (a x1 x2 : ℝ) (h_deriv: ∀ (x : ℝ), f' x a = 0 → x = x1 ∨ x = x2) (h_lt: x1 < x2) (h_extreme: f (x1) a = f (x2) a):
  (-2 < a ∧ a < 0) → 
  (f (x1) a / x2 + 1 < 0) :=
by
  sorry

end range_a_and_inequality_l180_180459


namespace one_third_pow_3_eq_3_pow_nineteen_l180_180389

theorem one_third_pow_3_eq_3_pow_nineteen (y : ℤ) (h : (1 / 3 : ℝ) * (3 ^ 20) = 3 ^ y) : y = 19 :=
by
  sorry

end one_third_pow_3_eq_3_pow_nineteen_l180_180389


namespace determine_c_d_l180_180902

theorem determine_c_d (c d : ℝ) :
  (∀ z : ℂ, (z * z + (15 + c * complex.I) * z + (35 + d * complex.I) = 0) → 
   (∃ u v : ℝ, z = u + v * complex.I ∧ z.conj = u - v * complex.I)) → 
   (c = 0 ∧ d = 0) :=
by
  sorry

end determine_c_d_l180_180902


namespace inequality_correct_l180_180323

noncomputable def a : ℝ := Real.exp (-0.5)
def b : ℝ := 0.5
noncomputable def c : ℝ := Real.log 1.5

theorem inequality_correct : a > b ∧ b > c :=
by
  sorry

end inequality_correct_l180_180323


namespace minimum_colors_l180_180600

-- Define a regular decagon with vertices colored alternately
def vertices : list (ℕ × bool) :=
  [(0, tt), (1, ff), (2, tt), (3, ff), (4, tt), (5, ff), (6, tt), (7, ff), (8, tt), (9, ff)]

-- Define the condition that every black point must be connected to every white point
def connection_condition (black white : ℕ) : Prop :=
  black % 2 = 0 ∧ white % 2 = 1

-- Define the condition that no two line segments of the same color intersect except at their endpoints
def nonintersecting_condition (colored_connections : list (ℕ × ℕ × ℕ)) : Prop :=
  ∀ ⦃c₁ c₂ : ℕ × ℕ × ℕ⦄, c₁ ∈ colored_connections → c₂ ∈ colored_connections →
  (c₁.0 = c₂.0 ∧ c₁.1 = c₂.1 → c₁.2 = c₂.2) ∧
  (¬(c₁.0 = c₂.0 ∧ c₁.1 = c₂.1) → ¬(c₁.2 = c₂.2 ∧ line_segments_intersect c₁.0 c₁.1 c₂.0 c₂.1))

-- Define the problem to prove the minimum number of colors needed.
theorem minimum_colors : ∃ C : ℕ, (∀ (colored_connections : list (ℕ × ℕ × ℕ)),
  (∀ (p : ℕ × ℕ), p ∈ colored_connections → connection_condition p.0 p.1) →
  nonintersecting_condition colored_connections → colored_connections.length ≤ C) ∧ C = 5 :=
sorry

end minimum_colors_l180_180600


namespace KimSweaterTotal_l180_180438

theorem KimSweaterTotal :
  let monday := 8
  let tuesday := monday + 2
  let wednesday := tuesday - 4
  let thursday := wednesday
  let friday := monday / 2
  monday + tuesday + wednesday + thursday + friday = 34 := by
  sorry

end KimSweaterTotal_l180_180438


namespace radius_increase_proof_l180_180106

noncomputable def increase_in_radius
  (r : ℝ) (d_original : ℝ) (d_snow : ℝ) (actual_distance : ℝ) (conversion_factor : ℝ) 
  (new_radius : ℝ) (Δr : ℝ) : Prop :=
  let C_original := 2 * real.pi * r in
  let miles_per_rotation_original := C_original / conversion_factor in
  let rotations_original := d_original / miles_per_rotation_original in
  let C_new := 2 * real.pi * new_radius in
  let miles_per_rotation_new := C_new / conversion_factor in
  let rotations_new := actual_distance / miles_per_rotation_new in
  rotations_original = rotations_new ∧ Δr = new_radius - r

theorem radius_increase_proof : 
  increase_in_radius 15 450 440 450 63360 15.34 0.34 := by
  sorry

end radius_increase_proof_l180_180106


namespace no_snow_probability_l180_180283

theorem no_snow_probability (p1 p2 p3 p4 : ℚ) 
  (h1 : p1 = 2 / 3) 
  (h2 : p2 = 3 / 4) 
  (h3 : p3 = 5 / 6) 
  (h4 : p4 = 1 / 2) : 
  (1 - p1) * (1 - p2) * (1 - p3) * (1 - p4) = 1 / 144 :=
by
  sorry

end no_snow_probability_l180_180283


namespace smallest_gcd_yz_l180_180837

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

theorem smallest_gcd_yz {x y z : ℕ} (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  gcd x y = 360 → gcd x z = 1176 → gcd y z = 24 :=
begin
  intros,
  sorry
end

end smallest_gcd_yz_l180_180837


namespace Ivan_can_transport_all_prisoners_l180_180073

-- Definitions of the conditions
def prisoners : ℕ := 43
def boat_capacity : ℕ := 2
def werewolves_known (P : ℕ) : Prop := P ≥ 40

-- Main theorem statement
theorem Ivan_can_transport_all_prisoners :
  ∃ (strategy : (ℕ → bool)), -- strategy to determine if a prisoner is ready to be transported based on conditions
  ∀ (k : ℕ), k ∈ finset.range prisoners → werewolves_known k →  
  ∃ (success : bool), success := 
  sorry

end Ivan_can_transport_all_prisoners_l180_180073


namespace ivan_rescue_ivan_succeeds_l180_180070

-- Definitions of Conditions
def boat_capacity (n : ℕ) : Prop := n = 2
def return_trip_capacity (n : ℕ) : Prop := n = 1
def werewolf_info (p : ℕ) (k : ℕ) : Prop := p >= 40
def behavior_constraint (will_ride : ℕ → ℕ → Prop) : Prop :=
  ∀ i j, (will_ride i j) == (¬ (werewolf_info i j))

-- Define the problem "Ivan can transport all prisoners to the mainland" as a theorem
theorem ivan_rescue (prisoners : ℕ) (boat_capacity : ℕ) (return_trip_capacity : ℕ) (werewolf_info : ℕ → ℕ → Prop)
  (will_ride : ℕ → ℕ → Prop) (behavior_constraint : Prop) : Prop :=
  prisoners = 43 ∧
  boat_capacity = 2 ∧
  return_trip_capacity = 1 ∧
  (∀ p, werewolf_info p ≥ 40) ∧
  (∀ i j, (will_ride i j) == (¬ (werewolf_info i j))) →
  true -- indicating Ivan can successfully transport all prisoners

-- Proof omitted
theorem ivan_succeeds : (∀ (prisoners : ℕ) (boat_capacity : ℕ) (return_trip_capacity : ℕ) (werewolf_info : ℕ → ℕ → Prop)
  (will_ride : ℕ → ℕ → Prop) (behavior_constraint : Prop), 
  prisoners = 43 ∧ boat_capacity = 2 ∧ return_trip_capacity = 1 ∧ 
  (∀ p, werewolf_info p ≥ 40) ∧ 
  (∀ i j, (will_ride i j) == (¬ (werewolf_info i j))) →
  true) :=
by {
  assume prisoners boat_capacity return_trip_capacity werewolf_info will_ride behavior_constraint,
  assume h₁ : prisoners = 43 ∧ boat_capacity = 2 ∧ return_trip_capacity = 1 ∧
    (∀ p, werewolf_info p ≥ 40) ∧
    (∀ i j, (will_ride i j) == (¬ (werewolf_info i j))),
  exact true.intro,
}

end ivan_rescue_ivan_succeeds_l180_180070


namespace percentage_of_money_saved_l180_180306

variable (eff_old cost_gasoline : ℝ)

def eff_new : ℝ := 1.5 * eff_old
def cost_diesel : ℝ := 1.3 * cost_gasoline
def distance : ℝ := 1000

noncomputable def percentage_savings : ℝ :=
  let old_car_cost := (distance / eff_old) * cost_gasoline
  let new_car_cost := (distance / eff_new) * cost_diesel
  let savings := old_car_cost - new_car_cost
  (savings / old_car_cost) * 100

theorem percentage_of_money_saved :
  percentage_savings eff_old cost_gasoline = 13 + 1/3 := by
sorry

end percentage_of_money_saved_l180_180306


namespace matrix_identity_26_l180_180093

-- Define the problem statement variables and matrix product conditions.
theorem matrix_identity_26 (a b c d : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0):
  (λ ⟨a1, a2, b1, b2⟩, ⟨5*a1, 5*a2, 3*b1, 3*b2⟩) ⟨a, b, c, d⟩ =
  (λ ⟨a1, a2, b1, b2⟩, ⟨15*a1 - 9*a2, 10*a1 - 6*a2, 15*b1 - 9*b2, 10*b1 - 6*b2⟩) ⟨a, b, c, d⟩ →
  a + b + c + d = 26 :=
by
  -- Conditions on matrices are provided in the implications
  assume h,
  -- Insert proof here
  sorry

end matrix_identity_26_l180_180093


namespace problem1_problem2_l180_180356

section geometry_problem

def C : ℝ → ℝ → Prop := λ x y, (x - 2)^2 + y^2 = 25

variables (P : ℝ × ℝ) (x y : ℝ)

-- Conditions
def P_condition_1 : Prop := P = (-1, 3/2)
def P_condition_2 : Prop := P ∈ { (x, y) | x + y + 6 = 0 }

-- Correct answer for the first part
def line_l_1_eq_1 : Prop := ∀ (x y : ℝ), 3*x - 4*y + 9 = 0 → C x y
def line_l_1_eq_2 : Prop := ∀ (x y : ℝ), x = -1 → C x y

-- Correct answer for the second part
def fixed_points : set (ℝ × ℝ) := { (2, 0), (-2, -4) }

-- Proof problems
theorem problem1 : (P_condition_1 P) → 
                 ∃ line_l : (ℝ → ℝ → Prop), (∀ (A B : ℝ × ℝ), 
                 line_l A.1 A.2 ∧ line_l B.1 B.2 → 
                 C A.1 A.2 ∧ C B.1 B.2 ∧ 
                 dist A B = 8) :=
by sorry

theorem problem2 : (P_condition_2 P) → 
                 ∀ circle : (ℝ → ℝ → Prop), 
                 circle P.1 P.2 → 
                 circle 2 0 → 
                 circle (-2) (-4) :=
by sorry

end geometry_problem

end problem1_problem2_l180_180356


namespace restore_arithmetic_operations_l180_180992

/--
Given the placeholders \(A, B, C, D, E\) for operations in the equations:
1. \(4 A 2 = 2\)
2. \(8 = 4 C 2\)
3. \(2 D 3 = 5\)
4. \(4 = 5 E 1\)

Prove that:
(a) \(A = ÷\)
(b) \(B = =\)
(c) \(C = ×\)
(d) \(D = +\)
(e) \(E = -\)
-/
theorem restore_arithmetic_operations {A B C D E : String} (h1 : B = "=") 
    (h2 : "4" ++ A  ++ "2" ++ B ++ "2" = "4" ++ "÷" ++ "2" ++ "=" ++ "2")
    (h3 : "8" ++ "=" ++ "4" ++ C ++ "2" = "8" ++ "=" ++ "4" ++ "×" ++ "2")
    (h4 : "2" ++ D ++ "3" ++ "=" ++ "5" = "2" ++ "+" ++ "3" ++ "=" ++ "5")
    (h5 : "4" ++ "=" ++ "5" ++ E ++ "1" = "4" ++ "=" ++ "5" ++ "-" ++ "1") :
  (A = "÷") ∧ (B = "=") ∧ (C = "×") ∧ (D = "+") ∧ (E = "-") := by
    sorry

end restore_arithmetic_operations_l180_180992


namespace f_neg_a_l180_180970

def f (x : ℝ) : ℝ := x + Real.sin x

theorem f_neg_a (a : ℝ) (h : f a = 1) : f (-a) = -1 := 
by
  sorry

end f_neg_a_l180_180970


namespace form_divisibility_l180_180474

-- Definition of the set X containing numbers of a specific form
def X :=
  {n : ℕ | ∃ (k : ℕ) (a : fin (k + 1) → ℕ), (∀ i, a i ∈ {1, 2, ..., 9}) ∧
    n = ∑ i in finset.range (k + 1), a i * 10^(2 * i) }

-- Statement of the proof problem 
theorem form_divisibility (p q : ℕ) :
  ∃ m ∈ X, (2^p * 3^q) ∣ m :=
sorry

end form_divisibility_l180_180474


namespace min_number_of_gennadys_l180_180664

theorem min_number_of_gennadys (a b v g : ℕ) (h_a : a = 45) (h_b : b = 122) (h_v : v = 27)
    (h_needed_g : g = 49) :
    (b - 1) - (a + v) = g :=
by
  -- We include sorry because we are focusing on the statement, not the proof itself.
  sorry

end min_number_of_gennadys_l180_180664


namespace perimeter_triangle_XYZ_l180_180578

-- Given definitions
def isMidpoint (p q r : Point) : Prop :=
  dist p q = dist q r

def isEquilateral (triangle: Triangle) : Prop :=
  triangle.a = 10 ∧ triangle.b = 10 ∧ triangle.c = 10

def isSolidRightPrism (prism : Prism) : Prop :=
  prism.height = 20 ∧ isEquilateral(prism.base1) ∧ isEquilateral(prism.base2)

-- Key Points
variables (A B C D E F X Y Z : Point)
variable (prism : Prism)
variable (triangle : Triangle)
variable (base1 base2 : Triangle)

-- The theorem we need to prove
theorem perimeter_triangle_XYZ : isSolidRightPrism prism 
  ∧ prism.base1 = triangle 
  ∧ isMidpoint A X C 
  ∧ isMidpoint B Y C 
  ∧ isMidpoint D Z C 
  → (triangle.perimeter = 5 + 10 * Real.sqrt 5) :=
sorry

end perimeter_triangle_XYZ_l180_180578


namespace problem_statement_l180_180383

-- Definition of the conditions
variables {a : ℝ} (h₀ : a > 0) (h₁ : a ≠ 1)

-- The Lean 4 statement for the problem
theorem problem_statement (h : 0 < a ∧ a < 1) : 
  (∀ x y : ℝ, x < y → a^x > a^y) → 
  (∀ x : ℝ, (2 - a) * x^3 > 0) ∧ 
  (∀ x : ℝ, (2 - a) * x^3 > 0 → 0 < a ∧ a < 2 ∧ (∀ x y : ℝ, x < y → a^x > a^y) → False) :=
by
  intros
  sorry

end problem_statement_l180_180383


namespace digit_in_sequence_2021st_position_l180_180103

theorem digit_in_sequence_2021st_position :
    let seq := (list.range 1000).bind (λ n, n.digits 10).reverse in
    seq.get? (2021 - 1) = some 1 :=
by
  sorry

end digit_in_sequence_2021st_position_l180_180103


namespace choose_marbles_l180_180134

theorem choose_marbles (m : Fin 15 → Prop)
  (h_total : ∃ (reds : Set (Fin 15)), reds = {i : Fin 15 | i ∈ {1, 2, 3, 4, 5}})
  (h_black : ∃ (b : Fin 15), m b)
  (h_choose : ∀ (S : Set (Fin 15)), S ⊆ {i | m i} → S.card = 5 → (S ∩ {i | i ∈ {1, 2, 3, 4, 5}}).card = 1 ∧ b ∉ S) :
  ∃ (n : ℕ), n = 630 :=
sorry

end choose_marbles_l180_180134


namespace exponent_of_five_in_factorial_l180_180056

theorem exponent_of_five_in_factorial:
  (nat.factors 30!).count 5 = 7 :=
begin
  sorry
end

end exponent_of_five_in_factorial_l180_180056


namespace scientific_notation_of_508_billion_l180_180160

theorem scientific_notation_of_508_billion:
  (508 * (10:ℝ)^9) = (5.08 * (10:ℝ)^11) := 
begin
  sorry
end

end scientific_notation_of_508_billion_l180_180160


namespace solve_quadratic_eq_l180_180129

theorem solve_quadratic_eq (x : ℝ) : x^2 + 8 * x = 9 ↔ x = -9 ∨ x = 1 :=
by
  sorry

end solve_quadratic_eq_l180_180129


namespace solve_l180_180444

variable (a b : ℕ)

def is_positive (x : ℕ) : Prop := x > 0

def is_prime (n : ℕ) : Prop := Nat.Prime n

def divides (d n : ℕ) : Prop := d ∣ n

theorem solve (h1 : is_positive a)
  (h2 : is_positive b)
  (h3 : is_prime (a + b + 1))
  (h4 : divides (a + b + 1) (4 * a * b - 1)) : a = b := 
by
  sorry

end solve_l180_180444


namespace perimeter_of_rectangle_l180_180420

theorem perimeter_of_rectangle (DC BC P : ℝ) (hDC : DC = 12) (hArea : 1/2 * DC * BC = 30) : P = 2 * (DC + BC) → P = 34 :=
by
  sorry

end perimeter_of_rectangle_l180_180420


namespace min_gennadys_l180_180641

theorem min_gennadys (alexanders borises vasilies : ℕ) (x : ℕ) 
    (h1 : alexanders = 45)
    (h2 : borises = 122)
    (h3 : vasilies = 27)
    (h4 : x = 49)
    (h5 : borises - 1 = alexanders + vasilies + x) :
  x = 49 := 
begin
  sorry,
end

end min_gennadys_l180_180641


namespace triangle_inequality_l180_180061
-- Step d: Lean 4 Statement

theorem triangle_inequality 
  (A B C : ℝ) 
  (h1 : A + B + C = π) 
  (h2 : 0 < A ∧ A < π) 
  (h3 : 0 < B ∧ B < π) 
  (h4 : 0 < C ∧ C < π) :
  ( (cos A / cos B) ^ 2 + (cos B / cos C) ^ 2 + (cos C / cos A) ^ 2) 
  ≥ 4 * (cos A ^ 2 + cos B ^ 2 + cos C ^ 2) :=
by {
  sorry 
}

end triangle_inequality_l180_180061


namespace min_period_f_pi_l180_180351

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ := 2 * sqrt 3 * cos (ω * x) * cos (ω * x + π / 2) + 2 * sin (ω * x)^2

theorem min_period_f_pi (ω : ℝ) (hω : ω > 0) (hT : is_periodic (f x ω) π) :
  (ω = 1) ∧ (∀ k : ℤ, ∀ x : ℝ, x ∈ set.Icc (k * π + π / 6) (k * π + 2 * π / 3) → monotone_increasing_on (f x ω) x) ∧
  (∀ x ∈ set.Icc (π / 3) π, f x ω ∈ set.Icc 0 3) :=
sorry

end min_period_f_pi_l180_180351


namespace angle_between_AM_CN_is_60_l180_180856

open EuclideanGeometry

variables {A B C N K M : Point} {α : ℝ}

-- Assuming there exists a right triangle ABC with ∠A = 60°
variable (h_triangle: ∠A = 60° ∧ is_right_triangle A B C)

-- Point N is on the hypotenuse AB and point K is the midpoint of CN
variable (h_N_on_hypotenuse :  ∃ N, N ∈ AB) 
variable (h_K_midpoint : K = midpoint C N)

-- AK = AC
variable (h_AK_eq_AC : AK = AC)

-- The medians of triangle BCN intersect at point M
variable (h_M_median : M = centroid B C N)

-- Prove the angle between lines AM and CN is 60°
theorem angle_between_AM_CN_is_60 :
  ∠AM CN = 60°
:= sorry

end angle_between_AM_CN_is_60_l180_180856


namespace limit_of_S_n_l180_180715

theorem limit_of_S_n
  (S_n : ℕ → ℝ)
  (h : ∀ n : ℕ, S_n = ∑ k in range n, ((2 * k + 1) * real.pi / (4 * n + 1) - (2 * k * real.pi / (4 * n - 1))))
  : tendsto S_n at_top (𝓝 (real.pi / 8)) :=
sorry

end limit_of_S_n_l180_180715


namespace unique_triangle_solution_l180_180268

noncomputable def triangle_solutions (a b A : ℝ) : ℕ :=
sorry -- Placeholder for actual function calculating number of solutions

theorem unique_triangle_solution : triangle_solutions 30 25 150 = 1 :=
sorry -- Proof goes here

end unique_triangle_solution_l180_180268


namespace termite_ridden_not_collapsing_l180_180927

theorem termite_ridden_not_collapsing
  (total_homes : ℕ)
  (termite_ridden_fraction : ℚ)
  (collapsing_fraction_of_termite_ridden : ℚ)
  (h1 : termite_ridden_fraction = 1/3)
  (h2 : collapsing_fraction_of_termite_ridden = 1/4) :
  (termite_ridden_fraction - (termite_ridden_fraction * collapsing_fraction_of_termite_ridden)) = 1/4 := 
by {
  sorry
}

end termite_ridden_not_collapsing_l180_180927


namespace snakes_that_can_add_are_happy_l180_180994

-- Definitions for snakes and their properties
structure Snake where
  id : Nat
  is_purple : Bool
  is_happy : Bool
  can_add : Bool
  can_subtract : Bool
  is_magical : Bool

variables {snakes : List Snake}

-- Conditions based on the problem statement
def condition1 := ∀ s ∈ snakes, s.is_happy → s.can_add
def condition2 := ∀ s ∈ snakes, s.is_purple → ¬s.can_subtract
def condition3 := ∀ s ∈ snakes, ¬s.can_subtract → ¬s.can_add
def condition4 := ∃ s ∈ snakes, s.is_magical ∧ s.can_add ∧ s.can_subtract
def count_purple := (snakes.countp (λ s, s.is_purple)) = 6
def count_happy := (snakes.countp (λ s, s.is_happy)) = 7
def count_magical := (snakes.countp (λ s, s.is_magical)) = 3
def total_snakes := snakes.length = 15

-- The theorem to prove
theorem snakes_that_can_add_are_happy 
  (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) 
  (h5 : count_purple) (h6 : count_happy) (h7 : count_magical) (h8 : total_snakes) : 
  ∀ s ∈ snakes, s.can_add → s.is_happy := 
sorry

end snakes_that_can_add_are_happy_l180_180994


namespace _l180_180917

noncomputable def is_isosceles_right_triangle (P F1 F2 : (ℝ × ℝ)) : Prop :=
  let d1 := (P.1 - F2.1)^2 + (P.2 - F2.2)^2
  let d2 := (F1.1 - F2.1)^2 + (F1.2 - F2.2)^2
  d1 = d2

noncomputable theorem ellipse_eccentricity (a b c e : ℝ) (F1 F2 P : (ℝ × ℝ))
  (h1 : F1 = (0, 0))
  (h2 : F2 = (c, 0))
  (h3 : P = (c, b^2 / a))
  (h4 : is_isosceles_right_triangle P F1 F2)
  : e = sqrt 2 - 1 := by
  sorry

end _l180_180917


namespace sequence_fifth_term_l180_180171

noncomputable def sequence : ℕ → ℕ
| 0       := 0
| (n + 1) := 4 * sequence n + 3

theorem sequence_fifth_term : sequence 5 = 255 := sorry

end sequence_fifth_term_l180_180171


namespace min_gennadys_l180_180634

theorem min_gennadys (alexanders borises vasilies : ℕ) (x : ℕ) 
    (h1 : alexanders = 45)
    (h2 : borises = 122)
    (h3 : vasilies = 27)
    (h4 : x = 49)
    (h5 : borises - 1 = alexanders + vasilies + x) :
  x = 49 := 
begin
  sorry,
end

end min_gennadys_l180_180634


namespace part1_part2_l180_180763

def f (x : ℝ) := |2 * x - 1| + |2 * x + 2|

theorem part1 (x : ℝ) (M : ℝ) (hM : M = 3) :
  f(x) < M + |2 * x + 2| ↔ -1 < x ∧ x < 2 :=
by
  sorry

theorem part2 (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (M : ℝ) (hM : M = 3) (h : a^2 + 2 * b^2 = M) :
  2 * a + b ≤ (3 * Real.sqrt 6) / 2 :=
by
  sorry

end part1_part2_l180_180763


namespace valid_paths_from_jack_to_jill_l180_180873

noncomputable def number_of_paths_avoiding_dangerous_intersection : ℕ :=
  17

theorem valid_paths_from_jack_to_jill :
  let jills_house := (4, 3)
      dangerous_intersection := (2, 2)
      total_paths := binomial 7 4
      paths_dangerous_intersection :=
        (binomial 4 2) * (binomial 3 2)
  in total_paths - paths_dangerous_intersection = number_of_paths_avoiding_dangerous_intersection :=
by
  let jills_house := (4, 3)
  let dangerous_intersection := (2, 2)
  let total_paths := binomial 7 4
  let paths_dangerous_intersection := (binomial 4 2) * (binomial 3 2)
  let number_of_paths_avoiding_dangerous_intersection := 17
  sorry

end valid_paths_from_jack_to_jill_l180_180873


namespace sam_days_to_do_job_l180_180944

theorem sam_days_to_do_job (S : ℝ) :
  (1 / S + 1 / 6 + 1 / 2 = 1 / (12 / 11)) → S = 4 :=
by
  intro h
  have eq : 12 / 11 = 1.09090909091 := by linarith
  rw eq at h
  sorry

end sam_days_to_do_job_l180_180944


namespace prove_problem_a_prove_problem_b_l180_180081

noncomputable def problem_a (f : ℝ → ℝ) (h_cont : Continuous f) : Prop :=
  (∀ α : ℝ, ∫ x in 0..1, f (real.sin (x + α)) = 0) → 
  (∀ x : ℝ, x ∈ Icc (-1) 1 → f x = 0)

noncomputable def problem_b (f : ℝ → ℝ) (h_cont : Continuous f) : Prop :=
  (∀ n : ℤ, ∫ x in 0..1, f (real.sin (↑n * x)) = 0) → 
  (∀ x : ℝ, x ∈ Icc (-1) 1 → f x = 0)

theorem prove_problem_a (f : ℝ → ℝ) (h_cont : Continuous f) :
  problem_a f h_cont :=
sorry

theorem prove_problem_b (f : ℝ → ℝ) (h_cont : Continuous f) :
  problem_b f h_cont :=
sorry

end prove_problem_a_prove_problem_b_l180_180081


namespace smallest_congruent_difference_l180_180455

theorem smallest_congruent_difference :
  let m := 111 in -- the smallest three-digit integer congruent to 7 (mod 13)
  let n := 1008 in -- the smallest four-digit integer congruent to 7 (mod 13)
  n - m = 897 := by
  sorry

end smallest_congruent_difference_l180_180455


namespace checker_arrangements_five_digit_palindromes_l180_180858

noncomputable def comb (n k : ℕ) : ℕ := Nat.choose n k

theorem checker_arrangements :
  comb 32 12 * comb 20 12 = Nat.choose 32 12 * Nat.choose 20 12 := by
  sorry

theorem five_digit_palindromes :
  9 * 10 * 10 = 900 := by
  sorry

end checker_arrangements_five_digit_palindromes_l180_180858


namespace axis_of_symmetry_even_function_l180_180393

theorem axis_of_symmetry_even_function
  (f : ℝ → ℝ)
  (h_even : ∀ x, f x = f (-x)) :
  ∃ a, (a = -1) ∧ ∀ x, f (x + 1) = f (a - (x + 1)) :=
begin
  sorry
end

end axis_of_symmetry_even_function_l180_180393


namespace product_mod_self_inverse_l180_180906

theorem product_mod_self_inverse 
  {n : ℕ} (hn : 0 < n) (a b : ℤ) (ha : a * a % n = 1) (hb : b * b % n = 1) :
  (a * b) % n = 1 := 
sorry

end product_mod_self_inverse_l180_180906


namespace solve_equation_l180_180130

theorem solve_equation :
  ∀ (x : ℝ), (sqrt(5 * x - 6) + 8 / sqrt(5 * x - 6) = 6) → (x = 22 / 5 ∨ x = 2) :=
by
  intros x h
  sorry

end solve_equation_l180_180130


namespace like_terms_monomials_l180_180395

theorem like_terms_monomials (m n : ℕ) (h₁ : m = 2) (h₂ : n = 1) : m + n = 3 := 
by
  sorry

end like_terms_monomials_l180_180395


namespace num_possible_values_of_a_l180_180112

theorem num_possible_values_of_a : 
  (Exclusive α : Type) [LinearOrder α] (a b c d : α) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h1 : a > b) (h2 : b > c) (h3 : c > d) 
  (sum_eq : a + b + c + d = 2010)
  (sq_sum_eq : a^2 - b^2 + c^2 - d^2 = 2010) :
  ∃ n : ℕ, n = 501 := 
sorry

end num_possible_values_of_a_l180_180112


namespace cone_central_angle_l180_180168

theorem cone_central_angle (r l : ℝ) (π : ℝ) (C : ℝ) (n : ℝ) :
  r = 3 →
  l = 10 →
  π := Real.pi →
  C = 2 * π * r →
  C = (n / 360) * 2 * π * l →
  n = 108 :=
by
  intros
  sorry

end cone_central_angle_l180_180168


namespace min_g_at_e_l180_180759

-- Define the function f(x)
def f (x : ℝ) (k : ℝ) (a : ℝ) : ℝ := k * x - a^x

-- Define the conditions
variables (k : ℝ) (a : ℝ)
variable h0 : a > 0
variable h1 : a ≠ 1

-- Define the derivative of f(x)
def f' (x : ℝ) := k - a^x * log a

-- Specific definitions for the problem
def g (a : ℝ) : ℝ := 
  let t := (log (1 / log a)) / log a in
    t - a^t

-- Prove that the minimum value of g(a) when a = exp 1 is -1
theorem min_g_at_e : g (Real.exp 1) = -1 := by
  sorry

end min_g_at_e_l180_180759


namespace solve_for_3a_l180_180750

-- Define the variables
variables {a b c : ℝ}

-- Define the conditions as hypotheses
hypothesis h1 : 3 * a - 2 * b - 2 * c = 30
hypothesis h2 : sqrt (3 * a) - sqrt (2 * b + 2 * c) = 4
hypothesis h3 : a + b + c = 10

theorem solve_for_3a : 3 * a = 30 :=
by sorry

end solve_for_3a_l180_180750


namespace find_c_for_circle_radius_five_l180_180690

theorem find_c_for_circle_radius_five
  (c : ℝ)
  (h : ∀ x y : ℝ, x^2 + 8 * x + y^2 + 2 * y + c = 0) :
  c = -8 :=
sorry

end find_c_for_circle_radius_five_l180_180690


namespace cost_of_5_cans_of_juice_is_2_l180_180579

-- Define the conditions
variable (original_price : ℕ) (sale_discount : ℕ) (total_cost : ℕ)
variable (ice_cream_tubs : ℕ) (juice_cans : ℕ)

-- Provide the values for the conditions
def original_price_ice_cream : original_price := 12
def sale_discount_ice_cream : sale_discount := 2
def cost_for_two_tubs_and_10_cans : total_cost := 24
def number_of_ice_cream_tubs : ice_cream_tubs := 2
def number_of_juice_cans : juice_cans := 10

-- Define the sale price of ice cream
def sale_price_of_ice_cream := original_price_ice_cream - sale_discount_ice_cream

-- Define the total cost for two tubs of ice cream
def total_cost_for_ice_cream := number_of_ice_cream_tubs * sale_price_of_ice_cream

-- Define the cost of 10 cans of juice
def cost_for_10_cans_of_juice := total_cost - total_cost_for_ice_cream

-- Define the cost of 5 cans of juice
def cost_for_5_cans_of_juice := cost_for_10_cans_of_juice / 2

-- The proof problem statement
theorem cost_of_5_cans_of_juice_is_2 :
  cost_for_5_cans_of_juice = 2 := by
  -- proof steps go here
  sorry

end cost_of_5_cans_of_juice_is_2_l180_180579


namespace find_b1008_l180_180331

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

axiom a1 (n : ℕ) : a n = (3 / 4) * (S n) + 2

axiom Sn_def (n : ℕ) : S n = ∑ i in finset.range n, a i

noncomputable def b (n : ℕ) := real.log (a n) / real.log 2

theorem find_b1008 : b a 1008 = 2017 :=
by
  sorry

end find_b1008_l180_180331


namespace problem1_problem2_problem3_l180_180365

def f_k (k : ℤ) (a x : ℝ) : ℝ := a^x - (k-1)*a^(-x)
def g (a x : ℝ) : ℝ := f_k 2 a x / f_k 0 a x
noncomputable def f_1 (a x : ℝ) : ℝ := a^x
noncomputable def h (a m x : ℝ) : ℝ := f_k 0 a (2*x) + 2*m*f_k 2 a x

-- Problem 1
theorem problem1 (a : ℝ) (h1 : a > 1) :
  ∀ x : ℝ, strict_mono (g a) :=
sorry

-- Problem 2
theorem problem2 (a : ℝ) (ha : a = 2) :
  ( ∃ x1 x2 : ℝ, x1 ∈ Icc 1 2 ∧ x2 ∈ Icc 1 2 ∧ f_1 a x1 - f_1 a x2 = 2 ) →
  odd (g 2) :=
sorry

-- Problem 3
theorem problem3 (a m : ℝ) (h_zero : ∃ x : ℝ, x ∈ Ici (1 : ℝ) ∧ h a m x = 0) :
  m ≤ -17/12 :=
sorry

end problem1_problem2_problem3_l180_180365


namespace satellite_modular_units_l180_180575

variables (N S T U : ℕ)
variable (h1 : N = S / 3)
variable (h2 : S / T = 1 / 9)
variable (h3 : U * N = 8 * T / 9)

theorem satellite_modular_units :
  U = 24 :=
by sorry

end satellite_modular_units_l180_180575


namespace polynomial_identity_l180_180711

noncomputable def p(x : ℝ) : ℝ := (1 + Real.sqrt 5) / 2 * x + (3 - Real.sqrt 5) / 2

theorem polynomial_identity (x : ℝ) : 
  p(p(x)) = x * (p(x) - 1) + x ^ 2 := by
  sorry

end polynomial_identity_l180_180711


namespace sec_minus_tan_l180_180794

theorem sec_minus_tan (x : ℝ) (h : real.sec x + real.tan x = 7 / 3) :
  real.sec x - real.tan x = 3 / 7 :=
sorry

end sec_minus_tan_l180_180794


namespace trisectors_form_equilateral_triangle_l180_180157

theorem trisectors_form_equilateral_triangle
    (A B C : Point)
    (triangle_ABC : Triangle A B C)
    (trisected : ∀ (p : Point), p ∈ trisectors_of(triangle_ABC)) :
    is_equilateral (triangle formed by trisectors_of(triangle_ABC)) :=
by
  sorry

end trisectors_form_equilateral_triangle_l180_180157


namespace sequence_2023_value_l180_180264

theorem sequence_2023_value : ∃ (p q : ℕ), Nat.gcd p q = 1 ∧ a₁ = 2 ∧ a₂ = 4 / 9 ∧
  (∀ n ≥ 3, a n = (a (n-2) * a (n-1)) / (3 * a (n-2) - 2 * a (n-1))) ∧
  (∀ n, a n = 8 / (4 * n + 5)) ∧
  (a 2023 = p / q) ∧ (p + q = 8115) :=
sorry

end sequence_2023_value_l180_180264


namespace midpoints_and_equal_angles_implies_equal_diagonals_l180_180406

variables {A B C D M N K : Type} [EuclideanGeometry.cm A] [EuclideanGeometry.cm B] [EuclideanGeometry.cm C] [EuclideanGeometry.cm D] [EuclideanGeometry.cm M] [EuclideanGeometry.cm N] [EuclideanGeometry.cm K]

-- Suppose ABCD is a convex quadrilateral
-- M and N are the midpoints of AB and CD respectively
-- and a line through M and N forms equal angles with the diagonals AC and BD.
-- We need to show that AC = BD.

theorem midpoints_and_equal_angles_implies_equal_diagonals 
  (ABCD_convex : convex_quadrilateral A B C D)
  (M_mid_AB : midpoint M A B)
  (N_mid_CD : midpoint N C D)
  (MN_equal_angles : ∃ K, is_midpoint K A D ∧ 
    (angle_eq (line_through M N) (line_through A C) (line_through M N) (line_through B D))) :
  (length A C) = (length B D) := 
sorry

end midpoints_and_equal_angles_implies_equal_diagonals_l180_180406


namespace expression_for_f_value_of_sin_a_l180_180753

open Real

noncomputable def f (x ϕ : ℝ) : ℝ := sin (2 * x) * cos ϕ + cos (2 * x) * sin ϕ

theorem expression_for_f (ϕ : ℝ) (hϕ1 : 0 < ϕ) (hϕ2 : ϕ < π) : 
  f (π / 4) ϕ = √3 / 2 ↔ ϕ = π / 6 := by
  sorry

theorem value_of_sin_a (a : ℝ) (ϕ : ℝ) (hϕ1 : 0 < ϕ) (hϕ2 : ϕ < π) 
  (ha1 : π / 2 < a) (ha2 : a < π) :
  (f (a / 2 - π / 3) ϕ = 5 / 13 ↔ sin a = 12 / 13) := by
  sorry

end expression_for_f_value_of_sin_a_l180_180753


namespace piecewise_inequality_l180_180361

def piecewise_function (x : ℝ) : ℝ :=
if x >= 0 then x^2 + x else x - x^2

theorem piecewise_inequality (a : ℝ) (h : piecewise_function a > piecewise_function (2 - a)) :
  a > 1 :=
by
  sorry

end piecewise_inequality_l180_180361


namespace total_colored_pencils_l180_180676

-- Define Cheryl's number of colored pencils
def Cheryl := ℕ

-- Define Cyrus's number of colored pencils
def Cyrus := ℕ

-- Define Madeline's number of colored pencils
def Madeline := ℕ

-- Given conditions as Lean definitions
def cheryl_thrice_cyrus (C : Cheryl) (Y : Cyrus) : Prop := C = 3 * Y
def madeline_half_cheryl (C : Cheryl) (M : Madeline) : Prop := M = 63 ∧ 63 = C / 2

-- Total colored pencils theorem
theorem total_colored_pencils (C : Cheryl) (Y : Cyrus) (M : Madeline) 
  (h1 : cheryl_thrice_cyrus C Y) (h2 : madeline_half_cheryl C M) : C + Y + M = 231 :=
by
  sorry

end total_colored_pencils_l180_180676


namespace less_than_reciprocal_l180_180752

theorem less_than_reciprocal (n : ℚ) : 
  n = -3 ∨ n = 3/4 ↔ (n = -1/2 → n >= 1/(-1/2)) ∧
                           (n = -3 → n < 1/(-3)) ∧
                           (n = 3/4 → n < 1/(3/4)) ∧
                           (n = 3 → n > 1/3) ∧
                           (n = 0 → false) := sorry

end less_than_reciprocal_l180_180752


namespace area_of_circle_l180_180111

namespace MathProof

/-- 
Points A and B lie on circle ω.
The tangent lines to ω at A and B intersect at a point on the x-axis.
Determine the area of ω.
-/
theorem area_of_circle (A B : ℝ × ℝ) (O : ℝ × ℝ) (hA : A = (4, 15)) (hB : B = (14, 7)) (hO : O = (-1, 0))
  (h_tangent_intersect : ∀ (T : ℝ × ℝ), (TangentLineThrough A) = T → T.2 = 0)
  : circle_area (circumference_through A B O) = 250 * π := sorry

end MathProof

end area_of_circle_l180_180111


namespace sec_minus_tan_l180_180805

theorem sec_minus_tan (x : ℝ) (h : Real.sec x + Real.tan x = 7 / 3) : Real.sec x - Real.tan x = 3 / 7 :=
by
  sorry

end sec_minus_tan_l180_180805


namespace find_r4_l180_180424

-- Definitions of the problem conditions
variable (r1 r2 r3 r4 r5 r6 r7 : ℝ)
-- Given radius of the smallest circle
axiom smallest_circle : r1 = 6
-- Given radius of the largest circle
axiom largest_circle : r7 = 24
-- Given that radii of circles form a geometric sequence
axiom geometric_sequence : r2 = r1 * (r7 / r1)^(1/6) ∧ 
                            r3 = r1 * (r7 / r1)^(2/6) ∧
                            r4 = r1 * (r7 / r1)^(3/6) ∧
                            r5 = r1 * (r7 / r1)^(4/6) ∧
                            r6 = r1 * (r7 / r1)^(5/6)

-- Statement to prove
theorem find_r4 : r4 = 12 :=
by
  sorry

end find_r4_l180_180424


namespace square_area_l180_180417

-- Definition of the problem conditions
def VertexCoordinates := {0, 2, 6, 8}

def VerticallyAligned (x_0 : ℝ) := (A : ℝ × ℝ) × (C : ℝ × ℝ) where 
  A = (x_0, 0) 
  C = (x_0, 8)

-- Statement to prove the area of the square given the conditions
theorem square_area (x_0 x_1 x_2 : ℝ) (A B C D : ℝ × ℝ) 
  (vc : VertexCoordinates = {0, 2, 6, 8}) 
  (va : VerticallyAligned x_0) :
  A = (x_0, 0) → B = (x_1, 2) → C = (x_0, 8) → D = (x_2, 6) → 
  ∃ (s : ℝ), s^2 = 64 :=
begin
  sorry
end

end square_area_l180_180417


namespace range_of_k_l180_180335

theorem range_of_k (k : ℝ) : (∀ x : ℝ, |x + 1| + |x - 2| > k) → k > 3 := 
sorry

end range_of_k_l180_180335


namespace exponent_of_5_in_30_fact_l180_180021

def count_powers_of_5 (n : ℕ) : ℕ :=
  if n < 5 then 0
  else n / 5 + count_powers_of_5 (n / 5)

theorem exponent_of_5_in_30_fact : count_powers_of_5 30 = 7 := 
  by
    sorry

end exponent_of_5_in_30_fact_l180_180021


namespace sec_tan_identity_l180_180827

theorem sec_tan_identity (x : ℝ) (h : Real.sec x + Real.tan x = 7 / 3) : Real.sec x - Real.tan x = 3 / 7 := 
by
  sorry

end sec_tan_identity_l180_180827


namespace no_integer_solutions_l180_180119

theorem no_integer_solutions (x y z : ℤ) (h : ¬ (x = 0 ∧ y = 0 ∧ z = 0)) : 2 * x^4 + y^4 ≠ 7 * z^4 :=
sorry

end no_integer_solutions_l180_180119


namespace perpendicular_lines_l180_180415

-- Define the lines and their slopes
def line1 (a : ℝ) : ℝ × ℝ × ℝ := (2, a, -1)  -- coefficients of 2x + ay - 1 = 0
def line2 (a : ℝ) : ℝ × ℝ × ℝ := (2a - 1, -1, 1) -- coefficients of (2a-1)x - y + 1 = 0

-- Define the slope of a line with coefficients (A, B, C)
noncomputable def slope (A B : ℝ) : ℝ := -A / B

-- Prove the condition given the perpendicularity of the lines
theorem perpendicular_lines (a : ℝ) (h : slope 2 a * slope (2a - 1) (-1) = -1) : a = 2 / 3 :=
by
  -- Simply state that the proof is omitted
  sorry

end perpendicular_lines_l180_180415


namespace lulu_quadratic_l180_180461

theorem lulu_quadratic (b : ℝ) (h1 : b > 0)
  (h2 : ∃ m : ℝ, (x^2 + bx + 36) = (x + m)^2 + 4) : b = 8 * sqrt 2 :=
sorry

end lulu_quadratic_l180_180461


namespace XY_squared_l180_180448

-- Define the triangle ABC, tangents, and specified conditions.
variables {A B C T X Y : Type}
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited T] [Inhabited X] [Inhabited Y]

-- Define the given conditions as assumptions.
axiom triangle_acute_scalene (△ABC : A) (ω : Type) : Prop
axiom tangents_intersect (T : Type) (ω : Type) (B C : A) : Prop
axiom projections_to_lines (T : Type) (X : A) (Y : A) (AB AC : Type) : Prop
axiom BT_CT_constant (BT CT : Type) : BT = CT := 18
axiom BC_constant (BC : Type) : BC := 24
axiom sum_of_squares (TX TY XY : Type) : (TX^2 + TY^2 + XY^2) = 1458

-- Define the main theorem
theorem XY_squared : XY^2 = 858 :=
by
  -- Assuming the given conditions
  assume (h1 : triangle_acute_scalene △ABC ω),
  assume (h2 : tangents_intersect T ω B C),
  assume (h3 : projections_to_lines T X Y AB AC),
  assume (h4 : BT_CT_constant BT CT),
  assume (h5 : BC_constant BC),
  assume (h6 : sum_of_squares TX TY XY),
  -- Proof will be provided here
  sorry

end XY_squared_l180_180448


namespace possible_degrees_of_remainder_l180_180205

-- Let f be a polynomial and g be the polynomial 3x^3 - 5x^2 + 3x - 20
def g : ℚ[X] := 3 * X ^ 3 - 5 * X ^ 2 + 3 * X - 20

-- The degree of g is 3
lemma deg_g : g.degree = 3 := by sorry

-- The degrees of the possible remainders when dividing by g
def possible_remainder_degrees : Finset ℕ := {0, 1, 2}

-- The statement we need to prove
theorem possible_degrees_of_remainder (f r q : ℚ[X]) (h : f = q * g + r) :
  r.degree < g.degree → r.degree ∈ possible_remainder_degrees := by sorry

end possible_degrees_of_remainder_l180_180205


namespace arithmetic_sequence_a1_l180_180334

theorem arithmetic_sequence_a1 
  (a : ℕ → ℝ)
  (d : ℝ)
  (h1 : ∀ (k : ℕ), a (k + 1) = a k + d)
  (sum_first_100 : ∑ k in Finset.range 100, a k = 100)
  (sum_last_100 : ∑ k in Finset.range 100, a (k + 900) = 1000) :
  a 0 = 0.505 :=
sorry

end arithmetic_sequence_a1_l180_180334


namespace find_angle_l180_180865

theorem find_angle {s : ℝ} (h_square : ∀ (A B C D : ℝ), (AB = s) → (BC = s) → (CD = s) → (DA = s))
  (h_parallel : ∀ (B D E C : ℝ), BD ∥ CE)
  (h_equal : ∀ (B D E : ℝ), (BE = BD)) :
  ∠E = 30 :=
sorry

end find_angle_l180_180865


namespace minimum_gennadys_l180_180618

theorem minimum_gennadys (alexs borises vasilies x : ℕ) (h₁ : alexs = 45) (h₂ : borises = 122) (h₃ : vasilies = 27)
    (h₄ : ∀ i, i ∈ list.range (borises-1) → alexs + vasilies + x > i) : 
    x = 49 :=
by 
    sorry

end minimum_gennadys_l180_180618


namespace trapezoid_area_l180_180961

theorem trapezoid_area (BC AD AB CD : ℝ) (CK : ℝ) :
  (BC = 3) ∧ (AB = 13) ∧ (CD = 13) ∧ (AD = 13) ∧ (AB + BC + CD + AD = 42) ∧ (CK = 12) →
  ((1 / 2) * (AD + BC) * CK = 96) :=
begin
  sorry
end

end trapezoid_area_l180_180961


namespace a10_equals_512_l180_180714

theorem a10_equals_512 (n : ℕ) (a : ℕ → ℝ) (r : ℝ) (a1_pos: 0 < a 1)
  (S_succ_mul : ∀ n, (∑ i in finset.range (2 * n + 1), a i) = 3 * (∑ i in finset.range n, a (2 * i)))
  (h_product : a 1 * a 2 * a 3 = 8) :
  a 10 = 512 :=
sorry

end a10_equals_512_l180_180714


namespace two_non_intersecting_similar_polyhedra_inside_convex_l180_180938

-- Define the notion of a polyhedron, convexity, similarity, and scaling in this simplified context
variables {M : Type} [Polyhedron M]

def is_convex (P : Polyhedron) : Prop := sorry
def is_similar (P Q : Polyhedron) (r : ℝ) : Prop := sorry -- P is similar to Q with scale factor r
def is_subset (P Q : Polyhedron) : Prop := sorry
def is_disjoint (P Q : Polyhedron) : Prop := P ∩ Q = ∅

theorem two_non_intersecting_similar_polyhedra_inside_convex (M : Polyhedron) (h_convex : is_convex M) :
  ∃ (M_A M_B : Polyhedron), 
    is_similar M_A M (1/2) ∧ 
    is_similar M_B M (1/2) ∧ 
    is_subset M_A M ∧ 
    is_subset M_B M ∧ 
    is_disjoint M_A M_B := 
sorry

end two_non_intersecting_similar_polyhedra_inside_convex_l180_180938


namespace angle_between_vectors_l180_180523

variables {a b : EuclideanSpace ℝ (Fin 3)}
variables {θ : ℝ}

-- Conditions
def valid_vectors (a b : EuclideanSpace ℝ (Fin 3)) : Prop :=
  ((a + b) ⬝ (2 • a - b) = -4) ∧
  (EuclideanSpace.norm a = 2) ∧
  (EuclideanSpace.norm b = 4)

-- Theorem to prove
theorem angle_between_vectors (h : valid_vectors a b) : θ = real.arccos (1 / 2) :=
sorry

end angle_between_vectors_l180_180523


namespace surface_area_of_cone_l180_180505

-- Definitions for the parameters involved
def slant_height : ℝ := 2
def radius : ℝ := 1

-- Definition for the surface area of a cone given slant height and radius
def surface_area_cone (l r : ℝ) : ℝ := 
  let lateral_area := π * r * l
  let base_area := π * r^2
  lateral_area + base_area

-- The theorem we need to prove
theorem surface_area_of_cone : surface_area_cone slant_height radius = 3 * π :=
  by sorry

end surface_area_of_cone_l180_180505


namespace find_s_l_l180_180978

def line_eq (x : ℝ) : ℝ := (3/4) * x + 5

def vec_eq (t : ℝ) (s l : ℝ) : ℝ × ℝ :=
  ( -7 + t * l, s + t * (-5))

theorem find_s_l :
  (∃ (s l : ℝ), 
  s = line_eq (-7) ∧
  let p := vec_eq 1 s l in p.2 = line_eq p.1) 
  ↔ (s, l) = (-1/4, -20/3) := by
  sorry

end find_s_l_l180_180978


namespace probability_heads_all_three_tosses_l180_180516

theorem probability_heads_all_three_tosses :
  (1 / 2) * (1 / 2) * (1 / 2) = 1 / 8 := 
sorry

end probability_heads_all_three_tosses_l180_180516


namespace find_a_l180_180757

def f (x : ℝ) : ℝ :=
  if x > 0 then 2 * x
  else x + 1

theorem find_a (a : ℝ) (h : f a + f 1 = 0) : a = -3 := by
  sorry

end find_a_l180_180757


namespace range_of_a_l180_180901

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 9 * x + a^2 / x + 7
  else 9 * x + a^2 / x - 7

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f a x ≥ a + 1) → a ≤ -8/7 :=
by
  intros h
  -- Detailed proof would go here
  sorry

end range_of_a_l180_180901


namespace fair_game_expected_winnings_l180_180181

theorem fair_game_expected_winnings (num_players : ℕ) (total_pot : ℝ) 
  (p : ℕ → ℝ) (stakes : ℕ → ℝ) :
  num_players = 36 →
  (∀ k, p k = (35 / 36) ^ (k - 1) * p 1) →
  (∀ k, stakes k = total_pot * p k) →
  (∀ k, let L_k := stakes k in total_pot * p k - L_k * p k - L_k + L_k * p k = 0) :=
sorry

end fair_game_expected_winnings_l180_180181


namespace sum_of_odd_binom_coeff_eq_256_constant_term_eq_84_l180_180864

noncomputable theory

-- Definitions based on the conditions
def binom_coeff (n k : ℕ) : ℕ := Nat.choose n k
def binom_exp (x : ℚ) (n : ℕ) : ℚ := x + 1 / x^(1/2)

-- Statement for the first part of the problem
theorem sum_of_odd_binom_coeff_eq_256 : 
  ∑ i in List.range 10, if i % 2 = 1 then binom_coeff 9 i else 0 = 2^8 :=
sorry

-- Statement for the second part of the problem
theorem constant_term_eq_84 :
  binom_coeff 9 6 = 84 :=
sorry

end sum_of_odd_binom_coeff_eq_256_constant_term_eq_84_l180_180864


namespace exists_xy_for_cube_difference_l180_180126

theorem exists_xy_for_cube_difference (a : ℕ) (h : 0 < a) :
  ∃ x y : ℤ, x^2 - y^2 = a^3 :=
sorry

end exists_xy_for_cube_difference_l180_180126


namespace lisa_flight_time_l180_180921

theorem lisa_flight_time
  (distance : ℕ) (speed : ℕ) (time : ℕ)
  (h_distance : distance = 256)
  (h_speed : speed = 32)
  (h_time : time = distance / speed) :
  time = 8 :=
by sorry

end lisa_flight_time_l180_180921


namespace sphere_radius_eq_three_l180_180398

theorem sphere_radius_eq_three (r : ℝ) (h : (4 / 3) * Real.pi * r^3 = 4 * Real.pi * r^2) : r = 3 := 
sorry

end sphere_radius_eq_three_l180_180398


namespace ellipse_equation_l180_180153

-- Definitions for the conditions
def ellipse (x y a b : ℝ) : Prop := 
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

def ellipse_foci (F₁ F₂ : ℝ × ℝ) (a b : ℝ) : Prop := 
  F₁ = (-ℝ.sqrt (a^2 - b^2), 0) ∧ F₂ = (ℝ.sqrt (a^2 - b^2), 0)

def upper_vertex (A : ℝ × ℝ) (a b : ℝ) : Prop := 
  A = (0, b)

def triangle_area (A F₁ F₂ : ℝ × ℝ) : ℝ :=
  0.5 * (A.1 * (F₁.2 - F₂.2) + F₁.1 * (F₂.2 - A.2) + F₂.1 * (A.2 - F₁.2))

def angle_condition (F₁ A F₂ : ℝ × ℝ) (θ : ℝ) : Prop := 
  θ = 4 * (π / 6)

-- The main statement to be proven
theorem ellipse_equation (a b c : ℝ) 
  (F₁ F₂ A : ℝ × ℝ) 
  (h_ellipse : ellipse F₁.1 F₁.2 a b) 
  (h_foci : ellipse_foci F₁ F₂ a b) 
  (h_vertex : upper_vertex A a b) 
  (h_area : triangle_area A F₁ F₂ = ℝ.sqrt 3) 
  (h_angle : angle_condition F₁ A F₂ 30): 
  a = 2 ∧ b = 1 ∧ c = ℝ.sqrt 3 ∧ (F₁.1 / (2:ℝ))^2 + F₁.2^2 = 1 :=
sorry

end ellipse_equation_l180_180153


namespace solution_set_f_2_minus_x_l180_180971

def f (x : ℝ) (a : ℝ) (b : ℝ) := (x - 2) * (a * x + b)

theorem solution_set_f_2_minus_x (a b : ℝ) (h_even : b - 2 * a = 0)
  (h_mono : 0 < a) :
  {x : ℝ | f (2 - x) a b > 0} = {x : ℝ | x < 0 ∨ x > 4} :=
by
  sorry

end solution_set_f_2_minus_x_l180_180971


namespace min_gennadys_needed_l180_180654

variables (A B V G : ℕ)

theorem min_gennadys_needed
  (hA : A = 45)
  (hB : B = 122)
  (hV : V = 27)
  (hG : ∀ i, i < 121 → A + V < 121 → G ≥ 49) :
  G = 49 :=
sorry

end min_gennadys_needed_l180_180654


namespace slope_angle_of_y_eq_0_l180_180397

theorem slope_angle_of_y_eq_0  :
  ∀ (α : ℝ), (∀ (y x : ℝ), y = 0) → α = 0 :=
by
  intros α h
  sorry

end slope_angle_of_y_eq_0_l180_180397


namespace solve_for_x_l180_180723

noncomputable def solve_determinant (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) : ℂ :=
  (3 * b^2 + a * b) / (a + b)

theorem solve_for_x (a b x : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) :
  det ![
    [x + a, x, x],
    [x, x + b, x],
    [2 * x + a + b, 2 * x, 2 * x]
  ] = 0 → x = solve_determinant a b ha hb := 
by
  sorry

end solve_for_x_l180_180723


namespace sec_sub_tan_l180_180800

theorem sec_sub_tan (x : ℝ) (h : sec x + tan x = 7 / 3) : sec x - tan x = 3 / 7 := by
  sorry

end sec_sub_tan_l180_180800


namespace sec_tan_eq_l180_180831

theorem sec_tan_eq (x : ℝ) (h : Real.cos x ≠ 0) : 
  Real.sec x + Real.tan x = 7 / 3 → Real.sec x - Real.tan x = 3 / 7 :=
by
  intro h1
  sorry

end sec_tan_eq_l180_180831


namespace min_gennadies_l180_180633

theorem min_gennadies 
  (n_Alexanders : ℕ) (n_Borises : ℕ) (n_Vasilies : ℕ) 
  (x_Gennadies : ℕ) 
  (h_Alexanders : n_Alexanders = 45) 
  (h_Borises   : n_Borises = 122) 
  (h_Vasilies  : n_Vasilies = 27) 
  (h_condition : ∀ p : ℕ, p = n_Borises - 1 → p = 121) 
  (h_total     : ∀ q : ℕ, q = 45 + 27 → q = 72)
  : x_Gennadies = 49 := 
sorry

end min_gennadies_l180_180633


namespace problem_statement_l180_180327

theorem problem_statement (n : ℕ) : (-1 : ℤ) ^ n * (-1) ^ (2 * n + 1) * (-1) ^ (n + 1) = 1 := 
by
  sorry

end problem_statement_l180_180327


namespace total_colored_pencils_l180_180678

noncomputable def total_pencils (Cheryl Cyrus Madeline : ℕ) : ℕ :=
  Cheryl + Cyrus + Madeline

theorem total_colored_pencils : 
  ∀ (Cheryl Cyrus Madeline : ℕ), Madeline = 63 →
  Madeline * 2 = Cheryl → 
  Cheryl = 3 * Cyrus → 
  total_pencils Cheryl Cyrus Madeline = 231 :=
by 
  intros Cheryl Cyrus Madeline hMadeline hCheryl hCyrus
  rw [hMadeline, hCheryl, hCyrus]
  simp [total_pencils]
  sorry

end total_colored_pencils_l180_180678


namespace minimum_value_am_gm_l180_180366

noncomputable def fixed_point_value (a : ℝ) (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) : Prop :=
log a 1 - 1 = -1

theorem minimum_value_am_gm (m n : ℝ) (hmn_pos : 0 < m * n) (fixed_point_on_line : 2 * m + n = 1) :
  1/m + 1/n ≥ 3 + 2 * Real.sqrt 2 :=
by
  sorry

end minimum_value_am_gm_l180_180366


namespace proof_parallel_D1D2_l_l180_180890

-- Definitions based on conditions
def Circle (α : Type) := α → Prop
def Point (α : Type) := α
def Line (α : Type) := α → α → Prop
def TangentCircles (α : Type) (Γ Γ₁ Γ₂ : Circle α) := Γ₁ ∩ Γ₂ = ∅ ∧ ∃ z : Point α, Γ z ∧ Γ₁ z ∧ Γ₂ z
def TangentToLine (α : Type) (Γ : Circle α) (l : Line α) (A : Point α) := ∀ B : Point α, Γ B → l A B

variables {α : Type} [Nonempty α]
variables (Γ Γ₁ Γ₂ : Circle α) (l : Line α)
variables (A A₁ A₂ B₁ B₂ C D₁ D₂ : Point α)

-- Problem statement using Lean 4:
theorem proof_parallel_D1D2_l
  (h1 : TangentCircles α Γ Γ₁)
  (h2 : TangentCircles α Γ Γ₂)
  (h3 : TangentCircles α Γ₁ Γ₂)
  (h4 : TangentToLine α Γ l A)
  (h5 : TangentToLine α Γ₁ l A₁)
  (h6 : TangentToLine α Γ₂ l A₂)
  (h7 : ∀ x, (A₁ = x ∧ x = A ∧ A = A₂) → True)
  (h8 : ∃ M : Point α, (A₁C ∩ (Line A₂ B₂)) M = D₁)
  (h9 : ∃ N : Point α, (A₂C ∩ (Line A₁ B₁)) N = D₂) :
  Parallel (Line D₁ D₂) l := 
by sorry

end proof_parallel_D1D2_l_l180_180890


namespace min_gennadies_l180_180629

theorem min_gennadies 
  (n_Alexanders : ℕ) (n_Borises : ℕ) (n_Vasilies : ℕ) 
  (x_Gennadies : ℕ) 
  (h_Alexanders : n_Alexanders = 45) 
  (h_Borises   : n_Borises = 122) 
  (h_Vasilies  : n_Vasilies = 27) 
  (h_condition : ∀ p : ℕ, p = n_Borises - 1 → p = 121) 
  (h_total     : ∀ q : ℕ, q = 45 + 27 → q = 72)
  : x_Gennadies = 49 := 
sorry

end min_gennadies_l180_180629


namespace exponent_of_five_in_factorial_l180_180053

theorem exponent_of_five_in_factorial:
  (nat.factors 30!).count 5 = 7 :=
begin
  sorry
end

end exponent_of_five_in_factorial_l180_180053


namespace proof_problem_l180_180773

def point (x : ℝ) (y : ℝ) (z : ℝ) := (x, y, z)

def vector (p1 p2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p2.1 - p1.1, p2.2 - p1.2, p2.3 - p1.3)

def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2 + v.3^2)

def unit_vector (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let mag := magnitude v
  in (v.1 / mag, v.2 / mag, v.3 / mag)

def projection_magnitude (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  (dot_product v1 v2).abs / magnitude v1

theorem proof_problem :
  let A := point (-1) 2 1
  let B := point 1 3 1
  let C := point (-2) 4 2
  let AB := vector A B
  let AC := vector A C
  let BC := vector B C
  let unit_opposite_BC := (-3 / Real.sqrt 11, 1 / Real.sqrt 11, 1 / Real.sqrt 11)
  in
    (dot_product AB AC) = 0 ∧
    unit_vector (-BC) = unit_opposite_BC ∧
    dot_product AC BC / (magnitude AC * magnitude BC) = Real.sqrt 66 / 11 ∧
    projection_magnitude AB BC = Real.sqrt 5 := sorry

end proof_problem_l180_180773


namespace triangle_proof_l180_180060

-- Definitions based on the conditions
def angle_C : ℝ := Real.pi / 3
def side_b : ℝ := 8
def area_triangle : ℝ := 10 * Real.sqrt 3

noncomputable def find_side_c (a b : ℝ) (angle_C : ℝ) : ℝ :=
Real.sqrt (a^2 + b^2 - 2 * a * b * Real.cos angle_C)

noncomputable def find_cos_B_minus_C (a b c : ℝ) : ℝ :=
let cos_B := (a^2 + c^2 - b^2) / (2 * a * c) in
let sin_B := Real.sqrt (1 - cos_B^2) in
cos_B * (1 / 2) + sin_B * (Real.sqrt 3 / 2)

theorem triangle_proof:
  ∃ a c, (∃ (a : ℝ), side_b * a * Real.sin angle_C = area_triangle * 2) ∧
  c = find_side_c 5 8 angle_C ∧
  find_cos_B_minus_C 5 8 7 = 13 / 14 := 
by {
  use 5,
  use 7,
  split,
  {
    use 5,
    simp [angle_C, side_b, area_triangle],
    linarith,
  },
  split,
  {
    simp [find_side_c, angle_C, side_b],
    norm_num,
  },
  {
    simp [find_cos_B_minus_C, angle_C, side_b],
    sorry,
  }
}

end triangle_proof_l180_180060


namespace normal_distribution_interval_probability_l180_180401

noncomputable def normal_cdf (μ σ : ℝ) (x : ℝ) : ℝ :=
sorry

theorem normal_distribution_interval_probability
  (σ : ℝ) (hσ : σ > 0)
  (hprob : normal_cdf 1 σ 2 - normal_cdf 1 σ 0 = 0.8) :
  (normal_cdf 1 σ 2 - normal_cdf 1 σ 1) = 0.4 :=
sorry

end normal_distribution_interval_probability_l180_180401


namespace mode_is_3_6_8_variance_is_475_l180_180235

def datalist := [2, 6, 8, 3, 3, 4, 6, 8]

theorem mode_is_3_6_8 : Multiset.mode (datalist : Multiset ℕ) = {3, 6, 8} :=
by
  sorry

theorem variance_is_475 : variance datalist = 4.75 :=
by
  sorry 

end mode_is_3_6_8_variance_is_475_l180_180235


namespace fair_game_condition_l180_180196

variables (n : ℕ) (L : ℝ) {p : ℕ → ℝ}

-- Define the probability p_k for the k-th player.
def probability (k : ℕ) : ℝ := (35.0 / 36.0) ^ k

-- Define the expected value of the k-th player.
def expected_value (L : ℝ) (Lk : ℝ) (k : ℕ) : ℝ := L * probability k - Lk

-- Define the conditions of the fair game.
def fair_game := ∀ k, expected_value L (L * probability k) k = 0

-- Main theorem stating that the game is fair if stakes decrease proportionally by a factor of 35/36.
theorem fair_game_condition (k : ℕ) (L : ℝ) :
  fair_game :=
by
  sorry

end fair_game_condition_l180_180196


namespace total_people_in_line_l180_180065

theorem total_people_in_line (people_in_front : ℕ) (people_behind : ℕ) (J : ℕ) : 
  people_in_front = 4 → 
  people_behind = 7 → 
  J = 1 → 
  people_in_front + J + people_behind = 12 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end total_people_in_line_l180_180065


namespace cube_mod_inverse_l180_180090

theorem cube_mod_inverse (p : ℕ) [hp : fact (nat.prime p)] (a : ℤ) (ha : a * a % p = 1 % p) : 
  (a ^ 3 % p) = (a % p) :=
sorry

end cube_mod_inverse_l180_180090


namespace min_gennadys_l180_180609

-- Defining the basic constants for each name type
def Alexanders : Nat := 45
def Borises : Nat := 122
def Vasilies : Nat := 27

-- Define the proof statement to check the minimum number of Gennadys needed
theorem min_gennadys (a b v : Nat) (no_adjacent: a = 45 ∧ b = 122 ∧ v = 27) : ∃ g : Nat, g = 49 :=
by
  -- Using provided conditions
  cases no_adjacent with h_a h_bv
  cases h_bv with h_b h_v
  -- Correct answer derived from the solution
  use 49
  -- skipping proof details 
  sorry

end min_gennadys_l180_180609


namespace total_gold_value_l180_180883

def legacy_bars : ℕ := 5
def aleena_bars : ℕ := legacy_bars - 2
def value_per_bar : ℕ := 2200
def total_bars : ℕ := legacy_bars + aleena_bars
def total_value : ℕ := total_bars * value_per_bar

theorem total_gold_value : total_value = 17600 :=
by
  -- Begin proof
  sorry

end total_gold_value_l180_180883


namespace exponent_of_5_in_30_factorial_l180_180003

theorem exponent_of_5_in_30_factorial : 
  (nat.factors 30!).count 5 = 7 :=
sorry

end exponent_of_5_in_30_factorial_l180_180003


namespace magic_square_proof_l180_180866

theorem magic_square_proof
    (a b c d e S : ℕ)
    (h1 : 35 + e + 27 = S)
    (h2 : 30 + c + d = S)
    (h3 : a + 32 + b = S)
    (h4 : 35 + c + b = S)
    (h5 : a + c + 27 = S)
    (h6 : 35 + c + b = S)
    (h7 : 35 + c + 27 = S)
    (h8 : a + c + d = S) :
  d + e = 35 :=
  sorry

end magic_square_proof_l180_180866


namespace min_number_of_gennadys_l180_180660

theorem min_number_of_gennadys (a b v g : ℕ) (h_a : a = 45) (h_b : b = 122) (h_v : v = 27)
    (h_needed_g : g = 49) :
    (b - 1) - (a + v) = g :=
by
  -- We include sorry because we are focusing on the statement, not the proof itself.
  sorry

end min_number_of_gennadys_l180_180660


namespace sum_powers_mod_p_l180_180908

theorem sum_powers_mod_p (p : ℕ) (n : ℕ) [fact (nat.prime p)] :
  let S := ∑ k in finset.range p, k ^ n
  in S % p = if n % (p - 1) = 0 then p - 1 else 0 :=
sorry

end sum_powers_mod_p_l180_180908


namespace minimum_gennadies_l180_180612

theorem minimum_gennadies (A B V G : ℕ) (hA : A = 45) (hB : B = 122) (hV : V = 27) (hGap : G + A + V >= B - 1) :
  G >= 49 :=
by 
  have := by linarith [hGap, hA, hB, hV]
  exact this

end minimum_gennadies_l180_180612


namespace exponent_of_five_in_30_factorial_l180_180035

theorem exponent_of_five_in_30_factorial : 
  nat.factorial_prime_exponent 30 5 = 7 := 
sorry

end exponent_of_five_in_30_factorial_l180_180035


namespace john_weekly_earnings_after_raise_l180_180878

theorem john_weekly_earnings_after_raise (original_earnings : ℝ) (raise_percentage : ℝ) (raise_amount new_earnings : ℝ) 
  (h1 : original_earnings = 50) (h2 : raise_percentage = 60) (h3 : raise_amount = (raise_percentage / 100) * original_earnings) 
  (h4 : new_earnings = original_earnings + raise_amount) : 
  new_earnings = 80 := 
by sorry

end john_weekly_earnings_after_raise_l180_180878


namespace only_solution_l180_180143

theorem only_solution (x : ℝ) : (3 / (x - 3) = 5 / (x - 5)) ↔ (x = 0) := 
sorry

end only_solution_l180_180143


namespace fair_game_stakes_ratio_l180_180183

theorem fair_game_stakes_ratio (n : ℕ) (deck_size : ℕ) (player_count : ℕ)
  (L : ℕ → ℝ) : 
  deck_size = 36 → player_count = 36 → 
  (∀ k : ℕ, k < player_count - 1 → 
    (L (k + 1)) / (L k) = 35 / 36) :=
by
  intros h_deck_size h_player_count k hk
  simp [h_deck_size, h_player_count, hk]
  sorry

end fair_game_stakes_ratio_l180_180183


namespace car_distance_covered_l180_180557

theorem car_distance_covered :
  let speed := 97.5 -- in km/h
  let time := 4 -- in hours
  let distance := speed * time
  distance = 390 :=
by
  let speed := 97.5
  let time := 4
  let distance := speed * time
  sorry

end car_distance_covered_l180_180557


namespace count_distinct_four_digit_numbers_l180_180778

-- Conditions as definitions
def digits := {1, 2, 3, 4, 5}
def no_repetition (l : List ℕ) := l.Nodup

-- Statement of the problem
theorem count_distinct_four_digit_numbers : 
  {l : List ℕ // l.length = 4 ∧ no_repetition l ∧ ∀ x ∈ l, x ∈ digits}.card = 120 :=
sorry

end count_distinct_four_digit_numbers_l180_180778


namespace solve_sqrt_eq_l180_180475

theorem solve_sqrt_eq (x : ℝ) (hx : x = 19881 / 576) : sqrt x + sqrt (x + 3) = 12 :=
by
  sorry

end solve_sqrt_eq_l180_180475


namespace area_triangle_le_quarter_l180_180693

theorem area_triangle_le_quarter (S : ℝ) (S₁ S₂ S₃ S₄ S₅ S₆ S₇ : ℝ)
  (h₁ : S₃ + (S₂ + S₇) = S / 2)
  (h₂ : S₁ + S₆ + (S₂ + S₇) = S / 2) :
  S₁ ≤ S / 4 :=
by
  -- Proof skipped
  sorry

end area_triangle_le_quarter_l180_180693


namespace chelsea_guaranteed_victory_l180_180853

noncomputable def minimum_bullseye_shots_to_win (k : ℕ) (n : ℕ) : ℕ :=
  if (k + 5 * n + 500 > k + 930) then n else sorry

theorem chelsea_guaranteed_victory (k : ℕ) :
  minimum_bullseye_shots_to_win k 87 = 87 :=
by
  sorry

end chelsea_guaranteed_victory_l180_180853


namespace vector_addition_example_l180_180372

def vector_addition (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)

theorem vector_addition_example : vector_addition (1, -1) (-1, 2) = (0, 1) := 
by 
  unfold vector_addition 
  simp
  sorry

end vector_addition_example_l180_180372


namespace fill_4x4_table_with_constraints_l180_180726

theorem fill_4x4_table_with_constraints :
  ∃ (configuration_count : ℕ), configuration_count = 511 ∧
    ∀ (table : ℕ → ℕ → ℕ), (∀ i j, i < 4 → j < 4 → table i j ∈ {0, 1}) →
      (∀ i j, i < 3 → j < 4 → table i j * table (i + 1) j = 0) ∧
      (∀ i j, i < 4 → j < 3 → table i j * table i (j + 1) = 0) :=
begin
  use 511,
  split,
  { refl },
  sorry
end

end fill_4x4_table_with_constraints_l180_180726


namespace inequality_for_positive_numbers_l180_180120

theorem inequality_for_positive_numbers (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a - b)^2 / (2 * (a + b)) ≤ (sqrt ((a^2 + b^2) / 2) - sqrt (a * b)) ∧ 
  (sqrt ((a^2 + b^2) / 2) - sqrt (a * b)) ≤ (a - b)^2 / (sqrt 2 * (a + b)) :=
by
  sorry

end inequality_for_positive_numbers_l180_180120


namespace part1_max_min_part2_triangle_inequality_l180_180697

noncomputable def f (x k : ℝ) : ℝ :=
  (x^4 + k*x^2 + 1) / (x^4 + x^2 + 1)

theorem part1_max_min (k : ℝ): 
  (∀ x : ℝ, k ≥ 1 → 1 ≤ f x k ∧ f x k ≤ (1/3) * (k + 2)) ∧ 
  (∀ x : ℝ, k < 1 → (1/3) * (k + 2) ≤ f x k ∧ f x k ≤ 1) := 
sorry

theorem part2_triangle_inequality (k : ℝ) : 
  -1/2 < k ∧ k < 4 ↔ (∀ a b c : ℝ, (f a k + f b k > f c k) ∧ (f b k + f c k > f a k) ∧ (f c k + f a k > f b k)) :=
sorry

end part1_max_min_part2_triangle_inequality_l180_180697


namespace PA2_minus_PB2_eq_PBPD_minus_PAPC_l180_180931

-- Define the geometric entities
variable {R : ℝ} -- circumradius
variable {O : ℝ × ℝ} -- center of the circumcircle
variable {A B C D P : ℝ × ℝ} -- vertices and point on circumcircle

-- Assume P is on the circumcircle of square ABCD between C and D
variable (hP_on_circumcircle : P.1^2 + P.2^2 = R^2)
variable (hP_between_CD : ∃ x y, P = (x, y) ∧ (C.1 < x ∧ x < D.1 ∨ C.2 < y ∧ y < D.2))

-- Goal: Prove the given trigonometric identity
theorem PA2_minus_PB2_eq_PBPD_minus_PAPC :
  let PA := dist P A
  let PB := dist P B
  let PC := dist P C 
  let PD := dist P D 
  in PA^2 - PB^2 = PB * PD - PA * PC :=
by
  sorry

end PA2_minus_PB2_eq_PBPD_minus_PAPC_l180_180931


namespace kim_knit_sweaters_total_l180_180441

theorem kim_knit_sweaters_total :
  ∀ (M T W R F : ℕ), 
    M = 8 →
    T = M + 2 →
    W = T - 4 →
    R = T - 4 →
    F = M / 2 →
    M + T + W + R + F = 34 :=
by
  intros M T W R F hM hT hW hR hF
  rw [hM, hT, hW, hR, hF]
  norm_num
  sorry

end kim_knit_sweaters_total_l180_180441


namespace tens_digit_of_7_pow_35_l180_180203

theorem tens_digit_of_7_pow_35 : 
  (7 ^ 35) % 100 / 10 % 10 = 4 :=
by
  sorry

end tens_digit_of_7_pow_35_l180_180203


namespace concurrency_of_lines_l180_180888

open Function

-- Define the cyclic hexagon and intersection conditions.
variables {A1 A2 A3 B1 B2 B3 C1 C2 C3 D1 D2 D3 : Point}

-- Placeholder for cyclic check
def cyclic_hexagon : Prop :=
  cyclic {A1, B3, A2, B1, A3, B2}

-- Define the intersection conditions at a single point.
def intersect_at_single_point : Prop :=
  ∃ P : Point, collinear {A1, P, B1} ∧ collinear {A2, P, B2} ∧ collinear {A3, P, B3}

-- Define the intersection of lines for C1, C2, C3.
def define_intersection_points : Prop :=
  C1 = line_intersection (line A1 B1) (line A2 A3) ∧
  C2 = line_intersection (line A2 B2) (line A1 A3) ∧
  C3 = line_intersection (line A3 B3) (line A1 A2)

-- Define the conditional tangency and corresponding point on the circumcircle for D1, D2, D3.
def circumcircle_and_tangency : Prop :=
  lie_on_circumcircle D1 {A1, B3, A2, B1, A3, B2} ∧
  tangent (C1, B1, D1) (A2, A3) ∧
  lie_on_circumcircle D2 {A1, B3, A2, B1, A3, B2} ∧
  tangent (C2, B2, D2) (A1, A3) ∧
  lie_on_circumcircle D3 {A1, B3, A2, B1, A3, B2} ∧
  tangent (C3, B3, D3) (A1, A2)


-- The final theorem stating that A1D1, A2D2, A3D3 concur.
theorem concurrency_of_lines :
  cyclic_hexagon →
  intersect_at_single_point →
  define_intersection_points →
  circumcircle_and_tangency →
  concurrent {line A1 D1, line A2 D2, line A3 D3} :=
by sorry

end concurrency_of_lines_l180_180888


namespace recruits_total_l180_180982

theorem recruits_total (P N D : ℕ) (total_recruits : ℕ) 
  (h1 : P = 50) 
  (h2 : N = 100) 
  (h3 : D = 170)
  (h4 : (∃ x y, (x = 50) ∧ (y = 100) ∧ (x = 4 * y))
        ∨ (∃ x z, (x = 50) ∧ (z = 170) ∧ (x = 4 * z))
        ∨ (∃ y z, (y = 100) ∧ (z = 170) ∧ (y = 4 * z))) : 
  total_recruits = 211 :=
by
  sorry

end recruits_total_l180_180982


namespace num_of_sets_M_l180_180787

open Finset

theorem num_of_sets_M :
  ∀ (M P Q : Finset ℕ),
  (M ⊆ P) →
  (M ⊆ Q) →
  (P = {0, 1, 2}) →
  (Q = {0, 2, 4}) →
  M ⊆ ({0, 2}) ∧ (M.card = 0 ∨ M.card = 1 ∨ M.card = 2) :=
by
  intros M P Q hMP hMQ hP hQ
  rw [hP, hQ]
  have h : ({0, 1, 2} ∩ {0, 2, 4}) = {0, 2} := by simp
  sorry

end num_of_sets_M_l180_180787


namespace exponent_of_5_in_30_factorial_l180_180011

theorem exponent_of_5_in_30_factorial : 
  (nat.factors 30!).count 5 = 7 :=
sorry

end exponent_of_5_in_30_factorial_l180_180011


namespace part1_part2_l180_180687

def companions (a b : ℝ) : Prop := a + b = a * b

theorem part1 : companions (-1) 0.5 := 
by 
  unfold companions
  linarith

theorem part2 (m n : ℝ) (h : companions m n) : 
  -2 * m * n + (1 / 2) * (3 * m + 2 * (1 / 2 * n - m) + 3 * m * n - 6) = -3 := 
by 
  obtain ⟨m, n, h⟩ := h
  rw [←h]
  linarith

end part1_part2_l180_180687


namespace probability_A_is_70_l180_180839

open ProbabilityTheory

variables {Ω : Type*} [ProbabilitySpace Ω]

def P (A B : Event Ω) := Prob A * Prob B

theorem probability_A_is_70 (A B : Event Ω)
  (hB : Prob B = 0.6)
  (hA_and_B : Prob (A ∩ B) = 0.42)
  (indep : IndepEvents A B) :
  Prob A = 0.7 :=
by
  sorry

end probability_A_is_70_l180_180839


namespace surface_area_of_cube_l180_180552

theorem surface_area_of_cube (a : ℝ) : 
  let edge_length := 7 * a in
  let face_area := edge_length^2 in
  let surface_area := 6 * face_area in
  surface_area = 294 * a^2 :=
by
  sorry

end surface_area_of_cube_l180_180552


namespace triangle_is_isosceles_l180_180851

theorem triangle_is_isosceles
  (A B C : Type)
  [triangle : EuclideanGeometry.Triangle A B C]
  (a b c : ℝ)
  (h : c = 2 * a * Real.cos B) :
  a = b :=
sorry

end triangle_is_isosceles_l180_180851


namespace min_gennadies_l180_180648

noncomputable section

def minGennadiesNeeded (alexanders borises vasilies : Nat) : Nat :=
  let needed_gaps := borises - 1
  let total_others := alexanders + vasilies
  if needed_gaps > total_others then needed_gaps - total_others else 0

theorem min_gennadies (alexanders borises vasilies : Nat) (h_alex: alexanders = 45) (h_boris: borises = 122) (h_vasil: vasilies = 27):
  minGennadiesNeeded alexanders borises vasilies = 49 := by
  rw [h_alex, h_boris, h_vasil]
  simp [minGennadiesNeeded]
  sorry

end min_gennadies_l180_180648


namespace connie_s_problem_l180_180287

theorem connie_s_problem (y : ℕ) (h : 3 * y = 90) : y / 3 = 10 :=
by
  sorry

end connie_s_problem_l180_180287


namespace min_gennadies_l180_180645

noncomputable section

def minGennadiesNeeded (alexanders borises vasilies : Nat) : Nat :=
  let needed_gaps := borises - 1
  let total_others := alexanders + vasilies
  if needed_gaps > total_others then needed_gaps - total_others else 0

theorem min_gennadies (alexanders borises vasilies : Nat) (h_alex: alexanders = 45) (h_boris: borises = 122) (h_vasil: vasilies = 27):
  minGennadiesNeeded alexanders borises vasilies = 49 := by
  rw [h_alex, h_boris, h_vasil]
  simp [minGennadiesNeeded]
  sorry

end min_gennadies_l180_180645


namespace parabola_shifted_left_and_down_l180_180147

-- Define the original parabolic equation
def original_parabola (x : ℝ) : ℝ := 2 * x ^ 2 - 1

-- Define the transformed parabolic equation
def transformed_parabola (x : ℝ) : ℝ := 2 * (x + 1) ^ 2 - 3

-- Theorem statement
theorem parabola_shifted_left_and_down :
  ∀ x : ℝ, transformed_parabola x = 2 * (x + 1) ^ 2 - 3 :=
by 
  -- Proof Left as an exercise.
  sorry

end parabola_shifted_left_and_down_l180_180147


namespace chord_length_eq_l180_180154

noncomputable def length_of_chord (R : ℝ) (c : ℝ × ℝ) (A B C : ℝ) : ℝ :=
  2 * real.sqrt (R^2 - (abs (A * c.1 + B * c.2 + C) / real.sqrt (A^2 + B^2))^2)

theorem chord_length_eq (R : ℝ) (hR : R = 3) (c : ℝ × ℝ)
(hc : c = (0, 0)) (A B C : ℝ)
(h_line : A = 1 ∧ B = -2 ∧ C = 3) :
  length_of_chord R c A B C = 12 * real.sqrt 5 / 5 :=
by
  sorry

end chord_length_eq_l180_180154


namespace range_of_expression_l180_180338

open Real

theorem range_of_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1) :
  1 ≤ x^2 + y^2 + sqrt (x * y) ∧ x^2 + y^2 + sqrt (x * y) ≤ 9 / 8 :=
sorry

end range_of_expression_l180_180338


namespace base_triangle_not_equilateral_l180_180977

-- Define the lengths of the lateral edges
def SA := 1
def SB := 2
def SC := 4

-- Main theorem: the base triangle is not equilateral
theorem base_triangle_not_equilateral 
  (a : ℝ)
  (equilateral : a = a)
  (triangle_inequality1 : SA + SB > a)
  (triangle_inequality2 : SA + a > SC) : 
  a ≠ a :=
by 
  sorry

end base_triangle_not_equilateral_l180_180977


namespace boys_in_play_l180_180988

theorem boys_in_play 
  (girls : ℕ)
  (total_parents : ℕ)
  (parents_per_child : ℕ)
  (children := girls + 8)
  (total_parents = 2 * children) :
  ∃ boys : ℕ, boys = 8 :=
by {
  let B := 8,
  use B,
  exact rfl,
  sorry
}

end boys_in_play_l180_180988


namespace tara_spent_more_on_icecream_l180_180217

def iceCreamCount : ℕ := 19
def yoghurtCount : ℕ := 4
def iceCreamCost : ℕ := 7
def yoghurtCost : ℕ := 1

theorem tara_spent_more_on_icecream :
  (iceCreamCount * iceCreamCost) - (yoghurtCount * yoghurtCost) = 129 := 
  sorry

end tara_spent_more_on_icecream_l180_180217


namespace fair_game_condition_l180_180197

variables (n : ℕ) (L : ℝ) {p : ℕ → ℝ}

-- Define the probability p_k for the k-th player.
def probability (k : ℕ) : ℝ := (35.0 / 36.0) ^ k

-- Define the expected value of the k-th player.
def expected_value (L : ℝ) (Lk : ℝ) (k : ℕ) : ℝ := L * probability k - Lk

-- Define the conditions of the fair game.
def fair_game := ∀ k, expected_value L (L * probability k) k = 0

-- Main theorem stating that the game is fair if stakes decrease proportionally by a factor of 35/36.
theorem fair_game_condition (k : ℕ) (L : ℝ) :
  fair_game :=
by
  sorry

end fair_game_condition_l180_180197


namespace product_fractions_l180_180314

theorem product_fractions : 
  (∏ n in Finset.range (2006 - 2 + 1) + 2, (1 / n - 1 / (n + 1)) / (1 / (n + 1) - 1 / (n + 2))) = 1004 := 
by
  sorry

end product_fractions_l180_180314


namespace slope_of_intersection_points_l180_180317

theorem slope_of_intersection_points : 
  (∀ t : ℝ, ∃ x y : ℝ, (2 * x + 3 * y = 10 * t + 4) ∧ (x + 4 * y = 3 * t + 3)) → 
  (∀ t1 t2 : ℝ, t1 ≠ t2 → ((2 * ((10 * t1 + 4)  / 2) + 3 * ((-5/3 * t1 - 2/3)) = (10 * t1 + 4)) ∧ (2 * ((10 * t2 + 4) / 2) + 3 * ((-5/3 * t2 - 2/3)) = (10 * t2 + 4))) → 
  (31 * (((-5/3 * t1 - 2/3) - (-5/3 * t2 - 2/3)) / ((10 * t1 + 4) / 2 - (10 * t2 + 4) / 2)) = -4)) :=
sorry

end slope_of_intersection_points_l180_180317


namespace find_smallest_n_l180_180373

noncomputable def a (n : ℕ) : ℕ := 
  if n = 1 then 9 else a (n - 1) * (-1 / 3)^n + 1

noncomputable def S (n : ℕ) : ℕ :=
  ∑ k in range n, a k

theorem find_smallest_n :
  ∃ n : ℕ, n = 7 ∧ |(S n) - n - 6| < (1 / 125) :=
by
  sorry

end find_smallest_n_l180_180373


namespace sec_minus_tan_l180_180789

theorem sec_minus_tan (x : ℝ) (h : real.sec x + real.tan x = 7 / 3) :
  real.sec x - real.tan x = 3 / 7 :=
sorry

end sec_minus_tan_l180_180789


namespace meeting_time_l180_180477

/--
The Racing Magic takes 150 seconds to circle the racing track once.
The Charging Bull makes 40 rounds of the track in an hour.
Prove that Racing Magic and Charging Bull meet at the starting point for the second time 
after 300 minutes.
-/
theorem meeting_time (rac_magic_time : ℕ) (chrg_bull_rounds_hour : ℕ)
  (h1 : rac_magic_time = 150) (h2 : chrg_bull_rounds_hour = 40) : 
  ∃ t: ℕ, t = 300 := 
by
  sorry

end meeting_time_l180_180477


namespace minimum_gennadys_l180_180620

theorem minimum_gennadys (alexs borises vasilies x : ℕ) (h₁ : alexs = 45) (h₂ : borises = 122) (h₃ : vasilies = 27)
    (h₄ : ∀ i, i ∈ list.range (borises-1) → alexs + vasilies + x > i) : 
    x = 49 :=
by 
    sorry

end minimum_gennadys_l180_180620


namespace minimum_gennadys_l180_180625

theorem minimum_gennadys (alexs borises vasilies x : ℕ) (h₁ : alexs = 45) (h₂ : borises = 122) (h₃ : vasilies = 27)
    (h₄ : ∀ i, i ∈ list.range (borises-1) → alexs + vasilies + x > i) : 
    x = 49 :=
by 
    sorry

end minimum_gennadys_l180_180625


namespace arithmetic_geometric_sequence_S6_l180_180747

noncomputable def S_6 (a : Nat) (q : Nat) : Nat :=
  (q ^ 6 - 1) / (q - 1)

theorem arithmetic_geometric_sequence_S6 (a : Nat) (q : Nat) (h1 : a * q ^ 1 = 2) (h2 : a * q ^ 3 = 8) (hq : q > 0) : S_6 a q = 63 :=
by
  sorry

end arithmetic_geometric_sequence_S6_l180_180747


namespace frustum_radius_l180_180481

theorem frustum_radius (C1 C2 l: ℝ) (S_lateral: ℝ) (r: ℝ) :
  (C1 = 2 * r * π) ∧ (C2 = 6 * r * π) ∧ (l = 3) ∧ (S_lateral = 84 * π) → (r = 7) :=
by
  sorry

end frustum_radius_l180_180481


namespace interval_solution_l180_180984

noncomputable def f (x : ℝ) : ℝ := 2^(x-2) + x - 6

theorem interval_solution : 
  (f 3 = -1) →
  (f 4 = 2) →
  ∃ x : ℝ, x ∈ set.Ioo 3 4 ∧ f x = 0 :=
by
  intros h₁ h₂
  sorry

end interval_solution_l180_180984


namespace total_colored_pencils_l180_180677

noncomputable def total_pencils (Cheryl Cyrus Madeline : ℕ) : ℕ :=
  Cheryl + Cyrus + Madeline

theorem total_colored_pencils : 
  ∀ (Cheryl Cyrus Madeline : ℕ), Madeline = 63 →
  Madeline * 2 = Cheryl → 
  Cheryl = 3 * Cyrus → 
  total_pencils Cheryl Cyrus Madeline = 231 :=
by 
  intros Cheryl Cyrus Madeline hMadeline hCheryl hCyrus
  rw [hMadeline, hCheryl, hCyrus]
  simp [total_pencils]
  sorry

end total_colored_pencils_l180_180677


namespace arithmetic_sequences_grid_problem_l180_180288

theorem arithmetic_sequences_grid_problem
  (row : Fin 7 → ℚ)
  (col1 col2 : Fin 5 → ℚ)
  (h1 : row 0 = 25)
  (h2 : col1 1 = 16)
  (h3 : col1 2 = 20)
  (h4 : col2 0 = M)
  (h5 : col2 4 = -21)
  (h6 : ∀ i j, row (i+1) - row i = row (i+2) - row (i+1))
  (h7 : ∀ i j, col1 (i+1) - col1 i = col1 (i+2) - col1 (i+1))
  (h8 : ∀ i j, col2 (i+1) - col2 i = col2 (i+2) - col2 (i+1)) :
  M = 1021/12 := 
begin
  sorry
end

end arithmetic_sequences_grid_problem_l180_180288


namespace number_of_divisors_of_n_l180_180377

def n : ℕ := 2^3 * 3^4 * 5^3 * 7^2

theorem number_of_divisors_of_n : ∃ d : ℕ, d = 240 ∧ ∀ k : ℕ, k ∣ n ↔ ∃ a b c d : ℕ, 0 ≤ a ∧ a ≤ 3 ∧ 0 ≤ b ∧ b ≤ 4 ∧ 0 ≤ c ∧ c ≤ 3 ∧ 0 ≤ d ∧ d ≤ 2 := 
sorry

end number_of_divisors_of_n_l180_180377


namespace exponent_of_5_in_30_factorial_l180_180015

theorem exponent_of_5_in_30_factorial : 
  (nat.factors 30!).count 5 = 7 :=
sorry

end exponent_of_5_in_30_factorial_l180_180015


namespace find_principal_amount_l180_180483

-- Definitions of the conditions
def rate_of_interest : ℝ := 0.20
def time_period : ℕ := 2
def interest_difference : ℝ := 144

-- Definitions for Simple Interest (SI) and Compound Interest (CI)
def simple_interest (P : ℝ) : ℝ := P * rate_of_interest * time_period
def compound_interest (P : ℝ) : ℝ := P * (1 + rate_of_interest)^time_period - P

-- Statement to prove the principal amount given the conditions
theorem find_principal_amount (P : ℝ) : 
    compound_interest P - simple_interest P = interest_difference → P = 3600 := by
    sorry

end find_principal_amount_l180_180483


namespace quadratic_trinomial_form_l180_180220

noncomputable def quadratic_form (a b c : ℝ) (h : a ≠ 0) : Prop :=
  ∀ x : ℝ, 
    (a * (3.8 * x - 1)^2 + b * (3.8 * x - 1) + c) = (a * (-3.8 * x)^2 + b * (-3.8 * x) + c)

theorem quadratic_trinomial_form (a b c : ℝ) (h : a ≠ 0) : b = a → quadratic_form a b c h :=
by
  intro hba
  unfold quadratic_form
  intro x
  rw [hba]
  sorry

end quadratic_trinomial_form_l180_180220


namespace difference_max_min_2a_sub_b_l180_180322

theorem difference_max_min_2a_sub_b (a b : ℝ) (h : a^2 + b^2 - 2a - 4 = 0) : 
  let t := 2 * a - b in
  let max_t := 7 in
  let min_t := -3 in
  max_t - min_t = 10 := 
by
  sorry

end difference_max_min_2a_sub_b_l180_180322


namespace exponentiation_equation_l180_180672

theorem exponentiation_equation : 4^2011 * (-0.25)^2010 - 1 = 3 := 
by { sorry }

end exponentiation_equation_l180_180672


namespace fair_game_expected_winnings_l180_180182

theorem fair_game_expected_winnings (num_players : ℕ) (total_pot : ℝ) 
  (p : ℕ → ℝ) (stakes : ℕ → ℝ) :
  num_players = 36 →
  (∀ k, p k = (35 / 36) ^ (k - 1) * p 1) →
  (∀ k, stakes k = total_pot * p k) →
  (∀ k, let L_k := stakes k in total_pot * p k - L_k * p k - L_k + L_k * p k = 0) :=
sorry

end fair_game_expected_winnings_l180_180182


namespace KimSweaterTotal_l180_180436

theorem KimSweaterTotal :
  let monday := 8
  let tuesday := monday + 2
  let wednesday := tuesday - 4
  let thursday := wednesday
  let friday := monday / 2
  monday + tuesday + wednesday + thursday + friday = 34 := by
  sorry

end KimSweaterTotal_l180_180436


namespace exponent_of_five_in_factorial_l180_180057

theorem exponent_of_five_in_factorial:
  (nat.factors 30!).count 5 = 7 :=
begin
  sorry
end

end exponent_of_five_in_factorial_l180_180057


namespace exponent_of_5_in_30_fact_l180_180019

def count_powers_of_5 (n : ℕ) : ℕ :=
  if n < 5 then 0
  else n / 5 + count_powers_of_5 (n / 5)

theorem exponent_of_5_in_30_fact : count_powers_of_5 30 = 7 := 
  by
    sorry

end exponent_of_5_in_30_fact_l180_180019


namespace intersection_point_on_circle_l180_180252

open_locale big_operators

variables {A B C P Q Y : Point}
variables (hAC : diameter (circle A C))
variables (hCB : diameter (circle C B))
variables (hPC : tangent (circle A C) at P)
variables (hQC : tangent (circle C B) at Q)
variables (h_Y_on_common_tangent : ∃ Y, is_common_tangent Y (circle A C) (circle C B) ∧
    lies_on (line AP) Y ∧ lies_on (line BQ) Y)

theorem intersection_point_on_circle :
  ∃ Y, lies_on (circle_on_diameter A B) Y ∧ 
  lies_on (line AP) Y ∧ lies_on (line BQ) Y ∧ 
  is_common_tangent Y (circle A C) (circle C B) :=
begin
  sorry
end

end intersection_point_on_circle_l180_180252


namespace number_of_shifts_worked_l180_180066

theorem number_of_shifts_worked
  (hourly_wage : ℝ)
  (tip_rate : ℝ)
  (orders_per_hour : ℝ)
  (total_earnings : ℝ)
  (hours_per_shift : ℝ)
  (total_shifts : ℕ) :
  hourly_wage = 4 → 
  tip_rate = 0.15 → 
  orders_per_hour = 40 → 
  total_earnings = 240 → 
  hours_per_shift = 8 →
  total_shifts = (240 / ((4 + (0.15 * 40)) * 8)) :=
by
  intros
  sorry

end number_of_shifts_worked_l180_180066


namespace functional_equation_product_zero_l180_180903

theorem functional_equation_product_zero :
  (let f : ℝ → ℝ := sorry in
    let n := sorry in
    let s := sorry in
    (f (x * f y + x) = x * y + f x) → n * s = 0) := sorry

end functional_equation_product_zero_l180_180903


namespace nine_pointed_star_angles_sum_l180_180925

theorem nine_pointed_star_angles_sum :
  let n := 9
  let arc_measure := 360 / n
  let star_points := List.range n
  let fourth_point := 4
  let minor_arc := 3 * arc_measure
  let inscribed_angle := minor_arc / 2
  ∑ i in Finset.range n, inscribed_angle = 540 := by
{
  -- Additional definitions and proof would go here
  sorry
}

end nine_pointed_star_angles_sum_l180_180925


namespace teachers_invitation_l180_180559

theorem teachers_invitation:
  let total_combinations := Nat.choose 10 6 in
  let combinations_with_A_and_B := Nat.choose 8 4 in
  total_combinations - combinations_with_A_and_B = 140 :=
by
  let total_combinations := Nat.choose 10 6
  let combinations_with_A_and_B := Nat.choose 8 4
  have h1 : total_combinations = 210 := by norm_num [Nat.choose]
  have h2 : combinations_with_A_and_B = 70 := by norm_num [Nat.choose]
  have h3 : 210 - 70 = 140 := by norm_num
  exact h3 sorry

end teachers_invitation_l180_180559


namespace total_surface_area_of_prism_l180_180412

/-- Define the parameters of the triangular prism -/
parameters (a b c l h : ℝ) (p : ℝ := (a + b + c) / 2)

/-- The total surface area of the prism -/
theorem total_surface_area_of_prism : 
  let S_total := (2 * l / h) * (Real.sqrt (p * (p - a) * (p - b) * (p - c)) + p * h) in
  S_total = (2 * l / h) * (Real.sqrt (p * (p - a) * (p - b) * (p - c)) + p * h) :=
begin
  sorry
end

end total_surface_area_of_prism_l180_180412


namespace problem_part1_problem_part2_l180_180423

open ProbabilityTheory

noncomputable def total_questions : ℕ := 8
noncomputable def listening_questions : ℕ := 3
noncomputable def written_response_questions : ℕ := 5

-- The probability that Student A draws a listening question and Student B draws a written response question
def prob_A_listening_B_written : ℚ :=
  (listening_questions * written_response_questions) / (total_questions * (total_questions - 1))

-- The probability that at least one of the students draws a listening question
def prob_at_least_one_listening : ℚ :=
  1 - (written_response_questions * (written_response_questions - 1)) / (total_questions * (total_questions - 1))

theorem problem_part1 : prob_A_listening_B_written = 15 / 56 := sorry

theorem problem_part2 : prob_at_least_one_listening = 9 / 14 := sorry

end problem_part1_problem_part2_l180_180423


namespace sodium_bisulfite_moles_combined_l180_180708

def NaHSO3 : Type := ℝ
def HCl : Type := ℝ
def H2O : Type := ℝ
def NaCl : Type := ℝ
def SO2 : Type := ℝ

constant reaction_stoichiometry : NaHSO3 × HCl → NaCl × H2O × SO2

theorem sodium_bisulfite_moles_combined
  (h_moles_hcl : HCl)
  (h_moles_h2o : H2O)
  (h_react : ∀ (n : NaHSO3 × HCl), reaction_stoichiometry n)
  : ∃ (moles_nahso3 : NaHSO3), moles_nahso3 = h_moles_h2o  :=
sorry

end sodium_bisulfite_moles_combined_l180_180708


namespace total_value_of_gold_is_l180_180885

-- Definitions based on the conditions
def legacyBars : ℕ := 5
def aleenaBars : ℕ := legacyBars - 2
def valuePerBar : ℝ := 2200
def totalValue : ℝ := (legacyBars + aleenaBars) * valuePerBar

-- Theorem statement
theorem total_value_of_gold_is :
  totalValue = 17600 := by
  -- We add sorry here to skip the proof
  sorry

end total_value_of_gold_is_l180_180885


namespace pentagon_area_ratio_l180_180458

-- Given definitions
structure Pentagon (point : Type) :=
(A B C D E : point)
(convex : ConvexPolygon Point)
(parallel_AB_CE : Parallel A B C E)
(parallel_BC_AD : Parallel B C A D)
(parallel_AC_DE : Parallel A C D E)
(angle_ABC_100 : angle A B C = 100)
(length_AB : dist A B = 4)
(length_BC : dist B C = 6)
(length_DE : dist D E = 18)

-- The problem statement
theorem pentagon_area_ratio (A B C D E : Point) 
    (h : Pentagon Point) : 
  let m : ℕ := 11 in
  let n : ℕ := 140 in
  m + n = 151 := 
by
  sorry

end pentagon_area_ratio_l180_180458


namespace zero_point_six_six_six_is_fraction_l180_180318

def is_fraction (x : ℝ) : Prop := ∃ (n d : ℤ), d ≠ 0 ∧ x = (n : ℝ) / (d : ℝ)

theorem zero_point_six_six_six_is_fraction:
  let sqrt_2_div_3 := (Real.sqrt 2) / 3
  let neg_sqrt_4 := - Real.sqrt 4
  let zero_point_six_six_six := 0.666
  let one_seventh := 1 / 7
  is_fraction zero_point_six_six_six :=
by sorry

end zero_point_six_six_six_is_fraction_l180_180318


namespace rectangle_perimeter_l180_180121

variables {x y a b : ℝ}

def conditions : Prop :=
  x * y = 2006 ∧
  a * b = 2006 ∧
  (a = (x + y) / 2) ∧
  (b = 4012 / (x + y))

theorem rectangle_perimeter (hx : conditions) : 2 * (x + y) = 8 * Real.sqrt 1003 :=
sorry

end rectangle_perimeter_l180_180121


namespace xiao_ming_error_step_l180_180210

theorem xiao_ming_error_step (x : ℝ) :
  (1 / (x + 1) = (2 * x) / (3 * x + 3) - 1) → 
  3 = 2 * x - (3 * x + 3) → 
  (3 = 2 * x - 3 * x + 3) ↔ false := by
  sorry

end xiao_ming_error_step_l180_180210


namespace g_seven_iterations_l180_180905

theorem g_seven_iterations :
  let g : ℕ → ℕ := λ x, x^2 - 3 * x + 1 in
  g (g (g (g (g (g (g 2)))))) = 3431577212128939 :=
by
  sorry

end g_seven_iterations_l180_180905


namespace no_solution_b_l180_180208

-- Definition of the conditions as functions or properties
def eq_a (x : ℝ) : Prop := (x - 4) ^ 2 = 0
def eq_b (x : ℝ) : Prop := |-5 * x| + 10 = 0
def eq_c (x : ℝ) : Prop := real.sqrt (-x) - 3 = 0
def eq_d (x : ℝ) : Prop := real.sqrt x - 7 = 0
def eq_e (x : ℝ) : Prop := |-5 * x| - 6 = 0

-- Theorem statement
theorem no_solution_b : ¬ ∃ x : ℝ, eq_b x := 
sorry

end no_solution_b_l180_180208


namespace line_equation_of_point_and_slope_angle_l180_180146

theorem line_equation_of_point_and_slope_angle 
  (p : ℝ × ℝ) (θ : ℝ)
  (h₁ : p = (-1, 2))
  (h₂ : θ = 45) :
  ∃ (a b c : ℝ), a * (p.1) + b * (p.2) + c = 0 ∧ (a * 1 + b * 1 = c) :=
sorry

end line_equation_of_point_and_slope_angle_l180_180146


namespace sin_sum_of_zero_points_l180_180741

-- Defines the function f
def f (x m : ℝ) := 2 * sin (2 * x) + cos (2 * x) - m

-- Defines the theorem to be proven
theorem sin_sum_of_zero_points (x1 x2 m : ℝ) (h1 : 0 ≤ x1 ∧ x1 ≤ π / 2) (h2 : 0 ≤ x2 ∧ x2 ≤ π / 2) 
(h3 : f x1 m = 0) (h4 : f x2 m = 0) :
sin (x1 + x2) = (2 * sqrt 5) / 5 := 
sorry

end sin_sum_of_zero_points_l180_180741


namespace Denise_age_l180_180590

-- Define the ages of Amanda, Carlos, Beth, and Denise
variables (A C B D : ℕ)

-- State the given conditions
def condition1 := A = C - 4
def condition2 := C = B + 5
def condition3 := D = B + 2
def condition4 := A = 16

-- The theorem to prove
theorem Denise_age (A C B D : ℕ) (h1 : condition1 A C) (h2 : condition2 C B) (h3 : condition3 D B) (h4 : condition4 A) : D = 17 :=
by
  sorry

end Denise_age_l180_180590


namespace abs_sum_of_binomial_expansion_l180_180751

theorem abs_sum_of_binomial_expansion :
  let a := (λ n, (2 - x)^n) in
  a 6 = 64 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + a 6 * x^6 →
  |a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| = 665 :=
by
  sorry

end abs_sum_of_binomial_expansion_l180_180751


namespace sec_tan_identity_l180_180829

theorem sec_tan_identity (x : ℝ) (h : Real.sec x + Real.tan x = 7 / 3) : Real.sec x - Real.tan x = 3 / 7 := 
by
  sorry

end sec_tan_identity_l180_180829


namespace square_division_l180_180062

theorem square_division (n k : ℕ) (m : ℕ) (h : n * k = m * m) :
  ∃ u v d : ℕ, (gcd u v = 1) ∧ (n = d * u * u) ∧ (k = d * v * v) ∧ (m = d * u * v) :=
by sorry

end square_division_l180_180062


namespace sin_cos_product_l180_180754

def f (x : ℝ) : ℝ := Real.sin x

theorem sin_cos_product (α : ℝ) (hα1 : f (Real.sin α) + f (Real.cos α - 1/2) = 0) 
  (domain_cond : ∀ x, x ∈ set.Icc (-Real.pi/2) (Real.pi/2)) :
  Real.sin α * Real.cos α = -3 / 8 :=
sorry

end sin_cos_product_l180_180754


namespace exponent_of_five_in_factorial_l180_180055

theorem exponent_of_five_in_factorial:
  (nat.factors 30!).count 5 = 7 :=
begin
  sorry
end

end exponent_of_five_in_factorial_l180_180055


namespace exponent_of_5_in_30_fact_l180_180028

def count_powers_of_5 (n : ℕ) : ℕ :=
  if n < 5 then 0
  else n / 5 + count_powers_of_5 (n / 5)

theorem exponent_of_5_in_30_fact : count_powers_of_5 30 = 7 := 
  by
    sorry

end exponent_of_5_in_30_fact_l180_180028


namespace sec_minus_tan_l180_180804

theorem sec_minus_tan (x : ℝ) (h : Real.sec x + Real.tan x = 7 / 3) : Real.sec x - Real.tan x = 3 / 7 :=
by
  sorry

end sec_minus_tan_l180_180804


namespace count_perfect_square_factors_l180_180781

theorem count_perfect_square_factors : 
  let n := (2^10) * (3^12) * (5^15) * (7^7)
  ∃ (count : ℕ), count = 1344 ∧
    (∀ (a b c d : ℕ), 0 ≤ a ∧ a ≤ 10 ∧ 0 ≤ b ∧ b ≤ 12 ∧ 0 ≤ c ∧ c ≤ 15 ∧ 0 ≤ d ∧ d ≤ 7 →
      ((a % 2 = 0) ∧ (b % 2 = 0) ∧ (c % 2 = 0) ∧ (d % 2 = 0) →
        ∃ (k : ℕ), (2^a * 3^b * 5^c * 7^d) = k ∧ k ∣ n)) :=
by
  sorry

end count_perfect_square_factors_l180_180781


namespace rational_terms_not_adjacent_probability_l180_180422

noncomputable def probability_rational_terms_not_adjacent (n : ℕ) (x : ℝ) : ℚ :=
  let k_vals := [0, 4, 8]
  let irrational_count := 9 - k_vals.length
  let total_permutations := Nat.factorial 9
  let invalid_permutations := Nat.factorial irrational_count * Nat.comb (irrational_count + 1) k_vals.length
  invalid_permutations / total_permutations

theorem rational_terms_not_adjacent_probability :
  probability_rational_terms_not_adjacent 8 x = 5 / 12 :=
sorry

end rational_terms_not_adjacent_probability_l180_180422


namespace leading_coefficient_poly_l180_180707

theorem leading_coefficient_poly : 
  (let p := -5 * (X^5 - X^4 + X^3) + 8 * (X^5 - X + 1) - 6 * (3 * X^5 + X^3 + 2)
   in leadingCoeff p) = -15 := 
by 
  sorry

end leading_coefficient_poly_l180_180707


namespace OI_half_BC_l180_180446

-- Definitions of the points and conditions
variables (A B C D O I : Type)
variables [Circle.Center O A B C D]
variables (ACPerpBD : IsLemma (Angle.Perpendicular A C B D))
variables (FootOI : Foot O I A D)

-- The main theorem statement
theorem OI_half_BC {A B C D O I : Type}
  [Circle.Center O A B C D]
  (ACPerpBD : IsLemma (Angle.Perpendicular A C B D))
  (FootOI : Foot O I A D) : 
  OI = BC / 2 :=
sorry

end OI_half_BC_l180_180446


namespace exponent_of_5_in_30_fact_l180_180023

def count_powers_of_5 (n : ℕ) : ℕ :=
  if n < 5 then 0
  else n / 5 + count_powers_of_5 (n / 5)

theorem exponent_of_5_in_30_fact : count_powers_of_5 30 = 7 := 
  by
    sorry

end exponent_of_5_in_30_fact_l180_180023


namespace function_decreasing_on_interval_l180_180758

noncomputable def f (x : ℝ) : ℝ := log (2 + x) + log (2 - x)

theorem function_decreasing_on_interval : ∀ x y : ℝ, (0 < x ∧ x < 2) ∧ (0 < y ∧ y < 2) ∧ x < y → f y < f x :=
by
  sorry

end function_decreasing_on_interval_l180_180758


namespace triangle_XYZ_perimeter_l180_180995

theorem triangle_XYZ_perimeter (PQ PR QR lP lQ lR : ℝ) (hPQ : PQ = 150) (hPR : PR = 210) (hQR : QR = 270) (hlP : lP = 75) (hlQ : lQ = 35) (hlR : lR = 25) : 
  let k := min (lP / QR) (min (lQ / PQ) (lR / PR)) in
  lP = k * QR ∧ lQ = k * PQ ∧ lR = k * PR → 
  (k * QR) + (k * PQ) + (k * PR) = 105 :=
by 
  sorry

end triangle_XYZ_perimeter_l180_180995


namespace max_value_of_f_on_interval_l180_180491

noncomputable def f (x : ℝ) : ℝ := 2 / (x^2 - 1)

theorem max_value_of_f_on_interval :
  (∀ x ∈ set.Icc 2 6, f x ≤ f 2) ∧ (f 2 = 2 / 3) :=
by
  sorry

end max_value_of_f_on_interval_l180_180491


namespace find_b_l180_180701

theorem find_b (b : ℝ) : log b 512 = -5 / 3 → b = 32 ^ (-1.08) :=
by
  sorry

end find_b_l180_180701


namespace evaluate_expression_at_3_l180_180713

theorem evaluate_expression_at_3 :
  (1 / (3 + 1 / (3 + 1 / (3 - 1 / 3)))) = 0.30337078651685395 :=
  sorry

end evaluate_expression_at_3_l180_180713


namespace root_in_interval_l180_180151

def f (x : ℝ) : ℝ := x^3 - 3 * x - 3

theorem root_in_interval : ∃ x ∈ set.Ioo (2 : ℝ) (3 : ℝ), f x = 0 :=
by
  -- The proof will be added here.
  sorry

end root_in_interval_l180_180151


namespace zero_in_interval_3_4_l180_180487

def f (x : ℝ) : ℝ := |x - 2| - Real.log x

theorem zero_in_interval_3_4 : ∃ (x : ℝ), 3 < x ∧ x < 4 ∧ f x = 0 :=
by
  sorry

end zero_in_interval_3_4_l180_180487


namespace sin_double_angle_plus_pi_over_2_l180_180321

def cos_add_pi (theta : ℝ) : Prop := cos (theta + π) = -1/3

theorem sin_double_angle_plus_pi_over_2 
  (h : cos_add_pi θ) : 
  sin (2 * θ + π / 2) = -7 / 9 := 
by
  -- Use the given condition
  have h1 : cos θ = 1 / 3 := sorry
  -- Compute the necessary transformations
  have h2 : cos (2 * θ) = 2 * (cos θ)^2 - 1 := sorry
  -- Substitute cos θ
  have h3 : cos (2 * θ) = 2 * (1/3)^2 - 1 := sorry
  -- Simplify the expression
  have h4 : cos (2 * θ) = 2 / 9 - 1 := sorry
  have h5 : cos (2 * θ) = -7 / 9 := sorry
  -- Use the trigonometric identity to get the final result
  show sin (2 * θ + π / 2) = -7 / 9 from h5

end sin_double_angle_plus_pi_over_2_l180_180321


namespace lateral_surface_area_of_cone_l180_180237

def slant_height : ℝ := 5
def base_radius : ℝ := 3

theorem lateral_surface_area_of_cone : ∃ (A : ℝ), A = π * base_radius * slant_height ∧ A = 15 * π := by
  exists (π * base_radius * slant_height)
  split
  . rfl
  . sorry

end lateral_surface_area_of_cone_l180_180237


namespace probability_girl_selection_l180_180256

-- Define the conditions
def total_candidates : ℕ := 3 + 1
def girl_candidates : ℕ := 1

-- Define the question in terms of probability
def probability_of_selecting_girl (total: ℕ) (girl: ℕ) : ℚ :=
  girl / total

-- Lean statement to prove
theorem probability_girl_selection : probability_of_selecting_girl total_candidates girl_candidates = 1 / 4 :=
by
  sorry

end probability_girl_selection_l180_180256


namespace hyperbola_equation_l180_180490

theorem hyperbola_equation (x y : ℝ) :
  (hyperbola_shares_foci_with_ellipse : ∃ c : ℝ, 4x^2 + y^2 = 1 ∧ is_focus c (0, c) ∧ is_focus c (0, -c)) ∧
  (asymptote : y = \sqrt 2 * x) →
  4 * y^2 - 2 * x^2 = 1 :=
sorry

end hyperbola_equation_l180_180490


namespace find_F_l180_180385

theorem find_F (C F : ℝ) 
  (h1 : C = 7 / 13 * (F - 40))
  (h2 : C = 26) :
  F = 88.2857 :=
by
  sorry

end find_F_l180_180385


namespace rectangle_area_l180_180570

variables (y : ℝ) (length : ℝ) (width : ℝ)

-- Definitions based on conditions
def is_diagonal_y (length width y : ℝ) : Prop :=
  y^2 = length^2 + width^2

def is_length_three_times_width (length width : ℝ) : Prop :=
  length = 3 * width

-- Statement to prove
theorem rectangle_area (y : ℝ) (length width : ℝ)
  (h1 : is_diagonal_y length width y)
  (h2 : is_length_three_times_width length width) :
  length * width = 3 * (y^2 / 10) :=
sorry

end rectangle_area_l180_180570


namespace solve_for_x_l180_180947

-- Define the given equation as a predicate
def equation (x: ℚ) : Prop := (x + 4) / (x - 3) = (x - 2) / (x + 2)

-- State the problem in a Lean theorem
theorem solve_for_x : ∃ x : ℚ, equation x ∧ x = -2 / 11 :=
by
  existsi -2 / 11
  constructor
  repeat { sorry }

end solve_for_x_l180_180947


namespace arithmetic_sequence_l180_180084

-- Definition of S_m as the sum of the first m elements of the sequence a_m
def S (a : ℕ → ℝ) (m : ℕ) : ℝ :=
  ∑ i in finset.range m, a (i + 1)

-- Statement to prove that a sequence (a_m) is arithmetic given the condition
theorem arithmetic_sequence (a : ℕ → ℝ) (h : ∀ (n k : ℕ), n ≠ k → 
    (S a (n + k)) / (n + k) = (S a n - S a k) / (n - k)) : 
    ∃ d : ℝ, ∀ m : ℕ, a m = a 1 + (m - 1) * d := 
begin
  sorry
end

end arithmetic_sequence_l180_180084


namespace fair_game_expected_winnings_l180_180179

theorem fair_game_expected_winnings (num_players : ℕ) (total_pot : ℝ) 
  (p : ℕ → ℝ) (stakes : ℕ → ℝ) :
  num_players = 36 →
  (∀ k, p k = (35 / 36) ^ (k - 1) * p 1) →
  (∀ k, stakes k = total_pot * p k) →
  (∀ k, let L_k := stakes k in total_pot * p k - L_k * p k - L_k + L_k * p k = 0) :=
sorry

end fair_game_expected_winnings_l180_180179


namespace h_k_a_b_sum_eq_14_l180_180893

noncomputable def h_k_a_b_sum : ℝ :=
  let F1 : ℝ × ℝ := (0, 2)
  let F2 : ℝ × ℝ := (6, 2)
  let a : ℝ := 5
  let c : ℝ := 3
  let b : ℝ := Math.sqrt (a^2 - c^2)
  let h : ℝ := (F1.1 + F2.1) / 2
  let k : ℝ := (F1.2 + F2.2) / 2
  h + k + a + b

theorem h_k_a_b_sum_eq_14 : h_k_a_b_sum = 14 := by
  sorry

end h_k_a_b_sum_eq_14_l180_180893


namespace tetrahedron_rhombus_cross_section_l180_180468

theorem tetrahedron_rhombus_cross_section (A B C D P Q R S : Point) (Pi1 Pi2 Pi : Plane)
  (h1 h2 : ℝ) (AB CD : ℝ) :
  tetrahedron A B C D →
  plane_parallel_to_edge Pi1 A B →
  plane_parallel_to_edge Pi2 C D →
  plane_contains_edge Pi1 A B →
  plane_contains_edge Pi2 C D →
  plane_parallel_between Pi Pi1 Pi2 →
  distance_ratio Pi Pi1 Pi2 h1 h2 →
  distance_ratio h1 h2 AB CD →
  plane_intersection_points Pi A C B D P Q R S →
  is_rhombus P Q R S :=
sorry

end tetrahedron_rhombus_cross_section_l180_180468


namespace Arabi_fifth_place_l180_180852

-- Introduce variables for the positions
variable {Farida Lian Marzuq Rafsan Arabi Nabeel : ℕ}

-- Define the conditions
def conditions : Prop :=
  Lian = 7 ∧
  Arabi = Lian - 2 ∧
  Rafsan = Arabi + 3 ∧
  Nabeel = Rafsan - 4 ∧
  Nabeel = Marzuq - 2 ∧
  Farida = Marzuq + 1

-- Define the theorem to be proven
theorem Arabi_fifth_place (h : conditions) : Arabi = 5 :=
  by sorry

end Arabi_fifth_place_l180_180852


namespace cherries_count_l180_180242

theorem cherries_count (b s r c : ℝ) 
  (h1 : b + s + r + c = 360)
  (h2 : s = 2 * b)
  (h3 : r = 4 * s)
  (h4 : c = 2 * r) : 
  c = 640 / 3 :=
by 
  sorry

end cherries_count_l180_180242


namespace john_spends_on_pins_l180_180068

theorem john_spends_on_pins :
  let original_price := 20
  let discount := 0.15
  let num_pins := 10
  (num_pins * (original_price - discount * original_price)) = 170 :=
by
  let original_price := 20
  let discount := 0.15
  let num_pins := 10
  let discounted_price := original_price - discount * original_price
  let total_cost := num_pins * discounted_price
  have h: total_cost = 170 := by sorry
  exact h

end john_spends_on_pins_l180_180068


namespace exponent_of_5_in_30_factorial_l180_180014

theorem exponent_of_5_in_30_factorial : 
  (nat.factors 30!).count 5 = 7 :=
sorry

end exponent_of_5_in_30_factorial_l180_180014


namespace train_length_l180_180582

noncomputable def speed_kph := 56  -- speed in km/hr
def time_crossing := 9  -- time in seconds
noncomputable def speed_mps := speed_kph * 1000 / 3600  -- converting km/hr to m/s

theorem train_length : speed_mps * time_crossing = 140 := by
  -- conversion and result approximation
  sorry

end train_length_l180_180582


namespace remaining_volume_correct_l180_180265

def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * π * r^3
def volume_of_cylinder (r h : ℝ) : ℝ := π * r^2 * h

variables (r_sphere : ℝ) (r_cyl1 r_cyl2 : ℝ) (h : ℝ)

noncomputable def remaining_volume_sphere : ℝ :=
  volume_of_sphere r_sphere - 2 * volume_of_cylinder r_cyl1 h - volume_of_cylinder r_cyl2 h

theorem remaining_volume_correct :
  remaining_volume_sphere 10 (1.5 / 2) 1 (4) = ((3956 : ℝ) / 3) * π := 
by simp [volume_of_sphere, volume_of_cylinder]; sorry

end remaining_volume_correct_l180_180265


namespace number_of_non_divisible_g_l180_180904

def is_proper_divisor (d n : ℕ) : Prop :=
  d ∣ n ∧ d < n

def g (n : ℕ) : ℕ :=
  ∏ d in (Finset.filter (λ d, is_proper_divisor d n) (Finset.range n)), d

def n_does_not_divide_g (n : ℕ) : Prop :=
  ¬ (n ∣ g n)

theorem number_of_non_divisible_g :
  (Finset.filter (λ n, n_does_not_divide_g n) (Finset.Icc 2 100)).card = 29 :=
by
  sorry

end number_of_non_divisible_g_l180_180904


namespace pesticide_washing_properties_l180_180554

noncomputable def f : ℝ → ℝ := λ x, 1 / (1 + x^2)

theorem pesticide_washing_properties :
  f 0 = 1 ∧
  f 1 = 1 / 2 ∧
  (∀ x y : ℝ, 0 ≤ x ∧ x ≤ y → f x ≥ f y) ∧
  (∀ x : ℝ, 0 < f x ∧ f x ≤ 1) ∧
  (∀ a : ℝ, 0 < a →
    (a > 2 * Real.sqrt 2 →
      f a < f (a / 2) ^ 2) ∧
    (a ≤ 2 * Real.sqrt 2 →
      f a ≥ f (a / 2) ^ 2)) :=
by
  sorry

end pesticide_washing_properties_l180_180554


namespace hyperbola_equation_l180_180390

theorem hyperbola_equation (a b : ℝ) (λ : ℝ) (focus : ℝ × ℝ)
    (asymp : ∀ x : ℝ, y = ± 3 * x) (focus_eq : focus = (sqrt 10, 0)):
  x^2 - (y^2 / 9) = 1 :=
by
  -- Conditions from the problem
  have asymptotes : ∀ x, y = ± 3 * x := asymp
  have focus_is : focus = (sqrt 10, 0) := focus_eq
  sorry

end hyperbola_equation_l180_180390


namespace quadratic_cos2x_equation_specific_quadratic_cos2x_eqn_l180_180445

-- Define the question and conditions as a Lean statement
theorem quadratic_cos2x_equation (a b c : ℝ) :
  let t := 2*b^2 - 4*a*c - a^2 in
  let u := 4*c^2 - 2*b^2 + 4*a*c + a^2 in
  (a ≠ 0) →
  ∃ (x : ℝ), a * cos x ^ 2 + b * cos x + c = 0 →
  (a^2 * cos (2 * x) ^ 2 + t * cos (2 * x) + u = 0) :=
by
  sorry

-- Specific case for a = 4, b = 2, c = -1
theorem specific_quadratic_cos2x_eqn :
  (let a := 4
   let b := 2
   let c := -1 in
   let t := 2 * b^2 - 4 * a * c - a^2 in
   let u := 4 * c^2 - 2 * b^2 + 4 * a * c + a^2 in
   a^2 * cos (2 * x) ^ 2 + t * cos (2 * x) + u) = 
   (16 * cos (2 * x) ^ 2 + 8 * cos (2 * x) - 4) :=
by
  sorry

end quadratic_cos2x_equation_specific_quadratic_cos2x_eqn_l180_180445


namespace find_angle_between_vectors_l180_180345

variables {a b : ℝ → ℝ} -- Assume vector functions a and b
variable  θ : ℝ -- θ is the angle we are looking for

axiom norm_a : ‖a‖ = 6 * real.sqrt 3
axiom norm_b : ‖b‖ = 1 / 3
axiom dot_product_ab : (λ x, a x * b x) = -3

theorem find_angle_between_vectors : θ = 5 * real.pi / 6 := 
by
  sorry

end find_angle_between_vectors_l180_180345


namespace infinite_ns_dividing_factorial_l180_180469

-- Conditions
def infinite_solutions_pells_equation := ∀ x:Nrat, ∀ y:Nrat, x^2 - 5 * y^2 = -1
def exists_y_for_n (n:ℕ) : Prop := ∃ y:ℕ, n^2 + 1 = 5 * y^2

-- The goal statement using these conditions
theorem infinite_ns_dividing_factorial :
  (∀ k:ℕ, ∃ n:ℕ, n > k ∧ exists_y_for_n n ∧ (n^2 + 1) ∣ n!) :=
by
  -- Skipping proof. This is a placeholder to ensure the statement is complete.
  sorry

end infinite_ns_dividing_factorial_l180_180469


namespace fifteenth_term_l180_180728

variable (a b : ℤ)

def sum_first_n_terms (n : ℕ) : ℤ := n * (2 * a + (n - 1) * b) / 2

axiom sum_first_10 : sum_first_n_terms 10 = 60
axiom sum_first_20 : sum_first_n_terms 20 = 320

def nth_term (n : ℕ) : ℤ := a + (n - 1) * b

theorem fifteenth_term : nth_term 15 = 25 :=
by
  sorry

end fifteenth_term_l180_180728


namespace min_gennadys_l180_180608

-- Defining the basic constants for each name type
def Alexanders : Nat := 45
def Borises : Nat := 122
def Vasilies : Nat := 27

-- Define the proof statement to check the minimum number of Gennadys needed
theorem min_gennadys (a b v : Nat) (no_adjacent: a = 45 ∧ b = 122 ∧ v = 27) : ∃ g : Nat, g = 49 :=
by
  -- Using provided conditions
  cases no_adjacent with h_a h_bv
  cases h_bv with h_b h_v
  -- Correct answer derived from the solution
  use 49
  -- skipping proof details 
  sorry

end min_gennadys_l180_180608


namespace equal_diagonals_of_convex_quadrilateral_l180_180403

theorem equal_diagonals_of_convex_quadrilateral 
  {A B C D M N : Type} [ConvexQuadrilateral A B C D]
  (mid_AB : midpoint M A B)
  (mid_CD : midpoint N C D)
  (equal_angles : ∀ (l : Line), l.pass_through M N ∧ 
    angle_with_diagonals l A C = angle_with_diagonals l B D) :
  distance A C = distance B D :=
sorry

end equal_diagonals_of_convex_quadrilateral_l180_180403


namespace company_C_more_than_A_l180_180563

theorem company_C_more_than_A (A B C D: ℕ) (hA: A = 30) (hB: B = 2 * A)
    (hC: C = A + 10) (hD: D = C - 5) (total: A + B + C + D = 165) : C - A = 10 := 
by 
  sorry

end company_C_more_than_A_l180_180563


namespace correct_answers_diff_l180_180922

variable (n : ℕ) (p : ℚ) (m : ℕ)
variable (Hn : n = 120) (Hp : p = 0.25) (Hm : m = 17)

theorem correct_answers_diff (Hn : n = 120) (Hp : p = 0.25) (Hm : m = 17) :
  let lyssa_correct := (1 - p) * n
  let precious_correct := n - m
  precious_correct - lyssa_correct = 13 :=
by
  unfold lyssa_correct precious_correct
  rw [Hn, Hp, Hm]
  sorry

end correct_answers_diff_l180_180922


namespace min_gennadys_needed_l180_180657

variables (A B V G : ℕ)

theorem min_gennadys_needed
  (hA : A = 45)
  (hB : B = 122)
  (hV : V = 27)
  (hG : ∀ i, i < 121 → A + V < 121 → G ≥ 49) :
  G = 49 :=
sorry

end min_gennadys_needed_l180_180657


namespace daniel_initial_noodles_l180_180223

variable (give : ℕ)
variable (left : ℕ)
variable (initial : ℕ)

theorem daniel_initial_noodles (h1 : give = 12) (h2 : left = 54) (h3 : initial = left + give) : initial = 66 := by
  sorry

end daniel_initial_noodles_l180_180223


namespace divisor_between_l180_180569

theorem divisor_between (n a b : ℕ) (h_n_gt_8 : n > 8) (h_a_dvd_n : a ∣ n) (h_b_dvd_n : b ∣ n) 
    (h_a_lt_b : a < b) (h_n_eq_asq_plus_b : n = a^2 + b) (h_a_ne_b : a ≠ b) :
  ∃ d : ℕ, d ∣ n ∧ a < d ∧ d < b :=
sorry

end divisor_between_l180_180569


namespace minimize_unrolled_area_l180_180562

theorem minimize_unrolled_area (x : ℝ) (h1 : x ≥ 0) (h2 : x ≤ 1) : 
  ∃ x_min : ℝ, x_min = 4 / (20 - real.pi) ∧ 
  ∀ y : ℝ, y ≥ 0 ∧ y ≤ 1 → (S y ≤ S x_min) :=
sorry

end minimize_unrolled_area_l180_180562


namespace average_income_B_and_C_l180_180479

variables (A_income B_income C_income : ℝ)

noncomputable def average_monthly_income_B_and_C (A_income : ℝ) :=
  (B_income + C_income) / 2

theorem average_income_B_and_C
  (h1 : (A_income + B_income) / 2 = 5050)
  (h2 : (A_income + C_income) / 2 = 5200)
  (h3 : A_income = 4000) :
  average_monthly_income_B_and_C 4000 = 6250 :=
by
  sorry

end average_income_B_and_C_l180_180479


namespace total_number_of_pages_l180_180782

variable (x : ℕ)

-- Conditions
def first_day_remaining : ℕ := x - (x / 6 + 10)
def second_day_remaining : ℕ := first_day_remaining x - (first_day_remaining x / 5 + 20)
def third_day_remaining : ℕ := second_day_remaining x - (second_day_remaining x / 4 + 25)
def final_remaining : Prop := third_day_remaining x = 100

-- Theorem statement
theorem total_number_of_pages : final_remaining x → x = 298 :=
by
  intros h
  sorry

end total_number_of_pages_l180_180782


namespace teacher_total_score_l180_180263

/-- Conditions -/
def written_test_score : ℝ := 80
def interview_score : ℝ := 60
def written_test_weight : ℝ := 0.6
def interview_weight : ℝ := 0.4

/-- Prove the total score -/
theorem teacher_total_score :
  written_test_score * written_test_weight + interview_score * interview_weight = 72 :=
by
  sorry

end teacher_total_score_l180_180263


namespace minimum_gennadies_l180_180611

theorem minimum_gennadies (A B V G : ℕ) (hA : A = 45) (hB : B = 122) (hV : V = 27) (hGap : G + A + V >= B - 1) :
  G >= 49 :=
by 
  have := by linarith [hGap, hA, hB, hV]
  exact this

end minimum_gennadies_l180_180611


namespace quadratic_factorization_l180_180964

theorem quadratic_factorization :
  ∃ a b : ℕ, (a > b) ∧ (x^2 - 20 * x + 96 = (x - a) * (x - b)) ∧ (4 * b - a = 20) := sorry

end quadratic_factorization_l180_180964


namespace probability_kyle_catherine_not_david_l180_180881

/--
Kyle, David, and Catherine each try independently to solve a problem. 
Their individual probabilities for success are 1/3, 2/7, and 5/9.
Prove that the probability that Kyle and Catherine, but not David, will solve the problem is 25/189.
-/
theorem probability_kyle_catherine_not_david :
  let P_K := 1 / 3
  let P_D := 2 / 7
  let P_C := 5 / 9
  let P_D_c := 1 - P_D
  P_K * P_C * P_D_c = 25 / 189 :=
by
  sorry

end probability_kyle_catherine_not_david_l180_180881


namespace exponent_of_five_in_factorial_l180_180046

theorem exponent_of_five_in_factorial:
  (nat.factors 30!).count 5 = 7 :=
begin
  sorry
end

end exponent_of_five_in_factorial_l180_180046


namespace find_c_l180_180682

theorem find_c (a b d : ℝ) (c : ℝ) : 
  a = -3 → 
  b = 3 → 
  d = 3 → 
  (∀ x, -3 * real.cos (3 * x) + c * real.sin (3 * x) ≤ 5) → 
  c = 4 ∨ c = -4 :=
by
  sorry

end find_c_l180_180682


namespace fair_game_stakes_ratio_l180_180185

theorem fair_game_stakes_ratio (n : ℕ) (deck_size : ℕ) (player_count : ℕ)
  (L : ℕ → ℝ) : 
  deck_size = 36 → player_count = 36 → 
  (∀ k : ℕ, k < player_count - 1 → 
    (L (k + 1)) / (L k) = 35 / 36) :=
by
  intros h_deck_size h_player_count k hk
  simp [h_deck_size, h_player_count, hk]
  sorry

end fair_game_stakes_ratio_l180_180185


namespace min_gennadys_needed_l180_180652

variables (A B V G : ℕ)

theorem min_gennadys_needed
  (hA : A = 45)
  (hB : B = 122)
  (hV : V = 27)
  (hG : ∀ i, i < 121 → A + V < 121 → G ≥ 49) :
  G = 49 :=
sorry

end min_gennadys_needed_l180_180652


namespace min_gennadies_l180_180631

theorem min_gennadies 
  (n_Alexanders : ℕ) (n_Borises : ℕ) (n_Vasilies : ℕ) 
  (x_Gennadies : ℕ) 
  (h_Alexanders : n_Alexanders = 45) 
  (h_Borises   : n_Borises = 122) 
  (h_Vasilies  : n_Vasilies = 27) 
  (h_condition : ∀ p : ℕ, p = n_Borises - 1 → p = 121) 
  (h_total     : ∀ q : ℕ, q = 45 + 27 → q = 72)
  : x_Gennadies = 49 := 
sorry

end min_gennadies_l180_180631


namespace sec_tan_eq_l180_180835

theorem sec_tan_eq (x : ℝ) (h : Real.cos x ≠ 0) : 
  Real.sec x + Real.tan x = 7 / 3 → Real.sec x - Real.tan x = 3 / 7 :=
by
  intro h1
  sorry

end sec_tan_eq_l180_180835


namespace cube_labelings_possible_l180_180238

theorem cube_labelings_possible :
  ∃ (labeling : (Fin 12 → Fin 2)), 
  (∀ face_edges : Fin 6, (finset.sum (finset.filter (λ edge, edge ∈ face_edges) univ.card)) = 3) ∧ 
  (∀ adjacent_faces : (Fin 6 × Fin 6), 
    finset.sum (finset.filter (λ edge, edge ∈ (adjacent_faces.1 ∪ adjacent_faces.2)) univ.card) ≤ 4) → 
  finset.card (set_of λ labeling, ∀ face_edges, (finset.sum (finset.filter (λ edge, edge ∈ face_edges) univ.card)) = 3 ∧ 
    ∀ adjacent_faces, finset.sum (finset.filter (λ edge, edge ∈ (adjacent_faces.1 ∪ adjacent_faces.2)) univ.card) ≤ 4 ) = 16 :=
begin
  sorry
end

end cube_labelings_possible_l180_180238


namespace triangle_sine_half_angle_triangle_tangent_half_angle_triangle_cosine_half_angle_l180_180896

variables {α β γ r R p : ℝ}
variables {α β γ : ℝ} -- angles of the triangle
variables {r R p : ℝ} -- r -> inscribed circle radius, R -> circumcircle radius, p -> semiperimeter

-- Given that α, β, and γ are the angles of a triangle, prove the following statements:
theorem triangle_sine_half_angle (hα : 0 < α ∧ α < π) (hβ : 0 < β ∧ β < π) (hγ : 0 < γ ∧ γ < π) (sum_angles : α + β + γ = π) :
  sin (α / 2) * sin (β / 2) * sin (γ / 2) = r / (4 * R) := sorry

theorem triangle_tangent_half_angle (hα : 0 < α ∧ α < π) (hβ : 0 < β ∧ β < π) (hγ : 0 < γ ∧ γ < π) (sum_angles : α + β + γ = π) :
  tan (α / 2) * tan (β / 2) * tan (γ / 2) = r / p := sorry

theorem triangle_cosine_half_angle (hα : 0 < α ∧ α < π) (hβ : 0 < β ∧ β < π) (hγ : 0 < γ ∧ γ < π) (sum_angles : α + β + γ = π) :
  cos (α / 2) * cos (β / 2) * cos (γ / 2) = p / (4 * R) := sorry

end triangle_sine_half_angle_triangle_tangent_half_angle_triangle_cosine_half_angle_l180_180896


namespace total_whales_seen_is_178_l180_180872

/-
Ishmael's monitoring of whales yields the following:
- On the first trip, he counts 28 male whales and twice as many female whales.
- On the second trip, he sees 8 baby whales, each traveling with their parents.
- On the third trip, he counts half as many male whales as the first trip and the same number of female whales as on the first trip.
-/

def number_of_whales_first_trip : ℕ := 28
def number_of_female_whales_first_trip : ℕ := 2 * number_of_whales_first_trip
def total_whales_first_trip : ℕ := number_of_whales_first_trip + number_of_female_whales_first_trip

def number_of_baby_whales_second_trip : ℕ := 8
def total_whales_second_trip : ℕ := number_of_baby_whales_second_trip * 3

def number_of_male_whales_third_trip : ℕ := number_of_whales_first_trip / 2
def number_of_female_whales_third_trip : ℕ := number_of_female_whales_first_trip
def total_whales_third_trip : ℕ := number_of_male_whales_third_trip + number_of_female_whales_third_trip

def total_whales_seen : ℕ := total_whales_first_trip + total_whales_second_trip + total_whales_third_trip

theorem total_whales_seen_is_178 : total_whales_seen = 178 :=
by
  -- skip the actual proof
  sorry

end total_whales_seen_is_178_l180_180872


namespace quadratic_factorization_l180_180965

theorem quadratic_factorization :
  ∃ a b : ℕ, (a > b) ∧ (x^2 - 20 * x + 96 = (x - a) * (x - b)) ∧ (4 * b - a = 20) := sorry

end quadratic_factorization_l180_180965


namespace verify_equation_l180_180999

theorem verify_equation : (3^2 + 5^2)^2 = 16^2 + 30^2 := by
  sorry

end verify_equation_l180_180999


namespace ratio_of_b_to_a_is_4_l180_180846

theorem ratio_of_b_to_a_is_4 (b a : ℚ) (h1 : b = 4 * a) (h2 : b = 15 - 4 * a) : a = 15 / 8 := by
  sorry

end ratio_of_b_to_a_is_4_l180_180846


namespace kim_knit_sweaters_total_l180_180439

theorem kim_knit_sweaters_total :
  ∀ (M T W R F : ℕ), 
    M = 8 →
    T = M + 2 →
    W = T - 4 →
    R = T - 4 →
    F = M / 2 →
    M + T + W + R + F = 34 :=
by
  intros M T W R F hM hT hW hR hF
  rw [hM, hT, hW, hR, hF]
  norm_num
  sorry

end kim_knit_sweaters_total_l180_180439


namespace gino_white_bears_l180_180320

theorem gino_white_bears
  (brown_bears : ℕ)
  (black_bears : ℕ)
  (total_bears : ℕ) :
  brown_bears = 15 →
  black_bears = 27 →
  total_bears = 66 →
  total_bears - (brown_bears + black_bears) = 24 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact rfl

end gino_white_bears_l180_180320


namespace area_sum_of_triangles_l180_180135

theorem area_sum_of_triangles 
  (A B C D E : Type*)
  [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] 
  [has_area A] [has_area B] [has_area C] [has_area D] [has_area E]
  (h_parallelogram : parallelogram A B C D) 
  (h_opposite_sides : opposite_sides A B D C)
  (h_point_inside : point_inside_parallelogram E A B C D) :
  area (triangle A E B) + area (triangle D E C) = area (triangle A E D) + area (triangle B E C) := 
sorry

end area_sum_of_triangles_l180_180135


namespace farmer_flax_acres_l180_180565

-- Definitions based on conditions
def total_acres : ℕ := 240
def extra_sunflower_acres : ℕ := 80

-- Problem statement
theorem farmer_flax_acres (F : ℕ) (S : ℕ) 
    (h1 : F + S = total_acres) 
    (h2 : S = F + extra_sunflower_acres) : 
    F = 80 :=
by
    -- Proof goes here
    sorry

end farmer_flax_acres_l180_180565


namespace similarity_of_triangles_l180_180207

theorem similarity_of_triangles (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] 
  (a1 a2 : A) (b1 b2 : B) (c1 c2 : C) (d1 d2 : D)
  (h1 : dist a1 a2 / dist b1 b2 = dist c1 c2 / dist d1 d2)
  (h2 : dist a1 a2 / dist b1 b2 = dist a2 a1 / dist b2 b1) :
  ¬ (dist a1 a2 / dist b1 b2 ≠ dist a2 a1 / dist b2 b1) :=
begin
  assume h,
  contradiction,
end

end similarity_of_triangles_l180_180207


namespace mirror_country_two_transfers_l180_180990

variables {City : Type} (OrdinaryCountry MirrorCountry : Type) 
  [Fintype OrdinaryCountry] [Fintype MirrorCountry]

-- Mapping city in ordinary country to corresponding city in mirror country
def Mirror (c : OrdinaryCountry) : MirrorCountry := sorry

-- Railway connections in ordinary country
def Connected (c1 c2 : OrdinaryCountry) : Prop := sorry

-- Railway connections in mirror country
def MirrorConnected (c1 c2 : MirrorCountry) : Prop :=
  ¬ Connected (Mirror c1) (Mirror c2)

-- Condition 1: There are two countries: an ordinary country and a mirror country
-- Provided by types checking Fintype == Country

-- Condition 2: For each city in the ordinary country, there is a corresponding city in the mirror country, and vice versa
-- Provided by function Mirror

-- Condition 3: Rail connections in ordinary country and mirror country are converse.
def railConverse : Prop :=
  ∀ c1 c2 : OrdinaryCountry, Connected c1 c2 ↔ ¬ MirrorConnected (Mirror c1) (Mirror c2)

-- Condition 4: In the ordinary country, Alisa needs at least two transfers to travel from city \( A \) to city \( B \)
def at_least_two_transfers (A B : OrdinaryCountry) : Prop :=
  ¬ Connected A B ∧ ∀ C : OrdinaryCountry,
    (Connected A C ∧ Connected C B) → False

theorem mirror_country_two_transfers (OrdinaryCountry MirrorCountry : Type)
  [Fintype OrdinaryCountry] [Fintype MirrorCountry]
  (Mirror : OrdinaryCountry → MirrorCountry)
  (Connected : OrdinaryCountry → OrdinaryCountry → Prop)
  (A B : OrdinaryCountry)
  (H3 : ∀ c1 c2 : OrdinaryCountry, Connected c1 c2 ↔ ¬ MirrorConnected (Mirror c1) (Mirror c2))
  (H4 : at_least_two_transfers A B) :
  ∀ (C' D' : MirrorCountry), C' ≠ D' → 
  ∃ t₁ t₂ t₃ : MirrorCountry, MirrorConnected C' t₁ ∧ MirrorConnected t₁ t₂ ∧ MirrorConnected t₂ t₃ ∧ MirrorConnected t₃ D' ∧ t₁ ≠ t₃ :=
sorry

end mirror_country_two_transfers_l180_180990


namespace complex_number_solution_l180_180727

-- Define the conditions
def z : ℂ := 1 + I

theorem complex_number_solution (z : ℂ) (h : z * (1 - I) = 2) : z = 1 + I :=
by sorry

end complex_number_solution_l180_180727


namespace fair_game_condition_l180_180195

variables (n : ℕ) (L : ℝ) {p : ℕ → ℝ}

-- Define the probability p_k for the k-th player.
def probability (k : ℕ) : ℝ := (35.0 / 36.0) ^ k

-- Define the expected value of the k-th player.
def expected_value (L : ℝ) (Lk : ℝ) (k : ℕ) : ℝ := L * probability k - Lk

-- Define the conditions of the fair game.
def fair_game := ∀ k, expected_value L (L * probability k) k = 0

-- Main theorem stating that the game is fair if stakes decrease proportionally by a factor of 35/36.
theorem fair_game_condition (k : ℕ) (L : ℝ) :
  fair_game :=
by
  sorry

end fair_game_condition_l180_180195


namespace negation_of_forall_x_squared_nonnegative_l180_180158

theorem negation_of_forall_x_squared_nonnegative :
  ¬ (∀ x : ℝ, x^2 ≥ 0) ↔ ∃ x : ℝ, x^2 < 0 :=
sorry

end negation_of_forall_x_squared_nonnegative_l180_180158


namespace find_A_l180_180513

theorem find_A (A : ℝ) : 
    (∀ (side_length : ℝ) (increase_horizontal : ℝ) (area_new_rectangle : ℝ),
    side_length = 12 ∧ increase_horizontal = 3 ∧ area_new_rectangle = 120 →
    (area_new_rectangle = (side_length + increase_horizontal) * (side_length - A))) →
    A = 4 :=
by
    intro h
    have side_length := 12
    have increase_horizontal := 3
    have area_new_rectangle := 120
    specialize h side_length increase_horizontal area_new_rectangle
    have h1 : 12 = side_length := rfl
    have h2 : 3 = increase_horizontal := rfl
    have h3 : 120 = area_new_rectangle := rfl
    rw [h1, h2, h3] at h
    exact sorry

end find_A_l180_180513


namespace min_gennadys_l180_180603

-- Defining the basic constants for each name type
def Alexanders : Nat := 45
def Borises : Nat := 122
def Vasilies : Nat := 27

-- Define the proof statement to check the minimum number of Gennadys needed
theorem min_gennadys (a b v : Nat) (no_adjacent: a = 45 ∧ b = 122 ∧ v = 27) : ∃ g : Nat, g = 49 :=
by
  -- Using provided conditions
  cases no_adjacent with h_a h_bv
  cases h_bv with h_b h_v
  -- Correct answer derived from the solution
  use 49
  -- skipping proof details 
  sorry

end min_gennadys_l180_180603


namespace hyperbola_eccentricity_l180_180387

theorem hyperbola_eccentricity
  (a b c : ℝ)
  (ha : a ≠ 0)
  (hb : b = -4 * a / 3)
  (hc : c = (Real.sqrt (a ^ 2 + b ^ 2)))
  (point_on_asymptote : ∃ x y : ℝ, x = 3 ∧ y = -4 ∧ (y = b / a * x ∨ y = -b / a * x)) :
  (c / a) = 5 / 3 :=
by
  sorry

end hyperbola_eccentricity_l180_180387


namespace necessary_not_sufficient_l180_180224

def line_tangent_to_parabola (l C : Set Point) : Prop :=
  ∃! p : Point, p ∈ l ∧ p ∈ C

def line_one_point_common (l C : Set Point) : Prop :=
  ∃! p : Point, p ∈ l ∧ p ∈ C

theorem necessary_not_sufficient (l C : Set Point) :
  (line_one_point_common l C → line_tangent_to_parabola l C)
  ∧ ¬(line_one_point_common l C ← line_tangent_to_parabola l C) :=
by
  sorry

end necessary_not_sufficient_l180_180224


namespace sec_minus_tan_l180_180813

-- Define the problem in Lean 4
theorem sec_minus_tan (x : ℝ) (h : Real.sec x + Real.tan x = 7 / 3) : Real.sec x - Real.tan x = 3 / 7 := by
  -- One could also include here the necessary mathematical facts and connections.
  sorry -- Proof to be provided

end sec_minus_tan_l180_180813


namespace pentagon_diagonal_sum_l180_180447

theorem pentagon_diagonal_sum (FG HI GH IJ FJ : ℝ) (hFG : FG = 4) (hHI : HI = 4)
  (hGH : GH = 11) (hIJ : IJ = 11) (hFJ : FJ = 15) : 
  ∑ (diagonal_length : ℝ in {d | is_diagonal FGHIJ d}, diagonal_length) = 1021 / 44 :=
by
  sorry

end pentagon_diagonal_sum_l180_180447


namespace solveEquation1_proof_solveEquation2_proof_l180_180949

noncomputable def solveEquation1 : Set ℝ :=
  { x | 2 * x^2 - 5 * x = 0 }

theorem solveEquation1_proof :
  solveEquation1 = { 0, (5 / 2 : ℝ) } :=
by
  sorry

noncomputable def solveEquation2 : Set ℝ :=
  { x | x^2 + 3 * x - 3 = 0 }

theorem solveEquation2_proof :
  solveEquation2 = { ( (-3 + Real.sqrt 21) / 2 : ℝ ), ( (-3 - Real.sqrt 21) / 2 : ℝ ) } :=
by
  sorry

end solveEquation1_proof_solveEquation2_proof_l180_180949


namespace different_color_socks_l180_180380

theorem different_color_socks :
  let black_socks := 5
      white_socks := 6
      blue_socks := 7 in
  (black_socks * white_socks) + (black_socks * blue_socks) + (white_socks * blue_socks) = 107 :=
by
  let black_socks := 5
  let white_socks := 6
  let blue_socks := 7
  have h1 : black_socks * white_socks = 30 := rfl
  have h2 : black_socks * blue_socks = 35 := rfl
  have h3 : white_socks * blue_socks = 42 := rfl
  have h_sum := h1 + h2 + h3
  show 107, by rw [h_sum]; rfl


end different_color_socks_l180_180380


namespace exponent_of_5_in_30_factorial_l180_180000

theorem exponent_of_5_in_30_factorial : Nat.factorial 30 ≠ 0 → (nat.factorization (30!)).coeff 5 = 7 :=
by
  sorry

end exponent_of_5_in_30_factorial_l180_180000


namespace coefficient_of_x3_in_expansion_l180_180391

noncomputable def C (n k : ℕ) : ℚ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def constant_term_condition (n : ℕ) : Prop :=
  ∃ r : ℕ, 3 * r = n ∧ C n r * (-1 : ℚ)^r = 15

def coefficient_x3_condition (n : ℕ) (coeff : ℚ) : Prop :=
  ∃ r : ℕ, 3 * r - n = 3 ∧ coeff = -C n r

theorem coefficient_of_x3_in_expansion (n : ℕ) (coeff : ℚ) :
  constant_term_condition n → (n = 6) → coefficient_x3_condition n coeff → coeff = -20 :=
begin
  sorry,
end

end coefficient_of_x3_in_expansion_l180_180391


namespace sigma_is_multiplicative_tau_is_multiplicative_phi_is_multiplicative_mu_is_multiplicative_l180_180247

noncomputable def is_multiplicative_number_theoretic_function (f : ℕ → ℕ) :=
  ∀ (m n : ℕ), Nat.coprime m n → f (m * n) = f m * f n

noncomputable def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ d in Nat.divisors n, d

noncomputable def number_of_divisors (n : ℕ) : ℕ :=
  Nat.divisors n.card

noncomputable def euler_totient_function (n : ℕ) : ℕ :=
  ∑ k in Finset.filter (λ k, Nat.coprime k n) (Finset.range (n + 1)), 1

noncomputable def mobius_function (n : ℕ) : ℤ :=
  if ∃ p, p ^ 2 ∣ n then 0
  else if Nat.card (Nat.factors n) % 2 = 0 then 1
  else - 1

#check is_multiplicative_number_theoretic_function sum_of_divisors
#check is_multiplicative_number_theoretic_function number_of_divisors
#check is_multiplicative_number_theoretic_function euler_totient_function
#check is_multiplicative_number_theoretic_function mobius_function

theorem sigma_is_multiplicative : is_multiplicative_number_theoretic_function sum_of_divisors :=
  sorry

theorem tau_is_multiplicative : is_multiplicative_number_theoretic_function number_of_divisors :=
  sorry

theorem phi_is_multiplicative : is_multiplicative_number_theoretic_function euler_totient_function :=
  sorry

theorem mu_is_multiplicative : is_multiplicative_number_theoretic_function mobius_function :=
  sorry

end sigma_is_multiplicative_tau_is_multiplicative_phi_is_multiplicative_mu_is_multiplicative_l180_180247


namespace sum_first_n_terms_general_formula_l180_180330

-- Definition of the sequence
def sequence (n : ℕ) (a : ℕ → ℤ) (k : ℤ) : Prop :=
  ∀n > 0, 2 * a (n + 1) = a n + a (n + 2) + k

-- Initial conditions
def initial_conditions (a : ℕ → ℤ) : Prop :=
  a 1 = 2 ∧ a 3 + a 5 = -4

-- Sum of first n terms when k = 0
theorem sum_first_n_terms (a : ℕ → ℤ) (n k : ℤ) (h : initial_conditions a) (hk : k = 0) :
  ∑ i in finset.range n, a i = (-2 * n^2 + 8 * n) / 3 := sorry

-- General formula for the sequence when a_4 = -1
theorem general_formula (a : ℕ → ℤ) (h1 : initial_conditions a) (h2 : a 4 = -1) :
  ∀n, a n = -n^2 + 4 * n - 1 := sorry

end sum_first_n_terms_general_formula_l180_180330


namespace kim_knit_sweaters_total_l180_180440

theorem kim_knit_sweaters_total :
  ∀ (M T W R F : ℕ), 
    M = 8 →
    T = M + 2 →
    W = T - 4 →
    R = T - 4 →
    F = M / 2 →
    M + T + W + R + F = 34 :=
by
  intros M T W R F hM hT hW hR hF
  rw [hM, hT, hW, hR, hF]
  norm_num
  sorry

end kim_knit_sweaters_total_l180_180440


namespace box_weight_l180_180989

theorem box_weight (W : ℝ) (h : 7 * (W - 20) = 3 * W) : W = 35 := by
  sorry

end box_weight_l180_180989


namespace concurrent_circles_at_circumcenter_l180_180599

theorem concurrent_circles_at_circumcenter (A B C D O M N: Point) 
  (h_cyclic: IsCyclic A B C D) 
  (h_midpoints_AB_M: M = midpoint A B) 
  (h_midpoints_AD_N: N = midpoint A D) 
  (h_circumcenter: O = circumcenter A B C D) 
  (h_circle_AMN_A_M_N: CircleThroughPoints A M N) 
  (h_circle_BML_B_M_L: CircleThroughPoints B M L) 
  (h_circle_CNX_C_N_X: CircleThroughPoints C N X) 
  (h_circle_DNY_D_N_Y: CircleThroughPoints D N Y) : 
  ∃ P, P = O ∧ 
    OnCircleThroughPoints P A M N ∧ 
    OnCircleThroughPoints P B M L ∧ 
    OnCircleThroughPoints P C N X ∧ 
    OnCircleThroughPoints P D N Y := 
by 
  sorry

end concurrent_circles_at_circumcenter_l180_180599


namespace initial_percentage_female_workers_l180_180507

theorem initial_percentage_female_workers
(E : ℕ) (F : ℝ) 
(h1 : E + 30 = 360) 
(h2 : (F / 100) * E = (55 / 100) * (E + 30)) :
F = 60 :=
by
  -- proof omitted
  sorry

end initial_percentage_female_workers_l180_180507


namespace solve_quadratic_eq_l180_180128

theorem solve_quadratic_eq (x : ℝ) :
  x^2 - 7 * x + 6 = 0 ↔ x = 1 ∨ x = 6 :=
by
  sorry

end solve_quadratic_eq_l180_180128


namespace right_triangles_congruent_by_ASA_l180_180930

theorem right_triangles_congruent_by_ASA
  {ABC DEF : Type}
  [IsRightTriangle ABC] [IsRightTriangle DEF]
  {AB DE : ℝ} {AC DF : Angle}:
  (AB = DE) → (angleOppositeLeg ABC AB = angleOppositeLeg DEF DE) →
  (congruent ABC DEF) :=
by
  sorry

end right_triangles_congruent_by_ASA_l180_180930


namespace find_income_l180_180974

-- Define the conditions
def income_and_expenditure (income expenditure : ℕ) : Prop :=
  5 * expenditure = 3 * income

def savings (income expenditure : ℕ) (saving : ℕ) : Prop :=
  income - expenditure = saving

-- State the theorem
theorem find_income (expenditure : ℕ) (saving : ℕ) (h1 : income_and_expenditure 5 3) (h2 : savings (5 * expenditure) (3 * expenditure) saving) :
  5 * expenditure = 10000 :=
by
  -- Use the provided hint or conditions
  sorry

end find_income_l180_180974


namespace ordered_pair_a_82_a_28_l180_180597

-- Definitions for the conditions
def a (i j : ℕ) : ℕ :=
  if i % 2 = 1 then
    if j = 1 then i * i else i * i - (j - 1)
  else
    if j = 1 then (i-1) * i + 1 else i * i - (j - 1)

theorem ordered_pair_a_82_a_28 : (a 8 2, a 2 8) = (51, 63) := by
  sorry

end ordered_pair_a_82_a_28_l180_180597


namespace min_gennadys_l180_180640

theorem min_gennadys (alexanders borises vasilies : ℕ) (x : ℕ) 
    (h1 : alexanders = 45)
    (h2 : borises = 122)
    (h3 : vasilies = 27)
    (h4 : x = 49)
    (h5 : borises - 1 = alexanders + vasilies + x) :
  x = 49 := 
begin
  sorry,
end

end min_gennadys_l180_180640


namespace exponent_of_5_in_30_fact_l180_180024

def count_powers_of_5 (n : ℕ) : ℕ :=
  if n < 5 then 0
  else n / 5 + count_powers_of_5 (n / 5)

theorem exponent_of_5_in_30_fact : count_powers_of_5 30 = 7 := 
  by
    sorry

end exponent_of_5_in_30_fact_l180_180024


namespace tangency_of_circles_l180_180276

/-- Given a triangle ABC with incenter I, points P and Q where perpendiculars from I intersect AB and AC respectively, 
and a circle T_A' tangent to AB at P and to AC at Q, proving that T_A' is tangent to the circumcircle of triangle ABC --/
theorem tangency_of_circles
  (A B C I P Q : Point)
  (h_incenter: is_incenter_of_triangle I A B C)
  (h_perp_P: is_perpendicular I P A B)
  (h_perp_Q: is_perpendicular I Q A C)
  (T_A' : Circle)
  (h_tangent_P: is_tangent_at_point T_A' (segment A P))
  (h_tangent_Q: is_tangent_at_point T_A' (segment A Q))
  :  is_tangent (circumcircle_of_triangle A B C) T_A' :=
sorry

end tangency_of_circles_l180_180276


namespace triangle_lengths_ce_l180_180419

theorem triangle_lengths_ce (AE BE CE : ℝ) (angle_AEB angle_BEC angle_CED : ℝ) (h1 : angle_AEB = 30)
  (h2 : angle_BEC = 45) (h3 : angle_CED = 45) (h4 : AE = 30) (h5 : BE = AE / 2) (h6 : CE = BE) : CE = 15 :=
by sorry

end triangle_lengths_ce_l180_180419


namespace part1_proof_part2_proof_l180_180347

variable {A B C a b c : ℝ} -- angles A, B, C and sides a, b, c of triangle ABC

-- Condition: In ΔABC, sin^2(A) = sin(B) sin(C)
def condition1 : Prop := sin A * sin A = sin B * sin C

-- Additional condition for question 1: A = π/3
def condition2 (A : ℝ) : Prop := A = real.pi / 3

-- Additional condition for question 2: b * c = 1
def condition3 (b c : ℝ) : Prop := b * c = 1

-- (1) Prove angle B given the conditions
theorem part1_proof : condition1 → condition2 A → B = real.pi / 3 := by
  sorry

-- (2) Prove the maximum area of ΔABC given the conditions
theorem part2_proof (ΔA : A ∈ set.Ioo 0 real.pi) : condition1 → condition3 b c → 
  (1 / 2) * b * c * sin A ≤ sqrt 3 / 4 := by
  sorry

end part1_proof_part2_proof_l180_180347


namespace fair_game_expected_winnings_l180_180180

theorem fair_game_expected_winnings (num_players : ℕ) (total_pot : ℝ) 
  (p : ℕ → ℝ) (stakes : ℕ → ℝ) :
  num_players = 36 →
  (∀ k, p k = (35 / 36) ^ (k - 1) * p 1) →
  (∀ k, stakes k = total_pot * p k) →
  (∀ k, let L_k := stakes k in total_pot * p k - L_k * p k - L_k + L_k * p k = 0) :=
sorry

end fair_game_expected_winnings_l180_180180


namespace locus_of_A2_l180_180442

-- Definitions for the problem setup
variables {A B C I A1 A2 : Type}
  [Poncelet_triangle : Prop]
  [Reflection_about_incenter : I.A1 = A.reflect_about I]
  [Isogonal_conjugate : ∀ (X Y Z : Type), Isogonal_conjugate X Y Z A1 A2]

-- The main theorem statement
theorem locus_of_A2 :
  ∀ (A B C I A1 A2 : Type), Poncelet_triangle A B C →
  Reflection_about_incenter A I A1 →
  Isogonal_conjugate A B C A1 A2 →
  exists (line : Type), ∀ (P : A2), On_line P line :=
sorry

end locus_of_A2_l180_180442


namespace S_two_eq_l180_180891

def S (n : ℕ) [NeZero n] : ℚ :=
  ∑ k in Finset.range (n^2 - n + 1), (1 : ℚ) / (n + k)

theorem S_two_eq : S 2 = 1/2 + 1/3 + 1/4 := by
  sorry

end S_two_eq_l180_180891


namespace sec_minus_tan_l180_180812

-- Define the problem in Lean 4
theorem sec_minus_tan (x : ℝ) (h : Real.sec x + Real.tan x = 7 / 3) : Real.sec x - Real.tan x = 3 / 7 := by
  -- One could also include here the necessary mathematical facts and connections.
  sorry -- Proof to be provided

end sec_minus_tan_l180_180812


namespace minimum_gennadys_l180_180621

theorem minimum_gennadys (alexs borises vasilies x : ℕ) (h₁ : alexs = 45) (h₂ : borises = 122) (h₃ : vasilies = 27)
    (h₄ : ∀ i, i ∈ list.range (borises-1) → alexs + vasilies + x > i) : 
    x = 49 :=
by 
    sorry

end minimum_gennadys_l180_180621


namespace total_ways_to_draw_cards_l180_180508

noncomputable def draw_cards : ℕ :=
  let total_cards := 16 in
  let red_cards := 4 in
  let yellow_cards := 4 in
  let blue_cards := 4 in
  let green_cards := 4 in
  let drawn_cards := 3 in
  -- Define the conditions for drawing cards accordingly
  sorry

theorem total_ways_to_draw_cards : draw_cards = 472 := by
  sorry

end total_ways_to_draw_cards_l180_180508


namespace evaluate_f_at_2_l180_180095

noncomputable def f : ℝ → ℝ
| x => if x ≤ 0 then Real.log (3 * x + 2) / Real.log 2 else f (x - 1)

theorem evaluate_f_at_2 : f 2 = 1 := by
  sorry

end evaluate_f_at_2_l180_180095


namespace definite_integral_of_odd_function_l180_180696

variable (x : ℝ)
def f (x : ℝ) : ℝ := exp x - exp (-x)

theorem definite_integral_of_odd_function : ∫ x in -1..1, f x = 0 := by
  sorry

end definite_integral_of_odd_function_l180_180696


namespace sixth_graders_forgot_homework_percentage_l180_180219

-- Definitions of the conditions
def num_students_A : ℕ := 20
def num_students_B : ℕ := 80
def percent_forgot_A : ℚ := 20 / 100
def percent_forgot_B : ℚ := 15 / 100

-- Statement to be proven
theorem sixth_graders_forgot_homework_percentage :
  (num_students_A * percent_forgot_A + num_students_B * percent_forgot_B) /
  (num_students_A + num_students_B) = 16 / 100 :=
by
  sorry

end sixth_graders_forgot_homework_percentage_l180_180219


namespace prob_at_least_one_goes_l180_180305

-- Conditions: probabilities of the persons going to Beijing and independence
variable (P_A P_B P_C : ℚ) -- Probabilities of A, B, C going
variable (independence : independent {P_A, P_B, P_C})

-- The given probabilities
def P_A_prob : P_A = 1 / 3 := sorry
def P_B_prob : P_B = 1 / 4 := sorry
def P_C_prob : P_C = 1 / 5 := sorry

-- Proof goal: the probability of at least one of A, B, or C going to Beijing
theorem prob_at_least_one_goes (P_A P_B P_C : ℚ) (independence : independent {P_A, P_B, P_C})
    (h1 : P_A = 1 / 3) (h2 : P_B = 1 / 4) (h3 : P_C = 1 / 5) :
    1 - ((1 - P_A) * (1 - P_B) * (1 - P_C)) = 3 / 5 := by
  sorry

end prob_at_least_one_goes_l180_180305


namespace value_of_a_l180_180375

-- Given the universal set U = ℝ
def U : Set ℝ := Set.univ

-- Define the set M = {x | x + 2a ≥ 0}
def M (a : ℝ) : Set ℝ := { x | x + 2 * a ≥ 0 }

-- Define the set N = {x | log₂(x - 1) < 1}
def N : Set ℝ := { x | 1 < x ∧ x < 3 }

-- Define the complement of N in U
def complement_N : Set ℝ := { x | x ≤ 1 ∨ x ≥ 3 }

-- Define the intersection condition
def condition (a : ℝ) : Prop := M(a) ∩ complement_N = { x | x = 1 ∨ x ≥ 3 }

-- The theorem to prove
theorem value_of_a (a : ℝ) : condition a ↔ a = -1 / 2 :=
by
  sorry

end value_of_a_l180_180375


namespace kettle_cannot_fill_100_cups_l180_180244

theorem kettle_cannot_fill_100_cups (liters_in_kettle : ℕ) (milliliters_per_cup : ℕ) :
  liters_in_kettle = 2.5 * 1000 ∧ milliliters_per_cup = 250 → 100 * milliliters_per_cup > liters_in_kettle :=
by
  intros
  unfold_coes
  sorry

end kettle_cannot_fill_100_cups_l180_180244


namespace product_of_solutions_of_abs_eq_l180_180712

theorem product_of_solutions_of_abs_eq (x : ℝ) (h : |x^2 - 4| = 2 * (|x| - 1)) : 
  ∃ a b c d : ℝ, (a = 1 + real.sqrt 3 ∧ b = 1 - real.sqrt 3 ∧ c = -1 + real.sqrt 3 ∧ d = -1 - real.sqrt 3) 
  ∧ (a * b * c * d = 4) :=
by
  sorry

end product_of_solutions_of_abs_eq_l180_180712


namespace four_points_all_edges_red_l180_180243

def color := ℕ -- Using ℕ to represent colors, where 0 can be red and 1 can be blue.

def is_red (c : color) : Prop := c = 0

def has_red_edge (E : Finset (Finset ℕ)) (e : Finset ℕ) (col : ℕ → ℕ → color) : Prop :=
  ∃ x ∈ e, ∃ y ∈ e, x ≠ y ∧ is_red (col x y)

noncomputable def exists_red_K4 (V : Finset ℕ) (E : Finset (Finset ℕ)) (col : ℕ → ℕ → color) :=
  (V.card = 9) ∧
  (E.card = 36) ∧
  (∀ t ∈ V.powersetLen 3, has_red_edge E t col) →
  ∃ S ∈ V.powersetLen 4, ∀ e ∈ S.powersetLen 2, is_red (col e.1 e.2)

theorem four_points_all_edges_red (V : Finset ℕ) (E : Finset (Finset ℕ)) (col : ℕ → ℕ → color) :
  exists_red_K4 V E col :=
sorry

end four_points_all_edges_red_l180_180243


namespace AP_over_PC_eq_AB_over_BC_l180_180738

variables (A B C P D E F : Type*)
variables [Point A] [Point B] [Point C] [Point P]
variables [Point D] [Point E] [Point F]
variables [OnCircumcircle P A B C]

-- Let D, E, F be the feet of the perpendiculars from P to BC, CA, and AB respectively.
variables (hD : Perpendicular D P BC)
variables (hE : Perpendicular E P CA)
variables (hF : Perpendicular F P AB)

-- E is the midpoint of segment DF
variables (h_midpoint : Midpoint E D F)

-- Prove \(\frac{AP}{PC} = \frac{AB}{BC}\)
theorem AP_over_PC_eq_AB_over_BC :
  D, E, F feet of perpendiculars from P to BC, CA, AB respectively →
  (E is the midpoint of line segment DF) →
  \(\frac{AP}{PC} = \frac{AB}{BC}\) :=
sorry

end AP_over_PC_eq_AB_over_BC_l180_180738


namespace num_valid_quadruples_l180_180313

def is_factor_of_30 (n : ℕ) : Prop :=
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 6 ∨ n = 10 ∨ n = 15 ∨ n = 30

def valid_quadruple (a b c d : ℕ) : Prop :=
  is_factor_of_30 a ∧ is_factor_of_30 b ∧ is_factor_of_30 c ∧ is_factor_of_30 d ∧ a * b * c * d > 900

theorem num_valid_quadruples : 
  (finset.univ.filter (λ (t : ℕ × ℕ × ℕ × ℕ), 
    valid_quadruple t.1 t.2.1 t.2.2.1 t.2.2.2)).card = 1940 :=
sorry

end num_valid_quadruples_l180_180313


namespace percentage_of_green_ducks_is_28_24_l180_180855

def smallest_pond_ducks : ℕ := 45
def medium_pond_ducks : ℕ := 55
def largest_pond_ducks : ℕ := 70

def green_ducks_smallest_pond : ℕ := (0.20 * smallest_pond_ducks).to_nat
def green_ducks_medium_pond : ℕ := (0.40 * medium_pond_ducks).to_nat
def green_ducks_largest_pond : ℕ := (0.25 * largest_pond_ducks).to_nat

def total_green_ducks : ℕ := green_ducks_smallest_pond + green_ducks_medium_pond + green_ducks_largest_pond
def total_ducks : ℕ := smallest_pond_ducks + medium_pond_ducks + largest_pond_ducks

noncomputable def percentage_green_ducks : ℝ := (total_green_ducks.to_real / total_ducks.to_real) * 100

theorem percentage_of_green_ducks_is_28_24 :
  abs (percentage_green_ducks - 28.24) < 0.01 := by
  sorry

end percentage_of_green_ducks_is_28_24_l180_180855


namespace probability_of_at_least_two_consecutive_heads_equals_11_over_16_l180_180241

-- Definitions of the conditions
def fair_coin_toss_outcome_space : ℕ := 2^4

def unfavorable_outcomes : fin 5 := ⟨[0, 1, 2, 4, 8], sorry⟩ -- positions of TTTT, TTTH, TTHT, THTT, HTTT as bit representations

def probability_of_unfavorable : ℚ := 5 * (1 / fair_coin_toss_outcome_space)

-- Theorem statement
theorem probability_of_at_least_two_consecutive_heads_equals_11_over_16 :
  1 - probability_of_unfavorable = 11 / 16 :=
sorry

end probability_of_at_least_two_consecutive_heads_equals_11_over_16_l180_180241


namespace distance_between_Hyosung_and_Mimi_after_15_minutes_l180_180962

theorem distance_between_Hyosung_and_Mimi_after_15_minutes :
  ∀ (initial_distance : ℝ) (mimi_speed : ℝ) (hyosung_speed_minute : ℝ) (time_minutes : ℕ),
  initial_distance = 2.5 →
  mimi_speed = 2.4 →
  hyosung_speed_minute = 0.08 →
  time_minutes = 15 →
  let hyosung_speed_hour := hyosung_speed_minute * 60 in
  let time_hour := (time_minutes: ℝ) / 60 in
  let mimi_distance := mimi_speed * time_hour in
  let hyosung_distance := hyosung_speed_hour * time_hour in
  let total_distance_covered := mimi_distance + hyosung_distance in
  let remaining_distance := initial_distance - total_distance_covered in
  remaining_distance = 0.7 :=
begin
  intros,
  -- A placeholder for the proof
  sorry
end

end distance_between_Hyosung_and_Mimi_after_15_minutes_l180_180962


namespace area_triangle_AED_l180_180278

-- Define points in the plane
variable {Point : Type}
variable [EuclideanGeometry Point]

-- Define collinearity and perpendicularity of points
variable {A B C D E : Point}
variable (h_collinear : Collinear ℝ [C, E, B])
variable (h_perp : Perpendicular ℝ CB AB)
variable (h_parallel : Parallel ℝ AE DC)
variable (AB_length : distance A B = 8)
variable (CE_length : distance C E = 5)

-- Define the problem
theorem area_triangle_AED : 
  area (triangle A E D) = 20 :=
by 
  sorry

end area_triangle_AED_l180_180278


namespace discount_percentage_is_25_l180_180463

-- Definitions representing the conditions of the problem
def n : ℕ := 2
def c : ℕ := 650
def p : ℕ := 975

-- The goal is to prove that the discount percentage is 25%
theorem discount_percentage_is_25 :
  let total_cost := n * c in
  let discount_amount := total_cost - p in
  let discount_percent := (discount_amount * 100) / total_cost in
  discount_percent = 25 := by
  sorry

end discount_percentage_is_25_l180_180463


namespace teacher_total_score_l180_180260

variable (written_score : ℕ)
variable (interview_score : ℕ)
variable (weight_written : ℝ)
variable (weight_interview : ℝ)

theorem teacher_total_score :
  (written_score = 80) → (interview_score = 60) → (weight_written = 0.6) → (weight_interview = 0.4) →
  (written_score * weight_written + interview_score * weight_interview = 72) :=
by
  sorry

end teacher_total_score_l180_180260


namespace number_of_smaller_cubes_l180_180564

theorem number_of_smaller_cubes (edge : ℕ) (N : ℕ) (h_edge : edge = 5)
  (h_divisors : ∃ (a b c : ℕ), a + b + c = N ∧ a * 1^3 + b * 2^3 + c * 3^3 = edge^3 ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
  N = 22 :=
by
  sorry

end number_of_smaller_cubes_l180_180564


namespace foci_of_ellipse_shortest_major_axis_ellipse_l180_180767

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 9 = 0

-- Define the parametric form of the ellipse C
def ellipse_C (θ : ℝ) : ℝ × ℝ := (2 * sqrt 3 * Real.cos θ, sqrt 3 * Real.sin θ)

-- Problem 1: Prove the coordinates of the two foci F1 and F2
theorem foci_of_ellipse : ∃ (F1 F2 : ℝ × ℝ), 
  F1 = (-3, 0) ∧ F2 = (3, 0) ∧ 
  -- Ensure F1 and F2 are the foci of the ellipse defined by ellipse_C
  ∀ θ, let (x, y) := ellipse_C θ in x^2 / 12 + y^2 / 3 = 1 := sorry

-- Problem 2: Prove the equation of the ellipse with the shortest major axis
theorem shortest_major_axis_ellipse : 
  ∃ (a b : ℝ), a = 3 * sqrt 5 ∧ b = 6 ∧ 
  ∀ (x y : ℝ), 
    (2 * sqrt 3 * Real.cos θ, sqrt 3 * Real.sin θ) = (x, y) ∨ line_l x y → 
    x^2 / (a^2) + y^2 / (b^2) = 1 := sorry

end foci_of_ellipse_shortest_major_axis_ellipse_l180_180767


namespace c_share_of_profit_l180_180549

-- Definitions for the investments and total profit
def investments_a := 800
def investments_b := 1000
def investments_c := 1200
def total_profit := 1000

-- Definition for the share of profits based on the ratio of investments
def share_of_c : ℕ :=
  let ratio_a := 4
  let ratio_b := 5
  let ratio_c := 6
  let total_ratio := ratio_a + ratio_b + ratio_c
  (ratio_c * total_profit) / total_ratio

-- The theorem to be proved
theorem c_share_of_profit : share_of_c = 400 := by
  sorry

end c_share_of_profit_l180_180549


namespace min_gennadys_l180_180602

-- Defining the basic constants for each name type
def Alexanders : Nat := 45
def Borises : Nat := 122
def Vasilies : Nat := 27

-- Define the proof statement to check the minimum number of Gennadys needed
theorem min_gennadys (a b v : Nat) (no_adjacent: a = 45 ∧ b = 122 ∧ v = 27) : ∃ g : Nat, g = 49 :=
by
  -- Using provided conditions
  cases no_adjacent with h_a h_bv
  cases h_bv with h_b h_v
  -- Correct answer derived from the solution
  use 49
  -- skipping proof details 
  sorry

end min_gennadys_l180_180602


namespace convert_to_cylindrical_l180_180295

-- Define the function for converting rectangular to cylindrical coordinates
def rectangular_to_cylindrical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := Real.atan2 y x
  (r, θ, z)

-- Given conditions
def point_rectangular : ℝ × ℝ × ℝ := (3, -3 * Real.sqrt 3, 2)
def expected_result : ℝ × ℝ × ℝ := (6, 5 * Real.pi / 3, 2)

-- The theorem to prove
theorem convert_to_cylindrical :
  rectangular_to_cylindrical 3 (-3 * Real.sqrt 3) 2 = expected_result := by
  sorry

end convert_to_cylindrical_l180_180295


namespace sec_minus_tan_l180_180792

theorem sec_minus_tan (x : ℝ) (h : real.sec x + real.tan x = 7 / 3) :
  real.sec x - real.tan x = 3 / 7 :=
sorry

end sec_minus_tan_l180_180792


namespace exponent_of_5_in_30_factorial_l180_180001

theorem exponent_of_5_in_30_factorial : Nat.factorial 30 ≠ 0 → (nat.factorization (30!)).coeff 5 = 7 :=
by
  sorry

end exponent_of_5_in_30_factorial_l180_180001


namespace exponent_of_five_in_30_factorial_l180_180033

theorem exponent_of_five_in_30_factorial : 
  nat.factorial_prime_exponent 30 5 = 7 := 
sorry

end exponent_of_five_in_30_factorial_l180_180033


namespace angle_between_vectors_l180_180344

theorem angle_between_vectors
    (a b : ℝ)
    (ha : |a| = 4 * real.cos (real.pi / 8))
    (hb : |b| = 2 * real.sin (real.pi / 8))
    (dot_ab : a * b = -real.sqrt 2) :
  ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ real.pi ∧ real.cos θ = -1 / 2 ∧ θ = 2 * real.pi / 3 :=
begin
  sorry -- Proof to be completed
end

end angle_between_vectors_l180_180344


namespace find_x_average_is_60_l180_180480

theorem find_x_average_is_60 : 
  ∃ x : ℕ, (54 + 55 + 57 + 58 + 59 + 62 + 62 + 63 + x) / 9 = 60 ∧ x = 70 :=
by
  existsi 70
  sorry

end find_x_average_is_60_l180_180480


namespace run_to_friends_house_l180_180783

theorem run_to_friends_house
  (constant_pace : ∀ (d1 d2 : ℕ), d2 ≠ 0 → (18 : ℕ) / (2 : ℕ) = (d1 / (d2 : ℕ)) → d1 = 9)
  (distance_friend_house : ℕ)
  (H_distance : distance_friend_house = 1) :
  ∃ (time : ℕ), time = 9 :=
by
  exists 9
  sorry

end run_to_friends_house_l180_180783


namespace slope_range_l180_180911

def coord := ℝ × ℝ

def satisfies_ellipse (p : coord) : Prop :=
  let (x, y) := p in
  x^2 / 4 + y^2 / 2 = 1

def satisfies_function (p : coord) : Prop :=
  let (x, y) := p in
  y = x^3

def range_of_slope_PA (A P : coord) : Prop :=
  let (x1, y1) := A in
  let (x0, y0) := P in
  ∃ k : ℝ, k ∈ set.Icc (-3 : ℝ) (-1 : ℝ) ∧ k = -1/2 * ((x0 + x1) / (y0 + y1))

def range_of_slope_PB (B P : coord) : Prop :=
  let (x1, y1) := B in
  let (x0, y0) := P in
  ∃ k : ℝ, k ∈ set.Icc (1 / 6 : ℝ) (1 / 2 : ℝ) ∧ k = (y0 + y1) / (x0 + x1)

theorem slope_range (A B P : coord) (hA : satisfies_ellipse A ∧ satisfies_function A)
  (hB : satisfies_ellipse B ∧ satisfies_function B)
  (hP : satisfies_ellipse P ∧ P ≠ A ∧ P ≠ B)
  (hPA : range_of_slope_PA A P) :
  range_of_slope_PB B P := 
sorry

end slope_range_l180_180911


namespace product_xy_eq_3_l180_180346

variable {x y : ℝ}
variables (h₀ : x ≠ y) (h₁ : x ≠ 0) (h₂ : y ≠ 0)
variable (h₃ : x + (3 / x) = y + (3 / y))

theorem product_xy_eq_3 : x * y = 3 := by
  sorry

end product_xy_eq_3_l180_180346


namespace hyperbola_vertex_distance_l180_180705

theorem hyperbola_vertex_distance :
  ∀ (x y : ℝ), (x * x / 36) - (y * y / 16) = 1 → 12 = 12 :=
by
  intros x y h
  exact rfl
  sorry

end hyperbola_vertex_distance_l180_180705


namespace min_gennadies_l180_180644

noncomputable section

def minGennadiesNeeded (alexanders borises vasilies : Nat) : Nat :=
  let needed_gaps := borises - 1
  let total_others := alexanders + vasilies
  if needed_gaps > total_others then needed_gaps - total_others else 0

theorem min_gennadies (alexanders borises vasilies : Nat) (h_alex: alexanders = 45) (h_boris: borises = 122) (h_vasil: vasilies = 27):
  minGennadiesNeeded alexanders borises vasilies = 49 := by
  rw [h_alex, h_boris, h_vasil]
  simp [minGennadiesNeeded]
  sorry

end min_gennadies_l180_180644


namespace sec_tan_eq_l180_180833

theorem sec_tan_eq (x : ℝ) (h : Real.cos x ≠ 0) : 
  Real.sec x + Real.tan x = 7 / 3 → Real.sec x - Real.tan x = 3 / 7 :=
by
  intro h1
  sorry

end sec_tan_eq_l180_180833


namespace overall_average_score_l180_180400

structure Club where
  members : Nat
  average_score : Nat

def ClubA : Club := { members := 40, average_score := 90 }
def ClubB : Club := { members := 50, average_score := 81 }

theorem overall_average_score : 
  (ClubA.members * ClubA.average_score + ClubB.members * ClubB.average_score) / 
  (ClubA.members + ClubB.members) = 85 :=
by
  sorry

end overall_average_score_l180_180400


namespace sec_minus_tan_l180_180815

-- Define the problem in Lean 4
theorem sec_minus_tan (x : ℝ) (h : Real.sec x + Real.tan x = 7 / 3) : Real.sec x - Real.tan x = 3 / 7 := by
  -- One could also include here the necessary mathematical facts and connections.
  sorry -- Proof to be provided

end sec_minus_tan_l180_180815


namespace exponent_of_5_in_30_factorial_l180_180005

theorem exponent_of_5_in_30_factorial : 
  (nat.factors 30!).count 5 = 7 :=
sorry

end exponent_of_5_in_30_factorial_l180_180005


namespace monotonicity_and_extremum_inequality_range_l180_180914

noncomputable theory

-- Definitions based on given conditions
def f (x : ℝ) (a b : ℝ) := -(1 / 3) * x ^ 3 + 2 * a * x ^ 2 - 3 * a ^ 2 * x + b
def f' (x : ℝ) (a : ℝ) := -x ^ 2 + 4 * a * x - 3 * a ^ 2

-- Statements to prove
theorem monotonicity_and_extremum (a b : ℝ) (h_a : 0 < a ∧ a < 1) (h_b : b ∈ Set.univ) :
  (∀ x, a < x ∧ x < 3 * a → f' x a > 0) ∧ 
  (∀ x, x < a ∨ x > 3 * a → f' x a < 0) ∧ 
  f a a b < f 3 * a a b :=
sorry

theorem inequality_range (a : ℝ) (h : ∀ x, a + 1 ≤ x ∧ x ≤ a + 2 → |f' x a| ≤ a) :
  4 / 5 ≤ a ∧ a < 1 :=
sorry

end monotonicity_and_extremum_inequality_range_l180_180914


namespace not_all_pairs_of_vectors_in_plane_can_serve_as_basis_l180_180595

theorem not_all_pairs_of_vectors_in_plane_can_serve_as_basis (u v : ℝ × ℝ) :
  ¬(∀ u v, ¬collinear u v → is_basis {u, v}) :=
sorry

end not_all_pairs_of_vectors_in_plane_can_serve_as_basis_l180_180595


namespace ballroom_majority_l180_180692

-- Definitions of beauty and intelligence rankings
def beauty (n : ℕ) : ℕ := n
def intelligence (n : ℕ) : ℕ := if n = 10 then 0 else n

-- Definitions of first and second dance pairings
def first_dance_pair (k : ℕ) : ℕ × ℕ := (k, k)
def second_dance_pair (k : ℕ) : ℕ × ℕ := if k < 10 then (k, k+1) else (k, 1)

-- Condition checks for beauty and intelligence
def more_beautiful (p1 p2 : ℕ × ℕ) : Prop := beauty p2.2 > beauty p1.2
def more_intelligent (p1 p2 : ℕ × ℕ) : Prop := intelligence p2.2 > intelligence p1.2

-- Statement of the problem
theorem ballroom_majority (young_men: fin 10) (young_women: fin 10) :
  (finset.filter (λ k:fin 10, 
    more_beautiful (first_dance_pair k) (second_dance_pair k) 
    ∧ more_intelligent (first_dance_pair k) (second_dance_pair k)) 
  finset.univ).card ≥ 8 :=
sorry

end ballroom_majority_l180_180692


namespace like_terms_monomials_l180_180394

theorem like_terms_monomials (m n : ℕ) (h₁ : m = 2) (h₂ : n = 1) : m + n = 3 := 
by
  sorry

end like_terms_monomials_l180_180394


namespace hannah_money_left_l180_180777

variable (initial_amount : ℕ) (amount_spent_rides : ℕ) (amount_spent_dessert : ℕ)
  (remaining_after_rides : ℕ) (remaining_money : ℕ)

theorem hannah_money_left :
  initial_amount = 30 →
  amount_spent_rides = initial_amount / 2 →
  remaining_after_rides = initial_amount - amount_spent_rides →
  amount_spent_dessert = 5 →
  remaining_money = remaining_after_rides - amount_spent_dessert →
  remaining_money = 10 := by
  sorry

end hannah_money_left_l180_180777


namespace min_gennadys_needed_l180_180653

variables (A B V G : ℕ)

theorem min_gennadys_needed
  (hA : A = 45)
  (hB : B = 122)
  (hV : V = 27)
  (hG : ∀ i, i < 121 → A + V < 121 → G ≥ 49) :
  G = 49 :=
sorry

end min_gennadys_needed_l180_180653


namespace angle_equality_l180_180898

open EuclideanGeometry

noncomputable def triangleABC (A B C I A1 B1 C1 Q P R : Point) :=
  let incircle := Circle I (dist I A1)
  Triangle A B C ∧
  Incircle A B C I incircle ∧
  TangentToSide incircle A1 B C ∧
  TangentToSide incircle B1 C A ∧
  TangentToSide incircle C1 A B ∧
  Line A A1 ∧ LineIntersectCircle A A1 incircle Q ∧
  LineParallel A B C ∧
  LinePassesThrough l A ∧
  LineIntersectLine A1 C1 l P ∧
  LineIntersectLine A1 B1 l R

theorem angle_equality (A B C I A1 B1 C1 Q P R : Point) (l : Line) :
  triangleABC A B C I A1 B1 C1 Q P R →
  ∠ P Q R = ∠ B1 Q C1 := by
  sorry

end angle_equality_l180_180898


namespace sec_minus_tan_l180_180820

theorem sec_minus_tan
  (x : ℝ)
  (h1 : Real.sec x + Real.tan x = 7 / 3)
  (h2 : (Real.sec x + Real.tan x) * (Real.sec x - Real.tan x) = 1) :
  Real.sec x - Real.tan x = 3 / 7 :=
by
  sorry

end sec_minus_tan_l180_180820


namespace exponent_of_five_in_factorial_l180_180048

theorem exponent_of_five_in_factorial:
  (nat.factors 30!).count 5 = 7 :=
begin
  sorry
end

end exponent_of_five_in_factorial_l180_180048


namespace find_sum_of_n_l180_180177

theorem find_sum_of_n :
  ∃ (r : ℕ) (n : fin r → ℕ) (a : fin r → ℤ),
  (∀ i j, i < j → n i > n j) ∧
  (∀ k, a k = 1 ∨ a k = -1) ∧
  (∑ k in finset.univ, a k * 3 ^ (n k) = 2023) ∧
  (∑ k in finset.univ, n k = 19) :=
by
  sorry

end find_sum_of_n_l180_180177


namespace part1_part2_l180_180762

def f (x : ℝ) := |2 * x - 1| + |2 * x + 2|

theorem part1 (x : ℝ) (M : ℝ) (hM : M = 3) :
  f(x) < M + |2 * x + 2| ↔ -1 < x ∧ x < 2 :=
by
  sorry

theorem part2 (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (M : ℝ) (hM : M = 3) (h : a^2 + 2 * b^2 = M) :
  2 * a + b ≤ (3 * Real.sqrt 6) / 2 :=
by
  sorry

end part1_part2_l180_180762


namespace sequence_inequality_l180_180080

theorem sequence_inequality
  (a : ℕ → ℝ)
  (h : ∀ n : ℕ, 0 < n → (a (n - 1) + a (n + 1)) / 2 ≥ a n) :
  ∀ n : ℕ, 0 < n → (a 0 + a (n + 1)) / 2 ≥ (finset.range n).sum (λ i, a (i + 1)) / n :=
by
  sorry

end sequence_inequality_l180_180080


namespace price_of_large_cup_is_3_l180_180515

noncomputable def price_of_large_cup (small_sales medium_sales total_sales large_cups : ℕ) :=
  (total_sales - (small_sales + medium_sales)) / large_cups

theorem price_of_large_cup_is_3 :
  ∀ (small_sales medium_sales total_sales large_cups price_large : ℕ),
  small_sales = 11 →
  medium_sales = 24 →
  total_sales = 50 →
  large_cups = 5 →
  price_large = price_of_large_cup small_sales medium_sales total_sales large_cups →
  price_large = 3 :=
by
  intros small_sales medium_sales total_sales large_cups price_large h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, price_of_large_cup] at h5
  linarith

end price_of_large_cup_is_3_l180_180515


namespace total_volume_l180_180150

noncomputable def volume_of_cone (r h : ℝ) : ℝ :=
  (1 / 3) * Real.pi * r^2 * h

noncomputable def volume_of_hemisphere (r : ℝ) : ℝ :=
  (2 / 3) * Real.pi * r^3

theorem total_volume (r h : ℝ) (h_r : r = 3) (h_h : h = 12) :
  volume_of_cone r h + volume_of_hemisphere r = 54 * Real.pi :=
by {
  -- Use the given conditions
  rw [h_r, h_h],
  -- Calculate individual volumes
  have cone_volume : volume_of_cone 3 12 = 36 * Real.pi, {
    simp [volume_of_cone],
    norm_num,
  },
  have hemisphere_volume : volume_of_hemisphere 3 = 18 * Real.pi, {
    simp [volume_of_hemisphere],
    norm_num,
  },
  -- Combine volumes
  rw [cone_volume, hemisphere_volume],
  norm_num,
}

end total_volume_l180_180150


namespace total_cost_of_paths_l180_180254

noncomputable def plot_length : ℝ := 120 -- in meters
noncomputable def plot_width : ℝ := 0.85 -- in meters
noncomputable def gravel_path_width : ℝ := 0.05 -- in meters
noncomputable def concrete_path_width : ℝ := 0.07 -- in meters
noncomputable def gravel_cost_per_sqm : ℝ := 0.80 -- in Rs. per square meter
noncomputable def concrete_cost_per_sqm : ℝ := 1.50 -- in Rs. per square meter

noncomputable def gravel_path_area : ℝ := 2 * (plot_length * gravel_path_width)
noncomputable def concrete_path_area : ℝ := 2 * (plot_width * concrete_path_width)

noncomputable def cost_of_gravelling : ℝ := gravel_path_area * gravel_cost_per_sqm
noncomputable def cost_of_concreting : ℝ := concrete_path_area * concrete_cost_per_sqm

noncomputable def total_cost : ℝ := cost_of_gravelling + cost_of_concreting

theorem total_cost_of_paths :
  real.round (total_cost * 100) / 100 = 9.78 := by
  sorry

end total_cost_of_paths_l180_180254


namespace solve_number_reordering_l180_180854

def number_reordering_possible_in_99_moves : Prop :=
  ∀ (A : Array (Array ℕ)), A.size = 10 → (∀ i, (A[i]).size = 10) →
  (∃ B : Array (Array ℕ), B.size = 10 ∧ (∀ i, (B[i]).size = 10) ∧
    (∀ i j, (i < 10 ∧ j < 10) → B[i][j] < B[i][j + 1]) ∧
    (∀ i j, (i < 10 ∧ j < 10) → B[j][i] < B[j + 1][i]) :=
    sorry
 
theorem solve_number_reordering : number_reordering_possible_in_99_moves :=
  sorry

end solve_number_reordering_l180_180854


namespace min_gennadies_l180_180646

noncomputable section

def minGennadiesNeeded (alexanders borises vasilies : Nat) : Nat :=
  let needed_gaps := borises - 1
  let total_others := alexanders + vasilies
  if needed_gaps > total_others then needed_gaps - total_others else 0

theorem min_gennadies (alexanders borises vasilies : Nat) (h_alex: alexanders = 45) (h_boris: borises = 122) (h_vasil: vasilies = 27):
  minGennadiesNeeded alexanders borises vasilies = 49 := by
  rw [h_alex, h_boris, h_vasil]
  simp [minGennadiesNeeded]
  sorry

end min_gennadies_l180_180646


namespace police_positions_l180_180279

-- Define the intersections and the police officers with their respective visibility constraints
def intersection := {i : Nat // i ≥ 1 ∧ i ≤ 9}

def A_visible (x : intersection) (y : intersection) := 
  -- Placeholder condition for A's visibility statement. 
  sorry

def B_visible (x : intersection) (y : intersection) := 
  -- Placeholder condition for B's visibility statement.
  sorry

def C_visible (x : intersection) (y : intersection) := 
  -- Placeholder condition for C's visibility statement.
  sorry

def D_visible (x : intersection) (y : intersection) := 
  -- Placeholder condition for D's visibility statement.
  sorry

def E_visible (x : intersection) (y : intersection) := 
  -- Placeholder condition for E's visibility statement.
  sorry

-- Define the main theorem to prove
theorem police_positions :
  ∃ (a b c d e : intersection),
    A_visible a ⟨2, sorry⟩ ∧ 
    B_visible b ⟨5, sorry⟩ ∧ 
    C_visible c ⟨4, sorry⟩ ∧ 
    D_visible d ⟨9, sorry⟩ ∧ 
    E_visible e ⟨6, sorry⟩ ∧ 
    ⟨2, sorry⟩ = a ∧ 
    ⟨5, sorry⟫ = b ∧ 
    ⟨4, sorry⟩ = c ∧ 
    ⟨9, sorry⟩ = d ∧ 
    ⟨6, sorry⟩ = e := 
  sorry

end police_positions_l180_180279


namespace dot_product_necessary_condition_l180_180476

variable {a b : EuclideanSpace ℝ (Fin 2)} -- assuming a 2D Euclidean space for simplicity
variable {θ : ℝ} -- The angle between vectors a and b

def is_unit_vector (v : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ‖v‖ = 1

def is_acute_angle (θ : ℝ) : Prop :=
  0 ≤ θ ∧ θ < Real.pi / 2

theorem dot_product_necessary_condition
  (ha : is_unit_vector a)
  (hb : is_unit_vector b)
  (H1 : 0 ≤ θ ∧ θ ≤ Real.pi) 
  (H : θ = Real.angleBetween a b)
  : (0 < (a ⬝ b)) ↔ is_acute_angle θ :=
sorry

end dot_product_necessary_condition_l180_180476


namespace length_of_CC_length_of_EF_l180_180170

/-- Given a rectangle ABCD with AB = 240 cm and BC = 288 cm, where the paper is folded along
    segment EF such that point C lies over the midpoint of AB. -/
theorem length_of_CC' (AB BC : ℝ) (fold_condition : True) (h_AB : AB = 240) (h_BC : BC = 288) :
  let BM := AB / 2 in
  let CC' := real.sqrt (BC ^ 2 + BM ^ 2) in
  CC' = 312 :=
by
  simp [BM, CC', h_AB, h_BC]
  sorry

/-- Given a rectangle ABCD with AB = 240 cm and BC = 288 cm, where the paper is folded along
    segment EF such that point C lies over the midpoint of AB, and CC' has been found to be 312 cm,
    what is the length of EF? -/
theorem length_of_EF (AB BC CC' : ℝ) (fold_condition : True) 
  (h_AB : AB = 240) (h_BC : BC = 288) (h_CC' : CC' = 312) :
  let EF := (13 / 12) * (AB / 2) in
  EF = 260 :=
by
  simp [EF, h_AB]
  sorry

end length_of_CC_length_of_EF_l180_180170


namespace book_club_boys_count_l180_180959

theorem book_club_boys_count (B G : ℕ) 
  (h1 : B + G = 30) 
  (h2 : B + (1 / 3 : ℝ) * G = 18) :
  B = 12 :=
by
  have h3 : 3 • B + G = 54 := sorry
  have h4 : 3 • B + G - (B + G) = 54 - 30 := sorry
  have h5 : 2 • B = 24 := sorry
  have h6 : B = 12 := sorry
  exact h6

end book_club_boys_count_l180_180959


namespace total_whales_correct_l180_180869

def first_trip_male_whales : ℕ := 28
def first_trip_female_whales : ℕ := 2 * first_trip_male_whales
def first_trip_total_whales : ℕ := first_trip_male_whales + first_trip_female_whales

def second_trip_baby_whales : ℕ := 8
def second_trip_parent_whales : ℕ := 2 * second_trip_baby_whales
def second_trip_total_whales : ℕ := second_trip_baby_whales + second_trip_parent_whales

def third_trip_male_whales : ℕ := first_trip_male_whales / 2
def third_trip_female_whales : ℕ := first_trip_female_whales
def third_trip_total_whales : ℕ := third_trip_male_whales + third_trip_female_whales

def total_whales_seen : ℕ :=
  first_trip_total_whales + second_trip_total_whales + third_trip_total_whales

theorem total_whales_correct : total_whales_seen = 178 := by
  sorry

end total_whales_correct_l180_180869


namespace intersection_of_P_and_Q_l180_180771

def P : Set ℤ := {-3, -2, 0, 2}
def Q : Set ℤ := {-1, -2, -3, 0, 1}

theorem intersection_of_P_and_Q : P ∩ Q = {-3, -2, 0} := by
  sorry

end intersection_of_P_and_Q_l180_180771


namespace regression_example_l180_180253

noncomputable def linear_regression (n : ℕ) (xs ys : Fin n → ℝ) : (ℝ × ℝ) :=
  let x_sum := ∑ i, xs i
  let y_sum := ∑ i, ys i
  let xy_sum := ∑ i, xs i * ys i
  let x2_sum := ∑ i, xs i * xs i
  let n := (n : ℝ)
  let x̄ := x_sum / n
  let ȳ := y_sum / n
  let β := (xy_sum - n * x̄ * ȳ) / (x2_sum - n * x̄ * x̄)
  let α := ȳ - β * x̄
  (β, α)

theorem regression_example :
  let n : ℕ := 10
  let xs : Fin n → ℝ := ![8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
  let ys : Fin n → ℝ := ![2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
  let (β, α) := linear_regression n xs ys
  β = 0.3 ∧ α = -0.4 := by
  sorry

end regression_example_l180_180253


namespace transformed_sine_function_l180_180518

theorem transformed_sine_function :
  (∀ x : ℝ, 
    (∃ z : ℝ, z = x - (π / 10) ∧ y = sin(z)) →
    y = sin((1 / 2) * x - (π / 20))) :=
sorry

end transformed_sine_function_l180_180518


namespace plant_lamp_arrangements_total_arrangements_l180_180939

-- Define the plants and lamps setup
structure Setup where
  basil1  : Prop
  basil2  : Prop
  aloe    : Prop
  cactus  : Prop
  white1  : Prop
  white2  : Prop
  red     : Prop
  yellow  : Prop

-- State the theorem for the number of ways to arrange the plants under the lamps
theorem plant_lamp_arrangements (s : Setup) : 
  (∃ (arr : Π (p : Prop), Prop), (arr s.basil1 = s.white1 ∨ arr s.basil1 = s.white2 ∨ arr s.basil1 = s.red ∨ arr s.basil1 = s.yellow) ∧
                                (arr s.basil2 = arr s.basil1) ∧
                                (arr s.aloe = s.white1 ∨ arr s.aloe = s.white2 ∨ arr s.aloe = s.red ∨ arr s.aloe = s.yellow) ∧
                                (arr s.cactus = s.white1 ∨ arr s.cactus = s.white2 ∨ arr s.cactus = s.red ∨ arr s.cactus = s.yellow) ∧
                                (arr s.basil1 ≠ arr s.aloe ∨ arr s.aloe ≠ arr s.cactus ∨ arr s.basil1 ≠ arr s.cactus)) := sorry

-- Combined statement reflecting the necessary constraints and the expected outcome
theorem total_arrangements (s : Setup) : 
  (number_of_ways : nat) → 
  number_of_ways = 13 := sorry

end plant_lamp_arrangements_total_arrangements_l180_180939


namespace interval_monotonicity_a3_g_extreme_points_range_a_l180_180756

noncomputable def f (x : ℝ) (a : ℝ) := 2 * x - 1 / x - a * Real.log x

noncomputable def g (x : ℝ) (a : ℝ) := f x a - x + 2 * a * Real.log x

theorem interval_monotonicity_a3 :
  (∀ x : ℝ, x > 0 → f x 3 > 0 → 0 < x → x < 1 / 2 ∨ 1 < x → (2 * x - 1 / x - 3 * Real.log x)’(x) > 0)
  ∧ (∀ x : ℝ, x > 0 → f x 3 < 0 → 1 / 2 < x → x < 1 → (2 * x - 1 / x - 3 * Real.log x)’(x) < 0) :=
sorry

theorem g_extreme_points_range_a :
  (∀ (x1 x2 : ℝ), x1 ≠ x2 → x1 + x2 =  -a → x1 * x2 = 1 → a^2 - 4 > 0 → a < -2) :=
sorry

end interval_monotonicity_a3_g_extreme_points_range_a_l180_180756


namespace maria_tom_weather_probability_l180_180486

noncomputable def probability_exactly_two_clear_days (p : ℝ) (n : ℕ) : ℝ :=
  (n.choose 2) * (p ^ (n - 2)) * ((1 - p) ^ 2)

theorem maria_tom_weather_probability :
  probability_exactly_two_clear_days 0.6 5 = 1080 / 3125 :=
by
  sorry

end maria_tom_weather_probability_l180_180486


namespace probability_not_adjacent_in_row_of_10_chairs_l180_180132

theorem probability_not_adjacent_in_row_of_10_chairs :
  let total_ways := Nat.choose 10 2 in
  let adjacent_pairs := 9 in
  let probability_adjacent := adjacent_pairs.toRat / total_ways.toRat in
  let probability_not_adjacent := 1 - probability_adjacent in
  probability_not_adjacent = 4 / 5 :=
by
  let total_ways := Nat.choose 10 2
  let adjacent_pairs := 9
  let probability_adjacent := adjacent_pairs.toRat / total_ways.toRat
  let probability_not_adjacent := 1 - probability_adjacent
  have h1 : total_ways = 45 := by rfl
  have h2 : probability_adjacent = (9 : ℚ) / 45 := by simp [adjacent_pairs, h1]
  have h3 : probability_adjacent = 1 / 5 := by norm_num [h2]
  have h4 : probability_not_adjacent = 1 - 1 / 5 := by simp [probability_adjacent, h3]
  have h5 : probability_not_adjacent = (4 : ℚ) / 5 := by norm_num [h4]
  exact h5

end probability_not_adjacent_in_row_of_10_chairs_l180_180132


namespace minimum_gennadies_l180_180610

theorem minimum_gennadies (A B V G : ℕ) (hA : A = 45) (hB : B = 122) (hV : V = 27) (hGap : G + A + V >= B - 1) :
  G >= 49 :=
by 
  have := by linarith [hGap, hA, hB, hV]
  exact this

end minimum_gennadies_l180_180610


namespace polynomial_irreducible_over_Z_iff_Q_l180_180936

theorem polynomial_irreducible_over_Z_iff_Q (f : Polynomial ℤ) :
  Irreducible f ↔ Irreducible (f.map (Int.castRingHom ℚ)) :=
sorry

end polynomial_irreducible_over_Z_iff_Q_l180_180936


namespace problem1_problem2_l180_180225

-- Problem 1
theorem problem1 :
  -1^4 + real.sqrt ((real.sqrt 3 - 2)^2) + 3 * real.tan (real.pi / 6) + (-1/3)^(-2) - (4^2022 * 0.25^2021) = 6 :=
sorry

-- Problem 2
theorem problem2 (a : ℝ)
  (h : a = real.sqrt 2 - real.floor (real.sqrt 2)) :
  (a^2 - 1) / (a^2 - a) / (2 + (a^2 + 1) / a) = real.sqrt 2 / 2 :=
sorry

end problem1_problem2_l180_180225


namespace math_problem_l180_180679

-- Define the individual numbers
def a : Int := 153
def b : Int := 39
def c : Int := 27
def d : Int := 21

-- Define the entire expression and its expected result
theorem math_problem : (a + b + c + d) * 2 = 480 := by
  sorry

end math_problem_l180_180679


namespace sec_tan_eq_l180_180836

theorem sec_tan_eq (x : ℝ) (h : Real.cos x ≠ 0) : 
  Real.sec x + Real.tan x = 7 / 3 → Real.sec x - Real.tan x = 3 / 7 :=
by
  intro h1
  sorry

end sec_tan_eq_l180_180836


namespace distance_between_points_l180_180425

open Real -- opening real number namespace

noncomputable def polar_to_cartesian (r θ : ℝ) : ℝ × ℝ :=
  (r * cos θ, r * sin θ)

theorem distance_between_points :
  let A := polar_to_cartesian 2 (π / 3)
  let B := polar_to_cartesian 2 (2 * π / 3)
  dist A B = 2 :=
by
  sorry

end distance_between_points_l180_180425


namespace find_k_l180_180350

theorem find_k (k : ℝ) (h : 0.5 * |-2 * k| * |k| = 1) : k = 1 ∨ k = -1 :=
sorry

end find_k_l180_180350


namespace probability_fx_lt_0_l180_180724

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x - 3

theorem probability_fx_lt_0 :
  (∫ x in -Real.pi..Real.pi, if f x < 0 then 1 else 0) / (2 * Real.pi) = 2 / Real.pi :=
by sorry

end probability_fx_lt_0_l180_180724


namespace part_a_part_b_l180_180209

def distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem part_a :
  let A := (0, -7)
  let B := (-4, 0)
  let origin := (0, 0)
  distance A origin > distance B origin :=
by
  let A := (0, -7)
  let B := (-4, 0)
  let origin := (0, 0)
  have d_A := distance A origin
  have d_B := distance B origin
  show d_A > d_B
  sorry

theorem part_b :
  let A := (0, -7)
  let B := (-4, 0)
  let C := (-4, -7)
  distance B C > distance A C :=
by
  let A := (0, -7)
  let B := (-4, 0)
  let C := (-4, -7)
  have d_CA := distance A C
  have d_CB := distance B C
  show d_CB > d_CA
  sorry

end part_a_part_b_l180_180209


namespace school_students_l180_180163

theorem school_students (x y : ℕ) (h1 : x + y = 432) (h2 : x - 16 = (y + 16) + 24) : x = 244 ∧ y = 188 := by
  sorry

end school_students_l180_180163


namespace find_XY_squared_l180_180451

/- Let ABC be an acute scalene triangle with circumcircle ω. 
The tangents to ω at B and C intersect at T. 
Let X and Y be the projections of T onto lines AB and AC, respectively. 
Suppose BT = CT = 18, BC = 24, and TX^2 + TY^2 + XY^2 = 1458. 
We need to prove XY^2 = 1386. -/

variables {A B C T X Y : Type}
variables [triangle A B C] [circumcircle ω A B C]
variables [tangent_to ω B T] [tangent_to ω C T]
variables [projection T X (line A B)] [projection T Y (line A C)]
variables {BT CT BC TX TY XY : ℝ}
variables (h1 : BT = 18) (h2 : CT = 18) (h3 : BC = 24) 
variables (h4 : TX^2 + TY^2 + XY^2 = 1458)

theorem find_XY_squared : XY^2 = 1386 := 
by 
  sorry

end find_XY_squared_l180_180451


namespace number_of_ordered_pairs_l180_180895

open Set

def satisfies_conditions (C D : Finset ℕ) : Prop :=
  (C ∪ D = (Finset.range 15).erase 0) ∧
  (C ∩ D = ∅) ∧
  (C.card ∉ C) ∧
  (D.card ∉ D)

theorem number_of_ordered_pairs : ∃ M : ℕ, M = 2100 ∧
  M = (Finset.powerset (Finset.range 15).erase 0).filter
    (λ C, satisfies_conditions C (Finset.range 15).erase 0 \ C).card :=
by
  sorry

end number_of_ordered_pairs_l180_180895


namespace min_gennadies_l180_180628

theorem min_gennadies 
  (n_Alexanders : ℕ) (n_Borises : ℕ) (n_Vasilies : ℕ) 
  (x_Gennadies : ℕ) 
  (h_Alexanders : n_Alexanders = 45) 
  (h_Borises   : n_Borises = 122) 
  (h_Vasilies  : n_Vasilies = 27) 
  (h_condition : ∀ p : ℕ, p = n_Borises - 1 → p = 121) 
  (h_total     : ∀ q : ℕ, q = 45 + 27 → q = 72)
  : x_Gennadies = 49 := 
sorry

end min_gennadies_l180_180628


namespace closest_integer_to_expression_l180_180149

theorem closest_integer_to_expression : 
  let expr := (3 / 2) * (4 / 9) + (7 / 2)
  in Int.closest expr = 4 := 
sorry

end closest_integer_to_expression_l180_180149


namespace circle_radius_tangent_to_four_given_circles_l180_180910

theorem circle_radius_tangent_to_four_given_circles (
  EF GH HE FG: ℕ) (r1 r2 r3 r4: ℕ)
  (h1: EFGH_is_isosceles_trapezoid EF GH HE FG)
  (h2: EF = 8) (h3: GH = 5) (h4: HE = 6) (h5: FG = 6)
  (h6: radius_centered_at E = 4) (h7: radius_centered_at F = 4)
  (h8: radius_centered_at G = 3) (h9: radius_centered_at H = 3)
  (internal_circle_radius: ℚ) 
  (radius_form : internal_circle_radius = \frac{-110 + 84 * Real.sqrt(6)}{29})
  : ∀ r a b c d (proof_eq : r = \frac{-a + b * Real.sqrt(c)}{d}), 
    (a + b + c + d = 229) := by
    sorry

end circle_radius_tangent_to_four_given_circles_l180_180910


namespace cleaning_time_ratio_l180_180430

noncomputable def trees_in_grove : Nat := 4 * 5
noncomputable def time_per_tree_without_help : Nat := 6
noncomputable def total_time_with_help : Nat := 60 (convert from 1 hour to minutes)

theorem cleaning_time_ratio :
  let trees := 4 * 5,
      time_per_tree_without_help := 6,
      total_time_with_help := 60 in
  let time_per_tree_with_help := total_time_with_help / trees in
  ratio (time_per_tree_with_help : ℚ) / (time_per_tree_without_help : ℚ) = 1 / 2 :=
by
  let trees := 4 * 5
  let time_per_tree_without_help := 6
  let total_time_with_help := 60
  let time_per_tree_with_help := total_time_with_help / trees
  
  -- We need to prove that the ratio of the times is 1/2
  have : (time_per_tree_with_help : ℚ) / (time_per_tree_without_help : ℚ) = 1 / 2 := sorry
  exact this

end cleaning_time_ratio_l180_180430


namespace solve_for_x_l180_180229

-- Define the given equation as a Lean function
def equation (x : ℕ) : ℕ := 90 + 5 * x / (180 / 3)

-- State the theorem to prove
theorem solve_for_x : ∃ x, equation x = 91 ↔ x = 12 :=
by {
  sorry,
}

end solve_for_x_l180_180229


namespace construct_triangle_ABX_l180_180572

-- Define the variables and conditions
def regular_octaon_in_circle (A B C D E F G H : Type) (radius : ℝ) (is_octagon : Prop) (inscribed_in_circle: Prop) : Prop :=
  radius = 5 ∧ is_octagon ∧ inscribed_in_circle

def triangle_ABC_with_orthocenter (A B X D : Type) (is_point_D_orthocenter : Prop) : Prop :=
  is_point_D_orthocenter

-- The theorem statement that constructs triangle ABX
theorem construct_triangle_ABX
  {A B C D E F G H X : Type}
  (h : regular_octaon_in_circle A B C D E F G H 5 (is_regular_octagon A B C D E F G H) (is_inscribed_in_circle A B C D E F G H 5))
  (h_orthocenter : triangle_ABC_with_orthocenter A B X D (is_orthocenter D A B X)) :
  ∃ X, is_orthocenter D A B X :=
by sorry

end construct_triangle_ABX_l180_180572


namespace sec_minus_tan_l180_180818

theorem sec_minus_tan
  (x : ℝ)
  (h1 : Real.sec x + Real.tan x = 7 / 3)
  (h2 : (Real.sec x + Real.tan x) * (Real.sec x - Real.tan x) = 1) :
  Real.sec x - Real.tan x = 3 / 7 :=
by
  sorry

end sec_minus_tan_l180_180818


namespace exponent_of_five_in_factorial_l180_180045

theorem exponent_of_five_in_factorial:
  (nat.factors 30!).count 5 = 7 :=
begin
  sorry
end

end exponent_of_five_in_factorial_l180_180045


namespace sec_sub_tan_l180_180797

theorem sec_sub_tan (x : ℝ) (h : sec x + tan x = 7 / 3) : sec x - tan x = 3 / 7 := by
  sorry

end sec_sub_tan_l180_180797


namespace fair_betting_scheme_fair_game_l180_180188

noncomputable def fair_game_stakes (L L_k : ℕ → ℚ) : Prop :=
  ∀ k: ℕ, 1 < k → L_k (k+1) = (35/36) * L_k k

theorem fair_betting_scheme_fair_game :
  ∃ L_k : ℕ → ℚ, fair_game_stakes (λ k, L_k k) (λ k, L_k k) :=
begin
  let L_k : ℕ → ℚ := λ k, (35 / 36) ^ k,
  use L_k,
  unfold fair_game_stakes,
  intros k hk,
  rw [mul_assoc, ←mul_pow],
  ring,
end

end fair_betting_scheme_fair_game_l180_180188


namespace nonagon_digit_assignments_l180_180122

/-- Considering a regular nonagon ABCDEFGHI with its center at J, 
each vertex and the center is to be associated with one of the digits 1 through 10 (each digit used once). 
We want to prove that the number of ways the digits can be assigned so that the sums of the numbers on the lines 
AJE, BJF, CJG, DJH, and EJI are all equal is 768. -/
theorem nonagon_digit_assignments :
  let digits := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} in
  let lines := [(A, J, E), (B, J, F), (C, J, G), (D, J, H), (E, J, I)] in
  let common_sum := sorry in
  let valid_assignments := sorry in
  let count_valid_assignments := sorry in
  count_valid_assignments = 768 := sorry

end nonagon_digit_assignments_l180_180122


namespace James_beat_record_by_72_l180_180431

-- Define the conditions as given in the problem
def touchdowns_per_game : ℕ := 4
def points_per_touchdown : ℕ := 6
def games_in_season : ℕ := 15
def conversions : ℕ := 6
def points_per_conversion : ℕ := 2
def old_record : ℕ := 300

-- Define the necessary calculations based on the conditions
def points_from_touchdowns_per_game : ℕ := touchdowns_per_game * points_per_touchdown
def points_from_touchdowns_in_season : ℕ := games_in_season * points_from_touchdowns_per_game
def points_from_conversions : ℕ := conversions * points_per_conversion
def total_points_in_season : ℕ := points_from_touchdowns_in_season + points_from_conversions
def points_above_old_record : ℕ := total_points_in_season - old_record

-- State the proof problem
theorem James_beat_record_by_72 : points_above_old_record = 72 :=
by
  sorry

end James_beat_record_by_72_l180_180431


namespace c_work_rate_l180_180212

noncomputable def work_rate (days : ℕ) : ℝ := 1 / days

theorem c_work_rate (A B C: ℝ) 
  (h1 : A + B = work_rate 28) 
  (h2 : A + B + C = work_rate 21) : C = work_rate 84 := by
  -- Proof will go here
  sorry

end c_work_rate_l180_180212


namespace sinA_sinC_B_value_l180_180396

variable (A B C a b c S : ℝ)

-- Conditions
axiom angle_condition : A + B + C = π
axiom side_opposite_angle : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0
axiom area_condition : S = b^2 / (3 * sin B)
axiom cos_condition : cos A * cos C = 1 / 6

-- Required to prove
theorem sinA_sinC : sin A * sin C = 2 / 3 :=
by
  sorry

theorem B_value : B = π / 3 :=
by
  sorry

end sinA_sinC_B_value_l180_180396


namespace min_gennadies_l180_180649

noncomputable section

def minGennadiesNeeded (alexanders borises vasilies : Nat) : Nat :=
  let needed_gaps := borises - 1
  let total_others := alexanders + vasilies
  if needed_gaps > total_others then needed_gaps - total_others else 0

theorem min_gennadies (alexanders borises vasilies : Nat) (h_alex: alexanders = 45) (h_boris: borises = 122) (h_vasil: vasilies = 27):
  minGennadiesNeeded alexanders borises vasilies = 49 := by
  rw [h_alex, h_boris, h_vasil]
  simp [minGennadiesNeeded]
  sorry

end min_gennadies_l180_180649


namespace number_of_i_with_b_i_2016_l180_180954

def a_n (n : ℕ) : ℕ := 2^(n-1)

def b_n (n : ℕ) : ℕ :=
  (Finset.filter (λ m, a_n m < n) (Finset.range (n+1))).card

theorem number_of_i_with_b_i_2016 :
  (Finset.filter (λ i, b_n i = 2016) (Finset.range (2^(2016) + 1))).card = 2^2015 :=
by sorry

end number_of_i_with_b_i_2016_l180_180954


namespace sasha_initial_questions_l180_180473

theorem sasha_initial_questions :
  (∀ (q_per_hour q_remaining q_total: ℕ), 
    q_per_hour = 15 → 
    q_total = 30 → 
    q_remaining = 30 → 
    q_per_hour * 2 + q_remaining = q_total + q_remaining) →
  (∃ (q_initial : ℕ), q_initial = 60) :=
by
  intro h
  use 60
  sorry

end sasha_initial_questions_l180_180473


namespace exists_fibonacci_ending_in_six_9s_l180_180429

def fibonacci : ℤ → ℤ
| 0       := 1
| 1       := 1
| (n + 2) := fibonacci (n + 1) + fibonacci n
| -1      := 0
| -2      := 1
| (n - 1) :=
  have h : (n + 1) < n - 1 + 2, from int.add_lt_add_left (int.of_nat_lt (nat.succ_lt_succ (nat.succ_pos n))) ↑(-1),
  -(fibonacci (n - 2) - fibonacci (n - 1))

theorem exists_fibonacci_ending_in_six_9s : 
  ∃ i : ℕ, fibonacci i % 1000000 = 999999 :=
sorry

end exists_fibonacci_ending_in_six_9s_l180_180429


namespace find_k_l180_180245

-- Defining the points
def point1 : ℕ × ℕ := (0, 7)
def point3 : ℕ × ℕ := (20, 3)

-- Function to calculate the slope between two points
def slope (p1 p2 : ℕ × ℕ) : ℚ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

-- Given points
def point2 (k : ℕ) : ℕ × ℕ := (15, k)

-- The main theorem to prove
theorem find_k (k : ℕ) :
  slope point1 (point2 k) = slope (point2 k) point3 → k = 4 :=
by
  -- Proof outline: Solve for k from slope equality
  sorry

end find_k_l180_180245


namespace area_above_x_axis_of_parallelogram_is_four_fifths_l180_180932

structure Point where
  x : ℝ
  y : ℝ
  
noncomputable def probability_above_x_axis (A B C D : Point) (x_axis : ℝ → Prop) : ℝ :=
  let area_parallelogram := 21 * 5
  let area_above_x := 21 * 4
  area_above_x / area_parallelogram

theorem area_above_x_axis_of_parallelogram_is_four_fifths :
  probability_above_x_axis ⟨4, 5⟩ ⟨-2, 1⟩ ⟨-8, 1⟩ ⟨-2, 5⟩ (λ x, x = 0) = 4 / 5 :=
by
  sorry

end area_above_x_axis_of_parallelogram_is_four_fifths_l180_180932


namespace total_votes_all_proposals_l180_180993

theorem total_votes_all_proposals : 
  ∀ (A B C : ℝ),
  (A = 0.4 * (A + (A + 70))) ∧ (B = 0.35 * (B + (B + 120))) ∧ (C = 0.3 * (C + (C + 150))) →
  A + (A + 70) + B + (B + 120) + C + (C + 150) = 1126 := 
by 
  intros A B C h,
  sorry

end total_votes_all_proposals_l180_180993


namespace maggie_hourly_wage_l180_180923

variables (x : ℕ) {hours_office hours_tractor hourly_wage_office total_income : ℕ}

theorem maggie_hourly_wage 
  (hourly_wage_office : ℕ) (hours_tractor : ℕ) (hours_office : ℕ) (total_income : ℕ) (h1 : hourly_wage_office = 10)
  (h2 : hours_office = 2 * hours_tractor) (h3 : total_income = 416) 
  (h4 : hours_tractor = 13) : x = 12 :=
begin
  sorry
end

end maggie_hourly_wage_l180_180923


namespace sec_minus_tan_l180_180821

theorem sec_minus_tan
  (x : ℝ)
  (h1 : Real.sec x + Real.tan x = 7 / 3)
  (h2 : (Real.sec x + Real.tan x) * (Real.sec x - Real.tan x) = 1) :
  Real.sec x - Real.tan x = 3 / 7 :=
by
  sorry

end sec_minus_tan_l180_180821


namespace simplify_fraction_l180_180127

theorem simplify_fraction :
  (144 : ℤ) / (1296 : ℤ) = 1 / 9 := 
by sorry

end simplify_fraction_l180_180127


namespace cylinder_cone_volume_l180_180240

theorem cylinder_cone_volume (V_total : ℝ) (Vc Vcone : ℝ)
  (h1 : V_total = 48)
  (h2 : V_total = Vc + Vcone)
  (h3 : Vc = 3 * Vcone) :
  Vc = 36 ∧ Vcone = 12 :=
by
  sorry

end cylinder_cone_volume_l180_180240


namespace minimum_gennadies_l180_180613

theorem minimum_gennadies (A B V G : ℕ) (hA : A = 45) (hB : B = 122) (hV : V = 27) (hGap : G + A + V >= B - 1) :
  G >= 49 :=
by 
  have := by linarith [hGap, hA, hB, hV]
  exact this

end minimum_gennadies_l180_180613


namespace draw_8_cards_ensure_even_product_l180_180551

def integers_on_cards : Finset ℕ := Finset.range 15 \ {0}

def draw_without_replacement {α : Type*} (s : Finset α) : List α → Finset α := sorry

def product_is_even (lst : List ℕ) : Prop := (Finset.prod (Finset.of_list lst)) % 2 = 0

theorem draw_8_cards_ensure_even_product :
  ∀ (drawn : List ℕ), 
    drawn.length = 8 → 
    drawn ⊆ integers_on_cards → 
    product_is_even drawn :=
sorry

end draw_8_cards_ensure_even_product_l180_180551


namespace brother_and_sister_ages_l180_180671

theorem brother_and_sister_ages :
  ∃ (b s : ℕ), (b - 3 = 7 * (s - 3)) ∧ (b - 2 = 4 * (s - 2)) ∧ (b - 1 = 3 * (s - 1)) ∧ (b = 5 / 2 * s) ∧ b = 10 ∧ s = 4 :=
by 
  sorry

end brother_and_sister_ages_l180_180671


namespace smaller_tablet_diagonal_l180_180983

theorem smaller_tablet_diagonal :
  ∀ (A_large A_small : ℝ)
    (d : ℝ),
    A_large = (8 / Real.sqrt 2) ^ 2 →
    A_small = (d / Real.sqrt 2) ^ 2 →
    A_large = A_small + 7.5 →
    d = 7
:= by
  intros A_large A_small d h1 h2 h3
  sorry

end smaller_tablet_diagonal_l180_180983


namespace sqrt7_sub_m_div_n_gt_inv_mn_l180_180092

variables (m n : ℤ)
variables (h_m_nonneg : m ≥ 1) (h_n_nonneg : n ≥ 1)
variables (h_ineq : Real.sqrt 7 - (m : ℝ) / (n : ℝ) > 0)

theorem sqrt7_sub_m_div_n_gt_inv_mn : 
  Real.sqrt 7 - (m : ℝ) / (n : ℝ) > 1 / ((m : ℝ) * (n : ℝ)) :=
by
  sorry

end sqrt7_sub_m_div_n_gt_inv_mn_l180_180092


namespace height_of_model_l180_180101

noncomputable def calc_height_of_model (volume_actual : ℝ) (volume_model : ℝ) (height_actual : ℝ) : ℝ :=
  let volume_ratio := volume_actual / volume_model
  let scale_factor := volume_ratio ** (1 / 3)
  height_actual / scale_factor

theorem height_of_model : calc_height_of_model 100000 0.2 40 = 0.5 :=
by
  unfold calc_height_of_model
  sorry

end height_of_model_l180_180101


namespace real_part_of_product_l180_180729

variable (α β : ℝ)
def z1 : ℂ := complex.mk (real.cos α) (real.sin α)
def z2 : ℂ := complex.mk (real.cos β) (real.sin β)

theorem real_part_of_product (α β : ℝ) :
  (z1 α β * z2 α β).re = real.cos (α + β) :=
by
  sorry

end real_part_of_product_l180_180729


namespace total_gold_value_l180_180884

def legacy_bars : ℕ := 5
def aleena_bars : ℕ := legacy_bars - 2
def value_per_bar : ℕ := 2200
def total_bars : ℕ := legacy_bars + aleena_bars
def total_value : ℕ := total_bars * value_per_bar

theorem total_gold_value : total_value = 17600 :=
by
  -- Begin proof
  sorry

end total_gold_value_l180_180884


namespace exponent_of_5_in_30_fact_l180_180029

def count_powers_of_5 (n : ℕ) : ℕ :=
  if n < 5 then 0
  else n / 5 + count_powers_of_5 (n / 5)

theorem exponent_of_5_in_30_fact : count_powers_of_5 30 = 7 := 
  by
    sorry

end exponent_of_5_in_30_fact_l180_180029


namespace sec_minus_tan_l180_180817

theorem sec_minus_tan
  (x : ℝ)
  (h1 : Real.sec x + Real.tan x = 7 / 3)
  (h2 : (Real.sec x + Real.tan x) * (Real.sec x - Real.tan x) = 1) :
  Real.sec x - Real.tan x = 3 / 7 :=
by
  sorry

end sec_minus_tan_l180_180817


namespace expected_value_ξ_l180_180916

-- Define the set A based on the condition
def A : set ℤ := {x : ℤ | x ∈ [-4, -3, -2, -1, 0, 1, 2] }

-- Define the random variable ξ = x^2 where x is from A
def ξ (x : ℤ) : ℤ := x^2

-- Define the probability distribution of ξ
noncomputable def P (n : ℤ) : ℝ :=
if n = 16 ∨ n = 9 then 1/7
else if n = 4 ∨ n = 1 then 2/7
else if n = 0 then 1/7
else 0

-- Define the expected value E(ξ)
noncomputable def E : ℝ := (0 * 1/7) + (1 * 2/7) + (4 * 2/7) + (9 * 1/7) + (16 * 1/7)

-- Statement of the proof problem
theorem expected_value_ξ : E = 5 := by
  sorry

end expected_value_ξ_l180_180916


namespace area_of_rhombus_formed_by_square_midpoints_l180_180573

theorem area_of_rhombus_formed_by_square_midpoints 
    (side_length : ℝ) 
    (H_square_side : side_length = 4) : 
    ∃ area : ℝ, area = 8 :=
by
  have d1 : ℝ := side_length,
  have d2 : ℝ := side_length,
  have area : ℝ := (d1 * d2) / 2,
  use area,
  rw [H_square_side],
  norm_num,
  sorry

end area_of_rhombus_formed_by_square_midpoints_l180_180573


namespace minimum_gennadies_l180_180617

theorem minimum_gennadies (A B V G : ℕ) (hA : A = 45) (hB : B = 122) (hV : V = 27) (hGap : G + A + V >= B - 1) :
  G >= 49 :=
by 
  have := by linarith [hGap, hA, hB, hV]
  exact this

end minimum_gennadies_l180_180617


namespace non_congruent_triangles_with_perimeter_15_l180_180378

theorem non_congruent_triangles_with_perimeter_15 :
  { t : Finset (Finset ℕ) //
    ∀ x ∈ t, ∃ (a b c : ℕ), 
      x = {a, b, c} ∧ a + b + c = 15 ∧ 
      a < b + c ∧ b < a + c ∧ c < a + b 
  }.val.card = 7 :=
by
  sorry

end non_congruent_triangles_with_perimeter_15_l180_180378


namespace value_of_a_probability_one_high_sales_day_l180_180236

-- Definition of Conditions
def sales_distribution (x : ℕ) (n : ℤ) (a : ℝ) : ℝ :=
  if (50 ≤ x ∧ x < 100) then
    if (10 * n ≤ x ∧ x < 10 * (n + 1) ∧ even n) then 
      n / 10 - 0.5
    else if (10 * n ≤ x ∧ x < 10 * (n + 1) ∧ odd n) then 
      n / 20 - a
    else 0
  else 0

-- Part I: Prove the value of a
theorem value_of_a : 
  (∃ (a : ℝ), (∀ (n : ℤ), 5 ≤ n ∧ n ≤ 9 →
      sales_distribution 60 n a + sales_distribution 70 n a + 
      sales_distribution 80 n a + sales_distribution 90 n a =
      1)) → a = 0.15 :=
sorry

-- Part II: Probability of exactly one high-sales day
theorem probability_one_high_sales_day :
  let high_sales_frequency := 0.6 in -- sum of high sales frequency
  let low_sales_frequency := 0.4 in -- sum of low sales frequency
  let total_days := 50 in
  let selected_days := 5 in
  let high_sales_days := 3 in
  let low_sales_days := 2 in
  let all_possible_outcomes := (selected_days choose 2) in
  let favorable_outcomes := (low_sales_days * high_sales_days) in
  favorable_outcomes / all_possible_outcomes = 3 / 5 :=
sorry

end value_of_a_probability_one_high_sales_day_l180_180236


namespace XY_squared_l180_180449

-- Define the triangle ABC, tangents, and specified conditions.
variables {A B C T X Y : Type}
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited T] [Inhabited X] [Inhabited Y]

-- Define the given conditions as assumptions.
axiom triangle_acute_scalene (△ABC : A) (ω : Type) : Prop
axiom tangents_intersect (T : Type) (ω : Type) (B C : A) : Prop
axiom projections_to_lines (T : Type) (X : A) (Y : A) (AB AC : Type) : Prop
axiom BT_CT_constant (BT CT : Type) : BT = CT := 18
axiom BC_constant (BC : Type) : BC := 24
axiom sum_of_squares (TX TY XY : Type) : (TX^2 + TY^2 + XY^2) = 1458

-- Define the main theorem
theorem XY_squared : XY^2 = 858 :=
by
  -- Assuming the given conditions
  assume (h1 : triangle_acute_scalene △ABC ω),
  assume (h2 : tangents_intersect T ω B C),
  assume (h3 : projections_to_lines T X Y AB AC),
  assume (h4 : BT_CT_constant BT CT),
  assume (h5 : BC_constant BC),
  assume (h6 : sum_of_squares TX TY XY),
  -- Proof will be provided here
  sorry

end XY_squared_l180_180449


namespace minimum_value_2x_3y_l180_180742

theorem minimum_value_2x_3y (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (hxy : x^2 * y * (4 * x + 3 * y) = 3) :
  2 * x + 3 * y ≥ 2 * Real.sqrt 3 := by
  sorry

end minimum_value_2x_3y_l180_180742


namespace exponent_of_5_in_30_fact_l180_180026

def count_powers_of_5 (n : ℕ) : ℕ :=
  if n < 5 then 0
  else n / 5 + count_powers_of_5 (n / 5)

theorem exponent_of_5_in_30_fact : count_powers_of_5 30 = 7 := 
  by
    sorry

end exponent_of_5_in_30_fact_l180_180026


namespace probability_selecting_girl_l180_180258

def boys : ℕ := 3
def girls : ℕ := 1
def total_candidates : ℕ := boys + girls
def favorable_outcomes : ℕ := girls

theorem probability_selecting_girl : 
  ∃ p : ℚ, p = (favorable_outcomes : ℚ) / (total_candidates : ℚ) ∧ p = 1 / 4 :=
sorry

end probability_selecting_girl_l180_180258


namespace equal_angles_imply_equal_sides_l180_180116

theorem equal_angles_imply_equal_sides (P : Type) [MetricSpace P] (circle : P → Prop)
  (pentagon : Fin 5 → P) 
  (inscribed : ∀ i, circle (pentagon i)) 
  (equal_angles : ∀ i j k, i ≠ j → j ≠ k → k ≠ i → ∠(pentagon i) (pentagon j) (pentagon k) = π / 5) : 
  (∀ i j, i ≠ j → dist (pentagon i) (pentagon (i + 1)) = dist (pentagon j) (pentagon (j + 1))) := 
by
  sorry

end equal_angles_imply_equal_sides_l180_180116


namespace exponent_of_five_in_factorial_l180_180050

theorem exponent_of_five_in_factorial:
  (nat.factors 30!).count 5 = 7 :=
begin
  sorry
end

end exponent_of_five_in_factorial_l180_180050


namespace parallel_line_slope_l180_180689

theorem parallel_line_slope (a b c : ℝ) (m : ℝ) :
  (5 * a + 10 * b = -35) →
  (∃ m : ℝ, b = m * a + c) →
  m = -1/2 :=
by sorry

end parallel_line_slope_l180_180689


namespace set_of_creases_is_on_or_outside_ellipse_l180_180104

noncomputable def ellipse_creases (O : Point) (R : ℝ) (A : Point) (a : ℝ) : Set Point :=
  { P | ∃ (A' : Point), distance O A' = R ∧ 
    let MN := perpendicular_bisector (segment A A') in 
    let P := intersection (line_segment O A') MN in 
    distance O A + distance A P = R
  }

theorem set_of_creases_is_on_or_outside_ellipse 
  (O A : Point) (R a : ℝ) 
  (hR : 0 < R) 
  (ha : 0 < a < R)
  : 
  (⋃ A', distance O A' = R → { Q | Q ∈ line_of_crease A A' }) =
  (ellipse_creases O R A a) :=
sorry

end set_of_creases_is_on_or_outside_ellipse_l180_180104


namespace cylindrical_coordinates_of_point_l180_180299

theorem cylindrical_coordinates_of_point :
  ∀ (x y z : ℝ), x = 3 → y = -3 * Real.sqrt 3 → z = 2 →
  ∃ r θ, r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ r = 6 ∧ θ = 5 * Real.pi / 3 ∧ z = 2 :=
by
  intros x y z hx hy hz
  use 6, (5 * Real.pi / 3)
  split
  { exact zero_lt_six },
  split
  { norm_num },
  split
  { norm_num },
  split
  { exact hx.symm ▸ hy.symm ▸ rfl },
  split
  { exact hz },
  sorry

end cylindrical_coordinates_of_point_l180_180299


namespace find_T_n_find_S_n_l180_180085

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, b (n + 1) = b n * r

noncomputable def T_n (b : ℕ → ℝ) : ℕ → ℝ
| 0       => 0
| (n + 1) => T_n n + b (n + 1)

noncomputable def S_n (a : ℕ → ℝ) : ℕ → ℝ
| 0       => 0
| 1       => 1
| (n + 1) => S_n n + |a (n + 1)|

-- Main proof statements
theorem find_T_n (a b : ℕ → ℝ) 
  (ha : arithmetic_sequence a) (hb : geometric_sequence b)
  (ha1 : a 1 = 1) (hb1 : b 1 = 1)
  (h_condition : sqrt (a 2 + 2) + sqrt (b 2 - 2) = 2 * sqrt 2)
  (h_min : a 2 + b 2 = 4):
  T_n b = λ n, (1 / 3) * (4^n - 1) :=
sorry

theorem find_S_n (a : ℕ → ℝ)
  (ha : arithmetic_sequence a)
  (ha1 : a 1 = 1)
  (h_condition : ∀ n ≥ 2, |a n| = n - 2):
  S_n a = λ n, if n = 1 then 1 else (n^2 - 3*n + 4) / 2 :=
sorry

end find_T_n_find_S_n_l180_180085


namespace max_value_ellipse_l180_180312

def condition (x y : ℝ) : Prop :=
  (x^2) / 9 + (y^2) / 4 = 1

theorem max_value_ellipse :
  ∀ x y : ℝ, condition x y → 2 * x - y ≤ 2 * real.sqrt 10 := sorry

end max_value_ellipse_l180_180312


namespace total_cost_l180_180535

noncomputable def cost_sandwich : ℝ := 2.44
noncomputable def quantity_sandwich : ℕ := 2
noncomputable def cost_soda : ℝ := 0.87
noncomputable def quantity_soda : ℕ := 4

noncomputable def total_cost_sandwiches : ℝ := cost_sandwich * quantity_sandwich
noncomputable def total_cost_sodas : ℝ := cost_soda * quantity_soda

theorem total_cost (total_cost_sandwiches total_cost_sodas : ℝ) : (total_cost_sandwiches + total_cost_sodas = 8.36) :=
by
  sorry

end total_cost_l180_180535


namespace felicia_flour_amount_l180_180699

-- Define the conditions as constants
def white_sugar := 1 -- cups
def brown_sugar := 1 / 4 -- cups
def oil := 1 / 2 -- cups
def scoop := 1 / 4 -- cups
def total_scoops := 15 -- number of scoops

-- Define the proof statement
theorem felicia_flour_amount : 
  (total_scoops * scoop - (white_sugar + brown_sugar / scoop + oil / scoop)) * scoop = 2 :=
by
  sorry

end felicia_flour_amount_l180_180699


namespace equal_angles_imply_equal_sides_l180_180117

theorem equal_angles_imply_equal_sides (P : Type) [MetricSpace P] (circle : P → Prop)
  (pentagon : Fin 5 → P) 
  (inscribed : ∀ i, circle (pentagon i)) 
  (equal_angles : ∀ i j k, i ≠ j → j ≠ k → k ≠ i → ∠(pentagon i) (pentagon j) (pentagon k) = π / 5) : 
  (∀ i j, i ≠ j → dist (pentagon i) (pentagon (i + 1)) = dist (pentagon j) (pentagon (j + 1))) := 
by
  sorry

end equal_angles_imply_equal_sides_l180_180117


namespace number_of_terminal_zeros_l180_180556

theorem number_of_terminal_zeros (n1 n2 n3 : ℕ) (h1 : n1 = 50) (h2 : n2 = 720) (h3 : n3 = 125) : 
  nat.min ((nat.factorial_count 2 n1) + (nat.factorial_count 2 n2) + (nat.factorial_count 2 n3)) 
          ((nat.factorial_count 5 n1) + (nat.factorial_count 5 n2) + (nat.factorial_count 5 n3)) = 5 :=
by {
  sorry
}

end number_of_terminal_zeros_l180_180556


namespace midpoints_collinear_l180_180867

-- Basic geometric definitions for the proof setup
variables {Point : Type} [metric_space Point]
variables {A B C D A' B' : Point}
variables {m₁ m₂ m₃ : Point}

-- Definitions for the problem conditions
def is_trapezoid (A B C D : Point) : Prop :=
  (dist A B) = (dist C D) ∧ parallel (line_through A B) (line_through C D)

def is_rotation (C A B A' B' : Point) : Prop :=
  (dist C B) = (dist C B') ∧
  ∃ θ : ℝ, θ ≠ 0 ∧
  rotation_around C θ A = A' ∧
  rotation_around C θ B = B'

def midpoint (P Q : Point) : Point :=
  (P + Q) / 2

-- The proposed theorem in Lean 4 statement
theorem midpoints_collinear 
  (A B C D A' B' m₁ m₂ m₃ : Point)
  (h1 : is_trapezoid A B C D)
  (h2 : is_rotation C A B A' B')
  (h3 : m₁ = midpoint A' D)
  (h4 : m₂ = midpoint B C)
  (h5 : m₃ = midpoint B' C) : collinear {m₁, m₂, m₃} :=
by {
  sorry
}

end midpoints_collinear_l180_180867


namespace standard_equation_of_ellipse_range_area_triangle_l180_180736

-- Definition of the ellipse and its properties
def ellipse (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) (h : b < a) :=
  ∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1

-- Point P on the ellipse
def P_on_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x = -1 ∧ y = sqrt 2 / 2

-- Line segment intersection properties
def midpoint_condition (P M F₂ : ℝ × ℝ) : Prop :=
  M = (P.1 + F₂.1) / 2 ∧ (P.2 + F₂.2) / 2

-- First part: Proving the standard equation
theorem standard_equation_of_ellipse : ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ b < a ∧
  ellipse a b ∧ P_on_ellipse a b (-1) (sqrt 2 / 2) ∧ 
  (∃ x y : ℝ, ((x / sqrt 2)^2) + y^2 = 1) :=
sorry

-- Second part: Range for the area of triangle
def intersecting_line (t : ℝ) := λ y, t * y + 1

theorem range_area_triangle (λ : ℝ) (h : λ ∈ set.Icc (2/3) 1) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ b < a ∧ 
  ellipse a b ∧ P_on_ellipse a b (-1) (sqrt 2 / 2) ∧ 
  (∀ t : ℝ, (1/3 ≤ t^2 ∧ t^2 ≤ 1/2) → 
   let S := sqrt (8*(t^2 + 1) / ((t^2 + 2)^2)) 
   in S ∈ set.Icc (4*sqrt 3 / 5) (4*sqrt 6 / 7)) :=
sorry

end standard_equation_of_ellipse_range_area_triangle_l180_180736


namespace sheela_monthly_income_l180_180216

theorem sheela_monthly_income (d : ℝ) (p : ℝ) (income : ℝ) (h1 : d = 4500) (h2 : p = 0.28) (h3 : d = p * income) : 
  income = 16071.43 :=
by
  sorry

end sheela_monthly_income_l180_180216


namespace equal_angle_implies_equal_side_not_equal_side_implies_not_equal_angle_l180_180467

theorem equal_angle_implies_equal_side (ABC A'B'C' : Triangle) (H : Point) (A' B' C' : Point) 
  (hA' : Reflect(H, ABC.sideBC) = A') (hB' : Reflect(H, ABC.sideCA) = B') (hC' : Reflect(H, ABC.sideAB) = C')
  (eq_angle : ∃ θ, ABC.angleA = θ ∧ A'B'C'.angleA = θ) :
  (∃ s, ABC.sideBC = s ∧ A'B'C'.sideBC = s) :=
by
  sorry

theorem not_equal_side_implies_not_equal_angle (ABC A'B'C' : Triangle) (H : Point) (A' B' C' : Point) 
  (hA' : Reflect(H, ABC.sideBC) = A') (hB' : Reflect(H, ABC.sideCA) = B') (hC' : Reflect(H, ABC.sideAB) = C')
  (eq_side : ABC.sideBC = A'B'C'.sideBC) :
  ¬(∃ θ, ABC.angleA = θ ∧ A'B'C'.angleA = θ) :=
by
  sorry

end equal_angle_implies_equal_side_not_equal_side_implies_not_equal_angle_l180_180467


namespace train_length_approx_140_l180_180584

noncomputable def km_per_hr_to_m_per_s (speed_km_hr : ℝ) : ℝ :=
  speed_km_hr * 1000 / 3600

def train_length (speed_km_hr : ℝ) (time_sec : ℝ) : ℝ :=
  let speed_m_s := km_per_hr_to_m_per_s speed_km_hr
  speed_m_s * time_sec

theorem train_length_approx_140 :
  train_length 56 9 ≈ 140 :=
by
  -- The proof can be filled in here
  sorry

end train_length_approx_140_l180_180584


namespace bacteria_doubling_time_l180_180980

noncomputable def doubling_time (N₀ N : ℕ) (t : ℝ) :=
  t / (Real.log (N / N₀) / Real.log 2)

theorem bacteria_doubling_time :
  doubling_time 1000 500000 44.82892142331043 ≈ 5 :=
by
  sorry

end bacteria_doubling_time_l180_180980


namespace cyclic_sum_inequality_l180_180079

variables (a b c : ℕ)
variable (h : 16 * (a + b + c) ≥ 1 / a + 1 / b + 1 / c)

theorem cyclic_sum_inequality (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∑ cyc ((1 : ℝ) / (a + b + real.sqrt (2 * a + 2 * c))) ^ 3 ≤ 8 / 9 :=
sorry

end cyclic_sum_inequality_l180_180079


namespace each_friend_pays_equally_l180_180694

-- Define the costs of the items
def taco_salad_cost := 10
def dave_single_cost := 5
def fries_cost := 2.5
def lemonade_cost := 2
def apple_pecan_cost := 6
def frosty_cost := 3

-- Number of items purchased
def dave_single_count := 5
def fries_count := 4
def lemonade_count := 5
def apple_pecan_count := 3
def frosty_count := 4

-- Number of friends
def num_friends := 8

-- Calculate individual categories total cost
def total_dave_single_cost := dave_single_cost * dave_single_count
def total_fries_cost := fries_cost * fries_count
def total_lemonade_cost := lemonade_cost * lemonade_count
def total_apple_pecan_cost := apple_pecan_cost * apple_pecan_count
def total_frosty_cost := frosty_cost * frosty_count

-- Calculate total cost
def total_cost := taco_salad_cost + total_dave_single_cost + total_fries_cost + total_lemonade_cost + total_apple_pecan_cost + total_frosty_cost

-- Calculate each friend's share
def each_friend_share := total_cost / num_friends

-- The goal statement to prove
theorem each_friend_pays_equally : each_friend_share = 10.63 :=
by
  -- Sorry is placed here to skip the proof steps
  sorry

end each_friend_pays_equally_l180_180694


namespace area_of_right_triangle_ABC_l180_180099

noncomputable def triangle_area {α : Type*} [field α] (x : α) (y : α) : α :=
  (1 / 2) * (x * y)

theorem area_of_right_triangle_ABC :
  ∃ (a b : ℝ), 
  let c_x := -a - b,
      c_y := -a - b - 1 in
  (a ≠ b) ∧
  ((b - a) ^ 2 + (b + 1 - a) ^ 2 = 900) ∧
  ((2 * a)^2 + (2 * (a + 0.5))^2 = 900) ∧
  triangle_area (a - c_x) (a - c_y) = 448 :=
begin
  sorry -- proof would go here
end

end area_of_right_triangle_ABC_l180_180099


namespace derivative_ge_two_range_of_a_l180_180097

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

theorem derivative_ge_two : ∀ x : ℝ, f' x ≥ 2 := by
  sorry

theorem range_of_a (a : ℝ) : (∀ x ≥ 0, f x ≥ a * x) ↔ (a ∈ set.Iic (2 : ℝ)) := by
  sorry


end derivative_ge_two_range_of_a_l180_180097


namespace total_weight_of_boxes_in_stack_l180_180410

theorem total_weight_of_boxes_in_stack :
  let individual_weight := 25 in
  let num_boxes := 1 * 3 + 4 * 2 + 3 * 1 in
  let total_weight := num_boxes * individual_weight in
  total_weight = 350 :=
by
  let individual_weight := 25
  let num_boxes := 1 * 3 + 4 * 2 + 3 * 1
  let total_weight := num_boxes * individual_weight
  have h1: num_boxes = 14 := by norm_num
  have h2: total_weight = 14 * 25 := by rw h1; norm_num
  norm_num at h2
  exact h2

end total_weight_of_boxes_in_stack_l180_180410


namespace exponent_of_five_in_factorial_l180_180052

theorem exponent_of_five_in_factorial:
  (nat.factors 30!).count 5 = 7 :=
begin
  sorry
end

end exponent_of_five_in_factorial_l180_180052


namespace exponent_of_five_in_30_factorial_l180_180043

theorem exponent_of_five_in_30_factorial : 
  nat.factorial_prime_exponent 30 5 = 7 := 
sorry

end exponent_of_five_in_30_factorial_l180_180043


namespace total_sweaters_knit_l180_180435

-- Definitions from condition a)
def monday_sweaters : ℕ := 8
def tuesday_sweaters : ℕ := monday_sweaters + 2
def wednesday_sweaters : ℕ := tuesday_sweaters - 4
def thursday_sweaters : ℕ := wednesday_sweaters
def friday_sweaters : ℕ := monday_sweaters / 2

-- Theorem statement
theorem total_sweaters_knit : 
  monday_sweaters + tuesday_sweaters + wednesday_sweaters + thursday_sweaters + friday_sweaters = 34 :=
  by
    sorry

end total_sweaters_knit_l180_180435


namespace part_I_part_II_l180_180368

noncomputable def f (x : ℝ) : ℝ := x^3 - 9 * x

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 + a

theorem part_I (a : ℝ) :
  let l := λ x, -9 * x in
  (∃ (m : ℝ), 6 * m = -9 ∧ g m a = l m) ↔ a = 27 / 4 := sorry

theorem part_II (a : ℝ) :
  -27 < a ∧ a < 5 ↔ ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ = g x₁ a ∧ f x₂ = g x₂ a ∧ f x₃ = g x₃ a := sorry

end part_I_part_II_l180_180368


namespace log_limit_l180_180275

open Real

theorem log_limit:
  (tendsto (λ x, log 5 (8*x - 3) - log 5 (4*x + 7)) at_top (nhds (log 5 2))) :=
sorry

end log_limit_l180_180275


namespace min_number_of_gennadys_l180_180658

theorem min_number_of_gennadys (a b v g : ℕ) (h_a : a = 45) (h_b : b = 122) (h_v : v = 27)
    (h_needed_g : g = 49) :
    (b - 1) - (a + v) = g :=
by
  -- We include sorry because we are focusing on the statement, not the proof itself.
  sorry

end min_number_of_gennadys_l180_180658


namespace min_gennadys_l180_180638

theorem min_gennadys (alexanders borises vasilies : ℕ) (x : ℕ) 
    (h1 : alexanders = 45)
    (h2 : borises = 122)
    (h3 : vasilies = 27)
    (h4 : x = 49)
    (h5 : borises - 1 = alexanders + vasilies + x) :
  x = 49 := 
begin
  sorry,
end

end min_gennadys_l180_180638


namespace number_of_books_about_trains_l180_180300

theorem number_of_books_about_trains
  (books_animals : ℕ)
  (books_outer_space : ℕ)
  (book_cost : ℕ)
  (total_spent : ℕ)
  (T : ℕ)
  (hyp1 : books_animals = 8)
  (hyp2 : books_outer_space = 6)
  (hyp3 : book_cost = 6)
  (hyp4 : total_spent = 102)
  (hyp5 : total_spent = (books_animals + books_outer_space + T) * book_cost)
  : T = 3 := by
  sorry

end number_of_books_about_trains_l180_180300


namespace ivy_covering_the_tree_l180_180285

def ivy_stripped_per_day := 6
def ivy_grows_per_night := 2
def days_to_strip := 10
def net_ivy_stripped_per_day := ivy_stripped_per_day - ivy_grows_per_night

theorem ivy_covering_the_tree : net_ivy_stripped_per_day * days_to_strip = 40 := by
  have h1 : net_ivy_stripped_per_day = 4 := by
    unfold net_ivy_stripped_per_day
    rfl
  rw [h1]
  show 4 * 10 = 40
  rfl

end ivy_covering_the_tree_l180_180285


namespace short_bingo_columns_possibilities_l180_180411

theorem short_bingo_columns_possibilities :
  let possible_values := finset.range 15
  ∃ first_set : finset ℕ, 
    first_set.card = 5 ∧ 
    ∀ x ∈ first_set, x ∈ possible_values ∧ 
    ∀ y ∈ possible_values, y ∉ first_set → 
    15 * 14 * 13 * 12 * 11 = 360360 :=
begin
  sorry
end

end short_bingo_columns_possibilities_l180_180411


namespace num_houses_with_digit_7_in_range_l180_180577

-- Define the condition for a number to contain a digit 7
def contains_digit_7 (n : Nat) : Prop :=
  (n / 10 = 7) || (n % 10 = 7)

-- The main theorem
theorem num_houses_with_digit_7_in_range (h : Nat) (H1 : 1 ≤ h ∧ h ≤ 70) : 
  ∃! n, 1 ≤ n ∧ n ≤ 70 ∧ contains_digit_7 n :=
sorry

end num_houses_with_digit_7_in_range_l180_180577


namespace sec_minus_tan_l180_180809

-- Define the problem in Lean 4
theorem sec_minus_tan (x : ℝ) (h : Real.sec x + Real.tan x = 7 / 3) : Real.sec x - Real.tan x = 3 / 7 := by
  -- One could also include here the necessary mathematical facts and connections.
  sorry -- Proof to be provided

end sec_minus_tan_l180_180809


namespace intersection_point_ratio_l180_180929

universe u
variables {A B C D K L M N O : Type u}

noncomputable def ratio_point {X Y: Type u} [A B: Type u] [division_ring X] [add_comm_group Y] [vector_space X Y]
    : X → A → A → A
| r a b := (1 - r) • a + r • b

axiom H₁ : ∀ {A B a b: Type u} {p: X}, ratio_point p a b = a + p * (b - a)
axiom H₂ : ∀ {C D c d: Type u} {p: X}, ratio_point p c d = d + p * (c - d)
axiom H₃ : ∀ {B C b c: Type u} {q: X}, ratio_point q b c = b + q * (c - b)
axiom H₄ : ∀ {A D a d: Type u} {q: X}, ratio_point q a d = d + q * (a - d)

theorem intersection_point_ratio {A B C D: Prop}
  (h₁ : ∀ (p ≠ p' : ℝ), ratio_point p' K M = ratio_point q N L)
  (h₂ : ∀ (x : ℝ), (ratio_point q K M = ratio_point p N L)) :
  ∃ (O : ℝ), ratio_point q K M = ratio_point q N L ∧ ratio_point p N L = ratio_point p O O :=
sorry

end intersection_point_ratio_l180_180929


namespace parallelogram_area_l180_180704

variable (a b : Vector ℝ 3)

theorem parallelogram_area :
  (∥a × b∥ = 15) → 
  ∥(3 • a + 4 • b) × (2 • a - 6 • b)∥ = 390 := 
by
  intro h
  sorry

end parallelogram_area_l180_180704


namespace exponent_of_5_in_30_fact_l180_180018

def count_powers_of_5 (n : ℕ) : ℕ :=
  if n < 5 then 0
  else n / 5 + count_powers_of_5 (n / 5)

theorem exponent_of_5_in_30_fact : count_powers_of_5 30 = 7 := 
  by
    sorry

end exponent_of_5_in_30_fact_l180_180018


namespace mean_of_combined_set_is_52_over_3_l180_180156

noncomputable def mean_combined_set : ℚ := 
  let mean_set1 := 10
  let size_set1 := 4
  let mean_set2 := 21
  let size_set2 := 8
  let sum_set1 := mean_set1 * size_set1
  let sum_set2 := mean_set2 * size_set2
  let total_sum := sum_set1 + sum_set2
  let combined_size := size_set1 + size_set2
  let combined_mean := total_sum / combined_size
  combined_mean

theorem mean_of_combined_set_is_52_over_3 :
  mean_combined_set = 52 / 3 :=
by
  sorry

end mean_of_combined_set_is_52_over_3_l180_180156


namespace inscribed_circle_radius_l180_180500

noncomputable def a : ℝ := 5
noncomputable def b : ℝ := 10
noncomputable def c : ℝ := 20

noncomputable def r : ℝ := 1 / (1 / a + 1 / b + 1 / c + 2 * Real.sqrt (1 / (a * b) + 1 / (a * c) + 1 / (b * c)))

theorem inscribed_circle_radius :
  r = 20 / (3.5 + 2 * Real.sqrt 14) :=
sorry

end inscribed_circle_radius_l180_180500


namespace charlene_gave_18_necklaces_l180_180674

theorem charlene_gave_18_necklaces
  (initial_necklaces : ℕ) (sold_necklaces : ℕ) (left_necklaces : ℕ)
  (h1 : initial_necklaces = 60)
  (h2 : sold_necklaces = 16)
  (h3 : left_necklaces = 26) :
  initial_necklaces - sold_necklaces - left_necklaces = 18 :=
by
  sorry

end charlene_gave_18_necklaces_l180_180674


namespace complete_the_square_l180_180538

theorem complete_the_square (x : ℝ) :
  x^2 - 8 * x + 5 = 0 ↔ (x - 4)^2 = 11 :=
by
  sorry

end complete_the_square_l180_180538


namespace decreasing_interval_g_l180_180359

noncomputable def g (x : ℝ) (φ : ℝ) : ℝ :=
  3 * sin (2 * x + φ / 2) + 3 * sqrt 3 * sin (π / 4 - 2 * x)

theorem decreasing_interval_g :
  ∀ (φ : ℝ), 
    (0 < φ ∧ φ < π) →
    ∃ a b : ℝ, 
      (a < b) ∧
      (-π < a ∧ a < -3 * π / 4 ∧ -3 * π / 4 < b ∧ b ≤ 0) ∧
      (∀ x : ℝ, (a < x ∧ x < b) →
      deriv (g x φ) x < 0) :=
sorry

end decreasing_interval_g_l180_180359


namespace probability_no_intersecting_chords_l180_180074

open Nat

def double_factorial (n : Nat) : Nat :=
  if n = 0 ∨ n = 1 then 1 else n * double_factorial (n - 2)

def catalan_number (n : Nat) : Nat :=
  (factorial (2 * n)) / (factorial n * factorial (n + 1))

theorem probability_no_intersecting_chords (n : Nat) (h : n > 0) :
  (catalan_number n) / (double_factorial (2 * n - 1)) = 2^n / (factorial (n + 1)) :=
by
  sorry

end probability_no_intersecting_chords_l180_180074


namespace xy_range_l180_180340

theorem xy_range (x y : ℝ)
  (h1 : x + y = 1)
  (h2 : 1 / 3 ≤ x ∧ x ≤ 2 / 3) :
  2 / 9 ≤ x * y ∧ x * y ≤ 1 / 4 :=
sorry

end xy_range_l180_180340


namespace iterative_average_difference_l180_180594

def iterative_average (seq : List ℝ) : ℝ :=
  seq.foldl (λ acc x, (acc + x) / 2) 0

theorem iterative_average_difference : 
  let nums := {1, 2, 3, 4, 5, 6} : Set ℝ
  ∃ (σ1 σ2 : List ℝ), 
    σ1.Permute nums ∧ σ2.Permute nums ∧ 
    abs (iterative_average σ1 - iterative_average σ2) = 1 :=
by
  -- setup and proof body
  sorry

end iterative_average_difference_l180_180594


namespace max_length_OB_l180_180997

theorem max_length_OB (O A B : ℝ) (a : ℝ)
  (AB : ℝ)
  (angle_AOB : ℝ)
  (OA : ℝ = 2)
  (AB_eq : AB = 1.5)
  (angle_eq : angle_AOB = real.pi / 4) :
  sorry := sorry

end max_length_OB_l180_180997


namespace home_electronics_percentage_l180_180560

def percentage_microphotonics : ℝ := 10
def percentage_food_additives : ℝ := 15
def percentage_genetically_modified_microorganisms : ℝ := 29
def percentage_industrial_lubricants : ℝ := 8
def degrees_basic_astrophysics : ℝ := 50.4

theorem home_electronics_percentage :
  let percentage_basic_astrophysics := 100 * degrees_basic_astrophysics / 360 in
  let total_known_percentage := percentage_microphotonics + percentage_food_additives + 
                                percentage_genetically_modified_microorganisms + 
                                percentage_industrial_lubricants + 
                                percentage_basic_astrophysics in
  100 - total_known_percentage = 24 := 
by
  sorry

end home_electronics_percentage_l180_180560


namespace tan_equals_one_iff_tan_identity_l180_180960

theorem tan_equals_one_iff_tan_identity (x : ℝ) (k : ℤ) : 
  (∃ k : ℤ, x = k * π + π/4) ↔ tan x = 1 :=
by sorry

end tan_equals_one_iff_tan_identity_l180_180960


namespace exponent_of_5_in_30_fact_l180_180020

def count_powers_of_5 (n : ℕ) : ℕ :=
  if n < 5 then 0
  else n / 5 + count_powers_of_5 (n / 5)

theorem exponent_of_5_in_30_fact : count_powers_of_5 30 = 7 := 
  by
    sorry

end exponent_of_5_in_30_fact_l180_180020


namespace cylindrical_coordinates_of_point_l180_180298

theorem cylindrical_coordinates_of_point :
  ∀ (x y z : ℝ), x = 3 → y = -3 * Real.sqrt 3 → z = 2 →
  ∃ r θ, r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ r = 6 ∧ θ = 5 * Real.pi / 3 ∧ z = 2 :=
by
  intros x y z hx hy hz
  use 6, (5 * Real.pi / 3)
  split
  { exact zero_lt_six },
  split
  { norm_num },
  split
  { norm_num },
  split
  { exact hx.symm ▸ hy.symm ▸ rfl },
  split
  { exact hz },
  sorry

end cylindrical_coordinates_of_point_l180_180298


namespace snow_shoveling_l180_180201

noncomputable def volume_of_snow_shoveled (length1 length2 width depth1 depth2 : ℝ) : ℝ :=
  (length1 * width * depth1) + (length2 * width * depth2)

theorem snow_shoveling :
  volume_of_snow_shoveled 15 15 4 1 (1 / 2) = 90 :=
by
  sorry

end snow_shoveling_l180_180201


namespace exponent_of_5_in_30_factorial_l180_180002

theorem exponent_of_5_in_30_factorial : Nat.factorial 30 ≠ 0 → (nat.factorization (30!)).coeff 5 = 7 :=
by
  sorry

end exponent_of_5_in_30_factorial_l180_180002


namespace max_value_of_n_l180_180530

theorem max_value_of_n (
   { n : ℕ } 
   { k : fin n → ℕ }
   (h1 : ∀ i j, i ≠ j → k i ≠ k j)
   (h2 : ∀ i, 1 ≤ k i ∧ k i < 20)
   (h3 : ∑ i in finset.range n, (k ⟨i, sorry⟩) ^ 2 = 2021)
   ) : n ≤ 17 :=
begin
  sorry
end

end max_value_of_n_l180_180530


namespace problem_proof_l180_180098

open Set

theorem problem_proof :
  let U : Set ℕ := {1, 2, 3, 4, 5, 6}
  let P : Set ℕ := {1, 2, 3, 4}
  let Q : Set ℕ := {3, 4, 5}
  P ∩ (U \ Q) = {1, 2} :=
by
  let U : Set ℕ := {1, 2, 3, 4, 5, 6}
  let P : Set ℕ := {1, 2, 3, 4}
  let Q : Set ℕ := {3, 4, 5}
  show P ∩ (U \ Q) = {1, 2}
  sorry

end problem_proof_l180_180098


namespace exists_root_in_interval_l180_180976

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 5

theorem exists_root_in_interval : ∃ x, (2 < x ∧ x < 3) ∧ f x = 0 := 
by
  -- Assuming f(2) < 0 and f(3) > 0
  have h1 : f 2 < 0 := sorry
  have h2 : f 3 > 0 := sorry
  -- From the intermediate value theorem, there exists a c in (2, 3) such that f(c) = 0
  sorry

end exists_root_in_interval_l180_180976


namespace ticket_difference_l180_180546

theorem ticket_difference (V G : ℕ) (h1 : V + G = 320) (h2 : 45 * V + 20 * G = 7500) :
  G - V = 232 :=
by
  sorry

end ticket_difference_l180_180546


namespace rachel_total_clothing_l180_180940

def box_1_scarves : ℕ := 2
def box_1_mittens : ℕ := 3
def box_1_hats : ℕ := 1
def box_2_scarves : ℕ := 4
def box_2_mittens : ℕ := 2
def box_2_hats : ℕ := 2
def box_3_scarves : ℕ := 1
def box_3_mittens : ℕ := 5
def box_3_hats : ℕ := 3
def box_4_scarves : ℕ := 3
def box_4_mittens : ℕ := 4
def box_4_hats : ℕ := 1
def box_5_scarves : ℕ := 5
def box_5_mittens : ℕ := 3
def box_5_hats : ℕ := 2
def box_6_scarves : ℕ := 2
def box_6_mittens : ℕ := 6
def box_6_hats : ℕ := 0
def box_7_scarves : ℕ := 4
def box_7_mittens : ℕ := 1
def box_7_hats : ℕ := 3
def box_8_scarves : ℕ := 3
def box_8_mittens : ℕ := 2
def box_8_hats : ℕ := 4
def box_9_scarves : ℕ := 1
def box_9_mittens : ℕ := 4
def box_9_hats : ℕ := 5

def total_clothing : ℕ := 
  box_1_scarves + box_1_mittens + box_1_hats +
  box_2_scarves + box_2_mittens + box_2_hats +
  box_3_scarves + box_3_mittens + box_3_hats +
  box_4_scarves + box_4_mittens + box_4_hats +
  box_5_scarves + box_5_mittens + box_5_hats +
  box_6_scarves + box_6_mittens + box_6_hats +
  box_7_scarves + box_7_mittens + box_7_hats +
  box_8_scarves + box_8_mittens + box_8_hats +
  box_9_scarves + box_9_mittens + box_9_hats

theorem rachel_total_clothing : total_clothing = 76 :=
by
  sorry

end rachel_total_clothing_l180_180940


namespace regular_discount_rate_l180_180249

theorem regular_discount_rate (MSRP : ℝ) (s : ℝ) (sale_price : ℝ) (d : ℝ) :
  MSRP = 35 ∧ s = 0.20 ∧ sale_price = 19.6 → d = 0.3 :=
by
  intro h
  sorry

end regular_discount_rate_l180_180249


namespace sec_minus_tan_l180_180788

theorem sec_minus_tan (x : ℝ) (h : real.sec x + real.tan x = 7 / 3) :
  real.sec x - real.tan x = 3 / 7 :=
sorry

end sec_minus_tan_l180_180788


namespace jaylene_saves_fraction_l180_180123

-- Statement of the problem
theorem jaylene_saves_fraction (r_saves : ℝ) (j_saves : ℝ) (m_saves : ℝ) 
    (r_salary_fraction : r_saves = 2 / 5) 
    (m_salary_fraction : m_saves = 1 / 2) 
    (total_savings : 4 * (r_saves * 500 + j_saves * 500 + m_saves * 500) = 3000) : 
    j_saves = 3 / 5 := 
by 
  sorry

end jaylene_saves_fraction_l180_180123


namespace triangle_inequality_l180_180850

theorem triangle_inequality (a b c : ℝ) (S : ℝ) (hS : S = (1/4) * Real.sqrt ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c))) :
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * S :=
sorry

end triangle_inequality_l180_180850


namespace incorrect_statements_l180_180541

-- Define geometrical concepts and assumptions
structure Point := (x y z : ℝ)

structure Plane := (normal : Point) (d : ℝ)

def three_points_determine_plane (p1 p2 p3 : Point) : Prop :=
  ¬ collinear p1 p2 p3 → ∃ plane : Plane, p1 ∈ plane ∧ p2 ∈ plane ∧ p3 ∈ plane

def two_points_outside_plane_determine_parallel_plane (p1 p2 : Point) (alpha : Plane) : Prop :=
  ¬ perpendicular (line_through p1 p2) alpha → ∃ beta : Plane, beta ∥ alpha ∧ p1 ∈ beta ∧ p2 ∈ beta

def three_planes_intersect_parallel_lines (plane1 plane2 plane3 : Plane) : Prop :=
  intersection_line plane1 plane2 ∥ intersection_line plane2 plane3 ∧ ∥ intersection_line plane1 plane3

def extensions_of_lateral_edges_frustum_intersect (frustum : Frustum) : Prop :=
  ∃ apex : Point, ∀ edge : Edge, edge.extended frustum → apex ∈ edge

-- Theorem statement
theorem incorrect_statements :
  (¬ ∀ (p1 p2 p3 : Point), three_points_determine_plane p1 p2 p3) ∧
  (¬ ∀ (p1 p2 : Point) (alpha : Plane), two_points_outside_plane_determine_parallel_plane p1 p2 alpha) ∧
  (¬ ∀ (plane1 plane2 plane3 : Plane), three_planes_intersect_parallel_lines plane1 plane2 plane3) ∧
  (∀ (frustum : Frustum), extensions_of_lateral_edges_frustum_intersect frustum) :=
sorry

end incorrect_statements_l180_180541


namespace exponent_of_five_in_factorial_l180_180047

theorem exponent_of_five_in_factorial:
  (nat.factors 30!).count 5 = 7 :=
begin
  sorry
end

end exponent_of_five_in_factorial_l180_180047


namespace largest_negative_angle_satisfies_condition_l180_180152

theorem largest_negative_angle_satisfies_condition :
  ∃ θ : ℝ, (θ > -π ∧ θ < 0 ∧ (1 - Math.sin θ + Math.cos θ) / (1 - Math.sin θ - Math.cos θ) + (1 - Math.sin θ - Math.cos θ) / (1 - Math.sin θ + Math.cos θ) = 2) ∧ θ = -π / 2 :=
by
  sorry

end largest_negative_angle_satisfies_condition_l180_180152


namespace monotonic_intervals_extreme_points_inequality_l180_180761

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - a * x + Real.log x

-- Part 1
theorem monotonic_intervals (x : ℝ) (h : 0 < x) (h₁ : x < ½ ∨ x > 1) :
  let f₃ := f x 3
  (∀ y, 0 < y ∧ y < ½ → 0 < Derivative (f y 3)) ∧
  (∀ y, y ≠ 1 → 0 < Derivative (f y 3)) ∧
  (∀ y, ½ < y ∧ y < 1 → Derivative (f y 3) < 0) := sorry

-- Part 2
theorem extreme_points_inequality (x₁ x₂ a : ℝ)
  (h₀ : a ∈ Set ℝ) (h₁ : f x₁ a - f x₂ a < -¾ + Real.log 2) :
  let critical_points := {c : ℝ | IsCriticalPoint (f c a)}
  x₁ ∈ critical_points ∧ x₁ > 1 ∧ x₂ ∈ critical_points →
  f x₁ a - f x₂ a < -¾ + Real.log 2 := sorry

end monotonic_intervals_extreme_points_inequality_l180_180761


namespace assignment_M_eq_M_plus_3_l180_180526

-- Definitions based on the conditions provided
def assignment_operator (variable expression : ℕ) : ℕ := expression

-- Conditions as axioms
axiom assignment_format (variable expression : ℕ) : Prop := 
  variable = assignment_operator variable expression

axiom assignment_purpose (variable expression : ℕ) : 
  assignment_format variable expression → (variable = expression)

-- The specific problem to prove
theorem assignment_M_eq_M_plus_3 (M : ℕ) : 
  assignment_format M (M + 3) ∧ assignment_purpose M (M + 3) → (M = M + 3) :=
by
  sorry

end assignment_M_eq_M_plus_3_l180_180526


namespace exponent_of_five_in_factorial_l180_180054

theorem exponent_of_five_in_factorial:
  (nat.factors 30!).count 5 = 7 :=
begin
  sorry
end

end exponent_of_five_in_factorial_l180_180054


namespace mean_eq_median_of_set_l180_180587

theorem mean_eq_median_of_set (x : ℕ) (hx : 0 < x) :
  let s := [1, 2, 4, 5, x]
  let mean := (1 + 2 + 4 + 5 + x) / 5
  let median := if x ≤ 2 then 2 else if x ≤ 4 then x else 4
  mean = median → (x = 3 ∨ x = 8) :=
by {
  sorry
}

end mean_eq_median_of_set_l180_180587


namespace fair_game_stakes_ratio_l180_180184

theorem fair_game_stakes_ratio (n : ℕ) (deck_size : ℕ) (player_count : ℕ)
  (L : ℕ → ℝ) : 
  deck_size = 36 → player_count = 36 → 
  (∀ k : ℕ, k < player_count - 1 → 
    (L (k + 1)) / (L k) = 35 / 36) :=
by
  intros h_deck_size h_player_count k hk
  simp [h_deck_size, h_player_count, hk]
  sorry

end fair_game_stakes_ratio_l180_180184


namespace infinite_squares_sum_l180_180958

theorem infinite_squares_sum (a m : ℝ) (h_a : a > 0) (h_m : m > 0) :
  let S := (λ (a m : ℝ), (a * m * m) / (a + 2 * m)) in 
  S a m = (a * m^2) / (a + 2 * m) :=
by
  sorry

end infinite_squares_sum_l180_180958


namespace l_triomino_tile_l180_180325

theorem l_triomino_tile (n : ℕ) (p : Fin (2^n) × Fin (2^n)) : 
  ∃ tiling : (Fin (2^n) × Fin (2^n) → option (Fin 3)), ∀ (i j : Fin (2^n)), ∃ t : Fin 3, 
    tiling (i, j) = some t ∧ 
    (i, j) ≠ p → 
    tiling (i.succ, j) = some t ∧ 
    tiling (i, j.succ) = some t ∧ 
    tiling (i.succ, j.succ) = some t 
  := sorry

end l_triomino_tile_l180_180325


namespace longer_side_length_l180_180561

-- Define the radius of the circle and the areas
def radius : ℝ := 6
def area_circle : ℝ := π * radius^2
def area_rectangle : ℝ := 3 * area_circle
def shorter_side : ℝ := 2 * radius

-- Define the length of the longer side of the rectangle
noncomputable def longer_side : ℝ := area_rectangle / shorter_side

-- The theorem to prove the length of the longer side
theorem longer_side_length : longer_side = 9 * π := by
  sorry

end longer_side_length_l180_180561


namespace geometric_sequence_product_l180_180329

theorem geometric_sequence_product :
  ∀ (a : ℕ → ℝ), (∀ n, a n > 0) →
  (∃ (a_1 a_99 : ℝ), (a_1 + a_99 = 10) ∧ (a_1 * a_99 = 16) ∧ a 1 = a_1 ∧ a 99 = a_99) →
  a 20 * a 50 * a 80 = 64 :=
by
  intro a hpos hex
  sorry

end geometric_sequence_product_l180_180329


namespace bread_cost_each_is_3_l180_180695

-- Define the given conditions
def initial_amount : ℕ := 86
def bread_quantity : ℕ := 3
def orange_juice_quantity : ℕ := 3
def orange_juice_cost_each : ℕ := 6
def remaining_amount : ℕ := 59

-- Define the variable for bread cost
variable (B : ℕ)

-- Lean 4 statement to prove the cost of each loaf of bread
theorem bread_cost_each_is_3 :
  initial_amount - remaining_amount = (bread_quantity * B + orange_juice_quantity * orange_juice_cost_each) →
  B = 3 :=
by
  sorry

end bread_cost_each_is_3_l180_180695


namespace length_AD_two_tangent_semicircles_l180_180998

theorem length_AD_two_tangent_semicircles 
  (r : ℝ) (A B C D : ℝ) 
  (tangent : ∀ x y, (x - y) = 2 * r)
  (radius : r = real.sqrt 2)
  (parallel : A = D) :
  let AD := (2 * real.sqrt 2) + (2 * real.sqrt 2)
  in AD = 4 * real.sqrt 2 :=
by
  sorry

end length_AD_two_tangent_semicircles_l180_180998


namespace exponent_of_five_in_30_factorial_l180_180041

theorem exponent_of_five_in_30_factorial : 
  nat.factorial_prime_exponent 30 5 = 7 := 
sorry

end exponent_of_five_in_30_factorial_l180_180041


namespace john_paid_more_l180_180877

theorem john_paid_more 
  (original_price : ℝ)
  (discount_percentage : ℝ) 
  (tip_percentage : ℝ) 
  (discounted_price : ℝ)
  (john_tip : ℝ) 
  (john_total : ℝ)
  (jane_tip : ℝ)
  (jane_total : ℝ) 
  (difference : ℝ) :
  original_price = 42.00000000000004 →
  discount_percentage = 0.10 →
  tip_percentage = 0.15 →
  discounted_price = original_price - (discount_percentage * original_price) →
  john_tip = tip_percentage * original_price →
  john_total = original_price + john_tip →
  jane_tip = tip_percentage * discounted_price →
  jane_total = discounted_price + jane_tip →
  difference = john_total - jane_total →
  difference = 4.830000000000005 :=
by 
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end john_paid_more_l180_180877


namespace sec_minus_tan_l180_180803

theorem sec_minus_tan (x : ℝ) (h : Real.sec x + Real.tan x = 7 / 3) : Real.sec x - Real.tan x = 3 / 7 :=
by
  sorry

end sec_minus_tan_l180_180803


namespace exist_triangle_with_given_altitudes_l180_180293

def triangle_construction (h_a h_b h_c : ℝ) :=
  ∃ (A B C : ℝ × ℝ), 
    -- altitudes from vertices A, B, and C respectively
    let altitude_A := (altitude_from_vertex A B C),
        altitude_B := (altitude_from_vertex B A C),
        altitude_C := (altitude_from_vertex C A B) in 
    altitude_A = h_a ∧ altitude_B = h_b ∧ altitude_C = h_c

noncomputable def altitude_from_vertex (A B C : ℝ × ℝ) : ℝ := sorry

theorem exist_triangle_with_given_altitudes (h_a h_b h_c : ℝ) :
  triangle_construction h_a h_b h_c :=
sorry

end exist_triangle_with_given_altitudes_l180_180293


namespace equilateral_triangle_area_ratio_l180_180485

theorem equilateral_triangle_area_ratio (a : ℝ) (ha : a > 0) :
  let small_area := (sqrt 3 / 4) * a^2
  let perimeter_factor := 4
  let large_edge := perimeter_factor * (a / 3)
  let large_area := (sqrt 3 / 4) * large_edge^2
  (4 * small_area) / large_area = 1 / 4 :=
by
  // result
  sorry

end equilateral_triangle_area_ratio_l180_180485


namespace correct_operation_l180_180539

theorem correct_operation : 
  (∀ π, π ≠ 3 → (π - 3)^0 = 1) ∧
  ¬ (∀ a, a^3 + a^3 = a^6) ∧
  ¬ (∀ x, x^9 / x^3 = x^3) ∧ 
  ¬ (∀ a, (a^3)^2 = a^5) :=
by {
  sorry
}

end correct_operation_l180_180539


namespace cyclic_hexagon_diagonal_intersect_iff_product_of_sides_l180_180718

theorem cyclic_hexagon_diagonal_intersect_iff_product_of_sides
    {A B C D E F O : Type}
    [circle_inscribed_hexagon A B C D E F O]
    (intersect_at_one_point : intersect AD BE CF = some O) :
    (|AB| * |CD| * |EF| = |BC| * |DE| * |FA|) ↔ (intersect_at_one_point = some O) := 
sorry

end cyclic_hexagon_diagonal_intersect_iff_product_of_sides_l180_180718


namespace range_of_m_l180_180339

-- Definitions of propositions p and q based on the problem statement
def proposition_p (m : ℝ) := ∀ x : ℝ, x^2 + m * x + 1 = 0 → x < 0 ∧ 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x^2 + m * x + 1 = 0
def proposition_q (m : ℝ) := ∄ x : ℝ, 4 * x^2 + 4 * (m - 2) * x + 1 = 0

-- Given conditions of the problem
axiom p_Or_q (m : ℝ) : proposition_p m ∨ proposition_q m
axiom not_p_And_q (m : ℝ) : ¬(proposition_p m ∧ proposition_q m)

-- The problem to prove
theorem range_of_m (m : ℝ) : (1 < m ∧ m ≤ 2) ∨ (3 ≤ m) :=
begin
  sorry -- proof steps to be filled in
end

end range_of_m_l180_180339


namespace sec_minus_tan_l180_180808

theorem sec_minus_tan (x : ℝ) (h : Real.sec x + Real.tan x = 7 / 3) : Real.sec x - Real.tan x = 3 / 7 :=
by
  sorry

end sec_minus_tan_l180_180808


namespace find_number_l180_180228

noncomputable def percentage_of (p : ℝ) (n : ℝ) := p / 100 * n

noncomputable def fraction_of (f : ℝ) (n : ℝ) := f * n

theorem find_number :
  ∃ x : ℝ, percentage_of 40 60 = fraction_of (4/5) x + 4 ∧ x = 25 :=
by
  sorry

end find_number_l180_180228


namespace max_value_expression_l180_180319

theorem max_value_expression (x : ℝ) (hx : 0 < x ∧ x < 3) :
  ∃ y, y = (sup (λ (x : ℝ), (x^2 - 2*x + 3) / (2*x - 2)) (set.Ioo 0 3)) ∧ y = 1 :=
by
  sorry

end max_value_expression_l180_180319


namespace regular_polygon_sides_l180_180200

theorem regular_polygon_sides (A B C : Type) (circle : Type) (inscribed_in_circle : A × B × C → circle)
  (angle_A angle_B angle_C : Real)
  (h_angle_sum : angle_B = 3 * angle_A ∧ angle_C = 3 * angle_A ∧ angle_A + angle_B + angle_C = 180)
  (polygon_sides : ℕ) (adjacent_vertices : ∃ n, n = polygon_sides ∧ ∀ x y : ℕ, x = 360 / polygon_sides ∧ y = 1080 / 7 ∧ (B, C) = (x, y)) :
  polygon_sides = 2 :=
begin
  sorry
end

end regular_polygon_sides_l180_180200


namespace proof_m_n_sum_162_l180_180059

-- Definition of the setup for the right triangle with given conditions
variables {A B C D : Type} -- Types representing the points
variables [right_triangle : triangle A B C] -- Triangle ABC with a right angle at C
variables (BD : ℕ) (cos_B : ℚ) -- Defining BD and cos(B)
variables (m n : ℕ) -- Positive integers m and n such that gcd(m, n) = 1

-- Hypotheses corresponding to the conditions in the problem
def given_conditions : Prop := 
  BD = 17^3 ∧
  ∃ (AB BC AC : ℕ), 
    (triangle A B C ∧
    right_triangle ∧
    integer_sides A B C ∧
    (AB^2 = AC^2 + BC^2) ∧ 
    (BC * BC = BD * AB) ∧
    relatively_prime m n ∧
    cos_B = m / n ∧
    BD = 17^3)

-- Proving the required condition
theorem proof_m_n_sum_162 : given_conditions → (m + n = 162) :=
begin
  sorry
end

end proof_m_n_sum_162_l180_180059


namespace negative_tg_15_to_2019_l180_180780

theorem negative_tg_15_to_2019 : 
  ∃ (cnt : ℕ), cnt = 1009 ∧ (∀ n ∈ finset.range 2020, 1 ≤ n → n ≤ 2019 → (if (15^n % 360 : ℕ) = 135 then cnt = cnt + 1 else true)) := 
sorry

end negative_tg_15_to_2019_l180_180780


namespace minimum_gennadies_l180_180615

theorem minimum_gennadies (A B V G : ℕ) (hA : A = 45) (hB : B = 122) (hV : V = 27) (hGap : G + A + V >= B - 1) :
  G >= 49 :=
by 
  have := by linarith [hGap, hA, hB, hV]
  exact this

end minimum_gennadies_l180_180615


namespace no_such_n_exists_l180_180703

theorem no_such_n_exists :
  ¬ ∃ (n : ℕ), n > 1 ∧ ∃ (a : ℕ → ℕ), (∀ i j, (i ≠ j) → Nat.coprime (a i) (a j)) ∧ (∀ m ∈ Finset.range(n).succ, S (Finset.range n a) ∣ S (Finset.range n (λ j, (a j) * m))) :=
by
  sorry

end no_such_n_exists_l180_180703


namespace number_of_valid_3_element_subsets_l180_180227

-- Define basic elements
def S : Finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Helper function to check if the sum of elements in a subset is divisible by 3
def sum_divisible_by_3 (s : Finset ℕ) : Prop :=
  s.sum id % 3 = 0

-- Define a property that counts the valid subsets
def valid_3_element_subsets_count (S : Finset ℕ) : ℕ :=
  (S.powerset.filter (λ t, t.card = 3 ∧ sum_divisible_by_3 t)).card

theorem number_of_valid_3_element_subsets :
  valid_3_element_subsets_count S = 42 :=
by
  -- Proof will go here
  sorry

end number_of_valid_3_element_subsets_l180_180227


namespace a_plus_c_eq_neg_300_l180_180088

namespace Polynomials

variable {α : Type*} [LinearOrderedField α]

def f (a b x : α) := x^2 + a * x + b
def g (c d x : α) := x^2 + c * x + d

theorem a_plus_c_eq_neg_300 
  {a b c d : α}
  (h1 : ∀ x, f a b x ≥ -144) 
  (h2 : ∀ x, g c d x ≥ -144)
  (h3 : f a b 150 = -200) 
  (h4 : g c d 150 = -200)
  (h5 : ∃ x, (2*x + a = 0) ∧ g c d x = 0)
  (h6 : ∃ x, (2*x + c = 0) ∧ f a b x = 0) :
  a + c = -300 := 
sorry

end Polynomials

end a_plus_c_eq_neg_300_l180_180088


namespace area_of_triangle_XPQ_l180_180427

def triangle_area_XPQ (XY YZ XZ XP XQ : ℝ) (hXY : XY = 8) (hYZ : YZ = 13) (hXZ : XZ = 15) (hXP : XP = 3) (hXQ : XQ = 10) : ℝ :=
  (1/2) * XP * XQ * (4 * Real.sqrt 3 / 13)

theorem area_of_triangle_XPQ : 
  ∃ XY YZ XZ XP XQ : ℝ, 
    8 = XY ∧ 13 = YZ ∧ 15 = XZ ∧ 3 = XP ∧ 10 = XQ ∧ 
    triangle_area_XPQ XY YZ XZ XP XQ 8 13 15 3 10 = (60 * Real.sqrt 3 / 13) :=
by
  use [8, 13, 15, 3, 10]
  exact ⟨rfl, rfl, rfl, rfl, rfl, sorry⟩

end area_of_triangle_XPQ_l180_180427


namespace pentagon_coloring_l180_180991

theorem pentagon_coloring (Colors : Type) (A B C D E : Colors)
  (differ : ∀ (X Y : Colors), X ≠ Y → Prop):
  (∀ (X Y : Colors), X ∈ [A, B, C, D, E] → Y ∈ [A, B, C, D, E] → differ X Y)
  → ∃ (count : ℕ), count = 30 := 
  by 
    sorry

end pentagon_coloring_l180_180991


namespace exponent_of_5_in_30_factorial_l180_180007

theorem exponent_of_5_in_30_factorial : 
  (nat.factors 30!).count 5 = 7 :=
sorry

end exponent_of_5_in_30_factorial_l180_180007


namespace prime_divisor_of_product_l180_180909

theorem prime_divisor_of_product 
  (p : ℕ) (hp : Nat.Prime p) 
  (a b : ℤ) 
  (h : p ∣ a * b) : 
  p ∣ a ∨ p ∣ b := 
sorry

end prime_divisor_of_product_l180_180909


namespace price_of_orange_l180_180525

-- Define relevant conditions
def price_apple : ℝ := 1.50
def morning_apples : ℕ := 40
def morning_oranges : ℕ := 30
def afternoon_apples : ℕ := 50
def afternoon_oranges : ℕ := 40
def total_sales : ℝ := 205

-- Define the proof problem
theorem price_of_orange (O : ℝ) 
  (h : (morning_apples * price_apple + morning_oranges * O) + 
       (afternoon_apples * price_apple + afternoon_oranges * O) = total_sales) : 
  O = 1 :=
by
  sorry

end price_of_orange_l180_180525


namespace exponent_of_5_in_30_fact_l180_180030

def count_powers_of_5 (n : ℕ) : ℕ :=
  if n < 5 then 0
  else n / 5 + count_powers_of_5 (n / 5)

theorem exponent_of_5_in_30_fact : count_powers_of_5 30 = 7 := 
  by
    sorry

end exponent_of_5_in_30_fact_l180_180030


namespace enough_buses_l180_180100

theorem enough_buses (students : ℕ) (bus_capacity : ℕ) (buses : ℕ): 
  students = 298 → 
  bus_capacity = 52 → 
  buses = 6 → 
  buses * bus_capacity ≥ students :=
begin
  intros h_students h_bus_capacity h_buses,
  rw [h_students, h_bus_capacity, h_buses],
  norm_num,
  exact dec_trivial
end

end enough_buses_l180_180100


namespace transform_1234_to_2002_impossible_l180_180566

theorem transform_1234_to_2002_impossible :
  ¬ (∃ (f : ℕ → ℕ → bool), 
    (f 1 2 = tt ∧ f 2 3 = tt ∧ f 3 4 = tt ∧ 
    (∀ a b, (a ≠ 1 ∨ b ≠ 2 ∨ a ≠ 9) ∧ (a ≠ 0 ∨ b ≠ 1 ∨ a ≠ 8) ) ∧
    (∀ a b c d, f a b = tt → f b c = tt → f c d = tt → 
    ∃ a' b' c' d' : ℕ, (f a' b' = tt ∧ f b' c' = tt ∧ f c' d' = tt)))) 1234 2002 := 
sorry

end transform_1234_to_2002_impossible_l180_180566


namespace jason_hourly_earnings_l180_180064

-- Defining the conditions
variables (x : ℝ) 
variables (saturday_hours weekly_hours total_earnings saturday_earnings : ℝ)
variables (saturday_rate : ℝ := 6)
variables (hours_worked_after_school : ℝ := weekly_hours - saturday_hours)
variables (earnings_after_school : ℝ := total_earnings - saturday_hours * saturday_rate)

-- Provided conditions
def jason_conditions : Prop :=
  saturday_rate = 6 ∧
  weekly_hours = 18 ∧
  total_earnings = 88 ∧
  saturday_hours = 8 ∧
  earnings_after_school = total_earnings - saturday_hours * saturday_rate ∧
  hours_worked_after_school = weekly_hours - saturday_hours ∧
  x * hours_worked_after_school = earnings_after_school

-- The theorem we need to prove
theorem jason_hourly_earnings (h : jason_conditions x saturday_hours weekly_hours total_earnings saturday_earnings saturday_rate hours_worked_after_school earnings_after_school) : 
  x = 4 :=
begin
  sorry
end

end jason_hourly_earnings_l180_180064


namespace exponent_of_five_in_30_factorial_l180_180031

theorem exponent_of_five_in_30_factorial : 
  nat.factorial_prime_exponent 30 5 = 7 := 
sorry

end exponent_of_five_in_30_factorial_l180_180031


namespace min_number_of_gennadys_l180_180662

theorem min_number_of_gennadys (a b v g : ℕ) (h_a : a = 45) (h_b : b = 122) (h_v : v = 27)
    (h_needed_g : g = 49) :
    (b - 1) - (a + v) = g :=
by
  -- We include sorry because we are focusing on the statement, not the proof itself.
  sorry

end min_number_of_gennadys_l180_180662


namespace find_b_l180_180700

theorem find_b (b : ℝ) (h : log b 15625 = -4/3) : b = 1 / real.sqrt (5^9) :=
by
  sorry

end find_b_l180_180700


namespace part1_part2_l180_180913

variables (z1 z2 A : ℂ)
variables (A_nonzero: A ≠ 0)
variables (eqn: z1 * z2 + conj A * z1 + A * z2 = 0)

theorem part1 : (abs (z1 + A) * abs (z2 + A) = abs A ^ 2) :=
sorry

theorem part2 : ((z1 + A) / (z2 + A) = abs ((z1 + A) / (z2 + A))) :=
sorry

end part1_part2_l180_180913


namespace derivative_at_zero_l180_180666

open Real

def f (x : ℝ) : ℝ := if x ≠ 0 then (log (cos x)) / x else 0

theorem derivative_at_zero :
  deriv f 0 = -1 / 2 :=
by
  sorry

end derivative_at_zero_l180_180666


namespace inscribed_cylinder_radius_l180_180574

-- Define the conditions and question in Lean
variables (r : ℝ)

-- Hypothesize the conditions
def diameter_cone := 16
def radius_cone := diameter_cone / 2  -- Hence 8
def height_cone := 24
def height_cylinder := 2 * r  -- Cylinder height

-- Define the similarity condition
def similar_triangles_condition : Prop :=
  (height_cone - height_cylinder) / r = height_cone / radius_cone

-- Translate the proof problem
theorem inscribed_cylinder_radius (h : similar_triangles_condition) : 
  r = 24 / 5 :=
by
  -- The proof itself would involve solving the equation, which is not required here.
  sorry

end inscribed_cylinder_radius_l180_180574


namespace polygon_product_l180_180221

variables {n : ℕ} {r p : ℝ}
noncomputable def A_i (i : ℕ) : ℂ := r * complex.exp (2 * real.pi * complex.I * (i - 1) / n.to_real)
noncomputable def P : ℂ := p
noncomputable def PA_i (i : ℕ) : ℝ := complex.abs (P - A_i i)

theorem polygon_product :
  ∏ i in finset.range n, PA_i i = p^n - r^n := sorry

end polygon_product_l180_180221


namespace total_pencils_l180_180987

theorem total_pencils  (a b c : Nat) (total : Nat) 
(h₀ : a = 43) 
(h₁ : b = 19) 
(h₂ : c = 16) 
(h₃ : total = a + b + c) : 
total = 78 := 
by
  sorry

end total_pencils_l180_180987


namespace convert_to_cylindrical_l180_180296

-- Define the function for converting rectangular to cylindrical coordinates
def rectangular_to_cylindrical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := Real.atan2 y x
  (r, θ, z)

-- Given conditions
def point_rectangular : ℝ × ℝ × ℝ := (3, -3 * Real.sqrt 3, 2)
def expected_result : ℝ × ℝ × ℝ := (6, 5 * Real.pi / 3, 2)

-- The theorem to prove
theorem convert_to_cylindrical :
  rectangular_to_cylindrical 3 (-3 * Real.sqrt 3) 2 = expected_result := by
  sorry

end convert_to_cylindrical_l180_180296


namespace area_sin_6phi_is_pi_over_2_l180_180553

noncomputable def area_enclosed_by_sin_6phi : ℝ :=
  (1 / 2) * 12 * (1 / 2) * ∫ (x : ℝ) in 0..(Real.pi / 6), (Real.sin (6 * x)) ^ 2

theorem area_sin_6phi_is_pi_over_2 :
  area_enclosed_by_sin_6phi = Real.pi / 2 :=
by
  -- Proof goes here
  sorry

end area_sin_6phi_is_pi_over_2_l180_180553


namespace math_proof_problem_l180_180118

noncomputable def proofStatements : Prop := 
  (∀ x : ℝ, (cos (arcsin x) = sqrt (1 - x^2))) ∧
  (∀ x : ℝ, (sin (arccos x) = sqrt (1 - x^2))) ∧
  (∀ x : ℝ, (tan (arcctg x) = 1 / x)) ∧
  (∀ x : ℝ, (cot (arctg x) = 1 / x)) ∧
  (∀ x : ℝ, (cos (arctg x) = 1 / sqrt (1 + x^2))) ∧
  (∀ x : ℝ, (sin (arctg x) = x / sqrt (1 + x^2))) ∧
  (∀ x : ℝ, (cos (arcctg x) = x / sqrt (1 + x^2))) ∧
  (∀ x : ℝ, (sin (arcctg x) = 1 / sqrt (1 + x^2)))

theorem math_proof_problem : proofStatements := by {
  sorry
}

end math_proof_problem_l180_180118


namespace sec_tan_identity_l180_180828

theorem sec_tan_identity (x : ℝ) (h : Real.sec x + Real.tan x = 7 / 3) : Real.sec x - Real.tan x = 3 / 7 := 
by
  sorry

end sec_tan_identity_l180_180828


namespace num_points_within_and_on_boundary_is_six_l180_180710

noncomputable def num_points_within_boundary : ℕ :=
  let points := [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (4, 1)]
  points.length

theorem num_points_within_and_on_boundary_is_six :
  num_points_within_boundary = 6 :=
  by
    -- proof steps would go here
    sorry

end num_points_within_and_on_boundary_is_six_l180_180710


namespace problem_conditions_l180_180952

theorem problem_conditions (a b c : ℝ) (h1 : a > 0) (h2 : a > b) (h3 : b > c) :
  (ab : a * b > b * c) ∧
  (ac : a * c > b * c) ∧
  (ab_ac : a * b > a * c) ∧
  (a_plus_b : a + b > b + c) ∧
  (c_div_a : c / a < 1) :=
by
  sorry

end problem_conditions_l180_180952


namespace matrix_not_invertible_l180_180304

noncomputable def determinant (a b c d : ℝ) : ℝ := a * d - b * c

theorem matrix_not_invertible (x : ℝ) :
  determinant (2*x + 1) 9 (4 - x) 10 = 0 ↔ x = 26/29 := by
  sorry

end matrix_not_invertible_l180_180304


namespace problem_l180_180862

-- Define the general term in the expansion
def general_term (n r : ℕ) (x : ℝ) : ℝ :=
  (Nat.choose n r) * x^(n - (3*r / 2))

-- Binomial theorem related facts
lemma binomial_theorem_expansion (n : ℕ) (a b : ℝ) :
  (a + b)^n = ∑ r in Finset.range (n + 1), (Nat.choose n r) * a^(n - r) * b^r := sorry

-- Sum of binomial coefficients.
lemma sum_binomial_coefficients (n : ℕ) : ∑ r in Finset.range (n + 1), (Nat.choose n r) = 2^n := sorry

-- Problem: Prove the correct options in the binomial expansion given the conditions.
theorem problem (n : ℕ) :
  let x : ℝ := n in
  let expansion := (x + 1/√x)^9 in
  -- Option B: The sum of the binomial coefficients of the odd terms is 256.
  (∑ r in Finset.range 10, if r % 2 = 1 then Nat.choose 9 r else 0) = 256 ∧
  -- Option C: The constant term is 84.
  (let r := 6 in Nat.choose 9 r) = 84 := sorry

end problem_l180_180862


namespace sec_minus_tan_l180_180802

theorem sec_minus_tan (x : ℝ) (h : Real.sec x + Real.tan x = 7 / 3) : Real.sec x - Real.tan x = 3 / 7 :=
by
  sorry

end sec_minus_tan_l180_180802


namespace sec_sub_tan_l180_180801

theorem sec_sub_tan (x : ℝ) (h : sec x + tan x = 7 / 3) : sec x - tan x = 3 / 7 := by
  sorry

end sec_sub_tan_l180_180801


namespace avg_stoppage_time_is_20_minutes_l180_180307

noncomputable def avg_stoppage_time : Real :=
let train1 := (60, 40) -- without stoppages, with stoppages (in kmph)
let train2 := (75, 50) -- without stoppages, with stoppages (in kmph)
let train3 := (90, 60) -- without stoppages, with stoppages (in kmph)
let time1 := (train1.1 - train1.2 : Real) / train1.1
let time2 := (train2.1 - train2.2 : Real) / train2.1
let time3 := (train3.1 - train3.2 : Real) / train3.1
let total_time := time1 + time2 + time3
(total_time / 3) * 60 -- convert hours to minutes

theorem avg_stoppage_time_is_20_minutes :
  avg_stoppage_time = 20 :=
sorry

end avg_stoppage_time_is_20_minutes_l180_180307


namespace area_PQR_is_4_l180_180466

noncomputable def P : ℝ × ℝ := (4, 5)
noncomputable def Q : ℝ × ℝ := (-P.1, P.2)
noncomputable def R : ℝ × ℝ := (-Q.2, -Q.1)

def sq (x : ℝ) : ℝ := x * x

def distance (A B : ℝ × ℝ) : ℝ := 
  real.sqrt (sq (A.1 - B.1) + sq (A.2 - B.2))

def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_PQR_is_4 : area_of_triangle P Q R = 4 := 
by
  -- The proof will be filled in later
  sorry

end area_PQR_is_4_l180_180466


namespace find_a_l180_180353

theorem find_a (a : ℝ)
  (h : (λ x : ℝ, ((x^2 + a / x)^5)).coeff 7 = -10) :
  a = -2 :=
sorry

end find_a_l180_180353


namespace problem_solution_l180_180527

theorem problem_solution (a b c d : ℕ) (h1 : a = 10) (h2 : b = 7) (h3 : c = 45) (h4 : d = 5) :
  (a * b)^3 + (c * d)^2 = 393625 :=
by
  rw [h1, h2, h3, h4]
  simp
  sorry

end problem_solution_l180_180527


namespace symmetric_parabola_eqn_l180_180497

theorem symmetric_parabola_eqn :
  ∀ C : ℝ → ℝ, 
  (C = λ x : ℝ, 2*(x+2)^2 - 5) →
  ∃ C_sym : ℝ → ℝ,
  (C_sym = λ x : ℝ, 2*x^2 - 8*x + 3) :=
by
  intro C hC
  use (λ x : ℝ, 2*x^2 - 8*x + 3)
  sorry

end symmetric_parabola_eqn_l180_180497


namespace darryl_parts_cost_l180_180686

-- Define the conditions
def patent_cost : ℕ := 4500
def machine_price : ℕ := 180
def break_even_units : ℕ := 45
def total_revenue := break_even_units * machine_price

-- Define the theorem using the conditions
theorem darryl_parts_cost :
  ∃ (parts_cost : ℕ), parts_cost = total_revenue - patent_cost ∧ parts_cost = 3600 := by
  sorry

end darryl_parts_cost_l180_180686


namespace real_roots_of_polynomial_l180_180688

theorem real_roots_of_polynomial :
  (∀ x : ℝ, (x^10 + 36 * x^6 + 13 * x^2 = 13 * x^8 + x^4 + 36) ↔ 
    (x = 1 ∨ x = -1 ∨ x = 3 ∨ x = -3 ∨ x = 2 ∨ x = -2)) :=
by 
  sorry

end real_roots_of_polynomial_l180_180688


namespace min_value_a_b_l180_180899

variable (a b : ℝ)

theorem min_value_a_b (ha : a > 1) (hb : b > 1) (hab : a * b - (a + b) = 1) : 
  a + b ≥ 2 * (Real.sqrt 2 + 1) :=
sorry

end min_value_a_b_l180_180899


namespace ride_time_l180_180880

def bike_rate : ℝ := 1 / 4 -- miles per minute
def distance_JC : ℝ := 5 -- miles
def stop_time : ℝ := 5 -- minutes

theorem ride_time : (distance_JC / bike_rate) + stop_time = 25 :=
by
  sorry

end ride_time_l180_180880


namespace volume_set_l180_180291

open Real

def total_volume (l w h r : ℝ) : ℝ :=
  let volume_box := l * w * h
  let volume_ext_parallelepipeds := 2 * (1 * l * w) + 2 * (1 * l * h) + 2 * (1 * w * h)
  let volume_cylinders := 1 / 2 * π * r^2 * (2 * l + 2 * w + 2 * h)
  let volume_spheres := 8 * (1 / 4 * 4 / 3 * π * r^3)
  volume_box + volume_ext_parallelepipeds + volume_cylinders + volume_spheres

theorem volume_set :
  total_volume 2 3 6 1 = 108 + (41 / 3) * π :=
by
  sorry

end volume_set_l180_180291


namespace min_value_prime_factorization_l180_180453

/-- Let x and y be positive integers and assume 5 * x ^ 7 = 13 * y ^ 11.
  If x has a prime factorization of the form a ^ c * b ^ d, then the minimum possible value of a + b + c + d is 31. -/
theorem min_value_prime_factorization (x y a b c d : ℕ) (hx_pos : x > 0) (hy_pos: y > 0) (ha_pos : a > 0) (hb_pos : b > 0) (hc_pos: c > 0) (hd_pos: d > 0)
    (h_eq : 5 * x ^ 7 = 13 * y ^ 11) (h_fact : x = a^c * b^d) : a + b + c + d = 31 :=
by
  sorry

end min_value_prime_factorization_l180_180453


namespace set_union_count_l180_180495

theorem set_union_count : 
  let M := {M : Set ℕ | M ∪ {1} = {1, 2, 3}} in
  M.size = 2 :=
by sorry

end set_union_count_l180_180495


namespace exponent_of_five_in_30_factorial_l180_180039

theorem exponent_of_five_in_30_factorial : 
  nat.factorial_prime_exponent 30 5 = 7 := 
sorry

end exponent_of_five_in_30_factorial_l180_180039


namespace equality_of_MK_and_ML_l180_180077

variables (A B C D X K L M : Type)
variables [EuclideanSpace ℝ V]

-- Assuming points are distinct and we are working in a right triangle context
variables [noncomputable_instance]
variables [hABC : Triangle ABC]
variables (hRight : ∠ BCA = π / 2)

-- Definitions for the conditions
variables (hD : FootOfAltitude C A B D)
variables (hX : X ∈ Segment C D)
variables (hBK : ∃ X, ∃ K, K ∈ Segment A X ∧ BK = BC)
variables (hAL : ∃ X, ∃ L, L ∈ Segment B X ∧ AL = AC)
variables (hM : Intersection AL BK M)

-- Goal
theorem equality_of_MK_and_ML 
    (A B C D X K L M : Type) [EuclideanSpace ℝ V]
    [Triangle ABC] (hRight : ∠ BCA = π / 2)
    (hD : FootOfAltitude C A B D) (hX : X ∈ Segment C D)
    (hBK : ∃ X, ∃ K, K ∈ Segment A X ∧ BK = BC)
    (hAL : ∃ X, ∃ L, L ∈ Segment B X ∧ AL = AC)
    (hM : Intersection AL BK M) : MK = ML :=
sorry

end equality_of_MK_and_ML_l180_180077


namespace constant_isomorphism_l180_180292
noncomputable def f (x : ℝ) : ℝ := Real.tan (π * x - 3 * π / 2)

theorem constant_isomorphism (P Q : Set ℝ) (f : ℝ → ℝ)
  (hP : P = Set.Ioo 1 2) (hQ : Q = Set.univ) : 
  (∀ x ∈ P, f x ∈ Q) ∧ (∀ x1 x2 ∈ P, x1 < x2 → f x1 < f x2) ↔ 
  (P = Set.Ioo 1 2) ∧ (Q = Set.univ) :=
by
  sorry

end constant_isomorphism_l180_180292


namespace sec_minus_tan_l180_180814

-- Define the problem in Lean 4
theorem sec_minus_tan (x : ℝ) (h : Real.sec x + Real.tan x = 7 / 3) : Real.sec x - Real.tan x = 3 / 7 := by
  -- One could also include here the necessary mathematical facts and connections.
  sorry -- Proof to be provided

end sec_minus_tan_l180_180814


namespace compute_w4_l180_180887

theorem compute_w4 :
  let w := (-1 + real.sqrt 3 * complex.I) / 2 in
  w ^ 4 = ( -1 + real.sqrt 3 * complex.I ) / 2 :=
by
  sorry

end compute_w4_l180_180887


namespace exponent_of_five_in_factorial_l180_180051

theorem exponent_of_five_in_factorial:
  (nat.factors 30!).count 5 = 7 :=
begin
  sorry
end

end exponent_of_five_in_factorial_l180_180051


namespace min_number_of_gennadys_l180_180661

theorem min_number_of_gennadys (a b v g : ℕ) (h_a : a = 45) (h_b : b = 122) (h_v : v = 27)
    (h_needed_g : g = 49) :
    (b - 1) - (a + v) = g :=
by
  -- We include sorry because we are focusing on the statement, not the proof itself.
  sorry

end min_number_of_gennadys_l180_180661


namespace sec_minus_tan_l180_180816

theorem sec_minus_tan
  (x : ℝ)
  (h1 : Real.sec x + Real.tan x = 7 / 3)
  (h2 : (Real.sec x + Real.tan x) * (Real.sec x - Real.tan x) = 1) :
  Real.sec x - Real.tan x = 3 / 7 :=
by
  sorry

end sec_minus_tan_l180_180816


namespace total_cost_of_fencing_l180_180843

theorem total_cost_of_fencing (side_count : ℕ) (cost_per_side : ℕ) (h1 : side_count = 4) (h2 : cost_per_side = 79) : side_count * cost_per_side = 316 := by
  sorry

end total_cost_of_fencing_l180_180843


namespace unit_prices_and_purchasing_schemes_l180_180558

theorem unit_prices_and_purchasing_schemes :
  ∃ (x y : ℕ),
    (14 * x + 8 * y = 1600) ∧
    (3 * x = 4 * y) ∧
    (x = 80) ∧ 
    (y = 60) ∧
    ∃ (m : ℕ), 
      (m ≥ 29) ∧ 
      (m ≤ 30) ∧ 
      (80 * m + 60 * (50 - m) ≤ 3600) ∧
      (m = 29 ∨ m = 30) := 
sorry

end unit_prices_and_purchasing_schemes_l180_180558


namespace shekar_math_marks_l180_180125

-- Define the number of subjects
def num_subjects : ℕ := 5

-- Define the scores in known subjects
def science_marks : ℕ := 65
def social_studies_marks : ℕ := 82
def english_marks : ℕ := 67
def biology_marks : ℕ := 85

-- Define the average marks
def average_marks : ℕ := 75

-- Axiom for the total marks using the given average marks formula
axiom average_formula : (M : ℕ) → average_marks * num_subjects = M + science_marks + social_studies_marks + english_marks + biology_marks

-- Goal: Prove that Shekar's marks in Mathematics is 76
theorem shekar_math_marks : (M : ℕ) → average_formula M → M = 76 :=
by 
  intro M h,
  sorry

end shekar_math_marks_l180_180125


namespace min_gennadies_l180_180627

theorem min_gennadies 
  (n_Alexanders : ℕ) (n_Borises : ℕ) (n_Vasilies : ℕ) 
  (x_Gennadies : ℕ) 
  (h_Alexanders : n_Alexanders = 45) 
  (h_Borises   : n_Borises = 122) 
  (h_Vasilies  : n_Vasilies = 27) 
  (h_condition : ∀ p : ℕ, p = n_Borises - 1 → p = 121) 
  (h_total     : ∀ q : ℕ, q = 45 + 27 → q = 72)
  : x_Gennadies = 49 := 
sorry

end min_gennadies_l180_180627


namespace coefficient_x_squared_in_expansion_l180_180482

theorem coefficient_x_squared_in_expansion : 
  let f := (2 * x + 1)^6 in 
  (∃ c, f = c * x^2 + O(x^3)) →
  c = 60 :=
sorry

end coefficient_x_squared_in_expansion_l180_180482


namespace num_bijective_functions_l180_180413

def A : Set ℕ := {1, 2, ..., 2021}
def B : Set ℕ := {2, 3, ..., 2022}

theorem num_bijective_functions (f : A → B) 
  (h1 : Function.Bijective f) 
  (h2 : ∀ k ∈ A, k ∣ f k): 
  (∃ n, n = 13) :=
sorry

end num_bijective_functions_l180_180413


namespace distinct_prime_factors_l180_180953

open Nat

theorem distinct_prime_factors (n : ℕ) (h3 : ¬ 3 ∣ n) :
  ∃ p : Finset ℕ, (∀ q ∈ p, Prime q) ∧ p.card ≥ 2 * (n.divisors.count) :=
sorry

end distinct_prime_factors_l180_180953


namespace five_peso_coins_count_l180_180381

theorem five_peso_coins_count (x y : ℕ) (h1 : x + y = 56) (h2 : 10 * x + 5 * y = 440) (h3 : x = 24 ∨ y = 24) : y = 24 :=
by sorry

end five_peso_coins_count_l180_180381


namespace correct_number_of_true_statements_l180_180731

noncomputable def number_of_true_statements (m n : Type)
  (α β : set m)
  (p1 : Prop := ∀ (m_parallel_α : m ∥ α) (n_parallel_α : n ∥ α), m ∥ n)
  (p2 : Prop := ∀ (m_parallel_α : m ∥ α) (n_perp_α : n ⊥ α), n ⊥ m)
  (p3 : Prop := ∀ (m_perp_α : m ⊥ α) (m_parallel_β : m ∥ β), α ⊥ β) : ℕ :=
if p1 then 1 else 0 + 
if p2 then 1 else 0 + 
if p3 then 1 else 0

theorem correct_number_of_true_statements (m n : Type) (α β : set m) :
  number_of_true_statements m n α β = 2 :=
by sorry

end correct_number_of_true_statements_l180_180731


namespace diophantine_solution_unique_l180_180702

theorem diophantine_solution_unique (k x y : ℕ) (hk : k > 0) (hx : x > 0) (hy : y > 0) :
  x^2 + y^2 = k * x * y - 1 ↔ k = 3 :=
by sorry

end diophantine_solution_unique_l180_180702


namespace radii_equal_l180_180522

-- Defining the problem conditions
variables {L1 L2 : Set Point} -- Two parallel lines
variables {C1 C2 : Set Point} -- Two circles
variables (O1 O2 : Point) -- Centers of the circles
variables (r1 r2 : ℝ) -- Radii of the circles

-- Defining that the circles are tangent to the lines
def tangent (L : Set Point) (C : Set Point) : Prop :=
  ∃ p ∈ C, p ∈ L

-- Two lines are parallel
def parallel (L1 L2 : Set Point) : Prop :=
  ∀ p1 ∈ L1, ∀ p2 ∈ L2, p1 = p2 + λ v, v ∈ ℝ ∧ v ≠ 0

-- The main theorem
theorem radii_equal
  (h_parallel : parallel L1 L2)
  (h_tangent_C1_L1 : tangent L1 C1) (h_tangent_C1_L2 : tangent L2 C1)
  (h_tangent_C2_L1 : tangent L1 C2) (h_tangent_C2_L2 : tangent L2 C2) :
  r1 = r2 :=
sorry -- Proof not provided

end radii_equal_l180_180522


namespace exponent_of_five_in_30_factorial_l180_180038

theorem exponent_of_five_in_30_factorial : 
  nat.factorial_prime_exponent 30 5 = 7 := 
sorry

end exponent_of_five_in_30_factorial_l180_180038


namespace general_solution_satisfies_diff_eq_l180_180706

-- Define the differential equation operator
def diff_eq_operator (y : ℝ → ℝ) (x : ℝ) : ℝ :=
  (deriv^[5] y x) - 2 * (deriv^[4] y x) + 2 * (deriv^[3] y x) - 4 * (deriv y '' x) + (deriv y x) - 2 * y x

-- Define the general solution
def general_solution (C1 C2 C3 C4 C5 : ℝ) (x : ℝ) : ℝ :=
  C1 * exp (2 * x) + (C2 + C3 * x) * cos x + (C4 + C5 * x) * sin x

-- Theorem statement: proving that the general solution satisfies the differential equation
theorem general_solution_satisfies_diff_eq (C1 C2 C3 C4 C5 : ℝ) :
  ∀ x, diff_eq_operator (general_solution C1 C2 C3 C4 C5) x = 0 :=
by
  -- Proof goes here
  sorry

end general_solution_satisfies_diff_eq_l180_180706


namespace remainder_div_14_l180_180545

variables (x k : ℕ)

theorem remainder_div_14 (h : x = 142 * k + 110) : x % 14 = 12 := by 
  sorry

end remainder_div_14_l180_180545


namespace percent_difference_l180_180848

variables (w q y z x : ℝ)

-- Given conditions
def cond1 : Prop := w = 0.60 * q
def cond2 : Prop := q = 0.60 * y
def cond3 : Prop := z = 0.54 * y
def cond4 : Prop := x = 1.30 * w

-- The proof problem
theorem percent_difference (h1 : cond1 w q)
                           (h2 : cond2 q y)
                           (h3 : cond3 z y)
                           (h4 : cond4 x w) :
  ((z - x) / w) * 100 = 20 :=
by
  sorry

end percent_difference_l180_180848


namespace Ivan_can_transport_all_prisoners_l180_180072

-- Definitions of the conditions
def prisoners : ℕ := 43
def boat_capacity : ℕ := 2
def werewolves_known (P : ℕ) : Prop := P ≥ 40

-- Main theorem statement
theorem Ivan_can_transport_all_prisoners :
  ∃ (strategy : (ℕ → bool)), -- strategy to determine if a prisoner is ready to be transported based on conditions
  ∀ (k : ℕ), k ∈ finset.range prisoners → werewolves_known k →  
  ∃ (success : bool), success := 
  sorry

end Ivan_can_transport_all_prisoners_l180_180072


namespace ellipse_problem_l180_180733

theorem ellipse_problem
  (F2 : ℝ) (a : ℝ) (A B : ℝ × ℝ)
  (on_ellipse_A : (A.1 ^ 2) / (a ^ 2) + (25 * (A.2 ^ 2)) / (9 * a ^ 2) = 1)
  (on_ellipse_B : (B.1 ^ 2) / (a ^ 2) + (25 * (B.2 ^ 2)) / (9 * a ^ 2) = 1)
  (focal_distance : |A.1 + F2| + |B.1 + F2| = 8 / 5 * a)
  (midpoint_to_directrix : |(A.1 + B.1) / 2 + 5 / 4 * a| = 3 / 2) :
  a = 1 → (∀ x y, (x^2 + (25 / 9) * y^2 = 1) ↔ ((x^2) / (a^2) + (25 * y^2) / (9 * a^2) = 1)) :=
by
  sorry

end ellipse_problem_l180_180733


namespace find_angle_length_l180_180399

noncomputable def triangle_ABC (a b : ℝ) (A B C : ℝ) :=
  ∃ (a b : ℝ), (a^2 - 2 * Real.sqrt 3 * a + 2 = 0) ∧ (b^2 - 2 * Real.sqrt 3 * b + 2 = 0) ∧ (a + b = 2 * Real.sqrt 3) ∧ (ab = 2)

theorem find_angle_length (a b AB A B C : ℝ)
  (h1 : 2 * Real.cos (A + B) = 1)
  (h2 : a^2 - 2 * Real.sqrt 3 * a + 2 = 0)
  (h3 : b^2 - 2 * Real.sqrt 3 * b + 2 = 0)
  (h4 : a + b = 2 * Real.sqrt 3)
  (h5 : a * b = 2)
  (h6 : ∃ (A B C : ℝ),  C = π - (A + B)) :
  C = (2 * π / 3) ∧ AB = Real.sqrt 10 :=
by
  sorry

end find_angle_length_l180_180399


namespace probability_selecting_girl_l180_180259

def boys : ℕ := 3
def girls : ℕ := 1
def total_candidates : ℕ := boys + girls
def favorable_outcomes : ℕ := girls

theorem probability_selecting_girl : 
  ∃ p : ℚ, p = (favorable_outcomes : ℚ) / (total_candidates : ℚ) ∧ p = 1 / 4 :=
sorry

end probability_selecting_girl_l180_180259


namespace ivan_rescue_ivan_succeeds_l180_180071

-- Definitions of Conditions
def boat_capacity (n : ℕ) : Prop := n = 2
def return_trip_capacity (n : ℕ) : Prop := n = 1
def werewolf_info (p : ℕ) (k : ℕ) : Prop := p >= 40
def behavior_constraint (will_ride : ℕ → ℕ → Prop) : Prop :=
  ∀ i j, (will_ride i j) == (¬ (werewolf_info i j))

-- Define the problem "Ivan can transport all prisoners to the mainland" as a theorem
theorem ivan_rescue (prisoners : ℕ) (boat_capacity : ℕ) (return_trip_capacity : ℕ) (werewolf_info : ℕ → ℕ → Prop)
  (will_ride : ℕ → ℕ → Prop) (behavior_constraint : Prop) : Prop :=
  prisoners = 43 ∧
  boat_capacity = 2 ∧
  return_trip_capacity = 1 ∧
  (∀ p, werewolf_info p ≥ 40) ∧
  (∀ i j, (will_ride i j) == (¬ (werewolf_info i j))) →
  true -- indicating Ivan can successfully transport all prisoners

-- Proof omitted
theorem ivan_succeeds : (∀ (prisoners : ℕ) (boat_capacity : ℕ) (return_trip_capacity : ℕ) (werewolf_info : ℕ → ℕ → Prop)
  (will_ride : ℕ → ℕ → Prop) (behavior_constraint : Prop), 
  prisoners = 43 ∧ boat_capacity = 2 ∧ return_trip_capacity = 1 ∧ 
  (∀ p, werewolf_info p ≥ 40) ∧ 
  (∀ i j, (will_ride i j) == (¬ (werewolf_info i j))) →
  true) :=
by {
  assume prisoners boat_capacity return_trip_capacity werewolf_info will_ride behavior_constraint,
  assume h₁ : prisoners = 43 ∧ boat_capacity = 2 ∧ return_trip_capacity = 1 ∧
    (∀ p, werewolf_info p ≥ 40) ∧
    (∀ i j, (will_ride i j) == (¬ (werewolf_info i j))),
  exact true.intro,
}

end ivan_rescue_ivan_succeeds_l180_180071


namespace decorations_total_l180_180685

def number_of_skulls : Nat := 12
def number_of_broomsticks : Nat := 4
def number_of_spiderwebs : Nat := 12
def number_of_pumpkins (spiderwebs : Nat) : Nat := 2 * spiderwebs
def number_of_cauldron : Nat := 1
def number_of_lanterns (trees : Nat) : Nat := 3 * trees
def number_of_scarecrows (trees : Nat) : Nat := 1 * (trees / 2)
def total_stickers : Nat := 30
def stickers_per_window (stickers : Nat) (windows : Nat) : Nat := (stickers / 2) / windows
def additional_decorations (bought : Nat) (used_percent : Nat) (leftover : Nat) : Nat := ((bought * used_percent) / 100) + leftover

def total_decorations : Nat :=
  number_of_skulls +
  number_of_broomsticks +
  number_of_spiderwebs +
  (number_of_pumpkins number_of_spiderwebs) +
  number_of_cauldron +
  (number_of_lanterns 5) +
  (number_of_scarecrows 4) +
  (additional_decorations 25 70 15)

theorem decorations_total : total_decorations = 102 := by
  sorry

end decorations_total_l180_180685


namespace exponent_of_5_in_30_factorial_l180_180004

theorem exponent_of_5_in_30_factorial : 
  (nat.factors 30!).count 5 = 7 :=
sorry

end exponent_of_5_in_30_factorial_l180_180004


namespace nathaniel_wins_probability_l180_180102

/-- Conditions: Nathaniel and Obediah take turns rolling a fair six-sided die.
    The game continues until the running tally of the sum of all rolls is a multiple of 7.
    Nathaniel goes first. -/
theorem nathaniel_wins_probability :
  let p_nathaniel_wins : ℚ := 5 / 11 in
  p_nathaniel_wins = 5 / 11 := 
sorry

end nathaniel_wins_probability_l180_180102


namespace sum_of_possible_perimeters_l180_180110

theorem sum_of_possible_perimeters (AB BC : ℕ) (hAB : AB = 9) (hBC : BC = 21) : 
  let AC := AB + BC in ∃ s : ℕ, s = 380 :=
by
  have hAC : AC = 30 := by rw [hAB, hBC]
  let possible_xys := [ (95, 94), (33, 30), (17, 10)]
  let perimeters := possible_xys.map (fun (x, _) => 2 * x)
  let s := perimeters.sum + 90 -- (3 * AC)
  exact Exists.intro s rfl
  sorry

end sum_of_possible_perimeters_l180_180110


namespace order_of_trig_values_l180_180091

noncomputable def a := Real.sin (Real.sin (2008 * Real.pi / 180))
noncomputable def b := Real.sin (Real.cos (2008 * Real.pi / 180))
noncomputable def c := Real.cos (Real.sin (2008 * Real.pi / 180))
noncomputable def d := Real.cos (Real.cos (2008 * Real.pi / 180))

theorem order_of_trig_values : b < a ∧ a < d ∧ d < c :=
by
  sorry

end order_of_trig_values_l180_180091


namespace angle_DRO_l180_180426

theorem angle_DRO 
  (DOG : Triangle)
  (angle_DGO_eq_DOG : DOG.angle DGO = DOG.angle DOG)
  (angle_GDO_eq_30 : DOG.angle GDO = 30)
  (OR_bisects_angle_DOG : is_bisector OR DOG.angle DOG) :
  DOG.angle DRO = 37.5 := 
by
  sorry

end angle_DRO_l180_180426


namespace min_gennadys_l180_180606

-- Defining the basic constants for each name type
def Alexanders : Nat := 45
def Borises : Nat := 122
def Vasilies : Nat := 27

-- Define the proof statement to check the minimum number of Gennadys needed
theorem min_gennadys (a b v : Nat) (no_adjacent: a = 45 ∧ b = 122 ∧ v = 27) : ∃ g : Nat, g = 49 :=
by
  -- Using provided conditions
  cases no_adjacent with h_a h_bv
  cases h_bv with h_b h_v
  -- Correct answer derived from the solution
  use 49
  -- skipping proof details 
  sorry

end min_gennadys_l180_180606


namespace exponent_of_five_in_30_factorial_l180_180034

theorem exponent_of_five_in_30_factorial : 
  nat.factorial_prime_exponent 30 5 = 7 := 
sorry

end exponent_of_five_in_30_factorial_l180_180034


namespace binomial_pmf_l180_180382

open ProbabilityTheory
open Finset

variables (X : ℕ → ℕ) [H : measure_theory.MeasureSpace (sample_space X)]

noncomputable def binomial_pmf (n : ℕ) (p : ℚ) : pmf (fin (n+1)) :=
pmf.uniform_of_finset (finset.range (n + 1)) (λ k, (finset.choose n k : ℚ) * p^k * (1 - p)^(n - k))

variable (n : ℕ)
variable (p : ℚ)
variable (h₁ : X ~ binomial_pmf n p)
variable (h₂ : (X 1) = 6)
variable (h₃ : (X 1) = 3)

theorem binomial_pmf.X_eq_1_prob : 
  (pmf.prob (binomial_pmf 12 (1 / 2)) (1 : fin 13)) = 3 / 2^10 := 
sorry

end binomial_pmf_l180_180382


namespace find_KQ_l180_180721

theorem find_KQ (A L K M P Q S : Point) (a b α : ℝ)
  (A_on_angle_bisector_L : angle_bisector A L)
  (perpendicular_AK_AM : perpendicular A K A M)
  (P_on_KM : on_segment P K M)
  (K_between_Q_L : between K Q L)
  (ML_intersects_S : intersects (ML) S)
  (angle_KLM_eq_alpha : ∠ K L M = α)
  (KM_eq_a : dist K M = a)
  (QS_eq_b : dist Q S = b) :
  dist K Q = sqrt (b^2 - a^2) / (2 * cos (α / 2)) := sorry

end find_KQ_l180_180721


namespace pirate_islands_probability_l180_180250

open Finset

/-- There are 7 islands.
There is a 1/5 chance of finding an island with treasure only (no traps).
There is a 1/10 chance of finding an island with treasure and traps.
There is a 1/10 chance of finding an island with traps only (no treasure).
There is a 3/5 chance of finding an island with neither treasure nor traps.
We want to prove that the probability of finding exactly 3 islands
with treasure only and the remaining 4 islands with neither treasure
nor traps is 81/2225. -/
theorem pirate_islands_probability :
  (Nat.choose 7 3 : ℚ) * ((1/5)^3) * ((3/5)^4) = 81 / 2225 :=
by
  /- Here goes the proof -/
  sorry

end pirate_islands_probability_l180_180250


namespace number_of_sets_condition_l180_180086

theorem number_of_sets_condition (a b : ℝ) (c : ℝ) (h₀ : c ∈ set.Ico 0 (2 * Real.pi))
  (h₁ : ∀ x : ℝ, 2 * Real.sin (3 * x - Real.pi / 3) = a * Real.sin (b * x + c)) :
  ∃ (a_sets b_sets c_sets : finset ℝ), 
    a_sets.card * b_sets.card * c_sets.card = 4 :=
sorry

end number_of_sets_condition_l180_180086


namespace shift_sin_to_cos_l180_180198

theorem shift_sin_to_cos
  (x : ℝ) :
  (∀ x, cos (3 * x - π / 3) = sin (3 * (x + π / 18))) :=
by sorry

end shift_sin_to_cos_l180_180198


namespace solid_circles_count_2006_l180_180580

def series_of_circles (n : ℕ) : List Char :=
  if n ≤ 0 then []
  else if n % 5 == 0 then '●' :: series_of_circles (n - 1)
  else '○' :: series_of_circles (n - 1)

def count_solid_circles (l : List Char) : ℕ :=
  l.count '●'

theorem solid_circles_count_2006 : count_solid_circles (series_of_circles 2006) = 61 := 
by
  sorry

end solid_circles_count_2006_l180_180580


namespace min_number_of_gennadys_l180_180659

theorem min_number_of_gennadys (a b v g : ℕ) (h_a : a = 45) (h_b : b = 122) (h_v : v = 27)
    (h_needed_g : g = 49) :
    (b - 1) - (a + v) = g :=
by
  -- We include sorry because we are focusing on the statement, not the proof itself.
  sorry

end min_number_of_gennadys_l180_180659


namespace min_asterisks_needed_l180_180105

-- Define a chessboard as a set of positions (i, j) where 1 ≤ i, j ≤ 8
def Chessboard : Type := { p : ℕ × ℕ // 1 ≤ p.1 ∧ p.1 ≤ 8 ∧ 1 ≤ p.2 ∧ p.2 ≤ 8 }

-- Define a marked square or asterisk
def IsMarked (board : Chessboard → Prop) (p : Chessboard) : Prop := board p

-- Define adjacency (sharing an edge or vertex)
def Adjacent (p q : Chessboard) : Prop :=
  (p.val.1 = q.val.1 ∨ p.val.2 = q.val.2) ∧ (abs (p.val.1 - q.val.1) ≤ 1) ∧ (abs (p.val.2 - q.val.2) ≤ 1)

-- Define the problem conditions
def NoCommonEdgeOrVertex (board : Chessboard → Prop) : Prop :=
  ∀ p q : Chessboard, IsMarked board p → IsMarked board q → p ≠ q → ¬ Adjacent p q

def EachUnmarkedAdjacentToMarked (board : Chessboard → Prop) : Prop :=
  ∀ p : Chessboard, ¬ IsMarked board p → ∃ q : Chessboard, IsMarked board q ∧ Adjacent p q

-- Define the minimal number of asterisks needed (as a constant defined in the problem)
def MinimallyMarked (board : Chessboard → Prop) : Prop :=
  ∃ s : Finset Chessboard, (∀ p : Chessboard, board p ↔ p ∈ s) ∧ s.card = 16

-- Define the main theorem
theorem min_asterisks_needed : 
  ∀ (board : Chessboard → Prop), 
  NoCommonEdgeOrVertex board →
  EachUnmarkedAdjacentToMarked board → 
  MinimallyMarked board :=
by
  sorry

end min_asterisks_needed_l180_180105


namespace total_whales_seen_is_178_l180_180871

/-
Ishmael's monitoring of whales yields the following:
- On the first trip, he counts 28 male whales and twice as many female whales.
- On the second trip, he sees 8 baby whales, each traveling with their parents.
- On the third trip, he counts half as many male whales as the first trip and the same number of female whales as on the first trip.
-/

def number_of_whales_first_trip : ℕ := 28
def number_of_female_whales_first_trip : ℕ := 2 * number_of_whales_first_trip
def total_whales_first_trip : ℕ := number_of_whales_first_trip + number_of_female_whales_first_trip

def number_of_baby_whales_second_trip : ℕ := 8
def total_whales_second_trip : ℕ := number_of_baby_whales_second_trip * 3

def number_of_male_whales_third_trip : ℕ := number_of_whales_first_trip / 2
def number_of_female_whales_third_trip : ℕ := number_of_female_whales_first_trip
def total_whales_third_trip : ℕ := number_of_male_whales_third_trip + number_of_female_whales_third_trip

def total_whales_seen : ℕ := total_whales_first_trip + total_whales_second_trip + total_whales_third_trip

theorem total_whales_seen_is_178 : total_whales_seen = 178 :=
by
  -- skip the actual proof
  sorry

end total_whales_seen_is_178_l180_180871


namespace find_eccentricity_l180_180416

open Real

variables {a b : ℝ} (h_ab_pos : a > b) (h_b_pos : b > 0)

def ellipse_eccentricity (a b : ℝ) : ℝ :=
  let c := sqrt (a^2 - b^2)
  in c / a

theorem find_eccentricity (a b : ℝ) (h_ab_pos : a > b) (h_b_pos : b > 0) :
  let B := (ellipse_intersections_some_condition) -- Assume necessary conditions for B and C
  let C := (ellipse_intersections_some_condition)
  let F1 := (-sqrt(a^2 - b^2), 0)
  let F2 := (sqrt(a^2 - b^2), 0)
  ∠ B F2 C = 90° →
  ellipse_eccentricity a b = sqrt(2) - 1 :=
sorry

end find_eccentricity_l180_180416


namespace coefficient_of_x3_l180_180202

theorem coefficient_of_x3 :
  let p1 := (3 * X^4 + 2 * X^3 - 4 * X + 5)
  let p2 := (X^2 - 7 * X + 3)
  (p1 * p2).coeff 3 = 6 := 
by
  let p1 := (3 * X^4 + 2 * X^3 - 4 * X + 5)
  let p2 := (X^2 - 7 * X + 3)
  let product := p1 * p2
  have h : (product.coeff 3) = 6 := sorry
  exact h

end coefficient_of_x3_l180_180202


namespace locus_of_X_equation_l180_180078

theorem locus_of_X_equation (L1 L2 L3 : Type) (H P Q R X T : Type) (d p : ℝ)
  (parallel_1_2 : L1 ∥ L2) (perpend_1_3 : L3 ⊥ L1) (perpend_2_3 : L3 ⊥ L2)
  (H_on_3 : H ∈ L3) (P_on_3 : P ∈ L3) (H_on_1 : H ∈ L1) (H_ne_Q : Q ≠ H)
  (QR_eq_PR : dist Q R = dist P R) (d_def : ∀ θ, inscribed_circle_diameter_of_triangle P Q R = d)
  (T_semi_plane : same_semiplane Q T L3)
  (TH_def : dist T H = 2 * inscribed_circle_radius_of_triangle P Q R)
  (X_int : ∃ x, x ∈ line_intersection PQ TH) :
  locus_eqn : ∀ Q (hq : Q ∈ L1), (let X := int_point_of_lines PQ TH in
    (X.1)^2 - 3*(X.2)^2 + 4*p*X.2 - p^2 = 0 ∧ X.1 < 0 ∧ X.2 ≤ p/3) :=
sorry

end locus_of_X_equation_l180_180078


namespace sec_sub_tan_l180_180795

theorem sec_sub_tan (x : ℝ) (h : sec x + tan x = 7 / 3) : sec x - tan x = 3 / 7 := by
  sorry

end sec_sub_tan_l180_180795


namespace books_difference_l180_180670

theorem books_difference (bobby_books : ℕ) (kristi_books : ℕ) (h₁ : bobby_books = 142) (h₂ : kristi_books = 78) :
  bobby_books - kristi_books = 64 :=
by
  rw [h₁, h₂]
  norm_num
  exact rfl

end books_difference_l180_180670


namespace domain_of_f_l180_180392

theorem domain_of_f (f : ℝ → ℝ) :
  (∀ x, -2 ≤ x ∧ x ≤ 4 → (x ∈ {x | -2 ≤ x ∧ x ≤ 4}) → (3 * x + 1) ∈ {x | -5 ≤ x ∧ x ≤ 13}) →
  (∀ x, -2 ≤ (x - 1) / 3 ∧ (x - 1) / 3 ≤ 4 → x ∈ {x | -5 ≤ x ∧ x ≤ 13}) :=
begin
  sorry
end

end domain_of_f_l180_180392


namespace no_four_distinct_sum_mod_20_l180_180732

theorem no_four_distinct_sum_mod_20 (R : Fin 9 → ℕ) (h : ∀ i, R i < 19) :
  ¬ ∃ (a b c d : Fin 9), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  (R a + R b) % 20 = (R c + R d) % 20 := sorry

end no_four_distinct_sum_mod_20_l180_180732


namespace max_value_sqrt_min_value_sqrt_l180_180926

noncomputable def max_sum_sqrt (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x^2 + y^2 + z^2 = 10) : ℝ :=
  max (sqrt (6 - x^2) + sqrt (6 - y^2) + sqrt (6 - z^2)) (2 * sqrt 6)

noncomputable def min_sum_sqrt (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x^2 + y^2 + z^2 = 10) : ℝ :=
  min (sqrt (6 - x^2) + sqrt (6 - y^2) + sqrt (6 - z^2)) (sqrt 6 + sqrt 2)

theorem max_value_sqrt (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x^2 + y^2 + z^2 = 10) :
  sqrt (6 - x^2) + sqrt (6 - y^2) + sqrt (6 - z^2) ≤ 2 * sqrt 6 :=
begin
  sorry, -- Proof omitted
end

theorem min_value_sqrt (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x^2 + y^2 + z^2 = 10) :
  sqrt (6 - x^2) + sqrt (6 - y^2) + sqrt (6 - z^2) ≥ sqrt 6 + sqrt 2 :=
begin
  sorry, -- Proof omitted
end

end max_value_sqrt_min_value_sqrt_l180_180926


namespace min_gennadies_l180_180632

theorem min_gennadies 
  (n_Alexanders : ℕ) (n_Borises : ℕ) (n_Vasilies : ℕ) 
  (x_Gennadies : ℕ) 
  (h_Alexanders : n_Alexanders = 45) 
  (h_Borises   : n_Borises = 122) 
  (h_Vasilies  : n_Vasilies = 27) 
  (h_condition : ∀ p : ℕ, p = n_Borises - 1 → p = 121) 
  (h_total     : ∀ q : ℕ, q = 45 + 27 → q = 72)
  : x_Gennadies = 49 := 
sorry

end min_gennadies_l180_180632


namespace meet_time_and_speed_ratio_l180_180996

variable (A B : Point)
variable (v1 v2 : ℝ) -- v1 and v2 are the speeds of the cyclists
variable (t : ℝ) -- t is the time to the meeting point in hours

-- Conditions
axiom cyclists_start : cyclists_start_simultaneously A B
axiom first_to_B : first_cyclist_time_to_B_after_meeting = 2/3
axiom second_to_A : second_cyclist_time_to_A_after_meeting = 3/2

-- Variables for speeds and distances
axiom speed_v1 : distance_covered v1 first_cyclist_time_to_B_after_meeting = v1 * 2/3
axiom speed_v2 : distance_covered v2 second_cyclist_time_to_A_after_meeting = v2 * 3/2

theorem meet_time_and_speed_ratio (mt : meet_time t) (sr : speed_ratio v1 v2) : 
  t = 1 ∧ sr = 3/2 := sorry

end meet_time_and_speed_ratio_l180_180996


namespace geometric_sequence_product_l180_180746

variable {a : ℕ → ℝ}
variable {r : ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

theorem geometric_sequence_product (h1 : is_geometric_sequence a r)
                                   (h2 : 0 < ∀ n, a n)
                                   (h3 : a 4 * a 8 = 4) :
  a 5 * a 6 * a 7 = 8 :=
by
  sorry

end geometric_sequence_product_l180_180746


namespace hyperbola_eccentricity_l180_180915

theorem hyperbola_eccentricity (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (F1 F2 A : ℝ × ℝ) (hF1 : F1 = (-real.sqrt (a^2 + b^2), 0))
  (hF2 : F2 = (real.sqrt (a^2 + b^2), 0))
  (hA : ∃ x : ℝ, A = (x, x - real.sqrt (a^2 + b^2)) ∧ (x / a)^2 - ((x - real.sqrt (a^2 + b^2)) / b)^2 = 1)
  (hIso : dist F1 A = dist F2 A) :
  let c := real.sqrt (a^2 + b^2) in
  ∃ e : ℝ, e = c / a ∧ e = real.sqrt 2 + 1 :=
sorry

end hyperbola_eccentricity_l180_180915


namespace volume_of_solid_area_of_region_l180_180360

noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- Question 1: Volume of the solid formed by rotating the region
theorem volume_of_solid : (∫ x in 0..2, π * (f x)^2) = (π / 2) * (Real.exp (2 * 2) - 1) :=
by
  sorry

-- Define the tangent line at x = 2
noncomputable def tangent_line (x : ℝ) : ℝ := (Real.exp 2) * (x - 2) + Real.exp 2

-- Question 2: Area of the region bounded by y=f(x), the tangent line, and the y-axis
theorem area_of_region : (∫ x in 0..2, f x - tangent_line x) = Real.exp 2 - 1 :=
by
  sorry

end volume_of_solid_area_of_region_l180_180360


namespace total_whales_correct_l180_180870

def first_trip_male_whales : ℕ := 28
def first_trip_female_whales : ℕ := 2 * first_trip_male_whales
def first_trip_total_whales : ℕ := first_trip_male_whales + first_trip_female_whales

def second_trip_baby_whales : ℕ := 8
def second_trip_parent_whales : ℕ := 2 * second_trip_baby_whales
def second_trip_total_whales : ℕ := second_trip_baby_whales + second_trip_parent_whales

def third_trip_male_whales : ℕ := first_trip_male_whales / 2
def third_trip_female_whales : ℕ := first_trip_female_whales
def third_trip_total_whales : ℕ := third_trip_male_whales + third_trip_female_whales

def total_whales_seen : ℕ :=
  first_trip_total_whales + second_trip_total_whales + third_trip_total_whales

theorem total_whales_correct : total_whales_seen = 178 := by
  sorry

end total_whales_correct_l180_180870


namespace relationship_among_variables_l180_180776

theorem relationship_among_variables (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
    (h1 : a^2 = 2) (h2 : b^3 = 3) (h3 : c^4 = 4) (h4 : d^5 = 5) : a = c ∧ a < d ∧ d < b := 
by
  sorry

end relationship_among_variables_l180_180776


namespace minimum_gennadies_l180_180614

theorem minimum_gennadies (A B V G : ℕ) (hA : A = 45) (hB : B = 122) (hV : V = 27) (hGap : G + A + V >= B - 1) :
  G >= 49 :=
by 
  have := by linarith [hGap, hA, hB, hV]
  exact this

end minimum_gennadies_l180_180614


namespace david_account_amount_l180_180550

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem david_account_amount : compound_interest 5000 0.06 2 1 = 5304.50 := by
  sorry

end david_account_amount_l180_180550


namespace average_age_of_combined_groups_l180_180478

theorem average_age_of_combined_groups :
  (let total_age_A := 8 * 38,
       total_age_B := 2 * 30,
       combined_total_age := total_age_A + total_age_B,
       total_number_of_people := 8 + 2
   in combined_total_age / total_number_of_people = 36.4) :=
by
  sorry

end average_age_of_combined_groups_l180_180478


namespace find_xyz_and_sum_of_squares_l180_180841

variables (x y z a b c : ℝ)

theorem find_xyz_and_sum_of_squares 
  (h1 : x * y = 2 * a)
  (h2 : x * z = 3 * b)
  (h3 : y * z = 4 * c) :
  (x^2 + y^2 + z^2 = (3 * a * b) / (2 * c) + (8 * a * c) / (3 * b) + (6 * b * c) / a) ∧
  (x * y * z = 2 * real.sqrt(6 * a * b * c)) :=
by 
  sorry

end find_xyz_and_sum_of_squares_l180_180841


namespace sec_minus_tan_l180_180790

theorem sec_minus_tan (x : ℝ) (h : real.sec x + real.tan x = 7 / 3) :
  real.sec x - real.tan x = 3 / 7 :=
sorry

end sec_minus_tan_l180_180790


namespace max_sum_surrounding_45_l180_180691

theorem max_sum_surrounding_45:
  ∃ (a b c d e f g h : ℕ), 
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
     b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
     c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
     d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ 
     e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
     f ≠ g ∧ f ≠ h ∧
     g ≠ h) ∧ 
    (45 * a * b = 3240) ∧ 
    (45 * c * d = 3240) ∧ 
    (45 * e * f = 3240) ∧ 
    (45 * g * h = 3240) ∧ 
    max_sum_surrounding 45 a b c d e f g h = 160 :=
sorry

noncomputable def max_sum_surrounding (a b c d e f g h : ℕ) : ℕ :=
  a + b + c + d + e + f + g + h

end max_sum_surrounding_45_l180_180691


namespace dispatch_3_male_2_female_dispatch_at_least_2_male_l180_180510

-- Define the number of male and female drivers
def male_drivers : ℕ := 5
def female_drivers : ℕ := 4
def total_drivers_needed : ℕ := 5

-- Define the combination formula (binomial coefficient)
def combination (n k : ℕ) : ℕ := Nat.choose n k

-- First part of the problem
theorem dispatch_3_male_2_female : 
  combination male_drivers 3 * combination female_drivers 2 = 60 :=
by sorry

-- Second part of the problem
theorem dispatch_at_least_2_male : 
  combination male_drivers 2 * combination female_drivers 3 + 
  combination male_drivers 3 * combination female_drivers 2 + 
  combination male_drivers 4 * combination female_drivers 1 + 
  combination male_drivers 5 * combination female_drivers 0 = 121 :=
by sorry

end dispatch_3_male_2_female_dispatch_at_least_2_male_l180_180510


namespace minimum_number_of_teachers_needed_l180_180215

theorem minimum_number_of_teachers_needed:
  ∀ (math_teachers physics_teachers chemistry_teachers : ℕ),
  math_teachers = 4 →
  physics_teachers = 3 →
  chemistry_teachers = 3 →
  ∀ (max_subjects_per_teacher : ℕ), 
  max_subjects_per_teacher = 2 →
  ∃ (minimum_teachers : ℕ), minimum_teachers = 6 := by
  intros math_teachers physics_teachers chemistry_teachers
  intros h1 h2 h3 max_subjects_per_teacher h4
  use 6
  sorry

end minimum_number_of_teachers_needed_l180_180215


namespace b_shaped_polyomino_count_l180_180933

-- Definition of the grid
def grid_size : ℕ := 8
def total_tiles : ℕ := grid_size * grid_size

-- Definition of the polyomino
def polyomino_tiles : ℕ := 4

-- Problem statement
theorem b_shaped_polyomino_count : 
  ∃ n : ℕ, 
  (n * polyomino_tiles) = total_tiles ∧
  equal_coverage_across_lines n := 
  sorry

end b_shaped_polyomino_count_l180_180933


namespace new_bottle_price_l180_180232

def volume (r h : ℝ) : ℝ :=
  π * r^2 * h

def price (volume₁ price₁ volume₂ : ℝ) : ℝ :=
  (volume₂ / volume₁) * price₁

def original_diameter := 8
def original_height := 10
def new_height := 15
def original_radius := original_diameter / 2
def original_volume := volume original_radius original_height 
def new_volume := volume original_radius new_height
def original_price := 5

theorem new_bottle_price : price original_volume original_price new_volume = 7.5 := by
  sorry

end new_bottle_price_l180_180232


namespace total_heartbeats_during_race_l180_180271

namespace Heartbeats

def avg_heart_beats_per_minute : ℕ := 160
def pace_minutes_per_mile : ℕ := 6
def race_distance_miles : ℕ := 20

theorem total_heartbeats_during_race :
  (race_distance_miles * pace_minutes_per_mile * avg_heart_beats_per_minute = 19200) :=
by
  sorry

end Heartbeats

end total_heartbeats_during_race_l180_180271


namespace ten_two_digit_integers_have_disjoint_subsets_with_same_sum_l180_180524

-- Definitions for the problem
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def subsets_with_same_sum_exists (s : Finset ℕ) : Prop :=
  ∃ A B : Finset ℕ, A ∩ B = ∅ ∧ s ⊆ Finset.range 100 ∧ 
  A.sum = B.sum

-- The theorem as described
theorem ten_two_digit_integers_have_disjoint_subsets_with_same_sum
  (s : Finset ℕ) (h : s.card = 10) (h2 : ∀ x ∈ s, is_two_digit x) :
  subsets_with_same_sum_exists s :=
  sorry

end ten_two_digit_integers_have_disjoint_subsets_with_same_sum_l180_180524


namespace geometric_series_evaluation_l180_180087

theorem geometric_series_evaluation (c d : ℝ) (h : (∑' n : ℕ, c / d^(n + 1)) = 3) :
  (∑' n : ℕ, c / (c + 2 * d)^(n + 1)) = (3 * d - 3) / (5 * d - 4) :=
sorry

end geometric_series_evaluation_l180_180087


namespace exponent_of_five_in_30_factorial_l180_180037

theorem exponent_of_five_in_30_factorial : 
  nat.factorial_prime_exponent 30 5 = 7 := 
sorry

end exponent_of_five_in_30_factorial_l180_180037


namespace exponent_of_5_in_30_factorial_l180_180008

theorem exponent_of_5_in_30_factorial : 
  (nat.factors 30!).count 5 = 7 :=
sorry

end exponent_of_5_in_30_factorial_l180_180008


namespace tom_sold_4_books_l180_180199

-- Definitions based on conditions from the problem
def initial_books : ℕ := 5
def new_books : ℕ := 38
def final_books : ℕ := 39

-- The number of books Tom sold
def books_sold (S : ℕ) : Prop := initial_books - S + new_books = final_books

-- Our goal is to prove that Tom sold 4 books
theorem tom_sold_4_books : books_sold 4 :=
  by
    -- Implicitly here would be the proof, but we use sorry to skip it
    sorry

end tom_sold_4_books_l180_180199


namespace min_gennadys_l180_180607

-- Defining the basic constants for each name type
def Alexanders : Nat := 45
def Borises : Nat := 122
def Vasilies : Nat := 27

-- Define the proof statement to check the minimum number of Gennadys needed
theorem min_gennadys (a b v : Nat) (no_adjacent: a = 45 ∧ b = 122 ∧ v = 27) : ∃ g : Nat, g = 49 :=
by
  -- Using provided conditions
  cases no_adjacent with h_a h_bv
  cases h_bv with h_b h_v
  -- Correct answer derived from the solution
  use 49
  -- skipping proof details 
  sorry

end min_gennadys_l180_180607


namespace minimum_sugar_amount_l180_180281

theorem minimum_sugar_amount (f s : ℕ) (h1 : f ≥ 9 + s / 2) (h2 : f ≤ 3 * s) : s ≥ 4 :=
by
  -- Provided conditions: f ≥ 9 + s / 2 and f ≤ 3 * s
  -- Goal: s ≥ 4
  sorry

end minimum_sugar_amount_l180_180281


namespace alice_needs_to_add_stamps_l180_180668

variable (A B E P D : ℕ)
variable (h₁ : B = 4 * E)
variable (h₂ : E = 3 * P)
variable (h₃ : P = 2 * D)
variable (h₄ : D = A + 5)
variable (h₅ : A = 65)

theorem alice_needs_to_add_stamps : (1680 - A = 1615) :=
by
  sorry

end alice_needs_to_add_stamps_l180_180668


namespace mrs_hilt_additional_rocks_l180_180462

-- Definitions from the conditions
def total_rocks : ℕ := 125
def rocks_she_has : ℕ := 64
def additional_rocks_needed : ℕ := total_rocks - rocks_she_has

-- The theorem to prove the question equals the answer given the conditions
theorem mrs_hilt_additional_rocks : additional_rocks_needed = 61 := 
by
  sorry

end mrs_hilt_additional_rocks_l180_180462


namespace player_a_wins_l180_180076

theorem player_a_wins (n : ℕ) (hn : n ≥ 3) (hn_odd : n % 2 = 1) :
  ∃ k, ∀ S : Finset ℤ, (S.card = k ∧ (∀ x ∈ (Finset.range (2 * n + 1)).map (Int.ofNat ∘ (λ i, -n + i)),
  ∃ T ⊆ S, T.card = n ∧ x = T.sum)) :=
begin
  use ⌈(3 * n : ℚ) / 2 + 1 / 2⌉,
  sorry
end

end player_a_wins_l180_180076


namespace largest_inscribed_parabola_area_l180_180720

noncomputable def maximum_parabolic_area_in_cone (r l : ℝ) : ℝ :=
  (l * r) / 2 * Real.sqrt 3

theorem largest_inscribed_parabola_area (r l : ℝ) : 
  ∃ t : ℝ, t = maximum_parabolic_area_in_cone r l :=
by
  let t_max := (l * r) / 2 * Real.sqrt 3
  use t_max
  sorry

end largest_inscribed_parabola_area_l180_180720


namespace problem_1_l180_180226

def vectors_orthogonal {x : ℝ} (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

theorem problem_1 (x : ℝ) (h : vectors_orthogonal (1, 3) (x, -1)) : x = 3 :=
sorry

end problem_1_l180_180226


namespace proof_problem_l180_180860

noncomputable theory

-- The parametric equations of curve C1
def curve_C1_x (t : ℝ) : ℝ := 1 + 2 * t
def curve_C1_y (t : ℝ) : ℝ := 2 - 2 * t

-- Convert C1 to Cartesian form x + y - 3 = 0
def C1_cartesian (x y : ℝ) : Prop := x + y - 3 = 0

-- The polar equation of curve C2: ρ - ρcos²θ - 2cosθ = 0
def C2_polar (ρ θ : ℝ) : Prop := ρ - ρ * (Real.cos θ)² - 2 * Real.cos θ = 0

-- Convert C2 to Cartesian form y² = 2x
def C2_cartesian (x y : ℝ) : Prop := y² = 2 * x

-- Prove the given conditions and derived result
theorem proof_problem (P : ℝ × ℝ) (A B : ℝ × ℝ) (t1 t2 : ℝ) (x y : ℝ) :
    (∃ t, curve_C1_x t = x ∧ curve_C1_y t = y) →
    (∃ ρ θ, C2_polar ρ θ ∧ Real.cos θ * ρ = x ∧ Real.sin θ * ρ = y) →
    P = (1, 2) →
    (curve_C1_x t1, curve_C1_y t1) = A →
    (curve_C1_x t2, curve_C1_y t2) = B →
    (t1 + t2 = 6 * Real.sqrt 2 ∧ t1 * t2 = -4) →
    |((P.1 - A.1) + (P.2 - A.2))| + |((P.1 - B.1) + (P.2 - B.2))| = 2 * Real.sqrt 22 :=
sorry

end proof_problem_l180_180860


namespace sixth_term_is_sixteen_l180_180968

-- Definition of the conditions
def first_term : ℝ := 512
def eighth_term (r : ℝ) : Prop := 512 * r^7 = 2

-- Proving the 6th term is 16 given the conditions
theorem sixth_term_is_sixteen (r : ℝ) (hr : eighth_term r) :
  512 * r^5 = 16 :=
by
  sorry

end sixth_term_is_sixteen_l180_180968


namespace ratio_Gladys_to_sum_Billy_Lucas_l180_180957

-- Define the ages of each person
def Gladys_age := 30
def Billy_age := Gladys_age / 3
def Lucas_future_age := 8
def Lucas_age := Lucas_future_age - 3

-- Calculate the sum of Billy's and Lucas' ages
def Sum := Billy_age + Lucas_age

-- Define the question as a proposition
theorem ratio_Gladys_to_sum_Billy_Lucas : Gladys_age / Sum = 2 := by
  -- Proof should be given here
  sorry

end ratio_Gladys_to_sum_Billy_Lucas_l180_180957


namespace largest_apartment_size_l180_180596

theorem largest_apartment_size (cost_per_sqft : ℝ) (monthly_budget : ℝ) (s : ℝ) 
  (h1 : cost_per_sqft = 1.2) (h2 : monthly_budget = 720) : 
  (monthly_budget / cost_per_sqft) = 600 :=
by
  rw [h1, h2]
  norm_num
  sorry

end largest_apartment_size_l180_180596


namespace C_optimal_start_time_l180_180555

-- Conditions and variables
def distance_MN : ℝ := 15 -- Distance between M and N in km
def walking_speed : ℝ := 6 -- Walking speed in km/h for A, B, and C
def biking_speed : ℝ := 15 -- Biking speed in km/h

-- Variables
variable (x : ℝ) -- distance each walks in km

-- Travel time computations
noncomputable def travel_time_walk (distance : ℝ) (speed : ℝ) : ℝ := distance / speed
noncomputable def travel_time_bike (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

-- The equation derived from the setup
def travel_equation (x : ℝ) : Prop := (distance_MN - 3 * x) = 2.5 * x

-- Solving the equation
lemma solve_x : ∃ x : ℝ, travel_equation x :=
begin
  use 60 / 11,
  sorry, -- actual proof steps would be provided here
end

-- Calculating individual times for B and C to synchronize
noncomputable def B_bike_distance (x : ℝ) : ℝ := distance_MN - (2 * x)
noncomputable def B_bike_time (x : ℝ) : ℝ := travel_time_bike (B_bike_distance x) biking_speed
noncomputable def C_walk_distance (x : ℝ) : ℝ := x
noncomputable def C_walk_time (x : ℝ) : ℝ := travel_time_walk (C_walk_distance x) walking_speed

-- Time C should leave before A and B
noncomputable def C_start_time (x : ℝ) : ℝ := C_walk_time x - B_bike_time x

-- Final theorem
theorem C_optimal_start_time : C_start_time (60 / 11) = 3 / 11 :=
begin
  unfold C_start_time,
  unfold C_walk_time,
  unfold travel_time_walk,
  unfold B_bike_time,
  unfold travel_time_bike,
  unfold B_bike_distance,
  rw [show 60 / 11 / walking_speed = 10 / 11, by norm_num],
  rw [show (distance_MN - 60 / 11) / biking_speed = 7 / 11, by norm_num],
  norm_num,
end

end C_optimal_start_time_l180_180555


namespace sum_of_odd_binom_coeff_eq_256_constant_term_eq_84_l180_180863

noncomputable theory

-- Definitions based on the conditions
def binom_coeff (n k : ℕ) : ℕ := Nat.choose n k
def binom_exp (x : ℚ) (n : ℕ) : ℚ := x + 1 / x^(1/2)

-- Statement for the first part of the problem
theorem sum_of_odd_binom_coeff_eq_256 : 
  ∑ i in List.range 10, if i % 2 = 1 then binom_coeff 9 i else 0 = 2^8 :=
sorry

-- Statement for the second part of the problem
theorem constant_term_eq_84 :
  binom_coeff 9 6 = 84 :=
sorry

end sum_of_odd_binom_coeff_eq_256_constant_term_eq_84_l180_180863


namespace correct_factorization_l180_180408

theorem correct_factorization :
  (∀ a x y, a * (x + y) = a * x + a * y) ∧
  (∀ x, x^2 - 4 * x + 4 ≠ x * (x - 4) + 4) ∧
  (∀ x, 10 * x^2 - 5 * x = 5 * x * (2 * x - 1)) ∧
  (∀ x, x^2 - 16 + 3 * x ≠ (x - 4) * (x + 4) + 3 * x) →
  "Xia Bohua is correct in the factorization problem." :=
begin
  assume h,
  sorry
end

end correct_factorization_l180_180408


namespace union_of_sets_l180_180770

noncomputable def M : Set ℝ := { x | log (x - 2) ≤ 0 }
def N : Set ℝ := { x | -1 ≤ x ∧ x ≤ 3 }

theorem union_of_sets :
  M ∪ N = { x : ℝ | -1 ≤ x ∧ x ≤ 3 } :=
by
  sorry

end union_of_sets_l180_180770


namespace range_of_a_l180_180918

def p (a : ℝ) : Prop := 0 < a ∧ a < 1
def q (a : ℝ) : Prop := a > 1 / 4

theorem range_of_a (a : ℝ) (h1 : p a ∨ q a) (h2 : ¬(p a ∧ q a)) : a ∈ Set.Ioc 0 (1 / 4) ∨ a ∈ Set.Ioi 1 :=
by
  sorry

end range_of_a_l180_180918


namespace fair_game_expected_winnings_l180_180178

theorem fair_game_expected_winnings (num_players : ℕ) (total_pot : ℝ) 
  (p : ℕ → ℝ) (stakes : ℕ → ℝ) :
  num_players = 36 →
  (∀ k, p k = (35 / 36) ^ (k - 1) * p 1) →
  (∀ k, stakes k = total_pot * p k) →
  (∀ k, let L_k := stakes k in total_pot * p k - L_k * p k - L_k + L_k * p k = 0) :=
sorry

end fair_game_expected_winnings_l180_180178


namespace total_number_of_shuttlecocks_l180_180206

theorem total_number_of_shuttlecocks (n_students : ℕ) (n_shuttlecocks_per_student : ℕ) (no_leftover : 24 * 19 = 456) : (24 = n_students) ∧ (19 = n_shuttlecocks_per_student) → 24 * 19 = 456 :=
by
  intros h
  exact no_leftover
  sorry

end total_number_of_shuttlecocks_l180_180206


namespace points_concyclic_l180_180082

-- Definitions of geometric properties and points
variables (A B C D K L P Q : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace K]
  [MetricSpace L] [MetricSpace P] [MetricSpace Q]

-- Trapezoid Condition
axiom trapezoid_condition : ∀ (A B C D : Type), parallel AB CD → AB > CD

-- Ratio Condition
axiom ratio_condition : ∀ (A B C D K L : Type) (h1 : K ∈ segment A B) (h2 : L ∈ segment C D),
  (dist A K / dist K B) = (dist D L / dist L C)

-- Angle Condition 1
axiom angle_condition_1 : ∀ (A B C D K L P : Type), (P ∈ segment K L) → angle A P B = angle B C D

-- Angle Condition 2
axiom angle_condition_2 : ∀ (A B C D K L Q : Type), (Q ∈ segment K L) → angle C Q D = angle A B C

-- The goal: Prove that points P, Q, B, C are concyclic
theorem points_concyclic : P ∈ (circumcircle B C Q) :=
by { sorry }

end points_concyclic_l180_180082


namespace exponent_of_5_in_30_factorial_l180_180012

theorem exponent_of_5_in_30_factorial : 
  (nat.factors 30!).count 5 = 7 :=
sorry

end exponent_of_5_in_30_factorial_l180_180012


namespace area_triangle_ABC_correct_l180_180289

noncomputable def rectangle_area : ℝ := 42

noncomputable def area_triangle_outside_I : ℝ := 9
noncomputable def area_triangle_outside_II : ℝ := 3.5
noncomputable def area_triangle_outside_III : ℝ := 12

noncomputable def area_triangle_ABC : ℝ :=
  rectangle_area - (area_triangle_outside_I + area_triangle_outside_II + area_triangle_outside_III)

theorem area_triangle_ABC_correct : area_triangle_ABC = 17.5 := by 
  sorry

end area_triangle_ABC_correct_l180_180289


namespace base7_to_base10_conversion_l180_180528

def convert_base_7_to_10 := 243

namespace Base7toBase10

theorem base7_to_base10_conversion :
  2 * 7^2 + 4 * 7^1 + 3 * 7^0 = 129 := by
  -- The original number 243 in base 7 is expanded and evaluated to base 10.
  sorry

end Base7toBase10

end base7_to_base10_conversion_l180_180528


namespace equal_sides_of_inscribed_pentagon_l180_180115

theorem equal_sides_of_inscribed_pentagon 
  (A B C D E : Type)
  [metric_space Type]
  [circle_inscribed (A B C D E : fin 5)]
  (angle_A angle_B angle_C angle_D angle_E : ℝ) 
  (h1 : angle_A = angle_B)
  (h2 : angle_B = angle_C)
  (h3 : angle_C = angle_D)
  (h4 : angle_D = angle_E) :
  side_length A B = side_length B C ∧
  side_length B C = side_length C D ∧
  side_length C D = side_length D E ∧
  side_length D E = side_length E A :=
begin
  sorry
end

end equal_sides_of_inscribed_pentagon_l180_180115


namespace circle_equation_coefficients_l180_180144

theorem circle_equation_coefficients (a : ℝ) (x y : ℝ) : 
  (a^2 * x^2 + (a + 2) * y^2 + 2 * a * x + a = 0) → (a = -1) :=
by 
  sorry

end circle_equation_coefficients_l180_180144


namespace sequence_sum_l180_180504

def general_term (n : ℕ) : ℚ := (2 * n - 1) + (1 / (2 ^ n))

def sum_of_first_n_terms (n : ℕ) : ℚ :=
  ∑ k in finset.range(n), general_term (k + 1)

theorem sequence_sum (n : ℕ) : 
  sum_of_first_n_terms n = n^2 - (1 / (2^n)) + 1 :=
by sorry

end sequence_sum_l180_180504


namespace teacher_total_score_l180_180262

/-- Conditions -/
def written_test_score : ℝ := 80
def interview_score : ℝ := 60
def written_test_weight : ℝ := 0.6
def interview_weight : ℝ := 0.4

/-- Prove the total score -/
theorem teacher_total_score :
  written_test_score * written_test_weight + interview_score * interview_weight = 72 :=
by
  sorry

end teacher_total_score_l180_180262


namespace max_b_value_l180_180766
noncomputable theory

-- Definitions of the functions f and g
def f (x a : ℝ) : ℝ := (1 / 2) * x^2 + 2 * a * x
def g (x a b : ℝ) : ℝ := 3 * a^2 * real.log x + b

-- Derivatives of the functions f and g
def f' (x a : ℝ) : ℝ := x + 2 * a
def g' (x a : ℝ) : ℝ := (3 * a^2) / x

-- Function h derived from b = (5/2)a^2 - 3a^2 ln a
def h (t : ℝ) : ℝ := (5 / 2) * t^2 - 3 * t^2 * real.log t

-- Theorem statement
theorem max_b_value (a : ℝ) (h₁ : a > 0) :
  (∃ x, f x a = g x a_ b ∧ f' x a = g' x a) →
  (∀ a > 0, b = (5 / 2) * a^2 - 3 * a^2 * real.log a) →
  ∃ b, b = (3 / 2) * real.exp (2 / 3) :=
sorry

end max_b_value_l180_180766


namespace speed_black_car_l180_180520

-- Definitions and conditions
def speed_red_car : ℝ := 40
def distance_ahead : ℝ := 30
def time_to_overtake : ℝ := 3

-- The theorem stating that the speed of the black car is 50 miles per hour
theorem speed_black_car : (let total_distance := distance_ahead + speed_red_car * time_to_overtake in total_distance / time_to_overtake = 50) :=
sorry

end speed_black_car_l180_180520


namespace least_value_solution_l180_180529

theorem least_value_solution :
  ∃ x : ℝ, 5 * x^2 + 7 * x + 3 = 6 ∧ x = (-(7 : ℝ) - real.sqrt 109) / 10 :=
by
  sorry

end least_value_solution_l180_180529


namespace cost_of_bottle_with_cork_l180_180233

theorem cost_of_bottle_with_cork (C : ℝ) (W : ℝ) (H1 : C = 2.05) (H2 : W = (C + 2.00) + C) : W = 6.10 :=
by {
  sorry,
}

end cost_of_bottle_with_cork_l180_180233


namespace prob_product_not_gt_4_prob_diff_less_2_l180_180511

open MeasureTheory

noncomputable def part_i (balls : Finset ℕ) : ℚ :=
  let outcomes := {(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)}.to_finset in
  let favorable := {(1, 2), (1, 3), (1, 4)}.to_finset in
  (favorable.card : ℚ) / outcomes.card

theorem prob_product_not_gt_4 : part_i {1, 2, 3, 4}.to_finset = 1/2 := by
  sorry

noncomputable def part_ii (balls : Finset ℕ) : ℚ :=
  let outcomes := (balls.product balls).filter (λ p, |p.1 - p.2| < 2) in
  (outcomes.card : ℚ) / (balls.card * balls.card)

theorem prob_diff_less_2 : part_ii {1, 2, 3, 4}.to_finset = 5/8 := by
  sorry

end prob_product_not_gt_4_prob_diff_less_2_l180_180511


namespace find_acute_angle_l180_180376

variables (θ : ℝ)
noncomputable def a := (1 - Real.sin θ, 1)
noncomputable def b := (1 / 2, 1 + Real.sin θ)

def vectors_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

theorem find_acute_angle (h : vectors_parallel (a θ) (b θ)) : θ = Real.pi / 4 :=
sorry

end find_acute_angle_l180_180376


namespace star_value_when_c_2_d_3_l180_180370

def star (c d : ℕ) : ℕ := c^3 + 3*c^2*d + 3*c*d^2 + d^3

theorem star_value_when_c_2_d_3 :
  star 2 3 = 125 :=
by
  sorry

end star_value_when_c_2_d_3_l180_180370


namespace semi_minor_axis_length_l180_180857

theorem semi_minor_axis_length (center focus endpoint : ℝ × ℝ)
  (hc : center = (0, 4)) 
  (hf : focus = (0, 1)) 
  (he : endpoint = (0, 9)) : 
  let c := Real.sqrt ((center.1 - focus.1)^2 + (center.2 - focus.2)^2) in
  let a := Real.sqrt ((center.1 - endpoint.1)^2 + (center.2 - endpoint.2)^2) in
  let b := Real.sqrt (a^2 - c^2) in
  b = 4 :=
by
  sorry

end semi_minor_axis_length_l180_180857


namespace min_number_of_gennadys_l180_180663

theorem min_number_of_gennadys (a b v g : ℕ) (h_a : a = 45) (h_b : b = 122) (h_v : v = 27)
    (h_needed_g : g = 49) :
    (b - 1) - (a + v) = g :=
by
  -- We include sorry because we are focusing on the statement, not the proof itself.
  sorry

end min_number_of_gennadys_l180_180663


namespace measure_of_angle_A_l180_180342

variable (a b c A B C : ℝ)

noncomputable def problem_statement : Prop :=
  (a + b) * (Real.sin A - Real.sin B) = (c - b) * Real.sin C →
  A = Real.pi / 3

theorem measure_of_angle_A (h : problem_statement a b c A B C) : A = Real.pi / 3 :=
sorry

end measure_of_angle_A_l180_180342


namespace tile_arrangement_possible_l180_180509

/-- 
There are 20 tiles, each divided into four quarters that may
be painted in different colors. We need to select 16 tiles 
and arrange them into a 4x4 grid such that adjacent tiles 
share the same color in their touching quarters.
-/
theorem tile_arrangement_possible 
  (tiles : Fin 20 → Tile)
  (color_match : ∀ (t1 t2 : Tile), touching_quarters t1 t2 → same_color t1 t2) :
  ∃ (selected_tiles : Fin 16 → Tile), valid_4x4_arrangement selected_tiles color_match :=
sorry

-- Definitions required for understanding
structure Tile :=
  (quarters : Fin 4 → Color)

def touching_quarters (t1 t2 : Tile) : Prop := sorry -- define when two tiles touch

def same_color (t1 t2 : Tile) : Prop := sorry -- define when two touching tiles have same color

def valid_4x4_arrangement (tiles : Fin 16 → Tile) 
    (color_match : ∀ (t1 t2 : Tile), touching_quarters t1 t2 → same_color t1 t2) : Prop := sorry -- define valid arrangement

end tile_arrangement_possible_l180_180509


namespace exponent_of_5_in_30_factorial_l180_180016

theorem exponent_of_5_in_30_factorial : 
  (nat.factors 30!).count 5 = 7 :=
sorry

end exponent_of_5_in_30_factorial_l180_180016


namespace sec_minus_tan_l180_180822

theorem sec_minus_tan
  (x : ℝ)
  (h1 : Real.sec x + Real.tan x = 7 / 3)
  (h2 : (Real.sec x + Real.tan x) * (Real.sec x - Real.tan x) = 1) :
  Real.sec x - Real.tan x = 3 / 7 :=
by
  sorry

end sec_minus_tan_l180_180822


namespace find_b_plus_k_l180_180274

open Real

noncomputable def semi_major_axis (f1 f2 : ℝ × ℝ) (p : ℝ × ℝ) : ℝ :=
  dist p f1 + dist p f2

def c_squared (a : ℝ) (b : ℝ) : ℝ :=
  a ^ 2 - b ^ 2

theorem find_b_plus_k :
  ∀ (f1 f2 : ℝ × ℝ) (p : ℝ × ℝ) (h k : ℝ) (a b : ℝ),
  f1 = (-2, 0) →
  f2 = (2, 0) →
  p = (6, 0) →
  (∃ a b, semi_major_axis f1 f2 p = 2 * a ∧ c_squared a b = 4) →
  h = 0 →
  k = 0 →
  b = 4 * sqrt 2 →
  b + k = 4 * sqrt 2 :=
by
  intros f1 f2 p h k a b f1_def f2_def p_def maj_axis_def h_def k_def b_def
  rw [b_def, k_def]
  exact add_zero (4 * sqrt 2)

end find_b_plus_k_l180_180274


namespace discount_percentage_in_february_l180_180248

theorem discount_percentage_in_february (C : ℝ) (h1 : C > 0) 
(markup1 : ℝ) (markup2 : ℝ) (profit : ℝ) (D : ℝ) :
  markup1 = 0.20 → markup2 = 0.25 → profit = 0.125 →
  1.50 * C * (1 - D) = 1.125 * C → D = 0.25 :=
by
  intros
  sorry

end discount_percentage_in_february_l180_180248


namespace wire_square_length_l180_180542

theorem wire_square_length (length_wire : ℕ) (h : length_wire = 60) : length_wire / 4 = 15 :=
by
  rw [h]
  exact Nat.div_eq_of_eq_mul_left (by norm_num : 0 < 4) rfl

end wire_square_length_l180_180542


namespace exponent_of_5_in_30_factorial_l180_180010

theorem exponent_of_5_in_30_factorial : 
  (nat.factors 30!).count 5 = 7 :=
sorry

end exponent_of_5_in_30_factorial_l180_180010


namespace remainder_of_combinations_l180_180979

theorem remainder_of_combinations (m n : ℕ) (a : Fin 10 → ℕ) 
  (h1 : ∀ i, 1 ≤ a i)
  (h2 : ∀ i j, i ≤ j → a i ≤ a j)
  (h3 : ∀ i, (a i - (i + 1)) % 2 = 0)  -- i + 1 because Fin 10 ranges from 0 to 9
  (h4 : (∑ i, a i) % 3 = 0)
  (total_combinations : m = choose 1014 10) : 
  m % 1000 = 662 := 
sorry

end remainder_of_combinations_l180_180979


namespace trajectory_of_Q_l180_180357

-- Define the points A, P, and Q
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 4, y := 0 }

-- Define the conditions
def onCircle (P : Point) : Prop :=
  P.x^2 + P.y^2 = 4

def AQ_equal_2QP (A Q P : Point) : Prop :=
  (Q.x - A.x, Q.y - A.y) = (2 * (P.x - Q.x), 2 * (P.y - Q.y))

-- Define the trajectory equation to prove
def trajectory_eq (Q : Point) : Prop :=
  (Q.x - 4/3)^2 + Q.y^2 = 16/9

-- Statement to prove
theorem trajectory_of_Q :
  ∃ Q : Point, ∃ P : Point, AQ_equal_2QP A Q P ∧ onCircle P → trajectory_eq Q :=
by
  intros Q P h₁ h₂
  sorry

end trajectory_of_Q_l180_180357


namespace min_gennadys_needed_l180_180656

variables (A B V G : ℕ)

theorem min_gennadys_needed
  (hA : A = 45)
  (hB : B = 122)
  (hV : V = 27)
  (hG : ∀ i, i < 121 → A + V < 121 → G ≥ 49) :
  G = 49 :=
sorry

end min_gennadys_needed_l180_180656


namespace octagon_diagonals_l180_180951

def num_sides := 8

def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem octagon_diagonals : num_diagonals num_sides = 20 :=
by
  sorry

end octagon_diagonals_l180_180951


namespace camp_weights_l180_180601

theorem camp_weights (m_e_w : ℕ) (m_e_w1 : ℕ) (c_w : ℕ) (m_e_w2 : ℕ) (d : ℕ)
  (h1 : m_e_w = 30) 
  (h2 : m_e_w1 = 28) 
  (h3 : c_w = 56)
  (h4 : m_e_w = m_e_w1 + d)
  (h5 : m_e_w1 = m_e_w2 + d)
  (h6 : c_w = m_e_w + m_e_w1 + d) :
  m_e_w = 28 ∧ m_e_w2 = 26 := 
by {
    sorry
}

end camp_weights_l180_180601


namespace length_sum_three_leq_three_l180_180735

-- Define the given vectors and their properties
variables {V : Type*} [inner_product_space ℝ V]
variables (v1 v2 v3 : V)

-- Given condition: For any two vectors, the length of their sum does not exceed 2
axiom given_condition : ∀ (u v : V), ∥u + v∥ ≤ 2

-- The theorem to prove: The length of the sum of any three vectors does not exceed 3
theorem length_sum_three_leq_three (v1 v2 v3 : V) :
  ∥v1 + v2 + v3∥ ≤ 3 :=
sorry

end length_sum_three_leq_three_l180_180735


namespace cos_three_theta_l180_180840

open Complex

theorem cos_three_theta (θ : ℝ) (h : cos θ = 1 / 2) : cos (3 * θ) = -1 / 2 :=
by
  sorry

end cos_three_theta_l180_180840


namespace problem_y_value_l180_180838

theorem problem_y_value :
  let y := (log 4 / log 3) * (log 5 / log 4) * ... * (log 15 / log 14)
  in y = 1 + log 5 / log 3 := 
sorry

end problem_y_value_l180_180838


namespace minimal_product_plus_100_l180_180109

-- Proposition: the minimal product of digits 4, 6, 7, and 9 placed in two 2-digit numbers plus 100 equals 3343.
theorem minimal_product_plus_100 : 
  ∃ (d1 d2 d3 d4 : ℕ), {d1, d2, d3, d4} = {4, 6, 7, 9} ∧ 
  let p1 := 10 * d1 + d2,
      p2 := 10 * d3 + d4,
      p3 := 10 * d1 + d3,
      p4 := 10 * d2 + d4 in
  let product1 := p1 * p2,
      product2 := p3 * p4 in
  min product1 product2 + 100 = 3343 :=
sorry

end minimal_product_plus_100_l180_180109


namespace cos_sin_gt_sin_cos_l180_180328

theorem cos_sin_gt_sin_cos (x : ℝ) (hx : 0 ≤ x ∧ x ≤ real.pi) : 
  real.cos (real.sin x) > real.sin (real.cos x) :=
sorry

end cos_sin_gt_sin_cos_l180_180328


namespace sec_minus_tan_l180_180811

-- Define the problem in Lean 4
theorem sec_minus_tan (x : ℝ) (h : Real.sec x + Real.tan x = 7 / 3) : Real.sec x - Real.tan x = 3 / 7 := by
  -- One could also include here the necessary mathematical facts and connections.
  sorry -- Proof to be provided

end sec_minus_tan_l180_180811


namespace num_ints_satisfying_condition_l180_180779

theorem num_ints_satisfying_condition :
  ∃ (n : ℕ), 200 < n ∧ n < 300 ∧ (∃ r, n % 7 = r ∧ n % 9 = r)  
    = 7 :=
sorry

end num_ints_satisfying_condition_l180_180779


namespace point_on_line_solve_for_a_l180_180744

theorem point_on_line_solve_for_a 
  (h : ∃ a : ℝ, (a, 1) ∈ set_of (λ p : ℝ × ℝ, p.2 = 3 * p.1 + 4)) 
  : ∃ a : ℝ, a = -1 :=
by
  rcases h with ⟨a, h1⟩
  sorry

end point_on_line_solve_for_a_l180_180744


namespace rotated_graph_equation_l180_180489

theorem rotated_graph_equation (x y : ℝ) (h : y = exp x) : x = exp (-y) :=
by
  sorry

end rotated_graph_equation_l180_180489


namespace fifth_pythagorean_triple_l180_180137

theorem fifth_pythagorean_triple : ∃ a b c : ℕ, a % 2 = 1 ∧ b = a^2 - 1 ∧ c = a^2 + 1 ∧ a^2 + b^2 = c^2 :=
by
  let a := 11
  let b := 60
  let c := 61
  have h1: a % 2 = 1 := by
    sorry -- a is an odd number
  have h2: b = a^2 - 1 := by
    sorry -- 60 = 11^2 - 1
  have h3: c = a^2 + 1 := by
    sorry -- 61 = 11^2 + 1
  have h4: a^2 + b^2 = c^2 := by
    sorry -- 11^2 + 60^2 = 61^2
  existsi [a, b, c]
  exact ⟨h1, h2, h3, h4⟩

end fifth_pythagorean_triple_l180_180137


namespace num_8_step_paths_l180_180543

def is_black (r c : ℕ) : Prop :=
  (r + c) % 2 = 1

def is_white (r c : ℕ) : Prop :=
  (r + c) % 2 = 0

def valid_step (r c dr dc : ℕ) : Prop :=
  r < 8 ∧ c < 8 ∧ dr = 1 ∧ (dc = -1 ∨ dc = 0 ∨ dc = 1)

def valid_path (path : List (ℕ × ℕ)) : Prop :=
  path.length = 9 ∧
  is_white 7 (path.head!.2) ∧
  List.pairwise (λ ⟨r1, c1⟩ ⟨r2, c2⟩, valid_step r1 c1 (r2 - r1) (c2 - c1) ∧ is_black (r2 - 1) c2 ∧ is_white r2 c2) path.tail!

theorem num_8_step_paths (P Q : ℕ × ℕ) (hP : P = (7, 0) ∧ is_white 7 0) (hQ : Q = (0, 7) ∧ is_black 0 7) :
  (List.filter valid_path (List.permutations [0, 0, 0, 0, 1, 1, 1, 1])).length = 70 := 
by
  sorry

end num_8_step_paths_l180_180543


namespace exponent_of_five_in_30_factorial_l180_180042

theorem exponent_of_five_in_30_factorial : 
  nat.factorial_prime_exponent 30 5 = 7 := 
sorry

end exponent_of_five_in_30_factorial_l180_180042


namespace simplify_expression_l180_180544

theorem simplify_expression (x : ℝ) : (x^2 - 4) * (x - 2) * (x + 2) = x^4 - 8x^2 + 16 :=
by
  sorry

end simplify_expression_l180_180544


namespace minimum_gennadys_l180_180623

theorem minimum_gennadys (alexs borises vasilies x : ℕ) (h₁ : alexs = 45) (h₂ : borises = 122) (h₃ : vasilies = 27)
    (h₄ : ∀ i, i ∈ list.range (borises-1) → alexs + vasilies + x > i) : 
    x = 49 :=
by 
    sorry

end minimum_gennadys_l180_180623


namespace y_is_multiple_of_2_y_is_multiple_of_3_y_is_multiple_of_6_y_is_multiple_of_9_l180_180892

def y : ℕ := 54 + 108 + 162 + 216 + 648 + 810 + 972

theorem y_is_multiple_of_2 : 2 ∣ y :=
sorry

theorem y_is_multiple_of_3 : 3 ∣ y :=
sorry

theorem y_is_multiple_of_6 : 6 ∣ y :=
sorry

theorem y_is_multiple_of_9 : 9 ∣ y :=
sorry

end y_is_multiple_of_2_y_is_multiple_of_3_y_is_multiple_of_6_y_is_multiple_of_9_l180_180892


namespace parabola_vertex_l180_180145

theorem parabola_vertex (x y : ℝ) :
  (x^2 - 4 * x + 3 * y + 8 = 0) → (x, y) = (2, -4 / 3) :=
by
  sorry

end parabola_vertex_l180_180145


namespace division_and_multiply_l180_180681

theorem division_and_multiply :
  (-128) / (-16) * 5 = 40 := 
by
  sorry

end division_and_multiply_l180_180681


namespace sec_sub_tan_l180_180796

theorem sec_sub_tan (x : ℝ) (h : sec x + tan x = 7 / 3) : sec x - tan x = 3 / 7 := by
  sorry

end sec_sub_tan_l180_180796


namespace probability_girl_selection_l180_180257

-- Define the conditions
def total_candidates : ℕ := 3 + 1
def girl_candidates : ℕ := 1

-- Define the question in terms of probability
def probability_of_selecting_girl (total: ℕ) (girl: ℕ) : ℚ :=
  girl / total

-- Lean statement to prove
theorem probability_girl_selection : probability_of_selecting_girl total_candidates girl_candidates = 1 / 4 :=
by
  sorry

end probability_girl_selection_l180_180257
