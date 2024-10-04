import Mathlib

namespace total_number_of_outliers_is_one_l725_725791

def dataset : List ℕ := [8, 20, 36, 36, 44, 46, 46, 48, 56, 62]
def Q1 : ℕ := 36
def Q3 : ℕ := 48

def IQR : ℕ := Q3 - Q1
def lower_threshold : ℕ := Q1 - 1.5 * IQR
def upper_threshold : ℕ := Q3 + 1.5 * IQR

def is_outlier (x : ℕ) : Prop :=
  x < lower_threshold ∨ x > upper_threshold

def num_outliers : ℕ :=
  dataset.countp is_outlier

theorem total_number_of_outliers_is_one :
  num_outliers = 1 :=
sorry

end total_number_of_outliers_is_one_l725_725791


namespace area_of_triangle_ABC_l725_725999

-- Given the constants in the problem.
def r : ℝ := 3
def bd : ℝ := 5
def ed : ℝ := 7
def AD : ℝ := r + r + bd -- 3 + 3 + 5
def AE : ℝ := Real.sqrt (AD^2 + ed^2) -- sqrt(11^2 + 7^2)
def EO : ℝ := r
def OD : ℝ := bd

-- Using the Power of a Point theorem.
def power_of_point_CA : ℝ := (AD^2 + ed^2 - (EO + OD)^2)
def CA : ℝ := power_of_point_CA / AE
def AC : ℝ := AE - CA
def BC : ℝ := Real.sqrt (r^2 - CA^2)

-- The Lean statement theorem to prove the area.
theorem area_of_triangle_ABC : (1 / 2) * AC * BC = 576 / 85 := 
by
  sorry

end area_of_triangle_ABC_l725_725999


namespace fraction_value_l725_725478

theorem fraction_value
  (m n : ℕ)
  (h : m / n = 2 / 3) :
  m / (m + n) = 2 / 5 :=
sorry

end fraction_value_l725_725478


namespace Phi_area_le_0_34_Phi_area_le_0_287_l725_725318

-- Definition of unit square and figure
def unit_square := set.univ : set (ℝ × ℝ)
variable (Phi : set (ℝ × ℝ))
variable (hPhi_in_K : Phi ⊆ unit_square)
variable (hPhi_no_close_pts : ∀ (x y ∈ Phi), dist x y ≠ 0.001)

-- Part (a) statement
theorem Phi_area_le_0_34 (hPhi_in_K : Phi ⊆ unit_square)
  (hPhi_no_close_pts : ∀ (x y ∈ Phi), dist x y ≠ 0.001) : 
  measure (Phi) ≤ 0.34 := 
sorry

-- Part (b) statement
theorem Phi_area_le_0_287 (hPhi_in_K : Phi ⊆ unit_square)
  (hPhi_no_close_pts : ∀ (x y ∈ Phi), dist x y ≠ 0.001) : 
  measure (Phi) ≤ 0.287 := 
sorry

end Phi_area_le_0_34_Phi_area_le_0_287_l725_725318


namespace table_properties_l725_725200

theorem table_properties (n : ℕ) (table : Matrix (Fin 2015) (Fin n) ℕ) :
  (∀ i : Fin 2015, ∃ j : Fin n, table i j > 0) →
  (∀ j : Fin n, ∃ i : Fin 2015, table i j > 0) →
  (∀ i : Fin 2015, ∀ j : Fin n, (table i j > 0 → (∑ x, table i x) = (∑ y, table y j))) →
  n = 2015 :=
  sorry

end table_properties_l725_725200


namespace gcd_of_given_lengths_l725_725396

def gcd_of_lengths_is_eight : Prop :=
  let lengths := [48, 64, 80, 120]
  ∃ d, d = 8 ∧ (∀ n ∈ lengths, d ∣ n)

theorem gcd_of_given_lengths : gcd_of_lengths_is_eight := 
  sorry

end gcd_of_given_lengths_l725_725396


namespace slope_AF_is_neg_four_thirds_l725_725118

noncomputable def point (x y : ℝ) : Prop := (x, y)

variable (x y : ℝ)

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  ( (x2 - x1)^2 + (y2 - y1)^2 )^.sqrt

def slope (x1 y1 x2 y2 : ℝ) : ℝ :=
  (y2 - y1) / (x2 - x1)

theorem slope_AF_is_neg_four_thirds 
  (x_A y_A : ℝ) 
  (hx : parabola x_A y_A) 
  (hx_neg : x_A < 0)
  (hx_dir : x_A + 1 = 5) :
  slope 1 0 x_A y_A = -4 / 3 :=
by
  sorry

end slope_AF_is_neg_four_thirds_l725_725118


namespace eval_expression_l725_725067

theorem eval_expression : 3 - (-3)^(3 - (-3)) = -726 := by
  sorry

end eval_expression_l725_725067


namespace face_opposite_to_A_l725_725744

-- Define the faces and their relationships
inductive Face : Type
| A | B | C | D | E | F
open Face

def adjacent (x y : Face) : Prop :=
  match x, y with
  | A, B => true
  | B, A => true
  | C, A => true
  | A, C => true
  | D, A => true
  | A, D => true
  | C, D => true
  | D, C => true
  | E, F => true
  | F, E => true
  | _, _ => false

-- Theorem stating that "F" is opposite to "A" given the provided conditions.
theorem face_opposite_to_A : ∀ x : Face, (adjacent A x = false) → (x = B ∨ x = C ∨ x = D → false) → (x = E ∨ x = F) → x = F := 
  by
    intros x h1 h2 h3
    sorry

end face_opposite_to_A_l725_725744


namespace tipping_condition_satisfied_l725_725768

variables {M a b θ μ_s : ℝ}

-- Definitions based on given conditions
def normal_force (mass : ℝ) (incline_angle : ℝ) : ℝ := mass * Real.cos incline_angle
def frictional_force (μ_s : ℝ) (normal : ℝ) : ℝ := μ_s * normal
def gravitational_component (mass : ℝ) (incline_angle : ℝ) : ℝ := mass * Real.sin incline_angle
def critical_angle (base height : ℝ) : ℝ := Real.atan (base / height)

-- Condition for no slipping
def no_slipping_condition (μ_s : ℝ) (mass : ℝ) (incline_angle : ℝ) : Prop :=
  μ_s * normal_force mass incline_angle ≥ gravitational_component mass incline_angle

-- Condition for tipping
def tipping_condition (base height : ℝ) : ℝ := base / height

-- The proof problem:
theorem tipping_condition_satisfied (hM : M > 0) (ha : a > 0) (hb : b > 0) (h_ineq: tipping_condition b a ≤ μ_s) :
  μ_s ≥ tipping_condition b a :=
sorry

end tipping_condition_satisfied_l725_725768


namespace total_daisies_l725_725558

-- Define the initial conditions
def white_daisies : Nat := 6
def pink_daisies : Nat := 9 * white_daisies
def red_daisies : Nat := 4 * pink_daisies - 3

-- The main theorem stating that the total number of daisies is 273
theorem total_daisies : white_daisies + pink_daisies + red_daisies = 273 := by
  -- The proof is left as an exercise
  sorry

end total_daisies_l725_725558


namespace tank_capacity_l725_725368

theorem tank_capacity (initial_fraction final_fraction: ℚ) (add_gallons : ℕ) (total_capacity : ℕ) :
  initial_fraction = 1 / 4 → final_fraction = 3 / 4 → add_gallons = 200 → 
  total_capacity * (final_fraction - initial_fraction) = add_gallons → 
  total_capacity = 400 :=
by
  intros h1 h2 h3 h4
  have h5 : final_fraction - initial_fraction = 1 / 2, by linarith [h1, h2]
  rw h5 at h4
  simp at h4
  linarith

end tank_capacity_l725_725368


namespace combination_k_values_l725_725831

theorem combination_k_values (k : ℕ) (h : nat.choose 18 k = nat.choose 18 (2 * k - 3)) :
  k = 3 ∨ k = 7 :=
sorry

end combination_k_values_l725_725831


namespace total_daisies_l725_725564

-- Define the conditions
def white_daisies : ℕ := 6
def pink_daisies : ℕ := 9 * white_daisies
def red_daisies : ℕ := 4 * pink_daisies - 3

-- Main statement to be proved
theorem total_daisies : white_daisies + pink_daisies + red_daisies = 273 := by
  sorry

end total_daisies_l725_725564


namespace area_of_bounded_region_l725_725427

theorem area_of_bounded_region :
  let region := {p : ℝ × ℝ | (p.1 = 0 ∧ p.2 ≥ 0) ∨ (p.2 = 0 ∧ p.1 ≥ 0) ∨ (p.1 = 2) ∨ (p.2 = 2)}
  ∃ (s : set (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ s ↔ (y = x))
    ∧ is_square s 
    ∧ area s = 4 := 
sorry

end area_of_bounded_region_l725_725427


namespace circle_a_lt_8_tangent_lines_perpendicular_circle_intersection_l725_725501

-- Problem (1)
theorem circle_a_lt_8 (x y a : ℝ) (h : x^2 + y^2 - 4*x - 4*y + a = 0) : 
  a < 8 :=
by
  sorry

-- Problem (2)
theorem tangent_lines (a : ℝ) (h : a = -17) : 
  ∃ (k : ℝ), k * 7 - 6 - 7 * k = 0 ∧
  ((39 * k + 80 * (-7) - 207 = 0) ∨ (k = 7)) :=
by
  sorry

-- Problem (3)
theorem perpendicular_circle_intersection (x1 x2 y1 y2 a : ℝ) 
  (h1: 2 * x1 - y1 - 3 = 0) 
  (h2: 2 * x2 - y2 - 3 = 0) 
  (h3: x1 * x2 + y1 * y2 = 0) 
  (hpoly : 5 * x1 * x2 - 6 * (x1 + x2) + 9 = 0): 
  a = -6 / 5 :=
by
  sorry

end circle_a_lt_8_tangent_lines_perpendicular_circle_intersection_l725_725501


namespace base_b_prime_digits_l725_725326

theorem base_b_prime_digits (b' : ℕ) (h1 : b'^4 ≤ 216) (h2 : 216 < b'^5) : b' = 3 :=
by {
  sorry
}

end base_b_prime_digits_l725_725326


namespace max_perimeter_isosceles_triangle_l725_725614

/-- Out of all triangles with the same base and the same angle at the vertex, 
    the triangle with the largest perimeter is isosceles -/
theorem max_perimeter_isosceles_triangle {α β γ : ℝ} (b : ℝ) (B : ℝ) (A C : ℝ) 
  (hB : 0 < B ∧ B < π) (hβ : α + C = B) (h1 : A = β) (h2 : γ = β) :
  α = γ := sorry

end max_perimeter_isosceles_triangle_l725_725614


namespace exists_circle_intersecting_B_R_exactly_l725_725952

-- Definitions based on the conditions
variable (n : ℕ) (h_pos_n : 0 < n)
variable (lines : list (line ℝ)) (h_lines_length : lines.length = 2 * n)
variable (h_no_parallel : ∀ i j, i ≠ j → ¬parallel (lines.nth_le i sorry) (lines.nth_le j sorry))
variable (colors : list bool) (h_colors_length : colors.length = 2 * n)
variable (h_blue_red_split : colors.count tt = n ∧ colors.count ff = n)

-- Sets of points defined by lines
def B : set (point ℝ) := { p | ∃ i, colors.nth_le i sorry = tt ∧ p ∈ lines.nth_le i sorry }
def R : set (point ℝ) := { p | ∃ i, colors.nth_le i sorry = ff ∧ p ∈ lines.nth_le i sorry }

-- The theorem to be proved
theorem exists_circle_intersecting_B_R_exactly (B R : set (point ℝ)):
  ∃ (C : set (point ℝ)), (C ∩ B).size = (2 * n - 1) ∧ (C ∩ R).size = (2 * n - 1) :=
sorry

end exists_circle_intersecting_B_R_exactly_l725_725952


namespace problem_result_l725_725736

def elongation_A : List ℕ := [545, 533, 551, 522, 575, 544, 541, 568, 596, 548]
def elongation_B : List ℕ := [536, 527, 543, 530, 560, 533, 522, 550, 576, 536]

def z_i : List ℤ := List.zipWith (λ x y => x - y) elongation_A elongation_B

def sample_mean (lst : List ℤ) : ℚ :=
  (List.sum lst : ℚ) / List.length lst

def sample_variance (lst : List ℤ) : ℚ :=
  let mean := sample_mean lst
  (List.sum (lst.map (λ z => (z - mean) * (z - mean))) : ℚ) / List.length lst

def improvement_significance (mean : ℚ) (variance : ℚ) : Prop :=
  mean ≥ 2 * Real.sqrt (variance / 10)

theorem problem_result :
  sample_mean z_i = 11 ∧
  sample_variance z_i = 61 ∧
  improvement_significance (sample_mean z_i) (sample_variance z_i) :=
by
  sorry

end problem_result_l725_725736


namespace least_possible_sections_l725_725364

theorem least_possible_sections (A C N : ℕ) (h1 : 7 * A = 11 * C) (h2 : N = A + C) : N = 18 :=
sorry

end least_possible_sections_l725_725364


namespace minimum_value_S_l725_725404

theorem minimum_value_S :
  ∀ (a : ℕ → ℤ), (∀ i, 1 ≤ i ∧ i ≤ 100 → (a i = 1 ∨ a i = -1)) →
  (∃ m n : ℕ, m + n = 100 ∧ (∑ i in finset.range 100, a i * a i) = 100 ∧
  2 * (∑ i in finset.range 100, ∑ j in finset.Ico i 100,  a i * a j) + 100 =
  (m - n) ^ 2  ∧  (∃ S : ℕ, S = (∑ i in finset.range 100, ∑ j in finset.Ico i 100, a i * a j) ∧ S = 22)) :=
begin
  sorry
end

end minimum_value_S_l725_725404


namespace monotonic_increasing_interval_l725_725058

noncomputable def f (x : ℝ) : ℝ :=
  Real.logb 0.5 (x^2 + 2 * x - 3)

theorem monotonic_increasing_interval :
  ∀ x, f x = Real.logb 0.5 (x^2 + 2 * x - 3) → 
  (∀ x₁ x₂, x₁ < x₂ ∧ x₁ < -3 ∧ x₂ < -3 → f x₁ ≤ f x₂) :=
sorry

end monotonic_increasing_interval_l725_725058


namespace circle_equation_line_equations_l725_725101

noncomputable def point := ℝ × ℝ

def C : point := (-2, 6)
def M : point := (0, 6 - 2 * Real.sqrt 3)
def r : ℝ := Real.sqrt ((0 + 2)^2 + (6 - 2 * Real.sqrt 3 - 6)^2)

theorem circle_equation : (∀ x y : ℝ, (x + 2)^2 + (y - 6)^2 = 16 ↔ (x, y) ∈ {p : point | (p.1 + 2)^2 + (p.2 - 6)^2 = r^2}) :=
sorry

def P : point := (0, 5)
def l_1 : ℝ → ℝ := λ x, (3/4) * x + 5
def l_2 : set point := {p : point | 3 * p.1 - 4 * p.2 + 20 = 0}
def l_3 : set point := {p : point | p.1 = 0}

theorem line_equations : (∀ p : ℝ, ((l_1 p, 5)) = P ∨ ∀ x y : ℝ, (3 * x - 4 * y + 20 = 0) ∨ ∀ x y : ℝ, (x = 0)) :=
sorry

end circle_equation_line_equations_l725_725101


namespace larger_solution_quadratic_l725_725440

theorem larger_solution_quadratic :
  (∃ a b : ℝ, a ≠ b ∧ (a = 9) ∧ (b = -2) ∧
              (∀ x : ℝ, x^2 - 7 * x - 18 = 0 → (x = a ∨ x = b))) →
  9 = max a b :=
by
  sorry

end larger_solution_quadratic_l725_725440


namespace central_cell_value_l725_725545

variable (f : Fin 5 → Fin 5 → ℕ)
hypothesis (h_sum_total : (Finset.univ.image (λ (i : Fin 5 × Fin 5), f i.1 i.2)).sum = 200)
hypothesis (h_sum_1x3 : ∀ (i : Fin 5) (j : Fin 3), f i j + f i (Fin.mk (j.1 + 1 % 5) sorry) + f i (Fin.mk (j.1 + 2 % 5) sorry) = 23)

theorem central_cell_value : f 2 2 = 16 := 
by
  sorry

end central_cell_value_l725_725545


namespace lcm_of_36_and_100_l725_725442

theorem lcm_of_36_and_100 : Nat.lcm 36 100 = 900 :=
by
  -- The proof is omitted
  sorry

end lcm_of_36_and_100_l725_725442


namespace diamond_value_l725_725053

def diamond (a b : ℕ) : ℕ := 4 * a - 2 * b

theorem diamond_value : diamond 6 3 = 18 :=
by
  sorry

end diamond_value_l725_725053


namespace range_of_x_l725_725283

theorem range_of_x (x : ℝ) : (∃ y : ℝ, y = 2 / real.sqrt (x - 1)) ↔ x > 1 :=
by
  sorry

end range_of_x_l725_725283


namespace harry_started_with_79_l725_725520

-- Definitions using the conditions
def harry_initial_apples (x : ℕ) : Prop :=
  (x + 5 = 84)

-- Theorem statement proving the initial number of apples Harry started with
theorem harry_started_with_79 : ∃ x : ℕ, harry_initial_apples x ∧ x = 79 :=
by
  sorry

end harry_started_with_79_l725_725520


namespace T_seq_has_maximum_no_minimum_l725_725544

noncomputable def arithmetic_seq (n : ℕ) : ℤ := -9 + (n - 1) * 2

def T_seq (n : ℕ) : ℤ :=
  (List.range n).map (λ i => arithmetic_seq (i + 1)).prod

theorem T_seq_has_maximum_no_minimum :
  (∃ m : ℕ, ∀ n : ℕ, T_seq n ≤ T_seq m) ∧
  (¬ ∃ l : ℤ, ∀ n : ℕ, l ≤ T_seq n) :=
sorry

end T_seq_has_maximum_no_minimum_l725_725544


namespace lcm_36_100_is_900_l725_725451

def prime_factors_36 : ℕ → Prop := 
  λ n, n = 36 → (2^2 * 3^2)

def prime_factors_100 : ℕ → Prop := 
  λ n, n = 100 → (2^2 * 5^2)

def lcm_36_100 := lcm 36 100

theorem lcm_36_100_is_900 : lcm_36_100 = 900 :=
by {
  sorry,
}

end lcm_36_100_is_900_l725_725451


namespace range_of_a_l725_725857

-- Define the even function property
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define the monotonically increasing property on [0, ∞)
def mono_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) :
  even_function f →
  mono_increasing_on_nonneg f →
  (f (Real.log a / Real.log 2) + f (Real.log a / Real.log (1/2)) ≤ 2 * f 1) →
  (0 < a ∧ a ≤ 2) :=
by
  intros h_even h_mono h_ineq
  sorry

end range_of_a_l725_725857


namespace intersection_of_lines_l725_725707

theorem intersection_of_lines :
  ∃ (x y : ℚ), 3 * y = -2 * x + 6 ∧ 2 * y = -7 * x - 2 ∧ x = -18 / 17 ∧ y = 46 / 17 :=
by
  sorry

end intersection_of_lines_l725_725707


namespace not_p_and_pq_false_not_necessarily_p_or_q_l725_725537

theorem not_p_and_pq_false_not_necessarily_p_or_q (p q : Prop) 
  (h1 : ¬p) 
  (h2 : ¬(p ∧ q)) : ¬(p ∨ q) ∨ (p ∨ q) := by
  sorry

end not_p_and_pq_false_not_necessarily_p_or_q_l725_725537


namespace rational_sqrt_addition_l725_725492

theorem rational_sqrt_addition (a b : ℚ) (h : ∃ r : ℚ, r = (a.toReal.sqrt + b.toReal.sqrt)) :
  ∃ (x y : ℚ), a.toReal.sqrt = x ∧ b.toReal.sqrt = y :=
begin
  sorry
end

end rational_sqrt_addition_l725_725492


namespace problem_statement_l725_725547

noncomputable def geometric_sequence (a_n : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a_n (n + 1) = r * a_n n

noncomputable def root_equation (a b : ℝ) : Prop :=
  polynomial.root (X^2 - 10*X + 16) a ∧ polynomial.root (X^2 - 10*X + 16) b

-- The main theorem statement according to the given problem
theorem problem_statement (a_n : ℕ → ℝ) (h_geo_seq: geometric_sequence a_n) 
  (h_root_eq: root_equation (a_n 1) (a_n 5)):
  a_n 3 = 4 :=
sorry

end problem_statement_l725_725547


namespace man_l725_725747

-- Define all given conditions using Lean definitions
def speed_with_current_wind : ℝ := 22
def speed_of_current : ℝ := 5
def wind_resistance_factor : ℝ := 0.15
def current_increase_factor : ℝ := 0.10

-- Define the key quantities (man's speed in still water, effective speed in still water, new current speed against)
def speed_in_still_water : ℝ := speed_with_current_wind - speed_of_current
def effective_speed_in_still_water : ℝ := speed_in_still_water - (wind_resistance_factor * speed_in_still_water)
def new_speed_of_current_against : ℝ := speed_of_current + (current_increase_factor * speed_of_current)

-- Proof goal: Prove that the man's speed against the current is 8.95 km/hr considering all the conditions
theorem man's_speed_against_current_is_correct : 
  (effective_speed_in_still_water - new_speed_of_current_against) = 8.95 := 
by
  sorry

end man_l725_725747


namespace horizontal_asymptote_of_rational_function_l725_725532

theorem horizontal_asymptote_of_rational_function :
  (tendsto (λ x : ℝ, (15 * x^4 + 6 * x^3 + 5 * x^2 + 2 * x + 7) / (5 * x^4 + 3 * x^3 + 4 * x^2 + 2 * x + 1)) at_top (𝓝 3)) :=
begin
  sorry
end

end horizontal_asymptote_of_rational_function_l725_725532


namespace unique_integral_root_l725_725804

theorem unique_integral_root {x : ℤ} :
  x - 12 / (x - 3) = 5 - 12 / (x - 3) ↔ x = 5 :=
by
  sorry

end unique_integral_root_l725_725804


namespace circular_garden_radius_l725_725717

theorem circular_garden_radius :
  ∀ (r : ℝ), (2 * Real.pi * r = (1 / 3) * Real.pi * r^2) → (r = 6) := 
begin
  intros,
  sorry
end

end circular_garden_radius_l725_725717


namespace sin_half_angle_product_l725_725902

theorem sin_half_angle_product (A B C : ℝ) 
  (hC : C = 60) 
  (h_tan_sum : tan (A / 2) + tan (B / 2) = 1) 
  (h_angle_sum : A + B + C = 180) :
  sin (A / 2) * sin (B / 2) = (sqrt 3 - 1) / 2 :=
by
  sorry

end sin_half_angle_product_l725_725902


namespace total_savings_in_2_months_l725_725007

def students : ℕ := 30
def contribution_per_student_per_week : ℕ := 2
def weeks_in_month : ℕ := 4
def months : ℕ := 2

def total_contribution_per_week : ℕ := students * contribution_per_student_per_week
def total_weeks : ℕ := months * weeks_in_month
def total_savings : ℕ := total_contribution_per_week * total_weeks

theorem total_savings_in_2_months : total_savings = 480 := by
  -- Proof goes here
  sorry

end total_savings_in_2_months_l725_725007


namespace not_perfect_square_infinitely_many_l725_725942

theorem not_perfect_square_infinitely_many (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_gt : b > a) (h_prime : Prime (b - a)) :
  ∃ᶠ n in at_top, ¬ IsSquare ((a ^ n + a + 1) * (b ^ n + b + 1)) :=
sorry

end not_perfect_square_infinitely_many_l725_725942


namespace generating_function_stirling_numbers_l725_725085

noncomputable def stirling_ogf (k : ℕ) : ℚ[[t]] :=
  ∑ n in Finset.range k.succ, (S_2 n k) * t^n

theorem generating_function_stirling_numbers (k : ℕ) :
  stirling_ogf k = t^k / ((1 - t) * (1 - 2 * t) * ... * (1 - k * t)) :=
sorry

end generating_function_stirling_numbers_l725_725085


namespace find_d_l725_725281

def line_param (x y : ℝ) (v d : ℝ × ℝ) (t : ℝ) : Prop :=
  (x, y) = (v.fst + t * d.fst, v.snd + t * d.snd)

def distance_constraint (x y : ℝ) (t : ℝ) : Prop :=
  real.sqrt ((x - 5) ^ 2 + (y - 2) ^ 2) = 2 * t

theorem find_d (x y t : ℝ) (d : ℝ × ℝ) :
  (y = (4 * x - 7) / 3) →
  (line_param x y (5, 2) d t) →
  (distance_constraint x y t) →
  (x >= 5) →
  d = (6/5, 8/5) :=
by
  sorry

end find_d_l725_725281


namespace tomatoes_left_l725_725684

theorem tomatoes_left (initial_tomatoes : ℕ) (birds : ℕ) (fraction : ℕ) (E1 : initial_tomatoes = 21) 
  (E2 : birds = 2) (E3 : fraction = 3) : 
  initial_tomatoes - initial_tomatoes / fraction = 14 :=
by 
  sorry

end tomatoes_left_l725_725684


namespace no_integers_with_cube_sum_l725_725817

theorem no_integers_with_cube_sum (a b : ℤ) (h1 : a^3 + b^3 = 4099) (h2 : Prime 4099) : false :=
sorry

end no_integers_with_cube_sum_l725_725817


namespace books_fit_in_advertising_bag_l725_725047

-- Definitions based on conditions:
def advertising_bag_width : ℝ := 28
def art_album_width : ℝ := 22
def cookbook_width : ℝ := 25
def book_thickness : ℝ := 1.5

-- Circumference of the bag's mouth:
def bag_circumference := 2 * advertising_bag_width 

-- Lean statement for the proof that the books can fit into the bag:
theorem books_fit_in_advertising_bag :
  let arrangement_perimeter := 
        2 * art_album_width + 2 * book_thickness + 6 * (book_thickness * real.sqrt 2) in
  arrangement_perimeter <= bag_circumference :=
by
  -- Proof steps would go here.
  sorry

end books_fit_in_advertising_bag_l725_725047


namespace local_maximum_at_x2_implies_c6_l725_725900

theorem local_maximum_at_x2_implies_c6 
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = x * (x - c)^2)
  (h_local_max : local_maximum f 2) :
  c = 6 :=
sorry

def local_maximum (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y, abs (y - x) < ε → f y ≤ f x

end local_maximum_at_x2_implies_c6_l725_725900


namespace find_f_neg2_l725_725666

theorem find_f_neg2 (a b : ℝ) (f : ℝ → ℝ) (h₁ : ∀ x, f x = x^5 + a*x^3 + x^2 + b*x + 2) (h₂ : f 2 = 3) : f (-2) = 9 :=
by
  sorry

end find_f_neg2_l725_725666


namespace lcm_36_100_l725_725459

theorem lcm_36_100 : Nat.lcm 36 100 = 900 :=
by
  sorry

end lcm_36_100_l725_725459


namespace joes_current_weight_l725_725937

theorem joes_current_weight (W : ℕ) (R : ℕ) : 
  (W = 222 - 4 * R) →
  (W - 3 * R = 180) →
  W = 198 :=
by
  intros h1 h2
  -- Skip the proof for now
  sorry

end joes_current_weight_l725_725937


namespace part1_part2_l725_725474

variable (a b : ℝ)

-- Part (1)
theorem part1 (hA : a^2 - 2 * a * b + b^2 = A) (hB: a^2 + 2 * a * b + b^2 = B) (h : a ≠ b) :
  A + B > 0 := sorry

-- Part (2)
theorem part2 (hA : a^2 - 2 * a * b + b^2 = A) (hB: a^2 + 2 * a * b + b^2 = B) (h: a * b = 1) : 
  A - B = -4 := sorry

end part1_part2_l725_725474


namespace pyramid_layers_l725_725640

theorem pyramid_layers (n : ℕ) (h : ∀ k ≤ n, ∑ i in (finset.range k), 4 * (i - 1) = 145) : n = 9 :=
sorry

end pyramid_layers_l725_725640


namespace cycling_problem_l725_725335

theorem cycling_problem (x : ℝ) (h₀ : x > 0) :
  30 / x - 30 / (x + 3) = 2 / 3 :=
sorry

end cycling_problem_l725_725335


namespace billboard_area_is_correct_l725_725363

def is_rectangular_billboard_area_correct (P W: ℕ) (L: ℕ) (A: ℕ) : Prop :=
  P = 2 * L + 2 * W ∧ L = 26 / 2 ∧ A = L * W

theorem billboard_area_is_correct : 
  ∀ (P W : ℕ), 
    P = 42 → W = 8 → 
    ∃ (L A : ℕ), 
      is_rectangular_billboard_area_correct P W L A ∧ A = 104 := 
by
  intros P W hP hW
  use 13
  use 104
  split
  · constructor
    · exact hP
    · exact rfl
    · exact rfl
  · exact rfl

end billboard_area_is_correct_l725_725363


namespace max_lines_with_specific_angles_l725_725995

def intersecting_lines : ℕ := 6

theorem max_lines_with_specific_angles :
  ∀ (n : ℕ), (∀ (i j : ℕ), i ≠ j → (∃ θ : ℝ, θ = 30 ∨ θ = 60 ∨ θ = 90)) → n ≤ 6 :=
  sorry

end max_lines_with_specific_angles_l725_725995


namespace intersection_A_Z_l725_725535

def A : Set ℝ := {x | Abs.abs (x - 1) < 2}

theorem intersection_A_Z : A ∩ (Set.of {x : ℤ | true}) = {0, 1, 2} := by
  sorry

end intersection_A_Z_l725_725535


namespace investment_payoff_period_l725_725992

noncomputable theory

def initialInvestment (systemUnitCost : ℕ) (graphicsCardCost : ℕ) (numGraphicsCards : ℕ) : ℕ :=
  systemUnitCost + (numGraphicsCards * graphicsCardCost)

def dailyRevenue (ethPerCardPerDay : ℝ) (numGraphicsCards : ℕ) (ethToRubRate : ℝ) : ℝ :=
  (ethPerCardPerDay * numGraphicsCards) * ethToRubRate

def dailyEnergyCost (systemUnitConsumption : ℕ) (graphicsCardConsumption : ℕ) (numGraphicsCards : ℕ) (electricityCostPerKWh : ℝ) : ℝ :=
  let totalWattage := systemUnitConsumption + (graphicsCardConsumption * numGraphicsCards)
  let dailyKWh := (totalWattage / 1000.0) * 24
  dailyKWh * electricityCostPerKWh

def netDailyProfit (dailyRevenue : ℝ) (dailyEnergyCost : ℝ) : ℝ :=
  dailyRevenue - dailyEnergyCost

def paybackPeriod (initialInvestment : ℕ) (netDailyProfit : ℝ) : ℝ :=
  initialInvestment / netDailyProfit

theorem investment_payoff_period
    (systemUnitCost : ℕ := 9499)
    (graphicsCardCost : ℕ := 20990)
    (numGraphicsCards : ℕ := 2)
    (ethPerCardPerDay : ℝ := 0.00630)
    (ethToRubRate : ℝ := 27790.37)
    (systemUnitConsumption : ℕ := 120)
    (graphicsCardConsumption : ℕ := 185)
    (electricityCostPerKWh : ℝ := 5.38)
    : paybackPeriod (initialInvestment systemUnitCost graphicsCardCost numGraphicsCards)
                    (netDailyProfit (dailyRevenue ethPerCardPerDay numGraphicsCards ethToRubRate)
                                    (dailyEnergyCost systemUnitConsumption graphicsCardConsumption numGraphicsCards electricityCostPerKWh)) ≈ 179 := by
  sorry

end investment_payoff_period_l725_725992


namespace number_of_six_digit_palindromes_l725_725170

def is_six_digit_palindrome (n : ℕ) : Prop := 
  100000 ≤ n ∧ n ≤ 999999 ∧ (∀ a b c : ℕ, 
    n = 100000 * a + 10000 * b + 1000 * c + 100 * c + 10 * b + a → a ≠ 0)

theorem number_of_six_digit_palindromes : 
  ∃ (count : ℕ), (count = 900 ∧ 
  ∀ n : ℕ, is_six_digit_palindrome n → true) 
:= 
by 
  use 900 
  sorry

end number_of_six_digit_palindromes_l725_725170


namespace part1_part2_l725_725927

variable (a b c : ℝ)
variable (A B C : ℝ)
variable (q p : ℝ × ℝ)

-- Conditions
def condition_1 : q = (2 * a, 1) := sorry
def condition_2 : p = (2 * b - c, Real.cos C) := sorry
def condition_3 : p = q.smul k := sorry  -- For some k since p || q
def condition_4 : TriangleSideAngle a b c A B C := sorry  -- Custom condition to establish sides-opposite-angles relation in a triangle

-- Questions to prove
theorem part1 : ∀ (a b c A B C : ℝ), (q = (2 * a, 1)) → (p = (2 * b - c, Real.cos C)) → (p = q.smul k) → 
    (a, b, c, A, B, C : TriangleSideAngle a b c A B C) → Real.sin A = Real.sqrt 3 / 2 :=
sorry

theorem part2 : ∀ (a b c A B C : ℝ), (q = (2 * a, 1)) → (p = (2 * b - c, Real.cos C)) → (p = q.smul k) → 
    (a, b, c, A, B, C : TriangleSideAngle a b c A B C) → 
    ∃ range, range = set.Ioo (-1 : ℝ) (Real.sqrt 2) :=
sorry

end part1_part2_l725_725927


namespace at_least_six_heads_in_10_flips_is_129_over_1024_l725_725348

def fair_coin_flip (n : ℕ) (prob_heads prob_tails : ℚ) : Prop :=
  (prob_heads = 1/2 ∧ prob_tails = 1/2)

noncomputable def at_least_six_consecutive_heads_probability (n : ℕ) : ℚ :=
  if n = 10 then 129 / 1024 else 0  -- this is specific to 10 flips and should be defined based on actual calculation for different n
  
theorem at_least_six_heads_in_10_flips_is_129_over_1024 :
  fair_coin_flip 10 (1/2) (1/2) →
  at_least_six_consecutive_heads_probability 10 = 129 / 1024 :=
by
  intros
  sorry

end at_least_six_heads_in_10_flips_is_129_over_1024_l725_725348


namespace find_y_l725_725997

variable (x y z : ℕ)

-- Conditions
def condition1 : Prop := 100 + 200 + 300 + x = 1000
def condition2 : Prop := 300 + z + 100 + x + y = 1000

-- Theorem to be proven
theorem find_y (h1 : condition1 x) (h2 : condition2 x y z) : z + y = 200 :=
sorry

end find_y_l725_725997


namespace susan_books_l725_725598

theorem susan_books (S : ℕ) (h1 : S + 4 * S = 3000) : S = 600 :=
by 
  sorry

end susan_books_l725_725598


namespace total_profit_is_correct_l725_725266

-- Define the conditions
variables (suresh_investment ramesh_investment ramesh_share total_profit : ℝ)

-- Assign the given values to the conditions
def suresh_investment : ℝ := 24000
def ramesh_investment : ℝ := 40000
def ramesh_share : ℝ := 11875

-- Statement to prove: Given the conditions, prove total profit is Rs. 19,000
theorem total_profit_is_correct (h1 : suresh_investment = 24000) 
                                (h2 : ramesh_investment = 40000)
                                (h3 : ramesh_share = 11875) : 
                                total_profit = 19000 :=
begin
  -- Here would be the proof
  sorry
end

end total_profit_is_correct_l725_725266


namespace find_m_find_max_value_l725_725869

noncomputable def function_f (x : ℝ) : ℝ := |x - 1|

lemma solution_set_inequality (x : ℝ) (m : ℝ) (h1 : m > 0) (h2 : -7 <= x) (h3 : x <= -1) :
  function_f (x + 5) <= 3 * m :=
by {
  sorry
}

theorem find_m (m : ℝ) (h : solution_set_inequality (-7) m (by linarith) (by linarith) (by linarith) ∧ 
                        solution_set_inequality (-1) m (by linarith) (by linarith) (by linarith)) :
  m = 1 :=
by {
  sorry
}

lemma maximum_value (a b : ℝ) (h1: a > 0) (h2: b > 0) (h3: 2*a^2 + b^2 = 3) :
  2*a*sqrt(1+b^2) <= 2*sqrt(2) :=
by {
  sorry
}

theorem find_max_value {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a^2 + b^2 = 3) :
  2*a*sqrt(1+b^2) = 2*sqrt(2) ↔ a = 1 ∧ b = 1 :=
by {
  sorry
}

end find_m_find_max_value_l725_725869


namespace part1_real_roots_part2_specific_roots_l725_725878

-- Part 1: Real roots condition
theorem part1_real_roots (m : ℝ) (h : ∃ x : ℝ, x^2 + (2 * m - 1) * x + m^2 = 0) : m ≤ 1/4 :=
by sorry

-- Part 2: Specific roots condition
theorem part2_specific_roots (m : ℝ) (x1 x2 : ℝ) 
  (h1 : x1^2 + (2 * m - 1) * x1 + m^2 = 0) 
  (h2 : x2^2 + (2 * m - 1) * x2 + m^2 = 0) 
  (h3 : x1 * x2 + x1 + x2 = 4) : m = -1 :=
by sorry

end part1_real_roots_part2_specific_roots_l725_725878


namespace correct_statements_l725_725502

def statement1 := "The radius of a sphere is the line segment from any point on the sphere to the center of the sphere."
def statement2 := "The diameter of a sphere is the line segment connecting any two points on the sphere."
def statement3 := "Cutting a sphere with a plane results in a circle."
def statement4 := "The sphere is often represented by the letter denoting its center."

theorem correct_statements :
  (statement1 = "The radius of a sphere is the line segment from any point on the sphere to the center of the sphere.") ∧
  (statement2 ≠ "The diameter of a sphere is the line segment connecting any two points on the sphere.") ∧
  (statement3 = "Cutting a sphere with a plane results in a circle.") ∧
  (statement4 = "The sphere is often represented by the letter denoting its center.") :=
begin
  sorry
end

end correct_statements_l725_725502


namespace conjugate_of_expr_l725_725126

-- Define the given complex number z
def z : ℂ := 1 - Complex.i

-- Define the expression to simplify
def expr : ℂ := (2 / z) - z^2

-- State the theorem to be proved
theorem conjugate_of_expr : Complex.conj expr = 1 - 3 * Complex.i := by
  sorry

end conjugate_of_expr_l725_725126


namespace fraction_of_triangle_area_l725_725630

open Real

def point := (ℝ × ℝ)

def area_of_triangle (A B C : point) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (abs ((x1 * (y2 - y3)) + (x2 * (y3 - y1)) + (x3 * (y1 - y2))) / 2)

def A : point := (2, 0)
def B : point := (8, 12)
def C : point := (14, 0)

def X : point := (6, 0)
def Y : point := (8, 4)
def Z : point := (10, 0)

theorem fraction_of_triangle_area :
  (area_of_triangle X Y Z) / (area_of_triangle A B C) = 1 / 9 :=
by
  sorry

end fraction_of_triangle_area_l725_725630


namespace projection_onto_same_vector_l725_725703

noncomputable def vec1 : ℝ × ℝ := (5, 2)
noncomputable def vec2 : ℝ × ℝ := (-2, 4)
noncomputable def p : ℝ × ℝ := (18 / 53, 83 / 53)

theorem projection_onto_same_vector :
  ∃ v : ℝ × ℝ, true -> (let proj := v in 
    proj = p) :=
begin
  sorry
end

end projection_onto_same_vector_l725_725703


namespace reflect_circle_center_l725_725655

theorem reflect_circle_center :
  let original_center : ℝ × ℝ := (4, -3)
  let reflected_center := (3, -4)
  let reflection_line (p : ℝ × ℝ) := (-p.2, -p.1)
  reflection_line original_center = reflected_center := 
by
  -- Definitions for conditions
  let original_center : ℝ × ℝ := (4, -3)
  let reflected_center := (3, -4)
  let reflection_line (p : ℝ × ℝ) := (-p.2, -p.1)

  -- Check that reflecting the original center gives the reflected center
  have H : reflection_line original_center = reflected_center := by
    simp [reflection_line, original_center, reflected_center]
  exact H

end reflect_circle_center_l725_725655


namespace chord_lengths_equal_l725_725672

theorem chord_lengths_equal (m n ω : ℝ) (h_intervals: ∀ x ∈ set.Icc (0:ℝ) (4 * π / ω), -1 ≤ sin (ω * x / 2) ∧ sin (ω * x / 2) ≤ 1)
(h_chords: ∀ x1 x2 : ℝ, (0 ≤ x1 ∧ x1 ≤ 4 * π / ω) ∧ (0 ≤ x2 ∧ x2 ≤ 4 * π / ω) ∧ (m * sin (ω * x1 / 2) + n = 5) ∧ (m * sin (ω * x2 / 2) + n = -1) → (x2 - x1) = (4 * π / ω - (x1 + x2))) :
  m > 3 ∧ n = 2 :=
sorry

end chord_lengths_equal_l725_725672


namespace part1_part2_l725_725183

-- Define the required variables and assumptions
variables {α β γ : Type} 

-- Define $\triangle ABC$ with respective angles and sides
variables (A B C : α) (a b c : β)

-- Define the conditions under Proposition 15
-- Proposition 15 specifics don't need to be defined, as we assume it is applicable and true.
axiom prop_15 : Prop

-- First part of the problem: Prove that aA + bB + cC \geq aB + bC + cA
theorem part1 (Triangle_ABC : prop_15) : 
  aA + bB + cC ≥ aB + bC + cA :=
sorry

-- Second part of the problem: Prove the inequality of mean values with trigonometric functions
theorem part2 (Triangle_ABC : prop_15) :
  (Aa + Bb + Cc) / (a + b + c) ≥ π / 3 ∧ π / 3 ≥ (A * cos A + B * cos B + C * cos C) / (cos A + cos B + cos C) :=
sorry

end part1_part2_l725_725183


namespace n_must_be_square_no_coloring_for_16_l725_725975

theorem n_must_be_square (n : ℕ) (color : Fin n → Bool) 
  (equal_length_brown_and_green_diags : ∀ l, (count (λ x, is_brown_diagonal x l) (all_diags n) = count (λ x, is_green_diagonal x l) (all_diags n)))
  (equal_brown_and_green_sides : (count_brown_sides n color = count_green_sides n color)) :
  ∃ k, n = k ^ 2 := sorry

-- Additional definition for the specific n = 16 case
theorem no_coloring_for_16 :
  ¬ ∃ color : Fin 16 → Bool, 
    (equal_length_brown_and_green_diags 16 color ∧ equal_brown_and_green_sides 16 color) := sorry

-- Auxiliary definitions and lemmas
def count {α : Type} (p : α → Prop) [DecidablePred p] (l : List α) : ℕ :=
  l.countp p

def all_diags (n : ℕ) : List (Fin n × Fin n) :=
  ⟨[x for x in Fin n, y for y in Fin n, x ≠ y]⟩  -- Pseudo-code for list of diagonals

def is_brown_diagonal {n : ℕ} (x : Fin n × Fin n) (l : ℕ) : Prop := sorry  -- Pseudo-code for brown diagonal

def is_green_diagonal {n : ℕ} (x : Fin n × Fin n) (l : ℕ) : Prop := sorry  -- Pseudo-code for green diagonal

def count_brown_sides (n : ℕ) (color : Fin n → Bool) : ℕ := sorry  -- Pseudo-code for count brown sides

def count_green_sides (n : ℕ) (color : Fin n → Bool) : ℕ := sorry  -- Pseudo-code for count green sides

end n_must_be_square_no_coloring_for_16_l725_725975


namespace area_bounded_by_graphs_eq_4_l725_725416

theorem area_bounded_by_graphs_eq_4 :
  let r₁ (θ : ℝ) := 2 / (cos θ)
  let r₂ (θ : ℝ) := 2 / (sin θ)
  ∀ (θ₁ θ₂ : ℝ) (x ℝ: ℝ), 0 ≤ θ₁ ∧ θ₁ ≤ π/2 ∧ 0 ≤ θ₂ ∧ θ₂ ≤ π/2 ∧
  x = r₁ θ₁ ∧ y = r₂ θ₂ →
  area (bounded_region r₁ r₂ 0 0) = 4 := by
  sorry

end area_bounded_by_graphs_eq_4_l725_725416


namespace number_of_six_digit_palindromes_l725_725167

def is_six_digit_palindrome (n : ℕ) : Prop := 
  100000 ≤ n ∧ n ≤ 999999 ∧ (∀ a b c : ℕ, 
    n = 100000 * a + 10000 * b + 1000 * c + 100 * c + 10 * b + a → a ≠ 0)

theorem number_of_six_digit_palindromes : 
  ∃ (count : ℕ), (count = 900 ∧ 
  ∀ n : ℕ, is_six_digit_palindrome n → true) 
:= 
by 
  use 900 
  sorry

end number_of_six_digit_palindromes_l725_725167


namespace mean_value_function_range_of_m_l725_725397

def is_mean_value_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x1 x2, a < x1 ∧ x1 < x2 ∧ x2 < b ∧
    (f'' x1 = (f b - f a) / (b - a)) ∧
    (f'' x2 = (f b - f a) / (b - a))

def f (x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * x^2 + Real.pi

theorem mean_value_function_range_of_m (m : ℝ) :
  is_mean_value_function f 0 m → m ∈ Set.Ioo (3 / 4 : ℝ) (3 / 2 : ℝ) :=
sorry

end mean_value_function_range_of_m_l725_725397


namespace polynomial_identity_solution_l725_725410

noncomputable def is_solution (P : ℝ[X]) : Prop :=
  P ≠ 0 ∧ (∀ x : ℝ, P.eval x * P.eval (2 * x^2) = P.eval (2 * x^3 + x))

noncomputable def final_result (P : ℝ[X]) : Prop :=
  ∃ k : ℕ, k > 0 ∧ P = (Polynomial.X^2 + 1)^k

theorem polynomial_identity_solution (P : ℝ[X]) (hP : is_solution P) : final_result P :=
sorry

end polynomial_identity_solution_l725_725410


namespace problem_monotonic_intervals_extreme_points_l725_725121

noncomputable def f (a x : ℝ) := ln x - 2 * a * x + 2 * a

theorem problem (a : ℝ) (a_ineq : a ∈ Icc 0 (1 / 4)) :
  ∀ x₁ x₂ ∈ Ioo 0 2, x₁ ≠ x₂ →
  abs (f a x₁ - f a x₂) < 2 * a * abs (1 / x₁ - 1 / x₂) ↔ a = 1 / 4 := 
sorry

theorem monotonic_intervals (a : ℝ) :
  (0 < a → 
   (∀ x ∈ Ioo 0 (1 / (2 * a)), f' a x > 0) ∧ 
   (∀ x ∈ Ioo (1 / (2 * a)) ∞, f' a x < 0)) ∧ 
  (a ≤ 0 → ∀ x > 0, f' a x > 0) := 
sorry

theorem extreme_points (a : ℝ) (ha : a ≤ 1 / 4) :
  ∀ x ∈ Ioo 0 2, (f (1, x) < f (1, 1) ↔ 0 < x < 1) ∧
  ∀ x ∈ Ioo 0 2, (f (1, x) > f (1, 1) ↔ 1 < x < 2) :=
sorry

end problem_monotonic_intervals_extreme_points_l725_725121


namespace total_tax_percent_l725_725726

-- Given conditions
variables (total_spent : ℝ)
variable (clothing_spent : ℝ := 0.5 * total_spent)
variable (food_spent : ℝ := 0.2 * total_spent)
variable (other_spent : ℝ := 0.3 * total_spent)
variable (clothing_tax_rate : ℝ := 0.04)
variable (food_tax_rate : ℝ := 0)
variable (other_tax_rate : ℝ := 0.1)

-- Proving the total tax percent paid
theorem total_tax_percent (total_spent : ℝ) :
  let clothing_tax := clothing_spent * clothing_tax_rate,
      food_tax := food_spent * food_tax_rate,
      other_tax := other_spent * other_tax_rate,
      total_tax := clothing_tax + food_tax + other_tax,
      total_tax_percent := (total_tax / total_spent) * 100
  in total_tax_percent = 5 := by {
  sorry
}

end total_tax_percent_l725_725726


namespace lcm_of_36_and_100_l725_725445

theorem lcm_of_36_and_100 : Nat.lcm 36 100 = 900 :=
by
  -- The proof is omitted
  sorry

end lcm_of_36_and_100_l725_725445


namespace clock_angle_at_9am_l725_725779

theorem clock_angle_at_9am : 
  let minute_on_12 := true,
      hour_on_9 := true,
      degrees_per_hour := 30 in
  (minute_on_12 ∧ hour_on_9 ∧ degrees_per_hour = 30) → 
  ∃ angle, angle = 90 :=
by
  sorry

end clock_angle_at_9am_l725_725779


namespace product_of_f_of_right_triangles_l725_725951

noncomputable def S := {p : ℕ × ℕ | (p.1 ∈ {0, 1, 2, 3}) ∧ (p.2 ∈ {0, 1, 2, 3, 4}) ∧ p ≠ (0, 4)}

def is_right_triangle (A B C : ℕ × ℕ) : Prop :=
  (A.1 = B.1 ∧ A.2 = C.2) ∨ (A.2 = B.2 ∧ A.1 = C.1)

def f (A B C : ℕ × ℕ) : ℚ :=
  if A.1 = B.1
  then (B.2 - A.2) / (C.1 - A.1)
  else (B.1 - A.1) / (C.2 - A.2)

noncomputable def T := {t : (ℕ × ℕ) × (ℕ × ℕ) × (ℕ × ℕ) | t.1 ∈ S ∧ t.2.1 ∈ S ∧ t.2.2 ∈ S ∧ is_right_triangle t.1 t.2.1 t.2.2}

theorem product_of_f_of_right_triangles :
  ∏ t in T, f t.1 t.2.1 t.2.2 = 9 / 2 :=
by
  sorry

end product_of_f_of_right_triangles_l725_725951


namespace intersection_point_l725_725078

theorem intersection_point :
  ∃ (x y : ℝ), (2 * x + 3 * y + 8 = 0) ∧ (x - y - 1 = 0) ∧ (x = -1) ∧ (y = -2) := 
by
  sorry

end intersection_point_l725_725078


namespace min_value_of_f_l725_725815

noncomputable def f (x : ℝ) : ℝ := cos x ^ 2 + sqrt 3 * sin x * cos x

theorem min_value_of_f : ∀ x : ℝ, f x ≥ -1/2 ∧ (∃ x, f x = -1/2) :=
by
  sorry

end min_value_of_f_l725_725815


namespace jane_bought_15_ice_cream_cones_l725_725935

def num_ice_cream_cones (x : ℕ) (pudding_cost : ℕ) (ice_cream_cost : ℕ) (price_difference : ℕ) : Prop :=
  5 * x = pudding_cost + price_difference

theorem jane_bought_15_ice_cream_cones :
  ∃ (x : ℕ), x = 15 ∧
  let pudding_cost := 5 * 2 in
  let ice_cream_cost := 5 in
  let price_difference := 65 in
  num_ice_cream_cones x pudding_cost ice_cream_cost price_difference :=
begin
  use 15,
  split,
  { refl },
  { dsimp [pudding_cost, ice_cream_cost, price_difference, num_ice_cream_cones],
    linarith }
end

end jane_bought_15_ice_cream_cones_l725_725935


namespace arcsin_arccos_eq_pi_over_2_l725_725983

theorem arcsin_arccos_eq_pi_over_2 (x : ℝ) (h₁ : arcsin x + arccos x = π / 2) (h₂ : arcsin x + arccos (1 - x) = π / 2) :
  x = sqrt 2 / 2 :=
sorry

end arcsin_arccos_eq_pi_over_2_l725_725983


namespace al_mass_percentage_not_10_11_in_Al2O3_l725_725216

-- Define constants for molar masses
def molar_mass_Al : ℝ := 26.98
def molar_mass_O : ℝ := 16.00

-- Define a function to compute the mass percentage of Al in a compound
def mass_percentage_Al_in_compound (x y : ℤ) : ℝ :=
  let molar_mass_compound := (x * molar_mass_Al) + (y * molar_mass_O)
  (x * molar_mass_Al / molar_mass_compound) * 100

-- Prove that Al₂O₃ does not satisfy the required mass percentage of 10.11%
theorem al_mass_percentage_not_10_11_in_Al2O3 :
    ¬ mass_percentage_Al_in_compound 2 3 = 10.11 :=
by 
  sorry

end al_mass_percentage_not_10_11_in_Al2O3_l725_725216


namespace trig_expression_simplification_l725_725819

theorem trig_expression_simplification (α : Real) :
  Real.cos (3/2 * Real.pi + 4 * α)
  + Real.sin (3 * Real.pi - 8 * α)
  - Real.sin (4 * Real.pi - 12 * α)
  = 4 * Real.cos (2 * α) * Real.cos (4 * α) * Real.sin (6 * α) :=
sorry

end trig_expression_simplification_l725_725819


namespace tomatoes_left_l725_725691

theorem tomatoes_left (initial_tomatoes : ℕ) (birds : ℕ) (fraction_eaten : ℚ) :
  initial_tomatoes = 21 ∧ birds = 2 ∧ fraction_eaten = 1/3 ->
  initial_tomatoes - initial_tomatoes * fraction_eaten = 14 :=
by
  intros h
  cases h with h1 h_rest
  cases h_rest with h2 h3
  rw [h1, h2, h3]
  norm_num
  rw [Nat.cast_sub 21 7 _, Nat.cast_mul, Nat.cast_div]; norm_num -- Converting to rational arithmetic and proving directly
  exact le_of_lt_nat (div_lt_self (zero_lt_nat 21) (zero_lt_nat 3))

end tomatoes_left_l725_725691


namespace minimal_polynomial_correct_l725_725816

noncomputable def minimal_polynomial : Polynomial ℚ :=
  (Polynomial.X^2 - 4 * Polynomial.X + 1) * (Polynomial.X^2 - 6 * Polynomial.X + 2)

theorem minimal_polynomial_correct :
  Polynomial.X^4 - 10 * Polynomial.X^3 + 29 * Polynomial.X^2 - 26 * Polynomial.X + 2 = minimal_polynomial :=
  sorry

end minimal_polynomial_correct_l725_725816


namespace prime_sum_divisibility_l725_725590

-- Conditions
def is_prime (p : ℕ) : Prop := Nat.Prime p
def sum_fractions (p r s : ℕ) : Prop :=
  1 + (1 / 2) + (1 / 3) + ... + (1 / (p-1)) + (1 / p) = r / (p * s)

-- Main theorem
theorem prime_sum_divisibility (p r s : ℕ)
  (h_prime : is_prime p)
  (h_p_gt_3 : p > 3)
  (h_sum_fractions : sum_fractions p r s)
  (h_coprime : Nat.gcd r s = 1) :
  p^3 ∣ (r - s) := by
  sorry

end prime_sum_divisibility_l725_725590


namespace g_eval_l725_725597

-- Define the function g
def g (a : ℚ) (b : ℚ) (c : ℚ) : ℚ := (2 * a + b) / (c - a)

-- Theorem to prove g(2, 4, -1) = -8 / 3
theorem g_eval :
  g 2 4 (-1) = -8 / 3 := 
by
  sorry

end g_eval_l725_725597


namespace cracked_to_broken_eggs_ratio_l725_725601

theorem cracked_to_broken_eggs_ratio (total_eggs : ℕ) (broken_eggs : ℕ) (P C : ℕ)
  (h1 : total_eggs = 24)
  (h2 : broken_eggs = 3)
  (h3 : P - C = 9)
  (h4 : P + C = 21) :
  (C : ℚ) / (broken_eggs : ℚ) = 2 :=
by
  sorry

end cracked_to_broken_eggs_ratio_l725_725601


namespace final_share_approx_equal_l725_725681

noncomputable def total_bill : ℝ := 211.0
noncomputable def number_of_people : ℝ := 6.0
noncomputable def tip_percentage : ℝ := 0.15
noncomputable def tip_amount : ℝ := tip_percentage * total_bill
noncomputable def total_amount : ℝ := total_bill + tip_amount
noncomputable def each_person_share : ℝ := total_amount / number_of_people

theorem final_share_approx_equal :
  abs (each_person_share - 40.44) < 0.01 :=
by
  sorry

end final_share_approx_equal_l725_725681


namespace quadratic_functions_intercept_difference_l725_725830

theorem quadratic_functions_intercept_difference :
  ∃ (m n p : ℕ), p ∉ (λ p, ∃ k : ℕ, p = k ^ 2) ∧
  (∃ x1 x2 x3 x4 : ℝ, 
    (∀ x : ℝ, g x = -f (120 - x)) ∧
    (∀ x : ℝ, is_vertex (f x) = is_vertex (g x)) ∧
    (∃ (x1 x2 x3 x4 : ℝ), 
      x1 < x2 ∧ x2 < x3 ∧ x3 < x4 ∧ 
      ∀ x : ℝ, is_intercept (f x) <-> x ∈ {x1, x2} ∧ 
      (is_intercept (g x) <-> x ∈ {x3, x4}) ∧
      x3 - x2 = 180) ∧ 
    (x4 - x1 = m + n * real.sqrt p)) :=
begin
  -- sorry to skip the proof
  sorry
end

end quadratic_functions_intercept_difference_l725_725830


namespace trigonometric_identities_alpha_trigonometric_identities_theta_l725_725729

-- Statement for proving trigonometric identities related to angle α
theorem trigonometric_identities_alpha (α : ℝ) (x : ℝ) (hx : x = -√3) (h1 : 90 < α) (h2 : α < 180) 
(h3 : cos α = (√2/4) * x) : 
  sin α = √10 / 4 ∧ tan α = -√15 / 3 :=
sorry

-- Statement for proving trigonometric identities related to angle θ
theorem trigonometric_identities_theta (θ : ℝ) (x : ℝ) 
(hx : x = 1 ∨ x = -1) (h1 : tan θ = -x) : 
  (x = 1 → sin θ = -√2 / 2 ∧ cos θ = √2 / 2) ∧
  (x = -1 → sin θ = -√2 / 2 ∧ cos θ = -√2 / 2) :=
sorry

end trigonometric_identities_alpha_trigonometric_identities_theta_l725_725729


namespace original_price_of_second_pair_l725_725934

variable (P : ℝ) -- original price of the second pair of shoes
variable (discounted_price : ℝ := P / 2)
variable (total_before_discount : ℝ := 40 + discounted_price)
variable (final_payment : ℝ := (3 / 4) * total_before_discount)
variable (payment : ℝ := 60)

theorem original_price_of_second_pair (h : final_payment = payment) : P = 80 :=
by
  -- Skipping the proof with sorry.
  sorry

end original_price_of_second_pair_l725_725934


namespace even_function_a_value_l725_725136

theorem even_function_a_value (a : ℝ) :
  (∀ x : ℝ, (a * (-x)^2 + (2 * a + 1) * (-x) - 1) = (a * x^2 + (2 * a + 1) * x - 1)) →
  a = - 1 / 2 :=
by sorry

end even_function_a_value_l725_725136


namespace area_of_bounded_region_l725_725436

theorem area_of_bounded_region : 
  let x1 := 2
  let y1 := 2
  let x0 := 0
  let y0 := 0
  in
  let area := (x1 - x0) * (y1 - y0)
  in
  area = 4 :=
by 
  -- definitions for x1, y1, x0, y0
  let x1 := 2
  let y1 := 2
  let x0 := 0
  let y0 := 0
  
  -- let area be the area of the square bounded by these lines
  let area := (x1 - x0) * (y1 - y0)
  
  -- assertion
  have h : area = (2 - 0) * (2 - 0), from rfl,
  
  -- proving the final statement
  show area = 4, by 
    rw [h],
    exact rfl

-- skipped proof step
sorry

end area_of_bounded_region_l725_725436


namespace probability_one_fork_two_spoons_one_knife_l725_725648

-- Define the drawer contents 
def num_forks : ℕ := 8
def num_spoons : ℕ := 5
def num_knives : ℕ := 10
def total_silverware : ℕ := num_forks + num_spoons + num_knives
def pieces_removed : ℕ := 4

-- Define the specific counts we are looking for
def desired_forks : ℕ := 1
def desired_spoons : ℕ := 2
def desired_knives : ℕ := 1

-- Define the total probability calculation
noncomputable def prob_fork_spoon_knife_exactly : ℚ :=
  (nat.choose num_forks desired_forks * nat.choose num_spoons desired_spoons *
   nat.choose num_knives desired_knives) /
  nat.choose total_silverware pieces_removed

-- The statement to prove
theorem probability_one_fork_two_spoons_one_knife :
  prob_fork_spoon_knife_exactly = 800 / 8855 :=
sorry

end probability_one_fork_two_spoons_one_knife_l725_725648


namespace area_of_bounded_region_l725_725424

theorem area_of_bounded_region :
  let region := {p : ℝ × ℝ | (p.1 = 0 ∧ p.2 ≥ 0) ∨ (p.2 = 0 ∧ p.1 ≥ 0) ∨ (p.1 = 2) ∨ (p.2 = 2)}
  ∃ (s : set (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ s ↔ (y = x))
    ∧ is_square s 
    ∧ area s = 4 := 
sorry

end area_of_bounded_region_l725_725424


namespace multiple_contains_all_digits_l725_725251

theorem multiple_contains_all_digits (n : ℕ) : ∃ M : ℕ, (M % n = 0) ∧ ∀ d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, ∃ k, (M / 10^k) % 10 = d := 
sorry

end multiple_contains_all_digits_l725_725251


namespace chemistry_club_student_count_l725_725807

theorem chemistry_club_student_count (x : ℕ) (h1 : x % 3 = 0)
  (h2 : x % 4 = 0) (h3 : x % 6 = 0)
  (h4 : (x / 3) = (x / 4) + 3) :
  (x / 6) = 6 :=
by {
  -- Proof goes here
  sorry
}

end chemistry_club_student_count_l725_725807


namespace ellipse_a_plus_k_l725_725035

/-- An ellipse has its foci at (2, 2) and (2, 6). Given that it passes through the point (-3, 4),
its equation is of the form (x-h)²/a² + (y-k)²/b² = 1 where a, b, h, k are constants, and a and b 
are positive. Prove that a + k = 4 + sqrt(29). -/
theorem ellipse_a_plus_k :
  ∃ (a b h k : ℝ), a > 0 ∧ b > 0 ∧
    ( ((2 - 2)^2 + (2 - 6)^2).sqrt * 2 = a * 2.sqrt + ( (2 - 2)^2 + (-3 - 2)^2).sqrt + ( (4 - 2)^2 + (-3 - 2)^2).sqrt) ∧
    (h = 2 ∧ k = 4 ∧ 4 = (a^2) - (b^2) ∧ b = 5) ∧
    (h = 2 ∧ k = 4 ∧ a + k = 4 + sqrt(29)) :=
sorry

end ellipse_a_plus_k_l725_725035


namespace trajectory_of_point_P_equation_of_line_m_l725_725677

def point_on_ellipse_trajectory (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in (sqrt ((x - 3 * sqrt 3) ^ 2 + y ^ 2)) / abs (x - 4 * sqrt 3) = sqrt 3 / 2

def equation_of_trajectory (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in x^2 / 36 + y^2 / 9 = 1

theorem trajectory_of_point_P (P : ℝ × ℝ) (h : point_on_ellipse_trajectory P) : equation_of_trajectory P :=
sorry    -- The proof would be added here

def midpoint_of_BC (B C : ℝ × ℝ) (M : ℝ × ℝ) : Prop :=
  let (x1, y1) := B in
  let (x2, y2) := C in
  let (xm, ym ) := M in
  xm = (x1 + x2) / 2 ∧ ym = (y1 + y2) / 2

def equation_of_ellipse (x y : ℝ) : Prop :=
  x^2 / 36 + y^2 / 9 = 1

def line (k : ℝ) (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in y = k * (x - 4) + 2

def line_m (k : ℝ) (M : ℝ × ℝ) : Prop :=
  let (x, y) := M in y - 2 = k * (x - 4)

theorem equation_of_line_m (k : ℝ) (B C M : ℝ × ℝ) (eq_line_m : line_m k M) 
  (intersect_ellipse : ∀ P, line k P → equation_of_ellipse P.1 P.2) 
  (midpoint : midpoint_of_BC B C M) : line k M :=
sorry    -- The proof would be added here

end trajectory_of_point_P_equation_of_line_m_l725_725677


namespace area_of_region_bounded_by_lines_l725_725429

theorem area_of_region_bounded_by_lines : 
  let x1 := λ θ : ℝ, 2 in
  let y1 := λ θ : ℝ, 2 in
  x1 (0:ℝ) * y1 (0:ℝ) = 4 :=
by {
  sorry,
}

end area_of_region_bounded_by_lines_l725_725429


namespace complex_sum_l725_725580

-- Definitions based on conditions
def A : ℂ := 3 - 4 * complex.I
def M : ℂ := -3 + 2 * complex.I
def S : ℂ := 2 * complex.I
def P : ℂ := -1

-- The goal is to prove the equality
theorem complex_sum :
  A - M + S + P = 5 - 4 * complex.I :=
by
  sorry

end complex_sum_l725_725580


namespace tangent_circumcircle_of_AKP_l725_725245

open Real

noncomputable def right_triangle (A B C : Point) : Prop :=
  ∃ O : Point, ∠OAB = 90 ∧ ∠OBC = 90 ∧ ∠OCA = 90

noncomputable def midpoint (X Y M : Point) : Prop :=
  dist M X = dist M Y

noncomputable def divide_ratio (X Y Z : Point) (r : ℝ) : Prop :=
  dist X Y = r * dist X Z

noncomputable def intersection (L1 L2 : Line) (P : Point) : Prop :=
  on_line P L1 ∧ on_line P L2

noncomputable def is_tangent (L : Line) (C : Circle) (P : Point) : Prop :=
  on_circle P C ∧ is_perpendicular L (tangent_line_at P C)

theorem tangent_circumcircle_of_AKP
  (A B C K M P : Point)
  (circumcircle_AKP : Circle)
  (line_KM : Line) :
  right_triangle A B C →
  midpoint A B K →
  divide_ratio B M C (2 / 1) →
  intersection (line_through A M) (line_through C K) P →
  on_circle K circumcircle_AKP →
  is_tangent line_KM circumcircle_AKP K :=
by
  sorry

end tangent_circumcircle_of_AKP_l725_725245


namespace magnified_image_diameter_l725_725309

variable (magnification_factor actual_diameter : ℝ)

def magnified_diameter (magnification_factor actual_diameter : ℝ) : ℝ :=
  magnification_factor * actual_diameter

theorem magnified_image_diameter 
  (h1 : magnification_factor = 1000) 
  (h2 : actual_diameter = 0.002) : 
  magnified_diameter magnification_factor actual_diameter = 2 := 
by 
  rw [magnified_diameter, h1, h2]
  norm_num
  exact sorry

end magnified_image_diameter_l725_725309


namespace range_of_y₀_l725_725139

variables {x y : ℝ}

def is_on_hyperbola (x₀ y₀ : ℝ) : Prop :=
  x₀^2 / 2 - y₀^2 = 1

def dot_product_non_positive (x₀ y₀ : ℝ) : Prop :=
  let F₁ := (-real.sqrt 3, 0) in
  let F₂ := (real.sqrt 3, 0) in
  ((- real.sqrt 3 - x₀) * (real.sqrt 3 - x₀) + (- y₀) * (- y₀)) ≤ 0

theorem range_of_y₀ (x₀ y₀ : ℝ) (h₁ : is_on_hyperbola x₀ y₀) (h₂ : dot_product_non_positive x₀ y₀) : 
  - real.sqrt 3 / 3 ≤ y₀ ∧ y₀ ≤ real.sqrt 3 / 3 := by
  sorry

end range_of_y₀_l725_725139


namespace number_of_six_digit_palindromes_l725_725168

def is_six_digit_palindrome (n : ℕ) : Prop := 
  100000 ≤ n ∧ n ≤ 999999 ∧ (∀ a b c : ℕ, 
    n = 100000 * a + 10000 * b + 1000 * c + 100 * c + 10 * b + a → a ≠ 0)

theorem number_of_six_digit_palindromes : 
  ∃ (count : ℕ), (count = 900 ∧ 
  ∀ n : ℕ, is_six_digit_palindrome n → true) 
:= 
by 
  use 900 
  sorry

end number_of_six_digit_palindromes_l725_725168


namespace find_a_2004_l725_725106

def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ a 2 = 1 ∧ ∀ n, a (n+2) = (1 / a (n+1)) + a n

theorem find_a_2004 (a : ℕ → ℝ) (h : sequence a) :
  a 2004 = real.double_factorial 2003 / real.double_factorial 2002 :=
sorry

end find_a_2004_l725_725106


namespace solve_system_l725_725986

theorem solve_system :
  ∃ x y : ℤ, (x - 3 * y = 7) ∧ (5 * x + 2 * y = 1) ∧ (x = 1) ∧ (y = -2) :=
by
  sorry

end solve_system_l725_725986


namespace exponents_subtraction_l725_725850

theorem exponents_subtraction (m n : ℕ) (hm : 3 ^ m = 8) (hn : 3 ^ n = 2) : 3 ^ (m - n) = 4 := 
by
  sorry

end exponents_subtraction_l725_725850


namespace geometric_sequence_n_value_l725_725211

theorem geometric_sequence_n_value (a₁ : ℕ) (q : ℕ) (a_n : ℕ) (n : ℕ) (h1 : a₁ = 1) (h2 : q = 2) (h3 : a_n = 64) (h4 : a_n = a₁ * q^(n-1)) : n = 7 :=
by
  sorry

end geometric_sequence_n_value_l725_725211


namespace hyperbola_asymptotes_angle_l725_725871

noncomputable def angle_between_asymptotes 
  (a b : ℝ) (e : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : e = 2 * Real.sqrt 3 / 3) : ℝ :=
  2 * Real.arctan (b / a)

theorem hyperbola_asymptotes_angle (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : e = 2 * Real.sqrt 3 / 3) 
  (b_eq : b = Real.sqrt (e^2 * a^2 - a^2)) : 
  angle_between_asymptotes a b e h1 h2 h3 = π / 3 := 
by
  -- proof omitted
  sorry
  
end hyperbola_asymptotes_angle_l725_725871


namespace sandwiches_left_l725_725969

theorem sandwiches_left (S G K L : ℕ) (h1 : S = 20) (h2 : G = 4) (h3 : K = 2 * G) (h4 : L = S - G - K) : L = 8 :=
sorry

end sandwiches_left_l725_725969


namespace sequence_sum_formula_l725_725482

-- Given conditions
def a : ℕ → ℕ
| 0       := 4
| (n+1)   := 2 * a n

def S (n : ℕ) : ℕ := (finset.range (n+1)).sum a

-- Statement to prove
theorem sequence_sum_formula (n : ℕ) : S n = 2^(n+1) :=
by sorry

end sequence_sum_formula_l725_725482


namespace winning_strategy_for_cycle_game_l725_725909

theorem winning_strategy_for_cycle_game (n : ℕ) (h : n ≥ 3) : 
  (even n ∧ player_A_loses) ∨ (odd n ∧ player_B_loses) := 
  sorry

-- Definitions for even, odd, player_A_loses, and player_B_loses
def even (n : ℕ) : Prop := n % 2 = 0
def odd (n : ℕ) : Prop := ¬ even n

def player_A_starts : Prop := true
def player_A_loses : Prop := even n ∧ player_A_starts
def player_B_loses : Prop := odd n ∧ player_A_starts

end winning_strategy_for_cycle_game_l725_725909


namespace max_sequence_value_l725_725107

noncomputable def sequence_max_value : ℝ :=
  let x : ℝ → ℝ := λ x₀, 2^(997)
  x 0

theorem max_sequence_value (x : ℕ → ℝ) (h₁ : x 0 = x 1995) (h₂ : ∀ i, 1 ≤ i ∧ i ≤ 1995 → x (i - 1) + 2 / x (i - 1) = 2 * x i + 1 / x i) :
  x 0 ≤ 2^(997) := 
sorry

end max_sequence_value_l725_725107


namespace sedrach_bite_size_samples_l725_725255

theorem sedrach_bite_size_samples (num_pies : ℕ) (total_people : ℕ) (num_halves : ℕ) (bite_size_samples : ℕ) :
  num_pies = 13 →
  total_people = 130 →
  num_halves = num_pies * 2 →
  bite_size_samples = total_people / num_halves →
  bite_size_samples = 5 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end sedrach_bite_size_samples_l725_725255


namespace largest_of_seven_consecutive_odd_numbers_l725_725641

theorem largest_of_seven_consecutive_odd_numbers (a b c d e f g : ℤ) 
  (h1: a % 2 = 1) (h2: b % 2 = 1) (h3: c % 2 = 1) (h4: d % 2 = 1) 
  (h5: e % 2 = 1) (h6: f % 2 = 1) (h7: g % 2 = 1)
  (h8 : a + b + c + d + e + f + g = 105)
  (h9 : b = a + 2) (h10 : c = a + 4) (h11 : d = a + 6)
  (h12 : e = a + 8) (h13 : f = a + 10) (h14 : g = a + 12) :
  g = 21 :=
by 
  sorry

end largest_of_seven_consecutive_odd_numbers_l725_725641


namespace probability_at_least_6_heads_in_10_flips_l725_725342

theorem probability_at_least_6_heads_in_10_flips : 
  let total_outcomes := 1024 in 
  let favorable_outcomes := 15 in 
  (favorable_outcomes / total_outcomes : ℚ) = 15 / 1024 :=
by
  sorry

end probability_at_least_6_heads_in_10_flips_l725_725342


namespace number_of_ordered_triples_modulo_1000000_l725_725578

def p : ℕ := 2017
def N : ℕ := sorry -- N is the number of ordered triples (a, b, c)

theorem number_of_ordered_triples_modulo_1000000 (N : ℕ) (h : ∀ (a b c : ℕ), 1 ≤ a ∧ a ≤ p * (p - 1) ∧ 1 ≤ b ∧ b ≤ p * (p - 1) ∧ a^b - b^a = p * c → true) : 
  N % 1000000 = 2016 :=
sorry

end number_of_ordered_triples_modulo_1000000_l725_725578


namespace sleep_ratio_l725_725782

theorem sleep_ratio (k : ℕ) : 
  let first_night := 6,
      second_night := first_night + 2,   -- 2 more hours than the previous night
      third_night := second_night / 2,   -- Half the previous amount
      fourth_night := k * third_night    -- Some multiple of the third night
  in (first_night + second_night + third_night + fourth_night = 30) →
     k * third_night / third_night = 3 :=
begin
  sorry
end

end sleep_ratio_l725_725782


namespace last_three_nonzero_digits_of_factorial_l725_725400

theorem last_three_nonzero_digits_of_factorial (n k : ℕ) (h1 : n = 80) (h2 : k = 19) :
  (nat.factorial 80 / 10^19) % 1000 = 712 := 
sorry

end last_three_nonzero_digits_of_factorial_l725_725400


namespace logarithmic_equation_solution_l725_725716

theorem logarithmic_equation_solution (x : ℝ) (h : x > 0) :
  3 ^ (∑ i in (finset.range 8 ).map (λ i, (i + 1) * real.log x) ) = 27 * x ^ 30 →
  x = real.sqrt 3 :=
by
  sorry

end logarithmic_equation_solution_l725_725716


namespace largest_integer_m_property_l725_725802

theorem largest_integer_m_property (M : ℝ) (hM : M > 1) :
  (∀ (s : Finset ℝ), s.card = 10 → (∀ (a b c : ℝ), a ∈ s → b ∈ s → c ∈ s → a < b → b < c → (a * b * c) < (b ^ 2))) ↔ (M ≤ 4 ^ 255) :=
sorry

end largest_integer_m_property_l725_725802


namespace discuss_monotonicity_inequality_proof_l725_725132

section Part1
variable {a : ℝ} (h : a ≠ 0)
def f (x : ℝ) : ℝ := a * x * exp x
theorem discuss_monotonicity : 
  (∀ x, f' x = a * (x+1) * exp x) →
  (a > 0 → ∀ x, (x < -1 → f' x < 0) ∧ (x > -1 → f' x > 0)) ∧
  (a < 0 → ∀ x, (x < -1 → f' x > 0) ∧ (x > -1 → f' x < 0)) :=
sorry
end Part1

section Part2
variable {a : ℝ} (h : a ≠ 0) (h₁ : a ≥ (4 / (exp 2)))
def f (x : ℝ) : ℝ := a * x * exp x
theorem inequality_proof (x : ℝ) (hx : x > 0) : 
  (f x / (x + 1)) - ((x + 1) * log x) > 0 :=
sorry
end Part2

end discuss_monotonicity_inequality_proof_l725_725132


namespace table_size_condition_l725_725190

-- Define the problem in Lean 4
theorem table_size_condition (n : ℕ) (a : ℕ → ℕ → ℝ) (_ : ∀ i j, 0 ≤ a i j) 
  (H1 : ∀ i, ∃ j, 0 < a i j) (H2 : ∀ j, ∃ i, 0 < a i j)
  (H3 : ∀ i j, 0 < a i j → (∑ k, a i k) = (∑ k, a k j)) : n = 2015 :=
sorry

end table_size_condition_l725_725190


namespace asha_remaining_money_l725_725039

theorem asha_remaining_money :
  let brother := 20
  let father := 40
  let mother := 30
  let granny := 70
  let savings := 100
  let total_money := brother + father + mother + granny + savings
  let spent := (3 / 4) * total_money
  let remaining := total_money - spent
  remaining = 65 :=
by
  sorry

end asha_remaining_money_l725_725039


namespace find_f_of_conditions_l725_725277

theorem find_f_of_conditions (f : ℝ → ℝ) :
  (f 1 = 1) →
  (∀ x y : ℝ, f (x + y) = 3^y * f x + 2^x * f y) →
  (∀ x : ℝ, f x = 3^x - 2^x) :=
by
  intros h1 h2
  sorry

end find_f_of_conditions_l725_725277


namespace nicky_and_cristina_race_time_l725_725243

theorem nicky_and_cristina_race_time 
    (h1 : ∀ t, (7 * t - 4 * t = 120)) : 
    ∃ t, t = 40 := 
by
    use 40
    sorry

end nicky_and_cristina_race_time_l725_725243


namespace frank_bought_2_bags_of_chips_l725_725825

theorem frank_bought_2_bags_of_chips
  (cost_choco_bar : ℕ)
  (num_choco_bar : ℕ)
  (total_money : ℕ)
  (change : ℕ)
  (cost_bag_chip : ℕ)
  (num_bags_chip : ℕ)
  (h1 : cost_choco_bar = 2)
  (h2 : num_choco_bar = 5)
  (h3 : total_money = 20)
  (h4 : change = 4)
  (h5 : cost_bag_chip = 3)
  (h6 : total_money - change = (cost_choco_bar * num_choco_bar) + (cost_bag_chip * num_bags_chip)) :
  num_bags_chip = 2 := by
  sorry

end frank_bought_2_bags_of_chips_l725_725825


namespace remove_candies_even_distribution_l725_725638

theorem remove_candies_even_distribution (candies friends : ℕ) (h_candies : candies = 30) (h_friends : friends = 4) :
  ∃ k, candies - k % friends = 0 ∧ k = 2 :=
by
  sorry

end remove_candies_even_distribution_l725_725638


namespace necessary_but_not_sufficient_l725_725529

theorem necessary_but_not_sufficient (a b : ℝ) : (a^2 > b^2) ↔ (a > |b|) :=
begin
  sorry,
end

end necessary_but_not_sufficient_l725_725529


namespace triangle_inequality_l725_725516

theorem triangle_inequality (a b c R r : ℝ) 
  (habc : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_area1 : a * b * c = 4 * R * S)
  (h_area2 : S = r * (a + b + c) / 2) :
  (b^2 + c^2) / (2 * b * c) ≤ R / (2 * r) := 
sorry

end triangle_inequality_l725_725516


namespace sum_of_integers_70_to_85_l725_725303

theorem sum_of_integers_70_to_85 :
  let range_start := 70
  let range_end := 85
  let n := range_end - range_start + 1
  let sum := (range_start + range_end) * n / 2
  sum = 1240 :=
by
  let range_start := 70
  let range_end := 85
  let n := range_end - range_start + 1
  let sum := (range_start + range_end) * n / 2
  sorry

end sum_of_integers_70_to_85_l725_725303


namespace curve_is_line_l725_725273

theorem curve_is_line (x y : ℝ) : (x^2 + y^2 - 2) * real.sqrt (x - 3) = 0 ↔ x = 3 :=
by sorry

end curve_is_line_l725_725273


namespace equal_angles_l725_725202

-- Definitions of points and properties in a convex quadrilateral
structure Point :=
(x : ℝ) (y : ℝ)

def midpoint (p1 p2 : Point) : Point :=
{ x := (p1.x + p2.x) / 2,
  y := (p1.y + p2.y) / 2 }

variable (A B C D K L M N S : Point)

-- Conditions of the problem
axiom H1 : K = midpoint A B
axiom H2 : L = midpoint B C
axiom H3 : M = midpoint C D
axiom H4 : N = midpoint D A
axiom H5 : ∃ S, S ∈ interior A B C D
axiom H6 : dist K S = dist L S
axiom H7 : dist N S = dist M S

-- Theorem to be proved
theorem equal_angles (A B C D K L M N S : Point) 
  (H1 : K = midpoint A B) 
  (H2 : L = midpoint B C)
  (H3 : M = midpoint C D)
  (H4 : N = midpoint D A)
  (H5 : ∃ S, S ∈ interior A B C D)
  (H6 : dist K S = dist L S)
  (H7 : dist N S = dist M S) :
  ∠ K S N = ∠ M S L := 
by 
  sorry

end equal_angles_l725_725202


namespace even_rolls_probability_l725_725354

noncomputable theory

-- Definition of rolling a die and event of rolling at least three even numbers in four rolls
def roll_even_prob : ℚ := 3 / 6

def at_least_three_even_rolls : Prop :=
  (comb 4 3 * (roll_even_prob ^ 3) * ((1 - roll_even_prob) ^ 1) +
   comb 4 4 * (roll_even_prob ^ 4)) = 5 / 16

theorem even_rolls_probability :
  at_least_three_even_rolls :=
sorry

end even_rolls_probability_l725_725354


namespace lcm_of_36_and_100_l725_725443

theorem lcm_of_36_and_100 : Nat.lcm 36 100 = 900 :=
by
  -- The proof is omitted
  sorry

end lcm_of_36_and_100_l725_725443


namespace incircle_midpoint_condition_l725_725669

theorem incircle_midpoint_condition 
  (A B C D E F Q M : Point)
  (h1 : AC = BC)
  (h2 : incircle_touching BC D ∧ incircle_touching CA E ∧ incircle_touching AB F)
  (h3 : line_meets_circle AD Q)
  (hM_mid : midpoint M A F) :
  (line_passes_through_midpoint EQ M) ↔ (AC = BC) :=
sorry

end incircle_midpoint_condition_l725_725669


namespace building_floors_l725_725031

theorem building_floors (top_floor : ℕ) (start_floor down1 up1 up2 : ℕ) 
  (h_start : start_floor = 9)
  (h_down1 : down1 = 7)
  (h_up1 : up1 = 3)
  (h_up2 : up2 = 8) 
  (h_end : start_floor - down1 + up1 + up2 = top_floor) : 
  top_floor = 13 := 
by 
  rw [h_start, h_down1, h_up1, h_up2] at h_end 
  simp at h_end 
  exact h_end 

end building_floors_l725_725031


namespace range_of_b_l725_725870

noncomputable def f (x : ℝ) : ℝ := x^2
noncomputable def g (x : ℝ) : ℝ := log x

theorem range_of_b (a b : ℝ) (h : f a = g b) : 1 ≤ b :=
by
  sorry

end range_of_b_l725_725870


namespace correct_calculation_l725_725173

theorem correct_calculation (x : ℕ) (h1 : x + 238 = 637) (h2 : x = 399) : 399 - 382 = 17 := 
by {
    sorry,
}

end correct_calculation_l725_725173


namespace functional_equation_solution_l725_725809

theorem functional_equation_solution (f : ℝ → ℝ)
  (h : ∀ x y, f (x ^ 2) - f (y ^ 2) + 2 * x + 1 = f (x + y) * f (x - y)) :
  (∀ x, f x = x + 1) ∨ (∀ x, f x = -x - 1) :=
by
  sorry

end functional_equation_solution_l725_725809


namespace probability_of_drawing_A_l725_725740

-- Definitions of the conditions
variables (n : ℕ) (A_units : ℕ := 10) (B_units : ℕ := 15) (total_products : ℕ := A_units + B_units)

-- Probabilities of combinations
noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

noncomputable def comb (n k : ℕ) : ℕ :=
factorial n / (factorial k * factorial (n - k))

noncomputable def prob_draw_A : ℚ :=
(comb A_units 2 + comb A_units 1 * comb B_units 1) / comb total_products 2

-- Main proof statement
theorem probability_of_drawing_A :
  prob_draw_A = 39 / 60 :=
sorry

end probability_of_drawing_A_l725_725740


namespace trajectory_equation_and_fixed_point_l725_725119

theorem trajectory_equation_and_fixed_point :
  (∀ (P M Q: ℝ × ℝ),
    (P.1^2 + P.2^2 = 6) → 
    (Q = (P.1, 0)) →
    ((1 - real.sqrt 3) * Q = P - real.sqrt 3 * M) →
    (∃ (x y : ℝ), (x, y) = M ∧ (x^2 / 6 + y^2 / 2 = 1))) ∧
  (∃ D : ℝ × ℝ, D = (7 / 3, 0) ∧ 
    ∀ (A B : ℝ × ℝ),
    (∃ m : ℝ, A.1 = m * A.2 + 2 ∧ B.1 = m * B.2 + 2 ∧
    (A.1^2 / 6 + A.2^2 / 2) = 1 ∧ (B.1^2 / 6 + B.2^2 / 2) = 1) →
    let DA := (A.1 - D.1, A.2) in
    let DB := (B.1 - D.1, B.2) in
    DA.1^2 + DA.2^2 + (DA.1 * DB.1 + DA.2 * DB.2) = -5 / 9) :=
sorry

end trajectory_equation_and_fixed_point_l725_725119


namespace min_value_of_diff_vec_l725_725886

noncomputable def minValue (a b : ℝ × ℝ) : ℝ := 
  (|a.1 - b.1|^2 + |a.2 - b.2|^2)^0.5

theorem min_value_of_diff_vec (a b : ℝ × ℝ) 
  (h1 : real.angle (a, b) = real.pi / 1.5) 
  (h2 : a.1 * b.1 + a.2 * b.2 = -1) : 
  minValue a b = real.sqrt 6 := 
sorry

end min_value_of_diff_vec_l725_725886


namespace perpendicular_lines_solve_a_l725_725468

theorem perpendicular_lines_solve_a (a : ℝ) (x y : ℝ) : 
  (∀ x, y = -3 * x - 7) →
  (∀ x, 9 * y + a * x = 15) →
  a = -3 :=
by
  intro line1 line2
  -- Given the conditions of the equations of the lines
  have slope1 : -3 := sorry
  have slope2 : - a / 9 := sorry
  -- proving perpendicular condition
  have perp_cond : slope1 * slope2 = -1 := sorry
  sorry

end perpendicular_lines_solve_a_l725_725468


namespace trajectory_midpoint_l725_725594

def point_on_circle (P : ℝ × ℝ) : Prop :=
  P.1^2 + P.2^2 = 4

def fixed_point : ℝ × ℝ := (8, 0)

def midpoint (P D M : ℝ × ℝ) : Prop :=
  M.1 = (P.1 + D.1) / 2 ∧ M.2 = (P.2 + D.2) / 2

theorem trajectory_midpoint 
  (P : ℝ × ℝ) (M : ℝ × ℝ)
  (hP : point_on_circle P)
  (hM : midpoint P fixed_point M) : 
  (M.1 - 4)^2 + (M.2)^2 = 1 := 
sorry

end trajectory_midpoint_l725_725594


namespace num_ordered_pairs_l725_725401

theorem num_ordered_pairs : 
  { n : ℕ // ∀ x y : ℤ, (xy ≥ 0) ∧ (x^3 + y^3 + 27*xy = 30^3) ∧ (x % 2 = y % 2) → n = 17} :=
begin
  sorry
end

end num_ordered_pairs_l725_725401


namespace sequence_a_n_value_l725_725215

noncomputable def a : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 1) := a n + n

theorem sequence_a_n_value : a 100 = 4951 :=
by
  sorry

end sequence_a_n_value_l725_725215


namespace sum_of_first_120_terms_l725_725668

noncomputable def a (n : ℕ) : ℝ := 
  if n = 0 then 0 else (-1) ^ n * (2 * n - 1) * Real.cos (n * Real.pi / 2) + 1

def S (n : ℕ) : ℝ := ∑ i in Finset.range (n + 1), a i

theorem sum_of_first_120_terms : S 120 = 240 := by
  sorry

end sum_of_first_120_terms_l725_725668


namespace systematic_sampling_intervals_l725_725373

theorem systematic_sampling_intervals (total_employees : ℕ) (selected_people : ℕ) (range_start : ℕ) (range_end : ℕ) (known_selected : ℕ) :
  total_employees = 840 → selected_people = 42 → range_start = 490 → range_end = 700 → known_selected = 13 →
  let k := total_employees / selected_people in
  let num_in_interval := (range_end - known_selected) / k - max ((range_start - known_selected + k - 1) / k) + 1 in
  num_in_interval = 11 :=
begin
  intros _ _ _ _ _,
  conv_rhs { rw [div_eq_div, mul_comm, mul_div_cancel'] };
  sorry,
end

end systematic_sampling_intervals_l725_725373


namespace area_of_flowerbed_l725_725787

theorem area_of_flowerbed :
  ∀ (a b : ℕ), 2 * (a + b) = 24 → b + 1 = 3 * (a + 1) → 
  let shorter_side := 3 * a
  let longer_side := 3 * b
  shorter_side * longer_side = 144 :=
by
  sorry

end area_of_flowerbed_l725_725787


namespace first_candidate_valid_votes_percentage_l725_725208

theorem first_candidate_valid_votes_percentage
  (total_votes : ℕ)
  (invalid_votes_percentage : ℚ)
  (valid_votes_second_candidate : ℚ)
  (total_votes = 9000)
  (invalid_votes_percentage = 0.30)
  (valid_votes_second_candidate = 2520) :
  let valid_votes_total := total_votes * (1 - invalid_votes_percentage)
  let valid_votes_first_candidate := valid_votes_total - valid_votes_second_candidate
  let percentage_first_candidate := (valid_votes_first_candidate / valid_votes_total) * 100
  percentage_first_candidate = 60 := 
by
  sorry

end first_candidate_valid_votes_percentage_l725_725208


namespace number_of_quadratic_functions_l725_725829

theorem number_of_quadratic_functions :
  set.univ.card = 8 ∧ 
  ∀ a b : ℤ,
    a ≠ 0 → 
    ((a < 0 ∧ b > 0) ∨ (a > 0 ∧ b > 0)) → 
    -3 ≤ a ∧ a ≤ 4 ∧ -3 ≤ b ∧ b ≤ 4 → 
    sum_of_combinations (8, 3, 24) :=
begin
  sorry
end

end number_of_quadratic_functions_l725_725829


namespace family_gathering_l725_725540

theorem family_gathering (P : ℕ) 
  (h1 : (P / 2 = P - 10)) : P = 20 :=
sorry

end family_gathering_l725_725540


namespace find_b_from_ellipse_l725_725854

-- Definitions used in conditions
variables {F₁ F₂ : ℝ → ℝ} -- foci
variables (a b c : ℝ) (a_pos : a > 0) (b_pos : b > 0) (a_gt_b : a > b)
noncomputable def ellipse (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Conditions
def point_on_ellipse (P : ℝ × ℝ) : Prop := ellipse a b P.1 P.2
def perpendicular_vectors (P : ℝ × ℝ) : Prop := true -- Simplified, use correct condition in detailed proof
def area_of_triangle (P : ℝ × ℝ) (F₁ F₂ : ℝ → ℝ) : ℝ := 9

-- The target statement
theorem find_b_from_ellipse (P : ℝ × ℝ) (condition1 : point_on_ellipse a b P)
  (condition2 : perpendicular_vectors P) 
  (condition3 : area_of_triangle P F₁ F₂ = 9) : 
  b = 3 := 
sorry

end find_b_from_ellipse_l725_725854


namespace pairs_of_opposite_numbers_l725_725379

-- define the notion of being opposite numbers
def are_opposite (a b : ℤ) : Prop := a = -b

-- condition definitions
def pair_1_is_opposite : Prop := are_opposite (- -3) (-| -3 |)
def pair_2_is_opposite : Prop := are_opposite ((-2) ^ 4) (- (2 ^ 4))
def pair_3_is_opposite : Prop := are_opposite ((-2) ^ 3) ((-3) ^ 2)
def pair_4_is_opposite : Prop := are_opposite ((-2) ^ 3) (- (2 ^ 3))

-- list of pairs that are the correct solution
def correct_pairs : List ℕ := [1, 2]

-- theorem stating that correct pairs are 1 and 2
theorem pairs_of_opposite_numbers :
  (if pair_1_is_opposite then 1 else 0) + (if pair_2_is_opposite then 2 else 0) + (if pair_3_is_opposite then 3 else 0) + (if pair_4_is_opposite then 4 else 0) = 3 :=
by sorry

end pairs_of_opposite_numbers_l725_725379


namespace repeated_letter_adjacent_probability_l725_725758

open Finset

noncomputable def probability_repeated_adjacent
  (letters : Finset Char) (X : Char) (H : X ∈ letters) (k : Nat) : ℝ :=
  let n := 10
  let factorial (n : Nat) := if n = 0 then 1 else n * factorial (n - 1)
  let total_arrangements := factorial 10 / factorial 2
  let adjacent_arrangements := factorial 9
  adjacent_arrangements / total_arrangements

theorem repeated_letter_adjacent_probability 
  (letters : Finset Char) (X : Char) (H : X ∈ letters) :
  letters = { 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', X } → 
  probability_repeated_adjacent letters X H 10 = 0.2 :=
by
  intro h1
  rw [probability_repeated_adjacent, h1]
  sorry

end repeated_letter_adjacent_probability_l725_725758


namespace mean_of_all_students_is_65_l725_725242

def morn_mean : ℚ := 88
def aft_mean : ℚ := 74
def eve_mean : ℚ := 80
def morn_aft_ratio : ℚ := 4 / 5
def morn_eve_ratio : ℚ := 2 / 3

theorem mean_of_all_students_is_65 :
  let m := 1     -- Assume m is 1 for simplicity of calculation
      a := 5 / 4 * m
      e := 3 / 2 * m
      total_students := m + a + e
      total_scores := morn_mean * m + aft_mean * a + eve_mean * e
  in (total_scores / total_students) = 65 := by
  sorry

end mean_of_all_students_is_65_l725_725242


namespace quadrilateral_is_trapezoid_l725_725923

noncomputable section

variables {A B C D O O1 : Type} 

-- Definitions of conditions
def is_cyclic (abcd : Quadrilateral A B C D) : Prop := sorry -- Placeholder for cyclic condition
def intersects (ac bd : Line A C B D) (o : Point O) : Prop := sorry -- Placeholder for intersection condition
def circumscribes (triangle_cod : Triangle C O D) (o1 : Point O1) : Prop := sorry -- Placeholder for circumscribing condition over O1

-- Original problem conditions
axiom abcd_cyclic : is_cyclic (Quadrilateral A B C D)
axiom diagonals_intersect : intersects (Line A C) (Line B D) O
axiom circle_through_o1 : circumscribes (Triangle C O D) O1

-- Statement to prove
theorem quadrilateral_is_trapezoid :
  quadrilateral ABCD → (∃ P : Point, parallel (Line P B) (Line P D)) :=
by
  sorry

end quadrilateral_is_trapezoid_l725_725923


namespace fraction_irreducible_l725_725642

theorem fraction_irreducible (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := 
by {
    sorry
}

end fraction_irreducible_l725_725642


namespace first_over_100_paperclips_is_friday_l725_725824

-- Definitions for the problem
def first_day_paperclips := 5
def increase_per_day := 3
def target_paperclips := 100
def days_in_week := 7
def day_of_week (day_count : ℕ) : String :=
  match day_count % days_in_week with
  | 0 => "Sunday"
  | 1 => "Monday"
  | 2 => "Tuesday"
  | 3 => "Wednesday"
  | 4 => "Thursday"
  | 5 => "Friday"
  | _ => "Saturday"

-- Proving the desired day of the week
theorem first_over_100_paperclips_is_friday :
  let n := (target_paperclips - (first_day_paperclips - increase_per_day)) / increase_per_day
  day_of_week (n + 1) = "Friday" :=
begin
  -- Calculate n
  let n := (target_paperclips - (first_day_paperclips - increase_per_day)) / increase_per_day,
  -- Prove that n + 1 corresponds to a Friday
  have h1 : n + 1 = 33, {
    sorry
  },
  have h2 : day_of_week 33 = "Friday", {
    sorry
  },
  rw h2,
  sorry
end

end first_over_100_paperclips_is_friday_l725_725824


namespace range_of_m_l725_725481

noncomputable def withinFourthQuadrant (m : ℝ) : Prop :=
  let z := (m + 1, m - 2) in
  z.1 > 0 ∧ z.2 < 0

theorem range_of_m (m : ℝ) (h : withinFourthQuadrant m) : -1 < m ∧ m < 2 :=
by {
  sorry
}

end range_of_m_l725_725481


namespace table_condition_l725_725193

-- Define the conditions used in the problem
variables (matrix : ℕ → ℕ → ℕ)
variables (rows cols : ℕ)
variables (positive_in_every_row : ∀ i < rows, ∃ j < cols, 0 < matrix i j)
variables (positive_in_every_col : ∀ j < cols, ∃ i < rows, 0 < matrix i j)
variables (sum_condition : ∀ i j, 0 < matrix i j → (∑ k, matrix i k) = (∑ k, matrix k j))

-- State the theorem we need to prove
theorem table_condition (h : rows = 2015) : cols = 2015 :=
sorry

end table_condition_l725_725193


namespace probability_at_least_6_heads_in_10_flips_l725_725344

theorem probability_at_least_6_heads_in_10_flips : 
  let total_outcomes := 1024 in 
  let favorable_outcomes := 15 in 
  (favorable_outcomes / total_outcomes : ℚ) = 15 / 1024 :=
by
  sorry

end probability_at_least_6_heads_in_10_flips_l725_725344


namespace max_non_intersecting_segments_l725_725836

theorem max_non_intersecting_segments (n : ℕ) (h : n ≥ 4) (no_three_collinear : ∀ {p1 p2 p3 : ℝ × ℝ}, (p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3) → ¬ collinear p1 p2 p3) : 
  (exists max_segments : ℕ, max_segments = 2 * n - 2) :=
by
  sorry

end max_non_intersecting_segments_l725_725836


namespace fraction_of_area_l725_725619

noncomputable def area_of_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  let base := (C.1 - A.1).abs
  let height := B.2
  (base * height) / 2

theorem fraction_of_area {A B C X Y Z: (ℝ × ℝ)}
  (hA : A = (2, 0)) (hB : B = (8, 12)) (hC : C = (14, 0))
  (hX : X = (6, 0)) (hY : Y = (8, 4)) (hZ : Z = (10, 0)):
  (area_of_triangle X Y Z) / (area_of_triangle A B C) = 1 / 9 :=
by
  -- Skipping the proof with 'sorry'
  sorry

end fraction_of_area_l725_725619


namespace total_cards_correct_l725_725959

-- Definitions based on the given conditions
def J (B : ℝ) : ℝ := B + 9.5
def M : ℝ := 210 - 60
def C (M : ℝ) : ℝ := 0.8 * M
def B (J : ℝ) : ℝ := J - 9.5
def total_cards (J B M C : ℝ) : ℝ := J + B + M + C

-- The theorem to prove
theorem total_cards_correct : 
  let M := 210 - 60 in
  let J := (M / 1.75 : ℝ) in
  let B := J - 9.5 in
  total_cards J B M (C M) = 431.92 :=
by
  -- Placeholder for the proof
  sorry

end total_cards_correct_l725_725959


namespace find_central_angle_l725_725125

-- Define the conditions
def radius (r : ℝ) := r = 2
def area (S : ℝ) := S = 4

-- Define the formula for the area of a sector
def sector_area (r α : ℝ) := (1 / 2) * α * r^2

theorem find_central_angle (r α S : ℝ) (hr : radius r) (hS : area S) : α = 2 :=
by
  unfold radius at hr
  unfold area at hS
  rw [hr, hS] at *
  have h1 : S = (1 / 2) * α * r^2 := by sorry
  have h2 : 4 = (1 / 2) * α * 2^2 := by sorry
  have h3 : 4 = 2 * α := by sorry
  have h4 : 2 = α := by linarith
  exact h4

end find_central_angle_l725_725125


namespace correct_intersection_l725_725880

def A := {x : ℝ | x^2 - x - 2 ≤ 0}
def B := {z : ℤ | true}

theorem correct_intersection : A ∩ (B : set ℝ) = {-1, 0, 1, 2} :=
by sorry

end correct_intersection_l725_725880


namespace range_shift_l725_725660

variable {α β : Type} [LinearOrder α] [LinearOrder β]

def transformed_range (f : α → β) (a b : β) (h : set.range f = set.Icc a b) : Prop :=
  set.range (λ x => f (x + 4)) = set.Icc a b

theorem range_shift (f : ℝ → β) (a b : β) (h_dom : set.Icc (-2 : ℝ) 3 ⊆ set.univ)
  (h_rng : set.range f = set.Icc a b) : transformed_range f a b h_rng :=
sorry

end range_shift_l725_725660


namespace proof_problem_l725_725109

namespace MathProof

-- Given conditions and data
def dataset (a : ℕ) : List ℕ := [4, 2, a, 10, 7]
def average (l : List ℕ) : ℕ := l.sum / l.length

-- Definitions of mode, median, and variance

def mode (l : List ℕ) : ℕ := l.foldr (λ x acc, if l.count x > l.count acc then x else acc) 0

def median (l : List ℕ) : ℕ :=
  let sortedL := l.qsort (≤)
  if sortedL.length % 2 == 1 then
    sortedL.get (sortedL.length / 2)
  else
    (sortedL.get (sortedL.length / 2 - 1) + sortedL.get (sortedL.length / 2)) / 2

def variance (l : List ℕ) : ℚ :=
  let avg := average l
  (l.foldr (λ x acc, acc + ((x - avg) * (x - avg))) 0 : ℚ) / (l.length : ℚ)

theorem proof_problem:
  ∀ (a : ℕ),
  average (dataset a) = 5 →
  a = 2 ∧ mode (dataset 2) = 2 ∧ median (dataset 2) = 4 ∧ variance (dataset 2) = 48 / 5 := 
by
  intros a h_avg
  sorry

end MathProof

end proof_problem_l725_725109


namespace fraction_equality_l725_725117

variable (a_n b_n : ℕ → ℝ)
variable (S_n T_n : ℕ → ℝ)

-- Conditions
axiom S_T_ratio (n : ℕ) : T_n n ≠ 0 → S_n n / T_n n = (2 * n + 1) / (4 * n - 2)
axiom Sn_def (n : ℕ) : S_n n = n / 2 * (2 * a_n 0 + (n - 1) * (a_n 1 - a_n 0))
axiom Tn_def (n : ℕ) : T_n n = n / 2 * (2 * b_n 0 + (n - 1) * (b_n 1 - b_n 0))
axiom an_def (n : ℕ) : a_n n = a_n 0 + n * (a_n 1 - a_n 0)
axiom bn_def (n : ℕ) : b_n n = b_n 0 + n * (b_n 1 - b_n 0)

-- Proof statement
theorem fraction_equality :
  (b_n 3 + b_n 18) ≠ 0 → (b_n 6 + b_n 15) ≠ 0 →
  (a_n 10 / (b_n 3 + b_n 18) + a_n 11 / (b_n 6 + b_n 15)) = (41 / 78) :=
by
  sorry

end fraction_equality_l725_725117


namespace average_of_roots_l725_725755

theorem average_of_roots (p q : ℝ) (h : ∃ x1 x2 : ℝ, 3*p*x1^2 - 6*p*x1 + q = 0 ∧ 3*p*x2^2 - 6*p*x2 + q = 0 ∧ x1 ≠ x2):
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 3*p*x1^2 - 6*p*x1 + q = 0 ∧ 3*p*x2^2 - 6*p*x2 + q = 0) → 
  (x1 + x2) / 2 = 1 :=
by
  sorry

end average_of_roots_l725_725755


namespace sixtieth_pair_is_5_7_l725_725142

-- Define the sequence of pairs based on the given pattern
def pair_sequence : ℕ → (ℕ × ℕ)
| n :=
  let sum := Nat.find (λ x, (x * (x + 1)) / 2 ≥ n) in
  let k := n - (sum * (sum - 1)) / 2 in
  (k, sum - k)

theorem sixtieth_pair_is_5_7 : pair_sequence 60 = (5, 7) := 
by {
  sorry
}

end sixtieth_pair_is_5_7_l725_725142


namespace cube_root_identity_l725_725528

variable {a : ℝ}

theorem cube_root_identity (h : real.cbrt (-a) = real.sqrt 2) : real.cbrt a = -real.sqrt 2 := by
  sorry

end cube_root_identity_l725_725528


namespace number_smaller_than_neg_one_l725_725773

theorem number_smaller_than_neg_one :
  let numbers := [-1.1, -0.9, 0, 1] in
  ∃ (x : ℝ), (x ∈ numbers ∧ x < -1) ∧ x = -1.1 :=
by
  let numbers := [-1.1, -0.9, 0, 1]
  use -1.1
  split
  { simp [numbers] }
  { norm_num[sorry] }

end number_smaller_than_neg_one_l725_725773


namespace julia_miles_l725_725794

theorem julia_miles (total_miles darius_miles julia_miles : ℕ) 
  (h1 : darius_miles = 679)
  (h2 : total_miles = 1677)
  (h3 : total_miles = darius_miles + julia_miles) :
  julia_miles = 998 :=
by
  sorry

end julia_miles_l725_725794


namespace arithmetic_prog_sum_bound_l725_725329

noncomputable def Sn (n : ℕ) (a1 : ℝ) (d : ℝ) : ℝ := n * a1 + (n * (n - 1) / 2) * d

theorem arithmetic_prog_sum_bound (n : ℕ) (a1 an : ℝ) (d : ℝ) (h_d_neg : d < 0) 
  (ha_n : an = a1 + (n - 1) * d) :
  n * an < Sn n a1 d ∧ Sn n a1 d < n * a1 :=
by 
  sorry

end arithmetic_prog_sum_bound_l725_725329


namespace find_angle_A_find_max_value_and_area_l725_725552

noncomputable theory

-- Definitions for the triangle △ABC
variables (a b c A B C : ℝ)
-- Condition given in the problem
variables (h₁ : b / a + sin (A - B) = sin C)

-- Prove that the value of angle A is π/4
theorem find_angle_A (h₁ : b / a + sin (A - B) = sin C) : A = π / 4 :=
sorry

-- Given that a = 2, prove the maximum value of √2 b + 2c and the area of triangle is 12/5
theorem find_max_value_and_area
  (h₁ : b / a + sin (A - B) = sin C)
  (ha : a = 2)
  (hA : A = π / 4) :
  (√2 * b + 2 * c, (1/2) * a * b * sin C) = (12 / 5,  12 / 5) :=
sorry

end find_angle_A_find_max_value_and_area_l725_725552


namespace number_of_players_sold_eq_2_l725_725011

def initial_balance : ℕ := 100
def selling_price_per_player : ℕ := 10
def buying_cost_per_player : ℕ := 15
def number_of_players_bought : ℕ := 4
def final_balance : ℕ := 60

theorem number_of_players_sold_eq_2 :
  ∃ x : ℕ, (initial_balance + selling_price_per_player * x - buying_cost_per_player * number_of_players_bought = final_balance) ∧ (x = 2) :=
by
  sorry

end number_of_players_sold_eq_2_l725_725011


namespace time_to_upload_file_l725_725319

-- Define the conditions
def file_size : ℕ := 160
def upload_speed : ℕ := 8

-- Define the question as a proof goal
theorem time_to_upload_file :
  file_size / upload_speed = 20 := 
sorry

end time_to_upload_file_l725_725319


namespace seating_arrangements_count_l725_725699

-- Definitions and conditions for the problem
def chairs := Fin 12
def married_couples := Fin 6
def seating_arrangements (couple_rule : married_couples → Prop) : Finset (chairs → chairs) := sorry

-- Condition stating that no one can sit across from their spouse and only one specific couple can sit next to each other
def valid_arrangement (arr : chairs → chairs) : Prop :=
  ∀ (i : chairs), ∃ (j : chairs),
    -- Ensure men and women alternate (for simplicity in definition we assume fixed alternating pattern)
    (arr i).to_nat % 2 = i.to_nat % 2 ∧ 
    -- Ensure no one sits across from their spouse
    (arr i + 6 ≠ arr j) ∨ 
    -- Allow one specified couple to sit next to each other
    couple_rule (arr i // 2)

-- The main theorem
theorem seating_arrangements_count :
  (seating_arrangements (λ c, c = 0)).card = 2880 := sorry

end seating_arrangements_count_l725_725699


namespace find_recip_sum_of_shifted_roots_l725_725583

noncomputable def reciprocal_sum_of_shifted_roots (α β γ : ℝ) (hαβγ : Polynomial.roots (Polynomial.C α * Polynomial.C β * Polynomial.C γ + Polynomial.X ^ 3 - 2 * Polynomial.X ^ 2 - Polynomial.X + Polynomial.C 2) = {α, β, γ}) : ℝ :=
  1 / (α + 2) + 1 / (β + 2) + 1 / (γ + 2)

theorem find_recip_sum_of_shifted_roots (α β γ : ℝ) (hαβγ : Polynomial.roots (Polynomial.C α * Polynomial.C β * Polynomial.C γ + Polynomial.X ^ 3 - 2 * Polynomial.X ^ 2 - Polynomial.X + Polynomial.C 2) = {α, β, γ}) :
  reciprocal_sum_of_shifted_roots α β γ hαβγ = -19 / 14 :=
  sorry

end find_recip_sum_of_shifted_roots_l725_725583


namespace determine_a_l725_725863

noncomputable def f (a x : ℝ) : ℝ :=
if x ≤ 0 then 2^x + 1 else log 2 x + a

theorem determine_a (a : ℝ) (h : f a (f a 0) = 3 * a) : a = 1 / 2 :=
by
  sorry

end determine_a_l725_725863


namespace combined_total_profit_percentage_l725_725593

noncomputable def o := 1000
noncomputable def b := 800
noncomputable def a := 750
noncomputable def p_o := 2.5
noncomputable def p_b := 1.5
noncomputable def p_a := 2.0
noncomputable def r_o := 0.12
noncomputable def r_b := 0.05
noncomputable def r_a := 0.10
noncomputable def P_o := 0.20
noncomputable def P_b := 0.25
noncomputable def P_a := 0.15

theorem combined_total_profit_percentage :
  let CP_oranges := o * p_o,
      CP_bananas := b * p_b,
      CP_apples := a * p_a,
      Total_CP := CP_oranges + CP_bananas + CP_apples,
      
      Good_oranges := o * (1 - r_o),
      Good_bananas := b * (1 - r_b),
      Good_apples := a * (1 - r_a),
      
      SP_oranges := Good_oranges * p_o * (1 + P_o),
      SP_bananas := Good_bananas * p_b * (1 + P_b),
      SP_apples := Good_apples * p_a * (1 + P_a),
      
      Total_SP := SP_oranges + SP_bananas + SP_apples,

      Total_Profit_or_Loss := Total_SP - Total_CP,
      
      Total_Profit_or_Loss_Percentage := (Total_Profit_or_Loss / Total_CP) * 100
  in Total_Profit_or_Loss_Percentage ≈ 8.03 := sorry

end combined_total_profit_percentage_l725_725593


namespace probability_of_6_consecutive_heads_l725_725351

/-- Define the probability of obtaining at least 6 consecutive heads in 10 flips of a fair coin. -/
def prob_at_least_6_consecutive_heads : ℚ :=
  129 / 1024

/-- Proof statement: The probability of getting at least 6 consecutive heads in 10 flips of a fair coin is 129/1024. -/
theorem probability_of_6_consecutive_heads : 
  prob_at_least_6_consecutive_heads = 129 / 1024 := 
by
  sorry

end probability_of_6_consecutive_heads_l725_725351


namespace area_of_region_bounded_by_lines_l725_725430

theorem area_of_region_bounded_by_lines : 
  let x1 := λ θ : ℝ, 2 in
  let y1 := λ θ : ℝ, 2 in
  x1 (0:ℝ) * y1 (0:ℝ) = 4 :=
by {
  sorry,
}

end area_of_region_bounded_by_lines_l725_725430


namespace find_a_for_tangent_line_l725_725918

theorem find_a_for_tangent_line (a : ℝ) :
  (∀ θ : ℝ, let x := a + Real.cos θ in let y := Real.sin θ in 
  let ρ := Math.sqrt (x^2 + y^2) in
  let line_l := (ρ * Real.sin (θ - (Real.pi / 4))) in 
  line_l = (Real.sqrt 2 / 2)) →
  (|a + 1| / Math.sqrt 2 = 1) →
  a = -1 + Real.sqrt 2 ∨ a = -1 - Real.sqrt 2 :=
by
  sorry

end find_a_for_tangent_line_l725_725918


namespace number_of_six_digit_palindromes_l725_725160

theorem number_of_six_digit_palindromes : 
  let count_palindromes : ℕ := 9 * 10 * 10 in
  count_palindromes = 900 :=
by
  sorry

end number_of_six_digit_palindromes_l725_725160


namespace tomatoes_left_l725_725688

theorem tomatoes_left (initial_tomatoes : ℕ) (fraction_eaten : ℚ) (eaters : ℕ) (final_tomatoes : ℕ)  
  (h_initial : initial_tomatoes = 21)
  (h_fraction : fraction_eaten = 1 / 3)
  (h_eaters : eaters = 2)
  (h_final : final_tomatoes = initial_tomatoes - initial_tomatoes * fraction_eaten) :
  final_tomatoes = 14 := by
  sorry

end tomatoes_left_l725_725688


namespace solve_for_x_l725_725538

theorem solve_for_x (x : ℝ) (h : 3 - (1 / (2 - x)) = (1 / (2 - x))) : x = 4 / 3 := 
by {
  sorry
}

end solve_for_x_l725_725538


namespace winter_hamburger_sales_l725_725752

theorem winter_hamburger_sales 
  (spring_sales_percent : ℝ) (summer_sales_percent : ℝ) (total_annual_sales : ℝ) :
  spring_sales_percent = 0.30 →
  summer_sales_percent = 0.35 →
  total_annual_sales = 20 →
  ∃ winter_sales, winter_sales = 3.5 :=
by
  intros h1 h2 h3
  have spring_sales : ℝ := spring_sales_percent * total_annual_sales
  have summer_sales : ℝ := summer_sales_percent * total_annual_sales
  have fall_sales: ℝ := 0.175 * total_annual_sales -- Assuming fall sales are 17.5% of total annual sales
  have winter_sales : ℝ := total_annual_sales - (spring_sales + summer_sales + fall_sales)
  use winter_sales
  rw [h1, h2, h3]
  norm_num at spring_sales
  norm_num at summer_sales
  norm_num at fall_sales
  norm_num at winter_sales
  exact winter_sales

end winter_hamburger_sales_l725_725752


namespace solve_system_l725_725260

theorem solve_system : ∃ (x y : ℝ), 
  (3 * x + 4 * y = 26) ∧
  (sqrt (x^2 + y^2 - 4 * x + 2 * y + 5) + sqrt (x^2 + y^2 - 20 * x - 10 * y + 125) = 10) ∧
  (x = 6 ∧ y = 2) :=
by {
  sorry
}

end solve_system_l725_725260


namespace percentage_error_l725_725764

theorem percentage_error (x : ℚ) : 
  let incorrect_result := (3/5 : ℚ) * x
  let correct_result := (5/3 : ℚ) * x
  let ratio := incorrect_result / correct_result
  let percentage_error := (1 - ratio) * 100
  percentage_error = 64 :=
by
  let incorrect_result := (3/5 : ℚ) * x
  let correct_result := (5/3 : ℚ) * x
  let ratio := incorrect_result / correct_result
  let percentage_error := (1 - ratio) * 100
  sorry

end percentage_error_l725_725764


namespace six_digit_numbers_count_l725_725299

noncomputable def number_of_valid_6_digit_numbers : Nat :=
  1440

theorem six_digit_numbers_count :
  ∃ n : Nat, n = number_of_valid_6_digit_numbers ∧ 
  ∀ digits : List Nat, digits.length = 6 → 
    (∀ d, d ∈ digits → d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4) →
    (∀ d, count digits d ≤ 2) → 
      n = 1440 :=
by
  sorry

end six_digit_numbers_count_l725_725299


namespace lcm_36_100_eq_900_l725_725446

/-- Definition for the prime factorization of 36 -/
def factorization_36 : Prop := 36 = 2^2 * 3^2

/-- Definition for the prime factorization of 100 -/
def factorization_100 : Prop := 100 = 2^2 * 5^2

/-- The least common multiple problem statement -/
theorem lcm_36_100_eq_900 (h₁ : factorization_36) (h₂ : factorization_100) : Nat.lcm 36 100 = 900 := 
by
  sorry

end lcm_36_100_eq_900_l725_725446


namespace distance_squared_between_intersections_l725_725297

-- Definitions of the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 25
def circle2 (x y : ℝ) : Prop := (x - 4)^2 + (y + 2)^2 = 9

-- Hypothesis about points of intersection
def pointC (x y : ℝ) : Prop := circle1 x y ∧ circle2 x y
def pointD (x y : ℝ) : Prop := circle1 x y ∧ circle2 x y ∧ (x, y) ≠ (4, -2)

-- The requirement to prove
theorem distance_squared_between_intersections :
  ∃ (x1 y1 x2 y2 : ℝ),
  pointC x1 y1 ∧ pointD x2 y2 ∧
  ((x1 - x2)^2 + (y1 - y2)^2 = 396.8) :=
sorry

end distance_squared_between_intersections_l725_725297


namespace probability_of_zero_product_l725_725294

open Set

-- Define the set of numbers
def S : Set ℤ := {-3, -2, -1, 0, 2, 3, 5}

-- Define the condition that the product of three numbers is zero
def product_is_zero (a b c : ℤ) : Prop := a * b * c = 0

-- Define the total number of ways to choose 3 different numbers
def total_ways : ℕ := Nat.choose (to_finset S).card 3

-- Define the number of favorable outcomes where one of the numbers is zero
def favorable_ways : ℕ := Nat.choose (to_finset (S \ {0})).card 2

-- The theorem stating the probability of the product being zero
theorem probability_of_zero_product : (Rational.ofNat favorable_ways) / (Rational.ofNat total_ways) = 3 / 7 := by
  sorry

end probability_of_zero_product_l725_725294


namespace initial_oak_trees_l725_725290

theorem initial_oak_trees (n : ℕ) (h : n - 2 = 7) : n = 9 := 
by
  sorry

end initial_oak_trees_l725_725290


namespace smallest_sum_Q_lt_7_9_l725_725050

def Q (N k : ℕ) : ℚ := (N + 1) / (N + k + 1)

theorem smallest_sum_Q_lt_7_9 : 
    ∃ N k : ℕ, (N + k) % 4 = 0 ∧ Q N k < 7 / 9 ∧ (∀ N' k' : ℕ, (N' + k') % 4 = 0 ∧ Q N' k' < 7 / 9 → N' + k' ≥ N + k) ∧ N + k = 4 :=
by
  sorry

end smallest_sum_Q_lt_7_9_l725_725050


namespace area_of_region_l725_725422

noncomputable def sec (θ : ℝ) := (cos θ)⁻¹
noncomputable def csc (θ : ℝ) := (sin θ)⁻¹

def region (r θ : ℝ) : Prop :=
  (r = 2 * sec θ ∧ (0 ≤ θ ∧ θ ≤ π / 2)) ∨ 
  (r = 2 * csc θ ∧ (0 ≤ θ ∧ θ ≤ π / 2))

theorem area_of_region :
  let bounded_region := { (x, y) | 0 ≤ x ∧ x ≤ 2 ∧ 0 ≤ y ∧ y ≤ 2 } in
  ∃ (A : ℝ), A = 4 ∧ (∀ (a b : ℝ), bounded_region (a, b)) :=
begin
  let bounded_region := { p | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2 },
  use 4,
  split,
  { refl, },
  { intros a b hb,
    exact hb, },
end

end area_of_region_l725_725422


namespace chip_placement_unique_l725_725895

theorem chip_placement_unique :
  ∀ (grid_size : ℕ) 
    (red_chips blue_chips green_chips : ℕ), 
    grid_size = 3 → 
    red_chips = 4 → 
    blue_chips = 3 → 
    green_chips = 2 →
    (∀ grid : Matrix (Fin 3) (Fin 3) (Option ℕ),
      (∀ i j, grid i j = some 1 → (∀ di dj, i + di < 3 → j + dj < 3 → grid (i + di) (j + dj) ≠ some 1)) ∧ 
      (∀ i j, grid i j = some 2 → (∀ di dj, i + di < 3 → j + dj < 3 → grid (i + di) (j + dj) ≠ some 2)) ∧
      (∀ i j, grid i j = some 3 → (∀ di dj, i + di < 3 → j + dj < 3 → grid (i + di) (j + dj) ≠ some 3)) 
        → (grid_fun (Matrix.to_fun grid)).count some = 1) :=
begin
  intros grid_size red_chips blue_chips green_chips h_size h_red h_blue h_green grid h_no_adj,
  sorry,
end

end chip_placement_unique_l725_725895


namespace tomatoes_left_l725_725686

theorem tomatoes_left (initial_tomatoes : ℕ) (birds : ℕ) (fraction : ℕ) (E1 : initial_tomatoes = 21) 
  (E2 : birds = 2) (E3 : fraction = 3) : 
  initial_tomatoes - initial_tomatoes / fraction = 14 :=
by 
  sorry

end tomatoes_left_l725_725686


namespace sum_of_parts_l725_725733

theorem sum_of_parts (x y : ℝ) (h1 : x + y = 52) (h2 : y = 30.333333333333332) :
  10 * x + 22 * y = 884 :=
sorry

end sum_of_parts_l725_725733


namespace total_daisies_l725_725561

theorem total_daisies (white pink red : ℕ) (h1 : pink = 9 * white) (h2 : red = 4 * pink - 3) (h3 : white = 6) : 
    white + pink + red = 273 :=
by
  sorry

end total_daisies_l725_725561


namespace min_value_of_expr_l725_725591

noncomputable def min_value_expr : ℝ → ℝ → ℝ :=
  λ x y, sqrt ((2 * x^2 + y^2) * (4 * x^2 + y^2)) / (x * y)

theorem min_value_of_expr (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  min_value_expr x y = 2 * real.sqrt 3 ^ (1/4) :=
sorry

end min_value_of_expr_l725_725591


namespace asha_remaining_money_l725_725040

theorem asha_remaining_money :
  let brother := 20
  let father := 40
  let mother := 30
  let granny := 70
  let savings := 100
  let total_money := brother + father + mother + granny + savings
  let spent := (3 / 4) * total_money
  let remaining := total_money - spent
  remaining = 65 :=
by
  sorry

end asha_remaining_money_l725_725040


namespace field_trip_savings_l725_725005

-- Define the parameters given in the conditions
def num_students : ℕ := 30
def contribution_per_student_per_week : ℕ := 2
def weeks_per_month : ℕ := 4
def num_months : ℕ := 2

-- Define the weekly savings for the class
def weekly_savings : ℕ := num_students * contribution_per_student_per_week

-- Define the total weeks in the given number of months
def total_weeks : ℕ := num_months * weeks_per_month

-- Define the total savings in the given number of months
def total_savings : ℕ := weekly_savings * total_weeks

-- Now, we state the theorem
theorem field_trip_savings : total_savings = 480 :=
by {
  -- calculations are skipped
  sorry
}

end field_trip_savings_l725_725005


namespace six_digit_palindromes_count_l725_725165

def is_digit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9
def is_non_zero_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

theorem six_digit_palindromes_count : 
  (∃a b c : ℕ, is_non_zero_digit a ∧ is_digit b ∧ is_digit c) → 
  (∃ n : ℕ, n = 900) :=
by
  sorry

end six_digit_palindromes_count_l725_725165


namespace quadratic_function_correct_options_c_d_l725_725546

theorem quadratic_function_correct_options_c_d
  (a b c : ℝ) (h1 : a ≠ 0)
  (h2 : -b / (2 * a) = 2)
  (h3 : c > -1) :
  true :=
by {
  sorry
}

end quadratic_function_correct_options_c_d_l725_725546


namespace angles_equal_l725_725658

variables {A B C D O M M' M'' L L' L'' : Type*}
variables [parallelogram A B C D] : Prop
variables (O_intersection: ∀ {A B C D O : Type*} [parallelogram A B C D], diagonals_intersect A B C D O) 
variables (medians: (OM : O → A → Type*) (OM' : O → B → Type*) (OM'' : O → C → Type*)
                   [is_median OM] [is_median OM'] [is_median OM''])
variables (angle_bisectors: (OL : O → A → Type*) (OL' : O → B → Type*) (OL'' : O → C → Type*)
                   [is_angle_bisector OL] [is_angle_bisector OL'] [is_angle_bisector OL''])

theorem angles_equal {A B C D O M M' M'' L L' L'' : Type*}
  [parallelogram A B C D]
  (O_intersection: ∀ {A B C D O : Type*} [parallelogram A B C D], diagonals_intersect A B C D O)
  (medians: (OM : O → A → Type*) (OM' : O → B → Type*) (OM'' : O → C → Type*)
               [is_median OM] [is_median OM'] [is_median OM''])
  (angle_bisectors: (OL : O → A → Type*) (OL' : O → B → Type*) (OL'' : O → C → Type*) 
               [is_angle_bisector OL] [is_angle_bisector OL'] [is_angle_bisector OL'']) :
    angle MM' M'' M = angle LL' L'' L := 
sorry

end angles_equal_l725_725658


namespace truck_driver_gas_l725_725372

variables (miles_per_gallon distance_to_station gallons_to_add gallons_in_tank total_gallons_needed : ℕ)
variables (current_gas_in_tank : ℕ)
variables (h1 : miles_per_gallon = 3)
variables (h2 : distance_to_station = 90)
variables (h3 : gallons_to_add = 18)

theorem truck_driver_gas :
  current_gas_in_tank = 12 :=
by
  -- Prove that the truck driver already has 12 gallons of gas in his tank,
  -- given the conditions provided.
  sorry

end truck_driver_gas_l725_725372


namespace function_relation4_l725_725882

open Set

section
  variable (M : Set ℤ) (N : Set ℤ)

  def relation1 (x : ℤ) := x ^ 2
  def relation2 (x : ℤ) := x + 1
  def relation3 (x : ℤ) := x - 1
  def relation4 (x : ℤ) := abs x

  theorem function_relation4 : 
    M = {-1, 1, 2, 4} →
    N = {1, 2, 4} →
    (∀ x ∈ M, relation4 x ∈ N) :=
  by
    intros hM hN
    simp [relation4]
    sorry
end

end function_relation4_l725_725882


namespace count_integers_leq_zero_l725_725798

def P (x : ℝ) : ℝ :=
  (List.foldr (*) 1 (List.map (λ k : ℕ, x - (k ^ 2)) (List.range 50)))

theorem count_integers_leq_zero :
  ∃ (n : ℕ), (count (λ x : ℤ, P x ≤ 0) (-1) > 1300) - 1 sorry :=
  sorry

end count_integers_leq_zero_l725_725798


namespace distinct_triples_l725_725412

theorem distinct_triples (a b c : ℕ) (h₁: 2 * a - 1 = k₁ * b) (h₂: 2 * b - 1 = k₂ * c) (h₃: 2 * c - 1 = k₃ * a) :
  (a, b, c) = (7, 13, 25) ∨ (a, b, c) = (13, 25, 7) ∨ (a, b, c) = (25, 7, 13) := sorry

end distinct_triples_l725_725412


namespace susann_can_color_boxes_l725_725845

noncomputable def fill_grid (n : ℕ) : matrix (fin n) (fin n) ℤ :=
sorry -- Details of grid filling with distinct integers in each row and column

def is_destroyed (n : ℕ) (destroyed_boxes : fin n × fin n → Prop) :=
∀ i j, destroyed_boxes (i, j) → true

def can_color_boxes (n : ℕ) 
  (grid : matrix (fin n) (fin n) ℤ)
  (destroyed_boxes : fin n × fin n → Prop)
  (coloring : fin n × fin n → Prop) : Prop :=
  (∀ i, ∀ j1 j2, j1 ≠ j2 → coloring (i, j1) → ¬coloring (i, j2)) ∧ 
  (∀ j, ∀ i1 i2, i1 ≠ i2 → coloring (i1, j) → ¬coloring (i2, j)) ∧ 
  (∀ i j, ¬destroyed_boxes (i, j) ∧ ¬coloring (i, j) → 
          (∃ k, coloring (i, k) ∧ grid (i, j) < grid (i, k)) ∨ 
          (∃ k, coloring (k, j) ∧ grid (i, j) > grid (k, j)))

theorem susann_can_color_boxes (n : ℕ)
  (h_pos : 0 < n)
  (grid : matrix (fin n) (fin n) ℤ)
  (h_distinct_rows : ∀ i j1 j2, j1 ≠ j2 → grid i j1 ≠ grid i j2)
  (h_distinct_cols : ∀ j i1 i2, i1 ≠ i2 → grid i1 j ≠ grid i2 j)
  (destroyed_boxes : fin n × fin n → Prop)
  (h_is_destroyed : is_destroyed n destroyed_boxes)
  : ∃ coloring : fin n × fin n → Prop, can_color_boxes n grid destroyed_boxes coloring :=
sorry

end susann_can_color_boxes_l725_725845


namespace area_of_bounded_region_l725_725434

theorem area_of_bounded_region : 
  let x1 := 2
  let y1 := 2
  let x0 := 0
  let y0 := 0
  in
  let area := (x1 - x0) * (y1 - y0)
  in
  area = 4 :=
by 
  -- definitions for x1, y1, x0, y0
  let x1 := 2
  let y1 := 2
  let x0 := 0
  let y0 := 0
  
  -- let area be the area of the square bounded by these lines
  let area := (x1 - x0) * (y1 - y0)
  
  -- assertion
  have h : area = (2 - 0) * (2 - 0), from rfl,
  
  -- proving the final statement
  show area = 4, by 
    rw [h],
    exact rfl

-- skipped proof step
sorry

end area_of_bounded_region_l725_725434


namespace box_surface_area_l725_725286

theorem box_surface_area (a b c : ℝ) (h₁ : 4 * (a + b + c) = 172) (h₂ : sqrt (a^2 + b^2 + c^2) = 21) :
  2 * (a * b + b * c + c * a) = 1408 := 
sorry

end box_surface_area_l725_725286


namespace tangent_line_at_1_m1_f_increasing_comparison_2017_2016_m_bound_l725_725513

noncomputable section

open Real

def f (x : ℝ) : ℝ := (log x) / x

def g (m x : ℝ) : ℝ := m / x + 1 / 2

def g_prime (x : ℝ) : ℝ := -1 / x^2

def e : ℝ := 2.718281828459045

theorem tangent_line_at_1_m1 : (2 * (1 : ℝ) + 2 * ((3 / 2) : ℝ) - 5 = 0) :=
by simp

theorem f_increasing : ∀ x ∈ Ioi (0 : ℝ), if x < e then (f x)' > 0 else (f x)' < 0 :=
by
  intro x hx
  split_ifs
  . exact (1 - log x) / x^2 > 0,
  . exact (1 - log x) / x^2 < 0

theorem comparison_2017_2016 : real_exp_log 2017 (1 / 2017) < real_exp_log 2016 (1 / 2016) :=
sorry

theorem m_bound (b : ℝ) (x : ℝ) (hb : 1 ≤ b ∧ b ≤ e) (hx : 0 < x ∧ x ≤ e)
    (h : ∀ x, b * f x > g m x) : m < - (e / 2) :=
sorry

end tangent_line_at_1_m1_f_increasing_comparison_2017_2016_m_bound_l725_725513


namespace ant_travel_distance_l725_725333

theorem ant_travel_distance (R : ℝ) (h : ℝ) (C : ℝ) (π : ℝ) 
  (planet_radius : R = 156)
  (alien_height : h = 13)
  (circumference : C = 2 * π * (sqrt ((R + h) ^ 2 - R ^ 2))) : 
  C = 130 * π := 
by {
  rw [planet_radius, alien_height],
  norm_num,
  rw [pow_two, pow_two, add_comm],
  ring
} sorry

end ant_travel_distance_l725_725333


namespace tangent_line_at_origin_l725_725094

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

theorem tangent_line_at_origin (a : ℝ) (h1 : is_even_function (λ x, 3 * x^2 + 2 * a * x + (a - 3))) :
  let f := λ x, x^3 + a * x^2 + (a - 3) * x in
  (∃ m : ℝ, (∀ h2 : m = (f' 0), f 0 = m * 0) ∧ (0, f 0) = (0, 0) ∧ f' 0 = -3) :=
by
  sorry

end tangent_line_at_origin_l725_725094


namespace find_angle_NCB_l725_725901

def triangle_ABC_with_point_N (A B C N : Point) : Prop :=
  ∃ (angle_ABC angle_ACB angle_NAB angle_NBC : ℝ),
    angle_ABC = 50 ∧
    angle_ACB = 20 ∧
    angle_NAB = 40 ∧
    angle_NBC = 30 

theorem find_angle_NCB (A B C N : Point) 
  (h : triangle_ABC_with_point_N A B C N) :
  ∃ (angle_NCB : ℝ), 
  angle_NCB = 10 :=
sorry

end find_angle_NCB_l725_725901


namespace magnitude_z_l725_725271

-- Definitions and conditions

def complex_z (z : ℂ) : Prop :=
  z / ((1 : ℂ) - (complex.I))^2 = (1 + complex.I) / 2

-- The main statement
theorem magnitude_z (z : ℂ) (h : complex_z z) : complex.abs z = real.sqrt 2 :=
sorry

end magnitude_z_l725_725271


namespace interest_percentage_is_correct_l725_725320

def purchase_price : ℝ := 110
def down_payment : ℝ := 10
def monthly_payment : ℝ := 10
def number_of_monthly_payments : ℝ := 12

def total_amount_paid : ℝ := down_payment + number_of_monthly_payments * monthly_payment
def interest_paid : ℝ := total_amount_paid - purchase_price
def interest_percentage : ℝ := (interest_paid / purchase_price) * 100

theorem interest_percentage_is_correct :
  interest_percentage = 18.2 := by
  sorry

end interest_percentage_is_correct_l725_725320


namespace least_of_consecutive_odd_integers_l725_725176

def average_arithmetic_mean (nums : List ℤ) : ℤ :=
  nums.sum / nums.length

theorem least_of_consecutive_odd_integers (l : List ℤ) (h_len : l.length = 16) 
(h_consecutive_odd : ∀ i < 15, l.get i + 2 = l.get (i + 1))
(h_average : average_arithmetic_mean l = 414) : 
  l.get 0 = 399 :=
by
  sorry

end least_of_consecutive_odd_integers_l725_725176


namespace table_properties_l725_725198

theorem table_properties (n : ℕ) (table : Matrix (Fin 2015) (Fin n) ℕ) :
  (∀ i : Fin 2015, ∃ j : Fin n, table i j > 0) →
  (∀ j : Fin n, ∃ i : Fin 2015, table i j > 0) →
  (∀ i : Fin 2015, ∀ j : Fin n, (table i j > 0 → (∑ x, table i x) = (∑ y, table y j))) →
  n = 2015 :=
  sorry

end table_properties_l725_725198


namespace savings_per_person_savings_per_person_correct_l725_725555

def total_years : ℕ := 3
def months_per_year : ℕ := 12
def total_downpayment : ℕ := 108000
def total_months : ℕ := total_years * months_per_year
def joint_monthly_savings : ℕ := total_downpayment / total_months

theorem savings_per_person : (joint_monthly_savings / 2) = 1500 :=
by
  -- Definitions and intermediate steps can be used here for Lean to verify
  sorry

-- To make it thorough, let's explicitly define the intermediate steps
lemma total_months_correct : total_months = 36 :=
by
  -- total_months = total_years * months_per_year = 3 * 12 = 36
  sorry

lemma joint_monthly_savings_correct : joint_monthly_savings = 3000 :=
by
  -- joint_monthly_savings = total_downpayment / total_months = 108000 / 36 = 3000
  sorry

theorem savings_per_person_correct :
  (total_downpayment / total_months / 2) = 1500 :=
by
  rw [←joint_monthly_savings_correct, ←total_months_correct]
  exact savings_per_person
  sorry

end savings_per_person_savings_per_person_correct_l725_725555


namespace functional_equation_solution_l725_725810

theorem functional_equation_solution (f : ℚ → ℚ)
  (H : ∀ x y : ℚ, f (x + y) + f (x - y) = 2 * f x + 2 * f y) :
  ∃ a : ℚ, ∀ x : ℚ, f x = a * x^2 :=
by
  sorry

end functional_equation_solution_l725_725810


namespace total_players_l725_725731

theorem total_players (kabadi kho_kho_only both_games : ℕ) (h_kabadi : kabadi = 10) 
  (h_kho_kho_only : kho_kho_only = 25) (h_both_games : both_games = 5) : 
  kabadi + kho_kho_only - both_games = 30 := 
by
  rw [h_kabadi, h_kho_kho_only, h_both_games]
  norm_num
  sorry

end total_players_l725_725731


namespace three_f_ln2_gt_two_f_ln3_l725_725102

variable (f : ℝ → ℝ)
variable (h_deriv : ∀ x : ℝ, f(x) > deriv f x)

theorem three_f_ln2_gt_two_f_ln3 : 3 * f (Real.log 2) > 2 * f (Real.log 3) :=
  sorry

end three_f_ln2_gt_two_f_ln3_l725_725102


namespace lattice_points_count_l725_725357

noncomputable def countLatticePoints (boundary1 boundary2 : ℤ × ℤ → Prop) : ℕ :=
  Finset.card (Finset.filter (λ p, boundary1 p ∨ boundary2 p)
    (Finset.univ.filter (λ p : ℤ × ℤ, p.1 * p.1 + p.2 * p.2 <= 64)))

def boundary1 (p : ℤ × ℤ) : Prop :=
  p.2 = 2 * Int.natAbs p.1

def boundary2 (p : ℤ × ℤ) : Prop :=
  p.2 = - p.1 * p.1 + 8

theorem lattice_points_count : countLatticePoints boundary1 boundary2 = 23 :=
sorry

end lattice_points_count_l725_725357


namespace parabola_solution_l725_725392

noncomputable def parabola_coefficients (a b c : ℝ) : Prop :=
  (6 : ℝ) = a * (5 : ℝ)^2 + b * (5 : ℝ) + c ∧
  0 = a * (3 : ℝ)^2 + b * (3 : ℝ) + c

theorem parabola_solution :
  ∃ (a b c : ℝ), parabola_coefficients a b c ∧ (a + b + c = 6) :=
by {
  -- definitions and constraints based on problem conditions
  sorry
}

end parabola_solution_l725_725392


namespace magnitude_of_sum_l725_725891

-- Define the vectors and the condition of orthogonality
def vector_a : ℝ × ℝ := (2, 1)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -6)

-- Assuming orthogonality of the vectors a and b
def orthogonal (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0
def z : ℝ := 3
def b := vector_b z

-- Define the length of a vector
def vector_length (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 * v.1 + v.2 * v.2)

-- Prove that the magnitude of a + b is 5√2
theorem magnitude_of_sum : vector_length (vector_a.1 + b.1, vector_a.2 + b.2) = 5 * real.sqrt 2 :=
by 
  sorry

end magnitude_of_sum_l725_725891


namespace original_number_l725_725360

theorem original_number (x : ℕ) : 3 * (2 * x + 5) = 117 → x = 17 :=
by
  intro h,
  sorry

end original_number_l725_725360


namespace triangle_area_ratio_l725_725928

theorem triangle_area_ratio (A B C X : Type) {AB BC AC BX AX : ℝ} (h1 : AB = 50)
  (h2 : BC = 36) (h3 : AC = 42) (h4 : perpendicular CX AB): 
  (BCX.area / ACX.area) = 6/7 :=
sorry

end triangle_area_ratio_l725_725928


namespace relationship_abc_l725_725095

noncomputable def a : ℝ := 2011^0.6
noncomputable def b : ℝ := 0.6^2011
noncomputable def c : ℝ := Real.log 2011 / Real.log 0.6

theorem relationship_abc : a > b ∧ b > c :=
by
  -- Conditions provided from the problem
  have ha : a = 2011 ^ 0.6 := rfl
  have hb : b = 0.6 ^ 2011 := rfl
  have hc : c = Real.log 2011 / Real.log 0.6 := rfl
  
  -- Required to prove a > b and b > c
  sorry

end relationship_abc_l725_725095


namespace ellipse_equation_and_properties_l725_725924

/-- Given conditions for the ellipse C --/
theorem ellipse_equation_and_properties
  (a b : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (eccentricity : a^2 - b^2 = (a^2) * (3 / 4))
  (point_on_ellipse : (sqrt 3)^2 / a^2 + (1 / 2)^2 / b^2 = 1) :

  /-- Part (1): Prove the equation of ellipse C --/
  (∃ a b : ℝ, (a = 2 ∧ b = 1) ∧ ∀ x y, (x^2 / 4) + y^2 = 1) ∧
  
  /-- Part (2)(i): Prove the ratio |OQ| / |OP| is 2 --/
  (∀ P Q : ℝ × ℝ, (P = (sqrt 3, 1 / 2)) → (Q = (-(sqrt 3) * 2, -(1 / 2) * 2)) → |Q| / |P| = 2) ∧
  
  /-- Part (2)(ii): Prove the maximum area of ΔABQ is 6√3 --/
  (∀ A B Q : ℝ × ℝ, 
    (∃ k m : ℝ, k * AX + m * AY = 0) 
    → (AX^2 / 16) + AY^2 / 4 = 1 
    → (OQ = (0, OQy))
    → (4 * sin(π/3) = 6√3)),
 sorry

end ellipse_equation_and_properties_l725_725924


namespace derivative_bound_l725_725576

noncomputable def f (x : ℝ) : ℝ := if x ≠ 0 then (Real.sin x) / x else 1

theorem derivative_bound (x : ℝ) (n : ℕ) (h_x : x > 0) (h_n : Nat.isStrictPos n) : 
  |(deriv^[n] f) x| < 1 / (n + 1) :=
sorry

end derivative_bound_l725_725576


namespace money_problem_solution_l725_725980

theorem money_problem_solution (a b : ℝ) (h1 : 7 * a + b < 100) (h2 : 4 * a - b = 40) (h3 : b = 0.5 * a) : 
  a = 80 / 7 ∧ b = 40 / 7 :=
by
  sorry

end money_problem_solution_l725_725980


namespace quadrilateral_area_l725_725203

-- Definitions of the lines
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 12 = 0
def line2 (x y : ℝ) : Prop := 5 * x - 4 * y - 10 = 0
def line3 (x : ℝ) : Prop := x = 3
def line4 (y : ℝ) : Prop := y = 1

-- Definition of intersection points
def intersection1 : ℝ × ℝ := (3, 0.75) -- line1 and line3
def intersection2 : ℝ × ℝ := (3, 1) -- line2 and line3
def intersection3 : ℝ × ℝ := intersection2 -- line3 and line4
def intersection4 : ℝ × ℝ := (8 / 3, 1) -- line1 and line4

-- The shoelace formula
def shoelace_formula (pts : list (ℝ × ℝ)) : ℝ :=
  0.5 * ((pts.head.1 * pts.tail.head.2 + pts.tail.head.1 * pts.tail.tail.head.2 + pts.tail.tail.head.1 * pts.tail.tail.tail.head.2 + pts.tail.tail.tail.head.1 * pts.head.2)
        - (pts.head.2 * pts.tail.head.1 + pts.tail.head.2 * pts.tail.tail.head.1 + pts.tail.tail.head.2 * pts.tail.tail.tail.head.1 + pts.tail.tail.tail.head.2 * pts.head.1)).abs

-- The list of intersection points
def points : list (ℝ × ℝ) := [intersection1, intersection2, intersection3, intersection4]

-- Proving the area
theorem quadrilateral_area :
  shoelace_formula points = 0.125 :=
by
  sorry

end quadrilateral_area_l725_725203


namespace sqrt_polynomial_expression_l725_725801

theorem sqrt_polynomial_expression :
  let a := Real.sin (π / 9)
  let b := Real.sin (2 * π / 9)
  let c := Real.sin (4 * π / 9)
  (512 * (a ^ 2) * (b ^ 2) * (c ^ 2) - 1152 * (a ^ 2 * b ^ 2 + b ^ 2 * c ^ 2 + c ^ 2 * a ^ 2) + 576 * (a ^ 2 + b ^ 2 + c ^ 2) - 27 = 0) →
  Real.sqrt (((3 - a ^ 2) * (3 - b ^ 2) * (3 - c ^ 2))) = 96 / 8 :=
by
sory

end sqrt_polynomial_expression_l725_725801


namespace herman_days_per_week_l725_725150

-- Defining the given conditions as Lean definitions
def total_meals : ℕ := 4
def cost_per_meal : ℕ := 4
def total_weeks : ℕ := 16
def total_cost : ℕ := 1280

-- Calculating derived facts based on given conditions
def cost_per_day : ℕ := total_meals * cost_per_meal
def cost_per_week : ℕ := total_cost / total_weeks

-- Our main theorem that states Herman buys breakfast combos 5 days per week
theorem herman_days_per_week : cost_per_week / cost_per_day = 5 :=
by
  -- Skipping the proof
  sorry

end herman_days_per_week_l725_725150


namespace ellipse_equation_and_constant_area_l725_725847

-- Definitions of terms used in the problem
variables (a b x y : ℝ)
def ellipse_pred (a b x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1 ∧ a > b ∧ b > 0
def point_on_ellipse (a b : ℝ) (x y : ℝ) : Prop := ellipse_pred a b x y ∧ (x = 2 ∧ y = sqrt 2)
def eccentricity (a b : ℝ) : ℝ := sqrt (1 - (b^2 / a^2))

-- The statement of the math problem in Lean 4
theorem ellipse_equation_and_constant_area (a b : ℝ) :
  point_on_ellipse a b 2 (sqrt 2) ∧ eccentricity a b = sqrt 2 / 2 →
  (\sum_{x,y : ℝ} ellipse_pred a b x y ↔ a^2 = 8 ∧ b^2 = 4) ∧
  (∀ (M N : ℝ → Prop), ∃ (k_AP k_BP k_OM k_ON : ℝ),
    (M ≠ N ∧ k_AP k_BP ≠ 0 ∧ k_OM k_ON = -1/2) →
    let t := (k_AP + k_BP) in
    let y1 := 0 in let y2 := 0 in
    let S := (1 / 2) * abs t * abs (y1 - y2) in
    ∀ t y1 y2, S = 2 * sqrt 2) :=
sorry

end ellipse_equation_and_constant_area_l725_725847


namespace area_bounded_by_graphs_eq_4_l725_725418

theorem area_bounded_by_graphs_eq_4 :
  let r₁ (θ : ℝ) := 2 / (cos θ)
  let r₂ (θ : ℝ) := 2 / (sin θ)
  ∀ (θ₁ θ₂ : ℝ) (x ℝ: ℝ), 0 ≤ θ₁ ∧ θ₁ ≤ π/2 ∧ 0 ≤ θ₂ ∧ θ₂ ≤ π/2 ∧
  x = r₁ θ₁ ∧ y = r₂ θ₂ →
  area (bounded_region r₁ r₂ 0 0) = 4 := by
  sorry

end area_bounded_by_graphs_eq_4_l725_725418


namespace equilateral_triangles_and_center_line_l725_725323

-- Definitions of the basic geometric constructs and conditions
structure Triangle (α : Type) [MetricSpace α] := 
  (A B C : α)
  (non_equilateral : A ≠ B ∧ B ≠ C ∧ C ≠ A)

-- Basic definition of a point not coinciding with triangle vertices
structure PointInPlane (α : Type) [MetricSpace α] (T : Triangle α) := 
  (P : α)
  (P_not_vertex : P ≠ T.A ∧ P ≠ T.B ∧ P ≠ T.C)

-- Given a triangle and a point P, define the second intersection points with the circumcircle
noncomputable def second_intersection_points {α : Type} [MetricSpace α] (T : Triangle α) (P : PointInPlane α T) :=
  (A_P B_P C_P : α) -- Defining points here

-- Theorem statement 
theorem equilateral_triangles_and_center_line {α : Type} [MetricSpace α] (T : Triangle α) (P : PointInPlane α T) :
  ∃ (P Q : α),
    (let ⟨A_P, B_P, C_P⟩ := second_intersection_points T P in
      is_equilateral (⟨A_P, B_P, C_P⟩ : Triangle α)) ∧
    (let ⟨A_Q, B_Q, C_Q⟩ := second_intersection_points T Q in
      is_equilateral (⟨A_Q, B_Q, C_Q⟩ : Triangle α)) ∧
    passes_through_center (P, Q, circumcenter T) :=
sorry

end equilateral_triangles_and_center_line_l725_725323


namespace correctly_subtracted_value_l725_725732

theorem correctly_subtracted_value (x : ℤ) (h1 : 122 = x - 64) : 
  x - 46 = 140 :=
by
  -- Proof goes here
  sorry

end correctly_subtracted_value_l725_725732


namespace at_least_six_heads_in_10_flips_is_129_over_1024_l725_725346

def fair_coin_flip (n : ℕ) (prob_heads prob_tails : ℚ) : Prop :=
  (prob_heads = 1/2 ∧ prob_tails = 1/2)

noncomputable def at_least_six_consecutive_heads_probability (n : ℕ) : ℚ :=
  if n = 10 then 129 / 1024 else 0  -- this is specific to 10 flips and should be defined based on actual calculation for different n
  
theorem at_least_six_heads_in_10_flips_is_129_over_1024 :
  fair_coin_flip 10 (1/2) (1/2) →
  at_least_six_consecutive_heads_probability 10 = 129 / 1024 :=
by
  intros
  sorry

end at_least_six_heads_in_10_flips_is_129_over_1024_l725_725346


namespace vector_at_t_neg1_vector_3_604_neg6_not_on_line_l725_725012

open Real

noncomputable def a : ℝ × ℝ × ℝ := (0, 4, 1)
noncomputable def d : ℝ × ℝ × ℝ := (-1, -4, -4)
noncomputable def line (t : ℝ) : ℝ × ℝ × ℝ := (a.1 + t * d.1, a.2 + t * d.2, a.3 + t * d.3)

theorem vector_at_t_neg1 :
  line (-1) = (1, 8, 5) :=
  by
  sorry

theorem vector_3_604_neg6_not_on_line :
  ∀ t : ℝ, line t ≠ (3, 604, -6) :=
  by
  sorry

end vector_at_t_neg1_vector_3_604_neg6_not_on_line_l725_725012


namespace problem_1_problem_2_l725_725944

open Set Real

noncomputable def A : Set ℝ := {x | x^2 - 3 * x - 18 ≤ 0}

noncomputable def B (m : ℝ) : Set ℝ := {x | m - 8 ≤ x ∧ x ≤ m + 4}

theorem problem_1 : (m = 3) → ((compl A) ∩ (B m) = {x | (-5 ≤ x ∧ x < -3) ∨ (6 < x ∧ x ≤ 7)}) :=
by
  sorry

theorem problem_2 : (A ∩ (B m) = A) → (2 ≤ m ∧ m ≤ 5) :=
by
  sorry

end problem_1_problem_2_l725_725944


namespace percentage_error_calculation_l725_725761

theorem percentage_error_calculation (x : ℝ) :
  let correct_value := x * (5 / 3)
  let incorrect_value := x * (3 / 5)
  let difference := correct_value - incorrect_value
  let percentage_error := (difference / correct_value) * 100
  percentage_error = 64 := 
by
  let correct_value := x * (5 / 3)
  let incorrect_value := x * (3 / 5)
  let difference := correct_value - incorrect_value
  let percentage_error := (difference / correct_value) * 100
  sorry

end percentage_error_calculation_l725_725761


namespace total_games_l725_725723

variable (G R : ℕ)

-- Condition 1: Team won 75 percent of its first 100 games
def cond1 : Prop := (0.75 * 100 = 75)

-- Condition 2: Team won 50 percent of its remaining games
def cond2 : Prop := (0.50 * R = 0.50 * R)

-- Condition 3: Team won 70 percent of its games for the entire season
def cond3 : Prop := (0.70 * G = 75 + 0.50 * R)

-- Equation linking total games and remaining games
def cond4 : Prop := (G = 100 + R)

-- The proof problem: Prove G = 125 given the conditions above
theorem total_games
  (h1 : cond1)
  (h2 : cond2)
  (h3 : cond3)
  (h4 : cond4) : G = 125 := by
  sorry

end total_games_l725_725723


namespace number_of_polynomials_l725_725154

def satisfies_condition (f : ℚ[X]) : Prop :=
  ∀ x : ℚ, f.eval (x^2 + 1) = (f.eval x)^2 + 1 ∧ f.eval (f.eval x + 1) = (f.eval x)^2 + 1

theorem number_of_polynomials : ∃! f : ℚ[X], degree f ≥ 2 ∧ satisfies_condition f :=
sorry

end number_of_polynomials_l725_725154


namespace find_principal_amount_l725_725765

def SI : ℝ := 7434.50
def R : ℝ := 3.5
def T : ℕ := 6
def expected_P : ℝ := 35402.38

theorem find_principal_amount (SI : ℝ) (R : ℝ) (T : ℕ) (P : ℝ) : 
  P = (SI * 100) / (R * T) := 
begin
  have h1 : P = (7434.50 * 100) / (3.5 * 6),
  { sorry },
  exact h1
end

end find_principal_amount_l725_725765


namespace invoice_total_correct_l725_725570

def cost_of_sofa : ℕ := 1250
def cost_of_one_armchair : ℕ := 425
def cost_of_coffee_table : ℕ := 330
def cost_of_two_armchairs : ℕ := 2 * cost_of_one_armchair
def total_invoice_amount : ℕ := cost_of_sofa + cost_of_two_armchairs + cost_of_coffee_table

theorem invoice_total_correct : total_invoice_amount = 2430 :=
by
  have h1 : cost_of_two_armchairs = 850 := by
    unfold cost_of_two_armchairs cost_of_one_armchair
    norm_num
  have h2 : total_invoice_amount = 1250 + 850 + 330 := by
    unfold total_invoice_amount cost_of_sofa cost_of_two_armchairs cost_of_coffee_table
    exact rfl
  rw [h1] at h2
  norm_num at h2
  exact h2

end invoice_total_correct_l725_725570


namespace find_wrongly_written_height_l725_725653

noncomputable def wrongly_written_height (n : ℕ) (average_wrong : ℝ) (h_actual : ℝ) (average_correct : ℝ) :=
  let total_wrong := n * average_wrong
  let total_correct_wrong := total_wrong - x + h_actual
  let total_correct := n * average_correct
  total_correct = total_correct_wrong → x = 176

theorem find_wrongly_written_height :
  wrongly_written_height 35 185 106 183 = 176 :=
begin
  sorry
end

end find_wrongly_written_height_l725_725653


namespace jacob_walked_8_miles_l725_725933

theorem jacob_walked_8_miles (rate time : ℝ) (h_rate : rate = 4) (h_time : time = 2) :
  rate * time = 8 := by
  -- conditions
  have hr : rate = 4 := h_rate
  have ht : time = 2 := h_time
  -- problem
  sorry

end jacob_walked_8_miles_l725_725933


namespace absolute_value_constant_term_l725_725027

theorem absolute_value_constant_term :
  ∃ (p q : ℤ[x]), monic p ∧ monic q ∧ degree p = 3 ∧ degree q = 5 ∧
  (∀ a b: ℤ, polynomial.coeff p 0 = -a ∧ polynomial.coeff q 0 = -b ∧ a = b) ∧
  p * q = X^8 + 2 * X^7 + X^6 + 2 * X^5 + 3 * X^4 + 2 * X^3 + X^2 + 2 * X + 9 →
  |polynomial.coeff q 0| = 3 :=
by
  sorry

end absolute_value_constant_term_l725_725027


namespace parabola_directrix_is_y_eq_2_l725_725661

-- Define the given parabola equation
def parabola_eq (x : ℝ) : ℝ :=
  - (1/8) * x^2

-- Define the property that the equation represents a parabola whose directrix is to be proven
def directrix_eq (y : ℝ) : Prop :=
  y = 2

-- The main theorem stating the directrix of the given parabola
theorem parabola_directrix_is_y_eq_2 : ∀ x : ℝ, directrix_eq (- (x^2 / 8)) :=
by
  sorry

end parabola_directrix_is_y_eq_2_l725_725661


namespace min_function_value_in_domain_l725_725814

theorem min_function_value_in_domain :
  ∃ (x y : ℝ), (1 / 3 ≤ x ∧ x ≤ 3 / 5) ∧ (1 / 4 ≤ y ∧ y ≤ 1 / 2) ∧ (∀ (x y : ℝ), (1 / 3 ≤ x ∧ x ≤ 3 / 5) ∧ (1 / 4 ≤ y ∧ y ≤ 1 / 2) → (xy / (x^2 + y^2)) ≥ (60 / 169)) :=
sorry

end min_function_value_in_domain_l725_725814


namespace hyperbola_eccentricity_l725_725140

theorem hyperbola_eccentricity (m : ℤ) (h : m^2 - 4 > 0) :
  let a := m
  let b := real.sqrt (m^2 - 4)
  real.sqrt (1 + (b^2 / a^2)) = 2 :=
by
  -- Proof goes here
  sorry

end hyperbola_eccentricity_l725_725140


namespace investment_payoff_period_l725_725991

noncomputable theory

def initialInvestment (systemUnitCost : ℕ) (graphicsCardCost : ℕ) (numGraphicsCards : ℕ) : ℕ :=
  systemUnitCost + (numGraphicsCards * graphicsCardCost)

def dailyRevenue (ethPerCardPerDay : ℝ) (numGraphicsCards : ℕ) (ethToRubRate : ℝ) : ℝ :=
  (ethPerCardPerDay * numGraphicsCards) * ethToRubRate

def dailyEnergyCost (systemUnitConsumption : ℕ) (graphicsCardConsumption : ℕ) (numGraphicsCards : ℕ) (electricityCostPerKWh : ℝ) : ℝ :=
  let totalWattage := systemUnitConsumption + (graphicsCardConsumption * numGraphicsCards)
  let dailyKWh := (totalWattage / 1000.0) * 24
  dailyKWh * electricityCostPerKWh

def netDailyProfit (dailyRevenue : ℝ) (dailyEnergyCost : ℝ) : ℝ :=
  dailyRevenue - dailyEnergyCost

def paybackPeriod (initialInvestment : ℕ) (netDailyProfit : ℝ) : ℝ :=
  initialInvestment / netDailyProfit

theorem investment_payoff_period
    (systemUnitCost : ℕ := 9499)
    (graphicsCardCost : ℕ := 20990)
    (numGraphicsCards : ℕ := 2)
    (ethPerCardPerDay : ℝ := 0.00630)
    (ethToRubRate : ℝ := 27790.37)
    (systemUnitConsumption : ℕ := 120)
    (graphicsCardConsumption : ℕ := 185)
    (electricityCostPerKWh : ℝ := 5.38)
    : paybackPeriod (initialInvestment systemUnitCost graphicsCardCost numGraphicsCards)
                    (netDailyProfit (dailyRevenue ethPerCardPerDay numGraphicsCards ethToRubRate)
                                    (dailyEnergyCost systemUnitConsumption graphicsCardConsumption numGraphicsCards electricityCostPerKWh)) ≈ 179 := by
  sorry

end investment_payoff_period_l725_725991


namespace tetrahedron_in_sphere_l725_725063

theorem tetrahedron_in_sphere :
  (∀ face, ∃ circle, (face ⊆ circle) ∧ (radius circle = 1)) →
  ∃ sphere, (tetrahedron ⊆ sphere) ∧ (radius sphere = 3 / (2 * sqrt 2)) :=
by
  intros h
  sorry

end tetrahedron_in_sphere_l725_725063


namespace part1_part2_l725_725876

theorem part1 (m : ℝ) :
  ∀ x : ℝ, x^2 + ( (2 * m - 1) : ℝ) * x + m^2 = 0 → m ≤ 1 / 4 :=
sorry

theorem part2 (m : ℝ) 
  (h : ∀ x1 x2 : ℝ, (x1^2 + (2*m -1)*x1 + m^2 = 0) ∧ (x2^2 + (2*m -1)*x2 + m^2 = 0) ∧ (x1*x2 + x1 + x2 = 4)) :
    m = -1 :=
sorry

end part1_part2_l725_725876


namespace correct_group_l725_725029

def atomic_number (element : String) : Nat :=
  match element with
  | "Be" => 4
  | "C" => 6
  | "B" => 5
  | "Cl" => 17
  | "O" => 8
  | "Li" => 3
  | "Al" => 13
  | "S" => 16
  | "Si" => 14
  | "Mg" => 12
  | _ => 0

def is_descending (lst : List Nat) : Bool :=
  match lst with
  | [] => true
  | [x] => true
  | x :: y :: xs => if x > y then is_descending (y :: xs) else false

theorem correct_group : is_descending [atomic_number "Cl", atomic_number "O", atomic_number "Li"] = true ∧
                        is_descending [atomic_number "Be", atomic_number "C", atomic_number "B"] = false ∧
                        is_descending [atomic_number "Al", atomic_number "S", atomic_number "Si"] = false ∧
                        is_descending [atomic_number "C", atomic_number "S", atomic_number "Mg"] = false :=
by
  -- Prove the given theorem based on the atomic number function and is_descending condition
  sorry

end correct_group_l725_725029


namespace attic_junk_percentage_l725_725239

noncomputable def percentage_junk (T : ℕ) (nj : ℕ) : ℕ := 
  (nj * 100) / T

theorem attic_junk_percentage : 
  ∃ (T : ℕ), 0.20 * T = 8 ∧ percentage_junk T 28 = 70 := 
by 
  sorry

end attic_junk_percentage_l725_725239


namespace isosceles_triangle_perimeter_l725_725915

-- Define the isosceles triangle with given sides
structure IsoscelesTriangle where
  a b : ℝ
  side1_eq_side2 : a = b

-- Define the sides of the triangle
def triangle1 : IsoscelesTriangle := {
  a := 4,
  b := 4,
  side1_eq_side2 := rfl
}

def triangle2 : IsoscelesTriangle := {
  a := 3,
  b := 3,
  side1_eq_side2 := rfl
}

-- Statement: Proving the perimeter can be either 10 or 11
theorem isosceles_triangle_perimeter (t1 t2 : IsoscelesTriangle) (side3_1 side3_2 : ℝ) :
  (t1.a + t1.b + side3_1 = 11 ∨ t2.a + t2.b + side3_2 = 10) :=
by
  -- t1: {a, b} with both sides equal to 4, and side3_1 equal to 3
  -- t2: {a, b} with both sides equal to 3, and side3_2 equal to 4
  have h1 : t1.a + t1.b + side3_1 = 4 + 4 + 3, by linarith,
  have h2 : t2.a + t2.b + side3_2 = 3 + 3 + 4, by linarith,
  have h3 : 4 + 4 + 3 = 11, by linarith,
  have h4 : 3 + 3 + 4 = 10, by linarith,
  left,
  exact h3,
  right,
  exact h4,
  sorry

end isosceles_triangle_perimeter_l725_725915


namespace probability_product_even_and_greater_than_12_l725_725643

theorem probability_product_even_and_greater_than_12 :
  let balls := {1, 2, 3, 4, 5, 6}
  let counts := (balls.toFinset.product balls.toFinset).filter (λ (n : ℕ × ℕ), (n.1 * n.2) % 2 = 0 ∧ n.1 * n.2 > 12) |>.card
  (counts : ℚ) / (balls.card ^ 2 : ℚ) = 1 / 4 :=
by
  sorry

end probability_product_even_and_greater_than_12_l725_725643


namespace first_pipe_time_l725_725753

noncomputable def pool_filling_time (T : ℝ) : Prop :=
  (1 / T + 1 / 12 = 1 / 4.8) → (T = 8)

theorem first_pipe_time :
  ∃ T : ℝ, pool_filling_time T := by
  use 8
  sorry

end first_pipe_time_l725_725753


namespace smallest_n_with_conditions_l725_725082

def num_divisors (n : ℕ) : ℕ :=
  (List.range (n + 1)).count (λ d, n % d = 0)

theorem smallest_n_with_conditions :
  ∃ n : ℕ, (num_divisors n = 144 ∧ ∃ (m : ℕ), ∀ k : ℕ, k < 10 → (m + k) ∣ n) →
  n = 110880 :=
by
  sorry

end smallest_n_with_conditions_l725_725082


namespace proposition_false_at_6_l725_725362

variable (P : ℕ → Prop)

theorem proposition_false_at_6 (h1 : ∀ k : ℕ, 0 < k → P k → P (k + 1)) (h2 : ¬P 7): ¬P 6 :=
by
  sorry

end proposition_false_at_6_l725_725362


namespace find_k_l725_725472

theorem find_k (k : ℚ) : (∀ x y : ℚ, (x, y) = (2, 1) → 3 * k * x - k = -4 * y - 2) → k = -(6 / 5) :=
by
  intro h
  have key := h 2 1 rfl
  have : 3 * k * 2 - k = -4 * 1 - 2 := key
  linarith

end find_k_l725_725472


namespace sum_of_remainders_l725_725785

-- Definitions of the given problem
def a : ℕ := 1234567
def b : ℕ := 123

-- First remainder calculation
def r1 : ℕ := a % b

-- Second remainder calculation with the power
def r2 : ℕ := (2 ^ r1) % b

-- The proof statement
theorem sum_of_remainders : r1 + r2 = 29 := by
  sorry

end sum_of_remainders_l725_725785


namespace simson_lines_equal_angle_l725_725252

noncomputable def simson_angle (A B C P Q : Type) [circle ABC] [point_on_circle P ABC] [point_on_circle Q ABC]
  (P1 P2 Q1 Q2 : Type)
  [foot_perpendicular P AB AC P1 P2] [foot_perpendicular Q AB AC Q1 Q2] : angle between :=
  sorry

theorem simson_lines_equal_angle (A B C P Q : Type) [triangle ABC] [circle ABC] [point_on_circle P ABC]
  [point_on_circle Q ABC] (P1 P2 Q1 Q2 : Type)
  [foot_perpendicular P AB AC P1 P2] [foot_perpendicular Q AB AC Q1 Q2]
  (φ : angle between (P1P2) (Q1Q2)) :
  φ = function_of_chord PQ :=
  sorry

end simson_lines_equal_angle_l725_725252


namespace min_pos_period_sin3x_pi4_l725_725673

theorem min_pos_period_sin3x_pi4 : 
  (∀ x : ℝ, f x = Real.sin (3 * x + Real.pi / 4)) → 
  (∃ T > 0, T = 2 * Real.pi / 3 ∧ f (x + T) = f x) :=
by
  let f := λ x : ℝ, Real.sin (3 * x + Real.pi / 4)
  let ω := 3
  let T := 2 * Real.pi / ω
  sorry

end min_pos_period_sin3x_pi4_l725_725673


namespace points_of_contact_coplanar_l725_725696

open EuclideanGeometry

noncomputable def Hemisphere : Type := sorry
noncomputable def Sphere : Type := sorry

variables (vase : Hemisphere) (orange grapefruit : Sphere)
variables (G A₁ A₂ A₃ A₄ : Point)
variables (K₁ K₂ K₃ K₄ : Point)
variables (v g a : ℝ)

-- Conditions
axiom vase_is_hemisphere_closed_with_flat_lid (V : Point) : True
axiom four_identical_oranges_touch_vase :
  dist A₁ A₂ = dist A₂ A₃ ∧ 
  dist A₂ A₃ = dist A₃ A₄ ∧ 
  dist A₃ A₄ = dist A₁ A₄ ∧ 
  ∃ P₁ P₂ P₃ P₄ : Point, 
  ∀ i, 1 ≤ i ∧ i ≤ 4 → dist (A i) (P i) = a ∧ dist (A i) V = v - a
axiom grapefruit_touches_four_oranges :
  ∀ i, 1 ≤ i ∧ i ≤ 4 → dist G (A i) = g + a ∧
         dist G (K i) = g ∧ 
         dist (K i) (A i) = a
axiom all_fruits_are_spheres :
  ∀ i, 1 ≤ i ∧ i ≤ 4 → IsSphere (A i) a ∧ IsSphere G g

theorem points_of_contact_coplanar :
  ∃ plane : Plane, ∀ i, 1 ≤ i ∧ i ≤ 4 → (K i ∈ plane)
  sorry

end points_of_contact_coplanar_l725_725696


namespace moles_C2H6_for_HCl_l725_725466

theorem moles_C2H6_for_HCl 
  (form_HCl : ℕ)
  (moles_Cl2 : ℕ)
  (reaction : ℕ) : 
  (6 * (reaction * moles_Cl2)) = form_HCl * (6 * reaction) :=
by
  -- The necessary proof steps will go here
  sorry

end moles_C2H6_for_HCl_l725_725466


namespace cos_value_l725_725490

variables {α : ℝ}

-- Given condition
axiom sin_condition : sin(5 * Real.pi / 2 + α) = 1 / 5

-- Goal to prove
theorem cos_value : cos α = 1 / 5 :=
by
  sorry

end cos_value_l725_725490


namespace table_properties_l725_725197

theorem table_properties (n : ℕ) (table : Matrix (Fin 2015) (Fin n) ℕ) :
  (∀ i : Fin 2015, ∃ j : Fin n, table i j > 0) →
  (∀ j : Fin n, ∃ i : Fin 2015, table i j > 0) →
  (∀ i : Fin 2015, ∀ j : Fin n, (table i j > 0 → (∑ x, table i x) = (∑ y, table y j))) →
  n = 2015 :=
  sorry

end table_properties_l725_725197


namespace sqrt_frac_add_l725_725385

theorem sqrt_frac_add : sqrt ((1 / 8) + (1 / 18)) = sqrt 26 / 12 := 
by sorry

end sqrt_frac_add_l725_725385


namespace lcm_36_100_is_900_l725_725454

def prime_factors_36 : ℕ → Prop := 
  λ n, n = 36 → (2^2 * 3^2)

def prime_factors_100 : ℕ → Prop := 
  λ n, n = 100 → (2^2 * 5^2)

def lcm_36_100 := lcm 36 100

theorem lcm_36_100_is_900 : lcm_36_100 = 900 :=
by {
  sorry,
}

end lcm_36_100_is_900_l725_725454


namespace box_surface_area_l725_725988

theorem box_surface_area (w l s tab : ℕ):
  w = 40 → l = 60 → s = 8 → tab = 2 →
  (40 * 60 - 4 * 8 * 8 + 2 * (2 * (60 - 2 * 8) + 2 * (40 - 2 * 8))) = 2416 :=
by
  intros _ _ _ _
  sorry

end box_surface_area_l725_725988


namespace gray_eyed_brunettes_l725_725539

-- Given conditions
def total_students : ℕ := 60
def brunettes : ℕ := 35
def green_eyed_blondes : ℕ := 20
def gray_eyed_total : ℕ := 25

-- Conclude that the number of gray-eyed brunettes is 20
theorem gray_eyed_brunettes :
    (gray_eyed_total - (total_students - brunettes - green_eyed_blondes)) = 20 := by
    sorry

end gray_eyed_brunettes_l725_725539


namespace angle_B_value_l725_725498

noncomputable def degree_a (A : ℝ) : Prop := A = 30 ∨ A = 60

noncomputable def degree_b (A B : ℝ) : Prop := B = 3 * A - 60

theorem angle_B_value (A B : ℝ) 
  (h1 : B = 3 * A - 60)
  (h2 : A = 30 ∨ A = 60) :
  B = 30 ∨ B = 120 :=
by
  sorry

end angle_B_value_l725_725498


namespace sarah_correct_answer_percentage_l725_725639

theorem sarah_correct_answer_percentage
  (q1 q2 q3 : ℕ)   -- Number of questions in the first, second, and third tests.
  (p1 p2 p3 : ℕ → ℝ)   -- Percentages of questions Sarah got right in the first, second, and third tests.
  (m : ℕ)   -- Number of calculation mistakes:
  (h_q1 : q1 = 30) (h_q2 : q2 = 20) (h_q3 : q3 = 50)
  (h_p1 : p1 q1 = 0.85) (h_p2 : p2 q2 = 0.75) (h_p3 : p3 q3 = 0.90)
  (h_m : m = 3) :
  ∃ pct_correct : ℝ, pct_correct = 83 :=
by
  sorry

end sarah_correct_answer_percentage_l725_725639


namespace eval_abs_expr_l725_725406

-- Define the complex number ω
def ω : ℂ := 7 + 3 * complex.i

-- Define the expression to be evaluated 
def expr : ℂ := ω^2 + 8 * ω + 98

-- Prove that the absolute value of the expression equals √41605
theorem eval_abs_expr : abs expr = real.sqrt 41605 := 
by
-- Proof goes here
sorry

end eval_abs_expr_l725_725406


namespace slip_4_5_in_box_R_l725_725938

noncomputable def slip_distribution (numbers : List ℝ) (boxes : Finset Nat) (distribution : Finset Nat -> List ℝ) : Prop :=
  let sum_boxes := boxes.toList.map (λ b => (distribution b).sum)
  let sums := sum_boxes.to_list
  sums.sorted = List.range (List.length sums) + (sums.min' sorry)

def distribution : String :=
  "John has sixteen slips of paper with numbers written on them: 1, 1.5, 1.5, 2, 2, 2, 2.5, 2.5, 3, 3, 3.5, 3.5, " ++
  "4, 4.5, 5, and 5.5. He decides to distribute these into six boxes labeled P, Q, R, S, T, U. Each box should " ++
  "have the sum of numbers on slips that is an integer. He desires the integers to be consecutive starting from box " ++
  "P to box U. A slip with 1 goes into box U and a slip with 2 goes into box Q. Show that given these conditions, " ++
  "the slip with 4.5 should go into box R."

theorem slip_4_5_in_box_R :
  slip_distribution
  [1, 1.5, 1.5, 2, 2, 2, 2.5, 2.5, 3, 3, 3.5, 3.5, 4, 4.5, 5, 5.5]
  ({P := 0, Q := 1, R := 2, S := 3, T := 4, U := 5}: Finset Nat)
  λ b, match b with
  | P => [/* Slips for box P */]
  | Q => [2, /* Other slips for box Q */]
  | R => [4.5, /* Other slips for box R */]
  | S => [/* Slips for box S */]
  | T => [/* Slips for box T */]
  | U => [1, /* Other slips for box U */]
  end := sorry

end slip_4_5_in_box_R_l725_725938


namespace table_condition_l725_725196

-- Define the conditions used in the problem
variables (matrix : ℕ → ℕ → ℕ)
variables (rows cols : ℕ)
variables (positive_in_every_row : ∀ i < rows, ∃ j < cols, 0 < matrix i j)
variables (positive_in_every_col : ∀ j < cols, ∃ i < rows, 0 < matrix i j)
variables (sum_condition : ∀ i j, 0 < matrix i j → (∑ k, matrix i k) = (∑ k, matrix k j))

-- State the theorem we need to prove
theorem table_condition (h : rows = 2015) : cols = 2015 :=
sorry

end table_condition_l725_725196


namespace cross_out_number_l725_725259

theorem cross_out_number (n : ℤ) (h1 : 5 * n + 10 = 10085) : n = 2015 → (n + 5 = 2020) :=
by
  sorry

end cross_out_number_l725_725259


namespace longest_route_uncrossed_streets_l725_725390

/-- In a city divided by one-way streets into an n x n grid of 100 x 100 meter blocks, under the given conditions, the longest path between two specified points W and E will leave 4(n-1) streets uncrossed. -/
theorem longest_route_uncrossed_streets (n : ℕ) (h_even : Even n) :
  let num_streets_uncrossed := 4 * (n - 1)
  in ∃ (longest_route : ℕ), longest_route = (2 * n * (n + 1)) - num_streets_uncrossed :=
by
  sorry

end longest_route_uncrossed_streets_l725_725390


namespace f_is_odd_l725_725663

noncomputable
def f (x : ℝ) : ℝ := x * log (1 + x^2)

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by
  intros x
  calc
    f (-x) = (-x) * log (1 + (-x)^2) : by sorry
        ... = -x * log (1 + x^2)     : by sorry
        ... = -f x                   : by sorry
  sorry

end f_is_odd_l725_725663


namespace petya_numbers_correct_l725_725248

def numbers_petyas_thought_of (sums : List ℕ) (nums : List ℕ) : Prop :=
  ∃ (x1 x2 x3 x4 x5: ℕ), List.perm sums [x1 + x2, x1 + x3, x1 + x4, x1 + x5, x2 + x3, x2 + x4, x2 + x5, x3 + x4, x3 + x5, x4 + x5] ∧
    (List.perm nums [x1, x2, x3, x4, x5])

theorem petya_numbers_correct : 
  numbers_petyas_thought_of [7, 9, 12, 16, 17, 19, 20, 21, 22, 29] [2, 5, 7, 14, 15] :=
sorry

end petya_numbers_correct_l725_725248


namespace original_weight_l725_725021

variable (W : ℝ) -- Let W be the original weight of the side of beef

-- Conditions
def condition1 : ℝ := 0.80 * W -- Weight after first stage
def condition2 : ℝ := 0.70 * condition1 W -- Weight after second stage
def condition3 : ℝ := 0.75 * condition2 W -- Weight after third stage

-- Final weight is given as 570 pounds
theorem original_weight (h : condition3 W = 570) : W = 1357.14 :=
by 
  sorry

end original_weight_l725_725021


namespace shortest_distance_Dasha_to_Vasya_l725_725920

structure Point :=
  (name: String)

structure Distance :=
  (from to: Point)
  (value : ℕ)

def Asya : Point := {name := "Asya"}
def Galia : Point := {name := "Galia"}
def Boria : Point := {name := "Boria"}
def Dasha : Point := {name := "Dasha"}
def Vasya : Point := {name := "Vasya"}

def distances : List Distance :=
  [ {from := Asya, to := Galia, value := 12},
    {from := Galia, to := Boria, value := 10},
    {from := Asya, to := Boria, value := 8},
    {from := Dasha, to := Galia, value := 15},
    {from := Vasya, to := Galia, value := 17} ]

theorem shortest_distance_Dasha_to_Vasya : 
  ∃ d : ℕ, d = 18 ∧ 
    ∀ (p1 p2 : Point), 
      (p1 ≠ Dasha ∨ p2 ≠ Vasya) → 
      (∃ dist : Distance, dist ∈ distances) → 
      (dist.from = p1 ∧ dist.to = p2 → dist.value ≥ d) := 
begin
  sorry
end

end shortest_distance_Dasha_to_Vasya_l725_725920


namespace probability_of_purple_l725_725265

def total_faces := 10
def purple_faces := 3

theorem probability_of_purple : (purple_faces : ℚ) / (total_faces : ℚ) = 3 / 10 := 
by 
  sorry

end probability_of_purple_l725_725265


namespace man_l725_725359

theorem man's_speed_against_the_current (vm vc : ℝ) 
(h1: vm + vc = 15) 
(h2: vm - vc = 10) : 
vm - vc = 10 := 
by 
  exact h2

end man_l725_725359


namespace finite_set_has_basis_l725_725355

noncomputable def exists_basis (M : Finset ℝ) : Prop :=
  ∃ B : Finset ℝ, ∀ m ∈ M, ∃! (exponents : B → ℤ),
    m = B.prod (λ b, b ^ exponents b)

theorem finite_set_has_basis (M : Finset ℝ) (hM : ∀ m ∈ M, 0 < m) : exists_basis M :=
sorry

end finite_set_has_basis_l725_725355


namespace periodicity_iff_condition_l725_725122

-- Define the given conditions
variable (f : ℝ → ℝ)
variable (h_even : ∀ x, f (-x) = f x)

-- State the problem
theorem periodicity_iff_condition :
  (∀ x, f (1 - x) = f (1 + x)) ↔ (∀ x, f (x + 2) = f x) :=
sorry

end periodicity_iff_condition_l725_725122


namespace range_of_m_l725_725097

variable (m : ℝ)

def condition_p (m : ℝ) : Prop :=
  ∃ x₀ : ℝ, m * |sin x₀ + 2| - 9 ≥ 0

def condition_q (m : ℝ) : Prop :=
  ∃ x₀ : ℝ, x₀^2 + 2*m*x₀ + 1 < 0

def not_p (m : ℝ) : Prop :=
  ∀ x : ℝ, m * |sin x + 2| - 9 < 0

def not_q (m : ℝ) : Prop :=
  ∃ x₀ : ℝ, x₀^2 + 2*m*x₀ + 1 < 0

theorem range_of_m (m : ℝ) : not_p m ∧ not_q m → (m < -1 ∨ 1 < m ∧ m < 3) :=
by {
  sorry
}

end range_of_m_l725_725097


namespace total_daisies_l725_725563

-- Define the conditions
def white_daisies : ℕ := 6
def pink_daisies : ℕ := 9 * white_daisies
def red_daisies : ℕ := 4 * pink_daisies - 3

-- Main statement to be proved
theorem total_daisies : white_daisies + pink_daisies + red_daisies = 273 := by
  sorry

end total_daisies_l725_725563


namespace range_m_plus_2n_l725_725510

noncomputable def f (x : ℝ) : ℝ := Real.log x - 1 / x
noncomputable def m_value (t : ℝ) : ℝ := 1 / t + 1 / (t ^ 2)

noncomputable def n_value (t : ℝ) : ℝ := Real.log t - 2 / t - 1

noncomputable def g (x : ℝ) : ℝ := (1 / (x ^ 2)) + 2 * Real.log x - (3 / x) - 2

theorem range_m_plus_2n :
  ∀ m n : ℝ, (∃ t > 0, m = m_value t ∧ n = n_value t) →
  (m + 2 * n) ∈ Set.Ici (-2 * Real.log 2 - 4) := by
  sorry

end range_m_plus_2n_l725_725510


namespace table_conditions_2015_l725_725185

theorem table_conditions_2015 (n : ℕ) (table : Fin 2015 → Fin n → ℕ) :
  (∀ i : Fin 2015, ∃ j : Fin n, 0 < table i j) ∧   -- Each row has a positive number
  (∀ j : Fin n, ∃ i : Fin 2015, 0 < table i j) ∧   -- Each column has a positive number
  (∀ i : Fin 2015, ∀ j : Fin n, 
     0 < table i j → 
     (∑ k : Fin n, table i k) = (∑ k : Fin 2015, table k j))   -- Positive cell condition
  → n = 2015 := 
by
  sorry

end table_conditions_2015_l725_725185


namespace log_equation_solution_l725_725898

open Real

theorem log_equation_solution (x : ℝ) (h : log 10 (x ^ 2 - 5 * x + 8) = 2) : x = 13 ∨ x = -8 :=
sorry

end log_equation_solution_l725_725898


namespace fence_order_fulfillment_l725_725751

-- Definitions for conditions
def total_length : ℕ := 297
def total_sections : ℕ := 16
def max_sections : ℕ := 8
def max_length : ℕ := 20
def section_length_variants : List ℕ := [18, 17]

-- Predicate for sections' lengths, ensuring remaining lengths are shorter by 1, 2, or 3 meters
def is_valid_length (x : ℕ) (l : ℕ) := x - l ∈ {1, 2, 3}

-- Formalize that the solution meets the conditions
theorem fence_order_fulfillment :
  ∑ (section : ℕ) in [max_length, max_length, max_length, max_length, max_length, max_length, max_length, max_length, 18, 17, 17, 17, 17, 17, 17, 17], id section = total_length ∧
  ∀ section ∈ section_length_variants, is_valid_length max_length section :=
by {
  sorry
}

end fence_order_fulfillment_l725_725751


namespace inequality_proof_l725_725100

open Real

theorem inequality_proof {x y : ℝ} (hx : x < 0) (hy : y < 0) : 
    (x ^ 4 / y ^ 4) + (y ^ 4 / x ^ 4) - (x ^ 2 / y ^ 2) - (y ^ 2 / x ^ 2) + (x / y) + (y / x) >= 2 := 
by
    sorry

end inequality_proof_l725_725100


namespace angle_COD_65_l725_725174

variable (A B C D O : Type)
variable (straight_angle : ∠AOD = 180)
variable (bisect : ∠BOD / 2 = ∠COD)
variable (angle_AOB : ∠AOB = 50)

theorem angle_COD_65 : ∠COD = 65 :=
by
  -- We'll add the proof here in the full solution,
  -- but for now, we just show the statement format.
  sorry

end angle_COD_65_l725_725174


namespace total_ages_l725_725682

theorem total_ages (groom_age : ℕ) (bride_age : ℕ) (h_groom : groom_age = 83) (h_bride : bride_age = groom_age + 19) :
  groom_age + bride_age = 185 := 
by {
  rw [h_groom, h_bride],
  exact rfl,
}

end total_ages_l725_725682


namespace seashells_initial_count_l725_725970

theorem seashells_initial_count (S : ℝ) (h : S + 4.0 = 10) : S = 6.0 :=
by
  sorry

end seashells_initial_count_l725_725970


namespace fraction_of_area_l725_725631

noncomputable section

open Real

-- Definitions of points A, B, C, X, Y, and Z with their given coordinates
def A := (2, 0) : ℝ × ℝ
def B := (8, 12) : ℝ × ℝ
def C := (14, 0) : ℝ × ℝ

def X := (6, 0) : ℝ × ℝ
def Y := (8, 4) : ℝ × ℝ
def Z := (10, 0) : ℝ × ℝ

-- Definition of the area of a triangle given vertices
def area (p₁ p₂ p₃ : ℝ × ℝ) : ℝ := 
  abs ((p₂.1 - p₁.1) * (p₃.2 - p₁.2) - (p₂.2 - p₁.2) * (p₃.1 - p₁.1)) / 2

-- Areas of triangles ABC and XYZ
def area_ABC := area A B C
def area_XYZ := area X Y Z

-- The Lean statement
theorem fraction_of_area : (area_XYZ / area_ABC) = 1 / 9 := by
  sorry

end fraction_of_area_l725_725631


namespace problem_1_problem_2_l725_725135

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x + Real.log x)
def g (x : ℝ) : ℝ := x^2

theorem problem_1 (a : ℝ) (ha : a ≠ 0) : 
  (∀ (x : ℝ), f a x = a * (x + Real.log x)) →
  deriv (f a) 1 = deriv g 1 → a = 1 := 
by 
  sorry

theorem problem_2 (a : ℝ) (ha : 0 < a) (hb : a < 1) (x1 x2 : ℝ) 
  (hx1 : 1 ≤ x1) (hx2 : x2 ≤ 2) (hx12 : x1 ≠ x2) : 
  |f a x1 - f a x2| < |g x1 - g x2| := 
by 
  sorry

end problem_1_problem_2_l725_725135


namespace tangent_line_condition_l725_725860

theorem tangent_line_condition (t : ℝ) : t = 2 * Real.sqrt 2 → 
  (∀ x y : ℝ, x^2 + y^2 = 4 → x + y = t) → 
  ∃x y : ℝ, x^2 + y^2 = 4 ∧ x + y = t :=
begin
  sorry
end

end tangent_line_condition_l725_725860


namespace optimal_room_rate_to_maximize_income_l725_725356

noncomputable def max_income (x : ℝ) : ℝ := x * (300 - 0.5 * (x - 200))

theorem optimal_room_rate_to_maximize_income :
  ∀ x, 200 ≤ x → x ≤ 800 → max_income x ≤ max_income 400 :=
by
  sorry

end optimal_room_rate_to_maximize_income_l725_725356


namespace area_of_bounded_region_l725_725428

theorem area_of_bounded_region :
  let region := {p : ℝ × ℝ | (p.1 = 0 ∧ p.2 ≥ 0) ∨ (p.2 = 0 ∧ p.1 ≥ 0) ∨ (p.1 = 2) ∨ (p.2 = 2)}
  ∃ (s : set (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ s ↔ (y = x))
    ∧ is_square s 
    ∧ area s = 4 := 
sorry

end area_of_bounded_region_l725_725428


namespace fraction_of_triangle_area_l725_725627

open Real

def point := (ℝ × ℝ)

def area_of_triangle (A B C : point) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (abs ((x1 * (y2 - y3)) + (x2 * (y3 - y1)) + (x3 * (y1 - y2))) / 2)

def A : point := (2, 0)
def B : point := (8, 12)
def C : point := (14, 0)

def X : point := (6, 0)
def Y : point := (8, 4)
def Z : point := (10, 0)

theorem fraction_of_triangle_area :
  (area_of_triangle X Y Z) / (area_of_triangle A B C) = 1 / 9 :=
by
  sorry

end fraction_of_triangle_area_l725_725627


namespace number_of_suits_sold_l725_725066

theorem number_of_suits_sold
  (commission_rate: ℝ)
  (price_per_suit: ℝ)
  (price_per_shirt: ℝ)
  (price_per_loafer: ℝ)
  (number_of_shirts: ℕ)
  (number_of_loafers: ℕ)
  (total_commission: ℝ)
  (suits_sold: ℕ)
  (total_sales: ℝ)
  (total_sales_from_non_suits: ℝ)
  (sales_needed_from_suits: ℝ)
  : 
  (commission_rate = 0.15) → 
  (price_per_suit = 700.0) → 
  (price_per_shirt = 50.0) → 
  (price_per_loafer = 150.0) → 
  (number_of_shirts = 6) → 
  (number_of_loafers = 2) → 
  (total_commission = 300.0) →
  (total_sales = total_commission / commission_rate) →
  (total_sales_from_non_suits = number_of_shirts * price_per_shirt + number_of_loafers * price_per_loafer) →
  (sales_needed_from_suits = total_sales - total_sales_from_non_suits) →
  (suits_sold = sales_needed_from_suits / price_per_suit) →
  suits_sold = 2 :=
by
  sorry

end number_of_suits_sold_l725_725066


namespace range_of_a_l725_725177

theorem range_of_a (x a : ℝ) : (∃ x : ℝ,  |x + 2| + |x - 3| ≤ |a - 1| ) ↔ (a ≤ -4 ∨ a ≥ 6) :=
by
  sorry

end range_of_a_l725_725177


namespace probability_one_odd_is_9_over_10_l725_725783

-- Define the set and subsets for odd and even numbers
def number_set : set ℕ := { 1, 2, 3, 4, 5 }
def odd_numbers : set ℕ := { 1, 3, 5 }
def even_numbers : set ℕ := { 2, 4 }

-- Define the total number of ways to pick any two numbers from the set
def total_ways_to_pick_two : ℕ := nat.choose 5 2 

-- Define the number of ways to pick two even numbers (i.e., choosing from {2, 4})
def ways_to_pick_two_evens : ℕ := nat.choose 2 2 

-- Define the probability of selecting both even numbers
def probability_both_evens : ℚ := ways_to_pick_two_evens / total_ways_to_pick_two

-- Define the probability of selecting at least one odd number as the complement
def probability_at_least_one_odd : ℚ := 1 - probability_both_evens

-- Theorem: Prove the probability of selecting at least one odd number is 9/10
theorem probability_one_odd_is_9_over_10 : probability_at_least_one_odd = 9 / 10 :=
by 
  sorry

end probability_one_odd_is_9_over_10_l725_725783


namespace correct_calculation_l725_725310

theorem correct_calculation (a b c d : ℤ) (h1 : a = -1) (h2 : b = -3) (h3 : c = 3) (h4 : d = -3) :
  a * b = c :=
by 
  rw [h1, h2]
  exact h3.symm

end correct_calculation_l725_725310


namespace tortoise_wins_l725_725912

-- Definitions based on the problem conditions
structure Movement :=
  (initial_speed : ℕ)
  (dynamics : ℕ → ℕ)  -- a function to represent changing speed dynamics over time

def hare : Movement := {
  initial_speed := 10,
  dynamics := λ t, if t < 2 then 10 else if t < 5 then 0 else if t < 8 then 2 else 10
}

def tortoise : Movement := {
  initial_speed := 1,
  dynamics := λ t, t + 1
}

-- Proving that the tortoise wins based on the given conditions
theorem tortoise_wins (hare tortoise : Movement) : tortoise = tortoise ∧ hare = hare → "Tortoise wins" :=
by {
  unfold Movement at hare tortoise,
  sorry
}

end tortoise_wins_l725_725912


namespace find_point_R_positions_l725_725950

theorem find_point_R_positions :
  ∃ (P Q : ℝ × ℝ), dist P Q = 10 ∧
  let S := (R : ℝ × ℝ) in
  (∃ R : ℝ × ℝ, 0 < R.1 ∧ 0 < R.2 ∧
   is_right_triangle ∆ P Q R ∧
   triangle_area ∆ P Q R = 15) ∧
  (∃! R' : ℝ × ℝ, 0 < R'.1 ∧ 0 < R'.2 ∧
   is_right_triangle ∆ P Q R' ∧
   triangle_area ∆ P Q R' = 15) ∧
  card S = 6 := by
  sorry

end find_point_R_positions_l725_725950


namespace solve_for_h_l725_725589

theorem solve_for_h (h : ℝ → ℝ) (x : ℝ) (h_def : ∀ x : ℝ, h(5*x - 2) = 3*x + 10) : h x = x → x = 28 :=
by
  intros hx
  sorry -- this part will contain the proof

end solve_for_h_l725_725589


namespace cylinder_distance_l725_725910

-- Define the problem
theorem cylinder_distance :
  ∀ (points : Fin 9 → ℝ × ℝ × ℝ), 
  (∀ i, i < 9 → 
    (points i).1 ∈ Icc (-1 : ℝ) (1) ∧ 
    (points i).2 ∈ Icc (-1 : ℝ) (1) ∧ 
    (points i).3 ∈ Icc (-1 : ℝ) (1)) → 
  ∃ (i j : Fin 9), i ≠ j ∧ 
    (dist (points i) (points j) ≤ √3) :=
by
  sorry

end cylinder_distance_l725_725910


namespace table_condition_l725_725195

-- Define the conditions used in the problem
variables (matrix : ℕ → ℕ → ℕ)
variables (rows cols : ℕ)
variables (positive_in_every_row : ∀ i < rows, ∃ j < cols, 0 < matrix i j)
variables (positive_in_every_col : ∀ j < cols, ∃ i < rows, 0 < matrix i j)
variables (sum_condition : ∀ i j, 0 < matrix i j → (∑ k, matrix i k) = (∑ k, matrix k j))

-- State the theorem we need to prove
theorem table_condition (h : rows = 2015) : cols = 2015 :=
sorry

end table_condition_l725_725195


namespace cyclic_cosine_inequality_l725_725571

theorem cyclic_cosine_inequality
  (α β γ : ℝ)
  (hα : 0 ≤ α ∧ α ≤ π / 2)
  (hβ : 0 ≤ β ∧ β ≤ π / 2)
  (hγ : 0 ≤ γ ∧ γ ≤ π / 2)
  (cos_sum : Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 = 1) :
  2 ≤ (1 + Real.cos α ^ 2) ^ 2 * (Real.sin α) ^ 4
       + (1 + Real.cos β ^ 2) ^ 2 * (Real.sin β) ^ 4
       + (1 + Real.cos γ ^ 2) ^ 2 * (Real.sin γ) ^ 4 ∧
    (1 + Real.cos α ^ 2) ^ 2 * (Real.sin α) ^ 4
       + (1 + Real.cos β ^ 2) ^ 2 * (Real.sin β) ^ 4
       + (1 + Real.cos γ ^ 2) ^ 2 * (Real.sin γ) ^ 4
      ≤ (1 + Real.cos α ^ 2) * (1 + Real.cos β ^ 2) * (1 + Real.cos γ ^ 2) :=
by 
  sorry

end cyclic_cosine_inequality_l725_725571


namespace participate_in_all_curriculums_l725_725204

variable (yoga cooking weaving cooking_only cooking_and_yoga cooking_and_weaving all_three : ℕ)
variable (total_cooking : ℕ)

-- Specify the known quantities
axiom yoga_eq : yoga = 25
axiom cooking_eq : cooking = 15
axiom weaving_eq : weaving = 8
axiom cooking_only_eq : cooking_only = 2
axiom cooking_and_yoga_eq : cooking_and_yoga = 7
axiom cooking_and_weaving_eq : cooking_and_weaving = 3
axiom total_cooking_eq : total_cooking = 15

-- The equation from the problem
axiom cooking_distribution : cooking_only + cooking_and_yoga + cooking_and_weaving + all_three = total_cooking

-- Prove the number of people who participate in all curriculums
theorem participate_in_all_curriculums : all_three = 3 :=
by
  have h := congr_arg (λ t, total_cooking_eq) cooking_distribution
  simp [*, add_assoc, add_comm, add_left_comm] at h
  sorry

end participate_in_all_curriculums_l725_725204


namespace area_of_region_l725_725420

noncomputable def sec (θ : ℝ) := (cos θ)⁻¹
noncomputable def csc (θ : ℝ) := (sin θ)⁻¹

def region (r θ : ℝ) : Prop :=
  (r = 2 * sec θ ∧ (0 ≤ θ ∧ θ ≤ π / 2)) ∨ 
  (r = 2 * csc θ ∧ (0 ≤ θ ∧ θ ≤ π / 2))

theorem area_of_region :
  let bounded_region := { (x, y) | 0 ≤ x ∧ x ≤ 2 ∧ 0 ≤ y ∧ y ≤ 2 } in
  ∃ (A : ℝ), A = 4 ∧ (∀ (a b : ℝ), bounded_region (a, b)) :=
begin
  let bounded_region := { p | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2 },
  use 4,
  split,
  { refl, },
  { intros a b hb,
    exact hb, },
end

end area_of_region_l725_725420


namespace semi_circle_radius_l725_725282

/-- The radius of a semi-circle with a perimeter of 33.42035224833366 cm is 6.5 cm. -/
theorem semi_circle_radius
  (P : ℝ) (hP : P = 33.42035224833366) :
  let π := Real.pi in
  let r := 6.5 in
  P = π * r + 2 * r :=
by
  -- Given that the perimeter is 33.42035224833366 cm
  have h1 : P = 33.42035224833366 := by exact hP
  -- We need to show that the given radius satisfies the formula
  have h2 : r = 6.5 := by rfl
  -- Final equation to check
  exact h1


end semi_circle_radius_l725_725282


namespace I_union_II_range_a_l725_725859

noncomputable def solve_I (a : ℝ) : Set ℝ :=
  {x | 2 * x - a < 0} ∪ {x | x ≤ 2 ∨ x ≥ 1 + a}

theorem I_union (a : ℝ) (h : a = -4) : solve_I a = { x | x < -2 } ∪ { x | x ≥ 2 } :=
by
  sorry

noncomputable def solve_II (a : ℝ) : Set ℝ × Set ℝ :=
  ({x | 2 * x - a < 0}, {x | x^2 - (3 + a) * x + 2 * (1 + a) ≥ 0})

theorem II_range_a (A B : Set ℝ) (hA : A = {x | 2 * x - a < 0})
  (hB : B = {x | x² - (3 + a) * x + 2 * (1 + a) ≥ 0}) (h : A ⊆ B) :
  -1 ≤ a ∧ a ≤ 4 :=
by
  sorry

end I_union_II_range_a_l725_725859


namespace range_of_a_l725_725089

theorem range_of_a (a : ℝ) : -7 / 3 < a ∧ a ≤ -2 ↔
  (∃ (xs : Finset ℤ), xs = (Finset.range 3).map (λ i, 11 - i) ∧ xs.card = 3 ∧
  ∀ x ∈ xs, x ≤ 11 ∧ (2 * ↑x + 2) / 3 < ↑x + a) :=
by
  sorry

end range_of_a_l725_725089


namespace intersection_of_M_and_N_l725_725515

open Set

def M : Set ℕ := {0, 1, 2, 3}
def N : Set ℕ := {2, 3}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := 
by 
  sorry

end intersection_of_M_and_N_l725_725515


namespace grid_distinct_numbers_l725_725543

noncomputable def floor_sqrt (n : ℕ) : ℕ :=
nat.floor (real.sqrt n)

theorem grid_distinct_numbers
  (n : ℕ)
  (grid : fin n → fin n → ℕ)
  (h_range : ∀ (i j : fin n), 1 ≤ grid i j ∧ grid i j ≤ n)
  (h_occurrence : ∀ (k : ℕ), 1 ≤ k ∧ k ≤ n → (finset.univ.bUnion (λ i, finset.filter (λ j, grid i j = k) finset.univ)).card = n) :
  ∃ (i : fin n), (finset.univ.image (λ j, grid i j)).card ≥ floor_sqrt n ∨ ∃ (j : fin n), (finset.univ.image (λ i, grid i j)).card ≥ floor_sqrt n :=
by
  sorry

end grid_distinct_numbers_l725_725543


namespace perpendicular_lines_in_square_circle_l725_725221

open EuclideanGeometry

noncomputable def square_property : Prop :=
  ∀ (ABCD : Square) (Γ : circle) (M P R Q S : point),
  let A := ABCD.A,
      B := ABCD.B,
      C := ABCD.C,
      D := ABCD.D,
      O := ABCD.center
  in
  (∃ (h1 : M ∈ Γ), 
  ∃ (h2 : M ∈ arc C D) (h3 : M ≠ A),
  let P := (line_through A M) ∩ (line_through B D),
      R := (line_through A M) ∩ (line_through C D),
      Q := (line_through B M) ∩ (line_through A C),
      S := (line_through B M) ∩ (line_through D C)
  in
  is_perpendicular (line_through P S) (line_through Q R))

theorem perpendicular_lines_in_square_circle : square_property :=
sorry

end perpendicular_lines_in_square_circle_l725_725221


namespace molecular_weight_compound_l725_725708

-- Definitions of atomic weights
def atomic_weight_Cu : ℝ := 63.546
def atomic_weight_C : ℝ := 12.011
def atomic_weight_O : ℝ := 15.999

-- Definitions of the number of atoms in the compound
def num_Cu : ℝ := 1
def num_C : ℝ := 1
def num_O : ℝ := 3

-- The molecular weight of the compound
def molecular_weight : ℝ := (num_Cu * atomic_weight_Cu) + (num_C * atomic_weight_C) + (num_O * atomic_weight_O)

-- Statement to prove
theorem molecular_weight_compound : molecular_weight = 123.554 := by
  sorry

end molecular_weight_compound_l725_725708


namespace max_sum_of_distances_l725_725958

noncomputable def min_distance_to_integer (x : ℝ) : ℝ :=
  Inf (set.image (λ m : ℤ, |x - m|) set.univ)

theorem max_sum_of_distances {n : ℕ} (hn : n ≥ 2) (x : ℕ → ℝ)
  (h_sum_int : ∃ m : ℤ, (∑ k in finset.range n, x k) = m) :
  (∑ k in finset.range n, min_distance_to_integer (x k))
  ≤ ℝ.floor (n / 2) :=
sorry

end max_sum_of_distances_l725_725958


namespace total_molecular_weight_of_products_l725_725399

/-- Problem Statement: Determine the total molecular weight of the products formed when
    8 moles of Copper(II) carbonate (CuCO3) react with 6 moles of Diphosphorus pentoxide (P4O10)
    to form Copper(II) phosphate (Cu3(PO4)2) and Carbon dioxide (CO2). -/
theorem total_molecular_weight_of_products 
  (moles_CuCO3 : ℕ) 
  (moles_P4O10 : ℕ)
  (atomic_weight_Cu : ℝ := 63.55)
  (atomic_weight_P : ℝ := 30.97)
  (atomic_weight_O : ℝ := 16.00)
  (atomic_weight_C : ℝ := 12.01)
  (molecular_weight_CuCO3 : ℝ := atomic_weight_Cu + atomic_weight_C + 3 * atomic_weight_O)
  (molecular_weight_CO2 : ℝ := atomic_weight_C + 2 * atomic_weight_O)
  (molecular_weight_Cu3PO4_2 : ℝ := (3 * atomic_weight_Cu) + (2 * atomic_weight_P) + (8 * atomic_weight_O))
  (moles_Cu3PO4_2_formed : ℝ := (8 : ℝ) / 3)
  (moles_CO2_formed : ℝ := 8)
  (total_molecular_weight_Cu3PO4_2 : ℝ := moles_Cu3PO4_2_formed * molecular_weight_Cu3PO4_2)
  (total_molecular_weight_CO2 : ℝ := moles_CO2_formed * molecular_weight_CO2) : 
  (total_molecular_weight_Cu3PO4_2 + total_molecular_weight_CO2) = 1368.45 := by
  sorry

end total_molecular_weight_of_products_l725_725399


namespace functional_equation_divisibility_l725_725074

theorem functional_equation_divisibility (f : ℕ+ → ℕ+) :
  (∀ x y : ℕ+, (f x)^2 + y ∣ f y + x^2) → (∀ x : ℕ+, f x = x) :=
by
  sorry

end functional_equation_divisibility_l725_725074


namespace sum_of_possible_a_l725_725181

theorem sum_of_possible_a : 
  (∑ a in {a : ℤ | ∃ p q : ℤ, p + q = a ∧ p * q = 2 * a}, a) = 16 := 
by
  sorry

end sum_of_possible_a_l725_725181


namespace teacher_exchange_arrangements_l725_725914

theorem teacher_exchange_arrangements :
  ∃ (n : ℕ), n = 12 ∧
    (∃ C1 C2 : Type, 
     ∃ (teachers : (C1 ⊕ C2) → Prop),
     (∀ x : C1 ⊕ C2, x ∈ teachers x) ∧
     ∃ (schoolA schoolB : Finset (C1 ⊕ C2)), 
     schoolA.card = 3 ∧ schoolB.card = 3 ∧
     (∃ (hasChineseA : ∃ x : C1, x ∈ schoolA) ∧
      ∃ (hasMathA : ∃ y : C2, y ∈ schoolA) ∧
      ∃ (hasChineseB : ∃ x : C1, x ∈ schoolB) ∧
      ∃ (hasMathB : ∃ y : C2, y ∈ schoolB))) :=
sorry

end teacher_exchange_arrangements_l725_725914


namespace margo_pairing_probability_l725_725908

theorem margo_pairing_probability (students : Finset ℕ)
  (H_50_students : students.card = 50)
  (margo irma jess kurt : ℕ)
  (H_margo_in_students : margo ∈ students)
  (H_irma_in_students : irma ∈ students)
  (H_jess_in_students : jess ∈ students)
  (H_kurt_in_students : kurt ∈ students)
  (possible_partners : Finset ℕ := students.erase margo) :
  (3: ℝ) / 49 = ((3: ℝ) / (possible_partners.card: ℝ)) :=
by
  -- The actual steps of the proof will be here
  sorry

end margo_pairing_probability_l725_725908


namespace heart_and_face_card_probability_l725_725700

noncomputable def probability_heart_face_card : ℚ :=
  -- step-1: Calculate respective probabilities and sum them as in the given solution
  let P_ace_of_hearts_first := (1 / 52) * (11 / 51)
  let P_heart_not_ace_first := (12 / 52) * (12 / 51)
  P_ace_of_hearts_first + P_heart_not_ace_first

theorem heart_and_face_card_probability :
  probability_heart_face_card = 5 / 86 :=
begin
  sorry
end

end heart_and_face_card_probability_l725_725700


namespace work_completion_time_for_A_l725_725735

-- Defining the problem conditions and the proof goal
theorem work_completion_time_for_A
  (B_completion_time : ℝ) (combined_completion_time : ℝ) (B_work_rate : ℝ) (combined_work_rate : ℝ)
  (A_completion_time : ℝ) (A_work_rate : ℝ) :
  B_completion_time = 16 ∧
  combined_completion_time = 16/3 ∧
  B_work_rate = 1 / B_completion_time ∧
  combined_work_rate = 1 / combined_completion_time ∧
  A_work_rate = combined_work_rate - B_work_rate ∧
  A_completion_time = 1 / A_work_rate →
  A_completion_time = 8 :=
begin
  -- Proof to be inserted here
  sorry
end

end work_completion_time_for_A_l725_725735


namespace infinitely_many_n_squared_plus_one_no_special_divisor_l725_725330

theorem infinitely_many_n_squared_plus_one_no_special_divisor :
  ∃ (f : ℕ → ℕ), (∀ n, f n ≠ 0) ∧ ∀ n, ∀ k, f n^2 + 1 ≠ k^2 + 1 ∨ k^2 + 1 = 1 :=
by
  sorry

end infinitely_many_n_squared_plus_one_no_special_divisor_l725_725330


namespace stuart_segments_l725_725262

-- Definitions and conditions
variables (C_large C_small : Circle) (A B C : Point)
variable (chords : list (Chord C_large))
hypothesis tangent_smaller : ∀ (chord : Chord C_large), tangent chord C_small
hypothesis angle_ABC : ∃ (ABC : Triangle), mangle ABC = 60

-- Problem statement
theorem stuart_segments :
  ∃ n : ℕ, n = 3 ∧
    (n * 120 = 360) ∧
    (∀ i, (chords.nth i).is_tangent_to(C_small)) ∧
    starts_at_A (chords.nth 0) A ∧
    returns_to_start_point_A (chords.nth (n - 1)) A :=
by
  sorry

end stuart_segments_l725_725262


namespace shorter_piece_length_l725_725334

theorem shorter_piece_length (total_length : ℝ) (shorter_fraction : ℝ) (longer_ratio : ℝ) 
    (h_total : total_length = 70) (h_ratio : shorter_fraction = 3 / 7) (h_longer : longer_ratio = 7 / 3) :
    let x : ℝ := total_length / (1 + (1 / shorter_fraction))
    in x = 21 := 
by
  sorry

end shorter_piece_length_l725_725334


namespace hyperbola_symmetric_asymptotes_l725_725885

noncomputable def M : ℝ := 225 / 16

theorem hyperbola_symmetric_asymptotes (M_val : ℝ) :
  (∀ x y : ℝ, (x^2 / 9 - y^2 / 16 = 1 → y = x * (4 / 3) ∨ y = -x * (4 / 3))
  ∧ (y^2 / 25 - x^2 / M_val = 1 → y = x * (5 / Real.sqrt M_val) ∨ y = -x * (5 / Real.sqrt M_val)))
  → M_val = M := by
  sorry

end hyperbola_symmetric_asymptotes_l725_725885


namespace find_LD_l725_725322

theorem find_LD (A B C D L K : ℝ²) (h_square : sq ABCD) (hL_on_CD : L ∈ segment CD) (hK_on_DA_ext : K ∈ line DA \ point A)
(h_angle_KBL : ∠ KBL = 90) (KD CL : ℝ) (hKD : KD = 19) (hCL : CL = 6) :
  LD = 7 := 
sorry

end find_LD_l725_725322


namespace journey_second_half_speed_l725_725746

theorem journey_second_half_speed
  (t_total : ℕ)
  (v1 : ℕ)
  (d_total : ℕ)
  (v2 : ℕ)
  (h1 : t_total = 30)
  (h2 : v1 = 20)
  (h3 : d_total = 400)
  (half_distance : ℕ := d_total / 2)
  (t1 : ℕ := half_distance / v1)
  (t2 : ℕ := t_total - t1) :
  v2 = half_distance / t2 :=
begin
  -- Provided conditions
  rw [h1, h2, h3],
  -- half_distance := 400 / 2 = 200
  -- first half time t1 := 200 / 20 = 10
  -- second half time t2 := 30 - 10 = 20
  -- second half speed v2 := 200 / 20 = 10
  sorry -- Proof steps would go here
end

end journey_second_half_speed_l725_725746


namespace ellipse_a_plus_k_l725_725034

/-- An ellipse has its foci at (2, 2) and (2, 6). Given that it passes through the point (-3, 4),
its equation is of the form (x-h)²/a² + (y-k)²/b² = 1 where a, b, h, k are constants, and a and b 
are positive. Prove that a + k = 4 + sqrt(29). -/
theorem ellipse_a_plus_k :
  ∃ (a b h k : ℝ), a > 0 ∧ b > 0 ∧
    ( ((2 - 2)^2 + (2 - 6)^2).sqrt * 2 = a * 2.sqrt + ( (2 - 2)^2 + (-3 - 2)^2).sqrt + ( (4 - 2)^2 + (-3 - 2)^2).sqrt) ∧
    (h = 2 ∧ k = 4 ∧ 4 = (a^2) - (b^2) ∧ b = 5) ∧
    (h = 2 ∧ k = 4 ∧ a + k = 4 + sqrt(29)) :=
sorry

end ellipse_a_plus_k_l725_725034


namespace range_of_a_l725_725134

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 2 then -x^2 + 2*x + 3 else 6 + log a x

theorem range_of_a (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (range_f : ∀ x, f a x ≤ 4) : 
    ∀ x, a ∈ set.Ico (real.sqrt 2 / 2) 1 :=
sorry

end range_of_a_l725_725134


namespace num_clients_l725_725718

theorem num_clients (cars : ℕ) (selections_per_car : ℕ) (selections_per_client : ℕ) 
  (hc : cars = 16) (hsc : selections_per_car = 3) (hscpc : selections_per_client = 2) :
  (cars * selections_per_car) / selections_per_client = 24 :=
  by
    rw [hc, hsc, hscpc]
    calc
      (16 * 3) / 2 = 48 / 2 : by rw Nat.mul_comm
                   ... = 24 : by norm_num

end num_clients_l725_725718


namespace find_k_l725_725517

variables (a b : ℝ × ℝ) (k : ℝ)
def vector_a : ℝ × ℝ := (2, 1)
def vector_b : ℝ × ℝ := (1, k) - vector_a
def perpendicular (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem find_k : vector_a = (2, 1) ∧ (2, 1) + vector_b = (1, k) ∧ perpendicular vector_a vector_b → k = 3 :=
by 
  sorry

end find_k_l725_725517


namespace liza_butter_fraction_l725_725963

theorem liza_butter_fraction :
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧
  let total := 10 in
  let peanut := total / 5 in
  let remaining_after_peanut := total - peanut in
  let sugar := remaining_after_peanut / 3 in
  let remaining_after_sugar := remaining_after_peanut - sugar in
  let remaining_after_chocolate := remaining_after_sugar - (total * x - peanut - sugar) in
  remaining_after_chocolate = 2 → x = 1/3 :=
by
sorry

end liza_butter_fraction_l725_725963


namespace soap_pack_count_l725_725605

theorem soap_pack_count (total_bars packs : ℕ) (h1 : total_bars = 30) (h2 : packs = 6) : (total_bars / packs) = 5 :=
by
  rw [h1, h2]
  norm_num
  sorry

end soap_pack_count_l725_725605


namespace rhombus_diagonal_inverse_proportion_l725_725774

noncomputable def is_inverse_prop (f : ℝ → ℝ) : Prop :=
∃ k, k ≠ 0 ∧ (∀ x ≠ 0, f x = k / x)

theorem rhombus_diagonal_inverse_proportion (x y : ℝ) (h1 : 20 = 1/2 * x * y) : is_inverse_prop (λ x, 40 / x) :=
by
  -- add the necessary hypothesis 
  have h : ∀ x ≠ 0, y = 40 / x := 
    λ x h', by 
      rw [← mul_div_cancel (40 : ℝ) h', mul_comm, div_eq_mul_inv, ← mul_assoc, mul_inv_cancel h', ← h1, ← mul_assoc]
      simp
  simp only [is_inverse_prop, ge_iff_le, gt_iff_lt]
  use 40
  constructor
  norm_num
  intros x h'
  exact h x h'

end rhombus_diagonal_inverse_proportion_l725_725774


namespace monotonicity_f_inequality_f_div_l725_725131

noncomputable def f (a x : ℝ) : ℝ := a * x * real.exp x

theorem monotonicity_f {a : ℝ} (h₀ : a ≠ 0) :
  (∀ x < -1, deriv (f a) x < 0) ∧ (∀ x > -1, deriv (f a) x > 0)
  ∨ (∀ x < -1, deriv (f a) x > 0) ∧ (∀ x > -1, deriv (f a) x < 0) :=
sorry

theorem inequality_f_div (a x : ℝ) (h₀ : a ≠ 0) (h₁ : a ≥ 4 / real.exp 2) (h₂ : x > 0) :
  (f a x) / (x + 1) - (x + 1) * real.log x > 0 :=
sorry

end monotonicity_f_inequality_f_div_l725_725131


namespace michelle_sandwiches_l725_725967

def sandwiches_left (total : ℕ) (given_to_coworker : ℕ) (kept : ℕ) : ℕ :=
  total - given_to_coworker - kept

theorem michelle_sandwiches : sandwiches_left 20 4 (4 * 2) = 8 :=
by
  sorry

end michelle_sandwiches_l725_725967


namespace solve_quartic_eq_l725_725985

theorem solve_quartic_eq {x : ℝ} : (x - 4)^4 + (x - 6)^4 = 16 → (x = 4 ∨ x = 6) :=
by
  sorry

end solve_quartic_eq_l725_725985


namespace find_distance_ad_l725_725976

noncomputable def distance_abc_triangle (A B C : ℝ × ℝ) (AC_angle : ℝ) (AC_distance : ℝ) : Prop :=
0 < AC_angle ∧ AC_angle = 45 ∧
(AC_distance = 15 * Real.sqrt 2) ∧
(C.1 = B.1 ∧ C.2 = B.2 + AC_distance / Real.sqrt 2) ∧
(B.1 = A.1 + AC_distance / Real.sqrt 2 ∧ B.2 = A.2)

noncomputable def distance_ad_triangle (A C D : ℝ × ℝ) (distance_cd : ℝ) : Prop :=
(0 < distance_cd ∧ distance_cd = 30 ∧
 D.1 = C.1 - distance_cd / Real.sqrt 2 ∧ D.2 = C.2 + distance_cd / Real.sqrt 2)

theorem find_distance_ad {A B C D : ℝ × ℝ} 
(ABC_cond : ∃ AC_angle AC_distance, distance_abc_triangle A B C AC_angle AC_distance)
(CD_cond : ∃ distance_cd, distance_ad_triangle C D distance_cd):
∃ distance_ad, Real.dist A D = distance_ad := sorry

end find_distance_ad_l725_725976


namespace probability_of_different_colors_correct_l725_725827

def ball : Type := string

def red_balls : list ball := ["red", "red"]
def yellow_balls : list ball := ["yellow", "yellow"]
def white_balls : list ball := ["white"]

def all_balls : list ball := red_balls ++ yellow_balls ++ white_balls

def different_colors (b1 b2 : ball) : bool :=
  b1 ≠ b2

def number_of_ways_to_choose_two_balls : ℕ :=
  Nat.choose (List.length all_balls) 2

def number_of_ways_to_choose_two_balls_of_same_colors : ℕ :=
  Nat.choose 2 2 + Nat.choose 2 2

noncomputable def probability_of_different_colors : ℚ :=
  1 - (number_of_ways_to_choose_two_balls_of_same_colors : ℚ) / (number_of_ways_to_choose_two_balls : ℚ)

theorem probability_of_different_colors_correct :
  probability_of_different_colors = 4 / 5 :=
by
  -- Proof omitted
  sorry

end probability_of_different_colors_correct_l725_725827


namespace group_total_cost_l725_725383

structure GroupOrder :=
  (numAdults : ℕ)
  (numKids : ℕ)
  (adultMealCounts : List (ℕ × ℕ)) -- list of (count, cost) pairs for adult meals
  (numBeverages : ℕ)
  (adultBeverageCost : ℕ)
  (kidBeverageCost : ℕ)

def totalCost (order : GroupOrder) : ℕ :=
  let adultMealCost := order.adultMealCounts.foldr (λ (p : ℕ × ℕ) acc, acc + p.1 * p.2) 0
  let maxAdultBeverages := order.numAdults
  let adultBeverages := min maxAdultBeverages order.numBeverages
  let kidBeverages := order.numBeverages - adultBeverages
  adultMealCost + adultBeverages * order.adultBeverageCost + kidBeverages * order.kidBeverageCost

theorem group_total_cost :
  let order := GroupOrder.mk 7 7 [(4, 5), (2, 7), (1, 9)] 9 2 1 in
  totalCost order = 59 :=
by 
  sorry

end group_total_cost_l725_725383


namespace max_AB_CD_max_area_ABCD_l725_725500

-- Define the ellipse equation and condition for the chords
def ellipse (x y : ℝ) : Prop :=
  (x^2 / 4) + y^2 = 1

def vector_eqn_of_chords (a b c d : ℝ × ℝ) : Prop :=
  ⟨2 * (b.1 - a.1), 2 * (b.2 - a.2)⟩ = ⟨d.1 - c.1, d.2 - c.2⟩

-- Define the parabola equation for point P
def parabola (x y : ℝ) : Prop :=
  y = (1 / 4) * x^2 - 3

-- Prove the maximum value of |AB| + |CD| given the conditions
theorem max_AB_CD
  (a b c d : ℝ × ℝ)
  (h1 : ellipse a.1 a.2)
  (h2 : ellipse b.1 b.2)
  (h3 : ellipse c.1 c.2)
  (h4 : ellipse d.1 d.2)
  (h5 : vector_eqn_of_chords a b c d) :
  dist a b + dist c d ≤ 6 :=
sorry

-- Prove the maximum area of the quadrilateral ABCD given the conditions
theorem max_area_ABCD
  (a b c d p : ℝ × ℝ)
  (h1 : ellipse a.1 a.2)
  (h2 : ellipse b.1 b.2)
  (h3 : ellipse c.1 c.2)
  (h4 : ellipse d.1 d.2)
  (h5 : vector_eqn_of_chords a b c d)
  (h6 : parabola p.1 p.2) :
  area_quadrilateral_ABCD a b c d = ((3 * real.sqrt 13 - 6) * real.sqrt (2 + 2 * real.sqrt 13) / 4) :=
sorry

-- Placeholder for calculating the area of a quadrilateral given its vertices
noncomputable def area_quadrilateral_ABCD (a b c d : ℝ × ℝ) : ℝ :=
sorry

end max_AB_CD_max_area_ABCD_l725_725500


namespace whale_sixth_hour_l725_725719

noncomputable def total_consumption (x : ℕ) : ℕ :=
  ∑ n in range 9, (x + n * 3)

def consumption_in_hour (x n : ℕ) : ℕ :=
  x + (n - 1) * 3

theorem whale_sixth_hour :
  ∀ x : ℕ, 
  total_consumption x = 360 → 
  consumption_in_hour x 6 = 43 :=
by
  assume x hx,
  sorry

end whale_sixth_hour_l725_725719


namespace intersection_of_sets_l725_725143

noncomputable def A : Set ℤ := {x | x^2 - 1 = 0}
def B : Set ℤ := {-1, 2, 5}

theorem intersection_of_sets : A ∩ B = {-1} :=
by
  sorry

end intersection_of_sets_l725_725143


namespace statement_B_l725_725604

variable (Student : Type)
variable (nora : Student)
variable (correctly_answered_all_math_questions : Student → Prop)
variable (received_at_least_B : Student → Prop)

theorem statement_B :
  (∀ s : Student, correctly_answered_all_math_questions s → received_at_least_B s) →
  (¬ received_at_least_B nora → ∃ q : Student, ¬ correctly_answered_all_math_questions q) :=
by
  intros h hn
  sorry

end statement_B_l725_725604


namespace gamma_plus_delta_eq_neg_third_l725_725960

noncomputable def reciprocal_sum_of_roots (a b c : ℚ) : ℚ :=
  let roots_pair := quadratic_roots a b c in
  let γ := 1 / roots_pair.1 in
  let δ := 1 / roots_pair.2 in
  γ + δ

theorem gamma_plus_delta_eq_neg_third : reciprocal_sum_of_roots 7 2 6 = -1 / 3 :=
by
  -- roots_pair represents (c, d)
  let roots_pair := quadratic_roots 7 2 6
  let c := roots_pair.1
  let d := roots_pair.2
  -- define γ and δ as per the condition
  let γ := 1 / c
  let δ := 1 / d
  -- using Vieta’s formulas:
  have : c + d = -2 / 7, from sorry
  have : c * d = 6 / 7, from sorry
  show γ + δ = -1 / 3, from sorry

end gamma_plus_delta_eq_neg_third_l725_725960


namespace cube_roll_sums_l725_725220

def opposite_faces_sum_to_seven (a b : ℕ) : Prop := a + b = 7

def valid_cube_faces : Prop := 
  opposite_faces_sum_to_seven 1 6 ∧
  opposite_faces_sum_to_seven 2 5 ∧
  opposite_faces_sum_to_seven 3 4

def max_min_sums : ℕ × ℕ := (342, 351)

theorem cube_roll_sums (faces_sum_seven : valid_cube_faces) : 
  ∃ cube_sums : ℕ × ℕ, cube_sums = max_min_sums := sorry

end cube_roll_sums_l725_725220


namespace ratio_even_to_odd_divisors_l725_725386

-- Define M
def M : ℕ := 126 * 36 * 187

-- Define sum of the even divisors for a given number n
def sum_even_divisors (n : ℕ) : ℕ :=
  (finset.filter (λ d, d % 2 = 0) (n.divisors)).sum id

-- Define sum of the odd divisors for a given number n
def sum_odd_divisors (n : ℕ) : ℕ :=
  (finset.filter (λ d, d % 2 = 1) (n.divisors)).sum id

-- Define the proof statement
theorem ratio_even_to_odd_divisors : sum_even_divisors M / sum_odd_divisors M = 14 := 
  by
    sorry

end ratio_even_to_odd_divisors_l725_725386


namespace parabola_distance_minimum_l725_725249

noncomputable def minimum_distance_sum (P : Point) (d1 d2 : ℝ) :=
  (d1 + d2)

theorem parabola_distance_minimum:
  ∀ (P : Point), P ∈ {P : ℝ × ℝ | P.2 ^ 2 = 4 * P.1} ->
  (d1 : ℝ), (d1 = distance P (Directrix)) ->
  (d2 : ℝ), (d2 = distance P ({x : ℝ × ℝ | x.1 - 2 * x.2 + 10 = 0})) ->
  minimum_distance_sum P d1 d2 = (11 * real.sqrt 5) / 5 :=
begin
  sorry

end parabola_distance_minimum_l725_725249


namespace circle_tangent_to_parabola_directrix_l725_725494

theorem circle_tangent_to_parabola_directrix (m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + m * x - 1/4 = 0 → y^2 = 4 * x → x = -1) → m = 3/4 :=
by
  sorry

end circle_tangent_to_parabola_directrix_l725_725494


namespace PA_AB_ratio_l725_725904

-- Definitions of the conditions
variables {A B C P: Type} [LinearOrderedField A]

-- Definition of points and lengths
variables (a b c p : A)
variable (ratio1 ratio2 : A)

-- Given conditions
def triangle_AC_CB_ratio (A B C : A) := ratio1 = 2 ∧ ratio2 = 3
def bisector_exterior_angle (A B C P : A) := (A < P) ∧ (P < B)

-- Proof goal
theorem PA_AB_ratio 
  (h1 : ratio1 = (2 : A)) 
  (h2 : ratio2 = (3 : A)) 
  (h3 : A < P) 
  (h4 : P < B) 
  : (PA / AB) = (2 / 1) := by
  sorry

end PA_AB_ratio_l725_725904


namespace average_of_roots_l725_725756

theorem average_of_roots (p q : ℝ) (h : ∀ r : ℝ, r^2 * (3 * p) + r * (-6 * p) + q = 0 → ∃ a b : ℝ, r = a ∨ r = b) : 
  ∀ (r1 r2 : ℝ), (3 * p) * r1^2 + (-6 * p) * r1 + q = 0 ∧ (3 * p) * r2^2 + (-6 * p) * r2 + q = 0 → 
  (r1 + r2) / 2 = 1 :=
by {
  sorry
}

end average_of_roots_l725_725756


namespace angle_DAC_is_100_l725_725209

theorem angle_DAC_is_100 
  (D A B C : Type)
  [EuclideanGeometry E D A B C]
  [EqGeom D A B C]
  (h1 : DA = CB)
  (h2 : ∠BAC = 70)
  (h3 : ∠ABC = 55) :
  ∠DAC = 100 :=
by
  sorry

end angle_DAC_is_100_l725_725209


namespace find_f_log20_l725_725099

open Real

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  a * sin ((x + 1) * π) + b * cbrt (x - 1) + 2

theorem find_f_log20 (a b : ℝ) (h : f (log 5) a b = 5) : f (log 20) a b = -1 := by
  sorry

end find_f_log20_l725_725099


namespace tomatoes_left_l725_725687

theorem tomatoes_left (initial_tomatoes : ℕ) (fraction_eaten : ℚ) (eaters : ℕ) (final_tomatoes : ℕ)  
  (h_initial : initial_tomatoes = 21)
  (h_fraction : fraction_eaten = 1 / 3)
  (h_eaters : eaters = 2)
  (h_final : final_tomatoes = initial_tomatoes - initial_tomatoes * fraction_eaten) :
  final_tomatoes = 14 := by
  sorry

end tomatoes_left_l725_725687


namespace exists_ab_negated_l725_725675

theorem exists_ab_negated :
  ¬ (∀ a b : ℝ, (a + b = 0 → a^2 + b^2 = 0)) ↔ 
  ∃ a b : ℝ, (a + b = 0 ∧ a^2 + b^2 ≠ 0) :=
by
  sorry

end exists_ab_negated_l725_725675


namespace maximize_points_l725_725263

def points (x : ℤ) : ℤ := Int.max 0 (8 - Int.natAbs (8 * x - 100))

theorem maximize_points : points 12 = 4 ∧ points 13 = 4 := by
  simp [points, Int.natAbs, Int.max, Int.sub]
  split
  · norm_num
  · norm_num
  done

end maximize_points_l725_725263


namespace total_volume_of_rainfall_l725_725045

theorem total_volume_of_rainfall (a : ℝ) (r1 r2 t1 t2 : ℝ) (h1 : r1 = 5 * 0.001) (h2 : r2 = 10 * 0.001)
(ph1 : t1 = 1) (ph2 : t2 = 1) (pa : a = 100) : 
  (r1 * t1 * a + r2 * t2 * a = 1.5) :=
by
  -- rainfall rate converted to meters per hour
  have r1m : r1 = 0.005 := by rw [h1, mul_comm, mul_assoc, mul_comm 0.001]
  have r2m : r2 = 0.01 := by rw [h2, mul_comm, mul_assoc, mul_comm 0.001]

  -- time in hours
  have t1h : t1 = 1 := ph1
  have t2h : t2 = 1 := ph2

  -- area in square meters
  have area : a = 100 := pa

  -- volumes for each hour
  have vol1 : (r1 * t1 * a = 0.5) := by rw [r1m, t1h, area]; norm_num
  have vol2 : (r2 * t2 * a = 1) := by rw [r2m, t2h, area]; norm_num

  -- total volume
  rw [vol1, vol2]; norm_num

-- sorry -- Removing 'sorry' as we actually completed the proof for descriptive purposes

end total_volume_of_rainfall_l725_725045


namespace length_of_BC_l725_725551

theorem length_of_BC (A B C M : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M]
  (AB AC BC AM : ℝ)
  (h1 : AB = 2)
  (h2 : AC = 3)
  (h3 : BC = 2 * AM)
  (h4 : ∠BAC = 120) :
  BC = 10 := by
  sorry

end length_of_BC_l725_725551


namespace angle_terminal_side_equiv_l725_725996

theorem angle_terminal_side_equiv (k : ℤ) : 
  ∀ θ α : ℝ, θ = - (π / 3) → α = 5 * π / 3 → α = θ + 2 * k * π := by
  intro θ α hθ hα
  sorry

end angle_terminal_side_equiv_l725_725996


namespace purchase_price_mobile_l725_725637

-- Definitions of the given conditions
def purchase_price_refrigerator : ℝ := 15000
def loss_percent_refrigerator : ℝ := 0.05
def profit_percent_mobile : ℝ := 0.10
def overall_profit : ℝ := 50

-- Defining the statement to prove
theorem purchase_price_mobile (P : ℝ)
  (h1 : purchase_price_refrigerator = 15000)
  (h2 : loss_percent_refrigerator = 0.05)
  (h3 : profit_percent_mobile = 0.10)
  (h4 : overall_profit = 50) :
  (15000 * (1 - 0.05) + P * (1 + 0.10)) - (15000 + P) = 50 → P = 8000 :=
by {
  -- Proof is omitted
  sorry
}

end purchase_price_mobile_l725_725637


namespace find_q_l725_725887

theorem find_q 
  (p q : ℝ) 
  (h1 : 1 < p) 
  (h2 : p < q)
  (h3 : 1 / p + 1 / q = 3 / 2) 
  (h4 : p * q = 12) : 
  q = 9 + 3 * real.sqrt 23 := 
sorry

end find_q_l725_725887


namespace significant_improvement_l725_725738

-- Definition of experiment data
def experiment_data (x y : Fin 10 → ℝ) : Prop :=
  x = ![545, 533, 551, 522, 575, 544, 541, 568, 596, 548] ∧
  y = ![536, 527, 543, 530, 560, 533, 522, 550, 576, 536]

-- Definition of z_i
def z (x y : Fin 10 → ℝ) (i : Fin 10) : ℝ := x i - y i

-- Sample mean of z
def mean_z (z : Fin 10 → ℝ) : ℝ := (1 / 10) * ∑ i, z i

-- Sample variance of z
def variance_z (z : Fin 10 → ℝ) : ℝ := (1 / 10) * ∑ i, (z i - mean_z z)^2

-- Proof problem to check significant improvement
theorem significant_improvement (x y : Fin 10 → ℝ)
  (h_data : experiment_data x y) :
  mean_z (z x y) ≥ 2 * (Real.sqrt (variance_z (z x y) / 10)) :=
by
  sorry

end significant_improvement_l725_725738


namespace tangent_lines_through_A_area_triangle_AOC_l725_725487

-- Given circle C: x^2 + y^2 - 4x - 6y + 12 = 0
def circle (x y : ℝ) := x^2 + y^2 - 4 * x - 6 * y + 12 = 0

-- Given point A(3, 5)
def point_A : ℝ × ℝ := (3, 5)

-- The equation of the tangent lines passing through point A are 3x - 4y + 11 = 0 and x = 3
theorem tangent_lines_through_A :
  (∀ x y : ℝ, circle x y → 3 * x - 4 * y + 11 = 0 ∨ x = 3 ∧ 3 ≤ y) := 
sorry

-- O is the coordinate origin, point O (0, 0)
def point_O : ℝ × ℝ := (0, 0)

-- Center of the circle C is (2, 3)
def center_C : ℝ × ℝ := (2, 3)

-- The area S of triangle AOC is 1/2
theorem area_triangle_AOC : 
  let A := point_A in
  let O := point_O in
  let C := center_C in
  ∀ (S : ℝ), S = 1/2  := sorry

end tangent_lines_through_A_area_triangle_AOC_l725_725487


namespace total_cost_in_euros_after_discount_l725_725565

  theorem total_cost_in_euros_after_discount:
    (crayons_bought: ℕ) 
    (price_per_crayon_usd: ℝ) 
    (discount_rate: ℝ) 
    (exchange_rate: ℝ) 
    (total_cost_eur: ℝ): 
    crayons_bought = 4 * 6 → 
    price_per_crayon_usd = 2 → 
    discount_rate = 0.10 → 
    exchange_rate = 0.85 → 
    total_cost_eur = (4 * 6 * 2 * (1 - 0.10)) * 0.85 :=
  sorry
  
end total_cost_in_euros_after_discount_l725_725565


namespace imaginary_part_of_z_l725_725098

-- Given condition
def z : ℂ := (2 - complex.i) ^ 2

-- The proof problem statement
theorem imaginary_part_of_z : complex.im z = -4 :=
by sorry

end imaginary_part_of_z_l725_725098


namespace min_area_of_ellipse_containing_circles_l725_725036

noncomputable def ellipsoid_min_area (a b: ℝ) : ℝ := π * a * b

theorem min_area_of_ellipse_containing_circles :
  ∃ a b : ℝ, 
    (∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ∧ ((x - 2)^2 + y^2 = 4) ∧ ((x + 2)^2 + y^2 = 4) → 
    ellipsoid_min_area a b = π * (9 * Real.sqrt 3 / 4)) :=
by 
  sorry 

end min_area_of_ellipse_containing_circles_l725_725036


namespace probability_of_6_consecutive_heads_l725_725350

/-- Define the probability of obtaining at least 6 consecutive heads in 10 flips of a fair coin. -/
def prob_at_least_6_consecutive_heads : ℚ :=
  129 / 1024

/-- Proof statement: The probability of getting at least 6 consecutive heads in 10 flips of a fair coin is 129/1024. -/
theorem probability_of_6_consecutive_heads : 
  prob_at_least_6_consecutive_heads = 129 / 1024 := 
by
  sorry

end probability_of_6_consecutive_heads_l725_725350


namespace problem_result_l725_725737

def elongation_A : List ℕ := [545, 533, 551, 522, 575, 544, 541, 568, 596, 548]
def elongation_B : List ℕ := [536, 527, 543, 530, 560, 533, 522, 550, 576, 536]

def z_i : List ℤ := List.zipWith (λ x y => x - y) elongation_A elongation_B

def sample_mean (lst : List ℤ) : ℚ :=
  (List.sum lst : ℚ) / List.length lst

def sample_variance (lst : List ℤ) : ℚ :=
  let mean := sample_mean lst
  (List.sum (lst.map (λ z => (z - mean) * (z - mean))) : ℚ) / List.length lst

def improvement_significance (mean : ℚ) (variance : ℚ) : Prop :=
  mean ≥ 2 * Real.sqrt (variance / 10)

theorem problem_result :
  sample_mean z_i = 11 ∧
  sample_variance z_i = 61 ∧
  improvement_significance (sample_mean z_i) (sample_variance z_i) :=
by
  sorry

end problem_result_l725_725737


namespace total_daisies_l725_725560

theorem total_daisies (white pink red : ℕ) (h1 : pink = 9 * white) (h2 : red = 4 * pink - 3) (h3 : white = 6) : 
    white + pink + red = 273 :=
by
  sorry

end total_daisies_l725_725560


namespace sandwiches_left_l725_725968

theorem sandwiches_left (S G K L : ℕ) (h1 : S = 20) (h2 : G = 4) (h3 : K = 2 * G) (h4 : L = S - G - K) : L = 8 :=
sorry

end sandwiches_left_l725_725968


namespace maximum_students_l725_725766

theorem maximum_students (x : ℕ) (hx : x / 2 + x / 4 + x / 7 + 6 > x) : x ≤ 28 :=
by sorry

end maximum_students_l725_725766


namespace Total_toys_l725_725776

-- Definitions from the conditions
def Mandy_toys : ℕ := 20
def Anna_toys : ℕ := 3 * Mandy_toys
def Amanda_toys : ℕ := Anna_toys + 2

-- The statement to be proven
theorem Total_toys : Mandy_toys + Anna_toys + Amanda_toys = 142 :=
by
  -- Add proof here
  sorry

end Total_toys_l725_725776


namespace a_finish_work_alone_in_4_days_l725_725000

theorem a_finish_work_alone_in_4_days
  (a : ℕ) (ha : 0 < a)
  (b_can_finish_in_8_days : 8 > 0)
  (work_done_together : (2 * (1 / a + 1 / 8) : ℚ))
  (b_finishes_remaining_work : (2 * (1 / 8) : ℚ))
  (total_work : work_done_together + b_finishes_remaining_work = 1) :
  a = 4 :=
by
  sorry

end a_finish_work_alone_in_4_days_l725_725000


namespace largest_prime_divisor_to_test_l725_725697

theorem largest_prime_divisor_to_test (n : ℕ) (hn : 900 ≤ n ∧ n ≤ 950) : 29 = (Prime.max_divisor (sqrt n)) :=
by
  sorry

end largest_prime_divisor_to_test_l725_725697


namespace find_x_l725_725890

-- Define the vectors a, b, and c
def a : ℝ × ℝ := (-2, 0)
def b : ℝ × ℝ := (2, 1)
def c (x : ℝ) : ℝ × ℝ := (x, 1)

-- Define the collinearity condition
def collinear_with_3a_plus_b (x : ℝ) : Prop :=
  ∃ k : ℝ, c x = k • (3 • a + b)

theorem find_x :
  ∀ x : ℝ, collinear_with_3a_plus_b x → x = -4 := 
sorry

end find_x_l725_725890


namespace sufficient_not_necessary_example_l725_725852

lemma sufficient_but_not_necessary_condition (x y : ℝ) (hx : x >= 2) (hy : y >= 2) : x^2 + y^2 >= 4 :=
by
  -- We only need to state the lemma, so the proof is omitted.
  sorry

theorem sufficient_not_necessary_example :
  ¬(∀ x y : ℝ, (x^2 + y^2 >= 4) -> (x >= 2) ∧ (y >= 2)) :=
by 
  -- We only need to state the theorem, so the proof is omitted.
  sorry

end sufficient_not_necessary_example_l725_725852


namespace sqrt_9_minus_2_pow_0_plus_abs_neg1_l725_725046

theorem sqrt_9_minus_2_pow_0_plus_abs_neg1 :
  (Real.sqrt 9 - 2^0 + abs (-1) = 3) :=
by
  -- Proof omitted for brevity
  sorry

end sqrt_9_minus_2_pow_0_plus_abs_neg1_l725_725046


namespace even_digit_sequence_count_l725_725152

theorem even_digit_sequence_count : 
  ∃ (n : ℕ), 
  (∀ (d : ℕ), d ∈ {0, 2, 4, 6, 8}) ∧ -- Each digit d must be from the set of even digits
  (∀ (i : ℕ), 0 ≤ i ∧ i < 99 → |digits_list.nth i - digits_list.nth (i + 1)| = 2) ∧ -- The difference between any two adjacent digits is 2
  n = 7 * 3^49 -- The number of such sequences is 7 * 3^49
:= sorry

end even_digit_sequence_count_l725_725152


namespace necessary_condition_l725_725848

noncomputable def quadratic_inequality_holds (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + 2 * a * x + 4 ≥ 0

noncomputable def log_function_positive (a : ℝ) : Prop :=
  ∀ x : ℝ, 1 < x → log a (x - a + 2) > 0

theorem necessary_condition (a : ℝ) :
  quadratic_inequality_holds a → log_function_positive a :=
sorry

end necessary_condition_l725_725848


namespace area_bounded_by_graphs_eq_4_l725_725417

theorem area_bounded_by_graphs_eq_4 :
  let r₁ (θ : ℝ) := 2 / (cos θ)
  let r₂ (θ : ℝ) := 2 / (sin θ)
  ∀ (θ₁ θ₂ : ℝ) (x ℝ: ℝ), 0 ≤ θ₁ ∧ θ₁ ≤ π/2 ∧ 0 ≤ θ₂ ∧ θ₂ ≤ π/2 ∧
  x = r₁ θ₁ ∧ y = r₂ θ₂ →
  area (bounded_region r₁ r₂ 0 0) = 4 := by
  sorry

end area_bounded_by_graphs_eq_4_l725_725417


namespace problem_divisibility_l725_725250

open Set Function

noncomputable def probability_divisible_by_5 : ℚ :=
  let S := Finset.range 2021
  let count := S.filter (λ a, (a % 5 = 0)).card
  let P_a_div_5 := (count : ℚ) / (2020 : ℚ)
  let P_internal := (1 / 5 : ℚ)
  P_a_div_5 + P_internal * (1 - P_a_div_5)

theorem problem_divisibility : probability_divisible_by_5 = 9 / 25 :=
sorry

end problem_divisibility_l725_725250


namespace powers_of_5_mod_9_l725_725302

theorem powers_of_5_mod_9 :
  (∑ i in Finset.range 2011, 5^i) % 9 = 7 :=
by
  sorry

end powers_of_5_mod_9_l725_725302


namespace cannot_make_all_divisible_by_10_l725_725906

def natural_grid_increments (grid : Fin 8 → Fin 8 → ℕ) (select3x3 select4x4 : list (Fin 8 × Fin 8)) : Fin 8 → Fin 8 → ℕ :=
  let increment_block (grid : Fin 8 → Fin 8 → ℕ) (block : list (Fin 8 × Fin 8)) :=
    block.foldl (fun g c => fun x y => if (x, y) ∈ block then g x y + 1 else g x y) grid in
  ((select3x3.foldl increment_block grid)).foldl increment_block

theorem cannot_make_all_divisible_by_10 (initial_grid : Fin 8 → Fin 8 → ℕ) :
  ¬ (∃ (select3x3 select4x4 : list (Fin 8 × Fin 8)), ∀ x y, (natural_grid_increments initial_grid select3x3 select4x4 x y) % 10 = 0) :=
sorry

end cannot_make_all_divisible_by_10_l725_725906


namespace doctor_team_combinations_l725_725256

theorem doctor_team_combinations :
  let males := 5
      females := 4
      total_doctors := males + females
  in (∑ k in (finset.range 4), if k = 0 ∨ k = 3 then 0 else nat.choose males k * nat.choose females (3 - k)) = 70 := 
by
  sorry

end doctor_team_combinations_l725_725256


namespace polygon_perimeter_eq_l725_725230

noncomputable def minimal_perimeter (Q : ℂ → ℂ) : ℝ :=
  20 * real.cbrt (abs (6 * real.sqrt 2 + 11)) * real.sin (real.pi / 5)

theorem polygon_perimeter_eq :
  let Q (z : ℂ) := z^10 + (6 * complex.sqrt 2 + 10) * z^5 - (6 * complex.sqrt 2 + 11)
  in minimal_perimeter Q = 20 * real.cbrt (abs (6 * real.sqrt 2 + 11)) * real.sin (real.pi / 5) :=
sorry

end polygon_perimeter_eq_l725_725230


namespace bret_total_spend_l725_725384

/-- Bret and his team are working late along with another team of 4 co-workers.
He decides to order dinner for everyone. -/

def team_A : ℕ := 4 -- Bret’s team
def team_B : ℕ := 4 -- Other team

def main_meal_cost : ℕ := 12
def team_A_appetizers_cost : ℕ := 2 * 6  -- Two appetizers at $6 each
def team_B_appetizers_cost : ℕ := 3 * 8  -- Three appetizers at $8 each
def sharing_plates_cost : ℕ := 4 * 10    -- Four sharing plates at $10 each

def tip_percentage : ℝ := 0.20           -- Tip is 20%
def rush_order_fee : ℕ := 5              -- Rush order fee is $5
def sales_tax : ℝ := 0.07                -- Local sales tax is 7%

def total_cost_without_tip_and_tax : ℕ :=
  team_A * main_meal_cost + team_B * main_meal_cost + team_A_appetizers_cost +
  team_B_appetizers_cost + sharing_plates_cost

def total_cost_with_tip : ℝ :=
  total_cost_without_tip_and_tax + 
  (tip_percentage * total_cost_without_tip_and_tax)

def total_cost_before_tax : ℝ :=
  total_cost_with_tip + rush_order_fee

def final_total_cost : ℝ :=
  total_cost_before_tax + (sales_tax * total_cost_with_tip)


theorem bret_total_spend : final_total_cost = 225.85 := by
  sorry

end bret_total_spend_l725_725384


namespace trig_identity_l725_725849

theorem trig_identity (x : ℝ) (h : 2 * Real.cos x - 5 * Real.sin x = 3) :
  (Real.sin x + 2 * Real.cos x = 1 / 2) ∨ (Real.sin x + 2 * Real.cos x = 83 / 29) := sorry

end trig_identity_l725_725849


namespace ratio_between_two_numbers_l725_725013

noncomputable def first_number : ℕ := 48
noncomputable def lcm_value : ℕ := 432
noncomputable def second_number : ℕ := 9 * 24  -- Derived from the given conditions in the problem

def ratio (a b : ℕ) : ℚ := (a : ℚ) / (b : ℚ)

theorem ratio_between_two_numbers 
  (A B : ℕ) 
  (hA : A = first_number) 
  (hLCM : Nat.lcm A B = lcm_value) 
  (hB : B = 9 * 24) : 
  ratio A B = 1 / 4.5 :=
by
  -- Proof would go here
  sorry

end ratio_between_two_numbers_l725_725013


namespace at_least_6_heads_in_10_flips_l725_725339

def coin_flip : Type := bool

def is_heads (x : coin_flip) : Prop := x = tt

def num_consecutive_heads (l : list coin_flip) (n : ℕ) : Prop :=
  ∃ i : ℕ, i + n ≤ l.length ∧ l.drop i.take n = list.replicate n tt

def prob_at_least_n_consecutive_heads (l : list coin_flip) (n : ℕ) : Prop :=
  ∃ i ≤ l.length - n + 1, list.replicate n tt = l.drop (i - 1).take n

noncomputable def at_least_6_heads_in_10_flips_prob : ℚ :=
  (129:ℚ) / (1024:ℚ)

theorem at_least_6_heads_in_10_flips :
  prob_at_least_n_consecutive_heads (list.replicate 10 coin_flip) 6 = at_least_6_heads_in_10_flips_prob :=
by
  sorry

end at_least_6_heads_in_10_flips_l725_725339


namespace find_x_l725_725531

theorem find_x (x : ℕ) (h : 1 + 2 + 3 + 4 + 5 + x = 21 + 22 + 23 + 24 + 25) : x = 100 :=
by {
  sorry
}

end find_x_l725_725531


namespace mark_receives_right_amount_of_pennies_l725_725965

-- Defining the costs of goods
def bread_price := 4.79
def cheese_price := 6.55
def milk_price := 3.85
def strawberries_price := 2.15

-- Defining the quantity of each item Mark buys
def bread_qty := 3
def cheese_qty := 2
def milk_qty := 6
def strawberries_qty := 4

-- Total amount given by Mark
def amount_given := 100.00

-- Sales tax rate
def tax_rate := 0.065

-- Coins available with the cashier
def num_quarters := 5
def num_dimes := 10
def num_nickels := 15

-- Total costs for each item
def total_cost (quantity: ℕ) (price: ℝ) : ℝ := quantity * price
def bread_total := total_cost bread_qty bread_price
def cheese_total := total_cost cheese_qty cheese_price
def milk_total := total_cost milk_qty milk_price
def strawberries_total := total_cost strawberries_qty strawberries_price

-- Subtotal before tax
def subtotal := bread_total + cheese_total + milk_total + strawberries_total

-- Sales tax
def sales_tax := subtotal * tax_rate

-- Total cost including tax
def total_cost_with_tax := subtotal + sales_tax

-- Change due
def change := amount_given - total_cost_with_tax

-- Value of coins
def value_of_quarters := num_quarters * 0.25
def value_of_dimes := num_dimes * 0.10
def value_of_nickels := num_nickels * 0.05

-- Remaining change after using quarters, dimes, and nickels
def remaining_change := change - (value_of_quarters + value_of_dimes + value_of_nickels)

-- Number of pennies Mark receives
def num_pennies := remaining_change / 0.01

-- Theorems to prove
theorem mark_receives_right_amount_of_pennies : num_pennies = 3398 :=
by
  -- Proof would go here
  sorry

end mark_receives_right_amount_of_pennies_l725_725965


namespace arctan_sum_pi_over_4_l725_725954

noncomputable def poly (x : ℝ) : ℝ := x^3 - 10 * x + 11

def are_roots (x1 x2 x3 : ℝ) : Prop := 
  poly x1 = 0 ∧
  poly x2 = 0 ∧
  poly x3 = 0 ∧
  x1 + x2 + x3 = 0 ∧
  x1 * x2 + x2 * x3 + x3 * x1 = -10 ∧
  x1 * x2 * x3 = -11

theorem arctan_sum_pi_over_4 {x1 x2 x3 : ℝ} (h : are_roots x1 x2 x3) : 
  arctan x1 + arctan x2 + arctan x3 = π / 4 :=
sorry

end arctan_sum_pi_over_4_l725_725954


namespace find_line_and_circle_l725_725519

-- Define the given lines
def line1 (x y : ℝ) := 2 * x - y = 0
def line2 (x y : ℝ) := x + y + 2 = 0

-- Define the point P
def pointP := (1, 1 : ℝ × ℝ)

-- Define the line l
def line3 (x y : ℝ) := x + 2 * y - 3 = 0

-- Define circle conditions
def circle_eqn (a b r x y : ℝ) := (x - a)^2 + (y - b)^2 = r^2
def l1_condition (a b : ℝ) := 2 * a - b = 0
def tangent_y_axis_condition (a r : ℝ) := r = abs a
def chord_length_condition (a b r : ℝ) := (abs (a + b + 2) / (sqrt 2))^2 + (sqrt 2 / 2)^2 = r^2

-- Equivalent Lean Theorem Statement
theorem find_line_and_circle : 
  (∀ (x y : ℝ), line3 x y ↔ line1 1 1 = 0 ∧ y - 1 = -1/2 * (x - 1)) ∧
  (∃ (a b r : ℝ), 
    (circle_eqn a b r = (x → (x + 5/7)^2 + (y → y + 10/7)^2 = (5/7)^2) ∨ 
     circle_eqn a b r = (x → (x + 1)^2 + (y → y + 2)^2 = 1)) ∧
    l1_condition a b ∧
    tangent_y_axis_condition a r ∧
    chord_length_condition a b r) :=
begin
  sorry
end

end find_line_and_circle_l725_725519


namespace max_mark_is_500_l725_725022

-- Definition of the conditions
def passes_if_marks_ge (marks_needed : ℝ) (total_marks : ℝ) : Prop :=
  marks_needed >= 0.33 * total_marks

def marks_received : ℝ := 125
def marks_failed_by : ℝ := 40

-- Definition of the total marks
def total_marks (marks_received : ℝ) (marks_failed_by : ℝ) : ℝ :=
  marks_received + marks_failed_by

-- Total marks should be equal to 500
theorem max_mark_is_500 (marks_received marks_failed_by : ℝ) :
  0.33 * (total_marks marks_received marks_failed_by) = marks_received + marks_failed_by →
  total_marks marks_received marks_failed_by = 500 :=
by
  intros h
  have pass_marks : ℝ := marks_received + marks_failed_by
  have h2 : 0.33 * pass_marks = pass_marks := h
  sorry

end max_mark_is_500_l725_725022


namespace obliquely_cut_cylinder_l725_725743

variables (r a b : ℝ)

-- Define the expressions for volume and lateral surface area
def volume (r a b : ℝ) : ℝ := r^2 * Real.pi * (a + b) / 2
def lateral_surface_area (r a b : ℝ) : ℝ := Real.pi * r * (a + b)

-- The theorem statement, no proof necessary
theorem obliquely_cut_cylinder (r a b : ℝ) :
  volume r a b = r^2 * Real.pi * (a + b) / 2 ∧
  lateral_surface_area r a b = Real.pi * r * (a + b) :=
by
  -- Proof would go here
  sorry

end obliquely_cut_cylinder_l725_725743


namespace floor_eq_solution_l725_725054

theorem floor_eq_solution (a b : ℝ) :
  (∀ n : ℕ, 0 < n → a * (floor (b * n)) = b * (floor (a * n))) ↔ (a = b ∨ a = 0 ∨ b = 0) := by
  sorry

end floor_eq_solution_l725_725054


namespace angle_AMB_eq_angle_DMC_l725_725246

-- Define the geometric setup
variables {A B C D M : Type} [EuclideanGeometry A]
variables [Parallelogram A B C D]
variables (hMAB : Angle (M, A, B) = Angle (M, C, B))
variables (h1 : Triangle M A B)
variables (h2 : Triangle M C B)

-- Prove the required theorem
theorem angle_AMB_eq_angle_DMC : ∠ AMB = ∠ DMC :=
begin
  sorry
end

end angle_AMB_eq_angle_DMC_l725_725246


namespace proof_problem_l725_725713

noncomputable def proof_statement : Prop :=
  ∀ (a b : Type), (0 ∈ ({0} : Set ℕ)) ∧ (∅ ⊆ ({0} : Set ℕ)) ∧ ¬(({0, 1} : Set ℕ) ⊆ ({(0,1)} : Set (ℕ × ℕ))) ∧ ¬ (({(a,b)} : Set (Type × Type)) = ({(b,a)} : Set (Type × Type)))

theorem proof_problem : proof_statement :=
by
  sorry

end proof_problem_l725_725713


namespace find_AB2_AC2_BC2_l725_725222

variables {A B C G : Type} [field A] [metric_space A]

-- Define the points and the distances
def centroid (a b c : A) : A := (a + b + c) / 3
def dist_sq (x y : A) : B := norm (y - x)^2

-- Given conditions
axiom angle_BAC_90 (A B C : A) : angle B A C = 90
axiom sum_distances_squared (a b c : A) : 
  let g := centroid a b c in dist_sq g a + dist_sq g b + dist_sq g c = 72
axiom right_triangle (A B C : A) : dist_sq A B + dist_sq A C = dist_sq B C

-- The proof statement
theorem find_AB2_AC2_BC2 (A B C : A) :
  dist_sq A B + dist_sq A C + dist_sq B C = 108 :=
by
  sorry

end find_AB2_AC2_BC2_l725_725222


namespace cost_of_milkshake_is_correct_l725_725932

-- Definitions related to the problem conditions
def initial_amount : ℕ := 15
def spent_on_cupcakes : ℕ := initial_amount * (1 / 3)
def remaining_after_cupcakes : ℕ := initial_amount - spent_on_cupcakes
def spent_on_sandwich : ℕ := remaining_after_cupcakes * (20 / 100)
def remaining_after_sandwich : ℕ := remaining_after_cupcakes - spent_on_sandwich
def remaining_after_milkshake : ℕ := 4
def cost_of_milkshake : ℕ := remaining_after_sandwich - remaining_after_milkshake

-- The theorem stating the equivalent proof problem
theorem cost_of_milkshake_is_correct :
  cost_of_milkshake = 4 :=
sorry

end cost_of_milkshake_is_correct_l725_725932


namespace angle_C_is_pi_div_6_area_of_triangle_ABC_l725_725207

theorem angle_C_is_pi_div_6 (A B C a b : ℝ) (ha_neq_b : a ≠ b) :
  ∀ (h1: ∀ (cosA cosB sinA sinB : ℝ), c = sqrt 3 → (sqrt 3) * (cosA ^ 2) - (sqrt 3) * (cosB ^ 2) = sinA * cosA - sinB * cosB )
  , c = sqrt 3 -> sin A = 4/5 -> ∃ C, C = π / 6 := 
  begin
    sorry
  end

theorem area_of_triangle_ABC (A B C a b : ℝ) (ha_neq_b : a ≠ b) :
  ∀ (h1: ∀ (cosA cosB sinA sinB : ℝ), c = sqrt 3 → (sqrt 3) * (cosA ^ 2) - (sqrt 3) * (cosB ^ 2) = sinA * cosA - sinB * cosB )
  , c = sqrt 3 -> sin A = 4/5 -> ∃ area, area = ((24 * sqrt 3) + 18) / 25 := 
  begin
    sorry
  end

end angle_C_is_pi_div_6_area_of_triangle_ABC_l725_725207


namespace polynomial_coeff_sum_eq_neg_two_l725_725896

/-- If (1 - 2 * x) ^ 9 = a₉ * x ^ 9 + a₈ * x ^ 8 + ... + a₂ * x ^ 2 + a₁ * x + a₀, 
then a₁ + a₂ + ... + a₈ + a₉ = -2. -/
theorem polynomial_coeff_sum_eq_neg_two 
  (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℤ) 
  (h : (1 - 2 * x) ^ 9 = a₉ * x ^ 9 + a₈ * x ^ 8 + a₇ * x ^ 7 + a₆ * x ^ 6 + a₅ * x ^ 5 + a₄ * x ^ 4 + a₃ * x ^ 3 + a₂ * x ^ 2 + a₁ * x + a₀) : 
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = -2 :=
by sorry

end polynomial_coeff_sum_eq_neg_two_l725_725896


namespace maximum_value_proof_l725_725813

noncomputable def maximum_value : ℝ :=
  (∃ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ x * y * (x - y) = 1/4)

theorem maximum_value_proof : ∀ (x y : ℝ), (0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1) → x * y * (x - y) ≤ 1/4 := by
  sorry

end maximum_value_proof_l725_725813


namespace area_of_region_l725_725423

noncomputable def sec (θ : ℝ) := (cos θ)⁻¹
noncomputable def csc (θ : ℝ) := (sin θ)⁻¹

def region (r θ : ℝ) : Prop :=
  (r = 2 * sec θ ∧ (0 ≤ θ ∧ θ ≤ π / 2)) ∨ 
  (r = 2 * csc θ ∧ (0 ≤ θ ∧ θ ≤ π / 2))

theorem area_of_region :
  let bounded_region := { (x, y) | 0 ≤ x ∧ x ≤ 2 ∧ 0 ≤ y ∧ y ≤ 2 } in
  ∃ (A : ℝ), A = 4 ∧ (∀ (a b : ℝ), bounded_region (a, b)) :=
begin
  let bounded_region := { p | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2 },
  use 4,
  split,
  { refl, },
  { intros a b hb,
    exact hb, },
end

end area_of_region_l725_725423


namespace circle_through_point_and_same_center_l725_725079

theorem circle_through_point_and_same_center :
  ∃ (x_0 y_0 r : ℝ),
    (∀ (x y : ℝ), (x - x_0)^2 + (y - y_0)^2 = r^2 ↔
      x^2 + y^2 - 4 * x + 6 * y - 3 = 0)
    ∧
    ∀ (x y : ℝ), (x - x_0)^2 + (y - y_0)^2 = r^2 ↔
      (x - 2)^2 + (y + 3)^2 = 25 := sorry

end circle_through_point_and_same_center_l725_725079


namespace area_of_bounded_region_l725_725435

theorem area_of_bounded_region : 
  let x1 := 2
  let y1 := 2
  let x0 := 0
  let y0 := 0
  in
  let area := (x1 - x0) * (y1 - y0)
  in
  area = 4 :=
by 
  -- definitions for x1, y1, x0, y0
  let x1 := 2
  let y1 := 2
  let x0 := 0
  let y0 := 0
  
  -- let area be the area of the square bounded by these lines
  let area := (x1 - x0) * (y1 - y0)
  
  -- assertion
  have h : area = (2 - 0) * (2 - 0), from rfl,
  
  -- proving the final statement
  show area = 4, by 
    rw [h],
    exact rfl

-- skipped proof step
sorry

end area_of_bounded_region_l725_725435


namespace determine_n_l725_725805

theorem determine_n (n : ℕ) : (2 : ℕ)^n = 2 * 4^2 * 16^3 ↔ n = 17 := 
by
  sorry

end determine_n_l725_725805


namespace probability_of_non_defective_is_0_92_l725_725361

-- Definitions of given conditions
def P_GradeB := 0.05
def P_GradeC := 0.03
def P_Defective := P_GradeB + P_GradeC
def P_GradeA := 1 - P_Defective

-- The theorem we need to prove
theorem probability_of_non_defective_is_0_92 : P_GradeA = 0.92 :=
by 
  unfold P_GradeA P_Defective P_GradeB P_GradeC
  -- This step just simplifies the definitions to show the desired equality
  calc
    1 - (0.05 + 0.03) = 1 - 0.08   : by sorry
                      ... = 0.92   : by sorry

end probability_of_non_defective_is_0_92_l725_725361


namespace find_PR_l725_725585

-- Definition of the right-angled triangle \(\triangle PQR\) and midpoints A and B
variables {P Q R A B : Type}
variables [RightAngledTriangle P Q R]
variables [Midpoint A P Q]
variables [Midpoint B P R]

-- Given conditions
variables (QA RB PR : ℝ)
variables (QA_eq_25 : QA = 25)
variables (RB_eq_15 : RB = 15)

-- Goal: Find PR
theorem find_PR : 
  ∃ (x y : ℝ), 
  QA = sqrt (4 * x^2 + y^2) ∧ 
  RB = sqrt (x^2 + 4 * y^2) ∧ 
  QA = 25 ∧ 
  RB = 15 ∧ 
  PR = 2 * sqrt (55 / 3) :=
by {
  have h1 : QA^2 = 4 * x^2 + y^2, sorry,
  have h2 : RB^2 = x^2 + 4 * y^2, sorry,
  have h3 : QA = 25, exact QA_eq_25,
  have h4 : RB = 15, exact RB_eq_15,
  have h5 : 4 * x^2 + y^2 = 625, sorry,
  have h6 : x^2 + 4 * y^2 = 225, sorry,
  have h7 : 4 * x^2 + y^2 + x^2 + 4 * y^2 = 625 + 225, sorry,
  have h8 : 5 * (x^2 + y^2) = 850, sorry,
  have h9 : x^2 + y^2 = 170, sorry,
  have h10 : x^2 = 170 - y^2, sorry,
  have h11 : 170 - y^2 + 4 * y^2 = 225, sorry,
  have h12 : 3 * y^2 = 55, sorry,
  have h13 : y^2 = 55 / 3, sorry,
  have h14 : y = sqrt (55 / 3), sorry,
  have h15 : PR = 2 * sqrt (55 / 3), sorry,
  use [x, y],
  tauto
}

end find_PR_l725_725585


namespace calc_f_7_2_l725_725112

variable {f : ℝ → ℝ}

axiom f_odd : ∀ x, f (-x) = -f x
axiom f_periodic : ∀ x, f (x + 2) = f x
axiom f_sqrt_on_interval : ∀ x, 0 < x ∧ x ≤ 1 → f x = Real.sqrt x

theorem calc_f_7_2 : f (7 / 2) = -Real.sqrt 2 / 2 := by
  sorry

end calc_f_7_2_l725_725112


namespace triangle_BED_area_l725_725212

noncomputable def area_triangle_bed (A B C M D E : Type)
  [Point A] [Point B] [Point C] [Midpoint M (LineSeg A B)]
  [Perp M D (Line BC)] [Perp E C (Line BC)] :=
  let area_ABC := 36
  let ac := 2 * area_ABC
  ac / 8

theorem triangle_BED_area (A B C M D E : Type) 
  [Point A] [Point B] [Point C] [Midpoint M (LineSeg A B)]
  [RightAngle C (LineSeg A B)] [Midpoint M (LineSeg A B)]
  [Perp M D (Line BC)] [Perp E C (Line BC)]
  (area_ABC : ℝ) (h : area_ABC = 36) :
  area_triangle_bed A B C M D E = 9 :=
sorry

end triangle_BED_area_l725_725212


namespace inscribed_circle_sector_l725_725002

-- Let's define the relevant variables and conditions
variables (r a b : ℝ)
-- Condition: a circle with radius r is inscribed in a sector with radius a and a chord of length 2b.
-- Prove the relationship
theorem inscribed_circle_sector (h1 : a > 0) (h2 : r > 0) (h3 : b > 0) (h4 : 2 * b = 2 * a * Real.sin (Real.acos ((a - r) / a) / 2)):
  1 / r = 1 / a + 1 / b :=
begin
  -- Proof is skipped
  sorry
end

end inscribed_circle_sector_l725_725002


namespace area_of_triangle_ABC_l725_725300

open Real

theorem area_of_triangle_ABC :
  ∃ (ABC : Type) (A B C K : ABC → Prop),
  (AC = 15) → (BK = 9) → (BC = 20) →
  (AK^2 + (BC - BK)^2 = AC^2) →
  area_of_triangle ABC = 20*sqrt(26) :=
sorry

end area_of_triangle_ABC_l725_725300


namespace part1_l725_725730

   noncomputable def sin_20_deg_sq : ℝ := (Real.sin (20 * Real.pi / 180))^2
   noncomputable def cos_80_deg_sq : ℝ := (Real.sin (10 * Real.pi / 180))^2
   noncomputable def sqrt3_sin20_cos80 : ℝ := Real.sqrt 3 * Real.sin (20 * Real.pi / 180) * Real.sin (10 * Real.pi / 180)
   noncomputable def value : ℝ := sin_20_deg_sq + cos_80_deg_sq + sqrt3_sin20_cos80

   theorem part1 : value = 1 / 4 := by
     sorry
   
end part1_l725_725730


namespace stuffed_animals_correctness_l725_725962

def M : ℕ := 34
def K : ℕ := 2 * M
def T : ℕ := K + 5
def S : ℕ := M + K + T
def A : ℚ := S / 3
def F : ℚ := M / S

theorem stuffed_animals_correctness :
  K = 68 ∧ 
  T = 73 ∧ 
  S = 175 ∧ 
  A = 175 / 3 ∧ 
  F = 34 / 175 :=
begin
  sorry,
end

end stuffed_animals_correctness_l725_725962


namespace trents_average_speed_l725_725296

def block_length : ℕ := 50
def walk_distance : ℕ := 4 * block_length
def bus_distance : ℕ := 7 * block_length
def bike_distance : ℕ := 5 * block_length
def jog_distance : ℕ := 5 * block_length
def total_blocks_walked := 4 + 4
def total_blocks_bus := 7 + 7
def total_blocks_biked := 5
def total_distance_meters := walk_distance + bus_distance + bike_distance + jog_distance + bus_distance + walk_distance
def total_distance_km := total_distance_meters / 1000
def total_time_hours : ℕ := 2
def average_speed := total_distance_km / total_time_hours

theorem trents_average_speed : average_speed = 0.8 := 
by
  sorry

end trents_average_speed_l725_725296


namespace table_size_condition_l725_725192

-- Define the problem in Lean 4
theorem table_size_condition (n : ℕ) (a : ℕ → ℕ → ℝ) (_ : ∀ i j, 0 ≤ a i j) 
  (H1 : ∀ i, ∃ j, 0 < a i j) (H2 : ∀ j, ∃ i, 0 < a i j)
  (H3 : ∀ i j, 0 < a i j → (∑ k, a i k) = (∑ k, a k j)) : n = 2015 :=
sorry

end table_size_condition_l725_725192


namespace vector_dot_product_l725_725526

variables {a b c : V}
variable [normed_space ℝ V]

theorem vector_dot_product 
  (ha : ∥a∥ = 1)
  (hb : ∥b∥ = 1)
  (hab : ∥a + b∥ = 1)
  (h3 : c - a - 4 * b = 5 * (a × b)) :
  b ⬝ c = 7 / 2 :=
sorry

end vector_dot_product_l725_725526


namespace find_f_of_2_l725_725865

def f : ℝ → ℝ :=
  λ x, log 5 x

theorem find_f_of_2 (h : ∀ x : ℝ, f (5^x) = x) : f 2 = log 5 2 :=
by
  sorry

end find_f_of_2_l725_725865


namespace days_until_birthday_l725_725238

theorem days_until_birthday :
  ∀ (savings_per_day flower_cost savings_per_flower number_of_flowers total_savings days_to_birthday : ℕ),
    savings_per_day = 2 →
    flower_cost = 4 →
    savings_per_flower = 4 →
    number_of_flowers = 11 →
    total_savings = number_of_flowers * flower_cost →
    days_to_birthday = total_savings / savings_per_day →
    days_to_birthday = 22 :=
by
  intros
  rw [ ← h4, h3, h1, h5]
  simp [ ← h2]
  sorry

end days_until_birthday_l725_725238


namespace difference_between_numbers_l725_725274

theorem difference_between_numbers (x y : ℕ) (h : x - y = 9) :
  (10 * x + y) - (10 * y + x) = 81 :=
by
  sorry

end difference_between_numbers_l725_725274


namespace average_of_combined_results_l725_725268

theorem average_of_combined_results (s1 s2 : Finset ℝ) (cond1 : s1.card = 60) (cond2 : s2.card = 40)
  (cond3 : (s1.sum / 60) = 40) (cond4 : (s2.sum / 40) = 60) :
  ((s1.sum + s2.sum) / (s1.card + s2.card)) = 48 :=
by
  -- Proof will go here
  sorry

end average_of_combined_results_l725_725268


namespace min_value_f_on_interval_l725_725398

def f (x: ℝ) : ℝ := (1 / 3) ^ x

theorem min_value_f_on_interval : 
  (∃ x ∈ set.Icc (-1:ℝ) (0:ℝ), (∀ y ∈ set.Icc (-1:ℝ) (0:ℝ), f(x) ≤ f(y)) ∧ f(x) = 1) :=
sorry

end min_value_f_on_interval_l725_725398


namespace closest_to_zero_is_neg_1001_l725_725312

-- Definitions used in the conditions
def list_of_integers : List Int := [-1101, 1011, -1010, -1001, 1110]

-- Problem statement
theorem closest_to_zero_is_neg_1001 (x : Int) (H : x ∈ list_of_integers) :
  x = -1001 ↔ ∀ y ∈ list_of_integers, abs x ≤ abs y :=
sorry

end closest_to_zero_is_neg_1001_l725_725312


namespace chromatic_number_of_triangulation_l725_725233

-- Define a convex n-gon and its triangulation
structure Polygon (n : ℕ) :=
(is_convex : n ≥ 3)

structure Triangulation (n : ℕ) extends Polygon n :=
(diagonals : Finset (Fin n × Fin n))
(no_intersection : ∀ {i j k l : Fin n}, (i, j) ∈ diagonals → (k, l) ∈ diagonals → i ≠ k → j ≠ l → ¬(segment_intersects (i, j) (k, l)))

def triangulation_graph (n : ℕ) (t : Triangulation n) : SimpleGraph (Fin n) :=
{ adj := λ u v, (u, v) ∈ t.diagonals ∨ abs (u.1 - v.1) = 1 ∨ abs (u.1 - v.1) = n - 1,
  sym := sorry,
  loopless := sorry }

-- Prove that all triangulations of a convex n-gon have a chromatic number of 3.
theorem chromatic_number_of_triangulation (n : ℕ) (t : Triangulation n) (h : n ≥ 3) : 
  (triangulation_graph n t).chromatic_number = 3 := 
sorry

end chromatic_number_of_triangulation_l725_725233


namespace price_for_12kg_l725_725780

-- Define the conditions
def price_proportional_to_mass (k : ℝ) := k > 0 ∧ ∀ (m₁ m₂ : ℝ), price m₁ = k * m₁ → price m₂ = k * m₂

-- Define the specific prices in the conditions
def specific_price : ℝ := 36
def specific_mass : ℝ := 12

-- Given the proportionality constant k
variables (k : ℝ) (price : ℝ → ℝ)

-- Given conditions definition
axiom proportional_price_cond : price_proportional_to_mass k
axiom specific_price_paid : price specific_mass = specific_price

-- Statement of the Lean 4 proof problem
theorem price_for_12kg (mass : ℝ := 12) : price mass = 36 :=
by sorry

end price_for_12kg_l725_725780


namespace speed_on_downward_road_l725_725745

theorem speed_on_downward_road (v : ℝ) :
  (∀ (d : ℝ), 50 * (d / 50) + v * (d / v) = 50 * 12 ⇔ 2 * d = 800) → v = 100 := by
  sorry

end speed_on_downward_road_l725_725745


namespace monotonicity_f_inequality_f_div_l725_725130

noncomputable def f (a x : ℝ) : ℝ := a * x * real.exp x

theorem monotonicity_f {a : ℝ} (h₀ : a ≠ 0) :
  (∀ x < -1, deriv (f a) x < 0) ∧ (∀ x > -1, deriv (f a) x > 0)
  ∨ (∀ x < -1, deriv (f a) x > 0) ∧ (∀ x > -1, deriv (f a) x < 0) :=
sorry

theorem inequality_f_div (a x : ℝ) (h₀ : a ≠ 0) (h₁ : a ≥ 4 / real.exp 2) (h₂ : x > 0) :
  (f a x) / (x + 1) - (x + 1) * real.log x > 0 :=
sorry

end monotonicity_f_inequality_f_div_l725_725130


namespace university_A_pass_one_subject_university_B_pass_one_subject_when_m_3_5_preferred_range_of_m_l725_725608

-- Part 1
def probability_A_exactly_one_subject : ℚ :=
  3 * (1/2) * (1/2)^2

def probability_B_exactly_one_subject (m : ℚ) : ℚ :=
  (1/6) * (2/5)^2 + (5/6) * (3/5) * (2/5) * 2

theorem university_A_pass_one_subject : probability_A_exactly_one_subject = 3/8 :=
sorry

theorem university_B_pass_one_subject_when_m_3_5 : probability_B_exactly_one_subject (3/5) = 32/75 :=
sorry

-- Part 2
def expected_A : ℚ :=
  3 * (1/2)

def expected_B (m : ℚ) : ℚ :=
  ((17 - 7 * m) / 30) + (2 * (3 + 14 * m) / 30) + (3 * m / 10)

theorem preferred_range_of_m : 0 < m ∧ m < 11/15 → expected_A > expected_B m :=
sorry

end university_A_pass_one_subject_university_B_pass_one_subject_when_m_3_5_preferred_range_of_m_l725_725608


namespace daisy_dog_toys_l725_725609

-- Given conditions
def dog_toys_monday : ℕ := 5
def dog_toys_tuesday_left : ℕ := 3
def dog_toys_tuesday_bought : ℕ := 3
def dog_toys_wednesday_all_found : ℕ := 13

-- The question we need to answer
def dog_toys_bought_wednesday : ℕ := 7

-- Statement to prove
theorem daisy_dog_toys :
  (dog_toys_monday - dog_toys_tuesday_left + dog_toys_tuesday_left + dog_toys_tuesday_bought + dog_toys_bought_wednesday = dog_toys_wednesday_all_found) :=
sorry

end daisy_dog_toys_l725_725609


namespace lcm_36_100_l725_725460

theorem lcm_36_100 : Nat.lcm 36 100 = 900 :=
by
  sorry

end lcm_36_100_l725_725460


namespace table_properties_l725_725199

theorem table_properties (n : ℕ) (table : Matrix (Fin 2015) (Fin n) ℕ) :
  (∀ i : Fin 2015, ∃ j : Fin n, table i j > 0) →
  (∀ j : Fin n, ∃ i : Fin 2015, table i j > 0) →
  (∀ i : Fin 2015, ∀ j : Fin n, (table i j > 0 → (∑ x, table i x) = (∑ y, table y j))) →
  n = 2015 :=
  sorry

end table_properties_l725_725199


namespace algebraic_sum_of_coefficients_l725_725799

def sequence (n : ℕ) : ℕ → ℕ
| 0     := 7
| k + 1 := sequence k + 5 + 2 * k

theorem algebraic_sum_of_coefficients :
  (∃ (a b c : ℤ), ∀ (n : ℕ), sequence n = a * n^2 + b * n + c) → 1 + 2 + 4 = 7 :=
by
  intros h
  sorry

end algebraic_sum_of_coefficients_l725_725799


namespace shaded_to_circle_ratio_l725_725714

-- Conditions as definitions in Lean 4
def AC : ℝ := 1
def CB : ℝ := 3
def CD : ℝ := 4
def radius_of_semi_circle_AB : ℝ := AC + CB

-- Calculate the required areas
def area_of_semi_circle (radius : ℝ) : ℝ := (1 / 2) * Math.pi * radius^2
def area_of_circle (radius : ℝ) : ℝ := Math.pi * radius^2
def shaded_area : ℝ := 
  area_of_semi_circle radius_of_semi_circle_AB
  - area_of_semi_circle (AC / 2)
  - area_of_semi_circle (CB / 2)

def area_circle_CD : ℝ := area_of_circle CD

-- Defining the proof problem
theorem shaded_to_circle_ratio : (shaded_area / area_circle_CD) = (1 / 2) :=
by
  sorry

end shaded_to_circle_ratio_l725_725714


namespace max_sector_area_l725_725497

theorem max_sector_area (r θ : ℝ) (S : ℝ) (h_perimeter : 2 * r + θ * r = 16)
  (h_max_area : S = 1 / 2 * θ * r^2) :
  r = 4 ∧ θ = 2 ∧ S = 16 := by
  -- sorry, the proof is expected to go here
  sorry

end max_sector_area_l725_725497


namespace binom_divisibility_by_prime_l725_725721

-- Given definitions
variable (p k : ℕ) (hp : Nat.Prime p) (hk1 : 2 ≤ k) (hk2 : k ≤ p - 2)

-- Main theorem statement
theorem binom_divisibility_by_prime
  (hp : Nat.Prime p) (hk1 : 2 ≤ k) (hk2 : k ≤ p - 2) :
  Nat.choose (p - k + 1) k - Nat.choose (p - k - 1) (k - 2) ≡ 0 [MOD p] :=
sorry

end binom_divisibility_by_prime_l725_725721


namespace problem_statement_l725_725943

def euler_totient (n : ℕ) : ℕ :=
  (Finset.range n).filter (λ m, Nat.gcd m n = 1).card

def reduced_residue_system (n : ℕ) : Finset ℕ :=
  (Finset.range n).filter (λ m, Nat.gcd m n = 1)

theorem problem_statement (n : ℕ) (a : ℤ) (hn_pos : 0 < n)
  (hgcd : Nat.gcd a.natAbs n = 1)
  (hresidue : reduced_residue_system n) :
  (a^ (euler_totient n) - 1) / n 
  ≡ (Finset.sum (reduced_residue_system n) (λ i, 1 / (a * i) * ⌊(a * i) / n⌋)) [MOD n] :=
by
  sorry

end problem_statement_l725_725943


namespace probability_losing_ticket_l725_725178

theorem probability_losing_ticket (winning : ℕ) (losing : ℕ)
  (h_odds : winning = 5 ∧ losing = 8) :
  (losing : ℚ) / (winning + losing : ℚ) = 8 / 13 := by
  sorry

end probability_losing_ticket_l725_725178


namespace max_similar_triangles_is_4_l725_725694

-- Define the basic setup: five distinct points on a plane
variable (A B C D E : Point)

-- Define what it means for three points to form a triangle
def is_triangle (P1 P2 P3 : Point) : Prop := 
  P1 ≠ P2 ∧ P2 ≠ P3 ∧ P1 ≠ P3 ∧ 
  ¬ collinear P1 P2 P3 -- No three points should be collinear

-- Define similar triangles
def similar_triangle (P1 P2 P3 Q1 Q2 Q3 : Point) : Prop :=
  -- Define condition for similar triangles here
  sorry

-- Define the condition: There are exactly 5 distinct points on the plane
axiom distinct_5_points : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ C ≠ D ∧ C ≠ E ∧ D ≠ E

-- Main theorem: the maximum number of similar triangles that can be formed
theorem max_similar_triangles_is_4 (P : set Point) (hP : P = {A, B, C, D, E}) :
  ∃ S : finset (finset Point), S.card = 4 ∧ 
  ∀ T1 T2 ∈ S, ∃ (P1 P2 P3 Q1 Q2 Q3 : Point), P1 ∈ P ∧ P2 ∈ P ∧ P3 ∈ P ∧ Q1 ∈ P ∧ Q2 ∈ P ∧ Q3 ∈ P ∧ 
  T1 = {P1, P2, P3} ∧ T2 = {Q1, Q2, Q3} ∧ is_triangle P1 P2 P3 ∧ is_triangle Q1 Q2 Q3 ∧ 
  similar_triangle P1 P2 P3 Q1 Q2 Q3 :=
  sorry

end max_similar_triangles_is_4_l725_725694


namespace asha_remaining_money_l725_725042

-- Define the borrowed amounts, gift, and savings
def borrowed_from_brother : ℤ := 20
def borrowed_from_father : ℤ := 40
def borrowed_from_mother : ℤ := 30
def gift_from_granny : ℤ := 70
def savings : ℤ := 100

-- Total amount of money Asha has
def total_amount : ℤ := borrowed_from_brother + borrowed_from_father + borrowed_from_mother + gift_from_granny + savings

-- Amount spent by Asha
def amount_spent : ℤ := (3 * total_amount) / 4

-- Amount of money Asha remains with
def amount_left : ℤ := total_amount - amount_spent

-- The proof statement
theorem asha_remaining_money : amount_left = 65 := by
  sorry

end asha_remaining_money_l725_725042


namespace fraction_of_area_l725_725634

noncomputable section

open Real

-- Definitions of points A, B, C, X, Y, and Z with their given coordinates
def A := (2, 0) : ℝ × ℝ
def B := (8, 12) : ℝ × ℝ
def C := (14, 0) : ℝ × ℝ

def X := (6, 0) : ℝ × ℝ
def Y := (8, 4) : ℝ × ℝ
def Z := (10, 0) : ℝ × ℝ

-- Definition of the area of a triangle given vertices
def area (p₁ p₂ p₃ : ℝ × ℝ) : ℝ := 
  abs ((p₂.1 - p₁.1) * (p₃.2 - p₁.2) - (p₂.2 - p₁.2) * (p₃.1 - p₁.1)) / 2

-- Areas of triangles ABC and XYZ
def area_ABC := area A B C
def area_XYZ := area X Y Z

-- The Lean statement
theorem fraction_of_area : (area_XYZ / area_ABC) = 1 / 9 := by
  sorry

end fraction_of_area_l725_725634


namespace grid_points_circumference_l725_725972

def numGridPointsOnCircumference (R : ℝ) : ℕ := sorry

def isInteger (x : ℝ) : Prop := ∃ (n : ℤ), x = n

theorem grid_points_circumference (R : ℝ) (h : numGridPointsOnCircumference R = 1988) : 
  isInteger R ∨ isInteger (Real.sqrt 2 * R) :=
by
  sorry

end grid_points_circumference_l725_725972


namespace find_p_over_q_at_0_l725_725394

noncomputable def p (x : ℝ) := 3 * (x - 4) * (x - 1)
noncomputable def q (x : ℝ) := (x + 3) * (x - 1) * (x - 4)

theorem find_p_over_q_at_0 : (p 0) / (q 0) = 1 := 
by
  sorry

end find_p_over_q_at_0_l725_725394


namespace number_of_seven_digit_integers_with_two_adjacent_even_numbers_l725_725092

theorem number_of_seven_digit_integers_with_two_adjacent_even_numbers 
  : ∃ n, n = 2880 ∧
    (let digits := [1, 2, 3, 4, 5, 6, 7] in 
     ∃ numbers : list (list ℕ), 
     (∀ num ∈ numbers, 
      list.length num = 7 ∧ 
      (list.countp (λ x, x ∈ [2, 4, 6]) num = 3) ∧
      (1 < list.length (list.filter (λ p, match p with | (a, b) => a ∈ [2, 4, 6] ∧ b ∈ [2, 4, 6] end) (list.zip num (list.tail num))) ∧
       (list.length (list.filter (λ p, match p with | (a, b) => a ∈ [2, 4, 6] ∧ b ∈ [2, 4, 6] end) (list.zip num (list.tail num))) = 1)))) ↔
     (numbers.length = n)) :=
begin
  sorry
end

end number_of_seven_digit_integers_with_two_adjacent_even_numbers_l725_725092


namespace number_of_ordered_tuples_l725_725853

noncomputable def count_tuples 
  (a1 a2 a3 a4 : ℕ) 
  (H_distinct : a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4)
  (H_range : 1 ≤ a1 ∧ a1 ≤ 100 ∧ 1 ≤ a2 ∧ a2 ≤ 100 ∧ 1 ≤ a3 ∧ a3 ≤ 100 ∧ 1 ≤ a4 ∧ a4 ≤ 100)
  (H_eqn : (a1^2 + a2^2 + a3^2) * (a2^2 + a3^2 + a4^2) = (a1 * a2 + a2 * a3 + a3 * a4)^2): ℕ :=
40

theorem number_of_ordered_tuples 
  (a1 a2 a3 a4 : ℕ)
  (H_distinct : a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4)
  (H_range : 1 ≤ a1 ∧ a1 ≤ 100 ∧ 1 ≤ a2 ∧ a2 ≤ 100 ∧ 1 ≤ a3 ∧ a3 ≤ 100 ∧ 1 ≤ a4 ∧ a4 ≤ 100)
  (H_eqn : (a1^2 + a2^2 + a3^2) * (a2^2 + a3^2 + a4^2) = (a1 * a2 + a2 * a3 + a3 * a4)^2) : 
  count_tuples a1 a2 a3 a4 H_distinct H_range H_eqn = 40 :=
sorry

end number_of_ordered_tuples_l725_725853


namespace converse_statement_l725_725579

variable (a b : ℝ)
variable (vec_a vec_b : Vector3)
variable (length_vec_a length_vec_b : ℝ)

def vec_neg (v : Vector3) : Vector3 := -v

axiom length_eq : |vec_a| = length_vec_a
axiom length_eq_b : |vec_b| = length_vec_b

theorem converse_statement :
  (vec_a = vec_neg vec_b) → (|vec_a| = |vec_b|) → (|vec_a| = |vec_b|) → (vec_a = vec_neg vec_b) :=
by
  exact sorry

end converse_statement_l725_725579


namespace find_k_l725_725093

-- Define the variables and conditions
variables (x y k : ℤ)

-- State the theorem
theorem find_k (h1 : x = 2) (h2 : y = 1) (h3 : k * x - y = 3) : k = 2 :=
sorry

end find_k_l725_725093


namespace problem1_problem2_l725_725145

noncomputable theory

-- defining the vectors m and n
def m (x : ℝ) : ℝ × ℝ := (√3 * sin (2 * x) + 2, cos x)
def n (x : ℝ) : ℝ × ℝ := (1, 2 * cos x)

-- defining the function f
def f (x : ℝ) : ℝ := (m x).fst * (n x).fst + (m x).snd * (n x).snd

-- Lean statement for the first part of the problem
theorem problem1 :
  (∃ T > 0, ∀ x, f(x + T) = f(x)) ∧
  (∃ (k : ℤ), ∀ x ∈ set.Icc (k * π - π / 3) (k * π + π / 6), ∀ y ∈ set.Icc (k * π - π / 3) (k * π + π / 6), (x <= y) → f x <= f y) := sorry

-- Lean statement for the second part of the problem
theorem problem2 :
  ∀ (A : ℝ) (b : ℝ) (c : ℝ), f A = 4 → b = 1 → (1/2 * b * c * sin A = √3/2) → ∃ a, a = √3 := sorry

end problem1_problem2_l725_725145


namespace trajectory_of_point_Q_l725_725855

theorem trajectory_of_point_Q : 
  ∀ (Q P : ℝ × ℝ), 
  (∃ (y x : ℝ), P = (-2 - x, 4 - y) ∧ 2 * x - y + 3 = 0) ∧ 
  M = (-1, 2) ∧ 
  midpoint P Q = M ∧ 
  dist P M = dist M Q →
  ∃ (y x : ℝ), 2 * x - y + 5 = 0
:= sorry

end trajectory_of_point_Q_l725_725855


namespace inscribed_rectangle_area_l725_725269

theorem inscribed_rectangle_area (b h x : ℝ) (hb : 0 < b) (hh : 0 < h) (hx : 0 < x) (hxle : x ≤ h) :
  ∃ area : ℝ, area = (b * x / h) * (h - x) :=
by
  let n := (b * (h - x)) / h
  let area := x * n
  use area
  unfold n area
  sorry

end inscribed_rectangle_area_l725_725269


namespace inscribe_ngon_in_circle_l725_725929

-- Definitions based on conditions from part a)
variables (n : ℕ) (M : Point) (circle : Circle) (lines : List Line)

-- The theorem statement, ensuring the conditions are encapsulated.
theorem inscribe_ngon_in_circle (h_n_gt_2 : n > 2) :
  ∃ polygon : Polygon, 
  (polygon.inscribed_in circle) ∧ 
  (∃ side : polygon.Side, side.contains_point M) ∧ 
  (parallel_sides polygon lines) :=
sorry

end inscribe_ngon_in_circle_l725_725929


namespace total_amount_charged_l725_725702

-- Define the conditions
variable (hours_mechanic1 hours_mechanic2 rate_sum : ℝ)
variable (h1 : hours_mechanic1 = 10)
variable (h2 : hours_mechanic2 = 5)
variable (h3 : rate_sum = 160)

-- Define the question as a theorem
theorem total_amount_charged : 
  (hours_mechanic1 + hours_mechanic2) * rate_sum = 2400 :=
by
  -- Using given conditions
  rw [h1, h2, h3],
  -- Simplify
  norm_num

end total_amount_charged_l725_725702


namespace area_of_bounded_region_l725_725437

theorem area_of_bounded_region : 
  let x1 := 2
  let y1 := 2
  let x0 := 0
  let y0 := 0
  in
  let area := (x1 - x0) * (y1 - y0)
  in
  area = 4 :=
by 
  -- definitions for x1, y1, x0, y0
  let x1 := 2
  let y1 := 2
  let x0 := 0
  let y0 := 0
  
  -- let area be the area of the square bounded by these lines
  let area := (x1 - x0) * (y1 - y0)
  
  -- assertion
  have h : area = (2 - 0) * (2 - 0), from rfl,
  
  -- proving the final statement
  show area = 4, by 
    rw [h],
    exact rfl

-- skipped proof step
sorry

end area_of_bounded_region_l725_725437


namespace area_of_region_bounded_by_lines_l725_725431

theorem area_of_region_bounded_by_lines : 
  let x1 := λ θ : ℝ, 2 in
  let y1 := λ θ : ℝ, 2 in
  x1 (0:ℝ) * y1 (0:ℝ) = 4 :=
by {
  sorry,
}

end area_of_region_bounded_by_lines_l725_725431


namespace total_marks_l725_725367

-- Variables and conditions
variables (M C P : ℕ)
variable (h1 : C = P + 20)
variable (h2 : (M + C) / 2 = 40)

-- Theorem statement
theorem total_marks (M C P : ℕ) (h1 : C = P + 20) (h2 : (M + C) / 2 = 40) : M + P = 60 :=
sorry

end total_marks_l725_725367


namespace fraction_of_triangle_area_l725_725628

open Real

def point := (ℝ × ℝ)

def area_of_triangle (A B C : point) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (abs ((x1 * (y2 - y3)) + (x2 * (y3 - y1)) + (x3 * (y1 - y2))) / 2)

def A : point := (2, 0)
def B : point := (8, 12)
def C : point := (14, 0)

def X : point := (6, 0)
def Y : point := (8, 4)
def Z : point := (10, 0)

theorem fraction_of_triangle_area :
  (area_of_triangle X Y Z) / (area_of_triangle A B C) = 1 / 9 :=
by
  sorry

end fraction_of_triangle_area_l725_725628


namespace min_perimeter_of_rectangle_l725_725015

variables (a b : ℕ)
def side_lengths_condition : Prop := (2 * a + b = 3 * a + b + a + (b - a))

theorem min_perimeter_of_rectangle :
  side_lengths_condition a b →
  (∃ a b : ℕ, 2 * (3 * a + b + 12 * a - b) = 30) :=
begin
  intro h,
  use [1, 3],
  split,
  {
    exact nat.add_comm 1 3,
  },
  {
    rw [nat.mul_comm, nat.mul_comm],
    exact eq.refl 30,
  },
  sorry

end min_perimeter_of_rectangle_l725_725015


namespace part_I_min_value_part_II_a_range_l725_725512

noncomputable def f (x a : ℝ) : ℝ := abs (2 * x - a) - abs (x + 3)

theorem part_I_min_value (x : ℝ) : f x 1 ≥ -7 / 2 :=
by sorry 

theorem part_II_a_range (x a : ℝ) (hx : 0 ≤ x) (hx' : x ≤ 3) (hf : f x a ≤ 4) : -4 ≤ a ∧ a ≤ 7 :=
by sorry

end part_I_min_value_part_II_a_range_l725_725512


namespace fraction_value_condition_l725_725476

theorem fraction_value_condition (m n : ℚ) (h : m / n = 2 / 3) : m / (m + n) = 2 / 5 :=
sorry

end fraction_value_condition_l725_725476


namespace calculate_f_log3_l725_725127

def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + x^2) - x) + 1

theorem calculate_f_log3 : f (Real.log 3) + f (Real.log (1 / 3)) = 2 :=
by
  -- Definitions and theorems relevant, proof to be provided
  sorry

end calculate_f_log3_l725_725127


namespace solve_system_l725_725645

noncomputable theory

def system_of_equations (x y : ℝ) : Prop :=
  (x + 2 * y = 2) ∧ (x - 2 * y = 6)

theorem solve_system : ∃ x y : ℝ, system_of_equations x y ∧ x = 4 ∧ y = -1 :=
  by {
    use [4, -1],
    split,
    {
      unfold system_of_equations,
      split;
      simp,
    },
    split;
    simp,
    sorry
  }

end solve_system_l725_725645


namespace ratio_percent_increase_decrease_l725_725008

variable (P U : ℝ)

theorem ratio_percent_increase_decrease :
  let new_price := 0.80 * P,
      units_sold_increase_percent := 25,
      further_discounted_price := 0.72 * P,
      price_decrease_percent := 28 in
  (units_sold_increase_percent / price_decrease_percent) = 1 / 1.12 := by
  sorry

end ratio_percent_increase_decrease_l725_725008


namespace payback_period_l725_725994

def system_unit_cost : ℕ := 9499 -- cost in RUB
def graphics_card_cost : ℕ := 20990 -- cost per card in RUB
def num_graphics_cards : ℕ := 2
def system_unit_power : ℕ := 120 -- power in watts
def graphics_card_power : ℕ := 185 -- power per card in watts
def earnings_per_card_per_day_ethereum : ℚ := 0.00630
def ethereum_to_rub : ℚ := 27790.37 -- RUB per ETH
def electricity_cost_per_kwh : ℚ := 5.38 -- RUB per kWh
def total_investment : ℕ := system_unit_cost + num_graphics_cards * graphics_card_cost
def total_power_consumption_watts : ℕ := system_unit_power + num_graphics_cards * graphics_card_power
def total_power_consumption_kwh_per_day : ℚ := total_power_consumption_watts / 1000 * 24
def daily_earnings_rub : ℚ := earnings_per_card_per_day_ethereum * num_graphics_cards * ethereum_to_rub
def daily_energy_cost : ℚ := total_power_consumption_kwh_per_day * electricity_cost_per_kwh
def net_daily_profit : ℚ := daily_earnings_rub - daily_energy_cost

theorem payback_period : total_investment / net_daily_profit = 179 := by
  sorry

end payback_period_l725_725994


namespace min_value_expression_l725_725113

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2*x + y = 1) : 
  ∃ (xy_min : ℝ), xy_min = 9 ∧ (∀ (x y : ℝ), 0 < x ∧ 0 < y ∧ 2*x + y = 1 → (x + 2*y)/(x*y) ≥ xy_min) :=
sorry

end min_value_expression_l725_725113


namespace jason_borrowed_amount_l725_725566

theorem jason_borrowed_amount (hours cycles value_per_cycle remaining_hrs remaining_value total_value: ℕ) : 
  hours = 39 → cycles = (hours / 7) → value_per_cycle = 28 → remaining_hrs = (hours % 7) →
  remaining_value = (1 + 2 + 3 + 4) →
  total_value = (cycles * value_per_cycle + remaining_value) →
  total_value = 150 := 
by {
  sorry
}

end jason_borrowed_amount_l725_725566


namespace lauren_subscribers_l725_725569

theorem lauren_subscribers
  (money_per_commercial : ℝ)
  (money_per_subscription : ℝ)
  (num_commercials_watched : ℕ)
  (total_earnings : ℝ)
  (h1 : money_per_commercial = 0.50)
  (h2 : money_per_subscription = 1.00)
  (h3 : num_commercials_watched = 100)
  (h4 : total_earnings = 77) :
  ∃ (num_subscriptions : ℕ), num_subscriptions = 27 :=
by
  let total_from_commercials := (num_commercials_watched : ℝ) * money_per_commercial
  let total_from_subscriptions := total_earnings - total_from_commercials
  have h_total_from_commercials : total_from_commercials = 50, from sorry
  have h_total_from_subscriptions : total_from_subscriptions = 27, from sorry
  use nat_ceil total_from_subscriptions
  have result : nat_ceil total_from_subscriptions = 27, from sorry
  exact result

end lauren_subscribers_l725_725569


namespace mo_tea_cups_l725_725973

theorem mo_tea_cups (n t : ℤ) (h1 : 4 * n + 3 * t = 22) (h2 : 3 * t = 4 * n + 8) : t = 5 :=
by
  -- proof steps
  sorry

end mo_tea_cups_l725_725973


namespace tomatoes_left_l725_725692

theorem tomatoes_left (initial_tomatoes : ℕ) (birds : ℕ) (fraction_eaten : ℚ) :
  initial_tomatoes = 21 ∧ birds = 2 ∧ fraction_eaten = 1/3 ->
  initial_tomatoes - initial_tomatoes * fraction_eaten = 14 :=
by
  intros h
  cases h with h1 h_rest
  cases h_rest with h2 h3
  rw [h1, h2, h3]
  norm_num
  rw [Nat.cast_sub 21 7 _, Nat.cast_mul, Nat.cast_div]; norm_num -- Converting to rational arithmetic and proving directly
  exact le_of_lt_nat (div_lt_self (zero_lt_nat 21) (zero_lt_nat 3))

end tomatoes_left_l725_725692


namespace find_principal_sum_l725_725750

noncomputable def principal_sum_lent (r : ℝ) (n : ℕ) (t : ℕ) (interest_difference : ℝ) : ℝ :=
  let A : ℝ := P * (1 + r / (n : ℝ)) ^ (n * t) in
  P = A - interest_difference ∧
  P = 340 / (1.0 - ((1 + 0.02) ^ 24 - 1))

theorem find_principal_sum :
  principal_sum_lent 0.04 2 12 340 = 852.24 := sorry

end find_principal_sum_l725_725750


namespace area_of_region_bounded_by_lines_l725_725432

theorem area_of_region_bounded_by_lines : 
  let x1 := λ θ : ℝ, 2 in
  let y1 := λ θ : ℝ, 2 in
  x1 (0:ℝ) * y1 (0:ℝ) = 4 :=
by {
  sorry,
}

end area_of_region_bounded_by_lines_l725_725432


namespace GH_perpendicular_to_FC_l725_725820

open EuclideanGeometry

-- Definitions of points and distances
variables (A B C D E F : Point)
variables (d : Length)

-- Conditions
axiom distinct_points (hA : A ≠ B) (hB : B ≠ C) (hC : C ≠ D) (hD : D ≠ E) (hE : E ≠ A) : 
  True

axiom points_collinear (hABC : collinear A B C) (hBCD : collinear B C D) 
  (hCDE : collinear C D E) : True

axiom distances_equal (hAB : |A - B| = d) (hBC : |B - C| = d) 
  (hCD : |C - D| = d) (hDE : |D - E| = d) : True

-- Definition of circumcenters
def G := circumcenter A D F
def H := circumcenter B E F

-- The proof statement
theorem GH_perpendicular_to_FC :
  ⟨G, H⟩ = ⟨GH_perpendicular_FC⟩ :=
sorry

end GH_perpendicular_to_FC_l725_725820


namespace geometric_sequence_example_l725_725911

theorem geometric_sequence_example
  (a : ℕ → ℝ)
  (h1 : ∀ n, 0 < a n)
  (h2 : ∃ r, ∀ n, a (n + 1) = r * a n)
  (h3 : Real.log (a 2) / Real.log 2 + Real.log (a 8) / Real.log 2 = 1) :
  a 3 * a 7 = 2 :=
sorry

end geometric_sequence_example_l725_725911


namespace increasing_function_property_l725_725088

-- Definitions based on problem conditions
def V (n : ℕ) : ℕ := ∑ (i : ℕ) in (multiset.filter (λ p, p > 10^100) (multiset.prime_factors n)), 1

-- The statement of the theorem
theorem increasing_function_property
  (f : ℤ → ℤ)
  (hf : ∀ a b : ℤ, a > b → V (f (a) - f (b)) ≤ V (a - b))
  : ∃ a b : ℤ, f = λ x, a * x + b :=
sorry

end increasing_function_property_l725_725088


namespace max_value_of_g_is_sqrt_5_l725_725505

noncomputable def f : ℝ → ℝ → ℝ := λ x a, sin x + a * cos x

noncomputable def g (a : ℝ) : ℝ → ℝ := λ x, sin x + f x a

theorem max_value_of_g_is_sqrt_5 (a : ℝ) (h : ∀ x, f (π / 4 + x) a = f (π / 4 - x) a) : 
  ∃ x, g 1 x = sqrt (5) := 
by
  sorry

end max_value_of_g_is_sqrt_5_l725_725505


namespace projection_of_sum_a_b_on_a_is_zero_l725_725888

variables (a b : Vector) (theta : ℝ)
noncomputable def projection_problem :=
  let dot_product_ab := a + b in
  let projection_a_on_a := (dot_product_ab ∙ a) / 1 in
  projection_a_on_a

theorem projection_of_sum_a_b_on_a_is_zero 
  (ha : ∥a∥ = 1)
  (hb : ∥b∥ = 2)
  (h_angle : theta = 120 * π / 180) 
  (h_cos_angle : real.cos theta = -1 / 2) :
  projection_problem a b theta = 0 :=
  sorry

end projection_of_sum_a_b_on_a_is_zero_l725_725888


namespace lcm_36_100_l725_725457

theorem lcm_36_100 : Nat.lcm 36 100 = 900 :=
by
  sorry

end lcm_36_100_l725_725457


namespace sin_alpha_in_fourth_quadrant_l725_725493

-- Given conditions
variables (α : Real) 

-- Statement
theorem sin_alpha_in_fourth_quadrant (h_quadrant : 0 > sin α ∧ cos α > 0) (h_tan : tan α = -5/12) : sin α = -5/13 := by
  sorry

end sin_alpha_in_fourth_quadrant_l725_725493


namespace initials_count_l725_725153

/-- The number of different three-letter sets of initials possible using the letters 
A through J, with no repeated letters in any set, is 720. -/
theorem initials_count : 
  let letters := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'} in
  let num_letters := 10 in
  let count := num_letters * (num_letters - 1) * (num_letters - 2) in
  count = 720 :=
by
  let letters := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'}
  let num_letters := 10
  let count := num_letters * (num_letters - 1) * (num_letters - 2)
  have : count = 10 * 9 * 8 := rfl
  exact this

end initials_count_l725_725153


namespace sufficient_not_necessary_condition_l725_725489

variables (a b c : ℝ)

theorem sufficient_not_necessary_condition (h1 : c < b) (h2 : b < a) :
  (ac < 0 → ab > ac) ∧ (ab > ac → ac < 0) → false :=
sorry

end sufficient_not_necessary_condition_l725_725489


namespace f_sum_zero_l725_725667

-- Define the function f with the given properties
noncomputable def f : ℝ → ℝ := sorry

-- Define hypotheses based on the problem's conditions
axiom f_cube (x : ℝ) : f (x ^ 3) = (f x) ^ 3
axiom f_inj (x1 x2 : ℝ) (h : x1 ≠ x2) : f x1 ≠ f x2

-- State the proof problem
theorem f_sum_zero : f 0 + f 1 + f (-1) = 0 :=
sorry

end f_sum_zero_l725_725667


namespace fly_distance_is_314_l725_725010

noncomputable def total_fly_distance
  (r : ℝ) (d3 : ℝ) (d2 : ℝ) (d1 : ℝ) (h1 : r = 65) (h2 : d3 = 90) 
  (h3 : d1 = 2 * r)
  (h4 : d2 = Real.sqrt(d1^2 - d3^2)) : ℝ := d1 + d2 + d3

theorem fly_distance_is_314
  (r : ℝ) (h1 : r = 65) (h2 : Real.sqrt ((2 * r)^2 - 90^2) = 94) :
  total_fly_distance r 90 (Real.sqrt((2 * r)^2 - 90^2)) (2 * r) h1 90 (2 * r) 94 = 314 :=
by
  rw [total_fly_distance, Real.sqrt, h1]
  linarith

end fly_distance_is_314_l725_725010


namespace MI_perpendicular_AC_l725_725553

noncomputable def incenter (A B C : Point) : Point := sorry
noncomputable def intersection (L₁ L₂ : Line) : Point := sorry
noncomputable def is_perpendicular (L₁ L₂ : Line) : Prop := sorry

def Point := sorry
def Line := sorry

theorem MI_perpendicular_AC
  (A B C K L M I : Point)
  (AC_shortest : ∀ X Y Z : Point, (X = A → Y = C → Z = B → distance A C < distance A B ∧ distance A C < distance B C))
  (K_on_AB : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ K = A + t • (B - A))
  (L_on_CB : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ L = C + t • (B - C))
  (KA_CL_eq_AC : distance K A = distance C L ∧ distance K A = distance A C)
  (M_intersection : M = intersection (line A L) (line K C))
  (I_incenter : I = incenter A B C) :
  is_perpendicular (line M I) (line A C) :=
sorry

end MI_perpendicular_AC_l725_725553


namespace intersection_of_sets_l725_725144

noncomputable def universal_set (x : ℝ) := true

def set_A (x : ℝ) : Prop := x^2 - 2 * x - 3 < 0

def set_B (x : ℝ) : Prop := ∃ y, y = Real.log (1 - x)

def complement_U_B (x : ℝ) : Prop := ¬ set_B x

theorem intersection_of_sets :
  { x : ℝ | set_A x } ∩ { x | complement_U_B x } = { x : ℝ | 1 ≤ x ∧ x < 3 } :=
by
  sorry

end intersection_of_sets_l725_725144


namespace percentage_error_l725_725763

theorem percentage_error (x : ℚ) : 
  let incorrect_result := (3/5 : ℚ) * x
  let correct_result := (5/3 : ℚ) * x
  let ratio := incorrect_result / correct_result
  let percentage_error := (1 - ratio) * 100
  percentage_error = 64 :=
by
  let incorrect_result := (3/5 : ℚ) * x
  let correct_result := (5/3 : ℚ) * x
  let ratio := incorrect_result / correct_result
  let percentage_error := (1 - ratio) * 100
  sorry

end percentage_error_l725_725763


namespace responses_needed_l725_725533

-- Define the given conditions
def rate : ℝ := 0.80
def num_mailed : ℕ := 375

-- Statement to prove
theorem responses_needed :
  rate * num_mailed = 300 := by
  sorry

end responses_needed_l725_725533


namespace min_spend_for_free_delivery_l725_725378

theorem min_spend_for_free_delivery : 
  let chicken_price := 1.5 * 6.00
  let lettuce_price := 3.00
  let tomato_price := 2.50
  let sweet_potato_price := 4 * 0.75
  let broccoli_price := 2 * 2.00
  let brussel_sprouts_price := 2.50
  let current_total := chicken_price + lettuce_price + tomato_price + sweet_potato_price + broccoli_price + brussel_sprouts_price
  let additional_needed := 11.00 
  let minimum_spend := current_total + additional_needed
  minimum_spend = 35.00 :=
by
  sorry

end min_spend_for_free_delivery_l725_725378


namespace symmetry_of_K_star_diameter_width_of_K_star_length_of_K_star_area_of_K_star_l725_725588

-- Given conditions
def convex_curve (K : Type) : Prop := sorry
def symmetric_to (K K' : Type) (O : Type) : Prop := sorry
def arithmetic_mean_curve (K K' K_star : Type) : Prop :=
  K_star = (1/2 : ℝ) • (K + K')

-- Questions translated to Lean statements
theorem symmetry_of_K_star 
  (K K' K_star O : Type)
  (h1 : convex_curve K)
  (h2 : symmetric_to K K' O)
  (h3 : arithmetic_mean_curve K K' K_star) :
  symmetric_to K_star K_star O :=
sorry

theorem diameter_width_of_K_star 
  (K K' K_star : Type)
  (h1 : convex_curve K)
  (h2 : symmetric_to K K' K_star)
  (h3 : arithmetic_mean_curve K K' K_star)
  (diameter width : Type)
  (h4 : diameter K = diameter K')
  (h5 : width K = width K') :
diameter K_star = diameter K ∧ width K_star = width K :=
sorry

theorem length_of_K_star 
  (K K' K_star : Type)
  (h1 : convex_curve K)
  (h2 : symmetric_to K K' K_star)
  (h3 : arithmetic_mean_curve K K' K_star)
  (length : Type)
  (h4 : length K = length K') :
length K_star = length K :=
sorry

theorem area_of_K_star 
  (K K' K_star : Type)
  (h1 : convex_curve K)
  (h2 : symmetric_to K K' K_star)
  (h3 : arithmetic_mean_curve K K' K_star)
  (area : Type)
  (h4 : area K ≤ area K_star) :
area K_star ≥ area K :=
sorry

end symmetry_of_K_star_diameter_width_of_K_star_length_of_K_star_area_of_K_star_l725_725588


namespace at_least_six_heads_in_10_flips_is_129_over_1024_l725_725347

def fair_coin_flip (n : ℕ) (prob_heads prob_tails : ℚ) : Prop :=
  (prob_heads = 1/2 ∧ prob_tails = 1/2)

noncomputable def at_least_six_consecutive_heads_probability (n : ℕ) : ℚ :=
  if n = 10 then 129 / 1024 else 0  -- this is specific to 10 flips and should be defined based on actual calculation for different n
  
theorem at_least_six_heads_in_10_flips_is_129_over_1024 :
  fair_coin_flip 10 (1/2) (1/2) →
  at_least_six_consecutive_heads_probability 10 = 129 / 1024 :=
by
  intros
  sorry

end at_least_six_heads_in_10_flips_is_129_over_1024_l725_725347


namespace boys_laps_l725_725925

-- Definition of variables for the problem
def girls_laps := 54
def additional_laps := 20

-- The theorem we want to prove
theorem boys_laps :
  ∃ b : ℕ, girls_laps = b + additional_laps ∧ b = 34 :=
by
  existsi (34 : ℕ)
  split
  · simp only [girls_laps, additional_laps]
    norm_num
  · refl

end boys_laps_l725_725925


namespace larger_number_hcf_lcm_l725_725321

theorem larger_number_hcf_lcm (a b : ℕ) (hcf : ℕ) (factor1 factor2 : ℕ) 
  (h_hcf : hcf = 20) 
  (h_factor1 : factor1 = 13) 
  (h_factor2 : factor2 = 14) 
  (h_ab_hcf : Nat.gcd a b = hcf)
  (h_ab_lcm : Nat.lcm a b = hcf * factor1 * factor2) :
  max a b = 280 :=
by 
  sorry

end larger_number_hcf_lcm_l725_725321


namespace arctan_sum_lt_pi_div_two_iff_arctan_sum_lt_pi_iff_l725_725395

open Real

theorem arctan_sum_lt_pi_div_two_iff (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  arctan x + arctan y < (π / 2) ↔ x * y < 1 :=
sorry

theorem arctan_sum_lt_pi_iff (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  arctan x + arctan y + arctan z < π ↔ x * y * z < x + y + z :=
sorry

end arctan_sum_lt_pi_div_two_iff_arctan_sum_lt_pi_iff_l725_725395


namespace num_positive_integers_count_positive_integers_l725_725315

theorem num_positive_integers (x : ℕ) : 
  2 * x^4 + 10 * x^3 + 44 * x^2 + 160 * x + 192 < 1000 → x = 1 ∨ x = 2 :=
begin
  intros h,
  sorry
end

theorem count_positive_integers :
  ∃ n : ℕ, n = 2 ∧ ∀ x : ℕ, (2 * x^4 + 10 * x^3 + 44 * x^2 + 160 * x + 192 < 1000) → (x = 1 ∨ x = 2) :=
begin
  use 2,
  split,
  { refl },
  { intros x h,
    sorry }
end

end num_positive_integers_count_positive_integers_l725_725315


namespace fraction_of_areas_l725_725626

/-- Points A, B, C, X, Y, Z coordinates definitions --/
structure Point :=
(x : ℝ)
(y : ℝ)

def A := Point.mk 2 0
def B := Point.mk 8 12
def C := Point.mk 14 0

def X := Point.mk 6 0
def Y := Point.mk 8 4
def Z := Point.mk 10 0

/-- Area of a triangle given base and height --/
def area_triangle (base height : ℝ) : ℝ :=
  (base * height) / 2

/-- Area of triangle ABC --/
def Area_ABC := area_triangle (C.x - A.x) B.y

/-- Area of triangle XYZ --/
def Area_XYZ := area_triangle (Z.x - X.x) Y.y

theorem fraction_of_areas : Area_XYZ / Area_ABC = 1 / 9 := by
  sorry

end fraction_of_areas_l725_725626


namespace tom_initial_investment_l725_725698

-- Given conditions
variables (D : ℕ) (T : ℕ)
def JoseInvestment := 4500
def TotalProfit := 6300
def JoseProfit := 3500
def PercentProfitShare := 4 / 5

-- Given: Duration of the business in months
variable (duration : ℕ := 12)
-- Calculates Tom's share of the profit
def TomProfit := TotalProfit - JoseProfit

-- Proof goal: Prove Tom's initial investment is Rs. 3000
theorem tom_initial_investment : T = 3000 :=
by
  have H1: D = 12 := rfl
  have H2: T * D / (JoseInvestment * (D - 2)) = TomProfit / JoseProfit := sorry
  have H3: T * 12 / (4500 * 10) = 2800 / 3500 := sorry
  have H4: T * 12 / 45000 = 2800 / 3500 := sorry
  have H5: T * 12 = 36000 := sorry
  have H6: T = 3000 := by linarith
  exact H6

end tom_initial_investment_l725_725698


namespace sample_standard_deviation_l725_725105

theorem sample_standard_deviation (sample : List ℝ) (h : sample = [3, 7, 4, 6, 5]) : 
  let n := (sample.length : ℝ)
  let mean := (sample.sum / n)
  let variance := (∑ x in sample, (x - mean)^2) / n
  sqrt variance = sqrt 2 := 
by
  sorry

end sample_standard_deviation_l725_725105


namespace six_digit_palindromes_count_l725_725164

def is_digit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9
def is_non_zero_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

theorem six_digit_palindromes_count : 
  (∃a b c : ℕ, is_non_zero_digit a ∧ is_digit b ∧ is_digit c) → 
  (∃ n : ℕ, n = 900) :=
by
  sorry

end six_digit_palindromes_count_l725_725164


namespace range_of_a_l725_725137

-- Function definition
def f (x a : ℝ) : ℝ := -x^3 + 3 * a^2 * x - 4 * a

-- Main theorem statement
theorem range_of_a (a : ℝ) (h : a > 0) :
  (∀ x, f x a = 0) ↔ (a ∈ Set.Ioi (Real.sqrt 2)) :=
sorry

end range_of_a_l725_725137


namespace couples_at_prom_l725_725388

theorem couples_at_prom (total_students attending_alone attending_with_partners couples : ℕ) 
  (h1 : total_students = 123) 
  (h2 : attending_alone = 3) 
  (h3 : attending_with_partners = total_students - attending_alone) 
  (h4 : couples = attending_with_partners / 2) : 
  couples = 60 := 
by 
  sorry

end couples_at_prom_l725_725388


namespace sum_of_squares_iff_one_mod_4_l725_725234

theorem sum_of_squares_iff_one_mod_4 (p : ℕ) [Fact (nat.prime p)] (hp_odd : p % 2 = 1) :
  (∃ a b : ℕ, p = a^2 + b^2) ↔ (p % 4 = 1) :=
sorry

end sum_of_squares_iff_one_mod_4_l725_725234


namespace equilateral_triangle_CEF_l725_725974

variables {A B C D E F : Type} [AddCommGroup A] [AffineSpace A E] [HasSmul R A]

-- Definitions of geometric objects based on conditions
def is_parallelogram (A B C D : E) : Prop :=
  (B - A) + (D - A) = (C - A) + (D - C)

def is_equilateral_triangle (A B C : E) : Prop :=
  A = B ∧ B = C ∧ angle A B C = 60

-- The problem statement in Lean
theorem equilateral_triangle_CEF (A B C D E F : E)
  (h_parallelogram: is_parallelogram A B C D)
  (h_eq_tri_abf: is_equilateral_triangle A B F)
  (h_eq_tri_ade: is_equilateral_triangle A D E) :
  is_equilateral_triangle C E F :=
sorry

end equilateral_triangle_CEF_l725_725974


namespace conjugate_of_z_l725_725949

def z : ℂ := 10 * complex.i / (3 + complex.i)

theorem conjugate_of_z : complex.conj z = 1 - 3 * complex.i :=
by
  sorry

end conjugate_of_z_l725_725949


namespace smallest_x_max_f_l725_725049

noncomputable def f (x : ℝ) : ℝ := real.sin (x / 4) + real.sin (x / 7)

theorem smallest_x_max_f :
  ∃ (x : ℝ), (x > 0) ∧ (∀ y, y > 0 → f(y) ≤ f(x)) ∧ (x = 2610) :=
sorry

end smallest_x_max_f_l725_725049


namespace triangle_area_multiplication_factor_l725_725541

theorem triangle_area_multiplication_factor
  (a b : ℝ) (θ : ℝ) :
  let A := (a * b * Real.sin θ) / 2 in
  let A' := (3 * a * b * Real.sin (θ + 15 * Real.pi / 180)) / 2 in
  (A' / A) = 3 * (Real.sin (θ + 15 * Real.pi / 180) / Real.sin θ) :=
by
  sorry

end triangle_area_multiplication_factor_l725_725541


namespace concyclic_l725_725280

noncomputable theory

open EuclideanGeometry

variables {A B C D E F X Y Z : Point}

/-- The statement of the problem in geometric terms. -/
theorem concyclic {ABC XBC : Triangle} 
  (h₁ : incircle_touches_sides_triangle ABC [BC, CA, AB] [D, E, F])
  (h₂ : X ∈ interior ABC)
  (h₃ : incircle_touches_sides_triangle XBC [BC, CX, XB] [D, Y, Z]) :
  concyclic E F Z Y :=
sorry

end concyclic_l725_725280


namespace part1_part2_l725_725504

open Real

noncomputable def f (x a : ℝ) : ℝ := 45 * abs (x - a) + 45 * abs (x - 5)

theorem part1 (a : ℝ) :
    (∀ (x : ℝ), f x a ≥ 3) ↔ (a ≤ 2 ∨ a ≥ 8) :=
sorry

theorem part2 (a : ℝ) (ha : a = 2) :
    ∀ (x : ℝ), (f x 2 ≥ x^2 - 8*x + 15) ↔ (2 ≤ x ∧ x ≤ 5 + Real.sqrt 3) :=
sorry

end part1_part2_l725_725504


namespace proof_problem_l725_725120

theorem proof_problem (a b c : ℤ) 
  (h1 : real.cbrt ((6 : ℤ) * a + 34) = 4)
  (h2 : real.sqrt ((5 : ℤ) * a + b - 2) = 5)
  (h3 : c = real.sqrt 9) :
  a = 5 ∧ b = 2 ∧ c = 3 ∧ real.sqrt (3 * a - b + c) = 4 ∨ real.sqrt (3 * a - b + c) = -4 :=
by
  sorry

end proof_problem_l725_725120


namespace certain_event_among_basketball_volleyball_l725_725828

theorem certain_event_among_basketball_volleyball :
  ∀ (balls : Finset (Fin 8)), 
    (balls.card = 6 + 2) → 
    -- Definitions of events:
    let event_A := ∀ (b : Finset (Fin 3)), b ⊆ balls → b.card = 3 → (∀ i ∈ b, i < 6)
    let event_B := ∃ (b : Finset (Fin 3)), b ⊆ balls ∧ b.card = 3 ∧ (∃ i ∈ b, 6 ≤ i)
    let event_C := ∃ (b : Finset (Fin 3)), b ⊆ balls ∧ b.card = 3 ∧ (∀ i ∈ b, 6 ≤ i)
    let event_D := ∀ (b : Finset (Fin 3)), b ⊆ balls → b.card = 3 → (∃ i ∈ b, i < 6)
    event_D
:= sorry

end certain_event_among_basketball_volleyball_l725_725828


namespace units_digit_17_pow_27_l725_725062

-- Define the problem: the units digit of 17^27
theorem units_digit_17_pow_27 : (17 ^ 27) % 10 = 3 :=
sorry

end units_digit_17_pow_27_l725_725062


namespace take_one_card_l725_725236

theorem take_one_card (L R : ℕ) (hL : L = 15) (hR : R = 20) : L + R = 35 := by
  rw [hL, hR]
  exact rfl

end take_one_card_l725_725236


namespace part1_part2_l725_725957

noncomputable def f (x a : ℝ) : ℝ := abs (x - 1) - 2 * abs (x + a)

theorem part1 (x : ℝ) : (∃ a, a = 1) → f x 1 > 1 ↔ -2 < x ∧ x < -(2/3) := by
  sorry

theorem part2 (a : ℝ) : (∀ x, 2 ≤ x → x ≤ 3 → f x a > 0) ↔ (-5/2) < a ∧ a < -2 := by
  sorry

end part1_part2_l725_725957


namespace cosine_value_neg_085_l725_725522

theorem cosine_value_neg_085 (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 360) (h3 : cos x = -0.85) : 
  (∃! x in Ico 0 360, cos x = -0.85) :=
sorry

end cosine_value_neg_085_l725_725522


namespace total_daisies_l725_725562

-- Define the conditions
def white_daisies : ℕ := 6
def pink_daisies : ℕ := 9 * white_daisies
def red_daisies : ℕ := 4 * pink_daisies - 3

-- Main statement to be proved
theorem total_daisies : white_daisies + pink_daisies + red_daisies = 273 := by
  sorry

end total_daisies_l725_725562


namespace exist_integers_xy_divisible_by_p_l725_725953

theorem exist_integers_xy_divisible_by_p (p : ℕ) [Fact (Nat.Prime p)] : ∃ x y : ℤ, (x^2 + y^2 + 2) % p = 0 := by
  sorry

end exist_integers_xy_divisible_by_p_l725_725953


namespace function_decreasing_range_a_l725_725664

theorem function_decreasing_range_a :
  ∀ (a : ℝ),
  (∀ x ∈ Ioi (-1 : ℝ), log 0.8 (2 * x^2 - a * x + 3) ≤ log 0.8 (2 * (x + ε) ^ 2 - a * (x + ε) + 3)) →
  a ∈ [-5, -4] :=
by
  sorry

end function_decreasing_range_a_l725_725664


namespace max_cookies_l725_725219

-- Definitions for the conditions
def John_money : ℕ := 2475
def cookie_cost : ℕ := 225

-- Statement of the problem
theorem max_cookies (x : ℕ) : cookie_cost * x ≤ John_money → x ≤ 11 :=
sorry

end max_cookies_l725_725219


namespace largest_apartment_size_l725_725777

theorem largest_apartment_size (rent_per_sqft : ℝ) (budget : ℝ) (s : ℝ) :
  rent_per_sqft = 0.9 →
  budget = 630 →
  s = budget / rent_per_sqft →
  s = 700 :=
by
  sorry

end largest_apartment_size_l725_725777


namespace average_of_roots_l725_725757

theorem average_of_roots (p q : ℝ) (h : ∀ r : ℝ, r^2 * (3 * p) + r * (-6 * p) + q = 0 → ∃ a b : ℝ, r = a ∨ r = b) : 
  ∀ (r1 r2 : ℝ), (3 * p) * r1^2 + (-6 * p) * r1 + q = 0 ∧ (3 * p) * r2^2 + (-6 * p) * r2 + q = 0 → 
  (r1 + r2) / 2 = 1 :=
by {
  sorry
}

end average_of_roots_l725_725757


namespace min_value_expression_l725_725480

theorem min_value_expression (x y : ℝ) (h1 : x < 0) (h2 : y < 0) (h3 : x + y = -1) :
  xy + (1 / xy) = 17 / 4 :=
sorry

end min_value_expression_l725_725480


namespace seongmin_work_days_l725_725568

theorem seongmin_work_days:
  let W : ℝ := 1 in
  let R1 : ℝ := (7/96) in
  let Rj : ℝ := (1/24) in
  let Rs := R1 - Rj in
  let Ds := W / Rs in
  Ds = 32 :=
by
  sorry

end seongmin_work_days_l725_725568


namespace carla_games_won_l725_725826

theorem carla_games_won (F C : ℕ) (h1 : F + C = 30) (h2 : F = C / 2) : C = 20 :=
by
  sorry

end carla_games_won_l725_725826


namespace polynomial_sum_correct_l725_725228

def f (x : ℝ) : ℝ := -4 * x^3 + 2 * x^2 - x - 5
def g (x : ℝ) : ℝ := -6 * x^3 - 7 * x^2 + 4 * x - 2
def h (x : ℝ) : ℝ := 2 * x^3 + 8 * x^2 + 6 * x + 3
def sum_polynomials (x : ℝ) : ℝ := -8 * x^3 + 3 * x^2 + 9 * x - 4

theorem polynomial_sum_correct (x : ℝ) : f x + g x + h x = sum_polynomials x :=
by sorry

end polynomial_sum_correct_l725_725228


namespace most_likely_outcome_l725_725469

-- Defining the conditions
def equally_likely (n : ℕ) (k : ℕ) := (Nat.choose n k) * (1 / 2)^n

-- Defining the problem statement
theorem most_likely_outcome :
  (equally_likely 5 3 = 5 / 16 ∧ equally_likely 5 2 = 5 / 16) :=
sorry

end most_likely_outcome_l725_725469


namespace dot_product_expression_l725_725527

variables (u v : ℝ^3)

-- Define the vector cross product condition
axiom cross_product_condition : u × v = ⟨3, -1, 2⟩

-- Define the proof statement using Lean syntax
theorem dot_product_expression :
  2 * (u • (u + 3 • v)) = 2 * (∥u∥^2 + 3 * (u • v)) :=
by
  sorry

end dot_product_expression_l725_725527


namespace harmonic_mean_lcm_gcd_sum_l725_725650

theorem harmonic_mean_lcm_gcd_sum {m n : ℕ} (h_lcm : Nat.lcm m n = 210) (h_gcd : Nat.gcd m n = 6) (h_sum : m + n = 72) :
  (1 / (m : ℚ) + 1 / (n : ℚ)) = 2 / 35 := 
sorry

end harmonic_mean_lcm_gcd_sum_l725_725650


namespace frequency_of_group5_l725_725769

-- Define the total number of students and the frequencies of each group
def total_students : ℕ := 40
def freq_group1 : ℕ := 12
def freq_group2 : ℕ := 10
def freq_group3 : ℕ := 6
def freq_group4 : ℕ := 8

-- Define the frequency of the fifth group in terms of the above frequencies
def freq_group5 : ℕ := total_students - (freq_group1 + freq_group2 + freq_group3 + freq_group4)

-- The theorem to be proven
theorem frequency_of_group5 : freq_group5 = 4 := by
  -- Proof goes here, skipped with sorry
  sorry

end frequency_of_group5_l725_725769


namespace period_f_monotonically_increasing_interval_f_range_f_in_interval_l725_725866

noncomputable def f (x : ℝ) : ℝ := (sin x + cos x)^2 + 2 * sin (x - π / 4) * sin (x + π / 4)

theorem period_f : ∃ (T : ℝ), T = π ∧ (∀ (x : ℝ), f (x + T) = f x) := sorry

theorem monotonically_increasing_interval_f : ∃ (I : ℝ × ℝ) (k : ℤ), I = (-π / 8 + k * π, 3 * π / 4 + k * π] :=
sorry

theorem range_f_in_interval : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ π / 2 → 0 ≤ f x ∧ f x ≤ sqrt 2 + 1 := sorry

end period_f_monotonically_increasing_interval_f_range_f_in_interval_l725_725866


namespace min_sum_at_20_l725_725214

noncomputable def a (n : ℕ) : ℕ → ℚ
| 0       := -13
| (n + 1) := a n + 2 / 3

def S (n : ℕ) : ℚ := ∑ i in finset.range n, a i

theorem min_sum_at_20 : ∀ (n : ℕ), S 20 ≤ S n := sorry

end min_sum_at_20_l725_725214


namespace min_selling_price_l725_725759

-- Average sales per month
def avg_sales := 50

-- Cost per refrigerator
def cost_per_fridge := 1200

-- Shipping fee per refrigerator
def shipping_fee_per_fridge := 20

-- Monthly storefront fee
def monthly_storefront_fee := 10000

-- Monthly repair costs
def monthly_repair_costs := 5000

-- Profit margin requirement
def profit_margin := 0.2

-- The minimum selling price for the shop to maintain at least 20% profit margin
theorem min_selling_price 
  (avg_sales : ℕ) 
  (cost_per_fridge : ℕ) 
  (shipping_fee_per_fridge : ℕ) 
  (monthly_storefront_fee : ℕ) 
  (monthly_repair_costs : ℕ) 
  (profit_margin : ℝ) : 
  ∃ x : ℝ, 
    (50 * x - ((cost_per_fridge + shipping_fee_per_fridge) * avg_sales + monthly_storefront_fee + monthly_repair_costs)) 
    ≥ (cost_per_fridge + shipping_fee_per_fridge) * avg_sales + monthly_storefront_fee + monthly_repair_costs * profit_margin 
    → x ≥ 1824 :=
by 
  sorry

end min_selling_price_l725_725759


namespace base_eight_seventeen_five_is_one_two_five_l725_725705

def base_eight_to_base_ten (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | _ => (n / 100) * 8^2 + ((n % 100) / 10) * 8^1 + (n % 10) * 8^0

theorem base_eight_seventeen_five_is_one_two_five :
  base_eight_to_base_ten 175 = 125 :=
by
  sorry

end base_eight_seventeen_five_is_one_two_five_l725_725705


namespace curve_C_equation_no_line_B_PQ_equal_distance_l725_725611

namespace EquivalenceProblem

-- Step a): We define the conditions as given in the problem
def origin : ℝ × ℝ := ⟨0, 0⟩
def point_O : ℝ × ℝ := origin
def point_A : ℝ × ℝ := ⟨5, 0⟩
def point_B : ℝ × ℝ := ⟨1, 0⟩

def trajectory_of_T := { p : ℝ × ℝ | ∃ (x y : ℝ),
  p = (x, y) ∧ (x^2 / 5 + y^2 / 4 = 1) }

theorem curve_C_equation :
  ∀ (x y : ℝ), (x, y) ∈ trajectory_of_T ↔ (x^2 / 5 + y^2 / 4 = 1) :=
by sorry

theorem no_line_B_PQ_equal_distance :
  ∀ (k : ℝ), 
  (- (sqrt 5) / 5 < k ∧ k < (sqrt 5) / 5) →
  ¬ (∃ (P Q : ℝ × ℝ), 
      P ≠ Q ∧
      is_line_through_point A k P Q ∧
      abs ((fst B - fst P) * (fst B - fst P) + (snd B - snd P) * (snd B - snd P)) = 
      abs ((fst B - fst Q) * (fst B - fst Q) + (snd B - snd Q) * (snd B - snd Q))) :=
by sorry

end EquivalenceProblem

end curve_C_equation_no_line_B_PQ_equal_distance_l725_725611


namespace triangle_not_divisible_l725_725748

/-- A polygon is considered "good" if its area (measured in cm²) is numerically equal to its perimeter (measured in cm). -/
def is_good (polygon : Type) (area perimeter : ℝ) : Prop :=
  area = perimeter

/-- A right-angled triangle with side lengths 5 cm, 12 cm, and 13 cm cannot be subdivided into multiple "good" polygons (i.e., polygons each with an area equal to its perimeter). -/
theorem triangle_not_divisible (
  sides : ℝ → ℝ → ℝ → Type,
  good_triangle : sides 5 12 13
) : ¬ (∃ polygons : list (ℝ → ℝ → Prop), 
      list.length polygons > 1 ∧ 
      ∀ p ∈ polygons, ∃ (area perimeter : ℝ), is_good p area perimeter) :=
sorry

end triangle_not_divisible_l725_725748


namespace complement_of_angle_l725_725657

theorem complement_of_angle (x : ℝ) (h1 : 3 * x + 10 = 90 - x) : 3 * x + 10 = 70 :=
by
  sorry

end complement_of_angle_l725_725657


namespace max_face_sum_l725_725615

open Finset

-- Define the set of vertex values
def vertices : Finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7}

-- Define the set of prime numbers that can be the sum of two vertices
def primes : Finset ℕ := {2, 3, 5, 7, 11, 13}

-- Define a function to check if the sum of two numbers is prime
def is_prime_sum (a b : ℕ) : Prop := a + b ∈ primes

-- Define a cube face as a set of four vertex indices
def face_indices : ℕ → Finset (Fin 8) :=
  λ i, match i with
    | 0 => {0, 1, 2, 3}
    | 1 => {0, 1, 5, 4}
    | 2 => {1, 2, 6, 5}
    | 3 => {2, 3, 7, 6}
    | 4 => {0, 3, 7, 4}
    | 5 => {4, 5, 6, 7}
    | _ => ∅
  end

-- Define a function to compute the sum of values at given vertices
def face_sum (v : Fin 8 → ℕ) (f : Finset (Fin 8)) : ℕ :=
  f.sum (λ i, v i)

-- Define the condition that all edges of a cube have sums that are prime
def valid_cube (v : Fin 8 → ℕ) : Prop :=
  (∀ i j : Fin 8, (i ≠ j) → ((i, j).to_finset ⊆ face_indices 0) ∨ ((i, j).to_finset ⊆ face_indices 1) ∨
    ((i, j).to_finset ⊆ face_indices 2) ∨ ((i, j).to_finset ⊆ face_indices 3) ∨ ((i, j).to_finset ⊆ face_indices 4) ∨ ((i, j).to_finset ⊆ face_indices 5) →
    is_prime_sum (v i) (v j))

-- The main theorem stating the maximum sum of a face in the valid configuration
theorem max_face_sum : ∃ v : Fin 8 → ℕ, valid_cube v ∧ (∀ i, v i ∈ vertices) ∧ max (face_sum v (face_indices 0)) (max (face_sum v (face_indices 1)) (max (face_sum v (face_indices 2)) (max (face_sum v (face_indices 3)) (max (face_sum v (face_indices 4)) (face_sum v (face_indices 5))))) = 18 :=
sorry

end max_face_sum_l725_725615


namespace sum_of_integers_7_to_10_l725_725083

theorem sum_of_integers_7_to_10 (n : ℕ) (h1 : 1.5 * n - 5.5 > 4.5) (h2 : 7 ≤ n ∧ n ≤ 10) :
  ∑ k in Finset.filter (λ n, 7 ≤ n ∧ n ≤ 10) (Finset.range 11), k = 34 :=
sorry

end sum_of_integers_7_to_10_l725_725083


namespace h_of_2_l725_725175

theorem h_of_2 {h : ℝ → ℝ} (h_prop : ∀ x : ℝ, h(3 * x - 4) = 4 * x + 12) : h 2 = 20 :=
by
  sorry

end h_of_2_l725_725175


namespace dihedral_angle_and_distance_l725_725210

-- Definitions for the given conditions
def square {α : Type} [field α] (A B C D : α) : Prop :=
  (A - B) = (B - C) ∧ (B - C) = (C - D) ∧ (C - D) = (D - A) ∧ (A - C) = 2

def isosceles_right_triangle {α : Type} [field α] (A E B : α) : Prop :=
  (A = E) ∧ (E = B) ∧ (E - B) = (E - A) ∧ (E - A) = sqrt(2)

def perpendicular {α : Type} [field α] (X Y Z : α) : Prop :=
  (X - Y) ⊥ (Y - Z)

-- The problem in mathematical proof
theorem dihedral_angle_and_distance
  {α : Type} [field α]
  (A B C D E F : α)
  (h1 : square A B C D)
  (h2 : isosceles_right_triangle A E B)
  (h3 : perpendicular B F (A ∩ C ∩ E))
  (h4 : E - B = sqrt(2))
  (h5 : B - G = sqrt(2)) :
  (dihedral_angle (B - A - C - E) = arcsin(sqrt(6) / 3)) ∧
  (distance_from_point_to_plane D (A ∩ C ∩ E) = 2 * sqrt(3) / 3) :=
sorry

end dihedral_angle_and_distance_l725_725210


namespace part_a_l725_725331

theorem part_a (S : Finset ℕ) (hS : S ⊆ (Finset.range 201).filter (λ n, n > 0)) (h_card : S.card = 101) :
  ∃ a b ∈ S, a ≠ b ∧ (a ∣ b ∨ b ∣ a) :=
by
  sorry

end part_a_l725_725331


namespace impossible_to_obtain_105_piles_of_one_token_l725_725217

theorem impossible_to_obtain_105_piles_of_one_token :
  ¬ ∃ (piles : list ℕ), 
    (piles = [51, 49, 5]) ∧ 
    (∀ (op : list ℕ → list ℕ), (op = combine_piles ∨ op = divide_even_pile) → 
      (op (combine_piles piles) = *) ∨ 
      (op (divide_even_pile piles) = *)) ∧ 
    (length piles = 105) ∧ 
    (∀ (pile : ℕ), pile = 1) :=
sorry

end impossible_to_obtain_105_piles_of_one_token_l725_725217


namespace correct_transformation_l725_725987

theorem correct_transformation (a b c d : ℕ) (h1 : b < 10) (h2 : c < 10) (h3 : d < 10):
  (( ((a * 2 + 1) * 5 + b) * 2 + 1) * 5 + c) * 2 + 1) * 5 + d - 555 = 1000 * a + 100 * b + 10 * c + d :=
by
  sorry

end correct_transformation_l725_725987


namespace additional_miles_l725_725795

theorem additional_miles (distance1 : ℕ) (speed1 : ℕ) (speed2 : ℕ) (target_speed : ℕ) (x : ℕ) :
  distance1 = 8 →
  speed1 = 20 →
  speed2 = 40 →
  target_speed = 30 →
  (8 + x) / (0.4 + x / 40) = 30 →
  x = 16 :=
by
  sorry

end additional_miles_l725_725795


namespace base_square_eq_l725_725916

theorem base_square_eq (b : ℕ) (h : (3*b + 3)^2 = b^3 + 2*b^2 + 3*b) : b = 9 :=
sorry

end base_square_eq_l725_725916


namespace equivalent_octal_to_decimal_l725_725366

def octal_to_decimal (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | n+1 => (n % 10) + 8 * octal_to_decimal (n / 10)

theorem equivalent_octal_to_decimal : octal_to_decimal 753 = 491 :=
by
  sorry

end equivalent_octal_to_decimal_l725_725366


namespace minimum_a_value_l725_725488

noncomputable def min_ellipse_a (c b : ℝ) (h1 : 0 < c) (h2 : 0 < b) (h3 : b < a) (h4 : b * c = 8) : ℝ :=
  sqrt (b^2 + c^2)

theorem minimum_a_value (c b a : ℝ) (h1 : 0 < c) (h2 : 0 < b) (h3 : b < a) (h4 : b * c = 8) :
  a ≥ 4 :=
begin
  suffices : a^2 ≥ 16,
  { apply le_of_sqrt_le_sqrt,
    all_goals {linarith} },
  have : a^2 = b^2 + c^2, from sorry,
  linarith,
end

end minimum_a_value_l725_725488


namespace length_of_chord_l725_725670

-- Defining the conditions and the theorem
theorem length_of_chord (x y : ℝ) :
  (∃ x y, y = x + 1 ∧ (x^2 / 4 + y^2 = 1)) →
  (∃ (A B : ℝ), 
    A = -8 / 5 ∧
    B = 0 ∧
    (( (A - B) * Real.sqrt(2) = 8 * Real.sqrt(2) / 5) :=
begin
  sorry
end

end length_of_chord_l725_725670


namespace count_valid_n_l725_725393

theorem count_valid_n :
  let a b c d n : ℝ 
  in let vertices := [(a, b), (a, b + 2 * c), (a - 2 * d, b)] in
  let midpoint1 := (a, b + c) in
  let midpoint2 := (a - d, b) in
  let slope1 := (midpoint1.2 - b) / (midpoint1.1 - a) in
  let slope2 := (midpoint2.2 - b) / (midpoint2.1 - a) in
  slope1 = 2 ∧ slope2 = n →
  let median1 := (y : ℝ) -> 2 * ((x : ℝ) ∈ ℝ) + 1 in
  let median2 := (y : ℝ) -> n * ((x : ℝ) ∈ ℝ) + 2 in
  (slope1 = 2 → ( ∃ a, a = 1 )) :=
sorry

end count_valid_n_l725_725393


namespace prove_area_and_eccentricity_of_ellipse_l725_725075

noncomputable def area_and_eccentricity_of_ellipse : Prop :=
  let ellipse_eq : Prop := 4 * x^2 + 12 * x + 9 * y^2 - 27 * y + 36 = 0 
  let area : ℝ := 15 * Real.pi / 8 
  let eccentricity : ℝ := Real.sqrt 5 / 3 
  ∀ x y : ℝ, ellipse_eq → 
    (∃ a b : ℝ, area = (Real.pi * a * b) ∧ 
      a = 3 * Real.sqrt 5 / 4 ∧ b = Real.sqrt 5 / 2) ∧ 
    (∃ e : ℝ, e = eccentricity) 

theorem prove_area_and_eccentricity_of_ellipse : area_and_eccentricity_of_ellipse := 
  sorry

end prove_area_and_eccentricity_of_ellipse_l725_725075


namespace sum_b_lt_3_l725_725514

def geom_seq (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 0 < n → a n < a (n + 1) ∧ ∃ q > 0, ∀ n : ℕ, a (n + 1) = a n * q

def b_sequence (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, b n = a n / ((a n - 1) ^ 2)

def sum_b_sequence (b : ℕ → ℝ) (T : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, T n = (finset.range (n + 1)).sum b

theorem sum_b_lt_3 (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ) 
  (h1 : geom_seq a)
  (h2 : a 1 + a 2 = 6) 
  (h3 : a 3 + a 4 = 24)
  (h4 : ∀ n : ℕ, a n = 2 ^ n)
  (h5 : b_sequence a b)
  (h6 : sum_b_sequence b T) :
  ∀ n : ℕ, T n < 3 :=
by 
  sorry

end sum_b_lt_3_l725_725514


namespace concurrency_of_lines_l725_725470

theorem concurrency_of_lines 
  (A B C D E F H E' F' X Y : Point) 
  (h_non_right : ¬is_right_triangle A B C)
  (h_perp_AD : is_foot_perpendicular A D B C)
  (h_perp_BE : is_foot_perpendicular B E A C)
  (h_perp_CF : is_foot_perpendicular C F A B)
  (h_orthocenter : is_orthocenter H A B C)
  (h_reflection_E : is_reflection E' E (line_through A D))
  (h_reflection_F : is_reflection F' F (line_through A D))
  (h_intersect_X : intersection (line_through B F') (line_through C E') = X)
  (h_intersect_Y : intersection (line_through B E') (line_through C F') = Y) :
  concurrency (line_through A X) (line_through B C) (line_through H Y) :=
sorry

end concurrency_of_lines_l725_725470


namespace Olivia_hours_worked_on_Monday_l725_725607

/-- Olivia works on multiple days in a week with given wages per hour and total income -/
theorem Olivia_hours_worked_on_Monday 
  (M : ℕ)  -- Hours worked on Monday
  (rate_per_hour : ℕ := 9) -- Olivia’s earning rate per hour
  (hours_Wednesday : ℕ := 3)  -- Hours worked on Wednesday
  (hours_Friday : ℕ := 6)  -- Hours worked on Friday
  (total_income : ℕ := 117)  -- Total income earned this week
  (hours_total : ℕ := hours_Wednesday + hours_Friday + M)
  (income_calc : ℕ := rate_per_hour * hours_total) :
  -- Prove that the hours worked on Monday is 4 given the conditions
  income_calc = total_income → M = 4 :=
by
  sorry

end Olivia_hours_worked_on_Monday_l725_725607


namespace complement_of_intersection_l725_725961

open Set

-- Define the universal set U
def U := @univ ℝ
-- Define the sets M and N
def M : Set ℝ := {x | x >= 2}
def N : Set ℝ := {x | 0 <= x ∧ x < 5}

-- Define M ∩ N
def M_inter_N := M ∩ N

-- Define the complement of M ∩ N with respect to U
def C_U (A : Set ℝ) := Aᶜ

theorem complement_of_intersection :
  C_U M_inter_N = {x : ℝ | x < 2 ∨ x ≥ 5} := 
by 
  sorry

end complement_of_intersection_l725_725961


namespace area_of_region_l725_725421

noncomputable def sec (θ : ℝ) := (cos θ)⁻¹
noncomputable def csc (θ : ℝ) := (sin θ)⁻¹

def region (r θ : ℝ) : Prop :=
  (r = 2 * sec θ ∧ (0 ≤ θ ∧ θ ≤ π / 2)) ∨ 
  (r = 2 * csc θ ∧ (0 ≤ θ ∧ θ ≤ π / 2))

theorem area_of_region :
  let bounded_region := { (x, y) | 0 ≤ x ∧ x ≤ 2 ∧ 0 ≤ y ∧ y ≤ 2 } in
  ∃ (A : ℝ), A = 4 ∧ (∀ (a b : ℝ), bounded_region (a, b)) :=
begin
  let bounded_region := { p | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2 },
  use 4,
  split,
  { refl, },
  { intros a b hb,
    exact hb, },
end

end area_of_region_l725_725421


namespace collinear_and_magnitude_l725_725649

theorem collinear_and_magnitude (z : ℂ) (h1 : collinear ({1, ↑(1 : ℂ), z} : set ℂ))
  (h2 : complex.abs z = 5) : z = 4 - 3 * complex.I ∨ z = -3 + 4 * complex.I :=
by sorry

end collinear_and_magnitude_l725_725649


namespace eq_EF_FG_and_area_ratio_l725_725111

variables {α : Type} [euclidean_space α]
variables {A B C D G E F : α}
variable (Γ : circle α)

-- Conditions
axiom angle_ADB_eq_angle_ACB_add_90 (h1 : ∠ A D B = ∠ A C B + 90)
axiom AC_mul_BD_eq_AD_mul_BC (h2 : distance A C * distance B D = distance A D * distance B C)

-- To prove
theorem eq_EF_FG_and_area_ratio
  (h1 : ∠ A D B = ∠ A C B + 90)
  (h2 : distance A C * distance B D = distance A D * distance B C)
  (hEF : line A D ∩ Γ = G)
  (hBD : line B D ∩ Γ = E)
  (hCD : line C D ∩ Γ = F) :
  (distance E F = distance F G) ∧
  (area (triangle E F G) / area (circumcircle Γ) = 1 / π) := 
sorry

end eq_EF_FG_and_area_ratio_l725_725111


namespace lcm_36_100_eq_900_l725_725450

/-- Definition for the prime factorization of 36 -/
def factorization_36 : Prop := 36 = 2^2 * 3^2

/-- Definition for the prime factorization of 100 -/
def factorization_100 : Prop := 100 = 2^2 * 5^2

/-- The least common multiple problem statement -/
theorem lcm_36_100_eq_900 (h₁ : factorization_36) (h₂ : factorization_100) : Nat.lcm 36 100 = 900 := 
by
  sorry

end lcm_36_100_eq_900_l725_725450


namespace jamie_speed_l725_725377

theorem jamie_speed (alex_speed : ℝ) (sam_ratio : ℝ) (jamie_ratio : ℝ) (h1 : alex_speed = 6) (h2 : sam_ratio = 3 / 4) (h3 : jamie_ratio = 4 / 3) :
  let sam_speed := sam_ratio * alex_speed,
      jamie_speed := jamie_ratio * sam_speed in
  jamie_speed = 6 :=
sorry

end jamie_speed_l725_725377


namespace area_of_bounded_region_l725_725438

theorem area_of_bounded_region : 
  let x1 := 2
  let y1 := 2
  let x0 := 0
  let y0 := 0
  in
  let area := (x1 - x0) * (y1 - y0)
  in
  area = 4 :=
by 
  -- definitions for x1, y1, x0, y0
  let x1 := 2
  let y1 := 2
  let x0 := 0
  let y0 := 0
  
  -- let area be the area of the square bounded by these lines
  let area := (x1 - x0) * (y1 - y0)
  
  -- assertion
  have h : area = (2 - 0) * (2 - 0), from rfl,
  
  -- proving the final statement
  show area = 4, by 
    rw [h],
    exact rfl

-- skipped proof step
sorry

end area_of_bounded_region_l725_725438


namespace initial_decrease_l725_725536

-- Definitions derived from conditions
def initial_price (P : ℝ) : ℝ := P
def decreased_price (P x : ℝ) : ℝ := P * (1 - x / 100)
def increased_price_after_decrease (P x : ℝ) : ℝ := decreased_price P x * 1.10
def final_price_expected (P : ℝ) : ℝ := P * 1.065

-- The theorem to prove
theorem initial_decrease (P : ℝ) (x : ℝ) :
  increased_price_after_decrease P x = final_price_expected P → x = 3.18 := 
sorry

end initial_decrease_l725_725536


namespace how_many_pans_l725_725567

-- Definitions based on given conditions
def cost_per_pan : ℝ := 10
def sell_price_per_pan : ℝ := 25
def profit : ℝ := 300

-- The number of pans Jenny makes and sells over the weekend
def number_of_pans_made_and_sold : ℝ :=
  profit / (sell_price_per_pan - cost_per_pan)

theorem how_many_pans (x : ℝ) :
  x = profit / (sell_price_per_pan - cost_per_pan) :=
by
  sorry

end how_many_pans_l725_725567


namespace lcm_36_100_is_900_l725_725455

def prime_factors_36 : ℕ → Prop := 
  λ n, n = 36 → (2^2 * 3^2)

def prime_factors_100 : ℕ → Prop := 
  λ n, n = 100 → (2^2 * 5^2)

def lcm_36_100 := lcm 36 100

theorem lcm_36_100_is_900 : lcm_36_100 = 900 :=
by {
  sorry,
}

end lcm_36_100_is_900_l725_725455


namespace part1_part2_l725_725875

theorem part1 (m : ℝ) :
  ∀ x : ℝ, x^2 + ( (2 * m - 1) : ℝ) * x + m^2 = 0 → m ≤ 1 / 4 :=
sorry

theorem part2 (m : ℝ) 
  (h : ∀ x1 x2 : ℝ, (x1^2 + (2*m -1)*x1 + m^2 = 0) ∧ (x2^2 + (2*m -1)*x2 + m^2 = 0) ∧ (x1*x2 + x1 + x2 = 4)) :
    m = -1 :=
sorry

end part1_part2_l725_725875


namespace asha_remaining_money_l725_725041

theorem asha_remaining_money :
  let brother := 20
  let father := 40
  let mother := 30
  let granny := 70
  let savings := 100
  let total_money := brother + father + mother + granny + savings
  let spent := (3 / 4) * total_money
  let remaining := total_money - spent
  remaining = 65 :=
by
  sorry

end asha_remaining_money_l725_725041


namespace correct_propositions_l725_725587

-- Definitions based on problem conditions
variables {m n : Line} {α β : Plane} {l : Line}

-- Proposition 1: If m is contained in α and n is parallel to α, then m is parallel to n.
def proposition1 : Prop := (m ⊆ α ∧ n ∥ α) → (m ∥ n)

-- Proposition 2: If m is contained in α, n is contained in β, α is perpendicular to β,
-- and the intersection of α and β is l, and m is perpendicular to l, then m is perpendicular to n.
def proposition2 : Prop := (m ⊆ α ∧ n ⊆ β ∧ α ⟂ β ∧ α ∩ β = l ∧ m ⟂ l) → (m ⟂ n)

-- Proposition 3: If m is perpendicular to α and m is perpendicular to n, then n is parallel to α.
def proposition3 : Prop := (m ⟂ α ∧ m ⟂ n) → (n ∥ α)

-- Proposition 4: If m is perpendicular to α and m is perpendicular to β, then α is parallel to β.
def proposition4 : Prop := (m ⟂ α ∧ m ⟂ β) → (α ∥ β)

-- Proposition 5: If α is perpendicular to β, m is perpendicular to α, and n is parallel to β, then m is parallel to n.
def proposition5 : Prop := (α ⟂ β ∧ m ⟂ α ∧ n ∥ β) → (m ∥ n)

-- The goal is to prove that Proposition 2 and Proposition 4 are correct
theorem correct_propositions : proposition2 ∧ proposition4 := sorry

end correct_propositions_l725_725587


namespace megan_days_per_month_l725_725602

noncomputable def daily_earnings (hourly_wage : ℝ) (hours_per_day : ℕ) : ℝ :=
hourly_wage * hours_per_day

noncomputable def total_days (total_earnings : ℝ) (daily_earnings : ℝ) : ℕ :=
(total_earnings / daily_earnings).toNat

noncomputable def days_per_month (total_days : ℕ) (months : ℕ) : ℕ :=
total_days / months

theorem megan_days_per_month (hours_per_day : ℕ) (hourly_wage : ℝ) (total_earnings : ℝ) :
  hours_per_day = 8 → hourly_wage = 7.50 → total_earnings = 2400 → days_per_month (total_days total_earnings (daily_earnings hourly_wage hours_per_day)) 2 = 20 :=
by
  intros h_hours h_wage h_earnings
  sorry

end megan_days_per_month_l725_725602


namespace max_value_of_f_l725_725889

noncomputable def f (x m : ℝ) : ℝ := (sqrt 3 * sin x) * (cos x) + (m + cos x) * (-m + cos x)

theorem max_value_of_f 
  (h_min: ∀ x ∈ set.Icc (-real.pi / 6) (real.pi / 3), f x m ≥ -4) 
  (h_min_value: ∃ x ∈ set.Icc (-real.pi / 6) (real.pi / 3), f x m = -4)
  (hx : ∃ x ∈ set.Icc (-real.pi / 6) (real.pi / 3), 
        sin (2*x + real.pi/6) = 1 ∧ cos (2*x) = 1 ∧ x = real.pi / 6) :
  ∃ x ∈ set.Icc (-real.pi / 6) (real.pi / 3), 
    f x 2 = -3 / 2 ∨ f x (- 2) = -3 / 2 :=
sorry

end max_value_of_f_l725_725889


namespace points_closer_to_B_l725_725838

-- Define the conditions
variable (C : Type) [metric_space C] 
variable (P B D : C)
variable (r : ℝ)
variable (on_circle_B : dist P B = r)
variable (on_circle_D : dist P D = r)

-- Define the question as a theorem
theorem points_closer_to_B :=
  -- We need to define the locus of points A satisfying the given condition
  ∃ (A : C), dist A B < dist A D :=
  sorry

end points_closer_to_B_l725_725838


namespace nth_element_formula_l725_725792

def sequence_nth_element (n : ℕ) : ℕ :=
    2 * n - Int.floor ((1 + Real.sqrt (8 * n - 7 : ℝ)) / 2)

theorem nth_element_formula (n : ℕ) : 
    -- Check if the n-th element follows the given formula.
    sequence_nth_element n = 
        2 * n - Int.floor ((1 + Real.sqrt (8 * n - 7 : ℝ)) / 2) := 
    by sorry

end nth_element_formula_l725_725792


namespace find_wrongly_written_height_l725_725654

noncomputable def wrongly_written_height (n : ℕ) (average_wrong : ℝ) (h_actual : ℝ) (average_correct : ℝ) :=
  let total_wrong := n * average_wrong
  let total_correct_wrong := total_wrong - x + h_actual
  let total_correct := n * average_correct
  total_correct = total_correct_wrong → x = 176

theorem find_wrongly_written_height :
  wrongly_written_height 35 185 106 183 = 176 :=
begin
  sorry
end

end find_wrongly_written_height_l725_725654


namespace find_f2_l725_725276

noncomputable def f (x : ℝ) : ℝ := sorry -- Placeholder function definition

theorem find_f2 (h : ∀ x : ℝ, f x + 2 * f (1 - x) = x^3 + 1) : f 2 = -3 :=
by
  -- Lean proof goes here
  sorry

end find_f2_l725_725276


namespace stone_division_ways_l725_725172

theorem stone_division_ways :
  ∃ k : ℕ, 2 ≤ k ∧ k ≤ 100 ∧ ∀ (q r : ℕ), 100 = k * q + r → r < k →
  ∀ a b : ℕ, a ≠ b →
  ((a < k → a * q + r = 100) ∧ (b < k → b * q + r = 100))
  ∧ ∑ (i : ℕ) in finset.range k, (if i < k - r then q else q + 1) = 100
 :=
begin
  sorry
end

end stone_division_ways_l725_725172


namespace fractional_part_tiled_l725_725405

def room_length : ℕ := 12
def room_width : ℕ := 20
def number_of_tiles : ℕ := 40
def tile_area : ℕ := 1

theorem fractional_part_tiled :
  (number_of_tiles * tile_area : ℚ) / (room_length * room_width) = 1 / 6 :=
by
  sorry

end fractional_part_tiled_l725_725405


namespace lcm_of_36_and_100_l725_725444

theorem lcm_of_36_and_100 : Nat.lcm 36 100 = 900 :=
by
  -- The proof is omitted
  sorry

end lcm_of_36_and_100_l725_725444


namespace table_conditions_2015_l725_725187

theorem table_conditions_2015 (n : ℕ) (table : Fin 2015 → Fin n → ℕ) :
  (∀ i : Fin 2015, ∃ j : Fin n, 0 < table i j) ∧   -- Each row has a positive number
  (∀ j : Fin n, ∃ i : Fin 2015, 0 < table i j) ∧   -- Each column has a positive number
  (∀ i : Fin 2015, ∀ j : Fin n, 
     0 < table i j → 
     (∑ k : Fin n, table i k) = (∑ k : Fin 2015, table k j))   -- Positive cell condition
  → n = 2015 := 
by
  sorry

end table_conditions_2015_l725_725187


namespace total_daisies_l725_725556

-- Define the initial conditions
def white_daisies : Nat := 6
def pink_daisies : Nat := 9 * white_daisies
def red_daisies : Nat := 4 * pink_daisies - 3

-- The main theorem stating that the total number of daisies is 273
theorem total_daisies : white_daisies + pink_daisies + red_daisies = 273 := by
  -- The proof is left as an exercise
  sorry

end total_daisies_l725_725556


namespace six_digit_palindromes_count_l725_725166

def is_digit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9
def is_non_zero_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

theorem six_digit_palindromes_count : 
  (∃a b c : ℕ, is_non_zero_digit a ∧ is_digit b ∧ is_digit c) → 
  (∃ n : ℕ, n = 900) :=
by
  sorry

end six_digit_palindromes_count_l725_725166


namespace figure_two_total_length_l725_725026

theorem figure_two_total_length (vertical_left : ℕ) (horizontal_top1 : ℕ) (vertical_middle : ℕ) (horizontal_top2 : ℕ) (vertical_bottom : ℕ)
  (H1 : vertical_left = 7) (H2 : horizontal_top1 = 3) (H3 : vertical_middle = 2)
  (H4 : horizontal_top2 = 4) (H5 : vertical_bottom = 3) :
  vertical_left + (horizontal_top1 + horizontal_top2) + (vertical_middle + vertical_bottom) + horizontal_top2 = 23 := 
by
  -- applying conditions
  calc
        vertical_left + (horizontal_top1 + horizontal_top2) + (vertical_middle + vertical_bottom) + horizontal_top2
      = 7 + (3 + 4) + (2 + 3) + 4 : by rw [H1, H2, H3, H4, H5]
  ... = 7 + 7 + 5 + 4 : by rfl
  ... = 23 : by norm_num

end figure_two_total_length_l725_725026


namespace find_x_l725_725837

def delta (x : ℝ) : ℝ := 5 * x + 6
def phi (x : ℝ) : ℝ := 7 * x + 4

theorem find_x (x : ℝ) (h : delta(phi(x)) = 1) : x = -5 / 7 :=
by {
  sorry
}

end find_x_l725_725837


namespace field_trip_savings_l725_725004

-- Define the parameters given in the conditions
def num_students : ℕ := 30
def contribution_per_student_per_week : ℕ := 2
def weeks_per_month : ℕ := 4
def num_months : ℕ := 2

-- Define the weekly savings for the class
def weekly_savings : ℕ := num_students * contribution_per_student_per_week

-- Define the total weeks in the given number of months
def total_weeks : ℕ := num_months * weeks_per_month

-- Define the total savings in the given number of months
def total_savings : ℕ := weekly_savings * total_weeks

-- Now, we state the theorem
theorem field_trip_savings : total_savings = 480 :=
by {
  -- calculations are skipped
  sorry
}

end field_trip_savings_l725_725004


namespace infinite_primes_of_the_year_2022_l725_725380

theorem infinite_primes_of_the_year_2022 :
  ∃ᶠ p in Filter.atTop, ∃ n : ℕ, p % 2 = 1 ∧ p ^ 2022 ∣ n ^ 2022 + 2022 :=
sorry

end infinite_primes_of_the_year_2022_l725_725380


namespace sin_x_plus_y_eq_two_thirds_l725_725897

variable {x y : ℝ}

theorem sin_x_plus_y_eq_two_thirds (h1 : cos x * cos y + sin x * sin y = 1 / 2) (h2 : sin (2 * x) + sin (2 * y) = 2 / 3) : sin (x + y) = 2 / 3 := 
by
  sorry

end sin_x_plus_y_eq_two_thirds_l725_725897


namespace combined_leak_rate_empty_time_l725_725003

-- The conditions
def cistern_fill_rate_without_leaks : ℚ := 1 / 5
def leakA_rate : ℚ := 1 / 10
def leakB_rate (x : ℚ) : ℚ := x
def cistern_fill_rate_with_leaks (x : ℚ) : ℚ := 1 / 7

-- Calculation of the combined leak rate and the proof of emptying time
theorem combined_leak_rate_empty_time (x : ℚ) : 
  (1 / 5) - (1 / 10 + x) = 1 / 7 → 
  (2 * (1 / 10 + x) = 1 / 7) :=
by
  intros h
  have : 1 / 10 + x = (1 / 10) + (-3 / 70) := by sorry
  rw this
  have combined_rate: 1 / 10 + (-3 / 70) = 1 / 7 := by sorry
  rw combined_rate
  exact rfl

end combined_leak_rate_empty_time_l725_725003


namespace smallest_positive_e_l725_725061

theorem smallest_positive_e :
  ∃ e : ℤ, e > 0 ∧ (∀ (a b c d : ℤ),
  ∃ (p : Polynomial ℤ),
  p.roots = {-3, 4, 10, -1/4} ∧
  p = Polynomial.C a * (Polynomial.X + 3) *
      (Polynomial.X - 4) *
      (Polynomial.X - 10) *
      (Polynomial.C 4 * Polynomial.X + 1) ∧
  e = Int.natAbs p.coeff 0) ∧
  e = 120 :=
sorry

end smallest_positive_e_l725_725061


namespace sandy_grew_6_carrots_l725_725254

theorem sandy_grew_6_carrots (sam_grew : ℕ) (total_grew : ℕ) (h1 : sam_grew = 3) (h2 : total_grew = 9) : ∃ sandy_grew : ℕ, sandy_grew = total_grew - sam_grew ∧ sandy_grew = 6 :=
by
  sorry

end sandy_grew_6_carrots_l725_725254


namespace ratio_product_l725_725550

theorem ratio_product (A B C A' B' C' O : Type)
(h1 : colinear A' B C)
(h2 : colinear B' A C)
(h3 : colinear C' A B)
(h4 : concurrent (line_through A A') (line_through B B') (line_through C C'))
(h5 : (ratio (segment A O) (segment O A')) = 24)
(h6 : (ratio (segment A O) (segment O A') 
     + ratio (segment B O) (segment O B') 
     + ratio (segment C O) (segment O C')) = 96) :
     (ratio (segment A O) (segment O A') 
      * ratio (segment B O) (segment O B') 
      * ratio (segment C O) (segment O C')) = 24 := 
sorry

end ratio_product_l725_725550


namespace area_of_bounded_region_l725_725426

theorem area_of_bounded_region :
  let region := {p : ℝ × ℝ | (p.1 = 0 ∧ p.2 ≥ 0) ∨ (p.2 = 0 ∧ p.1 ≥ 0) ∨ (p.1 = 2) ∨ (p.2 = 2)}
  ∃ (s : set (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ s ↔ (y = x))
    ∧ is_square s 
    ∧ area s = 4 := 
sorry

end area_of_bounded_region_l725_725426


namespace solve_combination_eq_l725_725644

theorem solve_combination_eq (x : ℕ) (h : x ≥ 3) : 
  (Nat.choose x 3 + Nat.choose x 2 = 12 * (x - 1)) ↔ (x = 9) := 
by
  sorry

end solve_combination_eq_l725_725644


namespace max_value_of_f_set_of_x_for_max_value_of_f_interval_where_f_is_monotonically_decreasing_l725_725129

noncomputable def f (x : ℝ) : ℝ := real.sin (π / 6 - 2 * x) + 3 / 2

theorem max_value_of_f :
  ∃ (y : ℝ), y = 1 / 2 ∧ ∀ x : ℝ, f x ≤ y :=
sorry

theorem set_of_x_for_max_value_of_f :
  {x : ℝ | f x = 1 / 2} = {kπ - π / 6 | k : ℤ} :=
sorry

theorem interval_where_f_is_monotonically_decreasing :
  ∀ (k : ℤ), ∀ x ∈ Icc (kπ - π / 6) (kπ + π / 3), f (x + ε) ≤ f x :=
sorry

end max_value_of_f_set_of_x_for_max_value_of_f_interval_where_f_is_monotonically_decreasing_l725_725129


namespace angle_bisector_projections_l725_725884

theorem angle_bisector_projections (a b c : ℝ) (A B C E F P R Q : Type) [triangle_ABC : triangle A B C] 
(h1 : a ≥ b ∧ b ≥ c)
(h2 : BE.interior_angle_bisector ∧ CF.interior_angle_bisector)
(h3 : P ∈ triangle A E F)
(h4 : is_projection_of P R A B)
(h5 : is_projection_of P Q A C) :
  PR + PQ + RQ < b := 
by
  -- Proof goes here.
  sorry

end angle_bisector_projections_l725_725884


namespace chord_length_is_3_sqrt_2_l725_725534

-- The polar equation of the line
def polar_line (ρ θ : ℝ) : Prop := ρ * Real.sin (θ - π / 4) = sqrt 2

-- The parametric equations of the curve
def parametric_curve (t : ℝ) : ℝ × ℝ := (t, t^2)

-- Condition to check intersection point
def intersection (x y : ℝ) : Prop := x - y + 2 = 0 ∧ y = x^2

-- Prove the length of the chord |AB|
theorem chord_length_is_3_sqrt_2 : 
  let A := (-1, 1)
  let B := (2, 4)
  dist A B = 3 * sqrt 2 :=
  by
  sorry

end chord_length_is_3_sqrt_2_l725_725534


namespace exists_stable_point_l725_725572

variable (A : Finset (ℝ × ℝ))
variable (ha : ∀ (p : ℝ × ℝ), p ∈ A → p.1 > 0)
variable (x0 y0: ℝ)

def X : ℕ → ℝ × ℝ 
| 0 => (x0, y0)
| (j + 1) => if ∃ (a b : ℝ) (h : (a, b) ∈ A), a * (X j).1 + b * (X j).2 ≤ 0 
    then let ⟨a, b, h⟩ := Classical.indefiniteDescription _ (Classical.exists_true_iff_nonempty.mpr (Classical.nonempty_of_exists (exists_mem_of_ne_empty (Finset.coe_nonempty.mpr A.nonempty))))
         in ((X j).1 + a, (X j).2 + b)
    else (X j)

theorem exists_stable_point: ∃ (N : ℕ), X A ha x0 y0 (N + 1) = X A ha x0 y0 N :=
sorry

end exists_stable_point_l725_725572


namespace find_eccentricity_find_lambda_l725_725485

-- Definitions from the conditions
def ellipse (x y a b : ℝ) : Prop := (x^2)/(a^2) + (y^2)/(b^2) = 1
def line_l (x e a : ℝ) : ℝ := e * x + a
def foci (a b : ℝ) (c : ℝ) : Prop := c^2 = a^2 - b^2
def reflection (P F₁ l : Point) : Prop := -- Add definition
def isosceles (P F₁ F₂ : Point) : Prop := -- Add definition

-- (I) Statement
theorem find_eccentricity (a b c : ℝ) (e : ℝ) (M : Point) (λ : ℝ) (hb : b > 0) 
                          (ha : a > b) (hlmb : M ∈ line_l e a) 
                          (hλ : λ = 3/4) 
                          (hellipse : M ∈ ellipse a b) : 
  e = 1/2 := by sorry

-- (II) Statement
theorem find_lambda (a b c e λ : ℝ) (F₁ F₂ : Point) (P : Point)
                    (hb : b > 0) 
                    (ha : a > b) 
                    (hfoci : foci a b c)
                    (htrieq : isosceles P F₁ F₂) : 
  λ = 2/3 := by sorry

end find_eccentricity_find_lambda_l725_725485


namespace approx_ineq_l725_725413

noncomputable def approx (x : ℝ) : ℝ := 1 + 6 * (-0.002 : ℝ)

theorem approx_ineq (x : ℝ) (h : x = 0.998) : 
  abs ((x^6) - approx x) < 0.001 :=
by
  sorry

end approx_ineq_l725_725413


namespace robotics_club_students_l725_725606

theorem robotics_club_students
  (total_students : ℕ)
  (cs_students : ℕ)
  (electronics_students : ℕ)
  (both_students : ℕ)
  (h1 : total_students = 80)
  (h2 : cs_students = 50)
  (h3 : electronics_students = 35)
  (h4 : both_students = 25) :
  total_students - (cs_students - both_students + electronics_students - both_students + both_students) = 20 :=
by
  sorry

end robotics_club_students_l725_725606


namespace percentage_died_by_bombardment_l725_725542

theorem percentage_died_by_bombardment
  (initial_population : ℕ)
  (final_population : ℕ)
  (left_by_fear : ℕ → ℕ)
  (died_by_bombardment : ℕ → ℕ → ℕ)
  (condition_initial_population : initial_population = 4500)
  (condition_final_population : final_population = 3240)
  (condition_left_by_fear : ∀ P : ℕ, left_by_fear P = P - P / 5)
  (condition_died_by_bombardment : ∀ P x : ℕ, died_by_bombardment P x = P - P * x / 100)
  : ∃ x : ℕ, died_by_bombardment initial_population 10 - left_by_fear (died_by_bombardment initial_population 10) = final_population ∧ x = 10 :=
by {
  admit,
  sorry
}

end percentage_died_by_bombardment_l725_725542


namespace length_PQ_circle_line_l725_725858

def circle (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 2*y = 0

def line_parametric (t : ℝ) (x y : ℝ) : Prop := 
  x = -1 + t ∧ y = t

def polar_ray (theta : ℝ) : Prop := theta = 3 * Real.pi / 4

theorem length_PQ_circle_line :
  let P := (2 * Real.sqrt 2, 3 * Real.pi / 4)
  let Q := (Real.sqrt 2, 3 * Real.pi / 4)
  dist P Q = Real.sqrt 2 :=
sorry

end length_PQ_circle_line_l725_725858


namespace incorrect_statement_l725_725917

-- Define the conditions part of the problem
def statement_A : Prop :=
  "Some statements are accepted without being proved."

def statement_B : Prop :=
  "In some instances there is more than one correct order in proving certain propositions."

def statement_C : Prop :=
  "Every term used in a proof must have been defined previously."

def statement_D : Prop :=
  "It is not possible to arrive by correct reasoning at a true conclusion if, in the given, there is an untrue proposition."

def statement_E : Prop :=
  "Indirect proof can be used whenever there are two or more contrary propositions."

-- The theorem stating that the incorrect statement is (E)
theorem incorrect_statement :
  (statement_A ∧ statement_B ∧ statement_C ∧ statement_D) → ¬ statement_E :=
by
  sorry

end incorrect_statement_l725_725917


namespace lcm_36_100_is_900_l725_725452

def prime_factors_36 : ℕ → Prop := 
  λ n, n = 36 → (2^2 * 3^2)

def prime_factors_100 : ℕ → Prop := 
  λ n, n = 100 → (2^2 * 5^2)

def lcm_36_100 := lcm 36 100

theorem lcm_36_100_is_900 : lcm_36_100 = 900 :=
by {
  sorry,
}

end lcm_36_100_is_900_l725_725452


namespace johns_friends_count_l725_725939

-- Define the conditions
def total_cost : ℕ := 12100
def cost_per_person : ℕ := 1100

-- Define the theorem to prove the number of friends John is going with
theorem johns_friends_count (total_cost cost_per_person : ℕ) (h1 : total_cost = 12100) (h2 : cost_per_person = 1100) : (total_cost / cost_per_person) - 1 = 10 := by
  -- Providing the proof is not required, so we use sorry to skip it
  sorry

end johns_friends_count_l725_725939


namespace sequence_divisible_by_4_l725_725548

theorem sequence_divisible_by_4 
  (n : ℕ)
  (x : ℕ → ℤ)
  (h1 : ∀ i, x i = 1 ∨ x i = -1)
  (h2 : ∑ i in (finset.range (n-3)).attach, x i * x (i+1) * x (i+2) * x (i+3) + 
        x (n-2) * x (n-1) * x n % n * x (1 % n) +
        x (n-1) * x n % n * x (1 % n) * x (2 % n) +
        x n * x (1 % n) * x (2 % n) * x (3 % n) = 0) :
  4 ∣ n := 
sorry

end sequence_divisible_by_4_l725_725548


namespace f_20_value_l725_725232

noncomputable def f (n : ℕ) : ℚ := sorry

axiom f_initial : f 1 = 3 / 2
axiom f_eq : ∀ x y : ℕ, 
  f (x + y) = (1 + y / (x + 1)) * f x + (1 + x / (y + 1)) * f y + x^2 * y + x * y + x * y^2

theorem f_20_value : f 20 = 4305 := 
by {
  sorry 
}

end f_20_value_l725_725232


namespace ellipse_standard_eq_l725_725851

-- Assumptions and definitions
variables {a b c x y : ℝ}
variables {F1 F2 P M : ℝ × ℝ}
variables λ : ℝ
variables t : ℝ
variables {A B C D : ℝ × ℝ}

-- Conditions
def ellipse_eq := x^2 / a^2 + y^2 / b^2 = 1
def point_on_ellipse := P = (-1, (sqrt 2) / 2) ∧ ellipse_eq P.1 P.2
def midpoint_condition := M = ((P.1 + F2.1) / 2, (P.2 + F2.2) / 2) ∧ (M = (0, M.2))
def dot_product := λ ∈ [2/3, 1]

-- Proving the range of area S given conditions
theorem ellipse_standard_eq (h1: ellipse_eq) (h2: point_on_ellipse) (h3: midpoint_condition) (h4: dot_product) :
  (∃ a² = 2, b² = 1) ∧ (∃ S ∈ [4 * sqrt 3 / 5, 4 * sqrt 6 / 7]) :=
by sorry

end ellipse_standard_eq_l725_725851


namespace pairs_satisfying_x2_minus_y2_eq_45_l725_725893

theorem pairs_satisfying_x2_minus_y2_eq_45 :
  (∃ p : Finset (ℕ × ℕ), (∀ (x y : ℕ), ((x, y) ∈ p → x^2 - y^2 = 45) ∧ (∀ (x y : ℕ), (x, y) ∈ p → 0 < x ∧ 0 < y)) ∧ p.card = 3) :=
by
  sorry

end pairs_satisfying_x2_minus_y2_eq_45_l725_725893


namespace number_of_valid_sequences_eq_catalan_l725_725171

def is_valid_sequence (n : ℕ) (seq : ℕ → ℤ) : Prop :=
  (∀ i, seq i = 1 ∨ seq i = -1) ∧
  (finset.sum (finset.range (2 * n)) seq = 0) ∧
  (∀ k, 0 ≤ finset.sum (finset.range (k + 1)) seq)

def catalan_number (n : ℕ) : ℕ :=
  nat.factorial (2 * n) / (nat.factorial (n + 1) * nat.factorial n)

theorem number_of_valid_sequences_eq_catalan (n : ℕ) :
  { seq : ℕ → ℤ // is_valid_sequence n seq }.card = catalan_number n :=
sorry

end number_of_valid_sequences_eq_catalan_l725_725171


namespace conversion_factor_l725_725240

-- Define the given conditions
def miles_per_minute : ℝ := 6
def kilometers_per_hour : ℝ := 600

-- Prove the conversion factor from kilometers to miles
theorem conversion_factor (miles_per_minute = 6) (kilometers_per_hour = 600) :
  1 = 0.6 * (600 / 360) :=
by
  sorry

end conversion_factor_l725_725240


namespace number_of_subsets_of_set_P_l725_725894

theorem number_of_subsets_of_set_P : 
  let P : Set ℕ := {1, 2, 3}
  in { s : Set ℕ | s ⊆ P }.card = 8 := by
  sorry

end number_of_subsets_of_set_P_l725_725894


namespace sqrt_subtraction_l725_725784

theorem sqrt_subtraction :
  sqrt (121 + 81) - sqrt (49 - 36) = sqrt 202 - sqrt 13 :=
by
  sorry

end sqrt_subtraction_l725_725784


namespace chewing_gum_company_revenue_l725_725201

theorem chewing_gum_company_revenue (R : ℝ) :
  let projected_revenue := 1.25 * R
  let actual_revenue := 0.75 * R
  (actual_revenue / projected_revenue) * 100 = 60 := 
by
  sorry

end chewing_gum_company_revenue_l725_725201


namespace max_value_of_expression_l725_725955

open Real

theorem max_value_of_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x + y + z = 1) : 
  x^4 * y^2 * z ≤ 1024 / 7^7 :=
sorry

end max_value_of_expression_l725_725955


namespace at_least_6_heads_in_10_flips_l725_725341

def coin_flip : Type := bool

def is_heads (x : coin_flip) : Prop := x = tt

def num_consecutive_heads (l : list coin_flip) (n : ℕ) : Prop :=
  ∃ i : ℕ, i + n ≤ l.length ∧ l.drop i.take n = list.replicate n tt

def prob_at_least_n_consecutive_heads (l : list coin_flip) (n : ℕ) : Prop :=
  ∃ i ≤ l.length - n + 1, list.replicate n tt = l.drop (i - 1).take n

noncomputable def at_least_6_heads_in_10_flips_prob : ℚ :=
  (129:ℚ) / (1024:ℚ)

theorem at_least_6_heads_in_10_flips :
  prob_at_least_n_consecutive_heads (list.replicate 10 coin_flip) 6 = at_least_6_heads_in_10_flips_prob :=
by
  sorry

end at_least_6_heads_in_10_flips_l725_725341


namespace solution_set_inequality_l725_725840

theorem solution_set_inequality (f : ℝ → ℝ) (h1 : f 1 = 1) (h2 : ∀ x : ℝ, f' x < 1 / 2) :
  {x : ℝ | f x < x / 2 + 1 / 2} = {x : ℝ | x > 1} :=
by 
  sorry

end solution_set_inequality_l725_725840


namespace period_of_repetend_of_39_over_1428_is_24_l725_725467

theorem period_of_repetend_of_39_over_1428_is_24 :
  let f : ℚ := 39 / 1428 in
  let binary_repetend_period (x : ℚ) : ℕ :=  -- some function to compute the repetend period in binary
    24 -- the final answer obtained by computation
  in binary_repetend_period f = 24 :=
by
  sorry

end period_of_repetend_of_39_over_1428_is_24_l725_725467


namespace percentage_of_children_who_like_math_l725_725473

theorem percentage_of_children_who_like_math (x : ℕ) :
  (∀ y, y ∈ children_like_math_participate_in_mo) ∧ 
  (∀ z, z ∈  children_do_not_like_math_not_participate_in_mo) ∧
  (children_participate_in_mo = 46) →
  x = 40 := 
sorry

end percentage_of_children_who_like_math_l725_725473


namespace sequence_problem_l725_725483

theorem sequence_problem (a : ℕ → ℝ) (pos_terms : ∀ n, a n > 0)
  (h1 : a 1 = 2)
  (recurrence : ∀ n, (a n + 1) * a (n + 2) = 1)
  (h2 : a 2 = a 6) :
  a 11 + a 12 = (11 / 18) + ((Real.sqrt 5 - 1) / 2) := by
  sorry

end sequence_problem_l725_725483


namespace part1_real_roots_part2_specific_roots_l725_725877

-- Part 1: Real roots condition
theorem part1_real_roots (m : ℝ) (h : ∃ x : ℝ, x^2 + (2 * m - 1) * x + m^2 = 0) : m ≤ 1/4 :=
by sorry

-- Part 2: Specific roots condition
theorem part2_specific_roots (m : ℝ) (x1 x2 : ℝ) 
  (h1 : x1^2 + (2 * m - 1) * x1 + m^2 = 0) 
  (h2 : x2^2 + (2 * m - 1) * x2 + m^2 = 0) 
  (h3 : x1 * x2 + x1 + x2 = 4) : m = -1 :=
by sorry

end part1_real_roots_part2_specific_roots_l725_725877


namespace point_lies_on_line_through_midpoint_parallel_to_diagonal_l725_725574

variables {A B C D M : Type} [convex_quadrilateral A B C D]
variable (inside_ABCD : M ∈ interior_of_quadrilateral A B C D)
variable (area_eq : area_of_ABC M = area_of_AMD M)

theorem point_lies_on_line_through_midpoint_parallel_to_diagonal 
  (h : convex_quadrilateral A B C D) 
  (hM : M ∈ interior_of_quadrilateral A B C D)
  (h_area : area_of_ABC M = area_of_AMD M) : 
  lies_on_line_parallel_to_diagonal_through_midpoint M A B C D :=
  sorry

end point_lies_on_line_through_midpoint_parallel_to_diagonal_l725_725574


namespace stock_index_approximation_l725_725337

noncomputable def stock_index_after_days (initial_index : ℝ) (daily_increase : ℝ) (days : ℕ) : ℝ :=
  initial_index * (1 + daily_increase / 100) ^ (days - 1)

theorem stock_index_approximation :
  let initial_index := 2
  let daily_increase := 0.02
  let days := 100
  abs (stock_index_after_days initial_index daily_increase days - 2.041) < 0.001 :=
by
  sorry

end stock_index_approximation_l725_725337


namespace horses_meet_in_nine_days_l725_725693

def fine_horse_distance (m : ℕ) : ℝ :=
  103 + 13 * (m - 1)

def inferior_horse_distance (m : ℕ) : ℝ :=
  97 + (-0.5) * (m - 1)

def total_distance (m : ℕ) : ℝ :=
  (103 * m + (m * (m - 1) * 13) / 2) + (97 * m + (m * (m - 1) * (-0.5)) / 2)

theorem horses_meet_in_nine_days : total_distance 9 = 1125 * 2 :=
  sorry

end horses_meet_in_nine_days_l725_725693


namespace part1_l725_725328

theorem part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h : (a^2 + b^2 + c^2)^2 > 2 * (a^4 + b^4 + c^4)) : 
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) :=
sorry

end part1_l725_725328


namespace find_equation_of_line_l725_725104

variable {ℝ : Type*}

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def equidistant (p1 p2 : ℝ × ℝ) (line : ℝ × ℝ → Prop) : Prop :=
  ∀ (p : ℝ × ℝ), line p → dist p p1 = dist p p2

noncomputable def equation_of_line (p1 p2 : ℝ × ℝ) (line : ℝ × ℝ → Prop) : Prop :=
  let m := midpoint p1 p2 in
  (line = (λ p, p.1 = m.1)) ∨ (line = (λ p, 4 * p.1 - p.2 - 2 = 0))

theorem find_equation_of_line :
  ∀ (line : ℝ × ℝ → Prop),
    line (1,2) →
    equidistant (2,3) (0,-5) line →
    equation_of_line (2,3) (0,-5) line :=
by
  intros
  sorry

end find_equation_of_line_l725_725104


namespace incorrect_statements_l725_725821

variables (a b c : ℝ × ℝ)

def is_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k • w)

def dot_product (v w : ℝ × ℝ) : ℝ :=
  (v.1 * w.1) + (v.2 * w.2)

theorem incorrect_statements :
  ¬ ((is_parallel a b ∧ is_parallel b c) → is_parallel a c) ∧
  (dot_product a b = dot_product a c ∧ a ≠ (0, 0) → b ≠ c) ∧
  ¬ ((dot_product a b) * c = a * (dot_product b c)) :=
sorry

end incorrect_statements_l725_725821


namespace number_of_sets_l725_725676

theorem number_of_sets (M : Set ℕ) : 
  {M | {0, 1} ⊆ M ∧ M ⊂ {0, 1, 3, 5}}.card = 3 := 
sorry

end number_of_sets_l725_725676


namespace exists_four_digit_sum_21_divisible_by_14_l725_725402

-- Define a function to calculate the sum of the digits of a number
def digit_sum (n : ℕ) : ℕ :=
  (n % 10) + (n / 10 % 10) + (n / 100 % 10) + (n / 1000 % 10)

-- Define a predicate to check if a number is four-digit
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

-- Define a predicate to check if a number is divisible by 14
def divisible_by_14 (n : ℕ) : Prop :=
  n % 14 = 0

-- Our main theorem statement
theorem exists_four_digit_sum_21_divisible_by_14 : 
  ∃ (n : ℕ), is_four_digit n ∧ digit_sum n = 21 ∧ divisible_by_14 n :=
begin
  use 6384,
  split,
  { -- is_four_digit 6384
    split,
    { -- 1000 ≤ 6384
      exact nat.le_succ_of_le (nat.le_of_lt 6384),
    },
    { -- 6384 < 10000
      linarith,
    }
  },
  split,
  { -- digit_sum 6384 = 21
    norm_num,
  },
  { -- divisible_by_14 6384
    norm_num,
  }
end
end exists_four_digit_sum_21_divisible_by_14_l725_725402


namespace behavior_on_interval_1_2_l725_725947

noncomputable def f (x : ℝ) : ℝ := if (0 < x ∧ x < 1) then log (1 - x) else f x -- Placeholder for conditions

theorem behavior_on_interval_1_2 :
  (∀ x : ℝ, f (-x) = f x) →
  (∀ x : ℝ, f (x + 2) = f x) →
  (∀ x : ℝ, (0 < x ∧ x < 1) → f x = log (1 - x)) →
  (∀ x : ℝ, (1 < x ∧ x < 2) → f x > 0 ∧ ∀ y : ℝ, 1 < y ∧ y < x → f y > f x) :=
sorry

end behavior_on_interval_1_2_l725_725947


namespace find_a_100_l725_725284

def s (a : ℕ) : ℕ := a.digits.sum

def a : ℕ → ℕ 
| 0 := 2^20
| (n + 1) := s (a n)

theorem find_a_100 : a 100 = 5 := by
  sorry

end find_a_100_l725_725284


namespace sum_of_four_consecutive_integers_is_even_l725_725292

theorem sum_of_four_consecutive_integers_is_even (n : ℤ) : 2 ∣ ((n - 1) + n + (n + 1) + (n + 2)) :=
by sorry

end sum_of_four_consecutive_integers_is_even_l725_725292


namespace smallest_multiplier_to_perfect_square_l725_725374

-- Definitions for the conditions
def y := 2^3 * 3^2 * 4^3 * 5^3 * 6^6 * 7^5 * 8^6 * 9^6

-- The theorem statement itself
theorem smallest_multiplier_to_perfect_square : ∃ k : ℕ, (∀ m : ℕ, (y * m = k) → (∃ n : ℕ, (k * y) = n^2)) :=
by
  let y := 2^3 * 3^2 * 4^3 * 5^3 * 6^6 * 7^5 * 8^6 * 9^6
  let smallest_k := 70
  have h : y = 2^33 * 3^20 * 5^3 * 7^5 := by sorry
  use smallest_k
  intros m hm
  use (2^17 * 3^10 * 5 * 7)
  sorry

end smallest_multiplier_to_perfect_square_l725_725374


namespace part1_part2_max_val_part3_l725_725509

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log x - (1 / 2) * a * x ^ 2 + x

theorem part1 (h_a : a = 0) : 
    let f' := fun x => (1 / x) + 1
    let tangent_line := (2 : ℝ) * x - y - 1 = 0
    tangent_line := 2 * (1 : ℝ) - (1 : ℝ) - 1 := 0
  sorry

theorem part2_max_val (a : ℝ) (h_a_pos : 0 < a) : 
  let g := fun x => log x - (1 / 2) * a * x ^ 2 + (1 - a) * x + 1
  ∃ x, g x = 1 / (2 * a) - log a
  sorry

theorem part3 (x₁ x₂ : ℝ) (h_a : a = -2) (h_pos : ∀ (x : ℝ), 0 < x)
    (eq_condition : f x₁ (-2 : ℝ) + f x₂ (-2 : ℝ) + x₁ * x₂ = 0) : 
  x₁ + x₂ ≥ (sqrt 5 - 1) / 2 :=
begin
  sorry
end

end part1_part2_max_val_part3_l725_725509


namespace probability_three_specific_cards_l725_725293

theorem probability_three_specific_cards :
  let deck_size := 52
  let diamonds := 13
  let spades := 13
  let hearts := 13
  let p1 := diamonds / deck_size
  let p2 := spades / (deck_size - 1)
  let p3 := hearts / (deck_size - 2)
  p1 * p2 * p3 = 169 / 5100 :=
by
  sorry

end probability_three_specific_cards_l725_725293


namespace induction_inequality_proof_l725_725298

theorem induction_inequality_proof (k : ℕ) :
  (∑ i in finset.range (2*k), 1 / (k + 1 + i)) - 
  (∑ i in finset.range (2*(k+1)), 1 / (k + 2 + i)) = 
  (1 / (3*k + 1)) + (1 / (3*k + 2)) + (1 / (3*k + 3)) - (1 / (k + 1)) :=
sorry

end induction_inequality_proof_l725_725298


namespace no_statistical_properties_preserved_l725_725391

variable {x1 x2 x3 x4 : ℝ}
variable (original_data := [x1, x2, x3, x4])
variable (transformed_data := [3 * x1 + 2, 3 * x2 + 2, 3 * x3 + 2, 3 * x4 + 2])

theorem no_statistical_properties_preserved :
  let mean (data : List ℝ) := (data.sum) / (data.length)
  let stddev (data : List ℝ) := Float.sqrt ((data.map (λ x, (x - mean data)^2)).sum / (data.length - 1))
  let median (data : List ℝ) := 
    let sorted := data.qsort (· ≤ ·)
    if sorted.length % 2 = 0 
    then (sorted.get! (sorted.length / 2 - 1) + sorted.get! (sorted.length / 2)) / 2
    else sorted.get! (sorted.length / 2)
  mean transformed_data ≠ mean original_data ∧
  stddev transformed_data ≠ stddev original_data ∧
  median transformed_data ≠ median original_data := 
by
  sorry

end no_statistical_properties_preserved_l725_725391


namespace binomial_expansion_coefficient_l725_725116

theorem binomial_expansion_coefficient :
  let a := ∫ x in 0..2, x
  (binomial_coeff : ℚ) := 5.choose 2 * 2^(5 - 2) = 80 :=
sorry

end binomial_expansion_coefficient_l725_725116


namespace combined_weight_of_Leo_and_Kendra_l725_725530

-- Define the weights of Leo and Kendra
variables (L K : ℕ)
constant Leo_current_weight : L = 104
constant Leo_new_weight_gain : L' = L + 10
constant Leo_new_weight_relation : L' = K + K / 2

-- Prove the combined weight of Leo and Kendra is 180 pounds
theorem combined_weight_of_Leo_and_Kendra : L + K = 180 :=
by
  -- Variables and constants
  let L := 104
  have Leo_current_weight : L = 104 := rfl
  have Leo_new_weight : L + 10 = 114 := rfl
  have Leo_new_weight_relation : 114 = K + K / 2
  -- Solve for K to find their combined weight
  sorry

end combined_weight_of_Leo_and_Kendra_l725_725530


namespace find_common_ratio_l725_725842

theorem find_common_ratio 
  (a : ℕ → ℝ)
  (a1 : a 1 = 1) 
  (h_q : 0 < q) 
  (q_lt_half : q < 1 / 2) 
  (seq_property : ∀ k : ℕ, k > 0 → (a k) - (a (k + 1) + a (k + 2)) ∈ {n | ∃ m : ℕ, n = a m}) :
  q = Real.sqrt 2 - 1 :=
begin
  sorry
end

end find_common_ratio_l725_725842


namespace anthony_pets_final_count_l725_725038

theorem anthony_pets_final_count :
  let initial_pets := 45
  let lost_pets := (0.12 * initial_pets).natAbs
  let pets_after_loss := initial_pets - lost_pets
  let contest_rewarded_pets := 7
  let pets_after_contest := pets_after_loss + contest_rewarded_pets
  let birth_giving_pets := (pets_after_contest / 4).natAbs
  let offspring := birth_giving_pets * 2
  let pets_after_birth := pets_after_contest + offspring
  let dead_pets := (pets_after_birth / 10).natAbs
  let final_pets := pets_after_birth - dead_pets
  final_pets = 62 :=
by
  sorry

end anthony_pets_final_count_l725_725038


namespace problem_equiv_math_problem_l725_725977
-- Lean Statement for the proof problem

variable {x y z : ℝ}

theorem problem_equiv_math_problem (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (eq1 : x^2 + x * y + y^2 / 3 = 25) 
  (eq2 : y^2 / 3 + z^2 = 9) 
  (eq3 : z^2 + z * x + x^2 = 16) :
  x * y + 2 * y * z + 3 * z * x = 24 * Real.sqrt 3 :=
by
  sorry

end problem_equiv_math_problem_l725_725977


namespace squirrel_hid_acorns_l725_725913

theorem squirrel_hid_acorns (a b c : ℕ) 
  (h1 : 4 * a = 5 * b)
  (h2 : b = a - 5)
  (h3 : 2 * c = 5 * b)
  (h4 : c = b + 10) : 5 * b = 100 :=
begin
  sorry, -- The proof will go here
end

end squirrel_hid_acorns_l725_725913


namespace distance_between_intersection_points_l725_725064

noncomputable def C (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos θ, Real.sqrt 3 * Real.sin θ)

noncomputable def l (t : ℝ) : ℝ × ℝ :=
  (-2 * t + 2, 3 * t)

theorem distance_between_intersection_points :
  ∃ (A B : ℝ × ℝ), 
    (∃ θ : ℝ, C θ = A) ∧
    (∃ t : ℝ, l t = A) ∧
    (∃ θ : ℝ, C θ = B) ∧
    (∃ t : ℝ, l t = B) ∧
    dist A B = Real.sqrt 13 / 2 :=
sorry

end distance_between_intersection_points_l725_725064


namespace find_line_eq_l725_725844

-- Define the point M
structure Point where
  x : ℝ
  y : ℝ

-- Define M(2, 1)
def M : Point := { x := 2, y := 1 }

-- Define the conditions:
-- Ellipse equation: (x^2 / 16) + (y^2 / 4) = 1 for points A and B
def on_ellipse (p : Point) : Prop :=
  p.x^2 / 16 + p.y^2 / 4 = 1

-- Points A and B with given properties
structure LineIntersectsEllipse where
  A : Point
  B : Point
  l_eqn : ∀ (P : Point), P.on_line := sorry -- Placeholder for the valid equation
  M_eq_A_B_midpoint : M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

-- The required proof.
theorem find_line_eq : 
  ∀ (le : LineIntersectsEllipse), (∀ P, le.l_eqn P) ↔ (P.x + 2 * P.y - 4 = 0) := sorry

end find_line_eq_l725_725844


namespace rectangle_area_l725_725014

theorem rectangle_area (w l : ℕ) (h1 : l = w + 8) (h2 : 2 * l + 2 * w = 176) :
  l * w = 1920 :=
by
  sorry

end rectangle_area_l725_725014


namespace edge_length_of_smaller_cube_l725_725613

theorem edge_length_of_smaller_cube (a : ℝ) (h₁ : a < 1) (h₂ : ∃ (points : fin 8 → ℝ × ℝ × ℝ), ∀ i, |fst (points i)| ≤ 1 ∧ |snd (points i)| ≤ 1 ∧ |trd (points i)| ≤ 1 ∧ geometrically_form_cubical_structure points a ) :
  1 / Real.sqrt 2 ≤ a ∧ a < 1 := by
  sorry

end edge_length_of_smaller_cube_l725_725613


namespace conical_hat_height_l725_725018

-- Define the conditions
def sector_radius := 5
def sector_angle_deg : ℝ := 108
def sector_angle_rad := sector_angle_deg * Real.pi / 180

-- The problem statement: prove the height of the cone is sqrt(91) / 2 cm.
theorem conical_hat_height :
  ∀ (r : ℝ) (θ : ℝ),
    r = sector_radius →
    θ = sector_angle_rad →
      let arc_length := θ * r in
      let cone_radius := arc_length / (2 * Real.pi) in
      let slant_height := r in
      let height := Real.sqrt (slant_height ^ 2 - cone_radius ^ 2) in
      height = Real.sqrt 91 / 2 :=
by {
  intros r θ hr hθ,
  rw [hr, hθ],
  let arc_length := θ * r,
  let cone_radius := arc_length / (2 * Real.pi),
  let slant_height := r,
  let height := Real.sqrt (slant_height ^ 2 - cone_radius ^ 2),
  have : height = Real.sqrt 91 / 2, sorry,
}

end conical_hat_height_l725_725018


namespace K_time_for_distance_l725_725327

theorem K_time_for_distance (s : ℝ) (hs : s > 0) :
  (let K_time := 45 / s
   let M_speed := s - 1 / 2
   let M_time := 45 / M_speed
   K_time = M_time - 3 / 4) -> K_time = 45 / s := 
by
  sorry

end K_time_for_distance_l725_725327


namespace tram_speed_l725_725982

/-- 
Given:
1. The pedestrian's speed is 1 km per 10 minutes, which converts to 6 km/h.
2. The speed of the trams is V km/h.
3. The relative speed of oncoming trams is V + 6 km/h.
4. The relative speed of overtaking trams is V - 6 km/h.
5. The ratio of the number of oncoming trams to overtaking trams is 700/300.
Prove:
The speed of the trams V is 15 km/h.
-/
theorem tram_speed (V : ℝ) (h1 : (V + 6) / (V - 6) = 700 / 300) : V = 15 :=
by
  sorry

end tram_speed_l725_725982


namespace women_in_department_l725_725921

theorem women_in_department : 
  ∀ (total_students men women : ℕ) (men_percentage women_percentage : ℝ),
  men_percentage = 0.70 →
  women_percentage = 0.30 →
  men = 420 →
  total_students = men / men_percentage →
  women = total_students * women_percentage →
  women = 180 :=
by
  intros total_students men women men_percentage women_percentage
  intros h1 h2 h3 h4 h5
  sorry

end women_in_department_l725_725921


namespace part1_part2_l725_725235

-- Define set A
def set_A : Set ℝ := { x | -3 ≤ x ∧ x ≤ 4 }

-- Define set B depending on m
def set_B (m : ℝ) : Set ℝ := { x | 2 * m - 1 ≤ x ∧ x ≤ m + 1 }

-- Part 1: When m = -3, find A ∩ B
theorem part1 : set_B (-3) ∩ set_A = { x | -3 ≤ x ∧ x ≤ -2 } := 
sorry

-- Part 2: Find the range of m such that B ⊆ A
theorem part2 (m : ℝ) : set_B m ⊆ set_A ↔ m ≥ -1 :=
sorry

end part1_part2_l725_725235


namespace exists_triangle_BXYD_l725_725025

-- Define the points and line segments in terms of a square
variables {A B C D P Q X Y : Type} 
variables [Square A B C D] [OnSide B C P] [OnSide C D Q]
variables {BP : length B P} {QC : length C Q} (hBPQC : BP = QC)
variables [OnSegment A P X] [OnSegment A Q Y]

-- The proof problem statement
theorem exists_triangle_BXYD 
  (hSquare : is_square A B C D) 
  (hP_on_BC : is_point_on_side P B C)
  (hQ_on_CD : is_point_on_side Q C D)
  (hDistinct_P : P ≠ B ∧ P ≠ C)
  (hDistinct_Q : Q ≠ C ∧ Q ≠ D)
  (hBP_eq_CQ : length B P = length C Q)
  (hX_on_AP : is_point_on_segment X A P)
  (hY_on_AQ : is_point_on_segment Y A Q) :

  ∃ Δ : Triangle, 
    Δ.has_sides (length B X) (length X Y) (length Y D) := 
sorry

end exists_triangle_BXYD_l725_725025


namespace ab_value_l725_725725

theorem ab_value (a b : ℝ) (h₁ : a - b = 3) (h₂ : a^2 + b^2 = 33) : a * b = 18 := 
by
  sorry

end ab_value_l725_725725


namespace sequence_properties_l725_725879

-- Define the sequence according to the problem
def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 3 ∧ (∀ n : ℕ, n ≥ 2 → a n = (n * a (n - 1)) / (n - 1))

-- State the theorem to be proved
theorem sequence_properties :
  ∃ (a : ℕ → ℕ), 
    seq a ∧ a 2 = 6 ∧ a 3 = 9 ∧ (∀ n : ℕ, n ≥ 1 → a n = 3 * n) :=
by
  -- Existence quantifier and properties (sequence definition, first three terms, and general term)
  sorry

end sequence_properties_l725_725879


namespace count_diff_of_squares_in_range_l725_725056

-- Let us first define what it means for a number to be expressible as the difference of squares of two nonnegative integers.
def is_diff_of_squares (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a^2 - b^2

-- We also need to define the range of numbers we are considering: between 1 and 500, inclusive.
def in_range (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 500

-- Now, we define the set of numbers in this range which can be expressed as the difference of squares of two nonnegative integers.
def nums_in_range_expressible_as_diff_of_squares : ℕ → Prop :=
  λ n, in_range n ∧ is_diff_of_squares n

-- Finally, we want to state that the number of such numbers is 375.
theorem count_diff_of_squares_in_range :
  (finset.filter nums_in_range_expressible_as_diff_of_squares (finset.range 501)).card = 375 :=
sorry

end count_diff_of_squares_in_range_l725_725056


namespace fraction_of_area_l725_725620

noncomputable def area_of_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  let base := (C.1 - A.1).abs
  let height := B.2
  (base * height) / 2

theorem fraction_of_area {A B C X Y Z: (ℝ × ℝ)}
  (hA : A = (2, 0)) (hB : B = (8, 12)) (hC : C = (14, 0))
  (hX : X = (6, 0)) (hY : Y = (8, 4)) (hZ : Z = (10, 0)):
  (area_of_triangle X Y Z) / (area_of_triangle A B C) = 1 / 9 :=
by
  -- Skipping the proof with 'sorry'
  sorry

end fraction_of_area_l725_725620


namespace machine_A_produces_4_sprockets_per_hour_l725_725237

theorem machine_A_produces_4_sprockets_per_hour (A : ℝ) (T_Q : ℝ) 
    (hP: ∀P T_Q, 440 = A * (T_Q + 10))
    (hQ: ∀Q T_Q, 440 = 1.10 * A * T_Q):
    A = 4 :=
by
  sorry

end machine_A_produces_4_sprockets_per_hour_l725_725237


namespace functional_inequality_solution_l725_725073

-- Define the functional inequality
def functional_inequality (f : ℝ → ℝ) (a b c d : ℝ) :=
  f(a-b) * f(c-d) + f(a-d) * f(b-c) ≤ (a-c) * f(b-d)

-- State the problem: we need to prove the inequality given the conditions
theorem functional_inequality_solution (f : ℝ → ℝ) :
  (∀ a b c d : ℝ, functional_inequality f a b c d) :=
by
  let f₁ : ℝ → ℝ := λ x, 0
  let f₂ : ℝ → ℝ := λ x, x
  sorry

end functional_inequality_solution_l725_725073


namespace lcm_36_100_eq_900_l725_725449

/-- Definition for the prime factorization of 36 -/
def factorization_36 : Prop := 36 = 2^2 * 3^2

/-- Definition for the prime factorization of 100 -/
def factorization_100 : Prop := 100 = 2^2 * 5^2

/-- The least common multiple problem statement -/
theorem lcm_36_100_eq_900 (h₁ : factorization_36) (h₂ : factorization_100) : Nat.lcm 36 100 = 900 := 
by
  sorry

end lcm_36_100_eq_900_l725_725449


namespace distance_between_parallel_sides_l725_725076

-- Definitions of the given conditions
def length_side1 : ℝ := 20
def length_side2 : ℝ := 18
def area_trapezium : ℝ := 190

-- Definition of the formula for the area of a trapezium
def trapezium_area (a b h : ℝ) : ℝ :=
  (1 / 2) * (a + b) * h

-- The theorem statement to prove the distance between the parallel sides given the area
theorem distance_between_parallel_sides :
  ∃ h : ℝ, trapezium_area length_side1 length_side2 h = area_trapezium ∧ h = 10 :=
begin
  sorry
end

end distance_between_parallel_sides_l725_725076


namespace measure_of_angle_A_in_triangle_ABC_l725_725926

noncomputable def triangle_cosine_law_angle_A (a b c : ℝ) (h : a = sqrt 3) (h2 : b = 1) (h3 : c = 2) : ℝ :=
  let cos_A := (b ^ 2 + c ^ 2 - a ^ 2) / (2 * b * c) in
  if cos_A = 1/2 then π / 3 else 0

theorem measure_of_angle_A_in_triangle_ABC : triangle_cosine_law_angle_A (sqrt 3) 1 2 (by rfl) (by rfl) (by rfl) = π / 3 :=
by {
  sorry
}

end measure_of_angle_A_in_triangle_ABC_l725_725926


namespace c1_minus_c5_eq_16_l725_725956

noncomputable def f (x c1 c2 c3 c4 c5 : ℝ) : ℝ := 
  (x^2 - 10*x + c1) * (x^2 - 10*x + c2) * (x^2 - 10*x + c3) * (x^2 - 10*x + c4) * (x^2 - 10*x + c5)

def M (f : ℝ → ℝ) : set ℝ := {x | f x = 0}

theorem c1_minus_c5_eq_16 (c1 c2 c3 c4 c5 : ℝ) (h_order : c1 ≥ c2 ∧ c2 ≥ c3 ∧ c3 ≥ c4 ∧ c4 ≥ c5) 
  (h_eqs : {x | f x c1 c2 c3 c4 c5 = 0} ⊆ {x | x ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : set ℕ)}) : 
  c1 - c5 = 16 :=
sorry

end c1_minus_c5_eq_16_l725_725956


namespace length_of_train_is_360_l725_725371

-- Define the given conditions
def train_speed_km_hr : ℝ := 45
def time_to_pass_platform_s : ℝ := 39.2
def platform_length_m : ℝ := 130
def train_speed_m_s : ℝ := train_speed_km_hr * (1000 / 3600)
def total_distance_m : ℝ := train_speed_m_s * time_to_pass_platform_s

-- The theorem to prove
theorem length_of_train_is_360 :
  let L_train := total_distance_m - platform_length_m in
  L_train = 360 :=
by
  -- Definitions and conditions in the proof
  let L_train := total_distance_m - platform_length_m
  let L_train_correct := 360
  sorry

end length_of_train_is_360_l725_725371


namespace cricket_target_run_rate_cricket_wicket_partnership_score_l725_725922

noncomputable def remaining_runs_needed (initial_runs : ℕ) (target_runs : ℕ) : ℕ :=
  target_runs - initial_runs

noncomputable def required_run_rate (remaining_runs : ℕ) (remaining_overs : ℕ) : ℚ :=
  (remaining_runs : ℚ) / remaining_overs

theorem cricket_target_run_rate (initial_runs : ℕ) (target_runs : ℕ) (remaining_overs : ℕ)
  (initial_wickets : ℕ) :
  initial_runs = 32 → target_runs = 282 → remaining_overs = 40 → initial_wickets = 3 →
  required_run_rate (remaining_runs_needed initial_runs target_runs) remaining_overs = 6.25 :=
by
  sorry


theorem cricket_wicket_partnership_score (initial_runs : ℕ) (target_runs : ℕ)
  (initial_wickets : ℕ) :
  initial_runs = 32 → target_runs = 282 → initial_wickets = 3 →
  remaining_runs_needed initial_runs target_runs = 250 :=
by
  sorry

end cricket_target_run_rate_cricket_wicket_partnership_score_l725_725922


namespace scientific_notation_of_2102000_l725_725382

theorem scientific_notation_of_2102000 : ∃ (x : ℝ) (n : ℤ), 2102000 = x * 10 ^ n ∧ x = 2.102 ∧ n = 6 :=
by
  sorry

end scientific_notation_of_2102000_l725_725382


namespace fill_pool_time_l725_725936

theorem fill_pool_time 
  (pool_volume : ℕ) (num_hoses : ℕ) (flow_rate_per_hose : ℕ)
  (H_pool_volume : pool_volume = 36000)
  (H_num_hoses : num_hoses = 6)
  (H_flow_rate_per_hose : flow_rate_per_hose = 3) :
  (pool_volume : ℚ) / (num_hoses * flow_rate_per_hose * 60) = 100 / 3 :=
by sorry

end fill_pool_time_l725_725936


namespace fraction_of_areas_l725_725623

/-- Points A, B, C, X, Y, Z coordinates definitions --/
structure Point :=
(x : ℝ)
(y : ℝ)

def A := Point.mk 2 0
def B := Point.mk 8 12
def C := Point.mk 14 0

def X := Point.mk 6 0
def Y := Point.mk 8 4
def Z := Point.mk 10 0

/-- Area of a triangle given base and height --/
def area_triangle (base height : ℝ) : ℝ :=
  (base * height) / 2

/-- Area of triangle ABC --/
def Area_ABC := area_triangle (C.x - A.x) B.y

/-- Area of triangle XYZ --/
def Area_XYZ := area_triangle (Z.x - X.x) Y.y

theorem fraction_of_areas : Area_XYZ / Area_ABC = 1 / 9 := by
  sorry

end fraction_of_areas_l725_725623


namespace larger_root_eq_5_over_8_l725_725862

noncomputable def find_larger_root : ℝ := 
    let x := ((5:ℝ) / 8)
    let y := ((23:ℝ) / 48)
    if x > y then x else y

theorem larger_root_eq_5_over_8 (x : ℝ) (y : ℝ) : 
  (x - ((5:ℝ) / 8)) * (x - ((5:ℝ) / 8)) + (x - ((5:ℝ) / 8)) * (x - ((1:ℝ) / 3)) = 0 → 
  find_larger_root = ((5:ℝ) / 8) :=
by
  intro h
  -- proof goes here
  sorry

end larger_root_eq_5_over_8_l725_725862


namespace find_abc_exists_l725_725930

theorem find_abc_exists:
  ∃ (a b c : ℚ), ∀ n : ℕ, n > 0 → (∑ k in Finset.range n, k * (n^2 - k^2)) = a * n^4 + b * n^2 + c := by
  sorry

end find_abc_exists_l725_725930


namespace third_divisor_l725_725308

/-- 
Given that the new number after subtracting 7 from 3,381 leaves a remainder of 8 when divided by 9 
and 11, prove that the third divisor that also leaves a remainder of 8 is 17.
-/
theorem third_divisor (x : ℕ) (h1 : x = 3381 - 7)
                      (h2 : x % 9 = 8)
                      (h3 : x % 11 = 8) :
  ∃ (d : ℕ), d = 17 ∧ x % d = 8 := sorry

end third_divisor_l725_725308


namespace sqrt_logarithm_expression_l725_725306

theorem sqrt_logarithm_expression :
  sqrt (log 2 8 + log 4 8 + log 2 4) = sqrt 6.5 := by
  have h1 : log 2 8 = 3 := sorry
  have h2 : log 2 4 = 2 := sorry
  have h3 : log 4 8 = log 2 8 / log 2 4 := sorry
  rw [h1, h2, h3]
  sorry

end sqrt_logarithm_expression_l725_725306


namespace probability_at_least_6_heads_in_10_flips_l725_725343

theorem probability_at_least_6_heads_in_10_flips : 
  let total_outcomes := 1024 in 
  let favorable_outcomes := 15 in 
  (favorable_outcomes / total_outcomes : ℚ) = 15 / 1024 :=
by
  sorry

end probability_at_least_6_heads_in_10_flips_l725_725343


namespace parabola_focus_directrix_distance_l725_725659

theorem parabola_focus_directrix_distance :
  (distance_focus_directrix_parabola (λ x y, y^2 = x) = 0.5) :=
begin
  sorry
end

end parabola_focus_directrix_distance_l725_725659


namespace derek_history_test_l725_725800

theorem derek_history_test :
  let ancient_questions := 20
  let medieval_questions := 25
  let modern_questions := 35
  let total_questions := ancient_questions + medieval_questions + modern_questions

  let derek_ancient_correct := 0.60 * ancient_questions
  let derek_medieval_correct := 0.56 * medieval_questions
  let derek_modern_correct := 0.70 * modern_questions

  let derek_total_correct := derek_ancient_correct + derek_medieval_correct + derek_modern_correct

  let passing_score := 0.65 * total_questions
  (derek_total_correct < passing_score) →
  passing_score - derek_total_correct = 2
  := by
  sorry

end derek_history_test_l725_725800


namespace lcm_36_100_eq_900_l725_725447

/-- Definition for the prime factorization of 36 -/
def factorization_36 : Prop := 36 = 2^2 * 3^2

/-- Definition for the prime factorization of 100 -/
def factorization_100 : Prop := 100 = 2^2 * 5^2

/-- The least common multiple problem statement -/
theorem lcm_36_100_eq_900 (h₁ : factorization_36) (h₂ : factorization_100) : Nat.lcm 36 100 = 900 := 
by
  sorry

end lcm_36_100_eq_900_l725_725447


namespace table_conditions_2015_l725_725188

theorem table_conditions_2015 (n : ℕ) (table : Fin 2015 → Fin n → ℕ) :
  (∀ i : Fin 2015, ∃ j : Fin n, 0 < table i j) ∧   -- Each row has a positive number
  (∀ j : Fin n, ∃ i : Fin 2015, 0 < table i j) ∧   -- Each column has a positive number
  (∀ i : Fin 2015, ∀ j : Fin n, 
     0 < table i j → 
     (∑ k : Fin n, table i k) = (∑ k : Fin 2015, table k j))   -- Positive cell condition
  → n = 2015 := 
by
  sorry

end table_conditions_2015_l725_725188


namespace intersect_x_axis_unique_l725_725665

theorem intersect_x_axis_unique (a : ℝ) : (∀ x, (ax^2 + (3 - a) * x + 1) = 0 → x = 0) ↔ (a = 0 ∨ a = 1 ∨ a = 9) := by
  sorry

end intersect_x_axis_unique_l725_725665


namespace flag_arrangements_modulo_l725_725695

open Nat

theorem flag_arrangements_modulo :
  let M := 12.choose(9) * (13 - 1) in
  M % 1000 = 295 :=
by
  let M := nat.choose 12 9 * 13
  have : M = 9295 := by sorry
  rw [this]
  exact rfl

end flag_arrangements_modulo_l725_725695


namespace valid_probability_is_two_over_fifteen_l725_725861

def total_permutations : ℕ := 5!
def valid_permutations : ℕ := 8

def valid_probability : ℚ := valid_permutations / total_permutations

theorem valid_probability_is_two_over_fifteen :
  valid_probability = 2 / 15 := 
by sorry

end valid_probability_is_two_over_fifteen_l725_725861


namespace fraction_of_area_l725_725622

noncomputable def area_of_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  let base := (C.1 - A.1).abs
  let height := B.2
  (base * height) / 2

theorem fraction_of_area {A B C X Y Z: (ℝ × ℝ)}
  (hA : A = (2, 0)) (hB : B = (8, 12)) (hC : C = (14, 0))
  (hX : X = (6, 0)) (hY : Y = (8, 4)) (hZ : Z = (10, 0)):
  (area_of_triangle X Y Z) / (area_of_triangle A B C) = 1 / 9 :=
by
  -- Skipping the proof with 'sorry'
  sorry

end fraction_of_area_l725_725622


namespace needed_correct_to_pass_l725_725376

def total_questions : Nat := 120
def genetics_questions : Nat := 20
def ecology_questions : Nat := 50
def evolution_questions : Nat := 50

def correct_genetics : Nat := (60 * genetics_questions) / 100
def correct_ecology : Nat := (50 * ecology_questions) / 100
def correct_evolution : Nat := (70 * evolution_questions) / 100
def total_correct : Nat := correct_genetics + correct_ecology + correct_evolution

def passing_rate : Nat := 65
def passing_score : Nat := (passing_rate * total_questions) / 100

theorem needed_correct_to_pass : (passing_score - total_correct) = 6 := 
by
  sorry

end needed_correct_to_pass_l725_725376


namespace pool_ratio_l725_725247

theorem pool_ratio 
  (total_pools : ℕ)
  (ark_athletic_wear_pools : ℕ)
  (total_pools_eq : total_pools = 800)
  (ark_athletic_wear_pools_eq : ark_athletic_wear_pools = 200)
  : ((total_pools - ark_athletic_wear_pools) / ark_athletic_wear_pools) = 3 :=
by
  sorry

end pool_ratio_l725_725247


namespace ordered_pairs_29_l725_725081

noncomputable def number_of_ordered_pairs : ℕ :=
  (λ (a b : ℂ), a^3 * b^5 = 1 ∧ a^7 * b^2 = 1).count

theorem ordered_pairs_29 : number_of_ordered_pairs = 29 := by
  sorry

end ordered_pairs_29_l725_725081


namespace asha_remaining_money_l725_725043

-- Define the borrowed amounts, gift, and savings
def borrowed_from_brother : ℤ := 20
def borrowed_from_father : ℤ := 40
def borrowed_from_mother : ℤ := 30
def gift_from_granny : ℤ := 70
def savings : ℤ := 100

-- Total amount of money Asha has
def total_amount : ℤ := borrowed_from_brother + borrowed_from_father + borrowed_from_mother + gift_from_granny + savings

-- Amount spent by Asha
def amount_spent : ℤ := (3 * total_amount) / 4

-- Amount of money Asha remains with
def amount_left : ℤ := total_amount - amount_spent

-- The proof statement
theorem asha_remaining_money : amount_left = 65 := by
  sorry

end asha_remaining_money_l725_725043


namespace smallest_positive_period_maximum_value_and_points_intervals_of_monotonic_increase_axis_of_symmetry_l725_725138

noncomputable def y (x : ℝ) : ℝ :=
  4 * (Real.cos x) ^ 2 - 4 * Real.sqrt 3 * Real.sin x * Real.cos x - 1

theorem smallest_positive_period (x : ℝ) : (∃ T, T = Real.pi ∧ ∀ y, y (x + T) = y x) :=
  sorry

theorem maximum_value_and_points (x : ℝ) : 
  let maxval := 6
  in y x ≤ maxval ∧ (∀ k : ℤ, 2 * x - Real.pi / 6 = -Real.pi / 2 + 2 * k * Real.pi) ->
     (y x = maxval ∧ ∃ k, x = -Real.pi / 6 + k * Real.pi) :=
  sorry

theorem intervals_of_monotonic_increase (x : ℝ) :
  (∃ k : ℤ, Real.pi / 3 + k * Real.pi ≤ x ∧ x ≤ 5 * Real.pi / 6 + k * Real.pi ∧
            (∀ y, ∃ T, y x < y T → (Real.pi / 3 + k * Real.pi ≤ T ∧
            T ≤ 5 * Real.pi / 6 + k * Real.pi))) :=
  sorry

theorem axis_of_symmetry (x : ℝ) : (∃ k : ℤ, x = Real.pi / 3 + k * Real.pi / 2) :=
  sorry

end smallest_positive_period_maximum_value_and_points_intervals_of_monotonic_increase_axis_of_symmetry_l725_725138


namespace correct_statement_is_D_l725_725775

def correct_instrumentation_for_heat_of_neutralization_exp (beaker1 beaker2 : String) 
    (measuring_cylinders : Nat) (thermometer : Bool) (stirring_rod : Bool) : Prop :=
  beaker1 ≠ beaker2 ∧ measuring_cylinders >= 2 ∧ thermometer = true ∧ stirring_rod = true

def required_temperature_measurements (sets : Nat) (measurements_per_set : Nat) : Nat :=
  sets * measurements_per_set

def correct_temp_measurement_requirement : Prop := required_temperature_measurements 2 3 = 6

def impact_of_naoh_excess : Prop := ∀ (naoh_excess hcl : Nat), naoh_excess > hcl → 
  (heat_of_neutralization naoh_excess - heat_of_neutralization hcl = 0)

def efficiency_of_thermos_usage : Prop := ∀ (thermos beaker : String) (insulation_effect : Nat), 
  thermos = "thermos" ∧ beaker = "beaker" → insulation_effect thermos > insulation_effect beaker

theorem correct_statement_is_D : ∀ (A B C D : Prop),
  (A = correct_instrumentation_for_heat_of_neutralization_exp "beaker1" "beaker1" 2 true true) →
  (B = correct_temp_measurement_requirement) →
  (C = impact_of_naoh_excess) →
  (D = efficiency_of_thermos_usage) →
  ¬A ∧ ¬B ∧ ¬C ∧ D := by
  intros
  sorry

end correct_statement_is_D_l725_725775


namespace smallest_positive_period_of_f_minimum_value_of_f_and_set_of_x_values_l725_725507

noncomputable def f (x : ℝ) : ℝ := 2 * sin x * cos x + cos x ^ 2 - sin x ^ 2

theorem smallest_positive_period_of_f : ∃ p > 0, ∀ x, f (x + p) = f x ∧ p = π := sorry

theorem minimum_value_of_f_and_set_of_x_values : 
  ∃ (min_val : ℝ) (x_set : Set ℝ), 
    (∀ x ∈ Icc 0 (π / 2), f x ≥ min_val) ∧ 
    min_val = -√2 ∧ 
    x_set = {π / 2} ∧ 
    ∀ x, x ∈ x_set → f x = min_val := sorry

end smallest_positive_period_of_f_minimum_value_of_f_and_set_of_x_values_l725_725507


namespace candice_spending_l725_725241

variable (total_budget : ℕ) (remaining_money : ℕ) (mildred_spending : ℕ)

theorem candice_spending 
  (h1 : total_budget = 100)
  (h2 : remaining_money = 40)
  (h3 : mildred_spending = 25) :
  (total_budget - remaining_money) - mildred_spending = 35 := 
by
  sorry

end candice_spending_l725_725241


namespace measure_of_angle_l725_725998

variable (x : ℝ)

theorem measure_of_angle :
  let complement := 90 - x in
  complement = 3 * x - 7 → x = 24.25 :=
by
  assume complement_eq : 90 - x = 3 * x - 7
  sorry

end measure_of_angle_l725_725998


namespace area_ratio_of_similar_triangles_l725_725180

-- To state the conditions of similarity and the problem
variables (A B C D E F : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]

-- Assume similarity ratio of triangles ABC and DEF
variables {similarity_ratio : ℝ} (similarity_ratio_eq : similarity_ratio = 1 / 3)

-- Define areas of the triangles
noncomputable def area_ABC (A B C : Type) : ℝ := sorry
noncomputable def area_DEF (D E F : Type) : ℝ := sorry

-- The theorem we need to prove
theorem area_ratio_of_similar_triangles (A B C D E F : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]
  (similarity_ratio : ℝ) (similarity_ratio_eq : similarity_ratio = 1 / 3) :
  let ratio_of_areas := area_ABC A B C / area_DEF D E F in
  ratio_of_areas = 1 / 9 :=
sorry

end area_ratio_of_similar_triangles_l725_725180


namespace table_conditions_2015_l725_725186

theorem table_conditions_2015 (n : ℕ) (table : Fin 2015 → Fin n → ℕ) :
  (∀ i : Fin 2015, ∃ j : Fin n, 0 < table i j) ∧   -- Each row has a positive number
  (∀ j : Fin n, ∃ i : Fin 2015, 0 < table i j) ∧   -- Each column has a positive number
  (∀ i : Fin 2015, ∀ j : Fin n, 
     0 < table i j → 
     (∑ k : Fin n, table i k) = (∑ k : Fin 2015, table k j))   -- Positive cell condition
  → n = 2015 := 
by
  sorry

end table_conditions_2015_l725_725186


namespace maximal_value_S_l725_725461

theorem maximal_value_S (a b c d : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) (h_sum : a + b + c + d = 100) :
  (sqrt3 (a / (b + 7))) + (sqrt3 (b / (c + 7))) + (sqrt3 (c / (d + 7))) + (sqrt3 (d / (a + 7))) ≤ (8 / sqrt3 7) :=
sorry

end maximal_value_S_l725_725461


namespace limit_series_sum_l725_725873

def series_limit_sum : ℝ :=
  ∑' n, (n : ℝ) / 10^n

theorem limit_series_sum :
  series_limit_sum = 10 / 81 :=
by
  -- Proof goes here
  sorry

end limit_series_sum_l725_725873


namespace tomatoes_left_l725_725690

theorem tomatoes_left (initial_tomatoes : ℕ) (birds : ℕ) (fraction_eaten : ℚ) :
  initial_tomatoes = 21 ∧ birds = 2 ∧ fraction_eaten = 1/3 ->
  initial_tomatoes - initial_tomatoes * fraction_eaten = 14 :=
by
  intros h
  cases h with h1 h_rest
  cases h_rest with h2 h3
  rw [h1, h2, h3]
  norm_num
  rw [Nat.cast_sub 21 7 _, Nat.cast_mul, Nat.cast_div]; norm_num -- Converting to rational arithmetic and proving directly
  exact le_of_lt_nat (div_lt_self (zero_lt_nat 21) (zero_lt_nat 3))

end tomatoes_left_l725_725690


namespace fraction_of_areas_l725_725625

/-- Points A, B, C, X, Y, Z coordinates definitions --/
structure Point :=
(x : ℝ)
(y : ℝ)

def A := Point.mk 2 0
def B := Point.mk 8 12
def C := Point.mk 14 0

def X := Point.mk 6 0
def Y := Point.mk 8 4
def Z := Point.mk 10 0

/-- Area of a triangle given base and height --/
def area_triangle (base height : ℝ) : ℝ :=
  (base * height) / 2

/-- Area of triangle ABC --/
def Area_ABC := area_triangle (C.x - A.x) B.y

/-- Area of triangle XYZ --/
def Area_XYZ := area_triangle (Z.x - X.x) Y.y

theorem fraction_of_areas : Area_XYZ / Area_ABC = 1 / 9 := by
  sorry

end fraction_of_areas_l725_725625


namespace table_size_condition_l725_725189

-- Define the problem in Lean 4
theorem table_size_condition (n : ℕ) (a : ℕ → ℕ → ℝ) (_ : ∀ i j, 0 ≤ a i j) 
  (H1 : ∀ i, ∃ j, 0 < a i j) (H2 : ∀ j, ∃ i, 0 < a i j)
  (H3 : ∀ i j, 0 < a i j → (∑ k, a i k) = (∑ k, a k j)) : n = 2015 :=
sorry

end table_size_condition_l725_725189


namespace length_of_AB_l725_725272

-- Define the parametric equations for the curve
def curve_parametric (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, 1 + Real.sin θ)

-- Define the line equation
def on_line (p : ℝ × ℝ) : Prop :=
  p.1 + p.2 - 1 = 0

-- Main theorem to be proved
theorem length_of_AB : ∀ θ1 θ2 : ℝ, on_line (curve_parametric θ1) ∧ on_line (curve_parametric θ2) ∧ θ1 ≠ θ2 -> 
  (Real.distance (curve_parametric θ1) (curve_parametric θ2) = 2) :=
by
  -- Sorry is used to skip the proof
  sorry

end length_of_AB_l725_725272


namespace sets_inequality_l725_725899

variable {X : Type*} (A B : Set X)

theorem sets_inequality :
  (|A| * |B| ≤ |A ∪ B| * |A ∩ B|) :=
sorry

end sets_inequality_l725_725899


namespace ratio_of_volumes_and_surface_areas_l725_725017

def V_AC (d1 d2 : ℝ) : ℝ := (2 / 3) * π * d1^2 * d2
def V_BD (d1 d2 : ℝ) : ℝ := (2 / 3) * π * d2^2 * d1
def S_AC (a d1 : ℝ) : ℝ := 2 * π * a * d1
def S_BD (a d2 : ℝ) : ℝ := 2 * π * a * d2

theorem ratio_of_volumes_and_surface_areas (a d1 d2 : ℝ) (h_d2_ne_0 : d2 ≠ 0) :
    (V_AC d1 d2 / V_BD d1 d2) = (S_AC a d1 / S_BD a d2) :=
by
  sorry

end ratio_of_volumes_and_surface_areas_l725_725017


namespace train_crosses_pole_in_9_seconds_l725_725767

-- Define the speed of the train in kilometers per hour
def train_speed_km_per_hr : ℝ := 80

-- Define the length of the train in meters
def train_length_m : ℝ := 200

-- Convert speed to meters per second
def km_per_hr_to_m_per_s (speed : ℝ) : ℝ := speed * 1000 / 3600

-- Compute the time it takes for the train to cross the pole
def time_to_cross_pole (length : ℝ) (speed_in_m_per_s : ℝ) : ℝ := length / speed_in_m_per_s

-- Use the converted speed in m/s to calculate the actual time
def train_speed_m_per_s : ℝ := km_per_hr_to_m_per_s train_speed_km_per_hr

theorem train_crosses_pole_in_9_seconds : time_to_cross_pole train_length_m train_speed_m_per_s = 9 := by
  sorry

end train_crosses_pole_in_9_seconds_l725_725767


namespace algebraic_expression_value_l725_725525

theorem algebraic_expression_value
  (a b x y : ℤ)
  (h1 : x = a)
  (h2 : y = b)
  (h3 : x - 2 * y = 7) :
  -a + 2 * b + 1 = -6 :=
by
  -- the proof steps are omitted as instructed
  sorry

end algebraic_expression_value_l725_725525


namespace area_of_region_bounded_by_lines_l725_725433

theorem area_of_region_bounded_by_lines : 
  let x1 := λ θ : ℝ, 2 in
  let y1 := λ θ : ℝ, 2 in
  x1 (0:ℝ) * y1 (0:ℝ) = 4 :=
by {
  sorry,
}

end area_of_region_bounded_by_lines_l725_725433


namespace evaluate_expression_l725_725069

theorem evaluate_expression : (10^9) / ((2 * 10^6) * 3) = 500 / 3 :=
by sorry

end evaluate_expression_l725_725069


namespace min_flight_routes_l725_725151

-- Defining a problem of connecting cities with flight routes such that 
-- every city can be reached from any other city with no more than two layovers.
theorem min_flight_routes (n : ℕ) (h : n = 50) : ∃ (r : ℕ), (r = 49) ∧
  (∀ (c1 c2 : ℕ), c1 ≠ c2 → c1 < n → c2 < n → ∃ (a b : ℕ),
    a < n ∧ b < n ∧ (a = c1 ∨ a = c2) ∧ (b = c1 ∨ b = c2) ∧
    ((c1 = a ∧ c2 = b) ∨ (c1 = a ∧ b = c2) ∨ (a = c2 ∧ b = c1))) :=
by {
  sorry
}

end min_flight_routes_l725_725151


namespace angle_between_vectors_is_pi_over_6_l725_725114

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

def is_angle_pi_over_6 (a b : V) : Prop :=
  real.angle_between a b = real.pi / 6

theorem angle_between_vectors_is_pi_over_6
  (h1 : inner a (a + b) b = 7)
  (h2 : ∥a∥ = real.sqrt 3)
  (h3 : ∥b∥ = 2) :
  is_angle_pi_over_6 a b :=
sorry

end angle_between_vectors_is_pi_over_6_l725_725114


namespace simplest_form_sqrt_sum_l725_725387

theorem simplest_form_sqrt_sum :
  (sqrt 2 + sqrt (2 + 4) + sqrt (2 + 4 + 6) + sqrt (2 + 4 + 6 + 8)) =
  (sqrt 2 + sqrt 6 + 2 * sqrt 3 + 2 * sqrt 5) :=
by
  sorry

end simplest_form_sqrt_sum_l725_725387


namespace standard_equation_hyperbola_l725_725103

-- Define necessary conditions
def condition_hyperbola (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) :=
  ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1)

def condition_asymptote (a b : ℝ) :=
  b / a = Real.sqrt 3

def condition_focus_hyperbola_parabola (a b : ℝ) :=
  (a^2 + b^2).sqrt = 4

-- Define the proof problem
theorem standard_equation_hyperbola (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0)
  (h_asymptote : condition_asymptote a b)
  (h_focus : condition_focus_hyperbola_parabola a b) :
  ∀ x y : ℝ, (x^2 / 4 - y^2 / 12 = 1) :=
sorry

end standard_equation_hyperbola_l725_725103


namespace number_of_six_digit_palindromes_l725_725156

def is_six_digit_palindrome (n : ℕ) : Prop :=
  let d1 := n / 100000 % 10
  let d2 := n / 10000 % 10
  let d3 := n / 1000 % 10
  let d4 := n / 100 % 10
  let d5 := n / 10 % 10
  let d6 := n % 10
  n >= 100000 ∧ n < 1000000 ∧ d1 > 0 ∧ d1 = d6 ∧ d2 = d5 ∧ d3 = d4

theorem number_of_six_digit_palindromes : 
  {n : ℕ | is_six_digit_palindrome n}.card = 900 := 
sorry

end number_of_six_digit_palindromes_l725_725156


namespace projection_matrix_correct_l725_725812

noncomputable def vec_proj_mat (v : ℝ^3) : ℝ^3 :=
  let a := ![1, 1, 2] in
  let factor := (v.dot a) / (a.dot a) in
  factor • a

theorem projection_matrix_correct (v : ℝ^3) :
  let P : matrix (fin 3) (fin 3) ℝ := ![
    ![1/6, 1/6, 1/3],
    ![1/6, 1/6, 1/3],
    ![1/3, 1/3, 2/3]
  ] in
  P.mul_vec v = vec_proj_mat v :=
by
  sorry

end projection_matrix_correct_l725_725812


namespace find_point_A_equidistant_l725_725077

noncomputable def is_equidistant (A B C : ℝ × ℝ × ℝ) :=
  let dist (P Q : ℝ × ℝ × ℝ) := real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2 + (P.3 - Q.3) ^ 2)
  dist A B = dist A C

theorem find_point_A_equidistant :
  ∃ z : ℝ, is_equidistant (0, 0, z) (5, 1, 0) (0, 2, 3) ∧ z = -(13 / 6) := 
by
  use -13 / 6
  -- Proof would go here
  sorry

end find_point_A_equidistant_l725_725077


namespace a_n_formula_l725_725846

variable {a : ℕ+ → ℝ}  -- Defining a_n as a sequence from positive natural numbers to real numbers
variable {S : ℕ+ → ℝ}  -- Defining S_n as a sequence from positive natural numbers to real numbers

-- Given conditions
axiom S_def (n : ℕ+) : S n = a n / 2 + 1 / a n - 1
axiom a_pos (n : ℕ+) : a n > 0

-- Conjecture to be proved
theorem a_n_formula (n : ℕ+) : a n = Real.sqrt (2 * n + 1) - Real.sqrt (2 * n - 1) := 
sorry -- proof to be done

end a_n_formula_l725_725846


namespace option_B_correct_l725_725311

theorem option_B_correct (a b : ℝ) (h : a < b) : a^3 < b^3 := sorry

end option_B_correct_l725_725311


namespace analytic_expression_and_domain_of_g_max_min_values_of_g_l725_725495

-- Define the function f(x) = 2^x
def f (x : ℝ) : ℝ := 2^x

-- Define the function g(x) = f(2x) - f(x + 2)
def g (x : ℝ) : ℝ := f(2 * x) - f(x + 2)

-- Define the domain of f(x) = 2^x
noncomputable def domain_f : set ℝ := {x | 0 ≤ x ∧ x ≤ 3}

-- Define the domain of g(x)
noncomputable def domain_g : set ℝ := {x | 0 ≤ x ∧ x ≤ 1}

-- Theorem 1: Analytic expression and domain of g(x)
theorem analytic_expression_and_domain_of_g : 
  (∀ (x : ℝ), x ∈ domain_g → g(x) = 2^(2*x) - 2^(x + 2)) ∧ (domain_g = {x | 0 ≤ x ∧ x ≤ 1}) :=
by 
  sorry

-- Theorem 2: Maximum and minimum values of g(x) when x ∈ [0,1]
theorem max_min_values_of_g :
  (∀ (x : ℝ), x ∈ {x | 0 ≤ x ∧ x ≤ 1} → (g x ≤ -3 ∧ g x ≥ -4)) ∧ 
  (∃ (x_max x_min : ℝ), x_max = -3 ∧ x_min = -4) :=
by 
  sorry

end analytic_expression_and_domain_of_g_max_min_values_of_g_l725_725495


namespace sum_of_integers_70_to_85_l725_725304

theorem sum_of_integers_70_to_85 :
  let range_start := 70
  let range_end := 85
  let n := range_end - range_start + 1
  let sum := (range_start + range_end) * n / 2
  sum = 1240 :=
by
  let range_start := 70
  let range_end := 85
  let n := range_end - range_start + 1
  let sum := (range_start + range_end) * n / 2
  sorry

end sum_of_integers_70_to_85_l725_725304


namespace lcm_36_100_l725_725458

theorem lcm_36_100 : Nat.lcm 36 100 = 900 :=
by
  sorry

end lcm_36_100_l725_725458


namespace range_of_m_l725_725471

theorem range_of_m (m : ℝ) : 
  (∃ x : ℕ, x.val > m + 3 ∧ 5 * x - 2 < 4 * x + 1 ∧ 1 ≤ x ∧ x ≤ 2
        ∧ (∀ x1 x2 : ℕ, x1 > m + 3 ∧ x1 ≤ 2 → x2 > m + 3 ∧ x2 ≤ 2 → x1 = x2 → False))
  ↔ (-5 ≤ m ∧ m < -4) := 
begin
  sorry
end

end range_of_m_l725_725471


namespace simplify_fraction_l725_725257

theorem simplify_fraction :
  (45 * (14 / 25) * (1 / 18) * (5 / 11) : ℚ) = 7 / 11 := 
by sorry

end simplify_fraction_l725_725257


namespace carlos_improved_lap_time_l725_725048

-- Define the initial condition using a function to denote time per lap initially
def initial_lap_time : ℕ := (45 * 60) / 15

-- Define the later condition using a function to denote time per lap later on
def current_lap_time : ℕ := (42 * 60) / 18

-- Define the proof that calculates the improvement in seconds
theorem carlos_improved_lap_time : initial_lap_time - current_lap_time = 40 := by
  sorry

end carlos_improved_lap_time_l725_725048


namespace problem_1_problem_2_l725_725586

variables {R : Type*} [CommRing R]

noncomputable def A (x y z : R) : R := sorry
noncomputable def B (x y z : R) : R := sorry
noncomputable def C (x y z : R) : R := sorry
noncomputable def f (x y z : R) : R := sorry

theorem problem_1 
  (h1: ∀ w : R, f w w w = 0) :
  (∃ A B C : R → R → R → R,
    (∀ w, A w w w + B w w w + C w w w = 0) ∧
    (∀ x y z, f x y z = A x y z * (x - y) + B x y z * (y - z) + C x y z * (z - x))) :=
begin
  sorry
end

theorem problem_2
  (A B C : R → R → R → R)
  (h2: ∀ w : R, f w w w = 0)
  (h3: ∀ w, A w w w + B w w w + C w w w = 0)
  (h4: ∀ x y z, f x y z = A x y z * (x - y) + B x y z * (y - z) + C x y z * (z - x)) :
  ¬ (∀ A' B' C' : R → R → R → R,
    (∀ w, A' w w w + B' w w w + C' w w w = 0) ∧
    (∀ x y z, f x y z = A' x y z * (x - y) + B' x y z * (y - z) + C' x y z * (z - x)) →
    (A, B, C) = (A', B', C')) :=
begin
  sorry
end

end problem_1_problem_2_l725_725586


namespace complete_square_solution_l725_725771

theorem complete_square_solution (a b c : ℤ) (h1 : a^2 = 25) (h2 : 10 * b = 30) (h3 : (a * x + b)^2 = 25 * x^2 + 30 * x + c) :
  a + b + c = -58 :=
by
  sorry

end complete_square_solution_l725_725771


namespace organization_members_count_l725_725365

noncomputable def numMembers (n_committees : ℕ) : ℕ :=
  (n_committees * (n_committees - 1)) / 2

theorem organization_members_count : 
  ∃ (num_members : ℕ), num_members = numMembers 5 :=
by {
  use 10,
  have h : numMembers 5 = 10 := by
    simp [numMembers],
    norm_num,
  exact h,
}

end organization_members_count_l725_725365


namespace total_daisies_l725_725557

-- Define the initial conditions
def white_daisies : Nat := 6
def pink_daisies : Nat := 9 * white_daisies
def red_daisies : Nat := 4 * pink_daisies - 3

-- The main theorem stating that the total number of daisies is 273
theorem total_daisies : white_daisies + pink_daisies + red_daisies = 273 := by
  -- The proof is left as an exercise
  sorry

end total_daisies_l725_725557


namespace triangle_inequality_point_inside_l725_725324

theorem triangle_inequality_point_inside {A B C M : Point} 
(hM_inside : M ∈ triangle ABC) : 
MB + MC < AB + AC := 
sorry

end triangle_inequality_point_inside_l725_725324


namespace exists_irrational_l725_725584

noncomputable def a_sequence (a1 : ℝ) : ℕ → ℝ
| 0     := a1
| (n+1) := real.sqrt (a_sequence n + 1)

theorem exists_irrational (a1 : ℝ) (h_pos : a1 > 0) :
  ∃ n, irrational (a_sequence a1 n) :=
by sorry

end exists_irrational_l725_725584


namespace value_of_a_l725_725278

noncomputable def f (a : ℝ) : ℝ → ℝ := λ x, x^3 + a * x + 3 * x - 9

theorem value_of_a
  (a : ℝ)
  (h_extreme : ∃ c, f a c = 0 ∧ deriv (f a) c = 0)
  (h_c : c = -3) :
  a = 5 :=
sorry

end value_of_a_l725_725278


namespace number_of_six_digit_palindromes_l725_725169

def is_six_digit_palindrome (n : ℕ) : Prop := 
  100000 ≤ n ∧ n ≤ 999999 ∧ (∀ a b c : ℕ, 
    n = 100000 * a + 10000 * b + 1000 * c + 100 * c + 10 * b + a → a ≠ 0)

theorem number_of_six_digit_palindromes : 
  ∃ (count : ℕ), (count = 900 ∧ 
  ∀ n : ℕ, is_six_digit_palindrome n → true) 
:= 
by 
  use 900 
  sorry

end number_of_six_digit_palindromes_l725_725169


namespace fraction_of_largest_circle_shaded_l725_725760

noncomputable def square_inscribed_circle (s : ℝ) := 
  circle (s / 2)

noncomputable def rectangle_inscribed_square (k l : ℝ) := 
  k ≤ l ∧ inside_square k l

noncomputable def circumscribed_circle_rectangle (k l : ℝ) := 
  circle (sqrt (k^2 + l^2) / 2)

noncomputable def inscribed_circle_rectangle (k : ℝ) := 
  circle (k / 2)

noncomputable def shaded_area_condition (s k l : ℝ) :=
  (s^2 - (k^2 + l^2) = 8*k^2)

noncomputable def fraction_shaded_largest_circle (s k : ℝ) :=
  (9 * (π * k^2 / 4)) / (π * (s^2) / 4) = 9 / 25

theorem fraction_of_largest_circle_shaded (s k l : ℝ) (h1 : square_inscribed_circle s) 
  (h2 : rectangle_inscribed_square k l) 
  (h3 : circumscribed_circle_rectangle k l) 
  (h4 : inscribed_circle_rectangle k) 
  (h5 : shaded_area_condition s k l) 
  : fraction_shaded_largest_circle s k := 
begin
  sorry
end

end fraction_of_largest_circle_shaded_l725_725760


namespace order_of_abc_l725_725835

noncomputable def a : ℝ := 0.1 * Real.exp 0.1
noncomputable def b : ℝ := 1 / 9
noncomputable def c : ℝ := -Real.log 0.9

theorem order_of_abc : b > a ∧ a > c :=
by
  sorry

end order_of_abc_l725_725835


namespace ABCD_is_square_if_PQRS_is_square_l725_725573

variables (A B C D P Q R S : Type) [convex_quadrilateral A B C D]
variables [on_extension P A B] [on_extension Q B C] [on_extension R C D] [on_extension S D A]
variables [BP CQ DR AS : = eq_points]

theorem ABCD_is_square_if_PQRS_is_square :
  square PQRS → square ABCD :=
begin
  sorry
end

end ABCD_is_square_if_PQRS_is_square_l725_725573


namespace lcm_of_36_and_100_l725_725441

theorem lcm_of_36_and_100 : Nat.lcm 36 100 = 900 :=
by
  -- The proof is omitted
  sorry

end lcm_of_36_and_100_l725_725441


namespace seeds_per_packet_l725_725389

theorem seeds_per_packet (total_seedlings packets : ℕ) (h1 : total_seedlings = 420) (h2 : packets = 60) : total_seedlings / packets = 7 :=
by 
  sorry

end seeds_per_packet_l725_725389


namespace cost_price_eq_l725_725020

variable (SP : Real) (profit_percentage : Real)

theorem cost_price_eq : SP = 100 → profit_percentage = 0.15 → (100 / (1 + profit_percentage)) = 86.96 :=
by
  intros hSP hProfit
  sorry

end cost_price_eq_l725_725020


namespace average_value_l725_725087

noncomputable def average_sum (a : Fin 12 → ℕ) : ℚ :=
  (1 / ↑(list.permutations [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).length.to_rat) *
  list.sum (list.map (λ l, abs (l.nth 0 - l.nth 1) + abs (l.nth 2 - l.nth 3) + abs (l.nth 4 - l.nth 5) + abs (l.nth 6 - l.nth 7) + abs (l.nth 8 - l.nth 9) + abs (l.nth 10 - l.nth 11))
            (list.permutations [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))

theorem average_value :
  (average_sum = 143 / 33) :=
sorry

end average_value_l725_725087


namespace wrongly_written_height_l725_725651

-- Define the conditions
def avg_height {n : ℕ} (h : Fin n → ℝ) := (∑ i, h i) / n

axiom num_boys : ℕ
axiom original_avg_height : ℝ
axiom actual_avg_height : ℝ
axiom wrong_height : ℝ
axiom correct_height : ℝ

-- Given conditions
def conditions : Prop := 
  num_boys = 35 ∧ 
  original_avg_height = 185 ∧ 
  correct_height = 106 ∧
  actual_avg_height = 183

-- The theorem to prove the wrongly written height
theorem wrongly_written_height (H : conditions) : wrong_height = 176 :=
by sorry

end wrongly_written_height_l725_725651


namespace ordering_of_means_l725_725946

variable {a b : ℝ}

-- Define that a and b are positive and unequal
def conditions : Prop := a > 0 ∧ b > 0 ∧ a ≠ b

-- Define the arithmetic mean
def AM (a b : ℝ) : ℝ := (a^2 + b^2) / 2

-- Define the geometric mean
def GM (a b : ℝ) : ℝ := a * b

-- Define the harmonic mean
def HM (a b : ℝ) : ℝ := 2 * a^2 * b^2 / (a^2 + b^2)

theorem ordering_of_means :
  conditions → AM a b > GM a b ∧ GM a b > HM a b :=
by
  intros
  sorry

end ordering_of_means_l725_725946


namespace common_area_ratio_l725_725149

noncomputable theory

/-- Given two identical intersecting circles with radius R where the distance between their centers is 2mR,
and a third circle externally touching the two original circles and their common tangent,
the ratio of the area of the common part of the first two circles to the area of the third circle
is equal to 2 * (arccos m - m * sqrt(1 - m^2)) / (π * m^2) --/
theorem common_area_ratio (R m : ℝ) (h : 0 ≤ m ∧ m ≤ 1) :
  (2 * (Real.arccos m - m * Real.sqrt (1 - m^2))) / (Real.pi * m^2) =
  let x := m * R in
  let third_circle_area := Real.pi * (x^2) in
  let common_area := 2 * (R^2) * (Real.arccos m - m * Real.sqrt (1 - m^2)) in
  common_area / third_circle_area :=
by sorry

end common_area_ratio_l725_725149


namespace projection_of_vec_l725_725678

open Real

def vec2 := (ℝ × ℝ)

def proj (u v : vec2) : vec2 :=
  let dot_uu := (u.1 * u.1 + u.2 * u.2)
  let dot_vu := (v.1 * u.1 + v.2 * u.2)
  ((dot_vu / dot_uu) * u.1, (dot_vu / dot_uu) * u.2)

def vec_eq (u v : vec2) : Prop :=
  u.1 = v.1 ∧ u.2 = v.2

theorem projection_of_vec :
  vec_eq (proj ⟨1, -2⟩ ⟨3, -2⟩) ⟨7/5, -14/5⟩ :=
by
  sorry

end projection_of_vec_l725_725678


namespace atomic_weight_of_Cl_l725_725465

noncomputable def atomic_weight_of_Ba : ℝ := 137.33
noncomputable def molecular_weight_of_BaCl2 : ℝ := 207

theorem atomic_weight_of_Cl :
  ∃ Cl : ℝ, molecular_weight_of_BaCl2 = atomic_weight_of_Ba + 2 * Cl ∧ Cl = 34.835 :=
by 
  use 34.835
  split
  · sorry
  · sorry

end atomic_weight_of_Cl_l725_725465


namespace cakes_baked_at_lunch_today_l725_725016

variables (sold_today left_today baked_yesterday : ℕ)

theorem cakes_baked_at_lunch_today (h1 : sold_today = 6) (h2 : left_today = 2) (h3 : baked_yesterday = 3) :
  let L := 11 - baked_yesterday in
  L = 8 :=
by
  have h_total_available_today := eq_add_of_sub_eq' (rfl : (sold_today + left_today) = 8),
  have h_total_today := eq_add_of_sub_eq' (rfl : (sold_today + left_today + baked_yesterday) = 11),
  show let L := 11 - baked_yesterday from L = 8, sorry

end cakes_baked_at_lunch_today_l725_725016


namespace six_digit_palindromes_count_l725_725163

def is_digit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9
def is_non_zero_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

theorem six_digit_palindromes_count : 
  (∃a b c : ℕ, is_non_zero_digit a ∧ is_digit b ∧ is_digit c) → 
  (∃ n : ℕ, n = 900) :=
by
  sorry

end six_digit_palindromes_count_l725_725163


namespace tomatoes_left_l725_725689

theorem tomatoes_left (initial_tomatoes : ℕ) (fraction_eaten : ℚ) (eaters : ℕ) (final_tomatoes : ℕ)  
  (h_initial : initial_tomatoes = 21)
  (h_fraction : fraction_eaten = 1 / 3)
  (h_eaters : eaters = 2)
  (h_final : final_tomatoes = initial_tomatoes - initial_tomatoes * fraction_eaten) :
  final_tomatoes = 14 := by
  sorry

end tomatoes_left_l725_725689


namespace equilateral_triangle_properties_l725_725439

noncomputable def height_of_equilateral_triangle (a : ℝ) : ℝ :=
  a * real.sqrt 3 / 2

noncomputable def radius_inscribed_circle (a : ℝ) : ℝ :=
  (a * real.sqrt 3) / 6

noncomputable def radius_circumscribed_circle (a : ℝ) : ℝ :=
  (a * real.sqrt 3) / 3

theorem equilateral_triangle_properties (a : ℝ) :
    (height_of_equilateral_triangle a = a * real.sqrt 3 / 2) ∧
    (radius_inscribed_circle a = a * real.sqrt 3 / 6) ∧
    (radius_circumscribed_circle a = a * real.sqrt 3 / 3) :=
by
  -- Proof goes here, but it's skipped as per instruction.
  sorry

end equilateral_triangle_properties_l725_725439


namespace range_of_f_l725_725060

noncomputable def f (x : ℝ) : ℝ := if x ≠ 5 then (3 * (x - 5) * (x + 2)) / (x - 5) else 0

theorem range_of_f :
  set.range (λ x : ℝ, if x ≠ 5 then f x else 0) = set.univ \ {21} :=
sorry

end range_of_f_l725_725060


namespace sum_of_ages_l725_725704

def Tyler_age : ℕ := 5

def Clay_age (T C : ℕ) : Prop :=
  T = 3 * C + 1

theorem sum_of_ages (C : ℕ) (h : Clay_age Tyler_age C) :
  Tyler_age + C = 6 :=
sorry

end sum_of_ages_l725_725704


namespace area_bounded_by_graphs_eq_4_l725_725415

theorem area_bounded_by_graphs_eq_4 :
  let r₁ (θ : ℝ) := 2 / (cos θ)
  let r₂ (θ : ℝ) := 2 / (sin θ)
  ∀ (θ₁ θ₂ : ℝ) (x ℝ: ℝ), 0 ≤ θ₁ ∧ θ₁ ≤ π/2 ∧ 0 ≤ θ₂ ∧ θ₂ ≤ π/2 ∧
  x = r₁ θ₁ ∧ y = r₂ θ₂ →
  area (bounded_region r₁ r₂ 0 0) = 4 := by
  sorry

end area_bounded_by_graphs_eq_4_l725_725415


namespace torn_pages_count_l725_725907

theorem torn_pages_count (pages : Finset ℕ) (h1 : ∀ p ∈ pages, 1 ≤ p ∧ p ≤ 100) (h2 : pages.sum id = 4949) : 
  100 - pages.card = 3 := 
by
  sorry

end torn_pages_count_l725_725907


namespace weighted_average_correct_l725_725711

theorem weighted_average_correct :
  let weights := [0.5, 0.25, 1, 2.5, 0.75, 1.25, 1.5]
  let values := [1200, 1300, 1400, 1510, 1520, 1530, 1200]
  let weighted_sum := (values.zip weights).map (λ ⟨v, w⟩, v * w).sum
  let total_weight := weights.sum
  weighted_sum / total_weight = 1413.23 :=
by
  let weights := [0.5, 0.25, 1, 2.5, 0.75, 1.25, 1.5]
  let values := [1200, 1300, 1400, 1510, 1520, 1530, 1200]
  have weighted_sum : (values.zip weights).map (λ ⟨v, w⟩, v * w).sum = 10952.5 := sorry
  have total_weight : weights.sum = 7.75 := sorry
  show 10952.5 / 7.75 = 1413.23, from sorry

end weighted_average_correct_l725_725711


namespace sum_of_squares_of_coeffs_l725_725305

theorem sum_of_squares_of_coeffs :
  let p := 3 * (X^5 - 2 * X^3 + 4 * X - 1)
  (p.coeff 0)^2 + (p.coeff 1)^2 + (p.coeff 2)^2 + (p.coeff 3)^2 + (p.coeff 4)^2 + (p.coeff 5)^2 = 198 := by
  sorry

end sum_of_squares_of_coeffs_l725_725305


namespace matrix_no_inverse_implies_x_eq_5_l725_725496

theorem matrix_no_inverse_implies_x_eq_5 (x : ℝ) :
  let M : Matrix (Fin 2) (Fin 2) ℝ := ![![x, 5], ![6, 6]] in
  ¬ (M.det ≠ 0) → x = 5 :=
by
  sorry

end matrix_no_inverse_implies_x_eq_5_l725_725496


namespace percent_freshmen_psych_majors_l725_725722

-- Define the total number of students as 100 for simplicity.
def total_students : ℕ := 100

-- Define the percentage of students who are freshmen.
def freshmen_percent : ℝ := 0.60

-- Define the percentage of freshmen who are enrolled in the School of Liberal Arts.
def sla_freshmen_percent : ℝ := 0.40

-- Define the percentage of Liberal Arts freshmen who are psychology majors.
def psych_major_percent : ℝ := 0.20

-- Define the number of freshmen.
def number_of_freshmen : ℕ := (total_students * freshmen_percent).toNat

-- Define the number of Liberal Arts freshmen.
def number_of_sla_freshmen : ℕ := (number_of_freshmen * sla_freshmen_percent).toNat

-- Define the number of psychology major freshmen in the School of Liberal Arts.
def number_of_psych_majors : ℝ := number_of_sla_freshmen * psych_major_percent

-- Define the percentage of total students who are freshmen psychology majors in the School of Liberal Arts.
def percentage_of_psych_majors : ℝ := (number_of_psych_majors / total_students) * 100

theorem percent_freshmen_psych_majors : percentage_of_psych_majors = 4.8 :=
by
  sorry

end percent_freshmen_psych_majors_l725_725722


namespace linear_equation_has_infinitely_many_solutions_l725_725671

theorem linear_equation_has_infinitely_many_solutions :
  ∃ f : ℚ → ℚ × ℚ, (∀ t : ℚ, let (a, b) := f t in 5 * a - 11 * b = 21) :=
sorry

end linear_equation_has_infinitely_many_solutions_l725_725671


namespace problem1_problem2_l725_725479

theorem problem1 (
  f : ℝ → ℝ
  (h_even : ∀ x : ℝ, f (-x) = f x)
  (h_passes : f 2 = 5)
  (g : ℝ → ℝ := λ x, (x + a) * f x)
) : 
  (-∞ < a ∧ a ≤ -real.sqrt 3) ∨ (real.sqrt 3 ≤ a ∧ a < +∞) :=
begin
  -- Proof goes here
  sorry
end

theorem problem2 (
  g' : ℝ → ℝ := λ x, 3 * x ^ 2 + 2 * 2 * x + 1,
  (h_extremum : g' (-1) = 0) -- given conditions for a=2
) : 
  (∀ x, x ∈ Iio (-1) → g' x > 0) ∧ 
  (∀ x, x ∈ Ioo (-1, -1 / 3) → g' x < 0) ∧ 
  (∀ x, x ∈ Ioi (-1 / 3) → g' x > 0) :=
begin
  -- Proof goes here
  sorry
end

end problem1_problem2_l725_725479


namespace min_f_prime_convex_f_min_f_sum_l725_725867

noncomputable def f (x : ℝ) : ℝ := Real.exp x - (1 / 2) * x^2
noncomputable def f' (x : ℝ) : ℝ := Real.exp x - x

theorem min_f_prime : ∃ x : ℝ, f' x = 1 := sorry

theorem convex_f (x1 x2 : ℝ) (hx1 : 0 ≤ x1) (hx2 : 0 ≤ x2) (λ1 λ2 : ℝ) (hλ1 : λ1 ≥ 0) (hλ2 : λ2 ≥ 0) (hλ_sum : λ1 + λ2 = 1) :
  f (λ1 * x1 + λ2 * x2) ≤ λ1 * f x1 + λ2 * f x2 :=
sorry

theorem min_f_sum (x1 x2 x3 : ℝ) (hx1 : 0 ≤ x1) (hx2 : 0 ≤ x2) (hx3 : 0 ≤ x3) (hsum : x1 + x2 + x3 = 3) :
  f x1 + f x2 + f x3 = 3 * Real.exp 1 - (3 / 2) :=
sorry

end min_f_prime_convex_f_min_f_sum_l725_725867


namespace part_a_part_b_l725_725720

variables {F K : Type*} [Field F] [Field K]
variables (α β : F) (m n : ℕ)
open Classical

noncomputable theory

-- Part (a)
theorem part_a (h_fin_ext : IsFiniteExtension F K) (h_eq : K = F(α, β)) :
  degree K F ≤ degree F(α) F * degree F(β) F := 
sorry

-- Part (b)
theorem part_b (h_fin_ext : IsFiniteExtension F K) (h_eq : K = F(α, β)) 
  (h_gcd : gcd (degree F(α) F) (degree F(β) F) = 1) :
  degree K F = degree F(α) F * degree F(β) F := 
sorry

-- Part (c)
example : degree ℚ(√2, √3) ℚ = degree ℚ(√2) ℚ * degree ℚ(√3) ℚ := 
begin
  have h1 : degree ℚ(√2) ℚ = 2 := sorry,
  have h2 : degree ℚ(√3) ℚ = 2 := sorry,
  show degree ℚ(√2, √3) ℚ = 2 * 2,
  sorry
end

end part_a_part_b_l725_725720


namespace common_chord_equation_l725_725275

theorem common_chord_equation
  (r : ℝ) (h : r > 0) :
  (∃ (ρ θ : ℝ), (ρ = r ∨ ρ = -2 * r * sin (θ + π / 4)) →
    sqrt 2 * ρ * (sin θ + cos θ) = -r) :=
sorry

end common_chord_equation_l725_725275


namespace angle_ACB_is_30_l725_725728

theorem angle_ACB_is_30 (A B C D : Type)
  (α β γ : Prop)
  [is_internal_angle_A : α > 60]
  [is_internal_angle_B : β < 60]
  [is_equilateral_ABD : triangle_equi ABD]
  [is_isosceles_ACD : triangle_iso ACD]
  [is_isosceles_BCD : triangle_iso BCD]:
  γ = 30 :=
by sorry

-- Custom definitions for necessary geometric properties
axiom triangle_equi (t : Type) : Prop -- denoting equilateral triangle
axiom triangle_iso (t : Type) : Prop  -- denoting isosceles triangle

end angle_ACB_is_30_l725_725728


namespace find_possible_phi_l725_725279

theorem find_possible_phi (k : ℤ) :
  ∃ φ : ℝ, translated_odd_function (φ, k, 2, 1/6) ∧
           φ = k * real.pi + 5 * real.pi / 6 :=
sorry

/- Additional definitions related to conditions from part a) -/
def translated_odd_function (φ : ℝ, k : ℤ, ω : ℝ, h : ℝ) :=
  let new_φ := 2 * h * ω in
  ∃ a : ℤ, φ - new_φ = a * real.pi + real.pi / 2

end find_possible_phi_l725_725279


namespace range_of_a_l725_725872

theorem range_of_a (a : ℝ) (x y : ℝ) (hxy : x * y > 0) (hx : 0 < x) (hy : 0 < y) :
  (x + y) * (1 / x + a / y) ≥ 9 → a ≥ 4 :=
by
  intro h
  sorry

end range_of_a_l725_725872


namespace kenneth_distance_past_finish_l725_725781

noncomputable def distance_past_finish_line (race_distance : ℕ) (biff_speed : ℕ) (kenneth_speed : ℕ) : ℕ :=
  let biff_time := race_distance / biff_speed
  let kenneth_distance := kenneth_speed * biff_time
  kenneth_distance - race_distance

theorem kenneth_distance_past_finish (race_distance : ℕ) (biff_speed : ℕ) (kenneth_speed : ℕ) (finish_line_distance : ℕ) : 
  race_distance = 500 ->
  biff_speed = 50 -> 
  kenneth_speed = 51 ->
  finish_line_distance = 10 ->
  distance_past_finish_line race_distance biff_speed kenneth_speed = finish_line_distance := by
  sorry

end kenneth_distance_past_finish_l725_725781


namespace event_probabilities_l725_725734

namespace DiceGame

def total_outcomes := 36

def EventA := {(1, 2), (2, 1), (2, 3), (3, 2), (3, 4), 
               (4, 3), (4, 5), (5, 4), (5, 6), (6, 5)}

def EventB := {(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)}

def EventC := {(1, 3), (1, 5), (3, 1), (3, 5), (5, 1), (5, 3), 
               (2, 4), (2, 6), (4, 2), (4, 6), (6, 2), (6, 4)}

def probability (event : Set (ℕ × ℕ)) : ℚ :=
  (event.to_finset.card : ℚ) / total_outcomes

theorem event_probabilities :
  probability EventB = 1 / 6 ∧
  probability EventA = 5 / 18 ∧
  probability EventC = 1 / 3 ∧
  probability EventB + probability EventA + probability EventC + (1 - (probability EventB + probability EventA + probability EventC)) = 1 :=
by
  sorry

end DiceGame

end event_probabilities_l725_725734


namespace berries_from_fourth_bush_l725_725600

def number_of_berries (n : ℕ) : ℕ :=
  match n with
  | 1 => 3
  | 2 => 4
  | 3 => 7
  | 5 => 19
  | _ => sorry  -- Assume the given pattern

theorem berries_from_fourth_bush : number_of_berries 4 = 12 :=
by sorry

end berries_from_fourth_bush_l725_725600


namespace payback_period_l725_725993

def system_unit_cost : ℕ := 9499 -- cost in RUB
def graphics_card_cost : ℕ := 20990 -- cost per card in RUB
def num_graphics_cards : ℕ := 2
def system_unit_power : ℕ := 120 -- power in watts
def graphics_card_power : ℕ := 185 -- power per card in watts
def earnings_per_card_per_day_ethereum : ℚ := 0.00630
def ethereum_to_rub : ℚ := 27790.37 -- RUB per ETH
def electricity_cost_per_kwh : ℚ := 5.38 -- RUB per kWh
def total_investment : ℕ := system_unit_cost + num_graphics_cards * graphics_card_cost
def total_power_consumption_watts : ℕ := system_unit_power + num_graphics_cards * graphics_card_power
def total_power_consumption_kwh_per_day : ℚ := total_power_consumption_watts / 1000 * 24
def daily_earnings_rub : ℚ := earnings_per_card_per_day_ethereum * num_graphics_cards * ethereum_to_rub
def daily_energy_cost : ℚ := total_power_consumption_kwh_per_day * electricity_cost_per_kwh
def net_daily_profit : ℚ := daily_earnings_rub - daily_energy_cost

theorem payback_period : total_investment / net_daily_profit = 179 := by
  sorry

end payback_period_l725_725993


namespace unique_solution_positive_integers_l725_725411

theorem unique_solution_positive_integers :
  ∀ (a b : ℕ), (0 < a ∧ 0 < b ∧ ∃ k m : ℤ, a^3 + 6 * a * b + 1 = k^3 ∧ b^3 + 6 * a * b + 1 = m^3) → (a = 1 ∧ b = 1) :=
by
  -- Proof goes here
  sorry

end unique_solution_positive_integers_l725_725411


namespace fraction_of_triangle_area_l725_725629

open Real

def point := (ℝ × ℝ)

def area_of_triangle (A B C : point) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (abs ((x1 * (y2 - y3)) + (x2 * (y3 - y1)) + (x3 * (y1 - y2))) / 2)

def A : point := (2, 0)
def B : point := (8, 12)
def C : point := (14, 0)

def X : point := (6, 0)
def Y : point := (8, 4)
def Z : point := (10, 0)

theorem fraction_of_triangle_area :
  (area_of_triangle X Y Z) / (area_of_triangle A B C) = 1 / 9 :=
by
  sorry

end fraction_of_triangle_area_l725_725629


namespace lcm_36_100_eq_900_l725_725448

/-- Definition for the prime factorization of 36 -/
def factorization_36 : Prop := 36 = 2^2 * 3^2

/-- Definition for the prime factorization of 100 -/
def factorization_100 : Prop := 100 = 2^2 * 5^2

/-- The least common multiple problem statement -/
theorem lcm_36_100_eq_900 (h₁ : factorization_36) (h₂ : factorization_100) : Nat.lcm 36 100 = 900 := 
by
  sorry

end lcm_36_100_eq_900_l725_725448


namespace initial_loss_percentage_is_10_l725_725024

-- Given conditions
def cost_price : ℝ := 1500
def selling_price_gain : ℝ := cost_price + 0.04 * cost_price
def selling_price_loss : ℝ := selling_price_gain - 210

-- Question: What was the initial loss percentage?
def initial_loss_percentage (cp sp : ℝ) : ℝ := ((cp - sp) / cp) * 100

-- Prove that the initial loss percentage is 10%
theorem initial_loss_percentage_is_10 :
  initial_loss_percentage cost_price selling_price_loss = 10 := 
by 
  sorry

end initial_loss_percentage_is_10_l725_725024


namespace at_least_6_heads_in_10_flips_l725_725338

def coin_flip : Type := bool

def is_heads (x : coin_flip) : Prop := x = tt

def num_consecutive_heads (l : list coin_flip) (n : ℕ) : Prop :=
  ∃ i : ℕ, i + n ≤ l.length ∧ l.drop i.take n = list.replicate n tt

def prob_at_least_n_consecutive_heads (l : list coin_flip) (n : ℕ) : Prop :=
  ∃ i ≤ l.length - n + 1, list.replicate n tt = l.drop (i - 1).take n

noncomputable def at_least_6_heads_in_10_flips_prob : ℚ :=
  (129:ℚ) / (1024:ℚ)

theorem at_least_6_heads_in_10_flips :
  prob_at_least_n_consecutive_heads (list.replicate 10 coin_flip) 6 = at_least_6_heads_in_10_flips_prob :=
by
  sorry

end at_least_6_heads_in_10_flips_l725_725338


namespace table_condition_l725_725194

-- Define the conditions used in the problem
variables (matrix : ℕ → ℕ → ℕ)
variables (rows cols : ℕ)
variables (positive_in_every_row : ∀ i < rows, ∃ j < cols, 0 < matrix i j)
variables (positive_in_every_col : ∀ j < cols, ∃ i < rows, 0 < matrix i j)
variables (sum_condition : ∀ i j, 0 < matrix i j → (∑ k, matrix i k) = (∑ k, matrix k j))

-- State the theorem we need to prove
theorem table_condition (h : rows = 2015) : cols = 2015 :=
sorry

end table_condition_l725_725194


namespace inequality_proof_l725_725231

theorem inequality_proof 
  (a b c : ℝ) 
  (h1 : a ≥ b) 
  (h2 : b ≥ c) 
  (h3 : c > 0) :
  (a^2 - b^2) / c + (c^2 - b^2) / a + (a^2 - c^2) / b ≥ 3 * a - 4 * b + c :=
  sorry

end inequality_proof_l725_725231


namespace evaluate_fraction_l725_725068

theorem evaluate_fraction :
  (20 - 18 + 16 - 14 + 12 - 10 + 8 - 6 + 4 - 2) / (2 - 4 + 6 - 8 + 10 - 12 + 14 - 16 + 18) = 1 :=
by
  sorry

end evaluate_fraction_l725_725068


namespace integral_sqrt_x_squared_plus_one_l725_725080

theorem integral_sqrt_x_squared_plus_one (C : ℝ) :
  ∫ (x : ℝ) in  -∞..∞, (x / (sqrt (x^2 + 1))) = sqrt (x^2 + 1) + C :=
sorry

end integral_sqrt_x_squared_plus_one_l725_725080


namespace monotonicity_of_f_sum_of_squares_of_roots_l725_725128

noncomputable def f (x a : Real) : Real := Real.log x - a * x^2

theorem monotonicity_of_f (a : Real) :
  (a ≤ 0 → ∀ x y : Real, 0 < x → x < y → f x a < f y a) ∧
  (a > 0 → ∀ x y : Real, 0 < x → x < Real.sqrt (1/(2 * a)) → Real.sqrt (1/(2 * a)) < y → f x a < f (Real.sqrt (1/(2 * a))) a ∧ f (Real.sqrt (1/(2 * a))) a > f y a) :=
by sorry

theorem sum_of_squares_of_roots (a x1 x2 : Real) (h1 : f x1 a = 0) (h2 : f x2 a = 0) (h3 : x1 ≠ x2) :
  x1^2 + x2^2 > 2 * Real.exp 1 :=
by sorry

end monotonicity_of_f_sum_of_squares_of_roots_l725_725128


namespace percentage_greater_than_l725_725883

-- Definitions of the variables involved
variables (X Y Z : ℝ)

-- Lean statement to prove the formula
theorem percentage_greater_than (X Y Z : ℝ) : 
  (100 * (X - Y)) / (Y + Z) = (100 * (X - Y)) / (Y + Z) :=
by
  -- skipping the actual proof
  sorry

end percentage_greater_than_l725_725883


namespace tetrahedron_circumscribed_sphere_volume_l725_725549

-- Define the lengths of PA, PB, PC
def PA : ℝ := Real.sqrt 2
def PB : ℝ := Real.sqrt 3
def PC : ℝ := 2

-- Define mutually perpendicular condition
def mutually_perpendicular (x y z : ℝ) : Prop :=
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x*y = 0 ∧ y*z = 0 ∧ x*z = 0

-- Volume of circumscribed sphere
def circumscribed_sphere_volume (PA PB PC : ℝ) : ℝ :=
  let r := (Real.sqrt (PA^2 + PB^2 + PC^2)) / 2
  in (4 / 3) * Real.pi * r^3

-- The statement to prove
theorem tetrahedron_circumscribed_sphere_volume :
  mutually_perpendicular PA PB PC →
  circumscribed_sphere_volume PA PB PC = (9 * Real.pi) / 2 :=
by
  sorry

end tetrahedron_circumscribed_sphere_volume_l725_725549


namespace area_triangle_POF_l725_725223

theorem area_triangle_POF (O : ℝ × ℝ) (F : ℝ × ℝ) (P : ℝ × ℝ)
  (hO : O = (0, 0))
  (hF : F = (0, 1))
  (hP_on_parabola : P.snd = (1 / 4) * P.fst^2)
  (hPF : real.sqrt ((P.fst - F.fst)^2 + (P.snd - F.snd)^2) = 4) :
  abs (P.fst * 1 / 2) = real.sqrt 3 := by
  sorry

end area_triangle_POF_l725_725223


namespace number_of_six_digit_palindromes_l725_725161

theorem number_of_six_digit_palindromes : 
  let count_palindromes : ℕ := 9 * 10 * 10 in
  count_palindromes = 900 :=
by
  sorry

end number_of_six_digit_palindromes_l725_725161


namespace scientific_notation_104000000_l725_725971

theorem scientific_notation_104000000 :
  104000000 = 1.04 * 10^8 :=
sorry

end scientific_notation_104000000_l725_725971


namespace simplify_vector_expression_l725_725258

-- Definitions for vectors
variables {A B C D : Type} [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D]

-- Defining the vectors
variables (AB CA BD CD : A)

-- A definition using the head-to-tail addition of vectors.
def vector_add (v1 v2 : A) : A := v1 + v2

-- Statement to prove
theorem simplify_vector_expression :
  vector_add (vector_add AB CA) BD = CD :=
sorry

end simplify_vector_expression_l725_725258


namespace minimum_pips_on_surface_of_glued_dice_l725_725244

theorem minimum_pips_on_surface_of_glued_dice :
  (∀ n : ℕ, ∃ d1 d2 d3 d4 : ℕ, 
    (d1 + d2 = 7 ∧ d3 + d4 = 7) ∧
    (d1 + d3 = 7 ∧ d2 + d4 = 7) ∧
    n = d1 + d2 + d3 + d4) →
  (minimum_pips (dice_config : list ℕ) = 58) := 
sorry

end minimum_pips_on_surface_of_glued_dice_l725_725244


namespace convex_sets_common_point_l725_725636

variables {α : Type*} [linear_ordered_field α] [convex_space α]

-- Definitions from conditions
variables (n : ℕ) (M : fin n → set α)
variable (h1 : 3 ≤ n)
variable (h2 : ∀ i j k : fin n, (M i ∩ M j ∩ M k).nonempty)

-- The proof that needs to be written
theorem convex_sets_common_point :
  ∃ x, ∀ i : fin n, x ∈ M i :=
sorry

end convex_sets_common_point_l725_725636


namespace at_least_six_heads_in_10_flips_is_129_over_1024_l725_725349

def fair_coin_flip (n : ℕ) (prob_heads prob_tails : ℚ) : Prop :=
  (prob_heads = 1/2 ∧ prob_tails = 1/2)

noncomputable def at_least_six_consecutive_heads_probability (n : ℕ) : ℚ :=
  if n = 10 then 129 / 1024 else 0  -- this is specific to 10 flips and should be defined based on actual calculation for different n
  
theorem at_least_six_heads_in_10_flips_is_129_over_1024 :
  fair_coin_flip 10 (1/2) (1/2) →
  at_least_six_consecutive_heads_probability 10 = 129 / 1024 :=
by
  intros
  sorry

end at_least_six_heads_in_10_flips_is_129_over_1024_l725_725349


namespace x_coordinate_of_point_M_l725_725683

-- Define the parabola equation
def parabola (x y : ℝ) : Prop := y^2 = (1/4) * x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1/16, 0)

-- Define a point on the parabola at a distance of 1 from the focus
def on_parabola_at_distance_1 (x y : ℝ) : Prop :=
  parabola x y ∧ dist (x, y) focus = 1

-- Prove the x-coordinate of such a point
theorem x_coordinate_of_point_M : ∃ (x : ℝ), ∃ (y : ℝ), on_parabola_at_distance_1 x y ∧ x = 15 / 16 := by
  sorry

end x_coordinate_of_point_M_l725_725683


namespace polynomial_evaluation_l725_725141

theorem polynomial_evaluation (p : ℕ → ℝ) (n : ℕ) (h : ∀ k, k ≤ n → p k = 1 / (Nat.choose (n + 1) k)) :
  p (n + 1) = if Nat.even n then 1 else 0 :=
by
  sorry

end polynomial_evaluation_l725_725141


namespace square_land_plot_area_l725_725314

theorem square_land_plot_area (side_length : ℕ) (h1 : side_length = 40) : side_length * side_length = 1600 :=
by
  sorry

end square_land_plot_area_l725_725314


namespace sugar_recipes_l725_725742

theorem sugar_recipes (container_sugar recipe_sugar : ℚ) 
  (h1 : container_sugar = 56 / 3) 
  (h2 : recipe_sugar = 3 / 2) :
  container_sugar / recipe_sugar = 112 / 9 := sorry

end sugar_recipes_l725_725742


namespace sum_of_coordinates_of_point_S_l725_725618

-- Definitions for the points
def P : ℝ × ℝ := (2, 7)
def Q : ℝ × ℝ := (3, 2)
def R : ℝ × ℝ := (6, 4)

-- Statement to prove
theorem sum_of_coordinates_of_point_S : ∃ S : ℝ × ℝ, 
  (let midpoint_PQ := (P.1 + Q.1) / 2, (P.2 + Q.2) / 2,
       midpoint_QR := (Q.1 + R.1) / 2, (Q.2 + R.2) / 2,
       RS_midpoint := 6, 5 in
   midpoint_PQ = (2.5, 4.5) ∧ midpoint_QR = (4.5, 3) ∧ 
   (S = (6, 6) ∧ S.1 + S.2 = 12)) :=
sorry

end sum_of_coordinates_of_point_S_l725_725618


namespace hemisphere_containers_needed_l725_725375

theorem hemisphere_containers_needed 
  (total_volume : ℕ) (volume_per_hemisphere : ℕ) 
  (h₁ : total_volume = 11780) 
  (h₂ : volume_per_hemisphere = 4) : 
  total_volume / volume_per_hemisphere = 2945 := 
by
  sorry

end hemisphere_containers_needed_l725_725375


namespace vertical_line_divides_equal_areas_l725_725225

-- Define vertices
def A : ℝ × ℝ := (0, 3)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (10, 0)

-- Define triangle area function
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

-- Prove that the vertical line dividing the triangle into two equal areas is at x = 5
theorem vertical_line_divides_equal_areas (a : ℝ) :
  let area := triangle_area A B C in
  (∃ a, 2 * triangle_area A B (a, 0) = area) ↔ a = 5 :=
by sorry

end vertical_line_divides_equal_areas_l725_725225


namespace length_of_AD_l725_725001

theorem length_of_AD (A B C D M : Point) (circle : Circle) : 
  (circle.passesThrough A) ∧ 
  (circle.passesThrough B) ∧ 
  (circle.passesThrough C) ∧ 
  (circle.isTangentTo (lineThrough A D)) ∧ 
  (circle.intersects (lineThrough C D) M) ∧ 
  (distance B M = 9) ∧ 
  (distance D M = 8) ∧ 
  (parallelogram A B C D) → 
  distance A D = 6 * sqrt 2 := 
by 
  sorry

end length_of_AD_l725_725001


namespace rank_eq_n_iff_comm_zero_l725_725264

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (T U : V →ₗ[ℝ] V) 

-- T + U being invertible is expressed.
variable (h_inv : LinearMap.ker (T + U) = ⊥ ∧ LinearMap.range (T + U) = ⊤)

-- Prove the equivalence statement.
theorem rank_eq_n_iff_comm_zero (n : ℕ) :
  LinearMap.ker (T + U) = ⊥ ∧ LinearMap.range (T + U) = ⊤ →
  (T.comp U = 0 ∧ U.comp T = 0) ↔
  (T.range.dim + U.range.dim = finrank ℝ V) :=
sorry

end rank_eq_n_iff_comm_zero_l725_725264


namespace maria_distributed_money_l725_725964

-- Definitions based on conditions
def rene_received : ℝ := 600
def florence_received : ℝ := 2 * rene_received
noncomputable def isha_received : ℝ := Real.sqrt(2 * florence_received)
def maria_total_money : ℝ := 4 * isha_received
def john_received : ℝ := florence_received / 3
def emma_received : ℝ := (john_received / 2) / 2

-- Total money distributed
noncomputable def total_distributed : ℝ :=
  isha_received + florence_received + rene_received + (john_received / 2) + emma_received

-- Prove that Maria distributed $2149 among her five friends
theorem maria_distributed_money : total_distributed = 2149 := by
  sorry

end maria_distributed_money_l725_725964


namespace probability_of_6_consecutive_heads_l725_725352

/-- Define the probability of obtaining at least 6 consecutive heads in 10 flips of a fair coin. -/
def prob_at_least_6_consecutive_heads : ℚ :=
  129 / 1024

/-- Proof statement: The probability of getting at least 6 consecutive heads in 10 flips of a fair coin is 129/1024. -/
theorem probability_of_6_consecutive_heads : 
  prob_at_least_6_consecutive_heads = 129 / 1024 := 
by
  sorry

end probability_of_6_consecutive_heads_l725_725352


namespace solution_set_of_inequality_l725_725285

theorem solution_set_of_inequality :
  { x : ℝ | |x^2 - 3 * x| > 4 } = { x : ℝ | x < -1 ∨ x > 4 } :=
sorry

end solution_set_of_inequality_l725_725285


namespace fraction_of_areas_l725_725624

/-- Points A, B, C, X, Y, Z coordinates definitions --/
structure Point :=
(x : ℝ)
(y : ℝ)

def A := Point.mk 2 0
def B := Point.mk 8 12
def C := Point.mk 14 0

def X := Point.mk 6 0
def Y := Point.mk 8 4
def Z := Point.mk 10 0

/-- Area of a triangle given base and height --/
def area_triangle (base height : ℝ) : ℝ :=
  (base * height) / 2

/-- Area of triangle ABC --/
def Area_ABC := area_triangle (C.x - A.x) B.y

/-- Area of triangle XYZ --/
def Area_XYZ := area_triangle (Z.x - X.x) Y.y

theorem fraction_of_areas : Area_XYZ / Area_ABC = 1 / 9 := by
  sorry

end fraction_of_areas_l725_725624


namespace two_digit_integers_count_l725_725521

/-- Prove that the number of positive two-digit integers where each digit is either a prime number or a square of a prime number is 36. -/
theorem two_digit_integers_count : 
  let valid_digits := {2, 3, 4, 5, 7, 9}
  in (valid_digits.card * valid_digits.card) = 36 := by
  sorry

end two_digit_integers_count_l725_725521


namespace intersection_A_B_l725_725518

def A (x : ℝ) : Prop := 0 < x ∧ x < 2
def B (x : ℝ) : Prop := -1 < x ∧ x < 1
def C (x : ℝ) : Prop := 0 < x ∧ x < 1

theorem intersection_A_B : ∀ x, A x ∧ B x ↔ C x := by
  sorry

end intersection_A_B_l725_725518


namespace weight_of_each_bag_of_planks_is_14_l725_725796

-- Definitions
def crate_capacity : Nat := 20
def num_crates : Nat := 15
def num_bags_nails : Nat := 4
def weight_bag_nails : Nat := 5
def num_bags_hammers : Nat := 12
def weight_bag_hammers : Nat := 5
def num_bags_planks : Nat := 10
def weight_to_leave_out : Nat := 80

-- Total weight calculations
def weight_nails := num_bags_nails * weight_bag_nails
def weight_hammers := num_bags_hammers * weight_bag_hammers
def total_weight_nails_hammers := weight_nails + weight_hammers
def total_crate_capacity := num_crates * crate_capacity
def weight_that_can_be_loaded := total_crate_capacity - weight_to_leave_out
def weight_available_for_planks := weight_that_can_be_loaded - total_weight_nails_hammers
def weight_each_bag_planks := weight_available_for_planks / num_bags_planks

-- Theorem statement
theorem weight_of_each_bag_of_planks_is_14 : weight_each_bag_planks = 14 :=
by {
  sorry
}

end weight_of_each_bag_of_planks_is_14_l725_725796


namespace number_of_six_digit_palindromes_l725_725157

def is_six_digit_palindrome (n : ℕ) : Prop :=
  let d1 := n / 100000 % 10
  let d2 := n / 10000 % 10
  let d3 := n / 1000 % 10
  let d4 := n / 100 % 10
  let d5 := n / 10 % 10
  let d6 := n % 10
  n >= 100000 ∧ n < 1000000 ∧ d1 > 0 ∧ d1 = d6 ∧ d2 = d5 ∧ d3 = d4

theorem number_of_six_digit_palindromes : 
  {n : ℕ | is_six_digit_palindrome n}.card = 900 := 
sorry

end number_of_six_digit_palindromes_l725_725157


namespace trajectory_is_circle_line_pass_through_N_l725_725148

-- Define the points on the coordinate plane
def A : ℝ × ℝ := (0, 4)
def O : ℝ × ℝ := (0, 0)

-- Define the condition for point M
def condition_M (M : ℝ × ℝ) : Prop := 
  Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2) = 3 * Real.sqrt (M.1^2 + M.2^2)

-- Define the equation of the trajectory
def trajectory_eq (x y : ℝ) : Prop :=
  x^2 + (y + 1/2)^2 = 9/4

-- Define the point N
def N : ℝ × ℝ := (-1/2, 1)

-- Define the line conditions
def vertical_line (x : ℝ) : Prop := x = -1/2
def slope_condition_line (x y : ℝ) : Prop := 4 * x + 3 * y - 1 = 0

-- Theorems to be proved
theorem trajectory_is_circle (M : ℝ × ℝ) (hM : condition_M M) : 
  trajectory_eq M.1 M.2 := sorry

theorem line_pass_through_N (l : ℝ → ℝ → Prop) (hN : l N.1 N.2) 
  (h_segment_length : (∃ x y, trajectory_eq x y ∧ l x y) → 2 * Real.sqrt 2) :
  (vertical_line = l ∨ slope_condition_line = l) := sorry

end trajectory_is_circle_line_pass_through_N_l725_725148


namespace total_daisies_l725_725559

theorem total_daisies (white pink red : ℕ) (h1 : pink = 9 * white) (h2 : red = 4 * pink - 3) (h3 : white = 6) : 
    white + pink + red = 273 :=
by
  sorry

end total_daisies_l725_725559


namespace increasing_function_when_a_eq_2_range_of_a_for_solution_set_l725_725508

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  Real.log x - a * (x - 1) / (x + 1)

theorem increasing_function_when_a_eq_2 :
  ∀ ⦃x⦄, x > 0 → (f 2 x - f 2 1) * (x - 1) > 0 := sorry

theorem range_of_a_for_solution_set :
  ∀ ⦃a x⦄, f a x ≥ 0 ↔ (x ≥ 1) → a ≤ 1 := sorry

end increasing_function_when_a_eq_2_range_of_a_for_solution_set_l725_725508


namespace a_b_condition_l725_725295

theorem a_b_condition (a b : ℂ) (h : (a + b) / a = b / (a + b)) :
  (∃ x y : ℂ, x = a ∧ y = b ∧ ((¬ x.im = 0 ∧ y.im = 0) ∨ (x.im = 0 ∧ ¬ y.im = 0) ∨ (¬ x.im = 0 ∧ ¬ y.im = 0))) :=
by
  sorry

end a_b_condition_l725_725295


namespace max_value_on_interval_l725_725464

-- Define the function f(x) = x(4 - x)
def f (x : ℝ) := x * (4 - x)

-- State the proposition that the maximum value of the function on the interval (0, 4) is 4
theorem max_value_on_interval : 
  ∃ x ∈ Ioo (0 : ℝ) 4, ∀ y ∈ Ioo (0 : ℝ) 4, f y ≤ f x ∧ f x = 4 := 
sorry

end max_value_on_interval_l725_725464


namespace determinant_of_sine_matrix_is_zero_l725_725789

open Matrix

theorem determinant_of_sine_matrix_is_zero :
  let A := λ (i j : Fin 3) => sin ((i.val * 3 + j.val + 1) : ℝ)
  det A = 0 := 
by
  let A := λ (i j : Fin 3) => sin ((i.val * 3 + j.val + 1) : ℝ)
  have : det A = 0 := sorry
  exact this

end determinant_of_sine_matrix_is_zero_l725_725789


namespace transformed_variance_l725_725484

open Real

-- Define the original variance condition
def original_variance_is_2 (n : ℕ) (x : Fin n → ℝ) : Prop :=
  variance x = 2

-- Define the transformation applied to each data point
def transform (n : ℕ) (x : Fin n → ℝ) : Fin n → ℝ :=
  λ i, 3 * x i + 4

-- State the main theorem to be proved
theorem transformed_variance (n : ℕ) (x : Fin n → ℝ) (h : original_variance_is_2 n x) :
  variance (transform n x) = 18 :=
sorry

end transformed_variance_l725_725484


namespace possible_values_of_k_l725_725989

theorem possible_values_of_k
  (x y z : ℝ) (k : ℝ)
  (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) (h4 : y = k * x) 
  (h5 : ((y/x) - (z/y)) - ((z/y) - (x/z)) = (z/y) - (x/z)) :
  k ∈ {-1, 2} :=
sorry

end possible_values_of_k_l725_725989


namespace evaluate_expression_l725_725806

theorem evaluate_expression : (1 / (2 + (1 / (3 + (1 / 4))))) = (13 / 30) :=
by
  sorry

end evaluate_expression_l725_725806


namespace parallelogram_base_length_l725_725727

theorem parallelogram_base_length (b : ℝ) (A : ℝ) (h : ℝ)
  (H1 : A = 288) 
  (H2 : h = 2 * b) 
  (H3 : A = b * h) : 
  b = 12 := 
by 
  sorry

end parallelogram_base_length_l725_725727


namespace fraction_of_area_l725_725621

noncomputable def area_of_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  let base := (C.1 - A.1).abs
  let height := B.2
  (base * height) / 2

theorem fraction_of_area {A B C X Y Z: (ℝ × ℝ)}
  (hA : A = (2, 0)) (hB : B = (8, 12)) (hC : C = (14, 0))
  (hX : X = (6, 0)) (hY : Y = (8, 4)) (hZ : Z = (10, 0)):
  (area_of_triangle X Y Z) / (area_of_triangle A B C) = 1 / 9 :=
by
  -- Skipping the proof with 'sorry'
  sorry

end fraction_of_area_l725_725621


namespace closest_yellow_percentage_is_26_l725_725823

-- Define the number of jelly beans in each bag
def jelly_beans_A : Nat := 24
def jelly_beans_B : Nat := 32
def jelly_beans_C : Nat := 36
def jelly_beans_D : Nat := 40

-- Define the percentage of yellow jelly beans in each bag
def percentage_yellow_A : Float := 0.40
def percentage_yellow_B : Float := 0.30
def percentage_yellow_C : Float := 0.25
def percentage_yellow_D : Float := 0.15

-- Calculate the number of yellow jelly beans in each bag
def yellow_beans_A : Float := jelly_beans_A * percentage_yellow_A
def yellow_beans_B : Float := jelly_beans_B * percentage_yellow_B
def yellow_beans_C : Float := jelly_beans_C * percentage_yellow_C
def yellow_beans_D : Float := jelly_beans_D * percentage_yellow_D

-- Total yellow jelly beans
def total_yellow_beans : Float := yellow_beans_A + yellow_beans_B + yellow_beans_C + yellow_beans_D

-- Total jelly beans
def total_beans : Nat := jelly_beans_A + jelly_beans_B + jelly_beans_C + jelly_beans_D

-- Calculate the ratio and convert to a percentage
def yellow_bean_percentage : Float := (total_yellow_beans / total_beans.toFloat) * 100

theorem closest_yellow_percentage_is_26 :
  yellow_bean_percentage ≈ 26 :=
by
  -- skipping the actual proof
  sorry

end closest_yellow_percentage_is_26_l725_725823


namespace fraction_evaporated_l725_725091

theorem fraction_evaporated (x : ℝ) (h : (1 - x) * (1/4) = 1/6) : x = 1/3 :=
by
  sorry

end fraction_evaporated_l725_725091


namespace angle_sum_l725_725919

theorem angle_sum (y : ℝ) (h : 3 * y + y = 120) : y = 30 :=
sorry

end angle_sum_l725_725919


namespace average_of_roots_l725_725754

theorem average_of_roots (p q : ℝ) (h : ∃ x1 x2 : ℝ, 3*p*x1^2 - 6*p*x1 + q = 0 ∧ 3*p*x2^2 - 6*p*x2 + q = 0 ∧ x1 ≠ x2):
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 3*p*x1^2 - 6*p*x1 + q = 0 ∧ 3*p*x2^2 - 6*p*x2 + q = 0) → 
  (x1 + x2) / 2 = 1 :=
by
  sorry

end average_of_roots_l725_725754


namespace ellipse_sum_a_k_l725_725032

theorem ellipse_sum_a_k :
  let foci1 := (2, 2) in
  let foci2 := (2, 6) in
  let passing_point := (-3, 4) in
  let h := 2 in
  let k := 4 in
  let a := (Real.sqrt 29 + 5) / 2 in
  (a + k) = (Real.sqrt 29 + 13) / 2 := 
  by 
    sorry

end ellipse_sum_a_k_l725_725032


namespace number_of_six_digit_palindromes_l725_725159

theorem number_of_six_digit_palindromes : 
  let count_palindromes : ℕ := 9 * 10 * 10 in
  count_palindromes = 900 :=
by
  sorry

end number_of_six_digit_palindromes_l725_725159


namespace number_of_red_pieces_l725_725979

noncomputable def puzzle_conditions : Prop :=
  ∃ (pieces : ℕ) (red_pieces : ℕ) (non_red_pieces : ℕ),
  pieces = 91 ∧
  red_pieces = pieces - non_red_pieces ∧
  non_red_pieces ∈ ℕ ∧
  (2 * non_red_pieces) = 24 ∧
  red_pieces = 79 

theorem number_of_red_pieces : puzzle_conditions := sorry

end number_of_red_pieces_l725_725979


namespace max_z_in_circle_l725_725462

open Real

noncomputable def z (x y a : ℝ) : ℝ := x^2 - y^2 + 2 * a^2
noncomputable def circle (x y a : ℝ) : Prop := x^2 + y^2 ≤ a^2

theorem max_z_in_circle (a : ℝ) (ha : a ≥ 0) :
  ∃ (p_max p_min : ℝ), 
    (p_max = 3 * a^2 ∧ p_min = a^2) ∧
    (∀ x y, circle x y a → z x y a ≤ p_max) ∧
    (∀ x y, circle x y a → z x y a ≥ p_min) := 
sorry

end max_z_in_circle_l725_725462


namespace stack_map_front_view_correct_l725_725071

def stack_map.front_view (top_view : list (list ℕ)) : list ℕ :=
  top_view.map (λ stack => stack.maximumD 0)

theorem stack_map_front_view_correct :
  stack_map.front_view [[3, 2], [4, 2, 2], [1, 5], [3]] = [3, 4, 5, 3] :=
by
  -- Proof is omitted.
  sorry

end stack_map_front_view_correct_l725_725071


namespace fourth_car_is_lynn_l725_725369

variable {Person : Type}
variables {T J E L M C : Person}
variable (C1 C2 C3 C4 C5 C6 : Person → Prop)

-- Conditions
axiom lead_car (h : C1 T)
axiom jamie_eden_direct (h : ∀ x, C2 x → C3 E)
axiom lynn_ahead_mira (h : ∀ x, C4 x → C5 M)
axiom mira_not_last (h : ∀ x, ¬C6 M)
axiom two_between_cory_lynn (h : ∃ x y, C1 x ∧ C4 L ∧ C5 y ∧ ¬(x = L) ∧ ∃ z, C4 z)

-- Fact to prove
theorem fourth_car_is_lynn : C4 L := sorry

end fourth_car_is_lynn_l725_725369


namespace points_concyclic_triangles_similar_l725_725224

noncomputable theory

open_locale euclidean_geometry

variables (A B C P A₁ B₁ C₁ : Point)
  (hABC : right_triangle A B C)
  (hPAB : line_p A B P)
  (hPA₁ : orth_proj P A A₁)
  (hPB₁ : orth_proj P B B₁)
  (hCC₁ : orth_proj C (line_through A B) C₁)
  (hRight_A₁PB₁C : right_angle A₁ P B₁ ∧ right_angle P B₁ C ∧ right_angle P A₁ C)

theorem points_concyclic : circle P A₁ C B₁ C₁ :=
sorry

theorem triangles_similar : similar (triangle A₁ B₁ C₁) (triangle A B C) :=
sorry

end points_concyclic_triangles_similar_l725_725224


namespace number_of_six_digit_palindromes_l725_725155

def is_six_digit_palindrome (n : ℕ) : Prop :=
  let d1 := n / 100000 % 10
  let d2 := n / 10000 % 10
  let d3 := n / 1000 % 10
  let d4 := n / 100 % 10
  let d5 := n / 10 % 10
  let d6 := n % 10
  n >= 100000 ∧ n < 1000000 ∧ d1 > 0 ∧ d1 = d6 ∧ d2 = d5 ∧ d3 = d4

theorem number_of_six_digit_palindromes : 
  {n : ℕ | is_six_digit_palindrome n}.card = 900 := 
sorry

end number_of_six_digit_palindromes_l725_725155


namespace kn_divides_am_ratio_l725_725635

theorem kn_divides_am_ratio (A B C K N M P : Point) (K_midpoint_AB: Midpoint K A B) (N_ratio_AN_NC: ratio N A C 2 1) (M_midpoint_BC: Midpoint M B C) (P_intersection_AM_KN: Intersection P (Segment A M) (Segment K N)) :
  ratio (Segment A P) (Segment P N) 4 3 :=
sorry

end kn_divides_am_ratio_l725_725635


namespace part1_part2_l725_725506

-- Define the function f(x) as given
def f (x : ℝ) (a : ℝ) : ℝ := (2 * x - 2) * Real.exp x - a * x^2 + 2 * a^2

-- Part 1: For a = 1, prove the solution set of f(x) > 0 is (0, +∞)
theorem part1 : ∀ x : ℝ, f x 1 > 0 ↔ 0 < x := by
  sorry

-- Part 2: For 0 < a < 1, prove that f(x) has exactly one root x₀, and ax₀ < 3/2
theorem part2 {a : ℝ} (ha : 0 < a ∧ a < 1) :
  ∃! x₀ : ℝ, f x₀ a = 0 ∧ a * x₀ < (3 : ℝ) / 2 :=
by
  sorry

end part1_part2_l725_725506


namespace function_neither_odd_nor_even_l725_725662

def f (x : ℝ) : ℝ := (1 / 3^(x - 1)) - 3

theorem function_neither_odd_nor_even : 
  ¬ (∀ x, f(-x) = f(x)) ∧ ¬ (∀ x, f(-x) = -f(x)) :=
by
  -- Initial steps to show neither even nor odd.
  sorry

end function_neither_odd_nor_even_l725_725662


namespace seating_arrangements_l725_725028

inductive Person : Type
| Alice | Bob | Carla | Derek | Eric | Fiona

open Person

def adjacent (p : List Person) (a b : Person) : Prop :=
  ∃ (i : ℕ), i < p.length - 1 ∧ (p.nth i = some a ∧ p.nth (i + 1) = some b ∨ p.nth i = some b ∧ p.nth (i + 1) = some a)

def validArrangements (p : List Person) : Prop :=
  ¬(adjacent p Alice Bob) ∧ ¬(adjacent p Alice Carla) ∧ ¬(adjacent p Derek Eric) ∧ ¬(adjacent p Derek Fiona)

theorem seating_arrangements : ∃ sits : Set (List Person), 
  sits.card = 360 ∧ ∀ p ∈ sits, validArrangements p :=
by
  sorry

end seating_arrangements_l725_725028


namespace log_graphs_log2_graph_log2_square_graph_log2_square_min_l725_725616

theorem log_graphs (x : ℝ) (hx : x ≠ 0) : 
  (y = log (x^2)) ↔ (y = 2 * log (abs x)) :=
by
sorry

theorem log2_graph (x : ℝ) (h0 : 0 < x) :
  ∀ x > 1, 0 < y → y = 2 * log x := 
by
sorry

theorem log2_square_graph (x : ℝ) (h0 : 0 < x) :
  ∃! y, y = (log x)^2 ∧ (∀ x, y ≥ 0) :=
by
sorry

theorem log2_square_min (x : ℝ) (h0 : 0 < x) :
  (∀ x > 0, (log x)^2 ≥ 0) ∧ ((log 1)^2 = 0) :=
by
sorry

end log_graphs_log2_graph_log2_square_graph_log2_square_min_l725_725616


namespace wrongly_written_height_l725_725652

-- Define the conditions
def avg_height {n : ℕ} (h : Fin n → ℝ) := (∑ i, h i) / n

axiom num_boys : ℕ
axiom original_avg_height : ℝ
axiom actual_avg_height : ℝ
axiom wrong_height : ℝ
axiom correct_height : ℝ

-- Given conditions
def conditions : Prop := 
  num_boys = 35 ∧ 
  original_avg_height = 185 ∧ 
  correct_height = 106 ∧
  actual_avg_height = 183

-- The theorem to prove the wrongly written height
theorem wrongly_written_height (H : conditions) : wrong_height = 176 :=
by sorry

end wrongly_written_height_l725_725652


namespace surface_area_of_T_is_630_l725_725581

noncomputable def s : ℕ := 582
noncomputable def t : ℕ := 42
noncomputable def u : ℕ := 6

theorem surface_area_of_T_is_630 : s + t + u = 630 :=
by
  sorry

end surface_area_of_T_is_630_l725_725581


namespace water_pump_rate_l725_725267

theorem water_pump_rate (hourly_rate : ℕ) (minutes : ℕ) (calculated_gallons : ℕ) : 
  hourly_rate = 600 → minutes = 30 → calculated_gallons = (hourly_rate * (minutes / 60)) → 
  calculated_gallons = 300 :=
by 
  sorry

end water_pump_rate_l725_725267


namespace perfect_cubes_l725_725822

theorem perfect_cubes (n : ℕ) (h : n > 0) : 
  (n = 7 ∨ n = 11 ∨ n = 12 ∨ n = 25) ↔ ∃ k : ℤ, (n^3 - 18*n^2 + 115*n - 391) = k^3 :=
by exact sorry

end perfect_cubes_l725_725822


namespace internship_arrangement_l725_725336

theorem internship_arrangement (n_students n_departments : ℕ) (h_students : n_students = 4) (h_departments : n_departments = 5)  :  
  (1/2 * (Nat.choose n_students 2) * Nat.choose 2 2 * (Nat.perm n_departments 2)) = 60 :=
by
  sorry

end internship_arrangement_l725_725336


namespace lcm_36_100_is_900_l725_725453

def prime_factors_36 : ℕ → Prop := 
  λ n, n = 36 → (2^2 * 3^2)

def prime_factors_100 : ℕ → Prop := 
  λ n, n = 100 → (2^2 * 5^2)

def lcm_36_100 := lcm 36 100

theorem lcm_36_100_is_900 : lcm_36_100 = 900 :=
by {
  sorry,
}

end lcm_36_100_is_900_l725_725453


namespace yuko_in_front_of_yuri_l725_725316

theorem yuko_in_front_of_yuri (a b c d e f : ℕ) (h₁ : a + b + c = 12) (h₂ : d + e + f < 12) :
  ∃ (y : ℕ), y = d + e + f ∧ y < a + b + c :=
by
  use d + e + f
  split
  · refl
  · exact h₂

end yuko_in_front_of_yuri_l725_725316


namespace find_k_l725_725146

-- Define the points A, B, and C
def A : ℝ × ℝ × ℝ := (-2, 0, 2)
def B : ℝ × ℝ × ℝ := (-1, 1, 2)
def C : ℝ × ℝ × ℝ := (-3, 0, 4)

-- Define the vectors a and b
def a : ℝ × ℝ × ℝ := (B.1 - A.1, B.2 - A.2, B.3 - A.3)
def b : ℝ × ℝ × ℝ := (C.1 - A.1, C.2 - A.2, C.3 - A.3)

-- Define the vector functions 
def k_a_plus_b (k : ℝ) : ℝ × ℝ × ℝ := (k * a.1 + b.1, k * a.2 + b.2, k * a.3 + b.3)
def k_a_minus_2b (k : ℝ) : ℝ × ℝ × ℝ := (k * a.1 - 2 * b.1, k * a.2 - 2 * b.2, k * a.3 - 2 * b.3)

-- Define the dot product
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Define the perpendicular condition
def is_perpendicular (k : ℝ) : Prop := dot_product (k_a_plus_b k) (k_a_minus_2b k) = 0

-- The statement to prove
theorem find_k : ∃ k : ℝ, is_perpendicular k ∧ (k = -5 / 2 ∨ k = 2) := by
  sorry

end find_k_l725_725146


namespace min_value_ineq_l725_725491

noncomputable theory

variables (a b : ℝ)

-- Conditions
def ab_pos : Prop := a * b > 0
def eq_sum : Prop := 2 * a + b = 5

-- Problem Statement
theorem min_value_ineq (ha : ab_pos a b) (hb : eq_sum a b) : 
  (inf { y : ℝ | ∃ (a b : ℝ), ab_pos a b ∧ eq_sum a b ∧ y = (2 / (a + 1) + 1 / (b + 1)) }) = 9 / 8 :=
sorry

end min_value_ineq_l725_725491


namespace greatest_mondays_in_45_days_l725_725706

-- Define the days in a week
def days_in_week : ℕ := 7

-- Define the total days being considered
def total_days : ℕ := 45

-- Calculate the complete weeks in the total days
def complete_weeks : ℕ := total_days / days_in_week

-- Calculate the extra days
def extra_days : ℕ := total_days % days_in_week

-- Define that the period starts on Monday (condition)
def starts_on_monday : Bool := true

-- Prove that the greatest number of Mondays in the first 45 days is 7
theorem greatest_mondays_in_45_days (h1 : days_in_week = 7) (h2 : total_days = 45) (h3 : starts_on_monday = true) : 
  (complete_weeks + if starts_on_monday && extra_days >= 1 then 1 else 0) = 7 := 
by
  sorry

end greatest_mondays_in_45_days_l725_725706


namespace proof_tan_x_proof_fraction_l725_725832

variable {x : ℝ}
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (Real.sin x, Real.cos x)
def parallel (a b : ℝ × ℝ) : Prop := ∃ k : ℝ, a = (k * b.1, k * b.2)

theorem proof_tan_x {x : ℝ} (h : parallel a b) : Real.tan x = 2 := sorry

theorem proof_fraction {x : ℝ} (h : parallel a b) (h_tan : Real.tan x = 2) : 
  (3 * Real.sin x - Real.cos x) / (Real.sin x + 3 * Real.cos x) = 1 := sorry

end proof_tan_x_proof_fraction_l725_725832


namespace part_I_part_IIa_part_IIb_part_IIc_part_III_l725_725864

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + x^2

theorem part_I (a : ℝ) (h : a = -4) : ∀ x : ℝ, x ∈ Ioi (Real.sqrt 2) → monotone_on (f a) (Ioi (Real.sqrt 2)) :=
by sorry

theorem part_IIa (a : ℝ) (h : a ≥ -2) : ∃ x : ℝ, x ∈ Icc 1 Real.exp ∧ min_on (f a) (Icc 1 Real.exp) f 1 1 :=
by sorry

theorem part_IIb (a : ℝ) (h1 : -2 * Real.exp ^ 2 < a) (h2 : a < -2) :
∃ x : ℝ, x ∈ Icc 1 Real.exp ∧ min_on (f a) (Icc 1 Real.exp) (f a) (-a/2 * Real.log (-a/2) - a/2) (Real.sqrt (-a/2)) :=
by sorry

theorem part_IIc (a : ℝ) (h : a ≤ -2 * Real.exp ^ 2) : 
∃ x : ℝ, x ∈ Icc 1 Real.exp ∧ min_on (f a) (Icc 1 Real.exp) f (a + Real.exp ^ 2) Real.exp :=
by sorry

theorem part_III (a : ℝ): 
(∃ x : ℝ, x ∈ Icc 1 Real.exp ∧ (f a x) ≤ (a + 2) * x) → a ∈ Icc -1 Real.Top :=
by sorry

end part_I_part_IIa_part_IIb_part_IIc_part_III_l725_725864


namespace smallest_a10_l725_725086

noncomputable def set_sigma (S : Finset ℤ) : ℤ := S.sum id

theorem smallest_a10:
  ∀ (A : Finset ℕ) (n : ℕ) 
  (hA : A = {1, 2, a₃, a₄, a₅, a₆, a₇, a₈, a₉, a₁₀, a₁₁}) 
  (h_sorted : (A : List _) = (A : List _).eraseDuplicatesSorted (<=))
  (hne : {n} ⊆ Finset.range 1501), 
  ∃ S ⊆ A, set_sigma S = n → a₁₀ = 248 := 
sorry

end smallest_a10_l725_725086


namespace DM_passes_through_midpoint_BC_EM_bisects__area_AEBM_l725_725325

variables (A B C D I E M : Type) [has_coords A ℝ] [has_coords B ℝ] [has_coords C ℝ] [has_coords D ℝ]
          [has_coords I ℝ] [has_coords E ℝ] [has_coords M ℝ]
variables [rectangle ABCD] (x : ℝ)

-- Given conditions as definitions
def midpoint_I : Prop := is_midpoint I C D
def BI_intersects_AC_at_M : Prop := intersects M BI AC
def BE_eq_BC : Prop := distance B E = distance B C
def BE_eq_x : Prop := distance B E = x
def AE_eq_BE : Prop := distance A E = distance B E
def ∠AEB_90 : Prop := angle A E B = 90

-- Prove that the line DM passes through the midpoint of BC
theorem DM_passes_through_midpoint_BC (h1 : rectangle ABCD) (h2 : midpoint_I I C D) (h3 : BI_intersects_AC_at_M M BI AC) :
  passes_through_midpoint DM BC := sorry

-- Prove that EM bisects angle AMB
theorem EM_bisects_∠AMB (h1 : BE_eq_x B E x) (h2 : BE_eq_x B C x) (h3 : AE_eq_BE A E B E) (h4 : ∠AEB_90 A E B) :
  bisects EM (angle A M B) := sorry

-- Prove that the area of AEBM is given by specific formula
theorem area_AEBM (h1 : BE_eq_x B E x) (h2 : BE_eq_x B C x) (h3 : AE_eq_BE A E B E) (h4 : ∠AEB_90 A E B) :
  area A E B M = x^2 * (3 + 2 * sqrt 2) / 6 := sorry

end DM_passes_through_midpoint_BC_EM_bisects__area_AEBM_l725_725325


namespace partition_X_into_3k_pairs_l725_725229

variable {n k : ℕ}
variable (X : Set (Fin n))
variable (A : Finset (Fin n × Fin n × Fin n))
variables (l : ℕ) [hl : l = n^2 / 6]
variable [H : ∀ j, 1 ≤ j ∧ j ≤ l → ∀ x y, x ∈ X ∧ y ∈ X ∧ x ≠ y → ∃ i ≤ l, Finset.insert (x,y) ∈ A]

theorem partition_X_into_3k_pairs 
  (hl_eq : l = n^2 / 6)
  (H_A : ∀ (j : ℕ), 1 ≤ j ∧ j ≤ l → ∀ x y : Fin n, (x ∈ X) ∧ (y ∈ X) ∧ (x ≠ y) → ∃ (i : ℕ), i ≤ l ∧ (x, y) ∈ (A : Finset (Fin n × Fin n × Fin n)))
  : ∃ P : Finset (Fin n × Fin n), P.card = 3 * k ∧ ∀ (p ∈ P), ∃ (j₁ j₂), ((x, y) ∈ (A j₁) ∧ (x, y) ∈ (A j₂)) ∧ ((x, y) ∉ ∪ A) := 
sorry

end partition_X_into_3k_pairs_l725_725229


namespace sum_of_digits_ABCED_l725_725674

theorem sum_of_digits_ABCED {A B C D E : ℕ} (hABCED : 3 * (10000 * A + 1000 * B + 100 * C + 10 * D + E) = 111111) :
  A + B + C + D + E = 20 := 
by
  sorry

end sum_of_digits_ABCED_l725_725674


namespace avg_of_five_consecutive_from_b_l725_725084

-- Conditions
def avg_of_five_even_consecutive (a : ℕ) : ℕ := (2 * a + (2 * a + 2) + (2 * a + 4) + (2 * a + 6) + (2 * a + 8)) / 5

-- The main theorem
theorem avg_of_five_consecutive_from_b (a : ℕ) : 
  avg_of_five_even_consecutive a = 2 * a + 4 → 
  ((2 * a + 4 + (2 * a + 4 + 1) + (2 * a + 4 + 2) + (2 * a + 4 + 3) + (2 * a + 4 + 4)) / 5) = 2 * a + 6 :=
by
  sorry

end avg_of_five_consecutive_from_b_l725_725084


namespace count_divisors_phi_factorial_add_one_l725_725797

-- Define \phi^{!}(n) as the product of all positive integers less than or equal to n and relatively prime to n
def phi_factorial (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (nat.coprime n).prod id

-- Define the condition that n divides phi_factorial(n) + 1
def divides_phi_factorial_add_one (n : ℕ) : Prop :=
  n ∣ (phi_factorial n + 1)

-- The main theorem statement
theorem count_divisors_phi_factorial_add_one :
  (finset.range 50).filter (λ n, 2 ≤ n ∧ divides_phi_factorial_add_one n).card = 30 :=
by
  sorry

end count_divisors_phi_factorial_add_one_l725_725797


namespace area_bounded_by_graphs_eq_4_l725_725414

theorem area_bounded_by_graphs_eq_4 :
  let r₁ (θ : ℝ) := 2 / (cos θ)
  let r₂ (θ : ℝ) := 2 / (sin θ)
  ∀ (θ₁ θ₂ : ℝ) (x ℝ: ℝ), 0 ≤ θ₁ ∧ θ₁ ≤ π/2 ∧ 0 ≤ θ₂ ∧ θ₂ ≤ π/2 ∧
  x = r₁ θ₁ ∧ y = r₂ θ₂ →
  area (bounded_region r₁ r₂ 0 0) = 4 := by
  sorry

end area_bounded_by_graphs_eq_4_l725_725414


namespace visiting_sequences_count_l725_725023

theorem visiting_sequences_count :
  (∃ (arr : List Char), arr.length = 4 ∧ arr.head = 'A' ∧
    ((arr.contains 'C' → arr.last ≠ 'C') ∧
    (arr.to_set = {'A', 'B', 'C', 'D', 'E'} \ 'E' ∨ 
     arr.to_set = {'A', 'B', 'C', 'D', 'E'} \ 'D' ∨ 
     arr.to_set = {'A', 'B', 'C', 'D', 'E'} \ 'B' ∨ 
     arr.to_set = {'A', 'B', 'D', 'E'}))) → 
  (List.countP (λ arr : List Char, arr.length = 4 ∧ arr.head = 'A' ∧
    ((arr.contains 'C' → arr.last ≠ 'C') ∧
    (arr.to_set = {'A', 'B', 'C', 'D', 'E'} \ 'E' ∨ 
     arr.to_set = {'A', 'B', 'C', 'D', 'E'} \ 'D' ∨ 
     arr.to_set = {'A', 'B', 'C', 'D', 'E'} \ 'B' ∨ 
     arr.to_set = {'A', 'B', 'D', 'E'}))) List.permutations ('A'::'B'::'C'::'D'::'E'::[])) = 18 :=
begin
  sorry
end

end visiting_sequences_count_l725_725023


namespace school_population_proof_l725_725206

variables (x y z: ℕ)
variable (B: ℕ := (50 * y) / 100)

theorem school_population_proof (h1 : 162 = (x * B) / 100)
                               (h2 : B = (50 * y) / 100)
                               (h3 : z = 100 - 50) :
  z = 50 :=
  sorry

end school_population_proof_l725_725206


namespace integral_result_l725_725407

noncomputable def f (x : ℝ) : ℝ := real.sqrt (1 - x^2) + x

theorem integral_result : 
  ∫ x in 0..1, f x = (Real.pi / 4) + (1 / 2) := 
by
  sorry

end integral_result_l725_725407


namespace triangle_ratios_equal_l725_725115

theorem triangle_ratios_equal 
  (A B C P Q R A' B' C' : Type)
  [LineSegment AB A B]
  [LineSegment BC B C]
  [LineSegment CA C A]
  [PointOnSegment P AB]
  [PointOnSegment Q BC]
  [PointOnSegment R CA]
  [PointOnSegment A' RP]
  [PointOnSegment B' PQ]
  [PointOnSegment C' QR]
  (h1 : Parallel AB A'B')
  (h2 : Parallel BC B'C')
  (h3 : Parallel CA C'A')
  (area_ABC : ℝ)
  (area_PQR : ℝ)
  (area_A'B'C' : ℝ) :
  (AB / A'B') = (area_PQR / area_A'B'C') := sorry

end triangle_ratios_equal_l725_725115


namespace proof_n_perp_β_l725_725834

-- Assume the existence and basic properties of lines and planes, including parallelism and perpendicularity.
variables {L : Type*} {P : Type*} [Line L] [Plane P]

-- Variables for lines m, n and planes α, β
variables (m n : L) (α β : P)

-- Given conditions
variables (non_coincident_lines : m ≠ n)
variables (non_coincident_planes : α ≠ β)
variables (m_perp_α : m ⊥ α) (m_parallel_n : m ∥ n)
variables (α_parallel_β : α ∥ β)

-- Proof statement
theorem proof_n_perp_β : n ⊥ β :=
sorry

end proof_n_perp_β_l725_725834


namespace ellipse_sum_a_k_l725_725033

theorem ellipse_sum_a_k :
  let foci1 := (2, 2) in
  let foci2 := (2, 6) in
  let passing_point := (-3, 4) in
  let h := 2 in
  let k := 4 in
  let a := (Real.sqrt 29 + 5) / 2 in
  (a + k) = (Real.sqrt 29 + 13) / 2 := 
  by 
    sorry

end ellipse_sum_a_k_l725_725033


namespace collinear_imp_m_eq_l725_725499

variable (m : ℝ)

def AB : (ℝ × ℝ) := (7, 6)
def BC : (ℝ × ℝ) := (-3, m)
def AD : (ℝ × ℝ) := (-1, 2 * m)

def AC : (ℝ × ℝ) := (AB.1 + BC.1, AB.2 + BC.2)

theorem collinear_imp_m_eq : (AD.1 / AC.1 = AD.2 / AC.2) → m = -2 / 3 := by
  sorry

end collinear_imp_m_eq_l725_725499


namespace no_root_of_equation_l725_725037

theorem no_root_of_equation (x : ℝ) (h : x - 9 / (x - 5) = 5 - 9 / (x - 5)) : false :=
begin
  sorry
end

end no_root_of_equation_l725_725037


namespace least_overlap_coffee_tea_l725_725990

open BigOperators

-- Define the percentages in a way that's compatible in Lean
def percentage (n : ℕ) := n / 100

noncomputable def C := percentage 75
noncomputable def T := percentage 80
noncomputable def B := percentage 55

-- The theorem statement
theorem least_overlap_coffee_tea : C + T - 1 = B := sorry

end least_overlap_coffee_tea_l725_725990


namespace sum_in_range_l725_725793

theorem sum_in_range : 
    let a := (2:ℝ) + 1/8
    let b := (3:ℝ) + 1/3
    let c := (5:ℝ) + 1/18
    10.5 < a + b + c ∧ a + b + c < 11 := 
by 
    sorry

end sum_in_range_l725_725793


namespace calculate_b_c_range_of_a_harmonic_inequality_l725_725511

-- Definition for the function f(x)
def f (x a b c : ℝ) : ℝ := a * x + b / x + c

-- Requirement 1: Calculate b and c given the tangent line at (1, f(1)) is y = x - 1
theorem calculate_b_c (a b c : ℝ) (h1 : 0 < a) 
  (h2 : f 1 a b c = 1) 
  (h3 : ∀ x, (deriv (λ x, f x a b c)) 1 = 1) :
  b = a - 1 ∧ c = 1 - 2a := by
  sorry
  
-- Requirement 2: Given that \( f(x) \geq \ln x \) for \( x \in [1, +\infty) \)
theorem range_of_a (a : ℝ) (h1 : 0 < a) 
  (h2 : ∀ x ≥ 1, f x a (a - 1) (1 - 2 * a) ≥ Real.log x) :
  a ≥ 0.5 := by
  sorry
  
-- Requirement 3: Prove inequality for harmonic series
theorem harmonic_inequality (n : ℕ) (h : n ≥ 1) :
  (∑ i in Finset.range (n + 1), 1 / (i + 1)) > Real.log (n + 1) + n / (2 * (n + 1)) := by
  sorry

end calculate_b_c_range_of_a_harmonic_inequality_l725_725511


namespace find_BN_l725_725903

theorem find_BN
  (A B C N : Type)
  [InnerProductSpace ℝ N]
  (AB AC BC : ℝ)
  (h1 : AB = 13)
  (h2 : AC = 13)
  (h3 : BC = 12)
  (hN_midpoint : ∃ M : N, M ∈ line [A, C])
  : distance B N = 7 * real.sqrt 2 :=
by
  sorry

end find_BN_l725_725903


namespace term_11_is_4sqrt2_l725_725108

def sequence_term (n : ℕ) : ℝ := real.sqrt (3 * n - 1)

theorem term_11_is_4sqrt2 :
  sequence_term 11 = 4 * real.sqrt 2 :=
sorry

end term_11_is_4sqrt2_l725_725108


namespace nonagon_diagonals_l725_725749

theorem nonagon_diagonals (n : ℕ) (hn : n = 9) (right_angles : 2) : 
  ∃ d, d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  use (9 * (9 - 3)) / 2
  split
  . -- Proof that the formula applies
    calc 
      (n * (n - 3)) / 2 = (9 * (9 - 3)) / 2 : by rw [hn]
                     ... = (9 * 6) / 2       : by norm_num
                     ... = 54 / 2            : by norm_num
                     ... = 27                : by norm_num
  . -- Proof that the result is 27
    refl

end nonagon_diagonals_l725_725749


namespace pebbles_partition_l725_725288

open Finset

def color (x : Fin 4n) : Fin n := sorry -- define the coloring function

theorem pebbles_partition {n : ℕ} (h : 0 < n) :
  (∃ (A B : Finset (Fin (4 * n))),
    (A ∪ B = univ) ∧
    (A ∩ B = ∅) ∧
    (A.card = 2 * n) ∧
    (B.card = 2 * n) ∧
    (∑ x in A, ↑x = ∑ x in B, ↑x) ∧
    (∀ i : Fin n, (A.filter (λ x, color x = i)).card = 2 ∧ (B.filter (λ x, color x = i)).card = 2)) :=
sorry

end pebbles_partition_l725_725288


namespace number_of_six_digit_palindromes_l725_725158

def is_six_digit_palindrome (n : ℕ) : Prop :=
  let d1 := n / 100000 % 10
  let d2 := n / 10000 % 10
  let d3 := n / 1000 % 10
  let d4 := n / 100 % 10
  let d5 := n / 10 % 10
  let d6 := n % 10
  n >= 100000 ∧ n < 1000000 ∧ d1 > 0 ∧ d1 = d6 ∧ d2 = d5 ∧ d3 = d4

theorem number_of_six_digit_palindromes : 
  {n : ℕ | is_six_digit_palindrome n}.card = 900 := 
sorry

end number_of_six_digit_palindromes_l725_725158


namespace amicable_pairs_at_least_l725_725577

noncomputable theory
open_locale classical

-- Definitions for amicable pair and binomial coefficient
def amicable (G : SimpleGraph V) (x y : V) : Prop :=
∃ z : V, G.Adj x z ∧ G.Adj y z

def binomial (n k : ℕ) : ℕ := nat.choose n k

-- The main statement
theorem amicable_pairs_at_least
  (n : ℕ)
  (h_even : n % 2 = 0)
  (G : SimpleGraph (fin n))
  (h_edges : G.edge_count = n^2 / 4) :
  ∃ (m : ℕ), m ≥ 2 * binomial (n / 2) 2 ∧ 
  ∀ (x y : fin n), (x ≠ y) → amicable G x y :=
sorry

end amicable_pairs_at_least_l725_725577


namespace quadrilateral_is_parallelogram_l725_725575

theorem quadrilateral_is_parallelogram
  (P O1 O2 O3 O4 : Point)
  (convex_quad : ConvexQuadrilateral O1 O2 O3 O4)
  (l1 l2 l3 l4 : Line)
  (passes_through_P : ∀ l, l1 = l ∨ l2 = l ∨ l3 = l ∨ l4 = l → l.passes_through P)
  (meets_rays : ∀ i l,
    (i = 1 → (∃ Ai Bi, Ai.distinct_from_point O1 ∧ Bi.distinct_from_point O3 ∧ l1.meets (Ray O1 O0) Ai ∧ l1.meets (Ray O1 O2) Bi)) ∧
    (i = 2 → (∃ Ai Bi, Ai.distinct_from_point O2 ∧ Bi.distinct_from_point O4 ∧ l2.meets (Ray O2 O1) Ai ∧ l2.meets (Ray O2 O3) Bi)) ∧
    (i = 3 → (∃ Ai Bi, Ai.distinct_from_point O3 ∧ Bi.distinct_from_point O1 ∧ l3.meets (Ray O3 O2) Ai ∧ l3.meets (Ray O3 O4) Bi)) ∧
    (i = 4 → (∃ Ai Bi, Ai.distinct_from_point O4 ∧ Bi.distinct_from_point O2 ∧ l4.meets (Ray O4 O3) Ai ∧ l4.meets (Ray O4 O1) Bi)))
  (fi_minimizes : ∀ i, l = li ↔ fi(l) ≤ fi l')
  (l1_eq_l3 : l1 = l3)
  (l2_eq_l4 : l2 = l4) :
  is_parallelogram O1 O2 O3 O4 :=
by
  sorry

end quadrilateral_is_parallelogram_l725_725575


namespace percentage_error_calculation_l725_725762

theorem percentage_error_calculation (x : ℝ) :
  let correct_value := x * (5 / 3)
  let incorrect_value := x * (3 / 5)
  let difference := correct_value - incorrect_value
  let percentage_error := (difference / correct_value) * 100
  percentage_error = 64 := 
by
  let correct_value := x * (5 / 3)
  let incorrect_value := x * (3 / 5)
  let difference := correct_value - incorrect_value
  let percentage_error := (difference / correct_value) * 100
  sorry

end percentage_error_calculation_l725_725762


namespace max_min_difference_l725_725948

theorem max_min_difference (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) :
  let Z := (|x^2 - y^2|) / (|x^2| + |y^2|) in 
  (max (Z) - min (Z) = 1) :=
sorry

end max_min_difference_l725_725948


namespace exposed_surface_area_l725_725287

theorem exposed_surface_area (r h : ℝ) (π : ℝ) (sphere_surface_area : ℝ) (cylinder_lateral_surface_area : ℝ) 
  (cond1 : r = 10) (cond2 : h = 5) (cond3 : sphere_surface_area = 4 * π * r^2) 
  (cond4 : cylinder_lateral_surface_area = 2 * π * r * h) :
  let hemisphere_curved_surface_area := sphere_surface_area / 2
  let hemisphere_base_area := π * r^2
  let total_surface_area := hemisphere_curved_surface_area + hemisphere_base_area + cylinder_lateral_surface_area
  total_surface_area = 400 * π :=
by
  sorry

end exposed_surface_area_l725_725287


namespace sum_of_digits_of_d_l725_725612

noncomputable def exchange_rate (d : ℝ) : ℝ := (4 / 3) * d

theorem sum_of_digits_of_d (d : ℝ) (h1 : exchange_rate d - 96 = d) :
  d = 288 ∧ (2 + 8 + 8 = 18) := 
by 
  -- First, solve for d from the given condition
  have h2 : (4 / 3) * d - d = 96, from h1,
  have h3 : ((4 / 3) - 1) * d = 96,
  { rw sub_eq_add_neg, convert sub_eq_add_neg (4 / 3) (1 : ℝ) },
  have h4 : d / 3 = 96, from (by linarith : (1 / 3) * d = 96),
  have d_val : d = 288, from (by linarith : d = 288),
  use d_val,
  -- Calculate the sum of the digits of d
  have sum_digits_d : 2 + 8 + 8 = 18, by linarith,
  exact ⟨d_val, sum_digits_d⟩

end sum_of_digits_of_d_l725_725612


namespace minimum_occupied_seats_l725_725289

theorem minimum_occupied_seats (total_seats : ℕ) (min_empty_seats : ℕ) (occupied_seats : ℕ)
  (h1 : total_seats = 150)
  (h2 : min_empty_seats = 2)
  (h3 : occupied_seats = 2 * (total_seats / (occupied_seats + min_empty_seats + min_empty_seats)))
  : occupied_seats = 74 := by
  sorry

end minimum_occupied_seats_l725_725289


namespace tan_add_l725_725833

theorem tan_add (α β : ℝ) (h1 : Real.tan (α - π / 6) = 3 / 7) (h2 : Real.tan (π / 6 + β) = 2 / 5) : Real.tan (α + β) = 1 := by
  sorry

end tan_add_l725_725833


namespace min_t_value_l725_725778

theorem min_t_value (M : Point) (A B C D : Point) (AB CD : Line)
  (AB_eq_1 : AB.length = 1)
  (BC_eq_2 : BC.length = 2)
  (M_in_rect : M ∈ interior ABCD) :
  ∃ t_min : ℝ, t_min = 2 ∧ ∀ t : ℝ, t = (AM.length * MC.length + BM.length * MD.length) → t ≥ t_min :=
sorry

end min_t_value_l725_725778


namespace inscribed_circle_radius_of_rhombus_l725_725709

theorem inscribed_circle_radius_of_rhombus (d1 d2 : ℝ) (a r : ℝ) : 
  d1 = 15 → d2 = 24 → a = Real.sqrt ((15 / 2)^2 + (24 / 2)^2) → 
  (d1 * d2) / 2 = 2 * a * r → 
  r = 60.07 / 13 :=
by
  intros h1 h2 h3 h4
  sorry

end inscribed_circle_radius_of_rhombus_l725_725709


namespace rounding_to_nearest_tenth_l725_725253

theorem rounding_to_nearest_tenth : 
  (let num := 3967149.1847234 in
   let first_decimal := (num * 10) % 10 in
   let second_decimal := (num * 100) % 10 in
   if second_decimal >= 5 then (num + (10 - first_decimal) / 10) else (num - first_decimal / 10)) = 3967149.2 := 
sorry

end rounding_to_nearest_tenth_l725_725253


namespace prove_nabla_squared_l725_725680

theorem prove_nabla_squared:
  ∃ (odot nabla : ℕ), odot < 20 ∧ nabla < 20 ∧ odot ≠ nabla ∧
  (nabla * nabla * odot = nabla) ∧ (nabla * nabla = 64) :=
by
  sorry

end prove_nabla_squared_l725_725680


namespace train_time_to_pass_platform_l725_725370

-- Definitions as per the conditions
def length_of_train : ℕ := 720 -- Length of train in meters
def speed_of_train_kmh : ℕ := 72 -- Speed of train in km/hr
def length_of_platform : ℕ := 280 -- Length of platform in meters

-- Conversion factor and utility functions
def kmh_to_ms (speed : ℕ) : ℕ :=
  speed * 1000 / 3600

def total_distance (train_len platform_len : ℕ) : ℕ :=
  train_len + platform_len

def time_to_pass (distance speed_ms : ℕ) : ℕ :=
  distance / speed_ms

-- Main statement to be proven
theorem train_time_to_pass_platform :
  time_to_pass (total_distance length_of_train length_of_platform) (kmh_to_ms speed_of_train_kmh) = 50 :=
by
  sorry

end train_time_to_pass_platform_l725_725370


namespace problem1_problem2_problem3_l725_725595

-- Definitions for conditions
def setA (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 5
def setB (m x : ℝ) : Prop := m + 1 ≤ x ∧ x ≤ 2 * m - 1

-- Problem 1: if B ⊆ A, then the range of m is m ≤ 3
theorem problem1 (m : ℝ) (subset_condition : ∀ x, setB m x → setA x) : m ≤ 3 := 
sorry

-- Definitions for a specific condition in Problem 2
def integer_setA : set ℤ := {-2, -1, 0, 1, 2, 3, 4, 5}

-- Problem 2: the number of non-empty proper subsets of A is 254
theorem problem2 : (2 ^ 8 - 2) = 254 := 
by norm_num

-- Problem 3: if A ∩ B = ∅, then the range of m is m < 2 or m > 4
theorem problem3 (m : ℝ) (disjoint_condition : ∀ x, setA x ∧ setB m x → False) : m < 2 ∨ m > 4 := 
sorry

end problem1_problem2_problem3_l725_725595


namespace eq_cont_fracs_l725_725984

noncomputable def cont_frac : Nat -> Rat
| 0       => 0
| (n + 1) => (n : Rat) + 1 / (cont_frac n)

theorem eq_cont_fracs (n : Nat) : 
  1 - cont_frac n = cont_frac n - 1 :=
sorry

end eq_cont_fracs_l725_725984


namespace evaluate_g_5_times_l725_725592

def g (x : ℕ) : ℕ :=
if x % 2 = 0 then x + 2 else 3 * x + 1

theorem evaluate_g_5_times : g (g (g (g (g 1)))) = 12 := by
  sorry


end evaluate_g_5_times_l725_725592


namespace area_of_bounded_region_l725_725425

theorem area_of_bounded_region :
  let region := {p : ℝ × ℝ | (p.1 = 0 ∧ p.2 ≥ 0) ∨ (p.2 = 0 ∧ p.1 ≥ 0) ∨ (p.1 = 2) ∨ (p.2 = 2)}
  ∃ (s : set (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ s ↔ (y = x))
    ∧ is_square s 
    ∧ area s = 4 := 
sorry

end area_of_bounded_region_l725_725425


namespace functional_eq_f800_l725_725226

theorem functional_eq_f800
  (f : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) = f x / y)
  (h2 : f 1000 = 6)
  : f 800 = 7.5 := by
  sorry

end functional_eq_f800_l725_725226


namespace fraction_value_condition_l725_725475

theorem fraction_value_condition (m n : ℚ) (h : m / n = 2 / 3) : m / (m + n) = 2 / 5 :=
sorry

end fraction_value_condition_l725_725475


namespace fraction_value_l725_725477

theorem fraction_value
  (m n : ℕ)
  (h : m / n = 2 / 3) :
  m / (m + n) = 2 / 5 :=
sorry

end fraction_value_l725_725477


namespace probability_at_least_6_heads_in_10_flips_l725_725345

theorem probability_at_least_6_heads_in_10_flips : 
  let total_outcomes := 1024 in 
  let favorable_outcomes := 15 in 
  (favorable_outcomes / total_outcomes : ℚ) = 15 / 1024 :=
by
  sorry

end probability_at_least_6_heads_in_10_flips_l725_725345


namespace correct_option_e_l725_725313

theorem correct_option_e : 15618 = 1 + 5^6 - 1 * 8 :=
by sorry

end correct_option_e_l725_725313


namespace part1_part2_l725_725945

-- Definitions
def is_root (α : ℝ) := α^2 - α - 1 = 0
def seq (α β n : ℕ) := (α^n.to_real - β^n.to_real) / (α.to_real - β.to_real)

-- Part 1
theorem part1 (α β : ℝ) (hα : is_root α) (hβ : is_root β) :
  ∀ n : ℕ, seq α β (n + 2) = seq α β (n + 1) + seq α β n :=
sorry

-- Part 2
theorem part2 :
  ∃ a b : ℕ, a < b ∧ ∀ n : ℕ, b ∣ seq ϕ ψ n - 2 * n * a^n ∧ a = 3 ∧ b = 5 :=
sorry

end part1_part2_l725_725945


namespace find_k_max_profit_l725_725009

-- Definitions based on the given conditions
def cost (x : ℝ) : ℝ := 3 + x

def revenue (x : ℝ) (k : ℝ) : ℝ :=
  if 0 < x ∧ x < 6 then
    3 * x + k / (x - 8) + 5
  else if x >= 6 then
    14
  else
    0  -- This case should not occur based on the problem statement

def profit (x : ℝ) (k : ℝ) : ℝ :=
  if 0 < x ∧ x < 6 then
    2 * x + k / (x - 8) + 2
  else if x >= 6 then
    11 - x
  else
    0  -- This case should not occur based on the problem statement

-- Condition given in the problem
axiom profit_condition : ∀ (k : ℝ), profit 2 k = 3

-- Problem (1): Find the value of k
theorem find_k : ∃ k : ℝ, profit 2 k = 3 := by
  use 18
  sorry

-- Problem (2): Find the daily production quantity that maximizes daily profit and determine the maximum value
theorem max_profit : ∃ (x_max : ℝ) (L_max : ℝ), 
  (∀ (x : ℝ), profit x 18 ≤ L_max) ∧ L_max = profit x_max 18 := by
  use 5, 6
  sorry

end find_k_max_profit_l725_725009


namespace weight_calculation_l725_725307

def molar_mass_Na : ℝ := 22.99
def molar_mass_Ca : ℝ := 40.08
def molar_mass_C : ℝ := 12.01
def molar_mass_O : ℝ := 16.00

def molar_mass_Na2CaCO32 : ℝ :=
  2 * molar_mass_Na + molar_mass_Ca + 2 * molar_mass_C + 6 * molar_mass_O

def moles_Na2CaCO32 : ℝ := 3.75

def weight_Na2CaCO32 : ℝ :=
  moles_Na2CaCO32 * molar_mass_Na2CaCO32

theorem weight_calculation :
  weight_Na2CaCO32 = 772.8 := by
  -- Definitions of molar masses
  have h1 : molar_mass_Na = 22.99 := rfl
  have h2 : molar_mass_Ca = 40.08 := rfl
  have h3 : molar_mass_C = 12.01 := rfl
  have h4 : molar_mass_O = 16.00 := rfl

  -- Calculation of molar mass of Na2Ca(CO3)2
  have molar_mass_calc : molar_mass_Na2CaCO32 =
    2 * molar_mass_Na + molar_mass_Ca + 2 * molar_mass_C + 6 * molar_mass_O := rfl
  
  -- Substitute the values of molar masses
  calc
    weight_Na2CaCO32
      = 3.75 * molar_mass_Na2CaCO32 : rfl
  ... = 3.75 * (2 * 22.99 + 40.08 + 2 * 12.01 + 6 * 16.00) : by rw [molar_mass_calc, h1, h2, h3, h4]
  ... = 772.8 : by norm_num

-- Ensuring the Lean statement can be synthesized correctly
#print axioms weight_calculation

end weight_calculation_l725_725307


namespace intersection_of_A_and_B_l725_725881

def A := {x : ℝ | ∃ y : ℝ, y = Real.log (x - 1) / Real.log 2}
def B := {x : ℝ | x < 2}

theorem intersection_of_A_and_B : (A ∩ B) = {x : ℝ | 1 < x ∧ x < 2} := by
  sorry

end intersection_of_A_and_B_l725_725881


namespace discuss_monotonicity_inequality_proof_l725_725133

section Part1
variable {a : ℝ} (h : a ≠ 0)
def f (x : ℝ) : ℝ := a * x * exp x
theorem discuss_monotonicity : 
  (∀ x, f' x = a * (x+1) * exp x) →
  (a > 0 → ∀ x, (x < -1 → f' x < 0) ∧ (x > -1 → f' x > 0)) ∧
  (a < 0 → ∀ x, (x < -1 → f' x > 0) ∧ (x > -1 → f' x < 0)) :=
sorry
end Part1

section Part2
variable {a : ℝ} (h : a ≠ 0) (h₁ : a ≥ (4 / (exp 2)))
def f (x : ℝ) : ℝ := a * x * exp x
theorem inequality_proof (x : ℝ) (hx : x > 0) : 
  (f x / (x + 1)) - ((x + 1) * log x) > 0 :=
sorry
end Part2

end discuss_monotonicity_inequality_proof_l725_725133


namespace set_A_cannot_form_right_triangle_l725_725030

-- Definitions for the side lengths
def set_A : ℕ × ℕ × ℕ := (5, 7, 10)
def set_B : ℕ × ℕ × ℕ := (3, 4, 5)
def set_C : ℕ × ℕ × ℕ := (1, 3, 2)
def set_D : ℕ × ℕ × ℕ := (7, 24, 25)

-- Define the Pythagorean theorem check
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

-- Define the triangle inequality theorem check
def satisfies_triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the condition for forming a right triangle
def can_form_right_triangle (a b c : ℕ) : Prop :=
  is_right_triangle a b c ∧ satisfies_triangle_inequality a b c

-- Prove that set_A cannot form a right triangle
theorem set_A_cannot_form_right_triangle : ¬ can_form_right_triangle (set_A.1) (set_A.2) (set_A.3) :=
  begin
    sorry
  end

end set_A_cannot_form_right_triangle_l725_725030


namespace frank_bought_5_chocolate_bars_l725_725090

theorem frank_bought_5_chocolate_bars :
  ∃ (x : ℕ), (2 * x + 2 * 3 = 16) ∧ x = 5 :=
by
  use 5
  split
  sorry
  rfl

end frank_bought_5_chocolate_bars_l725_725090


namespace number_of_six_digit_palindromes_l725_725162

theorem number_of_six_digit_palindromes : 
  let count_palindromes : ℕ := 9 * 10 * 10 in
  count_palindromes = 900 :=
by
  sorry

end number_of_six_digit_palindromes_l725_725162


namespace overtime_pay_correct_l725_725381

theorem overtime_pay_correct
  (overlap_slow : ℝ := 69) -- Slow clock minute-hand overlap in minutes
  (overlap_normal : ℝ := 12 * 60 / 11) -- Normal clock minute-hand overlap in minutes
  (hours_worked : ℝ := 8) -- The normal working hours a worker believes working
  (hourly_wage : ℝ := 4) -- The normal hourly wage
  (overtime_rate : ℝ := 1.5) -- Overtime pay rate
  (expected_overtime_pay : ℝ := 2.60) -- The expected overtime pay
  
  : hours_worked * (overlap_slow / overlap_normal) * hourly_wage * (overtime_rate - 1) = expected_overtime_pay :=
by
  sorry

end overtime_pay_correct_l725_725381


namespace area_of_JKLM_l725_725646

theorem area_of_JKLM 
  (A B C D J K L M : Point)
  (hABCD: square A B C D)
  (hJKLM: square J K L M)
  (hInside: inside J K L M A B C D)
  (side_ABCD: dist A B = 10)
  (dist_AJ: dist A J = 2) :
  area_square J K L M = 204 - 40 * Real.sqrt 2 :=
  sorry -- proof is omitted

end area_of_JKLM_l725_725646


namespace probability_range_l725_725184

theorem probability_range (p : ℝ) (h1 : 0 ≤ p) (h2 : p ≤ 1)
  (h3 : (4 * p * (1 - p)^3) ≤ (6 * p^2 * (1 - p)^2)) : 
  2 / 5 ≤ p ∧ p ≤ 1 :=
by {
  sorry
}

end probability_range_l725_725184


namespace roots_are_distinct_and_irrational_l725_725051

noncomputable def quadratic_equation_nature_of_roots (b : ℝ) (h: (b: ℝ)) : Prop :=
  let a := 3
  let d := (-6 * real.sqrt 3)
  let discriminant := d^2 - 4 * a * b
  discriminant = 12 ∧ ∀ x: ℝ, 3 * x^2 - 6 * x * real.sqrt 3 + b = 0

theorem roots_are_distinct_and_irrational (b : ℝ) (h₁ : 3 * has_sqrt.sqrt 3 * 3 = 9 * 3) 
  (h₂ : (108 - 4 * 3 * b = 12) ∨ (108 - 12 * b = 12)) :  quadratic_equation_nature_of_roots b h₁ h₂ :=
sorry

end roots_are_distinct_and_irrational_l725_725051


namespace categorize_numbers_correctly_l725_725072

-- Definitions for each set of numbers given
def given_numbers : Set ℚ :=
  {200 / 100, -3 / 4, 0, -9, 1.98, 4 / 15, 0.89, 102, -3 / 2, 3 / 20, real.pi, -100 / 25}

-- Convert percentages and decimals to rational numbers
notation "200%" := 2
notation "15%" := 3 / 20

-- Constructing expected sets
def rationalNumbers : Set ℚ := {2, -3 / 4, 0, -9, 1.98, 4 / 15, 89 / 100, 102, -3 / 2, 3 / 20, -4}

def negativeIntegers : Set ℚ := {-9, -4}

def positiveFractions : Set ℚ := {1.98, 4 / 15, 89 / 100, 3 / 20}

def negativeRationalNumbers : Set ℚ := {-3 / 4, -9, -3 / 2, -4}

theorem categorize_numbers_correctly :
  (∃ r, r = rationalNumbers) ∧
  (∃ ni, ni = negativeIntegers) ∧
  (∃ pf, pf = positiveFractions) ∧
  (∃ nrn, nrn = negativeRationalNumbers) :=
by
  sorry

end categorize_numbers_correctly_l725_725072


namespace steve_speed_back_l725_725610

open Real

noncomputable def steves_speed_on_way_back : ℝ := 15

theorem steve_speed_back
  (distance_to_work : ℝ)
  (traffic_time_to_work : ℝ)
  (traffic_time_back : ℝ)
  (total_time : ℝ)
  (speed_ratio : ℝ) :
  distance_to_work = 30 →
  traffic_time_to_work = 30 →
  traffic_time_back = 15 →
  total_time = 405 →
  speed_ratio = 2 →
  steves_speed_on_way_back = 15 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end steve_speed_back_l725_725610


namespace solve_trig_equation_l725_725261

open Real

theorem solve_trig_equation (x : ℝ) (n : ℤ) :
  (2 * tan (6 * x) ^ 4 + 4 * sin (4 * x) * sin (8 * x) - cos (8 * x) - cos (16 * x) + 2) / sqrt (cos x - sqrt 3 * sin x) = 0 
  ∧ cos x - sqrt 3 * sin x > 0 →
  ∃ (k : ℤ), x = 2 * π * k ∨ x = -π / 6 + 2 * π * k ∨ x = -π / 3 + 2 * π * k ∨ x = -π / 2 + 2 * π * k ∨ x = -2 * π / 3 + 2 * π * k :=
sorry

end solve_trig_equation_l725_725261


namespace general_formula_a_n_sum_T_n_l725_725486

-- Definitions used in the conditions
def a : ℕ → ℝ
def S : ℕ → ℝ

-- -- Conditions: a_3 - 2a_2 = 0 and S_3 = 7
axiom cond_a3_minus_2a2 : a 3 - 2 * a 2 = 0
axiom cond_S3_eq_7 : S 3 = 7

-- -- The general formula for the sequence {a_n}
theorem general_formula_a_n : ∀ n : ℕ, a n = 2^(n-1) :=
sorry

-- -- Definition and sum of the first n terms of the sequence {n/a_n}
def b (n : ℕ) := n / (a n)
def T (n : ℕ) := ∑ i in Finset.range n, b (i + 1)

-- -- The sum of the first n terms for the sequence {n / a_n}
theorem sum_T_n (n : ℕ) : T n = 4 - (n + 2) / 2^(n-1) :=
sorry

end general_formula_a_n_sum_T_n_l725_725486


namespace expand_and_simplify_l725_725409

theorem expand_and_simplify (x : ℝ) : (x^2 + 4) * (x - 5) = x^3 - 5 * x^2 + 4 * x - 20 := 
sorry

end expand_and_simplify_l725_725409


namespace example_counterexample_l725_725808

theorem example_counterexample : ∃ (a b c : ℤ), a = -1 ∧ b = -2 ∧ c = -3 ∧ a > b ∧ b > c ∧ ¬ (a + b > c) :=
by
  use -1
  use -2
  use -3
  simp
  sorry

end example_counterexample_l725_725808


namespace speed_of_man_in_still_water_l725_725358

theorem speed_of_man_in_still_water (v_m v_s : ℝ) (h1 : v_m + v_s = 18) (h2 : v_m - v_s = 13) : v_m = 15.5 :=
by {
  -- Proof is not required as per the instructions
  sorry
}

end speed_of_man_in_still_water_l725_725358


namespace find_n_l725_725524

theorem find_n (n : ℤ) (h : 13^(3 * n) = (1 / 13)^(n - 24)) : n = 6 :=
by
  sorry

end find_n_l725_725524


namespace remainder_7_pow_253_mod_12_l725_725710

theorem remainder_7_pow_253_mod_12 : (7 ^ 253) % 12 = 7 := by
  sorry

end remainder_7_pow_253_mod_12_l725_725710


namespace length_of_larger_sheet_l725_725656

theorem length_of_larger_sheet : 
  ∃ L : ℝ, 2 * (L * 11) = 2 * (5.5 * 11) + 100 ∧ L = 10 :=
by
  sorry

end length_of_larger_sheet_l725_725656


namespace lcm_36_100_l725_725456

theorem lcm_36_100 : Nat.lcm 36 100 = 900 :=
by
  sorry

end lcm_36_100_l725_725456


namespace binomial_expansion_l725_725523

theorem binomial_expansion (a : ℕ → ℝ) (x : ℝ) :
  (2 * x - 1) ^ 2013 = ∑ k in Finset.range 2014, a k * x ^ k →
  (1 / 2 + ∑ k in Finset.Ico 2 2014, a k / (2 ^ k * a 1) = 1 / 4026) :=
begin
  sorry
end

end binomial_expansion_l725_725523


namespace trajectory_equation_necessary_not_sufficient_l725_725679

theorem trajectory_equation_necessary_not_sufficient :
  ∀ (x y : ℝ), (|x| = |y|) → (y = |x|) ↔ (necessary_not_sufficient) :=
by
  sorry

end trajectory_equation_necessary_not_sufficient_l725_725679


namespace solution_set_l725_725841

variable {f : ℝ → ℝ}
variable {f' : ℝ → ℝ}
variable (h_deriv : ∀ x, deriv f x = f' x)
variable (h_domain : ∀ x, x ∈ ℝ)
variable (h_odd : ∀ x : ℝ, f (x - 1) = -f (-x + 1))
variable (h_condition : ∀ x < -1, (x + 1) * (f x + (x + 1) * f' x) < 0)

theorem solution_set :
  { x : ℝ | x * f (x - 1) > f 0 } = { x : ℝ | -1 < x ∧ x < 1 } :=
sorry

end solution_set_l725_725841


namespace min_value_of_f_exists_x_such_that_f_ge_33_l725_725839

def f (a x : ℝ) : ℝ := 4^x - 2 * 2^(x + 1) + a

-- Part 1
theorem min_value_of_f (a : ℝ) (h : ∀ x ∈ Icc 0 3, f a x ≥ 1) : a = 5 := sorry

-- Part 2
theorem exists_x_such_that_f_ge_33 (a : ℝ) (hx : ∃ x ∈ Icc 0 3, f a x ≥ 33) : a ≥ 1 := sorry

end min_value_of_f_exists_x_such_that_f_ge_33_l725_725839


namespace max_real_roots_l725_725463

noncomputable theory

/-- A function that defines the polynomial in question. -/
def polynomial (n : ℕ) (c : ℝ) : ℝ[X] :=
  ∑ i in (finset.range (n + 1)), polynomial.X ^ i + polynomial.C c

/-- Statement of the problem. -/
theorem max_real_roots (n : ℕ) (c : ℝ) (h_n_pos : 0 < n) (h_c_nonzero : c ≠ 0) :
  let p := polynomial n c in
  (if (n % 2 = 1) ∧ (c = 2) then
    ∃ x : ℝ, is_root p x
  else
    ¬ ∃ x : ℝ, is_root p x) :=
by
  let p := polynomial n c
  sorry

end max_real_roots_l725_725463


namespace total_stickers_used_l725_725065

-- Define all the conditions as given in the problem
def initially_water_bottles : ℕ := 20
def lost_at_school : ℕ := 5
def found_at_park : ℕ := 3
def stolen_at_dance : ℕ := 4
def misplaced_at_library : ℕ := 2
def acquired_from_friend : ℕ := 6
def stickers_per_bottle_school : ℕ := 4
def stickers_per_bottle_dance : ℕ := 3
def stickers_per_bottle_library : ℕ := 2

-- Prove the total number of stickers used
theorem total_stickers_used : 
  (lost_at_school * stickers_per_bottle_school)
  + (stolen_at_dance * stickers_per_bottle_dance)
  + (misplaced_at_library * stickers_per_bottle_library)
  = 36 := 
by
  sorry

end total_stickers_used_l725_725065


namespace probability_of_6_consecutive_heads_l725_725353

/-- Define the probability of obtaining at least 6 consecutive heads in 10 flips of a fair coin. -/
def prob_at_least_6_consecutive_heads : ℚ :=
  129 / 1024

/-- Proof statement: The probability of getting at least 6 consecutive heads in 10 flips of a fair coin is 129/1024. -/
theorem probability_of_6_consecutive_heads : 
  prob_at_least_6_consecutive_heads = 129 / 1024 := 
by
  sorry

end probability_of_6_consecutive_heads_l725_725353


namespace evelyn_family_tv_hours_l725_725408

theorem evelyn_family_tv_hours
    (watched_week_before : ℕ)
    (watched_next_week : ℕ)
    (average : ℕ)
    (x : ℕ)
    (h1 : watched_week_before = 8)
    (h2 : watched_next_week = 12)
    (h3 : average = 10) :
    (watched_week_before + x + watched_next_week) / 3 = average → x = 10 :=
by 
  intros h
  have h_eq : 8 + x + 12 = 30 := by 
    calc
      watched_week_before + x + watched_next_week = 8 + x + 12 := 
        by rw [h1, h2]
      ... = 30 := by linarith
        
  have h_sol : x + 20 = 30 := by 
    rw [add_assoc, add_comm 8, ← h_eq]
      
  have final_result : x = 10 := by 
    linarith

  exact final_result

end evelyn_family_tv_hours_l725_725408


namespace min_value_f_n_l725_725582

theorem min_value_f_n : 
  (∀ (p q : ℕ), 0 < p → 0 < q → a₁ = 2 ∧ a (p+q) = a p + a q ∧ 
  let S_n : ℕ → ℕ := λ n, 2 * n + (n * (n - 1)) / 2 * 2 in 
  f n = S_n n + 60 / (n + 1) → 
  min_value f = 29 / 2 :=
sorry

end min_value_f_n_l725_725582


namespace prove_ordered_pair_l725_725811

-- Definition of the problem
def satisfies_equation1 (x y : ℚ) : Prop :=
  3 * x - 4 * y = -7

def satisfies_equation2 (x y : ℚ) : Prop :=
  7 * x - 3 * y = 5

-- Definition of the correct answer
def correct_answer (x y : ℚ) : Prop :=
  x = -133 / 57 ∧ y = 64 / 19

-- Main theorem to prove
theorem prove_ordered_pair :
  correct_answer (-133 / 57) (64 / 19) :=
by
  unfold correct_answer
  constructor
  { sorry }
  { sorry }

end prove_ordered_pair_l725_725811


namespace harry_did_not_get_an_A_l725_725403

theorem harry_did_not_get_an_A
  (emily_Imp_frank : Prop)
  (frank_Imp_gina : Prop)
  (gina_Imp_harry : Prop)
  (exactly_one_did_not_get_an_A : ¬ (emily_Imp_frank ∧ frank_Imp_gina ∧ gina_Imp_harry)) :
  ¬ harry_Imp_gina :=
  sorry

end harry_did_not_get_an_A_l725_725403


namespace johnson_class_more_students_l725_725603

theorem johnson_class_more_students
  (finley_class_students : ℕ)
  (johnson_class_students : ℕ)
  (h_finley : finley_class_students = 24)
  (h_johnson : johnson_class_students = 22) :
  johnson_class_students - finley_class_students / 2 = 10 :=
  sorry

end johnson_class_more_students_l725_725603


namespace ball_stops_at_vertex_l725_725332

open Nat

def parity (n : ℕ) : Prop := if n % 2 = 0 then True else False

theorem ball_stops_at_vertex (m n : ℕ) : 
  ((¬ (parity m) ∧ ¬ (parity n)) → ((m, n) : ℕ × ℕ)) ∧
  ((parity m ∧ ¬ (parity n)) → ((m, 0) : ℕ × ℕ)) ∧
  ((¬ (parity m) ∧ parity n) → ((0, n) : ℕ × ℕ)) := 
sorry

end ball_stops_at_vertex_l725_725332


namespace right_triangle_area_l725_725213

theorem right_triangle_area (a b c : ℝ) (h1 : a + b = 21) (h2 : c = 15) (h3 : a^2 + b^2 = c^2):
  (1/2) * a * b = 54 :=
by
  sorry

end right_triangle_area_l725_725213


namespace probability_log_a_b_integer_l725_725701

theorem probability_log_a_b_integer :
  let S := finset.image (λ n : ℕ, 3^n) (finset.range 13)
  let valid_pairs := S.filter (λ a b : ℕ, log a b ∈ ℤ)
  (valid_pairs.card : ℚ) / (S.card.choose 2 : ℚ) = 2 / 13 := 
sorry

end probability_log_a_b_integer_l725_725701


namespace stacked_height_probability_is_50ft_l725_725070

noncomputable def probability_of_exact_stack_height : ℚ :=
  let totalWays := 3^15
  let combinations := (choose 15 7) * (choose 8 2) * (choose 6 6) +
                      (choose 15 5) * (choose 10 5) * (choose 5 5) +
                      (choose 15 3) * (choose 12 8) * (choose 4 4)
  let prob := (1 /  totalWays) * combinations
  by
    have h_totalWays := nat.pow 3 15
    have h_combinations := (choose 15 7) * (choose 8 2) * (choose 6 6) +
                           (choose 15 5) * (choose 10 5) * (choose 5 5) +
                           (choose 15 3) * (choose 12 8) * (choose 4 4)
    have h_prob := h_combinations / h_totalWays
    exact h_prob

theorem stacked_height_probability_is_50ft :
  probability_of_exact_stack_height = 1162161 / 14348907 :=
by
  sorry

end stacked_height_probability_is_50ft_l725_725070


namespace locus_of_P_is_single_ray_l725_725843
  
noncomputable def M : ℝ × ℝ := (1, 0)
noncomputable def N : ℝ × ℝ := (3, 0)

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2))

theorem locus_of_P_is_single_ray (P : ℝ × ℝ) (h : distance P M - distance P N = 2) : 
∃ α : ℝ, P = (3 + α * (P.1 - 3), α * P.2) :=
sorry

end locus_of_P_is_single_ray_l725_725843


namespace simplify_expression_l725_725981

theorem simplify_expression (x : ℝ) : 
  x * (x * (x * (3 - x) - 6) + 7) + 2 = -x^4 + 3 * x^3 - 6 * x^2 + 7 * x + 2 := 
by 
  sorry

end simplify_expression_l725_725981


namespace sum_of_squares_eq_product_l725_725941

/-- statement: Given a set A = {1, 2, ..., n} with n > 5, 
prove that there exists a finite set B of positive integers such that A ⊆ B 
and ∑ (x ∈ B) x^2 = ∏ (x ∈ B) x -/
theorem sum_of_squares_eq_product (n : ℕ) (h : n > 5) : 
  ∃ (B : Finset ℕ), {1, 2, ..., n} ⊆ B ∧ (∑ x in B, x^2) = (∏ x in B, x) := 
sorry

end sum_of_squares_eq_product_l725_725941


namespace time_after_12345_seconds_is_13_45_45_l725_725931

def seconds_in_a_minute := 60
def minutes_in_an_hour := 60
def initial_hour := 10
def initial_minute := 45
def initial_second := 0
def total_seconds := 12345

def time_after_seconds (hour minute second : Nat) (elapsed_seconds : Nat) : (Nat × Nat × Nat) :=
  let total_initial_seconds := hour * 3600 + minute * 60 + second
  let total_final_seconds := total_initial_seconds + elapsed_seconds
  let final_hour := total_final_seconds / 3600
  let remaining_seconds_after_hour := total_final_seconds % 3600
  let final_minute := remaining_seconds_after_hour / 60
  let final_second := remaining_seconds_after_hour % 60
  (final_hour, final_minute, final_second)

theorem time_after_12345_seconds_is_13_45_45 :
  time_after_seconds initial_hour initial_minute initial_second total_seconds = (13, 45, 45) :=
by
  sorry

end time_after_12345_seconds_is_13_45_45_l725_725931


namespace probability_of_selecting_point_between_C_and_D_l725_725617

-- Definitions and theorems
def points_on_line (A B C D : Type) := true
def length_eq (AB AD BC : ℝ) : Prop := AB = 4 * AD ∧ AB = 5 * BC
def prob_between_C_and_D (AB AD CD : ℝ) : ℝ := CD / AB

-- Main theorem
theorem probability_of_selecting_point_between_C_and_D (A B C D : Type)
  (AB AD BC : ℝ) (h : length_eq AB AD BC) :
  prob_between_C_and_D AB AD (AD * ((CD : ℝ) / AD)) = 11 / 20 :=
by
  sorry

end probability_of_selecting_point_between_C_and_D_l725_725617


namespace max_a_value_l725_725123

noncomputable def f (x : ℝ) : ℝ :=
if h : 0 ≤ x ∧ x ≤ 6 then Real.log 6 (x + 1)
else f (12 - x)

theorem max_a_value : 
  (∀ x : ℝ, f x = f (-x)) → 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 6 → f x = f (12 - x)) → 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 6 → f x = Real.log 6 (x + 1)) → 
  (∃ a : ℝ, 0 ≤ a ∧ a ≤ 2020 ∧ f a = 1) → 
  ∃ a : ℝ, a = 2011 ∧ f a = 1 := 
sorry

end max_a_value_l725_725123


namespace tomatoes_left_l725_725685

theorem tomatoes_left (initial_tomatoes : ℕ) (birds : ℕ) (fraction : ℕ) (E1 : initial_tomatoes = 21) 
  (E2 : birds = 2) (E3 : fraction = 3) : 
  initial_tomatoes - initial_tomatoes / fraction = 14 :=
by 
  sorry

end tomatoes_left_l725_725685


namespace functions_equivalence_problem_solution_l725_725772

theorem functions_equivalence :
  (∀ x : ℝ, x = (∛x^3)) :=
by sorry

-- Notation for cube root
def ∛ (x : ℝ) : ℝ := x ^ (1 / 3)

-- Definition for the group of functions C
def group_C_equivalence : Prop :=
  ∀ x : ℝ, 
    (x = ∛(x^3))

-- Main theorem to prove the problem statement
theorem problem_solution (hC : group_C_equivalence) :
  -- Among groups of functions, only group C represents the same function
  hC :=
by sorry

end functions_equivalence_problem_solution_l725_725772


namespace four_gt_sqrt_fifteen_l725_725788

theorem four_gt_sqrt_fifteen : 4 > Real.sqrt 15 := 
sorry

end four_gt_sqrt_fifteen_l725_725788


namespace find_a_l725_725503

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log (2 : ℝ) (-x^2 + a * x + 3)

theorem find_a (a : ℝ) (hf : f 1 a = 2) : a = 2 := 
sorry

end find_a_l725_725503


namespace arithmetic_sequence_cubed_sum_l725_725790

theorem arithmetic_sequence_cubed_sum (x : ℤ) (n : ℕ) (hxodd : x % 2 = 1) (hn_gt_4 : n > 4) :
  (∑ i in Finset.range (n + 1), (x + 2 * i) ^ 3) = -1331 → n = 5 := 
sorry

end arithmetic_sequence_cubed_sum_l725_725790


namespace michelle_sandwiches_l725_725966

def sandwiches_left (total : ℕ) (given_to_coworker : ℕ) (kept : ℕ) : ℕ :=
  total - given_to_coworker - kept

theorem michelle_sandwiches : sandwiches_left 20 4 (4 * 2) = 8 :=
by
  sorry

end michelle_sandwiches_l725_725966


namespace find_k_l725_725724

theorem find_k (k : ℕ) : (1/2)^18 * (1/81)^k = (1/18)^18 → k = 9 :=
by
  intro h
  sorry

end find_k_l725_725724


namespace tan_theta_parallel_l725_725892

theorem tan_theta_parallel (θ : ℝ) : 
  let a := (2, 3)
  let b := (Real.cos θ, Real.sin θ)
  (b.1 * a.2 = b.2 * a.1) → Real.tan θ = 3 / 2 :=
by
  intros h
  sorry

end tan_theta_parallel_l725_725892


namespace value_of_f_at_2_l725_725227

def f (x : ℤ) : ℤ := x^3 - x

theorem value_of_f_at_2 : f 2 = 6 := by
  sorry

end value_of_f_at_2_l725_725227


namespace original_price_double_value_l725_725179

theorem original_price_double_value :
  ∃ (P : ℝ), P + 0.30 * P = 351 ∧ 2 * P = 540 :=
by
  sorry

end original_price_double_value_l725_725179


namespace ratio_BE_EC_l725_725182

-- Define points A, B, C, D, E, F, G such that the conditions are met.
variables (A B C D E F G : Type) [is_point A] [is_point B] [is_point C]
          [is_point D] [is_point E] [is_point F] [is_point G]

-- Conditions:
def F_divides_AB_3_1 (A B F : Type) : Prop :=
  divides (3, 1) A B F -- Point F divides AB in the ratio 3:1

def D_mid_AC (A C D : Type) : Prop :=
  midpoint A C D -- Point D is the midpoint of AC

def G_mid_DF (D F G : Type) : Prop :=
  midpoint D F G -- Point G is the midpoint of DF

def E_intersects_BC_DG (B C D E G : Type) : Prop :=
  intersection B C D G E -- Point E is the intersection of BC and DG

-- Statement to prove:
theorem ratio_BE_EC (A B C D E F G : Type) [is_point A] [is_point B] [is_point C]
  [is_point D] [is_point E] [is_point F] [is_point G] :
  F_divides_AB_3_1 A B F →
  D_mid_AC A C D →
  G_mid_DF D F G →
  E_intersects_BC_DG B C D E G →
  divides (1, 1) B C E :=
by sorry -- Proof to be completed

end ratio_BE_EC_l725_725182


namespace perimeter_even_l725_725770

theorem perimeter_even 
  (n : ℕ) (V : Fin n → ℤ × ℤ) 
  (S : Fin n → Fin n → ℤ) 
  (all_integer_length : ∀ i : Fin n, ∃ k : ℤ, S i (V (i+1) - V i) = k) :
  ∃ k : ℕ, ∑ i in Finset.range n, ∥ V ((i + 1) % n) - V i ∥ = 2 * k :=
by
  sorry

end perimeter_even_l725_725770


namespace value_of_expression_l725_725124

theorem value_of_expression (a b : ℝ) (h1 : a = Real.floor (Real.sqrt 5))
                            (h2 : b = Real.sqrt 5 - Real.floor (Real.sqrt 5)) :
  a - 2 * b + Real.sqrt 5 = 6 - Real.sqrt 5 :=
sorry

end value_of_expression_l725_725124


namespace table_size_condition_l725_725191

-- Define the problem in Lean 4
theorem table_size_condition (n : ℕ) (a : ℕ → ℕ → ℝ) (_ : ∀ i j, 0 ≤ a i j) 
  (H1 : ∀ i, ∃ j, 0 < a i j) (H2 : ∀ j, ∃ i, 0 < a i j)
  (H3 : ∀ i j, 0 < a i j → (∑ k, a i k) = (∑ k, a k j)) : n = 2015 :=
sorry

end table_size_condition_l725_725191


namespace sum_of_products_l725_725096

theorem sum_of_products (n : ℕ) : (∑ i in Finset.range n, i) = n * (n - 1) / 2 :=
by
  sorry

end sum_of_products_l725_725096


namespace trader_profit_l725_725317

def original_price (P : ℝ) :=
  P

def bought_price (P : ℝ) :=
  0.80 * P

def sold_price (P : ℝ) :=
  1.24 * P

def profit (P : ℝ) :=
  sold_price P - original_price P

def profit_percentage (P : ℝ) :=
  (profit P / original_price P) * 100

theorem trader_profit (P : ℝ) (h : P > 0) : profit_percentage P = 24 := by
  unfold profit_percentage
  unfold profit
  unfold sold_price
  unfold original_price
  ring_nf
  sorry

end trader_profit_l725_725317


namespace line_intersects_circle_l725_725059

theorem line_intersects_circle : 
  ∀ (x y : ℝ), 
  (2 * x + y = 0) ∧ (x^2 + y^2 + 2 * x - 4 * y - 4 = 0) ↔
    ∃ (x0 y0 : ℝ), (2 * x0 + y0 = 0) ∧ ((x0 + 1)^2 + (y0 - 2)^2 = 9) :=
by
  sorry

end line_intersects_circle_l725_725059


namespace find_ellipse_foci_l725_725057

def ellipse_foci (θ : ℝ) : ℝ × ℝ :=
  (3 + 3 * Real.cos θ, -1 + 5 * Real.sin θ)

theorem find_ellipse_foci :
  ∃ h k c: ℝ, h = 3 ∧ k = -1 ∧ c = 4 ∧
  (ellipse_foci 3, ellipse_foci (-1)) = ((3, 3), (3, -5)) := 
by
  sorry

end find_ellipse_foci_l725_725057


namespace radius_OA_formula_l725_725715

variable (A B C O : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace O]
variable (α β : Real)
variable (a : Real)

-- Conditions
variable (AC : A → C)  -- AC is a segment between points A and C
variable (tangent_to_BC_at_A : Prop) -- AC is tangent to BC at A
variable (O_center_of_inscribe : O → Type)  -- O is the center of the inscribed circle
variable (OA_radius : O → A → Real)  -- OA is the radius of the circle
variable (angle_ACO_half_ACB : Real) -- Given angle ACO = 1/2 * angle ACB
variable (ABC_sine_theorem : Real) -- Sine theorem

-- Conclusion to prove
theorem radius_OA_formula :
  tangent_to_BC_at_A →
  O_center_of_inscribe O →
  ∀ (A C : Type) [MetricSpace A] [MetricSpace C],
  ∃ (OA : Real), 
    OA = (a * sin β * cot ((α + β) / 2)) / (sin α) :=
by
  sorry

end radius_OA_formula_l725_725715


namespace correct_operation_B_incorrect_operation_A_incorrect_operation_C_incorrect_operation_D_l725_725712

theorem correct_operation_B (a : ℝ) : a^3 / a = a^2 := 
by sorry

theorem incorrect_operation_A (a : ℝ) : a^2 + a^5 ≠ a^7 := 
by sorry

theorem incorrect_operation_C (a : ℝ) : (3 * a^2)^2 ≠ 6 * a^4 := 
by sorry

theorem incorrect_operation_D (a b : ℝ) : (a - b)^2 ≠ a^2 - b^2 := 
by sorry

end correct_operation_B_incorrect_operation_A_incorrect_operation_C_incorrect_operation_D_l725_725712


namespace coefficient_x3y3_in_expansion_l725_725301

theorem coefficient_x3y3_in_expansion (x y : ℕ) : 
  (∑ k in finset.range 7, (nat.choose 6 k) * x^(6-k) * y^k) = 20 :=
by sorry

end coefficient_x3y3_in_expansion_l725_725301


namespace an_arithmetic_sequence_sum_reciprocal_arithmetic_sum_reciprocal_general_arithmetic_sequence_l725_725596

-- Equivalent proof problem (1)
theorem an_arithmetic_sequence : 
  ∀ (n : ℕ) (h : n > 0), 
  let a_n := 2 * n - 1 in 
  (a_n = 2 * n - 1 ∧ ∀ (m : ℕ), (m > 0) → (a_(m+1) - a_m = 2)) :=
by
  sorry

-- Equivalent proof problem (2)
theorem sum_reciprocal_arithmetic : 
  ∀ (m k p : ℕ) (hm : m > 0) (hk : k > 0) (hp : p > 0) (h : m + p = 2 * k),
  (let S := λ x : ℕ, x^2 in (1 / S m + 1 / S p ≥ 2 / S k)) :=
by
  sorry

-- Equivalent proof problem (3)
theorem sum_reciprocal_general_arithmetic_sequence : 
  ∀ (m k p : ℕ) (hm : m > 0) (hk : k > 0) (hp : p > 0) (h : m + p = 2 * k)
      (a_1 : ℝ) (d : ℝ) (hpos : ∀ n : ℕ, n > 0 → a_1 + (n - 1) * d > 0),
  (let S := λ n : ℕ, (n * (2 * a_1 + (n - 1) * d)) / 2 in 
  1 / S m + 1 / S p ≥ 2 / S k) :=
by
  sorry

end an_arithmetic_sequence_sum_reciprocal_arithmetic_sum_reciprocal_general_arithmetic_sequence_l725_725596


namespace number_of_books_Mary_has_l725_725218

variable (Jason_books : Nat) (total_books : Nat)

theorem number_of_books_Mary_has : Jason_books = 18 → total_books = 60 → (total_books - Jason_books) = 42 :=
by
  intros h1 h2
  rw [h1, h2]
  exact rfl

end number_of_books_Mary_has_l725_725218


namespace length_PC_l725_725205

noncomputable def length_PA : ℝ := 8
noncomputable def length_PB : ℝ := 5
noncomputable def angle_APB : ℝ := 120
noncomputable def angle_BPC : ℝ := 120
noncomputable def angle_CPA : ℝ := 120

theorem length_PC 
  (a b c : ℝ)  -- lengths of sides of triangle ABC
  (right_triangle_ABC : a^2 + b^2 = c^2) -- ABC is right triangle with right angle at B
  (PA : ℝ := 8) (PB : ℝ := 5) (APB : ℝ := 120) (BPC : ℝ := 120) (CPA : ℝ := 120) 
  (in_triangle_condition : ∃ P, P inside triangle ABC ∧ PA = 8 ∧ PB = 5 
   ∧ angle APB = 120 ∧ angle BPC = 120 ∧ angle CPA = 120) :
  PC = 12.17 := 
sorry

end length_PC_l725_725205


namespace significant_improvement_l725_725739

-- Definition of experiment data
def experiment_data (x y : Fin 10 → ℝ) : Prop :=
  x = ![545, 533, 551, 522, 575, 544, 541, 568, 596, 548] ∧
  y = ![536, 527, 543, 530, 560, 533, 522, 550, 576, 536]

-- Definition of z_i
def z (x y : Fin 10 → ℝ) (i : Fin 10) : ℝ := x i - y i

-- Sample mean of z
def mean_z (z : Fin 10 → ℝ) : ℝ := (1 / 10) * ∑ i, z i

-- Sample variance of z
def variance_z (z : Fin 10 → ℝ) : ℝ := (1 / 10) * ∑ i, (z i - mean_z z)^2

-- Proof problem to check significant improvement
theorem significant_improvement (x y : Fin 10 → ℝ)
  (h_data : experiment_data x y) :
  mean_z (z x y) ≥ 2 * (Real.sqrt (variance_z (z x y) / 10)) :=
by
  sorry

end significant_improvement_l725_725739


namespace asha_remaining_money_l725_725044

-- Define the borrowed amounts, gift, and savings
def borrowed_from_brother : ℤ := 20
def borrowed_from_father : ℤ := 40
def borrowed_from_mother : ℤ := 30
def gift_from_granny : ℤ := 70
def savings : ℤ := 100

-- Total amount of money Asha has
def total_amount : ℤ := borrowed_from_brother + borrowed_from_father + borrowed_from_mother + gift_from_granny + savings

-- Amount spent by Asha
def amount_spent : ℤ := (3 * total_amount) / 4

-- Amount of money Asha remains with
def amount_left : ℤ := total_amount - amount_spent

-- The proof statement
theorem asha_remaining_money : amount_left = 65 := by
  sorry

end asha_remaining_money_l725_725044


namespace bread_remaining_after_five_days_l725_725599

-- Defining the initial conditions and the procedure of consumption
def initial_bread : ℕ := 1500
def consume_day1 (n : ℕ) : ℚ := 3/8 * n
def consume_day2 (n : ℚ) : ℚ := 7/10 * n
def consume_day3 (n : ℚ) : ℚ := 1/6 * n
def consume_day4 (n : ℚ) : ℚ := 4/9 * n
def consume_day5 (n : ℚ) : ℚ := 5/18 * n

-- Defining how much bread is left each day
def remaining_day1 : ℚ := initial_bread - consume_day1 initial_bread
def remaining_day2 : ℚ := remaining_day1 - consume_day2 remaining_day1
def remaining_day3 : ℚ := remaining_day2 - consume_day3 remaining_day2
def remaining_day4 : ℚ := remaining_day3 - consume_day4 remaining_day3
def remaining_day5 : ℚ := remaining_day4 - consume_day5 remaining_day4

-- The theorem we aim to prove
theorem bread_remaining_after_five_days : remaining_day5.to_nat = 94 :=
by
  sorry

end bread_remaining_after_five_days_l725_725599


namespace line_through_intersection_points_l725_725147

noncomputable def circle1 := { p : ℝ × ℝ | p.1^2 + p.2^2 = 10 }
noncomputable def circle2 := { p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 3)^2 = 10 }

theorem line_through_intersection_points (p : ℝ × ℝ) (hp1 : p ∈ circle1) (hp2 : p ∈ circle2) :
  p.1 + 3 * p.2 - 5 = 0 :=
sorry

end line_through_intersection_points_l725_725147


namespace triangle_is_obtuse_l725_725905

theorem triangle_is_obtuse (A B C : ℝ) (h_sum : A + B + C = 180)
  (h_tan : 0 < tan A * tan B ∧ tan A * tan B < 1) :
  A < 90 ∧ B < 90 ∧ C > 90 :=
sorry

end triangle_is_obtuse_l725_725905


namespace trapezoid_PQRS_perimeter_l725_725052

noncomputable def trapezoid_perimeter (PQ RS : ℝ) (height : ℝ) (PS QR : ℝ) : ℝ :=
  PQ + RS + PS + QR

theorem trapezoid_PQRS_perimeter :
  ∀ (PQ RS : ℝ) (height : ℝ)
  (PS QR : ℝ),
  PQ = 6 →
  RS = 10 →
  height = 5 →
  PS = Real.sqrt (5^2 + 4^2) →
  QR = Real.sqrt (5^2 + 4^2) →
  trapezoid_perimeter PQ RS height PS QR = 16 + 2 * Real.sqrt 41 :=
by
  intros
  sorry

end trapezoid_PQRS_perimeter_l725_725052


namespace unique_triple_solution_l725_725055

theorem unique_triple_solution (a b c : ℝ) 
  (h1 : a * (b ^ 2 + c) = c * (c + a * b))
  (h2 : b * (c ^ 2 + a) = a * (a + b * c))
  (h3 : c * (a ^ 2 + b) = b * (b + c * a)) : 
  a = b ∧ b = c := 
sorry

end unique_triple_solution_l725_725055


namespace fraction_of_area_l725_725633

noncomputable section

open Real

-- Definitions of points A, B, C, X, Y, and Z with their given coordinates
def A := (2, 0) : ℝ × ℝ
def B := (8, 12) : ℝ × ℝ
def C := (14, 0) : ℝ × ℝ

def X := (6, 0) : ℝ × ℝ
def Y := (8, 4) : ℝ × ℝ
def Z := (10, 0) : ℝ × ℝ

-- Definition of the area of a triangle given vertices
def area (p₁ p₂ p₃ : ℝ × ℝ) : ℝ := 
  abs ((p₂.1 - p₁.1) * (p₃.2 - p₁.2) - (p₂.2 - p₁.2) * (p₃.1 - p₁.1)) / 2

-- Areas of triangles ABC and XYZ
def area_ABC := area A B C
def area_XYZ := area X Y Z

-- The Lean statement
theorem fraction_of_area : (area_XYZ / area_ABC) = 1 / 9 := by
  sorry

end fraction_of_area_l725_725633


namespace distance_between_ports_l725_725019

theorem distance_between_ports
  (ship_speed : ℝ) (current_speed : ℝ) (time_difference : ℝ) (upstream_speed : ℝ) (downstream_speed : ℝ) 
  (D : ℝ) (h_ship_speed : ship_speed = 24) (h_current_speed : current_speed = 3) (h_time_difference : time_difference = 5)
  (h_upstream_speed : upstream_speed = ship_speed - current_speed) (h_downstream_speed : downstream_speed = ship_speed + current_speed)
  (h_time_difference_equation : D / upstream_speed - D / downstream_speed = time_difference) :
  D = 350 :=
by {
  rw [h_ship_speed, h_current_speed] at *,
  have h1 : upstream_speed = 21, by simp [h_upstream_speed, h_ship_speed, h_current_speed],
  have h2 : downstream_speed = 30, by simp [h_downstream_speed, h_ship_speed, h_current_speed],
  rw [h1, h2] at h_time_difference_equation,
  linarith [h_time_difference_equation],
}

end distance_between_ports_l725_725019


namespace carbon_atoms_in_compound_l725_725741

theorem carbon_atoms_in_compound
    (h_atoms : ℕ)
    (o_atoms : ℕ)
    (total_weight : ℕ)
    (carbon_weight : ℕ)
    (hydrogen_weight : ℕ)
    (oxygen_weight : ℕ)
    (h_atoms = 6) 
    (o_atoms = 2) 
    (total_weight = 122) 
    (carbon_weight = 12) 
    (hydrogen_weight = 1)
    (oxygen_weight = 16) : 
    ((total_weight - (h_atoms * hydrogen_weight + o_atoms * oxygen_weight)) / carbon_weight) = 7 := 
by
    sorry

end carbon_atoms_in_compound_l725_725741


namespace at_least_6_heads_in_10_flips_l725_725340

def coin_flip : Type := bool

def is_heads (x : coin_flip) : Prop := x = tt

def num_consecutive_heads (l : list coin_flip) (n : ℕ) : Prop :=
  ∃ i : ℕ, i + n ≤ l.length ∧ l.drop i.take n = list.replicate n tt

def prob_at_least_n_consecutive_heads (l : list coin_flip) (n : ℕ) : Prop :=
  ∃ i ≤ l.length - n + 1, list.replicate n tt = l.drop (i - 1).take n

noncomputable def at_least_6_heads_in_10_flips_prob : ℚ :=
  (129:ℚ) / (1024:ℚ)

theorem at_least_6_heads_in_10_flips :
  prob_at_least_n_consecutive_heads (list.replicate 10 coin_flip) 6 = at_least_6_heads_in_10_flips_prob :=
by
  sorry

end at_least_6_heads_in_10_flips_l725_725340


namespace fraction_of_area_l725_725632

noncomputable section

open Real

-- Definitions of points A, B, C, X, Y, and Z with their given coordinates
def A := (2, 0) : ℝ × ℝ
def B := (8, 12) : ℝ × ℝ
def C := (14, 0) : ℝ × ℝ

def X := (6, 0) : ℝ × ℝ
def Y := (8, 4) : ℝ × ℝ
def Z := (10, 0) : ℝ × ℝ

-- Definition of the area of a triangle given vertices
def area (p₁ p₂ p₃ : ℝ × ℝ) : ℝ := 
  abs ((p₂.1 - p₁.1) * (p₃.2 - p₁.2) - (p₂.2 - p₁.2) * (p₃.1 - p₁.1)) / 2

-- Areas of triangles ABC and XYZ
def area_ABC := area A B C
def area_XYZ := area X Y Z

-- The Lean statement
theorem fraction_of_area : (area_XYZ / area_ABC) = 1 / 9 := by
  sorry

end fraction_of_area_l725_725632


namespace semicircle_arc_length_l725_725270

theorem semicircle_arc_length (A B C O D E : Point) (AC : Line)
  (h_right : right_angle A B C)
  (h_hypotenuse : segment A O + segment O C = 30 + 40) :
  arc_length D E = 12 * Real.pi :=
sorry

end semicircle_arc_length_l725_725270


namespace total_savings_in_2_months_l725_725006

def students : ℕ := 30
def contribution_per_student_per_week : ℕ := 2
def weeks_in_month : ℕ := 4
def months : ℕ := 2

def total_contribution_per_week : ℕ := students * contribution_per_student_per_week
def total_weeks : ℕ := months * weeks_in_month
def total_savings : ℕ := total_contribution_per_week * total_weeks

theorem total_savings_in_2_months : total_savings = 480 := by
  -- Proof goes here
  sorry

end total_savings_in_2_months_l725_725006


namespace find_other_package_size_l725_725786

variable (total_coffee : ℕ)
variable (total_5_ounce_packages : ℕ)
variable (num_other_packages : ℕ)
variable (other_package_size : ℕ)

theorem find_other_package_size
  (h1 : total_coffee = 85)
  (h2 : total_5_ounce_packages = num_other_packages + 2)
  (h3 : num_other_packages = 5)
  (h4 : 5 * total_5_ounce_packages + other_package_size * num_other_packages = total_coffee) :
  other_package_size = 10 :=
sorry

end find_other_package_size_l725_725786


namespace library_books_proof_l725_725940

theorem library_books_proof :
  ∀ (X : ℕ), 100 + 4 * X = 300 → X = 50 :=
by
  assume X,
  assume h : 100 + 4 * X = 300,
  have h1 : 4 * X = 300 - 100, from eq_sub_of_add_eq h,
  have h2 : 4 * X = 200, from h1,
  have h3 : X = 200 / 4, from eq_div_of_mul_eq h2,
  have h4 : X = 50, from h3,
  show X = 50, from h4

end library_books_proof_l725_725940


namespace count_ways_to_make_200_yuan_l725_725291

noncomputable def count_methods (count_100 count_50 count_20 count_10 : Nat) (target_sum : Nat) : Nat :=  -- function to count methods
  let methods := { (x, y, z, w) : Nat × Nat × Nat × Nat | x ≤ count_100 ∧ y ≤ count_50 ∧ z ≤ count_20 ∧ w ≤ count_10 ∧ (x * 100 + y * 50 + z * 20 + w * 10 = target_sum) }
  methods.to_finset.card

theorem count_ways_to_make_200_yuan :
  count_methods 1 2 5 10 200 = 20 := by
  sorry

end count_ways_to_make_200_yuan_l725_725291


namespace polynomial_zero_l725_725978

open Polynomial

-- Given Conditions
def poly_deg_lt (P : Polynomial ℝ) (m : ℕ) : Prop :=
  P.degree < m

noncomputable def cond1 (P Q R : Polynomial ℝ) (m : ℕ) (hPdeg : poly_deg_lt P m) (hQdeg : poly_deg_lt Q m) (hRdeg : poly_deg_lt R m) : Prop :=
  ∀ x y : ℝ, x^(2 * m) * P.eval2 x y + y^(2 * m) * Q.eval2 x y = (x + y)^(2 * m) * R.eval2 x y 

-- Proposition to Prove
theorem polynomial_zero (P Q R : Polynomial ℝ) (m : ℕ)
  (hPdeg : poly_deg_lt P m) (hQdeg : poly_deg_lt Q m) (hRdeg : poly_deg_lt R m)
  (hc : cond1 P Q R m hPdeg hQdeg hRdeg) : P = 0 ∧ Q = 0 ∧ R = 0 :=
sorry

end polynomial_zero_l725_725978


namespace area_of_region_l725_725419

noncomputable def sec (θ : ℝ) := (cos θ)⁻¹
noncomputable def csc (θ : ℝ) := (sin θ)⁻¹

def region (r θ : ℝ) : Prop :=
  (r = 2 * sec θ ∧ (0 ≤ θ ∧ θ ≤ π / 2)) ∨ 
  (r = 2 * csc θ ∧ (0 ≤ θ ∧ θ ≤ π / 2))

theorem area_of_region :
  let bounded_region := { (x, y) | 0 ≤ x ∧ x ≤ 2 ∧ 0 ≤ y ∧ y ≤ 2 } in
  ∃ (A : ℝ), A = 4 ∧ (∀ (a b : ℝ), bounded_region (a, b)) :=
begin
  let bounded_region := { p | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2 },
  use 4,
  split,
  { refl, },
  { intros a b hb,
    exact hb, },
end

end area_of_region_l725_725419


namespace range_of_a_l725_725868

noncomputable def f (x a : ℝ) : ℝ := x - a * real.log x

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x > 1 → f x a > 0) ↔ a < real.exp 1 :=
sorry

end range_of_a_l725_725868


namespace sum_positive_integers_divisible_expression_l725_725818

theorem sum_positive_integers_divisible_expression :
  let N := 5000
  let p := 2477
  let expression := λ n: ℤ, n^2 + 2475 * n + 2454 + (-1)^n
  ∑ n in Finset.filter (λ n, (1 ≤ n ∧ n ≤ N) ∧ (expression n) % p = 0) (Finset.range (N + 1)) = 9912 :=
sorry

end sum_positive_integers_divisible_expression_l725_725818


namespace problem_statement_l725_725874

-- Variables and definitions for conditions
def line_l1 (t : ℝ) : ℝ × ℝ := (t, t * real.sqrt 3)
def circle_C1 (x y : ℝ) : Prop := (x - real.sqrt 3) ^ 2 + (y - 2) ^ 2 = 1

-- Polar coordinates transformation
def polar (ρ θ : ℝ) : ℝ × ℝ := (ρ * real.cos θ, ρ * real.sin θ)

-- Problem statement
theorem problem_statement :
  ∀ (ρ θ : ℝ),
    (polar ρ θ = (line_l1 ρ).fst → θ = real.pi / 3) ∧
    ((polar ρ θ).fst ^ 2 - 2 * real.sqrt 3 * (polar ρ θ).fst * real.cos θ 
    - 4 * (polar ρ θ).snd * real.sin θ + 6 = 0) ∧
    (∃ (ρ₁ ρ₂ : ℝ), (ρ₁, ρ₂) ∈ ({ρ | polar ρ (real.pi / 3) ∈ λ x y, circle_C1 x y } 
    → |ρ₁ - ρ₂| = real.sqrt 3) ∧ 
    (∃ A M N : ℝ × ℝ, 
      A = (0, 0) ∧
      M ∈ { M : ℝ × ℝ | circle_C1 M.fst M.snd ∧ (polar M.fst (real.pi / 3)) } ∧
      N ∈ { N : ℝ × ℝ | circle_C1 N.fst N.snd ∧ (polar N.fst (real.pi / 3)) } ∧
      (1/2 * |M - N| * 1/2 = real.sqrt 3 / 4))) := sorry

end problem_statement_l725_725874


namespace option_C_true_l725_725803

theorem option_C_true (a b : ℝ):
    (a^2 + b^2 ≥ 2 * a * b) ↔ ((a^2 + b^2 > 2 * a * b) ∨ (a^2 + b^2 = 2 * a * b)) :=
by
  sorry

end option_C_true_l725_725803


namespace test_passing_students_l725_725554

theorem test_passing_students (total_students : ℕ)
  (long_jump_pass : ℕ)
  (shot_put_pass : ℕ)
  (failed_both : ℕ)
  (passed_at_least_one : ℕ)
  (both_tests_pass : ℕ) :
  total_students = 50 →
  long_jump_pass = 40 →
  shot_put_pass = 31 →
  failed_both = 4 →
  passed_at_least_one = total_students - failed_both →
  passed_at_least_one = long_jump_pass + shot_put_pass - both_tests_pass →
  both_tests_pass = 25 :=
by {
  intros,
  linarith,
}

end test_passing_students_l725_725554


namespace problem1_problem2_problem3_l725_725647

noncomputable def basketball_price : ℝ := 120
noncomputable def jump_rope_price : ℝ := 25
noncomputable def num_basketballs : ℕ := 40

def cost_store_A (x : ℕ) : ℝ :=
  let actual_jumpropes := x - num_basketballs in
  (num_basketballs * basketball_price) + (actual_jumpropes * jump_rope_price)

def cost_store_B (x : ℕ) : ℝ :=
  0.9 * ((num_basketballs * basketball_price) + (x * jump_rope_price))

theorem problem1 (x : ℕ) (h : x > num_basketballs) :
  cost_store_A x = 3800 + 25 * x ∧ cost_store_B x = 4320 + 22.5 * x := sorry

theorem problem2 :
  cost_store_A 80 = 5800 ∧ cost_store_B 80 = 6120 ∧ cost_store_A 80 < cost_store_B 80 := sorry

theorem problem3 :
  (num_basketballs * basketball_price + num_basketballs * 0) + (40 * jump_rope_price * 0.9) = 5700 := sorry

end problem1_problem2_problem3_l725_725647


namespace triangle_angles_l725_725110

noncomputable def angle1 := real.arccos (5 / 8) * 180 / real.pi
noncomputable def angle2 := (180 - angle1) / 2

theorem triangle_angles :
  ∀ (a b c : ℝ), a = 4 → b = 4 → c = real.sqrt 12 →
  let θ := real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) * 180 / real.pi,
      α := (180 - θ) / 2 in
  θ ≈ 51.06 ∧ α ≈ 64.47 ∧ α ≈ 64.47 :=
begin
  intros a b c ha hb hc,
  rw [ha, hb, hc],
  let θ := real.arccos ((4^2 + 4^2 - (real.sqrt 12)^2) / (2 * 4 * 4)) * 180 / real.pi,
  have θ_val : θ = angle1 := rfl,
  let α := (180 - θ) / 2,
  have α_val : α = angle2 := rfl,
  split,
  { rw θ_val, exact sorry }, -- proof that angle1 ≈ 51.06
  split,
  { rw α_val, exact sorry }, -- proof that angle2 ≈ 64.47
  { rw α_val, exact sorry }  -- proof that angle2 ≈ 64.47
end

end triangle_angles_l725_725110


namespace eccentricity_of_ellipse_l725_725856

theorem eccentricity_of_ellipse :
  ∃ (e : ℝ),
    (∀ (a b c : ℝ),
      a = 3 ∧
      b = sqrt 3 ∧
      c = sqrt (a^2 - b^2) ∧
      e = c / a) ∧
    e = sqrt 6 / 3 :=
begin
  sorry
end

end eccentricity_of_ellipse_l725_725856
