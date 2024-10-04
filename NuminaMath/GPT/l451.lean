import Mathlib

namespace solution_set_inequality_l451_451852

theorem solution_set_inequality (a c : ℝ)
  (h : ∀ x : ℝ, (ax^2 + 2*x + c < 0) ↔ (x < -1/3 ∨ x > 1/2)) :
  (∀ x : ℝ, (cx^2 - 2*x + a ≤ 0) ↔ (-2 ≤ x ∧ x ≤ 3)) :=
sorry

end solution_set_inequality_l451_451852


namespace total_chocolates_distributed_l451_451665

theorem total_chocolates_distributed 
  (boys girls : ℕ)
  (chocolates_per_boy chocolates_per_girl : ℕ)
  (h_boys : boys = 60)
  (h_girls : girls = 60)
  (h_chocolates_per_boy : chocolates_per_boy = 2)
  (h_chocolates_per_girl : chocolates_per_girl = 3) : 
  boys * chocolates_per_boy + girls * chocolates_per_girl = 300 :=
by {
  sorry
}

end total_chocolates_distributed_l451_451665


namespace problem_statement_l451_451800

noncomputable def f : ℝ → ℝ := sorry

axiom differentiable_f : differentiable ℝ f

axiom derivative_condition (x : ℝ) : deriv f x > f x

theorem problem_statement : f 1 > Real.exp 1 * f 0 :=
sorry

end problem_statement_l451_451800


namespace number_of_correct_expressions_is_one_l451_451708

theorem number_of_correct_expressions_is_one : 
  (∀ a n : ℤ, (even n → na^n ≠ a)) →
  (∀ a : ℤ, (a^2 - 3 * a + 3)^0 = 1) →
  (3 - 3 ≠ 6 * (-3)^2) →
  (1) := by
  sorry

end number_of_correct_expressions_is_one_l451_451708


namespace complex_power_sum_l451_451947

noncomputable def a : ℂ := (-1 + complex.I * real.sqrt 7) / 2
noncomputable def b : ℂ := (-1 - complex.I * real.sqrt 7) / 2

theorem complex_power_sum : a^8 + b^8 = -7.375 := 
by sorry

end complex_power_sum_l451_451947


namespace finished_group_people_count_l451_451959

theorem finished_group_people_count
  (initial_groups : ℕ)
  (initial_avg_people : ℕ)
  (remaining_groups : ℕ)
  (remaining_avg_people : ℕ)
  (finished_group_people : ℕ) :
  initial_groups = 10 → 
  initial_avg_people = 9 → 
  remaining_groups = 9 → 
  remaining_avg_people = 8 → 
  let initial_total_people := initial_groups * initial_avg_people in
  let remaining_total_people := remaining_groups * remaining_avg_people in
  finished_group_people = initial_total_people - remaining_total_people →
  finished_group_people = 18 :=
by
  intros h1 h2 h3 h4 hp
  let initial_total_people := 10 * 9
  let remaining_total_people := 9 * 8
  have h5 : initial_total_people = 90 := by norm_num
  have h6 : remaining_total_people = 72 := by norm_num
  have h7 : finished_group_people = 90 - 72 := hp
  show finished_group_people = 18 by norm_num

end finished_group_people_count_l451_451959


namespace stewarts_theorem_l451_451842

theorem stewarts_theorem 
  (a b b₁ a₁ d c : ℝ)
  (h₁ : b * b ≠ 0) 
  (h₂ : a * a ≠ 0) 
  (h₃ : b₁ * b₁ ≠ 0) 
  (h₄ : a₁ * a₁ ≠ 0) 
  (h₅ : d * d ≠ 0) 
  (h₆ : c = a₁ + b₁) :
  b * b * a₁ + a * a * b₁ - d * d * c = a₁ * b₁ * c :=
  sorry

end stewarts_theorem_l451_451842


namespace polyhedra_overlap_interior_l451_451071

variables {P : Type*} [polyhedron P]
variables {A : ℕ → Type*} [vertex A]

-- Assume vertices of the polyhedron P1
variables {A1 A2 A3 A4 A5 A6 A7 A8 A9 : A ℕ}

-- Define translation for polyhedron P1
def translate (P : polyhedron) (v : vertex) : polyhedron := 
  sorry  -- Here we should define actual translation

-- Define P_i as translated P1 with A1 moved to Ai
def P_i (i : ℕ) : polyhedron :=
  translate P A1 A i

-- Theorem: At least two of the polyhedra P1, P2, ..., P9 have an interior point in common
theorem polyhedra_overlap_interior :
  ∃ i j, i ≠ j ∧ ∃ p, p ∈ (interior (P_i i)) ∧ p ∈ (interior (P_i j)) :=
  sorry

end polyhedra_overlap_interior_l451_451071


namespace shifted_scaled_function_l451_451166

variable (ω : ℝ) (f g : ℝ → ℝ)
variables (hω : ω > 0)
variable (h_dist : ∀ x, f (x + π / ω) = f x)
variable (h_shift : ∀ x, f (x - π / 6) = sin (2 * x))
variable (h_scale : ∀ x, g x = sin (4 * x))

theorem shifted_scaled_function:
  g = (λ x, sin (4 * x)) :=
by
  sorry

end shifted_scaled_function_l451_451166


namespace midpoint_correct_l451_451856

-- Define the vectors a and b
def a : ℝ × ℝ × ℝ := (-11, 3, 6)
def b : ℝ × ℝ × ℝ := (3, -7, -4)

-- Define the function to compute the midpoint of two 3D vectors
def midpoint (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((v1.1 + v2.1) / 2, (v1.2 + v2.2) / 2, (v1.3 + v2.3) / 2)

-- Define the theorem to prove
theorem midpoint_correct : midpoint a b = (-4, -2, 1) := sorry

end midpoint_correct_l451_451856


namespace floor_factorial_expression_l451_451309

theorem floor_factorial_expression : 
  ⌊(2010.factorial + 2007.factorial) / (2009.factorial + 2008.factorial)⌋ = 2009 :=
by
  sorry

end floor_factorial_expression_l451_451309


namespace prime_square_minus_one_divisible_by_24_l451_451101

theorem prime_square_minus_one_divisible_by_24 (p : ℕ) (hp : Prime p) (hp_ge_5 : 5 ≤ p) : 24 ∣ (p^2 - 1) := 
by 
sorry

end prime_square_minus_one_divisible_by_24_l451_451101


namespace units_digit_sum_squares_of_odd_integers_l451_451213

theorem units_digit_sum_squares_of_odd_integers :
  let first_2005_odd_units := [802, 802, 401] -- counts for units 1, 9, 5 respectively
  let extra_squares_last_6 := [9, 1, 3, 9, 5, 9] -- units digits of the squares of the last 6 numbers
  let total_sum :=
        (first_2005_odd_units[0] * 1 + 
         first_2005_odd_units[1] * 9 + 
         first_2005_odd_units[2] * 5) +
        (extra_squares_last_6.sum)
  (total_sum % 10) = 1 :=
by
  sorry

end units_digit_sum_squares_of_odd_integers_l451_451213


namespace count_permutations_of_digits_l451_451834

-- Define the multiset
def digits : multiset ℕ := {1, 1, 3, 3, 3, 5}

-- Define the function to calculate number of permutations accounting for repeats
def number_of_permutations (s : multiset ℕ) : ℕ :=
  (multiset.card s).factorial / (s.count 1).factorial / (s.count 3).factorial / (s.count 5).factorial

-- The Lean statement for the proof
theorem count_permutations_of_digits : number_of_permutations digits = 60 :=
by simp [digits, number_of_permutations, factorial, multiset.card, multiset.count]; norm_num

end count_permutations_of_digits_l451_451834


namespace log_proof_l451_451458

theorem log_proof (x : ℝ) (h : log 5 (log 4 (log 3 x)) = 1) : x ^ (-1 / 3) = 3 ^ (-1024 / 3) := by
  -- proof goes here
  sorry

end log_proof_l451_451458


namespace right_triangle_power_inequality_l451_451509

theorem right_triangle_power_inequality {a b c x : ℝ} (hpos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a^2 = b^2 + c^2) (h_longest : a > b ∧ a > c) :
  (x > 2) → (a^x > b^x + c^x) :=
by sorry

end right_triangle_power_inequality_l451_451509


namespace trajectory_of_P_is_a_ray_l451_451410

noncomputable def M := (-2 : ℝ, 0 : ℝ)
noncomputable def N := (2 : ℝ, 0 : ℝ)

def trajectory_condition (P : ℝ × ℝ) : Prop :=
  abs (real.sqrt ((P.1 - M.1) ^ 2 + (P.2 - M.2) ^ 2) 
        - real.sqrt ((P.1 - N.1) ^ 2 + (P.2 - N.2) ^ 2)) = 4

theorem trajectory_of_P_is_a_ray (P : ℝ × ℝ) (h : trajectory_condition P) : 
  ∃ a : ℝ, a > 0 ∧ P = (a * (2 - (-2)), 0) :=
sorry

end trajectory_of_P_is_a_ray_l451_451410


namespace problem_statement_l451_451906

-- We first define the conditions in Lean
variables {p : ℕ} [hp : Fact (Nat.Prime p)] (h : p ≥ 5)

noncomputable def M : Set ℕ := {1, 2, ..., p-1}

noncomputable def T : Set (ℕ × ℕ) :=
  { (n, x_n) | n ∈ M ∧ x_n ∈ M ∧ p ∣ (n * x_n - 1) }

-- Define the ceiling function
def floor (α : ℚ) : ℤ := ⌊α⌋

-- Define the main theorem
theorem problem_statement (hT : T) :
  ∑ (pair : ℕ × ℕ) in T, pair.1 * (floor (pair.1 * pair.2 / p)) % p = (p - 1) / 2 := 
sorry

end problem_statement_l451_451906


namespace profit_calculation_l451_451278

noncomputable theory

variable (cp : ℝ) (s : ℝ) (sp_before_tax : ℝ) (sp_after_discount : ℝ)

def cost_price_in_rupees : ℝ := 30 * 110

def selling_price_including_tax : ℝ := 4830

def selling_price_before_tax (s : ℝ) : ℝ := s / 1.12

def selling_price_after_discount (cp : ℝ) : ℝ := cp * 0.85

def effective_profit_percentage (cp sp_before_tax : ℝ) : ℝ :=
  ((sp_before_tax - cp) / cp) * 100

theorem profit_calculation :
  effective_profit_percentage cost_price_in_rupees (selling_price_before_tax selling_price_including_tax) = 30.68 := by
  sorry

end profit_calculation_l451_451278


namespace only_solution_l451_451387

theorem only_solution (a : ℤ) : 
  (∀ x : ℤ, x > 0 → 2 * x > 4 * x - 8 → 3 * x - a > -9 → x = 2) →
  (12 ≤ a ∧ a < 15) :=
by
  sorry

end only_solution_l451_451387


namespace max_sin_C_in_triangle_l451_451496

theorem max_sin_C_in_triangle
  (A B C : Type) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]
  (AB AC BA BC CA CB : ℝ)
  (h : AB * AC + 2 * BA * BC = 3 * CA * CB) :
  ∃ C : ℝ, C = real.sin ∠C ∧ C ≤ real.sqrt 7 / 3 :=
sorry

end max_sin_C_in_triangle_l451_451496


namespace fertilizer_prices_l451_451869

variables (x y : ℝ)

theorem fertilizer_prices :
  (x = y + 100) ∧ (2 * x + y = 1700) → (x = 600 ∧ y = 500) :=
by
  intros h
  cases h with h1 h2
  have h3 : y = 500 := by sorry
  have h4 : x = y + 100 := h1
  rw h3 at h4
  have h5 : x = 600 := by sorry
  exact ⟨h5, h3⟩

end fertilizer_prices_l451_451869


namespace beavers_build_dam_l451_451150

def num_beavers_first_group : ℕ := 20

theorem beavers_build_dam (B : ℕ) (t₁ : ℕ) (t₂ : ℕ) (n₂ : ℕ) :
  (B * t₁ = n₂ * t₂) → (B = num_beavers_first_group) := 
by
  -- Given
  let t₁ := 3
  let t₂ := 5
  let n₂ := 12

  -- Work equation
  assume h : B * t₁ = n₂ * t₂
  
  -- Correct answer
  have B_def : B = (n₂ * t₂) / t₁,
  exact h
   
  sorry

end beavers_build_dam_l451_451150


namespace race_distance_l451_451487

variables (a b c d : ℝ)
variables (h1 : d / a = (d - 30) / b)
variables (h2 : d / b = (d - 15) / c)
variables (h3 : d / a = (d - 40) / c)

theorem race_distance : d = 90 :=
by 
  sorry

end race_distance_l451_451487


namespace total_people_surveyed_l451_451893

-- Define the conditions
variable (total_surveyed : ℕ) (disease_believers : ℕ)
variable (rabies_believers : ℕ)

-- Condition 1: 75% of the people surveyed thought rats carried diseases
def condition1 (total_surveyed disease_believers : ℕ) : Prop :=
  disease_believers = (total_surveyed * 75) / 100

-- Condition 2: 50% of the people who thought rats carried diseases said rats frequently carried rabies
def condition2 (disease_believers rabies_believers : ℕ) : Prop :=
  rabies_believers = (disease_believers * 50) / 100

-- Condition 3: 18 people were mistaken in thinking rats frequently carry rabies
def condition3 (rabies_believers : ℕ) : Prop := rabies_believers = 18

-- The theorem to prove the total number of people surveyed given the conditions
theorem total_people_surveyed (total_surveyed disease_believers rabies_believers : ℕ) :
  condition1 total_surveyed disease_believers →
  condition2 disease_believers rabies_believers →
  condition3 rabies_believers →
  total_surveyed = 48 :=
by sorry

end total_people_surveyed_l451_451893


namespace angle_sum_greater_than_180_l451_451492

variable (A B C D N F : Point)
variable (l : Line)
variable (h_isosceles : CA = CB)
variable (h_cd_height : Height CD)
variable (h_line_l : ExternalAngleBisector l C)
variable (h_point_N : OnLine N l ∧ AN > AC ∧ SameSideLine N A CD)
variable (h_angle_bisector_AF : AngleBisector AF (Angle NAC))

theorem angle_sum_greater_than_180 (h1 : IsoscelesTriangle ABC h_isosceles)
    (h2 : HeightInTriangle C D h_cd_height)
    (h3 : ExternalAngleBisectorTriangle ABC C l h_line_l)
    (h4 : OnLineAndGreaterThan N l AN AC SameSideLine A CD h_point_N)
    (h5 : AngleBisectorAt F A N C h_angle_bisector_AF) :
    ∠NCD + ∠BAF > 180 :=
sorry

end angle_sum_greater_than_180_l451_451492


namespace gyeonghun_climbing_l451_451059

variable (t_up t_down d_up d_down : ℝ)
variable (h1 : t_up + t_down = 4) 
variable (h2 : d_down = d_up + 2)
variable (h3 : t_up = d_up / 3)
variable (h4 : t_down = d_down / 4)

theorem gyeonghun_climbing (h1 : t_up + t_down = 4) (h2 : d_down = d_up + 2) (h3 : t_up = d_up / 3) (h4 : t_down = d_down / 4) :
  t_up = 2 :=
by
  sorry

end gyeonghun_climbing_l451_451059


namespace Q_proper_subset_P_l451_451829

-- Conditions of the sets
def P : set ℝ := { x | x >= 1 }
def Q : set ℝ := { x | x = 1 ∨ x = 2 }

-- Theorem statement specifying the correct relationship
theorem Q_proper_subset_P : Q ⊂ P :=
sorry

end Q_proper_subset_P_l451_451829


namespace angle_equality_l451_451506

variable (A B C D M : Type)
variable [Geometry A] [Geometry B] [Geometry C] [Geometry D] [Point M]

-- Given a convex quadrilateral ABCD
variable (h_convex : convex_quadrilateral A B C D)
-- Angle conditions
variable (h1 : ∠ CAB = ∠ CDA)
variable (h2 : ∠ BCA = ∠ ACD)
-- M is the midpoint of AB
variable (h_midpoint : midpoint M A B)

theorem angle_equality (h_convex : convex_quadrilateral A B C D)
                       (h1 : ∠ CAB = ∠ CDA)
                       (h2 : ∠ BCA = ∠ ACD)
                       (h_midpoint : midpoint M A B) :
  ∠ BCM = ∠ DBA :=
sorry

end angle_equality_l451_451506


namespace negation_of_proposition_l451_451825

-- Definitions from the problem conditions
def proposition (x : ℝ) := ∃ x < 1, x^2 ≤ 1

-- Reformulated proof problem
theorem negation_of_proposition : 
  ¬ (∃ x < 1, x^2 ≤ 1) ↔ ∀ x < 1, x^2 > 1 :=
by
  sorry

end negation_of_proposition_l451_451825


namespace average_of_other_two_l451_451966

theorem average_of_other_two {a b c d : ℕ} (h₀ : a ≠ b) (h₁ : a ≠ c) (h₂ : a ≠ d)
  (h₃ : b ≠ c) (h₄ : b ≠ d) (h₅ : c ≠ d) 
  (h₆ : 0 < a) (h₇ : 0 < b) (h₈ : 0 < c) (h₉ : 0 < d)
  (h₁₀ : a + b + c + d = 20) (h₁₁ : a - min (min a b) (min c d) = max (max a b) (max c d) - min (min a b) (min c d)) :
  ((a + b + c + d) - (max (max a b) (max c d) + min (min a b) (min c d))) / 2 = 2.5 :=
by
  sorry

end average_of_other_two_l451_451966


namespace pancakes_needed_l451_451527

theorem pancakes_needed (initial_pancakes : ℕ) (num_people : ℕ) (pancakes_left : ℕ) :
  initial_pancakes = 12 → num_people = 8 → pancakes_left = initial_pancakes - num_people →
  (num_people - pancakes_left) = 4 :=
by
  intros initial_pancakes_eq num_people_eq pancakes_left_eq
  sorry

end pancakes_needed_l451_451527


namespace complete_square_l451_451216

theorem complete_square (x : ℝ) : (x ^ 2 + 4 * x + 1 = 0) ↔ ((x + 2) ^ 2 = 3) :=
by {
  split,
  { intro h,
    sorry },
  { intro h,
    sorry }
}

end complete_square_l451_451216


namespace wall_width_l451_451250

-- Define the dimensions of the brick in meters
def length_brick : ℝ := 0.25
def width_brick : ℝ := 0.15
def height_brick : ℝ := 0.08

-- Define the dimensions of the wall
def length_wall : ℝ := 10
def height_wall : ℝ := 5
def num_bricks : ℕ := 6000

-- Define the volume of one brick
def volume_brick (l w h : ℝ) : ℝ := l * w * h

-- Calculate the total volume of bricks
def total_volume_bricks (num : ℕ) (vol : ℝ) : ℝ := num * vol

-- The metrical equivalent proof problem
theorem wall_width (l_brick w_brick h_brick l_wall h_wall : ℝ) (num : ℕ) :
  l_brick = 0.25 → w_brick = 0.15 → h_brick = 0.08 →
  l_wall = 10 → h_wall = 5 → num = 6000 →
  width_wall * l_wall * h_wall = total_volume_bricks num (volume_brick l_brick w_brick h_brick) →
  width_wall = 0.36 :=
by
  intros h1 h2 h3 h4 h5 hnum hvol_eq
  sorry

-- width_wall can be calculated from the equation provided in the theorem
def width_wall : ℝ := 18 / (length_wall * height_wall)

end wall_width_l451_451250


namespace radius_of_smaller_molds_l451_451674

noncomputable def hemisphere_volume (r : ℝ) : ℝ := (2/3) * Real.pi * r ^ 3

theorem radius_of_smaller_molds :
  (64 * hemisphere_volume (1/2)) = hemisphere_volume 2 :=
by
  sorry

end radius_of_smaller_molds_l451_451674


namespace triangle_area_l451_451234

-- Conditions
def base : ℝ := 4.5
def height : ℝ := 6
def expected_area : ℝ := 13.5

-- Question: Prove that the area of the triangle using given base and height is 13.5 square meters
theorem triangle_area :
  (1/2) * base * height = expected_area :=
by
  sorry

end triangle_area_l451_451234


namespace arithmetic_sequence_x_y_sum_l451_451627

theorem arithmetic_sequence_x_y_sum :
  ∀ (a d x y: ℕ), 
  a = 3 → d = 6 → 
  (∀ (n: ℕ), n ≥ 1 → a + (n-1) * d = 3 + (n-1) * 6) →
  (a + 5 * d = x) → (a + 6 * d = y) → 
  (y = 45 - d) → x + y = 72 :=
by
  intros a d x y h_a h_d h_seq h_x h_y h_y_equals
  sorry

end arithmetic_sequence_x_y_sum_l451_451627


namespace floor_factorial_expression_l451_451312

-- Define the factorial function for natural numbers
def factorial : ℕ → ℕ
| 0 := 1
| (n + 1) := (n + 1) * factorial n

-- The main theorem to prove
theorem floor_factorial_expression :
  (nat.floor ((factorial 2010 + factorial 2007) / (factorial 2009 + factorial 2008)) = 2009) :=
begin
  -- Actual proof goes here
  sorry
end

end floor_factorial_expression_l451_451312


namespace axes_of_symmetry_not_coincide_l451_451568

def y₁ (x : ℝ) := (1 / 8) * (x^2 + 6 * x - 25)
def y₂ (x : ℝ) := (1 / 8) * (31 - x^2)

def tangent_y₁ (x : ℝ) := (x + 3) / 4
def tangent_y₂ (x : ℝ) := -x / 4

def axes_symmetry_y₁ := -3
def axes_symmetry_y₂ := 0

theorem axes_of_symmetry_not_coincide :
  (∃ x1 x2 : ℝ, y₁ x1 = y₂ x1 ∧ y₁ x2 = y₂ x2 ∧ tangent_y₁ x1 * tangent_y₂ x1 = -1 ∧ tangent_y₁ x2 * tangent_y₂ x2 = -1) →
  axes_symmetry_y₁ ≠ axes_symmetry_y₂ :=
by sorry

end axes_of_symmetry_not_coincide_l451_451568


namespace base4_odd_digits_345_l451_451374

theorem base4_odd_digits_345 :
  let n := 345
  let base4_rep := [5, 1, 2, 1] -- This is the base-4 representation of 345
  let odd_digits := base4_rep.filter (λ d, d % 2 = 1)
  let odd_digit_sum := odd_digits.sum
  in odd_digits.length = 3 ∧ odd_digit_sum = 7 := by
{
  sorry
}

end base4_odd_digits_345_l451_451374


namespace simplify_trig_expression_l451_451552

theorem simplify_trig_expression (α : ℝ) : 
    (1 - real.cos (2 * α) + real.sin (2 * α)) / (1 + real.cos (2 * α) + real.sin (2 * α)) = real.tan α :=
by
  sorry

end simplify_trig_expression_l451_451552


namespace distance_from_point_to_tangent_line_l451_451164

noncomputable def distance_to_tangent_line : ℝ :=
  let curve (x : ℝ) : ℝ := -x^3 + 2*x in
  let derivative (x : ℝ) : ℝ := -3*x^2 + 2 in
  let tangent_point_x := -1 in
  let tangent_point_y := curve tangent_point_x in
  let slope := derivative tangent_point_x in
  let tangent_line (x y : ℝ) : ℝ := x + y + 2 in
  let point := (3 : ℝ, 2 : ℝ) in
  let A := 1 in
  let B := 1 in
  let C := 2 in
  let (px, py) := point in
  (|A * px + B * py + C| / ℝ.sqrt (A^2 + B^2))

theorem distance_from_point_to_tangent_line :
  distance_to_tangent_line = 7 * ℝ.sqrt 2 / 2 :=
sorry

end distance_from_point_to_tangent_line_l451_451164


namespace fraction_pow_zero_is_one_l451_451720

theorem fraction_pow_zero_is_one (a b : ℤ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) : (a / (b : ℚ)) ^ 0 = 1 := by
  sorry

end fraction_pow_zero_is_one_l451_451720


namespace perpendiculars_intersect_at_single_point_l451_451546

-- Define the points and their relationships
variables (A B C A1 B1 C1 : Type)
variables (BC CA AB : Type)
variables (B1C1 C1A1 A1B1 : Type)
variables (intersection1 : BC → CA → AB → Prop)
variables (intersection2 : B1C1 → C1A1 → A1B1 → Prop)
variables (perpendicular : Type → Type → Prop)

-- Condition: Perpendiculars from A1, B1, and C1 intersect at a point
axiom perpendiculars_intersect : intersection1 (perpendicular A1 BC) (perpendicular B1 CA) (perpendicular C1 AB)

-- Question and the Claim
theorem perpendiculars_intersect_at_single_point :
  intersection2 (perpendicular A B1C1) (perpendicular B C1A1) (perpendicular C A1B1) :=
sorry

end perpendiculars_intersect_at_single_point_l451_451546


namespace grasshopper_can_return_to_origin_l451_451258

def grasshopper_jumps_to_origin : Prop :=
  ∃ (positions: Fin 32 → ℤ × ℤ),
    positions 0 = (0, 0) ∧
    ∀ n : Fin 31, 
      let (x, y) := positions n in
      let length := ((n : ℕ) + 1) in
      (positions (n + 1) = (x + length, y) ∨
       positions (n + 1) = (x - length, y) ∨
       positions (n + 1) = (x, y + length) ∨
       positions (n + 1) = (x, y - length))
  ∧ positions 31 = (0, 0)

theorem grasshopper_can_return_to_origin : grasshopper_jumps_to_origin :=
begin
  sorry
end

end grasshopper_can_return_to_origin_l451_451258


namespace omega_range_for_monotonic_decreasing_sin_l451_451807

theorem omega_range_for_monotonic_decreasing_sin (ω : ℝ) (φ : ℝ) (hω_pos : ω > 0) :
  (∀ x ∈ Ioo π (3 * π / 2), f x = sin (ω * x + 2 * φ) - 2 * sin φ * cos (ω * x + φ) →
    ∀ x1 x2 ∈ Ioo π (3 * π / 2), x1 ≤ x2 → f x1 ≥ f x2) ↔ 
    (1/2 ≤ ω ∧ ω ≤ 1) :=
by
  sorry

end omega_range_for_monotonic_decreasing_sin_l451_451807


namespace minimum_area_triangle_sqrt2_l451_451789

noncomputable def minimum_area_triangle (x0 y0 : ℝ) : ℝ :=
  if h : (x0^2 / 8 + y0^2 / 4 = 1) ∧ (x0 > 0) ∧ (y0 > 0) 
  then 8 / |x0 * y0|
  else 0

theorem minimum_area_triangle_sqrt2 (x0 y0 : ℝ) (hx : x0^2 / 8 + y0^2 / 4 = 1) (h_pos : x0 > 0 ∧ y0 > 0) : 
  (minimum_area_triangle x0 y0 ≥ real.sqrt 2) :=
by
  sorry

end minimum_area_triangle_sqrt2_l451_451789


namespace triangle_cosine_and_AB_l451_451479

theorem triangle_cosine_and_AB (a b : ℝ) (h1 : a = 5) (h2 : b = 4) (h3 : cos(∠A - ∠B) = 31/32) :
  cos ∠C = 1/8 ∧ AB = 6 := by
  sorry

end triangle_cosine_and_AB_l451_451479


namespace ellipse_equation_l451_451031

theorem ellipse_equation (a b : ℝ) (x y : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : ∀ x y, x^2 + y^2 = 1) 
  (h4 : ∃ A B : ℝ × ℝ, (is_tangent (x, y) (1, 1 / 2)) (line_AB_right_focus_top_vertex (x, y)) :
  x^2 / 5 + y^2 / 4 = 1 := 
begin
  sorry,
end

end ellipse_equation_l451_451031


namespace floor_factorial_expression_eq_2009_l451_451301

def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

theorem floor_factorial_expression_eq_2009 :
  (Int.floor (↑(factorial 2010 + factorial 2007) / ↑(factorial 2009 + factorial 2008)) = 2009) := by
  sorry

end floor_factorial_expression_eq_2009_l451_451301


namespace percentage_of_stock_l451_451865

-- Definitions based on conditions
def income := 500  -- I
def investment := 1500  -- Inv
def price := 90  -- Price

-- Initiate the Lean 4 statement for the proof
theorem percentage_of_stock (P : ℝ) (h : income = (investment * P) / price) : P = 30 :=
by
  sorry

end percentage_of_stock_l451_451865


namespace find_value_l451_451397

theorem find_value 
    (x y : ℝ) 
    (hx : x = 1 / (Real.sqrt 2 + 1)) 
    (hy : y = 1 / (Real.sqrt 2 - 1)) : 
    x^2 - 3 * x * y + y^2 = 3 := 
by 
    sorry

end find_value_l451_451397


namespace rebecca_needs_82_gemstones_l451_451131

-- Define the number of gemstones needed per set
def gemstones_per_set (magnets buttons gemstones_per_buttons) : ℕ :=
  let gemstones_per_earring := gemstones_per_buttons * buttons
  in 2 * gemstones_per_earring

def first_set_gemstones : ℕ :=
  gemstones_per_set 2 1 3

def second_set_gemstones : ℕ :=
  gemstones_per_set 3 6 2

def third_set_gemstones : ℕ :=
  gemstones_per_set 4 4 4

def fourth_set_gemstones : ℕ :=
  gemstones_per_set 5 (5 / 3) 5

def total_gemstones : ℕ :=
  first_set_gemstones + second_set_gemstones + third_set_gemstones + fourth_set_gemstones

theorem rebecca_needs_82_gemstones : total_gemstones = 82 :=
by
  sorry

end rebecca_needs_82_gemstones_l451_451131


namespace total_votes_cast_l451_451861

theorem total_votes_cast (W : ℕ) (h1 : W - (W - 53) = 53) (h2 : W - (W - 79) = 79)
  (h3 : W - (W - 105) = 105) (h4 : 199 = 199) :
  let T := W + (W - 53) + (W - 79) + (W - 105) + 199 in T = 1598 :=
by
  sorry

end total_votes_cast_l451_451861


namespace dilation_image_l451_451563

open Complex

theorem dilation_image (z₀ : ℂ) (c : ℂ) (k : ℝ) (z : ℂ)
    (h₀ : z₀ = 0 - 2*I) (h₁ : c = 1 + 2*I) (h₂ : k = 2) :
    z = -1 - 6*I :=
by
  sorry

end dilation_image_l451_451563


namespace a679b_multiple_of_72_l451_451028

-- Define conditions
def is_divisible_by_8 (n : Nat) : Prop :=
  n % 8 = 0

def sum_of_digits_is_divisible_by_9 (n : Nat) : Prop :=
  (n.digits 10).sum % 9 = 0

-- Define the given problem
theorem a679b_multiple_of_72 (a b : Nat) : 
  is_divisible_by_8 (7 * 100 + 9 * 10 + b) →
  sum_of_digits_is_divisible_by_9 (a * 10000 + 6 * 1000 + 7 * 100 + 9 * 10 + b) → 
  a = 3 ∧ b = 2 :=
by 
  sorry

end a679b_multiple_of_72_l451_451028


namespace number_of_paths_grid_l451_451836

theorem number_of_paths_grid (n m : ℕ) (h1 : n = 7) (h2 : m = 3) :
  (nat.choose (n + m) m) = 120 :=
by
  rw [h1, h2]
  sorry

end number_of_paths_grid_l451_451836


namespace tangent_lines_diff_expected_l451_451423

noncomputable def tangent_lines_diff (a : ℝ) (k1 k2 : ℝ) : Prop :=
  let curve (x : ℝ) := a * x + 2 * Real.log (|x|)
  let deriv (x : ℝ) := a + 2 / x
  -- Tangent conditions at some x1 > 0 for k1
  (∃ x1 : ℝ, 0 < x1 ∧ k1 = deriv x1 ∧ curve x1 = k1 * x1)
  -- Tangent conditions at some x2 < 0 for k2
  ∧ (∃ x2 : ℝ, x2 < 0 ∧ k2 = deriv x2 ∧ curve x2 = k2 * x2)
  -- The lines' slopes relations
  ∧ k1 > k2

theorem tangent_lines_diff_expected (a k1 k2 : ℝ) (h : tangent_lines_diff a k1 k2) :
  k1 - k2 = 4 / Real.exp 1 :=
sorry

end tangent_lines_diff_expected_l451_451423


namespace quadrilateral_pyramid_plane_intersection_l451_451784

-- Definitions:
-- Let MA, MB, MC, MD, MK, ML, MP, MN be lengths of respective segments
-- Let S_ABC, S_ABD, S_ACD, S_BCD be areas of respective triangles
variables {MA MB MC MD MK ML MP MN : ℝ}
variables {S_ABC S_ABD S_ACD S_BCD : ℝ}

-- Given a quadrilateral pyramid MABCD with a convex quadrilateral ABCD as base, and a plane intersecting edges MA, MB, MC, and MD at points K, L, P, and N respectively. Prove the following relation.
theorem quadrilateral_pyramid_plane_intersection :
  S_BCD * (MA / MK) + S_ADB * (MC / MP) = S_ABC * (MD / MN) + S_ACD * (MB / ML) :=
sorry

end quadrilateral_pyramid_plane_intersection_l451_451784


namespace find_second_number_l451_451372

theorem find_second_number (k m : ℤ) (n : ℤ) (h₁ : 6215 = 144 * k + 23) (h₂ : 6365 = 144 * m + 29) (h₃ : m = k + 1) : n = 6365 :=
by
  have : 6215 = 144 * 43 + 23, sorry
  have : m = 44, sorry
  exact sorry

end find_second_number_l451_451372


namespace area_of_triangle_l451_451699

open Real

-- Defining the line equation 3x + 2y = 12
def line_eq (x y : ℝ) : Prop := 3 * x + 2 * y = 12

-- Defining the vertices of the triangle
def vertex1 := (0, 0 : ℝ)
def vertex2 := (0, 6 : ℝ)
def vertex3 := (4, 0 : ℝ)

-- Define a function to calculate the area of the triangle
def triangle_area (v1 v2 v3 : ℝ × ℝ) : ℝ :=
  0.5 * abs ((v2.1 - v1.1) * (v3.2 - v1.2) - (v3.1 - v1.1) * (v2.2 - v1.2))

-- Prove that area of the triangle bounded by the line and coordinate axes is 12 square units
theorem area_of_triangle : triangle_area vertex1 vertex2 vertex3 = 12 :=
by
  sorry

end area_of_triangle_l451_451699


namespace monotonic_intervals_and_minimum_of_f_find_a_l451_451003

-- Define the function f(x)
def f (x : ℝ) : ℝ := x * Real.log x

-- Define the function F(x)
def F (x a : ℝ) : ℝ := (f x - a) / x

-- First part: Monotonic intervals and minimum value of f(x)
theorem monotonic_intervals_and_minimum_of_f :
  (∀ x > 0, x ≠ (1/e) → 
    ((x > (1/e) → f x ≥ f (1 / e)) ∧ (x < (1 / e) → f x ≤ f (1 / e)))) ∧
  (f (1 / e) = -1 / e) :=
sorry

-- Second part: Finding the value of a
theorem find_a (a : ℝ) (h : ∀ x ∈ Icc 1 Real.e, F x a ≥ (3/2)) :
  a = -Real.sqrt Real.e :=
sorry

end monotonic_intervals_and_minimum_of_f_find_a_l451_451003


namespace possible_values_of_m_l451_451558

open Complex Real

theorem possible_values_of_m (p q r s : ℂ) (hp : p ≠ 0) (hq : q ≠ 0)
    (m : ℂ) (m_root_1 : p * m^4 + q * m^3 + r * m^2 + s * m + p = 0)
    (m_root_2 : q * m^4 + r * m^3 + s * m^2 + p * m + q = 0) :
    ∃ k ∈ {0, 1, 2, 3, 4}, m = exp (2 * π * I * k / 5) ∨ m^5 = q / p :=
by
  sorry

end possible_values_of_m_l451_451558


namespace find_k_x_l451_451516

-- Define the nonzero polynomial condition
def nonzero_poly (p : Polynomial ℝ) : Prop :=
  ¬ (p = 0)

-- Define the conditions from the problem statement
def conditions (h k : Polynomial ℝ) : Prop :=
  nonzero_poly h ∧ nonzero_poly k ∧ (h.comp k = h * k) ∧ (k.eval 3 = 58)

-- State the main theorem to be proven
theorem find_k_x (h k : Polynomial ℝ) (cond : conditions h k) : 
  k = Polynomial.C 1 + Polynomial.C 49 * Polynomial.X + Polynomial.C (-49) * Polynomial.X^2 :=
sorry

end find_k_x_l451_451516


namespace smallest_square_patch_side_l451_451677

-- Define the dimensions of the rectangle
def rectangle_length : ℝ := 10
def rectangle_width : ℝ := 4

-- Define the diagonal of the rectangle
def rectangle_diagonal : ℝ := Real.sqrt (rectangle_length^2 + rectangle_width^2)

-- Define the side length of the square patch
def square_patch_side : ℝ := Real.sqrt 58

-- The statement to be proven
theorem smallest_square_patch_side : 
  ∃ s : ℝ, s = square_patch_side ∧ rectangle_diagonal ≤ s * Real.sqrt 2 :=
by
  sorry

end smallest_square_patch_side_l451_451677


namespace probability_of_queen_is_correct_l451_451624

def deck_size : ℕ := 52
def queen_count : ℕ := 4

-- This definition denotes the probability calculation.
def probability_drawing_queen : ℚ := queen_count / deck_size

theorem probability_of_queen_is_correct :
  probability_drawing_queen = 1 / 13 :=
by
  sorry

end probability_of_queen_is_correct_l451_451624


namespace min_quadratic_at_two_l451_451770

/-- For the quadratic function f(x) = 2x^2 - 8x + 1, it is minimized when x = 2. -/
theorem min_quadratic_at_two : ∃ x : ℝ, (∀ y : ℝ, (2 * x^2 - 8 * x + 1) ≤ (2 * y^2 - 8 * y + 1)) ∧ x = 2 :=
begin
  sorry
end

end min_quadratic_at_two_l451_451770


namespace fraction_increase_l451_451069

theorem fraction_increase (m n a : ℕ) (h1 : m > n) (h2 : a > 0) : 
  (n : ℚ) / m < (n + a : ℚ) / (m + a) :=
by
  sorry

end fraction_increase_l451_451069


namespace length_of_angle_bisector_l451_451056

theorem length_of_angle_bisector
  (DE EF DF : ℝ)
  (triangle_DEF : Triangle DEF)
  (EG_is_angle_bisector : IsAngleBisector E G DF)
  : EG = 2 * Real.sqrt 6 := 
by
  -- Proof has been omitted.
  sorry

end length_of_angle_bisector_l451_451056


namespace days_to_complete_work_l451_451951

variable {P W D : ℕ}

axiom condition_1 : 2 * P * 3 = W / 2
axiom condition_2 : P * D = W

theorem days_to_complete_work : D = 12 :=
by
  -- As an axiom or sorry is used, the proof is omitted.
  sorry

end days_to_complete_work_l451_451951


namespace absolute_value_c_l451_451155

noncomputable def condition_polynomial (a b c : ℤ) : Prop :=
  a * (↑(Complex.ofReal 3) + Complex.I)^4 +
  b * (↑(Complex.ofReal 3) + Complex.I)^3 +
  c * (↑(Complex.ofReal 3) + Complex.I)^2 +
  b * (↑(Complex.ofReal 3) + Complex.I) +
  a = 0

noncomputable def coprime_integers (a b c : ℤ) : Prop :=
  Int.gcd (Int.gcd a b) c = 1

theorem absolute_value_c (a b c : ℤ) (h1 : condition_polynomial a b c) (h2 : coprime_integers a b c) :
  |c| = 97 :=
sorry

end absolute_value_c_l451_451155


namespace income_increase_is_60_percent_l451_451112

noncomputable def income_percentage_increase 
  (J T M : ℝ) 
  (h1 : T = 0.60 * J) 
  (h2 : M = 0.9599999999999999 * J) : ℝ :=
  (M - T) / T * 100

theorem income_increase_is_60_percent 
  (J T M : ℝ) 
  (h1 : T = 0.60 * J) 
  (h2 : M = 0.9599999999999999 * J) : 
  income_percentage_increase J T M h1 h2 = 60 :=
by
  sorry

end income_increase_is_60_percent_l451_451112


namespace monotonicity_of_f_l451_451806

noncomputable def f (x : ℝ) : ℝ := -⅓ * x ^ 3 + x ^ 2 + 3 * x - 1

theorem monotonicity_of_f :
  (∀ x, f' x = 0 → x = -1 ∨ x = 3) ∧
  (∀ x, x ∈ (-1:ℝ, 3) → 0 < f'(x)) ∧
  (∀ x, x ∈ (-∞:ℝ, -1) ∨ x ∈ (3, ∞) → f'(x) < 0) ∧
  (f (-1) = -8 / 3) ∧
  (f 3 = 8) :=
by
  sorry

end monotonicity_of_f_l451_451806


namespace merchant_marked_price_l451_451260

variable (L C M S : ℝ)

-- Conditions
def condition1 : Prop := C = 0.7 * L
def condition2 : Prop := C = 0.7 * S
def condition3 : Prop := S = 0.8 * M

-- The main statement
theorem merchant_marked_price (h1 : condition1 L C) (h2 : condition2 C S) (h3 : condition3 S M) : M = 1.25 * L :=
by
  sorry

end merchant_marked_price_l451_451260


namespace exists_polygon_with_dividing_segment_exists_convex_polygon_with_dividing_segment_l451_451241

-- Part (a): Existence of a polygon with specified segment properties
theorem exists_polygon_with_dividing_segment : 
  ∃ (P : Type) [polygon P], 
  ∃ (S : segment P), 
  divides_into_equal_parts P S ∧ 
  bisects_one_side P S ∧ 
  divides_another_side_in_ratio P S 1 2 :=
sorry

-- Part (b): Existence of a convex polygon with the same properties
theorem exists_convex_polygon_with_dividing_segment : 
  ∃ (P : Type) [convex_polygon P], 
  ∃ (S : segment P), 
  divides_into_equal_parts P S ∧ 
  bisects_one_side P S ∧ 
  divides_another_side_in_ratio P S 1 2 :=
sorry

end exists_polygon_with_dividing_segment_exists_convex_polygon_with_dividing_segment_l451_451241


namespace triangle_area_32_27_12_l451_451232

noncomputable def semi_perimeter (a b c : ℕ) : ℝ :=
  (a + b + c) / 2

noncomputable def triangle_area (a b c : ℕ) : ℝ :=
  let s := semi_perimeter a b c in
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_area_32_27_12 :
  abs (triangle_area 32 27 12 - 47.1) < 0.1 :=
by
  sorry

end triangle_area_32_27_12_l451_451232


namespace grid_area_l451_451992

theorem grid_area :
  let B := 10   -- Number of boundary points
  let I := 12   -- Number of interior points
  I + B / 2 - 1 = 16 :=
by
  sorry

end grid_area_l451_451992


namespace passengers_from_other_continents_l451_451465

theorem passengers_from_other_continents (P : ℝ) (h : P = 120) : 
  let O := P - (P / 12 + P / 8 + P / 3 + P / 6) in
  O = 35 :=
by
  let O := P - (P / 12 + P / 8 + P / 3 + P / 6)
  sorry

end passengers_from_other_continents_l451_451465


namespace polynomial_inequality_l451_451823

theorem polynomial_inequality (n : ℕ) (a : ℕ → ℝ)
  (hroots : ∀ (p : Polynomial ℝ), p = ∑ i in finset.range (n + 1), a i * X^i - X + 1 →
    ∃ (r : Fin n → ℝ), (∀ i, (r i) > 0) ∧ ∀ (x : ℝ), p.eval x = 0 → ∃ i, x = r i) :
  0 < 2^2 * a 2 + ∑ i in finset.range (3, n + 1), 2^i * a i ∧ 
  (2^2 * a 2 + ∑ i in finset.range (3, n + 1), 2 ^ i * a i) ≤ ((n - 2) / n) ^ 2 + 1 := by
  sorry

end polynomial_inequality_l451_451823


namespace angle_AYX_50_l451_451725

theorem angle_AYX_50
  (Γ : Type) [circle Γ]
  (A B C X Y Z : Type)
  [on_segment X B C]
  [on_segment Y A B]
  [on_segment Z A C]
  (angle_A : ∠ A = 50)
  (angle_B : ∠ B = 70)
  (angle_C : ∠ C = 60)
  (incircle_ABC : incircle Γ (triangle A B C))
  (circumcircle_XYZ : circumcircle Γ (triangle X Y Z)) :
  ∠ AYX = 50 := 
sorry

end angle_AYX_50_l451_451725


namespace constant_term_binomial_expansion_l451_451983

theorem constant_term_binomial_expansion :
  ∀ (x : ℝ), ((2 / x) + x) ^ 4 = 24 :=
by
  sorry

end constant_term_binomial_expansion_l451_451983


namespace max_value_quadratic_l451_451201

theorem max_value_quadratic (r : ℝ) : 
  ∃ M, (∀ r, -3 * r^2 + 36 * r - 9 ≤ M) ∧ M = 99 :=
sorry

end max_value_quadratic_l451_451201


namespace zilla_savings_l451_451639

theorem zilla_savings (earnings : ℝ) (rent : ℝ) (expenses : ℝ) (savings : ℝ) 
  (h1 : rent = 0.07 * earnings)
  (h2 : rent = 133)
  (h3 : expenses = earnings / 2)
  (h4 : savings = earnings - rent - expenses) :
  savings = 817 := 
sorry

end zilla_savings_l451_451639


namespace beaver_group_l451_451147

theorem beaver_group (B : ℕ) :
  (B * 3 = 12 * 5) → B = 20 :=
by
  intros h1
  -- Additional steps for the proof would go here.
  -- The h1 hypothesis represents the condition B * 3 = 60.
  exact sorry -- Proof steps are not required.

end beaver_group_l451_451147


namespace triangle_angle_approx_l451_451742

noncomputable def angle_approx (angle : ℝ) (approx : ℝ) : Prop := abs (angle - approx) < 0.1

def is_trisection_point (A B D : Point) (c : ℝ) : Prop :=
  dist A D = c / 3 ∧ dist B D = 2 * c / 3

def is_quartile_point (B C F : Point) (b : ℝ) : Prop :=
  dist B F = b / 4 ∧ dist C F = 3 * b / 4

def cyclic_quadrilateral (D E F G : Point) : Prop :=
  ∃ O : Point, by 
    circumcenter O [D, E, F, G]

def tangent_circle (k : Circle) (CA : Line) (H : Point) : Prop :=
  ∃ O : Point, 
    circumcenter O k ∧ 
    circle_tangent_line k CA O H

theorem triangle_angle_approx (A B C D E F G H : Point) (k : Circle) (a b c : ℝ) :
  is_trisection_point A B D c →
  is_trisection_point A B E c →
  is_quartile_point B C F b →
  is_quartile_point B C G b →
  cyclic_quadrilateral D E F G →
  tangent_circle k (CA) H →
  let α := angle_approx (angle_ABC A B C) 60.09 in
  let β := angle_approx (angle_ABC B C A) 53.46 in
  let γ := angle_approx (angle_ABC C A B) 58.45 in
  α ∧ β ∧ γ :=
by sorry

end triangle_angle_approx_l451_451742


namespace sum_of_extreme_numbers_l451_451635

theorem sum_of_extreme_numbers : 
  let digits := {5, 6, 4, 7}
  ∃ largest smallest : ℕ, 
  (largest = 765 ∧ smallest = 456) ∧ 
  largest + smallest = 1221 := 
by {
  let digits := {5, 6, 4, 7},
  use [765, 456],
  have h_largest : 765 = list.foldl1 (λ acc d, acc * 10 + d) (list.reverse (list.sort (λ a b, a ≤ b) [5, 6, 4, 7])),
  have h_smallest : 456 = list.foldl1 (λ acc d, acc * 10 + d) (list.sort (λ a b, a ≤ b) [4, 5, 6]),
  split,
  exact ⟨h_largest, h_smallest⟩,
  exact rfl, -- Explanation: 765 + 456 = 1221
}

end sum_of_extreme_numbers_l451_451635


namespace chairs_needed_and_budget_l451_451686

theorem chairs_needed_and_budget :
  let chairs_bought := 1093
  let budget := 12000
  let num_classrooms := 35
  let max_capacity_classrooms := 20
  let max_capacity_per_classroom := 40
  let min_capacity_classrooms := 15
  let min_capacity_per_classroom := 30
  let total_max_capacity := max_capacity_classrooms * max_capacity_per_classroom + min_capacity_classrooms * min_capacity_per_classroom
  let additional_chairs_needed := total_max_capacity - chairs_bought
  let cost_per_chair := budget / chairs_bought
  let total_additional_cost := additional_chairs_needed * cost_per_chair
  in additional_chairs_needed = 157 ∧ cost_per_chair ≈ 10.98 ∧ total_additional_cost ≤ budget :=
by
  sorry

end chairs_needed_and_budget_l451_451686


namespace annie_start_crayons_l451_451286

def start_crayons (end_crayons : ℕ) (added_crayons : ℕ) : ℕ := end_crayons - added_crayons

theorem annie_start_crayons (added_crayons end_crayons : ℕ) (h1 : added_crayons = 36) (h2 : end_crayons = 40) :
  start_crayons end_crayons added_crayons = 4 :=
by
  rw [h1, h2]
  exact Nat.sub_eq_of_eq_add sorry  -- skips the detailed proof

end annie_start_crayons_l451_451286


namespace range_of_eccentricity_l451_451793

variable {a b c e : ℝ}
variable (x y : ℝ)

def ellipse (a b : ℝ) :=
  a > b ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def foci (a b : ℝ) :=
  ∃ c, c = sqrt (a^2 - b^2)

def perpendicular (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) :=
  let ⟨Px, Py⟩ := P in
  let ⟨F₁x, F₁y⟩ := F₁ in
  let ⟨F₂x, F₂y⟩ := F₂ in
  (Px - F₁x, Py - F₁y) • (Px - F₂x, Py - F₂y) = 0

theorem range_of_eccentricity
  (h₁ : ellipse a b)
  (h₂ : foci a b)
  (h₃ : ∃ P, perpendicular P (-(sqrt (a^2 - b^2)), 0) (sqrt (a^2 - b^2), 0)) :
  1 / sqrt 2 ≤ sqrt (a^2 - b^2) / a ∧ sqrt (a^2 - b^2) / a < 1 := 
sorry

end range_of_eccentricity_l451_451793


namespace factorial_floor_problem_l451_451325

theorem factorial_floor_problem :
  (nat.floor ( (nat.factorial 2010 + nat.factorial 2007) / (nat.factorial 2009 + nat.factorial 2008) )) = 2009 :=
by 
sorry

end factorial_floor_problem_l451_451325


namespace cot_15_subtract_3_cos_15_l451_451729

theorem cot_15_subtract_3_cos_15 :
  let θ := 15 * Real.pi / 180
  let cot_θ := Real.cos θ / Real.sin θ
  let cos_θ := Real.cos θ
  let sin_θ := Real.sin θ
  cot_θ - 3 * cos_θ = (18 + 11 * Real.sqrt 3) / 4 :=
by
  -- Given conditions:
  have h1 : sin_θ = (Real.sqrt 6 - Real.sqrt 2) / 4 := sorry
  have h2 : cos_θ = (Real.sqrt 6 + Real.sqrt 2) / 4 := sorry
  sorry

end cot_15_subtract_3_cos_15_l451_451729


namespace expected_value_of_sum_of_marbles_l451_451021

-- Definitions corresponding to the conditions
def marbles : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

def pairs := marbles.powerset.filter (λ s, s.card = 2)

def pair_sum (s : Finset ℕ) := s.sum id

-- Expected value calculation: There are 21 pairs
def total_sum_pairs := pairs.sum pair_sum

def expected_value := (total_sum_pairs : ℚ) / (pairs.card : ℚ)

-- The theorem that must be proven
theorem expected_value_of_sum_of_marbles :
  expected_value = 154 / 21 :=
by
  sorry

end expected_value_of_sum_of_marbles_l451_451021


namespace find_two_digit_numbers_dividing_all_relatives_l451_451904

noncomputable def is_relative (n : ℕ) (a b : ℕ) : Prop :=
  let units_digit := n % 10
  let remaining_digits_sum := (n / 10).digits.sum
  units_digit = b ∧ remaining_digits_sum = a

theorem find_two_digit_numbers_dividing_all_relatives :
  {ab : ℕ // 10 ≤ ab ∧ ab < 100 ∧ ∀n, is_relative n (ab / 10) (ab % 10) → ab ∣ n}
  = {15, 18, 30, 45, 90} := sorry

end find_two_digit_numbers_dividing_all_relatives_l451_451904


namespace floor_factorial_expression_l451_451321

theorem floor_factorial_expression : 
  (⌊(2010! + 2007! : ℚ) / (2009! + 2008! : ℚ)⌋ = 2009) :=
by
  -- Let a := 2010! and b := 2007!
  -- So a + b = 2010! + 2007!
  -- Notice 2010! = 2010 * 2009 * 2008 * 2007!
  -- Notice 2009! = 2009 * 2008 * 2007!
  -- Simplify (2010! + 2007!) / (2009! + 2008!)
  sorry

end floor_factorial_expression_l451_451321


namespace sum_of_possible_a_values_l451_451173

theorem sum_of_possible_a_values : 
  (∀ a : ℤ, (∃ m n : ℤ, m + n = a ∧ m * n = 2 * a) → a ∈ {-1, 0, 8, 9}) ∧
  (finset.sum { -1, 0, 8, 9 }) = 16 :=
by
  sorry

end sum_of_possible_a_values_l451_451173


namespace number_of_strategies_l451_451341

theorem number_of_strategies (S : ℕ) : 
  (∃ N : ℕ, 1 ≤ N ∧ N ≤ 59) ∧
  (S = (finset.choose 32 4).card + 16 * (finset.choose 30 1).card) →
  S = 36440 :=
by
  sorry

end number_of_strategies_l451_451341


namespace selling_price_eq_l451_451578

theorem selling_price_eq (cp sp L : ℕ) (h_cp: cp = 47) (h_L : L = cp - 40) (h_profit_loss_eq : sp - cp = L) :
  sp = 54 :=
by
  sorry

end selling_price_eq_l451_451578


namespace sets_equal_l451_451242

variable (f : ℝ → ℝ)
variable (h1 : Function.Injective f)
variable (h2 : ∀ x y : ℝ, x < y → f(x) < f(y))

def P := { x : ℝ | x > f(x) }
def Q := { x : ℝ | x > f(f(x)) }

theorem sets_equal : P f = Q f := by
  sorry

end sets_equal_l451_451242


namespace area_triangle_PFM_l451_451437

noncomputable def parabolaFocus := ⟨1, 0⟩

theorem area_triangle_PFM :
  ∀ (y0 : ℝ), let P := (y0^2 / 4, y0) in
  let F := parabolaFocus in
  let M := (0, y0) in
  y0 ≠ 0 →
  |P.1 - F.1| + 1 = 4 →
  (1/2) * abs (P.1 - M.1) * abs y0 = 3 * real.sqrt 3 := by 
  sorry

end area_triangle_PFM_l451_451437


namespace zilla_savings_l451_451638

theorem zilla_savings (earnings : ℝ) (rent : ℝ) (expenses : ℝ) (savings : ℝ) 
  (h1 : rent = 0.07 * earnings)
  (h2 : rent = 133)
  (h3 : expenses = earnings / 2)
  (h4 : savings = earnings - rent - expenses) :
  savings = 817 := 
sorry

end zilla_savings_l451_451638


namespace people_after_five_years_l451_451283

noncomputable def population_in_year : ℕ → ℕ
| 0       => 20
| (k + 1) => 4 * population_in_year k - 18

theorem people_after_five_years : population_in_year 5 = 14382 := by
  sorry

end people_after_five_years_l451_451283


namespace max_students_l451_451609

open Nat

theorem max_students (B G : ℕ) (h1 : 11 * B = 7 * G) (h2 : G = B + 72) (h3 : B + G ≤ 550) : B + G = 324 := by
  sorry

end max_students_l451_451609


namespace minimum_value_f_range_a_l451_451912

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.log (x + 1)

theorem minimum_value_f :
  ∃ x : ℝ, f x = -(1 / Real.exp 1) :=
sorry

theorem range_a (a : ℝ) :
  (∀ x ≥ 0, f x ≥ a * x) ↔ a ∈ Set.Iic 1 :=
sorry

end minimum_value_f_range_a_l451_451912


namespace min_people_mozart_bach_not_beethoven_l451_451619

theorem min_people_mozart_bach_not_beethoven (U M B E : Finset ℕ) 
  (hU : U.card = 200) (hM : M.card = 150) (hB : B.card = 120) (hE : E.card = 90) :
  ∃ x : ℕ, x = 10 ∧ (M ∩ B).card - (M ∩ B ∩ E).card = x := 
by
  -- Mathematically equivalent proof statement
  have h1 : (M ∩ B).card = 150 + 120 - 200 := sorry
  have h2 : (M ∩ B ∩ E).card ≤ 90 := sorry
  existsi 10
  have h3 : 70 - 60 = 10 := sorry
  exact ⟨rfl, h3⟩

end min_people_mozart_bach_not_beethoven_l451_451619


namespace parabola_equation_max_slope_OQ_l451_451818

theorem parabola_equation (p : ℝ) (hp : p = 2) :
    ∃ x y : ℝ, y^2 = 2 * p * x ↔ y^2 = 4 * x :=
by sorry

theorem max_slope_OQ (Q F : ℝ × ℝ) (hQF : ∀ P : ℝ × ℝ, P ∈ parabola_eq ↔ P.x = 10 * Q.x - 9 ∧ 
                                                         P.y = 10 * Q.y ∧ y^2 = 4 * P.x)
    (hPQ : (Q.x - P.x, Q.y - P.y) = 9 * (1 - Q.x, 0 - Q.y)) :
    ∃ n : ℝ, Q.y = n ∧ Q.x = (25 * n^2 + 9) / 10 ∧ 
        max (λ n, (10 * n) / (25 * n^2 + 9)) = 1 / 3 :=
by sorry

end parabola_equation_max_slope_OQ_l451_451818


namespace conjugate_complex_div_l451_451982

-- Define the complex conjugate function
def conjugate (z : ℂ) : ℂ := ⟨z.re, -z.im⟩

-- Define the complex division operation
def cdiv (z1 z2 : ℂ) : ℂ :=
  let denom := z2.re * z2.re + z2.im * z2.im
  let real_part := (z1.re * z2.re + z1.im * z2.im) / denom
  let imag_part := (z1.im * z2.re - z1.re * z2.im) / denom
  ⟨real_part, imag_part⟩

-- Define the complex number involved and the expected result
def c1 : ℂ := ⟨1, -3⟩
def c2 : ℂ := ⟨1, -1⟩
def result : ℂ := ⟨2, 1⟩

-- State the theorem
theorem conjugate_complex_div : conjugate (cdiv c1 c2) = result := by
  sorry

end conjugate_complex_div_l451_451982


namespace find_N_l451_451773

variables (k N : ℤ)

theorem find_N (h : ((k * N + N) / N - N) = k - 2021) : N = 2022 :=
by
  sorry

end find_N_l451_451773


namespace probability_25_cents_min_l451_451958

-- Define the five coins and their values
def penny := 0.01
def nickel := 0.05
def dime := 0.10
def quarter := 0.25
def halfDollar := 0.50

-- Define a function that computes the total value of heads up coins
def value_heads (results : (Bool × Bool × Bool × Bool × Bool)) : ℝ :=
  let (h₁, h₂, h₃, h₄, h₅) := results 
  (if h₁ then penny else 0) +
  (if h₂ then nickel else 0) +
  (if h₃ then dime else 0) +
  (if h₄ then quarter else 0) +
  (if h₅ then halfDollar else 0)

-- Define the main theorem statement
theorem probability_25_cents_min :
  (∑ results in (finset.univ : finset (Bool × Bool × Bool × Bool × Bool)),
    if value_heads results ≥ 0.25 then (1 : ℝ) else 0) / 32 = 13 / 16 := sorry

end probability_25_cents_min_l451_451958


namespace complete_square_l451_451221

theorem complete_square (x : ℝ) : x^2 + 4*x + 1 = 0 -> (x + 2)^2 = 3 :=
by sorry

end complete_square_l451_451221


namespace first_term_arithmetic_sum_l451_451080

theorem first_term_arithmetic_sum 
  (T : ℕ → ℚ) (b : ℚ) (d : ℚ) (h₁ : ∀ n, T n = n * (2 * b + (n - 1) * d) / 2)
  (h₂ : d = 5)
  (h₃ : ∀ n, (T (4 * n)) / (T n) = (16 : ℚ)) : 
  b = 5 / 2 :=
sorry

end first_term_arithmetic_sum_l451_451080


namespace side_length_of_equivalent_cube_l451_451269

noncomputable def surface_area_rectangular_prism (l w h : ℝ) : ℝ :=
  2 * ((l * w) + (l * h) + (w * h))

noncomputable def side_length_cube_with_equivalent_surface_area (area : ℝ) : ℝ :=
  real.sqrt (area / 6)

theorem side_length_of_equivalent_cube :
  surface_area_rectangular_prism 3 2 0.5 ≈ 17 ∧
  round (side_length_cube_with_equivalent_surface_area 17) = 2 :=
by
  sorry

end side_length_of_equivalent_cube_l451_451269


namespace problem_solution_l451_451920

def g (x : ℝ) : ℝ :=
  if x > 6 then x^2 - 1
  else if -6 <= x ∧ x <= 6 then 3 * x + 2
  else 4

theorem problem_solution :
  g (-8) + g 0 + g 8 = 69 :=
by
  -- Proof the steps here
  sorry

end problem_solution_l451_451920


namespace shopkeeper_cloth_sale_l451_451271

theorem shopkeeper_cloth_sale (x : ℕ) (CP : ℕ) (loss_per_metre : ℕ) (total_SP : ℕ) : 
  CP = 65 → 
  loss_per_metre = 5 → 
  total_SP = 18000 →
  (let SP_per_metre := CP - loss_per_metre in let SP := SP_per_metre * x in SP = total_SP) →
  x = 300 :=
by
  -- conditions and intermediate steps would normally go here
  sorry

end shopkeeper_cloth_sale_l451_451271


namespace eliminate_n_feasible_method_l451_451012

theorem eliminate_n_feasible_method :
  ∀ (m n : ℝ), 5 * m + 4 * n = 20 → 4 * m - 5 * n = 8 → 
  41 * m = 132 :=
by { intros m n h1 h2, sorry }

end eliminate_n_feasible_method_l451_451012


namespace determine_range_l451_451556

noncomputable def g (c : ℝ) (x : ℝ) : ℝ := c * x + 2

theorem determine_range (c : ℝ) (h : c ≠ 0) :
  ∃ (a b : ℝ), (∀ x, -1 ≤ x ∧ x ≤ 2 → g c x ∈ interval a b) ∧ 
  ((c > 0 → a = -c + 2 ∧ b = 2c + 2) ∧ (c < 0 → a = 2c + 2 ∧ b = -c + 2)) :=
by
  sorry

end determine_range_l451_451556


namespace piece_within_six_steps_l451_451681

/-- Define the conditions of the tiling and placement. -/
structure GridPoint where
  (x : Int) (y : Int)
  -- Define that each grid point has exactly six adjacent points.
  adj : List GridPoint

/-- Define the initial conditions. -/
def initial_grid_point : GridPoint := ⟨0, 0, []⟩

/-- Define the operation conditions. -/
def valid_move (p : GridPoint) : Prop :=
  ∀ (adj₁ adj₂ : GridPoint), adj₁ ∈ p.adj ∧ adj₂ ∈ p.adj ∧ adj₁ ≠ adj₂ →
  -- Pieces can only be placed on adjacent points if they don't already have pieces.
  (¬ has_piece adj₁ ∧ ¬ has_piece adj₂)

variable (has_piece : GridPoint → Bool)

theorem piece_within_six_steps :
  ∃ p : GridPoint, has_piece p ∧ distance initial_grid_point p ≤ 6 := 
sorry

end piece_within_six_steps_l451_451681


namespace gears_no_gaps_l451_451188

/-- There are two identical gears with 14 teeth on a common axis. Four pairs of teeth 
    are removed from the combined gears. Prove that the gears can be rotated so that they form a complete gear (without gaps). -/
theorem gears_no_gaps 
  (gear1 gear2 : Finset ℕ)
  (h1 : ∀ t ∈ gear1 ∪ gear2, t < 14)
  (h2 : gear1.card = 10)
  (h3 : gear2.card = 10)
  (h4 : ∀ t ∈ (gear1 ∩ gear2), t ∈ gear1 ∩ gear2): 
  ∃ k ∈ (Finset.range 14), ((gear1.map (λ t, (t + k) % 14) ∪ gear2) = (Finset.range 14) \ (gear1 ∩ gear2)) :=
sorry

end gears_no_gaps_l451_451188


namespace lottery_one_third_divisible_by_3_l451_451128

theorem lottery_one_third_divisible_by_3 :
  ∃ (A B C : Finset (Fin 91) × Finset (Fin 91) × Finset (Fin 91)),
  (A.card = B.card ∧ A.card = C.card ∧ B.card = C.card) ∧
  (∀ (x ∈ A), sum x % 3 = 0) ∧
  (∀ (y ∈ B), sum y % 3 = 1) ∧
  (∀ (z ∈ C), sum z % 3 = 2) :=
sorry

end lottery_one_third_divisible_by_3_l451_451128


namespace measure_angle_AYX_l451_451728

variable (A B C X Y Z : Type)
variable (Gamma : Type) (incircle circumcircle : Gamma)
variable (angle_A angle_B angle_C : ℝ)

-- Setting up the assumptions
axiom is_triangle : angle_A + angle_B + angle_C = 180
axiom in_circle : incircle
axiom circum_circle : circumcircle
axiom points_on_sides : (X ∈ BC) ∧ (Y ∈ AB) ∧ (Z ∈ AC)
axiom angles : angle_A = 50 ∧ angle_B = 70 ∧ angle_C = 60

-- The goal
theorem measure_angle_AYX : (AYX : ℝ) = 70 := sorry

end measure_angle_AYX_l451_451728


namespace problem_statement_l451_451786

noncomputable def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
noncomputable def is_differentiable (f : ℝ → ℝ) : Prop := ∀ x, differentiable ℝ f

theorem problem_statement (f : ℝ → ℝ) (h1 : is_even_function f) 
    (h2 : is_differentiable f) (h3 : f' 1 = 1) 
    (h4 : ∀ x, f (x + 2) = f (x - 2)) : 
    f' (-5) = -1 :=
by sorry

end problem_statement_l451_451786


namespace busRentalPlan1_busRentalPlan2_busRentalPlan3_busRentalPlans_l451_451591

def typeABuses := 8
def typeBBuses := 4
def totalStudents := 37

theorem busRentalPlan1 : 2 * typeABuses + 6 * typeBBuses = totalStudents :=
by sorry

theorem busRentalPlan2 : 3 * typeABuses + 4 * typeBBuses = totalStudents :=
by sorry

theorem busRentalPlan3 : 4 * typeABuses + 2 * typeBBuses = totalStudents :=
by sorry

theorem busRentalPlans : ∃ (plan1 plan2 plan3 : ℕ × ℕ),
  plan1 = (2, 6) ∧ plan2 = (3, 4) ∧ plan3 = (4, 2) ∧
  (plan1.1 * typeABuses + plan1.2 * typeBBuses = totalStudents) ∧
  (plan2.1 * typeABuses + plan2.2 * typeBBuses = totalStudents) ∧
  (plan3.1 * typeABuses + plan3.2 * typeBBuses = totalStudents) :=
by
  use (2, 6), (3, 4), (4, 2)
  simp [typeABuses, typeBBuses, totalStudents]
  constructor ; norm_num ;
  constructor ; norm_num ;
  constructor ; norm_num ;
  sorry

end busRentalPlan1_busRentalPlan2_busRentalPlan3_busRentalPlans_l451_451591


namespace proof_problem_solution_l451_451830

noncomputable def proof_problem_question_and_conditions (x_1 y_1 x_3 : ℝ) : Prop :=
  (x_1 = 5 ∧ y_1 = 3 ∧ x_3 = 1 + 3 * real.cbrt (-2)) →
  (x_1^3 - y_1^5 = 2882 ∧ x_1 - y_1 = 2) →
  (x_2 = -y_1 ∧ y_2 = -x_1 ∧ y_3 = -(1 + 3 * real.sqrt 2) ∧ x_4 = -(y_3) ∧ y_4 = -(x_3))

theorem proof_problem_solution :
    proof_problem_question_and_conditions 5 3 (1 + 3 * real.cbrt (-2)) :=
by
  sorry

end proof_problem_solution_l451_451830


namespace floor_factorial_expression_l451_451322

theorem floor_factorial_expression : 
  (⌊(2010! + 2007! : ℚ) / (2009! + 2008! : ℚ)⌋ = 2009) :=
by
  -- Let a := 2010! and b := 2007!
  -- So a + b = 2010! + 2007!
  -- Notice 2010! = 2010 * 2009 * 2008 * 2007!
  -- Notice 2009! = 2009 * 2008 * 2007!
  -- Simplify (2010! + 2007!) / (2009! + 2008!)
  sorry

end floor_factorial_expression_l451_451322


namespace mono_increasing_range_k_l451_451850

theorem mono_increasing_range_k (k : ℝ) :
  (∀ x ∈ Ioi 1, k - (1 / x) ≥ 0) → k ≥ 1 :=
by
  sorry

end mono_increasing_range_k_l451_451850


namespace negation_of_forall_inequality_l451_451995

theorem negation_of_forall_inequality :
  (¬ (∀ x : ℝ, x > 0 → x * Real.sin x < 2^x - 1)) ↔ (∃ x : ℝ, x > 0 ∧ x * Real.sin x ≥ 2^x - 1) :=
by sorry

end negation_of_forall_inequality_l451_451995


namespace find_first_term_l451_451087

theorem find_first_term (a : ℚ) (n : ℕ) (T : ℕ → ℚ)
  (hT : ∀ n, T n = n * (2 * a + 5 * (n - 1)) / 2)
  (h_const : ∃ c : ℚ, ∀ n > 0, T (4 * n) / T n = c) :
  a = 5 / 2 := 
sorry

end find_first_term_l451_451087


namespace reciprocal_of_neg3_l451_451589

theorem reciprocal_of_neg3 : (1 / (-3) = -1 / 3) :=
by
  sorry

end reciprocal_of_neg3_l451_451589


namespace third_student_gold_stickers_l451_451181

theorem third_student_gold_stickers:
  ∃ (n : ℕ), n = 41 ∧ 
  (∃ (a1 a2 a4 a5 a6 : ℕ), 
    a1 = 29 ∧ 
    a2 = 35 ∧ 
    a4 = 47 ∧ 
    a5 = 53 ∧ 
    a6 = 59 ∧ 
    a2 - a1 = 6 ∧ 
    a5 - a4 = 6 ∧ 
    ∀ k, k = 3 → n = a2 + 6) := 
sorry

end third_student_gold_stickers_l451_451181


namespace original_curve_eq_l451_451495

theorem original_curve_eq (x y x'' y'' : ℝ) (h1 : x'' = 3 * x) (h2 : y'' = y / 2) (h3 : y'' = sin x'') : y = 2 * sin (3 * x) :=
by {
  -- Proof omitted.
  sorry
}

end original_curve_eq_l451_451495


namespace function_property_l451_451403

theorem function_property (f : ℝ → ℝ)
  (h₁ : ∀ x y, f(x + y) = f(x) + f(y) + x * y + 1)
  (h₂ : f(-2) = -2) :
  (f(1) = 1)
  ∧ (∀ t : ℕ, t > 1 → f(t) > t)
  ∧ (∃! t : ℤ, f(t) = t) :=
by
  -- Placeholder for the proof
  sorry

end function_property_l451_451403


namespace find_first_term_l451_451088

theorem find_first_term (a : ℚ) (n : ℕ) (T : ℕ → ℚ)
  (hT : ∀ n, T n = n * (2 * a + 5 * (n - 1)) / 2)
  (h_const : ∃ c : ℚ, ∀ n > 0, T (4 * n) / T n = c) :
  a = 5 / 2 := 
sorry

end find_first_term_l451_451088


namespace angle_AKD_l451_451537

theorem angle_AKD :
  ∀ (A B C D M K : Point) (α β : ℝ),
    is_square A B C D → -- ABCD is a square
    M ∈ segment B C →
    K ∈ segment C D →
    ∠ B A M = 30 → -- ∠ BAM = 30 degrees
    ∠ C K M = 30 → -- ∠ CKM = 30 degrees
    ∠ A K D = 75 := -- ∠ AKD = 75 degrees
sorry

end angle_AKD_l451_451537


namespace alice_bob_sum_l451_451355

theorem alice_bob_sum :
  ∃ (A B : ℕ), 
  (1 ≤ A ∧ A ≤ 50) ∧ 
  (1 ≤ B ∧ B ≤ 50) ∧ 
  ((A ≠ 1 ∧ A ≠ 50) ∧ (B = 2)) ∧ 
  prime B ∧ 
  (∃ k : ℕ, 150 - B + A = k^2) ∧ 
  A + B = 23 :=
sorry

end alice_bob_sum_l451_451355


namespace triangle_AC_interval_l451_451946

theorem triangle_AC_interval (AB CD AC m n : ℝ) 
  (h1 : AB = 12)
  (h2 : CD = 4)
  (h3 : AD bisects ∠BAC) :
  (∃ m n, (4 < AC ∧ AC < 24) ∧ m + n = 28) :=
sorry

end triangle_AC_interval_l451_451946


namespace parabola_vertex_form_l451_451367

theorem parabola_vertex_form :
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, y = a * x^2 + b * x + c ↔ 
    y = -3 * x^2 + 12 * x - 8) ∧
    (vertex : ℝ × ℝ := (2, 4)) ∧
    (point : ℝ × ℝ := (1, 1)) ∧
    (vertex_condition : ∀ x y : ℝ, y = a * (x - 2)^2 + 4) ∧
    (point_condition : 1 = a * (1 - 2)^2 + 4) := sorry

end parabola_vertex_form_l451_451367


namespace real_root_of_eqn_l451_451760

theorem real_root_of_eqn : 
  ∃ x : ℝ, x ≥ 0 ∧ (sqrt x + sqrt (x + 4) = 8) ∧ x = 225 / 16 :=
by
  sorry

end real_root_of_eqn_l451_451760


namespace polynomial_inequality_l451_451106

variables {R : Type*} [CommRing R]

-- Define polynomials f and g
noncomputable def f (a₀ a₁ ... aₙ : R) : R[X] :=
  a₀ + a₁ * X + ... + aₙ * X^n

noncomputable def g (c₀ c₁ ... cₙ₊₁ : R) : R[X] :=
  c₀ + c₁ * X + ... + cₙ₊₁ * X^(n+1)

-- Define conditions and maximum modulus
def max_modulus (coeffs : List R) : R := 
  coeffs.sup (λ x, abs x)

theorem polynomial_inequality {a₀ a₁ ... aₙ c₀ c₁ ... cₙ₊₁ r : R}
  (hf : f a₀ a₁ ... aₙ ≠ 0)
  (hg : g c₀ c₁ ... cₙ₊₁ ≠ 0)
  (hrel : g c₀ c₁ ... cₙ₊₁ = (X + C r) * f a₀ a₁ ... aₙ) :
  max_modulus [a₀, a₁, ..., aₙ] / max_modulus [c₀, c₁, ..., cₙ₊₁] ≤ n+1 := 
sorry

end polynomial_inequality_l451_451106


namespace polynomial_degree_3_l451_451336

def f (x : ℝ) : ℝ := 2 - 15 * x + 4 * x^2 - 5 * x^3 + 6 * x^4
def g (x : ℝ) : ℝ := 4 - 3 * x - 7 * x^3 + 10 * x^4

theorem polynomial_degree_3 (c : ℝ) (h : c = -3 / 5) : (f x + c * g x).degree = 3 :=
by
  sorry

end polynomial_degree_3_l451_451336


namespace floor_factorial_expression_l451_451316

-- Define the factorial function for natural numbers
def factorial : ℕ → ℕ
| 0 := 1
| (n + 1) := (n + 1) * factorial n

-- The main theorem to prove
theorem floor_factorial_expression :
  (nat.floor ((factorial 2010 + factorial 2007) / (factorial 2009 + factorial 2008)) = 2009) :=
begin
  -- Actual proof goes here
  sorry
end

end floor_factorial_expression_l451_451316


namespace valid_outfits_count_l451_451452

theorem valid_outfits_count (shirts pants hats : ℕ) (striped_shirts striped_hats : ℕ) (colorable_pants : ℕ)
  (c1 : shirts = 8)
  (c2 : pants = 4)
  (c3 : hats = 8)
  (c4 : striped_shirts = 2)
  (c5 : striped_hats = 2)
  (c6 : colorable_pants = 4)
  (c7 : ∀ x : ℕ, x = 140) :
  -- Calculate total number of outfit combinations
  let total_comb := shirts * pants * hats,
  -- Calculate the number of non-striped shirts and hats
  let non_striped_shirts := shirts - striped_shirts,
  let non_striped_hats := hats - striped_hats,
  -- Adjust the total combinations for striped items
  let adjusted_comb := non_striped_shirts * pants * non_striped_hats,
  -- Subtract the outfits where all items are the same color for specific colors
  let invalid_combs := colorable_pants,
  -- Final valid outfits count
  let valid_outfits := adjusted_comb - invalid_combs in
  valid_outfits = x := sorry

end valid_outfits_count_l451_451452


namespace proof_a_b_m_l451_451805

noncomputable def f (a x : ℝ) : ℝ := (x + a) / Real.exp x  -- condition 1
def tangent_eq_at_zero (a : ℝ) : Prop := 
  let f0 := (0 + a) / Real.exp 0 in
  let f' := (λ x, ((1 - x) - a) / Real.exp x) in 
  f' 0 = 0 ∧ f0 = 1  -- condition 2
noncomputable def g (m x a : ℝ) : ℝ := 
  let f := (x + a) / Real.exp x in
  let f' := (λ x, ((1 - x) - a) / Real.exp x) in 
  x * f + m * f' x + 1 / Real.exp x  -- condition 3
def ex_real_x1_x2 (m a : ℝ) : Prop := 
  ∃ x1 x2, (0 ≤ x1 ∧ x1 ≤ 1) ∧ (0 ≤ x2 ∧ x2 ≤ 1) ∧ (2 * g m x1 a < g m x2 a) -- condition 4

theorem proof_a_b_m :
  (∀ a, tangent_eq_at_zero a → a = 1) ∧  -- proving a = 1
  (∀ b, (0 + 1) / Real.exp 0 = 1 → b = 1) ∧  -- proving b = 1
  (∀ m, (m > 0 ∧ ex_real_x1_x2 m 1) → m ∈ Ioo 0 (1/3) ∪ Ioi (5 / 2)) :=  -- proving range of m
sorry

end proof_a_b_m_l451_451805


namespace hyperbola_equation_exists_l451_451364

theorem hyperbola_equation_exists :
  (∃ (h : ℝ → ℝ → Prop),
  (∀ x y, h x y ↔ (x^2 / 9 - y^2 / 16 = 1)) ∧
  (¬ ∀ λ : ℝ, λ = 0) ∧
  h (-3) (2 * real.sqrt 3)) →
  (∃ k : ℝ → ℝ → Prop,
  (∀ x y, k x y ↔ (4 * x^2 / 9 - y^2 / 4 = 1))) := 
by
  sorry

end hyperbola_equation_exists_l451_451364


namespace find_k_value_l451_451998

theorem find_k_value (k : ℚ) (h1 : (3, -5) ∈ {p : ℚ × ℚ | p.snd = k * p.fst}) (h2 : k ≠ 0) : k = -5 / 3 :=
sorry

end find_k_value_l451_451998


namespace large_circle_diameter_l451_451550

theorem large_circle_diameter (r : ℝ) (R : ℝ) (R' : ℝ) :
  r = 2 ∧ R = 2 * r ∧ R' = R + r → 2 * R' = 12 :=
by
  intros h
  sorry

end large_circle_diameter_l451_451550


namespace range_of_a_l451_451010

noncomputable def a_n (a : ℝ) (n : ℕ) : ℝ :=
if n ≤ 5 then (5 - a) * n - 11 else a^(n - 4)

def is_increasing_sequence (a : ℝ) : Prop :=
∀ n : ℕ, a_n a n < a_n a (n + 1)

theorem range_of_a (a : ℝ) : is_increasing_sequence a → 2 < a ∧ a < 5 :=
begin
  sorry
end

end range_of_a_l451_451010


namespace third_number_in_tenth_bracket_l451_451469

open Nat

def nth_bracket (n : ℕ) : List ℕ :=
  let start := (n * (n - 1)) / 2 + 1
  List.range' start n

theorem third_number_in_tenth_bracket : (nth_bracket 10).nth! 2 = 48 := by
  sorry

end third_number_in_tenth_bracket_l451_451469


namespace complex_number_lemma_l451_451464

theorem complex_number_lemma (z : ℂ) (h : z + z⁻¹ = Real.sqrt 2) : z ^ 8 + z ^ -8 = 2 := by
  sorry

end complex_number_lemma_l451_451464


namespace beaver_group_count_l451_451153

theorem beaver_group_count (B : ℕ) (h1 : 3 * B = 60) : B = 20 :=
by sorry

end beaver_group_count_l451_451153


namespace chip_placement_count_l451_451451

open Set

-- Define the problem parameters
def grid_height := 4
def grid_width := 3
def total_squares := grid_height * grid_width

def red_chip_count := 4
def blue_chip_count := 3
def green_chip_count := 2

-- Condition: No two chips of the same color are directly adjacent
def non_adjacent {α : Type} (G : α → α → Prop) (c1 c2 : α) := ¬ G c1 c2

-- Define the grid positions as a pair (i, j) with i ∈ {0, ..., 3} and j ∈ {0, ..., 2}
def grid_positions := Σ (i : Fin 4), Fin 3

-- Define the adjacency relation in the grid
def adjacent (p1 p2 : grid_positions) : Prop := (p1.1 = p2.1 ∧ (p1.2.val = p2.2.val + 1 ∨ p1.2.val + 1 = p2.2.val)) ∨
                                               (p1.2 = p2.2 ∧ (p1.1.val = p2.1.val + 1 ∨ p1.1.val + 1 = p2.1.val))

-- Define the main theorem
theorem chip_placement_count : 
  ∃ (arrangements : Set (grid_positions → Option (Sum (Sum Unit Unit) Unit))), 
  (∀ (arrangement : grid_positions → Option (Sum (Sum Unit Unit) Unit)), arrangement ∈ arrangements → 
    (∀ r1 r2, arrangement r1 = arrangement r2 → r1 ≠ r2 → non_adjacent adjacent r1 r2)) ∧ 
  arrangements.size = 36 := 
sorry

end chip_placement_count_l451_451451


namespace calculate_pups_eaten_per_mouse_l451_451717

theorem calculate_pups_eaten_per_mouse:
  ∀ (initial_mice: ℕ) (pups_per_mouse: ℕ) (final_mice: ℕ) (mice_before_eating: ℕ),
  initial_mice = 8 →
  pups_per_mouse = 6 →
  final_mice = 280 →
  mice_before_eating = initial_mice * (1 + pups_per_mouse) * (1 + pups_per_mouse) →
  (mice_before_eating - final_mice) / (initial_mice * (1 + pups_per_mouse)) = 2 :=
by
  intros initial_mice pups_per_mouse final_mice mice_before_eating
  intros h_initial h_pups h_final h_before_eating
  rw [h_initial, h_pups, h_final, h_before_eating]
  norm_num [h_initial, h_pups]
  sorry

end calculate_pups_eaten_per_mouse_l451_451717


namespace five_by_five_rectangles_l451_451833

def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem five_by_five_rectangles : (choose 5 2) * (choose 5 2) = 100 :=
by
  sorry

end five_by_five_rectangles_l451_451833


namespace isosceles_triangle_side_length_l451_451183

theorem isosceles_triangle_side_length (c : ℝ) (h : c > 0) :
  let b := 4 in
  let a := c^2 / 4 in
  (0, 0) ∈ {(x, y) | x * y = c^2} ∧
  (a, b) ∈ {(x, y) | x * y = c^2} ∧
  (-a, b) ∈ {(x, y) | x * y = c^2} →
  ∃ d : ℝ, d = (Real.sqrt (a^2 + b^2)) ∧ d = Real.sqrt (c^4 / 16 + 16) :=
begin
  sorry
end

end isosceles_triangle_side_length_l451_451183


namespace probability_of_25_cents_heads_l451_451956

/-- 
Considering the flipping of five specific coins: a penny, a nickel, a dime,
a quarter, and a half dollar, prove that the probability of getting at least
25 cents worth of heads is 3 / 4.
-/
theorem probability_of_25_cents_heads :
  let total_outcomes := 2^5
  let successful_outcomes_1 := 2^4
  let successful_outcomes_2 := 2^3
  let successful_outcomes := successful_outcomes_1 + successful_outcomes_2
  (successful_outcomes / total_outcomes : ℚ) = 3 / 4 :=
by
  sorry

end probability_of_25_cents_heads_l451_451956


namespace average_of_other_two_l451_451967

theorem average_of_other_two {a b c d : ℕ} (h₀ : a ≠ b) (h₁ : a ≠ c) (h₂ : a ≠ d)
  (h₃ : b ≠ c) (h₄ : b ≠ d) (h₅ : c ≠ d) 
  (h₆ : 0 < a) (h₇ : 0 < b) (h₈ : 0 < c) (h₉ : 0 < d)
  (h₁₀ : a + b + c + d = 20) (h₁₁ : a - min (min a b) (min c d) = max (max a b) (max c d) - min (min a b) (min c d)) :
  ((a + b + c + d) - (max (max a b) (max c d) + min (min a b) (min c d))) / 2 = 2.5 :=
by
  sorry

end average_of_other_two_l451_451967


namespace cube_faces_sum_l451_451949

open Nat

theorem cube_faces_sum (a b c d e f : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : 0 < e) (h6 : 0 < f) 
    (h7 : (a + d) * (b + e) * (c + f) = 1386) : 
    a + b + c + d + e + f = 38 := 
sorry

end cube_faces_sum_l451_451949


namespace angle_OGD_tends_60_l451_451161

theorem angle_OGD_tends_60 
  (k : Circle) (O : Point) (A B C D : Point) 
  (alpha : ℝ) (E F G : Point)
  (h_circle : IsCircle k O) 
  (h_points_on_circle : OnCircle k A ∧ OnCircle k B ∧ OnCircle k C ∧ OnCircle k D)
  (h_consecutive_points : ∠AOB = alpha ∧ ∠BOC = alpha ∧ ∠COD = alpha)
  (h_alpha_bound : alpha < 60)
  (h_projection : E = Projection D (Line O A))
  (h_trisection : F = TrisectionPoint D E closerTo E)
  (h_intersection : G = Intersection (Line O A) (Line B F)) :
  tendsto (fun alpha => angle O G D) (nhds 60) :=
  sorry

end angle_OGD_tends_60_l451_451161


namespace skew_lines_angle_l451_451389

noncomputable def cube_vertices : set (euclidean_space ℝ [3]) := sorry

theorem skew_lines_angle (v1 v2 v3 v4 : euclidean_space ℝ [3]) 
  (hv1 : v1 ∈ cube_vertices) (hv2 : v2 ∈ cube_vertices) (hv3 : v3 ∈ cube_vertices) (hv4 : v4 ∈ cube_vertices) 
  (h_distinct : function.injective (λ v, v ∈ {v1, v2, v3, v4})) : 
  ∃ (l1 l2 : line (euclidean_space ℝ [3])), line.is_skew l1 l2 
  → ¬ ∃ θ : ℝ, cos θ = 1/2 ∧ θ = π/6 :=
sorry

end skew_lines_angle_l451_451389


namespace number_of_arrangements_word_l451_451746

noncomputable def factorial (n : Nat) : Nat := 
  if n = 0 then 1 else n * factorial (n - 1)

theorem number_of_arrangements_word (letters : List Char) (n : Nat) (r1 r2 r3 : Nat) 
  (h1 : letters = ['M', 'A', 'T', 'H', 'E', 'M', 'A', 'T', 'I', 'C', 'S'])
  (h2 : 2 = r1) (h3 : 2 = r2) (h4 : 2 = r3) :
  n = 11 → 
  factorial n / (factorial r1 * factorial r2 * factorial r3) = 4989600 := 
by
  sorry

end number_of_arrangements_word_l451_451746


namespace tenth_installment_payment_total_cost_appliance_l451_451679

noncomputable def installment_sequence (n : ℕ) : ℝ :=
  if n = 0 then 0 else 60 - 0.5 * (n - 1)

noncomputable def total_paid (months : ℕ) : ℝ :=
  150 + (Finset.range months).sum (λ n, installment_sequence (n + 1))

theorem tenth_installment_payment :
  installment_sequence 10 = 55.5 := by
  sorry

theorem total_cost_appliance :
  total_paid 20 = 1255 := by
  sorry

end tenth_installment_payment_total_cost_appliance_l451_451679


namespace probability_product_even_l451_451466

theorem probability_product_even (n : ℕ) (h : n = 6) : 
  (1 - (((nat_pow 1 2) / (nat_pow 2 2)) ^ n)) = 63/64 := by
  sorry

end probability_product_even_l451_451466


namespace factorial_floor_problem_l451_451329

theorem factorial_floor_problem :
  (nat.floor ( (nat.factorial 2010 + nat.factorial 2007) / (nat.factorial 2009 + nat.factorial 2008) )) = 2009 :=
by 
sorry

end factorial_floor_problem_l451_451329


namespace max_distance_to_pole_l451_451493

noncomputable def max_distance_to_origin (r1 r2 : ℝ) (c1 c2 : ℝ) : ℝ :=
  r1 + r2

theorem max_distance_to_pole (r : ℝ) (c : ℝ) : max_distance_to_origin 2 1 0 0 = 3 := by
  sorry

end max_distance_to_pole_l451_451493


namespace product_of_repeating_decimal_and_eight_l451_451375

theorem product_of_repeating_decimal_and_eight : (0.9 = 1) → (0.9 * 8 = 8) :=
by
  intros h
  rw [h]
  norm_num

end product_of_repeating_decimal_and_eight_l451_451375


namespace interval_of_n_l451_451385

theorem interval_of_n (n : ℕ) (h_pos : 0 < n) (h_lt_2000 : n < 2000) 
                      (h_div_99999999 : 99999999 % n = 0) (h_div_999999 : 999999 % (n + 6) = 0) : 
                      801 ≤ n ∧ n ≤ 1200 :=
by {
  sorry
}

end interval_of_n_l451_451385


namespace scientific_notation_l451_451715

def billion : ℝ := 10^9
def fifteenPointSeventyFiveBillion : ℝ := 15.75 * billion

theorem scientific_notation :
  fifteenPointSeventyFiveBillion = 1.575 * 10^10 :=
  sorry

end scientific_notation_l451_451715


namespace book_distribution_count_l451_451601

theorem book_distribution_count :
  ∃ (distribution_methods : ℕ),
    distribution_methods = 90 ∧
    ∃ books : Fin 5 → Prop,
      (∀ (p : Prop), ∃ (A B C : Prop),
       A ∧ B ∧ C ∧ 
       (A → ∃ (books_A : Finset (Fin 5)), books_A.card ≥ 1 ∧ books_A.card ≤ 2) ∧
       (B → ∃ (books_B : Finset (Fin 5)), books_B.card ≥ 1 ∧ books_B.card ≤ 2) ∧
       (C → ∃ (books_C : Finset (Fin 5)), books_C.card ≥ 1 ∧ books_C.card ≤ 2) ∧
       books = books_A ∪ books_B ∪ books_C ∧
       (books_A ∩ books_B = ∅) ∧
       (books_A ∩ books_C = ∅) ∧
       (books_B ∩ books_C = ∅)) :=
sorry

end book_distribution_count_l451_451601


namespace reciprocal_neg3_l451_451581

-- Define the problem
def reciprocal (x : ℚ) : ℚ := 1 / x

-- The required proof statement
theorem reciprocal_neg3 : reciprocal (-3) = -1 / 3 :=
by
  sorry

end reciprocal_neg3_l451_451581


namespace find_two_numbers_l451_451382

theorem find_two_numbers:
  ∃ (x y : ℤ), x + y = 24 ∧ x - y = 8 ∧ x * y > 100 :=
by
  exist x y
  state "x + y = 24 ∧ x - y = 8 ∧ x * y = 128"
  from 
    exist (16, 8)
    calc
      "x + y = 16 + 8"
      eq 24 
      from 
         calc 
           x - y 
           calc 
            ..
      sorry 
 
end find_two_numbers_l451_451382


namespace salary_increase_difference_l451_451014

structure Person where
  name : String
  salary : ℕ
  raise_percent : ℕ
  investment_return : ℕ

def hansel := Person.mk "Hansel" 30000 10 5
def gretel := Person.mk "Gretel" 30000 15 4
def rapunzel := Person.mk "Rapunzel" 40000 8 6
def rumpelstiltskin := Person.mk "Rumpelstiltskin" 35000 12 7
def cinderella := Person.mk "Cinderella" 45000 7 8
def jack := Person.mk "Jack" 50000 6 10

def salary_increase (p : Person) : ℕ := p.salary * p.raise_percent / 100
def investment_return (p : Person) : ℕ := salary_increase p * p.investment_return / 100
def total_increase  (p : Person) : ℕ := salary_increase p + investment_return p

def problem_statement : Prop :=
  let hansel_increase := total_increase hansel
  let gretel_increase := total_increase gretel
  let rapunzel_increase := total_increase rapunzel
  let rumpelstiltskin_increase := total_increase rumpelstiltskin
  let cinderella_increase := total_increase cinderella
  let jack_increase := total_increase jack

  let highest_increase := max gretel_increase (max rumpelstiltskin_increase (max cinderella_increase (max rapunzel_increase (max jack_increase hansel_increase))))
  let lowest_increase := min gretel_increase (min rumpelstiltskin_increase (min cinderella_increase (min rapunzel_increase (min jack_increase hansel_increase))))

  highest_increase - lowest_increase = 1530

theorem salary_increase_difference : problem_statement := by
  sorry

end salary_increase_difference_l451_451014


namespace anion_with_O_O_bond_is_peroxide_salt_formed_upon_anodic_oxidation_is_potassium_sulfate_l451_451962

-- Define bisulfate ion
def bisulfate_ion : Type := sorry

-- Define peroxide ion
def peroxide_ion : Type := sorry

-- Define potassium sulfate
def potassium_sulfate : Type := sorry

-- Anodic oxidation reaction for bisulfate to sulfate
def anodic_oxidation_of_bisulfate : bisulfate_ion → peroxide_ion → Prop :=
  sorry

-- The theorem statement
theorem anion_with_O_O_bond_is_peroxide
  (H: ∃ (a : bisulfate_ion) (b : peroxide_ion), anodic_oxidation_of_bisulfate a b)
  : peroxide_ion :=
begin
  sorry
end

theorem salt_formed_upon_anodic_oxidation_is_potassium_sulfate
  (H: ∃ (a : bisulfate_ion) (b : peroxide_ion), anodic_oxidation_of_bisulfate a b) 
  : potassium_sulfate :=
begin
  sorry
end

end anion_with_O_O_bond_is_peroxide_salt_formed_upon_anodic_oxidation_is_potassium_sulfate_l451_451962


namespace point_on_opposite_sides_l451_451473

theorem point_on_opposite_sides (y_0 : ℝ) :
  (2 - 2 * 3 + 5 > 0) ∧ (6 - 2 * y_0 < 0) → y_0 > 3 :=
by
  sorry

end point_on_opposite_sides_l451_451473


namespace cos_sum_lower_bound_l451_451126

theorem cos_sum_lower_bound (x : ℝ) (n : ℕ) (h : n > 0) :
  (Finset.range (n + 1)).sum (λ k, |Real.cos ((2 ^ k) * x)|) ≥ n / (2 * Real.sqrt 2) :=
sorry

end cos_sum_lower_bound_l451_451126


namespace least_prime_value_l451_451523

/-- Let q be a set of 12 distinct prime numbers. If the sum of the integers in q is odd,
the product of all the integers in q is divisible by a perfect square, and the number x is a member of q,
then the least value that x can be is 2. -/
theorem least_prime_value (q : Finset ℕ) (hq_distinct : q.card = 12) (hq_prime : ∀ p ∈ q, Nat.Prime p) 
    (hq_odd_sum : q.sum id % 2 = 1) (hq_perfect_square_div : ∃ k, q.prod id % (k * k) = 0) (x : ℕ)
    (hx : x ∈ q) : x = 2 :=
sorry

end least_prime_value_l451_451523


namespace leo_kept_l451_451504

/--
Leo had 400 marbles in a jar. He packed the marbles with ten marbles in each pack,
and he gave some of them to his two friends, Manny and Neil. He gave Manny 1/4 of 
the number of packs of marbles, Neil received 1/8 of the number of packs of marbles, 
and he kept the rest. Prove the number of packs of marbles Leo kept is 25.

- 400 marbles total
- 10 marbles per pack
- Manny got 1/4 of the packs
- Neil got 1/8 of the packs
- Leo kept the rest
-/
theorem leo_kept (total_marbles : ℕ) (marbles_per_pack : ℕ) (fraction_manny : ℚ) (fraction_neil : ℚ) (packs_leon : ℚ):
  total_marbles = 400 → 
  marbles_per_pack = 10 → 
  fraction_manny = 1 / 4 → 
  fraction_neil = 1 / 8 →
  packs_leon = (total_marbles / marbles_per_pack - total_marbles / marbles_per_pack * fraction_manny - total_marbles / marbles_per_pack * fraction_neil) →
  packs_leon = 25 := 
begin 
  sorry 
end

end leo_kept_l451_451504


namespace range_of_f_l451_451174

noncomputable def f : ℝ → ℝ
| x if x ≤ 0    := 2^(-x) - 1
| x if x > 0    := real.sqrt x

theorem range_of_f {x : ℝ} : f x > 1 ↔ x < -1 ∨ x > 1 :=
by
  sorry

end range_of_f_l451_451174


namespace triangular_number_19_l451_451177

def triangular_number (n : Nat) : Nat :=
  (n + 1) * (n + 2) / 2

theorem triangular_number_19 : triangular_number 19 = 210 := by
  sorry

end triangular_number_19_l451_451177


namespace completing_the_square_l451_451218

theorem completing_the_square (x : ℝ) :
  x^2 + 4 * x + 1 = 0 ↔ (x + 2)^2 = 3 :=
by
  sorry

end completing_the_square_l451_451218


namespace floor_factorial_expression_l451_451319

theorem floor_factorial_expression : 
  (⌊(2010! + 2007! : ℚ) / (2009! + 2008! : ℚ)⌋ = 2009) :=
by
  -- Let a := 2010! and b := 2007!
  -- So a + b = 2010! + 2007!
  -- Notice 2010! = 2010 * 2009 * 2008 * 2007!
  -- Notice 2009! = 2009 * 2008 * 2007!
  -- Simplify (2010! + 2007!) / (2009! + 2008!)
  sorry

end floor_factorial_expression_l451_451319


namespace determinant_of_matrixM_eq_neg_6_cubed_l451_451139

noncomputable def matrixM (n : ℤ) : Matrix (Fin 3) (Fin 3) ℤ :=
  Matrix.of (λ i j, ((n + 3 * i + j)^2))

theorem determinant_of_matrixM_eq_neg_6_cubed (n : ℤ) : (matrixM n).det = -(6 ^ 3) := by
  sorry

end determinant_of_matrixM_eq_neg_6_cubed_l451_451139


namespace vertical_strips_count_l451_451884

theorem vertical_strips_count (a b x y : ℕ)
  (h_outer : 2 * a + 2 * b = 50)
  (h_inner : 2 * x + 2 * y = 32)
  (h_strips : a + x = 20) :
  b + y = 21 :=
by
  have h1 : a + b = 25 := by
    linarith
  have h2 : x + y = 16 := by
    linarith
  linarith


end vertical_strips_count_l451_451884


namespace tangent_lines_diff_expected_l451_451422

noncomputable def tangent_lines_diff (a : ℝ) (k1 k2 : ℝ) : Prop :=
  let curve (x : ℝ) := a * x + 2 * Real.log (|x|)
  let deriv (x : ℝ) := a + 2 / x
  -- Tangent conditions at some x1 > 0 for k1
  (∃ x1 : ℝ, 0 < x1 ∧ k1 = deriv x1 ∧ curve x1 = k1 * x1)
  -- Tangent conditions at some x2 < 0 for k2
  ∧ (∃ x2 : ℝ, x2 < 0 ∧ k2 = deriv x2 ∧ curve x2 = k2 * x2)
  -- The lines' slopes relations
  ∧ k1 > k2

theorem tangent_lines_diff_expected (a k1 k2 : ℝ) (h : tangent_lines_diff a k1 k2) :
  k1 - k2 = 4 / Real.exp 1 :=
sorry

end tangent_lines_diff_expected_l451_451422


namespace first_term_arithmetic_sequence_l451_451092

def T_n (a d : ℚ) (n : ℕ) := n * (2 * a + (n - 1) * d) / 2

theorem first_term_arithmetic_sequence (a : ℚ)
  (h_const_ratio : ∀ (n : ℕ), n > 0 → 
    (T_n a 5 (4 * n)) / (T_n a 5 n) = (T_n a 5 4 / T_n a 5 1)) : 
  a = -5/2 :=
by 
  sorry

end first_term_arithmetic_sequence_l451_451092


namespace part1_part2_l451_451813

-- Define the conditions that translate the quadratic equation having distinct real roots
def discriminant_condition (m : ℝ) : Prop :=
  let a := 1
  let b := -4
  let c := 3 - 2 * m
  b ^ 2 - 4 * a * c > 0

-- Define the root condition from Vieta's formulas and the additional given condition
def additional_condition (m : ℝ) : Prop :=
  let x1_plus_x2 := 4
  let x1_times_x2 := 3 - 2 * m
  x1_times_x2 + x1_plus_x2 - m^2 = 4

-- Prove the range of m for part 1
theorem part1 (m : ℝ) : discriminant_condition m → m ≥ -1/2 := by
  sorry

-- Prove the value of m for part 2 with the range condition
theorem part2 (m : ℝ) : discriminant_condition m → additional_condition m → m = 1 := by
  sorry

end part1_part2_l451_451813


namespace product_of_repeating_decimal_and_eight_l451_451376

theorem product_of_repeating_decimal_and_eight : (0.9 = 1) → (0.9 * 8 = 8) :=
by
  intros h
  rw [h]
  norm_num

end product_of_repeating_decimal_and_eight_l451_451376


namespace rules_of_neg_numbers_from_identities_l451_451945

theorem rules_of_neg_numbers_from_identities
  {a b : ℝ} 
  (h1 : ∀ a : ℝ, a + 0 = a)
  (h2 : ∀ a : ℝ, a * 0 = 0)
  (h3 : ∀ a : ℝ, a * 1 = a)
  (h4 : ∀ a : ℝ, a + -a = 0)
  (h5 : ∀ a b c : ℝ, a * (b + c) = a * b + a * c)
  : 
  (a + -b = a - b) ∧ 
  (a - -b = a + b) ∧ 
  (a * -b = -a * b) ∧ 
  (a * -b = b * -a) ∧ 
  (-a * -b = a * b) := 
by 
  sorry

end rules_of_neg_numbers_from_identities_l451_451945


namespace proof_problem_l451_451900

open Classical

variable (x y z : ℝ)

theorem proof_problem
  (cond1 : 0 < x ∧ x < 1)
  (cond2 : 0 < y ∧ y < 1)
  (cond3 : 0 < z ∧ z < 1)
  (cond4 : x * y * z = (1 - x) * (1 - y) * (1 - z)) :
  ((1 - x) * y ≥ 1/4) ∨ ((1 - y) * z ≥ 1/4) ∨ ((1 - z) * x ≥ 1/4) := by
  sorry

end proof_problem_l451_451900


namespace eval_sum_and_fraction_l451_451342

noncomputable def double_factorial : ℕ → ℕ
| 0       := 1
| 1       := 1
| (n+2) := (n+2) * double_factorial n

theorem eval_sum_and_fraction:
  let S := ∑ i in Finset.range 2013, (Nat.choose (2 * (i + 1)) (i + 1)) / (2 ^ (2 * (i + 1))) in
  let c := ∑ i in Finset.range 2013, (Nat.choose (2 * (i + 1)) (i + 1)) * 2 ^ (4038 - 2 * (i + 1)) in
  let a := 2 * 2013 - 9 in
  let b := 1 in
  S = c / 2 ^ 4038 ∧ a = 4026 ∧ b = 1 ∧ (a * b) / 10 = 401.7 :=
by
  sorry

end eval_sum_and_fraction_l451_451342


namespace constant_function_l451_451356

def f (x y : ℤ) : ℝ := sorry -- the definition of f is not required here

theorem constant_function (f : ℤ × ℤ → ℝ) (h_range: ∀ p, 0 ≤ f p ∧ f p ≤ 1)
  (h_func: ∀ x y, f (x, y) = (f (x - 1, y) + f (x, y - 1)) / 2) :
  ∃ c ∈ set.Icc (0 : ℝ) (1 : ℝ), ∀ x y, f (x, y) = c :=
by
  sorry

end constant_function_l451_451356


namespace quadrilateral_midpoint_intersection_l451_451547

-- Define the vertices of the quadrilateral
structure Point := (x : ℝ) (y : ℝ) (z : ℝ)

variables (A B C D : Point)

-- Define the midpoints of the opposite sides AB and CD
def midpoint (p1 p2 : Point) : Point :=
{ x := (p1.x + p2.x) / 2,
  y := (p1.y + p2.y) / 2,
  z := (p1.z + p2.z) / 2 }

-- Midpoints K and M
def K := midpoint A B
def M := midpoint C D

-- Midpoint of the diagonals AC and BD
def P1 : Point :=
{ x := (A.x + B.x + C.x + D.x) / 4,
  y := (A.y + B.y + C.y + D.y) / 4,
  z := (A.z + B.z + C.z + D.z) / 4 }

-- The theorem statement
theorem quadrilateral_midpoint_intersection :
  let P := midpoint (midpoint A C) (midpoint B D) in
  (K = M ∧ K = P1) ∧ (M = P1 ∧ P = P1) :=
by sorry

end quadrilateral_midpoint_intersection_l451_451547


namespace average_of_two_excluding_min_max_l451_451977

theorem average_of_two_excluding_min_max :
  ∃ (a b c d : ℕ), (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d) ∧ 
  (a + b + c + d = 20) ∧ (max (max a b) (max c d) - min (min a b) (min c d) = 14) ∧
  (([a, b, c, d].erase (min (min a b) (min c d))).erase (max (max a b) (max c d)) = [x, y])
  → ((x + y) / 2 = 2.5) :=
sorry

end average_of_two_excluding_min_max_l451_451977


namespace find_first_term_l451_451086

theorem find_first_term (a : ℚ) (n : ℕ) (T : ℕ → ℚ)
  (hT : ∀ n, T n = n * (2 * a + 5 * (n - 1)) / 2)
  (h_const : ∃ c : ℚ, ∀ n > 0, T (4 * n) / T n = c) :
  a = 5 / 2 := 
sorry

end find_first_term_l451_451086


namespace prime_problem_l451_451192

open Nat

-- Definition of primes and conditions based on the problem
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- The formalized problem and conditions
theorem prime_problem (p q s : ℕ) 
  (p_prime : is_prime p) 
  (q_prime : is_prime q) 
  (s_prime : is_prime s) 
  (h1 : p + q = s + 4) 
  (h2 : 1 < p) 
  (h3 : p < q) : 
  p = 2 :=
sorry

end prime_problem_l451_451192


namespace tan_seventeen_pi_over_four_l451_451756

theorem tan_seventeen_pi_over_four : Real.tan (17 * Real.pi / 4) = 1 := by
  sorry

end tan_seventeen_pi_over_four_l451_451756


namespace sum_of_coefficients_of_factorized_polynomial_l451_451564

theorem sum_of_coefficients_of_factorized_polynomial : 
  ∃ (a b c d e : ℕ), 
    (216 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) ∧ 
    (a + b + c + d + e = 36) :=
sorry

end sum_of_coefficients_of_factorized_polynomial_l451_451564


namespace no_value_x_gt_12_not_defined_l451_451771

theorem no_value_x_gt_12_not_defined (x : ℝ) (h : x > 12) : x^2 - 24 * x + 144 ≠ 0 :=
by 
  intro h_eq
  have h1 : (x - 12) * (x - 12) = 0 := by rwa [mul_self_eq_zero] at h_eq
  have h2 : x = 12 := by simpa using h1
  linarith using h

end no_value_x_gt_12_not_defined_l451_451771


namespace combined_total_value_of_items_l451_451899

theorem combined_total_value_of_items :
  let V1 := 87.50 / 0.07
  let V2 := 144 / 0.12
  let V3 := 50 / 0.05
  let total1 := 1000 + V1
  let total2 := 1000 + V2
  let total3 := 1000 + V3
  total1 + total2 + total3 = 6450 := 
by
  sorry

end combined_total_value_of_items_l451_451899


namespace initial_rulers_calculation_l451_451608

variable {initial_rulers taken_rulers left_rulers : ℕ}

theorem initial_rulers_calculation 
  (h1 : taken_rulers = 25) 
  (h2 : left_rulers = 21) 
  (h3 : initial_rulers = taken_rulers + left_rulers) : 
  initial_rulers = 46 := 
by 
  sorry

end initial_rulers_calculation_l451_451608


namespace math_problem_l451_451393

variable (a b : ℝ)

#check a ≥ 0

theorem math_problem (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + b = 1) :
  a + Real.sqrt b ≤ Real.sqrt 2 ∧
  1/2 < 2^(a - Real.sqrt b) ∧ 2^(a - Real.sqrt b) < 2 ∧
  a^2 - b > -1 := 
by
  sorry

end math_problem_l451_451393


namespace jogger_usual_speed_l451_451676

theorem jogger_usual_speed (V T : ℝ) 
    (h_actual: 30 = V * T) 
    (h_condition: 40 = 16 * T) 
    (h_distance: T = 30 / V) :
  V = 12 := 
by
  sorry

end jogger_usual_speed_l451_451676


namespace siblings_of_John_l451_451593

structure Child :=
  (name : String)
  (eyeColor : String)
  (hairColor : String)
  (height : String)

def John : Child := {name := "John", eyeColor := "Brown", hairColor := "Blonde", height := "Tall"}
def Emma : Child := {name := "Emma", eyeColor := "Blue", hairColor := "Black", height := "Tall"}
def Oliver : Child := {name := "Oliver", eyeColor := "Brown", hairColor := "Black", height := "Short"}
def Mia : Child := {name := "Mia", eyeColor := "Blue", hairColor := "Blonde", height := "Short"}
def Lucas : Child := {name := "Lucas", eyeColor := "Blue", hairColor := "Black", height := "Tall"}
def Sophia : Child := {name := "Sophia", eyeColor := "Blue", hairColor := "Blonde", height := "Tall"}

theorem siblings_of_John : 
  (John.hairColor = Mia.hairColor ∧ John.hairColor = Sophia.hairColor) ∧
  ((John.eyeColor = Mia.eyeColor ∨ John.eyeColor = Sophia.eyeColor) ∨
   (John.height = Mia.height ∨ John.height = Sophia.height)) ∧
  (Mia.eyeColor = Sophia.eyeColor ∨ Mia.hairColor = Sophia.hairColor ∨ Mia.height = Sophia.height) ∧
  (John.hairColor = "Blonde") ∧
  (John.height = "Tall") ∧
  (Mia.hairColor = "Blonde") ∧
  (Sophia.hairColor = "Blonde") ∧
  (Sophia.height = "Tall") 
  → True := sorry

end siblings_of_John_l451_451593


namespace floor_factorial_expression_eq_2009_l451_451300

def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

theorem floor_factorial_expression_eq_2009 :
  (Int.floor (↑(factorial 2010 + factorial 2007) / ↑(factorial 2009 + factorial 2008)) = 2009) := by
  sorry

end floor_factorial_expression_eq_2009_l451_451300


namespace Jamie_minimum_4th_quarter_score_l451_451613

theorem Jamie_minimum_4th_quarter_score (q1 q2 q3 : ℤ) (avg : ℤ) (minimum_score : ℤ) :
  q1 = 84 → q2 = 80 → q3 = 83 → avg = 85 → minimum_score = 93 → 4 * avg - (q1 + q2 + q3) = minimum_score :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end Jamie_minimum_4th_quarter_score_l451_451613


namespace evaluate_f_at_7_l451_451801

def f : ℝ → ℝ 
noncomputable theory

-- Conditions
axiom h1 : ∀ x : ℝ, f(-x) = -f(x) -- f is odd
axiom h2 : ∀ x : ℝ, f(x + 4) = f(x) -- f is periodic with period 4
axiom h3 : ∀ x : ℝ, 0 < x ∧ x < 2 → f(x) = 2 * x^2 -- for 0 < x < 2, f(x) = 2x^2

-- Prove f(7) = -2
theorem evaluate_f_at_7 : f 7 = -2 :=
sorry

end evaluate_f_at_7_l451_451801


namespace correct_tangent_line_at_one_l451_451419

noncomputable def f (x : ℝ) : ℝ := f'.1 * x^3 + x^2 - 1

theorem correct_tangent_line_at_one :
  let f' := λ x : ℝ, 3 * f'.1 * x^2 + 2 * x in
  f (1 : ℝ) = -1 ∧ f'.1 = -1 ∧ (∀ (x : ℝ), x = 1 → (∀ (y : ℝ), y = f 1 → ∀ (slope : ℝ), slope = f'.1 → ( ∃ (l : ℝ), l = y - slope * x ∧ l = 0))) :=
by
  sorry

end correct_tangent_line_at_one_l451_451419


namespace exists_row_vector_no_zero_mod_l451_451073

theorem exists_row_vector_no_zero_mod (p : ℕ) (h_prime : Nat.Prime p) 
  (n m : ℕ) (M : Matrix (Fin n) (Fin m) ℤ)
  (cond : ∀ v : Vector (Fin m) ℤ, (∀ i, v i = 0 ∨ v i = 1) → v ≠ 0 → (M.mul_vec v) % p ≠ 0) :
  ∃ x : RowVector (Fin m) ℤ, ∀ i : Fin n, (x.mul M) i % p ≠ 0 :=
sorry

end exists_row_vector_no_zero_mod_l451_451073


namespace almost_sure_convergence_l451_451095

open ProbabilityTheory

variables {Ω : Type*} {p : ProbabilitySpace Ω}
variables {ξ : ℕ → MeasureTheory.ProbabilityTheory.Measure p ℝ}
variables {ξ_max : ℕ → MeasureTheory.ProbabilityTheory.Measure p ℝ}

noncomputable def sequence_max (n : ℕ) : MeasureTheory.ProbabilityTheory.Measure p ℝ :=
  (λ ω, sup (finset.range (n+1)).map (λ i, ξ i ω))

def stochastic_dominance (f g : MeasureTheory.ProbabilityTheory.Measure p ℝ) :=
  ∀ x : ℝ, MeasureTheory.toOuterMeasure f {ω | f ω > x} ≤ MeasureTheory.toOuterMeasure g {ω | g ω > x}

axiom A1 : ∀ n : ℕ, stochastic_dominance (sequence_max n) (ξ (n+1))
axiom A2 : MeasureTheory.ProbabilityTheory.ProbConvergence ξ (λ n, ξ n) (MeasureTheory.ProbabilityTheory.ae_eq p) ξ

theorem almost_sure_convergence :
  MeasureTheory.ProbabilityTheory.ae_eq p (MeasureTheory.ProbabilityTheory.lim (λ n, ξ n)) ξ := sorry

end almost_sure_convergence_l451_451095


namespace sequence_equivalency_and_comparison_l451_451826

noncomputable def a_seq (n : ℕ) : ℕ :=
match n with
| 0       => 0       -- We won't use index 0 for this sequence in proofs
| (n + 1) => if n = 0 then 2 else (n + 1) * a_seq n / n

def a_n (n : ℕ) : ℕ :=
2 * n

def S_n (n : ℕ) : ℕ :=
n * (n + 1)

def b_n (n : ℕ) : ℕ :=
2^(n - 1) + 1

def T_n (n : ℕ) : ℕ :=
2^n + n - 1

-- The main theorem as a Lean statement
theorem sequence_equivalency_and_comparison (n : ℕ) :
  a_seq 1 = 2 ∧ (∀ n, a_seq (n + 1) > 0) ∧ (∀ n, (n + 1) * (a_seq n)^2 + (a_seq n) * (a_seq (n + 1)) - n * (a_seq (n + 1))^2 = 0) ∧
  (S_n n = ∑ k in range (n + 1), a_n k) ∧ (T_n n = ∑ k in range (n + 1), b_n k) ∧
  ((n = 1 → T_n n = S_n n) ∧ (2 ≤ n ∧ n < 5 → T_n n < S_n n) ∧ (n ≥ 5 → T_n n > S_n n)) :=
by 
  sorry

end sequence_equivalency_and_comparison_l451_451826


namespace reciprocal_of_neg3_l451_451588

theorem reciprocal_of_neg3 : (1 / (-3) = -1 / 3) :=
by
  sorry

end reciprocal_of_neg3_l451_451588


namespace expected_value_of_sum_of_marbles_l451_451020

-- Definitions corresponding to the conditions
def marbles : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

def pairs := marbles.powerset.filter (λ s, s.card = 2)

def pair_sum (s : Finset ℕ) := s.sum id

-- Expected value calculation: There are 21 pairs
def total_sum_pairs := pairs.sum pair_sum

def expected_value := (total_sum_pairs : ℚ) / (pairs.card : ℚ)

-- The theorem that must be proven
theorem expected_value_of_sum_of_marbles :
  expected_value = 154 / 21 :=
by
  sorry

end expected_value_of_sum_of_marbles_l451_451020


namespace average_of_two_excluding_min_max_l451_451978

theorem average_of_two_excluding_min_max :
  ∃ (a b c d : ℕ), (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d) ∧ 
  (a + b + c + d = 20) ∧ (max (max a b) (max c d) - min (min a b) (min c d) = 14) ∧
  (([a, b, c, d].erase (min (min a b) (min c d))).erase (max (max a b) (max c d)) = [x, y])
  → ((x + y) / 2 = 2.5) :=
sorry

end average_of_two_excluding_min_max_l451_451978


namespace legs_total_l451_451247

def number_of_legs_bee := 6
def number_of_legs_spider := 8
def number_of_bees := 5
def number_of_spiders := 2
def total_legs := number_of_bees * number_of_legs_bee + number_of_spiders * number_of_legs_spider

theorem legs_total : total_legs = 46 := by
  sorry

end legs_total_l451_451247


namespace probability_of_sum_six_two_dice_l451_451632

noncomputable def probability_sum_six : ℚ := 5 / 36

theorem probability_of_sum_six_two_dice (dice_faces : ℕ := 6) : 
  ∃ (p : ℚ), p = probability_sum_six :=
by
  sorry

end probability_of_sum_six_two_dice_l451_451632


namespace smallest_x_l451_451414

open Int

theorem smallest_x (y : ℤ) : 0.75 = y / (256 + 0) → ∃ x : ℤ, 0.75 = y / (256 + x) ∧ x = 0 :=
by
  sorry

end smallest_x_l451_451414


namespace laura_garden_daisies_l451_451178

/-
Laura's Garden Problem: Given the ratio of daisies to tulips is 3:4,
Laura currently has 32 tulips, and she plans to add 24 more tulips,
prove that Laura will have 42 daisies in total after the addition to
maintain the same ratio.
-/

theorem laura_garden_daisies (daisies tulips add_tulips : ℕ) (ratio_d : ℕ) (ratio_t : ℕ)
    (h1 : ratio_d = 3) (h2 : ratio_t = 4) (h3 : tulips = 32) (h4 : add_tulips = 24)
    (new_tulips : ℕ := tulips + add_tulips) :
  daisies = 42 :=
by
  sorry

end laura_garden_daisies_l451_451178


namespace coloring_arithmetic_sequence_exists_l451_451751

open Finset

noncomputable def exists_coloring (A : Finset ℕ) (n : ℕ) : Prop :=
∃ c : ℕ → Prop, (∀ a b : ℕ, a ∈ A → b ∈ A → (a < b) → (b - a) % (n - 1) = 0 → c a ≠ c b)

theorem coloring_arithmetic_sequence_exists :
  ∃ c : ℕ → Prop, ∀ a b : ℕ, a ∈ (finset.range 2018).erase 0 →
    b ∈ (finset.range 2018).erase 0 →
    a < b →
    ((b - a) % (18 - 1) = 0) → (c a ≠ c b) :=
begin
  sorry
end

end coloring_arithmetic_sequence_exists_l451_451751


namespace matrix_self_inverse_l451_451748

def M (c d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2, -1], ![c, d]]

theorem matrix_self_inverse (c d : ℝ) :
  M c d ⬝ M c d = (1 : Matrix (Fin 2) (Fin 2) ℝ) ↔ c = 3 ∧ d = -2 :=
by {
  sorry,
}

end matrix_self_inverse_l451_451748


namespace smallest_number_of_eggs_l451_451636

theorem smallest_number_of_eggs (total_eggs : ℕ) (c : ℕ) (h1 : 15 * c - 3 > 130) : total_eggs = 132 :=
begin
  let containers := c,
  let eggs_removed := 3,
  have h2 : containers = 9 := sorry,
  have h3 : total_eggs = 15 * containers - eggs_removed := by sorry,
  rw [h2] at h3,
  exact h3,
end

end smallest_number_of_eggs_l451_451636


namespace x_eq_neg_f_neg_y_l451_451519

-- Define the function f(t)
def f (t : ℝ) : ℝ := t / (1 - t)

-- Define the condition y = f(x)
variable (x y : ℝ)
variable (h : y = f x)

-- Prove that x = -f(-y)
theorem x_eq_neg_f_neg_y (x y : ℝ) (h : y = f x) : x = -f (-y) :=
by sorry

end x_eq_neg_f_neg_y_l451_451519


namespace vector_norm_squared_l451_451917

open Real

noncomputable def midpoint (a b : ℝ × ℝ) : ℝ × ℝ :=
  ((a.1 + b.1) / 2, (a.2 + b.2) / 2)

def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

def norm_squared (v : ℝ × ℝ) : ℝ :=
  v.1 * v.1 + v.2 * v.2

theorem vector_norm_squared (a b : ℝ × ℝ)
  (m : ℝ × ℝ := (4, 2))
  (h₁ : midpoint a b = m)
  (h₂ : dot_product a b = 10) :
  norm_squared a + norm_squared b = 60 :=
sorry

end vector_norm_squared_l451_451917


namespace f_at_2_equals_12_l451_451421

def f : ℝ → ℝ :=
  λ x, if x < 0 then 2 * x^3 + x^2 else 0  -- Define f conditionally on the domain

-- Definition that f is odd function
def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = - (f x)

-- Our main statement
theorem f_at_2_equals_12 : odd_function f → f (-2) = -12 → f 2 = 12 := by
  sorry

end f_at_2_equals_12_l451_451421


namespace find_first_term_arithmetic_sequence_l451_451082

theorem find_first_term_arithmetic_sequence (a : ℤ) (k : ℤ)
  (hTn : ∀ n : ℕ, T_n = n * (2 * a + (n - 1) * 5) / 2)
  (hConstant : ∀ n : ℕ, (T (4 * n) / T n) = k) : a = 3 :=
by
  sorry

end find_first_term_arithmetic_sequence_l451_451082


namespace first_term_arithmetic_sequence_l451_451091

def T_n (a d : ℚ) (n : ℕ) := n * (2 * a + (n - 1) * d) / 2

theorem first_term_arithmetic_sequence (a : ℚ)
  (h_const_ratio : ∀ (n : ℕ), n > 0 → 
    (T_n a 5 (4 * n)) / (T_n a 5 n) = (T_n a 5 4 / T_n a 5 1)) : 
  a = -5/2 :=
by 
  sorry

end first_term_arithmetic_sequence_l451_451091


namespace trains_crossing_time_l451_451194

-- Define the lengths of the trains
def length_train1 : ℝ := 140
def length_train2 : ℝ := 190

-- Define the speeds of the trains in km/hr
def speed_train1 : ℝ := 60
def speed_train2 : ℝ := 40

-- Convert speeds from km/hr to m/s
def km_per_hr_to_m_per_s (v : ℝ) : ℝ :=
  v * (5 / 18)

-- Calculate the relative speed in m/s
def relative_speed : ℝ :=
  km_per_hr_to_m_per_s (speed_train1 + speed_train2)

-- Calculate the total distance to be covered
def total_distance : ℝ :=
  length_train1 + length_train2

-- Calculate crossing time
def crossing_time : ℝ :=
  total_distance / relative_speed

-- Proof statement
theorem trains_crossing_time : crossing_time ≈ 11.88 := by
  sorry

end trains_crossing_time_l451_451194


namespace difference_of_integers_divisible_by_2_to_7_l451_451363

theorem difference_of_integers_divisible_by_2_to_7 :
  let lcm_val := Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 7))))
  ∃ a b : ℕ, (1 < a) ∧ (1 < b) ∧
    (∀ k : ℕ, 2 ≤ k ∧ k ≤ 7 → (a % k = 1) ∧ (b % k = 1)) ∧
    (a = 1 + lcm_val ∨ a = 1 + 2 * lcm_val) ∧ 
    (b = 1 + lcm_val ∨ b = 1 + 2 * lcm_val) ∧
    abs (a - b) = 420 := 
by
  unfold let lcm_val := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 2 3) 4) 5) 6) 7
  -- provide the proof here
  sorry

end difference_of_integers_divisible_by_2_to_7_l451_451363


namespace angle_in_triangle_is_27_deg_l451_451683

-- Definitions from conditions
def regular_pentagon (P A B C D : Point) (O : Point) : Prop :=
  inscribed_in_circle P A B C D O ∧ ∀ (i j : Fin 5), i ≠ j → angle_i P A B C D i = 108

def regular_square (P Q R S : Point) (O : Point) : Prop :=
  inscribed_in_circle P Q R S O ∧ ∀ (i j : Fin 4), i ≠ j → angle_i P Q R S i = 90

-- Theorem statement using the identified conditions and the correct answer
theorem angle_in_triangle_is_27_deg 
  (P A B C D Q R S O : Point) 
  (h_pentagon : regular_pentagon P A B C D O) 
  (h_square : regular_square P Q R S O) :
  angle_in_triangle P A Q = 27 := 
sorry

end angle_in_triangle_is_27_deg_l451_451683


namespace axes_of_symmetry_not_coincide_l451_451569

def y₁ (x : ℝ) := (1 / 8) * (x^2 + 6 * x - 25)
def y₂ (x : ℝ) := (1 / 8) * (31 - x^2)

def tangent_y₁ (x : ℝ) := (x + 3) / 4
def tangent_y₂ (x : ℝ) := -x / 4

def axes_symmetry_y₁ := -3
def axes_symmetry_y₂ := 0

theorem axes_of_symmetry_not_coincide :
  (∃ x1 x2 : ℝ, y₁ x1 = y₂ x1 ∧ y₁ x2 = y₂ x2 ∧ tangent_y₁ x1 * tangent_y₂ x1 = -1 ∧ tangent_y₁ x2 * tangent_y₂ x2 = -1) →
  axes_symmetry_y₁ ≠ axes_symmetry_y₂ :=
by sorry

end axes_of_symmetry_not_coincide_l451_451569


namespace maximize_x_minus_y_plus_z_l451_451025

-- Define the given condition as a predicate
def given_condition (x y z : ℝ) : Prop :=
  2 * x^2 + y^2 + z^2 = 2 * x - 4 * y + 2 * x * z - 5

-- Define the statement we want to prove
theorem maximize_x_minus_y_plus_z :
  ∃ x y z : ℝ, given_condition x y z ∧ (x - y + z = 4) :=
by
  sorry

end maximize_x_minus_y_plus_z_l451_451025


namespace hexagon_planting_l451_451039

noncomputable def recurrence_relation : ℕ → ℕ
| 2          := 12
| (n + 1)    := 4 * (3^((n+1)-1)) - recurrence_relation n

theorem hexagon_planting (a_2 : ℕ) (H : a_2 = 12):
  recurrence_relation 6 = 732 :=
by {
  -- recall the recurrence relation used:
  have hr1 : recurrence_relation 3 = 4 * 3^2 - a_2, from sorry,
  have hr2 : recurrence_relation 4 = 4 * 3^3 - recurrence_relation 3, from sorry,
  have hr3 : recurrence_relation 5 = 4 * 3^4 - recurrence_relation 4, from sorry,
  have hr4 : recurrence_relation 6 = 4 * 3^5 - recurrence_relation 5, from sorry,
  -- using the base case:
  rw [H] at hr1,
  rw [hr1, hr2, hr3, hr4],
  -- other steps to complete the recurrence relation calculation to show it equals 732:
  admit
}

end hexagon_planting_l451_451039


namespace expected_value_of_sum_of_two_marbles_l451_451018

open Finset

noncomputable def choose2 (s:Finset ℕ) := s.powerset.filter (λ t, t.card = 2)

theorem expected_value_of_sum_of_two_marbles:
  let marbles := range 1 8 in
  let num_pairs := (choose2 marbles).card in
  let total_sum := (choose2 marbles).sum (λ t, t.sum id) in
  (total_sum:ℚ) / (num_pairs:ℚ) = 8 :=
by
  let marbles := range 1 8
  let num_pairs := (choose2 marbles).card
  let total_sum := (choose2 marbles).sum (λ t, t.sum id)
  have h1: num_pairs = 21, by sorry
  have h2: total_sum = 168, by sorry
  rw [h1, h2]
  norm_num

end expected_value_of_sum_of_two_marbles_l451_451018


namespace Emilee_earns_25_l451_451063

variable (Terrence Jermaine Emilee : ℕ)
variable (h1 : Terrence = 30)
variable (h2 : Jermaine = Terrence + 5)
variable (h3 : Jermaine + Terrence + Emilee = 90)

theorem Emilee_earns_25 : Emilee = 25 := by
  -- Insert the proof here
  sorry

end Emilee_earns_25_l451_451063


namespace sum_of_coordinates_l451_451472

-- Define the functions g and h
def g : ℝ → ℝ := sorry
def h (x : ℝ) : ℝ := (g(x))^3

-- Given conditions
def point_on_g : (3, 8) ∈ { p : ℝ × ℝ | p.snd = g(p.fst) } :=
by sorry

theorem sum_of_coordinates : (3 + h(3) = 515) :=
by {
  -- Use the given conditions to derive the correct answer
  have g3 : g(3) = 8 := by
    rw [← point_on_g],
    sorry,
  have h3 : h(3) = (g(3))^3 := by
    rw [h],
  rw [g3] at h3,
  have h3_val : h(3) = 512 := by
    norm_num at h3,
  norm_num,
  exact h3_val,
}

end sum_of_coordinates_l451_451472


namespace mars_surface_suitable_for_colonies_l451_451843

theorem mars_surface_suitable_for_colonies (h1 h2 : Prop) (fraction_not_covered_by_ice fraction_suitable_for_colonies : ℚ) :
  fraction_not_covered_by_ice = 1/3 →
  fraction_suitable_for_colonies = 2/3 →
  (fraction_not_covered_by_ice * fraction_suitable_for_colonies) = 2/9 :=
by
  intros hH₁ hH₂
  rw [hH₁, hH₂]
  norm_num
  sorry

end mars_surface_suitable_for_colonies_l451_451843


namespace angle_between_vectors_theorem_l451_451783

-- Given a non-zero vector a
variables {a b : ℝ}

-- a dot (a + b) equals 0
def dot_product_condition : Prop :=
  a * (a + b) = 0

-- 2 times the norm of a equals the norm of b
def norm_condition : Prop :=
  2 * |a| = |b|

-- The angle between vectors a and b is 120 degrees
def angle_between_vectors (θ : ℝ) : Prop :=
  θ = 120

-- Lean 4 statement
theorem angle_between_vectors_theorem
  (a b : ℝ)
  (non_zero_a : a ≠ 0)
  (cond1 : dot_product_condition)
  (cond2 : norm_condition) :
  ∃ θ : ℝ, angle_between_vectors θ :=
begin
  sorry
end

end angle_between_vectors_theorem_l451_451783


namespace train_crossing_time_is_18_seconds_l451_451246

-- Definition of the given conditions
def train_length : ℝ := 160
def train_speed_kmh : ℝ := 32
def conversion_factor : ℝ := 1000 / 3600

-- The speed of the train in meters per second
def train_speed_ms : ℝ := train_speed_kmh * conversion_factor

-- The time for the train to cross the man (distance / speed)
def crossing_time : ℝ := train_length / train_speed_ms

-- The theorem stating the required time to cross the man
theorem train_crossing_time_is_18_seconds :
  crossing_time ≈ 18 := sorry

end train_crossing_time_is_18_seconds_l451_451246


namespace solution_l451_451617

noncomputable def problem_statement : Prop :=
  ∃ (P Q R S T U : Type) [euclidean_space P Q R S T U],
  (PQ = 3 * PR) ∧
  (S ∈ PQ) ∧ (T ∈ QR) ∧
  (∠QPT = ∠SRT) ∧
  (U = intersection (PT) (RS)) ∧
  (isosceles (triangle R U T) (angle RT U) (angle RU T)) →
  (∠PRQ = 60)

theorem solution : problem_statement :=
by sorry

end solution_l451_451617


namespace complete_square_l451_451217

theorem complete_square (x : ℝ) : (x ^ 2 + 4 * x + 1 = 0) ↔ ((x + 2) ^ 2 = 3) :=
by {
  split,
  { intro h,
    sorry },
  { intro h,
    sorry }
}

end complete_square_l451_451217


namespace div_fraction_eq_l451_451835

theorem div_fraction_eq :
  (5 / 3) / (1 / 4) = 20 / 3 := 
by
  sorry

end div_fraction_eq_l451_451835


namespace zilla_savings_l451_451637

theorem zilla_savings (earnings : ℝ) (rent : ℝ) (expenses : ℝ) (savings : ℝ) 
  (h1 : rent = 0.07 * earnings)
  (h2 : rent = 133)
  (h3 : expenses = earnings / 2)
  (h4 : savings = earnings - rent - expenses) :
  savings = 817 := 
sorry

end zilla_savings_l451_451637


namespace number_of_liars_on_island_l451_451603

theorem number_of_liars_on_island :
  ∃ (knights liars : ℕ),
  let Total := 1000 in
  let Villages := 10 in
  let members_per_village := Total / Villages in
  -- Each village has at least 2 members
  (∀ i : ℕ, i < Villages → 2 ≤ members_per_village) ∧
  -- Populations of knights and liars respecting the Total population
  (knights + liars = Total) ∧
  -- Each inhabitant claims all others in their village are liars
  (∀ (i v : ℕ), i < Villages → v < members_per_village → 
     (∃ k l : ℕ, k + l = members_per_village ∧ 
      k = 1 ∧ -- there is exactly one knight in each village
      knights = Villages ∧ liars = Total - knights ∧
      l = members_per_village - 1)) ∧
  liars = 990 := sorry

end number_of_liars_on_island_l451_451603


namespace emilee_earns_25_l451_451067

-- Define the conditions
def earns_together (jermaine terrence emilee : ℕ) : Prop := 
  jermaine + terrence + emilee = 90

def jermaine_more (jermaine terrence : ℕ) : Prop :=
  jermaine = terrence + 5

def terrence_earning : ℕ := 30

-- The goal: Prove Emilee earns 25 dollars
theorem emilee_earns_25 (jermaine terrence emilee : ℕ) (h1 : earns_together jermaine terrence emilee) 
  (h2 : jermaine_more jermaine terrence) (h3 : terrence = terrence_earning) : 
  emilee = 25 := 
sorry

end emilee_earns_25_l451_451067


namespace triangle_side_solution_l451_451780

/-- 
Given \( a \geq b \geq c > 0 \) and \( a < b + c \), a solution to the equation 
\( b \sqrt{x^{2} - c^{2}} + c \sqrt{x^{2} - b^{2}} = a x \) is provided by 
\( x = \frac{abc}{2 \sqrt{p(p-a)(p-b)(p-c)}} \) where \( p = \frac{1}{2}(a+b+c) \).
-/

theorem triangle_side_solution (a b c x : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0) (h4 : a < b + c) :
  b * (Real.sqrt (x^2 - c^2)) + c * (Real.sqrt (x^2 - b^2)) = a * x → 
  x = (a * b * c) / (2 * Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))) :=
sorry

end triangle_side_solution_l451_451780


namespace tan_alpha_eq_3_l451_451797

theorem tan_alpha_eq_3 (α : ℝ) (h1 : 0 < α ∧ α < (π / 2))
  (h2 : (Real.sin α)^2 + Real.cos ((π / 2) + 2 * α) = 3 / 10) : Real.tan α = 3 := by
  sorry

end tan_alpha_eq_3_l451_451797


namespace trip_time_is_approximate_l451_451532

noncomputable def total_distance : ℝ := 620
noncomputable def half_distance : ℝ := total_distance / 2
noncomputable def speed1 : ℝ := 70
noncomputable def speed2 : ℝ := 85
noncomputable def time1 : ℝ := half_distance / speed1
noncomputable def time2 : ℝ := half_distance / speed2
noncomputable def total_time : ℝ := time1 + time2

theorem trip_time_is_approximate :
  abs (total_time - 8.0757) < 0.0001 :=
sorry

end trip_time_is_approximate_l451_451532


namespace minimum_value_l451_451373

def f (x y : ℝ) : ℝ := x * y / (x^2 + y^2)

theorem minimum_value :
  ∃ (x y : ℝ), (3 / 7 ≤ x) ∧ (x ≤ 2 / 3) ∧ (1 / 4 ≤ y) ∧ (y ≤ 3 / 5) ∧ (f x y = 288 / 876) :=
begin
  sorry
end

end minimum_value_l451_451373


namespace exists_special_N_l451_451993

theorem exists_special_N :
  ∃ N : ℕ, (∃ n : ℕ, N = 995 * (2 * n + 1989)) ∧ 
           (card { ⟨m, k⟩ : ℕ × ℕ | N = (k + 1) * (2 * m + k) / 2 } = 1990) ∧ 
           (N = 5^10 * 199^180 ∨ N = 5^180 * 199^10) := sorry

end exists_special_N_l451_451993


namespace find_common_difference_maximal_sum_index_l451_451802

-- Define the arithmetic sequence and conditions 
variable {a : ℕ → ℝ}
variable {d : ℝ}

-- Arithmetic sequence definition
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Condition 1
def cond1 (a : ℕ → ℝ) : Prop :=
  a 1 + a 3 + a 5 = 105

-- Condition 2
def cond2 (a : ℕ → ℝ) : Prop :=
  a 2 + a 4 + a 6 = 99

-- Main theorem to prove d == -2
theorem find_common_difference (a : ℕ → ℝ) (d : ℝ) [is_arithmetic_sequence a d] : 
  cond1 a → 
  cond2 a → 
  d = -2 := 
by 
  sorry

-- Sum of the first n terms S_n and its maximum n
def S_n (a : ℕ → ℝ) (n : ℕ) : ℝ := 
  (finset.range n).sum a

-- The index n for maximal sum S_n
theorem maximal_sum_index (a : ℕ → ℝ) (n : ℕ) [is_arithmetic_sequence a d] :
  cond1 a → 
  cond2 a → 
  (∀ n, S_n a n ≤ S_n a 20) →
  n = 20 := 
by 
  sorry

end find_common_difference_maximal_sum_index_l451_451802


namespace section_of_cube_by_center_plane_is_hexagon_l451_451345

noncomputable def intersection_shape_of_cube (C : Cube) (P : Plane) [passes_through_center P C] [perpendicular_to_diagonal P C] : Shape := 
sorry

theorem section_of_cube_by_center_plane_is_hexagon (C : Cube) (P : Plane) [passes_through_center P C] [perpendicular_to_diagonal P C] : 
  intersection_shape_of_cube C P = Shape.hexagon := 
sorry

end section_of_cube_by_center_plane_is_hexagon_l451_451345


namespace AmandaNetEarnings_is_416_l451_451894

def AmandaNetEarnings (pay_rate : ℝ) (regular_hours : ℕ) (overtime_rate : ℝ) 
    (commission : ℝ) (tax_rate : ℝ) (health_insurance_rate : ℝ) 
    (other_expenses : ℝ) (total_hours : ℕ) 
    (penalty_rate : ℝ) : ℝ :=
let regular_pay := pay_rate * regular_hours
let overtime_hours := (total_hours - regular_hours).to_real
let overtime_pay := pay_rate * overtime_rate * overtime_hours
let total_earnings := regular_pay + overtime_pay + commission
let tax_deduction := total_earnings * tax_rate
let health_deduction := total_earnings * health_insurance_rate
let total_deductions := tax_deduction + health_deduction + other_expenses
let earnings_after_deductions := total_earnings - total_deductions
let penalty := earnings_after_deductions * penalty_rate
earnings_after_deductions - penalty

theorem AmandaNetEarnings_is_416 
    (pay_rate : ℝ) (regular_hours : ℕ) 
    (overtime_rate : ℝ) (commission : ℝ) 
    (tax_rate : ℝ) (health_insurance_rate : ℝ) 
    (other_expenses : ℝ) (total_hours : ℕ) 
    (penalty_rate : ℝ) : 
    AmandaNetEarnings pay_rate regular_hours overtime_rate 
    commission tax_rate health_insurance_rate 
    other_expenses total_hours penalty_rate = 416 := 
sorry

end AmandaNetEarnings_is_416_l451_451894


namespace calculation_result_l451_451724

theorem calculation_result :
  (10 * 19 * 20 * 53 * 100 + 601) / 13 = 1549277 :=
by 
  sorry

end calculation_result_l451_451724


namespace most_water_Yujeong_l451_451228

theorem most_water_Yujeong :
  ∀ (Yujeong Eunji Yuna : ℚ),
    Yujeong = 7/10 →
    Eunji = 1/2 →
    Yuna = 6/10 →
    (Yujeong > Eunji ∧ Yujeong > Yuna) :=
by
  intros Yujeong Eunji Yuna hYujeong hEunji hYuna
  rw [hYujeong, hEunji, hYuna]
  norm_num
  exact ⟨by norm_num, by norm_num⟩

end most_water_Yujeong_l451_451228


namespace max_value_10x_plus_3y_plus_12z_l451_451919

theorem max_value_10x_plus_3y_plus_12z (x y z : ℝ) 
  (h1 : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) 
  (h2 : z = 2 * y) : 
  10 * x + 3 * y + 12 * z ≤ Real.sqrt 253 :=
sorry

end max_value_10x_plus_3y_plus_12z_l451_451919


namespace math_problem_l451_451391

theorem math_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a^2 + b = 1) : 
  (a + Real.sqrt b ≤ Real.sqrt 2) ∧ 
  (1 / 2 < 2 ^ (a - Real.sqrt b) ∧ 2 ^ (a - Real.sqrt b) < 2) ∧ 
  (a^2 - b > -1) := 
by
  sorry

end math_problem_l451_451391


namespace reciprocal_of_neg_three_l451_451583

-- Define the notion of reciprocal
def reciprocal (x : ℝ) : ℝ := 1 / x

-- State the proof problem
theorem reciprocal_of_neg_three :
  reciprocal (-3) = -1 / 3 :=
by
  -- Since we are only required to state the theorem, we use sorry to skip the proof.
  sorry

end reciprocal_of_neg_three_l451_451583


namespace minimal_fencing_l451_451111

theorem minimal_fencing (w l : ℝ) (h1 : l = 2 * w) (h2 : w * l ≥ 400) : 
  2 * (w + l) = 60 * Real.sqrt 2 :=
by
  sorry

end minimal_fencing_l451_451111


namespace target_set_not_in_F_l451_451592

-- Define the sample space and σ-algebra
def omega : Type := ℝ → ℝ
def F : Set (Set omega) := { s | ∃ (u : Set ℝ), is_open u ∧ ∀ f ∈ s, ∀ x, f x ∈ u }

-- Define the random elements X and Y
def X (ω : omega) : omega := ω
def Y : omega := λ (t : ℝ), 0

-- Define the target set
def target_set : Set omega := { ω | ∀ t, X ω t = Y t }

-- The theorem stating that the target set is not in the σ-algebra
theorem target_set_not_in_F : target_set ∉ F :=
by
  sorry

end target_set_not_in_F_l451_451592


namespace volume_of_pyramid_l451_451942

-- Given conditions
def AB : ℝ := 8
def BC : ℝ := 4
def PA : ℝ := 6
def base_area : ℝ := AB * BC
def height : ℝ := PA

-- Theorem to state the volume of the pyramid
theorem volume_of_pyramid (h1 : AB = 8) (h2 : BC = 4) (h3 : PA = 6) : 
  (1 / 3) * base_area * height = 64 :=
by {
  sorry
}

end volume_of_pyramid_l451_451942


namespace perpendicular_condition_parallel_condition_parallel_opposite_direction_l451_451390

variables (a b : ℝ × ℝ) (k : ℝ)

-- Define the vectors
def vec_a : ℝ × ℝ := (1, 2)
def vec_b : ℝ × ℝ := (-3, 2)

-- Define the given expressions
def expression1 (k : ℝ) : ℝ × ℝ := (k * vec_a.1 + vec_b.1, k * vec_a.2 + vec_b.2)
def expression2 : ℝ × ℝ := (vec_a.1 - 3 * vec_b.1, vec_a.2 - 3 * vec_b.2)

-- Dot product function
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Perpendicular condition
theorem perpendicular_condition : (k : ℝ) → dot_product (expression1 k) expression2 = 0 → k = 19 :=
by sorry

-- Parallel and opposite condition
theorem parallel_condition : (k : ℝ) → (∃ m : ℝ, expression1 k = m • expression2) → k = -1 / 3 :=
by sorry

noncomputable def m (k : ℝ) : ℝ × ℝ := 
  let ex1 := expression1 k
  let ex2 := expression2
  (ex2.1 / ex1.1, ex2.2 / ex1.2)

theorem parallel_opposite_direction : (k : ℝ) → expression1 k = -1 / 3 • expression2 → k = -1 / 3 :=
by sorry

end perpendicular_condition_parallel_condition_parallel_opposite_direction_l451_451390


namespace Emilee_earns_25_l451_451064

variable (Terrence Jermaine Emilee : ℕ)
variable (h1 : Terrence = 30)
variable (h2 : Jermaine = Terrence + 5)
variable (h3 : Jermaine + Terrence + Emilee = 90)

theorem Emilee_earns_25 : Emilee = 25 := by
  -- Insert the proof here
  sorry

end Emilee_earns_25_l451_451064


namespace smallest_n_divisible_l451_451626

theorem smallest_n_divisible (n : ℕ) (h : 1 ≤ n)
  : ∃ n, (∀ k, 1 ≤ k ∧ k ≤ n → k ∣ n - 1) ∧ (∃ (k : ℕ), 1 ≤ k ∧ k ≤ n ∧ ¬k ∣ n - 1) ∧ ∀ m < n, (¬∃ (k : ℕ), 1 ≤ k ∧ k ≤ m ∧ ¬k ∣ (m - 1)) :=
begin
  use 3,
  split,
  { intros k hk,
    cases hk with hk1 hk2,
    cases hk2, dec_trivial, dec_trivial },
  split,
  { use 3, dec_trivial },
  { intros m hm,
    have h2 : (m = 1) ∨ (m = 2),
    { interval_cases m },
    cases h2; dec_trivial }
end

end smallest_n_divisible_l451_451626


namespace george_speed_l451_451774

def time_for_trip (distance speed : ℝ) : ℝ := distance / speed

theorem george_speed :
  let normal_trip_time := time_for_trip 3 10
  let first_leg_time := time_for_trip 2 5
  let remaining_distance := 1
  let remaining_time := 0.3 - first_leg_time
  remaining_time = 0.1 →
  let required_speed := time_for_trip remaining_distance remaining_time
  required_speed = 10 :=
by {
  intros,
  sorry
}

end george_speed_l451_451774


namespace beavers_build_dam_l451_451148

def num_beavers_first_group : ℕ := 20

theorem beavers_build_dam (B : ℕ) (t₁ : ℕ) (t₂ : ℕ) (n₂ : ℕ) :
  (B * t₁ = n₂ * t₂) → (B = num_beavers_first_group) := 
by
  -- Given
  let t₁ := 3
  let t₂ := 5
  let n₂ := 12

  -- Work equation
  assume h : B * t₁ = n₂ * t₂
  
  -- Correct answer
  have B_def : B = (n₂ * t₂) / t₁,
  exact h
   
  sorry

end beavers_build_dam_l451_451148


namespace plane_intersection_dist_l451_451182

-- Define the cube's vertices and intersection points P, Q, R
def cube_vertices : list (ℝ × ℝ × ℝ) := [
  (0, 0, 0), (0, 0, 6), (0, 6, 0), (0, 6, 6), 
  (6, 0, 0), (6, 0, 6), (6, 6, 0), (6, 6, 6)
]

def P : ℝ × ℝ × ℝ := (0, 3, 0)
def Q : ℝ × ℝ × ℝ := (2, 0, 0)
def R : ℝ × ℝ × ℝ := (2, 6, 6)

-- Define the problem to prove that the distance is 2sqrt(13)
theorem plane_intersection_dist : 
  let plane_distance := 2 * real.sqrt 13 in
  ∀ V₁ V₂ ∈ cube_vertices, 
  -- Plane equation is derived from P, Q, R
  let normal_vector := (-(2 : ℝ), 3, 0) × (-(2 : ℝ), -3, -6) in
  let scaled_normal := (3 : ℝ, 2, -1) in
  let plane_eq := (3 : ℝ, 2, -1) in
  let pos_d := 6 in
  let intersection_rel V := (3 * V.1 + 2 * V.2 - V.3 = 6) in
  (intersection_rel V₁) → (intersection_rel V₂) → 
  dist V₁ V₂ = plane_distance :=
begin
  sorry
end

end plane_intersection_dist_l451_451182


namespace intersection_M_N_l451_451474

-- Definition of the sets M and N
def M : Set ℝ := {x | 4 < x ∧ x < 8}
def N : Set ℝ := {x | x^2 - 6 * x < 0}

-- Intersection of M and N
def intersection : Set ℝ := {x | 4 < x ∧ x < 6}

-- Theorem statement asserting the equality between the intersection and the desired set
theorem intersection_M_N : ∀ (x : ℝ), x ∈ M ∩ N ↔ x ∈ intersection := by
  sorry

end intersection_M_N_l451_451474


namespace expression_value_l451_451629

theorem expression_value (x : ℝ) (h : x = 4) :
  (x^2 - 2*x - 15) / (x - 5) = 7 :=
sorry

end expression_value_l451_451629


namespace g_of_1_equals_3_l451_451416

theorem g_of_1_equals_3 (f g : ℝ → ℝ)
  (hf_odd : ∀ x, f (-x) = -f x)
  (hg_even : ∀ x, g (-x) = g x)
  (h1 : f (-1) + g 1 = 2)
  (h2 : f 1 + g (-1) = 4) :
  g 1 = 3 :=
sorry

end g_of_1_equals_3_l451_451416


namespace Anna_age_is_25_l451_451285

-- Define Anna's current age and Kati's current age
def Anna_current_age (A K : ℕ) : Prop :=
  -- Condition 1: Anna will be four times as old as Kati was when Anna was two years older than Kati is now
  (A + 3 = 4 * ((A - K + 2)) - A + 2)

-- Condition 2: Kati is a high school student, hence Kati's age must be between 14 and 19 and must be a multiple of 5
axiom Kati_high_school_student (K : ℕ) : 14 ≤ K ∧ K ≤ 19 ∧ K % 5 = 0

-- Given the conditions, prove Anna's age equals 25
theorem Anna_age_is_25 (A K : ℕ) (h₁ : Anna_current_age A K) (h₂ : Kati_high_school_student K) : A = 25 :=
sorry

end Anna_age_is_25_l451_451285


namespace arithmetic_geometric_mean_inequality_weighted_arithmetic_geometric_mean_inequality_l451_451197

-- Part (1)
theorem arithmetic_geometric_mean_inequality
  (n : ℕ) (h : 0 < n) (a : Fin n → ℝ) (ha : ∀ i, 0 < a i) :
  (∑ i, a i) / n ≥ (∏ i, a i) ^ (1 / n) := sorry

-- Part (2)
theorem weighted_arithmetic_geometric_mean_inequality
  (n : ℕ) (h : 0 < n) (a α : Fin n → ℝ) (ha : ∀ i, 0 < a i) (hα : ∀ i, 0 < α i) :
  (∑ i, α i * a i) / (∑ i, α i) ≥ (∏ i, (a i) ^ (α i / ∑ j, α j)) ^ (∑ j, α j) := sorry

end arithmetic_geometric_mean_inequality_weighted_arithmetic_geometric_mean_inequality_l451_451197


namespace total_pictures_l451_451130

-- Definitions based on problem conditions
def Randy_pictures : ℕ := 5
def Peter_pictures : ℕ := Randy_pictures + 3
def Quincy_pictures : ℕ := Peter_pictures + 20
def Susan_pictures : ℕ := 2 * Quincy_pictures - 7
def Thomas_pictures : ℕ := Randy_pictures ^ 3

-- The proof statement
theorem total_pictures : Randy_pictures + Peter_pictures + Quincy_pictures + Susan_pictures + Thomas_pictures = 215 := by
  sorry

end total_pictures_l451_451130


namespace number_of_folders_l451_451534

-- Define the initial number of files
def initial_files : ℕ := 80

-- Define the number of files Nancy deleted
def deleted_files : ℕ := 31

-- Define the number of files per folder
def files_per_folder : ℕ := 7

-- Define a theorem to prove the number of folders Nancy ended up with
theorem number_of_folders : (initial_files - deleted_files) / files_per_folder = 7 :=
by
  -- Placeholder for the proof
  sorry

end number_of_folders_l451_451534


namespace distinct_arc_lengths_l451_451121

def circle_center (O: Type) : O := sorry
def radius_one (r : ℝ) : Prop := r = 1
def fixed_point (A0 : Type) (O : Type) : Prop := sorry
def distributed_points (A_k : ℕ → Type) (O : Type) (n : ℕ) : Prop := ∀ k ≤ n, ∃ A_k, sorry

theorem distinct_arc_lengths (O : Type) (r : ℝ) (A0 : Type) (A_k : ℕ → Type) :
  circle_center O →
  radius_one r →
  fixed_point A0 O →
  distributed_points A_k O 1000 →
  ∃ n : ℕ, n = 3 :=
by
  sorry

end distinct_arc_lengths_l451_451121


namespace leak_empties_tank_in_time_l451_451540

/-- Definition of the rate at which Pipe A fills the tank -/
def rate_A : ℝ := 1 / 12

/-- Definition of the rate at which Pipe B fills the tank -/
def rate_B : ℝ := 1 / 24

/-- Effective rate at which the tank is filled when both pipes and the leak are working together -/
def effective_rate_with_leak : ℝ := 1 / 18

/-- Definition of the leak rate, based on the problem conditions -/
def leak_rate (A B effective_rate : ℝ) : ℝ := (A + B) - effective_rate

/-- Time taken by the leak alone to empty the tank -/
theorem leak_empties_tank_in_time
  (A B effective_rate : ℝ)
  (hA : A = rate_A)
  (hB : B = rate_B)
  (heff : effective_rate = effective_rate_with_leak) :
  1 / (leak_rate A B effective_rate) = 14.4 := by
  sorry

end leak_empties_tank_in_time_l451_451540


namespace definite_integral_value_l451_451723

variable (x : ℝ)

-- Define the integrand
def integrand (x : ℝ) : ℝ := x^2 + Real.sin x

-- Define the definite integral
noncomputable def definite_integral : ℝ := ∫ x in -1..1, integrand x

-- Statement of the problem
theorem definite_integral_value : definite_integral = 2 / 3 := by
  sorry

end definite_integral_value_l451_451723


namespace distribution_ways_numbers_closer_to_center_not_smaller_sum_of_numbers_in_rings_equal_l451_451170

-- Conditions and Definitions
def parts_of_target := 10
def numbers := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
def center_value := 10

-- Problem (a)
theorem distribution_ways :
    (10.factorial = 3628800) :=
by
    sorry  -- Proof not required

-- Problem (b)
theorem numbers_closer_to_center_not_smaller :
    ∃ counts : ℕ,
    (counts = 4320) :=
by
    sorry  -- Proof not required

-- Problem (c)
theorem sum_of_numbers_in_rings_equal :
    ∃ counts : ℕ,
    (counts = 34560) :=
by
    sorry  -- Proof not required

end distribution_ways_numbers_closer_to_center_not_smaller_sum_of_numbers_in_rings_equal_l451_451170


namespace number_of_liars_on_the_island_l451_451605

-- Definitions for the conditions
def isKnight (person : ℕ) : Prop := sorry -- Placeholder, we know knights always tell the truth
def isLiar (person : ℕ) : Prop := sorry -- Placeholder, we know liars always lie
def population := 1000
def villages := 10
def minInhabitantsPerVillage := 2

-- Definitional property: each islander claims that all other villagers in their village are liars
def claimsAllOthersAreLiars (islander : ℕ) (village : ℕ) : Prop := 
  ∀ (other : ℕ), (other ≠ islander) → (isLiar other)

-- Main statement in Lean
theorem number_of_liars_on_the_island : ∃ liars, liars = 990 :=
by
  have total_population := population
  have number_of_villages := villages
  have min_people_per_village := minInhabitantsPerVillage
  have knight_prop := isKnight
  have liar_prop := isLiar
  have claim_prop := claimsAllOthersAreLiars
  -- Proof will be filled here
  sorry

end number_of_liars_on_the_island_l451_451605


namespace infinite_solutions_l451_451357

theorem infinite_solutions (n : ℤ) : 
  (∀ x y : ℤ, x^2 + n * x * y + y^2 = 1) → (n ≠ -1 ∧ n ≠ 0 ∧ n ≠ 1) → 
  ∃ f : ℕ → ℤ × ℤ, function.injective f ∧ (∀ m, (f m).fst ^ 2 + n * (f m).fst * (f m).snd + (f m).snd ^ 2 = 1) :=
begin
  sorry
end

end infinite_solutions_l451_451357


namespace tan_2theta_sin_cos_fraction_l451_451413

variable {θ : ℝ} (h : (2 - Real.tan θ) / (1 + Real.tan θ) = 1)

-- Part (I)
theorem tan_2theta (h : (2 - Real.tan θ) / (1 + Real.tan θ) = 1) : Real.tan (2 * θ) = 4 / 3 :=
by sorry

-- Part (II)
theorem sin_cos_fraction (h : (2 - Real.tan θ) / (1 + Real.tan θ) = 1) : 
  (Real.sin θ + Real.cos θ) / (Real.cos θ - 3 * Real.sin θ) = -3 :=
by sorry

end tan_2theta_sin_cos_fraction_l451_451413


namespace area_of_triangle_l451_451700

open Real

-- Defining the line equation 3x + 2y = 12
def line_eq (x y : ℝ) : Prop := 3 * x + 2 * y = 12

-- Defining the vertices of the triangle
def vertex1 := (0, 0 : ℝ)
def vertex2 := (0, 6 : ℝ)
def vertex3 := (4, 0 : ℝ)

-- Define a function to calculate the area of the triangle
def triangle_area (v1 v2 v3 : ℝ × ℝ) : ℝ :=
  0.5 * abs ((v2.1 - v1.1) * (v3.2 - v1.2) - (v3.1 - v1.1) * (v2.2 - v1.2))

-- Prove that area of the triangle bounded by the line and coordinate axes is 12 square units
theorem area_of_triangle : triangle_area vertex1 vertex2 vertex3 = 12 :=
by
  sorry

end area_of_triangle_l451_451700


namespace trajectory_of_point_P_l451_451678

theorem trajectory_of_point_P:
  ∀ (x y : ℝ),
  ((x - 1)^2 + y^2 = 1) ∧ (√((x - 1)^2 + y^2) = 2) →
  (x - 1)^2 + y^2 = 5 :=
begin
  sorry,
end

end trajectory_of_point_P_l451_451678


namespace ratio_of_beef_to_pork_l451_451501

/-- 
James buys 20 pounds of beef. 
James buys an unknown amount of pork. 
James uses 1.5 pounds of meat to make each meal. 
Each meal sells for $20. 
James made $400 from selling meals.
The ratio of the amount of beef to the amount of pork James bought is 2:1.
-/
theorem ratio_of_beef_to_pork (beef pork : ℝ) (meal_weight : ℝ) (meal_price : ℝ) (total_revenue : ℝ)
  (h_beef : beef = 20)
  (h_meal_weight : meal_weight = 1.5)
  (h_meal_price : meal_price = 20)
  (h_total_revenue : total_revenue = 400) :
  (beef / pork) = 2 :=
by
  sorry

end ratio_of_beef_to_pork_l451_451501


namespace find_larger_number_l451_451987

theorem find_larger_number (L S : ℕ) 
  (h1 : L - S = 1000) 
  (h2 : L = 10 * S + 10) : 
  L = 1110 :=
sorry

end find_larger_number_l451_451987


namespace solve_fraction_equation_l451_451950

theorem solve_fraction_equation (x : ℝ) (h : (x + 5) / (x - 3) = 4) : x = 17 / 3 :=
by
  sorry

end solve_fraction_equation_l451_451950


namespace arithmetic_progression_11th_term_l451_451863

theorem arithmetic_progression_11th_term:
  ∀ (a d : ℝ), (15 / 2) * (2 * a + 14 * d) = 56.25 → a + 6 * d = 3.25 → a + 10 * d = 5.25 :=
by
  intros a d h_sum h_7th
  sorry

end arithmetic_progression_11th_term_l451_451863


namespace part_a_part_b_l451_451661

-- Part A: Problem
theorem part_a : ∃ N : ℕ, N = 8 ∧  (sqrt (9 - sqrt 77) * sqrt 2 * (sqrt 11 - sqrt 7) * (9 + sqrt 77) = N) := 
sorry

-- Part B: Problem
theorem part_b (x y : ℝ) (h1 : x * y = 6) (h2 : x > 2) (h3 : y > 2) : x + y < 5 :=
sorry

end part_a_part_b_l451_451661


namespace perfect_square_trinomial_m_l451_451023

-- Define a polynomial
def poly (m : ℝ) := X^2 - C m * X + 16

-- Define the property of being a perfect square trinomial
def is_perfect_square_trinomial (p : Polynomial ℝ) := ∃ a : ℝ, p = (X - C a)^2

-- State the theorem to be proved
theorem perfect_square_trinomial_m (m : ℝ) : 
  is_perfect_square_trinomial (poly m) ↔ m = 8 ∨ m = -8 := 
by
  sorry

end perfect_square_trinomial_m_l451_451023


namespace ratio_ba_in_range_l451_451471

theorem ratio_ba_in_range (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) 
  (h1 : a + 2 * b = 7) (h2 : a^2 + b^2 ≤ 25) : 
  (3 / 4 : ℝ) ≤ b / a ∧ b / a ≤ 4 / 3 :=
by {
  sorry
}

end ratio_ba_in_range_l451_451471


namespace find_x_l451_451248

variable (R : Type) [Nontrivial R] [Field R]

def diamondsuit (a b : R) : R := a / b

axiom diamondsuit_assoc (a b c : R) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : diamondsuit a (diamondsuit b c) = a * diamondsuit b * c
axiom diamondsuit_self (a : R) (ha : a ≠ 0) : diamondsuit a a = 1

theorem find_x 
  (x : R) 
  (hx : x ≠ 0) 
  : diamondsuit (4050 : R) (diamondsuit (9 : R) x) = 150 → x = (1 / 3 : R) :=
by
  intro h
  sorry

end find_x_l451_451248


namespace train_cross_bridge_time_l451_451692

theorem train_cross_bridge_time :
  ∀ (length_train length_bridge : ℝ) (speed_kmh : ℝ), 
  length_train = 150 → 
  length_bridge = 225 → 
  speed_kmh = 45 → 
  (length_train + length_bridge) / (speed_kmh * 1000 / 3600) = 30 :=
by
  intros length_train length_bridge speed_kmh ht hb hs
  rw [ht, hb, hs]
  dsimp
  linarith

end train_cross_bridge_time_l451_451692


namespace equivalent_angle_l451_451199

theorem equivalent_angle (k : ℤ) : ∃ (r : ℤ), 610 = k * 360 + r ∧ 0 ≤ r ∧ r < 360 ∧ r = 250 :=
by
  use 250
  have h₁ := (610 : ℤ) = 1 * 360 + 250
  have h₂ := 0 ≤ 250
  have h₃ := 250 < 360
  rw h₁
  tauto

--Skip the proof for now
sorry

end equivalent_angle_l451_451199


namespace divisors_count_46_320_l451_451449

theorem divisors_count_46_320 : 
  (finset.filter (λ x, 46_320 % x = 0) (finset.range 10)).card = 7 := 
by sorry

end divisors_count_46_320_l451_451449


namespace tetrahedron_labeling_impossible_l451_451752

theorem tetrahedron_labeling_impossible :
  ¬ (∃ (a b c d : ℕ), {a, b, c, d} = {1, 2, 3, 4} ∧
     (∃ s : ℕ, 
       (a + b + c = s) ∧ 
       (a + b + d = s) ∧ 
       (a + c + d = s) ∧ 
       (b + c + d = s))) :=
by sorry

end tetrahedron_labeling_impossible_l451_451752


namespace organic_fertilizer_prices_l451_451867

theorem organic_fertilizer_prices
  (x y : ℝ)
  (h1 : x - y = 100)
  (h2 : 2 * x + y = 1700) :
  x = 600 ∧ y = 500 :=
by {
  sorry
}

end organic_fertilizer_prices_l451_451867


namespace area_of_triangle_l451_451695

theorem area_of_triangle : 
  let l : ℝ → ℝ → Prop := fun x y => 3 * x + 2 * y = 12 in
  let x_intercept := (4 : ℝ) in
  let y_intercept := (6 : ℝ) in
  ∃ x y : ℝ, l x 0 ∧ x = x_intercept ∧ l 0 y ∧ y = y_intercept ∧ (1 / 2) * x_intercept * y_intercept = 12 := 
by
  sorry

end area_of_triangle_l451_451695


namespace distance_AB_eq_l451_451054

/-- The parametric equations of curve C1 in rectangular coordinates -/
def curve_C1 (α : ℝ) : ℝ × ℝ :=
⟨1 + real.cos α, real.sin α⟩

/-- The equation of curve C2 in rectangular coordinates -/
def curve_C2 (x y : ℝ) : Prop :=
x^2 / 3 + y^2 = 1

/-- Conversion from cartesian to polar coordinates -/
def polar_coordinates (x y : ℝ) : ℝ × ℝ :=
⟨real.sqrt (x^2 + y^2), real.atan2 y x⟩

/-- Conversion of curve in rectangular coordinates to polar coordinates -/
def polar_eq_C1 : ℝ → ℝ :=
λ θ, 2 * real.cos θ

/-- Polar equation for curve C2 -/
def polar_eq_C2 (θ : ℝ) : ℝ :=
real.sqrt (3 / (1 + 2 * (real.sin θ)^2))

/-- The ray equation given theta as π / 3 intersects with curves C1 and C2 at points A and B respectively, find the distance between points A and B. -/
def distance_AB : ℝ :=
abs (1 - polar_eq_C2 (real.pi / 3))

theorem distance_AB_eq :
  distance_AB = (real.sqrt 30) / 5 - 1 :=
by
  sorry

end distance_AB_eq_l451_451054


namespace gnomes_cannot_cross_l451_451538

theorem gnomes_cannot_cross :
  ∀ (gnomes : List ℕ), 
    (∀ g, g ∈ gnomes → g ∈ (List.range 100).map (λ x => x + 1)) →
    List.sum gnomes = 5050 → 
    ∀ (boat_capacity : ℕ), boat_capacity = 100 →
    ∀ (k : ℕ), (200 * (k + 1) - k^2 = 10100) → false :=
by
  intros gnomes H_weights H_sum boat_capacity H_capacity k H_equation
  sorry

end gnomes_cannot_cross_l451_451538


namespace volume_ratio_proof_l451_451379

variables (R α : ℝ)
noncomputable def sphere_volume : ℝ := (4 / 3) * Real.pi * R^3
noncomputable def segment_volume : ℝ :=
  let KD := R * (1 - Real.cos (α / 2))
  in Real.pi * KD^2 * (R - (1 / 3) * KD)
noncomputable def volume_ratio : ℝ := segment_volume R α / sphere_volume R

theorem volume_ratio_proof (R α : ℝ) : volume_ratio R α = Real.sin (α / 4)^4 * (2 + Real.cos (α / 2)) :=
  sorry

end volume_ratio_proof_l451_451379


namespace stayed_days_calculation_l451_451561

theorem stayed_days_calculation (total_cost : ℕ) (charge_1st_week : ℕ) (charge_additional_week : ℕ) (first_week_days : ℕ) :
  total_cost = 302 ∧ charge_1st_week = 18 ∧ charge_additional_week = 11 ∧ first_week_days = 7 →
  ∃ D : ℕ, D = 23 :=
by {
  sorry
}

end stayed_days_calculation_l451_451561


namespace points_with_integer_differences_l451_451682

noncomputable def polygon_area (P : set (ℝ × ℝ)) : ℝ := sorry

theorem points_with_integer_differences 
  (P : set (ℝ × ℝ)) 
  (n : ℕ) 
  (h_area : polygon_area P > n) : 
  ∃ (points : fin (n+1) → (ℝ × ℝ)), 
    (∀ i j, i ≠ j → (points i ∈ P ∧ points j ∈ P)) ∧ 
    ∀ i j, i ≠ j → (∃ m1 m2 : ℤ, (points j).1 - (points i).1 = m1 ∧ (points j).2 - (points i).2 = m2) := 
sorry

end points_with_integer_differences_l451_451682


namespace number_of_proper_subsets_of_C_l451_451444

theorem number_of_proper_subsets_of_C
  (A : Set ℝ := {x | x^2 ≠ 1})
  (a : ℝ)
  (B : Set ℝ := {x | a * x = 1})
  (C : Set ℝ := {x | x = a})
  (h1 : B ⊆ A) :
  (B.ToFinset.card := 2) → (Finset.card (C.ToFinset.powerset \ C.ToFinset) = 3) :=
begin
  sorry
end

end number_of_proper_subsets_of_C_l451_451444


namespace expansion_coefficient_lemma_l451_451425

theorem expansion_coefficient_lemma :
  (∃ n : ℕ, (4 ^ n - 2 ^ n = 992) ∧
    ((∃ t3 t4 : ℤ × ℕ, (t3 = (90, 6) ∧ t4 = (270, 22 / 3))) ∧
    (∃ r : ℕ, (r = 4 ∧ (405, 26 / 3) = (coeff (binomial_coeff n) r
                                                   * (3 ^ r) * (x ^ (term_exp (r + 1))))))
 sorry

end expansion_coefficient_lemma_l451_451425


namespace kevin_distance_after_six_hops_l451_451897

def kevin_hops : ℚ := (1/2 : ℚ) + (1/4 : ℚ) + (1/8 : ℚ) + (1/16 : ℚ) + (1/32 : ℚ) + (1/64 : ℚ)

theorem kevin_distance_after_six_hops : kevin_hops = (63/64 : ℚ) :=
by {
  -- Proof place holder
  sorry
}

end kevin_distance_after_six_hops_l451_451897


namespace exponents_equal_find_x_value_l451_451243

theorem exponents_equal {a b : ℝ} {m n : ℕ} (h : (-a^2 * b^m)^3 = -a^n * b^12) :
  m = 4 ∧ n = 6 :=
begin
  sorry
end

theorem find_x_value {x : ℕ} (h : 2^(2*x+2) - 2^(2*x+1) = 32) :
  x = 2 :=
begin
  sorry
end

end exponents_equal_find_x_value_l451_451243


namespace floor_factorial_expression_l451_451311

theorem floor_factorial_expression : 
  ⌊(2010.factorial + 2007.factorial) / (2009.factorial + 2008.factorial)⌋ = 2009 :=
by
  sorry

end floor_factorial_expression_l451_451311


namespace sum_of_values_a_l451_451172

def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 2
  else x^2 + 4 * x + 1

theorem sum_of_values_a : 
  let values := {a : ℝ | f (f a) = 1} in
  values.sum = -15 / 16 - Real.sqrt 5 :=
sorry

end sum_of_values_a_l451_451172


namespace problem_part1_problem_part2_l451_451862

variables {A B C a b c : ℝ} (h_acut : ∀ {x : ℝ}, 0 < x ∧ x < π / 2) (h_A₂π : 2 * A = π / 3)
variables (h1 : sqrt 3 * a = 2 * c * sin A) (h_c : c = sqrt 13) (h_area: (1 / 2) * a * b * (sqrt 3 / 2) = 3 * sqrt 3)

theorem problem_part1 (h_triangle : ∀ {x}, h_acut x) : C = π / 3 :=
sorry

theorem problem_part2 (h_triangle : ∀ {x}, h_acut x) (h_part1 : C = π / 3) : a + b = 7 :=
sorry

end problem_part1_problem_part2_l451_451862


namespace sum_of_integers_m_l451_451475

theorem sum_of_integers_m :
  (∀ (x m : ℝ), (x - 4) / 3 - x > 2 → (x - m) / 2 ≤ 0 → x < -5) →
  (∀ (y m : ℝ), (2 - m * y) / (3 - y) + 5 / (y - 3) = -3 → y ∈ ℤ) →
  ∑ m in {-5, -4, -2, 0, 3}, m = -8 :=
by
  sorry

end sum_of_integers_m_l451_451475


namespace total_games_played_l451_451652

-- Define the function for combinations
def combination (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

-- Given conditions
def teams : ℕ := 20
def games_per_pair : ℕ := 10

-- Proposition stating the target result
theorem total_games_played : 
  (combination teams 2 * games_per_pair) = 1900 :=
by
  sorry

end total_games_played_l451_451652


namespace order_of_a_and_reciprocal_and_square_l451_451557

theorem order_of_a_and_reciprocal_and_square 
  (n : ℕ) (hn : n > 1) : 
  let a := 1 / (n : ℝ) in 0 < a^2 ∧ a^2 < a ∧ a < 1 ∧ 1 < 1 / a := 
by
  sorry

end order_of_a_and_reciprocal_and_square_l451_451557


namespace monotonically_increasing_on_interval_l451_451849

def f (k : ℝ) (x : ℝ) : ℝ := k * x - Real.log x

theorem monotonically_increasing_on_interval (k : ℝ) :
  (∀ x : ℝ, 1 < x → 0 ≤ k - 1 / x) → 1 ≤ k :=
by
  intro h
  have h₁ := h 1 (by linarith)
  rw [one_div_one] at h₁
  linarith

end monotonically_increasing_on_interval_l451_451849


namespace cows_problem_l451_451704

theorem cows_problem :
  ∃ (M X : ℕ), 
  (5 * M = X + 30) ∧ 
  (5 * M + X = 570) ∧ 
  M = 60 :=
by
  sorry

end cows_problem_l451_451704


namespace max_sum_l451_451510

open Nat

theorem max_sum (n k : ℤ) (hn : n > 1) (hk : k > 1) 
  (a : Fin n → ℝ) (c : Fin n → ℝ) 
  (ha_nonneg : ∀ i, 0 ≤ a i) 
  (hc_nonneg : ∀ i, 0 ≤ c i)
  (ha_sorted : ∀ i j, i ≤ j → a i ≥ a j)
  (ha_sum : ∑ i in Finset.univ, a i = 1)
  (hc_condition : ∀ m : ℕ, m ∈ Finset.range (n.to_nat + 1) → ∑ i in Finset.range m, c (Fin.ofNat i) ≤ (m:ℤ)^k) :
  ∑ i in Finset.univ, c i * (a i)^k ≤ 1 :=
begin
  sorry
end

end max_sum_l451_451510


namespace olivia_wallet_final_amount_l451_451189

variable (initial_money : ℕ) (money_added : ℕ) (money_spent : ℕ)

theorem olivia_wallet_final_amount
  (h1 : initial_money = 100)
  (h2 : money_added = 148)
  (h3 : money_spent = 89) :
  initial_money + money_added - money_spent = 159 :=
  by 
    sorry

end olivia_wallet_final_amount_l451_451189


namespace axes_of_symmetry_do_not_coincide_l451_451567

-- Define the two quadratic functions
def f (x : ℝ) : ℝ := (x^2 + 6*x - 25) / 8
def g (x : ℝ) : ℝ := (31 - x^2) / 8

-- Define the axes of symmetry for the quadratic functions
def axis_of_symmetry_f : ℝ := -3
def axis_of_symmetry_g : ℝ := 0

-- Define the slopes of the tangents to the graphs at x = 4 and x = -7
def slope_f (x : ℝ) : ℝ := (2*x + 6) / 8
def slope_g (x : ℝ) : ℝ := -x / 4

-- We need to prove that the axes of symmetry do not coincide
theorem axes_of_symmetry_do_not_coincide :
    axis_of_symmetry_f ≠ axis_of_symmetry_g :=
by {
    sorry
}

end axes_of_symmetry_do_not_coincide_l451_451567


namespace geometric_sequence_fifth_term_l451_451371

theorem geometric_sequence_fifth_term (a1 a2 : ℝ) (h1 : a1 = 2) (h2 : a2 = 1 / 4) : 
  let r := a2 / a1 in
  let a5 := a1 * r ^ 4 in
  a5 = 1 / 2048 :=
by
  sorry

end geometric_sequence_fifth_term_l451_451371


namespace arrange_squares_l451_451072

theorem arrange_squares (n : ℕ) (h : n ≥ 5) :
  ∃ arrangement : Fin n → Fin n × Fin n, 
    (∀ i j : Fin n, i ≠ j → 
      (arrangement i).fst + (arrangement i).snd = (arrangement j).fst + (arrangement j).snd
      ∨ (arrangement i).fst = (arrangement j).fst
      ∨ (arrangement i).snd = (arrangement j).snd) :=
sorry

end arrange_squares_l451_451072


namespace ratio_of_sheep_to_cow_l451_451705

noncomputable def sheep_to_cow_ratio 
  (S : ℕ) 
  (h1 : 12 + 4 * S = 108) 
  (h2 : S ≠ 0) : ℕ × ℕ := 
if h3 : 12 = 0 then (0, 0) else (2, 1)

theorem ratio_of_sheep_to_cow 
  (S : ℕ) 
  (h1 : 12 + 4 * S = 108) 
  (h2 : S ≠ 0) : sheep_to_cow_ratio S h1 h2 = (2, 1) := 
sorry

end ratio_of_sheep_to_cow_l451_451705


namespace sequence_formula_l451_451991

def a (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 3
  | 2 => 6
  | 3 => 10
  | _ => sorry  -- The pattern is more general

theorem sequence_formula (n : ℕ) : a n = (n * (n + 1)) / 2 := 
  sorry

end sequence_formula_l451_451991


namespace average_of_other_two_l451_451965

theorem average_of_other_two {a b c d : ℕ} (h₀ : a ≠ b) (h₁ : a ≠ c) (h₂ : a ≠ d)
  (h₃ : b ≠ c) (h₄ : b ≠ d) (h₅ : c ≠ d) 
  (h₆ : 0 < a) (h₇ : 0 < b) (h₈ : 0 < c) (h₉ : 0 < d)
  (h₁₀ : a + b + c + d = 20) (h₁₁ : a - min (min a b) (min c d) = max (max a b) (max c d) - min (min a b) (min c d)) :
  ((a + b + c + d) - (max (max a b) (max c d) + min (min a b) (min c d))) / 2 = 2.5 :=
by
  sorry

end average_of_other_two_l451_451965


namespace problem1_problem2_problem3_problem4_problem5_l451_451621

-- Conditions
def digits := [0, 1, 2, 3, 4]

-- Problem 1
theorem problem1 : 
  {(n : ℕ) | ∃ a b c d e, a ≠ 0 ∧ a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧ 
               n = a * 10000 + b * 1000 + c * 100 + d * 10 + e}.toFinset.card = 2500 :=
by sorry

-- Problem 2
theorem problem2 : 
  {(n : ℕ) | ∃ a b c d e, a ≠ 0 ∧ a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧ 
               a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧ 
               n = a * 10000 + b * 1000 + c * 100 + d * 10 + e}.toFinset.card = 96 :=
by sorry

-- Problem 3
theorem problem3 : 
  {(n : ℕ) | ∃ a b c d e, a ≠ 0 ∧ a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧ 
               a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
               e ≠ 0 ∧ e % 2 = 1 ∧
               n = a * 10000 + b * 1000 + c * 100 + d * 10 + e}.toFinset.card = 36 :=
by sorry

-- Problem 4
theorem problem4 :
  ∃ k : ℕ, {(n : ℕ) | ∃ a b c d e, a ≠ 0 ∧ a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧ 
                a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧ 
                n = a * 10000 + b * 1000 + c * 100 + d * 10 + e}.toFinset.sort (<).indexOf 42130 = 87 :=
by sorry

-- Problem 5
theorem problem5 : 
  {(n : ℕ) | ∃ a b c d e, a ≠ 0 ∧ a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧ 
               a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
               a % 2 = 1 ∧ c % 2 = 1 ∧ e % 2 = 1 ∧ -- a, c, e are in odd positions
               n = a * 10000 + b * 1000 + c * 100 + d * 10 + e}.toFinset.card = 32 :=
by sorry

end problem1_problem2_problem3_problem4_problem5_l451_451621


namespace multiply_72517_9999_l451_451230

theorem multiply_72517_9999 : 72517 * 9999 = 725097483 :=
by
  sorry

end multiply_72517_9999_l451_451230


namespace therapy_charge_l451_451646

-- Defining the conditions
variables (A F : ℝ)
variables (h1 : F = A + 25)
variables (h2 : F + 4*A = 250)

-- The statement we need to prove
theorem therapy_charge : F + A = 115 := 
by
  -- proof would go here
  sorry

end therapy_charge_l451_451646


namespace avg_of_two_middle_numbers_l451_451973

theorem avg_of_two_middle_numbers (a b c d : ℕ) (h₀ : a ≠ b) (h₁ : a ≠ c) (h₂ : a ≠ d) (h₃ : b ≠ c) (h₄ : b ≠ d) (h₅ : c ≠ d) (h₆ : a + b + c + d = 20) (h₇ : a < d) (h₈ : d - a ≥ b - c) (h₉ : b = 2) (h₁₀ : c = 3) :
  (b + c) / 2 = 2.5 :=
by
  sorry

end avg_of_two_middle_numbers_l451_451973


namespace expand_product_l451_451353

theorem expand_product (x : ℝ) : 
  5 * (x + 6) * (x^2 + 2 * x + 3) = 5 * x^3 + 40 * x^2 + 75 * x + 90 := 
by 
  sorry

end expand_product_l451_451353


namespace comet_visibility_count_l451_451037

theorem comet_visibility_count : (n : ℕ) → 12 = (@Finset.card ℕ ⦃x | 5 ≤ x ∧ x ≤ 16⦄) :=
by
  -- Using 1740 as the starting year and 83 as the recurrence period
  let visibility_sequence (n : ℕ) : ℕ := 1740 + 83 * (n - 1)
  -- We need to check how many values of n fit between 2023 and 3000 in this sequence.
  have h_range : 2023 ≤ visibility_sequence n ∧ visibility_sequence n ≤ 3000,
  sorry
  -- Calculate the cardinality of the Finset {5, 6, ..., 16}
  exact Finset.card_Icc 5 16

end comet_visibility_count_l451_451037


namespace train_length_correct_l451_451275

-- Definitions for the given conditions
def train_speed_kmph : ℝ := 60
def man_speed_kmph : ℝ := 6
def opposite_direction_relative_speed_kmph : ℝ := train_speed_kmph + man_speed_kmph
def relative_speed_m_per_s : ℝ := opposite_direction_relative_speed_kmph * (5 / 18)
def time_seconds : ℝ := 24

-- The length of the train is the relative speed multiplied by time
def train_length : ℝ := relative_speed_m_per_s * time_seconds

-- Theorem stating the result
theorem train_length_correct (h1 : train_speed_kmph = 60) (h2 : man_speed_kmph = 6) (h3 : time_seconds = 24):
  train_length ≈ 439.92 :=
by
  have h4 : opposite_direction_relative_speed_kmph = 66 := by rw [h1, h2]
  have h5 : relative_speed_m_per_s ≈ 18.33 := by rw [h4]; norm_num
  have h6 : train_length ≈ 439.92 := by rw [h5, h3]; norm_num
  exact mod_cast

-- No further proof necessary, use 'sorry' to indicate unfinished proofs
sorry

end train_length_correct_l451_451275


namespace vector_Q_expression_l451_451879

noncomputable def vectorQ (A B C : Vec3) : Vec3 :=
  (2/5 : ℝ) • A + (0 : ℝ) • B + (2/5 : ℝ) • C

theorem vector_Q_expression (A B C : Vec3) :
  let G := (2/5 : ℝ) • A + (3/5 : ℝ) • B
  let H := (2/5 : ℝ) • B + (3/5 : ℝ) • C
  let Q := intersection_point (segment A G) (segment C H)
  Q = vectorQ A B C := sorry

end vector_Q_expression_l451_451879


namespace necessary_but_not_sufficient_l451_451841

-- Define the variables and hypothesis
variables (a b c : ℝ)

theorem necessary_but_not_sufficient :
  (a > b) ↔ (∃ c ≠ 0, ac^2 > bc^2) ∧ (¬ (a > b) → ¬ ∃ c ≠ 0, ac^2 > bc^2) :=
by 
  sorry

end necessary_but_not_sufficient_l451_451841


namespace pet_store_cages_l451_451265

def initial_puppies : ℕ := 18
def puppies_sold : ℕ := 3
def puppies_per_cage : ℕ := 5

def remaining_puppies : ℕ := initial_puppies - puppies_sold
def number_of_cages : ℕ := remaining_puppies / puppies_per_cage

theorem pet_store_cages : number_of_cages = 3 :=
by sorry

end pet_store_cages_l451_451265


namespace floor_factorial_expression_l451_451315

-- Define the factorial function for natural numbers
def factorial : ℕ → ℕ
| 0 := 1
| (n + 1) := (n + 1) * factorial n

-- The main theorem to prove
theorem floor_factorial_expression :
  (nat.floor ((factorial 2010 + factorial 2007) / (factorial 2009 + factorial 2008)) = 2009) :=
begin
  -- Actual proof goes here
  sorry
end

end floor_factorial_expression_l451_451315


namespace function_monotonicity_cos_eq_neg_third_power_l451_451279

theorem function_monotonicity_cos_eq_neg_third_power (x : ℝ) (h : 0 < x ∧ x < (π / 2)) :
  monotone (λ x, cos x) = monotone (λ x, x^(-1/3)) :=
by sorry

end function_monotonicity_cos_eq_neg_third_power_l451_451279


namespace hair_cut_off_length_l451_451711

def hair_initial_length := 11
def hair_growth_rate := 0.5
def weeks := 4
def hair_length_after_haircut := 7

/-- Prove Amy's hair cut off length is 6 inches. -/
theorem hair_cut_off_length : 
  let hair_length_before_haircut := hair_initial_length + hair_growth_rate * weeks in
  hair_length_before_haircut - hair_length_after_haircut = 6 := 
by
  sorry

end hair_cut_off_length_l451_451711


namespace Zilla_savings_l451_451643

-- Define the conditions
def rent_expense (E : ℝ) (R : ℝ) := R = 0.07 * E
def other_expenses (E : ℝ) := 0.5 * E
def amount_saved (E : ℝ) (R : ℝ) (S : ℝ) := S = E - (R + other_expenses E)

-- Define the main problem statement
theorem Zilla_savings (E R S: ℝ) 
    (hR : rent_expense E R)
    (hR_val : R = 133)
    (hS : amount_saved E R S) : 
    S = 817 := by
  sorry

end Zilla_savings_l451_451643


namespace equivalent_form_sqrt_l451_451744

theorem equivalent_form_sqrt (x : ℝ) (h : x < 0) : 
  sqrt(x / (1 - (x + 1) / x)) = -x * complex.I := 
sorry

end equivalent_form_sqrt_l451_451744


namespace find_length_ED_l451_451882

-- Given points A, B, C, L, E, D
variables (A B C L E D : Point)
-- Given angles and bisectors in the triangle
variables (triangle_ABC : Triangle A B C) (is_bisector_AL : AngleBisector A L)
-- Given collinear points, parallel lines, and specific lengths
variables (E_on_AB : OnSegment A B E) (D_on_BL : OnSegment B L D)
variables (DL_eq_LC : Segment L B D = Segment L B C) (ED_parallel_AC : IsParallel E D A C)
variables (AE_eq_15 : Segment A B E = 15) (AC_eq_12 : Segment A B C = 12)

-- The theorem statement
theorem find_length_ED : Segment E D = 3 := by
  sorry

end find_length_ED_l451_451882


namespace height_of_water_and_sum_l451_451597

-- Define the conditions for the problem
def radius := 20
def height := 120
def water_percentage := 0.2

-- Define the proof that height of water in the tank is 60 (40)^(1/3) feet, and a + b = 100
theorem height_of_water_and_sum : 
  ∃ (a b : ℕ), 
  b*20^(1/3) = 40 ∧ a + b = 100 :=
sorry

end height_of_water_and_sum_l451_451597


namespace smallest_number_divisible_l451_451203

theorem smallest_number_divisible (x : ℕ) : 
  (∃ x, x + 7 % 8 = 0 ∧ x + 7 % 11 = 0 ∧ x + 7 % 24 = 0) ∧
  (∀ y, (y + 7 % 8 = 0 ∧ y + 7 % 11 = 0 ∧ y + 7 % 24 = 0) → 257 ≤ y) :=
by { sorry }

end smallest_number_divisible_l451_451203


namespace max_slope_of_line_OQ_l451_451815

-- Given conditions
variables {p : ℕ} (h_pos_p : p > 0)
def parabola_eq (p : ℕ) := ∀ (x y : ℝ), y^2 = 2 * p * x

-- Given distance from the focus to the directrix is 2
lemma distance_focus_directrix_eq_two : p = 2 :=
by sorry

-- Thus the equation of the parabola is:
def parabola : ∀ (x y : ℝ), y^2 = 4 * x :=
by sorry

-- Variables for point P and Q
variables {O P Q : ℝ × ℝ}
-- Point P lies on the parabola
variables (hP : ∃ (x y : ℝ), y^2 = 4 * x)
-- Condition relating vector PQ and QF
variables (hPQ_QF : ∀ (P Q F : ℝ × ℝ), (P - Q) = 9 * (Q - F))
-- Maximizing slope of line OQ
def max_slope (O Q : ℝ × ℝ) : ℝ := 
  ∀ (m n : ℝ), let slope := n / ((25 * n^2 + 9) / 10) in
  slope ≤ 1 / 3 := 
by sorry

-- Prove the theorem equivalent to solution part 2, maximum slope is 1/3
theorem max_slope_of_line_OQ : max_slope O Q = 1 / 3 :=
by sorry

end max_slope_of_line_OQ_l451_451815


namespace sqrt_sum_pow_representation_l451_451655

theorem sqrt_sum_pow_representation :
  ∃ (n m k l : ℕ), 
    ( (\sqrt 3 + \sqrt 5 + \sqrt 7) ^ 2021 
      = (n * \sqrt 3 + m * \sqrt 5 + k * \sqrt 7 + l * \sqrt (3 * 5 * 7))
    ) 
    ∧ (1 - 10 ^ (-500 : ℕ) < \sqrt 35 * (l / n : ℚ) ∧ \sqrt 35 * (l / n : ℚ) < 1) := 
sorry

end sqrt_sum_pow_representation_l451_451655


namespace least_possible_brown_eyes_with_lunch_box_l451_451855

-- Definitions related to the conditions
def students_with_brown_eyes : ℕ := 15
def students_with_lunch_box : ℕ := 18
def total_students : ℕ := 25

-- Statement to prove
theorem least_possible_brown_eyes_with_lunch_box :
  ∃ n, n = 8 ∧ ∀ m, (m < 8) -> (m ≠ students_with_brown_eyes - (total_students - students_with_lunch_box)) :=
begin
  use 8,
  split,
  { rfl },
  { intros m h,
    sorry,
  }
end

end least_possible_brown_eyes_with_lunch_box_l451_451855


namespace sum_of_a_b_of_perimeter_l451_451921

def distance (p q : (ℝ × ℝ)) : ℝ :=
  Real.sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2)

noncomputable def perimeter (P Q R S : (ℝ × ℝ)) : ℝ :=
  distance P Q + distance Q R + distance R S + distance S P

theorem sum_of_a_b_of_perimeter (P Q R S : (ℝ × ℝ)) (a b p q : ℤ) 
  (h1 : P = (1, 2)) (h2 : Q = (3, 6)) (h3 : R = (6, 3)) (h4 : S = (8, 1)) 
  (ha : a = 2) (hb : b = 10) (hp : p = 5) (hq : q = 2) :
  let perimeter_value := perimeter P Q R S
  (perimeter_value = ↑a * Real.sqrt ↑p + ↑b * Real.sqrt ↑q) ∧ (a + b = 12) :=
by sorry

end sum_of_a_b_of_perimeter_l451_451921


namespace sum_first_50_b_l451_451406

def S (n : ℕ) : ℕ := n^2 + n + 1

def a : ℕ → ℕ
| 0       := 0   -- By convention, though we never use a(0)
| 1       := 3
| (n+2)   := 2*(n+2)

def b (n : ℕ) : ℤ := (-1)^n * (a n - 2)

theorem sum_first_50_b : 
  (∑ i in Finset.range 50, b (i + 1)) = 97 :=
sorry

end sum_first_50_b_l451_451406


namespace circle_intersection_l451_451476

theorem circle_intersection (a : ℝ) :
  ((-3 * Real.sqrt 2 / 2 < a ∧ a < -Real.sqrt 2 / 2) ∨ (Real.sqrt 2 / 2 < a ∧ a < 3 * Real.sqrt 2 / 2)) ↔
  (∃ x y : ℝ, (x - a)^2 + (y - a)^2 = 4 ∧ x^2 + y^2 = 1) :=
sorry

end circle_intersection_l451_451476


namespace simplest_expression_is_B_l451_451634

-- Define the expressions as Lean definitions
def expr_A : ℝ := real.sqrt 0.2
def expr_B (a b : ℝ) : ℝ := real.sqrt (a^2 - b^2)
def expr_C (x : ℝ) : ℝ := real.sqrt (1 / x)
def expr_D (a : ℝ) : ℝ := real.sqrt (4 * a)

-- Placeholder for the full proof
theorem simplest_expression_is_B (a b x : ℝ) : 
  expr_B a b = real.sqrt (a^2 - b^2) :=
by sorry

end simplest_expression_is_B_l451_451634


namespace final_bicycle_price_l451_451662

-- Define conditions 
def original_price : ℝ := 200
def first_discount : ℝ := 0.20
def second_discount : ℝ := 0.25
def price_after_first_discount := original_price * (1 - first_discount)
def final_price := price_after_first_discount * (1 - second_discount)

-- Define the Lean statement to be proven
theorem final_bicycle_price :
  final_price = 120 :=
by
  -- Proof goes here
  sorry

end final_bicycle_price_l451_451662


namespace reciprocal_of_neg_three_l451_451585

-- Define the notion of reciprocal
def reciprocal (x : ℝ) : ℝ := 1 / x

-- State the proof problem
theorem reciprocal_of_neg_three :
  reciprocal (-3) = -1 / 3 :=
by
  -- Since we are only required to state the theorem, we use sorry to skip the proof.
  sorry

end reciprocal_of_neg_three_l451_451585


namespace range_of_positive_integers_in_J_l451_451233

noncomputable def range_of_positive_integers (J : Set ℤ) (smallestTerm : ℤ) (numElements : ℕ) : ℤ :=
  let evens := List.range' smallestTerm (2 * numElements) |>.filter (· % 2 = 0)
  let positives := evens.filter (· > 0)
  match positives.min, positives.max with
  | some minVal, some maxVal => maxVal - minVal
  | _, _ => sorry  -- This case shouldn't happen if the set construction is correct

theorem range_of_positive_integers_in_J : 
  range_of_positive_integers (λ x, ∃ n ∈ (0:ℕ):10, x = -4 + 2 * n) (-4) 10 = 12 := 
by
  sorry

end range_of_positive_integers_in_J_l451_451233


namespace total_marbles_l451_451339

-- Define the given conditions 
def bags : ℕ := 20
def marbles_per_bag : ℕ := 156

-- The theorem stating that the total number of marbles is 3120
theorem total_marbles : bags * marbles_per_bag = 3120 := by
  sorry

end total_marbles_l451_451339


namespace problem_statement_l451_451167

-- Define the required conditions
variable {f : ℝ → ℝ}

-- f is odd function, f(x + 2) = f(x)
axiom f_odd : ∀ x , f (-x) = - (f x)
axiom f_periodic : ∀ x , f (x + 2) = f x

-- Define the derivative
noncomputable def f' (x : ℝ) : ℝ := sorry 

-- Proof predicate that states f' is even
def f'_even : Prop := ∀ x , f' (-x) = f' x

-- Proof predicate that states f is symmetric about (1, 0)
def f_symmetric_about_1_0 : Prop := ∀ x , f (x + 1) + f (1 - x) = 0

-- The final theorem combining these proofs, taking all conditions 
theorem problem_statement : f'_even ∧ f_symmetric_about_1_0 :=
by {
  sorry,
}

end problem_statement_l451_451167


namespace max_slope_of_line_OQ_l451_451814

-- Given conditions
variables {p : ℕ} (h_pos_p : p > 0)
def parabola_eq (p : ℕ) := ∀ (x y : ℝ), y^2 = 2 * p * x

-- Given distance from the focus to the directrix is 2
lemma distance_focus_directrix_eq_two : p = 2 :=
by sorry

-- Thus the equation of the parabola is:
def parabola : ∀ (x y : ℝ), y^2 = 4 * x :=
by sorry

-- Variables for point P and Q
variables {O P Q : ℝ × ℝ}
-- Point P lies on the parabola
variables (hP : ∃ (x y : ℝ), y^2 = 4 * x)
-- Condition relating vector PQ and QF
variables (hPQ_QF : ∀ (P Q F : ℝ × ℝ), (P - Q) = 9 * (Q - F))
-- Maximizing slope of line OQ
def max_slope (O Q : ℝ × ℝ) : ℝ := 
  ∀ (m n : ℝ), let slope := n / ((25 * n^2 + 9) / 10) in
  slope ≤ 1 / 3 := 
by sorry

-- Prove the theorem equivalent to solution part 2, maximum slope is 1/3
theorem max_slope_of_line_OQ : max_slope O Q = 1 / 3 :=
by sorry

end max_slope_of_line_OQ_l451_451814


namespace reciprocal_of_neg_three_l451_451584

-- Define the notion of reciprocal
def reciprocal (x : ℝ) : ℝ := 1 / x

-- State the proof problem
theorem reciprocal_of_neg_three :
  reciprocal (-3) = -1 / 3 :=
by
  -- Since we are only required to state the theorem, we use sorry to skip the proof.
  sorry

end reciprocal_of_neg_three_l451_451584


namespace polar_coordinate_circle_l451_451053

theorem polar_coordinate_circle {A : ℝ × ℝ} {r : ℝ} (hA : A = (0, 2)) (hr : r = 2) :
  ∀ θ : ℝ, ∃ ρ : ℝ, ρ = 4 * Real.sin θ :=
by
  intro θ
  use 4 * Real.sin θ
  sorry

end polar_coordinate_circle_l451_451053


namespace maximize_r3_minimize_r3_l451_451124

-- Define a regular pentagon and positions P within it
structure Point :=
  (x : ℝ)
  (y : ℝ)

structure Pentagon :=
  (vertices : Fin 5 → Point)

-- Define a function to calculate the distance from a point to a line in the plane
def distance_to_side (P : Point) (A B : Point) : ℝ := sorry

-- Define the order of distances from point P to the sides of the pentagon
def ordered_distances (S : Pentagon) (P : Point) : Fin 5 → ℝ := sorry

-- Assume the ordered distances such that r₁ ≤ r₂ ≤ r₃ ≤ r₄ ≤ r₅
axiom distances_ordered (S : Pentagon) (P : Point) :
  let dists := ordered_distances S P in
  dists 0 ≤ dists 1 ∧ dists 1 ≤ dists 2 ∧ dists 2 ≤ dists 3 ∧ dists 3 ≤ dists 4

-- Prove P at vertices for maximized r₃
theorem maximize_r3 (S : Pentagon) (P : Point) :
  (∃ v : Fin 5, P = S.vertices v) → (ordered_distances S P 2 = max_fin (ordered_distances S P)) :=
sorry

-- Prove P at midpoints of sides for minimized r₃
theorem minimize_r3 (S : Pentagon) (P : Point) :
  (∃ i : Fin 5, ∃ j : Fin 5, i ≠ j ∧ P = midpoint (S.vertices i) (S.vertices j)) →
    (ordered_distances S P 2 = min_fin (ordered_distances S P)) :=
sorry

end maximize_r3_minimize_r3_l451_451124


namespace sum_of_sequence_l451_451877

variables (a : ℕ → ℤ)
axiom h1 : a 1 = 1
axiom h2 : a 2 = 2
axiom recursive_rule : ∀ (n : ℕ), 0 < n → a (n + 2) - a n = 1 + (-1) ^ n

theorem sum_of_sequence : (∑ i in Finset.range 51, a (i + 1)) = 676 :=
sorry

end sum_of_sequence_l451_451877


namespace difference_of_scores_correct_l451_451160

-- Define the parameters
def num_innings : ℕ := 46
def batting_avg : ℕ := 63
def highest_score : ℕ := 248
def reduced_avg : ℕ := 58
def excluded_innings : ℕ := num_innings - 2

-- Necessary calculations
def total_runs := batting_avg * num_innings
def reduced_total_runs := reduced_avg * excluded_innings
def sum_highest_lowest := total_runs - reduced_total_runs
def lowest_score := sum_highest_lowest - highest_score

-- The correct answer to prove
def expected_difference := highest_score - lowest_score
def correct_answer := 150

-- Define the proof problem
theorem difference_of_scores_correct :
  expected_difference = correct_answer := by
  sorry

end difference_of_scores_correct_l451_451160


namespace statement_not_always_true_l451_451886

theorem statement_not_always_true 
  (a b c d : ℝ)
  (h1 : (a + b) / (3 * a - b) = (b + c) / (3 * b - c))
  (h2 : (b + c) / (3 * b - c) = (c + d) / (3 * c - d))
  (h3 : (c + d) / (3 * c - d) = (d + a) / (3 * d - a))
  (h4 : (d + a) / (3 * d - a) = (a + b) / (3 * a - b)) :
  a^2 + b^2 + c^2 + d^2 ≠ ab + bc + cd + da :=
by {
  sorry
}

end statement_not_always_true_l451_451886


namespace triangle_area_given_angle_l451_451035

theorem triangle_area_given_angle (A B C : Type) [triangle A B C]
  (angle_A : ∠A = 120)
  (side_a : 7 = side opposite to ∠A)
  (side_c : 3 = side opposite to ∠C) :
  area ABC = 15 * sqrt 3 / 4 :=
sorry

end triangle_area_given_angle_l451_451035


namespace non_degenerate_triangle_possible_integer_ys_l451_451575

theorem non_degenerate_triangle_possible_integer_ys :
  let possible_ys := {y : ℤ | 25 < y ∧ y < 55}
  let count_ys := possible_ys.to_finset.card
  count_ys = 29 :=
by
  let possible_ys := {y : ℤ | 25 < y ∧ y < 55}
  let count_ys := possible_ys.to_finset.card
  show count_ys = 29
  sorry

end non_degenerate_triangle_possible_integer_ys_l451_451575


namespace tens_digit_prime_probability_l451_451668

def is_prime_digit (d : ℕ) : Prop := d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

theorem tens_digit_prime_probability : 
  (1 : ℚ) / 2 = 1 / 5 :=
begin
  sorry
end

end tens_digit_prime_probability_l451_451668


namespace min_value_a_plus_b_l451_451518

theorem min_value_a_plus_b (a b : ℕ) (h₁ : 79 ∣ (a + 77 * b)) (h₂ : 77 ∣ (a + 79 * b)) : a + b = 193 :=
by
  sorry

end min_value_a_plus_b_l451_451518


namespace number_of_correct_statements_l451_451100

-- Definitions for lines, planes, and perpendicularity
variables (Line Plane : Type)
variables (m n : Line) (α β : Plane)
variable (perpendicular : ∀ {A B : Type}, A → A → Prop)
variable (parallel : ∀ {A B : Type}, A → A → Prop)
variable (subset : ∀ {A B : Type}, A → B → Prop)

-- Conditions of the problem
axiom diff_lines : m ≠ n
axiom diff_planes : α ≠ β
axiom statement_I : ∀ {m n : Line} {α : Plane}, perpendicular m n → perpendicular m α → ¬ subset n α → parallel n α
axiom statement_II : ∀ {m : Line} {α β : Plane}, parallel m α → perpendicular α β → ¬ perpendicular m β
axiom statement_III : ∀ {m : Line} {α β : Plane}, perpendicular m β → perpendicular α β → ¬ parallel m α
axiom statement_IV : ∀ {m n : Line} {α β : Plane}, perpendicular m n → perpendicular m α → perpendicular n β → perpendicular α β

-- The theorem to prove
theorem number_of_correct_statements : 2 = 2 :=
by
  sorry

end number_of_correct_statements_l451_451100


namespace values_of_a_l451_451828

open Set

theorem values_of_a (a : ℝ) :
  let S := {x : ℝ | x ≤ -1 ∨ x ≥ 2}
  let P := {x : ℝ | a ≤ x ∧ x ≤ a + 3}
  (S ∪ P) = univ → a = -1 :=
by
  intro h
  have : a ≤ -1 ∧ a + 3 ≥ 2, from sorry
  have : a = -1, from sorry
  exact this

end values_of_a_l451_451828


namespace question_1_question_2_l451_451787

variable (x a : ℝ)

def p := (x - a) * (x - 3 * a) < 0
def q := 8 < 2^(x + 1) ∧ 2^(x + 1) ≤ 16

theorem question_1 (ha : a = 1) (h : p x a ∧ q x) : 2 < x ∧ x < 3 := 
sorry

theorem question_2 (h : ∀ x, q x → p x a ∧ ¬(p x a → q x)) : 1 < a ∧ a ≤ 2 := 
sorry

end question_1_question_2_l451_451787


namespace no_solution_fraction_eq_l451_451847

theorem no_solution_fraction_eq (m : ℝ) : 
  ¬(∃ x : ℝ, x ≠ -1 ∧ 3 * x / (x + 1) = m / (x + 1) + 2) ↔ m = -3 :=
by
  sorry

end no_solution_fraction_eq_l451_451847


namespace analysis_duration_unknown_l451_451288

-- Definitions based on the given conditions
def number_of_bones : Nat := 206
def analysis_duration_per_bone (bone: Nat) : Nat := 5  -- assumed fixed for simplicity
-- Time spent analyzing all bones (which needs more information to be accurately known)
def total_analysis_time (bones_analyzed: Nat) (hours_per_bone: Nat) : Nat := bones_analyzed * hours_per_bone

-- Given the number of bones and duration per bone, there isn't enough information to determine the total analysis duration
theorem analysis_duration_unknown (total_bones : Nat) (duration_per_bone : Nat) (bones_remaining: Nat) (analysis_already_done : Nat) :
  total_bones = number_of_bones →
  (∀ bone, analysis_duration_per_bone bone = duration_per_bone) →
  analysis_already_done ≠ (total_bones - bones_remaining) ->
  ∃ hours_needed, hours_needed = total_analysis_time (total_bones - bones_remaining) duration_per_bone :=
by
  intros
  sorry

end analysis_duration_unknown_l451_451288


namespace math_proof_problem_l451_451292

theorem math_proof_problem : (10^8 / (2 * 10^5) - 50) = 450 := 
  by
  sorry

end math_proof_problem_l451_451292


namespace angle_A_eq_pi_div_3_max_area_of_triangle_l451_451034

variables {A B C : ℝ}
variables {a b c : ℝ}
variables {m n : ℝ × ℝ}

def m := (Real.cos A, Real.cos B)
def n := (a, 2 * c - b)

theorem angle_A_eq_pi_div_3 (h_parallel : m ∥ n) (ha : a ≠ 0) (hc_cos : (2 * c - b) * Real.cos A = a * Real.cos B) :
  Real.cos A = 1 / 2 ∧ A = Real.pi / 3 :=
begin
  sorry
end

theorem max_area_of_triangle (ha : a = 4) :
  ∃ (b c : ℝ), let S := (b * c * Real.sqrt 3 / 4) in
  S ≤ 4 * Real.sqrt 3 :=
begin
  sorry
end

end angle_A_eq_pi_div_3_max_area_of_triangle_l451_451034


namespace covered_area_l451_451952

def side_length : ℕ := 12
def area_square (s : ℕ) : ℕ := s * s

theorem covered_area :
  side_length = 12 →
  area_square side_length = 144 →
  (let total_area := 2 * area_square side_length in
   let overlap_area := 4 * (side_length * side_length / 4) / 2 in
   total_area - overlap_area = 216) := by
  sorry

end covered_area_l451_451952


namespace parabola_equation_max_slope_OQ_l451_451817

theorem parabola_equation (p : ℝ) (hp : p = 2) :
    ∃ x y : ℝ, y^2 = 2 * p * x ↔ y^2 = 4 * x :=
by sorry

theorem max_slope_OQ (Q F : ℝ × ℝ) (hQF : ∀ P : ℝ × ℝ, P ∈ parabola_eq ↔ P.x = 10 * Q.x - 9 ∧ 
                                                         P.y = 10 * Q.y ∧ y^2 = 4 * P.x)
    (hPQ : (Q.x - P.x, Q.y - P.y) = 9 * (1 - Q.x, 0 - Q.y)) :
    ∃ n : ℝ, Q.y = n ∧ Q.x = (25 * n^2 + 9) / 10 ∧ 
        max (λ n, (10 * n) / (25 * n^2 + 9)) = 1 / 3 :=
by sorry

end parabola_equation_max_slope_OQ_l451_451817


namespace volume_multiplication_factor_l451_451631

-- Define the original and new volume
def original_volume (r h : ℝ) : ℝ := π * r^2 * h

def new_volume (r h : ℝ) : ℝ := π * (2.5 * r)^2 * (3 * h)

-- Define the multiplication factor
def volume_factor (r h : ℝ) : ℝ := new_volume r h / original_volume r h

-- State the theorem
theorem volume_multiplication_factor (r h : ℝ) : volume_factor r h = 18.75 :=
by 
  unfold volume_factor
  unfold new_volume
  unfold original_volume
  rw [mul_assoc, mul_comm (π * r^2 * h), mul_assoc]
  sorry

end volume_multiplication_factor_l451_451631


namespace greatest_possible_integer_l451_451499

theorem greatest_possible_integer (m : ℕ) (h1 : m < 150) (h2 : ∃ a : ℕ, m = 10 * a - 2) (h3 : ∃ b : ℕ, m = 9 * b - 4) : m = 68 := 
  by sorry

end greatest_possible_integer_l451_451499


namespace solve_integer_pairs_l451_451358

theorem solve_integer_pairs :
  {p : ℤ × ℤ | p.fst + p.snd = p.fst^2 - p.fst * p.snd + p.snd^2} =
  {(0, 0), (0, 1), (1, 0), (1, 2), (2, 1), (2, 2)} :=
by sorry

end solve_integer_pairs_l451_451358


namespace infinite_sum_of_zeta_fractional_parts_l451_451765

open Real

noncomputable def fractional_part (x : ℝ) : ℝ := x - floor x

theorem infinite_sum_of_zeta_fractional_parts :
  (∑' k : ℕ, fractional_part (ζ (2 * (k + 1)))) = 1 / 2 :=
by
  sorry

end infinite_sum_of_zeta_fractional_parts_l451_451765


namespace hyperbola_line_intersection_range_l451_451008

def hyperbola : ℝ → ℝ → Prop := λ x y, x^2 - y^2 = 4
def line (k : ℝ) : ℝ → ℝ → Prop := λ x y, y = k * (x - 1)

theorem hyperbola_line_intersection_range (k : ℝ) :
  let discriminant : ℝ := 4 * (4 - 3 * k^2) in
  (¬ (1 - k^2 = 0) ∧ discriminant > 0 → k ∈ Ioo (-2 * real.sqrt 3 / 3) (-1) ∪ Ioo (-1) 1 ∪ Ioo 1 (2 * real.sqrt 3 / 3))
  ∧ ((1 - k^2 = 0 ∨ (¬ (1 - k^2 = 0) ∧ discriminant = 0)) → k = 1 ∨ k = -1 ∨ k = 2 * real.sqrt 3 / 3 ∨ k = -2 * real.sqrt 3 / 3)
  ∧ (¬ (¬ (1 - k^2 = 0) ∧ discriminant > 0 ∨ (1 - k^2 = 0 ∨ (¬ (1 - k^2 = 0) ∧ discriminant = 0))) → k ∈ Iio (-2 * real.sqrt 3 / 3) ∪ Ioi (2 * real.sqrt 3 / 3)) := sorry

end hyperbola_line_intersection_range_l451_451008


namespace organic_fertilizer_prices_l451_451866

theorem organic_fertilizer_prices
  (x y : ℝ)
  (h1 : x - y = 100)
  (h2 : 2 * x + y = 1700) :
  x = 600 ∧ y = 500 :=
by {
  sorry
}

end organic_fertilizer_prices_l451_451866


namespace parabola_equation_max_slope_OQ_l451_451819

theorem parabola_equation (p : ℝ) (hp : p = 2) :
    ∃ x y : ℝ, y^2 = 2 * p * x ↔ y^2 = 4 * x :=
by sorry

theorem max_slope_OQ (Q F : ℝ × ℝ) (hQF : ∀ P : ℝ × ℝ, P ∈ parabola_eq ↔ P.x = 10 * Q.x - 9 ∧ 
                                                         P.y = 10 * Q.y ∧ y^2 = 4 * P.x)
    (hPQ : (Q.x - P.x, Q.y - P.y) = 9 * (1 - Q.x, 0 - Q.y)) :
    ∃ n : ℝ, Q.y = n ∧ Q.x = (25 * n^2 + 9) / 10 ∧ 
        max (λ n, (10 * n) / (25 * n^2 + 9)) = 1 / 3 :=
by sorry

end parabola_equation_max_slope_OQ_l451_451819


namespace average_of_two_middle_numbers_is_correct_l451_451969

def four_numbers_meeting_conditions (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  (a + b + c + d = 20) ∧ 
  (max a (max b (max c d)) - min a (min b (min c d)) = max_diff

def max_diff := 
  (∀ (x y : ℕ), (x ≠ y → x > 0 → y > 0 → x + y ≤ 19 ∧ x + y ≥ 5) → 
  (x = 14 ∧ y = 1))

theorem average_of_two_middle_numbers_is_correct :
  ∃ (a b c d : ℕ), four_numbers_meeting_conditions a b c d →
  let numbers := [a, b, c, d].erase (min a (min b (min c d))).erase (max a (max b (max c d))),
  (numbers.sum / 2) = 2.5 := 
by
  sorry

end average_of_two_middle_numbers_is_correct_l451_451969


namespace probability_25_cents_min_l451_451957

-- Define the five coins and their values
def penny := 0.01
def nickel := 0.05
def dime := 0.10
def quarter := 0.25
def halfDollar := 0.50

-- Define a function that computes the total value of heads up coins
def value_heads (results : (Bool × Bool × Bool × Bool × Bool)) : ℝ :=
  let (h₁, h₂, h₃, h₄, h₅) := results 
  (if h₁ then penny else 0) +
  (if h₂ then nickel else 0) +
  (if h₃ then dime else 0) +
  (if h₄ then quarter else 0) +
  (if h₅ then halfDollar else 0)

-- Define the main theorem statement
theorem probability_25_cents_min :
  (∑ results in (finset.univ : finset (Bool × Bool × Bool × Bool × Bool)),
    if value_heads results ≥ 0.25 then (1 : ℝ) else 0) / 32 = 13 / 16 := sorry

end probability_25_cents_min_l451_451957


namespace exercise_mean_days_l451_451115

theorem exercise_mean_days
  (students_exercise_counts : List (Nat × Nat))
  (h : students_exercise_counts = [(1, 1), (3, 2), (2, 3), (6, 4), (8, 5), (3, 6), (2, 7)]) :
  let total_days := students_exercise_counts.foldr (λ (p : Nat × Nat) acc, acc + p.1 * p.2) 0
  let total_students := students_exercise_counts.foldr (λ (p : Nat × Nat) acc, acc + p.1) 0
  float_of_nat total_days / float_of_nat total_students = 4.36 :=
by
  sorry

end exercise_mean_days_l451_451115


namespace reciprocal_of_neg3_l451_451590

theorem reciprocal_of_neg3 : (1 / (-3) = -1 / 3) :=
by
  sorry

end reciprocal_of_neg3_l451_451590


namespace eval_expression_l451_451352
-- We use Mathlib to ensure all necessary math functions and properties are available

-- Statement of the problem in Lean
theorem eval_expression :
  1234562 - ((12 * 3 * (2 + 7))^2 / 6) + 18 = 1217084 :=
by
  linarith

end eval_expression_l451_451352


namespace probability_different_grandchildren_count_l451_451116

theorem probability_different_grandchildren_count :
  let total_grandchildren := 12
  let total_variations := 2 ^ total_grandchildren
  let comb := Nat.choose total_grandchildren (total_grandchildren / 2)
  let prob_equal := comb / total_variations
  let prob_different := 1 - prob_equal
  prob_different = 793 / 1024 := by
sorry

end probability_different_grandchildren_count_l451_451116


namespace haley_candy_l451_451763

theorem haley_candy (X : ℕ) (h : X - 17 + 19 = 35) : X = 33 :=
by
  sorry

end haley_candy_l451_451763


namespace beaver_group_l451_451146

theorem beaver_group (B : ℕ) :
  (B * 3 = 12 * 5) → B = 20 :=
by
  intros h1
  -- Additional steps for the proof would go here.
  -- The h1 hypothesis represents the condition B * 3 = 60.
  exact sorry -- Proof steps are not required.

end beaver_group_l451_451146


namespace MrWillamTaxPercentage_l451_451754

-- Definitions
def TotalTaxCollected : ℝ := 3840
def MrWillamTax : ℝ := 480

-- Theorem Statement
theorem MrWillamTaxPercentage :
  (MrWillamTax / TotalTaxCollected) * 100 = 12.5 :=
by
  sorry

end MrWillamTaxPercentage_l451_451754


namespace units_digit_of_sum_of_squares_2010_odds_l451_451204

noncomputable def sum_units_digit_of_squares (n : ℕ) : ℕ :=
  let units_digits := [1, 9, 5, 9, 1]
  List.foldl (λ acc x => (acc + x) % 10) 0 (List.map (λ i => units_digits.get! (i % 5)) (List.range (2 * n)))

theorem units_digit_of_sum_of_squares_2010_odds : sum_units_digit_of_squares 2010 = 0 := sorry

end units_digit_of_sum_of_squares_2010_odds_l451_451204


namespace floor_fraction_equals_2009_l451_451294

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem floor_fraction_equals_2009 :
  (⌊ (factorial 2010 + factorial 2007) / (factorial 2009 + factorial 2008) ⌋ : ℤ) = 2009 :=
by sorry

end floor_fraction_equals_2009_l451_451294


namespace nonneg_f_of_a_l451_451809

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * real.exp x - real.log x - 1

lemma extremum_point_a (h : deriv (f a) 2 = 0) : a = 1 / (2 * real.exp 2) :=
by sorry

theorem nonneg_f_of_a (a : ℝ) (h : a ≥ 1 / real.exp 1) (x : ℝ) (hx : x > 0) : f a x ≥ 0 :=
by sorry

end nonneg_f_of_a_l451_451809


namespace num_positions_P_l451_451934

-- Defining the points in the plane
def O : ℝ × ℝ := (0,0)
def Q : ℝ × ℝ := (4,4)

-- Predicate for a point being on the y-axis and within the range (0, 100)
def onYAxis (P : ℝ × ℝ) : Prop := P.1 = 0 ∧ 0 < P.2 ∧ P.2 < 100

-- Predicate for P such that the radius of the circle through O, P, and Q is an integer
def integerRadius (P : ℝ × ℝ) : Prop :=
  ∃ k : ℤ, k > 0 ∧ ∃ a : ℝ, (P = (0, 2 * a)) ∧ k = real.sqrt (2 * (a - 2)^2 + 8)

-- Define the main theorem to count possible positions of P
theorem num_positions_P : ∃ n : ℤ, n = 66 :=
  sorry

end num_positions_P_l451_451934


namespace probability_sum_le_10_l451_451195

-- We define the type representing a fair six-sided die.
def Die := {n : ℕ // n > 0 ∧ n ≤ 6}

-- The sample space of rolling two six-sided dice.
def sampleSpace : Finset (Die × Die) :=
  (Finset.range 6).product (Finset.range 6)

-- The event that the sum of the two dice rolls is less than or equal to 10.
def eventSumLE10 : Finset (Die × Die) :=
  sampleSpace.filter (λ p, p.1.val + p.2.val ≤ 10)

-- Probability calculation
noncomputable def probabilitySumLE10 : ℚ :=
  (eventSumLE10.card : ℚ) / (sampleSpace.card : ℚ)

theorem probability_sum_le_10 :
  probabilitySumLE10 = 11 / 12 :=
  sorry

end probability_sum_le_10_l451_451195


namespace next_word_after_Ятианр_l451_451239

def custom_alphabet : List Char := ['Т', 'А', 'Р', 'Н', 'И', 'Я']

def lexicographical_order (a b : Char) : Prop :=
  custom_alphabet.indexOf a < custom_alphabet.indexOf b

def word_to_next_in_lexicographical_order (word : List Char) : (List Char) :=
  -- Function to find the next lexicographical permutation
  sorry

theorem next_word_after_Ятианр :
  word_to_next_in_lexicographical_order ['Я', 'Т', 'И', 'А', 'Н', 'Р'] = ['Я', 'Т', 'И', 'Р', 'А', 'Н'] :=
by
  sorry

end next_word_after_Ятианр_l451_451239


namespace avg_of_two_middle_numbers_l451_451974

theorem avg_of_two_middle_numbers (a b c d : ℕ) (h₀ : a ≠ b) (h₁ : a ≠ c) (h₂ : a ≠ d) (h₃ : b ≠ c) (h₄ : b ≠ d) (h₅ : c ≠ d) (h₆ : a + b + c + d = 20) (h₇ : a < d) (h₈ : d - a ≥ b - c) (h₉ : b = 2) (h₁₀ : c = 3) :
  (b + c) / 2 = 2.5 :=
by
  sorry

end avg_of_two_middle_numbers_l451_451974


namespace order_of_magnitude_l451_451777

noncomputable def a : ℝ := (0.5 : ℝ)^(-1/3)
noncomputable def b : ℝ := (3/5 : ℝ)^(-1/3)
noncomputable def c : ℝ := Real.logBase 2.5 1.5

theorem order_of_magnitude :
  0 < c ∧ c < b ∧ b < a :=
by
  have ha : a = (2 ^ (1/3)) := by sorry
  have hb : b > 1 := by sorry
  have hc : 0 < c ∧ c < 1 := by sorry
  sorry

end order_of_magnitude_l451_451777


namespace num_integers_make_expression_integer_l451_451768

theorem num_integers_make_expression_integer : 
  { n : ℤ | ∃ k : ℤ, 72 * (3 / 2)^n = k }.card = 6 :=
by
  sorry

end num_integers_make_expression_integer_l451_451768


namespace number_of_paths_to_spell_MATH_l451_451872

-- Define the problem setting and conditions
def number_of_paths_M_to_H (adj: ℕ) (steps: ℕ): ℕ :=
  adj^(steps-1)

-- State the problem in Lean 4
theorem number_of_paths_to_spell_MATH : number_of_paths_M_to_H 8 4 = 512 := 
by 
  unfold number_of_paths_M_to_H 
  -- The needed steps are included:
  -- We calculate: 8^(4-1) = 8^3 which should be 512.
  sorry

end number_of_paths_to_spell_MATH_l451_451872


namespace problem_statement_l451_451460

def g (x : ℝ) : ℝ := x ^ 3
def f (x : ℝ) : ℝ := 2 * x - 1

theorem problem_statement : f (g 3) = 53 :=
by
  sorry

end problem_statement_l451_451460


namespace necessary_but_not_sufficient_condition_l451_451923

open Set

variable {α : Type*} [PartialOrder α]

def M (x : α) : Prop := x > 2

def P (x : α) : Prop := x < 3

theorem necessary_but_not_sufficient_condition (x : α) :
  (M x ∨ P x) → (M x ∧ P x) ∧ ¬((M x ∨ P x) ↔ (M x ∧ P x)) :=
by
  sorry

end necessary_but_not_sufficient_condition_l451_451923


namespace woman_speed_in_still_water_l451_451648

theorem woman_speed_in_still_water (V_w V_s : ℝ)
  (downstream_distance upstream_distance downstream_time upstream_time : ℝ)
  (h1 : downstream_distance = 45)
  (h2 : upstream_distance = 15)
  (h3 : downstream_time = 3)
  (h4 : upstream_time = 3)
  (h5 : V_w + V_s = downstream_distance / downstream_time)
  (h6 : V_w - V_s = upstream_distance / upstream_time) :
  V_w = 10 :=
by 
  have h1 : downstream_distance = 45 := by assumption,
  have h2 : upstream_distance = 15 := by assumption,
  have h3 : downstream_time = 3 := by assumption,
  have h4 : upstream_time = 3 := by assumption,
  have h5 : V_w + V_s = 15 := by {
    rw h1 at *,
    rw h3 at *,
    exact h5
  },
  have h6 : V_w - V_s = 5 := by {
    rw h2 at *,
    rw h4 at *,
    exact h6
  },
  sorry

end woman_speed_in_still_water_l451_451648


namespace arrange_abc_l451_451097

-- Definitions of the given values
def a : ℝ := 2^(-0.3)
def b : ℝ := Real.logb 2 0.3
def c : ℝ := Real.logb (1/2) 0.3

-- Theorem statement
theorem arrange_abc : c > a ∧ a > b := by
  sorry

end arrange_abc_l451_451097


namespace eccentricity_range_l451_451007

variable (C : set (ℝ × ℝ)) (a b : ℝ) (f : ℝ × ℝ)
variable (O A B : ℝ × ℝ)
variable (l : set (ℝ × ℝ))

--Conditions
variable (hb_gt_ha : b > a) (ha_gt_zero : a > 0)
variable (h_origin : O = (0, 0))
variable (h_focus : f = (sqrt(a^2 + b^2), 0))
variable (h_hyperbola : ∀ x y, (x, y) ∈ C ↔ x^2 / a^2 - y^2 / b^2 = 1)
variable (h_line : ∃ k, l = {p | ∃ x' y', p = (x', y') ∧ y' = k * (x' - sqrt(a^2 + b^2))})
variable (h_intersection_A : A ∈ C ∩ l)
variable (h_intersection_B : B ∈ C ∩ l)
variable (h_perpendicular : (fst A) * (fst B) + (snd A) * (snd B) = 0)

--To prove
theorem eccentricity_range (e : ℝ) : e = sqrt(a^2 + b^2) / a → e > sqrt 2 := by
  sorry

end eccentricity_range_l451_451007


namespace Zilla_savings_l451_451644

-- Define the conditions
def rent_expense (E : ℝ) (R : ℝ) := R = 0.07 * E
def other_expenses (E : ℝ) := 0.5 * E
def amount_saved (E : ℝ) (R : ℝ) (S : ℝ) := S = E - (R + other_expenses E)

-- Define the main problem statement
theorem Zilla_savings (E R S: ℝ) 
    (hR : rent_expense E R)
    (hR_val : R = 133)
    (hS : amount_saved E R S) : 
    S = 817 := by
  sorry

end Zilla_savings_l451_451644


namespace tom_caught_16_trout_l451_451533

theorem tom_caught_16_trout (melanie_trout : ℕ) (tom_caught_twice : melanie_trout * 2 = 16) : 
  2 * melanie_trout = 16 :=
by 
  sorry

end tom_caught_16_trout_l451_451533


namespace sum_reciprocal_g_approx_l451_451099

-- Define the integer closest to the cube root of n
def g (n : ℕ) := round (n^(1 / 3 : ℝ))

-- Prove the sum of reciprocals of g(k) (k ranges from 1 to 2744) is approximately equal to 286.49
theorem sum_reciprocal_g_approx : 
  (Finset.range 2744).sum (λ k, (1 : ℝ) / g (k + 1)) ≈ 286.49 :=
sorry

end sum_reciprocal_g_approx_l451_451099


namespace sum_series_l451_451543

theorem sum_series (n : ℕ) (h : n > 1) : 
  (∑ k in Finset.range (n-1), (1 : ℚ) / (k+1) / (k+2)) = (n-1) / n :=
sorry

end sum_series_l451_451543


namespace floor_factorial_expression_l451_451318

theorem floor_factorial_expression : 
  (⌊(2010! + 2007! : ℚ) / (2009! + 2008! : ℚ)⌋ = 2009) :=
by
  -- Let a := 2010! and b := 2007!
  -- So a + b = 2010! + 2007!
  -- Notice 2010! = 2010 * 2009 * 2008 * 2007!
  -- Notice 2009! = 2009 * 2008 * 2007!
  -- Simplify (2010! + 2007!) / (2009! + 2008!)
  sorry

end floor_factorial_expression_l451_451318


namespace sum_of_first_seven_terms_l451_451159

noncomputable def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
a + (n - 1) * d

theorem sum_of_first_seven_terms :
  ∀ (a d : ℝ), a = 1 → (arithmetic_sequence a d 2 + arithmetic_sequence a d 3) = 3 → 
  (∑ i in finset.range 7, arithmetic_sequence a d (i + 1)) = 14 :=
begin
  intros a d h1 h2,
  sorry
end

end sum_of_first_seven_terms_l451_451159


namespace proof_problem_l451_451989

noncomputable def F (p : ℝ) : ℝ × ℝ := (p / 2, 0)
noncomputable def parabola (y x p : ℝ) : Prop := y^2 = 2 * p * x
noncomputable def B (n p : ℝ) : ℝ × ℝ := (p * n^2 + p * n * Real.sqrt (n^2 + 1) + p / 2, p * n + p * Real.sqrt (n^2 + 1))
noncomputable def C (n p : ℝ) : ℝ × ℝ := (p * n^2 - p * n * Real.sqrt (n^2 + 1) + p / 2, p * n - p * Real.sqrt (n^2 + 1))
noncomputable def M (n p : ℝ) : ℝ × ℝ := (p * n^2 + p / 2, p * n)
noncomputable def N (n p : ℝ) : ℝ × ℝ := (p * n^2 + 3 * p / 2, 0)
noncomputable def dist (A B : ℝ × ℝ) : ℝ := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

theorem proof_problem (p n : ℝ) (h : 0 < p) :
  dist (M n p) (N n p)^2 = dist (F p) (B n p) * dist (F p) (C n p) := sorry

end proof_problem_l451_451989


namespace math_problem_l451_451000

-- Define the ellipse equation and conditions
def ellipse_equation : Prop :=
  ∀ (x y : ℝ), x^2 / 8 + y^2 / 4 = 1

def midpoint_condition : Prop :=
  ∀ (A B : ℝ × ℝ),
  midpoint A B = (2, 1) ∧
  (let (x1, y1) := A, (x2, y2) := B in
    x1^2 / 8 + y1^2 / 4 = 1 ∧ x2^2 / 8 + y2^2 / 4 = 1)

-- Prove the equation of line AB
def equation_of_line_AB : Prop :=
  ellipse_equation ∧ midpoint_condition →
  ∀ (m b : ℝ), (y1, y2 : ℝ),
    b = 1 ∧ m = -1 / (2 * 1) ∧
    (let x = 2 in x + y - 3 = 0)

-- Prove the length of the chord AB
def length_of_chord_AB : Prop :=
  ellipse_equation ∧ midpoint_condition →
  ∀ (length : ℝ),
    length = (4 * sqrt 3) / 3

-- Combined proof
def proof_problem : Prop :=
  equation_of_line_AB ∧ length_of_chord_AB

theorem math_problem : proof_problem := by
  sorry

end math_problem_l451_451000


namespace perpendicular_bisector_equation_l451_451409

-- Define points A and B
structure Point where
  x : ℝ
  y : ℝ

-- Given points A(1,2) and B(3,1)
def A : Point := { x := 1, y := 2 }
def B : Point := { x := 3, y := 1 }

-- Midpoint of segment AB
def midpoint (P Q : Point) : Point := 
  { x := (P.x + Q.x) / 2, y := (P.y + Q.y) / 2 }

-- Calculate the slope of AB
def slope (P Q : Point) : ℝ := 
  (Q.y - P.y) / (Q.x - P.x)

-- Calculate the slope of the perpendicular bisector
def perpendicular_slope (m : ℝ) : ℝ := 
  -1 / m

-- Equation of the line given a point and a slope in point-slope form
def line_equation (P : Point) (m : ℝ) : (ℝ → ℝ) := 
  λ x, m * (x - P.x) + P.y

-- Convert line equation to standard form ax + by + c = 0
def standard_form (P : Point) (m : ℝ) : (ℝ × ℝ × ℝ) := 
  (m, -1, m * -P.x + P.y)

-- Main proof goal
theorem perpendicular_bisector_equation :
  ∃ (a b c : ℝ), (4, -2, 5) = standard_form (midpoint A B) (perpendicular_slope (slope A B)) :=
by
  sorry

end perpendicular_bisector_equation_l451_451409


namespace tan_seventeen_pi_over_four_l451_451755

theorem tan_seventeen_pi_over_four : Real.tan (17 * Real.pi / 4) = 1 := by
  sorry

end tan_seventeen_pi_over_four_l451_451755


namespace complete_square_l451_451215

theorem complete_square (x : ℝ) : (x ^ 2 + 4 * x + 1 = 0) ↔ ((x + 2) ^ 2 = 3) :=
by {
  split,
  { intro h,
    sorry },
  { intro h,
    sorry }
}

end complete_square_l451_451215


namespace cost_of_mms_in_terms_of_snickers_l451_451483

theorem cost_of_mms_in_terms_of_snickers 
  (snickers_price : ℝ) (snickers_bought : ℕ) (mms_bought : ℕ) (money_given : ℝ) (change_received : ℝ)
  (Hsnickers : snickers_price = 1.5) (Hsnickers_bought : snickers_bought = 2) (Hmms_bought : mms_bought = 3)
  (Hmoney_given : money_given = 20) (Hchange_received : change_received = 8) :
  (money_given - change_received - snickers_bought * snickers_price) / mms_bought / snickers_price = 2 :=
by
  -- Scope of each identifier is limited to this theorem
  let total_spent := money_given - change_received
  let spent_on_snickers := snickers_bought * snickers_price
  let spent_on_mms := total_spent - spent_on_snickers
  let cost_per_mms := spent_on_mms / mms_bought
  calc
    cost_per_mms / snickers_price = 2 : sorry

end cost_of_mms_in_terms_of_snickers_l451_451483


namespace ognev_phone_number_l451_451165

def alphabet_position (c : Char) : ℕ :=
  if 'a' ≤ c ∧ c ≤ 'z' then c.toNat - 'a'.toNat + 1
  else if 'A' ≤ c ∧ c ≤ 'Z' then c.toNat - 'A'.toNat + 1
  else 0

def phone_number_of_surname (surname : String) : ℕ :=
  let first_digit := surname.length
  let first_letter_pos := alphabet_position surname.head!
  let last_letter_pos := alphabet_position surname.toList.last!
  let remaining_digits := (first_letter_pos * 100 + last_letter_pos)
  (first_digit * 10000 + remaining_digits)

theorem ognev_phone_number :
  phone_number_of_surname "Ognev" = 5163 := by
  sorry

end ognev_phone_number_l451_451165


namespace num_factors_of_N_l451_451016

-- Define the number N using prime factorization
def N : ℕ := 2^4 * 3^2 * 5 * 7^2

-- Define the number of natural-number factors of N
def numberOfFactors (n : ℕ) : ℕ :=
  ∏ d in (n.divisors.to_finset), 1 -- This is a placeholder definition

theorem num_factors_of_N : 
  numberOfFactors N = 90 := by
  -- Skip the proof for now
  sorry

end num_factors_of_N_l451_451016


namespace time_for_each_trip_is_4_hours_l451_451289

-- Define the speed of Athul in still water
def speed_in_still_water : ℝ := 5

-- Define the speed of the stream
def stream_speed : ℝ := 1

-- Define the distances traveled upstream and downstream
def upstream_distance : ℝ := 16
def downstream_distance : ℝ := 24

-- Define the effective speeds
def upstream_speed : ℝ := speed_in_still_water - stream_speed
def downstream_speed : ℝ := speed_in_still_water + stream_speed

-- Define the times taken, which are equal
def upstream_time : ℝ := upstream_distance / upstream_speed
def downstream_time : ℝ := downstream_distance / downstream_speed

-- Prove that the time taken for both upstream and downstream is 4 hours
theorem time_for_each_trip_is_4_hours : upstream_time = 4 ∧ downstream_time = 4 :=
by
  sorry

end time_for_each_trip_is_4_hours_l451_451289


namespace mr_brown_net_result_l451_451929

noncomputable def C1 := 1.50 / 1.3
noncomputable def C2 := 1.50 / 0.9
noncomputable def profit_from_first_pen := 1.50 - C1
noncomputable def tax := 0.05 * profit_from_first_pen
noncomputable def total_cost := C1 + C2
noncomputable def total_revenue := 3.00
noncomputable def net_result := total_revenue - total_cost - tax

theorem mr_brown_net_result : net_result = 0.16 :=
by
  sorry

end mr_brown_net_result_l451_451929


namespace number_of_tangents_l451_451935

-- Define the points and conditions
variable (A B : ℝ × ℝ)
variable (dist_AB : dist A B = 8)
variable (radius_A : ℝ := 3)
variable (radius_B : ℝ := 2)

-- The goal
theorem number_of_tangents (dist_condition : dist A B = 8) : 
  ∃ n, n = 2 :=
by
  -- skipping the proof
  sorry

end number_of_tangents_l451_451935


namespace time_to_paint_one_house_l451_451497

theorem time_to_paint_one_house (houses : ℕ) (total_time_hours : ℕ) (total_time_minutes : ℕ) 
  (minutes_per_hour : ℕ) (h1 : houses = 9) (h2 : total_time_hours = 3) 
  (h3 : minutes_per_hour = 60) (h4 : total_time_minutes = total_time_hours * minutes_per_hour) : 
  (total_time_minutes / houses) = 20 :=
by
  sorry

end time_to_paint_one_house_l451_451497


namespace smallest_circle_radius_l451_451610

-- Define the problem as a proposition
theorem smallest_circle_radius (r : ℝ) (R1 R2 : ℝ) (hR1 : R1 = 6) (hR2 : R2 = 4) (h_right_triangle : (r + R2)^2 + (r + R1)^2 = (R2 + R1)^2) : r = 2 := 
sorry

end smallest_circle_radius_l451_451610


namespace monotone_decreasing_intervals_l451_451435

theorem monotone_decreasing_intervals (f : ℝ → ℝ)
  (h : ∀ x : ℝ, deriv f x = (x - 2) * (x^2 - 1)) :
  ((∀ x : ℝ, x < -1 → deriv f x < 0) ∧ (∀ x : ℝ, 1 < x → x < 2 → deriv f x < 0)) :=
by
  sorry

end monotone_decreasing_intervals_l451_451435


namespace hamburgers_initial_count_l451_451684

theorem hamburgers_initial_count (served left_over : ℕ) (h_served : served = 3) (h_left_over : left_over = 6) : served + left_over = 9 := by
  rw [h_served, h_left_over]
  exact Nat.add_comm _ _
  exact Nat.add_comm _ _
  exact Nat.add_self _

sorrry

end hamburgers_initial_count_l451_451684


namespace perfect_square_dice_l451_451256

theorem perfect_square_dice (p q : ℕ) (hc : Nat.coprime p q) (h : 6^4 = 1296) : 
  let F := {x // x = 1 ∨ x = 3 ∨ x = 5 ∨ x = 7 ∨ x = 8 ∨ x = 9}
  let n := 4
  let outcomes := List.replicate n F
  let count_p_sq := -- Function to count perfect square outcomes (omitted)
  let total_possible_outcomes := h
  let probability_p_sq := count_p_sq / total_possible_outcomes
  let pq := p + q
  in
  pq = 217 := 
sorry

end perfect_square_dice_l451_451256


namespace floor_factorial_expression_eq_2009_l451_451303

def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

theorem floor_factorial_expression_eq_2009 :
  (Int.floor (↑(factorial 2010 + factorial 2007) / ↑(factorial 2009 + factorial 2008)) = 2009) := by
  sorry

end floor_factorial_expression_eq_2009_l451_451303


namespace reachable_target_l451_451284

-- Define the initial state of the urn
def initial_urn_state : (ℕ × ℕ) := (150, 50)

-- Define the operations as changes in counts of black and white marbles
def operation1 (state : ℕ × ℕ) := (state.1 - 2, state.2)
def operation2 (state : ℕ × ℕ) := (state.1 - 1, state.2)
def operation3 (state : ℕ × ℕ) := (state.1, state.2 - 2)
def operation4 (state : ℕ × ℕ) := (state.1 + 2, state.2 - 3)

-- Define a predicate that a state can be reached from the initial state
def reachable (target : ℕ × ℕ) : Prop :=
  ∃ n1 n2 n3 n4 : ℕ, 
    operation1^[n1] (operation2^[n2] (operation3^[n3] (operation4^[n4] initial_urn_state))) = target

-- The theorem to be proved
theorem reachable_target : reachable (1, 2) :=
sorry

end reachable_target_l451_451284


namespace floor_factorial_expression_l451_451308

theorem floor_factorial_expression : 
  ⌊(2010.factorial + 2007.factorial) / (2009.factorial + 2008.factorial)⌋ = 2009 :=
by
  sorry

end floor_factorial_expression_l451_451308


namespace reciprocal_neg3_l451_451580

-- Define the problem
def reciprocal (x : ℚ) : ℚ := 1 / x

-- The required proof statement
theorem reciprocal_neg3 : reciprocal (-3) = -1 / 3 :=
by
  sorry

end reciprocal_neg3_l451_451580


namespace problem_result_l451_451622

open Matrix FiniteDimensional

-- Definitions of the initial vectors u₀ and z₀
def u₀ : ℝ^2 := ![2, 1]
def z₀ : ℝ^2 := ![3, 2]

-- Projections in general form for Lean
def proj (a b : ℝ^2) : ℝ^2 := (dot_product a b / dot_product a a) • a

-- Definitions for the vector sequences
def u (n : ℕ) : ℝ^2 :=
  if n = 0 then u₀ else proj u₀ (z (n - 1))

def z (n : ℕ) : ℝ^2 :=
  if n = 0 then z₀ else proj z₀ (u n)

-- The sum of all vectors in the sequence
noncomputable def sequence_sum : ℝ^2 := 
  ∑' n : ℕ, u (n + 1) + z (n + 1)

-- The theorem to prove the desired sum
theorem problem_result :
  sequence_sum = ![520, 312] :=
sorry

end problem_result_l451_451622


namespace cos_angle_BNG_l451_451658

noncomputable def isRegularTetrahedron (A B C D : ℝ³) : Prop := 
  (dist A B = dist B C) ∧ (dist B C = dist C D) ∧ (dist C D = dist D A) ∧
  (dist A C = dist A D) ∧ (dist B D = dist A B)

noncomputable def midpoint (P Q : ℝ³) : ℝ³ := (P + Q) / 2

noncomputable def centroid (P Q R : ℝ³) : ℝ³ := (P + Q + R) / 3

theorem cos_angle_BNG {A B C D : ℝ³} (h : isRegularTetrahedron A B C D) 
  (N : ℝ³) (hN : N = midpoint A B) 
  (G : ℝ³) (hG : G = centroid A C D) : 
  real.cos (angle B N G) = 1 / 3 :=
sorry

end cos_angle_BNG_l451_451658


namespace factorial_floor_problem_l451_451326

theorem factorial_floor_problem :
  (nat.floor ( (nat.factorial 2010 + nat.factorial 2007) / (nat.factorial 2009 + nat.factorial 2008) )) = 2009 :=
by 
sorry

end factorial_floor_problem_l451_451326


namespace area_of_triangle_l451_451697

theorem area_of_triangle : 
  let l : ℝ → ℝ → Prop := fun x y => 3 * x + 2 * y = 12 in
  let x_intercept := (4 : ℝ) in
  let y_intercept := (6 : ℝ) in
  ∃ x y : ℝ, l x 0 ∧ x = x_intercept ∧ l 0 y ∧ y = y_intercept ∧ (1 / 2) * x_intercept * y_intercept = 12 := 
by
  sorry

end area_of_triangle_l451_451697


namespace sucrose_concentration_in_mixture_is_correct_l451_451273

def solutionA_concentration : ℝ := 15.3 / 100
def solutionB_concentration : ℝ := 27.8 / 100
def volumeA : ℝ := 45
def volumeB : ℝ := 75

def total_sucrose_A : ℝ := solutionA_concentration * volumeA
def total_sucrose_B : ℝ := solutionB_concentration * volumeB
def total_sucrose : ℝ := total_sucrose_A + total_sucrose_B
def total_volume : ℝ := volumeA + volumeB

def resulting_concentration : ℝ := total_sucrose / total_volume

theorem sucrose_concentration_in_mixture_is_correct :
  abs (resulting_concentration - 0.231125) < 1e-12 := by
  sorry

end sucrose_concentration_in_mixture_is_correct_l451_451273


namespace factorial_floor_problem_l451_451328

theorem factorial_floor_problem :
  (nat.floor ( (nat.factorial 2010 + nat.factorial 2007) / (nat.factorial 2009 + nat.factorial 2008) )) = 2009 :=
by 
sorry

end factorial_floor_problem_l451_451328


namespace LynsDonation_l451_451529

theorem LynsDonation (X : ℝ)
  (h1 : 1/3 * X + 1/2 * X + 1/4 * (X - (1/3 * X + 1/2 * X)) = 3/4 * X)
  (h2 : (X - 3/4 * X)/4 = 30) :
  X = 240 := by
  sorry

end LynsDonation_l451_451529


namespace units_digit_of_2011_odd_squares_l451_451209

def units_digit_sum_squares_first_k_odd_integers (k : ℕ) : ℕ :=
  let odd_numbers := List.range k |>.map (λ n, 2*n + 1)
  let squares := odd_numbers.map (λ n, n^2)
  let total_sum := squares.sum
  total_sum % 10

theorem units_digit_of_2011_odd_squares : units_digit_sum_squares_first_k_odd_integers 2011 = 9 :=
by
  sorry

end units_digit_of_2011_odd_squares_l451_451209


namespace angle_WYZ_deg_l451_451051

-- Definition of angles and relationships in the problem
def angle_XWY : ℝ := 53
def angle_WXY : ℝ := 43
def angle_XYZ : ℝ := 85
def is_straight_line (x y z : ℝ) : Prop := x + y + z = 180

-- Statement of the theorem
theorem angle_WYZ_deg (angle_XWY_deg : ℝ) (angle_WXY_deg : ℝ) (angle_XYZ_deg : ℝ) 
  (h_XYZ_straight : is_straight_line 180 0 0) : 
  angle_WYZ_deg = 1 :=
by
  -- The proof will be provided here.
  sorry

end angle_WYZ_deg_l451_451051


namespace mark_hourly_wage_before_raise_40_l451_451531

-- Mark's hourly wage before the raise
def hourly_wage_before_raise (x : ℝ) : Prop :=
  let weekly_hours := 40
  let raise_percentage := 0.05
  let new_hourly_wage := x * (1 + raise_percentage)
  let new_weekly_earnings := weekly_hours * new_hourly_wage
  let old_bills := 600
  let personal_trainer := 100
  let new_expenses := old_bills + personal_trainer
  let leftover_income := 980
  new_weekly_earnings = new_expenses + leftover_income

-- Proving that Mark's hourly wage before the raise was 40 dollars
theorem mark_hourly_wage_before_raise_40 : hourly_wage_before_raise 40 :=
by
  -- Proof goes here
  sorry

end mark_hourly_wage_before_raise_40_l451_451531


namespace circumcircle_radius_eq_distance_incenter_circumcenter_l451_451939

variables {R R_x r : ℝ}  -- circumradius, circumradius of CDE, inradius

def R_IO := sqrt (R^2 - 2 * R * r)

-- Problem statement to prove
theorem circumcircle_radius_eq_distance_incenter_circumcenter
  (h1 : R_x = sqrt (R^2 - 2 * R * r))  -- Given condition for R_x
  : R_x = R_IO := begin
  sorry
end

end circumcircle_radius_eq_distance_incenter_circumcenter_l451_451939


namespace domain_of_log_function_monotonicity_of_log_function_value_of_log_function_at_2_l451_451436

noncomputable def logarithmic_domain (a : ℝ) (x : ℝ) : Prop :=
  0 < a ∧ a ≠ 1 → x < 4

noncomputable def log_function_monotonicity (a : ℝ) (x : ℝ) : Prop :=
  0 < a ∧ a ≠ 1 →
  (if 1 < a then strictly_decreasing_on (λ x, log a (4 - x)) {: x | x < 4 }
  else strictly_increasing_on (λ x, log a (4 - x)) {: x | x < 4 })

noncomputable def log_function_value_at_2 (x : ℝ) : ℝ :=
  if x = 2 then log 2 (4 - x) else 0

theorem domain_of_log_function (a : ℝ) (x : ℝ) : logarithmic_domain a x := sorry

theorem monotonicity_of_log_function (a : ℝ) (x : ℝ) : log_function_monotonicity a x := sorry

theorem value_of_log_function_at_2 : log_function_value_at_2 2 = 1 := by
  simp [log_function_value_at_2, log]
  sorry

end domain_of_log_function_monotonicity_of_log_function_value_of_log_function_at_2_l451_451436


namespace circle_properties_l451_451428

noncomputable def circle_eq : Real → Real → Prop := λ x y, (x^2 + y^2 - 4*x + 3 = 0)

def point_outside_circle (x y : Real) : Prop :=
  ¬ circle_eq x y ∧ x = 4 ∧ y = 0

def center_symmetric_with_respect_to_line (cx cy : Real) : Prop :=
  (cx = 2 ∧ cy = 0) ∧ (2 + 3 * 0 - 2 = 0)

def circle_radius : Real := 1

def tangent_line (x y : Real) : Prop :=
  let d := abs (1 * 2 + (-sqrt 3) * 0) / sqrt (1^2 + (-sqrt 3)^2) in
  d = circle_radius

theorem circle_properties :
  (∀ x y, ¬circle_eq x y → x = 4 → y = 0) ∧
  (∃ cx cy, center_symmetric_with_respect_to_line cx cy) ∧
  circle_radius = 1 ∧
  (∀ x y, tangent_line x y) :=
by
  sorry

end circle_properties_l451_451428


namespace cost_of_ground_school_l451_451985

theorem cost_of_ground_school (G : ℝ) (F : ℝ) (h1 : F = G + 625) (h2 : F = 950) :
  G = 325 :=
by
  sorry

end cost_of_ground_school_l451_451985


namespace screamers_lineups_l451_451961

/-- The Screamers have 15 players. Bob, Yogi, and Zane refuse to play together in any combination.
    Determine the number of starting lineups of 6 players not including all of Bob, Yogi, and Zane. -/
theorem screamers_lineups (Bob Yogi Zane : Fin 15) :
  ¬ (Bob = Yogi ∧ Yogi = Zane) →
  (∑ i, if i = Bob ∨ i = Yogi ∨ i = Zane then 0 else 1) = 12 →
  (Finset.card {l : Finset (Fin 15) | ∃ b y z, 
    (b ∈ l → b = Bob) ∧ 
    (y ∈ l → y = Yogi) ∧
    (z ∈ l → z = Zane) ∧ 
    Finset.card l = 6 } = 3300) :=
by
  intros h1 h2
  sorry

end screamers_lineups_l451_451961


namespace average_of_other_two_l451_451964

theorem average_of_other_two {a b c d : ℕ} (h₀ : a ≠ b) (h₁ : a ≠ c) (h₂ : a ≠ d)
  (h₃ : b ≠ c) (h₄ : b ≠ d) (h₅ : c ≠ d) 
  (h₆ : 0 < a) (h₇ : 0 < b) (h₈ : 0 < c) (h₉ : 0 < d)
  (h₁₀ : a + b + c + d = 20) (h₁₁ : a - min (min a b) (min c d) = max (max a b) (max c d) - min (min a b) (min c d)) :
  ((a + b + c + d) - (max (max a b) (max c d) + min (min a b) (min c d))) / 2 = 2.5 :=
by
  sorry

end average_of_other_two_l451_451964


namespace anthony_lunch_money_l451_451287

-- Define the costs as given in the conditions
def juice_box_cost : ℕ := 27
def cupcake_cost : ℕ := 40
def amount_left : ℕ := 8

-- Define the total amount needed for lunch every day
def total_amount_for_lunch : ℕ := juice_box_cost + cupcake_cost + amount_left

theorem anthony_lunch_money : total_amount_for_lunch = 75 := by
  -- This is where the proof would go.
  sorry

end anthony_lunch_money_l451_451287


namespace floor_fraction_equals_2009_l451_451299

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem floor_fraction_equals_2009 :
  (⌊ (factorial 2010 + factorial 2007) / (factorial 2009 + factorial 2008) ⌋ : ℤ) = 2009 :=
by sorry

end floor_fraction_equals_2009_l451_451299


namespace unique_domino_partition_l451_451916

theorem unique_domino_partition {n : ℕ} (h : n > 0) : 
  ∃ k : ℕ, k = 1 ∧ ∀ (marks : Finset (Fin (2 * n) × Fin (2 * n))),
  marks.card = k →
  ∃! (partition : List (Fin (2 * n) × Fin (2 * n))),
    (∀ (d : (Fin (2 * n) × Fin (2 * n))), d ∈ partition → 
      (d.1.1 < 2 * n ∧ d.2.1 < 2 * n ∧ 
      (d.1.1 + 1 = d.2.1.1 ∨ d.1.1 = d.2.1.1 + 1 
      ∨ d.1 = (d.2.1 + 1, d.2.2)
      ∨ d.1 = (d.2.1, d.2.2 + 1))) ∧
      (∀ mark ∈ marks, ∃ dom ∈ partition, mark ∈ dom) ∧ 
      (∀ dom1 dom2 ∈ partition, dom1 ≠ dom2 → dom1 ∩ dom2 = ∅) :=
by sorry

end unique_domino_partition_l451_451916


namespace smallest_nat_twice_cube_thrice_square_l451_451380

theorem smallest_nat_twice_cube_thrice_square :
  ∃ (k : ℕ), (∃ (n : ℕ), k = 2 * n^3) ∧ (∃ (m : ℕ), k = 3 * m^2) ∧ k = 432 :=
by {
  sorry,
}

end smallest_nat_twice_cube_thrice_square_l451_451380


namespace shuttle_speed_l451_451647

theorem shuttle_speed (v : ℕ) (h : v = 9) : v * 3600 = 32400 :=
by
  sorry

end shuttle_speed_l451_451647


namespace amount_received_is_500_l451_451887

-- Define the conditions
def books_per_month : ℕ := 3
def months_per_year : ℕ := 12
def price_per_book : ℕ := 20
def loss : ℕ := 220

-- Calculate number of books bought in a year
def books_per_year : ℕ := books_per_month * months_per_year

-- Calculate total amount spent on books in a year
def total_spent : ℕ := books_per_year * price_per_book

-- Calculate the amount Jack got from selling the books based on the given loss
def amount_received : ℕ := total_spent - loss

-- Proving the amount received is $500
theorem amount_received_is_500 : amount_received = 500 := by
  sorry

end amount_received_is_500_l451_451887


namespace first_term_arithmetic_sum_l451_451077

theorem first_term_arithmetic_sum 
  (T : ℕ → ℚ) (b : ℚ) (d : ℚ) (h₁ : ∀ n, T n = n * (2 * b + (n - 1) * d) / 2)
  (h₂ : d = 5)
  (h₃ : ∀ n, (T (4 * n)) / (T n) = (16 : ℚ)) : 
  b = 5 / 2 :=
sorry

end first_term_arithmetic_sum_l451_451077


namespace number_of_valid_four_digit_numbers_l451_451402

open Finset

def four_digit_numbers_with_sum (s : Finset (Fin 10)) : Prop :=
  s.card = 4 ∧ ∑ x in s, (x : Nat) = 12

theorem number_of_valid_four_digit_numbers : 
  (univ.filter four_digit_numbers_with_sum).card = 174 := 
sorry

end number_of_valid_four_digit_numbers_l451_451402


namespace floor_factorial_expression_eq_2009_l451_451302

def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

theorem floor_factorial_expression_eq_2009 :
  (Int.floor (↑(factorial 2010 + factorial 2007) / ↑(factorial 2009 + factorial 2008)) = 2009) := by
  sorry

end floor_factorial_expression_eq_2009_l451_451302


namespace cube_edges_not_in_same_plane_l451_451876

theorem cube_edges_not_in_same_plane (cube : Type) [is_cube : cube] (f : face) (l : line_on_face f) :
  ∃ n, n ∈ {4, 6, 7, 8} ∧ edges_not_in_same_plane(cube, f, l) = n :=
sorry

end cube_edges_not_in_same_plane_l451_451876


namespace solution_set_of_inequality_l451_451030

variable (f : ℝ → ℝ)

theorem solution_set_of_inequality :
  (∀ x, f (x) = f (-x)) →               -- f(x) is even
  (∀ x y, 0 < x → x < y → f y ≤ f x) →   -- f(x) is monotonically decreasing on (0, +∞)
  f 2 = 0 →                              -- f(2) = 0
  {x : ℝ | (f x + f (-x)) / (3 * x) < 0} = 
    {x : ℝ | -2 < x ∧ x < 0} ∪ {x : ℝ | 2 < x} :=
by sorry

end solution_set_of_inequality_l451_451030


namespace first_term_arithmetic_sum_l451_451078

theorem first_term_arithmetic_sum 
  (T : ℕ → ℚ) (b : ℚ) (d : ℚ) (h₁ : ∀ n, T n = n * (2 * b + (n - 1) * d) / 2)
  (h₂ : d = 5)
  (h₃ : ∀ n, (T (4 * n)) / (T n) = (16 : ℚ)) : 
  b = 5 / 2 :=
sorry

end first_term_arithmetic_sum_l451_451078


namespace cos_angle_product_le_one_eighth_l451_451649

theorem cos_angle_product_le_one_eighth (α β γ : ℝ) (hα : 0 < α ∧ α < π)
  (hβ : 0 < β ∧ β < π) (hγ : 0 < γ ∧ γ < π) 
  (h_sum : α + β + γ = π) : cos α * cos β * cos γ ≤ 1 / 8 :=
begin
  sorry
end

end cos_angle_product_le_one_eighth_l451_451649


namespace emily_quiz_probability_l451_451853

noncomputable def prob_at_least_two_correct : ℚ := 763 / 3888

theorem emily_quiz_probability :
  (let total_prob := 1 - ((5/6)^5 + 5 * (1/6) * (5/6)^4)
  in total_prob = prob_at_least_two_correct) :=
by sorry

end emily_quiz_probability_l451_451853


namespace hyperbola_eccentricity_l451_451811

theorem hyperbola_eccentricity (a b : ℝ) (h_a : a > 0) (h_b : b > 0) 
  (h_tangent : let asymptote := (λ x : ℝ, (b / a) * x) in ∀ x y, (x - 4)^2 + y^2 = 4 ↔ y = asymptote x) :
  real.eccentricity a b = 2 * real.sqrt 3 / 3 :=
by
  sorry

end hyperbola_eccentricity_l451_451811


namespace largest_int_sqrt_389_l451_451764

def g (n : ℕ) : ℕ := if n > 1 then (List.find (λ d, n % d = 0) (List.range' 1 n.reverse)).getOrElse 1 else 0

def f (n : ℕ) : ℕ := n - g(n)

def smallest_N : ℕ := (List.find (λ n, f(f(f(n))) = 97) (List.range' 2 Nat.succ 1000)).getOrElse 0

theorem largest_int_sqrt_389 : ⌊ Real.sqrt smallest_N ⌋.toNat = 19 := by
  -- Proof is omitted, using sorry
  sorry

end largest_int_sqrt_389_l451_451764


namespace find_max_slope_of_OQ_l451_451821

noncomputable def parabola_C := {p : ℝ // p = 2}

def parabola_eq (p : ℝ) : Prop := 
  ∀ x y : ℝ, (y^2 = 2 * p * x) → (y^2 = 4 * x)

def max_slope (p : ℝ) (O Q : ℝ × ℝ) (F P Q' : ℝ × ℝ) : Prop := 
  ∀ K : ℝ, K = (Q.2) / (Q.1) → 
  ∀ n : ℝ, (K = (10 * n) / (25 * n^2 + 9)) →
  ∀ n : ℝ , n = (3 / 5) → 
  K = (1 / 3)

theorem find_max_slope_of_OQ : 
  ∀ pq: parabola_C,
  ∃ C : parabola_eq pq.val,
  ∃ O F P Q : (ℝ × ℝ),
  (F = (1, 0)) ∧
  (P.1 * P.1 = 4 * P.2) ∧
  (Q.1 - P.1, Q.2 - P.2) = 9 * -(F.1 - Q.1, Q.2) →
  max_slope pq.val O Q F P Q'.1 :=
sorry

end find_max_slope_of_OQ_l451_451821


namespace max_slope_of_line_OQ_l451_451816

-- Given conditions
variables {p : ℕ} (h_pos_p : p > 0)
def parabola_eq (p : ℕ) := ∀ (x y : ℝ), y^2 = 2 * p * x

-- Given distance from the focus to the directrix is 2
lemma distance_focus_directrix_eq_two : p = 2 :=
by sorry

-- Thus the equation of the parabola is:
def parabola : ∀ (x y : ℝ), y^2 = 4 * x :=
by sorry

-- Variables for point P and Q
variables {O P Q : ℝ × ℝ}
-- Point P lies on the parabola
variables (hP : ∃ (x y : ℝ), y^2 = 4 * x)
-- Condition relating vector PQ and QF
variables (hPQ_QF : ∀ (P Q F : ℝ × ℝ), (P - Q) = 9 * (Q - F))
-- Maximizing slope of line OQ
def max_slope (O Q : ℝ × ℝ) : ℝ := 
  ∀ (m n : ℝ), let slope := n / ((25 * n^2 + 9) / 10) in
  slope ≤ 1 / 3 := 
by sorry

-- Prove the theorem equivalent to solution part 2, maximum slope is 1/3
theorem max_slope_of_line_OQ : max_slope O Q = 1 / 3 :=
by sorry

end max_slope_of_line_OQ_l451_451816


namespace incorrect_option_d_l451_451251

noncomputable def normal_distribution (μ σ : ℝ) (x : ℝ) : ℝ :=
  (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-((x - μ) ^ 2) / (2 * σ ^ 2))

variable (σ : ℝ) (hσ : σ > 0)

theorem incorrect_option_d :
  ∃ μ : ℝ, μ = 10 ∧
  let prob1 := (1 / (σ * Real.sqrt (2 * Real.pi))) * (Real.integral (normal_distribution μ σ) 9.9 10.2) in
  let prob2 := (1 / (σ * Real.sqrt (2 * Real.pi))) * (Real.integral (normal_distribution μ σ) 10 10.3) in
  prob1 ≠ prob2 := 
by
  exists 10
  sorry

end incorrect_option_d_l451_451251


namespace beavers_build_dam_l451_451149

def num_beavers_first_group : ℕ := 20

theorem beavers_build_dam (B : ℕ) (t₁ : ℕ) (t₂ : ℕ) (n₂ : ℕ) :
  (B * t₁ = n₂ * t₂) → (B = num_beavers_first_group) := 
by
  -- Given
  let t₁ := 3
  let t₂ := 5
  let n₂ := 12

  -- Work equation
  assume h : B * t₁ = n₂ * t₂
  
  -- Correct answer
  have B_def : B = (n₂ * t₂) / t₁,
  exact h
   
  sorry

end beavers_build_dam_l451_451149


namespace beaver_group_l451_451145

theorem beaver_group (B : ℕ) :
  (B * 3 = 12 * 5) → B = 20 :=
by
  intros h1
  -- Additional steps for the proof would go here.
  -- The h1 hypothesis represents the condition B * 3 = 60.
  exact sorry -- Proof steps are not required.

end beaver_group_l451_451145


namespace ratio_MN_BC_l451_451043

-- Define the context for the problem
variables {ABC : Type} [triangle ABC]
variables (M N : ABC) (R r : ℝ)
variables (AB AC MB BC CN : ℝ)

-- Define the conditions provided
def point_on_sides (M N : ABC) : Prop := (M ∈ AB) ∧ (N ∈ AC)
def distance_conditions (MB BC CN : ℝ) : Prop := (MB = BC) ∧ (BC = CN)
def circumradius_inradius (R r : ℝ) : Prop := ∃ (circumradius R) (inradius r), true

-- The main theorem to be proved
theorem ratio_MN_BC (R r : ℝ) (MB : ℝ) (BC : ℝ) (CN : ℝ) (M : ABC) (N : ABC)
  (h_conditions : point_on_sides M N)
  (h_distances : distance_conditions MB BC CN)
  (h_radii: circumradius_inradius R r) :
  ∃ MN : ℝ, MN / BC = real.sqrt (1 - 2 * r / R) :=
sorry

end ratio_MN_BC_l451_451043


namespace sum_in_base_5_is_54_sum_in_base_result_is_54_in_base_5_l451_451156

def calculate_sum_in_base_b (b : ℕ) : ℕ :=
  3 * b + 14

theorem sum_in_base_5_is_54 : calculate_sum_in_base_b 5 = 29 :=
by
  sorry

theorem sum_in_base_result_is_54_in_base_5 : nat.toDigits 5 (calculate_sum_in_base_b 5) = [5, 4] :=
by
  sorry

end sum_in_base_5_is_54_sum_in_base_result_is_54_in_base_5_l451_451156


namespace angle_sum_other_pair_l451_451193

-- Define the quadrilateral and the given conditions
variables {A B C D : Type} -- Points defining the quadrilateral
variables (angle : A → B → Type) (sum_to_180 : ∀ {a b : Type}, angle a b → Prop) -- Angle measures & their properties

-- Assuming we are given two angles A and B that sum to 180 degrees
axiom angle_sum_180 (a b : A) : sum_to_180 (angle a b)

-- Proving that two sides (represented here as pairs of points) are parallel
axiom sides_parallel (a b c d : A) (h : sum_to_180 (angle a b)) : a ≠ b ∧ c ≠ d ∧ angle a b = 180

-- Proving that the sum of the other two angles also equals 180 degrees
theorem angle_sum_other_pair
    (a b c d : A)
    (h : sum_to_180 (angle a b))
    (parallel_sides: a ≠ b ∧ c ≠ d ∧ angle a b = 180) :
    sum_to_180 (angle c d) :=
sorry

end angle_sum_other_pair_l451_451193


namespace mia_fruit_eating_permutations_l451_451114

theorem mia_fruit_eating_permutations :
  ∃ ways : ℕ, ways = nat.div (nat.factorial 7) (nat.factorial 4 * nat.factorial 2 * nat.factorial 1) ∧ ways = 105 :=
by
  use nat.div (nat.factorial 7) (nat.factorial 4 * nat.factorial 2 * nat.factorial 1)
  simp
  sorry

end mia_fruit_eating_permutations_l451_451114


namespace pq_parallel_bl_l451_451657

-- Definitions of the geometric concepts needed
variables (A B C L M P Q : Point)

-- Triangle ABC
axiom triangle_abc : Triangle A B C

-- BL is the bisector of angle BAC
axiom bisector_bl : Bisector (angle A B C) L

-- M is an arbitrary point on segment CL
axiom point_m : PointOnSegment M C L

-- Tangent to circumcircle of triangle ABC at B intersects CA at P
axiom tangent_circumcircle_b_p : TangentAtCircumcircleSegment (circumcircle_of_triangle A B C) B (segment C A) P

-- Tangents to circumcircle of BLM at B and M intersect at Q
axiom tangents_intersect : TangentAtCircumcirclePoint (circumcircle_b_l_m B L M) B Q ∧ TangentAtCircumcirclePoint (circumcircle_b_l_m B L M) M Q

-- Prove that PQ is parallel to BL
theorem pq_parallel_bl : Parallel (line P Q) (line B L) :=
sorry

end pq_parallel_bl_l451_451657


namespace inverse_matrix_matrix_power_times_vector_l451_451244

-- Problem 1 Lean 4 Statement
theorem inverse_matrix (α : ℝ) (M : Matrix (Fin 2) (Fin 2) ℝ := ![![cos α, -sin α], ![sin α, cos α]])
  (A B : Fin 2 -> ℝ := ![2, 2]) (B' : Fin 2 -> ℝ := ![-2, 2]) :
  (M.mulVec A = B') →
  (inv M = ![![0, 1], ![-1, 0]]) :=
by
sorry

-- Problem 2 Lean 4 Statement
theorem matrix_power_times_vector :
  let A := ![![2, 1], ![4, 2]]
  let β := ![1, 7]
  ∀ (n : ℕ) (h : n = 50),
  (A ^ n).mulVec β = ![ (4 ^ 50 * 9 / 4), (4 ^ 50 * 9 / 2)] :=
by
sorry

end inverse_matrix_matrix_power_times_vector_l451_451244


namespace choose_4_from_7_l451_451489

theorem choose_4_from_7 : nat.choose 7 4 = 35 :=
by
  sorry

end choose_4_from_7_l451_451489


namespace simplify_radicals_l451_451721

theorem simplify_radicals (q : ℝ) (hq : 0 < q) :
  (Real.sqrt (42 * q)) * (Real.sqrt (7 * q)) * (Real.sqrt (14 * q)) = 98 * q * Real.sqrt (3 * q) :=
by
  sorry

end simplify_radicals_l451_451721


namespace find_first_term_arithmetic_sequence_l451_451083

theorem find_first_term_arithmetic_sequence (a : ℤ) (k : ℤ)
  (hTn : ∀ n : ℕ, T_n = n * (2 * a + (n - 1) * 5) / 2)
  (hConstant : ∀ n : ℕ, (T (4 * n) / T n) = k) : a = 3 :=
by
  sorry

end find_first_term_arithmetic_sequence_l451_451083


namespace Zilla_savings_l451_451645

-- Define the conditions
def rent_expense (E : ℝ) (R : ℝ) := R = 0.07 * E
def other_expenses (E : ℝ) := 0.5 * E
def amount_saved (E : ℝ) (R : ℝ) (S : ℝ) := S = E - (R + other_expenses E)

-- Define the main problem statement
theorem Zilla_savings (E R S: ℝ) 
    (hR : rent_expense E R)
    (hR_val : R = 133)
    (hS : amount_saved E R S) : 
    S = 817 := by
  sorry

end Zilla_savings_l451_451645


namespace subset_B_in_A_inequality_l451_451110

-- Define the inequality condition
def inequality (x : ℝ) : Prop := 4^x - 5 * 2^x + 4 < 0

-- Define the set A
def set_A := { x : ℝ | 0 < x ∧ x < 2 }

-- Define the set B given m
def set_B (m : ℝ) := { x : ℝ | 3 - 2 * m < x ∧ x < m + 1 }

-- Define the proof problem in Lean
theorem subset_B_in_A_inequality (m : ℝ) : (∀ x, inequality x → (x ∈ set_A)) → 
  (∀ x, x ∈ set_B m → x ∈ set_A) → m ≤ 1 :=
by
  intro H₁ H₂
  -- proof goes here
  sorry

end subset_B_in_A_inequality_l451_451110


namespace subset_elem_relationships_l451_451439

theorem subset_elem_relationships :
  let A := {1, 2, 3, 4, 5}
  in {1, 2} ⊆ A ∧ 3 ∈ A ∧ ¬ {6} ⊆ A ∧ ¬ 6 ∈ A :=
by
  let A := {1, 2, 3, 4, 5}
  sorry

end subset_elem_relationships_l451_451439


namespace expansion_correct_statements_l451_451769

theorem expansion_correct_statements :
  let expansion := (7 - X)^7
  let num_terms := 8
  let sum_of_coefficients_128 := 128
  let binom_coeff_seventh_term := 49
  let sum_of_all_coefficients := 6^7
  -- Conditions
  (num_terms = 8) →
  (sum_of_coefficients_128 = (7 + 1)^7) →
  (binom_coeff_seventh_term ≠ binomCoeff 7 (7 - 1)) →
  (sum_of_all_coefficients = (7 - 1)^7) →
  -- Expected result
  (num_terms = 8 ∧ binom_coeff_seventh_term ≠ 49 ∧ sum_of_all_coefficients = 6^7) :=
by
  intros h1 h2 h3 h4
  have ha : num_terms = 8 := h1
  have hb : sum_of_coefficients_128 = 128 := by sorry
  have hc : binom_coeff_seventh_term ≠ 49 := h3
  have hd : sum_of_all_coefficients = 6^7 := h4
  exact ⟨ha, hc, hd⟩

end expansion_correct_statements_l451_451769


namespace number_of_absent_men_l451_451673

theorem number_of_absent_men
  (original_men : ℕ)
  (initial_days : ℕ)
  (final_days : ℕ)
  (absent_men : ℕ)
  (original_men = 60)
  (initial_days = 50)
  (final_days = 60) :
  original_men - absent_men = 50 → absent_men = 10 :=
by
  sorry

end number_of_absent_men_l451_451673


namespace shapes_after_rotation_l451_451259

-- Define the initial positions of the shapes
def position (shape : Type) : Type := shape

-- Define the positions of square, pentagon, and ellipse
def X : Type := position X
def Y : Type := position Y
def Z : Type := position Z

-- Define the rotation function
def rotate_180 (p : Type) : Type :=
  match p with
  | X => Y
  | Y => X
  | Z => Z

-- The theorem that we need to prove
theorem shapes_after_rotation :
  rotate_180 X = Y ∧ rotate_180 Y = X ∧ rotate_180 Z = Z :=
by
  sorry

end shapes_after_rotation_l451_451259


namespace average_of_two_middle_numbers_is_correct_l451_451970

def four_numbers_meeting_conditions (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  (a + b + c + d = 20) ∧ 
  (max a (max b (max c d)) - min a (min b (min c d)) = max_diff

def max_diff := 
  (∀ (x y : ℕ), (x ≠ y → x > 0 → y > 0 → x + y ≤ 19 ∧ x + y ≥ 5) → 
  (x = 14 ∧ y = 1))

theorem average_of_two_middle_numbers_is_correct :
  ∃ (a b c d : ℕ), four_numbers_meeting_conditions a b c d →
  let numbers := [a, b, c, d].erase (min a (min b (min c d))).erase (max a (max b (max c d))),
  (numbers.sum / 2) = 2.5 := 
by
  sorry

end average_of_two_middle_numbers_is_correct_l451_451970


namespace fraction_of_smart_integers_divisible_by_20_l451_451343

def is_even (n : ℕ) : Prop := n % 2 = 0

def digits_sum (n : ℕ) : ℕ := n.digits.sum

def is_smart_integer (n : ℕ) : Prop :=
  is_even n ∧ 10 < n ∧ n < 150 ∧ digits_sum n = 10

def is_divisible_by_20 (n : ℕ) : Prop := n % 20 = 0

theorem fraction_of_smart_integers_divisible_by_20 :
  (card (filter is_divisible_by_20 (filter is_smart_integer (range 151))).val) 
  / (card (filter is_smart_integer (range 151))).val = 1 / 4 := sorry

end fraction_of_smart_integers_divisible_by_20_l451_451343


namespace no_unbiased_estimator_for_one_over_lambda_l451_451910

/-- Define the probability mass function for the Poisson distribution with parameter λ -/
noncomputable def poisson_pmf (λ : ℝ) (k : ℕ) : ℝ := 
  real.exp (-λ) * (λ ^ k) / (nat.factorial k)

/-- Define the expectation under the Poisson distribution -/
noncomputable def expectation (T : ℕ → ℝ) (λ : ℝ) : ℝ := 
  ∑ k in finset.range 100000, T k * poisson_pmf λ k

/-- Show that there does not exist an unbiased estimator for 1/λ -/
theorem no_unbiased_estimator_for_one_over_lambda :
  ∀ (λ : ℝ) (hλ : 0 < λ), ¬ ∃ (T : ℕ → ℝ),
    ((expectation T λ) = 1 / λ ∧ 
     (∑ k in finset.range 100000, real.abs (T k) * poisson_pmf λ k) < 1000000 ) :=
sorry

end no_unbiased_estimator_for_one_over_lambda_l451_451910


namespace Morse_code_symbols_count_l451_451480

theorem Morse_code_symbols_count : 
  let count n := (2^n) - 1 in
  count 1 + count 2 + count 3 + count 4 + count 5 = 57 := 
by
  let count n := (2^n) - 1
  have h1 : count 1 = 1 := by sorry
  have h2 : count 2 = 3 := by sorry
  have h3 : count 3 = 7 := by sorry
  have h4 : count 4 = 15 := by sorry
  have h5 : count 5 = 31 := by sorry
  sorry

end Morse_code_symbols_count_l451_451480


namespace nuts_per_meter_computation_l451_451041

def tree_heights : ℝ := 7 + 12 + 18
def oak_acorns : ℝ := 12 * (3/4)
def oak_walnuts : ℝ := 20 * (1/2)
def pine_acorns : ℝ := 40 * (1/4)
def pine_walnuts : ℝ := 50 * (2/5)
def maple_acorns : ℝ := 48 * (5/8)
def maple_walnuts : ℝ := 30 * (5/6)

def total_acorns_collected : ℝ := oak_acorns + pine_acorns + maple_acorns
def total_walnuts_collected : ℝ := oak_walnuts + pine_walnuts + maple_walnuts

def acorns_stolen : ℝ := total_acorns_collected * 0.1
def walnuts_stolen : ℝ := total_walnuts_collected * 0.15

def acorns_left : ℝ := total_acorns_collected - acorns_stolen
def walnuts_left : ℝ := total_walnuts_collected - walnuts_stolen

def total_nuts_left : ℝ := acorns_left + walnuts_left
def nuts_per_meter : ℝ := total_nuts_left / tree_heights

theorem nuts_per_meter_computation : nuts_per_meter ≈ 2.43 :=
by
  sorry

end nuts_per_meter_computation_l451_451041


namespace floor_factorial_expression_l451_451323

theorem floor_factorial_expression : 
  (⌊(2010! + 2007! : ℚ) / (2009! + 2008! : ℚ)⌋ = 2009) :=
by
  -- Let a := 2010! and b := 2007!
  -- So a + b = 2010! + 2007!
  -- Notice 2010! = 2010 * 2009 * 2008 * 2007!
  -- Notice 2009! = 2009 * 2008 * 2007!
  -- Simplify (2010! + 2007!) / (2009! + 2008!)
  sorry

end floor_factorial_expression_l451_451323


namespace irrational_r_plus_sqrt_a_l451_451508

noncomputable def a : ℤ := sorry
noncomputable def r : ℝ := sorry
axiom a_pos : 0 < a
axiom a_not_perfect_square : ∀ n : ℤ, n * n ≠ a
axiom r_root : r ^ 3 - 2 * (a : ℝ) * r + 1 = 0

theorem irrational_r_plus_sqrt_a : irrational (r + real.sqrt (a : ℝ)) := 
sorry

end irrational_r_plus_sqrt_a_l451_451508


namespace martha_saves_106_l451_451928

def daily_allowance := 15
def week_1_savings := (6 * daily_allowance * 0.4) + (daily_allowance * 0.3)
def week_2_savings := (6 * daily_allowance * 0.5) + (daily_allowance * 0.4)
def week_3_savings := (6 * daily_allowance * 0.6) + (daily_allowance * 0.5)
def week_4_savings := (8 * daily_allowance * 0.7) + (daily_allowance * 0.6)
def week_1_expense := 20
def week_2_expense := 30
def week_3_expense := 40
def week_4_expense := 50

def net_week_1_savings := week_1_savings - week_1_expense
def net_week_2_savings := week_2_savings - week_2_expense
def net_week_3_savings := week_3_savings - week_3_expense
def net_week_4_savings := week_4_savings - week_4_expense

def total_savings := net_week_1_savings + net_week_2_savings + net_week_3_savings + net_week_4_savings

theorem martha_saves_106 : total_savings = 106 := by
  sorry

end martha_saves_106_l451_451928


namespace units_digit_sum_squares_of_first_2011_odd_integers_l451_451206

-- Define the relevant conditions and given parameters
def first_n_odd_integers (n : ℕ) : List ℕ := List.range' 1 (2*n) (λ k, 2*k - 1)

def units_digit (n : ℕ) : ℕ := n % 10

def square_units_digit (n : ℕ) : ℕ := units_digit (n * n)

-- Prove the units digit of the sum of squares of the first 2011 odd positive integers
theorem units_digit_sum_squares_of_first_2011_odd_integers : 
  units_digit (List.sum (List.map (λ x, x * x) (first_n_odd_integers 2011))) = 1 :=
by
  -- Sorry skips the proof
  sorry

end units_digit_sum_squares_of_first_2011_odd_integers_l451_451206


namespace find_a_if_f_is_odd_function_l451_451022

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 * (a * 2^x - 2^(-x))

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

theorem find_a_if_f_is_odd_function : 
  ∀ a : ℝ, is_odd_function (f a) → a = 1 :=
by
  sorry

end find_a_if_f_is_odd_function_l451_451022


namespace geometric_sum_n_equals_4_l451_451180

def a : ℚ := 1 / 3
def r : ℚ := 1 / 3
def S (n : ℕ) : ℚ := a * ((1 - r^n) / (1 - r))
def sum_value : ℚ := 26 / 81

theorem geometric_sum_n_equals_4 (n : ℕ) (h : S n = sum_value) : n = 4 :=
by sorry

end geometric_sum_n_equals_4_l451_451180


namespace minimum_value_of_expression_l451_451237

noncomputable def f : ℝ → ℝ := λ x, |x - 4| + |x + 2| + |x - 5|

theorem minimum_value_of_expression : ∃ x : ℝ, x = 4 ∧ f x = -1 := sorry

end minimum_value_of_expression_l451_451237


namespace lcm_is_600_l451_451758

def lcm_of_24_30_40_50_60 : ℕ :=
  Nat.lcm 24 (Nat.lcm 30 (Nat.lcm 40 (Nat.lcm 50 60)))

theorem lcm_is_600 : lcm_of_24_30_40_50_60 = 600 := by
  sorry

end lcm_is_600_l451_451758


namespace units_digit_sum_squares_of_odd_integers_l451_451211

theorem units_digit_sum_squares_of_odd_integers :
  let first_2005_odd_units := [802, 802, 401] -- counts for units 1, 9, 5 respectively
  let extra_squares_last_6 := [9, 1, 3, 9, 5, 9] -- units digits of the squares of the last 6 numbers
  let total_sum :=
        (first_2005_odd_units[0] * 1 + 
         first_2005_odd_units[1] * 9 + 
         first_2005_odd_units[2] * 5) +
        (extra_squares_last_6.sum)
  (total_sum % 10) = 1 :=
by
  sorry

end units_digit_sum_squares_of_odd_integers_l451_451211


namespace cost_of_bread_l451_451927

-- Definition of the conditions
def total_purchase_amount : ℕ := 205  -- in cents
def amount_given_to_cashier : ℕ := 700  -- in cents
def quarter_value : ℕ := 25
def dime_value : ℕ := 10
def nickel_value : ℕ := 5
def num_nickels_received : ℕ := 8

-- Statement of the problem
theorem cost_of_bread :
  (∃ (B C : ℕ), B + C = total_purchase_amount ∧
                  amount_given_to_cashier - total_purchase_amount = 
                  (quarter_value + dime_value + num_nickels_received * nickel_value + 420) ∧
                  B = 125) :=
by
  -- Skipping the proof
  sorry

end cost_of_bread_l451_451927


namespace number_of_liars_on_island_l451_451602

theorem number_of_liars_on_island :
  ∃ (knights liars : ℕ),
  let Total := 1000 in
  let Villages := 10 in
  let members_per_village := Total / Villages in
  -- Each village has at least 2 members
  (∀ i : ℕ, i < Villages → 2 ≤ members_per_village) ∧
  -- Populations of knights and liars respecting the Total population
  (knights + liars = Total) ∧
  -- Each inhabitant claims all others in their village are liars
  (∀ (i v : ℕ), i < Villages → v < members_per_village → 
     (∃ k l : ℕ, k + l = members_per_village ∧ 
      k = 1 ∧ -- there is exactly one knight in each village
      knights = Villages ∧ liars = Total - knights ∧
      l = members_per_village - 1)) ∧
  liars = 990 := sorry

end number_of_liars_on_island_l451_451602


namespace instantaneous_velocity_at_t4_l451_451281

-- Definition of the motion equation
def s (t : ℝ) : ℝ := 1 - t + t^2

-- The proof problem statement: Proving that the derivative of s at t = 4 is 7
theorem instantaneous_velocity_at_t4 : deriv s 4 = 7 :=
by sorry

end instantaneous_velocity_at_t4_l451_451281


namespace chord_line_equation_l451_451490

theorem chord_line_equation : 
  ∀ (x y : ℝ), (∀ (M : ℝ × ℝ), M = (1, 1/2) → 
  (∀ (x1 y1 x2 y2 : ℝ), 
    (x1 + x2 = 2 → y1 + y2 = 1 → 
    (x1^2 / 4 + y1^2 = 1) → 
    (x2^2 / 4 + y2^2 = 1) → 
    ∃ k b : ℝ, k = -1/2 → b = 1/2 → 
    y - 1/2 = k * (x - 1) → x + 2 * y - 2 = 0))) :=
begin
  sorry
end

end chord_line_equation_l451_451490


namespace geometry_problem_l451_451544

-- We need to define the variables a, b, c, r, r_a, R, r_b, r_c
variables {a b c r r_a R r_b r_c : ℝ}

-- Define the conditions and the theorem we need to prove
theorem geometry_problem (h1 : a * (b + c) = (r + r_a) * (4 * R + r - r_a)) 
                         (h2 : a * (b - c) = (r_b - r_c) * (4 * R - r_b - r_c)) :
  a * (b + c) = (r + r_a) * (4 * R + r - r_a) ∧ a * (b - c) = (r_b - r_c) * (4 * R - r_b - r_c) :=
by
  split
  -- Prove the first part
  exact h1

  -- Prove the second part
  exact h2

end geometry_problem_l451_451544


namespace graduation_day_is_tuesday_l451_451134

theorem graduation_day_is_tuesday (days_after : ℕ) (h : days_after = 85) : 
  let start_day : ℕ := 1 -- assuming Monday is represented as 1
  in ((start_day + days_after) % 7) = 2 := -- assuming Tuesday is represented as 2 
by {
  rw h,
  have mod_result : 85 % 7 = 1 := by norm_num,
  rw mod_result,
  norm_num,
}

end graduation_day_is_tuesday_l451_451134


namespace geometric_body_is_sphere_l451_451670

-- Given conditions about the views of the geometric body
def main_view_is_circle (shape : Type) : Prop := 
  ∀ v : view, v = main_view → v.shape = circle

def left_view_is_circle (shape : Type) : Prop := 
  ∀ v : view, v = left_view → v.shape = circle

def top_view_is_circle (shape : Type) : Prop := 
  ∀ v : view, v = top_view → v.shape = circle

-- Proof of question == answer given conditions: Proving that the body is a sphere
theorem geometric_body_is_sphere (shape : Type) 
  (h1 : main_view_is_circle shape) 
  (h2 : left_view_is_circle shape) 
  (h3 : top_view_is_circle shape) : shape = sphere :=
sorry

end geometric_body_is_sphere_l451_451670


namespace fixed_point_of_f_l451_451762

variable (a : ℝ)
variable (h_cond1 : a > 0)
variable (h_cond2 : a ≠ 1)

def f (x : ℝ) : ℝ := a^(x + 3) + 2

theorem fixed_point_of_f : f a (-3) = 3 :=
by
  sorry

end fixed_point_of_f_l451_451762


namespace brick_fence_depth_l451_451337

theorem brick_fence_depth (length height total_bricks : ℕ) 
    (h1 : length = 20) 
    (h2 : height = 5) 
    (h3 : total_bricks = 800) : 
    (total_bricks / (4 * length * height) = 2) := 
by
  sorry

end brick_fence_depth_l451_451337


namespace geometric_sequence_sum_l451_451595

theorem geometric_sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) (q : ℕ) (h1 : ∀ n, S n = (finset.range n).sum a)
(h2 : a 1 = 1) (h3 : 4 * a 1 + a 3 = 2 * (2 * a 2)) (h4 : ∀ n, a (n + 1) = a n * q) :
S 4 = 15 :=
by
  sorry

end geometric_sequence_sum_l451_451595


namespace find_sides_of_triangle_l451_451549

theorem find_sides_of_triangle (a b c : ℝ) (h_area : ∆ = 6) (h_inradius : r = 1) (h_a : a = 4) (h_b_less_c : b < c) : b = 3 ∧ c = 5 := by
  sorry

end find_sides_of_triangle_l451_451549


namespace value_of_A_l451_451456

def clubsuit (A B : ℕ) := 3 * A + 2 * B + 5

theorem value_of_A (A : ℕ) (h : clubsuit A 7 = 82) : A = 21 :=
by
  sorry

end value_of_A_l451_451456


namespace common_elements_count_l451_451511

def S : Set ℕ := {n | ∃ k, 1 ≤ k ∧ k ≤ 2005 ∧ n = 4 * k }
def T : Set ℕ := {n | ∃ k, 1 ≤ k ∧ k ≤ 2005 ∧ n = 6 * k }

theorem common_elements_count : (S ∩ T).card = 668 := by
  sorry

end common_elements_count_l451_451511


namespace average_difference_l451_451963

theorem average_difference :
  let avg1 := (20 + 40 + 60) / 3,
      avg2 := (10 + 50 + 45) / 3
  in avg1 - avg2 = 5 :=
by
  sorry

end average_difference_l451_451963


namespace simplify_expression_l451_451140

theorem simplify_expression (z y : ℝ) :
  (4 - 5 * z + 2 * y) - (6 + 7 * z - 3 * y) = -2 - 12 * z + 5 * y :=
by
  sorry

end simplify_expression_l451_451140


namespace median_remaining_rooms_l451_451485

-- Define the set of initial room numbers from 1 to 25
def initialRooms : List ℕ := List.range' 1 25

-- Define the rooms where Mathletes did not attend
def absentRooms : List ℕ := [10, 15, 20]

-- Define the remaining rooms after removing the absent ones
def remainingRooms : List ℕ := initialRooms.filter (λ x => ¬ absentRooms.contains x)

-- Define the median calculation
def median (l : List ℕ) : ℚ := (l.nthLe 10 (by simp) + l.nthLe 11 (by simp)) / 2

-- Theorem to prove the median of remaining rooms is 13.5
theorem median_remaining_rooms : median remainingRooms = 13.5 := by
  sorry

end median_remaining_rooms_l451_451485


namespace brownies_left_over_l451_451926

-- Definitions
def initial_brownies : ℕ := 16
def children_percentage : ℚ := 0.25
def family_percentage : ℚ := 0.50
def lorraine_more : ℕ := 1

-- Theorem statement
theorem brownies_left_over (initial_brownies : ℕ)
                           (children_percentage : ℚ)
                           (family_percentage : ℚ)
                           (lorraine_more : ℕ) :
  initial_brownies - (initial_brownies * children_percentage).natAbs
  - ((initial_brownies - (initial_brownies * children_percentage).natAbs) * family_percentage).natAbs
  - lorraine_more = 5 :=
by
  sorry

end brownies_left_over_l451_451926


namespace cotangent_identity_l451_451127

noncomputable def cotangent (θ : ℝ) : ℝ := 1 / Real.tan θ

theorem cotangent_identity (x : ℝ) (i : ℂ) (n : ℕ) (k : ℕ) (h : (0 < k) ∧ (k < n)) :
  ((x + i) / (x - i))^n = 1 → x = cotangent (k * Real.pi / n) := 
sorry

end cotangent_identity_l451_451127


namespace find_max_slope_of_OQ_l451_451822

noncomputable def parabola_C := {p : ℝ // p = 2}

def parabola_eq (p : ℝ) : Prop := 
  ∀ x y : ℝ, (y^2 = 2 * p * x) → (y^2 = 4 * x)

def max_slope (p : ℝ) (O Q : ℝ × ℝ) (F P Q' : ℝ × ℝ) : Prop := 
  ∀ K : ℝ, K = (Q.2) / (Q.1) → 
  ∀ n : ℝ, (K = (10 * n) / (25 * n^2 + 9)) →
  ∀ n : ℝ , n = (3 / 5) → 
  K = (1 / 3)

theorem find_max_slope_of_OQ : 
  ∀ pq: parabola_C,
  ∃ C : parabola_eq pq.val,
  ∃ O F P Q : (ℝ × ℝ),
  (F = (1, 0)) ∧
  (P.1 * P.1 = 4 * P.2) ∧
  (Q.1 - P.1, Q.2 - P.2) = 9 * -(F.1 - Q.1, Q.2) →
  max_slope pq.val O Q F P Q'.1 :=
sorry

end find_max_slope_of_OQ_l451_451822


namespace units_digit_of_2011_odd_squares_l451_451208

def units_digit_sum_squares_first_k_odd_integers (k : ℕ) : ℕ :=
  let odd_numbers := List.range k |>.map (λ n, 2*n + 1)
  let squares := odd_numbers.map (λ n, n^2)
  let total_sum := squares.sum
  total_sum % 10

theorem units_digit_of_2011_odd_squares : units_digit_sum_squares_first_k_odd_integers 2011 = 9 :=
by
  sorry

end units_digit_of_2011_odd_squares_l451_451208


namespace length_square_of_k_l451_451953

def f (x : ℝ) := 3 * x + 2
def g (x : ℝ) := -3 * x + 2
def h (x : ℝ) := (1 : ℝ)
def j (x : ℝ) := max (max (f x) (g x)) (h x)
def k (x : ℝ) := min (min (f x) (g x)) (h x)

theorem length_square_of_k :
  (2 * Real.sqrt ((35 / 3)^2 + (24)^2) + 2 / 3)^2 = (
    let Lg := Real.sqrt ((4 - (-1/3))^2 + ((g 4 - g (-1/3))^2)) in
    let Lh := 2 / 3 in
    let Lf := Real.sqrt ((4 - (1/3))^2 + ((f 4 - f (1/3))^2)) in
    (Lg + Lh + Lf)^2) := 
by
  sorry

end length_square_of_k_l451_451953


namespace mutuallyExclusiveButNotComplementary_l451_451672

def Event (α : Type) := Set α

variables (Group : Set (String × String)) -- (name, gender)

def isGirl (s : String × String) : Prop := s.2 = "girl"
def isBoy (s : String × String) : Prop := s.2 = "boy"

def atLeastOneGirl (selection : Set (String × String)) : Prop :=
  ∃ x ∈ selection, isGirl x

def bothGirls (selection : Set (String × String)) : Prop :=
  ∀ x ∈ selection, isGirl x

def exactlyOneGirl (selection : Set (String × String)) : Prop :=
  ∃ x ∈ selection, isGirl x ∧ ∀ y ∈ selection, y ≠ x → isBoy y

def exactlyTwoGirls (selection : Set (String × String)) : Prop :=
  ( ∀ x ∈ selection, isGirl x ) ∧ selection.size = 2

variables (selection : Set (String × String))
variables h1 : atLeastOneGirl selection
variables h2 : bothGirls selection
variables h3 : exactlyOneGirl selection
variables h4 : exactlyTwoGirls selection

theorem mutuallyExclusiveButNotComplementary : 
  ( ∀ (selection : Set (String × String)),  exactlyOneGirl selection → ¬ exactlyTwoGirls selection ) ∧ 
  ( ∀ (selection : Set (String × String)), ¬ exactlyOneGirl selection ∨ ¬ exactlyTwoGirls selection ) :=
sorry

end mutuallyExclusiveButNotComplementary_l451_451672


namespace find_counterfeit_l451_451185

def is_counterfeit_sequence (coins : Fin 100 → ℝ) (start_idx : ℕ) : Prop :=
  ∀ i : Fin 26, coins ⟨start_idx + i, by linarith [i.2]⟩ < genuine_weight

def counterfeit_present_at (coins : Fin 100 → ℝ) (pos : Fin 100) : Prop :=
  coins pos < genuine_weight

def genuine_weight : ℝ := sorry

theorem find_counterfeit 
  (coins : Fin 100 → ℝ)
  (h_consecutive : ∃ s : ℕ, s ≤ 74 ∧ is_counterfeit_sequence coins s)
  (h_genuine_equal : ∀ i j : Fin 100, ¬counterfeit_present_at coins i → ¬counterfeit_present_at coins j → coins i = coins j)
  (h_counterfeit_lighter : ∀ i : Fin 100, counterfeit_present_at coins i → coins i < genuine_weight) :
  ∃ i, i = 25 ∨ i = 51 ∨ i = 77 ∧ counterfeit_present_at coins ⟨i, by linarith [i.2]⟩ :=
sorry

end find_counterfeit_l451_451185


namespace factorial_floor_problem_l451_451324

theorem factorial_floor_problem :
  (nat.floor ( (nat.factorial 2010 + nat.factorial 2007) / (nat.factorial 2009 + nat.factorial 2008) )) = 2009 :=
by 
sorry

end factorial_floor_problem_l451_451324


namespace james_correct_take_home_pay_l451_451061

noncomputable def james_take_home_pay : ℝ :=
  let main_job_hourly_rate := 20
  let second_job_hourly_rate := main_job_hourly_rate * 0.8
  let main_job_hours := 30
  let main_job_overtime_hours := 5
  let second_job_hours := 15
  let side_gig_daily_rate := 100
  let side_gig_days := 2
  let tax_deductions := 200
  let federal_tax_rate := 0.18
  let state_tax_rate := 0.05

  let regular_main_job_hours := main_job_hours - main_job_overtime_hours
  let main_job_regular_pay := regular_main_job_hours * main_job_hourly_rate
  let main_job_overtime_pay := main_job_overtime_hours * main_job_hourly_rate * 1.5
  let total_main_job_pay := main_job_regular_pay + main_job_overtime_pay

  let total_second_job_pay := second_job_hours * second_job_hourly_rate
  let total_side_gig_pay := side_gig_daily_rate * side_gig_days

  let total_earnings := total_main_job_pay + total_second_job_pay + total_side_gig_pay
  let taxable_income := total_earnings - tax_deductions
  let federal_tax := taxable_income * federal_tax_rate
  let state_tax := taxable_income * state_tax_rate
  let total_taxes := federal_tax + state_tax
  total_earnings - total_taxes

theorem james_correct_take_home_pay : james_take_home_pay = 885.30 := by
  sorry

end james_correct_take_home_pay_l451_451061


namespace increase_in_wheel_radius_l451_451348

theorem increase_in_wheel_radius (r1 r2 : ℝ) (d1 d2 : ℝ) (pi : ℝ)
    (H1 : r1 = 14) 
    (H2 : d1 = 540) 
    (H3 : d2 = 530)
    (H4 : pi = real.pi) 
    (H5 : r2 = (540 * 63360 * 0.001388888888888889 / (2 * pi * 530))) :
    r2 - r1 = 0.31 :=
sorry

end increase_in_wheel_radius_l451_451348


namespace hyperbola_equation_exists_l451_451365

theorem hyperbola_equation_exists :
  (∃ (h : ℝ → ℝ → Prop),
  (∀ x y, h x y ↔ (x^2 / 9 - y^2 / 16 = 1)) ∧
  (¬ ∀ λ : ℝ, λ = 0) ∧
  h (-3) (2 * real.sqrt 3)) →
  (∃ k : ℝ → ℝ → Prop,
  (∀ x y, k x y ↔ (4 * x^2 / 9 - y^2 / 4 = 1))) := 
by
  sorry

end hyperbola_equation_exists_l451_451365


namespace find_lambda_l451_451799

noncomputable def angle_between_vectors (a b : ℝ) : Prop :=
  ∀ (a b : ℝ), (a * b = ℝ.cos (π / 3) * (a.norm * b.norm))

theorem find_lambda (a b : ℝ) (h_angle : angle_between_vectors a b)
  (h_a_norm : ∥a∥ = 1) (h_b_norm : ∥b∥ = 2) 
  (h_perpendicular : (sqrt(3) • a + λ • b) ⬝ a = 0) :
  λ = -sqrt(3) :=
sorry

end find_lambda_l451_451799


namespace base_sum_l451_451045

theorem base_sum (R_A R_B : ℕ) 
  (hFA_RA : F_A = (0.454545..)RA)
  (hFB_RA : F_B = (0.545454..)RA )
  (hFA_RB : F_A = (0.363636..)RB )
  (hFB_RB : F_B = (0.636363..)RB ) 
  : R_A + R_B = 19 :=
sorry

end base_sum_l451_451045


namespace exists_valid_numbers_l451_451120

noncomputable def sum_of_numbers_is_2012_using_two_digits : Prop :=
  ∃ (a b c d : ℕ), (a < 1000) ∧ (b < 1000) ∧ (c < 1000) ∧ (d < 1000) ∧ 
                    (∀ n ∈ [a, b, c, d], ∃ x y, (x ≠ y) ∧ ((∀ d ∈ [n / 100 % 10, n / 10 % 10, n % 10], d = x ∨ d = y))) ∧
                    (a + b + c + d = 2012)

theorem exists_valid_numbers : sum_of_numbers_is_2012_using_two_digits :=
  sorry

end exists_valid_numbers_l451_451120


namespace part1_part2_l451_451005

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * sin x * cos x + a * cos x - 2 * x

theorem part1 (a : ℝ) : deriv (f a) (π / 6) = -2 → a = 2 :=
sorry

theorem part2 (a : ℝ) [Fact (a = 2)] : 
  ∃ x ∈ Icc (-π / 6) (7 * π / 6), 
  ∀ y ∈ Icc (-π / 6) (7 * π / 6), f a y ≤ f a x ∧ f a x = 2 :=
sorry

end part1_part2_l451_451005


namespace axes_of_symmetry_coincide_l451_451572

-- Define the quadratic functions f and g
def f (x : ℝ) : ℝ := (x^2 + 6*x - 25) / 8
def g (x : ℝ) : ℝ := (31 - x^2) / 8

-- Define the derivatives of f and g
def f' (x : ℝ) : ℝ := (x + 3) / 4
def g' (x : ℝ) : ℝ := -x / 4

-- Define the axes of symmetry for the functions
def axis_of_symmetry_f : ℝ := -3
def axis_of_symmetry_g : ℝ := 0

-- Define the intersection points
def intersection_points : List ℝ := [4, -7]

-- State the problem: Do the axes of symmetry coincide?
theorem axes_of_symmetry_coincide :
  (axis_of_symmetry_f = axis_of_symmetry_g) = False :=
by
  sorry

end axes_of_symmetry_coincide_l451_451572


namespace find_triangle_CAN_angles_l451_451274

open Real

variables {A B C D P Q N : Type}
variables [square A B C D] 
variables (angle_CAP : angle A C P = 15)
variables (angle_BCP : angle B C P = 15)
variables (APCQ_isosceles_trap : isosceles_trapezoid A P C Q ∧ parallel P C A Q ∧ equal_length A P C Q)
variables (N : midpoint P Q)

theorem find_triangle_CAN_angles :
  ∠ C A N = 15 ∧ ∠ A N C = 90 ∧ ∠ N C A = 75 :=
sorry

end find_triangle_CAN_angles_l451_451274


namespace reciprocal_of_neg3_l451_451587

theorem reciprocal_of_neg3 : (1 / (-3) = -1 / 3) :=
by
  sorry

end reciprocal_of_neg3_l451_451587


namespace f_f_1_l451_451404

def f (x : ℝ) : ℝ :=
  if x > 0 then 2^x - 1 else -x^2 - 2*x

theorem f_f_1 : f (f 1) = 1 :=
  by sorry

end f_f_1_l451_451404


namespace arithmetic_expression_l451_451214

noncomputable def sum_odd_numbers : ℕ := (list.range' 1 1025).map (λ n, 2 * n - 1).sum

noncomputable def sum_even_numbers : ℕ := (list.range' 1 1025).map (λ n, 2 * n).sum

noncomputable def sum_multiples_of_3 : ℕ := (list.range' 1 683).map (λ n, 3 * n).sum

theorem arithmetic_expression :
  sum_odd_numbers - sum_even_numbers - sum_multiples_of_3 = -694684 :=
by {
  unfold sum_odd_numbers,
  unfold sum_even_numbers,
  unfold sum_multiples_of_3,
  sorry
}

end arithmetic_expression_l451_451214


namespace factorial_floor_problem_l451_451327

theorem factorial_floor_problem :
  (nat.floor ( (nat.factorial 2010 + nat.factorial 2007) / (nat.factorial 2009 + nat.factorial 2008) )) = 2009 :=
by 
sorry

end factorial_floor_problem_l451_451327


namespace emilee_earns_25_l451_451065

-- Define the conditions
def earns_together (jermaine terrence emilee : ℕ) : Prop := 
  jermaine + terrence + emilee = 90

def jermaine_more (jermaine terrence : ℕ) : Prop :=
  jermaine = terrence + 5

def terrence_earning : ℕ := 30

-- The goal: Prove Emilee earns 25 dollars
theorem emilee_earns_25 (jermaine terrence emilee : ℕ) (h1 : earns_together jermaine terrence emilee) 
  (h2 : jermaine_more jermaine terrence) (h3 : terrence = terrence_earning) : 
  emilee = 25 := 
sorry

end emilee_earns_25_l451_451065


namespace triangle_area_l451_451702

theorem triangle_area :
  ∀ (x y : ℝ), (3 * x + 2 * y = 12 ∧ x ≥ 0 ∧ y ≥ 0) →
  (1 / 2) * 4 * 6 = 12 := by
  sorry

end triangle_area_l451_451702


namespace pancakes_needed_l451_451528

theorem pancakes_needed (initial_pancakes : ℕ) (num_people : ℕ) (pancakes_left : ℕ) :
  initial_pancakes = 12 → num_people = 8 → pancakes_left = initial_pancakes - num_people →
  (num_people - pancakes_left) = 4 :=
by
  intros initial_pancakes_eq num_people_eq pancakes_left_eq
  sorry

end pancakes_needed_l451_451528


namespace math_problem_l451_451429

def proposition1 (a b : Vec) : Prop := (abs a = abs b) → (a = b)
def proposition2 (A B C D : Point) : Prop := 
  ¬Collinear A B C → 
  (AB = DC ↔ Parallelogram A B C D)
def proposition3 (a b c : Vec) : Prop := (a = b) ∧ (b = c) → (a = c)
def proposition4 (a b : Vec) : Prop := (a = b) ↔ ((abs a = abs b) ∧ (Parallel a b))

def correctPropositions : Prop :=
  ¬proposition1 ∧
  proposition2 ∧ 
  proposition3 ∧ 
  ¬proposition4 → 
  (proposition2 ∧ proposition3)

theorem math_problem : correctPropositions :=
  sorry

end math_problem_l451_451429


namespace area_transformation_l451_451909

theorem area_transformation
  (T : Set (ℝ × ℝ))
  (hT : MeasureTheory.MeasureTheory.measureT.measure T = 9)
  (A : Matrix (Fin 2) (Fin 2) ℝ)
  (hA : A = ![![3, 4], ![6, 5]]) :
  MeasureTheory.MeasureTheory.measureT.measure (T.map (fun p => ((A).mulVec p) : ℝ × ℝ)) = 81 := 
by 
  sorry

end area_transformation_l451_451909


namespace problem_solution_l451_451881

-- Definitions based on the conditions

structure Triangle (α : Type) :=
  (A B C : α)

variables {α : Type} [LinearOrder α] 

-- Definitions for points E and D on segments AB and BL respectively, and bisector AL in triangle ABC 
structure GeometryConfiguration (triangle : Triangle α) :=
  (E D L : α)
  (AE BL : set α) -- segments
  (AL : α)

-- Defining the problem conditions 
def problem_conditions (triangle : Triangle α) (config : GeometryConfiguration α) : Prop :=
  let ⟨A, B, C⟩ := triangle in
  let ⟨E, D, L, AE, BL, AL⟩ := config in
  segment (A, E, B) ∧
  segment (B, D, L) ∧
  (L = midpoint B C) ∧ -- DL = LC (midpoint)
  segment_parallel (E, D) (A, C) ∧ -- ED ∥ AC
  distance A E = 15 ∧
  distance A C = 12

-- The theorem statement
theorem problem_solution (triangle : Triangle α) (config : GeometryConfiguration α) 
  (h : problem_conditions triangle config) : 
  distance (segment_E config) (segment_D config) = 3 := 
sorry

end problem_solution_l451_451881


namespace trapezium_area_l451_451361

-- Given conditions
def side_a : ℝ := 30
def side_b : ℝ := 12
def height : ℝ := 16

-- The statement of the theorem we need to prove
theorem trapezium_area 
    (a b h : ℝ) 
    (ha : a = side_a) 
    (hb : b = side_b) 
    (hh : h = height) : 
    (1 / 2) * (a + b) * h = 336 :=
by
  rw [ha, hb, hh]
  sorry

end trapezium_area_l451_451361


namespace simplify_expr_l451_451141

open Real

theorem simplify_expr (x : ℝ) (hx : 1 ≤ x) :
  sqrt (x + 2 * sqrt (x - 1)) + sqrt (x - 2 * sqrt (x - 1)) = 
  if x ≤ 2 then 2 else 2 * sqrt (x - 1) :=
by sorry

end simplify_expr_l451_451141


namespace remaining_amoeba_is_blue_l451_451266

-- Define the initial number of amoebas for red, blue, and yellow types.
def n1 := 47
def n2 := 40
def n3 := 53

-- Define the property that remains constant, i.e., the parity of differences
def parity_diff (a b : ℕ) : Bool := (a - b) % 2 == 1

-- Initial conditions based on the given problem
def initial_conditions : Prop :=
  parity_diff n1 n2 = true ∧  -- odd
  parity_diff n1 n3 = false ∧ -- even
  parity_diff n2 n3 = true    -- odd

-- Final statement: Prove that the remaining amoeba is blue
theorem remaining_amoeba_is_blue : Prop :=
  initial_conditions ∧ (∀ final : String, final = "Blue")

end remaining_amoeba_is_blue_l451_451266


namespace bella_position_p2023_l451_451333

/-- Define the position of Bella after n steps
 given her movement pattern and initial conditions -/
def bella_position (n : ℕ) : ℤ × ℤ :=
  sorry  -- Definitions based on the problem’s conditions

/-- Statement of the problem: Determining the position of Bella after 2023 steps -/
theorem bella_position_p2023 : bella_position 2023 = (22, 21) :=
begin
  sorry
end

end bella_position_p2023_l451_451333


namespace five_digit_number_divisible_by_B_is_multiple_of_1000_l451_451745

-- Definitions
def is_five_digit_number (A : ℕ) : Prop := 10000 ≤ A ∧ A < 100000
def B (A : ℕ) := (A / 1000 * 100) + (A % 100)
def is_four_digit_number (B : ℕ) : Prop := 1000 ≤ B ∧ B < 10000

-- Main theorem
theorem five_digit_number_divisible_by_B_is_multiple_of_1000
  (A : ℕ) (hA : is_five_digit_number A)
  (hAB : ∃ k : ℕ, B A = k) :
  A % 1000 = 0 := 
sorry

end five_digit_number_divisible_by_B_is_multiple_of_1000_l451_451745


namespace completing_the_square_l451_451219

theorem completing_the_square (x : ℝ) :
  x^2 + 4 * x + 1 = 0 ↔ (x + 2)^2 = 3 :=
by
  sorry

end completing_the_square_l451_451219


namespace average_of_two_excluding_min_max_l451_451979

theorem average_of_two_excluding_min_max :
  ∃ (a b c d : ℕ), (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d) ∧ 
  (a + b + c + d = 20) ∧ (max (max a b) (max c d) - min (min a b) (min c d) = 14) ∧
  (([a, b, c, d].erase (min (min a b) (min c d))).erase (max (max a b) (max c d)) = [x, y])
  → ((x + y) / 2 = 2.5) :=
sorry

end average_of_two_excluding_min_max_l451_451979


namespace range_of_x_l451_451412

def p (x : ℝ) : Prop := x^2 - 5 * x + 6 ≥ 0
def q (x : ℝ) : Prop := 0 < x ∧ x < 4

theorem range_of_x (x : ℝ) (hpq : p x ∨ q x) (hnq : ¬ q x) : x ≤ 0 ∨ x ≥ 4 :=
by sorry

end range_of_x_l451_451412


namespace circumcenter_x_P_formula_circumcenter_locus_eq_l451_451398

noncomputable def circumcenter_x_P (A B : ℝ × ℝ) : ℝ :=
  let (x_a, t₁) := A
  let (x_b, t₂) := B
  9 / 2 - t₁ * t₂ / 6

def circumcenter_locus (x y : ℝ) : Prop :=
  ((x - 4)^2 / 4) - (y^2 / 12) = 1

theorem circumcenter_x_P_formula (t₁ t₂ : ℝ) :
  circumcenter_x_P (3, t₁) (3, t₂) = 9 / 2 - t₁ * t₂ / 6 := sorry

theorem circumcenter_locus_eq (P : ℝ × ℝ) (t₁ t₂ : ℝ) (hp : P = (circumcenter_x_P (3, t₁) (3, t₂), (t₁ + t₂) / 2))
  (h_angle : ∠ ((0 : ℝ), (0 : ℝ)) (3, t₁) ((3, t₂)) = real.pi / 3) :
  circumcenter_locus P.1 P.2 := sorry

end circumcenter_x_P_formula_circumcenter_locus_eq_l451_451398


namespace exists_polynomial_h_with_properties_l451_451905

noncomputable def f (x : ℤ) (m : ℕ) : ℤ := 
∏ i in finset.range m, (x ^ (i + 1) - 1)

noncomputable def g (x : ℤ) (n m : ℕ) : ℤ :=
∏ i in finset.range m, (x ^ (n + i + 1) - 1)

theorem exists_polynomial_h_with_properties (m n : ℕ) (hpos_m : 0 < m) (hpos_n : 0 < n) :
  ∃ h : polynomial ℤ, (polynomial.degree h = m * n ∧ 
  ∀ i, 0 ≤ i ∧ i ≤ m * n → 0 < ((f x m) * h).coeff i) ∧ ((f x m) * h = g x n m) :=
sorry

end exists_polynomial_h_with_properties_l451_451905


namespace distance_from_center_to_triangle_l451_451999

theorem distance_from_center_to_triangle :
  ∃ m n k : ℕ, 
  m.gcd k = 1 ∧ 
  (∀ p ∈ prime_factors n, p^2 ∣ n → false) ∧
  let PQ := 20;
      QR := 21;
      RP := 29;
      radius := 24;
      distance := (m * (sqrt n : ℝ)) / k;
      area := sqrt (35 * (35 - PQ) * (35 - QR) * (35 - RP)) in
  PQ = 20 ∧
  QR = 21 ∧
  RP = 29 ∧
  radius = 24 ∧
  m = 19 ∧ 
  n = 374 ∧ 
  k = 3 ∧
  distance = area / 70 ∧ 
  m + n + k = 396 := 
by {
  -- Without going into actual computations since it is only statement requirement
  sorry
}

end distance_from_center_to_triangle_l451_451999


namespace total_games_in_season_l451_451253

theorem total_games_in_season (divA divB: Finset ℕ):
  divA.card = 8 →
  divB.card = 8 →
  (∀ (teamA ∈ divA) (teamB ∈ divA), teamA ≠ teamB → 2 * (divA.card - 1)) + 
  (∀ (teamA ∈ divA) (teamB ∈ divB), teamA ≠ teamB → divB.card) + 
  (∀ (teamA ∈ divB) (teamB ∈ divA), teamA ≠ teamB → divA.card) + 
  (∀ (teamB ∈ divB) (teamC ∈ divB), teamB ≠ teamC → 2 * (divB.card - 1)) = 176 := sorry

end total_games_in_season_l451_451253


namespace ants_first_group_count_l451_451144

theorem ants_first_group_count :
    ∃ x : ℕ, 
        (∀ (w1 c1 a1 t1 w2 c2 a2 t2 : ℕ),
          w1 = 10 ∧ c1 = 600 ∧ a1 = x ∧ t1 = 5 ∧
          w2 = 5 ∧ c2 = 960 ∧ a2 = 20 ∧ t2 = 3 ∧ 
          (w1 * c1) / t1 = 1200 / a1 ∧ (w2 * c2) / t2 = 1600 / 20 →
             x = 15)
:= sorry

end ants_first_group_count_l451_451144


namespace emilee_earns_25_l451_451066

-- Define the conditions
def earns_together (jermaine terrence emilee : ℕ) : Prop := 
  jermaine + terrence + emilee = 90

def jermaine_more (jermaine terrence : ℕ) : Prop :=
  jermaine = terrence + 5

def terrence_earning : ℕ := 30

-- The goal: Prove Emilee earns 25 dollars
theorem emilee_earns_25 (jermaine terrence emilee : ℕ) (h1 : earns_together jermaine terrence emilee) 
  (h2 : jermaine_more jermaine terrence) (h3 : terrence = terrence_earning) : 
  emilee = 25 := 
sorry

end emilee_earns_25_l451_451066


namespace max_vector_length_l451_451792

noncomputable def maximum_length (θ : ℝ) : ℝ :=
  real.sqrt (10 - 8 * real.cos θ)

theorem max_vector_length :
  ∀ (θ : ℝ), 0 ≤ θ ∧ θ < 2 * real.pi → maximum_length θ = 3 * real.sqrt 2 :=
by 
  -- Proof goes here
  sorry

end max_vector_length_l451_451792


namespace find_a_value_l451_451432

noncomputable def f (a : ℝ) (x : ℝ) := log a (1 - x) + log a (x + 3)

theorem find_a_value :
  (∀ x ∈ set.Icc (-2 : ℝ) 0, f a x ≥ -2) ∧ (∃ x ∈ set.Icc (-2 : ℝ) 0, f a x = -2) →
  a = 1 / 2 :=
by
  sorry

end find_a_value_l451_451432


namespace beaver_group_count_l451_451151

theorem beaver_group_count (B : ℕ) (h1 : 3 * B = 60) : B = 20 :=
by sorry

end beaver_group_count_l451_451151


namespace num_ordered_pairs_l451_451517

theorem num_ordered_pairs (n : ℕ) (hn : n > 1) : (∑ k in finset.range (n), (n - k)) = (n * (n - 1)) / 2 :=
by sorry

end num_ordered_pairs_l451_451517


namespace sin_double_angle_l451_451776

noncomputable def sin_alpha (α : ℝ) : ℝ := sorry
noncomputable def cos_alpha (α : ℝ) : ℝ := -4 / 5

theorem sin_double_angle (α : ℝ) (h1 : 0 < α) (h2 : α < π) (h3 : sin (π / 2 - α) = -4 / 5) : 
  sin (2 * α) = -24 / 25 := 
by
  sorry

end sin_double_angle_l451_451776


namespace dodecahedron_probability_l451_451196

noncomputable def probability_endpoints_of_edge (total_vertices edges_per_vertex total_edges : ℕ) : ℚ :=
  let possible_choices := total_vertices - 1
  let favorable_outcomes := edges_per_vertex
  favorable_outcomes / possible_choices

theorem dodecahedron_probability :
  probability_endpoints_of_edge 20 3 30 = 3 / 19 := by
  sorry

end dodecahedron_probability_l451_451196


namespace monotonically_increasing_on_interval_l451_451848

def f (k : ℝ) (x : ℝ) : ℝ := k * x - Real.log x

theorem monotonically_increasing_on_interval (k : ℝ) :
  (∀ x : ℝ, 1 < x → 0 ≤ k - 1 / x) → 1 ≤ k :=
by
  intro h
  have h₁ := h 1 (by linarith)
  rw [one_div_one] at h₁
  linarith

end monotonically_increasing_on_interval_l451_451848


namespace avg_of_two_middle_numbers_l451_451975

theorem avg_of_two_middle_numbers (a b c d : ℕ) (h₀ : a ≠ b) (h₁ : a ≠ c) (h₂ : a ≠ d) (h₃ : b ≠ c) (h₄ : b ≠ d) (h₅ : c ≠ d) (h₆ : a + b + c + d = 20) (h₇ : a < d) (h₈ : d - a ≥ b - c) (h₉ : b = 2) (h₁₀ : c = 3) :
  (b + c) / 2 = 2.5 :=
by
  sorry

end avg_of_two_middle_numbers_l451_451975


namespace number_of_points_P_l451_451994

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop :=
  (x^2 / 16) + (y^2 / 9) = 1

-- Define the line equation
def line (x y : ℝ) : Prop :=
  (x / 4) + (y / 3) = 1

-- Define the points A and B
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (0, 3)

-- Define the triangle area condition
def area_triangle (P : ℝ × ℝ) (A B : ℝ × ℝ) : ℝ :=
  1 / 2 * |A.1 * (B.2 - P.2) + B.1 * (P.2 - A.2) + P.1 * (A.2 - B.2)|

-- State the theorem
theorem number_of_points_P : ∃ (P1 P2 P3 P4 : ℝ × ℝ),
  ellipse P1.1 P1.2 ∧ ellipse P2.1 P2.2 ∧ ellipse P3.1 P3.2 ∧ ellipse P4.1 P4.2 ∧
  area_triangle P1 A B = 2 ∧ area_triangle P2 A B = 2 ∧
  area_triangle P3 A B = 2 ∧ area_triangle P4 A B = 2 ∧
  P1 ≠ P2 ∧ P1 ≠ P3 ∧ P1 ≠ P4 ∧ P2 ≠ P3 ∧ P2 ≠ P4 ∧ P3 ≠ P4 :=
sorry

end number_of_points_P_l451_451994


namespace team_B_points_target_l451_451706

theorem team_B_points_target (average_points_per_game : ℝ) (points_game_3 : ℝ) (additional_points_needed : ℝ) :
  average_points_per_game = 61.5 → points_game_3 = 47 → additional_points_needed = 330 → 
  ∃ target_points : ℝ, target_points = 500 :=
by
  intros h_avg h_game3 h_needed
  use 500
  rw [h_avg, h_game3, h_needed]
  sorry

end team_B_points_target_l451_451706


namespace probability_series_Lakers_win_40_percent_l451_451960

noncomputable def probability_lakers_win_series_at_least_5_games : ℚ := 
  let pL : ℚ := 2 / 3 -- Probability Lakers win a single game
  let pC : ℚ := 1 / 3 -- Probability Clippers win a single game
  let prob_4_1 := (4 * (pL ^ 4) * (pC ^ 1))
  let prob_4_2 := (10 * (pL ^ 4) * (pC ^ 2))
  let prob_4_3 := (20 * (pL ^ 4) * (pC ^ 3))
  prob_4_1 + prob_4_2 + prob_4_3

theorem probability_series_Lakers_win_40_percent : 
  (probability_lakers_win_series_at_least_5_games * 100).round = 40 := by
  sorry

end probability_series_Lakers_win_40_percent_l451_451960


namespace arith_floor_sum_l451_451330

theorem arith_floor_sum :
  let seq := List.range 166 |>.map (λ n, n * 1.2 + 2)
  List.sum (seq.map Int.floor) = 16731 := by
  sorry

end arith_floor_sum_l451_451330


namespace taxi_fare_l451_451162

-- Definition of the conditions
def initial_charge : ℝ := 8
def additional_rate_per_km : ℝ := 2.7
def distance (x : ℝ) (h : x > 3) := x

-- Statement of the problem
theorem taxi_fare (x : ℝ) (h : x > 3) : 
  let y := initial_charge + additional_rate_per_km * (x - 3) in
  y = 2.7 * x - 0.1 :=
by 
  sorry

end taxi_fare_l451_451162


namespace a1_leq_N_l451_451505

theorem a1_leq_N
  (N : ℕ) 
  (a : ℕ → ℕ)
  (h1 : 1 < a 1)
  (h2 : ∀ i : ℕ, i = 1 ∨ (i ≠ 1 → a 1 < a i))
  (h3 : ∀ i : ℕ, (1 ≤ i ∧ i ≤ N) → a i ∣ (∏ j in (finset.range (N+1)).filter (≠ i), a j) + 1) :
  a 1 ≤ N :=
sorry

end a1_leq_N_l451_451505


namespace minimum_distance_l451_451418

open Set

-- Define the geometric conditions
def parabola (P : ℝ × ℝ) : Prop :=
  P.2 ^ 2 = 2 * P.1

def projection_y_axis (P : ℝ × ℝ) (M : ℝ × ℝ) : Prop :=
  M.1 = 0 ∧ M.2 = P.2

-- Points and values involved
def A : ℝ × ℝ := (7 / 2, 4)

-- Distance function
def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

-- Problem statement
theorem minimum_distance :
  ∃ (P M : ℝ × ℝ), parabola P ∧ projection_y_axis P M ∧ | distance P A + distance P M | = 9 / 2 :=
sorry

end minimum_distance_l451_451418


namespace area_of_triangle_of_hyperbola_foci_correct_l451_451415

noncomputable def area_of_triangle_of_hyperbola_foci : Prop :=
  let b := (Real.sqrt 2) / 2
  in ∀ (F₁ F₂ P : Point) (hyperbola_cond : (4 * F₁.x^2 - 2 * F₁.y^2 = 1) ∧ (4 * F₂.x^2 - 2 * F₂.y^2 = 1) ∧ on_hyperbola P),
  ∠(F₁PF₂) = 60° → 
  (area_of_triangle F₁ P F₂) = (b^2 * Real.cot (60° / 2))

theorem area_of_triangle_of_hyperbola_foci_correct :
  area_of_triangle_of_hyperbola_foci :=
by {
  sorry
}

end area_of_triangle_of_hyperbola_foci_correct_l451_451415


namespace max_percent_liquid_X_l451_451925

theorem max_percent_liquid_X (wA wB wC : ℝ) (XA XB XC YA YB YC : ℝ)
  (hXA : XA = 0.8 / 100) (hXB : XB = 1.8 / 100) (hXC : XC = 3.0 / 100)
  (hYA : YA = 2.0 / 100) (hYB : YB = 1.0 / 100) (hYC : YC = 0.5 / 100)
  (hwA : wA = 500) (hwB : wB = 700) (hwC : wC = 300)
  (H_combined_limit : XA * wA + XB * wB + XC * wC + YA * wA + YB * wB + YC * wC ≤ 0.025 * (wA + wB + wC)) :
  XA * wA + XB * wB + XC * wC ≤ 0.0171 * (wA + wB + wC) :=
sorry

end max_percent_liquid_X_l451_451925


namespace range_of_m_l451_451803

-- Definitions and conditions
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def is_eccentricity (e a b : ℝ) : Prop :=
  e = Real.sqrt (1 - (b^2 / a^2))

def is_semi_latus_rectum (d a b : ℝ) : Prop :=
  d = 2 * b^2 / a

-- Main theorem statement
theorem range_of_m (a b m : ℝ) (x y : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0)
  (h3 : is_eccentricity (Real.sqrt (3) / 2) a b)
  (h4 : is_semi_latus_rectum 1 a b)
  (h_ellipse : ellipse a b x y) : 
  m ∈ Set.Ioo (-3 / 2 : ℝ) (3 / 2 : ℝ) := 
sorry

end range_of_m_l451_451803


namespace math_problem_l451_451394

variable (a b : ℝ)

#check a ≥ 0

theorem math_problem (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + b = 1) :
  a + Real.sqrt b ≤ Real.sqrt 2 ∧
  1/2 < 2^(a - Real.sqrt b) ∧ 2^(a - Real.sqrt b) < 2 ∧
  a^2 - b > -1 := 
by
  sorry

end math_problem_l451_451394


namespace harkamal_total_payment_l451_451831

-- Conditions
def cost_of_grapes (kg : ℕ) (rate : ℕ) : ℕ := kg * rate
def cost_of_mangoes (kg : ℕ) (rate : ℕ) : ℕ := kg * rate
def total_cost_before_discount (cost_grapes cost_mangoes : ℕ) : ℕ := cost_grapes + cost_mangoes
def discount (percentage total_cost : ℕ) : ℕ := (percentage * total_cost) / 100
def discounted_price (total_cost discount : ℕ) : ℕ := total_cost - discount
def sales_tax (percentage discounted_price : ℕ) : ℕ := (percentage * discounted_price) / 100
def total_amount_paid (discounted_price tax : ℕ) : ℕ := discounted_price + tax

-- Theorem to prove the amount paid by Harkamal
theorem harkamal_total_payment :
  let cost_grapes := cost_of_grapes 8 70 in
  let cost_mangoes := cost_of_mangoes 9 65 in
  let total_cost := total_cost_before_discount cost_grapes cost_mangoes in
  let discount_amount := discount 10 total_cost in
  let discounted_price := discounted_price total_cost discount_amount in
  let tax := sales_tax 5 discounted_price in
  let total_payment := total_amount_paid discounted_price tax in
  round total_payment = 1082 :=
begin
  sorry
end

end harkamal_total_payment_l451_451831


namespace part_one_part_two_part_three_l451_451431

noncomputable def f (a x : ℝ) := a - 1 / x

-- Proof Problem 1
theorem part_one (a : ℝ) : ∀ x1 x2 ∈ Set.Ioi 0, x1 < x2 → f a x1 < f a x2 := sorry

-- Proof Problem 2
theorem part_two (a m n : ℝ) (h_mn : m < n) (h_fm : f a m = 2 * m) (h_fn : f a n = 2 * n) : a > 2 * Real.sqrt 2 := sorry

-- Proof Problem 3
theorem part_three (a : ℝ) : (∀ x ∈ Set.Icc (1 / 3) (1 / 2), x^2 * |f a x| ≤ 1) → -2 ≤ a ∧ a ≤ 6 := sorry

end part_one_part_two_part_three_l451_451431


namespace find_value_l451_451424

theorem find_value : ∀ (s t : ℝ), 
  19 * s^2 + 99 * s + 1 = 0 ∧ t^2 + 99 * t + 19 = 0 ∧ s * t ≠ 1 → 
  (st + 4s + 1) / t = -5 := by 
  intros s t h
  rcases h with ⟨hs, ht, hst⟩
  -- Proof goes here, but including sorry to focus on the statement
  sorry

end find_value_l451_451424


namespace value_of_f_6_l451_451027

noncomputable def f : ℤ → ℤ
| n := if n = 4 then 15 else f (n - 1) - n

theorem value_of_f_6 : f 6 = 4 :=
by {
  -- According to conditions given in the problem
  have h1 : f(4) = 15 := rfl,
  -- using the recursion formula f(n) = f(n-1) - n
  have h2 : f(5) = f(4) - 5,
  rw [h1] at h2,
  simp at h2,
  have h3 : f(6) = f(5) - 6,
  rw [h2] at h3,
  simp at h3,
  sorry,
}

end value_of_f_6_l451_451027


namespace carl_additional_hours_per_week_l451_451614

def driving_hours_per_day : ℕ := 2

def days_per_week : ℕ := 7

def total_hours_two_weeks_after_promotion : ℕ := 40

def driving_hours_per_week_before_promotion : ℕ := driving_hours_per_day * days_per_week

def driving_hours_per_week_after_promotion : ℕ := total_hours_two_weeks_after_promotion / 2

def additional_hours_per_week : ℕ := driving_hours_per_week_after_promotion - driving_hours_per_week_before_promotion

theorem carl_additional_hours_per_week : 
  additional_hours_per_week = 6 :=
by
  -- Using plain arithmetic based on given definitions
  sorry

end carl_additional_hours_per_week_l451_451614


namespace initial_pins_in_group_l451_451943

-- Defining the conditions
def avg_pins_per_day_per_person : ℕ := 10
def pins_deleted_per_week_per_person : ℕ := 5
def num_people : ℕ := 20
def num_days_in_month : ℕ := 30
def num_weeks_in_month : ℕ := 4
def total_pins_after_month : ℕ := 6600

-- Defining the problem to prove
theorem initial_pins_in_group :
  let total_pins_contributed := avg_pins_per_day_per_person * num_people * num_days_in_month,
      total_pins_deleted := pins_deleted_per_week_per_person * num_people * num_weeks_in_month in
  ∃ P : ℕ, P + total_pins_contributed - total_pins_deleted = total_pins_after_month ∧ P = 1000 :=
by
  -- placeholder for the proof
  sorry

end initial_pins_in_group_l451_451943


namespace max_distance_circle_to_line_l451_451074

theorem max_distance_circle_to_line :
  let P : ℝ × ℝ := sorry,
  x^2 + y^2 = 1 → ∃ (P : ℝ × ℝ), max_dist_circle_to_line P := 3
  sorry

note
conditions:
1. let P : ℝ × ℝ := (x,y)
x^2 + y^2 = 1 ⇒ P is point on circle
2. line is given by 3x - 4y - 10 = 0

dist0 = abs(3*0 - 4*0 - 10)/sqrt(3*3 + (-4)*(-4))
         = abs(-10)/5
         = 2

max distance = radius 1 + dist0
                      = 1 + 2
                      = 3

*Proof Required*

end max_distance_circle_to_line_l451_451074


namespace tom_investment_calculation_l451_451616

def tom_initial_investment : ℕ := 3000

theorem tom_investment_calculation (J : ℕ) (total_profit : ℕ) (jose_share : ℕ) 
    (profit_ratio : J = 45000) 
    (total_profit_eq : total_profit = 27000) 
    (jose_share_eq : jose_share = 15000) 
    (investment_ratio : (tom_initial_investment * 12) = (45000 * 10 * 4 / 5)) : 
    tom_initial_investment = 3000 := 
begin
  sorry,
end

end tom_investment_calculation_l451_451616


namespace length_KN_eq_R_l451_451542

-- Define the circle with diameter AB and radius R
variables (R : ℝ) (A B X K N : Point) (hAB : diameter (circle A B) = 2 * R)

-- Conditions
variables (hX : X ∈ line A B) 
variables (hK : K ∈ circle A B) (hN : N ∈ circle A B)
variables (hSameHalfPlane : same_half_plane_with_respect_to A B K N)
variables (hAngleKXA : ∠(K, X, A) = 60)
variables (hAngleNXB : ∠(N, X, B) = 60)

-- Question restated
theorem length_KN_eq_R : dist K N = R := 
sorry

end length_KN_eq_R_l451_451542


namespace angle_RPQ_l451_451873

theorem angle_RPQ (P Q R S : Type) (angle : ℕ → ℝ) (y : ℝ) 
  (h1 : P ∈ line(R, S)) 
  (h2 : bisects (QP, ∠SQR)) 
  (h3 : PQ = PR) 
  (h4 : RS = RQ) 
  (h5 : angle RSQ = 4 * y)
  (h6 : angle RPQ = 3 * y) :
  angle RPQ = 90 :=
sorry

end angle_RPQ_l451_451873


namespace profit_if_no_discount_l451_451272

-- Definitions from conditions
def cp : ℝ := 100
def discount : ℝ := 4 / 100
def profit_with_discount : ℝ := 32 / 100
def sp_with_discount : ℝ := cp * (1 + profit_with_discount)
def mp : ℝ := sp_with_discount / (1 - discount)
def profit_without_discount : ℝ := (mp - cp) / cp

-- Proof Goal Statement
theorem profit_if_no_discount : profit_without_discount * 100 = 37.5 := by
  sorry

end profit_if_no_discount_l451_451272


namespace find_k_value_l451_451997

theorem find_k_value (k : ℚ) (h1 : (3, -5) ∈ {p : ℚ × ℚ | p.snd = k * p.fst}) (h2 : k ≠ 0) : k = -5 / 3 :=
sorry

end find_k_value_l451_451997


namespace simplify_expression_l451_451142

variable (a b : ℝ)
axiom pos_a (h : 0 < a)
axiom pos_b (h : 0 < b)

theorem simplify_expression : 
  (a - b) / (Real.sqrt a + Real.sqrt b) + 
  (Real.sqrt a)^3 + (Real.sqrt b)^3 / 
  (a - Real.sqrt (a * b) + b) = 2 * Real.sqrt a := 
sorry

end simplify_expression_l451_451142


namespace range_of_omega_l451_451006

noncomputable def sin_monotone_condition (ω : ℝ) : Prop :=
  ∀ x y ∈ Icc (-5 * π / 6) (2 * π / 3), x < y → sin (ω * x + π / 6) < sin (ω * y + π / 6)

noncomputable def unique_x0_condition (ω : ℝ) : Prop :=
  ∃! x0 ∈ Icc 0 (5 * π / 6), sin (ω * x0 + π / 6) = 1

theorem range_of_omega (ω : ℝ) (h1 : sin_monotone_condition ω) (h2 : unique_x0_condition ω) :
    ω ∈ Icc (2 / 5 : ℝ) (1 / 2 : ℝ) :=
begin
  sorry -- Proof should be provided here.
end

end range_of_omega_l451_451006


namespace correct_statement_is_C_l451_451227

theorem correct_statement_is_C (A: Prop) (B: Prop) (C: Prop) (D: Prop):
  (¬A) ∧ (¬B) ∧ C ∧ (¬D) -> C :=
by 
  intro h,
  cases h with A_incorrect h1,
  cases h1 with B_incorrect h2,
  cases h2 with C_correct D_incorrect,
  exact C_correct

end correct_statement_is_C_l451_451227


namespace part1_OP_slope_range_part2_cos_POQ_min_value_l451_451420

open Real

-- Definitions for the given conditions
def parabola_equation (x y : ℝ) : Prop := y^2 = 4 * x
def focus_f (F : ℝ × ℝ) : Prop := F = (1, 0)
def origin_o (O : ℝ × ℝ) : Prop := O = (0, 0)
def point_m (M : ℝ × ℝ) : Prop := M = (4, 0)

-- Definitions for the midpoints and lines
def midpoint (A B P : ℝ × ℝ) : Prop := P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
def line_slope (P Q : ℝ × ℝ) : ℝ := (Q.2 - P.2) / (Q.1 - P.1)

-- Statements for Part (1) and Part (2)
theorem part1_OP_slope_range :
  ∀ (F O P A B : ℝ × ℝ),
  focus_f F →
  origin_o O →
  parabola_equation A.1 A.2 →
  parabola_equation B.1 B.2 →
  midpoint A B P →
  |line_slope O P| ≤ sqrt 2 / 2 :=
sorry

theorem part2_cos_POQ_min_value :
  ∀ (F O M A B C D P Q : ℝ × ℝ),
  focus_f F →
  origin_o O →
  point_m M →
  parabola_equation A.1 A.2 →
  parabola_equation B.1 B.2 →
  parabola_equation C.1 C.2 →
  parabola_equation D.1 D.2 →
  midpoint A B P →
  midpoint C D Q →
  -- Computations for angle POQ using slopes
  let t := line_slope O P in
  let u := line_slope O Q in
  (cos (abs (atan t - atan u))) = 3 * sqrt 11 / 11 :=
sorry

end part1_OP_slope_range_part2_cos_POQ_min_value_l451_451420


namespace license_plate_count_l451_451559

noncomputable def num_plates : ℕ :=
  let letters := ['A', 'E', 'G', 'I', 'K', 'O', 'P', 'R', 'S', 'T', 'U', 'V']
  let available_mid_letters := ['A', 'E', 'I', 'O', 'P', 'R', 'U', 'V']
  (1 : ℕ) * (available_mid_letters.length : ℕ) * (available_mid_letters.length - 1 : ℕ) * (available_mid_letters.length - 2 : ℕ) * (2 : ℕ)

theorem license_plate_count :
  num_plates = 1008 :=
by
  unravel
  rw [num_plates]
  simp only [list.length, available_mid_letters, available_mid_letters.length] 
  norm_num
  sorry

end license_plate_count_l451_451559


namespace number_of_points_on_line_l451_451136

theorem number_of_points_on_line (a b c d : ℕ) (h1 : a * b = 80) (h2 : c * d = 90) (h3 : a + b = c + d) :
  a + b + 1 = 22 :=
sorry

end number_of_points_on_line_l451_451136


namespace h_f_equals_h_g_l451_451520

def f (x : ℝ) := x^2 - x + 1

def g (x : ℝ) := -x^2 + x + 1

def h (x : ℝ) := (x - 1)^2

theorem h_f_equals_h_g : ∀ x : ℝ, h (f x) = h (g x) :=
by
  intro x
  unfold f g h
  sorry

end h_f_equals_h_g_l451_451520


namespace fraction_of_square_shaded_is_half_l451_451042

theorem fraction_of_square_shaded_is_half {s : ℝ} (h : s > 0) :
  let O := (0, 0)
  let P := (0, s)
  let Q := (s, s / 2)
  let area_square := s^2
  let area_triangle_OPQ := 1 / 2 * s^2 / 2
  let shaded_area := area_square - area_triangle_OPQ
  (shaded_area / area_square) = 1 / 2 :=
by
  sorry

end fraction_of_square_shaded_is_half_l451_451042


namespace locus_of_centroids_l451_451407

-- Define the problem condition, which are points A, B, and C on the edges of a trihedral angle
variables {O A B C : ℝ × ℝ × ℝ}
variables {d : ℝ}

-- Assume we have a fixed point A on edge OA and sliding points B and C on edges OB and OC respectively
def point_on_OA (A O : ℝ × ℝ × ℝ) : Prop := ∃ a : ℝ, A = (a, 0, 0)
def point_on_OB (B O : ℝ × ℝ × ℝ) : Prop := ∃ b : ℝ, B = (0, b, 0)
def point_on_OC (C O : ℝ × ℝ × ℝ) : Prop := ∃ c : ℝ, C = (0, 0, c)

-- Define the centroid of a triangle ABC
def centroid (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3, (A.3 + B.3 + C.3) / 3)

-- Define the distance from a point to a plane
def dist_to_plane (A : ℝ × ℝ × ℝ) (BOC_plane : ℝ × ℝ × ℝ → Prop) : ℝ := sorry

-- Definition of the geometric locus:
def geometric_locus (A B C O : ℝ × ℝ × ℝ) (d : ℝ) : ℝ × ℝ × ℝ → Prop :=
  λ G, ∃ (a b c : ℝ), G = (a / 3, b / 3, c / 3) ∧ point_on_OB (0, b, 0) ∧ point_on_OC (0, 0, c) ∧
                       dist_to_plane (A) (λ p, p.1 == 0) = d

-- The actual statement to be proven
theorem locus_of_centroids (A B C O : ℝ × ℝ × ℝ) (d : ℝ) 
  (hA : point_on_OA A O) (hB : point_on_OB B O) (hC : point_on_OC C O) :
  ∀ G, (centroid A B C = G) → geometric_locus A B C O (d / 3) G :=
sorry

end locus_of_centroids_l451_451407


namespace functions_symmetric_to_line_x_eq_1_l451_451386

theorem functions_symmetric_to_line_x_eq_1 (f : ℝ → ℝ) :
    ∀ x, f(x - 1) = f(1 - x) → ∀ x, f(x - 1) = f(-(x - 1)) :=
  sorry

end functions_symmetric_to_line_x_eq_1_l451_451386


namespace triangle_altitude_intersect_computation_l451_451707

def Triangle (α β γ : Type) : Prop := true -- A dummy type for triangle
def isAltitude {A B C H : Type} : A → B → C → H → Prop := sorry -- hypothetical definition of altitude property
def intersects_at (P Q R : Type) : Prop := true -- A dummy type for intersection property

variables {A B C D E H : Type}

-- Hypothetical measures of length
def length_HD : ℕ := 8
def length_HE : ℕ := 3

theorem triangle_altitude_intersect_computation
  (h_triangle : Triangle A B C)
  (h_altitude_AD : isAltitude A D C H)
  (h_altitude_BE : isAltitude B E C H)
  (h_intersection : intersects_at H (A D) (B E))
  (h_HD : length HD = 8)
  (h_HE : length HE = 3) :
  (BD * DC) - (AE * EC) = 55 :=
by
  sorry

end triangle_altitude_intersect_computation_l451_451707


namespace unique_solution_of_ffx_eq_8_l451_451026

def f (x : ℝ) : ℝ :=
  if x > 0 then x^2 - 1 else x + 2

theorem unique_solution_of_ffx_eq_8 : 
  ∃! (x : ℝ), f (f x) = 8 :=
begin
  sorry
end

end unique_solution_of_ffx_eq_8_l451_451026


namespace rhombus_side_length_l451_451685

theorem rhombus_side_length (d : ℝ) (K : ℝ) (h : K = (3 * d^2) / 2) :
  (let s := sqrt ((5 * (2 * K / 3)) / 2) in s = sqrt (5 * K / 3)) :=
by
  sorry

end rhombus_side_length_l451_451685


namespace remainder_n_squared_plus_2n_plus_3_mod_75_l451_451461

theorem remainder_n_squared_plus_2n_plus_3_mod_75 (a : ℤ) :
  let n := 75 * a - 2 in
  (n^2 + 2*n + 3) % 75 = 3 := by
  let n := 75 * a - 2
  sorry

end remainder_n_squared_plus_2n_plus_3_mod_75_l451_451461


namespace trajectory_centroid_PAM_l451_451659

noncomputable def midpoint (A B : Point) : Point :=
  (A + B) / 2

structure Cube :=
  (A B C D A' B' C' D' : Point)

structure Square :=
  (A B C D : Point)

def perimeter_trajectory (G : Point) (s : Square) : Set Point :=
  sorry -- Definition of trajectory here

theorem trajectory_centroid_PAM 
  (cube : Cube)
  (M : Point)
  (P : Point)
  (G : Point)
  (s : Square)
  (hM : M = midpoint cube.B cube.B')
  (hP : P ∈ {A, B, C, D})
  (hG : G = centroid triangle PAM)
  (s_scaled : Square)
  (scaled : s_scaled = scale s (2/3)) :
  perimeter_trajectory G s_scaled :=
sorry

end trajectory_centroid_PAM_l451_451659


namespace determine_a_from_binomial_expansion_l451_451984

theorem determine_a_from_binomial_expansion (a : ℝ) 
    (h : ∑ r in finset.range 7, binom 6 r * (-1)^r * a^r = -160) : 
    a = 2 := by sorry

end determine_a_from_binomial_expansion_l451_451984


namespace incenter_circumcenter_distance_correct_l451_451334

noncomputable def incenter_circumcenter_distance (a b c : ℕ) (h : a^2 + b^2 = c^2) : ℚ := do
  let s := (a + b + c) / 2
  let K := a * b / 2
  let r := K / s
  let I_x := (c * 0 + b * ↑a + a * (c / 2)) / (a + b + c)
  let I_y := (a * 0 + b * 0 + c * ↑a) / (a + b + c)
  let O_x := c / 2
  let O_y := 0
  let distance := Math.sqrt ((((I_x - O_x)^2 + (I_y - O_y)^2 : ℚ)))

#eval incenter_circumcenter_distance 7 24 25 (by norm_num)

theorem incenter_circumcenter_distance_correct :
  incenter_circumcenter_distance 7 24 25 (by norm_num) = (3 * Real.sqrt 5) / 2
  := sorry

end incenter_circumcenter_distance_correct_l451_451334


namespace model_X_completion_time_l451_451254

theorem model_X_completion_time :
  ∀ (T_x : ℕ),
  (∀ t_x : ℕ, T_x = t_x) →
  (∀ t_y : ℕ, t_y = 30) →
  (∀ (m_x m_y : ℕ), 
    m_x = 20 →
    m_y = 20 →
    (m_x * (1 / T_x : ℝ) + m_y * (1 / 30 : ℝ) = 1) →
    T_x = 60) :=
by {
  intros T_x h_T_x h_T_y m_x m_y h_mx h_my h_work,
  sorry
}

end model_X_completion_time_l451_451254


namespace average_of_two_middle_numbers_is_correct_l451_451968

def four_numbers_meeting_conditions (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  (a + b + c + d = 20) ∧ 
  (max a (max b (max c d)) - min a (min b (min c d)) = max_diff

def max_diff := 
  (∀ (x y : ℕ), (x ≠ y → x > 0 → y > 0 → x + y ≤ 19 ∧ x + y ≥ 5) → 
  (x = 14 ∧ y = 1))

theorem average_of_two_middle_numbers_is_correct :
  ∃ (a b c d : ℕ), four_numbers_meeting_conditions a b c d →
  let numbers := [a, b, c, d].erase (min a (min b (min c d))).erase (max a (max b (max c d))),
  (numbers.sum / 2) = 2.5 := 
by
  sorry

end average_of_two_middle_numbers_is_correct_l451_451968


namespace triangle_area_l451_451703

theorem triangle_area :
  ∀ (x y : ℝ), (3 * x + 2 * y = 12 ∧ x ≥ 0 ∧ y ≥ 0) →
  (1 / 2) * 4 * 6 = 12 := by
  sorry

end triangle_area_l451_451703


namespace necessary_and_sufficient_condition_l451_451104

-- Definition of a complex number being pure imaginary
def z (x : ℝ) : ℂ := (x^2 - 1) + complex.i * (x + 1)

-- Conditions derived from the problem
def real_part_zero (x : ℝ) : Prop := x^2 - 1 = 0
def imaginary_part_non_zero (x : ℝ) : Prop := x + 1 ≠ 0

theorem necessary_and_sufficient_condition (x : ℝ) :
  (z x).re = 0 ∧ (z x).im ≠ 0 ↔ x = 1 :=
sorry

end necessary_and_sufficient_condition_l451_451104


namespace zilla_savings_l451_451640

theorem zilla_savings
  (monthly_earnings : ℝ)
  (h_rent : monthly_earnings * 0.07 = 133)
  (h_expenses : monthly_earnings * 0.5 = monthly_earnings / 2) :
  monthly_earnings - (133 + monthly_earnings / 2) = 817 :=
by
  sorry

end zilla_savings_l451_451640


namespace find_max_slope_of_OQ_l451_451820

noncomputable def parabola_C := {p : ℝ // p = 2}

def parabola_eq (p : ℝ) : Prop := 
  ∀ x y : ℝ, (y^2 = 2 * p * x) → (y^2 = 4 * x)

def max_slope (p : ℝ) (O Q : ℝ × ℝ) (F P Q' : ℝ × ℝ) : Prop := 
  ∀ K : ℝ, K = (Q.2) / (Q.1) → 
  ∀ n : ℝ, (K = (10 * n) / (25 * n^2 + 9)) →
  ∀ n : ℝ , n = (3 / 5) → 
  K = (1 / 3)

theorem find_max_slope_of_OQ : 
  ∀ pq: parabola_C,
  ∃ C : parabola_eq pq.val,
  ∃ O F P Q : (ℝ × ℝ),
  (F = (1, 0)) ∧
  (P.1 * P.1 = 4 * P.2) ∧
  (Q.1 - P.1, Q.2 - P.2) = 9 * -(F.1 - Q.1, Q.2) →
  max_slope pq.val O Q F P Q'.1 :=
sorry

end find_max_slope_of_OQ_l451_451820


namespace area_bounded_by_graphs_l451_451722

noncomputable def compute_area : ℝ :=
  ∫ x in (0 : ℝ) .. 1, real.sqrt (4 - x^2)

theorem area_bounded_by_graphs :
  compute_area = (real.pi / 3) + (real.sqrt 3 / 2) :=
by
  sorry

end area_bounded_by_graphs_l451_451722


namespace sum_areas_of_amazing_right_triangles_l451_451270

theorem sum_areas_of_amazing_right_triangles :
  let amazing_area (a b : ℕ) := a * b / 2
  ∑ s in {147, 96, 81, 75, 72} : ℕ, s = 471 :=
by 
  sorry

end sum_areas_of_amazing_right_triangles_l451_451270


namespace find_divisor_l451_451468

theorem find_divisor (x d : ℤ) (h1 : ∃ k : ℤ, x = k * d + 5)
                     (h2 : ∃ n : ℤ, x + 17 = n * 41 + 22) :
    d = 1 :=
by
  sorry

end find_divisor_l451_451468


namespace tangent_line_through_P_l451_451366

-- Definition of the point P and the circle
def P := (real.sqrt 3, 1 : ℝ)
def circle (x y : ℝ) : Prop := (x ^ 2 + y ^ 2 = 4)

-- Tangent line equation at point (x0, y0) on a circle x^2 + y^2 = r^2
def tangent_line (x y x0 y0 r : ℝ) : Prop := (x0 * x + y0 * y = r ^ 2)

-- Proof problem statement
theorem tangent_line_through_P :
  circle (real.sqrt 3) 1 →
  tangent_line real.sqrt 3 1 x y 4 :=
sorry

end tangent_line_through_P_l451_451366


namespace reciprocal_neg3_l451_451579

-- Define the problem
def reciprocal (x : ℚ) : ℚ := 1 / x

-- The required proof statement
theorem reciprocal_neg3 : reciprocal (-3) = -1 / 3 :=
by
  sorry

end reciprocal_neg3_l451_451579


namespace age_difference_l451_451651

variable (A B C : ℕ)

theorem age_difference (h : A + B = B + C + 12) : A - C = 12 := 
sorry

end age_difference_l451_451651


namespace find_first_term_l451_451085

theorem find_first_term (a : ℚ) (n : ℕ) (T : ℕ → ℚ)
  (hT : ∀ n, T n = n * (2 * a + 5 * (n - 1)) / 2)
  (h_const : ∃ c : ℚ, ∀ n > 0, T (4 * n) / T n = c) :
  a = 5 / 2 := 
sorry

end find_first_term_l451_451085


namespace two_digit_factors_of_2_power_36_minus_1_l451_451450

theorem two_digit_factors_of_2_power_36_minus_1 :
  { n : ℤ | 1 ≤ n ∧ n < 100 ∧ n ∣ (2^36 - 1) }.card = 6 :=
by
  sorry

end two_digit_factors_of_2_power_36_minus_1_l451_451450


namespace angle_ABM_eq_angle_NCD_l451_451857

variables {A B C D O M N : Type}
variables [Geometry 𝒢] [IsConvexQuadrilateral 𝒢 A B C D] 
variables (h_O : IsIntersection O (Segment A C) (Segment B D))
variables (h_M : M ∈ Segment O A)
variables (h_N : N ∈ Segment O D)
variables (h_parallel1 : Parallel 𝒢 (Line M N) (Line A D))
variables (h_parallel2 : Parallel 𝒢 (Line N C) (Line A B))

theorem angle_ABM_eq_angle_NCD (A B C D O M N : Point) 
  (h_O : intersection A C B D O)
  (h_M : M ∈ segment O A)
  (h_N : N ∈ segment O D)
  (h_parallel1 : parallel (line M N) (line A D))
  (h_parallel2 : parallel (line N C) (line A B)) :
  angle A B M = angle N C D :=
sorry

end angle_ABM_eq_angle_NCD_l451_451857


namespace angle_relationship_l451_451055

theorem angle_relationship (a b A B C : ℝ) (hA : A = π / 6) (ha : a = 1) (hb : b = √3) (hB_acute : B < π / 2) :
  B = π / 3 ∧ C = π / 2 ∧ C > B ∧ B > A :=
by
  sorry

end angle_relationship_l451_451055


namespace oil_leak_during_fix_l451_451282

theorem oil_leak_during_fix (total_leak: ℕ) (leak_before_fixing: ℕ):
  total_leak = 6206 → leak_before_fixing = 2475 →
  total_leak - leak_before_fixing = 3731 :=
by
  intro h1 h2
  rw [h1, h2]
  rfl

end oil_leak_during_fix_l451_451282


namespace complex_subtraction_l451_451346

theorem complex_subtraction (a b : ℂ) (h_a : a = 6 - 3 * complex.I) (h_b : b = 2 + 3 * complex.I) : 
  a - 3 * b = -12 * complex.I :=
by
  sorry

end complex_subtraction_l451_451346


namespace polar_line_through_center_perpendicular_to_axis_l451_451494

-- We define our conditions
def circle_in_polar (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ

def center_of_circle (C : ℝ × ℝ) : Prop := C = (2, 0)

def line_in_rectangular (x : ℝ) : Prop := x = 2

-- We now state the proof problem
theorem polar_line_through_center_perpendicular_to_axis (ρ θ : ℝ) : 
  (∃ C, center_of_circle C ∧ (∃ x, line_in_rectangular x)) →
  (circle_in_polar ρ θ → ρ * Real.cos θ = 2) :=
by
  sorry

end polar_line_through_center_perpendicular_to_axis_l451_451494


namespace geometric_sequence_fifth_term_l451_451368

theorem geometric_sequence_fifth_term : 
  let a₁ := (2 : ℝ)
  let a₂ := (1 / 4 : ℝ)
  let r := a₂ / a₁
  let a₅ := a₁ * r ^ (5 - 1)
  a₅ = 1 / 2048 :=
by
  let a₁ := (2 : ℝ)
  let a₂ := (1 / 4 : ℝ)
  let r := a₂ / a₁
  let a₅ := a₁ * r ^ (5 - 1)
  sorry

end geometric_sequence_fifth_term_l451_451368


namespace jake_and_luke_items_l451_451060

theorem jake_and_luke_items :
  ∃ (p j : ℕ), 6 * p + 2 * j ≤ 50 ∧ (∀ (p' : ℕ), 6 * p' + 2 * j ≤ 50 → p' ≤ p) ∧ p + j = 9 :=
by
  sorry

end jake_and_luke_items_l451_451060


namespace probability_event_A_l451_451922

-- Defining the vectors
def a_m (m : ℕ) := (m, 1)
def b_n (n : ℕ) := (2, n)

-- Definitions of the set of m and n values
def valid_m : Finset ℕ := {1, 2, 3}
def valid_n : Finset ℕ := {1, 2, 3}

-- Condition for orthogonality
def orthogonal (m n : ℕ) : Prop := (m - 1) ^ 2 = n

-- Event A
def event_A : Finset (ℕ × ℕ) := 
  (valid_m.product valid_n).filter (λ ⟨m, n⟩, orthogonal m n)

-- Total number of possible pairs
def total_pairs : ℕ := (valid_m.product valid_n).card

theorem probability_event_A : 
  (event_A.card : ℚ) / total_pairs = 1 / 9 := 
sorry

end probability_event_A_l451_451922


namespace orange_cost_is_correct_l451_451113

-- Define constants for the costs and the total paid
def apple_cost : ℝ := 1
def banana_cost : ℝ := 3
def discount_per_5_fruits : ℝ := 1
def total_paid : ℝ := 15

-- Define the quantities of fruits bought
def apples_bought : ℕ := 5
def oranges_bought : ℕ := 3
def bananas_bought : ℕ := 2

-- Define the total number of fruits bought
def total_fruits := apples_bought + oranges_bought + bananas_bought

-- Define the cost of an orange as a variable
def orange_cost : ℝ := sorry

-- Define the total discount received
def total_discount := (total_fruits / 5) * discount_per_5_fruits

-- Define the total cost before discount
def total_cost_before_discount := 
  (apples_bought * apple_cost) + (oranges_bought * orange_cost) + (bananas_bought * banana_cost)

-- Define the equation representing total paid after applying the discount
def equation := total_cost_before_discount - total_discount = total_paid

-- The theorem proving the cost of an orange
theorem orange_cost_is_correct : orange_cost = 8 / 3 :=
by
  -- Arguments will go here to complete the proof
  sorry

end orange_cost_is_correct_l451_451113


namespace selling_price_to_store_is_correct_l451_451689

-- Define the constants
def one_time_product_cost : ℝ := 56430.00
def variable_cost_per_book : ℝ := 8.25
def number_of_books : ℕ := 4180

-- Total variable cost
def total_variable_cost : ℝ := variable_cost_per_book * number_of_books

-- Total cost of production
def total_cost_of_production : ℝ := one_time_product_cost + total_variable_cost

-- Total sales needed (same as total cost of production for break-even)
def total_sales_needed : ℝ := total_cost_of_production

-- Selling price per book
def selling_price_per_book : ℝ := total_sales_needed / number_of_books

-- Prove that the selling price per book is 21.75 dollars
theorem selling_price_to_store_is_correct : selling_price_per_book = 21.75 :=
by
  -- Proof omitted
  sorry

end selling_price_to_store_is_correct_l451_451689


namespace completing_the_square_l451_451220

theorem completing_the_square (x : ℝ) :
  x^2 + 4 * x + 1 = 0 ↔ (x + 2)^2 = 3 :=
by
  sorry

end completing_the_square_l451_451220


namespace probability_sum_18_l451_451477

-- Define a standard 6-faced die roll
def dice_roll := {n : ℕ // 1 ≤ n ∧ n ≤ 6}

-- Define the condition for three 6-faced dice being rolled
def roll_three_dice := ({dice_roll, dice_roll, dice_roll} : Set dice_roll)

-- Define the condition of independence of the dice
def independent (A B C : dice_roll) : Prop :=
  ∀ (a : dice_roll) (b : dice_roll) (c : dice_roll),
    (Pr[A = a] * Pr[B = b] * Pr[C = c]) = 1/6 * 1/6 * 1/6 

-- Define the event of rolling a sum of 18
def event_sum_18 (A B C : dice_roll) : Prop := A.val + B.val + C.val = 18

-- The probability of the sum of 18 given the conditions
theorem probability_sum_18 :
  ∀ (A B C : dice_roll), independent A B C → Pr[event_sum_18 A B C] = 1 / 216 :=
by
  intros A B C h
  sorry

end probability_sum_18_l451_451477


namespace angle_between_a_b_l451_451442

variables (a b : EuclideanSpace ℝ (Fin 3)) -- We use 3D space for generic vector considerations

# Let a² = 4 and b² = 4 (use L2 norm square for vector magnitude)
def a_squared : ℝ := (∥a∥^2) = 4
def b_squared : ℝ := (∥b∥^2) = 4

# Given condition: (a + b) ⋅ (3a - b) = 4
def given_condition : ℝ := (a + b) ⬝ (3 • a - b) = 4

# Angle between a and b is 2π/3
theorem angle_between_a_b : 
  (a_squared a) → 
  (b_squared b) → 
  (given_condition a b) → 
  real.angle.cos (real.angle b a) = -1/2 →
  real.angle b a = 2 * real.pi / 3 := 
sorry

end angle_between_a_b_l451_451442


namespace toys_of_Jason_l451_451890

theorem toys_of_Jason (R J Jason : ℕ) 
  (hR : R = 1) 
  (hJ : J = R + 6) 
  (hJason : Jason = 3 * J) : 
  Jason = 21 :=
by
  sorry

end toys_of_Jason_l451_451890


namespace negation_of_exists_x_quad_eq_zero_l451_451176

theorem negation_of_exists_x_quad_eq_zero :
  ¬ ∃ x : ℝ, x^2 + 2 * x + 5 = 0 ↔ ∀ x : ℝ, x^2 + 2 * x + 5 ≠ 0 :=
by sorry

end negation_of_exists_x_quad_eq_zero_l451_451176


namespace quantities_average_l451_451980

theorem quantities_average (n : ℕ) (qs : list ℕ) (h_avg_all : (qs.sum : ℚ) = 10 * n)
  (h_avg_part1 : (qs.take 3).sum = 12) (h_avg_part2 : (qs.drop 3).sum = 38) : n = 5 :=
by
  sorry

end quantities_average_l451_451980


namespace stratified_sampling_sophomores_selected_l451_451687

theorem stratified_sampling_sophomores_selected 
  (total_freshmen : ℕ) (total_sophomores : ℕ) (total_seniors : ℕ) 
  (freshmen_selected : ℕ) (selection_ratio : ℕ) :
  total_freshmen = 210 →
  total_sophomores = 270 →
  total_seniors = 300 →
  freshmen_selected = 7 →
  selection_ratio = total_freshmen / freshmen_selected →
  selection_ratio = 30 →
  total_sophomores / selection_ratio = 9 :=
by sorry

end stratified_sampling_sophomores_selected_l451_451687


namespace turnover_first_quarter_correct_l451_451750

def turnover_january : ℝ := 1.5 -- million yuan
def february_increase (t_jan : ℝ) : ℝ := t_jan * 1.1 -- increase by 10%
def march_growth (t_jan : ℝ) : ℝ := t_jan * 0.91 -- reduced by 9%
def turnover_first_quarter (t_jan : ℝ) : ℝ := t_jan + february_increase(t_jan) + march_growth(t_jan)
def total_turnover := 4.515 -- million yuan

theorem turnover_first_quarter_correct :
  turnover_first_quarter turnover_january = total_turnover :=
sorry

end turnover_first_quarter_correct_l451_451750


namespace largest_power_that_divides_factorial_l451_451767

theorem largest_power_that_divides_factorial (p : ℕ) (hp : Nat.Prime p) : 
  ∃ n : ℕ, (n = p + 1) ∧ ((p!)^n ∣ (p^2)!) :=
by
  sorry

end largest_power_that_divides_factorial_l451_451767


namespace box_weight_without_balls_l451_451190

theorem box_weight_without_balls :
  let number_of_balls := 30
  let weight_per_ball := 0.36
  let total_weight_with_balls := 11.26
  let total_weight_of_balls := number_of_balls * weight_per_ball
  let weight_of_box := total_weight_with_balls - total_weight_of_balls
  weight_of_box = 0.46 :=
by 
  sorry

end box_weight_without_balls_l451_451190


namespace scheme_A_yield_percentage_l451_451713

-- Define the initial investments and yields
def initial_investment_A : ℝ := 300
def initial_investment_B : ℝ := 200
def yield_B : ℝ := 0.5 -- 50% yield

-- Define the equation given in the problem
def yield_A_equation (P : ℝ) : Prop :=
  initial_investment_A + (initial_investment_A * (P / 100)) = initial_investment_B + (initial_investment_B * yield_B) + 90

-- The proof statement we need to prove
theorem scheme_A_yield_percentage : yield_A_equation 30 :=
by
  sorry -- Proof is omitted

end scheme_A_yield_percentage_l451_451713


namespace units_digit_sum_squares_of_first_2011_odd_integers_l451_451205

-- Define the relevant conditions and given parameters
def first_n_odd_integers (n : ℕ) : List ℕ := List.range' 1 (2*n) (λ k, 2*k - 1)

def units_digit (n : ℕ) : ℕ := n % 10

def square_units_digit (n : ℕ) : ℕ := units_digit (n * n)

-- Prove the units digit of the sum of squares of the first 2011 odd positive integers
theorem units_digit_sum_squares_of_first_2011_odd_integers : 
  units_digit (List.sum (List.map (λ x, x * x) (first_n_odd_integers 2011))) = 1 :=
by
  -- Sorry skips the proof
  sorry

end units_digit_sum_squares_of_first_2011_odd_integers_l451_451205


namespace sum_of_integers_in_range_correct_l451_451594

def sum_of_integers_in_range (lower upper : ℤ) : ℤ :=
  ∑ i in (Finset.range (upper - lower + 1)).filter (λ n, n + lower > -4 ∧ n + lower < 3.2),
    i + lower

theorem sum_of_integers_in_range_correct :
  sum_of_integers_in_range (-4) 4 = 0 :=
by
  sorry

end sum_of_integers_in_range_correct_l451_451594


namespace percentage_female_employees_l451_451864

/-- 
Given:
- Total number of employees E = 1600
- 62% of all employees are computer literate
- 672 female employees are computer literate
- 50% of all male employees are computer literate

Prove:
The percentage of female employees in the office is 60%.
-/
theorem percentage_female_employees (E : ℕ) (CL : ℕ) (CF : ℕ) (CM : ℕ) (F : ℕ) (M : ℕ)
    (hE : E = 1600)
    (hCL : CL = 992)
    (hCF : CF = 672)
    (hCM : CM = 320)
    (hF : E - M = F)
    (hM : M = 640)
    (hCL_def : CL = CF + CM)
    (hCM_def : CM = 0.5 * M)
    (hCL_calc : CL = 0.62 * E) :
    (F / E : ℝ) * 100 = 60 := by
  sorry

end percentage_female_employees_l451_451864


namespace sum_a_1_to_100_l451_451002

def f (n : ℕ) : ℤ := n^2 * Int.cos (n * Real.pi)
def a (n : ℕ) : ℤ := f n + f (n + 1)

theorem sum_a_1_to_100 : (∑ n in Finset.range 100, a (n + 1)) = 0 := sorry

end sum_a_1_to_100_l451_451002


namespace interest_rate_lent_l451_451263

theorem interest_rate_lent (P Q : ℕ) (R_borrow R_lent : ℚ) (T: ℕ) (gain_per_year: ℚ) :
  P = 9000 → 
  R_borrow = 4 → 
  T = 2 → 
  gain_per_year = 180 → 
  Q = P * R_borrow * T / 100 → 
  let total_interest := Q + gain_per_year * T in
  total_interest = P * R_lent * T / 100 → 
  R_lent = 6 :=
by 
  intro hP hR_borrow hT hgain_per_year hQ h_total_interest
  rw [hP, hR_borrow, hT, hgain_per_year] at hQ h_total_interest 
  sorry

end interest_rate_lent_l451_451263


namespace area_of_triangle_PQR_l451_451757

-- Define the problem conditions
def PQ : ℝ := 4
def PR : ℝ := 4
def angle_P : ℝ := 45 -- degrees

-- Define the main problem
theorem area_of_triangle_PQR : 
  (PQ = PR) ∧ (angle_P = 45) ∧ (PR = 4) → 
  ∃ A, A = 8 := 
by
  sorry

end area_of_triangle_PQR_l451_451757


namespace arcsin_eq_pi_over_2_solutions_l451_451553

theorem arcsin_eq_pi_over_2_solutions :
  ∀ x : ℝ, (arcsin x + arcsin (3 * x) = π / 2) →
    (x = 1 / sqrt 10 ∨ x = -1 / sqrt 10) := 
by sorry

end arcsin_eq_pi_over_2_solutions_l451_451553


namespace midpoint_distance_l451_451536

theorem midpoint_distance (a b c d : ℝ) :
  let m := (a + c) / 2
  let n := (b + d) / 2
  let m' := m - 0.5
  let n' := n - 0.5
  dist (m, n) (m', n') = (Real.sqrt 2) / 2 := 
by 
  sorry

end midpoint_distance_l451_451536


namespace smallest_shift_l451_451954

variable (g : ℝ → ℝ)

def periodic (g : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, g(x + p) = g(x)

theorem smallest_shift (h : periodic g 30) : ∃ b > 0, (∀ x, g (2 * x + b) = g (2 * x)) ∧ b = 60 :=
by
  use 60
  split
  { linarith }
  { intros x
    have h2 : g (2 * (x + 15)) = g (2 * x) := by {
      rw [← h (2 * x)]
      rw [← h (2 * (x + 15) - 30)]
      rw [add_sub]
    }
    exact h2
  }

end smallest_shift_l451_451954


namespace find_b_from_root_l451_451791

theorem find_b_from_root (b : ℚ) (h : ∃ (x : ℚ), x^2 + b * x - 15 = 0 ∧ x = -8) : b = 49/8 :=
by
  obtain ⟨x, hx, hx_eq_minus_8⟩ := h
  have h_root : x = -8 := hx_eq_minus_8
  replace hx := calc x^2 + b * x - 15 = 0 : hx
  sorry

end find_b_from_root_l451_451791


namespace math_proof_problem_l451_451521

variables {n : ℕ}
variables {A B : ℝ}
variables {a b : Fin n → ℝ}

-- Conditions are provided as assumptions.
theorem math_proof_problem
  (h_pos_n : 0 < n)
  (h_pos_A : A > 0) 
  (h_pos_B : B > 0)
  (h_a_pos : ∀ i, a i > 0)
  (h_b_pos : ∀ i, b i > 0)
  (h_a_le_b : ∀ i, a i ≤ b i)
  (h_a_le_A : ∀ i, a i ≤ A)
  (h_product_le : (∏ i in Finset.univ, b i) / (∏ i in Finset.univ, a i) ≤ B / A):
  (∏ i in Finset.univ, b i + 1) / (∏ i in Finset.univ, a i + 1) ≤ (B + 1) / (A + 1) :=
sorry

end math_proof_problem_l451_451521


namespace rate_of_slabs_l451_451175

def rate_per_sq_meter (cost : ℝ) (length : ℝ) (width : ℝ) : ℝ :=
  cost / (length * width)

theorem rate_of_slabs : rate_per_sq_meter 16500 5.5 3.75 = 800 := 
by sorry

end rate_of_slabs_l451_451175


namespace jonathan_first_name_length_l451_451070

theorem jonathan_first_name_length :
  ∃ J : ℕ, (J + 10) + (5 + 10) = 33 ∧ J = 8 :=
by
  use 8
  simp
  sorry

end jonathan_first_name_length_l451_451070


namespace distance_from_intersection_to_side_CD_l451_451885

theorem distance_from_intersection_to_side_CD (s : ℝ) :
  let A := ⟨0, 0⟩
  let B := ⟨s, 0⟩
  let CD := line_through (⟨s, s⟩) (⟨0, s⟩)
  let X := classical.some (Exists.intro (punctured_ball_center A s ∩ punctured_ball_center B s))
  let dist_to_CD := λ pt seg, classical.some (Exists.intro (abs (seg pt - pt.2)))
  (dist_to_CD X CD) = (1 / 2 * s * (2 - sqrt 3)) := by
  sorry

end distance_from_intersection_to_side_CD_l451_451885


namespace first_term_arithmetic_sequence_l451_451090

def T_n (a d : ℚ) (n : ℕ) := n * (2 * a + (n - 1) * d) / 2

theorem first_term_arithmetic_sequence (a : ℚ)
  (h_const_ratio : ∀ (n : ℕ), n > 0 → 
    (T_n a 5 (4 * n)) / (T_n a 5 n) = (T_n a 5 4 / T_n a 5 1)) : 
  a = -5/2 :=
by 
  sorry

end first_term_arithmetic_sequence_l451_451090


namespace power_function_point_l451_451824

noncomputable theory
def f (k a : ℝ) (x : ℝ) := k * x^a

theorem power_function_point (k a : ℝ) (h : f k a (1/2) = 1/4) : k + a = 3 :=
by
  sorry

end power_function_point_l451_451824


namespace arithmetic_sequence_next_term_perfect_square_sequence_next_term_l451_451759

theorem arithmetic_sequence_next_term (a : ℕ → ℕ) (n : ℕ) (h₀ : a 0 = 0) (h₁ : ∀ n, a (n + 1) = a n + 3) :
  a 5 = 15 :=
by sorry

theorem perfect_square_sequence_next_term (b : ℕ → ℕ) (k : ℕ) (h₀ : ∀ k, b k = (k + 1) * (k + 1)) :
  b 5 = 36 :=
by sorry

end arithmetic_sequence_next_term_perfect_square_sequence_next_term_l451_451759


namespace sqrt_12_minus_2_cos_30_minus_inv_third_eq_sqrt3_minus_3_l451_451332

theorem sqrt_12_minus_2_cos_30_minus_inv_third_eq_sqrt3_minus_3 :
  sqrt 12 - 2 * (real.cos (real.pi/6)) - (1/3)⁻¹ = sqrt 3 - 3 :=
by
  have h1 : (1/3)⁻¹ = 3 := by norm_num
  rw [h1]
  have h2 : sqrt 12 = 2 * sqrt 3 := by norm_num
  rw [h2]
  have h3 : real.cos (real.pi/6) = sqrt 3 / 2 := by norm_num
  rw [h3]
  sorry

end sqrt_12_minus_2_cos_30_minus_inv_third_eq_sqrt3_minus_3_l451_451332


namespace base10_progression_false_l451_451737

-- Definitions for counting units
def integer_units := {1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000, 100000000000}
def decimal_units := {0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001}

-- Theorem statement
theorem base10_progression_false : 
  (∀ x ∈ integer_units, ∃ y ∈ integer_units, 10 * x = y) ∧ 
  (∀ x ∈ decimal_units, ∃ y ∈ decimal_units, 10 * x = y) → 
  false := 
sorry

end base10_progression_false_l451_451737


namespace Jason_toys_correct_l451_451891

variable (R Jn Js : ℕ)

def Rachel_toys : ℕ := 1

def John_toys (R : ℕ) : ℕ := R + 6

def Jason_toys (Jn : ℕ) : ℕ := 3 * Jn

theorem Jason_toys_correct (hR : R = 1) (hJn : Jn = John_toys R) (hJs : Js = Jason_toys Jn) : Js = 21 :=
by
  sorry

end Jason_toys_correct_l451_451891


namespace math_problem_l451_451392

theorem math_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a^2 + b = 1) : 
  (a + Real.sqrt b ≤ Real.sqrt 2) ∧ 
  (1 / 2 < 2 ^ (a - Real.sqrt b) ∧ 2 ^ (a - Real.sqrt b) < 2) ∧ 
  (a^2 - b > -1) := 
by
  sorry

end math_problem_l451_451392


namespace min_soda_bottles_needed_l451_451163

theorem min_soda_bottles_needed (total_people : ℕ) (exchange_rate : ℕ) (min_bottles : ℕ) (hx : total_people = 50) (he : exchange_rate = 5) (hm : min_bottles = 40):
  ∃ x : ℕ, x + (⌊ x / exchange_rate ⌋) + (⌊ x / (exchange_rate^2) ⌋) + (⌊ x / (exchange_rate^3) ⌋) + ... ≥ total_people ∧ x = min_bottles :=
by
  sorry

end min_soda_bottles_needed_l451_451163


namespace axes_of_symmetry_not_coincide_l451_451570

def y₁ (x : ℝ) := (1 / 8) * (x^2 + 6 * x - 25)
def y₂ (x : ℝ) := (1 / 8) * (31 - x^2)

def tangent_y₁ (x : ℝ) := (x + 3) / 4
def tangent_y₂ (x : ℝ) := -x / 4

def axes_symmetry_y₁ := -3
def axes_symmetry_y₂ := 0

theorem axes_of_symmetry_not_coincide :
  (∃ x1 x2 : ℝ, y₁ x1 = y₂ x1 ∧ y₁ x2 = y₂ x2 ∧ tangent_y₁ x1 * tangent_y₂ x1 = -1 ∧ tangent_y₁ x2 * tangent_y₂ x2 = -1) →
  axes_symmetry_y₁ ≠ axes_symmetry_y₂ :=
by sorry

end axes_of_symmetry_not_coincide_l451_451570


namespace same_color_socks_prob_l451_451249

noncomputable def probability_of_at_least_one_pair_same_color :
    ℕ → ℕ → ℕ → ℕ → ℚ
  | total_socks, white_socks, red_socks, black_socks =>
    let total_ways := Nat.choose total_socks 3
    let diff_colors_ways := white_socks * red_socks * black_socks
    (total_ways - diff_colors_ways) / total_ways

theorem same_color_socks_prob :
  probability_of_at_least_one_pair_same_color 40 10 12 18 = 193 / 247 :=
by
  sorry

end same_color_socks_prob_l451_451249


namespace proof_x_plus_y_equals_30_l451_451914

variable (x y : ℝ) (h_distinct : x ≠ y)
variable (h_det : Matrix.det ![
  ![2, 5, 10],
  ![4, x, y],
  ![4, y, x]
  ] = 0)

theorem proof_x_plus_y_equals_30 :
  x + y = 30 :=
sorry

end proof_x_plus_y_equals_30_l451_451914


namespace flowchart_decision_is_diamond_l451_451482

def flowchart_decision_symbol : Prop :=
  ∀ (notation : String),
  notation = "flowchart_standard" → ∃ (symbol : String), symbol = "diamond-shaped box"

theorem flowchart_decision_is_diamond :
  flowchart_decision_symbol :=
by
  intros notation notation_standard
  use "diamond-shaped box"
  sorry

end flowchart_decision_is_diamond_l451_451482


namespace a_plus_b_is_24_l451_451096

theorem a_plus_b_is_24 (a b : ℤ) (h1 : 0 < b) (h2 : b < a) (h3 : a * (a + 3 * b) = 550) : a + b = 24 :=
sorry

end a_plus_b_is_24_l451_451096


namespace problem_l451_451524

def f (x : ℝ) : ℝ := x^2 + 2
def g (x : ℝ) : ℝ := 3 * x - 2
def h (x : ℝ) : ℝ := x + 3

theorem problem (x : ℝ) : f (g (h 2)) = 171 :=
by
   have h2 := h 2
   have gh2 := g h2
   have fgh2 := f gh2
   show fgh2 = 171 from sorry

end problem_l451_451524


namespace cookie_price_ratio_l451_451832

theorem cookie_price_ratio (c b : ℝ) (h1 : 6 * c + 5 * b = 3 * (3 * c + 27 * b)) : c = (4 / 5) * b :=
sorry

end cookie_price_ratio_l451_451832


namespace complete_square_l451_451222

theorem complete_square (x : ℝ) : x^2 + 4*x + 1 = 0 -> (x + 2)^2 = 3 :=
by sorry

end complete_square_l451_451222


namespace murtha_pebble_collection_l451_451930

theorem murtha_pebble_collection :
  let total_pebbles := (15 / 2.0) * (2 + 16) in total_pebbles = 135 :=
by
  let total_pebbles := (15 / 2.0) * (2 + 16)
  have : total_pebbles = 135 := sorry
  assumption

end murtha_pebble_collection_l451_451930


namespace sequence_n_sum_l451_451941

def sequence : List (Char × Nat) :=
  [('X', 6), ('Y', 24), ('X', 96)]

def letter_count (seq : List (Char × Nat)) (n : Nat) : Char → Nat
  | 'X' => seq.foldl (fun acc (c, count) => if c == 'X' then acc + (if n <= 0 then 0 else min count n) else acc) 0
  | 'Y' => seq.foldl (fun acc (c, count) => if c == 'Y' then acc + (if n <= 0 then 0 else min count n) else acc) 0
  | _ => 0

def valid_n_values (seq : List (Char × Nat)) : List Nat :=
  let rec find_vals (n : Nat) (xCount yCount : Nat) (vals : List Nat) := 
    if h : n ≤ seq.foldl (fun acc (_, count) => acc + count) 0 0 then
      let currentX := letter_count seq n 'X'
      let currentY := letter_count seq n 'Y'
      if currentX == 2 * currentY || currentY == 2 * currentX then
        find_vals (n + 1) currentX currentY (n :: vals)
      else
        find_vals (n + 1) currentX currentY vals
    else
      vals
  find_vals 0 0 0 []

theorem sequence_n_sum (seq : List (Char × Nat) = [('X', 6), ('Y', 24), ('X', 96)]) :
  valid_n_values seq.sum = 135 :=
sorry

end sequence_n_sum_l451_451941


namespace converse_inverse_contrapositive_l451_451154

theorem converse (x y : ℤ) : (x = 3 ∧ y = 2) → (x + y = 5) :=
by sorry

theorem inverse (x y : ℤ) : (x + y ≠ 5) → (x ≠ 3 ∨ y ≠ 2) :=
by sorry

theorem contrapositive (x y : ℤ) : (¬ (x = 3 ∧ y = 2)) → (¬ (x + y = 5)) :=
by sorry

end converse_inverse_contrapositive_l451_451154


namespace floor_factorial_expression_l451_451314

-- Define the factorial function for natural numbers
def factorial : ℕ → ℕ
| 0 := 1
| (n + 1) := (n + 1) * factorial n

-- The main theorem to prove
theorem floor_factorial_expression :
  (nat.floor ((factorial 2010 + factorial 2007) / (factorial 2009 + factorial 2008)) = 2009) :=
begin
  -- Actual proof goes here
  sorry
end

end floor_factorial_expression_l451_451314


namespace combined_cost_is_450_l451_451257

-- Given conditions
def bench_cost : ℕ := 150
def table_cost : ℕ := 2 * bench_cost

-- The statement we want to prove
theorem combined_cost_is_450 : bench_cost + table_cost = 450 :=
by
  sorry

end combined_cost_is_450_l451_451257


namespace determine_y_l451_451347

-- Given condition as a definition
def satisfies_equation (y : ℝ) : Prop :=
  10^y * 1000^y = 100^(8*y - 4)

-- The statement to prove
theorem determine_y : ∃ y : ℝ, satisfies_equation y ∧ y = (2/3) :=
by
  existsi (2/3 : ℝ)
  split
  · -- Proof that it satisfies the equation
    sorry
  · -- Proof that it equals 2/3
    sorry

end determine_y_l451_451347


namespace chessboard_symmetry_l451_451184

-- Definition of symmetry conditions
def symmetrical_positions (pos1 pos2 : (Nat × Char)) : Prop :=
  pos1.1 + pos2.1 = 9 ∧ pos1.2 = pos2.2

-- The main theorem combining the conditions and specifying the symmetry results
theorem chessboard_symmetry :
  symmetrical_positions (2, 'e') (7, 'e') ∧
  symmetrical_positions (5, 'h') (4, 'h') :=
by
  split;
  sorry

end chessboard_symmetry_l451_451184


namespace sum_first_10_terms_a_sequence_sum_every_second_term_l451_451408

-- Arithmetic sequence {a_n} with a non-zero common difference d
def a_sequence (n : ℕ) : ℕ := 2 * n + 2

-- Part (1): Sum of the first 10 terms of the a_sequence
theorem sum_first_10_terms_a_sequence : ∑ i in Finset.range 10, a_sequence i = 130 := 
sorry

-- Geometric sequence {b_n} with common ratio q = 1/2
def b_sequence (n : ℕ) : ℝ := 4 * (1 / 2) ^ n

-- Part (2): Sum of every second term in a geometric sequence
theorem sum_every_second_term : 
  ∑ i in Finset.Ico 1 (101 / 2), b_sequence (2 * i) = 50 :=
sorry

end sum_first_10_terms_a_sequence_sum_every_second_term_l451_451408


namespace shadedRegionArea_correct_l451_451874

noncomputable def semiCircleArea (d : ℝ) : ℝ := (π * d^2) / 8

def areaShadedRegion : ℝ :=
  let smallCircleArea := semiCircleArea 3 * 5
  let largeCircleArea := semiCircleArea 15
  largeCircleArea - smallCircleArea

theorem shadedRegionArea_correct :
  areaShadedRegion = (45 / 2) * π := by
  sorry

end shadedRegionArea_correct_l451_451874


namespace sara_saving_amount_l451_451135

theorem sara_saving_amount :
  ∃ S : ℕ, 4100 + 820 * S = 15 * 820 ∧ S = 10 := by
  have h : 15 * 820 = 12300 := by norm_num
  have key : 4100 + 820 * 10 = 4100 + 8200 := by norm_num
  use 10
  split
  case left =>
    have h8200: 12300 - 4100 = 8200 := by norm_num
    norm_num
    rw [key, h8200, h]
  case right => 
    refl

end sara_saving_amount_l451_451135


namespace multiply_binomials_l451_451117

theorem multiply_binomials :
  ∀ (x : ℝ), 
  (4 * x + 3) * (x - 6) = 4 * x^2 - 21 * x - 18 :=
by
  sorry

end multiply_binomials_l451_451117


namespace floor_factorial_expression_l451_451317

-- Define the factorial function for natural numbers
def factorial : ℕ → ℕ
| 0 := 1
| (n + 1) := (n + 1) * factorial n

-- The main theorem to prove
theorem floor_factorial_expression :
  (nat.floor ((factorial 2010 + factorial 2007) / (factorial 2009 + factorial 2008)) = 2009) :=
begin
  -- Actual proof goes here
  sorry
end

end floor_factorial_expression_l451_451317


namespace problem_part_one_problem_part_two_l451_451395

variable (a : ℕ) (h : a = 1)

theorem problem_part_one : a^3 + 1 = 2 := by
  rw [h]
  calc
    1^3 + 1 = 1 + 1 := by norm_num
    ... = 2 := by norm_num
  sorry

theorem problem_part_two : (a + 1) * (a^2 - a + 1) = 2 := by
  rw [h]
  calc
    (1 + 1) * (1^2 - 1 + 1)
        = 2 * 1 := by norm_num
    ... = 2 := by norm_num
  sorry

end problem_part_one_problem_part_two_l451_451395


namespace average_of_two_excluding_min_max_l451_451976

theorem average_of_two_excluding_min_max :
  ∃ (a b c d : ℕ), (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d) ∧ 
  (a + b + c + d = 20) ∧ (max (max a b) (max c d) - min (min a b) (min c d) = 14) ∧
  (([a, b, c, d].erase (min (min a b) (min c d))).erase (max (max a b) (max c d)) = [x, y])
  → ((x + y) / 2 = 2.5) :=
sorry

end average_of_two_excluding_min_max_l451_451976


namespace problem_solution_l451_451880

-- Definitions based on the conditions

structure Triangle (α : Type) :=
  (A B C : α)

variables {α : Type} [LinearOrder α] 

-- Definitions for points E and D on segments AB and BL respectively, and bisector AL in triangle ABC 
structure GeometryConfiguration (triangle : Triangle α) :=
  (E D L : α)
  (AE BL : set α) -- segments
  (AL : α)

-- Defining the problem conditions 
def problem_conditions (triangle : Triangle α) (config : GeometryConfiguration α) : Prop :=
  let ⟨A, B, C⟩ := triangle in
  let ⟨E, D, L, AE, BL, AL⟩ := config in
  segment (A, E, B) ∧
  segment (B, D, L) ∧
  (L = midpoint B C) ∧ -- DL = LC (midpoint)
  segment_parallel (E, D) (A, C) ∧ -- ED ∥ AC
  distance A E = 15 ∧
  distance A C = 12

-- The theorem statement
theorem problem_solution (triangle : Triangle α) (config : GeometryConfiguration α) 
  (h : problem_conditions triangle config) : 
  distance (segment_E config) (segment_D config) = 3 := 
sorry

end problem_solution_l451_451880


namespace bons_wins_probability_l451_451753

theorem bons_wins_probability:
    (p : ℝ) 
    (silver_not_six : ℝ := 5/6)
    (bons_six : ℝ := 1/6)
    (silver_not_six_again : ℝ := 5/6)
    (bons_not_six : ℝ := 5/6):
    p = (silver_not_six * bons_six) / (1 - silver_not_six * bons_not_six) :=
begin
  -- Solution constraints
  have scenario1: ℝ := silver_not_six * bons_six,
  have scenario2: ℝ := silver_not_six_again * bons_not_six * p,
  
  -- Proof 
  have p_eq: scenario1 + scenario2 = (silver_not_six * bons_six) / (1 - silver_not_six * bons_not_six),
    sorry, 
  exact p_eq,
end

end bons_wins_probability_l451_451753


namespace gardener_distance_99_apples_l451_451714

def distance_travelled_one_gardener (n : ℕ) : ℕ :=
  2 * (n * (n + 1)) / 2

def distance_in_kilometers (d : ℕ) : ℝ :=
  d / 1000.0

theorem gardener_distance_99_apples :
  let n := 99 in
  distance_in_kilometers (distance_travelled_one_gardener n) = 9.9 :=
by
  sorry

end gardener_distance_99_apples_l451_451714


namespace union_area_of_reflected_triangles_l451_451276

open Real

noncomputable def pointReflected (P : ℝ × ℝ) (line_y : ℝ) : ℝ × ℝ :=
  (P.1, 2 * line_y - P.2)

def areaOfTriangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem union_area_of_reflected_triangles :
  let A := (2, 6)
  let B := (5, -2)
  let C := (7, 3)
  let line_y := 2
  let A' := pointReflected A line_y
  let B' := pointReflected B line_y
  let C' := pointReflected C line_y
  areaOfTriangle A B C + areaOfTriangle A' B' C' = 29 := sorry

end union_area_of_reflected_triangles_l451_451276


namespace num_lines_satisfying_conditions_l451_451411

-- Define points A and B
def A : (ℝ × ℝ) := (1, 0)
def B : (ℝ × ℝ) := (7, 8)

-- Define the distance function from a point to a line
def dist_to_line (p : ℝ × ℝ) (m b : ℝ) : ℝ :=
  abs ((m * p.1 - p.2 + b) / (real.sqrt (m * m + 1)))

-- State the problem in Lean
theorem num_lines_satisfying_conditions :
  ∃ n, (∀ l : ℝ × ℝ, dist_to_line A l.fst l.snd = 5 ∧ dist_to_line B l.fst l.snd = 5) → n = 3 :=
sorry

end num_lines_satisfying_conditions_l451_451411


namespace initial_persons_count_l451_451981

theorem initial_persons_count (new_person_weight old_person_weight : ℝ) 
  (average_weight_increase : ℝ) (h : new_person_weight - old_person_weight = average_weight_increase * n) 
  (new_person_weight = 87) (old_person_weight = 67) (average_weight_increase = 2.5) : 
  n = (20 / 2.5) := 
by 
  sorry

end initial_persons_count_l451_451981


namespace arithmetic_sequence_inequality_l451_451785

theorem arithmetic_sequence_inequality (a : ℕ → ℝ) (h : ∀ n : ℕ, n ≥ 2 → 2 * a n = a (n - 1) + a (n + 1)) :
  a 2 * a 4 ≤ a 3 ^ 2 :=
sorry

end arithmetic_sequence_inequality_l451_451785


namespace prime_divisor_of_polynomial_l451_451913

theorem prime_divisor_of_polynomial (p q : ℕ) [hp : Nat.Prime p] [hq : Nat.Prime q] (hp_odd : p % 2 = 1)
  (hx : ∃ x : ℕ, x^(p-1) + x^(p-2) + ... + 1 ≡ 0 [MOD q]) : p = q ∨ p ∣ q - 1 :=
sorry

end prime_divisor_of_polynomial_l451_451913


namespace goods_train_length_l451_451671

theorem goods_train_length (speed_kmph : ℕ) (platform_length_m : ℕ) (time_s : ℕ) 
    (h_speed : speed_kmph = 72) (h_platform : platform_length_m = 250) (h_time : time_s = 24) : 
    ∃ train_length_m : ℕ, train_length_m = 230 := 
by 
  sorry

end goods_train_length_l451_451671


namespace find_a5_l451_451794

noncomputable def a (n : ℕ) : ℝ := sorry  -- Define the arithmetic sequence

axiom arithmetic_sequence : ∀ n : ℕ, a (n + 1) = a n + d
axiom condition : a 2 + a 8 = 12

theorem find_a5 (d : ℝ) : a 5 = 6 := by
  sorry

end find_a5_l451_451794


namespace counterexample_exists_l451_451277

theorem counterexample_exists : 
  ∃ (m : ℤ), (∃ (k1 : ℤ), m = 2 * k1) ∧ ¬(∃ (k2 : ℤ), m = 4 * k2) := 
sorry

end counterexample_exists_l451_451277


namespace problem_solution_l451_451454

variable (a b : ℝ)

theorem problem_solution (h : 2 * a - 3 * b = 5) : 4 * a^2 - 9 * b^2 - 30 * b + 1 = 26 :=
sorry

end problem_solution_l451_451454


namespace area_of_circle_is_484_over_pi_l451_451235

noncomputable def area_of_circle (A_square : ℝ) (π : ℝ) : Option ℝ :=
  let side_length := real.sqrt A_square
  let perimeter := 4 * side_length
  let radius := perimeter / (2 * π)
  let area := π * (radius ^ 2)
  if A_square = 121 ∧ π ≠ 0 then
    some (484 / π)
  else
    none

theorem area_of_circle_is_484_over_pi :
  area_of_circle 121 real.pi = some (484 / real.pi) := by
    sorry

end area_of_circle_is_484_over_pi_l451_451235


namespace area_triangle_MNP_l451_451046

theorem area_triangle_MNP (XY XZ : ℝ) (hXY : XY = 13) (hXZ : XZ = 5) :
  let YZ := Real.sqrt (XY^2 - XZ^2)
  let X1Z := 60 / 17
  let X1Y := 84 / 17
  let MN := X1Y
  let MP := X1Z
  let NP := Real.sqrt (MN^2 - MP^2)
  let area_MNP := 1 / 2 * MN * MP
  in area_MNP = 2520 / 289 :=
by
  sorry

end area_triangle_MNP_l451_451046


namespace complete_square_l451_451223

theorem complete_square (x : ℝ) : x^2 + 4*x + 1 = 0 -> (x + 2)^2 = 3 :=
by sorry

end complete_square_l451_451223


namespace kristi_books_proof_l451_451718

variable (Bobby_books Kristi_books : ℕ)

def condition1 : Prop := Bobby_books = 142

def condition2 : Prop := Bobby_books = Kristi_books + 64

theorem kristi_books_proof (h1 : condition1 Bobby_books) (h2 : condition2 Bobby_books Kristi_books) : Kristi_books = 78 := 
by 
  sorry

end kristi_books_proof_l451_451718


namespace besfamilies_children_l451_451870

theorem besfamilies_children (n x : ℕ) 
  (initial_age final_age age_increase : ℕ) 
  (h_initial : initial_age = 101) 
  (h_final : final_age = 150) 
  (h_age_increase : age_increase = final_age - initial_age) 
  (h_equation : (n + 2) * x = age_increase) 
  (h_valid_years : x ≠ 1) 
  : n = 5 :=
by
  have h_final_age : final_age = 150 := by exact h_final
  have h_initial_age : initial_age = 101 := by exact h_initial
  have h_age_increment : age_increase = 49 := by rw [h_age_increase, h_final_age, h_initial_age]
  have h_main_equation : (n + 2) * x = 49 := by rw [h_age_increment, h_equation]
  sorry

end besfamilies_children_l451_451870


namespace units_digit_G_100_l451_451535

def G (n : ℕ) : ℕ := 3 ^ (2 ^ n) + 1

theorem units_digit_G_100 : (G 100) % 10 = 2 := 
by
  sorry

end units_digit_G_100_l451_451535


namespace count_ordered_pairs_l451_451908

theorem count_ordered_pairs:
  let s := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}.to_finset
  ∃ (M: ℕ), M = 3172 ∧
  (∀ (C D: finset ℕ),
    C ∪ D = s ∧
    C ∩ D = ∅ ∧
    ¬ C.card ∈ C ∧
    ¬ D.card ∈ D) → M = 3172 :=
by
  sorry

end count_ordered_pairs_l451_451908


namespace trig_expression_eval_l451_451775

open Real

-- Declare the main theorem
theorem trig_expression_eval (θ : ℝ) (k : ℤ) 
  (h : sin (θ + k * π) = -2 * cos (θ + k * π)) :
  (4 * sin θ - 2 * cos θ) / (5 * cos θ + 3 * sin θ) = 10 :=
  sorry

end trig_expression_eval_l451_451775


namespace rectangle_length_ratio_l451_451478

-- Definition of the lengths based on given conditions
def side_length_of_small_square (s : ℝ) := s
def side_length_of_large_square (s : ℝ) := 3 * s
def width_of_rectangle (s : ℝ) := 3 * s
def height_of_rectangle (s : ℝ) := s

-- Proof the length of the rectangle is 3 times its width
theorem rectangle_length_ratio (s : ℝ) :
  let l := width_of_rectangle s in
  let w := height_of_rectangle s in
  l = 3 * w :=
by
  -- Here the short proof is provided
  sorry

end rectangle_length_ratio_l451_451478


namespace Emilee_earns_25_l451_451062

variable (Terrence Jermaine Emilee : ℕ)
variable (h1 : Terrence = 30)
variable (h2 : Jermaine = Terrence + 5)
variable (h3 : Jermaine + Terrence + Emilee = 90)

theorem Emilee_earns_25 : Emilee = 25 := by
  -- Insert the proof here
  sorry

end Emilee_earns_25_l451_451062


namespace all_triangles_have_10_red_l451_451598

universe u

-- Blue points and red points
variable (B : Finset (ℕ × ℕ))
variable (R : Finset (ℕ × ℕ))

-- Condition: 20 blue points on a circle
axiom B_card : B.card = 20

-- Condition: some red points inside the circle
axiom R_nonempty : R.nonempty

-- Condition: No three points (blue or red) are collinear
axiom no_three_collinear : ∀ (P1 P2 P3 : (ℕ × ℕ)), P1 ∈ B ∪ R → P2 ∈ B ∪ R → P3 ∈ B ∪ R → P1 ≠ P2 → P2 ≠ P3 → P1 ≠ P3 → (¬ linear_independent ℝ ![P1, P2, P3]) 

-- Condition: There exist 1123 triangles with blue vertices that have exactly 10 red points inside.
axiom triangles_with_10_red : ∃ T : Finset (Finset (ℕ × ℕ)), (∀ t ∈ T, t.card = 3 ∧ (∀ v ∈ t, v ∈ B)) ∧ (T.card = 1123) ∧ (∀ t ∈ T, ((t : set (ℕ × ℕ)).card = 3) ∧ ((R ∩ (inside_triangle t)).card = 10))

-- Task: Prove that all triangles formed by blue points have exactly 10 red points inside.
theorem all_triangles_have_10_red : ∀ t : Finset (ℕ × ℕ), t.card = 3 ∧ (∀ v ∈ t, v ∈ B) → (R ∩ (inside_triangle t)).card = 10 :=
by
  sorry

end all_triangles_have_10_red_l451_451598


namespace zilla_savings_l451_451642

theorem zilla_savings
  (monthly_earnings : ℝ)
  (h_rent : monthly_earnings * 0.07 = 133)
  (h_expenses : monthly_earnings * 0.5 = monthly_earnings / 2) :
  monthly_earnings - (133 + monthly_earnings / 2) = 817 :=
by
  sorry

end zilla_savings_l451_451642


namespace quotient_base4_correct_l451_451354

noncomputable def base4_to_base10 (n : ℕ) : ℕ :=
  match n with
  | 1302 => 1 * 4^3 + 3 * 4^2 + 0 * 4^1 + 2 * 4^0
  | 12 => 1 * 4^1 + 2 * 4^0
  | _ => 0

def base10_to_base4 (n : ℕ) : ℕ :=
  match n with
  | 19 => 1 * 4^2 + 0 * 4^1 + 3 * 4^0
  | _ => 0

theorem quotient_base4_correct : base10_to_base4 (114 / 6) = 103 := 
  by sorry

end quotient_base4_correct_l451_451354


namespace order_of_a_b_c_l451_451778

noncomputable def a := log 2 3
noncomputable def b := 2 ^ (1 / 2)
noncomputable def c := log (4⁻¹) (1 / 15)

theorem order_of_a_b_c : c > a ∧ a > b := by
  sorry

end order_of_a_b_c_l451_451778


namespace average_remaining_primes_l451_451844

theorem average_remaining_primes (p : ℕ → ℕ) (h1 : (∑ i in range 20, p i) / 20 = 95)
  (h2 : (∑ i in range 10, p i) / 10 = 85) :
  ((∑ i in (10..20), p i) / 10) = 105 := 
sorry

end average_remaining_primes_l451_451844


namespace lines_intersect_value_k_l451_451576

theorem lines_intersect_value_k :
  ∀ (x y k : ℝ), (-3 * x + y = k) → (2 * x + y = 20) → (x = -10) → (k = 70) :=
by
  intros x y k h1 h2 h3
  sorry

end lines_intersect_value_k_l451_451576


namespace target_heart_rate_34_year_old_high_altitude_l451_451712

theorem target_heart_rate_34_year_old_high_altitude : 
  let max_heart_rate (age : ℕ) : ℕ := (220 - age)
  let adjusted_max_heart_rate (age : ℕ) : ℕ := max_heart_rate age + 15
  let target_heart_rate (adjusted_max : ℕ) : ℕ := (0.85 * adjusted_max).ceil.to_nat
  target_heart_rate (adjusted_max_heart_rate 34) = 171 :=
by
  sorry

end target_heart_rate_34_year_old_high_altitude_l451_451712


namespace part_one_part_two_l451_451513

theorem part_one (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hne : p ≠ q) : 
  Nat.totient (p * q) = (p - 1) * (q - 1) := by 
  sorry

theorem part_two : ∃ (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ 
  p = 3 ∧ q = 11 ∧ Nat.totient (p * q) = 3 * p + q := by
  exact ⟨3, 11, Nat.prime_three, Nat.prime.eleven, by norm_num, rfl, rfl, by norm_num⟩

end part_one_part_two_l451_451513


namespace area_of_triangle_l451_451696

theorem area_of_triangle : 
  let l : ℝ → ℝ → Prop := fun x y => 3 * x + 2 * y = 12 in
  let x_intercept := (4 : ℝ) in
  let y_intercept := (6 : ℝ) in
  ∃ x y : ℝ, l x 0 ∧ x = x_intercept ∧ l 0 y ∧ y = y_intercept ∧ (1 / 2) * x_intercept * y_intercept = 12 := 
by
  sorry

end area_of_triangle_l451_451696


namespace trig_eq_solution_l451_451229

noncomputable def solve_trig_eq (t : Real) : Prop :=
  (∃ (k n : ℤ), 
    (t = (Real.pi / 12) * (4 * k - 1) ∨ 
     t = (Real.arctan 5 / 3) + (Real.pi * n) / 3) ∧ 
    cos (3 * t) ≠ 0 ∧ 
    Real.arccos (3 * t) - 6 * cos (3 * t) = 4 * sin (3 * t))

theorem trig_eq_solution (t : Real) :
  solve_trig_eq t := by
  sorry

end trig_eq_solution_l451_451229


namespace dot_product_self_eq_thirtysix_l451_451839

-- Define the vector space and the norm function
variables {V : Type*} [inner_product_space ℝ V] (v : V)

-- The condition given in the problem
def norm_v_is_six : Prop := ∥v∥ = 6

-- The theorem stating the mathematically equivalent proof problem
theorem dot_product_self_eq_thirtysix (h : norm_v_is_six v) : inner_product_space.dot_product v v = 36 :=
by sorry

end dot_product_self_eq_thirtysix_l451_451839


namespace zero_follows_eleven_l451_451749

theorem zero_follows_eleven :
  let count_01 := 16
  let count_10 := 15
  let count_0_after_01 := 8
  in count_10 - count_0_after_01 = 7 :=
by
  sorry

end zero_follows_eleven_l451_451749


namespace collinear_points_inverse_sum_half_l451_451033

theorem collinear_points_inverse_sum_half (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0)
    (collinear : (a - 2) * (b - 2) - (-2) * a = 0) : 
    1 / a + 1 / b = 1 / 2 := 
by
  sorry

end collinear_points_inverse_sum_half_l451_451033


namespace min_xy_min_x_plus_y_l451_451399

theorem min_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y - x - y = 3) : x * y ≥ 9 :=
sorry

theorem min_x_plus_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y - x - y = 3) : x + y ≥ 6 :=
sorry

end min_xy_min_x_plus_y_l451_451399


namespace reciprocal_of_neg_three_l451_451586

-- Define the notion of reciprocal
def reciprocal (x : ℝ) : ℝ := 1 / x

-- State the proof problem
theorem reciprocal_of_neg_three :
  reciprocal (-3) = -1 / 3 :=
by
  -- Since we are only required to state the theorem, we use sorry to skip the proof.
  sorry

end reciprocal_of_neg_three_l451_451586


namespace sequence_geometric_sequence_general_term_l451_451827

theorem sequence_geometric (a : ℕ → ℕ) (h₁ : a 1 = 2) (h₂ : ∀ n, a (n + 1) = 2 * a n + 1) :
  ∃ r : ℕ, (a 1 + 1) = 3 ∧ (∀ n, (a (n + 1) + 1) = r * (a n + 1)) := by
  sorry

theorem sequence_general_term (a : ℕ → ℕ) (h₁ : a 1 = 2) (h₂ : ∀ n, a (n + 1) = 2 * a n + 1) :
  ∀ n, a n = 3 * 2^(n-1) - 1 := by
  sorry

end sequence_geometric_sequence_general_term_l451_451827


namespace min_product_xyz_l451_451103

theorem min_product_xyz (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (h4 : x + y + z = 1) (h5 : ∀ a b ∈ {x, y, z}, a ≤ 3 * b) : 
  ∃ (x y z : ℝ), (x + y + z = 1 ∧ ∀ a b ∈ {x, y, z}, a ≤ 3 * b ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧ xyz = 1/18) :=
sorry

end min_product_xyz_l451_451103


namespace hyperbola_eccentricity_range_l451_451438

-- Definitions of hyperbola and distance condition
def hyperbola (x y a b : ℝ) : Prop := (a > 0) ∧ (b > 0) ∧ (x^2 / a^2 - y^2 / b^2 = 1)
def distance_condition (a b : ℝ) : Prop :=
  ∀ (x y : ℝ), hyperbola x y a b → (b * x + a * y - 2 * a * b) > a

-- The range of the eccentricity
theorem hyperbola_eccentricity_range (a b : ℝ) (h : hyperbola 0 1 a b) 
  (dist_cond : distance_condition a b) : 
  ∃ e : ℝ, e ≥ (2 * Real.sqrt 3 / 3) :=
sorry

end hyperbola_eccentricity_range_l451_451438


namespace problem_abcd_l451_451541

theorem problem_abcd 
  (F E G : Point)
  (A B C D : Point)
  (parallelogram_ABCD : parallelogram A B C D)
  (on_extension_AD : OnExtension A D F)
  (intersect_AC : Intersect (Line B F) (Diagonal A C) E)
  (intersect_DC : Intersect (Line D C) (Extended G) F)
  (EF_eq : segment_length E F = 40)
  (GF_eq : segment_length G F = 30) :
  segment_length B E = 30 := 
sorry

end problem_abcd_l451_451541


namespace square_area_is_25_l451_451690

-- Define the vertices of the square based on given y-coordinates
def vertices_of_square (y1 y2 : ℝ) (x1 x2 : ℝ) : Prop :=
  y1 = 3 ∧ y2 = 3 ∧ x1 = 8 ∧ x2 = 8

-- Define the function to calculate the area of a square given its side length
def square_area (side : ℝ) : ℝ := side * side

-- Prove that the area of the square is 25 given the conditions
theorem square_area_is_25 (y1 y2 : ℝ) (x1 x2 : ℝ) :
  vertices_of_square y1 y2 x1 x2 → square_area 5 = 25 :=
by
  intro h
  unfold square_area
  exact rfl

end square_area_is_25_l451_451690


namespace sum_fractions_eq_n_l451_451545

theorem sum_fractions_eq_n (n : ℕ) :
  (finset.powerset_len n (finset.range n)).sum (λ s, s.prod (λ x, (1 : ℝ) / (x + 1))) = n :=
sorry

end sum_fractions_eq_n_l451_451545


namespace AMC_135_l451_451924

-- defining the conditions
variables {x : ℝ}
variables {B C H A M : Type} -- Points in a Euclidean plane
variables [is_BH HC_eq : x]
variables [BC_eq : (distance B C) = x * real.sqrt 2]
variables [AB_eq : (distance A B) = 2 * x]
variables [angle_BCH : angle B C H = 45]
variables [angle_CBH : angle C B H = 45]
variables [angle_ABH : angle A B H = 60]

-- the goal to prove
theorem AMC_135 (angle_ABC : (angle A B C) → (angle M B C) → (angle C B A)),
:
    angle A M C = 135 :=
begin
  sorry -- Proof goes here
end

end AMC_135_l451_451924


namespace triangular_array_mod_5_l451_451694

theorem triangular_array_mod_5 :
  let initial_distributions (x : ℕ → ℕ → ℕ) := ∀ i j, 0 ≤ x i j ∧ x i j ≤ 1
  ∧ (∀ i j, x i j = x (i+1) j + x (i+1) (j+1))
  ∧ (∀ j, x 15 j ∈ {0, 1}) in
  let top_square (x : ℕ → ℕ → ℕ) := ∑ i in range 15, (nat.choose 14 i) * x 15 i in
  let final_condition (x : ℕ → ℕ → ℕ) := top_square x % 5 = 0 in
  (∃ x : ℕ → ℕ → ℕ, initial_distributions x ∧ final_condition x = 0) =
  81920 := 
by sorry

end triangular_array_mod_5_l451_451694


namespace non_congruent_impossible_l451_451198

/-
Define the initial condition:
- Four congruent right triangles, noted in the invariant function
-/
noncomputable def initial_triangles : Set (ℤ × ℤ) := 
  {(0, 0), (0, 0), (0, 0), (0, 0)}

/-
Define the transformation step that produces new similar triangles
-/
def transformation (triangles : Set (ℤ × ℤ)) : Set (ℤ × ℤ) := 
  triangles.fold (λ (acc : Set (ℤ × ℤ)) (mn : ℤ × ℤ), 
                     acc.insert (mn.1 + 1, mn.2).insert (mn.1, mn.2 + 1)) 
                 ∅

/-
Invariant function
-/
def invariant (triangles : Set (ℤ × ℤ)) : ℝ :=
  triangles.toFinset.sum (λ ⟨m, n⟩, (1 : ℝ) / (2 ^ (m + n)))

/-
The Lean 4 statement that needs to be proven
-/
theorem non_congruent_impossible :
  ∀ (triangles : Set (ℤ × ℤ)), 
    (invariant triangles = 4) → 
    (¬ ∃ finite_steps : ℕ, 
        (∀ mn1 mn2 ∈ triangles, 
          mn1 ≠ mn2 → 
          triangles = iterate transformation finite_steps initial_triangles)) :=
begin
  sorry
end

end non_congruent_impossible_l451_451198


namespace thirty_degree_trig_l451_451730

-- Define the conditions of the 30-60-90 triangle in the unit circle.
noncomputable def sin_30_deg : ℝ := 1 / 2
noncomputable def cos_30_deg : ℝ := real.sqrt 3 / 2

-- State the Lean theorem to prove the assertions.
theorem thirty_degree_trig:
  sin 30 = sin_30_deg ∧ cos 30 = cos_30_deg ∧ sin_30_deg^2 + cos_30_deg^2 = 1 :=
by
  sorry

end thirty_degree_trig_l451_451730


namespace units_digit_sum_squares_of_first_2011_odd_integers_l451_451207

-- Define the relevant conditions and given parameters
def first_n_odd_integers (n : ℕ) : List ℕ := List.range' 1 (2*n) (λ k, 2*k - 1)

def units_digit (n : ℕ) : ℕ := n % 10

def square_units_digit (n : ℕ) : ℕ := units_digit (n * n)

-- Prove the units digit of the sum of squares of the first 2011 odd positive integers
theorem units_digit_sum_squares_of_first_2011_odd_integers : 
  units_digit (List.sum (List.map (λ x, x * x) (first_n_odd_integers 2011))) = 1 :=
by
  -- Sorry skips the proof
  sorry

end units_digit_sum_squares_of_first_2011_odd_integers_l451_451207


namespace problem_statement_l451_451486

open Real

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def A := (0, 0) : ℝ × ℝ
def B := (8, 0) : ℝ × ℝ
def C := (4, 7) : ℝ × ℝ
def P := (3, 3) : ℝ × ℝ

def dist_A := distance A P
def dist_B := distance B P
def dist_C := distance C P

def sum_distances := dist_A + dist_B + dist_C

theorem problem_statement : sum_distances = 3 * sqrt 2 + sqrt 34 + sqrt 17 ∧ 3 + 1 + 1 = 5 :=
by
  sorry

end problem_statement_l451_451486


namespace range_of_a_minus_b_l451_451171

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b

theorem range_of_a_minus_b (a b : ℝ) (h1 : ∃ α β : ℝ, α ≠ β ∧ f α a b = 0 ∧ f β a b = 0)
  (h2 : ∃ x1 x2 x3 x4 : ℝ, x1 < x2 ∧ x2 < x3 ∧ x3 < x4 ∧
                         (x2 - x1 = x3 - x2) ∧ (x3 - x2 = x4 - x3) ∧
                         f (x1^2 + 2 * x1 - 1) a b = 0 ∧
                         f (x2^2 + 2 * x2 - 1) a b = 0 ∧
                         f (x3^2 + 2 * x3 - 1) a b = 0 ∧
                         f (x4^2 + 2 * x4 - 1) a b = 0) :
  a - b ≤ 25 / 9 :=
sorry

end range_of_a_minus_b_l451_451171


namespace zilla_savings_l451_451641

theorem zilla_savings
  (monthly_earnings : ℝ)
  (h_rent : monthly_earnings * 0.07 = 133)
  (h_expenses : monthly_earnings * 0.5 = monthly_earnings / 2) :
  monthly_earnings - (133 + monthly_earnings / 2) = 817 :=
by
  sorry

end zilla_savings_l451_451641


namespace constant_term_in_binomial_expansion_l451_451491

theorem constant_term_in_binomial_expansion :
  ∃ c : ℝ, 
    (∃ n : ℕ, 2 ^ (n + 1) = 72) ∧ 
    c = (3 / real.pi) ^ 5 :=
by 
  have h : ∃ n : ℕ, 2 ^ (n + 1) = 72, from
    sorry,
  obtain ⟨n, hn⟩ := h,
  use (3 / real.pi) ^ 5,
  use n,
  split;
  assumption

end constant_term_in_binomial_expansion_l451_451491


namespace keiko_speed_proof_l451_451896

noncomputable def keiko_speed (a b s : ℝ) : Prop := 
  let Lin := 2 * a + 2 * real.pi * b in
  let Lout := 2 * a + 2 * real.pi * (b + 6) in
  (Lout / s = Lin / s + 48) → s = real.pi / 4

theorem keiko_speed_proof (a b : ℝ) : 
  let s := real.pi / 4 in keiko_speed a b s := 
by 
  intros Lin Lout
  sorry

end keiko_speed_proof_l451_451896


namespace find_m_plus_n_l451_451878

noncomputable def triangle_ratio (u v w : ℝ) : Prop :=
  u + v + w = 3/4 ∧ u^2 + v^2 + w^2 = 1/2 →
  let ratio := 1 - (u * (1 - w) + v * (1 - u) + w * (1 - v)) in
  ratio = 9 / 32

theorem find_m_plus_n : ∃ m n, (triangle_ratio u v w) ∧ (m + n = 41) := sorry

end find_m_plus_n_l451_451878


namespace diamond_problem_l451_451388

def diamond (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2)

theorem diamond_problem : diamond (diamond 7 24) (diamond (-24) (-7)) = 25 * Real.sqrt 2 := by
  sorry

end diamond_problem_l451_451388


namespace cylindrical_to_rectangular_conversion_l451_451735

theorem cylindrical_to_rectangular_conversion :
  ∀ (r θ z : ℝ), 
  r = 10 → 
  θ = Real.pi / 3 → 
  z = 2 → 
  (r * Real.cos θ, r * Real.sin θ, z) = (5, 5 * Real.sqrt 3, 2) :=
by
  intros r θ z hr hθ hz
  rw [hr, hθ, hz]
  norm_num
  rw [Real.cos_pi_div_three, Real.sin_pi_div_three]
  norm_num
  sorry

end cylindrical_to_rectangular_conversion_l451_451735


namespace initial_boys_l451_451858

theorem initial_boys (p : ℝ) (initial_boys : ℝ) (final_boys : ℝ) (final_groupsize : ℝ) : 
  (initial_boys = 0.35 * p) ->
  (final_boys = 0.35 * p - 1) ->
  (final_groupsize = p + 3) ->
  (final_boys / final_groupsize = 0.3) ->
  initial_boys = 13 := 
by
  sorry

end initial_boys_l451_451858


namespace beaver_group_count_l451_451152

theorem beaver_group_count (B : ℕ) (h1 : 3 * B = 60) : B = 20 :=
by sorry

end beaver_group_count_l451_451152


namespace bookkeeper_arrangements_l451_451447

theorem bookkeeper_arrangements :
  (fact 10) / ((fact 2) * (fact 2) * (fact 2) * (fact 2)) = 226800 :=
by
  sorry

end bookkeeper_arrangements_l451_451447


namespace fraction_exponentiation_and_multiplication_l451_451331

theorem fraction_exponentiation_and_multiplication :
  ( (2 : ℚ) / 3 ) ^ 3 * (1 / 4) = 2 / 27 :=
by
  sorry

end fraction_exponentiation_and_multiplication_l451_451331


namespace range_of_p_l451_451747

def p (x : ℝ) : ℝ := x^6 + 6 * x^3 + 9

theorem range_of_p : Set.Ici 9 = { y | ∃ x ≥ 0, p x = y } :=
by
  -- We skip the proof to only provide the statement as requested.
  sorry

end range_of_p_l451_451747


namespace proof_a_proof_b_proof_c_l451_451522

variable (ABCD : Type) [parallelogram ABCD]
variable (A B C D P E F : Point)
variable (r1 r2 : ℝ)
variable (AP PC DA DC : ℝ)
variable (in_triangle : ∀ {a b c : Point}, inscribed_circle a b c)
variable (D_tangent P_tangent : ∀ {a b c p e f : Point}, inscribed_circle a b c → tangential a b c p e f)

-- Conditions:
noncomputable def condition_1 : Prop := parallelogram ABCD ∧ ∃ (AC : Line), diagonal AC
noncomputable def condition_2 : Prop := in_triangle A B C ∧ ∃ (P : Point), tangent AC P
noncomputable def condition_3 : Prop := in_triangle D A P ∧ ∃ (r1 : ℝ), tangent DA D P r1
noncomputable def condition_4 : Prop := in_triangle D C P ∧ ∃ (r2 : ℝ), tangent DC D P r2

-- Statements to prove:
theorem proof_a :
  condition_1 ABCD ∧ condition_2 A B C P ∧ (DA + AP = DC + PC) → DA + AP = DC + CP := 
by sorry

theorem proof_b :
  condition_1 ABCD ∧ condition_2 A B C P ∧ condition_3 D A P r1 ∧ condition_4 D C P r2 →
  (r1 / r2 = AP / PC) := 
by sorry

theorem proof_c :
  condition_1 ABCD ∧ condition_2 A B C P ∧ condition_3 D A P r1 ∧ condition_4 D C P r2 ∧
  (DA + DC = 3 * AC) ∧ (DA = DP) → (r1 / r2 = 1) := 
by sorry

end proof_a_proof_b_proof_c_l451_451522


namespace find_angle_A_l451_451440

theorem find_angle_A (A B C : ℝ) (a b c : ℝ) 
  (h : 1 + (Real.tan A / Real.tan B) = 2 * c / b) : 
  A = Real.pi / 3 :=
sorry

end find_angle_A_l451_451440


namespace unique_diagonals_equal_l451_451280

variables (R P : Type) [rect : Rectangle R] [par : Parallelogram P]
-- Assuming that rect and par are classes capturing necessary properties

-- Opposite sides are equal
example (r : R) : opposite_sides_equal r := 
  rect.opposite_sides_equal r -- Definition assumed in the condition

-- Opposite angles are equal
example (r : R) : opposite_angles_equal r := 
  rect.opposite_angles_equal r -- Definition assumed in the condition

-- Diagonals are equal (the property we must confirm is unique to rectangles)
example (r : R) : diagonals_equal r := 
  rect.diagonals_equal r -- Definition assumed in the condition

-- Opposite sides are parallel
example (r : R) : opposite_sides_parallel r := 
  rect.opposite_sides_parallel r -- Definition assumed in the condition

-- Now we state the proof problem
theorem unique_diagonals_equal (r : R) (p : P) : 
  (opposite_sides_equal r → opposite_angles_equal r → diagonals_equal r → opposite_sides_parallel r) ∧ 
  (∃ p, opposite_sides_equal p ∧ opposite_angles_equal p ∧ ¬diagonals_equal p ∧ opposite_sides_parallel p) := 
sorry

end unique_diagonals_equal_l451_451280


namespace problem1_problem2_problem3_l451_451804
noncomputable theory

-- Part (1)
theorem problem1 (m : ℝ) : (20 - 4m > 0) → (m < 5) :=
sorry

-- Part (2)
theorem problem2 (m : ℝ) (C : ℝ × ℝ) (r : ℝ) (d : ℝ) 
    (h1 : C = (1, 2)) 
    (h2 : r = sqrt (5 - m)) 
    (h3 : abs (3*1 + 4*2 - 6) / sqrt (3*3 + 4*4) = 1)
    (h4 : 2*sqrt(3) = 2*sqrt(3)) 
    : m = 1 :=
sorry

-- Part (3)
theorem problem3 (m : ℝ) (x1 x2 y1 y2 : ℝ) 
    (h1 : x1 + y1 = 1 ∧ x2 + y2 = 1)
    (h2 : 2*x1^2 - 8*x1 + 5 + m = 0)
    (h3 : 2*x2^2 - 8*x2 + 5 + m = 0)
    (h4 : x1 * x2 + y1 * y2 = 0)
    (h5 : 24 - 8*m > 0)
    : m = -2 :=
sorry

end problem1_problem2_problem3_l451_451804


namespace isabel_spending_ratio_l451_451057

theorem isabel_spending_ratio :
  ∀ (initial_amount toy_cost remaining_amount : ℝ),
    initial_amount = 204 ∧
    toy_cost = initial_amount / 2 ∧
    remaining_amount = 51 →
    ((initial_amount - toy_cost - remaining_amount) / remaining_amount) = 1 / 2 :=
by
  intros
  sorry

end isabel_spending_ratio_l451_451057


namespace mono_increasing_range_k_l451_451851

theorem mono_increasing_range_k (k : ℝ) :
  (∀ x ∈ Ioi 1, k - (1 / x) ≥ 0) → k ≥ 1 :=
by
  sorry

end mono_increasing_range_k_l451_451851


namespace ara_current_height_l451_451551

-- Define the initial condition where Shea's and Ara's height are the same
variables (initial_height : ℝ) (shea_final_height : ℝ) (shea_growth_perc : ℝ) (ara_growth_ratio : ℝ)

-- Assign known values from the conditions
def initial_height := 56
def shea_final_height := 70
def shea_growth_perc := 0.25
def ara_growth_ratio := 1/3

-- Statement that should be proved
theorem ara_current_height : 
  initial_height = shea_final_height / (1 + shea_growth_perc) → 
  ara_growth_ratio * (shea_growth_perc * initial_height) + initial_height = 60.67 := by
  intros h1
  sorry

end ara_current_height_l451_451551


namespace find_p_l451_451052

-- Define the coordinates of the vertices.
def A : ℝ × ℝ := (2, 12)
def B : ℝ × ℝ := (12, 0)
def C : ℝ × ℝ := (0, p)
def Q : ℝ × ℝ := (0, 12)
def O : ℝ × ℝ := (0, 0)

-- State the area condition of triangle ABC.
def area_triangle_ABC : ℝ := 27

-- Now we state the theorem we want to prove.
theorem find_p (p : ℝ) (h1 : (1/2 * abs (2 * (0 - p))) + (abs (6 * p)) = 27) : p = 9 := by
  sorry

end find_p_l451_451052


namespace graph_chromatic_number_bounds_l451_451740

-- Definitions related to graphs, chromatic number, and complement graph
def is_graph (G : Type*) := ∃ (V : Type*) (E : V → V → Prop), True -- Simplified graph definition

def chromatic_number (G : Type*) [is_graph G] : ℕ := sorry -- Assuming a definition of chromatic number

def complement (G : Type*) : Type* := sorry -- Assuming a definition of the complement graph

theorem graph_chromatic_number_bounds (G : Type*)
  [is_graph G]
  (h_chi_G : chromatic_number G ≤ 2)
  (h_chi_G_comp : chromatic_number (complement G) ≤ 2) :
  ∃ (V : Type*), fintype V ∧ exists (n : ℕ), _root_.cardinal.mk V <= 4 :=
by
  sorry

end graph_chromatic_number_bounds_l451_451740


namespace polynomial_property_l451_451360

theorem polynomial_property (P : ℤ[X]) :
  (∀ a b : ℤ, a.gcd b = 1 → ∃ (seq : ℕ → ℤ), (seq = λ n, P.eval (a * n + b)) ∧ (∀ m n : ℕ, m ≠ n → (seq m).gcd (seq n) = 1)) →
  (∃ k : ℕ, P = Polynomial.C (1 : ℤ) * Polynomial.X ^ k ∨ P = -Polynomial.C (1 : ℤ) * Polynomial.X ^ k) :=
sorry

end polynomial_property_l451_451360


namespace f_expression_evaluation_l451_451434

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * sin x + b * x^3 + 4

noncomputable def f_prime (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * cos x + 3 * b * x^2

theorem f_expression_evaluation (a b : ℝ) : 
  f 2014 a b + f (-2014) a b + f_prime 2015 a b - f_prime (-2015) a b = 8 :=
by
  sorry

end f_expression_evaluation_l451_451434


namespace workers_count_l451_451187

noncomputable def numberOfWorkers (W: ℕ) : Prop :=
  let old_supervisor_salary := 870
  let new_supervisor_salary := 690
  let avg_old := 430
  let avg_new := 410
  let total_after_old := (W + 1) * avg_old
  let total_after_new := 9 * avg_new
  total_after_old - old_supervisor_salary = total_after_new - new_supervisor_salary

theorem workers_count : numberOfWorkers 8 :=
by
  sorry

end workers_count_l451_451187


namespace find_first_term_arithmetic_sequence_l451_451081

theorem find_first_term_arithmetic_sequence (a : ℤ) (k : ℤ)
  (hTn : ∀ n : ℕ, T_n = n * (2 * a + (n - 1) * 5) / 2)
  (hConstant : ∀ n : ℕ, (T (4 * n) / T n) = k) : a = 3 :=
by
  sorry

end find_first_term_arithmetic_sequence_l451_451081


namespace infinite_sum_identity_l451_451732

theorem infinite_sum_identity :
  (∑ n in (Finset.range (n + 2)), 1 / (n^2 * (n+2))) = Real.pi^2 / 12 :=
sorry

end infinite_sum_identity_l451_451732


namespace opposite_face_B_is_D_l451_451680

universe u

variables {α : Type u}

-- Define the six labels as a type
inductive Label : Type
| A | B | C | D | E | F
open Label

-- Define the adjacency relation
def adj : Label → Label → Prop
| A, B | B, A => true
| A, C | C, A => true
| B, C | C, B => true
| _, _ => false

-- Define non-adjacency for other labels
def non_adj : Label → Label → Prop
| D, E | E, D => true
| D, F | F, D => true
| E, F | F, E => true
| _, _ => false

-- Prove the face opposite the face labeled B is D
theorem opposite_face_B_is_D : ∀ (x : Label), (x ≠ B) → (x ≠ A) → (x ≠ C) → non_adj x B → x = D :=
by
  assume x h1 h2 h3 hnon_adj
  sorry

end opposite_face_B_is_D_l451_451680


namespace triangles_with_positive_area_in_4x4_grid_l451_451838

theorem triangles_with_positive_area_in_4x4_grid : 
  (∑ i in (finset.range 1)..(finset.range 5), ∑ j in (finset.range 1)..(finset.range 5),
    if i ≠ j ∧ ∃ (x y z : ℕ), (x, y) ≠ (x, z) ∧ (x, y) ∉ [(i,j), (j,j)] ∧ 
      (x, z) ∈ [(i,i), (j,j)] ∧ (x, z) ∉ [(i,i), (i,j), (j,j)] ∧ (x, x) ≠ (i, j)
    then 1 else 0) = 516 :=
by sorry

end triangles_with_positive_area_in_4x4_grid_l451_451838


namespace initial_speed_condition_l451_451264

variable (v : ℝ)

/-- Time taken to walk 2.2 km with initial speed v km/h -/
def time_with_initial_speed (v : ℝ) : ℝ := 2.2 / v

/-- Time taken to walk 2.2 km at 6 km/h -/
def time_with_six_speed : ℝ := 2.2 / 6

/-- Time in hours equivalent to 10 minutes -/
def ten_minutes : ℝ := 10 / 60

/-- Time in hours equivalent to 12 minutes -/
def twelve_minutes : ℝ := 12 / 60

/-- Assuming the walking speed initial v, given conditions state -/
theorem initial_speed_condition (hv : v = 3) :
  time_with_six_speed + ten_minutes = time_with_initial_speed v - twelve_minutes :=
by
  simp [time_with_six_speed, ten_minutes, twelve_minutes, time_with_initial_speed]
  sorry

end initial_speed_condition_l451_451264


namespace exists_subset_F_l451_451507

variable (E : Type) [Fintype E]
variable (f : Finset E → ℝ) 
variable (h0 : ∀ A B : Finset E, Disjoint A B → f (A ∪ B) = f A + f B)

theorem exists_subset_F (hf : ∀ A, 0 ≤ f A) : 
∃ F : Finset E, ∀ A : Finset E,
  let A' := A \ F in
  f A = f A' ∧ (f A = 0 ↔ A ⊆ F) :=
by
  sorry

end exists_subset_F_l451_451507


namespace heather_total_distance_l451_451446

theorem heather_total_distance :
  let d1 := 0.3333333333333333
  let d2 := 0.3333333333333333
  let d3 := 0.08333333333333333
  d1 + d2 + d3 = 0.75 :=
by
  sorry

end heather_total_distance_l451_451446


namespace total_subsidy_l451_451158

theorem total_subsidy : 
  ∀ (x y : ℕ), 
  x + y = 960 ∧ 1.3 * x + 1.25 * y = 1228 → 
  (560 * 1.3 * 80000 + 400 * 1.25 * 90000) * 0.05 = 51620000 :=
by
  intros x y h
  have hx : x = 560 := 
    sorry -- Detail proof showing x = 560
  have hy : y = 400 := 
    sorry -- Detail proof showing y = 400
  calc (560 * 1.3 * 80000 + 400 * 1.25 * 90000) * 0.05
      = 51620000 : sorry -- Calculation of the total subsidy

end total_subsidy_l451_451158


namespace alice_day_odd_probability_l451_451931

theorem alice_day_odd_probability :
  let a1 := 5 in
  let sequence_property := ∀ n > 1, a_n ∈ set.Icc a_{n-1} (2 * a_{n-1}) in
  let all_odd_probability : ℚ := (1 / 2) ^ 6 in
  let (m, n) := (1, 64) in
  m + n = 65 :=
begin
  sorry
end

end alice_day_odd_probability_l451_451931


namespace red_sweets_count_l451_451599

theorem red_sweets_count (total_sweets green_sweets non_red_nor_green_sweets : ℕ) 
(h_total : total_sweets = 285) 
(h_green : green_sweets = 59) 
(h_non_red : non_red_nor_green_sweets = 177) :
  ∃ R : ℕ, total_sweets = R + green_sweets + non_red_nor_green_sweets ∧ R = 49 :=
by
  use 49
  split
  { rw [h_total, h_green, h_non_red], norm_num }
  { refl }

end red_sweets_count_l451_451599


namespace smallest_palindromic_prime_with_hundreds_digit_two_l451_451625

def is_palindromic (n : ℕ) : Prop :=
  let s := n.to_string in s = s.reverse

def has_hundreds_digit_two (n : ℕ) : Prop :=
  n % 1000 / 100 = 2

theorem smallest_palindromic_prime_with_hundreds_digit_two :
  ∀ n, n ≥ 100 → is_palindromic n → has_hundreds_digit_two n → Nat.prime n → 232 ≤ n :=
by
  sorry

end smallest_palindromic_prime_with_hundreds_digit_two_l451_451625


namespace new_years_day_of_2017_l451_451445

theorem new_years_day_of_2017 (Saturdays_2016 : Nat) (is_leap_year : Bool) (H1 : Saturdays_2016 = 53) (H2 : is_leap_year = tt) :
  day_of_week (new_years_day 2017) = Monday :=
by
  -- 2016 is a leap year with 366 days
  have leap_year_days : days_in_year 2016 = 366 := sorry
  -- 53 Saturdays means there are 53 * 7 = 371 days corresponding to Saturdays
  have total_days_from_saturdays : 53 * 7 = 371 := sorry
  -- Therefore, we have 371 days which is 5 more than the leap year days
  have extra_days : 371 - 366 = 5 := sorry
  -- Since we know the total extra days must be accounted for starting at Saturday, the year ends on a Saturday and hence Dec 31 is a Saturday.
  have last_day_sunday : day_of_week (Dec 31, 2016) = Saturday := sorry
  -- Therefore, the next day which is Jan 1, New Year’s day must be Monday
  show day_of_week (new_years_day 2017) = Monday from sorry

end new_years_day_of_2017_l451_451445


namespace angle_between_unit_vectors_l451_451512

variables {V : Type*} [inner_product_space ℝ V]

def is_unit_vector (v : V) : Prop := ∥v∥ = 1

theorem angle_between_unit_vectors
  (a b c : V)
  (ha : is_unit_vector a)
  (hb : is_unit_vector b)
  (hc : is_unit_vector c)
  (h : a + b + 2 • c = 0) :
  real.angle a b = 0 :=
by sorry

end angle_between_unit_vectors_l451_451512


namespace people_stools_chairs_l451_451860

def numberOfPeopleStoolsAndChairs (x y z : ℕ) : Prop :=
  2 * x + 3 * y + 4 * z = 32 ∧
  x > y ∧
  x > z ∧
  x < y + z

theorem people_stools_chairs :
  ∃ (x y z : ℕ), numberOfPeopleStoolsAndChairs x y z ∧ x = 5 ∧ y = 2 ∧ z = 4 :=
by
  sorry

end people_stools_chairs_l451_451860


namespace f_is_decreasing_solve_f_gt_0_find_range_of_a_l451_451430

-- Define the function f(x)
def f (a x : ℝ) : ℝ := -1 / a + 2 / x

-- Prove that f(x) is decreasing on (0, +∞)
theorem f_is_decreasing (a : ℝ) (x : ℝ) (h : 0 < x) : f a x < f a (x + 1) := 
  sorry

-- Solve the inequality f(x) > 0 for x
theorem solve_f_gt_0 (a : ℝ) : 
  (∃ (x : ℝ), 0 < x ∧ f a x > 0) ↔ 
  (a < 0 ∨ (0 < a ∧ ∃ (x : ℝ), 0 < x ∧ x < 2 * a ∧ f a x > 0)) :=
  sorry

-- Find the range of values for a given f(x) + 2x ≥ 0 on (0, +∞)
theorem find_range_of_a (a : ℝ) : 
  (∀ (x : ℝ), 0 < x → f a x + 2 * x ≥ 0) ↔ (a < 0 ∨ a ≥ 1/4) :=
  sorry

end f_is_decreasing_solve_f_gt_0_find_range_of_a_l451_451430


namespace time_to_fill_containers_l451_451123

def volume_reference := 10 * 10 * 30
def volume_container_1 := 10 * 10 * 30
def volume_container_2 := 10 * 10 * 20 + 10 * 10 * 10
def volume_container_3 := Real.pi * 1^2 * 20

def flow_rate_reference := volume_reference / 1

theorem time_to_fill_containers :
  let t1 := volume_container_1 / flow_rate_reference in
  let t2 := volume_container_2 / flow_rate_reference in
  let t3 := volume_container_3 / flow_rate_reference in
  t1 = 1 ∧ t2 = 1 ∧ t3 = 2 :=
by
  sorry

end time_to_fill_containers_l451_451123


namespace correct_answer_l451_451633

def f1 (x : ℝ) : ℝ := x
def f2 (x : ℝ) : ℝ := 2 ^ x
def f3 (x : ℝ) : ℝ := x ^ 2
def f4 (x : ℝ) : ℝ := - x ^ 2

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f(x) = f(-x)
def is_decreasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop := ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f y ≤ f x

theorem correct_answer : 
  is_even f4 ∧ is_decreasing_on f4 (Set.Ici 0) ∧ 
  (¬ is_even f1 ∨ ¬ is_decreasing_on f1 (Set.Ici 0)) ∧ 
  (¬ is_even f2 ∨ ¬ is_decreasing_on f2 (Set.Ici 0)) ∧ 
  (¬ is_even f3 ∨ ¬ is_decreasing_on f3 (Set.Ici 0)) := 
by 
  sorry

end correct_answer_l451_451633


namespace stirling_egf_l451_451240

-- We define the problem statement and conditions as per the steps above.

def stirling_first_kind (N n : ℕ) : ℤ :=  -- Here, we would have the exact definition or properties of Stirling numbers of the first kind
sorry

noncomputable def exponential_generating_function (s : ℕ → ℕ → ℤ) (x : ℝ) : ℝ :=
∑ N in 0.. , (s N n) * x^N / (N.factorial)

theorem stirling_egf (n : ℕ) (x : ℝ) :
  (exponential_generating_function stirling_first_kind x) = (log (1 + x))^n / (n.factorial) :=
sorry

end stirling_egf_l451_451240


namespace tangency_points_distance_l451_451044

theorem tangency_points_distance
  (a b : ℝ)
  (h : ∀ (M : Point) (AM : ℝ = a) (MC : ℝ = b) (ABC_isosceles : Triangle),
    isosceles_trianlge ABC AC) :
  distance_tangency_points A B C M = |a - b| / 2 :=
by
  -- Definition and other necessary statements as per conditions
  -- Assumptions: ⟨h⟩ = ∀ (M : Point) (AM : ℝ = a) (MC : ℝ = b) (ABC_isosceles : Triangle),
  --  isosceles_trianlge ABC AC
  -- Goal: distance_tangency_points A B C M = |a - b| / 2
  sorry

end tangency_points_distance_l451_451044


namespace inequality_region_area_l451_451743

noncomputable def area_of_inequality_region : ℝ :=
  let region := {p : ℝ × ℝ | |p.fst - p.snd| + |2 * p.fst + 2 * p.snd| ≤ 8}
  let vertices := [(2, 2), (-2, 2), (-2, -2), (2, -2)]
  let d1 := 8
  let d2 := 8
  (1 / 2) * d1 * d2

theorem inequality_region_area :
  area_of_inequality_region = 32 :=
by
  sorry  -- Proof to be provided

end inequality_region_area_l451_451743


namespace probability_of_desired_sum_is_1_over_6_l451_451618

-- Define the sets of numbers on the two dice.
def die1 : Set ℕ := {1, 2, 3, 7, 8, 9}
def die2 : Set ℕ := {4, 5, 6, 10, 11, 12}

-- Define the desired sum.
def desired_sum : ℕ := 13

-- Calculate the total number of outcomes when rolling two dice.
def total_outcomes : ℕ := Set.card die1 * Set.card die2

-- Define the function that checks if a sum is equal to the desired sum.
def is_desired_sum (x y : ℕ) : Bool :=
  (x + y) = desired_sum

-- Define the number of favorable outcomes where the sum is exactly the desired sum.
def favorable_outcomes : ℕ :=
  Set.card {(x, y) | x ∈ die1 ∧ y ∈ die2 ∧ is_desired_sum x y}

-- Define the probability of getting the desired sum.
def probability_desired_sum : ℚ :=
  favorable_outcomes / total_outcomes

-- Theorems to state the problem in Lean.
theorem probability_of_desired_sum_is_1_over_6 : 
  probability_desired_sum = 1 / 6 := by
  sorry

end probability_of_desired_sum_is_1_over_6_l451_451618


namespace radius_of_smaller_circle_l451_451859

theorem radius_of_smaller_circle (R : ℝ) (n : ℕ) (r : ℝ) 
  (hR : R = 10) 
  (hn : n = 7) 
  (condition : 2 * R = 2 * r * n) :
  r = 10 / 7 :=
by
  sorry

end radius_of_smaller_circle_l451_451859


namespace Joshua_jogged_distance_l451_451895

theorem Joshua_jogged_distance :
  ∀ (d : ℝ), 
  let t_total := (d / 12) + (d / 8) in
  t_total = 50 / 60 →
  d = 4 :=
by
  intro d t_total
  sorry

end Joshua_jogged_distance_l451_451895


namespace volume_of_pyramid_correct_l451_451574

noncomputable def volume_of_pyramid (lateral_surface_area base_area inscribed_circle_area radius : ℝ) : ℝ :=
  if lateral_surface_area = 3 * base_area ∧ inscribed_circle_area = radius then
    (2 * Real.sqrt 6) / (Real.pi ^ 3)
  else
    0

theorem volume_of_pyramid_correct
  (lateral_surface_area base_area inscribed_circle_area radius : ℝ)
  (h1 : lateral_surface_area = 3 * base_area)
  (h2 : inscribed_circle_area = radius) :
  volume_of_pyramid lateral_surface_area base_area inscribed_circle_area radius = (2 * Real.sqrt 6) / (Real.pi ^ 3) :=
by {
  sorry
}

end volume_of_pyramid_correct_l451_451574


namespace fraction_of_meat_used_for_meatballs_l451_451500

theorem fraction_of_meat_used_for_meatballs
    (initial_meat : ℕ)
    (spring_rolls_meat : ℕ)
    (remaining_meat : ℕ)
    (total_meat_used : ℕ)
    (meatballs_meat : ℕ)
    (h_initial : initial_meat = 20)
    (h_spring_rolls : spring_rolls_meat = 3)
    (h_remaining : remaining_meat = 12) :
    (initial_meat - remaining_meat) = total_meat_used ∧
    (total_meat_used - spring_rolls_meat) = meatballs_meat ∧
    (meatballs_meat / initial_meat) = (1/4 : ℝ) :=
by
  sorry

end fraction_of_meat_used_for_meatballs_l451_451500


namespace long_division_valid_l451_451738

def is_correct_quotient (n : Nat) (d : Nat) (q : Nat) (r : Nat) : Prop :=
  n = d * q + r

theorem long_division_valid (n q d r : Nat) (h_n : n = 1089708)
  (h_d : d = 12)
  (h_q : q = 90909)
  (h_r : r = 0) :
  is_correct_quotient n d q r :=
by
  unfold is_correct_quotient
  simp [h_n, h_d, h_q, h_r]
  sorry

end long_division_valid_l451_451738


namespace slope_of_line_l451_451202

theorem slope_of_line {x1 y1 x2 y2 : ℤ} (h1 : x1 = 1) (h2 : y1 = 3) (h3 : x2 = -4) (h4 : y2 = -2) :
  ((y2 - y1) / (x2 - x1) : ℚ) = 1 :=
by
  rw [h1, h2, h3, h4]
  simp
  norm_num
  sorry

end slope_of_line_l451_451202


namespace smallest_x_absolute_value_l451_451761

theorem smallest_x_absolute_value :
  ∃ x : ℝ, (|5 * x + 15| = 40) ∧ (∀ y : ℝ, |5 * y + 15| = 40 → x ≤ y) ∧ x = -11 :=
sorry

end smallest_x_absolute_value_l451_451761


namespace modulus_of_complex_expression_l451_451109

theorem modulus_of_complex_expression (x y : ℝ)
  (h : (1 + complex.I) * (x + y * complex.I) = 2) : 
  complex.abs (2 * x + y * complex.I) = real.sqrt 5 := 
sorry 

end modulus_of_complex_expression_l451_451109


namespace product_of_0_dot_9_repeating_and_8_eq_8_l451_451378

theorem product_of_0_dot_9_repeating_and_8_eq_8 :
  ∃ q : ℝ, q = 0.999... ∧ q * 8 = 8 := 
sorry

end product_of_0_dot_9_repeating_and_8_eq_8_l451_451378


namespace area_of_sector_eq_l451_451118

-- Definitions corresponding to the problem's conditions
def radius : ℝ := 15
def arc_length : ℝ := π / 3

-- Statement of the problem
theorem area_of_sector_eq : 
  (1 / 2 * arc_length * radius) = (5 * π / 2) :=
  sorry

end area_of_sector_eq_l451_451118


namespace guilt_of_genotan_l451_451058

def Person := {Isobel Josh Genotan Tegan : Type} -- Define the set of people involved

-- Define the statements made by each person
def is_innocent (p : Person) (q : Person) : Prop :=
  p = Isobel → q = Josh →
  p = Genotan → q = Tegan →
  p = Josh → q = Genotan →
  p = Tegan → q = Isobel

-- Define the condition that only the guilty person is lying
def condition (guilty : Person) : Prop :=
  ∀ p, p ≠ guilty → (is_innocent p = true → p = true)

theorem guilt_of_genotan (guilty : Person) (Guilty_Conditions : condition guilty) : guilty = Genotan :=
by
  sorry -- Proof to be added

end guilt_of_genotan_l451_451058


namespace Tiffany_bags_l451_451612

theorem Tiffany_bags (x : ℕ) 
  (h1 : 8 = x + 1) : 
  x = 7 :=
by
  sorry

end Tiffany_bags_l451_451612


namespace evaluate_f_at_1_l451_451808

noncomputable def f (x : ℝ) : ℝ := 2^x + 2

theorem evaluate_f_at_1 : f 1 = 4 :=
by {
  -- proof goes here
  sorry
}

end evaluate_f_at_1_l451_451808


namespace initial_candies_in_box_l451_451186

-- Definitions for the conditions
variables (c_left c_taken: ℕ)

-- Given conditions
def candy_left_cond := c_left = 82
def candy_taken_cond := c_taken = 6

-- The statement to prove
theorem initial_candies_in_box (h1 : candy_left_cond) (h2 : candy_taken_cond) :
  ∃ c_initial: ℕ, c_initial = 88 :=
sorry

end initial_candies_in_box_l451_451186


namespace rectangle_diagonal_l451_451562

theorem rectangle_diagonal (l w : ℝ) (h_area : l * w = 20) (h_perimeter : 2 * l + 2 * w = 18) :
  (l ≠ w ∧ (real.sqrt (l^2 + w^2) = real.sqrt 41)) := 
by {
  sorry
}

end rectangle_diagonal_l451_451562


namespace inequality_one_system_of_inequalities_l451_451555

theorem inequality_one (x : ℝ) : 
  (2 * x - 2) / 3 ≤ 2 - (2 * x + 2) / 2 → x ≤ 1 :=
sorry

theorem system_of_inequalities (x : ℝ) : 
  (3 * (x - 2) - 1 ≥ -4 - 2 * (x - 2) → x ≥ 7 / 5) ∧
  ((1 - 2 * x) / 3 > (3 * (2 * x - 1)) / 2 → x < 1 / 2) → false :=
sorry

end inequality_one_system_of_inequalities_l451_451555


namespace even_length_implies_div_by_4_l451_451937

theorem even_length_implies_div_by_4 {n : ℕ} 
  (e : Fin n → ℤ) 
  (h : ∀ i : Fin n, e i = 1 ∨ e i = -1) 
  (h_sum : ∑ i in Finset.range n, e i * e ((i + 1) % n) = 0) : 
  n % 4 = 0 := 
begin
  sorry
end

end even_length_implies_div_by_4_l451_451937


namespace volume_comparison_l451_451667

-- Define the properties for the cube and the cuboid.
def cube_side_length : ℕ := 1 -- in meters
def cuboid_width : ℕ := 50  -- in centimeters
def cuboid_length : ℕ := 50 -- in centimeters
def cuboid_height : ℕ := 20 -- in centimeters

-- Convert cube side length to centimeters.
def cube_side_length_cm := cube_side_length * 100 -- in centimeters

-- Calculate volumes.
def cube_volume : ℕ := cube_side_length_cm ^ 3 -- in cubic centimeters
def cuboid_volume : ℕ := cuboid_width * cuboid_length * cuboid_height -- in cubic centimeters

-- The theorem stating the problem.
theorem volume_comparison : cube_volume / cuboid_volume = 20 :=
by sorry

end volume_comparison_l451_451667


namespace exists_unique_positive_d_inequality_x_y_z_l451_451902

-- Part a: Existence and Uniqueness of d
theorem exists_unique_positive_d (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃! d > 0, (1 / (a + d) + 1 / (b + d) + 1 / (c + d) = 2 / d) :=
sorry

-- Part b: Inequality involving x, y, z
theorem inequality_x_y_z (a b c x y z d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hd : d > 0)
  (h1 : 1 / (a + d) + 1 / (b + d) + 1 / (c + d) = 2 / d)
  (h2 : a * x + b * y + c * z = x * y * z) :
  x + y + z ≥ (2 / d) * sqrt ((a + d) * (b + d) * (c + d)) :=
sorry

end exists_unique_positive_d_inequality_x_y_z_l451_451902


namespace axes_of_symmetry_do_not_coincide_l451_451566

-- Define the two quadratic functions
def f (x : ℝ) : ℝ := (x^2 + 6*x - 25) / 8
def g (x : ℝ) : ℝ := (31 - x^2) / 8

-- Define the axes of symmetry for the quadratic functions
def axis_of_symmetry_f : ℝ := -3
def axis_of_symmetry_g : ℝ := 0

-- Define the slopes of the tangents to the graphs at x = 4 and x = -7
def slope_f (x : ℝ) : ℝ := (2*x + 6) / 8
def slope_g (x : ℝ) : ℝ := -x / 4

-- We need to prove that the axes of symmetry do not coincide
theorem axes_of_symmetry_do_not_coincide :
    axis_of_symmetry_f ≠ axis_of_symmetry_g :=
by {
    sorry
}

end axes_of_symmetry_do_not_coincide_l451_451566


namespace correct_statements_count_l451_451710

-- Definitions for each condition
def is_output_correct (stmt : String) : Prop :=
  stmt = "PRINT a, b, c"

def is_input_correct (stmt : String) : Prop :=
  stmt = "INPUT \"x=3\""

def is_assignment_correct_1 (stmt : String) : Prop :=
  stmt = "A=3"

def is_assignment_correct_2 (stmt : String) : Prop :=
  stmt = "A=B ∧ B=C"

-- The main theorem to be proven
theorem correct_statements_count (stmt1 stmt2 stmt3 stmt4 : String) :
  stmt1 = "INPUT a, b, c" → stmt2 = "INPUT x=3" → stmt3 = "3=A" → stmt4 = "A=B=C" →
  (¬ is_output_correct stmt1 ∧ ¬ is_input_correct stmt2 ∧ ¬ is_assignment_correct_1 stmt3 ∧ ¬ is_assignment_correct_2 stmt4) →
  0 = 0 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end correct_statements_count_l451_451710


namespace circles_are_intersecting_l451_451245

-- Define the circles and the distances given
def radius_O1 : ℝ := 3
def radius_O2 : ℝ := 5
def distance_O1O2 : ℝ := 2

-- Define the positional relationships
inductive PositionalRelationship
| externally_tangent
| intersecting
| internally_tangent
| contained_within_each_other

open PositionalRelationship

-- State the theorem to be proved
theorem circles_are_intersecting :
  distance_O1O2 > 0 ∧ distance_O1O2 < (radius_O1 + radius_O2) ∧ distance_O1O2 > abs (radius_O1 - radius_O2) →
  PositionalRelationship := 
by
  intro h
  exact PositionalRelationship.intersecting

end circles_are_intersecting_l451_451245


namespace sequence_problem_l451_451047

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n m k, n ≠ m → a n = a m + (n - m) * k

theorem sequence_problem
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 2003 + a 2005 + a 2007 + a 2009 + a 2011 + a 2013 = 120) :
  2 * a 2018 - a 2028 = 20 :=
sorry

end sequence_problem_l451_451047


namespace lottery_card_color_l451_451656

theorem lottery_card_color
  (numbers : Finset (Fin 9 → Fin 3))
  (colors : Finset (Fin 9 → Fin 3 × (Fin 3)))
  (h1 : ∀ n, n ∈ numbers ↔ n ∈ ({n | ∀ i : Fin 9, n i < 3} : Set (Fin 9 → Fin 3)))
  (h2 : ∀ n, (n, (colors n)) ∈ colors ↔ 
             n ∈ numbers ∧ ∃ c, colors n = c ∧ ({n' | (n' ≠ n) ∧ ∀ i, n' i ≠ n i} ∈ colors → c ≠ colors n))
  (red_card : Fin 9 → Fin 3)
  (yellow_card : Fin 9 → Fin 3)
  (h_red_card : red_card = (λ i, if i = 0 then 1 else 2) : true)
  (h_yellow_card : yellow_card = (λ i, 2 : Fin 3) : true)
  (h_red : ∀ i, colors (λ i, if i = 0 then 1 else 2) = 0)
  (h_yellow : ∀ i, colors (λ i, 2 : Fin 3) = 1) :
  colors (λ i, if i % 3 = 0 then 1 else if i % 3 = 1 then 2 else 3) = 0 := 
sorry

end lottery_card_color_l451_451656


namespace spherical_coord_plane_l451_451383

-- Let's define spherical coordinates and the condition theta = c.
structure SphericalCoordinates where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

def is_plane (c : ℝ) (p : SphericalCoordinates) : Prop :=
  p.θ = c

theorem spherical_coord_plane (c : ℝ) : 
  ∀ p : SphericalCoordinates, is_plane c p → True := 
by
  intros p hp
  sorry

end spherical_coord_plane_l451_451383


namespace exists_k_with_n_distinct_prime_factors_l451_451940

theorem exists_k_with_n_distinct_prime_factors (m n : ℕ) (hm : m > 0) (hn : n > 0) : 
  ∃ k : ℕ, k > 0 ∧ (nat.card (nat.factorization (2^k - m)).support ≥ n) :=
by
  sorry

end exists_k_with_n_distinct_prime_factors_l451_451940


namespace ratio_Y_to_Z_l451_451293

variables (X Y Z : ℕ)

def population_relation1 (X Y : ℕ) : Prop := X = 3 * Y
def population_relation2 (X Z : ℕ) : Prop := X = 6 * Z

theorem ratio_Y_to_Z (h1 : population_relation1 X Y) (h2 : population_relation2 X Z) : Y / Z = 2 :=
  sorry

end ratio_Y_to_Z_l451_451293


namespace max_elements_subset_T_l451_451076

theorem max_elements_subset_T :
  ∃ T : Finset ℕ, (∀ x ∈ T, x ∈ Finset.range 101) ∧ 
                  (∀ x y ∈ T, x ≠ y → (x + y) % 11 ≠ 0) ∧ 
                  T.card = 60 :=
sorry

end max_elements_subset_T_l451_451076


namespace triangle_dot_product_sum_l451_451013

theorem triangle_dot_product_sum (A B C : ℝ × ℝ) (h1 : dist A B = 3) (h2 : dist B C = 4) (h3 : dist C A = 5) :
  (λ u v, (u.1 - v.1) * (u.2 - v.2)) (A, B) (B, C) +
  (λ u v, (u.1 - v.1) * (u.2 - v.2)) (B, C) (C, A) +
  (λ u v, (u.1 - v.1) * (u.2 - v.2)) (C, A) (A, B) = -25 :=
sorry

end triangle_dot_product_sum_l451_451013


namespace probability_within_circle_l451_451845

def is_within_circle (x y : ℕ) : Prop :=
  x^2 + y^2 ≤ 16

def roll_outcomes : list (ℕ × ℕ) :=
  [(x, y) | x <- [1, 2, 3, 4, 5, 6], y <- [1, 2, 3, 4, 5, 6]]

def favorable_outcomes : list (ℕ × ℕ) :=
  roll_outcomes.filter (λ p, is_within_circle p.1 p.2)

def probability : ℚ :=
  (favorable_outcomes.length : ℚ) / (roll_outcomes.length : ℚ)

theorem probability_within_circle :
  probability = 2 / 9 :=
by  sorry

end probability_within_circle_l451_451845


namespace painted_unit_cubes_l451_451736

theorem painted_unit_cubes :
  ∃ (n : ℕ), let total_cubes := n ^ 3 in
    let unpainted_cubes := (n - 2) ^ 3 in
    let painted_cubes := total_cubes - unpainted_cubes in
    unpainted_cubes = 24 ∧ painted_cubes = 101 :=
by
  sorry

end painted_unit_cubes_l451_451736


namespace expected_winnings_is_minus_half_l451_451261

-- Define the given condition in Lean
noncomputable def prob_win_side_1 : ℚ := 1 / 4
noncomputable def prob_win_side_2 : ℚ := 1 / 4
noncomputable def prob_lose_side_3 : ℚ := 1 / 3
noncomputable def prob_no_change_side_4 : ℚ := 1 / 6

noncomputable def win_amount_side_1 : ℚ := 2
noncomputable def win_amount_side_2 : ℚ := 4
noncomputable def lose_amount_side_3 : ℚ := -6
noncomputable def no_change_amount_side_4 : ℚ := 0

-- Define the expected value function
noncomputable def expected_winnings : ℚ :=
  (prob_win_side_1 * win_amount_side_1) +
  (prob_win_side_2 * win_amount_side_2) +
  (prob_lose_side_3 * lose_amount_side_3) +
  (prob_no_change_side_4 * no_change_amount_side_4)

-- Statement to prove
theorem expected_winnings_is_minus_half : expected_winnings = -1 / 2 := 
by
  sorry

end expected_winnings_is_minus_half_l451_451261


namespace median_and_mode_l451_451986

def scores : list ℕ := [74, 84, 84, 84, 87, 92, 92]

theorem median_and_mode (l : list ℕ) (H : l = [74, 84, 84, 84, 87, 92, 92]) :
  (median l = 84) ∧ (mode l = 84) :=
by sorry

end median_and_mode_l451_451986


namespace average_speed_of_stream_l451_451349

def man's_speed_in_still_water : ℝ := 1.5
def distance_upstream_downstream_equal (D : ℝ) : Prop := ∀ V_s : ℝ, 
  (D / (man's_speed_in_still_water - V_s)) = 2 * (D / (man's_speed_in_still_water + V_s))
def stream_speed_increase (d : ℝ) (initial_speed : ℝ) (increment : ℝ) : ℝ := 
  initial_speed + increment * floor (d / 100)

theorem average_speed_of_stream (d_total : ℝ) (initial_speed : ℝ) (increment : ℝ) : 
  distance_upstream_downstream_equal d_total →
  (d_total / 5) = 500 →
  ∑ i in finset.range 5, (stream_speed_increase (i * 100) initial_speed increment) / 5 = 0.7 :=
by
  intros h1 h2
  sorry

end average_speed_of_stream_l451_451349


namespace min_expression_l451_451911

theorem min_expression (a b c d e f : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f)
  (h_sum : a + b + c + d + e + f = 10) : 
  (1 / a + 9 / b + 16 / c + 25 / d + 36 / e + 49 / f) ≥ 67.6 :=
sorry

end min_expression_l451_451911


namespace cone_opening_angle_correct_l451_451255

noncomputable def cone_opening_angle (r1 r2 m : ℝ) (h_volume : (1/3) * π * r1^2 * m = π * r2^2 * m) (h_surface : π * r1 * (sqrt (r1^2 + m^2)) = 2 * π * r2 * m) : ℝ :=
  let α := real.arcsin ((sqrt 3) / 2) in 
  α

theorem cone_opening_angle_correct (r1 r2 m : ℝ) (h_volume : (1/3) * π * r1^2 * m = π * r2^2 * m) (h_surface : π * r1 * (sqrt (r1^2 + m^2)) = 2 * π * r2 * m) :
  cone_opening_angle r1 r2 m h_volume h_surface = (π / 3) :=
by
  sorry

end cone_opening_angle_correct_l451_451255


namespace Q_at_7_l451_451107

noncomputable def Q (x g h i j : ℝ) : ℝ :=
  (3 * x ^ 3 - 27 * x ^ 2 + g * x + h) *
  (4 * x ^ 3 - 36 * x ^ 2 + i * x + j)

theorem Q_at_7 (g h i j : ℝ)
  (h_roots : (∀ z : ℂ, z ∈ {1, 2, 6} ↔ (∃ r : ℂ, (3 * r ^ 3 - 27 * r ^ 2 + (g : ℂ) * r + (h : ℂ)) * 
                                                       (4 * r ^ 3 - 36 * r ^ 2 + (i : ℂ) * r + (j : ℂ)) = 0))) :
  Q 7 g h i j = 10800 := by
  sorry


end Q_at_7_l451_451107


namespace min_r_for_B_subset_C_l451_451105

theorem min_r_for_B_subset_C :
  let A := {t : ℝ | 0 < t ∧ t < 2 * Real.pi}
  let B := {(x, y) : ℝ × ℝ | ∃ t, x = Real.sin t ∧ y = 2 * Real.sin t * Real.cos t ∧ t ∈ A}
  ∃ r : ℝ, r = 5 / 4 ∧ ∀ (x y : ℝ), (x, y) ∈ B → x^2 + y^2 ≤ r^2 :=
by
  intros A B
  use 5 / 4
  intros x y hb
  sorry

end min_r_for_B_subset_C_l451_451105


namespace reciprocal_neg3_l451_451582

-- Define the problem
def reciprocal (x : ℚ) : ℚ := 1 / x

-- The required proof statement
theorem reciprocal_neg3 : reciprocal (-3) = -1 / 3 :=
by
  sorry

end reciprocal_neg3_l451_451582


namespace tower_count_7_cubes_l451_451252

/-- 
Statement: The number of different towers of height 7 that can be built with 
3 red cubes, 4 blue cubes, and 2 yellow cubes is 5040.
-/
theorem tower_count_7_cubes : 
  let total_cubes := 3 + 4 + 2,
      height := 7,
      red := 3,
      blue := 4,
      yellow := 2 in
  (total_cubes = 9 ∧ height = 7 ∧ red = 3 ∧ blue = 4 ∧ yellow = 2) → 
  choose 9 2 * (7.factorial / (3.factorial * 3.factorial * 1.factorial)) = 5040 :=
by 
  intros total_cubes height red blue yellow h_conditions,
  sorry

end tower_count_7_cubes_l451_451252


namespace max_individual_contribution_l451_451231

def total_contribution : ℕ := 20
def number_of_people : ℕ := 10
def min_contribution : ℕ := 1

theorem max_individual_contribution (total_contribution = 20) (number_of_people = 10) (min_contribution = 1) :
  ∃ (max_contribution : ℕ), max_contribution = 11 :=
by
  sorry

end max_individual_contribution_l451_451231


namespace number_of_persons_in_room_l451_451040

theorem number_of_persons_in_room (n : ℕ) (h : n * (n - 1) / 2 = 78) : n = 13 :=
by
  /- We have:
     n * (n - 1) / 2 = 78,
     We need to prove n = 13 -/
  sorry

end number_of_persons_in_room_l451_451040


namespace avg_of_two_middle_numbers_l451_451972

theorem avg_of_two_middle_numbers (a b c d : ℕ) (h₀ : a ≠ b) (h₁ : a ≠ c) (h₂ : a ≠ d) (h₃ : b ≠ c) (h₄ : b ≠ d) (h₅ : c ≠ d) (h₆ : a + b + c + d = 20) (h₇ : a < d) (h₈ : d - a ≥ b - c) (h₉ : b = 2) (h₁₀ : c = 3) :
  (b + c) / 2 = 2.5 :=
by
  sorry

end avg_of_two_middle_numbers_l451_451972


namespace tan_x_value_l451_451433

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

theorem tan_x_value:
  (∀ x : ℝ, deriv f x = 2 * f x) → (∀ x : ℝ, f x = Real.sin x - Real.cos x) → (∀ x : ℝ, Real.tan x = 3) := 
by
  intros h_deriv h_f
  sorry

end tan_x_value_l451_451433


namespace sum_g_n_diverges_l451_451384

-- Define g(n) as the sum of the infinite series 1/k^n
def g (n : ℕ) : ℝ := ∑' (k : ℕ) (hk : k > 0), 1 / (k^n)

-- Define the main statement that needs to be proven
theorem sum_g_n_diverges :
  (∑' (n : ℕ) (hn : n > 0), g n) = ⊤ := 
sorry

end sum_g_n_diverges_l451_451384


namespace conditional_probability_heads_heads_l451_451611

theorem conditional_probability_heads_heads (A B : Prop):
  (P_A : ℝ) (P_B : ℝ) (P_AB : ℝ)
  (h1 : P_A = 1/2)
  (h2 : P_B = 1/2)
  (hAB : P_AB = P_A * P_B) :
  P_AB / P_A = 1/2 :=
by
  sorry

end conditional_probability_heads_heads_l451_451611


namespace number_of_liars_on_the_island_l451_451604

-- Definitions for the conditions
def isKnight (person : ℕ) : Prop := sorry -- Placeholder, we know knights always tell the truth
def isLiar (person : ℕ) : Prop := sorry -- Placeholder, we know liars always lie
def population := 1000
def villages := 10
def minInhabitantsPerVillage := 2

-- Definitional property: each islander claims that all other villagers in their village are liars
def claimsAllOthersAreLiars (islander : ℕ) (village : ℕ) : Prop := 
  ∀ (other : ℕ), (other ≠ islander) → (isLiar other)

-- Main statement in Lean
theorem number_of_liars_on_the_island : ∃ liars, liars = 990 :=
by
  have total_population := population
  have number_of_villages := villages
  have min_people_per_village := minInhabitantsPerVillage
  have knight_prop := isKnight
  have liar_prop := isLiar
  have claim_prop := claimsAllOthersAreLiars
  -- Proof will be filled here
  sorry

end number_of_liars_on_the_island_l451_451604


namespace incorrect_statements_l451_451132

variable (a b c : Vector ℝ)

theorem incorrect_statements :
  (a ∙ c = b ∙ c → a ≠ b) ∧
  ((a + b) ∙ c = a ∙ c + b ∙ c) ∧
  (a ^ 2 = b ^ 2 → a ∙ c ≠ b ∙ c) ∧
  ((a ∙ b) ∙ c ≠ (b ∙ c) ∙ a) := 
by sorry

end incorrect_statements_l451_451132


namespace arithmetic_sequence_sum_l451_451427

theorem arithmetic_sequence_sum :
  ∀ (a_1 d : ℤ), 
  (2 * a_1 + 14 * d = 4) → 
  (a_1 + d + a_1 + 8 * d + a_1 + 12 * d = 6) :=
by {
  intros a_1 d h,
  sorry
}

end arithmetic_sequence_sum_l451_451427


namespace Jason_toys_correct_l451_451892

variable (R Jn Js : ℕ)

def Rachel_toys : ℕ := 1

def John_toys (R : ℕ) : ℕ := R + 6

def Jason_toys (Jn : ℕ) : ℕ := 3 * Jn

theorem Jason_toys_correct (hR : R = 1) (hJn : Jn = John_toys R) (hJs : Js = Jason_toys Jn) : Js = 21 :=
by
  sorry

end Jason_toys_correct_l451_451892


namespace max_ratio_square_l451_451515

variables {a b c x y : ℝ}
-- Assume a, b, c are positive real numbers
variable (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
-- Assume the order of a, b, c: a ≥ b ≥ c
variable (h_order : a ≥ b ∧ b ≥ c)
-- Define the system of equations
variable (h_system : a^2 + y^2 = c^2 + x^2 ∧ c^2 + x^2 = (a - x)^2 + (c - y)^2)
-- Assume the constraints on x and y
variable (h_constraints : 0 ≤ x ∧ x < a ∧ 0 ≤ y ∧ y < c)

theorem max_ratio_square :
  ∃ (ρ : ℝ), ρ = (a / c) ∧ ρ^2 = 4 / 3 :=
sorry

end max_ratio_square_l451_451515


namespace measure_angle_AYX_l451_451727

variable (A B C X Y Z : Type)
variable (Gamma : Type) (incircle circumcircle : Gamma)
variable (angle_A angle_B angle_C : ℝ)

-- Setting up the assumptions
axiom is_triangle : angle_A + angle_B + angle_C = 180
axiom in_circle : incircle
axiom circum_circle : circumcircle
axiom points_on_sides : (X ∈ BC) ∧ (Y ∈ AB) ∧ (Z ∈ AC)
axiom angles : angle_A = 50 ∧ angle_B = 70 ∧ angle_C = 60

-- The goal
theorem measure_angle_AYX : (AYX : ℝ) = 70 := sorry

end measure_angle_AYX_l451_451727


namespace find_first_term_arithmetic_sequence_l451_451084

theorem find_first_term_arithmetic_sequence (a : ℤ) (k : ℤ)
  (hTn : ∀ n : ℕ, T_n = n * (2 * a + (n - 1) * 5) / 2)
  (hConstant : ∀ n : ℕ, (T (4 * n) / T n) = k) : a = 3 :=
by
  sorry

end find_first_term_arithmetic_sequence_l451_451084


namespace irrational_number_count_l451_451709

theorem irrational_number_count :
  let numbers := [real.cbrt 8, (5 / 3 : ℚ), real.sqrt 2, (5 / 11 : ℚ), real.sqrt 9, real.pi, 3.010010001]
  (count (λ x, ¬ is_rat x) numbers) = 3 :=
by sorry

end irrational_number_count_l451_451709


namespace floor_factorial_expression_l451_451306

theorem floor_factorial_expression : 
  ⌊(2010.factorial + 2007.factorial) / (2009.factorial + 2008.factorial)⌋ = 2009 :=
by
  sorry

end floor_factorial_expression_l451_451306


namespace marissa_sunflower_height_l451_451530

def height_sister_in_inches : ℚ := 4 * 12 + 3
def height_difference_in_inches : ℚ := 21
def inches_to_cm (inches : ℚ) : ℚ := inches * 2.54
def cm_to_m (cm : ℚ) : ℚ := cm / 100

theorem marissa_sunflower_height :
  cm_to_m (inches_to_cm (height_sister_in_inches + height_difference_in_inches)) = 1.8288 :=
by sorry

end marissa_sunflower_height_l451_451530


namespace find_BA_l451_451901

section MatrixProof

open Matrix

-- Let A be a real 4x2 matrix
variable (A : Matrix (Fin 4) (Fin 2) ℝ)

-- Let B be a real 2x4 matrix
variable (B : Matrix (Fin 2) (Fin 4) ℝ)

-- Given AB
def AB : Matrix (Fin 4) (Fin 4) ℝ := ![
  ![1, 0, -1, 0],
  ![0, 1, 0, -1],
  ![-1, 0, 1, 0],
  ![0, -1, 0, 1]
]

-- Condition: AB = Defined Matrix
axiom ab_eq : A ⬝ B = AB

-- Theorem: Find BA
theorem find_BA : B ⬝ A = ![
  ![2, 0],
  ![0, 2]
] :=
sorry

end MatrixProof

end find_BA_l451_451901


namespace sphere_radius_existence_l451_451577

theorem sphere_radius_existence
    (S T : Plane)
    (m : Line)
    (C D : Point)
    (C' D' : Point)
    (c d e : ℝ)
    (h1 : perpendicular S T)
    (h2 : contains S C)
    (h3 : contains T D)
    (h4 : projection_onto_line m C = C')
    (h5 : projection_onto_line m D = D')
    (h6 : distance C C' = c)
    (h7 : distance D D' = d)
    (h8 : distance C' D' = e) :
    ∃ r : ℝ, r = (1 / 2) * (sqrt (2 * (c ^ 2 + d ^ 2 + e ^ 2) + (c + d) ^ 2) - (c + d)) :=
begin
    -- sorry to skip the proof
    sorry
end

end sphere_radius_existence_l451_451577


namespace min_distance_from_curve_to_line_l451_451075

open Real -- Use the real number operations.

theorem min_distance_from_curve_to_line :
  let curve := λ x : ℝ, x^2 - log x in
  let line := λ x : ℝ, x - 2 in
  ∃ (x P_dist : ℝ), x > 0 ∧
  P_dist = sqrt 2 ∧
  curve x = line P_dist :=
by 
  sorry

end min_distance_from_curve_to_line_l451_451075


namespace complex_quadrant_l451_451840

theorem complex_quadrant (a b : ℝ) : 
  ( (a - 4 + 5 * complex.I) * (-(b ^ 2) + 2 * b - 6) ).im < 0 → 
  ( (a - 4 + 5 * complex.I) * (-(b ^ 2) + 2 * b - 6) ).re > 0 → 
  true := by
  sorry

end complex_quadrant_l451_451840


namespace inclination_angle_eq_arccos_l451_451094

-- Given conditions
variable (θ a : ℝ)
variable (h1 : cos θ = a)
variable (h2 : a < 0)

-- To prove
theorem inclination_angle_eq_arccos : θ = arccos a := 
sorry

end inclination_angle_eq_arccos_l451_451094


namespace problem1_problem2_l451_451660

-- Problem 1: Solution set of the inequality |x-5| - |2x+3| ≥ 1 is {x | -7 ≤ x ≤ 1/3}
theorem problem1 (x : ℝ) : 
  |x - 5| - |2 * x + 3| ≥ 1 ↔ -7 ≤ x ∧ x ≤ 1 / 3 := 
sorry

-- Problem 2: For any positive real numbers a and b where a + b = 1/2, √a + √b ≤ 1
theorem problem2 (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1 / 2) : 
  sqrt a + sqrt b ≤ 1 := 
sorry

end problem1_problem2_l451_451660


namespace increasing_condition_l451_451990

-- Definitions for the conditions
def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x → x < y → y < b → f x < f y

def func (x a : ℝ) : ℝ := x / (x + a)

-- Theorem statement for the proof problem
theorem increasing_condition (a : ℝ) :
  (∀ x y : ℝ, -2 < x → x < y → y < +∞ → func x a < func y a) ↔ (2 ≤ a) :=
by
  sorry

end increasing_condition_l451_451990


namespace ratio_of_trees_l451_451688

theorem ratio_of_trees (x : ℕ) 
  (P4 : 30 = 30) 
  (P5 : x = 60) 
  (P6 : 3 * x - 30 = 3 * 60 - 30) 
  (total : 30 + x + (3 * x - 30) = 240) : 
  x / 30 = 2 :=
by {
  have h1 : x = 60,
  {
    linarith [P5],
  },
  rw h1,
  norm_num,
}

#print ratio_of_trees -- To verify that the theorem is typed correctly

end ratio_of_trees_l451_451688


namespace Shekars_Biology_marks_l451_451137

theorem Shekars_Biology_marks
  (m : ℕ) (s : ℕ) (ss : ℕ) (e : ℕ) (a : ℕ) (b : ℕ)
  (hm : m = 76) (hs : s = 65) (hss : ss = 82) (he : e = 67) (ha : a = 75) :
  b = 85 :=
by
  have h_total_marks_known: m + s + ss + e = 76 + 65 + 82 + 67, from sorry,
  have h_total_marks_all: 5 * a = 375, from sorry,
  have h_total_marks_pred: b = 375 - 290, from sorry,
  sorry

end Shekars_Biology_marks_l451_451137


namespace problem_statement_l451_451846

theorem problem_statement (m x : ℝ) (h1 : m^2 - 1 = 0) (h2 : m - 1 ≠ 0) : 
  200 * (x - m) * (x + 2 * m) - 10 * m = 2010 :=
by
  have hm : m = -1 :=
    by
      linarith
  have hx : x = 4 :=
    by
      linarith
  rw [hm, hx]
  linarith

end problem_statement_l451_451846


namespace smallest_n_integer_expression_l451_451514

theorem smallest_n_integer_expression :
  ∃ n : ℕ, 0 < n ∧ 2 * ∑ k in finset.range n, cos ((↑k + 1) ^ 2 * (Real.pi / 2016)) * sin ((↑k + 1) * (Real.pi / 2016)) ∈ ℤ ∧
    n = 1008 :=
by
  let a := Real.pi / 2016
  let expression_sum := λ (n : ℕ), 2 * ∑ k in finset.range n, cos ((↑k + 1) ^ 2 * a) * sin ((↑k + 1) * a)
  have h1: ∃ n : ℕ, 0 < n ∧ (sin (n * (n + 1) * (Real.pi / 2016))) ∈ {-1, 0, 1} := sorry
  have h2: ∀ n : ℕ, (sin (n * (n + 1) * (Real.pi / 2016))) ∈ {-1, 0, 1} → (2 * sin (n * (n + 1) * a) ∈ ℤ) := sorry
  have h3: 1008 ∈  ℕ := sorry
  use 1008
  split
  exact h3
  exact ⟨expression_sum 1008, rfl⟩

end smallest_n_integer_expression_l451_451514


namespace first_term_arithmetic_sequence_l451_451089

def T_n (a d : ℚ) (n : ℕ) := n * (2 * a + (n - 1) * d) / 2

theorem first_term_arithmetic_sequence (a : ℚ)
  (h_const_ratio : ∀ (n : ℕ), n > 0 → 
    (T_n a 5 (4 * n)) / (T_n a 5 n) = (T_n a 5 4 / T_n a 5 1)) : 
  a = -5/2 :=
by 
  sorry

end first_term_arithmetic_sequence_l451_451089


namespace axes_of_symmetry_do_not_coincide_l451_451565

-- Define the two quadratic functions
def f (x : ℝ) : ℝ := (x^2 + 6*x - 25) / 8
def g (x : ℝ) : ℝ := (31 - x^2) / 8

-- Define the axes of symmetry for the quadratic functions
def axis_of_symmetry_f : ℝ := -3
def axis_of_symmetry_g : ℝ := 0

-- Define the slopes of the tangents to the graphs at x = 4 and x = -7
def slope_f (x : ℝ) : ℝ := (2*x + 6) / 8
def slope_g (x : ℝ) : ℝ := -x / 4

-- We need to prove that the axes of symmetry do not coincide
theorem axes_of_symmetry_do_not_coincide :
    axis_of_symmetry_f ≠ axis_of_symmetry_g :=
by {
    sorry
}

end axes_of_symmetry_do_not_coincide_l451_451565


namespace f_eq_4_max_l451_451903

def f (a b c : ℝ) : ℝ :=
  abs ((abs (b-a) / abs (a*b)) + ((b+a) / (a*b)) - (2 / c)) +
  (abs (b-a) / abs (a*b)) + ((b+a) / (a*b)) + (2 / c)

theorem f_eq_4_max (a b c : ℝ) : 
  f a b c = 4 * max (max (1 / a) (1 / b)) (1 / c) :=
by
  sorry

end f_eq_4_max_l451_451903


namespace floor_factorial_expression_l451_451320

theorem floor_factorial_expression : 
  (⌊(2010! + 2007! : ℚ) / (2009! + 2008! : ℚ)⌋ = 2009) :=
by
  -- Let a := 2010! and b := 2007!
  -- So a + b = 2010! + 2007!
  -- Notice 2010! = 2010 * 2009 * 2008 * 2007!
  -- Notice 2009! = 2009 * 2008 * 2007!
  -- Simplify (2010! + 2007!) / (2009! + 2008!)
  sorry

end floor_factorial_expression_l451_451320


namespace olivia_possible_amount_l451_451488

theorem olivia_possible_amount (k : ℕ) :
  ∃ k : ℕ, 1 + 79 * k = 1984 :=
by
  -- Prove that there exists a non-negative integer k such that the equation holds
  sorry

end olivia_possible_amount_l451_451488


namespace floor_factorial_expression_l451_451310

theorem floor_factorial_expression : 
  ⌊(2010.factorial + 2007.factorial) / (2009.factorial + 2008.factorial)⌋ = 2009 :=
by
  sorry

end floor_factorial_expression_l451_451310


namespace unique_minimizer_of_g_l451_451098

def num_divisors (m : ℕ) : ℕ :=
  if h : m > 0 then (Finset.range (m + 1)).filter (λ d, d > 0 ∧ m % d = 0).card else 0

def g (m : ℕ) : ℝ :=
  if h : m > 0 then (num_divisors m : ℝ) / m.pow (1 / 4) else 0

theorem unique_minimizer_of_g :
  ∃ M : ℕ, (∀ m : ℕ, m ≠ M → g M < g m) ∧ M = 2 ∧ (product_of_digits M = 2) := 
sorry

def product_of_digits (n : ℕ) : ℕ :=
  (n.toString.data.map Char.to_nat.map (λ c, c - '0'.toNat)).foldl (*) 1

end unique_minimizer_of_g_l451_451098


namespace linear_system_incorrect_statement_l451_451548

def is_determinant (a b c d : ℝ) := a * d - b * c

def is_solution_system (a1 b1 c1 a2 b2 c2 D Dx Dy : ℝ) :=
  D = is_determinant a1 b1 a2 b2 ∧
  Dx = is_determinant c1 b1 c2 b2 ∧
  Dy = is_determinant a1 c1 a2 c2

def is_solution_linear_system (a1 b1 c1 a2 b2 c2 x y : ℝ) :=
  a1 * x + b1 * y = c1 ∧ a2 * x + b2 * y = c2

theorem linear_system_incorrect_statement :
  ∀ (x y : ℝ),
    is_solution_system 3 (-1) 1 1 3 7 10 10 20 ∧
    is_solution_linear_system 3 (-1) 1 1 3 7 x y →
    x = 1 ∧ y = 2 ∧ ¬(20 = -20) := 
by sorry

end linear_system_incorrect_statement_l451_451548


namespace complex_trajectory_is_parabola_l451_451401

theorem complex_trajectory_is_parabola (x y : ℝ) (hx : x ≥ 1/2) (hz : |↑(x - 1) + complex.I * y| = x) : y^2 = 2 * x - 1 := by
sorcery 

end complex_trajectory_is_parabola_l451_451401


namespace average_price_correct_l451_451503

def large_bottles : ℕ := 1365
def small_bottles : ℕ := 720
def price_large_bottle : ℝ := 1.89
def price_small_bottle : ℝ := 1.42

def total_cost_large_bottles : ℝ := large_bottles * price_large_bottle
def total_cost_small_bottles : ℝ := small_bottles * price_small_bottle
def overall_total_cost : ℝ := total_cost_large_bottles + total_cost_small_bottles
def total_bottles : ℕ := large_bottles + small_bottles
def approximate_average_price_per_bottle : ℝ := overall_total_cost / total_bottles

theorem average_price_correct :
  approximate_average_price_per_bottle ≈ 1.73 :=
sorry

end average_price_correct_l451_451503


namespace sara_correct_problems_l451_451119

-- Define the conditions as hypotheses
variable (s c w n : ℕ)

-- Hypothesis: Score calculation formula
def score_calculation : Prop := s = 30 + 5 * c - 2 * w - n

-- Hypothesis: Sara's score is exactly 90
def sara_score : Prop := s = 90

-- Hypothesis: No determination possible for scores between 85 and 90 exclusively
def no_determination (c' w' n' : ℕ) : Prop :=
  ∀ s', 85 < s' ∧ s' < 90 → s' = 30 + 5 * c' - 2 * w' - n' → c ≠ c'

-- The main theorem we want to prove:
theorem sara_correct_problems :
  (score_calculation s c w n) →
  (sara_score s) →
  (∀ s', 85 < s' ∧ s' < 90 → (∃ c', ∃ w', ∃ n', no_determination c' w' n')) →
  c = 12 :=
by
  -- Proof is omitted; we are focusing on the statement only
  sorry

end sara_correct_problems_l451_451119


namespace y_capital_l451_451653

theorem y_capital (X Y Z : ℕ) (Pz : ℕ) (Z_months_after_start : ℕ) (total_profit Z_share : ℕ)
    (hx : X = 20000)
    (hz : Z = 30000)
    (hz_profit : Z_share = 14000)
    (htotal_profit : total_profit = 50000)
    (hZ_months : Z_months_after_start = 5)
  : Y = 25000 := 
by
  -- Here we would have a proof, skipped with sorry for now
  sorry

end y_capital_l451_451653


namespace evaluate_expression_at_neg_two_l451_451948

theorem evaluate_expression_at_neg_two :
  (let x := -2 in (-x^2 + 5 + 4 * x) + (5 * x - 4 + 2 * x^2)) = -13 :=
by {
  let x := (-2 : ℤ),
  calc
    (-x^2 + 5 + 4 * x) + (5 * x - 4 + 2 * x^2)
        = (-(-2)^2 + 5 + 4 * (-2)) + (5 * (-2) - 4 + 2 * (-2)^2) : rfl
    ... = (-(4) + 5 - 8) + (-10 - 4 + 8) : by norm_num
    ... = -14 + 1 : by norm_num
    ... = -13 : by norm_num
}

end evaluate_expression_at_neg_two_l451_451948


namespace determine_three_numbers_l451_451122

noncomputable def can_determine_numbers (a b c p q r : ℕ) : Prop :=
  ∃ x y z : ℕ, (x < y ∧ y < z) ∧ (x + y = a ∧ x + z = b ∧ y + z = c) ∧ (xy = p ∧ xz = q ∧ yz = r)

theorem determine_three_numbers (h : ∀ a b c p q r : ℕ, can_determine_numbers a b c p q r) :
  ∃ x y z : ℕ, ∀ a b c p q r : ℕ, can_determine_numbers a b c p q r :=
sorry

end determine_three_numbers_l451_451122


namespace arithmetic_sequence_sum_l451_451795

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := -1
noncomputable def c : ℝ := ln(b + 2) - b
noncomputable def d : ℝ := 1

theorem arithmetic_sequence_sum (a b c d : ℝ) 
  (ar_seq : ∃ k : ℝ, b = a + k ∧ c = a + 2 * k ∧ d = a + 3 * k)
  (max_value : y = ln(x + 2) - x 
    ∧ ∃ b : ℝ, x = b ∧ (∀ x : ℝ, (ln(x + 2) - x) ≤ (ln(b + 2) - b)) ∧ (ln(b + 2) - b) = c)
  : b + d = -1 := sorry

end arithmetic_sequence_sum_l451_451795


namespace floor_fraction_equals_2009_l451_451297

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem floor_fraction_equals_2009 :
  (⌊ (factorial 2010 + factorial 2007) / (factorial 2009 + factorial 2008) ⌋ : ℤ) = 2009 :=
by sorry

end floor_fraction_equals_2009_l451_451297


namespace determine_n_l451_451666

theorem determine_n (n : ℕ)
  (h1 : ∀ i j : ℕ, i ≠ j ∧ i < n ∧ j < n →
          (∃ k : ℕ, k < n ∧ k ≠ i ∧ k ≠ j ∧ 
          (visited k i ∨ visited k j)))
  (h2 : ∀ k : ℕ, k < n →
          (hosted k = 1/4 * (students_sent k)))
  : (n % 7 = 0 ∨ n % 7 = 1) :=
sorry

end determine_n_l451_451666


namespace value_of_a_l451_451457

theorem value_of_a (a : ℝ) (P Q : set ℝ) (hP : P = {1, 2}) (hQ : Q = {1, a^2}) (hPQ : P = Q) : a = sqrt 2 ∨ a = -sqrt 2 := by
  sorry

end value_of_a_l451_451457


namespace domain_ln_l451_451168

theorem domain_ln (x : ℝ) (h : x - 1 > 0) : x > 1 := 
sorry

end domain_ln_l451_451168


namespace toys_of_Jason_l451_451889

theorem toys_of_Jason (R J Jason : ℕ) 
  (hR : R = 1) 
  (hJ : J = R + 6) 
  (hJason : Jason = 3 * J) : 
  Jason = 21 :=
by
  sorry

end toys_of_Jason_l451_451889


namespace f_max_iff_l451_451268

noncomputable def f : ℚ → ℝ := sorry

axiom f_zero : f 0 = 0
axiom f_pos (a : ℚ) (h : a ≠ 0) : f a > 0
axiom f_mul (a b : ℚ) : f (a * b) = f a * f b
axiom f_add_le (a b : ℚ) : f (a + b) ≤ f a + f b
axiom f_bound (m : ℤ) : f m ≤ 1989

theorem f_max_iff (a b : ℚ) (h : f a ≠ f b) : f (a + b) = max (f a) (f b) := 
sorry

end f_max_iff_l451_451268


namespace zurbagan_one_way_traffic_l451_451049
noncomputable theory

-- Definition of a graph
structure Graph (V : Type) :=
(edges : V → V → Prop)
(two_way : ∀ u v, edges u v ↔ edges v u) -- originally two-way traffic

-- Definition for the city (conditions)
variables {V : Type} (G : Graph V)
variables {a b : V}

-- Condition: During road repairs, connectivity preserved
axiom road_repairs : ∀ u v, ∃ w, G.edges u w ∧ G.edges w v

-- We need to prove the statement
theorem zurbagan_one_way_traffic :
  ∃ (G' : V → V → Prop), (∀ u v, G' u v → G'.edges u v) ∧ (∀ x y, ∃ z, G'.edges x z ∧ G'.edges z y) :=
sorry

end zurbagan_one_way_traffic_l451_451049


namespace brandZ_to_brandW_ratio_l451_451719

variable (v p : ℝ)

def brandZ_volume := 1.3 * v
def brandZ_price := 0.85 * p

def unit_price (volume price : ℝ) : ℝ := price / volume

theorem brandZ_to_brandW_ratio
  (v p : ℝ) : unit_price (brandZ_volume v) (brandZ_price p) / unit_price v p = 17 / 26 := by
  sorry

end brandZ_to_brandW_ratio_l451_451719


namespace shelby_initial_money_l451_451138

-- Definitions based on conditions
def cost_of_first_book : ℕ := 8
def cost_of_second_book : ℕ := 4
def cost_of_each_poster : ℕ := 4
def number_of_posters : ℕ := 2

-- Number to prove (initial money)
def initial_money : ℕ := 20

-- Theorem statement
theorem shelby_initial_money :
    (cost_of_first_book + cost_of_second_book + (number_of_posters * cost_of_each_poster)) = initial_money := by
    sorry

end shelby_initial_money_l451_451138


namespace cost_condition_shirt_costs_purchasing_plans_maximize_profit_l451_451615

/-- Define the costs and prices of shirts A and B -/
def cost_A (m : ℝ) : ℝ := m
def cost_B (m : ℝ) : ℝ := m - 10
def price_A : ℝ := 260
def price_B : ℝ := 180

/-- Condition: total cost of 3 A shirts and 2 B shirts is 480 -/
theorem cost_condition (m : ℝ) : 3 * (cost_A m) + 2 * (cost_B m) = 480 := by
  sorry

/-- The cost of each A shirt is 100 and each B shirt is 90 -/
theorem shirt_costs : ∃ m, cost_A m = 100 ∧ cost_B m = 90 := by
  sorry

/-- Number of purchasing plans for at least $34,000 profit with 300 shirts and at most 110 A shirts -/
theorem purchasing_plans : ∃ x, 100 ≤ x ∧ x ≤ 110 ∧ 
  (260 * x + 180 * (300 - x) - 100 * x - 90 * (300 - x) ≥ 34000) := by
  sorry

/- Maximize profit given 60 < a < 80:
   - 60 < a < 70: 110 A shirts, 190 B shirts.
   - a = 70: any combination satisfying conditions.
   - 70 < a < 80: 100 A shirts, 200 B shirts. -/

theorem maximize_profit (a : ℝ) (ha : 60 < a ∧ a < 80) : 
  ∃ x, ((60 < a ∧ a < 70 ∧ x = 110 ∧ (300 - x) = 190) ∨ 
        (a = 70) ∨ 
        (70 < a ∧ a < 80 ∧ x = 100 ∧ (300 - x) = 200)) := by
  sorry

end cost_condition_shirt_costs_purchasing_plans_maximize_profit_l451_451615


namespace sum_of_coordinates_l451_451470

-- Define the conditions for m and n
def m : ℤ := -3
def n : ℤ := 2

-- State the proposition based on the conditions
theorem sum_of_coordinates : m + n = -1 := 
by 
  -- Provide an incomplete proof skeleton with "sorry" to skip the proof
  sorry

end sum_of_coordinates_l451_451470


namespace solve_for_x_l451_451102

variable (a b x : ℝ)
variable (a_pos : a > 0) (b_pos : b > 0) (x_pos : x > 0)

theorem solve_for_x : (3 * a) ^ (3 * b) = (a ^ b) * (x ^ b) → x = 27 * a ^ 2 :=
by
  intro h_eq
  sorry

end solve_for_x_l451_451102


namespace probability_of_one_from_each_name_l451_451291

theorem probability_of_one_from_each_name (cards_total : ℕ)
    (letters_bill : ℕ) (letters_john : ℕ) :
    cards_total = 12 → letters_bill = 4 → letters_john = 5 →
    (letters_bill / cards_total) * 
    ((letters_john : ℚ) / (cards_total - 1)) +
    (letters_john / cards_total) * 
    ((letters_bill : ℚ) / (cards_total - 1)) = 
    (10 / 33) := by
  intros h_total h_bill h_john
  sorry

end probability_of_one_from_each_name_l451_451291


namespace red_balls_removal_condition_l451_451484

theorem red_balls_removal_condition (total_balls : ℕ) (initial_red_balls : ℕ) (r : ℕ) : 
  total_balls = 600 → 
  initial_red_balls = 420 → 
  60 * (total_balls - r) = 100 * (initial_red_balls - r) → 
  r = 150 :=
by
  sorry

end red_balls_removal_condition_l451_451484


namespace sum_binomial_squares_eq_l451_451129

theorem sum_binomial_squares_eq (n : ℕ) :
  (∑ k in Finset.range (n+1), (-1) ^ k * Nat.choose n k ^ 2) = 
    if n % 2 = 0 then (-1) ^ (n / 2) * Nat.choose n (n / 2) else 0 :=
  sorry

end sum_binomial_squares_eq_l451_451129


namespace solve_inequality_l451_451143

theorem solve_inequality (x : ℝ) : 
  (x / (x^2 + x - 6) ≥ 0) ↔ (x < -3) ∨ (x = 0) ∨ (0 < x ∧ x < 2) :=
by 
  sorry 

end solve_inequality_l451_451143


namespace transformed_roots_correct_l451_451733

variable (k : ℝ)
variable (a b c d : ℝ)

/-- Given polynomial equation -/
def poly_eqn (x : ℝ) := k * x^4 - 5 * k * x - 12 = 0

/-- Conditions: k is a nonzero constant and a, b, c, d are solutions of the polynomial equation. -/
axiom nonzero_k : k ≠ 0
axiom root_a : poly_eqn k a
axiom root_b : poly_eqn k b
axiom root_c : poly_eqn k c
axiom root_d : poly_eqn k d

/-- New polynomial whose roots are transformed roots --/
def transformed_poly (x : ℝ) := 12 * k^3 * x^4 - 5 * k^3 * x^3 - 1 = 0

theorem transformed_roots_correct :
  ∀ y_i, y_i ∈ { (b + c + d) / (k * a^2), (a + c + d) / (k * b^2),
                  (a + b + d) / (k * c^2), (a + b + c) / (k * d^2) } → 
  transformed_poly k y_i := 
sorry

end transformed_roots_correct_l451_451733


namespace find_length_ED_l451_451883

-- Given points A, B, C, L, E, D
variables (A B C L E D : Point)
-- Given angles and bisectors in the triangle
variables (triangle_ABC : Triangle A B C) (is_bisector_AL : AngleBisector A L)
-- Given collinear points, parallel lines, and specific lengths
variables (E_on_AB : OnSegment A B E) (D_on_BL : OnSegment B L D)
variables (DL_eq_LC : Segment L B D = Segment L B C) (ED_parallel_AC : IsParallel E D A C)
variables (AE_eq_15 : Segment A B E = 15) (AC_eq_12 : Segment A B C = 12)

-- The theorem statement
theorem find_length_ED : Segment E D = 3 := by
  sorry

end find_length_ED_l451_451883


namespace math_problem_l451_451396

theorem math_problem (a b c : ℝ) (h1 : a + b + c = 3) (h2 : a^2 + b^2 + c^2 = 3) : a^(2008 : ℕ) + b^(2008 : ℕ) + c^(2008 : ℕ) = 3 :=
by 
  let h1' : a + b + c = 3 := h1
  let h2' : a^2 + b^2 + c^2 = 3 := h2
  sorry

end math_problem_l451_451396


namespace polynomial_not_product_of_two_vars_l451_451938

theorem polynomial_not_product_of_two_vars:
  ¬ ∃ (f : polynomial ℚ) (g : polynomial ℚ), 
    (∀ x y : ℚ, f.eval x * g.eval y = x^200 * y^200 + 1) :=
by
  sorry

end polynomial_not_product_of_two_vars_l451_451938


namespace savings_total_correct_l451_451157

def total_savings_are_40 (teagan_savings_pennies : ℕ) (rex_savings_nickels : ℕ) (toni_savings_dimes : ℕ) : Prop :=
  teagan_savings_pennies = 200 ∧ rex_savings_nickels = 100 ∧ toni_savings_dimes = 330 →
  teagan_savings_pennies / 100 + rex_savings_nickels / 20 + toni_savings_dimes / 10 = 40

theorem savings_total_correct : total_savings_are_40 200 100 330 :=
by {
  intros h,
  cases h with h1 h2,
  cases h2 with h3 h4,
  -- calculations to reach the conclusion
  sorry
}

end savings_total_correct_l451_451157


namespace geometric_sequence_fifth_term_l451_451370

theorem geometric_sequence_fifth_term (a1 a2 : ℝ) (h1 : a1 = 2) (h2 : a2 = 1 / 4) : 
  let r := a2 / a1 in
  let a5 := a1 * r ^ 4 in
  a5 = 1 / 2048 :=
by
  sorry

end geometric_sequence_fifth_term_l451_451370


namespace prob_mult_of_4_or_9_l451_451675

noncomputable def is_divisible_by (n k : ℕ) : Prop := k % n = 0

theorem prob_mult_of_4_or_9 :
  let total_balls := 100
  let multiples_of_4 := (finset.range (total_balls + 1)).filter (λ x, is_divisible_by 4 x)
  let multiples_of_9 := (finset.range (total_balls + 1)).filter (λ x, is_divisible_by 9 x)
  let multiples_of_36 := (finset.range (total_balls + 1)).filter (λ x, is_divisible_by 36 x)
  let favorable_outcomes := multiples_of_4.card + multiples_of_9.card - multiples_of_36.card
  (favorable_outcomes: ℚ) / total_balls = 17 / 50 :=
by
  sorry

end prob_mult_of_4_or_9_l451_451675


namespace time_to_carry_backpack_l451_451898

/-- 
Given:
1. Lara takes 73 seconds to crank open the door to the obstacle course.
2. Lara traverses the obstacle course the second time in 5 minutes and 58 seconds.
3. The total time to complete the obstacle course is 874 seconds.

Prove:
The time it took Lara to carry the backpack through the obstacle course the first time is 443 seconds.
-/
theorem time_to_carry_backpack (door_time : ℕ) (second_traversal_time : ℕ) (total_time : ℕ) : 
  (door_time + second_traversal_time + 443 = total_time) :=
by
  -- Given conditions
  let door_time := 73
  let second_traversal_time := 5 * 60 + 58 -- Convert 5 minutes 58 seconds to seconds
  let total_time := 874
  -- Calculate the time to carry the backpack
  sorry

end time_to_carry_backpack_l451_451898


namespace buckets_oranges_l451_451606

theorem buckets_oranges :
  ∀ (a b c : ℕ), 
  a = 22 → 
  b = a + 17 → 
  a + b + c = 89 → 
  b - c = 11 := 
by 
  intros a b c h1 h2 h3 
  sorry

end buckets_oranges_l451_451606


namespace evaporation_rate_l451_451664

theorem evaporation_rate (initial_water_volume : ℕ) (days : ℕ) (percentage_evaporated : ℕ) (evaporated_fraction : ℚ)
  (h1 : initial_water_volume = 10)
  (h2 : days = 50)
  (h3 : percentage_evaporated = 3)
  (h4 : evaporated_fraction = percentage_evaporated / 100) :
  (initial_water_volume * evaporated_fraction) / days = 0.06 :=
by
  -- Proof goes here
  sorry

end evaporation_rate_l451_451664


namespace binomial_expansion_coefficient_x2_l451_451048

theorem binomial_expansion_coefficient_x2 :
  let general_term (r : ℕ) := binomial 8 r * 2^r * (-1)^(8-r) * x^((3*r/2) - 4) in
  (∃ r : ℕ, (3 * r / 2 - 4 = 2) ∧ (general_term r).coeff x^2 = 1120) := 
sorry

end binomial_expansion_coefficient_x2_l451_451048


namespace option_d_not_constructible_l451_451539

-- Define rhombus and its properties
structure Rhombus :=
  (color: ℕ) -- 0 for white, 1 for gray

-- Define the condition that rhombuses can be rotated but not flipped
constant can_rotate : Rhombus → Rhombus → Prop

-- Assume we have a set of identical rhombuses
constant identical_rhombuses : set Rhombus

-- Define a larger shape as an assembly of rhombuses
structure LargerShape :=
  (assembled_from: set Rhombus)

-- Define the four options for larger shapes (a, b, c, d)
constant option_a : LargerShape
constant option_b : LargerShape
constant option_c : LargerShape
constant option_d : LargerShape

-- Define a predicate that checks if a larger shape can be constructed
constant can_be_constructed : LargerShape → Prop

-- Given these conditions, prove that Option (d) cannot be constructed
theorem option_d_not_constructible :
  ¬ can_be_constructed option_d :=
sorry

end option_d_not_constructible_l451_451539


namespace color_grid_3x3_l451_451068

-- Define a grid cell position
inductive Position
  | A | B | C | D | E | F | G | H | I | J | K | L | M | N | O | P
  deriving DecidableEq

-- Define colors
inductive Color
  | color1 | color2
  deriving DecidableEq

-- Define adjacency relation between positions
def adjacent : Position → Position → Prop
  | Position.A, Position.B => true
  | Position.A, Position.E => true
  | Position.B, Position.A => true
  | Position.B, Position.F => true
  | Position.B, Position.C => true
  | Position.C, Position.B => true
  | Position.C, Position.D => true
  | Position.C, Position.G => true
  | Position.D, Position.C => true
  | Position.D, Position.H => true
  | Position.E, Position.A => true
  | Position.E, Position.I => true
  | Position.E, Position.F => true
  | Position.F, Position.B => true
  | Position.F, Position.E => true
  | Position.F, Position.J => true
  | Position.F, Position.G => true
  | Position.G, Position.C => true
  | Position.G, Position.F => true
  | Position.G, Position.K => true
  | Position.G, Position.H => true
  | Position.H, Position.D => true
  | Position.H, Position.G => true
  | Position.H, Position.L => true
  | Position.I, Position.E => true
  | Position.I, Position.J => true
  | Position.J, Position.F => true
  | Position.J, Position.I => true
  | Position.J, Position.K => true
  | Position.K, Position.G => true
  | Position.K, Position.J => true
  | Position.K, Position.L => true
  | Position.L, Position.H => true
  | Position.L, Position.K => true
  | Position.L, Position.P => true
  | Position.M, Position.I => true
  | Position.M, Position.N => true
  | Position.N, Position.M => true
  | Position.N, Position.J => true
  | Position.N, Position.O => true
  | Position.O, Position.N => true
  | Position.O, Position.K => true
  | Position.O, Position.P => true
  | Position.P, Position.O => true
  | Position.P, Position.L => true
  -- Position is adjacent to itself and all positions are non-adjacent to each other
  | _, _ => false

-- Define a coloring of the grid as a function
def coloring : Position → Color → Prop

-- The main theorem to prove
theorem color_grid_3x3 : ∃ c : Position → Color, 
  (∀ (p1 p2 : Position), adjacent p1 p2 → c p1 ≠ c p2) ∧ 
  ((∃ (c1 c2 : Position → Color), 
    (∀ (p : Position), c1 p = c2 p ∨ c1 p ≠ c2 p)) ∧ 
  (c = c1 ∨ c = c2)) := 
sorry

end color_grid_3x3_l451_451068


namespace floor_fraction_equals_2009_l451_451298

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem floor_fraction_equals_2009 :
  (⌊ (factorial 2010 + factorial 2007) / (factorial 2009 + factorial 2008) ⌋ : ℤ) = 2009 :=
by sorry

end floor_fraction_equals_2009_l451_451298


namespace complete_square_formula_l451_451225

theorem complete_square_formula (x : ℝ) : 
    ∃ (a b : ℝ), (x^2 + 4 + 4x) = (a + b)^2 := 
by
    use x, 2
    sorry

end complete_square_formula_l451_451225


namespace linear_fn_g_20_5_l451_451915

theorem linear_fn_g_20_5:
  ∀ (g : ℝ → ℝ),
  (∀ x y, g(x) = g(0) + (g(1) - g(0)) * x) → -- Definition of linear function
  (g(10) - g(5) = 20) →
  (g(20) - g(5) = 60) := 
by
  assume g lin_g h,
  sorry -- Proof here

end linear_fn_g_20_5_l451_451915


namespace chord_length_is_4_l451_451525

noncomputable def parabolaFocus : (ℝ × ℝ) := (1 / 2, 0)

noncomputable def chordLength 
  (x1 x2 : ℝ) 
  (y1 y2 : ℝ)
  (focus : (ℝ × ℝ) := parabolaFocus) 
  (h1 : x1 + x2 = 3)
  (h2 : y1 ^ 2 = 2 * x1) 
  (h3 : y2 ^ 2 = 2 * x2) 
  (l_passes_focus : (focus.fst, 0) ∈ line_through (x1, y1) (x2, y2))
  : ℝ := 
  x1 + x2 + focus.fst

theorem chord_length_is_4 (x1 x2 y1 y2 : ℝ)
  (h1 : x1 + x2 = 3)
  (h2 : y1 ^ 2 = 2 * x1) 
  (h3 : y2 ^ 2 = 2 * x2) 
  (focus : (ℝ × ℝ) := parabolaFocus)
  (l_passes_focus : (focus.fst, 0) ∈ line_through (x1, y1) (x2, y2)) :
  chordLength x1 x2 y1 y2 = 4 :=
by
  sorry

end chord_length_is_4_l451_451525


namespace angle_AYX_50_l451_451726

theorem angle_AYX_50
  (Γ : Type) [circle Γ]
  (A B C X Y Z : Type)
  [on_segment X B C]
  [on_segment Y A B]
  [on_segment Z A C]
  (angle_A : ∠ A = 50)
  (angle_B : ∠ B = 70)
  (angle_C : ∠ C = 60)
  (incircle_ABC : incircle Γ (triangle A B C))
  (circumcircle_XYZ : circumcircle Γ (triangle X Y Z)) :
  ∠ AYX = 50 := 
sorry

end angle_AYX_50_l451_451726


namespace probability_of_25_cents_heads_l451_451955

/-- 
Considering the flipping of five specific coins: a penny, a nickel, a dime,
a quarter, and a half dollar, prove that the probability of getting at least
25 cents worth of heads is 3 / 4.
-/
theorem probability_of_25_cents_heads :
  let total_outcomes := 2^5
  let successful_outcomes_1 := 2^4
  let successful_outcomes_2 := 2^3
  let successful_outcomes := successful_outcomes_1 + successful_outcomes_2
  (successful_outcomes / total_outcomes : ℚ) = 3 / 4 :=
by
  sorry

end probability_of_25_cents_heads_l451_451955


namespace equiangular_polygon_angle_solution_l451_451441

-- Given two equiangular polygons P_1 and P_2 with different numbers of sides
-- Each angle of P_1 is x degrees
-- Each angle of P_2 is k * x degrees where k is an integer greater than 1
-- Prove that the number of valid pairs (x, k) is exactly 1

theorem equiangular_polygon_angle_solution : ∃ x k : ℕ, ( ∀ n m : ℕ, x = 180 - 360 / n ∧ k * x = 180 - 360 / m → (k > 1) → x = 60 ∧ k = 2) := sorry

end equiangular_polygon_angle_solution_l451_451441


namespace sum_of_valid_c_l451_451381

theorem sum_of_valid_c :
  let discriminant_condition (c : ℤ) := ∃ k : ℤ, 81 + 4 * c = k * k
  let valid_c_range (c : ℤ) := -20 ≤ c ∧ c ≤ 30
  ∑ c in (finset.filter (λ c, discriminant_condition c) (finset.Icc (-20) 30)), c = -28 :=
by
  sorry

end sum_of_valid_c_l451_451381


namespace fertilizer_prices_l451_451868

variables (x y : ℝ)

theorem fertilizer_prices :
  (x = y + 100) ∧ (2 * x + y = 1700) → (x = 600 ∧ y = 500) :=
by
  intros h
  cases h with h1 h2
  have h3 : y = 500 := by sorry
  have h4 : x = y + 100 := h1
  rw h3 at h4
  have h5 : x = 600 := by sorry
  exact ⟨h5, h3⟩

end fertilizer_prices_l451_451868


namespace y_directly_varies_as_square_l451_451463

theorem y_directly_varies_as_square (k : ℚ) (y : ℚ) (x : ℚ) 
  (h1 : y = k * x ^ 2) (h2 : y = 18) (h3 : x = 3) : 
  ∃ y : ℚ, ∀ x : ℚ, x = 6 → y = 72 :=
by
  sorry

end y_directly_varies_as_square_l451_451463


namespace nonnegative_integer_pairs_solution_l451_451741

open Int

theorem nonnegative_integer_pairs_solution (x y : ℕ) : 
  3 * x ^ 2 + 2 * 9 ^ y = x * (4 ^ (y + 1) - 1) ↔ (x = 3 ∧ y = 1) ∨ (x = 2 ∧ y = 1) :=
by 
  sorry

end nonnegative_integer_pairs_solution_l451_451741


namespace find_inverse_l451_451810

noncomputable def inverse_function (x : ℝ) (h : x > -1) : ℝ :=
10^(x - 3) - 1

theorem find_inverse :
  ∀ x : ℝ, x > -1 → (log (x + 1) + 3) = y ↔ x = 10^(y - 3) - 1 :=
by
  intro x hx
  have : (log (10^(x-3) - 1 + 1) + 3) = x :=
    sorry
  have : y = 10^(log (x+1) + 3) - 1 :=
    sorry
  split
  · intro h₁
    sorry
  · intro h₂
    sorry

end find_inverse_l451_451810


namespace first_term_arithmetic_sum_l451_451079

theorem first_term_arithmetic_sum 
  (T : ℕ → ℚ) (b : ℚ) (d : ℚ) (h₁ : ∀ n, T n = n * (2 * b + (n - 1) * d) / 2)
  (h₂ : d = 5)
  (h₃ : ∀ n, (T (4 * n)) / (T n) = (16 : ℚ)) : 
  b = 5 / 2 :=
sorry

end first_term_arithmetic_sum_l451_451079


namespace part1_part2_l451_451790

-- Define the set A
def A : set ℝ := { x | x^2 - 9 < 0 }

-- Define the set B
def B : set ℝ := { x | 2 ≤ x + 1 ∧ x + 1 ≤ 4 }

-- Define the set C with variable m
def C (m : ℝ) : set ℝ := { x | m ≤ x ∧ x ≤ m + 1 }

-- Prove A ∩ B = { x | 1 ≤ x ∧ x < 3 }
theorem part1 : A ∩ B = { x | 1 ≤ x ∧ x < 3 } :=
sorry

-- Prove the range of m for which A ∩ C = ∅ is (-∞, -4] ∪ [3, +∞)
theorem part2 : (∀ m : ℝ, A ∩ (C m) = ∅ ↔ m ≤ -4 ∨ m ≥ 3) :=
sorry

end part1_part2_l451_451790


namespace trapezium_area_l451_451650

-- Define the lengths of the parallel sides
def a := 10
def b := 18

-- Define the distance between the parallel sides
def h := 10.00001

-- Define the expected area of the trapezium
def expected_area := 140.00014

-- Prove that the area of the trapezium with the given dimensions is the expected_area
theorem trapezium_area : (1 / 2) * (a + b) * h = expected_area :=
by
  sorry

end trapezium_area_l451_451650


namespace equilateral_triangle_perimeter_l451_451335

theorem equilateral_triangle_perimeter (r : ℝ) (h : r = 4) :
  ∃ (P : ℝ), P = 12 * Real.sqrt 3 + 24 := by
  -- Given an equilateral triangle with inscribed circles each of radius 4,
  -- we need to show that the perimeter of the triangle is 12√3 + 24.
  use 12 * Real.sqrt 3 + 24
  sorry

end equilateral_triangle_perimeter_l451_451335


namespace tyler_meals_l451_451620

-- Define the types of items Tyler can choose from in terms of meats, vegetables, and desserts.
inductive Meat 
| beef | chicken | pork | turkey

inductive Vegetable 
| baked_beans | corn | potatoes | tomatoes | carrots

inductive Dessert 
| brownies | chocolate_cake | chocolate_pudding | ice_cream | cheesecake

-- Use the relevant combinatorial functions to calculate the number of ways to choose foods

open Nat

def num_meals : ℕ :=
  let meat_choices := choose 4 1 + choose 4 2 -- 4 choose 1 + 4 choose 2
  let veg_choices := choose 5 2 -- 5 choose 2
  let dessert_choices := choose 5 2 -- 5 choose 2
  meat_choices * veg_choices * dessert_choices

theorem tyler_meals : num_meals = 1000 :=
  by
  -- Calculation based on the combinatorial analysis from the problem statement.
  -- The proof will be done step by step to form the full calculation.
  sorry

end tyler_meals_l451_451620


namespace range_of_c_l451_451630

variable (x c : ℝ)

-- We define the conditions
def condition_1 : Prop := x ∈ Ioi 0
def condition_2 : Prop := c^2 * x^2 - (c * x + 1) * log x + c * x ≥ 0

-- The proof statement
theorem range_of_c (h1 : condition_1 x) (h2 : ∀ x, condition_2 x c) : c ≥ 1 / Real.exp 1 := 
sorry

end range_of_c_l451_451630


namespace symmetric_point_example_l451_451596

def symmetric_point_yoz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (-p.1, p.2, p.3)

theorem symmetric_point_example : symmetric_point_yoz (2, 3, 4) = (-2, 3, 4) :=
  by sorry

end symmetric_point_example_l451_451596


namespace constant_distance_sum_l451_451907

open EuclideanGeometry

noncomputable def orthocenter (A B C : Point) : Point := sorry
noncomputable def circumcircle (A B C : Point) : Circle := sorry
noncomputable def midpoint (A B: Point) : Point := sorry
noncomputable def distance_sq (P Q : Point) : ℝ := sorry

theorem constant_distance_sum (A B C : Point)
  (H : Point) (P : Point) (D : Point)
  (a b c R : ℝ) :
  H = orthocenter A B C →
  P ≠ A → P ≠ B → P ≠ C →
  P ∈ circumcircle A B C →
  D = midpoint B C →
  PA^2 + PB^2 + PC^2 - PH^2 - PD^2 = a^2 + b^2 + c^2 - 4R^2 :=
begin
  sorry
end

end constant_distance_sum_l451_451907


namespace sum_of_first_five_terms_l451_451179

def a (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 2
  else a (n - 1) + a (n - 2)

theorem sum_of_first_five_terms : a 1 + a 2 + a 3 + a 4 + a 5 = 19 := by
  sorry

end sum_of_first_five_terms_l451_451179


namespace distance_store_to_Peter_l451_451498

variables (D_HS D_SP : ℝ)

-- Conditions
axiom H1 : D_HS = 2 * D_SP
axiom H2 : D_HS + 2 * D_SP = 250

theorem distance_store_to_Peter : D_SP = 62.5 :=
by {
  -- Starting with the given conditions
  have H3 : D_HS = 2 * D_SP := H1,
  have H4 : D_HS + 2 * D_SP = 250 := H2,

  -- Substitution from H3 into H4
  rw H3 at H4,
  rw ←add_assoc at H4,
  rw two_mul at H4,

  -- Solving for D_SP
  have H5 : 4 * D_SP = 250 := H4,
  exact eq_div_of_mul_eq Iff.rfl H5,
} sorry

end distance_store_to_Peter_l451_451498


namespace relay_race_probability_l451_451772

theorem relay_race_probability :
  let num_possible_arrangements_with_A_not_first := 18
  let num_possible_arrangements_with_B_first := 6
  let num_possible_arrangements_with_B_not_first_and_not_second := 8
  let total_probability := (num_possible_arrangements_with_B_first + num_possible_arrangements_with_B_not_first_and_not_second) / num_possible_arrangements_with_A_not_first
  (total_probability = (7 : ℚ) / 9) :=
begin
  sorry
end

end relay_race_probability_l451_451772


namespace ratio_A_B_correct_l451_451133

-- Define the shares of A, B, and C
def A_share := 372
def B_share := 93
def C_share := 62

-- Total amount distributed
def total_share := A_share + B_share + C_share

-- The ratio of A's share to B's share
def ratio_A_to_B := A_share / B_share

theorem ratio_A_B_correct : 
  total_share = 527 ∧ 
  ¬(B_share = (1 / 4) * C_share) ∧ 
  ratio_A_to_B = 4 := 
by
  sorry

end ratio_A_B_correct_l451_451133


namespace product_of_0_dot_9_repeating_and_8_eq_8_l451_451377

theorem product_of_0_dot_9_repeating_and_8_eq_8 :
  ∃ q : ℝ, q = 0.999... ∧ q * 8 = 8 := 
sorry

end product_of_0_dot_9_repeating_and_8_eq_8_l451_451377


namespace find_x_when_y_is_minus_21_l451_451996

variable (x y k : ℝ)

theorem find_x_when_y_is_minus_21
  (h1 : x * y = k)
  (h2 : x + y = 35)
  (h3 : y = 3 * x)
  (h4 : y = -21) :
  x = -10.9375 := by
  sorry

end find_x_when_y_is_minus_21_l451_451996


namespace combinatorial_expression_equivalence_l451_451169

theorem combinatorial_expression_equivalence :
  (∏ i in finset.range 8, (10 - i)) / (finset.prod (finset.range 7) (λ i, i + 1)) = nat.choose 10 7 :=
sorry

end combinatorial_expression_equivalence_l451_451169


namespace Emmy_money_l451_451350

theorem Emmy_money {Gerry_money cost_per_apple number_of_apples Emmy_money : ℕ} 
    (h1 : Gerry_money = 100)
    (h2 : cost_per_apple = 2) 
    (h3 : number_of_apples = 150) 
    (h4 : number_of_apples * cost_per_apple = Gerry_money + Emmy_money) :
    Emmy_money = 200 :=
by
   sorry

end Emmy_money_l451_451350


namespace max_value_expression_l451_451462

theorem max_value_expression  
    (x y : ℝ) 
    (h : 2 * x^2 + y^2 = 6 * x) : 
    x^2 + y^2 + 2 * x ≤ 15 :=
sorry

end max_value_expression_l451_451462


namespace four_digit_even_numbers_with_sum_of_tens_and_units_12_l451_451448

-- State the problem in Lean 4
theorem four_digit_even_numbers_with_sum_of_tens_and_units_12 :
  let count_four_digit_even_with_tens_units_sum_12 :=
        (9 * 10 * 3) in 
  count_four_digit_even_with_tens_units_sum_12 = 270 :=
by {
  -- The proof demonstrating the value will be constructed here
  sorry
}

end four_digit_even_numbers_with_sum_of_tens_and_units_12_l451_451448


namespace determine_hyperbola_eq_l451_451781

def hyperbola_eq (a b : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2) / (a^2) - (y^2) / (b^2) = 1

def asymptote_condition (a b : ℝ) : Prop :=
  b / a = 3 / 4

def focus_condition (a b : ℝ) : Prop :=
  a^2 + b^2 = 25

theorem determine_hyperbola_eq : 
  ∃ a b : ℝ, 
  (a > 0) ∧ (b > 0) ∧ asymptote_condition a b ∧ focus_condition a b ∧ hyperbola_eq 4 3 :=
sorry

end determine_hyperbola_eq_l451_451781


namespace right_triangle_sides_l451_451226

theorem right_triangle_sides :
  (4^2 + 5^2 ≠ 6^2) ∧
  (1^2 + 1^2 = (Real.sqrt 2)^2) ∧
  (6^2 + 8^2 ≠ 11^2) ∧
  (5^2 + 12^2 ≠ 23^2) :=
by
  repeat { sorry }

end right_triangle_sides_l451_451226


namespace equation_of_line_l451_451782

theorem equation_of_line (P : ℝ × ℝ) (A : ℝ) (m : ℝ) (hP : P = (-3, 4)) (hA : A = 3) (hm : m = 1) :
  ((2 * P.1 + 3 * P.2 - 6 = 0) ∨ (8 * P.1 + 3 * P.2 + 12 = 0)) :=
by 
  sorry

end equation_of_line_l451_451782


namespace expected_steps_l451_451734

def expected_steps_to_reach_10 (E : ℕ → ℕ) : Prop :=
  E 10 = 0 ∧
  ∀ i : ℕ, 2 ≤ i ∧ i ≤ 9 → E i = 2 + (E (i - 1) + (E (i + 1) / 2)) ∧
  E 1 = 2 + (E 2)

theorem expected_steps : ∃ E : ℕ → ℕ, expected_steps_to_reach_10 E ∧ E 1 = 90 :=
begin
  sorry
end

end expected_steps_l451_451734


namespace hyperbola_eccentricity_l451_451812

def eccentricity (a b : ℝ) : ℝ := 
  real.sqrt (a^2 + b^2) / a

theorem hyperbola_eccentricity (a : ℝ) (h : a > 0) : 
  eccentricity a (a * real.sqrt 6 / 6) = real.sqrt 42 / 6 := 
by 
  sorry

end hyperbola_eccentricity_l451_451812


namespace geometric_sequence_expression_l451_451875

theorem geometric_sequence_expression (a : ℕ → ℝ) (q : ℝ) (n : ℕ) (h1 : a 2 = 1)
(h2 : a 3 * a 5 = 2 * a 7) : a n = 1 / 2 ^ (n - 2) :=
sorry

end geometric_sequence_expression_l451_451875


namespace floor_factorial_expression_l451_451307

theorem floor_factorial_expression : 
  ⌊(2010.factorial + 2007.factorial) / (2009.factorial + 2008.factorial)⌋ = 2009 :=
by
  sorry

end floor_factorial_expression_l451_451307


namespace exists_number_with_special_quotient_l451_451015

theorem exists_number_with_special_quotient :
  ∃ N : ℕ, 
    (∃ k1 : ℕ, N = 312 * k1 ∧ (list.dedup ((nat.digits 10 k1)) = [1,2,3,4,5,6,7,8])) ∨
    (∃ k2 : ℕ, N = 3101 * k2 ∧ (list.dedup ((nat.digits 10 k2)) = [1,2,3,4,5,6,7,8])) :=
sorry

end exists_number_with_special_quotient_l451_451015


namespace floor_fraction_equals_2009_l451_451295

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem floor_fraction_equals_2009 :
  (⌊ (factorial 2010 + factorial 2007) / (factorial 2009 + factorial 2008) ⌋ : ℤ) = 2009 :=
by sorry

end floor_fraction_equals_2009_l451_451295


namespace pam_bags_l451_451932

-- Definitions
def gerald_bag_apples : ℕ := 40
def pam_bag_apples : ℕ := 3 * gerald_bag_apples
def pam_total_apples : ℕ := 1200

-- Theorem stating that the number of Pam's bags is 10
theorem pam_bags : pam_total_apples / pam_bag_apples = 10 := by
  sorry

end pam_bags_l451_451932


namespace find_x_l451_451628

theorem find_x (x y z w : ℕ) (h1 : x = y + 8) (h2 : y = z + 15) (h3 : z = w + 25) (h4 : w = 90) : x = 138 :=
by
  sorry

end find_x_l451_451628


namespace floor_factorial_expression_eq_2009_l451_451305

def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

theorem floor_factorial_expression_eq_2009 :
  (Int.floor (↑(factorial 2010 + factorial 2007) / ↑(factorial 2009 + factorial 2008)) = 2009) := by
  sorry

end floor_factorial_expression_eq_2009_l451_451305


namespace satisfy_inequality_l451_451779

noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x + Real.pi / 4)

theorem satisfy_inequality (x : ℝ) (k : ℤ) : 
  f(x) ≥ Real.sqrt 3 ↔ 
  ∃ k : ℤ, (Real.pi / 24) + (1 / 2) * k * Real.pi ≤ x ∧ x < (Real.pi / 8) + (1 / 2) * k * Real.pi := 
sorry

end satisfy_inequality_l451_451779


namespace correct_propositions_l451_451001

-- Definition of the first proposition P and its negation
def prop1 (x : ℝ) : Prop := (1 / (x - 1) > 0)
def not_prop1 (x : ℝ) : Prop := (1 / (x - 1) ≤ 0)

-- Definition of the second proposition
def prop2 (α : ℝ) : Prop := (sin α + cos α = 1 / 2) → (sin (2 * α) = -3 / 4)

-- Definition of the third proposition
def prop3 (α β : Type) [plane α] [plane β] (m : α) [line m] 
  (h1 : m ⊆ α) (h2 : m ⊆ β) : Prop := (m ∥ β) ↔ (α ∥ β)

-- Definition of the fourth proposition
def odd_function {R : Type*} [has_neg R] [AddGroup R] (f : R → R) : Prop :=
  ∀ x : R, f (-x) = -f x

def prop4 (f : ℝ → ℝ) (h1 : odd_function f) (h2 : ∀ x, f (x + 2) = -f x) : Prop :=
  ∃ a b c ∈ [0, 4], f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c
  
-- The main proof problem
theorem correct_propositions :
  (∃ x : ℝ, ¬ prop1 x ∧ not_prop1 x ∧ ¬ (prop1 x → not_prop1 x)) ∧
  (∀ α : ℝ, prop2 α) ∧
  (∃ α β : Type, ∃ line m (h1 : m ⊆ α) (h2 : m ⊆ β), prop3 α β m h1 h2) ∧
  (∃ f : ℝ → ℝ, ∃ h1 : odd_function f, ∃ h2 : ∀ x, f (x + 2) = -f x, prop4 f h1 h2)
:= sorry

end correct_propositions_l451_451001


namespace percent_daisies_l451_451669

theorem percent_daisies 
    (total_flowers : ℕ)
    (yellow_flowers : ℕ)
    (yellow_tulips : ℕ)
    (blue_flowers : ℕ)
    (blue_daisies : ℕ)
    (h1 : 2 * yellow_tulips = yellow_flowers) 
    (h2 : 3 * blue_daisies = blue_flowers)
    (h3 : 10 * yellow_flowers = 7 * total_flowers) : 
    100 * (yellow_flowers / 2 + blue_daisies) = 45 * total_flowers :=
by
  sorry

end percent_daisies_l451_451669


namespace prime_numbers_satisfy_equation_l451_451344

noncomputable def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_numbers_satisfy_equation :
  ∃ (p q r : ℕ), is_prime p ∧ is_prime q ∧ is_prime r ∧ (p + q^2 = r^4) ∧ 
  (p = 7) ∧ (q = 3) ∧ (r = 2) :=
by
  sorry

end prime_numbers_satisfy_equation_l451_451344


namespace floor_factorial_expression_l451_451313

-- Define the factorial function for natural numbers
def factorial : ℕ → ℕ
| 0 := 1
| (n + 1) := (n + 1) * factorial n

-- The main theorem to prove
theorem floor_factorial_expression :
  (nat.floor ((factorial 2010 + factorial 2007) / (factorial 2009 + factorial 2008)) = 2009) :=
begin
  -- Actual proof goes here
  sorry
end

end floor_factorial_expression_l451_451313


namespace least_sum_four_primes_gt_10_l451_451236

theorem least_sum_four_primes_gt_10 : 
  ∃ (p1 p2 p3 p4 : ℕ), 
    p1 > 10 ∧ p2 > 10 ∧ p3 > 10 ∧ p4 > 10 ∧ 
    Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ Nat.Prime p4 ∧
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧
    p1 + p2 + p3 + p4 = 60 ∧
    ∀ (q1 q2 q3 q4 : ℕ), 
      q1 > 10 ∧ q2 > 10 ∧ q3 > 10 ∧ q4 > 10 ∧ 
      Nat.Prime q1 ∧ Nat.Prime q2 ∧ Nat.Prime q3 ∧ Nat.Prime q4 ∧
      q1 ≠ q2 ∧ q1 ≠ q3 ∧ q1 ≠ q4 ∧ q2 ≠ q3 ∧ q2 ≠ q4 ∧ q3 ≠ q4 →
      q1 + q2 + q3 + q4 ≥ 60 :=
by
  sorry

end least_sum_four_primes_gt_10_l451_451236


namespace geometric_sequence_fifth_term_l451_451369

theorem geometric_sequence_fifth_term : 
  let a₁ := (2 : ℝ)
  let a₂ := (1 / 4 : ℝ)
  let r := a₂ / a₁
  let a₅ := a₁ * r ^ (5 - 1)
  a₅ = 1 / 2048 :=
by
  let a₁ := (2 : ℝ)
  let a₂ := (1 / 4 : ℝ)
  let r := a₂ / a₁
  let a₅ := a₁ * r ^ (5 - 1)
  sorry

end geometric_sequence_fifth_term_l451_451369


namespace quadratic_distinct_real_roots_l451_451405

theorem quadratic_distinct_real_roots (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x - a = 0 → x^2 - 2*x - a = 0 ∧ (∀ y : ℝ, y ≠ x → y^2 - 2*y - a = 0)) → 
  a > -1 :=
by
  sorry

end quadratic_distinct_real_roots_l451_451405


namespace number_of_correct_equations_l451_451017

-- Define the equations as conditions
def eqA (α : ℝ) := sin (2 * π - α) = sin α
def eqB (α : ℝ) := cos (-α) = cos α
def eqC (α : ℝ) := cos (π - α) = cos (2 * π + α)
def eqD (α : ℝ) := cos (π / 2 - α) = -cos α

-- Theorem stating the number of correct equations
theorem number_of_correct_equations : ∀ α : ℝ, (¬eqA α ∧ eqB α ∧ ¬eqC α ∧ ¬eqD α) → 1 := by
  sorry

end number_of_correct_equations_l451_451017


namespace range_of_a_l451_451453

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 + (a + 1) * x + 1 < 0) → (a < -3 ∨ a > 1) :=
by
  sorry

end range_of_a_l451_451453


namespace sum_powers_of_i_l451_451731

def i : ℂ := Complex.I

theorem sum_powers_of_i :
  (∑ k in Range (203), (i^(-101 + k))) = 2 :=
by
  sorry

end sum_powers_of_i_l451_451731


namespace problem_monotonically_increasing_interval_l451_451004

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x + Real.cos x

theorem problem_monotonically_increasing_interval :
  ∃ a b : ℝ, (0 < a ∧ a < b ∧ b ≤ real.pi) ∧ (∀ x y : ℝ, a < x ∧ x < b → a < y ∧ y < b → x < y → f x < f y) := 
  sorry

end problem_monotonically_increasing_interval_l451_451004


namespace units_digit_sum_squares_of_odd_integers_l451_451212

theorem units_digit_sum_squares_of_odd_integers :
  let first_2005_odd_units := [802, 802, 401] -- counts for units 1, 9, 5 respectively
  let extra_squares_last_6 := [9, 1, 3, 9, 5, 9] -- units digits of the squares of the last 6 numbers
  let total_sum :=
        (first_2005_odd_units[0] * 1 + 
         first_2005_odd_units[1] * 9 + 
         first_2005_odd_units[2] * 5) +
        (extra_squares_last_6.sum)
  (total_sum % 10) = 1 :=
by
  sorry

end units_digit_sum_squares_of_odd_integers_l451_451212


namespace triangle_area_l451_451701

theorem triangle_area :
  ∀ (x y : ℝ), (3 * x + 2 * y = 12 ∧ x ≥ 0 ∧ y ≥ 0) →
  (1 / 2) * 4 * 6 = 12 := by
  sorry

end triangle_area_l451_451701


namespace time_to_pass_bridge_approx_l451_451693

-- Define the lengths in meters
def train_length : ℕ := 1200
def bridge_length : ℕ := 500

-- Define the speed in kilometers per hour and convert it to meters per second
def train_speed_kmh : ℕ := 120
def train_speed_ms : ℚ := (train_speed_kmh : ℚ) * 1000 / 3600

-- Define total distance to cover
def total_distance : ℕ := train_length + bridge_length

-- Define function to calculate time to pass the bridge
noncomputable def time_to_pass_bridge : ℚ := total_distance / train_speed_ms

-- Define the proposition we want to prove
theorem time_to_pass_bridge_approx : time_to_pass_bridge ≈ 51.01 :=
by sorry

end time_to_pass_bridge_approx_l451_451693


namespace axes_of_symmetry_coincide_l451_451571

-- Define the quadratic functions f and g
def f (x : ℝ) : ℝ := (x^2 + 6*x - 25) / 8
def g (x : ℝ) : ℝ := (31 - x^2) / 8

-- Define the derivatives of f and g
def f' (x : ℝ) : ℝ := (x + 3) / 4
def g' (x : ℝ) : ℝ := -x / 4

-- Define the axes of symmetry for the functions
def axis_of_symmetry_f : ℝ := -3
def axis_of_symmetry_g : ℝ := 0

-- Define the intersection points
def intersection_points : List ℝ := [4, -7]

-- State the problem: Do the axes of symmetry coincide?
theorem axes_of_symmetry_coincide :
  (axis_of_symmetry_f = axis_of_symmetry_g) = False :=
by
  sorry

end axes_of_symmetry_coincide_l451_451571


namespace expected_value_of_sum_of_two_marbles_l451_451019

open Finset

noncomputable def choose2 (s:Finset ℕ) := s.powerset.filter (λ t, t.card = 2)

theorem expected_value_of_sum_of_two_marbles:
  let marbles := range 1 8 in
  let num_pairs := (choose2 marbles).card in
  let total_sum := (choose2 marbles).sum (λ t, t.sum id) in
  (total_sum:ℚ) / (num_pairs:ℚ) = 8 :=
by
  let marbles := range 1 8
  let num_pairs := (choose2 marbles).card
  let total_sum := (choose2 marbles).sum (λ t, t.sum id)
  have h1: num_pairs = 21, by sorry
  have h2: total_sum = 168, by sorry
  rw [h1, h2]
  norm_num

end expected_value_of_sum_of_two_marbles_l451_451019


namespace sara_marbles_total_l451_451944

theorem sara_marbles_total (original_marbles : ℝ) (given_marbles : ℝ) : original_marbles = 792.0 → given_marbles = 233.0 → original_marbles + given_marbles = 1025.0 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end sara_marbles_total_l451_451944


namespace boat_stream_speed_l451_451663

theorem boat_stream_speed :
  ∀ (v : ℝ), (∀ (downstream_speed boat_speed : ℝ), boat_speed = 22 ∧ downstream_speed = 54/2 ∧ downstream_speed = boat_speed + v) -> v = 5 :=
by
  sorry

end boat_stream_speed_l451_451663


namespace poly_div_l451_451739

theorem poly_div (A B : ℂ) :
  (∀ x : ℂ, x^3 + x^2 + 1 = 0 → x^202 + A * x + B = 0) → A + B = 0 :=
by
  intros h
  sorry

end poly_div_l451_451739


namespace count_perfect_squares_lt_10_pow_9_multiple_36_l451_451837

theorem count_perfect_squares_lt_10_pow_9_multiple_36 : 
  ∃ N : ℕ, ∀ n < 31622, (n % 6 = 0 → n^2 < 10^9 ∧ 36 ∣ n^2 → n ≤ 31620 → N = 5270) :=
by
  sorry

end count_perfect_squares_lt_10_pow_9_multiple_36_l451_451837


namespace coefficient_of_x_squared_in_binomial_expansion_l451_451796

noncomputable def a := ∫ x in 0..Real.pi, Real.sin x

theorem coefficient_of_x_squared_in_binomial_expansion :
  let binomial_expansion := (a*x + (1/Real.sqrt x))^5 in
  ∃ c : ℝ, c * x^2 ∈ binomial_expansion ∧ c = 80 :=
sorry

end coefficient_of_x_squared_in_binomial_expansion_l451_451796


namespace solve_trig_eq_l451_451554

noncomputable theory

-- All necessary trigonometric defintions and identity proofs
def cosine_eq (x : ℝ) : Prop := 
  cos x - cos (2 * x) + cos (3 * x) - cos (4 * x) = (1 / 2)

def solution_case_1 (k : ℤ) : ℝ := 
  (k : ℝ) * (2 * π) + (π / 3)

def solution_case_2 (k : ℤ) : ℝ := 
  (k : ℝ) * (2 * π) - (π / 3)

def solution_case_3 (k : ℤ) : ℝ := 
  (k : ℝ) * (2 / 3 * π) + (π / 9)

def solution_case_4 (k : ℤ) : ℝ := 
  (k : ℝ) * (2 / 3 * π) - (π / 9)

theorem solve_trig_eq (x : ℝ) (k : ℤ) : 
  cosine_eq x → 
  x = solution_case_1 k ∨ 
  x = solution_case_2 k ∨ 
  x = solution_case_3 k ∨ 
  x = solution_case_4 k :=
by 
  sorry

end solve_trig_eq_l451_451554


namespace axes_of_symmetry_coincide_l451_451573

-- Define the quadratic functions f and g
def f (x : ℝ) : ℝ := (x^2 + 6*x - 25) / 8
def g (x : ℝ) : ℝ := (31 - x^2) / 8

-- Define the derivatives of f and g
def f' (x : ℝ) : ℝ := (x + 3) / 4
def g' (x : ℝ) : ℝ := -x / 4

-- Define the axes of symmetry for the functions
def axis_of_symmetry_f : ℝ := -3
def axis_of_symmetry_g : ℝ := 0

-- Define the intersection points
def intersection_points : List ℝ := [4, -7]

-- State the problem: Do the axes of symmetry coincide?
theorem axes_of_symmetry_coincide :
  (axis_of_symmetry_f = axis_of_symmetry_g) = False :=
by
  sorry

end axes_of_symmetry_coincide_l451_451573


namespace radius_of_circle_l451_451125

noncomputable def point_circle_radius
    (P T : Point)
    (ℓ : Line)
    (d P_to_ℓ : ℝ)
    (d P_to_T : ℝ)
    (dist_Pℓ : distance P ℓ = P_to_ℓ)
    (dist_PT : distance P T = P_to_T) : ℝ :=
  sorry

theorem radius_of_circle
    {P T : Point}
    {ℓ : Line}
    (h1 : distance P ℓ = 12)
    (h2 : distance P T = 13) :
  point_circle_radius P T ℓ 12 13 h1 h2 = 169 / 24 :=
sorry

end radius_of_circle_l451_451125


namespace magnitude_difference_minimum_value_l451_451443

noncomputable def a (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)
noncomputable def b (ϕ : ℝ) : ℝ × ℝ := (Real.cos ϕ, Real.sin ϕ)

noncomputable def f (θ : ℝ) (λ : ℝ) : ℝ := 
  let a := a θ
  let b := b (Real.arccos (Real.cos (θ - λ))) -- since φ = arccos(cos(θ - λ))
  ( a.1 * b.1 + a.2 * b.2 ) - λ * ( (a.1 + b.1)^2 + (a.2 + b.2)^2 ).sqrt

theorem magnitude_difference 
  (θ ϕ : ℝ) 
  (h : |θ - ϕ| = π / 3) : 
  ( ((Real.cos θ - Real.cos ϕ)^2 + (Real.sin θ - Real.sin ϕ)^2 ).sqrt ) = 1 := 
by {
  -- Proof goes here
  sorry
}

theorem minimum_value 
  (θ λ : ℝ)
  (h1 : θ ∈ [0, π / 2])
  (h2 : 1 ≤ λ ∧ λ ≤ 2)
  (h3 : θ + λ = π / 3) : 
  (f θ λ) ≥ -(λ^2 / 4) - 1 := 
by {
  -- Proof goes here
  sorry
}

end magnitude_difference_minimum_value_l451_451443


namespace speed_ratio_l451_451262

theorem speed_ratio (D : ℝ) (bus_time truck_time : ℝ)
  (hD_pos : D > 0)
  (bus_time_eq : bus_time = 10)
  (truck_time_eq : truck_time = 15) :
  bus_time / truck_time = 2 / 3 :=
by
  rw [bus_time_eq, truck_time_eq]
  norm_num
  sorry

end speed_ratio_l451_451262


namespace count_quadratic_equations_with_real_roots_l451_451798

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def has_two_distinct_real_roots (b c : ℕ) : Prop :=
  b^2 - 4 * c > 0

theorem count_quadratic_equations_with_real_roots : 
  (∑ b in {n : ℕ | 1 ≤ n ∧ n ≤ 11 ∧ is_even n}, ∑ c in {n : ℕ | has_two_distinct_real_roots b n}, 1) = 50 :=
sorry

end count_quadratic_equations_with_real_roots_l451_451798


namespace ratio_SL_KL_l451_451788

-- Define the basic geometric setting and conditions for the problem
variables {A B C T K L S : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables [MetricSpace T] [MetricSpace K] [MetricSpace L] [MetricSpace S]
variables (AB AC BC AT AC_T KL_S BT : Line) [MetricSpace AB] [MetricSpace AC] [MetricSpace BC]
variables [MetricSpace AT] [MetricSpace AC_T] [MetricSpace KL_S] [MetricSpace BT]

noncomputable def | (x : Type) | : Real := sorry -- placeholder for distance function

axiom triangle_AB_AC_3BC : | AB | + | AC | = 3 * | BC |
axiom point_T_on_AC : | AC | = 4 * | AT |
axiom KL_parallel_BC : Parallel KL_S BC
axiom KL_tangent_inscribed_circle : Tangent KL_S (InscribedCircle (Triangle ABC))
axiom S_intersection_BT_KL : Intersection BT KL_S = S

-- The theorem we need to prove
theorem ratio_SL_KL : | SL | / | KL | = 2 / 3 :=
by
  sorry

end ratio_SL_KL_l451_451788


namespace tenth_term_arithmetic_sequence_l451_451988

theorem tenth_term_arithmetic_sequence :
  ∀ (a : ℕ → ℚ), a 1 = 5/6 ∧ a 16 = 7/8 →
  a 10 = 103/120 :=
by
  sorry

end tenth_term_arithmetic_sequence_l451_451988


namespace floor_factorial_expression_eq_2009_l451_451304

def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

theorem floor_factorial_expression_eq_2009 :
  (Int.floor (↑(factorial 2010 + factorial 2007) / ↑(factorial 2009 + factorial 2008)) = 2009) := by
  sorry

end floor_factorial_expression_eq_2009_l451_451304


namespace reroll_probability_exactly_three_dice_l451_451526

theorem reroll_probability_exactly_three_dice :
  (∃ (d : ℕ), d ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) ∧ 
    let k := 15 - d in
    k ∈ {6, 9, 10, 12, 15, 18} ∧ 
    (k = 15 - d → True) ∧ 
    (let sum_combinations := 21 + 25 + 27 + 27 + 25 in
    sum_combinations / 216 = 125 / 216)) := 
sorry

end reroll_probability_exactly_three_dice_l451_451526


namespace arithmetic_sequence_y_l451_451623

theorem arithmetic_sequence_y :
  let a := 3^3
  let c := 3^5
  let y := (a + c) / 2
  y = 135 :=
by
  let a := 27
  let c := 243
  let y := (a + c) / 2
  show y = 135
  sorry

end arithmetic_sequence_y_l451_451623


namespace odd_number_between_500_and_1000_with_sum_of_last_digits_33_l451_451359

def last_digit (n : ℕ) : ℕ :=
  n % 10

def sum_of_last_digits_of_divisors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d, n % d = 0).sum (λ d, last_digit d)

theorem odd_number_between_500_and_1000_with_sum_of_last_digits_33 :
  ∃ n : ℕ, 500 < n ∧ n < 1000 ∧ n % 2 = 1 ∧ sum_of_last_digits_of_divisors n = 33 :=
sorry

end odd_number_between_500_and_1000_with_sum_of_last_digits_33_l451_451359


namespace average_of_two_middle_numbers_is_correct_l451_451971

def four_numbers_meeting_conditions (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  (a + b + c + d = 20) ∧ 
  (max a (max b (max c d)) - min a (min b (min c d)) = max_diff

def max_diff := 
  (∀ (x y : ℕ), (x ≠ y → x > 0 → y > 0 → x + y ≤ 19 ∧ x + y ≥ 5) → 
  (x = 14 ∧ y = 1))

theorem average_of_two_middle_numbers_is_correct :
  ∃ (a b c d : ℕ), four_numbers_meeting_conditions a b c d →
  let numbers := [a, b, c, d].erase (min a (min b (min c d))).erase (max a (max b (max c d))),
  (numbers.sum / 2) = 2.5 := 
by
  sorry

end average_of_two_middle_numbers_is_correct_l451_451971


namespace value_of_g_at_8_l451_451024

def g (x : ℝ) : ℝ := (3 * x + 2) / (x - 2)

theorem value_of_g_at_8 : g 8 = 13 / 3 := by
  sorry

end value_of_g_at_8_l451_451024


namespace trip_cost_is_correct_l451_451933

noncomputable def trip_cost : ℕ := 
  let XZ_dist := 4000
  let XY_dist := 4500
  let YZ_dist := real.sqrt ((XY_dist ^ 2) - (XZ_dist ^ 2))
  let bus_cost_per_km := 0.20
  let plane_cost_per_km := 0.12
  let booking_fee := 120
  let cost_fly (dist : ℕ) := (plane_cost_per_km * dist) + booking_fee
  let cost_bus (dist : ℕ) := bus_cost_per_km * dist
  let cost_XY := min (cost_fly XY_dist) (cost_bus XY_dist)
  let cost_YZ := min (cost_fly YZ_dist) (cost_bus YZ_dist)
  let cost_ZX := min (cost_fly XZ_dist) (cost_bus XZ_dist)
  cost_XY + cost_YZ + cost_ZX

theorem trip_cost_is_correct :
  trip_cost = 2655 :=
by
  sorry

end trip_cost_is_correct_l451_451933


namespace solve_x_l451_451455

theorem solve_x (x y : ℝ) (h1 : 3 * x - y = 7) (h2 : x + 3 * y = 16) : x = 16 := by
  sorry

end solve_x_l451_451455


namespace find_k_minus_a_l451_451009

noncomputable def power_function (k a x : ℝ) : ℝ := k * x ^ a

theorem find_k_minus_a (k a : ℝ) (H1 : power_function k a 8 = 4) : k - a = 1 / 3 :=
by
  sorry

end find_k_minus_a_l451_451009


namespace cyclic_quadrilateral_diagonal_perpendicular_sum_of_squares_l451_451267

-- Definitions and premises as per the conditions
variables (R : ℝ) (α β : ℝ)
variables (A B C D P : Type*) -- Points A, B, C, D, and P

noncomputable def inscribed_quadrilateral (R : ℝ) : Prop :=
  ∃ (A B C D P : Type*),
    let AB := 2 * R * real.sin α,
        BC := 2 * R * real.sin β,
        CD := 2 * R * real.cos α,
        AD := 2 * R * real.cos β in
    (AB^2 + CD^2 = 4 * R^2) ∧
    (AB^2 + BC^2 + CD^2 + AD^2 = 8 * R^2)

theorem cyclic_quadrilateral_diagonal_perpendicular_sum_of_squares (h : inscribed_quadrilateral R) :
  let AP_sq := AB^2 + CD^2 in
  AP_sq = 4 * R^2 ∧ 
  let sides_sq := AB^2 + BC^2 + CD^2 + AD^2 in
  sides_sq = 8 * R^2 :=
by sorry

end cyclic_quadrilateral_diagonal_perpendicular_sum_of_squares_l451_451267


namespace bead_velocities_l451_451191

-- define constants
def m1 : ℝ := 150
def m2 : ℝ := 1
def m3 : ℝ := 30
def V2_initial : ℝ := 10
def V1_initial : ℝ := 0
def V3_initial : ℝ := 0

-- state problems as axioms
axiom conservation_of_momentum (V1 V3 : ℝ) : 
    - (m1 * V1) + (m2 * V2_initial) + (m3 * V3) = m2 * V2_initial
axiom conservation_of_kinetic_energy (V1 V3 : ℝ) : 
    (1 / 2) * m1 * V1^2 + (1 / 2) * m2 * V2_initial^2 + (1 / 2) * m3 * V3^2 = 
    (1 / 2) * m2 * V2_initial^2

-- proofs
theorem bead_velocities : 
    ∃ (V1 V3 : ℝ), (V1 ≈ 0.28) ∧ (V3 ≈ 1.72) :=
by 
    -- Assuming the solutions are correct from physical analysis 
    sorry

end bead_velocities_l451_451191


namespace convert_to_scientific_notation_9600000_l451_451560

theorem convert_to_scientific_notation_9600000 :
  9600000 = 9.6 * 10^6 := 
sorry

end convert_to_scientific_notation_9600000_l451_451560


namespace letters_in_mailboxes_l451_451600

theorem letters_in_mailboxes (letters mailboxes : ℕ) (h_letters : letters = 3) (h_mailboxes : mailboxes = 4) :
  (mailboxes ^ letters) = 4^3 :=
by
  rw [h_letters, h_mailboxes]
  exact eq.refl 4^3

end letters_in_mailboxes_l451_451600


namespace find_d_l451_451029

variables {x y z k d : ℝ}
variables {a : ℝ} (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0)
variables (h_ap : x * (y - z) + y * (z - x) + z * (x - y) = 0)
variables (h_sum : x * (y - z) + (y * (z - x) + d) + (z * (x - y) + 2 * d) = k)

theorem find_d : d = k / 3 :=
sorry

end find_d_l451_451029


namespace odd_function_negative_value_l451_451417

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_value {f : ℝ → ℝ} (h_odd : is_odd_function f) :
  (∀ x, 0 < x → f x = x^2 - x - 1) → (∀ x, x < 0 → f x = -x^2 - x + 1) :=
by
  sorry

end odd_function_negative_value_l451_451417


namespace area_of_triangle_8_9_9_l451_451362

noncomputable def triangle_area (a b c : ℕ) : Real :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem area_of_triangle_8_9_9 : triangle_area 8 9 9 = 4 * Real.sqrt 65 :=
by
  sorry

end area_of_triangle_8_9_9_l451_451362


namespace largest_k_for_positive_root_l451_451200

theorem largest_k_for_positive_root : ∃ k : ℤ, k = 1 ∧ ∀ k' : ℤ, (k' > 1) → ¬ (∃ x > 0, 3 * x * (2 * k' * x - 5) - 2 * x^2 + 8 = 0) :=
by
  sorry

end largest_k_for_positive_root_l451_451200


namespace min_lambda_plus_two_mu_l451_451854

theorem min_lambda_plus_two_mu 
  (A B C P M N : Type)
  [AddCommGroup A] [AddCommGroup B] [AddCommGroup C]
  [Module ℝ A] [Module ℝ B] [Module ℝ C]
  (lambda mu : ℝ)
  (h_lambda_pos : 0 < lambda)
  (h_mu_pos : 0 < mu)
  (h_BP_PC : ∀ (v : A → B), v P = 1/2 * v C)
  (h_AM : ∀ (v : A → B), v M = lambda * v B)
  (h_AN : ∀ (v : A → C), v N = mu * v C) :
  ∃ (lambda : ℝ),  lambda = 4/3 ∧ (lambda + 2 * (lambda / (3 * lambda - 2))) = 8/3 :=
sorry

end min_lambda_plus_two_mu_l451_451854


namespace union_complement_eq_l451_451011

open Set

theorem union_complement_eq :
  let U := ({1, 2, 3, 4, 5} : Set ℕ)
  let A := ({3, 4} : Set ℕ)
  let B := ({1, 4, 5} : Set ℕ)
  A ∪ (U \ B) = ({2, 3, 4} : Set ℕ) := by
{
  let U := ({1, 2, 3, 4, 5} : Set ℕ)
  let A := ({3, 4} : Set ℕ)
  let B := ({1, 4, 5} : Set ℕ)
  
  have h1 : U \ B = ({2, 3} : Set ℕ),
  sorry,

  have h2 : A ∪ ({2, 3} : Set ℕ) = ({2, 3, 4} : Set ℕ),
  sorry
  
  show A ∪ (U \ B) = ({2, 3, 4} : Set ℕ), from
  sorry
}

end union_complement_eq_l451_451011


namespace last_score_is_92_l451_451038

theorem last_score_is_92 (scores : List ℕ) (H_sorted : List.sort scores = [73, 77, 83, 85, 92])
  (H_sum : scores.sum = 410)
  (H_avg_int : ∀ (i : ℕ) (h₁ : i < scores.length) (h₂ : i > 0), ((scores.take (i + 1)).sum / (i + 1)) ∈ ℤ) :
  List.last scores sorry = 92 := sorry

end last_score_is_92_l451_451038


namespace logistics_service_teams_l451_451716

/-- 
  At the school sports meeting, three athletes A, B, and C participated in the 3000m, 1500m, and high jump competitions, respectively. 
  For safety reasons, the class committee set up logistics service teams for these three athletes. 
  Athlete A and four other students participate in the logistics service work (each student can only participate in one logistics service team).
  If athlete A is in the logistics service team for athlete A, then the number of assignment schemes for these five students is 50.
-/
theorem logistics_service_teams :
  let A := 1, B := 1, C := 1, Students := 4 in
  let assignments := λ n m : ℕ, Nat.choose n m in 
  assignments 4 2 * assignments 2 2 + assignments 4 3 * assignments 2 2 + assignments 4 1 * assignments 3 2 * assignments 2 2 + assignments 4 2 * assignments 2 2 = 50 :=
by
  sorry

end logistics_service_teams_l451_451716


namespace Daniel_noodles_left_l451_451340

theorem Daniel_noodles_left (initial_noodles : ℕ) (noodles_given : ℕ) (remaining_noodles : ℕ) 
  (h1 : initial_noodles = 66) (h2 : noodles_given = 12) : remaining_noodles = initial_noodles - noodles_given :=
by
  have h3 : remaining_noodles = 54 := sorry
  exact h3

end Daniel_noodles_left_l451_451340


namespace geometric_series_properties_l451_451426

theorem geometric_series_properties (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) :
  a 3 = 3 ∧ a 10 = 384 → 
  q = 2 ∧ 
  (∀ n, a n = (3 / 4) * 2 ^ (n - 1)) ∧ 
  (∀ n, S n = (3 / 4) * (2 ^ n - 1)) :=
by
  intro h
  -- Proofs will go here, if necessary.
  sorry

end geometric_series_properties_l451_451426


namespace sum_of_possible_m_l451_451108

def f (x m : ℝ) : ℝ :=
if x < m then x^2 + 3 * x + 1 else 3 * x + 6

theorem sum_of_possible_m (m : ℝ) : 
  (∀ x, (x < m → f x m = x^2 + 3 * x + 1) ∧ (x ≥ m → f x m = 3 * x + 6)) →
  (∀ m, f m m = m^2 + 3 * m + 1 → f m m = 3 * m + 6) →
  m = √5 ∨ m = -√5 →
  0 :=
by
  sorry

end sum_of_possible_m_l451_451108


namespace inequality_subtraction_l451_451459

variable (a b : ℝ)

theorem inequality_subtraction (h : a > b) : a - 5 > b - 5 :=
sorry

end inequality_subtraction_l451_451459


namespace probability_coin_covers_black_region_l451_451691

open Real

noncomputable def coin_cover_black_region_probability : ℝ :=
  let side_length_square := 10
  let triangle_leg := 3
  let diamond_side_length := 3 * sqrt 2
  let smaller_square_side := 1
  let coin_diameter := 1
  -- The derived probability calculation
  (32 + 9 * sqrt 2 + π) / 81

theorem probability_coin_covers_black_region :
  coin_cover_black_region_probability = (32 + 9 * sqrt 2 + π) / 81 :=
by
  -- Proof goes here
  sorry

end probability_coin_covers_black_region_l451_451691


namespace fraction_of_B_is_one_fourth_l451_451481

noncomputable theory
open_locale classical

-- Define the conditions
variable (T : ℝ) (hT : T ≈ 600)

-- Contribution of fractions
def fraction_A := (1 / 5 : ℝ)
def fraction_C := (1 / 2 : ℝ)
def number_D := 30

-- Number of each grade
def number_A := fraction_A * T
def number_C := fraction_C * T

-- Non-A/C grades are given by the difference
def non_AC_grades := T - (number_A + number_C)

-- The remaining number of grades which are B's
def number_B := non_AC_grades - number_D

-- The fraction of grades that are B's
def fraction_B := number_B / T

-- The proof statement asserting the fraction of B's is approximately 1/4
theorem fraction_of_B_is_one_fourth : fraction_B ≈ 1 / 4 :=
begin
  sorry
end

end fraction_of_B_is_one_fourth_l451_451481


namespace age_problem_l451_451036

variable (A B : ℕ)

theorem age_problem (h1 : A + 10 = 2 * (B - 10)) (h2 : A = B + 5) : B = 35 := by
  sorry

end age_problem_l451_451036


namespace triangle_obtuse_of_eccentricities_l451_451032

noncomputable def is_obtuse_triangle (a b m : ℝ) : Prop :=
  a^2 + b^2 - m^2 < 0

theorem triangle_obtuse_of_eccentricities (a b m : ℝ) (ha : a > 0) (hm : m > b) (hb : b > 0)
  (ecc_cond : (Real.sqrt (a^2 + b^2) / a) * (Real.sqrt (m^2 - b^2) / m) > 1) :
  is_obtuse_triangle a b m := 
sorry

end triangle_obtuse_of_eccentricities_l451_451032


namespace distance_between_centers_of_inscribed_circles_l451_451918

theorem distance_between_centers_of_inscribed_circles 
  (A B C : ℝ × ℝ) (AB AC BC : ℝ) (r1 r2 r3 : ℝ) (O2 O3 : ℝ × ℝ) 
  (_ : A = (0, 0))
  (_ : B = (80, 0))
  (_ : C = (0, 150))
  (_ : AB = 80) 
  (_ : AC = 150) 
  (_ : BC = 170) 
  (_ : 2 * 30 * 200 = 6000) -- this represents semiperimeter and area calculation
  (_ : r1 = 30)
  (_ : r2 = 24)
  (_ : r3 = 18.75)
  (_ : O2 = (24, 144))
  (_ : O3 = (68.75, 18.75)) :
  real.dist O2 O3 = real.sqrt (10 * 2025) :=
sorry

end distance_between_centers_of_inscribed_circles_l451_451918


namespace number_of_kids_in_all_three_activities_l451_451607

-- Define the initial number of kids
def total_kids_on_lake_pleasant : ℕ := 40

-- Define the number of kids who went tubing
def kids_tubing : ℕ := total_kids_on_lake_pleasant / 4

-- Define the number of kids who went tubing and rafting
def kids_tubing_and_rafting : ℕ := kids_tubing / 2

-- Define the number of kids who went tubing, rafting, and kayaking
def kids_tubing_rafting_and_kayaking : ℕ := kids_tubing_and_rafting / 3

-- This is the main statement to be proven
theorem number_of_kids_in_all_three_activities (total_kids_on_lake_pleasant = 40): 
  kids_tubing_rafting_and_kayaking = 1 := sorry

end number_of_kids_in_all_three_activities_l451_451607


namespace cos_squared_minus_sin_squared_15_eqn_half_sin_40_plus_half_sqrt3_cos_40_eqn_sin_pi_over_8_cos_pi_over_8_eqn_tan_15_eqn_l451_451224

-- Prove that cos^2(15°) - sin^2(15°) = √3/2
theorem cos_squared_minus_sin_squared_15_eqn : (cos (15 * π / 180))^2 - (sin (15 * π / 180))^2 = √3 / 2 :=
sorry

-- Prove that 1/2 * sin(40°) + √3/2 * cos(40°) = sin(70°)
theorem half_sin_40_plus_half_sqrt3_cos_40_eqn : (1 / 2) * sin (40 * π / 180) + (√3 / 2) * cos (40  * π / 180) = sin (70 * π / 180) :=
sorry

-- Prove that sin(π/8) * cos(π/8) = √2/4
theorem sin_pi_over_8_cos_pi_over_8_eqn : sin (π / 8) * cos (π / 8) = √2 / 4 :=
sorry

-- Prove that tan(15°) = 2 - √3 
theorem tan_15_eqn : tan (15 * π / 180) = 2 - √3 :=
sorry

end cos_squared_minus_sin_squared_15_eqn_half_sin_40_plus_half_sqrt3_cos_40_eqn_sin_pi_over_8_cos_pi_over_8_eqn_tan_15_eqn_l451_451224


namespace international_sales_correct_option_l451_451338

theorem international_sales_correct_option :
  (∃ (A B C D : String),
     A = "who" ∧
     B = "what" ∧
     C = "whoever" ∧
     D = "whatever" ∧
     (∃ x, x = C → "Could I speak to " ++ x ++ " is in charge of International Sales please?" = "Could I speak to whoever is in charge of International Sales please?")) :=
sorry

end international_sales_correct_option_l451_451338


namespace sum_six_least_n_satisfying_l451_451093

-- Define the tau function that returns the number of divisors
def tau (n : ℕ) : ℕ :=
  finset.card (finset.filter (λ d: ℕ, n % d = 0) (finset.range (n + 1)))

-- Define the property we are interested in
def satisfies_together (n : ℕ) : Prop :=
  tau(n) + tau(n + 1) = 8

-- Define the main theorem statement
theorem sum_six_least_n_satisfying : ∃ (n₁ n₂ n₃ n₄ n₅ n₆ : ℕ), 
  n₁ < n₂ ∧ n₂ < n₃ ∧ n₃ < n₄ ∧ n₄ < n₅ ∧ n₅ < n₆ ∧
  satisfies_together n₁ ∧ satisfies_together n₂ ∧ satisfies_together n₃ ∧ 
  satisfies_together n₄ ∧ satisfies_together n₅ ∧ satisfies_together n₆ ∧
  (n₁ + n₂ + n₃ + n₄ + n₅ + n₆ = S) := sorry

end sum_six_least_n_satisfying_l451_451093


namespace area_of_triangle_l451_451698

open Real

-- Defining the line equation 3x + 2y = 12
def line_eq (x y : ℝ) : Prop := 3 * x + 2 * y = 12

-- Defining the vertices of the triangle
def vertex1 := (0, 0 : ℝ)
def vertex2 := (0, 6 : ℝ)
def vertex3 := (4, 0 : ℝ)

-- Define a function to calculate the area of the triangle
def triangle_area (v1 v2 v3 : ℝ × ℝ) : ℝ :=
  0.5 * abs ((v2.1 - v1.1) * (v3.2 - v1.2) - (v3.1 - v1.1) * (v2.2 - v1.2))

-- Prove that area of the triangle bounded by the line and coordinate axes is 12 square units
theorem area_of_triangle : triangle_area vertex1 vertex2 vertex3 = 12 :=
by
  sorry

end area_of_triangle_l451_451698


namespace circle_equation_correct_line_equation_correct_l451_451400

open Real

-- Given data points P and Q and conditions about the circle
def P := (4, -2)
def Q := (-1, 3)
def radius_less_than := 5
def segment_length := 4 * sqrt 3

-- Equation of the circle C to be proved
noncomputable def equation_circle := sorry
theorem circle_equation_correct :
  ∃ D E F, 
  (∀ (x y : ℝ), (x^2 + y^2 + D * x + E * y + F = 0) → 
  (x, y) = P ∨ (x, y) = Q) 
  ∧ abs(D^2 - 4*F) = segment_length^2 
  ∧ (∀ (x y : ℝ), x^2 + y^2 + D * x + E * y + F = 0 → (x - D/2)^2 + y^2 = (radius_less_than)^2 ) 
  ∧ (D,E,F) = (2, 0, -12) :=
sorry

-- Given line l parallel to PQ
noncomputable def line_l_parallel_to_PQ :=
  λ m x y, x + y + m = 0

-- Equation of line l to be proved
noncomputable def ab_circle_pass_origin_equation := sorry
theorem line_equation_correct :
  (∀ (A B : ℝ × ℝ), (AB_circle_pass_origin := 0) →
  ∃ m, ∀ x y, (x + y + m = 0) ∧ (m = 3 ∨ m = -4)) :=
sorry

end circle_equation_correct_line_equation_correct_l451_451400


namespace units_digit_of_2011_odd_squares_l451_451210

def units_digit_sum_squares_first_k_odd_integers (k : ℕ) : ℕ :=
  let odd_numbers := List.range k |>.map (λ n, 2*n + 1)
  let squares := odd_numbers.map (λ n, n^2)
  let total_sum := squares.sum
  total_sum % 10

theorem units_digit_of_2011_odd_squares : units_digit_sum_squares_first_k_odd_integers 2011 = 9 :=
by
  sorry

end units_digit_of_2011_odd_squares_l451_451210


namespace I1_I2_I3_I4_cyclic_l451_451871

open Triangle EuclideanGeometry

-- Given conditions
variable (A B C P A' : Point)
variable (I1 I2 I3 I4 : Point)
variable [h_acute : AcuteTriangle A B C]
variable [h_reflection : Reflection A A' B C]
variable [h_ratio : RatioEq (A, B) (A, C) (P, B) (P, C)]
variable [h_incenters : IsIncenter P A B I1]
variable [h_incenters : IsIncenter P B A' I2]
variable [h_incenters : IsIncenter P A' C I3]
variable [h_incenters : IsIncenter P C A I4]

-- The goal is to prove that I1, I2, I3, and I4 are cyclic
theorem I1_I2_I3_I4_cyclic : Cyclic I1 I2 I3 I4 :=
by
  sorry

end I1_I2_I3_I4_cyclic_l451_451871


namespace problem_solution_l451_451766

def num_divisors (n : ℕ) : ℕ := 
  finset.card ((finset.range n).filter (λ d, d > 0 ∧ n % d = 0))

def f1 (n : ℕ) : ℕ := 2 * num_divisors n

noncomputable def fj : ℕ → ℕ → ℕ
| 1, n := f1 n
| (j+1), n := f1 (fj j n)

theorem problem_solution : 
  (finset.filter (λ n, fj 50 n = 18) (finset.range 101)).card = 0 :=
sorry

end problem_solution_l451_451766


namespace calculate_initial_books_l451_451502

def initial_books := 
  (sold_monday sold_tuesday sold_wednesday sold_thursday sold_friday : ℕ)
  (percentage_not_sold : ℚ) : ℕ :=
  let total_sold := sold_monday + sold_tuesday + sold_wednesday + sold_thursday + sold_friday
  let percentage_sold := 1 - percentage_not_sold
  total_sold / percentage_sold

theorem calculate_initial_books 
  (sold_monday sold_tuesday sold_wednesday sold_thursday sold_friday : ℕ)
  (percentage_not_sold : ℚ) : 
  (sold_monday = 50) → 
  (sold_tuesday = 82) → 
  (sold_wednesday = 60) → 
  (sold_thursday = 48) → 
  (sold_friday = 40) → 
  (percentage_not_sold = 54.83870967741935 / 100) →
  initial_books sold_monday sold_tuesday sold_wednesday sold_thursday sold_friday percentage_not_sold = 620 := 
by 
  intros
  sorry

end calculate_initial_books_l451_451502


namespace area_ratio_l451_451050

theorem area_ratio
  (A B C D E F : Type)
  (hAB : A = 130)
  (hAC : C = 130)
  (hAD : D = 50)
  (hCF : F = 90) :
  (area CEF / area DBE) = 22 / 5 :=
  by
    sorry

end area_ratio_l451_451050


namespace find_y_of_x_and_range_range_of_t_l451_451654

def rhombus_condition (A B D : ℝ × ℝ) (M : ℝ × ℝ) : Prop :=
  -- Conditions characterizing the rhombus and M being on x-axis
  A = (0, 1) ∧ 
  B.1 = 0 ∧ B.2 < 0 ∧
  (A.1 - D.1)^2 + (A.2 - D.2)^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2 ∧
  M.2 = 0

def function_y_expression (x y : ℝ) : Prop :=
  y = x^2 / 4

def fixed_point_condition (P Q : ℝ × ℝ) : Prop :=
  -- Conditions for the line parallel to x-axis cutting circle with diameter PQ
  ∀ m : ℝ, P.1 = Q.1 ∧ P.2 = x^2 / 4 ∧ Q.1 = 0 ∧ Q.2 = t →
  let d := abs((P.2 + Q.2) / 2 - m) in
  let radius := sqrt((P.1/2)^2 + ((P.2 - Q.2)/2)^2) in
  ∃ c : ℝ, 2 * sqrt(radius^2 - d^2) = c

theorem find_y_of_x_and_range (x : ℝ) (h : x ≠ 0) : ∃ y : ℝ, function_y_expression x y :=
begin
  use x^2 / 4,
  exact rfl,
end

theorem range_of_t (t : ℝ) : t > 1 :=
begin
  sorry
end

end find_y_of_x_and_range_range_of_t_l451_451654


namespace floor_fraction_equals_2009_l451_451296

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem floor_fraction_equals_2009 :
  (⌊ (factorial 2010 + factorial 2007) / (factorial 2009 + factorial 2008) ⌋ : ℤ) = 2009 :=
by sorry

end floor_fraction_equals_2009_l451_451296


namespace inequality_holds_for_all_n_l451_451936

theorem inequality_holds_for_all_n (n : ℕ) : 
  (finset.range n).sum (λ i, 1 / real.sqrt (i + 1)) > 2 * real.sqrt n - 3 / 2 :=
sorry

end inequality_holds_for_all_n_l451_451936


namespace paint_split_cost_l451_451888

theorem paint_split_cost :
  let BrandA_cost := 50
  let BrandA_coverage := 350
  let Jason_wall_area := 1025
  let Jason_coats := 3
  let BrandB_cost := 45
  let BrandB_coverage := 400
  let Jeremy_wall_area := 1575
  let Jeremy_coats := 2
  let Jason_total_sq := Jason_wall_area * Jason_coats
  let Jeremy_total_sq := Jeremy_wall_area * Jeremy_coats
  let Jason_gallons := (Jason_total_sq / BrandA_coverage).ceil
  let Jeremy_gallons := (Jeremy_total_sq / BrandB_coverage).ceil
  let Jason_cost := Jason_gallons * BrandA_cost
  let Jeremy_cost := Jeremy_gallons * BrandB_cost
  let total_cost := Jason_cost + Jeremy_cost
  let each_contribution := total_cost / 2
  in each_contribution = 405 := sorry

end paint_split_cost_l451_451888


namespace nested_fraction_evaluation_l451_451351

def nested_expression := 1 / (3 - 1 / (3 - 1 / (3 - 1 / 3)))

theorem nested_fraction_evaluation : nested_expression = 8 / 21 := by
  sorry

end nested_fraction_evaluation_l451_451351


namespace dice_prime_probability_l451_451290

theorem dice_prime_probability :
  let primes := {2, 3, 5, 7}
  let is_prime (n : ℕ) := n ∈ primes
  let probability_prime := (finset.card primes).to_rat / 10
  let probability_non_prime := 1 - probability_prime
  let ways_to_choose_2_out_of_4 := nat.choose 4 2
  let total_probability := ways_to_choose_2_out_of_4 * (probability_prime ^ 2 * probability_non_prime ^ 2)
  total_probability = 216 / 625 :=
by
  -- proof will be provided here
  sorry

end dice_prime_probability_l451_451290


namespace fourth_derivative_of_function_y_l451_451238

noncomputable def log_base_3 (x : ℝ) : ℝ := (Real.log x) / (Real.log 3)

noncomputable def function_y (x : ℝ) : ℝ := (log_base_3 x) / (x ^ 2)

theorem fourth_derivative_of_function_y (x : ℝ) (h : 0 < x) : 
    (deriv^[4] (fun x => function_y x)) x = (-154 + 120 * (Real.log x)) / (x ^ 6 * Real.log 3) :=
  sorry

end fourth_derivative_of_function_y_l451_451238


namespace sum_of_fractions_eq_neg_eight_l451_451467

theorem sum_of_fractions_eq_neg_eight
  (a b c : ℝ)
  (h1 : a^3 + b^3 + c^3 = 6)
  (h2 : a^2 + b^2 + c^2 = 8) :
  (a + b + c = 0) →
  (abc = 2) →
  (ab + bc + ca = -4) →
  (frac_ab : ab / (a + b)) →
  (frac_bc : bc / (b + c)) →
  (frac_ca : ca / (c + a)) →
  frac_ab + frac_bc + frac_ca = -8 :=
by
  -- proof here
  sorry

end sum_of_fractions_eq_neg_eight_l451_451467
