import Mathlib

namespace additional_miles_needed_l3_3971

-- Definitions for the given conditions
def initial_distance := 20
def initial_speed := 40
def additional_speed := 60
def desired_avg_speed := 45

-- Prove that given the conditions, the additional miles needed is 50
theorem additional_miles_needed :
  let t := (desired_avg_speed * (initial_distance / initial_speed + t) - initial_distance) / additional_speed = 5 / 6 in
  additional_speed * (5 / 6) = 50 :=
by
  sorry

end additional_miles_needed_l3_3971


namespace vector_parallel_find_k_l3_3554

theorem vector_parallel_find_k (k : ℝ) (a : ℝ × ℝ) (b : ℝ × ℝ) 
  (h₁ : a = (3 * k + 1, 2)) 
  (h₂ : b = (k, 1)) 
  (h₃ : ∃ c : ℝ, a = c • b) : k = -1 := 
by 
  sorry

end vector_parallel_find_k_l3_3554


namespace _l3_3641

-- Definitions
def diagonal : ℝ := 50
def aspect_ratio_width : ℝ := 16
def aspect_ratio_height : ℝ := 9

-- Conditions
def aspect_ratio_condition (w h : ℝ) : Prop := (w / h) = (aspect_ratio_width / aspect_ratio_height)
def pythagorean_theorem (w h d : ℝ) : Prop := (w^2 + h^2 = d^2)

-- Target lemma
lemma horizontal_length_of_tv (w h : ℝ) 
    (d_eq : diagonal = 50)
    (ar_cond : aspect_ratio_condition w h)
    (pt_cond : pythagorean_theorem w h diagonal) :
    w ≈ 43.6 :=
by
  -- Proper proofs come here
  sorry

end _l3_3641


namespace sum_of_reciprocals_l3_3143

noncomputable theory

open Polynomial

variables {R : Type*} [CommRing R] [IsDomain R]

theorem sum_of_reciprocals (a b c : R) (h : (X ^ 3 - X - (2 : R)).splits (ring_hom.id R))
  (h_roots : (X ^ 3 - X - (2 : R)).roots = [a, b, c]) :
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) = 2) :=
sorry

end sum_of_reciprocals_l3_3143


namespace coefficient_x_squared_in_expansion_l3_3041

theorem coefficient_x_squared_in_expansion :
  let p := (x^2 - 3*x + 2)
  in ∑ (m : ℕ) in antidiagonal 4, binomial 4 m.fst * C(m.snd + 2, 2) - 3 * C(m.snd, 1) + 2 * C(m.snd, 0) = 248 sorry

end coefficient_x_squared_in_expansion_l3_3041


namespace bug_problem_l3_3248

def P : ℕ → ℚ
| 0     := 1
| 2     := 0
| (n + 2) := 1 / 3 * (1 - P n)

theorem bug_problem 
: (P 10 = 20 / 81) → ∃ n : ℕ, n = 182 ∧ (n / 2187) = 20 / 81 :=
by
  assume h : P 10 = 20 / 81
  apply Exists.intro 182
  split
  {
    rfl
  }
  {
    rw [← h]
    simp
  }

end bug_problem_l3_3248


namespace a_n_add_a_n1_add_a_n2_find_a_2001_l3_3873

noncomputable def a : ℕ → ℤ 
| 1   := 5
| 5   := 8
| n   := sorry   -- This will be defined recursively in proofs, to depend on a_n + a_{n+1} + a_{n+2} = 7

theorem a_n_add_a_n1_add_a_n2 (n : ℕ) : a n + a (n+1) + a (n+2) = 7 :=
sorry

theorem find_a_2001 : a 2001 = -6 :=
sorry

end a_n_add_a_n1_add_a_n2_find_a_2001_l3_3873


namespace trains_cross_time_l3_3013

theorem trains_cross_time (speed_A_kmh : ℝ) (time_A_cross_pole : ℝ) (speed_B_kmh : ℝ) 
  (length_B_ratio : ℝ) (expected_time: ℝ) : 
  speed_A_kmh = 30 → 
  time_A_cross_pole = 24 → 
  speed_B_kmh = 40 → 
  length_B_ratio = 1.5 → 
  expected_time ≈ 25.72 → 
  let speed_A_ms := speed_A_kmh * (1000 / 3600) in
  let length_A := speed_A_ms * time_A_cross_pole in
  let length_B := length_B_ratio * length_A in
  let speed_B_ms := speed_B_kmh * (1000 / 3600) in
  let relative_speed := speed_A_ms + speed_B_ms in
  let total_length := length_A + length_B in
  total_length / relative_speed ≈ expected_time := 
by 
  intros hsA htAt hsb hr hb;
  sorry

end trains_cross_time_l3_3013


namespace anton_wins_infinitely_often_l3_3076

theorem anton_wins_infinitely_often :
  ∃ᶠ (n : ℕ) in (𝓟 (λ n, ∃ k : ℕ, n = 3 * k ^ 2 - 1)), ∃ m : ℕ, m ^ 2 = 6 * (n + 1) + 1 := 
sorry

end anton_wins_infinitely_often_l3_3076


namespace sphere_equation_l3_3672

noncomputable def sphere_center (a b c : ℝ) : ℝ × ℝ × ℝ := (a, b, c)
noncomputable def sphere_radius (R : ℝ) : ℝ := R

theorem sphere_equation {a b c R x y z : ℝ} :
  let Q := sphere_center a b c in
  let r_squared := (x - a) * (x - a) + (y - b) * (y - b) + (z - c) * (z - c) in
  r_squared = R^2 -> 
  (x - a)^2 + (y - b)^2 + (z - c)^2 = R^2 :=
by 
  intros
  sorry

end sphere_equation_l3_3672


namespace floor_painting_cost_l3_3213

noncomputable def floor_painting_problem : Prop := 
  ∃ (B L₁ L₂ B₂ Area₁ Area₂ CombinedCost : ℝ),
  L₁ = 2 * B ∧
  Area₁ = L₁ * B ∧
  484 = Area₁ * 3 ∧
  L₂ = 0.8 * L₁ ∧
  B₂ = 1.3 * B ∧
  Area₂ = L₂ * B₂ ∧
  CombinedCost = 484 + (Area₂ * 5) ∧
  CombinedCost = 1320.8

theorem floor_painting_cost : floor_painting_problem :=
by
  sorry

end floor_painting_cost_l3_3213


namespace f_n_2_l3_3831

def f (m n : ℕ) : ℝ :=
if h : m = 1 ∧ n = 1 then 1 else
if h : n > m then 0 else 
sorry -- This would be calculated based on the recursive definition

lemma f_2_2 : f 2 2 = 2 :=
sorry

theorem f_n_2 (n : ℕ) (hn : n ≥ 1) : f n 2 = 2^(n - 1) :=
sorry

end f_n_2_l3_3831


namespace jonah_first_intermission_lemonade_l3_3474

theorem jonah_first_intermission_lemonade :
  ∀ (l1 l2 l3 l_total : ℝ)
  (h1 : l2 = 0.42)
  (h2 : l3 = 0.25)
  (h3 : l_total = 0.92)
  (h4 : l_total = l1 + l2 + l3),
  l1 = 0.25 :=
by sorry

end jonah_first_intermission_lemonade_l3_3474


namespace part1_part2_l3_3912

-- Part 1 Definition
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * Real.log (x + 1) + 1 / (x + 1) + 3 * x - 1

-- Part 1: Formalizing the problem statement
theorem part1 (a : ℝ) : (∀ x ≥ 0, f a x ≥ 0) → (a ≥ -2) :=
sorry

-- Part 2: Formalizing the inequality
theorem part2 (n : ℕ) (h : n > 0) :
  ∑ k in Finset.range n, (k + 1 : ℝ) / (4 * (k + 1)^2 - 1) > (1 / 4) * Real.log (2 * n + 1) :=
sorry

end part1_part2_l3_3912


namespace sqrt_inequality_l3_3949

-- Defining the polyline distance between two points A and B
def polyline_distance (x1 y1 x2 y2 : ℝ) : ℝ := abs (x1 - x2) + abs (y1 - y2)

-- Given points O(0, 0) and C(x, y)
variables (x y : ℝ)

-- Given condition: d(O, C) = 1, i.e., |x| + |y| = 1
def distance_condition : Prop := abs x + abs y = 1

-- The inequality we need to prove: sqrt(x^2 + y^2) ≥ √2 / 2
theorem sqrt_inequality (h : distance_condition x y) : sqrt (x^2 + y^2) ≥ sqrt 2 / 2 :=
by
  admit  -- The placeholder for the actual proof

end sqrt_inequality_l3_3949


namespace cups_from_two_tablespoons_l3_3394

-- Defining the conditions as noncomputable (to skip the proof complexities)
noncomputable def popcornConditions := and.intro
  (3 + 4 + 6 + 3 = 16) -- Total cups wanted
  (16 / 8 = 2)         -- Cups per tablespoon

theorem cups_from_two_tablespoons :
  (2 * 2 = 4) :=
by
  sorry

end cups_from_two_tablespoons_l3_3394


namespace correct_equation_l3_3389

variable (x : ℤ)
variable (cost_of_chickens : ℤ)

-- Condition 1: If each person contributes 9 coins, there will be an excess of 11 coins.
def condition1 : Prop := 9 * x - cost_of_chickens = 11

-- Condition 2: If each person contributes 6 coins, there will be a shortage of 16 coins.
def condition2 : Prop := 6 * x - cost_of_chickens = -16

-- The goal is to prove the correct equation given the conditions.
theorem correct_equation (h1 : condition1 (x) (cost_of_chickens)) (h2 : condition2 (x) (cost_of_chickens)) :
  9 * x - 11 = 6 * x + 16 :=
sorry

end correct_equation_l3_3389


namespace length_AP_right_angle_l3_3229

-- Definitions based on the problem conditions
def A' : ℝ × ℝ × ℝ := (0, 0, 0)
def B' : ℝ × ℝ × ℝ := (12 * sqrt 3, 0, 0)
def P (z : ℝ) : ℝ × ℝ × ℝ := (0, 0, z)
def N : ℝ × ℝ × ℝ := (9 * sqrt 3, 0, 0)
def M (y : ℝ) : ℝ × ℝ × ℝ := (12 * sqrt 3, y, 18)

-- Vector calculations
def vector_mn (y : ℝ) : ℝ × ℝ × ℝ := (3 * sqrt 3, y, 18)
def vector_np (z : ℝ) : ℝ × ℝ × ℝ := (-9 * sqrt 3, 0, z)

-- Dot product condition for right angle
def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- Lean statement to be proved
theorem length_AP_right_angle (z : ℝ) :
  (∃ y : ℝ, dot_product (vector_mn y) (vector_np z) = 0) → z = 27 / 2 :=
by {
  sorry
}

end length_AP_right_angle_l3_3229


namespace division_by_negative_divisor_l3_3084

theorem division_by_negative_divisor : 15 / (-3) = -5 :=
by sorry

end division_by_negative_divisor_l3_3084


namespace additional_oil_needed_l3_3140

def car_cylinders := 6
def car_oil_per_cylinder := 8
def truck_cylinders := 8
def truck_oil_per_cylinder := 10
def motorcycle_cylinders := 4
def motorcycle_oil_per_cylinder := 6

def initial_car_oil := 16
def initial_truck_oil := 20
def initial_motorcycle_oil := 8

theorem additional_oil_needed :
  let car_total_oil := car_cylinders * car_oil_per_cylinder
  let truck_total_oil := truck_cylinders * truck_oil_per_cylinder
  let motorcycle_total_oil := motorcycle_cylinders * motorcycle_oil_per_cylinder
  let car_additional_oil := car_total_oil - initial_car_oil
  let truck_additional_oil := truck_total_oil - initial_truck_oil
  let motorcycle_additional_oil := motorcycle_total_oil - initial_motorcycle_oil
  car_additional_oil = 32 ∧
  truck_additional_oil = 60 ∧
  motorcycle_additional_oil = 16 :=
by
  repeat (exact sorry)

end additional_oil_needed_l3_3140


namespace sqrt_a_sqrt_a_eq_a_rational_exponent_l3_3142

theorem sqrt_a_sqrt_a_eq_a_rational_exponent (a : ℝ) (h : 0 < a) : (sqrt (a * sqrt a)) = a^(3/4) :=
by sorry

end sqrt_a_sqrt_a_eq_a_rational_exponent_l3_3142


namespace minimum_cans_needed_l3_3419

/-- Each can holds 15 ounces, and we need at least 192 ounces of soda.
    Prove that the minimum number of cans needed is 13. -/
theorem minimum_cans_needed : ∃ n : ℕ, 15 * n ≥ 192 ∧ (∀ m : ℕ, 15 * m ≥ 192 → m ≥ n) := 
by
  use 13
  split
  -- Here we verify the sufficiency of 13 cans:
  obviously (15 * 13 = 195)
  apply nat.find_min
  -- Prove that no smaller number of cans would be sufficient:
  imply obviously (15 * n < 192).imply obviously (n < 13)
  sorry -- actual proofs would go here.

end minimum_cans_needed_l3_3419


namespace opposite_face_of_A_is_E_l3_3420

-- Definitions for the conditions
def joined_in_sequence (squares : List Char) : Prop :=
  squares = ['A', 'B', 'C', 'D', 'E', 'F']

-- The statement of the problem
theorem opposite_face_of_A_is_E 
  (squares : List Char)
  (h : joined_in_sequence squares) :
  (opposite_face (fold_to_cube squares 'A')) = 'E' := 
sorry

end opposite_face_of_A_is_E_l3_3420


namespace distances_sum_inequality_l3_3283

theorem distances_sum_inequality
  (n : ℕ) (a : ℝ)
  (X : Π (i : fin n), fin n → ℝ) 
  (h : Π (i : fin n), ℝ)
  (h_pos : ∀ i, 0 < h i)
  (lines_defined_by_sides : ∀ i, is_line (Π (i : fin n), fin n → ℝ))
  (distances_from_X_to_lines : ∀ i, distance (X i) (lines_defined_by_sides i) = h i) :
  (∑ i, 1 / (h i)) > 2 * Real.pi / a :=
sorry

end distances_sum_inequality_l3_3283


namespace problem1_problem2_problem3_l3_3288

def a_level_associated_point (a : ℝ) (x y : ℝ) : ℝ × ℝ :=
  (a * x + y, x + a * y)

-- Problem 1
theorem problem1 : a_level_associated_point (1/2) 2 6 = (7 : ℝ, 5 : ℝ) :=
by sorry

-- Problem 2
theorem problem2 (a b : ℝ) (h1 : a_level_associated_point a 2 (-1) = (9, b)) :
  a + b = 2 :=
by sorry

-- Problem 3
theorem problem3 (m : ℝ)
  (N := a_level_associated_point (-4) (m - 1) (2 * m)) :
  N = (30 / 7, 0) ∨ N = (0, -15) :=
by sorry


end problem1_problem2_problem3_l3_3288


namespace sqrt_expression_simplification_l3_3453

theorem sqrt_expression_simplification : 
  (Real.sqrt 48 - Real.sqrt 2 * Real.sqrt 6 - Real.sqrt 15 / Real.sqrt 5) = Real.sqrt 3 := 
  by
    sorry

end sqrt_expression_simplification_l3_3453


namespace simplify_expression_value_given_condition_value_of_m_if_independent_l3_3511

variable (m y : ℝ)
def A := 2 * m^2 + 3 * m * y + 2 * y - 1
def B := m^2 - m * y

theorem simplify_expression :
  3 * A - 2 * (A + B) = 5 * m * y + 2 * y - 1 := by
  sorry

theorem value_given_condition :
  (m - 1)^2 + |y + 2| = 0 → 3 * A - 2 * (A + B) = -15 := by
  sorry

theorem value_of_m_if_independent (h : ∀ y, 3 * A - 2 * (A + B) = 5 * m * y + 2 * y - 1) :
  5 * m + 2 = 0 → m = -2 / 5 := by
  sorry

end simplify_expression_value_given_condition_value_of_m_if_independent_l3_3511


namespace probability_vertices_same_face_l3_3347

-- Define the structure of a cube
structure Cube where
  vertices : Fin 8
  faces : Fin 6
  vertices_per_face : Fin 4

-- Given a cube with the necessary properties
axiom cube : Cube

-- Define the theorem to prove the probability
theorem probability_vertices_same_face : 
  (let total_pairs := (8.choose 2) in
   let favorable_pairs := 24 in
   (favorable_pairs : ℚ) / total_pairs = 6 / 7) :=
by
  -- Placeholder for the actual proof
  sorry

end probability_vertices_same_face_l3_3347


namespace sum_a_b_l3_3936

variables {a b : ℝ}

def are_collinear (p1 p2 p3 : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, (p2.1 - p1.1, p2.2 - p1.2, p2.3 - p1.3) = k • (p3.1 - p2.1, p3.2 - p2.2, p3.3 - p2.3)

theorem sum_a_b {a b : ℝ} (h : are_collinear (1, a, b) (a, 2, b) (a, b, 3)) : a + b = 4 :=
sorry

end sum_a_b_l3_3936


namespace length_of_bridge_l3_3380

-- Given conditions
def lengthOfTrain : ℝ := 110
def speedInKmHr : ℝ := 45
def timeInSeconds : ℝ := 30

-- Convert speed from km/hr to m/s
def speedInMetersPerSecond : ℝ := speedInKmHr * 1000 / 3600

-- Calculate the total distance covered in 30 seconds
def totalDistance : ℝ := speedInMetersPerSecond * timeInSeconds

-- Define the problem: Prove that bridge length is 265 meters
theorem length_of_bridge
  (h1 : lengthOfTrain = 110)
  (h2 : speedInKmHr = 45)
  (h3 : timeInSeconds = 30)
  (h4 : speedInMetersPerSecond = speedInKmHr * 1000 / 3600)
  (h5 : totalDistance = speedInMetersPerSecond * timeInSeconds) :
  totalDistance - lengthOfTrain = 265 :=
sorry

end length_of_bridge_l3_3380


namespace line_intersects_circle_l3_3516

theorem line_intersects_circle (a : ℝ) (x y : ℝ)
  (C_eq : x^2 + y^2 = 4)
  (l_eq : a * x + y + 2 * a = 0)
  (AB_eq : |sqrt((x - x')^2 + (y - y')^2)| = 2 * sqrt 2) :
  (a = 1 ∨ a = -1) → (∀ x y, a * x + y + 2 * a = 0) :=
by
  sorry

end line_intersects_circle_l3_3516


namespace base7_digit_divisibility_by_13_l3_3869

theorem base7_digit_divisibility_by_13 (d : ℕ) (h : d ∈ {0, 1, 2, 3, 4, 5, 6}) :
  (693 + 56 * d) % 13 = 0 ↔ d = 0 :=
by
  sorry

end base7_digit_divisibility_by_13_l3_3869


namespace smallest_possible_sector_angle_l3_3964

theorem smallest_possible_sector_angle :
  ∃ (a1 : ℕ) (d : ℕ), (∀ n, (1 ≤ n ∧ n ≤ 10) → (∃ an : ℕ, an = a1 + (n - 1) * d ∧ an ≤ 360)) ∧
    (10 * (a1 + (a1 + 9 * d)) / 2 = 360) ∧
    (∃ a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℕ,
      a2 = a1 + d ∧ a3 = a1 + 2 * d ∧ a4 = a1 + 3 * d ∧ a5 = a1 + 4 * d ∧ 
      a6 = a1 + 5 * d ∧ a7 = a1 + 6 * d ∧ a8 = a1 + 7 * d ∧ a9 = a1 + 8 * d ∧ 
      a10 = a1 + 9 * d ∧ 
      10 * (a1 + a10) / 2 = 360 ∧ a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a1 = 360) ∧ 
    (∃ d, (∀ x, (2 * a1 + 9 * d = 72) → (a1 = 9))) := sorry

end smallest_possible_sector_angle_l3_3964


namespace determine_f_1789_l3_3034

theorem determine_f_1789
  (f : ℕ → ℕ)
  (h1 : ∀ n : ℕ, 0 < n → f (f n) = 4 * n + 9)
  (h2 : ∀ k : ℕ, f (2^k) = 2^(k+1) + 3) :
  f 1789 = 3581 :=
sorry

end determine_f_1789_l3_3034


namespace find_missing_even_number_l3_3376

theorem find_missing_even_number {n : ℕ} (h_series : (finset.range n).sum (λ i, 2 * (i + 1)) - k = 2012) : 
  k = 58 :=
sorry

end find_missing_even_number_l3_3376


namespace pizzas_difference_l3_3914

def pizzas (craig_first_day craig_second_day heather_first_day heather_second_day total_pizzas: ℕ) :=
  heather_first_day = 4 * craig_first_day ∧
  heather_second_day = craig_second_day - 20 ∧
  craig_first_day = 40 ∧
  craig_first_day + heather_first_day + craig_second_day + heather_second_day = total_pizzas

theorem pizzas_difference :
  ∀ (craig_first_day craig_second_day heather_first_day heather_second_day : ℕ),
  pizzas craig_first_day craig_second_day heather_first_day heather_second_day 380 →
  craig_second_day - craig_first_day = 60 :=
by
  intros craig_first_day craig_second_day heather_first_day heather_second_day h
  sorry

end pizzas_difference_l3_3914


namespace soldier_rearrangement_20x20_soldier_rearrangement_21x21_l3_3036

theorem soldier_rearrangement_20x20 (d : ℝ) : d ≤ 10 * Real.sqrt 2 :=
by
  -- Problem (a) setup and conditions
  sorry

theorem soldier_rearrangement_21x21 (d : ℝ) : d ≤ 10 * Real.sqrt 2 :=
by
  -- Problem (b) setup and conditions
  sorry

end soldier_rearrangement_20x20_soldier_rearrangement_21x21_l3_3036


namespace number_of_correct_propositions_is_two_l3_3517

open Set -- Open necessary namespaces

-- Define necessary entities: lines, planes, relationships
variables (Point : Type) (Line Plane : Type)
variables (l m : Line) (alpha beta : Plane)

-- Define the perpendicular and subset relationships
variable [Perp : has_perp Plane Plane]
variable [Subset : has_subset Line Plane]
variable [LinePerp : has_perp Line Plane]
variable [Parallel : has_parallel Plane Plane]
variable [LineParallel : has_parallel Line Line]

-- Given conditions
axiom l_perpendicular_alpha : l ⟂ alpha
axiom m_subset_beta : m ⊆ beta

-- Proposition definitions in Lean
def proposition1 : Prop := (α ∥ β) → (l ⟂ m)
def proposition2 : Prop := (l ⟂ m) → (α ∥ β)
def proposition3 : Prop := (α ⟂ β) → (l ⟂ m)
def proposition4 : Prop := (l ∥ m) → (α ⟂ β)

-- The proof goal: number of correct propositions
theorem number_of_correct_propositions_is_two :
  (proposition1 ∧ proposition4) ∧ ¬(proposition2 ∨ proposition3) :=
sorry

end number_of_correct_propositions_is_two_l3_3517


namespace max_value_of_vector_sum_l3_3881

open Real

variables {a b : ℝ} (A B : (EuclideanSpace ℝ (Fin 2)))

def non_zero_vectors (A B : EuclideanSpace ℝ (Fin 2)) : Prop := 
A ≠ 0 ∧ B ≠ 0

def angle_of_vectors (A B : EuclideanSpace ℝ (Fin 2)) : Prop := 
let θ := real.angle A B in
θ = π / 3

def vectors_difference_one (A B : EuclideanSpace ℝ (Fin 2)) : Prop :=
∥A - B∥ = 1

theorem max_value_of_vector_sum (h1 : non_zero_vectors A B) (h2 : angle_of_vectors A B) (h3 : vectors_difference_one A B) :
  ∥A + B∥ ≤ sqrt 3 := sorry

end max_value_of_vector_sum_l3_3881


namespace distance_focus_directrix_parabola_l3_3309

theorem distance_focus_directrix_parabola (x : ℝ) :
  let parabola := λ x, (1 / 4) * x ^ 2 in
  let focus := (0, 1 : ℝ) in
  let directrix := λ x, -1 in
  let distance := focus.snd - directrix 0 in
  distance = 2 :=
by
  -- The necessary proof should go here
  sorry

end distance_focus_directrix_parabola_l3_3309


namespace cos_pi_five_sin_sum_seven_sin_sum_l3_3656

theorem cos_pi_five:
  cos (Real.pi / 5) - cos (2 * Real.pi / 5) = 1 / 2 := by
  sorry

theorem sin_sum_seven:
  1 / sin (Real.pi / 7) = 1 / sin (2 * Real.pi / 7) + 1 / sin (3 * Real.pi / 7) := by
  sorry

theorem sin_sum:
  (List.range' 9 321).map (λ x, Real.sin (x * Real.pi / 180)).sum = 0 := by
  sorry

end cos_pi_five_sin_sum_seven_sin_sum_l3_3656


namespace pencils_placed_by_dan_l3_3694

-- Definitions based on the conditions provided
def pencils_in_drawer : ℕ := 43
def initial_pencils_on_desk : ℕ := 19
def new_total_pencils : ℕ := 78

-- The statement to be proven
theorem pencils_placed_by_dan : pencils_in_drawer + initial_pencils_on_desk + 16 = new_total_pencils :=
by
  sorry

end pencils_placed_by_dan_l3_3694


namespace max_intersections_in_first_quadrant_l3_3481

/-- Given 15 points on the positive x-axis and 10 points on the positive y-axis, each point
on the x-axis is connected to each point on the y-axis. Additionally, each point on the x-axis
is connected to a single common point in the first quadrant. This theorem proves that the 
maximum number of intersection points of these segments, solely within the interior of the 
first quadrant (excluding intersections involving the common point), is 4725. -/
theorem max_intersections_in_first_quadrant 
  (x_points : ℕ) (y_points : ℕ)
  (hx : x_points = 15) (hy : y_points = 10) : 
  ∃ n : ℕ, n = 4725 := 
by 
  have binom_x : ℕ := Nat.choose (x_points, 2)
  have binom_y : ℕ := Nat.choose (y_points, 2)
  have h1 : binom_x = 105 := by sorry
  have h2 : binom_y = 45 := by sorry
  let intersections := binom_x * binom_y
  have h3: intersections = 4725 := by sorry
  exact ⟨4725, h3⟩

end max_intersections_in_first_quadrant_l3_3481


namespace number_of_selection_methods_l3_3870

theorem number_of_selection_methods (M F : ℕ) (M_eq : M = 3) (F_eq : F = 6) :
  (combinatorics.combine (M + F) 5) - (combinatorics.combine F 5) = 120 :=
by sorry

end number_of_selection_methods_l3_3870


namespace path_dependency_time_lowest_point_cycloid_time_lowest_point_straight_l3_3763

-- Define the problem conditions
structure Cycloid where
  a : ℝ -- parameter of the cycloid
  g : ℝ -- acceleration due to gravity
  (Ha : a > 0) -- ensuring a is positive
  (Hg : g > 0) -- ensuring g is positive

-- Define the path traveled by the center of gravity on time
noncomputable def path_on_cycloid (cyc : Cycloid) (t : ℝ) : ℝ :=
  4 * cyc.a * (1 - Real.cos (t * Real.sqrt (cyc.g / (4 * cyc.a))))

-- Prove the path dependency of the ball on time
theorem path_dependency (cyc : Cycloid) (t : ℝ) : 
  path_on_cycloid cyc t = 4 * cyc.a * (1 - Real.cos (t * Real.sqrt (cyc.g / (4 * cyc.a)))) :=
sorry

-- Define the time to reach the lowest point on the cycloid
noncomputable def time_to_lowest_point_cycloid (cyc : Cycloid) : ℝ :=
  π * Real.sqrt (cyc.a / cyc.g)

-- Prove the time to reach the lowest point on cycloid
theorem time_lowest_point_cycloid (cyc : Cycloid) : 
  time_to_lowest_point_cycloid cyc = π * Real.sqrt (cyc.a / cyc.g) :=
sorry

-- Define the time to reach the lowest point on the straight line path
noncomputable def time_to_lowest_point_straight (cyc : Cycloid) : ℝ :=
  Real.sqrt (cyc.a * (4 + π^2) / cyc.g)

-- Prove the time to reach the lowest point on straight line
theorem time_lowest_point_straight (cyc : Cycloid) : 
  time_to_lowest_point_straight cyc = Real.sqrt (cyc.a * (4 + π^2) / cyc.g) :=
sorry

end path_dependency_time_lowest_point_cycloid_time_lowest_point_straight_l3_3763


namespace polar_to_cartesian_distance_l3_3589

open Real

theorem polar_to_cartesian_distance :
  let point_polar := (1 : ℝ, π / 2)
  let line_polar  := 2
  let point_cartesian := (0, 1 : ℝ × ℝ)
  let line_cartesian := 2
  distance point_cartesian (line_cartesian, 0 : ℝ × ℝ) = 2 :=
by 
  -- Definitions
  let point_polar := (1 : ℝ, π / 2)
  let line_polar := 2
  let x := point_polar.1 * cos point_polar.2
  let y := point_polar.1 * sin point_polar.2
  let point_cartesian := (x, y)
  let line_cartesian := (line_polar, 0 : ℝ × ℝ)
  -- Calculate distance
  have dist := abs (point_cartesian.1 - line_polar)
  -- Final distance
  show dist = 2, from sorry

end polar_to_cartesian_distance_l3_3589


namespace expected_OP_squared_proof_l3_3456

noncomputable def expected_OP_squared : ℝ := 10004

theorem expected_OP_squared_proof 
  (A B : ℝ × ℝ)
  (hA : A ∈ set.unit_circle)
  (hB : B ∈ (100 : ℝ) • set.unit_circle)
  (ℓ : ℝ → ℝ) -- tangent line at point A
  (P : ℝ × ℝ) -- reflection of B across ℓ
  (hℓ : tangent_to_circle ℓ A)
  (hP : reflected P B ℓ)
: ∑ P in {P | valid_reflection P hB hℓ}, OP_squared P hℓ := expected_OP_squared :=
sorry

end expected_OP_squared_proof_l3_3456


namespace problem_statement_l3_3744

def vector := (ℝ × ℝ × ℝ)

def are_parallel (v1 v2 : vector) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ v1 = (k * v2.1, k * v2.2, k * v2.3)

def are_perpendicular (v1 v2 : vector) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3 = 0

theorem problem_statement :
  (are_parallel (2, 3, -1) (-2, -3, 1)) ∧
  (¬ are_perpendicular (1, -1, 2) (6, 4, -1)) ∧
  (are_perpendicular (2, 2, -1) (-3, 4, 2)) ∧
  (¬ are_parallel (0, 3, 0) (0, -5, 0)) :=
by
  sorry

end problem_statement_l3_3744


namespace coefficient_y_in_second_eq_l3_3895

variables (x y z : ℝ)

-- Conditions
def eq1 : Prop := 6 * x - 5 * y + 3 * z = 22
def eq2 : Prop := 4 * x + y - 11 * z = 7 / 8
def eq3 : Prop := 5 * x - 6 * y + 2 * z = 12
def sum_eq : Prop := x + y + z = 10

-- Proof Problem
theorem coefficient_y_in_second_eq
  (h1 : eq1 x y z) (h2 : eq2 x y z) (h3 : eq3 x y z) (h_sum : sum_eq x y z) :
  coefficient_of_y_eq2 : y := y sorry

end coefficient_y_in_second_eq_l3_3895


namespace point_A_is_closer_to_origin_l3_3279

theorem point_A_is_closer_to_origin (A B : ℤ) (hA : A = -2) (hB : B = 3) : abs A < abs B := by 
sorry

end point_A_is_closer_to_origin_l3_3279


namespace sum_of_y_coeffs_is_69_l3_3366

-- Introduction of the polynomials involved
def poly1 : ℤ[X, Y] := 4 * (X^1) + 3 * (Y^1) + 2
def poly2 : ℤ[X, Y] := 2 * (X^1) + 5 * (Y^1) + 6

-- The expanded form of the polynomials
noncomputable def expanded : ℤ[X, Y] := poly1 * poly2

-- Summing the coefficients of the terms containing a nonzero power of y
noncomputable def sum_of_coefficients_of_y_terms : ℤ :=
  coeff (monomial 0 1) expanded + coeff (monomial 1 1) expanded + coeff (monomial 0 2) expanded

-- Theorem stating the sum of the coefficients is 69
theorem sum_of_y_coeffs_is_69 : sum_of_coefficients_of_y_terms = 69 := by
  sorry

end sum_of_y_coeffs_is_69_l3_3366


namespace probability_at_most_one_match_l3_3353

theorem probability_at_most_one_match :
  let emails := {1, 2, 3, 4}
  let websites := {1, 2, 3, 4}
  let total_outcomes := 24
  let matching_outcomes := 7
  let probability_more_than_one_match := matching_outcomes / total_outcomes
  let probability_at_most_one_match := 1 - probability_more_than_one_match
  probability_at_most_one_match = 17 / 24 :=
by
  sorry

end probability_at_most_one_match_l3_3353


namespace S_5_is_121_l3_3148

-- Definitions of the sequence and its terms
def S : ℕ → ℕ := sorry  -- Define S_n
def a : ℕ → ℕ := sorry  -- Define a_n

-- Conditions
axiom S_2 : S 2 = 4
axiom recurrence_relation : ∀ n : ℕ, S (n + 1) = 1 + 2 * S n

-- Proof that S_5 = 121 given the conditions
theorem S_5_is_121 : S 5 = 121 := by
  sorry

end S_5_is_121_l3_3148


namespace james_out_of_pocket_cost_l3_3239

-- Definitions
def doctor_charge : ℕ := 300
def insurance_coverage_percentage : ℝ := 0.80

-- Proof statement
theorem james_out_of_pocket_cost : (doctor_charge : ℝ) * (1 - insurance_coverage_percentage) = 60 := 
by sorry

end james_out_of_pocket_cost_l3_3239


namespace constant_term_in_expansion_l3_3115

noncomputable def constant_term_binomial_expansion (a b : ℕ) : ℕ :=
  let n := 6
  let r := 2 in
  binom n r * (2^(n-r) : ℕ)

theorem constant_term_in_expansion :
  constant_term_binomial_expansion 2 (-1) = 240 := 
sorry

end constant_term_in_expansion_l3_3115


namespace find_k_values_l3_3847

open Real

noncomputable def vector_norm (v : ℝ × ℝ) : ℝ :=
  sqrt (v.1^2 + v.2^2)

theorem find_k_values (k : ℝ) :
  (vector_norm (k * (3, -4) - (5, 8)) = 5 * sqrt 10) ∧
  (vector_norm (2 * k * (3, -4) + (10, 16)) = 10 * sqrt 10) ↔
  (k = 2.08) ∨ (k = -3.44) :=
by sorry

end find_k_values_l3_3847


namespace Arman_worked_last_week_l3_3246

variable (H : ℕ) -- hours worked last week
variable (wage_last_week wage_this_week : ℝ)
variable (hours_this_week worked_this_week two_weeks_earning : ℝ)
variable (worked_last_week : Prop)

-- Define assumptions based on the problem conditions
def condition1 : wage_last_week = 10 := by sorry
def condition2 : wage_this_week = 10.5 := by sorry
def condition3 : hours_this_week = 40 := by sorry
def condition4 : worked_this_week = wage_this_week * hours_this_week := by sorry
def condition5 : worked_this_week = 420 := by sorry -- 10.5 * 40
def condition6 : two_weeks_earning = wage_last_week * (H : ℝ) + worked_this_week := by sorry
def condition7 : two_weeks_earning = 770 := by sorry

-- Proof statement
theorem Arman_worked_last_week : worked_last_week := by
  have h1 : wage_last_week * (H : ℝ) + worked_this_week = two_weeks_earning := sorry
  have h2 : wage_last_week * (H : ℝ) + 420 = 770 := sorry
  have h3 : wage_last_week * (H : ℝ) = 350 := sorry
  have h4 : (10 : ℝ) * (H : ℝ) = 350 := sorry
  have h5 : H = 35 := sorry
  sorry

end Arman_worked_last_week_l3_3246


namespace product_diff_is_square_l3_3457

noncomputable def is_square (n : Int) : Prop :=
  ∃ m : Int, m * m = n

theorem product_diff_is_square
  (a1 a2 a3 a4 b1 b2 b3 b4 : Int)
  (h1 : a1 ≠ b1) (h2 : a2 ≠ b2) (h3 : a3 ≠ b3) (h4 : a4 ≠ b4)
  (h5 : {a1 + a2, a1 + b2, b1 + a2, b1 + b2} = {a3 + a4, a3 + b4, b3 + a4, b3 + b4}) :
  is_square (|(a1 - b1) * (a2 - b2) * (a3 - b3) * (a4 - b4)|) := sorry

end product_diff_is_square_l3_3457


namespace arm_wrestling_tournament_min_rounds_l3_3397

theorem arm_wrestling_tournament_min_rounds (N : ℕ) (hN : N = 510) :
  ∃ r : ℕ, r = 9 ∧ 
    ∀ (points : ℕ → ℕ) (meet : ℕ → ℕ → Prop),
      (∀ i, points i = 0 ∨ points i = 0) →
      (∀ i j, meet i j → |points i - points j| ≤ 1) →
      (∀ i j, meet i j → if points i < points j then points i + 1 = points i) →
      ∃ leader, leader ∈ {i | points i = max points i} := 
sorry

end arm_wrestling_tournament_min_rounds_l3_3397


namespace c_20_eq_49_Sn_arithmetic_exists_arithmetic_seq_l3_3911

section Problem1
variable {a b : ℕ → ℕ}
variables (h1 : a 0 = 1) (h2 : b 0 = 1)
variables (h3 : a 1 = b 2) (h4 : a 5 = b 4)

noncomputable def arithmetic_sequence (n : ℕ) : ℕ := 3 * n - 2
noncomputable def geometric_sequence (n : ℕ) : ℕ := 2^(n-1)
noncomputable def combined_sequence (n : ℕ) : ℕ := if n < 17 then arithmetic_sequence n else geometric_sequence (n - 16)
theorem c_20_eq_49 : combined_sequence 19 = 49 :=
sorry
end Problem1

section Problem2
variable {a : ℕ → ℕ}
variables (h1 : a 0 = 1) (h2 : ∀ n, a n > 0)
variables (h3 : ∀ n, ∃ k, a k = 3^n)
noncomputable def c_sequence_arithmetic (n d : ℕ) : ℕ := d * n + 1 - d
noncomputable def sum_first_n_terms (n d : ℕ) : ℕ := if d = 1 then Nat.succ n * n / 2 else n * n

theorem Sn_arithmetic (d n : ℕ) (h4 : (∀ k, ∃ m, a m = 3^k) → (c_sequence_arithmetic k d = 3^k)) :
sum_first_n_terms n d = if d = 1 then n * (n + 1) / 2 else n * n :=
sorry
end Problem2

section Problem3
variables {a : ℕ → ℕ} {q : ℕ}
variables (h1 : 1 < q)
variables (h2 : ∃ a0, 1 < a0 ∧ a0 < q)
noncomputable def b_seq (n : ℕ) : ℕ := q^(n-1)
noncomputable def a_seq (n : ℕ) (d : ℕ) : ℕ := (n - 1) * d + a 0 
theorem exists_arithmetic_seq (h3 : ∀ n, a (n + b_seq n - 1) < a_seq (n + 1) (q - 1)):
∃ d, ∃ a0 ∈ (1 : ℕ, q), a_seq n d :=
sorry
end Problem3

end c_20_eq_49_Sn_arithmetic_exists_arithmetic_seq_l3_3911


namespace largest_possible_sum_l3_3327

theorem largest_possible_sum (f g h j : ℕ) (hf : f ∈ ({3, 5, 7, 9} : Finset ℕ))
  (hg : g ∈ ({3, 5, 7, 9} : Finset ℕ))
  (hh : h ∈ ({3, 5, 7, 9} : Finset ℕ))
  (hj : j ∈ ({3, 5, 7, 9} : Finset ℕ))
  (hf_ne_hg : f ≠ g)
  (hf_ne_hh : f ≠ h)
  (hf_ne_hj : f ≠ j)
  (hg_ne_hh : g ≠ h)
  (hg_ne_hj : g ≠ j)
  (hh_ne_hj : h ≠ j) :
  fg + gh + hj + fj ≤ 144 := by
  sorry

end largest_possible_sum_l3_3327


namespace city_a_vs_city_b_l3_3683

-- Working with average costs.
def city_a_highest_avg_cost := 45
def city_b_lowest_avg_cost := 25

-- Define the mathematical relationship to calculate percentage increase.
def percentage_increase (highest lowest : ℝ) : ℝ := ((highest - lowest) / lowest) * 100

-- The target theorem statement, proving the percentage increase is 80%.
theorem city_a_vs_city_b : percentage_increase city_a_highest_avg_cost city_b_lowest_avg_cost = 80 := 
by 
  -- Placeholder proof
  sorry

end city_a_vs_city_b_l3_3683


namespace partI_partII_l3_3526

noncomputable def a := ∫ x in 0..π, (sin x - 1 + 2 * (cos x / 2) ^ 2)

theorem partI : a = 2 := by
  sorry

def expr (x : ℝ) : ℝ := (2 * sqrt x - 1 / sqrt x) ^ 6 * (x ^ 2 + 2)

theorem partII : constant_term (expr x) = -334 := by
  sorry

end partI_partII_l3_3526


namespace smallest_y_value_l3_3355

noncomputable def f (y : ℝ) : ℝ := 3 * y ^ 2 + 27 * y - 90
noncomputable def g (y : ℝ) : ℝ := y * (y + 15)

theorem smallest_y_value (y : ℝ) : (∀ y, f y = g y → y ≠ -9) → false := by
  sorry

end smallest_y_value_l3_3355


namespace integral_transform_eq_l3_3747

open MeasureTheory

variable (f : ℝ → ℝ)

theorem integral_transform_eq (hf_cont : Continuous f) (hL_exists : ∃ L, ∫ x in (Set.univ : Set ℝ), f x = L) :
  ∃ L, ∫ x in (Set.univ : Set ℝ), f (x - 1/x) = L :=
by
  cases' hL_exists with L hL
  use L
  have h_transform : ∫ x in (Set.univ : Set ℝ), f (x - 1/x) = ∫ x in (Set.univ : Set ℝ), f x := sorry
  rw [h_transform]
  exact hL

end integral_transform_eq_l3_3747


namespace smallest_B_to_divisible_3_l3_3603

-- Define the problem
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Define the digits in the integer
def digit_sum (B : ℕ) : ℕ := 8 + B + 4 + 6 + 3 + 5

-- Prove that the smallest digit B that makes 8B4,635 divisible by 3 is 1
theorem smallest_B_to_divisible_3 : ∃ B : ℕ, B ≥ 0 ∧ B ≤ 9 ∧ is_divisible_by_3 (digit_sum B) ∧ ∀ B' : ℕ, B' < B → ¬ is_divisible_by_3 (digit_sum B') ∧ B = 1 :=
sorry

end smallest_B_to_divisible_3_l3_3603


namespace snowman_volume_correct_l3_3596

def volume_of_snowman (r1 r2 r3 reduction : ℝ) : ℝ :=
  let vol r := (4 / 3) * Real.pi * r^3
  vol (r1 - reduction) + vol (r2 - reduction) + vol (r3 - reduction)

theorem snowman_volume_correct :
  volume_of_snowman 4 6 8 0.5 = (841.5 / 3) * Real.pi := by
  sorry

end snowman_volume_correct_l3_3596


namespace PMID_concyclic_or_collinear_l3_3630

theorem PMID_concyclic_or_collinear
  (A B C I D E F P M: Point)
  (hI : is_incenter I A B C)
  (h_incircle : tangent_point I A B C D E F)
  (h_AD_second_meet : second_intersection AD_incircle A P)
  (h_M_midpoint : midpoint M E F) :
  Cyclic P M I D ∨ Collinear P M I D := 
sorry

end PMID_concyclic_or_collinear_l3_3630


namespace problem_statement_l3_3135

def g (n : ℕ) : ℝ := Real.log (n ^ 3) / Real.log 3003

theorem problem_statement : g 7 + g 11 + g 13 = 9 / 4 := sorry

end problem_statement_l3_3135


namespace Marcy_spears_l3_3275

def makeSpears (saplings: ℕ) (logs: ℕ) (branches: ℕ) (trunks: ℕ) : ℕ :=
  3 * saplings + 9 * logs + 7 * branches + 15 * trunks

theorem Marcy_spears :
  makeSpears 12 1 6 0 - (3 * 2) + makeSpears 0 4 0 0 - (9 * 4) + makeSpears 0 0 6 1 - (7 * 0) + makeSpears 0 0 0 2 = 81 := by
  sorry

end Marcy_spears_l3_3275


namespace smaller_of_two_integers_l3_3302

noncomputable def smaller_integer (m n : ℕ) : ℕ :=
if m < n then m else n

theorem smaller_of_two_integers :
  ∀ (m n : ℕ),
  100 ≤ m ∧ m < 1000 ∧ 100 ≤ n ∧ n < 1000 ∧
  (m + n) / 2 = m + n / 200 →
  smaller_integer m n = 891 :=
by
  intros m n h
  -- Assuming m, n are positive three-digit integers and satisfy the condition
  have h1 := h.1
  have h2 := h.2.1
  have h3 := h.2.2.1
  have h4 := h.2.2.2.1
  have h5 := h.2.2.2.2
  sorry

end smaller_of_two_integers_l3_3302


namespace min_sum_a1_a2_l3_3326

noncomputable def sequence (a : ℕ → ℕ) : Prop :=
∀ n, a (n + 2) = (a n + 3007) / (1 + a (n + 1))

theorem min_sum_a1_a2 :
  ∀ (a : ℕ → ℕ), (∀ i, a i > 0) → sequence a → (∃ a1 a2, a1 = a 1 ∧ a2 = a 2 ∧ a1 + a2 = 114) :=
by
  sorry

end min_sum_a1_a2_l3_3326


namespace sweetsies_remainder_l3_3479

-- Each definition used in Lean 4 statement should be directly from the conditions in a)
def number_of_sweetsies_in_one_bag (m : ℕ): Prop :=
  m % 8 = 5

theorem sweetsies_remainder (m : ℕ) (h : number_of_sweetsies_in_one_bag m) : 
  (4 * m) % 8 = 4 := by
  -- Proof will be provided here.
  sorry

end sweetsies_remainder_l3_3479


namespace find_line_equation_l3_3053

-- Define point A
def point_A : ℝ × ℝ := (-3, 4)

-- Define the conditions
def passes_through_point_A (line_eq : ℝ → ℝ → ℝ) : Prop :=
  line_eq (-3) 4 = 0

def intercept_condition (line_eq : ℝ → ℝ → ℝ) : Prop :=
  ∃ a : ℝ, a ≠ 0 ∧ line_eq (2 * a) 0 = 0 ∧ line_eq 0 a = 0

-- Define the equations of the line
def line1 (x y : ℝ) : ℝ := 3 * y + 4 * x
def line2 (x y : ℝ) : ℝ := 2 * x - y - 5

-- Statement of the problem
theorem find_line_equation : 
  (passes_through_point_A line1 ∧ intercept_condition line1) ∨
  (passes_through_point_A line2 ∧ intercept_condition line2) :=
sorry

end find_line_equation_l3_3053


namespace line_eq_y_eq_x_plus_3_line_above_x_axis_l3_3546

-- Define the points A and B
def A : ℝ × ℝ := (-1, 2)
def B : ℝ × ℝ := (2, 5)

-- Define the line equation given slope k and intercept b
def line (k b : ℝ) (x : ℝ) : ℝ := k * x + b

-- Prove that the line passing through A and B is y = x + 3
theorem line_eq_y_eq_x_plus_3 : 
  ∃ (k b : ℝ), (∀ x y : ℝ, (y = line k b x) ↔ (y = x + 3)) ∧ (line k b (-1) = 2) ∧ (line k b 2 = 5) :=
by
  -- Definitions derived from given conditions
  existsi 1
  existsi 3
  split
  {
    -- Prove that the line equation is consistent with y = x + 3
    intros
    split
    {
      intro
      exact h
    }
    {
      intro
      exact h
    }
  }
  split
  {
    show line 1 3 (-1) = 2, by sorry
  }
  {
    show line 1 3 2 = 5, by sorry
  }

-- Prove that the line y = x + 3 lies above the x-axis for x > -3
theorem line_above_x_axis (x : ℝ) : 
  (line 1 3 x > 0) ↔ (x > -3) :=
by 
  -- Show that x + 3 > 0 implies x > -3
  transitivity
  exact x > -3
  sorry

end line_eq_y_eq_x_plus_3_line_above_x_axis_l3_3546


namespace students_in_section_A_l3_3330

theorem students_in_section_A (x : ℕ) (h1 : (40 : ℝ) * x + 44 * 35 = 37.25 * (x + 44)) : x = 36 :=
by
  sorry

end students_in_section_A_l3_3330


namespace Leibniz_Theorem_l3_3614

-- Define the coordinates of points
variables {M A B C : Point}
variables {G : Point}

-- Assume G is the centroid of triangle ABC
axiom G_centroid : G = centroid A B C

-- Assume the coordinate of point M
axiom M_arbitrary : Point M

-- Define squared distance function
def dist_squared (P Q : Point) : ℝ := (P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2

theorem Leibniz_Theorem (M A B C G : Point) :
  3 * dist_squared M G =  dist_squared M A + dist_squared M B + dist_squared M C - 
                          (dist_squared A B + dist_squared B C + dist_squared C A) / 3 := 
by
  sorry

end Leibniz_Theorem_l3_3614


namespace state_tax_percentage_l3_3444

theorem state_tax_percentage (weekly_salary federal_percent health_insurance life_insurance parking_fee final_paycheck : ℝ)
  (h_weekly_salary : weekly_salary = 450)
  (h_federal_percent : federal_percent = 1/3)
  (h_health_insurance : health_insurance = 50)
  (h_life_insurance : life_insurance = 20)
  (h_parking_fee : parking_fee = 10)
  (h_final_paycheck : final_paycheck = 184) :
  (36 / 450) * 100 = 8 :=
by
  sorry

end state_tax_percentage_l3_3444


namespace problem_solution_l3_3359

theorem problem_solution : (2^0 - 1 + 5^2 - 0)⁻¹ * 5 = 1 / 5 := by
  sorry

end problem_solution_l3_3359


namespace solve_linear_system_l3_3293

theorem solve_linear_system (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) :
  2 * x₁ + 2 * x₂ - x₃ + x₄ + 4 * x₆ = 0 ∧
  x₁ + 2 * x₂ + 2 * x₃ + 3 * x₅ + x₆ = -2 ∧
  x₁ - 2 * x₂ + x₄ + 2 * x₅ = 0 →
  x₁ = -1 / 4 - 5 / 8 * x₄ - 9 / 8 * x₅ - 9 / 8 * x₆ ∧
  x₂ = -1 / 8 + 3 / 16 * x₄ - 7 / 16 * x₅ + 9 / 16 * x₆ ∧
  x₃ = -3 / 4 + 1 / 8 * x₄ - 11 / 8 * x₅ + 5 / 8 * x₆ :=
by
  sorry

end solve_linear_system_l3_3293


namespace motorcycle_race_l3_3941

theorem motorcycle_race (
  (s x : ℝ)
  (h1 : 0 < s)
  (h2 : 0 < x)
  (eq1 : s / (x + 15) = s / x - 12 / 60)
  (eq2 : s / x = s / (x - 3) - 3 / 60)
) : 
  s = 90 ∧ x = 75 ∧ (x + 15) = 90 ∧ (x - 3) = 72 :=
by
  sorry

end motorcycle_race_l3_3941


namespace modulo_residue_l3_3260

theorem modulo_residue (T : ℤ) (h : T = ∑ i in (range 2020), (-1)^i * (i+1)) : T % 2020 = 0 :=
sorry

end modulo_residue_l3_3260


namespace find_f_1789_l3_3032

-- Given conditions as definitions

def f : ℕ → ℕ := sorry
axiom f_a (n : ℕ) (hn : n > 0) : f(f(n)) = 4 * n + 9
axiom f_b (k : ℕ) : f(2^k) = 2^(k+1) + 3

-- The theorem to prove f(1789) = 3581 given the conditions
theorem find_f_1789 : f(1789) = 3581 := sorry

end find_f_1789_l3_3032


namespace sum_of_squares_real_solutions_l3_3469

theorem sum_of_squares_real_solutions :
  ∑ x in {x : ℝ | x^81 = 3^36}, x^2 = 13122 :=
sorry

end sum_of_squares_real_solutions_l3_3469


namespace freshmen_psychology_majors_percentage_l3_3806

-- Definitions based on the conditions
def school_distribution := {liberal_arts := 0.35, science := 0.25, business := 0.30, engineering := 0.10}
def school_demographics := {freshmen := 0.40, sophomores := 0.25, juniors := 0.20, seniors := 0.15}
def liberal_arts_distribution := {freshmen := 0.60, sophomores := 0.10, juniors := 0.20, seniors := 0.10}
def psychology_distribution_in_liberal_arts := {freshmen := 0.30}

-- Theorem to prove the required percentage
theorem freshmen_psychology_majors_percentage :
  let total_students := 1
  let lib_arts_students := school_distribution.liberal_arts * total_students
  let lib_arts_freshmen := lib_arts_students * liberal_arts_distribution.freshmen
  let freshmen_psychology_majors := lib_arts_freshmen * psychology_distribution_in_liberal_arts.freshmen
  let percentage := freshmen_psychology_majors / total_students
  percentage = 0.063 :=
by
  -- This is where the proof would go
  sorry

end freshmen_psychology_majors_percentage_l3_3806


namespace kellys_apples_l3_3613

def apples_kelly_needs_to_pick := 49
def total_apples := 105

theorem kellys_apples :
  ∃ x : ℕ, x + apples_kelly_needs_to_pick = total_apples ∧ x = 56 :=
sorry

end kellys_apples_l3_3613


namespace mode_and_median_of_scores_l3_3939

def scores : List ℕ := [90, 89, 90, 95, 92, 94, 93, 90]

theorem mode_and_median_of_scores :
  Multiset.mode (scores : Multiset ℕ) = 90 ∧
  Multiset.median (scores : Multiset ℕ) = 91 := by
  sorry

end mode_and_median_of_scores_l3_3939


namespace limit_problem_l3_3527

noncomputable def f := λ x : ℝ, 1 / x

theorem limit_problem : 
  (∀ x : ℝ, f(x) = 1 / x) →
  (∃ L, tendsto (λ Δx : ℝ, (-f (2 + Δx) + f 2) / Δx) (𝓝 0) (𝓝 L) ∧ L = 1 / 4) :=
  by
    intros h_f
    use 1 / 4
    sorry

end limit_problem_l3_3527


namespace distance_train_A_when_meeting_l3_3014

noncomputable def distance_traveled_by_train_A : ℝ :=
  let distance := 375
  let time_A := 36
  let time_B := 24
  let speed_A := distance / time_A
  let speed_B := distance / time_B
  let relative_speed := speed_A + speed_B
  let time_meeting := distance / relative_speed
  speed_A * time_meeting

theorem distance_train_A_when_meeting :
  distance_traveled_by_train_A = 150 := by
  sorry

end distance_train_A_when_meeting_l3_3014


namespace g_range_l3_3264

-- Define the piecewise function f(x)
def f (x : ℝ) : ℝ :=
  if abs x < 1 then x else x^2

-- Define the condition g function range so that the range of f(g(x)) is [0, +∞)
theorem g_range {g : ℝ → ℝ} (h : ∀ y, 0 ≤ y → ∃ x, g x = y) : range g = set.Ici 0 :=
by
  sorry

end g_range_l3_3264


namespace young_people_in_sample_l3_3768

-- Define the conditions
def total_population (elderly middle_aged young : ℕ) : ℕ :=
  elderly + middle_aged + young

def sample_proportion (sample_size total_pop : ℚ) : ℚ :=
  sample_size / total_pop

def stratified_sample (group_size proportion : ℚ) : ℚ :=
  group_size * proportion

-- Main statement to prove
theorem young_people_in_sample (elderly middle_aged young : ℕ) (sample_size : ℚ) :
  total_population elderly middle_aged young = 108 →
  sample_size = 36 →
  stratified_sample (young : ℚ) (sample_proportion sample_size 108) = 17 :=
by
  intros h_total h_sample_size
  sorry -- proof omitted

end young_people_in_sample_l3_3768


namespace closest_to_sin_2016_deg_is_neg_half_l3_3487

/-- Given the value of \( \sin 2016^\circ \), show that the closest number from the given options is \( -\frac{1}{2} \).
Options:
A: \( \frac{11}{2} \)
B: \( -\frac{1}{2} \)
C: \( \frac{\sqrt{2}}{2} \)
D: \( -1 \)
-/
theorem closest_to_sin_2016_deg_is_neg_half :
  let sin_2016 := Real.sin (2016 * Real.pi / 180)
  |sin_2016 - (-1 / 2)| < |sin_2016 - 11 / 2| ∧
  |sin_2016 - (-1 / 2)| < |sin_2016 - Real.sqrt 2 / 2| ∧
  |sin_2016 - (-1 / 2)| < |sin_2016 - (-1)| :=
by
  sorry

end closest_to_sin_2016_deg_is_neg_half_l3_3487


namespace algebraic_expression_value_l3_3557

variable {R : Type} [CommRing R]

theorem algebraic_expression_value (m n : R) (h1 : m - n = -2) (h2 : m * n = 3) :
  -m^3 * n + 2 * m^2 * n^2 - m * n^3 = -12 :=
sorry

end algebraic_expression_value_l3_3557


namespace modulus_of_z_l3_3165

-- Given condition
def z : ℂ := (2 + (complex.I : ℂ)) / (1 - (complex.I : ℂ))

-- Proof statement
theorem modulus_of_z : complex.abs z = real.sqrt 10 / 2 :=
by
  sorry

end modulus_of_z_l3_3165


namespace no_increasing_sequence_with_unique_sum_l3_3473

theorem no_increasing_sequence_with_unique_sum :
  ¬ (∃ (a : ℕ → ℕ), (∀ n, 0 < a n) ∧ (∀ n, a n < a (n + 1)) ∧ 
  (∀ N, ∃ k ≥ N, ∀ m ≥ k, 
    (∃! (i j : ℕ), a i + a j = m))) := sorry

end no_increasing_sequence_with_unique_sum_l3_3473


namespace birthday_cake_cost_is_25_l3_3476

def cost_of_gift : ℝ := 250
def erika_savings : ℝ := 155
def rick_savings : ℝ := cost_of_gift / 2
def total_savings : ℝ := erika_savings + rick_savings
def amount_left : ℝ := 5
def total_expenditure : ℝ := total_savings - amount_left
def cost_of_birthday_cake : ℝ := total_expenditure - cost_of_gift

theorem birthday_cake_cost_is_25 : cost_of_birthday_cake = 25 := by
  sorry

end birthday_cake_cost_is_25_l3_3476


namespace select_one_each_category_select_two_different_types_l3_3910

-- Problem 1: Number of ways to choose one painting from each category
theorem select_one_each_category (T O W : ℕ) (hT : T = 5) (hO : O = 2) (hW : W = 7) :
  T * O * W = 70 := by
  rw [hT, hO, hW]
  norm_num
  sorry

-- Problem 2: Number of ways to choose two paintings of different types
theorem select_two_different_types (T O W : ℕ) (hT : T = 5) (hO : O = 2) (hW : W = 7) :
  T * O + T * W + O * W = 59 := by
  rw [hT, hO, hW]
  norm_num
  sorry

end select_one_each_category_select_two_different_types_l3_3910


namespace find_2019th_integer_not_divisible_by_5_l3_3849

/-- Function to calculate the 5-adic valuation of a binomial coefficient -/
def binomial_5_adic_valuation (n : ℕ) : ℕ :=
  PadicVal.eval 5 (Nat.choose (2 * n) n)

/-- Find the 2019th positive integer n such that the binomial coefficient 
  (2n choose n) is not divisible by 5. -/
theorem find_2019th_integer_not_divisible_by_5 : 
  ∃ n : ℕ, (∀ i < 2019, ∃ m : ℕ, m > 0 ∧ binomial_5_adic_valuation m = 0 ∧ m < n) 
  ∧ binomial_5_adic_valuation n = 0 ∧ n = 37805 :=
sorry

end find_2019th_integer_not_divisible_by_5_l3_3849


namespace count_4_digit_mountain_numbers_l3_3015

def is_mountain_number (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  n >= 1000 ∧ n < 10000 ∧ 
  (∃ i, digits !! i > digits !! (i - 1) ∧ digits !! i > digits !! (i + 1)) ∧
  ∀ j ≠ i, digits !! j ≤ digits !! i

theorem count_4_digit_mountain_numbers : ∃ n, (∃ f : fin 4 → ℕ, 
  ∀ i, f i ∈ {0,1,2,3,4,5,6,7,8,9}) ∧ 
  (∀ x ≠ y, f x ≠ f y) ∧
  (∃ i, f i > f (i - 1) ∧ f i > f (i + 1)) ∧ 
  ∀ j ≠ i, f j ≤ f i) ∧
  n = 162 := sorry

end count_4_digit_mountain_numbers_l3_3015


namespace original_price_of_iWatch_l3_3108

theorem original_price_of_iWatch (P : ℝ) (h1 : 800 > 0) (h2 : P > 0)
    (h3 : 680 + 0.90 * P > 0) (h4 : 0.98 * (680 + 0.90 * P) = 931) :
    P = 300 := by
  sorry

end original_price_of_iWatch_l3_3108


namespace domain_f_value_at_pi_over_4_interval_monotonic_increase_l3_3899

noncomputable def f (x : ℝ) : ℝ := (sin (2 * x) + 2 * cos (x)^2) / cos (x)

-- Proving the domain
theorem domain_f : { x : ℝ | ∃ k : ℤ, x = k * π + π / 2 } = { x | x ≠ kπ + π / 2 ∧ k ∈ ℤ } :=
sorry

-- Proving the value at π/4
theorem value_at_pi_over_4 : f (π / 4) = 2 * sqrt 2 :=
sorry

-- Proving the interval of monotonic increase
theorem interval_monotonic_increase : (∀ x ∈ Ioo 0 (π / 2), f x) = Ioo 0 (π / 4) :=
sorry

end domain_f_value_at_pi_over_4_interval_monotonic_increase_l3_3899


namespace hyperbola_eccentricity_value_l3_3538

noncomputable def hyperbola_eccentricity (a b : ℝ) (cond_a : 0 < a) (cond_b : 0 < b)
  (dist_formula : ∀ (c : ℝ), c = Real.sqrt (a^2 + b^2) → (abs (b * c) / Real.sqrt (b^2 + a^2)) = (Real.sqrt 5 / 3) * c) : ℝ := 
c / a

theorem hyperbola_eccentricity_value (a b c : ℝ) (cond_a : 0 < a) (cond_b : 0 < b)
  (cond_c : c = Real.sqrt (a^2 + b^2)) 
  (dist_formula : (abs (b * c) / Real.sqrt (b^2 + a^2)) = (Real.sqrt 5 / 3) * c) : 
  hyperbola_eccentricity a b cond_a cond_b dist_formula = 3 / 2 :=
sorry

end hyperbola_eccentricity_value_l3_3538


namespace sum_multiples_3_to_1000_l3_3201

theorem sum_multiples_3_to_1000 : 
  (∑ k in finset.range (1000 + 1), if k % 3 = 0 then k else 0) = 166833 := by sorry

end sum_multiples_3_to_1000_l3_3201


namespace ant_at_C_after_4_minutes_l3_3801

open ProbabilityTheory

-- Definitions of points on the lattice
structure Point :=
  (x : Int)
  (y : Int)

-- Define movement on the lattice
def move (p : Point) (d : Point) : Point :=
  ⟨p.x + d.x, p.y + d.y⟩

-- Adjacent moves (up, down, left, right)
def adjacent_moves : Set Point :=
  {⟨0, 1⟩, ⟨0, -1⟩, ⟨1, 0⟩, ⟨-1, 0⟩}

-- Probabilistic function to determine reach
noncomputable def transition_prob (start : Point) (end : Point) (n : Nat) : ℝ := sorry

theorem ant_at_C_after_4_minutes (A C : Point) (H_A : A = ⟨0, 0⟩) (H_C : C = ⟨2, 0⟩) :
  transition_prob A C 4 = 1/3 := 
sorry

end ant_at_C_after_4_minutes_l3_3801


namespace sum_g_47_l3_3980

noncomputable def f (x : ℝ) : ℝ := 4 * x ^ 2 - 3
noncomputable def g (y : ℝ) : ℝ := 
  if h : ∃ x : ℝ, f x = y then 
    let x := Classical.choose h 
    x ^ 2 - x + 2 
  else 0

theorem sum_g_47 : g 47 = 29 := 
by 
  sorry

end sum_g_47_l3_3980


namespace coffee_tea_soda_l3_3277

theorem coffee_tea_soda (Pcoffee Ptea Psoda Pboth_no_soda : ℝ)
  (H1 : 0.9 = Pcoffee)
  (H2 : 0.8 = Ptea)
  (H3 : 0.7 = Psoda) :
  0.0 = Pboth_no_soda :=
  sorry

end coffee_tea_soda_l3_3277


namespace max_value_of_seq_ratio_l3_3164

-- Defining the sequence {b_n} and its properties.
def b_n (n : ℕ) : ℝ := n - 35

-- Defining the recurrence relationship of {a_n}.
def a_n : ℕ → ℕ → ℝ
| 0 _ := 0
| 1 _ := 2
| (n + 1) rec := rec + 2^n

-- Specifying the sequence of b_n/a_n.
def seq_ratio (n : ℕ) : ℝ := b_n n / a_n n n

-- Max value condition
def max_seq_ratio : Prop := seq_ratio 36 = 1 / 2^36

-- The lean theorem statement.
theorem max_value_of_seq_ratio : max_seq_ratio :=
by {
  sorry -- Proof is omitted.
}

end max_value_of_seq_ratio_l3_3164


namespace positive_divisors_30030_l3_3120

theorem positive_divisors_30030 : 
  let n := 30030
  let factorization := [(2, 1), (3, 1), (5, 1), (7, 1), (11, 1), (13, 1)]
  number_of_divisors n factorization = 64 := 
by 
  sorry

end positive_divisors_30030_l3_3120


namespace power_function_quadrants_l3_3147

theorem power_function_quadrants :
  ∃ (a : ℝ), (∀ x : ℝ, 0 < x → x ^ a > 0) ∧ (∀ x : ℝ, x < 0 → x ^ a > 0) ∧ (1 / 3) ^ a = 9 :=
begin
  use -2,
  split,
  { intros x hx,
    exact pow_pos hx (-2) },
  split,
  { intros x hx,
    exact pow_pos_of_neg hx (-2) },
  { norm_num }
end

end power_function_quadrants_l3_3147


namespace options_a_and_c_correct_l3_3729

theorem options_a_and_c_correct :
  (let a₁ := (2, 3, -1) in
   let b₁ := (-2, -3, 1) in
   let a₁_is_parallel_b₁ := a₁ = (-1 : ℝ) • b₁ in
   (let a₂ := (1, -1, 2) in
    let u₂ := (6, 4, -1) in
    let not_parallel_or_perpendicular := ¬(a₂ = (6 : ℝ) • u₂) ∧ a₂.1 * u₂.1 + a₂.2 * u₂.2 + a₂.3 * u₂.3 ≠ 0 in
    (let u₃ := (2, 2, -1) in
     let v₃ := (-3, 4, 2) in
     let dot_product_zero := u₃.1 * v₃.1 + u₃.2 * v₃.2 + u₃.3 * v₃.3 = 0 in
     (let a₄ := (0, 3, 0) in
      let u₄ := (0, -5, 0) in
      let parallel_but_not_perpendicular := a₄ = (3 / 5 : ℝ) • u₄) in
     a₁_is_parallel_b₁ ∧ dot_product_zero))) :=
sorry

end options_a_and_c_correct_l3_3729


namespace queenie_worked_4_days_l3_3658

-- Conditions
def daily_earning : ℕ := 150
def overtime_rate : ℕ := 5
def overtime_hours : ℕ := 4
def total_pay : ℕ := 770

-- Question
def number_of_days_worked (d : ℕ) : Prop := 
  daily_earning * d + overtime_rate * overtime_hours * d = total_pay

-- Theorem statement
theorem queenie_worked_4_days : ∃ d : ℕ, number_of_days_worked d ∧ d = 4 := 
by 
  use 4
  unfold number_of_days_worked 
  sorry

end queenie_worked_4_days_l3_3658


namespace percent_increase_in_combined_cost_is_correct_l3_3707

noncomputable def percentage_increase (initial final : ℝ) : ℝ :=
  ((final - initial) / initial) * 100

noncomputable def scooter_initial_cost : ℝ := 200
noncomputable def safety_gear_initial_cost : ℝ := 60
noncomputable def maintenance_kit_initial_cost : ℝ := 20

noncomputable def scooter_increase_percentage : ℝ := 0.15
noncomputable def safety_gear_increase_percentage : ℝ := 0.20
noncomputable def maintenance_kit_increase_percentage : ℝ := 0.25

noncomputable def scooter_final_cost := scooter_initial_cost * (1 + scooter_increase_percentage)
noncomputable def safety_gear_final_cost := safety_gear_initial_cost * (1 + safety_gear_increase_percentage)
noncomputable def maintenance_kit_final_cost := maintenance_kit_initial_cost * (1 + maintenance_kit_increase_percentage)

noncomputable def initial_total_cost := scooter_initial_cost + safety_gear_initial_cost + maintenance_kit_initial_cost
noncomputable def final_total_cost := scooter_final_cost + safety_gear_final_cost + maintenance_kit_final_cost

theorem percent_increase_in_combined_cost_is_correct : 
  percentage_increase initial_total_cost final_total_cost ≈ 16.79 :=
by 
  -- The proof will be provided here
  sorry

end percent_increase_in_combined_cost_is_correct_l3_3707


namespace vector_magnitude_calculation_l3_3183

theorem vector_magnitude_calculation (a b : ℝ × ℝ) (h1 : a = (3, -4)) (h2 : ∥b∥ = 2) (h3 : real.angle a b = real.pi / 3) :
  ∥(a.1 + 2 * b.1, a.2 + 2 * b.2)∥ = real.sqrt 61 := by
  sorry

end vector_magnitude_calculation_l3_3183


namespace length_HM_l3_3586

theorem length_HM (A B C M H : Type) 
  (AB BC CA : ℝ) (AM MB AH HM : ℝ)
  (triangle_ABC : triangle A B C)
  (h1 : AB = 12)
  (h2 : BC = 20)
  (h3 : CA = 16)
  (h4 : AM = 2 * MB)
  (h5 : ∃ H, is_foot (altitude A BC) H) :
  HM = 2.85 := 
sorry

end length_HM_l3_3586


namespace find_line_equation_intercepts_l3_3535

noncomputable def line_equation : Prop :=
  ∃ (k m : ℝ), 
    (∀ (x : ℝ), x intercepts analytic function definition) ∧
    (∀ (y : ℝ), y intercepts analytic function definition) ∧
    (∃ (P : ℝ × ℝ), P = (1, 2) ∧ (P belongs to line and is substituted into the equation)) ∧
    (k = 2 ∧ m = 0 ∨ equation in general form )

theorem find_line_equation_intercepts 
  (hx_eq_hy : intercepts of line l on the x-axis and y-axis are equal)
  (hP : (1,2) ∈ line) : 
  line_equation :=
by
  sorry

end find_line_equation_intercepts_l3_3535


namespace sum_of_squares_of_cosines_l3_3092

theorem sum_of_squares_of_cosines :
  (∑ k in (Finset.range 19), (Real.cos (k * 10 * Real.pi / 180))^2) = 19 / 2 :=
by
  -- Proof to be provided
  sorry

end sum_of_squares_of_cosines_l3_3092


namespace compare_volumes_l3_3340

variables (a b c : ℝ) -- dimensions of the second block
variables (a1 b1 c1 : ℝ) -- dimensions of the first block

-- Length, width, and height conditions
def length_cond := a1 = 1.5 * a
def width_cond := b1 = 0.8 * b
def height_cond := c1 = 0.7 * c

-- Volumes of the blocks
def V1 := a1 * b1 * c1 -- Volume of the first block
def V2 := a * b * c -- Volume of the second block

-- Main theorem
theorem compare_volumes (h1 : length_cond) (h2 : width_cond) (h3 : height_cond) :
  V2 = (25/21) * V1 :=
sorry

end compare_volumes_l3_3340


namespace convergence_divergence_series_l3_3489

theorem convergence_divergence_series (c : ℝ) (c_pos : 0 < c) :
  (∑' n, (n! : ℝ) / (c * n)^n).converges ↔ c > 1 / Real.exp 1 :=
sorry

end convergence_divergence_series_l3_3489


namespace exp_mul_l3_3451

variable {a : ℝ}

-- Define a theorem stating the problem: proof that a^2 * a^3 = a^5
theorem exp_mul (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end exp_mul_l3_3451


namespace transform_sequences_l3_3371

structure Domino :=
  (a : ℕ)
  (b : ℕ)

-- Define a sequence of dominoes
def Sequence := List Domino

-- Function to flip a section of dominoes
def flip_section (seq : Sequence) (start end_ : ℕ) : Sequence :=
  seq.take start ++ (seq.drop start).take (end_ - start + 1).reverse ++ (seq.drop (end_ + 1))

-- Main theorem statement
theorem transform_sequences (seq1 seq2 : Sequence) (h_identical_sets : (seq1.map id) ~ (seq2.map id))
  (h_same_endpoints : (seq1.head?.getD (Domino.mk 0 0)) = (seq2.head?.getD (Domino.mk 0 0)) 
    ∧ (seq1.reverse.head?.getD (Domino.mk 0 0)) = (seq2.reverse.head?.getD (Domino.mk 0 0))) :
  ∃ operations : List (ℕ × ℕ), (operations.foldl (λ s op, flip_section s op.1 op.2) seq2) = seq1 :=
by
  sorry

end transform_sequences_l3_3371


namespace total_carrots_computation_l3_3039

-- Definitions
def initial_carrots : ℕ := 19
def thrown_out_carrots : ℕ := 4
def next_day_carrots : ℕ := 46

def total_carrots (c1 c2 t : ℕ) : ℕ := (c1 - t) + c2

-- The statement to prove
theorem total_carrots_computation :
  total_carrots initial_carrots next_day_carrots thrown_out_carrots = 61 :=
by sorry

end total_carrots_computation_l3_3039


namespace minimum_value_expr_l3_3991

variable (a b : ℝ)

theorem minimum_value_expr (h1 : 0 < a) (h2 : 0 < b) : 
  (a + 1 / b) * (a + 1 / b - 1009) + (b + 1 / a) * (b + 1 / a - 1009) ≥ -509004.5 :=
sorry

end minimum_value_expr_l3_3991


namespace percentage_of_second_year_students_l3_3953

-- Define the conditions
def N : ℕ := 240
def A : ℕ := 423
def B : ℕ := 134
def F : ℕ := 663

-- Define the total number of second-year students
def T : ℕ := N + A - B

-- Define the percentage of second-year students in the faculty
def P : ℚ := (T : ℚ) / (F : ℚ) * 100

-- Provide the main statement we want to prove
theorem percentage_of_second_year_students :
  P ≈ 79.79 := sorry

end percentage_of_second_year_students_l3_3953


namespace find_x_l3_3599

variable (x : ℝ)
variable (s : ℝ)

-- Conditions as hypothesis
def square_perimeter_60 (s : ℝ) : Prop := 4 * s = 60
def triangle_area_150 (x s : ℝ) : Prop := (1 / 2) * x * s = 150
def height_equals_side (s : ℝ) : Prop := true

-- Proof problem statement
theorem find_x 
  (h1 : square_perimeter_60 s)
  (h2 : triangle_area_150 x s)
  (h3 : height_equals_side s) : 
  x = 20 := 
sorry

end find_x_l3_3599


namespace product_of_x_values_l3_3567

theorem product_of_x_values : (|15 / x + 4| = 3) → ∏ x in ({-15, -15 / 7}), x = 225 / 7 :=
by
  sorry

end product_of_x_values_l3_3567


namespace problem_statement_l3_3144

noncomputable def f : ℝ → ℝ := sorry -- We define "f" as a non-computable function for generality.

theorem problem_statement : (∀ x, 0 ≤ x ∧ x ≤ π/2 → f (sin x) = x) → f (1/2) = π / 6 :=
begin
  intro h,
  sorry -- Proof goes here.
end

end problem_statement_l3_3144


namespace james_out_of_pocket_cost_l3_3238

theorem james_out_of_pocket_cost (total_cost : ℝ) (coverage : ℝ) (out_of_pocket_cost : ℝ)
  (h1 : total_cost = 300) (h2 : coverage = 0.8) :
  out_of_pocket_cost = 60 :=
by
  sorry

end james_out_of_pocket_cost_l3_3238


namespace num_pos_divisors_of_30030_l3_3123

def prime_factors (n : ℕ) (factors : List ℕ) :=
  factors.prod = n ∧ factors.all prime

theorem num_pos_divisors_of_30030 :
  ∀ (n : ℕ), prime_factors n [2, 3, 5, 7, 11, 13] → 
  n = 30030 → 
  (finset.divisors n).card = 64 :=
by
  intros n h_factors h_eq
  sorry

end num_pos_divisors_of_30030_l3_3123


namespace possible_values_of_a_l3_3550

def P : Set ℝ := {x | x^2 = 1}
def M (a : ℝ) : Set ℝ := {x | a * x = 1}

theorem possible_values_of_a :
  {a | M a ⊆ P} = {1, -1, 0} :=
sorry

end possible_values_of_a_l3_3550


namespace num_pos_divisors_of_30030_l3_3125

def prime_factors (n : ℕ) (factors : List ℕ) :=
  factors.prod = n ∧ factors.all prime

theorem num_pos_divisors_of_30030 :
  ∀ (n : ℕ), prime_factors n [2, 3, 5, 7, 11, 13] → 
  n = 30030 → 
  (finset.divisors n).card = 64 :=
by
  intros n h_factors h_eq
  sorry

end num_pos_divisors_of_30030_l3_3125


namespace g_at_3_l3_3681

noncomputable def g : ℝ → ℝ := sorry

theorem g_at_3 (h : ∀ x : ℝ, g (3 ^ x) - x * g (3 ^ (-x)) = x) : g 3 = 0 :=
by
  sorry

end g_at_3_l3_3681


namespace tan_product_inequality_l3_3624

theorem tan_product_inequality (a : Fin (n + 1) → ℝ) (h₀ : ∀ i, 0 < a i ∧ a i < π / 2)
    (h₁ : ∑ i, tan (a i - π / 4) ≥ n - 1) : (∏ i, tan (a i)) ≥ n ^ (n + 1) := 
sorry

end tan_product_inequality_l3_3624


namespace square_of_1024_l3_3823

theorem square_of_1024 : 1024^2 = 1048576 :=
by
  sorry

end square_of_1024_l3_3823


namespace overlap_coordinates_l3_3409

theorem overlap_coordinates :
  ∃ m n : ℝ, 
    (m + n = 6.8) ∧ 
    ((2 * (7 + m) / 2 - 3) = (3 + n) / 2) ∧ 
    ((2 * (7 + m) / 2 - 3) = - (m - 7) / 2) :=
by
  sorry

end overlap_coordinates_l3_3409


namespace cos_2013_lt_sin_2013_l3_3089

theorem cos_2013_lt_sin_2013 : cos (2013 * real.pi / 180) < sin (2013 * real.pi / 180) := 
begin
  sorry
end

end cos_2013_lt_sin_2013_l3_3089


namespace sum_inverse_less_than_one_l3_3349

theorem sum_inverse_less_than_one (n : ℕ) (h : n ≥ 2) : 
  (∑ i in finset.range (2 * n + 1), 1 / (n + i)) < 1 := 
by
  sorry

end sum_inverse_less_than_one_l3_3349


namespace maximum_value_x_plus_reciprocal_x_l3_3004

theorem maximum_value_x_plus_reciprocal_x
  (n : ℕ) (x y : ℝ) (xs : Fin n → ℝ)
  (H1 : n = 2000)
  (H2 : ∀ i, 0 < xs i)
  (H3 : 0 < x)
  (Hsum : (∑ i, xs i) + x = 2002)
  (Hrecip_sum : (∑ i, 1 / xs i) + 1 / x = 2002) :
  x + 1 / x ≤ 8008001 / 2002 := 
begin
  sorry
end

end maximum_value_x_plus_reciprocal_x_l3_3004


namespace compute_power_of_complex_l3_3817

open Complex

noncomputable def cos_deg (θ: ℝ) : ℂ := Complex.cos (Real.pi * θ / 180)
noncomputable def sin_deg (θ: ℝ) : ℂ := Complex.sin (Real.pi * θ / 180)

theorem compute_power_of_complex :
  (cos_deg 150 + Complex.i * sin_deg 150)^30 = -1 :=
by sorry

end compute_power_of_complex_l3_3817


namespace f_inequality_l3_3636

open Real

variable {f : ℝ → ℝ}

axiom f_pos : ∀ x : ℝ, 0 < x → 0 < f(x)
axiom f_condition : ∀ x y : ℝ, 0 < x → 0 < y → f(x * y) ≤ f(x) * f(y)

theorem f_inequality (x : ℝ) (hx : 0 < x) (n : ℕ) (hn : 0 < n) :
  f(x^n) ≤ (list.prod (list.map (λ k, f(x^(k / n))) (list.range (n + 1)))) := sorry

end f_inequality_l3_3636


namespace barbara_wins_l3_3799

-- Define the game board size
def board_size := 2008

-- Define the winning conditions for Barbara
def barbara_winning_strategy (A : Matrix (Fin board_size) (Fin board_size) ℝ) : Prop :=
  (∃ i j : Fin board_size, i ≠ j ∧ A i = A j) ∨ (∃ i j : Fin board_size, i ≠ j ∧ (λ k, A k i) = (λ k, A k j))

-- The main statement to be proved in Lean
theorem barbara_wins : ∀ (turn : Fin (board_size * board_size)) (A : Matrix (Fin board_size) (Fin board_size) ℝ),
  (∀ k : Fin (board_size * board_size), k ≤ turn → (A (k / board_size) (k % board_size) = 0 ∨ ∃ n : ℝ, A (k / board_size) (k % board_size) = n)) →
   ∃ A_final : Matrix (Fin board_size) (Fin board_size) ℝ, ( (∀ k : Fin (board_size * board_size), k ≤ turn → (A_final (k / board_size) (k % board_size) = A (k / board_size) (k % board_size) ∨ True)) 
     →  det A_final = 0) :=
begin
  sorry,
end

end barbara_wins_l3_3799


namespace part1_part2_l3_3900

section
noncomputable def f (x : ℝ) (m : ℝ) : ℝ := Real.exp x - m * x^2

theorem part1 (h : f (1 : ℝ) m = 0) : m = Real.exp 1 / 2 :=
sorry

theorem part2 (n : ℕ) (h : 2 ≤ n) : 
  (∑ k in Finset.range n, 1/Real.sqrt (k+1)) < Real.log (n+1) + n * (1 - Real.log 2) :=
sorry
end

end part1_part2_l3_3900


namespace min_value_of_expression_l3_3850

noncomputable def given_expression (x : ℝ) : ℝ := 
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) + Real.sqrt ((2 + x)^2 + x^2)

theorem min_value_of_expression : ∃ x : ℝ, given_expression x = 6 * Real.sqrt 2 := 
by 
  use 0
  sorry

end min_value_of_expression_l3_3850


namespace james_out_of_pocket_cost_l3_3240

-- Definitions
def doctor_charge : ℕ := 300
def insurance_coverage_percentage : ℝ := 0.80

-- Proof statement
theorem james_out_of_pocket_cost : (doctor_charge : ℝ) * (1 - insurance_coverage_percentage) = 60 := 
by sorry

end james_out_of_pocket_cost_l3_3240


namespace range_a_if_B_subset_AU_compl_l3_3637

-- Definition of sets A and B
def U := ℝ
def A : set ℝ := {x | (x + 1) / (x - 2) < 0}
def B (a : ℝ) : set ℝ := {x | a * x - 1 > 0}

-- Complement of A in U
def AU_compl : set ℝ := {x | x ≤ -1 ∨ x ≥ 2}

-- Problem statement
theorem range_a_if_B_subset_AU_compl (a : ℝ) (h : a > 0) : B a ⊆ AU_compl ↔ (0 < a ∧ a ≤ 1/2) :=
  sorry

end range_a_if_B_subset_AU_compl_l3_3637


namespace b_n_formula_S_n_bounds_l3_3323

open Nat

def a : Nat → ℝ 
| 0       => 1  -- so that a_1 corresponds to a 2 in the mathematical problem
| (n + 1) => (2 ^ (n + 1) * a n) / ((n + 1 + 1/2) * a n + 2 ^ n)

def b (n : Nat) : ℝ := 2 ^ n / a n

theorem b_n_formula (n : Nat) : b n = (n^2 + 1) / 2 := sorry

def c (n : Nat) : ℝ := 1 / (n * (n + 1) * a (n + 1))

def S (n : Nat) : ℝ := ∑ i in range 1, n + 1, c i

theorem S_n_bounds (n : Nat) : 5/16 ≤ S n ∧ S n < 1/2 := sorry

end b_n_formula_S_n_bounds_l3_3323


namespace arm_wrestling_tournament_min_rounds_l3_3398

theorem arm_wrestling_tournament_min_rounds (N : ℕ) (hN : N = 510) :
  ∃ r : ℕ, r = 9 ∧ 
    ∀ (points : ℕ → ℕ) (meet : ℕ → ℕ → Prop),
      (∀ i, points i = 0 ∨ points i = 0) →
      (∀ i j, meet i j → |points i - points j| ≤ 1) →
      (∀ i j, meet i j → if points i < points j then points i + 1 = points i) →
      ∃ leader, leader ∈ {i | points i = max points i} := 
sorry

end arm_wrestling_tournament_min_rounds_l3_3398


namespace optimal_fruit_combination_l3_3241

structure FruitPrices :=
  (price_2_apples : ℕ)
  (price_6_apples : ℕ)
  (price_12_apples : ℕ)
  (price_2_oranges : ℕ)
  (price_6_oranges : ℕ)
  (price_12_oranges : ℕ)

def minCostFruits : ℕ :=
  sorry

theorem optimal_fruit_combination (fp : FruitPrices) (total_fruits : ℕ)
  (mult_2_or_3 : total_fruits = 15) :
  fp.price_2_apples = 48 →
  fp.price_6_apples = 126 →
  fp.price_12_apples = 224 →
  fp.price_2_oranges = 60 →
  fp.price_6_oranges = 164 →
  fp.price_12_oranges = 300 →
  minCostFruits = 314 :=
by
  sorry

end optimal_fruit_combination_l3_3241


namespace proof_result_l3_3732

noncomputable def direction_vector (a b : ℝ × ℝ × ℝ) := 
  ∃ k : ℝ, (k • b) = a

noncomputable def perpendicular_vector (a b : ℝ × ℝ × ℝ) := 
  (a.1 * b.1 + a.2 * b.2 + a.3 * b.3) = 0

def parallel_line_condition := 
  direction_vector (2, 3, -1) (-2, -3, 1) 

def perpendicular_plane_condition := 
  perpendicular_vector (2, 2, -1) (-3, 4, 2)

theorem proof_result: parallel_line_condition ∧ perpendicular_plane_condition :=
by
  have h1 : parallel_line_condition := sorry
  have h2 : perpendicular_plane_condition := sorry
  exact ⟨h1, h2⟩

end proof_result_l3_3732


namespace ratio_of_area_l3_3957

noncomputable def area_of_triangle_ratio (AB CD height : ℝ) (h : CD = 2 * AB) : ℝ :=
  let ABCD_area := (AB + CD) * height / 2
  let EAB_area := ABCD_area / 3
  EAB_area / ABCD_area

theorem ratio_of_area (AB CD : ℝ) (height : ℝ) (h1 : AB = 10) (h2 : CD = 20) (h3 : height = 5) : 
  area_of_triangle_ratio AB CD height (by rw [h1, h2]; ring) = 1 / 3 :=
sorry

end ratio_of_area_l3_3957


namespace estimated_value_of_y_l3_3178

theorem estimated_value_of_y (x : ℝ) (h : x = 25) : 
  let y := 0.50 * x - 0.81 in
  y = 11.69 :=
by
  rw [h]
  let y := 0.50 * 25 - 0.81
  sorry

end estimated_value_of_y_l3_3178


namespace isabel_paper_left_l3_3960

theorem isabel_paper_left :
  ∀ (bought used : ℕ), bought = 900 → used = 156 → (bought - used = 744) :=
by
  intros bought used h_bought h_used
  rw [h_bought, h_used]
  rfl

end isabel_paper_left_l3_3960


namespace parabola_focus_coords_l3_3673

-- Define the parabola equation
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the focus coordinates
def focus (x y : ℝ) : Prop := (x, y) = (1, 0)

-- The math proof problem statement
theorem parabola_focus_coords :
  ∀ x y, parabola x y → focus x y :=
by
  intros x y hp
  sorry

end parabola_focus_coords_l3_3673


namespace pencils_on_desk_l3_3693

theorem pencils_on_desk (pencils_in_drawer pencils_on_desk_initial pencils_total pencils_placed : ℕ)
  (h_drawer : pencils_in_drawer = 43)
  (h_desk_initial : pencils_on_desk_initial = 19)
  (h_total : pencils_total = 78) :
  pencils_placed = 16 := by
  sorry

end pencils_on_desk_l3_3693


namespace number_of_valid_ordered_pairs_l3_3057

theorem number_of_valid_ordered_pairs (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : b > a) 
(h4 : (a - 4) * (b - 4) * 3 = a * b) : 
∃ n, n = 3 ∧ (n = card { (a, b) | ∃ (a_ b_ : ℕ), a_ > 0 ∧ b_ > 0 ∧ b_ > a_ ∧ (a - 4) * (b - 4) * 3 = a_ * b_ } ) :=
sorry

end number_of_valid_ordered_pairs_l3_3057


namespace constant_term_expansion_l3_3226

theorem constant_term_expansion : 
    (∃ (C : ℚ), ∀ (x y : ℚ), 
    (∑ i j k : ℕ in (x^4 + y^2 + 1/(2*x*y))^7) = C) 
    → C = 105/16 := by
  sorry

end constant_term_expansion_l3_3226


namespace initial_ratio_milk_water_l3_3210

theorem initial_ratio_milk_water (M W : ℕ) 
  (h1 : M + W = 45) 
  (h2 : M = 3 * (W + 3)) 
  : M / W = 4 := 
sorry

end initial_ratio_milk_water_l3_3210


namespace number_of_divisors_30030_l3_3126

theorem number_of_divisors_30030 : 
  let n := 30030 in 
  let prime_factors := [2, 3, 5, 7, 11, 13] in
  (∀ p ∈ prime_factors, nat.prime p) → 
  (∀ p ∈ prime_factors, n % p = 0) → 
  (∀ p₁ p₂ ∈ prime_factors, p₁ ≠ p₂ → ∀ m, n % (p₁ * p₂ * m) ≠ 0) →
  (∀ p ∈ prime_factors, ∃ m, n = p ^ 1 * m ∧ nat.gcd(p, m) = 1) → 
  ∃ t, t = 64 ∧ t = ∏ p in prime_factors.to_finset, ((1 : ℕ) + 1) :=
by
  intro n prime_factors hprimes hdivisors hunique hfactored
  use 64
  rw list.prod_eq_foldr at *
  suffices : 64 = list.foldr (λ _ r, 2 * r) 1 prime_factors, from this
  simp [list.foldr, prime_factors]
  sorry

end number_of_divisors_30030_l3_3126


namespace slices_left_l3_3643

variable (total_pieces: ℕ) (joe_fraction: ℚ) (darcy_fraction: ℚ)
variable (carl_fraction: ℚ) (emily_fraction: ℚ)

theorem slices_left 
  (h1 : total_pieces = 24)
  (h2 : joe_fraction = 1/3)
  (h3 : darcy_fraction = 1/4)
  (h4 : carl_fraction = 1/6)
  (h5 : emily_fraction = 1/8) :
  total_pieces - (total_pieces * joe_fraction + total_pieces * darcy_fraction + total_pieces * carl_fraction + total_pieces * emily_fraction) = 3 := 
  by 
  sorry

end slices_left_l3_3643


namespace calf_rope_length_l3_3786

noncomputable def new_rope_length (initial_length : ℝ) (additional_area : ℝ) : ℝ :=
  let A1 := Real.pi * initial_length ^ 2
  let A2 := A1 + additional_area
  let new_length_squared := A2 / Real.pi
  Real.sqrt new_length_squared

theorem calf_rope_length :
  new_rope_length 12 565.7142857142857 = 18 := by
  sorry

end calf_rope_length_l3_3786


namespace number_of_solutions_l3_3488

noncomputable theory
open Complex

def complex_number_satisfying_conditions (z : ℂ) : Prop :=
  abs z = 1 ∧ abs ((z / conj z) + (conj z / z)) = 1

theorem number_of_solutions : 
  ∃ (solutions : set ℂ), 
    (∀ z ∈ solutions, complex_number_satisfying_conditions z) ∧ 
    solutions.to_finset.card = 8 :=
sorry

end number_of_solutions_l3_3488


namespace solve_system_l3_3665

theorem solve_system :
  ∃ a b c : ℝ, 
    a = 1 ∧ b = 1 ∧ c = 1 ∧ 
    (a^3 + 3*a*b^2 + 3*a*c^2 - 6*a*b*c = 1) ∧
    (b^3 + 3*b*a^2 + 3*b*c^2 - 6*a*b*c = 1) ∧
    (c^3 + 3*c*a^2 + 3*c*b^2 - 6*a*b*c = 1) :=
begin
  use 1,
  use 1,
  use 1,
  split,
  { refl },
  split,
  { refl },
  split,
  { refl },
  simp,
  split,
  { ring },
  split,
  { ring },
  { ring }
end

end solve_system_l3_3665


namespace Berengere_contribution_l3_3809

theorem Berengere_contribution (cake_cost_in_euros : ℝ) (emily_dollars : ℝ) (exchange_rate : ℝ)
  (h1 : cake_cost_in_euros = 6)
  (h2 : emily_dollars = 5)
  (h3 : exchange_rate = 1.25) :
  cake_cost_in_euros - emily_dollars * (1 / exchange_rate) = 2 := by
  sorry

end Berengere_contribution_l3_3809


namespace total_pages_read_proof_l3_3289

def reading_pages_total (lit_time_total : ℕ) (lit_speed : ℕ) (hist_time_total : ℕ) (hist_speed : ℕ)
    (lit_actual : ℕ) (hist_actual : ℕ) (sci_actual : ℕ) (sci_speed : ℕ) : ℕ :=
  let lit_pages := lit_actual * 60 / lit_speed in
  let hist_pages := hist_actual * 60 / hist_speed in
  let sci_pages := sci_actual * 60 / sci_speed in
  lit_pages + hist_pages + sci_pages

theorem total_pages_read_proof : 
  reading_pages_total 3 15 3 10 1.5 1 0.5 20 = 13 :=
by
  -- ( 1.5 * 60 / 15 ) + ( 1 * 60 / 10 ) + ( 0.5 * 60 / 20 ) = 13
  sorry

end total_pages_read_proof_l3_3289


namespace acute_angle_sum_l3_3266

theorem acute_angle_sum (n : ℕ) (hn : n ≥ 4) (M m: ℕ) 
  (hM : M = 3) (hm : m = 0) : M + m = 3 := 
by 
  sorry

end acute_angle_sum_l3_3266


namespace count_valid_arrangements_l3_3838

-- Definitions for individuals and constraints
inductive Person : Type
| A
| B
| C
| D
| E
| F

open Person

def is_adjacent (p1 p2 : Person) (arrangement : List Person) : Prop :=
  ∃ i : ℕ, i < arrangement.length - 1 ∧ arrangement.nth i = some p1 ∧ arrangement.nth (i + 1) = some p2

def valid_arrangement (arrangement : List Person) : Prop :=
  arrangement.length = 6 ∧
  ∀ p, p ∈ arrangement → p ∈ [A, B, C, D, E, F] ∧
  ¬ is_adjacent A B arrangement ∧
  ¬ is_adjacent B A arrangement ∧
  ¬ is_adjacent C D arrangement ∧
  ¬ is_adjacent D C arrangement

-- The theorem to be proven
theorem count_valid_arrangements : 
  (finset.univ.filter valid_arrangement).card = 336 := by
  sorry

end count_valid_arrangements_l3_3838


namespace range_of_a_l3_3549

noncomputable def setA : set ℝ := { x | -4 < x ∧ x ≤ 2 }
noncomputable def setB (a : ℝ) : set ℝ := { x | (x - a) * (x - 2 * a + 1) ≤ 0 }

theorem range_of_a (a : ℝ) (H : setB a ⊆ setA) : -3 / 2 < a ∧ a ≤ 3 / 2 :=
sorry

end range_of_a_l3_3549


namespace quadratic_limit_l3_3430

noncomputable def quadratic_sequence (a b : ℝ) : ℕ → ℝ
| 0 => a
| 1 => b
| n+2 => if h : quadratic_sequence a b (n+1) = 0 then 0 else
  let p := quadratic_sequence a b n in
  let q := quadratic_sequence a b (n+1) in
  if p <= q then - (p + q) else p * q

theorem quadratic_limit (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) : 
  ∀ n > 5, |quadratic_sequence a b n| = 0 :=
begin
  sorry,
end

end quadratic_limit_l3_3430


namespace minimum_force_is_0_06_n_l3_3372

noncomputable def minimum_force_to_submerge_cube 
  (V : ℝ) (ρ_cube : ℝ) (ρ_water : ℝ) (g : ℝ) : ℝ :=
  let V_m3 := V * 10^(-6)  -- Convert cm^3 to m^3
  let m_cube := ρ_cube * V_m3  -- Mass of cube in kg
  let F_g := m_cube * g  -- Gravitational force in N
  let F_b := ρ_water * V_m3 * g  -- Buoyant force in N
  let F_ext := F_b - F_g  -- External force required in N
  F_ext

theorem minimum_force_is_0_06_n 
  (V : ℝ := 10) 
  (ρ_cube : ℝ := 400) 
  (ρ_water : ℝ := 1000) 
  (g : ℝ := 10) :
  minimum_force_to_submerge_cube V ρ_cube ρ_water g = 0.06 := 
by 
  sorry

end minimum_force_is_0_06_n_l3_3372


namespace positive_divisors_30030_l3_3121

theorem positive_divisors_30030 : 
  let n := 30030
  let factorization := [(2, 1), (3, 1), (5, 1), (7, 1), (11, 1), (13, 1)]
  number_of_divisors n factorization = 64 := 
by 
  sorry

end positive_divisors_30030_l3_3121


namespace sum_first_2014_terms_l3_3000

-- Definitions and initial conditions
def sequence (n : ℕ) : ℤ :=
  if n = 0 then 0 else if n = 1 then 1 else if n = 2 then 2
  else sequence (n - 1) - sequence (n - 2)

def sum_sequence (n : ℕ) : ℤ :=
  (Finset.range n).sum sequence

theorem sum_first_2014_terms :
  sum_sequence 2014 = 3 :=
sorry

end sum_first_2014_terms_l3_3000


namespace no_n_in_range_l3_3486

theorem no_n_in_range :
  ¬ ∃ n : ℤ, 10 ≤ n ∧ n ≤ 15 ∧ n % 7 = 10467 % 7 := by
  sorry

end no_n_in_range_l3_3486


namespace theta_is_in_second_quadrant_l3_3994

theorem theta_is_in_second_quadrant (θ : ℝ) (hθ : π < θ ∧ θ < 3 * π / 2) 
(hcos : |cos (θ / 2)| = -cos (θ / 2)) : π / 2 < θ / 2 ∧ θ / 2 < π :=
by
  sorry

end theta_is_in_second_quadrant_l3_3994


namespace discount_percentage_is_ten_l3_3778

-- Definitions based on given conditions
def cost_price : ℝ := 42
def markup (S : ℝ) : ℝ := 0.30 * S
def selling_price (S : ℝ) : Prop := S = cost_price + markup S
def profit : ℝ := 6

-- To prove the discount percentage
theorem discount_percentage_is_ten (S SP : ℝ) 
  (h_sell_price : selling_price S) 
  (h_SP : SP = S - profit) : 
  ((S - SP) / S) * 100 = 10 := 
by
  sorry

end discount_percentage_is_ten_l3_3778


namespace min_colors_tessellation_l3_3096

/-
We need to prove that the minimum number of colors needed to shade a tessellation,
composed of alternating rows of hexagons and triangles, ensuring no two tiles
sharing a side are the same color, is 3. 
-/

theorem min_colors_tessellation : 
  (∃ (colors : Finset ℕ), (∀ (hexagon_neighbors : Finset (Finset ℕ)), 
    hexagon_neighbors.card = 6 → 
    ∀ triangle_neighbors (n : ℕ), 
    triangle_neighbors.card = 3 → 
    ∀ tile ∈ colors, 
    ∀ neighbor ∈ triangle_neighbors, 
    neighbor ≠ tile →^tile != neighbor) ∧ colors.card = 3) :=
sorry

end min_colors_tessellation_l3_3096


namespace boat_speed_ratio_l3_3765

theorem boat_speed_ratio (speed_still_water : ℝ) (current_speed : ℝ) (distance : ℝ) :
  speed_still_water = 20 ∧ current_speed = 8 ∧ distance = 10 →
  (let downstream_speed := speed_still_water + current_speed in
   let upstream_speed := speed_still_water - current_speed in
   let time_downstream := distance / downstream_speed in
   let time_upstream := distance / upstream_speed in
   let total_time := time_downstream + time_upstream in
   let total_distance := 2 * distance in
   let average_speed := total_distance / total_time in
   average_speed / speed_still_water = 42 / 65) :=
begin
  intros h,
  rcases h with ⟨hsw, hcs, hd⟩,
  rw [hsw, hcs, hd],
  let downstream_speed := 20 + 8,
  let upstream_speed := 20 - 8,
  let time_downstream := 10 / downstream_speed,
  let time_upstream := 10 / upstream_speed,
  let total_time := time_downstream + time_upstream,
  let total_distance := 2 * 10,
  let average_speed := total_distance / total_time,
  have h_average_speed : average_speed = 840 / 65,
  { sorry }, -- Proof to substantiate this claim
  rw h_average_speed,
  norm_num,
end

end boat_speed_ratio_l3_3765


namespace abs_diff_base_6_l3_3482

theorem abs_diff_base_6 (A B : ℕ) (hA : A < 6) (hB : B < 6)
  (eq1 : A_6 = 6) (eq2 : B_6 = 5) :
  (|A - B|_6) = 1 :=
by
  sorry

end abs_diff_base_6_l3_3482


namespace DivisionExpressionCannotHaveZeroDivisor_l3_3600

-- Define the context in which Δ ÷ ☆ makes sense
structure DivisionExpression where
  Δ : ℝ
  ☆ : ℝ

-- Define the condition that the expression is meaningless if ☆ is 0
def expressionMeaninglessIfZero (e : DivisionExpression) : Prop :=
  e.☆ = 0

-- The statement to be proved: "If the expression is meaningless when the divisor is 0, 
-- then it is necessarily true that the divisor cannot be 0."
theorem DivisionExpressionCannotHaveZeroDivisor (e : DivisionExpression) :
  ¬expressionMeaninglessIfZero e → e.☆ ≠ 0 :=
by
  sorry

end DivisionExpressionCannotHaveZeroDivisor_l3_3600


namespace not_power_of_two_l3_3934

theorem not_power_of_two (S : Finset ℕ) (h1 : ∀ n, n ∈ S → 11111 ≤ n ∧ n ≤ 99999) :
  ∀ (l : List ℕ), (∀ n, n ∈ l → n ∈ S) → 
  ¬(∃ k, 2^k = l.foldr (λ x y, x * 10^5 + y) 0) :=
by { sorry }

end not_power_of_two_l3_3934


namespace correct_intersection_l3_3909

-- Define the universal set
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 1}

-- Define set B
def B : Set ℝ := {x : ℝ | x < -2 ∨ x > 2}

-- Define the complement of B in U
def complement_B : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}

-- Define the intersection we are interested in
def intersection := A ∩ complement_B

-- Theorem statement of what we need to prove
theorem correct_intersection : A ∩ complement_B = {x : ℝ | -2 ≤ x ∧ x ≤ 1} :=
by
  sorry

end correct_intersection_l3_3909


namespace PB_value_l3_3648

theorem PB_value 
  (A B P : Point) 
  (h : is_perpendicular_bisector A B P) 
  (h_PA : dist P A = 3) : 
  dist P B = 3 := 
by 
  sorry

end PB_value_l3_3648


namespace intersection_of_A_and_B_l3_3270

theorem intersection_of_A_and_B :
  let A := {1, 2, 3}
  let B := {2, 4}
  A ∩ B = {2} :=
by
  sorry

end intersection_of_A_and_B_l3_3270


namespace find_abc_l3_3129

noncomputable def p_n (n : ℕ) : ℚ := (10^n - 1) / 9

theorem find_abc :
  (∀ n : ℕ, (a : ℚ) (b : ℚ) (c : ℚ),
    (a, b, c) = (0, 0, 0) ∨ (a, b, c) = (1, 5, 3) ∨ (a, b, c) = (4, 8, 6) ∨ (a, b, c) = (9, 9, 9) →
    a * p_n n + b * p_n n + 1 = (c * p_n n + 1)^2) :=
  sorry

end find_abc_l3_3129


namespace closest_integer_to_10_minus_sqrt_12_l3_3071

theorem closest_integer_to_10_minus_sqrt_12 (a b c d : ℤ) (h_a : a = 4) (h_b : b = 5) (h_c : c = 6) (h_d : d = 7) :
  d = 7 :=
by
  sorry

end closest_integer_to_10_minus_sqrt_12_l3_3071


namespace isosceles_right_triangle_hyperbola_l3_3217

theorem isosceles_right_triangle_hyperbola :
  ∀ (A B C F : Point) (D : Hyperbola) (a : ℝ),
  is_isosceles_right_triangle A B C ∧
  ∠A = 90 ∧
  on_hyperbola A D ∧
  on_hyperbola B D ∧
  on_focus F D ∧
  passes_through AB F ∧
  is_left_focus C D →
  ( |AF| / |BF| = sqrt 2 - 1 ) :=
by
  sorry

end isosceles_right_triangle_hyperbola_l3_3217


namespace nested_g_3_l3_3572

def g (x : ℝ) : ℝ := -1 / x^2

theorem nested_g_3 :
  g (g (g (g (g 3)))) = -1 / (43046721^2) := by
  sorry

end nested_g_3_l3_3572


namespace area_enclosed_by_line_and_parabola_l3_3484

theorem area_enclosed_by_line_and_parabola :
  let f := λ x : ℝ, 2 * x + 3
  let g := λ x : ℝ, x ^ 2
  (∫ x in -1..3, f x - g x) = 32 / 3 := 
by
  let f := λ x : ℝ, 2 * x + 3
  let g := λ x : ℝ, x ^ 2
  have h : ∫ x in -1..3, f x - g x = 32 / 3 := sorry
  exact h

end area_enclosed_by_line_and_parabola_l3_3484


namespace hyperbola_from_ellipse_l3_3745

noncomputable def ellipse_foci (a b : ℝ) : (ℝ × ℝ) := (sqrt (a^2 - b^2), 0)

noncomputable def hyperbola_equation (a b : ℝ) (hyperbola_eccentricity : ℝ) : ℝ :=
  let foci_distance := sqrt (a^2 - b^2)
  let vertices_a := foci_distance / hyperbola_eccentricity
  let vertices_b := sqrt (foci_distance^2 - vertices_a^2)
  in (x : ℝ) × (y : ℝ) × (ℝ) := ((x^2 / vertices_a^2) - (y^2 / vertices_b^2) - 1)

theorem hyperbola_from_ellipse (a b hyperbola_eccentricity : ℝ) :
  a = 5 → b = 4 → hyperbola_eccentricity = 2 →
  hyperbola_equation 5 4 2 =
    ((x : ℝ) × (y : ℝ) × (ℝ) := ((x^2 / 9) - (y^2 / 27) - 1)) :=
begin
  intros ha hb he,
  rw [ha, hb, he],
  -- The proof should go here.
  sorry
end

end hyperbola_from_ellipse_l3_3745


namespace wall_building_time_l3_3232

theorem wall_building_time
  (m1 m2 : ℕ) 
  (d1 d2 : ℝ)
  (h1 : m1 = 20)
  (h2 : d1 = 3.0)
  (h3 : m2 = 30)
  (h4 : ∃ k, m1 * d1 = k ∧ m2 * d2 = k) :
  d2 = 2.0 :=
by
  sorry

end wall_building_time_l3_3232


namespace number_of_solutions_abs_eq_five_l3_3118

theorem number_of_solutions_abs_eq_five : 
  ((setOf (λ x : ℝ, abs (x - abs (3 * x + 2)) = 5)).finite ∧ 
   (setOf (λ x : ℝ, abs (x - abs (3 * x + 2)) = 5)).to_finset.card = 2) :=
by
  -- The formulation states that the set of solutions to the given equation is finite, and the cardinality of its finite set is 2.
  sorry

end number_of_solutions_abs_eq_five_l3_3118


namespace max_value_a_plus_2b_l3_3533

theorem max_value_a_plus_2b {a b : ℝ} (h_positive : 0 < a ∧ 0 < b) (h_eqn : a^2 + 2 * a * b + 4 * b^2 = 6) :
  a + 2 * b ≤ 2 * Real.sqrt 2 :=
sorry

end max_value_a_plus_2b_l3_3533


namespace part1_part2_part3_l3_3221

-- Definition of a companion point
structure Point where
  x : ℝ
  y : ℝ

def isCompanion (P Q : Point) : Prop :=
  Q.x = P.x + 2 ∧ Q.y = P.y - 4

-- Part (1) proof statement
theorem part1 (P Q : Point) (hPQ : isCompanion P Q) (hP : P = ⟨2, -1⟩) (hQ : Q.y = -20 / Q.x) : Q.x = 4 ∧ Q.y = -5 ∧ -20 / 4 = -5 :=
  sorry

-- Part (2) proof statement
theorem part2 (P Q : Point) (hPQ : isCompanion P Q) (hPLine : P.y = P.x - (-5)) (hQ : Q = ⟨-1, -2⟩) : P.x = -3 ∧ P.y = -3 - (-5) ∧ Q.x = -1 ∧ Q.y = -2 :=
  sorry

-- Part (3) proof statement
noncomputable def line2 (Q : Point) := 2*Q.x - 5

theorem part3 (P Q : Point) (hPQ : isCompanion P Q) (hP : P.y = 2*P.x + 3) (hQLine : Q.y = line2 Q) : line2 Q = 2*(P.x + 2) - 5 :=
  sorry

end part1_part2_part3_l3_3221


namespace ratio_volumes_tetrahedron_octahedron_l3_3855

theorem ratio_volumes_tetrahedron_octahedron (a b : ℝ) (h_eq_areas : a^2 * (Real.sqrt 3) = 2 * b^2 * (Real.sqrt 3)) :
  (a^3 * (Real.sqrt 2) / 12) / (b^3 * (Real.sqrt 2) / 3) = 1 / Real.sqrt 2 :=
by
  sorry

end ratio_volumes_tetrahedron_octahedron_l3_3855


namespace perfect_squares_multiple_of_72_number_of_perfect_squares_multiple_of_72_l3_3189

theorem perfect_squares_multiple_of_72 (N : ℕ) : 
  (N^2 < 1000000) ∧ (N^2 % 72 = 0) ↔ N ≤ 996 :=
sorry

theorem number_of_perfect_squares_multiple_of_72 : 
  ∃ upper_bound : ℕ, upper_bound = 83 ∧ ∀ n : ℕ, (n < 1000000) → (n % 144 = 0) → n ≤ (12 * upper_bound) :=
sorry

end perfect_squares_multiple_of_72_number_of_perfect_squares_multiple_of_72_l3_3189


namespace magnitude_of_angle_A_value_of_side_b_l3_3607

open Real

-- Definitions based on given problem
def sides_opposite_equal (A B C a b c : ℝ) : Prop := true -- Placeholder condition

-- Conditions for part I
def vectors_perpendicular (m n : ℝ × ℝ) : Prop :=
  m.1 * n.1 + m.2 * n.2 = 0

def m_vector (A : ℝ) := (sqrt 3, cos A + 1)
def n_vector (A : ℝ) := (sin A, -1)

-- Proving the magnitude of angle A
theorem magnitude_of_angle_A (A : ℝ) :
  vectors_perpendicular (m_vector A) (n_vector A) →
  A = π / 3 :=
by sorry

-- Conditions for part II
def law_of_sines (a A b B : ℝ) : Prop :=
  a / sin A = b / sin B

-- Additional Given conditions
def given_cos_B (B : ℝ) : Prop := cos B = sqrt 3 / 3

-- Proving value of side b
theorem value_of_side_b (a A cos_B b B : ℝ) :
  a = 2 →
  given_cos_B B →
  A = π / 3 →
  sin B = sqrt 6 / 3 →
  law_of_sines a A b B →
  b = 4 * sqrt 2 / 3 :=
by sorry

end magnitude_of_angle_A_value_of_side_b_l3_3607


namespace min_segments_required_l3_3382

theorem min_segments_required (side_length diagonal_length : ℝ) : 
  ∃ (segments : ℕ), 
  (segments = 4 ∧ 
   ∃ (line : list ℝ), 
     (∀ seg ∈ line, seg ≤ side_length ∨ seg ≤ diagonal_length) ∧ 
     (∑ seg in line.filter (λ seg, seg ≤ side_length), seg = side_length) ∧
     (∑ seg in line.filter (λ seg, seg > side_length), seg = diagonal_length)) ∧
  -- Ensuring this broken line divides the square into two equal areas
  (area_division (side_length:ℝ) (diagonal_length:ℝ) (line:list ℝ) = true) :=
-- The actual proof is omitted
sorry

end min_segments_required_l3_3382


namespace probability_drawing_red_l3_3805

/-- The probability of drawing a red ball from a bag that contains 1 red ball and 2 yellow balls. -/
theorem probability_drawing_red : 
  let N_red := 1
  let N_yellow := 2
  let N_total := N_red + N_yellow
  let P_red := (N_red : ℝ) / N_total
  P_red = (1 : ℝ) / 3 :=
by {
  sorry
}

end probability_drawing_red_l3_3805


namespace emilys_number_l3_3109

theorem emilys_number : ∃ n : ℕ, 250 ∣ n ∧ 60 ∣ n ∧ 1000 < n ∧ n < 4000 ∧ n = 3000 := by
  exists 3000
  sorry

end emilys_number_l3_3109


namespace math_problem_l3_3360

theorem math_problem :
  (2^0 - 1 + 5^2 - 0)⁻¹ * 5 = 1 / 5 :=
by
  sorry

end math_problem_l3_3360


namespace vector_perpendicular_l3_3555

def vec_a : ℝ × ℝ := (-1, 2)
def vec_b : ℝ × ℝ := (1, 3)
def vec_diff : ℝ × ℝ := (vec_a.1 - vec_b.1, vec_a.2 - vec_b.2)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem vector_perpendicular :
  dot_product vec_a vec_diff = 0 := by
  sorry

end vector_perpendicular_l3_3555


namespace solution_set_l3_3269

noncomputable def f : ℝ → ℝ := sorry

axiom deriv_f_pos (x : ℝ) : deriv f x > 1 - f x
axiom f_at_zero : f 0 = 3

theorem solution_set (x : ℝ) : e^x * f x > e^x + 2 ↔ x > 0 :=
by sorry

end solution_set_l3_3269


namespace tangent_at_origin_l3_3680

noncomputable def f (a x : ℝ) : ℝ := x^3 + a * x

theorem tangent_at_origin (a : ℝ) (h_extremum : (deriv (f a) 1 = 0)) :
  tangent_line_eq (f a) 0 = 3 * x + y := by
sorry

end tangent_at_origin_l3_3680


namespace problem_statement_l3_3626

noncomputable def f (x : ℝ) : ℝ := sorry -- define f
noncomputable def g (x : ℝ) : ℝ := sorry -- define g

theorem problem_statement :
  (∀ x y : ℝ, f(x + y) + f(x - y) = 2 * f(x) * g(y)) →
  f(0) = 0 →
  (∃ x : ℝ, f(x) ≠ 0) →
  (∀ x : ℝ, f(-x) = -f(x)) ∧ (∀ x : ℝ, g(-x) = g(x)) :=
begin
  intros h1 h2 h3,
  split,
  { -- prove f is odd
    sorry },
  { -- prove g is even
    sorry }
end

end problem_statement_l3_3626


namespace most_likely_maximum_people_in_room_l3_3044

theorem most_likely_maximum_people_in_room :
  ∃ k, 1 ≤ k ∧ k ≤ 3000 ∧
    (∃ p : ℕ → ℕ → ℕ → ℕ, (p 1000 1000 1000) = 1019) ∧
    (∀ a b c : ℕ, a + b + c = 3000 → a ≤ 1019 ∧ b ≤ 1019 ∧ c ≤ 1019 → max a (max b c) = 1019) :=
sorry

end most_likely_maximum_people_in_room_l3_3044


namespace qinJiushao_value_l3_3679

/-- A specific function f(x) with given a and b -/
def f (x : ℤ) : ℤ :=
  x^5 + 47 * x^4 - 37 * x^2 + 1

/-- Qin Jiushao algorithm to find V3 at x = -1 -/
def qinJiushao (x : ℤ) : ℤ :=
  let V0 := 1
  let V1 := V0 * x + 47
  let V2 := V1 * x + 0
  let V3 := V2 * x - 37
  V3

theorem qinJiushao_value :
  qinJiushao (-1) = 9 :=
by
  sorry

end qinJiushao_value_l3_3679


namespace determine_sixth_face_l3_3331

-- Define a cube configuration and corresponding functions
inductive Color
| black
| white

structure Cube where
  faces : Fin 6 → Fin 9 → Color

noncomputable def sixth_face_color (cube : Cube) : Fin 9 → Color := sorry

-- The statement of the theorem proving the coloring of the sixth face
theorem determine_sixth_face (cube : Cube) : 
  (exists f : (Fin 9 → Color), f = sixth_face_color cube) := 
sorry

end determine_sixth_face_l3_3331


namespace infinite_sequence_even_numbers_l3_3657

def sequence (n : ℕ) : ℕ :=
  if n = 0 then 2 else 2 ^ sequence (n - 1) + 2

lemma seq_even (k : ℕ) : ∃ n : ℕ, even (sequence n) :=
begin
  use n,
  unfold sequence,
  sorry
end

lemma seq_divisibility (k : ℕ) : ∃ n : ℕ, sequence n ∣ (2 ^ sequence n + 2) ∧ (sequence n - 1) ∣ (2 ^ sequence n + 1) :=
begin
  use n,
  unfold sequence,
  sorry
end

theorem infinite_sequence_even_numbers : ∃ (n : ℕ) (k : ℕ), 
  (∀ k, n < sequence (n + k)) 
  ∧ ∀ k, sequence (n + k) ∣ (2 ^ sequence (n + k) + 2) ∧ (sequence (n + k) - 1) ∣ (2 ^ sequence (n + k) + 1) :=
begin
  use 0,
  use 1,
  intros k,
  split,
  { sorry },
  { sorry }
end

end infinite_sequence_even_numbers_l3_3657


namespace points_difference_l3_3446

-- Define the given data
def points_per_touchdown : ℕ := 7
def brayden_gavin_touchdowns : ℕ := 7
def cole_freddy_touchdowns : ℕ := 9

-- Define the theorem to prove the difference in points
theorem points_difference :
  (points_per_touchdown * cole_freddy_touchdowns) - 
  (points_per_touchdown * brayden_gavin_touchdowns) = 14 :=
  by sorry

end points_difference_l3_3446


namespace find_hyperbola_equation_l3_3174

noncomputable def hyperbola_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

theorem find_hyperbola_equation :
  ∃ (a b : ℝ), a = 2 ∧ b = real.sqrt 5 ∧ hyperbola_equation a b x y :=
begin
  -- Conditions provided:
  -- 1. Asymptote y = (sqrt 5 / 2) * x
  -- 2. Common focus with the ellipse x^2 / 12 + y^2 / 3 = 1
  -- 3. Foci of the ellipse are (±3, 0), hence c = 3.
  sorry
end

end find_hyperbola_equation_l3_3174


namespace alarm_system_codes_count_l3_3752

theorem alarm_system_codes_count :
  ∃ n : ℕ, n = (∏ i in finset.range(5), (10 - i)) + 
             (10 * (nat.choose 5 2) * 9 * 8 * 7) + 
             (10 * 9 * (nat.choose 5 2) * (nat.choose 3 2) * 8) ∧ 
             n = 102240 :=
by {
  sorry
}

end alarm_system_codes_count_l3_3752


namespace smallest_value_l3_3385

theorem smallest_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  ∃ (v : ℝ), (∀ x y : ℝ, 0 < x → 0 < y → v ≤ (16 / x + 108 / y + x * y)) ∧ v = 36 :=
sorry

end smallest_value_l3_3385


namespace geometric_mean_of_4_and_9_l3_3312

theorem geometric_mean_of_4_and_9 : ∃ G : ℝ, (4 / G = G / 9) ∧ (G = 6 ∨ G = -6) := 
by
  sorry

end geometric_mean_of_4_and_9_l3_3312


namespace melanie_sold_4_gumballs_l3_3640

theorem melanie_sold_4_gumballs (price_per_gumball total_money : ℕ) 
  (h1 : price_per_gumball = 8) 
  (h2 : total_money = 32) : 
  total_money / price_per_gumball = 4 := 
by 
  rw [h1, h2] 
  norm_num
  sorry

end melanie_sold_4_gumballs_l3_3640


namespace reciprocal_sum_of_products_roots_l3_3256

theorem reciprocal_sum_of_products_roots 
  (p q r s : ℚ) 
  (h_roots : (Polynomial.C (5 : ℚ) * Polynomial.X ^ 0 + 
              Polynomial.C (7 : ℚ) * Polynomial.X ^ 1 + 
              Polynomial.C (11 : ℚ) * Polynomial.X ^ 2 + 
              Polynomial.C (6 : ℚ) * Polynomial.X ^ 3 + 
              Polynomial.C (1 : ℚ) * Polynomial.X ^ 4).roots = 
              {(p, 1), (q, 1), (r, 1), (s, 1)}) :
  (\frac{1}{p * q} + \frac{1}{p * r} + \frac{1}{p * s} + 
   \frac{1}{q * r} + \frac{1}{q * s} + \frac{1}{r * s}) = 
   \frac{11}{5} :=
by sorry

end reciprocal_sum_of_products_roots_l3_3256


namespace roots_geometric_progression_condition_l3_3688

theorem roots_geometric_progression_condition 
  (a b c : ℝ) 
  (x1 x2 x3 : ℝ)
  (h1 : x1 + x2 + x3 = -a)
  (h2 : x1 * x2 + x2 * x3 + x1 * x3 = b)
  (h3 : x1 * x2 * x3 = -c)
  (h4 : x2^2 = x1 * x3) :
  a^3 * c = b^3 :=
sorry

end roots_geometric_progression_condition_l3_3688


namespace building_height_is_correct_l3_3718

noncomputable def height_of_building : real :=
  let shadow_building := 120 -- shadow length of building in meters
  let height_lamp := 15 -- height of the lamp post in meters
  let shadow_lamp := 25 -- shadow length of the lamp post in meters
  let ratio := height_lamp / shadow_lamp -- ratio derived from lamp post
  in shadow_building * ratio

theorem building_height_is_correct : height_of_building = 72 := by
  let shadow_building := 120
  let height_lamp := 15
  let shadow_lamp := 25
  let ratio := height_lamp / shadow_lamp
  have ratio_is_correct : ratio = 3 / 5 := by
    calc
      ratio = height_lamp / shadow_lamp : rfl
      ... = 15 / 25 : rfl
      ... = 3 / 5 : by norm_num
  calc
    height_of_building
        = shadow_building * ratio : rfl
    ... = 120 * (3 / 5) : by rw [ratio_is_correct]
    ... = 72 : by norm_num
  sorry -- ensure code builds

end building_height_is_correct_l3_3718


namespace quadrilateral_area_formula_l3_3055

structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨1, 4⟩
def B : Point := ⟨1, 1⟩
def C : Point := ⟨3, 1⟩
def D : Point := ⟨1003, 1004⟩

noncomputable def distance (P1 P2 : Point) : ℝ :=
  ((P2.x - P1.x) ^ 2 + (P2.y - P1.y) ^ 2).sqrt

noncomputable def area_triangle (P1 P2 P3 : Point) : ℝ :=
  let a := distance P1 P2
  let b := distance P2 P3
  let c := distance P3 P1
  let s := (a + b + c) / 2
  (s * (s - a) * (s - b) * (s - c)).sqrt

noncomputable def quadrilateral_area (A B C D : Point) : ℝ :=
  area_triangle A B C + area_triangle A C D

theorem quadrilateral_area_formula : 
  quadrilateral_area A B C D = 
    area_triangle A B C + area_triangle A C D := 
by 
  rw quadrilateral_area
  sorry

end quadrilateral_area_formula_l3_3055


namespace sector_area_l3_3303

open Real

theorem sector_area (θ : ℝ) (r : ℝ) (hθ : θ = (2 * π) / 3) (hr : r = sqrt 3) :
  (1 / 2) * θ * r^2 = π :=
by
  have h1: r^2 = 3 := by
    rw hr
    exact sqr_sqrt 3
  rw [hθ, h1]
  simp
  sorry

end sector_area_l3_3303


namespace segments_can_form_triangle_l3_3280

-- Definitions
variables {A B C A' B' C' : Type} [EuclideanGeometry A B C] [EquilateralTriangle A B C]

open EquilateralTriangle

def outward_triangles_constructed (ΔABC : EquilateralTriangle A B C) : Prop :=
  let AC' := AB' in let BA' := BC' in let CB' := CA' in
  let angles_greater_than_120 := angle A'BC' > 120 ∧ angle C'AB' > 120 ∧ angle B'CA' > 120 in 
  ∃ A' B' C', (angles_greater_than_120 ∧
               A'B' = AC' ∧ 
               B'C' = BA' ∧ 
               C'A' = CB')

-- Theorem and statement that needs proof
theorem segments_can_form_triangle (ΔABC : EquilateralTriangle A B C) : 
  outward_triangles_constructed ΔABC → 
  (segment AB' + segment BC' > segment CA' ∧
   segment BC' + segment CA' > segment AB' ∧
   segment CA' + segment AB' > segment BC') :=
by 
  sorry

end segments_can_form_triangle_l3_3280


namespace ratio_female_to_male_l3_3807

variables {f m : ℕ} -- f for female members and m for male members
variables (avgF avgM avgTotal : ℕ) -- average ages for females, males, and total

-- Conditions
def condition1 := avgF = 45
def condition2 := avgM = 20
def condition3 (f m : ℕ) := avgTotal = (45 * f + 20 * m) / (f + m)

-- Theorem to prove
theorem ratio_female_to_male : condition1 → condition2 → condition3 f m → f / m = 1 / 4 :=
by
  intros h1 h2 h3
  sorry

end ratio_female_to_male_l3_3807


namespace machine_present_value_l3_3054

/-- A machine depreciates at a certain rate annually.
    Given the future value after a certain number of years and the depreciation rate,
    prove the present value of the machine. -/
theorem machine_present_value
  (depreciation_rate : ℝ := 0.25)
  (future_value : ℝ := 54000)
  (years : ℕ := 3)
  (pv : ℝ := 128000) :
  (future_value = pv * (1 - depreciation_rate) ^ years) :=
sorry

end machine_present_value_l3_3054


namespace total_chairs_needed_l3_3233

theorem total_chairs_needed (tables_4_seats tables_6_seats seats_per_table_4 seats_per_table_6 : ℕ) : 
  tables_4_seats = 6 → 
  seats_per_table_4 = 4 → 
  tables_6_seats = 12 → 
  seats_per_table_6 = 6 → 
  (tables_4_seats * seats_per_table_4 + tables_6_seats * seats_per_table_6) = 96 := 
by
  intros h1 h2 h3 h4
  -- sorry

end total_chairs_needed_l3_3233


namespace sum_first_n_terms_l3_3868

variable (a : ℕ → ℕ)

axiom a1_condition : a 1 = 2
axiom diff_condition : ∀ n : ℕ, a (n + 1) - a n = 2^n

-- Define the sum of the first n terms of the sequence
noncomputable def S : ℕ → ℕ
| 0 => 0
| (n + 1) => S n + a (n + 1)

theorem sum_first_n_terms (n : ℕ) : S a n = 2^(n + 1) - 2 :=
by
  sorry

end sum_first_n_terms_l3_3868


namespace total_tickets_sold_is_336_l3_3061

-- Define the costs of the tickets
def cost_vip_ticket : ℕ := 45
def cost_ga_ticket : ℕ := 20

-- Define the total cost collected
def total_cost_collected : ℕ := 7500

-- Define the difference in the number of tickets sold
def vip_less_ga : ℕ := 276

-- Define the main theorem to be proved
theorem total_tickets_sold_is_336 (V G : ℕ) 
  (h1 : cost_vip_ticket * V + cost_ga_ticket * G = total_cost_collected)
  (h2 : V = G - vip_less_ga) : V + G = 336 :=
  sorry

end total_tickets_sold_is_336_l3_3061


namespace smallest_n_with_digits_315_l3_3307

-- Defining the conditions
def relatively_prime (m n : ℕ) := Nat.gcd m n = 1
def valid_fraction (m n : ℕ) := (m < n) ∧ relatively_prime m n

-- Predicate for the sequence 3, 1, 5 in the decimal representation of m/n
def contains_digits_315 (m n : ℕ) : Prop :=
  ∃ k d : ℕ, 10^k * m % n = 315 * 10^(d - 3) ∧ d ≥ 3

-- The main theorem: smallest n for which the conditions are satisfied
theorem smallest_n_with_digits_315 :
  ∃ n : ℕ, valid_fraction m n ∧ contains_digits_315 m n ∧ n = 159 :=
sorry

end smallest_n_with_digits_315_l3_3307


namespace range_of_k_eq_l3_3836

theorem range_of_k_eq (k : ℝ) : 
  (∃ x : ℝ, cos (2 * x) - 2 * sqrt 3 * sin x * cos x = k + 1) ↔ k ∈ set.Icc (-3 : ℝ) (1 : ℝ) := 
by
  sorry

end range_of_k_eq_l3_3836


namespace max_area_of_unit_hexagon_l3_3408

/-- Define a unit hexagon by its properties. -/
structure UnitHexagon where
  A B C D E F : ℝ 
  AX DX AY DY : ℝ 
  AB AC DE DF : ℝ 
  angleBAC : ℝ

/-- A hexagon is called a "unit" if it has four diagonals of length 1,
    whose endpoints include all the vertices of the hexagon.
    Show that the largest possible area for such a hexagon is 3√3/4. -/
theorem max_area_of_unit_hexagon : 
  ∀ (H : UnitHexagon), 
    H.AB = 1 → H.AC = 1 → H.DE = 1 → H.DF = 1 → 
    H.AX = 2/3 → H.DX = 2/3 → H.AY = 2/3 → H.DY = 2/3 → 
  ∃ θ ≤ (π / 4), (area_hexagon H = sin (2*θ)) → 
  ∃ area, area = (3*sqrt 3) / 4 :=
begin
  sorry
end

end max_area_of_unit_hexagon_l3_3408


namespace options_a_and_c_correct_l3_3725

theorem options_a_and_c_correct :
  (let a₁ := (2, 3, -1) in
   let b₁ := (-2, -3, 1) in
   let a₁_is_parallel_b₁ := a₁ = (-1 : ℝ) • b₁ in
   (let a₂ := (1, -1, 2) in
    let u₂ := (6, 4, -1) in
    let not_parallel_or_perpendicular := ¬(a₂ = (6 : ℝ) • u₂) ∧ a₂.1 * u₂.1 + a₂.2 * u₂.2 + a₂.3 * u₂.3 ≠ 0 in
    (let u₃ := (2, 2, -1) in
     let v₃ := (-3, 4, 2) in
     let dot_product_zero := u₃.1 * v₃.1 + u₃.2 * v₃.2 + u₃.3 * v₃.3 = 0 in
     (let a₄ := (0, 3, 0) in
      let u₄ := (0, -5, 0) in
      let parallel_but_not_perpendicular := a₄ = (3 / 5 : ℝ) • u₄) in
     a₁_is_parallel_b₁ ∧ dot_product_zero))) :=
sorry

end options_a_and_c_correct_l3_3725


namespace gcd_expression_infinite_composite_pairs_exists_l3_3616

-- Part (a)
theorem gcd_expression (n : ℕ) (a : ℕ) (b : ℕ) (hn : n > 0) (ha : a > 0) (hb : b > 0) :
  Nat.gcd (n^a + 1) (n^b + 1) ≤ n^(Nat.gcd a b) + 1 :=
by
  sorry

-- Part (b)
theorem infinite_composite_pairs_exists (n : ℕ) (hn : n > 0) :
  ∃ (pairs : ℕ × ℕ → Prop), (∀ a b, pairs (a, b) → a > 1 ∧ b > 1 ∧ ∃ d, d > 1 ∧ a = d ∧ b = dn) ∧
  (∀ a b, pairs (a, b) → Nat.gcd (n^a + 1) (n^b + 1) = n^(Nat.gcd a b) + 1) ∧
  (∀ x y, x > 1 → y > 1 → x ∣ y ∨ y ∣ x → ¬pairs (x, y)) :=
by
  sorry

end gcd_expression_infinite_composite_pairs_exists_l3_3616


namespace quadratic_roots_l3_3251

noncomputable theory

def complex_omega (ω : ℂ) : Prop := ω^5 = 1 ∧ ω ≠ 1

def alpha (ω : ℂ) : ℂ := ω + ω^2

def beta (ω : ℂ) : ℂ := ω^3 + ω^4

theorem quadratic_roots (ω : ℂ) (h : complex_omega ω) :
  ∃ a b : ℝ, a = 1 ∧ b = 1 ∧ (∀ x, x^2 + (a : ℂ) * x + (b : ℂ) = 0 → (x = alpha ω ∨ x = beta ω)) :=
sorry

end quadratic_roots_l3_3251


namespace relationship_between_exponents_l3_3925

section
variables {a b c d : ℝ} {x y q z : ℝ}
hypothesis h1 : a^(3 * x) = c^(2 * q) ∧ c^(2 * q) = b
hypothesis h2 : c^(4 * y) = a^(5 * z) ∧ a^(5 * z) = d

theorem relationship_between_exponents : 5 * q * z = 6 * x * y :=
by sorry
end

end relationship_between_exponents_l3_3925


namespace flagpole_height_in_inches_l3_3401

theorem flagpole_height_in_inches
  (height_lamppost shadow_lamppost : ℚ)
  (height_flagpole shadow_flagpole : ℚ)
  (h₁ : height_lamppost = 50)
  (h₂ : shadow_lamppost = 12)
  (h₃ : shadow_flagpole = 18 / 12) :
  height_flagpole * 12 = 75 :=
by
  -- Note: To keep the theorem concise, proof steps are omitted
  sorry

end flagpole_height_in_inches_l3_3401


namespace position_relationship_correct_l3_3736

noncomputable def problem_data :=
  (((2:ℝ), 3, -1), ((-2:ℝ), -3, 1), ((1:ℝ), -1, 2), ((6:ℝ), 4, -1), 
   ((2:ℝ), 2, -1), ((-3:ℝ), 4, 2), ((0:ℝ), 3, 0), ((0:ℝ), -5, 0))

theorem position_relationship_correct :
  let ⟨a, b, a2, u2, u3, v3, a4, u4⟩ := problem_data in
  (a = ((-1):ℝ) • b ∧
  (∀ k : ℝ, a2 ≠ k • u2) ∧
  (u3 ⬝ v3 = 0 ∧
  ¬ (a4 = (0:ℝ) • u4 ∧ a4 = (-3/5:ℝ) • u4 ))) :=
sorry

end position_relationship_correct_l3_3736


namespace periodic_derivative_l3_3506

theorem periodic_derivative {f : ℝ → ℝ} (h_diff : differentiable ℝ f) (h_periodic : ∃ T ≠ 0, ∀ x, f (x + T) = f x) :
  ∃ T ≠ 0, ∀ x, deriv f (x + T) = deriv f x :=
by sorry

end periodic_derivative_l3_3506


namespace number_of_older_females_l3_3945

theorem number_of_older_females (total_population : ℕ) (num_groups : ℕ) (one_group_population : ℕ) :
  total_population = 1000 → num_groups = 5 → total_population = num_groups * one_group_population →
  one_group_population = 200 :=
by
  intro h1 h2 h3
  sorry

end number_of_older_females_l3_3945


namespace factorization_correct_l3_3368

theorem factorization_correct :
  ∀ (y : ℝ), (y^2 - 1 = (y + 1) * (y - 1)) :=
by
  intro y
  sorry

end factorization_correct_l3_3368


namespace triathlete_average_speed_l3_3434

theorem triathlete_average_speed (d_swim d_bike d_run d_kayak : ℕ)
  (s_swim s_bike s_run s_kayak : ℕ)
  (d_total time_total v_avg : ℚ) (v_approx : v_avg ≈ 8.3) :
  d_swim = 2 →
  d_bike = 8 →
  d_run = 5 →
  d_kayak = 3 →
  s_swim = 2 →
  s_bike = 25 →
  s_run = 12 →
  s_kayak = 7 →
  d_total = d_swim + d_bike + d_run + d_kayak →
  time_total = (d_swim / s_swim) + (d_bike / s_bike) + (d_run / s_run) + (d_kayak / s_kayak) →
  v_avg = d_total / time_total →
  v_avg ≈ 8.3 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11
  sorry

end triathlete_average_speed_l3_3434


namespace Elliot_reads_20_pages_per_day_l3_3005

def pages_in_book (P : ℕ) : Prop := P = 381
def pages_already_read (R : ℕ) : Prop := R = 149
def pages_remaining (L : ℕ) : Prop := L = 92
def days_in_week (D : ℕ) : Prop := D = 7

theorem Elliot_reads_20_pages_per_day (P R L D pages_read_per_day : ℕ)
  (h1 : pages_in_book P) (h2 : pages_already_read R) (h3 : pages_remaining L) (h4 : days_in_week D) :
  pages_read_per_day = 20 :=
begin
  sorry
end

end Elliot_reads_20_pages_per_day_l3_3005


namespace residue_T_modulo_2020_l3_3263

theorem residue_T_modulo_2020 :
  let
    T := (Finset.range 2020).sum (λ n, if even n then - (n + 1) else (n + 1))
  in
  T % 2020 = 1010 :=
by
  sorry

end residue_T_modulo_2020_l3_3263


namespace proof_problem_l3_3666

noncomputable def problem : ℕ :=
  let p := 588
  let q := 0
  let r := 1
  p + q + r

theorem proof_problem
  (AB : ℝ) (P Q : ℝ) (AP BP PQ : ℝ) (angle_POQ : ℝ) 
  (h1 : AB = 1200)
  (h2 : AP + PQ = BP)
  (h3 : BP - Q = 600)
  (h4 : angle_POQ = 30)
  (h5 : PQ = 500)
  : problem = 589 := by
    sorry

end proof_problem_l3_3666


namespace problem_statement_l3_3370

variables (x a : ℝ) -- Define variables as real numbers
def is_polynomial (expr : ℝ → ℝ) : Prop := ∀ y : ℝ, differentiable ℝ expr -- Assume differentiable implies polynomial (for simplicity in this example)

theorem problem_statement : ¬ is_polynomial (λ y, y + (a / y) + 1) :=
  by sorry

end problem_statement_l3_3370


namespace S_le_131_exists_tau_max_S_ge_57_exists_tau_min_num_permutations_max_num_permutations_min_l3_3862

open BigOperators

def S (tau : Fin 10 → Fin 10) : ℤ :=
  ∑ k in Finset.range 10, | (2 : ℤ) * (tau k).val - 3 * (tau ((k + 1) % 10)).val |

theorem S_le_131 : ∀ (tau : Fin 10 → Fin 10), S tau ≤ 131 :=
by sorry

theorem exists_tau_max : ∃ (tau : Fin 10 → Fin 10), S tau = 131 :=
by sorry

theorem S_ge_57 : ∀ (tau : Fin 10 → Fin 10), S tau ≥ 57 :=
by sorry

theorem exists_tau_min : ∃ (tau : Fin 10 → Fin 10), S tau = 57 :=
by sorry

theorem num_permutations_max : Finset.card {tau : Fin 10 → Fin 10 | S tau = 131} = 28800 :=
by sorry

theorem num_permutations_min : Finset.card {tau : Fin 10 → Fin 10 | S tau = 57} = 720 :=
by sorry

end S_le_131_exists_tau_max_S_ge_57_exists_tau_min_num_permutations_max_num_permutations_min_l3_3862


namespace sufficient_but_not_necessary_l3_3390

theorem sufficient_but_not_necessary (x : ℝ) :
  (x > 3 → x^2 > 4) ∧ ¬(x^2 > 4 → x > 3) :=
by sorry

end sufficient_but_not_necessary_l3_3390


namespace probability_multiple_4_or_15_l3_3780

-- Definitions of natural number range and a set of multiples
def first_30_nat_numbers : Finset ℕ := Finset.range 30
def multiples_of (n : ℕ) (s : Finset ℕ) : Finset ℕ := s.filter (λ x => x % n = 0)

-- Conditions
def multiples_of_4 := multiples_of 4 first_30_nat_numbers
def multiples_of_15 := multiples_of 15 first_30_nat_numbers

-- Proof that probability of selecting a multiple of 4 or 15 is 3 / 10
theorem probability_multiple_4_or_15 : 
  let favorable_outcomes := (multiples_of_4 ∪ multiples_of_15).card
  let total_outcomes := first_30_nat_numbers.card
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 :=
by
  -- correct answer based on the computation
  sorry

end probability_multiple_4_or_15_l3_3780


namespace eccentricity_of_cylinder_intersection_l3_3771

-- Define the given conditions and the proof goal in Lean.
theorem eccentricity_of_cylinder_intersection (R : ℝ) (hR : 0 < R) :
  let θ := 30 * (Real.pi / 180) in
  let b := R * Real.cos θ in
  let e := Real.sqrt (1 - (b / R)^2) in
  e = 1 / 2 :=
by
  sorry

end eccentricity_of_cylinder_intersection_l3_3771


namespace unpainted_face_area_correct_l3_3046

-- Define the right circular cylinder properties
def radius := 6
def height := 8
def theta := 120 -- degrees

-- Define the unpainted face area
def unpainted_face_area (r h : ℕ) (angle : ℕ) : ℝ :=
  let semi_major_axis := r
  let semi_minor_axis := h
  let ellipse_area := Real.pi * semi_major_axis * semi_minor_axis
  let sector_area := ellipse_area * (angle / 360)
  let chord := 2 * r * Real.sin (Real.toRadians (angle / 2))
  let triangle_area := (Real.sqrt 3 / 4) * (chord ^ 2)
  sector_area + triangle_area

-- Statement to prove
theorem unpainted_face_area_correct :
  ∃ (a b c : ℕ), (unpainted_face_area radius height theta = a * Real.pi + b * Real.sqrt c) ∧ (a + b + c = 46) :=
by
  sorry

end unpainted_face_area_correct_l3_3046


namespace minimum_moves_to_find_coin_l3_3334

theorem minimum_moves_to_find_coin (n : ℕ) (h : n ≥ 9) : 
  ∃ k : ℕ, k = (⌈(n - 3 : ℝ) / 2⌉.to_nat - 1) ∧ 
    ∀ (choose_cups : finset ℕ → Prop),
    (∀ coin_pos ∈ finset.range n, ∃ i ≤ k, choose_cups (set_of (λ j, j ∈ finset.range (coin_pos + 2) ∪ finset.range (coin_pos - 2)))) :=
sorry

end minimum_moves_to_find_coin_l3_3334


namespace breadth_of_plot_l3_3754

-- Definition of the given problem conditions
def area_of_plot : ℝ := 360
def breadth := length / 0.75
def length (b : ℝ) := b * 0.75

-- Statement of the proof problem
theorem breadth_of_plot (b : ℝ):
  b * 0.75 * b = area_of_plot → b = 4 * Real.sqrt 30 ∨ b = - (4 * Real.sqrt 30) :=
by
  sorry

end breadth_of_plot_l3_3754


namespace planted_area_fraction_of_triangle_with_square_l3_3111

noncomputable def planted_area_fraction (leg1 leg2 hypotenuse_distance: ℕ) : ℚ :=
  let area_triangle := (leg1 * leg2 : ℚ) / 2
  let s := 3  -- Shortest distance from the square to the hypotenuse, given as 3
  let planted_area := area_triangle - (s^2 : ℚ)
  in planted_area / area_triangle

theorem planted_area_fraction_of_triangle_with_square :
  planted_area_fraction 5 12 3 = 13 / 40 :=
by
  sorry

end planted_area_fraction_of_triangle_with_square_l3_3111


namespace compute_expression_l3_3090

theorem compute_expression :
  (75 * 1313 - 25 * 1313 + 50 * 1313 = 131300) :=
by
  sorry

end compute_expression_l3_3090


namespace g_range_l3_3490

noncomputable def g (x : ℝ) : ℝ := Real.arcsin x + Real.arccos x + Real.arctan x + (π / 2 - Real.arctan x)

theorem g_range : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → g x = π := 
by
  intro x hx
  have h₁ : Real.arcsin x + Real.arccos x = π / 2 :=
    sorry -- Arcsin and Arccos property for x in [-1, 1]
  have h₂ : g x = Real.arcsin x + Real.arccos x + Real.arctan x + (π / 2 - Real.arctan x) :=
    rfl
  rw [h₂, h₁]
  have h₃ : Real.symm Add.add_∗ a₁ a₂ (Real.arctan x + (π / 2 - Real.arctan x)) = π :=
    sorry -- Simplify the addition
  assumption h₃  
  sorry

end g_range_l3_3490


namespace integral_of_ellipse_zero_l3_3081

theorem integral_of_ellipse_zero (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) :
  ∮ (fun (x y : ℝ) => (x^2 - y^2)) (fun (x y : ℝ) => 2*x*y)
    {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1} = 0 :=
by
  -- Proof skipped
  sorry

end integral_of_ellipse_zero_l3_3081


namespace amount_of_c_l3_3748

theorem amount_of_c (A B C : ℕ) (h1 : A + B + C = 350) (h2 : A + C = 200) (h3 : B + C = 350) : C = 200 :=
sorry

end amount_of_c_l3_3748


namespace tan_theta_eq_minus_four_thirds_frac_expr_eq_minus_seven_l3_3141

-- Given conditions
variables {θ : ℝ}
axiom h1 : sin θ + cos θ = 1 / 5
axiom h2 : 0 < θ ∧ θ < π

-- Prove the first part: tan θ = -4/3
theorem tan_theta_eq_minus_four_thirds : tan θ = -4 / 3 := by
  sorry

-- Prove the second part: (1 - 2 * sin θ * cos θ) / (cos^2 θ - sin^2 θ) = -7
theorem frac_expr_eq_minus_seven 
  : (1 - 2 * sin θ * cos θ) / (cos θ ^ 2 - sin θ ^ 2) = -7 := by
  sorry

end tan_theta_eq_minus_four_thirds_frac_expr_eq_minus_seven_l3_3141


namespace solve_for_y_l3_3664

theorem solve_for_y :
  (∃ y : ℝ, (∛(30*y + ∛(30*y + 26)) = 26)) → (y = 585) := by
  sorry

end solve_for_y_l3_3664


namespace solve_fx_l3_3874

def f : ℝ → ℝ
| x := if x ≤ 0 then real.cos (π * x / 2) else f (x - 1) + 1

theorem solve_fx : f 2 = 3 :=
by
  sorry

end solve_fx_l3_3874


namespace speed_of_boat_in_still_water_l3_3218

-- Define a structure for the conditions
structure BoatConditions where
  V_b : ℝ    -- Speed of the boat in still water
  V_s : ℝ    -- Speed of the stream
  goes_along_stream : V_b + V_s = 11
  goes_against_stream : V_b - V_s = 5

-- Define the target theorem
theorem speed_of_boat_in_still_water (c : BoatConditions) : c.V_b = 8 :=
by
  sorry

end speed_of_boat_in_still_water_l3_3218


namespace emily_days_off_l3_3110

/-
Emily took a day off from work twice every month and occasionally took additional unpaid leaves. She also enjoyed 10 public holidays that fell on weekdays.
Given that she took 3 unpaid leaves throughout the year, calculate the total number of days off Emily took.
-/
theorem emily_days_off
  (months_in_year : ℕ) (days_off_per_month : ℕ) (public_holidays : ℕ) (unpaid_leaves : ℕ) :
  months_in_year = 12 →
  days_off_per_month = 2 →
  public_holidays = 10 →
  unpaid_leaves = 3 →
  months_in_year * days_off_per_month + public_holidays + unpaid_leaves = 37 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end emily_days_off_l3_3110


namespace _l3_3069

open Nat

/-- Function to check the triangle inequality theorem -/
def canFormTriangle (a b c : Nat) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

example : canFormTriangle 6 4 5 := by
  /- Proof omitted -/
  sorry

end _l3_3069


namespace maximal_value_fraction_l3_3202

noncomputable def maximum_value_ratio (a b c : ℝ) (S : ℝ) : ℝ :=
  if S = c^2 / 4 then 2 * Real.sqrt 2 else 0

theorem maximal_value_fraction (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (area_cond : 1/2 * a * b * Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) = c^2 / 4) :
  maximum_value_ratio a b c (c^2/4) = 2 * Real.sqrt 2 :=
sorry

end maximal_value_fraction_l3_3202


namespace quarters_flipped_by_Eric_l3_3454

open Nat

theorem quarters_flipped_by_Eric
  (dimes : ℕ := 5)
  (nickels : ℕ := 8)
  (pennies : ℕ := 60)
  (total_amount : ℕ := 200) :
  (quarters : ℕ) → quarters * 25 = total_amount - (dimes * 10 + nickels * 5 + pennies * 1) :=
by 
  intro quarters
  have h : dimes * 10 + nickels * 5 + pennies * 1 = 50 + 40 + 60 := by norm_num
  have h_total : quarters * 25 = 50 := by sorry
  exact h_total

end quarters_flipped_by_Eric_l3_3454


namespace division_points_form_regular_hexagon_l3_3584

-- Define an equilateral triangle with points A, B, C
structure EquilateralTriangle (Point : Type) [MetricSpace Point] (A B C : Point) : Prop :=
  (AB_eq_BC : dist A B = dist B C)
  (BC_eq_CA : dist B C = dist C A)

-- Define the division points as mentioned in the problem
structure DivisionPoints (Point : Type) [MetricSpace Point] (A B C D E F G H I : Point) : Prop :=
  (A_D_D_E_E_B : dist A D = dist D E ∧ dist D E = dist E B)
  (B_F_F_G_G_C : dist B F = dist F G ∧ dist F G = dist G C)
  (C_H_H_I_I_A : dist C H = dist H I ∧ dist H I = dist I A)
  (segment_length : ∀ s, dist A B = s → dist A D = s / 3)

-- The vertices of the hexagon
structure HexagonVertices (Point : Type) [MetricSpace Point] (D E F G H I : Point) : Prop :=
  (regular_hexagon : ∀ a b c d e f, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e ∧ e ≠ f ∧ f ≠ a → 
    ∀ (A B C : Point), EquilateralTriangle Point A B C →
    DivisionPoints Point A B C D E F G H I → True)

theorem division_points_form_regular_hexagon (Point : Type) [MetricSpace Point] : 
  ∀ (A B C D E F G H I : Point), 
  EquilateralTriangle Point A B C →
  DivisionPoints Point A B C D E F G H I →
  HexagonVertices Point D E F G H I :=
by
  -- Proof elided
  sorry

end division_points_form_regular_hexagon_l3_3584


namespace unique_triangle_count_l3_3187

theorem unique_triangle_count : ∃ a b c : ℕ,
  a + b + c = 8 ∧
  (a ≤ 4 ∨ b ≤ 4 ∨ c ≤ 4) ∧
  (a + b > c ∧ a + c > b ∧ b + c > a) ∧
  ∀ (a' b' c' : ℕ), 
    a' + b' + c' = 8 ∧ 
    (a' ≤ 4 ∨ b' ≤ 4 ∨ c' ≤ 4) ∧ 
    (a' + b' > c' ∧ a' + c' > b' ∧ b' + c' > a')
    → (multiset {a, b, c} = multiset {a', b', c'}) :=
by sorry

end unique_triangle_count_l3_3187


namespace batsman_average_increases_by_one_l3_3764

noncomputable def batsman_average_after_11th_inning (avg_after_10 : ℕ) : ℕ :=
let total_runs_after_10 := 10 * avg_after_10 in
let total_runs_after_11 := total_runs_after_10 + 69 in
let new_avg_after_11 := total_runs_after_11 / 11 in
new_avg_after_11

theorem batsman_average_increases_by_one (avg_after_10 : ℕ) (h : batsman_average_after_11th_inning avg_after_10 = avg_after_10 + 1) :
  batsman_average_after_11th_inning avg_after_10 = 59 :=
by
  let x := avg_after_10 in
  have total_runs_after_10 : 10 * x := by sorry
  have total_runs_after_11 : 10 * x + 69 := by sorry
  have new_avg_after_11 : total_runs_after_11 / 11 := by sorry
  have increased_avg_x : avg_after_10 + 1 = new_avg_after_11 := by sorry
  show batsman_average_after_11th_inning avg_after_10 = 59, from sorry

end batsman_average_increases_by_one_l3_3764


namespace intersect_at_one_point_l3_3986

theorem intersect_at_one_point
  (A B C X I_1 I_2 I_3 : Type) 
  [geometry A B C X I_1 I_2 I_3] 
  (h : X ∈ triangle A B C) 
  (h1 : XA * BC = XB * AC) 
  (h2 : XB * AC = XC * AB) 
  (h3 : XC * AB = XA * BC)
  (hI1 : is_incircle_center X B C I_1)
  (hI2 : is_incircle_center X C A I_2)
  (hI3 : is_incircle_center X A B I_3) :
  intersect AI_1 BI_2 CI_3 := sorry

end intersect_at_one_point_l3_3986


namespace find_1500th_letter_in_sequence_l3_3717

-- Definition of the repeating sequence
def sequence : List Char := ['A', 'B', 'C', 'D', 'D', 'C', 'B', 'A']

-- Length of the repeating sequence
def sequence_length : Nat := 8

-- Position to be found in the sequence
def position : Nat := 1500

-- Correct answer for the position
def expected_letter : Char := 'D'

theorem find_1500th_letter_in_sequence 
  (h_seq_length : sequence.length = sequence_length)
  (h_pos : position % sequence_length = 4):
  sequence.get ⟨4, by linarith [h_seq_length]⟩ = expected_letter := 
  sorry

end find_1500th_letter_in_sequence_l3_3717


namespace angle_bisector_properties_l3_3392

-- Definitions based on the given conditions
variables {ABC : Type*} [triangle ABC]
variables {Γ : Type*} [circle Γ ABC]
variables {O : point}
variables {I : point}
variables {I_A : point}
variables (A B C : point) (I_O: line) (BC : line)

-- Property of point I_A as per given conditions
axiom IA_def : internal_angle_bisector A I_A ∧ second_intersection I_A Γ

-- Problem statement conversion to Lean theorem
theorem angle_bisector_properties (h : IA_def A B C I_A) :
  (perpendicular (line_through I_A O) (line_through B C)) ∧
  (circumcircle_passing_through_points I_A B C I) :=
sorry

end angle_bisector_properties_l3_3392


namespace arrival_time_difference_l3_3437

-- Define the times in minutes, with 600 representing 10:00 AM.
def my_watch_time_planned := 600
def my_watch_fast := 5
def my_watch_slow := 10

def friend_watch_time_planned := 600
def friend_watch_fast := 5

-- Calculate actual arrival times.
def my_actual_arrival_time := my_watch_time_planned - my_watch_fast + my_watch_slow
def friend_actual_arrival_time := friend_watch_time_planned - friend_watch_fast

-- Prove the arrival times and difference.
theorem arrival_time_difference :
  friend_actual_arrival_time < my_actual_arrival_time ∧
  my_actual_arrival_time - friend_actual_arrival_time = 20 :=
by
  -- Proof terms can be filled in later.
  sorry

end arrival_time_difference_l3_3437


namespace min_sum_dist_ellipse_l3_3495

theorem min_sum_dist_ellipse (a b : ℝ) (h : a > b) (x0 y0 : ℝ) 
  (P : x0^2 / a^2 + y0^2 / b^2 = 1) (hneq : y0 ≠ b ∧ y0 ≠ -b) :
  let M := (b * x0 / (b - y0), 0)
  let N := (b * x0 / (b + y0), 0)
  2 * a ≤ abs (b * x0 / (b - y0)) + abs (b * x0 / (b + y0)) :=
begin
  sorry
end

end min_sum_dist_ellipse_l3_3495


namespace circumradii_form_triangle_with_half_area_l3_3098

variables {A B C A1 B1 C1 A' B' C' A'' B'' C'' : Type*}
variables [EuclideanGeometry A B C]
variables (R R' R'' : Real)
variables (area_ABC : Real)
variables (circumradius : A B C Real)

-- Conditions
variables (acute_triangle : acute_triangle A B C)
variables (altitudes : (altitude A A1 B C) ∧ (altitude B B1 A C) ∧ (altitude C C1 A B))
variables (points_def : (point_on_extension B1 A1 A1 C' B1 C1) ⟨point_on_extension A1 C1 C1 B' A1 B1⟩ )
variables (sym_points : (symmetric_point A_ A' BC) ∧ (symmetric_point B_ B' CA) ∧ (symmetric_point C_ C' AB))

-- Correct Answer Translation
theorem circumradii_form_triangle_with_half_area :
    ∃ (R R' R'' : Real), circumradius (A B C) = R ∧ circumradius (A' B' C') = R' ∧ circumradius (A'' B'' C'') = R'' ∧
    (area (R R' R'') = (area_ABC / 2)) :=
sorry

end circumradii_form_triangle_with_half_area_l3_3098


namespace find_a2_l3_3927

noncomputable theory

open Complex

def z : ℂ := (1/2) + (sqrt 3 / 2) * I

def polynomial_expansion := ((x : ℂ) - z) ^ 4

theorem find_a2 :
  let a2 := 6 * (- ((1/2) + (sqrt 3 / 2) * I))^2 in
  a2 = -3 + 3 * sqrt 3 * I :=
by
  sorry

end find_a2_l3_3927


namespace regression_estimate_l3_3176

theorem regression_estimate :
  ∀ (x y : ℝ), (y = 0.50 * x - 0.81) → x = 25 → y = 11.69 :=
by
  intros x y h_eq h_x_val
  sorry

end regression_estimate_l3_3176


namespace number_of_store_DVDs_l3_3639

theorem number_of_store_DVDs (total_DVDs online_DVDs store_DVDs : ℕ) (h1 : total_DVDs = 10) (h2 : online_DVDs = 2) :
  store_DVDs = total_DVDs - online_DVDs :=
by
  have h : store_DVDs = 10 - 2 := sorry
  exact h

end number_of_store_DVDs_l3_3639


namespace min_t_value_l3_3499

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x - 1

theorem min_t_value :
  ∃ t : ℝ, (∀ x1 x2 : ℝ, x1 ∈ Set.Icc (-3 : ℝ) 2 → x2 ∈ Set.Icc (-3 : ℝ) 2 → |f x1 - f x2| ≤ t) ∧ t = 20 :=
begin
  sorry
end

end min_t_value_l3_3499


namespace volume_ratio_l3_3045

-- Definitions for the conditions
def length : ℕ := 9
def width : ℕ := 6

-- The radii based on rolling along the width and length
def radius1 : ℝ := width / (2 * Real.pi)
def radius2 : ℝ := length / (2 * Real.pi)

-- The heights of the corresponding cylinders
def height1 : ℕ := length
def height2 : ℕ := width

-- Volumes of the corresponding cylinders
def volume1 : ℝ := Real.pi * radius1^2 * height1
def volume2 : ℝ := Real.pi * radius2^2 * height2

-- Theorem statement for the ratio of the volumes
theorem volume_ratio : (if volume1 > volume2 then volume1 / volume2 else volume2 / volume1) = 3 / 2 := sorry

end volume_ratio_l3_3045


namespace probability_abs_diff_gt_half_eq_l3_3660

noncomputable def probability_abs_diff_gt_half : ℝ :=
  ∫ x in set.Icc (0:ℝ) 1, ∫ y in set.Icc (0:ℝ) 1, 
    if (|x - y| > 1/2) then 
      ((1 / 2) * (1 / 6) * (1 / 2) * (1 / 6)) + -- both by die
      ((1 / 2) * (1 / 6) * (1 / 2)) + -- one by die, one uniformly
      ((1 / 2) * (1 / 2) * (1 / 2) * (1 / 2)) -- both uniformly
    else 0

theorem probability_abs_diff_gt_half_eq : probability_abs_diff_gt_half = 1/4 :=
sorry

end probability_abs_diff_gt_half_eq_l3_3660


namespace num_pos_divisors_of_30030_l3_3124

def prime_factors (n : ℕ) (factors : List ℕ) :=
  factors.prod = n ∧ factors.all prime

theorem num_pos_divisors_of_30030 :
  ∀ (n : ℕ), prime_factors n [2, 3, 5, 7, 11, 13] → 
  n = 30030 → 
  (finset.divisors n).card = 64 :=
by
  intros n h_factors h_eq
  sorry

end num_pos_divisors_of_30030_l3_3124


namespace binomial_expansion_terms_l3_3894

theorem binomial_expansion_terms (n : ℕ) (x : ℝ) (h : Nat.choose n (n-2) = 45) :
  (∃ c : ℝ, (sqrt (sqrt x) + sqrt (x ^ 3)) ^ n = c * x^5 ∧ c = 45) ∧
  (∃ c' : ℝ, (sqrt (sqrt x) + sqrt (x ^ 3)) ^ n = c' * x^(35 / 4) ∧ c' = 252) := by
  sorry

end binomial_expansion_terms_l3_3894


namespace common_point_on_x_axis_distance_between_intersections_l3_3598

def C1_parametric (t : ℝ) : ℝ × ℝ := (t + 1, 1 - 2 * t)
def C2_parametric (a θ : ℝ) : ℝ × ℝ := (a * Real.cos θ, 3 * Real.sin θ)

theorem common_point_on_x_axis :
  (∃ t, (t + 1, 1 - 2 * t) = (a, 0)) →
  a = (3 / 2) := by sorry

theorem distance_between_intersections (a : ℝ) :
  a = 3 →
  (∃ A B : ℝ × ℝ, C1_parametric (A.1) = C2_parametric a (A.2) ∧ C1_parametric (B.1) = C2_parametric a (B.2) ∧ A ≠ B) →
  (dist A B = (12 * Real.sqrt 5) / 5) := by sorry

end common_point_on_x_axis_distance_between_intersections_l3_3598


namespace correct_descriptions_count_l3_3438

-- Definitions for the conditions
def cond1 : Prop := 
  "The dilution spreading plate method easily obtains single colonies and counts the number of viable bacteria"

def cond2 : Prop := 
  "In the experiment observing mitochondria under a high-power microscope, cells die after staining with Janus Green"

def cond3 : Prop := 
  "The orange solution of potassium dichromate reacts with alcohol under alkaline conditions to turn gray-green"

def cond4 : Prop := 
  "In the experiment observing mitosis in the meristematic tissue cells of onion root tips, the basic dye pyronin can be used for staining"

def cond5 : Prop := 
  "When treating apple pulp with pectinase, significantly increasing the reaction temperature can make pectinase catalyze the reaction more fully"

-- Main statement of the proof
theorem correct_descriptions_count : (cond1 = True) ∧ (cond2 = False) ∧ (cond3 = False) ∧ (cond4 = False) ∧ (cond5 = False) → 1 =
sorry

end correct_descriptions_count_l3_3438


namespace money_left_after_transaction_l3_3028

def initial_amount := 200.00
def bread := 5.00
def candy := 3.00 * 3.00
def cereal := 6.00
def milk := 2.00 * 4.00
def eggs := 3.00
def cheese := 4.00

def initial_items_total := bread + candy + cereal + milk + eggs + cheese
def remaining_after_initial_items := initial_amount - initial_items_total

def fruits := 0.15 * remaining_after_initial_items
def remaining_after_fruits := remaining_after_initial_items - fruits

def must_buy_initial := 10.00
def must_buy_discount := 0.05 * must_buy_initial
def must_buy := must_buy_initial - must_buy_discount
def remaining_after_must_buy := remaining_after_fruits - must_buy

def vegetable_mix := remaining_after_must_buy / 5
def remaining_after_vegetable_mix := remaining_after_must_buy - vegetable_mix

def pre_tax_total := initial_items_total + fruits + must_buy + vegetable_mix
def sales_tax := 0.07 * pre_tax_total
def post_tax_total := pre_tax_total + sales_tax

def remaining_money := remaining_after_vegetable_mix - post_tax_total

theorem money_left_after_transaction : abs (remaining_money - 2.52) < 0.01 :=
  by
  sorry

end money_left_after_transaction_l3_3028


namespace max_area_triangle_DAB_eqn_l3_3633

noncomputable def circle_M (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 15 = 0

def center_M : ℝ × ℝ := (-1, 0)

noncomputable def point_N : ℝ × ℝ := (1, 0)

-- Define E as an ellipse passing through certain conditions 
noncomputable def curve_E (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define line l with the given slope m in R
def line_l (m : ℝ) (x y : ℝ) : Prop := x = m * y - 4

-- The main theorem to be proved
theorem max_area_triangle_DAB_eqn (m : ℝ) : 
  curve_E x y ∧ line_l m x y → 
  m = ℝ.sqrt(28/3) / 3 ∨ 
  m = -ℝ.sqrt(28/3) / 3 :=
sorry

end max_area_triangle_DAB_eqn_l3_3633


namespace Zoey_finished_15th_book_on_Monday_l3_3746

open Nat

inductive DayOfWeek
  | Sunday | Monday | Tuesday | Wednesday 
  | Thursday | Friday | Saturday
  deriving DecidableEq, Repr

def reading_days : List ℕ := List.range' 1 16 -- [1, 2, ..., 15]

def total_reading_days := reading_days.sum

def starting_day : DayOfWeek := DayOfWeek.Monday

def calculate_day_of_week (starting_day : DayOfWeek) (days_passed : ℕ) : DayOfWeek :=
  match starting_day, days_passed % 7 with
  | DayOfWeek.Sunday, 0 => DayOfWeek.Sunday
  | DayOfWeek.Monday, 0 => DayOfWeek.Monday
  | DayOfWeek.Tuesday, 0 => DayOfWeek.Tuesday
  | DayOfWeek.Wednesday, 0 => DayOfWeek.Wednesday
  | DayOfWeek.Thursday, 0 => DayOfWeek.Thursday
  | DayOfWeek.Friday, 0 => DayOfWeek.Friday
  | DayOfWeek.Saturday, 0 => DayOfWeek.Saturday
  | _, n => calculate_day_of_week DayOfWeek.Sunday n

theorem Zoey_finished_15th_book_on_Monday :
  calculate_day_of_week starting_day total_reading_days = DayOfWeek.Monday := by
  sorry

end Zoey_finished_15th_book_on_Monday_l3_3746


namespace square_of_1024_l3_3825

theorem square_of_1024 : (1024 : ℤ)^2 = 1048576 := by
  let a := 1020
  let b := 4
  have h : (1024 : ℤ) = a + b := by
    norm_num
  rw [h] 
  norm_num
  sorry
  -- expand (a+b)^2 = a^2 + 2ab + b^2
  -- prove that 1020^2 = 1040400
  -- prove that 2 * 1020 * 4 = 8160
  -- prove that 4^2 = 16
  -- sum these results 
  -- result = 1048576

end square_of_1024_l3_3825


namespace basketball_team_selection_l3_3652

open BigOperators

theorem basketball_team_selection :
  (∑ i in finset.range 18.succ, nat.choose 18 i) - (∑ i in finset.range 16.succ, nat.choose 16 i) = 20384 :=
by {
    sorry
}

end basketball_team_selection_l3_3652


namespace finite_2a_next_equals_3a_l3_3973

noncomputable def seq (n : ℕ) : ℕ
| 0     := 2
| (n+1) := Nat.find (λ m, m > seq n ∧ Nat.totient m > Nat.totient (seq n))

theorem finite_2a_next_equals_3a (n : ℕ): 
  (∃ N, ∀ n > N, 2 * seq (n + 1) ≠ 3 * seq n) :=
sorry

end finite_2a_next_equals_3a_l3_3973


namespace nursery_school_students_l3_3037

def num_students (S : ℕ) : Prop :=
  ∃ (S : ℕ), (1 / 10 : ℚ) * S = 30 ∧ 50 = 30 + 20

theorem nursery_school_students : ∃ S, num_students S :=
by {
  -- Step to assert the existence
  use 300,
  -- Skip the detailed proof
  sorry
}

end nursery_school_students_l3_3037


namespace range_of_real_number_m_l3_3632

open Set

variable {m : ℝ}

theorem range_of_real_number_m (U : Set ℝ) (A : Set ℝ) (B : Set ℝ) (h1 : U = univ) (h2 : A = { x | x < 1 }) (h3 : B = { x | x ≥ m }) (h4 : compl A ⊆ B) : m ≤ 1 := by
  sorry

end range_of_real_number_m_l3_3632


namespace find_n_series_sum_l3_3345

theorem find_n_series_sum 
  (first_term_I : ℝ) (second_term_I : ℝ) (first_term_II : ℝ) (second_term_II : ℝ) (sum_multiplier : ℝ) (n : ℝ)
  (h_I_first_term : first_term_I = 12)
  (h_I_second_term : second_term_I = 4)
  (h_II_first_term : first_term_II = 12)
  (h_II_second_term : second_term_II = 4 + n)
  (h_sum_multiplier : sum_multiplier = 5) :
  n = 152 :=
by
  sorry

end find_n_series_sum_l3_3345


namespace compute_difference_of_squares_l3_3094

theorem compute_difference_of_squares : 
  let a := 23 + 15
  let b := 23 - 15
  (a^2 - b^2) = 1380 := 
by
  rw [pow_two, pow_two, add_assoc, sub_eq_add_neg, add_assoc, add_comm, sub_eq_add_neg]
  -- The proof step can be filled later
  sorry

end compute_difference_of_squares_l3_3094


namespace triangle_area_division_l3_3650

theorem triangle_area_division (ABC : Triangle) (h_equilateral : ABC.is_equilateral) (M N O : Point)
  (h_AM : M ∈ segment(ABC.AC)) (h_BN : N ∈ segment(ABC.CB)) (h_AMeq : dist(ABC.A M) = dist(ABC.B N))
  (h_AM_val : dist(ABC.A M) = (1/4) * dist(ABC.A B)) (h_O_center : O = ABC.centroid)
  (h_M_val : segment_ratio(ABC.AC M (3/4))) (h_N_val : segment_ratio(ABC.CB N (3/4))) :
  divides_into_two_equal_parts_by_line ABC MON := sorry

end triangle_area_division_l3_3650


namespace exists_finite_set_geometric_mean_integer_not_exists_infinite_set_geometric_mean_integer_l3_3391

-- Problem (1) Statement
theorem exists_finite_set_geometric_mean_integer (n : ℕ) (h : 0 < n) :
  ∃ S_n : Finset ℕ, (S_n.card = n ∧
                     ∀ (T : Finset ℕ), (T ⊆ S_n) → 
                     ∃ (k : ℕ), ∏ i in T, i = k ^ T.card) := 
sorry

-- Problem (2) Statement
theorem not_exists_infinite_set_geometric_mean_integer :
  ¬(∃ S : Set ℕ, (S.infinite ∧
                  ∀ (T : Finset ℕ), (T ⊆ S) → 
                  ∃ (k : ℕ), ∏ i in T, i = k ^ T.card)) :=
sorry

end exists_finite_set_geometric_mean_integer_not_exists_infinite_set_geometric_mean_integer_l3_3391


namespace probability_two_cards_l3_3344

noncomputable def probability_first_spade_second_ace : ℚ :=
  let total_cards := 52
  let total_spades := 13
  let total_aces := 4
  let remaining_cards := total_cards - 1
  
  let first_spade_non_ace := (total_spades - 1) / total_cards
  let second_ace_after_non_ace := total_aces / remaining_cards
  
  let probability_case1 := first_spade_non_ace * second_ace_after_non_ace
  
  let first_ace_spade := 1 / total_cards
  let second_ace_after_ace := (total_aces - 1) / remaining_cards
  
  let probability_case2 := first_ace_spade * second_ace_after_ace
  
  probability_case1 + probability_case2

theorem probability_two_cards {p : ℚ} (h : p = 1 / 52) : 
  probability_first_spade_second_ace = p := 
by 
  simp only [probability_first_spade_second_ace]
  sorry

end probability_two_cards_l3_3344


namespace range_of_a_l3_3387

theorem range_of_a (a : ℝ) : (-1/Real.exp 1 < a ∧ a < 0) ∨ (0 < a ∧ a < 1/Real.exp 1) :=
  sorry

end range_of_a_l3_3387


namespace distinct_values_even_sum_product_l3_3458

theorem distinct_values_even_sum_product :
  let evens := {n // n < 15 ∧ n % 2 = 0} in
  let expr (p q : ℕ) := p * q + p + q in
  let values := {expr p.val q.val | p q : {n // n < 15 ∧ n % 2 = 0}} in
  values.card = 27 :=
by
  sorry

end distinct_values_even_sum_product_l3_3458


namespace inequality_sum_l3_3525

-- Definitions based on conditions
def a (n : ℕ) : ℝ := 2 + ((3/2) * (n - 1))
def b (n : ℕ) : ℝ := 2 ^ n
def S (n : ℕ) : ℝ := (3 / 4) * n ^ 2 + (5 / 4) * n
def c (n : ℕ) : ℝ :=
  if n % 2 = 1 then a n else a n / b n

-- Sum of the first 2n terms of sequence c
def sum_c (n : ℕ) : ℝ :=
  (∑ k in finset.range n, a (2 * k + 1)) + 
  (∑ k in finset.range n, (a (2 * k + 2)) / b (2 * k + 2))

-- The theorem to be proven
theorem inequality_sum {n : ℕ} (hn : n > 0) :
  (∑ i in finset.range n, (1 / (b (i + 1) * real.sqrt (S (i + 1))))) < (17 / 24) :=
sorry

end inequality_sum_l3_3525


namespace part1_part2_l3_3173

-- Problem 1
theorem part1 (x : ℝ) (a : ℝ) (h : a = 2) : (|4 * x + 1| - |4 * x + a| + x < 0) -> x ∈ Iio (1/9) :=
sorry

-- Problem 2
theorem part2 (a : ℝ) (h : ∃ x : ℝ, (|4 * x + 1| - |4 * x + a|) ≤ -5) : a ∈ Iic (-4) ∨ a ∈ Ici (6) :=
sorry

end part1_part2_l3_3173


namespace range_of_x_when_a_equals_1_range_of_a_l3_3981

variable {a x : ℝ}

-- Definitions for conditions p and q
def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := (x - 3) / (x - 2) < 0

-- Part (1): Prove the range of x when a = 1 and p ∨ q is true.
theorem range_of_x_when_a_equals_1 (h : a = 1) (h1 : p 1 x ∨ q x) : 1 < x ∧ x < 3 :=
by sorry

-- Part (2): Prove the range of a when p is a necessary but not sufficient condition for q.
theorem range_of_a (h2 : ∀ x, q x → p a x) (h3 : ¬ ∀ x, p a x → q x) : 1 ≤ a ∧ a ≤ 2 :=
by sorry

end range_of_x_when_a_equals_1_range_of_a_l3_3981


namespace prime_divisible_by_57_is_zero_l3_3562

open Nat

theorem prime_divisible_by_57_is_zero :
  (∀ p, Prime p → (57 ∣ p) → False) :=
by
  intro p hp hdiv
  have h57 : 57 = 3 * 19 := by norm_num
  have h1 : p = 57 ∨ p = 3 ∨ p = 19 := sorry
  have hp1 : p ≠ 57 := sorry
  have hp2 : p ≠ 3 := sorry
  have hp3 : p ≠ 19 := sorry
  exact Or.elim h1 hp1 (Or.elim hp2 hp3)


end prime_divisible_by_57_is_zero_l3_3562


namespace ants_meeting_time_l3_3702

open Real

theorem ants_meeting_time :
  ∃ t : ℕ, t = Nat.lcm 3 2 := sorry

end ants_meeting_time_l3_3702


namespace intersection_of_medians_divide_in_ratio_l3_3946

variable {α : Type} [LinearOrder α]

-- Define a triangle as a set of three points
structure Triangle (α : Type) [LinearOrder α] :=
(A B C : α)

-- Define a median from a vertex to the midpoint of the opposite side
def isMedian (T : Triangle α) (K O : α) : Prop := 
  ∃ M, M = midpoint T.B T.C ∧ line_segment T.A M = line_segment T.A K

-- Define the property of the intersection point of the two medians
def divides_in_ratio (T : Triangle α) (O : α) (ratio : ℝ) : Prop :=
  ∃ (A K C L : α),
  isMedian T K O ∧ isMedian T L O ∧
  distance T.A O / distance O K = ratio ∧
  distance <| T.C O / distance O L = ratio

-- The theorem statement
theorem intersection_of_medians_divide_in_ratio (T : Triangle α) (O : α) :
  divides_in_ratio T O 2 :=
sorry

end intersection_of_medians_divide_in_ratio_l3_3946


namespace winning_candidate_votes_l3_3696

theorem winning_candidate_votes 
  (V W : ℝ)
  (h1 : W = 71.42857142857143 / 100 * V)
  (h2 : V = W + 5000 + 20000) : 
  W ≈ 62500 :=
by
  sorry

end winning_candidate_votes_l3_3696


namespace pond_length_is_8_l3_3685

noncomputable def pond_length (L W A_field A_pond L_pond : ℝ) :=
  L = 112 ∧ L = 2 * W ∧ A_field = L * W ∧ A_pond = (1 / 98) * A_field ∧ L_pond = real.sqrt A_pond

theorem pond_length_is_8 : pond_length 112 56 6272 64 8 :=
by
  unfold pond_length
  simp
  sorry -- proof not required

end pond_length_is_8_l3_3685


namespace hyperbola_standard_equation_l3_3675

theorem hyperbola_standard_equation
  (h1 : ∀ x y : ℝ, y = (1/3) * x ∨ y = -(1/3) * x)
  (h2 : (0, 2 * real.sqrt 5) ∈ set_of {p : ℝ × ℝ | ∀ h k: ℝ, p = (0, 2 * real.sqrt 5)}) :
  ∀ (x y : ℝ), (y^2) / 2 - (x^2) / 18 = 1 :=
sorry

end hyperbola_standard_equation_l3_3675


namespace hidden_die_letter_l3_3088

theorem hidden_die_letter (dice : Fin 8 → char) 
  (hidden : Fin 8) :
  (∀ i j: Fin 8, i ≠ j → touching i j → dice i ≠ dice j) → 
  (dice (above hidden) = 'R') → 
  (dice (below_left hidden) = 'P') → 
  (dice (below_right hidden) = 'S') → 
  dice hidden = 'Q' :=
by
  sorry

end hidden_die_letter_l3_3088


namespace log_inequality_l3_3628

theorem log_inequality (t : ℝ) (n : ℕ) (ht : t > 0) :
  (∑ i in finset.range (2*n+1), (-1)^i * (t^(i+1)) / (i+1)) < 
  real.log(1 + t) ∧ 
  real.log(1 + t) < 
  (∑ i in finset.range (2*n+2), (-1)^i * (t^(i+1)) / (i+1)) := by
  sorry

end log_inequality_l3_3628


namespace factor_polynomial_l3_3539

variable {R : Type*} [CommRing R]

theorem factor_polynomial (a b c : R) :
  a^4 * (b^3 - c^3) + b^4 * (c^3 - a^3) + c^4 * (a^3 - b^3) =
  (a - b) * (b - c) * (c - a) * (-(a + b + c) * (a^2 + b^2 + c^2 + ab + bc + ac)) :=
by
  sorry

end factor_polynomial_l3_3539


namespace gcd_a5000_b501_l3_3689

-- Define the sequence a_n according to the given recurrence relation and initial values
noncomputable def a : ℕ → ℕ 
| 1     := 1
| 2     := 8
| (n+2) := 7 * a (n + 1) - a n

-- Define the sequence b_n according to the given recurrence relation and initial values
noncomputable def b : ℕ → ℕ 
| 1     := 1
| 2     := 2
| (n+2) := 3 * b (n + 1) - b n

-- Now use the conditions to state the gcd problem.
theorem gcd_a5000_b501 : 
  Int.gcd (a 5000) (b 501) = 89 := 
by sorry

end gcd_a5000_b501_l3_3689


namespace john_trip_to_distant_city_l3_3965

theorem john_trip_to_distant_city (t : ℝ) :
  (∀ d₁ : ℝ, ∀ d₂ : ℝ, ∀ dist : ℝ,
    d₁ = 60 * t ∧ d₂ = 72 * 5 ∧ dist = d₁ ∧ d₁ = d₂) → t = 30 :=
by 
  intros,
  sorry

end john_trip_to_distant_city_l3_3965


namespace overlap_region_area_l3_3772

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  1/2 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

noncomputable def overlap_area : ℝ := 
  let A : ℝ × ℝ := (0, 0);
  let B : ℝ × ℝ := (6, 2);
  let C : ℝ × ℝ := (2, 6);
  let D : ℝ × ℝ := (6, 6);
  let E : ℝ × ℝ := (0, 2);
  let F : ℝ × ℝ := (2, 0);
  let P1 : ℝ × ℝ := (2, 2);
  let P2 : ℝ × ℝ := (4, 2);
  let P3 : ℝ × ℝ := (3, 3);
  let P4 : ℝ × ℝ := (2, 3);
  1/2 * abs (P1.1 * (P2.2 - P4.2) + P2.1 * (P3.2 - P1.2) + P3.1 * (P4.2 - P2.2) + P4.1 * (P1.2 - P3.2))

theorem overlap_region_area :
  let A : ℝ × ℝ := (0, 0);
  let B : ℝ × ℝ := (6, 2);
  let C : ℝ × ℝ := (2, 6);
  let D : ℝ × ℝ := (6, 6);
  let E : ℝ × ℝ := (0, 2);
  let F : ℝ × ℝ := (2, 0);
  triangle_area A B C > 0 →
  triangle_area D E F > 0 →
  overlap_area = 0.5 :=
by { sorry }

end overlap_region_area_l3_3772


namespace altered_solution_ratio_l3_3206

variable (b d w : ℕ)
variable (b' d' w' : ℕ)
variable (ratio_orig_bd_ratio_orig_dw_ratio_orig_bw : Rat)
variable (ratio_new_bd_ratio_new_dw_ratio_new_bw : Rat)

noncomputable def orig_ratios (ratio_orig_bd ratio_orig_bw : Rat) (d w : ℕ) : Prop := 
    ratio_orig_bd = 2 / 40 ∧ ratio_orig_bw = 40 / 100

noncomputable def new_ratios (ratio_new_bd : Rat) (d' : ℕ) : Prop :=
    ratio_new_bd = 6 / 40 ∧ d' = 60

noncomputable def new_solution (w' : ℕ) : Prop :=
    w' = 300

theorem altered_solution_ratio : 
    ∀ (orig_ratios: Prop) (new_ratios: Prop) (new_solution: Prop),
    orig_ratios ∧ new_ratios ∧ new_solution →
    (d' / w = 2 / 5) :=
by
    sorry

end altered_solution_ratio_l3_3206


namespace ellie_loan_difference_l3_3475

noncomputable def principal : ℝ := 8000
noncomputable def simple_rate : ℝ := 0.10
noncomputable def compound_rate : ℝ := 0.08
noncomputable def time : ℝ := 5
noncomputable def compounding_periods : ℝ := 1

noncomputable def simple_interest_total (P r t : ℝ) : ℝ :=
  P + (P * r * t)

noncomputable def compound_interest_total (P r t n : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem ellie_loan_difference :
  (compound_interest_total principal compound_rate time compounding_periods) -
  (simple_interest_total principal simple_rate time) = -245.36 := 
  by sorry

end ellie_loan_difference_l3_3475


namespace triangle_BCD_area_l3_3209

variable (A B C D : Type) [LinearOrder A]
variable (area : Type → ℝ)
variable (length : Type → ℝ)
variable (triangle : Type → Type)
variable (h : Type → ℝ)
variable (l : Type → Type → ℝ)

theorem triangle_BCD_area
  (h : Type) [LinearOrder h]
  {ΔABC ΔBCD : triangle Type} {AC CD : Type}
  (area_ΔABC_45 : area ΔABC = 45)
  (length_AC_10 : length AC = 10)
  (length_CD_30 : length CD = 30)
  (h_AC_9 : ∃ h, area ΔABC = 1/2 * length AC * h ∧ h = 9) :
  area ΔBCD = 135 := by
  sorry

end triangle_BCD_area_l3_3209


namespace temperature_comparison_l3_3760

theorem temperature_comparison: ¬ (-3 > -0.3) :=
by
  sorry -- Proof goes here, skipped for now.

end temperature_comparison_l3_3760


namespace triangle_BC_squared_compute_ab_value_l3_3615

-- Problem 1
theorem triangle_BC_squared (A B C D E F : Type)
  [ht : ∃ (A B C : Type) [P : Triangle A B C], 
      (AB : Segment A B 3) (AC : Segment A C 5) (h_perp : Perp (AD, BC)) 
      (h_mid : Midpoint E D F) (h_BAE_eq_CAE: ∠BAE = ∠CAE)
      (h_BF_eq_CF: BF = CF)] :
  (BC^2 = 64) := sorry

-- Problem 2
theorem compute_ab_value (a b : ℝ) (X : ℤ) (Y : ℤ)
  (h_eq1 : a + 1 / b = Y)
  (h_eq2 : b / a = X) :
  ((a * b)^4 + 1 / (a * b)^4) = 9602 := sorry

end triangle_BC_squared_compute_ab_value_l3_3615


namespace medium_stores_in_sample_l3_3205

theorem medium_stores_in_sample :
  let total_stores := 300
  let large_stores := 30
  let medium_stores := 75
  let small_stores := 195
  let sample_size := 20
  sample_size * (medium_stores/total_stores) = 5 :=
by
  sorry

end medium_stores_in_sample_l3_3205


namespace proof_solve_x_l3_3663

def solve_x : ℤ :=
  let x := 260 / 7
  x

theorem proof_solve_x : solve_x = 37.142857 :=
by
  -- The conditions and manipulations are implicitly handled in the definition and theorem.
  sorry

end proof_solve_x_l3_3663


namespace range_of_abscissa_of_P_l3_3529

noncomputable def line (P : ℝ × ℝ) : Prop := P.1 - P.2 + 1 = 0

noncomputable def circle (M : ℝ × ℝ) : Prop := (M.1 - 2) ^ 2 + (M.2 - 1) ^ 2 = 1

noncomputable def condition_on_P (P : ℝ × ℝ) : Prop :=
  line P ∧ ∃ M N : ℝ × ℝ, circle M ∧ circle N ∧ angle_eq60 M P N

theorem range_of_abscissa_of_P :
  ∃ x : set ℝ, ∀ P : ℝ × ℝ, condition_on_P P → P.1 ∈ set.range x → P.1 ∈ set.Icc 0 2 :=
sorry

end range_of_abscissa_of_P_l3_3529


namespace sum_ai_over_s_minus_ai_l3_3988

theorem sum_ai_over_s_minus_ai (n : ℕ) (hn : n > 1) (a : Fin n → ℝ) (hpos : ∀ i, a i > 0) (s : ℝ) (hsum : ∑ i, a i = s) :
  ∑ i in Fin.elems (Fin n), a i / (s - a i) ≥ n / (n - 1) :=
sorry

end sum_ai_over_s_minus_ai_l3_3988


namespace lacy_correct_percentage_l3_3645

variables (x : ℝ)

-- Definitions based on conditions
def total_problems : ℝ := 6 * x
def missed_problems : ℝ := 2 * x
def correct_problems : ℝ := total_problems - missed_problems
def percent_correct : ℝ := (correct_problems / total_problems) * 100

-- Statement to prove
theorem lacy_correct_percentage :
  percent_correct x = 66.67 :=
sorry

end lacy_correct_percentage_l3_3645


namespace necessary_but_not_sufficient_condition_l3_3324

theorem necessary_but_not_sufficient_condition (p : ℝ) : 
  p < 2 → (¬(p^2 - 4 < 0) → ∃ q, q < p ∧ q^2 - 4 < 0) :=
by
  sorry

end necessary_but_not_sufficient_condition_l3_3324


namespace weighted_means_inequalities_weighted_means_equality_special_case_l3_3990

variables {p1 p2 x1 x2 : ℝ}
variables (hp1 : 0 < p1) (hp2 : 0 < p2)

noncomputable def q1 := p1 / (p1 + p2)
noncomputable def q2 := p2 / (p1 + p2)

theorem weighted_means_inequalities :
  (1 / (q1 x1 / x1 + q2 x1 / x2)) ≤ (q1 x1 * x1) + (q2 x2 * x2) ∧
  (q1 x1 * x1) + (q2 x2 * x2) ≤ real.sqrt (q1 x1 * x1^2 + q2 x2 * x2^2) :=
by
  sorry

theorem weighted_means_equality_special_case :
  p1 = p2 →
  real.sqrt ((x1^2 + x2^2) / 2) ≥ ((x1 + x2) / 2) ∧
  ((x1 + x2) / 2) ≥ (2 * x1 * x2 / (x1 + x2)) :=
by
  sorry

end weighted_means_inequalities_weighted_means_equality_special_case_l3_3990


namespace find_a_for_pure_imaginary_quotient_l3_3192

theorem find_a_for_pure_imaginary_quotient (a : ℝ) :
  let z1 := a + 2 * complex.I,
      z2 := 3 - 4 * complex.I,
      quotient := z1 / z2 in
  (quotient.re = 0) → a = 8 / 3 :=
by
  sorry

end find_a_for_pure_imaginary_quotient_l3_3192


namespace probability_of_one_absent_l3_3215

open Classical

noncomputable def probability_one_absent_two_present_in_three (p_absent : ℚ) (p_present : ℚ) :=
  p_absent * p_present * p_present * 3

theorem probability_of_one_absent :
  let p_absent := (1 : ℚ) / 40
  let p_present := 1 - p_absent
  let prob := probability_one_absent_two_present_in_three p_absent p_present in
  prob * 100 ≈ 7.1 :=
by
  have p_absent : ℚ := 1 / 40
  have p_present : ℚ := 1 - p_absent
  have prob := probability_one_absent_two_present_in_three p_absent p_present
  have rounded := Float.ofRat (prob * 100).toFloat
  have percent_rounded := rounded.toRat.round
  show percent_rounded ≈ 7.1
  sorry

end probability_of_one_absent_l3_3215


namespace fourth_root_of_256000000_is_400_l3_3816

def x : ℕ := 256000000

theorem fourth_root_of_256000000_is_400 : Nat.root x 4 = 400 := by
  sorry

end fourth_root_of_256000000_is_400_l3_3816


namespace correct_statement_l3_3190

theorem correct_statement (a b : ℝ) (ha : a < b) (hb : b < 0) : |a| / |b| > 1 :=
sorry

end correct_statement_l3_3190


namespace possible_values_of_P_l3_3987

variable (X : Set (Fin n)) (P : Set (Set (Fin n)))

theorem possible_values_of_P (h_subset : 
  (∀ (A B : Set (Fin n)), 
    A ∈ P → 
    B ∈ P →
    (X \ A) ∈ P ∧ 
    (A ∪ B) ∈ P ∧ 
    (A ∩ B) ∈ P)) : 
  ∃ (m : ℕ), ∃ (k : ℕ), k ≤ n ∧ |P| = 2^m := 
sorry

end possible_values_of_P_l3_3987


namespace decaf_coffee_percent_l3_3052

theorem decaf_coffee_percent (initial_stock : ℕ) (initial_decaf_percent : ℚ) 
                             (additional_stock : ℕ) (additional_decaf_percent : ℚ) :
  initial_stock = 400 →
  initial_decaf_percent = 0.25 →
  additional_stock = 100 →
  additional_decaf_percent = 0.60 →
  (let total_decaf := initial_stock * initial_decaf_percent + additional_stock * additional_decaf_percent,
       total_stock := initial_stock + additional_stock,
       decaf_percent := (total_decaf / total_stock) * 100 in
    decaf_percent = 32) :=
by
  intros h1 h2 h3 h4
  let total_decaf := initial_stock * initial_decaf_percent + additional_stock * additional_decaf_percent
  let total_stock := initial_stock + additional_stock
  let decaf_percent := (total_decaf / total_stock) * 100
  sorry

end decaf_coffee_percent_l3_3052


namespace max_possible_cups_l3_3335

theorem max_possible_cups (a b : ℕ) 
  (h : Nat.choose a 2 * Nat.choose b 3 = 1200) : 
  a + b ≤ 29 :=
begin
  sorry
end

end max_possible_cups_l3_3335


namespace crease_length_l3_3231

theorem crease_length 
  (AB AC : ℝ) (BC : ℝ) (BA' : ℝ) (A'C : ℝ)
  (h1 : AB = 10) (h2 : AC = 10) (h3 : BC = 8) (h4 : BA' = 3) (h5 : A'C = 5) :
  ∃ PQ : ℝ, PQ = (Real.sqrt 7393) / 15 := by
  sorry

end crease_length_l3_3231


namespace problem1_solution_problem2_solution_l3_3758

noncomputable def problem1_expr : ℝ :=
  (real.cbrt (-8) + abs (3 - 2 * real.sqrt 3) - (real.sqrt 12 + real.sqrt 27) / real.sqrt 3)

theorem problem1_solution : problem1_expr = -10 + 2 * real.sqrt 3 :=
by simp [real.cbrt, real.sqrt, abs]; sorry


variables (x y : ℝ)

-- Defining the equations for problem 2
def eq1 := 2 * x + 3 * y = 17
def eq2 := (x + y) / 2 = y - 2

theorem problem2_solution : eq1 x y ∧ eq2 x y → x = 1 ∧ y = 5 :=
by simp [eq1, eq2]; sorry

end problem1_solution_problem2_solution_l3_3758


namespace triangle_inequality_l3_3606

-- Definitions of terms and expressions
variables {α : Type*} [OrderedField α]
variables (A B C A1 B1 C1 O : α)
variables (R r : ℝ) -- circumradius and inradius

-- Assumptions
def triangle (A B C : α) : Prop := True
def angle_bisectors_intersect_at (A B C A1 B1 C1 O : α) : Prop := True -- simplified assumption for angle bisectors intersecting conditions

-- Theorem statement
theorem triangle_inequality 
  (h1 : triangle A B C)
  (h2 : angle_bisectors_intersect_at A B C A1 B1 C1 O)
  (h3: circumradius A B C = R)
  (h4: inradius A B C = r)
  : 8 ≤ (AO / A1O) * (BO / B1O) * (CO / C1O) ∧ (AO / A1O) * (BO / B1O) * (CO / C1O) ≤ (4 * R) / r := 
sorry

end triangle_inequality_l3_3606


namespace range_of_f_l3_3854

theorem range_of_f:
  ∀ x : ℝ, (sin x ≠ 2) →
  let f := (λ x, (sin x)^3 + 10 * (sin x)^2 + 3 * (sin x) + 4 * (1 - (sin x)^2) - 12) / (sin x - 2)
  (f x) ∈ set.Icc (-4 : ℝ) 0 :=
by
  intros x h
  let f := λ x, ((sin x)^3 + 10 * (sin x)^2 + 3 * (sin x) + 4 * (1 - (sin x)^2) - 12) / (sin x - 2)
  sorry

end range_of_f_l3_3854


namespace radius_of_sphere_l3_3773

theorem radius_of_sphere 
  (shadow_length_sphere : ℝ)
  (stick_height : ℝ)
  (stick_shadow : ℝ)
  (parallel_sun_rays : Prop) 
  (tan_θ : ℝ) 
  (h1 : tan_θ = stick_height / stick_shadow)
  (h2 : tan_θ = shadow_length_sphere / 20) :
  shadow_length_sphere / 20 = 1/4 → shadow_length_sphere = 5 := by
  sorry

end radius_of_sphere_l3_3773


namespace problem_statement_l3_3740

def vector := (ℝ × ℝ × ℝ)

def are_parallel (v1 v2 : vector) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ v1 = (k * v2.1, k * v2.2, k * v2.3)

def are_perpendicular (v1 v2 : vector) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3 = 0

theorem problem_statement :
  (are_parallel (2, 3, -1) (-2, -3, 1)) ∧
  (¬ are_perpendicular (1, -1, 2) (6, 4, -1)) ∧
  (are_perpendicular (2, 2, -1) (-3, 4, 2)) ∧
  (¬ are_parallel (0, 3, 0) (0, -5, 0)) :=
by
  sorry

end problem_statement_l3_3740


namespace circumcenter_DEX_perp_bisector_OA_l3_3411

open EuclideanGeometry

variables (A B C X D E O : Point) (h_circum_o : C = circumcenter (triangle A B X)) 
(h_cyclic_quad : cyclic A B X C) (h_AD_eq_BD : dist A D = dist B D) 
(h_AE_eq_CE : dist A E = dist C E)

theorem circumcenter_DEX_perp_bisector_OA : 
    let O1 := circumcenter (triangle D E X) in
    point_on_perpendicular_bisector O1 A O :=
begin
  sorry
end

end circumcenter_DEX_perp_bisector_OA_l3_3411


namespace area_of_large_rectangle_ABCD_l3_3678

-- Definitions for conditions and given data
def shaded_rectangle_area : ℕ := 2
def area_of_rectangle_ABCD (a b c : ℕ) : ℕ := a + b + c

-- The theorem to prove
theorem area_of_large_rectangle_ABCD
  (a b c : ℕ) 
  (h1 : shaded_rectangle_area = a)
  (h2 : shaded_rectangle_area = b)
  (h3 : a + b + c = 8) : 
  area_of_rectangle_ABCD a b c = 8 :=
by
  sorry

end area_of_large_rectangle_ABCD_l3_3678


namespace quadratic_roots_two_l3_3955

theorem quadratic_roots_two (a b c : ℝ) (h : a * c < 0) : 
  let Δ := b^2 - 4 * a * c in Δ > 0 → ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0 := 
by
  sorry

end quadratic_roots_two_l3_3955


namespace pm_nq_eq_one_min_area_apq_l3_3757

-- Steps to define the geometrical setup using Lean
variables {A B C I P Q M N : Type} [RightTriangle ABC (90 : Real)] [Incenter I ABC]
variables {AP AQ : Real}
variables (x y : Real) (h_dist : Distance I BC = 1)

-- Part (a): PM * NQ = 1
theorem pm_nq_eq_one (h_dist : Distance I BC = 1) : PM * NQ = 1 := by
  sorry

-- Part (b): Minimum area of triangle APQ
theorem min_area_apq (h_xy_is_1: x * y = 1) (h_ineq: x + y ≥ 2 * √ (x * y)) : 
  ∃ (min_area : Real), min_area = 2 := by
  sorry

end pm_nq_eq_one_min_area_apq_l3_3757


namespace football_points_difference_l3_3447

theorem football_points_difference :
  let points_per_touchdown := 7
  let brayden_gavin_touchdowns := 7
  let cole_freddy_touchdowns := 9
  let brayden_gavin_points := brayden_gavin_touchdowns * points_per_touchdown
  let cole_freddy_points := cole_freddy_touchdowns * points_per_touchdown
  cole_freddy_points - brayden_gavin_points = 14 :=
by sorry

end football_points_difference_l3_3447


namespace lakers_win_nba_l3_3669

noncomputable def probability_lakers_win (p : ℕ) : ℕ :=
  rounded_nearest (100 * (prob_four_wins p (2/3)))

theorem lakers_win_nba 
  (prob_win : ℚ := 2/3) 
  (no_ties : ∀ (g : Game), g.tie = false) :
  probability_lakers_win 4 = 80 := sorry

/-- Helper definition to calculate probability of winning 4 games given that 
     the probability of winning a single game is p. This constructs the 
     combination and binomial calculation for each case of k wins by the Celtics. -/
def prob_four_wins (p : ℚ) : ℚ :=
  (p^4) + 
  (binomial (1 + 3) 1) * (p^4) * ((1-p)^1) + 
  (binomial (2 + 3) 2) * (p^4) * ((1-p)^2) + 
  (binomial (3 + 3) 3) * (p^4) * ((1-p)^3)

def rounded_nearest (q : ℚ) : ℕ :=
  let num : ℕ := q.num;
  let den : ℕ := q.denom;
  let rnd : ℕ := if (10 * (num % den)) / den >= 5 then (num + den - 1) / den else num / den;
  rnd

end lakers_win_nba_l3_3669


namespace geometric_sequence_general_formula_lambda_value_l3_3160

theorem geometric_sequence_general_formula :
  ∃ a_n : ℕ → ℝ, (∀ n, 0 ≤ a_n n) ∧ a_n 1 = 1 ∧ a_n 4 + 3 * a_n 3 + a_n 5 = (a_n 3 + a_n 4 + a_n 5) / 3 → (∀ n, a_n n = 2^(n-1)) :=
sorry

theorem lambda_value :
  ∀ (λ : ℝ) (b : ℕ → ℝ), (∀ n, b n = 2^n - λ * 2^(n-1)) ∧ (∃ S_n : ℕ → ℝ, ∀ n, S_n n = 2^n - 1) → λ = 1 :=
sorry

end geometric_sequence_general_formula_lambda_value_l3_3160


namespace part_one_part_two_case1_part_two_case2_part_two_case3_l3_3271

noncomputable def f (x : ℝ) : ℝ := |2 * x + 1| + x

theorem part_one : 
  { x : ℝ | f x ≤ 5 } = set.Icc (-6 : ℝ) (4 / 3 : ℝ) :=
by
  sorry -- Proof to be provided

noncomputable def g (a x : ℝ) := f x - |a * x - 1| - x

theorem part_two_case1 (a : ℝ) (h : a > 2) : 
  set.range (g a) = set.Iic (2 / a + 1) := 
by
  sorry -- Proof to be provided

theorem part_two_case2 (a : ℝ) (h : 0 < a) (h' : a < 2) : 
  set.range (g a) = set.Ioi (-a / 2 - 1) := 
by
  sorry -- Proof to be provided

theorem part_two_case3 (a : ℝ) (h : a = 2) : 
  set.range (g a) = set.Icc (-2 : ℝ) (2 : ℝ) := 
by
  sorry -- Proof to be provided

end part_one_part_two_case1_part_two_case2_part_two_case3_l3_3271


namespace sum_of_elements_in_A_star_B_l3_3519

-- Definitions of the sets A and B
def A := {4, 5, 6}
def B := {2, 3}

-- Definition of the operation A * B
def star (A B : Set ℤ) : Set ℤ := {x | ∃ (m ∈ A) (n ∈ B), x = m - n}

-- The sets as Lean sets
def setA : Set ℤ := {4, 5, 6}
def setB : Set ℤ := {2, 3}

-- The set (A * B) according to the given operation
def A_star_B : Set ℤ := star setA setB

-- We now state the theorem
theorem sum_of_elements_in_A_star_B : ((star setA setB).toFinset.sum id) = 10 := by
  sorry

end sum_of_elements_in_A_star_B_l3_3519


namespace total_points_scored_l3_3970

theorem total_points_scored (layla_score nahima_score : ℕ)
  (h1 : layla_score = 70)
  (h2 : layla_score = nahima_score + 28) :
  layla_score + nahima_score = 112 :=
by
  sorry

end total_points_scored_l3_3970


namespace points_difference_l3_3445

-- Define the given data
def points_per_touchdown : ℕ := 7
def brayden_gavin_touchdowns : ℕ := 7
def cole_freddy_touchdowns : ℕ := 9

-- Define the theorem to prove the difference in points
theorem points_difference :
  (points_per_touchdown * cole_freddy_touchdowns) - 
  (points_per_touchdown * brayden_gavin_touchdowns) = 14 :=
  by sorry

end points_difference_l3_3445


namespace diane_age_proof_l3_3025

noncomputable def diane_age (A Al D : ℕ) : Prop :=
  ((A + (30 - D) = 60) ∧ (Al + (30 - D) = 15) ∧ (A + Al = 47)) → (D = 16)

theorem diane_age_proof : ∃ (D : ℕ), ∃ (A Al : ℕ), diane_age A Al D :=
by {
  sorry
}

end diane_age_proof_l3_3025


namespace smart_integers_divisible_by_25_fraction_l3_3461

def is_smart_integer (n : ℤ) : Prop :=
  n % 2 = 0 ∧ 20 < n ∧ n < 120 ∧ (n.digits.map (fun x => x.to_nat)).sum = 10

theorem smart_integers_divisible_by_25_fraction : 
  (∃ n : ℤ, is_smart_integer n ∧ n % 25 = 0) = false :=
by
  sorry

end smart_integers_divisible_by_25_fraction_l3_3461


namespace telescoping_series_sum_l3_3818

open BigOperators

theorem telescoping_series_sum :
  ∑ n in Finset.range 500 | 1 ≤ n, (1 / (n^2 + 2 * n)) = 3 / 4 :=
by
  sorry

end telescoping_series_sum_l3_3818


namespace gauss_lucas_theorem_l3_3286

noncomputable def convex_hull (s : Set ℂ) : Set ℂ := sorry

theorem gauss_lucas_theorem (P : ℂ[X]) (roots : List ℂ) (h_roots : ∀ z ∈ roots, P.eval z = 0) :
  ∀ w, w ∈ (P.derivative).roots → w ∈ convex_hull (↑roots.to_finset) :=
sorry

end gauss_lucas_theorem_l3_3286


namespace polar_coord_eq_min_distance_l3_3950

noncomputable def line_eq (x y : ℝ) : Prop := x - y + 4 = 0

noncomputable def param_eq (α : ℝ) : ℝ × ℝ :=
  (sqrt 3 * Real.cos α, Real.sin α)

theorem polar_coord_eq (x y ρ θ : ℝ) (hx: x = ρ * Real.cos θ)
  (hy : y = ρ * Real.sin θ) (hl : line_eq x y) : 
  ρ * Real.sin (θ - π/4) = 2 * sqrt 2 := by
  sorry

theorem min_distance (α : ℝ) (Q : ℝ × ℝ) (Q_eq : Q = param_eq α)
  (d : ℝ) (hd : d = abs (√3 * Real.cos α - Real.sin α + 4) / sqrt 2) :
  ∃ d_min : ℝ, d_min = sqrt 2 ∧ ∀ d', d' = hd → d' ≥ d_min := by
  sorry

end polar_coord_eq_min_distance_l3_3950


namespace circles_intersect_l3_3835

-- Define the first circle C_1
def circle1 : set (ℝ × ℝ) := { p | (p.1 + 1)^2 + p.2^2 = 4 }

-- Define the second circle C_2
def circle2 : set (ℝ × ℝ) := { p | p.1^2 + (p.2 - 2)^2 = 1 }

-- Assume we have the centers and radii information
def center1 := (-1, 0) -- Center of C_1
def radius1 := 2 -- Radius of C_1
def center2 := (0, 2) -- Center of C_2
def radius2 := 1 -- Radius of C_2

-- Calculate the distance between centers
def distance_centers := Real.sqrt ((center1.1 - center2.1)^2 + (center1.2 - center2.2)^2)

-- Prove the circles are intersecting
theorem circles_intersect (h1 : distance_centers = Real.sqrt 5)
                          (h2 : radius1 = 2)
                          (h3 : radius2 = 1) :
                          1 < distance_centers ∧ distance_centers < 3 :=
by {
  have h_dist : distance_centers = Real.sqrt 5 := h1,
  have h_r : radius1 = 2 := h2,
  have h_R : radius2 = 1 := h3,
  sorry -- Proof steps are not required
}

end circles_intersect_l3_3835


namespace ab_minus_a_inv_b_l3_3570

theorem ab_minus_a_inv_b (a : ℝ) (b : ℚ) (h1 : a > 1) (h2 : 0 < (b : ℝ)) (h3 : (a ^ (b : ℝ)) + (a ^ (-(b : ℝ))) = 2 * Real.sqrt 2) :
  (a ^ (b : ℝ)) - (a ^ (-(b : ℝ))) = 2 := 
sorry

end ab_minus_a_inv_b_l3_3570


namespace train_length_approx_200_l3_3794

noncomputable def train_length (speed_kmph : ℕ) (time_sec : ℕ) : ℝ :=
  (speed_kmph * 1000) / 3600 * time_sec

theorem train_length_approx_200
  (speed_kmph : ℕ)
  (time_sec : ℕ)
  (h_speed : speed_kmph = 120)
  (h_time : time_sec = 6) :
  train_length speed_kmph time_sec ≈ 200 := 
by sorry

end train_length_approx_200_l3_3794


namespace wooden_box_height_l3_3066

def volume_rectangular_box (length : ℝ) (width : ℝ) (height : ℝ) : ℝ :=
  length * width * height

def total_volume (box_volume : ℝ) (num_boxes : ℕ) : ℝ :=
  box_volume * num_boxes

theorem wooden_box_height (length_wb : ℝ) (width_wb : ℝ) (num_boxes : ℕ) 
[length_wb = 8] [width_wb = 7] [num_boxes = 2000000] : 
  let length_rb := 0.04 in 
  let width_rb := 0.07 in 
  let height_rb := 0.06 in
  let volume_rb := volume_rectangular_box length_rb width_rb height_rb in
  let total_vol := total_volume volume_rb num_boxes in 
  height :=
  total_vol / (length_wb * width_wb)
:=
  sorry

end wooden_box_height_l3_3066


namespace dartboard_area_ratio_l3_3788

theorem dartboard_area_ratio
    (larger_square_side_length : ℝ)
    (inner_square_side_length : ℝ)
    (angle_division : ℝ)
    (s : ℝ)
    (p : ℝ)
    (h1 : larger_square_side_length = 4)
    (h2 : inner_square_side_length = 2)
    (h3 : angle_division = 45)
    (h4 : s = 1/4)
    (h5 : p = 3) :
    p / s = 12 :=
by
    sorry

end dartboard_area_ratio_l3_3788


namespace number_of_non_similar_regular_12_pointed_double_layered_stars_l3_3103

def vertices := Finset.range 12

def is_coprime_with_12 (m: ℕ): Prop := Nat.gcd m 12 = 1

def star_vertices (k m: ℕ): Finset (ℕ × ℕ) :=
  Finset.singleton (k % 12, (k + m) % 12) ∪ Finset.singleton (k % 12, (k + 2*m) % 12)

noncomputable def number_of_distinct_stars: ℕ := 2

theorem number_of_non_similar_regular_12_pointed_double_layered_stars: 
  (∃ m_values : Finset ℕ, (∀ m ∈ m_values, is_coprime_with_12 m) ∧ m_values.card = 4) ∧ number_of_distinct_stars = 2 :=
  sorry

end number_of_non_similar_regular_12_pointed_double_layered_stars_l3_3103


namespace point_in_multiple_k_gons_l3_3515

theorem point_in_multiple_k_gons (n k : ℕ) (h : 1 ≤ k)
    (h1 : ∀ (P_i P_j : convex_polygon) (H : P_i.is_k_gon k ∧ P_j.is_k_gon k ∧ P_i.intersects P_j),
        ∃ (O : point) (r : ℝ) (hr: r > 0), P_j = P_i.homothety_at O r) :
    ∃ (p : point), ∃ (m : ℕ), 1 + (n - 1) / (2 * k) ≤ m ∧ p.belongs_to_at_least_m_k_gons p m :=
sorry

end point_in_multiple_k_gons_l3_3515


namespace probability_sum_to_4_l3_3300

noncomputable def rounding_scenarios_probability : ℝ :=
  let intervals := [(0.5, 1.5), (1.5, 2.5), (2.5, 3.5)] in
  let total_length := intervals.foldr (λ ⟨a, b⟩ acc => acc + (b - a)) 0 in
  total_length / 3.5

theorem probability_sum_to_4 :
  rounding_scenarios_probability = 6 / 7 :=
by
  sorry

end probability_sum_to_4_l3_3300


namespace sum_distances_eq_6sqrt2_l3_3951

-- Define the curves C1 and C2 in Cartesian coordinates
def curve_C1 := { p : ℝ × ℝ | p.1 + p.2 = 3 }
def curve_C2 := { p : ℝ × ℝ | p.2^2 = 2 * p.1 }

-- Defining the point P in ℝ²
def point_P : ℝ × ℝ := (1, 2)

-- Find the sum of distances |PA| + |PB|
theorem sum_distances_eq_6sqrt2 : 
  ∃ A B : ℝ × ℝ, A ∈ curve_C1 ∧ A ∈ curve_C2 ∧ 
                B ∈ curve_C1 ∧ B ∈ curve_C2 ∧ 
                (dist point_P A) + (dist point_P B) = 6 * Real.sqrt 2 := 
sorry

end sum_distances_eq_6sqrt2_l3_3951


namespace find_x3_plus_y3_l3_3295

theorem find_x3_plus_y3 (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 + y^2 = 167) : x^3 + y^3 = 2005 :=
sorry

end find_x3_plus_y3_l3_3295


namespace value_of_m_l3_3893

-- Define what it means to be a pure imaginary number
def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0

-- Define the given complex number
def complex_expression (m : ℝ) : ℂ :=
  (1 + m * complex.I) * (1 - complex.I)

-- Define the condition for the problem
def condition (m : ℝ) : Prop :=
  is_pure_imaginary (complex_expression m)

-- State the theorem that proves the value of m
theorem value_of_m (m : ℝ) : condition m → m = -1 :=
by
  sorry

end value_of_m_l3_3893


namespace mode_of_set_l3_3787

-- Define the data set
def data_set (x : ℕ) : List ℕ := [4, 5, x, 7, 9]

-- Define the condition for the average of the data set
def average_condition (x : ℕ) : Prop := (4 + 5 + x + 7 + 9) / 5 = 6

-- Define the proof goal to show that the mode is 5
theorem mode_of_set (x : ℕ) (h : average_condition x) : mode (data_set x) = 5 :=
sorry

end mode_of_set_l3_3787


namespace find_z_l3_3581

open Complex

theorem find_z (z : ℂ) (h : 2 * z + conj z = 3 - 2 * I) : z = 1 - 2 * I :=
sorry

end find_z_l3_3581


namespace vasya_always_wins_l3_3224

def player := ℕ → Prop

noncomputable def move_like_queen (x y : ℕ × ℕ) : Prop :=
  x.1 = y.1 ∨ x.2 = y.2 ∨ x.1 - y.1 = x.2 - y.2 ∨ x.1 - y.1 = y.2 - x.2

noncomputable def move_like_king (x y : ℕ × ℕ) : Prop :=
  abs (x.1 - y.1) ≤ 1 ∧ abs (x.2 - y.2) ≤ 1

def valid_move (current_pos : ℕ × ℕ) (board : set (ℕ × ℕ)) (move : ℕ × ℕ → Prop) : Prop :=
  ∀ new_pos, move current_pos new_pos → new_pos ∉ board

noncomputable def initial_conditions : Prop :=
  ∃ board: set (ℕ × ℕ), 
  ‒ (0, 0) ∈ board ∧ ∀ x y : ℕ × ℕ, x.1 < 8 ∧ x.2 < 8 ∧ y.1 < 8 ∧ y.2 < 8

noncomputable def winning_strategy : Prop :=
  ∀ board: set (ℕ × ℕ),
  (valid_move (0,0) board move_like_queen → 
  (∀ pos, valid_move pos board (λ x y, move_like_king x y ∧ move_like_king y x) →
  ¬valid_move (0,0) board move_like_queen)) → sorry
  
theorem vasya_always_wins:
  initial_conditions → winning_strategy :=
by sorry

end vasya_always_wins_l3_3224


namespace right_angled_triangle_l3_3918

-- Define the lengths of the sides of the triangle
def a : ℕ := 9
def b : ℕ := 12
def c : ℕ := 15

-- State the theorem using the Pythagorean theorem
theorem right_angled_triangle : a^2 + b^2 = c^2 :=
by
  sorry

end right_angled_triangle_l3_3918


namespace fourth_number_in_12th_row_is_92_l3_3311

-- Define the number of elements per row and the row number
def elements_per_row := 8
def row_number := 12

-- Define the last number in a row function
def last_number_in_row (n : ℕ) := elements_per_row * n

-- Define the starting number in a row function
def starting_number_in_row (n : ℕ) := (elements_per_row * (n - 1)) + 1

-- Define the nth number in a specified row function
def nth_number_in_row (n : ℕ) (k : ℕ) := starting_number_in_row n + (k - 1)

-- Prove that the fourth number in the 12th row is 92
theorem fourth_number_in_12th_row_is_92 : nth_number_in_row 12 4 = 92 :=
by
  -- state the required equivalences
  sorry

end fourth_number_in_12th_row_is_92_l3_3311


namespace octahedron_tetrahedron_surface_area_ratio_l3_3784

theorem octahedron_tetrahedron_surface_area_ratio :
  ∀ (octahedron tetrahedron : ℝ → Prop),
    (∀ s, octahedron s → s = 1) →
    (∀ s, tetrahedron s → s = sqrt 2) →
    (∃ (s₀ s₁ : ℝ), octahedron s₀ ∧ tetrahedron s₁ ∧ ratio_surface_area s₀ s₁ = 1) :=
by
  sorry

end octahedron_tetrahedron_surface_area_ratio_l3_3784


namespace num_divisors_of_36_l3_3188

/-
  The theorem states that 36 has exactly 9 positive divisors, and lists all of them.
-/
theorem num_divisors_of_36 : 
  let divisors := [1, 2, 3, 4, 6, 9, 12, 18, 36] in
  (∀ d, d ∣ 36 → d ∈ divisors) ∧ divisors.length = 9 :=
by
  sorry

end num_divisors_of_36_l3_3188


namespace general_formula_arithmetic_sequence_exists_max_integer_t_l3_3513

noncomputable def arithmetic_sequence (n : ℕ) : ℕ :=
  2 * n - 1

noncomputable def b (n : ℕ) : ℚ :=
  (arithmetic_sequence n + 1)^2 / (arithmetic_sequence n * arithmetic_sequence (n + 1))

noncomputable def S (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i, b (i + 1))

theorem general_formula_arithmetic_sequence :
  ∀ n : ℕ, arithmetic_sequence n = 2 * n - 1 :=
by sorry

theorem exists_max_integer_t :
  ∃ (t : ℤ), t = 1 ∧ ∀ n : ℕ, S n > t :=
by sorry

end general_formula_arithmetic_sequence_exists_max_integer_t_l3_3513


namespace sum_sin_squared_l3_3860

theorem sum_sin_squared (n : ℕ) (hn : 0 < n) : 
  let x := (Real.pi / (2 * n : ℝ)) in
  (Finset.sum (Finset.range n) (λ k, Real.sin (x * (k + 1 : ℝ)) ^ 2))
  = (n + 1) / 2 := 
by
  sorry

end sum_sin_squared_l3_3860


namespace convert_to_base8_l3_3459

theorem convert_to_base8 (n : ℕ) (h : n = 1024) : 
  (∃ (d3 d2 d1 d0 : ℕ), n = d3 * 8^3 + d2 * 8^2 + d1 * 8^1 + d0 * 8^0 ∧ d3 = 2 ∧ d2 = 0 ∧ d1 = 0 ∧ d0 = 0) :=
by
  sorry

end convert_to_base8_l3_3459


namespace determine_function_f_l3_3827

noncomputable def f (c x : ℝ) : ℝ := c ^ (1 / Real.log x)

theorem determine_function_f (f : ℝ → ℝ) (c : ℝ) (Hc : c > 1) :
  (∀ x, 1 < x → 1 < f x) →
  (∀ (x y : ℝ) (u v : ℝ), 1 < x → 1 < y → 0 < u → 0 < v →
    f (x ^ 4 * y ^ v) ≤ (f x) ^ (1 / (4 * u)) * (f y) ^ (1 / (4 * v))) →
  (∀ x : ℝ, 1 < x → f x = c ^ (1 / Real.log x)) :=
by
  sorry

end determine_function_f_l3_3827


namespace continuous_f_identity_l3_3995

noncomputable def satisfies_integral_eqns (n : ℕ) (hn : Odd n) (hge : n ≥ 3) (f : ℝ → ℝ) : Prop :=
  ∀ k : ℕ, k ∈ Finset.range (n - 1).succ → 
    ∫ x in 0..1, (f (x ^ (1 / k.to_real))) ^ (n - k) = k / n

theorem continuous_f_identity {n : ℕ} (hn : Odd n) (hge : n ≥ 3) :
  ∀ f : ℝ → ℝ, Continuous f → satisfies_integral_eqns n hn hge f → ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → f t = t :=
sorry

end continuous_f_identity_l3_3995


namespace rods_in_one_mile_l3_3523

theorem rods_in_one_mile (chains_in_mile : ℕ) (rods_in_chain : ℕ) (mile_to_chain : 1 = 10 * chains_in_mile) (chain_to_rod : 1 = 22 * rods_in_chain) :
  1 * 220 = 10 * 22 :=
by sorry

end rods_in_one_mile_l3_3523


namespace probability_red_ball_l3_3802

def total_balls : ℕ := 3
def red_balls : ℕ := 1
def yellow_balls : ℕ := 2

theorem probability_red_ball : (red_balls : ℚ) / (total_balls : ℚ) = 1 / 3 :=
by
  sorry

end probability_red_ball_l3_3802


namespace annie_spent_28_dollars_l3_3441

def cost_per_classmate (ca cb cc : ℕ) (costA costB costC : ℝ) : ℝ :=
  3 * costA + 2 * costB + costC

def total_cost (n : ℕ) (cost_per_classmate: ℝ) : ℝ :=
  n * cost_per_classmate

theorem annie_spent_28_dollars :
  let ca := 35
  let costA := 0.1
  let costB := 0.15
  let costC := 0.2
  total_cost ca (cost_per_classmate ca costA costB costC) = 28 :=
by
  sorry

end annie_spent_28_dollars_l3_3441


namespace bart_interest_after_three_years_l3_3299

def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

theorem bart_interest_after_three_years :
  let P := 500
  let r := 0.02
  let n := 3
  let A := compound_interest P r n
  let interest := A - P
  interest ≈ 30 :=
by
  sorry

end bart_interest_after_three_years_l3_3299


namespace number_of_8_tuples_l3_3463

-- Define the constraints for a_k
def valid_a (a : ℕ) (k : ℕ) : Prop := 0 ≤ a ∧ a ≤ k

-- Define the condition for the 8-tuple
def valid_8_tuple (a1 a2 a3 a4 b1 b2 b3 b4 : ℕ) : Prop :=
  valid_a a1 1 ∧ valid_a a2 2 ∧ valid_a a3 3 ∧ valid_a a4 4 ∧ 
  (a1 + a2 + a3 + a4 + 2 * b1 + 3 * b2 + 4 * b3 + 5 * b4 = 19)

theorem number_of_8_tuples : 
  ∃ (n : ℕ), n = 1540 ∧ 
  ∃ (a1 a2 a3 a4 b1 b2 b3 b4 : ℕ), valid_8_tuple a1 a2 a3 a4 b1 b2 b3 b4 := 
sorry

end number_of_8_tuples_l3_3463


namespace total_beats_together_in_week_l3_3662

theorem total_beats_together_in_week :
  let samantha_beats_per_min := 250
  let samantha_hours_per_day := 3
  let michael_beats_per_min := 180
  let michael_hours_per_day := 2.5
  let days_per_week := 5

  let samantha_beats_per_day := samantha_beats_per_min * 60 * samantha_hours_per_day
  let samantha_beats_per_week := samantha_beats_per_day * days_per_week
  let michael_beats_per_day := michael_beats_per_min * 60 * michael_hours_per_day
  let michael_beats_per_week := michael_beats_per_day * days_per_week
  let total_beats_per_week := samantha_beats_per_week + michael_beats_per_week

  total_beats_per_week = 360000 := 
by
  -- The proof will go here
  sorry

end total_beats_together_in_week_l3_3662


namespace ratio_problem_l3_3921

theorem ratio_problem (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 := 
by
  sorry

end ratio_problem_l3_3921


namespace satisfies_condition_l3_3146

def f : ℝ → ℝ := λ x, if x ∈ ℚ then 1 else 0

theorem satisfies_condition (f : ℝ → ℝ) (h : ∀ x : ℝ, f (2 * x) = f (x + 1))
  (hf : f = λ x, if x ∈ ℚ then 1 else 0) : ∀ x : ℝ, f (2 * x) = f (x + 1) :=
sorry

end satisfies_condition_l3_3146


namespace sum_mod_11_l3_3720

theorem sum_mod_11 :
  let s := (List.range 15).map (λ x => x + 1) in
  (s.sum % 11) = 10 :=
by
  let s := (List.range 15).map (λ x => x + 1)
  have h : s.sum = 120 := sorry
  show s.sum % 11 = 10
  calc
    s.sum % 11 = 120 % 11  : by rw [h]
           ... = 10        : by norm_num

end sum_mod_11_l3_3720


namespace quadrilateral_DFGE_is_cyclic_l3_3976

open EuclideanGeometry

theorem quadrilateral_DFGE_is_cyclic 
  (Γ : Circle)
  (B C : Point)
  (hBC_chord : Chord Γ B C)
  (A : Point)
  (hA_midpoint : midpoint_arc Γ B C A)
  (D E : Point)
  (hAD_chord : Chord Γ A D)
  (hAE_chord : Chord Γ A E)
  (F G : Point)
  (hF_intersection : intersection_points AD BC F)
  (hG_intersection : intersection_points AE BC G) :
  cyclic_quadrilateral (LineSegment D F) (LineSegment F G) (LineSegment G E) (LineSegment E D) :=
sorry

end quadrilateral_DFGE_is_cyclic_l3_3976


namespace hyperbola_standard_equation_l3_3676

theorem hyperbola_standard_equation
  (h1 : ∀ x y : ℝ, y = (1/3) * x ∨ y = -(1/3) * x)
  (h2 : (0, 2 * real.sqrt 5) ∈ set_of {p : ℝ × ℝ | ∀ h k: ℝ, p = (0, 2 * real.sqrt 5)}) :
  ∀ (x y : ℝ), (y^2) / 2 - (x^2) / 18 = 1 :=
sorry

end hyperbola_standard_equation_l3_3676


namespace intersection_correct_l3_3155

def setA := {x : ℝ | (x - 2) * (2 * x + 1) ≤ 0}
def setB := {x : ℝ | x < 1}
def expectedIntersection := {x : ℝ | -1 / 2 ≤ x ∧ x < 1}

theorem intersection_correct : (setA ∩ setB) = expectedIntersection := by
  sorry

end intersection_correct_l3_3155


namespace conditional_probability_calc_l3_3365

-- Definitions of the events M and N
def eventM (red_die : ℕ) : Prop :=
  red_die = 3 ∨ red_die = 6

def eventN (red_die blue_die : ℕ) : Prop :=
  red_die + blue_die > 8

-- Probability of M and MN
def P_M : ℚ := 12 / 36
def P_MN : ℚ := 5 / 36

-- Conditional Probability
def P_N_given_M := P_MN / P_M

theorem conditional_probability_calc :
  P_N_given_M = 5 / 12 :=
by
  unfold P_N_given_M P_M P_MN
  sorry

end conditional_probability_calc_l3_3365


namespace least_multiple_of_25_gt_475_l3_3719

theorem least_multiple_of_25_gt_475 : ∃ n : ℕ, n > 475 ∧ n % 25 = 0 ∧ ∀ m : ℕ, (m > 475 ∧ m % 25 = 0) → n ≤ m := 
  sorry

end least_multiple_of_25_gt_475_l3_3719


namespace angle_A_O2_B_gt_90_deg_l3_3684

-- Given conditions
def circle (x y : ℝ) (r : ℝ) : Prop := x^2 + y^2 = r^2
def circle_transformed (x y : ℝ) (h k r : ℝ) : Prop := (x + h)^2 + (y + k)^2 = r^2

noncomputable def O1 (x y : ℝ) : Prop := circle x y 2
noncomputable def O2 (x y : ℝ) : Prop := circle_transformed x y 1 (-2) (sqrt 5)

-- The proposition to prove
theorem angle_A_O2_B_gt_90_deg (A B : ℝ × ℝ) :
  (O1 A.1 A.2 ∧ O2 A.1 A.2) ∧
  (O1 B.1 B.2 ∧ O2 B.1 B.2) →
  ∃ α, 90 < α ∧ α = angle O2 A B :=
sorry

end angle_A_O2_B_gt_90_deg_l3_3684


namespace king_position_probability_l3_3646

open scoped BigOperators

def prob_of_king_on_even_coord_after_2008_turns : ℚ :=
  1 / 4 + 3 / (4 * 5 ^ 2008)

theorem king_position_probability :
  let p_even_after_2008_turns := prob_of_king_on_even_coord_after_2008_turns in
  p_even_after_2008_turns =
    1 / 4 + 3 / (4 * 5 ^ 2008) :=
sorry

end king_position_probability_l3_3646


namespace possible_to_achieve_2017_with_79_ones_l3_3059

theorem possible_to_achieve_2017_with_79_ones :
  ∃ (s : list string), s.length = 78 ∧ 
  (∀ (i : ℕ), i < 78 → (s[i] = "+" ∨ s[i] = "-")) ∧ 
  ((79 :: s).foldr (λ (x y : string), if x = "+" then y + 1 else y - 1) 0 = 2017) :=
sorry

end possible_to_achieve_2017_with_79_ones_l3_3059


namespace min_dot_product_l3_3553

variables {a b : EuclideanSpace 3 ℝ} (x y : ℝ)

-- Conditions
def unit_vec (v : EuclideanSpace 3 ℝ) := ∥v∥ = 1

def acute_angle (v w : EuclideanSpace 3 ℝ) := v.dot w > 0

def satisfies_condition (x y : ℝ) (a b : EuclideanSpace 3 ℝ) := 
  (∥x • a + y • b∥ = 1) ∧ (x * y ≥ 0)

def satisfies_inequality (x y : ℝ) :=
  |x + 2 * y| ≤ 8 / Real.sqrt 15

-- Proof problem
theorem min_dot_product :
  unit_vec a ∧ unit_vec b ∧ acute_angle a b →
  (∀ (x y : ℝ), satisfies_condition x y a b → satisfies_inequality x y) →
  a.dot b ≥ 1 / 4 :=
by
  sorry

end min_dot_product_l3_3553


namespace num_valid_8_tuples_is_1540_l3_3465

def valid_8_tuple (a1 a2 a3 a4 b1 b2 b3 b4 : ℕ) := 
  (0 ≤ a1 ∧ a1 ≤ 1) ∧ 
  (0 ≤ a2 ∧ a2 ≤ 2) ∧ 
  (0 ≤ a3 ∧ a3 ≤ 3) ∧
  (0 ≤ a4 ∧ a4 ≤ 4) ∧ 
  a1 + a2 + a3 + a4 + 2 * b1 + 3 * b2 + 4 * b3 + 5 * b4 = 19

theorem num_valid_8_tuples_is_1540 :
  {t : ℕ × ℕ × ℕ × ℕ × ℕ × ℕ × ℕ × ℕ // valid_8_tuple t.1 t.2.1 t.2.2.1 t.2.2.2.1 t.2.2.2.2.1 t.2.2.2.2.2.1 t.2.2.2.2.2.2.1 t.2.2.2.2.2.2.2 } = 1540 :=
by
  -- Proof goes here
  sorry

end num_valid_8_tuples_is_1540_l3_3465


namespace range_of_f_l3_3321

noncomputable def f (x : ℝ) : ℝ := (1/2) ^ |x + 1|

theorem range_of_f : Set.Ioo 0 1 ∪ {1} = {y : ℝ | ∃ x : ℝ, f x = y} :=
by
  sorry

end range_of_f_l3_3321


namespace similar_triangle_leg_l3_3424

theorem similar_triangle_leg (x : Real) : 
  (12 / x = 9 / 7) → x = 84 / 9 := by
  intro h
  sorry

end similar_triangle_leg_l3_3424


namespace solve_first_equation_solve_second_equation_l3_3856

theorem solve_first_equation (x : ℝ) : 3 * (x - 2)^2 - 27 = 0 ↔ x = 5 ∨ x = -1 :=
by {
  sorry
}

theorem solve_second_equation (x : ℝ) : 2 * (x + 1)^3 + 54 = 0 ↔ x = -4 :=
by {
  sorry
}

end solve_first_equation_solve_second_equation_l3_3856


namespace arithmetic_expression_evaluation_l3_3079

theorem arithmetic_expression_evaluation : 
  (5 * 7 - (3 * 2 + 5 * 4) / 2) = 22 := 
by
  sorry

end arithmetic_expression_evaluation_l3_3079


namespace general_formula_an_general_formula_bn_Tn_lt_half_l3_3163

-- Conditions
variable (a : ℕ → ℝ)
variable (b : ℕ → ℝ)
variable (T : ℕ → ℝ)
variable {q : ℝ} (h_pos : ∀ n, 0 < a n) 
variable {h1 : real.log_base 2 (4 * a 1 + 2 * a 1 * q) = 2 * real.log_base 2 (a 1 * q^2)}
variable {h2 : real.log_base 2 (a 5) = real.log_base 2 32}
variable {log_def : ∀ n, b n = real.log_base 2 (a (2 * n - 1))}

-- Theorem to prove the general formula for {a_n}
theorem general_formula_an (h_pos : ∀ n, 0 < a n) 
  (h1 : 4 * a 1 + 2 * (a 1 * q) = 2 * (a 1 * q^2)) 
  (h2 : a 5 = 32)
  (q_def : q = 2)
  (a1_def : a 1 = 2):
  ∀ n, a n = 2 ^ n :=
by
  sorry

-- Theorem to prove the general formula for {b_n}
theorem general_formula_bn (h_pos : ∀ n, 0 < a n)
  (log_def : ∀ n, b n = real.log_base 2 (a (2 * n - 1)))
  (a_def : ∀ n, a n = 2 ^ n):
  ∀ n, b n = 2 * n - 1 :=
by
  sorry

-- Theorem to prove T_n < 1 / 2
theorem Tn_lt_half 
  (log_def : ∀ n, b n = real.log_base 2 (a (2 * n - 1)))
  (a_def : ∀ n, a n = 2 ^ n)
  (bn_def : ∀ n, b n = 2 * n - 1)
  (T_def : ∀ n, T n = (1 / 2) * (1 - (1 / (2*(n + 1) + 1))))
  (n : ℕ) :
  T n < 1 / 2 :=
by
  sorry

end general_formula_an_general_formula_bn_Tn_lt_half_l3_3163


namespace square_of_1024_l3_3822

theorem square_of_1024 : 1024^2 = 1048576 :=
by
  sorry

end square_of_1024_l3_3822


namespace jade_transactions_l3_3644

theorem jade_transactions 
    (mabel_transactions : ℕ)
    (anthony_transactions : ℕ)
    (cal_transactions : ℕ)
    (jade_transactions : ℕ)
    (h_mabel : mabel_transactions = 90)
    (h_anthony : anthony_transactions = mabel_transactions + mabel_transactions / 10)
    (h_cal : cal_transactions = 2 * anthony_transactions / 3)
    (h_jade : jade_transactions = cal_transactions + 14) : 
    jade_transactions = 80 :=
sorry

end jade_transactions_l3_3644


namespace square_coverable_find_ks_l3_3350

open Nat

noncomputable def sum_of_areas (n : ℕ) : ℕ :=
  (∑ i in finset.range (n + 1), 2^i * (i + 1))

theorem square_coverable (k : ℕ) (hk : k > 1) :
  ∃ n, k^2 = 2^(n+1) * n + 1 :=
by
  sorry

theorem find_ks : {k | k > 1 ∧ ∃ n, k^2 = 2^(n+1) * n + 1} = {7} :=
by
  sorry

end square_coverable_find_ks_l3_3350


namespace find_number_l3_3713

theorem find_number (x : ℝ) : 
  10 * ((2 * (x * x + 2) + 3) / 5) = 50 → x = 3 := 
by
  sorry

end find_number_l3_3713


namespace members_didnt_show_up_l3_3796

theorem members_didnt_show_up (total_members : ℕ) (points_per_member : ℕ) (total_points : ℕ) :
  total_members = 14 →
  points_per_member = 5 →
  total_points = 35 →
  total_members - (total_points / points_per_member) = 7 :=
by
  intros
  sorry

end members_didnt_show_up_l3_3796


namespace number_in_parentheses_l3_3193

theorem number_in_parentheses (x : ℤ) (h : x - (-2) = 3) : x = 1 :=
by {
  sorry
}

end number_in_parentheses_l3_3193


namespace probability_weight_between_24_8_and_25_4_l3_3047

-- Define the normal distribution parameters
def mu := 25.0
def sigma_squared := 0.04
def sigma := Math.sqrt sigma_squared

-- Define the transformation to Z-score
def z (x : Float) := (x - mu) / sigma

-- Define known probability values for the standard normal distribution
def P_abs_z_less_sigma := 0.6826
def P_abs_z_less_two_sigma := 0.9544
def P_abs_z_less_three_sigma := 0.9974

-- Given conditions
def P_neg_one_le_z_le_one := P_abs_z_less_sigma
def P_one_le_z_le_two := (P_abs_z_less_two_sigma - P_abs_z_less_sigma) / 2

def P_X_in_interval := P_neg_one_le_z_le_one + P_one_le_z_le_two

theorem probability_weight_between_24_8_and_25_4 :
  P_X_in_interval = 0.8185 := by
  -- This will be proven when the needed properties of normal distribution are formalized
  sorry

end probability_weight_between_24_8_and_25_4_l3_3047


namespace range_proof_l3_3840

noncomputable def ellipse (x y a b : ℝ) : Prop := (a > b ∧ b > 0) ∧ (x^2 / a^2 + y^2 / b^2 = 1)
def ecc (c a : ℝ) : Prop := c / a = (2 / 3)
def inner_product (a c : ℝ) : Prop := (a^2 - c^2 = 5)

theorem range_proof (a b c m n : ℝ) (FM FN : ℝ) :
  ellipse 1 1 a b → ecc c a → inner_product a c →
  (m ∈ set.Icc 1 5) →
  ∀ (f : ℝ → ℝ), f = (λ m, 1 / m + 4 / (6 - m)) →
  ( ∀ M N : ℝ, (M = |FM| ∧ N = |FN| ∧ (M + N = 6)) →
    ∃ x y : ℝ, f x ≤ 21 / 5 ∧ f y ≥ 3 / 2 ) :=
by
  intros
  sorry

end range_proof_l3_3840


namespace generating_function_A_generating_function_B_l3_3080

-- Problem statements

-- Part (a)
theorem generating_function_A (m : ℕ) :
  (∑ n : ℕ, (nat.choose (m + n) m : ℚ) * x ^ n) = 1 / (1 - x) ^ (m + 1) := sorry

-- Part (b)
theorem generating_function_B (m : ℕ) :
  (∑ n : ℕ, (nat.choose n m : ℚ) * x ^ n) = (x ^ m) / (1 - x) ^ (m + 1) := sorry

end generating_function_A_generating_function_B_l3_3080


namespace total_cost_is_90_l3_3609

variable (jackets : ℕ) (shirts : ℕ) (pants : ℕ)
variable (price_jacket : ℕ) (price_shorts : ℕ) (price_pants : ℕ)

theorem total_cost_is_90 
  (h1 : jackets = 3)
  (h2 : price_jacket = 10)
  (h3 : shirts = 2)
  (h4 : price_shorts = 6)
  (h5 : pants = 4)
  (h6 : price_pants = 12) : 
  (jackets * price_jacket + shirts * price_shorts + pants * price_pants) = 90 := by 
  sorry

end total_cost_is_90_l3_3609


namespace probability_green_or_blue_l3_3667

-- Define the properties of the 10-sided die
def total_faces : ℕ := 10
def red_faces : ℕ := 4
def yellow_faces : ℕ := 3
def green_faces : ℕ := 2
def blue_faces : ℕ := 1

-- Define the number of favorable outcomes
def favorable_outcomes : ℕ := green_faces + blue_faces

-- Define the probability function
def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

-- The theorem to prove
theorem probability_green_or_blue :
  probability favorable_outcomes total_faces = 3 / 10 :=
by
  sorry

end probability_green_or_blue_l3_3667


namespace unique_integer_sequence_l3_3254

theorem unique_integer_sequence (a : ℕ → ℤ)
  (H_inf_pos : ∀ N : ℕ, ∃ n : ℕ, n > N ∧ 0 < a n)
  (H_inf_neg : ∀ N : ℕ, ∃ n : ℕ, n > N ∧ a n < 0)
  (H_distinct_remainders : ∀ (n : ℕ), (list.range n).pairwise (λ i j, (a i % n ≠ a j % n))) :
  ∀ (m : ℤ), ∃! k : ℕ, a k = m :=
sorry

end unique_integer_sequence_l3_3254


namespace smallest_y_l3_3314

theorem smallest_y 
  (y : ℕ)
  (hy_factors : (nat.factors y).length = 8) 
  (h18 : 18 ∣ y) 
  (h20 : 20 ∣ y) 
  : y = 180 :=
sorry

end smallest_y_l3_3314


namespace vasya_number_l3_3711

theorem vasya_number (a b c : ℕ) (h1 : 100 ≤ 100*a + 10*b + c) (h2 : 100*a + 10*b + c < 1000) 
  (h3 : a + c = 1) (h4 : a * b = 4) (h5 : a ≠ 0) : 100*a + 10*b + c = 140 :=
by
  sorry

end vasya_number_l3_3711


namespace correct_transformation_l3_3710

theorem correct_transformation (m : ℤ) (h : 2 * m - 1 = 3) : 2 * m = 4 :=
by
  sorry

end correct_transformation_l3_3710


namespace compute_ratio_l3_3888

variable {p q r u v w : ℝ}

theorem compute_ratio
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hu : u > 0) (hv : v > 0) (hw : w > 0) 
  (h1 : p^2 + q^2 + r^2 = 49) 
  (h2 : u^2 + v^2 + w^2 = 64) 
  (h3 : p * u + q * v + r * w = 56) : 
  (p + q + r) / (u + v + w) = 7 / 8 := 
sorry

end compute_ratio_l3_3888


namespace problem_value_of_f_l3_3635

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 1 - x^2 else x^2 + x - 2

theorem problem_value_of_f : 
  f (1 / f 2) = 15 / 16 :=
by
  sorry

end problem_value_of_f_l3_3635


namespace parallel_condition_l3_3040

theorem parallel_condition (a : ℝ) : (a = -2) ↔ (∀ x y : ℝ, ax + 2 * y = 0 → (-a / 2) = 1) :=
by
  sorry

end parallel_condition_l3_3040


namespace solve_for_x_l3_3931

variable {a b c x : ℝ}
variable (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
variable (h_eq : (x - a - b) / c + (x - b - c) / a + (x - c - a) / b = 3)

theorem solve_for_x (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : (x - a - b) / c + (x - b - c) / a + (x - c - a) / b = 3) : 
  x = a + b + c :=
sorry

end solve_for_x_l3_3931


namespace jonathan_needs_more_money_l3_3243

def cost_dictionary : ℕ := 11
def cost_dinosaur_book : ℕ := 19
def cost_childrens_cookbook : ℕ := 7
def saved_money : ℕ := 8

def total_cost : ℕ := cost_dictionary + cost_dinosaur_book + cost_childrens_cookbook
def amount_needed : ℕ := total_cost - saved_money

theorem jonathan_needs_more_money : amount_needed = 29 := by
  have h1 : total_cost = 37 := by
    show 11 + 19 + 7 = 37
    sorry
  show 37 - 8 = 29
  sorry

end jonathan_needs_more_money_l3_3243


namespace distribution_X_Y_l3_3010

noncomputable theory

variables {Ω : Type*} [MeasurableSpace Ω]
variables (P : MeasureTheory.ProbabilityMeasure Ω)
variables (hit_first: MeasureTheory.MeasurableSet (set.univ : set Ω))
variables (hit_second: MeasureTheory.MeasurableSet (set.univ : set Ω))
variables (miss_first: MeasureTheory.MeasurableSet (set.univ : set Ω))
variables (miss_second: MeasureTheory.MeasurableSet (set.univ : set Ω))

-- Given conditions
def prob_hit_first : ℝ := 0.3
def prob_hit_second : ℝ := 0.7

-- Definitions of the distribution laws for X and Y
def dist_X (k : ℕ) : ℝ := if k = 0 then 0 else 0.79 * 0.21^(k-1)
def dist_Y (k : ℕ) : ℝ := 0.3 * if k = 0 then 1 else 0.553 * 0.21^(k-1)

-- Theorem stating the required distributions
theorem distribution_X_Y :
  (∀ k : ℕ, MeasureTheory.ProbabilityMassFunction.probability P (dist_X k)) ∧ 
  (∀ k : ℕ, MeasureTheory.ProbabilityMassFunction.probability P (dist_Y k)) :=
by { sorry }

end distribution_X_Y_l3_3010


namespace no_prime_divisible_by_57_l3_3564

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. --/
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Given that 57 is equal to 3 times 19.--/
theorem no_prime_divisible_by_57 : ∀ p : ℕ, is_prime p → ¬ (57 ∣ p) :=
by
  sorry

end no_prime_divisible_by_57_l3_3564


namespace probability_not_square_l3_3944

theorem probability_not_square 
  (total_figures : ℕ)
  (triangles : ℕ)
  (squares : ℕ)
  (circles : ℕ)
  (htotal : total_figures = 10)
  (htri : triangles = 5)
  (hsq : squares = 3)
  (hcirc : circles = 2) :
  ∃ (p : ℚ), p = 7 / 10 :=
by
  use 7 / 10
  sorry

end probability_not_square_l3_3944


namespace new_car_distance_l3_3779

theorem new_car_distance (older_car_distance : ℕ) (h : older_car_distance = 150) : 
  let new_car_distance := older_car_distance + (older_car_distance * 30 / 100)
  in new_car_distance = 195 :=
by
  have h1 : older_car_distance = 150 := h
  let new_car_distance := older_car_distance + (older_car_distance * 30 / 100)
  show new_car_distance = 195 by sorry

end new_car_distance_l3_3779


namespace log_12_7_eq_2rs_over_2r_plus_1_l3_3923

-- Define the conditions given in the problem
variable (r s : ℝ)
axiom log_9_4_eq_r : log 9 4 = r
axiom log_4_7_eq_s : log 4 7 = s

-- State the theorem we need to prove
theorem log_12_7_eq_2rs_over_2r_plus_1 : log 12 7 = (2 * r * s) / (2 * r + 1) :=
by
  sorry

end log_12_7_eq_2rs_over_2r_plus_1_l3_3923


namespace total_hours_worked_l3_3436

-- Definitions of the conditions
def ordinary_pay_rate := 0.60  -- 60 cents per hour
def overtime_pay_rate := 0.90  -- 90 cents per hour
def total_weekly_earnings := 32.40  -- total earnings for the week in dollars
def overtime_hours := 8  -- hours of overtime worked

-- Theorem stating the total number of hours worked
theorem total_hours_worked 
  (ordinary_pay_rate_def : ordinary_pay_rate = 0.60)
  (overtime_pay_rate_def : overtime_pay_rate = 0.90)
  (total_weekly_earnings_def : total_weekly_earnings = 32.40)
  (overtime_hours_def : overtime_hours = 8) :
  let ordinary_hours := (total_weekly_earnings - (overtime_pay_rate * overtime_hours)) / ordinary_pay_rate 
  in ordinary_hours + overtime_hours = 50 :=
by sorry

end total_hours_worked_l3_3436


namespace right_triangle_perimeter_l3_3578

theorem right_triangle_perimeter (hypotenuse : ℝ) (angle : ℝ) 
  (h_hyp : hypotenuse = 1) (h_angle : angle = 60) : 
  let a := hypotenuse / 2
  let b := hypotenuse * (sqrt 3) / 2
  let perimeter := hypotenuse + a + b
  perimeter = (3 + sqrt 3) / 2 := 
by
  sorry

end right_triangle_perimeter_l3_3578


namespace number_of_ways_to_write_528_as_sum_of_consecutive_integers_l3_3216

theorem number_of_ways_to_write_528_as_sum_of_consecutive_integers : 
  ∃ (n : ℕ), (2 ≤ n ∧ ∃ k : ℕ, n * (2 * k + n - 1) = 1056) ∧ n = 15 :=
by
  sorry

end number_of_ways_to_write_528_as_sum_of_consecutive_integers_l3_3216


namespace sqrt_nested_eq_x_pow_eleven_eighths_l3_3837

theorem sqrt_nested_eq_x_pow_eleven_eighths (x : ℝ) (hx : 0 ≤ x) : 
  Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x))) = x ^ (11 / 8) :=
  sorry

end sqrt_nested_eq_x_pow_eleven_eighths_l3_3837


namespace cube_surface_area_l3_3181

structure Point3D := (x : ℝ) (y : ℝ) (z : ℝ)

def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2 + (p2.z - p1.z)^2)

noncomputable def side_length_of_cube {A B C : Point3D} (h_ab : distance A B = Real.sqrt 98)
  (h_ac : distance A C = Real.sqrt 98)
  (h_bc : distance B C = Real.sqrt 98) : ℝ :=
  Real.sqrt 98 / Real.sqrt 2

theorem cube_surface_area {A B C : Point3D} (h_ab : distance A B = Real.sqrt 98)
  (h_ac : distance A C = Real.sqrt 98)
  (h_bc : distance B C = Real.sqrt 98) :
  6 * (side_length_of_cube h_ab h_ac h_bc)^2 = 294 :=
sorry

def A : Point3D := ⟨5, 9, 5⟩
def B : Point3D := ⟨6, 5, -4⟩
def C : Point3D := ⟨9, 0, 4⟩

example : 6 * (side_length_of_cube (distance A B) (distance A C) (distance B C))^2 = 294 :=
cube_surface_area (distance A B).sound (distance A C).sound (distance B C).sound


end cube_surface_area_l3_3181


namespace complex_polynomial_divides_p1992_l3_3770

noncomputable def polynomial_divides_p1992 (q : ℂ[X]) (z : ℕ → ℂ) : Prop :=
  ∃ p : ℕ → ℂ[X], 
    (∀ n, p 1 = X - C (z 1)) ∧ 
    (∀ n, p (n + 1) = (p n)^2 - C (z (n + 1))) ∧ 
    q ∣ p 1992

theorem complex_polynomial_divides_p1992 (q : ℂ[X]) (h_deg : q.degree = 1992)
    (h_roots : q.roots.nodup) :
    ∃ z : ℕ → ℂ, polynomial_divides_p1992 q z :=
sorry

end complex_polynomial_divides_p1992_l3_3770


namespace player1_has_winning_strategy_l3_3509

def is_in_M (x y : ℤ) : Prop :=
  x^2 + y^2 ≤ 10^10

structure state :=
  (marked : set (ℤ × ℤ))
  (last : ℤ × ℤ)

def is_symmetric (p q : ℤ × ℤ) : Prop :=
  ∃ θ : ℤ, θ ≠ 0 ∧ θ ≠ 180 ∧ θ ≠ 360 ∧ (x: ℤ, y: ℤ) = (cos θ * p.1 - sin θ * p.2, sin θ * p.1 + cos θ * p.2)

def valid_move (s : state) (p : ℤ × ℤ) : Prop :=
  p ∈ M ∧ p ∉ s.marked ∧ ¬ is_symmetric s.last p

def game_over (s : state) : Prop :=
  ∀ p, ¬valid_move s p

noncomputable def player1_winning_strategy : Prop :=
  ∃ f : state → ℤ × ℤ, ∀ s, valid_move s (f s) ∧ game_over (state.mk (s.marked ∪ {f s}) (f s))

theorem player1_has_winning_strategy : player1_winning_strategy :=
sorry

end player1_has_winning_strategy_l3_3509


namespace expand_expression_l3_3844

theorem expand_expression (x : ℝ) : (x + 3) * (2 * x ^ 2 - x + 4) = 2 * x ^ 3 + 5 * x ^ 2 + x + 12 :=
by
  sorry

end expand_expression_l3_3844


namespace reflected_ray_eq_l3_3783

noncomputable def point := ℝ × ℝ
def A := (-3 : ℝ, 4 : ℝ)
def B := (-2 : ℝ, 6 : ℝ)
def A₁ := (-3 : ℝ, -4 : ℝ)
def A₂ := (3 : ℝ, -4 : ℝ)

theorem reflected_ray_eq :
  ∃ c : ℝ, A₂.1 * 2 + A₂.2 + c = 0 ∧
  ∃ c : ℝ, B.1 * 2 + B.2 + c = 0 :=
sorry

end reflected_ray_eq_l3_3783


namespace hyperbola_condition_l3_3467

theorem hyperbola_condition (m : ℝ) : 
  (∃ x y : ℝ, (x^2 / (m + 2)) + (y^2 / (m + 1)) = 1) ↔ (-2 < m ∧ m < -1) :=
by
  sorry

end hyperbola_condition_l3_3467


namespace area_of_square_l3_3227

theorem area_of_square (ABCD MN : ℝ) (h1 : 4 * (ABCD / 4) = ABCD) (h2 : MN = 3) : ABCD = 64 :=
by
  sorry

end area_of_square_l3_3227


namespace area_of_square_with_side_15_l3_3017

theorem area_of_square_with_side_15 : 
  ∀ (side_length : ℝ), side_length = 15 → (side_length * side_length) = 225 :=
by {
  assume side_length,
  assume h : side_length = 15,
  sorry
}

end area_of_square_with_side_15_l3_3017


namespace parallel_line_slope_l3_3023

-- Define the points p1 and p2
structure Point where
  x : ℤ
  y : ℤ

def p1 : Point := { x := 2, y := -3 }
def p2 : Point := { x := -4, y := 5 }

-- Define the slope calculation
def slope (p1 p2 : Point) : ℚ :=
  (p2.y - p1.y : ℚ) / (p2.x - p1.x : ℚ)

-- Statement: Prove that the slope of a line parallel to the line through points p1 and p2 is -4/3
theorem parallel_line_slope : slope p1 p2 = -4/3 :=
sorry

end parallel_line_slope_l3_3023


namespace eat_together_time_l3_3642

noncomputable def time_to_eat_together (weight : ℕ) (rate_fat : ℚ) (rate_thin : ℚ) : ℚ := 
  weight / (rate_fat + rate_thin)

theorem eat_together_time :
  let rate_fat := 1 / 20
  let rate_thin := 1 / 40
  time_to_eat_together 4 rate_fat rate_thin ≈ 53 := by
  sorry

end eat_together_time_l3_3642


namespace exists_six_digit_no_identical_six_endings_l3_3472

theorem exists_six_digit_no_identical_six_endings :
  ∃ (A : ℕ), (100000 ≤ A ∧ A < 1000000) ∧ ∀ (k : ℕ), (1 ≤ k ∧ k ≤ 500000) → 
  (∀ d, d ≠ 0 → d < 10 → (k * A) % 1000000 ≠ d * 111111) :=
by
  sorry

end exists_six_digit_no_identical_six_endings_l3_3472


namespace min_black_cells_needed_l3_3947

theorem min_black_cells_needed (N : ℕ) (black_cells_initial : set (fin 8 × fin 12)) :
  (∀ cells : set (fin 8 × fin 12),
  (∀ a b c : fin 8 × fin 12, a ≠ b ∧ b ≠ c ∧ c ≠ a) →
  (∃ op : (fin 8 × fin 12) → Prop, op a ∧ op b ∧ op c)) →
  (¬ ∃ ops : fin 25 → set (fin 8 × fin 12), 
   ∀ i : fin 25, ∃ a b c : fin 8 × fin 12, 
   (black_cells_initial ∩ ops i).subset {a, b, c} ∧ ops i = {a, b, c}) →
  N = 27 :=
sorry

end min_black_cells_needed_l3_3947


namespace max_1x2_rectangles_in_3x4_grid_l3_3677

theorem max_1x2_rectangles_in_3x4_grid: 
  ∀ (grid : matrix (fin 3) (fin 4) (ℕ)), (∃ rectangles : list (list (fin 3 × fin 4)), 
    (∀ r ∈ rectangles, r.length = 2 ∧ r.pairwise (λ (a b : (fin 3 × fin 4)), 
    a ≠ b)) ∧ rectangles.length = 5) :=
by {
  sorry
}

end max_1x2_rectangles_in_3x4_grid_l3_3677


namespace sum_of_possible_values_d_l3_3767

noncomputable def first_base := 4
noncomputable def second_base := 16

-- Define the range of numbers that have exactly 5 digits in base 4.
def min_value_base4 := first_base ^ 4
def max_value_base4 := first_base ^ 5 - 1

-- Define the number of digits function in base 16.
def num_digits_in_base (n : ℕ) (b : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log b n + 1

-- Define the theorem that states the sum of all possible values of d is 3.
theorem sum_of_possible_values_d : 
  ∑ d in {d | ∃ n, min_value_base4 ≤ n ∧ n ≤ max_value_base4 ∧ num_digits_in_base n second_base = d}, d = 3 :=
  sorry

end sum_of_possible_values_d_l3_3767


namespace find_magnitude_of_a_plus_b_l3_3185

def vector := ℝ × ℝ

def dot_product (v₁ v₂ : vector) : ℝ :=
  v₁.1 * v₂.1 + v₁.2 * v₂.2

def magnitude (v : vector) : ℝ :=
  sqrt (v.1 ^ 2 + v.2 ^ 2)

variable (x y : ℝ)
def a : vector := (x, 1)
def b : vector := (1, y)
def c : vector := (2, -4)

theorem find_magnitude_of_a_plus_b
  (h₁ : dot_product a c = 0)
  (h₂ : ∃ k : ℝ, b = k • c) :
  magnitude (a + b) = sqrt 10 :=
  sorry

end find_magnitude_of_a_plus_b_l3_3185


namespace tan_15_deg_product_l3_3565

theorem tan_15_deg_product : (1 + Real.tan 15) * (1 + Real.tan 15) = 2.1433 := by
  sorry

end tan_15_deg_product_l3_3565


namespace length_of_KN_l3_3405

variable {R : ℝ} -- radius of the circle
variable (l : Set ℤ) -- tangent line
variable (A B C D E K N : Set ℤ) -- defining the points mentioned in the problem
variable (circle : Set ℤ)  -- the circle itself

/- Definitions and Conditions -/
def is_tangent_at (circle : Set ℤ) (line : Set ℤ) (pt : Set ℤ) : Prop :=
  -- type out the mathematical definition of tangency
  sorry

def is_diameter (circle : Set ℤ) (segment : Set ℤ) : Prop :=
  -- type out the definition of diameter
  sorry

def is_perpendicular (a b c : Set ℤ) : Prop :=
  -- type out the definition of perpendicularity
  sorry

def is_on_extension (a b c : Set ℤ) (dist_ab dist_ac : ℝ) : Prop :=
  -- type out the definition of a point lying on the extension with specific length
  sorry

axiom tangents_through_point
  (circle : Set ℤ) (pt : Set ℤ) : Prop := sorry -- axiom for constructing tangents passing through a point

/- Proof Statement -/
theorem length_of_KN (h1 : is_tangent_at circle l A)
  (h2 : is_diameter circle (A ∪ B))
  (h3 : ∃ B C, ¬Collinear A B C)
  (h4 : is_perpendicular C D A)
  (h5 : is_on_extension C D E (|BC|) (|ED|))
  (h6 : tangents_through_point circle E) :
  |KN| = 2 * R :=
sorry

end length_of_KN_l3_3405


namespace polynomial_to_linear_form_is_divisible_by_49_l3_3067

   variable (a b x : ℕ)

   def transform_polynomial (p : ℕ → ℕ) : (ℕ → ℕ) :=
     sorry  -- Placeholder for arbitrary differentiation and multiplication operations.

   theorem polynomial_to_linear_form_is_divisible_by_49 :
     let p := (λ x, x^8 + x^7) in
     let q := transform_polynomial p in
     (q 1 - q 0) % 49 = 0 :=
   sorry
   
end polynomial_to_linear_form_is_divisible_by_49_l3_3067


namespace fixed_equidistant_point_exists_l3_3346

-- Definitions based on conditions
structure MovingPoints :=
  (pos1 : ℝ × ℝ → ℝ × ℝ)
  (pos2 : ℝ × ℝ → ℝ × ℝ)
  (equal_speeds : ∀ t₁ t₂, (pos1 (t₁ + 1) - pos1 t₁) = (pos2 (t₂ + 1) - pos2 t₂))
  (intersect_at : ℝ × ℝ)

-- Main proposition based on the question and correct answer
theorem fixed_equidistant_point_exists (points : MovingPoints) : 
  ∃ M : ℝ × ℝ, ∀ t : ℝ, (dist (points.pos1 t) M) = (dist (points.pos2 t) M) :=
sorry

end fixed_equidistant_point_exists_l3_3346


namespace sin_cos_product_is_obtuse_triangle_tan_value_l3_3892

variable {α : Type*} [LinearOrderedField α]

-- Condition 1: Given in the problem
def sin_cos_sum (A : α) : Prop := sin A + cos A = 1 / 5

-- Question 1: Prove \(\sin A * \cos A = -\frac{12}{25}\)
theorem sin_cos_product (A : α) (h : sin_cos_sum A) : sin A * cos A = -12 / 25 :=
sorry

-- Condition 2: Given along with question 2
def angle_range (A : α) : Prop := 0 < A ∧ A < π

-- Question 2: Prove \(\triangle ABC\) is an obtuse triangle
theorem is_obtuse_triangle (A : α) (h1 : sin_cos_sum A) (h2 : angle_range A) : ¬ (cos A > 0) :=
sorry

-- Condition for Question 3
def is_obtuse (A : α) : Prop := cos A < 0

-- Question 3: Prove \(\tan A = -\frac{4}{3}\)
theorem tan_value (A : α) (h1 : sin_cos_sum A) (h2 : angle_range A) (h3 : is_obtuse A) : tan A = -4 / 3 :=
sorry

end sin_cos_product_is_obtuse_triangle_tan_value_l3_3892


namespace problem_l3_3928

theorem problem (x z : ℝ) (h : |2 * x - log z| = 2 * x + log z) : x * (z - 1) = 0 := sorry

end problem_l3_3928


namespace area_of_shaded_region_l3_3018

noncomputable def sqrt2 : ℝ := real.sqrt 2

theorem area_of_shaded_region (d : ℝ) (A_shaded : ℝ) (π : ℝ := real.pi) (approx : ℝ ≈ 3.1416) :
  d = 28 →
  A_shaded ≈ 84.043 :=
by
  let s := d / sqrt2
  let A_square := s^2
  let r := d / 2
  let A_semi_circle := (1 / 2) * π * r^2
  let A_shaded := A_square - A_semi_circle
  sorry

end area_of_shaded_region_l3_3018


namespace range_of_m_l3_3898

noncomputable def f (x a : ℝ) : ℝ :=
  2 * Real.log x + x^2 - a * x + 2

theorem range_of_m (x0 : ℝ) (a m : ℝ) (h0 : 0 < x0 ∧ x0 ≤ 1) (ha : a ∈ Ioc (-2) 0)
  (h : f x0 a > a^2 + 3 * a + 2 - 2 * m * Real.exp(a) * (a + 1)) :
  m ∈ Icc (-1 / 2 : ℝ) (5 * Real.exp(2) / 2) :=
sorry

end range_of_m_l3_3898


namespace sum_even_integers_202_to_300_is_12550_l3_3001

-- Definitions
def sum_first_n_even_integers (n : ℕ) : ℕ :=
  n * (n + 1)

def sum_even_integers_in_range (start end_ : ℕ) : ℕ :=
  let n := (end_ - start) / 2 + 1 in
  n * (start + end_) / 2

-- Theorem Statement
theorem sum_even_integers_202_to_300_is_12550 :
  sum_even_integers_in_range 202 300 = 12550 := by
sorry

end sum_even_integers_202_to_300_is_12550_l3_3001


namespace todd_ate_8_cupcakes_l3_3966

-- Define the conditions
def cupcakes_baked := 18
def packages := 5
def cupcakes_per_package := 2

-- Define the question as a statement that needs to be proved
theorem todd_ate_8_cupcakes : 
  let cupcakes_left := packages * cupcakes_per_package in
  let cupcakes_eaten := cupcakes_baked - cupcakes_left in
  cupcakes_eaten = 8 := 
by
  sorry

end todd_ate_8_cupcakes_l3_3966


namespace positive_difference_between_balances_l3_3440

noncomputable def angela_balance (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * ((1 + r / n) ^ (n * t))

noncomputable def bob_balance (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r * t)

theorem positive_difference_between_balances :
  let angela_P := 9000
  let angela_r := 0.06
  let angela_n := 2
  let angela_t := 25
  let bob_P := 11000
  let bob_r := 0.07
  let bob_t := 25
  let A := angela_balance angela_P angela_r angela_n angela_t
  let B := bob_balance bob_P bob_r bob_t
  A - B ≈ 9206 := 
by 
  sorry

end positive_difference_between_balances_l3_3440


namespace muffins_ounces_correct_l3_3811

-- Define the given constants
def blueberry_cost_per_carton : ℝ := 5.00
def blueberry_ounces_per_carton : ℝ := 6

def raspberry_cost_per_carton : ℝ := 3.00
def raspberry_ounces_per_carton : ℝ := 8

-- Derived constants
def blueberry_cost_per_ounce : ℝ := blueberry_cost_per_carton / blueberry_ounces_per_carton
def raspberry_cost_per_ounce : ℝ := raspberry_cost_per_carton / raspberry_ounces_per_carton

def cost_difference_per_ounce : ℝ := blueberry_cost_per_ounce - raspberry_cost_per_ounce

-- Define the given conditions
def total_savings : ℝ := 22.0
def number_of_batches : ℝ := 4.0

-- Goal: Find the ounces of fruit per batch
def ounces_of_fruit_per_batch : ℝ := total_savings / (cost_difference_per_ounce * number_of_batches)

-- Statement to prove
theorem muffins_ounces_correct : ounces_of_fruit_per_batch = 12 := by
  sorry

end muffins_ounces_correct_l3_3811


namespace nth_odd_and_sum_first_n_odds_l3_3716

noncomputable def nth_odd (n : ℕ) : ℕ := 2 * n - 1

noncomputable def sum_first_n_odds (n : ℕ) : ℕ := n ^ 2

theorem nth_odd_and_sum_first_n_odds :
  nth_odd 100 = 199 ∧ sum_first_n_odds 100 = 10000 :=
by
  sorry

end nth_odd_and_sum_first_n_odds_l3_3716


namespace number_of_zeros_l3_3145

-- Define the function f with its properties
noncomputable def f (x : ℝ) : ℝ := 
  if x ∈ set.Ioc 0 (3/2) then real.sin (real.pi * x) else 
  if x = 3/2 then 0 else 
  f (x - 3 * real.floor (x / 3) : ℝ) * if even (real.floor (x / 1.5 : ℝ)) then 1 else -1

-- State the main proof problem
theorem number_of_zeros (f : ℝ → ℝ)
  (H1 : ∀ x, f (-x) = -f (x)) -- f is odd
  (H2 : ∀ x, f (x + 3) = f (x)) -- f is periodic with period 3
  (H3 : ∀ x, x ∈ set.Ioc 0 (3/2) → f (x) = real.sin (real.pi * x)) -- f(x) = sin(πx) for 0 < x < 3/2
  (H4 : f (3/2) = 0) -- f(3/2) = 0
  : ∃ n, n = 9 ∧ ∀ a b, [0,6].has_subset (set.Icc a b) → (set.filter (λ x, f x = 0) (set.Icc a b)).card = n :=
sorry

end number_of_zeros_l3_3145


namespace distinct_value_expression_l3_3384

def tri (a b : ℕ) : ℕ := min a b
def nabla (a b : ℕ) : ℕ := max a b

theorem distinct_value_expression (x : ℕ) : (nabla 5 (nabla 4 (tri x 4))) = 5 := 
by
  sorry

end distinct_value_expression_l3_3384


namespace intersection_on_y_axis_l3_3547

theorem intersection_on_y_axis (A C : ℝ) 
  (l1 : ∀ (x y : ℝ), A * x + 3 * y + C = 0) 
  (l2 : ∀ (x y : ℝ), 2 * x - 3 * y + 4 = 0) 
  (h : ∃ y, l1 0 y = 0 ∧ l2 0 y = 0) : 
  C = -4 := 
by
  sorry

end intersection_on_y_axis_l3_3547


namespace trains_meet_at_point_l3_3338

theorem trains_meet_at_point :
  ∃ (X : ℝ) (distance : ℝ),
    X = 45 ∧ distance = 180 ∧
    (∀ (t : ℝ), 0 ≤ t →
      30 * (t + 2) = distance ∧
      36 * (t + 1) = distance ∧
      X * t = distance) :=
begin
  use [45, 180],
  split,
  { refl },
  split,
  { refl },
  intros t ht,
  split,
  { calc 30 * (t + 2) = 30 * t + 60 : by ring },
  split,
  { calc 36 * (t + 1) = 36 * t + 36 : by ring },
  { calc 45 * t = 45 * t : by refl }
end

end trains_meet_at_point_l3_3338


namespace eric_running_time_l3_3841

-- Define the conditions
variables (jog_time to_park_time return_time : ℕ)
axiom jog_time_def : jog_time = 10
axiom return_time_def : return_time = 90
axiom trip_relation : return_time = 3 * to_park_time

-- Define the question
def run_time : ℕ := to_park_time - jog_time

-- State the problem: Prove that given the conditions, the running time is 20 minutes.
theorem eric_running_time : run_time = 20 :=
by
  -- Proof goes here
  sorry

end eric_running_time_l3_3841


namespace quadruplets_babies_l3_3077

/-- The conditions given in the problem -/
variables (t r q p : ℕ)
variables (total_babies : ℕ)
variables (h1 : t = 4 * p)
variables (h2 : r = 2 * q)
variables (h3 : q = 2 * p)
variables (h4 : 2 * t + 3 * r + 4 * q + 5 * p = 1250)

/-- The theorem we need to prove: Number of babies from sets of quadruplets is 303 -/
theorem quadruplets_babies : 4 * q = 303 :=
by {
  sorry,
}

end quadruplets_babies_l3_3077


namespace polynomial_division_result_l3_3255

-- Define the given polynomials
def f (x : ℝ) : ℝ := 4 * x ^ 4 + 12 * x ^ 3 - 9 * x ^ 2 + 2 * x + 3
def d (x : ℝ) : ℝ := x ^ 2 + 2 * x - 3

-- Define the computed quotient and remainder
def q (x : ℝ) : ℝ := 4 * x ^ 2 + 4
def r (x : ℝ) : ℝ := -12 * x + 42

theorem polynomial_division_result :
  (∀ x : ℝ, f x = q x * d x + r x) ∧ (q 1 + r (-1) = 62) :=
by
  sorry

end polynomial_division_result_l3_3255


namespace probability_sin_in_interval_half_l3_3421

noncomputable def probability_sin_interval : ℝ :=
  let a := - (Real.pi / 2)
  let b := Real.pi / 2
  let interval_length := b - a
  (b - 0) / interval_length

theorem probability_sin_in_interval_half :
  probability_sin_interval = 1 / 2 := by
  sorry

end probability_sin_in_interval_half_l3_3421


namespace last_page_dave_should_read_l3_3940

theorem last_page_dave_should_read :
  ∃ n : ℕ, 
    let total_pages := 950
      Dave_time_per_page := 40
      Emma_time_per_page := 50
      reading_ratio := (Emma_time_per_page + Dave_time_per_page)
      Dave_fraction := Dave_time_per_page / reading_ratio
      Dave_pages := (Dave_fraction * total_pages).natFloor
    in n = Dave_pages ∧ n = 422 := 
begin
  sorry
end

end last_page_dave_should_read_l3_3940


namespace move_plane_make_lines_parallel_l3_3617

-- Defining the problem statement in Lean
theorem move_plane_make_lines_parallel (A1 A2 A3 A4 : Point)
                                        (Pi' Pi'' : Plane)
                                        (P1 P2 P3 P4 : Point -> Plane -> Point)
                                        (project : ∀ A : Point, P1 A Pi' -> P2 A Pi'' -> LineSegment) :
    (∃ P : Plane, ∀ i, (move_plane Pi' P)) → 
    (∀ i, parallel (project (list.nth [A1, A2, A3, A4] i) Pi' Pi'')) :=
sorry

end move_plane_make_lines_parallel_l3_3617


namespace sum_of_values_l3_3503

def f : ℝ → ℝ :=
λ x, if x < 1 then cos (π * x) else f (x - 1) - 1

theorem sum_of_values : f (1/3) + f (4/3) = 0 := by
  sorry

end sum_of_values_l3_3503


namespace focus_of_parabola_l3_3305

theorem focus_of_parabola (m : ℝ) (hm : m ≠ 0) :
    (0, m / 4) = let p := m / 4 in (0, p) := by
  sorry

end focus_of_parabola_l3_3305


namespace pencils_placed_by_dan_l3_3695

-- Definitions based on the conditions provided
def pencils_in_drawer : ℕ := 43
def initial_pencils_on_desk : ℕ := 19
def new_total_pencils : ℕ := 78

-- The statement to be proven
theorem pencils_placed_by_dan : pencils_in_drawer + initial_pencils_on_desk + 16 = new_total_pencils :=
by
  sorry

end pencils_placed_by_dan_l3_3695


namespace count_squares_sharing_vertices_with_triangles_l3_3250

theorem count_squares_sharing_vertices_with_triangles
  (A B C D : Type)
  [equilateral_triangle A B C]
  [equilateral_triangle A B D]
  (shared_side : same_side A B) :
  ∃ n : ℕ, n = 12 :=
sorry

end count_squares_sharing_vertices_with_triangles_l3_3250


namespace speed_of_stream_l3_3756

def canoe_speed (upstream downstream : ℝ) : ℝ :=
  (downstream + upstream) / 2

def stream_speed (upstream downstream : ℝ) : ℝ :=
  (downstream - upstream) / 2

theorem speed_of_stream (upstream downstream : ℝ) (h_up : upstream = 3) (h_down : downstream = 12) :
  stream_speed upstream downstream = 4.5 :=
by
  rw [h_up, h_down]
  unfold stream_speed
  norm_num

end speed_of_stream_l3_3756


namespace tan_double_alpha_plus_pi_over_four_sin_cos_fraction_l3_3500

-- Condition: Given tan(α) = 2
variable (α : ℝ) (h₀ : Real.tan α = 2)

-- Statement (1): Prove tan(2α + π/4) = 9
theorem tan_double_alpha_plus_pi_over_four :
  Real.tan (2 * α + Real.pi / 4) = 9 := by
  sorry

-- Statement (2): Prove (6 sin α + cos α) / (3 sin α - 2 cos α) = 13 / 4
theorem sin_cos_fraction :
  (6 * Real.sin α + Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 13 / 4 := by
  sorry

end tan_double_alpha_plus_pi_over_four_sin_cos_fraction_l3_3500


namespace lena_muffins_l3_3282

theorem lena_muffins (x y z : Real) 
  (h1 : x + 2 * y + 3 * z = 3 * x + z)
  (h2 : 3 * x + z = 6 * y)
  (h3 : x + 2 * y + 3 * z = 6 * y)
  (lenas_spending : 2 * x + 2 * z = 6 * y) :
  ∃ (n : ℕ), n = 5 :=
by
  sorry

end lena_muffins_l3_3282


namespace expr_for_pos_x_min_value_l3_3162

section
variable {f : ℝ → ℝ}
variable {a : ℝ}

def even_func (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def func_def (f : ℝ → ℝ) (a : ℝ) : Prop := ∀ x, x ≤ 0 → f x = 4^(-x) - a * 2^(-x)

-- Assuming f is even and specified as in the problem for x ≤ 0
axiom ev_func : even_func f
axiom f_condition : 0 < a

theorem expr_for_pos_x (f : ℝ → ℝ) (a : ℝ) (h1 : even_func f) (h2 : func_def f a) : 
  ∀ x, 0 < x → f x = 4^x - a * 2^x :=
sorry -- this aims to prove the function's form for positive x.

theorem min_value (f : ℝ → ℝ) (a : ℝ) (h1 : even_func f) (h2 : func_def f a) :
  (0 < a ∧ a ≤ 2 → ∃ x, 0 < x ∧ f x = 1 - a) ∧
  (2 < a → ∃ x, 0 < x ∧ f x = -a^2 / 4) :=
sorry -- this aims to prove the minimum value on the interval (0, +∞).
end

end expr_for_pos_x_min_value_l3_3162


namespace inclination_angle_l3_3907

noncomputable def polar_to_cartesian_eqn (rho theta : ℝ) : (ℝ × ℝ) :=
  let x := rho * real.cos theta in
  let y := rho * real.sin theta in
  (x, y)

def curve_C_cartesian : ℝ × ℝ → Prop :=
  λ p, let (x, y) := p in (x - 2) ^ 2 + y ^ 2 = 4

def line_l_parametric (t α : ℝ) : ℝ × ℝ :=
  (1 + t * real.cos α, t * real.sin α)

theorem inclination_angle (α : ℝ) (pt1 pt2 : ℝ) :
  curve_C_cartesian (line_l_parametric pt1 α) ∧
  curve_C_cartesian (line_l_parametric pt2 α) ∧
  (∥⟨1 + pt1 * real.cos α, pt1 * real.sin α⟩ - ⟨1 + pt2 * real.cos α, pt2 * real.sin α⟩∥ = real.sqrt 14)
  → (α = real.pi / 4 ∨ α = 3 * real.pi / 4) :=
sorry

end inclination_angle_l3_3907


namespace distribute_money_equation_l3_3220

theorem distribute_money_equation (x : ℕ) (hx : x > 0) : 
  (10 : ℚ) / x = (40 : ℚ) / (x + 6) := 
sorry

end distribute_money_equation_l3_3220


namespace area_difference_l3_3029

theorem area_difference (P : ℕ) (L : ℕ) (side length : ℕ) (width : ℕ)
  (h1 : P = 52)
  (h2 : 4 * side length = P)
  (h3 : 2 * (L + width) = P)
  (h4 : L = 15) :
  (side length * side length) - (L * width) = 4 :=
by
  sorry

end area_difference_l3_3029


namespace shooting_competition_sequences_l3_3592

/--
In a shooting competition, ten clay targets are arranged in three hanging columns as follows:
Column A has 4 targets, Column B has 3 targets, Column C has 3 targets.
A marksman must break all the targets following these rules:
1) The marksman first chooses a column.
2) The marksman must then break the lowest unbroken target in the chosen column.

We need to determine the number of different sequences in which all ten targets can be broken.
-/
theorem shooting_competition_sequences :
  let total_targets := 10
  let targets_A := 4
  let targets_B := 3
  let targets_C := 3
  let total_permutations := Nat.factorial total_targets
  let permutations_A := Nat.factorial targets_A
  let permutations_B := Nat.factorial targets_B
  let permutations_C := Nat.factorial targets_C in
  total_permutations / (permutations_A * permutations_B * permutations_C) = 4200 :=
by
  -- Definitions and calculations based on the problem conditions
  let total_targets := 10
  let targets_A := 4
  let targets_B := 3
  let targets_C := 3
  let total_permutations := Nat.factorial total_targets
  let permutations_A := Nat.factorial targets_A
  let permutations_B := Nat.factorial targets_B
  let permutations_C := Nat.factorial targets_C
  -- Expected outcome based on calculations in the solution
  have correct_answer : total_permutations / (permutations_A * permutations_B * permutations_C) = 4200 := by sorry
  exact correct_answer

end shooting_competition_sequences_l3_3592


namespace sector_angle_solution_l3_3536

theorem sector_angle_solution (R α : ℝ) (h1 : 2 * R + α * R = 6) (h2 : (1/2) * R^2 * α = 2) : α = 1 ∨ α = 4 := 
sorry

end sector_angle_solution_l3_3536


namespace total_number_of_students_l3_3751

namespace StudentRanking

def rank_from_right := 17
def rank_from_left := 5
def total_students (rank_from_right rank_from_left : ℕ) := rank_from_right + rank_from_left - 1

theorem total_number_of_students : total_students rank_from_right rank_from_left = 21 :=
by
  sorry

end StudentRanking

end total_number_of_students_l3_3751


namespace stamps_in_scrapbook_l3_3336

variable (Parker_initial_stamps : ℕ) (Addie_total_stamps : ℕ)

theorem stamps_in_scrapbook (h1 : Parker_initial_stamps = 18) (h2 : Addie_total_stamps = 72) :
  let Addie_stamps_added := Addie_total_stamps / 4
  let Parker_final_stamps := Parker_initial_stamps + Addie_stamps_added
  Parker_final_stamps = 36 :=
by
  -- Conditions
  have Addie_stamps := Addie_total_stamps / 4
  have Parker_add_stamps := Parker_initial_stamps + Addie_stamps
  -- Replacing the values
  rw [h1, h2]
  have Addie_stamps_calculation : Addie_stamps = 72 / 4 := by sorry
  have Parker_stamps_calculation : Parker_add_stamps = 18 + (72 / 4) := by sorry
  rw Addie_stamps_calculation
  simp at Parker_stamps_calculation
  exact Parker_stamps_calculation
  sorry

end stamps_in_scrapbook_l3_3336


namespace toaster_customer_count_l3_3460

theorem toaster_customer_count :
  ∀ (p₁ p₂ : ℕ) (c₁ c₂ : ℕ) (d : ℕ),
  (p₁ = 12) →
  (c₁ = 600) →
  (c₂ = 400) →
  (d = 10) →
  (p₁ * c₁ = p₂ * (c₂ * 2 * (100 - d) / 100)) →
  p₂ = 10 :=
by
  intros,
  sorry

end toaster_customer_count_l3_3460


namespace x5_plus_x16_eq_20_l3_3576

-- Definition that the sequence {1/x_n} is harmonic
def harmonic_sequence (x : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → (1 / x (n + 1)) - (1 / x n) = d

-- Given conditions
variables {x : ℕ → ℝ} {d : ℝ}
axiom harmonic_seq : harmonic_sequence x
axiom sum_eq_200 : ∑ i in finset.range 20, x (i + 1) = 200

-- The main theorem to prove
theorem x5_plus_x16_eq_20 : x 5 + x 16 = 20 :=
sorry

end x5_plus_x16_eq_20_l3_3576


namespace complement_intersection_l3_3922

def S := {1, 2, 3, 4, 5}
def M := {1, 4}
def N := {2, 4}

def complement (S : Set ℕ) (A : Set ℕ) := {x ∈ S | x ∉ A}

theorem complement_intersection : 
  complement S M ∩ complement S N = {3, 5} :=
by
  sorry

end complement_intersection_l3_3922


namespace computation_l3_3455

theorem computation :
  4.165 * 4.8 + 4.165 * 6.7 - 4.165 / (2 / 3) = 41.65 :=
by
  sorry

end computation_l3_3455


namespace maisie_flyers_count_l3_3274

theorem maisie_flyers_count (M : ℕ) (h1 : 71 = 2 * M + 5) : M = 33 :=
by
  sorry

end maisie_flyers_count_l3_3274


namespace max_bead_volume_l3_3315

noncomputable def max_volume_bead (diameter : ℝ) (cone_volume : ℝ) : ℝ :=
  if diameter = 12 ∧ cone_volume = 96 * real.pi then 36 * real.pi else 0

theorem max_bead_volume : max_volume_bead 12 (96 * real.pi) = 36 * real.pi :=
by
  sorry

end max_bead_volume_l3_3315


namespace problem_statement_l3_3742

def vector := (ℝ × ℝ × ℝ)

def are_parallel (v1 v2 : vector) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ v1 = (k * v2.1, k * v2.2, k * v2.3)

def are_perpendicular (v1 v2 : vector) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3 = 0

theorem problem_statement :
  (are_parallel (2, 3, -1) (-2, -3, 1)) ∧
  (¬ are_perpendicular (1, -1, 2) (6, 4, -1)) ∧
  (are_perpendicular (2, 2, -1) (-3, 4, 2)) ∧
  (¬ are_parallel (0, 3, 0) (0, -5, 0)) :=
by
  sorry

end problem_statement_l3_3742


namespace multiplicative_inverse_CD_mod_1000000_l3_3993

theorem multiplicative_inverse_CD_mod_1000000 :
  let C := 123456
  let D := 166666
  let M := 48
  M * (C * D) % 1000000 = 1 := by
  sorry

end multiplicative_inverse_CD_mod_1000000_l3_3993


namespace no_such_a_exists_for_slope_k_l3_3998

def f (x a : ℝ) := x - (1 / x) - a * Real.log x

theorem no_such_a_exists_for_slope_k (a x₁ x₂ k : ℝ) (h_extreme_points : f x₁ a = (x₁ - (1 / x₁) - a * Real.log x₁) ∧ 
                                             f x₂ a = (x₂ - (1 / x₂) - a * Real.log x₂)) 
                                     (h_slope : k = 2 - a) 
                                     (h_x1_x2 : x₁ * x₂ = 1) : False :=
sorry

end no_such_a_exists_for_slope_k_l3_3998


namespace triangle_is_isosceles_l3_3585

theorem triangle_is_isosceles
    (a b c : ℝ)
    (A B C : ℝ)
    (h1 : a = 2 * c * Real.cos B)
    (h2 : b = c * Real.cos A) 
    (h3 : c = a * Real.cos C) 
    : a = b := 
sorry

end triangle_is_isosceles_l3_3585


namespace max_median_cans_l3_3281

-- Definitions based on conditions
def total_cans : ℕ := 305
def total_customers : ℕ := 120

-- Theorem statement
theorem max_median_cans (h : ∀ i : ℕ, i < total_customers → 1 ≤ i) : 
  ∃ m : ℝ, m = 5.0 ∧ 
  (let c := list.repeat 1 59 ++ list.repeat 5 61 in 
    (c.nth 59).getD 0.0 = m ∧ (c.nth 60).getD 0.0 = m) := 
sorry

end max_median_cans_l3_3281


namespace greatest_common_divisor_of_72_and_m_l3_3704

-- Definitions based on the conditions
def is_power_of_prime (m : ℕ) : Prop :=
  ∃ (p : ℕ) (k : ℕ), Nat.Prime p ∧ m = p^k

-- Main theorem based on the question and conditions
theorem greatest_common_divisor_of_72_and_m (m : ℕ) :
  (Nat.gcd 72 m = 9) ↔ (m = 3^2) ∨ ∃ k, k ≥ 2 ∧ m = 3^k :=
by
  sorry

end greatest_common_divisor_of_72_and_m_l3_3704


namespace sin_alpha_plus_7_pi_over_6_l3_3871

theorem sin_alpha_plus_7_pi_over_6
  (α : ℝ)
  (h : sin (π / 3 + α) + sin α = 4 * real.sqrt 3 / 5) : sin (α + 7 * π / 6) = -4 / 5 :=
sorry

end sin_alpha_plus_7_pi_over_6_l3_3871


namespace gcd_polynomial_l3_3528

theorem gcd_polynomial (b : ℕ) (h : 570 ∣ b) : Nat.gcd (5 * b^3 + 2 * b^2 + 5 * b + 95) b = 95 :=
by
  sorry

end gcd_polynomial_l3_3528


namespace order_of_reading_amounts_l3_3593

variable (a b c d : ℝ)

theorem order_of_reading_amounts (h1 : a + c = b + d) (h2 : a + b > c + d) (h3 : d > b + c) :
  a > d ∧ d > b ∧ b > c :=
by
  sorry

end order_of_reading_amounts_l3_3593


namespace inequality_amgm_l3_3287

-- Defining the statement of our problem
theorem inequality_amgm (k n : ℕ) (x : ℕ → ℝ) (h : ∀ i, 1 ≤ i ∧ i ≤ k → x i > 0) : 
  (∏ i in range k, x i) * (∑ i in range k, x i ^ (n - 1)) ≤ ∑ i in range k, x i ^ (n + k - 1) := 
sorry

end inequality_amgm_l3_3287


namespace solve_quartic_equation_l3_3292

theorem solve_quartic_equation (a b c : ℤ) (x : ℤ) : 
  x^4 + a * x^2 + b * x + c = 0 :=
sorry

end solve_quartic_equation_l3_3292


namespace bisection_method_next_interval_contains_root_l3_3709

def f (x : ℝ) : ℝ := x^3 - 2 * x - 5

theorem bisection_method_next_interval_contains_root 
  (x₀ : ℝ)
  (h_interval : 2 < x₀ ∧ x₀ < 3)
  (h_midpoint : x₀ = (2 + 3) / 2)
  (h_sign_change : f 2 * f x₀ < 0) :
  ∃ c, c ∈ set.Ioo 2 x₀ ∧ f c = 0 := 
by {
  -- sorry is a placeholder for the actual proof which is not required
  sorry 
}

end bisection_method_next_interval_contains_root_l3_3709


namespace university_theater_receipts_l3_3348

def total_receipts
  (total_tickets : ℕ)
  (adult_ticket_price : ℕ)
  (senior_ticket_price : ℕ)
  (senior_tickets_sold : ℕ)
  : ℕ :=
  let adult_tickets_sold := total_tickets - senior_tickets_sold in
  let total_adult_receipts := adult_ticket_price * adult_tickets_sold in
  let total_senior_receipts := senior_ticket_price * senior_tickets_sold in
  total_adult_receipts + total_senior_receipts

theorem university_theater_receipts :
  total_receipts 529 25 15 348 = 9745 := by
  unfold total_receipts
  let adult_tickets_sold := 529 - 348
  let total_adult_receipts := 25 * adult_tickets_sold
  let total_senior_receipts := 15 * 348
  calc 
    total_adult_receipts + total_senior_receipts
      = (25 * adult_tickets_sold) + (15 * 348) : by rw [←total_adult_receipts, ←total_senior_receipts]
  ... = (25 * 181) + (15 * 348) : by rw [sub_eq_iff_eq_add, nat.sub_self, nat.add_sub_self, nat.add_self_eq]
  ... = 4525 + 5220 : by norm_num
  ... = 9745 : by norm_num

end university_theater_receipts_l3_3348


namespace angle_between_a_and_neg_b_l3_3889

variables {V : Type*} [inner_product_space ℝ V]

def magnitude (v : V) : ℝ := real.sqrt (inner_product_space.inner v v)

theorem angle_between_a_and_neg_b
  (a b : V)
  (ha : magnitude a = 1)
  (hb : magnitude b = real.sqrt 2)
  (h_perp : inner_product_space.inner a (a - b) = 0) :
  real.angle a (-b) = 3 * real.pi / 4 :=
sorry

end angle_between_a_and_neg_b_l3_3889


namespace inverse_of_f_l3_3316

def f (x : ℝ) : ℝ := real.sqrt x + 1

theorem inverse_of_f : ∀ (y : ℝ), y ≥ 0 → f ((y - 1)^2) = y := by 
  sorry

end inverse_of_f_l3_3316


namespace polar_coordinates_of_point_l3_3101

def convert_to_polar (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let theta :=
    if y = 0 ∧ x > 0 then 0
    else if y > 0   then Real.arctan (y / x)
    else if y < 0 ∧ x > 0 then Real.arctan (y / x) + 2 * Real.pi
    else Real.arctan (y / x) + Real.pi
  (r,theta)

theorem polar_coordinates_of_point (x y : ℝ) (r θ: ℝ) (h1: x = 2) (h2: y = -2) :
  convert_to_polar x y = (r, θ) ↔ (r = 2 * Real.sqrt 2) ∧ (θ = 7 * Real.pi / 4) :=
by
  sorry
 
end polar_coordinates_of_point_l3_3101


namespace decagonal_pyramid_volume_l3_3058

noncomputable def volume_of_decagonal_pyramid (m : ℝ) (apex_angle : ℝ) : ℝ :=
  let sin18 := Real.sin (18 * Real.pi / 180)
  let sin36 := Real.sin (36 * Real.pi / 180)
  let cos18 := Real.cos (18 * Real.pi / 180)
  (5 * m^3 * sin36) / (3 * (1 + 2 * cos18))

theorem decagonal_pyramid_volume : volume_of_decagonal_pyramid 39 (18 * Real.pi / 180) = 20023 :=
  sorry

end decagonal_pyramid_volume_l3_3058


namespace description_of_T_l3_3975

def T : Set (ℝ × ℝ) := { p | ∃ c, (4 = p.1 + 3 ∨ 4 = p.2 - 2 ∨ p.1 + 3 = p.2 - 2) 
                           ∧ (p.1 + 3 ≤ c ∨ p.2 - 2 ≤ c ∨ 4 ≤ c) }

theorem description_of_T : 
  (∀ p ∈ T, (∃ x y : ℝ, p = (x, y) ∧ ((x = 1 ∧ y ≤ 6) ∨ (y = 6 ∧ x ≤ 1) ∨ (y = x + 5 ∧ x ≥ 1 ∧ y ≥ 6)))) :=
sorry

end description_of_T_l3_3975


namespace tan_angle_PAB_l3_3284

noncomputable theory
open_locale real

-- Definitions of the triangle and the point
variables {A B C P : Type} [euclidean_geometry A B C P]
variables (AB BC CA : ℝ) (PAB PBC PCA : ℝ)

-- Given conditions
-- Point P lies inside triangle ABC such that angles are equal
def point_inside_triangle_with_equal_angles (A B C P : Type) [euclidean_geometry A B C P]
  (PAB PBC PCA : ℝ) : Prop :=
  ∠A P B = PAB ∧ ∠B P C = PBC ∧ ∠C P A = PCA

-- Given side lengths
def triangle_sides (A B C : Type) [euclidean_geometry A B C] (AB BC CA : ℝ) : Prop :=
  AB = 8 ∧ BC = 17 ∧ CA = 15

-- The main theorem to prove
theorem tan_angle_PAB {A B C P : Type} [euclidean_geometry A B C P]
  (PAB PBC PCA : ℝ) (AB BC CA : ℝ)
  (H1 : point_inside_triangle_with_equal_angles A B C P PAB PBC PCA)
  (H2 : triangle_sides A B C AB BC CA) :
  tan PAB = 168 / 289 :=
by
  sorry

end tan_angle_PAB_l3_3284


namespace ramu_profit_percent_l3_3379

noncomputable def profit_percent (purchase_price repair_cost selling_price : ℝ) : ℝ :=
  let total_cost := purchase_price + repair_cost
  let profit := selling_price - total_cost
  (profit / total_cost) * 100

theorem ramu_profit_percent :
  profit_percent 42000 13000 64500 = 17.27 :=
by
  -- Placeholder for the proof
  sorry

end ramu_profit_percent_l3_3379


namespace veggies_minus_fruits_l3_3006

-- Definitions of quantities as given in the conditions
def cucumbers : ℕ := 6
def tomatoes : ℕ := 8
def apples : ℕ := 2
def bananas : ℕ := 4

-- Problem Statement
theorem veggies_minus_fruits : (cucumbers + tomatoes) - (apples + bananas) = 8 :=
by 
  -- insert proof here
  sorry

end veggies_minus_fruits_l3_3006


namespace smallestProduct_l3_3690

def numSet : Set ℤ := { -8, -4, -2, 1, 5 }

theorem smallestProduct : ∃ a b ∈ numSet, a ≠ b ∧ a * b = -40 :=
by
  use [-8, 5]
  split
  repeat { try { split, simp [numSet] } }
  sorry

end smallestProduct_l3_3690


namespace ratio_of_salmon_sold_l3_3697

theorem ratio_of_salmon_sold 
  (first_week_salmon_sold : ℝ) (total_salmon_sold : ℝ) 
  (h1 : first_week_salmon_sold = 50) 
  (h2 : total_salmon_sold = 200) : 
  let second_week_salmon_sold := total_salmon_sold - first_week_salmon_sold in
  (second_week_salmon_sold / first_week_salmon_sold) = 3 :=
by 
  sorry

end ratio_of_salmon_sold_l3_3697


namespace area_of_triangle_is_one_fourth_l3_3753

def area_of_region (AC BC : ℝ) : ℝ := (1/2) * (((AC / 2) + (BC / 2)) * ((AC / 2) + (BC / 2)) - (AC / 2) * (AC / 2) - (BC / 2) * (BC / 2))

theorem area_of_triangle_is_one_fourth (CD AB AC BC : ℝ) (h1 : CD = 1) (h2 : CD * AB = 1):
  area_of_region AC BC = 1/4 := 
by
  -- Skip the proof for now
  sorry

end area_of_triangle_is_one_fourth_l3_3753


namespace total_non_overlapping_area_of_squares_l3_3654

theorem total_non_overlapping_area_of_squares 
  (side_length : ℕ) 
  (num_squares : ℕ)
  (overlapping_areas_count : ℕ)
  (overlapping_width : ℕ)
  (overlapping_height : ℕ)
  (total_area_with_overlap: ℕ)
  (final_missed_patch_ratio: ℕ)
  (final_adjustment: ℕ) 
  (total_area: ℕ :=  total_area_with_overlap-final_missed_patch_ratio ):
  side_length = 2 ∧ 
  num_squares = 4 ∧ 
  overlapping_areas_count = 3 ∧ 
  overlapping_width = 1 ∧ 
  overlapping_height = 2 ∧
  total_area_with_overlap = 16- 3  ∧
  final_missed_patch_ratio = 3-> 
  total_area = 13 := 
 by sorry

end total_non_overlapping_area_of_squares_l3_3654


namespace kittens_and_mice_count_l3_3969

theorem kittens_and_mice_count :
  let children := 12
  let baskets_per_child := 3
  let cats_per_basket := 1
  let kittens_per_cat := 12
  let mice_per_kitten := 4
  let total_kittens := children * baskets_per_child * cats_per_basket * kittens_per_cat
  let total_mice := total_kittens * mice_per_kitten
  total_kittens + total_mice = 2160 :=
by
  sorry

end kittens_and_mice_count_l3_3969


namespace total_hours_spent_l3_3102

theorem total_hours_spent :
  let w := 24 in
  let c1 := 2 in
  let h1 := 3 in
  let c2 := 1 in
  let h2 := 4 in
  let hw := 4 in
  let ls := 8 in
  let lh := 6 in
  let p1 := 10 in
  let p2 := 14 in
  let p3 := 18 in
  let class_hours := w * (c1 * h1 + c2 * h2) in
  let homework_hours := w * hw in
  let lab_hours := ls * lh in
  let project_hours := p1 + p2 + p3 in
  class_hours + homework_hours + lab_hours + project_hours = 426 :=
by
  sorry

end total_hours_spent_l3_3102


namespace cos_monotonically_increasing_l3_3106

noncomputable def interval_increasing (k : ℤ) : set ℝ :=
{ x | 2*k*π - (3*π)/4 ≤ x ∧ x ≤ 2*k*π + π/4 }

theorem cos_monotonically_increasing :
  ∀ k : ℤ, ∀ x : ℝ, (x ∈ interval_increasing k) ↔ ∃ k ∈ ℤ, 
    2*k*π - (3*π)/4 ≤ x ∧ x ≤ 2*k*π + π/4 :=
by
  sorry

end cos_monotonically_increasing_l3_3106


namespace options_a_and_c_correct_l3_3726

theorem options_a_and_c_correct :
  (let a₁ := (2, 3, -1) in
   let b₁ := (-2, -3, 1) in
   let a₁_is_parallel_b₁ := a₁ = (-1 : ℝ) • b₁ in
   (let a₂ := (1, -1, 2) in
    let u₂ := (6, 4, -1) in
    let not_parallel_or_perpendicular := ¬(a₂ = (6 : ℝ) • u₂) ∧ a₂.1 * u₂.1 + a₂.2 * u₂.2 + a₂.3 * u₂.3 ≠ 0 in
    (let u₃ := (2, 2, -1) in
     let v₃ := (-3, 4, 2) in
     let dot_product_zero := u₃.1 * v₃.1 + u₃.2 * v₃.2 + u₃.3 * v₃.3 = 0 in
     (let a₄ := (0, 3, 0) in
      let u₄ := (0, -5, 0) in
      let parallel_but_not_perpendicular := a₄ = (3 / 5 : ℝ) • u₄) in
     a₁_is_parallel_b₁ ∧ dot_product_zero))) :=
sorry

end options_a_and_c_correct_l3_3726


namespace triangle_area_ABC_l3_3880

variable {A : Prod ℝ ℝ}
variable {B : Prod ℝ ℝ}
variable {C : Prod ℝ ℝ}

noncomputable def area_of_triangle (A B C : Prod ℝ ℝ ) : ℝ :=
  (1 / 2) * (abs ((A.1 * (B.2 - C.2)) + (B.1 * (C.2 - A.2)) + (C.1 * (A.2 - B.2))))

theorem triangle_area_ABC : 
  ∀ {A B C : Prod ℝ ℝ}, 
  A = (2, 3) → 
  B = (5, 7) → 
  C = (6, 1) → 
  area_of_triangle A B C = 11 
:= by
  intros
  subst_vars
  simp [area_of_triangle]
  sorry

end triangle_area_ABC_l3_3880


namespace sum_first_9_terms_eq_99_l3_3222

variable {a : ℕ → ℤ} -- The sequence in question

-- Conditions from the problem
def cond1 : Prop := a 1 + a 4 + a 7 = 39
def cond2 : Prop := a 3 + a 6 + a 9 = 27

-- Definition of the sum of the first 9 terms
def sum_first_9_terms : ℤ := a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9

-- The statement we need to prove, with the necessary conditions
theorem sum_first_9_terms_eq_99 (h1 : cond1) (h2 : cond2) : sum_first_9_terms = 99 := 
begin 
  sorry 
end

end sum_first_9_terms_eq_99_l3_3222


namespace highest_third_term_of_geometric_progression_l3_3426

theorem highest_third_term_of_geometric_progression :
  ∃ d : ℝ, (5 : ℝ) + d ≥ 0 ∧ 
  let a2 := (10 : ℝ) + d in 
  let a3 := (35 : ℝ) + 2 * d in
  (a2 * a2 = (5 : ℝ) * a3) ∧
  (a3 = 45) :=
sorry

end highest_third_term_of_geometric_progression_l3_3426


namespace probability_blue_is_7_over_50_l3_3766

-- Define the total number of tiles
def total_tiles : ℕ := 100

-- Define what it means for a tile to be blue
def is_blue (n : ℕ) : Prop := n % 7 = 3

-- Define the set of blue tiles
def blue_tiles_set : Finset ℕ := (Finset.range total_tiles).filter is_blue

-- Define the number of blue tiles
def num_blue_tiles : ℕ := blue_tiles_set.card

-- Define the probability calculation
def probability_blue : ℚ := num_blue_tiles / total_tiles

-- Prove the probability of selecting a blue tile is 7/50
theorem probability_blue_is_7_over_50 : probability_blue = 7 / 50 := 
by
  sorry

end probability_blue_is_7_over_50_l3_3766


namespace dice_sum_not_11_l3_3963

/-- Jeremy rolls three standard six-sided dice, with each showing a different number and the product of the numbers on the upper faces is 72.
    Prove that the sum 11 is not possible. --/
theorem dice_sum_not_11 : 
  ∃ (a b c : ℕ), 
    (1 ≤ a ∧ a ≤ 6) ∧ 
    (1 ≤ b ∧ b ≤ 6) ∧ 
    (1 ≤ c ∧ c ≤ 6) ∧ 
    (a ≠ b ∧ a ≠ c ∧ b ≠ c) ∧ 
    (a * b * c = 72) ∧ 
    (a > 4 ∨ b > 4 ∨ c > 4) → 
    a + b + c ≠ 11 := 
by
  sorry

end dice_sum_not_11_l3_3963


namespace find_equation_of_line_l3_3507

variable (l : Line) (P : Point) (α : ℝ)
axiom inclination_angle (l : Line) : ℝ
axiom passes_through (l : Line) (P : Point) : Prop
axiom perpendicular_to (l m : Line) : Prop
axiom intercept_sum_zero (l : Line) : Prop

def Point := ℝ × ℝ
noncomputable def Line := { f : ℝ × ℝ → ℝ // ∀ p : Point, f p = 0 }

theorem find_equation_of_line :
  ∃ l : Line,
  passes_through l (2, 3) ∧
  inclination_angle l = 120 ∧
  perpendicular_to l ⟨fun p => p.1 - 2 * p.2 + 1⟩ ∧
  intercept_sum_zero l ∧
  (l = ⟨fun p => √3 * p.1 + p.2 - 3 - 2 * √3⟩ ∨
   l = ⟨fun p => 2 * p.1 + p.2 - 7⟩ ∨
   l = ⟨fun p => 3 * p.1 - 2 * p.2⟩ ∨
   l = ⟨fun p => p.1 - p.2 + 1⟩) :=
by {
  sorry
}

end find_equation_of_line_l3_3507


namespace snow_probability_l3_3320

theorem snow_probability :
  let p_snow_day1 := 1 / 2,
      p_snow_day2 := 3 / 4,
      p_snow_day3 := 2 / 3 in
  1 - ((1 - p_snow_day1) * (1 - p_snow_day2) * (1 - p_snow_day3)) = 23 / 24 :=
by
  sorry

end snow_probability_l3_3320


namespace parking_methods_count_l3_3332

theorem parking_methods_count : ∃ n : ℕ, n = 528 ∧
  (let spaces := 6 in
   let cars := 3 in
   let adjacent_face_same_direction := true in
   let non_adjacent_no_direction_limit := true in
   -- The following constraints are expressed in plain English for clarity.
   -- The total number of different parking methods for 3 cars in 6 consecutive parking spaces,
   -- given that adjacent cars must face the same direction and non-adjacent cars can face any direction,
   -- is 528).
   sorry)

end parking_methods_count_l3_3332


namespace prob_A_fee_exactly_6_yuan_prob_sum_fees_A_B_36_yuan_l3_3587

section ParkingProblem

variable (P_A_more_1_no_more_2 : ℚ) (P_A_more_than_14 : ℚ)

theorem prob_A_fee_exactly_6_yuan :
  (P_A_more_1_no_more_2 = 1/3) →
  (P_A_more_than_14 = 5/12) →
  (1 - (P_A_more_1_no_more_2 + P_A_more_than_14)) = 1/4 :=
by
  -- Skipping the proof
  intros _ _
  sorry

theorem prob_sum_fees_A_B_36_yuan :
  (1/4 : ℚ) = 1/4 :=
by
  -- Skipping the proof
  exact rfl

end ParkingProblem

end prob_A_fee_exactly_6_yuan_prob_sum_fees_A_B_36_yuan_l3_3587


namespace volume_of_pyramid_proof_l3_3062

def volume_of_pyramid (a : ℝ) : ℝ := (4 * a^3) / 7

theorem volume_of_pyramid_proof {a : ℝ} (h₁ : a > 0) 
    (h₂ : ∃ r : ℝ, r > 0 ∧ abstract_height : r * 16 = a * 21) : 
    volume_of_pyramid a = (4 * a^3) / 7 := 
sorry

end volume_of_pyramid_proof_l3_3062


namespace maximize_three_digit_product_l3_3601

-- Define the problem conditions
def is_valid_digit (d : Nat) : Prop := d > 0 ∧ d < 10

def three_digit_number (a b c : Nat) : Prop :=
  is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c

-- Function definitions for sum of reciprocals and the number itself
def reciprocal_sum (a b c : Nat) := (1 / (a : Real)) + (1 / (b : Real)) + (1 / (c : Real))

def number (a b c : Nat) := 100 * a + 10 * b + c

-- Define the problem statement
def max_product : Real :=
  1923.222

theorem maximize_three_digit_product :
  ∃ (a b c : Nat), three_digit_number a b c ∧ 
  (number a b c * reciprocal_sum a b c = max_product) :=
sorry

end maximize_three_digit_product_l3_3601


namespace Iris_total_spent_l3_3611

theorem Iris_total_spent :
  let jackets := 3
  let cost_per_jacket := 10
  let shorts := 2
  let cost_per_short := 6
  let pants := 4
  let cost_per_pant := 12
  jackets * cost_per_jacket + shorts * cost_per_short + pants * cost_per_pant = 90 := by
  sorry

end Iris_total_spent_l3_3611


namespace expected_volunteers_in_2004_l3_3605

noncomputable def volunteer_count (n : ℕ) : ℕ → ℝ
| 0       := 500
| (n + 1) := volunteer_count n * 1.5

theorem expected_volunteers_in_2004 : volunteer_count 4 = 2532 := 
by sorry

end expected_volunteers_in_2004_l3_3605


namespace right_angled_triangles_l3_3258

-- Define the problem conditions
variable {p : ℚ}  -- Rational number p

-- The point M in Cartesian coordinates
def M : ℚ × ℚ := (p * 2002, 7 * p * 2002)

-- Systematically define conditions and the problem
-- Here we just state the problem, and so we leave the proof as sorry

theorem right_angled_triangles (h1 : M = (p * 2002, 7 * p * 2002)) 
  (h2 : ∃ x y z : ℚ × ℚ, (
    (x = (0, 0) ∧ -- Origin is one vertex
    y = M ∧      -- M is the right angle vertex
    z ∈ ℚ × ℚ ∧ -- z is another lattice point
    is_lattice_point x ∧ is_lattice_point z ∧ is_lattice_point y ∧ -- All are lattice points
    is_right_angle x y z) -- M is the right angle vertex
  )) : 
  let num_triangles := if p = 2 then 162 else if p = 7 ∨ p = 11 ∨ p = 13 then 180 else 324 in
  num_right_angled_triangles_with_incenter_origin M = num_triangles :=
sorry

-- Helper definitions
def is_lattice_point (x : ℚ × ℚ) : Prop :=
  ∃ m n : ℤ, x = (m, n)

def is_right_angle (a b c : ℚ × ℚ) : Prop := 
  let angle := (b.1 - a.1) * (c.1 - a.1) + (b.2 - a.2) * (c.2 - a.2) in
  angle = 0

def num_right_angled_triangles_with_incenter_origin (M : ℚ × ℚ) : ℕ :=
  -- Placeholder function, implementation needed
  sorry

end right_angled_triangles_l3_3258


namespace eval_expr_eq_2_l3_3024

noncomputable def evaluate_expression : ℕ :=
  let term1 := 3 ^ (0 ^ (2 ^ 3))
  let term2 := ((3 ^ 1) ^ 0) ^ 2
  term1 + term2

theorem eval_expr_eq_2 : evaluate_expression = 2 := 
by
  let term1 := 3 ^ (0 ^ (2 ^ 3))
  let term2 := ((3 ^ 1) ^ 0) ^ 2
  have h1 : 2 ^ 3 = 8 := by norm_num
  have h2 : 0 ^ 8 = 0 := by norm_num
  have h3 : 3 ^ 0 = 1 := by simp
  have h4 : term1 = 1 := by rw [h1, h2, h3]
  have h5 : 3 ^ 1 = 3 := by norm_num
  have h6 : 3 ^ 0 = 1 := by simp
  have h7 : term2 = 1 := by rw [h5, h6]; norm_num
  show 1 + 1 = 2
  exact eq.refl 2

#eval eval_expr_eq_2

end eval_expr_eq_2_l3_3024


namespace find_b_when_a_is_5_l3_3328

variable (a b : ℝ)

-- Given conditions
def inversely_proportional (a b : ℝ) : Prop := ∃ k, a * b = k
axiom sum_condition : a + b = 24
axiom diff_condition : a - b = 6
axiom a_equals_5 : a = 5

-- The goal is to prove b = 27
theorem find_b_when_a_is_5
    (h1 : inversely_proportional a b)
    (h2 : sum_condition)
    (h3 : diff_condition)
    (h4 : a_equals_5) : b = 27 := by
  sorry

end find_b_when_a_is_5_l3_3328


namespace cos_minus_sin_eq_neg_sqrt_three_over_two_l3_3884

theorem cos_minus_sin_eq_neg_sqrt_three_over_two (θ : ℝ) (h1 : sin θ * cos θ = 1/8) (h2 : π/4 < θ ∧ θ < π/2) : 
  cos θ - sin θ = -√3/2 := 
by 
  sorry

end cos_minus_sin_eq_neg_sqrt_three_over_two_l3_3884


namespace tan_half_ineq1_cot_tan_half_ineq2_tan_cot_half_ineq3_cot_csc_ineq4_l3_3937

theorem tan_half_ineq1 (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (hABC : A + B + C = π) :
  tan (A / 2) + tan (B / 2) + tan (C / 2) - tan (A / 2) * tan (B / 2) * tan (C / 2) ≥ (8 / 9) * sqrt 3 := sorry

theorem cot_tan_half_ineq2 (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (hABC : A + B + C = π) :
  cot (A / 2) * cot (B / 2) * cot (C / 2) - tan (A / 2) * tan (B / 2) * tan (C / 2) ≥ (26 / 9) * sqrt 3 := sorry

theorem tan_cot_half_ineq3 (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (hABC : A + B + C = π) :
  tan (A / 2) + tan (B / 2) + tan (C / 2) + cot (A / 2) + cot (B / 2) + cot (C / 2) ≥ 4 * sqrt 3 := sorry

theorem cot_csc_ineq4 (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (hABC : A + B + C = π) :
  cot A + cot B + cot C ≥ (1 / 2) * (csc A + csc B + csc C) := sorry

end tan_half_ineq1_cot_tan_half_ineq2_tan_cot_half_ineq3_cot_csc_ineq4_l3_3937


namespace angle_between_v1_v2_l3_3113

-- Define vectors
def v1 : ℝ × ℝ := (3, -4)
def v2 : ℝ × ℝ := (4, 6)

-- Define the dot product function
def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

-- Define the magnitude function
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Define the cosine of the angle between two vectors
noncomputable def cos_theta (a b : ℝ × ℝ) : ℝ := (dot_product a b) / (magnitude a * magnitude b)

-- Define the angle in degrees between two vectors
noncomputable def angle_between_vectors (a b : ℝ × ℝ) : ℝ := Real.arccos (cos_theta a b) * (180 / Real.pi)

-- The statement to prove
theorem angle_between_v1_v2 : angle_between_vectors v1 v2 = Real.arccos (-6 * Real.sqrt 13 / 65) * (180 / Real.pi) :=
sorry

end angle_between_v1_v2_l3_3113


namespace find_range_of_m_l3_3154

noncomputable def p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + 2 * x + m ≠ 0

noncomputable def q (m : ℝ) : Prop :=
  ∀ x : ℝ, mx^2 - x + (1 / 16) * m > 0

theorem find_range_of_m (m : ℝ) : ((p m) ∨ (q m)) ∧ ¬ ((p m) ∧ (q m)) ↔ 1 < m ∧ m ≤ 2 :=
begin
  sorry
end

end find_range_of_m_l3_3154


namespace card_M_cap_N_l3_3574

def M : set (ℝ × ℝ) := 
  {p | ∃ x y, p = (x, y) ∧ (tan (π * y) + sin (π * x) * sin (π * x) = 0)}

def N : set (ℝ × ℝ) := 
  {p | ∃ x y, p = (x, y) ∧ (x^2 + y^2 ≤ 2)}

theorem card_M_cap_N : 
  fintype.card ({p : ℤ × ℤ | p ∈ (M ∩ N)} : set (ℤ × ℤ)) = 9 :=
sorry

end card_M_cap_N_l3_3574


namespace sum_sin_squared_l3_3858

theorem sum_sin_squared (n : ℕ) (hn : 0 < n) : 
    \\
  let x := Real.pi / (2 * n) in
  \\
  \sum_{k \in Finset.range (n + 1)} (Real.sin ((k : ℝ) * x))^2 = (n + 1) / 2 := 
by sorry

end sum_sin_squared_l3_3858


namespace units_digit_of_T_l3_3566

noncomputable def factorial_units_digit (n : ℕ) : ℕ :=
match n with
| 0 => 1
| 1 => 1
| 2 => 2
| 3 => 6
| 4 => 4
| _ => 0

noncomputable def sum_units (n : ℕ) : ℕ :=
(nat.range n).map factorial_units_digit |>.sum % 10

theorem units_digit_of_T : sum_units 16 = 3 :=
by
  sorry

end units_digit_of_T_l3_3566


namespace proof_l3_3552

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := { x | x ^ 2 ≤ 1 }

-- Define set B
def B : Set ℝ := { x | 2 ^ x ≤ 1 }

-- Define the complement of B
def complement_B : Set ℝ := { x | x > 0 }

-- Define A intersection complement of B
def A_inter_complement_B : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 ∧ x > 0 }

-- State the theorem to prove
theorem proof : A_inter_complement_B = { x | 0 < x ∧ x ≤ 1 } :=
by sorry

end proof_l3_3552


namespace log5_rounded_nearest_l3_3715

theorem log5_rounded_nearest (log_inc : ∀ x y : ℝ, x < y → Real.logb 5 x < Real.logb 5 y)
  (log_3125 : Real.logb 5 3125 = 5) (log_625 : Real.logb 5 625 = 4) : 
  Int.round (Real.logb 5 3120) = 5 := 
by
  sorry

end log5_rounded_nearest_l3_3715


namespace distinct_values_of_d_l3_3979

theorem distinct_values_of_d (d : ℂ) (x y z : ℂ) (h_distinct: x ≠ y ∧ y ≠ z ∧ z ≠ x) 
  (h_eq : ∀ w : ℂ, (w - x)*(w - y)*(w - z) = (w - d*x)*(w - d*y)*(w - d*z)) : 
  {d : ℂ | ∃ (d' : ℂ), d' = d}.toFinset.card = 4 :=
by 
  sorry

end distinct_values_of_d_l3_3979


namespace tan_ratio_proof_l3_3252

noncomputable def tan_ratio (a b : ℝ) : ℝ := Real.tan a / Real.tan b

theorem tan_ratio_proof (a b : ℝ) (h1 : Real.sin (a + b) = 5 / 8) (h2 : Real.sin (a - b) = 1 / 3) : 
tan_ratio a b = 23 / 7 := by
  sorry

end tan_ratio_proof_l3_3252


namespace line_through_fixed_point_l3_3497

theorem line_through_fixed_point (a : ℝ) :
  ∃ P : ℝ × ℝ, (P = (1, 2)) ∧ (∀ x y, a * x + y - a - 2 = 0 → P = (x, y)) ∧
  ((∃ a, x + y = a ∧ x = 1 ∧ y = 2) → (a = 3)) :=
by
  sorry

end line_through_fixed_point_l3_3497


namespace sum_of_all_possible_x_l3_3952

theorem sum_of_all_possible_x : 
  (∀ x : ℝ, |x - 5| - 4 = -1 → (x = 8 ∨ x = 2)) → ( ∃ (x1 x2 : ℝ), (x1 = 8) ∧ (x2 = 2) ∧ (x1 + x2 = 10) ) :=
by
  admit

end sum_of_all_possible_x_l3_3952


namespace proof_result_l3_3734

noncomputable def direction_vector (a b : ℝ × ℝ × ℝ) := 
  ∃ k : ℝ, (k • b) = a

noncomputable def perpendicular_vector (a b : ℝ × ℝ × ℝ) := 
  (a.1 * b.1 + a.2 * b.2 + a.3 * b.3) = 0

def parallel_line_condition := 
  direction_vector (2, 3, -1) (-2, -3, 1) 

def perpendicular_plane_condition := 
  perpendicular_vector (2, 2, -1) (-3, 4, 2)

theorem proof_result: parallel_line_condition ∧ perpendicular_plane_condition :=
by
  have h1 : parallel_line_condition := sorry
  have h2 : perpendicular_plane_condition := sorry
  exact ⟨h1, h2⟩

end proof_result_l3_3734


namespace merchant_profit_percentage_is_five_l3_3777

-- Assume the following constants and calculations
def cost_price : ℝ := 100
def marked_up_percentage : ℝ := 0.75
def discount_percentage : ℝ := 0.40

-- Calculate mark up amount
def marked_price : ℝ := cost_price * (1 + marked_up_percentage)

-- Calculate discount amount
def discount : ℝ := marked_price * discount_percentage

-- Calculate selling price
def selling_price : ℝ := marked_price - discount

-- Calculate profit
def profit : ℝ := selling_price - cost_price

-- Calculate profit percentage
def profit_percentage : ℝ := (profit / cost_price) * 100

-- The main theorem to prove that the profit percentage is 5%
theorem merchant_profit_percentage_is_five : profit_percentage = 5 :=
by sorry

end merchant_profit_percentage_is_five_l3_3777


namespace sqrt_sum_of_powers_l3_3724

theorem sqrt_sum_of_powers :
  sqrt (2^4 + 2^4 + 4^2) = 4 * sqrt 3 :=
by 
  sorry

end sqrt_sum_of_powers_l3_3724


namespace find_annual_interest_rate_l3_3301

def compoundInterest (P A : ℝ) (n t : ℕ) (r : ℝ) :=
  A = P * (1 + r / n) ^ (n * t)

theorem find_annual_interest_rate :
  compoundInterest 8000 8400 1 1 0.05 :=
by
  unfold compoundInterest
  rw [Real.rpow, Real.pow]
  sorry

end find_annual_interest_rate_l3_3301


namespace line_intersects_parabola_at_vertex_l3_3105

theorem line_intersects_parabola_at_vertex :
  ∃ (a : ℝ), (∀ x : ℝ, -x + a = x^2 + a^2) ↔ a = 0 ∨ a = 1 :=
by
  sorry

end line_intersects_parabola_at_vertex_l3_3105


namespace median_first_fifteen_positive_even_integers_l3_3020

noncomputable def a (n : ℕ) : ℕ := 2 * n

theorem median_first_fifteen_positive_even_integers : 
  (median (list.map a (list.range 15))) = 16 :=
by
  sorry

end median_first_fifteen_positive_even_integers_l3_3020


namespace sum_a_equals_half_sum_b_l3_3247

-- Define the set S as described in the problem conditions
def S : Set (ℤ × ℤ) :=
  { p | let a := p.1
        let b := p.2
        0 < 2 * a ∧ 2 * a < 2 * b ∧ 2 * b < 2017 ∧ (a ^ 2 + b ^ 2) % 2017 = 0 }

-- The statement to be proven: the sum of the first components of elements in S equals half the sum of the second components
theorem sum_a_equals_half_sum_b (S : Set (ℤ × ℤ)) :
  (∑ (p : ℤ × ℤ) in S, p.1) = (1 / 2) * (∑ (p : ℤ × ℤ) in S, p.2) := sorry

end sum_a_equals_half_sum_b_l3_3247


namespace cubes_fit_and_occupy_volume_l3_3422

-- Define the dimensions of the container and the cube
def container_length := 8
def container_width := 4
def container_height := 9
def cube_side := 2

-- Volume calculations
def container_volume := container_length * container_width * container_height
def cube_volume := cube_side * cube_side * cube_side

-- Number of cubes that can fit along each dimension (length, width, height)
def num_cubes_length := container_length / cube_side
def num_cubes_width := container_width / cube_side
def num_cubes_height := container_height / cube_side

-- Total number of cubes fitting inside the container
def total_num_cubes := (num_cubes_length.toNat) * (num_cubes_width.toNat) * (num_cubes_height.toNat)

-- Total volume of cubes fitting inside the container
def total_cube_volume := total_num_cubes * cube_volume

-- Fraction of container volume occupied by cubes
def fraction_occupied := total_cube_volume / container_volume

-- Percent of container volume occupied by cubes
def percent_occupied := fraction_occupied * 100

theorem cubes_fit_and_occupy_volume :
  total_num_cubes = 32 ∧ percent_occupied = 88.89 :=
by
  sorry

end cubes_fit_and_occupy_volume_l3_3422


namespace tom_tim_age_sum_l3_3701

theorem tom_tim_age_sum (T : ℕ) (h1 : 15 = 15) (h2 : 15 + 3 = 2 * (T + 3)) : 15 + T = 21 :=
begin
  sorry
end

end tom_tim_age_sum_l3_3701


namespace police_catch_up_time_l3_3785

-- Define the conditions and constants
def v : ℝ := 1 -- speed of rogue spaceship in units per hour, use 1 for simplicity of calculations
def v_police : ℝ := 1.12 * v  -- speed of police spaceship in units per hour
def head_start_minutes : ℝ := 54.0 -- head start time in minutes
def head_start_hours : ℝ := head_start_minutes / 60.0 -- convert head start to hours
def distance_head_start : ℝ := v * head_start_hours  -- distance covered by rogue spaceship during head start
def relative_speed : ℝ := v_police - v  -- relative speed of police spaceship compared to rogue spaceship

-- Define the formula to calculate the catch up time
def catch_up_time_hours : ℝ := distance_head_start / relative_speed  -- catch up time in hours
def catch_up_time_minutes : ℝ := catch_up_time_hours * 60.0  -- catch up time in minutes

-- The theorem to prove
theorem police_catch_up_time : catch_up_time_minutes = 450 :=
by
  sorry

end police_catch_up_time_l3_3785


namespace grandma_olga_daughters_l3_3558

theorem grandma_olga_daughters :
  ∃ (D : ℕ), ∃ (S : ℕ),
  S = 3 ∧
  (∃ (total_grandchildren : ℕ), total_grandchildren = 33) ∧
  (∀ D', 6 * D' + 5 * S = 33 → D = D')
:=
sorry

end grandma_olga_daughters_l3_3558


namespace problem_statement_l3_3741

def vector := (ℝ × ℝ × ℝ)

def are_parallel (v1 v2 : vector) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ v1 = (k * v2.1, k * v2.2, k * v2.3)

def are_perpendicular (v1 v2 : vector) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3 = 0

theorem problem_statement :
  (are_parallel (2, 3, -1) (-2, -3, 1)) ∧
  (¬ are_perpendicular (1, -1, 2) (6, 4, -1)) ∧
  (are_perpendicular (2, 2, -1) (-3, 4, 2)) ∧
  (¬ are_parallel (0, 3, 0) (0, -5, 0)) :=
by
  sorry

end problem_statement_l3_3741


namespace punctures_covered_l3_3400

theorem punctures_covered (P1 P2 P3 : ℝ) (h1 : 0 ≤ P1) (h2 : P1 < P2) (h3 : P2 < P3) (h4 : P3 < 3) :
    ∃ x, x ≤ P1 ∧ x + 2 ≥ P3 := 
sorry

end punctures_covered_l3_3400


namespace T_n_sum_l3_3618

open Real

noncomputable def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q ^ (n - 1)

noncomputable def sum_of_sequence (f : ℕ → ℝ) (n : ℕ) : ℝ :=
  (finset.range n).sum f

theorem T_n_sum {n : ℕ} (hpos : n > 0) :
  let a₁ := 3
  let q := 3
  let a := geometric_sequence a₁ q
  let b := λ n, log 3 (a (2 * n - 1))
  let S := sum_of_sequence b n
  let c := λ n, 1 / (4 * S - 1)
  let T := sum_of_sequence c n
  T = n / (2 * n + 1) :=
by {
  sorry
}

end T_n_sum_l3_3618


namespace cos_product_identity_l3_3883

theorem cos_product_identity (n : ℕ) (hn : 0 < n) :
  (finset.prod (finset.range n) (λ k, real.cos (↑k.succ * real.pi / (2 * n + 1)))) = (1 / (2^n)) :=
by
  sorry

end cos_product_identity_l3_3883


namespace sum_of_areas_is_correct_l3_3132

/-- Define the lengths of the rectangles -/
def lengths : List ℕ := [4, 16, 36, 64, 100]

/-- Define the common base width of the rectangles -/
def base_width : ℕ := 3

/-- Define the area of a rectangle given its length and a common base width -/
def area (length : ℕ) : ℕ := base_width * length

/-- Compute the total area of the given rectangles -/
def total_area : ℕ := (lengths.map area).sum

/-- Theorem stating that the total area of the five rectangles is 660 -/
theorem sum_of_areas_is_correct : total_area = 660 := by
  sorry

end sum_of_areas_is_correct_l3_3132


namespace hexagon_unique_intersection_points_are_45_l3_3828

-- Definitions related to hexagon for the proof problem
def hexagon_vertices : ℕ := 6
def sides_of_hexagon : ℕ := 6
def diagonals_of_hexagon : ℕ := 9
def total_line_segments : ℕ := 15
def total_intersections : ℕ := 105
def vertex_intersections_per_vertex : ℕ := 10
def total_vertex_intersections : ℕ := 60

-- Final Proof Statement that needs to be proved
theorem hexagon_unique_intersection_points_are_45 :
  total_intersections - total_vertex_intersections = 45 :=
by
  sorry

end hexagon_unique_intersection_points_are_45_l3_3828


namespace maple_pine_ratio_l3_3588

theorem maple_pine_ratio 
  (total_trees : ℕ) (ancient_oaks : ℕ) (medium_firs : ℕ) (tall_palms : ℕ) 
  (saplings : ℕ) (maple_saplings : ℕ) (pine_saplings : ℕ) 
  (total_trees = 150) (ancient_oaks = 20) (medium_firs = 35) (tall_palms = 25) 
  (saplings + ancient_oaks + medium_firs + tall_palms = total_trees) 
  (maple_saplings = 2 * pine_saplings) 
  (maple_saplings + pine_saplings = saplings) : 
  maple_saplings / pine_saplings = 2 := 
sorry

end maple_pine_ratio_l3_3588


namespace proof_f_1_add_g_2_l3_3619

def f (x : ℤ) : ℤ := 3 * x - 4
def g (x : ℤ) : ℤ := x + 1

theorem proof_f_1_add_g_2 : f (1 + g 2) = 8 := by
  sorry

end proof_f_1_add_g_2_l3_3619


namespace value_of_ab_l3_3932

theorem value_of_ab (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 25) : ab = 8 :=
by
  sorry

end value_of_ab_l3_3932


namespace f_f_minus_two_l3_3885

def f (x : ℚ) : ℚ := x⁻¹ + (x⁻¹ / (1 + x⁻¹))

theorem f_f_minus_two : f (f (-2)) = -8 / 3 := by
  sorry

end f_f_minus_two_l3_3885


namespace train_length_l3_3792

theorem train_length (speed_kmph : ℝ) (time_sec : ℝ) (speed_ms : ℝ) (length_m : ℝ)
  (h1 : speed_kmph = 120) 
  (h2 : time_sec = 6)
  (h3 : speed_ms = 33.33)
  (h4 : length_m = 200) : 
  speed_kmph * 1000 / 3600 * time_sec = length_m :=
by
  sorry

end train_length_l3_3792


namespace marys_next_birthday_l3_3276

theorem marys_next_birthday (d s m : ℝ) (h1 : s = 0.7 * d) (h2 : m = 1.3 * s) (h3 : m + s + d = 25.2) : m + 1 = 9 :=
by
  sorry

end marys_next_birthday_l3_3276


namespace jackson_chairs_l3_3236

theorem jackson_chairs (a b c d : ℕ) (h1 : a = 6) (h2 : b = 4) (h3 : c = 12) (h4 : d = 6) : a * b + c * d = 96 := 
by sorry

end jackson_chairs_l3_3236


namespace nth_equation_holds_l3_3278

theorem nth_equation_holds (n : ℕ) (h : 0 < n) :
  1 / (n + 2) + 2 / (n^2 + 2 * n) = 1 / n :=
by
  sorry

end nth_equation_holds_l3_3278


namespace alpha_values_l3_3273

-- Define the given values and conditions
variables (b c : ℝ) (α : ℝ)
def AB := 2 * c
def AC := 2 * b
def angleBAC := α

def AM := b / (Real.cos α)
def AN := c / (Real.cos α)

-- Given equations from cosine theorem
def NM_squared := (c ^ 2 + b ^ 2 - 2 * b * c * (Real.cos α)) / (Real.cos α) ^ 2
def BC_squared := 4 * (c ^ 2 + b ^ 2 - 2 * b * c * (Real.cos α))

-- MN equals BC
def MN_eq_BC : Prop := NM_squared = BC_squared

-- Prove the angles
theorem alpha_values (h : MN_eq_BC) : α = 60 * Real.pi / 180 ∨ α = 120 * Real.pi / 180 :=
by sorry

end alpha_values_l3_3273


namespace decrease_travel_time_l3_3968

variable (distance : ℕ) (initial_speed : ℕ) (speed_increase : ℕ)

def original_travel_time (distance initial_speed : ℕ) : ℕ :=
  distance / initial_speed

def new_travel_time (distance new_speed : ℕ) : ℕ :=
  distance / new_speed

theorem decrease_travel_time (h₁ : distance = 600) (h₂ : initial_speed = 50) (h₃ : speed_increase = 25) :
  original_travel_time distance initial_speed - new_travel_time distance (initial_speed + speed_increase) = 4 :=
by
  sorry

end decrease_travel_time_l3_3968


namespace additional_time_due_to_leak_l3_3407

theorem additional_time_due_to_leak : 
  let R := (1:ℝ) / 6 in 
  let L := (1:ℝ) / 24 in 
  (1 / (R - L)) - 6 = 2 :=
by
  sorry

end additional_time_due_to_leak_l3_3407


namespace has_odd_prime_factor_distinct_from_p1_to_pk_l3_3504

theorem has_odd_prime_factor_distinct_from_p1_to_pk
  (k : ℕ)
  (h_k : k ≥ 2)
  (p : ℕ → ℕ)
  (h_p : ∀ i, 1 ≤ i ∧ i ≤ k → Prime (p i))
  (h_odd : ∀ i, 1 ≤ i ∧ i ≤ k → Odd (p i))
  (a : ℕ)
  (h_gcd : Nat.gcd a (List.prod (List.map p (List.range k))) = 1) :
  ∃ q : ℕ, Prime q ∧ q ≠ p 0 ∧ (∀ i, 1 ≤ i ∧ i ≤ k → q ≠ p i) ∧ q ∣ (a ^ (List.prod (List.map (λ i, p i - 1) (List.range k))) - 1) := by
  sorry

end has_odd_prime_factor_distinct_from_p1_to_pk_l3_3504


namespace simplify_expression_l3_3357

theorem simplify_expression : 
  (Real.sqrt 1 + Real.sqrt (1 + 2) + Real.sqrt (1 + 2 + 3) + Real.sqrt (1 + 2 + 3 + 4)) = 1 + Real.sqrt 3 + Real.sqrt (1 + 2 + 3) + Real.sqrt (1 + 2 + 3 + 4) :=
begin
  sorry
end

end simplify_expression_l3_3357


namespace convert_to_cartesian_l3_3686

theorem convert_to_cartesian (ρ θ : ℝ) (h : ρ = 4 * Real.sin θ) : 
  (∃ (x y : ℝ), x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ x^2 + (y - 2)^2 = 4) :=
by
  use [ρ * Real.cos θ, ρ * Real.sin θ]
  split
  { rw h, rw Real.sin, rw Real.cos }
  split
  { rw h, rw Real.sin }
  sorry

end convert_to_cartesian_l3_3686


namespace isabel_pictures_l3_3388

theorem isabel_pictures
  (phone_pics : ℕ)
  (camera_pics : ℕ)
  (total_albums : ℕ)
  (h_phone_pics : phone_pics = 2)
  (h_camera_pics : camera_pics = 4)
  (h_total_albums : total_albums = 3) :
  (phone_pics + camera_pics) / total_albums = 2 :=
by
  sorry

end isabel_pictures_l3_3388


namespace evaluate_log8_64_l3_3477

noncomputable def log8_64 : ℝ := Real.logBase 8 64

theorem evaluate_log8_64 : log8_64 = 2 := by
  sorry

end evaluate_log8_64_l3_3477


namespace vectors_orthogonal_and_norm_diff_l3_3272

variables (a b : ℝ³)

-- Given conditions
axiom norm_a : ∥a∥ = 1
axiom norm_b : ∥b∥ = 1
axiom norm_diff : ∥b - 2 • a∥ = sqrt 5

-- Proof problem
theorem vectors_orthogonal_and_norm_diff (a b : ℝ³) 
  (norm_a : ∥a∥ = 1) 
  (norm_b : ∥b∥ = 1) 
  (norm_diff : ∥b - 2 • a∥ = sqrt 5) : 
  (a ⬝ b = 0) ∧ (∥a - b∥ = sqrt 2) := 
sorry

end vectors_orthogonal_and_norm_diff_l3_3272


namespace range_of_abscissa_of_P_l3_3530

noncomputable def line (P : ℝ × ℝ) : Prop := P.1 - P.2 + 1 = 0

noncomputable def circle (M : ℝ × ℝ) : Prop := (M.1 - 2) ^ 2 + (M.2 - 1) ^ 2 = 1

noncomputable def condition_on_P (P : ℝ × ℝ) : Prop :=
  line P ∧ ∃ M N : ℝ × ℝ, circle M ∧ circle N ∧ angle_eq60 M P N

theorem range_of_abscissa_of_P :
  ∃ x : set ℝ, ∀ P : ℝ × ℝ, condition_on_P P → P.1 ∈ set.range x → P.1 ∈ set.Icc 0 2 :=
sorry

end range_of_abscissa_of_P_l3_3530


namespace jogger_distance_l3_3575

theorem jogger_distance (t : ℝ) (h1 : (20 * t = 12 * t + 15)) : (12 * t = 22.5) := by
  -- We use the equation 20t = 12t + 15
  have h2 : 8 * t = 15 := by linarith
  -- Solve for t
  have h3 : t = 15 / 8 := by linarith
  -- Substitute t back into 12 * t
  show 12 * t = 22.5 by
    rw [h3]
    norm_num

end jogger_distance_l3_3575


namespace find_common_difference_l3_3948

variable (a an Sn d : ℚ)
variable (n : ℕ)

def arithmetic_sequence (a : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  a + (n - 1) * d

def sum_arithmetic_sequence (a : ℚ) (an : ℚ) (n : ℕ) : ℚ :=
  n * (a + an) / 2

theorem find_common_difference
  (h1 : a = 3)
  (h2 : an = 50)
  (h3 : Sn = 318)
  (h4 : an = arithmetic_sequence a d n)
  (h5 : Sn = sum_arithmetic_sequence a an n) :
  d = 47 / 11 :=
by
  sorry

end find_common_difference_l3_3948


namespace no_three_pairwise_coprime_divisible_squares_l3_3471

theorem no_three_pairwise_coprime_divisible_squares :
  ¬ ∃ (a b c : ℕ), Nat.coprime a b ∧ Nat.coprime b c ∧ Nat.coprime c a ∧
    (a^2 ∣ (b + c)) ∧ (b^2 ∣ (a + c)) ∧ (c^2 ∣ (a + b)) :=
by
  sorry

end no_three_pairwise_coprime_divisible_squares_l3_3471


namespace watch_cost_price_l3_3435

theorem watch_cost_price (C : ℝ) (h1 : 0.85 * C = SP1) (h2 : 1.06 * C = SP2) (h3 : SP2 - SP1 = 350) : 
  C = 1666.67 := 
  sorry

end watch_cost_price_l3_3435


namespace sum_sin_squared_l3_3857

theorem sum_sin_squared (n : ℕ) (hn : 0 < n) : 
    \\
  let x := Real.pi / (2 * n) in
  \\
  \sum_{k \in Finset.range (n + 1)} (Real.sin ((k : ℝ) * x))^2 = (n + 1) / 2 := 
by sorry

end sum_sin_squared_l3_3857


namespace options_a_and_c_correct_l3_3728

theorem options_a_and_c_correct :
  (let a₁ := (2, 3, -1) in
   let b₁ := (-2, -3, 1) in
   let a₁_is_parallel_b₁ := a₁ = (-1 : ℝ) • b₁ in
   (let a₂ := (1, -1, 2) in
    let u₂ := (6, 4, -1) in
    let not_parallel_or_perpendicular := ¬(a₂ = (6 : ℝ) • u₂) ∧ a₂.1 * u₂.1 + a₂.2 * u₂.2 + a₂.3 * u₂.3 ≠ 0 in
    (let u₃ := (2, 2, -1) in
     let v₃ := (-3, 4, 2) in
     let dot_product_zero := u₃.1 * v₃.1 + u₃.2 * v₃.2 + u₃.3 * v₃.3 = 0 in
     (let a₄ := (0, 3, 0) in
      let u₄ := (0, -5, 0) in
      let parallel_but_not_perpendicular := a₄ = (3 / 5 : ℝ) • u₄) in
     a₁_is_parallel_b₁ ∧ dot_product_zero))) :=
sorry

end options_a_and_c_correct_l3_3728


namespace tan_L_eq_l3_3590

-- Given conditions definitions
variables {JKL : Type} [metric_space JKL]
variables {J K L : JKL}
variable (right_angle_J : ∠ J K L = 90)
variable (length_KL : dist K L = 13)
variable (length_JL : dist J L = 12)

-- Definition of JK using Pythagorean theorem
def length_JK : ℝ := sqrt (dist K L ^ 2 - dist J L ^ 2)

-- Statement to prove
theorem tan_L_eq : tan L = 12 / 5 :=
by 
  -- Skipping the proof with sorry
  sorry

end tan_L_eq_l3_3590


namespace center_of_symmetry_l3_3114

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * Real.tan (2 * x - π / 4)

-- Define the statement to prove the centers of symmetry
theorem center_of_symmetry (k : ℤ) : ∃ x : ℝ, f x = 0 ∧ x = (k * π / 4 + π / 8) := 
sorry

end center_of_symmetry_l3_3114


namespace number_of_children_riding_tricycles_l3_3078

-- Definitions
def bicycles_wheels := 2
def tricycles_wheels := 3

def adults := 6
def total_wheels := 57

-- Problem statement
theorem number_of_children_riding_tricycles (c : ℕ) (H : 12 + 3 * c = total_wheels) : c = 15 :=
by
  sorry

end number_of_children_riding_tricycles_l3_3078


namespace cos_75_cos_15_plus_sin_75_sin_15_l3_3452

theorem cos_75_cos_15_plus_sin_75_sin_15 :
  (Real.cos (75 * Real.pi / 180) * Real.cos (15 * Real.pi / 180) + 
   Real.sin (75 * Real.pi / 180) * Real.sin (15 * Real.pi / 180)) = (1 / 2) := by
  sorry

end cos_75_cos_15_plus_sin_75_sin_15_l3_3452


namespace min_possible_value_of_a1_a2_l3_3002

noncomputable def min_sum_a1_a2 : ℕ :=
  let n := 2028 in
  let f : ℕ × ℕ → ℕ := λ (a₁ a₂: ℕ), a₁ + a₂ in
  if h : ∃ (a₁ a₂ : ℕ), a₁ > 0 ∧ a₂ > 0 ∧ (1 + a₁) * (1 + a₂) = 2029
  then 104
  else 0

theorem min_possible_value_of_a1_a2 :
    ∀ (a₁ a₂: ℕ), a₁ > 0 → a₂ > 0 → a₁ * a₂ = 2028 → min_sum_a1_a2 = 104 :=
begin
  intros,
  sorry,
end

end min_possible_value_of_a1_a2_l3_3002


namespace sum_of_valid_member_counts_eq_2807_l3_3049

-- Define the problem conditions
def isValidMemberCount (s : ℕ) : Prop :=
  150 ≤ s ∧ s ≤ 250 ∧ ∃ k : ℕ, s = 7 * k + 1

-- Prove that the sum of all valid member counts is 2807
theorem sum_of_valid_member_counts_eq_2807 :
  ∑ s in Finset.filter isValidMemberCount (Finset.range 251), s = 2807 :=
by
  sorry

end sum_of_valid_member_counts_eq_2807_l3_3049


namespace find_base_h_l3_3131

noncomputable def base_h_addition_equation (h : ℕ) : Bool :=
  let units := 4 + 3 = 7
  let tens := 6 + 2 = 8
  let hundreds := 7 + 6 = 1 * h + 3
  let thousands := 8 + 9 + 1 = 1 * h + 8
  let ten_thousands := 1 = 2
 in units && tens && (hundreds && thousands && ten_thousands)

theorem find_base_h : ∃ h : ℕ, base_h_addition_equation h = true :=
  ⟨10, by { sorry }⟩

end find_base_h_l3_3131


namespace technical_class_average_age_l3_3214

noncomputable def average_age_in_technical_class : ℝ :=
  let average_age_arts := 21
  let num_arts_classes := 8
  let num_technical_classes := 5
  let overall_average_age := 19.846153846153847
  let total_classes := num_arts_classes + num_technical_classes
  let total_age_university := overall_average_age * total_classes
  ((total_age_university - (average_age_arts * num_arts_classes)) / num_technical_classes)

theorem technical_class_average_age :
  average_age_in_technical_class = 990.4 :=
by
  sorry  -- Proof to be provided

end technical_class_average_age_l3_3214


namespace original_number_is_16_l3_3573

theorem original_number_is_16 (x : ℕ) : 213 * x = 3408 → x = 16 :=
by
  sorry

end original_number_is_16_l3_3573


namespace solve_ineq_l3_3502

theorem solve_ineq (a x : ℝ) :
  (a < 0 → (x ≤ 3 / a ∨ x ≥ 1) ↔ ax^2 - (a + 3)x + 3 ≤ 0) ∧
  (a = 0 → (x ≥ 1) ↔ a*x^2 - (a + 3)*x + 3 ≤ 0) ∧
  (0 < a ∧ a < 3 → (1 ≤ x ∧ x ≤ 3 / a) ↔ ax^2 - (a + 3)x + 3 ≤ 0) ∧
  (a = 3 → (x = 1) ↔ a*x^2 - (a + 3)*x + 3 ≤ 0) ∧
  (a > 3 → (3 / a ≤ x ∧ x ≤ 1) ↔ ax^2 - (a + 3)*x + 3 ≤ 0) :=
by
  sorry

end solve_ineq_l3_3502


namespace gwendolyn_reading_time_l3_3559

/--
Gwendolyn can read 200 sentences in 1 hour. 
Each paragraph has 10 sentences. 
There are 20 paragraphs per page. 
The book has 50 pages. 
--/
theorem gwendolyn_reading_time : 
  let sentences_per_hour := 200
  let sentences_per_paragraph := 10
  let paragraphs_per_page := 20
  let pages := 50
  let sentences_per_page := sentences_per_paragraph * paragraphs_per_page
  let total_sentences := sentences_per_page * pages
  (total_sentences / sentences_per_hour) = 50 := 
by
  let sentences_per_hour : ℕ := 200
  let sentences_per_paragraph : ℕ := 10
  let paragraphs_per_page : ℕ := 20
  let pages : ℕ := 50
  let sentences_per_page : ℕ := sentences_per_paragraph * paragraphs_per_page
  let total_sentences : ℕ := sentences_per_page * pages
  have h : (total_sentences / sentences_per_hour) = 50 := by sorry
  exact h

end gwendolyn_reading_time_l3_3559


namespace monotonicity_find_ab_l3_3542

-- Definitions and conditions
def f (ω x : ℝ) : ℝ := sqrt 3 * sin (ω * x) * cos (ω * x) - cos (ω * x) ^ 2 + 1/2
def ω_pos (ω : ℝ) : Prop := ω > 0
def axis_symmetry (ω x : ℝ) : Prop := x = π / 3
def zero_adjacent (ω x : ℝ) : Prop := x = π / 12
def c_value : ℝ := sqrt 3
def vector_m (A : ℝ) : ℝ × ℝ := (1, sin A)
def vector_n (B : ℝ) : ℝ × ℝ := (2, sin B)
def collinear (A B : ℝ) : Prop := sin B = 2 * sin A

-- Monotonicity in the given interval
theorem monotonicity (ω : ℝ) (hω : ω_pos ω) :
  ∀ x ∈ Icc (-π / 12) (5 * π / 12),
    if x ∈ Icc (-π / 12) (π / 3) then 
         monotone_on (f ω) (Icc (-π / 12) (π / 3))
    else 
         antitone_on (f ω) (Icc (π / 3) (5 * π / 12)) :=
sorry

-- Given the triangle conditions, find values of a and b
theorem find_ab (a b A B C : ℝ) (hC : C = π / 3) (hcol : collinear A B) :
  (b = 2 * a) →
  (c_value ^ 2 = a ^ 2 + b ^ 2 - a * b) →
  a = 1 ∧ b = 2 :=
by
  intros h1 h2
  -- Proof would follow here
  sorry

end monotonicity_find_ab_l3_3542


namespace graph_representation_l3_3714

theorem graph_representation {x y : ℝ} (h : x^2 * (x - y - 2) = y^2 * (x - y - 2)) :
  ( ∃ a : ℝ, ∀ (x : ℝ), y = a * x ) ∨ 
  ( ∃ b : ℝ, ∀ (x : ℝ), y = b * x ) ∨ 
  ( ∃ c : ℝ, ∀ (x : ℝ), y = x - 2 ) ∧ 
  (¬ ∃ d : ℝ, ∀ (x : ℝ), y = d * x ∧ y = d * x - 2) :=
sorry

end graph_representation_l3_3714


namespace total_cost_is_15_75_l3_3415

def price_sponge : ℝ := 4.20
def price_shampoo : ℝ := 7.60
def price_soap : ℝ := 3.20
def tax_rate : ℝ := 0.05
def total_cost_before_tax : ℝ := price_sponge + price_shampoo + price_soap
def tax_amount : ℝ := tax_rate * total_cost_before_tax
def total_cost_including_tax : ℝ := total_cost_before_tax + tax_amount

theorem total_cost_is_15_75 : total_cost_including_tax = 15.75 :=
by sorry

end total_cost_is_15_75_l3_3415


namespace find_g_sum_l3_3137

def g (n : ℕ) : ℝ := log 3003 (n ^ 3)

theorem find_g_sum : g 7 + g 11 + g 13 = 9 / 4 :=
by sorry

end find_g_sum_l3_3137


namespace event_of_three_6s_is_random_l3_3790

/-- A student rolls 3 dice at once, and the event of getting three 6s is a random event. -/
theorem event_of_three_6s_is_random : 
  let dice_rolls := [1, 2, 3, 4, 5, 6]
  in let event_getting_three_6s := ((6 ∈ dice_rolls) ∧ (6 ∈ dice_rolls) ∧ (6 ∈ dice_rolls))
  in ∃ (probability : ℝ), probability > 0 :=
by 
  sorry

end event_of_three_6s_is_random_l3_3790


namespace self_intersections_cannot_divide_half_l3_3769

noncomputable def is_closed_self_intersecting_polygonal_path {V : Type*}
  (points : list V) (segments : list (set V)) : Prop :=
  -- Define the conditions for a closed, self-intersecting polygonal path here
  sorry

noncomputable def intersects_once_per_segment {V : Type*}
  (segments : list (set V)) : Prop :=
  -- Define condition that each segment intersects exactly once
  sorry

noncomputable def two_segments_per_intersection {V : Type*}
  (points : list V) (segments : list (set V)) : Prop :=
  -- Define condition that exactly two segments pass through each intersection
  sorry

noncomputable def no_intersection_at_vertices {V : Type*}
  (points : list V) (segments : list (set V)) : Prop :=
  -- Define condition that there are no self-intersections at vertices
  sorry

noncomputable def no_common_segments {V : Type*}
  (segments : list (set V)) : Prop :=
  -- Define condition that no segments share a common line
  sorry

theorem self_intersections_cannot_divide_half {V : Type*}
  (points : list V) (segments : list (set V))
  (h1 : is_closed_self_intersecting_polygonal_path points segments)
  (h2 : intersects_once_per_segment segments)
  (h3 : two_segments_per_intersection points segments)
  (h4 : no_intersection_at_vertices points segments)
  (h5 : no_common_segments segments) :
  ¬ (∀ p ∈ points, ∀ s₁ s₂ ∈ segments, divides_in_half p s₁ s₂) :=
begin
  sorry
end

end self_intersections_cannot_divide_half_l3_3769


namespace solve_problem_l3_3631

def f (x : ℝ) : ℝ :=
if x > 8 then x^3 - 1
else if x >= -8 then 3 * x + 2
else 4

theorem solve_problem :
  f (-9) + f (0) + f (10) = 1005 :=
by
  sorry

end solve_problem_l3_3631


namespace train_length_approx_200_l3_3795

noncomputable def train_length (speed_kmph : ℕ) (time_sec : ℕ) : ℝ :=
  (speed_kmph * 1000) / 3600 * time_sec

theorem train_length_approx_200
  (speed_kmph : ℕ)
  (time_sec : ℕ)
  (h_speed : speed_kmph = 120)
  (h_time : time_sec = 6) :
  train_length speed_kmph time_sec ≈ 200 := 
by sorry

end train_length_approx_200_l3_3795


namespace fraction_of_dark_tiles_is_correct_l3_3413

def num_tiles_in_block : ℕ := 64
def num_dark_tiles : ℕ := 18
def expected_fraction_dark_tiles : ℚ := 9 / 32

theorem fraction_of_dark_tiles_is_correct :
  (num_dark_tiles : ℚ) / num_tiles_in_block = expected_fraction_dark_tiles := by
sorry

end fraction_of_dark_tiles_is_correct_l3_3413


namespace sport_vs_std_ratio_comparison_l3_3604

/-- Define the ratios for the standard formulation. -/
def std_flavor_syrup_ratio := 1 / 12
def std_flavor_water_ratio := 1 / 30

/-- Define the conditions for the sport formulation. -/
def sport_water := 15 -- ounces of water in the sport formulation
def sport_syrup := 1 -- ounce of corn syrup in the sport formulation

/-- The ratio of flavoring to water in the sport formulation is half that of the standard formulation. -/
def sport_flavor_water_ratio := std_flavor_water_ratio / 2

/-- Calculate the amount of flavoring in the sport formulation. -/
def sport_flavor := sport_water * sport_flavor_water_ratio

/-- The ratio of flavoring to corn syrup in the sport formulation. -/
def sport_flavor_syrup_ratio := sport_flavor / sport_syrup

/-- The proof problem statement. -/
theorem sport_vs_std_ratio_comparison : sport_flavor_syrup_ratio = 3 * std_flavor_syrup_ratio := 
by
  -- proof would go here
  sorry

end sport_vs_std_ratio_comparison_l3_3604


namespace range_of_a_l3_3886

noncomputable def function_monotonicity (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x ∈ Ico (-1 : ℝ) 2, (has_deriv_at f (2 * x * a + 2) x) → (2 * x * a + 2) ≥ 0

theorem range_of_a : 
  (∀ f, f = (λ x : ℝ, a * x^2 + 2 * x - 2 * a) → function_monotonicity f a) → 
  a ∈ Icc (-1 / 2 : ℝ) 1 :=
sorry

end range_of_a_l3_3886


namespace position_relationship_correct_l3_3735

noncomputable def problem_data :=
  (((2:ℝ), 3, -1), ((-2:ℝ), -3, 1), ((1:ℝ), -1, 2), ((6:ℝ), 4, -1), 
   ((2:ℝ), 2, -1), ((-3:ℝ), 4, 2), ((0:ℝ), 3, 0), ((0:ℝ), -5, 0))

theorem position_relationship_correct :
  let ⟨a, b, a2, u2, u3, v3, a4, u4⟩ := problem_data in
  (a = ((-1):ℝ) • b ∧
  (∀ k : ℝ, a2 ≠ k • u2) ∧
  (u3 ⬝ v3 = 0 ∧
  ¬ (a4 = (0:ℝ) • u4 ∧ a4 = (-3/5:ℝ) • u4 ))) :=
sorry

end position_relationship_correct_l3_3735


namespace find_f_1789_l3_3031

-- Given conditions as definitions

def f : ℕ → ℕ := sorry
axiom f_a (n : ℕ) (hn : n > 0) : f(f(n)) = 4 * n + 9
axiom f_b (k : ℕ) : f(2^k) = 2^(k+1) + 3

-- The theorem to prove f(1789) = 3581 given the conditions
theorem find_f_1789 : f(1789) = 3581 := sorry

end find_f_1789_l3_3031


namespace jackie_more_apples_oranges_l3_3797

-- Definitions of initial conditions
def adams_apples : ℕ := 25
def adams_oranges : ℕ := 34
def jackies_apples : ℕ := 43
def jackies_oranges : ℕ := 29

-- The proof statement
theorem jackie_more_apples_oranges :
  (jackies_apples - adams_apples) + (jackies_oranges - adams_oranges) = 13 :=
by
  sorry

end jackie_more_apples_oranges_l3_3797


namespace maximum_sequence_length_y_l3_3843

theorem maximum_sequence_length_y : 
  ∃ y : ℕ, 927 < y ∧ y < 928 ∧ 
  (∀ n, ∃ b : ℕ → ℤ, b 1 = 1500 ∧ 
                     b 2 = y ∧ 
                     (∀ k, b (k + 2) = b (k + 1) - b k) ∧ 
                     (∃ m, b m < 0 ∧ ∀ l < m, b l ≥ 0)) → y = 927 :=
begin
  sorry
end

end maximum_sequence_length_y_l3_3843


namespace solution_set_inequality_l3_3919

variable (a x : ℝ)

-- Conditions
theorem solution_set_inequality (h₀ : 0 < a) (h₁ : a < 1) :
  ((a - x) * (x - (1 / a)) > 0) ↔ (a < x ∧ x < 1 / a) := 
by 
  sorry

end solution_set_inequality_l3_3919


namespace clock_angle_at_5_50_l3_3442

theorem clock_angle_at_5_50 :
    let h := 5
    let m := 50
    (abs ((60 * h - 11 * m) / 2)) = 125 :=
by
  let h := 5
  let m := 50
  have angle_formula := abs ((60 * h - 11 * m) / 2)
  calc
    angle_formula
        = abs ((60 * 5 - 11 * 50) / 2) : by rw [h, m]
    ... = abs ((300 - 550) / 2) : by norm_num
    ... = abs ((-250) / 2) : by norm_num
    ... = abs (-125) : by norm_num
    ... = 125 : by norm_num

end clock_angle_at_5_50_l3_3442


namespace am_gm_inequality_l3_3625

open Real

theorem am_gm_inequality {a : ℕ → ℝ} {n : ℕ} (hpos : ∀ i, 0 < a i) (hprod : ∏ i in finset.range n, a i = 1) :
  (∏ i in finset.range n, (2 + a i)) ≥ 3 ^ n :=
sorry

end am_gm_inequality_l3_3625


namespace value_of_a_l3_3534

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + x^2

-- Define the tangent line condition at point (1,1) parallel to line x+y=0
theorem value_of_a (a : ℝ) :
  (∀ x : ℝ, f a 1 = 1 ∧ (2 + a / 1 = -1)) → a = -3 :=
begin
  sorry
end

end value_of_a_l3_3534


namespace compare_a_b_c_l3_3872

noncomputable def a : ℝ := 2^(-1/3)
noncomputable def b : ℝ := Real.log 1 / (Real.log 2)^3
noncomputable def c : ℝ := Real.log 3 / Real.log (1/2)

theorem compare_a_b_c : c > a ∧ a > b := by
  -- The following sorry is a placeholder for the proof.
  sorry

end compare_a_b_c_l3_3872


namespace replacement_digit_is_9_l3_3933

theorem replacement_digit_is_9 :
  ∀ (d : ℕ), 
    (let num_sixes_units := 10 in
     let num_sixes_tens := 10 in
     let total_difference := num_sixes_units * (d - 6) + num_sixes_tens * 10 * (d - 6) in
     total_difference = 330) →
    d = 9 :=
by
  intros d h
  -- definitions and equations derived from conditions
  let num_sixes_units := 10
  let num_sixes_tens := 10
  let total_difference := num_sixes_units * (d - 6) + num_sixes_tens * 10 * (d - 6)
  -- proof would go here
  sorry

end replacement_digit_is_9_l3_3933


namespace degree_of_difficulty_approx_l3_3204

-- Definitions for the problem
def scores : List ℝ := [7.5, 8.0, 9.0, 6.0, 8.8]
def point_value : ℝ := 77.76

-- Define a function for processing the scores and computing the difficulty
def degree_of_difficulty := 
  let highest := scores.maximum
  let lowest := scores.minimum
  let remaining_scores := scores.erase highest |>.erase lowest
  point_value / remaining_scores.sum

-- The statement to prove the degree of difficulty is approximately 3.2
theorem degree_of_difficulty_approx : |degree_of_difficulty- 3.2| < 0.01 :=
by
  sorry

end degree_of_difficulty_approx_l3_3204


namespace no_prime_divisible_by_57_l3_3563

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. --/
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Given that 57 is equal to 3 times 19.--/
theorem no_prime_divisible_by_57 : ∀ p : ℕ, is_prime p → ¬ (57 ∣ p) :=
by
  sorry

end no_prime_divisible_by_57_l3_3563


namespace perfect_squares_represented_as_diff_of_consecutive_cubes_l3_3917

theorem perfect_squares_represented_as_diff_of_consecutive_cubes : ∃ (count : ℕ), 
  count = 40 ∧ 
  ∀ n : ℕ, 
  (∃ a : ℕ, a^2 = ( ( n + 1 )^3 - n^3 ) ∧ a^2 < 20000) → count = 40 := by 
sorry

end perfect_squares_represented_as_diff_of_consecutive_cubes_l3_3917


namespace number_of_outcomes_for_champions_l3_3399

def num_events : ℕ := 3
def num_competitors : ℕ := 6
def total_possible_outcomes : ℕ := num_competitors ^ num_events

theorem number_of_outcomes_for_champions :
  total_possible_outcomes = 216 :=
by
  sorry

end number_of_outcomes_for_champions_l3_3399


namespace similarity_proportion_l3_3759

theorem similarity_proportion (YZ VZ WZ : ℝ) (hYZ : YZ = 30) (hVZ : VZ = 18) (hWZ : WZ = 15) (h_similarity : ΔXYZ ∼ ΔWZV) : ZV = 16.2 :=
by
  sorry

end similarity_proportion_l3_3759


namespace solve_for_w_l3_3378

theorem solve_for_w (w : ℝ) : (2 : ℝ)^(2 * w) = (8 : ℝ)^(w - 4) → w = 12 := by
  sorry

end solve_for_w_l3_3378


namespace no_exact_cover_l3_3830

theorem no_exact_cover (large_w : ℕ) (large_h : ℕ) (small_w : ℕ) (small_h : ℕ) (n : ℕ) :
  large_w = 13 → large_h = 7 → small_w = 2 → small_h = 3 → n = 15 →
  ¬ (small_w * small_h * n = large_w * large_h) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end no_exact_cover_l3_3830


namespace quadrant_of_z_l3_3304

theorem quadrant_of_z (z : ℂ) (h : (1 + 2 * Complex.i) * z = Complex.abs (1 + 3 * Complex.i) ^ 2) : 
  let p := (z.re, z.im) in p.snd < 0 ∧ p.fst > 0 :=
by
  -- proof goes here
  sorry

end quadrant_of_z_l3_3304


namespace filled_circles_2006_l3_3431

-- Definition of the repeating sequence
def sequence : List Char := ['○', '●', '○', '○', '●', '○', '○', '○', '●', '○', '○', '○', '○', '●', '○', '○', '○', '○', '○', '●']

-- Function to get n-th circle in the repeated sequence
def get_nth_circle (n : Nat) : Char :=
  sequence[(n % sequence.length)]

-- Function to count filled circles in the first n circles
def count_filled_circles (n : Nat) : Nat :=
  ((List.range n).map get_nth_circle).filter (fun c => c = '●').length

-- Main statement: There are 61 filled circles among the first 2006 circles
theorem filled_circles_2006 : count_filled_circles 2006 = 61 := 
  sorry

end filled_circles_2006_l3_3431


namespace Iris_total_spent_l3_3610

theorem Iris_total_spent :
  let jackets := 3
  let cost_per_jacket := 10
  let shorts := 2
  let cost_per_short := 6
  let pants := 4
  let cost_per_pant := 12
  jackets * cost_per_jacket + shorts * cost_per_short + pants * cost_per_pant = 90 := by
  sorry

end Iris_total_spent_l3_3610


namespace value_at_v3_using_horner_l3_3813

noncomputable def f (x : ℝ) : ℝ := 2 * x^5 - 3 * x^3 + 2 * x^2 + x - 3

theorem value_at_v3_using_horner (x : ℝ) (h : x = 2) : 
  let v0 := 2,
      v1 := v0 * x + 0,
      v2 := v1 * x - 3,
      v3 := v2 * x + 2,
      v4 := v3 * x + 1,
      v5 := v4 * x - 3 in
  v3 = 12 :=
by
  sorry

end value_at_v3_using_horner_l3_3813


namespace total_copies_correct_l3_3373

-- Definitions based on given conditions
def rate_A : ℝ := 100 / 8
def rate_B : ℝ := 150 / 10
def rate_C : ℝ := 200 / 12
def rate_D : ℝ := 250 / 15

def time : ℝ := 40

-- Number of copies produced by each machine in 40 minutes
def copies_A : ℝ := rate_A * time
def copies_B : ℝ := rate_B * time
def copies_C : ℝ := rate_C * time
def copies_D : ℝ := rate_D * time

-- Total number of copies produced by all machines in 40 minutes
def total_copies : ℝ := copies_A + copies_B + copies_C + copies_D

-- The proof problem statement
theorem total_copies_correct : total_copies = 2434 := by
  -- Note: The proof is omitted, only the statement is written according to the procedure
  sorry

end total_copies_correct_l3_3373


namespace circle_equation_l3_3161

theorem circle_equation (center : ℝ × ℝ) (chord_length : ℝ) (line : ℝ → ℝ → ℝ) :
  center = (0, -2) → chord_length = 4 * Real.sqrt 5 → 
  line = (λ x y, 2 * x - y + 3) →
  ∃ r : ℝ, x^2 + (y + 2)^2 = r^2 :=
by
  intros
  use 5
  sorry

end circle_equation_l3_3161


namespace min_number_of_rounds_l3_3395

theorem min_number_of_rounds (n : ℕ) (hn : n = 510) :
  ∃ r : ℕ, r = 9 ∧
    (∀ (participants : ℕ), participants = n → 
      (∀ (points : ℕ → ℕ), 
        (∀ p, p ≥ 0) ∧ 
        (∀ round, 
          ∃ matches : ℕ, 
            (∀ match_index, match_index < matches → 
              (abs (points match_index.succ - points match_index) ≤ 1)) ∧ 
                (participants / 2 ≤ matches →
                  (points match_index = participants.succ + 1))) →
          ((∀ round_number, round_number ≤ r → 
            ∃ final_leader_points, 
              (final_leader_points > 0) ∧ 
              (∀ other_points, other_points ≠ final_leader_points → 
                other_points < final_leader_points))))) :=
by
  intro n hn
  use 9
  split
  · rfl
  · intros participants hparticipants points hpoints round hround
    sorry

end min_number_of_rounds_l3_3395


namespace compute_x2_y2_l3_3257

theorem compute_x2_y2 (x y : ℝ) (h1 : 1 < x) (h2 : 1 < y)
  (h3 : (Real.log x / Real.log 4)^3 + (Real.log y / Real.log 5)^3 + 27 = 9 * (Real.log x / Real.log 4) * (Real.log y / Real.log 5)) :
  x^2 + y^2 = 189 := 
by sorry

end compute_x2_y2_l3_3257


namespace tangent_angle_double_l3_3383

-- We assume that the circles, tangents, and points exist as described
variables (Ω1 Ω2 : Circle) (A P X Y Q R : Point)

-- Conditions:
-- Ω1 is internally tangent to Ω2 at point A
-- P is a point on Ω2
-- Tangents from P to Ω1 pass through X and Y on Ω1
-- Tangents intersect Ω2 again at Q and R respectively

-- Definitions of angles on the circles based on the problem statement
def angle_XAY : Angle := sorry -- Replace with the actual definition of ∠XAY
def angle_QAR : Angle := sorry -- Replace with the actual definition of ∠QAR

-- The theorem we need to prove
theorem tangent_angle_double :
  ∠ QAR = 2 * ∠ XAY :=
sorry

end tangent_angle_double_l3_3383


namespace position_relationship_correct_l3_3738

noncomputable def problem_data :=
  (((2:ℝ), 3, -1), ((-2:ℝ), -3, 1), ((1:ℝ), -1, 2), ((6:ℝ), 4, -1), 
   ((2:ℝ), 2, -1), ((-3:ℝ), 4, 2), ((0:ℝ), 3, 0), ((0:ℝ), -5, 0))

theorem position_relationship_correct :
  let ⟨a, b, a2, u2, u3, v3, a4, u4⟩ := problem_data in
  (a = ((-1):ℝ) • b ∧
  (∀ k : ℝ, a2 ≠ k • u2) ∧
  (u3 ⬝ v3 = 0 ∧
  ¬ (a4 = (0:ℝ) • u4 ∧ a4 = (-3/5:ℝ) • u4 ))) :=
sorry

end position_relationship_correct_l3_3738


namespace amy_quadrilateral_rod_count_l3_3800

/-- Given that Amy has 40 thin rods, each of a different integer length ranging from 1 cm to 40 cm,
and she has placed rods of 4 cm, 8 cm, and 16 cm on the table,
proves that there are exactly 23 remaining rods that she can choose as the fourth rod
such that it is possible to form a quadrilateral with positive area. -/
theorem amy_quadrilateral_rod_count :
  ∀ (rods : Finset ℕ),
    rods = (Finset.range 41).filter (λ x, 1 ≤ x) →
    ∀ {table_rods : Finset ℕ},
    table_rods = {4, 8, 16} →
    ∀ {valid_rods : Finset ℕ},
    valid_rods = (rods \ table_rods).filter (λ x, 5 ≤ x ∧ x < 28) →
    valid_rods.card = 23 :=
by
  intros rods rods_def table_rods table_rods_def valid_rods valid_rods_def
  rw [rods_def, table_rods_def, valid_rods_def]
  sorry

end amy_quadrilateral_rod_count_l3_3800


namespace part1_part2_l3_3903

noncomputable def f (x a : ℝ) : ℝ := abs (x + 2 * a) + abs (x - 1)

noncomputable def g (a : ℝ) : ℝ := abs ((1 : ℝ) / a + 2 * a) + abs ((1 : ℝ) / a - 1)

theorem part1 (x : ℝ) : f x 1 ≤ 5 ↔ -3 ≤ x ∧ x ≤ 2 := by
  sorry

theorem part2 (a : ℝ) (h : a ≠ 0) : g a ≤ 4 ↔ (1 / 2) ≤ a ∧ a ≤ (3 / 2) := by
  sorry

end part1_part2_l3_3903


namespace solve_for_x_l3_3878

noncomputable def avg (a b : ℝ) := (a + b) / 2

noncomputable def B (t : List ℝ) : List ℝ :=
  match t with
  | [a, b, c, d, e] => [avg a b, avg b c, avg c d, avg d e]
  | _ => []

noncomputable def B_iter (m : ℕ) (t : List ℝ) : List ℝ :=
  match m with
  | 0 => t
  | k + 1 => B (B_iter k t)

theorem solve_for_x (x : ℝ) (h1 : 0 < x) (h2 : B_iter 4 [1, x, x^2, x^3, x^4] = [1/4]) :
  x = Real.sqrt 2 - 1 :=
sorry

end solve_for_x_l3_3878


namespace evaluate_expression_l3_3842

theorem evaluate_expression (a : ℚ) (h : a = 4 / 3) : (6 * a^2 - 8 * a + 3) * (3 * a - 4) = 0 :=
by
  rw [h]
  sorry

end evaluate_expression_l3_3842


namespace xiaohong_height_l3_3375

theorem xiaohong_height 
  (father_height_cm : ℕ)
  (height_difference_dm : ℕ)
  (father_height : father_height_cm = 170)
  (height_difference : height_difference_dm = 4) :
  ∃ xiaohong_height_cm : ℕ, xiaohong_height_cm + height_difference_dm * 10 = father_height_cm :=
by
  use 130
  sorry

end xiaohong_height_l3_3375


namespace sum_partial_fraction_l3_3821

theorem sum_partial_fraction :
  (∑ n in Finset.range 500, (1 : ℚ) / (n + 1)^2 + 2*(n + 1)) = 1499 / 2008 :=
by
  sorry

end sum_partial_fraction_l3_3821


namespace apothem_and_radius_limit_l3_3449
open Real

theorem apothem_and_radius_limit (h r : ℕ → ℝ) (h₀ : ∀ p, h (p+1) = (h p + r p) / 2)
  (r₀ : ∀ p, r (p+1) = sqrt (h (p+1) * r p)) :
  ∃ l, tendsto h at_top (𝓝 l) ∧ tendsto r at_top (𝓝 l) :=
sorry

end apothem_and_radius_limit_l3_3449


namespace problem_statement_l3_3136

def g (n : ℕ) : ℝ := Real.log (n ^ 3) / Real.log 3003

theorem problem_statement : g 7 + g 11 + g 13 = 9 / 4 := sorry

end problem_statement_l3_3136


namespace figure_property_invariant_l3_3313

variable {m n L : ℕ}
variable (F : Type)
variable (numbers : List ℝ)
variable (P : ℕ → ℕ → ℝ)
variable (deviation θ : List ℝ)

theorem figure_property_invariant (hF_invariant : ∀ (x : ℝ) i j, 0 ≤ i → i < m * n → 0 ≤ j → j < L → 
  deviation j = (θ j - x * P i j) → 
  let new_deviation := (θ j - x * P i j) in
  sum (new_deviation.map (λ x => x^2)) = sum (θ.map (λ x => x^2)) - 2 * x * sum (deviation.map (λ d => d * P i j)) + x^2 * sum (λ i, P i j^2)) :
  true := sorry

end figure_property_invariant_l3_3313


namespace shortest_distance_plane_ellipsoid_l3_3022

-- Defining the parameters and conditions
variables {A B C a b c : ℝ}

-- Definitions of h and m
def h := (A^2 + B^2 + C^2)⁻¹ ^ (1 / 2)
def m := (a^2 * A^2 + b^2 * B^2 + c^2 * C^2) ^ (1 / 2)

-- Main theorem statement
theorem shortest_distance_plane_ellipsoid (h : ℝ) (m : ℝ) :
    (h = (A^2 + B^2 + C^2)⁻¹ ^ (1 / 2)) →
    (m = (a^2 * A^2 + b^2 * B^2 + c^2 * C^2) ^ (1 / 2)) →
    ((h * (1 - m) if m < 1 else 0) = 
    if m < 1 then h * (1 - m) else 0) :=
by sorry

end shortest_distance_plane_ellipsoid_l3_3022


namespace four_letter_words_count_l3_3612

theorem four_letter_words_count :
  (∑ x in (finset.range 26).image (λ i, (prod.mk i i)), 1) *
  (∑ v in (finset.filter (λ c : ℕ, c = 'a' ∨ c = 'e' ∨ c = 'i' ∨ c = 'o' ∨ c = 'u') 
         (finset.range 26)), 1) *
  (∑ x in (finset.range 26), 1) = 3380 :=
by
  sorry

end four_letter_words_count_l3_3612


namespace total_charcoal_is_212_l3_3962

def charcoal_first_batch_ratio : ℝ := 2 / 30
def charcoal_second_batch_ratio : ℝ := 3 / 50
def charcoal_third_batch_ratio : ℝ := 4 / 80

def water_first_batch : ℝ := 900
def water_second_batch : ℝ := 1200
def water_third_batch : ℝ := 1600

def charcoal_needed_first_batch : ℝ := charcoal_first_batch_ratio * water_first_batch
def charcoal_needed_second_batch : ℝ := charcoal_second_batch_ratio * water_second_batch
def charcoal_needed_third_batch : ℝ := charcoal_third_batch_ratio * water_third_batch

def total_charcoal_needed : ℝ := charcoal_needed_first_batch + charcoal_needed_second_batch + charcoal_needed_third_batch

theorem total_charcoal_is_212 :
  total_charcoal_needed = 212 := by
  sorry

end total_charcoal_is_212_l3_3962


namespace regression_estimate_l3_3175

theorem regression_estimate :
  ∀ (x y : ℝ), (y = 0.50 * x - 0.81) → x = 25 → y = 11.69 :=
by
  intros x y h_eq h_x_val
  sorry

end regression_estimate_l3_3175


namespace lowest_test_score_dropped_l3_3242

theorem lowest_test_score_dropped (S L : ℕ)
  (h1 : S = 5 * 42) 
  (h2 : S - L = 4 * 48) : 
  L = 18 :=
by
  sorry

end lowest_test_score_dropped_l3_3242


namespace problem_l3_3171

def f (x : ℝ) : ℝ :=
  4 * sin x * cos (x + π / 3) + 4 * sqrt 3 * sin x ^ 2 - sqrt 3

theorem problem (x : ℝ): 
  f (π / 3) = sqrt 3 ∧
  (∀ k : ℤ, f (x) = 2 * sin (2 * x - π / 3) → x = k * (π / 2) + 5 * (π / 12)) ∧
  (∀ x ∈ set.Icc (-π / 4) (π / 3), f x ≤ sqrt 3 ∧ f x ≥ -2) :=
by
  sorry

end problem_l3_3171


namespace line_length_limit_is_correct_l3_3417

noncomputable def line_length_limit : ℝ :=
  2 + (∑' n, (1/(3^(n+1)) * (1 + (sqrt 3))))

theorem line_length_limit_is_correct : line_length_limit = (5 + sqrt 3) / 2 :=
by
  sorry

end line_length_limit_is_correct_l3_3417


namespace sequence_third_term_l3_3682

theorem sequence_third_term (n : ℕ) (a : ℕ → ℤ) (h : ∀ n, a n = 3 * n - 5) : a 3 = 4 := by
  sorry

end sequence_third_term_l3_3682


namespace sign_of_slope_equals_sign_of_correlation_l3_3668

theorem sign_of_slope_equals_sign_of_correlation {x y : Type} [LinearOrder x] [LinearOrder y]
  (r b : ℝ) (lin_rel : ∀ x : ℝ, y = a + b * x)
  (corr_coeff : ℝ) (h_corr : r = corr_coeff) :
  (b > 0) ↔ (r > 0) :=
sorry

end sign_of_slope_equals_sign_of_correlation_l3_3668


namespace Ganesh_avg_speed_l3_3038

theorem Ganesh_avg_speed (D : ℝ) : 
  (∃ (V : ℝ), (39.6 = (2 * D) / ((D / 44) + (D / V))) ∧ V = 36) :=
by
  sorry

end Ganesh_avg_speed_l3_3038


namespace geo_seq_b_sum_bound_l3_3548

-- Conditions
def a : ℕ → ℤ
| 0     := 0
| 1     := -1
| (n+1) := (3 * n + 3) * a n + 4 * n + 6 / n

-- Proving Question 1
theorem geo_seq (n : ℕ) (h : n ≥ 1) :
  (∃ r, ∀ n, (a n + 2) / n = r^(n - 1)) := sorry

-- Proving Question 2
def b (n : ℕ) : ℚ := (3^(n-1) : ℚ) / (a n + 2)

theorem b_sum_bound (n : ℕ) (h : n ≥ 2) :
  (∑ i in (finset.range (2 * n + 1)).filter (λ i, i ≥ n + 1), b i) < 4/5 - 1 / (2 * n + 1) := sorry

end geo_seq_b_sum_bound_l3_3548


namespace permutation_count_l3_3119

theorem permutation_count :
  let perms := {p : (Fin 7) → (Fin 7) // Function.Bijective p}
  let valid := 
    ∀ p : perms, 
    (∏ i in Finset.univ, (p.val i + Fin i + 1 : ℚ) / 2) > Nat.factorial 7
  let total_valid := (perms : Set _).filter valid)
  perms.to_finset.card - 1 = 5039 :=
  by
    sorry

end permutation_count_l3_3119


namespace volume_ratio_inscribed_circumscribed_sphere_regular_tetrahedron_l3_3687

theorem volume_ratio_inscribed_circumscribed_sphere_regular_tetrahedron (R r : ℝ) (h : r = R / 3) : 
  (4/3 * π * r^3) / (4/3 * π * R^3) = 1 / 27 :=
by
  sorry

end volume_ratio_inscribed_circumscribed_sphere_regular_tetrahedron_l3_3687


namespace quadrilateral_is_parallelogram_l3_3012

-- Define the points A, B, C, D, M, N, P with appropriate assumptions
variables {A B C D M N P : Type} 
variables [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] 
          [InnerProductSpace ℝ C] [InnerProductSpace ℝ D] 
          [InnerProductSpace ℝ M] [InnerProductSpace ℝ N]
          [InnerProductSpace ℝ P]

-- Define two sides of the quadrilateral as parallel
variable (par : Parallel A B A C ∨ Parallel B C C D)

-- Define M and N as midpoints of sides BC and CD, respectively
variable (midM : midpoint B C = M)
variable (midN : midpoint C D = N)

-- Define P as the intersection point of AN and DM
variable (intersectP : Line A N ∩ Line D M = P)

-- Define the length relationship AP = 4PN
variable (lenAP_4PN : dist A P = 4 * dist P N)

theorem quadrilateral_is_parallelogram :
  parallelogram A B C D :=
sorry

end quadrilateral_is_parallelogram_l3_3012


namespace correct_propositions_l3_3498

variables {m n : Line} {α β γ : Plane}

def proposition_1 := (m ∥ α ∧ m ⊥ n) → n ⊥ α

def proposition_2 := (m ⊥ α ∧ m ⊥ n) → n ∥ α

def proposition_3 := (α ⊥ β ∧ γ ⊥ β) → α ∥ γ

def proposition_4 := (m ⊥ α ∧ m ∥ n ∧ n ⊆ β) → α ⊥ β

theorem correct_propositions : proposition_2 ∧ proposition_4 :=
by
  split,
  -- Proof for proposition_2
  sorry,
  -- Proof for proposition_4
  sorry

end correct_propositions_l3_3498


namespace spadesuit_evaluation_l3_3104

def spadesuit (a b : ℝ) : ℝ := abs (a - b)

theorem spadesuit_evaluation : spadesuit 1.5 (spadesuit 2.5 (spadesuit 4.5 6)) = 0.5 :=
by
  sorry

end spadesuit_evaluation_l3_3104


namespace smallest_difference_l3_3839

theorem smallest_difference :
  ∃ (n m : ℕ), 
    let digits := {3, 5, 6, 7, 8} in
    (∀ d ∈ digits, d ∈ finset.digits n ∨ d ∈ finset.digits m) ∧
    (∀ d ∈ digits, (finset.digits n).count d + (finset.digits m).count d = 1) ∧
    100 ≤ n ∧ n ≤ 999 ∧ 10 ≤ m ∧ m ≤ 99 ∧
    (n - m) = 269 :=
  sorry

end smallest_difference_l3_3839


namespace smallest_hw_assignments_needed_l3_3776

theorem smallest_hw_assignments_needed : 
  (∑ i in Finset.range (30 + 1), (Int.ceil (i / 3))) = 165 := by
  sorry

end smallest_hw_assignments_needed_l3_3776


namespace ratio_of_weights_l3_3290

variable (x : ℝ)

-- Conditions as definitions in Lean 4
def seth_loss : ℝ := 17.5
def jerome_loss : ℝ := 17.5 * x
def veronica_loss : ℝ := 17.5 + 1.5 -- 19 pounds
def total_loss : ℝ := seth_loss + jerome_loss x + veronica_loss

-- Statement to prove
theorem ratio_of_weights (h : total_loss x = 89) : jerome_loss x / seth_loss = 3 :=
by sorry

end ratio_of_weights_l3_3290


namespace remaining_battery_life_l3_3798

theorem remaining_battery_life (inactive_hours : ℕ) (active_hours : ℤ) (active_use : ℚ) (total_time : ℚ) :
  inactive_hours = 18 → 
  active_use = 2 → 
  active_hours = 1 / 2 → 
  total_time = 6 → 
  let remaining_life := total_time - active_hours,
      inactive_consumption := remaining_life / inactive_hours,
      active_consumption := active_hours / active_use,
      total_consumption := inactive_consumption + active_consumption,
      battery_remaining := 1 - total_consumption
  in battery_remaining / (1 / inactive_hours) = 8 :=
begin
  intros h_inactive h_activeuse h_activehours h_totaltime,
  let remaining_life := 6 - 1 / 2,
  let inactive_consumption := remaining_life / 18,
  let active_consumption := (1 / 2) / 2,
  let total_consumption := inactive_consumption + active_consumption,
  let battery_remaining := 1 - total_consumption,
  suffices : battery_remaining / (1 / 18) = 8,
  from this,
  sorry
end

end remaining_battery_life_l3_3798


namespace min_even_integers_zero_l3_3703

theorem min_even_integers_zero (x y a b m n : ℤ)
(h1 : x + y = 28) 
(h2 : x + y + a + b = 46) 
(h3 : x + y + a + b + m + n = 64) : 
∃ e, e = 0 :=
by {
  -- The conditions assure the sums of pairs are even including x, y, a, b, m, n.
  sorry
}

end min_even_integers_zero_l3_3703


namespace last_triangle_perimeter_l3_3268

theorem last_triangle_perimeter (a b c : ℕ) (h1 : a = 13) (h2 : b = 14) (h3 : c = 15) :
  let T1 := (a, b, c),
      AD := (b + c - a) / 2,
      BE := (a + c - b) / 2,
      CF := (a + b - c) / 2
      T2 := (AD, BE, CF)
  in (fst T2) + (fst (snd T2)) + (snd (snd T2)) = 14 :=
begin
  let AD := (b + c - a) / 2,
  let BE := (a + c - b) / 2,
  let CF := (a + b - c) / 2,
  have h_AD : AD = 8, by { rw [h1, h2, h3], norm_num },
  have h_BE : BE = 7, by { rw [h1, h2, h3], norm_num },
  have h_CF : CF = 6, by { rw [h1, h2, h3], norm_num },
  rw [h_AD, h_BE, h_CF],
  norm_num,
sorry
end

end last_triangle_perimeter_l3_3268


namespace position_relationship_correct_l3_3739

noncomputable def problem_data :=
  (((2:ℝ), 3, -1), ((-2:ℝ), -3, 1), ((1:ℝ), -1, 2), ((6:ℝ), 4, -1), 
   ((2:ℝ), 2, -1), ((-3:ℝ), 4, 2), ((0:ℝ), 3, 0), ((0:ℝ), -5, 0))

theorem position_relationship_correct :
  let ⟨a, b, a2, u2, u3, v3, a4, u4⟩ := problem_data in
  (a = ((-1):ℝ) • b ∧
  (∀ k : ℝ, a2 ≠ k • u2) ∧
  (u3 ⬝ v3 = 0 ∧
  ¬ (a4 = (0:ℝ) • u4 ∧ a4 = (-3/5:ℝ) • u4 ))) :=
sorry

end position_relationship_correct_l3_3739


namespace inequality_four_a_cubed_sub_l3_3877

theorem inequality_four_a_cubed_sub (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  4 * a^3 * (a - b) ≥ a^4 - b^4 :=
sorry

end inequality_four_a_cubed_sub_l3_3877


namespace second_polygon_sides_l3_3011

-- Definitions and given conditions:
def Polygon1 (s : ℝ) := 50 * 3 * s  -- Perimeter of the first polygon: 50 sides, each side length 3s
def Polygon2 (n : ℕ) (s : ℝ) := n * s  -- Perimeter of the second polygon: n sides, each side length s

-- Theorem statement to prove the second polygon has 150 sides
theorem second_polygon_sides (s : ℝ) (h : Polygon1 s = Polygon2 150 s) :
  ∃ n : ℕ, Polygon2 n s = Polygon1 s ∧ n = 150 :=
by
  use 150
  split
  case left =>
    exact h
  case right =>
    rfl


end second_polygon_sides_l3_3011


namespace sum_of_tangency_points_l3_3310

theorem sum_of_tangency_points :
  let E := λ x y, y^2 = x^3 + 1
  let C := λ x y, (x - 4)^2 + y^2 = c (where c ≥ 0)
  let dy_dx_E := λ x y, 2 * y * (derivative y x) = 3 * x^2
  let dy_dx_C := λ x y, 2 * (x - 4) + 2 * y * (derivative y x) = 0
  let F := factorize (3 * x^2 + x - 4)
  in (1:ℝ) + -4 / 3 == 1
:= sorry

end sum_of_tangency_points_l3_3310


namespace positive_divisors_30030_l3_3122

theorem positive_divisors_30030 : 
  let n := 30030
  let factorization := [(2, 1), (3, 1), (5, 1), (7, 1), (11, 1), (13, 1)]
  number_of_divisors n factorization = 64 := 
by 
  sorry

end positive_divisors_30030_l3_3122


namespace length_of_QS_l3_3978

-- Define the geometric elements and conditions
variable (P Q R S : Type) [RightTriangle P Q R]
variable (circle : Circle (segment Q R))
variable (area_PQR : Real := 98)
variable (PR_length : segment P R = 14)

-- Define the length of QS
def length_QS : Real := 14

-- State the theorem
theorem length_of_QS (h1 : RightTriangle P Q R) (h2 : Circle (segment Q R)) (h3 : area_of_triangle P Q R = 98) (h4 : segment_length P R = 14) : 
  length_of_segment Q S = 14 :=
sorry

end length_of_QS_l3_3978


namespace sequence_properties_l3_3974

theorem sequence_properties (S : ℕ → ℕ) (a : ℕ → ℕ) (T : ℕ → ℕ) :
  (a 1 ≠ 0) →
  (∀ n : ℕ, n > 0 → 2 * a n - a 1 = S 1 * S n) →
  (∀ n : ℕ, S n = finset.sum (finset.range n) a) →
  a 1 = 1 ∧ (∀ n, n > 0 → a n = 2^(n-1))
  ∧ (∀ n, T n = finset.sum (finset.range n) (λ i, (i + 1) * a (i+1)))
  ∧ (∀ n, T n = (n-1) * 2^n + 1) :=
by
  -- Proof is omitted
  sorry

end sequence_properties_l3_3974


namespace total_flower_beds_l3_3653

theorem total_flower_beds :
  let section1_seeds := 470
  let section2_seeds := 320
  let section3_seeds := 210
  let seeds_per_bed1 := 10
  let seeds_per_bed2 := 10
  let seeds_per_bed3 := 8
  let beds1 := section1_seeds / seeds_per_bed1
  let beds2 := section2_seeds / seeds_per_bed2
  let beds3 := section3_seeds / seeds_per_bed3
  let total_beds := beds1 + beds2 + beds3
  ⌊total_beds⌋ = 47 + 32 + 26 →
  47 + 32 + 26 = 105 :=
by
  intros
  have h_beds1 : beds1 = 47 := by norm_num
  have h_beds2 : beds2 = 32 := by norm_num
  have h_beds3 : beds3 = 210 / 8 := by norm_num
  have h_total_beds : total_beds = beds1 + beds2 + beds3 := rfl
  rw [h_beds1, h_beds2, ←h_beds3] at h_total_beds
  have h_floor_beds3 : ⌊210 / 8⌋ = 26 := by norm_num
  rw h_floor_beds3 at h_total_beds
  rw [add_assoc, add_comm 32 26, add_assoc]
  norm_num
  
  sorry

end total_flower_beds_l3_3653


namespace find_n_l3_3097

def seq (n : ℕ) : ℚ :=
  if n = 1 then 2
  else if n % 3 = 0 then 2 + seq (n / 3)
  else 1 / seq (n - 1)

theorem find_n (n : ℕ) (hn : seq n = 23 / 105) : n = 19 := by
  sorry

end find_n_l3_3097


namespace number_of_8_tuples_l3_3464

-- Define the constraints for a_k
def valid_a (a : ℕ) (k : ℕ) : Prop := 0 ≤ a ∧ a ≤ k

-- Define the condition for the 8-tuple
def valid_8_tuple (a1 a2 a3 a4 b1 b2 b3 b4 : ℕ) : Prop :=
  valid_a a1 1 ∧ valid_a a2 2 ∧ valid_a a3 3 ∧ valid_a a4 4 ∧ 
  (a1 + a2 + a3 + a4 + 2 * b1 + 3 * b2 + 4 * b3 + 5 * b4 = 19)

theorem number_of_8_tuples : 
  ∃ (n : ℕ), n = 1540 ∧ 
  ∃ (a1 a2 a3 a4 b1 b2 b3 b4 : ℕ), valid_8_tuple a1 a2 a3 a4 b1 b2 b3 b4 := 
sorry

end number_of_8_tuples_l3_3464


namespace circumscribed_circle_radius_l3_3167

noncomputable def radius_of_circumscribed_circle (a b : ℝ) : ℝ :=
  (Real.sqrt (a^2 + b^2)) / 2

theorem circumscribed_circle_radius (a r l b R : ℝ)
  (h1 : r = 1)
  (h2 : a = 2 * Real.sqrt 3)
  (h3 : b = 3)
  (h4 : l = a)
  (h5 : R = radius_of_circumscribed_circle l b) :
  R = Real.sqrt 21 / 2 :=
by
  sorry

end circumscribed_circle_radius_l3_3167


namespace new_cube_volume_l3_3363

theorem new_cube_volume (V_ref : ℝ) (A_ref : ℝ) (V_new : ℝ) (A_new : ℝ) 
  (hVref : V_ref = 8)
  (hAref : A_ref = 6 * (3√V_ref) ^ 2)
  (hAnew : A_new = 3 * A_ref)
  (hVnew : V_new = 4 * V_ref) :
  V_new = 32 := by
  sorry

end new_cube_volume_l3_3363


namespace fourth_root_of_256000000_is_400_l3_3815

def x : ℕ := 256000000

theorem fourth_root_of_256000000_is_400 : Nat.root x 4 = 400 := by
  sorry

end fourth_root_of_256000000_is_400_l3_3815


namespace find_x_l3_3112

noncomputable def series := λ (x : ℝ), 2 + 7 * x + 12 * x^2 + 17 * x^3 + infinite_sum x

theorem find_x (x : ℝ) (h_series : series x = 100) (h_bound : |x| < 1) : x = 0.6 :=
sorry

end find_x_l3_3112


namespace find_radius_l3_3762

-- Definitions
def side_length : ℝ := 2
def probability_visible (r : ℝ) : ℝ := 1 / 2

-- Statement to be proved
theorem find_radius (r : ℝ) : probability_visible r = 1 / 2 → r = 3 * Real.sqrt 2 + Real.sqrt 6 :=
sorry

end find_radius_l3_3762


namespace smallest_positive_period_range_of_f_l3_3184

noncomputable def m (x : ℝ) : ℝ × ℝ :=
( sin x ^ 2 + (1 + cos (2 * x)) / 2, sin x )

noncomputable def n (x : ℝ) : ℝ × ℝ :=
( (1 / 2) * cos (2 * x) - (sqrt 3 / 2) * sin (2 * x), 2 * sin x )

noncomputable def f (x : ℝ) : ℝ := 
( sin x ^ 2 + (1 + cos (2 * x)) / 2 ) * ( (1 / 2) * cos (2 * x) - (sqrt 3 / 2) * sin (2 * x) ) + sin x * (2 * sin x)

theorem smallest_positive_period (x : ℝ) : 
  (f x) = 1 - sin (2 * x + (Real.pi / 6)) → 
  ∃ T > 0, ∀ x, f (x + T) = f x :=
sorry

theorem range_of_f (x : ℝ) : 
  (f x) = 1 - sin (2 * x + (Real.pi / 6)) ∧ 0 ≤ x ∧ x ≤ Real.pi / 2 → 
  ∃ y, y ∈ set.Icc 0 (3 / 2) ∧ f x = y :=
sorry

end smallest_positive_period_range_of_f_l3_3184


namespace determine_f_1789_l3_3033

theorem determine_f_1789
  (f : ℕ → ℕ)
  (h1 : ∀ n : ℕ, 0 < n → f (f n) = 4 * n + 9)
  (h2 : ∀ k : ℕ, f (2^k) = 2^(k+1) + 3) :
  f 1789 = 3581 :=
sorry

end determine_f_1789_l3_3033


namespace options_a_and_c_correct_l3_3727

theorem options_a_and_c_correct :
  (let a₁ := (2, 3, -1) in
   let b₁ := (-2, -3, 1) in
   let a₁_is_parallel_b₁ := a₁ = (-1 : ℝ) • b₁ in
   (let a₂ := (1, -1, 2) in
    let u₂ := (6, 4, -1) in
    let not_parallel_or_perpendicular := ¬(a₂ = (6 : ℝ) • u₂) ∧ a₂.1 * u₂.1 + a₂.2 * u₂.2 + a₂.3 * u₂.3 ≠ 0 in
    (let u₃ := (2, 2, -1) in
     let v₃ := (-3, 4, 2) in
     let dot_product_zero := u₃.1 * v₃.1 + u₃.2 * v₃.2 + u₃.3 * v₃.3 = 0 in
     (let a₄ := (0, 3, 0) in
      let u₄ := (0, -5, 0) in
      let parallel_but_not_perpendicular := a₄ = (3 / 5 : ℝ) • u₄) in
     a₁_is_parallel_b₁ ∧ dot_product_zero))) :=
sorry

end options_a_and_c_correct_l3_3727


namespace angle_gac_equals_angle_eac_l3_3228

variables {A B C D E F G : Type} [AffinePlane A]
variables {a b c d e f g : A}

-- Given conditions
def quadrilateral (a b c d : A) : Prop := True
def diagonal_bisects_angle_bad (a b c d : A) : Prop := 
  ∃ (line_ac : Line A),
  line_ac.contains a ∧ line_ac.contains c ∧ bisects_angle line_ac (angle a b d)

def point_e_on_cd (c d e : A) : Prop := 
  ∃ (line_cd : Line A),
  line_cd.contains c ∧ line_cd.contains d ∧ line_cd.contains e

def be_intersects_ac_at_f (b e a c f : A) : Prop := 
  ∃ (line_be : Line A), ∃ (line_ac : Line A),
  line_be.contains b ∧ line_be.contains e ∧ line_ac.contains a ∧ line_ac.contains c ∧ 
  line_be.contains f ∧ line_ac.contains f

def df_intersects_bc_at_g (d f b c g : A) : Prop := 
  ∃ (line_df : Line A), ∃ (line_bc : Line A),
  line_df.contains d ∧ line_df.contains f ∧ line_bc.contains b ∧ line_bc.contains c ∧
  line_df.contains g ∧ line_bc.contains g

-- The theorem to prove
theorem angle_gac_equals_angle_eac 
  [quadrilateral a b c d]
  [diagonal_bisects_angle_bad a b c d]
  [point_e_on_cd c d e]
  [be_intersects_ac_at_f b e a c f]
  [df_intersects_bc_at_g d f b c g] :
  angle g a c = angle e a c :=
by sorry

end angle_gac_equals_angle_eac_l3_3228


namespace find_principal_amount_l3_3035

noncomputable def principal_amount (R : ℚ) : ℚ :=
  let P := 3000 in
  let T := 9 in
  let ΔR := 5 in
  let extra_interest := 1350 in
  let I1 := (P * R * T) / 100 in
  let I2 := (P * (R + ΔR) * T) / 100 in
  I2 = I1 + extra_interest 

theorem find_principal_amount (R : ℚ) : principal_amount R = 3000 :=
  sorry

end find_principal_amount_l3_3035


namespace compute_difference_of_squares_l3_3093

theorem compute_difference_of_squares : 
  let a := 23 + 15
  let b := 23 - 15
  (a^2 - b^2) = 1380 := 
by
  rw [pow_two, pow_two, add_assoc, sub_eq_add_neg, add_assoc, add_comm, sub_eq_add_neg]
  -- The proof step can be filled later
  sorry

end compute_difference_of_squares_l3_3093


namespace integer_solutions_to_equation_l3_3848

theorem integer_solutions_to_equation :
  {p : ℤ × ℤ | (p.fst^2 - 2 * p.fst * p.snd - 3 * p.snd^2 = 5)} =
  {(4, 1), (2, -1), (-4, -1), (-2, 1)} :=
by {
  sorry
}

end integer_solutions_to_equation_l3_3848


namespace proof_problem_l3_3521

-- Define the point M on the parabola and the parabola equation
def pointM : ℝ × ℝ := (1, 2)

def parabola (p : ℝ) (x y : ℝ) := y^2 = 2 * p * x

-- Define the directrix of the parabola
def directrix (p : ℝ) := x = -p / 2

-- Define the line l passing through point T
def line_l (k x : ℝ) := k * x + 1

-- Define the complementary slopes condition
def complementary_slopes (x1 y1 x2 y2 : ℝ) :=
  (y1 - 2) / (x1 - 1) + (y2 - 2) / (x2 - 1) = 0

-- Define the lengths |TA| and |TB| and their product
def distance (x1 y1 x2 y2 : ℝ) := sqrt ((x2 - x1)^2 + (y2 - y1)^2)

noncomputable def TA_TB (T A B : ℝ × ℝ) := 
  let (x1, y1) := T
  let (x2, y2) := A
  let (x3, y3) := B
  distance x1 y1 x2 y2 * distance x1 y1 x3 y3

theorem proof_problem (p : ℝ) (T A B : ℝ × ℝ) (x1 x2 y1 y2 k : ℝ) :
  pointM ∈ parabola p ∧ 
  directrix p = -1 ∧
  line_l k 0 = 1 ∧
  parabola p x1 y1 ∧ parabola p x2 y2 ∧ complementary_slopes x1 y1 x2 y2 ∧ 
  T = (0, 1) ∧ 
  A = (x1, y1) ∧ B = (x2, y2) →
  TA_TB T A B = 2 :=
sorry

end proof_problem_l3_3521


namespace minimum_RS_value_l3_3249

theorem minimum_RS_value {A B C D M R S : Point}
    (h_rhombus : is_rhombus A B C D)
    (h_diag1 : dist A C = 24)
    (h_diag2 : dist B D = 10)
    (h_M_on_AD : lies_on M A D)
    (h_R : foot M A C R)
    (h_S : foot M B D S) :
  ∃ M on A D, dist R S = 0 :=
by
  sorry

end minimum_RS_value_l3_3249


namespace ron_spends_on_chocolate_bars_l3_3042

/-- Ron is hosting a camp for 15 scouts where each scout needs 2 s'mores.
    Each chocolate bar costs $1.50 and can be broken into 3 sections to make 3 s'mores.
    A discount of 15% applies if 10 or more chocolate bars are purchased.
    Calculate the total amount Ron will spend on chocolate bars after applying the discount if applicable. -/
theorem ron_spends_on_chocolate_bars :
  let cost_per_bar := 1.5
  let s'mores_per_bar := 3
  let scouts := 15
  let s'mores_per_scout := 2
  let total_s'mores := scouts * s'mores_per_scout
  let bars_needed := total_s'mores / s'mores_per_bar
  let discount := 0.15
  let total_cost := bars_needed * cost_per_bar
  let discount_amount := if bars_needed >= 10 then discount * total_cost else 0
  let final_cost := total_cost - discount_amount
  final_cost = 12.75 := by sorry

end ron_spends_on_chocolate_bars_l3_3042


namespace three_digit_numbers_divisible_by_11_are_550_or_803_l3_3832

theorem three_digit_numbers_divisible_by_11_are_550_or_803 :
  ∀ (N : ℕ), (100 ≤ N ∧ N < 1000 ∧ ∃ (a b c : ℕ), N = 100 * a + 10 * b + c ∧ a ≠ 0 ∧ 11 ∣ N ∧ (N / 11 = a^2 + b^2 + c^2)) → (N = 550 ∨ N = 803) :=
by
  sorry

end three_digit_numbers_divisible_by_11_are_550_or_803_l3_3832


namespace inscribed_sphere_surface_area_l3_3508

theorem inscribed_sphere_surface_area (V S : ℝ) (hV : V = 2) (hS : S = 3) : 4 * Real.pi * (3 * V / S)^2 = 16 * Real.pi := by
  sorry

end inscribed_sphere_surface_area_l3_3508


namespace inequality_AB_CD_max_AM_DM_BN_CN_l3_3655

noncomputable def midpoint (x y : Point) : Point := sorry
noncomputable def length (a b : Point) : ℝ := sorry

variable {A B C D M N : Point}
variable (AB CD AM DM BN CN : ℝ)

-- assuming convex quadrilateral ABCD
axiom is_convex_quadrilateral : ∀ A B C D : Point, ConvexQuadrilateral A B C D

-- defining midpoints
axiom midpoint_M : M = midpoint B C
axiom midpoint_N : N = midpoint A D

-- lengths of sides
axiom length_AB : length A B = AB
axiom length_CD : length C D = CD
axiom length_AM : length A M = AM
axiom length_DM : length D M = DM
axiom length_BN : length B N = BN
axiom length_CN : length C N = CN

-- the theorem statement
theorem inequality_AB_CD_max_AM_DM_BN_CN :
    AB + CD > max (AM + DM) (BN + CN) := by
  sorry

end inequality_AB_CD_max_AM_DM_BN_CN_l3_3655


namespace no_partition_special_sets_l3_3712

def is_special (s : Set ℤ) : Prop :=
  ∃ a b c d : ℤ, s = {a, b, c, d} ∧ (a * b - c * d = 1)

theorem no_partition_special_sets (n : ℕ) (hn : n > 0) :
  ¬ ∃ (S : Finset (Set ℤ)), S.card = n ∧ Finset.univ = (Finset.range (4 * n) : Set ℤ) ∧ ∀ s ∈ S, is_special s :=
by
  sorry

end no_partition_special_sets_l3_3712


namespace sum_legs_of_larger_triangle_l3_3706

open Real

noncomputable def area_of_triangle (a b : ℝ) : ℝ := (a * b) / 2

def is_similar_right_triangle_with_given_hypotenuse 
  (hypotenuse_small : ℝ) 
  (area_small area_large : ℝ) 
  (sum_legs_large : ℝ) : Prop :=
  ∃ a b : ℝ, 
    a^2 + b^2 = hypotenuse_small^2 ∧ 
    area_of_triangle a b = area_small ∧ 
    let scale_factor := sqrt (area_large / area_small) in
    let a_large := scale_factor * a in
    let b_large := scale_factor * b in
    a_large + b_large = sum_legs_large

theorem sum_legs_of_larger_triangle 
  (hypotenuse_small : ℝ) 
  (area_small area_large : ℝ) :
  hypotenuse_small = 8 ∧ area_small = 10 ∧ area_large = 250 → 
  is_similar_right_triangle_with_given_hypotenuse hypotenuse_small area_small area_large 51 :=
sorry

end sum_legs_of_larger_triangle_l3_3706


namespace descent_time_proof_l3_3374

section XiaohongMountainProblem

variables (time_ascent_hours : ℕ) (time_ascent_minutes : ℕ)
variables (rest_up : ℕ) (walk_cycle_up : ℕ)
variables (rest_down : ℕ) (walk_cycle_down : ℕ)
variables (speed_ratio : ℚ)

def total_ascent_time : ℕ := (time_ascent_hours * 60) + time_ascent_minutes

def ascent_cycles : ℕ := total_ascent_time walk_cycle_up rest_up / (walk_cycle_up + rest_up)
def ascent_remainder : ℕ := total_ascent_time walk_cycle_up rest_up % (walk_cycle_up + rest_up)

def total_rest_time_ascent : ℕ := ascent_cycles * rest_up

def actual_walk_time_ascent : ℕ := total_ascent_time walk_cycle_up rest_up - total_rest_time_ascent walk_cycle_up rest_up

def actual_walk_time_descent : ℕ := actual_walk_time_ascent walk_cycle_up rest_up / speed_ratio

def descent_cycles : ℕ := actual_walk_time_descent actual_walk_time_ascent walk_cycle_up rest_up rest_down / walk_cycle_down
def descent_remainder : ℕ := actual_walk_time_descent actual_walk_time_ascent walk_cycle_up rest_up rest_down % walk_cycle_down

def total_descent_time : ℕ := actual_walk_time_descent actual_walk_time_ascent walk_cycle_up rest_up rest_down + (descent_cycles - 1) * rest_down

-- Given conditions
variable (h_ascent_time : total_ascent_time 3 50 = 230)
variable (h_rest_up : rest_up = 10)
variable (h_walk_cycle_up : walk_cycle_up = 30)
variable (h_rest_down : rest_down = 5)
variable (h_walk_cycle_down : walk_cycle_down = 30)
variable (h_speed_ratio : speed_ratio = 3 / 2)

-- Prove that the total descent time is 135 minutes
theorem descent_time_proof : total_descent_time 3 50 10 30 5 30 (3 / 2) = 135 := by
  sorry

end XiaohongMountainProblem

end descent_time_proof_l3_3374


namespace problem_ns_k_divisibility_l3_3834

theorem problem_ns_k_divisibility (n k : ℕ) (h1 : 0 < n) (h2 : 0 < k) :
  (∃ (a b : ℕ), (a = 1 ∨ a = 5) ∧ (b = 1 ∨ b = 5) ∧ a = n ∧ b = k) ↔ 
  n * k ∣ (2^(2^n) + 1) * (2^(2^k) + 1) := 
sorry

end problem_ns_k_divisibility_l3_3834


namespace grandfather_sent_150_l3_3812

variable {T G : ℝ}
variable (aunt : ℝ) (bank_ratio : ℝ) (bank_amount : ℝ)

-- Conditions
def received_from_aunt := aunt = 75
def fraction_put_in_bank := bank_ratio = 1/5
def amount_put_in_bank := bank_amount = 45

-- Derived Conditions
def total_money_received := bank_ratio * T = bank_amount
def grandfathers_contribution := T - aunt = G

-- Theorem to prove
theorem grandfather_sent_150:
  received_from_aunt aunt →
  fraction_put_in_bank bank_ratio →
  amount_put_in_bank bank_amount →
  total_money_received T bank_ratio bank_amount →
  grandfathers_contribution T aunt G →
  G = 150 :=
by
  intros
  sorry

end grandfather_sent_150_l3_3812


namespace initial_customers_l3_3065

theorem initial_customers (tables : ℕ) (people_per_table : ℕ) (customers_left : ℕ) (h1 : tables = 5) (h2 : people_per_table = 9) (h3 : customers_left = 17) :
  tables * people_per_table + customers_left = 62 :=
by
  sorry

end initial_customers_l3_3065


namespace circle_radius_tangent_to_semicircles_l3_3649

theorem circle_radius_tangent_to_semicircles:
  ∀ (A B C O1 O2 O3 : ℝ) (AB AC O1O3 O2O3 : ℝ),
  AC = 12 ∧ AB = 4 ∧ O1O3 = x + 2 ∧ O2O3 = 6 - x ∧ sqrt(12 * x * (4 - x)) = 2 * x →
  x = 3 := by
  sorry

end circle_radius_tangent_to_semicircles_l3_3649


namespace jackson_chairs_l3_3235

theorem jackson_chairs (a b c d : ℕ) (h1 : a = 6) (h2 : b = 4) (h3 : c = 12) (h4 : d = 6) : a * b + c * d = 96 := 
by sorry

end jackson_chairs_l3_3235


namespace compare_functions_l3_3156

variables {ℝ : Type*} [Real : LinearOrderedField ℝ]

theorem compare_functions
  (f g : ℝ → ℝ) 
  (m n : ℝ) 
  (hmn : m < n)
  (hf_diff : DifferentiableOn ℝ f (Set.Icc m n))
  (hg_diff : DifferentiableOn ℝ g (Set.Icc m n))
  (h_der : ∀ x ∈ (Set.Ioo m n), deriv f x < deriv g x):
  ∀ x ∈ (Set.Ioo m n), f x + g n < g x + f n :=
by
  sorry

end compare_functions_l3_3156


namespace probability_two_cards_l3_3343

noncomputable def probability_first_spade_second_ace : ℚ :=
  let total_cards := 52
  let total_spades := 13
  let total_aces := 4
  let remaining_cards := total_cards - 1
  
  let first_spade_non_ace := (total_spades - 1) / total_cards
  let second_ace_after_non_ace := total_aces / remaining_cards
  
  let probability_case1 := first_spade_non_ace * second_ace_after_non_ace
  
  let first_ace_spade := 1 / total_cards
  let second_ace_after_ace := (total_aces - 1) / remaining_cards
  
  let probability_case2 := first_ace_spade * second_ace_after_ace
  
  probability_case1 + probability_case2

theorem probability_two_cards {p : ℚ} (h : p = 1 / 52) : 
  probability_first_spade_second_ace = p := 
by 
  simp only [probability_first_spade_second_ace]
  sorry

end probability_two_cards_l3_3343


namespace find_x_value_l3_3781

theorem find_x_value :
  ∀ (x : ℝ), 0.3 + 0.1 + 0.4 + x = 1 → x = 0.2 :=
by
  intros x h
  sorry

end find_x_value_l3_3781


namespace remainder_3_302_plus_302_div_by_3_151_plus_3_101_plus_1_l3_3021

-- Definitions from the conditions
def a : ℕ := 3^302
def b : ℕ := 3^151 + 3^101 + 1

-- Theorem: Prove that the remainder when a + 302 is divided by b is 302.
theorem remainder_3_302_plus_302_div_by_3_151_plus_3_101_plus_1 :
  (a + 302) % b = 302 :=
by {
  sorry
}

end remainder_3_302_plus_302_div_by_3_151_plus_3_101_plus_1_l3_3021


namespace finite_integer_k_l3_3518

theorem finite_integer_k (m n : ℕ) (hm : m > 0) (hn : n > 0) : 
  {k : ℕ | (m + 1/2)^k + (n + 1/2)^k ∈ ℤ}.finite :=
by
  sorry

end finite_integer_k_l3_3518


namespace estimated_value_of_y_l3_3177

theorem estimated_value_of_y (x : ℝ) (h : x = 25) : 
  let y := 0.50 * x - 0.81 in
  y = 11.69 :=
by
  rw [h]
  let y := 0.50 * 25 - 0.81
  sorry

end estimated_value_of_y_l3_3177


namespace max_min_f_l3_3972

-- Defining a and the set A
def a : ℤ := 2001

def A : Set (ℤ × ℤ) := {p | p.snd ≠ 0 ∧ p.fst < 2 * a ∧ (2 * p.snd) ∣ ((2 * a * p.fst) - (p.fst * p.fst) + (p.snd * p.snd)) ∧ ((p.snd * p.snd) - (p.fst * p.fst) + (2 * p.fst * p.snd) ≤ (2 * a * (p.snd - p.fst)))}

-- Defining the function f
def f (m n : ℤ): ℤ := (2 * a * m - m * m - m * n) / n

-- Main theorem: Proving that the maximum and minimum values of f over A are 3750 and 2 respectively
theorem max_min_f : 
  ∃ p ∈ A, f p.fst p.snd = 3750 ∧
  ∃ q ∈ A, f q.fst q.snd = 2 :=
sorry

end max_min_f_l3_3972


namespace f_neg_a_eq_neg_11_l3_3997

def f (x : Real) : Real := x^2 * Real.sin x + 2

variable (a : Real)

-- Given conditions
axiom fa_eq_15 : f a = 15
axiom fa_plus_fneg_a_eq_4 : f a + f (-a) = 4

-- Theorem to prove
theorem f_neg_a_eq_neg_11 : f (-a) = -11 :=
by
  -- Placeholder for proof
  sorry

end f_neg_a_eq_neg_11_l3_3997


namespace inequality_solution_l3_3864

theorem inequality_solution (a x : ℝ) (h : 0 ≤ a ∧ a ≤ 4) :
  (x^2 + a * x > 4 * x + a - 3) ↔ (x < -1 ∨ x > 3)
:=
sorry

end inequality_solution_l3_3864


namespace eq_holds_at_most_finitely_u_n_a_b_l3_3267

theorem eq_holds_at_most_finitely_u_n_a_b (u : ℕ) (hu : u > 0) :
  ∃ m : ℕ, ∀ (n a b : ℕ), (n! = u ^ a - u ^ b) → n < m := sorry

end eq_holds_at_most_finitely_u_n_a_b_l3_3267


namespace x_100_eq_neg1_S_100_eq_2_l3_3879

def seq (x : ℕ → ℤ) : Prop :=
  ∀ n ≥ 2, x (n + 1) = x n - x (n - 1)

variables (x : ℕ → ℤ) (S : ℕ → ℤ)

-- Conditions
axiom x1 : x 1 = 1
axiom x2 : x 2 = 2
axiom seq_cond : seq x
axiom S_def : ∀ n, S n = ∑ i in Finset.range (n + 1), x i

-- Questions/proofs to show
theorem x_100_eq_neg1 : x 100 = -1 := 
sorry

theorem S_100_eq_2 : S 100 = 2 := 
sorry

end x_100_eq_neg1_S_100_eq_2_l3_3879


namespace sector_area_correct_l3_3194

-- Define the conditions: central angle and arc length
def central_angle : ℝ := 2  -- 2 radians
def arc_length : ℕ := 2  -- 2 cm

-- Define the question: area of the sector formed by the central angle
def sector_area (r θ : ℝ) : ℝ := (1 / 2) * r^2 * θ

-- Assuming the radius is derived from the arc length and central angle
def radius (l θ : ℝ) : ℝ := l / θ

-- Theorem statement: the area of the sector is 1 cm^2
theorem sector_area_correct :
  sector_area (radius arc_length central_angle) central_angle = 1 := by
  sorry

end sector_area_correct_l3_3194


namespace range_of_m_l3_3996

open Set

def M (m : ℝ) : Set ℝ := {x | x ≤ m}
def N : Set ℝ := {y | ∃ x : ℝ, y = 2^(-x)}

theorem range_of_m (m : ℝ) : (M m ∩ N).Nonempty ↔ m > 0 := sorry

end range_of_m_l3_3996


namespace constant_angle_of_segment_of_tangent_l3_3808

-- Define the circle, points, and tangents
variables {O A B M C : Type}
variable (circle : Set O) -- the circle
variable (O : O) [incircle : ∀ x : O, x ∈ circle]
variable (A B M : O)
variables (tangentA tangentB tangentM : Set O)
variable (C : Type)

-- Define the tangents and their properties
axiom tangentA_is_tangent : ∀ x : O, x ∉ circle ⇔ x ∈ tangentA
axiom tangentB_is_tangent : ∀ x : O, x ∉ circle ⇔ x ∈ tangentB
axiom tangentM_is_tangent : ∀ x : O, x ∉ circle ⇔ x ∈ tangentM

-- Define intersection point of tangents at A and B
axiom pointC : C ∈ tangentA ∧ C ∈ tangentB

-- Define the theorem
theorem constant_angle_of_segment_of_tangent :
  ∃ angle : ℝ, angle = 90 - (angle of intersection between tangentA and tangentB) / 2 :=
sorry

end constant_angle_of_segment_of_tangent_l3_3808


namespace sin_double_angle_l3_3522

theorem sin_double_angle (α : ℝ) (h : Real.tan α = 2) : Real.sin (2 * α) = 4 / 5 :=
by sorry

end sin_double_angle_l3_3522


namespace proof_result_l3_3731

noncomputable def direction_vector (a b : ℝ × ℝ × ℝ) := 
  ∃ k : ℝ, (k • b) = a

noncomputable def perpendicular_vector (a b : ℝ × ℝ × ℝ) := 
  (a.1 * b.1 + a.2 * b.2 + a.3 * b.3) = 0

def parallel_line_condition := 
  direction_vector (2, 3, -1) (-2, -3, 1) 

def perpendicular_plane_condition := 
  perpendicular_vector (2, 2, -1) (-3, 4, 2)

theorem proof_result: parallel_line_condition ∧ perpendicular_plane_condition :=
by
  have h1 : parallel_line_condition := sorry
  have h2 : perpendicular_plane_condition := sorry
  exact ⟨h1, h2⟩

end proof_result_l3_3731


namespace no_trisector_divides_triangle_area_eq_l3_3959

theorem no_trisector_divides_triangle_area_eq (n : ℕ) (h : n > 2) : 
  ∀ (A B C : Point), 
  ¬(exists (AF : (LineSegment A B)), exists (AF1 AF2 AF3 : LineSegment A B), 
  divides_triangle_area_eq (triangle B A C) (AF1 AF2 AF3)) := sorry

end no_trisector_divides_triangle_area_eq_l3_3959


namespace a_2018_value_l3_3890

theorem a_2018_value (S a : ℕ -> ℕ) (h₁ : S 1 = a 1) (h₂ : a 1 = 1) (h₃ : ∀ n : ℕ, n > 0 -> S (n + 1) = 3 * S n) :
  a 2018 = 2 * 3 ^ 2016 :=
sorry

end a_2018_value_l3_3890


namespace Cherry_weekly_earnings_l3_3814

theorem Cherry_weekly_earnings :
  let charge_small_cargo := 2.50
  let charge_large_cargo := 4.00
  let daily_small_cargo := 4
  let daily_large_cargo := 2
  let days_in_week := 7
  let daily_earnings := (charge_small_cargo * daily_small_cargo) + (charge_large_cargo * daily_large_cargo)
  let weekly_earnings := daily_earnings * days_in_week
  weekly_earnings = 126 := sorry

end Cherry_weekly_earnings_l3_3814


namespace problem_solution_l3_3358

theorem problem_solution : (2^0 - 1 + 5^2 - 0)⁻¹ * 5 = 1 / 5 := by
  sorry

end problem_solution_l3_3358


namespace angle_between_hands_at_3_15_l3_3915

noncomputable def angle_at_3_15 : ℝ :=
let minute_angle := 15 * (360 / 60) in
let hour_angle := 3 * (360 / 12) + 0.25 * (360 / 12) in
abs (minute_angle - hour_angle)

theorem angle_between_hands_at_3_15 : angle_at_3_15 = 7.5 := by
  sorry

end angle_between_hands_at_3_15_l3_3915


namespace euclidean_div_remainder_l3_3083

noncomputable def P (X : ℝ) : ℝ := X^100 - 2*X^51 + 1
noncomputable def D (X : ℝ) : ℝ := X^2 - 1

theorem euclidean_div_remainder :
  ∃ Q R : ℝ → ℝ, P = λ X, D X * Q X + R X ∧ (∃ a b : ℝ, R = λ X, a * X + b ∧ a = -2 ∧ b = 2) :=
by
  sorry

end euclidean_div_remainder_l3_3083


namespace range_of_a_l3_3543

noncomputable def g (a x : ℝ) : ℝ := a * real.exp x - x + 2 * a^2 - 3

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, a * real.exp x - x + 2 * a^2 - 3 > 0) ↔ (a ≤ 0 ∨ (0 < a ∧ a ≤ 1)) :=
by
  sorry

end range_of_a_l3_3543


namespace volume_of_hemisphere_l3_3412

theorem volume_of_hemisphere (d : ℝ) (h : d = 10) : 
  let r := d / 2
  let V := (2 / 3) * π * r^3
  V = 250 / 3 * π := by
sorry

end volume_of_hemisphere_l3_3412


namespace number_of_divisors_30030_l3_3128

theorem number_of_divisors_30030 : 
  let n := 30030 in 
  let prime_factors := [2, 3, 5, 7, 11, 13] in
  (∀ p ∈ prime_factors, nat.prime p) → 
  (∀ p ∈ prime_factors, n % p = 0) → 
  (∀ p₁ p₂ ∈ prime_factors, p₁ ≠ p₂ → ∀ m, n % (p₁ * p₂ * m) ≠ 0) →
  (∀ p ∈ prime_factors, ∃ m, n = p ^ 1 * m ∧ nat.gcd(p, m) = 1) → 
  ∃ t, t = 64 ∧ t = ∏ p in prime_factors.to_finset, ((1 : ℕ) + 1) :=
by
  intro n prime_factors hprimes hdivisors hunique hfactored
  use 64
  rw list.prod_eq_foldr at *
  suffices : 64 = list.foldr (λ _ r, 2 * r) 1 prime_factors, from this
  simp [list.foldr, prime_factors]
  sorry

end number_of_divisors_30030_l3_3128


namespace sum_harmonic_series_l3_3134

noncomputable def H (n : ℕ) : ℝ := ∑ i in Finset.range (n + 1), (1 / (i + 1 : ℝ))

theorem sum_harmonic_series :
  ∑' n, (1 / (n ^ 2 * H n * H (n + 1))) = 1 :=
sorry

end sum_harmonic_series_l3_3134


namespace PA_dot_PC_range_l3_3219

-- Define the conditions for the parallelogram.
variables (A B C D P : Type) [InnerProductSpace ℝ]

-- Let AB = 4, AD = 2, and AB ⋅ AD = 4.
variables (AB AD : A ≃ₗᵢ[ℝ] B)
variables (hAB : norm (AB - A) = 4)
variables (hAD : norm (AD - A) = 2)
variables (hAB_AD_dot : ⟪AB - A, AD - A⟫ = 4)

-- Point P lies on edge CD.
variables (λ : ℝ) (hλ : 0 ≤ λ ∧ λ ≤ 1)
variables (hDP : P = D + λ • (C - D))

-- Define PA and PC.
def PA := P - A
def PC := P - C

-- Prove the range of PA ⋅ PC.
theorem PA_dot_PC_range : -25 / 4 ≤ ⟪PA, PC⟫ ∧ ⟪PA, PC⟫ ≤ 0 := sorry

end PA_dot_PC_range_l3_3219


namespace distance_from_A_to_B_l3_3196

-- Define points A and B
structure Point where
  x : ℝ
  y : ℝ

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2)

-- Define point A and point B
def A : Point := ⟨-3, 5⟩
def B : Point := ⟨2, 10⟩

-- Define the symmetric point of A w.r.t the x-axis
def A_sym : Point := ⟨A.x, -A.y⟩

-- State the theorem about the distance from A to B
theorem distance_from_A_to_B : distance A_sym B = 5 * Real.sqrt 10 := by
  sorry

end distance_from_A_to_B_l3_3196


namespace equal_medians_isosceles_isosceles_equal_medians_l3_3285

noncomputable theory

-- Define the triangle and its properties
structure Triangle :=
  (A B C : Point) -- vertices

def median (A B C : Point) : Segment :=
  segment (midpoint B C) A

-- Define the property that two segments are equal
def segments_equal (m1 m2 : Segment) : Prop :=
  m1.length = m2.length

-- Prove the main theorem: If two medians of a triangle are equal, then the triangle is isosceles
theorem equal_medians_isosceles {A B C : Point} (h1 : segments_equal (median A B C) (median B A C)) :
  is_isosceles (Triangle.mk A B C) :=
sorry

-- Prove the converse theorem: If a triangle is isosceles, then two medians are equal
theorem isosceles_equal_medians {A B C : Point} (h2 : is_isosceles (Triangle.mk A B C)) :
  segments_equal (median A B C) (median B A C) :=
sorry

end equal_medians_isosceles_isosceles_equal_medians_l3_3285


namespace conor_chop_eggplants_l3_3826

theorem conor_chop_eggplants (E : ℕ) 
  (condition1 : E + 9 + 8 = (E + 17))
  (condition2 : 4 * (E + 9 + 8) = 116) :
  E = 12 :=
by {
  sorry
}

end conor_chop_eggplants_l3_3826


namespace multiplication_pattern_correct_l3_3913

theorem multiplication_pattern_correct :
  (1 * 9 + 2 = 11) ∧
  (12 * 9 + 3 = 111) ∧
  (123 * 9 + 4 = 1111) ∧
  (1234 * 9 + 5 = 11111) ∧
  (12345 * 9 + 6 = 111111) →
  123456 * 9 + 7 = 1111111 :=
by
  sorry

end multiplication_pattern_correct_l3_3913


namespace cos_arcsin_l3_3091

theorem cos_arcsin (h : (7:ℝ) / 25 ≤ 1) : Real.cos (Real.arcsin ((7:ℝ) / 25)) = (24:ℝ) / 25 := by
  -- Proof to be provided
  sorry

end cos_arcsin_l3_3091


namespace sum_of_three_equal_numbers_l3_3671

theorem sum_of_three_equal_numbers
  (mean : ℝ)
  (a1 a2 : ℝ)
  (a3 a4 a5 : ℝ)
  (h_mean : mean = 20)
  (h_first : a1 = 12)
  (h_second : a2 = 24)
  (h_equal : a3 = a4 ∧ a4 = a5) :
  a3 + a4 + a5 = 64 :=
by 
  have h_sum : mean * 5 = 100 := by calc
    mean * 5 = 20 * 5 : by rw [h_mean]
    ... = 100 : by norm_num,
  have h_total_sum : a1 + a2 + a3 + a4 + a5 = 100 := by calc
    a1 + a2 + a3 + a4 + a5 = 12 + 24 + a3 + a4 + a5 : by rw [h_first, h_second]
    ... = 100 : by rw [←h_sum, h_mean]; norm_num,
  have h_unknown_sum : a3 + a4 + a5 = 100 - 12 - 24 := by linarith,
  rw [←h_unknown_sum, h_equal.left, h_equal.right],
  norm_num,
  sorry

end sum_of_three_equal_numbers_l3_3671


namespace find_base_b_l3_3130

theorem find_base_b :
  ∃ b : ℕ, (b > 7) ∧ (b > 10) ∧ (b > 8) ∧ (b > 12) ∧ 
    (4 + 3 = 7) ∧ ((2 + 7 + 1) % b = 3) ∧ ((3 + 4 + 1) % b = 5) ∧ 
    ((5 + 6 + 1) % b = 2) ∧ (1 + 1 = 2)
    ∧ b = 13 :=
by
  sorry

end find_base_b_l3_3130


namespace virtual_set_divisors_count_l3_3428

def is_virtual_set (A: set ℕ) : Prop :=
  A.card = 5 ∧ 
  (∀ B ⊆ A, B.card = 3 → (gcd (B.to_finset.to_list.nth 0).get_or_else 1 
                           (gcd (B.to_finset.to_list.nth 1).get_or_else 1 
                                (B.to_finset.to_list.nth 2).get_or_else 1) > 1)) ∧
  (∀ C ⊆ A, C.card = 4 → (gcd (C.to_finset.to_list.nth 0).get_or_else 1 
                           (gcd (C.to_finset.to_list.nth 1).get_or_else 1 
                                (gcd (C.to_finset.to_list.nth 2).get_or_else 1 
                                      (C.to_finset.to_list.nth 3).get_or_else 1)) = 1))

theorem virtual_set_divisors_count (A : set ℕ) (hA : is_virtual_set A) :
  (∏ x in A, x).factors.to_finset.card >= 2020 := sorry

end virtual_set_divisors_count_l3_3428


namespace area_triangle_NQP_eq_area_quad_TSQR_l3_3182

-- Define triangle and corresponding points with given properties
variables {A B C M N P R S Q T : Type}
variables [DecidableEq A] [DecidableEq B] [DecidableEq C]
variables [DecidableEq M] [DecidableEq N] [DecidableEq P]
variables [DecidableEq R] [DecidableEq S] [DecidableEq Q]
variables [DecidableEq T]

-- Assumptions given in the problem
variable {triangleABC : Triangle}
variable {points_on_sides : PointsOnSides ABC M N P}
variable {parallelogram_CPMN : Parallelogram CPMN}
variable {intersection_AN_MP_R : Intersection AN MP R}
variable {intersection_BP_MN_S : Intersection BP MN S}
variable {intersection_AN_BP_Q : Intersection AN BP Q}
variable {projection_Q_AB_T : Projection Q AB T}

-- State the theorem to be proved
theorem area_triangle_NQP_eq_area_quad_TSQR :
  (area (triangle N Q P) = area (quad T S Q R)) := sorry

end area_triangle_NQP_eq_area_quad_TSQR_l3_3182


namespace units_digit_of_power_l3_3920

theorem units_digit_of_power (n : ℕ) : units_digit (4589 ^ 1276) = 1 :=
by sorry

end units_digit_of_power_l3_3920


namespace largest_fraction_l3_3068

noncomputable def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem largest_fraction {m n : ℕ} (hm : 1000 ≤ m ∧ m ≤ 9999) (hn : 1000 ≤ n ∧ n ≤ 9999) (h_sum : digit_sum m = digit_sum n) :
  (m = 9900 ∧ n = 1089) ∧ m / n = 9900 / 1089 :=
by
  sorry

end largest_fraction_l3_3068


namespace circumcircle_DPQTangent_BF_l3_3629

-- Definitions of points A, B, C, D, E, F on the circumcircle of a cyclic hexagon.
variables {A B C D E F P Q : Type}
variables [ConvexCyclicHexagon A B C D E F] 
variables [Quadrilateral A B D F] [Square A B D F]
variables {angle}
variables (incenter_ACE_on_BF : LiesOnIncenter (IncenterOfTriangle A C E) (LineThrough B F))
variables (P_on_CE_BD : IntersectsAt P (LineThrough C E) (LineThrough B D))
variables (Q_on_CE_DF : IntersectsAt Q (LineThrough C E) (LineThrough D F))

-- Prove that the circumcircle of triangle DPQ is tangent to line BF.
theorem circumcircle_DPQTangent_BF
  (h1 : convex_cyclic_hexagon A B C D E F)
  (h2 : quadrilateral A B D F)
  (h3 : square A B D F)
  (h4 : incenter_ACE_on_BF)
  (h5 : P_on_CE_BD)
  (h6 : Q_on_CE_DF)
  : Tangent (Circumcircle (Triangle D P Q)) (LineThrough B F) :=
sorry

end circumcircle_DPQTangent_BF_l3_3629


namespace marks_in_modern_literature_l3_3789

theorem marks_in_modern_literature
    (marks_geography : ℕ)
    (marks_history_government : ℕ)
    (marks_art : ℕ)
    (marks_computer_science : ℕ)
    (average_marks : ℚ)
    (total_subjects : ℕ)
    (total_marks : ℚ) :
  marks_geography = 56 →
  marks_history_government = 60 →
  marks_art = 72 →
  marks_computer_science = 85 →
  average_marks = 70.6 →
  total_subjects = 5 →
  total_marks = average_marks * total_subjects →
  ∑ (i : ℕ) in {marks_geography, marks_history_government, marks_art, marks_computer_science, modern_literature}, i = total_marks →
  modern_literature = 80 :=
by {
  intros,
  sorry
}

end marks_in_modern_literature_l3_3789


namespace arithmetic_sequence_length_l3_3833

theorem arithmetic_sequence_length :
  ∃ n : ℕ, ∀ a d l, a = -48 ∧ d = 6 ∧ l = 72 → l = a + (n-1) * d := by
  sorry

end arithmetic_sequence_length_l3_3833


namespace remainder_of_sum_of_first_15_natural_numbers_divided_by_11_l3_3723

theorem remainder_of_sum_of_first_15_natural_numbers_divided_by_11 :
  (∑ i in finset.range 16, i) % 11 = 10 :=
by 
  sorry

end remainder_of_sum_of_first_15_natural_numbers_divided_by_11_l3_3723


namespace totalSurfaceArea_l3_3064

namespace BoxSurfaceArea

-- Define the dimensions of the original box and the cuts.
def originalBox : ℝ × ℝ × ℝ := (2, 1, 1)
def firstCut : ℝ := 1/4
def secondCut : ℝ := 1/3

-- Define the heights of pieces A, B, and C.
def heightA : ℝ := firstCut
def heightB : ℝ := secondCut
def heightC : ℝ := 1 - (firstCut + secondCut)

-- Define the surface areas.
def topBottomArea : ℝ := 2 * (2 * 1)
def sideArea : ℝ := 2 * (1 * 1)
def frontBackArea : ℝ := 2 * (3 * 1)

-- Prove that the total surface area is 12 square feet.
theorem totalSurfaceArea : topBottomArea + sideArea + frontBackArea = 12 := by
  -- (2 * (1 * 1)) + (2 * (1 * 1)) + (2 * (3 * 1)) = 4 + 2 + 6 = 12
  sorry

end BoxSurfaceArea

end totalSurfaceArea_l3_3064


namespace find_value_l3_3621

noncomputable def f (x : ℝ) : ℝ := sorry

axiom odd_function : ∀ x, f (-x) = -f x
axiom symmetric_about_one : ∀ x, f (x - 1) = f (1 - x)
axiom equation_on_interval : ∀ x, 0 < x ∧ x < 1 → f x = 9^x

theorem find_value : f (5 / 2) + f 2 = -3 := 
by sorry

end find_value_l3_3621


namespace symmetric_polynomials_identity_l3_3480

theorem symmetric_polynomials_identity (x y z : ℝ) :
  let σ1 := x + y + z,
      σ2 := x * y + y * z + z * x in
  (x ^ 3 + y ^ 3 + z ^ 3 - 3 * x * y * z) = σ1 * (σ1 ^ 2 - 3 * σ2) :=
by
  sorry

end symmetric_polynomials_identity_l3_3480


namespace increase_number_correct_l3_3393

-- Definitions for the problem
def originalNumber : ℕ := 110
def increasePercent : ℝ := 0.5

-- Statement to be proved
theorem increase_number_correct : originalNumber + (originalNumber * increasePercent) = 165 := by
  sorry

end increase_number_correct_l3_3393


namespace find_a_l3_3520

noncomputable def A (a : ℝ) : Set ℝ := {2^a, 3}
def B : Set ℝ := {2, 3}
def C : Set ℝ := {1, 2, 3}

theorem find_a (a : ℝ) (h : A a ∪ B = C) : a = 0 :=
sorry

end find_a_l3_3520


namespace odd_integer_solution_l3_3846

theorem odd_integer_solution
  (y : ℤ) (hy_odd : y % 2 = 1)
  (h : ∃ x : ℤ, x^2 + 2*y^2 = y*x^2 + y + 1) :
  y = 1 :=
sorry

end odd_integer_solution_l3_3846


namespace greatest_integer_gcd_is_4_l3_3352

theorem greatest_integer_gcd_is_4 : 
  ∀ (n : ℕ), n < 150 ∧ (Nat.gcd n 24 = 4) → n ≤ 148 := 
by
  sorry

end greatest_integer_gcd_is_4_l3_3352


namespace total_gain_percentage_l3_3060

def cloth_costs (A B C : ℕ):= A = 4 ∧ B = 6 ∧ C = 8
def markups (A B C : ℕ → ℚ):= A 25 ∧ B 30 ∧ C 20
def quantities (A B C : ℕ):= A = 25 ∧ B = 15 ∧ C = 10
def discount (d : ℚ) := d = 0.05
def taxes (t : ℚ) := t = 0.05

theorem total_gain_percentage
    (cost_A cost_B cost_C : ℕ) (markup_A markup_B markup_C : ℕ → ℚ)
    (quantity_A quantity_B quantity_C : ℕ)
    (disc tax : ℚ)
    (gain_percentage : ℚ) :

    cloth_costs cost_A cost_B cost_C →
    markups markup_A markup_B markup_C →
    quantities quantity_A quantity_B quantity_C →
    discount disc →
    taxes tax →
    gain_percentage = 24.87 := 
sorry

end total_gain_percentage_l3_3060


namespace range_of_f_l3_3896

noncomputable def f (x : ℝ) : ℝ := (2^(x+1)) / (2^x + 1)

theorem range_of_f : set.range f = {y | 0 < y ∧ y < 2} :=
sorry

end range_of_f_l3_3896


namespace hyperbola_foci_coords_l3_3116

theorem hyperbola_foci_coords :
  ∀ x y, (x^2) / 8 - (y^2) / 17 = 1 → (x, y) = (5, 0) ∨ (x, y) = (-5, 0) :=
by
  sorry

end hyperbola_foci_coords_l3_3116


namespace combination_sum_eq_l3_3085

theorem combination_sum_eq :
  ∀ (n : ℕ), (2 * n ≥ 10 - 2 * n) ∧ (3 + n ≥ 2 * n) →
  Nat.choose (2 * n) (10 - 2 * n) + Nat.choose (3 + n) (2 * n) = 16 :=
by
  intro n h
  cases' h with h1 h2
  sorry

end combination_sum_eq_l3_3085


namespace correct_option_is_C_l3_3027

variable (a b : ℝ)

def option_A : Prop := (a - b) ^ 2 = a ^ 2 - b ^ 2
def option_B : Prop := a ^ 2 + a ^ 2 = a ^ 4
def option_C : Prop := (a ^ 2) ^ 3 = a ^ 6
def option_D : Prop := a ^ 2 * a ^ 2 = a ^ 6

theorem correct_option_is_C : option_C a :=
by
  sorry

end correct_option_is_C_l3_3027


namespace find_a_l3_3904

noncomputable def f (x : ℝ) : ℝ :=
  if -1 < x ∧ x < 0 then Real.sin (Real.pi * x^2)
  else Real.exp (x - 1)

theorem find_a (a : ℝ) (h_cond : f 1 + f a = 2) : 
  a = 1 ∨ a = -Real.sqrt 2 / 2 :=
by
  have h_f1 : f 1 = 1 := by sorry
  have h_fa : f a = 1 := by sorry
  cases lt_or_ge a 0 with h_neg h_nonneg,
  { -- Case: -1 < a < 0
    have h_apart : -1 < a ∧ a < 0 := by sorry
    rw [if_pos h_apart] at h_fa,
    rw Real.sin_eq_one_iff at h_fa,
    cases h_fa with k hk,
    by_cases h_even : k.even,
    { rw (Int.ofNat_le_iff).symm at hk,
      have : k = 1 := by sorry,
      rw this at hk,
      have : a^2 = 1 / 2 := by sorry,
      linarith only [this] },
    { exfalso, sorry } },
  { -- Case: a ≥ 0
    rw if_neg at h_fa,
    { have : a = 1 := by sorry,
      cc },
    { linarith only [h_nonneg] } }

end find_a_l3_3904


namespace train_passing_platform_time_l3_3432

theorem train_passing_platform_time :
  (500 : ℝ) / (50 : ℝ) > 0 →
  (500 : ℝ) + (500 : ℝ) / ((500 : ℝ) / (50 : ℝ)) = 100 := by
  sorry

end train_passing_platform_time_l3_3432


namespace value_of_Q_when_n_is_100_l3_3985

noncomputable def Q (n : ℕ) : ℚ := 
  ∏ k in Finset.range (n - 1) + 1, 2 * (1 - (1 / (k + 2)))

theorem value_of_Q_when_n_is_100 : Q 100 = 2^99 / 100 :=
by
  sorry

end value_of_Q_when_n_is_100_l3_3985


namespace find_principal_l3_3852

noncomputable def compound_interest (P r : ℝ) (n : ℝ) (t : ℝ) : ℝ := P * (1 + r / n)^(n * t)

theorem find_principal :
  let A := 1120
  let r := 0.0625
  let n := 2
  let t := 10/3
  P = 921.68 -> A = compound_interest P r n t :=
by
  let P := 921.68
  let A := 1120
  let r := 0.0625
  let n := 2
  let t := 10/3
  show A = compound_interest P r n t
  sorry

end find_principal_l3_3852


namespace inequality_l3_3253

noncomputable def a : ℝ := (1 / 3) ^ (2 / 5)
noncomputable def b : ℝ := 2 ^ (4 / 3)
noncomputable def c : ℝ := Real.log 1 / 3 / Real.log 2

theorem inequality : c < a ∧ a < b := 
by 
  sorry

end inequality_l3_3253


namespace residue_T_modulo_2020_l3_3262

theorem residue_T_modulo_2020 :
  let
    T := (Finset.range 2020).sum (λ n, if even n then - (n + 1) else (n + 1))
  in
  T % 2020 = 1010 :=
by
  sorry

end residue_T_modulo_2020_l3_3262


namespace triangle_area_l3_3433

theorem triangle_area (a b c : ℕ) (h₁ : a = 9) (h₂ : b = 40) (h₃ : c = 41) (h₄ : a^2 + b^2 = c^2) : 
  let A := (1 / 2 : ℚ) * a * b in
  A = 180 :=
by
  -- h₁, h₂, and h₃ are assumptions directly from the conditions
  rw [h₁, h₂, h₃]
  -- conclude the proof with sorry as actual steps are omitted
  sorry

end triangle_area_l3_3433


namespace period_T_min_k_bound_l3_3875

noncomputable def f₁ (x : ℝ) : ℝ := Real.sin x

noncomputable def f (n : ℕ) : (ℝ → ℝ) :=
  if n = 1 then f₁
  else (λ x, f (n - 1) x * (deriv (f (n - 1))) x)

def T (n : ℕ) : ℝ :=
  if n = 0 then 0  -- This shouldn’t really be necessary since n ≥ 1 in the problem.
  else π / 2^(n-2)

theorem period_T (n : ℕ) (h : 0 < n) : T n = π / 2^(n-2) :=
by
  sorry

theorem min_k_bound (n : ℕ) (h : 0 < n) : T 1 + T 2 + T 3 + ... + T n < 4 * π :=
by
  sorry

end period_T_min_k_bound_l3_3875


namespace corrected_variance_calculation_l3_3485

variables (x : ℕ → ℚ) (n : ℕ → ℕ)

def sample_data :=
    [(x 0, n 0), (x 1, n 1), (x 2, n 2)]

def transformed_data :=
    sample_data |> (λ (p : ℚ × ℕ), (100 * p.1, p.2))

noncomputable def sum_ni_ui :=
    transformed_data.foldr (λ p acc, p.2 * p.1 + acc) 0

noncomputable def sum_ni_ui_sq :=
    transformed_data.foldr (λ p acc, p.2 * p.1^2 + acc) 0

def sample_size : ℕ :=
    sample_data.foldr (λ p acc, p.2 + acc) 0

noncomputable def sample_variance :=
    (sum_ni_ui_sq - ((sum_ni_ui)^2 / sample_size)) / (sample_size - 1)

def corrected_sample_variance :=
    sample_variance / 10000

theorem corrected_variance_calculation :
    corrected_sample_variance x n = 0.0010844 :=
sorry

end corrected_variance_calculation_l3_3485


namespace complex_number_quadrant_l3_3580

theorem complex_number_quadrant (z : ℂ) (h : z * (1 + complex.I) = 1 - 2 * complex.I) : 
  z.re < 0 ∧ z.im < 0 :=
sorry

end complex_number_quadrant_l3_3580


namespace teddy_hamburgers_l3_3661

def total_spending_both := 106
def robert_boxes_of_pizza := 5
def cost_per_pizza := 10
def robert_soft_drinks := 10
def cost_per_soft_drink := 2
def teddy_hamburgers_cost := 3
def teddy_soft_drinks := 10

def robert_total_spending := robert_boxes_of_pizza * cost_per_pizza + robert_soft_drinks * cost_per_soft_drink
def teddy_total_spending := total_spending_both - robert_total_spending

theorem teddy_hamburgers :
  let x := teddy_total_spending - teddy_soft_drinks * cost_per_soft_drink
  in x / teddy_hamburgers_cost = 5 :=
by
  sorry

end teddy_hamburgers_l3_3661


namespace smallest_number_is_27_l3_3698

theorem smallest_number_is_27 (a b c : ℕ) (h_mean : (a + b + c) / 3 = 30) (h_median : b = 28) (h_largest : c = b + 7) : a = 27 :=
by {
  sorry
}

end smallest_number_is_27_l3_3698


namespace shoelace_quadrilateral_area_l3_3016

theorem shoelace_quadrilateral_area :
  let vertices := [(4, 0), (0, 5), (3, 4), (10, 10)]
  let area := (1 / 2 : ℝ) * (| 
    (vertices[0].1 * vertices[1].2 + vertices[1].1 * vertices[2].2 + vertices[2].1 * vertices[3].2 + vertices[3].1 * vertices[0].2) - 
    (vertices[1].1 * vertices[0].2 + vertices[2].1 * vertices[1].2 + vertices[3].1 * vertices[2].2 + vertices[0].1 * vertices[3].2) |) in
  area = 22.5 :=
by 
  sorry

end shoelace_quadrilateral_area_l3_3016


namespace isosceles_triangle_points_count_l3_3051

theorem isosceles_triangle_points_count :
  let geoboard := { (x, y) : ℕ × ℕ | x < 7 ∧ y < 7 }
  let segment_CD := [(2, 3), (5, 3)]
  ∀ (E : ℕ × ℕ), E ∈ geoboard ∧ E ≠ (2, 3) ∧ E ≠ (5, 3) →
  let triangle_CDE_is_isosceles := 
    dist (2, 3) E = dist (5, 3) E ∨ dist (2, 3) E = 3 ∨ dist (5, 3) E = 3 in
  { E ∈ geoboard | triangle_CDE_is_isosceles }.card = 11 :=
by
  sorry

end isosceles_triangle_points_count_l3_3051


namespace cross_product_scalar_multiplication_l3_3568

variables (a b : ℝ^3)

theorem cross_product_scalar_multiplication :
  a × b = ⟨-3, 2, 6⟩ → 
  a × (5 • b) = ⟨-15, 10, 30⟩ :=
by
  sorry

end cross_product_scalar_multiplication_l3_3568


namespace volume_of_intersection_of_octahedra_l3_3364

theorem volume_of_intersection_of_octahedra : 
  let S := {p : ℝ × ℝ × ℝ | |p.1| + |p.2| + |p.3| ≤ 2}
  let T := {p : ℝ × ℝ × ℝ | |p.1| + |p.2| + |p.3 - 2| ≤ 2}
  volume (S ∩ T) = 4 / 3 := 
sorry

end volume_of_intersection_of_octahedra_l3_3364


namespace number_of_pecan_pies_is_4_l3_3404

theorem number_of_pecan_pies_is_4 (apple_pies pumpkin_pies total_pies pecan_pies : ℕ) 
  (h1 : apple_pies = 2) 
  (h2 : pumpkin_pies = 7) 
  (h3 : total_pies = 13) 
  (h4 : pecan_pies = total_pies - (apple_pies + pumpkin_pies)) 
  : pecan_pies = 4 := 
by 
  sorry

end number_of_pecan_pies_is_4_l3_3404


namespace range_of_abscissa_of_P_l3_3532

noncomputable def point_lies_on_line (P : ℝ × ℝ) : Prop :=
  P.1 - P.2 + 1 = 0

noncomputable def point_lies_on_circle_c (M N : ℝ × ℝ) : Prop :=
  (M.1 - 2)^2 + (M.2 - 1)^2 = 1 ∧ (N.1 - 2)^2 + (N.2 - 1)^2 = 1

noncomputable def angle_mpn_eq_60 (P M N : ℝ × ℝ) : Prop :=
  true -- This is a placeholder because we have to define the geometrical angle condition which is complex.

theorem range_of_abscissa_of_P :
  ∀ (P M N : ℝ × ℝ),
  point_lies_on_line P →
  point_lies_on_circle_c M N →
  angle_mpn_eq_60 P M N →
  0 ≤ P.1 ∧ P.1 ≤ 2 := sorry

end range_of_abscissa_of_P_l3_3532


namespace minimal_marked_cells_theorem_l3_3989

-- Definitions: Let n be an even positive integer.
def n (k : Nat) : Nat := 2 * k

-- The minimal number of cells that must be marked.
def minimal_marked_cells (k : Nat) : Nat := k * (k + 1)

-- Main theorem: The minimal number of cells on an n x n board that must be marked
-- so that every cell has a marked neighbor is k(k+1), where n = 2k.
theorem minimal_marked_cells_theorem (k : Nat) (hk : k > 0) :
  ∃ m, m = minimal_marked_cells k ∧ 
    ∀ (i j : Nat), i < n k ∧ j < n k → 
    ∃ (a b : Nat), a < n k ∧ b < n k ∧ (a = i ∧ (b = j + 1 ∨ b = j - 1) ∨ b = j ∧ (a = i + 1 ∨ a = i - 1)) ∧ 
    marked a b m :=
begin
  sorry
end

end minimal_marked_cells_theorem_l3_3989


namespace sin_cos_product_pos_l3_3924

variables {θ : ℝ}

theorem sin_cos_product_pos 
  (h1 : sin θ * tan θ > 0) 
  (h2 : cos θ * tan θ < 0) :
  sin θ * cos θ > 0 := 
sorry

end sin_cos_product_pos_l3_3924


namespace probability_two_correct_positions_sum_mn_l3_3294

noncomputable def permutations (l : List ℕ) : List (List ℕ) := 
List.permutations l

theorem probability_two_correct_positions_sum_mn (d1 d2 d3 d4 : ℕ)
  (h : d1 ≠ d2 ∧ d2 ≠ d3 ∧ d3 ≠ d4 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d4) :
  ∃ (m n : ℕ), (m + n = 5) :=
by
  let digits := [d1, d2, d3, d4]
  let perms := permutations digits
  have h_total_perms : perms.length = 24 := by sorry
  have h_two_correct : (∃ p ∈ perms, ∑ i, if p[i] = digits[i] then 1 else 0 = 2) := by sorry
  have h_probability : (6 : ℚ) / 24 = 1 / 4 := by norm_num
  have h_m_n := h_probability -- given that m = 1 and n = 4
  existsi 1, 4
  trivial

end probability_two_correct_positions_sum_mn_l3_3294


namespace proof_result_l3_3733

noncomputable def direction_vector (a b : ℝ × ℝ × ℝ) := 
  ∃ k : ℝ, (k • b) = a

noncomputable def perpendicular_vector (a b : ℝ × ℝ × ℝ) := 
  (a.1 * b.1 + a.2 * b.2 + a.3 * b.3) = 0

def parallel_line_condition := 
  direction_vector (2, 3, -1) (-2, -3, 1) 

def perpendicular_plane_condition := 
  perpendicular_vector (2, 2, -1) (-3, 4, 2)

theorem proof_result: parallel_line_condition ∧ perpendicular_plane_condition :=
by
  have h1 : parallel_line_condition := sorry
  have h2 : perpendicular_plane_condition := sorry
  exact ⟨h1, h2⟩

end proof_result_l3_3733


namespace arithmetic_sequence_ratio_l3_3149

-- Given an arithmetic sequence {a_n} with the sum of the first n terms denoted as S_n,
constant a : ℕ → ℝ
constant S : ℕ → ℝ

-- Condition: S_8 = 2 * S_4
axiom h1 : S 8 = 2 * S 4

-- Theorem: Prove that a_3 / a_1 = 1
theorem arithmetic_sequence_ratio : a 3 / a 1 = 1 :=
sorry

end arithmetic_sequence_ratio_l3_3149


namespace incorrect_statement_g2_l3_3930

def g (x : ℚ) : ℚ := (2 * x + 3) / (x - 2)

theorem incorrect_statement_g2 : g 2 ≠ 0 := by
  sorry

end incorrect_statement_g2_l3_3930


namespace game_strategy_winner_l3_3003

theorem game_strategy_winner
  (initial_balls : ℕ)
  (can_draw : ℕ → Prop)
  (turn : ℕ → bool)
  (smart_players : Prop)
  (draw_range : ∀ n, can_draw n ↔ 1 ≤ n ∧ n ≤ 5)
  (initial_turn : turn 0 = tt)
  (final_turn : ∀ n, (n < 6) → ¬can_draw n):
  initial_balls = 100 → smart_players → turn 0 = tt → (∃ turns, turn 100 = tt) :=
by
  sorry

end game_strategy_winner_l3_3003


namespace points_total_l3_3208

/--
In a game, Samanta has 8 more points than Mark,
and Mark has 50% more points than Eric. Eric has 6 points.
How many points do Samanta, Mark, and Eric have in total?
-/
theorem points_total (Samanta Mark Eric : ℕ)
  (h1 : Samanta = Mark + 8)
  (h2 : Mark = Eric + Eric / 2)
  (h3 : Eric = 6) :
  Samanta + Mark + Eric = 32 := by
  sorry

end points_total_l3_3208


namespace inequality_proof_l3_3627

theorem inequality_proof 
  (n : ℕ) 
  (h_pos : 0 < n) 
  (x : ℕ → ℝ) 
  (h_sorted : ∀ i j : ℕ, 1 ≤ i → i ≤ n → 1 ≤ j → j ≤ n → i ≤ j → x i ≤ x j) : 
  (∑ i in finset.range n, ∑ j in finset.range n, |x (i + 1) - x (j + 1)|)^2 
  ≤ (2 * (n^2 - 1) / 3) * ∑ i in finset.range n, ∑ j in finset.range n, (x (i + 1) - x (j + 1))^2 :=
by
  sorry

end inequality_proof_l3_3627


namespace g_value_l3_3199

def f (x : ℝ) : ℝ := (1 - 2^x) / (1 + 2^x)
def g (y : ℝ) : ℝ := if h : ∃ x, f x = y then Classical.choose h else y -- Using classical logic to define g as inverse

theorem g_value (h : g (2 / 7) = ℤ) : g (2 / 7) = Real.log 5 / Real.log 2 - Real.log 9 / Real.log 2 := by
    -- Replace with the correct proof steps
    sorry

end g_value_l3_3199


namespace triangle_equilateral_l3_3577

noncomputable def omega : ℂ := - (1/2 : ℂ) + (complex.I) * (real.sqrt 3 / 2)

theorem triangle_equilateral
  (a b c : ℂ)
  (h_neq : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_omega : omega = - (1/2 : ℂ) + (complex.I) * (real.sqrt 3 / 2))
  (h_eq : a + omega * b + (ω^2) * c = 0) : 
  ∃ A B C : ℂ, A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ (triangle_type A B C = equilateral) :=
sorry

end triangle_equilateral_l3_3577


namespace number_of_zeros_h_g_less_than_linear_l3_3634

noncomputable def f (x : ℝ) : ℝ := max (x^2 - 1) (2 * Real.log x)
noncomputable def g (x a : ℝ) : ℝ := max (x + Real.log x) (-x^2 + (a^2 - 1/2) * x + 2 * a^2 + 4 * a)
noncomputable def h (x : ℝ) : ℝ := f x - 3 * (x - 1/2) * (x - 1)^2

theorem number_of_zeros_h :
  (∃ x1 x2 ∈ Ioo (0 : ℝ) 1, h x1 = 0 ∧ h x2 = 0 ∧ x1 ≠ x2) :=
sorry

theorem g_less_than_linear (a : ℝ) :
  (∃ a ∈ Ioo ((Real.log 2 - 1)/4) 2, ∀ x ∈ Ioi (a + 2), g x a < (3/2) * x + 4 * a) :=
sorry

end number_of_zeros_h_g_less_than_linear_l3_3634


namespace count_correct_propositions_l3_3070

theorem count_correct_propositions :
  let p1 := ∀ a b : ℝ, (a + b ≠ 6) → (a ≠ 3 ∨ b ≠ 3)
  let p2 := ∀ p q : Prop, (p ∨ q) → (p ∧ q)
  let p3 := (¬∀ a b : ℝ, a^2 + b^2 ≥ 2 * (a - b - 1)) ↔ (∃ a b : ℝ, a^2 + b^2 ≤ 2 * (a - b - 1))
  (count_true [p1, p2, p3] == 1) :=
sorry

end count_correct_propositions_l3_3070


namespace tyler_meals_l3_3708

-- Definitions for the meat, vegetables, and dessert choices
def meats := 4
def vegetables := 5
def desserts := 3

-- Conditions derived from the problem statement
def chosen_meats := nat.choose meats 2
def chosen_vegetables := nat.choose vegetables 3
def chosen_desserts := desserts

-- Lean 4 statement to prove the total number of different meals
theorem tyler_meals : 
  chosen_meats * chosen_vegetables * chosen_desserts = 180 :=
by
  sorry

end tyler_meals_l3_3708


namespace marketing_firm_surveyed_households_l3_3775

theorem marketing_firm_surveyed_households :
  ∃ (total_households : ℕ),
  let neither := 80 in
  let only_A := 60 in
  let both := 10 in
  let ratio_B_to_both := 3 in
  let only_B := ratio_B_to_both * both in
  total_households = neither + only_A + both + only_B ∧ total_households = 200 :=
begin
  sorry
end

end marketing_firm_surveyed_households_l3_3775


namespace find_m_for_parallel_vectors_l3_3638

def vec_parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

theorem find_m_for_parallel_vectors :
  ∀ (m : ℝ), vec_parallel (m, 1) (1 + m, 3) → m = 1 / 2 :=
begin
  intro m,
  intro h,
  sorry
end

end find_m_for_parallel_vectors_l3_3638


namespace quadratic_square_binomial_l3_3470

theorem quadratic_square_binomial (c : ℝ) : 
  (x : ℝ) → (x^2 + 10 * x + c = (x + 5)^2) ↔ (c = 25) :=
begin
  sorry
end

end quadratic_square_binomial_l3_3470


namespace probability_spade_then_ace_l3_3342

theorem probability_spade_then_ace :
  let total_cards := 52
  let total_aces := 4
  let total_spades := 13
  let ace_of_spades := 1
  let non_ace_spades := total_spades - ace_of_spades
  (non_ace_spades / total_cards) * (total_aces / (total_cards - 1)) +
  (ace_of_spades / total_cards) * ((total_aces - ace_of_spades) / (total_cards - 1)) = (1 / 52) :=
by
  sorry

end probability_spade_then_ace_l3_3342


namespace arithmetic_seq_sum_l3_3595

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h : a 5 + a 6 + a 7 = 1) : a 3 + a 9 = 2 / 3 :=
sorry

end arithmetic_seq_sum_l3_3595


namespace circle_center_l3_3501

-- Define the conditions
variable (a : ℝ)

def circle_equation (a : ℝ) (x y : ℝ) : ℝ :=
  a^2 * x^2 + (a + 2) * y^2 + 4 * x + 8 * y + 5 * a

theorem circle_center (a : ℝ) 
    (h₁ : a^2 = a + 2)
    (h₂ : a ≠ 0)
    (h_circle : ∃ x y : ℝ, circle_equation a x y = 0) :
    a = -1 → ∃ c₁ c₂ : ℝ, (c₁, c₂) = (-2, -4) :=
by
  intros ha
  use [-2, -4]
  sorry

end circle_center_l3_3501


namespace evaluate_log8_64_l3_3478

noncomputable def log8_64 : ℝ := Real.logBase 8 64

theorem evaluate_log8_64 : log8_64 = 2 := by
  sorry

end evaluate_log8_64_l3_3478


namespace correct_statements_l3_3072

theorem correct_statements (s1 s2 s3 s4 : Prop)
  (h1 : ∀ (σ^2 : ℝ), ∀ (c : ℝ), σ^2 = σ^2 + c)
  (h2 : ∀ (x : ℝ), let y := 3 - 5 * x in y ≠ y + 5)
  (h3 : ∀ (r : ℝ), | r | < 1 → ¬ (| r | = 0))
  (h4 : ∀ (K^2 : ℝ), (K^2 > 0) → true) :
  (s1 = true) ∧ (s4 = true) ∧ ¬ (s2 = true) ∧ ¬ (s3 = true) :=
by
  sorry

end correct_statements_l3_3072


namespace compare_volumes_l3_3339

variables (a b c : ℝ) -- dimensions of the second block
variables (a1 b1 c1 : ℝ) -- dimensions of the first block

-- Length, width, and height conditions
def length_cond := a1 = 1.5 * a
def width_cond := b1 = 0.8 * b
def height_cond := c1 = 0.7 * c

-- Volumes of the blocks
def V1 := a1 * b1 * c1 -- Volume of the first block
def V2 := a * b * c -- Volume of the second block

-- Main theorem
theorem compare_volumes (h1 : length_cond) (h2 : width_cond) (h3 : height_cond) :
  V2 = (25/21) * V1 :=
sorry

end compare_volumes_l3_3339


namespace meet_at_eleven_l3_3651

noncomputable def meeting_time (d U V : ℝ) (t : ℝ) : ℝ := 
  if U = d / 6 ∧ V = d / 3 ∧ t = 2 then 9 + t else 0

theorem meet_at_eleven 
  (d U V : ℝ) (t : ℝ) 
  (h₀ : Fedya_left_at_nine : True)   
  (h₁ : Vera_left_at_nine : True)  
  (h₂ : distance_fedya_covered : U * t = d / 3)
  (h₃ : time_vara_covered : V * t = 2 * d / 3)
  (h₄ : earlier_departure_condition : U * (t + 1) = d / 2)
  : meeting_time d U V t = 11 :=
by 
  sorry

end meet_at_eleven_l3_3651


namespace total_distance_covered_l3_3418

theorem total_distance_covered :
  let speed_upstream := 12 -- km/h
  let time_upstream := 2 -- hours
  let speed_downstream := 38 -- km/h
  let time_downstream := 1 -- hour
  let distance_upstream := speed_upstream * time_upstream
  let distance_downstream := speed_downstream * time_downstream
  distance_upstream + distance_downstream = 62 := by
  sorry

end total_distance_covered_l3_3418


namespace rectangle_width_length_ratio_l3_3225

theorem rectangle_width_length_ratio
  (w l P : ℕ)
  (hl : l = 10)
  (hP : P = 30)
  (hp : P = 2 * l + 2 * w) :
  Nat.gcd w l = 1 →
  (w : l) = (1 : 2) :=
by
  intro h_gcd
  apply sorry

end rectangle_width_length_ratio_l3_3225


namespace sum_last_two_digits_l3_3356

theorem sum_last_two_digits (a b : ℕ) (h1 : a = 8) (h2 : b = 12) :
  let x := a ^ 50
      y := b ^ 50 in
  ((x + y) % 100) = 48 :=
by
  sorry

end sum_last_two_digits_l3_3356


namespace cos_theta_correct_projection_correct_l3_3556

noncomputable def vec_a : ℝ × ℝ := (2, 3)
noncomputable def vec_b : ℝ × ℝ := (-2, 4)

noncomputable def cos_theta (a b : ℝ × ℝ) : ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let norm_a := Real.sqrt (a.1 ^ 2 + a.2 ^ 2)
  let norm_b := Real.sqrt (b.1 ^ 2 + b.2 ^ 2)
  dot_product / (norm_a * norm_b)

noncomputable def projection (b : ℝ × ℝ) (cosθ : ℝ) : ℝ :=
  let norm_b := Real.sqrt (b.1 ^ 2 + b.2 ^ 2)
  norm_b * cosθ

theorem cos_theta_correct :
  cos_theta vec_a vec_b = 4 * Real.sqrt 65 / 65 :=
by
  sorry

theorem projection_correct :
  projection vec_b (cos_theta vec_a vec_b) = 8 * Real.sqrt 13 / 13 :=
by
  sorry

end cos_theta_correct_projection_correct_l3_3556


namespace unique_real_x_l3_3462

theorem unique_real_x (x : ℝ) : sqrt (-(2 * x - 3) ^ 2) = 0 ↔ x = 3 / 2 :=
by
  sorry

end unique_real_x_l3_3462


namespace quadratic_has_exactly_one_zero_in_interval_l3_3571

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a * x + 1

theorem quadratic_has_exactly_one_zero_in_interval (a : ℝ) (h : a > 3) :
  ∃! x : ℝ, 0 < x ∧ x < 2 ∧ f a x = 0 :=
begin
  sorry
end

end quadratic_has_exactly_one_zero_in_interval_l3_3571


namespace new_code_correct_l3_3582

noncomputable def newDigit (d : Nat) : Nat := 9 - d

def applyRule (code : String) : String :=
  let codeDigits := code.toList.map (λ c => c.toNat - '0'.toNat)
  let transformedDigits :=
    List.zipWith3 (λ t h u => t*1000 + h*100 + u*10 + newDigit (u % 10))
    (codeDigits.indices.map (λ i => if i % 4 == 0 then codeDigits.get! i else 0))  -- thousands
    (codeDigits.indices.map (λ i => if i % 4 == 1 then newDigit (codeDigits.get! i) else 0))  -- hundreds
    (codeDigits.indices.map (λ i => if i % 4 == 2 then codeDigits.get! i else 0))  -- tens
    (codeDigits.indices.map (λ i => if i % 4 == 3 then newDigit (codeDigits.get! i) else 0))  -- units
  transformedDigits.map (λ d => (d + '0'.toNat).toChar).asString

theorem new_code_correct :
  (applyRule "4022" = "4927") ∧
  (applyRule "0710" = "0219") ∧
  (applyRule "4199" = "4890") :=
by
  -- proofs for individual segments
  sorry

end new_code_correct_l3_3582


namespace total_distance_in_12_hours_l3_3691

def d : ℕ → ℕ
| 1       := 30
| (n + 1) := d n + 2

theorem total_distance_in_12_hours : (∑ n in Finset.range 12, d (n + 1)) = 492 := 
sorry

end total_distance_in_12_hours_l3_3691


namespace option_c_correct_l3_3367

theorem option_c_correct (a b : ℝ) : (a * b^2)^2 = a^2 * b^4 := by
  sorry

end option_c_correct_l3_3367


namespace function_has_one_zero_l3_3583

-- Define the function f
def f (x m : ℝ) : ℝ := (m - 1) * x^2 + 2 * (m + 1) * x - 1

-- State the theorem
theorem function_has_one_zero (m : ℝ) :
  (∃! x : ℝ, f x m = 0) ↔ m = 0 ∨ m = -3 := 
sorry

end function_has_one_zero_l3_3583


namespace number_of_planes_l3_3333

-- Define our points and the condition that no three points are collinear.
structure Points (α : Type) :=
(x : α) (y : α) (z : α) (w : α)

-- Assume all four points are non-collinear, which implies they form a tetrahedron.
noncomputable def nonCollinear {α : Type} [LinearOrderedField α] (p : Points α) : Prop :=
¬ ∃ (a b c : α), 
  ({a, b, c} : finset _).card = 3 ∧ 
  (p.x = a ∨ p.x = b ∨ p.x = c) ∧ 
  (p.y = a ∨ p.y = b ∨ p.y = c) ∧ 
  (p.z = a ∨ p.z = b ∨ p.z = c) ∧ 
  (p.w = a ∨ p.w = b ∨ p.w = c)

-- The theorem statement that there are exactly 4 planes defined by any three out of these four points.
theorem number_of_planes {α : Type} [LinearOrderedField α] (p : Points α) (h : nonCollinear p) : 
  ∃! (planeSet : finset (finset α)), 
  (planeSet.card = 4) ∧ 
  ∀ (pl: finset α), pl ∈ planeSet → (pl.card = 3) :=
sorry

end number_of_planes_l3_3333


namespace integral_f_l3_3169

def f (x : ℝ) : ℝ :=
  if x ≥ -2 ∧ x ≤ 0 then x^2 else
  if x > 0 ∧ x ≤ 2 then x + 1 else 0

theorem integral_f : ∫ x in -2..2, f x = 20 / 3 :=
by sorry

end integral_f_l3_3169


namespace net_population_change_is_minus_8_percent_l3_3329

variable (initial_population : ℝ)

def year_1_increase : ℝ := initial_population * (6 / 5)
def year_2_increase : ℝ := year_1_increase * (6 / 5)
def year_3_decrease : ℝ := year_2_increase * (4 / 5)
def year_4_decrease : ℝ := year_3_decrease * (4 / 5)

-- Statement of the proof problem
theorem net_population_change_is_minus_8_percent :
  (year_4_decrease / initial_population - 1) * 100 = -8 :=
by sorry

end net_population_change_is_minus_8_percent_l3_3329


namespace exterior_angle_regular_octagon_l3_3211

theorem exterior_angle_regular_octagon : 
  ∀ (interior_angle : ℝ) (exterior_angle : ℝ), 
  (∀ (n : ℕ), n = 8 → interior_angle = 135) → exterior_angle = 180 - interior_angle → exterior_angle = 45 :=
by
  intros interior_angle exterior_angle h1 h2
  rw h1
  rw h2
  sorry

end exterior_angle_regular_octagon_l3_3211


namespace area_of_triangle_formed_by_tangent_line_l3_3902
-- Import necessary libraries from Mathlib

-- Set up the problem
theorem area_of_triangle_formed_by_tangent_line
  (f : ℝ → ℝ) (h_f : ∀ x, f x = x^2) :
  let slope := (deriv f 1)
  let tangent_line (x : ℝ) := slope * (x - 1) + f 1
  let x_intercept := (0 : ℝ)
  let y_intercept := tangent_line 0
  let area := 0.5 * abs x_intercept * abs y_intercept
  area = 1 / 4 :=
by
  sorry -- Proof to be completed

end area_of_triangle_formed_by_tangent_line_l3_3902


namespace area_of_rhombus_is_110_l3_3755

-- Define the length of the diagonals
def d1 : ℝ := 11
def d2 : ℝ := 20

-- Define the formula for the area of the rhombus
def area_of_rhombus (d1 d2 : ℝ) : ℝ :=
  (d1 * d2) / 2

-- State the theorem
theorem area_of_rhombus_is_110 :
  area_of_rhombus d1 d2 = 110 :=
by
  -- skip the proof
  sorry

end area_of_rhombus_is_110_l3_3755


namespace sum_of_rational_roots_eq_two_l3_3491
noncomputable def h (x : ℚ) : ℚ := x^3 - 6 * x^2 + 9 * x - 2

theorem sum_of_rational_roots_eq_two : 
  (∑ root in {root | h(root) = 0 ∧ root ∈ ℚ}, root) = 2 :=
sorry

end sum_of_rational_roots_eq_two_l3_3491


namespace telescoping_series_sum_l3_3819

open BigOperators

theorem telescoping_series_sum :
  ∑ n in Finset.range 500 | 1 ≤ n, (1 / (n^2 + 2 * n)) = 3 / 4 :=
by
  sorry

end telescoping_series_sum_l3_3819


namespace gcf_154_252_l3_3351

theorem gcf_154_252 : ∃ g : ℕ, g = Nat.gcd 154 252 ∧ g = 14 :=
by
  use 14
  have h1 : Nat.gcd 154 252 = 14 := by sorry
  show 14 = Nat.gcd 154 252
  exact h1

end gcf_154_252_l3_3351


namespace min_platforms_needed_l3_3943

theorem min_platforms_needed :
  let slabs_7_tons := 120
  let slabs_9_tons := 80
  let weight_7_tons := 7
  let weight_9_tons := 9
  let max_weight_per_platform := 40
  let total_weight := slabs_7_tons * weight_7_tons + slabs_9_tons * weight_9_tons
  let platforms_needed_per_7_tons := slabs_7_tons / 3
  let platforms_needed_per_9_tons := slabs_9_tons / 2
  platforms_needed_per_7_tons = 40 ∧ platforms_needed_per_9_tons = 40 ∧ 3 * platforms_needed_per_7_tons = slabs_7_tons ∧ 2 * platforms_needed_per_9_tons = slabs_9_tons →
  platforms_needed_per_7_tons = 40 ∧ platforms_needed_per_9_tons = 40 :=
by
  sorry

end min_platforms_needed_l3_3943


namespace sin_alpha_value_l3_3929

theorem sin_alpha_value (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : cos (α + π / 3) = -sqrt 3 / 3) :
  sin α = (sqrt 6 + 3) / 6 :=
sorry

end sin_alpha_value_l3_3929


namespace exists_nonneg_poly_div_l3_3494

theorem exists_nonneg_poly_div (P : Polynomial ℝ) 
  (hP_pos : ∀ x : ℝ, x > 0 → P.eval x > 0) :
  ∃ (Q R : Polynomial ℝ), (∀ n, Q.coeff n ≥ 0) ∧ (∀ n, R.coeff n ≥ 0) ∧ (P = Q / R) := 
sorry

end exists_nonneg_poly_div_l3_3494


namespace gcd_factorials_l3_3496

noncomputable def factorial : ℕ → ℕ
| 0     => 1
| (n+1) => (n+1) * factorial n

theorem gcd_factorials (n m : ℕ) (hn : n = 8) (hm : m = 10) :
  Nat.gcd (factorial n) (factorial m) = 40320 := by
  sorry

end gcd_factorials_l3_3496


namespace passes_through_fixed_point_l3_3545

variable (a : ℝ) (h : a ≠ 0)

def f (x : ℝ) : ℝ := a^(x-2) + 3

theorem passes_through_fixed_point : f a 2 = 4 := by
  -- sorry indicates the proof is omitted
  sorry

end passes_through_fixed_point_l3_3545


namespace find_a8_l3_3537

-- Define the arithmetic sequence and the given conditions
variable {α : Type} [AddCommGroup α] [MulAction ℤ α]

def is_arithmetic_sequence (a : ℕ → α) := ∃ d, ∀ n, a (n + 1) = a n + d

-- Given conditions
variables {a : ℕ → ℝ}
axiom h1 : is_arithmetic_sequence a
axiom h2 : a 5 + a 6 = 22
axiom h3 : a 3 = 7

theorem find_a8 : a 8 = 15 :=
by
  -- Proof omitted
  sorry

end find_a8_l3_3537


namespace position_relationship_correct_l3_3737

noncomputable def problem_data :=
  (((2:ℝ), 3, -1), ((-2:ℝ), -3, 1), ((1:ℝ), -1, 2), ((6:ℝ), 4, -1), 
   ((2:ℝ), 2, -1), ((-3:ℝ), 4, 2), ((0:ℝ), 3, 0), ((0:ℝ), -5, 0))

theorem position_relationship_correct :
  let ⟨a, b, a2, u2, u3, v3, a4, u4⟩ := problem_data in
  (a = ((-1):ℝ) • b ∧
  (∀ k : ℝ, a2 ≠ k • u2) ∧
  (u3 ⬝ v3 = 0 ∧
  ¬ (a4 = (0:ℝ) • u4 ∧ a4 = (-3/5:ℝ) • u4 ))) :=
sorry

end position_relationship_correct_l3_3737


namespace sum_of_sequence_l3_3866

-- Definitions of sequence and difference sequence conditions
def a : ℕ → ℕ
| 0     := 2
| (n+1) := a n + 2^n

-- Sum of the first n terms of sequence a
def S (n : ℕ) : ℕ := ∑ i in Finset.range n, a i

-- Theorem statement to prove
theorem sum_of_sequence (n : ℕ) : S n = 2^(n+1) - 2 :=
by sorry

end sum_of_sequence_l3_3866


namespace max_quotient_is_20_l3_3107

theorem max_quotient_is_20 : 
  ∃ a b ∈ ({-15, -5, -4, 1, 5, 20} : Set ℤ), 
    (∀ x y ∈ ({-15, -5, -4, 1, 5, 20} : Set ℤ), (x : ℚ) / y ≤ a / b) 
    ∧ a / b = 20 :=
by 
  sorry

end max_quotient_is_20_l3_3107


namespace count_integer_n_for_integer_expression_l3_3887

theorem count_integer_n_for_integer_expression : 
  let i := Complex.I in
  i^2 = -1 →
  card { n : ℤ | |n| ≤ 5 ∧ Im ((n + i)^5) = 0 } = 5 :=
begin
  sorry
end

end count_integer_n_for_integer_expression_l3_3887


namespace proof_result_l3_3730

noncomputable def direction_vector (a b : ℝ × ℝ × ℝ) := 
  ∃ k : ℝ, (k • b) = a

noncomputable def perpendicular_vector (a b : ℝ × ℝ × ℝ) := 
  (a.1 * b.1 + a.2 * b.2 + a.3 * b.3) = 0

def parallel_line_condition := 
  direction_vector (2, 3, -1) (-2, -3, 1) 

def perpendicular_plane_condition := 
  perpendicular_vector (2, 2, -1) (-3, 4, 2)

theorem proof_result: parallel_line_condition ∧ perpendicular_plane_condition :=
by
  have h1 : parallel_line_condition := sorry
  have h2 : perpendicular_plane_condition := sorry
  exact ⟨h1, h2⟩

end proof_result_l3_3730


namespace num_students_left_l3_3063

variable (Joe_weight : ℝ := 45)
variable (original_avg_weight : ℝ := 30)
variable (new_avg_weight : ℝ := 31)
variable (final_avg_weight : ℝ := 30)
variable (diff_avg_weight : ℝ := 7.5)

theorem num_students_left (n : ℕ) (x : ℕ) (W : ℝ := n * original_avg_weight)
  (new_W : ℝ := W + Joe_weight) (A : ℝ := Joe_weight - diff_avg_weight) : 
  new_W = (n + 1) * new_avg_weight →
  W + Joe_weight - x * A = (n + 1 - x) * final_avg_weight →
  x = 2 :=
by
  sorry

end num_students_left_l3_3063


namespace intervals_of_monotonicity_range_of_values_for_a_g_has_two_zeros_l3_3170

-- Definition of the function f
def f (x : ℝ) (a : ℝ) : ℝ := (2 / x) + a * Real.log x - 2

-- Conditions
variable {a : ℝ} (ha : a > 0) {b : ℝ}
variable {x : ℝ} (hx : x > 0)

-- 1. Intervals of monotonicity
theorem intervals_of_monotonicity (hx : x ≠ 1) (ha1 : a = 1) : 
  (∀ x, f x a > f x 1 -> x > 2) ∧ 
  (∀ x, f x a < f x 1 -> 0 < x ∧ x < 2) := 
sorry

-- 2. Range of values for a
theorem range_of_values_for_a : 
  (∀ x, f x a > 2 * (a - 1) -> 0 < a ∧ a < 2 / Real.exp 1) := 
sorry

-- 3. Range of values for b given g has two zeros in [e^{-1}, e]
theorem g_has_two_zeros (ha1 : a = 1) (hx : e⁻¹ ≤ x ∧ x ≤ e) :
  (∀ x, f x 1 + x - b > 0 -> 1 < b ≤ 2 / Real.exp 1 + Real.exp 1 - 1) := 
sorry

end intervals_of_monotonicity_range_of_values_for_a_g_has_two_zeros_l3_3170


namespace cross_product_scalar_multiplication_l3_3569

variables (a b : ℝ^3)

theorem cross_product_scalar_multiplication :
  a × b = ⟨-3, 2, 6⟩ → 
  a × (5 • b) = ⟨-15, 10, 30⟩ :=
by
  sorry

end cross_product_scalar_multiplication_l3_3569


namespace modulo_residue_l3_3261

theorem modulo_residue (T : ℤ) (h : T = ∑ i in (range 2020), (-1)^i * (i+1)) : T % 2020 = 0 :=
sorry

end modulo_residue_l3_3261


namespace income_ratio_l3_3322

variable (U B: ℕ) -- Uma's and Bala's incomes
variable (x: ℕ)  -- Common multiplier for expenditures
variable (savings_amt: ℕ := 2000)  -- Savings amount for both
variable (ratio_expenditure_uma : ℕ := 7)
variable (ratio_expenditure_bala : ℕ := 6)
variable (uma_income : ℕ := 16000)
variable (bala_expenditure: ℕ)

-- Conditions of the problem
-- Uma's Expenditure Calculation
axiom ua_exp_calc : savings_amt = uma_income - ratio_expenditure_uma * x
-- Bala's Expenditure Calculation
axiom bala_income_calc : savings_amt = B - ratio_expenditure_bala * x

theorem income_ratio (h1: U = uma_income) (h2: B = bala_expenditure):
  U * ratio_expenditure_bala = B * ratio_expenditure_uma :=
sorry

end income_ratio_l3_3322


namespace count_positive_integers_l3_3560

def is_divisible_by (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def count_divisibles (n m : ℕ) : ℕ :=
  (list.range (n + 1)).count (λ x, is_divisible_by x m)

theorem count_positive_integers (n : ℕ) :
  let divisible_by_2_and_3_not_5 := count_divisibles n 6 - count_divisibles n 30
  divisible_by_2_and_3_not_5 1000 = 133 :=
by
  simp [is_divisible_by, count_divisibles]
  sorry

end count_positive_integers_l3_3560


namespace expected_replanted_seeds_l3_3861

open MeasureTheory ProbabilityTheory

-- Define the conditions
def probability_of_germination : ℝ := 0.9
def total_seeds : ℕ := 1000
def seeds_to_replant_per_failure : ℕ := 2

-- Define the binomial distribution and the expected value for failures
noncomputable def probability_of_failure : ℝ := 1 - probability_of_germination
noncomputable def number_of_failures : BinomialDistribution where
  n := total_seeds
  p := probability_of_failure

-- Main theorem: Expected number of replanted seeds
theorem expected_replanted_seeds : 
  (2 * (total_seeds * probability_of_failure : ℝ)) = 200 :=
by
  -- The proper proof goes here, but we use sorry to complete the theorem statement
  sorry

end expected_replanted_seeds_l3_3861


namespace base4_representation_of_123_l3_3019

theorem base4_representation_of_123 : base_repr 4 123 = [1, 3, 2, 3] :=
sorry

end base4_representation_of_123_l3_3019


namespace gain_percent_is_80_l3_3750

noncomputable def cost_price : ℝ := 600
noncomputable def selling_price : ℝ := 1080
noncomputable def gain : ℝ := selling_price - cost_price
noncomputable def gain_percent : ℝ := (gain / cost_price) * 100

theorem gain_percent_is_80 :
  gain_percent = 80 := by
  sorry

end gain_percent_is_80_l3_3750


namespace probability_red_ball_l3_3803

def total_balls : ℕ := 3
def red_balls : ℕ := 1
def yellow_balls : ℕ := 2

theorem probability_red_ball : (red_balls : ℚ) / (total_balls : ℚ) = 1 / 3 :=
by
  sorry

end probability_red_ball_l3_3803


namespace express_inequality_l3_3845

theorem express_inequality (x : ℝ) : x + 4 ≥ -1 := sorry

end express_inequality_l3_3845


namespace casey_marathon_time_l3_3087

theorem casey_marathon_time (C : ℝ) (h : (C + (4 / 3) * C) / 2 = 7) : C = 10.5 :=
by
  sorry

end casey_marathon_time_l3_3087


namespace selection_methods_count_l3_3954

theorem selection_methods_count :
  (∃ (boys girls : ℕ), boys = 5 ∧ girls = 3 ∧ 
  ∃ (n k : ℕ), 
    n = 3 ∧ 
    (∃ (a b : ℕ), a = (nat.choose 5 2 * nat.choose 3 1) ∧ 
                   b = (nat.choose 5 1 * nat.choose 3 2) ∧ 
                   a + b = 45)) :=
sorry

end selection_methods_count_l3_3954


namespace trajectory_of_C_is_line_segment_l3_3512

-- Defining the conditions
variable (A B C : Point)
variable (alpha : Real)
variable (O X Y : Point)
variable (OA OB : Line)
variable (ABC : Triangle)
variable [TriangleSidesSlideAlongAngle α OA OB ABC A B]
variable [AngleSum α (angle A B C)]

-- Statement of the problem
theorem trajectory_of_C_is_line_segment :
  trajectory C = segment XY := sorry

end trajectory_of_C_is_line_segment_l3_3512


namespace max_value_a_plus_b_plus_c_l3_3133

-- Definitions used in the problem
def A_n (a n : ℕ) : ℕ := a * (10^n - 1) / 9
def B_n (b n : ℕ) : ℕ := b * (10^n - 1) / 9
def C_n (c n : ℕ) : ℕ := c * (10^(2 * n) - 1) / 9

-- Main statement of the problem
theorem max_value_a_plus_b_plus_c (n : ℕ) (a b c : ℕ) (h : n > 0) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_eq : ∃ n1 n2 : ℕ, n1 ≠ n2 ∧ C_n c n1 - B_n b n1 = 2 * (A_n a n1)^2 ∧ C_n c n2 - B_n b n2 = 2 * (A_n a n2)^2) :
  a + b + c ≤ 18 :=
sorry

end max_value_a_plus_b_plus_c_l3_3133


namespace right_triangle_hypotenuse_l3_3591

-- Each leg of the right triangle is defined as 2x and 2y, respectively.
variable (x y : ℝ)

-- Conditions for medians
axiom median_x : sqrt (y^2 + (x / 2)^2) = sqrt 18
axiom median_y : sqrt (x^2 + (y / 2)^2) = 3

-- Determine the hypotenuse of this triangle.
theorem right_triangle_hypotenuse :
  sqrt (4 * (x^2 + y^2)) = sqrt 86.4 :=
by
  sorry

end right_triangle_hypotenuse_l3_3591


namespace angle_between_vectors_l3_3906

variables {a b : euclidean_space ℝ (fin 2)}

-- Given conditions
def vector_a_norm_one : |a| = 1 := sorry
def vector_b_norm_two : |b| = 2 := sorry
def dot_product_condition : inner a b = 1 := sorry

-- Theorem to prove
theorem angle_between_vectors : 
  ∃ θ : ℝ, θ = real.arccos (1 / 2) ∧ θ = real.pi / 3 :=
begin
  use real.pi / 3,
  split,
  { rw [real.arccos, real.cos, real.pi],
    -- additional necessary rewrites and simplifications
    sorry },
  exact sorry
end

end angle_between_vectors_l3_3906


namespace prob_question_correct_answers_l3_3938

theorem prob_question_correct_answers {A B C D : Type} (correct_answers : set (option A) := {A, B, C}) :
  (selecting one option, probability of getting points = 3/4 ∨
   selecting two options, probability of getting points = 3/6 ∨ 
   selecting three options, probability of getting points = 1/4 ∨ 
   selecting four options, probability of getting points = 1) → 
   (conclusion C := true ∧ conclusion D := true) :=
sorry

end prob_question_correct_answers_l3_3938


namespace hyperbola_eccentricity_l3_3317

-- Definitions for conditions
def hyperbola (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) := 
  ∀ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1

-- Define the eccentricity condition function
def eccentricity (a b : ℝ) : ℝ := 
  real.sqrt (1 + (b^2 / a^2))

-- Main theorem: proving the eccentricity is 9/7 given the conditions
theorem hyperbola_eccentricity (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b)
  (perpendicular : ∀ (M F1 F2 : Point), M.on_asymptote a b → ⟂ M F1 M F2) 
  (sin_condition : ∀ (MF1F2 : ℝ), MF1F2 = 1 / 3) : 
  eccentricity a b = 9 / 7 :=
  sorry

end hyperbola_eccentricity_l3_3317


namespace min_number_of_rounds_l3_3396

theorem min_number_of_rounds (n : ℕ) (hn : n = 510) :
  ∃ r : ℕ, r = 9 ∧
    (∀ (participants : ℕ), participants = n → 
      (∀ (points : ℕ → ℕ), 
        (∀ p, p ≥ 0) ∧ 
        (∀ round, 
          ∃ matches : ℕ, 
            (∀ match_index, match_index < matches → 
              (abs (points match_index.succ - points match_index) ≤ 1)) ∧ 
                (participants / 2 ≤ matches →
                  (points match_index = participants.succ + 1))) →
          ((∀ round_number, round_number ≤ r → 
            ∃ final_leader_points, 
              (final_leader_points > 0) ∧ 
              (∀ other_points, other_points ≠ final_leader_points → 
                other_points < final_leader_points))))) :=
by
  intro n hn
  use 9
  split
  · rfl
  · intros participants hparticipants points hpoints round hround
    sorry

end min_number_of_rounds_l3_3396


namespace minimum_value_of_f_l3_3983

noncomputable def f (u v : ℝ) : ℝ :=
  (u - v) ^ 2 + (sqrt (4 - u ^ 2) - (2 * v + 5)) ^ 2

theorem minimum_value_of_f : ∀ u v : ℝ, 
  u ∈ Icc (-2 : ℝ) 2 → 
  (∃ u v, f u v = 9 - 4 * sqrt 5) := 
by
  sorry

end minimum_value_of_f_l3_3983


namespace minimum_waste_l3_3030

/-- Zenobia's cookout problem setup -/
def LCM_hot_dogs_buns : Nat := Nat.lcm 10 12

def hot_dog_packages : Nat := LCM_hot_dogs_buns / 10
def bun_packages : Nat := LCM_hot_dogs_buns / 12

def waste_hot_dog_packages : ℝ := hot_dog_packages * 0.4
def waste_bun_packages : ℝ := bun_packages * 0.3
def total_waste : ℝ := waste_hot_dog_packages + waste_bun_packages

theorem minimum_waste :
  hot_dog_packages = 6 ∧ bun_packages = 5 ∧ total_waste = 3.9 :=
by
  sorry

end minimum_waste_l3_3030


namespace mode_and_median_of_survey_l3_3337

/-- A data structure representing the number of students corresponding to each sleep time. -/
structure SleepSurvey :=
  (time7 : ℕ)
  (time8 : ℕ)
  (time9 : ℕ)
  (time10 : ℕ)

def survey : SleepSurvey := { time7 := 6, time8 := 9, time9 := 11, time10 := 4 }

theorem mode_and_median_of_survey (s : SleepSurvey) :
  (mode=9 ∧ median = 8.5) :=
by
  -- proof would go here
  sorry

end mode_and_median_of_survey_l3_3337


namespace part1_part2_l3_3172

-- Definitions
def f (x : ℝ) (λ ω : ℝ) : ℝ := λ * sin (ω * x) + cos (ω * x)
def g (x : ℝ) (λ ω : ℝ) : ℝ := f x λ ω + cos (2 * x - (2 * π / 3))

-- Conditions
def ω_pos (ω : ℝ) : Prop := ω > 0
def symmetry_distance_eq (ω : ℝ) : Prop := (π / ω) = (π / 2)
def symmetry_axis (λ ω : ℝ) : Prop := f (π / 6) λ ω = f 0 λ ω

-- Questions turned into propositions
def find_λ (λ ω : ℝ) (h₁ : ω_pos ω) (h₂ : symmetry_distance_eq ω) (h₃ : symmetry_axis λ ω) : Prop :=
  λ = sqrt 3

def range_of_g (λ ω : ℝ) (h₁ : ω_pos ω) (h₂ : symmetry_distance_eq ω) (h₃ : symmetry_axis λ ω) : Prop :=
  ∀ x : ℝ, (-π / 3 <= x ∧ x <= π / 6) → -3 <= g x λ ω ∧ g x λ ω <= (3 * sqrt 3 / 2)

-- Lean 4 statements
theorem part1 (λ ω : ℝ) (h₁ : ω_pos ω) (h₂ : symmetry_distance_eq ω) (h₃ : symmetry_axis λ ω) : find_λ λ ω h₁ h₂ h₃ :=
  sorry

theorem part2 (λ ω : ℝ) (h₁ : ω_pos ω) (h₂ : symmetry_distance_eq ω) (h₃ : symmetry_axis λ ω) : range_of_g λ ω h₁ h₂ h₃ :=
  sorry

end part1_part2_l3_3172


namespace college_students_total_l3_3207

theorem college_students_total
  (B G N : ℕ)
  (h_ratio : B : G : N = 8 : 5 : 3)
  (h_girls : G = 400) :
  B + G + N = 1280 :=
by
  -- use the given conditions to derive the proof
  sorry

end college_students_total_l3_3207


namespace ab_abs_value_l3_3296

theorem ab_abs_value {a b : ℤ} (ha : a ≠ 0) (hb : b ≠ 0)
  (hroots : ∃ r s : ℤ, (x - r)^2 * (x - s) = x^3 + a * x^2 + b * x + 9 * a) :
  |a * b| = 1344 := 
sorry

end ab_abs_value_l3_3296


namespace surface_area_of_circumsphere_l3_3151

-- Definitions for the conditions
def is_isosceles_right_triangle (A B C : Type) (hypotenuse : ℝ) : Prop :=
  hypotenuse = 2 ∧
  -- Other properties of an isosceles right triangle can be defined as needed

noncomputable def dihedral_angle (angle : ℝ) : Prop :=
  angle = π / 3

-- Statement of the theorem
theorem surface_area_of_circumsphere (A B C D : Type)
(hABC : is_isosceles_right_triangle A B C 2)
(h_angle : dihedral_angle (π / 3)) :
  let circumsphere_surface_area : ℝ := 32 * π / 3
in circumsphere_surface_area = 32 * π / 3 :=
by sorry

end surface_area_of_circumsphere_l3_3151


namespace sequence_problem_l3_3891

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n - a (n - 1) = a 1 - a 0

noncomputable def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∀ n, b n * b (n - 1) = b 1 * b 0

theorem sequence_problem
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (ha : a 0 = -9) (ha1 : a 3 = -1) (ha_seq : arithmetic_sequence a)
  (hb : b 0 = -9) (hb4 : b 4 = -1) (hb_seq : geometric_sequence b) :
  b 2 * (a 2 - a 1) = -8 :=
sorry

end sequence_problem_l3_3891


namespace correct_statements_l3_3602

noncomputable def star (a b : ℝ) : ℝ :=
a * b + a + b

def f (x : ℝ) : ℝ :=
x * (1 / x) + x + 1 / x

theorem correct_statements :
  let condition1 := ∀ a b : ℝ, star a b = star b a,
      condition2 := ∀ a : ℝ, star a 0 = a,
      condition3 := ∀ a b c : ℝ, star (star a b) c = star c (a * b) + star a c + star c b - 2 * c,
      statement1 := ∀ x : ℝ, 0 < x → (f x = 1 + x + 1 / x) ∧ (f x ≥ 3),
      statement2 := ¬(∀ x : ℝ, x ∈ (-∞, 0) ∪ (0, +∞) → f (-x) = -f (x)),
      statement3 := ∀ x : ℝ, (x < -1) ∨ (1 < x) → f' x > 0,
      correct_statements := statement1 ∧ statement3 in
  (condition1 ∧ condition2 ∧ condition3) → correct_statements → counter_examples ∨ correct_statements
  sorry

end correct_statements_l3_3602


namespace binary_remainder_l3_3354

theorem binary_remainder (n : ℕ) (h : n = 0b111100010111) : n % 4 = 3 := 
by
  rw [h]
  sorry

end binary_remainder_l3_3354


namespace prove_a_geq_1_l3_3901

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * Real.log x + a / x + 3
noncomputable def g (x : ℝ) : ℝ := x^3 - x^2

theorem prove_a_geq_1
  (a : ℝ)
  (h : ∀ x1 x2 : ℝ, x1 ∈ Icc (1/3) 2 → x2 ∈ Icc (1/3) 2 → f x1 a - g x2 ≥ 0) :
  a ≥ 1 := sorry

end prove_a_geq_1_l3_3901


namespace monotonic_intervals_min_value_l3_3541

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * x

theorem monotonic_intervals (a : ℝ) :
  (∀ x : ℝ, x > 0 → f x a) ∧ 
  (a ≤ 0 → ∀ x y : ℝ, 0 < x ∧ x < y → f x a < f y a) ∧ 
  (a > 0 → ∀ x : ℝ, x > 0 → (f' x a = 0 → x = 1 / a) ∧ (x < 1 / a → f x a < f (1 / a) a) ∧ (x > 1 / a → f (1 / a) a > f x a)) :=
  sorry

theorem min_value (a : ℝ) (h : a > 0):
  ( (1 ≤ 2) ∧ (a ≥ 1 → ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x a ≥ f 2 a) ∧
    (0 < a ∧ a < 1 / 2 → ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x a ≥ f 1 a) ∧
    (1 / a > 1 ∧ 1 / a < 2 → min (f 1 a) (f 2 a) ) ) :=
  sorry

end monotonic_intervals_min_value_l3_3541


namespace candy_total_l3_3007

theorem candy_total (r b : ℕ) (hr : r = 145) (hb : b = 3264) : r + b = 3409 := by
  -- We can use Lean's rewrite tactic to handle the equalities, but since proof is skipped,
  -- it's not necessary to write out detailed tactics here.
  sorry

end candy_total_l3_3007


namespace angle_sum_around_point_l3_3362

theorem angle_sum_around_point (y : ℝ) (h1 : 150 + y + y = 360) : y = 105 :=
by sorry

end angle_sum_around_point_l3_3362


namespace find_price_of_each_part_l3_3074

def original_price (total_cost : ℝ) (num_parts : ℕ) (price_per_part : ℝ) :=
  num_parts * price_per_part = total_cost

theorem find_price_of_each_part :
  original_price 439 7 62.71 :=
by
  sorry

end find_price_of_each_part_l3_3074


namespace neighboring_difference_at_least_five_l3_3594

theorem neighboring_difference_at_least_five :
  ∃ (r1 c1 r2 c2 : ℕ), 
  (r1 < 8) ∧ (c1 < 8) ∧ (r2 < 8) ∧ (c2 < 8) ∧ 
  ((r1 = r2 ∧ (c2 = c1 + 1 ∨ c1 = c2 + 1)) ∨ (c1 = c2 ∧ (r2 = r1 + 1 ∨ r1 = r2 + 1))) ∧
  ∃ (f : ℕ → ℕ → ℕ), 
  ((∀ r c : ℕ, r < 8 → c < 8 → 1 ≤ f r c ∧ f r c ≤ 64) ∧
  (∀ i j k l : ℕ, (i < 8 ∧ j < 8 ∧ k < 8 ∧ l < 8 ∧ ((i = k ∧ (j = l + 1 ∨ j = l - 1)) ∨ (j = l ∧ (i = k + 1 ∨ i = k - 1))) → (|f i j - f k l| < 64))) →
  (|f r1 c1 - f r2 c2| ≥ 5)) sorry

end neighboring_difference_at_least_five_l3_3594


namespace work_days_per_week_l3_3297

theorem work_days_per_week (d : ℕ) 
  (terry_income_per_day : ℕ)
  (jordan_income_per_day : ℕ)
  (weekly_income_difference : ℕ)
  (h1 : terry_income_per_day = 24)
  (h2 : jordan_income_per_day = 30)
  (h3 : weekly_income_difference = 42) : 
  30 * d - 24 * d = weekly_income_difference → d = 7 := 
by
  intro h
  have h_d : 6 * d = 42, from Eq.subst h (by rw [mul_sub; mul_add]; symmetry), 
  sorry

end work_days_per_week_l3_3297


namespace probability_drawing_red_l3_3804

/-- The probability of drawing a red ball from a bag that contains 1 red ball and 2 yellow balls. -/
theorem probability_drawing_red : 
  let N_red := 1
  let N_yellow := 2
  let N_total := N_red + N_yellow
  let P_red := (N_red : ℝ) / N_total
  P_red = (1 : ℝ) / 3 :=
by {
  sorry
}

end probability_drawing_red_l3_3804


namespace sum_of_reciprocals_l3_3468

namespace Problem

def quadr_eqn : Polynomial ℝ := Polynomial.C 6 + Polynomial.X * (Polynomial.C (-15)) + Polynomial.X ^ 2

theorem sum_of_reciprocals (r1 r2 : ℝ) (h : quadr_eqn = Polynomial.X^2 - 15*Polynomial.X + 6) (hr1 : Polynomial.IsRoot quadr_eqn r1) (hr2 : Polynomial.IsRoot quadr_eqn r2) :
  1 / r1 + 1 / r2 = 5 / 2 := by
  have h_sum : r1 + r2 = 15 := by
    sorry
  have h_prod : r1 * r2 = 6 := by
    sorry
  calc
    1 / r1 + 1 / r2 = (r1 + r2) / (r1 * r2) : by sorry
    ... = 15 / 6   : by sorry
    ... = 5 / 2    : by sorry

end Problem

end sum_of_reciprocals_l3_3468


namespace smallest_n_with_347_digits_l3_3308

theorem smallest_n_with_347_digits (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : m < n) 
  (h4 : Nat.coprime m n) 
  (h5 : ∃ k : ℕ, (∀ j : ℕ, j < (k + 3) → m * 10^(k + 3) / n % 10^(k + 3) = 347)) 
  : n = 999 :=
sorry

end smallest_n_with_347_digits_l3_3308


namespace sum_of_sequence_l3_3865

-- Definitions of sequence and difference sequence conditions
def a : ℕ → ℕ
| 0     := 2
| (n+1) := a n + 2^n

-- Sum of the first n terms of sequence a
def S (n : ℕ) : ℕ := ∑ i in Finset.range n, a i

-- Theorem statement to prove
theorem sum_of_sequence (n : ℕ) : S n = 2^(n+1) - 2 :=
by sorry

end sum_of_sequence_l3_3865


namespace sum_partial_fraction_l3_3820

theorem sum_partial_fraction :
  (∑ n in Finset.range 500, (1 : ℚ) / (n + 1)^2 + 2*(n + 1)) = 1499 / 2008 :=
by
  sorry

end sum_partial_fraction_l3_3820


namespace f_log2_3_eq_one_twelfth_l3_3897

def f : ℝ → ℝ :=
  λ x, if x ≥ 3 then 2^(-x) else f (x + 1)

theorem f_log2_3_eq_one_twelfth : f (Real.log 3 / Real.log 2) = 1 / 12 :=
by
  sorry

end f_log2_3_eq_one_twelfth_l3_3897


namespace sequence_sum_mod_l3_3425

def sequence (n : ℕ) : ℕ :=
  match n with
  | 0     => 2
  | 1     => 3
  | 2     => 5
  | (n+3) => sequence n + sequence (n+1) + sequence (n+2)

theorem sequence_sum_mod :
  let T := ∑ k in Finset.range 20, sequence k
  in T % 500 = remainder :=
sorry

end sequence_sum_mod_l3_3425


namespace illuminated_cube_surface_area_l3_3410

noncomputable def edge_length : ℝ := Real.sqrt (2 + Real.sqrt 3)
noncomputable def radius : ℝ := Real.sqrt 2
noncomputable def illuminated_area (a ρ : ℝ) : ℝ := Real.sqrt 3 * (Real.pi + 3)

theorem illuminated_cube_surface_area :
  illuminated_area edge_length radius = Real.sqrt 3 * (Real.pi + 3) := sorry

end illuminated_cube_surface_area_l3_3410


namespace inscribed_cube_volume_l3_3429

noncomputable def side_length_of_inscribed_cube (d : ℝ) : ℝ :=
d / Real.sqrt 3

noncomputable def volume_of_inscribed_cube (s : ℝ) : ℝ :=
s^3

theorem inscribed_cube_volume :
  (volume_of_inscribed_cube (side_length_of_inscribed_cube 12)) = 192 * Real.sqrt 3 :=
by
  sorry

end inscribed_cube_volume_l3_3429


namespace minimal_area_OMNA_is_maximized_l3_3043

variables {O B C A M N : Type} [EuclideanGeometry O B C A M N]

-- Define the given angles and conditions
def angle_OBC : ℝ := α
def ∠BOC : ℝ := α
def ∠MAN : ℝ := β
def length_MA : ℝ := |MA|
def length_AN : ℝ := |AN|

-- Points
axiom point_A_on_BC : IsOnLineSegment A BC
axiom point_M_on_OB : IsOnLineSegment M OB
axiom point_N_on_OC : IsOnLineSegment N OC

-- Conditions
axiom angle_MAN_equal_beta : ∠MAN = β
axiom length_MA_equals_length_AN : |MA| = |AN|
axiom line_MN_parallel_BC : Parallel MN BC

-- Problem statement
theorem minimal_area_OMNA_is_maximized :
  (minimized (area_of_quadrilateral OMNA)) :=
begin
  -- We assume the stated conditions as axioms and derive the required theorem
  assume (h : condition1) (h2 : condition2) (h3 : condition3),
  -- The proof of this statement follows from geometric analysis
  sorry
end

end minimal_area_OMNA_is_maximized_l3_3043


namespace find_x_l3_3318

-- Given conditions
variables {x : ℤ}
def numbers := [21, x, 56, 63, 9]
def is_negative_integer (x : ℤ) : Prop := x < 0
def median (l : List ℤ) : ℤ := List.nthLe (l.sort (≤)) (l.length / 2) (by simp [List.length_sort])
def mean (l : List ℤ) : ℤ := l.sum / l.length

-- Statement to prove
theorem find_x
  (h1 : is_negative_integer x)
  (h2 : median numbers = mean numbers + 20) :
  x = -144 :=
sorry

end find_x_l3_3318


namespace monomial_combined_l3_3200

theorem monomial_combined (n m : ℕ) (h₁ : 2 = n) (h₂ : m = 4) : n^m = 16 := by
  sorry

end monomial_combined_l3_3200


namespace range_of_m_value_of_m_when_perpendicular_equation_of_circle_with_diameter_MN_l3_3166

-- Define the equation of the circle with a parameter m and state the first condition.
def represents_circle (m : ℝ) : Prop :=
  ∀ (x y : ℝ), (x - 1)^2 + (y - 2)^2 = 5 - m

-- Theorem to prove the range of m for the equation to represent a circle.
theorem range_of_m (m : ℝ) : represents_circle m → m < 5 :=
by
  intros h,
  sorry

-- Define the line equation and state the second condition of intersection and orthogonality.
def line (x y : ℝ) : Prop :=
  x + 2 * y = 4

def perpendicular (OM ON : ℝ × ℝ) : Prop :=
  OM.fst * ON.fst + OM.snd * ON.snd = 0

-- Theorem to find the value of m when OM ⊥ ON.
theorem value_of_m_when_perpendicular (m : ℝ) (OM ON : ℝ × ℝ) :
  represents_circle m ∧ line OM.fst OM.snd ∧ line ON.fst ON.snd ∧ perpendicular OM ON → m = 8/5 :=
by
  intros h,
  sorry

-- Theorem to find the equation of the circle with diameter MN under the given conditions.
theorem equation_of_circle_with_diameter_MN (m : ℝ) (OM ON : ℝ × ℝ) :
  represents_circle m ∧ line OM.fst OM.snd ∧ line ON.fst ON.snd ∧ perpendicular OM ON ∧ m = 8/5 →
  (∀ (x y : ℝ), x^2 + y^2 = (8/5) * x + (16/5) * y) :=
by
  intros h,
  sorry

end range_of_m_value_of_m_when_perpendicular_equation_of_circle_with_diameter_MN_l3_3166


namespace area_enclosed_shape_l3_3670

open Real

theorem area_enclosed_shape : 
    let f1 := λ x : ℝ, 2 * x
    let f2 := λ x : ℝ, 2 / x
    let x1 := 1
    let x2 := e
    ∫ x in x1..x2, f1 x - f2 x = e^2 - 3 :=
by
  sorry

end area_enclosed_shape_l3_3670


namespace total_chairs_needed_l3_3234

theorem total_chairs_needed (tables_4_seats tables_6_seats seats_per_table_4 seats_per_table_6 : ℕ) : 
  tables_4_seats = 6 → 
  seats_per_table_4 = 4 → 
  tables_6_seats = 12 → 
  seats_per_table_6 = 6 → 
  (tables_4_seats * seats_per_table_4 + tables_6_seats * seats_per_table_6) = 96 := 
by
  intros h1 h2 h3 h4
  -- sorry

end total_chairs_needed_l3_3234


namespace probability_spade_then_ace_l3_3341

theorem probability_spade_then_ace :
  let total_cards := 52
  let total_aces := 4
  let total_spades := 13
  let ace_of_spades := 1
  let non_ace_spades := total_spades - ace_of_spades
  (non_ace_spades / total_cards) * (total_aces / (total_cards - 1)) +
  (ace_of_spades / total_cards) * ((total_aces - ace_of_spades) / (total_cards - 1)) = (1 / 52) :=
by
  sorry

end probability_spade_then_ace_l3_3341


namespace remainder_of_sum_of_first_15_natural_numbers_divided_by_11_l3_3722

theorem remainder_of_sum_of_first_15_natural_numbers_divided_by_11 :
  (∑ i in finset.range 16, i) % 11 = 10 :=
by 
  sorry

end remainder_of_sum_of_first_15_natural_numbers_divided_by_11_l3_3722


namespace sara_staircase_steps_l3_3942

-- Define the problem statement and conditions
theorem sara_staircase_steps (n : ℕ) :
  (3 * n * (n + 1) / 2 = 270) → n = 12 := 
by
  intro h
  sorry

end sara_staircase_steps_l3_3942


namespace value_of_f1_l3_3195

def f : ℕ → ℕ := sorry

variable {x : ℕ}

axiom h1 : ∀ x, x ∈ {1, 2, 3} → f(x) ∈ {1, 2, 3}
axiom h2 : f(1) < f(2) ∧ f(2) < f(3)
axiom h3 : ∀ x, x ∈ {1, 2, 3} → f(f(x)) = 3 * x

theorem value_of_f1 : f(1) = 2 :=
by
  sorry

end value_of_f1_l3_3195


namespace total_cost_is_90_l3_3608

variable (jackets : ℕ) (shirts : ℕ) (pants : ℕ)
variable (price_jacket : ℕ) (price_shorts : ℕ) (price_pants : ℕ)

theorem total_cost_is_90 
  (h1 : jackets = 3)
  (h2 : price_jacket = 10)
  (h3 : shirts = 2)
  (h4 : price_shorts = 6)
  (h5 : pants = 4)
  (h6 : price_pants = 12) : 
  (jackets * price_jacket + shirts * price_shorts + pants * price_pants) = 90 := by 
  sorry

end total_cost_is_90_l3_3608


namespace madrid_time_correct_l3_3026

def time_difference (city1 city2 : String) (h1 h2 : ℕ) :=
  if city1 = "London" ∧ city2 = "Madrid" then 1
  else if city1 = "San Francisco" ∧ city2 = "London" then 8
  else if city1 = "San Francisco" ∧ city2 = "Madrid" then 9
  else 0

theorem madrid_time_correct :
  ∀ (current_time_sf : ℕ) (time_sf_bed : ℕ), 
  time_sf_bed = 21 →
  current_time_sf = time_sf_bed + time_difference "San Francisco" "Madrid" 0 0 →
  current_time_sf % 24 = 6 :=
by 
  intros current_time_sf time_sf_bed h1 h2
  rw h1 at h2
  rw h2
  sorry

end madrid_time_correct_l3_3026


namespace num_valid_8_tuples_is_1540_l3_3466

def valid_8_tuple (a1 a2 a3 a4 b1 b2 b3 b4 : ℕ) := 
  (0 ≤ a1 ∧ a1 ≤ 1) ∧ 
  (0 ≤ a2 ∧ a2 ≤ 2) ∧ 
  (0 ≤ a3 ∧ a3 ≤ 3) ∧
  (0 ≤ a4 ∧ a4 ≤ 4) ∧ 
  a1 + a2 + a3 + a4 + 2 * b1 + 3 * b2 + 4 * b3 + 5 * b4 = 19

theorem num_valid_8_tuples_is_1540 :
  {t : ℕ × ℕ × ℕ × ℕ × ℕ × ℕ × ℕ × ℕ // valid_8_tuple t.1 t.2.1 t.2.2.1 t.2.2.2.1 t.2.2.2.2.1 t.2.2.2.2.2.1 t.2.2.2.2.2.2.1 t.2.2.2.2.2.2.2 } = 1540 :=
by
  -- Proof goes here
  sorry

end num_valid_8_tuples_is_1540_l3_3466


namespace volume_of_region_l3_3492

noncomputable def f (x y z : ℝ) : ℝ :=
  |x + y + z| + |x + y - z| + |x - y + z| + |-x + y + z| + |x|

theorem volume_of_region :
  (∫ z in -∞..∞, ∫ y in -∞..∞, ∫ x in -∞..∞, if f x y z ≤ 6 then 1 else 0) = 288 / 125 := sorry

end volume_of_region_l3_3492


namespace james_out_of_pocket_cost_l3_3237

theorem james_out_of_pocket_cost (total_cost : ℝ) (coverage : ℝ) (out_of_pocket_cost : ℝ)
  (h1 : total_cost = 300) (h2 : coverage = 0.8) :
  out_of_pocket_cost = 60 :=
by
  sorry

end james_out_of_pocket_cost_l3_3237


namespace prime_divisible_by_57_is_zero_l3_3561

open Nat

theorem prime_divisible_by_57_is_zero :
  (∀ p, Prime p → (57 ∣ p) → False) :=
by
  intro p hp hdiv
  have h57 : 57 = 3 * 19 := by norm_num
  have h1 : p = 57 ∨ p = 3 ∨ p = 19 := sorry
  have hp1 : p ≠ 57 := sorry
  have hp2 : p ≠ 3 := sorry
  have hp3 : p ≠ 19 := sorry
  exact Or.elim h1 hp1 (Or.elim hp2 hp3)


end prime_divisible_by_57_is_zero_l3_3561


namespace monotonic_intervals_range_of_a_for_zeros_min_value_g_l3_3905

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1 / 3) * x^3 + (1 - a) / 2 * x^2 - a * x - a

theorem monotonic_intervals (a : ℝ) (h : a > 0) : 
  (∀ x ∈ (-∞:ℝ, -1), 0 < derivative (f x a)) ∧
  (∀ x ∈ (a:ℝ, +∞), 0 < derivative (f x a)) ∧
  (∀ x ∈ (-1:ℝ, a), 0 > derivative (f x a)) :=
  sorry

theorem range_of_a_for_zeros (a : ℝ) (h : 0 < a ∧ a < 1 / 3) :
  (∃ x1 x2 ∈ (-3:ℝ, 0), f x1 a = 0 ∧ f x2 a = 0 ∧ x1 ≠ x2) :=
  sorry

noncomputable def M (t : ℝ) (ft : ℝ → ℝ) : ℝ := 
  max (ft t) (ft (t + 3))

noncomputable def m (t : ℝ) (ft : ℝ → ℝ) : ℝ := 
  min (ft t) (ft (t + 3))

noncomputable def g (t : ℝ) (ft : ℝ → ℝ) : ℝ := 
  M t ft - m t ft

theorem min_value_g (t : ℝ) :
  let a := 1
  let ft := f · a
  (-4 ≤ t ∧ t ≤ -1) → (∃ t0 ∈ (-4:ℝ, -1), g t0 ft = 4 / 3) :=
  sorry

end monotonic_intervals_range_of_a_for_zeros_min_value_g_l3_3905


namespace probability_function_increasing_on_interval_l3_3699

theorem probability_function_increasing_on_interval :
  let possible_values := {1, 2, 3, 4, 5, 6}
  let occurrences := (possible_values × possible_values).card
  let favorable_outcomes := ((possible_values × possible_values).filter (λ (mn : ℕ × ℕ), mn.2 * 2 ≤ mn.1)).card
  favorable_outcomes / occurrences = 1 / 4 := by
    let possible_values : finset ℕ := {1, 2, 3, 4, 5, 6}
    let occurrences : ℕ := (possible_values.product possible_values).card
    let favorable_outcomes : ℕ := (possible_values.product possible_values).filter (λ (mn : ℕ × ℕ), 2 * mn.2 ≤ mn.1).card
    have h : favorable_outcomes = 9 := sorry
    have h2 : occurrences = 36 := sorry
    rw [h, h2]
    norm_num
    exact eq_refl (1 / 4)

end probability_function_increasing_on_interval_l3_3699


namespace vector_on_line_at_t4_eq_expected_l3_3774

theorem vector_on_line_at_t4_eq_expected :
  let a : ℝ × ℝ × ℝ := (1, 5, 9)
  let b : ℝ × ℝ × ℝ := (6, 0, 4)
  let d : ℝ × ℝ × ℝ := (5, -5, -5)
  ∀ t : ℝ,
    (a + t * d) = (a.1 + t * d.1, a.2 + t * d.2, a.3 + t * d.3) →
    (a + 4 * d) = (21, -15, -11) :=
by
  let a : ℝ × ℝ × ℝ := (1, 5, 9)
  let b : ℝ × ℝ × ℝ := (6, 0, 4)
  let d : ℝ × ℝ × ℝ := (5, -5, -5)
  assume t : ℝ,
  have h : a + t * d = (a.1 + t * d.1, a.2 + t * d.2, a.3 + t * d.3), from sorry,
  have ht4 : a + 4 * d = (21, -15, -11), from sorry,
  show a + 4 * d = (21, -15, -11), from ht4

example : (1 + 4 * 5, 5 + 4 * (-5), 9 + 4 * (-5)) = (21, -15, -11) := by
  unfold_coes
  simp

end vector_on_line_at_t4_eq_expected_l3_3774


namespace football_points_difference_l3_3448

theorem football_points_difference :
  let points_per_touchdown := 7
  let brayden_gavin_touchdowns := 7
  let cole_freddy_touchdowns := 9
  let brayden_gavin_points := brayden_gavin_touchdowns * points_per_touchdown
  let cole_freddy_points := cole_freddy_touchdowns * points_per_touchdown
  cole_freddy_points - brayden_gavin_points = 14 :=
by sorry

end football_points_difference_l3_3448


namespace common_chord_through_orthocenter_l3_3882

/-- Given points M and N on lines AB and AC respectively in triangle ABC,
prove that the common chord of the two circles with diameters CM and BN 
passes through the orthocenter of triangle ABC. -/
theorem common_chord_through_orthocenter
  (A B C M N : Point)
  (hM : M ∈ line_through A B)
  (hN : N ∈ line_through A C) :
  passes_through (common_chord (circle_with_diameter C M) (circle_with_diameter B N)) (orthocenter A B C) :=
  sorry

end common_chord_through_orthocenter_l3_3882


namespace probability_path_A_to_B_through_C_and_D_l3_3095

theorem probability_path_A_to_B_through_C_and_D :
  let p_A_C := Nat.choose 5 3, -- Paths from A to C: 3 east and 2 south
      p_C_D := Nat.choose 3 2, -- Paths from C to D: 2 east and 1 south
      p_D_B := Nat.choose 3 1, -- Paths from D to B: 1 east and 2 south
      total_paths_via_C_D := p_A_C * p_C_D * p_D_B, -- Total paths via C and D
      total_paths_A_B := Nat.choose 11 6 -- Total paths directly from A to B: 6 east and 5 south
  in (total_paths_via_C_D : ℚ) / total_paths_A_B = 15 / 77 :=
sorry

end probability_path_A_to_B_through_C_and_D_l3_3095


namespace area_triangle_KPN_l3_3597

-- Definitions of the points and their coordinates
def A : (ℝ × ℝ) := (0, 0)
def B : (ℝ × ℝ) := (2, 0)
def C : (ℝ × ℝ) := (2, 4)
def D : (ℝ × ℝ) := (0, 4)
def K : (ℝ × ℝ) := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
def L : (ℝ × ℝ) := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
def M : (ℝ × ℝ) := ((D.1 + A.1) / 2, (D.2 + A.2) / 2)
def N : (ℝ × ℝ) := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
def P : (ℝ × ℝ) := ((K.1 + M.1) / 2, (K.2 + M.2) / 2)

-- Function to calculate the area of a triangle given its vertices
def triangle_area (A B C : (ℝ × ℝ)) : ℝ :=
  (1 / 2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- Prove that the area of triangle KPN is 3/2
theorem area_triangle_KPN : triangle_area K P N = 3 / 2 :=
  by
    -- Proof omitted
    sorry

end area_triangle_KPN_l3_3597


namespace kaarel_wins_game_l3_3386

theorem kaarel_wins_game (p : ℕ) (h_prime : Nat.Prime p) (h_p_gt_3 : p > 3) :
  ∀ (R K : Finset ℕ) (hR : ∀ x ∈ R, x ∈ Finset.range (p - 1))
    (hK : ∀ x ∈ K, x ∈ Finset.range (p - 1))
    (h_no_common : R ∩ K = ∅) (h_size : (R ∪ K).card = p - 1),
    let sum_products := ∑ x in R, x * (p - x)
    sum_products % p = 0 :=
by
  sorry

end kaarel_wins_game_l3_3386


namespace part1_monotonic_increasing_part2_exists_p_gt_condition_part3_sum_squares_log_l3_3999

open Real

def f (x p : ℝ) : ℝ := p * x - p / x - 2 * log x
def g (x : ℝ) : ℝ := 2 * exp 1 / x

-- Proposition for (1)
theorem part1_monotonic_increasing (p : ℝ) (h : ∀ x > 0, differentiable_at ℝ (f x p) x → (f x p)' ≥ 0) : p ≥ 1 :=
sorry

-- Proposition for (2)
theorem part2_exists_p_gt_condition (p : ℝ) (h_pos : p > 0) (h : ∃ x ∈ Icc 1 (exp 1), f x p > g x) : p > 4 * exp 1 / (exp 1 ^ 2 - 1) :=
sorry

-- Proposition for (3)
theorem part3_sum_squares_log (n : ℕ) (h_pos : 0 < n) : (Finset.range n).sum (λ k, log (1 + 2 / (k + 1) : ℝ)^2) < 3 :=
sorry

end part1_monotonic_increasing_part2_exists_p_gt_condition_part3_sum_squares_log_l3_3999


namespace math_problem_l3_3361

theorem math_problem :
  (2^0 - 1 + 5^2 - 0)⁻¹ * 5 = 1 / 5 :=
by
  sorry

end math_problem_l3_3361


namespace cost_per_adult_meal_l3_3443

theorem cost_per_adult_meal (total_people : ℕ) (num_kids : ℕ) (total_cost : ℕ) (cost_per_kid : ℕ) :
  total_people = 12 →
  num_kids = 7 →
  cost_per_kid = 0 →
  total_cost = 15 →
  (total_cost / (total_people - num_kids)) = 3 :=
by
  intros
  sorry

end cost_per_adult_meal_l3_3443


namespace exists_linear_combination_of_proper_fractions_l3_3152

theorem exists_linear_combination_of_proper_fractions 
  (n : ℕ) (hn : 0 < n) 
  (a : ℕ → ℕ) (h_a : ∀ i, a i < n + i)
  (x : ℕ → ℚ) (h_x : ∀ i, x i = a i / (n + i)) :
  ∃ k (c : ℕ → ℤ), (∑ i in finset.range k, c i * x (i + 1) = 1) := 
sorry

end exists_linear_combination_of_proper_fractions_l3_3152


namespace range_of_abscissa_of_P_l3_3531

noncomputable def point_lies_on_line (P : ℝ × ℝ) : Prop :=
  P.1 - P.2 + 1 = 0

noncomputable def point_lies_on_circle_c (M N : ℝ × ℝ) : Prop :=
  (M.1 - 2)^2 + (M.2 - 1)^2 = 1 ∧ (N.1 - 2)^2 + (N.2 - 1)^2 = 1

noncomputable def angle_mpn_eq_60 (P M N : ℝ × ℝ) : Prop :=
  true -- This is a placeholder because we have to define the geometrical angle condition which is complex.

theorem range_of_abscissa_of_P :
  ∀ (P M N : ℝ × ℝ),
  point_lies_on_line P →
  point_lies_on_circle_c M N →
  angle_mpn_eq_60 P M N →
  0 ≤ P.1 ∧ P.1 ≤ 2 := sorry

end range_of_abscissa_of_P_l3_3531


namespace polynomial_evaluation_at_3_l3_3259

noncomputable def P (x : ℝ) : ℝ := b_0 + b_1 * x + b_2 * x^2 + b_3 * x^3 + b_4 * x^4 + b_5 * x^5

theorem polynomial_evaluation_at_3 (b_0 b_1 b_2 b_3 b_4 b_5 : ℤ) 
  (h0 : 0 ≤ b_0 ∧ b_0 < 4)
  (h1 : 0 ≤ b_1 ∧ b_1 < 4)
  (h2 : 0 ≤ b_2 ∧ b_2 < 4)
  (h3 : 0 ≤ b_3 ∧ b_3 < 4)
  (h4 : 0 ≤ b_4 ∧ b_4 < 4)
  (h5 : 0 ≤ b_5 ∧ b_5 < 4)
  (hP : P (Real.sqrt 2) = 30 + 26 * Real.sqrt 2) : 
  P 3 = 458 :=
by
  sorry

end polynomial_evaluation_at_3_l3_3259


namespace sum_of_coefficients_l3_3982

theorem sum_of_coefficients (u v w : ℝ) (α β γ : ℝ) (t : ℕ → ℝ) :
  (polynomial.eval u (polynomial.C (-7) + polynomial.X * (polynomial.C 6 + polynomial.X * (polynomial.C (-5) + polynomial.X)))) = 0 →
  (polynomial.eval v (polynomial.C (-7) + polynomial.X * (polynomial.C 6 + polynomial.X * (polynomial.C (-5) + polynomial.X)))) = 0 →
  (polynomial.eval w (polynomial.C (-7) + polynomial.X * (polynomial.C 6 + polynomial.X * (polynomial.C (-5) + polynomial.X)))) = 0 →
  t 0 = 3 →
  t 1 = 5 →
  t 2 = 9 →
  (∀ k ≥ 2, t (k+1) = α * t k + β * t (k-1) + γ * t (k-2) + 2) →
  α + β + γ = 3 :=
sorry

end sum_of_coefficients_l3_3982


namespace b7_value_l3_3992

theorem b7_value (a : ℕ → ℚ) (b : ℕ → ℚ)
  (h₀a : a 0 = 3) (h₀b : b 0 = 4)
  (h₁ : ∀ n, a (n + 1) = a n ^ 2 / b n)
  (h₂ : ∀ n, b (n + 1) = b n ^ 2 / a n) :
  b 7 = 4 ^ 730 / 3 ^ 1093 :=
by
  sorry

end b7_value_l3_3992


namespace symmetric_point_about_x_axis_l3_3306

theorem symmetric_point_about_x_axis (P : ℝ × ℝ) (hx : P.1 = 3) (hy : P.2 = -4) :
  ∃ P' : ℝ × ℝ, P'.1 = 3 ∧ P'.2 = 4 :=
by
  let P' := (P.1, - P.2)
  use P'
  split
  { rw hx }
  { rw hy, exact neg_neg 4 }
  sorry

end symmetric_point_about_x_axis_l3_3306


namespace geom_sum_eq_six_l3_3524

variable (a : ℕ → ℝ)
variable (r : ℝ) -- common ratio for geometric sequence

-- Conditions
axiom geom_seq (n : ℕ) : a (n + 1) = a n * r
axiom pos_seq (n : ℕ) : a (n + 1) > 0
axiom given_eq : a 1 * a 3 + 2 * a 2 * a 5 + a 4 * a 6 = 36

-- Proof statement
theorem geom_sum_eq_six : a 2 + a 5 = 6 :=
sorry

end geom_sum_eq_six_l3_3524


namespace relation_among_AP_PB_AB_DF_l3_3230

-- Definitions derived from conditions in a)
variables (r : ℝ)

def AB := 3 * r
def AO := (AB^2 - r^2).sqrt
def DO := r
def AD := AO - DO
def AP := AD
def PB := AB - AP
def DE := DO -- as DE = DO is assumed to create the line ADOE

-- Stating the mathematically equivalent problem
theorem relation_among_AP_PB_AB_DF (r : ℝ) : PB * AB = AP^2 + 3 * r^2 :=
by
  sorry  -- proof to be provided

end relation_among_AP_PB_AB_DF_l3_3230


namespace cube_root_neg3_is_neg3_l3_3674

theorem cube_root_neg3_is_neg3 (x : ℝ) (h : x = real.cbrt (-3)) : x = -3 :=
by
  rw [h, real.cbrt_eq_iff_pow_eq (by norm_num : 3 ≠ 0)]
  norm_num
  sorry

end cube_root_neg3_is_neg3_l3_3674


namespace smallest_positive_period_of_f_max_min_values_of_f_on_interval_l3_3540

def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x)^2 + Real.cos (2 * x)

theorem smallest_positive_period_of_f:
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') :=
begin
  use π,
  split,
  { exact real.pi_pos },
  sorry
end

theorem max_min_values_of_f_on_interval:
  ∃ (max min : ℝ), (∀ x ∈ Icc (π/4) (3*π/4), f x ≤ max) ∧ (∀ x ∈ Icc (π/4) (3*π/4), min ≤ f x) ∧ 
  max = 2 ∧ min = -Real.sqrt 2 + 1 :=
begin
  use [2, -Real.sqrt 2 + 1],
  sorry
end

end smallest_positive_period_of_f_max_min_values_of_f_on_interval_l3_3540


namespace questionnaire_C_count_l3_3791

theorem questionnaire_C_count (n : ℕ) (hn : n ∈ (1 : ℕ)..=960)
  (h₁ : ∀ k ∈ finset.range 32, a_n k = 30 * k + 5)
  (h₃ : ∀ m ∈ finset.range 32, (a_n m ≤ 450) ∨ (451 ≤ a_n m ∧ a_n m ≤ 750) ∨ (751 ≤ a_n m ∧ a_n m ≤ 960)) :
  (fintype.card {m // 751 ≤ a_n m ∧ a_n m ≤ 960} = 7) :=
by
  sorry

end questionnaire_C_count_l3_3791


namespace distance_next_second_l3_3075

variable (g : ℝ) (t : ℝ) (h : ℝ)

theorem distance_next_second (g_pos : 0 < g) (h_def : h = 0.5 * g) : 
  let distance := (λ t, 0.5 * g * t^2) in
  distance 2 - distance 1 = 3 * h := by
  sorry

end distance_next_second_l3_3075


namespace only_prime_square_l3_3782

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

structure Point :=
(x : ℤ)
(y : ℤ)

def is_integer_point (p : Point) : Prop := true

def is_prime_point (p : Point) : Prop :=
  is_prime p.x.natAbs ∧ is_prime p.y.natAbs

def is_prime_square (p1 p2 p3 p4 : Point) : Prop :=
  let boundary_points := [p1, p2, p3, p4] in
  ∀ p ∈ boundary_points, is_prime_point p

theorem only_prime_square :
  ∃ (p1 p2 p3 p4 : Point), 
    is_prime_square p1 p2 p3 p4 ∧
    ((p1, p2, p3, p4) = (⟨2, 2⟩, ⟨2, 3⟩, ⟨3, 2⟩, ⟨3, 3⟩)) :=
by
  sorry

end only_prime_square_l3_3782


namespace quad_diagonal_dot_product_eq_zero_l3_3510

variables (A B C D : ℝ^3)
variable (distance : ℝ^3 → ℝ^3 → ℝ)
variable {dist_AB : distance A B = 3}
variable {dist_BC : distance B C = 7}
variable {dist_CD : distance C D = 11}
variable {dist_DA : distance D A = 9}

-- Define the distance function
def distance (P Q : ℝ^3) : ℝ := (P - Q).norm

-- The statement we need to prove
theorem quad_diagonal_dot_product_eq_zero (A B C D : ℝ^3)
  (h_AB : distance A B = 3)
  (h_BC : distance B C = 7)
  (h_CD : distance C D = 11)
  (h_DA : distance D A = 9) :
  (C - A) • (B - D) = 0 :=
by
sorry

end quad_diagonal_dot_product_eq_zero_l3_3510


namespace length_of_MN_correct_l3_3009

noncomputable def length_of_MN (Q R S M N : Point) (hQRS : IsIsoscelesTriangle Q R S) (areaQRS : ℝ)
  (baseRS : ℝ) (heightQS : ℝ) (area_trapezoid : ℝ) (parallel_MN_RS : Line) : ℝ :=
  if hQRS_area : areaQRS = 180 ∧ baseRS = 12 ∧ heightQS = 30 
     ∧ area_trapezoid = 120 ∧ parallel_MN_RS then 4 * Real.sqrt 3
  else 0

theorem length_of_MN_correct (Q R S M N : Point) (hQRS : IsIsoscelesTriangle Q R S) 
  (areaQRS : ℝ := 180) (baseRS : ℝ := 12) (heightQS : ℝ := 30) 
  (area_trapezoid : ℝ := 120) (parallel_MN_RS : Line) :
  length_of_MN Q R S M N hQRS areaQRS baseRS heightQS area_trapezoid parallel_MN_RS = 4 * Real.sqrt 3 :=
  sorry

end length_of_MN_correct_l3_3009


namespace equal_radii_for_odd_l3_3984

-- Conditions translated to Lean definitions
def is_polygon_inscribed (A : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i, A i ∈ (circle ℝ) ∧ ∃ center, inside center (A 0) ∧ inside center (A (n - 1))

def inscribed_circle (A : ℕ → ℝ) (k : ℕ) : circle ℝ :=
  { center := A k, radius := some_radius }

def circles_tangent (A : ℕ → ℝ) (k : ℕ) : Prop :=
  tangent (inscribed_circle A k) (circle ℝ) ∧
  ∃ B, point_on_chord B (A k) (A (k + 1)) ∧ 
    intersection_tangent (inscribed_circle A k) (inscribed_circle A (k + 1)) B 

def odd (n : ℕ) : Prop :=
  n % 2 = 1

-- Problem Statement in Lean 4
theorem equal_radii_for_odd (A : ℕ → ℝ) (n : ℕ) 
    (h_polygon : is_polygon_inscribed A n)
    (h_circles_tangent : ∀ k, circles_tangent A k)
    (h_odd : odd n) : 
    ∀ k, radii_equality (inscribed_circle A k) (inscribed_circle A k) :=
sorry

end equal_radii_for_odd_l3_3984


namespace train_length_l3_3793

theorem train_length (speed_kmph : ℝ) (time_sec : ℝ) (speed_ms : ℝ) (length_m : ℝ)
  (h1 : speed_kmph = 120) 
  (h2 : time_sec = 6)
  (h3 : speed_ms = 33.33)
  (h4 : length_m = 200) : 
  speed_kmph * 1000 / 3600 * time_sec = length_m :=
by
  sorry

end train_length_l3_3793


namespace find_g_sum_l3_3138

def g (n : ℕ) : ℝ := log 3003 (n ^ 3)

theorem find_g_sum : g 7 + g 11 + g 13 = 9 / 4 :=
by sorry

end find_g_sum_l3_3138


namespace maximize_profit_l3_3159

noncomputable def g (x : ℝ) : ℝ :=
if h : (0 < x ∧ x ≤ 10) then 13.5 - (1 / 30) * x^2
else if h : (x > 10) then 168 / x - 2000 / (3 * x^2)
else 0 -- default case included for totality

noncomputable def y (x : ℝ) : ℝ :=
if h : (0 < x ∧ x ≤ 10) then 8.1 * x - (1 / 30) * x^3 - 20
else if h : (x > 10) then 148 - 2 * (1000 / (3 * x) + 2.7 * x)
else 0 -- default case included for totality

theorem maximize_profit (x : ℝ) : 0 < x → y 9 = 28.6 :=
by sorry

end maximize_profit_l3_3159


namespace cost_difference_l3_3245

-- Given conditions
def first_present_cost : ℕ := 18
def third_present_cost : ℕ := first_present_cost - 11
def total_cost : ℕ := 50

-- denoting costs of the second present via variable
def second_present_cost (x : ℕ) : Prop :=
  first_present_cost + x + third_present_cost = total_cost

-- Goal statement
theorem cost_difference (x : ℕ) (h : second_present_cost x) : x - first_present_cost = 7 :=
  sorry

end cost_difference_l3_3245


namespace reading_homework_pages_l3_3659

variable (total_pages math_pages : ℕ)
variable (h1 : total_pages = 7)
variable (h2 : math_pages = 5)

theorem reading_homework_pages : total_pages - math_pages = 2 :=
by
  rw [h1, h2]
  norm_num
  sorry

end reading_homework_pages_l3_3659


namespace trigonometric_identity_l3_3381

theorem trigonometric_identity (α : ℝ) : 
  (2 * cos (π / 6 - 2 * α) - sqrt 3 * sin (5 * π / 2 - 2 * α)) /
    (cos (9 * π / 2 - 2 * α) + 2 * cos (π / 6 + 2 * α)) = 
  tan (2 * α) / sqrt 3 :=
by
  sorry

end trigonometric_identity_l3_3381


namespace min_value_fraction_l3_3863

theorem min_value_fraction (x : ℝ) (h : x > 4) : 
  ∃ y : ℝ, (y = 2 * Real.sqrt 19) ∧ (∀ z : ℝ, (z = (x + 15) / Real.sqrt (x - 4)) → z ≥ y) :=
by
  sorry

end min_value_fraction_l3_3863


namespace prime_iff_k_t_greater_than_n_over_four_l3_3514

noncomputable def odd_integer (n : ℕ) : Prop := n % 2 = 1

theorem prime_iff_k_t_greater_than_n_over_four (n k t : ℕ) (hn : odd_integer n) (h_gt_3 : n > 3)
  (h_kn_sq : ∃ a : ℕ, k * n + 1 = a ^ 2) (h_tn_sq : ∃ b : ℕ, t * n = b ^ 2) :
  (nat.prime n ↔ k > n / 4 ∧ t > n / 4) :=
sorry

end prime_iff_k_t_greater_than_n_over_four_l3_3514


namespace sum_mod_11_l3_3721

theorem sum_mod_11 :
  let s := (List.range 15).map (λ x => x + 1) in
  (s.sum % 11) = 10 :=
by
  let s := (List.range 15).map (λ x => x + 1)
  have h : s.sum = 120 := sorry
  show s.sum % 11 = 10
  calc
    s.sum % 11 = 120 % 11  : by rw [h]
           ... = 10        : by norm_num

end sum_mod_11_l3_3721


namespace hyperbola_eccentricity_l3_3416

theorem hyperbola_eccentricity (center_x center_y : ℝ) (radius : ℝ) (h_center : (center_x, center_y) = (2, 0))
    (h_radius : radius = real.sqrt 3)
    (hyperbola_at_origin : true)
    (asymptotes_tangent_to_circle : true) :
    ∃ e : ℝ, e = 2 ∨ e = 2 * real.sqrt 3 / 3 := sorry

end hyperbola_eccentricity_l3_3416


namespace average_monthly_increase_l3_3403

theorem average_monthly_increase (x : ℝ) (turnover_january turnover_march : ℝ)
  (h_jan : turnover_january = 2)
  (h_mar : turnover_march = 2.88)
  (h_growth : turnover_march = turnover_january * (1 + x) * (1 + x)) :
  x = 0.2 :=
by
  sorry

end average_monthly_increase_l3_3403


namespace insect_leg_discrepancy_l3_3810

theorem insect_leg_discrepancy : 
  ∀ (total_legs legs_6 legs_8 : ℕ), 
  total_legs = 190 → legs_6 = 78 → legs_8 = 24 → 
  ¬ ∃ (x y z : ℕ), 6 * x = legs_6 ∧ 8 * y = legs_8 ∧ 10 * z = total_legs - (legs_6 + legs_8)
by
  intro total_legs legs_6 legs_8 h_total h_legs6 h_legs8
  rw [h_total, h_legs6, h_legs8]
  have h1 : 6 = 6 := rfl
  have h2 : 78 = 78 := rfl
  have h3 : (total_legs - (legs_6 + legs_8)) = 88 := rfl
  intro ⟨x, y, z, hx, hy, hz⟩
  rw [hx, hy, hz] at h3
  have hz2 : z * 10 = 88 := rfl
  have hz3 : z = 88 / 10 := by sorry
  have hz4 : ¬ (∃ z : ℕ, z * 10 = 88) := by sorry
  exact hz4 ⟨z, hz⟩

end insect_leg_discrepancy_l3_3810


namespace frog_probability_vertical_side_l3_3050

noncomputable def P : ℕ × ℕ → ℚ
| (0, y) := 1  -- boundary condition on left vertical side
| (6, y) := 1  -- boundary condition on right vertical side
| (x, 0) := 0  -- boundary condition on bottom horizontal side
| (x, 6) := 0  -- boundary condition on top horizontal side
| (2, 2) := (P (1, 2) + P (3, 2) + P (2, 1) + P (2, 3)) / 4
| (3, 2) := (P (2, 2) + 1) / 2
| (2, 1) := P (2, 2) / 2
| _ := sorry  -- Other points not used in this problem's specific solution

theorem frog_probability_vertical_side :
  P (2, 2) = 2 / 3 :=
sorry

end frog_probability_vertical_side_l3_3050


namespace ant_population_percentage_l3_3402

theorem ant_population_percentage (R : ℝ) 
  (h1 : 0.45 * R = 46.75) 
  (h2 : R * 0.55 = 46.75) : 
  R = 0.85 := 
by 
  sorry

end ant_population_percentage_l3_3402


namespace green_ball_probability_l3_3100

theorem green_ball_probability :
  let prob := (1 / 4) * (4 / 12) + (1 / 4) * (5 / 7) + (1 / 4) * (5 / 7) + (1 / 4) * (4 / 8)
  in prob = 25 / 56 :=
by
  sorry

end green_ball_probability_l3_3100


namespace missing_number_sum_eq_nine_l3_3427

theorem missing_number_sum_eq_nine (x : ℝ) (h1 : {10, 2, 5, 2, 4, 2, x}.to_finset.sum / 7 = (list.median [10, 2, 5, 2, 4, 2, x]) ∧
  (list.median [10, 2, 5, 2, 4, 2, x]) - (list.mode [10, 2, 5, 2, 4, 2, x]).to_finset = 
  2 * ((list.median [10, 2, 5, 2, 4, 2, x]) - {10, 2, 5, 2, 4, 2, x}.to_finset.sum / 7))
  : 
  x = 9 :=
sorry

end missing_number_sum_eq_nine_l3_3427


namespace fine_on_fifth_day_l3_3749

-- Define the initial conditions
def initial_fine : ℝ := 0.05

-- Define the recursive function to compute the fine, selecting the smaller increase
def daily_fine (n : ℕ) : ℝ :=
  match n with
  | 0       => initial_fine
  | (n + 1) => min (daily_fine n + 0.30) (daily_fine n * 2)

-- Define the proof problem that the fine on the fifth day is $0.70
theorem fine_on_fifth_day : daily_fine 5 = 0.70 :=
  sorry

end fine_on_fifth_day_l3_3749


namespace quadratic_inequality_solution_l3_3180

theorem quadratic_inequality_solution (m : ℝ) : 
  (∀ x : ℝ, x^2 + m * x + 1 ≥ 0) ↔ (-2 ≤ m ∧ m ≤ 2) :=
by
  sorry

end quadratic_inequality_solution_l3_3180


namespace num_elements_intersection_set_l3_3908

def is_on_unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1
def is_on_parabola (x y : ℝ) : Prop := y = 4*x^2 - 1

def set_A : Set (ℝ × ℝ) := { p : ℝ × ℝ | is_on_unit_circle p.1 p.2 }
def set_B : Set (ℝ × ℝ) := { p : ℝ × ℝ | is_on_parabola p.1 p.2 }

def intersection_set : Set (ℝ × ℝ) := set_A ∩ set_B

theorem num_elements_intersection_set :
  Finset.card (Set.toFinset intersection_set) = 3 :=
by
  sorry

end num_elements_intersection_set_l3_3908


namespace count_divisors_36_432_l3_3916

def is_divisor (n : ℕ) (d : ℕ) : Prop := d ∣ n

theorem count_divisors_36_432 :
  (finset.filter (is_divisor 36432) (finset.range 10)).card = 7 :=
by
  sorry

end count_divisors_36_432_l3_3916


namespace currency_notes_total_l3_3244

theorem currency_notes_total (num_50_notes total_amount remaining_amount num_100_notes : ℕ) 
  (h1 : remaining_amount = total_amount - (num_50_notes * 50))
  (h2 : num_50_notes = 3500 / 50)
  (h3 : total_amount = 5000)
  (h4 : remaining_amount = 1500)
  (h5 : num_100_notes = remaining_amount / 100) : 
  num_50_notes + num_100_notes = 85 :=
by sorry

end currency_notes_total_l3_3244


namespace find_value_of_sum_l3_3622

noncomputable theory

variables (p q r s : ℝ)

-- Conditions
def distinct_real_numbers : Prop := p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s
def roots_eq1 : Prop := r + s = 12 * p ∧ r * s = -8 * q
def roots_eq2 : Prop := p + q = 12 * r ∧ p * q = -8 * s

-- Goal
theorem find_value_of_sum (h1 : distinct_real_numbers p q r s)
                          (h2 : roots_eq1 p q r s)
                          (h3 : roots_eq2 p q r s) :
    p + q + r + s = 1248 :=
sorry

end find_value_of_sum_l3_3622


namespace gcd_qr_l3_3926

theorem gcd_qr (p q r : ℕ) (hpq : Nat.gcd p q = 210) (hpr : Nat.gcd p r = 770) : Nat.gcd q r = 70 := sorry

end gcd_qr_l3_3926


namespace incorrect_statement_B_for_linear_function_l3_3369

theorem incorrect_statement_B_for_linear_function :
  ∀ x y : ℝ, (y = -2 * x + 2) →
    (¬ (x = 2 ∧ y = 0)) :=
begin
  sorry
end

end incorrect_statement_B_for_linear_function_l3_3369


namespace graph_passes_through_point_l3_3439

theorem graph_passes_through_point (x y : ℝ) :
  ((x = y⁻¹) ∧ y = 1 ∧ x = 1) ↔
  ((x = 1) → (y = 1⁻¹) ∧ (∀ z, (z = 2^1 → y ≠ 1) ∧ (z = log 2 1 → y ≠ 1) ∧ (z = tan 1 → y ≠ 1))) := 
by
  sorry

end graph_passes_through_point_l3_3439


namespace construct_triangle_l3_3099

theorem construct_triangle 
  (A B C : Type) 
  (BC AC : ℝ) 
  (α β : ℝ) 
  (h1 : BC = 4)
  (h2 : AC = 5)
  (h3 : β = 2 * α) :
  ∃ (A B C : Point), is_triangle A B C ∧ side_length B C = 4 ∧ side_length A C = 5 ∧ angle B A C = 2 * angle A B C :=
sorry

end construct_triangle_l3_3099


namespace num_possible_values_ω_l3_3544

def period (ω : ℝ) : ℝ := 2 * Real.pi / ω

theorem num_possible_values_ω :
  {ω : ℤ | 200 * Real.pi < ω ∧ ω < 100 * Real.pi}.card = 314 :=
by sorry

end num_possible_values_ω_l3_3544


namespace number_of_true_propositions_is_one_l3_3139

-- Define propositions
def prop1 (a b c : ℝ) : Prop := a > b ∧ c ≠ 0 → a * c > b * c
def prop2 (a b c : ℝ) : Prop := a > b → a * c^2 > b * c^2
def prop3 (a b c : ℝ) : Prop := a * c^2 > b * c^2 → a > b
def prop4 (a b : ℝ) : Prop := a > b → (1 / a) < (1 / b)
def prop5 (a b c d : ℝ) : Prop := a > b ∧ b > 0 ∧ c > d → a * c > b * d

-- The main theorem stating the number of true propositions
theorem number_of_true_propositions_is_one (a b c d : ℝ) :
  (prop3 a b c) ∧ (¬ prop1 a b c) ∧ (¬ prop2 a b c) ∧ (¬ prop4 a b) ∧ (¬ prop5 a b c d) :=
by
  sorry

end number_of_true_propositions_is_one_l3_3139


namespace arithmetic_progression_sum_and_squares_l3_3325

theorem arithmetic_progression_sum_and_squares (a1 a2 a3 : ℚ)
  (h1 : a1 + a2 + a3 = 2)
  (h2 : a1^2 + a2^2 + a3^2 = 14 / 9) :
  {a1, a2, a3} = {1/3, 2/3, 1} ∨ {a1, a2, a3} = {1, 2/3, 1/3} :=
by
  sorry

end arithmetic_progression_sum_and_squares_l3_3325


namespace tuple_possible_values_2014_l3_3265

def f : ℕ → ℕ := sorry

axiom f_property1 : f 1 = 1
axiom f_property2 : ∀ (a b : ℕ), a ≤ b → f a ≤ f b
axiom f_property3 : ∀ (a : ℕ), f (2 * a) = f a + 1

theorem tuple_possible_values_2014 : 
  ∃ (n : ℕ), n = 1007 ∧ 
  ∀ (tuple_set : set (vector ℕ 2014)), 
  (∀ (t : (vector ℕ 2014)), t ∈ tuple_set ↔ (∀ i, (1 ≤ i ∧ i ≤ 2014) → (f i))) → tuple_set.finite → tuple_set.card = n :=
sorry

end tuple_possible_values_2014_l3_3265


namespace number_of_multiples_of_4_l3_3450

theorem number_of_multiples_of_4 (a b : ℤ) (h1 : 100 < a) (h2 : b < 500) (h3 : a % 4 = 0) (h4 : b % 4 = 0) : 
  ∃ n : ℤ, n = 99 :=
by
  sorry

end number_of_multiples_of_4_l3_3450


namespace triangle_angle_division_l3_3483

theorem triangle_angle_division 
  (A B C D M: Type) 
  [triangle: Triangle ABC]
  (CD_perp_BM: Perpendicular CD BM)
  (median_CM: Median CM)
  (equal_division: ∀ {P Q R: Type}, CM = ∠ B C D = ∠ D C M = ∠ A C M )
  : ∠ A = 30° ∧ ∠ B = 60° ∧ ∠ C = 90° := 
sorry

end triangle_angle_division_l3_3483


namespace total_amount_due_l3_3761

noncomputable def original_bill : ℝ := 500
noncomputable def late_charge_rate : ℝ := 0.02
noncomputable def annual_interest_rate : ℝ := 0.05

theorem total_amount_due (n : ℕ) (initial_amount : ℝ) (late_charge_rate : ℝ) (interest_rate : ℝ) : 
  initial_amount = 500 → 
  late_charge_rate = 0.02 → 
  interest_rate = 0.05 → 
  n = 3 → 
  (initial_amount * (1 + late_charge_rate)^n * (1 + interest_rate) = 557.13) :=
by
  intros h_initial_amount h_late_charge_rate h_interest_rate h_n
  sorry

end total_amount_due_l3_3761


namespace trig_proof_l3_3876

theorem trig_proof (α : ℝ) (h1 : Real.tan α = Real.sqrt 3) (h2 : π < α ∧ α < 3 * π / 2) :
  Real.cos (2 * α) - Real.sin (π / 2 + α) = 0 :=
sorry

end trig_proof_l3_3876


namespace even_three_digit_numbers_with_digit_sum_24_l3_3186

theorem even_three_digit_numbers_with_digit_sum_24 :
  { n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (∃ a b c : ℕ, n = 100*a + 10*b + c ∧ a + b + c = 24 ∧ n % 2 = 0) } = 3 :=
sorry

end even_three_digit_numbers_with_digit_sum_24_l3_3186


namespace curve_intersects_self_at_6_6_l3_3829

-- Definitions for the given conditions
def x (t : ℝ) : ℝ := t^2 - 3
def y (t : ℝ) : ℝ := t^4 - t^2 - 9 * t + 6

-- Lean statement stating that the curve intersects itself at the coordinate (6, 6)
theorem curve_intersects_self_at_6_6 :
  ∃ t1 t2 : ℝ, t1 ≠ t2 ∧ x t1 = x t2 ∧ y t1 = y t2 ∧ x t1 = 6 ∧ y t1 = 6 :=
by
  sorry

end curve_intersects_self_at_6_6_l3_3829


namespace pentagon_projection_l3_3008

theorem pentagon_projection (a : ℝ) (h1 : 1 < a) (h2 : a < 5) : 
  sqrt 5 - 2 < a ∧ a < sqrt 5 := 
sorry

end pentagon_projection_l3_3008


namespace volume_ratio_l3_3935

theorem volume_ratio (R r : ℝ) (h : (4 * real.pi * R^2) / (4 * real.pi * r^2) = 4 / 9) :
  (4 / 3 * real.pi * R^3) / (4 / 3 * real.pi * r^3) = 8 / 27 :=
by
  sorry

end volume_ratio_l3_3935


namespace pipe_filling_problem_l3_3705

theorem pipe_filling_problem (x : ℝ) (h : (2 / 15) * x + (1 / 20) * (10 - x) = 1) : x = 6 :=
sorry

end pipe_filling_problem_l3_3705


namespace find_initial_weight_solution_Y_l3_3291

-- Define the initial weight of solution Y
def initial_weight (W : ℝ) : Prop :=
  let concentration_initial_Y := 0.30 * W in
  let evaporation_kilograms := 3 in
  let remaining_weight := W - evaporation_kilograms in
  let added_solution_Y := 3 in
  let added_liquid_X := 0.30 * added_solution_Y in
  let new_total_weight := remaining_weight + added_solution_Y in
  let new_liquid_X := concentration_initial_Y + added_liquid_X in
  let final_concentration := new_liquid_X / new_total_weight in
  (final_concentration = 0.4125) -> (W = 8)

-- Proof omitted
theorem find_initial_weight_solution_Y : ∃ W : ℝ, initial_weight W :=
begin
  use 8,
  sorry
end

end find_initial_weight_solution_Y_l3_3291


namespace square_of_1024_l3_3824

theorem square_of_1024 : (1024 : ℤ)^2 = 1048576 := by
  let a := 1020
  let b := 4
  have h : (1024 : ℤ) = a + b := by
    norm_num
  rw [h] 
  norm_num
  sorry
  -- expand (a+b)^2 = a^2 + 2ab + b^2
  -- prove that 1020^2 = 1040400
  -- prove that 2 * 1020 * 4 = 8160
  -- prove that 4^2 = 16
  -- sum these results 
  -- result = 1048576

end square_of_1024_l3_3824


namespace probability_matching_pair_l3_3377

theorem probability_matching_pair :
  let total_shoes := 200
  let total_pairs := 100
  let total_combinations := (total_shoes * (total_shoes - 1)) / 2
  let matching_combinations := total_pairs
  (matching_combinations : ℚ) / (total_combinations : ℚ) = 1 / 199 :=
by
  let total_shoes := 200
  let total_pairs := 100
  let total_combinations := (total_shoes * (total_shoes - 1)) / 2
  let matching_combinations := total_pairs
  have h1 : total_combinations = (200 * 199) / 2 := by rfl
  have h2 : matching_combinations = 100 := by rfl
  calc
    (matching_combinations : ℚ) / (total_combinations : ℚ)
        = 100 / ((200 * 199) / 2) : by rw [h1, h2]
    ... = 100 / 19900 : by norm_num
    ... = 1 / 199 : by norm_num

end probability_matching_pair_l3_3377


namespace value_of_f_neg2013_plus_f2014_l3_3157

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x < 2 then log x + 1 else f (x % 2)

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_periodic : ∀ x : ℝ, 0 ≤ x → f (x + 2) = f x

theorem value_of_f_neg2013_plus_f2014 : f (-2013) + f 2014 = -1 := by
  sorry

end value_of_f_neg2013_plus_f2014_l3_3157


namespace arithmetic_sequence_S9_l3_3150

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Assume the sequence is an arithmetic sequence
def arithmetic_seq (a : ℕ → ℝ) : Prop := ∀ n m : ℕ, n < m → a (m - n) = a m - a n

-- Conditions: S is the sum of the first n terms, and a_2 + a_8 = 10
axiom sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2 
axiom condition_a2_a8 (a : ℕ → ℝ) : a 2 + a 8 = 10

-- Question: Prove that S_9 = 45
theorem arithmetic_sequence_S9 (a : ℕ → ℝ) (S : ℕ → ℝ) 
    [arithmetic_seq a] [sum_of_first_n_terms a S] [condition_a2_a8 a] : S 9 = 45 := 
by 
    sorry

end arithmetic_sequence_S9_l3_3150


namespace unused_paper_area_correct_l3_3700

-- Definitions derived from problem conditions
def square_side : ℝ := 10     -- Side length of the square in cm
def num_circles : ℕ := 9      -- Number of circles

-- Derived quantities
def circle_diameter : ℝ := square_side / 3  -- Diameter of one circle
def circle_radius : ℝ := circle_diameter / 2 -- Radius of one circle

-- Areas
def square_area : ℝ := square_side ^ 2
def circle_area : ℝ := Real.pi * (circle_radius ^ 2)
def total_circle_area : ℝ := num_circles * circle_area

-- Calculation of unused paper area
def unused_area : ℝ := square_area - total_circle_area

-- Approximate value of unused area with π approximated to 3.14
def pi_approx : ℝ := 3.14
def circle_area_approx : ℝ := pi_approx * (circle_radius ^ 2)
def total_circle_area_approx : ℝ := num_circles * circle_area_approx
def unused_area_approx : ℝ := square_area - total_circle_area_approx

theorem unused_paper_area_correct : unused_area_approx = 21.5 := by
  sorry

end unused_paper_area_correct_l3_3700


namespace multiplication_in_base7_l3_3082

def base7_to_base10 (n : list ℕ) : ℕ :=
  n.reverse.enum.map (λ ⟨i, a⟩, a * 7^i).sum

def base10_to_base7 (n : ℕ) : list ℕ :=
  let rec loop (n : ℕ) (acc : list ℕ) : list ℕ :=
    if n = 0 then acc else loop (n / 7) ((n % 7) :: acc)
  loop n []

def multiply_base7 (a b : list ℕ) : list ℕ :=
  base10_to_base7 ((base7_to_base10 a) * (base7_to_base10 b))

theorem multiplication_in_base7 :
  multiply_base7 [3, 2, 4] [3] = [1, 3, 0, 5] :=
by
  sorry

end multiplication_in_base7_l3_3082


namespace sum_sin_squared_l3_3859

theorem sum_sin_squared (n : ℕ) (hn : 0 < n) : 
  let x := (Real.pi / (2 * n : ℝ)) in
  (Finset.sum (Finset.range n) (λ k, Real.sin (x * (k + 1 : ℝ)) ^ 2))
  = (n + 1) / 2 := 
by
  sorry

end sum_sin_squared_l3_3859


namespace coeff_is_neg4_l3_3579

noncomputable def find_coeff (a : ℝ) : ℝ :=
  let poly := ((1 : polynomial ℝ) + polynomial.X) * ((2 * polynomial.X^2) + (a * polynomial.X) + 1)
  in poly.coeff 2 -- Returns the coefficient of the x^2 term 

theorem coeff_is_neg4 : ∃ (a : ℝ), find_coeff a = -4 :=
begin
  existsi (-6),
  sorry, -- The proof is omitted as specified
end

end coeff_is_neg4_l3_3579


namespace intersection_points_distance_l3_3423

theorem intersection_points_distance (A B C D : ℝ^3) (r : ℝ) (S : Set (ℝ^3)) :
  -- Conditions for a regular tetrahedron A, B, C, D with edge length 1:
  dist A B = 1 ∧ dist A C = 1 ∧ dist A D = 1 ∧ dist B C = 1 ∧ dist B D = 1 ∧ dist C D = 1 ∧

  -- Six spheres with each edge as a diameter:
  ∀ p ∈ S, (dist A p = r) ∨ (dist B p = r) ∨ (dist C p = r) ∨ (dist D p = r) ∨
                     (dist (B + C) / 2 p = r/2) ∨ (dist (B + D) / 2 p = r/2) ∨ (dist (C + D) / 2 p = r/2) →
  -- Prove that there are two points in S such that their distance is sqrt(6)/6:
  ∃ p₁ p₂ ∈ S, dist p₁ p₂ = √6 / 6 :=
by
  sorry

end intersection_points_distance_l3_3423


namespace probability_same_foot_l3_3212

theorem probability_same_foot (pairs : ℕ := 4) :
  let shoes := 2 * pairs
  let total_ways := Nat.choose shoes 2
  let ways_same_foot := 2 * Nat.choose pairs 2
  (ways_same_foot : ℚ) / (total_ways : ℚ) = 3 / 7 :=
by
  let shoes := 2 * pairs
  let total_ways := Nat.choose shoes 2
  let ways_same_foot := 2 * Nat.choose pairs 2
  have h1 : total_ways = 28 := by
    simp [shoes]
  have h2 : ways_same_foot = 12 := by
    simp [pairs]
  exact calc
    (ways_same_foot : ℚ) / (total_ways : ℚ) = 12 / 28 : by congr
    ... = 3 / 7 : by norm_num

end probability_same_foot_l3_3212


namespace intersection_volume_of_reflected_tetrahedron_l3_3153

theorem intersection_volume_of_reflected_tetrahedron (V : ℝ) (h : V = 1) : ∃ I : ℝ, I = 1 / 2 :=
by
  use 1 / 2
  sorry

end intersection_volume_of_reflected_tetrahedron_l3_3153


namespace rectangular_box_has_volume_240_l3_3056

theorem rectangular_box_has_volume_240 (x : ℕ) (V : ℕ) (h1 : 2 * x * 3 * x * 5 * x = V) :
  V ∈ {60, 90, 120, 180, 240} → V = 240 :=
by {
  intros h2,
  by_cases V = 60,
  { exfalso, -- contradiction will show that this case is impossible
    revert h,
    sorry },
  by_cases V = 90,
  { exfalso,
    revert h,
    sorry },
  by_cases V = 120,
  { exfalso,
    revert h,
    sorry },
  by_cases V = 180,
  { exfalso,
    revert h,
    sorry },
  by_cases V = 240,
  { assumption }
}

end rectangular_box_has_volume_240_l3_3056


namespace alpha_i_l3_3623

noncomputable def i : ℂ := complex.I

def alpha (z : ℂ) : ℕ :=
  Nat.find (exists_pow_eq_one z)

theorem alpha_i : alpha i = 4 := by
  sorry

end alpha_i_l3_3623


namespace largest_minus_smallest_eq_13_l3_3073

theorem largest_minus_smallest_eq_13 :
  let a := (-1 : ℤ) ^ 3
  let b := (-1 : ℤ) ^ 2
  let c := -(2 : ℤ) ^ 2
  let d := (-3 : ℤ) ^ 2
  max (max a (max b c)) d - min (min a (min b c)) d = 13 := by
  sorry

end largest_minus_smallest_eq_13_l3_3073


namespace pencils_on_desk_l3_3692

theorem pencils_on_desk (pencils_in_drawer pencils_on_desk_initial pencils_total pencils_placed : ℕ)
  (h_drawer : pencils_in_drawer = 43)
  (h_desk_initial : pencils_on_desk_initial = 19)
  (h_total : pencils_total = 78) :
  pencils_placed = 16 := by
  sorry

end pencils_on_desk_l3_3692


namespace number_of_divisors_30030_l3_3127

theorem number_of_divisors_30030 : 
  let n := 30030 in 
  let prime_factors := [2, 3, 5, 7, 11, 13] in
  (∀ p ∈ prime_factors, nat.prime p) → 
  (∀ p ∈ prime_factors, n % p = 0) → 
  (∀ p₁ p₂ ∈ prime_factors, p₁ ≠ p₂ → ∀ m, n % (p₁ * p₂ * m) ≠ 0) →
  (∀ p ∈ prime_factors, ∃ m, n = p ^ 1 * m ∧ nat.gcd(p, m) = 1) → 
  ∃ t, t = 64 ∧ t = ∏ p in prime_factors.to_finset, ((1 : ℕ) + 1) :=
by
  intro n prime_factors hprimes hdivisors hunique hfactored
  use 64
  rw list.prod_eq_foldr at *
  suffices : 64 = list.foldr (λ _ r, 2 * r) 1 prime_factors, from this
  simp [list.foldr, prime_factors]
  sorry

end number_of_divisors_30030_l3_3127


namespace keith_spent_on_tires_l3_3967

noncomputable def money_spent_on_speakers : ℝ := 136.01
noncomputable def money_spent_on_cd_player : ℝ := 139.38
noncomputable def total_expenditure : ℝ := 387.85
noncomputable def total_spent_on_speakers_and_cd_player : ℝ := money_spent_on_speakers + money_spent_on_cd_player
noncomputable def money_spent_on_new_tires : ℝ := total_expenditure - total_spent_on_speakers_and_cd_player

theorem keith_spent_on_tires :
  money_spent_on_new_tires = 112.46 :=
by
  sorry

end keith_spent_on_tires_l3_3967


namespace cistern_fill_time_l3_3406

theorem cistern_fill_time:
  (F E net_rate : ℝ) (time_to_fill: ℝ)
  (hF : F = 1 / 3)
  (hE : E = 1 / 9)
  (h_net_rate : net_rate = F - E)
  (h_time : time_to_fill = 1 / net_rate) :
  time_to_fill = 4.5 :=
by
  sorry

end cistern_fill_time_l3_3406


namespace part1_part2_l3_3223

variable {a : ℕ → ℤ}

-- Conditions
def common_difference (d : ℤ) := d > 0
def a_2_plus_a_6 := a 2 + a 6 = 8
def a_3_times_a_5 := a 3 * a 5 = 12
def a_general_term := ∀ n, a n = 2 * n - 4

-- Sequence definitions
def b (n : ℕ) := (a n + 4) * 2^n

-- Sum function definition
def T (n : ℕ) := Finset.sum (Finset.range n) (λ i, b (i + 1))

-- Theorem statements
theorem part1 (d : ℤ) (h1 : common_difference d) (h2 : a_2_plus_a_6) (h3 : a_3_times_a_5) : a_general_term :=
by
  sorry

theorem part2 (n : ℕ) (d : ℤ) (h1 : common_difference d) (h2 : a_2_plus_a_6) (h3 : a_3_times_a_5) (h4 : a_general_term) :
  T n = (n - 1) * 2^(n + 2) + 4 :=
by
  sorry

end part1_part2_l3_3223


namespace ratio_PD_PC_l3_3647

-- Define the setup conditions
variables {A B C D P : Type} -- Points that define the isosceles trapezoid ABCD and point P on AB
variables (AB CD AP BP : ℝ) -- Lengths involved
variables (angle_CPD angle_PAD : ℝ) -- Angles in radians
variable  [Isosceles_Trapezoid :  ABCD]
variable  [AP_BP_Ratio : AP / BP = 4] 
variable  [angle_Equivalence : angle_CPD = angle_PAD] 

-- Define the desired proof statement
theorem ratio_PD_PC : PD / PC = 2 :=
sorry

end ratio_PD_PC_l3_3647


namespace conjugate_of_z_l3_3505

open Complex

theorem conjugate_of_z (z : ℂ) (h : (1 - I) * z = 2 + I) : conj z = (1 / 2) - (3 / 2) * I :=
by
  sorry

end conjugate_of_z_l3_3505


namespace no_valid_a_exists_l3_3179

theorem no_valid_a_exists 
  (a : ℝ)
  (h1: ∀ x : ℝ, x^2 + 2*(a+1)*x - (a-1) = 0 → (1 < x ∨ x < 1)) :
  false := by
  sorry

end no_valid_a_exists_l3_3179


namespace incorrect_statements_l3_3956

-- Definitions needed from the problem
def is_ratio_arithmetic_seq (p : ℕ → ℤ) (k : ℤ) : Prop :=
∀ n ≥ 2, (p (n+1)) / (p n) - (p n) / (p (n-1)) = k

def is_geometric_seq (a : ℕ → ℤ) (q : ℤ) : Prop :=
∀ n, a(n+1) = q * a(n)

def is_arithmetic_seq (b : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n, b(n+1) = b(n) + d

-- Main statement
theorem incorrect_statements : 
  (∃ (a : ℕ → ℤ) (q : ℤ), is_geometric_seq a q ∧ ¬is_ratio_arithmetic_seq a 1) ∧
  (∃ (b : ℕ → ℤ), is_arithmetic_seq b 0 ∧ is_ratio_arithmetic_seq b 0) ∧
  (∃ (a b : ℕ → ℤ) (q : ℤ), is_geometric_seq b q ∧ is_arithmetic_seq a 0 ∧ ∀ n, a(n) = 0 ∧ ¬is_ratio_arithmetic_seq (λ n, a(n) * b(n)) 0) ∧
  (∀ (a : ℕ → ℤ), a 1 = 1 → a 2 = 1 → (∀ n ≥ 2, a (n+1) = a n + a (n-1)) → ¬is_ratio_arithmetic_seq a 0) :=
begin
  sorry
end

end incorrect_statements_l3_3956


namespace find_coefficients_l3_3620

noncomputable def f (x : ℝ) : ℝ := x^3 + 2 * x^2 + 3 * x + 4

noncomputable def h (a b c x : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

theorem find_coefficients :
  ∃ a b c : ℝ, (∀ s : ℝ, f s = 0 → h a b c (s^3) = 0) ∧
    (a, b, c) = (-6, -9, 20) :=
sorry

end find_coefficients_l3_3620


namespace each_child_consumes_3_bottles_per_day_l3_3048

noncomputable def bottles_per_child_per_day : ℕ :=
  let first_group := 14
  let second_group := 16
  let third_group := 12
  let fourth_group := (first_group + second_group + third_group) / 2
  let total_children := first_group + second_group + third_group + fourth_group
  let cases_of_water := 13
  let bottles_per_case := 24
  let initial_bottles := cases_of_water * bottles_per_case
  let additional_bottles := 255
  let total_bottles := initial_bottles + additional_bottles
  let bottles_per_child := total_bottles / total_children
  let days := 3
  bottles_per_child / days

theorem each_child_consumes_3_bottles_per_day :
  bottles_per_child_per_day = 3 :=
by
  sorry

end each_child_consumes_3_bottles_per_day_l3_3048


namespace find_interest_rate_l3_3853

theorem find_interest_rate
  (P : ℝ)  -- Principal amount
  (A : ℝ)  -- Final amount
  (T : ℝ)  -- Time period in years
  (H1 : P = 1000)
  (H2 : A = 1120)
  (H3 : T = 2.4)
  : ∃ R : ℝ, (A - P) = (P * R * T) / 100 ∧ R = 5 :=
by
  -- Proof with calculations to be provided here
  sorry

end find_interest_rate_l3_3853


namespace tan_increasing_intervals_l3_3117

theorem tan_increasing_intervals (k : ℤ) :
  ∃ I : set ℝ, I = { x : ℝ | 2 * k - 5 / 3 < x ∧ x < 2 * k + 1 / 3 } ∧
    ∀ x ∈ I, ∀ y ∈ I, x < y → 
      (tan ((π / 2) * x + π / 3) < tan ((π / 2) * y + π / 3)) :=
by
  sorry

end tan_increasing_intervals_l3_3117


namespace angle_between_PB_and_AC_is_90_degrees_l3_3961

open EuclideanGeometry AffineSpace

-- Definitions of the given conditions and the theorem
theorem angle_between_PB_and_AC_is_90_degrees
  (A B C D P : Point)
  (h_square : square A B C D)
  (h_outside : P ∉ plane A B C)
  (h_perpendicular : ∀ Q : Point, Q ∈ (plane A B C) → ∠ P Q = π / 2)
  (h_equal : distance P A = distance A B) :
  ∠ (P - B) (A - B) = π / 2 :=
sorry

end angle_between_PB_and_AC_is_90_degrees_l3_3961


namespace evaluate_f_5_minus_f_neg_5_l3_3191

def f (x : ℝ) : ℝ := x^4 + x^2 + 5 * x^3

theorem evaluate_f_5_minus_f_neg_5 : f 5 - f (-5) = 1250 :=
by 
  sorry

end evaluate_f_5_minus_f_neg_5_l3_3191


namespace diagonal_can_not_be_1_diagonal_can_be_2_diagonal_can_be_1001_l3_3319

theorem diagonal_can_not_be_1 (P : ℝ) (d1 d2 : ℝ) (P_eq : P = 2004) (d1_eq : d1 = 1001) (d2_eq : d2 = 1) : 
  ¬(2 * (d1 + d2) > P) :=
by
  sorry

theorem diagonal_can_be_2 (P : ℝ) (d1 d2 : ℝ) (P_eq : P = 2004) (d1_eq : d1 = 1001) (d2_eq : d2 = 2) : 
  ∃ K, 2 * (d1 + K + d2) = P :=
by
  sorry

theorem diagonal_can_be_1001 (P : ℝ) (d1 d2 : ℝ) (P_eq : P = 2004) (d1_eq : d1 = 1001) (d2_eq : d2 = 1001) : 
  ∃ θ, 2 * ((d1 * cos θ) + (d2 * sin θ)) = P :=
by
  sorry

end diagonal_can_not_be_1_diagonal_can_be_2_diagonal_can_be_1001_l3_3319


namespace quadratic_eq_roots_l3_3977

open Complex

noncomputable def eta : ℂ := sorry -- eta is a complex number

def gamma (η : ℂ) : ℂ := η + η^2 + η^3
def delta (η : ℂ) : ℂ := η^4 + η^5 + η^6 + η^7

lemma eta_property : eta^8 = 1 ∧ eta ≠ 1 := sorry

lemma quadratic_roots : (gamma eta)^2 + (delta eta) = -1 ∧ (gamma eta) * (delta eta) = 1 := sorry

theorem quadratic_eq_roots (γ δ : ℂ) (hγ : γ = gamma eta) (hδ : δ = delta eta) :
    ∃ a b : ℂ, (a = 1 ∧ b = 1) ∧ (γ + δ + a = 0 ∧ γ * δ + b = 0) :=
begin
  use [1, 1],
  split,
  { split; refl, },
  { rw [←hγ, ←hδ],
    have h1 : gamma eta + delta eta = -1,
    { sorry, }, -- This follows from eta_property and the definitions of gamma and delta
    have h2 : gamma eta * delta eta = 1,
    { sorry, }, -- This also follows from eta_property and the definitions of gamma and delta
    exact ⟨h1, h2⟩, }
end

end quadratic_eq_roots_l3_3977


namespace circle_equation_k_range_l3_3198

theorem circle_equation_k_range (k : ℝ) :
  ∀ x y: ℝ, x^2 + y^2 + 4*k*x - 2*y + 4*k^2 - k = 0 →
  k > -1 := 
sorry

end circle_equation_k_range_l3_3198


namespace prob_second_day_A_l3_3298

-- Definitions based on conditions
def prob_first_day_A : ℝ := 1 / 2
def prob_first_day_B : ℝ := 1 / 2
def prob_second_day_given_A_first_day_A : ℝ := 0.7
def prob_second_day_given_B_first_day_A : ℝ := 0.5

-- Required proof statement
theorem prob_second_day_A :
  (prob_first_day_A * prob_second_day_given_A_first_day_A) +
  (prob_first_day_B * prob_second_day_given_B_first_day_A) = 0.6 :=
by
  sorry

end prob_second_day_A_l3_3298


namespace tangent_line_at_x_1_monotonicity_range_a_l3_3168

def f (x a : ℝ) : ℝ := x * abs (x^2 - a)

theorem tangent_line_at_x_1 (a : ℝ) (h : a = 2) : 
  (∃ (m b : ℝ), m = -1 ∧ b = 2 ∧ ∀ (x y : ℝ), y = f x a → y = -x + 2) :=
sorry

theorem monotonicity (a : ℝ) (h : 0 < a) : 
  (∀ x : ℝ, 
    (-∞ < x ∧ x < -sqrt a) → 0 < f' x a) ∧ 
  (∀ x : ℝ, 
    (-sqrt (a / 3) < x ∧ x < sqrt (a / 3)) → 0 < f' x a) ∧ 
  (∀ x : ℝ, 
    (sqrt a < x) → 0 < f' x a) ∧ 
  (∀ x : ℝ, 
    (-sqrt a < x ∧ x < -sqrt (a / 3)) → f' x a < 0) ∧ 
  (∀ x : ℝ, 
    (sqrt (a / 3) < x ∧ x < sqrt a) → f' x a < 0) :=
sorry

theorem range_a (h : 0 < a) : 
  (∀ x₁ x₂ : ℝ, 
    (0 ≤ x₁ ∧ x₁ ≤ 1) → (0 ≤ x₂ ∧ x₂ ≤ 1) → 
    abs (f x₁ a - f x₂ a) < sqrt 2 / 2) ↔ 
    a ∈ (Ioo (1 - sqrt 2 / 2) (3 / 2)) :=
sorry

end tangent_line_at_x_1_monotonicity_range_a_l3_3168


namespace unique_center_size_of_regular_ngon_l3_3493
noncomputable theory
open_locale real

-- Problem statement and conditions
variables (n : ℕ) (points : fin n → ℝ × ℝ)
-- Assume n >= 3
axiom n_ge_3 : 3 ≤ n

-- Definition of the operation on points by rotating around their midpoint
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def rotate_around_midpoint (A B : ℝ × ℝ) (θ : ℝ) : ℝ × ℝ × (ℝ × ℝ) :=
let M := midpoint A B in
let cosθ := real.cos θ in
let sinθ := real.sin θ in
let A' := (cosθ * (A.1 - M.1) - sinθ * (A.2 - M.2) + M.1, sinθ * (A.1 - M.1) + cosθ * (A.2 - M.2) + M.2) in
let B' := (cosθ * (B.1 - M.1) - sinθ * (B.2 - M.2) + M.1, sinθ * (B.1 - M.1) + cosθ * (B.2 - M.2) + M.2) in
(A', B')

-- Theorem: If points can be transformed into a regular n-gon, its center and size are unique
theorem unique_center_size_of_regular_ngon (h : ∃ (operations : list (fin n × fin n × ℝ)),
  (∃ points' : fin n → ℝ × ℝ, list.foldl (λ pts op, 
    let ptA := pts.snd (op.fst).fst in
    let ptB := pts.snd (op.fst).snd in
    let (new_ptA, new_ptB) := rotate_around_midpoint ptA ptB op.snd in
    λ i, if i = op.fst.fst then new_ptA else if i = op.fst.snd then new_ptB else pts i
  ) points operations = points' ∧
  ∃ O : ℝ × ℝ, ∀ i j, dist ((points' i).fst, (points' i).snd) O = dist ((points' j).fst, (points' j).snd)))
  : ∀ (O₁ O₂ : ℝ × ℝ) (r₁ r₂ : ℝ), 
   (∀ i, dist ((points i).fst, (points i).snd) O₁ = r₁) 
  ∧ (∀ i, dist ((points i).fst, (points i).snd) O₂ = r₂) → (O₁ = O₂ ∧ r₁ = r₂) :=
begin
  sorry
end

end unique_center_size_of_regular_ngon_l3_3493


namespace part_1_part_2_l3_3551

open Set

variable (U : Set ℝ)
variable (A : Set ℝ)
variable (B : Set ℝ)

-- Condition definitions
def U : Set ℝ := univ
def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {x | -1 < x ∧ x < 1}

-- Proof for (1)
theorem part_1 : A ∩ B = B ∧ B ⊆ A := by
  sorry

-- Proof for (2)
theorem part_2 : (A \ B) = {x | x ≤ -1 ∨ (1 ≤ x ∧ x < 2)} := by
  sorry

end part_1_part_2_l3_3551


namespace problem_statement_l3_3743

def vector := (ℝ × ℝ × ℝ)

def are_parallel (v1 v2 : vector) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ v1 = (k * v2.1, k * v2.2, k * v2.3)

def are_perpendicular (v1 v2 : vector) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3 = 0

theorem problem_statement :
  (are_parallel (2, 3, -1) (-2, -3, 1)) ∧
  (¬ are_perpendicular (1, -1, 2) (6, 4, -1)) ∧
  (are_perpendicular (2, 2, -1) (-3, 4, 2)) ∧
  (¬ are_parallel (0, 3, 0) (0, -5, 0)) :=
by
  sorry

end problem_statement_l3_3743


namespace canteen_distance_proof_l3_3414

noncomputable theory

def girls_camp_distance := 300 -- Distance of the girls' camp from the road
def boys_camp_distance := 500 -- Distance between the girls' camp and the boys' camp
def canteen_is_equidistant (x : ℝ) := 
  let ag := girls_camp_distance
  let bg := boys_camp_distance
  let ab := 400 -- This is inferred from a 3-4-5 triangle relationship
  let ac := ab - x in
  (ag^2 + ac^2 = x^2)

theorem canteen_distance_proof : ∃ x, canteen_is_equidistant x ∧ x = 312.5 :=
by -- The proof goes here
  sorry

end canteen_distance_proof_l3_3414


namespace area_of_triangle_l3_3958

theorem area_of_triangle (A B C : ℝ) (a b : ℝ) (A_eq : A = 60) (b_eq : b = 4) (a_eq : a = 2 * Real.sqrt 3) : 
  let C := 30 in
  0.5 * a * b * Real.sin (Real.pi / 6) = 2 * Real.sqrt 3 :=
by 
  sorry

end area_of_triangle_l3_3958


namespace x_intercepts_count_l3_3851

noncomputable def count_x_intercepts : ℕ :=
  let lower_bound_k := (10000 : ℝ) / Real.pi
  let upper_bound_k := (100000 : ℝ) / Real.pi
  (Real.floor upper_bound_k).toNat - (Real.floor lower_bound_k).toNat

theorem x_intercepts_count : count_x_intercepts = 28647 :=
by
  sorry

end x_intercepts_count_l3_3851


namespace sum_first_n_terms_l3_3867

variable (a : ℕ → ℕ)

axiom a1_condition : a 1 = 2
axiom diff_condition : ∀ n : ℕ, a (n + 1) - a n = 2^n

-- Define the sum of the first n terms of the sequence
noncomputable def S : ℕ → ℕ
| 0 => 0
| (n + 1) => S n + a (n + 1)

theorem sum_first_n_terms (n : ℕ) : S a n = 2^(n + 1) - 2 :=
by
  sorry

end sum_first_n_terms_l3_3867


namespace pure_imaginary_m_l3_3197

theorem pure_imaginary_m (m : ℝ) (h : (m^2 - m - 2 : ℝ) = 0) : (z : ℂ) = (m + 1) * complex.I :=
begin
  sorry
end

end pure_imaginary_m_l3_3197


namespace ellis_orion_card_difference_l3_3203

def total_cards : ℕ := 2500
def ratio_ellis : ℕ := 17
def ratio_orion : ℕ := 13
def total_ratio : ℕ := ratio_ellis + ratio_orion
def cards_per_part (total_cards : ℕ) (total_ratio : ℕ) : ℚ :=
  total_cards / total_ratio
def ellis_share (ratio_ellis : ℕ) (cards_per_part : ℚ) : ℕ :=
  ratio_ellis * (cards_per_part.to_nat)
def orion_share (ratio_orion : ℕ) (cards_per_part : ℚ) : ℕ :=
  ratio_orion * (cards_per_part.to_nat)
def card_difference (ellis_share : ℕ) (orion_share : ℕ) : ℕ :=
  ellis_share - orion_share

theorem ellis_orion_card_difference :
  card_difference (ellis_share ratio_ellis (cards_per_part total_cards total_ratio))
                  (orion_share ratio_orion (cards_per_part total_cards total_ratio)) = 332 :=
by 
  sorry

end ellis_orion_card_difference_l3_3203


namespace volume_of_revolution_triangle_l3_3086

noncomputable def volume_of_cone (r h : ℝ) : ℝ := (1 / 3) * π * r^2 * h

theorem volume_of_revolution_triangle:
  let r₁ := 4
  let h₁ := 3
  let V₁ := volume_of_cone r₁ h₁
  V₁ = 16 * π ∧
  
  let r₂ := 3
  let h₂ := 4
  let V₂ := volume_of_cone r₂ h₂
  V₂ = 12 * π := by 
  sorry

end volume_of_revolution_triangle_l3_3086


namespace problem1_l3_3158

theorem problem1 (x₀ : ℝ) (ω : ℝ) (hω : ω > 0) (hx₀ : x₀ + π/2) : 
  (∀ x : ℝ, f x = cos((ω * x) - π/6) ^ 2 - sin(ω * x) ^ 2 → 
  f (π / 12) = sqrt(3) / 2) ∧
  (∀ m : ℝ, (∀ x : ℝ, x ∈ set.Icc (-7 * π / 12) 0 → |f x - m| ≤ 1) → 
  m ∈ set.Icc (-1 / 4) (1 - sqrt(3) / 2)) 
  :=
sorry

end problem1_l3_3158
