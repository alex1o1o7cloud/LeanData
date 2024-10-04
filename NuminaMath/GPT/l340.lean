import Mathlib

namespace AB_side_length_l340_340771

noncomputable def P := (x : ℝ) × (y : ℝ)

def is_foot_perpendicular (P : P) (A B : P) : P := sorry

def equilateral_triangle (A B C : P) : Prop := sorry

theorem AB_side_length (A B C P Q R S : P)
  (h_equilateral : equilateral_triangle A B C)
  (h_P_inside : sorry /* P inside ABC */)
  (h_Q_foot : Q = is_foot_perpendicular P A B) 
  (h_R_foot : R = is_foot_perpendicular P B C)
  (h_S_foot : S = is_foot_perpendicular P C A)
  (h_PQ : (dist P Q) = 2)
  (h_PR : (dist P R) = 3)
  (h_PS : (dist P S) = 4) :
  dist A B = 6 * real.sqrt 3 := 
sorry

end AB_side_length_l340_340771


namespace beth_total_packs_l340_340920

-- Define the initial conditions
def initial_packs : ℝ := 4
def friends_including_beth : ℕ := 10
def packs_found_in_drawer : ℝ := 6

-- Calculate the packs Beth initially keeps
def packs_beth_initially_keeps : ℝ := initial_packs / friends_including_beth

-- Calculate the total packs Beth has
def total_packs_beth_has : ℝ := packs_beth_initially_keeps + packs_found_in_drawer

-- The theorem to be proven
theorem beth_total_packs : total_packs_beth_has = 6.4 :=
by
  sorry

end beth_total_packs_l340_340920


namespace function_derivative_unique_l340_340403

theorem function_derivative_unique (f : ℝ → ℝ) (m a : ℝ) (h : ∀ x, f x = x^m + a * x) :
  (∀ x, deriv f x = 2 * x + 1) → m = 3 ∧ a = 1 :=
by {
  intro h1,
  have h2 : ∀ x, deriv f x = m * x^(m - 1) + a,
  { intro x, simp only [h x], exact deriv_add (deriv_pow x m) (deriv_const_mul x a) },
  specialize h1 1,
  specialize h2 1,
  rw [h1, h2] at *,
  sorry
}

end function_derivative_unique_l340_340403


namespace Angelina_speeds_l340_340915

def distance_home_to_grocery := 960
def distance_grocery_to_gym := 480
def distance_gym_to_library := 720
def time_diff_grocery_to_gym := 40
def time_diff_gym_to_library := 20

noncomputable def initial_speed (v : ℝ) :=
  (distance_home_to_grocery : ℝ) = (v * (960 / v)) ∧
  (distance_grocery_to_gym : ℝ) = (2 * v * (240 / v)) ∧
  (distance_gym_to_library : ℝ) = (3 * v * (720 / v))

theorem Angelina_speeds (v : ℝ) :
  initial_speed v →
  v = 18 ∧ 2 * v = 36 ∧ 3 * v = 54 :=
by
  sorry

end Angelina_speeds_l340_340915


namespace max_value_of_a_l340_340332

theorem max_value_of_a (a : ℝ) : (∀ x : ℝ, x^2 - 2*x - a ≥ 0) → a ≤ -1 := 
by 
  sorry

end max_value_of_a_l340_340332


namespace red_cells_count_l340_340612

theorem red_cells_count (infinite_grid : ℕ → ℕ → bool) (coloring_property : ∀ (i j : ℕ), (infinite_grid i j = true) → ∃ (R : list (ℕ × ℕ)), R.length = 2) :
  ∃ (R : list (ℕ × ℕ)), R.length = 33 :=
by
  sorry

end red_cells_count_l340_340612


namespace multiplier_of_reciprocal_l340_340171

theorem multiplier_of_reciprocal (x m : ℝ) (h1 : x = 7) (h2 : x - 4 = m * (1 / x)) : m = 21 :=
by
  sorry

end multiplier_of_reciprocal_l340_340171


namespace trigonometric_identity_solution_l340_340147

theorem trigonometric_identity_solution (k : ℤ) :
  (∃ x, x = 7 / 12 + 2 * k ∧ sin (π * x) + cos (π * x) = sqrt 2 / 2) ∨
  (∃ x, x = -1 / 4 + 2 * k ∧ sin (π * x) + cos (π * x) = sqrt 2 / 2) :=
sorry

end trigonometric_identity_solution_l340_340147


namespace largest_prime_factor_7799_l340_340486

theorem largest_prime_factor_7799 : ∃ (p : ℕ), prime p ∧ divides p 7799 ∧ ∀ q, prime q ∧ divides q 7799 → q ≤ p :=
sorry

end largest_prime_factor_7799_l340_340486


namespace R_depends_on_d_and_n_l340_340718

-- Define the given properties of the arithmetic progression sums
def s1 (a d n : ℕ) : ℕ := (n * (2 * a + (n - 1) * d)) / 2
def s3 (a d n : ℕ) : ℕ := (3 * n * (2 * a + (3 * n - 1) * d)) / 2
def s5 (a d n : ℕ) : ℕ := (5 * n * (2 * a + (5 * n - 1) * d)) / 2

-- Define R in terms of s1, s3, and s5
def R (a d n : ℕ) : ℕ := s5 a d n - s3 a d n - s1 a d n

-- The main theorem to prove the statement about R's dependency
theorem R_depends_on_d_and_n (a d n : ℕ) : R a d n = 7 * d * n^2 := by 
  sorry

end R_depends_on_d_and_n_l340_340718


namespace range_of_x_l340_340290

variable {ℝ : Type*} [LinearOrderedField ℝ]

-- Conditions -- 
def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f (x)
def derivative (f : ℝ → ℝ) := ∃ (f' : ℝ → ℝ), ∀ x, HasDerivAt f (f' x) x
def condition (f : ℝ → ℝ) := ∀ x ∈ Set.Iic (0 : ℝ), x * f x < f (-x)

-- Definition of F
def F (f : ℝ → ℝ) (x : ℝ) := x * f x

theorem range_of_x (f : ℝ → ℝ) (odd_fn : odd_function f) (deriv_exists : derivative f) (cond : condition f) :
  { x : ℝ | F f 3 > F f (2 * x - 1) } = Set.Ioo (-1 : ℝ) 2 :=
sorry

end range_of_x_l340_340290


namespace cars_left_in_parking_lot_l340_340826

-- Define constants representing the initial number of cars and cars that went out.
def initial_cars : ℕ := 24
def first_out : ℕ := 8
def second_out : ℕ := 6

-- State the theorem to prove the remaining cars in the parking lot.
theorem cars_left_in_parking_lot : 
  initial_cars - first_out - second_out = 10 := 
by {
  -- Here, 'sorry' is used to indicate the proof is omitted.
  sorry
}

end cars_left_in_parking_lot_l340_340826


namespace radius_to_BC_ratio_l340_340695

section
variables {A B C P : Point}
variables {AC AB BC BP PC : ℝ}
variables (x : ℝ)

-- Conditions
def right_triangle (A B C : Point) := ∠C = π / 2
def ratio_AC_AB (AC AB : ℝ) := AC / AB = 4 / 5
def ratio_BP_PC (BP PC : ℝ) := BP / PC = 2 / 3

-- Circle tangent to hypotenuse with certain properties
def circle_center_on_AC_tangent_to_AB (A B C : Point) (O : Point) (circle : circle) :=
  circle.center = O ∧ O ∈ AC ∧ circle.tangent_to (line_AB : Line)

-- Ratio of the radius to the leg \(BC\)
theorem radius_to_BC_ratio
  (A B C P : Point)
  (AC AB BC BP PC : ℝ)
  (C_right : right_triangle A B C)
  (ratio_AB_AC : ratio_AC_AB AC AB)
  (circle_properties : ∃ O r, circle_center_on_AC_tangent_to_AB A B C O (circle.mk O r) 
  ∧ BP = (2 / 5) * BC ∧ PC = (3 / 5) * BC) :
  ∃ r : ℝ, ratio_r_BC : r / BC = 13 / 20 :=
sorry

end

end radius_to_BC_ratio_l340_340695


namespace pirate_coins_l340_340885

theorem pirate_coins :
  ∃ x : ℕ, let coins_at_pirate (n : ℕ) (coins : ℕ) : ℕ :=
    match n with
    | 1       => (14 * coins) / 15
    | (n + 1) => ((15 - n) * coins_at_pirate n coins) / 15
    in coins_at_pirate 14 x = 64512 :=
sorry

end pirate_coins_l340_340885


namespace five_students_in_a_row_five_students_with_constraints_five_students_into_three_classes_l340_340470

-- Definition: Number of ways to arrange n items in a row
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Question (1)
theorem five_students_in_a_row : factorial 5 = 120 :=
by sorry

-- Question (2) - Rather than performing combinatorial steps directly, we'll assume a function to calculate the specific arrangement
def specific_arrangement (students: ℕ) : ℕ :=
  if students = 5 then 24 else 0

theorem five_students_with_constraints : specific_arrangement 5 = 24 :=
by sorry

-- Question (3) - Number of ways to divide n students into k classes with at least one student in each class
def number_of_ways_to_divide (students: ℕ) (classes: ℕ) : ℕ :=
  if students = 5 ∧ classes = 3 then 150 else 0

theorem five_students_into_three_classes : number_of_ways_to_divide 5 3 = 150 :=
by sorry

end five_students_in_a_row_five_students_with_constraints_five_students_into_three_classes_l340_340470


namespace number_of_pages_500_l340_340337

-- Define the conditions as separate constants
def cost_per_page : ℕ := 3 -- cents
def total_cents : ℕ := 1500 

-- Define the number of pages calculation
noncomputable def number_of_pages := total_cents / cost_per_page

-- Statement we want to prove
theorem number_of_pages_500 : number_of_pages = 500 :=
sorry

end number_of_pages_500_l340_340337


namespace Dana_has_25_more_pencils_than_Marcus_l340_340943

theorem Dana_has_25_more_pencils_than_Marcus (JaydenPencils : ℕ) (h1 : JaydenPencils = 20) :
  let DanaPencils := JaydenPencils + 15,
      MarcusPencils := JaydenPencils / 2
  in DanaPencils - MarcusPencils = 25 := 
by
  sorry -- proof to be filled in

end Dana_has_25_more_pencils_than_Marcus_l340_340943


namespace gray_region_area_l340_340556

variable (r_small r_large : ℝ)
variable (d_small : ℝ)
variable (area_gray : ℝ)

def small_circle_radius (d_small : ℝ) : ℝ :=
  d_small / 2

def large_circle_radius (r_small : ℝ) : ℝ :=
  5 * r_small

def circle_area (radius : ℝ) : ℝ :=
  Real.pi * (radius ^ 2)

def gray_area (large_area small_area : ℝ) : ℝ :=
  large_area - small_area

theorem gray_region_area : 
  d_small = 6 → r_small = small_circle_radius d_small → r_large = large_circle_radius r_small →
  area_gray = gray_area (circle_area r_large) (circle_area r_small) →
  area_gray = 216 * Real.pi :=
by
  sorry

end gray_region_area_l340_340556


namespace perpendicular_vectors_m_solution_l340_340597

theorem perpendicular_vectors_m_solution (m : ℝ) 
  (a : ℝ × ℝ := (1, 2)) 
  (b : ℝ × ℝ := (m, 1)) 
  (h_perp : a.1 * b.1 + a.2 * b.2 = 0) : m = -2 := by
  sorry

end perpendicular_vectors_m_solution_l340_340597


namespace salary_recovery_l340_340892

theorem salary_recovery (S : ℝ) : 
  (0.80 * S) + (0.25 * (0.80 * S)) = S :=
by
  sorry

end salary_recovery_l340_340892


namespace balloon_difference_l340_340511

theorem balloon_difference (your_balloons : ℕ) (friend_balloons : ℕ) (h1 : your_balloons = 7) (h2 : friend_balloons = 5) : your_balloons - friend_balloons = 2 :=
by
  sorry

end balloon_difference_l340_340511


namespace game_winning_strategy_l340_340542

theorem game_winning_strategy (piles : List ℕ) :
  let xor_sum := List.foldr (λ x acc => x ⊕ acc) 0 piles in
  (xor_sum = 0 ↔ ∀ k, ¬ ∃ s, s < piles[k] ∧ s = piles[k] ⊕ xor_sum) ∧
  (xor_sum ≠ 0 ↔ ∃ k, ∃ s, s < piles[k] ∧ s = piles[k] ⊕ xor_sum) :=
by
  sorry

end game_winning_strategy_l340_340542


namespace new_total_lines_l340_340416

theorem new_total_lines (L : ℕ) (h : L + 80 = 1.5 * L) : L + 80 = 240 :=
by
  have h1 : L = 160,
  { linarith, },
  rw h1,
  norm_num,
  exact L

end new_total_lines_l340_340416


namespace ratio_books_to_decorations_l340_340204

noncomputable def books := 42
noncomputable def books_per_shelf := 2
noncomputable def decorations_per_shelf := 1
noncomputable def shelves := 3

theorem ratio_books_to_decorations :
  let total_books := books in
  let shelves_needed := total_books / (books_per_shelf * shelves) in
  let total_decorations := shelves_needed * decorations_per_shelf in
  (total_books / total_decorations) = 6 := by
  sorry

end ratio_books_to_decorations_l340_340204


namespace monotonic_increasing_derivative_local_minimum_generally_monotonic_l340_340628

-- Define the function f(x) = ax^2 + cos x with a ∈ ℝ
def f (a : ℝ) (x : ℝ) := a * x^2 + Real.cos x

-- Question 1: Prove f'(x) = x - sin x is monotonically increasing when a = 1/2
theorem monotonic_increasing_derivative (x : ℝ) : 
  let a := 1/2,
  let f_x := f a x in
  let g_x := deriv (f a) x in
  deriv g_x x ≥ 0 :=
sorry

-- Question 2: Prove f(x) attains a local minimum at x = 0 if and only if a ∈ [1/2, +∞)
theorem local_minimum (a : ℝ) : 
  (∃ x : ℝ, x = 0 ∧ ∀ ε > 0, () → f (a) (x - ε) > f a x ∧ f (a) (x + ε) > f a x) ↔ 
  a ∈ { b : ℝ | b ≥ 1/2 } :=
sorry

-- Define the function y = f(x) - x ln x
def y (a : ℝ) (x : ℝ) := f(a) x - x * Real.log x

-- Question 3: Prove y is generally monotonic on (0, +∞)
theorem generally_monotonic (a : ℝ) : 
  let domain := { x : ℝ | 0 < x },
  let y_x := y a in
  (∃ m : ℝ, m ∈ domain ∧ ∀ x > m, monotonic_on y_x (Ioi m)) ∨ 
  (∃ m : ℝ, m ∈ domain ∧ ∀ x > m, monotonic_on y_x (Iio m)) :=
sorry

end monotonic_increasing_derivative_local_minimum_generally_monotonic_l340_340628


namespace AB_side_length_l340_340773

noncomputable def P := (x : ℝ) × (y : ℝ)

def is_foot_perpendicular (P : P) (A B : P) : P := sorry

def equilateral_triangle (A B C : P) : Prop := sorry

theorem AB_side_length (A B C P Q R S : P)
  (h_equilateral : equilateral_triangle A B C)
  (h_P_inside : sorry /* P inside ABC */)
  (h_Q_foot : Q = is_foot_perpendicular P A B) 
  (h_R_foot : R = is_foot_perpendicular P B C)
  (h_S_foot : S = is_foot_perpendicular P C A)
  (h_PQ : (dist P Q) = 2)
  (h_PR : (dist P R) = 3)
  (h_PS : (dist P S) = 4) :
  dist A B = 6 * real.sqrt 3 := 
sorry

end AB_side_length_l340_340773


namespace Hari_contribution_in_capital_l340_340860

theorem Hari_contribution_in_capital (P H : ℝ) (join_months : ℝ) (profit_ratio_H P_ratio_H : ℝ):
  P = 3220 →
  join_months = 5 →
  profit_ratio_H / P_ratio_H = 3 / 2 →
  H = (P * (12 / 7) * (3 / 2)) →
  H = 8280 :=
by
  intros hP hjoin hprofit hratio
  rw [hP, hjoin]
  sorry

end Hari_contribution_in_capital_l340_340860


namespace range_of_values_for_sqrt_l340_340092

theorem range_of_values_for_sqrt (x : ℝ) : (x + 3 ≥ 0) ↔ (x ≥ -3) :=
by
  sorry

end range_of_values_for_sqrt_l340_340092


namespace find_valid_six_digit_numbers_l340_340576

def is_six_digit_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000

def move_last_digit_to_front (n : ℕ) : ℕ :=
  let a := n / 10
  let b := n % 10
  b * 10^5 + a

def is_integer_multiple (x y : ℕ) : Prop :=
  ∃ k : ℕ, k * x = y

theorem find_valid_six_digit_numbers (N : ℕ) :
  is_six_digit_number N →
  is_integer_multiple N (move_last_digit_to_front N) →
  N ∈ {142857, 102564, 128205, 153846, 179487, 205128, 230769} :=
by
  sorry

end find_valid_six_digit_numbers_l340_340576


namespace coplanar_points_l340_340578

theorem coplanar_points (a : ℂ) : 
  (Determinant.mat33 
    ![
      ![1, 0, a^2],
      ![a^2, 1, 0],
      ![0, a^2, a]
    ]) = 0 ↔ a = 0 ∨ a = 1 ∨ a = -1 ∨ a = Complex.I ∨ a = -Complex.I :=
by
  sorry

end coplanar_points_l340_340578


namespace no_unique_sum_grid_l340_340211

noncomputable def impossible_fill_grid (n : ℕ) : Prop :=
  ¬ ∃ (grid : Fin n × Fin n → ℕ),
    (∀ i : Fin n, (∀ j, grid (i, j) ∈ {1, 2, 3})) ∧
    (Finset.range n).disjoint (λ i => (Finset.univ.sum (λ j => grid (i, j)))) ∧
    (Finset.range n).disjoint (λ j => (Finset.univ.sum (λ i => grid (i, j)))) ∧
    (Finset.univ.sum (λ i => grid (i, i)) ≠ Finset.univ.sum (λ i => grid (i, n - 1 - i)))

theorem no_unique_sum_grid (n : ℕ) : impossible_fill_grid n :=
  sorry

end no_unique_sum_grid_l340_340211


namespace not_in_nat_set_l340_340464

def N : Set ℕ := { n | true } -- Define the set of natural numbers

theorem not_in_nat_set : -3 ∈ N = false := 
by 
  sorry -- Placeholder for the proof

end not_in_nat_set_l340_340464


namespace total_spending_march_to_july_l340_340452

-- Define the conditions
def beginning_of_march_spending : ℝ := 1.2
def end_of_july_spending : ℝ := 4.8

-- State the theorem to prove
theorem total_spending_march_to_july : 
  end_of_july_spending - beginning_of_march_spending = 3.6 :=
sorry

end total_spending_march_to_july_l340_340452


namespace monotonic_increasing_interval_l340_340459

def quadratic (x : ℝ) := x^2 - 4 * x + 3

def domain := {x : ℝ | quadratic x > 0}

def log_function_part (x : ℝ) := log (quadratic x) (1/3)

theorem monotonic_increasing_interval :
  - ∞ < x ∧ x < 1 → log_function_part x = log (quadratic x) (1/3) :=
sorry

end monotonic_increasing_interval_l340_340459


namespace acute_angle_between_planes_l340_340241

theorem acute_angle_between_planes :
  let n1 := (2, -1, -3 : ℝ × ℝ × ℝ)
  let n2 := (1, 1, 0 : ℝ × ℝ × ℝ)
  let dot_product (a b : ℝ × ℝ × ℝ) := a.1 * b.1 + a.2 * b.2 + a.3 * b.3
  let magnitude (v : ℝ × ℝ × ℝ) := Real.sqrt (v.1^2 + v.2^2 + v.3^2)
  let cos_angle := dot_product n1 n2 / (magnitude n1 * magnitude n2)
  let alpha := Real.arccos (Real.abs cos_angle)
  
  n1 = (2, -1, -3 : ℝ × ℝ × ℝ) ∧ n2 = (1, 1, 0 : ℝ × ℝ × ℝ) →
  alpha = Real.arccos (1 / (2 * Real.sqrt 7)) := 
by {
  intros,
  sorry
}

end acute_angle_between_planes_l340_340241


namespace suitable_for_census_l340_340498

-- Define types for each survey option.
inductive SurveyOption where
  | A : SurveyOption -- Understanding the vision of middle school students in our province
  | B : SurveyOption -- Investigating the viewership of "The Reader"
  | C : SurveyOption -- Inspecting the components of a newly developed fighter jet to ensure successful test flights
  | D : SurveyOption -- Testing the lifespan of a batch of light bulbs

-- Theorem statement asserting that Option C is the suitable one for a census.
theorem suitable_for_census : SurveyOption.C = SurveyOption.C :=
by
  exact rfl

end suitable_for_census_l340_340498


namespace option_b_option_c_l340_340395

variable {z1 z2 : Complex}

theorem option_b (h : |z1| = |z2|) : z1 * Complex.conj z1 = z2 * Complex.conj z2 := 
  by sorry

theorem option_c (h : |z1 / z2| > 1) : |z1| > |z2| := 
  by sorry

end option_b_option_c_l340_340395


namespace range_of_m_l340_340273

-- Define the function f and the inequality condition
def f (x m : ℝ) := x^2 - 2 * m * x + 4

def inequality (m x : ℝ) := m * x^2 + 4 * (m - 2) * x + 4 > 0

-- Conditions for p
def condition_p (m : ℝ) := ∀ x : ℝ, x ≥ 2 → (∀ y : ℝ, (y ≥ x) → f y m ≥ f x m)

-- Conditions for q
def condition_q (m : ℝ) := ∀ x : ℝ, inequality m x

-- Combined conditions
def condition_p_or_q (m : ℝ) := condition_p m ∨ condition_q m
def condition_p_and_not_q (m : ℝ) := condition_p m ∧ ¬ condition_q m

-- The main theorem statement
theorem range_of_m (m : ℝ) : condition_p_or_q m ∧ ¬ condition_p_and_not_q m → m ∈ set.Iic 1 ∪ set.Ioo 2 4 :=
by
  sorry

end range_of_m_l340_340273


namespace number_of_correct_propositions_l340_340626

variable (Ω : Type) (R : Type) [Nonempty Ω] [Nonempty R]

-- Definitions of the conditions
def carsPassingIntersection (t : ℝ) : Ω → ℕ := sorry
def passengersInWaitingRoom (t : ℝ) : Ω → ℕ := sorry
def maximumFlowRiverEachYear : Ω → ℝ := sorry
def peopleExitingTheater (t : ℝ) : Ω → ℕ := sorry

-- Statement to prove the number of correct propositions
theorem number_of_correct_propositions : 4 = 4 := sorry

end number_of_correct_propositions_l340_340626


namespace average_marks_l340_340224

-- Conditions
def marks_english : ℕ := 73
def marks_mathematics : ℕ := 69
def marks_physics : ℕ := 92
def marks_chemistry : ℕ := 64
def marks_biology : ℕ := 82
def number_of_subjects : ℕ := 5

-- Problem Statement
theorem average_marks :
  (marks_english + marks_mathematics + marks_physics + marks_chemistry + marks_biology) / number_of_subjects = 76 :=
by
  sorry

end average_marks_l340_340224


namespace maria_work_end_time_l340_340035

-- Define the conditions
def maria_start_time : ℕ := 7 * 60 + 25  -- Start time in minutes after midnight
def maria_lunch_start : ℕ := 12 * 60  -- Noon in minutes after midnight
def working_hours_excluding_lunch : ℕ := 9 * 60 -- 9 hours in minutes
def lunch_duration : ℕ := 60 -- 1 hour in minutes

-- Define the main theorem to prove
theorem maria_work_end_time : 
  let time_worked_before_lunch := maria_lunch_start - maria_start_time,
      remaining_work_time := working_hours_excluding_lunch - time_worked_before_lunch,
      resume_work_time := maria_lunch_start + lunch_duration,
      end_work_time := resume_work_time + remaining_work_time
  in end_work_time = 17 * 60 + 25 := by 
  sorry

end maria_work_end_time_l340_340035


namespace sum_of_first_nine_terms_l340_340387

noncomputable def geometric_sum (a : ℕ → ℝ) : ℕ → ℝ
| 0       := 0
| (n + 1) := a n + geometric_sum (n)

variable {a : ℕ → ℝ}
variable {r : ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n, a (n + 1) = r * a n

theorem sum_of_first_nine_terms
  (h1 : is_geometric_sequence a r)
  (S3 : geometric_sum a 3 = 10)
  (S6 : geometric_sum a 6 = 30) :
  geometric_sum a 9 = 70 :=
sorry

end sum_of_first_nine_terms_l340_340387


namespace distinct_collections_COMPUTATIONS_l340_340360

theorem distinct_collections_COMPUTATIONS : 
  let vowels := {O, U, A, I}
  let consonants := {C, M, P, T, T, S, N}
  let binom := Nat.choose
  (binom 4 3 * 
    (binom 6 4 + binom 6 3 * binom 2 1 + binom 6 2 * binom 2 2)) = 200 := 
by 
  unfold vowels consonants binom 
  sorry

end distinct_collections_COMPUTATIONS_l340_340360


namespace asymptote_equation_l340_340776

noncomputable def hyperbola_asymptotes (a b : ℝ) (P F1 F2 : ℝ × ℝ) (k : ℝ) : Prop :=
  ∃ (a b : ℝ), 
    a > 0 ∧ b > 0 ∧
    P ∈ {p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1} ∧ 
    ∠ F1 P F2 = 90 ∧ 
    |F1 P| = 4k ∧ 
    |F2 P| = 3k ∧ 
    |F1 F2| = 5k ∧
    y = 2 * sqrt(6) * x

theorem asymptote_equation :
  ∀ (a b : ℝ) (P F1 F2 : ℝ × ℝ) (k : ℝ),
    hyperbola_asymptotes a b P F1 F2 k → 
    y = 2 * sqrt(6) * x :=
begin
  intros,
  sorry,
end

end asymptote_equation_l340_340776


namespace minimum_k_value_l340_340721

theorem minimum_k_value :
  ∃ k ∈ ℕ, (∀ A ⊆ {i | i ∈ Finset.range 101}, 
  Finset.card A = k → 
  ∃ a b ∈ A, a ≠ b ∧ |a - b| ≤ 4) ∧ 
  (∀ n ∈ ℕ, n < k → (∃ A ⊆ {i | i ∈ Finset.range 101}, 
  Finset.card A = n ∧ (∀ a b ∈ A, a ≠ b → |a - b| > 4))) := by
  let S := Finset.range 101
  sorry

end minimum_k_value_l340_340721


namespace number_of_planes_l340_340482

theorem number_of_planes (total_wings: ℕ) (wings_per_plane: ℕ) 
  (h1: total_wings = 50) (h2: wings_per_plane = 2) : 
  total_wings / wings_per_plane = 25 := by 
  sorry

end number_of_planes_l340_340482


namespace factor_expression_l340_340963

variable (b : ℝ)

theorem factor_expression : 221 * b * b + 17 * b = 17 * b * (13 * b + 1) :=
by
  sorry

end factor_expression_l340_340963


namespace find_sin_theta_l340_340711

open real

variables {a b c : ℝ^3}
variables (theta : ℝ)

-- Conditions
axiom nonzero_a : a ≠ 0
axiom nonzero_b : b ≠ 0
axiom nonzero_c : c ≠ 0
axiom non_parallel_ab : ¬ collinear a b
axiom non_parallel_bc : ¬ collinear b c
axiom non_parallel_ca : ¬ collinear c a
axiom cross_product_condition : ((a × b) × c) = (1/3) * ∥b∥ * ∥c∥ • a

-- Question: Prove that sin θ = 2sqrt(2)/3
theorem find_sin_theta : sin theta = (2 * sqrt(2)) / 3 :=
  sorry

end find_sin_theta_l340_340711


namespace curve1_line_and_circle_curve2_two_points_l340_340447

-- Define the first condition: x(x^2 + y^2 - 4) = 0
def curve1 (x y : ℝ) : Prop := x * (x^2 + y^2 - 4) = 0

-- Define the second condition: x^2 + (x^2 + y^2 - 4)^2 = 0
def curve2 (x y : ℝ) : Prop := x^2 + (x^2 + y^2 - 4)^2 = 0

-- The corresponding theorem statements
theorem curve1_line_and_circle : ∀ x y : ℝ, curve1 x y ↔ (x = 0 ∨ (x^2 + y^2 = 4)) := 
sorry 

theorem curve2_two_points : ∀ x y : ℝ, curve2 x y ↔ (x = 0 ∧ (y = 2 ∨ y = -2)) := 
sorry 

end curve1_line_and_circle_curve2_two_points_l340_340447


namespace sequence_sum_l340_340790

theorem sequence_sum (n : ℕ) (x : ℕ → ℕ) 
  (h₀ : x 1 = 2) 
  (h₁ : ∀ k, 1 ≤ k → k < n → x (k + 1) = x k + k) : 
  ∑ i in finset.range n.succ, x i = 2 * n + (n^3 - n) / 6 := 
sorry

end sequence_sum_l340_340790


namespace sum_of_squares_of_diagonals_l340_340587

variable (OP R : ℝ)

theorem sum_of_squares_of_diagonals (AC BD : ℝ) :
  AC^2 + BD^2 = 8 * R^2 - 4 * OP^2 :=
sorry

end sum_of_squares_of_diagonals_l340_340587


namespace complement_union_eq_l340_340033

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def S : Set ℕ := {1, 3, 5}
def T : Set ℕ := {3, 6}

theorem complement_union_eq : (U \ (S ∪ T)) = {2, 4, 7, 8} :=
by {
  sorry
}

end complement_union_eq_l340_340033


namespace a_share_514_29_l340_340857
noncomputable theory

-- Definitions based on conditions
def a_plus_b_plus_c (A B C : ℝ) : Prop := A + B + C = 1800

def a_as_fraction_of_b_c (A B C : ℝ) : Prop := A = (2 / 5) * (B + C)

def b_as_fraction_of_a_c (B A C : ℝ) : Prop := B = (1 / 5) * (A + C)

-- Statement of the problem
theorem a_share_514_29 
  (A B C : ℝ) 
  (h1 : a_plus_b_plus_c A B C) 
  (h2 : a_as_fraction_of_b_c A B C)
  (h3 : b_as_fraction_of_a_c B A C) : 
  A = 514.29 := 
sorry

end a_share_514_29_l340_340857


namespace factorization_l340_340574

theorem factorization (m n : ℝ) : 
  m^2 - n^2 + 2 * m - 2 * n = (m - n) * (m + n + 2) :=
by
  sorry

end factorization_l340_340574


namespace find_total_amount_l340_340152

noncomputable def total_amount (A : ℝ) (annual_income : ℝ) : ℝ :=
  let T := 2083.3333333333335
  in T

theorem find_total_amount :
  ∀ (T A : ℝ),
    A = 500.0000000000002 →
    0.05 * A + 0.06 * (T - A) = 145 →
    T = 2083.3333333333335 :=
by
  intros T A hA ha
  sorry

end find_total_amount_l340_340152


namespace collinear_LEF_l340_340047

open EuclideanGeometry

variables {A B C D E O₁ O₂ F L : Point}
variables {triangleABC : Triangle A B C}
variables {heightBD : Line}
variables {circumcenterAEB : Circumcenter O₁ (Triangle A E B)}
variables {circumcenterCEB : Circumcenter O₂ (Triangle C E B)}
variables {midpointAC : Midpoint F A C}
variables {midpointO₁O₂ : Midpoint L O₁ O₂}
variables {pointE_onBD : E ∈ heightBD}
variables {angleAEC_90 : angle A E C = 90}

theorem collinear_LEF (hBD : height heightBD triangleABC)
    (hE_on_BD : pointE_onBD)
    (h_angle_AEC_90 : angleAEC_90)
    (h_circumcenter_AEB : circumcenterAEB)
    (h_circumcenter_CEB : circumcenterCEB)
    (h_midpoint_AC : midpointAC)
    (h_midpoint_O₁O₂ : midpointO₁O₂) :
    collinear L E F :=
by
  sorry

end collinear_LEF_l340_340047


namespace factor_correct_l340_340969

noncomputable def p (b : ℝ) : ℝ := 221 * b^2 + 17 * b
def factored_form (b : ℝ) : ℝ := 17 * b * (13 * b + 1)

theorem factor_correct (b : ℝ) : p b = factored_form b := by
  sorry

end factor_correct_l340_340969


namespace period_2pi_tan_half_period_2pi_cos_abs_not_period_2pi_sin_half_not_period_2pi_sin_abs_l340_340131

def period (f : ℝ → ℝ) (T : ℝ) : Prop :=
∀ x, f (x + T) = f x

def tan_half (x : ℝ) : ℝ := tan (x / 2)
def sin_half (x : ℝ) : ℝ := sin (x / 2)
def sin_abs (x : ℝ) : ℝ := sin (abs x)
def cos_abs (x : ℝ) : ℝ := cos (abs x)

theorem period_2pi_tan_half : period tan_half (2 * Real.pi) := sorry
theorem period_2pi_cos_abs : period cos_abs (2 * Real.pi) := sorry
theorem not_period_2pi_sin_half : ¬ period sin_half (2 * Real.pi) := sorry
theorem not_period_2pi_sin_abs : ¬ period sin_abs (2 * Real.pi) := sorry

end period_2pi_tan_half_period_2pi_cos_abs_not_period_2pi_sin_half_not_period_2pi_sin_abs_l340_340131


namespace largest_increase_is_2007_2008_l340_340919

-- Define the number of students each year
def students_2005 : ℕ := 50
def students_2006 : ℕ := 55
def students_2007 : ℕ := 60
def students_2008 : ℕ := 70
def students_2009 : ℕ := 72
def students_2010 : ℕ := 80

-- Define the percentage increase function
def percentage_increase (old new : ℕ) : ℚ :=
  ((new - old) : ℚ) / old * 100

-- Define percentage increases for each pair of consecutive years
def increase_2005_2006 := percentage_increase students_2005 students_2006
def increase_2006_2007 := percentage_increase students_2006 students_2007
def increase_2007_2008 := percentage_increase students_2007 students_2008
def increase_2008_2009 := percentage_increase students_2008 students_2009
def increase_2009_2010 := percentage_increase students_2009 students_2010

-- State the theorem
theorem largest_increase_is_2007_2008 :
  (max (max increase_2005_2006 (max increase_2006_2007 increase_2008_2009))
       increase_2009_2010) < increase_2007_2008 := 
by
  -- Add proof steps if necessary.
  sorry

end largest_increase_is_2007_2008_l340_340919


namespace third_candidate_votes_l340_340473

theorem third_candidate_votes (V A B W: ℕ) (hA : A = 2500) (hB : B = 15000) 
  (hW : W = (2 * V) / 3) (hV : V = W + A + B) : (V - (A + B)) = 35000 := by
  sorry

end third_candidate_votes_l340_340473


namespace find_f_inequality_on_interval_f_equals_x_l340_340716

section problem1
  variables {a : ℝ} (f g : ℝ → ℝ)
  -- Define conditions
  def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
  def symmetric_about_x_1 (f g : ℝ → ℝ) : Prop := ∀ x, f x = g (2 - x)
  def g_def : ℝ → ℝ := λ x, a * (x - 2) - (x - 2)^3

  hypothesis odd_f : odd_function f
  hypothesis sym_fg : symmetric_about_x_1 f g
  hypothesis g_eqn : ∀ x, g x = a * (x - 2) - (x - 2)^3
  hypothesis f_extreme_at_1 : ∀ x, x = 1 → f x ∂ = 0
  hypothesis f_monotonic_on_ge_1 : monotonic_on f (λ x : ℝ, x ≥ 1)
  hypothesis f_ge_1_on_ge_1 : ∀ x, x ≥ 1 → f x ≥ 1
  hypothesis f_self_inverse_on_ge_1 : ∀ x, x ≥ 1 → f (f x) = x
  
  -- Assertions
  theorem find_f : f = λ x, -a * x + x^3 :=
  sorry

  theorem inequality_on_interval : ∀ x1 x2, (-1 < x1) ∧ (x1 < 1) ∧ (-1 < x2) ∧ (x2 < 1) → |f x1 - f x2| < 4 :=
  sorry

  theorem f_equals_x : (∀ x, x ≥ 1 → f x ≥ 1 ∧ f (f x) = x) → (∀ x, f x = x) :=
  sorry
end problem1

end find_f_inequality_on_interval_f_equals_x_l340_340716


namespace symmetry_distance_l340_340689

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.sin (2 * x + Real.pi / 6)

theorem symmetry_distance :
  ∀ x : ℝ, ∃ d : ℝ, d = ℝ.pi / 2 ∧ ∃ k : ℤ, x = k * d :=
begin
  -- The proof would go here.
  sorry
end

end symmetry_distance_l340_340689


namespace number_of_elements_in_intersection_l340_340320

def M : Set ℕ := { x | x ≤ 3 }
def N : Set ℕ := { 0, 1, 2, 3, 4, 5, ... } -- natural numbers are included as infinite set in Mathlib

theorem number_of_elements_in_intersection : (M ∩ N).card = 4 := by
  sorry

end number_of_elements_in_intersection_l340_340320


namespace smallest_N_mod_25_l340_340707

theorem smallest_N_mod_25 : 
  let N := Nat.find (λ N, (∀ m, (2008 * N = m * m) ↔ ∃ k, m = k * k) ∧ (∀ n, (2007 * N = n * n * n) ↔ ∃ l, n = l * l * l))
  in N % 25 = 17 :=
by
  let N := Nat.find (λ N, (∀ m, (2008 * N = m * m) ↔ ∃ k, m = k * k) ∧ (∀ n, (2007 * N = n * n * n) ↔ ∃ l, n = l * l * l))
  sorry

end smallest_N_mod_25_l340_340707


namespace triangle_inequality_l340_340288

theorem triangle_inequality (a b c Δ : ℝ) (h_Δ: Δ = Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))) :
    a^2 + b^2 + c^2 ≥ 4 * Real.sqrt (3) * Δ + (a - b)^2 + (b - c)^2 + (c - a)^2 :=
by
  sorry

end triangle_inequality_l340_340288


namespace initial_amount_correct_l340_340179

-- Define the initial conditions
def total_amount_received : ℝ := 500
def interest_rate : ℝ := 0.04
def time_period : ℝ := 5 / 3

-- Define the initial amount lent out 'P'
noncomputable def initial_amount_lent_out : ℝ :=
  (total_amount_received / (1 + (interest_rate * time_period)))

-- Verify the initial amount lent out is approximately 7031.72
theorem initial_amount_correct : initial_amount_lent_out ≈ 7031.72 :=
by
  -- Replace with equivalent Lean tactics
  have h : initial_amount_lent_out = 7031.71859, from by sorry -- calculation should be handled by assuming correct math done 
  show initial_amount_lent_out ≈ 7031.72, from calc_proof sorry

end initial_amount_correct_l340_340179


namespace quadratic_no_real_roots_l340_340085

theorem quadratic_no_real_roots
  {a b c : ℝ}
  (h : ∀ x, (a * x ^ 2 + b * x + c = 0) → (2 * a * x + b = 0))
  (divides_plane_into_four_parts : ∀ f, ∃ x1 x2, (f x1 = f x2) ∧ (x1 ≠ x2)) :
  ¬ ∃ x : ℝ, a * x ^ 2 + b * x + c = 0 := 
begin
  sorry
end

end quadratic_no_real_roots_l340_340085


namespace bananas_removed_l340_340949

theorem bananas_removed (initial: ℕ) (remaining: ℕ) (h1: initial = 46) (h2: remaining = 41) :
  initial - remaining = 5 :=
by 
  rw [h1, h2]
  rfl

end bananas_removed_l340_340949


namespace euler_relation_l340_340382

-- Definitions of the circumcenter and incenter
variables {Δ : Type*} [triangle Δ]
variables (O I : Δ → point)
variables (R r : ℝ)

-- Definition of conditions: 
-- O is the circumcenter with circumradius R and I is the incenter with inradius r
def is_circumcenter (ABC : Δ) : Prop :=
  ∃ (O : Δ → point), circumradius ABC = R

def is_incenter (ABC : Δ) : Prop :=
  ∃ (I : Δ → point), inradius ABC = r

-- Theorem statement
theorem euler_relation (ABC : Δ) (h_circ : is_circumcenter ABC) (h_in : is_incenter ABC) :
  dist O I ^ 2 = R * (R - 2 * r) ∧ R ≥ 2 * r := 
sorry

end euler_relation_l340_340382


namespace g_x_plus_three_l340_340024

variable (x : ℝ)

def g (x : ℝ) : ℝ := x^2 - x

theorem g_x_plus_three : g (x + 3) = x^2 + 5 * x + 6 := by
  sorry

end g_x_plus_three_l340_340024


namespace joan_gave_mike_seashells_l340_340700

noncomputable def original_seashells : ℕ := 79
noncomputable def current_seashells : ℕ := 16

theorem joan_gave_mike_seashells (h : original_seashells = 79) (h1 : current_seashells = 16) :
        original_seashells - current_seashells = 63 :=
begin
  sorry
end

end joan_gave_mike_seashells_l340_340700


namespace triangle_geometry_bn_lt_im_l340_340706
-- Import the required library

-- Define the theorem
theorem triangle_geometry_bn_lt_im
  (ABC : Triangle) -- Let ABC be a triangle
  (A B C H I M N : Point) -- Points involved, including incenter and midpoint
  (HA : A ∈ ABC)
  (HB : B ∈ ABC)
  (HC : C ∈ ABC)
  (H_AB_BC : dist A B < dist B C) -- AB < BC
  (H_H_on_AC : H ∈ line_span A C) -- H is on AC
  (H_BH_perpendicular_AC : perpendicular (line_span B H) (line_span A C)) -- BH is height from B to AC
  (H_I_incenter : is_incenter I ABC) -- I is the incenter of triangle ABC
  (H_M_midpoint_AC : is_midpoint M A C) -- M is the midpoint of AC
  (H_M_I_intersects_BH_at_N : intersects (line_span M I) (line_span B H) N) -- MI intersects BH at N
  : dist B N < dist I M := -- Prove that BN < IM
sorry

end triangle_geometry_bn_lt_im_l340_340706


namespace football_area_regions_II_III_l340_340430

-- Definitions based on conditions from step a)
def PQRS_square (P Q R S : Type) [AddGroup P] [HasNorm P] [HasInner P] : Prop :=
  ∃ (a b c d : P), PQ = a ∧ QR = b ∧ RS = c ∧ SP = d ∧ 
  norm (P - Q) = 3 ∧ norm (Q - R) = 3 ∧ norm (R - S) = 3 ∧ norm (S - P) = 3 ∧
  inner (Q - P) (S - P) = 0 ∧ inner (R - Q) (P - Q) = 0 ∧
  inner (S - R) (Q - R) = 0 ∧ inner (P - S) (R - S) = 0

-- Main theorem based on step c)
theorem football_area_regions_II_III
  (P Q R S : Type) [AddGroup P] [HasNorm P] [HasInner P]
  (h1 : PQRS_square P Q R S)
  (h2 : ∀(P T U : Type), Arc_from_CenterToP P T U (center := Q))
  (h3 : ∀(P Z U : Type), Arc_from_CenterToP P Z U (center := R)) :
  total_area_regions_II_III PQRS = 5.1 :=
sorry

end football_area_regions_II_III_l340_340430


namespace digit_sum_equiv_l340_340089

-- Definitions assuming given conditions
def is_permutation (A B : ℕ) : Prop :=
  ∃ (a b : list ℕ), (A = a.join_digit ∧ B = b.join_digit ∧ a.perm b)

-- Defining the sum of the digits of a number
def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

-- Translation to Lean statement
theorem digit_sum_equiv (A B : ℕ) (h : is_permutation A B) : 
  digit_sum (5 * A) = digit_sum (5 * B) :=
by
  sorry

end digit_sum_equiv_l340_340089


namespace AB_side_length_l340_340774

noncomputable def P := (x : ℝ) × (y : ℝ)

def is_foot_perpendicular (P : P) (A B : P) : P := sorry

def equilateral_triangle (A B C : P) : Prop := sorry

theorem AB_side_length (A B C P Q R S : P)
  (h_equilateral : equilateral_triangle A B C)
  (h_P_inside : sorry /* P inside ABC */)
  (h_Q_foot : Q = is_foot_perpendicular P A B) 
  (h_R_foot : R = is_foot_perpendicular P B C)
  (h_S_foot : S = is_foot_perpendicular P C A)
  (h_PQ : (dist P Q) = 2)
  (h_PR : (dist P R) = 3)
  (h_PS : (dist P S) = 4) :
  dist A B = 6 * real.sqrt 3 := 
sorry

end AB_side_length_l340_340774


namespace binomial_sum_of_coefficients_l340_340998

theorem binomial_sum_of_coefficients :
  let a := (∫ x in 0..2, (1 - 3 * x^2)) + 4 in
  (∃ n : ℕ, ∀ k : ℕ, (a ≠ 0 ∧ 15 = Nat.choose n 2) ∧
    (∑ i in Finset.range (n + 1), (1 : ℚ)) = (1 - 1/2)^n → (1 - 1/2) ^ n = 1/64) :=
by
  let a := (∫ x in 0..2, (1 - 3 * x^2)) + 4
  have integral_calculation : ∫ x in 0..2, (1 - 3 * x^2) = -6 := sorry
  have a_value : a = -2 := sorry
  have coeff_condition : ∃ n : ℕ, 15 = Nat.choose n 2 := by
    use 6
  have n_value : 15 = Nat.choose 6 2 := sorry
  have sum_of_coeffs : (1 - 1/2)^6 = 1/64 := sorry
  exact ⟨6, fun _ ⟨h1, h2⟩ => by
    exact sum_of_coeffs
  ⟩
  sorry

end binomial_sum_of_coefficients_l340_340998


namespace weights_less_than_90_l340_340167

variable (a b c : ℝ)
-- conditions
axiom h1 : a + b = 100
axiom h2 : a + c = 101
axiom h3 : b + c = 102

theorem weights_less_than_90 (a b c : ℝ) (h1 : a + b = 100) (h2 : a + c = 101) (h3 : b + c = 102) : a < 90 ∧ b < 90 ∧ c < 90 := 
by sorry

end weights_less_than_90_l340_340167


namespace scientific_notation_of_78200000000_l340_340791

theorem scientific_notation_of_78200000000 :
  (78_200_000_000 : ℝ) = 7.82 * (10 : ℝ) ^ 10 :=
by sorry

end scientific_notation_of_78200000000_l340_340791


namespace num_ways_to_select_3_colors_from_9_l340_340267

def num_ways_select_colors (n k : ℕ) : ℕ := Nat.choose n k

theorem num_ways_to_select_3_colors_from_9 : num_ways_select_colors 9 3 = 84 := by
  sorry

end num_ways_to_select_3_colors_from_9_l340_340267


namespace equilateral_triangle_side_length_l340_340753

open Classical

noncomputable section

variable {P Q R S : Type}

def is_perpendicular_feet (P Q R S : P) : Prop :=
  sorry -- Definition for Q, R, S being the feet of perpendiculars from P

structure EquilateralTriangle (A B C P Q R S : P) where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  AB : ℝ
  (h_perpendicular : is_perpendicular_feet P Q R S)
  (h_area_eq : ∀ (h₁ : PQ = 2) (h₂ : PR = 3) (h₃ : PS = 4), AB = 12 * Real.sqrt 3)

theorem equilateral_triangle_side_length {A B C P Q R S : P} 
    (h_eq_triangle : EquilateralTriangle A B C P Q R S) : h_eq_triangle.AB = 12 * Real.sqrt 3 :=
  by
    cases h_eq_triangle with
    | mk PQ PR PS AB h_perpendicular h_area_eq =>
        apply h_area_eq
        · exact rfl
        · exact rfl
        · exact rfl

end equilateral_triangle_side_length_l340_340753


namespace exists_n_sum_gt_10_exists_n_sum_gt_1000_l340_340006

noncomputable def harmonic_sum (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ i, 1 / (i + 1 : ℝ))

theorem exists_n_sum_gt_10 : ∃ n : ℕ, harmonic_sum n > 10 :=
sorry

theorem exists_n_sum_gt_1000 : ∃ n : ℕ, harmonic_sum n > 1000 :=
sorry

end exists_n_sum_gt_10_exists_n_sum_gt_1000_l340_340006


namespace part1_part2_l340_340629

open Real

/-- (1) Prove monotonic intervals of f(x) = (2ae^x - x)e^x for a = 0. -/
theorem part1 (x : ℝ) : 
  (let f := λ x : ℝ, (0 - x) * exp x in
  (∀ x < -1, (deriv f) x > 0) ∧ (∀ x > -1, (deriv f) x < 0)) := sorry

/-- (2) Prove the minimum value of a such that f(x) + 1/a ≤ 0 always holds for all x ∈ ℝ is -e^3/2. -/
theorem part2 (x : ℝ) : 
  (let f := λ x : ℝ, (2 * a * exp x - x) * exp x in
  (∃ a : ℝ, a = -exp 3 / 2 ∧ ∀ x, f x + 1 / a ≤ 0)) := sorry

end part1_part2_l340_340629


namespace side_length_eq_l340_340748

namespace EquilateralTriangle

variables (A B C P Q R S : Type) [HasVSub Type P] [MetricSpace P]
variables [HasDist P] [HasEquilateralTriangle ABC] [InsideTriangle P ABC]
variables [Perpendicular PQ AB] [Perpendicular PR BC] [Perpendicular PS CA]
variables [Distance PQ 2] [Distance PR 3] [Distance PS 4]

theorem side_length_eq : side_length ABC = 6 * √3 :=
sorry
end EquilateralTriangle

end side_length_eq_l340_340748


namespace geometric_sequence_property_l340_340797

-- Define the sequence and the conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the main property we are considering
def given_property (a: ℕ → ℝ) (n : ℕ) : Prop :=
  a (n + 1) * a (n - 1) = (a n) ^ 2

-- State the theorem
theorem geometric_sequence_property {a : ℕ → ℝ} (n : ℕ) (hn : n ≥ 2) :
  (is_geometric_sequence a → given_property a n ∧ ∀ a, given_property a n → ¬ is_geometric_sequence a) := sorry

end geometric_sequence_property_l340_340797


namespace blocks_added_l340_340102

theorem blocks_added (a b : Nat) (h₁ : a = 86) (h₂ : b = 95) : b - a = 9 :=
by
  sorry

end blocks_added_l340_340102


namespace function_properties_l340_340537

def f (x : ℝ) : ℝ := real.sqrt (x^2 + 4) + real.sqrt (x^2 - 8 * x + 20)

theorem function_properties :
  (∀ x, f (2 - x) = f (2 + x)) ∧
  (∀ x, f (-x) ≠ f x ∧ f (-x) ≠ -f x) ∧
  (¬ ∃ x, f (f x) = 2 + 2 * real.sqrt (5)) ∧
  (∀ x, f x ≥ 4 * real.sqrt (2)) :=
by
  sorry

end function_properties_l340_340537


namespace count_simple_numbers_lt_one_million_l340_340170

-- Definition of a simple number
def is_simple_number (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d = 1 ∨ d = 2

-- The main theorem stating the problem
theorem count_simple_numbers_lt_one_million : 
  (finset.filter (λ n, is_simple_number n) (finset.range 1000000)).card = 126 :=
sorry

end count_simple_numbers_lt_one_million_l340_340170


namespace find_X_l340_340095

theorem find_X :
  ∃ X : ℝ, 
    let top_row := [30, 30 - 6, 30 - 2 * 6, 30 - 3 * 6, 30 - 4 * 6, 30 - 5 * 6, 30 - 6 * 6],
        first_column := [20, 24, 20 - 4, 20 - 2 * 4],
        second_column := [X, X - (30 - 30 + (-2.5)), X - 2 * (30 - 30 + (-2.5)), X - 3 * (30 - 30 + (-2.5)), -10]
  in
  top_row !! 3 = some 12 ∧
  first_column !! 0 = some 20 ∧
  first_column !! 1 = some 24 ∧
  second_column !! 4 = some (-10) ∧
  X = 2.5 :=
sorry

end find_X_l340_340095


namespace proper_subsets_A_range_of_m_l340_340729

-- Problem 1
def A (x : ℤ) : Prop := -2 ≤ x ∧ x ≤ 5

theorem proper_subsets_A : 
  (∃ s : finset ℤ, (∀ x, A x → x ∈ s.to_set) ∧ s.card = 8) → 
  2^8 - 1 = 253 :=
by 
  sorry

-- Problem 2
def B (m : ℝ) (x : ℝ) : Prop := m - 1 ≤ x ∧ x < 2 * m + 1
def set_A : set ℝ := {x | -1 ≤ x + 1 ∧ x + 1 ≤ 6}
def set_B (m : ℝ) : set ℝ := {x | B m x}

theorem range_of_m (m : ℝ) : 
  (∀ x, set_B m x → set_A x) → 
  m < -2 ∨ (-1 ≤ m ∧ m ≤ 2) :=
by 
  sorry

end proper_subsets_A_range_of_m_l340_340729


namespace area_of_circle_l340_340950

noncomputable def area_of_region : ℝ := 
  let eq := λ (x y : ℝ), x^2 + y^2 - 4 * x + 6 * y = -9
  in 4 * real.pi

theorem area_of_circle :
  (let is_circle := ∀ x y, x^2 + y^2 - 4 * x + 6 * y = -9 -> (x - 2)^2 + (y + 3)^2 = 4 in
  is_circle → area_of_region = 4 * real.pi) :=
by
  let is_circle := ∀ x y, x^2 + y^2 - 4 * x + 6 * y = -9 -> (x - 2)^2 + (y + 3)^2 = 4
  exact λ _ => sorry

end area_of_circle_l340_340950


namespace product_of_four_consecutive_odd_numbers_is_perfect_square_l340_340492

theorem product_of_four_consecutive_odd_numbers_is_perfect_square (n : ℤ) :
    (n + 0) * (n + 2) * (n + 4) * (n + 6) = 9 :=
sorry

end product_of_four_consecutive_odd_numbers_is_perfect_square_l340_340492


namespace sum_of_sequence_l340_340044

theorem sum_of_sequence (n m : ℕ) (h : n^2 = (list.range (m - 100 + 1)).sum + 100) (h2 : n = (m + 100) / 2) : n + m = 497 :=
sorry

end sum_of_sequence_l340_340044


namespace zero_points_of_f_l340_340308

def f (x : ℝ) : ℝ := (1 / 3) * x - Real.log x

theorem zero_points_of_f :
  ¬(∃ x : ℝ, 1 / Real.exp 1 < x ∧ x < 1 ∧ f(x) = 0) ∧ (∃ x : ℝ, 1 < x ∧ x < Real.exp 1 ∧ f(x) = 0) :=
by
  sorry

end zero_points_of_f_l340_340308


namespace find_a_l340_340978

theorem find_a :
  (∃ x1 x2, (x1 + x2 = -2 ∧ x1 * x2 = a) ∧ (∃ y1 y2, (y1 + y2 = - a ∧ y1 * y2 = 2) ∧ (x1^2 + x2^2 = y1^2 + y2^2))) → 
  (a = -4) := 
by
  sorry

end find_a_l340_340978


namespace sum_of_all_integers_m_eq_18_l340_340588

noncomputable def sum_of_integers_satisfying_conditions : ℤ :=
let S := {m : ℤ | ∃ x : ℤ, 3 * x + 2 > m ∧ (x - 1) / 2 ≤ 1 ∧ 
  (∀ y : ℤ, (3 * y + 2 > m ∧ (y - 1) / 2 ≤ 1) ↔ (y = x ∨ y = x + 1))} in
@Finset.sum ℤ ℤ (Finset.filter (λ m : ℤ, m ∈ S) (Finset.range 8)) id

theorem sum_of_all_integers_m_eq_18 : sum_of_integers_satisfying_conditions = 18 :=
sorry

end sum_of_all_integers_m_eq_18_l340_340588


namespace dana_more_pencils_than_marcus_l340_340945

theorem dana_more_pencils_than_marcus :
  ∀ (Jayden Dana Marcus : ℕ), 
  (Jayden = 20) ∧ 
  (Dana = Jayden + 15) ∧ 
  (Jayden = 2 * Marcus) → 
  (Dana - Marcus = 25) :=
by
  intros Jayden Dana Marcus h
  rcases h with ⟨hJayden, hDana, hMarcus⟩
  sorry

end dana_more_pencils_than_marcus_l340_340945


namespace find_pairs_l340_340236

def polyRoots (a b c : ℂ) : List ℂ :=
  let Δ := b^2 - 4 * a * c
  [(-b + complex.sqrt Δ) / (2 * a), (-b - complex.sqrt Δ) / (2 * a)]

noncomputable def complexPolynomialDivisibility (n : ℕ) (r : ℂ) : Prop :=
  ∀ x : ℂ, x ∈ polyRoots 2 2 1 → (x + 1)^n = r 

theorem find_pairs :
  ∀ k : ℕ+, complexPolynomialDivisibility (4 * k) (-(1/4)^k) := 
by 
  -- Proof skipping with sorry
  sorry

end find_pairs_l340_340236


namespace minor_arc_circumference_l340_340383

-- Let P, Q, and R be points on a circle of radius 10
-- Let angle PRQ = 60 degrees
namespace CircleProof

def radius : ℝ := 10
def angle_PRQ : ℝ := 60

theorem minor_arc_circumference (radius : ℝ) (angle_PRQ : ℝ) : 
  0 < radius ∧ angle_PRQ = 60 → 
  let circumference := 2 * Real.pi * radius in
  let angle_center := 2 * angle_PRQ in
  let minor_arc := circumference * (angle_center / 360) in
  minor_arc = (20 * Real.pi / 3) :=
by
  intros h
  let r := h.1
  let a := h.2
  have circumference := 2 * Real.pi * radius
  have angle_center := 2 * angle_PRQ
  have minor_arc := circumference * (angle_center / 360)
  sorry
end CircleProof

end minor_arc_circumference_l340_340383


namespace max_wrappers_l340_340914

-- Definitions for the conditions
def total_wrappers : ℕ := 49
def andy_wrappers : ℕ := 34

-- The problem statement to prove
theorem max_wrappers : total_wrappers - andy_wrappers = 15 :=
by
  sorry

end max_wrappers_l340_340914


namespace triangle_properties_l340_340001

variable (A B C a b c : ℝ)
variable (h1 : a = Real.sqrt 7)
variable (h2 : b = 2)
variable (h3 : a * Real.sin B - Real.sqrt 3 * b * Real.cos A = 0)
variable (h4 : ∀ t : ℝ,  t ≠ 0 → ( Real.sin A - Real.sqrt 3 * Real.cos A = 0))

theorem triangle_properties : 
  A = Real.pi / 3 ∧ 
  (let c := 3 in 
    Areaₓ (triangle A B C) = (3 * Real.sqrt 3) / 2) := by
  sorry

end triangle_properties_l340_340001


namespace prime_n_if_power_of_prime_l340_340723

theorem prime_n_if_power_of_prime (n : ℕ) (h1 : n ≥ 2) (b : ℕ) (h2 : b > 0) (p : ℕ) (k : ℕ) 
  (hk : k > 0) (hb : (b^n - 1) / (b - 1) = p^k) : Nat.Prime n :=
sorry

end prime_n_if_power_of_prime_l340_340723


namespace f_is_odd_function_l340_340809

noncomputable def f (x : ℝ) := (Real.tan x) / (1 + Real.cos x)

theorem f_is_odd_function : ∀ x : ℝ, f x = -f (-x) := 
by
  intro x
  have f_def : f x = (Real.tan x) / (1 + Real.cos x) := rfl
  have f_neg_def : f (-x) = (Real.tan (-x)) / (1 + Real.cos (-x)) := rfl
  rw [Real.tan_neg, Real.cos_neg]
  rw [of_real_neg]
  sorry 

end f_is_odd_function_l340_340809


namespace sum_evaluation_l340_340111

theorem sum_evaluation :
  (∑ k in Finset.range 50, (-1)^(k + 1) * (k^3 + k^2 + k + 1) / k.factorial) = 2601 / 50.factorial - 1 := 
sorry

end sum_evaluation_l340_340111


namespace min_balls_for_same_color_l340_340344

theorem min_balls_for_same_color (R Y B : ℕ) 
    (h_total : R + Y + B = 88) 
    (h_cond : ∀ S : Finset (Fin 88), S.card = 24 → ∃ c, S.count (λ b, color b = c) ≥ 10) : 
    ∃ S : Finset (Fin 88), S.card = 44 ∧ ∃ c, S.count (λ b, color b = c) ≥ 20 := 
sorry

end min_balls_for_same_color_l340_340344


namespace find_solutions_l340_340565

theorem find_solutions (n k : ℕ) (hn : n > 0) (hk : k > 0) : 
  n! + n = n^k → (n, k) = (2, 2) ∨ (n, k) = (3, 2) ∨ (n, k) = (5, 3) :=
sorry

end find_solutions_l340_340565


namespace coefficient_x2y3_l340_340076

theorem coefficient_x2y3 (C : ℕ → ℕ → ℕ)
  (hC : ∀ n k, C n k = Nat.choose n k) :
  (∑ r in Finset.range (6), C 5 r * ((1/2) ^ (5 - r)) * (-2) ^ r * (r.choose 2) * (x ^ (5 - r)) * (y ^ r)) = -20 :=
by
  sorry

end coefficient_x2y3_l340_340076


namespace f_symmetric_about_point_l340_340829

def h (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 4)

def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 4) + 2

theorem f_symmetric_about_point (x : ℝ) : f(x) = 2 - h(-x) :=
sorry

end f_symmetric_about_point_l340_340829


namespace S5_sum_l340_340355

noncomputable def arithmetic_sequence_S5 (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  (a 2 + a 3 + a 4 = 3) ∧ (∀ n, S n = ∑ i in finset.range n, a (i + 1)) 

theorem S5_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (h : arithmetic_sequence_S5 a S) : S 5 = 5 := 
  sorry

end S5_sum_l340_340355


namespace total_bananas_l340_340413

theorem total_bananas (bunches_8 bunches_7 : ℕ) (bananas_8 bananas_7 : ℕ) (h_bunches_8 : bunches_8 = 6) (h_bananas_8 : bananas_8 = 8) (h_bunches_7 : bunches_7 = 5) (h_bananas_7 : bananas_7 = 7) :
  bunches_8 * bananas_8 + bunches_7 * bananas_7 = 83 :=
by
  rw [h_bunches_8, h_bananas_8, h_bunches_7, h_bananas_7]
  norm_num

end total_bananas_l340_340413


namespace abs_value_1_minus_z_times_z_conj_l340_340303

noncomputable def z : ℂ := 2 + complex.I
noncomputable def z_conj : ℂ := 2 - complex.I

theorem abs_value_1_minus_z_times_z_conj : 
  |(1 - z) * z_conj| = √10 := 
sorry

end abs_value_1_minus_z_times_z_conj_l340_340303


namespace martin_initial_spending_l340_340409

theorem martin_initial_spending :
  ∃ (x : ℝ), 
    ∀ (a b : ℝ), 
      a = x - 100 →
      b = a - 0.20 * a →
      x - b = 280 →
      x = 1000 :=
by
  sorry

end martin_initial_spending_l340_340409


namespace min_value_inequality_l340_340719

noncomputable def min_value (x y z w : ℝ) : ℝ :=
  x^2 + 4 * x * y + 9 * y^2 + 6 * y * z + 8 * z^2 + 3 * x * w + 4 * w^2

theorem min_value_inequality 
  (x y z w : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0)
  (h_prod : x * y * z * w = 3) : 
  min_value x y z w ≥ 81.25 := 
sorry

end min_value_inequality_l340_340719


namespace option_A_option_B_option_C_option_D_l340_340134

theorem option_A : (-4:ℤ)^2 ≠ -(4:ℤ)^2 := sorry
theorem option_B : (-2:ℤ)^3 = -2^3 := sorry
theorem option_C : (-1:ℤ)^2020 ≠ (-1:ℤ)^2021 := sorry
theorem option_D : ((2:ℚ)/(3:ℚ))^3 = ((2:ℚ)/(3:ℚ))^3 := sorry

end option_A_option_B_option_C_option_D_l340_340134


namespace find_multiple_l340_340549

theorem find_multiple :
  ∃ m : ℕ, 46 = m * 20 + 6 :=
begin
  use 2,
  -- proof goes here
  sorry
end

end find_multiple_l340_340549


namespace min_distinct_sums_l340_340283

theorem min_distinct_sums (n : ℕ) (h : n ≥ 5) : 
  ∃ S : Finset ℕ, S.card = n ∧ 
  ∀ T : Finset (ℕ × ℕ), (∀ x, x ∈ T → x.1 < x.2) → T ⊆ S ×ˢ S → 
  Finset.card (T.image (λ x, x.1 + x.2)) = 2 * n - 3 :=
sorry

end min_distinct_sums_l340_340283


namespace area_of_region_l340_340953

noncomputable def enclosed_area : ℝ :=
  let circle_equation := λ (x y : ℝ), x ^ 2 + y ^ 2 - 4 * x + 6 * y = -9
  in 4 * Real.pi

theorem area_of_region : enclosed_area = 4 * Real.pi :=
  sorry

end area_of_region_l340_340953


namespace smoking_correlation_l340_340064

-- Define the data points for X and Y
def X : list ℝ := [16, 18, 20, 22]
def Y : list ℝ := [15.10, 12.81, 9.72, 3.21]

-- Define the data points for U and V
def U : list ℝ := [10, 20, 30]
def V : list ℝ := [7.5, 9.5, 16.6]

-- Define the correlation coefficients
def r1 : ℝ := sorry  -- Placeholder for the linear correlation coefficient between X and Y
def r2 : ℝ := sorry  -- Placeholder for the linear correlation coefficient between U and V

-- State the theorem to be proven
theorem smoking_correlation :
  r1 < 0 ∧ r2 > 0 :=
by
  sorry -- Proof follows from the given conditions and needs to be filled in

end smoking_correlation_l340_340064


namespace sixth_year_fee_l340_340191

def first_year_fee : ℕ := 80
def yearly_increase : ℕ := 10

def membership_fee (year : ℕ) : ℕ :=
  first_year_fee + (year - 1) * yearly_increase

theorem sixth_year_fee : membership_fee 6 = 130 :=
  by sorry

end sixth_year_fee_l340_340191


namespace range_of_a_l340_340257

noncomputable def f (x a : ℝ) : ℝ :=
  x^2 + abs (x - a)

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f 0.5 a > f x) ∧ (∀ x : ℝ, f (-0.5) a > f x) → -0.5 < a ∧ a < 0.5 :=
by 
  intros H
  sorry

end range_of_a_l340_340257


namespace problem_l340_340388

-- Define the problem conditions
variable (a b : ℝ)
theorem problem (h_root : (1 : ℂ) + complex.i * real.sqrt 3 = 0) (h_sum_real_roots : -4) :
  a + b = 15 :=
by
  sorry

end problem_l340_340388


namespace total_broken_marbles_l340_340262

theorem total_broken_marbles (marbles_set1 marbles_set2 : ℕ) 
  (percentage_broken_set1 percentage_broken_set2 : ℚ) 
  (h1 : marbles_set1 = 50) 
  (h2 : percentage_broken_set1 = 0.1) 
  (h3 : marbles_set2 = 60) 
  (h4 : percentage_broken_set2 = 0.2) : 
  (marbles_set1 * percentage_broken_set1 + marbles_set2 * percentage_broken_set2 = 17) := 
by 
  sorry

end total_broken_marbles_l340_340262


namespace ChocolateBallAverage_l340_340200

noncomputable def f : ℕ → ℝ
| 2       := 1
| (n + 1) := 1  -- based on the induction hypothesis that f(n) = 1

theorem ChocolateBallAverage (n : ℕ) (h : n ≥ 2) : f n = 1 :=
by
  sorry

end ChocolateBallAverage_l340_340200


namespace number_of_non_symmetric_letters_is_3_l340_340202

def letters_in_JUNIOR : List Char := ['J', 'U', 'N', 'I', 'O', 'R']

def axis_of_symmetry (c : Char) : Bool :=
  match c with
  | 'J' => false
  | 'U' => true
  | 'N' => false
  | 'I' => true
  | 'O' => true
  | 'R' => false
  | _   => false

def letters_with_no_symmetry : List Char :=
  letters_in_JUNIOR.filter (λ c => ¬axis_of_symmetry c)

theorem number_of_non_symmetric_letters_is_3 :
  letters_with_no_symmetry.length = 3 :=
by
  sorry

end number_of_non_symmetric_letters_is_3_l340_340202


namespace molecular_weight_of_compound_l340_340841

theorem molecular_weight_of_compound :
  let atomic_weight_H := 1.008
  let atomic_weight_Cr := 51.996
  let atomic_weight_O := 15.999
  let num_H := 2
  let num_Cr := 1
  let num_O := 4
  let molecular_weight := (num_H * atomic_weight_H) + (num_Cr * atomic_weight_Cr) + (num_O * atomic_weight_O)
  molecular_weight = 118.008 :=
by
  let atomic_weight_H := 1.008
  let atomic_weight_Cr := 51.996
  let atomic_weight_O := 15.999
  let num_H := 2
  let num_Cr := 1
  let num_O := 4
  let molecular_weight := (num_H * atomic_weight_H) + (num_Cr * atomic_weight_Cr) + (num_O * atomic_weight_O)
  show molecular_weight = 118.008
  sorry

end molecular_weight_of_compound_l340_340841


namespace centroid_of_V_l340_340017

-- Define the set of points (x, y) satisfying the given conditions
noncomputable def V : Set (ℝ × ℝ) :=
  {p | (abs p.1) ≤ p.2 ∧ p.2 ≤ (abs p.1 + 3) ∧ p.2 ≤ 4}

-- Statement that the centroid of V is (0, 2.31)
theorem centroid_of_V :
  centroid V = (0, 2.31) :=
sorry

end centroid_of_V_l340_340017


namespace ratio_B_to_A_l340_340199

theorem ratio_B_to_A (A B S : ℕ) 
  (h1 : A = 2 * S)
  (h2 : A = 80)
  (h3 : B - S = 200) :
  B / A = 3 :=
by sorry

end ratio_B_to_A_l340_340199


namespace equilateral_triangle_side_length_l340_340769

theorem equilateral_triangle_side_length (P Q R S A B C : Type)
  [Point P] [Point Q] [Point R] [Point S] [Point A] [Point B] [Point C] :
  (within_triangle P A B C) →
  orthogonal_projection P Q A B →
  orthogonal_projection P R B C →
  orthogonal_projection P S C A →
  distance P Q = 2 →
  distance P R = 3 →
  distance P S = 4 →
  distance A B = 6 * √3 :=
by
  sorry

end equilateral_triangle_side_length_l340_340769


namespace area_ratio_of_quad_is_1_over_10_l340_340673

variables {A B C D E F O : Type} 

/- Conditions as per given problem -/
variables [Triangle A B C]   -- ABC is a triangle
variable (D : Point (BC)) 
variable (E : Point (AC)) 
variable (O : Point)
variable (F : Point)

-- D is a point such that BD/DC = 1/3
axiom BD_DC : ∀ (BD DC : Length), (BD / DC) = (1 / 3)

-- E is the midpoint of AC
axiom E_mid_AC : ∀ (AC : Segment), midpoint E AC 

-- AD and BE intersect at O
axiom AD_BE_intersect_O : ∀ (AD BE : Line), intersection AD BE = O

-- CO intersects AB at F
axiom CO_AB_intersect_F : ∀ (CO AB : Line), intersection CO AB = F

-- Question: Prove the ratio of the area of quadrilateral BDOF to the area of triangle ABC is 1/10
theorem area_ratio_of_quad_is_1_over_10 (S_BDOF S_ABC : ℝ) : 
  (S_BDOF / S_ABC) = (1 / 10) :=
sorry

end area_ratio_of_quad_is_1_over_10_l340_340673


namespace imaginary_part_zero_l340_340338

-- Definitions
def z : ℂ := 1 + complex.i
def z_conj : ℂ := 1 - complex.i
def z_squared : ℂ := z^2
def z_conj_squared : ℂ := z_conj^2
def sum_squares : ℂ := z_squared + z_conj_squared

-- Statement of the theorem we need to prove
theorem imaginary_part_zero : (sum_squares).im = 0 := by
  sorry

end imaginary_part_zero_l340_340338


namespace find_BL_l340_340744

-- Given points and lines as defined in the problem
variables {A B C M L K : Type}
variables {circumcircle : Set (Point → Point → Point → Prop)}

-- Assume given lengths
variables (a b c : ℝ)
variables (AL AK CK : ℝ)
variables (BL : ℝ)

-- Define conditions according to problem statement
def eq_length_AL := AL = a
def eq_length_BK := BK = b
def eq_length_CK := CK = c

-- Goal: Prove that BL = ab/c
theorem find_BL (h1 : eq_length_AL) (h2 : eq_length_BK) (h3 : eq_length_CK) : BL = a * b / c :=
by
  sorry

end find_BL_l340_340744


namespace first_player_strategy_l340_340478

-- Define the sequence and the game conditions
def sequence := List.range 102 |>.tail!  -- creates the list [1, 2, ..., 101]

-- Definition of the game state at any point
structure GameState where
  remaining : List ℕ
  turn : ℕ

-- Initial game state
def initial_state : GameState :=
  { remaining := sequence, turn := 0 }

-- Function to simulate a move by eliminating n elements from the game state
def make_move (state : GameState) (eliminated : List ℕ) : GameState :=
  { state with remaining := state.remaining.filter (λ x => ¬ x ∈ eliminated), turn := state.turn + 1 }

-- Define what it means for the first player to ensure scoring at least 55 points
def first_player_can_score (state : GameState) : Prop :=
  ∃ remaining_1 remaining_2, remaining_1 ∈ state.remaining ∧ remaining_2 ∈ state.remaining ∧
  (remaining_1 - remaining_2).natAbs ≥ 55

-- The theorem to be proven
theorem first_player_strategy : first_player_can_score initial_state :=
sorry

end first_player_strategy_l340_340478


namespace equilateral_triangle_side_length_l340_340758

theorem equilateral_triangle_side_length 
  {P Q R S : Point} 
  {A B C : Triangle}
  (h₁ : is_inside P A B C)
  (h₂ : is_perpendicular P Q A B)
  (h₃ : is_perpendicular P R B C)
  (h₄ : is_perpendicular P S C A)
  (h₅ : distance P Q = 2)
  (h₆ : distance P R = 3)
  (h₇ : distance P S = 4)
  : side_length A B C = 6 * sqrt 3 :=
by 
  sorry

end equilateral_triangle_side_length_l340_340758


namespace factor_expression_l340_340930

theorem factor_expression :
  (8 * x ^ 4 + 34 * x ^ 3 - 120 * x + 150) - (-2 * x ^ 4 + 12 * x ^ 3 - 5 * x + 10) 
  = 5 * x * (2 * x ^ 3 + (22 / 5) * x ^ 2 - 23 * x + 28) :=
sorry

end factor_expression_l340_340930


namespace greatest_divisor_l340_340581

theorem greatest_divisor (d : ℕ) :
  (6215 % d = 23 ∧ 7373 % d = 29 ∧ 8927 % d = 35) → d = 36 :=
by
  sorry

end greatest_divisor_l340_340581


namespace find_internal_angles_l340_340363

noncomputable def triangle_angles (A B C : ℝ) : Prop :=
  sin A + cos A = sqrt 2 ∧
  sqrt 3 * cos A = -sqrt 2 * cos (π - B) ∧
  A + B + C = π

theorem find_internal_angles (A B C : ℝ) :
  triangle_angles A B C → 
  A = π / 4 ∧ B = π / 6 ∧ C = 7 * π / 12 :=
by
  intro h
  sorry

end find_internal_angles_l340_340363


namespace verify_n_l340_340349

noncomputable def find_n (n : ℕ) : Prop :=
  let widget_rate1 := 3                             -- Widgets per worker-hour from the first condition
  let whoosit_rate1 := 2                            -- Whoosits per worker-hour from the first condition
  let widget_rate3 := 1                             -- Widgets per worker-hour from the third condition
  let minutes_per_widget := 1                       -- Arbitrary unit time for one widget
  let minutes_per_whoosit := 2                      -- 2 times unit time for one whoosit based on problem statement
  let whoosit_rate3 := 2 / 3                        -- Whoosits per worker-hour from the third condition
  let widget_rate2 := 540 / (90 * 3 : ℕ)            -- Widgets per hour in the second condition
  let whoosit_rate2 := n / (90 * 3 : ℕ)             -- Whoosits per hour in the second condition
  widget_rate2 = 2 ∧ whoosit_rate2 = 4 / 3 ∧
  (minutes_per_widget < minutes_per_whoosit) ∧
  (whoosit_rate2 = (4 / 3 : ℚ) ↔ n = 360)

theorem verify_n : find_n 360 :=
by sorry

end verify_n_l340_340349


namespace Dana_has_25_more_pencils_than_Marcus_l340_340941

theorem Dana_has_25_more_pencils_than_Marcus (JaydenPencils : ℕ) (h1 : JaydenPencils = 20) :
  let DanaPencils := JaydenPencils + 15,
      MarcusPencils := JaydenPencils / 2
  in DanaPencils - MarcusPencils = 25 := 
by
  sorry -- proof to be filled in

end Dana_has_25_more_pencils_than_Marcus_l340_340941


namespace sum_of_m_l340_340324

open Real EuclideanSpace

-- We assume the conditions: two non-zero perpendicular vectors a and b, with |b| = 1
variables {a b : EuclideanSpace ℝ (Fin 2)}

-- Conditions from the problem
axiom a_nonzero : a ≠ 0
axiom b_nonzero : b ≠ 0
axiom ab_perpendicular : inner a b = 0
axiom b_normalized : ∥b∥ = 1

-- The theorem states the sum of all real numbers m such that (a + m * b) and (a + (1 - m) * b) are perpendicular is 1.
theorem sum_of_m (m : ℝ) : (inner (a + m • b) (a + (1 - m) • b) = 0) → m = 1 := sorry

end sum_of_m_l340_340324


namespace overall_profit_is_600_l340_340009

def grinder_cp := 15000
def mobile_cp := 10000
def laptop_cp := 20000
def camera_cp := 12000

def grinder_loss_percent := 4 / 100
def mobile_profit_percent := 10 / 100
def laptop_loss_percent := 8 / 100
def camera_profit_percent := 15 / 100

def grinder_sp := grinder_cp * (1 - grinder_loss_percent)
def mobile_sp := mobile_cp * (1 + mobile_profit_percent)
def laptop_sp := laptop_cp * (1 - laptop_loss_percent)
def camera_sp := camera_cp * (1 + camera_profit_percent)

def total_cp := grinder_cp + mobile_cp + laptop_cp + camera_cp
def total_sp := grinder_sp + mobile_sp + laptop_sp + camera_sp

def overall_profit_or_loss := total_sp - total_cp

theorem overall_profit_is_600 : overall_profit_or_loss = 600 := by
  sorry

end overall_profit_is_600_l340_340009


namespace leap_day_l340_340014

theorem leap_day (h : nat) :
  let feb_29_1996 := 4 in -- Thursday
  let years_between := 2024 - 1996 in
  let days_between := 21 * 365 + 7 * 366 in
  (feb_29_1996 + days_between % 7) % 7 = 2 := -- Tuesday
by
  let feb_29_1996 := 4
  let years_between := 2024 - 1996
  let days_between := 21 * 365 + 7 * 366
  show (feb_29_1996 + days_between % 7) % 7 = 2 from sorry

end leap_day_l340_340014


namespace train_speed_correct_l340_340184

noncomputable def train_speed_problem : ℝ :=
let distance_km := 220 / 1000 in
let time_hr := 12 / 3600 in
let relative_speed := distance_km / time_hr in
let man_speed := 6 in
relative_speed - man_speed

theorem train_speed_correct (distance_m : ℝ) (time_s : ℝ) (man_speed : ℝ) 
  (distance_km : ℝ := distance_m / 1000)
  (time_hr : ℝ := time_s / 3600)
  (relative_speed : ℝ := distance_km / time_hr)
  (train_speed : ℝ := relative_speed - man_speed) 
  (h1 : distance_m = 220) 
  (h2 : time_s = 12) 
  (h3 : man_speed = 6) 
  : train_speed = 60 :=
by
  sorry

end train_speed_correct_l340_340184


namespace min_value_of_f_range_of_t_l340_340317

noncomputable def f (x : ℝ) : ℝ := abs (x + 2) + abs (x - 4)

theorem min_value_of_f : ∃ m, (∀ x, f x ≥ m) ∧ (∃ x, f x = m) :=
by
  use 6
  sorry

theorem range_of_t (t : ℝ) :
  (∃ x ∈ Icc (-3 : ℝ) 5, f x ≤ t^2 - t) ↔ t ≤ -2 ∨ t ≥ 3 :=
by
  sorry

end min_value_of_f_range_of_t_l340_340317


namespace forty_percent_of_number_l340_340859

theorem forty_percent_of_number (N : ℝ) 
  (h : (1/4) * (1/3) * (2/5) * N = 35) : 0.4 * N = 420 :=
by
  sorry

end forty_percent_of_number_l340_340859


namespace calculate_gas_volumes_l340_340507

variable (gas_volume_western : ℝ) 
variable (total_gas_volume_non_western : ℝ)
variable (population_non_western : ℝ)
variable (total_gas_percentage_russia : ℝ)
variable (gas_volume_russia : ℝ)
variable (population_russia : ℝ)

theorem calculate_gas_volumes 
(h_western : gas_volume_western = 21428)
(h_non_western : total_gas_volume_non_western = 185255)
(h_population_non_western : population_non_western = 6.9)
(h_percentage_russia : total_gas_percentage_russia = 68.0)
(h_gas_volume_russia : gas_volume_russia = 30266.9)
(h_population_russia : population_russia = 0.147)
: 
  (total_gas_volume_non_western / population_non_western = 26848.55) ∧ 
  (gas_volume_russia / population_russia ≈ 302790.13) := 
  sorry

end calculate_gas_volumes_l340_340507


namespace number_of_values_a_l340_340440

theorem number_of_values_a (a : ℕ) (h1 : 3 ∣ a) (h2 : a ∣ 12) (h3 : 0 < a) : 3 = {b | 3 ∣ b ∧ b ∣ 12 ∧ 0 < b}.to_finset.card :=
by
  sorry

end number_of_values_a_l340_340440


namespace packed_oranges_l340_340521

theorem packed_oranges (oranges_per_box : ℕ) (boxes_used : ℕ) (total_oranges : ℕ) 
  (h1 : oranges_per_box = 10) (h2 : boxes_used = 265) : 
  total_oranges = 2650 :=
by 
  sorry

end packed_oranges_l340_340521


namespace percent_Asian_in_West_l340_340550

noncomputable def NE_Asian := 2
noncomputable def MW_Asian := 2
noncomputable def South_Asian := 2
noncomputable def West_Asian := 6

noncomputable def total_Asian := NE_Asian + MW_Asian + South_Asian + West_Asian

theorem percent_Asian_in_West (h1 : total_Asian = 12) : (West_Asian / total_Asian) * 100 = 50 := 
by sorry

end percent_Asian_in_West_l340_340550


namespace radius_of_sphere_eq_l340_340469

theorem radius_of_sphere_eq (r : ℝ) : 
  (4 / 3) * Real.pi * r^3 = 4 * Real.pi * r^2 → r = 3 :=
by
  sorry

end radius_of_sphere_eq_l340_340469


namespace number_of_correct_propositions_l340_340683

-- Define the propositions as separate statements
def prop1 : Prop := ∀ (l₁ l₂ l₃ : Line), (l₁ ∥ l₃ ∧ l₂ ∥ l₃) → l₁ ∥ l₂
def prop2 : Prop := ∀ (P₁ P₂ : Plane) (l : Line), (P₁ ∥ l ∧ P₂ ∥ l) → P₁ ∥ P₂
def prop3 : Prop := ∀ (P₁ P₂ P₃ : Plane), (P₁ ⊥ P₃ ∧ P₂ ⊥ P₃) → P₁ ∥ P₂
def prop4 : Prop := ∀ (l₁ l₂ : Line) (P : Plane), (l₁ ⊥ P ∧ l₂ ⊥ P) → l₁ ∥ l₂

-- Theorem statement
theorem number_of_correct_propositions : (∃ (propositions : List Prop), propositions = [prop1, prop2, prop3, prop4] ∧ (prop1 → prop4 → prop2 → prop3) = 2) :=
by {
  sorry
}

end number_of_correct_propositions_l340_340683


namespace parabola_chord_length_l340_340604

theorem parabola_chord_length :
  ∀ (A B : Point) (M : Point),
  is_on_parabola A (λ x, x^2 - 7) ∧
  is_on_parabola B (λ x, x^2 - 7) ∧
  line_symmetric A B (λ p, p.1 + p.2 = 0) →
  length_of_line_segment A B = 5 * Real.sqrt 2 := sorry

end parabola_chord_length_l340_340604


namespace seokgi_share_is_67_l340_340783

-- The total length of the wire
def length_of_wire := 150

-- Seokgi's share is 16 cm shorter than Yeseul's share
def is_shorter_by (Y S : ℕ) := S = Y - 16

-- The sum of Yeseul's and Seokgi's shares equals the total length
def total_share (Y S : ℕ) := Y + S = length_of_wire

-- Prove that Seokgi's share is 67 cm
theorem seokgi_share_is_67 (Y S : ℕ) (h1 : is_shorter_by Y S) (h2 : total_share Y S) : 
  S = 67 :=
sorry

end seokgi_share_is_67_l340_340783


namespace value_of_k_l340_340792

-- Define the values and conditions of the problem
def circumference (r : ℝ) : ℝ := 2 * Real.pi * r
def area (r : ℝ) : ℝ := Real.pi * r^2

theorem value_of_k :
  ∀ (r : ℝ), circumference r = 18 * Real.pi → area r = 81 * Real.pi :=
by
  intro r
  intro hcircumference
  -- circumference r = 2 * Real.pi * r = 18 * Real.pi
  have h_radius : r = 9 :=
    by sorry
  -- area r = Real.pi * r^2 = Real.pi * (9^2) = 81 * Real.pi
  rw [h_radius]
  sorry

end value_of_k_l340_340792


namespace broken_more_than_perfect_spiral_l340_340647

theorem broken_more_than_perfect_spiral :
  let perfect_shells := 17
  let broken_shells := 52
  let broken_spiral_shells := broken_shells / 2
  let perfect_non_spiral_shells := 12
  let perfect_spiral_shells := perfect_shells - perfect_non_spiral_shells
  in broken_spiral_shells - perfect_spiral_shells = 21 :=
by
  sorry

end broken_more_than_perfect_spiral_l340_340647


namespace asymptotes_of_hyperbola_l340_340580

theorem asymptotes_of_hyperbola :
  (∀ x y : ℝ, (x^2 / 4 - y^2 / 9 = 1) → (y = (3 / 2) * x ∨ y = -(3 / 2) * x)) :=
begin
  sorry
end

end asymptotes_of_hyperbola_l340_340580


namespace convex_inequality_l340_340730

-- Conditions for the problem
def convex (f : ℝ → ℝ) : Prop :=
∀ x₁ x₂ : ℝ, ∀ t : ℝ, 0 < t → t < 1 → f (t * x₁ + (1 - t) * x₂) ≤ t * f x₁ + (1 - t) * f x₂

-- Conclusion to be proved
theorem convex_inequality (f : ℝ → ℝ) (h : convex f) :
  ∀ (n : ℕ) (a : ℕ → ℝ),
  2 ≤ n →
  (∀ k : ℕ, k < n → a k ≥ a (k + 1)) →
  a (n + 1) = a 1 →
  ∑ k in Finset.range n, f (a (k + 1)) * a k ≤ ∑ k in Finset.range n, f (a k) * a (k + 1) := sorry

end convex_inequality_l340_340730


namespace ellipse_standard_form_exists_line_intersection_range_l340_340289

-- Define the conditions of the ellipse
def isEllipse (a b c : ℝ) (e : ℝ) (l : ℝ) : Prop :=
  e = c / a ∧ e = 1 / 2 ∧ a - c = l ∧ l = 1 ∧ a > b ∧ b > 0

-- Define the standard form of the ellipse
def standardEllipse (a b : ℝ) : Prop :=
  (∀ x y : ℝ, x^2 / (a^2) + y^2 / (b^2) = 1)

-- Ellipse C satisfies the given conditions
theorem ellipse_standard_form_exists (a b c : ℝ) :
  isEllipse a b c 1/2 1 →
  standardEllipse 2 (sqrt 3) :=
sorry

-- Define the line equation and intersecting condition
def isLineIntersectsEllipse (k m a b : ℝ) : Prop :=
  ∃ x1 x2 y1 y2 : ℝ, y1 = k * x1 + m ∧ y2 = k * x2 + m ∧
  x1^2 / a^2 + y1^2 / b^2 = 1 ∧ x2^2 / a^2 + y2^2 / b^2 = 1 ∧
  (x1 * x2 + y1 * y2 = 0)

-- Prove the range of m for the line intersection condition
theorem line_intersection_range (k m : ℝ) :
  (3 + 4 * k^2 > m^2) → (7 * m^2 = 12 + 12 * k^2) →
  isLineIntersectsEllipse k m 2 (sqrt 3) →
  m ∈ {m : ℝ | m ≥ 2 * sqrt 21 / 7} ∪ {m : ℝ | m ≤ -2 * sqrt 21 / 7} :=
sorry

end ellipse_standard_form_exists_line_intersection_range_l340_340289


namespace factor_expression_l340_340967

variable (b : ℤ)

theorem factor_expression : 221 * b^2 + 17 * b = 17 * b * (13 * b + 1) :=
by
  sorry

end factor_expression_l340_340967


namespace find_marked_price_l340_340532

noncomputable def marked_price 
  (purchase_price : ℝ) 
  (purchase_discount : ℝ) 
  (cost_price : ℝ) 
  (profit_margin : ℝ) 
  (selling_price : ℝ) 
  (selling_discount : ℝ) 
  (marked_price : ℝ) : Prop :=
  purchase_price - (purchase_price * purchase_discount) = cost_price ∧
  cost_price + (cost_price * profit_margin) = selling_price ∧
  selling_price / (1 - selling_discount) = marked_price

theorem find_marked_price : ∃ x, 
  marked_price 50 0.20 40 0.25 50 0.10 x ∧ x ≈ 55.56 :=
begin
  sorry
end

end find_marked_price_l340_340532


namespace school_students_count_l340_340680

def students_in_school (c n : ℕ) : ℕ := n * c

theorem school_students_count
  (c n : ℕ)
  (h1 : n * c = (n - 6) * (c + 5))
  (h2 : n * c = (n - 16) * (c + 20)) :
  students_in_school c n = 900 :=
by
  sorry

end school_students_count_l340_340680


namespace at_least_two_zeros_l340_340083

variable (f : ℝ → ℝ)

theorem at_least_two_zeros (h_cont : ∀ x ∈ set.Icc 1 3, continuous_at f x)
  (h1 : f 1 * f 2 < 0)
  (h2 : f 2 * f 3 < 0) :
  ∃ x1 x2 ∈ set.Ioo 1 3, x1 ≠ x2 ∧ f x1 = 0 ∧ f x2 = 0 :=
by
  sorry

end at_least_two_zeros_l340_340083


namespace memorable_numbers_count_l340_340561

def is_memorable_number (d : Fin 10 → Fin 8 → ℕ) : Prop :=
  d 0 0 = d 1 0 ∧ d 0 1 = d 1 1 ∧ d 0 2 = d 1 2 ∧ d 0 3 = d 1 3

theorem memorable_numbers_count : 
  ∃ n : ℕ, n = 10000 ∧ ∀ (d : Fin 10 → Fin 8 → ℕ), is_memorable_number d → n = 10000 :=
sorry

end memorable_numbers_count_l340_340561


namespace echo_books_inequality_l340_340780

def C (n : ℕ) : ℕ :=
  if 1 ≤ n ∧ n ≤ 30 then 15 * n
  else if 31 ≤ n ∧ n ≤ 60 then 13 * n
  else if 61 ≤ n then 12 * n
  else 0

theorem echo_books_inequality : 
  ∃ n1 n2 n3 n4 n5 : ℕ, 
  n1 ≠ n2 ∧ n1 ≠ n3 ∧ n1 ≠ n4 ∧ n1 ≠ n5 ∧
  n2 ≠ n3 ∧ n2 ≠ n4 ∧ n2 ≠ n5 ∧
  n3 ≠ n4 ∧ n3 ≠ n5 ∧
  n4 ≠ n5 ∧
  (1 ≤ n1 ∧ C(n1) > C(n1 + 1)) ∧
  (1 ≤ n2 ∧ C(n2) > C(n2 + 1)) ∧
  (1 ≤ n3 ∧ C(n3) > C(n3 + 1)) ∧
  (1 ≤ n4 ∧ C(n4) > C(n4 + 1)) ∧
  (1 ≤ n5 ∧ C(n5) > C(n5 + 1)) ∧
  (∀ m : ℕ, 1 ≤ m → C(m) > C(m + 1) → (m = n1 ∨ m = n2 ∨ m = n3 ∨ m = n4 ∨ m = n5)) :=
sorry

end echo_books_inequality_l340_340780


namespace complex_multiplication_l340_340619

def i := Complex.I

theorem complex_multiplication (i := Complex.I) : (-1 + i) * (2 - i) = -1 + 3 * i := 
by 
    -- The actual proof steps would go here.
    sorry

end complex_multiplication_l340_340619


namespace range_and_smallest_mode_l340_340342

def stem_and_leaf_scores : List ℕ := [51, 51, 52, 52, 55, 55, 67, 68, 69, 72, 72, 74, 74, 74, 76, 81, 83, 83, 83, 85, 90, 91, 92, 95, 95, 95, 101, 101, 101, 104, 105, 119, 119, 119]

theorem range_and_smallest_mode
    (scores : List ℕ)
    (highest_score : ℕ := scores.maximum |> Option.getD 0)
    (lowest_score : ℕ := scores.minimum |> Option.getD 0)
    (range : ℕ := highest_score - lowest_score)
    (frequencies : List (ℕ ×ℕ) := List.countp (λ x => x ∈ scores))
    (max_frequency : ℕ := frequencies.maximumBy (λ p => p.snd) |> Option.getD (0, 0) |> Prod.snd)
    (modes : List ℕ := frequencies.filter (λ p => p.snd = max_frequency) |> List.map Prod.fst)
    (smallest_mode : ℕ := modes.minimum |> Option.getD 0)
    : range = 68 ∧ smallest_mode = 51 :=
by
  sorry

end range_and_smallest_mode_l340_340342


namespace sum_of_b_sequence_l340_340319

-- Define the sequence b_n
def b (n : ℕ) : ℕ := n^2 * 2^n

-- Define the partial sum of the sequence b_n
def T (n : ℕ) : ℕ := ∑ i in Finset.range (n + 1), b i

-- Define the sequence a_n
def a (n : ℕ) : ℕ := (2 * n - 1) * 2^n

-- Define the partial sum of the sequence a_n
def S (n : ℕ) : ℕ := (2*n - 3) * 2^(n+1) + 6

-- The statement to prove
theorem sum_of_b_sequence (n : ℕ) : T n = (n^2 - 2*n + 3) * 2^(n+1) - 6 :=
by { sorry }

end sum_of_b_sequence_l340_340319


namespace geometric_sequence_term_formula_sum_of_terms_of_n_an_l340_340602

theorem geometric_sequence_term_formula
  (a : ℕ → ℕ)
  (h1 : a 2 = 4)
  (h2 : a 5 = 32)
  (h3 : ∀ n : ℕ, a n = a 1 * (2 : ℕ) ^ (n - 1)) :
  ∀ n : ℕ, a n = 2 ^ n :=
sorry

theorem sum_of_terms_of_n_an
  (a : ℕ → ℕ)
  (h : ∀ n : ℕ, a n = 2 ^ n) :
  ∀ (n : ℕ), ∑ i in Finset.range n, (i + 1) * a (i + 1) = (n - 1) * 2 ^ (n + 1) + 2 :=
sorry

end geometric_sequence_term_formula_sum_of_terms_of_n_an_l340_340602


namespace product_multiplication_rule_l340_340205

theorem product_multiplication_rule (a : ℝ) : (a * a^3)^2 = a^8 := 
by  
  -- The proof will apply the rule of product multiplication here
  sorry

end product_multiplication_rule_l340_340205


namespace quadratic_roots_specific_solution_l340_340259

theorem quadratic_roots :
  ∀ (m x : ℝ),
    x^2 - 3 * x - m * x + m - 1 = 0 → 
    let Δ := (3 + m)^2 - 4 * (m - 1) in
    (Δ > 0) :=
begin
  intros m x h,
  let Δ := (3 + m)^2 - 4 * (m - 1),
  have pos_discriminant : Δ > 0 := by {
    sorry -- proof to be filled
  },
  exact pos_discriminant
end

theorem specific_solution :
  ∀ (x1 x2 m : ℝ),
    (x1^2 - 3 * x1 - m * x1 + m - 1 = 0)  ∧ 
    (x2^2 - 3 * x2 - m * x2 + m - 1 = 0) ∧
    (3 * x1 - x1 * x2 + 3 * x2 = 12) →
    m = 1 ∧ x1 = 0 ∧ x2 = 4 :=
begin
  intros x1 x2 m h,
  have condition_satisfied : (3 * x1 - x1 * x2 + 3 * x2 = 12) → 
    m = 1 ∧ x1 = 0 ∧ x2 = 4 := by {
    sorry -- proof to be filled
  },
  exact condition_satisfied h
end

end quadratic_roots_specific_solution_l340_340259


namespace find_angle_B_l340_340366

-- Conditions
variable (A B C a b : ℝ)
variable (h1 : a = Real.sqrt 6)
variable (h2 : b = Real.sqrt 3)
variable (h3 : b + a * (Real.sin C - Real.cos C) = 0)

-- Target
theorem find_angle_B : B = Real.pi / 6 :=
sorry

end find_angle_B_l340_340366


namespace bus_stop_l340_340523

theorem bus_stop (M H : ℕ) 
  (h1 : H = 2 * (M - 15))
  (h2 : M - 15 = 5 * (H - 45)) :
  M = 40 ∧ H = 50 := 
sorry

end bus_stop_l340_340523


namespace relation_of_s1_s2_l340_340020

variables {A B C G : Type} [metric_space G]
variables (GA GB GC : ℝ) (AB BC CA : ℝ)

-- Assuming G is the centroid of triangle ABC
def is_centroid (G : Type) [metric_space G] (A B C : G) (G : G) : Prop :=
  (1/3) * (dist G A + dist G B + dist G C)

theorem relation_of_s1_s2
  (h1 : is_centroid A B C G)
  (h2 : GA + GB + GC = s_1)
  (h3 : AB + BC + CA = s_2) :
  (s_1 > (1/2) * s_2) ∧ (s_1 < s_2) :=
sorry

end relation_of_s1_s2_l340_340020


namespace area_of_triangle_ADE_l340_340691

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
let s := (a + b + c) / 2 in
Real.sqrt (s * (s - a) * (s - b) * (s - c))

noncomputable def sin_angle (a b area : ℝ) : ℝ :=
2 * area / (a * b)

noncomputable def area_triangle (a b sin_angle : ℝ) : ℝ :=
1 / 2 * a * b * sin_angle

theorem area_of_triangle_ADE (AB BC AC AD AE : ℝ) (h1 : AB = 10) (h2 : BC = 12) (h3 : AC = 13)
  (h4 : AD = 5) (h5 : AE = 8) : triangle_area AD AE (sin_angle 12 13 (triangle_area 10 12 13)) = 15.5 :=
by
  simp [h1, h2, h3, h4, h5]
  sorry

end area_of_triangle_ADE_l340_340691


namespace domain_of_f_l340_340485

noncomputable def f (t : ℝ) : ℝ :=  1 / ((abs (t - 1))^2 + (abs (t + 1))^2)

theorem domain_of_f : ∀ t : ℝ, (abs (t - 1))^2 + (abs (t + 1))^2 ≠ 0 :=
by
  intro t
  sorry

end domain_of_f_l340_340485


namespace number_of_possible_choices_leq_l340_340449

variable (r m : ℕ)

-- Representing the graph as a complete bipartite graph K_{r,r}
def complete_bipartite_graph (r : ℕ) : Type := {G : Graph // G.is_complete_bipartite r r }

-- Definition of flights in the graph. 
def flights (G : complete_bipartite_graph r) : ℕ := (G.val.edge_set).card

-- Predicate to check if we can choose a pair of non-intersecting groups of r cities each such that 
-- every city in the first group is connected to every city in the second group.
def valid_pair (G : complete_bipartite_graph r) (P Q : set G.val.V) : Prop :=
  P.card = r ∧ Q.card = r ∧ P ∩ Q = ∅ ∧ ∀ p ∈ P, ∀ q ∈ Q, (p, q) ∈ G.val.edge_set ∧ (q, p) ∈ G.val.edge_set

-- The theorem we need to prove
theorem number_of_possible_choices_leq (F : complete_bipartite_graph r) (h_flight_num : flights F = m) :
  ∃ k, k ≤ 2 * m^r :=
sorry


end number_of_possible_choices_leq_l340_340449


namespace side_length_eq_l340_340746

namespace EquilateralTriangle

variables (A B C P Q R S : Type) [HasVSub Type P] [MetricSpace P]
variables [HasDist P] [HasEquilateralTriangle ABC] [InsideTriangle P ABC]
variables [Perpendicular PQ AB] [Perpendicular PR BC] [Perpendicular PS CA]
variables [Distance PQ 2] [Distance PR 3] [Distance PS 4]

theorem side_length_eq : side_length ABC = 6 * √3 :=
sorry
end EquilateralTriangle

end side_length_eq_l340_340746


namespace smallest_number_is_21_5_l340_340544

-- Definitions of the numbers in their respective bases
def num1 := 3 * 4^0 + 3 * 4^1
def num2 := 0 + 1 * 2^1 + 1 * 2^2 + 1 * 2^3
def num3 := 2 * 3^0 + 2 * 3^1 + 1 * 3^2
def num4 := 1 * 5^0 + 2 * 5^1

-- Statement asserting that num4 is the smallest number
theorem smallest_number_is_21_5 : num4 < num1 ∧ num4 < num2 ∧ num4 < num3 := by
  sorry

end smallest_number_is_21_5_l340_340544


namespace range_of_a_l340_340609

theorem range_of_a (a_n : ℕ → ℝ) (d q : ℝ) (b_n : ℕ → ℝ) (a : ℝ)
  (h1 : ∀ n : ℕ, a_n n = a_n 0 + (n - 1) * d)
  (h2 : ∀ n : ℕ, b_n n = b_n 1 * q ^ (n - 1))
  (h3 : 1 < q) (h4 : 0 < d) (b1_pos : 0 < b_n 1)
  (ineq_cond : ∀ n : ℕ, 1 < n → a_n n - a_n 0 > log a (b_n n) - log a (b_n 1)) :
  a ∈ (Set.Ioo 0 1) ∪ Set.Ioi (q^(1/d)) :=
sorry

end range_of_a_l340_340609


namespace tan_beta_of_tan_alpha_and_tan_alpha_plus_beta_l340_340997

theorem tan_beta_of_tan_alpha_and_tan_alpha_plus_beta (α β : ℝ)
  (h1 : Real.tan α = 2)
  (h2 : Real.tan (α + β) = 1 / 5) :
  Real.tan β = -9 / 7 :=
sorry

end tan_beta_of_tan_alpha_and_tan_alpha_plus_beta_l340_340997


namespace more_spent_on_keychains_bracelets_than_tshirts_l340_340552

-- Define the conditions as variables
variable (spent_keychains_bracelets spent_total_spent : ℝ)
variable (spent_keychains_bracelets_eq : spent_keychains_bracelets = 347.00)
variable (spent_total_spent_eq : spent_total_spent = 548.00)

-- Using these conditions, define the problem to prove the desired result
theorem more_spent_on_keychains_bracelets_than_tshirts :
  spent_keychains_bracelets - (spent_total_spent - spent_keychains_bracelets) = 146.00 :=
by
  rw [spent_keychains_bracelets_eq, spent_total_spent_eq]
  sorry

end more_spent_on_keychains_bracelets_than_tshirts_l340_340552


namespace product_fraction_simplification_l340_340125

theorem product_fraction_simplification :
  (∏ n in Finset.range (100-3+1), 1 - (1 / (n + 3))) = (1 / 50) :=
by
  sorry

end product_fraction_simplification_l340_340125


namespace triangle_area_of_parallelogram_l340_340443

theorem triangle_area_of_parallelogram (area_parallelogram : ℝ) (h : area_parallelogram = 128) : 
  let area_triangle := 1 / 2 * area_parallelogram in
  area_triangle = 64 := 
by 
  rw [h]
  norm_num

end triangle_area_of_parallelogram_l340_340443


namespace fifth_term_is_correct_l340_340080

variables (x y : ℝ)

def a1 : ℝ := 2 * x + 3 * y
def a2 : ℝ := 2 * x - 3 * y
def a3 : ℝ := 2 * x * y
def a4 : ℝ := 2 * x / (3 * y)

noncomputable def common_difference : ℝ := -6 * y

-- Calculate the fifth term in the sequence
noncomputable def a5 : ℝ := a4 + common_difference

theorem fifth_term_is_correct :
  a1 - a2 = common_difference ∧
  a2 - a3 = common_difference ∧
  a3 - a4 = common_difference ∧
  a5 = 4.5 :=
by
  sorry

end fifth_term_is_correct_l340_340080


namespace moon_temp_difference_l340_340419

def temp_difference (T_day T_night : ℤ) : ℤ := T_day - T_night

theorem moon_temp_difference :
  temp_difference 127 (-183) = 310 :=
by
  sorry

end moon_temp_difference_l340_340419


namespace tan_of_a_pi_over_6_l340_340301

theorem tan_of_a_pi_over_6 (a : ℝ) (h : 3 ^ a = 9) : Real.tan (a * Real.pi / 6) = Real.sqrt 3 :=
by
  sorry

end tan_of_a_pi_over_6_l340_340301


namespace percentage_of_profit_is_20_l340_340901

def selling_price : ℤ := 1110
def cost_price : ℤ := 925

def profit (selling_price cost_price : ℤ) : ℤ := selling_price - cost_price

def percentage_of_profit (profit cost_price : ℤ) : ℚ :=
  (profit.toRat / cost_price.toRat) * 100

theorem percentage_of_profit_is_20 :
  percentage_of_profit (profit selling_price cost_price) cost_price = 20 := by
  sorry

end percentage_of_profit_is_20_l340_340901


namespace distance_from_origin_to_line_AB_l340_340633

-- Definitions of points A and B
def A := (0, 6)
def B := (-8, 0)

-- Function to calculate the distance from a point (x0, y0) to a line given by Ax + By + C = 0
def distance_from_point_to_line (x0 y0 A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / sqrt (A^2 + B^2)

-- The equation of line AB in standard form obtained from the points A and B is 3x - 4y + 24 = 0
-- We now prove the distance from origin (0,0) to this line is 24/5
theorem distance_from_origin_to_line_AB : distance_from_point_to_line 0 0 3 (-4) 24 = 24 / 5 :=
by 
  sorry

end distance_from_origin_to_line_AB_l340_340633


namespace cost_of_items_l340_340798

theorem cost_of_items (M R F : ℝ)
  (h1 : 10 * M = 24 * R) 
  (h2 : F = 2 * R) 
  (h3 : F = 20.50) : 
  4 * M + 3 * R + 5 * F = 231.65 := 
by
  sorry

end cost_of_items_l340_340798


namespace complex_fraction_simplification_l340_340554

theorem complex_fraction_simplification : 
  (complex.mk 2 2) / complex.i + (complex.mk 1 1) / (complex.mk 1 (-1)) = complex.mk 2 (-1) :=
by
  sorry

end complex_fraction_simplification_l340_340554


namespace marbles_per_friend_l340_340375

theorem marbles_per_friend (total_marbles friends : ℕ) (h1 : total_marbles = 5504) (h2 : friends = 64) :
  total_marbles / friends = 86 :=
by {
  -- Proof will be added here
  sorry
}

end marbles_per_friend_l340_340375


namespace max_value_of_expressions_l340_340245

theorem max_value_of_expressions :
  ∃ x ∈ ℝ, (∀ y ∈ ℝ, (2^y - 4^y) ≤ (2^x - 4^x)) ∧ (2^x - 4^x = 1 / 4) :=
by
  sorry

end max_value_of_expressions_l340_340245


namespace benny_total_hours_l340_340553

theorem benny_total_hours (hours_per_day : ℕ) (days : ℕ) (total_hours : ℕ) : 
(hours_per_day = 3) → (days = 6) → (total_hours = hours_per_day * days) → total_hours = 18 :=
by
  intros hpd d t
  rw [hpd, d] at *
  rw [t]
  exact rfl

end benny_total_hours_l340_340553


namespace find_principal_l340_340888

noncomputable def principal_amount_borrowed
    (SI : ℝ)
    (R : ℝ)
    (T : ℝ)
    (P : ℝ) : Prop := SI = (P * R * T) / 100

theorem find_principal
    (SI : ℝ)
    (R : ℝ)
    (T : ℝ)
    (P : ℝ)
    (h : SI = 5400) : principal_amount_borrowed SI R T P :=
by
  have hR: R = 12 := rfl
  have hT: T = 3 := rfl
  have hP: P = 15000 := rfl
  have hSI := hR.symm ▸ hT.symm ▸ hP.symm ▸ rfl
  exact h

end find_principal_l340_340888


namespace amount_of_juice_p_in_a_l340_340149

  def total_p : ℚ := 24
  def total_v : ℚ := 25
  def ratio_a : ℚ := 4 / 1
  def ratio_y : ℚ := 1 / 5

  theorem amount_of_juice_p_in_a :
    ∃ P_a : ℚ, ∃ V_a : ℚ, ∃ P_y : ℚ, ∃ V_y : ℚ,
      P_a / V_a = ratio_a ∧ P_y / V_y = ratio_y ∧
      P_a + P_y = total_p ∧ V_a + V_y = total_v ∧ P_a = 20 :=
  by
    sorry
  
end amount_of_juice_p_in_a_l340_340149


namespace area_of_triangle_F1PF2_is_one_l340_340710

noncomputable theory

open Real

-- Definitions and conditions
def is_foci_of_ellipse (F1 F2 : ℝ × ℝ) : Prop :=
  F1 = (-sqrt 3, 0) ∧ F2 = (sqrt 3, 0)

def on_ellipse (P : ℝ × ℝ) : Prop :=
  (P.1 ^ 2 / 4 + P.2 ^ 2 = 1)

def perpendicular (F1 P F2 : ℝ × ℝ) : Prop :=
  ∃ (a b : ℝ), a * (F2.1 - F1.1) + b * (F2.2 - F1.2) = 0 ∧
               a * (F2.1 - P.1) + b * (F2.2 - P.2) = 0

-- Main theorem statement
theorem area_of_triangle_F1PF2_is_one
  (F1 F2 P : ℝ × ℝ)
  (hFoci : is_foci_of_ellipse F1 F2)
  (hOnEllipse : on_ellipse P)
  (hPerpendicular : perpendicular F1 P F2) :
  let PF1 := sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2),
      PF2 := sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2)
  in (1/2) * PF1 * PF2 = 1 := 
sorry

end area_of_triangle_F1PF2_is_one_l340_340710


namespace same_calendar_1990_1996_l340_340861

-- Given conditions in the problem:
def year := ℕ
def isLeapYear (y: year) : Prop := (y % 4 = 0 ∧ y % 100 ≠ 0) ∨ (y % 400 = 0)
def sameCalendar (y1 y2: year) : Prop :=
  ∀ (d : ℕ), d % 365 = (d + (if isLeapYear y1 then 1 else 0) + (if isLeapYear (y1 + 1) then 2 else 0) + .. + (if isLeapYear (y2 - 1) then 1 else 0)) % 365

-- Conditions for the specific years:
def y1990 := 1990
def y1991 := 1991
def y1992 := 1992

lemma y1990_not_leap : ¬ isLeapYear y1990 := by
  simp [isLeapYear, y1990]
  exact and.intro (dec_trivial) (dec_trivial)

lemma y1992_is_leap : isLeapYear y1992 := by
  simp [isLeapYear, y1992]
  exact or.intro_left _ (and.intro (dec_trivial) (dec_trivial))

-- Mathematical equivalent proof problem statement:
theorem same_calendar_1990_1996 : sameCalendar 1990 1996 := sorry

end same_calendar_1990_1996_l340_340861


namespace ratio_of_green_to_blue_l340_340104

-- Definitions of the areas and the circles
noncomputable def red_area : ℝ := Real.pi * (1 : ℝ) ^ 2
noncomputable def middle_area : ℝ := Real.pi * (2 : ℝ) ^ 2
noncomputable def large_area: ℝ := Real.pi * (3 : ℝ) ^ 2

noncomputable def blue_area : ℝ := middle_area - red_area
noncomputable def green_area : ℝ := large_area - middle_area

-- The proof that the ratio of the green area to the blue area is 5/3
theorem ratio_of_green_to_blue : green_area / blue_area = 5 / 3 := by
  sorry

end ratio_of_green_to_blue_l340_340104


namespace carlos_biked_more_than_daniel_l340_340345

-- Definitions modeled from conditions
def distance_carlos : ℕ := 108
def distance_daniel : ℕ := 90
def time_hours : ℕ := 6

-- Lean statement to prove the difference in distance
theorem carlos_biked_more_than_daniel : distance_carlos - distance_daniel = 18 := 
  by 
    sorry

end carlos_biked_more_than_daniel_l340_340345


namespace garrett_total_spent_l340_340994

/-- Garrett bought 6 oatmeal raisin granola bars, each costing $1.25. -/
def oatmeal_bars_count : Nat := 6
def oatmeal_bars_cost_per_unit : ℝ := 1.25

/-- Garrett bought 8 peanut granola bars, each costing $1.50. -/
def peanut_bars_count : Nat := 8
def peanut_bars_cost_per_unit : ℝ := 1.50

/-- The total amount spent on granola bars is $19.50. -/
theorem garrett_total_spent : oatmeal_bars_count * oatmeal_bars_cost_per_unit + peanut_bars_count * peanut_bars_cost_per_unit = 19.50 :=
by
  sorry

end garrett_total_spent_l340_340994


namespace smallest_positive_b_factors_l340_340986

theorem smallest_positive_b_factors (b : ℤ) : 
  (∃ p q : ℤ, x^2 + b * x + 2016 = (x + p) * (x + q) ∧ p + q = b ∧ p * q = 2016 ∧ p > 0 ∧ q > 0) → b = 95 := 
by {
  sorry
}

end smallest_positive_b_factors_l340_340986


namespace acquaintances_unique_l340_340148

theorem acquaintances_unique (N : ℕ) : ∃ acquaintances : ℕ → ℕ, 
  (∀ i j k : ℕ, i < N → j < N → k < N → i ≠ j → j ≠ k → i ≠ k → 
    acquaintances i ≠ acquaintances j ∨ acquaintances j ≠ acquaintances k ∨ acquaintances i ≠ acquaintances k) :=
sorry

end acquaintances_unique_l340_340148


namespace smallest_natural_number_rearrange_l340_340984

theorem smallest_natural_number_rearrange (N : ℕ) (hN : N = 1089) :
  ∀ (digits1 digits2 : List ℕ), 
  digits1 = N.digits 
  → digits2 = (9 * N).digits 
  → digits2.eraseAll digits1 = []
  → digits1.eraseAll digits2 = [] :=
by
  sorry

end smallest_natural_number_rearrange_l340_340984


namespace exists_three_numbers_sum_to_zero_l340_340053

theorem exists_three_numbers_sum_to_zero (s : Finset ℤ) (h_card : s.card = 101) (h_abs : ∀ x ∈ s, |x| ≤ 99) :
  ∃ (a b c : ℤ), a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c = 0 :=
by {
  sorry
}

end exists_three_numbers_sum_to_zero_l340_340053


namespace final_position_of_T_l340_340087

structure Position where
  base : Vector2 -- Assuming Vector2 as a structure for 2D vectors
  stem : Vector2

noncomputable def rotate_180_clockwise (pos : Position) : Position :=
  { base := -pos.base, stem := -pos.stem }

noncomputable def reflect_x_axis (pos : Position) : Position :=
  { base := ⟨pos.base.x, -pos.base.y⟩, stem := ⟨pos.stem.x, -pos.stem.y⟩ }

noncomputable def rotate_90_clockwise (pos : Position) : Position :=
  { base := ⟨pos.base.y, -pos.base.x⟩, stem := ⟨pos.stem.y, -pos.stem.x⟩ }

theorem final_position_of_T :
  let initial_pos := Position.mk ⟨1, 0⟩ ⟨0, 1⟩ in
  let pos_after_180_rotation := rotate_180_clockwise initial_pos in
  let pos_after_reflection := reflect_x_axis pos_after_180_rotation in
  let final_pos := rotate_90_clockwise pos_after_reflection in
  final_pos.base = ⟨0, -1⟩ ∧ final_pos.stem = ⟨-1, 0⟩ :=
by
  intros
  sorry

end final_position_of_T_l340_340087


namespace probability_sum_gt_six_l340_340138

def set_s : set ℤ := {12, 34}

def possible_pairs (s : set ℤ) : set (ℤ × ℤ) :=
  { (x, y) | x ∈ s ∧ y ∈ s }

def sum_greater_than_six (p : ℤ × ℤ) : Prop :=
  p.1 + p.2 > 6

theorem probability_sum_gt_six : 
  ∀ (p : ℤ × ℤ), p ∈ possible_pairs set_s → sum_greater_than_six p :=
by {
  intros p hp,
  sorry
}

end probability_sum_gt_six_l340_340138


namespace minimal_bailing_rate_needed_l340_340203

-- Definitions of the conditions
def distance_to_shore : ℝ := 1.5
def rate_of_water_intake : ℝ := 8 -- gallons per minute
def boat_capacity_before_sinking : ℝ := 40 -- gallons
def rowing_speed : ℝ := 3 -- miles per hour

-- The question and the answer together form the proof goal
theorem minimal_bailing_rate_needed :
  ∃ r : ℝ, r = 7 ∧ -- gallons per minute
  let time_to_shore := (distance_to_shore / rowing_speed) * 60 in -- converted to minutes
  let total_water_intake := rate_of_water_intake * time_to_shore in
  let excess_water_to_bail := total_water_intake - boat_capacity_before_sinking in
  r >= (excess_water_to_bail / time_to_shore) :=
sorry

end minimal_bailing_rate_needed_l340_340203


namespace magnitude_of_inverse_z_square_of_conjugate_z_l340_340291

-- Define the complex number z
def z : ℂ := -1/2 + (sqrt 3)/2 * complex.I

-- Define the conjugate of z
def z_conj : ℂ := -1/2 - (sqrt 3)/2 * complex.I

-- Statement 1: Magnitude of 1/z
theorem magnitude_of_inverse_z : complex.abs (1/z) = 2 := sorry

-- Statement 2: Square of the conjugate of z
theorem square_of_conjugate_z : z_conj * z_conj = 1 - (sqrt 3) * complex.I := sorry

end magnitude_of_inverse_z_square_of_conjugate_z_l340_340291


namespace probability_of_condition_l340_340547

-- Define the list of integers
def list_of_integers : List ℤ := [-9, -7, -5, -3, -1, 1, 3, 5, 7, 9]

-- Define the condition that an integer's fourth power is greater than 100
def condition (m : ℤ) : Prop := m^4 > 100

-- Define the total count of integers in the list
def total_count : ℕ := list_of_integers.length

-- Define the count of integers satisfying the condition
def count_condition_satisfied : ℕ := list_of_integers.countp condition

-- The theorem to prove the probability
theorem probability_of_condition : ↑count_condition_satisfied / ↑total_count = (3 : ℚ) / 5 := by
  sorry

end probability_of_condition_l340_340547


namespace weight_of_hollow_golden_sphere_l340_340687

theorem weight_of_hollow_golden_sphere : 
  let diameter := 12
  let thickness := 0.3
  let pi := (3 : Real)
  let outer_radius := diameter / 2
  let inner_radius := (outer_radius - thickness)
  let outer_volume := (4 / 3) * pi * outer_radius^3
  let inner_volume := (4 / 3) * pi * inner_radius^3
  let gold_volume := outer_volume - inner_volume
  let weight_per_cubic_inch := 1
  let weight := gold_volume * weight_per_cubic_inch
  weight = 123.23 :=
by
  sorry

end weight_of_hollow_golden_sphere_l340_340687


namespace g_of_neg15_l340_340392

def f (x : ℝ) : ℝ := 4 * x - 7
def g (y : ℝ) : ℝ := 3 * (y / 4 + 7 / 4)^2 - 2 * (y / 4 + 7 / 4) + 1

theorem g_of_neg15 : g (-15) = 17 :=
by
  sorry

end g_of_neg15_l340_340392


namespace part1_i_part1_ii_part2_l340_340868

-- Definitions of sets A, B, and C
def A: Set ℤ := {x | |x| ≤ 6}
def B: Set ℕ := {1, 2, 3}
def C: Set ℕ := {3, 4, 5, 6}

-- Lean statement for part (1) question (i)
theorem part1_i: A ∩ (B ∩ C) = {3} :=
sorry

-- Lean definition for the set complement and theorem for part (1) question (ii)
def complement_A (S : Set ℤ) := {x | x ∈ A ∧ x ∉ S}

theorem part1_ii: A ∩ (complement_A (B ∪ C)) = { -6, -5, -4, -3, -2, -1, 0 } :=
sorry

-- Lean statement for part (2)
theorem part2: log 25 + log 2 * log 50 + (log 2)^2 = 2 :=
sorry

end part1_i_part1_ii_part2_l340_340868


namespace symmetric_point_y_axis_l340_340292

def M : ℝ × ℝ := (-5, 2)
def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)
theorem symmetric_point_y_axis :
  symmetric_point M = (5, 2) :=
by
  sorry

end symmetric_point_y_axis_l340_340292


namespace factorization_2210_l340_340653

theorem factorization_2210 : 
  (∃ a b : ℕ, 10 ≤ a ∧ a < 100 ∧ 100 ≤ b ∧ b < 1000 ∧ a * b = 2210) ∧ 
  (∀ c d : ℕ, 10 ≤ c ∧ c < 100 ∧ 100 ≤ d ∧ d < 1000 ∧ c * d = 2210 → (a = c ∧ b = d) ∨ (a = d ∧ b = c)) :=
by 
  have h2210 : 2210 = 2 * 5 * 13 * 17 := by norm_num,
  sorry

end factorization_2210_l340_340653


namespace min_value_is_1_over_6_l340_340617

noncomputable def min_value (x y z : ℝ) (h : x + 2 * y + z = 1) : ℝ :=
  x^2 + y^2 + z^2

theorem min_value_is_1_over_6 (x y z : ℝ) (h : x + 2 * y + z = 1) :
  min_value x y z h ≥ (1 / 6) :=
begin
  sorry
end

end min_value_is_1_over_6_l340_340617


namespace equilateral_triangle_side_length_l340_340762

theorem equilateral_triangle_side_length 
  {P Q R S : Point} 
  {A B C : Triangle}
  (h₁ : is_inside P A B C)
  (h₂ : is_perpendicular P Q A B)
  (h₃ : is_perpendicular P R B C)
  (h₄ : is_perpendicular P S C A)
  (h₅ : distance P Q = 2)
  (h₆ : distance P R = 3)
  (h₇ : distance P S = 4)
  : side_length A B C = 6 * sqrt 3 :=
by 
  sorry

end equilateral_triangle_side_length_l340_340762


namespace distance_between_points_l340_340115

/-- Define the distance function in 3D space -/
def distance (x1 y1 z1 x2 y2 z2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

/-- Define the specific points -/
def point1 : ℝ × ℝ × ℝ := (1, 3, -5)
def point2 : ℝ × ℝ × ℝ := (4, -2, 0)

/-- Prove that the distance between point1 and point2 is √59 -/
theorem distance_between_points :
  distance point1.1 point1.2 point1.3 point2.1 point2.2 point2.3 = real.sqrt 59 :=
by 
  sorry

end distance_between_points_l340_340115


namespace missing_dog_number_l340_340551

theorem missing_dog_number {S : Finset ℕ} (h₁ : S =  Finset.range 25 \ {24}) (h₂ : S.sum id = 276) :
  (∃ y ∈ S, y = (S.sum id - y) / (S.card - 1)) ↔ 24 ∉ S :=
by
  sorry

end missing_dog_number_l340_340551


namespace tax_diminishment_percentage_l340_340824

variable (T C : ℝ) -- Define original tax and consumption as real numbers
variable (X : ℝ) -- Define the percentage diminishment in tax as a real number

-- Conditions as definitions
def original_revenue := T * C
def new_tax_rate := T * (1 - X / 100)
def new_consumption := C * 1.15
def new_revenue := new_tax_rate * new_consumption

-- Effect on revenue is an 8% decrease
def effect_on_revenue := new_revenue = 0.92 * original_revenue

theorem tax_diminishment_percentage :
  effect_on_revenue T C X → X = 20 :=
by
  sorry

end tax_diminishment_percentage_l340_340824


namespace total_broken_marbles_l340_340263

theorem total_broken_marbles (marbles_set1 marbles_set2 : ℕ) 
  (percentage_broken_set1 percentage_broken_set2 : ℚ) 
  (h1 : marbles_set1 = 50) 
  (h2 : percentage_broken_set1 = 0.1) 
  (h3 : marbles_set2 = 60) 
  (h4 : percentage_broken_set2 = 0.2) : 
  (marbles_set1 * percentage_broken_set1 + marbles_set2 * percentage_broken_set2 = 17) := 
by 
  sorry

end total_broken_marbles_l340_340263


namespace cos_phi_eq_sin_alpha_sin_beta_l340_340359

variables (A B C D : Type) [InnerProductSpace ℝ A] 
variables (α β φ : ℝ)
variables (angle_cad : α = angle A C D)
variables (angle_cbd : β = angle C B D)
variables (angle_acb : φ = angle A C B)
variables (right_angle_cda : angle C D A = π / 2)
variables (right_angle_cdb : angle C D B = π / 2)
variables (right_angle_adb : angle A D B = π / 2)

theorem cos_phi_eq_sin_alpha_sin_beta :
  real.cos (φ) = real.sin (α) * real.sin (β) :=
sorry

end cos_phi_eq_sin_alpha_sin_beta_l340_340359


namespace six_digit_odd_number_count_l340_340993

-- Definition of the problem with the conditions
theorem six_digit_odd_number_count : 
  let digits := {0, 1, 3, 5, 7, 9}
  let count := 5 * 4 * Nat.factorial 4 
  count = 480 := 
by
  sorry

end six_digit_odd_number_count_l340_340993


namespace eccentricity_of_hyperbola_l340_340603

theorem eccentricity_of_hyperbola (a b : ℝ) (ha : a > 0) (hb: b > 0)
  (h: b = 2 * a) : 
  let e := (sqrt ((a^2 + b^2) / a^2)) in 
  e = sqrt 5 := 
by
  sorry

end eccentricity_of_hyperbola_l340_340603


namespace num_different_pairs_l340_340396

theorem num_different_pairs :
  (∃ (A B : Finset ℕ), A ∪ B = {1, 2, 3, 4} ∧ A ≠ B ∧ (A, B) ≠ (B, A)) ∧
  (∃ n : ℕ, n = 81) :=
by
  -- Proof would go here, but it's skipped per instructions
  sorry

end num_different_pairs_l340_340396


namespace triangle_CP_PA_ratio_l340_340362

theorem triangle_CP_PA_ratio (ABC : Triangle Point) (A B C D M P : Point) 
(h1 : distance A B = 15) 
(h2 : distance A C = 18) 
(h3 : angle_bisector A B C D) 
(h4 : midpoint M A D) 
(h5 : intersection P A C B M) : 
  let m := 11 in -- the ratio numerator
  let n := 5 in -- the ratio denominator
  m + n = 16 :=
by
  sorry

end triangle_CP_PA_ratio_l340_340362


namespace h_1_eq_2_l340_340724

def h (p : ℕ) : ℕ :=
  Nat.find (λ n : ℕ, ∀ m : ℕ, (n ≠ left_over m p) )
where left_over (m : ℕ) (p : ℕ) : ℕ := (2^m / 10^p)

theorem h_1_eq_2 : h 1 = 2 :=
sorry

end h_1_eq_2_l340_340724


namespace period_sin_cos_l340_340849

theorem period_sin_cos (x : ℝ) : ∀ y : ℝ, y = sin x + 2 * cos x → ∃ T > 0, ∀ t : ℝ, y = sin (x + t + T) + 2 * cos (x + t + T) := 
sorry

end period_sin_cos_l340_340849


namespace equilateral_triangle_side_length_l340_340765

theorem equilateral_triangle_side_length (P Q R S A B C : Type)
  [Point P] [Point Q] [Point R] [Point S] [Point A] [Point B] [Point C] :
  (within_triangle P A B C) →
  orthogonal_projection P Q A B →
  orthogonal_projection P R B C →
  orthogonal_projection P S C A →
  distance P Q = 2 →
  distance P R = 3 →
  distance P S = 4 →
  distance A B = 6 * √3 :=
by
  sorry

end equilateral_triangle_side_length_l340_340765


namespace value_of_x_l340_340852

theorem value_of_x : 
  let x := (2021^2 - 2021) / 2021 
  in x = 2020 :=
by
  let x := (2021^2 - 2021) / 2021
  have h : x = 2020 := sorry
  exact h

end value_of_x_l340_340852


namespace tangent_line_at_1_4_monotonic_intervals_l340_340314

-- Definition of the function
def f (x : ℝ) : ℝ := x^2 - 8 * Real.log x + 3

-- The statement proving the tangent line equation at the point (1, 4)
theorem tangent_line_at_1_4 : 
  ∃ m b : ℝ, (∀ x : ℝ, f(x) = m * x + b) ∧ 
            m = -6 ∧ b = 10 :=
by
  sorry

-- The statement proving the monotonicity intervals
theorem monotonic_intervals :
  (∀ x : ℝ, x > 2 → (f' x > 0)) ∧ (∀ x : ℝ, 0 < x ∧ x < 2 → (f' x < 0)) :=
by
  sorry

-- Define the derivative of the function
noncomputable def f' (x : ℝ) : ℝ := 2 * x - 8 / x

end tangent_line_at_1_4_monotonic_intervals_l340_340314


namespace solve_system_l340_340238

theorem solve_system (x₁ x₂ x₃ : ℝ) (h₁ : 2 * x₁^2 / (1 + x₁^2) = x₂) (h₂ : 2 * x₂^2 / (1 + x₂^2) = x₃) (h₃ : 2 * x₃^2 / (1 + x₃^2) = x₁) :
  (x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0) ∨ (x₁ = 1 ∧ x₂ = 1 ∧ x₃ = 1) :=
sorry

end solve_system_l340_340238


namespace trip_to_market_distance_l340_340101

theorem trip_to_market_distance 
  (school_trip_one_way : ℝ) (school_days_per_week : ℕ) 
  (weekly_total_mileage : ℝ) (round_trips_per_day : ℕ) (market_trip_count : ℕ) :
  (school_trip_one_way = 2.5) →
  (school_days_per_week = 4) →
  (round_trips_per_day = 2) →
  (weekly_total_mileage = 44) →
  (market_trip_count = 1) →
  let school_mileage := (school_trip_one_way * 2 * round_trips_per_day * school_days_per_week)
  let total_market_mileage := weekly_total_mileage - school_mileage
  let market_trip_distance := total_market_mileage / (2 * market_trip_count)
  market_trip_distance = 2 :=
by
  intros h1 h2 h3 h4 h5
  let school_mileage := (school_trip_one_way * 2 * round_trips_per_day * school_days_per_week)
  let total_market_mileage := weekly_total_mileage - school_mileage
  let market_trip_distance := total_market_mileage / (2 * market_trip_count)
  sorry

end trip_to_market_distance_l340_340101


namespace area_of_pentagon_BCFXE_l340_340472

noncomputable def area_pentagon (AB BC : ℝ) (AE_div_EB CF_div_FD : ℝ) : ℝ :=
  let A := (0 : ℝ, 0 : ℝ)
  let B := (12 : ℝ, 0 : ℝ)
  let C := (12 : ℝ, 7 : ℝ)
  let D := (0 : ℝ, 7 : ℝ)
  let E := ((A.1 + B.1) / 2, A.2)
  let F := (C.1, (2 / 3 * C.2))
  let AF_slope := (F.2 - A.2) / (F.1 - A.1)
  let DE_slope := (E.2 - D.2) / (E.1 - D.1)
  let x := (7 : ℝ) / (AF_slope + DE_slope)
  let y := AF_slope * x
  let X := (x, y)
  let area_rect := AB * BC
  let area_AEX := 1 / 2 * |E.1 - A.1| * |X.2|
  let area_DFX := 1 / 2 * |D.2 - F.2| * |X.1 - D.1|
  let area_ADF := 1 / 2 * |A.1 - D.1| * |F.2|
  area_rect - (area_AEX + area_DFX + area_ADF)

theorem area_of_pentagon_BCFXE :
  area_pentagon 12 7 1 (1 / 2) = 47 := by
  sorry

end area_of_pentagon_BCFXE_l340_340472


namespace length_of_bridge_l340_340540

theorem length_of_bridge
  (T : ℕ) (t : ℕ) (s : ℕ)
  (hT : T = 250)
  (ht : t = 20)
  (hs : s = 20) :
  ∃ L : ℕ, L = 150 :=
by
  sorry

end length_of_bridge_l340_340540


namespace geometric_seq_product_l340_340601

theorem geometric_seq_product (a : ℕ → ℝ) (r : ℝ) (h : ∀ n, a (n + 1) = r * a n) 
  (h_log : log 2 (a 2 * a 98) = 4) : a 40 * a 60 = 16 := 
by
  sorry

end geometric_seq_product_l340_340601


namespace age_of_new_employee_l340_340071

/-- Given 13 employees with an average age of 35 years, and a new employee is hired making the total number of employees 14 with an average age of 34 years, the age of the new employee is 21 years. -/
theorem age_of_new_employee (total_before : ℕ) (total_after : ℕ) (average_before : ℕ) (average_after : ℕ) (new_employee_count : ℕ) : unit :=
  let employees_before := 13
  let employees_after := 14
  let age_before := employees_before * average_before
  let age_after := employees_after * average_after
  total_before = age_before →
  total_after = age_after →
  new_employee_count = (total_after - total_before) →
  new_employee_count = 21
  by
  sorry

end age_of_new_employee_l340_340071


namespace inequality_proof_l340_340424

theorem inequality_proof (x y : ℝ) (hx_pos : x > 0) (hy_pos : y > 0) (hxy : x < y) : 
  x + real.sqrt (y^2 + 2) < y + real.sqrt (x^2 + 2) :=
sorry

end inequality_proof_l340_340424


namespace factors_of_210_l340_340569

theorem factors_of_210 : 
    let n := 210 in
    let factors := [7, 2, 3, 5] in
    n = factors.product 
    → ∃ F, F = (list.length factors).succ * (list.length factors).succ * (list.length factors).succ * (list.length factors).succ ∧ F = 16 :=
by
  -- complete this part with correct steps if necessary
  sorry

end factors_of_210_l340_340569


namespace part1_part2_l340_340313

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  cos (2 * x) + a * sin x + b

noncomputable def g (x : ℝ) (m : ℝ) : ℝ :=
  m * sin x + 2 * m

theorem part1 (h₀ : ∀ x, f x a b ≤ 9 / 8) (h₁ : ∀ x, f x a b ≥ -2) (ha_neg : a < 0) :
  a = -1 ∧ b = 0 :=
sorry

theorem part2 (h₀ : a = -2) (h₁ : b = 1) (h₂ : ∀ x ∈ Icc (π / 6) (2 * π / 3), f x a b > g x m) :
  m < -2 / 3 :=
sorry

end part1_part2_l340_340313


namespace slope_of_tangent_line_at_point_is_negative_reciprocal_of_slope_of_radius_l340_340120

def slope (p1 p2 : ℝ × ℝ) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)

theorem slope_of_tangent_line_at_point_is_negative_reciprocal_of_slope_of_radius :
  ∀ (point center : ℝ × ℝ), center = (2,1) ∧ point = (6,3) →
  let radius_slope := slope center point in
  let tangent_slope := - (1 / radius_slope) in
  tangent_slope = -2 :=
by intros point center h
   let radius_slope := slope center point
   let tangent_slope := - (1 / radius_slope)
   cases h
   sorry

end slope_of_tangent_line_at_point_is_negative_reciprocal_of_slope_of_radius_l340_340120


namespace triangle_angle_and_perimeter_l340_340364

/-
In a triangle ABC, given c * sin B = sqrt 3 * cos C,
prove that angle C equals pi / 3,
and given a + b = 6, find the minimum perimeter of triangle ABC.
-/
theorem triangle_angle_and_perimeter (A B C : ℝ) (a b c : ℝ) 
  (h1 : c * Real.sin B = Real.sqrt 3 * Real.cos C)
  (h2 : a + b = 6) :
  C = Real.pi / 3 ∧ a + b + (Real.sqrt (36 - a * b)) = 9 :=
by
  sorry

end triangle_angle_and_perimeter_l340_340364


namespace distances_equal_l340_340018

-- Defining the conditions
variables {A B C D M P Q R S : Type*}

-- Assume A, B, C, D are points on the circle
axiom points_on_circle : circle A B C D

-- Assume AC and BD intersect at point M inside the circle
axiom intersect_M : intersect AC BD = M

-- Assume M to be inside the circle
axiom M_inside_circle : inside_circle M

-- Assume perpendiculars dropped from M to lines AB, BC, CD, DA
axiom perp_MP_AB : perpendicular M P AB
axiom perp_MQ_BC : perpendicular M Q BC
axiom perp_MR_CD : perpendicular M R CD
axiom perp_MS_DA : perpendicular M S DA

-- The feet of these perpendiculars are P, Q, R, S respectively
axiom foot_MP : foot M P AB
axiom foot_MQ : foot M Q BC
axiom foot_MR : foot M R CD
axiom foot_MS : foot M S DA

-- The proof problem statement
theorem distances_equal :
  distances_from M to (lines PQ QR RS SP) = equal :=
sorry

end distances_equal_l340_340018


namespace ranking_of_girls_l340_340346

variables (Cassie Bridget Hannah Ella : ℝ)

def proof_problem : Prop :=
  (Cassie > Ella ∨ Cassie > Hannah) ∧
  (Bridget > Ella ∨ Bridget > Hannah) ∧
  (∃ y, y ∈ {Cassie, Bridget, Hannah} ∧ y > Ella) →
  (Cassie > Bridget ∧ Bridget > Ella ∧ Ella > Hannah)

theorem ranking_of_girls :
  proof_problem Cassie Bridget Hannah Ella :=
sorry

end ranking_of_girls_l340_340346


namespace melissa_gave_x_books_l340_340735

-- Define the initial conditions as constants
def initial_melissa_books : ℝ := 123
def initial_jordan_books : ℝ := 27
def final_melissa_books (x : ℝ) : ℝ := initial_melissa_books - x
def final_jordan_books (x : ℝ) : ℝ := initial_jordan_books + x

-- The main theorem to prove how many books Melissa gave to Jordan
theorem melissa_gave_x_books : ∃ x : ℝ, final_melissa_books x = 3 * final_jordan_books x ∧ x = 10.5 :=
sorry

end melissa_gave_x_books_l340_340735


namespace value_of_x2_minus_y2_l340_340336

theorem value_of_x2_minus_y2 (x y : ℚ) (h1 : x + y = 9 / 17) (h2 : x - y = 1 / 19) : x^2 - y^2 = 9 / 323 :=
by
  -- the proof would go here
  sorry

end value_of_x2_minus_y2_l340_340336


namespace Gretchen_walking_time_l340_340642

theorem Gretchen_walking_time
  (hours_worked : ℕ)
  (minutes_per_hour : ℕ)
  (sit_per_walk : ℕ)
  (walk_per_brake : ℕ)
  (h1 : hours_worked = 6)
  (h2 : minutes_per_hour = 60)
  (h3 : sit_per_walk = 90)
  (h4 : walk_per_brake = 10) :
  let total_minutes := hours_worked * minutes_per_hour,
      breaks := total_minutes / sit_per_walk,
      walking_time := breaks * walk_per_brake in
  walking_time = 40 := 
by
  sorry

end Gretchen_walking_time_l340_340642


namespace period_sin_cos_l340_340850

theorem period_sin_cos (x : ℝ) : ∀ y : ℝ, y = sin x + 2 * cos x → ∃ T > 0, ∀ t : ℝ, y = sin (x + t + T) + 2 * cos (x + t + T) := 
sorry

end period_sin_cos_l340_340850


namespace stratified_sampling_seniors_l340_340524

theorem stratified_sampling_seniors (total_students freshmen male_sophomores : ℕ)
  (prob_female_sophomore : ℝ) (n : ℕ) :
  total_students = 1000 →
  freshmen = 380 →
  male_sophomores = 180 →
  prob_female_sophomore = 0.19 →
  n = 100 →
  let female_sophomores := total_students * prob_female_sophomore in
  let sophomores := male_sophomores + female_sophomores in
  let seniors := total_students - sophomores - freshmen in
  seniors * n / total_students = 25 :=
by
  intros h1 h2 h3 h4 h5
  let female_sophomores := total_students * prob_female_sophomore
  let sophomores := male_sophomores + female_sophomores
  let seniors := total_students - sophomores - freshmen
  have h6 : seniors * n / total_students = 25 := sorry
  exact h6

end stratified_sampling_seniors_l340_340524


namespace smallest_m_for_reflection_l340_340733

noncomputable def theta : Real := Real.arctan (1 / 3)
noncomputable def pi_8 : Real := Real.pi / 8
noncomputable def pi_12 : Real := Real.pi / 12
noncomputable def pi_4 : Real := Real.pi / 4
noncomputable def pi_6 : Real := Real.pi / 6

/-- The smallest positive integer m such that R^(m)(l) = l
where the transformation R(l) is described as:
l is reflected in l1 (angle pi/8), then the resulting line is
reflected in l2 (angle pi/12) -/
theorem smallest_m_for_reflection :
  ∃ (m : ℕ), m > 0 ∧ ∀ (k : ℤ), m = 12 * k + 12 := by
sorry

end smallest_m_for_reflection_l340_340733


namespace minimum_area_of_triangle_l340_340143

def point := (ℤ × ℤ)

def A : point := (0, 0)
def B : point := (24, 18)

def area (A B C : point) : ℚ :=
  1/2 * |(B.1 * C.2 - B.2 * C.1 : ℤ)|

theorem minimum_area_of_triangle (C : point) (h1 : C.1 ∈ ℤ) (h2 : C.2 ∈ ℤ) :
  ∃ p q : ℤ, area A B (p, q) = 3 :=
sorry

end minimum_area_of_triangle_l340_340143


namespace infinite_possible_values_for_c_l340_340672

theorem infinite_possible_values_for_c (a b c : ℤ) (h1 : a * b + c = 40) (h2 : a + b ≠ 18) : ∃∞ c, a * b + c = 40 ∧ a + b ≠ 18 :=
sorry

end infinite_possible_values_for_c_l340_340672


namespace largest_non_sum_of_multiple_of_30_and_composite_l340_340840

theorem largest_non_sum_of_multiple_of_30_and_composite :
  ∃ (n : ℕ), n = 211 ∧ ∀ a b : ℕ, (a > 0) → (b > 0) → (b < 30) → 
  n ≠ 30 * a + b ∧ ¬ ∃ k : ℕ, k > 1 ∧ k < b ∧ b % k = 0 :=
sorry

end largest_non_sum_of_multiple_of_30_and_composite_l340_340840


namespace chebyshev_inequality_l340_340713

variables {n : ℕ} {ε : ℝ} (ξ : Fin n → ℝ) (C : Matrix (Fin n) (Fin n) ℝ)

-- Assume the expectation of ξ is zero
-- Assume that C is the covariance matrix which is nonsingular
def CovarianceMatrix (ξ : Fin n → ℝ) : Matrix (Fin n) (Fin n) ℝ := sorry
def IsNonsingular (C : Matrix (Fin n) (Fin n) ℝ) : Prop := sorry

noncomputable def Expec (ξ : Fin n → ℝ) : Fin n → ℝ := λ i, 0

-- Chebyshev's Inequality statement in Lean
theorem chebyshev_inequality (hC : IsNonsingular C) (hC_def : CovarianceMatrix ξ = C) (hε : ε > 0) :
  P (λ ξ, ((ξ - Expec ξ)ᵀ ⬝ C⁻¹ ⬝ (ξ - Expec ξ) > ε)) ≤ n / ε :=
sorry

end chebyshev_inequality_l340_340713


namespace max_value_of_a_l340_340333

theorem max_value_of_a (a : ℝ) : (∀ x : ℝ, x^2 - 2*x - a ≥ 0) → a ≤ -1 := 
by 
  sorry

end max_value_of_a_l340_340333


namespace intersection_eq_l340_340635

def setA : Set ℝ := { y | ∃ x : ℝ, y = 2^x }
def setB : Set ℝ := { y | ∃ x : ℝ, y = x^2 }
def expectedIntersection : Set ℝ := { y | 0 < y }

theorem intersection_eq :
  setA ∩ setB = expectedIntersection :=
sorry

end intersection_eq_l340_340635


namespace freshman_count_630_l340_340816

-- Define the variables and conditions
variables (f o j s : ℕ)
variable (total_students : ℕ)

-- Define the ratios given in the problem
def freshmen_to_sophomore : Prop := f = (5 * o) / 4
def sophomore_to_junior : Prop := j = (8 * o) / 7
def junior_to_senior : Prop := s = (7 * j) / 9

-- Total number of students condition
def total_students_condition : Prop := f + o + j + s = total_students

theorem freshman_count_630
  (h1 : freshmen_to_sophomore f o)
  (h2 : sophomore_to_junior o j)
  (h3 : junior_to_senior j s)
  (h4 : total_students_condition f o j s 2158) :
  f = 630 :=
sorry

end freshman_count_630_l340_340816


namespace divisors_of_sums_eq_coprimes_l340_340030

noncomputable def M (a : ℕ) : Set ℕ :=
{m : ℕ | ∃ (n : ℕ), m ∣ (∑ k in Finset.range (n + 1), a ^ k)}

theorem divisors_of_sums_eq_coprimes (a : ℕ) (h : a > 1) :
  M a = {m : ℕ | Nat.gcd m a = 1} := sorry

end divisors_of_sums_eq_coprimes_l340_340030


namespace emma_room_width_l340_340233

noncomputable def width_of_room (w : ℕ) : Prop :=
  let area_of_room := w * 20 in
  let area_covered_by_tiles := 40 in
  let fraction_of_room_covered := area_covered_by_tiles * 6 = area_of_room in
  fraction_of_room_covered

theorem emma_room_width : ∃ (w : ℕ), width_of_room w ∧ w = 12 :=
by
  use 12
  unfold width_of_room
  simp
  sorry

end emma_room_width_l340_340233


namespace increase_in_expenditure_l340_340538

-- Given conditions
def annual_income := ℝ -- x in ten thousand yuan
def annual_food_expenditure := ℝ -- y in ten thousand yuan
def regression_line (x : annual_income) : annual_food_expenditure := 0.254 * x + 0.321

-- Proof statement
theorem increase_in_expenditure (x : annual_income) :
  regression_line (x + 1) - regression_line x = 0.254 :=
by sorry

end increase_in_expenditure_l340_340538


namespace evaluate_expression_l340_340234

theorem evaluate_expression : (10^(-1) * 3^(0)) / (2 * 10^(-2)) = 5 := 
sorry

end evaluate_expression_l340_340234


namespace midpoint_sum_coordinates_l340_340810

theorem midpoint_sum_coordinates (x y : ℝ) 
  (midpoint_cond_x : (x + 10) / 2 = 4) 
  (midpoint_cond_y : (y + 4) / 2 = -8) : 
  x + y = -22 :=
by
  sorry

end midpoint_sum_coordinates_l340_340810


namespace solve_for_y_l340_340787

theorem solve_for_y 
  (y: ℝ) 
  (h: (8*y^2 + 50*y + 5) / (3*y + 21) = 4*y + 3) :
  y = (-43 + real.sqrt 921) / 8 ∨ y = (-43 - real.sqrt 921) / 8 :=
sorry

end solve_for_y_l340_340787


namespace sum_first_9_terms_l340_340819

-- Definitions of the arithmetic sequence and sum.
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n m : ℕ, a (n + m) = a n + a m

def sum_first_n (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = n * (a 1 + a n) / 2

-- Conditions
def a_n (n : ℕ) : ℤ := sorry -- we assume this function gives the n-th term of the arithmetic sequence
def S_n (n : ℕ) : ℤ := sorry -- sum of first n terms
axiom a_5_eq_2 : a_n 5 = 2
axiom arithmetic_sequence_proof : arithmetic_sequence a_n
axiom sum_first_n_proof : sum_first_n a_n S_n

-- Statement to prove
theorem sum_first_9_terms : S_n 9 = 18 :=
by
  sorry

end sum_first_9_terms_l340_340819


namespace median_of_dataset_l340_340898

theorem median_of_dataset (a : ℝ) (h_avg : (a + 5 + 6 + 7 + 7 + 8 + 11 + 12) / 8 = 8) :
  let dataset := [a, 5, 6, 7, 7, 8, 11, 12].qsort (· < ·)
  dataset.nth (dataset.length / 2 - 1).get_or_else 0 + dataset.nth (dataset.length / 2).get_or_else 0 = 15 :=
by
  sorry

end median_of_dataset_l340_340898


namespace part1_part2_l340_340269

def a (x : ℝ) : ℝ × ℝ := (sqrt 3 * Real.sin x, 1)
def b (x : ℝ) : ℝ × ℝ := (1, -Real.cos x)
def f (x : ℝ) : ℝ := sqrt 3 * Real.sin x - Real.cos x

theorem part1 (x : ℝ) (h : x = 0) : a x.1 * b x.1 + a x.2 * b x.2 = -1 := by
  sorry

theorem part2 (x : ℝ) (h : f x = 2 * Real.sin (x - Real.pi / 6)) : 
  ∃ k : ℤ, (x ∈ ([-Real.pi / 3 + 2 * k * Real.pi, 2 * Real.pi / 3 + 2 * k * Real.pi] : set ℝ)) := by
  sorry

end part1_part2_l340_340269


namespace length_of_bridge_l340_340905

-- Condition Definitions
def train_length : ℝ := 180  -- in meters
def train_speed_kmh : ℝ := 60  -- in km/hr
def crossing_time : ℕ := 45  -- in seconds

-- Conversion from km/hr to m/s
def train_speed_ms := (train_speed_kmh * 1000) / 3600 -- in m/s

-- Total distance covered by the train while crossing the bridge
def total_distance := train_speed_ms * crossing_time

-- Proof that the length of the bridge is 570.15 meters
theorem length_of_bridge : total_distance = train_length + 570.15 := by
  sorry

end length_of_bridge_l340_340905


namespace sprinter_time_no_wind_l340_340190

theorem sprinter_time_no_wind :
  ∀ (x y : ℝ), (90 / (x + y) = 10) → (70 / (x - y) = 10) → x = 8 * y → 100 / x = 12.5 :=
by
  intros x y h1 h2 h3
  sorry

end sprinter_time_no_wind_l340_340190


namespace smallest_range_seven_observations_l340_340899

theorem smallest_range_seven_observations (observations : Fin 7 → ℝ)
    (h_sum : ∑ i, observations i = 105)
    (h_median : observations 3 = 17) :
  ∃ (min_observation max_observation : ℝ), (min_observation = min_value observations) ∧ 
                                           (max_observation = max_value observations) ∧
                                           (max_observation - min_observation = 11) :=
by
  sorry

end smallest_range_seven_observations_l340_340899


namespace problem_l340_340023

noncomputable def f (x : ℝ) (a b c : ℝ) := a * x ^ 3 + b * x + c
noncomputable def g (x : ℝ) (d e f : ℝ) := d * x ^ 3 + e * x + f

theorem problem (a b c d e f : ℝ) :
    (∀ x : ℝ, f (g x d e f) a b c = g (f x a b c) d e f) ↔ d = a ∨ d = -a :=
by
  sorry

end problem_l340_340023


namespace limit_of_sequence_l340_340833

theorem limit_of_sequence {ε : ℝ} (hε : ε > 0) : 
  ∃ (N : ℝ), ∀ (n : ℝ), n > N → |(2 * n^3) / (n^3 - 2) - 2| < ε :=
by
  sorry

end limit_of_sequence_l340_340833


namespace area_of_circle_l340_340951

noncomputable def area_of_region : ℝ := 
  let eq := λ (x y : ℝ), x^2 + y^2 - 4 * x + 6 * y = -9
  in 4 * real.pi

theorem area_of_circle :
  (let is_circle := ∀ x y, x^2 + y^2 - 4 * x + 6 * y = -9 -> (x - 2)^2 + (y + 3)^2 = 4 in
  is_circle → area_of_region = 4 * real.pi) :=
by
  let is_circle := ∀ x y, x^2 + y^2 - 4 * x + 6 * y = -9 -> (x - 2)^2 + (y + 3)^2 = 4
  exact λ _ => sorry

end area_of_circle_l340_340951


namespace doughnuts_left_l340_340874

theorem doughnuts_left (dozen : ℕ) (total_initial : ℕ) (eaten : ℕ) (initial : total_initial = 2 * dozen) (d : dozen = 12) : total_initial - eaten = 16 :=
by
  rcases d
  rcases initial
  sorry

end doughnuts_left_l340_340874


namespace math_test_score_l340_340073

theorem math_test_score (K E M : ℕ) 
  (h₁ : (K + E) / 2 = 92) 
  (h₂ : (K + E + M) / 3 = 94) : 
  M = 98 := 
by 
  sorry

end math_test_score_l340_340073


namespace f_sum_lt_zero_l340_340393

-- Define a monotonically decreasing odd function f
def is_monotonically_decreasing_odd (f : ℝ → ℝ) : Prop := 
  (∀ x y : ℝ, x < y → f(y) < f(x)) ∧ (∀ x : ℝ, f(-x) = -f(x))

-- Define the conditions
variables (f : ℝ → ℝ) (x1 x2 x3 : ℝ)
hypothesis (h1 : x1 + x2 > 0) (h2 : x2 + x3 > 0) (h3 : x3 + x1 > 0)
hypothesis (h_f : is_monotonically_decreasing_odd f)

-- Prove the main statement
theorem f_sum_lt_zero : f(x1) + f(x2) + f(x3) < 0 :=
by 
  sorry

end f_sum_lt_zero_l340_340393


namespace leap_years_in_105_years_l340_340918

theorem leap_years_in_105_years : 
  (λ n, (⌊n / 4⌋ : ℕ) + (⌊n / 5⌋ : ℕ) - (⌊n / 20⌋ : ℕ)) 105 = 42 :=
by
  sorry

end leap_years_in_105_years_l340_340918


namespace initial_amount_100000_l340_340242

noncomputable def compound_interest_amount (P r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def future_value (P CI : ℝ) : ℝ :=
  P + CI

theorem initial_amount_100000
  (CI : ℝ) (P : ℝ) (r : ℝ) (n t : ℕ) 
  (h1 : CI = 8243.216)
  (h2 : r = 0.04)
  (h3 : n = 2)
  (h4 : t = 2)
  (h5 : future_value P CI = compound_interest_amount P r n t) :
  P = 100000 :=
by
  sorry

end initial_amount_100000_l340_340242


namespace sequence_bounded_iff_perfect_cube_l340_340659

def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k^3 = n

def largest_cube_le (k : ℕ) : ℕ := (Nat.cbrt k)^3

def sequence_bounded (p : ℕ) : Prop :=
  let a : ℕ → ℕ := λ n, Nat.recOn n p (λ _ an, 3 * an - 2 * largest_cube_le an)
  -- A sequence is bounded if there exists an M such that for all n, a_n ≤ M
  ∃ M : ℕ, ∀ n : ℕ, a n ≤ M

theorem sequence_bounded_iff_perfect_cube (p : ℕ) : sequence_bounded p ↔ is_perfect_cube p := by
  sorry

end sequence_bounded_iff_perfect_cube_l340_340659


namespace even_function_when_a_zero_range_of_a_for_domain_real_l340_340999

def f (a x : ℝ) : ℝ := Real.log (x^2 + 2 * a * x + a^2 + a + 1)

-- Problem (1)
theorem even_function_when_a_zero : 
  (∀ x : ℝ, f 0 x = f 0 (-x)) := by sorry

-- Problem (2)
theorem range_of_a_for_domain_real : 
  (∀ x : ℝ, 0 < x^2 + 2 * a * x + a^2 + a + 1) ↔ a > -1 := by sorry

end even_function_when_a_zero_range_of_a_for_domain_real_l340_340999


namespace find_n_l340_340282

theorem find_n (n : ℕ) (hn_pos : 0 < n) (hn_greater_30 : 30 < n) 
  (divides : (4 * n - 1) ∣ 2002 * n) : n = 36 := 
by
  sorry

end find_n_l340_340282


namespace sin_225_cos_225_l340_340933

noncomputable def sin_225_eq_neg_sqrt2_div_2 : Prop :=
  Real.sin (225 * Real.pi / 180) = -Real.sqrt 2 / 2

noncomputable def cos_225_eq_neg_sqrt2_div_2 : Prop :=
  Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2

theorem sin_225 : sin_225_eq_neg_sqrt2_div_2 := by
  sorry

theorem cos_225 : cos_225_eq_neg_sqrt2_div_2 := by
  sorry

end sin_225_cos_225_l340_340933


namespace average_closer_to_larger_set_l340_340825

/-- 
  Given two sets of numbers with their combined average being 80 and
  one set has more numbers than the other, the average is closer
  to the set with more numbers.
-/
theorem average_closer_to_larger_set 
  (S T : Set ℝ) 
  (avg : (Set.sum S + Set.sum T) / (S.card + T.card) = 80) 
  (h : S.card > T.card) 
  : avg = 80 → are_closer_to (S ∪ T) (S) := 
sorry

/-- 
  Helper definition to express that one set's average is closer 
  to a given set
-/
def are_closer_to (combined_set : Set ℝ) (target_set : Set ℝ) : Prop :=
  let combined_avg := (Set.sum combined_set) / (Set.card combined_set)
  let target_avg := (Set.sum target_set) / (Set.card target_set)
  |combined_avg - target_avg| < |combined_avg - (Set.sum (combined_set \ target_set)) / ((Set.card combined_set) - (Set.card target_set))|

end average_closer_to_larger_set_l340_340825


namespace divide_triangle_area_l340_340278

theorem divide_triangle_area (A B C P : Point) (P_in_triangle : PointInTriangle P A B C) : 
  ∃ Q : Point, PointOnPerimeter Q A B C ∧ Area (Triangle A P Q) + Area (Triangle P Q B) ≠ Area (Triangle P Q C) :=
sorry

end divide_triangle_area_l340_340278


namespace arithmetic_mean_correct_l340_340321

noncomputable def arithmetic_mean (n : ℕ) (h : n > 1) : ℝ :=
  let one_minus_one_div_n := 1 - (1 / n : ℝ)
  let rest_ones := (n - 1 : ℕ) • 1
  let total_sum : ℝ := rest_ones + one_minus_one_div_n
  total_sum / n

theorem arithmetic_mean_correct (n : ℕ) (h : n > 1) :
  arithmetic_mean n h = 1 - (1 / (n * n : ℝ)) := sorry

end arithmetic_mean_correct_l340_340321


namespace upstream_travel_time_l340_340517

def boat_speed_still_water : ℝ := 12 -- kmph
def downstream_distance : ℝ := 42 -- km
def downstream_time : ℝ := 3 -- hours
def downstream_current : ℝ := 2 -- kmph

def first_third_distance : ℝ := downstream_distance / 3 -- km
def second_third_distance : ℝ := downstream_distance / 3 -- km
def final_third_distance : ℝ := downstream_distance / 3 -- km

def first_third_current : ℝ := 4 -- kmph
def second_third_current : ℝ := 2 -- kmph
def final_third_current : ℝ := 3 -- kmph

def stops : ℕ := 2
def stop_time : ℝ := 15 / 60 -- hours

theorem upstream_travel_time :
  let first_third_speed := boat_speed_still_water - first_third_current in
  let second_third_speed := boat_speed_still_water - second_third_current in
  let final_third_speed := boat_speed_still_water - final_third_current in
  let first_third_time := first_third_distance / first_third_speed in
  let second_third_time := second_third_distance / second_third_speed in
  let final_third_time := final_third_distance / final_third_speed in
  let total_travel_time := first_third_time + second_third_time + final_third_time in
  let total_stop_time := stops * stop_time in
  total_travel_time + total_stop_time = 5.2056 :=
sorry

end upstream_travel_time_l340_340517


namespace amplitude_of_f_phase_shift_of_f_l340_340957

noncomputable def f (x : ℝ) : ℝ :=
  -5 * Real.sin (x + (Real.pi / 3))

theorem amplitude_of_f : ∀ x : ℝ, |f(x)| ≤ 5 := by
  sorry

theorem phase_shift_of_f : ∃ φ : ℝ, ∀ x : ℝ, f (x - φ) = -5 * Real.sin x ∧ φ = Real.pi / 3 := by
  sorry

end amplitude_of_f_phase_shift_of_f_l340_340957


namespace sum_of_sequence_S_2011_l340_340607

-- Define the sequence aₙ
def a : ℕ → ℕ
| 1 := 1
| 2 := 2
| n+(n+(n+1)) := if (a n * a (n+1) * a (n+2) = a n + a (n+1) + a (n+2) ∧ a (n+1) * a (n+2) ≠ 1) 
                  then -- Define based on conditions 
                  3
                  else if _ _ _ then -- further logic assuming periodicity
                      sorry -- logic for extended terms based on periodicity
                  else sorry -- otherwise sorry

-- Define the sum Sₙ
noncomputable def S : ℕ → ℕ
| 0 := 0
| n+1 := S n + a (n+1)

-- State the problem as a theorem
theorem sum_of_sequence_S_2011 : S 2011 = 4021 :=
begin
  sorry
end

end sum_of_sequence_S_2011_l340_340607


namespace extended_cross_cannot_form_cube_l340_340135

-- Define what it means to form a cube from patterns
def forms_cube (pattern : Type) : Prop := 
  sorry -- Definition for forming a cube would be detailed here

-- Define the Extended Cross pattern in a way that captures its structure
def extended_cross : Type := sorry -- Definition for Extended Cross structure

-- Define the L shape pattern in a way that captures its structure
def l_shape : Type := sorry -- Definition for L shape structure

-- The theorem statement proving that the Extended Cross pattern cannot form a cube
theorem extended_cross_cannot_form_cube : ¬(forms_cube extended_cross) := 
  sorry

end extended_cross_cannot_form_cube_l340_340135


namespace find_k_l340_340394

noncomputable def series_sum (k : ℝ) : ℝ :=
  ∑' n, (7 * n + 2) / k^n

theorem find_k (k : ℝ) (h : k > 1) (hk : series_sum k = 5) : 
  k = (7 + Real.sqrt 14) / 5 :=
by 
  sorry

end find_k_l340_340394


namespace coplanar_k_values_l340_340048

noncomputable def coplanar_lines_possible_k (k : ℝ) : Prop :=
  ∃ (t u : ℝ), (2 + t = 1 + k * u) ∧ (3 + t = 4 + 2 * u) ∧ (4 - k * t = 5 + u)

theorem coplanar_k_values :
  ∀ k : ℝ, coplanar_lines_possible_k k ↔ (k = 0 ∨ k = -3) :=
by
  sorry

end coplanar_k_values_l340_340048


namespace triangle_area_AC_1_AD_BC_circumcircle_l340_340876

noncomputable def area_triangle_ABC (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_area_AC_1_AD_BC_circumcircle (A B C D E : ℝ × ℝ) (hAC : dist A C = 1)
  (hAD : dist A D = (2 / 3) * dist A B)
  (hMidE : E = ((B.1 + C.1) / 2, (B.2 + C.2) / 2))
  (hCircum : dist E ((A.1 + C.1) / 2, (A.2 + C.2) / 2) = 1 / 2) :
  area_triangle_ABC A B C = (Real.sqrt 5) / 6 :=
by
  sorry

end triangle_area_AC_1_AD_BC_circumcircle_l340_340876


namespace money_spent_correct_l340_340702

-- Define the number of plays, acts per play, wigs per act, and the cost of each wig
def num_plays := 3
def acts_per_play := 5
def wigs_per_act := 2
def wig_cost := 5
def sell_price := 4

-- Given the total number of wigs he drops and sells from one play
def dropped_plays := 1
def total_wigs_dropped := dropped_plays * acts_per_play * wigs_per_act
def money_from_selling_dropped_wigs := total_wigs_dropped * sell_price

-- Calculate the initial cost
def total_wigs := num_plays * acts_per_play * wigs_per_act
def initial_cost := total_wigs * wig_cost

-- The final spent money should be calculated by subtracting money made from selling the wigs of the dropped play
def final_spent_money := initial_cost - money_from_selling_dropped_wigs

-- Specify the expected amount of money John spent
def expected_final_spent_money := 110

theorem money_spent_correct :
  final_spent_money = expected_final_spent_money := by
  sorry

end money_spent_correct_l340_340702


namespace find_equation_of_line_l340_340088

theorem find_equation_of_line :
∀ (l : ℝ → ℝ), 
  (∃ k b, (∀ x, l x = k * x + b) ∧ (l 2 = 2 * 2 + 1) ∧ (l 1 = -1 + 2)) -> 
  (∃ k b, (∀ x, l x = k * x + b) ∧ k = 4 ∧ b = -3) :=
by
  intro l
  rintro ⟨k, b, hl, h1, h2⟩
  use [4, -3]
  split
  { intro x
    rw [←hl x, hl 2, hl 1, hl (2*x + 1), hl (-(x/x)+2),(4*x-3)] }
  { exact equations to calculate  k+= 4<= -3 }
  sorry

end find_equation_of_line_l340_340088


namespace geometric_progression_first_term_and_ratio_l340_340820

theorem geometric_progression_first_term_and_ratio (
  b_1 q : ℝ
) :
  b_1 * (1 + q + q^2) = 21 →
  b_1^2 * (1 + q^2 + q^4) = 189 →
  (b_1 = 12 ∧ q = 1/2) ∨ (b_1 = 3 ∧ q = 2) :=
by
  intros hsum hsumsq
  sorry

end geometric_progression_first_term_and_ratio_l340_340820


namespace pos_int_solutions_l340_340331

theorem pos_int_solutions (w x y z : ℕ) (h_eq : w + x + y + z = 20) (h_wx : w + x ≥ 5) (h_yz : y + z ≥ 5) : 
  ∃ n, n = 873 ∧ (w + x + y + z = 20 ∧ w + x ≥ 5 ∧ y + z ≥ 5) :=
by
  exists 873
  split
  ·
    rfl
  ·
    split
    ·
      exact h_eq
    ·
      split
      ·
        exact h_wx
      ·
        exact h_yz

end pos_int_solutions_l340_340331


namespace part1_part2_l340_340731

noncomputable theory

section Problem1

def f : ℝ → ℝ := sorry
def a : ℕ → ℝ := sorry

axiom domain_f (x : ℝ) : true
axiom f_pos_for_neg_x {x : ℝ} (h : x < 0) : f x > 1
axiom f_add_eq_mul (x y : ℝ) : f (x + y) = f x * f y
axiom a_initial : a 1 = f 0
axiom a_recursion {n : ℕ} : f (a (n + 1)) = 1 / f (-2 - a n)

theorem part1 : a 2003 = 4005 := sorry

end Problem1

section Problem2

def k := 2 * Real.sqrt 3 / 3

axiom sequence_inequality {n : ℕ} :
  (∏ i in Finset.range (n + 1), 1 + 1 / a (i + 1)) ≥ k * Real.sqrt (2 * n + 1)

theorem part2 : is_maximum k :=
begin
  sorry
end

end Problem2

end part1_part2_l340_340731


namespace eq_nine_l340_340136

theorem eq_nine (x y : ℝ) (h1 : x^2 + y^2 = 15) (h2 : x * y = 3) : (x - y)^2 = 9 := by
  sorry

end eq_nine_l340_340136


namespace sum_product_eq_l340_340276

theorem sum_product_eq (n : ℕ) (x : fin n → ℝ) (h_distinct : function.injective x) : 
  (∑ i : fin n, ∏ j : fin n, if i ≠ j then (1 - x i * x j) / (x i - x j) else 1) = (if even n then 0 else 1) := 
sorry

end sum_product_eq_l340_340276


namespace car_mileage_correct_l340_340527

def skips_digits_4_and_7 (n : Nat) : Prop :=
  ¬ (Nat.has_digit n 4 ∨ Nat.has_digit n 7)

theorem car_mileage_correct (odometer_reading : Nat) (actual_miles : Nat) :
  odometer_reading = 3008 → skips_digits_4_and_7 3008 →
  actual_miles = 1542 :=
    by
    sorry

end car_mileage_correct_l340_340527


namespace distance_to_work_l340_340106

theorem distance_to_work (
    bike_speed : ℝ,
    biking_hours_per_week : ℝ,
    workdays_per_week : ℕ,
    weekend_distance : ℝ,
    total_weekly_distance : ℝ
  ) : 
  bike_speed = 25 → 
  biking_hours_per_week = 16 → 
  workdays_per_week = 5 → 
  weekend_distance = 200 → 
  total_weekly_distance = bike_speed * biking_hours_per_week → 
  2 * workdays_per_week * D + weekend_distance = total_weekly_distance → 
  D = 20 :=
by
  sorry

end distance_to_work_l340_340106


namespace compute_value_3_std_devs_less_than_mean_l340_340070

noncomputable def mean : ℝ := 15
noncomputable def std_dev : ℝ := 1.5
noncomputable def skewness : ℝ := 0.5
noncomputable def kurtosis : ℝ := 0.6

theorem compute_value_3_std_devs_less_than_mean : 
  ¬∃ (value : ℝ), value = mean - 3 * std_dev :=
sorry

end compute_value_3_std_devs_less_than_mean_l340_340070


namespace angle_PEQ_is_180_l340_340779

-- The conditions that define our geometric setup
variables (A B C D P Q E : Point)
variables {O O1 O2 : Point} -- Centers of circles

-- The circle ω with center O passing through A, B, C, D
def circle_omega (O : Point) (A B C D : Point) : Prop :=
  circle O A B ∧ circle O B C ∧ circle O C D ∧ circle O D A

-- Circle ω₁ touching ω externally at C
def circle_omega1 (O1 : Point) (C : Point) : Prop :=
  circle O1 C ∧ tangent_outside O C O1

-- Circle ω₂ touching ω at D and ω₁ at E
def circle_omega2 (O2 : Point) (D E : Point) : Prop :=
  circle O2 D ∧ circle O2 E ∧ tangent_inside O D O2 ∧ tangent_outside O2 E O1

-- Line conditions for intersections at point P and point Q
def line_conditions (B D P A C Q : Point) : Prop :=
  lines_intersect B D P ∧ 
  lines_intersect A C Q

-- Condition for the angle we need to find
def angle_PEQ (P E Q : Point) : ℝ := angle_deg P E Q

-- Proof that angle PEQ is 180 degrees
theorem angle_PEQ_is_180
  (h1 : circle_omega O A B C D)
  (h2 : circle_omega1 O1 C)
  (h3 : circle_omega2 O2 D E)
  (h4 : line_conditions B D P A C Q) :
  angle_PEQ P E Q = 180 :=
sorry

end angle_PEQ_is_180_l340_340779


namespace units_digit_of_sum_is_7_l340_340456

noncomputable def original_num (a b c : ℕ) : ℕ := 100 * a + 10 * b + c
noncomputable def reversed_num (a b c : ℕ) : ℕ := 100 * c + 10 * b + a

theorem units_digit_of_sum_is_7 (a b c : ℕ) (h : a = 2 * c - 3) :
  (original_num a b c + reversed_num a b c) % 10 = 7 := by
  sorry

end units_digit_of_sum_is_7_l340_340456


namespace willam_tax_correct_l340_340575

-- Define the constants and conditions given in the problem
def total_tax_collected : ℝ := 3840 
def willam_land_percentage : ℝ := 21.701388888888893 / 100 -- Convert percentage to decimal

-- Define the expected tax that Mr. Willam paid (approximate)
def willam_farm_tax : ℝ := total_tax_collected * willam_land_percentage

-- The goal is to prove this statement
theorem willam_tax_correct : willam_farm_tax = 833.89 := by
  sorry

end willam_tax_correct_l340_340575


namespace find_freshmen_count_l340_340814

theorem find_freshmen_count
  (F S J R : ℕ)
  (h1 : F : S = 5 : 4)
  (h2 : S : J = 7 : 8)
  (h3 : J : R = 9 : 7)
  (total_students : F + S + J + R = 2158) :
  F = 630 :=
by 
  sorry

end find_freshmen_count_l340_340814


namespace gcd_30_45_is_15_l340_340228

theorem gcd_30_45_is_15 : Nat.gcd 30 45 = 15 := by
  sorry

end gcd_30_45_is_15_l340_340228


namespace necessarily_true_II_l340_340535

def digit := Nat
def statements := Σ (d : digit), d > 0 ∧ d < 6

-- I: The digit is 1.
def statement_I (d : digit) : Prop := d = 1

-- II: The digit is not 2 or 4.
def statement_II (d : digit) : Prop := d ≠ 2 ∧ d ≠ 4

-- III: The digit is 3.
def statement_III (d : digit) : Prop := d = 3

-- IV: The digit is not 5.
def statement_IV (d : digit) : Prop := d ≠ 5

theorem necessarily_true_II (d : digit) (h : d > 0 ∧ d < 6) 
  (h1 : (statement_I d ∨ statement_II d ∨ statement_III d ∨ statement_IV d) ∧ ¬(statement_I d ∧ statement_II d ∧ statement_III d ∧ statement_IV d) 
  ∧ ∃ (s1 s2 s3 : digit → Prop), (s1 = statement_I ∨ s1 = statement_II ∨ s1 = statement_III ∨ s1 = statement_IV)
  ∧ (s2 = statement_I ∨ s2 = statement_II ∨ s2 = statement_III ∨ s2 = statement_IV)
  ∧ (s3 = statement_I ∨ s3 = statement_II ∨ s3 = statement_III ∨ s3 = statement_IV)
  ∧ s1 ≠ s2 ∧ s2 ≠ s3 ∧ s1 ≠ s3 ∧ (s1 d ∧ s2 d ∧ s3 d)) : 
  statement_II d := sorry

end necessarily_true_II_l340_340535


namespace min_pizzas_needed_l340_340703

noncomputable def pizzas_needed (car_cost earnings_per_pizza expenses_per_pizza : ℕ) : ℕ :=
  let net_earning_per_pizza := earnings_per_pizza - expenses_per_pizza
  in (car_cost + net_earning_per_pizza - 1) / net_earning_per_pizza  -- Ceiling division

theorem min_pizzas_needed (car_cost : ℕ) (earnings_per_pizza : ℕ) (expenses_per_pizza : ℕ) :
  cars_cost = 6000 -> earnings_per_pizza = 12 -> expenses_per_pizza = 4 -> pizzas_needed car_cost earnings_per_pizza expenses_per_pizza = 750 :=
  by
    intro h1 h2 h3
    rw [h1, h2, h3]
    exact sorry

end min_pizzas_needed_l340_340703


namespace inequality_M_l340_340057

noncomputable def M : ℝ := (∏ i in finset.range 50, (2*i + 1)) / (∏ i in finset.range 50, (2*i + 2))

theorem inequality_M : (1 : ℝ) / 15 < M ∧ M < (1 : ℝ) / 10 := by
  sorry

end inequality_M_l340_340057


namespace equilateral_triangle_side_length_l340_340755

open Classical

noncomputable section

variable {P Q R S : Type}

def is_perpendicular_feet (P Q R S : P) : Prop :=
  sorry -- Definition for Q, R, S being the feet of perpendiculars from P

structure EquilateralTriangle (A B C P Q R S : P) where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  AB : ℝ
  (h_perpendicular : is_perpendicular_feet P Q R S)
  (h_area_eq : ∀ (h₁ : PQ = 2) (h₂ : PR = 3) (h₃ : PS = 4), AB = 12 * Real.sqrt 3)

theorem equilateral_triangle_side_length {A B C P Q R S : P} 
    (h_eq_triangle : EquilateralTriangle A B C P Q R S) : h_eq_triangle.AB = 12 * Real.sqrt 3 :=
  by
    cases h_eq_triangle with
    | mk PQ PR PS AB h_perpendicular h_area_eq =>
        apply h_area_eq
        · exact rfl
        · exact rfl
        · exact rfl

end equilateral_triangle_side_length_l340_340755


namespace fixed_point_YZ_l340_340379

theorem fixed_point_YZ {A B C X Y Z B' C' M G : Point} 
  (hABC : Triangle A B C)
  (hXBC : X ∈ segment B C)
  (hBB' : collinear X B B')
  (hCC' : collinear X C C')
  (hB'X_eq_BC : dist B' X = dist B C)
  (hC'X_eq_BC : dist C' X = dist B C)
  (hY : parallel_line_through X Y AB' AC)
  (hZ : parallel_line_through X Z AC' AB)
  (hM : M = midpoint B C)
  (hG : G = centroid A B C) :
  ∀ (X : segment B C), line_through Y Z G :=
sorry

end fixed_point_YZ_l340_340379


namespace AB_side_length_l340_340772

noncomputable def P := (x : ℝ) × (y : ℝ)

def is_foot_perpendicular (P : P) (A B : P) : P := sorry

def equilateral_triangle (A B C : P) : Prop := sorry

theorem AB_side_length (A B C P Q R S : P)
  (h_equilateral : equilateral_triangle A B C)
  (h_P_inside : sorry /* P inside ABC */)
  (h_Q_foot : Q = is_foot_perpendicular P A B) 
  (h_R_foot : R = is_foot_perpendicular P B C)
  (h_S_foot : S = is_foot_perpendicular P C A)
  (h_PQ : (dist P Q) = 2)
  (h_PR : (dist P R) = 3)
  (h_PS : (dist P S) = 4) :
  dist A B = 6 * real.sqrt 3 := 
sorry

end AB_side_length_l340_340772


namespace workers_combined_time_l340_340266

theorem workers_combined_time (g_rate a_rate c_rate : ℝ)
  (hg : g_rate = 1 / 70)
  (ha : a_rate = 1 / 30)
  (hc : c_rate = 1 / 42) :
  1 / (g_rate + a_rate + c_rate) = 14 :=
by
  sorry

end workers_combined_time_l340_340266


namespace nathan_has_83_bananas_l340_340415

def nathan_bananas (bunches_eight bananas_eight bunches_seven bananas_seven: Nat) : Nat :=
  bunches_eight * bananas_eight + bunches_seven * bananas_seven

theorem nathan_has_83_bananas (h1 : bunches_eight = 6) (h2 : bananas_eight = 8) (h3 : bunches_seven = 5) (h4 : bananas_seven = 7) : 
  nathan_bananas bunches_eight bananas_eight bunches_seven bananas_seven = 83 := by
  sorry

end nathan_has_83_bananas_l340_340415


namespace average_marks_correct_l340_340225

-- Define constants for the marks in each subject
def english_marks : ℕ := 76
def math_marks : ℕ := 65
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 67
def biology_marks : ℕ := 85

-- Define the total number of subjects
def num_subjects : ℕ := 5

-- Define the total marks as the sum of individual subjects
def total_marks : ℕ := english_marks + math_marks + physics_marks + chemistry_marks + biology_marks

-- Define the average marks
def average_marks : ℕ := total_marks / num_subjects

-- Prove that the average marks is as expected
theorem average_marks_correct : average_marks = 75 :=
by {
  -- skip the proof
  sorry
}

end average_marks_correct_l340_340225


namespace triple_hash_100_l340_340226

def hash (N : ℝ) : ℝ :=
  0.5 * N + N

theorem triple_hash_100 : hash (hash (hash 100)) = 337.5 :=
by
  sorry

end triple_hash_100_l340_340226


namespace distance_between_trees_l340_340543

theorem distance_between_trees
  (yard_length : ℕ)
  (num_trees : ℕ)
  (h_yard_length : yard_length = 441)
  (h_num_trees : num_trees = 22) :
  (yard_length / (num_trees - 1)) = 21 :=
by
  sorry

end distance_between_trees_l340_340543


namespace correct_inequality_l340_340132

theorem correct_inequality :
  1.6 ^ 0.3 > 0.9 ^ 3.1 :=
sorry

end correct_inequality_l340_340132


namespace gift_cost_l340_340010

theorem gift_cost (half_cost : ℝ) (h : half_cost = 14) : 2 * half_cost = 28 :=
by
  sorry

end gift_cost_l340_340010


namespace negation_of_p_is_neg_p_l340_340400

-- Define the proposition p
def p : Prop := ∀ n : ℕ, 3^n ≥ n^2 + 1

-- Define the negation of p
def neg_p : Prop := ∃ n_0 : ℕ, 3^n_0 < n_0^2 + 1

-- The proof statement
theorem negation_of_p_is_neg_p : ¬p ↔ neg_p :=
by sorry

end negation_of_p_is_neg_p_l340_340400


namespace net_salary_change_l340_340187

theorem net_salary_change (S : ℝ) : 
  let after_first_increase := S * 1.20 in
  let after_second_increase := after_first_increase * 1.40 in
  let after_first_decrease := after_second_increase * 0.65 in
  let final_salary := after_first_decrease * 0.75 in
  let net_change := final_salary - S in
  let percentage_change := (net_change / S) * 100 in
  percentage_change = -18.1 := 
by
  sorry

end net_salary_change_l340_340187


namespace street_lights_per_side_l340_340182

theorem street_lights_per_side
  (neighborhoods : ℕ)
  (roads_per_neighborhood : ℕ)
  (total_street_lights : ℕ)
  (total_neighborhoods : neighborhoods = 10)
  (roads_in_each_neighborhood : roads_per_neighborhood = 4)
  (street_lights_in_town : total_street_lights = 20000) :
  (total_street_lights / (neighborhoods * roads_per_neighborhood * 2) = 250) :=
by
  sorry

end street_lights_per_side_l340_340182


namespace number_of_a_values_l340_340987

theorem number_of_a_values (a : ℝ) : 
  (∃ a : ℝ, ∃ b : ℝ, a = 0 ∨ a = 1) := sorry

end number_of_a_values_l340_340987


namespace vertical_asymptote_unique_l340_340566

theorem vertical_asymptote_unique : 
  ∀ (x : ℝ), x ≠ 2 → (y = (x + 2) / (x ^ 2 - 4)) → (x = 2 ↔ ∃ l : ℝ, tendsto (λ x, (x + 2) / (x ^ 2 - 4)) (𝓝[≠] x) l) := 
by 
{
  sorry
}

end vertical_asymptote_unique_l340_340566


namespace side_length_eq_l340_340747

namespace EquilateralTriangle

variables (A B C P Q R S : Type) [HasVSub Type P] [MetricSpace P]
variables [HasDist P] [HasEquilateralTriangle ABC] [InsideTriangle P ABC]
variables [Perpendicular PQ AB] [Perpendicular PR BC] [Perpendicular PS CA]
variables [Distance PQ 2] [Distance PR 3] [Distance PS 4]

theorem side_length_eq : side_length ABC = 6 * √3 :=
sorry
end EquilateralTriangle

end side_length_eq_l340_340747


namespace smallest_possible_value_d_l340_340586

noncomputable def smallest_d : ℝ := 4

theorem smallest_possible_value_d (d : ℝ) :
  d = 4 →
  (sqrt ((4 * sqrt 5) ^ 2 + (2 * d - 4) ^ 2) = 4 * d) :=
by
  sorry

end smallest_possible_value_d_l340_340586


namespace smallest_positive_period_f_l340_340585

def f (x : ℝ) : ℝ := (Real.cos x + Real.sin x) / (Real.cos x - Real.sin x)

theorem smallest_positive_period_f : ∃ T > 0, ∀ x, f (x + T) = f x ∧ (∀ T' > 0, T' < T → ¬ (∀ x, f (x + T') = f x)) := 
sorry

end smallest_positive_period_f_l340_340585


namespace final_position_correct_l340_340804

-- Define the initial position and the transformations
def initial_position : (ℝ × ℝ) := (-1, 1)

def rotate (angle: ℝ) (pos: ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := pos in
  let rad := angle * (Float.pi / 180.0) in
  (x * Float.cos rad - y * Float.sin rad, x * Float.sin rad + y * Float.cos rad)

def reflect_x (pos: ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := pos in
  (x, -y)

def reflect_y (pos: ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := pos in
  (-x, y)

def half_turn (pos: ℝ × ℝ) : ℝ × ℝ :=
  rotate 180 pos

-- Defining the series of transformations
def transform (pos: ℝ × ℝ) : ℝ × ℝ :=
  let pos1 := rotate 270 pos in
  let pos2 := reflect_x pos1 in
  let pos3 := reflect_y pos2 in
  half_turn pos3

-- The final position after all transformations starting from initial_position
def final_position := transform initial_position

-- Proof Statement
theorem final_position_correct : final_position = (-1, -1) :=
  sorry

end final_position_correct_l340_340804


namespace permutations_b_before_e_l340_340330

theorem permutations_b_before_e {α : Type*} (s : Finset α) (hp : s = {a, b, c, d, e, f}) : 
  (∃ t : Finset (List α), t.card = 360 ∧ ∀ l ∈ t, (l.index_of b < l.index_of e)) := by
  sorry

end permutations_b_before_e_l340_340330


namespace grocer_profit_l340_340159

theorem grocer_profit
  (purch_price : ℝ := 0.50)
  (purch_qty : ℝ := 3)
  (sell_price : ℝ := 1.00)
  (sell_qty : ℝ := 4)
  (total_qty : ℝ := 132) :
  let cost_price_per_pound := purch_price / purch_qty,
      total_cost_price := total_qty * cost_price_per_pound,
      selling_price_per_pound := sell_price / sell_qty,
      total_selling_price := total_qty * selling_price_per_pound,
      profit := total_selling_price - total_cost_price
  in  profit = 10.9956 := 
by 
  -- Definitions of cost price per pound, total cost price, 
  -- selling price per pound, and total selling price
  let cost_price_per_pound := purch_price / purch_qty
  let total_cost_price := total_qty * cost_price_per_pound
  let selling_price_per_pound := sell_price / sell_qty
  let total_selling_price := total_qty * selling_price_per_pound
  let profit := total_selling_price - total_cost_price
  -- Assert the calculated profit
  have h : profit = 10.9956 := sorry
  exact h

end grocer_profit_l340_340159


namespace side_length_eq_l340_340750

namespace EquilateralTriangle

variables (A B C P Q R S : Type) [HasVSub Type P] [MetricSpace P]
variables [HasDist P] [HasEquilateralTriangle ABC] [InsideTriangle P ABC]
variables [Perpendicular PQ AB] [Perpendicular PR BC] [Perpendicular PS CA]
variables [Distance PQ 2] [Distance PR 3] [Distance PS 4]

theorem side_length_eq : side_length ABC = 6 * √3 :=
sorry
end EquilateralTriangle

end side_length_eq_l340_340750


namespace parts_of_milk_in_drink_A_l340_340153

theorem parts_of_milk_in_drink_A (x : ℝ) (h : 63 * (4 * x) / (7 * (x + 3)) = 63 * 3 / (x + 3) + 21) : x = 16.8 :=
by
  sorry

end parts_of_milk_in_drink_A_l340_340153


namespace second_quadrant_point_l340_340352

theorem second_quadrant_point (x : ℝ) (h1 : x < 2) (h2 : x > 1/2) : 
  (x-2 < 0) ∧ (2*x-1 > 0) ↔ (1/2 < x ∧ x < 2) :=
by
  sorry

end second_quadrant_point_l340_340352


namespace largest_common_term_arith_seq_l340_340196

theorem largest_common_term_arith_seq :
  ∃ a, a < 90 ∧ (∃ n : ℤ, a = 3 + 8 * n) ∧ (∃ m : ℤ, a = 5 + 9 * m) ∧ a = 59 :=
by
  sorry

end largest_common_term_arith_seq_l340_340196


namespace arrangement_possible_in_ShapeSh_arrangement_not_possible_in_strips_l340_340004

noncomputable def sum_1_to_8 : ℕ := 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8

theorem arrangement_possible_in_ShapeSh (digits : List ℕ) (partition : List (List ℕ)) (h_dig_sum : sum_1_to_8 = 36) 
(h_digits_unique : digits.nodup) (h_digits_range : ∀ d ∈ digits, d ≥ 1 ∧ d ≤ 8) (h_partition_valid : partition.join = digits) 
(h_partition_div : ∀ part1 part2 ∈ partition, part1.sum % part2.sum = 0) : 
  ∃ (arrangement : List (List (Σ' i : ℕ, (i ≥ 1 ∧ i ≤ 8)))), arrangement = partition := sorry

theorem arrangement_not_possible_in_strips (digits : List ℕ) (partition : List (List ℕ)) (h_dig_sum : sum_1_to_8 = 36) 
(h_digits_unique : digits.nodup) (h_digits_range : ∀ d ∈ digits, d ≥ 1 ∧ d ≤ 8) (h_partition_valid : partition.join = digits) 
(h_partition_div : ∀ part1 part2 ∈ partition, part1.sum % part2.sum = 0) : 
  ¬ ∃ (arrangement : List (List (Σ' i : ℕ, (i ≥ 1 ∧ i ≤ 8)))), arrangement = partition := sorry

end arrangement_possible_in_ShapeSh_arrangement_not_possible_in_strips_l340_340004


namespace father_cannot_see_boy_more_than_half_time_l340_340862

-- Definitions based on the problem conditions
structure SquareSchool :=
  (perimeter : ℝ)

structure Boy :=
  (speed_boy : ℝ := 10)

structure Father :=
  (speed_father : ℝ := 5)
  (can_change_direction : bool := true)

constant same_side : SquareSchool → Boy → Father → (ℝ → ℝ → Prop)

-- Main theorem statement
theorem father_cannot_see_boy_more_than_half_time
  (school : SquareSchool)
  (boy : Boy)
  (father : Father)
  (t : ℝ) (T_b : ℝ) (T_f : ℝ)
  (hT_b : T_b = school.perimeter / boy.speed_boy)
  (hT_f : T_f = school.perimeter / father.speed_father) :
  (∃ t₁ t₂ : ℝ, 0 ≤ t₁ ∧ t₁ < t₂ ∧ t₂ ≤ t ∧ ¬same_side school boy father t₁ t₂) → 
  t / 2 ≥ t :=
sorry

end father_cannot_see_boy_more_than_half_time_l340_340862


namespace angle_APB_always_acute_dot_product_BQ_AQ_rhombus_condition_l340_340615

def point := (ℝ × ℝ)

def A : point := (-1, 0)
def B : point := (0, 1)

def P (x : ℝ) : point := (x, x - 1)
def Q : point := (0, -1)

-- Step (I): Prove that ∠APB is always acute.
theorem angle_APB_always_acute (x : ℝ) :
  let PA := (fst A - x, snd A - (x - 1)),
      PB := (fst B - x, snd B - (x - 1)) in
  ∃ θ : ℝ, 0 < θ ∧ θ < π / 2 ∧
  (cos θ = (fst PA * fst PB + snd PA * snd PB) / 
           (sqrt (fst PA ^ 2 + snd PA ^ 2) * sqrt (fst PB ^ 2 + snd PB ^ 2))) := by sorry

-- Step (II): Prove that BQ ⋅ AQ = 2 when ABPQ is a rhombus with specific P and Q.
theorem dot_product_BQ_AQ_rhombus_condition :
  let BQ := (fst Q - fst B, snd Q - snd B),
      AQ := (fst Q - fst A, snd Q - snd A) in
  (fst BQ * fst AQ + snd BQ * snd AQ) = 2 := by sorry

end angle_APB_always_acute_dot_product_BQ_AQ_rhombus_condition_l340_340615


namespace max_area_cost_constraint_l340_340879

-- Define the Lagrange multiplier problem and the conditions.
variable {x y: ℝ} (x_pos: 0 < x) (y_pos: 0 < y)

def material_cost := 900 * x + 400 * y + 200 * x * y
def area := x * y

-- The total cost should not exceed 32000 yuan.
def cost_constraint := material_cost x y ≤ 32000

-- The main theorem we want to show.
theorem max_area_cost_constraint :
  ∃ (x y : ℝ), (0 < x) ∧ (0 < y) ∧ cost_constraint x y ∧ (area x y) = 100 ∧ x = 20 / 3 :=
sorry

end max_area_cost_constraint_l340_340879


namespace teresas_age_is_43_l340_340528

noncomputable def teresasAge (guesses : List ℕ) : ℕ :=
  if h : ∃ age, prime age ∧ age ∈ guesses ∧
                (guesses.filter (λ g => g < age)).length ≥ (guesses.length / 2) ∧
                (guesses.filter (λ g => g == 43)).length == 3 ∧
                (∀ g ∈ guesses, g ≠ age → abs (g - age) ≥ 2) ∧
                (guesses.forall (λ g => g ≥ 35)) then
    Classical.choose h
  else 0

theorem teresas_age_is_43 :
  let guesses := [35, 39, 41, 43, 47, 49, 51, 53, 58, 60]
  teresasAge guesses = 43 :=
by
  sorry

end teresas_age_is_43_l340_340528


namespace quadratic_rewrite_sum_l340_340463

theorem quadratic_rewrite_sum (a b c : ℝ) (x : ℝ) :
  -3 * x^2 + 15 * x + 75 = a * (x + b)^2 + c → (a + b + c) = 88.25 :=
sorry

end quadratic_rewrite_sum_l340_340463


namespace division_twice_correct_l340_340253

-- Define a, b, FirstDivisionResult, and SecondDivisionResult
def a : ℝ := 166.08
def b : ℝ := 4

def FirstDivisionResult : ℝ := a / b
def SecondDivisionResult : ℝ := FirstDivisionResult / b

-- Lean statement to prove
theorem division_twice_correct : SecondDivisionResult = 10.38 := 
by 
  -- Proof is omitted; sorry is a placeholder here
  sorry

end division_twice_correct_l340_340253


namespace number_of_correct_propositions_is_zero_l340_340912

theorem number_of_correct_propositions_is_zero :
  (¬(∀ p : Prop, p = (∃ (P : Type), is_polyhedron P ∧ has_faces P 5 ∧ is_triangular_prism P)) ∧
   ¬(∀ p : Prop, p = (∃ (P : Type), is_pyramid P ∧ is_cut_with_plane P ∧ is_frustum P)) ∧
   ¬(∀ p : Prop, p = (∃ (P : Type), is_pentahedron P ∧ has_one_pair_parallel_faces P ∧ is_frustum P)) ∧
   ¬(∀ p : Prop, p = (∃ (P : Type), is_geometric_body P ∧ has_polygon_face P ∧ has_triangular_faces_with_common_vertex P ∧ is_pyramid P))) →
  (count_correct_propositions ([false, false, false, false]) = 0) :=
begin
  sorry
end

end number_of_correct_propositions_is_zero_l340_340912


namespace min_sum_of_first_n_terms_l340_340354

-- Define arithmetic sequence and terms
def arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + ↑n * d

-- Definitions based on the problem conditions
def a₁ : ℤ := -3
def d : ℤ := 2

-- Conditions derived from the problem
def condition1 (a₁ : ℤ) (d : ℤ) : Prop :=
  11 * (arithmetic_sequence a₁ d 4) = 5 * (arithmetic_sequence a₁ d 7)

def condition2 (a₁ : ℤ) : Prop := 
  a₁ = -3

-- Statement to prove
theorem min_sum_of_first_n_terms : 
  condition1 a₁ d → condition2 a₁ → S_n arithmetic_sequence (-4) :=
begin
  sorry -- Proof to be provided
end

end min_sum_of_first_n_terms_l340_340354


namespace length_BC_l340_340745

theorem length_BC (O A M B C : Point) (alpha : Real) :
  (AO : Segment) is_radius O ∧
  M ∈ AO ∧
  B ∈ Circle(O, 10) ∧
  C ∈ Circle(O, 10) ∧
  ∠(A M B) = alpha ∧
  ∠(O M C) = alpha ∧
  sin(alpha) = sqrt(24) / 5 →
  length(B, C) = 4 :=
by
  sorry

end length_BC_l340_340745


namespace periodic_sum_f_l340_340596

noncomputable def f1 (x : ℝ) : ℝ := Math.sin x + Math.cos x

noncomputable def f : ℕ → (ℝ → ℝ)
| 0 => f1
| (n + 1) => (f n)' -- f1 derivative if n=1, etc

theorem periodic_sum_f (n : ℕ) (hn : n % 4 = 1) (x : ℝ) :
  ∑ i in Finset.range (n+1), (f i x) = Math.sin (x) + Math.cos (x) := by
  sorry

example : 
  (∑ i in Finset.range 2017, (f i) (Real.pi / 3)) = (1 + Real.sqrt 3) / 2 := 
  periodic_sum_f 2016 (by norm_num) (Real.pi / 3)

end periodic_sum_f_l340_340596


namespace triangle_construction_exists_l340_340223

noncomputable def construct_triangle (a : ℝ) (A : ℝ) (r : ℝ) : Prop :=
  ∃ (ABC : Triangle), 
    ABC.side_a = a ∧
    ABC.angle_A = A ∧
    ABC.incircle_radius = r

-- Use the statement to declare a theorem
theorem triangle_construction_exists (a : ℝ) (A : ℝ) (r : ℝ) : 
  construct_triangle a A r :=
sorry

end triangle_construction_exists_l340_340223


namespace polygon_E_has_largest_area_l340_340495

/-- Areas of the polygons --/
def area_Polygon_A := 5
def area_Polygon_B := 5
def area_Polygon_C := 5
def area_Polygon_D := 4 + 1 * 0.5
def area_Polygon_E := 5 + 1 * 0.5

/-- Prove that polygon E has the largest area --/
theorem polygon_E_has_largest_area :
  area_Polygon_E > area_Polygon_A ∧
  area_Polygon_E > area_Polygon_B ∧
  area_Polygon_E > area_Polygon_C ∧
  area_Polygon_E > area_Polygon_D :=
by {
  sorry
}

end polygon_E_has_largest_area_l340_340495


namespace martin_crayons_l340_340408

theorem martin_crayons : (8 * 7 = 56) := by
  sorry

end martin_crayons_l340_340408


namespace sin_double_angle_plus_pi_over_six_l340_340995

variable (θ : ℝ)
variable (h : 7 * Real.sqrt 3 * Real.sin θ = 1 + 7 * Real.cos θ)

theorem sin_double_angle_plus_pi_over_six :
  Real.sin (2 * θ + π / 6) = 97 / 98 :=
by
  sorry

end sin_double_angle_plus_pi_over_six_l340_340995


namespace alyssa_spent_total_l340_340911

theorem alyssa_spent_total {g c : ℝ} (h1 : g = 12.08) (h2 : c = 9.85) : g + c = 21.93 :=
by 
  rw [h1, h2]
  norm_num
  sorry

end alyssa_spent_total_l340_340911


namespace total_number_of_rats_l340_340012

theorem total_number_of_rats (Kenia Hunter Elodie Teagan : ℕ) 
  (h1 : Elodie = 30)
  (h2 : Elodie = Hunter + 10)
  (h3 : Kenia = 3 * (Hunter + Elodie))
  (h4 : Teagan = 2 * Elodie)
  (h5 : Teagan = Kenia - 5) : 
  Kenia + Hunter + Elodie + Teagan = 260 :=
by 
  sorry

end total_number_of_rats_l340_340012


namespace DE_parallel_FG_l340_340022

open Circle Geometry

variable (Γ : Circle) (A B C D E F G: Point)

axiom acute_triangle_ABC : Triangle A B C
axiom circumscribed_ABC : circumscribed Γ (Triangle A B C)
axiom D_on_segment_AB : on_segment D (segment A B)
axiom E_on_segment_AC : on_segment E (segment A C)
axiom AD_eq_AE : segment A D = segment A E 
axiom F_on_arc_AB : on_arc F (arc A B Γ) 
axiom G_on_arc_AC : on_arc G (arc A C Γ)
axiom perp_bisector_BD : is_perp_bisector (perp_bisector (segment B D)) (line A B)
axiom perp_bisector_CE : is_perp_bisector (perp_bisector (segment C E)) (line A C)

theorem DE_parallel_FG : parallel (line D E) (line F G) :=
sorry

end DE_parallel_FG_l340_340022


namespace cory_prime_sum_l340_340563

def primes_between_30_and_60 : List ℕ := [31, 37, 41, 43, 47, 53, 59]

theorem cory_prime_sum :
  let smallest := 31
  let largest := 59
  let median := 43
  smallest ∈ primes_between_30_and_60 ∧
  largest ∈ primes_between_30_and_60 ∧
  median ∈ primes_between_30_and_60 ∧
  primes_between_30_and_60 = [31, 37, 41, 43, 47, 53, 59] → 
  smallest + largest + median = 133 := 
by
  intros; sorry

end cory_prime_sum_l340_340563


namespace transform_log_graph_l340_340084

theorem transform_log_graph :
  ∀ x : ℝ, y = log 10 x → r = (y + 2) ↔ ∃ x', x' = x + 2 := by
  sorry

end transform_log_graph_l340_340084


namespace num_bad_carrots_l340_340213

theorem num_bad_carrots (carol_picked : ℕ) (mom_picked : ℕ) (brother_picked : ℕ) (good_carrots : ℕ) :
  carol_picked = 29 → 
  mom_picked = 16 →
  brother_picked = 23 →
  good_carrots = 52 →
  (carol_picked + mom_picked + brother_picked - good_carrots) = 16 := by
  intro h_carol
  intro h_mom
  intro h_brother
  intro h_good
  rw [h_carol, h_mom, h_brother, h_good]
  sorry

end num_bad_carrots_l340_340213


namespace greatest_ratio_irrational_l340_340622

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2)

theorem greatest_ratio_irrational (A B C D : ℝ × ℝ)
  (hA : A.1 ^ 2 + A.2 ^ 2 = 16) (hB : B.1 ^ 2 + B.2 ^ 2 = 16)
  (hC : C.1 ^ 2 + C.2 ^ 2 = 16) (hD : D.1 ^ 2 + D.2 ^ 2 = 16)
  (hAB_irrational : ¬ ∃ n : ℕ, distance A B = n)
  (hCD_irrational : ¬ ∃ n : ℕ, distance C D = n) :
  ∃ (k : ℝ), k = 2 * real.sqrt 2 ∧ (distance A B) / (distance C D) = k :=
begin
  sorry
end

end greatest_ratio_irrational_l340_340622


namespace interval_of_increase_l340_340229

def f (x : ℝ) : ℝ := log (1 / 2) (x ^ 2 - 2 * x - 3)

theorem interval_of_increase :
    ∀ x ∈ set.Iio (-1), x ^ 2 - 2 * x - 3 > 0 ∧ (∀ y, y ∈ set.Iio (-1) → f y < f x) :=
by
  sorry

end interval_of_increase_l340_340229


namespace sum_of_diagonals_l340_340160

noncomputable def length_AB : ℝ := 31
noncomputable def length_sides : ℝ := 81

def hexagon_inscribed_in_circle (A B C D E F : Type) : Prop :=
-- Assuming A, B, C, D, E, F are suitable points on a circle
-- Definitions to be added as per detailed proof needs
sorry

theorem sum_of_diagonals (A B C D E F : Type) :
    hexagon_inscribed_in_circle A B C D E F →
    (length_AB + length_sides + length_sides + length_sides + length_sides + length_sides = 384) := 
by
  sorry

end sum_of_diagonals_l340_340160


namespace find_marks_of_a_l340_340072

theorem find_marks_of_a (A B C D E : ℕ)
  (h1 : (A + B + C) / 3 = 48)
  (h2 : (A + B + C + D) / 4 = 47)
  (h3 : E = D + 3)
  (h4 : (B + C + D + E) / 4 = 48) : 
  A = 43 :=
by
  sorry

end find_marks_of_a_l340_340072


namespace exists_point_with_min_distance_to_vertices_l340_340195

theorem exists_point_with_min_distance_to_vertices
  (T : Triangle ℝ) (h_acute : ∀ (α : Angle), α ∈ T.angles → 0 < α ∧ α < π / 2)
  (h_area : T.area = 1) :
  ∃ P : Point ℝ, T.interior P ∧ ∀ v ∈ T.vertices, dist P v ≥ 2 / 27^(1/4) :=
by
  sorry

end exists_point_with_min_distance_to_vertices_l340_340195


namespace loss_percentage_l340_340164

theorem loss_percentage {C : ℝ} (hC : 20 * (C + 0.20 * C) = 60) (num_articles : ℝ) (h_num_articles : num_articles ≈ 34.99999562500055) :
  let cost_per_article := C / 20 in
  let total_cost_price := 35 * cost_per_article in
  let selling_price := 70 in
  let loss := total_cost_price - selling_price in
  let loss_percentage := (loss / total_cost_price) * 100 in
  loss_percentage = 20 := by
    -- proof omitted
    sorry

end loss_percentage_l340_340164


namespace inequality_proof_equality_condition_l340_340785

theorem inequality_proof (x y : ℝ) (hx : x > -1) (hy : y > -1) (hxy : x + y = 1) :
  (x / (y + 1) + y / (x + 1)) ≥ (2 / 3) :=
begin
  sorry
end

theorem equality_condition (x y : ℝ) (hx : x > -1) (hy : y > -1) (hxy : x + y = 1) :
  (x / (y + 1) + y / (x + 1)) = (2 / 3) ↔ x = 1 / 2 ∧ y = 1 / 2 :=
begin
  sorry
end

end inequality_proof_equality_condition_l340_340785


namespace sum_of_a_for_repeated_root_l340_340568

theorem sum_of_a_for_repeated_root :
  ∀ a : ℝ, (∀ x : ℝ, 2 * x^2 + a * x + 10 * x + 16 = 0 → 
               (a + 10 = 8 * Real.sqrt 2 ∨ a + 10 = -8 * Real.sqrt 2)) → 
               (a = -10 + 8 * Real.sqrt 2 ∨ a = -10 - 8 * Real.sqrt 2) → 
               ((-10 + 8 * Real.sqrt 2) + (-10 - 8 * Real.sqrt 2) = -20) := by
sorry

end sum_of_a_for_repeated_root_l340_340568


namespace different_distributions_l340_340871

def arrangement_methods (students teachers: Finset ℕ) : ℕ :=
  students.card.factorial * (students.card - 1).factorial * ((students.card - 1) - 1).factorial

theorem different_distributions :
  ∀ (students teachers : Finset ℕ), 
  students.card = 3 ∧ teachers.card = 3 →
  arrangement_methods students teachers = 72 :=
by sorry

end different_distributions_l340_340871


namespace term_added_from_k_to_k_plus_1_l340_340832

theorem term_added_from_k_to_k_plus_1 
    (S : ℕ → ℕ → ℚ) 
    (hS : ∀ n : ℕ, S n (n + 1) = 1 + ∑ i in finset.range n, (1 / ∑ j in finset.range (i + 1), (j + 1))) :
    ∀ k : ℕ, S (k + 1) ((k + 1) + 1) - S k (k + 1) = 2 / ((k + 1) * (k + 2)) := 
by 
  intro k
  rw [hS (k + 1), hS k]
  sorry

end term_added_from_k_to_k_plus_1_l340_340832


namespace jack_final_apples_l340_340698

-- Jack's transactions and initial count as conditions
def initial_count : ℕ := 150
def sold_to_jill : ℕ := initial_count * 30 / 100
def remaining_after_jill : ℕ := initial_count - sold_to_jill
def sold_to_june : ℕ := remaining_after_jill * 20 / 100
def remaining_after_june : ℕ := remaining_after_jill - sold_to_june
def donated_to_charity : ℕ := 5
def final_count : ℕ := remaining_after_june - donated_to_charity

-- Proof statement
theorem jack_final_apples : final_count = 79 := by
  sorry

end jack_final_apples_l340_340698


namespace trig_225_deg_l340_340931

noncomputable def sin_225 : Real := Real.sin (225 * Real.pi / 180)
noncomputable def cos_225 : Real := Real.cos (225 * Real.pi / 180)

theorem trig_225_deg :
  sin_225 = -Real.sqrt 2 / 2 ∧ cos_225 = -Real.sqrt 2 / 2 := by
  sorry

end trig_225_deg_l340_340931


namespace parity_of_f_l340_340461

def f (x : ℝ) : ℝ := if x ≠ 0 then (Real.sin x * sqrt (1 - abs x)) / (abs (x + 2) - 2) else 0

theorem parity_of_f : 
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ x ≠ 0 → abs (x + 2) - 2 ≠ 0) →
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ x ≠ 0 → f(-x) = -f(x) :=
begin
  sorry
end

end parity_of_f_l340_340461


namespace new_interest_rate_is_five_l340_340803

theorem new_interest_rate_is_five :
  let I := 202.50 in
  let R := 4.5 in
  let T := 1 in
  let I_additional := 22.5 in
  let P := I * 100 / R in
  let R_new := I_additional * 100 / P + R in
  R_new = 5 :=
by
  let I := 202.50
  let R := 4.5
  let T := 1
  let I_additional := 22.5
  let P := I * 100 / R
  let R_new := I_additional * 100 / P + R
  show R_new = 5 from sorry

end new_interest_rate_is_five_l340_340803


namespace system_of_equations_solution_l340_340865

theorem system_of_equations_solution :
  ∃ x₁ x₂ : ℝ, (1 ≤ x₁ ∧ x₁ < 2) ∧ x₂ = -1/2 ∧ 
  (2 * Real.floor x₁ + x₂ = 3/2) ∧ (3 * Real.floor x₁ - 2 * x₂ = 4) :=
sorry

end system_of_equations_solution_l340_340865


namespace angle_D_calculation_l340_340268

theorem angle_D_calculation (A B E C D : ℝ)
  (h1 : A + B = 180)
  (h2 : C = D)
  (h3 : A = 50)
  (h4 : E = 60)
  (h5 : A + B + E = 180)
  (h6 : B + C + D = 180) :
  D = 55 :=
by
  sorry

end angle_D_calculation_l340_340268


namespace max_value_of_exp_diff_l340_340248

open Real

theorem max_value_of_exp_diff : ∀ x : ℝ, ∃ y : ℝ, y = 2^x - 4^x ∧ y ≤ 1/4 := sorry

end max_value_of_exp_diff_l340_340248


namespace find_a_l340_340613

noncomputable def f (a : ℝ) (b : ℝ) (x : ℝ) := a^x + b * a^(-x)

theorem find_a (a b : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : ∀ x, f a b x = -f a b (-x)) 
    (h4 : ∃ x ∈ Icc (-1 : ℝ) 1, f a b x = 8 / 3) : a = 3 ∨ a = 1 / 3 :=
  sorry

end find_a_l340_340613


namespace gas_volumes_correct_l340_340508

noncomputable def west_gas_vol_per_capita : ℝ := 21428
noncomputable def non_west_gas_vol : ℝ := 185255
noncomputable def non_west_population : ℝ := 6.9
noncomputable def non_west_gas_vol_per_capita : ℝ := non_west_gas_vol / non_west_population

noncomputable def russia_gas_vol_68_percent : ℝ := 30266.9
noncomputable def russia_gas_vol : ℝ := russia_gas_vol_68_percent * 100 / 68
noncomputable def russia_population : ℝ := 0.147
noncomputable def russia_gas_vol_per_capita : ℝ := russia_gas_vol / russia_population

theorem gas_volumes_correct :
  west_gas_vol_per_capita = 21428 ∧
  non_west_gas_vol_per_capita = 26848.55 ∧
  russia_gas_vol_per_capita = 302790.13 := by
    sorry

end gas_volumes_correct_l340_340508


namespace smoking_related_lung_cancer_l340_340690

theorem smoking_related_lung_cancer :
  (∀ (S : Type) (e : S → Prop) (p : S → Prop), 
   (¬ (∀ x, e x → p x) ∨ (e = p) ∨ (¬ ∃ x, e x ∧ p x) ∨ (∃ x, e x ∧ ¬ p x)) ∧
   (¬ (∀ (x : S), p x ↔ e x) ) := 
begin 
  sorry 
end) 
→ ∃ (s : ℕ), s = 1 :=
begin
  intro h,
  use 1,
  sorry,
end

end smoking_related_lung_cancer_l340_340690


namespace middle_four_cells_third_row_l340_340688

-- Define the grid constraints for a 6x6 grid with cells filled by letters A to F.
def is_valid_grid (grid : ℕ → ℕ → char) : Prop :=
  ∀ i j, (i < 6) ∧ (j < 6) ∧ ('A' ≤ grid i j ∧ grid i j ≤ 'F') ∧
    (∀ k, (k < 6) → (k ≠ j → grid i j ≠ grid i k) ∧ (k ≠ i → grid i j ≠ grid k j)) ∧
    (∀ a b, (a < 2) ∧ (b < 3) → grid (2 * (i / 2) + a) (3 * (j / 3) + b) ≠ grid i j)

-- Define the specific problem statement.
theorem middle_four_cells_third_row (grid : ℕ → ℕ → char) 
  (H : is_valid_grid grid) :
  (grid 2 1, grid 2 2, grid 2 3, grid 2 4) = ('D', 'F', 'C', 'E') :=
sorry

end middle_four_cells_third_row_l340_340688


namespace sum_of_center_and_radius_l340_340019

-- Define the circle equation and necessary constants
def circle_eq (x y : ℝ) : Prop := 2 * x^2 + 3 * y - 25 = -y^2 + 12 * x + 4

-- Define the center and radius
def center (a b : ℝ) := a = 3 ∧ b = -3/2
def radius (r : ℝ) := r = real.sqrt 27.5

-- Main theorem stating that a + b + r = 6.744
theorem sum_of_center_and_radius (a b r : ℝ) (h_center : center a b) (h_radius : radius r) : 
  a + b + r = 6.744 := 
sorry

end sum_of_center_and_radius_l340_340019


namespace pharmacist_weights_exist_l340_340169

theorem pharmacist_weights_exist :
  ∃ (a b c : ℝ), a + b = 100 ∧ a + c = 101 ∧ b + c = 102 ∧ a < 90 ∧ b < 90 ∧ c < 90 :=
by
  sorry

end pharmacist_weights_exist_l340_340169


namespace factor_expression_l340_340235

theorem factor_expression :
  (12 * x ^ 6 + 40 * x ^ 4 - 6) - (2 * x ^ 6 - 6 * x ^ 4 - 6) = 2 * x ^ 4 * (5 * x ^ 2 + 23) :=
by sorry

end factor_expression_l340_340235


namespace trigonometric_expression_identity_l340_340216

open Real

theorem trigonometric_expression_identity :
  (1 - 1 / cos (35 * (pi / 180))) * 
  (1 + 1 / sin (55 * (pi / 180))) * 
  (1 - 1 / sin (35 * (pi / 180))) * 
  (1 + 1 / cos (55 * (pi / 180))) = 1 := by
  sorry

end trigonometric_expression_identity_l340_340216


namespace total_salad_dressing_weight_l340_340410

noncomputable def bowl_volume := 150 -- Volume of the bowl in ml
def oil_fraction := 2 / 3 -- Fraction of the bowl that is oil
def vinegar_fraction := 1 / 3 -- Fraction of the bowl that is vinegar
def oil_density := 5 -- Density of oil (g/ml)
def vinegar_density := 4 -- Density of vinegar (g/ml)

def oil_volume := oil_fraction * bowl_volume -- Volume of oil in ml
def vinegar_volume := vinegar_fraction * bowl_volume -- Volume of vinegar in ml
def oil_weight := oil_volume * oil_density -- Weight of the oil in grams
def vinegar_weight := vinegar_volume * vinegar_density -- Weight of the vinegar in grams
def total_weight := oil_weight + vinegar_weight -- Total weight of the salad dressing in grams

theorem total_salad_dressing_weight : total_weight = 700 := by
  sorry

end total_salad_dressing_weight_l340_340410


namespace find_fourth_vertex_l340_340989

def isSquare (a b c d : ℂ) : Prop :=
  ∃ (u : ℂ), (u ≠ 0 ∧ a = b + u ∧ c = b + u * matrix.complexRotation 0.5 ∧ d = b + u * matrix.complexRotation 1)

theorem find_fourth_vertex (z1 z2 z3 z4 : ℂ) (h₁ : z1 = 2 + 3 * complex.I) (h₂ : z2 = -3 + 2 * complex.I) (h₃ : z3 = -2 - 3 * complex.I) :
  isSquare z1 z2 z3 z4 → z4 = 3 - 2 * complex.I :=
begin
  intro h,
  sorry
end

end find_fourth_vertex_l340_340989


namespace freshman_count_630_l340_340817

-- Define the variables and conditions
variables (f o j s : ℕ)
variable (total_students : ℕ)

-- Define the ratios given in the problem
def freshmen_to_sophomore : Prop := f = (5 * o) / 4
def sophomore_to_junior : Prop := j = (8 * o) / 7
def junior_to_senior : Prop := s = (7 * j) / 9

-- Total number of students condition
def total_students_condition : Prop := f + o + j + s = total_students

theorem freshman_count_630
  (h1 : freshmen_to_sophomore f o)
  (h2 : sophomore_to_junior o j)
  (h3 : junior_to_senior j s)
  (h4 : total_students_condition f o j s 2158) :
  f = 630 :=
sorry

end freshman_count_630_l340_340817


namespace joe_total_time_to_school_l340_340701

theorem joe_total_time_to_school:
  ∀ (d r_w: ℝ), (1 / 3) * d = r_w * 9 →
                  4 * r_w * (2 * (r_w * 9) / (3 * (4 * r_w))) = (2 / 3) * d →
                  (1 / 3) * d / r_w + (2 / 3) * d / (4 * r_w) = 13.5 :=
by
  intros d r_w h1 h2
  sorry

end joe_total_time_to_school_l340_340701


namespace tan_double_angle_l340_340271

theorem tan_double_angle (α : ℝ) (h : Real.tan (π - α) = 2) : Real.tan (2 * α) = 4 / 3 :=
by
  sorry

end tan_double_angle_l340_340271


namespace incorrect_equation_l340_340660

noncomputable def x : ℂ := (-1 + Real.sqrt 3 * Complex.I) / 2
noncomputable def y : ℂ := (-1 - Real.sqrt 3 * Complex.I) / 2

theorem incorrect_equation : x^9 + y^9 ≠ -1 := sorry

end incorrect_equation_l340_340660


namespace mitch_total_scoops_l340_340737

theorem mitch_total_scoops :
  (3 : ℝ) / (1/3 : ℝ) + (2 : ℝ) / (1/3 : ℝ) = 15 :=
by
  sorry

end mitch_total_scoops_l340_340737


namespace cubic_polynomial_Q_l340_340714

noncomputable def Q (x : ℝ) : ℝ := 27 * x^3 - 162 * x^2 + 297 * x - 156

theorem cubic_polynomial_Q {a b c : ℝ} 
  (h_roots : ∀ x, x^3 - 6 * x^2 + 11 * x - 6 = 0 → x = a ∨ x = b ∨ x = c)
  (h_vieta_sum : a + b + c = 6)
  (h_vieta_prod_sum : ab + bc + ca = 11)
  (h_vieta_prod : abc = 6)
  (hQ : Q a = b + c) 
  (hQb : Q b = a + c) 
  (hQc : Q c = a + b) 
  (hQ_sum : Q (a + b + c) = -27) :
  Q x = 27 * x^3 - 162 * x^2 + 297 * x - 156 :=
by { sorry }

end cubic_polynomial_Q_l340_340714


namespace proof_a_lt_b_lt_c_l340_340658

-- Define the conditions
def a : ℝ := (4 / 5) ^ 2.1
def b : ℝ := (4 / 5) ^ -1
def c : ℝ := Real.logBase 2 3

-- Define the theorem to prove
theorem proof_a_lt_b_lt_c : a < b ∧ b < c := by
  sorry

end proof_a_lt_b_lt_c_l340_340658


namespace equilateral_triangle_side_length_l340_340752

open Classical

noncomputable section

variable {P Q R S : Type}

def is_perpendicular_feet (P Q R S : P) : Prop :=
  sorry -- Definition for Q, R, S being the feet of perpendiculars from P

structure EquilateralTriangle (A B C P Q R S : P) where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  AB : ℝ
  (h_perpendicular : is_perpendicular_feet P Q R S)
  (h_area_eq : ∀ (h₁ : PQ = 2) (h₂ : PR = 3) (h₃ : PS = 4), AB = 12 * Real.sqrt 3)

theorem equilateral_triangle_side_length {A B C P Q R S : P} 
    (h_eq_triangle : EquilateralTriangle A B C P Q R S) : h_eq_triangle.AB = 12 * Real.sqrt 3 :=
  by
    cases h_eq_triangle with
    | mk PQ PR PS AB h_perpendicular h_area_eq =>
        apply h_area_eq
        · exact rfl
        · exact rfl
        · exact rfl

end equilateral_triangle_side_length_l340_340752


namespace projective_transformation_affine_l340_340503

theorem projective_transformation_affine (P : ProjectiveTransformation) 
  (h₁ : P.maps_line_at_infinity_to_itself) : P.is_affine :=
sorry

end projective_transformation_affine_l340_340503


namespace sqrt_16_eq_4_l340_340436

theorem sqrt_16_eq_4 : real.sqrt 16 = 4 := by
  sorry

end sqrt_16_eq_4_l340_340436


namespace complete_square_identity_l340_340221

theorem complete_square_identity (x d e : ℤ) (h : x^2 - 10 * x + 15 = 0) :
  (x + d)^2 = e → d + e = 5 :=
by
  intros hde
  sorry

end complete_square_identity_l340_340221


namespace difference_of_extremes_l340_340003

def digits : List ℕ := [2, 0, 1, 3]

def largest_integer : ℕ := 3210
def smallest_integer_greater_than_1000 : ℕ := 1023
def expected_difference : ℕ := 2187

theorem difference_of_extremes :
  largest_integer - smallest_integer_greater_than_1000 = expected_difference := by
  sorry

end difference_of_extremes_l340_340003


namespace cot_tan_sum_l340_340434

theorem cot_tan_sum (h1 : Real.angle)
    (cot_def : ∀ x : Real.Angle, Real.cot x = Real.cos x / Real.sin x)
    (tan_def : ∀ x : Real.Angle, Real.tan x = Real.sin x / Real.cos x)
    (angle_addition_formula : ∀ x y : Real.Angle, Real.cos (x + y) = Real.cos x * Real.cos y - Real.sin x * Real.sin y) :
    Real.cot (20 * π / 180) + Real.tan (10 * π / 180) = Real.csc (20 * π / 180) := 
by
  sorry

end cot_tan_sum_l340_340434


namespace prime_factors_of_difference_l340_340405

theorem prime_factors_of_difference (A B : ℕ) (h_neq : A ≠ B) : 
  ∃ p, Nat.Prime p ∧ p ∣ (Nat.gcd (9 * A - 9 * B + 10) (9 * B - 9 * A - 10)) :=
by
  sorry

end prime_factors_of_difference_l340_340405


namespace problem_statement_l340_340973

-- Definitions derived from conditions
def isMultipleOf (n m : ℕ) : Prop := ∃ k, n = m * k

def construct_numbers : List ℕ :=
  List.range 15 |>.map (λ i, 5^(70 + i) * 7^(84 - i))

theorem problem_statement :
  ∃ (nums : List ℕ), 
    nums = construct_numbers ∧
    (∀ n ∈ nums, isMultipleOf n 35) ∧
    (∀ n m ∈ nums, n ≠ m → ¬(isMultipleOf n m)) ∧
    (∀ n m ∈ nums, (n^6) % (m^5) = 0) :=
by 
  use construct_numbers
  sorry

end problem_statement_l340_340973


namespace distinct_sum_inequality_l340_340295

def is_distinct_sequence (a : ℕ → ℕ) : Prop :=
∀ i j : ℕ, i ≠ j → a i ≠ a j

theorem distinct_sum_inequality (a : ℕ → ℕ) (n : ℕ) (h : is_distinct_sequence a) (h_pos : n > 0) :
  (∑ k in Finset.range n, (a k : ℚ) / (k + 1 : ℚ)^2) ≥ ∑ k in Finset.range n, 1 / (k + 1 : ℚ) := by
  sorry

end distinct_sum_inequality_l340_340295


namespace colored_square_number_possibilities_l340_340484

theorem colored_square_number_possibilities :
  let primes := {2, 3, 5, 7, 11, 13, 17, 19}
  let pairs := { (11, 13), (7, 17), (5, 19) }
  let sum_pair := 24
  let initial := 5
  let total_sum := 192
  let possible_numbers := {7, 11, 13, 17}
  ∀ roll_pattern : list ℕ,
  roll_pattern.length = 8 →
  (∀ x ∈ roll_pattern, x ∈ primes) →
  (∀ x ∈ roll_pattern, ∃ a b, (a, b) ∈ pairs ∧ (x = a ∨ x = b)) →
  list.sum roll_pattern = total_sum →
  roll_pattern.head = initial →
  roll_pattern.nth 7 ∈ possible_numbers := sorry

end colored_square_number_possibilities_l340_340484


namespace knights_prob_sum_l340_340477

theorem knights_prob_sum :
  let n := 25
  let total_ways := Nat.choose n 3
  let adj_all := n
  let adj_two_and_one := n * (n - 5)
  let favorable_ways := adj_all + adj_two_and_one
  let probability := favorable_ways % total_ways
  probability.numerator + probability.denominator =
    113 := 
by
  sorry

end knights_prob_sum_l340_340477


namespace equilateral_triangle_side_length_l340_340757

open Classical

noncomputable section

variable {P Q R S : Type}

def is_perpendicular_feet (P Q R S : P) : Prop :=
  sorry -- Definition for Q, R, S being the feet of perpendiculars from P

structure EquilateralTriangle (A B C P Q R S : P) where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  AB : ℝ
  (h_perpendicular : is_perpendicular_feet P Q R S)
  (h_area_eq : ∀ (h₁ : PQ = 2) (h₂ : PR = 3) (h₃ : PS = 4), AB = 12 * Real.sqrt 3)

theorem equilateral_triangle_side_length {A B C P Q R S : P} 
    (h_eq_triangle : EquilateralTriangle A B C P Q R S) : h_eq_triangle.AB = 12 * Real.sqrt 3 :=
  by
    cases h_eq_triangle with
    | mk PQ PR PS AB h_perpendicular h_area_eq =>
        apply h_area_eq
        · exact rfl
        · exact rfl
        · exact rfl

end equilateral_triangle_side_length_l340_340757


namespace triangle_DEF_perimeter_l340_340378

variables {A B C F D E : Point}
variables {BC : Line}
variables {altitude : Line}
variables {h : Line}
variables [hypotenuse_parallel : EF ∥ BC]

-- Given Points:
-- Points F, D, E lie on sides AB, BC, and CA respectively of triangle ABC
-- Triangle DEF is an isosceles right triangle with EF as the hypotenuse
-- The altitude of triangle ABC passing through A is 10 cm
-- BC has length 30 cm
-- EF is parallel to BC
-- We need to prove the perimeter of triangle DEF is 12√2 + 12

theorem triangle_DEF_perimeter :
  ∀ (A B C F D E : Point),
  ∃ (BC : Line) (altitude : Line),
  (hypotenuse_parallel : EF ∥ BC) →
  (altitude_length : ∀ A BC, length (altitude A BC) = 10) →
  (BC_length : length (segment B C) = 30) →
  (is_isosceles_right_triangle (triangle D E F) (hypotenuse E F)) →
  parallel EF BC →
  length (segment E F) = 12 →
  length (segment D E) = 6 * Real.sqrt 2 →
  length (segment D F) = 6 * Real.sqrt 2 →
  perimeter (triangle D E F) = 12 * Real.sqrt 2 + 12 :=
begin
  sorry
end

end triangle_DEF_perimeter_l340_340378


namespace compute_expression_l340_340935

theorem compute_expression :
  4 * (Real.sin (Float.pi / 3)) - abs (-2 : ℝ) - Real.sqrt 12 + (-1 : ℝ)^2016 = -1 :=
by
  -- The proof will go here.
  sorry

end compute_expression_l340_340935


namespace determinant_of_tan_matrix_l340_340027

theorem determinant_of_tan_matrix
  (A B C : ℝ)
  (h₁ : A = π / 4)
  (h₂ : A + B + C = π)
  : (Matrix.det ![
      ![Real.tan A, 1, 1],
      ![1, Real.tan B, 1],
      ![1, 1, Real.tan C]
    ]) = 2 :=
  sorry

end determinant_of_tan_matrix_l340_340027


namespace total_painting_cost_l340_340529

-- Define the arithmetic sequence for the east side.
def east_side_addresses (n : ℕ) : ℕ := 5 + 7 * (n - 1)

-- Define the arithmetic sequence for the west side.
def west_side_addresses (n : ℕ) : ℕ := 6 + 8 * (n - 1)

-- Define a function to count the number of digits in a number.
def num_digits (n : ℕ) : ℕ := n.digits.length

-- Define a function to calculate the painting cost for a list of house numbers.
def painting_cost (addresses : List ℕ) : ℕ :=
  addresses.map num_digits |> List.sum

theorem total_painting_cost : painting_cost (List.range' 1 26).map east_side_addresses +
                              painting_cost (List.range' 1 26).map west_side_addresses = 123 :=
by
  -- East side costs
  let east_cost := [5, 12, 19, 26, 33, 40, 47, 54, 61, 68, 75, 82, 89, 96, 103, 110, 117, 124, 131, 138, 145, 152, 159, 166, 173].map num_digits |> List.sum
  -- West side costs
  let west_cost := [6, 14, 22, 30, 38, 46, 54, 62, 70, 78, 86, 94, 102, 110, 118, 126, 134, 142, 150, 158, 166, 174, 182, 190, 198].map num_digits |> List.sum
  -- Summing both costs
  have : east_cost = 61 ∧ west_cost = 62 := by sorry
  rw [this.1, this.2]
  exact sorry

end total_painting_cost_l340_340529


namespace woman_wait_time_to_catch_up_l340_340165

noncomputable def man_speed : ℝ := 5 -- in miles per hour
noncomputable def woman_speed : ℝ := 25 -- in miles per hour
noncomputable def wait_time : ℝ := 5 -- in minutes

theorem woman_wait_time_to_catch_up : 
  let man_speed_per_minute := man_speed / 60 in
  let woman_speed_per_minute := woman_speed / 60 in
  let distance := woman_speed_per_minute * wait_time in
  let time := distance / man_speed_per_minute in
  time = 25 := 
by
  sorry

end woman_wait_time_to_catch_up_l340_340165


namespace fraction_values_l340_340335

theorem fraction_values (a b c : ℚ) (h1 : a / b = 2) (h2 : b / c = 4 / 3) : c / a = 3 / 8 := 
by
  sorry

end fraction_values_l340_340335


namespace total_copies_in_half_hour_l340_340518

-- Definitions of the machine rates and their time segments.
def machine1_rate := 35 -- copies per minute
def machine2_rate := 65 -- copies per minute
def machine3_rate1 := 50 -- copies per minute for the first 15 minutes
def machine3_rate2 := 80 -- copies per minute for the next 15 minutes
def machine4_rate1 := 90 -- copies per minute for the first 10 minutes
def machine4_rate2 := 60 -- copies per minute for the next 20 minutes

-- Time intervals for different machines
def machine3_time1 := 15 -- minutes
def machine3_time2 := 15 -- minutes
def machine4_time1 := 10 -- minutes
def machine4_time2 := 20 -- minutes

-- Proof statement
theorem total_copies_in_half_hour : 
  (machine1_rate * 30) + 
  (machine2_rate * 30) + 
  ((machine3_rate1 * machine3_time1) + (machine3_rate2 * machine3_time2)) + 
  ((machine4_rate1 * machine4_time1) + (machine4_rate2 * machine4_time2)) = 
  7050 :=
by 
  sorry

end total_copies_in_half_hour_l340_340518


namespace period_sin_x_plus_2cos_x_l340_340844

def period_of_sin_x_plus_2_cos_x : Prop :=
  ∃ (T : ℝ), T > 0 ∧ ∀ x : ℝ, sin x + 2 * cos x = sin (x + T)

theorem period_sin_x_plus_2cos_x : period_of_sin_x_plus_2_cos_x :=
by
  sorry

end period_sin_x_plus_2cos_x_l340_340844


namespace original_price_of_sarees_l340_340811

theorem original_price_of_sarees 
  (P : ℝ)
  (h1 : 0.98 * 0.95 * P = 502.74) : 
  P ≈ 540 :=
begin
  sorry
end

end original_price_of_sarees_l340_340811


namespace modulus_z_l340_340623

theorem modulus_z (z : ℂ) (h : (2 - 3 * complex.I) * z = 3 + 2 * complex.I) : complex.abs z = 1 := 
sorry

end modulus_z_l340_340623


namespace cube_assembly_possible_l340_340112

theorem cube_assembly_possible (n : ℕ) (h : n ≥ 23) : 
  let small_cubes := n^3 in 
  let large_cube := (2 * n)^3 in 
  let hollow_cube := (2 * n - 2)^3 in
  small_cubes ≥ large_cube - hollow_cube := 
by
  sorry

end cube_assembly_possible_l340_340112


namespace geometric_progression_common_ratio_l340_340665

theorem geometric_progression_common_ratio (x y z w r : ℂ) 
  (h_distinct : x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w)
  (h_nonzero : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ w ≠ 0)
  (h_geom : x * (y - w) = a ∧ y * (z - x) = a * r ∧ z * (w - y) = a * r^2 ∧ w * (x - z) = a * r^3) :
  1 + r + r^2 + r^3 = 0 :=
sorry

end geometric_progression_common_ratio_l340_340665


namespace determine_b_l340_340082

theorem determine_b (b : ℝ) : 
  (∀ x : ℝ, f x = if x < 1 then 3 * x - b else 2^x) ∧
  f (f (5 / 6)) = 4 → b = 1 / 2 := 
by
  let f (x : ℝ) := if x < 1 then 3 * x - b else 2^x
  sorry

end determine_b_l340_340082


namespace banana_nn_together_count_l340_340567

open Finset

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def arrangements_banana_with_nn_together : ℕ :=
  (factorial 4) / (factorial 3)

theorem banana_nn_together_count : arrangements_banana_with_nn_together = 4 := by
  sorry

end banana_nn_together_count_l340_340567


namespace cyclic_X_CQP_l340_340725

universe u
variables {Point : Type u} [MetricSpace Point]

open EuclideanGeometry

noncomputable def triangle := (A B C : Point) : Prop

noncomputable def on_segment (P Q : Point) (R : Point) : Prop := sorry -- TODO: on_segment implementation 

noncomputable def circumcircle (A B C : Point) : Set Point := sorry -- TODO: circumcircle definition

noncomputable def cyclic_quad (A B C D : Point) : Prop := sorry -- TODO: cyclic quadrilateral definition

theorem cyclic_X_CQP
  (A B C P Q R X : Point)
  (h_triangle : triangle A B C)
  (hP : on_segment B C P)
  (hQ : on_segment C A Q)
  (hR : on_segment A B R)
  (hX_circumcircle_AQR : X ∈ circumcircle A Q R)
  (hX_circumcircle_BRP : X ∈ circumcircle B R P) :
  X ∈ circumcircle C Q P := by
  sorry

end cyclic_X_CQP_l340_340725


namespace probability_correct_l340_340591

noncomputable def probability_drawing_ball_3_three_times : ℚ :=
  if (∃ (draws : list ℕ), draws.length = 3 ∧ (∀ x ∈ draws, x ∈ {1, 2, 3, 4}) ∧ (draws.sum = 9)) then 1 / 13 else 0

theorem probability_correct : probability_drawing_ball_3_three_times = 1 / 13 :=
by
  sorry

end probability_correct_l340_340591


namespace mitch_total_scoops_l340_340736

theorem mitch_total_scoops :
  (3 : ℝ) / (1/3 : ℝ) + (2 : ℝ) / (1/3 : ℝ) = 15 :=
by
  sorry

end mitch_total_scoops_l340_340736


namespace platform_length_l340_340904

theorem platform_length (train_length : ℝ) (speed_kmph : ℝ) (time_sec : ℝ) (platform_length : ℝ) :
  train_length = 150 ∧ speed_kmph = 75 ∧ time_sec = 20 →
  platform_length = 1350 :=
by
  sorry

end platform_length_l340_340904


namespace max_value_of_exp_diff_l340_340246

open Real

theorem max_value_of_exp_diff : ∀ x : ℝ, ∃ y : ℝ, y = 2^x - 4^x ∧ y ≤ 1/4 := sorry

end max_value_of_exp_diff_l340_340246


namespace _l340_340727

noncomputable theorem maximum_value_inv_sum 
  (x y z : ℝ)
  (hx : x > 0)
  (hy : y > 0)
  (hz : z > 0)
  (hxyz : x + y + z ≤ 1) :
  ∃ (M : ℝ), M = ∞ ∧ ∀ x y z : ℝ, 
  (x > 0) → (y > 0) → (z > 0) → 
  (x + y + z ≤ 1) → 
  \frac{1}{x} + \frac{1}{y} + \frac{1}{z} ≤ M := 
sorry

end _l340_340727


namespace trapezoid_inequality_l340_340906

theorem trapezoid_inequality (a b R : ℝ) (h : a > 0) (h1 : b > 0) (h2 : R > 0) 
  (circumscribed : ∃ (x y : ℝ), x + y = a ∧ R^2 * (1/x + 1/y) = b) : 
  a * b ≥ 4 * R^2 :=
by
  sorry

end trapezoid_inequality_l340_340906


namespace functional_relationship_maximize_profit_l340_340500

-- Definitions based on conditions
def profit_A : ℕ → ℤ  := fun (x : ℕ), 60 * x
def profit_B : ℕ → ℤ  := fun (x : ℕ), 80 * (40 - x)
def total_profit : ℕ → ℤ  := fun (x : ℕ), profit_A x + profit_B x
def constraint (x : ℕ) : Prop := (40 - x) ≤ 3 * x

-- Theorems to be proven
theorem functional_relationship : 
  ∀ (x : ℕ), total_profit x = -20 * x + 3200 := 
by 
  intros x 
  unfold total_profit profit_A profit_B 
  simp
  sorry

theorem maximize_profit : 
  ∃ (x : ℕ), constraint x ∧ ( ∀ y : ℕ, constraint y → total_profit y ≤ 3000) :=
by 
  use 10
  unfold constraint
  split
  {
    sorry
  },
  {
    intros y hy
    sorry
  }

end functional_relationship_maximize_profit_l340_340500


namespace magnitude_of_b_l340_340638

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2)

theorem magnitude_of_b (b : ℝ × ℝ) 
  (h1 : (2, 3) = (2, 3)) 
  (h2 : (2 + b.1, 3 + b.2) ⬝ (2 - b.1, 3 - b.2) = 0) :
  magnitude b = real.sqrt 13 :=
by
  sorry

end magnitude_of_b_l340_340638


namespace store_loss_l340_340178

noncomputable def calculation (x y : ℕ) : ℤ :=
  let revenue : ℕ := 60 * 2
  let cost : ℕ := x + y
  revenue - cost

theorem store_loss (x y : ℕ) (hx : (60 - x) * 2 = x) (hy : (y - 60) * 2 = y) :
  calculation x y = -40 := by
    sorry

end store_loss_l340_340178


namespace perpendicular_slope_l340_340208

theorem perpendicular_slope (x1 y1 x2 y2 : ℝ) (h1 : x1 = 8) (h2 : y1 = -3) (h3 : x2 = -2) (h4 : y2 = 10) :
  let m := (y2 - y1) / (x2 - x1) in
  let perpendicular_m := -1 / m in
  perpendicular_m = (10 : ℝ) / (13 : ℝ) :=
by
  rw [h1, h2, h3, h4]
  let m := (10 - (-3)) / (-2 - 8)
  let perpendicular_m := -1 / m
  show perpendicular_m = (10 : ℝ) / (13 : ℝ)
  sorry

end perpendicular_slope_l340_340208


namespace arnold_optimal_guess_l340_340916

theorem arnold_optimal_guess (m : ℕ) (h₁ : 1 ≤ m) (h₂ : m ≤ 1001) : 
  let k := 859 in 
  ∀ k, (1 ≤ k ∧ k ≤ 1001) → 
    (if m ≥ k then m - k else 10 + k) 
    ≤ (if m ≥ 859 then m - 859 else 10 + 859) := 
sorry

end arnold_optimal_guess_l340_340916


namespace closure_of_A_range_of_a_l340_340402

-- Definitions for sets A and B
def A (x : ℝ) : Prop := x < -1 ∨ x > -0.5
def B (x a : ℝ) : Prop := a - 1 ≤ x ∧ x ≤ a + 1

-- 1. Closure of A
theorem closure_of_A :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ -0.5) ↔ (∀ x : ℝ, A x) :=
sorry

-- 2. Range of a when A ∪ B = ℝ
theorem range_of_a (B_condition : ∀ x : ℝ, B x a) :
  (∀ a : ℝ, -1 ≤ x ∨ x ≥ -0.5) ↔ (-1.5 ≤ a ∧ a ≤ 0) :=
sorry

end closure_of_A_range_of_a_l340_340402


namespace right_triangle_perimeter_l340_340069

theorem right_triangle_perimeter (S m : ℝ) : 
  ∃ (a b c : ℝ), (1/2)*a*b = S ∧ c = 2*m ∧ a^2 + b^2 = c^2 ∧
  a + b + c = sqrt(4*m^2 + 4*S) + 2*m :=
by
  sorry

end right_triangle_perimeter_l340_340069


namespace lcm_gcd_product_l340_340041

theorem lcm_gcd_product (a b : ℕ) (ha : a = 12) (hb : b = 9) :
  Nat.lcm a b * Nat.gcd a b = 108 := by
  rw [ha, hb]
  -- Replace with Nat library functions and calculate
  sorry

end lcm_gcd_product_l340_340041


namespace kristy_gave_to_brother_l340_340013

def total_cookies : Nat := 22
def kristy_ate : Nat := 2
def first_friend_took : Nat := 3
def second_friend_took : Nat := 5
def third_friend_took : Nat := 5
def cookies_left : Nat := 6

theorem kristy_gave_to_brother :
  kristy_ate + first_friend_took + second_friend_took + third_friend_took = 15 ∧
  total_cookies - cookies_left - (kristy_ate + first_friend_took + second_friend_took + third_friend_took) = 1 :=
by
  sorry

end kristy_gave_to_brother_l340_340013


namespace diagonal_length_count_l340_340589

theorem diagonal_length_count {P Q R S : Type} [metric_space P] 
  (hPQ : dist P Q = 7) (hQR : dist Q R = 9)
  (hRS : dist R S = 14) (hSP : dist S P = 10) :
  ∃ (y : ℕ), y \in finset.range 17 ∧ ∀ y, (5 ≤ y ∧ y ≤ 15) ↔ (y = 5 ∨ y = 6 ∨ y = 7 ∨ y = 8 ∨ y = 9 ∨ y = 10 ∨ y = 11 ∨ y = 12 ∨ y = 13 ∨ y = 14 ∨ y = 15) :=
begin
  sorry
end

end diagonal_length_count_l340_340589


namespace ratio_lcm_gcf_l340_340488

theorem ratio_lcm_gcf (a b : ℕ) (h₁ : a = 252) (h₂ : b = 675) : 
  let lcm_ab := Nat.lcm a b
  let gcf_ab := Nat.gcd a b
  (lcm_ab / gcf_ab) = 2100 :=
by
  sorry

end ratio_lcm_gcf_l340_340488


namespace fibonacci_pair_l340_340778

-- Define Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| n + 2 => fib n + fib (n + 1)

-- Prove that if a^2 - ab - b^2 = ±1, then a and b are consecutive Fibonacci numbers
theorem fibonacci_pair (a b : ℕ) (h : a^2 - a * b - b^2 = 1 ∨ a^2 - a * b - b^2 = -1) :
  ∃ n : ℕ, a = fib (n + 1) ∧ b = fib n := by
  sorry

end fibonacci_pair_l340_340778


namespace broken_perfect_spiral_shells_difference_l340_340645

theorem broken_perfect_spiral_shells_difference :
  let perfect_shells := 17
  let broken_shells := 52
  let broken_spiral_shells := broken_shells / 2
  let not_spiral_perfect_shells := 12
  let spiral_perfect_shells := perfect_shells - not_spiral_perfect_shells
  broken_spiral_shells - spiral_perfect_shells = 21 := by
  sorry

end broken_perfect_spiral_shells_difference_l340_340645


namespace diamond_evaluation_l340_340988

def diamond (a b : ℝ) : ℝ := (a + b) * (a - b)

theorem diamond_evaluation : diamond 2 (diamond 3 (diamond 1 4)) = -46652 :=
  by
  sorry

end diamond_evaluation_l340_340988


namespace minimum_groups_l340_340990

noncomputable def num_groups : ℕ := 9

theorem minimum_groups {A B C D : Type} [fintype A] [fintype B] [fintype C] [fintype D] 
  (hA : fintype.card A = 3) (hB: fintype.card B = 3) (hC: fintype.card C = 3) (hD: fintype.card D = 3)
  (no_same_school : ∀ a a' ∈ (A ∪ B ∪ C ∪ D), a ≠ a' → ∀ g ∈ groups, a ∈ g → a' ∉ g)
  (one_group : ∀ (a ∈ A) (b ∈ B) (c ∈ C) (d ∈ D), ∃ g ∈ groups, a ∈ g ∧ b ∈ g ∧ c ∈ g ∧ d ∈ g)
  : ∃ n = num_groups, n = 9 :=
begin
  sorry
end

end minimum_groups_l340_340990


namespace frac_XQ_YQ_eq_4_l340_340002

-- Definitions of the conditions
variable (X Y Z Q : Type)
variables (Angle : Type) [LinearOrder Angle]
variables (angle_YXZ angle_XQZ angle_XZQ : Angle)
variable (XY XQ ZQ : ℝ)
variable [Triangle XYZ : Type]

-- Assumptions from the conditions
axiom angle_YXZ_90 : angle_YXZ = 90
axiom angle_XYZ_lt_45 : angle_YXZ < 45
axiom XY_eq_5 : XY = 5
axiom angle_XQZ_eq_2XZQ : angle_XQZ = 2 * angle_XZQ
axiom ZQ_eq_2 : ZQ = 2

-- Theorem to prove
theorem frac_XQ_YQ_eq_4 : (XQ / (XY - XQ)) = 4 :=
by sorry

end frac_XQ_YQ_eq_4_l340_340002


namespace not_obtain_other_than_given_set_l340_340340

theorem not_obtain_other_than_given_set : 
  ∀ (x : ℝ), x = 1 → 
  ∃ (n : ℕ → ℝ), (n 0 = 1) ∧ 
  (∀ k, n (k + 1) = n k + 1 ∨ n (k + 1) = -1 / n k) ∧
  (x = -2 ∨ x = 1/2 ∨ x = 5/3 ∨ x = 7) → 
  ∃ k, x = n k :=
sorry

end not_obtain_other_than_given_set_l340_340340


namespace C_completes_work_in_100_days_l340_340150

theorem C_completes_work_in_100_days :
  (∀ (A B : ℝ) (total_work : ℝ), 
   A = 1 / 20 ∧ B = 1 / 15 → -- work rates of A and B
   total_work = 1 → -- total work to be done
   (∀ (t1 t2: ℝ) (W1 W2: ℝ), 
     t1 = 6 ∧ -- A and B work together for 6 days
     t2 = 5 ∧ -- remaining work is done in next 5 days
     W1 = 7 / 10 ∧ -- work completed by A and B in first 6 days
     W2 = 3 / 10 ∧ -- remaining work
     (∀ (rate_A rate_C : ℝ),
       rate_A = 1 / 20 ∧ -- A's work rate
       rate_C = (W2 - (rate_A * t2)) / t2 → -- C's work rate over 5 days
       1 / rate_C = 100) -- C would take 100 days to complete the work alone
  )
:= sorry

end C_completes_work_in_100_days_l340_340150


namespace sum_of_squares_second_15_l340_340505

theorem sum_of_squares_second_15 :
  (∑ k in Finset.range (30 + 1), k^2) - (∑ k in Finset.range (15 + 1), k^2) = 8215 :=
by
  have h₁ : ∑ k in Finset.range (15 + 1), k^2 = 1240 := sorry
  have h₂ : ∑ k in Finset.range (30 + 1), k^2 = (30 * (30 + 1) * (2 * 30 + 1)) / 6 := sorry
  have h₃ : ∑ k in Finset.range (30 + 1), k^2 - ∑ k in Finset.range 16, k^2 = ∑ k in Finset.range (30 + 1) \ Finset.range 16, k^2 := sorry
  rw Finset.range_add_inter at h₃
  sorry

end sum_of_squares_second_15_l340_340505


namespace distance_to_lake_l340_340193

theorem distance_to_lake 
  {d : ℝ} 
  (h1 : ¬ (d ≥ 8))
  (h2 : ¬ (d ≤ 7))
  (h3 : ¬ (d ≤ 6)) : 
  (7 < d) ∧ (d < 8) :=
by
  sorry

end distance_to_lake_l340_340193


namespace units_digit_product_l340_340124

theorem units_digit_product :
  let nums : List Nat := [7, 17, 27, 37, 47, 57, 67, 77, 87, 97]
  let product := nums.prod
  (product % 10) = 9 :=
by
  sorry

end units_digit_product_l340_340124


namespace house_orderings_count_l340_340110

theorem house_orderings_count :
  let houses := ["P", "G", "B", "L", "Y"]
  ∃ (orders : List (List String)), 
    (∀ o ∈ orders, 
      (List.indexOf "G" o > List.indexOf "B" o) ∧
      (List.indexOf "P" o > List.indexOf "G" o) ∧
      (List.indexOf "L" o > List.indexOf "Y" o) ∧
      (List.indexOf "L" o + 1 ≠ List.indexOf "Y" o) ∧
      (List.indexOf "Y" o + 1 ≠ List.indexOf "L" o )) ∧
    (orders.length = 6)
:=
begin
  sorry
end

end house_orderings_count_l340_340110


namespace omit_infinitely_many_digits_l340_340369

noncomputable def is_nonterminating_nonrepeating (digits : Nat → Nat) : Prop :=
  ∀ N : Nat, ∃ M : Nat, M > N ∧ ∀ k : Nat, k > M → digits k ≠ digits (k + 1)

theorem omit_infinitely_many_digits (A : ℝ) (digits : ℕ → ℕ) 
  (h1 : A > 0) 
  (h2 : irrational A) 
  (h3 : ∀ n : ℕ, digits n = (int.fract(A) * 10^n).to_nat % 10) 
  (h4 : is_nonterminating_nonrepeating digits) :
  ∃ s : ℕ → ℕ, (∀ i : ℕ, digits i ≠ 0 → ∃ k > i, s k = digits i) → (decimal s).sum = A := 
sorry

end omit_infinitely_many_digits_l340_340369


namespace calculate_expression_l340_340296

theorem calculate_expression (f : ℕ → ℝ) (h1 : ∀ a b, f (a + b) = f a * f b) (h2 : f 1 = 2) : 
  (f 2 / f 1) + (f 4 / f 3) + (f 6 / f 5) = 6 := 
sorry

end calculate_expression_l340_340296


namespace discount_is_seventy_percent_l340_340808

-- Define the conditions
def firstReduction (P : ℝ) := 0.5 * P
def secondReduction (firstPrice : ℝ) := 0.6 * firstPrice

-- Define the final discount calculation
def totalDiscount (P thirdPrice : ℝ) := (P - thirdPrice) / P * 100

-- The theorem to prove that the total discount is 70%
theorem discount_is_seventy_percent (P : ℝ) (hP : 0 < P) :
  totalDiscount P (secondReduction (firstReduction P)) = 70 :=
by
  -- the actual proof steps would go here, skipped for now
  sorry

end discount_is_seventy_percent_l340_340808


namespace number_of_distinct_circles_l340_340021

theorem number_of_distinct_circles (S : set (ℝ × ℝ)) (hS : ∃ A B C D : ℝ × ℝ, 
  A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧ (A = (0, 0)) ∧ (B = (1, 0)) ∧ (C = (1, 1)) ∧ (D = (0, 1))) :
  ∃ circles : set (set (ℝ × ℝ)), circles.card = 5 ∧ 
    ∀ ab ∈ {{A, B}, {B, C}, {C, D}, {D, A}, {A, C}, {B, D}}, 
      ∃ circle : set (ℝ × ℝ), circle ∈ circles ∧ 
        ∃ x y : (ℝ × ℝ), x ∈ S ∧ y ∈ S ∧ ab = {x, y} ∧ (∃ d : ℝ, (dist x y) = 2 * d) := sorry

end number_of_distinct_circles_l340_340021


namespace union_setA_setB_l340_340401

noncomputable def setA : Set ℝ := { x : ℝ | 2 / (x + 1) ≥ 1 }
noncomputable def setB : Set ℝ := { y : ℝ | ∃ x : ℝ, y = 2^x ∧ x < 0 }

theorem union_setA_setB : setA ∪ setB = { x : ℝ | -1 < x ∧ x ≤ 1 } :=
by
  sorry

end union_setA_setB_l340_340401


namespace walking_time_l340_340644

-- Define the conditions as Lean definitions
def minutes_in_hour : Nat := 60

def work_hours : Nat := 6
def work_minutes := work_hours * minutes_in_hour
def sitting_interval : Nat := 90
def walking_time_per_interval : Nat := 10

-- State the main theorem
theorem walking_time (h1 : 10 * 90 = 600) (h2 : 10 * (work_hours * 60) / 90 = 40) : 
  work_minutes / sitting_interval * walking_time_per_interval = 40 :=
  sorry

end walking_time_l340_340644


namespace tan_sum_of_roots_l340_340386

theorem tan_sum_of_roots (alpha β : ℝ) (h1 : ∃ x y, (x = tan α ∧ y = tan β) ∧ (x^2 - 3 * x + 2 = 0) ∧ (y^2 - 3 * y + 2 = 0)) :
  tan (alpha + β) = -3 := 
sorry

end tan_sum_of_roots_l340_340386


namespace average_time_per_cut_l340_340907

theorem average_time_per_cut (segments : ℕ) (total_time : ℕ) 
  (H_segments : segments = 5) (H_total_time : total_time = 20) : 
  total_time / (segments - 1) = 5 :=
by
  rw [H_segments, H_total_time]
  sorry

end average_time_per_cut_l340_340907


namespace total_number_of_stops_is_16_l340_340232

-- Each Tuesday, a bus makes its first stop at Gauss Public Library at 1 p.m.
def first_stop := 1

-- The bus continues to stop at the library every 20 minutes.
def stop_interval := 20

-- The bus's last stop is at 6 p.m.
def last_stop := 18

-- Function to calculate total stops
def total_stops : Nat :=
  let hours := last_stop - first_stop
  let stops_per_hour := (60 / stop_interval)
  (hours * stops_per_hour + 1).toNat

-- Prove that the total number of stops is 16.
theorem total_number_of_stops_is_16 :
  total_stops = 16 :=
by
  sorry

end total_number_of_stops_is_16_l340_340232


namespace right_triangle_third_side_length_l340_340671

theorem right_triangle_third_side_length:
  ∀ (a b : ℝ), (sqrt (a^2 - 6 * a + 9) + abs (b - 4) = 0) →
  (a = 3) →
  (b = 4) →
  ((hypotenuse : ℝ) → (hypotenuse = 5 ∨ hypotenuse = sqrt 7)) :=
by
  intros a b h1 h2 h3 hypotenuse
  sorry

end right_triangle_third_side_length_l340_340671


namespace geometric_series_sum_is_correct_l340_340560

-- Define the first term, common ratio, and number of terms
def a : ℤ := 3
def r : ℤ := -2
def n : ℤ := 8

-- Define the formula for the sum of the geometric series
def geometric_series_sum (a r : ℤ) (n : ℤ) : ℤ :=
  a * (r^n - 1) / (r - 1)

-- State the theorem that the sum of the geometric series is -255
theorem geometric_series_sum_is_correct : geometric_series_sum a r n = -255 := by
  sorry

end geometric_series_sum_is_correct_l340_340560


namespace problem1_and_problem2_l340_340693

noncomputable theory
open Real

-- Given conditions
variables (A B C : ℝ) (a b : ℝ) (AC CD : ℝ)
variables [triangle_ABC : triangle A B C]
variables [AC_eq : AC = 2]
variables [CD_eq : CD = √3]
variables (D : point) [midpoint_D : midpoint D (AB : line A B)]

-- The main statement showing the given conditions lead to the required conclusions
theorem problem1_and_problem2 
  (h1 : 2 * sin A * cos B + b * sin (2 * A) + 2 * √3 * a * cos C = 0)
  (h2 : AC = 2)
  (h3 : CD = √3)
  (midp : D.is_midpoint A B) :
  (C = 2 * π / 3 ∧ 
   area B C D = √3) :=
by {
  sorry
}

end problem1_and_problem2_l340_340693


namespace valid_sequences_count_l340_340384

-- Definitions for the vertices of the triangle
def T_vert1 := (0, 0)
def T_vert2 := (6, 0)
def T_vert3 := (0, 4)

-- Transformations represented as functions on points in the plane
def rotate_120 (P : ℝ × ℝ) : ℝ × ℝ := (-P.2, P.1 + P.2)
def rotate_240 (P : ℝ × ℝ) : ℝ × ℝ := (P.2 - P.1, -P.1)
def reflect_y_eq_x (P : ℝ × ℝ) : ℝ × ℝ := (P.2, P.1)
def scale_2 (P : ℝ × ℝ) : ℝ × ℝ := (2 * P.1, 2 * P.2)

-- Type for sequences of three transformations
def Transformations := List ((ℝ × ℝ) → (ℝ × ℝ))

-- Applying a sequence of transformations to a point
def apply_transformations (seq : Transformations) (P : ℝ × ℝ) : ℝ × ℝ :=
  seq.foldl (λ Q f, f Q) P

-- Predicate to check if a sequence of transformations returns the triangle to its original shape and orientation
def preserves_shape_and_orientation (seq : Transformations) : Prop :=
  let P1 := apply_transformations seq T_vert1
  let P2 := apply_transformations seq T_vert2
  let P3 := apply_transformations seq T_vert3
  (T_vert1 = P1) ∧
  (P1.1 - P2.1) * (T_vert3.2 - T_vert2.2) = (P1.2 - P2.2) * (T_vert3.1 - T_vert2.1) ∧
  (P1.1 - P3.1) * (T_vert2.2 - T_vert3.2) = (P1.2 - P3.2) * (T_vert2.1 - T_vert3.1)

-- The set of all transformations
def all_transformations : List ((ℝ × ℝ) → (ℝ × ℝ)) :=
  [rotate_120, rotate_240, reflect_y_eq_x, scale_2]

-- The main proof statement: there are exactly 15 valid sequences
theorem valid_sequences_count : (all_transformations^3).filter preserves_shape_and_orientation |>.length = 15 := sorry

end valid_sequences_count_l340_340384


namespace inequality_proof_l340_340398

theorem inequality_proof (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 4) : 
  (a + 2) * (b + 2) ≥ c * d := 
sorry

def equality_case : 
  ∃ a b c d : ℝ, 
  a^2 + b^2 + c^2 + d^2 = 4 ∧ 
  (a + 2) * (b + 2) = c * d := 
begin
  use [-1, -1, -1, -1],
  split,
  { norm_num },
  { norm_num },
end

end inequality_proof_l340_340398


namespace sum_of_faces_edges_vertices_l340_340490

def pentagonal_prism_faces : ℕ := 7
def pentagonal_prism_edges : ℕ := 15
def pentagonal_prism_vertices : ℕ := 10

theorem sum_of_faces_edges_vertices (F E V : ℕ) (h1: F = 7) (h2: E = 15) (h3: V = 10) :
  F + E + V = 32 :=
by
  rw [h1, h2, h3]
  exact rfl

end sum_of_faces_edges_vertices_l340_340490


namespace addition_of_smallest_multiples_l340_340657

theorem addition_of_smallest_multiples :
  (let a := 10 in let b := 105 in a + b = 115) :=
by
  let a := 10
  let b := 105
  exact rfl

end addition_of_smallest_multiples_l340_340657


namespace inequality_positive_numbers_l340_340426

theorem inequality_positive_numbers (x y : ℝ) 
  (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy : x < y) : 
  x + real.sqrt (y^2 + 2) < y + real.sqrt (x^2 + 2) :=
sorry

end inequality_positive_numbers_l340_340426


namespace remaining_integers_in_S_after_removals_l340_340812

-- Define the set of the first 100 positive integers
def S : Set ℕ := {n | n ≥ 1 ∧ n ≤ 100}

-- Define the set of multiples of 2
def multiples_of_2 : Set ℕ := {n | n ∈ S ∧ n % 2 = 0}

-- Define the set of multiples of 3
def multiples_of_3 : Set ℕ := {n | n ∈ S ∧ n % 3 = 0}

-- Define the set of multiples of 5
def multiples_of_5 : Set ℕ := {n | n ∈ S ∧ n % 5 = 0}

-- Define the set S' which is S after removing all multiples of 2, 3, and 5
def S' : Set ℕ := S \ (multiples_of_2 ∪ multiples_of_3 ∪ multiples_of_5)

-- Prove the number of integers remaining in S' is 26
theorem remaining_integers_in_S_after_removals : S'.card = 26 :=
by sorry

end remaining_integers_in_S_after_removals_l340_340812


namespace chapter_page_difference_l340_340873

theorem chapter_page_difference (page_ch1 : Nat) (page_ch2 : Nat) (h1 : page_ch1 = 48) (h2 : page_ch2 = 11) : page_ch1 - page_ch2 = 37 :=
by
  rw [h1, h2]
  rw [Nat.sub_self]
  sorry

end chapter_page_difference_l340_340873


namespace tangent_slope_at_point_l340_340304

open Function

theorem tangent_slope_at_point : 
  ∀ (f : ℝ → ℝ) (f' : ℝ → ℝ) (x0 : ℝ) (y0 : ℝ),
  f = (λ x, 0.5 * x^2 - 3) →
  f' = (λ x, x) →
  f x0 = y0 →
  x0 = 1 →
  y0 = -5/2 →
  f' x0 = 1 :=
by
  intros f f' x0 y0 hf hf' hx0 hy0 hxy0
  rw [hf', hxy0]
  norm_num
  sorry

end tangent_slope_at_point_l340_340304


namespace dana_more_pencils_than_marcus_l340_340946

theorem dana_more_pencils_than_marcus :
  ∀ (Jayden Dana Marcus : ℕ), 
  (Jayden = 20) ∧ 
  (Dana = Jayden + 15) ∧ 
  (Jayden = 2 * Marcus) → 
  (Dana - Marcus = 25) :=
by
  intros Jayden Dana Marcus h
  rcases h with ⟨hJayden, hDana, hMarcus⟩
  sorry

end dana_more_pencils_than_marcus_l340_340946


namespace min_sum_of_factors_l340_340462

theorem min_sum_of_factors (a b c : ℕ) (h1 : a * b * c = 1806) (h2 : a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0) : a + b + c ≥ 112 :=
sorry

end min_sum_of_factors_l340_340462


namespace find_x_when_y_neg_10_l340_340789

def inversely_proportional (x y : ℝ) (k : ℝ) := x * y = k

theorem find_x_when_y_neg_10 (k : ℝ) (h₁ : inversely_proportional 4 (-2) k) (yval : y = -10) 
: ∃ x, inversely_proportional x y k ∧ x = 4 / 5 := by
  sorry

end find_x_when_y_neg_10_l340_340789


namespace hyperbola_properties_l340_340277

theorem hyperbola_properties (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) 
    (focus : ℝ × ℝ := (3, 0)) (asymptote_tangent : ℝ := 2 * Real.sqrt 2)
    (c := Real.sqrt (a^2 + b^2) := 3) (asymptote_condition : b / a = 2 * Real.sqrt 2) :
  (∃ (a b : ℝ), a^2 = 1 ∧ b^2 = 8 ∧ (∀ (x y : ℝ), x^2 - y^2 / 8 = 1)) ∧
  (∃ (k m : ℝ) (l_instruction : k ≠ 2 * Real.sqrt 2 ∧ m ≠ 0),
    ∃ (triangle_area : ℝ), triangle_area = 2 * Real.sqrt 2) :=
by
  sorry

end hyperbola_properties_l340_340277


namespace max_functional_value_l340_340510

noncomputable def kernel (x t : ℝ) : ℝ := cos x * cos (2 * t) + cos t * cos (2 * x) + 1

noncomputable def functional (K : ℝ → ℝ → ℝ) (ϕ : ℝ → ℝ) : ℝ :=
  |∫ x in 0..π, (∫ t in 0..π, K x t * ϕ x * ϕ t) dt|

noncomputable def norm (ϕ : ℝ → ℝ) : ℝ :=
  ∫ x in 0..π, (ϕ x)^2

noncomputable def maxFunctionalValue : ℝ :=
  2 * π

theorem max_functional_value {ϕ : ℝ → ℝ} (h_norm : norm ϕ = 1) :
  functional kernel ϕ ≤ maxFunctionalValue :=
sorry

end max_functional_value_l340_340510


namespace problem_solution_l340_340298

theorem problem_solution (P : ℝ) (h : P = (∛6 * ∛(1/162))⁻¹) : P = 3 :=
by
  sorry

end problem_solution_l340_340298


namespace arc_length_l340_340795

-- Given conditions
def circumference (D : Type) [MetricSpace D] : ℝ := 100
def central_angle (angle_EDF : ℝ) : Prop := angle_EDF = 45

-- Statement to prove
theorem arc_length (D : Type) [MetricSpace D] (angle_EDF : ℝ) 
  (h_circumference : circumference D = 100) (h_angle : central_angle angle_EDF) :
  let fraction_of_circle := angle_EDF / 360 in
  let arc_length := fraction_of_circle * circumference D in
  arc_length = 12.5 :=
by
  -- This 'sorry' is used to skip the actual proof
  sorry

end arc_length_l340_340795


namespace books_combination_l340_340181

theorem books_combination :
  (Nat.choose 15 3) = 455 := 
sorry

end books_combination_l340_340181


namespace period_sin_cos_l340_340848

theorem period_sin_cos (x : ℝ) : ∀ y : ℝ, y = sin x + 2 * cos x → ∃ T > 0, ∀ t : ℝ, y = sin (x + t + T) + 2 * cos (x + t + T) := 
sorry

end period_sin_cos_l340_340848


namespace vector_t_perpendicular_l340_340326

theorem vector_t_perpendicular (t : ℝ) :
  let a := (2, 4)
  let b := (-1, 1)
  let c := (2 + t, 4 - t)
  b.1 * c.1 + b.2 * c.2 = 0 → t = 1 := by
  sorry

end vector_t_perpendicular_l340_340326


namespace sixth_year_fee_l340_340192

def first_year_fee : ℕ := 80
def yearly_increase : ℕ := 10

def membership_fee (year : ℕ) : ℕ :=
  first_year_fee + (year - 1) * yearly_increase

theorem sixth_year_fee : membership_fee 6 = 130 :=
  by sorry

end sixth_year_fee_l340_340192


namespace y_min_max_l340_340996

noncomputable def y (x : ℝ) : ℝ := x⁻¹ - 4 * x + 2

theorem y_min_max {x : ℝ} (hx : 9^x - 10 * 3^x + 9 ≤ 0) : 
  (∀ x, 0 ≤ x ∧ x ≤ 2 → y x ≥ 1) ∧ (∀ x, 0 ≤ x ∧ x ≤ 2 → y x ≤ 2) :=
by
  sorry

end y_min_max_l340_340996


namespace equivalent_proof_problem_l340_340921
noncomputable def calculateExpression : ℝ :=
  [(-2 : ℝ)^6]^(1/3) - (-1)^0 + 3^(1 - Real.logb 3 6)

theorem equivalent_proof_problem :
  calculateExpression = (7 / 2 : ℝ) :=
  by
  sorry

end equivalent_proof_problem_l340_340921


namespace total_insects_eaten_is_159_l340_340674

/-- Number of geckos -/
def geckos := 5
/-- Number of lizards -/
def lizards := 3
/-- Number of chameleons -/
def chameleons := 4
/-- Number of iguanas -/
def iguanas := 2

/-- Insects eaten by each gecko -/
def insects_per_gecko := 6

/-- Insects eaten by each lizard -/
def insects_per_lizard := 2 * insects_per_gecko

/-- Insects eaten by each chameleon -/
def insects_per_chameleon := 3.5 * insects_per_gecko

/-- Insects eaten by each iguana -/
def insects_per_iguana := 0.75 * insects_per_gecko

/-- Total insects eaten by all reptiles -/
def total_insects_eaten := 
  geckos * insects_per_gecko + 
  lizards * insects_per_lizard + 
  chameleons * insects_per_chameleon + 
  iguanas * insects_per_iguana

theorem total_insects_eaten_is_159 : total_insects_eaten = 159 := by
  sorry

end total_insects_eaten_is_159_l340_340674


namespace fraction_subtraction_simplified_l340_340555

theorem fraction_subtraction_simplified :
  (8 / 19) - (5 / 57) = (1 / 3) :=
by
  sorry

end fraction_subtraction_simplified_l340_340555


namespace sin_4x_solution_l340_340856

theorem sin_4x_solution (x : ℝ) (k n : ℤ) :
  sin (4 * x) * (3 * sin (4 * x) - 2 * cos (4 * x)) =
  sin (2 * x) ^ 2 - 16 * sin (x) ^ 2 * cos (x) ^ 2 * cos (2 * x) ^ 2 + cos (2 * x) ^ 2 ↔
  (x = -1 / 4 * arctan (1 / 3) + ↑k * (π / 4)) ∨
  (x = (π / 16) * (4 * ↑n + 1)) := sorry

end sin_4x_solution_l340_340856


namespace gain_percentage_l340_340501

theorem gain_percentage (CP SP : ℕ) (h_sell : SP = 10 * CP) : 
  (10 * CP / 25 * CP) * 100 = 40 := by
  sorry

end gain_percentage_l340_340501


namespace star_to_circle_ratio_l340_340877

-- Define the radius
def radius : ℝ := 3

-- Define the area of the larger circle
def area_circle : ℝ := π * radius^2

-- Define the side length of the surrounding square
def side_length : ℝ := 2 * radius

-- Define the area of the surrounding square
def area_square : ℝ := side_length^2

-- Define the area of one quarter-circle
def area_quarter_circle : ℝ := (1 / 4) * π * radius^2

-- Define the area of the star figure
def area_star : ℝ := area_square - 4 * area_quarter_circle

-- Define the ratio of the area of the star figure to the area of the larger circle
def ratio : ℝ := area_star / area_circle

-- Proof statement
theorem star_to_circle_ratio : ratio = (4 - π) / π :=
by
  -- Let's leave the proof as sorry as instructed
  sorry

end star_to_circle_ratio_l340_340877


namespace correct_value_of_3_dollar_neg4_l340_340947

def special_operation (x y : Int) : Int :=
  x * (y + 2) + x * y + x

theorem correct_value_of_3_dollar_neg4 : special_operation 3 (-4) = -15 :=
by
  sorry

end correct_value_of_3_dollar_neg4_l340_340947


namespace arithmetic_sum_goal_l340_340302

section ArithmeticSequence

variable (a_1 d : ℝ)

-- Define the n-th term of the arithmetic sequence
def a_n (n : ℕ) := a_1 + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def S_n (n : ℕ) := (n / 2) * (2 * a_1 + (n - 1) * d)

-- Given that S_20 = 340
axiom h_sum : S_n a_1 d 20 = 340

-- The goal is to prove that a_6 + a_9 + a_{11} + a_{16} = 68
theorem arithmetic_sum_goal : a_n a_1 d 6 + a_n a_1 d 9 + a_n a_1 d 11 + a_n a_1 d 16 = 68 :=
sorry

end ArithmeticSequence

end arithmetic_sum_goal_l340_340302


namespace incorrect_equation_l340_340661

noncomputable def x : ℂ := (-1 + Real.sqrt 3 * Complex.I) / 2
noncomputable def y : ℂ := (-1 - Real.sqrt 3 * Complex.I) / 2

theorem incorrect_equation : x^9 + y^9 ≠ -1 := sorry

end incorrect_equation_l340_340661


namespace tangent_slope_l340_340121

-- Define the coordinates for the center of the circle and the point of tangency
def center : ℝ × ℝ := (2, 1)
def point_of_tangency : ℝ × ℝ := (6, 3)

-- Define a function to compute the slope between two points
def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

-- Show that the slope of the tangent line at the point of tangency is -2
theorem tangent_slope :
  slope center point_of_tangency = 1/2 → 
  slope point_of_tangency center = -2 :=
by
  intros h

  -- The slope of the radius
  have slope_radius : slope center point_of_tangency = 1/2 := h

  -- The slope of the tangent is the negative reciprocal of the radius slope
  have tangent_slope : slope point_of_tangency center = -1 / slope_radius := by
    rw slope_radius
    have reciprocal : 1 / (1 / 2) = 2 := sorry  -- Prove that the negative reciprocal is -2
    rw reciprocal
    exact sorry
  
  exact sorry

end tangent_slope_l340_340121


namespace multiply_exponents_l340_340206

theorem multiply_exponents (x : ℝ) : (x^2) * (x^3) = x^5 := 
sorry

end multiply_exponents_l340_340206


namespace find_angles_of_triangle_ABC_l340_340696

variable (A B C X Y : Type)
variable [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited X] [Inhabited Y]
variables (triangle : Type) [Inhabited triangle]
variables (Angle : Type) [Inhabited Angle]
variables (eq_angle_ABX_YAC : Angle) (eq_angle_AYB_BXC : Angle) (eq_length_XC_YB : Angle)
variables (angle_ABC : Angle) (angle_BCA : Angle) (angle_CAB : Angle)

-- Define the conditions
structure Triangle :=
  (A : Type) (B : Type) (C : Type)
  [inh_A : Inhabited A] [inh_B : Inhabited B] [inh_C : Inhabited C]

-- Define the points on the triangle
structure PointsOnTriangle :=
  (X : Type) (Y : Type)
  [inh_X : Inhabited X] [inh_Y : Inhabited Y]

-- Define the angle properties given in the problem
structure AngleProperties :=
  (eq_angle_ABX_YAC : Angle)
  (eq_angle_AYB_BXC : Angle)
  (eq_length_XC_YB : Angle)

-- Define the equation of the angles in triangle ABC
structure TriangleABC :=
  (angle_ABC : Angle)
  (angle_BCA : Angle)
  (angle_CAB : Angle)

-- State the problem as a Lean 4 theorem.
theorem find_angles_of_triangle_ABC (ABC : Triangle) (Points : PointsOnTriangle) (Angles : AngleProperties):
  ∃ (equal_angles : TriangleABC),
    equal_angles.angle_ABC = 60 ∧
    equal_angles.angle_BCA = 60 ∧
    equal_angles.angle_CAB = 60 := 
sorry

end find_angles_of_triangle_ABC_l340_340696


namespace max_value_of_exp_diff_l340_340247

open Real

theorem max_value_of_exp_diff : ∀ x : ℝ, ∃ y : ℝ, y = 2^x - 4^x ∧ y ≤ 1/4 := sorry

end max_value_of_exp_diff_l340_340247


namespace handrail_length_is_25_point_1_l340_340175

noncomputable def handrail_length (theta radius height : ℝ) : ℝ :=
  let circumference := 2 * real.pi * radius
  let width := (theta / 360) * circumference
  real.sqrt (height^2 + width^2)

theorem handrail_length_is_25_point_1 :
  handrail_length 315 4 12 = 25.1 :=
by
  sorry

end handrail_length_is_25_point_1_l340_340175


namespace angle_POQ_l340_340502

theorem angle_POQ (O P Q F1 F2 : Point) (α β : ℝ) 
  (angle_PFO1 : Angle) (angle_PFO2 : Angle) 
  (h1 : angle_PFO1 = α)
  (h2 : angle_PFO2 = β)
  (tangents : Tangents O P Q F1 F2) : 
  ∠POQ = π - (1 / 2) * (α + β) := 
begin 
  sorry 
end

end angle_POQ_l340_340502


namespace trig_225_deg_l340_340932

noncomputable def sin_225 : Real := Real.sin (225 * Real.pi / 180)
noncomputable def cos_225 : Real := Real.cos (225 * Real.pi / 180)

theorem trig_225_deg :
  sin_225 = -Real.sqrt 2 / 2 ∧ cos_225 = -Real.sqrt 2 / 2 := by
  sorry

end trig_225_deg_l340_340932


namespace range_of_k_l340_340625

theorem range_of_k {k : ℝ} :
  (∀ (k : ℝ), (0 < k ∧ k ≤ 1 ∧
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 ∈ set.Icc (k - 1) (k + 1) ∧ x2 ∈ set.Icc (k - 1) (k + 1) ∧
  (|x1 - k| = (sqrt 2 / 2) * k * sqrt x1)
  ∧ (|x2 - k| = (sqrt 2 / 2) * k * sqrt x2))))
→ 0 < k ∧ k ≤ 1 := sorry

end range_of_k_l340_340625


namespace net_salary_change_is_minus_18_point_1_percent_l340_340188

variable (S : ℝ)

def initial_salary := S
def after_first_increase := S * 1.20
def after_second_increase := after_first_increase * 1.40
def after_first_decrease := after_second_increase * (1 - 0.35)
def after_second_decrease := after_first_decrease * (1 - 0.25)
def final_salary := after_second_decrease

noncomputable def percentage_change := ((final_salary - initial_salary) / initial_salary) * 100

theorem net_salary_change_is_minus_18_point_1_percent :
  percentage_change S = -18.1 :=
by
  sorry

end net_salary_change_is_minus_18_point_1_percent_l340_340188


namespace number_of_integers_250_lt_nsqr_lt_1000_l340_340979

theorem number_of_integers_250_lt_nsqr_lt_1000 : 
  (finset.range 32).filter (λ n, 250 < n^2 ∧ n^2 < 1000)).card = 16 :=
by
  sorry

end number_of_integers_250_lt_nsqr_lt_1000_l340_340979


namespace median_property_l340_340800

variables (a b c a1 b1 c1 h : ℝ)

-- The conditions of the problem
def is_tetrahedron_edges := 
    (AB = c) ∧ (BC = a) ∧ (CA = b) ∧ (DA = a1) ∧ (DB = b1) ∧ (DC = c1)

def median_length (D A B C : ℝ) := 
      -- The length of the median from vertex D to the centroid of face ABC
     h

-- The goal to prove
theorem median_property 
    (AB BC CA DA DB DC A B C D : ℝ)
    (h : ℝ)
    (Hedges : is_tetrahedron_edges AB BC CA DA DB DC)
    (Hlength : h = median_length D A B C) :
     h^2 = (1 / 3) * (a1^2 + b1^2 + c1^2) - (1 / 9) * (a^2 + b^2 + c^2) := 
    sorry

end median_property_l340_340800


namespace apples_difference_l340_340007

-- Definitions based on conditions
def JackiesApples : Nat := 10
def AdamsApples : Nat := 8

-- Statement
theorem apples_difference : JackiesApples - AdamsApples = 2 := by
  sorry

end apples_difference_l340_340007


namespace total_legs_l340_340011

theorem total_legs 
  (johnny_legs : ℕ := 2) 
  (son_legs : ℕ := 2) 
  (dog_legs_per_dog : ℕ := 4) 
  (number_of_dogs : ℕ := 2) :
  johnny_legs + son_legs + dog_legs_per_dog * number_of_dogs = 12 := 
sorry

end total_legs_l340_340011


namespace area_of_equilateral_triangle_with_inscribed_circle_l340_340476

theorem area_of_equilateral_triangle_with_inscribed_circle 
  (r : ℝ) (A : ℝ) (area_circle_eq : A = 9 * Real.pi)
  (DEF_equilateral : ∀ {a b c : ℝ}, a = b ∧ b = c): 
  ∃ area_def : ℝ, area_def = 27 * Real.sqrt 3 :=
by
  -- proof omitted
  sorry

end area_of_equilateral_triangle_with_inscribed_circle_l340_340476


namespace circles_ordering_l340_340558

theorem circles_ordering :
  let rA := 3
  let AB := 12 * Real.pi
  let AC := 28 * Real.pi
  let rB := Real.sqrt 12
  let rC := Real.sqrt 28
  (rA < rB) ∧ (rB < rC) :=
by
  let rA := 3
  let AB := 12 * Real.pi
  let AC := 28 * Real.pi
  let rB := Real.sqrt 12
  let rC := Real.sqrt 28
  have rA_lt_rB: rA < rB := by sorry
  have rB_lt_rC: rB < rC := by sorry
  exact ⟨rA_lt_rB, rB_lt_rC⟩

end circles_ordering_l340_340558


namespace cos_product_prime_gt_3_eq_three_l340_340145

noncomputable def cos_product_prime_gt_3 (n : ℕ) (h_prime : nat.prime n) (h_gt_3 : 3 < n) : ℂ :=
  ∏ k in finset.range n, (1 + 2 * complex.cos ((2 * k : ℕ) * real.pi / n))

theorem cos_product_prime_gt_3_eq_three (n : ℕ) (h_prime : nat.prime n) (h_gt_3 : 3 < n) :
  cos_product_prime_gt_3 n h_prime h_gt_3 = 3 :=
sorry

end cos_product_prime_gt_3_eq_three_l340_340145


namespace sum_is_root_of_quadratic_l340_340818

open Real

theorem sum_is_root_of_quadratic (n : ℕ) : 
  let sum := ∑ k in Finset.range(n) + 1, 1 / (sqrt (2 * k - 1) + sqrt (2 * k + 1))
  in sum^2 + sum + -n/2 = 0 :=
by
  let sum := ∑ k in Finset.range(n+1), 1 / (sqrt (2 * k - 1) + sqrt (2 * k + 1))
  sorry

end sum_is_root_of_quadratic_l340_340818


namespace find_m_l340_340353

theorem find_m (m : ℝ) :
  let A := (m, 0)
  let A_prime := (-m, 0)
  let A_translated := (m - 6, 0)
  (A_translated = A_prime) → m = 3 :=
by
  intros A A_prime A_translated h
  have h1: A_translated = A_prime := h
  rw [A_prime, A_translated] at h1
  sorry

end find_m_l340_340353


namespace area_quadrilateral_minimum_l340_340429

noncomputable def quadrilateral_area (A B C D : Point) : Real := sorry 
noncomputable def point_on_segment (P Q : Point) (λ : Real) : Point := sorry

theorem area_quadrilateral_minimum (
  A B C D : Point
  (A' B' C' D' : Point)
  (λ : Real) 
  (hA' : point_on_segment A B λ = A')
  (hB' : point_on_segment B C λ = B')
  (hC' : point_on_segment C D λ = C')
  (hD' : point_on_segment D A λ = D')
) : quadrilateral_area A' B' C' D' ≥ (1/2) * quadrilateral_area A B C D := 
sorry

end area_quadrilateral_minimum_l340_340429


namespace geometric_progression_solution_l340_340823

-- Definitions and conditions as per the problem
def geometric_progression_first_term (b q : ℝ) : Prop :=
  b * (1 + q + q^2) = 21

def geometric_progression_sum_of_squares (b q : ℝ) : Prop :=
  b^2 * (1 + q^2 + q^4) = 189

-- The main theorem to be proven
theorem geometric_progression_solution (b q : ℝ) :
  (geometric_progression_first_term b q ∧ geometric_progression_sum_of_squares b q) →
  (b = 3 ∧ q = 2) ∨ (b = 12 ∧ q = 1 / 2) := 
by
  intros h
  sorry

end geometric_progression_solution_l340_340823


namespace matrix_transformation_l340_340240

open Matrix

-- Definitions
variable {α : Type*} [CommRing α]
def targetMatrix := (λ (N : Matrix (Fin 3) (Fin 3) α) =>
  ![
    ![N 2 0, N 2 1, N 2 2],
    ![3 * N 1 0, 3 * N 1 1, 3 * N 1 2],
    ![N 0 0, N 0 1, N 0 2]
  ]
)

-- The goal
theorem matrix_transformation :
  let M := ![
    ![0, 0, 1],
    ![0, 3, 0],
    ![1, 0, 0]
  ] in
  ∀ (N : Matrix (Fin 3) (Fin 3) α),
    M ⬝ N = targetMatrix N := by
  sorry

end matrix_transformation_l340_340240


namespace smallest_degree_of_polynomial_with_given_roots_l340_340441

theorem smallest_degree_of_polynomial_with_given_roots :
  ∃ (p : ℚ[X]), 
    (p ≠ 0) ∧ 
    (root p (2 - 3 * sqrt 3)) ∧ 
    (root p (-2 - 3 * sqrt 3)) ∧ 
    (root p (3 + sqrt 5)) ∧ 
    (root p (3 - sqrt 5)) ∧ 
    (polynomial.degree p).nat_degree = 6 :=
sorry

end smallest_degree_of_polynomial_with_given_roots_l340_340441


namespace problem_intersection_line_l340_340454

theorem problem_intersection_line (b : ℝ) :
  (∃ (x y : ℝ), x + y = b ∧ x = (1 + 5) / 2 ∧ y = (3 + 11) / 2) → b = 10 :=
by
  intro h
  cases h with x hx
  cases hx with y hy
  cases hy
  -- Proof would go here
  sorry

end problem_intersection_line_l340_340454


namespace carla_needs_30_leaves_l340_340557

-- Definitions of the conditions
def items_per_day : Nat := 5
def total_days : Nat := 10
def total_bugs : Nat := 20

-- Maths problem to be proved
theorem carla_needs_30_leaves :
  let total_items := items_per_day * total_days
  let required_leaves := total_items - total_bugs
  required_leaves = 30 :=
by
  sorry

end carla_needs_30_leaves_l340_340557


namespace smallest_n_l340_340123

theorem smallest_n (m l n : ℕ) :
  (∃ m : ℕ, 2 * n = m ^ 4) ∧ (∃ l : ℕ, 3 * n = l ^ 6) → n = 1944 :=
by
  sorry

end smallest_n_l340_340123


namespace solution_to_inequality_l340_340974

noncomputable def f (x : ℝ) : ℝ := (3 * x - 8) * (x - 5) / (x - 2)

theorem solution_to_inequality : {x : ℝ | f x ≤ 0} = set.Icc (8 / 3) 5 :=
by
  sorry

end solution_to_inequality_l340_340974


namespace integral_iterateP_2004_l340_340959

-- Define the polynomial function P(x)
def P (x : ℝ) : ℝ := x^3 - (3 / 2) * x^2 + x + (1 / 4)

-- Define the iteration P^[n](x)
def iterateP : ℕ → (ℝ → ℝ)
| 0     := id
| (n+1) := λ x, P (iterateP n x)

-- Prove that the integral of P^[2004](x) from 0 to 1 is 1/2
theorem integral_iterateP_2004 :
  ∫ x in 0..1, iterateP 2004 x = 1/2 :=
sorry

end integral_iterateP_2004_l340_340959


namespace period_of_sin_x_plus_2_cos_x_l340_340847

-- Definition of the function
def f (x : ℝ) : ℝ := sin x + 2 * cos x

-- The statement to prove the period is 2π
theorem period_of_sin_x_plus_2_cos_x : ∀ x : ℝ, f (x + 2 * π) = f x := by
  -- Proof goes here
  sorry

end period_of_sin_x_plus_2_cos_x_l340_340847


namespace conjugate_in_first_quadrant_l340_340599

open Complex

noncomputable def z : ℂ := (2 + I) / (1 + 2 * I)
noncomputable def z_conjugate : ℂ := conj z

theorem conjugate_in_first_quadrant (h : (1 + 2 * I) * z = 2 + I) :
  z_conjugate.re > 0 ∧ z_conjugate.im > 0 := by
sorry

end conjugate_in_first_quadrant_l340_340599


namespace product_consecutive_natural_not_equal_even_l340_340925

theorem product_consecutive_natural_not_equal_even (n m : ℕ) (h : m % 2 = 0 ∧ m > 0) : n * (n + 1) ≠ m * (m + 2) :=
sorry

end product_consecutive_natural_not_equal_even_l340_340925


namespace triangle_abc_isosceles_acute_l340_340130

theorem triangle_abc_isosceles_acute (A B C : Type) [triangle A B C] (h : AB = AC) : ¬(∠ B ≥ 90°) :=
by
  sorry

end triangle_abc_isosceles_acute_l340_340130


namespace sum_of_possible_values_f_is_124_l340_340562

-- Define the conditions specific to our problem
def is_multiplicative_magic_square (a b c d e f g : ℕ) : Prop :=
  let P := 75 * a * b in
  P = c * d * e ∧
  P = f * g * 3 ∧
  P = 75 * d * 3

noncomputable def possible_values_f (d : ℕ) : set ℕ :=
  {f | ∃ g, f * g = 75 * d}

theorem sum_of_possible_values_f_is_124 :
  ∃ (a b c d e f g : ℕ), is_multiplicative_magic_square a b c d e f g ∧
  (finset.univ.filter (possible_values_f 1)).sum = 124 :=
sorry

end sum_of_possible_values_f_is_124_l340_340562


namespace perfect_match_of_products_l340_340679

theorem perfect_match_of_products
  (x : ℕ)  -- number of workers assigned to produce nuts
  (h1 : 22 - x ≥ 0)  -- ensuring non-negative number of workers for screws
  (h2 : 1200 * (22 - x) = 2 * 2000 * x) :  -- the condition for perfect matching
  (2 * 1200 * (22 - x) = 2000 * x) :=  -- the correct equation
by sorry

end perfect_match_of_products_l340_340679


namespace correct_propositions_l340_340913

-- Definitions for propositions
def proposition_a (A B : ℝ) [Real.lt A B] : Prop :=
  A > B → Real.sin A > Real.sin B

def proposition_b (A B : ℝ) [Real.lt A π/2] [Real.lt B π/2] : Prop :=
  ∀ (A B : ℝ), (0 < A ∧ A < π/2) → (0 < B ∧ B < π/2) → Real.sin A > Real.cos B

def proposition_c (a b A B : ℝ) : Prop :=
  a * Real.cos A = b * Real.cos B → False -- translating the statement that this is an incorrect proposition

def proposition_d (a b c : ℝ) : Prop :=
  b = 60 * π / 180 ∧ b^2 = a * c → a = c

-- The mathematical statement to prove which propositions are correct
theorem correct_propositions :
  (proposition_a ∧ proposition_b ∧ proposition_d) ∧ ¬proposition_c ← sorry

end correct_propositions_l340_340913


namespace find_p8_eq_40328_l340_340026

theorem find_p8_eq_40328 (p : ℤ → ℤ) 
  (hp_monic : p.monotonic) (hp_deg : p.degree = 7)
  (hp_0 : p(0) = 0) (hp_1 : p(1) = 1) 
  (hp_2 : p(2) = 2) (hp_3 : p(3) = 3) 
  (hp_4 : p(4) = 4) (hp_5 : p(5) = 5)
  (hp_6 : p(6) = 6) (hp_7 : p(7) = 7) : 
  p(8) = 40328 := 
sorry

end find_p8_eq_40328_l340_340026


namespace sub_neg_seven_eq_neg_fourteen_l340_340142

theorem sub_neg_seven_eq_neg_fourteen : (-7) - 7 = -14 := 
  by
  sorry

end sub_neg_seven_eq_neg_fourteen_l340_340142


namespace no_email_days_during_2022_l340_340742

theorem no_email_days_during_2022
  (emails1 : ℕ → Prop)
  (emails2 : ℕ → Prop)
  (emails3 : ℕ → Prop)
  (h1 : ∀ n, emails1 (4 * n + 1))
  (h2 : ∀ n, emails2 (6 * n + 1))
  (h3 : ∀ n, emails3 (8 * n + 1)) :
  (∃ d, ∀ n, (n ≠ 0) → (¬ emails1 n ∧ ¬ emails2 n ∧ ¬ emails3 n) ∧ d = 244) :=
by
  let year_length := 365
  let emails := λ d, emails1 d ∨ emails2 d ∨ emails3 d
  have days_with_emails := λ d, emails d ∧ (d ≤ year_length)
  have days_with_no_emails := λ d, ¬ emails d ∧ (d ≤ year_length)
  have no_emails := 365 - 244
  sorry

end no_email_days_during_2022_l340_340742


namespace distance_from_home_to_school_l340_340499

variable (t : ℕ) (D : ℕ)

-- conditions
def condition1 := 60 * (t - 10) = D
def condition2 := 50 * (t + 4) = D

-- the mathematical equivalent proof problem: proving the distance is 4200 given conditions
theorem distance_from_home_to_school :
  (∃ t, condition1 t 4200 ∧ condition2 t 4200) :=
  sorry

end distance_from_home_to_school_l340_340499


namespace geometric_progression_solution_l340_340822

-- Definitions and conditions as per the problem
def geometric_progression_first_term (b q : ℝ) : Prop :=
  b * (1 + q + q^2) = 21

def geometric_progression_sum_of_squares (b q : ℝ) : Prop :=
  b^2 * (1 + q^2 + q^4) = 189

-- The main theorem to be proven
theorem geometric_progression_solution (b q : ℝ) :
  (geometric_progression_first_term b q ∧ geometric_progression_sum_of_squares b q) →
  (b = 3 ∧ q = 2) ∨ (b = 12 ∧ q = 1 / 2) := 
by
  intros h
  sorry

end geometric_progression_solution_l340_340822


namespace value_is_9_l340_340128

def x : ℝ := 4.5

def y : ℝ := x / 6

theorem value_is_9 : y * 12 = 9 := by
  sorry

end value_is_9_l340_340128


namespace complex_number_modulus_squared_l340_340878

theorem complex_number_modulus_squared (w : ℂ) (h : w + complex.abs w = 4 + 5 * complex.i) :
  complex.abs w ^ 2 = 1681 / 64 :=
sorry

end complex_number_modulus_squared_l340_340878


namespace find_m_value_l340_340616

/-- definition of the distance from a point to a line -/
def distance_to_line (x y m : ℝ) : ℝ := |x + m * y - 1| / Real.sqrt (1 + m^2)

/-- final statement with the conditions and the expected conclusion -/
theorem find_m_value
  (A_x A_y : ℝ) (B_x B_y : ℝ) (m : ℝ)
  (hA : A_x = -2) (hA' : A_y = 0)
  (hB : B_x = 0) (hB' : B_y = 4)
  (h_eq_dist : distance_to_line A_x A_y m = distance_to_line B_x B_y m) :
  m = -1/2 ∨ m = 1 :=
by
  rw [hA, hA', hB, hB'] at h_eq_dist
  simp only [distance_to_line] at h_eq_dist
  sorry

end find_m_value_l340_340616


namespace proof_problem_l340_340323
open Classical

-- Define the planes and their relationships
variables {α β γ : Type} [Plane α] [Plane β] [Plane γ]
variables {non_collinear_points_on_alpha : Set α} (h_non_collinear : NonCollinear non_collinear_points_on_alpha)

-- Define propositions p and q
def p : Prop := (Perpendicular α β) ∧ (Perpendicular β γ) → Parallel α γ
def q : Prop := (Equidistant β non_collinear_points_on_alpha) → Parallel α β

-- Prove the combined proposition "p and q is false"
theorem proof_problem : ¬ (p ∧ q) :=
by
  sorry

end proof_problem_l340_340323


namespace matrix_power_eq_l340_340015

def MatrixC : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![3, 4], ![-8, -10]]

def MatrixA : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![201, 200], ![-400, -449]]

theorem matrix_power_eq :
  MatrixC ^ 50 = MatrixA := 
  sorry

end matrix_power_eq_l340_340015


namespace minimum_strips_cover_circle_l340_340489

theorem minimum_strips_cover_circle (l R : ℝ) (hl : l > 0) (hR : R > 0) :
  ∃ (k : ℕ), (k : ℝ) * l ≥ 2 * R ∧ ((k - 1 : ℕ) : ℝ) * l < 2 * R :=
sorry

end minimum_strips_cover_circle_l340_340489


namespace vector_norm_sub_l340_340639

noncomputable def a : Vector ℝ 3 := sorry
noncomputable def b : Vector ℝ 3 := sorry

-- Given conditions 
axiom norm_a : ‖a‖ = 2
axiom norm_b : ‖b‖ = 3
axiom dot_ab : a ⬝ b = 2

-- The theorem to prove
theorem vector_norm_sub : ‖a - b‖ = 3 := by
  -- Proof goes here
  sorry

end vector_norm_sub_l340_340639


namespace rain_A_given_B_l340_340684

noncomputable def prob_rain_given_B (P_A P_B P_A_and_B : ℚ) : ℚ :=
  P_A_and_B / P_B

theorem rain_A_given_B :
  ∀ (P_A P_B P_A_and_B : ℚ),
    P_A = 0.20 ∧ P_B = 0.18 ∧ P_A_and_B = 0.12 → 
    prob_rain_given_B P_A P_B P_A_and_B = 2 / 3 :=
by
  intros P_A P_B P_A_and_B h
  cases h with h1 h2
  cases h2 with h3 h4
  rw [h1, h3, h4]
  norm_num
  sorry

end rain_A_given_B_l340_340684


namespace sequence_a_is_perfect_square_l340_340732

theorem sequence_a_is_perfect_square :
  let a : ℕ → ℤ := λ n, nat.recOn n 1 (λ n a_n, 7 * a_n + 6 * (nat.recOn n 0 (λ n b_n, 8 * (nat.recOn n 1 (λ n a_n2, 7 * a_n2 + 6 * b_n - 3)) + 7 * b_n - 4)) - 3)
  ∧ b : ℕ → ℤ := λ n, nat.recOn n 0 (λ n b_n, 8 * (nat.recOn n 1 (λ n a_n, 7 * (nat.recOn n (n + 1) 1)) + 6 * b_n - 3)) + 7 * b_n - 4 in
  ∀ (n : ℕ), ∃ (x_n : ℤ), a n = x_n * x_n := 
by
  sorry

end sequence_a_is_perfect_square_l340_340732


namespace eval_expression_l340_340573

variable {x : ℝ}

theorem eval_expression (x : ℝ) : x * (x * (x * (3 - x) - 5) + 8) + 2 = -x^4 + 3 * x^3 - 5 * x^2 + 8 * x + 2 :=
by
  sorry

end eval_expression_l340_340573


namespace Q_at_1_is_zero_l340_340217

def P (x : ℝ) : ℝ := 3 * x^3 - 2 * x^2 + 4 * x - 5

def mean_of_nonzero_coeffs : ℝ := (3 - 2 + 4 - 5) / 4

def Q (x : ℝ) : ℝ := mean_of_nonzero_coeffs * x^3 + mean_of_nonzero_coeffs * x^2 + mean_of_nonzero_coeffs * x + mean_of_nonzero_coeffs

theorem Q_at_1_is_zero : Q 1 = 0 :=
by
  sorry

end Q_at_1_is_zero_l340_340217


namespace volume_is_correct_l340_340086

noncomputable def volume_of_rectangular_parallelepiped (a b : ℝ) (h_diag : (2 * a^2 + b^2 = 1)) (h_surface_area : (4 * a * b + 2 * a^2 = 1)) : ℝ :=
  a^2 * b

theorem volume_is_correct (a b : ℝ)
  (h_diag : 2 * a^2 + b^2 = 1)
  (h_surface_area : 4 * a * b + 2 * a^2 = 1) :
  volume_of_rectangular_parallelepiped a b h_diag h_surface_area = (Real.sqrt 2) / 27 :=
sorry

end volume_is_correct_l340_340086


namespace range_of_r_for_lines_l340_340162

def parabola (x y : ℝ) : Prop := y^2 = 6 * x
def circle (x y r : ℝ) : Prop := (x - 6)^2 + y^2 = r^2
def midpoint (x0 y0 x1 y1 x2 y2 : ℝ) : Prop := 2 * x0 = x1 + x2 ∧ 2 * y0 = y1 + y2

theorem range_of_r_for_lines 
  (x1 y1 x2 y2 x0 y0 r : ℝ)
  (h_parabola1 : parabola x1 y1)
  (h_parabola2 : parabola x2 y2)
  (h_midpoint : midpoint x0 y0 x1 y1 x2 y2)
  (h_circle : circle x0 y0 r) :
  3 < r ∧ r < 3 * Real.sqrt 3 ↔
  ∃ l : ℝ, ∀ (x y : ℝ), (l * x + y = 0) ∧ 
                     ((l * x + 1) = (6 * x) / x^2) :=
sorry

end range_of_r_for_lines_l340_340162


namespace round_trip_time_l340_340096

variable (boat_speed : ℕ) (stream_speed : ℕ) (distance : ℕ)

theorem round_trip_time :
  boat_speed = 8 → 
  stream_speed = 6 → 
  distance = 210 → 
  let downstream_speed := boat_speed + stream_speed,
      upstream_speed := boat_speed - stream_speed,
      time_downstream := distance / downstream_speed,
      time_upstream := distance / upstream_speed
  in 
    (time_downstream + time_upstream) = 120 := 
by {
  intros,
  sorry
}

end round_trip_time_l340_340096


namespace ratio_of_radii_l340_340536

noncomputable theory

-- Define the problem conditions and the question to be proven
def sphere_in_truncated_cone (R r s : ℝ) : Prop :=
  s = Real.sqrt (R * r) ∧ -- Geometric mean theorem
  ∃ h H, -- There exist heights h and H such that
  H = h + 2 * s ∧
  h = 2 * s * (r / (R - r)) ∧
  (π * R^2 * H / 3) - (π * r^2 * h / 3) = 3 * (4 * π * s^3 / 3)

theorem ratio_of_radii (R r : ℝ) (h H s : ℝ)
  (hs : s = Real.sqrt (R * r))
  (hH : H = h + 2 * s)
  (hh : h = 2 * s * (r / (R - r)))
  (vol_relation : (π * R^2 * H / 3) - (π * r^2 * h / 3) = 3 * (4 * π * s^3 / 3)) :
  R / r = (5 + Real.sqrt 21) / 2 :=
by sorry

end ratio_of_radii_l340_340536


namespace tangent_segment_length_l340_340936

noncomputable def circumcenter (A B C : point) : point := sorry
noncomputable def radius (O : point) (O_C : point) : ℝ := sorry
noncomputable def power_of_point (O : point) (A B : point) : ℝ := sorry

theorem tangent_segment_length 
    (O : point := (0, 0)) 
    (A : point := (4, 3)) 
    (B : point := (8, 6)) 
    (C : point := (13, 5)) :
    let O_C := circumcenter A B C in
    let r := radius O O_C in
    let OA := (euclidean_distance O A) in
    let OB := (euclidean_distance O B) in
    let OT2 := power_of_point O A B in
    OA * OB = 100 ->
    sqrt OT2 = 10 := 
sorry

end tangent_segment_length_l340_340936


namespace find_a_l340_340977

theorem find_a :
  (∃ x1 x2, (x1 + x2 = -2 ∧ x1 * x2 = a) ∧ (∃ y1 y2, (y1 + y2 = - a ∧ y1 * y2 = 2) ∧ (x1^2 + x2^2 = y1^2 + y2^2))) → 
  (a = -4) := 
by
  sorry

end find_a_l340_340977


namespace weights_less_than_90_l340_340166

variable (a b c : ℝ)
-- conditions
axiom h1 : a + b = 100
axiom h2 : a + c = 101
axiom h3 : b + c = 102

theorem weights_less_than_90 (a b c : ℝ) (h1 : a + b = 100) (h2 : a + c = 101) (h3 : b + c = 102) : a < 90 ∧ b < 90 ∧ c < 90 := 
by sorry

end weights_less_than_90_l340_340166


namespace discount_on_glove_l340_340640

variables (total_money amount_cards amount_bat amount_cleats_glove_discount amount_glove_sale original_price_glove: ℝ)
variables (percentage_discount: ℝ)

-- Given conditions:
def total_money := 79
def amount_cards := 25
def amount_bat := 10
def amount_cleats := 20
def original_price_glove := 30
def amount_glove_sale := total_money - (amount_cards + amount_bat + amount_cleats)

-- Question: What is the percentage discount on the baseball glove?
def percentage_discount := ((original_price_glove - amount_glove_sale) / original_price_glove) * 100

theorem discount_on_glove : percentage_discount = 20 := by
  sorry

end discount_on_glove_l340_340640


namespace probability_acute_or_right_triangles_l340_340592

theorem probability_acute_or_right_triangles :
  let points_chosen_uniformly_at_random_on_circle := true
  let triangle_formed_with_center := true
  in
    let probability_of_all_acute_or_right_triangles := (1 : ℝ) / 64
    in
      probability_of_all_acute_or_right_triangles = 1 / 64 := 
sorry

end probability_acute_or_right_triangles_l340_340592


namespace minimum_n_for_all_columns_l340_340172

def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

-- Function to check if a given number covers all columns from 0 to 9
def covers_all_columns (n : ℕ) : Bool :=
  let columns := (List.range n).map (λ i => triangular_number i % 10)
  List.range 10 |>.all (λ c => c ∈ columns)

theorem minimum_n_for_all_columns : ∃ n, covers_all_columns n ∧ triangular_number n = 253 :=
by 
  sorry

end minimum_n_for_all_columns_l340_340172


namespace max_segments_no_triangle_l340_340197

theorem max_segments_no_triangle (n : ℕ) : 
  ∃ k, k = n * (n + 1) ∧ ∀ (chosen_segments : set (ℕ × ℕ)), 
  chosen_segments.card = k → 
  ∀ (a b c : ℕ × ℕ), (a ∈ chosen_segments ∧ b ∈ chosen_segments ∧ c ∈ chosen_segments) → 
  (a, b, c do not form a triangle) :=
sorry

end max_segments_no_triangle_l340_340197


namespace coeff_of_k1_term_in_expansion_l340_340077

theorem coeff_of_k1_term_in_expansion (k : ℕ) : 
  (∀ x y : ℝ, k ≤ 2016 → 
    (binomial 2016 k) * (2 ^ (2016 - k)) * (5 ^ k) = 
    coeff (expand_polynomial (2 * x + 5 * y) 2016) (multi_var_term k (2016 - k))) := sorry

end coeff_of_k1_term_in_expansion_l340_340077


namespace quilt_block_shading_l340_340938

theorem quilt_block_shading (total_squares shaded_triangles : ℕ) 
(total_area shaded_area : ℝ)
(h1 : total_squares = 16)
(h2 : shaded_triangles = 4)
(h3 : ∀ t, t ∈ {1,2,3,4} → t = 2 * shaded_triangles)
(h4 : ∀ a, a ∈ {1,2,3,4} → a = 0.5)
(h5 : shaded_area = shaded_triangles * 0.5)
(h6 : total_area = total_squares * 1) : 
(shaded_area / total_area) = 1 / 8 :=
by
  sorry

end quilt_block_shading_l340_340938


namespace line_properties_l340_340465

theorem line_properties (m x_intercept : ℝ) (y_intercept point_on_line : ℝ × ℝ) :
  m = -4 → x_intercept = -3 → y_intercept = (0, -12) → point_on_line = (2, -20) → 
    (∀ x y, y = -4 * x - 12 → (y_intercept = (0, y) ∧ point_on_line = (x, y))) := 
by
  sorry

end line_properties_l340_340465


namespace problem_p9_solution_l340_340513

open Real

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  in sqrt (s * (s - a) * (s - b) * (s - c))

theorem problem_p9_solution :
  ∀ (AB AC CD : ℝ), 
  AB = 12 → AC = 15 → CD = 10 →
  ∃ (m n p : ℕ), 
    m + n + p = 146 ∧
    ((gcd m p = 1) ∧ 
     ¬ ∃ k : ℕ, prime k ∧ k ^ 2 ∣ n ∧ 
     triangle_area AB AC (AB * CD / AC + CD) = (m : ℝ) * sqrt n / (p : ℝ)) :=
by { intros, sorry }

end problem_p9_solution_l340_340513


namespace gasoline_fraction_used_l340_340504

theorem gasoline_fraction_used
  (speed : ℕ) (gas_usage : ℕ) (initial_gallons : ℕ) (travel_time : ℕ)
  (h_speed : speed = 50) (h_gas_usage : gas_usage = 30) 
  (h_initial_gallons : initial_gallons = 15) (h_travel_time : travel_time = 5) :
  (speed * travel_time) / gas_usage / initial_gallons = 5 / 9 :=
by
  sorry

end gasoline_fraction_used_l340_340504


namespace AB_side_length_l340_340775

noncomputable def P := (x : ℝ) × (y : ℝ)

def is_foot_perpendicular (P : P) (A B : P) : P := sorry

def equilateral_triangle (A B C : P) : Prop := sorry

theorem AB_side_length (A B C P Q R S : P)
  (h_equilateral : equilateral_triangle A B C)
  (h_P_inside : sorry /* P inside ABC */)
  (h_Q_foot : Q = is_foot_perpendicular P A B) 
  (h_R_foot : R = is_foot_perpendicular P B C)
  (h_S_foot : S = is_foot_perpendicular P C A)
  (h_PQ : (dist P Q) = 2)
  (h_PR : (dist P R) = 3)
  (h_PS : (dist P S) = 4) :
  dist A B = 6 * real.sqrt 3 := 
sorry

end AB_side_length_l340_340775


namespace max_value_expression_l340_340293

/--
Given real numbers \( x, y, z, w \) satisfying \( x + y + z + w = 1 \),
prove that the maximum value of \( M = xw + 2yw + 3xy + 3zw + 4xz + 5yz \) is \( \frac{3}{2} \).
-/
theorem max_value_expression (x y z w : ℝ) (h : x + y + z + w = 1) :
  let M := x * w + 2 * y * w + 3 * x * y + 3 * z * w + 4 * x * z + 5 * y * z
  in M ≤ 3 / 2 :=
begin
  sorry
end

end max_value_expression_l340_340293


namespace verify_statements_l340_340632

def line1 (a x y : ℝ) : Prop := a * x - (a + 2) * y - 2 = 0
def line2 (a x y : ℝ) : Prop := (a - 2) * x + 3 * a * y + 2 = 0

theorem verify_statements (a : ℝ) :
  (∀ x y : ℝ, line1 a x y ↔ x = -1 ∧ y = -1) ∧
  (∀ x y : ℝ, (line1 a x y ∧ line2 a x y) → (a = 0 ∨ a = -4)) :=
by sorry

end verify_statements_l340_340632


namespace self_employed_tax_amount_l340_340098

-- Definitions for conditions
def gross_income : ℝ := 350000.0

def tax_rate_self_employed : ℝ := 0.06

-- Statement asserting the tax amount for self-employed individuals given the conditions
theorem self_employed_tax_amount :
  gross_income * tax_rate_self_employed = 21000.0 := by
  sorry

end self_employed_tax_amount_l340_340098


namespace probability_of_event_A_l340_340685

def point : Type := ℝ × ℝ
def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def K := {p : point | p.1 = -1 ∨ p.1 = 0 ∨ p.1 = 1 ∧ p.2 = -1 ∨ p.2 = 0 ∨ p.2 = 1}

def event_A (S : set point) : Prop :=
  ∃ (p1 p2 : point), p1 ∈ S ∧ p2 ∈ S ∧ p1 ≠ p2 ∧ distance p1 p2 = real.sqrt 5

noncomputable def pevent_A : ℝ :=
  fintype.card({S : finset point | S ⊆ K ∧ event_A S}.to_set) / 
  fintype.card({S : finset point | S ⊆ K ∧ S.card = 3}.to_set)

theorem probability_of_event_A :
  pevent_A = 4 / 7 :=
by {
  sorry
}

end probability_of_event_A_l340_340685


namespace inequality_proof_l340_340425

theorem inequality_proof (x y : ℝ) (hx_pos : x > 0) (hy_pos : y > 0) (hxy : x < y) : 
  x + real.sqrt (y^2 + 2) < y + real.sqrt (x^2 + 2) :=
sorry

end inequality_proof_l340_340425


namespace total_treat_value_is_339100_l340_340564

def hotel_cost (cost_per_night : ℕ) (nights : ℕ) (discount : ℕ) : ℕ :=
  let total_cost := cost_per_night * nights
  total_cost - (total_cost * discount / 100)

def car_cost (base_price : ℕ) (tax : ℕ) : ℕ :=
  base_price + (base_price * tax / 100)

def house_cost (car_base_price : ℕ) (multiplier : ℕ) (property_tax : ℕ) : ℕ :=
  let house_value := car_base_price * multiplier
  house_value + (house_value * property_tax / 100)

def yacht_cost (hotel_value : ℕ) (car_value : ℕ) (multiplier : ℕ) (discount : ℕ) : ℕ :=
  let combined_value := hotel_value + car_value
  let yacht_value := combined_value * multiplier
  yacht_value - (yacht_value * discount / 100)

def gold_coins_cost (yacht_value : ℕ) (multiplier : ℕ) (tax : ℕ) : ℕ :=
  let gold_value := yacht_value * multiplier
  gold_value + (gold_value * tax / 100)

theorem total_treat_value_is_339100 :
  let hotel_value := hotel_cost 4000 2 5
  let car_value := car_cost 30000 10
  let house_value := house_cost 30000 4 2
  let yacht_value := yacht_cost 8000 30000 2 7
  let gold_coins_value := gold_coins_cost 76000 3 3
  hotel_value + car_value + house_value + yacht_value + gold_coins_value = 339100 :=
by sorry

end total_treat_value_is_339100_l340_340564


namespace first_term_geometric_sequence_b_n_bounded_l340_340404

-- Definition: S_n = 3a_n - 5n for any n in ℕ*
def S (n : ℕ) (a : ℕ → ℝ) : ℝ := 3 * a n - 5 * n

-- The sequence a_n is given such that
-- Proving the first term a_1
theorem first_term (a : ℕ → ℝ) (h : ∀ n, S (n + 1) a = S n a + a n + 1 - 5) : 
  a 1 = 5 / 2 :=
sorry

-- Prove that {a_n + 5} is a geometric sequence with common ratio 3/2
theorem geometric_sequence (a : ℕ → ℝ) (h : ∀ n, S n a = 3 * a n - 5 * n) : 
  ∃ r, (∀ n, a (n + 1) + 5 = r * (a n + 5)) ∧ r = 3 / 2 :=
sorry

-- Prove that there exists m such that b_n < m always holds for b_n = (9n + 4) / (a_n + 5)
theorem b_n_bounded (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h1 : ∀ n, b n = (9 * ↑n + 4) / (a n + 5)) 
  (h2 : ∀ n, a n = (15 / 2) * (3 / 2)^(n-1) - 5) :
  ∃ m, ∀ n, b n < m ∧ m = 88 / 45 :=
sorry

end first_term_geometric_sequence_b_n_bounded_l340_340404


namespace compute_100m_plus_n_l340_340708

noncomputable def set_S : Set ℕ := {c | c ∣ 630000 ∧ c % 70 = 0}

lemma probability_condition (c : ℕ) (h : c ∈ set_S) :
  ∃ d : ℕ, gcd c d = 70 ∧ lcm c d = 630000 := sorry

theorem compute_100m_plus_n : 100 * 1 + 6 = 106 := by
  have m : ℕ := 1
  have n : ℕ := 6
  have h : m = 1 ∧ n = 6 := ⟨rfl, rfl⟩
  sorry

end compute_100m_plus_n_l340_340708


namespace percentage_rejected_products_l340_340705

theorem percentage_rejected_products : 
  ∀ (P : ℝ), P > 0 →
  let J_inspected := (1 / 6) * P in 
  let Jane_inspected := (5 / 6) * P in
  let J_rejected := (0.5 / 100) * J_inspected in
  let Jane_rejected := (0.8 / 100) * Jane_inspected in
  let total_rejected := J_rejected + Jane_rejected in
  let percentage_rejected := (total_rejected / P) * 100 in
  percentage_rejected = 0.75 :=
by {
  intros P hP J_inspected Jane_inspected J_rejected Jane_rejected total_rejected percentage_rejected,
  sorry
}

end percentage_rejected_products_l340_340705


namespace coins_value_percentage_l340_340483

theorem coins_value_percentage :
  let penny_value := 1
  let nickel_value := 5
  let dime_value := 10
  let quarter_value := 25
  let total_value_cents := (1 * penny_value) + (2 * nickel_value) + (1 * dime_value) + (2 * quarter_value)
  (total_value_cents / 100) * 100 = 71 :=
by
  sorry

end coins_value_percentage_l340_340483


namespace buttons_per_shirt_l340_340059

theorem buttons_per_shirt :
  let shirts_monday := 4 in
  let shirts_tuesday := 3 in
  let shirts_wednesday := 2 in
  let total_shirts := shirts_monday + shirts_tuesday + shirts_wednesday in
  let total_buttons := 45 in
  total_buttons / total_shirts = 5 :=
by
  let shirts_monday := 4
  let shirts_tuesday := 3
  let shirts_wednesday := 2
  let total_shirts := shirts_monday + shirts_tuesday + shirts_wednesday
  let total_buttons := 45
  sorry

end buttons_per_shirt_l340_340059


namespace factor_expression_l340_340968

variable (b : ℤ)

theorem factor_expression : 221 * b^2 + 17 * b = 17 * b * (13 * b + 1) :=
by
  sorry

end factor_expression_l340_340968


namespace number_of_valid_four_digit_numbers_l340_340433

def four_digit_numbers := {n : Nat | ∃ a b c d : Nat, 
  n = a * 1000 + b * 100 + c * 10 + d ∧ 
  0 ≤ a ∧ a ≤ 4 ∧ 0 ≤ b ∧ b ≤ 4 ∧ 0 ≤ c ∧ c ≤ 4 ∧ 0 ≤ d ∧ d ≤ 4 ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  n > 3200 ∧ a ≠ 0}

noncomputable def count_valid_numbers : Nat := four_digit_numbers.toFinset.card

theorem number_of_valid_four_digit_numbers : count_valid_numbers = 36 :=
sorry

end number_of_valid_four_digit_numbers_l340_340433


namespace max_value_of_expressions_l340_340243

theorem max_value_of_expressions :
  ∃ x ∈ ℝ, (∀ y ∈ ℝ, (2^y - 4^y) ≤ (2^x - 4^x)) ∧ (2^x - 4^x = 1 / 4) :=
by
  sorry

end max_value_of_expressions_l340_340243


namespace max_distance_on_curve_l340_340279

open Real -- Open the real number module for easier access to real number functions

theorem max_distance_on_curve :
  ∀ (x y : ℝ), (y^2 = 4 - 2 * x^2) → -2 ≤ y ∧ y ≤ 2 →
  ∃ (p a : ℝ), p = sqrt(x^2 + (y + sqrt(2))^2) ∧ p = 2 + sqrt(2) :=
by sorry

end max_distance_on_curve_l340_340279


namespace inscribe_triangle_similar_l340_340418

variables {A B C P X Y L M N : Type} [Point A] [Point B] [Point C] [Point P] [Point X] [Point Y] [Point L] [Point M] [Point N]
variables {k : ℝ} {phi : ℝ}
variables [hp : OnLineSegment A B P] [hx : OnLineSegment A C X] [hy : OnLineSegment C B Y]

def triangle_similar (A B C D E F : Point) : Prop :=
  ∃ (k : ℝ) (phi : ℝ), 
    ∀ (σ : triangle A B C → triangle D E F), 
    (∃ p : Point, 
       p = P ∧ 
       rotation_homothety k phi p (triangle A B C) (triangle D E F) ∧
       ∀ (θ : angle A C), 
         θ = AngleBetween (p, X, Y) ∧ 
         θ = AngleBetween (L, M, N) ∧ 
         CorrespondingSides InProportion (triangle A B C) (triangle D E F))

theorem inscribe_triangle_similar : 
  OnLineSegment A B P →
  OnLineSegment A C X →
  OnLineSegment C B Y →
  ∃ k : ℝ,
  k = (Length (P, Y)) / (Length (P, X)) →
  ∃ phi : ℝ,
  phi = AngleBetween (P, X, Y) →
  triangle_similar A C P L M N :=
by sorry

end inscribe_triangle_similar_l340_340418


namespace count_int_solutions_ineq_l340_340249

theorem count_int_solutions_ineq : 
  (finset.filter (λ x : ℤ, -6 ≤ 3 * x - 2 ∧ 3 * x - 2 ≤ 9) (finset.Icc (-10) 10)).card = 5 := 
by 
  sorry

end count_int_solutions_ineq_l340_340249


namespace number_of_ways_l340_340992

theorem number_of_ways (students : Fin 10 → Type) (A B C : Fin 10) (chosen : Finset (Fin 10)) :
  chosen.card = 3 →
  (B ∉ chosen) →
  (A ∈ chosen ∨ C ∈ chosen) →
  nat.choose 9 3 - nat.choose 7 3 = 49 := 
by
  sorry

end number_of_ways_l340_340992


namespace height_of_middle_person_is_l340_340254

noncomputable def height_of_middle_person : ℝ :=
  let heights := [3, 3 * (real.sqrt (7/3)) ^ 1, 3 * (real.sqrt (7/3)) ^ 2, 3 * (real.sqrt (7/3)) ^ 3, 7]
  heights.nth 2.get_or_else 0

theorem height_of_middle_person_is :
  -- Stating the conditions
  let h1 := 3
  let h5 := 7
  ∃ (r : ℝ), 
    (r ^ 4 = 7 / 3) ∧
    heights = [h1, h1 * r, h1 * r ^ 2, h1 * r ^ 3, h5] ∧
  
  -- Prove the middle person's height
  height_of_middle_person = real.sqrt 21 := 
by {
  sorry
}

end height_of_middle_person_is_l340_340254


namespace total_broken_marbles_l340_340264

def percentage_of (percent : ℝ) (total : ℝ) : ℝ := percent * total / 100

theorem total_broken_marbles :
  let first_set_total := 50
  let second_set_total := 60
  let first_set_percent_broken := 10
  let second_set_percent_broken := 20
  let first_set_broken := percentage_of first_set_percent_broken first_set_total
  let second_set_broken := percentage_of second_set_percent_broken second_set_total
  first_set_broken + second_set_broken = 17 :=
by
  sorry

end total_broken_marbles_l340_340264


namespace cube_eq_minus_one_l340_340126

theorem cube_eq_minus_one (x : ℝ) (h : x = -2) : (x + 1) ^ 3 = -1 :=
by
  sorry

end cube_eq_minus_one_l340_340126


namespace kids_go_to_camp_l340_340374

variable (total_kids staying_home going_to_camp : ℕ)

theorem kids_go_to_camp (h1 : total_kids = 313473) (h2 : staying_home = 274865) (h3 : going_to_camp = total_kids - staying_home) :
  going_to_camp = 38608 :=
by
  sorry

end kids_go_to_camp_l340_340374


namespace dealer_sold_92_fords_l340_340519

theorem dealer_sold_92_fords :
  ∀ (total_cars : ℕ) (bmw_percentage toyota_percentage nissan_percentage : ℕ),
  total_cars = 250 →
  bmw_percentage = 18 →
  toyota_percentage = 20 →
  nissan_percentage = 25 →
  let non_ford_percentage := bmw_percentage + toyota_percentage + nissan_percentage in
  let ford_percentage := 100 - non_ford_percentage in
  let ford_cars := total_cars * ford_percentage / 100 in
  ford_cars = 92 :=
begin
  sorry
end

end dealer_sold_92_fords_l340_340519


namespace candidates_difference_l340_340675

theorem candidates_difference
  (total_candidates : ℕ)
  (percentage_A : ℚ)
  (percentage_B : ℚ)
  (candidates_A : ℕ)
  (candidates_B : ℕ)
  (total_candidates_eq : total_candidates = 8400)
  (percentage_A_eq : percentage_A = 6 / 100)
  (percentage_B_eq : percentage_B = 7 / 100)
  (candidates_A_eq : candidates_A = (percentage_A * total_candidates).to_nat)
  (candidates_B_eq : candidates_B = (percentage_B * total_candidates).to_nat) :
  candidates_B - candidates_A = 84 := sorry

end candidates_difference_l340_340675


namespace find_k_l340_340958

theorem find_k 
  (x y k : ℚ) 
  (h1 : y = 4 * x - 1) 
  (h2 : y = -1 / 3 * x + 11) 
  (h3 : y = 2 * x + k) : 
  k = 59 / 13 :=
sorry

end find_k_l340_340958


namespace percentage_alcohol_solution_x_is_ten_percent_l340_340902

-- Define the conditions
variables (P : ℝ) -- P is the percentage of alcohol by volume in solution x

-- Solution y is 30% alcohol by volume, so we let Py = 0.30
def Py : ℝ := 0.30

-- The volume of solution y added
def volume_y : ℝ := 300

-- The volume of solution x
def volume_x : ℝ := 100

-- The desired alcohol percentage of the new solution is 25%, so we let Pmix = 0.25
def Pmix : ℝ := 0.25

-- The total volume of the new solution
def total_volume : ℝ := volume_y + volume_x

-- The total amount of alcohol from solution y
def alcohol_y : ℝ := volume_y * Py

-- Statement to be proven
theorem percentage_alcohol_solution_x_is_ten_percent :
  100 * P + alcohol_y = total_volume * Pmix → P = 0.10 :=
begin
  sorry,
end

end percentage_alcohol_solution_x_is_ten_percent_l340_340902


namespace factor_correct_l340_340970

noncomputable def p (b : ℝ) : ℝ := 221 * b^2 + 17 * b
def factored_form (b : ℝ) : ℝ := 17 * b * (13 * b + 1)

theorem factor_correct (b : ℝ) : p b = factored_form b := by
  sorry

end factor_correct_l340_340970


namespace find_f_find_range_lambda_find_lambda_min_value_l340_340606

noncomputable def quadratic_func (f : ℝ → ℝ) :=
  ∃ a b : ℝ, (a ≠ 0) ∧ (f = λ x, a * x^2 + b * x) ∧ (f 0 = 0) ∧ (-b / (2 * a) = -1 / 2) ∧ ( ∃ x, f x = x )

noncomputable def g_func (f : ℝ → ℝ) (λ : ℝ) :=
  (λ x, f x - (1 + 2 * λ) * x)

theorem find_f :
  quadratic_func (λ x, x^2 + x) :=
begin
  use [1, 1],
  split,
  { intro h, exact one_ne_zero h },
  split,
  { refl },
  split,
  { refl },
  split,
  { norm_num },
  { use [0],
    refl }
end

theorem find_range_lambda (λ : ℝ) :
  g_func (λ x, x^2 + x) λ ≥ -1 ↔ -1 ≤ λ ∧ λ ≤ 1 :=
begin
  sorry -- proof to be filled in by the mathematician.
end

theorem find_lambda_min_value (λ : ℝ) :
  (∃ x, x ∈ set.Icc (-2 : ℝ) 1 ∧ g_func (λ x, x^2 + x) λ x = -3) ↔ λ = 2 :=
begin
  sorry -- proof to be filled in by the mathematician.
end

end find_f_find_range_lambda_find_lambda_min_value_l340_340606


namespace coeff_x2_least_then_coeff_x7_156_l340_340307

-- Let's define the necessary conditions
variable (m n : Nat)

-- Define the function f(x) and its conditions
def f (x : ℕ) : ℕ := bin (m) x + bin (n) x

-- Define the main theorem to prove the problem statement
theorem coeff_x2_least_then_coeff_x7_156 (m n : ℕ) (h : m + n = 19) :
  f 1 = 19 → 
  (∃ k : ℕ, k = min (n^2 - 19*n + 171) :=
  ∃ l : ℕ, (n = 9 ↔ m = 10) ∨ (n = 10 ↔ m = 9) → fam_ecoeff_x7 = 156 := 
sorry

end coeff_x2_least_then_coeff_x7_156_l340_340307


namespace minimum_pebbles_on_chessboard_l340_340113

-- Defining the dimensions and the main conditions.
def chessboard_width : ℕ := 2013
def square_width : ℕ := 19
def minimum_pebbles_per_19x19_square : ℕ := 21

-- The main statement
theorem minimum_pebbles_on_chessboard :
  ∀ (board : Fin chessboard_width × Fin chessboard_width → ℕ),
  (∀ x y : Fin chessboard_width, board (x, y) ≤ 1) →
  (∀ x y : Fin (chessboard_width - square_width + 1),
     ((λ i j, board (i, j)) '' {i // (x : ℕ) ≤ i ∧ i < x + square_width} ×
                               {j // (y : ℕ) ≤ j ∧ j < y + square_width}).sum ≥ minimum_pebbles_per_19x19_square) →
  ∑ x y, board (x, y) ≥ 160175889 :=
begin
  sorry
end

end minimum_pebbles_on_chessboard_l340_340113


namespace problem_statement_l340_340709

open Nat

def gcd (a b : ℕ) : ℕ := if b = 0 then a else gcd b (a % b)
def lcm (a b : ℕ) : ℕ := a * (b / gcd a b)

theorem problem_statement :
  (∀ f : ℕ → ℕ, (∀ m n : ℕ, m > 0 ∧ n > 0 → gcd (f m) n + lcm m (f n) = gcd m (f n) + lcm (f m) n)
  → (∀ n : ℕ, n > 0 → f n = n)) :=
begin
  sorry,
end

end problem_statement_l340_340709


namespace count_triangles_in_hexagonal_grid_l340_340652

-- Define the number of smallest triangles in the figure.
def small_triangles : ℕ := 10

-- Define the number of medium triangles in the figure, composed of 4 small triangles each.
def medium_triangles : ℕ := 6

-- Define the number of large triangles in the figure, composed of 9 small triangles each.
def large_triangles : ℕ := 3

-- Define the number of extra-large triangle composed of 16 small triangles.
def extra_large_triangle : ℕ := 1

-- Define the total number of triangles in the figure.
def total_triangles : ℕ := small_triangles + medium_triangles + large_triangles + extra_large_triangle

-- The theorem we want to prove: the total number of triangles is 20.
theorem count_triangles_in_hexagonal_grid : total_triangles = 20 := by
  -- Placeholder for the proof.
  sorry

end count_triangles_in_hexagonal_grid_l340_340652


namespace area_GCD_48_l340_340066

/-- The given geometric setup and conditions -/
variables {A B C D E F G : Type} [geometry.space ℝ A B C D]
variables (hABCD : square ABCD 256)
variables (hE_ratio : divides BC E (3:1))
variables (hF : midpoint A E F)
variables (hG : midpoint D E G)
variables (hBEGF_area : area BEGF = 48)

/-- The main theorem to prove the area of triangle GCD is 48 square units -/
theorem area_GCD_48 (hABCD : square ABCD 256)
                     (hE_ratio : divides BC E (3:1))
                     (hF : midpoint A E F)
                     (hG : midpoint D E G)
                     (hBEGF_area : area BEGF = 48) :
  area GCD = 48 :=
sorry

end area_GCD_48_l340_340066


namespace student_selection_probability_l340_340533

theorem student_selection_probability :
  ∀ (n_students total_students eliminated_students sampled_students: ℕ),
  total_students = 2012 → eliminated_students = 12 → n_students = 50 → 
  sample_systematic (total_students - eliminated_students) =
  n_students →
  probability_of_selection total_students eliminated_students n_students = 50 / 2012 :=
begin
  sorry
end

-- Definitions used in the theorem statement
def sample_systematic (remaining_students: ℕ) : ℕ :=
  if remaining_students = 2000 then 50 else 0

def probability_of_selection (total_students eliminated_students n_students: ℕ) : ℚ :=
  (n_students.to_rat) / (total_students.to_rat - eliminated_students.to_rat)

end student_selection_probability_l340_340533


namespace weight_of_A_l340_340444

variable (A B C D E : ℕ)

axiom cond1 : A + B + C = 180
axiom cond2 : A + B + C + D = 260
axiom cond3 : E = D + 3
axiom cond4 : B + C + D + E = 256

theorem weight_of_A : A = 87 :=
by
  sorry

end weight_of_A_l340_340444


namespace maximum_subset_cardinality_l340_340397
noncomputable def max_subset_cardinality : ℕ :=
  let S := {1, 2, ..., 2002} in  -- This defines the set S
  max { T : set ℕ | T ⊆ S ∧ ∀ a b ∈ T, a * b ∉ T }

theorem maximum_subset_cardinality (S : set ℕ) (hS : S = {1..2002}) :
  let max_card := max { T : set ℕ | T ⊆ S ∧ ∀ a b ∈ T, a * b ∉ T } in
  max_card = 1958 := sorry

end maximum_subset_cardinality_l340_340397


namespace exists_triangle_with_angle_divided_into_four_equal_parts_l340_340005

theorem exists_triangle_with_angle_divided_into_four_equal_parts
  (a b c : ℝ) (α β γ : ℝ)
  (h_triangle : a ≠ b ∧ a > b)
  (h_gamma_division : γ = 22.5 * (Real.pi / 180) * 4) : 
  ∃ (triangle : Type) (angle : ℝ), 
  (triangle = Triangle a b c α β γ) ∧ 
  (angle = 90 * (Real.pi / 180)) := sorry

end exists_triangle_with_angle_divided_into_four_equal_parts_l340_340005


namespace fourth_vertex_of_parallelogram_l340_340091

structure Point where
  x : ℝ
  y : ℝ

def Q := Point.mk 1 (-1)
def R := Point.mk (-1) 0
def S := Point.mk 0 1
def V := Point.mk (-2) 2

theorem fourth_vertex_of_parallelogram (Q R S V : Point) :
  Q = ⟨1, -1⟩ ∧ R = ⟨-1, 0⟩ ∧ S = ⟨0, 1⟩ → V = ⟨-2, 2⟩ := by 
  sorry

end fourth_vertex_of_parallelogram_l340_340091


namespace ratio_of_pentagon_side_to_rectangle_width_l340_340174

-- Definitions based on the conditions
def pentagon_perimeter : ℝ := 60
def rectangle_perimeter : ℝ := 60
def rectangle_length (w : ℝ) : ℝ := 2 * w

-- The statement to be proven
theorem ratio_of_pentagon_side_to_rectangle_width :
  ∀ w : ℝ, 2 * (rectangle_length w + w) = rectangle_perimeter → (pentagon_perimeter / 5) / w = 6 / 5 :=
by
  sorry

end ratio_of_pentagon_side_to_rectangle_width_l340_340174


namespace Rhett_salary_l340_340058

universe u

variable (S : ℝ)

def monthly_rent : ℝ := 1350
def tax_rate : ℝ := 0.1
def rent_payments : ℝ := 2 * monthly_rent
def fraction_of_salary_after_tax : ℝ := 3 / 5
def salary_after_tax (S : ℝ) : ℝ := (1 - tax_rate) * S
def rent_payment_covered_fraction (S : ℝ) : Prop := (fraction_of_salary_after_tax * salary_after_tax S = rent_payments)

theorem Rhett_salary :
  rent_payment_covered_fraction S → S = 5000 := 
by
  sorry

end Rhett_salary_l340_340058


namespace registration_methods_count_l340_340255

theorem registration_methods_count (students groups : ℕ) (h_students : students = 5) (h_groups : groups = 2) : 
  (groups ^ students) = 32 := 
by 
  rw [h_students, h_groups]
  exact pow_succ 2 4 -- 2^5 = 32 can be simplified as 2^4 * 2 = 16 * 2 = 32
  sorry 

end registration_methods_count_l340_340255


namespace range_of_omega_l340_340453

theorem range_of_omega (ω : ℝ) (h : 0 < ω ∧ ω ≤ 2) :
  ∀ x ∈ Set.Ioo (π / 6) (π / 3), Deriv (λ x, 2 * sin (ω * x - π / 6)) x > 0 :=
by
  sorry

end range_of_omega_l340_340453


namespace line_tangent_circle_line_intersects_circle_line_outside_circle_l340_340917

theorem line_tangent_circle (O A B C P Q R : Point) (m l : Line) (r : ℝ) (x : ℝ) :
  (is_tangent l O r) ∧
  (on_line A l) ∧ (on_line B l) ∧ (on_line C l) ∧ 
  (on_line M m) ∧ (perpendicular l m) ∧ 
  (tangent_point A P O) ∧ (tangent_point B Q O) ∧ (tangent_point C R O) →
  (AB • CR + BC • AP = AC • BQ) := sorry

theorem line_intersects_circle (O A B C P Q R : Point) (m l : Line) (r : ℝ) (x : ℝ) :
  (intersects l O r) ∧
  (on_line A l) ∧ (on_line B l) ∧ (on_line C l) ∧ 
  (on_line M m) ∧ (perpendicular l m) ∧ 
  (tangent_point A P O) ∧ (tangent_point B Q O) ∧ (tangent_point C R O) →
  (AB • CR + BC • AP < AC • BQ) := sorry

theorem line_outside_circle (O A B C P Q R : Point) (m l : Line) (r : ℝ) (x : ℝ) :
  (does_not_intersect l O r) ∧
  (on_line A l) ∧ (on_line B l) ∧ (on_line C l) ∧ 
  (on_line M m) ∧ (perpendicular l m) ∧ 
  (tangent_point A P O) ∧ (tangent_point B Q O) ∧ (tangent_point C R O) →
  (AB • CR + BC • AP > AC • BQ) := sorry

end line_tangent_circle_line_intersects_circle_line_outside_circle_l340_340917


namespace equilateral_triangle_side_length_l340_340764

theorem equilateral_triangle_side_length (P Q R S A B C : Type)
  [Point P] [Point Q] [Point R] [Point S] [Point A] [Point B] [Point C] :
  (within_triangle P A B C) →
  orthogonal_projection P Q A B →
  orthogonal_projection P R B C →
  orthogonal_projection P S C A →
  distance P Q = 2 →
  distance P R = 3 →
  distance P S = 4 →
  distance A B = 6 * √3 :=
by
  sorry

end equilateral_triangle_side_length_l340_340764


namespace trip_cost_l340_340420

def distance (x y : ℝ) : ℝ := real.sqrt (x * x + y * y)

def travel_cost_bus (dist : ℝ) : ℝ := 0.20 * dist

def travel_cost_airplane (dist : ℝ) : ℝ := 120 + 0.12 * dist

def minimum (a b : ℝ) : ℝ := if a < b then a else b

theorem trip_cost : 
  let AB := 4250 
  let AC := 4000
  let BC := real.sqrt (AB * AB - AC * AC) -- Using Pythagorean theorem
  let cost_A_to_B := minimum (travel_cost_airplane AB) (travel_cost_bus AB)
  let cost_B_to_C := minimum (travel_cost_airplane BC) (travel_cost_bus BC)
  let cost_C_to_A := minimum (travel_cost_airplane AC) (travel_cost_bus AC) 
  in 
  cost_A_to_B + cost_B_to_C + cost_C_to_A = 1520 :=
by 
  -- Definitions and intermediate calculations are encapsulated in let bindings
  let AB := 4250 
  let AC := 4000
  let BC := real.sqrt (AB * AB - AC * AC) -- BC = √(AB² - AC²)
  let cost_A_to_B := minimum (travel_cost_airplane AB) (travel_cost_bus AB)
  let cost_B_to_C := minimum (travel_cost_airplane BC) (travel_cost_bus BC)
  let cost_C_to_A := minimum (travel_cost_airplane AC) (travel_cost_bus AC) 
  have BC_calculation : BC = 1450, -- Insert calculation or simplification steps here
    { sorry }
  have cost_A_to_B_correct : cost_A_to_B = 630, 
    { sorry }
  have cost_B_to_C_correct : cost_B_to_C = 290, 
    { sorry }
  have cost_C_to_A_correct : cost_C_to_A = 600, 
    { sorry }
  calc cost_A_to_B + cost_B_to_C + cost_C_to_A 
    = 630 + 290 + 600 : by rw [cost_A_to_B_correct, cost_B_to_C_correct, cost_C_to_A_correct]
... = 1520 : by norm_num
  -- proof ends here


end trip_cost_l340_340420


namespace largest_number_is_base6_l340_340194

noncomputable def base9_to_dec (n : ℕ) : ℕ :=
  8 * 9 + 5

noncomputable def base6_to_dec (n : ℕ) : ℕ :=
  2 * 36 + 1 * 6 + 0

noncomputable def base4_to_dec (n : ℕ) : ℕ :=
  1 * 64 + 0 * 16 + 0 * 4 + 0

noncomputable def base2_to_dec (n : ℕ) : ℕ :=
  1 * 16 + 1 * 8 + 1 * 4 + 1 * 2 + 1

theorem largest_number_is_base6 :
  ∀ n, base6_to_dec n > base9_to_dec n ∧ base6_to_dec n > base4_to_dec n ∧ base6_to_dec n > base2_to_dec n :=
by
  intro n
  simp [base9_to_dec, base6_to_dec, base4_to_dec, base2_to_dec]
  sorry

end largest_number_is_base6_l340_340194


namespace max_total_length_inside_circle_l340_340835

variable (P : Type) [MetricSpace P]

/-- Let ABC be an equilateral triangle with circumradius 1. -/
def equilateral_triangle (ABC : Triangle P) : Prop :=
  circumradius ABC = 1

/-- The total distance from a point P inside an equilateral triangle to its sides is constant. -/
def total_distance_to_sides (P : P) (ABC : Triangle P) : Prop :=
  ∑ (d : P → ℝ) in (point_to_sides_distances P ABC), d = 3 / 2

/-- Define the maximum total length of the sides lying inside a circle with center P and radius 1. -/
theorem max_total_length_inside_circle (P : P) (ABC : Triangle P) (h1 : equilateral_triangle ABC) (h2 : total_distance_to_sides P ABC) :
  maximum_sides_length_inside_circle P ABC 1 = 3 * Real.sqrt 3 :=
sorry

end max_total_length_inside_circle_l340_340835


namespace cube_volume_surface_area_l340_340491

-- Define volume and surface area conditions
def volume_condition (x : ℝ) (s : ℝ) : Prop := s^3 = 3 * x
def surface_area_condition (x : ℝ) (s : ℝ) : Prop := 6 * s^2 = x

-- The main theorem statement
theorem cube_volume_surface_area (x : ℝ) (s : ℝ) :
  volume_condition x s → surface_area_condition x s → x = 5832 :=
by
  intros h_volume h_area
  sorry

end cube_volume_surface_area_l340_340491


namespace coverage_of_circle_l340_340052

theorem coverage_of_circle (radius_main : ℝ) (radius_small : ℝ) (n : ℕ) 
  (h_main: radius_main = 2) 
  (h_small: radius_small = 1) 
  (h_num: n = 7) : 
  (∃ (positions : fin n → ℝ × ℝ), ∀ (x y : ℝ), (x ^ 2 + y ^ 2 ≤ radius_main ^ 2) → 
  ∃ (i : fin n), ((x - (positions i).1) ^ 2 + (y - (positions i).2) ^ 2 ≤ radius_small ^ 2)) ∧ 
  (∀ (R : ℝ), R > 2 → ¬(∃ (positions : fin n → ℝ × ℝ), ∀ (x y : ℝ), (x ^ 2 + y ^ 2 ≤ R ^ 2) → 
  ∃ (i : fin n), ((x - (positions i).1) ^ 2 + (y - (positions i).2) ^ 2 ≤ radius_small ^ 2))) := 
sorry

end coverage_of_circle_l340_340052


namespace net_salary_change_l340_340186

theorem net_salary_change (S : ℝ) : 
  let after_first_increase := S * 1.20 in
  let after_second_increase := after_first_increase * 1.40 in
  let after_first_decrease := after_second_increase * 0.65 in
  let final_salary := after_first_decrease * 0.75 in
  let net_change := final_salary - S in
  let percentage_change := (net_change / S) * 100 in
  percentage_change = -18.1 := 
by
  sorry

end net_salary_change_l340_340186


namespace pointP_outside_triangle_l340_340231

/-- Definition of the line equations and point P --/
def line1 (x y : ℝ) : Prop := 8*x - 15*y - 35 = 0
def line2 (x y : ℝ) : Prop := x - 2*y - 2 = 0
def line3 (x y : ℝ) : Prop := y = 0

def pointP : ℝ × ℝ := (15.2, 12.4)

/-- Theorem to determine if point P is outside the triangle formed by the lines --/
theorem pointP_outside_triangle 
    (P : ℝ × ℝ)
    (l1 : ℝ × ℝ → Prop)
    (l2 : ℝ × ℝ → Prop)
    (l3 : ℝ × ℝ → Prop) :
    (P = (15.2, 12.4)) →
    (l1 = line1) →
    (l2 = line2) →
    (l3 = line3) →
    ¬ (∃ (x : ℝ) (y : ℝ), l1 (x, y) ∧ l2 (x, y) ∧ l3 (x, y) ∧ x = 15.2 ∧ y = 12.4) :=
by
  intros P_eq l1_eq l2_eq l3_eq
  sorry

end pointP_outside_triangle_l340_340231


namespace simplify_expression_l340_340435

variable (x : ℝ)

theorem simplify_expression :
  (3 * x^6 + 2 * x^5 + x^4 + x - 5) - (x^6 + 3 * x^5 + 2 * x^3 + 6) = 2 * x^6 - x^5 + x^4 - 2 * x^3 + x + 1 := by
  sorry

end simplify_expression_l340_340435


namespace total_scoops_l340_340739

-- Define the conditions as variables
def flourCups := 3
def sugarCups := 2
def scoopSize := 1/3

-- Define what needs to be proved, i.e., the total amount of scoops needed
theorem total_scoops (flourCups sugarCups : ℚ) (scoopSize : ℚ) : 
  (flourCups / scoopSize) + (sugarCups / scoopSize) = 15 := 
by
  sorry

end total_scoops_l340_340739


namespace side_length_eq_l340_340749

namespace EquilateralTriangle

variables (A B C P Q R S : Type) [HasVSub Type P] [MetricSpace P]
variables [HasDist P] [HasEquilateralTriangle ABC] [InsideTriangle P ABC]
variables [Perpendicular PQ AB] [Perpendicular PR BC] [Perpendicular PS CA]
variables [Distance PQ 2] [Distance PR 3] [Distance PS 4]

theorem side_length_eq : side_length ABC = 6 * √3 :=
sorry
end EquilateralTriangle

end side_length_eq_l340_340749


namespace determine_a_l340_340954

theorem determine_a (a : ℝ): (∃ b : ℝ, 27 * x^3 + 9 * x^2 + 36 * x + a = (3 * x + b)^3) → a = 8 := 
by
  sorry

end determine_a_l340_340954


namespace minimum_f_x_l340_340146

noncomputable def f (x : ℝ) : ℝ := Real.sqrt(x^2 - 4 * x + 5) + Real.sqrt(x^2 + 4 * x + 8)

theorem minimum_f_x : ∃ x_min : ℝ, (∀ x : ℝ, f x_min ≤ f x) ∧ f x_min = 5 :=
by
  -- sorry is used as a placeholder for the actual proof
  sorry

end minimum_f_x_l340_340146


namespace four_nabla_seven_l340_340481

-- Define the operation ∇
def nabla (a b : ℤ) : ℚ :=
  (a + b) / (1 + a * b)

theorem four_nabla_seven :
  nabla 4 7 = 11 / 29 :=
by
  sorry

end four_nabla_seven_l340_340481


namespace last_three_digits_7_pow_80_l340_340980

theorem last_three_digits_7_pow_80 : (7^80) % 1000 = 961 := by
  sorry

end last_three_digits_7_pow_80_l340_340980


namespace concurrency_of_lines_l340_340621

/-- Given that Circle ΓA passes through points B and C, 
    and is tangent to the incircle Ι of triangle ABC at point KA,
    and similarly defining points KB and KC for Circles ΓB and ΓC respectively.
    Prove that lines AKA, BKB, and CKC are concurrent. --/
theorem concurrency_of_lines
  {A B C K_A K_B K_C I : Type}
  (circle_GammaA : ∀ {P}, P = B ∨ P = C ∨ (tangent P I) ∧ (P = K_A → on_circle P ΓA))
  (circle_GammaB : ∀ {P}, P = A ∨ P = C ∨ (tangent P I) ∧ (P = K_B → on_circle P ΓB))
  (circle_GammaC : ∀ {P}, P = A ∨ P = B ∨ (tangent P I) ∧ (P = K_C → on_circle P ΓC))
  (triangle_ABC : Triangle A B C) :
  concurrent [line_through A K_A, line_through B K_B, line_through C K_C] :=
sorry

end concurrency_of_lines_l340_340621


namespace solution_set_fraction_inequality_l340_340466

theorem solution_set_fraction_inequality (x : ℝ) : 
  (x + 1) / (x - 1) ≤ 0 ↔ -1 ≤ x ∧ x < 1 :=
sorry

end solution_set_fraction_inequality_l340_340466


namespace point_on_parabola_touching_x_axis_l340_340802

theorem point_on_parabola_touching_x_axis (a b c : ℤ) (h : ∃ r : ℤ, a * (r * r) + b * r + c = 0 ∧ (r * r) = 0) :
  ∃ (a' b' : ℤ), ∃ k : ℤ, (k * k) + a' * k + b' = 0 ∧ (k * k) = 0 :=
sorry

end point_on_parabola_touching_x_axis_l340_340802


namespace cos_x_half_lt_zero_sin_x_lt_cos_x_half_imp_cos_x_lt_half_l340_340256

theorem cos_x_half_lt_zero_sin_x_lt_cos_x_half_imp_cos_x_lt_half 
    {x : ℝ} 
    (h1 : sin x < cos (x / 2)) 
    (h2 : cos (x / 2) < 0) : 
    cos x < 1 / 2 := 
sorry

end cos_x_half_lt_zero_sin_x_lt_cos_x_half_imp_cos_x_lt_half_l340_340256


namespace vertex_is_correct_l340_340305

-- Define the equation of the parabola
def parabola_equation (x y : ℝ) : Prop :=
  y^2 + 10 * y + 4 * x + 9 = 0

-- The vertex of the parabola
def vertex_of_parabola : ℝ × ℝ := (4, -5)

-- The theorem stating that the given vertex satisfies the parabola equation
theorem vertex_is_correct : 
  parabola_equation vertex_of_parabola.1 vertex_of_parabola.2 :=
sorry

end vertex_is_correct_l340_340305


namespace standard_equation_of_hyperbola_l340_340870

noncomputable def ellipse_eccentricity_problem
  (e : ℚ) (a_maj : ℕ) (f_1 f_2 : ℝ × ℝ) (d : ℕ) : Prop :=
  e = 5 / 13 ∧
  a_maj = 26 ∧
  f_1 = (-5, 0) ∧
  f_2 = (5, 0) ∧
  d = 8 →
  ∃ b, (2 * b = 3) ∧ (2 * b ≠ 0) ∧
  ∃ h k : ℝ, (0 ≤  h) ∧ (0 ≤ k) ∧
  ((h^2)/(4^2)) - ((k^2)/(3^2)) = 1

-- problem statement: 
theorem standard_equation_of_hyperbola
  (e : ℚ) (a_maj : ℕ) (f_1 f_2 : ℝ × ℝ) (d : ℕ)
  (h : e = 5 / 13)
  (a_maj_length : a_maj = 26)
  (f1_coords : f_1 = (-5, 0))
  (f2_coords : f_2 = (5, 0))
  (distance_diff : d = 8) :
  ellipse_eccentricity_problem e a_maj f_1 f_2 d :=
sorry

end standard_equation_of_hyperbola_l340_340870


namespace equilateral_triangle_side_length_l340_340761

theorem equilateral_triangle_side_length 
  {P Q R S : Point} 
  {A B C : Triangle}
  (h₁ : is_inside P A B C)
  (h₂ : is_perpendicular P Q A B)
  (h₃ : is_perpendicular P R B C)
  (h₄ : is_perpendicular P S C A)
  (h₅ : distance P Q = 2)
  (h₆ : distance P R = 3)
  (h₇ : distance P S = 4)
  : side_length A B C = 6 * sqrt 3 :=
by 
  sorry

end equilateral_triangle_side_length_l340_340761


namespace mod_inverse_sum_l340_340837

theorem mod_inverse_sum :
  (7⁻¹ : ZMod 17) + (7⁻² : ZMod 17) = 13 :=
by
  sorry

end mod_inverse_sum_l340_340837


namespace maximal_number_blackboard_l340_340437

/-- 
 Given some distinct positive integers on a blackboard such that the sum of any two distinct integers is a power of 2. Prove that the maximal number on the blackboard can be 
 of the form 2^k - 1 for some large k.
 -/
theorem maximal_number_blackboard (S : Finset ℕ) (h : ∀ ⦃a b⦄, a ∈ S → b ∈ S → a ≠ b → ∃ k, a + b = 2^k) : 
  ∃ k, ∀ a ∈ S, a ≤ 2^k - 1 :=
sorry

end maximal_number_blackboard_l340_340437


namespace false_propositions_l340_340496

open Classical

theorem false_propositions :
  ¬ (∀ x : ℝ, x^2 + 3 < 0) ∧ ¬ (∀ x : ℕ, x^2 > 1) ∧ (∃ x : ℤ, x^5 < 1) ∧ ¬ (∃ x : ℚ, x^2 = 3) :=
by
  sorry

end false_propositions_l340_340496


namespace num_tangent_circles_l340_340028

-- Circle definition with center and radius
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Conditions
def C_1 : Circle := { center := (0, 0), radius := 2 }
def C_2 : Circle := { center := (4, 0), radius := 2 }

-- The statement to prove.
theorem num_tangent_circles (h1 : C_1.radius = 2) (h2 : C_2.radius = 2) (h3 : dist C_1.center C_2.center = 4) (h_line : ∀ x : ℝ, (0, 2) < x) : 
  ∃! c : Circle, c.radius = 4 ∧ tangent c C_1 ∧ tangent c C_2 ∧ tangent_to_line c h_line = 2 :=
by sorry

-- Definitions needed for the proof condition.
def dist (p1 p2 : ℝ × ℝ) : ℝ := (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

def tangent (c1 c2 : Circle) : Prop := 
  dist c1.center c2.center = c1.radius + c2.radius ∨ 
  dist c1.center c2.center = abs (c1.radius - c2.radius)

def tangent_to_line (c : Circle) (h_line : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, h_line x = c.radius

end num_tangent_circles_l340_340028


namespace angle_AOD_is_140_degrees_l340_340356

noncomputable def angle_AOD (x : ℝ) : ℝ := 3.5 * x

theorem angle_AOD_is_140_degrees (x : ℝ) 
  (H1 : ∠('OA) ⊥ ∠('OC))
  (H2 : ∠('OB) ⊥ ∠('OD))
  (H3 : angle_AOD x = 3.5 * x) :
  angle_AOD x = 140 :=
by
  sorry

end angle_AOD_is_140_degrees_l340_340356


namespace B_votes_correct_l340_340350

noncomputable def B_valid_votes (V : ℕ) : ℕ :=
  let valid_votes := (4 : ℕ) * V / 5  -- 80% of total votes
  let excess_votes := (3 : ℕ) * V / 20  -- 15% of total votes
  let B_votes := (valid_votes - excess_votes) / 2
  B_votes

theorem B_votes_correct : B_valid_votes 8720 = 2834 := by
  unfold B_valid_votes
  have V := 8720
  have valid_votes := (4 : ℕ) * V / 5
  have excess_votes := (3 : ℕ) * V / 20
  have B_votes := (valid_votes - excess_votes) / 2
  calc B_votes
      = (6976 - 1308) / 2 : by
        unfold valid_votes excess_votes
        rw [(mul_comm 4 V), (mul_comm 3 V)]
        unfold V
        norm_num
      ... = 2834 : by norm_num

end B_votes_correct_l340_340350


namespace sugar_per_larger_cookie_l340_340704

theorem sugar_per_larger_cookie (c₁ c₂ : ℕ) (s₁ s₂ : ℝ) (h₁ : c₁ = 50) (h₂ : s₁ = 1 / 10) (h₃ : c₂ = 25) (h₄ : c₁ * s₁ = c₂ * s₂) : s₂ = 1 / 5 :=
by
  simp [h₁, h₂, h₃, h₄]
  sorry

end sugar_per_larger_cookie_l340_340704


namespace amount_of_gain_l340_340891

-- Define the given conditions
def selling_price : ℝ := 195
def gain_percentage : ℝ := 0.30

-- Define the cost price and gain
def cost_price (SP : ℝ) (g : ℝ) : ℝ := SP / (1 + g)
def gain (SP : ℝ) (C : ℝ) : ℝ := SP - C

-- Define the main theorem
theorem amount_of_gain : gain selling_price (cost_price selling_price gain_percentage) = 45 :=
by 
  -- statements to be filled in the proof, using the above definitions and sorry for now
  sorry

end amount_of_gain_l340_340891


namespace find_tangent_points_l340_340468

def f (x : ℝ) : ℝ := x^3 + x - 2
def tangent_parallel_to_line (x : ℝ) : Prop := deriv f x = 4

theorem find_tangent_points :
  (tangent_parallel_to_line 1 ∧ f 1 = 0) ∧ 
  (tangent_parallel_to_line (-1) ∧ f (-1) = -4) :=
by
  sorry

end find_tangent_points_l340_340468


namespace sum_equals_target_l340_340559

-- Define the main sum expression.
noncomputable def sum_expression : ℚ := 
  ∑' (a : ℕ) (b : ℕ) (c : ℕ) in {p : ℕ × ℕ × ℕ | 1 ≤ p.1 ∧ p.1 < p.2 ∧ p.2 < p.3}.toFinset (λ p, 
    (1 : ℚ) / (3 ^ p.1 * 4 ^ p.2 * 6 ^ p.3)) 

-- Define the target value.
def target_value : ℚ := 1 / 106505

-- State the theorem to be proven.
theorem sum_equals_target : sum_expression = target_value :=
  sorry

end sum_equals_target_l340_340559


namespace lowest_score_is_C_l340_340681

variable (Score : Type) [LinearOrder Score]

/-- Scores of A, B, and C -/
variables (A B C : Score)

/-- Conditions given in the problem -/
theorem lowest_score_is_C (h1 : B < A → A < B ∧ A < C)
                         (h2 : C > A → A > B ∧ C < A) :
  C ≤ A ∧ C ≤ B :=
by {
  sorry
}

end lowest_score_is_C_l340_340681


namespace min_value_am_gm_seq_l340_340281

theorem min_value_am_gm_seq (a : ℕ → ℝ) (m n : ℕ) (a1 : ℝ)
  (h_pos : ∀ k, 0 < a k)
  (h_seq : a 7 = a 6 + 2 * a 5)
  (h_terms : sqrt (a m * a n) = 4 * a 1)
  (hmn : m + n = 6) :
  (1 / m + 4 / n) ≥ 3 / 2 :=
sorry

end min_value_am_gm_seq_l340_340281


namespace triangle_problems_l340_340692

variables {A B C : ℝ}
variables {a b c : ℝ}
variables {S : ℝ}

theorem triangle_problems
  (h1 : (2 * b - c) * Real.cos A = a * Real.cos C)
  (h2 : a = Real.sqrt 13)
  (h3 : b + c = 5) :
  (A = π / 3) ∧ (S = Real.sqrt 3) :=
by
  sorry

end triangle_problems_l340_340692


namespace equilateral_triangle_side_length_l340_340768

theorem equilateral_triangle_side_length (P Q R S A B C : Type)
  [Point P] [Point Q] [Point R] [Point S] [Point A] [Point B] [Point C] :
  (within_triangle P A B C) →
  orthogonal_projection P Q A B →
  orthogonal_projection P R B C →
  orthogonal_projection P S C A →
  distance P Q = 2 →
  distance P R = 3 →
  distance P S = 4 →
  distance A B = 6 * √3 :=
by
  sorry

end equilateral_triangle_side_length_l340_340768


namespace arithmetic_sequence_l340_340389

theorem arithmetic_sequence (a : ℕ → ℝ) 
    (h : ∀ m n, |a m + a n - a (m + n)| ≤ 1 / (m + n)) :
    ∃ d, ∀ k, a k = k * d := 
sorry

end arithmetic_sequence_l340_340389


namespace largest_four_digit_number_conditioned_l340_340882

theorem largest_four_digit_number_conditioned (a b c d : ℕ) :
  let n := 1000 * a + 100 * b + 10 * c + d in 
  10 ≤ 10*a + b ∧ 10*a + b ∣ 2014 ∧ c ≠ 0 ∧ 2014 ∣ (10*a + b) * (10*c + d) →
  n = 5376 :=
by
  sorry

end largest_four_digit_number_conditioned_l340_340882


namespace find_a_l340_340306

noncomputable def coefficient_of_x3_in_expansion (a : ℝ) : ℝ :=
  6 * a^2 - 15 * a + 20 

theorem find_a (a : ℝ) (h : coefficient_of_x3_in_expansion a = 56) : a = 6 ∨ a = -1 :=
  sorry

end find_a_l340_340306


namespace integer_values_of_x_for_positive_star_l340_340460

-- Definition of the operation star
def star (a b : ℕ) : ℚ := (a^2 : ℕ) / b

-- Problem statement
theorem integer_values_of_x_for_positive_star :
  ∃ (count : ℕ), count = 9 ∧ (∀ x : ℕ, (10^2 % x = 0) → (∃ n : ℕ, star 10 x = n)) :=
sorry

end integer_values_of_x_for_positive_star_l340_340460


namespace incorrect_eqn_x9_y9_neg1_l340_340662

theorem incorrect_eqn_x9_y9_neg1 (x y : ℂ) 
  (hx : x = (-1 + Complex.I * Real.sqrt 3) / 2) 
  (hy : y = (-1 - Complex.I * Real.sqrt 3) / 2) : 
  x^9 + y^9 ≠ -1 :=
sorry

end incorrect_eqn_x9_y9_neg1_l340_340662


namespace area_of_shaded_region_l340_340896

structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨0, 10⟩
def C : Point := ⟨12, 0⟩
def E : Point := ⟨16, 10⟩

def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

def area_of_triangle (p1 p2 p3 : Point) : ℝ :=
  (1/2) * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

theorem area_of_shaded_region : area_of_triangle A C E = 80 := by
  sorry

end area_of_shaded_region_l340_340896


namespace incorrect_eqn_x9_y9_neg1_l340_340663

theorem incorrect_eqn_x9_y9_neg1 (x y : ℂ) 
  (hx : x = (-1 + Complex.I * Real.sqrt 3) / 2) 
  (hy : y = (-1 - Complex.I * Real.sqrt 3) / 2) : 
  x^9 + y^9 ≠ -1 :=
sorry

end incorrect_eqn_x9_y9_neg1_l340_340663


namespace socks_ratio_l340_340910

/-- Alice ordered 6 pairs of green socks and some additional pairs of red socks. The price per pair
of green socks was three times that of the red socks. During the delivery, the quantities of the 
pairs were accidentally swapped. This mistake increased the bill by 40%. Prove that the ratio of the 
number of pairs of green socks to red socks in Alice's original order is 1:2. -/
theorem socks_ratio (r y : ℕ) (h1 : y * r ≠ 0) (h2 : 6 * 3 * y + r * y = (r * 3 * y + 6 * y) * 10 / 7) :
  6 / r = 1 / 2 :=
by
  sorry

end socks_ratio_l340_340910


namespace not_algebraic_expression_is_D_l340_340133

-- Define the four options
def optionA : Prop := ∃ (x : Real), 1 / x ∈ AlgebraicExpression
def optionB : Prop := ∃ (a : Real), 3 * a ^ 2 - a + 6 / (5 * π) ∈ AlgebraicExpression
def optionC : Prop := ∃ (π : Real), π / 3.14 ∈ AlgebraicExpression
def optionD : Prop := ¬ (∃ (π : Real), π ≈ 3.14 ∈ AlgebraicExpression)

-- The main theorem to prove
theorem not_algebraic_expression_is_D : optionD :=
by
  -- Proof is not necessary
  sorry

end not_algebraic_expression_is_D_l340_340133


namespace vector_eq_to_slope_intercept_form_l340_340526

theorem vector_eq_to_slope_intercept_form :
  ∀ (x y : ℝ), (2 * (x - 4) + 5 * (y - 1)) = 0 → y = -(2 / 5) * x + 13 / 5 := 
by 
  intros x y h
  sorry

end vector_eq_to_slope_intercept_form_l340_340526


namespace odd_function_example_l340_340948

def f (x : ℝ) (a : ℝ) : ℝ :=
  if 0 ≤ x then 1 / (2 ^ x) + a else -(1 / (2 ^ (-x))) + 1

theorem odd_function_example :
  ∀ (a : ℝ), (a = -1) → f (-1) a = 1 / 2 :=
by
  intro a ha
  rw ha
  sorry

end odd_function_example_l340_340948


namespace rearrange_infinite_decimal_l340_340055

-- Define the set of digits
def Digit : Type := Fin 10

-- Define the classes of digits
def Class1 (d : Digit) (dec : ℕ → Digit) : Prop :=
  ∃ n : ℕ, ∀ m : ℕ, m > n → dec m ≠ d

def Class2 (d : Digit) (dec : ℕ → Digit) : Prop :=
  ∀ n : ℕ, ∃ m : ℕ, m > n ∧ dec m = d

-- The statement to prove
theorem rearrange_infinite_decimal (dec : ℕ → Digit) (h : ∃ d : Digit, ¬ Class1 d dec) :
  ∃ rearranged : ℕ → Digit, (Class1 d rearranged ∧ Class2 d rearranged) →
  ∃ r : ℚ, ∃ n : ℕ, ∀ m ≥ n, rearranged m = rearranged (m + n) :=
sorry

end rearrange_infinite_decimal_l340_340055


namespace flower_pots_problem_l340_340036

noncomputable def cost_of_largest_pot (x : ℝ) : ℝ := x + 5 * 0.15

theorem flower_pots_problem
  (x : ℝ)       -- cost of the smallest pot
  (total_cost : ℝ) -- total cost of all pots
  (h_total_cost : total_cost = 8.25)
  (h_price_relation : total_cost = 6 * x + (0.15 + 2 * 0.15 + 3 * 0.15 + 4 * 0.15 + 5 * 0.15)) :
  cost_of_largest_pot x = 1.75 :=
by
  sorry

end flower_pots_problem_l340_340036


namespace find_n_times_s_l340_340390

noncomputable def f : ℝ → ℝ := λ x, 0 -- Placeholder, actual satisfying functions are f(x) = 0 and f(x) = x²

theorem find_n_times_s :
  (∃ f : ℝ → ℝ, (∀ x y : ℝ, f (f(x) + y) = f (x^2 - 3y) + 3 * f(x) * y) 
    ∧ (let n := {(f 2)}.card in
       let s := (∑ a in {(f 2)}, a) in
       n = 2 ∧ s = 4)) →
  8 = 8 :=
by
  sorry

end find_n_times_s_l340_340390


namespace rounding_addition_to_tenth_l340_340909

def number1 : Float := 96.23
def number2 : Float := 47.849

theorem rounding_addition_to_tenth (sum : Float) : 
    sum = number1 + number2 →
    Float.round (sum * 10) / 10 = 144.1 :=
by
  intro h
  rw [h]
  norm_num
  sorry -- Skipping the actual rounding proof

end rounding_addition_to_tenth_l340_340909


namespace mark_walks_longer_l340_340407

-- Define the conditions
def mark_speed : ℝ := 3  -- Mark's speed in miles per hour
def chris_speed : ℝ := 4  -- Chris's speed in miles per hour
def distance_to_school : ℝ := 9  -- Distance from house to school in miles

-- Calculate the time for Chris to walk to school
def chris_time : ℝ := distance_to_school / chris_speed  -- equals 2.25 hours

-- Calculate the total distance Mark walks
def mark_total_distance : ℝ := 3 * 2 + 2 * 2 + distance_to_school  -- equals 19 miles

-- Calculate the time for Mark to walk the total distance
def mark_time : ℝ := mark_total_distance / mark_speed  -- 19 / 3

-- Define the difference in time
def time_difference : ℝ := mark_time - chris_time  -- should be equal to 4.08

-- Statement to prove
theorem mark_walks_longer : time_difference = 4.08 :=
by 
  -- We are stating the theorem without proof
  sorry

end mark_walks_longer_l340_340407


namespace magnitude_of_earthquake_amplitude_ratio_l340_340450

-- Given conditions
variables (A A_0 : ℝ) (A_9 A_5 : ℝ)
variables (log : ℝ → ℝ) -- log function

-- Define the Richter magnitude formula
def richter_magnitude (A A_0 : ℝ) : ℝ := log A - log A_0

-- Problem 1: Calculate the magnitude of the earthquake
theorem magnitude_of_earthquake : richter_magnitude 1000 0.001 = 6 :=
by 
  sorry

-- Problem 2: Ratio of amplitudes between a 9 magnitude and a 5 magnitude earthquake
theorem amplitude_ratio : 9 - 5 = log A_9 - log A_5 → (A_9 / A_5) = 10^4 :=
by
  intro h1,
  have h2 : 9 - 5 = 4 := by norm_num,
  rw h2 at h1,
  have h3 : log (A_9 / A_5) = 4,
  {
    rw log_div,
    exact h1,
  },
  sorry

end magnitude_of_earthquake_amplitude_ratio_l340_340450


namespace integer_solutions_exist_l340_340579

theorem integer_solutions_exist : ∀ (n : ℤ), ∃ (x y z : ℤ), 
  x = n^2 + n + 1 ∧ y = n^2 - n + 1 ∧ z = 1 ∧ 
  x^2 + y^2 + z^2 = 2 * y * z + 2 * z * x + 2 * x * y - 3 :=
by {
  intro n,
  use [n^2 + n + 1, n^2 - n + 1, 1],
  split, reflexivity,
  split, reflexivity,
  split, reflexivity,
  sorry
}

end integer_solutions_exist_l340_340579


namespace pharmacist_weights_exist_l340_340168

theorem pharmacist_weights_exist :
  ∃ (a b c : ℝ), a + b = 100 ∧ a + c = 101 ∧ b + c = 102 ∧ a < 90 ∧ b < 90 ∧ c < 90 :=
by
  sorry

end pharmacist_weights_exist_l340_340168


namespace range_of_w_l340_340630

-- Definition of the function f
def f (x : ℝ) : ℝ := 4 * sin x * (sin (π / 4 + x / 2))^2 + cos (2 * x)

-- Definition of the rewritten function with parameter w
def g (w x : ℝ) : ℝ := 2 * sin (w * x) + 1

-- Hypothesized monotonicity condition of g
def isMonotonicOn (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, x ≤ y → f x ≤ f y

-- The main statement we need to prove
theorem range_of_w (w : ℝ) (h : w > 0) :
  isMonotonicOn (g w) (Icc (-π / 2) (2 * π / 3)) → w ∈ Ioo 0 (3 / 4) :=
sorry

end range_of_w_l340_340630


namespace monotonic_increasing_interval_l340_340458

noncomputable def f (x : ℝ) : ℝ := Real.log_base (1/3) (x^2 - 4)

def h (x : ℝ) : ℝ := x^2 - 4

theorem monotonic_increasing_interval :
  (∀ x y : ℝ, x < y → f x < f y) ↔ Iio (-2) = {x : ℝ | x < -2} :=
sorry

end monotonic_increasing_interval_l340_340458


namespace hyperbola_asymptote_a_value_l340_340318

theorem hyperbola_asymptote_a_value :
  ∃ (a : ℝ), (∀ x y : ℝ, (x^2 / a^2 - y^2 = 1) → (√3 * x + y = 0)) → a = Real.sqrt 3 / 3 :=
by
  sorry

end hyperbola_asymptote_a_value_l340_340318


namespace distinct_positive_integers_l340_340016

theorem distinct_positive_integers (a b : ℕ) (h_distinct : a ≠ b) (h_pos : a > 0 ∧ b > 0) 
  (h_div : a^2 + a * b + b^2 ∣ a * b * (a + b)) : |a - b| > (ab)^(1/3) := by
  sorry

end distinct_positive_integers_l340_340016


namespace angles_equal_l340_340620

theorem angles_equal (A B C : ℝ) (h1 : A + B + C = Real.pi) (h2 : Real.sin A = 2 * Real.cos B * Real.sin C) : B = C :=
by sorry

end angles_equal_l340_340620


namespace jar_capacity_l340_340699

theorem jar_capacity 
  (hives : ℕ)
  (hive_production : ℕ)
  (jars_needed : ℕ)
  (friend_jar_fraction : ℚ)
  (total_honey := hives * hive_production)
  (jars_for_half_honey := total_honey * (1 - friend_jar_fraction))
  (capacity_per_jar := jars_for_half_honey / jars_needed) :
  hives = 5 → 
  hive_production = 20 → 
  jars_needed = 100 → 
  friend_jar_fraction = 0.5 →
  capacity_per_jar = 0.5 := 
by
  intros,
  sorry

end jar_capacity_l340_340699


namespace min_value_AP_PB1_l340_340357

-- Define the existence of points and edge lengths of the rectangular prism
variable (A B C D A1 B1 C1 D1 P : Type) [Point (A B C D A1 B1 C1 D1 P : Type)]
variable (edges : AB = 6 ∧ BC = BB1 = sqrt(2))

noncomputable def minimum_distance : ℝ :=
  5 * sqrt(2)

theorem min_value_AP_PB1 (P : Point) (hP : P ∈ line_segment B C1) :
  AP + PB1 = min (AP + PB1) :=
by
  sorry

end min_value_AP_PB1_l340_340357


namespace MagicSquareDifference_l340_340734

noncomputable def magic_square_diff (A B C D E : ℕ) (Grid : Fin 3 → Fin 3 → ℕ) (S : ℕ) : Prop :=
  (Grid 0 0 = A) ∧ (Grid 0 1 = B) ∧ (Grid 0 2 = C) ∧
  (∀ i, ∑ j, Grid i j = S) ∧
  (∀ j, ∑ i, Grid i j = S) ∧
  (∑ i, Grid i i = S) ∧
  (∑ i, Grid i (2 - i) = S) ∧
  (A - B = 14) ∧
  (B - C = 14) ∧
  (D - E = 14)

theorem MagicSquareDifference :
  ∀ (A B C D E : ℕ) (Grid : Fin 3 → Fin 3 → ℕ) (S : ℕ),
  magic_square_diff A B C D E Grid S → (D - E = 14) := 
by
  intro A B C D E Grid S h
  sorry

end MagicSquareDifference_l340_340734


namespace bucket_full_weight_l340_340494

variable (p q : ℝ) -- Defining p and q as real numbers

-- Conditions from the problem
def three_fourths_full (x y : ℝ) : Prop := x + (3/4) * y = p
def one_third_full (x y : ℝ) : Prop := x + (1/3) * y = q

-- Theorem to prove the total weight when the bucket is full
theorem bucket_full_weight (x y : ℝ) (h1 : three_fourths_full p q x y) (h2 : one_third_full p q x y) : 
  x + y = (8 * p - 3 * q) / 5 :=
sorry

end bucket_full_weight_l340_340494


namespace log_108_eq_2a_add_3b_l340_340593

theorem log_108_eq_2a_add_3b (a b : ℝ) (h1 : log 10 2 = a) (h2 : 10^b = 3) : log 10 108 = 2 * a + 3 * b :=
by 
  sorry

end log_108_eq_2a_add_3b_l340_340593


namespace factor_count_x9_minus_x_l340_340493

theorem factor_count_x9_minus_x :
  ∃ (factors : List (Polynomial ℤ)), x^9 - x = factors.prod ∧ factors.length = 5 :=
sorry

end factor_count_x9_minus_x_l340_340493


namespace product_of_sisters_and_brothers_l340_340034

-- Lucy's family structure
def lucy_sisters : ℕ := 4
def lucy_brothers : ℕ := 6

-- Liam's siblings count
def liam_sisters : ℕ := lucy_sisters + 1  -- Including Lucy herself
def liam_brothers : ℕ := lucy_brothers    -- Excluding himself

-- Prove the product of Liam's sisters and brothers is 25
theorem product_of_sisters_and_brothers : liam_sisters * (liam_brothers - 1) = 25 :=
by
  sorry

end product_of_sisters_and_brothers_l340_340034


namespace smallest_number_of_eggs_l340_340855

-- Definitions based on conditions
def total_containers (c : ℕ) := 9
def eggs_per_container := 18
def total_eggs_if_all_full (c : ℕ) := eggs_per_container * c
def eggs_lost_three_containers := 3 * (18 - 16)
def eggs_lost_one_container := 18 - 17
def total_eggs_lost := eggs_lost_three_containers + eggs_lost_one_container
def total_eggs (c : ℕ) := total_eggs_if_all_full c - total_eggs_lost

-- Statement of the problem as a Lean 4 theorem
theorem smallest_number_of_eggs :
  total_eggs 9 = 155 :=
by
  rw [total_eggs, total_eggs_if_all_full, total_eggs_lost, eggs_lost_three_containers, eggs_lost_one_container],
  norm_num,
  sorry

end smallest_number_of_eggs_l340_340855


namespace limit_a_n_l340_340726

noncomputable def a_n (n : ℕ) : ℝ := 
  let rec nested_rad (k : ℕ) : ℝ :=
  if k = 0 then 1 else real.sqrt (1 + k * nested_rad (k - 1))
  in nested_rad n

theorem limit_a_n : 
  tendsto (λ n, a_n n) at_top (𝓝 3) := 
sorry

end limit_a_n_l340_340726


namespace bicycle_has_four_wheels_l340_340201

variables (Car : Type) (Bicycle : Car) (FourWheeled : Car → Prop)
axiom car_four_wheels : ∀ (c : Car), FourWheeled c

theorem bicycle_has_four_wheels : FourWheeled Bicycle :=
by {
  apply car_four_wheels
}

end bicycle_has_four_wheels_l340_340201


namespace difference_of_sums_l340_340114

-- Define the sum of the first 1001 odd numbers squared
def sum_of_odd_squares (n : ℕ) : ℕ :=
  n * (2 * n - 1) * (2 * n + 1) / 3

-- Define the sum of the first 1001 even numbers cubed
def sum_of_even_cubes (n : ℕ) : ℕ :=
  n^2 * (n + 1)^2

-- The main theorem stating the problem translated into Lean
theorem difference_of_sums :
  sum_of_even_cubes 1001 - sum_of_odd_squares 1001 = -799700002 :=
by
  -- Placeholder for the proof
  sorry

end difference_of_sums_l340_340114


namespace find_k_for_circle_radius_l340_340590

theorem find_k_for_circle_radius (k : ℝ) :
  (∃ x y : ℝ, x^2 + 14*x + y^2 + 8*y - k = 0 ∧ (x + 7)^2 + (y + 4)^2 = 10^2) ↔ k = 35 :=
by
  sorry

end find_k_for_circle_radius_l340_340590


namespace polynomial_necessary_but_not_sufficient_l340_340866

-- Definitions
def polynomial_condition (x : ℝ) : Prop := x^2 - 3 * x + 2 = 0

def specific_value : ℝ := 1

-- Theorem statement
theorem polynomial_necessary_but_not_sufficient :
  (polynomial_condition specific_value ∧ ¬ ∀ x, polynomial_condition x -> x = specific_value) :=
by
  sorry

end polynomial_necessary_but_not_sufficient_l340_340866


namespace red_ball_higher_probability_l340_340474

theorem red_ball_higher_probability : 
  let prob := λ k, 2^{-k}
  let bins := Set.Ici 1
  (∑ k in bins, ∏ ball in [red, green, blue], prob k ^ 3) = ∑ r in bins, ∑ g in bins, ∑ b in bins, if r > g ∧ r > b then ∏ ball in [r, g, b], prob ball else 0 = 2/7 :=
by
  sorry

end red_ball_higher_probability_l340_340474


namespace intersection_points_count_l340_340067

-- Define the function and conditions
variables {R : Type*} [linear_ordered_field R] (f : R → R)
variable (hf : function.injective f)

-- Define the theorem for the number of intersection points
theorem intersection_points_count : 
  ∃ n : ℕ, n = 2 ∧ ∀ x : R, f (x ^ 3) = f (x ^ 6) → x ∈ {0, 1} :=
by
  use 2
  intros
  sorry

end intersection_points_count_l340_340067


namespace interval_approx_14_l340_340805

def start_time : ℕ := 7078
def end_time : ℕ := 12047
def number_of_glows : ℝ := 354.92857142857144
def interval_between_glows : ℝ := (end_time - start_time : ℝ) / number_of_glows

theorem interval_approx_14 : abs (interval_between_glows - 14) < 1 := by
  sorry

end interval_approx_14_l340_340805


namespace tangent_line_through_P_l340_340180

variable (x y : ℝ)
def circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 5
def point_on_circle (P : ℝ × ℝ) : Prop := circle P.1 P.2
def is_tangent_line (l : ℝ → ℝ) : Prop := ∀ x y, point_on_circle (x, y) → (∃ k : ℝ, l x = y + k * (x - 2))

theorem tangent_line_through_P (x y : ℝ) (P : ℝ × ℝ) (H : point_on_circle (2, 4)) :
  x + 2 * y - 10 = 0 :=
by
  sorry

end tangent_line_through_P_l340_340180


namespace solve_for_a_l340_340656

open Complex

theorem solve_for_a (a : ℝ) (h : (2 + a * I) * (a - 2 * I) = -4 * I) : a = 0 :=
sorry

end solve_for_a_l340_340656


namespace factor_expression_l340_340965

variable (b : ℝ)

theorem factor_expression : 221 * b * b + 17 * b = 17 * b * (13 * b + 1) :=
by
  sorry

end factor_expression_l340_340965


namespace part_1_part_2_part_3_l340_340286

-- Define sequences and their properties

def T (n : Nat) : ℝ := -a n + 1 / 2

def a (n : Nat) : ℝ := (1 / 2)^(n + 1)

def b (n : Nat) : ℝ := 3 * n + 1

def c (n : Nat) : ℝ := a n * b n

-- Statement 1: General formula for b_n
theorem part_1 (n : Nat) (hn : 0 < n) : 
  b n = 3 * n + 1 := 
sorry

-- Statement 2: Sum of the first n terms of the sequence {c_n}
def S (n : Nat) : ℝ := ∑ i in Finset.range n, c (i + 1)

theorem part_2 (n : Nat) (hn : 0 < n) : 
  S n = 7 / 2 - (3 * n + 7) * (1 / 2)^(n + 1) := 
sorry

-- Statement 3: Given condition on c_n and range of m
theorem part_3 (n : Nat) (m : ℝ) (hn : 0 < n) : 
  (∀ (k : Nat), 0 < k → c k ≤ 1 / 4 * m^2 + m + 1) → m ≥ 0 ∨ m ≤ -4 := 
sorry

end part_1_part_2_part_3_l340_340286


namespace chord_square_sum_l340_340712

noncomputable def circle_radius : ℝ := 7
noncomputable def diameter_length : ℝ := 2 * circle_radius

-- Defining points: center O, points A, B, C, D, and intersection point E
variables (A B E C D : Type)
variables [metric_space A] [metric_space B] [metric_space E] [metric_space C] [metric_space D]

noncomputable def OA_length : ℝ := circle_radius
noncomputable def OB_length : ℝ := circle_radius

-- Intersecting at point E, with length constraints
noncomputable def BE_length : ℝ := 3
noncomputable def AE_length : ℝ := diameter_length - BE_length

-- Angle condition
axiom angle_AEC : ℝ := 45 * real.pi / 180  -- converting degrees to radians for Lean

-- Proving that CE^2 + DE^2 = 98
theorem chord_square_sum
  (h_diam : dist A B = diameter_length)
  (h_radius_A : dist (0 : ℝ) A = OA_length)
  (h_radius_B : dist (0 : ℝ) B = OB_length)
  (h_radius_C : dist (0 : ℝ) C = circle_radius)
  (h_radius_D : dist (0 : ℝ) D = circle_radius)
  (h_BE : dist B E = BE_length)
  (h_AE : dist A E = AE_length)
  (h_angle_AEC : real.angle A E C = angle_AEC) :
  dist C E^2 + dist D E^2 = 98 :=
sorry  -- Proof skipped

end chord_square_sum_l340_340712


namespace sum_inverses_mod_17_l340_340961

theorem sum_inverses_mod_17 : 
  (3⁻¹ + 3⁻² + 3⁻³ + 3⁻⁴ + 3⁻⁵ + 3⁻⁶) % 17 = 3 % 17 := 
by 
  sorry

end sum_inverses_mod_17_l340_340961


namespace insulation_cost_l340_340173

/-
  A rectangular tank needs to be coated with insulation. 
  The tank has dimensions of 3 feet (width), 7 feet (length), and 2 feet (height). 
  Each square foot of insulation costs $20.
  Prove that it will cost $1640 to cover the surface.
-/

theorem insulation_cost
  (l w h : ℕ)
  (cost_per_sqft : ℕ)
  (l_eq : l = 7)
  (w_eq : w = 3)
  (h_eq : h = 2)
  (cost_per_sqft_eq : cost_per_sqft = 20) :
  let SA := 2 * l * w + 2 * l * h + 2 * w * h in
  let C := SA * cost_per_sqft in
  C = 1640 := 
by {
  sorry
}

end insulation_cost_l340_340173


namespace John_l340_340372

-- Definitions based on the conditions in the problem
def John's initial gap (meters: ℕ) : ℕ := 15
def Steve's speed (m_per_s: ℝ) : ℝ := 3.7
def John's lead at finish (meters: ℕ) : ℕ := 2
def duration (seconds: ℕ) : ℕ := 34

-- Theorem based on the question and the correct answer
theorem John's_pace_is_correct : 
  ∃ (J : ℝ), J * (duration 34) = (Steve's speed 3.7) * (duration 34) + (John's initial gap 15 + John's lead at finish 2) := 
begin
  use 4.2,
  simp [duration, Steve's speed, John's initial gap, John's lead at finish],
  norm_num,
end

end John_l340_340372


namespace probability_of_at_least_one_two_l340_340158

theorem probability_of_at_least_one_two (x y z : ℕ) (hx : 1 ≤ x ∧ x ≤ 6) (hy : 1 ≤ y ∧ y ≤ 6) (hz : 1 ≤ z ∧ z ≤ 6) (hxy : x + y = z) :
  let outcomes := [(1, 1), (1, 2), (2, 1), (1, 3), (2, 2), (3, 1), (1, 4), (2, 3), (3, 2), (4, 1), (1, 5), (2, 4), (3, 3), (4, 2), (5, 1)] in
  let valid_outcomes := (1, 1) :: (1, 2) :: (2, 1) :: (1, 3) :: (2, 2) :: (3, 1) :: (1, 4) :: (2, 3) :: (3, 2) :: (4, 1) :: (1, 5) :: (2, 4) :: (3, 3) :: (4, 2) :: (5, 1) :: [] in
  let favorable_outcomes := list.countp (λ (p : ℕ × ℕ), p.fst = 2 ∨ p.snd = 2) valid_outcomes in
  let total_outcomes := list.length valid_outcomes in
  (favorable_outcomes.to_nat / total_outcomes.to_nat : ℚ) = 8 / 15 :=
sorry

end probability_of_at_least_one_two_l340_340158


namespace last_three_digits_7_pow_80_l340_340981

theorem last_three_digits_7_pow_80 : (7^80) % 1000 = 961 := by
  sorry

end last_three_digits_7_pow_80_l340_340981


namespace cos_theta_calculate_l340_340325

open_locale real_inner_product_space

variables {V : Type*} [inner_product_space ℝ V]

def unit_vector (u : V) := ∥u∥ = 1
def perpendicular (u v : V) := ⟪u, v⟫ = 0
def vector_c (a b : V) : V := (sqrt 5) • a - 2 • b
def cos_theta (a c : V) : ℝ := ⟪a, c⟫ / (∥a∥ * ∥c∥)

theorem cos_theta_calculate
  (a b : V) (hac : unit_vector a) (hbc : unit_vector b) (hapb : perpendicular a b) :
  cos_theta a (vector_c a b) = (sqrt 5) / 3 :=
by
  sorry

end cos_theta_calculate_l340_340325


namespace find_f_quarter_l340_340667

noncomputable def f (x : ℝ) (α : ℝ) := x^α

theorem find_f_quarter
  (α : ℝ)
  (h1 : f 3 α = real.sqrt 3) :
  f (1/4) α = 1/2 :=
by
  sorry

end find_f_quarter_l340_340667


namespace period_of_sin_x_plus_2_cos_x_l340_340846

-- Definition of the function
def f (x : ℝ) : ℝ := sin x + 2 * cos x

-- The statement to prove the period is 2π
theorem period_of_sin_x_plus_2_cos_x : ∀ x : ℝ, f (x + 2 * π) = f x := by
  -- Proof goes here
  sorry

end period_of_sin_x_plus_2_cos_x_l340_340846


namespace tetrahedron_distance_ratio_l340_340836

theorem tetrahedron_distance_ratio
  (T₁ T₂ : Tetrahedron)
  (h₁ : T₁.is_regular)
  (h₂ : T₂.is_regular)
  (h₃ : T₁.is_congruent T₂)
  (h₄ : T₁.is_joined_along_base_with T₂)
  (h₅ : T₁.dihedral_angle_is_equal (join_base T₁ T₂))
  : ratio (distance (apex T₁) (apex T₂)) (distance (centroid_face T₁) (centroid_face T₂)) = 2 / 3 :=
  sorry

end tetrahedron_distance_ratio_l340_340836


namespace C1_cartesian_equation_C2_general_equation_intersection_reciprocal_sum_l340_340351

noncomputable def polar_equation_C1 (rho theta : ℝ) : Prop := 
  sqrt 2 * rho * cos (theta - π / 4) + 1 = 0

def parametric_equation_C2 (α : ℝ) : (ℝ × ℝ) := 
  (2 * cos α, sqrt 3 * sin α)

def P : (ℝ × ℝ) := (0, -1)

theorem C1_cartesian_equation :
  ∀ (ρ θ : ℝ), polar_equation_C1 ρ θ → (ρ * cos θ + ρ * sin θ + 1 = 0) := by
  intros rho theta h
  sorry

theorem C2_general_equation :
  ∀ (α : ℝ), parametric_equation_C2 α = (2 * cos α, sqrt 3 * sin α) →
    (2 * cos α) ^ 2 / 4 + (sqrt 3 * sin α) ^ 2 / 3 = 1 := by
  intros α h
  sorry

theorem intersection_reciprocal_sum :
  let A B : (ℝ × ℝ) := (sorry, sorry) in
  let PA PB : ℝ := real.dist P A, real.dist P B in
  PA ≠ 0 ∧ PB ≠ 0 →
  PA + PB = 24 / 7 →
  PA * PB = 16 / 7 →
  (1 / PA + 1 / PB) = 3 / 2 := by
  intros A B PA PB not_zero PA_PB_sum PA_PB_product
  sorry

end C1_cartesian_equation_C2_general_equation_intersection_reciprocal_sum_l340_340351


namespace light_distance_in_500_years_l340_340799

theorem light_distance_in_500_years (d1: ℕ) (distance_one_year : d1 = 5_870_000_000_000) : 
  500 * d1 = 2935 * 10^12 := by
  rw [distance_one_year]
  sorry

end light_distance_in_500_years_l340_340799


namespace simplify_polynomial_expression_l340_340063

theorem simplify_polynomial_expression (r : ℝ) :
  (2 * r^3 + 5 * r^2 + 6 * r - 4) - (r^3 + 9 * r^2 + 4 * r - 7) = r^3 - 4 * r^2 + 2 * r + 3 :=
by
  sorry

end simplify_polynomial_expression_l340_340063


namespace arithmetic_sequence_line_l340_340334

theorem arithmetic_sequence_line (A B C x y : ℝ) :
  (2 * B = A + C) → (A * 1 + B * -2 + C = 0) :=
by
  intros h
  sorry

end arithmetic_sequence_line_l340_340334


namespace translate_function_down_l340_340686

theorem translate_function_down 
  (f : ℝ → ℝ) 
  (a k : ℝ) 
  (h : ∀ x, f x = a * x) 
  : ∀ x, (f x - k) = a * x - k :=
by
  sorry

end translate_function_down_l340_340686


namespace f_f_2_l340_340391

noncomputable def f (x : ℝ) : ℝ :=
if h : x < 2 then 2 * Real.exp (x - 1) else Real.log (2^x - 1) / Real.log 3

theorem f_f_2 : f (f 2) = 2 :=
by
  sorry

end f_f_2_l340_340391


namespace system_solution_l340_340322

theorem system_solution (x y m n : ℤ) (h1 : x + y = m) (h2 : x - y = n + 1) (hx : x = 3) (hy : y = 2) : m + n = 5 :=
by
  have m_val : m = 5 := by
    rw [hx, hy, h1]
    norm_num
  have n_val : n = 0 := by
    rw [hx, hy, h2]
    norm_num
  rw [m_val, n_val]
  norm_num
  sorry

end system_solution_l340_340322


namespace parabola_focus_constants_l340_340605

theorem parabola_focus_constants (p : ℝ) (h : p > 0) :
  let focus : ℝ × ℝ := (p / 2, 0)
      parabola (x y : ℝ) := y^2 = 2 * p * x
      
      -- Line passing through the focus and intersecting at points A and B
      intersects (x y : ℝ) (k : ℝ) := y = k * (x - p / 2)
      -- Points A and B
      A B : ℝ × ℝ := sorry -- Definitions of A and B based on the line equation and intersection can be written here

  in (A.1 * B.1 = p^2 / 4) ∧
     (1 / (abs A.1 + p / 2) + 1 / (abs B.1 + p / 2) = 2 / p) :=
by
  -- Adding the constants definitions to express the proof conditions
  sorry

end parabola_focus_constants_l340_340605


namespace inequality_proof_l340_340141

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 3) :
    (a^2 + 6) / (2*a^2 + 2*b^2 + 2*c^2 + 2*a - 1) +
    (b^2 + 6) / (2*a^2 + 2*b^2 + 2*c^2 + 2*b - 1) +
    (c^2 + 6) / (2*a^2 + 2*b^2 + 2*c^2 + 2*c - 1) ≤
    3 := by
  sorry

end inequality_proof_l340_340141


namespace count_distinct_x_values_l340_340220

/-- Define the sequence according to the problem conditions. -/
def sequence (x : ℝ) (n : ℕ) : ℝ :=
  match n with
  | 0 => x
  | 1 => 1000
  | _ => if n % 2 = 0 then
            (sequence x (n-1) + 1) / (sequence x (n-2))
         else
            sequence x (n-1)  -- This is a placeholder; actual sequence definition will iterate.

def appears_1001 (x : ℝ) : Prop :=
  ∃ n : ℕ, sequence x n = 1001

theorem count_distinct_x_values :
  {x : ℝ | appears_1001 x}.finite.toFinset.card = 4 :=
sorry

end count_distinct_x_values_l340_340220


namespace last_three_digits_7_pow_80_l340_340983

theorem last_three_digits_7_pow_80 : (7 ^ 80) % 1000 = 961 := 
by sorry

end last_three_digits_7_pow_80_l340_340983


namespace number_of_walking_methods_l340_340154

theorem number_of_walking_methods (n : ℕ) (h : n = 4) : ∃ k, k = 12 := 
by
  use 4 * 3
  rw [mul_comm]
  have h_eq : 4 * 3 = 12 := by norm_num
  exact h_eq

end number_of_walking_methods_l340_340154


namespace compare_f_values_l340_340272

noncomputable def f : ℝ → ℝ :=
  λ x, x * Real.sin x + Real.cos x

theorem compare_f_values :
  f (-3) < f 2 ∧ f 2 < f (Real.pi / 2) :=
by
  -- The proof will go here
  sorry

end compare_f_values_l340_340272


namespace log_arith_proof_l340_340230

theorem log_arith_proof :
  (10 ^ (Real.log (1 / 2)) * (1 / 10) ^ (Real.log 5)) = 1 / 10 :=
by
  sorry

end log_arith_proof_l340_340230


namespace total_broken_marbles_l340_340265

def percentage_of (percent : ℝ) (total : ℝ) : ℝ := percent * total / 100

theorem total_broken_marbles :
  let first_set_total := 50
  let second_set_total := 60
  let first_set_percent_broken := 10
  let second_set_percent_broken := 20
  let first_set_broken := percentage_of first_set_percent_broken first_set_total
  let second_set_broken := percentage_of second_set_percent_broken second_set_total
  first_set_broken + second_set_broken = 17 :=
by
  sorry

end total_broken_marbles_l340_340265


namespace moles_of_water_formed_l340_340329

-- Definitions for the conditions
def amyl_alcohol_moles : ℕ := 3
def hydrochloric_acid_moles : ℕ := 3
def amyl_alcohol_to_water_ratio : ℕ := 1 -- from balanced equation 1:1:1:1 for H2O
def hydrochloric_acid_to_water_ratio : ℕ := 1 -- from balanced equation 1:1:1:1 for H2O

-- Theorem stating the proof problem
theorem moles_of_water_formed :
  amyl_alcohol_moles = 3 →
  hydrochloric_acid_moles = 3 →
  amyl_alcohol_to_water_ratio = 1 →
  hydrochloric_acid_to_water_ratio = 1 →
  ∃ (water_moles: ℕ), water_moles = 3 :=
by
  intros h_amyl h_hcl h_amyl_ratio h_hcl_ratio
  use amyl_alcohol_moles -- since the ratio is 1:1:1:1
  exact h_amyl

end moles_of_water_formed_l340_340329


namespace average_of_ratios_4_5_6_l340_340827

theorem average_of_ratios_4_5_6 (x : ℕ) (h : 6 * x = 24) : 
  let a := 4 * x in
  let b := 5 * x in
  let c := 6 * x in
  (a + b + c) / 3 = 20 :=
by
  sorry

end average_of_ratios_4_5_6_l340_340827


namespace total_books_from_library_l340_340050

def initialBooks : ℕ := 54
def additionalBooks : ℕ := 23

theorem total_books_from_library : initialBooks + additionalBooks = 77 := by
  sorry

end total_books_from_library_l340_340050


namespace michael_total_revenue_l340_340040

def price_large : ℕ := 22
def price_medium : ℕ := 16
def price_small : ℕ := 7

def qty_large : ℕ := 2
def qty_medium : ℕ := 2
def qty_small : ℕ := 3

def total_revenue : ℕ :=
  (price_large * qty_large) +
  (price_medium * qty_medium) +
  (price_small * qty_small)

theorem michael_total_revenue : total_revenue = 97 :=
  by sorry

end michael_total_revenue_l340_340040


namespace total_clouds_count_l340_340927

-- Definitions based on the conditions
def carson_clouds := 12
def little_brother_clouds := 5 * carson_clouds
def older_sister_clouds := carson_clouds / 2

-- The theorem statement that needs to be proved
theorem total_clouds_count : carson_clouds + little_brother_clouds + older_sister_clouds = 78 := by
  -- Definitions
  have h1 : carson_clouds = 12 := rfl
  have h2 : little_brother_clouds = 5 * 12 := rfl
  have h3 : older_sister_clouds = 12 / 2 := rfl
  sorry

end total_clouds_count_l340_340927


namespace fraction_dutch_americans_has_window_l340_340045

variable (P D DA : ℕ)
variable (f_P_d d_P_w : ℚ)
variable (DA_w : ℕ)

-- Total number of people on the bus P 
-- Fraction of people who were Dutch f_P_d
-- Fraction of Dutch Americans who got window seats d_P_w
-- Number of Dutch Americans who sat at windows DA_w
-- Define the assumptions
def total_people_on_bus := P = 90
def fraction_dutch := f_P_d = 3 / 5
def fraction_dutch_americans_window := d_P_w = 1 / 3
def dutch_americans_window := DA_w = 9

-- Prove that fraction of Dutch people who were also American is 1/2
theorem fraction_dutch_americans_has_window (P D DA DA_w : ℕ) (f_P_d d_P_w : ℚ) :
  total_people_on_bus P ∧ fraction_dutch f_P_d ∧
  fraction_dutch_americans_window d_P_w ∧ dutch_americans_window DA_w →
  (DA: ℚ) / D = 1 / 2 :=
by
  sorry

end fraction_dutch_americans_has_window_l340_340045


namespace triangle_perimeter_triangle_side_c_l340_340341

theorem triangle_perimeter (A B C : ℝ) (a b c : ℝ) (h1 : b * (Real.sin (A/2))^2 + a * (Real.sin (B/2))^2 = C / 2) (h2 : c = 2) : 
  a + b + c = 6 := 
sorry

theorem triangle_side_c (A B C : ℝ) (a b c : ℝ) (h1 : b * (Real.sin (A/2))^2 + a * (Real.sin (B/2))^2 = C / 2) 
(h2 : C = Real.pi / 3) (h3 : 2 * Real.sqrt 3 = (1/2) * a * b * Real.sin (Real.pi / 3)) : 
c = 2 * Real.sqrt 2 := 
sorry

end triangle_perimeter_triangle_side_c_l340_340341


namespace find_y_l340_340127

theorem find_y (y : ℕ) (h1 : y % 6 = 5) (h2 : y % 7 = 6) (h3 : y % 8 = 7) : y = 167 := 
by
  sorry  -- Proof is omitted

end find_y_l340_340127


namespace crayon_count_per_row_l340_340972

theorem crayon_count_per_row (total_rows : Nat) (total_crayons : Nat) (crayons_per_row : Nat)
  (h1 : total_rows = 16) (h2 : total_crayons = 96) : crayons_per_row = 6 :=
by
  have eqn : total_rows * crayons_per_row = total_crayons := by sorry
  have h3 : 16 * crayons_per_row = 96 := by rw [h1, h2]; exact eqn
  have h4 : crayons_per_row = 6 := by
    have h5 : 16 * crayons_per_row = 16 * 6 := by rw h3
    sorry
  exact h4

end crayon_count_per_row_l340_340972


namespace odd_function_max_to_min_l340_340666

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

theorem odd_function_max_to_min (a b : ℝ) (f : ℝ → ℝ)
  (hodd : is_odd_function f)
  (hmax : ∃ x : ℝ, x > 0 ∧ (a * f x + b * x + 1) = 2) :
  ∃ y : ℝ, y < 0 ∧ (a * f y + b * y + 1) = 0 :=
sorry

end odd_function_max_to_min_l340_340666


namespace convert_rectangular_to_spherical_l340_340940

theorem convert_rectangular_to_spherical :
  ∀ (x y z : ℝ) (ρ θ φ : ℝ),
    (x, y, z) = (2, -2 * Real.sqrt 2, 2) →
    ρ = Real.sqrt (x^2 + y^2 + z^2) →
    z = ρ * Real.cos φ →
    x = ρ * Real.sin φ * Real.cos θ →
    y = ρ * Real.sin φ * Real.sin θ →
    0 < ρ ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ 0 ≤ φ ∧ φ ≤ Real.pi →
    (ρ, θ, φ) = (4, 2 * Real.pi - Real.arcsin (Real.sqrt 6 / 3), Real.pi / 3) :=
by
  intros x y z ρ θ φ H Hρ Hφ Hθ1 Hθ2 Hconditions
  sorry

end convert_rectangular_to_spherical_l340_340940


namespace count_ordered_pairs_l340_340651

theorem count_ordered_pairs : 
  {p : ℕ × ℕ | p.1 * p.2 = 50}.toFinset.card = 6 := by
  sorry

end count_ordered_pairs_l340_340651


namespace problem_XZ_eq_YT_l340_340377

open scoped EuclideanGeometry

-- Definitions based on the given problem
variables {A B C A1 B1 C1 K L X Y Z T : Point}
noncomputable theory

-- Assume the necessary geometrical configurations and properties
axiom midpoint_A1 : midpoint A1 B C A
axiom midpoint_B1 : midpoint B1 A C B
axiom midpoint_C1 : midpoint C1 A B C
axiom altitude_AK : altitude A K B C
axiom tangency_L : tangency_point L (incircle ABC) B C
axiom circumcircle_X : meets_second (circumcircle (triangle L K B1)) B1 C1 X
axiom circumcircle_Y : meets_second (circumcircle (triangle A1 L C1)) B1 C1 Y
axiom incircle_meets_ZT : meets (incircle ABC) B1 C1 Z T

-- The final problem to prove
theorem problem_XZ_eq_YT :
  XZ = YT :=
sorry

end problem_XZ_eq_YT_l340_340377


namespace simplify_expression_l340_340923

theorem simplify_expression : 
  1.5 * (Real.sqrt 1 + Real.sqrt (1+3) + Real.sqrt (1+3+5) + Real.sqrt (1+3+5+7) + Real.sqrt (1+3+5+7+9)) = 22.5 :=
by
  sorry

end simplify_expression_l340_340923


namespace last_triangle_perimeter_l340_340385

noncomputable def T1_sides := (1201, 1203, 1205)

-- Helper function to compute the sides of the next triangle based on current sides
def next_triangle_sides (a b c : ℕ) : (ℕ × ℕ × ℕ) :=
let x := (a + b - c)/2 in
let y := (b + c - a)/2 in
let z := (c + a - b)/2 in
(x, y, z)

noncomputable def T9_sides : (ℕ × ℕ × ℕ) :=
(next_triangle_sides^(8)) T1_sides.1 T1_sides.2 T1_sides.3 -- Apply next_triangle_sides 8 times

theorem last_triangle_perimeter :
  let (x, y, z) := T9_sides in
  (x + y + z : ℚ) = 903 / 128 :=
by
  let (x, y, z) := T9_sides
  have h : x + y + z = 302 + 301 + 300 / 2^7
  sorry

end last_triangle_perimeter_l340_340385


namespace number_of_equilateral_triangles_l340_340455

theorem number_of_equilateral_triangles : 
  let k_range := {-6, -5, ..., 5, 6} 
  let lines_yk := ∀ k ∈ k_range, ∀ x: ℝ, y = k
  let lines_x3k := ∀ k ∈ k_range, ∀ x: ℝ, y = x + 3 * k
  let lines_negx3k := ∀ k ∈ k_range, ∀ x: ℝ, y = -x + 3 * k
  let equilateral_triangles := count_equilateral_triangles(lines_yk, lines_x3k, lines_negx3k, triangle_side_length=1)
  equilateral_triangles = 444
:= sorry

end number_of_equilateral_triangles_l340_340455


namespace southern_northern_dynasties_congruence_l340_340571

theorem southern_northern_dynasties_congruence :
  let a := ∑ k in Finset.range 21, Nat.choose 20 k * 2^k
  ∃ b : ℤ, (a ≡ 2011 [MOD 10]) → b = 2011 :=
by { 
  sorry 
}

end southern_northern_dynasties_congruence_l340_340571


namespace probability_sum_odd_correct_l340_340090

noncomputable def probability_sum_odd : ℚ :=
  (6 * (8.factorial * 8.factorial : ℚ)) / 16.factorial

theorem probability_sum_odd_correct :
  probability_sum_odd = 1 / 2150 := by
  sorry

end probability_sum_odd_correct_l340_340090


namespace calculate_gas_volumes_l340_340506

variable (gas_volume_western : ℝ) 
variable (total_gas_volume_non_western : ℝ)
variable (population_non_western : ℝ)
variable (total_gas_percentage_russia : ℝ)
variable (gas_volume_russia : ℝ)
variable (population_russia : ℝ)

theorem calculate_gas_volumes 
(h_western : gas_volume_western = 21428)
(h_non_western : total_gas_volume_non_western = 185255)
(h_population_non_western : population_non_western = 6.9)
(h_percentage_russia : total_gas_percentage_russia = 68.0)
(h_gas_volume_russia : gas_volume_russia = 30266.9)
(h_population_russia : population_russia = 0.147)
: 
  (total_gas_volume_non_western / population_non_western = 26848.55) ∧ 
  (gas_volume_russia / population_russia ≈ 302790.13) := 
  sorry

end calculate_gas_volumes_l340_340506


namespace find_range_of_k_l340_340806

def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + (y - 2)^2 = 4

def line_eq (k x y : ℝ) : Prop := y = k * x + 2

def chord_length_condition (MN : ℝ) : Prop := MN ≥ 2 * real.sqrt 3

theorem find_range_of_k (k : ℝ) :
  (∀ x y : ℝ, circle_eq x y → line_eq k x y → chord_length_condition (2 * real.sqrt (4 - (3 * k / real.sqrt (1 + k^2))^2))) →
  (-real.sqrt 2 / 4 ≤ k ∧ k ≤ real.sqrt 2 / 4) :=
by
  sorry

end find_range_of_k_l340_340806


namespace find_abc_l340_340239

theorem find_abc
  (a b c : ℝ)
  (h : ∀ x y z : ℝ, |a * x + b * y + c * z| + |b * x + c * y + a * z| + |c * x + a * y + b * z| = |x| + |y| + |z|):
  (a = 1 ∧ b = 0 ∧ c = 0) ∨ (a = -1 ∧ b = 0 ∧ c = 0) ∨ 
  (a = 0 ∧ b = 1 ∧ c = 0) ∨ (a = 0 ∧ b = -1 ∧ c = 0) ∨ 
  (a = 0 ∧ b = 0 ∧ c = 1) ∨ (a = 0 ∧ b = 0 ∧ c = -1) :=
sorry

end find_abc_l340_340239


namespace equilateral_triangle_side_length_l340_340763

theorem equilateral_triangle_side_length 
  {P Q R S : Point} 
  {A B C : Triangle}
  (h₁ : is_inside P A B C)
  (h₂ : is_perpendicular P Q A B)
  (h₃ : is_perpendicular P R B C)
  (h₄ : is_perpendicular P S C A)
  (h₅ : distance P Q = 2)
  (h₆ : distance P R = 3)
  (h₇ : distance P S = 4)
  : side_length A B C = 6 * sqrt 3 :=
by 
  sorry

end equilateral_triangle_side_length_l340_340763


namespace min_tiles_tiling_l340_340937

noncomputable def minimum_tiles_required {m n : ℕ} (h_m : 4 ≤ m) (h_n : 4 ≤ n) : ℕ :=
  m * n

theorem min_tiles_tiling (m n : ℕ) (h_m : 4 ≤ m) (h_n : 4 ≤ n) :
  ∀ (tiles : ℕ), 
    (tiles → ∀ (r : ℕ), (r × r) <= tiles → (2*m-1)*(2*n-1) <= 3*r + 4*(tiles-r)) →
    tiles = minimum_tiles_required h_m h_n :=
sorry

end min_tiles_tiling_l340_340937


namespace solve_inequality_l340_340252

theorem solve_inequality (x : ℝ) :
  (abs ((6 - x) / 4) < 3) ∧ (2 ≤ x) ↔ (2 ≤ x) ∧ (x < 18) := 
by
  sorry

end solve_inequality_l340_340252


namespace trigonometric_comparison_l340_340929

theorem trigonometric_comparison :
  cos (- 2 * Real.pi / 5) < sin (3 * Real.pi / 5) ∧ sin (3 * Real.pi / 5) < tan (7 * Real.pi / 5) :=
by
  sorry

end trigonometric_comparison_l340_340929


namespace find_a_l340_340975

theorem find_a (a : ℝ) :
  let Δ1 := 4 - 4 * a, Δ2 := a^2 - 8
  in Δ1 > 0 ∧ Δ2 > 0 ∧ 4 - 2 * a = a^2 - 4 ↔ a = -4 := 
by {
  intros,
  sorry
}

end find_a_l340_340975


namespace find_b_l340_340598

theorem find_b (b : ℝ) :
  let C := {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 2)^2 = 2},
      line := {p : ℝ × ℝ | p.2 = 3 * p.1 + b},
      center : ℝ × ℝ := (1, 2),
      distance := |1 + b| / sqrt 10 in
  distance = 1 → b = -1 + sqrt 10 ∨ b = -1 - sqrt 10 :=
begin
  sorry
end

end find_b_l340_340598


namespace union_of_sets_l340_340869

def A := { x : ℝ | -1 ≤ x ∧ x < 3 }
def B := { x : ℝ | 2 < x ∧ x ≤ 5 }

theorem union_of_sets : A ∪ B = { x : ℝ | -1 ≤ x ∧ x ≤ 5 } := 
by sorry

end union_of_sets_l340_340869


namespace all_cells_equal_l340_340219

-- Define the infinite grid
def Grid := ℕ → ℕ → ℕ

-- Define the condition on the grid values
def is_min_mean_grid (g : Grid) : Prop :=
  ∀ i j : ℕ, g i j ≥ (g (i-1) j + g (i+1) j + g i (j-1) + g i (j+1)) / 4

-- Main theorem
theorem all_cells_equal (g : Grid) (h : is_min_mean_grid g) : ∃ a : ℕ, ∀ i j : ℕ, g i j = a := 
sorry

end all_cells_equal_l340_340219


namespace packed_oranges_l340_340520

theorem packed_oranges (oranges_per_box : ℕ) (boxes_used : ℕ) (total_oranges : ℕ) 
  (h1 : oranges_per_box = 10) (h2 : boxes_used = 265) : 
  total_oranges = 2650 :=
by 
  sorry

end packed_oranges_l340_340520


namespace movie_ticket_cost_l340_340480

theorem movie_ticket_cost (amount_brought : ℕ) (change_received : ℕ) (num_people : ℕ) :
  amount_brought = 25 → change_received = 9 → num_people = 2 → (amount_brought - change_received) / num_people = 8 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end movie_ticket_cost_l340_340480


namespace greatest_possible_value_of_median_l340_340793

-- Given conditions as definitions
variables (k m r s t : ℕ)

-- condition 1: The average (arithmetic mean) of the 5 integers is 10
def avg_is_10 : Prop := k + m + r + s + t = 50

-- condition 2: The integers are in a strictly increasing order
def increasing_order : Prop := k < m ∧ m < r ∧ r < s ∧ s < t

-- condition 3: t is 20
def t_is_20 : Prop := t = 20

-- The main statement to prove
theorem greatest_possible_value_of_median : 
  avg_is_10 k m r s t → 
  increasing_order k m r s t → 
  t_is_20 t → 
  r = 13 :=
by
  intros
  sorry

end greatest_possible_value_of_median_l340_340793


namespace broken_perfect_spiral_shells_difference_l340_340646

theorem broken_perfect_spiral_shells_difference :
  let perfect_shells := 17
  let broken_shells := 52
  let broken_spiral_shells := broken_shells / 2
  let not_spiral_perfect_shells := 12
  let spiral_perfect_shells := perfect_shells - not_spiral_perfect_shells
  broken_spiral_shells - spiral_perfect_shells = 21 := by
  sorry

end broken_perfect_spiral_shells_difference_l340_340646


namespace equilateral_triangle_side_length_l340_340766

theorem equilateral_triangle_side_length (P Q R S A B C : Type)
  [Point P] [Point Q] [Point R] [Point S] [Point A] [Point B] [Point C] :
  (within_triangle P A B C) →
  orthogonal_projection P Q A B →
  orthogonal_projection P R B C →
  orthogonal_projection P S C A →
  distance P Q = 2 →
  distance P R = 3 →
  distance P S = 4 →
  distance A B = 6 * √3 :=
by
  sorry

end equilateral_triangle_side_length_l340_340766


namespace inequality_for_positive_real_l340_340061

theorem inequality_for_positive_real (x : ℝ) (h : 0 < x) : x + 1/x ≥ 2 :=
by
  sorry

end inequality_for_positive_real_l340_340061


namespace inequality_proof_l340_340025

theorem inequality_proof (n : ℕ) (h_n : n > 4) (x : Fin n → ℝ) 
  (h_x : ∀ i, 0 ≤ x i ∧ x i ≤ 1) :
  2 * (∑ i in Finset.range n, (x i) ^ 3) - (∑ i in Finset.range n, (x i) ^ 2 * x (Fin.modN n i (i + 1))) ≤ n :=
by
  sorry

end inequality_proof_l340_340025


namespace last_three_digits_7_pow_80_l340_340982

theorem last_three_digits_7_pow_80 : (7 ^ 80) % 1000 = 961 := 
by sorry

end last_three_digits_7_pow_80_l340_340982


namespace find_large_number_l340_340858

theorem find_large_number (L S : ℕ) (h1 : L - S = 1515) (h2 : L = 16 * S + 15) : L = 1615 :=
sorry

end find_large_number_l340_340858


namespace num_last_digits_alex_likes_l340_340042

def divisible_by_3 (n : ℕ) : Prop := ∃ k : ℕ, n = 3 * k
def ends_in_6 (n : ℕ) : Prop := n % 10 = 6

theorem num_last_digits_alex_likes : 
  {d : ℕ | ∃ n : ℕ, (divisible_by_3 n ∨ ends_in_6 n) ∧ n % 10 = d}.finite.to_finset.card = 4 :=
by
  sorry

end num_last_digits_alex_likes_l340_340042


namespace blocks_differ_in_exactly_three_ways_l340_340155

noncomputable def num_blocks_differing_in_three_ways : ℕ :=
  let materials := 3
  let sizes := 3
  let colors := 4
  let shapes := 5
  let total_blocks := 120
  let target_block := ("metal", "medium", "blue", "rectangle")

  have different_material_size_color : nat := 2 * 2 * 3 * 1
  have different_material_size_shape : nat := 2 * 2 * 1 * 4
  have different_material_color_shape : nat := 2 * 1 * 3 * 4
  have different_size_color_shape : nat := 1 * 2 * 3 * 4

  different_material_size_color + different_material_size_shape + different_material_color_shape + different_size_color_shape

theorem blocks_differ_in_exactly_three_ways : num_blocks_differing_in_three_ways = 76 :=
  by
    -- Placeholder for the proof
    sorry

end blocks_differ_in_exactly_three_ways_l340_340155


namespace imaginary_part_of_z_l340_340297

noncomputable def i : ℂ := Complex.I
noncomputable def z : ℂ := i / (i - 1)

theorem imaginary_part_of_z : z.im = -1 / 2 := by
  sorry

end imaginary_part_of_z_l340_340297


namespace first_part_second_part_l340_340210

-- Define the expressions involved in the first question
def expr1 := (complex.cbrt 8) - ((Real.sqrt 12) * (Real.sqrt 6) / (Real.sqrt 3))
def result1 := 2 - 2 * (Real.sqrt 6)
theorem first_part : expr1 = result1 := sorry

-- Define the expressions involved in the second question
def expr2 := (Real.sqrt 3 + 1) ^ 2 - ((2 * Real.sqrt 2 + 3) * (2 * Real.sqrt 2 - 3))
def result2 := 5 + 2 * Real.sqrt 3
theorem second_part : expr2 = result2 := sorry

end first_part_second_part_l340_340210


namespace students_on_right_side_l340_340743

-- Define the total number of students and the number of students on the left side
def total_students : ℕ := 63
def left_students : ℕ := 36

-- Define the number of students on the right side using subtraction
def right_students (total_students left_students : ℕ) : ℕ := total_students - left_students

-- Theorem: Prove that the number of students on the right side is 27
theorem students_on_right_side : right_students total_students left_students = 27 := by
  sorry

end students_on_right_side_l340_340743


namespace base_2_base_3_product_is_144_l340_340962

def convert_base_2_to_10 (n : ℕ) : ℕ :=
  match n with
  | 1001 => 9
  | _ => 0 -- For simplicity, only handle 1001_2

def convert_base_3_to_10 (n : ℕ) : ℕ :=
  match n with
  | 121 => 16
  | _ => 0 -- For simplicity, only handle 121_3

theorem base_2_base_3_product_is_144 :
  convert_base_2_to_10 1001 * convert_base_3_to_10 121 = 144 :=
by
  sorry

end base_2_base_3_product_is_144_l340_340962


namespace period_of_sin_x_plus_2_cos_x_l340_340845

-- Definition of the function
def f (x : ℝ) : ℝ := sin x + 2 * cos x

-- The statement to prove the period is 2π
theorem period_of_sin_x_plus_2_cos_x : ∀ x : ℝ, f (x + 2 * π) = f x := by
  -- Proof goes here
  sorry

end period_of_sin_x_plus_2_cos_x_l340_340845


namespace imaginary_part_eq_neg_3_l340_340922

-- Define the imaginary unit
def i : ℂ := Complex.I

-- Define the complex number z
def z : ℂ := (3 + i) / i

-- Theorem stating the imaginary part of z is -3
theorem imaginary_part_eq_neg_3 : z.im = -3 :=
by 
  sorry

end imaginary_part_eq_neg_3_l340_340922


namespace trajectory_centroid_proof_existence_l340_340422

def trajectory_conditions (P G F1 F2 : ℝ × ℝ) : Prop :=
  let (m, n) := P in
  let (x, y) := G in
  let (f1_x, f1_y) := F1 in
  let (f2_x, f2_y) := F2 in
  (f1_x = 0) ∧ (f1_y = 1) ∧ (f2_x = 0) ∧ (f2_y = -1) ∧
  (x = m / 3) ∧ (y = (f1_y + f2_y + n) / 3) ∧
  (m = 3 * x) ∧ (n = 3 * y) ∧
  (m^2 / 3 + n^2 / 4 = 1)

def trajectory_equation (G : ℝ × ℝ) : Prop := 
  let (x, y) := G in
  (3 * x^2 + 9 * y^2 / 4 = 1) ∧ (x ≠ 0)

theorem trajectory_centroid_proof_existence (P G F1 F2 : ℝ × ℝ) :
  trajectory_conditions P G F1 F2 → trajectory_equation G := 
by {
  intros,
  sorry
}

end trajectory_centroid_proof_existence_l340_340422


namespace probability_parabola_vertex_third_quadrant_integer_solution_l340_340677

theorem probability_parabola_vertex_third_quadrant_integer_solution :
  let balls := [-5, -4, -3, -2, 2, 1] in
  let conditions (a : ℤ) : Prop :=
    (a ∈ balls) ∧
    ((-1, a + 1).1 < 0 ∧ (a + 1) < 0) ∧
    (∃ x : ℤ, ∀ (a : ℤ), a ∈ balls → (-6 : ℤ) % (a + 1) = 0) in
  let valid_balls := list.filter (λ a, conditions a) balls in
  (list.length valid_balls : ℚ) / (list.length balls : ℚ) = 1 / 3 := sorry

end probability_parabola_vertex_third_quadrant_integer_solution_l340_340677


namespace hexagon_perimeter_l340_340487

-- Define the points in the 2D plane and side lengths
universe u
variable {A B C D E F : Type u}
noncomputable theory

-- Given points A B C D E F forming 90 degree angles with each other, 
-- and their respective distances,
-- prove the perimeter of hexagon ABCDEF is 6
theorem hexagon_perimeter
  (A B C D E F : Type)
  (distAB distBC distCD distDE distEF : ℝ)
  (hAB : distAB = 1)
  (hBC : distBC = 1)
  (hCD : distCD = 2)
  (hDE : distDE = 1)
  (hEF : distEF = 1)
  (H90 : ∀ {X Y Z : Type u}, X = Y ∧ Y = Z → ∠ X Y Z = 90): 
  distAB + distBC + distCD + distDE + distEF = 6 :=
by
  sorry

end hexagon_perimeter_l340_340487


namespace spherical_to_cartesian_change_theta_l340_340894

theorem spherical_to_cartesian_change_theta (r θ φ : ℝ) :
  let x := r * sin φ * cos θ
  let y := r * sin φ * sin θ
  let z := r * cos φ
  let x' := r * sin φ * cos (-θ)
  let y' := r * sin φ * sin (-θ)
  let z' := r * cos φ
  (x, y, z) = (-3, 5, -2) → (x', y', z') = (-3, -5, -2) :=
by
  sorry

end spherical_to_cartesian_change_theta_l340_340894


namespace midpoints_and_center_are_collinear_l340_340928

-- Define the setup of the problem
variables {A B C D O K L : Type}
variables [Point A] [Point B] [Point C] [Point D] [Point O] [Point K] [Point L] 
variables {ω ω1 ω2 : Type}
variables [Circle ω] [Circle ω1] [Circle ω2]

-- Conditions of the problem
variables (inscribed_in_quadrilateral : InscribedInQuadrilateral ω A B C D)
variables (AB_not_parallel_CD : ¬ Parallel (Line A B) (Line C D))
variables (AB_intersects_CD_at_O : ∃ (O : Point), Intersection (Line A B) (Line C D) = O)
variables (circle_ω1_touches_BC_at_K : Touches ω1 (Side B C) K)
variables (circle_ω1_touches_AB_CD_outside : OutsideTouches ω1 (Line A B) ∧ OutsideTouches ω1 (Line C D))
variables (circle_ω2_touches_AD_at_L : Touches ω2 (Side A D) L)
variables (circle_ω2_touches_AB_CD_outside : OutsideTouches ω2 (Line A B) ∧ OutsideTouches ω2 (Line C D))
variables (O_K_L_collinear : Collinear O K L)

-- Midpoints notation
variables (M1 M2 I : Type)
variables [Midpoint_of_Side M1 (Side B C)] [Midpoint_of_Side M2 (Side A D)]
variables [Center_of_Circle I ω]

-- The final theorem statement
theorem midpoints_and_center_are_collinear 
  (inscribed_in_quadrilateral : InscribedInQuadrilateral ω A B C D)
  (AB_not_parallel_CD : ¬ Parallel (Line A B) (Line C D))
  (AB_intersects_CD_at_O : ∃ O, Intersection (Line A B) (Line C D) = O)
  (circle_ω1_touches_BC_at_K : Touches ω1 (Side B C) K)
  (circle_ω1_touches_AB_CD_outside : OutsideTouches ω1 (Line A B) ∧ OutsideTouches ω1 (Line C D))
  (circle_ω2_touches_AD_at_L : Touches ω2 (Side A D) L)
  (circle_ω2_touches_AB_CD_outside : OutsideTouches ω2 (Line A B) ∧ OutsideTouches ω2 (Line C D))
  (O_K_L_collinear : Collinear O K L)
  (M1 M2 I : Type)
  [Midpoint_of_Side M1 (Side B C)] [Midpoint_of_Side M2 (Side A D)]
  [Center_of_Circle I ω] :
  Collinear M1 M2 I :=
by 
  sorry

end midpoints_and_center_are_collinear_l340_340928


namespace period_sin_x_plus_2cos_x_l340_340843

def period_of_sin_x_plus_2_cos_x : Prop :=
  ∃ (T : ℝ), T > 0 ∧ ∀ x : ℝ, sin x + 2 * cos x = sin (x + T)

theorem period_sin_x_plus_2cos_x : period_of_sin_x_plus_2_cos_x :=
by
  sorry

end period_sin_x_plus_2cos_x_l340_340843


namespace uncolored_vertex_not_original_hexagon_vertex_l340_340284

theorem uncolored_vertex_not_original_hexagon_vertex
    (point_index : ℕ)
    (orig_hex_vertices : Finset ℕ) -- Assuming the vertices of the original hexagon are represented as a finite set of indices.
    (num_parts : ℕ := 1000) -- Each hexagon side is divided into 1000 parts
    (label : ℕ → Fin 3) -- A function labeling each point with 0, 1, or 2.
    (is_valid_labeling : ∀ (i j k : ℕ), label i ≠ label j ∧ label j ≠ label k ∧ label k ≠ label i) -- No duplicate labeling within a triangle.
    (is_single_uncolored : ∀ (p : ℕ), (p ∈ orig_hex_vertices ∨ ∃ (v : ℕ), v ∈ orig_hex_vertices ∧ p = v) → p ≠ point_index) -- Only one uncolored point
    : point_index ∉ orig_hex_vertices :=
by sorry

end uncolored_vertex_not_original_hexagon_vertex_l340_340284


namespace total_scoops_l340_340740

-- Define the conditions as variables
def flourCups := 3
def sugarCups := 2
def scoopSize := 1/3

-- Define what needs to be proved, i.e., the total amount of scoops needed
theorem total_scoops (flourCups sugarCups : ℚ) (scoopSize : ℚ) : 
  (flourCups / scoopSize) + (sugarCups / scoopSize) = 15 := 
by
  sorry

end total_scoops_l340_340740


namespace proposition_A_l340_340717

-- Define the necessary structures and properties for lines and planes
variables (l m n : Type) (α β : Type)
variables [Line l] [Line m] [Line n] [Plane α] [Plane β]

-- Define non-coincidence axiom for lines and planes
axiom non_coincident_lines (l m n : Type) [Line l] [Line m] [Line n] : l ≠ m ∧ m ≠ n ∧ l ≠ n
axiom non_coincident_planes (α β : Type) [Plane α] [Plane β] : α ≠ β

-- Define perpendicularity and parallelism between lines and planes
axiom perp_line_plane (l : Type) [Line l] (α : Type) [Plane α] : Prop
axiom parallel_line_plane (l : Type) [Line l] (β : Type) [Plane β] : Prop
axiom perp_plane_plane (α : Type) [Plane α] (β : Type) [Plane β] : Prop

-- State the theorem
theorem proposition_A (l : Type) [Line l] (α β : Type) [Plane α] [Plane β] :
  non_coincident_lines l l l → non_coincident_planes α β →
  perp_line_plane l α → parallel_line_plane l β → perp_plane_plane α β := 
by
  intros
  sorry

end proposition_A_l340_340717


namespace sum_inverses_mod_17_l340_340960

theorem sum_inverses_mod_17 : 
  (3⁻¹ + 3⁻² + 3⁻³ + 3⁻⁴ + 3⁻⁵ + 3⁻⁶) % 17 = 3 % 17 := 
by 
  sorry

end sum_inverses_mod_17_l340_340960


namespace repetend_of_five_over_eleven_l340_340251

noncomputable def repetend_of_decimal_expansion (n d : ℕ) : ℕ := sorry

theorem repetend_of_five_over_eleven : repetend_of_decimal_expansion 5 11 = 45 :=
by sorry

end repetend_of_five_over_eleven_l340_340251


namespace partition_space_by_planes_l340_340475

theorem partition_space_by_planes (n : ℕ) 
  (h1 : ∀ (p1 p2 p3 : α), 
    p1 ∩ p2 = ∅ → p1 ∩ p3 = ∅ → p2 ∩ p3 = ∅ → 
    n = 4 ∨ n = 6 ∨ n = 7 ∨ n = 8) : 
  ∃ (p1 p2 p3 : α), 
    p1 ∩ p2 = ∅ ∧ p1 ∩ p3 = ∅ ∧ p2 ∩ p3 = ∅ ∧
    (n = 4 ∨ n = 6 ∨ n = 7 ∨ n = 8) := sorry

end partition_space_by_planes_l340_340475


namespace women_decreased_by_3_l340_340367

noncomputable def initial_men := 12
noncomputable def initial_women := 27

theorem women_decreased_by_3 
  (ratio_men_women : 4 / 5 = initial_men / initial_women)
  (men_after_enter : initial_men + 2 = 14)
  (women_after_leave : initial_women - 3 = 24) :
  (24 - 27 = -3) :=
by
  sorry

end women_decreased_by_3_l340_340367


namespace derivative_check_l340_340807

theorem derivative_check :
  let statement1 := deriv (λ x : ℝ, 3^x) = λ x, 3^x * log 3
  let statement2 := deriv (λ x : ℝ, log x / log 2) = λ x, 1 / (x * log 2)
  let statement3 := deriv (λ x : ℝ, exp x) = λ x, exp x
  let statement4 := deriv (λ x : ℝ, 1 / log x) = λ x, - 1 / (x * (log x)^2)
  (if statement1 ≠ (λ x : ℝ, 3^x * log (3)) then 0 else 1) +
  (if statement2 = (λ x, 1 / (x * log 2)) then 1 else 0) +
  (if statement3 = (λ x, exp x) then 1 else 0) +
  (if statement4 ≠ (λ x, x) then 0 else 1) = 2 :=
by
  intro statement1 statement2 statement3 statement4
  -- statement1 corresponds to ①
  have h1 : ¬(statement1 ≠ (λ x : ℝ, 3^x * log (3))) := sorry
  -- statement2 corresponds to ②
  have h2 : (statement2 = (λ x, 1 / (x * log 2))) := sorry
  -- statement3 corresponds to ③
  have h3 : (statement3 = (λ x, exp x)) := sorry
  -- statement4 corresponds to ④
  have h4 : ¬(statement4 = (λ x, x)) := sorry
  rw [if_neg h1, if_pos h2, if_pos h3, if_neg h4]
  exact rfl

end derivative_check_l340_340807


namespace length_of_crease_l340_340218

theorem length_of_crease (θ : ℝ) : 
  let B := 5
  let DM := 5 * (Real.tan θ)
  DM = 5 * (Real.tan θ) := 
by 
  sorry

end length_of_crease_l340_340218


namespace sin_225_cos_225_l340_340934

noncomputable def sin_225_eq_neg_sqrt2_div_2 : Prop :=
  Real.sin (225 * Real.pi / 180) = -Real.sqrt 2 / 2

noncomputable def cos_225_eq_neg_sqrt2_div_2 : Prop :=
  Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2

theorem sin_225 : sin_225_eq_neg_sqrt2_div_2 := by
  sorry

theorem cos_225 : cos_225_eq_neg_sqrt2_div_2 := by
  sorry

end sin_225_cos_225_l340_340934


namespace exists_large_b_l340_340380

def sequence_b (a : ℕ → ℕ) (n : ℕ) : ℝ :=
(a (finset.range (n + 1)).prod / (finset.range (n + 1)).sum a : ℝ)

theorem exists_large_b (a : ℕ → ℕ) (hpos : ∀ n, 0 < a n)
  (hint : ∀ (m : ℕ), ∃ (n : ℕ), m ≤ n ∧ sequence_b a n ∈ ℤ) :
  ∃ k : ℕ, sequence_b a k > 2021 ^ 2021 := sorry

end exists_large_b_l340_340380


namespace chessboard_domino_tiling_impossible_l340_340834

theorem chessboard_domino_tiling_impossible : 
  ¬ ∃ (f : (Fin 8 × Fin 8 \ {((0, 0) : Fin 8 × Fin 8), (7, 7)}) → (Fin 8 × Fin 8)), 
  (∀ (x y : Fin 8 × Fin 8 \ {((0, 0) : Fin 8 × Fin 8), (7, 7)}), f x = y → (x ≠ y ∧ adjacent x y)) ∧
  (∀ (x y : Fin 8 × Fin 8), (x ≠ ((0,0): Fin 8 × Fin 8) ∧ x ≠ ((7,7): Fin 8 × Fin 8) ∧ y ≠ ((0,0): Fin 8 × Fin 8) ∧ y ≠ ((7,7): Fin 8 × Fin 8)) → adjacent x y → ∃ (a b : (Fin 8 × Fin 8)), a ≠ b ∧ f a = b)  :=
by sorry

end chessboard_domino_tiling_impossible_l340_340834


namespace max_sum_of_distances_l340_340624

noncomputable def max_sum_of_distances_to_focus 
  (ellipse : ℝ → ℝ → Prop) 
  (F1 F2 : ℝ × ℝ) 
  (line : ℝ × ℝ → Prop) : ℝ := 5

theorem max_sum_of_distances
  (ellipse : ℝ → ℝ → Prop := λ x y, x^2 / 4 + y^2 / 3 = 1)
  (F1 : ℝ × ℝ := (-1, 0))
  (F2 : ℝ × ℝ := (1, 0))
  (line : ℝ × ℝ → Prop) 
  (A B : ℝ × ℝ)
  (hA : ellipse A.1 A.2) 
  (hB : ellipse B.1 B.2)
  (hlA : line A) 
  (hlB : line B) : 
  max_sum_of_distances_to_focus ellipse F1 F2 line = 5 := sorry

end max_sum_of_distances_l340_340624


namespace inequality_proof_l340_340720

theorem inequality_proof (x y z : ℝ) 
  (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z)
  (hx1 : x ≤ 1) (hy1 : y ≤ 1) (hz1 : z ≤ 1) :
  2 * (x^3 + y^3 + z^3) - (x^2 * y + y^2 * z + z^2 * x) ≤ 3 :=
by
  sorry

end inequality_proof_l340_340720


namespace projection_relation_l340_340608

open EuclideanGeometry

structure Triangle (α : Type*) :=
(A B C : α)

variables {α : Type*} [MetricSpace α]

def plane (α : Type*) := set α

noncomputable def construct_line (A : α) (S : plane α) : α → α := sorry

def project_point (P : α) (g : α → α) : α := g P

theorem projection_relation (ABC : Triangle α) (S : plane α) (g : α → α)
  (h_g : ∃ A, g = construct_line A S) :
  ∀ (B1 C1 : α) (hB1 : B1 = project_point ABC.B g) (hC1 : C1 = project_point ABC.C g),
  dist ABC.A B1 = 2 * dist ABC.A C1 :=
begin
  sorry
end

end projection_relation_l340_340608


namespace find_angle_C_find_perimeter_l340_340294

theorem find_angle_C (a b c A B C : ℝ) (h1 : 2 * cos C * (a * cos B + b * cos A) = c) : C = π / 3 :=
by 
  sorry

theorem find_perimeter (a b c A B : ℝ) 
  (C : ℝ)
  (hC : C = π / 3)
  (h2 : c = sqrt 7)
  (area : ℝ) 
  (h3 : area = 3 * sqrt 3 / 2) 
  (h4 : 2 * (a * b * sin C / 2) = area) :
  (a + b + c) = 5 + sqrt 7 :=
by 
  sorry

end find_angle_C_find_perimeter_l340_340294


namespace inequality_for_positive_real_l340_340062

theorem inequality_for_positive_real (x : ℝ) (h : 0 < x) : x + 1/x ≥ 2 :=
by
  sorry

end inequality_for_positive_real_l340_340062


namespace obtuse_triangle_exists_l340_340614

/-- Given five points in the plane where no three are collinear,
    among these five points, there always exist three points that form an obtuse-angled triangle. -/
theorem obtuse_triangle_exists (points : Fin 5 → ℝ × ℝ)
    (h_no_three_collinear : ∀ i j k : Fin 5, i ≠ j → j ≠ k → i ≠ k → ¬Collinear ℝ {points i, points j, points k}) :
    ∃ (i j k : Fin 5), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ is_obtuse_angle (triangle_angle (points i) (points j) (points k)) :=
begin
    sorry
end

end obtuse_triangle_exists_l340_340614


namespace tony_age_at_end_of_period_l340_340107

theorem tony_age_at_end_of_period (x : ℕ) (a : ℕ):
  (∀ (x : ℕ), 0 ≤ x ≤ 100 →
  (1.9 * a * x + 1.9 * (a + 1) * (100 - x) = 3750) →
  (\exists b : ℕ, b = if x = 100 then a else a + 1) →
  15 = a ∨ 15 = a + 1) :=
begin
  sorry
end

end tony_age_at_end_of_period_l340_340107


namespace strict_decreasing_interval_l340_340582

open Real

noncomputable def f (x : ℝ) : ℝ := x / (log x)

theorem strict_decreasing_interval :
  ∀ x : ℝ, 1 < x ∧ x < exp 1 → deriv f x < 0 := 
by 
  intro x hx
  have h_diff : fderiv ℝ f x = (log x - 1) / (log x)^2 := by 
    -- here should be the computation of derivative
    sorry 
  have h_numer_neg : log x - 1 < 0 := by 
    -- here should be the proof that log x - 1 < 0 when 1 < x < e
    sorry
  have h_denom_pos : (log x)^2 > 0 := by 
    -- here should be the proof that (log x)^2 > 0 for all x > 0
    sorry
  exact 
    -- combining the results to show deriv f x < 0
    sorry

end strict_decreasing_interval_l340_340582


namespace gas_volumes_correct_l340_340509

noncomputable def west_gas_vol_per_capita : ℝ := 21428
noncomputable def non_west_gas_vol : ℝ := 185255
noncomputable def non_west_population : ℝ := 6.9
noncomputable def non_west_gas_vol_per_capita : ℝ := non_west_gas_vol / non_west_population

noncomputable def russia_gas_vol_68_percent : ℝ := 30266.9
noncomputable def russia_gas_vol : ℝ := russia_gas_vol_68_percent * 100 / 68
noncomputable def russia_population : ℝ := 0.147
noncomputable def russia_gas_vol_per_capita : ℝ := russia_gas_vol / russia_population

theorem gas_volumes_correct :
  west_gas_vol_per_capita = 21428 ∧
  non_west_gas_vol_per_capita = 26848.55 ∧
  russia_gas_vol_per_capita = 302790.13 := by
    sorry

end gas_volumes_correct_l340_340509


namespace ship_illuminated_by_lighthouse_l340_340074

theorem ship_illuminated_by_lighthouse (d v : ℝ) (hv : v > 0) (ship_speed : ℝ) 
    (hship_speed : ship_speed ≤ v / 8) (rock_distance : ℝ) 
    (hrock_distance : rock_distance = d):
    ∀ t : ℝ, ∃ t' : ℝ, t' ≤ t ∧ t' = (d * t / v) := sorry

end ship_illuminated_by_lighthouse_l340_340074


namespace no_prime_in_sequence_number_of_primes_in_sequence_l340_340222

def product_of_primes_up_to (n : ℕ) : ℕ :=
  (Nat.range (n + 1)).filter Nat.prime |>.prod

noncomputable def Q : ℕ := product_of_primes_up_to 67

def is_prime_seq (m : ℕ) : ℕ := Q + m

theorem no_prime_in_sequence : 
  ∀ m : ℕ, 2 ≤ m → m ≤ 65 → ¬ Nat.Prime (is_prime_seq m) :=
sorry

theorem number_of_primes_in_sequence : Finset.card ((Finset.range 64).filter (λ n => Nat.Prime (is_prime_seq (n + 2)))) = 0 :=
sorry

end no_prime_in_sequence_number_of_primes_in_sequence_l340_340222


namespace calculation_result_l340_340209

theorem calculation_result : ((55 * 45 - 37 * 43) - (3 * 221 + 1)) / 22 = 10 := by
  sorry

end calculation_result_l340_340209


namespace scientific_notation_29150000_l340_340908

theorem scientific_notation_29150000 :
  29150000 = 2.915 * 10^7 := sorry

end scientific_notation_29150000_l340_340908


namespace problem_statement_l340_340595

theorem problem_statement (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : b < a) (f : ℝ → ℝ) 
  (h_f : ∀ x, f x = |Real.log x|) (h_eq : f a = f b) :
  a * b = 1 ∧ Real.exp a + Real.exp b > 2 * Real.exp 1 ∧ (1 / a)^2 - b + 5 / 4 ≥ 1 :=
by
  sorry

end problem_statement_l340_340595


namespace number_of_valid_three_digit_numbers_l340_340650

def is_valid_digit (d : Nat) : Prop :=
  d ≠ 3

def is_valid_hundreds_digit (a : Nat) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ is_valid_digit a

def is_valid_tens_digit (b : Nat) : Prop :=
  0 ≤ b ∧ b ≤ 9 ∧ is_valid_digit b

def is_valid_units_digit (c : Nat) : Prop :=
  c ∈ {1, 5, 7, 9} ∧ is_valid_digit c

def is_valid_three_digit_number (a b c : Nat) : Prop :=
  is_valid_hundreds_digit a ∧ is_valid_tens_digit b ∧ is_valid_units_digit c ∧ 
  (a + b + c) % 3 = 0

def count_valid_three_digit_numbers : Nat :=
  Set.toFinset {n | ∃ a b c, n = 100 * a + 10 * b + c ∧ is_valid_three_digit_number a b c}.card

theorem number_of_valid_three_digit_numbers : count_valid_three_digit_numbers = 96 := 
  by
  sorry

end number_of_valid_three_digit_numbers_l340_340650


namespace tip_percentage_l340_340161

theorem tip_percentage (cost_of_crown : ℕ) (total_paid : ℕ) (h1 : cost_of_crown = 20000) (h2 : total_paid = 22000) :
  (total_paid - cost_of_crown) * 100 / cost_of_crown = 10 :=
by
  sorry

end tip_percentage_l340_340161


namespace equation_of_line_l_l340_340299

-- Define the conditions using Lean definitions

def is_perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

def line_equation (m b x : ℝ) : ℝ := m * x + b

noncomputable def given_y_intercept : ℝ := 1
noncomputable def given_slope_of_perpendicular_line : ℝ := 1 / 2
noncomputable def calculated_slope : ℝ :=
if is_perpendicular given_slope_of_perpendicular_line (-2) then -2 else 0

-- The theorem we need to prove
theorem equation_of_line_l (x y : ℝ) :
  y = line_equation calculated_slope given_y_intercept x :=
by
  sorry

end equation_of_line_l_l340_340299


namespace point_movement_end_position_l340_340051

theorem point_movement_end_position :
  ∀ (A : ℝ), (A = 3 ∨ A = -3) → (A + 4 - 1 = 6 ∨ A + 4 - 1 = 0) := by
  intros A h
  cases h
  . simp [h]
  . simp [h]
  sorry

end point_movement_end_position_l340_340051


namespace canteen_initial_water_l340_340327

/-- Harry started a 7-mile hike with a full canteen of water and finished the hike in 2 hours 
with 3 cups of water remaining in the canteen. The canteen leaked at the rate of 1 cup per hour 
and Harry drank 2 cups of water during the last mile. He drank 0.6666666666666666 cups per mile 
during the first 6 miles of the hike. -/
theorem canteen_initial_water :
  ∀ (remaining_water leaked_rate hours_last_mile drink_last_mile drink_per_mile miles_first last_miles: ℝ), 
  remaining_water = 3 ∧
  leaked_rate = 1 ∧
  hours_last_mile = 2 ∧
  drink_last_mile = 2 ∧
  drink_per_mile = 0.6666666666666666 ∧
  miles_first = 6 ∧
  last_miles = 1 →
  let total_leaked := hours_last_mile * leaked_rate in
  let total_drunk_first := miles_first * drink_per_mile in
  let total_remaining := remaining_water + total_leaked + drink_last_mile + total_drunk_first in
  total_remaining = 11 := 
by 
  intros;
  sorry

end canteen_initial_water_l340_340327


namespace complex_conjugate_l340_340079

theorem complex_conjugate (z : ℂ) (hz : z = (2 - ⅈ) / (1 + 2 * ⅈ)) : conj z = ⅈ := 
by 
  sorry

end complex_conjugate_l340_340079


namespace average_of_remaining_two_l340_340139

-- Given conditions
def average_of_six (S : ℝ) := S / 6 = 3.95
def average_of_first_two (S1 : ℝ) := S1 / 2 = 4.2
def average_of_next_two (S2 : ℝ) := S2 / 2 = 3.85

-- Prove that the average of the remaining 2 numbers equals 3.8
theorem average_of_remaining_two (S S1 S2 Sr : ℝ) (h1 : average_of_six S) (h2 : average_of_first_two S1) (h3: average_of_next_two S2) (h4 : Sr = S - S1 - S2) :
  Sr / 2 = 3.8 :=
by
  -- We can use the assumptions h1, h2, h3, and h4 to reach the conclusion
  sorry

end average_of_remaining_two_l340_340139


namespace z_share_profit_correct_l340_340140

-- Define the investments as constants
def x_investment : ℕ := 20000
def y_investment : ℕ := 25000
def z_investment : ℕ := 30000

-- Define the number of months for each investment
def x_months : ℕ := 12
def y_months : ℕ := 12
def z_months : ℕ := 7

-- Define the annual profit
def annual_profit : ℕ := 50000

-- Calculate the active investment
def x_share : ℕ := x_investment * x_months
def y_share : ℕ := y_investment * y_months
def z_share : ℕ := z_investment * z_months

-- Calculate the total investment
def total_investment : ℕ := x_share + y_share + z_share

-- Define Z's ratio in terms of the total investment
def z_ratio : ℚ := z_share / total_investment

-- Calculate Z's share of the annual profit
def z_profit_share : ℚ := z_ratio * annual_profit

-- Theorem to prove Z's share in the annual profit
theorem z_share_profit_correct : z_profit_share = 14000 := by
  sorry

end z_share_profit_correct_l340_340140


namespace basketball_substitution_mod_1000_l340_340872

theorem basketball_substitution_mod_1000 :
  let num_ways_substitutions := 1 + 5 * 9 + 45 * 4 * 8 + 1440 * 3 * 7 + 30240 * 2 * 6
  in num_ways_substitutions % 1000 = 606 :=
by
  let num_ways_substitutions := 1 + 5 * 9 + 45 * 4 * 8 + 1440 * 3 * 7 + 30240 * 2 * 6
  show num_ways_substitutions % 1000 = 606
  sorry

end basketball_substitution_mod_1000_l340_340872


namespace sum_of_three_sqrt_139_l340_340670

theorem sum_of_three_sqrt_139 {x y z : ℝ} (h1 : x >= 0) (h2 : y >= 0) (h3 : z >= 0)
  (hx : x^2 + y^2 + z^2 = 75) (hy : x * y + y * z + z * x = 32) : x + y + z = Real.sqrt 139 := 
by
  sorry

end sum_of_three_sqrt_139_l340_340670


namespace total_salad_dressing_weight_l340_340411

noncomputable def bowl_volume := 150 -- Volume of the bowl in ml
def oil_fraction := 2 / 3 -- Fraction of the bowl that is oil
def vinegar_fraction := 1 / 3 -- Fraction of the bowl that is vinegar
def oil_density := 5 -- Density of oil (g/ml)
def vinegar_density := 4 -- Density of vinegar (g/ml)

def oil_volume := oil_fraction * bowl_volume -- Volume of oil in ml
def vinegar_volume := vinegar_fraction * bowl_volume -- Volume of vinegar in ml
def oil_weight := oil_volume * oil_density -- Weight of the oil in grams
def vinegar_weight := vinegar_volume * vinegar_density -- Weight of the vinegar in grams
def total_weight := oil_weight + vinegar_weight -- Total weight of the salad dressing in grams

theorem total_salad_dressing_weight : total_weight = 700 := by
  sorry

end total_salad_dressing_weight_l340_340411


namespace lines_are_perpendicular_l340_340631

noncomputable def line1 := {x : ℝ | ∃ y : ℝ, x + y - 1 = 0}
noncomputable def line2 := {x : ℝ | ∃ y : ℝ, x - y + 1 = 0}

theorem lines_are_perpendicular : 
  let slope1 := -1
  let slope2 := 1
  slope1 * slope2 = -1 := sorry

end lines_are_perpendicular_l340_340631


namespace Thursday_total_rainfall_correct_l340_340370

def Monday_rainfall : ℝ := 0.9
def Tuesday_rainfall : ℝ := Monday_rainfall - 0.7
def Wednesday_rainfall : ℝ := Tuesday_rainfall + 0.5 * Tuesday_rainfall
def additional_rain : ℝ := 0.3
def decrease_factor : ℝ := 0.2
def Thursday_rainfall_before_addition : ℝ := Wednesday_rainfall - decrease_factor * Wednesday_rainfall
def Thursday_total_rainfall : ℝ := Thursday_rainfall_before_addition + additional_rain

theorem Thursday_total_rainfall_correct :
  Thursday_total_rainfall = 0.54 :=
by
  sorry

end Thursday_total_rainfall_correct_l340_340370


namespace angles_cos_eq_l340_340381

-- Definitions for angles of a triangle and given conditions
variables {A B C : ℝ} (ht : A + B + C = π) (hC : C > π/2)
variables {cosA cosB cosC sinA sinB sinC : ℝ}
  (hcosA : cosA = cos A) (hcosB : cosB = cos B) (hcosC : cosC = cos C)
  (hsinA : sinA = sin A) (hsinB : sinB = sin B) (hsinC : sinC = sin C)

-- Given equations
variables (eq1 : cosA ^ 2 + cosC ^ 2 + 2 * sinA * sinC * cosB = 17 / 9)
          (eq2 : cosC ^ 2 + cosB ^ 2 + 2 * sinC * sinB * cosA = 12 / 7)

-- The statement to be proved
theorem angles_cos_eq : ∃ (p q r s : ℤ), 
  p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0 ∧ p + q ≠ 0 ∧ Nat.gcd (Int.toNat p + Int.toNat q) (Int.toNat s) = 1 ∧ 
  ¬ (∃ k : ℤ, k ^ 2 = r) ∧ 
  cosB ^ 2 + cosA ^ 2 + 2 * sinB * sinA * cosC = (p - q * real.sqrt r) / s ∧ 
  p + q + r + s = 220 :=
by
  sorry

end angles_cos_eq_l340_340381


namespace total_scoops_l340_340741

-- Define the conditions as variables
def flourCups := 3
def sugarCups := 2
def scoopSize := 1/3

-- Define what needs to be proved, i.e., the total amount of scoops needed
theorem total_scoops (flourCups sugarCups : ℚ) (scoopSize : ℚ) : 
  (flourCups / scoopSize) + (sugarCups / scoopSize) = 15 := 
by
  sorry

end total_scoops_l340_340741


namespace intersection_point_of_MN_with_base_plane_l340_340777

-- Definitions of the points and the parallelepiped
variables {A B C D A1 B1 C1 D1 M N P : Type}
variables [Plane A B C D] [Parallelepiped A B C D A1 B1 C1 D1]

-- Definitions of edges
variables (BC : Line B C) (AA1 : Line A A1)
variables (base_plane : Plane A1 B1 C1 D1)

-- Conditions that M and N lie on respective edges
def M_on_BC : Prop := M ∈ BC
def N_on_AA1 : Prop := N ∈ AA1

-- The line MN
def MN : Line M N := Line.mk M N

-- Statement to prove the intersection
theorem intersection_point_of_MN_with_base_plane 
  (h1 : M_on_BC M)
  (h2 : N_on_AA1 N) :
  ∃ P, P ∈ (MN : Set Point) ∧ P ∈ (base_plane : Set Point) :=
sorry

end intersection_point_of_MN_with_base_plane_l340_340777


namespace no_eulerian_path_no_hamiltonian_path_l340_340534

variables {V : Type*} [DecidableEq V] (G : SimpleGraph V)

-- Problem (a): No Eulerian path exists in a graph with 19 vertices, 30 edges, and more than two odd degree vertices
theorem no_eulerian_path (hV : Fintype.card V = 19) (hE : G.edgeFinset.card = 30)
  (hOdd : ∃ v₁ v₂ v₃, v₁ ≠ v₂ ∧ v₂ ≠ v₃ ∧ v₁ ≠ v₃ ∧ G.degree v₁ % 2 = 1 ∧ G.degree v₂ % 2 = 1 ∧ G.degree v₃ % 2 = 1) :
  ¬ G.hasEulerianTrail :=
sorry

-- Problem (b): No Hamiltonian path exists in a bipartite graph with 7 black vertices and 12 white vertices
theorem no_hamiltonian_path (G : SimpleGraph V) [decidable_rel G.adj]
  (hsTripartite : G.isBipartite) (hBlack : G.part.card = 7) (hWhite : G.compPart.card = 12) :
  ¬ G.hasHamiltonianPath :=
sorry

end no_eulerian_path_no_hamiltonian_path_l340_340534


namespace divides_polynomial_at_integer_l340_340031

theorem divides_polynomial_at_integer 
    (f : ℤ[X]) 
    (a₀ a₁ : ℤ) 
    (n : ℕ)
    (a : Fin n.succ → ℤ)
    (p q : ℤ)
    (hpq_irreducible : isCoprime p q) 
    (root_rational : f.eval (Rat.mk p q) = 0)
    (k : ℤ) 
    : q * k - p ∣ f.eval k := 
sorry

end divides_polynomial_at_integer_l340_340031


namespace rows_of_potatoes_l340_340926

theorem rows_of_potatoes (total_potatoes : ℕ) (seeds_per_row : ℕ) (h1 : total_potatoes = 54) (h2 : seeds_per_row = 9) : total_potatoes / seeds_per_row = 6 := 
by
  sorry

end rows_of_potatoes_l340_340926


namespace exists_xy_for_a_cubed_l340_340137

theorem exists_xy_for_a_cubed (a : ℕ) (ha : a > 0) : 
  ∃ x y : ℤ, x^2 - y^2 = a^3 :=
by
  let x := (a^2 + a) / 2
  let y := (a^2 - a) / 2
  use [x, y]
  sorry

end exists_xy_for_a_cubed_l340_340137


namespace triangle_obtuse_l340_340365

theorem triangle_obtuse (A B : ℝ) (h : sin A * sin B < cos A * cos B) : 
  ∃ C, C > π / 2 ∧ ∃ a b c : ℝ, a + b + c = π ∧ a = A ∧ b = B ∧ c = C :=
by
  sorry

end triangle_obtuse_l340_340365


namespace lcm_of_two_numbers_l340_340830

-- Definitions based on the conditions
variable (a b l : ℕ)

-- The conditions from the problem
def hcf_ab : Nat := 9
def prod_ab : Nat := 1800

-- The main statement to prove
theorem lcm_of_two_numbers : Nat.lcm a b = 200 :=
by
  -- Skipping the proof implementation
  sorry

end lcm_of_two_numbers_l340_340830


namespace graphs_intersect_at_one_point_l340_340939

theorem graphs_intersect_at_one_point :
  ∃! x : ℝ, 1 < x ∧ 3 * log x = log (3 * x) := sorry

end graphs_intersect_at_one_point_l340_340939


namespace sum_of_first_50_primes_is_5356_l340_340851

open Nat

-- Define the first 50 prime numbers
def first_50_primes : List Nat := 
  [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 
   83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 
   179, 181, 191, 193, 197, 199, 211, 223, 227, 229]

-- Calculate their sum
def sum_first_50_primes : Nat := List.foldr (Nat.add) 0 first_50_primes

-- Now we state the theorem we want to prove
theorem sum_of_first_50_primes_is_5356 : 
  sum_first_50_primes = 5356 := 
by
  -- Placeholder for proof
  sorry

end sum_of_first_50_primes_is_5356_l340_340851


namespace part1_part2_l340_340722

noncomputable def f (a x : ℝ) : ℝ := a*x^2 + x - a
def abs (y : ℝ) : ℝ := if y >= 0 then y else -y

theorem part1 (a x : ℝ) (h₁ : abs a ≤ 1) (h₂ : abs x ≤ 1) : abs (f a x) ≤ 5/4 := 
sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, abs x ≤ 1 → f (-2) x ≤ 17/8) ∧ (∃ x : ℝ, abs x ≤ 1 ∧ f (-2) x = 17/8) := 
sorry

end part1_part2_l340_340722


namespace total_bananas_l340_340412

theorem total_bananas (bunches_8 bunches_7 : ℕ) (bananas_8 bananas_7 : ℕ) (h_bunches_8 : bunches_8 = 6) (h_bananas_8 : bananas_8 = 8) (h_bunches_7 : bunches_7 = 5) (h_bananas_7 : bananas_7 = 7) :
  bunches_8 * bananas_8 + bunches_7 * bananas_7 = 83 :=
by
  rw [h_bunches_8, h_bananas_8, h_bunches_7, h_bananas_7]
  norm_num

end total_bananas_l340_340412


namespace shape_volume_to_surface_area_ratio_l340_340890

/-- 
Define the volume and surface area of our specific shape with given conditions:
1. Five unit cubes in a straight line.
2. An additional cube on top of the second cube.
3. Another cube beneath the fourth cube.

Prove that the ratio of the volume to the surface area is \( \frac{1}{4} \).
-/
theorem shape_volume_to_surface_area_ratio :
  let volume := 7
  let surface_area := 28
  volume / surface_area = 1 / 4 :=
by
  sorry

end shape_volume_to_surface_area_ratio_l340_340890


namespace meal_cost_approx_l340_340198

variable (x : ℝ) -- Cost of the meal before tax and tip

-- Conditions
variable (tax_rate : ℝ := 0.12) -- 12% tax rate
variable (tip_rate : ℝ := 0.18) -- 18% tip rate
variable (total_amount : ℝ := 33) -- Total amount paid

-- Define the total cost function
def total_cost (x : ℝ) : ℝ := x + tax_rate * x + tip_rate * x

-- The goal is to prove that x is approximately 25.38
theorem meal_cost_approx : total_cost x = total_amount → x ≈ 25.38 :=
sorry

end meal_cost_approx_l340_340198


namespace jade_transactions_l340_340046

theorem jade_transactions 
  (mabel_trans : ℕ)
  (ant_more_perc : ℕ)
  (cal_frac : ℚ)
  (jade_more : ℕ)
  (h_mabel : mabel_trans = 90)
  (h_ant : ant_more_perc = 10)
  (h_cal : cal_frac = 2/3)
  (h_jade : jade_more = 17) : 
  (jade_handled : ℕ) :=
  have ant_handled : ℕ := mabel_trans + (ant_more_perc * mabel_trans / 100),
  have cal_handled : ℕ := (cal_frac * ant_handled).to_nat,
  have jade_handled := cal_handled + jade_more,
  show jade_handled = 83, from sorry

end jade_transactions_l340_340046


namespace size_of_first_type_package_is_5_l340_340212

noncomputable def size_of_first_type_package (total_coffee : ℕ) (num_first_type : ℕ) (num_second_type : ℕ) (size_second_type : ℕ) : ℕ :=
  (total_coffee - num_second_type * size_second_type) / num_first_type

theorem size_of_first_type_package_is_5 :
  size_of_first_type_package 70 (4 + 2) 4 10 = 5 :=
by
  sorry

end size_of_first_type_package_is_5_l340_340212


namespace range_of_modulus_l340_340078

theorem range_of_modulus (z : ℂ) (h : abs (z - 3 + 4 * complex.i) = 1) : 
  4 ≤ abs z ∧ abs z ≤ 6 := 
sorry

end range_of_modulus_l340_340078


namespace min_value_sin_cos_expression_l340_340584

theorem min_value_sin_cos_expression (x : ℝ) : 
  sin x ^ 4 + 2 * cos x ^ 4 + sin x ^ 2 ≥ 2 / 3 := 
sorry

end min_value_sin_cos_expression_l340_340584


namespace exists_committees_with_one_member_in_common_l340_340884

theorem exists_committees_with_one_member_in_common 
  (n : ℕ) (h : n ≥ 5) 
  (committees : fin (n + 1) → finset (fin n))
  (h_card : ∀ (i : fin (n + 1)), (committees i).card = 3)
  (h_distinct : ∀ (i j : fin (n + 1)), i ≠ j → committees i ≠ committees j) :
  ∃ (i j : fin (n + 1)), i ≠ j ∧ (committees i ∩ committees j).card = 1 :=
sorry

end exists_committees_with_one_member_in_common_l340_340884


namespace value_at_2_is_14_l340_340109

def polynomial (x : ℝ) : ℝ := 2 * x^6 + 3 * x^5 + 5 * x^3 + 6 * x^2 + 7 * x + 8

theorem value_at_2_is_14 : polynomial 2 = 14 := 
by 
  have v0 : ℝ := 2
  have v1 : ℝ := v0 * 2 + 3
  have v2 : ℝ := v1 * 2
  have h : 14 = v2 := by simp [v0, v1, v2]; norm_num
  rw [h]
  sorry

end value_at_2_is_14_l340_340109


namespace factor_expression_l340_340966

variable (b : ℤ)

theorem factor_expression : 221 * b^2 + 17 * b = 17 * b * (13 * b + 1) :=
by
  sorry

end factor_expression_l340_340966


namespace kids_all_three_activities_l340_340103

-- Definitions based on conditions
def total_kids : ℕ := 40
def kids_tubing : ℕ := total_kids / 4
def kids_tubing_rafting : ℕ := kids_tubing / 2
def kids_tubing_rafting_kayaking : ℕ := kids_tubing_rafting / 3

-- Theorem statement: proof of the final answer
theorem kids_all_three_activities : kids_tubing_rafting_kayaking = 1 := by
  sorry

end kids_all_three_activities_l340_340103


namespace AB_side_length_l340_340770

noncomputable def P := (x : ℝ) × (y : ℝ)

def is_foot_perpendicular (P : P) (A B : P) : P := sorry

def equilateral_triangle (A B C : P) : Prop := sorry

theorem AB_side_length (A B C P Q R S : P)
  (h_equilateral : equilateral_triangle A B C)
  (h_P_inside : sorry /* P inside ABC */)
  (h_Q_foot : Q = is_foot_perpendicular P A B) 
  (h_R_foot : R = is_foot_perpendicular P B C)
  (h_S_foot : S = is_foot_perpendicular P C A)
  (h_PQ : (dist P Q) = 2)
  (h_PR : (dist P R) = 3)
  (h_PS : (dist P S) = 4) :
  dist A B = 6 * real.sqrt 3 := 
sorry

end AB_side_length_l340_340770


namespace no_negative_roots_l340_340627
-- Import the entire Lean mathematical library

-- Define the function f given the parameters
def f (a : ℝ) (x : ℝ) : ℝ := a^x + (x - 2) / (x + 1)

-- State the theorem with given conditions
theorem no_negative_roots (a : ℝ) (h : a > 1) :
  ∀ x : ℝ, x < 0 → f(a, x) ≠ 0 :=
by
  sorry

end no_negative_roots_l340_340627


namespace Dana_has_25_more_pencils_than_Marcus_l340_340942

theorem Dana_has_25_more_pencils_than_Marcus (JaydenPencils : ℕ) (h1 : JaydenPencils = 20) :
  let DanaPencils := JaydenPencils + 15,
      MarcusPencils := JaydenPencils / 2
  in DanaPencils - MarcusPencils = 25 := 
by
  sorry -- proof to be filled in

end Dana_has_25_more_pencils_than_Marcus_l340_340942


namespace mod_sum_example_l340_340116

theorem mod_sum_example :
  (9^5 + 8^4 + 7^6) % 5 = 4 :=
by sorry

end mod_sum_example_l340_340116


namespace head_start_A_gives_C_l340_340348

theorem head_start_A_gives_C :
  let race_length : ℕ := 1000 in
  ∀ V_a V_b V_c T_a T_b T_c : ℝ,
    (V_a * T_a = race_length) →
    (V_b * T_b = race_length - 60) →
    (T_a = T_b) →
    (V_b * T_b = race_length) →
    (V_c * T_c = race_length - 148.936170212766) →
    (T_b = T_c) →
    ∃ (X : ℝ), X = 148.936170212766 := by
  intros V_a V_b V_c T_a T_b T_c h1 h2 h3 h4 h5 h6
  use 148.936170212766
  exact ⟨rfl⟩

end head_start_A_gives_C_l340_340348


namespace find_total_tennis_balls_l340_340176

noncomputable def original_white_balls : ℕ := sorry
noncomputable def original_yellow_balls : ℕ := sorry
noncomputable def dispatched_yellow_balls : ℕ := original_yellow_balls + 20

theorem find_total_tennis_balls
  (white_balls_eq : original_white_balls = original_yellow_balls)
  (ratio_eq : original_white_balls / dispatched_yellow_balls = 8 / 13) :
  original_white_balls + original_yellow_balls = 64 := sorry

end find_total_tennis_balls_l340_340176


namespace perimeters_sum_eq_240_l340_340546

noncomputable def sum_geometric_series (a r : ℕ) : ℕ :=
  a / (1 - r)

theorem perimeters_sum_eq_240 :
  let T1_side := 40 in
  let T1_perimeter := 3 * T1_side in
  let r := 1/2 in
  sum_geometric_series T1_perimeter r = 240 := by
  sorry

end perimeters_sum_eq_240_l340_340546


namespace platform_length_correct_l340_340539

noncomputable def train_length : ℝ := 120
noncomputable def initial_velocity_kmph : ℝ := 72
noncomputable def acceleration : ℝ := 0.5
noncomputable def time_seconds : ℝ := 25
noncomputable def initial_velocity_ms : ℝ := (initial_velocity_kmph * 1000) / (60 * 60)

theorem platform_length_correct :
  let s := initial_velocity_ms * time_seconds + (1 / 2) * acceleration * time_seconds^2 in
  let platform_length := s - train_length in
  platform_length = 536.25 :=
by
  sorry

end platform_length_correct_l340_340539


namespace range_of_k_l340_340311

-- Define the function f
def f (x : ℝ) : ℝ := if x ≥ 0 then sin x else -x^2 - 1

-- State the theorem
theorem range_of_k (k : ℝ) : (∀ x : ℝ, f x ≤ k * x) ↔ 1 ≤ k ∧ k ≤ 2 :=
by
  sorry

end range_of_k_l340_340311


namespace part1_part2_l340_340312

noncomputable def f (x : ℝ) : ℝ := 2 * sin x * (sin x + cos x)

theorem part1 (k : ℤ) : 
  ∃ x : ℝ, (x = k * π - (π / 8) ∧ f x = 1 - real.sqrt 2) :=
sorry

theorem part2 (k : ℤ) :
  ∀ x : ℝ, (k * π - (π / 8) ≤ x ∧ x ≤ k * π + (3 * π / 8)) → (2 * sin x * (sin x + cos x)).strict_mono_incr :=
sorry

end part1_part2_l340_340312


namespace triangle_perimeter_eqn_l340_340049

variable (a b : ℝ)

def first_side := a + b
def second_side := a + b + (a + 2)
def third_side := a + b + (a + 2) - 3

theorem triangle_perimeter_eqn 
  (h1 : first_side = a + b) 
  (h2 : second_side = a + b + (a + 2)) 
  (h3 : third_side = a + b + (a + 2) - 3) : 
  first_side + second_side + third_side = 5 * a + 3 * b + 1 := 
  sorry

end triangle_perimeter_eqn_l340_340049


namespace knights_rearrangement_impossible_l340_340863

theorem knights_rearrangement_impossible :
  ∀ (b : ℕ → ℕ → Prop), (b 0 0 = true) ∧ (b 0 2 = true) ∧ (b 2 0 = true) ∧ (b 2 2 = true) ∧
  (b 0 0 = b 0 2) ∧ (b 2 0 ≠ b 2 2) → ¬(∃ (b' : ℕ → ℕ → Prop), 
  (b' 0 0 ≠ b 0 0) ∧ (b' 0 2 ≠ b 0 2) ∧ (b' 2 0 ≠ b 2 0) ∧ (b' 2 2 ≠ b 2 2) ∧ 
  (b' 0 0 ≠ b' 0 2) ∧ (b' 2 0 ≠ b' 2 2)) :=
by { sorry }

end knights_rearrangement_impossible_l340_340863


namespace ab_value_l340_340479

theorem ab_value (a b : ℝ) (h1 : a + b = 8) (h2 : a^3 + b^3 = 107) : a * b = 25.3125 :=
by
  sorry

end ab_value_l340_340479


namespace sale_price_is_91_percent_of_original_price_l340_340828

variable (x : ℝ)
variable (h_increase : ∀ p : ℝ, p * 1.4)
variable (h_sale : ∀ p : ℝ, p * 0.65)

/--The sale price of an item is 91% of the original price.-/
theorem sale_price_is_91_percent_of_original_price {x : ℝ} 
  (h_increase : ∀ p, p * 1.4 = 1.40 * p)
  (h_sale : ∀ p, p * 0.65 = 0.65 * p): 
  (0.65 * 1.40 * x = 0.91 * x) := 
by 
  sorry

end sale_price_is_91_percent_of_original_price_l340_340828


namespace JacksonsGrade_l340_340008

theorem JacksonsGrade : 
  let hours_playing_video_games := 12
  let hours_studying := (1 / 3) * hours_playing_video_games
  let hours_kindness := (1 / 4) * hours_playing_video_games
  let grade_initial := 0
  let grade_per_hour_studying := 20
  let grade_per_hour_kindness := 40
  let grade_from_studying := grade_per_hour_studying * hours_studying
  let grade_from_kindness := grade_per_hour_kindness * hours_kindness
  let total_grade := grade_initial + grade_from_studying + grade_from_kindness
  total_grade = 200 :=
by
  -- Proof goes here
  sorry

end JacksonsGrade_l340_340008


namespace combined_age_l340_340654

theorem combined_age (H : ℕ) (Ryanne : ℕ) (Jamison : ℕ) 
  (h1 : Ryanne = H + 7) 
  (h2 : H + Ryanne = 15) 
  (h3 : Jamison = 2 * H) : 
  H + Ryanne + Jamison = 23 := 
by 
  sorry

end combined_age_l340_340654


namespace remaining_weight_after_deliveries_l340_340183

def initial_weight : ℝ := 50000
def first_unload_percentage : ℝ := 0.10
def second_unload_percentage : ℝ := 0.20

theorem remaining_weight_after_deliveries (w1 w2 : ℝ) : w1 = 50000 → w2 = 36000 := 
by
  intro h1
  have h2 : ℝ := w1 * first_unload_percentage
  have h3 : ℝ := w1 - h2
  have h4 : ℝ := h3 * second_unload_percentage
  have h5 : ℝ := h3 - h4
  exact calc
    w1 = 50000     : h1
       ... - (50000 * 0.10) = 45000
       ... - (45000 * 0.20) = 36000
       ... = w2

end remaining_weight_after_deliveries_l340_340183


namespace gear_q_revolutions_per_minute_l340_340215

noncomputable def gear_p_revolutions_per_minute : ℕ := 10

noncomputable def additional_revolutions : ℕ := 15

noncomputable def calculate_q_revolutions_per_minute
  (p_rev_per_min : ℕ) (additional_rev : ℕ) : ℕ :=
  2 * (p_rev_per_min / 2 + additional_rev)

theorem gear_q_revolutions_per_minute :
  calculate_q_revolutions_per_minute gear_p_revolutions_per_minute additional_revolutions = 40 :=
by
  sorry

end gear_q_revolutions_per_minute_l340_340215


namespace michael_total_revenue_l340_340039

def price_large : ℕ := 22
def price_medium : ℕ := 16
def price_small : ℕ := 7

def qty_large : ℕ := 2
def qty_medium : ℕ := 2
def qty_small : ℕ := 3

def total_revenue : ℕ :=
  (price_large * qty_large) +
  (price_medium * qty_medium) +
  (price_small * qty_small)

theorem michael_total_revenue : total_revenue = 97 :=
  by sorry

end michael_total_revenue_l340_340039


namespace total_bins_sum_l340_340570

def total_bins_soup : ℝ := 0.2
def total_bins_vegetables : ℝ := 0.35
def total_bins_fruits : ℝ := 0.15
def total_bins_pasta : ℝ := 0.55
def total_bins_canned_meats : ℝ := 0.275
def total_bins_beans : ℝ := 0.175

theorem total_bins_sum :
  total_bins_soup + total_bins_vegetables + total_bins_fruits + total_bins_pasta + total_bins_canned_meats + total_bins_beans = 1.7 :=
by
  sorry

end total_bins_sum_l340_340570


namespace required_speed_fourth_lap_l340_340043

variables {d : ℝ} -- distance of one lap
variables (speed_avg goal_speed : ℝ) (lap1 lap2 lap3 lap4 : ℝ)

def total_distance := 4 * d
def total_time := total_distance / goal_speed
def time_lap1 := d / lap1
def time_lap2 := d / lap2
def time_lap3 := d / lap3
def time_lap4 := d / lap4
def time_first_three_laps := time_lap1 + time_lap2 + time_lap3
def remaining_time := total_time - time_first_three_laps

theorem required_speed_fourth_lap
  (h1 : lap1 = 9) (h2 : lap2 = 9) (h3 : lap3 = 9) (h_goal : goal_speed = 10) :
  lap4 = 15 :=
by
  sorry

end required_speed_fourth_lap_l340_340043


namespace robot_4cube_moves_l340_340813

theorem robot_4cube_moves : 
  let quadruples := {p : (ℕ × ℕ × ℕ × ℕ) | ∀ x ∈ p, x = 0 ∨ x = 1},
      start := (0, 0, 0, 0),
      adjacent (a b : ℕ × ℕ × ℕ × ℕ) : Prop := 
        ∃ i, (a = b ⟨i⟩ 1),
      reachable (a b : ℕ × ℕ × ℕ × ℕ) (n : ℕ) : Prop := 
        ∃ p, length p = n ∧ (∀ i < n - 1, adjacent (p i) (p (i + 1))) ∧ p 0 = a ∧ p (n - 1) = b
  in 
  ∃! (n : ℕ), n = 4042 ∧ ∃ p : (nat → (ℕ × ℕ × ℕ × ℕ)), 
    (p 0 = start ∧ p (n - 1) = start ∧ ∀ i < n-1, adjacent (p i) (p (i + 1))) 
    ↔ 2^4041 + 2^8081 := 
sorry

end robot_4cube_moves_l340_340813


namespace letter_F_final_position_l340_340457

/-- 
  The final position of the letter F, initially rotated 90° clockwise, undergoes 
  a rotation of 45° clockwise around the origin, reflected in the x-axis, 
  then rotated 180° around the origin, is along the negative (x + y)-axis equally. 
--/
theorem letter_F_final_position :
  ∀ (F : Type) (initial_rotation : ℝ) (r1 : ℝ) (r2 : ℝ) (reflect_x : bool), 
    initial_rotation = 90 ∧ r1 = 45 ∧ r2 = 180 ∧ reflect_x = tt → 
    final_position F initial_rotation r1 r2 reflect_x = "negative (x+y)-axis equally" :=
by 
  sorry

end letter_F_final_position_l340_340457


namespace parallel_vectors_l340_340270

theorem parallel_vectors (x : ℝ) (a b : ℝ × ℝ)
  (ha : a = (x, 1))
  (hb : b = (-1, 3))
  (h_parallel : ∃ k : ℝ, a = (k * (-1), k * 3)) :
  x = -1/3 :=
by
  rw [ha, hb] at h_parallel
  sorry

end parallel_vectors_l340_340270


namespace inequality_positive_numbers_l340_340427

theorem inequality_positive_numbers (x y : ℝ) 
  (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy : x < y) : 
  x + real.sqrt (y^2 + 2) < y + real.sqrt (x^2 + 2) :=
sorry

end inequality_positive_numbers_l340_340427


namespace remainder_of_2n_div_7_l340_340664

theorem remainder_of_2n_div_7 (n : ℤ) (k : ℤ) (h : n = 7 * k + 2) : (2 * n) % 7 = 4 :=
by
  sorry

end remainder_of_2n_div_7_l340_340664


namespace problem_a_problem_b_l340_340237

-- Definitions

def points_on_unit_circle (P : ℕ → ℂ) : Prop :=
  ∀ i, P i = complex.exp (2 * real.pi * complex.I * i / n) -- P_i are on the unit circle

def sum_MP_k_fixed (P : ℕ → ℂ) (k : ℕ) : Prop :=
  ∀ M : ℂ, M = complex.exp (2 * real.pi * complex.I * θ) →  -- M is on the unit circle
    (∑ i in finset.range n, complex.abs (M - P i) ^ k) = fixed_val

-- Part (a) theorem

theorem problem_a (P : ℕ → ℂ) (n : ℕ) (k : ℕ := 2018) :
  points_on_unit_circle P →
  (∃ (fixed_val : ℝ), sum_MP_k_fixed P k) →
  n > 1009 :=
sorry

-- Part (b) theorem

theorem problem_b (P : ℕ → ℂ) (n : ℕ) (k : ℕ := 2019) :
  points_on_unit_circle P →
  (∃ (fixed_val : ℝ), sum_MP_k_fixed P k) →
  false :=
sorry

end problem_a_problem_b_l340_340237


namespace side_length_eq_l340_340751

namespace EquilateralTriangle

variables (A B C P Q R S : Type) [HasVSub Type P] [MetricSpace P]
variables [HasDist P] [HasEquilateralTriangle ABC] [InsideTriangle P ABC]
variables [Perpendicular PQ AB] [Perpendicular PR BC] [Perpendicular PS CA]
variables [Distance PQ 2] [Distance PR 3] [Distance PS 4]

theorem side_length_eq : side_length ABC = 6 * √3 :=
sorry
end EquilateralTriangle

end side_length_eq_l340_340751


namespace no_nonzero_solutions_l340_340924

theorem no_nonzero_solutions (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x^2 + x = y^2 - y) ∧ (y^2 + y = z^2 - z) ∧ (z^2 + z = x^2 - x) → false :=
by
  sorry

end no_nonzero_solutions_l340_340924


namespace odd_prime_divides_x_squared_plus_one_l340_340054

theorem odd_prime_divides_x_squared_plus_one (x p : ℤ) (hp : Nat.Prime p) (hodd : p % 2 = 1) (hdiv : p ∣ (x * x + 1)) : ∃ k : ℤ, p = 4 * k + 1 := 
sorry

end odd_prime_divides_x_squared_plus_one_l340_340054


namespace area_of_region_l340_340838

theorem area_of_region (x y : ℝ) : |4 * x - 24| + |3 * y + 10| ≤ 6 → ∃ A : ℝ, A = 12 :=
by
  sorry

end area_of_region_l340_340838


namespace volume_of_intersection_l340_340853

noncomputable section

def region1 (x y z : ℝ) : Prop := abs x + abs y + abs z ≤ 2
def region2 (x y z : ℝ) : Prop := abs x + abs y + abs (z - 2) ≤ 2

/-- The volume of the intersection of region1 and region2 is 2/3 -/
theorem volume_of_intersection : 
  (∃ W : Set (ℝ × ℝ × ℝ), 
    (∀ x y z, (x, y, z) ∈ W ↔ region1 x y z ∧ region2 x y z) 
    ∧ measurable_set W 
    ∧ volume W = (2 / 3)) := sorry

end volume_of_intersection_l340_340853


namespace overlap_triangles_area_l340_340108

-- Define the properties of the 45-45-90 triangle
def hypotenuse : ℝ := 8
def leg : ℝ := hypotenuse / Real.sqrt 2

-- Define the overlap property
def overlap_hypotenuse : ℝ := hypotenuse / 2
def overlap_leg : ℝ := overlap_hypotenuse / Real.sqrt 2

-- Define the calculation of the area of the overlapping region
def overlapping_area : ℝ := 1/2 * overlap_leg * overlap_leg

-- Statement to prove
theorem overlap_triangles_area : overlapping_area = 4 := 
by
  sorry

end overlap_triangles_area_l340_340108


namespace slope_of_tangent_line_at_point_is_negative_reciprocal_of_slope_of_radius_l340_340119

def slope (p1 p2 : ℝ × ℝ) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)

theorem slope_of_tangent_line_at_point_is_negative_reciprocal_of_slope_of_radius :
  ∀ (point center : ℝ × ℝ), center = (2,1) ∧ point = (6,3) →
  let radius_slope := slope center point in
  let tangent_slope := - (1 / radius_slope) in
  tangent_slope = -2 :=
by intros point center h
   let radius_slope := slope center point
   let tangent_slope := - (1 / radius_slope)
   cases h
   sorry

end slope_of_tangent_line_at_point_is_negative_reciprocal_of_slope_of_radius_l340_340119


namespace runner_speed_l340_340531

-- Given conditions
def track_length (radius : ℝ) : ℝ :=
  200 + 2 * radius * Real.pi

def time_difference (v : ℝ) : ℝ :=
  (track_length 52 - track_length 50) / v

-- To prove the runner's average speed
theorem runner_speed : 
  time_difference (Real.pi / 10) = 40 :=
by
  sorry

end runner_speed_l340_340531


namespace range_of_y_over_x_l340_340600

-- Definitions based on the conditions
def f (x : ℝ) : ℝ := sorry  -- Undefined, but we assume it exists
axiom condition_1 (x y : ℝ) : f (x^2 - 2*x) ≤ -f (2*y - y^2)
axiom condition_2 (x : ℝ) : f (x - 1) = f (2 - (x - 1))

-- The theorem statement
theorem range_of_y_over_x (x y : ℝ) (h1 : 1 ≤ x) (h2 : x ≤ 4) :
  ∃ (r : ℝ), r ∈ set.Icc (-1 / 2) 1 ∧ r = y / x :=
sorry

end range_of_y_over_x_l340_340600


namespace find_k_circle_radius_l340_340260

theorem find_k_circle_radius (k : ℝ) :
  (∀ x y : ℝ, (x^2 + 8 * x + y^2 + 4 * y - k = 0) → ((x + 4)^2 + (y + 2)^2 = 7^2)) → k = 29 :=
sorry

end find_k_circle_radius_l340_340260


namespace maximum_integer_solutions_l340_340895

-- Definition of a self-contained polynomial
def selfContained (p : ℤ → ℤ) : Prop :=
  ∀ x, ∃ c : ℤ, p x = c*x

-- The statement of the theorem
theorem maximum_integer_solutions (p : ℤ → ℤ) (h_self_contained : selfContained p) (h_p_50 : p 50 = 50) :
  ∃ k, (∀ x, p x = x^3 → x ∈ k) ∧ k.card = 6 :=
sorry

end maximum_integer_solutions_l340_340895


namespace area_of_region_l340_340952

noncomputable def enclosed_area : ℝ :=
  let circle_equation := λ (x y : ℝ), x ^ 2 + y ^ 2 - 4 * x + 6 * y = -9
  in 4 * Real.pi

theorem area_of_region : enclosed_area = 4 * Real.pi :=
  sorry

end area_of_region_l340_340952


namespace problem_B_problem_C_problem_D_l340_340287

theorem problem_B (a : ℕ → ℕ) (b : ℕ → ℕ) (a_def : ∀ n, a n = n) 
  (b_def : ∀ n, b n = a (n + 1) + (-1)^n * a n) :
  (finset.range 16).sum b = 160 := sorry

theorem problem_C (a : ℕ → ℕ) (b : ℕ → ℕ) (b_def : ∀ n, b n = n) 
  (a_def : ∀ n, b n = a (n + 1) + (-1)^n * a n) :
  (finset.range 16).sum a = 72 := sorry

theorem problem_D (a : ℕ → ℕ) (b : ℕ → ℕ) (b_def : ∀ n, b n = n) 
  (a_def : ∀ n, b n = a (n + 1) + (-1)^n * a n) (even_terms : ℕ) 
  (even_def : even even_terms):
  (finset.filter even (finset.range even_terms)).sum a >
  (finset.filter (λ n, ¬ even n) (finset.range even_terms)).sum a := sorry

end problem_B_problem_C_problem_D_l340_340287


namespace no_divisors_in_range_l340_340956

theorem no_divisors_in_range : ¬ ∃ n : ℕ, 80 < n ∧ n < 90 ∧ n ∣ (3^40 - 1) :=
by sorry

end no_divisors_in_range_l340_340956


namespace gain_percent_calculation_l340_340339

noncomputable def gain_percent (C S : ℝ) : ℝ :=
  (S - C) / C * 100

theorem gain_percent_calculation (C S : ℝ) (h : 50 * C = 46 * S) : 
  gain_percent C S = 100 / 11.5 :=
by
  sorry

end gain_percent_calculation_l340_340339


namespace max_soap_boxes_in_carton_l340_340883

noncomputable def V_carton : ℕ := 25 * 42 * 60
noncomputable def V_soap_box : ℕ := 7 * 6 * 5
noncomputable def max_soap_boxes : ℕ := V_carton / V_soap_box 

theorem max_soap_boxes_in_carton : max_soap_boxes = 300 :=
by
  unfold max_soap_boxes
  unfold V_carton
  unfold V_soap_box
  norm_num
  sorry

end max_soap_boxes_in_carton_l340_340883


namespace length_BP_length_QT_l340_340682

-- Let's define the given conditions
variables {A B C D P Q T S R : Type}
variables [rect : Rectangle A B C D] -- here A, B, C, D define the rectangle
variables [point_on_BC : PointOnLineSegment P B C] -- P is on the segment BC
variables [angle_APD : RightAngle A P D] -- ∠APD = 90°
variables [perpendicular_TS_BC : Perpendicular T S B C] -- TS ⊥ BC
variables [length_BP_PT : BP = PT] -- BP = PT
variables [line_intersect : LineIntersect PD TS Q] -- PD intersects TS at Q
variables [line_through_RA : LineThrough R A Q] -- RA passes through Q
variables [triangle_PQA : Triangle P Q A] -- ΔPQA
variables [length_PA : PA = 15]
variables [length_AQ : AQ = 20]
variables [length_QP : QP = 25]

-- Required theorems to be proved (in two parts for clarity)
theorem length_BP : BP = 12 := by sorry
theorem length_QT : QT = Real.sqrt 481 := by sorry

end length_BP_length_QT_l340_340682


namespace plane_rectangle_equality_space_rectangle_equality_prism_equality_l340_340144

-- Part 1: Rectangle in the plane
theorem plane_rectangle_equality (a b x y : ℝ) :
  let A := (-a, -b)
  let B := (a, -b)
  let C := (a, b)
  let D := (-a, b)
  let P := (x, y)
  (dist P A) ^ 2 + (dist P C) ^ 2 = (dist P B) ^ 2 + (dist P D) ^ 2 :=
sorry

-- Part 2: Rectangle in 3D space
theorem space_rectangle_equality (a b x y z : ℝ) :
  let A := (-a, -b, 0)
  let B := (a, -b, 0)
  let C := (a, b, 0)
  let D := (-a, b, 0)
  let P := (x, y, z)
  (dist3D P A) ^ 2 + (dist3D P C) ^ 2 = (dist3D P B) ^ 2 + (dist3D P D) ^ 2 :=
sorry

-- Part 3: Rectangular prism
theorem prism_equality (a b c x y z : ℝ) :
  let A := (-a, -b, 0)
  let B := (a, -b, 0)
  let C := (a, b, 0)
  let D := (-a, b, 0)
  let A1 := (-a, -b, c)
  let B1 := (a, -b, c)
  let C1 := (a, b, c)
  let D1 := (-a, b, c)
  let P := (x, y, z)
  (dist3D P A) ^ 2 + (dist3D P C) ^ 2 + (dist3D P B1) ^ 2 + (dist3D P D1) ^ 2 =
  (dist3D P B) ^ 2 + (dist3D P D) ^ 2 + (dist3D P A1) ^ 2 + (dist3D P C1) ^ 2 :=
sorry

-- Distance function definitions
noncomputable def dist {α : Type*} [RealInnerProductSpace α] (x y : α) : ℝ :=
  Real.sqrt (RealInnerProductSpace.inner (x - y) (x - y))

noncomputable def dist3D {α : Type*} [RealInnerProductSpace α] (x y : α) : ℝ3 :=
  Real.sqrt (RealInnerProductSpace.inner (x - y) (x - y))

end plane_rectangle_equality_space_rectangle_equality_prism_equality_l340_340144


namespace balanced_placement_exists_l340_340881

-- Defining the problem conditions in Lean 4
def is_domino (d : (ℕ × ℕ) × (ℕ × ℕ)) : Prop :=
  (d.snd.1 = d.fst.1 + 1 ∧ d.snd.2 = d.fst.2) ∨ (d.snd.1 = d.fst.1 ∧ d.snd.2 = d.fst.2 + 1)

def covers (d : (ℕ × ℕ) × (ℕ × ℕ)) (sq : ℕ × ℕ) : Prop :=
  sq = d.fst ∨ sq = d.snd

def no_overlap (dominos : list ((ℕ × ℕ) × (ℕ × ℕ))) : Prop :=
  ∀ (d1 d2 : (ℕ × ℕ) × (ℕ × ℕ)), d1 ∈ dominos → d2 ∈ dominos → d1 ≠ d2 → 
  ∀ sq, ¬ (covers d1 sq ∧ covers d2 sq)

def balanced (dominos : list ((ℕ × ℕ) × (ℕ × ℕ))) (n : ℕ) (k : ℕ) : Prop := 
  ∀ r c, 1 ≤ r ∧ r ≤ n ∧ 1 ≤ c ∧ c ≤ n → 
  (∃ doms_r, doms_r = (list.filter (λ (d : (ℕ × ℕ) × (ℕ × ℕ)), covers d (r, _)) dominos).length) ∧ 
  (∃ doms_c, doms_c = (list.filter (λ (d : (ℕ × ℕ) × (ℕ × ℕ)), covers d (_, c)) dominos).length) ∧
  doms_r = k ∧ doms_c = k

theorem balanced_placement_exists (n : ℕ) (h₁ : n ≥ 3) :  ∃ (dominos : list ((ℕ × ℕ) × (ℕ × ℕ))), 
  (dominos.length = n * n / 2) ∧ no_overlap dominos ∧ 
  (∃ k, balanced dominos n k) := sorry

end balanced_placement_exists_l340_340881


namespace probability_sum_eight_l340_340880

def total_outcomes : ℕ := 36
def favorable_outcomes : ℕ := 5

theorem probability_sum_eight :
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 36 := by
  sorry

end probability_sum_eight_l340_340880


namespace locus_of_tangent_circle_is_hyperbola_l340_340156

theorem locus_of_tangent_circle_is_hyperbola :
  ∀ (P : ℝ × ℝ) (r : ℝ),
    (P.1 ^ 2 + P.2 ^ 2).sqrt = 1 + r ∧ ((P.1 - 4) ^ 2 + P.2 ^ 2).sqrt = 2 + r →
    ∃ (a b : ℝ), (P.1 - a) ^ 2 / b ^ 2 - (P.2 / a) ^ 2 / b ^ 2 = 1 :=
sorry

end locus_of_tangent_circle_is_hyperbola_l340_340156


namespace moth_can_reach_vertex_B_in_48_ways_l340_340889

def vertex := ℕ
def steps := ℕ

axiom vertices_of_cube : set vertex
axiom edges_of_cube : set (vertex × vertex)
axiom valid_vertex_A : vertex
axiom valid_vertex_B : vertex
axiom vertex_opposite : valid_vertex_B ∈ vertices_of_cube

-- Define the cube and its properties
def cube : ℕ := 8
def opposite_vertices := (valid_vertex_A, valid_vertex_B) ∈ edges_of_cube

-- Distance in terms of steps
def distance_in_steps (A B : vertex) : steps := 3

-- Question: How many distinct paths from A to B in ≤ 5 steps?
def count_paths_to_B_in_five_or_fewer_steps (A B : vertex) : ℕ := 48

-- Proof statement
theorem moth_can_reach_vertex_B_in_48_ways 
  (A B : vertex)
  (h1 : A = valid_vertex_A)
  (h2 : B = valid_vertex_B)
  (h3 : opposite_vertices)
  (h4 : distance_in_steps A B = 3) : 
  count_paths_to_B_in_five_or_fewer_steps A B = 48 :=
sorry

end moth_can_reach_vertex_B_in_48_ways_l340_340889


namespace sum_of_squares_mod_16_l340_340117

/-- Prove that the remainder when the sum of squares from 1 to 15 is divided by 16 is 8. -/
theorem sum_of_squares_mod_16 :
  (∑ n in finset.range 16, n^2) % 16 = 8 :=
sorry

end sum_of_squares_mod_16_l340_340117


namespace product_of_nonreal_roots_l340_340250

theorem product_of_nonreal_roots (p : Polynomial ℂ) (hp : p = Polynomial.C (-119) + Polynomial.monomial 4 (1 : ℂ) - Polynomial.monomial 3 (6 : ℂ) + Polynomial.monomial 2 (15 : ℂ) - Polynomial.monomial 1 (20 : ℂ)) :
  let nonreal_roots := {r : ℂ | Polynomial.root p r ∧ r.im ≠ 0} in
  nonreal_roots.prod (fun x => x) = 4 + complex.sqrt 103 :=
sorry

end product_of_nonreal_roots_l340_340250


namespace sin_increasing_interval_l340_340955

theorem sin_increasing_interval :
  ∀ (x : ℝ) (k : ℤ), 
    (2 * k * real.pi - real.pi / 2 ≤ 2 * x - real.pi / 4 ∧ 2 * x - real.pi / 4 ≤ 2 * k * real.pi + real.pi / 2) ↔ 
    (-real.pi / 8 + k * real.pi ≤ x ∧ x ≤ 3 * real.pi / 8 + k * real.pi) :=
by sorry

end sin_increasing_interval_l340_340955


namespace part_1_part_2_l340_340315

noncomputable def f (a b x : ℝ) : ℝ := x^3 + a * x^2 + b * x + 1

noncomputable def a_domain : set ℝ := {a | 3 < a}

noncomputable def b_of_a (a : ℝ) : ℝ := (2 * a^2) / 9 + 3 / a

theorem part_1 (a : ℝ) (ha : a > 3) : 
  let b := b_of_a a in
  b^2 > 3 * a :=
by sorry

theorem part_2 : set.Icc 3 6 ⊆ {a | ∃ b, b = b_of_a a ∧ a > 3 ∧ b^2 > 3 * a } :=
by sorry

end part_1_part_2_l340_340315


namespace lines_perpendicular_slope_l340_340636

theorem lines_perpendicular_slope (k : ℝ) :
  (∀ (x : ℝ), k * 2 = -1) → k = (-1:ℝ)/2 :=
by
  sorry

end lines_perpendicular_slope_l340_340636


namespace solve_equation_l340_340065

theorem solve_equation (x : ℤ) (h1 : x ≠ 2) : x - 8 / (x - 2) = 5 - 8 / (x - 2) → x = 5 := by
  sorry

end solve_equation_l340_340065


namespace isosceles_triangle_congruent_side_length_l340_340794

theorem isosceles_triangle_congruent_side_length (BC : ℝ) (BM : ℝ) :
  BC = 4 * Real.sqrt 2 → BM = 5 → ∃ (AB : ℝ), AB = Real.sqrt 34 :=
by
  -- sorry is used here to indicate proof is not provided, but the statement is expected to build successfully.
  sorry

end isosceles_triangle_congruent_side_length_l340_340794


namespace probability_both_boys_probability_exactly_one_girl_probability_at_least_one_girl_l340_340782

-- Definitions for combinatorics
def combination (n k : ℕ) : ℕ := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

-- Given conditions
def boys : ℕ := 4
def girls : ℕ := 3
def total_people : ℕ := boys + girls
def selected_people : ℕ := 2

-- Computations relevant to the questions
def total_combinations := combination total_people selected_people
def combinations_two_boys := combination boys selected_people
def combinations_one_boy_one_girl := combination boys 1 * combination girls 1

-- Proof statements
theorem probability_both_boys : (combinations_two_boys : ℚ) / total_combinations = 2 / 7 := by
  sorry

theorem probability_exactly_one_girl : (combinations_one_boy_one_girl : ℚ) / total_combinations = 4 / 7 := by
  sorry

theorem probability_at_least_one_girl : 1 - (combinations_two_boys : ℚ) / total_combinations = 5 / 7 := by
  sorry

end probability_both_boys_probability_exactly_one_girl_probability_at_least_one_girl_l340_340782


namespace mitch_total_scoops_l340_340738

theorem mitch_total_scoops :
  (3 : ℝ) / (1/3 : ℝ) + (2 : ℝ) / (1/3 : ℝ) = 15 :=
by
  sorry

end mitch_total_scoops_l340_340738


namespace geometric_progression_first_term_and_ratio_l340_340821

theorem geometric_progression_first_term_and_ratio (
  b_1 q : ℝ
) :
  b_1 * (1 + q + q^2) = 21 →
  b_1^2 * (1 + q^2 + q^4) = 189 →
  (b_1 = 12 ∧ q = 1/2) ∨ (b_1 = 3 ∧ q = 2) :=
by
  intros hsum hsumsq
  sorry

end geometric_progression_first_term_and_ratio_l340_340821


namespace rectangle_area_l340_340530

theorem rectangle_area (x y : ℝ) (L W : ℝ) (h_diagonal : (L ^ 2 + W ^ 2) ^ (1 / 2) = x + y) (h_ratio : L / W = 3 / 2) : 
  L * W = (6 * (x + y) ^ 2) / 13 := 
sorry

end rectangle_area_l340_340530


namespace triangle_side_ratio_l340_340185

variables (a b c : ℝ)
-- Conditions: sides of triangle where a ≤ b ≤ c
axiom triangle_sides : a ≤ b ∧ b ≤ c

-- Definition of medians using Apollonius's theorem
def median (x y z : ℝ) : ℝ := sqrt ((2 * y^2 + 2 * z^2 - x^2) / 4)

def m_a := median a b c
def m_b := median b a c
def m_c := median c a b

-- Statement of the problem
theorem triangle_side_ratio (h : 2 * b^2 = a^2 + c^2) : 2 * b^2 = a^2 + c^2 := 
sorry

end triangle_side_ratio_l340_340185


namespace factor_expression_l340_340964

variable (b : ℝ)

theorem factor_expression : 221 * b * b + 17 * b = 17 * b * (13 * b + 1) :=
by
  sorry

end factor_expression_l340_340964


namespace area_of_expanded_quad_l340_340417

variables {V : Type*} [inner_product_space ℝ V]
variables (A B C D A' B' C' D' : V)

def is_extension (P Q R : V) : Prop :=
  P - Q = R - P

def area_of_quad (A B C D : V) : ℝ :=
  sorry -- Define the area calculation for a quadrilateral

theorem area_of_expanded_quad
  (h1 : is_extension B B' A B)
  (h2 : is_extension C C' B C)
  (h3 : is_extension D D' C D)
  (h4 : is_extension A A' D A) :
  area_of_quad A' B' C' D' = 5 * area_of_quad A B C D :=
begin
  sorry
end

end area_of_expanded_quad_l340_340417


namespace equilateral_triangle_side_length_l340_340760

theorem equilateral_triangle_side_length 
  {P Q R S : Point} 
  {A B C : Triangle}
  (h₁ : is_inside P A B C)
  (h₂ : is_perpendicular P Q A B)
  (h₃ : is_perpendicular P R B C)
  (h₄ : is_perpendicular P S C A)
  (h₅ : distance P Q = 2)
  (h₆ : distance P R = 3)
  (h₇ : distance P S = 4)
  : side_length A B C = 6 * sqrt 3 :=
by 
  sorry

end equilateral_triangle_side_length_l340_340760


namespace num_balls_picked_l340_340516

-- Definitions based on the conditions
def numRedBalls : ℕ := 4
def numBlueBalls : ℕ := 3
def numGreenBalls : ℕ := 2
def totalBalls : ℕ := numRedBalls + numBlueBalls + numGreenBalls
def probFirstRed : ℚ := numRedBalls / totalBalls
def probSecondRed : ℚ := (numRedBalls - 1) / (totalBalls - 1)

-- Theorem stating the problem
theorem num_balls_picked :
  probFirstRed * probSecondRed = 1 / 6 → 
  (∃ (n : ℕ), n = 2) :=
by 
  sorry

end num_balls_picked_l340_340516


namespace range_of_a_minus_abs_b_l340_340512

theorem range_of_a_minus_abs_b (a b : ℝ) (h1 : 1 < a ∧ a < 8) (h2 : -4 < b ∧ b < 2) : 
  -3 < a - |b| ∧ a - |b| < 8 :=
sorry

end range_of_a_minus_abs_b_l340_340512


namespace max_roots_poly_interval_l340_340280

noncomputable theory

open Polynomial

def max_roots_in_interval (P : Polynomial ℤ) : ℝ → ℝ → ℕ :=
  λ a b, (map (algebraMap ℤ ℝ) P).roots.countInInterval (Ioo a b)
  
theorem max_roots_poly_interval :
  ∀ P : Polynomial ℤ,
  degree P = 2022 ∧ leadingCoeff P = 1 →
  max_roots_in_interval P 0 1 ≤ 2021 :=
by sorry

end max_roots_poly_interval_l340_340280


namespace hours_per_day_rented_l340_340371

theorem hours_per_day_rented 
  (sailboat_cost : ℕ) (ski_boat_cost_per_hour : ℕ) 
  (extra_cost : ℕ) (days : ℕ)
  (ken_cost : ℕ) (aldrich_cost : ℕ) : 
  sailboat_cost = 60 → 
  ski_boat_cost_per_hour = 80 → 
  extra_cost = 120 → 
  days = 2 → 
  ken_cost = sailboat_cost * days → 
  aldrich_cost = ski_boat_cost_per_hour * h * days → 
  aldrich_cost = ken_cost + extra_cost → 
  (h : ℚ), 
  h = 1.5 := 
by 
  sorry

end hours_per_day_rented_l340_340371


namespace quadrilateral_is_rectangle_l340_340442

variable {a b c d : ℝ} -- lengths of the sides in sequential order
variable {A B C D : ℝ} -- angles of the quadrilateral
variable {S : ℝ} -- area of the quadrilateral

-- Given the Egyptian formula for the area: S = (a + c) * (b + d) / 4
def egyptian_area_formula (a c b d : ℝ) : ℝ := (a + c) * (b + d) / 4

-- To prove the quadrilateral must be a rectangle
theorem quadrilateral_is_rectangle (h : egyptian_area_formula a c b d = S)
    (h_quad : S = (1 / 2) * (a * b * sin B + c * d * sin D) + (1 / 2) * (a * d * sin A + b * c * sin C))
    (h_eq : egyptian_area_formula a c b d = S) :
    (A = 90 ∧ B = 90 ∧ C = 90 ∧ D = 90) ↔ (a = b ∧ a = c ∧ a = d) := sorry

end quadrilateral_is_rectangle_l340_340442


namespace first_player_wins_l340_340634

-- Define the polynomial with placeholders
def P (X : ℤ) (a3 a2 a1 a0 : ℤ) : ℤ :=
  X^4 + a3 * X^3 + a2 * X^2 + a1 * X + a0

-- The statement that the first player can always win
theorem first_player_wins :
  ∀ (a3 a2 a1 a0 : ℤ),
    (a0 ≠ 0) → (a1 ≠ 0) → (a2 ≠ 0) → (a3 ≠ 0) →
    ∃ (strategy : ℕ → ℤ),
      (∀ n, strategy n ≠ 0) ∧
      ¬ ∃ (x y : ℤ), x ≠ y ∧ P x (strategy 0) (strategy 1) (strategy 2) (strategy 3) = 0 ∧ P y (strategy 0) (strategy 1) (strategy 2) (strategy 3) = 0 :=
by
  sorry

end first_player_wins_l340_340634


namespace picasso_together_probability_correct_l340_340439

noncomputable def probability_of_picasso_together : ℚ :=
  let total_arrangements := (12!).nnreal,
      picasso_together_arrangements := (9! * 4!).nnreal
  in (picasso_together_arrangements / total_arrangements)

theorem picasso_together_probability_correct :
  probability_of_picasso_together = 1 / 55 :=
by
   sorry

end picasso_together_probability_correct_l340_340439


namespace period_sin_x_plus_2cos_x_l340_340842

def period_of_sin_x_plus_2_cos_x : Prop :=
  ∃ (T : ℝ), T > 0 ∧ ∀ x : ℝ, sin x + 2 * cos x = sin (x + T)

theorem period_sin_x_plus_2cos_x : period_of_sin_x_plus_2_cos_x :=
by
  sorry

end period_sin_x_plus_2cos_x_l340_340842


namespace ordered_pair_solution_l340_340788

theorem ordered_pair_solution :
  ∃ x y : ℚ, 3 * x - 4 * y = -2 ∧ 4 * x + 5 * y = 23 ∧ x = 82 / 31 ∧ y = 77 / 31 :=
by
  existsi 82 / 31
  existsi 77 / 31
  split
  { sorry }
  split
  { sorry }

end ordered_pair_solution_l340_340788


namespace tax_amount_self_employed_l340_340099

noncomputable def gross_income : ℝ := 350000.00
noncomputable def tax_rate : ℝ := 0.06

theorem tax_amount_self_employed :
  gross_income * tax_rate = 21000.00 :=
by
  sorry

end tax_amount_self_employed_l340_340099


namespace tangent_line_b_value_l340_340668

theorem tangent_line_b_value (b : ℝ) :
  (∃ x₀ : ℝ, y = 2*x₀ + b ∧ y = e^x₀ + x₀) ∧ 
  derivative y = e^x₀ + 1 = (b = 1) := 
by
  sorry

end tangent_line_b_value_l340_340668


namespace race_time_A_l340_340347

theorem race_time_A (v_A v_B : ℝ) (t_A t_B : ℝ) (hA_time_eq : v_A = 1000 / t_A)
  (hB_time_eq : v_B = 960 / t_B) (hA_beats_B_40m : 1000 / v_A = 960 / v_B)
  (hA_beats_B_8s : t_B = t_A + 8) : t_A = 200 := 
  sorry

end race_time_A_l340_340347


namespace correct_judgement_l340_340886

/-- Define the suspects and their statements --/
def suspect_A : Prop := C_committed
def suspect_B : Prop := D_culprit
def suspect_C : Prop := if C_committed then D_mastermind else true
def suspect_D : Prop := ¬D_committed

-- Define the assumptions
axiom one_statement_is_false : ∃ (s : Prop), s ∈ [suspect_A, suspect_B, suspect_C, suspect_D] ∧ ¬s
axiom D_lied : ¬suspect_D

theorem correct_judgement : D_lied ∧ 
  (C_committed ∧ D_committed) :=
by
  apply and.intro
  {
    exact D_lied
  }
  {
    have hd : D_committed := not_not.mp D_lied
    have hc : C_committed := sorry -- Prove that C must have committed the crime given the statements.
    exact and.intro hc hd
  }

end correct_judgement_l340_340886


namespace enclosed_area_of_curve_l340_340075

def side_length_of_hexagon  : ℝ := 3
def num_arcs                 : ℕ := 9
def arc_length               : ℝ := (5 * Real.pi) / 6
def expected_area            : ℝ := (27 * Real.sqrt 3) / 2 + (1125 * Real.pi ^ 2) / 96

theorem enclosed_area_of_curve :
  let r := 5 / 4 in
  let area_hexagon := (3 * Real.sqrt 3 / 2) * side_length_of_hexagon ^ 2 in
  let area_sector := (arc_length / (2 * Real.pi * r)) * (Real.pi * r ^ 2) in
  let total_area_of_sectors := num_arcs * area_sector in
  let net_area := area_hexagon + total_area_of_sectors in
  net_area = expected_area :=
sorry

end enclosed_area_of_curve_l340_340075


namespace successful_N_l340_340611

-- Definitions of colors
inductive Color
| red : Color
| white : Color
| blue : Color

-- Array of cubes
def Circle (N : Nat) := Fin N → Color

-- Function to simulate the replacement step for the robot
def replace (c1 c2 : Color) : Color :=
match c1, c2 with
| Color.red, Color.red => Color.red
| Color.white, Color.white => Color.white
| Color.blue, Color.blue => Color.blue
| Color.red, Color.white => Color.blue
| Color.white, Color.red => Color.blue
| Color.red, Color.blue => Color.white
| Color.blue, Color.red => Color.white
| Color.white, Color.blue => Color.red
| Color.blue, Color.white => Color.red

-- Function to simulate the process
noncomputable def finalColor (circle : Circle N) : Color := sorry

-- Define what makes an arrangement "good"
def good (N : Nat) :=
∀ (circle : Circle N) (start1 start2 : Fin N), finalColor circle start1 = finalColor circle start2

-- Main theorem statement
theorem successful_N (N : Nat) : (∃ k : Nat, N = 2 ^ k) ↔ good N := sorry

end successful_N_l340_340611


namespace find_a2008_infinite_constant_subsequences_l340_340358

variable {a : ℕ → ℤ}

-- Given conditions and initial values
def initial_conditions : Prop :=
  (a 15 = 2) ∧ (a 16 = -1) ∧ (∀ n, a (n + 2) = (a (n + 1) - a n).natAbs)

-- Prove that a_{2008} = 1
theorem find_a2008 (h : initial_conditions) : a 2008 = 1 := sorry

-- Prove that the sequence can be partitioned into infinitely many terms forming two distinct constant subsequences
theorem infinite_constant_subsequences (h : initial_conditions) :
  ∃ f1 f2 : ℕ → ℤ, (f1 ≠ f2) ∧ (∀ n, (a (3 * n) = f1 n ∧ a (3 * n + 1) = f1 n) ∨ (a (3 * n) = f2 n ∧ a (3 * n + 1) = f2 n)) := sorry

end find_a2008_infinite_constant_subsequences_l340_340358


namespace self_employed_tax_amount_l340_340097

-- Definitions for conditions
def gross_income : ℝ := 350000.0

def tax_rate_self_employed : ℝ := 0.06

-- Statement asserting the tax amount for self-employed individuals given the conditions
theorem self_employed_tax_amount :
  gross_income * tax_rate_self_employed = 21000.0 := by
  sorry

end self_employed_tax_amount_l340_340097


namespace tax_amount_self_employed_l340_340100

noncomputable def gross_income : ℝ := 350000.00
noncomputable def tax_rate : ℝ := 0.06

theorem tax_amount_self_employed :
  gross_income * tax_rate = 21000.00 :=
by
  sorry

end tax_amount_self_employed_l340_340100


namespace find_n_l340_340451

def sequence_term (n : ℕ) : ℚ := 1 / (n * (n + 1))

theorem find_n (n : ℕ) (h : ∑ i in Finset.range n, sequence_term (i + 1) = 10 / 11) : n = 10 :=
by sorry

end find_n_l340_340451


namespace distance_from_point_to_line_l340_340678

/-- Definition of point M in polar coordinates. --/
def polar_point_M := (2 : ℝ, real.pi / 3)

/-- Definition of the line in polar coordinates. --/
def polar_line (ρ θ : ℝ) := ρ * real.sin (θ + real.pi / 4) = real.sqrt 2 / 2

/-- Distance from the point M to the line is as given. --/
theorem distance_from_point_to_line :
  let (r, θ) := polar_point_M in
  let M_x := r * real.cos θ in
  let M_y := r * real.sin θ in
  let A := 1 in
  let B := 1 in
  let C := -1 in
  let d := (abs (A * M_x + B * M_y + C)) / (sqrt (A^2 + B^2)) in
  d = real.sqrt 6 / 2 :=
by
  sorry

end distance_from_point_to_line_l340_340678


namespace abe_proof_l340_340423

theorem abe_proof (A B C D E : Point) (h1 : collinear [A, B, C, D])
  (h2 : dist A B = dist C D) (h3 : dist B C = 8)
  (h4 : dist B E = 12) (h5 : dist C E = 12)
  (h6 : 3 * (dist B E + dist C E + dist B C) = dist A E + dist D E + dist A D + dist A E + dist D E) :
  dist A B = 18 :=
by
  sorry

end abe_proof_l340_340423


namespace students_in_line_l340_340438

theorem students_in_line (T N : ℕ) (hT : T = 1) (h_btw : N = T + 4) (h_behind: ∃ k, k = 8) : T + (N - T) + 1 + 8 = 13 :=
by
  sorry

end students_in_line_l340_340438


namespace plane_hover_central_time_l340_340545

theorem plane_hover_central_time (x : ℕ) (h1 : 3 + x + 2 + 5 + (x + 2) + 4 = 24) : x = 4 := by
  sorry

end plane_hover_central_time_l340_340545


namespace goods_train_length_l340_340522

noncomputable def length_of_goods_train
  (v : ℝ) -- speed in km/h
  (p : ℝ) -- platform length in meters
  (t : ℝ) -- time in seconds
  (conversion_factor : ℝ) -- conversion factor from km/h to m/s
  (expected_length : ℝ) -- expected length of the train in meters
  : Prop :=
  let v_m_s := v * conversion_factor in
  let distance_covered := v_m_s * t in
  let length_of_train := distance_covered - p in
  length_of_train = expected_length

-- Given conditions
theorem goods_train_length : 
  length_of_goods_train 72 250 30 (5/18) 350 :=
by
  sorry

end goods_train_length_l340_340522


namespace count_9digit_numbers_l340_340328

def num_9digit_numbers (no_zero_first: Prop) : ℕ :=
  if no_zero_first then 9 * 10^8 else 0

theorem count_9digit_numbers : 
  num_9digit_numbers (1 ≤ 9) = 900000000 := 
by {
  sorry
}

end count_9digit_numbers_l340_340328


namespace total_distance_is_66_gasoline_cost_is_52_8_l340_340572

-- Definition of the itinerary distances
def distances : List Int := [-15, 4, -5, 10, -12, 5, 8, -7]

-- Definition of the gasoline consumption and price
def gasoline_consumption_per_100km : Float := 10
def gasoline_price_per_liter : Float := 8

-- Total distance calculated from the sum of absolute values of itinerary
def total_distance : Int := List.sum (List.map Int.natAbs distances)

-- Total gasoline cost based on the total distance
def gasoline_cost : Float := (total_distance.toFloat / 100) * gasoline_consumption_per_100km * gasoline_price_per_liter

-- Proof that the total distance driven is 66 kilometers
theorem total_distance_is_66 : total_distance = 66 := by
  sorry

-- Proof that the gasoline cost is 52.8 yuan
theorem gasoline_cost_is_52_8 : gasoline_cost = 52.8 := by
  sorry

end total_distance_is_66_gasoline_cost_is_52_8_l340_340572


namespace solve_equation_l340_340864

noncomputable def sum_of_fractions (x : ℝ) : ℝ :=
  (List.range 10).foldr (λ n acc, (acc * (x + n + 1) + (10 - n)) / (x + n + 1)) 0

theorem solve_equation : {x : ℝ // sum_of_fractions x = 11} = -1 / 11 :=
by
  sorry

end solve_equation_l340_340864


namespace sales_tax_difference_l340_340548

theorem sales_tax_difference :
  let price : ℝ := 50
  let tax_rate1 : ℝ := 0.075
  let tax_rate2 : ℝ := 0.07
  (price * tax_rate1) - (price * tax_rate2) = 0.25 := by
  sorry

end sales_tax_difference_l340_340548


namespace sum_of_vectors_zero_l340_340056

-- Define Regular Polyhedron (Dummy definition for structure)
structure RegularPolyhedron (n : ℕ) :=
  (O : Point)
  (vertices : fin n → Point)
  (is_center : ∀ i, distance O (vertices i) = distance O (vertices 0))
  (is_symmetric : True) -- Placeholder for symmetry property.

-- Define Point for simplicity
structure Point :=
  (x y z : ℝ)

instance : Add Point := ⟨λ p q, Point.mk (p.x + q.x) (p.y + q.y) (p.z + q.z)⟩
instance : Zero Point := ⟨Point.mk 0 0 0⟩

-- The sum of the vertices of the regular polyhedron
noncomputable def sum_vectors {n : ℕ} (P : RegularPolyhedron n) : Point := ∑ i, P.vertices i

theorem sum_of_vectors_zero {n : ℕ} (P : RegularPolyhedron n) : sum_vectors P = 0 :=
  by
    sorry

end sum_of_vectors_zero_l340_340056


namespace find_freshmen_count_l340_340815

theorem find_freshmen_count
  (F S J R : ℕ)
  (h1 : F : S = 5 : 4)
  (h2 : S : J = 7 : 8)
  (h3 : J : R = 9 : 7)
  (total_students : F + S + J + R = 2158) :
  F = 630 :=
by 
  sorry

end find_freshmen_count_l340_340815


namespace find_angle_BAD_l340_340093

-- Define the given arc and tangent properties
def angle_BC : ℝ := 112
def ratio_BD_CD : ℝ := 7 / 9

-- Theorem to prove the angle BAD
theorem find_angle_BAD (α β : ℝ) (h1 : α + β = angle_BC) (h2 : α / β = ratio_BD_CD) : 
  ∠BAD = 31.5 :=
by
  sorry

end find_angle_BAD_l340_340093


namespace quadratic_properties_l340_340310

-- Conditions
def f (x : ℝ) : ℝ := -x^2 + 3*x + 2
def g (x : ℝ) : ℝ := x^2 - 2

-- Statement
theorem quadratic_properties :
  (∀ x : ℝ, ∃ a b c : ℝ, f(x) = a*x^2 + b*x + c) ∧
  ((∀ x, ∃ y, f(x) + g(x) = y) → (∀ x, f(-x) + g(-x) = -(f(x) + g(x)))) ∧
  (∃ h k ℝ, f(h) = 3*h + 2 ∧ f(k) = 3*k + 2 ∧ h = k) → 
  f x = -x^2 + 3*x + 2 ∧ 
  (∀ x, f(x) > g(x) → (frac 3 - sqrt 41 / 4 < x ∧ x < frac 3 + sqrt 41 / 4)) ∧ 
  (∃ m n : ℝ, m < n ∧ ∀ x ∈ set.Icc m n, f(x) ∈ set.Icc (2 * m) (2 * n)) :=
by
  sorry

end quadratic_properties_l340_340310


namespace Michael_made_97_dollars_l340_340037

def price_large : ℕ := 22
def price_medium : ℕ := 16
def price_small : ℕ := 7

def quantity_large : ℕ := 2
def quantity_medium : ℕ := 2
def quantity_small : ℕ := 3

def calculate_total_money (price_large price_medium price_small : ℕ) 
                           (quantity_large quantity_medium quantity_small : ℕ) : ℕ :=
  (price_large * quantity_large) + (price_medium * quantity_medium) + (price_small * quantity_small)

theorem Michael_made_97_dollars :
  calculate_total_money price_large price_medium price_small quantity_large quantity_medium quantity_small = 97 := 
by
  sorry

end Michael_made_97_dollars_l340_340037


namespace part_a_part_b_part_c_l340_340029

def S := {n : ℕ | ∃ m k : ℕ, m ≥ 2 ∧ k ≥ 2 ∧ n = m^k}

def f (n : ℕ) : ℕ := 
  {cardinality of distinct subsets from S summing to n}

theorem part_a : f 30 = 0 :=
  by sorry

theorem part_b (n : ℕ) (hn : n ≥ 31) : f n ≥ 1 :=
  by sorry

def T := {n : ℕ | f n = 3}

theorem part_c : (T ≠ ∅) ∧ (finite T) :=
  by sorry

end part_a_part_b_part_c_l340_340029


namespace equilateral_triangle_side_length_l340_340767

theorem equilateral_triangle_side_length (P Q R S A B C : Type)
  [Point P] [Point Q] [Point R] [Point S] [Point A] [Point B] [Point C] :
  (within_triangle P A B C) →
  orthogonal_projection P Q A B →
  orthogonal_projection P R B C →
  orthogonal_projection P S C A →
  distance P Q = 2 →
  distance P R = 3 →
  distance P S = 4 →
  distance A B = 6 * √3 :=
by
  sorry

end equilateral_triangle_side_length_l340_340767


namespace log2_sufficient_not_necessary_l340_340594

noncomputable def baseTwoLog (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem log2_sufficient_not_necessary (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (baseTwoLog a > baseTwoLog b) ↔ (a > b) :=
sorry

end log2_sufficient_not_necessary_l340_340594


namespace sum_f_inv_eq_94_l340_340801

open Real

noncomputable def f (x : ℝ) : ℝ :=
  if x < 5 then x - 3 else sqrt x

noncomputable def f_inv (y : ℝ) : ℝ :=
  if y < 2 then y + 3 else y * y

theorem sum_f_inv_eq_94 : ((Finset.range 13).sum (λ i, f_inv (i - 6))) = 94 :=
by
  sorry

end sum_f_inv_eq_94_l340_340801


namespace walking_time_l340_340643

-- Define the conditions as Lean definitions
def minutes_in_hour : Nat := 60

def work_hours : Nat := 6
def work_minutes := work_hours * minutes_in_hour
def sitting_interval : Nat := 90
def walking_time_per_interval : Nat := 10

-- State the main theorem
theorem walking_time (h1 : 10 * 90 = 600) (h2 : 10 * (work_hours * 60) / 90 = 40) : 
  work_minutes / sitting_interval * walking_time_per_interval = 40 :=
  sorry

end walking_time_l340_340643


namespace total_selling_price_correct_l340_340887

-- Definitions of parameters based on the given conditions
def cp_bicycle : ℝ := 1600
def cp_scooter : ℝ := 8000
def cp_motorcycle : ℝ := 15000
def loss_perc_bicycle : ℝ := 10 / 100
def loss_perc_scooter : ℝ := 5 / 100
def loss_perc_motorcycle : ℝ := 8 / 100
def discount_bicycle : ℝ := 2 / 100
def sales_tax_scooter : ℝ := 3 / 100
def commission_motorcycle : ℝ := 4 / 100

-- Proof Problem Statement
theorem total_selling_price_correct :
  let sp_bicycle := (cp_bicycle - loss_perc_bicycle * cp_bicycle) * (1 - discount_bicycle)
  let sp_scooter := (cp_scooter - loss_perc_scooter * cp_scooter) * (1 + sales_tax_scooter)
  let sp_motorcycle := (cp_motorcycle - loss_perc_motorcycle * cp_motorcycle) * (1 - commission_motorcycle)
  sp_bicycle + sp_scooter + sp_motorcycle = 23487.2 := 
by
  sorry

end total_selling_price_correct_l340_340887


namespace find_c_l340_340445

-- Define the given conditions as hypothesis
def parabola_vertex {x y : ℝ} (a : ℝ) := x = a * (y - 2.5)^2 - 3

def parabola_point (a : ℝ) := parabola_vertex a (-4) 5

-- Prove that the constant term c in the equation is -4
theorem find_c : ∃ c, ∀ (a : ℝ), parabola_vertex a (-3) 2.5 ∧ parabola_point a → c = -4 :=
by {
  intro a,
  use -4,
  intros h_vertex h_point,
  sorry
}

end find_c_l340_340445


namespace equilateral_triangle_side_length_l340_340754

open Classical

noncomputable section

variable {P Q R S : Type}

def is_perpendicular_feet (P Q R S : P) : Prop :=
  sorry -- Definition for Q, R, S being the feet of perpendiculars from P

structure EquilateralTriangle (A B C P Q R S : P) where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  AB : ℝ
  (h_perpendicular : is_perpendicular_feet P Q R S)
  (h_area_eq : ∀ (h₁ : PQ = 2) (h₂ : PR = 3) (h₃ : PS = 4), AB = 12 * Real.sqrt 3)

theorem equilateral_triangle_side_length {A B C P Q R S : P} 
    (h_eq_triangle : EquilateralTriangle A B C P Q R S) : h_eq_triangle.AB = 12 * Real.sqrt 3 :=
  by
    cases h_eq_triangle with
    | mk PQ PR PS AB h_perpendicular h_area_eq =>
        apply h_area_eq
        · exact rfl
        · exact rfl
        · exact rfl

end equilateral_triangle_side_length_l340_340754


namespace find_a_l340_340697

def f (x : ℝ) : ℝ := ∑ i in range 2001, |x - (i + 1)|

theorem find_a : ∃ a, (∀ x, f(x) = a → x = 1001) ∧ a = 1001000 :=
by {
  sorry
}

end find_a_l340_340697


namespace bacteria_growth_l340_340893

-- Define the original and current number of bacteria
def original_bacteria := 600
def current_bacteria := 8917

-- Define the increase in bacteria count
def additional_bacteria := 8317

-- Prove the statement
theorem bacteria_growth : current_bacteria - original_bacteria = additional_bacteria :=
by {
    -- Lean will require the proof here, so we use sorry for now 
    sorry
}

end bacteria_growth_l340_340893


namespace find_ratio_l340_340316

-- Definition of the function
def f (x : ℝ) (a b: ℝ) : ℝ := x^3 + a * x^2 + b * x - a^2 - 7 * a

-- Statement to be proved
theorem find_ratio (a b : ℝ) (h1: f 1 a b = 10) (h2 : (3 * 1^2 + 2 * a * 1 + b = 0)) : b = -a / 2 :=
by
  sorry

end find_ratio_l340_340316


namespace min_k_satisfying_l340_340285
open Nat

def a : ℕ → ℕ
| 1       := 1
| (2*n)   := a (2*n - 1) + 1
| (2*n+1) := 2 * a (2*n) + 1

def S : ℕ → ℕ
| 0     := 0
| (n+1) := S n + a (n+1)

theorem min_k_satisfying (S : ℕ → ℕ) (a : ℕ → ℕ) : 
  (∀k, (k > 0) → S k > 2018 ↔ k = 17) ∧
  (a 1 = 1) ∧ 
  (∀ n, a (2*n) = a (2*n-1) + 1) ∧ 
  (∀ n, a (2*n + 1) = 2 * a (2*n) + 1) := sorry

end min_k_satisfying_l340_340285


namespace equation_solutions_l340_340577

noncomputable def solve_equation : set ℂ :=
  {x | x^4 + 4*x^2 + 100 = 0}

theorem equation_solutions : solve_equation = 
  {3 + 10/3 * complex.I, 3 - 10/3 * complex.I, -3 + 10/3 * complex.I, -3 - 10/3 * complex.I} :=
by
  sorry

end equation_solutions_l340_340577


namespace sum_arithmetic_sequence_l340_340610

variable {α : Type} [linear_ordered_field α]

noncomputable def arithmetic_sequence (a2 a16 : α) (d : α) : ℕ → α
| 0     := a2 - d
| 1     := a2
| n + 2 := a2 + (n + 1) * d

theorem sum_arithmetic_sequence (a2 a16 : α) (d : α) (a : ℕ → α) 
  (h_seq : ∀ n : ℕ, a n = arithmetic_sequence a2 a16 d n) 
  (h_root : ∀ r ∈ [a2, a16], r^2 - 6 * r + 1 = 0) 
  (h_d : d = (a16 - a2) / 14) : 
  a 6 + a 7 + a 8 + a 9 + a 10 = 15 := 
  by 
    -- insert this proof here
    sorry

end sum_arithmetic_sequence_l340_340610


namespace cannot_sort_by_height_l340_340676

theorem cannot_sort_by_height (n : ℕ) (h : n = 1998)
    (swap_rule : ∀ i j : ℕ, 0 ≤ i < j < n → j = i + 2 ∨ i = j + 2):
  ¬ ∀ (heights : fin n → ℝ), ∃ (sorted_heights : fin n → ℝ), 
    (∀ i : fin (n-1), sorted_heights i ≤ sorted_heights (i+1)) ∧
    (∀ (i j : fin n) (hij : i < j), list.swap heights.to_list i.val j.val = 
    sorted_heights.to_list := sorry

end cannot_sort_by_height_l340_340676


namespace triple_nested_g_l340_340728

def g (x : ℝ) : ℝ :=
if x > 2 then x^3 else 2*x

theorem triple_nested_g : g(g(g(1))) = 64 := by
  sorry

end triple_nested_g_l340_340728


namespace area_quadrilateral_AEDC_l340_340694

theorem area_quadrilateral_AEDC
  (A B C D E P : Type*)
  [IsPoint A] [IsPoint B] [IsPoint C] [IsPoint D] [IsPoint E] [IsPoint P]
  (med_AD : IsMedian A D)
  (med_CE : IsMedian C E)
  (intersect_AD_CE : ∃ P, P = intersection AD CE)
  (PE_len : Length PE = 2)
  (PD_len : Length PD = 6)
  (DE_len : Length DE = 2 * sqrt 10) :
  Area (Quadrilateral A E D C) = 54 :=
sorry

end area_quadrilateral_AEDC_l340_340694


namespace first_discount_percentage_l340_340900

theorem first_discount_percentage
  (list_price : ℝ)
  (second_discount : ℝ)
  (third_discount : ℝ)
  (tax_rate : ℝ)
  (final_price : ℝ)
  (D1 : ℝ)
  (h_list_price : list_price = 150)
  (h_second_discount : second_discount = 12)
  (h_third_discount : third_discount = 5)
  (h_tax_rate : tax_rate = 10)
  (h_final_price : final_price = 105) :
  100 - 100 * (final_price / (list_price * (1 - D1 / 100) * (1 - second_discount / 100) * (1 - third_discount / 100) * (1 + tax_rate / 100))) = 24.24 :=
by
  sorry

end first_discount_percentage_l340_340900


namespace train_speed_on_time_l340_340129

theorem train_speed_on_time (v : ℕ) (t : ℕ) :
  (15 / v + 1 / 4 = 15 / 50) ∧ (t = 15) → v = 300 := by
  sorry

end train_speed_on_time_l340_340129


namespace sequence_an_l340_340094

theorem sequence_an (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h1 : a 1 = 1/6)
  (h2 : ∀ n, S n = n * (n + 1) / 2 * a n)
  (h3 : ∀ n, S n = (λ x, (x * (x + 1)) / 2 * a x) n) :
  ∀ n : ℕ, a n = 1 / ((n + 1) * (n + 2)) := sorry

end sequence_an_l340_340094


namespace tessa_initial_apples_l340_340068

theorem tessa_initial_apples (x : ℝ) (h : x + 5.0 - 4.0 = 11) : x = 10 :=
by
  sorry

end tessa_initial_apples_l340_340068


namespace ruffy_age_difference_l340_340431

theorem ruffy_age_difference (R O : ℕ) (hR : R = 9) (hRO : R = (3/4 : ℚ) * O) :
  (R - 4) - (1 / 2 : ℚ) * (O - 4) = 1 :=
by 
  sorry

end ruffy_age_difference_l340_340431


namespace slope_between_midpoints_is_zero_l340_340118

-- Define the midpoint of a segment
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Define the slope between two points
def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

-- Define the points and midpoints as given in the conditions
def P1 : ℝ × ℝ := (1, 2)
def P2 : ℝ × ℝ := (3, 6)
def Q1 : ℝ × ℝ := (4, 2)
def Q2 : ℝ × ℝ := (7, 6)

def M1 : ℝ × ℝ := midpoint P1 P2
def M2 : ℝ × ℝ := midpoint Q1 Q2

-- The proof statement: proving that the slope between M1 and M2 is 0
theorem slope_between_midpoints_is_zero : slope M1 M2 = 0 := by
  -- Use 'sorry' to omit the proof details
  sorry

end slope_between_midpoints_is_zero_l340_340118


namespace equilateral_triangle_side_length_l340_340759

theorem equilateral_triangle_side_length 
  {P Q R S : Point} 
  {A B C : Triangle}
  (h₁ : is_inside P A B C)
  (h₂ : is_perpendicular P Q A B)
  (h₃ : is_perpendicular P R B C)
  (h₄ : is_perpendicular P S C A)
  (h₅ : distance P Q = 2)
  (h₆ : distance P R = 3)
  (h₇ : distance P S = 4)
  : side_length A B C = 6 * sqrt 3 :=
by 
  sorry

end equilateral_triangle_side_length_l340_340759


namespace coloring_methods_l340_340514

def color_ways (n : ℕ) (k : ℕ) (diff : ℕ) : ℕ :=
  (n - diff - 1) * (n - diff) / 2

theorem coloring_methods : color_ways 10 2 2 = 28 := by
  unfold color_ways
  simp
  sorry

end coloring_methods_l340_340514


namespace broken_more_than_perfect_spiral_l340_340648

theorem broken_more_than_perfect_spiral :
  let perfect_shells := 17
  let broken_shells := 52
  let broken_spiral_shells := broken_shells / 2
  let perfect_non_spiral_shells := 12
  let perfect_spiral_shells := perfect_shells - perfect_non_spiral_shells
  in broken_spiral_shells - perfect_spiral_shells = 21 :=
by
  sorry

end broken_more_than_perfect_spiral_l340_340648


namespace relay_race_selection_l340_340261

-- Define the number of athletes
def total_athletes : ℕ := 6

-- Define the number of athletes to choose
def chosen_athletes : ℕ := 4

-- Specify that athletes A and B cannot run the first leg
def cannot_run_first_leg (A B : ℕ) : Prop := true

-- Define the problem statement
theorem relay_race_selection (A B : ℕ) (h : cannot_run_first_leg A B) :
  ((total_athletes - 2) * nat.perm (total_athletes - 1) (chosen_athletes - 1)) = 240 :=
sorry

end relay_race_selection_l340_340261


namespace find_a_l340_340976

theorem find_a (a : ℝ) :
  let Δ1 := 4 - 4 * a, Δ2 := a^2 - 8
  in Δ1 > 0 ∧ Δ2 > 0 ∧ 4 - 2 * a = a^2 - 4 ↔ a = -4 := 
by {
  intros,
  sorry
}

end find_a_l340_340976


namespace inclination_angle_range_l340_340669

theorem inclination_angle_range (m : ℝ) :
  let k := (1 + m^2) in
  π/4 ≤ real.arctan k ∧ real.arctan k < π/2 :=
by sorry

end inclination_angle_range_l340_340669


namespace women_more_than_men_l340_340471

theorem women_more_than_men 
(M W : ℕ) 
(h_ratio : (M:ℚ) / W = 5 / 9) 
(h_total : M + W = 14) :
W - M = 4 := 
by 
  sorry

end women_more_than_men_l340_340471


namespace apples_sum_l340_340406

theorem apples_sum
  (maggie_kelsey_sum : 40 + 28 = 68)
  (average : 30)
  (num_people : 4)
  (total_apples : average * num_people = 120) :
  (∃ L A, L + A = 120 - 68) :=
by
  sorry

end apples_sum_l340_340406


namespace intersection_x_axis_l340_340163

theorem intersection_x_axis (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (7, 3)) (h2 : (x2, y2) = (3, -1)) :
  ∃ x : ℝ, (x, 0) = (4, 0) :=
by sorry

end intersection_x_axis_l340_340163


namespace track_length_l340_340105

theorem track_length (V_A V_B V_C : ℝ) (x : ℝ) 
  (h1 : x / V_A = (x - 1) / V_B) 
  (h2 : x / V_A = (x - 2) / V_C) 
  (h3 : x / V_B = (x - 1.01) / V_C) : 
  110 - x = 9 :=
by 
  sorry

end track_length_l340_340105


namespace handed_out_apples_l340_340446

def total_apples : ℤ := 96
def pies : ℤ := 9
def apples_per_pie : ℤ := 6
def apples_for_pies : ℤ := pies * apples_per_pie
def apples_handed_out : ℤ := total_apples - apples_for_pies

theorem handed_out_apples : apples_handed_out = 42 := by
  sorry

end handed_out_apples_l340_340446


namespace cannot_divide_convex_polygon_into_nonconvex_quadrilaterals_l340_340368

-- Definition of a convex polygon
def is_convex_polygon (P : polygon) : Prop :=
  ∀ {a b : P}, segment a b ⊆ P

-- The proof problem statement
theorem cannot_divide_convex_polygon_into_nonconvex_quadrilaterals (P : polygon) 
(h_convex : is_convex_polygon P) : ¬ ∃ Q : finset (polygon), (∀ q ∈ Q, is_quadrilateral q ∧ ¬ is_convex_polygon q) :=
  sorry

end cannot_divide_convex_polygon_into_nonconvex_quadrilaterals_l340_340368


namespace necessary_but_not_sufficient_condition_l340_340000

def is_increasing_sequence {α : Type*} [linear_order α] (a : ℕ → α) : Prop := 
  ∀ n, a n < a (n + 1)

def abs_gt {α : Type*} [linear_ordered_add_comm_group α] (a : ℕ → α) : Prop :=
  ∀ n, abs (a (n + 1)) > a n

theorem necessary_but_not_sufficient_condition 
  {α : Type*} [linear_ordered_add_comm_group α] 
  (a : ℕ → α) : abs_gt a → ¬is_increasing_sequence a := 
sorry

end necessary_but_not_sufficient_condition_l340_340000


namespace lcm_24_90_35_l340_340583

-- Definition of least common multiple (can be found in Mathlib)
def lcm (a b : ℕ) : ℕ := sorry -- placeholder for actual LCM definition, if not directly usable from Mathlib

-- The LCM of three numbers
def lcm3 (a b c : ℕ) : ℕ := lcm (lcm a b) c

-- The given problem conditions
def a := 24
def b := 90
def c := 35

-- The statement to prove
theorem lcm_24_90_35 : lcm3 a b c = 2520 := by
  sorry

end lcm_24_90_35_l340_340583


namespace factor_correct_l340_340971

noncomputable def p (b : ℝ) : ℝ := 221 * b^2 + 17 * b
def factored_form (b : ℝ) : ℝ := 17 * b * (13 * b + 1)

theorem factor_correct (b : ℝ) : p b = factored_form b := by
  sorry

end factor_correct_l340_340971


namespace Gretchen_walking_time_l340_340641

theorem Gretchen_walking_time
  (hours_worked : ℕ)
  (minutes_per_hour : ℕ)
  (sit_per_walk : ℕ)
  (walk_per_brake : ℕ)
  (h1 : hours_worked = 6)
  (h2 : minutes_per_hour = 60)
  (h3 : sit_per_walk = 90)
  (h4 : walk_per_brake = 10) :
  let total_minutes := hours_worked * minutes_per_hour,
      breaks := total_minutes / sit_per_walk,
      walking_time := breaks * walk_per_brake in
  walking_time = 40 := 
by
  sorry

end Gretchen_walking_time_l340_340641


namespace winning_candidate_percentage_l340_340515

theorem winning_candidate_percentage (votes1 votes2 votes3 : ℕ) (h_votes1 : votes1 = 3000) (h_votes2 : votes2 = 5000) (h_votes3 : votes3 = 15000) :
  let total_votes := votes1 + votes2 + votes3 in
  let winning_votes := max (max votes1 votes2) votes3 in
  (winning_votes : ℝ) / (total_votes : ℝ) * 100 ≈ 65.22 :=
by sorry

end winning_candidate_percentage_l340_340515


namespace find_x_l340_340854

theorem find_x :
  ∃ x : ℝ, 0.2 * x + 0.6 * 0.8 = 0.56 ∧ x = 0.4 :=
begin
  -- Proof steps can go here...
  sorry
end

end find_x_l340_340854


namespace bullet_train_passing_time_l340_340151

-- Given definitions
def length_of_train : ℝ := 200  -- in meters
def speed_of_train_kmph : ℝ := 69  -- in kmph
def speed_of_man_kmph : ℝ := 3  -- in kmph

-- Conversion factor
def kmph_to_mps (kmph : ℝ) : ℝ := kmph * (1000 / 3600)

-- Relative speed calculation
def relative_speed_mps : ℝ := kmph_to_mps (speed_of_train_kmph + speed_of_man_kmph)

-- Time calculation
def passing_time : ℝ := length_of_train / relative_speed_mps

-- Theorem statement
theorem bullet_train_passing_time : passing_time = 10 :=
by
  -- Proof skipped
  sorry

end bullet_train_passing_time_l340_340151


namespace problem1_problem2_l340_340867

-- Problem 1
theorem problem1 (a b : ℤ) :
  (a - b)^2 = 25 := 
by
  assume h : 2 - b = 0,
  assume k : a + 3 = 0,
  calc (a - b)^2
      = ((-3) - 2)^2 : by rw [←k, h]
  ... = (-5)^2 : by rfl
  ... = 25 : by norm_num

-- Problem 2
theorem problem2 (m n : ℤ) :
  -1 - n = 0 → -m + 6 = 0 → m = 6 ∧ n = -1 := 
by
  assume h₁ : -1 - n = 0,
  assume h₂ : -m + 6 = 0,
  have n_eq := eq_neg_of_add_eq_zero_left h₁,
  have m_eq := eq_of_add_eq_zero_right (neg_eq_zero.mpr h₂),
  exact ⟨m_eq, n_eq⟩

end problem1_problem2_l340_340867


namespace solve_for_x_l340_340786

theorem solve_for_x (x : ℝ) : 2 ^ (32 ^ x) = 32 ^ (2 ^ x) → x = Real.log 5 / (4 * Real.log 2) :=
by intros h; sorry

end solve_for_x_l340_340786


namespace function_extreme_values_l340_340081

/-- Prove that the function y = 1 + 3x - x^3 has a minimum value of -1 and a maximum value of 3 -/
theorem function_extreme_values : 
  ∃ (x_min x_max : ℝ), 
    (∀ x ∈ (-∞, x_min) ∪ (x_max, ∞), (1 + 3 * x - x^3) < (1 + 3 * x_min - x_min^3) ∧
                                       (1 + 3 * x - x^3) < (1 + 3 * x_max - x_max^3)) ∧
    (∀ x ∈ (x_min, x_max), (1 + 3 * x - x^3) > (1 + 3 * x_min - x_min^3) ∧
                            (1 + 3 * x - x^3) > (1 + 3 * x_max - x_max^3)) ∧
    (1 + 3 * (-1) - (-1)^3 = -1) ∧
    (1 + 3 * 1 - 1^3 = 3) :=
by {
  sorry -- proof omitted
}

end function_extreme_values_l340_340081


namespace equilateral_triangle_side_length_l340_340756

open Classical

noncomputable section

variable {P Q R S : Type}

def is_perpendicular_feet (P Q R S : P) : Prop :=
  sorry -- Definition for Q, R, S being the feet of perpendiculars from P

structure EquilateralTriangle (A B C P Q R S : P) where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  AB : ℝ
  (h_perpendicular : is_perpendicular_feet P Q R S)
  (h_area_eq : ∀ (h₁ : PQ = 2) (h₂ : PR = 3) (h₃ : PS = 4), AB = 12 * Real.sqrt 3)

theorem equilateral_triangle_side_length {A B C P Q R S : P} 
    (h_eq_triangle : EquilateralTriangle A B C P Q R S) : h_eq_triangle.AB = 12 * Real.sqrt 3 :=
  by
    cases h_eq_triangle with
    | mk PQ PR PS AB h_perpendicular h_area_eq =>
        apply h_area_eq
        · exact rfl
        · exact rfl
        · exact rfl

end equilateral_triangle_side_length_l340_340756


namespace circle_has_max_perimeter_l340_340784

noncomputable def ellipse_perimeter (a b : ℝ) : ℝ :=
  4 * a * (∫ θ in 0..(Real.pi / 2), Real.sqrt (1 - (1 - (b/a)^2) * (Real.cos θ)^2))

def circle_perimeter (r : ℝ) : ℝ :=
  2 * Real.pi * r

theorem circle_has_max_perimeter (S : Set (ℝ × ℝ)) (r a b : ℝ) (h₁ : ∀ p ∈ S, (r * r) ≥ (p.1 * p.1 + p.2 * p.2))
  (h₂ : ∀ p ∈ S, (a * a) ≥ (p.1 * p.1) ∧ (b * b) ≥ (p.2 * p.2)) :
  ∀ (a b : ℝ) (h : b ≤ a), ellipse_perimeter a b ≤ circle_perimeter (Sqrt.sqrt (r^2 / 2)) :=
by
  sorry

end circle_has_max_perimeter_l340_340784


namespace coeff_x_squared_in_expansion_l340_340796

-- Define the polynomial expressions
def poly_exp1 (x : ℝ) := (x - 2) ^ 3
def poly_exp2 (x : ℝ) := (x + 1) ^ 4

-- State the problem as a theorem in Lean
theorem coeff_x_squared_in_expansion :
  polynomial.coeff ((polynomial.C 1 * (polynomial.X - 2) ^ 3) * ((polynomial.C 1 * (polynomial.X + 1) ^ 4))) 2 = -6 := by sorry

end coeff_x_squared_in_expansion_l340_340796


namespace option_c_correct_l340_340497

theorem option_c_correct (a : ℝ) (h : ∀ x : ℝ, 1 ≤ x ∧ x < 2 → x^2 - a ≤ 0) : 4 < a :=
by
  sorry

end option_c_correct_l340_340497


namespace star_six_three_l340_340655

-- Definition of the operation
def star (a b : ℕ) : ℕ := 4 * a + 5 * b - 2 * a * b

-- Statement to prove
theorem star_six_three : star 6 3 = 3 := by
  sorry

end star_six_three_l340_340655


namespace tan_theta_minus_pi_over_4_l340_340274

theorem tan_theta_minus_pi_over_4 (θ : ℝ) (h1 : sin θ = 3 / 5) (h2 : 0 < θ ∧ θ < π / 2) : 
  tan (θ - π / 4) = -1 / 7 := 
by sorry

end tan_theta_minus_pi_over_4_l340_340274


namespace total_time_christine_sees_pablo_l340_340214

-- Define the speeds in km/h
def v_j := 10 -- Christine's jogging speed in km/h
def v_c := 6  -- Pablo's cycling speed in km/h

-- Define the distances in meters
def d1 := 300 -- initial distance in meters
def d2 := 400 -- final distance in meters

-- Intermediate calculation of relative speed in km/h and meters per minute
def relative_speed_kmh := v_j - v_c
def relative_speed_mpm := (relative_speed_kmh * 1000) / 60

-- Target time calculation in minutes
def time_seen := (d1 / relative_speed_mpm) + (d2 / relative_speed_mpm)

theorem total_time_christine_sees_pablo : 
  time_seen = 10.5 :=
by sorry

end total_time_christine_sees_pablo_l340_340214


namespace domain_of_f_l340_340227

noncomputable def f (x : ℝ) := real.sqrt (4 - real.sqrt (6 - real.sqrt (7 - real.sqrt x)))

theorem domain_of_f : {x : ℝ | 0 ≤ x ∧ x ≤ 49} = {x : ℝ | ∃ y, f y ≠ f y} :=
by
  sorry

end domain_of_f_l340_340227


namespace point_coordinates_correct_l340_340300

theorem point_coordinates_correct (k : ℝ) : 
  (∀ x y : ℝ, y = k * x + 3 → (x = 1 → y = -2 )) → k < 0 :=
by
  intro h
  have h1 : -2 = k * 1 + 3 := h 1 -2
  linarith
  sorry

end point_coordinates_correct_l340_340300


namespace area_of_triangle_PQR_l340_340839

/-- The vertices of the triangle -/
def P : (ℝ × ℝ) := (-3, 2)
def Q : (ℝ × ℝ) := (1, 5)
def R : (ℝ × ℝ) := (4, -1)

/-- The area of the triangle PQR is 16.5 square units -/
theorem area_of_triangle_PQR : 
  let x1 := P.1, y1 := P.2
  let x2 := Q.1, y2 := Q.2
  let x3 := R.1, y3 := R.2
in (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) = 16.5 :=
by 
  let x1 := P.1, y1 := P.2
  let x2 := Q.1, y2 := Q.2
  let x3 := R.1, y3 := R.2
  -- calculate the area of the triangle
  have area := (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
  -- show that the area is correct
  have h : area = 16.5 := sorry
  exact h

end area_of_triangle_PQR_l340_340839


namespace max_value_of_expressions_l340_340244

theorem max_value_of_expressions :
  ∃ x ∈ ℝ, (∀ y ∈ ℝ, (2^y - 4^y) ≤ (2^x - 4^x)) ∧ (2^x - 4^x = 1 / 4) :=
by
  sorry

end max_value_of_expressions_l340_340244


namespace unique_two_color_codes_identify_subjects_l340_340897

theorem unique_two_color_codes_identify_subjects : 
  ∀ (n m : ℕ) (c : ℕ) (subjects : ℕ), 
  n = 5 → 
  c = n * n → 
  subjects = 16 → 
  c ≥ subjects → 
  subjects = 16 → 
  (c - subjects) = 0 := 
by
  intros n m c subjects hn hc hsubjects hcgtsubjects eqsubjects
  have hn : n = 5 := by assumption,
  have hc : c = n * n := by assumption,
  have hsubjects : subjects = 16 := by assumption,
  have hcgtsubjects : c ≥ subjects := by assumption,
  have eqsubjects : subjects = 16 := by assumption,
  rw [hn, hc] at *,
  rw hsubjects at *,
  simp,
  sorry

end unique_two_color_codes_identify_subjects_l340_340897


namespace Michael_made_97_dollars_l340_340038

def price_large : ℕ := 22
def price_medium : ℕ := 16
def price_small : ℕ := 7

def quantity_large : ℕ := 2
def quantity_medium : ℕ := 2
def quantity_small : ℕ := 3

def calculate_total_money (price_large price_medium price_small : ℕ) 
                           (quantity_large quantity_medium quantity_small : ℕ) : ℕ :=
  (price_large * quantity_large) + (price_medium * quantity_medium) + (price_small * quantity_small)

theorem Michael_made_97_dollars :
  calculate_total_money price_large price_medium price_small quantity_large quantity_medium quantity_small = 97 := 
by
  sorry

end Michael_made_97_dollars_l340_340038


namespace sean_div_julie_eq_two_l340_340781

def sum_n (n : ℕ) := n * (n + 1) / 2

def sean_sum := 2 * sum_n 500

def julie_sum := sum_n 500

theorem sean_div_julie_eq_two : sean_sum / julie_sum = 2 := 
by sorry

end sean_div_julie_eq_two_l340_340781


namespace nathan_has_83_bananas_l340_340414

def nathan_bananas (bunches_eight bananas_eight bunches_seven bananas_seven: Nat) : Nat :=
  bunches_eight * bananas_eight + bunches_seven * bananas_seven

theorem nathan_has_83_bananas (h1 : bunches_eight = 6) (h2 : bananas_eight = 8) (h3 : bunches_seven = 5) (h4 : bananas_seven = 7) : 
  nathan_bananas bunches_eight bananas_eight bunches_seven bananas_seven = 83 := by
  sorry

end nathan_has_83_bananas_l340_340414


namespace log_eq_neg_one_fourth_l340_340275

theorem log_eq_neg_one_fourth (x : ℝ) (hx : 1 < x)
  (h : log 2 (log 4 x) + log 4 (log 16 x) + log 16 (log 2 x) = 0) :
  log 2 (log 16 x) + log 16 (log 4 x) + log 4 (log 2 x) = -1 / 4 := by
  sorry

end log_eq_neg_one_fourth_l340_340275


namespace intersection_of_A_and_B_l340_340060

def A := {x : ℝ | x^2 - 5 * x + 6 > 0}
def B := {x : ℝ | x / (x - 1) < 0}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 1} :=
by sorry

end intersection_of_A_and_B_l340_340060


namespace ratio_of_ions_in_blood_l340_340831

noncomputable def log10 (x : ℝ) : ℝ :=
  Real.log x / Real.log 10

def pH (H_conc : ℝ) : ℝ :=
  -log10 H_conc

theorem ratio_of_ions_in_blood (H_conc OH_conc : ℝ) 
  (h1 : H_conc * OH_conc = 10^(-14))
  (h2 : 7.35 < pH H_conc)
  (h3 : pH H_conc < 7.45) :
  H_conc / OH_conc = 1 / 6 :=
by
  sorry

end ratio_of_ions_in_blood_l340_340831


namespace shaded_seats_cover_all_l340_340157

theorem shaded_seats_cover_all (table : Fin 10) (start : Fin 10) (k : ℕ) : 
    (∀ n : ℕ, n ∈ {0, 3, 6, 9, ... , k} → n mod 10 ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 0}) 
    → (∃ k, k = 13) :=
by
  -- We assume the set of shaded seats and their properties based on conditions provided.
  sorry

end shaded_seats_cover_all_l340_340157


namespace net_salary_change_is_minus_18_point_1_percent_l340_340189

variable (S : ℝ)

def initial_salary := S
def after_first_increase := S * 1.20
def after_second_increase := after_first_increase * 1.40
def after_first_decrease := after_second_increase * (1 - 0.35)
def after_second_decrease := after_first_decrease * (1 - 0.25)
def final_salary := after_second_decrease

noncomputable def percentage_change := ((final_salary - initial_salary) / initial_salary) * 100

theorem net_salary_change_is_minus_18_point_1_percent :
  percentage_change S = -18.1 :=
by
  sorry

end net_salary_change_is_minus_18_point_1_percent_l340_340189


namespace part1_intersection_when_m_3_part1_union_when_m_3_part2_if_intersection_eq_B_l340_340618

open Set

variable (m : ℝ)

def A : Set ℝ := {x | x ^ 2 - 2 * x - 3 ≤ 0}
def B : Set ℝ := {x | m - 2 ≤ x ∧ x ≤ m + 1}

theorem part1_intersection_when_m_3 :
  A ∩ B = {x | 1 ≤ x ∧ x ≤ 3} :=
sorry

theorem part1_union_when_m_3 :
  A ∪ B = {x | -1 ≤ x ∧ x ≤ 4} :=
sorry

theorem part2_if_intersection_eq_B :
  A ∩ B = B → 1 ≤ m ∧ m ≤ 2 :=
sorry

end part1_intersection_when_m_3_part1_union_when_m_3_part2_if_intersection_eq_B_l340_340618


namespace cards_given_away_l340_340991

open Nat

def initial_cards : Nat := 26
def additional_cards : Nat := 40
def current_cards : Nat := 48

theorem cards_given_away : ∃ (cards_given : Nat), cards_given = (initial_cards + additional_cards - current_cards) :=
by
  use (initial_cards + additional_cards - current_cards)
  sorry

end cards_given_away_l340_340991


namespace candle_height_half_burned_time_l340_340525

/-- A large candle is 150 centimeters tall. It burns at a variable
   rate, taking 15 seconds to burn the first centimeter, 30 seconds 
   for the second centimeter, up to 15k seconds to burn the kth centimeter. 
   Compute the height of the candle in centimeters after T/2 seconds from 
   the start, where T is the total burn time.

   The height of the candle at T/2 seconds is 44 centimeters. -/
theorem candle_height_half_burned_time :
  let burn_time := λ k : ℕ => 15 * k
  let total_burn_time (n : ℕ) := ∑ k in finset.range (n + 1), burn_time k
  let height_after_time (t : ℕ) := 150 - (∑ k in finset.Ico 1 (t + 1), burn_time k)

  total_burn_time 150 = 170325 →
  height_after_time (170325 / 2) = 44 :=
by
  intros
  sorry

end candle_height_half_burned_time_l340_340525


namespace doughnuts_left_l340_340875

theorem doughnuts_left (dozen : ℕ) (total_initial : ℕ) (eaten : ℕ) (initial : total_initial = 2 * dozen) (d : dozen = 12) : total_initial - eaten = 16 :=
by
  rcases d
  rcases initial
  sorry

end doughnuts_left_l340_340875


namespace infinite_A_l340_340376

theorem infinite_A 
  (A : Set ℝ) 
  (hA₀ : ∀ x ∈ A, 0 ≤ x ∧ x < 1) 
  (hA₁ : Finite { x | x ∈ A }) :
  (∃ a b c d ∈ A, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ ab + cd ∈ A) → False := 
sorry

end infinite_A_l340_340376


namespace smallest_number_starting_with_five_l340_340985

theorem smallest_number_starting_with_five :
  ∃ n : ℕ, ∃ m : ℕ, m = (5 * m + 5) / 4 ∧ 5 * n + m = 512820 ∧ m < 10^6 := sorry

end smallest_number_starting_with_five_l340_340985


namespace measure_diff_eq_l340_340399

noncomputable def P {n : ℕ} : MeasureTheory.Measure (EuclideanSpace ℝ n) := sorry
noncomputable def F {n : ℕ} (x : Fin n → ℝ) : ℝ := P { y | ∀ i, y i ≤ x i }

noncomputable def Delta {n : ℕ} (a b : ℝ) (i : Fin n) (F : (Fin n → ℝ) → ℝ) (x : (Fin n → ℝ)) : ℝ :=
  F (fun j => if j = i then b else x j) - F (fun j => if j = i then a else x j)

theorem measure_diff_eq {n : ℕ} (a b : Fin n → ℝ) :
  Delta (a 0) (b 0) 0 (Delta (a 1) (b 1) 1 (Delta (a 2) (b 2) 2 ... F)) = 
  P { x | ∀ i, a i < x i ∧ x i ≤ b i } := sorry

end measure_diff_eq_l340_340399


namespace M_minus_N_l340_340715

theorem M_minus_N (a b c d : ℕ) (h1 : a + b = 20) (h2 : a + c = 24) (h3 : a + d = 22) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) : 
  let M := 2 * b + 26
  let N := 2 * 1 + 26
  (M - N) = 36 :=
by
  sorry

end M_minus_N_l340_340715


namespace dana_more_pencils_than_marcus_l340_340944

theorem dana_more_pencils_than_marcus :
  ∀ (Jayden Dana Marcus : ℕ), 
  (Jayden = 20) ∧ 
  (Dana = Jayden + 15) ∧ 
  (Jayden = 2 * Marcus) → 
  (Dana - Marcus = 25) :=
by
  intros Jayden Dana Marcus h
  rcases h with ⟨hJayden, hDana, hMarcus⟩
  sorry

end dana_more_pencils_than_marcus_l340_340944


namespace exists_polygon_divisible_into_three_triangles_l340_340428

theorem exists_polygon_divisible_into_three_triangles :
  ∃ (p : polygon), (p = quadrilateral ∨ p = pentagon ∨ p = hexagon ∨ p = heptagon) → 
  (∃ l : straight_line, divides_polygon_into_three_triangles p l) :=
by
  sorry

end exists_polygon_divisible_into_three_triangles_l340_340428


namespace rectangle_length_increase_l340_340448

variable (L B : ℝ) -- Original length and breadth
variable (A : ℝ) -- Original area
variable (p : ℝ) -- Percentage increase in length
variable (A' : ℝ) -- New area

theorem rectangle_length_increase (hA : A = L * B) 
  (hp : L' = L + (p / 100) * L) 
  (hB' : B' = B * 0.9) 
  (hA' : A' = 1.035 * A)
  (hl' : L' = (1 + (p / 100)) * L)
  (hb_length : L' * B' = A') :
  p = 15 :=
by
  sorry

end rectangle_length_increase_l340_340448


namespace multiply_exponents_l340_340207

theorem multiply_exponents (x : ℝ) : (x^2) * (x^3) = x^5 := 
sorry

end multiply_exponents_l340_340207


namespace equi_spaced_on_unit_circle_l340_340649

theorem equi_spaced_on_unit_circle (n : ℕ) (h : n ≥ 2) : 
  (∃ z : ℕ → ℂ, (∀ i, |z i| = 1) ∧ (∑ i in finset.range n, z i) = 0 ∧ 
  (∀ i1 i2, z i1 ≠ z i2 → dist (z i1) (z i2) = dist (z (i1 + 1 % n)) (z (i2 + 1 % n)))) ↔ (n = 2 ∨ n = 3) :=
sorry

end equi_spaced_on_unit_circle_l340_340649


namespace max_value_of_m_l340_340309

-- Define the function f(x)
def f (x : ℝ) := x^2 + 2 * x

-- Define the property of t and m such that the condition holds for all x in [1, m]
def valid_t_m (t m : ℝ) : Prop :=
  ∀ x : ℝ, 1 ≤ x ∧ x ≤ m → f (x + t) ≤ 3 * x

-- The proof statement ensuring the maximum value of m is 8
theorem max_value_of_m 
  (t : ℝ) (m : ℝ) 
  (ht : ∃ x : ℝ, valid_t_m t x ∧ x = 8) : 
  ∀ m, valid_t_m t m → m ≤ 8 :=
  sorry

end max_value_of_m_l340_340309


namespace ratio_area_section4_section1_l340_340903

theorem ratio_area_section4_section1
  (r1 r2 r3 : ℝ)  -- radii of concentric circles
  (α β : ℝ)      -- angles appropriate to different sections
  (h1 : r1 < r2)
  (h2 : r2 < r3)
  (h3 : ∃ t2 t3, t2 = (r1^2 * β) / 2 ∧ t3 = ((r2^2 - r1^2) * α) / 2 ∧
        2 * t2 = t3) :
  let t4 := (r3^2 - r2^2) * β / 2
  let t1 := r1^2 * α / 2
  in t4 / t1 = 8 := by
  sorry

end ratio_area_section4_section1_l340_340903


namespace sandwich_and_soda_cost_l340_340432

theorem sandwich_and_soda_cost:
  let sandwich_cost := 4
  let soda_cost := 1
  let num_sandwiches := 6
  let num_sodas := 10
  let total_cost := num_sandwiches * sandwich_cost + num_sodas * soda_cost
  total_cost = 34 := 
by 
  sorry

end sandwich_and_soda_cost_l340_340432


namespace addition_equations_count_l340_340421

theorem addition_equations_count :
  let digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  ∃ (A B C D E a b c d e : ℕ),
  A ∈ digits ∧ B ∈ digits ∧ C ∈ digits ∧ D ∈ digits ∧ E ∈ digits ∧
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧
  (A + B + C+ D + E) + (a + b + c + d + e) = 45 ∧
  (A * 10000 + B * 1000 + C * 100 + D * 10 + E) + (a * 10000 + b * 1000 + c * 100 + d * 10 + e) = 99999
---
  1536 := sorry

end addition_equations_count_l340_340421


namespace adjacent_diff_at_least_six_l340_340343

theorem adjacent_diff_at_least_six :
  ∀ (table : Fin 9 → Fin 9 → ℕ), 
  (∀ i j, 1 ≤ table i j ∧ table i j ≤ 81) ∧ 
  (∀ i j i' j', (abs (i - i') + abs (j - j') = 1) → 1 ≤ table i j ∧ table i j ≤ 81 → 1 ≤ table i' j' ∧ table i' j' ≤ 81) → 
  ∃ (i j i' j'), (abs (i - i') + abs (j - j') = 1) ∧ abs (table i j - table i' j') ≥ 6 := 
by 
  sorry

end adjacent_diff_at_least_six_l340_340343


namespace probability_of_longer_segment_is_half_l340_340541

-- Define the length of the rod
def rod_length : ℝ := 4

-- Define the event A that one of the segments is longer than 1 meter
def event_A (x : ℝ) : Prop := (x > 1) ∨ (rod_length - x > 1)

-- Calculate the probability P(A)
def probability_event_A : ℝ :=
  (λ x, if event_A x then 1 else 0).integral (0, rod_length) / rod_length

theorem probability_of_longer_segment_is_half :
  probability_event_A = 1 / 2 :=
sorry

end probability_of_longer_segment_is_half_l340_340541


namespace keanu_fish_total_cost_l340_340373

def total_fish_cost (fish_dog : ℕ) (cost_fish : ℕ) (fish_cat_mul_factor : ℚ) : ℕ :=
  let cost_dog := fish_dog * cost_fish
  let fish_cat := (fish_cat_mul_factor * fish_dog).toNat
  let cost_cat := fish_cat * cost_fish
  cost_dog + cost_cat

theorem keanu_fish_total_cost (h1 : 40 = fish_dog) (h2 : 4 = cost_fish) 
  (h3 : 1 / 2 = fish_cat_mul_factor) : total_fish_cost 40 4 (1 / 2) = 240 := by
  sorry

end keanu_fish_total_cost_l340_340373


namespace problem1_problem2_l340_340637

open Real

noncomputable def a (x : ℝ) : ℝ × ℝ := (cos (3 * x / 2), sin (3 * x / 2))
noncomputable def b (x : ℝ) : ℝ × ℝ := (cos (x / 2), -sin (x / 2))

def perpendicular (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

def magnitude (u : ℝ × ℝ) : ℝ :=
  sqrt (u.1 ^ 2 + u.2 ^ 2)

theorem problem1 (x : ℝ) (h : x ∈ set.Icc (-π / 2) (π / 2)) :
  perpendicular (a x - b x) (a x + b x) := sorry

theorem problem2 (x : ℝ) (h : x ∈ set.Icc (-π / 2) (π / 2)) (h1 : magnitude (a x + b x) = 1 / 3) :
  2 * cos x = 1 / 3 := sorry

end problem1_problem2_l340_340637


namespace min_value_a_plus_3b_plus_9c_l340_340032

theorem min_value_a_plus_3b_plus_9c {a b c : ℝ} (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 27) :
  a + 3*b + 9*c ≥ 27 :=
sorry

end min_value_a_plus_3b_plus_9c_l340_340032


namespace tangent_slope_l340_340122

-- Define the coordinates for the center of the circle and the point of tangency
def center : ℝ × ℝ := (2, 1)
def point_of_tangency : ℝ × ℝ := (6, 3)

-- Define a function to compute the slope between two points
def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

-- Show that the slope of the tangent line at the point of tangency is -2
theorem tangent_slope :
  slope center point_of_tangency = 1/2 → 
  slope point_of_tangency center = -2 :=
by
  intros h

  -- The slope of the radius
  have slope_radius : slope center point_of_tangency = 1/2 := h

  -- The slope of the tangent is the negative reciprocal of the radius slope
  have tangent_slope : slope point_of_tangency center = -1 / slope_radius := by
    rw slope_radius
    have reciprocal : 1 / (1 / 2) = 2 := sorry  -- Prove that the negative reciprocal is -2
    rw reciprocal
    exact sorry
  
  exact sorry

end tangent_slope_l340_340122


namespace quadratic_equation_roots_quadratic_equation_discriminant_l340_340258

theorem quadratic_equation_roots (m : ℝ) :
  (-2 : ℝ) is_root_of_polynomial (λ x, x^2 + m * x + m - 2) → is_root_of_polynomial (λ x, x^2 + m * x + m - 2) (0 : ℝ) :=
sorry

theorem quadratic_equation_discriminant (m : ℝ) :
  (m - 2)^2 + 4 > 0 :=
by
  calc (m - 2)^2 + 4 = ... : sorry
       ... > 0 : sorry

end quadratic_equation_roots_quadratic_equation_discriminant_l340_340258


namespace sum_of_intervals_length_l340_340467

-- Given function
def floor_sqrt (k : ℕ) : ℤ := int.floor (real.sqrt k)

def expression (x : ℝ) : ℝ := 
  ∏ k in finset.range 150, (x - (k+1)) ^ floor_sqrt (k+1)

-- Definition of the actual proof statement
theorem sum_of_intervals_length : 
  let intervals := { I : set ℝ | is_interval I ∧ ∃ k, I = set.Ioc k (k + 1) ∧
                                   (∀ x ∈ I, expression x < 0) } in
  (finset.sum (finset.univ.filter λ I, I ∈ intervals) (λ I, set.Ioc.length I)).round 2 = 78.00 := 
sorry

end sum_of_intervals_length_l340_340467


namespace oranges_thrown_away_l340_340177

theorem oranges_thrown_away (initial_oranges new_oranges current_oranges : ℕ) (x : ℕ) 
  (h1 : initial_oranges = 50)
  (h2 : new_oranges = 24)
  (h3 : current_oranges = 34) : 
  initial_oranges - x + new_oranges = current_oranges → x = 40 :=
by
  intros h
  rw [h1, h2, h3] at h
  sorry

end oranges_thrown_away_l340_340177


namespace triangle_area_ABC_l340_340361

theorem triangle_area_ABC 
  (A B C D E : Type) 
  (h_tri : triangle A B C)
  (h_eq : AB = BC) 
  (h_alt : altitude D B)
  (h_BE : BE = 12)
  (h_tan_geo : geometric_prog (tan ∠ CBE) (tan ∠ DBE) (tan ∠ ABE))
  (h_cot_arith : arithmetic_prog (cot ∠ DBE) (cot ∠ CBE) (cot ∠ ABE)) 
  : area A B C = 24 :=
sorry

end triangle_area_ABC_l340_340361
