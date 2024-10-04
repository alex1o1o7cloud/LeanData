import Algebra.BigOperators
import Mathlib
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Factorial
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Order
import Mathlib.Analysis.SpecialFunctions.Integrals
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.LinearAlgebra.Basic
import Mathlib.MeasureTheory.Integral.IntervalIntegral
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.Basic

namespace analytic_expression_of_f_range_of_k_l738_738423

noncomputable def quadratic_function_minimum (a b : ℝ) : ℝ :=
a * (-1) ^ 2 + b * (-1) + 1

theorem analytic_expression_of_f (a b : ℝ) (ha : quadratic_function_minimum a b = 0)
  (hmin: -1 = -b / (2 * a)) : a = 1 ∧ b = 2 :=
by sorry

theorem range_of_k (k : ℝ) : ∃ k : ℝ, (k ∈ Set.Ici 3 ∨ k = 13 / 4) :=
by sorry

end analytic_expression_of_f_range_of_k_l738_738423


namespace bagel_machine_completion_time_l738_738977

theorem bagel_machine_completion_time :
  ∀ start completion one_fourth_time : ℕ,
  start = 7 ∧ completion = 19 ∧ one_fourth_time = 3 →
  ∀ t : ℕ, t = start + 4 * one_fourth_time → t = completion :=
begin
  sorry
end

end bagel_machine_completion_time_l738_738977


namespace john_treats_patients_per_year_l738_738843

theorem john_treats_patients_per_year :
  (let 
    patients_first_hospital_per_day := 20,
    patients_second_hospital_per_day := patients_first_hospital_per_day + (20 / 100 * patients_first_hospital_per_day),
    days_per_week := 5,
    weeks_per_year := 50,
    patients_first_hospital_per_week := patients_first_hospital_per_day * days_per_week,
    patients_second_hospital_per_week := patients_second_hospital_per_day * days_per_week,
    total_patients_per_week := patients_first_hospital_per_week + patients_second_hospital_per_week,
    total_patients_per_year := total_patients_per_week * weeks_per_year
  in total_patients_per_year = 11000) :=
by 
  sorry

end john_treats_patients_per_year_l738_738843


namespace toby_change_l738_738939

theorem toby_change :
  let cheeseburger_cost := 3.65
  let milkshake_cost := 2
  let coke_cost := 1
  let large_fries_cost := 4
  let cookie_cost := 0.5
  let num_cookies := 3
  let tax := 0.2
  let initial_amount := 15
  let total_cost := 2 * cheeseburger_cost + milkshake_cost + coke_cost + large_fries_cost + num_cookies * cookie_cost + tax
  let cost_per_person := total_cost / 2
  let toby_change := initial_amount - cost_per_person
  in toby_change = 7 :=
sorry

end toby_change_l738_738939


namespace scalar_existence_l738_738337

variable (v : ℝ^3)

def i : ℝ^3 := ![1, 0, 0]
def j : ℝ^3 := ![0, 1, 0]
def k : ℝ^3 := ![0, 0, 1]

theorem scalar_existence :
  i × (v × i) + j × (v × j) + k × (v × k) + v = 3 * v :=
by
  sorry

end scalar_existence_l738_738337


namespace max_area_triangle_l738_738839

theorem max_area_triangle (A B C : ℝ) (a b c : ℝ) (h1 : Real.sqrt 2 * Real.sin A = Real.sqrt 3 * Real.cos A) (h2 : a = Real.sqrt 3) :
  ∃ (max_area : ℝ), max_area = (3 * Real.sqrt 3) / (8 * Real.sqrt 5) := 
sorry

end max_area_triangle_l738_738839


namespace f_neg_one_f_two_l738_738129

def f (x : ℝ) : ℝ :=
if x < 0 then 3 * x + 4
else x^2 - 2 * x + 1

theorem f_neg_one : f (-1) = -1 :=
by sorry

theorem f_two : f 2 = 1 :=
by sorry

end f_neg_one_f_two_l738_738129


namespace simplify_cos_sum_simplify_sin_sum_l738_738632

-- Lean statement for the first problem
theorem simplify_cos_sum (α : ℝ) (n : ℕ) (C : ℕ → ℕ → ℕ) :
  (cos α + ∑ k in finset.range n, C n k * cos ((k + 2) * α)) = 
  2^n * (cos (α/2))^n * cos ((n+2) * α / 2) :=
sorry

-- Lean statement for the second problem
theorem simplify_sin_sum (α : ℝ) (n : ℕ) (C : ℕ → ℕ → ℕ) :
  (sin α + ∑ k in finset.range n, C n k * sin ((k + 2) * α)) = 
  2^n * (cos (α/2))^n * sin ((n+2) * α / 2) :=
sorry

end simplify_cos_sum_simplify_sin_sum_l738_738632


namespace exists_unequal_translatable_polygons_l738_738533

theorem exists_unequal_translatable_polygons :
  ∃ (P P' : Polygon), (area P = area P') ∧ (¬∃ (P1 P2 : Polygon), (P1 ⊆ P ∧ P2 ⊆ P') ∧ (translate P1 = P2)) :=
sorry

end exists_unequal_translatable_polygons_l738_738533


namespace purely_imaginary_z_point_on_line_z_l738_738414

-- Proof problem for (I)
theorem purely_imaginary_z (a : ℝ) (z : ℂ) (h : z = Complex.mk 0 (a+2)) 
: a = 2 :=
sorry

-- Proof problem for (II)
theorem point_on_line_z (a : ℝ) (x y : ℝ) (h1 : x = a^2-4) (h2 : y = a+2) (h3 : x + 2*y + 1 = 0) 
: a = -1 :=
sorry

end purely_imaginary_z_point_on_line_z_l738_738414


namespace find_rectangle_length_l738_738095

-- Define the problem conditions
def length_is_three_times_breadth (l b : ℕ) : Prop := l = 3 * b
def area_of_rectangle (l b : ℕ) : Prop := l * b = 6075

-- Define the theorem to prove the length of the rectangle given the conditions
theorem find_rectangle_length (l b : ℕ) (h1 : length_is_three_times_breadth l b) (h2 : area_of_rectangle l b) : l = 135 := 
sorry

end find_rectangle_length_l738_738095


namespace garden_perimeter_is_64_l738_738607

theorem garden_perimeter_is_64 :
    ∀ (width_garden length_garden width_playground length_playground : ℕ),
    width_garden = 24 →
    width_playground = 12 →
    length_playground = 16 →
    width_playground * length_playground = width_garden * length_garden →
    2 * length_garden + 2 * width_garden = 64 :=
by
  intros width_garden length_garden width_playground length_playground
  intro h1
  intro h2
  intro h3
  intro h4
  sorry

end garden_perimeter_is_64_l738_738607


namespace total_distance_travelled_l738_738948

/-- Proving that the total horizontal distance traveled by the centers of two wheels with radii 1 m and 2 m 
    after one complete revolution is 6π meters. -/
theorem total_distance_travelled (R1 R2 : ℝ) (h1 : R1 = 1) (h2 : R2 = 2) : 
    2 * Real.pi * R1 + 2 * Real.pi * R2 = 6 * Real.pi :=
by
  sorry

end total_distance_travelled_l738_738948


namespace problem1_problem2_l738_738010

variable (α : Real)
hypothesis (h : Real.tan α = 1 / 3)

theorem problem1 : (Real.sin α + Real.cos α) / (5 * Real.cos α - Real.sin α) = 3 / 10 :=
by
  sorry

theorem problem2 : 1 / (2 * Real.sin α * Real.cos α + Real.cos α ^ 2) = 2 / 3 :=
by
  sorry

end problem1_problem2_l738_738010


namespace ratio_addition_l738_738089

theorem ratio_addition (a b : ℕ) (h : a / b = 2 / 3) : (a + b) / b = 5 / 3 := 
by sorry

end ratio_addition_l738_738089


namespace monotonic_intervals_perimeter_range_l738_738067

noncomputable def vector_a (x : Real) : Real × Real := (Real.sin x, 2 * Real.sqrt 3 * Real.cos x)
noncomputable def vector_b (x : Real) : Real × Real := (2 * Real.sin x, Real.sin x)

noncomputable def f (x : Real) : Real :=
  let a := vector_a x
  let b := vector_b x
  a.fst * b.fst + a.snd * b.snd + 1

theorem monotonic_intervals (k : Int) :
  (∀ x : Real, f(x) = 2 * Real.sin (2 * x - Real.pi / 6) + 2) →
  ((k * Real.pi - Real.pi / 6) ≤ x ∧ x ≤ (k * Real.pi + Real.pi / 3)) ∨
  ((k * Real.pi - Real.pi / 3) ≤ x ∧ x ≤ (k * Real.pi + 5 * Real.pi / 6)) :=
sorry

theorem perimeter_range (A : Real) :
  (f(A) = 4 ∧ A = Real.pi / 3 ∧ 2 = 2) →
  ∃ l, (l = 2 + 4 * Real.sin (B + Real.pi / 6)) ∧
  (2 + 2 * Real.sqrt 3 < l) ∧ (l <= 6) :=
sorry

end monotonic_intervals_perimeter_range_l738_738067


namespace sum_distances_regular_tetrahedron_l738_738809

noncomputable def sum_distances_to_faces (P : ℝ×ℝ×ℝ) (d₁ d₂ d₃ d₄ : ℝ) : Prop :=
  ∀ (d₁ d₂ d₃ d₄ : ℝ), 
    d₁ + d₂ + d₃ + d₄ = ℝ

theorem sum_distances_regular_tetrahedron (P : ℝ×ℝ×ℝ) : 
  ∀ a b c d,
    a + b + c + d =  (real.sqrt 6) / 3 :=
begin
  sorry
end

end sum_distances_regular_tetrahedron_l738_738809


namespace trajectory_equation_l738_738462

noncomputable def A : ℝ × ℝ := (0, -1)
noncomputable def B (x_b : ℝ) : ℝ × ℝ := (x_b, -3)
noncomputable def M (x y : ℝ) : ℝ × ℝ := (x, y)

-- Conditions as definitions in Lean 4
def MB_parallel_OA (x y x_b : ℝ) : Prop :=
  ∃ k : ℝ, (x_b - x) = k * 0 ∧ (-3 - y) = k * (-1)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def condition (x y x_b : ℝ) : Prop :=
  let MA := (0 - x, -1 - y)
  let AB := (x_b - 0, -3 - (-1))
  let MB := (x_b - x, -3 - y)
  let BA := (-x_b, 2)

  dot_product MA AB = dot_product MB BA

theorem trajectory_equation : ∀ x y, (∀ x_b, MB_parallel_OA x y x_b) → condition x y x_b → y = (1 / 4) * x^2 - 2 :=
by
  intros
  sorry

end trajectory_equation_l738_738462


namespace union_of_sets_l738_738424

theorem union_of_sets (a b : ℕ) (h : a = 3 ∧ b = 3) : 
  ({1, 2, a} ∪ {b, 2} = {1, 2, 3}) :=
by 
  intro; sorry

end union_of_sets_l738_738424


namespace area_of_square_plot_l738_738610

theorem area_of_square_plot (price_per_foot : ℕ) (total_cost : ℕ) (h_price : price_per_foot = 58) (h_cost : total_cost = 2088) :
  ∃ s : ℕ, s^2 = 81 := by
  sorry

end area_of_square_plot_l738_738610


namespace interval_of_monotonic_increase_l738_738909

theorem interval_of_monotonic_increase (k : ℤ) :
  let y := λ x : ℝ, 2 * sin (π / 3 - 2 * x) in
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → y x₁ ≤ y x₂ ↔ k * π + 5 * π / 12 ≤ x₁ ∧ x₂ ≤ k * π + 11 * π / 12) :=
sorry

end interval_of_monotonic_increase_l738_738909


namespace one_four_cell_piece_l738_738284

/--
  Consider a 7x7 checkered board. Prove that only one piece, 
  each composed of four cells, could have been used to form the board,
  given:
    1. The board is 7x7 in size.
    2. The board is checkered, alternating black and white cells.
    3. Figures used must cover exactly four cells.
    4. There are a total of 25 white cells and 24 black cells on the board.
-/
theorem one_four_cell_piece (h1 : ∃ b : ℕ → ℕ → Prop, 
                                (∀ i j, (b i j ↔ ((i + j) % 2 = 0)) ∧
                                         (b i j ↔ i < 7 ∧ j < 7)) ∧ 
                                ((∃ w b : ℕ, w = 25 ∧ b = 24) ∧ 
                                (w + b = 49)) ∧ 
                                 ∃ f : ℕ → Π(ℕ → ℕ → Prop), 
                                 (∀ i, f i = λ(i,j), ∃ p q : ℕ, (p,q) ∈ set.range (some_function 4))) : 
                                ∃ N : ℕ, N = 1 :=
by {
  sorry
}

end one_four_cell_piece_l738_738284


namespace hiker_speeds_l738_738214

theorem hiker_speeds:
  ∃ (d : ℝ), 
  (d > 5) ∧ ((70 / (d - 5)) = (110 / d)) ∧ (d - 5 = 8.75) :=
by
  sorry

end hiker_speeds_l738_738214


namespace count_multiples_3_or_4_not_12_l738_738432

theorem count_multiples_3_or_4_not_12 (start: ℕ) (finish: ℕ) (h1: start = 1) (h2: finish = 2005) :
  let multiples_3 := Nat.floor (finish / 3)
  let multiples_4 := Nat.floor (finish / 4)
  let multiples_12 := Nat.floor (finish / 12)
  multiples_3 + multiples_4 - multiples_12 = 1002 :=
by
  sorry

end count_multiples_3_or_4_not_12_l738_738432


namespace number_of_ways_to_choose_one_book_is_correct_l738_738927

-- Definitions of the given problem conditions
def number_of_chinese_books : Nat := 10
def number_of_english_books : Nat := 7
def number_of_math_books : Nat := 5

-- Theorem stating the proof problem
theorem number_of_ways_to_choose_one_book_is_correct : 
  number_of_chinese_books + number_of_english_books + number_of_math_books = 22 := by
  -- This proof is left as an exercise.
  sorry

end number_of_ways_to_choose_one_book_is_correct_l738_738927


namespace ways_to_divide_day_l738_738985

theorem ways_to_divide_day : 
  ∃ nm_count: ℕ, nm_count = 72 ∧ ∀ n m: ℕ, 0 < n ∧ 0 < m ∧ n * m = 72000 → 
  ∃ nm_pairs: ℕ, nm_pairs = 72 * 2 :=
sorry

end ways_to_divide_day_l738_738985


namespace find_c_d_l738_738726

-- Define the matrix and its properties
def matrix (c d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![4, -2],
    ![c, d]]

def is_inverse (c d : ℝ) : Prop :=
  (matrix c d) ⬝ (matrix c d) = (1 : Matrix (Fin 2) (Fin 2) ℝ)

theorem find_c_d : ∃ c d, is_inverse c d ∧ c = 7.5 ∧ d = -4 :=
by
  use 7.5, -4
  split
  sorry -- this will skip the proof

end find_c_d_l738_738726


namespace domain_of_sqrt_function_l738_738740

noncomputable def domain_of_function (x : ℝ) : Prop :=
  2 - sqrt (4 - sqrt (5 - x)) ≥ 0

theorem domain_of_sqrt_function :
  ∀ x : ℝ, domain_of_function x ↔ x ∈ set.Icc (-11 : ℝ) (5 : ℝ) := sorry

end domain_of_sqrt_function_l738_738740


namespace bob_walking_rate_is_12_l738_738878

-- Definitions for the problem
def yolanda_distance := 24
def yolanda_rate := 3
def bob_distance_when_met := 12
def time_yolanda_walked := 2

-- The theorem we need to prove
theorem bob_walking_rate_is_12 : 
  (bob_distance_when_met / (time_yolanda_walked - 1) = 12) :=
by sorry

end bob_walking_rate_is_12_l738_738878


namespace find_f_value_l738_738380

noncomputable def f (α : ℝ) : ℝ := 
  (Real.cos (π/2 + α) * Real.cos (π - α)) / Real.sin (π + α)

axiom α_in_third_quadrant (α : ℝ) : (3 * π / 2 < α) ∧ (α < 2 * π)

axiom cos_alpha_shift (α : ℝ) : Real.cos (α - 3 * π / 2) = 1 / 5

theorem find_f_value (α : ℝ) (h1 : α_in_third_quadrant α) (h2 : cos_alpha_shift α) : 
  f α = (2 * Real.sqrt 6) / 5 :=
by
  sorry

end find_f_value_l738_738380


namespace train_ride_time_in_hours_l738_738509

-- Definition of conditions
def lukes_total_trip_time_hours : ℕ := 8
def bus_ride_minutes : ℕ := 75
def walk_to_train_center_minutes : ℕ := 15
def wait_time_minutes : ℕ := 2 * walk_to_train_center_minutes

-- Convert total trip time to minutes
def lukes_total_trip_time_minutes : ℕ := lukes_total_trip_time_hours * 60

-- Calculate the total time spent on bus, walking, and waiting
def bus_walk_wait_time_minutes : ℕ :=
  bus_ride_minutes + walk_to_train_center_minutes + wait_time_minutes

-- Calculate the train ride time in minutes
def train_ride_time_minutes : ℕ :=
  lukes_total_trip_time_minutes - bus_walk_wait_time_minutes

-- Prove the train ride time in hours
theorem train_ride_time_in_hours : train_ride_time_minutes / 60 = 6 :=
by
  sorry

end train_ride_time_in_hours_l738_738509


namespace average_player_time_l738_738521

theorem average_player_time:
  let pg := 130
  let sg := 145
  let sf := 85
  let pf := 60
  let c := 180
  let total_secs := pg + sg + sf + pf + c
  let total_mins := total_secs / 60
  let num_players := 5
  let avg_mins_per_player := total_mins / num_players
  avg_mins_per_player = 2 :=
by
  sorry

end average_player_time_l738_738521


namespace river_length_GSA_AWRA_l738_738638

-- Define the main problem statement
noncomputable def river_length_estimate (GSA_length AWRA_length GSA_error AWRA_error error_prob : ℝ) : Prop :=
  (GSA_length = 402) ∧ (AWRA_length = 403) ∧ 
  (GSA_error = 0.5) ∧ (AWRA_error = 0.5) ∧ 
  (error_prob = 0.04) ∧ 
  (abs (402.5 - GSA_length) ≤ GSA_error) ∧ 
  (abs (402.5 - AWRA_length) ≤ AWRA_error) ∧ 
  (error_prob = 1 - (2 * 0.02))

-- The main theorem statement
theorem river_length_GSA_AWRA :
  river_length_estimate 402 403 0.5 0.5 0.04 :=
by
  sorry

end river_length_GSA_AWRA_l738_738638


namespace part_I_part_II_l738_738506

noncomputable def f (x : ℝ) : ℝ := (Real.log (1 + x)) - (2 * x) / (x + 2)
noncomputable def g (x : ℝ) : ℝ := f x - (4 / (x + 2))

theorem part_I (x : ℝ) (h₀ : 0 < x) : f x > 0 := sorry

theorem part_II (a : ℝ) (h : ∀ x, g x < x + a) : -2 < a := sorry

end part_I_part_II_l738_738506


namespace min_value_frac_inv_l738_738036

theorem min_value_frac_inv {x y : ℝ} (hx : x > 0) (hy : y > 0) (hxy : x + y = 2) : 
  (∃ m, (∀ x y, x > 0 ∧ y > 0 ∧ x + y = 2 → m ≤ (1 / x + 1 / y)) ∧ (m = 2)) :=
by
  sorry

end min_value_frac_inv_l738_738036


namespace find_a_l738_738420

def f (x : ℝ) (a : ℝ) : ℝ := Real.log (x^2 + a) / Real.log 2

theorem find_a (a : ℝ) (h : f 3 a = 1) : a = -7 :=
by
  sorry

end find_a_l738_738420


namespace exists_triangle_l738_738835

noncomputable def construct_triangle (A B C C0 C1 C2 : Point) (AC BC CC0 CC1 CC2 : ℝ) : Prop :=
  let midpoint := λ A B C0, dist A C0 = dist B C0
  let angle_bisector := λ C C1 A B, angle A C C1 = angle B C C1
  let external_bisector := λ C C2 A B, angle A C C2 + angle B C C2 = 180
  let length := λ P Q r, dist P Q = r
  (AC > BC) ∧
  (midpoint A B C0) ∧
  (angle_bisector C C1 A B) ∧
  (external_bisector C C2 A B) ∧
  (length C C0 CC0) ∧
  (length C C1 CC1) ∧
  (length C C2 CC2)

theorem exists_triangle
    (A B C C0 C1 C2 : Point)
    (AC BC CC0 CC1 CC2 : ℝ)
    (h_cond : construct_triangle A B C C0 C1 C2 AC BC CC0 CC1 CC2):
    ∃ Δ : Triangle, Δ = triangle.mk A B C :=
by sorry

end exists_triangle_l738_738835


namespace max_value_min_value_l738_738743

noncomputable def y (x : ℝ) : ℝ := 2 * Real.sin (3 * x + (Real.pi / 3))

theorem max_value (x : ℝ) : (∃ k : ℤ, x = (2 * k * Real.pi) / 3 + Real.pi / 18) ↔ y x = 2 :=
sorry

theorem min_value (x : ℝ) : (∃ k : ℤ, x = (2 * k * Real.pi) / 3 - 5 * Real.pi / 18) ↔ y x = -2 :=
sorry

end max_value_min_value_l738_738743


namespace product_of_imaginary_parts_of_roots_l738_738497

theorem product_of_imaginary_parts_of_roots :
  let i := Complex.I in
  let z1 := (-3 + Complex.sqrt (-7 + 28 * i)) / 2 in
  let z2 := (-3 - Complex.sqrt (-7 + 28 * i)) / 2 in
  Im z1 * Im z2 = -14 :=
by
  sorry

end product_of_imaginary_parts_of_roots_l738_738497


namespace expansion_rational_and_largest_l738_738041

noncomputable def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

theorem expansion_rational_and_largest:
  let x : ℚ := 2 -- Any positive rational value for x
  let sqrt_x := Real.sqrt (x.toReal)
  let k := 1 / (2 * 4 * x)
  let term (n r : ℕ) := (binomial_coeff n r) * (sqrt_x^(n-r)) * (k^r)
  let first_coeff := term 8 0 -- Coefficient for r = 0
  let second_coeff := term 8 1 -- Coefficient for r = 1
  let third_coeff := term 8 2 -- Coefficient for r = 2
  let rational_terms := [term 8 4, term 8 16, term 8 8]
  let largest_coeff_terms := [term 8 2, term 8 3]
  first_coeff = 1 ∧
  second_coeff = binomial_coeff 8 1 * (1/2) ∧
  third_coeff = binomial_coeff 8 2 * (1/4) * 2 ∧
  2 * second_coeff = first_coeff + third_coeff ∧
  rational_terms = [x^4, (35/8)*x, (1/(256*x^2)) ] ∧
  largest_coeff_terms = [7 * x^(5/2), 7 * x^(7/4)]
:=
by
  -- skip the proof
  sorry

end expansion_rational_and_largest_l738_738041


namespace find_m_l738_738790

noncomputable def f : ℝ → ℝ := cos

def x1 := π / 2
def x2 := 3 * π / 2
def x3 := 5 * π / 6
def x4 := 7 * π / 6

theorem find_m :
  x1 < x3 ∧ x3 < x4 ∧ x4 < x2 ∧
  (x1 + x2) / 2 = (x3 + x4) / 2 ∧
  (x2 - x1) / 3 = (x4 - x3) ∧
  f x3 = -sqrt 3 / 2 → 
  -sqrt 3 / 2 = cos (5 * π / 6) :=
sorry

end find_m_l738_738790


namespace find_a_plus_b_l738_738027

theorem find_a_plus_b (a b : ℝ) (x y : ℝ) 
  (h1 : x = 2) 
  (h2 : y = -1) 
  (h3 : a * x - 2 * y = 4) 
  (h4 : 3 * x + b * y = -7) : a + b = 14 := 
by 
  -- Begin the proof
  sorry

end find_a_plus_b_l738_738027


namespace train_length_l738_738629

noncomputable def length_of_each_train (L : ℝ) : Prop :=
  let v1 := 46 -- speed of faster train in km/hr
  let v2 := 36 -- speed of slower train in km/hr
  let relative_speed := (v1 - v2) * (5/18) -- converting relative speed to m/s
  let time := 72 -- time in seconds
  2 * L = relative_speed * time -- distance equation

theorem train_length : ∃ (L : ℝ), length_of_each_train L ∧ L = 100 :=
by
  use 100
  unfold length_of_each_train
  sorry

end train_length_l738_738629


namespace lake_side_length_l738_738482

theorem lake_side_length
  (swim_time_per_mile : Float := 20.0 / 60.0) -- Jake's swimming time in hours
  (row_time_total : Float := 10.0)            -- Total rowing time in hours
  (row_to_swim_ratio : Float := 2.0)          -- Rowing speed is twice the swimming speed
  (sides : Nat := 4)                          -- Number of sides of the lake (since it's square)
  : let swim_speed := 1.0 / swim_time_per_mile                        -- Swim speed in miles per hour
    let row_speed := row_to_swim_ratio * swim_speed                   -- Row speed in miles per hour
    let row_distance := row_speed * row_time_total                    -- Total rowing distance in miles
    let side_length := row_distance / sides                           -- Length of one side of the lake in miles
    side_length = 15 := by
  sorry

end lake_side_length_l738_738482


namespace find_skirts_l738_738958

variable (blouses : ℕ) (skirts : ℕ) (slacks : ℕ)
variable (blouses_in_hamper : ℕ) (slacks_in_hamper : ℕ) (skirts_in_hamper : ℕ)
variable (clothes_in_hamper : ℕ)

-- Given conditions
axiom h1 : blouses = 12
axiom h2 : slacks = 8
axiom h3 : blouses_in_hamper = (75 * blouses) / 100
axiom h4 : slacks_in_hamper = (25 * slacks) / 100
axiom h5 : skirts_in_hamper = 3
axiom h6 : clothes_in_hamper = blouses_in_hamper + slacks_in_hamper + skirts_in_hamper
axiom h7 : clothes_in_hamper = 11

-- Proof goal: proving the total number of skirts
theorem find_skirts : skirts_in_hamper = (50 * skirts) / 100 → skirts = 6 :=
by sorry

end find_skirts_l738_738958


namespace alice_commission_percentage_l738_738699

-- Definitions from the given problem
def basic_salary : ℝ := 240
def total_sales : ℝ := 2500
def savings : ℝ := 29
def savings_percentage : ℝ := 0.10

-- The target percentage we want to prove
def commission_percentage : ℝ := 0.02

-- The statement we aim to prove
theorem alice_commission_percentage :
  commission_percentage =
  (savings / savings_percentage - basic_salary) / total_sales := 
sorry

end alice_commission_percentage_l738_738699


namespace focus_of_ellipse_l738_738318

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def length (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / 2

def foci_distance (a b : ℝ) : ℝ :=
  Real.sqrt (a * a - b * b)

theorem focus_of_ellipse
  (A B C D : ℝ × ℝ)
  (hA : A = (3, 0)) (hB : B = (3, 8))
  (hC : C = (1, 4)) (hD : D = (5, 4)) :
  let center := midpoint A B in
  let a := ‖A.2 - B.2‖ / 2 in
  let b := ‖C.1 - D.1‖ / 2 in
  let c := foci_distance a b in
  let focus_y := center.2 + c in
  center = (3, 4) ∧ a = 4 ∧ b = 2 ∧ c = 2 * Real.sqrt 3 ∧ 
  focus_y = 4 + 2 * Real.sqrt 3 :=
by
  sorry

end focus_of_ellipse_l738_738318


namespace correct_equations_l738_738207

-- Defining the problem statement
theorem correct_equations (m n : ℕ) :
  (∀ (m n : ℕ), 40 * m + 10 = 43 * m + 1 ∧ 
   (n - 10) / 40 = (n - 1) / 43) :=
by
  sorry

end correct_equations_l738_738207


namespace perimeter_of_figure_n_l738_738109

theorem perimeter_of_figure_n (n : ℕ) (h1 : ∀ k : ℕ, k ≥ 1 → (perimeter (figure k)) = 60 + (k - 1) * 10) (h2 : perimeter (figure n) = 710) : n = 66 :=
by 
  sorry

end perimeter_of_figure_n_l738_738109


namespace equation_of_tangent_line_l738_738059

-- Definitions for the given conditions
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 2 * x
def P : ℝ × ℝ := (-1, 4)
def slope_of_tangent (a : ℝ) (x : ℝ) : ℝ := -6 * x^2 - 2

-- The main theorem to prove the equation of the tangent line
theorem equation_of_tangent_line (a : ℝ) (ha : f a (-1) = 4) :
  8 * x + y + 4 = 0 := by
  sorry

end equation_of_tangent_line_l738_738059


namespace product_of_solutions_eq_l738_738361

noncomputable def equation (x : ℝ) : ℝ :=
  2 * (Real.cos (2 * x)) * ((Real.cos (2 * x)) - (Real.cos (2016 * Real.pi^2 / x))) - (Real.cos (4 * x)) + 1

theorem product_of_solutions_eq :
  ∃ (S : Set ℝ), (∀ x ∈ S, equation x = 0 ∧ 0 < x) → (S.prod id) = 441 * Real.pi^4 :=
by
  sorry

end product_of_solutions_eq_l738_738361


namespace arithmetic_expression_eval_l738_738656

theorem arithmetic_expression_eval :
  ((26.3 * 12 * 20) / 3) + 125 = 2229 :=
sorry

end arithmetic_expression_eval_l738_738656


namespace sum_of_two_numbers_l738_738923

theorem sum_of_two_numbers (a b : ℝ) (h1 : a + b = 25) (h2 : a * b = 144) (h3 : |a - b| = 7) : a + b = 25 := 
  by
  sorry

end sum_of_two_numbers_l738_738923


namespace linear_dependence_k_l738_738722

theorem linear_dependence_k :
  ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ 
    (a * (2 : ℝ) + b * (5 : ℝ) = 0) ∧ 
    (a * (3 : ℝ) + b * k = 0) →
  k = 15 / 2 := by
  sorry

end linear_dependence_k_l738_738722


namespace john_paid_in_total_l738_738481

-- Define the cost of various items and discounts
def dress_shirt_cost := 25
def pants_cost := 35
def socks_cost := 10

def dress_shirt_discount := 0.15
def pants_discount := 0.20
def socks_discount := 0.10

def tax_rate := 0.10
def shipping_fee := 12.50

-- Quantities purchased
def num_dress_shirts := 4
def num_pants := 2
def num_socks := 3

-- Total cost calculation in Lean
def total_cost : ℝ :=
  let dress_shirts_total := num_dress_shirts * dress_shirt_cost
  let pants_total := num_pants * pants_cost
  let socks_total := num_socks * socks_cost

  let dress_shirts_discounted := dress_shirts_total * (1 - dress_shirt_discount)
  let pants_discounted := pants_total * (1 - pants_discount)
  let socks_discounted := socks_total * (1 - socks_discount)

  let subtotal := dress_shirts_discounted + pants_discounted + socks_discounted
  
  let tax := subtotal * tax_rate
  let total_before_shipping := subtotal + tax
  
  total_before_shipping + shipping_fee

theorem john_paid_in_total :
  total_cost = 197.30 :=
by
  -- The proof is omitted
  sorry

end john_paid_in_total_l738_738481


namespace minimum_handshakes_l738_738273

-- Definitions
def people : ℕ := 30
def handshakes_per_person : ℕ := 3

-- Theorem statement
theorem minimum_handshakes : (people * handshakes_per_person) / 2 = 45 :=
by
  sorry

end minimum_handshakes_l738_738273


namespace track_circumference_l738_738258

theorem track_circumference (x : ℕ) 
  (A_B_uniform_speeds_opposite : True) 
  (diametrically_opposite_start : True) 
  (same_start_time : True) 
  (first_meeting_B_150_yards : True) 
  (second_meeting_A_90_yards_before_complete_lap : True) : 
  2 * x = 720 :=
by
  sorry

end track_circumference_l738_738258


namespace power_sum_mod_inverse_l738_738236

theorem power_sum_mod_inverse (h : 3^6 ≡ 1 [MOD 17]) : 
  (3^(-1) + 3^(-2) + 3^(-3) + 3^(-4) +  3^(-5) + 3^(-6)) ≡ 1 [MOD 17] := 
by
  sorry

end power_sum_mod_inverse_l738_738236


namespace calculate_expression_l738_738328

theorem calculate_expression :
  sqrt 12 - (-1)^0 + abs (sqrt 3 - 1) = 3 * sqrt 3 - 2 := 
by
  sorry

end calculate_expression_l738_738328


namespace nested_sqrt_approaches_l738_738721

def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n+1) + fibonacci n

noncomputable def nested_sqrt_expression (n : ℕ) : ℝ :=
  let b := 2023 in
  let fib2k_sum (k : ℕ) : ℝ := real.sqrt (b * (fibonacci (2^k))^2) + sorry
  in sorry

theorem nested_sqrt_approaches (n : ℕ) :
  ∃ (a b c : ℕ), 
    (∀ (n : ℕ), ∃ r : ℝ, nested_sqrt_expression n = r ∧ r ≈ (a + real.sqrt b) / c) ∧
    a + b + c = 8102 ∧ 
    nat.gcd a c = 1 := sorry

end nested_sqrt_approaches_l738_738721


namespace part1_part2_l738_738496

noncomputable def f (a x : ℝ) : ℝ := a * x^2 + (2 - a) * x + a

-- Part 1
theorem part1 (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 1) ↔ a ≥ 2 * Real.sqrt 3 / 3 := sorry

-- Part 2
theorem part2 (a x : ℝ) : 
  (f a x < a + 2) ↔ 
    (a = 0 ∧ x < 1) ∨ 
    (a > 0 ∧ -2 / a < x ∧ x < 1) ∨ 
    (-2 < a ∧ a < 0 ∧ (x < 1 ∨ x > -2 / a)) ∨ 
    (a = -2) ∨ 
    (a < -2 ∧ (x < -2 / a ∨ x > 1)) := sorry

end part1_part2_l738_738496


namespace minimum_value_of_f_l738_738368

noncomputable def f (a : ℝ) := ∫ t in 0..1, abs (a * t + t * log t)

theorem minimum_value_of_f : ∃ (a : ℝ), 0 < a ∧ f a = f (log 2 / 2) :=
by
  sorry

end minimum_value_of_f_l738_738368


namespace possible_values_of_x_l738_738828

-- Definitions based on the conditions given
structure Square (A B C D: Type) :=
  (AB: real)
  (BC: real)
  (CD: real)
  (DA: real)
  (AB_eq_BC: AB = 2)
  (BC_eq_2: BC = 2)
  (CD_eq_AB: CD = AB)
  (DA_eq_BC: DA = BC)

structure Midpoint (E: Type) :=
  (AB: real)
  (E_mid_AB: E = AB / 2)

structure PointOnSide (F: Type) :=
  (BF: real)
  (BC: real)
  (F_on_BC: 0 < BF ∧ BF < BC)

axiom folding_condition (x: real) 
  (ABCD: Square) 
  (E: Midpoint) 
  (F: PointOnSide)  
  (x_cond: F.BF = x):
  0 < x ∧ x < 4 / 3

-- The proof problem statement
theorem possible_values_of_x (x: real) 
  (ABCD: Square) 
  (E: Midpoint) 
  (F: PointOnSide)  
  (x_cond: F.BF = x): 
  0 < x ∧ x < 4 / 3 := 
by sorry

end possible_values_of_x_l738_738828


namespace cylinder_height_one_third_of_cone_l738_738983

theorem cylinder_height_one_third_of_cone
  (A_baseCone A_baseCyl : ℝ) (V_cone V_cyl : ℝ) (h_cone h_cyl : ℝ)
  (h1 : A_baseCone = A_baseCyl)
  (h2 : V_cone = V_cyl)
  (h3 : V_cone = (1 / 3) * A_baseCone * h_cone)
  (h4 : V_cyl = A_baseCyl * h_cyl) :
  h_cyl = (1 / 3) * h_cone :=
begin
  -- Proof goes here ...
  sorry
end

end cylinder_height_one_third_of_cone_l738_738983


namespace maximum_rectangle_area_in_circle_l738_738716

theorem maximum_rectangle_area_in_circle (ω : ℝ → ℝ → Prop) (A B C D E : ℝ × ℝ) 
  (h1 : ω (A.fst, A.snd)) (h2 : ω (B.fst, B.snd)) (h3 : ω (C.fst, C.snd)) (h4 : ω (D.fst, D.snd))
  (h_int : ω (E.fst, E.snd)) 
  (h_AE : (A, E).fst - A.snd = 8) 
  (h_BE : (B, E).fst - B.snd = 2) 
  (h_CD : C.fst - D.fst + C.snd - D.snd = 10) 
  (h_AEC : E.fst - A.fst = E.fst - C.fst ∧ 
            E.snd - A.snd = E.snd - C.snd) : 
  ∃ R : ℝ, ∀ x y : ℝ, R = 6 * Real.sqrt 17 :=
begin
  use 6 * Real.sqrt 17,
  intros x y,
  sorry
end

end maximum_rectangle_area_in_circle_l738_738716


namespace threshold_mu_l738_738366

/-- 
Find threshold values μ₁₀₀ and μ₁₀₀₀₀₀ such that 
F = m * n * sin (π / m) * sqrt (1 / n² + sin⁴ (π / m)) 
is definitely greater than 100 and 1,000,000 respectively for all m greater than μ₁₀₀ and μ₁₀₀₀₀₀, 
assuming n = m³. -/
theorem threshold_mu : 
  (∃ (μ₁₀₀ μ₁₀₀₀₀₀ : ℝ), ∀ (m : ℝ), m > μ₁₀₀ → 
    m * (m ^ 3) * (Real.sin (Real.pi / m)) * 
      (Real.sqrt ((1 : ℝ) / (m ^ 6) + (Real.sin (Real.pi / m)) ^ 4)) > 100) ∧ 
  (∃ (μ₁₀₀₀₀₀ μ₁₀₀₀₀₀ : ℝ), ∀ (m : ℝ), m > μ₁₀₀₀₀₀ → 
    m * (m ^ 3) * (Real.sin (Real.pi / m)) * 
      (Real.sqrt ((1 : ℝ) / (m ^ 6) + (Real.sin (Real.pi / m)) ^ 4)) > 1000000) :=
sorry

end threshold_mu_l738_738366


namespace probability_divisible_by_4_l738_738676

def set_of_numbers := Finset.range (800 - 100 + 1) |>.map (λ n => n + 100)
def set_of_multiples_of_4 := set_of_numbers.filter (λ n => n % 4 == 0)

theorem probability_divisible_by_4 :
  (set_of_multiples_of_4.card : ℚ) / (set_of_numbers.card : ℚ) = 176 / 701 :=
by
-- We start with the range [100, 800]
have card_set_of_numbers : set_of_numbers.card = 701 :=
  by simp [set_of_numbers, Finset.range, Finset.card, Nat.sub_add_cancel (200 - 100)]

-- Calculate the number of elements divisible by 4
have card_set_of_multiples_of_4 : set_of_multiples_of_4.card = 176 :=
  by
    -- Arithmetic progression starting at 100, ending at 800 with common difference 4
    have : set_of_multiples_of_4 = Finset.filter (λ n => n % 4 == 0) (Finset.range (800 - 100 + 1) |>.map (λ n => n + 100)) :=
      by simp [set_of_multiples_of_4, set_of_numbers]
    rw this
    simp [Finset.filter_card, Finset.count, Finset.card_map, Finset.range, Nat.sub_add_cancel (800 - 100)]
    have count_multiples := (Nat.div_eq_iff_eq_mul_right.mpr (by norm_num : 4 ≠ 0)).2
    have smallest_multiple_by_n := 100 -- Smallest n / common multiple ... 
    sorry -- Calculation skipped, provided accurate steps here for guidance

-- Prove final probability
rw [card_set_of_numbers, card_set_of_multiples_of_4]
simp [Rat.div_def, Rat.mk_eq_div, Int.mul_by_distrib_left]
norm_num

end probability_divisible_by_4_l738_738676


namespace find_s_l738_738678

namespace Parallelogram

-- Define s and other parameters
variables (s : ℝ)

-- Definitions of the given conditions
def base := 3 * s
def side := s
def angle := 30
def area := 9 * real.sqrt 3

-- The height relationship using properties of 30-60-90 triangle
def height := side / 2

-- Statement translating the area condition
theorem find_s (h : 1/2 * base * side = area) : s = real.sqrt 6 * real.sqrt (real.sqrt 3) := sorry 

end Parallelogram

end find_s_l738_738678


namespace sequence_inequality_l738_738508

def A (m k : ℕ) : ℕ := Nat.choose m k * Nat.factorial k

def a (n : ℕ) : ℝ := 
  Real.sqrt (A (n + 2) 1 * Real.cbrt (A (n + 3) 2 * Real.crt4 (A (n + 4) 3 * Real.crt5 (A (n + 5) 4))))

theorem sequence_inequality (n : ℕ) (hn : n ≥ 1) : 
  a n < (119 / 120 : ℝ) * n + 7 / 3 :=
sorry

end sequence_inequality_l738_738508


namespace exists_set_X_l738_738862

variable (n k : ℕ)
variable (h1 : n ≥ k ∧ k ≥ 2)
variable (S : Fin n → Set ℤ)
variable (h2 : ∀ i, (S i).Nonempty)
variable (h3 : ∀ (t : Fin k → Fin n), (∃ i j, i ≠ j ∧ (S (t i) ∩ S (t j)).Nonempty))

theorem exists_set_X (n k : ℕ) (h1 : n ≥ k ∧ k ≥ 2)
  (S : Fin n → Set ℤ)
  (h2 : ∀ i, (S i).Nonempty)
  (h3 : ∀ (t : Fin k → Fin n), (∃ i j, i ≠ j ∧ (S (t i) ∩ S (t j)).Nonempty))
  :
  ∃ X : Set ℤ, X.Finite ∧ X.card = k - 1 ∧ ∀ i, ∃ x ∈ S i, x ∈ X := sorry

end exists_set_X_l738_738862


namespace independence_test_categorical_l738_738840

theorem independence_test_categorical (V1 V2 : Type) [Categorical V1 V2] : 
  relationship_check IndependenceTest V1 V2 = categorical_relationship := 
by sorry

end independence_test_categorical_l738_738840


namespace sum_cd_of_product_seq_eq_16_l738_738896

theorem sum_cd_of_product_seq_eq_16 (c d : ℕ) (h : (∏ i in Finset.range (c - 3), (i + 4 : ℚ) / (i + 3)) = (16 : ℚ)) : c + d = 95 :=
sorry

end sum_cd_of_product_seq_eq_16_l738_738896


namespace impossible_to_arrange_parallelepipeds_l738_738117

theorem impossible_to_arrange_parallelepipeds (P : Fin 12 → Set (ℝ × ℝ × ℝ))
  (h_parallel: ∀ i, (∀ (x y z: ℝ × ℝ × ℝ), (x ∈ P i → ∃ a b c d e f, x = (a, b, c) ∧ 
                                            y ∈ P i → y = (d, e, f) → a = d ∨ b = e ∨ c = f)))
  (h_intersect: ∀ i, ∀ j ∈ ({set.range Finset.univ} : finset (finset ℕ)).to_Set, 
    (if j = i-1 mod 12 ∨ j = i+1 mod 12 then P i ∩ P j = ∅ else P i ∩ P j ≠ ∅)) :
  false :=
  sorry

end impossible_to_arrange_parallelepipeds_l738_738117


namespace exists_lines_with_1985_intersections_l738_738256

theorem exists_lines_with_1985_intersections :
  ∃ (lines : Fin 100 → AffinePlane ℝ), (∃ (pts : Set (AffinePlane ℝ)), pts = ⋃ (i j : Fin 100), i ≠ j -> intersection_of lines[i] lines[j] ∧ |pts| = 1985) :=
sorry

end exists_lines_with_1985_intersections_l738_738256


namespace extend_staircase_toothpicks_l738_738320

theorem extend_staircase_toothpicks :
  (4 -> 5 needs 12) ∧ (every step requires 2 more) ∧ ∀n m, (n -> m needs p) → Proof that (4 -> 6 needs 26)

end extend_staircase_toothpicks_l738_738320


namespace polygon_sides_l738_738048

theorem polygon_sides (h : ∑_angles = 1080) : sides = 8 := 
  sorry

end polygon_sides_l738_738048


namespace predicted_value_y_at_x_5_l738_738167

noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

theorem predicted_value_y_at_x_5 :
  let x_values := [-2, -1, 0, 1, 2]
  let y_values := [5, 4, 2, 2, 1]
  let x_bar := mean x_values
  let y_bar := mean y_values
  let a_hat := y_bar
  (∀ x, y = -x + a_hat) →
  (x = 5 → y = -2.2) :=
by
  sorry

end predicted_value_y_at_x_5_l738_738167


namespace total_socks_l738_738122

-- Definitions based on conditions
def red_pairs : ℕ := 20
def red_socks : ℕ := red_pairs * 2
def black_socks : ℕ := red_socks / 2
def white_socks : ℕ := 2 * (red_socks + black_socks)

-- The main theorem we want to prove
theorem total_socks :
  (red_socks + black_socks + white_socks) = 180 := by
  sorry

end total_socks_l738_738122


namespace EricBenJackMoneySum_l738_738734

noncomputable def EricBenJackTotal (E B J : ℕ) :=
  (E + B + J : ℕ)

theorem EricBenJackMoneySum :
  ∀ (E B J : ℕ), (E = B - 10) → (B = J - 9) → (J = 26) → (EricBenJackTotal E B J) = 50 :=
by
  intros E B J
  intro hE hB hJ
  rw [hJ] at hB
  rw [hB] at hE
  sorry

end EricBenJackMoneySum_l738_738734


namespace simplify_and_find_ratio_l738_738540

theorem simplify_and_find_ratio (k : ℤ) : (∃ (c d : ℤ), (∀ x y : ℤ, c = 1 ∧ d = 2 ∧ x = c ∧ y = d → ((6 * k + 12) / 6 = k + 2) ∧ (c / d = 1 / 2))) :=
by
  use 1
  use 2
  sorry

end simplify_and_find_ratio_l738_738540


namespace sum_of_inverses_mod_17_l738_738230

theorem sum_of_inverses_mod_17 :
  (3⁻¹ + 3⁻² + 3⁻³ + 3⁻⁴ + 3⁻⁵ + 3⁻⁶ : ℤ) % 17 = 7 :=
by {
  sorry
}

end sum_of_inverses_mod_17_l738_738230


namespace loss_percentage_is_10_l738_738663

-- Define the conditions
def cost_price (CP : ℝ) : Prop :=
  (550 : ℝ) = 1.1 * CP

def selling_price (SP : ℝ) : Prop :=
  SP = 450

-- Define the main proof statement
theorem loss_percentage_is_10 (CP SP : ℝ) (HCP : cost_price CP) (HSP : selling_price SP) :
  ((CP - SP) / CP) * 100 = 10 :=
by
  -- Translation of the condition into Lean statement
  sorry

end loss_percentage_is_10_l738_738663


namespace time_saved_is_zero_l738_738551

/-- Ted's grandfather's treadmill usage --/
def treadmill_usage : ℕ → ℤ
| 0 := 1 / 6 -- Monday
| 1 := 1 / 2 -- Tuesday
| 2 := 1 / 3 -- Thursday
| 3 := 1     -- Friday
| _ := 0

/-- The total distance walked --/
def total_distance : ℤ := 1 + 2 + 1 + 2

/-- The total time spent walking --/
def total_time : ℤ := treadmill_usage 0 + treadmill_usage 1 + treadmill_usage 2 + treadmill_usage 3

/-- The time if walked at 3 mph everyday --/
def time_at_3mph : ℤ := total_distance / 3

/-- The mathematically equivalent problem statement --/
theorem time_saved_is_zero : (total_time - time_at_3mph) * 60 = 0 := by
  sorry

end time_saved_is_zero_l738_738551


namespace sampling_method_correct_l738_738299

-- Define the condition in which there is an inspection every 10 minutes.
def interval_based_inspection (inspect : ℕ → bool) : Prop :=
  ∀ n : ℕ, inspect (10 * n)

-- Define what it means to be a systematic sampling
def is_systematic_sampling (inspect : ℕ → bool) : Prop :=
  interval_based_inspection inspect

-- Main theorem statement based on the problem and solution
theorem sampling_method_correct (inspect : ℕ → bool) :
  interval_based_inspection inspect → is_systematic_sampling inspect := by
  intros h
  exact h
  sorry -- proof skipped

end sampling_method_correct_l738_738299


namespace exists_p_and_q_l738_738490

open Real

theorem exists_p_and_q
  (n : ℕ) (h_n_pos : n > 0)
  (x : Fin (2 * n) → ℝ)
  (h_x_nonneg : ∀ i, 0 ≤ x i)
  (h_sum_x : (Finset.univ.sum x) = 4) :
  ∃ (p q : ℕ), (0 ≤ q ∧ q ≤ n - 1) ∧
    (Finset.range (q + 1)).sum (λ i, x ⟨(p + 2 * i - 1) % (2 * n), by linarith [mul_pos (zero_lt_two) h_n_pos]⟩) ≤ 1 ∧
    (Finset.range (n - q - 1)).sum (λ i, x ⟨(p + 2 * (q + 1 + i)) % (2 * n), by linarith [mul_pos (zero_lt_two) h_n_pos]⟩) ≤ 1 :=
sorry

end exists_p_and_q_l738_738490


namespace trapezoid_area_correct_l738_738471

-- Define the conditions
variables {P Q R S : Type} [affine.triangle P Q R] [affine.triangle Q R S] [affine.triangle S P Q] [affine.triangle R S P]
variables (area_ADE : ℝ) (area_BCE : ℝ)

-- Given the areas of triangles ADE and BCE
axiom area_ADE_eq : area_ADE = 12
axiom area_BCE_eq : area_BCE = 3

-- Define the trapezoid area in terms of the given triangles
noncomputable def trapezoid_area (area_ADE area_BCE : ℝ) : ℝ :=
area_ADE + area_BCE

-- Prove the total area of the trapezoid is 27
theorem trapezoid_area_correct :
  trapezoid_area 12 3 = 27 :=
by sorry

end trapezoid_area_correct_l738_738471


namespace find_a_l738_738404

open Real

def point_in_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 6 * x + 4 * y + 4 = 0

def line_equation (x y : ℝ) : Prop :=
  x + 2 * y - 3 = 0

theorem find_a (a : ℝ) :
  point_in_circle 1 a →
  line_equation 1 a →
  a = -2 :=
by
  intro h1 h2
  sorry

end find_a_l738_738404


namespace solve_for_x_l738_738174

theorem solve_for_x (x : ℝ) : 
  x^2 - 2 * x - 8 = -(x + 2) * (x - 6) → (x = 5 ∨ x = -2) :=
by
  intro h
  sorry

end solve_for_x_l738_738174


namespace max_n_monic_quadratic_l738_738340

theorem max_n_monic_quadratic 
  (p1 p2 p3 : ℤ → ℤ)
  (h1 : ∀ x, p1 x = x ^ 2 + a1 * x + b1)
  (h2 : ∀ x, p2 x = x ^ 2 + a2 * x + b2)
  (h3 : ∀ x, p3 x = x ^ 2 + a3 * x + b3)
  (monic1 : a1 = 1) 
  (monic2 : a2 = 1) 
  (monic3 : a3 = 1) 
  (coeff_integers : a1 ∈ ℤ ∧ a2 ∈ ℤ ∧ a3 ∈ ℤ ∧ b1 ∈ ℤ ∧ b2 ∈ ℤ ∧ b3 ∈ ℤ) 
  (cover_range : ∀ i ∈ (finset.range 10), ∃ j ∈ {1, 2, 3}, ∃ m ∈ ℤ, 
  if j = 1 then p1 m = i + 1 else if j = 2 then p2 m = i + 1 else p3 m = i + 1) :
  (∀ n, n ≤ 9) :=
begin
  sorry
end

end max_n_monic_quadratic_l738_738340


namespace loss_percentage_25_l738_738660

variable (C S : ℝ)
variable (h : 15 * C = 20 * S)

theorem loss_percentage_25 (h : 15 * C = 20 * S) : (C - S) / C * 100 = 25 := by
  sorry

end loss_percentage_25_l738_738660


namespace find_f4_l738_738044

-- Let f be a function from ℝ to ℝ with the following properties:
variable (f : ℝ → ℝ)

-- 1. f(x + 1) is an odd function
axiom f_odd : ∀ x, f (-(x + 1)) = -f (x + 1)

-- 2. f(x - 1) is an even function
axiom f_even : ∀ x, f (-(x - 1)) = f (x - 1)

-- 3. f(0) = 2
axiom f_zero : f 0 = 2

-- Prove that f(4) = -2
theorem find_f4 : f 4 = -2 :=
by
  sorry

end find_f4_l738_738044


namespace abc_sitting_together_probability_l738_738252

-- Definitions and conditions
def favourable_arrangements : ℕ := Nat.factorial 6 * Nat.factorial 3
def total_arrangements : ℕ := Nat.factorial 8
def probability : ℚ := favourable_arrangements / total_arrangements

-- Theorem: Prove that the probability of a, b, c sitting together is 1/9.375
theorem abc_sitting_together_probability : probability = 1 / 9.375 := by
  unfold favourable_arrangements
  unfold total_arrangements
  unfold probability
  sorry

end abc_sitting_together_probability_l738_738252


namespace cone_lateral_area_l738_738045

def is_embed_main_view_isosceles_triangle
  (main_view : triangle) 
  (base_length height : ℝ) : Prop := 
  main_view.isosceles ∧ main_view.base_length = base_length ∧ main_view.height = height 

theorem cone_lateral_area 
  (main_view : triangle) 
  (base_length height : ℝ) 
  (h : is_embed_main_view_isosceles_triangle main_view base_length height) : 
  lateral_area_of_the_cone = 60 * π := 
sorry

end cone_lateral_area_l738_738045


namespace exists_n_gt_2_divisible_by_1991_l738_738170

theorem exists_n_gt_2_divisible_by_1991 :
  ∃ n > 2, 1991 ∣ (2 * 10^(n+1) - 9) :=
by
  existsi (1799 : Nat)
  have h1 : 1799 > 2 := by decide
  have h2 : 1991 ∣ (2 * 10^(1799+1) - 9) := sorry
  constructor
  · exact h1
  · exact h2

end exists_n_gt_2_divisible_by_1991_l738_738170


namespace repair_cost_l738_738165

theorem repair_cost (purchase_price transport_cost sale_price : ℝ) (profit_percentage : ℝ) (repair_cost : ℝ) :
  purchase_price = 14000 →
  transport_cost = 1000 →
  sale_price = 30000 →
  profit_percentage = 50 →
  sale_price = (1 + profit_percentage / 100) * (purchase_price + repair_cost + transport_cost) →
  repair_cost = 5000 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end repair_cost_l738_738165


namespace magnitude_one_plus_ai_is_sqrt_five_l738_738032

variable (a : ℝ)

def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0

theorem magnitude_one_plus_ai_is_sqrt_five (h : is_pure_imaginary ((a - 2 * complex.I) / (1 + complex.I))) :
  abs (1 + (a : ℂ) * complex.I) = √5 :=
by
  sorry

end magnitude_one_plus_ai_is_sqrt_five_l738_738032


namespace specially_balanced_int_count_l738_738672

theorem specially_balanced_int_count : 
  { n : ℕ // 1000 ≤ n ∧ n ≤ 9999 ∧ 
                 ∃ a b c d : ℕ, 
                 a * 1000 + b * 100 + c * 10 + d = n ∧ 
                 a + b = c + d + 1 } = 540 := 
by sorry

end specially_balanced_int_count_l738_738672


namespace find_a_l738_738815

noncomputable def pure_imaginary (z : ℂ) : Prop := z.re = 0

theorem find_a (a : ℝ) 
  (h : pure_imaginary ((a + complex.I) / (1 + 2 * complex.I))) : a = -2 := 
by
  sorry

end find_a_l738_738815


namespace number_of_games_l738_738967

theorem number_of_games (n : ℕ) (h : n = 15) : ∃ (games : ℕ), games = n * (n - 1) / 2 ∧ games = 105 :=
by
  use 105
  split
  · rw [h]
    exact calc
      15 * (15 - 1) / 2 = 15 * 14 / 2 : by rfl
      ... = 210 / 2               : by rfl
      ... = 105                   : by rfl 
  · rfl

end number_of_games_l738_738967


namespace function_represents_y_eq_x_l738_738249

theorem function_represents_y_eq_x (x : ℝ) : x ≠ 0 → (x^2 / x = x) := by 
  intro hx
  -- Simplifies to x when x is not equal to zero
  have : x^2 / x = x, from div_self hx
  exact this

end function_represents_y_eq_x_l738_738249


namespace number_of_M_partitions_l738_738140

open Finset

-- Define sets and conditions
def A := (range 2002).image (λ x, x + 1)
def M := {1001, 2003, 3005}

def M_free (B : Finset ℕ) : Prop :=
  ∀ m n ∈ B, m + n ∉ M

def M_partition (A1 A2 : Finset ℕ) : Prop :=
  A1 ∪ A2 = A ∧ A1 ∩ A2 = ∅ ∧ M_free A1 ∧ M_free A2

-- Define the main theorem
theorem number_of_M_partitions : 
  (Finset.filter (λ (A1 : Finset ℕ), ∃ A2, M_partition A1 A2) (powerset A)).card = 2^501 :=
sorry

end number_of_M_partitions_l738_738140


namespace g_at_neg_two_g_at_three_l738_738504

def g (x : ℝ) : ℝ :=
  if x < 0 then 2 * x - 4 else 5 - 3 * x

theorem g_at_neg_two : g (-2) = -8 :=
by
  sorry

theorem g_at_three : g (3) = -4 :=
by
  sorry

end g_at_neg_two_g_at_three_l738_738504


namespace min_chord_length_l738_738383

open Real

theorem min_chord_length (m : ℝ) :
  let C := ∀ x y : ℝ, (x - 1)^2 + (y - 2)^2 = 25
  let l := ∀ x y : ℝ, (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0 in
  ∃ A B : Real × Real, 
  (C A.fst A.snd ∧ C B.fst B.snd ∧ l A.fst A.snd ∧ l B.fst B.snd) →
  dist A B = 4 * sqrt 5 := 
sorry

end min_chord_length_l738_738383


namespace arithmetic_sequence_weighted_sum_l738_738750

variable (a : ℕ → ℕ)
variable (f : ℕ → ℕ)
variable (n : ℕ)
hypothesis (H1 : a₂ : ℕ → ℕ → ℕ)
  (Ha : ∀ (n : ℕ), f(a n) = 2 / (2 - a n) ∧ a n ≠ 2)
  (Hsn : ∀ (n : ℕ), S n = 1/4 * (3 - 2 / f(a n)) ^ 2)

theorem arithmetic_sequence (H: ∀ n, Ha n) : ∃ a₁ d, ∀ n, a (n + 1) = a n + d :=
sorry

variable (b : ℕ → ℝ)
variable (T : ℕ → ℝ)
hypothesis (Hb : ∀ (n : ℕ), b n = a n / 2 ^ n)
hypothesis (Ht : ∀ (n : ℕ), T n = ∑ k in range n, b k)
include Ha Hb

theorem weighted_sum (H: ∀ n, Ha n) : ∀ n, T n = 3 - (2 * n + 3) / 2 ^ n :=
sorry

end arithmetic_sequence_weighted_sum_l738_738750


namespace triangle_possible_sides_l738_738946

theorem triangle_possible_sides (a b c : ℕ) (h₁ : a + b + c = 7) (h₂ : a + b > c) (h₃ : a + c > b) (h₄ : b + c > a) :
  a = 1 ∨ a = 2 ∨ a = 3 :=
by {
  sorry
}

end triangle_possible_sides_l738_738946


namespace consumption_percentage_l738_738683

open Real

theorem consumption_percentage (x y : ℝ) (h : y = 0.66 * x + 1.562) (h_y : y = 7.675) :
  (7.675 / x * 100 ≈ 83) :=
by
  sorry

end consumption_percentage_l738_738683


namespace f_value_at_2_l738_738017

noncomputable def f : ℝ → ℝ := sorry

theorem f_value_at_2:
  (∀ x, f(-x) + (-x)^2 = -(f(x) + x^2)) →
  (∀ x, f(-x) + (-x)^3 = f(x) + x^3) →
  f 2 = -12 :=
by
  intros h1 h2
  sorry

end f_value_at_2_l738_738017


namespace smallest_number_increased_by_three_divisible_l738_738628

theorem smallest_number_increased_by_three_divisible (n : ℕ) :
  (∃ n : ℕ, (∀ d ∈ {510, 4590, 105}, d ∣ (n + 3)) ∧ n = 32127) :=
sorry

end smallest_number_increased_by_three_divisible_l738_738628


namespace triangle_inradius_circumradius_ratio_l738_738606

theorem triangle_inradius_circumradius_ratio (T : Triangle)
  (ρ : ℝ) (r : ℝ)
  (hρ : T.inscribed_circle.radius = ρ)
  (hr : T.circumscribed_circle.radius = r) :
  0 < ρ / r ∧ ρ / r ≤ 1 / 2 ∧ (ρ / r = 1 / 2 ↔ T.is_equilateral) :=
sorry

end triangle_inradius_circumradius_ratio_l738_738606


namespace fran_speed_same_distance_l738_738479

theorem fran_speed_same_distance (joann_speed : ℝ) (joann_time : ℝ) (fran_time : ℝ) : 
  joann_speed = 12 → joann_time = 3.5 → fran_time = 3 → 
  ∃ fran_speed : ℝ, fran_speed = 14 :=
by
  intros h_joann_speed h_joann_time h_fran_time
  let joann_distance := joann_speed * joann_time
  have h_joann_distance : joann_distance = 42 :=
    by rw [h_joann_speed, h_joann_time]; norm_num
  let fran_speed := joann_distance / fran_time
  have h_fran_speed : fran_speed = 14 :=
    by rw [←h_fran_time, h_joann_distance]; norm_num
  use fran_speed
  exact h_fran_speed

end fran_speed_same_distance_l738_738479


namespace maximum_angle_B_in_triangle_l738_738654

theorem maximum_angle_B_in_triangle
  (A B C M : ℝ × ℝ)
  (hM : midpoint ℝ A B = M)
  (h_angle_MAC : ∃ angle_MAC : ℝ, angle_MAC = 15) :
  ∃ angle_B : ℝ, angle_B = 105 := 
by
  sorry

end maximum_angle_B_in_triangle_l738_738654


namespace middle_part_division_l738_738728

theorem middle_part_division 
  (x : ℝ) 
  (x_pos : x > 0) 
  (H : x + (1 / 4) * x + (1 / 8) * x = 96) :
  (1 / 4) * x = 17 + 21 / 44 :=
by
  sorry

end middle_part_division_l738_738728


namespace avg_writing_speed_l738_738696

theorem avg_writing_speed 
  (words1 hours1 words2 hours2 : ℕ)
  (h_words1 : words1 = 30000)
  (h_hours1 : hours1 = 60)
  (h_words2 : words2 = 50000)
  (h_hours2 : hours2 = 100) :
  (words1 + words2) / (hours1 + hours2) = 500 :=
by {
  sorry
}

end avg_writing_speed_l738_738696


namespace kindergarten_classes_l738_738461

theorem kindergarten_classes :
  ∃ (j a m : ℕ), j + a + m = 32 ∧
                  j > 0 ∧ a > 0 ∧ m > 0 ∧
                  j / 2 + a / 4 + m / 8 = 6 ∧
                  (j = 4 ∧ a = 4 ∧ m = 24) :=
by {
  sorry
}

end kindergarten_classes_l738_738461


namespace problem_part1_problem_part2_l738_738069

def A : Set ℝ := { x | 3 ≤ x ∧ x ≤ 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }
def CR_A : Set ℝ := { x | x < 3 ∨ x > 7 }

theorem problem_part1 : A ∪ B = { x | 3 ≤ x ∧ x ≤ 7 } := by
  sorry

theorem problem_part2 : (CR_A ∩ B) = { x | (2 < x ∧ x < 3) ∨ (7 < x ∧ x < 10) } := by
  sorry

end problem_part1_problem_part2_l738_738069


namespace cos_sub_angle_l738_738028

theorem cos_sub_angle (α : ℝ) (h1 : cos α = -4 / 5) (h2 : π / 2 < α ∧ α < π) :
  cos (π / 6 - α) = (3 - 4 * Real.sqrt 3) / 10 ∧ cos (π / 6 + α) = -(3 + 4 * Real.sqrt 3) / 10 :=
by
  sorry

end cos_sub_angle_l738_738028


namespace find_m_min_value_l738_738060

theorem find_m_min_value (m : ℝ) (f : ℝ → ℝ) (h_def : ∀ x ∈ Icc 1 2, f x = Real.exp x - m / x)
(h_min : ∃ x ∈ Icc 1 2, f x = 1) : m = Real.exp 1 - 1 := 
sorry

end find_m_min_value_l738_738060


namespace total_students_in_class_l738_738667

variables (num_cafeteria : ℕ) (num_no_lunch : ℕ)
variables (num_bring_lunch : ℕ := 3 * num_cafeteria)
noncomputable def total_students := num_cafeteria + num_bring_lunch + num_no_lunch

theorem total_students_in_class :
  num_cafeteria = 10 → num_no_lunch = 20 → total_students num_cafeteria num_no_lunch = 60 :=
by
  intros h_cafeteria h_no_lunch
  have h_bring_lunch : num_bring_lunch = 3 * 10 := by simp [h_cafeteria]
  have h_total : total_students num_cafeteria num_no_lunch = 10 + (3 * 10) + 20 := by simp [h_cafeteria, h_bring_lunch]
  simp [h_total]
  sorry

end total_students_in_class_l738_738667


namespace triangle_area_l738_738560

-- Define the curve function
def curve (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 3*x^2 - 6*x

-- Define the point of tangency
def point_of_tangency : ℝ × ℝ := (1, -1)

-- Define the tangent line given the point and the slope
def tangent_line (x : ℝ) : ℝ := -3*x + 2

-- Prove that the area of the triangle formed by the tangent line and coordinate axes is 2/3
theorem triangle_area :
  let p₁ := (0, tangent_line 0),
      p₂ := (tangent_line.symm 0, 0) in
  (1 / 2) * (fst p₁) * (snd p₁) = 2 / 3 := sorry

end triangle_area_l738_738560


namespace remainder_2519_div_7_l738_738303

theorem remainder_2519_div_7 : 2519 % 7 = 6 :=
by
  sorry

end remainder_2519_div_7_l738_738303


namespace decimal_expansion_repeats_l738_738538

/-- The decimal expansion of a rational number must eventually repeat. -/
theorem decimal_expansion_repeats (a b : ℕ) (h_b : b ≠ 0) (h_a : a < b) :
  ∃ (n₀ k : ℕ), ∀ n ≥ n₀, decimal_digit a b (n + k) = decimal_digit a b n := 
sorry

end decimal_expansion_repeats_l738_738538


namespace min_erase_factors_to_no_real_solutions_l738_738972

theorem min_erase_factors_to_no_real_solutions :
  ∃ k : ℕ, (∀ (x : real),
    (∀ f : ℕ → ℤ,
    (f 0 = (x - 1)) ∧
    (f 1 = (x - 2)) ∧
    (f 2015 = (x - 2016))) →
    (∀ g : ℕ → ℤ,
    (g 0 = (x - 1)) ∧
    (g 1 = (x - 2)) ∧
    (g 2015 = (x - 2016))) →
    ((∃ m n : ℕ, (m < k ∧ n < k) ∧ (f m ≠ 0) ∧ (g n ≠ 0))) ∧
     ((f k ≠ g k) → False)) → k = 2016 :=
by sorry

end min_erase_factors_to_no_real_solutions_l738_738972


namespace river_length_GSA_AWRA_l738_738642

-- Define the main problem statement
noncomputable def river_length_estimate (GSA_length AWRA_length GSA_error AWRA_error error_prob : ℝ) : Prop :=
  (GSA_length = 402) ∧ (AWRA_length = 403) ∧ 
  (GSA_error = 0.5) ∧ (AWRA_error = 0.5) ∧ 
  (error_prob = 0.04) ∧ 
  (abs (402.5 - GSA_length) ≤ GSA_error) ∧ 
  (abs (402.5 - AWRA_length) ≤ AWRA_error) ∧ 
  (error_prob = 1 - (2 * 0.02))

-- The main theorem statement
theorem river_length_GSA_AWRA :
  river_length_estimate 402 403 0.5 0.5 0.04 :=
by
  sorry

end river_length_GSA_AWRA_l738_738642


namespace prove_fn_eq_n_l738_738861

theorem prove_fn_eq_n (f : ℕ → ℕ) (h : ∀ n : ℕ, f(n + 1) > f(f(n))) : ∀ n : ℕ, f(n) = n :=
by
  sorry

end prove_fn_eq_n_l738_738861


namespace proportion_odd_smallest_element_l738_738765

-- Define the problem statement
theorem proportion_odd_smallest_element (n : ℕ) (h_pos : n > 0) :
  (∑ i in (finset.range (2 * n + 1)).filter (λ x, x % 2 = 1), 2^(2 * n - i)) 
  / ((∑ i in (finset.range (2 * n + 1)).filter (λ x, x % 2 = 1), 2^(2 * n - i)) 
  + (∑ i in (finset.range (2 * n + 1)).filter (λ x, x % 2 = 0), 2^(2 * n - i))) 
  = (2 / 3: ℚ) := 
sorry

end proportion_odd_smallest_element_l738_738765


namespace accurate_river_length_l738_738648

-- Define the given conditions
def length_GSA := 402
def length_AWRA := 403
def error_margin := 0.5
def probability_of_error := 0.04

-- State the theorem based on these conditions
theorem accurate_river_length : 
  ∀ Length_GSA Length_AWRA error_margin probability_of_error, 
  Length_GSA = 402 → 
  Length_AWRA = 403 → 
  error_margin = 0.5 → 
  probability_of_error = 0.04 → 
  (this based on independent measurements with above error margins)
  combined_length = 402.5 ∧ combined_probability_of_error = 0.04 :=
by 
  -- Proof to be completed
  sorry

end accurate_river_length_l738_738648


namespace seq_inequality_l738_738485

noncomputable def sequence_of_nonneg_reals (a : ℕ → ℝ) : Prop :=
  ∀ m n : ℕ, a (n + m) ≤ a n + a m

theorem seq_inequality
  (a : ℕ → ℝ)
  (h : sequence_of_nonneg_reals a)
  (h_nonneg : ∀ n, 0 ≤ a n) :
  ∀ n m : ℕ, m > 0 → n ≥ m → a n ≤ m * a 1 + ((n / m) - 1) * a m := 
by
  sorry

end seq_inequality_l738_738485


namespace both_miss_probability_l738_738157

-- Define the probabilities of hitting the target for Persons A and B 
def prob_hit_A : ℝ := 0.85
def prob_hit_B : ℝ := 0.8

-- Calculate the probabilities of missing the target
def prob_miss_A : ℝ := 1 - prob_hit_A
def prob_miss_B : ℝ := 1 - prob_hit_B

-- Prove that the probability of both missing the target is 0.03
theorem both_miss_probability : prob_miss_A * prob_miss_B = 0.03 :=
by
  sorry

end both_miss_probability_l738_738157


namespace tangent_circle_centers_form_sphere_l738_738719

noncomputable def proof_problem (σ : Type _) (S : σ) (r h : ℝ) (π : Type _) (N : π) :=
  ∃ ω : σ, locus_is_sphere_with_diameter ω (r^2 / h)

def locus_is_sphere_with_diameter (ω : Type _) (d : ℝ) :=
  sorry  -- To be filled in with the full definition of what it means for ω to be a sphere with diameter d

theorem tangent_circle_centers_form_sphere
    {σ π : Type _} [metric_space σ] [metric_space π]
    (S : σ) (r h : ℝ) (N : π) :
  proof_problem σ S r h π N :=
sorry  -- To be proved

end tangent_circle_centers_form_sphere_l738_738719


namespace proof_problem_l738_738019

variable {f : ℝ → ℝ}
variable (a b : ℝ)

theorem proof_problem (c : ℝ) (h₀ : 0 < c) (h₁ : f c >= 0) (h₂ : differentiable ℝ f) (h₃ : ∀ x > 0, x * (derivative f x) + f x ≤ -x^2 + x - 1) (h₄ : 0 < a) (h₅ : a < b) :
  a * f b ≤ b * f a :=
sorry

end proof_problem_l738_738019


namespace intersection_elements_0_or_1_l738_738018

theorem intersection_elements_0_or_1 (f : ℝ → ℝ) (a b : ℝ) (h : a ≤ b) :
  ∃ n : ℕ, n = 0 ∨ n = 1 ∧ n = (∃ y,  (a ≤ 0 ∧ 0 ≤ b) ∧ y = f 0) :=
by sorry

end intersection_elements_0_or_1_l738_738018


namespace number_of_primes_in_first_15_terms_prime_sequence_l738_738545

def prime_sequence (n : ℕ) : ℕ :=
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53] in
  primes.take n.foldl (+) 0

theorem number_of_primes_in_first_15_terms_prime_sequence : 
  let sequence := list.map prime_sequence (list.range 15)
  in list.countp is_prime sequence = 2 := sorry

end number_of_primes_in_first_15_terms_prime_sequence_l738_738545


namespace triangle_base_length_l738_738555

theorem triangle_base_length (base : ℝ) (h1 : ∃ (side : ℝ), side = 6 ∧ (side^2 = (base * 12) / 2)) : base = 6 :=
sorry

end triangle_base_length_l738_738555


namespace number_of_books_from_second_shop_l738_738534

theorem number_of_books_from_second_shop (books_first_shop : ℕ) (cost_first_shop : ℕ)
    (books_second_shop : ℕ) (cost_second_shop : ℕ) (average_price : ℕ) :
    books_first_shop = 50 →
    cost_first_shop = 1000 →
    cost_second_shop = 800 →
    average_price = 20 →
    average_price * (books_first_shop + books_second_shop) = cost_first_shop + cost_second_shop →
    books_second_shop = 40 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end number_of_books_from_second_shop_l738_738534


namespace assignment_statement_increases_l738_738910

theorem assignment_statement_increases (N : ℕ) : (N + 1 = N + 1) :=
sorry

end assignment_statement_increases_l738_738910


namespace find_a_l738_738197

theorem find_a (x y z a : ℝ) (k : ℝ) (h1 : x = 2 * k) (h2 : y = 3 * k) (h3 : z = 5 * k)
    (h4 : x + y + z = 100) (h5 : y = a * x - 10) : a = 2 :=
  sorry

end find_a_l738_738197


namespace equation_of_line_l738_738901

-- Given point A (1, -2) and slope 3
def pointA : ℝ × ℝ := (1, -2)
def slope : ℝ := 3

-- The target equation to prove
def target_equation (x y : ℝ) : Prop := 3 * x - y - 5 = 0

-- The form of the line equation through a point with a given slope using point-slope form
def line_equation (x y : ℝ) : Prop := y + 2 = 3 * (x - 1)

theorem equation_of_line : ∀ (x y : ℝ), line_equation x y ↔ target_equation x y :=
by
  intros x y
  split
  {
    intro h
    unfold line_equation at h
    unfold target_equation
    sorry -- Proof steps will go here
  }
  {
    intro h
    unfold target_equation at h
    unfold line_equation
    sorry -- Proof steps will go here
  }

end equation_of_line_l738_738901


namespace catch_up_distance_l738_738990

def distance_a_leg1 : ℝ := 10 * 2  -- Distance covered by A in the first leg
def distance_a_leg2 : ℝ := 5 * 3   -- Distance covered by A in the second leg
def distance_a_leg3 : ℝ := 8 * 2   -- Distance covered by A in the third leg
def total_distance_a_before_b : ℝ := distance_a_leg1 + distance_a_leg2 + distance_a_leg3  -- Total distance covered by A before B starts

def time_b_to_catch_up_a : ℝ := 51 / 12  -- Time it takes for B to catch up A

def distance_b_to_catch_up : ℝ := 20 * time_b_to_catch_up_a  -- Distance B travels in this time

theorem catch_up_distance : distance_b_to_catch_up = 85 := by
  have distance_a_covered := distance_a_leg1 + distance_a_leg2 + distance_a_leg3
  have total_distance_a := distance_a_covered + 8 * time_b_to_catch_up_a
  have distance_b := 20 * time_b_to_catch_up_a
  have eq1 : total_distance_a = distance_b := by
    calc
      total_distance_a = 51 + 8 * (51 / 12) : by rw [distance_a_covered, time_b_to_catch_up_a]
      ... = distance_b : by sorry -- Solve the arithmetic equation to reach the desired result
  sorry

end catch_up_distance_l738_738990


namespace proposition_true_l738_738160

variable (A : Set)

def p := ∃ (T : ℝ), ∀ (x : ℝ), sin (x + T) = sin x
def q := ∅ ⊆ A

theorem proposition_true :
  (p ∧ q) := 
by 
  sorry

end proposition_true_l738_738160


namespace three_digit_numbers_count_l738_738196

theorem three_digit_numbers_count : 
  let digits := {0, 1, 2, 3}
  ∃ nums :  set (list ℕ), 
  ∀ num ∈ nums, list.length num = 3 ∧ (∀ d ∈ num, d ∈ digits) ∧ 
  (num.head ≠ 0) ∧ (num.Nodup) 
  → nums.size = 18 :=
  sorry

end three_digit_numbers_count_l738_738196


namespace loss_equates_to_balls_l738_738515

theorem loss_equates_to_balls
    (SP_20 : ℕ) (CP_1: ℕ) (Loss: ℕ) (x: ℕ)
    (h1 : SP_20 = 720)
    (h2 : CP_1 = 48)
    (h3 : Loss = (20 * CP_1 - SP_20))
    (h4 : Loss = x * CP_1) :
    x = 5 :=
by
  sorry

end loss_equates_to_balls_l738_738515


namespace negation_of_prop_exists_negation_of_prop_forall_l738_738573

theorem negation_of_prop_exists (h : ∃ (x : ℝ), 2^x ≤ 0) : false :=
  by sorry

theorem negation_of_prop_forall : (∀ (x : ℝ), 2^x > 0) :=
  by {
    intro x,
    have h : ¬(2^x ≤ 0),
    { apply not_exists.1,
      intro h_exists,
      have := negation_of_prop_exists h_exists,
      contradiction, 
    },
    exact lt_of_not_le h,
  }

end negation_of_prop_exists_negation_of_prop_forall_l738_738573


namespace problem_sum_l738_738785

def f (x : ℝ) : ℝ := x^3 - 3 * x^2

theorem problem_sum :
  ∑ k in Finset.range 4045.succ, f (k / 2023) = -8090 :=
by sorry

end problem_sum_l738_738785


namespace vitya_and_vova_l738_738604

-- Definitions for the given problem
def is_solution (V W : ℕ) : Prop :=
  V + W = 27 ∧ 5 * V + 3 * W = 111

-- Correct answer definition
def correct_answer (V W : ℕ) : Prop :=
  V = 15 ∧ W = 12

-- Theorem for the proof problem
theorem vitya_and_vova :
  ∃ (V W : ℕ), is_solution V W ∧ correct_answer V W :=
by {
  existsi 15,
  existsi 12,
  split,
  { split; refl },
  { split; refl }
}

end vitya_and_vova_l738_738604


namespace lines_through_point_hyperbola_l738_738996

theorem lines_through_point_hyperbola :
  ∃! (l : ℝ → ℝ), 
    (∀ x, l x = k * x + 1) ∧ 
    (∃ k, k = 1 ∨ k = -1 ∨ k = √2 ∨ k = -√2) ∧ 
    ((∀ x y, x^2 - y^2 - 1 = 0) → 
      ∃! p, (p.1, p.2) = (0, 1) ∧ p.2 = l p.1) :=
sorry

end lines_through_point_hyperbola_l738_738996


namespace meal_cost_is_seven_l738_738155

-- Defining the given conditions
def total_cost : ℕ := 21
def number_of_meals : ℕ := 3

-- The amount each meal costs
def meal_cost : ℕ := total_cost / number_of_meals

-- Prove that each meal costs 7 dollars given the conditions
theorem meal_cost_is_seven : meal_cost = 7 :=
by
  -- The result follows directly from the definition of meal_cost
  unfold meal_cost
  have h : 21 / 3 = 7 := by norm_num
  exact h


end meal_cost_is_seven_l738_738155


namespace num_two_digit_integers_congruent_to_2_mod_5_l738_738081

theorem num_two_digit_integers_congruent_to_2_mod_5 :
  let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99 ∧ n % 5 = 2} in
  S.card = 18 := 
by
  sorry

end num_two_digit_integers_congruent_to_2_mod_5_l738_738081


namespace inverse_sum_mod_l738_738224

theorem inverse_sum_mod (h1 : ∃ k, 3^6 ≡ 1 [MOD 17])
                        (h2 : ∃ k, 3 * 6 ≡ 1 [MOD 17]) : 
  (6 + 9 + 2 + 1 + 6 + 1) % 17 = 8 :=
by
  cases h1 with k1 h1
  cases h2 with k2 h2
  sorry

end inverse_sum_mod_l738_738224


namespace triangle_max_area_l738_738116

open Real

theorem triangle_max_area (A B C : ℝ) (BC_eq_2 : BC = 2) (dot_product_eq : dot_product AB AC = 1) : 
  ∃ S, S ≤ sqrt 2 ∧ ∀ S', S' > sqrt 2 → S' ≠ area A B C :=
sorry

end triangle_max_area_l738_738116


namespace EricBenJackMoneySum_l738_738733

noncomputable def EricBenJackTotal (E B J : ℕ) :=
  (E + B + J : ℕ)

theorem EricBenJackMoneySum :
  ∀ (E B J : ℕ), (E = B - 10) → (B = J - 9) → (J = 26) → (EricBenJackTotal E B J) = 50 :=
by
  intros E B J
  intro hE hB hJ
  rw [hJ] at hB
  rw [hB] at hE
  sorry

end EricBenJackMoneySum_l738_738733


namespace even_number_of_tuples_l738_738133

theorem even_number_of_tuples :
  let n := { (a_1, a_2, a_3, a_4, a_5) : ℕ × ℕ × ℕ × ℕ × ℕ |
              a_1 > 0 ∧ a_2 > 0 ∧ a_3 > 0 ∧ a_4 > 0 ∧ a_5 > 0 ∧
              (1 / a_1 + 1 / a_2 + 1 / a_3 + 1 / a_4 + 1 / a_5 = 1) } in
  nat.even (n.card) :=
sorry

end even_number_of_tuples_l738_738133


namespace sum_of_interior_angles_dodecagon_l738_738205

theorem sum_of_interior_angles_dodecagon :
  let n := 12 in
  (n - 2) * 180 = 1800 :=
begin
  sorry
end

end sum_of_interior_angles_dodecagon_l738_738205


namespace factorial_simplification_l738_738889

theorem factorial_simplification :
  (13! / (11! + 2 * 10!)) = (156 / 31) := 
by
  sorry

end factorial_simplification_l738_738889


namespace h_value_l738_738084

theorem h_value (h : ℝ) : (∃ x : ℝ, x^3 + h * x + 5 = 0 ∧ x = 3) → h = -32 / 3 := by
  sorry

end h_value_l738_738084


namespace cos_of_angle_sum_l738_738810

variable (θ : ℝ)

-- Given condition
axiom sin_theta : Real.sin θ = 1 / 4

-- To prove
theorem cos_of_angle_sum : Real.cos (3 * Real.pi / 2 + θ) = -1 / 4 :=
by
  sorry

end cos_of_angle_sum_l738_738810


namespace problem_part1_problem_part2_l738_738416

noncomputable def f (x : ℝ) : ℝ := - (√3) * (sin x)^2 + (sin x) * (cos x)

theorem problem_part1 : f (25 * π / 6) = 0 :=
sorry

theorem problem_part2 (α : ℝ) (h1 : 0 < α) (h2 : α < π) (h3 : f (α / 2) = 1 / 4 - (√3) / 2) : 
  sin α = (1 + 3 * (√5)) / 8 :=
sorry

end problem_part1_problem_part2_l738_738416


namespace trigonometric_identity_l738_738759

variable {α β γ n : Real}

-- Condition:
axiom h : Real.sin (2 * (α + γ)) = n * Real.sin (2 * β)

-- Statement to be proved:
theorem trigonometric_identity : 
  Real.tan (α + β + γ) / Real.tan (α - β + γ) = (n + 1) / (n - 1) :=
by
  sorry

end trigonometric_identity_l738_738759


namespace distance_from_S_to_face_PQR_l738_738528

def Point := ℝ × ℝ × ℝ

def distance (A B : Point) : ℝ :=
  real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2 + (A.3 - B.3)^2)

def perpendicular (A B C D : Point) : Prop :=
  (A.1 - B.1) * (C.1 - D.1) + (A.2 - B.2) * (C.2 - D.2) + (A.3 - B.3) * (C.3 - D.3) = 0

theorem distance_from_S_to_face_PQR (P Q R S : Point)
  (h_perp1 : perpendicular S P S Q)
  (h_perp2 : perpendicular S P S R)
  (h_perp3 : perpendicular S Q S R)
  (h_SP : distance S P = 10)
  (h_SQ : distance S Q = 10)
  (h_SR : distance S R = 8) :
  ∃ d : ℝ, d = 4 :=
by
  sorry

end distance_from_S_to_face_PQR_l738_738528


namespace factorial_simplification_l738_738890

theorem factorial_simplification :
  (13! / (11! + 2 * 10!)) = (156 / 31) := 
by
  sorry

end factorial_simplification_l738_738890


namespace min_handshakes_l738_738278

theorem min_handshakes 
  (people : ℕ) 
  (handshakes_per_person : ℕ) 
  (total_people : people = 30) 
  (handshakes_rule : handshakes_per_person = 3) 
  (unique_handshakes : people * handshakes_per_person % 2 = 0) 
  (multiple_people : people > 0):
  (people * handshakes_per_person / 2) = 45 :=
by
  sorry

end min_handshakes_l738_738278


namespace sum_of_digits_in_base3_l738_738286

theorem sum_of_digits_in_base3 (n : ℕ) (h1 : (6561 ≤ n) ∧ (n ≤ 59048))
                               (h2 : ∀ k : ℕ, ∀ b : ℕ, 1 < b → k < b → nat.digits b k = [k]) :
  (let d := nat.digits 3 n in
    (d.length = 9 ∨ d.length = 10) →
      (d.length = 9 ∨ d.length = 10) * 1 + (d.length = 10 ∨ d.length = 9) * 10 = 19) :=
begin
  sorry -- proof to be filled out
end

end sum_of_digits_in_base3_l738_738286


namespace range_of_w_l738_738057

noncomputable def f (w x : ℝ) : ℝ := sin (w * x) - (sqrt 3) * cos (w * x)

theorem range_of_w (w : ℝ) (h_w : w > 0) (h_zeros : ∀ k : ℤ, k = 0 ∨ k = 1 ∨ k = 2 → 
    (0 < (k * real.pi + real.pi / 3) / w ∧ (k * real.pi + real.pi / 3) / w < real.pi ∧ 
    ∑_{k=0 to 2} (0 < (k * real.pi + real.pi / 3) / w ∧ (k * real.pi + real.pi / 3) / w < real.pi) = 3)) :
    (7 / 3 < w ∧ w ≤ 10 / 3) :=
sorry

end range_of_w_l738_738057


namespace p_q_sum_is_91_l738_738457

noncomputable def tetrahedron_vertices : List (ℝ × ℝ × ℝ) :=
[(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, -1, -1)]

noncomputable def midpoint (p1 p2 : ℝ × ℝ × ℝ) : (ℝ × ℝ × ℝ) :=
((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2, (p1.3 + p2.3) / 2)

noncomputable def edge_midpoints : List (ℝ × ℝ × ℝ) :=
[midpoint (1, 0, 0) (0, 1, 0),
 midpoint (1, 0, 0) (0, 0, 1),
 midpoint (1, 0, 0) (-1, -1, -1),
 midpoint (0, 1, 0) (0, 0, 1),
 midpoint (0, 1, 0) (-1, -1, -1),
 midpoint (0, 0, 1) (-1, -1, -1)]

noncomputable def volume_ratio_octahedron_tetrahedron : ℚ :=
27 / 64

theorem p_q_sum_is_91
  (p q : ℕ)
  (hpq_coprime : Nat.coprime p q)
  (h_ratio : (p : ℚ) / q = volume_ratio_octahedron_tetrahedron) :
  p + q = 91 :=
sorry

end p_q_sum_is_91_l738_738457


namespace incorrect_statements_l738_738587

-- Define the floor function [x] as ⌊x⌋
def floor (x : ℝ) : ℤ := Int.floor x

-- Define the function f(x) = x + ⌊x⌋
def f (x : ℝ) : ℝ := x + ↑(floor x)

-- State the theorem about the incorrect statements for function f
theorem incorrect_statements :
  ¬(∀ x : ℝ, f (-x) = - f x) ∧ ¬(∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, f (x + p) = f x) :=
by
  sorry

end incorrect_statements_l738_738587


namespace number_of_possible_values_of_a_l738_738529

theorem number_of_possible_values_of_a :
  ∃ (n : ℕ), n = 501 ∧
  ∀ (a b c d : ℕ), a > b ∧ b > c ∧ c > d ∧ 
  a + b + c + d = 2010 ∧
  a^2 - b^2 + c^2 - d^2 = 2010 →
  (∃ n : ℕ, n = a ∧ α ∈ set.univ) :=
begin
  -- The proof will go here
  sorry
end

end number_of_possible_values_of_a_l738_738529


namespace net_configuration_exists_l738_738973

/--
Given a \(2 \times 1\) paper parallelepiped that is cut into a net with ten squares, 
prove the existence of a net configuration such that removing one square 
leaves a net with nine remaining squares.
-/
theorem net_configuration_exists (initial_squares : ℕ) (remaining_squares : ℕ) (parallelepiped : ℕ × ℕ) :
  parallelepiped = (2, 1) ∧ initial_squares = 10 ∧ remaining_squares = 9 → 
  ∃ net : list (list bool), length (join net) = remaining_squares :=
begin
  sorry -- Proof to be filled in
end

end net_configuration_exists_l738_738973


namespace regular_polygon_sides_l738_738473

theorem regular_polygon_sides 
  (A B C : ℝ)
  (h₁ : A + B + C = 180)
  (h₂ : B = 3 * A)
  (h₃ : C = 6 * A) :
  ∃ (n : ℕ), n = 5 :=
by
  sorry

end regular_polygon_sides_l738_738473


namespace parabola_and_directrix_slope_sum_value_l738_738065

noncomputable theory

-- Define the parabola and conditions given
def parabola (p : ℝ) (hp : p > 0) : Prop := ∃ y, (2:ℝ)² = 2 * p * y ∧ (abs (0 - 2) = 2)

def point_on_parabola (x y₀ p : ℝ) (hp : p > 0) : Prop :=
  x = 2 ∧ ∃ y₀, 4 = 2 * p * y₀ ∧ (y₀ + p / 2 = 2) 

-- Define the equations and points
def parabola_equation : Prop := ∀ p (hp : p > 0), parabola p hp → True

def directrix_equation : Prop := ∀ p (hp : p > 0), parabola p hp → True

def intersection_points (k : ℝ) (hk : k ≠ 0) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ),
  x1 + x2 = 4 * k ∧
  x1 * x2 = -4 ∧
  y1 = k * x1 + 1 ∧
  y2 = k * x2 + 1

def slope_sums (k : ℝ) (hk : k ≠ 0) : Prop :=
  ∀ (x1 y1 x2 y2 : ℝ),
  intersection_points k hk →
  (-((2 - x1) / (k * x1 + 2)) - ((2 - x2) / (k * x2 + 2)) = -2)

-- Lean 4 statement for part (1)
theorem parabola_and_directrix :
  ∃ (p : ℝ) (hp : p > 0), parabola p hp ∧ point_on_parabola 2 (-1) p hp ∧ 
  parabola_equation ∧ directrix_equation := 
sorry

-- Lean 4 statement for part (2)
theorem slope_sum_value :
  ∀ (k : ℝ) (hk : k ≠ 0), slope_sums k hk := 
sorry

end parabola_and_directrix_slope_sum_value_l738_738065


namespace Rio_Coralio_Length_Estimate_l738_738645

def RioCoralioLength := 402.5
def GSA_length := 402
def AWRA_length := 403
def error_margin := 0.5
def error_probability := 0.04

theorem Rio_Coralio_Length_Estimate :
  ∀ (L_GSA L_AWRA : ℝ) (margin error_prob : ℝ),
  L_GSA = GSA_length ∧ L_AWRA = AWRA_length ∧ 
  margin = error_margin ∧ error_prob = error_probability →
  (RioCoralioLength = 402.5) ∧ (error_probability = 0.04) := 
by 
  intros L_GSA L_AWRA margin error_prob h,
  sorry

end Rio_Coralio_Length_Estimate_l738_738645


namespace chord_length_min_value_l738_738385

theorem chord_length_min_value :
  let circle (x y : ℝ) := (x - 1) ^ 2 + (y - 2) ^ 2 = 25
  let line (m x y : ℝ) := (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0
  ∃ m : ℝ, ∃ A B : ℝ × ℝ,
    circle A.1 A.2 ∧ circle B.1 B.2 ∧
    line m A.1 A.2 ∧ line m B.1 B.2 ∧
    (dist A B = 4 * sqrt 5) :=
  sorry

end chord_length_min_value_l738_738385


namespace balloon_count_l738_738928

theorem balloon_count (gold_balloon silver_balloon black_balloon blue_balloon green_balloon total_balloon : ℕ) (h1 : gold_balloon = 141) 
                      (h2 : silver_balloon = (gold_balloon / 3) * 5) 
                      (h3 : black_balloon = silver_balloon / 2) 
                      (h4 : blue_balloon = black_balloon / 2) 
                      (h5 : green_balloon = (blue_balloon / 4) * 3) 
                      (h6 : total_balloon = gold_balloon + silver_balloon + black_balloon + blue_balloon + green_balloon): 
                      total_balloon = 593 :=
by 
  sorry

end balloon_count_l738_738928


namespace find_remaining_angles_l738_738292

noncomputable def equal_segments (A B C : ℝ) (circle_segments : List ℝ) : Prop :=
  circle_segments.length = 3 ∧ 
  circle_segments.nth 0 = circle_segments.nth 1 ∧ 
  circle_segments.nth 1 = circle_segments.nth 2 ∧
  circle_segments.nth 2 = circle_segments.nth 0

theorem find_remaining_angles (α β γ : ℝ) 
  (h1 : α = 60) 
  (arcs : ℝ → ℝ) 
  (h2 : arcs 0 + arcs 1 + arcs 2 = 90) 
  (h3 : ∃ k > 0, arcs 1 = 2 * arcs 0 ∧ arcs 2 = k * arcs 0)
  (triangle_condition : equal_segments α β γ [arcs 0, arcs 1, arcs 2]) :
  (α, β, γ) = (60, 45, 75) ∨ (α, β, γ) = (60, 75, 45) := 
  sorry

end find_remaining_angles_l738_738292


namespace wheel_distance_l738_738695

theorem wheel_distance (total_distance : ℕ) (wheels_in_use : ℕ) (spare_wheels : ℕ) (total_wheels : ℕ) (equal_distance : ℕ) :
  total_distance = 100 →
  wheels_in_use = 3 →
  spare_wheels = 2 →
  total_wheels = wheels_in_use + spare_wheels →
  equal_distance * total_wheels = wheels_in_use * total_distance →
  equal_distance = 60 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at h5
  have h5' : equal_distance * 5 = 3 * 100 := h5
  linarith

end wheel_distance_l738_738695


namespace find_number_l738_738517

-- Given conditions
def one_third_of_1200 : ℝ := 1200 / 3
def percent_to_decimal : ℝ := 169.4915254237288 / 100

-- Define x and its condition
def x (n : ℝ) : Prop := one_third_of_1200 = percent_to_decimal * n

-- The proof problem statement
theorem find_number (n : ℝ) (h : x n) : n ≈ 236 :=
sorry

end find_number_l738_738517


namespace cone_volume_l738_738671

theorem cone_volume (V_cylinder V_frustum V_cone : ℝ)
  (h₁ : V_cylinder = 9)
  (h₂ : V_frustum = 63) :
  V_cone = 64 :=
sorry

end cone_volume_l738_738671


namespace limit_sin_kx_over_x_l738_738349

open Real

theorem limit_sin_kx_over_x (k : ℝ) : 
  filter.tendsto (λ x: ℝ, (sin (k * x)) / x) (𝓝 0) (𝓝 k) := 
sorry

end limit_sin_kx_over_x_l738_738349


namespace find_three_digit_numbers_l738_738356

def is_three_digit_number (n : Nat) : Prop :=
  100 ≤ n ∧ n < 1000
  
def perm_arith_mean_eq (a b c : Nat) : Prop :=
  let A := 100 * a + 10 * b + c
  let p1 := 100 * a + 10 * b + c
  let p2 := 100 * a + 10 * c + b
  let p3 := 100 * b + 10 * a + c
  let p4 := 100 * b + 10 * c + a
  let p5 := 100 * c + 10 * a + b
  let p6 := 100 * c + 10 * b + a
  A = 1/6 * (p1 + p2 + p3 + p4 + p5 + p6)

theorem find_three_digit_numbers :
  ∀ (n : Nat), is_three_digit_number n →
    (∃ (a b c : Nat), n = 100 * a + 10 * b + c ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ perm_arith_mean_eq a b c) →
    n ∈ {111, 222, 333, 444, 555, 666, 777, 888, 999, 407, 518, 629, 370, 481, 592} :=
by sorry

end find_three_digit_numbers_l738_738356


namespace range_of_a_l738_738192

def f (x a : ℝ) : ℝ := Real.exp x - a * Real.log (a * x - a) + a

theorem range_of_a (a : ℝ) (h : 0 < a) : (∀ x : ℝ, f x a > 0) ↔ 0 < a ∧ a < Real.exp 2 :=
by
  -- insert proof here
  sorry

end range_of_a_l738_738192


namespace sin_2A_result_l738_738472

variable (A : ℝ)

-- Given condition
def cos_condition : Prop := cos (π / 4 + A) = 5 / 13

-- Theorem statement
theorem sin_2A_result (A : ℝ) (h : cos_condition A) : sin (2 * A) = 119 / 169 := by
  sorry

end sin_2A_result_l738_738472


namespace part_a_part_b_l738_738621

theorem part_a (a b c d : ℕ) (h : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) : 
  ∃ (a b c d : ℕ), 1 / (a : ℝ) + 1 / (b : ℝ) = 1 / (c : ℝ) + 1 / (d : ℝ) := sorry

theorem part_b (a b c d e : ℕ) (h : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) : 
  ∃ (a b c d e : ℕ), 1 / (a : ℝ) + 1 / (b : ℝ) = 1 / (c : ℝ) + 1 / (d : ℝ) + 1 / (e : ℝ) := sorry

end part_a_part_b_l738_738621


namespace sum_of_inverses_mod_17_l738_738243

theorem sum_of_inverses_mod_17 :
  (3⁻¹ + 3⁻² + 3⁻³ + 3⁻⁴ + 3⁻⁵ + 3⁻⁶) % 17 = 9 := sorry

end sum_of_inverses_mod_17_l738_738243


namespace cylinder_inscribed_in_sphere_l738_738692

noncomputable def sphere_volume (r : ℝ) : ℝ := 
  (4 / 3) * Real.pi * r^3

theorem cylinder_inscribed_in_sphere 
  (r_cylinder : ℝ)
  (h₁ : r_cylinder > 0)
  (height_cylinder : ℝ)
  (radius_sphere : ℝ)
  (h₂ : radius_sphere = r_cylinder + 2)
  (h₃ : height_cylinder = r_cylinder + 1)
  (h₄ : 2 * radius_sphere = Real.sqrt ((2 * r_cylinder)^2 + (height_cylinder)^2))
  : sphere_volume 17 = 6550 * 2 / 3 * Real.pi :=
by
  -- solution steps and proof go here
  sorry

end cylinder_inscribed_in_sphere_l738_738692


namespace angle_quadrant_l738_738444

theorem angle_quadrant (θ : Real) (P : Real × Real) (h : P = (Real.sin θ * Real.cos θ, 2 * Real.cos θ) ∧ P.1 < 0 ∧ P.2 < 0) :
  π / 2 < θ ∧ θ < π :=
by
  sorry

end angle_quadrant_l738_738444


namespace leo_import_tax_excess_amount_l738_738612

theorem leo_import_tax_excess_amount (total_value : ℝ) (tax_paid : ℝ) (tax_rate : ℝ) 
  (htotal : total_value = 2250) (htax_paid : tax_paid = 87.50) (htax_rate : tax_rate = 0.07):
  ∃ (excess_amount : ℝ), excess_amount = 1000 ∧ (total_value - excess_amount) * tax_rate = tax_paid := 
by
  use 1000
  split
  { 
    refl
  }
  { 
    rw [htotal, htax_paid, htax_rate]
    norm_num
  }

end leo_import_tax_excess_amount_l738_738612


namespace triangle_c_value_l738_738476

noncomputable def triangle_problem (A B a : ℝ) (hA: A = 30) (hB: B = 105) (ha: a = 4) : ℝ :=
  let C := 180 - A - B in
  let sin_A := Real.sin (A * Real.pi / 180) in
  let sin_C := Real.sin (C * Real.pi / 180) in
  a * sin_C / sin_A

theorem triangle_c_value (A B a : ℝ) (hA : A = 30) (hB : B = 105) (ha : a = 4) :
  triangle_problem A B a hA hB ha = 4 * Real.sqrt 2 :=
by sorry

end triangle_c_value_l738_738476


namespace sequence_solution_l738_738794

variable {ℕ : Type} [Nonempty ℕ] [Inhabited ℕ]

def recurrence_relation (a : ℕ → ℕ) (n : ℕ) : Prop :=
  ∀ n (n > 1), (a (n + 1) + a n - 1) / (a (n + 1) - a n + 1) = n

def initial_condition (a : ℕ → ℕ) : Prop :=
  a 2 = 6

theorem sequence_solution (a : ℕ → ℕ) (h1 : recurrence_relation a) (h2 : initial_condition a) :
  ∀ n, a n = n * (2 * n - 1) :=
by
  sorry

end sequence_solution_l738_738794


namespace flutes_tried_out_l738_738868

theorem flutes_tried_out (flutes clarinets trumpets pianists : ℕ) 
  (percent_flutes_in : ℕ → ℕ) (percent_clarinets_in : ℕ → ℕ) 
  (percent_trumpets_in : ℕ → ℕ) (percent_pianists_in : ℕ → ℕ) 
  (total_in_band : ℕ) :
  percent_flutes_in flutes = 80 / 100 * flutes ∧
  percent_clarinets_in clarinets = 30 / 2 ∧
  percent_trumpets_in trumpets = 60 / 3 ∧
  percent_pianists_in pianists = 20 / 10 ∧
  total_in_band = 53 →
  flutes = 20 :=
by
  sorry

end flutes_tried_out_l738_738868


namespace composite_quadratic_l738_738201

theorem composite_quadratic (a b : Int) (x1 x2 : Int)
  (h1 : x1 + x2 = -a)
  (h2 : x1 * x2 = b)
  (h3 : abs x1 > 2)
  (h4 : abs x2 > 2) :
  ∃ m n : Int, a + b + 1 = m * n ∧ m > 1 ∧ n > 1 :=
by
  sorry

end composite_quadratic_l738_738201


namespace total_metal_rods_l738_738295

def rods_per_sheet : ℕ → ℕ
| 1 := 10  -- Aluminum
| 2 := 8   -- Bronze
| 3 := 12  -- Copper
| _ := 0   -- Default to capture any unsupported type

def rods_per_beam : ℕ → ℕ
| 1 := 6   -- Aluminum
| 2 := 4   -- Bronze
| 3 := 5   -- Copper
| _ := 0   -- Default to capture any unsupported type

def pattern_X_rods (panels : ℕ) : ℕ :=
  (2 * rods_per_sheet 1 + rods_per_sheet 2 + 2 * rods_per_beam 3) * panels

def pattern_Y_rods (panels : ℕ) : ℕ :=
  (rods_per_sheet 3 + 2 * rods_per_sheet 2 + 3 * rods_per_beam 1 + rods_per_beam 2) * panels

def total_rods (X_panels Y_panels : ℕ) : ℕ :=
  pattern_X_rods X_panels + pattern_Y_rods Y_panels

-- Lean statement that states the total number of rods required
theorem total_metal_rods :
  total_rods 7 3 = 416 :=
by
  -- Proof goes here 
  sorry

end total_metal_rods_l738_738295


namespace max_pies_without_ingredients_l738_738151

theorem max_pies_without_ingredients :
  let total_pies := 48
  let chocolate_pies := total_pies / 3
  let marshmallow_pies := total_pies / 2
  let cayenne_pies := 3 * total_pies / 8
  let soy_nut_pies := total_pies / 8
  total_pies - max chocolate_pies (max marshmallow_pies (max cayenne_pies soy_nut_pies)) = 24 := by
{
  sorry
}

end max_pies_without_ingredients_l738_738151


namespace find_positive_integers_l738_738355

theorem find_positive_integers
  (a b c : ℕ) 
  (h : a ≥ b ∧ b ≥ c ∧ a ≥ c)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0) :
  (1 + 1 / (a : ℚ)) * (1 + 1 / (b : ℚ)) * (1 + 1 / (c : ℚ)) = 2 →
  (a, b, c) ∈ [(15, 4, 2), (9, 5, 2), (7, 6, 2), (8, 3, 3), (5, 4, 3)] :=
by
  sorry

end find_positive_integers_l738_738355


namespace triangle_obtuse_triangle_is_obtuse_l738_738475

theorem triangle_obtuse (A B : ℝ) (hA : A = 10) (hB : B = 60) : A + B < 180 :=
begin
  -- Proof steps go here
  sorry
end

theorem triangle_is_obtuse (A B C : ℝ) (hA : A = 10) (hB : B = 60) (hC : C = 180 - A - B) : 
  C = 110 :=
begin
  -- Proof steps go here
  sorry
end

#print triangle_obtuse
#print triangle_is_obtuse

end triangle_obtuse_triangle_is_obtuse_l738_738475


namespace number_of_technicians_l738_738102

theorem number_of_technicians :
  ∃ (T R : ℕ), T + R = 21 ∧ 2 * T + R = 28 ∧ T = 7 :=
by {
  use [7, 14],
  split,
  { exact rfl },
  split,
  { exact rfl },
  { exact rfl }
}

end number_of_technicians_l738_738102


namespace students_in_diligence_before_transfer_l738_738099

theorem students_in_diligence_before_transfer (D I P : ℕ)
  (h_total : D + I + P = 75)
  (h_equal : D + 2 = I - 2 + 3 ∧ D + 2 = P - 3) :
  D = 23 :=
by
  sorry

end students_in_diligence_before_transfer_l738_738099


namespace second_car_distance_rate_l738_738456

variables {l : ℝ} {v1 v2 vm : ℝ}

-- The conditions as given in the problem
def condition1 : Prop := (l / v2) - (l / v1) = 1 / 60
def condition2 : Prop := v1 = 4 * vm
def condition3 : Prop := (v2 / 60) - (vm / 60) = l / 6
def condition4 : Prop := l / vm < 10

-- The final result to be proved under the given conditions
theorem second_car_distance_rate
  (h1 : condition1)
  (h2 : condition2)
  (h3 : condition3)
  (h4 : condition4) :
  v2 / 60 = 2 / 3 := 
sorry

end second_car_distance_rate_l738_738456


namespace minimum_p_for_required_profit_l738_738577

noncomputable def profit (x p : ℝ) : ℝ := p * x - (0.5 * x^2 - 2 * x - 10)
noncomputable def max_profit (p : ℝ) : ℝ := (p + 2)^2 / 2 + 10

theorem minimum_p_for_required_profit : ∀ (p : ℝ), 3 * max_profit p >= 126 → p >= 6 :=
by
  intro p
  unfold max_profit
  -- Given:  3 * ((p + 2)^2 / 2 + 10) >= 126
  sorry

end minimum_p_for_required_profit_l738_738577


namespace find_k_for_min_value_zero_l738_738594

theorem find_k_for_min_value_zero :
  (∃ k : ℝ, ∀ x y : ℝ, 9 * x^2 - 12 * k * x * y + (4 * k^2 + 3) * y^2 - 6 * x - 3 * y + 9 ≥ 0 ∧
                         ∃ x y : ℝ, 9 * x^2 - 12 * k * x * y + (4 * k^2 + 3) * y^2 - 6 * x - 3 * y + 9 = 0) →
  k = 3 / 2 :=
sorry

end find_k_for_min_value_zero_l738_738594


namespace intersection_P_Q_l738_738146

def setP : Set ℝ := {1, 2, 3, 4}
def setQ : Set ℝ := {x | abs x ≤ 2}

theorem intersection_P_Q : (setP ∩ setQ) = {1, 2} :=
by
  sorry

end intersection_P_Q_l738_738146


namespace CityResidentShouldInstallLampHimself_l738_738968

structure CityResidentLampDecision where
  P1 : ℕ -- 60W incandescent lamp
  P2 : ℕ -- 12W energy-efficient lamp
  T : ℕ -- Tariff 5 rubles/kWh
  t : ℕ -- Monthly usage time in hours
  costEfficientLamp : ℕ -- Initial cost of energy-efficient lamp
  serviceCompanyRate : ℝ -- Rate of payment to the energy service company

def monthlyConsumption (P : ℕ) (t : ℕ) : ℝ :=
  (P * t) / 1000.0

def monthlyCost (E : ℝ) (T : ℕ) : ℝ :=
  E * T

def totalCost (C : ℕ) (months : ℕ) : ℕ :=
  C * months

def serviceCompanyPayment (ΔC : ℝ) (rate : ℝ) : ℝ :=
  ΔC * rate

def validDecision10Months (r : CityResidentLampDecision) :=
  let E1 := monthlyConsumption r.P1 r.t
  let E2 := monthlyConsumption r.P2 r.t
  let C1 := monthlyCost E1 r.T
  let C2 := monthlyCost E2 r.T
  let total10MonthCost1 := totalCost C1 10
  let total10MonthCost2 := r.costEfficientLamp + totalCost C2 10
  let ΔE := E1 - E2
  let ΔC := ΔE * r.T
  let servicePayment := serviceCompanyPayment ΔC r.serviceCompanyRate
  let newC := C2 + servicePayment
  let totalCostCompany := totalCost newC 10

  total10MonthCost2 <= totalCostCompany

def validDecisionFullLifespan (r : CityResidentLampDecision) :=
  let E2 := monthlyConsumption r.P2 r.t
  let C2 := monthlyCost E2 r.T
  let total36MonthCost := r.costEfficientLamp + totalCost C2 36
  let remainingMonths := 36 - 10
  let remainingCostCompany := totalCost C2 remainingMonths
  let totalCostCompany := 240 + remainingCostCompany

  total36MonthCost <= totalCostCompany

theorem CityResidentShouldInstallLampHimself : ∀ r : CityResidentLampDecision,
  validDecision10Months r ∧ validDecisionFullLifespan r := by
    sorry

end CityResidentShouldInstallLampHimself_l738_738968


namespace minimize_K_at_0_l738_738807

def H (p q : ℝ) : ℝ := 3 * p * q - 2 * p * (1 - q) - 4 * (1 - p) * q + 5 * (1 - p) * (1 - q)

def K (p : ℝ) : ℝ := max (5 * p - 5) (4 * p - 1)

theorem minimize_K_at_0 : ∀ p, 0 ≤ p → p ≤ 1 → 
  (∀ q, 0 ≤ q → q ≤ 1 → H p q ≤ K p) → K p ≥ K 0 :=
by
  sorry

end minimize_K_at_0_l738_738807


namespace absolute_value_inequality_l738_738161

theorem absolute_value_inequality (x y z : ℝ) : 
  |x| + |y| + |z| ≤ |x + y - z| + |x - y + z| + |-x + y + z| :=
by
  sorry

end absolute_value_inequality_l738_738161


namespace rhombus_area_l738_738947

theorem rhombus_area (s : ℝ) (θ : ℝ) (hθ : θ = π / 3) (hs : s = 2) : 
  (s * s * real.sin θ = 2 * real.sqrt 3) :=
by {
  rw [hθ, hs],
  -- θ = 60 degrees, which is π / 3 radians
  have h1 : real.sin (π / 3) = real.sqrt 3 / 2,
  { exact real.sin_pi_div_three },
  rw [real.mul_assoc, real.mul_comm s _, real.mul_assoc, h1],
  norm_num,
}

end rhombus_area_l738_738947


namespace exists_rational_distance_points_l738_738884

theorem exists_rational_distance_points (n : ℕ) (h : n = 1975) :
  ∃ (P : Fin n → ℂ), (∀ i j, i ≠ j → ∃ (r : ℚ), abs (P i - P j) = r) ∧ 
  (∀ i, complex.abs (P i) = 1) := by
  sorry

end exists_rational_distance_points_l738_738884


namespace correct_options_l738_738959

theorem correct_options :
  let num_products := 1001
  let num_selected := 10
  let average_k := 2
  let variance_k := 3
  let k := [2, 6, 8, 3, 3, 4, 6, 8]
  let n := 160
  let mid_area := (1 / 9 : ℝ)
  let sum_other_areas := 1 - (1 / 9 : ℝ)
in
(A : min_digits num_products num_selected = 4) ∧
(B : avg_variance_transformed average_k variance_k ≠ 6) ∧
(C : freq_histogram n mid_area sum_other_areas = 16) ∧
(D : mode_median k = ([3, 6, 8], 5)) ∧
(solution : {A, C, D} = {A, B, C, D} \ {B}) := sorry

end correct_options_l738_738959


namespace sum_of_inverses_mod_17_l738_738239

noncomputable def inverse_sum_mod_17 : ℤ :=
  let a1 := Nat.gcdA 3 17 in -- 3^{-1} mod 17
  let a2 := Nat.gcdA (3^2) 17 in -- 3^{-2} mod 17
  let a3 := Nat.gcdA (3^3) 17 in -- 3^{-3} mod 17
  let a4 := Nat.gcdA (3^4) 17 in -- 3^{-4} mod 17
  let a5 := Nat.gcdA (3^5) 17 in -- 3^{-5} mod 17
  let a6 := Nat.gcdA (3^6) 17 in -- 3^{-6} mod 17
  (a1 + a2 + a3 + a4 + a5 + a6) % 17

theorem sum_of_inverses_mod_17 : inverse_sum_mod_17 = 7 := sorry

end sum_of_inverses_mod_17_l738_738239


namespace factor_expression_l738_738717

theorem factor_expression (x : ℝ) : 
  (10 * x^3 + 45 * x^2 - 5 * x) - (-5 * x^3 + 10 * x^2 - 5 * x) = 5 * x^2 * (3 * x + 7) :=
by 
  sorry

end factor_expression_l738_738717


namespace smallest_positive_n_l738_738957

theorem smallest_positive_n (n : ℕ) (h : 1023 * n % 30 = 2147 * n % 30) : n = 15 :=
by
  sorry

end smallest_positive_n_l738_738957


namespace range_of_a_decreasing_l738_738566

theorem range_of_a_decreasing (a : ℝ) (h : ∀ x : ℝ, x ≥ -1 → (2 * a * x + (a - 3)) ≤ 0) :
  a ∈ set.Icc (-3 : ℝ) 0 :=
sorry

end range_of_a_decreasing_l738_738566


namespace find_A_l738_738808

def clubsuit (A B : ℤ) : ℤ := 4 * A + 2 * B + 6

theorem find_A : ∃ A : ℤ, clubsuit A 6 = 70 → A = 13 := 
by
  sorry

end find_A_l738_738808


namespace acute_angle_AED_l738_738159

open EuclideanGeometry

theorem acute_angle_AED (A B C D E : Point)
  (h_collinear : Collinear A B C D)
  (h_plane : OnPlane E (PlaneOf [A, B, C, D]))
  (h1 : Distance A C = Distance C E)
  (h2 : Distance E B = Distance B D) :
  AcuteAngle (Angle A E D) :=
sorry

end acute_angle_AED_l738_738159


namespace laura_saves_more_with_promotion_A_l738_738986

def promotion_A_cost (pair_price : ℕ) : ℕ :=
  let second_pair_price := pair_price / 2
  pair_price + second_pair_price

def promotion_B_cost (pair_price : ℕ) : ℕ :=
  let discount := pair_price * 20 / 100
  pair_price + (pair_price - discount)

def savings (pair_price : ℕ) : ℕ :=
  promotion_B_cost pair_price - promotion_A_cost pair_price

theorem laura_saves_more_with_promotion_A :
  savings 50 = 15 :=
  by
  -- The detailed proof will be added here
  sorry

end laura_saves_more_with_promotion_A_l738_738986


namespace correlation_of_xy_l738_738753

/- 
  Prove that the uncertain relationship between the independent variable x and the 
  dependent variable y, when the value of x is certain and y has certain randomness, 
  is a correlation.
-/

theorem correlation_of_xy (x y : Type) 
  (hx : ∃ (xc : x), True) 
  (hy : ∃ (yc : y), xc → exists_bool_randomness (y)) : 
  relation_is_correlation x y :=
sorry

end correlation_of_xy_l738_738753


namespace find_natural_numbers_l738_738179

theorem find_natural_numbers (x y z : ℕ) (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_ordered : x < y ∧ y < z)
  (h_reciprocal_sum_nat : ∃ a : ℕ, 1/x + 1/y + 1/z = a) : (x, y, z) = (2, 3, 6) := 
sorry

end find_natural_numbers_l738_738179


namespace wrench_force_inv_proportional_l738_738907

theorem wrench_force_inv_proportional (F₁ : ℝ) (L₁ : ℝ) (F₂ : ℝ) (L₂ : ℝ) (k : ℝ)
  (h₁ : F₁ * L₁ = k) (h₂ : L₁ = 12) (h₃ : F₁ = 300) (h₄ : L₂ = 18) :
  F₂ = 200 :=
by
  sorry

end wrench_force_inv_proportional_l738_738907


namespace part1_part2_l738_738772

noncomputable section

open real

variables (A B C a b c : ℝ)
variables (h1 : a = b * cos C - c * sin B)
variables (h2 : a = 3*sqrt 2)
variables (h3 : b = 5)

theorem part1 : tan B = -1 -> B = 135 :=
by
  sorry

theorem part2 : 
  let c := 1 in 
  let S := (1/2) * a * c * sin B in 
  B = 135 -> S = 3 / 2 :=
by
  sorry

end part1_part2_l738_738772


namespace find_f_at_sin_70_l738_738437

noncomputable def f (x : ℝ) : ℝ := cos (3 * (real.acos x))

theorem find_f_at_sin_70 : f (real.sin (real.of_real (70 * (real.pi / 180)))) = 1 / 2 := sorry

end find_f_at_sin_70_l738_738437


namespace find_n_l738_738037

variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)

-- Given conditions as definitions in Lean 4.
def S_condition (n : ℕ) : Prop := S n = 2 * (a n) - 2
def S_value_condition : Prop := S 7 = 254

-- The statement to prove.
theorem find_n : (S_condition 7) ∧ S_value_condition → ∃ n : ℕ, n = 7 :=
by
  sorry

end find_n_l738_738037


namespace Mindy_tax_rate_l738_738153

theorem Mindy_tax_rate (M : ℝ) (r : ℝ) :
  let mork_tax := 0.40 * M,
      mindy_income := 4 * M,
      combined_income := 5 * M,
      combined_tax_rate := 0.32,
      mindy_tax := r * mindy_income,
      combined_tax := mork_tax + mindy_tax
  in combined_tax = combined_tax_rate * combined_income → r = 0.30 :=
by
  intros,
  let mork_tax := 0.40 * M,
  let mindy_income := 4 * M,
  let combined_income := 5 * M,
  let combined_tax_rate := 0.32,
  let mindy_tax := r * mindy_income,
  let combined_tax := mork_tax + mindy_tax,
  have h : combined_tax = combined_tax_rate * combined_income := ‹combined_tax = combined_tax_rate * combined_income›,
  sorry

end Mindy_tax_rate_l738_738153


namespace min_chord_length_l738_738384

open Real

theorem min_chord_length (m : ℝ) :
  let C := ∀ x y : ℝ, (x - 1)^2 + (y - 2)^2 = 25
  let l := ∀ x y : ℝ, (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0 in
  ∃ A B : Real × Real, 
  (C A.fst A.snd ∧ C B.fst B.snd ∧ l A.fst A.snd ∧ l B.fst B.snd) →
  dist A B = 4 * sqrt 5 := 
sorry

end min_chord_length_l738_738384


namespace simplify_fraction_l738_738887

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem simplify_fraction :
  (factorial 13) / (factorial 11 + 2 * factorial 10) = 12 := sorry

end simplify_fraction_l738_738887


namespace correct_adverb_for_sentence_l738_738182

theorem correct_adverb_for_sentence
  (context : String)
  (modify_verb : String → Prop)
  (prevent_dropping : String)
  (cautiously_works carefuly_works : Prop)
  (noncomputable) : 
  modify_verb "held" → context = "The boy held the milk bottle" →
  prevent_dropping = "for fear of dropping it onto the ground" →
  ((cautiously_works ∧ carefully_works) → 
  (modify_verb "held" = "cautiously" ∨ modify_verb "held" = "carefully")) := 
by
  intro h1 h2 h3
  sorry

end correct_adverb_for_sentence_l738_738182


namespace find_C_given_eq_statement_max_area_triangle_statement_l738_738837

open Real

noncomputable def find_C_given_eq (a b c A : ℝ) (C : ℝ) : Prop :=
  (2 * a = sqrt 3 * c * sin A - a * cos C) → 
  C = 2 * π / 3

noncomputable def max_area_triangle (a b c : ℝ) (C : ℝ) : Prop :=
  C = 2 * π / 3 →
  c = sqrt 3 →
  ∃ S, S = (sqrt 3 / 4) * a * b ∧ 
  ∀ a b : ℝ, a * b ≤ 1 → S = (sqrt 3 / 4)

-- Lean statements
theorem find_C_given_eq_statement (a b c A C : ℝ) : find_C_given_eq a b c A C := 
by sorry

theorem max_area_triangle_statement (a b c : ℝ) (C : ℝ) : max_area_triangle a b c C := 
by sorry

end find_C_given_eq_statement_max_area_triangle_statement_l738_738837


namespace coefficient_of_x3_in_expansion_l738_738724

theorem coefficient_of_x3_in_expansion : 
  let x := λ r, (-1:ℤ)^r * 2^(6 - 2 * r) * (Nat.choose 6 r) * (divepow (x:ℚ)^(6 - (3 / 2) * r)) 
  in (∀ (x: ℚ), 
  (2 ^ 2) * (Nat.choose 6 2) * x)== 60  := 
sorry

end coefficient_of_x3_in_expansion_l738_738724


namespace coef_x_in_det_eq_3_l738_738723

-- Define the 3x3 matrix from the conditions
def M : Matrix (Fin 3) (Fin 3) ℤ := ![![1, 4, -1], ![2, 7, 1], ![-1, x, 5]]

-- The question is to determine the coefficient of x in the determinant of this matrix
theorem coef_x_in_det_eq_3 (x : ℤ) : coefficient_of (Matrix.det M) x = 3 :=
sorry

end coef_x_in_det_eq_3_l738_738723


namespace integral_calculation_l738_738712

noncomputable def integral_result : ℝ :=
  ∫ x in Real.arcsin (1 / Real.sqrt 37).toReal..(Real.pi / 4),
  (6 * Real.tan x) / (3 * Real.sin (2 * x) + 5 * Real.cos x ^ 2)

theorem integral_calculation : integral_result = (5 / 6) * Real.log ((6 * Real.exp 1) / 11) := sorry

end integral_calculation_l738_738712


namespace james_total_socks_l738_738124

theorem james_total_socks :
  ∀ (red_socks_pairs : ℕ) (black_socks_ratio red_socks_ratio : ℕ),
    red_socks_pairs = 20 →
    black_socks_ratio = 2 →
    red_socks_ratio = 2 →
    let red_socks := red_socks_pairs * 2 in
    let black_socks := red_socks / black_socks_ratio in
    let red_black_combined := red_socks + black_socks in
    let white_socks := red_black_combined * red_socks_ratio in
    red_socks + black_socks + white_socks = 180 :=
by
  intros red_socks_pairs black_socks_ratio red_socks_ratio
  intro h1 h2 h3
  let red_socks := red_socks_pairs * 2
  let black_socks := red_socks / black_socks_ratio
  let red_black_combined := red_socks + black_socks
  let white_socks := red_black_combined * red_socks_ratio
  have step1 : red_socks = 40 := by rw [h1]; refl
  have step2 : black_socks = 20 := by rw [step1, h2]; refl
  have step3 : red_black_combined = 60 := by rw [step1, step2]; refl
  have step4 : white_socks = 120 := by rw [step3, h3]; refl
  calc 
    red_socks + black_socks + white_socks 
      = 40 + 20 + 120 : by rw [step1, step2, step4]
      ... = 180 : by norm_num

end james_total_socks_l738_738124


namespace sum_of_inverses_mod_17_l738_738245

theorem sum_of_inverses_mod_17 :
  (3⁻¹ + 3⁻² + 3⁻³ + 3⁻⁴ + 3⁻⁵ + 3⁻⁶) % 17 = 9 := sorry

end sum_of_inverses_mod_17_l738_738245


namespace Rio_Coralio_Length_Estimate_l738_738647

def RioCoralioLength := 402.5
def GSA_length := 402
def AWRA_length := 403
def error_margin := 0.5
def error_probability := 0.04

theorem Rio_Coralio_Length_Estimate :
  ∀ (L_GSA L_AWRA : ℝ) (margin error_prob : ℝ),
  L_GSA = GSA_length ∧ L_AWRA = AWRA_length ∧ 
  margin = error_margin ∧ error_prob = error_probability →
  (RioCoralioLength = 402.5) ∧ (error_probability = 0.04) := 
by 
  intros L_GSA L_AWRA margin error_prob h,
  sorry

end Rio_Coralio_Length_Estimate_l738_738647


namespace tap_B_fill_time_l738_738180

theorem tap_B_fill_time :
  ∃ t : ℝ, 
    (3 * 10 + (12 / t) * 10 = 36) →
    t = 20 :=
by
  sorry

end tap_B_fill_time_l738_738180


namespace problem1_problem2_l738_738052

-- Define the function
def f (x : ℝ) : ℝ := cos x * (sin x + cos x) - 1/2

-- Statement for Question 1
theorem problem1 (α : ℝ) (h1 : 0 < α) (h2 : α < π / 2) (h3 : sin α = sqrt 2 / 2) : f α = 1/2 :=
by sorry

-- Statement for Question 2
theorem problem2 : (∃ T : ℝ, T > 0 ∧ (∀ x : ℝ, f (x + T) = f x) ∧ T = π) ∧ 
    (∀ k : ℤ, ∀ x : ℝ, k * π - 3 * π / 8 ≤ x ∧ x ≤ k * π + π / 8 → (differentiable ℝ f) ∧ 
    (∀ y : ℝ, f (y) = 0 → (k * π - 3 * π / 8 ≤ y ∧ y ≤ k * π + π / 8))) :=
by sorry

end problem1_problem2_l738_738052


namespace hyperbola_eccentricity_l738_738040

noncomputable def parabola {p : ℝ} (hp : p > 0) : set (ℝ × ℝ) :=
{x | ∃ y, y^2 = 2 * p * x}

noncomputable def hyperbola {a b : ℝ} (ha : a > 0) (hb : b > 0) : set (ℝ × ℝ) :=
{x | ∃ x y, x^2 / a^2 - y^2 / b^2 = 1}

noncomputable def asymptote {a b : ℝ} (ha : a > 0) (hb : b > 0) : set (ℝ × ℝ) :=
{x | ∃ x y, y = b / a * x}

theorem hyperbola_eccentricity
  {p a b : ℝ} (hp : p > 0) (ha : a > 0) (hb : b > 0)
  (hA : ∃ x y, (y, x) ∈ parabola hp ∧ (y, x) ∈ asymptote ha hb)
  (hD : distance (classical.some hA) (parabola_directrix hp) = p) :
  sqrt (1 + b^2 / a^2) = sqrt 5 :=
sorry

end hyperbola_eccentricity_l738_738040


namespace division_of_decimals_l738_738221

theorem division_of_decimals : 0.25 / 0.005 = 50 := 
by
  sorry

end division_of_decimals_l738_738221


namespace exists_arith_prog_with_digit_sum_arith_prog_11_exists_arith_prog_with_digit_sum_arith_prog_10000_no_arith_prog_with_digit_sum_arith_prog_infinite_l738_738257

-- Definitions to help with the problem

def arith_prog (start diff : ℕ) (length : ℕ) : list ℕ :=
  (list.range length).map (λ n, start + n * diff)

def sum_of_digits (n : ℕ) : ℕ :=
  (n.toString.data.map (λ c, c.toNat - '0'.toNat)).sum

def sums_of_digits (l : list ℕ) : list ℕ :=
  l.map sum_of_digits

def is_arith_prog (l : list ℕ) : Prop :=
  ∃ d, ∀ i < l.length - 1, l.nthLe i sorry + d = l.nthLe (i + 1) sorry

-- 1. Existence of an increasing arithmetic progression of 11 numbers
--    such that the sums of the digits of its members also form an increasing arithmetic progression.

theorem exists_arith_prog_with_digit_sum_arith_prog_11 :
  ∃ (start diff : ℕ), is_arith_prog (arith_prog start diff 11) ∧ is_arith_prog (sums_of_digits (arith_prog start diff 11)) :=
sorry

-- 2. Existence of an increasing arithmetic progression of 10,000 numbers
--    such that the sums of the digits of its members also form an increasing arithmetic progression.

theorem exists_arith_prog_with_digit_sum_arith_prog_10000 :
  ∃ (start diff : ℕ), is_arith_prog (arith_prog start diff 10000) ∧ is_arith_prog (sums_of_digits (arith_prog start diff 10000)) :=
sorry

-- 3. Non-existence of an increasing arithmetic progression of an infinite number of natural numbers
--    such that the sums of the digits of its members also form an increasing arithmetic progression.

theorem no_arith_prog_with_digit_sum_arith_prog_infinite :
  ∀ (start diff : ℕ), ¬ (is_arith_prog (arith_prog start diff (length := ℕ)) ∧ is_arith_prog (sums_of_digits (arith_prog start diff (length := ℕ)))) :=
sorry

end exists_arith_prog_with_digit_sum_arith_prog_11_exists_arith_prog_with_digit_sum_arith_prog_10000_no_arith_prog_with_digit_sum_arith_prog_infinite_l738_738257


namespace solve_inequality_l738_738892

theorem solve_inequality (x : ℝ) : 1 + 2 * (x - 1) ≤ 3 → x ≤ 2 :=
by
  sorry

end solve_inequality_l738_738892


namespace correct_conclusions_l738_738248

-- Definitions based on conditions
def condition_1 (x : ℝ) : Prop := x ≠ 0 → x + |x| > 0
def condition_3 (a b c : ℝ) (Δ : ℝ) : Prop := a > 0 ∧ Δ ≤ 0 ∧ Δ = b^2 - 4*a*c → 
  ∀ x, a*x^2 + b*x + c ≥ 0

-- Stating the proof problem
theorem correct_conclusions (x a b c Δ : ℝ) :
  (condition_1 x) ∧ (condition_3 a b c Δ) :=
sorry

end correct_conclusions_l738_738248


namespace binomial_expansion_terms_largest_coeffs_l738_738365

theorem binomial_expansion_terms_largest_coeffs :
  let T_5 := (Nat.choose 8 4) * (sqrt x)^4 * (2 / x^2)^4,
      T_6 := (Nat.choose 8 5) * (sqrt x)^3 * (2 / x^2)^5,
      T_7 := (Nat.choose 8 6) * (sqrt x)^2 * (2 / x^2)^6,
      T_5_value := 1120 / x^6,
      T_6_value := 1792 * x^(-17/2),
      T_7_value := 1792 * x^(-11)
  in T_5 = T_5_value ∧ T_6 = T_6_value ∧ T_7 = T_7_value :=
by
  sorry

end binomial_expansion_terms_largest_coeffs_l738_738365


namespace semicircle_circle_tangent_l738_738166

theorem semicircle_circle_tangent
  (d : ℝ) (hd : d = 14)
  (QR : ℝ) (hQR : QR = 3 * real.sqrt 3)
  (angleQPR : ℝ) (hangleQPR : angleQPR = 60) :
  ∃ (a b c : ℕ), a * real.sqrt b / c = 27 * real.sqrt 3 / 4 ∧ nat.gcd a c = 1 ∧ (¬ ∃ p : ℕ, p.prime ∧ p * p ∣ b) ∧ a + b + c = 34 :=
begin
  sorry
end

end semicircle_circle_tangent_l738_738166


namespace logo_shaded_area_l738_738199

theorem logo_shaded_area {l1 l2 length: ℝ} (h_length : length = 2) (h1 : l1 = 2) (h2 : l2 = 2) : 
  let r1 := 1 in     -- radius of small semicircles
  let r2 := real.sqrt 2 in  -- radius of large semicircle
  let half_circle_area_radius_1 := 0.5 * real.pi * r1^2 in -- area of small semicircle
  let half_circle_area_radius_2 := 0.5 * real.pi * r2^2 in -- area of large semicircle
  let triangle_area := 0.5 * l1 * l2 in -- area of triangle PQR
  let top_half_area := triangle_area + half_circle_area_radius_2 - half_circle_area_radius_1 in -- area of top half component
  let total_shaded_area := 2 * top_half_area in -- total shaded area
total_shaded_area = 4 + real.pi :=
sorry

end logo_shaded_area_l738_738199


namespace average_minutes_per_player_l738_738524

theorem average_minutes_per_player
  (pg sg sf pf c : ℕ)
  (total_players : ℕ)
  (hp_pg : pg = 130)
  (hp_sg : sg = 145)
  (hp_sf : sf = 85)
  (hp_pf : pf = 60)
  (hp_c : c = 180)
  (hp_total_players : total_players = 5) :
  (pg + sg + sf + pf + c) / total_players / 60 = 2 :=
by
  sorry

end average_minutes_per_player_l738_738524


namespace slope_of_AB_constant_l738_738770

-- Defining the problem conditionally
def ellipse : Type := {p : ℝ × ℝ // (p.1^2 / 16) + (p.2^2 / 12) = 1}

variables (A B P Q : ellipse)
variables 
  (hp : P.1 = 2 ∧ P.2 = 3)
  (hq : Q.1 = 2 ∧ Q.2 = -3)
  (focus : (focus_x : -2) (focus_y : 0))
  (vertex : (vertex_x : -4) (vertex_y : 0))
  (angle_condition : ∀ A B P Q : ellipse, angle APQ = angle BPQ)

-- Proof problem
theorem slope_of_AB_constant : 
  ∀ (A B P Q : ellipse) (hp : P.1 = 2 ∧ P.2 = 3) (hq : Q.1 = 2 ∧ Q.2 = -3),
  (angle APQ = angle BPQ) → slope (line_through A B) = 1/2 := 
sorry

end slope_of_AB_constant_l738_738770


namespace max_a_decreasing_l738_738090

theorem max_a_decreasing (f : ℝ → ℝ) (a : ℝ) (h : ∀ x ∈ Icc (-a) a, f' x ≤ 0) : a ≤ π / 4 := by
  sorry

end max_a_decreasing_l738_738090


namespace circle_equation_tangent_line_equation_triangle_OAB_area_l738_738064

theorem circle_equation (a : ℝ) (h : 4 * (a / 2) + 3 * 0 - 8 = 0) : x^2 + y^2 - 4 * x = 0 :=
by
  -- Proof omitted
  sorry

theorem tangent_line_equation (h₁ : x^2 + y^2 - 4 * x = 0) (h₂ : x = 1 ∧ y = sqrt 3) :
  x - sqrt 3 * y + 2 = 0 :=
by
  -- Proof omitted
  sorry

theorem triangle_OAB_area (h₁ : x^2 + y^2 - 4 * x = 0) (h₂ : x = 1 ∧ y = sqrt 3) 
  (h₃ : 4 * x + 3 * y - 8 = 0) : area_of_triangle O A B = 16 / 5 :=
by
  -- Proof omitted
  sorry

end circle_equation_tangent_line_equation_triangle_OAB_area_l738_738064


namespace odd_function_f_neg_two_l738_738409

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x^3 + 1 else -(x^3 + 1)

theorem odd_function_f_neg_two : 
  (∀ x : ℝ, f(-x) = -f(x)) → 
  (∀ x : ℝ, x > 0 → f(x) = x^3 + 1) → 
  f(-2) = -9 := by
  intro h1 h2
  have h3 : f(2) = 2^3 + 1 := h2 2 (by linarith)
  rw [<-h3, h1 2]
  have h4 : 2^3 + 1 = 9 := by norm_num
  rw h4
  norm_num
  sorry

end odd_function_f_neg_two_l738_738409


namespace find_a_b_find_m_l738_738058

-- Define the primary function
def f (a b x : ℝ) := a * x^2 - 2 * a * x + 2 + b

-- Problem (1)
theorem find_a_b (a b : ℝ) (h_max : f a b 3 = 5) (h_min : f a b 2 = 2) : 
  (a = 1 ∧ b = 0) ∨ (a = -1 ∧ b = 3) :=
  by sorry

-- Define the function g based on f
def g (a b m x : ℝ) := f a b x - (2^m) * x

-- Monotonicity condition for g in the interval [2, 4]
theorem find_m (a b : ℝ) (m : ℝ) (h_b : b < 1) (h_f : a = 1 ∧ b = 0) 
  (h_mono : ∀ x y : ℝ, 2 ≤ x → x ≤ y → y ≤ 4 → g a b m x ≤ g a b m y) : 
  m ≤ 1 ∨ m ≥ Real.log 6 / Real.log 2 :=
  by sorry

end find_a_b_find_m_l738_738058


namespace least_clock_equivalent_l738_738874

theorem least_clock_equivalent (x : ℕ) : 
  x > 3 ∧ x % 12 = (x * x) % 12 → x = 12 := 
by
  sorry

end least_clock_equivalent_l738_738874


namespace Maplewood_Summer_Camp_l738_738706

/--
At Maplewood Summer Camp, 70% of the children play soccer, and 40% of the children swim.
If 30% of the soccer players are also swimmers, 
the percentage of the non-swimmers who play soccer is 80%.
-/
theorem Maplewood_Summer_Camp (N : ℕ) :
  (0.30 * 0.70) * N = 0.21 * N →
  0.49 / 0.60 ≈ 0.80 :=
by
  sorry

end Maplewood_Summer_Camp_l738_738706


namespace alice_expected_games_l738_738208

-- Defining the initial conditions
def skill_levels := Fin 21

def initial_active_player := 0

-- Defining Alice's skill level
def Alice_skill_level := 11

-- Define the tournament structure and conditions
def tournament_round (active: skill_levels) (inactive: Set skill_levels) : skill_levels :=
  sorry

-- Define the expected number of games Alice plays
noncomputable def expected_games_Alice_plays : ℚ :=
  sorry

-- Statement of the problem proving the expected number of games Alice plays
theorem alice_expected_games : expected_games_Alice_plays = 47 / 42 :=
sorry

end alice_expected_games_l738_738208


namespace distribution_of_books_l738_738975

theorem distribution_of_books :
  let A := 2 -- number of identical art albums (type A)
  let B := 3 -- number of identical stamp albums (type B)
  let friends := 4 -- number of friends
  let total_ways := 5 -- total number of ways to distribute books 
  (A + B) = friends + 1 →
  total_ways = 5 := 
by
  intros A B friends total_ways h
  sorry

end distribution_of_books_l738_738975


namespace dot_product_solution_l738_738091

variable {V : Type*} [inner_product_space ℝ V] -- V is a real inner product space

variables (u v w : V)

noncomputable def given_conditions : Prop := 
  (∥u∥ = 1) ∧ 
  (∥v∥ = 1) ∧ 
  (∥2 • u - v∥ = real.sqrt 7) ∧ 
  (w - 2 • u + 3 • v = 4 • (u × v))

theorem dot_product_solution (h : given_conditions u v w) : 
  v ⬝ w = -4 :=
sorry

end dot_product_solution_l738_738091


namespace tan_alpha_third_quadrant_l738_738777

variables (α : ℝ)
-- α is an angle in the third quadrant
-- f(α) = (sin(α - π/2) * cos(3π/2 + α) * tan(π - α)) / (tan(-α - π) * sin(-α - π))
def f (α : ℝ) : ℝ := (Real.sin (α - Real.pi / 2) * Real.cos (3 * Real.pi / 2 + α) * Real.tan (Real.pi - α)) / (Real.tan (-α - Real.pi) * Real.sin (-α - Real.pi))

theorem tan_alpha_third_quadrant (hα : f α = 4 / 5) :
  ∃ (α : ℝ), α ∈ Set.Icc (Real.pi) (3 * Real.pi / 2) ∧ Real.tan α = 3 / 4 :=
by
  sorry

end tan_alpha_third_quadrant_l738_738777


namespace no_solution_prob1_l738_738175

theorem no_solution_prob1 : ¬ ∃ x : ℝ, x ≠ 2 ∧ (1 / (x - 2) + 3 = (1 - x) / (2 - x)) :=
by
  sorry

end no_solution_prob1_l738_738175


namespace bowling_tournament_l738_738542

def num_possible_orders : ℕ := 32

theorem bowling_tournament : num_possible_orders = 2 * 2 * 2 * 2 * 2 := by
  -- The structure of the playoff with 2 choices per match until all matches are played,
  -- leading to a total of 5 rounds and 2 choices per round, hence 2^5 = 32.
  sorry

end bowling_tournament_l738_738542


namespace accurate_river_length_l738_738649

-- Define the given conditions
def length_GSA := 402
def length_AWRA := 403
def error_margin := 0.5
def probability_of_error := 0.04

-- State the theorem based on these conditions
theorem accurate_river_length : 
  ∀ Length_GSA Length_AWRA error_margin probability_of_error, 
  Length_GSA = 402 → 
  Length_AWRA = 403 → 
  error_margin = 0.5 → 
  probability_of_error = 0.04 → 
  (this based on independent measurements with above error margins)
  combined_length = 402.5 ∧ combined_probability_of_error = 0.04 :=
by 
  -- Proof to be completed
  sorry

end accurate_river_length_l738_738649


namespace Rio_Coralio_Length_Estimate_l738_738643

def RioCoralioLength := 402.5
def GSA_length := 402
def AWRA_length := 403
def error_margin := 0.5
def error_probability := 0.04

theorem Rio_Coralio_Length_Estimate :
  ∀ (L_GSA L_AWRA : ℝ) (margin error_prob : ℝ),
  L_GSA = GSA_length ∧ L_AWRA = AWRA_length ∧ 
  margin = error_margin ∧ error_prob = error_probability →
  (RioCoralioLength = 402.5) ∧ (error_probability = 0.04) := 
by 
  intros L_GSA L_AWRA margin error_prob h,
  sorry

end Rio_Coralio_Length_Estimate_l738_738643


namespace sum_of_inverses_mod_17_l738_738242

theorem sum_of_inverses_mod_17 :
  (3⁻¹ + 3⁻² + 3⁻³ + 3⁻⁴ + 3⁻⁵ + 3⁻⁶) % 17 = 9 := sorry

end sum_of_inverses_mod_17_l738_738242


namespace no_solution_system_l738_738371

theorem no_solution_system (v : ℝ) :
  (∀ x y z : ℝ, ¬(x + y + z = v ∧ x + v * y + z = v ∧ x + y + v^2 * z = v^2)) ↔ (v = -1) :=
  sorry

end no_solution_system_l738_738371


namespace interval_contains_root_l738_738568

noncomputable def f (x : ℝ) : ℝ := 2^x + x - 2

theorem interval_contains_root :
  ∃ x ∈ Ioo (0 : ℝ) 1, f x = 0 :=
by
  sorry

end interval_contains_root_l738_738568


namespace tangent_circles_DKH_FKM_l738_738491

open EuclideanGeometry

variables {A B C H M F D K : Point}

-- Conditions
def is_triangle (A B C : Point) : Prop := 
  A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ collinear A B C = false

def orthocenter (H : Point) (A B C : Point) : Prop :=
  altitude A H B C ∧ altitude B H A C ∧ altitude C H A B

def midpoint (M : Point) (B C : Point) : Prop :=
  ∃ (l : Line), M ∈ l ∧ M = midpoint_of_segment B C

def foot_of_altitude (F A : Point) (B C : Point) : Prop :=
  is_point_on_altitude F A B C

def point_on_circumcircle_with_angle (D : Point) (circumcircle : Circle) (H A : Point): Prop :=
  lies_on D circumcircle ∧ ∠ H D A = 90

def point_on_circumcircle_with_angle_2 (K : Point) (circumcircle : Circle) (D H : Point): Prop :=
  lies_on K circumcircle ∧ ∠ D K H = 90

-- Theorem to prove
theorem tangent_circles_DKH_FKM
  (h1 : is_triangle A B C)
  (h2 : orthocenter H A B C)
  (h3 : midpoint M B C)
  (h4 : foot_of_altitude F A B C)
  (h5 : ∃ (circumcircle : Circle), point_on_circumcircle_with_angle D circumcircle H A)
  (h6 : ∃ (circumcircle : Circle), point_on_circumcircle_with_angle_2 K circumcircle D H) :
  tangent_at K (circumcircle_of D K H) (circumcircle_of F K M) := 
begin
  sorry
end

end tangent_circles_DKH_FKM_l738_738491


namespace problem1_problem2_l738_738715

theorem problem1 : -7 + (+5) - (-10) = 8 :=
by
  sorry

theorem problem2 : 3 ÷ (-1/3) × (-3) = 27 :=
by
  sorry

end problem1_problem2_l738_738715


namespace impossible_square_construction_l738_738206

theorem impossible_square_construction (k : ℕ) (h : 2 ≤ k ∧ k ≤ 2024) : 
  ¬ ∃ (rectangles : list (ℕ × ℕ)), (∀ r ∈ rectangles, r.1 = 1 ∧ r.2 ∈ finset.range 2025 ∧ r.2 ≤ k) ∧ (rectangles_length rectangles = k * k) :=
sorry

end impossible_square_construction_l738_738206


namespace sum_of_inverses_mod_17_l738_738244

theorem sum_of_inverses_mod_17 :
  (3⁻¹ + 3⁻² + 3⁻³ + 3⁻⁴ + 3⁻⁵ + 3⁻⁶) % 17 = 9 := sorry

end sum_of_inverses_mod_17_l738_738244


namespace total_amount_l738_738732

-- Definitions based on the problem conditions
def jack_amount : ℕ := 26
def ben_amount : ℕ := jack_amount - 9
def eric_amount : ℕ := ben_amount - 10

-- Proof statement
theorem total_amount : jack_amount + ben_amount + eric_amount = 50 :=
by
  -- Sorry serves as a placeholder for the actual proof
  sorry

end total_amount_l738_738732


namespace prob_P_on_parabola_l738_738891

noncomputable def probability_on_parabola : ℚ :=
  let cubeA_faces := {1, 2, 3, 4, 5, 6}
  let cubeB_faces := {1, 2, 3, 4, 5, 6}
  let total_outcomes := finset.card (finset.product cubeA_faces cubeB_faces)
  let favorable_outcomes := finset.filter (λ p: ℕ × ℕ, p.snd = -(p.fst^2) + 4 * p.fst) (finset.product cubeA_faces cubeB_faces)
  finset.card favorable_outcomes / total_outcomes

theorem prob_P_on_parabola : probability_on_parabola = 1 / 12 :=
by
  sorry

end prob_P_on_parabola_l738_738891


namespace ann_older_than_susan_l738_738703

variables (A S : ℕ)

theorem ann_older_than_susan (h1 : S = 11) (h2 : A + S = 27) : A - S = 5 := by
  -- Proof is skipped
  sorry

end ann_older_than_susan_l738_738703


namespace price_per_book_sold_l738_738981

-- Definitions based on the given conditions
def total_books_before_sale : ℕ := 3 * 50
def books_sold : ℕ := 2 * 50
def total_amount_received : ℕ := 500

-- Target statement to be proved
theorem price_per_book_sold :
  (total_amount_received : ℚ) / books_sold = 5 :=
sorry

end price_per_book_sold_l738_738981


namespace peanuts_in_box_proof_l738_738447

variable (initial_peanuts : ℕ) (added_peanuts : ℕ)

def total_peanuts_in_box (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

theorem peanuts_in_box_proof 
  (h_initial : initial_peanuts = 4)
  (h_added : added_peanuts = 8) :
  total_peanuts_in_box initial_peanuts added_peanuts = 12 := by
  rw [h_initial, h_added]
  rw [total_peanuts_in_box]
  sorry

end peanuts_in_box_proof_l738_738447


namespace eq_holds_for_values_l738_738756

noncomputable def tg (α : Real) : Real := (Real.sin α) / (Real.cos α)
noncomputable def condition (α : Real) : Prop := (Real.sqrt (tg(α)^2 - Real.sin(α)^2)) = (tg(α) * Real.sin(α))
noncomputable def solution (α : Real) (k : Int) : Prop := (4*k - 1) * (Real.pi / 2) < α ∧ α < (4*k + 1) * (Real.pi / 2)

theorem eq_holds_for_values (α : Real) : 
  (∃ k : Int, solution α k) ↔ condition α :=
sorry

end eq_holds_for_values_l738_738756


namespace rectangle_length_l738_738575

noncomputable def length_of_rectangle (π : ℝ) (C_semi : ℝ) (b : ℝ) : ℝ :=
  let s := C_semi / (π / 2 + 1) in
  let P_square := 4 * s in
  let P_rectangle := P_square in
  let l := (P_rectangle - 2 * b) / 2 in
  l

theorem rectangle_length (π : ℝ) (C_semi : ℝ) (b : ℝ) (l : ℝ) :
  C_semi = 29.85 →
  b = 16 →
  π = 3.14159 →
  length_of_rectangle π C_semi b = 7.22 :=
by
  intros h1 h2 h3 
  rw [h1, h2, h3]
  have : length_of_rectangle 3.14159 29.85 16 = 7.22 := sorry
  exact this

end rectangle_length_l738_738575


namespace count_valid_integers_between_5000_6000_l738_738804

theorem count_valid_integers_between_5000_6000 : 
  ∃ n, n = 21 ∧ ∀ (d : ℕ) (x y z : ℕ), d = 5 → (5 ≤ 5 * 1000 + x * 100 + y * 10 + z < 6000) → (x + y + z = 5) :=
sorry

end count_valid_integers_between_5000_6000_l738_738804


namespace total_socks_l738_738120

-- Definitions based on conditions
def red_pairs : ℕ := 20
def red_socks : ℕ := red_pairs * 2
def black_socks : ℕ := red_socks / 2
def white_socks : ℕ := 2 * (red_socks + black_socks)

-- The main theorem we want to prove
theorem total_socks :
  (red_socks + black_socks + white_socks) = 180 := by
  sorry

end total_socks_l738_738120


namespace main_theorem_l738_738853

variables {α : Type} [metric_space α] [normed_group α] [add_comm_group α] [module ℝ α]

/-- Variables and coordinates -/
variables {A B C P I : α}
variables {a b c : ℝ}

/-- Distances from points PA, PB, PC, IA, IB and IC -/
noncomputable def dist_sq (x y : α) : ℝ := ∥x - y∥^2

/-- Assumptions -/
axiom incenter (h : is_incenter I A B C)
axiom side_lengths (h : ∃ (a b c : ℝ), a = dist A B ∧ b = dist B C ∧ c = dist C A)

theorem main_theorem
  (a b c : ℝ)
  (h₁ : dist_sq P A = dist_sq I A)
  (h₂ : dist_sq P B = dist_sq I B)
  (h₃ : dist_sq P C = dist_sq I C)
  (hI : is_incenter I A B C) :
a * (dist_sq P A) + b * (dist_sq P B) + c * (dist_sq P C) =
  a * (dist_sq I A) + b * (dist_sq I B) + c * (dist_sq I C) + 
  (a + b + c) * (dist_sq I P) := sorry


end main_theorem_l738_738853


namespace price_of_straight_paddle_price_of_horizontal_paddle_cost_effective_solution_l738_738290

-- Definitions for the constants and variables
def price_per_ball : ℕ := 2
def ball_count_per_pair : ℕ := 10

def total_pairs (straight_pairs horizontal_pairs : ℕ) : ℕ := straight_pairs + horizontal_pairs
def total_cost (straight_price horizontal_price balls_price straight_pairs horizontal_pairs : ℕ) : ℕ := 
  (straight_price + balls_price) * straight_pairs + (horizontal_price + balls_price) * horizontal_pairs

-- Given the problem conditions as hypotheses
axiom condition_1 : total_cost 20 15 (ball_count_per_pair * price_per_ball) 9000
axiom condition_2 : (10 * 260 + (ball_count_per_pair * price_per_ball)) = (5 * 220 + (ball_count_per_pair * price_per_ball)) + 1600
axiom condition_3 : ∀ m h, m + h = 40 → m ≤ 3 * h → 
  (220 + ball_count_per_pair * price_per_ball) * m + 
  (260 + ball_count_per_pair * price_per_ball) * h = 10000

-- Proof tasks as Lean theorems
theorem price_of_straight_paddle : (x : ℤ) → x = 220 :=
sorry

theorem price_of_horizontal_paddle : (y : ℤ) → y = 260 :=
sorry

theorem cost_effective_solution : ∃ m h, m = 30 ∧ h = 10 ∧ 
  (total_pairs m h = 40 ∧ m ≤ 3 * h → 
  total_cost 220 260 (ball_count_per_pair * price_per_ball) m h = 10000) :=
sorry

end price_of_straight_paddle_price_of_horizontal_paddle_cost_effective_solution_l738_738290


namespace fifth_term_in_geometric_sequence_l738_738339

variable (y : ℝ)

def geometric_sequence : ℕ → ℝ
| 0       => 3
| (n + 1) => geometric_sequence n * (3 * y)

theorem fifth_term_in_geometric_sequence (y : ℝ) : 
  geometric_sequence y 4 = 243 * y^4 :=
sorry

end fifth_term_in_geometric_sequence_l738_738339


namespace not_square_l738_738132

theorem not_square (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : ¬ ∃ n : ℕ, n * n = a ^ 2 + Int.ceil (4 * a ^ 2 / b) := 
sorry

end not_square_l738_738132


namespace total_pies_l738_738980

-- Define the number of each type of pie.
def apple_pies : Nat := 2
def pecan_pies : Nat := 4
def pumpkin_pies : Nat := 7

-- Prove the total number of pies.
theorem total_pies : apple_pies + pecan_pies + pumpkin_pies = 13 := by
  sorry

end total_pies_l738_738980


namespace regular_tetrahedron_varphi_equals_epsilon_l738_738308

noncomputable def regular_tetrahedron_angle_equality : Prop :=
  let varphi : ℝ := (toRadians 67 + toRadians (22 / 60)) in
  let epsilon : ℝ := varphi in
  varphi = epsilon

theorem regular_tetrahedron_varphi_equals_epsilon :
  regular_tetrahedron_angle_equality :=
by
  sorry  -- Proof omitted

end regular_tetrahedron_varphi_equals_epsilon_l738_738308


namespace seating_arrangements_valid_count_l738_738832

theorem seating_arrangements_valid_count :
  let Martians := 6;
  let Venusians := 5;
  let Earthlings := 4;
  (∃ (N : ℕ),
    (N = 1) ∧
    Martians! * Venusians! * Earthlings! = 720 * 120 * 24) :=
begin
  sorry
end

end seating_arrangements_valid_count_l738_738832


namespace find_value_l738_738848

noncomputable def equilateral_triangle (A B C: Type*) [EuclideanGeometry A] [EuclideanGeometry B] [EuclideanGeometry C] 
(AB BC CA: ℝ) (h1: AB = 1) (h2: is_equilateral A B C) := true

variables {A B C D E F X Y Z : Point}
variables [EuclideanGeometry A] [EuclideanGeometry B] [EuclideanGeometry C]
variables (h_ABC_eq : equilateral_triangle A B C 1)

lemma perpendicular_def (XY DE: Line) (h : XY ⊥ DE) := true

noncomputable def unique_value (h : ∀ DEF XYZ, 
(DEF : EuclideanTriangle) (XYZ : EuclideanTriangle),
DEF.side DE / 20 = DE.sideline EF / 22 ∧
EF.side FD / 38 ∧
(XYZ.sidelength XY ⊥ DE) ∧ 
(YZ ⊥ EF) ∧ 
(ZX ⊥ FD) → 
frac_value 1/DEF.area + 1/XYZ.area = (sqrt 2 97 + sqrt 3 40) / 15) : Prop := 
true

theorem find_value (A B C D E F X Y Z : Point)
(h1 : equilateral_triangle A B C 1)
(h2 : ∃ DEF XYZ : EuclideanTriangle, 
DEF.sidelength DE / 20 = DEF.sidelength EF / 22 ∧
DEF.angle FD = ∠(DEF)
∧ ZXY.sidelength XY ⊥ DE
∧ ZXY.sidelength YZ ⊥ EF
∧ ZXY.sidelength ZX ⊥ FD
)
:
∃ value : ℝ, 
frac_value 1/[DEF] + 1/[XYZ] = value :=
begin
    use (97 * sqrt(2) + 40 * sqrt(3)) / 15,
    sorry
end

end find_value_l738_738848


namespace smallest_n_13_l738_738503

theorem smallest_n_13 :
  ∃ n m r : ℝ, n = 13 ∧ m = (n + r)^3 ∧ n ∈ ℕ ∧ r > 0 ∧ r < 1 / 500 := sorry

end smallest_n_13_l738_738503


namespace combined_river_length_estimate_l738_738635

def river_length_GSA := 402 
def river_error_GSA := 0.5 
def river_prob_error_GSA := 0.04 

def river_length_AWRA := 403 
def river_error_AWRA := 0.5 
def river_prob_error_AWRA := 0.04 

/-- 
Given the measurements from GSA and AWRA, 
the combined estimate of the river's length, Rio-Coralio, is 402.5 km,
and the probability of error for this combined estimate is 0.04.
-/
theorem combined_river_length_estimate :
  ∃ l : ℝ, l = 402.5 ∧ ∀ p : ℝ, (p = 0.04) :=
sorry

end combined_river_length_estimate_l738_738635


namespace range_of_g_l738_738322

open Real

noncomputable def g (x : ℝ) : ℝ := (cos x) ^ 4 + (sin x) ^ 2

theorem range_of_g : set.range g = {y | 3 / 4 ≤ y ∧ y ≤ 1} :=
by
  sorry

end range_of_g_l738_738322


namespace largest_prime_divisor_of_sum_of_squares_24_75_l738_738358

theorem largest_prime_divisor_of_sum_of_squares_24_75 :
  ∃ p : ℕ, prime p ∧ p = 53 ∧ ∀ q : ℕ, prime q ∧ (q ∣ (24^2 + 75^2)) → q ≤ p :=
by
  sorry

end largest_prime_divisor_of_sum_of_squares_24_75_l738_738358


namespace find_boys_and_girls_l738_738100

noncomputable def number_of_boys_and_girls (a b c d : Nat) : (Nat × Nat) := sorry

theorem find_boys_and_girls : 
  ∃ m d : Nat,
  (∀ (a b c : Nat), 
    ((a = 15 ∨ b = 18 ∨ c = 13) ∧ 
    (a.mod 4 = 3 ∨ b.mod 4 = 2 ∨ c.mod 4 = 1)) 
    → number_of_boys_and_girls a b c d = (16, 14)) :=
sorry

end find_boys_and_girls_l738_738100


namespace angle_Q_of_extended_sides_of_dodecagon_l738_738171

theorem angle_Q_of_extended_sides_of_dodecagon (A M F G Q : Point)
  (dodecagon : Set Point)
  (AM FG : Line)
  (h_reg_dodecagon : regular_dodecagon dodecagon)
  (h_AM_in_dodecagon : AM ∈ dodecagon)
  (h_FG_in_dodecagon : FG ∈ dodecagon)
  (h_AM_extends : extends_AM_to Q)
  (h_FG_extends : extends_FG_to Q)
  (h_parallel : parallel AM FG) :
  angle Q = 60 := 
sorry

end angle_Q_of_extended_sides_of_dodecagon_l738_738171


namespace cube_colorings_distinguishable_l738_738296

-- Define the problem
def cube_construction_distinguishable_ways : Nat :=
  30

-- The theorem we need to prove
theorem cube_colorings_distinguishable :
  ∃ (ways : Nat), ways = cube_construction_distinguishable_ways :=
by
  sorry

end cube_colorings_distinguishable_l738_738296


namespace problem_1_problem_2_problem_3_l738_738054

noncomputable def f (a x : ℝ) : ℝ := a * x - Real.log x

theorem problem_1 :
  (∀ x : ℝ, f 1 x ≥ f 1 1) :=
by sorry

theorem problem_2 (x e : ℝ) (hx : x ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1)) (hf : f a x = 1) :
  0 ≤ a ∧ a ≤ 1 :=
by sorry

theorem problem_3 (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Ici 1 → f a x ≥ f a (1 / x)) → 1 ≤ a :=
by sorry

end problem_1_problem_2_problem_3_l738_738054


namespace octopus_legs_proof_l738_738825

section OctopusLegs

variable (Blue Green Red Yellow : ℕ)

-- Conditions:
-- - Each octopus can only have 6, 7, or 8 legs.
-- - Those with 7 legs always lie, while others (6 or 8 legs) always tell the truth.
-- - Blue states the total number of legs is 25
-- - Green states the total number of legs is 26
-- - Red states the total number of legs is 27
-- - Yellow states the total number of legs is 28

def isLie (legs : ℕ) : Prop := legs = 7
def isTruth (legs : ℕ) : Prop := legs = 6 ∨ legs = 8

variable (TotalLegs : ℕ)

-- The total number of legs according to each octopus:
def blueStatement := Blue + Green + Red + Yellow = 25 
def greenStatement := Blue + Green + Red + Yellow = 26 
def redStatement := Blue + Green + Red + Yellow = 27 
def yellowStatement := Blue + Green + Red + Yellow = 28 

theorem octopus_legs_proof :
  -- Given each octopus can have 6, 7, or 8 legs
  (Blue = 6 ∨ Blue = 7 ∨ Blue = 8) ∧
  (Green = 6 ∨ Green = 7 ∨ Green = 8) ∧
  (Red = 6 ∨ Red = 7 ∨ Red = 8) ∧
  (Yellow = 6 ∨ Yellow = 7 ∨ Yellow = 8) ∧

  -- And octopuses with 7 legs lie and with 6 or 8 tell truth
  (isLie Blue ↔ ¬blueStatement) ∧ (isTruth Blue ↔ blueStatement) ∧
  (isLie Green ↔ ¬greenStatement) ∧ (isTruth Green ↔ greenStatement) ∧
  (isLie Red ↔ ¬redStatement) ∧ (isTruth Red ↔ redStatement) ∧
  (isLie Yellow ↔ ¬yellowStatement) ∧ (isTruth Yellow ↔ yellowStatement) ∧

  -- Only one octopus tells the truth
  ((isTruth Blue ∧ isLie Green ∧ isLie Red ∧ isLie Yellow) ∨
   (isLie Blue ∧ isTruth Green ∧ isLie Red ∧ isLie Yellow) ∨
   (isLie Blue ∧ isLie Green ∧ isTruth Red ∧ isLie Yellow) ∨
   (isLie Blue ∧ isLie Green ∧ isLie Red ∧ isTruth Yellow)) ∧

  -- Prove:
  (Blue = 7 ∧ Green = 7 ∧ Red = 6 ∧ Yellow = 7) :=
sorry

end OctopusLegs

end octopus_legs_proof_l738_738825


namespace div_c_a_l738_738442

theorem div_c_a (a b c : ℚ) (h1 : a = 3 * b) (h2 : b = 2 / 5 * c) :
  c / a = 5 / 6 := 
by
  sorry

end div_c_a_l738_738442


namespace angle_projections_l738_738994

-- Define the parabola and its associated properties
def parabola (p : ℝ) : set (ℝ × ℝ) := {point | point.snd ^ 2 = 2 * p * point.fst}

-- Define the focus of the parabola
def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)

-- Define the directrix of the parabola
def directrix (p : ℝ) : set (ℝ × ℝ) := {point | point.fst = -p / 2}

-- Define the projections of points A and B onto the directrix
def projection (point : ℝ × ℝ) (p : ℝ) : ℝ × ℝ := (-p / 2, point.snd)

-- The angle ∠A₁FB₁ is 90° given the conditions
theorem angle_projections (p : ℝ) (A B : ℝ × ℝ)
  (hA : A ∈ parabola p) (hB : B ∈ parabola p) 
  (line_through_focus : ∃ m b, A.snd = m * A.fst + b ∧ B.snd = m * B.fst + b ∧ focus p = (m * (p / 2) + b, 0)) :
  ∠ (projection A p) (focus p) (projection B p) = 90 :=
sorry

end angle_projections_l738_738994


namespace find_a_given_b_l738_738895

theorem find_a_given_b
  (a b : ℝ) 
  (h_inv : ∀ a b : ℝ, a^3 * b^2 = 256)
  (h_a4_b2 : a = 4 ∧ b = 2) : 
  a = 4^(1/3) :=
by
  have k_val := h_inv 4 2
  have eq_k : 256 = 4^3 * 2^2 := by norm_num
  rw [h_a4_b2] at k_val
  sorry

end find_a_given_b_l738_738895


namespace winding_clock_available_time_l738_738605

theorem winding_clock_available_time
    (minute_hand_restriction_interval: ℕ := 5) -- Each interval the minute hand restricts
    (hour_hand_restriction_interval: ℕ := 60) -- Each interval the hour hand restricts
    (intervals_per_12_hours: ℕ := 2) -- Number of restricted intervals in each 12-hour cycle
    (minutes_in_day: ℕ := 24 * 60) -- Total minutes in 24 hours
    : (minutes_in_day - ((minute_hand_restriction_interval * intervals_per_12_hours * 12) + 
                         (hour_hand_restriction_interval * intervals_per_12_hours * 2))) = 1080 :=
by
  -- Skipping the proof steps
  sorry

end winding_clock_available_time_l738_738605


namespace bricks_needed_l738_738622

noncomputable def volume (length : ℝ) (width : ℝ) (height : ℝ) : ℝ := length * width * height

theorem bricks_needed
  (brick_length : ℝ)
  (brick_width : ℝ)
  (brick_height : ℝ)
  (wall_length : ℝ)
  (wall_height : ℝ)
  (wall_thickness : ℝ)
  (hl : brick_length = 40)
  (hw : brick_width = 11.25)
  (hh : brick_height = 6)
  (wl : wall_length = 800)
  (wh : wall_height = 600)
  (wt : wall_thickness = 22.5) :
  (volume wall_length wall_height wall_thickness / volume brick_length brick_width brick_height) = 4000 := by
  sorry

end bricks_needed_l738_738622


namespace extremum_f_range_a_for_no_zeros_l738_738791

noncomputable def f (a b x : ℝ) : ℝ :=
  (a * (x - 1) + b * Real.exp x) / Real.exp x

theorem extremum_f (a b : ℝ) (h_a_ne_zero : a ≠ 0) :
  (∃ (x : ℝ), a = -1 ∧ b = 0 ∧ f a b x = -1 / Real.exp 2) := sorry

theorem range_a_for_no_zeros (a : ℝ) :
  (∀ x : ℝ, a * x - a + Real.exp x ≠ 0) ↔ (-Real.exp 2 < a ∧ a < 0) := sorry

end extremum_f_range_a_for_no_zeros_l738_738791


namespace probability_heart_then_king_of_clubs_l738_738458

theorem probability_heart_then_king_of_clubs : 
  let deck := 52
  let hearts := 13
  let remaining_cards := deck - 1
  let king_of_clubs := 1
  let first_card_heart_probability := (hearts : ℝ) / deck
  let second_card_king_of_clubs_probability := (king_of_clubs : ℝ) / remaining_cards
  first_card_heart_probability * second_card_king_of_clubs_probability = 1 / 204 :=
by
  sorry

end probability_heart_then_king_of_clubs_l738_738458


namespace sum_of_inverses_mod_17_l738_738237

noncomputable def inverse_sum_mod_17 : ℤ :=
  let a1 := Nat.gcdA 3 17 in -- 3^{-1} mod 17
  let a2 := Nat.gcdA (3^2) 17 in -- 3^{-2} mod 17
  let a3 := Nat.gcdA (3^3) 17 in -- 3^{-3} mod 17
  let a4 := Nat.gcdA (3^4) 17 in -- 3^{-4} mod 17
  let a5 := Nat.gcdA (3^5) 17 in -- 3^{-5} mod 17
  let a6 := Nat.gcdA (3^6) 17 in -- 3^{-6} mod 17
  (a1 + a2 + a3 + a4 + a5 + a6) % 17

theorem sum_of_inverses_mod_17 : inverse_sum_mod_17 = 7 := sorry

end sum_of_inverses_mod_17_l738_738237


namespace sum_possible_values_l738_738092

theorem sum_possible_values (x : ℤ) (h : ∃ y : ℤ, y = (3 * x + 13) / (x + 6)) :
  ∃ s : ℤ, s = -2 + 8 + 2 + 4 :=
sorry

end sum_possible_values_l738_738092


namespace coin_toss_min_n_l738_738668

theorem coin_toss_min_n (n : ℕ) :
  (1 : ℝ) - (1 / (2 : ℝ)) ^ n ≥ 15 / 16 → n ≥ 4 :=
by
  sorry

end coin_toss_min_n_l738_738668


namespace pi_S_minus_S1_over_S2_l738_738469

noncomputable def triangle_areas (a b c : ℝ) : ℝ × ℝ × ℝ :=
  let S := (b * c) / 2
  let p := (a + b + c) / 2
  let r := S / p
  let S1 := π * r ^ 2
  let R := a / 2
  let S2 := π * R ^ 2
  (S, S1, S2)

theorem pi_S_minus_S1_over_S2 (a b c : ℝ) (h₁ : a^2 = b^2 + c^2) :
  let (S, S1, S2) := triangle_areas a b c
  in π * (S - S1) / S2 < 1 / (π - 1) :=
by
  sorry

end pi_S_minus_S1_over_S2_l738_738469


namespace bubble_bath_per_guest_l738_738118

def rooms_couple : ℕ := 13
def rooms_single : ℕ := 14
def total_bubble_bath : ℕ := 400

theorem bubble_bath_per_guest :
  (total_bubble_bath / (rooms_couple * 2 + rooms_single)) = 10 :=
by
  sorry

end bubble_bath_per_guest_l738_738118


namespace cos_of_transformed_angle_l738_738013

theorem cos_of_transformed_angle 
  (x : ℝ) 
  (h : sin (x + π / 6) = -1 / 3) :
  cos (2 * π /3 - 2 * x) = -7 / 9 :=
sorry

end cos_of_transformed_angle_l738_738013


namespace range_of_a_l738_738063

theorem range_of_a (a : ℝ) (h : a > 0) (h1 : ∀ x : ℝ, |a * x - 1| + |a * x - a| ≥ 1) : a ≥ 2 := 
sorry

end range_of_a_l738_738063


namespace x_coordinate_incenter_eq_l738_738305

theorem x_coordinate_incenter_eq {x y : ℝ} :
  (y = 0 → x + y = 3 → x = 0) → 
  (y = x → y = -x + 3 → x = 3 / 2) :=
by
  sorry

end x_coordinate_incenter_eq_l738_738305


namespace proof_problem_l738_738831

-- Conditions/Definitions
def C1_param_eq (φ : ℝ) : ℝ × ℝ := (Real.cos φ, Real.sin φ)

def C2_param_eq (a b φ : ℝ) : ℝ × ℝ := (a * Real.cos φ, b * Real.sin φ)

theorem proof_problem
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
  (h4 : Real.dist (1, 0) (a, 0) = 2)
  (h5 : Real.dist (0, 1) (0, b) = 0)
  (α1 α2 : ℝ)
  (intersect_alpha1 : α1 = π/4)
  (intersect_alpha2 : α2 = -π/4)
  : a = 3 ∧ b = 1 ∧
    (let A1 := (Real.cos α1, Real.sin α1),
         B1 := (3 / √10, 3 / √10): ℝ × ℝ in ρ * Real.cos θ = √2 / 2) ∧
    (let A2 := (Real.cos α2, Real.sin α2),
         B2 := (3 / √10, -3 / √10): ℝ × ℝ in ρ * Real.cos θ = 3*√10 / 10) :=
sorry

end proof_problem_l738_738831


namespace angle_A1_F_B1_90_deg_l738_738993

theorem angle_A1_F_B1_90_deg (p : ℝ) (A B A1 B1 F : ℝ × ℝ) :
  let parabola : set (ℝ × ℝ) := { point | (point.2)^2 = 2 * p * point.1 },
      focus : ℝ × ℝ := (p / 2, 0),
      directrix := { point : (ℝ × ℝ) | point.1 = - p / 2 },
      projection1 : (ℝ × ℝ) := (-(p / 2), A.2),
      projection2 : (ℝ × ℝ) := (-(p / 2), B.2) in
  A ∈ parabola ∧ B ∈ parabola ∧ 
  A1 = projection1 ∧ B1 = projection2 ∧ F = focus →
  ∠ A1 F B1 = 90 :=
begin
  sorry
end

end angle_A1_F_B1_90_deg_l738_738993


namespace negation_of_universal_l738_738195

theorem negation_of_universal :
  ¬(∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) :=
by sorry

end negation_of_universal_l738_738195


namespace find_f_sin_75_l738_738398

-- Define the conditions of the problem
def is_zero_of (x : ℝ) (f : ℝ → ℝ) : Prop :=
  f x = 0

noncomputable def cos_75 := (Real.sqrt 6 - Real.sqrt 2) / 4
noncomputable def sin_75 := (Real.sqrt 6 + Real.sqrt 2) / 4

def f (x : ℝ) (a4 a3 a2 a1 a0 : ℤ) : ℝ :=
  a4 * x^4 + a3 * x^3 + a2 * x^2 + a1 * x + a0

-- Define the main theorem to prove
theorem find_f_sin_75 {a4 a3 a2 a1 a0 : ℤ} 
  (h1 : a4 ≠ 0) 
  (h2 : is_zero_of cos_75 (f ⟨a4, h1⟩ a3 a2 a1 a0))
  : f (sin_75) ⟨a4, h1⟩ a3 a2 a1 a0 = 0 := 
sorry

end find_f_sin_75_l738_738398


namespace combined_river_length_estimate_l738_738637

def river_length_GSA := 402 
def river_error_GSA := 0.5 
def river_prob_error_GSA := 0.04 

def river_length_AWRA := 403 
def river_error_AWRA := 0.5 
def river_prob_error_AWRA := 0.04 

/-- 
Given the measurements from GSA and AWRA, 
the combined estimate of the river's length, Rio-Coralio, is 402.5 km,
and the probability of error for this combined estimate is 0.04.
-/
theorem combined_river_length_estimate :
  ∃ l : ℝ, l = 402.5 ∧ ∀ p : ℝ, (p = 0.04) :=
sorry

end combined_river_length_estimate_l738_738637


namespace sum_of_inverses_mod_17_l738_738231

theorem sum_of_inverses_mod_17 :
  (3⁻¹ + 3⁻² + 3⁻³ + 3⁻⁴ + 3⁻⁵ + 3⁻⁶ : ℤ) % 17 = 7 :=
by {
  sorry
}

end sum_of_inverses_mod_17_l738_738231


namespace num_girls_l738_738202

-- Define conditions as constants
def ratio (B G : ℕ) : Prop := B = (5 * G) / 8
def total (B G : ℕ) : Prop := B + G = 260

-- State the proof problem
theorem num_girls (B G : ℕ) (h1 : ratio B G) (h2 : total B G) : G = 160 :=
by {
  -- actual proof omitted
  sorry
}

end num_girls_l738_738202


namespace pentagon_area_constant_infinitely_many_pentagons_with_given_properties_l738_738704

theorem pentagon_area_constant (A B C D E : Point)
  (h1 : area (triangle A B C) = 1)
  (h2 : area (triangle B C D) = 1)
  (h3 : area (triangle C D E) = 1)
  (h4 : area (triangle D E A) = 1)
  (h5 : area (triangle E A B) = 1) :
  ∃! S, (area (pentagon A B C D E) = S) :=
sorry

theorem infinitely_many_pentagons_with_given_properties :
  ∃ inf (A B C D E : Point),
    (area (triangle A B C) = 1) ∧
    (area (triangle B C D) = 1) ∧
    (area (triangle C D E) = 1) ∧
    (area (triangle D E A) = 1) ∧
    (area (triangle E A B) = 1) :=
sorry

end pentagon_area_constant_infinitely_many_pentagons_with_given_properties_l738_738704


namespace square_of_999_l738_738746

theorem square_of_999 : 999 * 999 = 998001 := by
  sorry

end square_of_999_l738_738746


namespace areas_sum_l738_738518

-- Define the points A, B, C and the intersection point P
variables {A B C P Q R : Type}
variables [metric_space A] [metric_space B] [metric_space C]
[is_point_of A] [is_point_of B] [is_point_of C]

-- Define the areas of the parallelograms involved
noncomputable def area_ABPQ : ℝ := sorry
noncomputable def area_CBPR : ℝ := sorry
noncomputable def area_ACRQ : ℝ := sorry

-- Conditions regarding the construction of parallelograms and point P
variables (ABPQ : parallelogram A B P Q) (CBPR : parallelogram C B P R) (ACRQ : parallelogram A C R Q)
variables (h1: parallelogram_area ABPQ = area_ABPQ) (h2: parallelogram_area CBPR = area_CBPR)
variables (h3: parallelogram_area ACRQ = area_ACRQ)

-- The theorem to be proven
theorem areas_sum : 
  parallelogram_area ACRQ = parallelogram_area ABPQ + parallelogram_area CBPR :=
by sorry

end areas_sum_l738_738518


namespace edward_lawns_forgotten_l738_738730

theorem edward_lawns_forgotten (dollars_per_lawn : ℕ) (total_lawns : ℕ) (total_earned : ℕ) (lawns_mowed : ℕ) (lawns_forgotten : ℕ) :
  dollars_per_lawn = 4 →
  total_lawns = 17 →
  total_earned = 32 →
  lawns_mowed = total_earned / dollars_per_lawn →
  lawns_forgotten = total_lawns - lawns_mowed →
  lawns_forgotten = 9 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end edward_lawns_forgotten_l738_738730


namespace area_of_lawn_l738_738294

def diameter_island : ℝ := 50
def diameter_flowerbed : ℝ := 20
def radius (d : ℝ) := d / 2

def area_circle (r : ℝ) := Real.pi * r ^ 2

theorem area_of_lawn : area_circle (radius diameter_island) - area_circle (radius diameter_flowerbed) = 1648.5 := 
by
  sorry

end area_of_lawn_l738_738294


namespace find_a_over_b_l738_738410

-- Define the function f(x) = x * exp(x)
def f (x : ℝ) : ℝ := x * Real.exp x

-- Define the point P(1, e)
def P : ℝ × ℝ := (1, Real.exp 1)

-- Define the line ax - by - 3 = 0
def line (a b x y : ℝ) : Prop := a * x - b * y - 3 = 0

-- Define the condition that the tangent lines are perpendicular
def perpendicular_tangents (a b : ℝ) : Prop :=
  let slope_f := (λ x, Real.exp x + x * Real.exp x) 1 in
  slope_f = 2 * Real.exp 1 ∧ a / b = -(1 / (2 * Real.exp 1))

-- The theorem to prove the value of a/b
theorem find_a_over_b (a b : ℝ) (hline : line a b 1 (Real.exp 1))
  (hperp : perpendicular_tangents a b) : a / b = -(1 / (2 * Real.exp 1)) :=
by
  sorry

end find_a_over_b_l738_738410


namespace value_of_fraction_l738_738438

theorem value_of_fraction (x y : ℤ) (h1 : x = 3) (h2 : y = 4) : (x^5 + 3 * y^3) / 9 = 48 :=
by
  sorry

end value_of_fraction_l738_738438


namespace equal_segments_division_l738_738687

theorem equal_segments_division (total_length : ℝ) (n : ℕ) (segment_index : ℕ) :
  total_length = 3 ∧ n = 4 ∧ segment_index = 2 →
  (∃ segment_length : ℝ, segment_length = total_length / n ∧ segment_length = 3 / 4)
  ∧ (∃ segment_length : ℝ, segment_length = total_length / n ∧ segment_length = 3 / 4) :=
by
  intros given_conditions
  rcases given_conditions with ⟨h1, h2, h3⟩
  use 3 / 4
  split;
  intros

  { have h_segment_fraction := 3 / 4
    rw [h1, h2] at h_segment_fraction
    use h_segment_fraction
    assumption },

  { have h_segment_length := 3 / 4
    rw [h1, h2, h3] at h_segment_length 
    use h_segment_length
    assumption
  }

end equal_segments_division_l738_738687


namespace find_matrix_N_l738_738742

noncomputable def cross_product_matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, -7, -1], ![7, 0, 3], ![1, -3, 0]]

theorem find_matrix_N (w : Vector3 ℝ) : 
  (cross_product_matrix).vecMul w = 
  (vector.vec3 3 (-1) 7).cross w := 
sorry

end find_matrix_N_l738_738742


namespace simplify_1_simplify_2_simplify_3_simplify_4_simplify_5_simplify_6_simplify_7_simplify_8_l738_738541

theorem simplify_1 : 1 + (-0.5) = 0.5 := by
  sorry

theorem simplify_2 : 2 - (+10.1) = -10.1 := by
  sorry

theorem simplify_3 : 3 + (+7) = 7 := by
  sorry

theorem simplify_4 : 4 - (-20) = 20 := by
  sorry

theorem simplify_5 : 5 + |(-2) / 3| = 2 / 3 := by
  sorry

theorem simplify_6 : 6 - |(-4) / 5| = -4 / 5 := by
  sorry

theorem simplify_7 : 7 + [-( -10 )] = 10 := by
  sorry

theorem simplify_8 : 8 - [-(- 2 + 6 / 7)] = -(2 + 6 / 7) := by
  sorry

end simplify_1_simplify_2_simplify_3_simplify_4_simplify_5_simplify_6_simplify_7_simplify_8_l738_738541


namespace statement_B_statement_C_statement_D_l738_738960

-- Statement B
theorem statement_B (a b c : ℝ) (h1 : a > b) (h2 : c < 0) : a^3 * c < b^3 * c :=
sorry

-- Statement C
theorem statement_C (a b c : ℝ) (h1 : c > a) (h2 : a > b) (h3 : b > 0) : (a / (c - a)) > (b / (c - b)) :=
sorry

-- Statement D
theorem statement_D (a b : ℝ) (h1 : a > b) (h2 : 1 / a > 1 / b) : a > 0 ∧ b < 0 :=
sorry

end statement_B_statement_C_statement_D_l738_738960


namespace g_five_eq_one_l738_738193

noncomputable def g : ℝ → ℝ := sorry

theorem g_five_eq_one (hx : ∀ x y : ℝ, g (x * y) = g x * g y) (h1 : g 1 ≠ 0) : g 5 = 1 :=
sorry

end g_five_eq_one_l738_738193


namespace greatest_value_l738_738741

theorem greatest_value (x : ℝ) : -x^2 + 9 * x - 18 ≥ 0 → x ≤ 6 :=
by
  sorry

end greatest_value_l738_738741


namespace common_chord_length_l738_738595

noncomputable def chord_length {r : ℝ} (alpha : ℝ) : ℝ := 
  sqrt (2 - 2 * Real.cos alpha)

theorem common_chord_length (r : ℝ) (alpha theta : ℝ) 
  (h0 : r = 1) 
  (h1 : sqrt 2 = 2 * Real.sin (theta / 2))
  (h2 : alpha = (90 : ℝ) - theta / 2) :
  chord_length r alpha = sqrt (2 - sqrt 2) :=
sorry

end common_chord_length_l738_738595


namespace number_of_B_students_in_history_class_l738_738823

def history_class_grades (prob_A prob_B prob_C prob_D : ℕ → ℝ) (total_students : ℕ) : Prop :=
  ∀ n_B : ℕ, (prob_A n_B = 0.5 * prob_B n_B) ∧ (prob_C n_B = 2 * prob_B n_B) ∧ (prob_D n_B = 0.5 * prob_B n_B)
    -> (total_students = n_B + prob_A n_B + prob_C n_B + prob_D n_B)

theorem number_of_B_students_in_history_class 
  (prob_A prob_B prob_C prob_D : ℕ → ℝ) (total_students : ℕ) :
  history_class_grades prob_A prob_B prob_C prob_D total_students → total_students = 52 → (∃ n_B : ℕ, n_B = 13) :=
by
  assume h : history_class_grades prob_A prob_B prob_C prob_D total_students,
  assume h_total : total_students = 52,
  sorry

end number_of_B_students_in_history_class_l738_738823


namespace one_over_a_lt_one_over_b_iff_ab_over_a3_minus_b3_gt_zero_l738_738811

theorem one_over_a_lt_one_over_b_iff_ab_over_a3_minus_b3_gt_zero
  (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (1 / a < 1 / b) ↔ ((a * b) / (a^3 - b^3) > 0) := 
by
  sorry

end one_over_a_lt_one_over_b_iff_ab_over_a3_minus_b3_gt_zero_l738_738811


namespace max_pawns_on_chessboard_l738_738955

-- Definitions of conditions
def white_attacks (p : ℕ × ℕ) (q : ℕ × ℕ) : Prop :=
  (q.1 = p.1 + 1) ∧ (q.2 = p.2 - 1 ∨ q.2 = p.2 + 1)

def black_attacks (p : ℕ × ℕ) (q : ℕ × ℕ) : Prop :=
  (q.1 = p.1 - 1) ∧ (q.2 = p.2 - 1 ∨ q.2 = p.2 + 1)

-- Statement of the problem
theorem max_pawns_on_chessboard :
  ∃ (white black : fin 9 → fin 9 → bool), 
    (∀ i j, ¬ ( ∃ k l, white i j ∧ white_attacks (i, j) (k, l) )) ∧ 
    (∀ i j, ¬ ( ∃ k l, black i j ∧ black_attacks (i, j) (k, l) )) ∧
    (∀ i j, ¬ ( ∃ k l, white i j ∧ black_attacks (i, j) (k, l) )) ∧
    (∀ i j, ¬ ( ∃ k l, black i j ∧ white_attacks (i, j) (k, l) )) ∧
    (∑ i j, (if white i j then 1 else 0) + (if black i j then 1 else 0) = 56) :=
sorry

end max_pawns_on_chessboard_l738_738955


namespace half_sqrt_eq_one_implies_number_l738_738075

theorem half_sqrt_eq_one_implies_number (x : ℝ) (h : 1 / 2 * real.sqrt x = 1) : x = 4 :=
sorry

end half_sqrt_eq_one_implies_number_l738_738075


namespace part1_part2_l738_738150

open Nat

-- Given conditions
def S (n : ℕ) : ℕ := 2^n - 1
def a (n : ℕ) : ℕ := 2^(n-1)
def b (n : ℕ) : ℕ := n / (a n)

-- First part: Prove S_n + 1 is geometric and find a_n
theorem part1 (n : ℕ) (h : n ∈ Nat) (h1 : S 1 = 1) (h2 : S (n+1) - 2 * S n = 1) :
  ∃ c, ∀ n, S n + 1 = c * 2^n ∧ a n = 2^(n-1) := sorry

-- Second part: Find T_n given the a_n and b_n
def T (n : ℕ) : ℕ := ∑ i in range n, b i

theorem part2 (n : ℕ) (h : n ∈ Nat) :
  T n = 4 - (n + 2) / 2^(n-1) := sorry

end part1_part2_l738_738150


namespace prove_rhombus_solutions_l738_738309

noncomputable def rhombus_solutions (square_center : Point) (circle_center : Point) (rhombus_in_square : Rhombus) (circle_in_rhombus : Circle) : ℕ :=
  if valid_configuration square_center circle_center rhombus_in_square circle_in_rhombus then 
    if two_rhombuses square_center circle_center rhombus_in_square circle_in_rhombus then 2
    else if one_rhombus square_center circle_center rhombus_in_square circle_in_rhombus then 1
    else 0
  else 0

theorem prove_rhombus_solutions (square_center : Point) (circle_center : Point) 
                                (rhombus_in_square : Rhombus) (circle_in_rhombus : Circle) :
  valid_configuration square_center circle_center rhombus_in_square circle_in_rhombus →
  rhombus_solutions square_center circle_center rhombus_in_square circle_in_rhombus ∈ {0, 1, 2} :=
by {
  sorry
}

end prove_rhombus_solutions_l738_738309


namespace div_c_a_l738_738441

theorem div_c_a (a b c : ℚ) (h1 : a = 3 * b) (h2 : b = 2 / 5 * c) :
  c / a = 5 / 6 := 
by
  sorry

end div_c_a_l738_738441


namespace handshakes_mod_1000_l738_738822

-- Define the problem  and its conditions

def num_ways_handshakes (n : ℕ) : ℕ :=
  match n with
  | 3     => 1
  | 4     => 3
  | 5     => 6
  | 6     => 30
  | 9     => 
    let case1 := 30 -- f_6 
    let case2 := 6 * 6 -- 6 * f_5
    let case3 := 6 * 5 * 3 -- 6 * 5 * f_4
    let case4 := 4 * 1 -- 4 * f_3
    let case5 := 4 * 3 * 2 * 1
    28 * (case1 + case2 + case3 + case4 + case5)
  | _     => 0 -- default case

theorem handshakes_mod_1000 : (num_ways_handshakes 9) % 1000 = 152 :=
by
  have h : num_ways_handshakes 9 = 5152 := by
    simp [num_ways_handshakes]
  rw [h]
  norm_num
  sorry

end handshakes_mod_1000_l738_738822


namespace inverse_sum_mod_l738_738225

theorem inverse_sum_mod (h1 : ∃ k, 3^6 ≡ 1 [MOD 17])
                        (h2 : ∃ k, 3 * 6 ≡ 1 [MOD 17]) : 
  (6 + 9 + 2 + 1 + 6 + 1) % 17 = 8 :=
by
  cases h1 with k1 h1
  cases h2 with k2 h2
  sorry

end inverse_sum_mod_l738_738225


namespace trig_identity_proof_l738_738283

noncomputable theory

-- Defining the angles
def A : ℝ := real.pi * 10 / 180
def B : ℝ := real.pi * 50 / 180
def C : ℝ := real.pi * 70 / 180

-- Statement of the problem
theorem trig_identity_proof :
  6 * real.cos A * real.cos B * real.cos C + real.sin A * real.sin B * real.sin C = (1 + real.sqrt 3) / 4 :=
sorry

end trig_identity_proof_l738_738283


namespace sum_of_inverses_mod_17_l738_738228

theorem sum_of_inverses_mod_17 :
  (3⁻¹ + 3⁻² + 3⁻³ + 3⁻⁴ + 3⁻⁵ + 3⁻⁶ : ℤ) % 17 = 7 :=
by {
  sorry
}

end sum_of_inverses_mod_17_l738_738228


namespace kite_flying_weather_conditions_l738_738707

-- Definitions for the problem
variable (T : ℝ) (gusts : ℝ) (contest_held : Prop)

-- Conditions
def condition1 := (T >= 75) ∧ (gusts > 20)
def condition2 := ¬ contest_held

-- The statement to be proved
theorem kite_flying_weather_conditions :
  (condition1 → contest_held) → 
  condition2 → 
  (T < 75 ∨ gusts <= 20) := 
by
  intros h1 h2
  sorry

end kite_flying_weather_conditions_l738_738707


namespace hyperbola_vertex_distance_l738_738738

theorem hyperbola_vertex_distance :
  (∃ a : ℝ, a = 4 * real.sqrt 3 ∧ 2 * a = 8 * real.sqrt 3) ∧ (∀ x y : ℝ, (y^2 / 48 - x^2 / 16 = 1) → 2 * a = 8 * real.sqrt 3) :=
begin
  sorry
end

end hyperbola_vertex_distance_l738_738738


namespace log_inequality_l738_738631

theorem log_inequality : 
  log 2016 2018 > (∑ k in finset.range 2017, log 2016 (k + 1)) / 2017 :=
sorry

end log_inequality_l738_738631


namespace megs_cat_weight_l738_738581

/-- The ratio of the weight of Meg's cat to Anne's cat is 5:7 and Anne's cat weighs 8 kg more than Meg's cat. Prove that the weight of Meg's cat is 20 kg. -/
theorem megs_cat_weight
  (M A : ℝ)
  (h1 : M / A = 5 / 7)
  (h2 : A = M + 8) :
  M = 20 :=
sorry

end megs_cat_weight_l738_738581


namespace flowchart_correct_option_l738_738616

-- Definitions based on conditions
def typical_flowchart (start_points end_points : ℕ) : Prop :=
  start_points = 1 ∧ end_points ≥ 1

-- Theorem to prove
theorem flowchart_correct_option :
  ∃ (start_points end_points : ℕ), typical_flowchart start_points end_points ∧ "Option C" = "Option C" :=
by {
  sorry -- This part skips the proof itself,
}

end flowchart_correct_option_l738_738616


namespace angle_A_in_triangle_l738_738821

theorem angle_A_in_triangle :
  ∀ (A B C : ℝ) (a b c : ℝ),
  a = 2 * Real.sqrt 3 → b = 2 * Real.sqrt 2 → B = π / 4 → 
  (A = π / 3 ∨ A = 2 * π / 3) :=
by
  intros A B C a b c ha hb hB
  sorry

end angle_A_in_triangle_l738_738821


namespace guzman_boxes_l738_738512

noncomputable def total_doughnuts : Nat := 48
noncomputable def doughnuts_per_box : Nat := 12

theorem guzman_boxes :
  ∃ (N : Nat), N = total_doughnuts / doughnuts_per_box ∧ N = 4 :=
by
  use 4
  sorry

end guzman_boxes_l738_738512


namespace part_a_part_b_l738_738001

def A_n (n : ℕ) : ℝ := sqrt (2 - sqrt (2 + sqrt (2 + ... + sqrt(2) ... ))) -- n many radicals

theorem part_a (n : ℕ) (h : n ≥ 2) : A_n n = 2 * sin (π / 2^(n+1)) := by
  sorry

theorem part_b : tendsto (λ n, 2^n * A_n n) at_top (𝓝 π) := by
  sorry

end part_a_part_b_l738_738001


namespace symmetric_line_equation_l738_738902

theorem symmetric_line_equation (x y : ℝ) :
  (1, y = 2*1 + 1) ∧ (x, y) = (1, 1) → (y = 2*x - 3) :=
by
  sorry

end symmetric_line_equation_l738_738902


namespace profit_percentage_on_rest_of_sugar_l738_738311

variable (C : ℝ) -- Cost price per kg
variables (Q₁ Q₂ : ℝ) -- Quantities sold at different profit rates
variables (P : ℝ) -- Unknown profit percentage
variable (T : ℝ := 1600) -- Total quantity of sugar
variable (TSP : ℝ) -- Total selling price

-- Define the conditions
def total_quantity : Prop := Q₁ + Q₂ = T
def specific_quantity : Prop := Q₂ = 1200
def Q₁_val : Prop := Q₁ = 400
def selling_price_1 : ℝ := 432 * C
def selling_price_2 : ℝ := Q₂ * C + (P / 100) * Q₂ * C
def total_selling_price : ℝ := selling_price_1 + selling_price_2
def overall_selling_price_condition : Prop := total_selling_price = 1776 * C

-- The theorem we need to prove
theorem profit_percentage_on_rest_of_sugar :
  (total_quantity)
  ∧ (specific_quantity)
  ∧ (Q₁_val)
  ∧ (overall_selling_price_condition) →
  P = 12 := sorry

end profit_percentage_on_rest_of_sugar_l738_738311


namespace point_d_is_in_fourth_quadrant_l738_738106

-- Conditions
def pointA := (1, 2)
def pointB := (-3, 8)
def pointC := (-3, -5)
def pointD := (6, -7)

def inFourthQuadrant (p : ℤ × ℤ) := p.fst > 0 ∧ p.snd < 0

-- Theorem stating the solution
theorem point_d_is_in_fourth_quadrant : inFourthQuadrant pointD :=
by
  -- Sorry is used to skip the actual proof
  sorry

end point_d_is_in_fourth_quadrant_l738_738106


namespace train_distance_l738_738988

noncomputable def distance_between_stations : ℝ :=
  let d := distance_between_stations in
  let t1_fast := (d / 2) + 10 in  -- Distance fast train travels when they meet first
  let t2_fast := (3 * (d / 2)) + 30 in  -- Distance fast train travels when they meet second
  let t2_total := 2 * d - 40 in  -- Total distance fast train travels when they meet second
  d

theorem train_distance : distance_between_stations = 140 := 
  by
  let d := distance_between_stations in
  have h1 : (d / 2) + 10 + (d / 2) - 10 = d := by sorry  -- They meet first time 10 km west of midpoint
  have h2 : (3 * (d / 2)) + 30 = (2 * d) - 40 := by sorry  -- Distances when they meet second time
  have h3 : (3 / 2) * d + 30 = 2 * d - 40 := by sorry  -- Simplified equation for second meeting
  have h4 : (3 / 2) * d - d = -40 - 30 := by sorry  -- Combining terms
  have h5 : (1 / 2) * d = 70 := by sorry  -- Solving for d
  show d = 140 from by sorry  -- Final solution shows distance is 140

end train_distance_l738_738988


namespace positive_value_of_n_l738_738754

theorem positive_value_of_n (n : ℝ) : (|5 + 5 * Real.sqrt 5 * Complex.I| = 5 * Real.sqrt 6) :=
by
  sorry

end positive_value_of_n_l738_738754


namespace problem_statement_l738_738849

theorem problem_statement (n : ℕ) (hn : n ≥ 2) (x : Fin n → ℝ) (h_nonzero : ¬(∀ i, x i = 0))
  (Cn : ℝ) (hCn_pos : 0 < Cn) (h_sum : (Finset.sum (Finset.finRange n) (λ i, x i) = 0))
  (h_cond : ∀ i : Fin n, x i ≤ x (Fin.mod (i + 1) n) ∨ x i ≤ x (Fin.mod (i + 1) n) + Cn * x (Fin.mod (i + 2) n)) :
  (Cn ≥ 2) ∧ ((Cn = 2) ↔ (2 ∣ n)) := 
sorry

end problem_statement_l738_738849


namespace domain_of_sqrt_exponential_l738_738187

theorem domain_of_sqrt_exponential {x : ℝ} :
  (∃ y : ℝ, y = sqrt (2^x - 1)) ↔ x ≥ 0 :=
by sorry

end domain_of_sqrt_exponential_l738_738187


namespace binomial_expansion_problem_l738_738051

theorem binomial_expansion_problem
  (n : ℕ)
  (x : ℝ)
  (h_ratio : (nat.choose n 2) / (nat.choose n 1) = 5 / 2) :
  (n = 6) ∧ 
  (let r := 3 in nat.choose 6 r * 2^(6 - r) * 3^r = 4320) ∧ 
  (let r := 4 in nat.choose 6 r * 2^(6 - r) * 3^r = 4860 ∧ x^(6 - (4 / 3) * r) = x^(2 / 3)) :=
by
  sorry

end binomial_expansion_problem_l738_738051


namespace heartsuit_calculation_l738_738370

def heartsuit (x y : ℝ) : ℝ := x - Real.sqrt (1 / y)

theorem heartsuit_calculation : heartsuit 3 (heartsuit 3 3) = (27 - Real.sqrt (3 * (9 + Real.sqrt 3))) / 9 := 
by 
  sorry

end heartsuit_calculation_l738_738370


namespace angle_ABC_in_regular_octagon_rectangle_is_67_5_deg_l738_738904

-- Define the problem. Let's name the points involved, and bedsides some basics about regular octagons and rectangles.
variable (A B C D : Type) [decidable_eq A] [decidable_eq B] [decidable_eq C] [decidable_eq D]

def is_regular_octagon (O : fin 8 → A) := 
  -- Define what it means for O to be a regular octagon
  true -- simplified for demonstration

def is_rectangle (R : fin 4 → B) :=
  -- Define what it means for R to be a rectangle
  true -- simplified for demonstration

def shares_common_side (O : fin 8 → A) (R : fin 4 → B) (i : fin 8) (j : fin 4) :=
  -- Define what it means for O and R to share a common side
  true -- simplified for demonstration

variables (O : fin 8 → A) (R : fin 4 → B)
variables (S_T : fin 3 → C) -- For the triangle S, T, which are part of the proof explanation in steps

def m angle_ab (a b c : Type) := Type -- simplified since no type inference is used to define angle

def angle_in_degree (angle:D) := ℝ
  -- Degrees version of the angle

theorem angle_ABC_in_regular_octagon_rectangle_is_67_5_deg : 
  is_regular_octagon O → 
  is_rectangle R → 
  shares_common_side O R 0 0 → 
  angle_in_degree (m angle_ab D A B) = 67.5 :=
by
  sorry

end angle_ABC_in_regular_octagon_rectangle_is_67_5_deg_l738_738904


namespace no_integer_solution_exists_l738_738344

theorem no_integer_solution_exists :
  ¬ ∃ m n : ℤ, m^3 = 3 * n^2 + 3 * n + 7 := by
  sorry

end no_integer_solution_exists_l738_738344


namespace time_to_cross_l738_738603

-- Define the lengths of the trains in meters
def length_train1 : ℕ := 140
def length_train2 : ℕ := 160

-- Define the speeds of the trains in kmph
def speed_train1_kmph : ℕ := 60
def speed_train2_kmph : ℕ := 40

-- Convert speeds from kmph to meters per second
def speed_train1_mps : ℝ := (speed_train1_kmph * 1000) / 3600
def speed_train2_mps : ℝ := (speed_train2_kmph * 1000) / 3600

-- Calculate the relative speed in meters per second
def relative_speed_mps : ℝ := speed_train1_mps + speed_train2_mps

-- Calculate the total distance to be covered in meters
def total_distance : ℕ := length_train1 + length_train2

-- Define the theorem to be proved: time to cross each other is 10.8 seconds
theorem time_to_cross : 
  (total_distance / relative_speed_mps) = (2700 / 250) := 
sorry

end time_to_cross_l738_738603


namespace sequence_term_expression_l738_738047

theorem sequence_term_expression (a : ℕ → ℝ) (S : ℕ → ℝ) (C : ℝ)
  (h1 : a 1 = 1)
  (h2 : ∀ n, S n + n * a n = C)
  (h3 : ∀ n ≥ 2, (n + 1) * a n = (n - 1) * a (n - 1)) :
  ∀ n, a n = 2 / (n * (n + 1)) :=
by
  sorry

end sequence_term_expression_l738_738047


namespace four_times_sum_of_cubes_gt_cube_sum_l738_738799

theorem four_times_sum_of_cubes_gt_cube_sum
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  4 * (a^3 + b^3) > (a + b)^3 :=
by
  sorry

end four_times_sum_of_cubes_gt_cube_sum_l738_738799


namespace prop1_prop2_prop3_l738_738428

variables (a b c d : ℝ)

-- Proposition 1: ab > 0 ∧ bc - ad > 0 → (c/a - d/b > 0)
theorem prop1 (h1 : a * b > 0) (h2 : b * c - a * d > 0) : c / a - d / b > 0 :=
sorry

-- Proposition 2: ab > 0 ∧ (c/a - d/b > 0) → bc - ad > 0
theorem prop2 (h1 : a * b > 0) (h2 : c / a - d / b > 0) : b * c - a * d > 0 :=
sorry

-- Proposition 3: (bc - ad > 0) ∧ (c/a - d/b > 0) → ab > 0
theorem prop3 (h1 : b * c - a * d > 0) (h2 : c / a - d / b > 0) : a * b > 0 :=
sorry

end prop1_prop2_prop3_l738_738428


namespace max_quotient_l738_738435

theorem max_quotient (a b : ℝ) (h1 : 300 ≤ a) (h2 : a ≤ 500) (h3 : 900 ≤ b) (h4 : b ≤ 1800) :
  ∃ (q : ℝ), q = 5 / 9 ∧ (∀ (x y : ℝ), (300 ≤ x ∧ x ≤ 500) ∧ (900 ≤ y ∧ y ≤ 1800) → (x / y ≤ q)) :=
by
  use 5 / 9
  sorry

end max_quotient_l738_738435


namespace angle_B_is_pi_div_3_sin_C_value_l738_738449

-- Definitions and conditions
variable (A B C a b c : ℝ)
variable (cos_cos_eq : (2 * a - c) * Real.cos B = b * Real.cos C)
variable (triangle_ineq : 0 < A ∧ A < Real.pi)
variable (sin_positive : Real.sin A > 0)
variable (a_eq_2 : a = 2)
variable (c_eq_3 : c = 3)

-- Proving B = π / 3 under given conditions
theorem angle_B_is_pi_div_3 : B = Real.pi / 3 := sorry

-- Proving sin C under given additional conditions
theorem sin_C_value : Real.sin C = 3 * Real.sqrt 14 / 14 := sorry

end angle_B_is_pi_div_3_sin_C_value_l738_738449


namespace division_of_decimals_l738_738220

theorem division_of_decimals : 0.25 / 0.005 = 50 := 
by
  sorry

end division_of_decimals_l738_738220


namespace largest_of_choices_l738_738250

theorem largest_of_choices :
  let A := 24680 + (1 / 13579)
  let B := 24680 - (1 / 13579)
  let C := 24680 * (1 / 13579)
  let D := 24680 / (1 / 13579)
  let E := 24680.13579
  A < D ∧ B < D ∧ C < D ∧ E < D :=
by
  let A := 24680 + (1 / 13579)
  let B := 24680 - (1 / 13579)
  let C := 24680 * (1 / 13579)
  let D := 24680 / (1 / 13579)
  let E := 24680.13579
  sorry

end largest_of_choices_l738_738250


namespace sum_of_inverses_mod_17_l738_738227

theorem sum_of_inverses_mod_17 :
  (3⁻¹ + 3⁻² + 3⁻³ + 3⁻⁴ + 3⁻⁵ + 3⁻⁶ : ℤ) % 17 = 7 :=
by {
  sorry
}

end sum_of_inverses_mod_17_l738_738227


namespace frank_total_points_l738_738963

def points_defeating_enemies (enemies : ℕ) (points_per_enemy : ℕ) : ℕ :=
  enemies * points_per_enemy

def total_points (points_from_enemies : ℕ) (completion_points : ℕ) : ℕ :=
  points_from_enemies + completion_points

theorem frank_total_points :
  let enemies := 6
  let points_per_enemy := 9
  let completion_points := 8
  let points_from_enemies := points_defeating_enemies enemies points_per_enemy
  total_points points_from_enemies completion_points = 62 :=
by
  let enemies := 6
  let points_per_enemy := 9
  let completion_points := 8
  let points_from_enemies := points_defeating_enemies enemies points_per_enemy
  -- Placeholder for proof
  sorry

end frank_total_points_l738_738963


namespace car_speeds_l738_738598

theorem car_speeds (slower_faster_diff : ℕ) (total_distance : ℕ) (time : ℕ)
  (h1 : slower_faster_diff = 10)
  (h2 : total_distance = 500)
  (h3 : time = 5) :
  ∃ x y : ℕ, y = x + 10 ∧ 
             5 * x + 5 * y = 500 ∧ 
             x = 45 ∧ 
             y = 55 := 
by {
  use 45,
  use 55,
  sorry
}

end car_speeds_l738_738598


namespace is_isosceles_triangle_l738_738818

theorem is_isosceles_triangle
  {A B C : ℝ} {a b c : ℝ} {triangle_ABC : Triangle ℝ} 
  (h₁ : Triangle.interiorAngles triangle_ABC A B C) 
  (h₂ : 2 * (Real.sin B) * (Real.cos C) = Real.sin A) 
  (h₃ : Triangle.sineLaw a b c A B C)
  (h₄ : Triangle.cosineLaw a b c A B C) :
  is_isosceles b c :=
sorry

end is_isosceles_triangle_l738_738818


namespace problem_solution_l738_738448

variable (x y z : Set ℤ)
variable (card_x card_y card_z : ℕ)
variable (xy_common yz_common xz_common : ℕ)

-- Conditions
def cond1 : card_x = 8 := by sorry
def cond2 : card_y = 18 := by sorry
def cond3 : card_z = 12 := by sorry
def cond4 : xy_common = 6 := by sorry
def cond5 : yz_common = 4 := by sorry
def cond6 : xz_common = 3 := by sorry
def cond7 : ∀ i, (i ∈ x ∧ i ∈ y ∧ i ∈ z) → False := by sorry

-- Definition of the symmetric difference (x # y)
def symmetric_difference (a b : Set ℤ) : Set ℤ := { i | (i ∈ a ∧ i ∉ b) ∨ (i ∈ b ∧ i ∉ a) }

-- Theorem
theorem problem_solution : (symmetric_difference (symmetric_difference x y) z).card = 19 :=
  by
  have cond1 : card_x = 8 := sorry
  have cond2 : card_y = 18 := sorry
  have cond3 : card_z = 12 := sorry
  have cond4 : xy_common = 6 := sorry
  have cond5 : yz_common = 4 := sorry
  have cond6 : xz_common = 3 := sorry
  have cond7 : ∀ i, (i ∈ x ∧ i ∈ y ∧ i ∈ z) → False := sorry
  sorry

end problem_solution_l738_738448


namespace max_f_at_x0_l738_738128

noncomputable def f (x : ℝ) := Real.sqrt (x * (60 - x)) + Real.sqrt (x * (5 - x))

theorem max_f_at_x0 : ∃ x0 M, (0 ≤ x0 ∧ x0 ≤ 5) ∧ M = Real.sqrt 275 ∧ (∀ x ∈ set.Icc (0 : ℝ) 5, f x ≤ M) :=
by
  use 5, Real.sqrt 275
  split
  exact And.intro (by linarith) (by linarith)
  split
  rfl
  intro x hx
  sorry

end max_f_at_x0_l738_738128


namespace geometric_common_ratio_l738_738782

noncomputable def geometric_seq (a : ℕ → ℝ) (q : ℝ) : Prop := ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_common_ratio (a : ℕ → ℝ) (q : ℝ) (h1 : q > 0) 
  (h2 : geometric_seq a q) (h3 : a 3 * a 7 = 4 * (a 4)^2) : q = 2 := 
by 
  sorry

end geometric_common_ratio_l738_738782


namespace simplify_radical_product_l738_738710

theorem simplify_radical_product (q : ℝ) (hq : q ≥ 0) :
  (sqrt (50 * q)) * (sqrt (10 * q)) * (sqrt (15 * q)) = 50 * q * sqrt q :=
by
  sorry

end simplify_radical_product_l738_738710


namespace toby_change_l738_738937

theorem toby_change :
  let cheeseburger_cost := 3.65
  let milkshake_cost := 2
  let coke_cost := 1
  let large_fries_cost := 4
  let cookie_cost := 0.5
  let num_cookies := 3
  let tax := 0.2
  let initial_amount := 15
  let total_cost := 2 * cheeseburger_cost + milkshake_cost + coke_cost + large_fries_cost + num_cookies * cookie_cost + tax
  let cost_per_person := total_cost / 2
  let toby_change := initial_amount - cost_per_person
  in toby_change = 7 :=
sorry

end toby_change_l738_738937


namespace xyz_value_l738_738198

noncomputable def x : ℝ := real.sqrt (12 - 3 * real.sqrt 7) - real.sqrt (12 + 3 * real.sqrt 7)
noncomputable def y : ℝ := real.sqrt (7 - 4 * real.sqrt 3) - real.sqrt (7 + 4 * real.sqrt 3)
noncomputable def z : ℝ := real.sqrt (2 + real.sqrt 3) - real.sqrt (2 - real.sqrt 3)

theorem xyz_value : x * y * z = 12 := by
  sorry

end xyz_value_l738_738198


namespace car_rental_cost_l738_738979

def daily_rental_rate : ℝ := 29
def per_mile_charge : ℝ := 0.08
def rental_duration : ℕ := 1
def distance_driven : ℝ := 214.0

theorem car_rental_cost : 
  (daily_rental_rate * rental_duration + per_mile_charge * distance_driven) = 46.12 := 
by 
  sorry

end car_rental_cost_l738_738979


namespace muthu_investment_time_l738_738885

-- Define the conditions given in the problem
variables (x m : ℝ) -- amount invested by Raman and the number of months after which Muthu invests
constant annual_gain : ℝ := 36000
constant lakshmi_share : ℝ := 12000

-- Define the proportional shares
def raman_share := x * 12
def lakshmi_share_proportional := 2 * x * 6
def muthu_share_proportional := 3 * x * (12 - m)

-- Define the total gain
def total_share := raman_share + lakshmi_share_proportional + muthu_share_proportional

-- Lean statement for the proof problem
theorem muthu_investment_time : 
  lakshmi_share / annual_gain = lakshmi_share_proportional / total_share -> m = 8 :=
by
  -- The proof steps would go here
  sorry

end muthu_investment_time_l738_738885


namespace parallelogram_side_length_l738_738681

theorem parallelogram_side_length 
  (s : ℝ) -- s is a real number
  (area : ℝ) 
  (h : ℝ)
  (H_side1 : 3 * s) -- Length of the first side
  (H_side2 : s) -- Length of the second side
  (H_angle : 30) -- Angle between the sides in degrees
  (H_area : s * s * (√3) = 9 * √3) -- Area condition using heights derived from 30-60-90 triangle properties
  : s = 3 := 
by sorry -- Proof that s = 3 with the given conditions (to be completed)

end parallelogram_side_length_l738_738681


namespace sequence_sum_l738_738169

theorem sequence_sum {n : ℕ} (h : n ≥ 2) :
  (∑ i in Finset.range (n - 1), 1 / (i + 1) / (i + 2)) = 1 - 1 / n :=
sorry

end sequence_sum_l738_738169


namespace Sn_minimum_value_l738_738149

theorem Sn_minimum_value {a : ℕ → ℤ} (n : ℕ) (S : ℕ → ℤ)
  (h1 : a 1 = -11)
  (h2 : a 4 + a 6 = -6)
  (S_def : ∀ n, S n = n * (-12 + n)) :
  ∃ n, S n = S 6 :=
sorry

end Sn_minimum_value_l738_738149


namespace cos_2alpha_plus_pi_six_result_l738_738817

noncomputable def cos_2alpha_plus_pi_six : Real :=
  let α := Real.arccos (-4/5) in
  Real.cos (2 * α + Real.pi / 6)

theorem cos_2alpha_plus_pi_six_result :
  cos_2alpha_plus_pi_six = (7 * Real.sqrt 3 + 24) / 50 := by
  sorry

end cos_2alpha_plus_pi_six_result_l738_738817


namespace constant_term_expansion_l738_738035

theorem constant_term_expansion :
  let n := ∫ x in 0..(Real.pi / 2), 4 * Real.sin x
  let expansion (x : ℝ) : ℝ := (x - (2 / x))^n
  \sum i in Finset.range (n + 1), binomial n i * (-2) ^ i * x ^ (n - 2 * i) == 24 := by
sorry

end constant_term_expansion_l738_738035


namespace general_term_and_sum_l738_738412

noncomputable def arithmeticSequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem general_term_and_sum (a₁ d : ℕ) (S₅ a₃ a₇ : ℕ) 
  (h₁ : S₅ = 20)
  (h₂ : a₁ * a₇ = a₃ * a₃)
  (h₃ : a₁ + a₃ + a₃ + a₃ + a₃ = S₅)
  (h₄ : a₁ + 2 * d = a₃)
  (h₅ : a₃ + 4 * d = a₇):
  (∀ n, arithmeticSequence a₁ d n = n + 1) ∧
  (∀ n, (let b := λ (n : ℕ), $ (n * (n + 1) / 2)$ in
           let T := λ (n : ℕ), 
             2 * (finset.range n).sum (λ i, (1 / b (i + 1))) in
           T n = 2 * (1 - 1 / (n + 1))) ∧
  (∀ n, T n = 2 * (1 - 1 / (n + 1)) = (2 * n) / (n + 1)) :=
sorry

end general_term_and_sum_l738_738412


namespace odd_function_value_l738_738093

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then x^2 + x else -(x^2 + x)

theorem odd_function_value : f (-3) = -12 :=
by
  -- proof goes here
  sorry

end odd_function_value_l738_738093


namespace accurate_river_length_l738_738652

-- Define the given conditions
def length_GSA := 402
def length_AWRA := 403
def error_margin := 0.5
def probability_of_error := 0.04

-- State the theorem based on these conditions
theorem accurate_river_length : 
  ∀ Length_GSA Length_AWRA error_margin probability_of_error, 
  Length_GSA = 402 → 
  Length_AWRA = 403 → 
  error_margin = 0.5 → 
  probability_of_error = 0.04 → 
  (this based on independent measurements with above error margins)
  combined_length = 402.5 ∧ combined_probability_of_error = 0.04 :=
by 
  -- Proof to be completed
  sorry

end accurate_river_length_l738_738652


namespace find_perimeter_l738_738360

noncomputable def problem_statement : ℝ :=
  let length_rectangle := 12 -- cm
  let width_rectangle := 6.5 -- cm
  let radius_semicircle := 6.5 -- cm
  let π := Real.pi
  let perimeter_rectangle := 2 * (length_rectangle + width_rectangle)
  let perimeter_semicircle := (π * radius_semicircle) + (2 * radius_semicircle)
  perimeter_rectangle + perimeter_semicircle

theorem find_perimeter (h1 : problem_statement ≈ 70.42) : 
  problem_statement ≈ 70.42 :=
by
  sorry

end find_perimeter_l738_738360


namespace total_cost_eq_57_l738_738200

namespace CandyCost

-- Conditions
def cost_of_caramel : ℕ := 3
def cost_of_candy_bar : ℕ := 2 * cost_of_caramel
def cost_of_cotton_candy : ℕ := (4 * cost_of_candy_bar) / 2

-- Define the total cost calculation
def total_cost : ℕ :=
  (6 * cost_of_candy_bar) + (3 * cost_of_caramel) + cost_of_cotton_candy

-- Theorem we want to prove
theorem total_cost_eq_57 : total_cost = 57 :=
by
  sorry  -- Proof to be provided

end CandyCost

end total_cost_eq_57_l738_738200


namespace value_diff_l738_738003

def greatest_even_le (z : ℝ) : ℕ :=
  if z < 0 then 0 else nat.floor (z / 2) * 2

theorem value_diff (z : ℝ) (h : z = 6.30) : 
  z - greatest_even_le z = 0.30 :=
by
  simp [h, greatest_even_le]
  -- proof would go here.
  sorry

end value_diff_l738_738003


namespace calculate_expression_l738_738329

theorem calculate_expression :
  sqrt 12 - (-1)^0 + abs (sqrt 3 - 1) = 3 * sqrt 3 - 2 := 
by
  sorry

end calculate_expression_l738_738329


namespace area_of_triangle_AOB_l738_738038

-- Define the polar coordinates for points A and B
def pointA : ℝ × ℝ := (2, Real.pi / 6)
def pointB : ℝ × ℝ := (4, 2 * Real.pi / 3)

-- Define the distances OA and OB
def OA : ℝ := pointA.1
def OB : ℝ := pointB.1

-- Define the angle AOB
def angleAOB : ℝ := pointB.2 - pointA.2

-- Prove that the area of the triangle AOB is 4
theorem area_of_triangle_AOB :
  (OA = 2 ∧ OB = 4 ∧ angleAOB = Real.pi / 2) → 1/2 * OA * OB = 4 :=
by
  sorry

end area_of_triangle_AOB_l738_738038


namespace equilateral_triangle_perimeter_twice_side_area_l738_738559

noncomputable def triangle_side_length (s : ℝ) :=
  s * s * Real.sqrt 3 / 4 = 2 * s

noncomputable def triangle_perimeter (s : ℝ) := 3 * s

theorem equilateral_triangle_perimeter_twice_side_area (s : ℝ) (h : triangle_side_length s) : 
  triangle_perimeter s = 8 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_perimeter_twice_side_area_l738_738559


namespace monochromatic_triangle_in_K17_l738_738345

theorem monochromatic_triangle_in_K17 :
  ∀ (V : Type) (E : V → V → ℕ), (∀ v1 v2, 0 ≤ E v1 v2 ∧ E v1 v2 < 3) →
    (∃ (v1 v2 v3 : V), v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 ∧ (E v1 v2 = E v2 v3 ∧ E v2 v3 = E v1 v3)) :=
by
  intro V E Hcl
  sorry

end monochromatic_triangle_in_K17_l738_738345


namespace find_average_daily_production_l738_738620

def january_production : ℕ := 4000
def increase_per_month : ℕ := 100
def days_in_year : ℕ := 365

noncomputable def monthly_production (n : ℕ) : ℕ := 
if n = 0 then january_production else january_production + n * increase_per_month

noncomputable def total_production : ℕ := 
finset.sum (finset.range 12) monthly_production

noncomputable def average_daily_production : ℚ := total_production / days_in_year

theorem find_average_daily_production : average_daily_production = 152.05 := 
by
  sorry

end find_average_daily_production_l738_738620


namespace sqrt_x_minus_3_defined_iff_l738_738213

theorem sqrt_x_minus_3_defined_iff (x : ℝ) : (∃ y : ℝ, y = real.sqrt (x - 3)) ↔ x ≥ 3 :=
by
  sorry

end sqrt_x_minus_3_defined_iff_l738_738213


namespace hexagon_inscribed_cos_angle_product_l738_738261

theorem hexagon_inscribed_cos_angle_product
  (A B C D E F : Point)
  (circle : Circle)
  (hexagon_inscribed : InscribedHexagon circle A B C D E F)
  (side_length : ∀ {X Y : Point}, {h : IsConsecutiveSide hexagon_inscribed X Y} → dist X Y = 3)
  (AD_eq_2 : dist A D = 2) :
  (1 - cos (angle B A D)) * (1 - cos (angle A D F)) = 2 / 9 := by
  sorry

end hexagon_inscribed_cos_angle_product_l738_738261


namespace minimum_handshakes_l738_738272

-- Definitions
def people : ℕ := 30
def handshakes_per_person : ℕ := 3

-- Theorem statement
theorem minimum_handshakes : (people * handshakes_per_person) / 2 = 45 :=
by
  sorry

end minimum_handshakes_l738_738272


namespace number_of_B_is_14_l738_738455

-- Define the problem conditions
variable (num_students : ℕ)
variable (num_A num_B num_C num_D : ℕ)
variable (h1 : num_A = 8 * num_B / 10)
variable (h2 : num_C = 13 * num_B / 10)
variable (h3 : num_D = 5 * num_B / 10)
variable (h4 : num_students = 50)
variable (h5 : num_A + num_B + num_C + num_D = num_students)

-- Formalize the statement to be proved
theorem number_of_B_is_14 :
  num_B = 14 := by
  sorry

end number_of_B_is_14_l738_738455


namespace noah_closets_capacity_l738_738154

theorem noah_closets_capacity (ali_capacity : ℕ) (fraction : ℚ) (noah_closets : ℕ) :
  (ali_capacity = 200) →
  (fraction = 1/4) →
  (noah_closets = 2) →
  noah_closets * (fraction * ali_capacity).toNat = 100 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry

end noah_closets_capacity_l738_738154


namespace determine_m_to_satisfy_conditions_l738_738191

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 5) * x^(m - 1)

theorem determine_m_to_satisfy_conditions : 
  ∃ (m : ℝ), (m = 3) ∧ ∀ (x : ℝ), (0 < x → (m^2 - m - 5 > 0) ∧ (m - 1 > 0)) :=
by
  sorry

end determine_m_to_satisfy_conditions_l738_738191


namespace find_value_of_n_l738_738395

def is_prime (p : ℕ) : Prop := Nat.Prime p

theorem find_value_of_n
  (a b c n : ℕ)
  (ha : is_prime a)
  (hb : is_prime b)
  (hc : is_prime c)
  (h1 : 2 * a + 3 * b = c)
  (h2 : 4 * a + c + 1 = 4 * b)
  (h3 : n = a * b * c)
  (h4 : n < 10000) :
  n = 1118 :=
by
  sorry

end find_value_of_n_l738_738395


namespace power_sum_mod_inverse_l738_738234

theorem power_sum_mod_inverse (h : 3^6 ≡ 1 [MOD 17]) : 
  (3^(-1) + 3^(-2) + 3^(-3) + 3^(-4) +  3^(-5) + 3^(-6)) ≡ 1 [MOD 17] := 
by
  sorry

end power_sum_mod_inverse_l738_738234


namespace h_inverse_left_h_inverse_right_l738_738864

noncomputable def f (x : ℝ) : ℝ := 4 * x + 5
noncomputable def g (x : ℝ) : ℝ := x^2 - 2 * x - 1
noncomputable def h (x : ℝ) : ℝ := f (g x)
noncomputable def h_inv (y : ℝ) : ℝ := 1 + (Real.sqrt (3 * y + 12)) / 4 -- Correct answer

-- Theorem statements to prove the inverse relationship
theorem h_inverse_left (x : ℝ) : h (h_inv x) = x :=
by
  sorry -- Proof of the left inverse

theorem h_inverse_right (y : ℝ) : h_inv (h y) = y :=
by
  sorry -- Proof of the right inverse

end h_inverse_left_h_inverse_right_l738_738864


namespace figure_C_is_impossible_l738_738477

-- Define the types and properties of the shapes.
structure Shape :=
(area : ℕ)

-- Defining the shapes
def square_1x1 := Shape.mk 1
def rectangle_1x3 := Shape.mk 3
def rectangle_2x1 := Shape.mk 2
def l_shape := Shape.mk 5

-- List of available shapes
def shapes : List Shape := [square_1x1, square_1x1, rectangle_1x3, rectangle_2x1, l_shape]

-- Define the total area of available shapes
def total_area : ℕ := shapes.foldl (λ a s, a + s.area) 0

-- Prove that the total area is 13
example : total_area = 13 := by
  simp [total_area, shapes]

-- Define the property that no rotation is allowed, only translation
def no_rotation (s : Shape) : Prop := true -- Placeholder for the actual property

-- Define that a figure is impossible to form with given shapes
def impossible_to_form (figure : List Shape) : Prop :=
  ¬ ∃ subsets_of_shapes : List (List Shape), (∀ subset ∈ subsets_of_shapes, no_rotation subset) ∧
  (subsets_of_shapes.foldl (λ a s, a + s.area) 0 = 13) ∧
  (subsets_of_shapes.concat figure = shapes)

-- Define specific figure C (for illustration purposes, assume it's figure C)
def figure_C : List Shape := [] -- Placeholder for the actual shape list if known

-- Formal statement: Prove that figure C cannot be formed
theorem figure_C_is_impossible : impossible_to_form figure_C := 
  sorry

end figure_C_is_impossible_l738_738477


namespace problem_I_problem_II_problem_III_l738_738793

section ProblemI
variables {h : ℝ → ℝ} (a : ℝ)
def h1 (x : ℝ) : ℝ := x^2 - a * x + real.log x

theorem problem_I (H : ∀ x ∈ set.Ioo (1 / 2) 1, 0 < deriv h1 x) : a = 3 :=
sorry
end ProblemI

section ProblemII
variables {f g : ℝ → ℝ} (a : ℝ)
def f2 (x : ℝ) : ℝ := x^2 - a * x
def g2 (x : ℝ) : ℝ := real.log x

theorem problem_II (H : ∀ x : ℝ, x > 0 → f2 x ≥ g2 x) : a ≤ 1 :=
sorry
end ProblemII

section ProblemIII
variables {h : ℝ → ℝ} (a x1 x2 m : ℝ)
def h3 (x : ℝ) : ℝ := x^2 - a * x + real.log x

theorem problem_III (H1 : deriv h3 x1 = 0 ∧ deriv h3 x2 = 0) (H2 : x1 ∈ set.Ioo 0 (1 / 2)) (H3 : h3 x1 - h3 x2 > m) : m ≤ (3 / 4) - real.log 2 :=
sorry
end ProblemIII

end problem_I_problem_II_problem_III_l738_738793


namespace min_value_reciprocal_sum_l738_738779

theorem min_value_reciprocal_sum (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 1) : 
  (∃ c, (∀ x y, x > 0 → y > 0 → x + y = 1 → (1/x + 1/y) ≥ c) ∧ (1/a + 1/b = c)) 
:= 
sorry

end min_value_reciprocal_sum_l738_738779


namespace speeding_tickets_l738_738875

theorem speeding_tickets (p1 p2 : ℝ)
  (h1 : p1 = 16.666666666666664)
  (h2 : p2 = 40) :
  (p1 * (100 - p2) / 100 = 10) :=
by sorry

end speeding_tickets_l738_738875


namespace range_of_m_l738_738068

theorem range_of_m : 
  ∀ (m : ℝ), (∀ x, -3 ≤ x ∧ x ≤ 4 → (1 < x ∧ x < m)) → (1 < m ∧ m ≤ 4) :=
by
  intros m h
  split
  · sorry
  · sorry

end range_of_m_l738_738068


namespace num_subsets_intersecting_B_not_empty_l738_738505

-- Definitions of the sets A and B
def A : Set := {a, b, c}
def B : Set := {b, c}

-- The proof problem statement
theorem num_subsets_intersecting_B_not_empty : 
  (finite {S : Set | S ⊆ A ∧ S ∩ B ≠ ∅}).card = 6 :=
by
  sorry

end num_subsets_intersecting_B_not_empty_l738_738505


namespace coincide_nine_point_circles_l738_738847

open EuclideanGeometry

theorem coincide_nine_point_circles (ABC : Triangle) (N : Point) 
(hN : is_orthocenter N ABC) 
(hM1 : is_midpoint_of AN N ABC.a M1)
(hM2 : is_midpoint_of BN N ABC.b M2)
(hM3 : is_midpoint_of CN N ABC.c M3)
(hLine1 : line_parallel_through_point ABC.bC M1)
(hLine2 : line_parallel_through_point ABC.cA M2)
(hLine3 : line_parallel_through_point ABC.aB M3)
(hAN : is_intersection_point M2 M3 A_N)
(hBN : is_intersection_point M3 M1 B_N)
(hCN : is_intersection_point M1 M2 C_N) :
nine_point_circle ABC = nine_point_circle (Triangle.mk A_N B_N C_N) :=
sorry

end coincide_nine_point_circles_l738_738847


namespace Simson_line_tangent_fixed_circle_l738_738158

theorem Simson_line_tangent_fixed_circle
  (circle : Type) [metric_space circle]
  (P C : circle) (A B : circle)
  (hC : C ∈ circle)
  (hP : P ∈ circle)
  (h_angle_const : ∀ (A B : circle), ∠ A C B = const) :
  ∃ fixed_circle : Type, tangent (Simson_line P ABC) fixed_circle := 
sorry

end Simson_line_tangent_fixed_circle_l738_738158


namespace average_minutes_per_player_l738_738522

theorem average_minutes_per_player
  (pg sg sf pf c : ℕ)
  (total_players : ℕ)
  (hp_pg : pg = 130)
  (hp_sg : sg = 145)
  (hp_sf : sf = 85)
  (hp_pf : pf = 60)
  (hp_c : c = 180)
  (hp_total_players : total_players = 5) :
  (pg + sg + sf + pf + c) / total_players / 60 = 2 :=
by
  sorry

end average_minutes_per_player_l738_738522


namespace combined_river_length_estimate_l738_738633

def river_length_GSA := 402 
def river_error_GSA := 0.5 
def river_prob_error_GSA := 0.04 

def river_length_AWRA := 403 
def river_error_AWRA := 0.5 
def river_prob_error_AWRA := 0.04 

/-- 
Given the measurements from GSA and AWRA, 
the combined estimate of the river's length, Rio-Coralio, is 402.5 km,
and the probability of error for this combined estimate is 0.04.
-/
theorem combined_river_length_estimate :
  ∃ l : ℝ, l = 402.5 ∧ ∀ p : ℝ, (p = 0.04) :=
sorry

end combined_river_length_estimate_l738_738633


namespace range_of_a_l738_738062

def f (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x)) + x + 1

theorem range_of_a (a : ℝ) : f a + f (a + 1) > 2 → -1/2 < a ∧ a < 0 :=
by
  -- Conditions
  have domain : ∀ x, -1 < x ∧ x < 1 := sorry
  have g_def : ∀ x, g x = f x - 1 := sorry
  have g_odd : ∀ x, g (-x) = -g x := sorry
  have g_increasing : ∀ x, x₁ < x₂ → g x₁ < g x₂ ∧ -1 < x₁ ∧ x₂ < 1 := sorry
  have inequality : ∀ a, g a > g (-a - 1) := sorry
  have solution : ∀ a, -1/2 < a ∧ a < 0 := sorry
  sorry

end range_of_a_l738_738062


namespace determine_parallel_planes_l738_738898

-- Definition of planes and lines with parallelism
structure Plane :=
  (points : Set (ℝ × ℝ × ℝ))

structure Line :=
  (point1 point2 : ℝ × ℝ × ℝ)
  (in_plane : Plane)

def parallel_planes (α β : Plane) : Prop :=
  ∀ (l1 : Line) (l2 : Line), l1.in_plane = α → l2.in_plane = β → (l1 = l2)

def parallel_lines (l1 l2 : Line) : Prop :=
  ∀ p1 p2, l1.point1 = p1 → l1.point2 = p2 → l2.point1 = p1 → l2.point2 = p2


theorem determine_parallel_planes (α β γ : Plane)
  (h1 : parallel_planes γ α)
  (h2 : parallel_planes γ β)
  (l1 l2 : Line)
  (l1_in_alpha : l1.in_plane = α)
  (l2_in_alpha : l2.in_plane = α)
  (parallel_l1_l2 : ¬ (l1 = l2) → parallel_lines l1 l2)
  (l1_parallel_beta : ∀ l, l.in_plane = β → parallel_lines l l1)
  (l2_parallel_beta : ∀ l, l.in_plane = β → parallel_lines l l2) :
  parallel_planes α β := 
sorry

end determine_parallel_planes_l738_738898


namespace accurate_river_length_l738_738651

-- Define the given conditions
def length_GSA := 402
def length_AWRA := 403
def error_margin := 0.5
def probability_of_error := 0.04

-- State the theorem based on these conditions
theorem accurate_river_length : 
  ∀ Length_GSA Length_AWRA error_margin probability_of_error, 
  Length_GSA = 402 → 
  Length_AWRA = 403 → 
  error_margin = 0.5 → 
  probability_of_error = 0.04 → 
  (this based on independent measurements with above error margins)
  combined_length = 402.5 ∧ combined_probability_of_error = 0.04 :=
by 
  -- Proof to be completed
  sorry

end accurate_river_length_l738_738651


namespace max_ratio_polar_coordinates_l738_738829
noncomputable def cartesian_to_polar (x y : ℝ) : (ℝ × ℝ) := 
  let ρ := Real.sqrt (x^2 + y^2)
  let θ := Real.arctan2 y x
  (ρ, θ)

theorem max_ratio_polar_coordinates :
  let C1 := λ x y : ℝ, (sqrt 3) * x + y - 4 = 0
  let C2 := λ θ : ℝ, (cos θ, 1 + sin θ)
  let C3 := λ t α : ℝ, t > 0 ∧ 0 < α ∧ α < pi / 2 → (t * cos α, t * sin α)
  ∀ A B : ℝ × ℝ,
  (C1 A.1 A.2) → 
  (C2 B.2 = B) → 
  (C3 1 α = A) → 
  (C3 1 α = B) →
  let ρ1 := 4 / (sqrt 3 * cos α + sin α)
  let ρ2 := 2 * sin α
  let ratio := ρ2 / ρ1
  (2 * α - pi / 6 = pi / 2) → 
  α = pi / 3 ∧ (max ratio = 3 / 4)
by sorry

end max_ratio_polar_coordinates_l738_738829


namespace tet_dist_MN_is_1_l738_738388

-- Define the regular tetrahedron with edge length 2
variable (A B C D : Point) -- Points are geometry-specific and we assume a type Point for conceptual clarity.

-- Define point P on edge AB such that AP < 1
variable (P : Point)
variable (hP : P ∈ segment A B ∧ dist A P < 1)

-- Define distances PM and PN where M is on face ABD and N is on face ABC
variable (M : Point)
variable (N : Point)
variable (hPM : dist P M = 1)
variable (hPN : dist P N = 2/3)

-- Define the dihedral angle α between any two faces (which is arccos(1/3) for a regular tetrahedron)
noncomputable def α := Real.arccos (1/3)

-- Define the length MN using the cosine rule in tetrahedron geometry
noncomputable def MN (PM PN:ℝ) (α:ℝ) : ℝ :=
  Real.sqrt (PM^2 + PN^2 - 2 * PM * PN * Real.cos α)

-- Statement
theorem tet_dist_MN_is_1 :
  MN 1 (2 / 3) (Real.arccos (1 / 3)) = 1 := by
  sorry

end tet_dist_MN_is_1_l738_738388


namespace sum_of_three_squares_l738_738826

theorem sum_of_three_squares (n : ℕ) (h : n = 100) : 
  ∃ (a b c : ℕ), a = 4 ∧ b^2 + c^2 = 84 ∧ a^2 + b^2 + c^2 = 100 ∧ 
  (a ≠ b ∧ b ≠ c ∧ a ≠ c ∨ (b = c ∧ a ≠ b)) ∧
  (4^2 + 7^2 + 6^2 = 100 ∧ 4^2 + 8^2 + 5^2 = 100 ∧ 4^2 + 9^2 + 1^2 = 100) ∧
  (4^2 + 6^2 + 7^2 ≠ 100 ∧ 4^2 + 5^2 + 8^2 ≠ 100 ∧ 4^2 + 1^2 + 9^2 ≠ 100 ∧ 
   4^2 + 4^2 + 8^2 ≠ 100 ∨ 4^2 + 8^2 + 4^2 ≠ 100) :=
sorry

end sum_of_three_squares_l738_738826


namespace sin_sum_to_product_l738_738347

theorem sin_sum_to_product (x : ℝ) : 
  sin (7 * x) + sin (9 * x) = 2 * sin (8 * x) * cos (x) :=
by
  sorry

end sin_sum_to_product_l738_738347


namespace fraction_of_phone_numbers_l738_738705

theorem fraction_of_phone_numbers :
  let total_valid_numbers := 7 * 10^6 in
  let numbers_starting_with_9_and_ending_with_even := 5 * 10^5 in
  (numbers_starting_with_9_and_ending_with_even / total_valid_numbers : ℚ) = 1 / 14 :=
by
  -- to be proven
  sorry

end fraction_of_phone_numbers_l738_738705


namespace perimeter_dodecagon_equals_seventy_two_l738_738486

variables (ABCDEF : Type) [hexagon ABCDEF] (side_len: ℝ)
variables (ABGH BCIJ CDKL DEMN EFOP FAQR : ABCDEF → ABCDEF)
variable [hexagon_side : ∀ (A B C D E F : ABCDEF), distance A B = side_len]
variable [hexagon_squares : ∀ (A B C D E F : ABCDEF), 
                    (is_square ABGH A B) ∧ 
                    (is_square BCIJ B C) ∧ 
                    (is_square CDKL C D) ∧
                    (is_square DEMN D E) ∧
                    (is_square EFOP E F) ∧
                    (is_square FAQR F A)]

noncomputable def dodecagon_perimeter : ℝ := 12 * side_len

theorem perimeter_dodecagon_equals_seventy_two : 
    ∀ (A B C D E F : ABCDEF), hexagon_side A B → 
                               hexagon_squares A B C D E F →
                               side_len = 6 →
                               dodecagon_perimeter side_len = 72 := 
sorry

end perimeter_dodecagon_equals_seventy_two_l738_738486


namespace max_distance_cos_sin_to_line_l738_738570

open Real

noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  abs (A * x₀ + B * y₀ + C) / sqrt (A^2 + B^2)

theorem max_distance_cos_sin_to_line :
  ∃ θ : ℝ, ∃ A B C : ℝ,
  distance_point_to_line (cos θ) (sin θ) A B C = 9 / 5 :=
by
  let A := 3
  let B := 4
  let C := -4
  use θ
  use A
  use B
  use C
  sorry

end max_distance_cos_sin_to_line_l738_738570


namespace vanessa_age_l738_738546

def is_prime (n : ℕ) : Prop := nat.prime n

def exactly_correct_guess (age guesses : list ℕ) (n : ℕ) : Prop :=
  n ∈ guesses ∧ n = age

def off_by_one_guesses (age guesses : list ℕ) : Prop :=
  (age - 1 ∈ guesses) ∨ (age + 1 ∈ guesses)

def at_least_half_too_low (age : ℕ) (guesses : list ℕ) : Prop :=
  (guesses.count (< age)).to_real / guesses.length.to_real ≥ 0.5

def guess_vanessa_age (guesses : list ℕ) : ℕ :=
  53 -- based on the correct answer in the solution

theorem vanessa_age (guesses : list ℕ) (age : ℕ) :
  (at_least_half_too_low age guesses) ∧
  (three_guesses_off_by_one age guesses) ∧
  (is_prime age) ∧
  (exactly_correct_guess age guesses age) →
  age = 53 :=
sorry

end vanessa_age_l738_738546


namespace sum_numerator_divisible_by_1987_l738_738162

theorem sum_numerator_divisible_by_1987 :
    let S := ∑ m in Finset.range 662, 1 / ((3 * m + 1) * (3 * m + 2) * (3 * m + 3)) in
    (∃ n, n * 1987 = (S.num)) :=
by
  -- Definitions
  let S := ∑ m in Finset.range 662, 1 / ((3 * m + 1) * (3 * m + 2) * (3 * m + 3))
  -- Concluding statement
  sorry

end sum_numerator_divisible_by_1987_l738_738162


namespace min_sum_MP_MF_l738_738408

theorem min_sum_MP_MF {a b : ℝ} (hp : a > 0) (hq : b > 0)
  (focus_eq : ∀ {x y : ℝ}, x = 0 ∧ y = 1)
  (ellipse_eq : ∀ {x y : ℝ}, y^2/4 + x^2/(b^2) = 1 → b = sqrt 3)
  (parabola_eq : ∀ {x y : ℝ}, y = x^2/4)
  (M P : ℝ × ℝ) (hx : (P.1, P.2) = (3/2, 1))
: (|MP + |MF|) = 2 :=
sorry

end min_sum_MP_MF_l738_738408


namespace solve_for_a_l738_738426

theorem solve_for_a (x y a : ℝ) (h1 : 2 * x + y = 2 * a + 1) 
                    (h2 : x + 2 * y = a - 1) 
                    (h3 : x - y = 4) : a = 2 :=
by
  sorry

end solve_for_a_l738_738426


namespace number_of_valid_b_l738_738805

theorem number_of_valid_b :
  ∃ (count : ℕ), count = 4 ∧ ∀ (b : ℕ), (0 < b) → (∃ n : ℕ, n > 0 ∧ b^n = 256 → b ∈ {2, 4, 16, 256}) :=
by
  sorry

end number_of_valid_b_l738_738805


namespace vegetarian_gluten_free_fraction_l738_738513

theorem vegetarian_gluten_free_fraction :
  ∀ (total_dishes meatless_dishes gluten_free_meatless_dishes : ℕ),
  meatless_dishes = 4 →
  meatless_dishes = total_dishes / 5 →
  gluten_free_meatless_dishes = meatless_dishes - 3 →
  gluten_free_meatless_dishes / total_dishes = 1 / 20 :=
by sorry

end vegetarian_gluten_free_fraction_l738_738513


namespace area_of_triangle_ABC_l738_738952

noncomputable def triangle_altitude_area (AC BK BC : ℝ) (hAC : AC = 12) (hBK : BK = 9) (hBC : BC = 18) : ℝ :=
  let CK := BC - BK in
  let AK := Real.sqrt (AC^2 - CK^2) in
  (1 / 2) * BC * AK

theorem area_of_triangle_ABC :
  ∀ (AC BK BC : ℝ), AC = 12 → BK = 9 → BC = 18 → triangle_altitude_area AC BK BC = 27 * Real.sqrt 7 :=
by  
  intros AC BK BC hAC hBK hBC
  have h1 : CK = 9 := by rw [←hBC, hBK]; linarith
  have h2 : AK = 3 * Real.sqrt 7 := by 
    rw [Real.sqrt_eq_rpow, hAC, h1]; 
    norm_num1
  
  unfold triangle_altitude_area
  rw [hAC, hBC, h2]
  norm_num
  sorry

end area_of_triangle_ABC_l738_738952


namespace f_even_l738_738912

def f (x : ℝ) : ℝ := real.log (1 + 4^x) / real.log 2 - x

theorem f_even : ∀ x : ℝ, f (-x) = f x :=
by
  intro x
  apply sorry

end f_even_l738_738912


namespace women_in_company_l738_738452

variable (W : ℕ) -- Total number of workers
variable (men : ℕ) -- Number of men in the company
variable (women : ℕ) -- Number of women in the company

-- Conditions
axiom cond1 : W / 3 = (W : ℕ) \div 3 -- A third of the workers do not have a retirement plan
axiom cond2 : 0.2 * (W / 3) = ((1 / 15) : ℝ) * W -- 20% of the workers without a retirement plan are women
axiom cond3 : 0.4 * (2 / 3 * W) = ((4 / 15) : ℝ) * W -- 40% of the workers with a retirement plan are men
axiom cond4 : men = 144 -- There are 144 men in the company

-- Goal to be proved
theorem women_in_company : women = 252 :=
by 
  sorry

end women_in_company_l738_738452


namespace area_of_B_l738_738141

-- Define the set A
def A : Set ℝ := {a | -1 ≤ a ∧ a ≤ 2}

-- Define the set B
def B : Set (ℝ × ℝ) := {p | p.1 ∈ A ∧ p.2 ∈ A ∧ p.1 + p.2 ≥ 0}

-- State the theorem to prove the area of set B is 7
theorem area_of_B : measure_theory.measure_space.volume B = 7 := 
  sorry

end area_of_B_l738_738141


namespace margin_expression_l738_738445

variable (C S M : ℝ) (n : ℝ)
hypothesis h_n_gt_two : n > 2
hypothesis h_M_eq_2_over_n_C : M = (2 / n) * C
hypothesis h_S_minus_M_eq_C : S - M = C

theorem margin_expression (h_M_eq_2_over_n_C : M = (2 / n) * C) (h_S_minus_M_eq_C : S - M = C) (h_n_gt_two : n > 2) : M = 2 * S / (n + 2) := 
by
  sorry

end margin_expression_l738_738445


namespace total_money_l738_738433

theorem total_money 
  (n_pennies n_nickels n_dimes n_quarters n_half_dollars : ℝ) 
  (h_pennies : n_pennies = 9) 
  (h_nickels : n_nickels = 4) 
  (h_dimes : n_dimes = 3) 
  (h_quarters : n_quarters = 7) 
  (h_half_dollars : n_half_dollars = 5) : 
  0.01 * n_pennies + 0.05 * n_nickels + 0.10 * n_dimes + 0.25 * n_quarters + 0.50 * n_half_dollars = 4.84 :=
by 
  sorry

end total_money_l738_738433


namespace equilateral_triangle_perimeter_twice_side_area_l738_738558

noncomputable def triangle_side_length (s : ℝ) :=
  s * s * Real.sqrt 3 / 4 = 2 * s

noncomputable def triangle_perimeter (s : ℝ) := 3 * s

theorem equilateral_triangle_perimeter_twice_side_area (s : ℝ) (h : triangle_side_length s) : 
  triangle_perimeter s = 8 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_perimeter_twice_side_area_l738_738558


namespace missing_number_in_proportion_l738_738833

theorem missing_number_in_proportion (x : ℝ) :
  (2 / x) = ((4 / 3) / (10 / 3)) → x = 5 :=
by sorry

end missing_number_in_proportion_l738_738833


namespace average_player_time_l738_738519

theorem average_player_time:
  let pg := 130
  let sg := 145
  let sf := 85
  let pf := 60
  let c := 180
  let total_secs := pg + sg + sf + pf + c
  let total_mins := total_secs / 60
  let num_players := 5
  let avg_mins_per_player := total_mins / num_players
  avg_mins_per_player = 2 :=
by
  sorry

end average_player_time_l738_738519


namespace AL_LB_ratio_l738_738186

-- Define the circle and its diameters
variable {S : Type} [circle S]
variable {A B C D K L : Point}

-- Given conditions
variable (diamAB : ∀ (point : Point), point ∈ diameter A B ↔ ∃ (x : Real), point = (x, 0))
variable (diamCD : ∀ (point : Point), point ∈ diameter C D ↔ ∃ (y : Real), point = (0, y))
variable (perpendicular : ∀ (point : Point), point ∈ diameter A B ∩ diameter C D → point = (0, 0))
variable (CK_KD : ∀ (division_pt : Point), division_pt = K → ratio (C, K, D) = 2 / 3)

-- Chords intersecting diameters
variable (chordEA : ∀ (intersection_pt : Point), intersection_pt = K ↔ (intersection_pt ∈ chord E A ∧ intersection_pt ∈ diameter C D))
variable (chordEC : ∀ (intersection_pt : Point), intersection_pt = L ↔ (intersection_pt ∈ chord E C ∧ intersection_pt ∈ diameter A B))

-- Proof statement to show the final ratio
theorem AL_LB_ratio (Al_Deq : ∀ (L_ratio : Real), L_ratio = ratio (A, L, B)): Al_Deq = 3 / 4 :=
by sorry

end AL_LB_ratio_l738_738186


namespace quadrilateral_angles_l738_738934

theorem quadrilateral_angles 
(h : ∃ A B C D : Type, (∀ (a b c : A), a ≠ b → a ≠ c → b ≠ c → quadrilateral A B C D ∧ (side_length A B = side_length B C) ∧ (∠ABC = 90) ∧ (∠BCD = 150))) : 
(∠BAD = 75) ∧ (∠ADC = 45) :=
sorry

end quadrilateral_angles_l738_738934


namespace company_storage_payment_l738_738619

theorem company_storage_payment
  (length width height : ℕ)
  (total_volume : ℕ)
  (cost_per_box : ℝ)
  (h_length : length = 15)
  (h_width : width = 12)
  (h_height : height = 10)
  (h_total_volume : total_volume = 1080000)
  (h_cost_per_box : cost_per_box = 0.2) : 
  (total_payment : ℝ) (h_payment_eq : total_payment = (total_volume / (length * width * height)) * cost_per_box) :
  total_payment = 120 := 
by
  sorry

end company_storage_payment_l738_738619


namespace find_k_value_l738_738783

theorem find_k_value (k : ℝ) : 
  let line_eq := λ x : ℝ, k * x + 2
      circle_eq := λ x y : ℝ, (x - 3)^2 + (y - 1)^2 = 9
      chord_length := 4 * real.sqrt 2 in
  (∃ A B : ℝ × ℝ, 
    (circle_eq A.1 A.2) ∧ (circle_eq B.1 B.2) ∧ 
    (line_eq A.1 = A.2) ∧ (line_eq B.1 = B.2) ∧ 
    (real.dist A B = chord_length)) → (k = 0 ∨ k = -3 / 4) :=
sorry

end find_k_value_l738_738783


namespace g_even_problem_l738_738858

def g (x : ℝ) : ℝ := 2 * x^6 + 3 * x^4 - x^2 + 7

theorem g_even (x : ℝ) : g x = g (-x) := by sorry

theorem problem (h : g 5 = 29) : g 5 + g (-5) = 58 := by
  calc
    g 5 + g (-5) = g 5 + g 5 := by rw g_even
    ... = 29 + 29 := by rw h
    ... = 58 := by ring

end g_even_problem_l738_738858


namespace license_plates_count_l738_738078

/--
Define the conditions and constants.
-/
def num_letters := 26
def num_first_digit := 5  -- Odd digits
def num_second_digit := 5 -- Even digits

theorem license_plates_count : num_letters ^ 3 * num_first_digit * num_second_digit = 439400 := by
  sorry

end license_plates_count_l738_738078


namespace sufficient_condition_for_m_perp_beta_l738_738436

-- Define the planes and lines
variables (α β γ : Plane) (m n l : Line)

-- Conditions
axiom diff_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ
axiom diff_lines : m ≠ n ∧ n ≠ l ∧ m ≠ l

-- Specific perpendicularity and parallel conditions
axiom perp_n_alpha : n ⊥ α
axiom perp_n_beta : n ⊥ β
axiom perp_m_alpha : m ⊥ α

-- The required proof to show that m ⊥ β
theorem sufficient_condition_for_m_perp_beta (α β γ : Plane) (m n l : Line) 
  (h1 : α ≠ β) (h2 : β ≠ γ) (h3 : α ≠ γ)
  (h4 : m ≠ n) (h5 : n ≠ l) (h6 : m ≠ l)
  (h7 : n ⊥ α) (h8 : n ⊥ β) (h9 : m ⊥ α) : m ⊥ β :=
sorry


end sufficient_condition_for_m_perp_beta_l738_738436


namespace mean_daily_profit_l738_738571

theorem mean_daily_profit (d : ℕ) (fifteen_days_first : ℕ) (fifteen_days_last : ℕ) (mean_first : ℕ) (mean_second : ℕ) :
  d = 30 → fifteen_days_first = 15 → fifteen_days_last = 15 → mean_first = 225 → mean_second = 475 →
  let total_profit := (mean_first * fifteen_days_first) + (mean_second * fifteen_days_last),
      mean_daily_profit := total_profit / d
  in mean_daily_profit = 350 :=
by
  intros h1 h2 h3 h4 h5
  let total_profit := (mean_first * fifteen_days_first) + (mean_second * fifteen_days_last)
  have : total_profit = (225 * 15) + (475 * 15), by rw [h4, h5]
  have : total_profit = 3375 + 7125, by rw this
  have : total_profit = 10500, by norm_num [this]
  let mean_daily_profit := total_profit / d
  have : d = 30, by rw h1 
  have : mean_daily_profit = 10500 / 30, by norm_num [this, h1]
  have : mean_daily_profit = 350, by norm_num [this]
  assumption
  sorry

end mean_daily_profit_l738_738571


namespace length_of_arc_l738_738184

theorem length_of_arc (C : ℝ) (θ : ℝ) (DE : ℝ) (c_circ : C = 100) (angle : θ = 120) :
  DE = 100 / 3 :=
by
  -- Place the actual proof here.
  sorry

end length_of_arc_l738_738184


namespace flat_fee_for_solar_panel_equipment_l738_738935

theorem flat_fee_for_solar_panel_equipment
  (land_acreage : ℕ)
  (land_cost_per_acre : ℕ)
  (house_cost : ℕ)
  (num_cows : ℕ)
  (cow_cost_per_cow : ℕ)
  (num_chickens : ℕ)
  (chicken_cost_per_chicken : ℕ)
  (installation_hours : ℕ)
  (installation_cost_per_hour : ℕ)
  (total_cost : ℕ)
  (total_spent : ℕ) :
  land_acreage * land_cost_per_acre + house_cost +
  num_cows * cow_cost_per_cow + num_chickens * chicken_cost_per_chicken +
  installation_hours * installation_cost_per_hour = total_spent →
  total_cost = total_spent →
  total_cost - (land_acreage * land_cost_per_acre + house_cost +
  num_cows * cow_cost_per_cow + num_chickens * chicken_cost_per_chicken +
  installation_hours * installation_cost_per_hour) = 26000 := by 
  sorry

end flat_fee_for_solar_panel_equipment_l738_738935


namespace roots_sum_product_l738_738916

theorem roots_sum_product (p q : ℝ) (h_sum : p / 3 = 8) (h_prod : q / 3 = 12) : p + q = 60 := 
by 
  sorry

end roots_sum_product_l738_738916


namespace possible_k_values_l738_738115

def triangle_right_k_values (AB AC : ℝ × ℝ) (k : ℝ) : Prop :=
  let BC := (AC.1 - AB.1, AC.2 - AB.2)
  let angle_A := AB.1 * AC.1 + AB.2 * AC.2 = 0   -- Condition for ∠A = 90°
  let angle_B := AB.1 * BC.1 + AB.2 * BC.2 = 0   -- Condition for ∠B = 90°
  let angle_C := BC.1 * AC.1 + BC.2 * AC.2 = 0   -- Condition for ∠C = 90°
  (angle_A ∨ angle_B ∨ angle_C)

theorem possible_k_values (k : ℝ) :
  triangle_right_k_values (2, 3) (1, k) k ↔
  k = -2/3 ∨ k = 11/3 ∨ k = (3 + Real.sqrt 13) / 2 ∨ k = (3 - Real.sqrt 13) / 2 :=
by
  sorry

end possible_k_values_l738_738115


namespace problem_1_problem_2_l738_738023

variables {A B C : ℝ} {a b c : ℝ}

-- Assuming A = 2C, and ∆ABC is acute
def acute_triangle (A B C : ℝ) : Prop := 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2

theorem problem_1
  (h_acute : acute_triangle A B C)
  (h_A_eq_2C : A = 2 * C) :
  sqrt 2 < a / c ∧ a / c < sqrt 3 :=
sorry

theorem problem_2
  (h_acute : acute_triangle A B C)
  (h_A_eq_2C : A = 2 * C)
  (h_b : b = 1)
  (h_c : c = 3) :
  let S := 1 / 2 * b * a * sin C,
  S = sqrt 2 :=
sorry

end problem_1_problem_2_l738_738023


namespace grasshopper_cannot_move_3_cm_grasshopper_can_move_2_cm_grasshopper_can_move_1234_cm_l738_738300

def can_jump (x : Int) : Prop :=
  ∃ (k m : Int), x = k * 36 + m * 14

theorem grasshopper_cannot_move_3_cm :
  ¬ can_jump 3 :=
by
  sorry

theorem grasshopper_can_move_2_cm :
  can_jump 2 :=
by
  sorry

theorem grasshopper_can_move_1234_cm :
  can_jump 1234 :=
by
  sorry

end grasshopper_cannot_move_3_cm_grasshopper_can_move_2_cm_grasshopper_can_move_1234_cm_l738_738300


namespace perpendicular_planes_l738_738137

variables 
  (Line : Type) 
  (Plane : Type) 
  (m n : Line) 
  (α β : Plane)

-- Define perpendicularity relations
def perp (x y : Type) : Prop := sorry  -- assuming we have a definition for perpendicular

-- Given conditions
variables 
  (h1 : m ≠ n)
  (h2 : α ≠ β)
  (h3 : perp Line m n)
  (h4 : perp Line m α)
  (h5 : perp Line n β)

-- Prove that α is perpendicular to β
theorem perpendicular_planes : perp Plane α β :=
by
  sorry

end perpendicular_planes_l738_738137


namespace isosceles_triangle_semicircle_radius_l738_738334

noncomputable def semicircle_radius {base height : ℝ} (b h : ℝ) : ℝ :=
  if b ≠ 0 ∧ h ≠ 0 then
    let s := 2 * Real.sqrt (h^2 + (b/2)^2) in
    let area := b * h / 2 in
    area / s
  else 0

theorem isosceles_triangle_semicircle_radius :
  semicircle_radius 20 18 = 90 / Real.sqrt 106 :=
by
  sorry

end isosceles_triangle_semicircle_radius_l738_738334


namespace evaluate_expressions_for_pos_x_l738_738736

theorem evaluate_expressions_for_pos_x :
  (∀ x : ℝ, x > 0 → 6^x * x^3 = 6^x * x^3) ∧
  (∀ x : ℝ, x > 0 → (3 * x)^(3 * x) ≠ 6^x * x^3) ∧
  (∀ x : ℝ, x > 0 → 3^x * x^6 ≠ 6^x * x^3) ∧
  (∀ x : ℝ, x > 0 → (6 * x)^x ≠ 6^x * x^3) →
  ∃ n : ℕ, n = 1 := 
by
  sorry

end evaluate_expressions_for_pos_x_l738_738736


namespace distinct_remainders_l738_738850

theorem distinct_remainders (n : ℕ) (hn : 0 < n) : 
  ∀ (i j : ℕ), (i < n) → (j < n) → (2 * i + 1 ≠ 2 * j + 1) → 
  ((2 * i + 1) ^ (2 * i + 1) % 2^n ≠ (2 * j + 1) ^ (2 * j + 1) % 2^n) :=
by
  sorry

end distinct_remainders_l738_738850


namespace complex_division_l738_738401

theorem complex_division (i : ℂ) (hi : i = Complex.I) : (7 - i) / (3 + i) = 2 - i := by
  sorry

end complex_division_l738_738401


namespace carnival_rides_l738_738709

theorem carnival_rides (F : ℕ) (bumper_rides : ℕ) (ticket_per_ride : ℕ) (total_tickets_used : ℕ)
  (h1 : bumper_rides = 3) (h2 : ticket_per_ride = 5) (h3 : total_tickets_used = 50) :
  F = 7 :=
by
  -- Intuition: The remaining tickets after using them for bumper rides should equal the tickets used for ferris wheel.
  have t_bumper := (3 * 5), -- Tickets used for bumper cars
  have t_ferris := (50 - t_bumper), -- Tickets left for ferris wheel
  have number_of_rides := t_ferris / 5, -- Number of ferris wheel rides
  have final_f := (number_of_rides = 7), -- Based on provided conditions
  sorry

end carnival_rides_l738_738709


namespace sum_central_square_l738_738450

noncomputable def table_sum : ℕ := 10200
noncomputable def a : ℕ := 1200
noncomputable def central_sum : ℕ := 720

theorem sum_central_square :
  ∃ (a : ℕ), table_sum = a * (1 + (1 / 3) + (1 / 9) + (1 / 27)) * (1 + (1 / 4) + (1 / 16) + (1 / 64)) ∧ 
              central_sum = (a / 3) + (a / 12) + (a / 9) + (a / 36) :=
by
  sorry

end sum_central_square_l738_738450


namespace cos_double_angle_example_l738_738758

theorem cos_double_angle_example (θ : ℝ) (h : Real.cos θ = 1 / 3) : Real.cos (2 * θ) = -7 / 9 := by
  sorry

end cos_double_angle_example_l738_738758


namespace arithmetic_seq_sum_l738_738768

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h1 : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h2 : a 2 + a 5 + a 8 = 15) : a 3 + a 7 = 10 :=
sorry

end arithmetic_seq_sum_l738_738768


namespace car_b_speed_l738_738330

theorem car_b_speed (Speed_A Time_A Time_B : ℕ) (Distance_A : ℕ)
  (h1 : Speed_A = 80)
  (h2 : Time_A = 5)
  (h3 : Time_B = 2)
  (h4 : Distance_A = Speed_A * Time_A)
  (ratio : ∀ Distance_B, Distance_A / Distance_B = 2 -> Speed_B = Distance_B / Time_B)
  : ∃ Speed_B, Speed_B = 100 :=
by
  use 100
  sorry

end car_b_speed_l738_738330


namespace average_screen_time_per_player_l738_738526

def video_point_guard : ℕ := 130
def video_shooting_guard : ℕ := 145
def video_small_forward : ℕ := 85
def video_power_forward : ℕ := 60
def video_center : ℕ := 180
def total_video_time : ℕ := 
  video_point_guard + video_shooting_guard + video_small_forward + video_power_forward + video_center
def total_video_time_minutes : ℕ := total_video_time / 60
def number_of_players : ℕ := 5

theorem average_screen_time_per_player : total_video_time_minutes / number_of_players = 2 :=
  sorry

end average_screen_time_per_player_l738_738526


namespace toby_change_is_7_l738_738945

def cheeseburger_cost : ℝ := 3.65
def milkshake_cost : ℝ := 2
def coke_cost : ℝ := 1
def large_fries_cost : ℝ := 4
def cookie_cost : ℝ := 0.5
def tax : ℝ := 0.2
def toby_funds : ℝ := 15

def total_food_cost_before_tax : ℝ := 
  2 * cheeseburger_cost + milkshake_cost + coke_cost + large_fries_cost + 3 * cookie_cost

def total_bill_with_tax : ℝ := total_food_cost_before_tax + tax

def each_person_share : ℝ := total_bill_with_tax / 2

def toby_change : ℝ := toby_funds - each_person_share

theorem toby_change_is_7 : toby_change = 7 := by
  sorry

end toby_change_is_7_l738_738945


namespace problem_statement_l738_738468

theorem problem_statement (
  (Q : ℝ × ℝ) (hq : Q = (real.sqrt 3, 4 * real.sqrt 3)) 
  (m : ℝ → ℝ → Prop) (hm : ∀ x y, m x y ↔ x + 2 * y = 0) 
  (n : ℝ → ℝ → Prop) (hn : ∀ x y, n x y ↔ 2 * x - y + 2 * real.sqrt 3 = 0)
  (M N : ℝ × ℝ) (hmn : M = (-real.sqrt 3, 0) ∧ N = (real.sqrt 3, 0)) 
  (P : ℝ × ℝ → Prop) (hp : ∀ P, dist P M + dist P N = 4)
  (D E : ℝ × ℝ) (hde : D = (1, 0) ∧ E = (4, 1))
  (L : ℝ × ℝ → ℝ → Prop) (hl : ∀ x y, L (x, y) k ↔ y = k * (x - 1))
  (k1 k2 : ℝ)) :
  (trajectory : ℝ × ℝ → Prop) (hC : ∀ x y, trajectory (x, y) ↔ x^2 / 4 + y^2 = 1) ∧
  (constant_sum : k1 + k2 = 2 / 3) :=
begin
  sorry
end

end problem_statement_l738_738468


namespace div_identity_l738_738440

theorem div_identity (a b c : ℚ) (h1 : a / b = 3) (h2 : b / c = 2 / 5) : c / a = 5 / 6 := by
  sorry

end div_identity_l738_738440


namespace log_prime_fraction_l738_738816

theorem log_prime_fraction (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h : 3 * p + 5 * q = 31) :
    (Nat.log 2 (p / (3 * q + 1)) = -3) ∨ (Nat.log 2 (p / (3 * q + 1)) = 0) :=
by
  sorry

end log_prime_fraction_l738_738816


namespace find_s_l738_738679

namespace Parallelogram

-- Define s and other parameters
variables (s : ℝ)

-- Definitions of the given conditions
def base := 3 * s
def side := s
def angle := 30
def area := 9 * real.sqrt 3

-- The height relationship using properties of 30-60-90 triangle
def height := side / 2

-- Statement translating the area condition
theorem find_s (h : 1/2 * base * side = area) : s = real.sqrt 6 * real.sqrt (real.sqrt 3) := sorry 

end Parallelogram

end find_s_l738_738679


namespace sasha_can_afford_l738_738247

-- Definitions for constants
def sasha_budget : ℕ := 1800
def shashlik_price_per_kg : ℕ := 350
def tomato_sauce_price_per_can : ℕ := 70
def discount_threshold : ℕ := 1500
def discount_rate : ℕ := 26
def discount_factor : ℤ := 74 -- 100 - discount rate
def planned_shashlik_kg : ℕ := 5
def planned_tomato_sauce_can : ℕ := 1

-- Theorem statement
theorem sasha_can_afford (
  budget : ℕ := sasha_budget,
  shashlik_price : ℕ := shashlik_price_per_kg,
  sauce_price : ℕ := tomato_sauce_price_per_can,
  threshold : ℕ := discount_threshold,
  disc_rate : ℕ := discount_rate,
  disc_factor : ℤ := discount_factor,
  desired_shashlik : ℕ := planned_shashlik_kg,
  desired_sauce : ℕ := planned_tomato_sauce_can
) : (desired_shashlik * shashlik_price + sauce + shashlik_price * desired_shashlik * disc_factor / 100 ≤ budget) :=
sorry

-- Note: Instead of fully implementing the theorem with all the calculated steps,
--       'sorry' is used to indicate the place where the proof would go.

end sasha_can_afford_l738_738247


namespace simplify_fraction_l738_738749

noncomputable def a_n (n : ℕ) : ℝ := ∑ k in Finset.range (n + 1), 1 / (Nat.choose n k)

noncomputable def b_n (n : ℕ) : ℝ := ∑ k in Finset.range (n + 1), (k ^ 2 : ℝ) / (Nat.choose n k)

theorem simplify_fraction (n : ℕ) (h: n > 0) : a_n n / b_n n ≈ 4 / (n ^ 2) := sorry

end simplify_fraction_l738_738749


namespace division_of_decimals_l738_738219

theorem division_of_decimals : 0.25 / 0.005 = 50 := 
by
  sorry

end division_of_decimals_l738_738219


namespace best_and_stable_student_l738_738665

theorem best_and_stable_student :
  ∃ (m n : ℝ), m = 92 ∧ n = 8.5 ∧
  m > 91 ∧ n < 11 :=
by {
  use [92, 8.5],
  simp,
  split,
  { linarith, },
  { linarith, },
  sorry
}

end best_and_stable_student_l738_738665


namespace abs_neg_eight_l738_738554

theorem abs_neg_eight : abs (-8) = 8 := by
  sorry

end abs_neg_eight_l738_738554


namespace domain_of_function_correct_l738_738739

open Real

noncomputable def domain_of_function (x : ℝ) : Prop :=
  log (sin (2 * x)) + sqrt (9 - x^2)

theorem domain_of_function_correct : { x : ℝ | x ∈ [-3, -π / 2) ∪ (0, π / 2] } = { x : ℝ | 0 < sin (2 * x) ∧ -3 ≤ x ∧ x ≤ 3 } :=
by
  sorry

end domain_of_function_correct_l738_738739


namespace second_lowest_exam_score_l738_738871

noncomputable def avg (s : List ℤ) : ℤ := s.sum / s.length

theorem second_lowest_exam_score 
  (s : List ℤ) 
  (h_length : s.length = 6)
  (h_avg : avg s = 74)
  (h_mode : (s.count 76) = 2)
  (h_median : s.median = some 76)
  (h_min : s.minimum = some 50)
  (h_max : s.maximum = some 94)
  (h_unique_mode : ∀ x, s.count x ≤ 2) :
  {v : ℤ | ∃ t : List ℤ, t ⊂ s ∧ t.length = 6 ∧ (t.erase 50).minimum = some v} = {i | 55 ≤ i ∧ i ≤ 71}.to_finset.card = 17 :=
sorry

end second_lowest_exam_score_l738_738871


namespace path_counts_l738_738499

    noncomputable def x : ℝ := 2 + Real.sqrt 2
    noncomputable def y : ℝ := 2 - Real.sqrt 2

    theorem path_counts (n : ℕ) :
      ∃ α : ℕ → ℕ, (α (2 * n - 1) = 0) ∧ (α (2 * n) = (1 / Real.sqrt 2) * ((x ^ (n - 1)) - (y ^ (n - 1)))) :=
    by
      sorry
    
end path_counts_l738_738499


namespace X_on_circumcircle_CPQ_l738_738134

-- Definition of the geometric setup
variables {A B C P Q R X : Type}
variable [geometry A B C P Q R X] -- Assuming existence of the geometric entities

-- Basic condition: P on BC, Q on CA, R on AB
axiom P_on_BC : on P B C
axiom Q_on_CA : on Q C A
axiom R_on_AB : on R A B

-- Additional conditions: X as the second intersection point of given circumcircles
axiom X_on_circumcircle_AQR : on_circumcircle X A Q R
axiom X_on_circumcircle_BRP : on_circumcircle X B R P

-- The goal to prove that X is also on circumcircle of CPQ
theorem X_on_circumcircle_CPQ : on_circumcircle X C P Q :=
sorry

end X_on_circumcircle_CPQ_l738_738134


namespace ellipse_equation_dot_product_MA_MB_l738_738050

theorem ellipse_equation 
  (a b : ℝ) (h1 : ∀ x y : ℝ, (x^2)/(a^2) + (y^2)/(b^2) = 1)
  (h2 : ∀ x y : ℝ, x^2 + y^2 + 2*x = 0 → (x, y) = (-1, 0))
  (h3 : dist_on_ellipse : ∀ x y : ℝ, (x^2)/(a^2) + (y^2)/(b^2) = 1 → sqrt ((x + 1)^2 + y^2) = sqrt(2) - 1) :
  (a = sqrt 2 ∧ b = 1 ∧ (∀ x y : ℝ, (x^2)/2 + (y^2)/1 = 1)) :=
by {
  sorry
}

theorem dot_product_MA_MB 
  (x₁ y₁ x₂ y₂ k : ℝ)
  (h1 : ∀ x y : ℝ, (x^2)/(2) + y^2 = 1)
  (h2 : k > 0 ∨ k ≠ 0)
  (h3 : M = (-5/4, 0))
  (h4 : (x₁, y₁), (x₂, y₂) are points of intersection of the line and ellipse) :
  (x₁ * x₂ + (5/4) * (x₁ + x₂) + (25/16) + k^2 * (x₁ * x₂ + x₁ + x₂ + 1) = -7/16) :=
by {
  sorry
}

end ellipse_equation_dot_product_MA_MB_l738_738050


namespace percentage_of_girls_passed_l738_738824

theorem percentage_of_girls_passed
    (total_candidates : ℕ := 2000)
    (number_of_girls : ℕ := 900)
    (number_of_boys : ℕ := total_candidates - number_of_girls)
    (percentage_boys_passed : ℝ := 28)
    (percentage_total_failed : ℝ := 70.2) :
    (number_of_girls_passed / number_of_girls) * 100 = 32 :=
by
  let actual_number_of_boys_passed := 0.28 * number_of_boys
  let actual_number_of_failed := 0.702 * total_candidates
  let actual_number_of_passed := total_candidates - actual_number_of_failed
  let number_of_girls_passed := actual_number_of_passed - actual_number_of_boys_passed
  sorry

end percentage_of_girls_passed_l738_738824


namespace largest_k_l738_738264

namespace Proof

def seq_a : ℕ → ℤ
| 1       := -1
| (n + 2) := (1 + (n + 1) - (n + 2): ℤ) + seq_a (n + 1)

theorem largest_k (k : ℕ) (h : k = 4) : seq_a k + seq_a (k + 1) = seq_a (k + 2) :=
by 
  -- Sequence definition
  have h1 : seq_a 1 = -1 := rfl
  have h2 : ∀ (n ≥ 2), seq_a n = (1 + 2 + ... + n - (n + 1)) := 
    sorry  -- Proof of sequence formula from given conditions

  -- General form of a_n
  -- a_n = (n - 2) * (n + 1) / 2
  sorry

  -- Validating k = 4
  sorry

end Proof

end largest_k_l738_738264


namespace ratio_of_BH_to_HD_l738_738903

noncomputable def angle_DCA (arc_AD : ℝ) : ℝ := arc_AD / 2
noncomputable def angle_DBC (arc_CD : ℝ) : ℝ := arc_CD / 2

noncomputable def BH_div_HD (arc_AD arc_CD : ℝ) : ℝ :=
  let DCA := angle_DCA arc_AD
  let DBC := angle_DBC arc_CD
  let BHC : Triangle := ⟨45, 45, 90⟩ -- Isosceles right triangle properties
  let ratio : ℝ := 1 / (√3)
  ratio

theorem ratio_of_BH_to_HD (arc_AD arc_CD : ℝ) (hAD : arc_AD = 120) (hCD : arc_CD = 90) :
  BH_div_HD arc_AD arc_CD = 1 / (√3) :=
by
  rw [hAD, hCD]
  simp [BH_div_HD, angle_DCA, angle_DBC]
  sorry

end ratio_of_BH_to_HD_l738_738903


namespace find_width_of_prism_l738_738814

theorem find_width_of_prism (l h d : ℝ) (h_l : l = 4) (h_h : h = 9) (h_d : d = 15) :
    ∃ w : ℝ, sqrt (l^2 + w^2 + h^2) = d ∧ w = 8 * sqrt 2 :=
by
  sorry

end find_width_of_prism_l738_738814


namespace max_quotient_l738_738177

-- Define the given conditions
def conditions (a b : ℝ) :=
  100 ≤ a ∧ a ≤ 250 ∧ 700 ≤ b ∧ b ≤ 1400

-- State the theorem for the largest value of the quotient b / a
theorem max_quotient (a b : ℝ) (h : conditions a b) : b / a ≤ 14 :=
by
  sorry

end max_quotient_l738_738177


namespace handshakes_minimum_l738_738266

/-- Given 30 people and each person shakes hands with exactly three others,
    the minimum possible number of handshakes is 45. -/
theorem handshakes_minimum (n k : ℕ) (h_n : n = 30) (h_k : k = 3) :
  (n * k) / 2 = 45 :=
by
  sorry

end handshakes_minimum_l738_738266


namespace min_value_E_l738_738373

def E (x : ℝ) : ℝ := 4 + Real.tan (2 * Real.pi * Real.sin (Real.pi * x)) ^ 2 + 
                          (1 / Real.tan (3 * Real.pi * Real.cos (2 * Real.pi * x))) ^ 2

theorem min_value_E (x : ℝ) : 
  (∀ x, E x ≥ 4) ∧ 
  (E x = 4 ↔ (∃ m : ℤ, x = int.cast m  + 1 / 6 ∨ x = int.cast m - 1 / 6 ∨ x = int.cast m  + 5 / 6 ∨ x = int.cast m - 5 / 6)) :=
begin
  sorry
end

end min_value_E_l738_738373


namespace coefficient_x4_in_expansion_l738_738338

theorem coefficient_x4_in_expansion :
  (Polynomial.expand (fun x => (↑x : ℂ)) 10).coeff 4 = 45 :=
by sorry

end coefficient_x4_in_expansion_l738_738338


namespace verify_tin_amount_l738_738265

def ratio_to_fraction (part1 part2 : ℕ) : ℚ :=
  part2 / (part1 + part2 : ℕ)

def tin_amount_in_alloy (total_weight : ℚ) (ratio : ℚ) : ℚ :=
  total_weight * ratio

def alloy_mixture_tin_weight_is_correct
    (weight_A weight_B : ℚ)
    (ratio_A_lead ratio_A_tin : ℕ)
    (ratio_B_tin ratio_B_copper : ℕ) : Prop :=
  let tin_ratio_A := ratio_to_fraction ratio_A_lead ratio_A_tin
  let tin_ratio_B := ratio_to_fraction ratio_B_tin ratio_B_copper
  let tin_weight_A := tin_amount_in_alloy weight_A tin_ratio_A
  let tin_weight_B := tin_amount_in_alloy weight_B tin_ratio_B
  tin_weight_A + tin_weight_B = 146.57

theorem verify_tin_amount :
    alloy_mixture_tin_weight_is_correct 130 160 2 3 3 4 :=
by
  sorry

end verify_tin_amount_l738_738265


namespace minimum_pentagons_to_form_2011_gon_l738_738956

theorem minimum_pentagons_to_form_2011_gon : 
  (∀ n, (∑ i in finset.range (n-2), (180 : ℝ)) / 540 ≥ 2011 - 2) → 670 :=
by
  intro h
  have angle_sum_2011_gon : (2011 - 2) * 180 = 2009 * 180 := by sorry
  have angle_sum_pentagon : 540 := by sorry
  have min_pentagons : Real.ceil (2009 * 180 / 540) = 670 := by sorry
  exact min_pentagons

end minimum_pentagons_to_form_2011_gon_l738_738956


namespace river_length_GSA_AWRA_l738_738640

-- Define the main problem statement
noncomputable def river_length_estimate (GSA_length AWRA_length GSA_error AWRA_error error_prob : ℝ) : Prop :=
  (GSA_length = 402) ∧ (AWRA_length = 403) ∧ 
  (GSA_error = 0.5) ∧ (AWRA_error = 0.5) ∧ 
  (error_prob = 0.04) ∧ 
  (abs (402.5 - GSA_length) ≤ GSA_error) ∧ 
  (abs (402.5 - AWRA_length) ≤ AWRA_error) ∧ 
  (error_prob = 1 - (2 * 0.02))

-- The main theorem statement
theorem river_length_GSA_AWRA :
  river_length_estimate 402 403 0.5 0.5 0.04 :=
by
  sorry

end river_length_GSA_AWRA_l738_738640


namespace hexagon_inscribed_cos_angle_product_l738_738262

theorem hexagon_inscribed_cos_angle_product
  (A B C D E F : Point)
  (circle : Circle)
  (hexagon_inscribed : InscribedHexagon circle A B C D E F)
  (side_length : ∀ {X Y : Point}, {h : IsConsecutiveSide hexagon_inscribed X Y} → dist X Y = 3)
  (AD_eq_2 : dist A D = 2) :
  (1 - cos (angle B A D)) * (1 - cos (angle A D F)) = 2 / 9 := by
  sorry

end hexagon_inscribed_cos_angle_product_l738_738262


namespace part1_part2_l738_738074

variables {x A : ℝ}
noncomputable def vector_m (ω : ℝ) (x : ℝ) : ℝ × ℝ := (2 * Real.cos (ω * x), -1)
noncomputable def vector_n (ω : ℝ) (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin (ω * x) + Real.cos (ω * x), 1)

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  let m := vector_m ω x
  let n := vector_n ω x
  (m.1 * n.1 + m.2 * n.2)

theorem part1 (ω : ℝ) (hω : 0 < ω) 
  (hx_axis_intersect : ∃ x₁ x₂, x₁ < x₂ ∧ x₂ - x₁ = π / 2 ∧ f ω x₁ = 0 ∧ f ω x₂ = 0) :
  ∀ x ∈ Icc 0 (π / 2), -1 ≤ f ω x ∧ f ω x ≤ 2 := sorry

theorem part2 (a b c : ℝ) (h_height : a = 3 ∧ (3 * Real.sqrt 3) / 2 = (c / 2) * Real.sin A)
  (h_fA : f 1 A = 1) 
  (h_triangle_area : (3 * Real.sqrt 3) / 2 = (1 / 2) * a * ((c / 2) * Real.sin A)) :
  b = 3 ∧ c = 3 := sorry

end part1_part2_l738_738074


namespace power_sum_mod_inverse_l738_738233

theorem power_sum_mod_inverse (h : 3^6 ≡ 1 [MOD 17]) : 
  (3^(-1) + 3^(-2) + 3^(-3) + 3^(-4) +  3^(-5) + 3^(-6)) ≡ 1 [MOD 17] := 
by
  sorry

end power_sum_mod_inverse_l738_738233


namespace equalize_expenses_l738_738374

def total_expenses := 130 + 160 + 150 + 180
def per_person_share := total_expenses / 4
def tom_owes := per_person_share - 130
def dorothy_owes := per_person_share - 160
def sammy_owes := per_person_share - 150
def alice_owes := per_person_share - 180
def t := tom_owes
def d := dorothy_owes

theorem equalize_expenses : t - dorothy_owes = 30 := by
  sorry

end equalize_expenses_l738_738374


namespace circle_points_chord_intersections_l738_738488

/-
Let X₁, Z₂, Y₁, X₂, Z₁, Y₂ be six points lying on the periphery of a circle in this order.
Let the chords Y₁Y₂ and Z₁Z₂ meet at a point A;
Let the chords Z₁Z₂ and X₁X₂ meet at a point B;
Let the chords X₁X₂ and Y₁Y₂ meet at a point C.
We want to prove that (BX₂ - CX₁) * BC + (CY₂ - AY₁) * CA + (AZ₂ - BZ₁) * AB = 0.
-/

variables {X₁ Z₂ Y₁ X₂ Z₁ Y₂ A B C : Type}
variables [Field K] [has_sub K] [has_mul K] [has_add K] [EqNullSupp K]

theorem circle_points_chord_intersections
  (h1: ∀ {a b c d e f : K}, (X₁ ≤ Z₂) ∧ (Z₂ ≤ Y₁) ∧ (Y₁ ≤ X₂) ∧ (X₂ ≤ Z₁) ∧ (Z₁ ≤ Y₂) ∧ (Y₂ ≤ A) ∧
                           (Y₁ = Y₂) ∧ (Z₁ = Z₂) ∧ (X₁ = X₂)) :
  ((BX₂ - CX₁) * BC + (CY₂ - AY₁) * CA + (AZ₂ - BZ₁) * AB) = 0 :=
sorry

end circle_points_chord_intersections_l738_738488


namespace quadratic_inequality_solution_l738_738352

theorem quadratic_inequality_solution (x : ℝ) : -3 < x ∧ x < 4 → x^2 - x - 12 < 0 := by
  sorry

end quadratic_inequality_solution_l738_738352


namespace necessary_but_not_sufficient_l738_738761

variable {a b : ℝ}

theorem necessary_but_not_sufficient : (a < b + 1) ∧ ¬ (a < b + 1 → a < b) :=
by
  sorry

end necessary_but_not_sufficient_l738_738761


namespace circle_C_equations_trajectory_midpoint_M_l738_738382

-- First part: Finding the possible equations of circle C
theorem circle_C_equations (A B : ℝ × ℝ) (A_coord : A = (1, 2)) (B_coord : B = (1, 10)) 
                           (C_tangent_line : ∀ (x y : ℝ), (x = 2 * y + 1))
                           : ∃ (a r: ℝ), (( (x - a) ^ 2 + (y - 6) ^ 2 = r ^ 2 ) 
                                         ∧ (((1 - a) ^ 2 + 16 = r ^ 2) 
                                         ∧ ( abs(a - 13) = r * √5)
                                         ∨ ((1 - a) ^ 2 + 16 = r ^ 2) 
                                         ∧ ( abs(a - 13) = 4 * r )))
                           := sorry

-- Second part: Finding the trajectory of midpoint M
theorem trajectory_midpoint_M (P Q M : ℝ × ℝ) (P_on_C : ∀ P: ℝ × ℝ, 
                                                        ((P.fst - a) ^ 2 + (P.snd - 6) ^ 2 = r ^ 2))
                             (Q_coord : Q = (-3, -6)) (M_eq : M = ((-3 + P.fst) / 2, (-6 + P.snd) / 2))
                             : ∃ P: ℝ × ℝ, ((P.fst - 3) ^ 2 + (P.snd - 6) ^ 2 = 20 
                                         ∧ (M.fst ^ 2 + M.snd ^ 2 = 5)
                                         ∨ ((P.fst + 7)  ^ 2 + (P.snd - 6) ^ 2 = 80
                                         ∧ ( (M.fst + 5) ^ 2 + M.snd ^ 2 = 20 )))
                           := sorry

end circle_C_equations_trajectory_midpoint_M_l738_738382


namespace inequality_proof_l738_738883

variable (a : ℝ)

theorem inequality_proof (a : ℝ) : 
  (a^2 + a + 2) / (Real.sqrt (a^2 + a + 1)) ≥ 2 :=
sorry

end inequality_proof_l738_738883


namespace total_patients_in_a_year_l738_738844

-- Define conditions from the problem
def patients_per_day_first : ℕ := 20
def percent_increase_second : ℕ := 20
def working_days_per_week : ℕ := 5
def working_weeks_per_year : ℕ := 50

-- Lean statement for the problem
theorem total_patients_in_a_year (patients_per_day_first : ℕ) (percent_increase_second : ℕ) (working_days_per_week : ℕ) (working_weeks_per_year : ℕ) :
  (patients_per_day_first + ((patients_per_day_first * percent_increase_second) / 100)) * working_days_per_week * working_weeks_per_year = 11000 :=
by
  sorry

end total_patients_in_a_year_l738_738844


namespace interval_monotonicity_uniq_solution_range_l738_738865

noncomputable def f1 (x : ℝ) : ℝ :=
  Real.log x - (1 / 4) * x ^ 2 - (1 / 2) * x

noncomputable def f2 (x : ℝ) : ℝ :=
  Real.log x + x

theorem interval_monotonicity (a b : ℝ) (h : a = 1 / 2 ∧ b = 1 / 2) :
  (∀ x : ℝ, 0 < x ∧ x < 1 → 0 < derivative f1 x) ∧
  (∀ x : ℝ, 1 < x → derivative f1 x < 0) :=
sorry

theorem uniq_solution_range (a b : ℝ) (h : a = 0 ∧ b = -1) :
  (∀ m : ℝ, (1 ≤ m ∧ m < 1 + (2 / Real.exp 2)) →
    ∃! x : ℝ, 1 ≤ x ∧ x ≤ Real.exp 2 ∧ (Real.log x + x = m * x)) :=
sorry

end interval_monotonicity_uniq_solution_range_l738_738865


namespace female_officers_count_l738_738516

theorem female_officers_count {total_on_duty : ℝ} (h1 : total_on_duty = 204)
  (h2 : ∀ f_on_duty, f_on_duty = total_on_duty / 2)
  (p : ℝ) (h3 : p = 0.17) :
  (102 : ℝ) = f_on_duty →
  ∃ total_female_officers : ℝ, total_female_officers = 600 :=
by
  intro h_female_on_duty
  use 600
  sorry

end female_officers_count_l738_738516


namespace bank_discount_correct_l738_738669

variables (BillAmount : ℝ) (BillDueMonths : ℝ) (TrueDiscount : ℝ) (BankDiscountRate : ℝ)

def calc_bankers_discount (BillAmount : ℝ) (BankDiscountRate : ℝ) (BillDueYears : ℝ) : ℝ :=
  (BillAmount * BankDiscountRate * BillDueYears) / 100

def calc_effective_discount_rate (BillAmount : ℝ) (EffectiveDiscount : ℝ) : ℝ :=
  (EffectiveDiscount / BillAmount) * 100

theorem bank_discount_correct
  (BillAmount : ℝ := 12800)
  (BillDueMonths : ℝ := 6)
  (TrueDiscount : ℝ := (5/8) * 0.045 * 12800)
  (BankDiscountRate : ℝ := 7.5) :
  calc_bankers_discount BillAmount BankDiscountRate (BillDueMonths / 12) = 480
  ∧ calc_effective_discount_rate BillAmount 480 = 3.75 :=
by
  sorry

end bank_discount_correct_l738_738669


namespace general_term_of_sequence_l738_738031

def is_arithmetic_sequence {α : Type*} [AddCommGroup α] (a : ℕ → α) (d : α) : Prop :=
∀ n : ℕ, a (n + 1) - a n = d

theorem general_term_of_sequence (a : ℕ → ℝ) 
  (h_seq : is_arithmetic_sequence (λ n, Real.log (a n)) (Real.log 3))
  (h_sum : Real.log (a 0) + Real.log (a 1) + Real.log (a 2) = 6 * Real.log 3) :
  ∀ n, a n = 3^n :=
by
  sorry

end general_term_of_sequence_l738_738031


namespace seating_chart_representation_l738_738434

structure SeatingChartNotation (cols rows : ℕ)

def notation (n : ℕ × ℕ) : SeatingChartNotation n.1 n.2 := 
  ⟨n.1, n.2⟩

theorem seating_chart_representation :
  notation (2, 3) = SeatingChartNotation 2 3 ∧ notation (5, 4) = SeatingChartNotation 5 4 :=
by
  sorry

end seating_chart_representation_l738_738434


namespace proof_paddle_prices_and_cost_optimization_l738_738288

-- Define the conditions
variables (x y : ℕ)
variables (m : ℕ)
variables (cost : ℕ)

-- The first condition: Total cost for 20 straight + 15 horizontal pairs of paddles with balls is 9000 yuan
def condition1 := 20 * (x + 20 * 2) + 15 * (y + 20 * 2) = 9000

-- The second condition: Cost difference between 10 horizontal and 5 straight pairs is 1600 yuan
def condition2 := 5 * (x + 20 * 2) + 1600 = 10 * (y + 20 * 2)

-- The threshold condition: Number of straight paddles is not more than three times the number of horizontal paddles
def condition3 := m ≤ 3 * (40 - m)

-- The optimization problem: Minimizing total cost for 40 pairs of paddles
def total_cost := (220 + 20 * 2) * m + (260 + 20 * 2) * (40 - m)

theorem proof_paddle_prices_and_cost_optimization 
    (h1 : condition1)
    (h2 : condition2)
    (h3 : condition3)
    (h4 : x = 220)
    (h5 : y = 260) : total_cost = 10000 :=
sorry  -- Proof left as an exercise

end proof_paddle_prices_and_cost_optimization_l738_738288


namespace arithmetic_geometric_sequence_problems_l738_738021

/-- Given a non-zero common difference arithmetic sequence {a_n} with the sum of the first n
terms S_n, and the conditions S_6 = 60 and a_6 being the geometric mean of a_1 and a_21,
prove that a_n = 2n + 3 and S_n = n(n + 4).
Additionally, for the sequence {b_n} such that b_{n+1} - b_n = a_n and b_1 = 3, prove
that the sum of the first n terms T_n of the sequence {1 / b_n} is (3n^2 + 5n) / (4(n + 1)(n + 2)).
-/
theorem arithmetic_geometric_sequence_problems
  (a_n : ℕ → ℤ)
  (S_n : ℕ → ℤ)
  (b_n : ℕ → ℤ)
  (T_n : ℕ → ℤ)
  (d : ℤ)
  (n : ℕ)
  (h1 : ∀ n, a_n = 2 * n + 3)
  (h2 : ∀ n, S_n = n * (n + 4))
  (h3 : b_1 = 3)
  (h4 : ∀ n, b_{n+1} - b_n = a_n)
  (h5 : ∀ n, T_n = (3 * n ^ 2 + 5 * n) / (4 * (n + 1) * (n + 2))) :
  (a_6 = 13 ∧ S_6 = 60 ∧ b_n = n * (n + 2) ∧ T_n = (3 * n ^ 2 + 5 * n) / (4 * (n + 1) * (n + 2))) :=
by 
  sorry

end arithmetic_geometric_sequence_problems_l738_738021


namespace simplify_expression_l738_738565

theorem simplify_expression : (Real.sqrt (9 / 4) - Real.sqrt (4 / 9)) = 5 / 6 :=
by
  sorry

end simplify_expression_l738_738565


namespace range_of_x1_f_x2_l738_738056

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 4 * Real.exp 1 else Real.exp x / x^2

theorem range_of_x1_f_x2:
  ∃ (x1 x2 : ℝ), x1 ≤ 0 ∧ 0 < x2 ∧ f x1 = f x2 ∧ -4 * (Real.exp 1)^2 ≤ x1 * f x2 ∧ x1 * f x2 ≤ 0 :=
sorry

end range_of_x1_f_x2_l738_738056


namespace roots_sum_product_l738_738917

theorem roots_sum_product (p q : ℝ) (h_sum : p / 3 = 8) (h_prod : q / 3 = 12) : p + q = 60 := 
by 
  sorry

end roots_sum_product_l738_738917


namespace inequality_proof_l738_738531

theorem inequality_proof (a b x y : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : x / a < y / b) :
  (1 / 2) * (x / a + y / b) > (x + y) / (a + b) := by
  sorry

end inequality_proof_l738_738531


namespace tan_x_eq_sqrt3_when_parallel_interval_of_monotonic_increase_l738_738379

-- Definitions of vectors m and n
def m (x : ℝ) := (Real.sin (x - Real.pi / 6), 1)
def n (x : ℝ) := (Real.cos x, 1)

-- First statement: Prove tan x = sqrt 3 when m is parallel to n
theorem tan_x_eq_sqrt3_when_parallel (x : ℝ) (h : m x.1 / m x.2 = n x.1 / n x.2) : Real.tan x = Real.sqrt 3 :=
sorry

-- Function f definition
def f (x : ℝ) := (m x.1) • (n x.1)

-- Second statement: Prove the interval of monotonic increase for f(x)
theorem interval_of_monotonic_increase (x : ℝ) (k : ℤ) :
  ∃ k : ℤ, - (Real.pi / 6) + (k * Real.pi) ≤ x ∧ x ≤ (Real.pi / 3) + (k * Real.pi) ⇔
  IsMonotonic (f x) :=
sorry

end tan_x_eq_sqrt3_when_parallel_interval_of_monotonic_increase_l738_738379


namespace image_length_interval_two_at_least_four_l738_738686

noncomputable def quadratic_function (p q r : ℝ) : ℝ → ℝ :=
  fun x => p * (x - q)^2 + r

theorem image_length_interval_two_at_least_four (p q r : ℝ)
  (h : ∀ I : Set ℝ, (∀ a b : ℝ, I = Set.Icc a b ∨ I = Set.Ioo a b → |b - a| = 1 → |quadratic_function p q r b - quadratic_function p q r a| ≥ 1)) :
  ∀ I' : Set ℝ, (∀ a b : ℝ, I' = Set.Icc a b ∨ I' = Set.Ioo a b → |b - a| = 2 → |quadratic_function p q r b - quadratic_function p q r a| ≥ 4) :=
by
  sorry


end image_length_interval_two_at_least_four_l738_738686


namespace CityResidentShouldInstallLampHimself_l738_738969

structure CityResidentLampDecision where
  P1 : ℕ -- 60W incandescent lamp
  P2 : ℕ -- 12W energy-efficient lamp
  T : ℕ -- Tariff 5 rubles/kWh
  t : ℕ -- Monthly usage time in hours
  costEfficientLamp : ℕ -- Initial cost of energy-efficient lamp
  serviceCompanyRate : ℝ -- Rate of payment to the energy service company

def monthlyConsumption (P : ℕ) (t : ℕ) : ℝ :=
  (P * t) / 1000.0

def monthlyCost (E : ℝ) (T : ℕ) : ℝ :=
  E * T

def totalCost (C : ℕ) (months : ℕ) : ℕ :=
  C * months

def serviceCompanyPayment (ΔC : ℝ) (rate : ℝ) : ℝ :=
  ΔC * rate

def validDecision10Months (r : CityResidentLampDecision) :=
  let E1 := monthlyConsumption r.P1 r.t
  let E2 := monthlyConsumption r.P2 r.t
  let C1 := monthlyCost E1 r.T
  let C2 := monthlyCost E2 r.T
  let total10MonthCost1 := totalCost C1 10
  let total10MonthCost2 := r.costEfficientLamp + totalCost C2 10
  let ΔE := E1 - E2
  let ΔC := ΔE * r.T
  let servicePayment := serviceCompanyPayment ΔC r.serviceCompanyRate
  let newC := C2 + servicePayment
  let totalCostCompany := totalCost newC 10

  total10MonthCost2 <= totalCostCompany

def validDecisionFullLifespan (r : CityResidentLampDecision) :=
  let E2 := monthlyConsumption r.P2 r.t
  let C2 := monthlyCost E2 r.T
  let total36MonthCost := r.costEfficientLamp + totalCost C2 36
  let remainingMonths := 36 - 10
  let remainingCostCompany := totalCost C2 remainingMonths
  let totalCostCompany := 240 + remainingCostCompany

  total36MonthCost <= totalCostCompany

theorem CityResidentShouldInstallLampHimself : ∀ r : CityResidentLampDecision,
  validDecision10Months r ∧ validDecisionFullLifespan r := by
    sorry

end CityResidentShouldInstallLampHimself_l738_738969


namespace find_cos_C_l738_738112

noncomputable def cos_C (a b c : ℝ) : ℝ :=
(a^2 + b^2 - c^2) / (2 * a * b)

theorem find_cos_C (a b c : ℝ) (h1 : b^2 = a * c) (h2 : c = 2 * a) :
  cos_C a b c = -√2/4 :=
by
  sorry

end find_cos_C_l738_738112


namespace arithmetic_and_geometric_mean_l738_738535

theorem arithmetic_and_geometric_mean (x y : ℝ) (h₁ : (x + y) / 2 = 20) (h₂ : Real.sqrt (x * y) = Real.sqrt 150) : x^2 + y^2 = 1300 :=
by
  sorry

end arithmetic_and_geometric_mean_l738_738535


namespace perpendicular_line_parallel_planes_l738_738774

-- Define the conditions
def m_parallel_alpha (m alpha : Set ℝℝ) : Prop := Parallel m alpha
def m_perpendicular_alpha (m alpha : Set ℝℝ) : Prop := Perpendicular m alpha
def m_subset_alpha (m alpha : Set ℝℝ) : Prop := m ⊆ alpha
def alpha_parallel_beta (alpha beta : Set ℝℝ) : Prop := Parallel alpha beta
def m_perpendicular_beta (m beta : Set ℝℝ) : Prop := Perpendicular m beta

-- Simplified assumption definitions for the conditions given in the problem
variable (m alpha beta : Set ℝℝ)

-- Main theorem statement
theorem perpendicular_line_parallel_planes :
  m_perpendicular_alpha m alpha →
  alpha_parallel_beta alpha beta →
  m_perpendicular_beta m beta :=
by
  sorry

end perpendicular_line_parallel_planes_l738_738774


namespace security_deposit_percentage_l738_738846

theorem security_deposit_percentage
    (daily_rate : ℝ) (pet_fee : ℝ) (service_fee_rate : ℝ) (days : ℝ) (security_deposit : ℝ)
    (total_cost : ℝ) (expected_percentage : ℝ) :
    daily_rate = 125.0 →
    pet_fee = 100.0 →
    service_fee_rate = 0.20 →
    days = 14 →
    security_deposit = 1110 →
    total_cost = daily_rate * days + pet_fee + (daily_rate * days + pet_fee) * service_fee_rate →
    expected_percentage = (security_deposit / total_cost) * 100 →
    expected_percentage = 50 :=
by
  intros
  sorry

end security_deposit_percentage_l738_738846


namespace sequence_neither_arithmetic_nor_geometric_l738_738767

noncomputable def Sn (n : ℕ) : ℕ := 3 * n + 2
noncomputable def a (n : ℕ) : ℕ := if n = 1 then 5 else Sn n - Sn (n - 1)

def not_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ¬∃ d, ∀ n, a (n + 1) = a n + d

def not_geometric_sequence (a : ℕ → ℕ) : Prop :=
  ¬∃ r, ∀ n, a (n + 1) = r * a n

theorem sequence_neither_arithmetic_nor_geometric :
  not_arithmetic_sequence a ∧ not_geometric_sequence a :=
sorry

end sequence_neither_arithmetic_nor_geometric_l738_738767


namespace estimate_probability_concave_l738_738718

noncomputable def times_thrown : ℕ := 1000
noncomputable def frequency_convex : ℝ := 0.44

theorem estimate_probability_concave :
  (1 - frequency_convex) = 0.56 := by
  sorry

end estimate_probability_concave_l738_738718


namespace integral_evaluation_l738_738735

noncomputable def integral_1 : ℝ := ∫ x in 0..1, real.sqrt (1 - (x - 1)^2)

noncomputable def integral_2 : ℝ := ∫ x in 0..1, x^2

theorem integral_evaluation : integral_1 - integral_2 = (Real.pi / 4) - (1 / 3) :=
by 
  have integral1_val : integral_1 = Real.pi / 4 := by sorry,
  have integral2_val : integral_2 = 1 / 3 := by sorry,
  rw [integral1_val, integral2_val]

end integral_evaluation_l738_738735


namespace more_people_joined_l738_738212

def initial_people : Nat := 61
def final_people : Nat := 83

theorem more_people_joined :
  final_people - initial_people = 22 := by
  sorry

end more_people_joined_l738_738212


namespace problem1_problem2_l738_738976

namespace ProofProblems

-- Problem 1: Prove the inequality
theorem problem1 (x : ℝ) (h : x + |2 * x - 1| < 3) : -2 < x ∧ x < 4 / 3 := 
sorry

-- Problem 2: Prove the value of x + y + z 
theorem problem2 (x y z : ℝ) 
  (h1 : x^2 + y^2 + z^2 = 1) 
  (h2 : x + 2 * y + 3 * z = Real.sqrt 14) : 
  x + y + z = 3 * Real.sqrt 14 / 7 := 
sorry

end ProofProblems

end problem1_problem2_l738_738976


namespace min_handshakes_30_people_3_each_l738_738274

theorem min_handshakes_30_people_3_each : 
  ∃ (H : ℕ), (∀ (n k : ℕ), n = 30 ∧ k = 3 → H = (n * k) / 2) := 
by {
  use 45,
  intros n k h,
  rw [← h.1, ← h.2],
  norm_num,
  sorry
}

end min_handshakes_30_people_3_each_l738_738274


namespace num_possible_pairs_l738_738173

theorem num_possible_pairs (f m : ℕ) (h_f : f ≥ 0) (h_m : m ≥ 0)
  (seated : Finset (ℕ × ℕ)) :
  (seated = {(0, 6), (2, 6), (4, 6), (5, 6), (6, 6), (6, 0)}) →
  Finset.card seated = 6 :=
by
  intro h
  rw [h]
  simp [Finset.card]
  sorry

end num_possible_pairs_l738_738173


namespace tangent_line_at_neg1_neg1_l738_738564

def curve (x : ℝ) : ℝ := x / (x + 2)

theorem tangent_line_at_neg1_neg1 : 
  ∀ (x y : ℝ), (x, y) = (-1, -1) → 
  ∀ t : ℝ, y = 2 * t + 1 → curve x = y :=
sorry

end tangent_line_at_neg1_neg1_l738_738564


namespace argument_of_z_l738_738327

noncomputable def z : ℂ :=
  complex.exp ((19 * real.pi * complex.I) / 60) +
  complex.exp ((29 * real.pi * complex.I) / 60) +
  complex.exp ((39 * real.pi * complex.I) / 60) +
  complex.exp ((49 * real.pi * complex.I) / 60) +
  complex.exp ((59 * real.pi * complex.I) / 60)

theorem argument_of_z : complex.arg z = 13 * real.pi / 20 :=
by
  sorry

end argument_of_z_l738_738327


namespace number_of_ways_to_form_team_l738_738589

noncomputable def binomial : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n + 1, k + 1 => binomial n k + binomial n (k + 1)

theorem number_of_ways_to_form_team :
  let total_selections := binomial 11 5
  let all_boys_selections := binomial 8 5
  total_selections - all_boys_selections = 406 :=
by 
  sorry

end number_of_ways_to_form_team_l738_738589


namespace inequality_am_gm_l738_738881

theorem inequality_am_gm (a b : ℝ) (h1 : a < 1) (h2 : b < 1) (h3 : a + b ≥ 1/2) :
  (1 - a) * (1 - b) ≤ 9 / 16 := 
by
  sorry

end inequality_am_gm_l738_738881


namespace reciprocal_neg_sin60_eq_neg_2sqrt3_div_3_l738_738915

-- Define the conditions
def sin60 := Real.sin (Real.pi / 3)

-- Prove the statement
theorem reciprocal_neg_sin60_eq_neg_2sqrt3_div_3 : -sin60 ≠ 0 → (1 / -sin60) = - (2 * Real.sqrt 3) / 3 :=
by
  -- Conditions from part a)
  have h1 : sin60 = Real.sqrt 3 / 2 := by sorry
  intro h_neg_sin60_ne_zero
  -- Proof will go here, for now we skip it
  sorry

end reciprocal_neg_sin60_eq_neg_2sqrt3_div_3_l738_738915


namespace internal_external_segment_product_l738_738601

-- Define the structure of the problem
variables {α : Type*} [linear_ordered_field α] [inhabited α]

-- Main theorem statement
theorem internal_external_segment_product
  (Δ : Type*) [acute_triangle Δ]
  (H : point) [orthocenter Δ H]
  (L1 L2 : line) [perpendicular L1 L2]
  (F D : point) [on_line_through F H L1] [on_line_through D H L1]
  (F1 E1 : point) [on_line_through F1 H L2] [on_line_through E1 H L2]
  (E D1 : point) [on_line_through E H L1] [on_line_through D1 H L2] :
  segment_product FD F1E1 = segment_product FE F1D1 := sorry

end internal_external_segment_product_l738_738601


namespace candy_problem_l738_738700

theorem candy_problem
  (n : ℕ) (h1 : 100 ≤ n) (h2 : n ≤ 999)
  (h3 : n + 7 ≡ 0 [MOD 9])
  (h4 : n - 9 ≡ 0 [MOD 6]) :
  n = 101 :=
sorry

end candy_problem_l738_738700


namespace geometric_sequence_a_5_value_l738_738111

noncomputable def geometric_sequence_a_n
  (a : ℕ → ℝ) (n : ℕ) (r : ℝ) : Prop :=
∀ n, a n = a 1 * r ^ (n - 1)

theorem geometric_sequence_a_5_value
  {a : ℕ → ℝ}
  {r : ℝ}
  (h_pos : ∀ n, a n > 0)
  (h_geom : geometric_sequence_a_n a r)
  (h_product : a 3 * a 7 = 64) :
  a 5 = 8 :=
sorry

end geometric_sequence_a_5_value_l738_738111


namespace expected_value_of_ten_sided_die_l738_738297

theorem expected_value_of_ten_sided_die : 
  let outcomes := finset.range 10 in
  let total := (finset.sum outcomes (λ n, (n + 1))) in
  (total / 10 : ℝ) = 5.5 := 
by
  let outcomes := finset.range 10
  let total := finset.sum outcomes (λ n, (n + 1))
  have h : total = 55 := by sorry
  show (total / 10 : ℝ) = 5.5, from sorry

end expected_value_of_ten_sided_die_l738_738297


namespace volume_of_wedge_l738_738691

theorem volume_of_wedge (h : 2 * Real.pi * r = 18 * Real.pi) :
  let V := (4 / 3) * Real.pi * (r ^ 3)
  let V_wedge := V / 6
  V_wedge = 162 * Real.pi :=
by
  sorry

end volume_of_wedge_l738_738691


namespace smallest_possible_dots_to_remove_l738_738630

theorem smallest_possible_dots_to_remove :
  ∃ S : Finset (Fin 4 × Fin 4), S.card = 4 ∧
    (∀ x y : Fin 4, 
      (∃ c1 c2 c3 c4 : Finset (Fin 4 × Fin 4),
        c1.card = 4 ∧ (x, y) ∈ c1 ∧ ((x + 1) % 4, y) ∈ c1 ∧ ((x, y + 1) % 4) ∈ c1 ∧ ((x + 1) % 4, (y + 1) % 4) ∈ c1 →
        c1 ∩ S ≠ ∅)
    ∧
      (∃ c1 c2 c3 c4 : Finset (Fin 4 × Fin 4),
        c2.card = 4 ∧ (x, y) ∈ c2 ∧ ((x + 2) % 4, y) ∈ c2 ∧ ((x, y + 2) % 4) ∈ c2 ∧ ((x + 2) % 4, (y + 2) % 4) ∈ c2 →
        c2 ∩ S ≠ ∅)
    ∧
      (∃ c1 c2 c3 c4 : Finset (Fin 4 × Fin 4),
        c3.card = 4 ∧ (x, y) ∈ c3 ∧ ((x + 3) % 4, y) ∈ c3 ∧ ((x, y + 3) % 4) ∈ c3 ∧ ((x + 3) % 4, (y + 3) % 4) ∈ c3 →
        c3 ∩ S ≠ ∅)) := sorry

end smallest_possible_dots_to_remove_l738_738630


namespace inverse_sum_mod_l738_738223

theorem inverse_sum_mod (h1 : ∃ k, 3^6 ≡ 1 [MOD 17])
                        (h2 : ∃ k, 3 * 6 ≡ 1 [MOD 17]) : 
  (6 + 9 + 2 + 1 + 6 + 1) % 17 = 8 :=
by
  cases h1 with k1 h1
  cases h2 with k2 h2
  sorry

end inverse_sum_mod_l738_738223


namespace ant_travel_distance_l738_738933

theorem ant_travel_distance (r1 r2 r3 : ℝ) (h1 : r1 = 5) (h2 : r2 = 10) (h3 : r3 = 15) :
  let A_large := (1/3) * 2 * Real.pi * r3
  let D_radial := (r3 - r2) + (r2 - r1)
  let A_middle := (1/3) * 2 * Real.pi * r2
  let D_small := 2 * r1
  let A_small := (1/2) * 2 * Real.pi * r1
  A_large + D_radial + A_middle + D_small + A_small = (65 * Real.pi / 3) + 20 :=
by
  sorry

end ant_travel_distance_l738_738933


namespace double_inequality_pos_reals_equality_condition_l738_738974

theorem double_inequality_pos_reals (x y z : ℝ) (x_pos: 0 < x) (y_pos: 0 < y) (z_pos: 0 < z):
  0 < (1 / (x + y + z + 1)) - (1 / ((x + 1) * (y + 1) * (z + 1))) ∧
  (1 / (x + y + z + 1)) - (1 / ((x + 1) * (y + 1) * (z + 1))) ≤ (1 / 8) :=
  sorry

theorem equality_condition (x y z : ℝ) :
  ((1 / (x + y + z + 1)) - (1 / ((x + 1) * (y + 1) * (z + 1))) = (1 / 8)) ↔ x = 1 ∧ y = 1 ∧ z = 1 :=
  sorry

end double_inequality_pos_reals_equality_condition_l738_738974


namespace combined_river_length_estimate_l738_738634

def river_length_GSA := 402 
def river_error_GSA := 0.5 
def river_prob_error_GSA := 0.04 

def river_length_AWRA := 403 
def river_error_AWRA := 0.5 
def river_prob_error_AWRA := 0.04 

/-- 
Given the measurements from GSA and AWRA, 
the combined estimate of the river's length, Rio-Coralio, is 402.5 km,
and the probability of error for this combined estimate is 0.04.
-/
theorem combined_river_length_estimate :
  ∃ l : ℝ, l = 402.5 ∧ ∀ p : ℝ, (p = 0.04) :=
sorry

end combined_river_length_estimate_l738_738634


namespace minimum_handshakes_l738_738270

-- Definitions
def people : ℕ := 30
def handshakes_per_person : ℕ := 3

-- Theorem statement
theorem minimum_handshakes : (people * handshakes_per_person) / 2 = 45 :=
by
  sorry

end minimum_handshakes_l738_738270


namespace div_identity_l738_738439

theorem div_identity (a b c : ℚ) (h1 : a / b = 3) (h2 : b / c = 2 / 5) : c / a = 5 / 6 := by
  sorry

end div_identity_l738_738439


namespace handshakes_minimum_l738_738267

/-- Given 30 people and each person shakes hands with exactly three others,
    the minimum possible number of handshakes is 45. -/
theorem handshakes_minimum (n k : ℕ) (h_n : n = 30) (h_k : k = 3) :
  (n * k) / 2 = 45 :=
by
  sorry

end handshakes_minimum_l738_738267


namespace remainder_division_example_l738_738156

theorem remainder_division_example :
  ∀ n, (n = 8 * 8 + 0) → (∃ q r, n = 5 * q + r ∧ 0 ≤ r ∧ r < 5 ∧ r = 4) :=
by
  intros n h
  have hn : n = 64 := by { rw h }
  use 12 -- quotient
  use 4 -- remainder
  sorry

end remainder_division_example_l738_738156


namespace river_length_GSA_AWRA_l738_738639

-- Define the main problem statement
noncomputable def river_length_estimate (GSA_length AWRA_length GSA_error AWRA_error error_prob : ℝ) : Prop :=
  (GSA_length = 402) ∧ (AWRA_length = 403) ∧ 
  (GSA_error = 0.5) ∧ (AWRA_error = 0.5) ∧ 
  (error_prob = 0.04) ∧ 
  (abs (402.5 - GSA_length) ≤ GSA_error) ∧ 
  (abs (402.5 - AWRA_length) ≤ AWRA_error) ∧ 
  (error_prob = 1 - (2 * 0.02))

-- The main theorem statement
theorem river_length_GSA_AWRA :
  river_length_estimate 402 403 0.5 0.5 0.04 :=
by
  sorry

end river_length_GSA_AWRA_l738_738639


namespace tanx_parallel_phi_value_l738_738803

variables {x φ : ℝ} (k : ℤ)
def vec_a : ℝ × ℝ := (Real.sin x, Real.cos x)
def vec_b : ℝ × ℝ := (1, Real.sqrt 3)
def f (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x
def g (x : ℝ) (φ : ℝ) : ℝ := Real.sin (1/2 * x + 1/2 * φ + Real.pi / 3)

-- Prove that if vec_a is parallel to vec_b, then tanx = sqrt(3) / 3
theorem tanx_parallel (h_parallel : Real.sqrt 3 * Real.sin x = Real.cos x) :
  Real.tan x = Real.sqrt 3 / 3 :=
sorry

-- Prove that if g is symmetric about the y-axis, then φ = π / 3
theorem phi_value (h_sym : ∀ x, g x φ = g (-x) φ)
  (h_range : 0 < φ ∧ φ < Real.pi) :
  ∃ k : ℤ, φ = 2 * k * Real.pi + Real.pi / 3 :=
sorry

end tanx_parallel_phi_value_l738_738803


namespace angle_AST_is_90_degrees_l738_738143

variables (A B C D T S O : Point)
variables (R : Real)
variables (α : Angle)
variables [acute_triangle : Triangle ABC] 

-- Conditions in Lean
variables (AC_lt_AB : AC < AB)
variables (circumradius_R : circumradius ABC = R)
variables (D_foot : foot altitude A D B C)
variables (AT_length : AT = 2 * R)
variables (D_between_A_and_T : between A D T)
variables (S_midpoint_arc_BC : is_midpoint_of_arc S B C)

-- Question in Lean: to prove angle AST = 90 degrees
theorem angle_AST_is_90_degrees :
  ∠AST = 90 :=
sorry

end angle_AST_is_90_degrees_l738_738143


namespace euler_lines_concurrent_l738_738130

theorem euler_lines_concurrent
  (A B C F : Type)
  [Triangle A B C]
  (hF_interior : PointInInteriorOfTriangle F A B C)
  (h_angles : ∠AFB = 120 ∧ ∠BFC = 120 ∧ ∠CFA = 120) :
  ConcurrentEulerLines A B C F :=
sorry

end euler_lines_concurrent_l738_738130


namespace unique_right_triangle_construction_l738_738720

noncomputable def right_triangle_condition (c f : ℝ) : Prop :=
  f < c / 2

theorem unique_right_triangle_construction (c f : ℝ) (h_c : 0 < c) (h_f : 0 < f) :
  right_triangle_condition c f :=
  sorry

end unique_right_triangle_construction_l738_738720


namespace total_money_correct_l738_738886

-- Define the number of pennies and quarters Sam has
def pennies : ℕ := 9
def quarters : ℕ := 7

-- Define the value of one penny and one quarter
def penny_value : ℝ := 0.01
def quarter_value : ℝ := 0.25

-- Calculate the total value of pennies and quarters Sam has
def total_value : ℝ := pennies * penny_value + quarters * quarter_value

-- Proof problem: Prove that the total value of money Sam has is $1.84
theorem total_money_correct : total_value = 1.84 :=
sorry

end total_money_correct_l738_738886


namespace smallest_int_y_prime_l738_738343

def is_prime (n : ℤ) : Prop := n > 1 ∧ ∀ m : ℤ, m > 1 ∧ m < n → ¬ (m ∣ n)

def absolute_value (x : ℤ) : ℤ := if x < 0 then -x else x

def poly_value (y : ℤ) : ℤ := absolute_value (5 * y^2 - 34 * y + 7)

theorem smallest_int_y_prime :
  ∃ y : ℤ, (is_prime (poly_value y)) ∧ (∀ z : ℤ, (is_prime (poly_value z)) → z ≥ 0) :=
begin
  use 0,
  split,
  {
    -- Check if it is prime; solution shows 7 is prime for y = 0
    sorry
  },
  {
    -- Prove there is no smaller integer having the property; solution shows checking all other smaller integers
    sorry
  }
end

end smallest_int_y_prime_l738_738343


namespace parallelogram_side_length_l738_738680

theorem parallelogram_side_length 
  (s : ℝ) -- s is a real number
  (area : ℝ) 
  (h : ℝ)
  (H_side1 : 3 * s) -- Length of the first side
  (H_side2 : s) -- Length of the second side
  (H_angle : 30) -- Angle between the sides in degrees
  (H_area : s * s * (√3) = 9 * √3) -- Area condition using heights derived from 30-60-90 triangle properties
  : s = 3 := 
by sorry -- Proof that s = 3 with the given conditions (to be completed)

end parallelogram_side_length_l738_738680


namespace min_value_of_b_minus_2c_plus_1_over_a_l738_738784

theorem min_value_of_b_minus_2c_plus_1_over_a
  (a b c : ℝ)
  (h₁ : (a ≠ 0))
  (h₂ : ∀ x, -1 < x ∧ x < 3 → ax^2 + bx + c < 0) :
  b - 2 * c + (1 / a) = 4 :=
sorry

end min_value_of_b_minus_2c_plus_1_over_a_l738_738784


namespace sam_can_order_193_sandwiches_l738_738185

-- Define the types of bread, meat, and cheese
constant breads : Fin 5
constant meats : Fin 7
constant cheeses : Fin 6

-- Define the condition predicates
def prohibits_combination (b : Fin 5) (m : Fin 7) (c : Fin 6) : Prop :=
  (m = 1 ∧ c = 2) ∨ (b = 2 ∧ m = 3) ∨ (m = 1 ∧ b = 2)

-- Prove the number of valid sandwiches avoiding prohibited combinations
def valid_sandwiches_count : Nat :=
  5 * 7 * 6 - 17

theorem sam_can_order_193_sandwiches : valid_sandwiches_count = 193 :=
  by
    -- The proof would normally go here
    sorry

end sam_can_order_193_sandwiches_l738_738185


namespace cone_lateral_surface_area_l738_738016

theorem cone_lateral_surface_area (r h : ℝ) (hr : r = 3) (hh : h = 4) : 15 * Real.pi = Real.pi * r * (Real.sqrt (r^2 + h^2)) :=
by
  -- Prove that 15π = π * r * sqrt(r^2 + h^2) for r = 3 and h = 4
  sorry

end cone_lateral_surface_area_l738_738016


namespace sin_alpha_l738_738446

theorem sin_alpha (m : ℝ) (h : m > 0) :
  let P : ℝ × ℝ := (m, 2 * m) in
  let OP : ℝ := real.sqrt (m^2 + (2 * m)^2) in
  sin (real.atan2 (2 * m) m) = (2 * real.sqrt 5) / 5 :=
by
  sorry

end sin_alpha_l738_738446


namespace tangent_range_of_a_l738_738786

theorem tangent_range_of_a 
  (a : ℝ)
  (circle_eq : ∀ x y : ℝ, x^2 + y^2 + a * x + 2 * y + a^2 = 0)
  (A : ℝ × ℝ) 
  (A_eq : A = (1, 2)) :
  -2 * Real.sqrt 3 / 3 < a ∧ a < 2 * Real.sqrt 3 / 3 :=
by
  sorry

end tangent_range_of_a_l738_738786


namespace find_B_l738_738097

noncomputable def B_solution (A a b : ℝ) : set ℝ :=
  {B | A = π / 6 ∧ a = 1 ∧ b = sqrt 3 → B = π / 3 ∨ B = 2 * π / 3}

theorem find_B (A a b B : ℝ) (hA : A = π / 6) (ha : a = 1) (hb : b = sqrt 3) : B_solution A a b B :=
by
  have h : ∃ B, A = π / 6 ∧ a = 1 ∧ b = sqrt 3 → B = π / 3 ∨ B = 2 * π / 3 := 
  sorry
  exact h

end find_B_l738_738097


namespace probability_third_winning_l738_738391

-- Definitions based on the conditions provided
def num_tickets : ℕ := 10
def num_winning_tickets : ℕ := 3
def num_non_winning_tickets : ℕ := num_tickets - num_winning_tickets

-- Define the probability function
def probability_of_third_draw_winning : ℚ :=
  (num_non_winning_tickets / num_tickets) * 
  ((num_non_winning_tickets - 1) / (num_tickets - 1)) * 
  (num_winning_tickets / (num_tickets - 2))

-- The theorem to prove
theorem probability_third_winning : probability_of_third_draw_winning = 7 / 40 :=
  by sorry

end probability_third_winning_l738_738391


namespace equation_of_line_through_midpoint_l738_738429

-- Define the lines l1 and l2
def l1 (x y : ℝ) : Prop := 4 * x + y + 6 = 0
def l2 (x y : ℝ) : Prop := 3 * x - 5 * y - 6 = 0

-- Define the midpoint condition
def midpoint (x1 y1 x2 y2 : ℝ) (P : ℝ × ℝ) : Prop := 
  P = (0, 0) ∧ x1 + x2 = 0 ∧ y1 + y2 = 0

-- Lean theorem statement for the required proof
theorem equation_of_line_through_midpoint 
  (x1 y1 x2 y2 : ℝ)
  (h1 : l1 x1 y1)
  (h2 : l2 x2 y2)
  (P : ℝ × ℝ)
  (h_mid : midpoint x1 y1 x2 y2 P) :
  ∃ m : ℝ, m = 7 / 6 ∧ ∀ x : ℝ, ∃ y : ℝ, y = m * x := 
  by {
    sorry
  }

end equation_of_line_through_midpoint_l738_738429


namespace cover_2x9_with_dominoes_l738_738950

def num_ways_cover_2xn (n : ℕ) : ℕ :=
  match n with
  | 0     => 0
  | 1     => 1
  | 2     => 2
  | nat.succ (nat.succ n') => num_ways_cover_2xn (n') + num_ways_cover_2xn (n' + 1)

theorem cover_2x9_with_dominoes : num_ways_cover_2xn 9 = 55 := 
  by
    sorry

end cover_2x9_with_dominoes_l738_738950


namespace sum_difference_even_odd_l738_738254

-- Define the sum of even integers from 2 to 100
def sum_even (n : ℕ) : ℕ := (n / 2) * (2 + n)

-- Define the sum of odd integers from 1 to 99
def sum_odd (n : ℕ) : ℕ := (n / 2) * (1 + n)

theorem sum_difference_even_odd:
  let a := sum_even 100
  let b := sum_odd 99
  a - b = 50 :=
by
  sorry

end sum_difference_even_odd_l738_738254


namespace parallel_lines_a_value_l738_738411

theorem parallel_lines_a_value (a : ℝ) 
  (h1 : ∀ x y : ℝ, x + a * y - 1 = 0 → x = a * (-4 * y - 2)) 
  : a = 2 :=
sorry

end parallel_lines_a_value_l738_738411


namespace find_yellow_marbles_l738_738377

-- Definitions for the variables and conditions
def total_marbles : ℕ := 50
def white_marbles : ℕ := total_marbles / 2
def red_marbles : ℕ := 7
def yellow_and_green_marbles : ℕ := total_marbles - (white_marbles + red_marbles)
def green_marble_ratio : ℝ := 0.5

-- Variables representing the number of yellow and green marbles
def yellow_marbles : ℝ
def green_marbles : ℝ := green_marble_ratio * yellow_marbles

-- The final proof to be stated
theorem find_yellow_marbles : yellow_marbles + green_marbles = yellow_and_green_marbles → yellow_marbles = 12 :=
by
  unfold yellow_marbles green_marbles yellow_and_green_marbles green_marble_ratio total_marbles white_marbles red_marbles
  sorry

end find_yellow_marbles_l738_738377


namespace orthogonal_unit_vector_l738_738350

noncomputable def unit_vector (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let mag := Real.sqrt (x^2 + y^2 + z^2)
  (x / mag, y / mag, z / mag)

def vector1 : ℝ × ℝ × ℝ := (2, 1, 1)
def vector2 : ℝ × ℝ × ℝ := (1, 1, 3)

def cross_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v1.2.1 * v2.2 - v1.2 * v2.2.1, 
   v1.2 * v2.1 - v1.1 * v2.2, 
   v1.1 * v2.2.1 - v1.2.1 * v2.1)

def unit_orthogonal_vector : ℝ × ℝ × ℝ :=
  unit_vector (cross_product vector1 vector2).1 (cross_product vector1 vector2).2.1 (cross_product vector1 vector2).2.2

#eval unit_orthogonal_vector -- This will compute unit_orthogonal_vector (2/√30, -5/√30, 1/√30)

theorem orthogonal_unit_vector (v1 v2 : ℝ × ℝ × ℝ) :
  (cross_product v1 v2).1 / Real.sqrt ((cross_product v1 v2).1^2 + (cross_product v1 v2).2.1^2 + (cross_product v1 v2).2.2^2) = 2 / Real.sqrt 30 ∧
  (cross_product v1 v2).2.1 / Real.sqrt ((cross_product v1 v2).1^2 + (cross_product v1 v2).2.1^2 + (cross_product v1 v2).2.2^2) = -5 / Real.sqrt 30 ∧
  (cross_product v1 v2).2.2 / Real.sqrt ((cross_product v1 v2).1^2 + (cross_product v1 v2).2.1^2 + (cross_product v1 v2).2.2^2) = 1 / Real.sqrt 30 :=
by 
  sorry -- proof omitted

end orthogonal_unit_vector_l738_738350


namespace graphs_symmetric_l738_738507

variable (f : ℝ → ℝ)

theorem graphs_symmetric (f : ℝ → ℝ) : symmetric_with_respect_to (λ x y => y = f(x - 1)) (λ x y => y = f(1 - x)) (λ x => x = 1) :=
sorry

end graphs_symmetric_l738_738507


namespace largest_of_three_numbers_l738_738962

theorem largest_of_three_numbers :
  (0 < 2^(-3) ∧ 2^(-3) < 1) ∧
  (1 < 3^(1/2) ∧ 3^(1/2) < 2) ∧
  (log 2 5 > 2) →
  log 2 5 > 3^(1/2) ∧ log 2 5 > 2^(-3) :=
by
  intros h
  cases h with h1 h2h3
  cases h2h3 with h2 h3
  cases h1 with h1_left h1_right
  cases h2 with h2_left h2_right
  sorry

end largest_of_three_numbers_l738_738962


namespace symmetric_difference_card_l738_738253

variable (x y : Finset ℤ)
variable (h1 : x.card = 16)
variable (h2 : y.card = 18)
variable (h3 : (x ∩ y).card = 6)

theorem symmetric_difference_card :
  (x \ y ∪ y \ x).card = 22 := by sorry

end symmetric_difference_card_l738_738253


namespace concert_people_count_l738_738966

variable {W M : ℕ}

theorem concert_people_count (h1 : W * 2 = M) (h2 : (W - 12) * 3 = M - 29) : W + M = 21 := 
sorry

end concert_people_count_l738_738966


namespace problem1_problem2_problem3_l738_738005

-- Problem 1: Existence of q and r
theorem problem1 : ∃ q r : ℤ, 2011 = 91 * q + r ∧ 0 ≤ r ∧ r < 91 := 
  exists.intro 22 (exists.intro 9 (by
    simp [int.zero_le, int.one_mul, lt_add_one]
    norm_num
  ))

-- Problem 2: Possible values of t for which B is not a harmonic set
theorem problem2 (t : ℕ) (h : B ⊆ A) (hB : ¬is_harmonic_set B): 
  t ∈ {13, 17, 19, 23} := 
  sorry

-- Problem 3: Maximum value of m for 12-element harmonic subset
theorem problem3 (m : ℕ) (h : m ∈ A) (hC : ∀ C : set ℕ, C ⊆ A ∧ C.card = 12 ∧ m ∈ C → is_harmonic_set C): 
  m = 7 := 
  sorry

end problem1_problem2_problem3_l738_738005


namespace distance_from_point_to_circle_center_l738_738421

section
variable (P : ℝ × ℝ)
variable (C : ℝ × ℝ)

/-- Given:
 - a line l defined by y = 2x 
 - a circle C with center (8,1) and radius sqrt(2)
 - point P lies on the line y = 2x
 - tangents from point P to the circle are symmetric about line l

We want to prove the distance from point P to the center of circle C is 3 * sqrt 5.
--/
theorem distance_from_point_to_circle_center
  (hP_on_line : P.2 = 2 * P.1)
  (hC_center : C = (8, 1))
  (hC_eqn : ∀ (x y : ℝ), (x - 8)^2 + (y - 1)^2 = 2 → (x, y) = C)
  (h_symm_tangents : symmetric_about_line l P C) : 
  dist P C = 3 * real.sqrt 5 := sorry
end

end distance_from_point_to_circle_center_l738_738421


namespace sequence_term_2017_l738_738389

theorem sequence_term_2017 (a : ℕ → ℚ)
  (h₁ : ∀ n, (n + 1) * a (n + 1) = a n + n)
  (h₂ : a 1 = 2) :
  a 2017 = 1 + 1 / 2017! :=
by
  sorry

end sequence_term_2017_l738_738389


namespace two_T_n_minus_three_S_n_eq_neg_one_l738_738548

theorem two_T_n_minus_three_S_n_eq_neg_one (n : ℕ) (h₁ : n ≥ 1) :
  let T_n := (3^n - 1) / 2 in
  let S_n := 3^(n-1) in
  2 * T_n - 3 * S_n = -1 :=
by
  sorry

end two_T_n_minus_three_S_n_eq_neg_one_l738_738548


namespace part_a_part_b_part_c_l738_738932

/-- Part (a) -/
theorem part_a (a b : ℝ) (h1 : a = 16) (h2 : b = 4) :
  2 * real.sqrt (a * b) = 16 :=
by calc
  2 * real.sqrt (a * b)
    = 2 * real.sqrt (16 * 4) : by rw [h1, h2]
... = 2 * 8                 : by norm_num
... = 16                    : by norm_num

/-- Part (b) -/
theorem part_b (a b c : ℝ) (h1 : a = 16) (h2 : b = 4) (h3 : a ≠ 0) (h4 : b ≠ 0) :
  real.sqrt (b * c) + real.sqrt (a * c) = 8 → c = 16/9 :=
by intros h
   let x := real.sqrt c,
   have hx : x = 4 / 3,
   { sorry },
   rw [hx, real.sqr_eq, mul_div_assoc, of_div],
   field_simp,
   exact or.inl zero_lt_four

/-- Part (c) -/
theorem part_c (a b c : ℕ) (h1 : a = 36) (h2 : b = 9) (h3 : c = 4) :
  ∃ n : ℕ, ∀ (a b c : ℕ), n = c → integer (a) → integer (b) → integer (c) :=
by use 4
   intros a b c hn ha hb hc
   rw [hn, int.nat_cast_eq_coe_nat]
   exact or.inl zero_lt_four

end part_a_part_b_part_c_l738_738932


namespace solution_set_l738_738585

variable (x : ℝ)

def condition_1 : Prop := 2 * x - 4 ≤ 0
def condition_2 : Prop := -x + 1 < 0

theorem solution_set : (condition_1 x ∧ condition_2 x) ↔ (1 < x ∧ x ≤ 2) := by
sorry

end solution_set_l738_738585


namespace coin_stack_arrangements_l738_738164

-- Define conditions as given in the problem

def coins := Σ (c : Fin 10), if c.val < 5 then Unit else Unit -- Representing the 10 coins
def valid_orientation (c1 c2 : coins) : Prop := -- Validates no two adjacent coins face to face
  sorry

def valid_color_arrangement (coins : Fin 10 → coins) : Prop := -- Validates no three consecutive coins of the same color
  sorry

def valid_stack (coins : Fin 10 → coins) : Prop := -- Valid stack with both conditions
  ∀ i : Fin 10, valid_orientation coins[i] coins[i+1] ∧ valid_color_arrangement coins

theorem coin_stack_arrangements : 
  (Σ c : coins, valid_stack c).2 = 44 := 
sorry

end coin_stack_arrangements_l738_738164


namespace min_iterations_at_least_5_l738_738189

noncomputable def sequence : Type := Vector ℕ 100

def is_correct_order (seq : sequence) (selected : Finset ℕ) (reordered : Finset ℕ) : Prop :=
  ∀ (a b : ℕ), a ∈ selected → b ∈ selected → a < b → nth seq a < nth seq b

def min_iterations (seq : sequence) : ℕ :=
  ∀ n, n < 5 → exists (selections : Vector (Finset ℕ) n),
  ∀ (i : ℕ) (h : i < n), is_correct_order seq (selections.nth h) (reorder seq (selections.nth h))

theorem min_iterations_at_least_5 (seq : sequence) : min_iterations seq ≥ 5 := sorry

end min_iterations_at_least_5_l738_738189


namespace triangle_ABC_is_isosceles_right_l738_738838

variables {A B C : Type}

noncomputable def is_isosceles_right_triangle
  (AB AC : vector A) (angle : A → ℝ)
  (sin cos : (A → ℝ) → (A → ℝ))
  (h1 : |AB + AC| = |AB - AC|)
  (h2 : sin (angle A) = 2 * (sin (angle B)) * (cos (angle C))) : Prop :=
isosceles_triangle AB AC ∧ right_triangle AB AC

theorem triangle_ABC_is_isosceles_right (AB AC : vector ℝ) :
  (|AB + AC| = |AB - AC| ∧ sin (angle A) = 2 * sin (angle B) * cos (angle C)) →
  is_isosceles_right_triangle AB AC angle sin cos := sorry

end triangle_ABC_is_isosceles_right_l738_738838


namespace min_handshakes_l738_738279

theorem min_handshakes 
  (people : ℕ) 
  (handshakes_per_person : ℕ) 
  (total_people : people = 30) 
  (handshakes_rule : handshakes_per_person = 3) 
  (unique_handshakes : people * handshakes_per_person % 2 = 0) 
  (multiple_people : people > 0):
  (people * handshakes_per_person / 2) = 45 :=
by
  sorry

end min_handshakes_l738_738279


namespace cannot_use_square_difference_formula_l738_738614

theorem cannot_use_square_difference_formula (x y : ℝ) :
  ¬ ∃ a b : ℝ, (2 * x + 3 * y) * (-3 * y - 2 * x) = (a + b) * (a - b) :=
sorry

end cannot_use_square_difference_formula_l738_738614


namespace champion_is_certainly_chinese_l738_738453

theorem champion_is_certainly_chinese 
  (PlayerA : String) 
  (PlayerB : String)
  (is_chinese : String → Prop)
  (h1 : is_chinese PlayerA) 
  (h2 : is_chinese PlayerB) : 
  ∃ player : String, is_chinese player :=
by 
  exists PlayerA
  exact h1
-- Note: Proof is provided only to ensure clarity, but if omitted according to the request, the following will work.
-- sorry

end champion_is_certainly_chinese_l738_738453


namespace cos_value_given_sin_l738_738399

theorem cos_value_given_sin:
  ∀ (α : ℝ), sin (π / 3 - α) = 1 / 3 → cos (5 * π / 6 - α) = -1 / 3 :=
by
  intros α h
  sorry

end cos_value_given_sin_l738_738399


namespace correct_option_l738_738011

variable {a b : ℝ}
variable (h1 : a > b) (h2 : b > 0)

theorem correct_option :
  (a + b) / 2 > real.sqrt (a * b) ∧ real.sqrt (a * b) > (2 * a * b) / (a + b) :=
sorry

end correct_option_l738_738011


namespace two_digit_multiples_of_4_and_9_l738_738080

theorem two_digit_multiples_of_4_and_9 :
  ∃ (count : ℕ), 
    (∀ (n : ℕ), (10 ≤ n ∧ n ≤ 99) → (n % 4 = 0 ∧ n % 9 = 0) → (n = 36 ∨ n = 72)) ∧ count = 2 :=
by
  sorry

end two_digit_multiples_of_4_and_9_l738_738080


namespace dot_cross_equal_one_l738_738493

variables (a b c : ℝ^3)
variables (ha : ‖a‖ = 1) (hb : ‖b‖ = 1)
variables (h1 : a × b + a = c)
variables (h2 : c × a = b)

theorem dot_cross_equal_one : a ⋅ (b × c) = 1 :=
by sorry

end dot_cross_equal_one_l738_738493


namespace shaded_region_area_l738_738188

theorem shaded_region_area (r_s r_l chord_AB : ℝ) (hs : r_s = 40) (hl : r_l = 60) (hc : chord_AB = 100) :
    chord_AB / 2 = 50 →
    60^2 - (chord_AB / 2)^2 = r_s^2 →
    (π * r_l^2) - (π * r_s^2) = 2500 * π :=
by
  intros h1 h2
  sorry

end shaded_region_area_l738_738188


namespace find_z_l738_738854

-- Definition of the infinite nested square root operation
def bowtie (a b : ℝ) : ℝ := a + real.sqrt (b + real.sqrt (b + real.sqrt (b + ...)))

-- Theorem stating the question and conditions leading to the correct answer
theorem find_z (z : ℝ) : bowtie 7 z = 15 → z = 56 :=
by
  sorry

end find_z_l738_738854


namespace inverse_sum_mod_13_l738_738856

theorem inverse_sum_mod_13 :
  ∀ (a c d b : ℕ),
    (4 * a ≡ 1 [MOD 13]) →
    (6 * c ≡ 1 [MOD 13]) →
    (9 * d ≡ 1 [MOD 13]) →
    (b * (a + c + d) ≡ 1 [MOD 13]) →
    b ≡ 6 [MOD 13] :=
by {
  intros,
  sorry
}

end inverse_sum_mod_13_l738_738856


namespace sqrt_complement_to_9_l738_738376

theorem sqrt_complement_to_9 (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  ¬ (∃ a : ℤ, ∀ k, let dm := nat.digits 10 (nat.floor (real.sqrt m * 10^k))
                     in let dn := nat.digits 10 (nat.floor (real.sqrt n * 10^k))
                     in ∃ i, i < min dm.length dn.length ∧ (dm.nth i).getOrElse 0 + (dn.nth i).getOrElse 0 = 9) :=
sorry

end sqrt_complement_to_9_l738_738376


namespace infinite_natural_with_square_digit_sum_and_no_zero_digits_l738_738532

theorem infinite_natural_with_square_digit_sum_and_no_zero_digits :
  ∃ᶠ n : ℕ, ∃ k : ℕ, n = k^2 ∧ (∃ m : ℕ, S n = m^2) ∧ (∀ d ∈ n.digits 10, d ≠ 0) :=
by sorry

end infinite_natural_with_square_digit_sum_and_no_zero_digits_l738_738532


namespace quadratic_points_order_l738_738381

theorem quadratic_points_order {m y1 y2 y3 : ℝ} (h1 : m < -2)
  (h2 : y1 = (m-1)^2 + 2*(m-1))
  (h3 : y2 = m^2 + 2*m)
  (h4 : y3 = (m+1)^2 + 2*(m+1)) :
  y3 < y2 ∧ y2 < y1 :=
begin
  sorry
end

end quadratic_points_order_l738_738381


namespace Toby_change_l738_738942

def change (orders_cost per_person total_cost given_amount : ℝ) : ℝ :=
  given_amount - per_person

def total_cost (cheeseburgers milkshake coke fries cookies tax : ℝ) : ℝ :=
  cheeseburgers + milkshake + coke + fries + cookies + tax

theorem Toby_change :
  let cheeseburger_cost := 3.65
  let milkshake_cost := 2.0
  let coke_cost := 1.0
  let fries_cost := 4.0
  let cookie_cost := 3 * 0.5 -- Total cost for three cookies
  let tax := 0.2
  let total := total_cost (2 * cheeseburger_cost) milkshake_cost coke_cost fries_cost cookie_cost tax
  let per_person := total / 2
  let toby_arrival := 15.0
  change total per_person total toby_arrival = 7 :=
by
  sorry

end Toby_change_l738_738942


namespace inequality_proof_l738_738763

variables (a b c : ℝ)

noncomputable def proof_inequality : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 3 → 
  (a / (a + b) + b / (b + c) + c / (c + a) ≤ 3 / (1 + Real.sqrt (a * b * c)))

theorem inequality_proof (a b c : ℝ) : proof_inequality a b c :=
begin
  sorry
end

end inequality_proof_l738_738763


namespace total_patients_in_a_year_l738_738845

-- Define conditions from the problem
def patients_per_day_first : ℕ := 20
def percent_increase_second : ℕ := 20
def working_days_per_week : ℕ := 5
def working_weeks_per_year : ℕ := 50

-- Lean statement for the problem
theorem total_patients_in_a_year (patients_per_day_first : ℕ) (percent_increase_second : ℕ) (working_days_per_week : ℕ) (working_weeks_per_year : ℕ) :
  (patients_per_day_first + ((patients_per_day_first * percent_increase_second) / 100)) * working_days_per_week * working_weeks_per_year = 11000 :=
by
  sorry

end total_patients_in_a_year_l738_738845


namespace triangle_area_side_a_cos_2C_B_angle_l738_738836

/-- Given a triangle ABC with sides opposite angles A, B, and C labeled as a, b, and c respectively.
    Given b = 2, area of triangle ABC = 3 sqrt 3, and cos(A - π/3) = 2 cos A, prove a = 2 sqrt 7. --/
theorem triangle_area_side_a (A B C : ℝ) (a b c : ℝ) (h1 : b = 2) (h2 : 3 * sqrt 3 * 2 / (2 * 6 * sin A) = 1) (h3 : cos (A - π / 3) = 2 * cos A) : 
  a = 2 * sqrt 7 :=
sorry

/-- Given cos(2C) = 1 - a^2 / 6b^2,
    find B = π / 12 or B = 7π / 12. --/
theorem cos_2C_B_angle (A B C : ℝ) (a b : ℝ) (h1 : cos (2 * C) = 1 - (a ^ 2) / (6 * b ^ 2)) : 
  B = π / 12 ∨ B = 7 * π / 12 :=
sorry

end triangle_area_side_a_cos_2C_B_angle_l738_738836


namespace polyhedron_value_l738_738984

variable (T P H : ℕ)

-- Given conditions
def polyhedron_conditions : Prop :=
  T + P + H = 42 ∧ 
  ∃ E V, E = (3 * T + 5 * P + 6 * H) / 2 ∧ 
           V = (3 * T + 2 * P + H) / 6 ∧ 
           V - E + 42 = 2

-- Question to prove
theorem polyhedron_value (h : polyhedron_conditions T P H) :
  100 * H + 10 * P + T + (3 * T + 2 * P + H) / 6 = 714 :=
by {
  obtain ⟨E, V, hE, hV, _⟩ := h,
  sorry
}

end polyhedron_value_l738_738984


namespace minimum_area_of_triangle_l738_738494

theorem minimum_area_of_triangle (A B C Y J1 J2 : Type*)
  [inner A B C]
  (side_AB : nat)
  (side_BC : nat)
  (side_AC : nat)
  (on_segment_BC : between Y B C)
  (incenter1 : J1 = incenter of ABY)
  (incenter2 : J2 = incenter of ACY)
  (angle_constant : ∀ Y, angle J1AJ2 = constant)
  : 
  ∃ area_minimum, area_minimum = 4.5 :=
by
  sorry

end minimum_area_of_triangle_l738_738494


namespace angle_in_second_quadrant_l738_738085

/-- If α is an angle in the first quadrant, then π - α is an angle in the second quadrant -/
theorem angle_in_second_quadrant (α : Real) (h : 0 < α ∧ α < π / 2) : π - α > π / 2 ∧ π - α < π :=
by
  sorry

end angle_in_second_quadrant_l738_738085


namespace Janet_horses_l738_738126

theorem Janet_horses (acres : ℕ) (gallons_per_acre : ℕ) (spread_acres_per_day : ℕ) (total_days : ℕ)
  (gallons_per_day_per_horse : ℕ) (total_gallons_needed : ℕ) (total_gallons_spread : ℕ) (horses : ℕ) :
  acres = 20 ->
  gallons_per_acre = 400 ->
  spread_acres_per_day = 4 ->
  total_days = 25 ->
  gallons_per_day_per_horse = 5 ->
  total_gallons_needed = acres * gallons_per_acre ->
  total_gallons_spread = spread_acres_per_day * gallons_per_acre * total_days ->
  horses = total_gallons_needed / (gallons_per_day_per_horse * total_days) ->
  horses = 64 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end Janet_horses_l738_738126


namespace compare_log_exp_l738_738762

theorem compare_log_exp: 
  let a := Real.logb 0.7 3 
  let b := 3^0.7 
  let c := 0.7^3 
  in a < c ∧ c < b :=
by
  let a := Real.logb 0.7 3
  let b := 3^0.7
  let c := 0.7^3
  sorry

end compare_log_exp_l738_738762


namespace right_angle_triangle_similarity_l738_738615

theorem right_angle_triangle_similarity {α β γ δ : ℝ} 
  (h1 : α + β = 90) (h2 : γ + δ = 90) (h3 : α = γ) : 
  (α, β, 90).angle = (γ, δ, 90).angle :=
by
  sorry

end right_angle_triangle_similarity_l738_738615


namespace min_liars_circle_l738_738543

-- Define the problem conditions
inductive Person : Type
| p1 | p2 | p3 | p4 | p5 | p6

open Person

def opposite : Person → Person
| p1 => p4
| p2 => p5
| p3 => p6
| p4 => p1
| p5 => p2
| p6 => p3

inductive Role
| Knight -- Always tells the truth
| Liar   -- Always lies

noncomputable def statement : Person → (Person → Role) → Prop :=
λ p r, (r p = Role.Knight → (r (p.next) = Role.Liar ∨ r (opposite p) = Role.Liar)) ∧
       (r p = Role.Liar → (r (p.next) ≠ Role.Liar ∧ r (opposite p) ≠ Role.Liar))

def minimal_liars (r : Person → Role) : Prop :=
  (∀ p, statement p r) → (∑ p, if r p = Role.Liar then 1 else 0) = 2

-- Lean 4 statement to be proven
theorem min_liars_circle :
  ∃ r : Person → Role, minimal_liars r :=
sorry

end min_liars_circle_l738_738543


namespace closest_integer_area_for_centroid_path_is_113_l738_738006

-- Let's define the conditions first
def diameter : ℝ := 36
def radius : ℝ := diameter / 2
def centroid_radius : ℝ := radius / 3
def traced_circle_area : ℝ := Real.pi * centroid_radius^2
def closest_integer_area : ℝ := Real.floor (traced_circle_area + 0.5)

-- Now, let's state the theorem
theorem closest_integer_area_for_centroid_path_is_113 :
  closest_integer_area = 113 :=
sorry

end closest_integer_area_for_centroid_path_is_113_l738_738006


namespace no_solutions_to_equation_l738_738911

theorem no_solutions_to_equation : ¬∃ x : ℝ, (x ≠ 0) ∧ (x ≠ 5) ∧ ((2 * x ^ 2 - 10 * x) / (x ^ 2 - 5 * x) = x - 3) :=
by
  sorry

end no_solutions_to_equation_l738_738911


namespace ellipse_eqn_no_line_exists_range_of_n_l738_738463

noncomputable def A := (-1 : ℝ, 0 : ℝ)
noncomputable def B := (1 : ℝ, 0 : ℝ)
noncomputable def C := (-1 : ℝ, (3 / 2 : ℝ))
noncomputable def D := (0 : ℝ, 1 : ℝ)

-- Problem (I)
theorem ellipse_eqn (a b : ℝ) (ha : a > b) (hb : b > 0) :
  (a^2 = 4) ∧ (b^2 = 3) → 
  (∀ (x y : ℝ), (x, y) ≠ A → (x, y) ≠ B → (x, y) ≠ C → ((x^2 / 4) + (y^2 / 3) = 1)) :=
sorry

-- Problem (II)
theorem no_line_exists (k m : ℝ) (hk : k ≠ 0) :
  ¬ (∃ (M N : ℝ × ℝ), 
      M ≠ N ∧ 
      (M.1^2 / 4 + M.2^2 / 3 = 1) ∧ 
      (N.1^2 / 4 + N.2^2 / 3 = 1) ∧ 
      ((D.1 - M.1 + D.2 - M.2) * (N.1 - M.1 + N.2 - M.2) = 0)) :=
sorry

-- Problem (III)
theorem range_of_n : 
  ∃ (n : ℝ), (0 < n ∧ n < sqrt(3) / 3) ∨ (-sqrt(3) / 3 < n ∧ n < 0) →
  (∃ (k m : ℝ), k ≠ 0 ∧ 
    ∃ (M N : ℝ × ℝ), 
      M ≠ N ∧ 
      (M.1^2 / 4 + M.2^2 / 3 = 1) ∧ 
      (N.1^2 / 4 + N.2^2 / 3 = 1) ∧ 
      ((n - P.1 + 0 - P.2) * (N.1 - M.1 + N.2 - M.2) = 0)) :=
sorry

end ellipse_eqn_no_line_exists_range_of_n_l738_738463


namespace intersection_point_is_P_distance_between_P_and_Q_is_4sqrt3_l738_738773

-- Define the lines and points in Lean
structure Line :=
  (x : ℝ → ℝ)
  (y : ℝ → ℝ)

def l1 : Line :=
  ⟨(λ t, 1 + t), (λ t, -5 + (Real.sqrt 3) * t)⟩

def l2 (x y : ℝ) : Prop :=
  x - y = 2 * Real.sqrt 3

def Q := (1 : ℝ, -5 : ℝ)

-- Define the points P and distances
def P : ℝ × ℝ := (1 + 2 * Real.sqrt 3, 1)

-- The first task is to prove that the intersection point of l1 and l2 is P.
theorem intersection_point_is_P : ∃ t : ℝ, 
  (l1.x t, l1.y t) = P ∧ l2 (l1.x t) (l1.y t) :=
sorry

-- The second task is to prove the distance between points P and Q
def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem distance_between_P_and_Q_is_4sqrt3 : 
  distance P Q = 4 * Real.sqrt 3 :=
sorry

end intersection_point_is_P_distance_between_P_and_Q_is_4sqrt3_l738_738773


namespace line_passing_through_M_l738_738301

-- Define the point M
def M : ℝ × ℝ := (-3, 4)

-- Define the predicate for a line equation having equal intercepts and passing through point M
def line_eq (x y : ℝ) (a b : ℝ) : Prop :=
  ∃ c : ℝ, ((a = 0 ∧ b = 0 ∧ 4 * x + 3 * y = 0) ∨ (a ≠ 0 ∧ b ≠ 0 ∧ a = b ∧ x + y = 1)) 

theorem line_passing_through_M (x y : ℝ) (a b : ℝ) (h₀ : (-3, 4) = M) (h₁ : ∃ c : ℝ, (a = 0 ∧ b = 0 ∧ 4 * x + 3 * y = 0) ∨ (a ≠ 0 ∧ b ≠ 0 ∧ a = b ∧ x + y = 1)) :
  (4 * x + 3 * y = 0) ∨ (x + y = 1) :=
by
  -- We add 'sorry' to skip the proof
  sorry

end line_passing_through_M_l738_738301


namespace initial_distance_is_190_05_l738_738684

-- Define the speeds in km/h
def criminal_speed_kmph : ℝ := 8
def policeman_speed_kmph : ℝ := 9

-- Convert speeds to km/min
def criminal_speed_kmpmin : ℝ := criminal_speed_kmph / 60
def policeman_speed_kmpmin : ℝ := policeman_speed_kmph / 60

-- Define the timespan in minutes
def time_minutes : ℝ := 3

-- Calculate the distances covered
def distance_criminal : ℝ := criminal_speed_kmpmin * time_minutes
def distance_policeman : ℝ := policeman_speed_kmpmin * time_minutes

-- Calculate the difference in distance covered in 3 minutes
def distance_difference : ℝ := distance_policeman - distance_criminal

-- Define the distance after 3 minutes
def distance_after_3_minutes : ℝ := 190

-- Define the initial distance
def initial_distance : ℝ := distance_after_3_minutes + distance_difference

-- Prove the initial distance is 190.05 km
theorem initial_distance_is_190_05 :
  initial_distance = 190.05 :=
by
  sorry

end initial_distance_is_190_05_l738_738684


namespace gain_percent_is_correct_l738_738563

variable (MP : ℝ) -- Marked Price
def CP : ℝ := 0.36 * MP -- Cost Price
def SP : ℝ := 0.80 * MP -- Selling Price
def Gain : ℝ := SP - CP -- Gain

theorem gain_percent_is_correct : (Gain / CP) * 100 = 122.22 := by
  sorry

end gain_percent_is_correct_l738_738563


namespace correct_statements_l738_738961

-- Define the conditions
def sample_population : ℕ := 50
def sample_size : ℕ := 5
def probability_selected (pop : ℕ) (sample : ℕ) : ℝ := (sample:ℝ) / (pop:ℝ)

def data_set := [1, 2, 4, 6, 7]
def average (l : List ℕ) : ℝ := l.sum / l.length

def variance_data_set : ℝ := 
  (1 - 4)^2 + (2 - 4)^2 + (4 - 4)^2 + (6 - 4)^2 + (7 - 4)^2

def variance_calculation (l : List ℕ) (avg : ℝ) : ℝ :=
  (l.map (λ x => (x - avg)^2)).sum / l.length

def data_for_percentile := [13, 27, 24, 12, 14, 30, 15, 17, 19, 23]
def sample_standard_deviation : ℝ := 8

-- Proof statement: verify the correctness of statements A and B
theorem correct_statements : 
  (probability_selected sample_population sample_size = 0.1) ∧
  (variance_calculation data_set 4 = 26 / 5) := 
by 
  sorry -- Proof can be filled in here

end correct_statements_l738_738961


namespace max_happy_students_4x4_l738_738929

def is_happy (scores : Matrix (Fin 4) (Fin 4) ℕ) (i j : Fin 4) : Prop :=
  let neighbors := [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
  neighbors.count (λ (pos : Fin 4 × Fin 4), scores pos.1 pos.2 > scores i j) ≤ 1

def max_happy_students (scores : Matrix (Fin 4) (Fin 4) ℕ) : ℕ :=
  Finset.univ.count (λ (i : Fin 4 × Fin 4), is_happy scores i.1 i.2)

theorem max_happy_students_4x4 (scores : Matrix (Fin 4) (Fin 4) ℕ) (h_unique: Function.Injective scores.toFun) :
  max_happy_students scores = 12 := sorry

end max_happy_students_4x4_l738_738929


namespace sum_of_m_values_l738_738675

theorem sum_of_m_values (a m : ℕ) (ha : a > 0) (hm : m > 0) (h : a * m = 50) : 
  ∑ (d : ℕ) in (finset.filter (λ d, d ∣ 50) (finset.range 51)), d = 93 :=
by
  sorry

end sum_of_m_values_l738_738675


namespace combined_river_length_estimate_l738_738636

def river_length_GSA := 402 
def river_error_GSA := 0.5 
def river_prob_error_GSA := 0.04 

def river_length_AWRA := 403 
def river_error_AWRA := 0.5 
def river_prob_error_AWRA := 0.04 

/-- 
Given the measurements from GSA and AWRA, 
the combined estimate of the river's length, Rio-Coralio, is 402.5 km,
and the probability of error for this combined estimate is 0.04.
-/
theorem combined_river_length_estimate :
  ∃ l : ℝ, l = 402.5 ∧ ∀ p : ℝ, (p = 0.04) :=
sorry

end combined_river_length_estimate_l738_738636


namespace correct_propositions_l738_738190

variables (a b : ℝ) (x : ℝ) (a_max : ℝ)

/-- Given propositions to analyze. -/
noncomputable def propositions :=
  ((a + b ≠ 5 → a ≠ 2 ∨ b ≠ 3) ∧
  ((¬ ∀ x : ℝ, x^2 + x - 2 > 0) ↔ ∃ x : ℝ, x^2 + x - 2 ≤ 0) ∧
  (a_max = 2 ∧ ∀ x > 0, x + 1/x ≥ a_max))

/-- The main theorem stating which propositions are correct -/
theorem correct_propositions (h1 : a + b ≠ 5 → a ≠ 2 ∨ b ≠ 3)
                            (h2 : (¬ ∀ x : ℝ, x^2 + x - 2 > 0) ↔ ∃ x : ℝ, x^2 + x - 2 ≤ 0)
                            (h3 : a_max = 2 ∧ ∀ x > 0, x + 1/x ≥ a_max) :
  propositions a b a_max :=
by
  sorry

end correct_propositions_l738_738190


namespace handshakes_minimum_l738_738268

/-- Given 30 people and each person shakes hands with exactly three others,
    the minimum possible number of handshakes is 45. -/
theorem handshakes_minimum (n k : ℕ) (h_n : n = 30) (h_k : k = 3) :
  (n * k) / 2 = 45 :=
by
  sorry

end handshakes_minimum_l738_738268


namespace assertion1_assertion2_l738_738026

noncomputable def A : Set ℝ := {x | 1 < x ∧ x < 3}

noncomputable def f (x : ℝ) : ℝ := (1 / Real.sqrt (5 - x)) + Real.log x

noncomputable def B : Set ℝ := {x | 0 < x ∧ x < 5}

-- Set C depending on parameter m
def C (m : ℝ) : Set ℝ := {x | 2 * m - 1 < x ∧ x < m}

-- Assertion 1: Prove B and (R \ A) ∩ B
theorem assertion1 : B = {x | 0 < x ∧ x < 5} ∧ (set.univ \ A) ∩ B = {x | (0 < x ∧ x ≤ 1) ∨ (3 ≤ x ∧ x < 5)} :=
by
  -- Proof omitted
  sorry

-- Assertion 2: A ∩ C = C and the range of m is [1, +∞)
theorem assertion2 : ∀ m, (A ∩ C m = C m) → (1 ≤ m) :=
by
  -- Proof omitted
  sorry

end assertion1_assertion2_l738_738026


namespace price_of_straight_paddle_price_of_horizontal_paddle_cost_effective_solution_l738_738289

-- Definitions for the constants and variables
def price_per_ball : ℕ := 2
def ball_count_per_pair : ℕ := 10

def total_pairs (straight_pairs horizontal_pairs : ℕ) : ℕ := straight_pairs + horizontal_pairs
def total_cost (straight_price horizontal_price balls_price straight_pairs horizontal_pairs : ℕ) : ℕ := 
  (straight_price + balls_price) * straight_pairs + (horizontal_price + balls_price) * horizontal_pairs

-- Given the problem conditions as hypotheses
axiom condition_1 : total_cost 20 15 (ball_count_per_pair * price_per_ball) 9000
axiom condition_2 : (10 * 260 + (ball_count_per_pair * price_per_ball)) = (5 * 220 + (ball_count_per_pair * price_per_ball)) + 1600
axiom condition_3 : ∀ m h, m + h = 40 → m ≤ 3 * h → 
  (220 + ball_count_per_pair * price_per_ball) * m + 
  (260 + ball_count_per_pair * price_per_ball) * h = 10000

-- Proof tasks as Lean theorems
theorem price_of_straight_paddle : (x : ℤ) → x = 220 :=
sorry

theorem price_of_horizontal_paddle : (y : ℤ) → y = 260 :=
sorry

theorem cost_effective_solution : ∃ m h, m = 30 ∧ h = 10 ∧ 
  (total_pairs m h = 40 ∧ m ≤ 3 * h → 
  total_cost 220 260 (ball_count_per_pair * price_per_ball) m h = 10000) :=
sorry

end price_of_straight_paddle_price_of_horizontal_paddle_cost_effective_solution_l738_738289


namespace intersection_complement_eq_l738_738796

open Set

def U := {1, 2, 3, 4, 5, 6}
def M := {1, 2}
def N := {2, 3, 4}

theorem intersection_complement_eq :
  M ∩ (U \ N) = {1} :=
by
  sorry

end intersection_complement_eq_l738_738796


namespace not_closed_under_subtraction_l738_738148

def v : Set ℕ := {1, 4, 9, 16, 25}

theorem not_closed_under_subtraction :
  ∃ a b ∈ v, a ≠ b ∧ (a - b) ∉ v := by
  sorry

end not_closed_under_subtraction_l738_738148


namespace focus_of_curve_is_4_0_l738_738357

noncomputable def is_focus (p : ℝ × ℝ) (curve : ℝ × ℝ → Prop) : Prop :=
  ∃ c : ℝ, ∀ x y : ℝ, curve (x, y) ↔ (y^2 = -16 * c * (x - 4))

def curve (p : ℝ × ℝ) : Prop := p.2^2 = -16 * p.1 + 64

theorem focus_of_curve_is_4_0 : is_focus (4, 0) curve :=
by
sorry

end focus_of_curve_is_4_0_l738_738357


namespace airplane_altitude_l738_738698

  theorem airplane_altitude (distance_ab : ℝ) (angle_alice : ℝ) (angle_bob : ℝ) (altitude : ℝ) 
    (h_ab : distance_ab = 15) (h_angle_alice : angle_alice = 25) (h_angle_bob : angle_bob = 45) : 
    altitude ≈ 3.8 := 
  by
    sorry
  
end airplane_altitude_l738_738698


namespace find_a_l738_738406

-- Define the conditions
def parabola_equation (a : ℝ) (x : ℝ) : ℝ := a * x^2
def axis_of_symmetry : ℝ := -2

-- The main theorem: proving the value of a
theorem find_a (a : ℝ) : (axis_of_symmetry = - (1 / (4 * a))) → a = 1/8 :=
by
  intro h
  sorry

end find_a_l738_738406


namespace area_triangle_min_distance_point_on_ellipse_l738_738787

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := (x^2 / 3) + y^2 = 1

-- Define the foci points F1 and F2
def F1 : ℝ × ℝ := (-sqrt(2), 0)
def F2 : ℝ × ℝ := (sqrt(2), 0)

-- Define the angle condition
def angle_F1_P_F2 (P : ℝ × ℝ) : Prop :=
  let (Px, Py) := P
  -- Calculate vectors PF1 and PF2
  let PF1 := (Px + sqrt(2), Py)
  let PF2 := (Px - sqrt(2), Py)
  -- Calculate the cosine of the angle using dot product and magnitudes
  let dot_prod := PF1.1 * PF2.1 + PF1.2 * PF2.2
  let mag_PF1 := Math.sqrt (PF1.1 ^ 2 + PF1.2 ^ 2)
  let mag_PF2 := Math.sqrt (PF2.1 ^ 2 + PF2.2 ^ 2)
  let cos_angle := dot_prod / (mag_PF1 * mag_PF2)
  cos_angle = -1 / 2

-- Define the line equation 
def line (P : ℝ × ℝ) : ℝ := (P.1 + P.2 + 4)

-- Define the conditions for the given math problem
def conditions (P : ℝ × ℝ) : Prop := 
  ellipse P.1 P.2 ∧ angle_F1_P_F2 P

-- Statement for Part (1): Area of triangle F1PF2 is sqrt(3)
theorem area_triangle (P : ℝ × ℝ) (hcond : conditions P) : 
  let dist_PF1 := Math.sqrt ((P.1 + sqrt(2))^2 + P.2^2)
  let dist_PF2 := Math.sqrt ((P.1 - sqrt(2))^2 + P.2^2)
  let area := (1/2) * dist_PF1 * dist_PF2 * Math.sin (2 * Math.pi / 3)
  area = Math.sqrt 3 :=
sorry

-- Statement for Part (2): Minimum distance from point P to line
theorem min_distance_point_on_ellipse :
  ∃ P : ℝ × ℝ, conditions P ∧ line P = -2 ∧ 
  let min_dist := (4 - 2) / Math.sqrt 2
  min_dist = Math.sqrt 2 :=
sorry

end area_triangle_min_distance_point_on_ellipse_l738_738787


namespace finite_squares_for_an2_plus_b_l738_738131

theorem finite_squares_for_an2_plus_b 
  (a b : ℕ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) :
  ∃ N : ℕ, ∀ n : ℕ, n > N → ¬(∃ x k : ℤ, an^2 + b = x^2 ∧ a(n+1)^2 + b = (x + k)^2) :=
sorry

end finite_squares_for_an2_plus_b_l738_738131


namespace number_of_possible_arrangements_l738_738766

theorem number_of_possible_arrangements
  (n : ℕ) (h_n : n = 48) :
  {p : ℕ × ℕ // p.1 * p.2 = n ∧ p.1 ≥ 3 ∧ p.2 ≥ 3}.size = 6 :=
by
  sorry

end number_of_possible_arrangements_l738_738766


namespace weighted_average_revenue_difference_is_57_80_l738_738553

open Nat

noncomputable def weighted_average_revenue_diff : ℝ :=
  let num_A_jerseys := 50
  let num_B_jerseys := 35
  let num_C_jerseys := 25
  let num_X_tshirts := 80
  let num_Y_tshirts := 97
  
  let revenue_per_A_jersey := 180
  let revenue_per_B_jersey := 210
  let revenue_per_C_jersey := 220
  let revenue_per_X_tshirt := 240
  let revenue_per_Y_tshirt := 270

  let total_A_revenue := num_A_jerseys * revenue_per_A_jersey
  let total_B_revenue := num_B_jerseys * revenue_per_B_jersey
  let total_C_revenue := num_C_jerseys * revenue_per_C_jersey
  let total_jersey_revenue := total_A_revenue + total_B_revenue + total_C_revenue
  let total_jerseys_sold := num_A_jerseys + num_B_jerseys + num_C_jerseys
  let avg_jersey_revenue := total_jersey_revenue.toReal / total_jerseys_sold.toReal

  let total_X_revenue := num_X_tshirts * revenue_per_X_tshirt
  let total_Y_revenue := num_Y_tshirts * revenue_per_Y_tshirt
  let total_tshirt_revenue := total_X_revenue + total_Y_revenue
  let total_tshirts_sold := num_X_tshirts + num_Y_tshirts
  let avg_tshirt_revenue := total_tshirt_revenue.toReal / total_tshirts_sold.toReal

  avg_tshirt_revenue - avg_jersey_revenue
  
theorem weighted_average_revenue_difference_is_57_80 :
  weighted_average_revenue_diff = 57.8 :=
sorry

end weighted_average_revenue_difference_is_57_80_l738_738553


namespace sum_of_inverses_mod_17_l738_738246

theorem sum_of_inverses_mod_17 :
  (3⁻¹ + 3⁻² + 3⁻³ + 3⁻⁴ + 3⁻⁵ + 3⁻⁶) % 17 = 9 := sorry

end sum_of_inverses_mod_17_l738_738246


namespace toby_change_l738_738938

theorem toby_change :
  let cheeseburger_cost := 3.65
  let milkshake_cost := 2
  let coke_cost := 1
  let large_fries_cost := 4
  let cookie_cost := 0.5
  let num_cookies := 3
  let tax := 0.2
  let initial_amount := 15
  let total_cost := 2 * cheeseburger_cost + milkshake_cost + coke_cost + large_fries_cost + num_cookies * cookie_cost + tax
  let cost_per_person := total_cost / 2
  let toby_change := initial_amount - cost_per_person
  in toby_change = 7 :=
sorry

end toby_change_l738_738938


namespace ellipse_equation_proof_k_value_proof_l738_738024

variable (a b k x1 y1 x2 y2 : Real)
variable (h1 : a > b) (h2 : b > 0)
variable (h3 : a^2 - b^2 = (3/4) * a^2)
variable (A : (a, 0)) (B : (0, -b))
variable (ecc : Real) (O : (0, 0))
variable (dist : Real)

-- Given the ellipse conditions
def ellipse_eq : Prop := ∀ (x y : Real), (x^2 / a^2) + (y^2 / b^2) = 1

-- Given the distance condition
def distance_origin_to_line := ∀ (a b : Real), (abs(a * b) / (sqrt(a^2 + b^2))) = (4 * sqrt(5)) / 5

-- Proof that the ellipse equation is as stated
theorem ellipse_equation_proof (h : ellipse_eq a b)
  (h_dist : distance_origin_to_line A B) (ecc_eq : ecc = sqrt(3) / 2) : 
  (x y : Real), (x^2 / 16) + (y^2 / 4) = 1 := 
sorry

-- Line and Circle conditions
def line_circle_intersect (k : Real) := ∀ (x y : Real), y = k * x + 1 → (x^2 + (y + 2)^2 = x^2 + (y+2)^2)

-- Proof that k is as stated
theorem k_value_proof (h_line_circle_intersect : line_circle_intersect k) 
  : k = sqrt(2) / 4 ∨ k = - sqrt(2) / 4 := 
sorry

end ellipse_equation_proof_k_value_proof_l738_738024


namespace num_ways_climb_8_steps_l738_738841

noncomputable def f : ℕ → ℕ
| 0     := 1
| 1     := 1
| 2     := 2
| (n+3) := f n + f (n+1) + f (n+2)

noncomputable def g (n : ℕ) : ℕ := f (n - 3)

theorem num_ways_climb_8_steps : g 8 = 13 := by
  sorry

end num_ways_climb_8_steps_l738_738841


namespace sum_f_values_l738_738771

def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x, f (-x) = - f x
axiom eq_condition : ∀ x, f x = f (2 - x)
axiom initial_condition : f (-1) = 2

theorem sum_f_values : (∑ i in finset.range 2017, f (i + 1)) = -2 := 
by
  sorry

end sum_f_values_l738_738771


namespace locus_of_centers_of_tangent_circles_l738_738183

noncomputable def C1 (x y : ℝ) : Prop := x^2 + y^2 = 1
noncomputable def C2 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 25
noncomputable def locus (a b : ℝ) : Prop := 4 * a^2 + 4 * b^2 - 6 * a - 25 = 0

theorem locus_of_centers_of_tangent_circles :
  (∃ (a b r : ℝ), a^2 + b^2 = (r + 1)^2 ∧ (a - 3)^2 + b^2 = (5 - r)^2) →
  (∃ a b : ℝ, locus a b) :=
sorry

end locus_of_centers_of_tangent_circles_l738_738183


namespace isosceles_triangle_three_times_ce_l738_738467

/-!
# Problem statement
In the isosceles triangle \( ABC \) with \( \overline{AC} = \overline{BC} \), 
\( D \) is the foot of the altitude through \( C \) and \( M \) is 
the midpoint of segment \( CD \). The line \( BM \) intersects \( AC \) 
at \( E \). Prove that \( AC \) is three times as long as \( CE \).
-/

-- Definition of isosceles triangle and related points
variables {A B C D E M : Type} 

-- Assume necessary conditions
variables (triangle_isosceles : A = B)
variables (D_foot : true) -- Placeholder, replace with proper definition if needed
variables (M_midpoint : true) -- Placeholder, replace with proper definition if needed
variables (BM_intersects_AC : true) -- Placeholder, replace with proper definition if needed

-- Main statement to prove
theorem isosceles_triangle_three_times_ce (h1 : A = B)
    (h2 : true) (h3 : true) (h4 : true) : 
    AC = 3 * CE :=
by
  sorry

end isosceles_triangle_three_times_ce_l738_738467


namespace find_y_l738_738859

variable (a b y : ℝ)
variable (h₀ : b ≠ 0)
variable (h₁ : (3 * a)^(3 * b) = a^b * y^b)

theorem find_y : y = 27 * a^2 :=
  by sorry

end find_y_l738_738859


namespace determine_m_type_l738_738372

theorem determine_m_type (m : ℝ) :
  ((m^2 + 2*m - 8 = 0) ↔ (m = -4)) ∧
  ((m^2 - 2*m = 0) ↔ (m = 0 ∨ m = 2)) ∧
  ((m^2 - 2*m ≠ 0) ↔ (m ≠ 0 ∧ m ≠ 2)) :=
by sorry

end determine_m_type_l738_738372


namespace calculate_sum_of_squares_l738_738083

variables {a b : ℤ}
theorem calculate_sum_of_squares (h1 : (a + b)^2 = 17) (h2 : (a - b)^2 = 11) : a^2 + b^2 = 14 :=
by
  sorry

end calculate_sum_of_squares_l738_738083


namespace fx_analytical_expression_fx_monotonically_increasing_fx_range_on_interval_l738_738418

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

theorem fx_analytical_expression :
  f = λ x, 2 * Real.sin (2 * x + Real.pi / 6) := by
  sorry

theorem fx_monotonically_increasing (k : ℤ) :
  ∀ (x : ℝ), k * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 6 →
  monotone_on f (Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6)) := by
  sorry

theorem fx_range_on_interval :
  Real.Icc (Real.min (f (Real.pi / 12)) (f (Real.pi / 2)))
            (Real.max (f (Real.pi / 12)) (f (Real.pi / 2))) =
  Real.Icc (-1) 2 := by
  sorry

end fx_analytical_expression_fx_monotonically_increasing_fx_range_on_interval_l738_738418


namespace bananas_per_truck_l738_738611

theorem bananas_per_truck (total_apples total_bananas apples_per_truck : ℝ) 
  (h_total_apples: total_apples = 132.6)
  (h_apples_per_truck: apples_per_truck = 13.26)
  (h_total_bananas: total_bananas = 6.4) :
  (total_bananas / (total_apples / apples_per_truck)) = 0.64 :=
by
  sorry

end bananas_per_truck_l738_738611


namespace cube_volume_l738_738670

theorem cube_volume (SA : ℝ) (h_SA: SA = 863.9999999999997) : 
  let side := Real.sqrt (SA / 6) in
  let V := side ^ 3 in
  V = 1728 :=
by
  sorry

end cube_volume_l738_738670


namespace find_big_apples_l738_738331
noncomputable def apple_costs (n : ℕ) : Prop :=
  let small_apple_cost := 1.5
  let medium_apple_cost := 2
  let big_apple_cost := 3
  let total_payment := 45
  let small_apples := 6
  let medium_apples := 6
  let big_apples := n
  total_payment - (small_apples * small_apple_cost + medium_apples * medium_apple_cost) = big_apples * big_apple_cost

theorem find_big_apples (n : ℕ) : apple_costs n ↔ n = 8 :=
by
  sorry

end find_big_apples_l738_738331


namespace total_cost_of_roads_l738_738307

/-- A rectangular lawn with dimensions 150 m by 80 m with two roads running 
through the middle, one parallel to the length and one parallel to the breadth. 
The first road has a width of 12 m, a base cost of Rs. 4 per sq m, and an additional section 
through a hill costing 25% more for a section of length 60 m. The second road has a width 
of 8 m and a cost of Rs. 5 per sq m. Prove that the total cost for both roads is Rs. 14000. -/
theorem total_cost_of_roads :
  let lawn_length := 150
  let lawn_breadth := 80
  let road1_width := 12
  let road2_width := 8
  let road1_base_cost := 4
  let road1_hill_length := 60
  let road1_hill_cost := road1_base_cost + (road1_base_cost / 4)
  let road2_cost := 5
  let road1_length := lawn_length
  let road2_length := lawn_breadth

  let road1_area_non_hill := road1_length * road1_width
  let road1_area_hill := road1_hill_length * road1_width
  let road1_cost_non_hill := road1_area_non_hill * road1_base_cost
  let road1_cost_hill := road1_area_hill * road1_hill_cost

  let total_road1_cost := road1_cost_non_hill + road1_cost_hill

  let road2_area := road2_length * road2_width
  let road2_total_cost := road2_area * road2_cost

  let total_cost := total_road1_cost + road2_total_cost

  total_cost = 14000 := by sorry

end total_cost_of_roads_l738_738307


namespace largest_possible_last_digit_l738_738905

def is_divisible_by(n : ℕ) (m : ℕ) : Prop := ∃ k, m * k = n

def is_valid_two_digit_multiple (x y : ℕ) : Prop :=
  let num := 10 * x + y in
  is_divisible_by num 13 ∨ is_divisible_by num 17

def ends_with_valid_last_digit (first_digit : ℕ) (length : ℕ) : ℕ :=
  if first_digit = 1 ∧ length = 1001 then 8 else sorry

theorem largest_possible_last_digit :
  (∀ x y, is_valid_two_digit_multiple x y) →
  ends_with_valid_last_digit 1 1001 = 8 :=
sorry

end largest_possible_last_digit_l738_738905


namespace Toby_change_l738_738940

def change (orders_cost per_person total_cost given_amount : ℝ) : ℝ :=
  given_amount - per_person

def total_cost (cheeseburgers milkshake coke fries cookies tax : ℝ) : ℝ :=
  cheeseburgers + milkshake + coke + fries + cookies + tax

theorem Toby_change :
  let cheeseburger_cost := 3.65
  let milkshake_cost := 2.0
  let coke_cost := 1.0
  let fries_cost := 4.0
  let cookie_cost := 3 * 0.5 -- Total cost for three cookies
  let tax := 0.2
  let total := total_cost (2 * cheeseburger_cost) milkshake_cost coke_cost fries_cost cookie_cost tax
  let per_person := total / 2
  let toby_arrival := 15.0
  change total per_person total toby_arrival = 7 :=
by
  sorry

end Toby_change_l738_738940


namespace cubic_meter_to_cubic_centimeters_l738_738431

theorem cubic_meter_to_cubic_centimeters :
  (1 : ℝ) ^ 3 = (100 : ℝ) ^ 3 := by
  sorry

end cubic_meter_to_cubic_centimeters_l738_738431


namespace chord_length_min_value_l738_738386

theorem chord_length_min_value :
  let circle (x y : ℝ) := (x - 1) ^ 2 + (y - 2) ^ 2 = 25
  let line (m x y : ℝ) := (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0
  ∃ m : ℝ, ∃ A B : ℝ × ℝ,
    circle A.1 A.2 ∧ circle B.1 B.2 ∧
    line m A.1 A.2 ∧ line m B.1 B.2 ∧
    (dist A B = 4 * sqrt 5) :=
  sorry

end chord_length_min_value_l738_738386


namespace max_det_matrix_l738_738744

noncomputable def matrix : ℝ → Matrix (Fin 3) (Fin 3) ℝ :=
  λ θ, ![
    ![1, 1, 1],
    ![1, 1 + 2 * Real.sin (θ + π / 4), 1],
    ![1, 1, 1 + 2 * Real.cos (θ + π / 4)]
  ]

theorem max_det_matrix : ∃ θ, (det (matrix θ)) = 2 * sqrt 2 + 1 :=
by
  sorry

end max_det_matrix_l738_738744


namespace function_f_correct_g_min_max_l738_738020

noncomputable def f (x : ℝ) : ℝ := sin (2 * x - π / 3) - 1

def g (x : ℝ) : ℝ := sin (2 * x - 2 * π / 3) + 5

theorem function_f_correct (x : ℝ) : 
  (sin ((2 * (x + π / 6) - π / 3)) + 1) - 1 = sin (2 * x - π / 3) - 1 :=
by sorry

theorem g_min_max (x : ℝ) (h : 0 ≤ x ∧ x ≤ π / 2) :  
  (4 ≤ g x ∧ g x ≤ 5 + sqrt 3 / 2) :=
by sorry

end function_f_correct_g_min_max_l738_738020


namespace sum_first_15_terms_arithmetic_l738_738586

-- Given conditions
def S₅ : ℕ := 48
def S₁₀ : ℕ := 60

-- Proving statement
theorem sum_first_15_terms_arithmetic : ∑ k in finset.range 15, a k = 36 :=
by 
  -- sorry here indicates proof is needed, but it is omitted for now.
  sorry

end sum_first_15_terms_arithmetic_l738_738586


namespace part1_part2_l738_738072

noncomputable def set_A : set ℝ := {x | x^2 - 4 * x - 5 ≤ 0}
noncomputable def set_B : set ℝ := {x | 1 < 2^x ∧ 2^x < 4}
noncomputable def set_C (m : ℝ) : set ℝ := {x | x < m}
noncomputable def complement_R_B : set ℝ := {x | x ≥ 2 ∨ x ≤ 0}

theorem part1 : set_A ∩ complement_R_B = {x : ℝ | (-1 ≤ x ∧ x ≤ 0) ∨ (2 ≤ x ∧ x ≤ 5)} :=
by
  sorry

theorem part2 (m : ℝ) (h1 : (set_A ∩ set_C m) ≠ set_A) (h2 : (set_B ∩ set_C m) ≠ ∅) : 0 < m ∧ m ≤ 5 :=
by
  sorry

end part1_part2_l738_738072


namespace distinct_solution_difference_l738_738500

theorem distinct_solution_difference :
  ∀ p q : ℝ, 
  (∀ x : ℝ, (x = p ∨ x = q) ↔ (x ≠ 3 ∧ \frac{6 * x - 18}{x^2 + 4 * x - 21} = x + 3)) 
  ∧ p ≠ q 
  ∧ p > q 
  → p - q = 10 :=
by 
  intros p q h
  sorry

end distinct_solution_difference_l738_738500


namespace polynomial_solutions_l738_738178

theorem polynomial_solutions (k : ℝ → ℝ) :
  (∀ x, k(x) = 3*x - 1 ∨ k(x) = -3*x + 1) ↔ (∀ x, k(x)^2 = 9*x^2 - 6*x + 1) := by
  sorry

end polynomial_solutions_l738_738178


namespace remainder_of_83_div_9_l738_738877

theorem remainder_of_83_div_9 : ∃ r : ℕ, 83 = 9 * 9 + r ∧ r = 2 :=
by {
  sorry
}

end remainder_of_83_div_9_l738_738877


namespace james_total_socks_l738_738125

theorem james_total_socks :
  ∀ (red_socks_pairs : ℕ) (black_socks_ratio red_socks_ratio : ℕ),
    red_socks_pairs = 20 →
    black_socks_ratio = 2 →
    red_socks_ratio = 2 →
    let red_socks := red_socks_pairs * 2 in
    let black_socks := red_socks / black_socks_ratio in
    let red_black_combined := red_socks + black_socks in
    let white_socks := red_black_combined * red_socks_ratio in
    red_socks + black_socks + white_socks = 180 :=
by
  intros red_socks_pairs black_socks_ratio red_socks_ratio
  intro h1 h2 h3
  let red_socks := red_socks_pairs * 2
  let black_socks := red_socks / black_socks_ratio
  let red_black_combined := red_socks + black_socks
  let white_socks := red_black_combined * red_socks_ratio
  have step1 : red_socks = 40 := by rw [h1]; refl
  have step2 : black_socks = 20 := by rw [step1, h2]; refl
  have step3 : red_black_combined = 60 := by rw [step1, step2]; refl
  have step4 : white_socks = 120 := by rw [step3, h3]; refl
  calc 
    red_socks + black_socks + white_socks 
      = 40 + 20 + 120 : by rw [step1, step2, step4]
      ... = 180 : by norm_num

end james_total_socks_l738_738125


namespace number_of_pencils_l738_738255

theorem number_of_pencils (P L : ℕ) (h1 : (P : ℚ) / L = 5 / 6) (h2 : L = P + 6) : L = 36 :=
sorry

end number_of_pencils_l738_738255


namespace volume_of_remaining_solid_l738_738951

noncomputable def volume_remaining_cube_with_spheres (a : ℝ) :=
  a^3 * 0.304

theorem volume_of_remaining_solid (a : ℝ) :
  let r := (a / 4) * Real.sqrt 5 in
  let sphere_volume := (4 / 3) * Real.pi * r^3 in
  let total_spheres_volume := 8 * (sphere_volume / 8) in
  let edge_intersections := 6 * (Real.pi * (a^3 / 96) * (5 * Real.sqrt 5 - 11)) in
  a^3 - total_spheres_volume + edge_intersections = volume_remaining_cube_with_spheres a :=
by
  sorry

end volume_of_remaining_solid_l738_738951


namespace number_of_methods_l738_738209

def doctors : ℕ := 6
def days : ℕ := 3

theorem number_of_methods : (days^doctors) = 729 := 
by sorry

end number_of_methods_l738_738209


namespace stock_yield_calculation_l738_738661

theorem stock_yield_calculation (par_value market_value annual_dividend : ℝ)
  (h1 : par_value = 100)
  (h2 : market_value = 80)
  (h3 : annual_dividend = 0.04 * par_value) :
  (annual_dividend / market_value) * 100 = 5 :=
by
  sorry

end stock_yield_calculation_l738_738661


namespace Rio_Coralio_Length_Estimate_l738_738644

def RioCoralioLength := 402.5
def GSA_length := 402
def AWRA_length := 403
def error_margin := 0.5
def error_probability := 0.04

theorem Rio_Coralio_Length_Estimate :
  ∀ (L_GSA L_AWRA : ℝ) (margin error_prob : ℝ),
  L_GSA = GSA_length ∧ L_AWRA = AWRA_length ∧ 
  margin = error_margin ∧ error_prob = error_probability →
  (RioCoralioLength = 402.5) ∧ (error_probability = 0.04) := 
by 
  intros L_GSA L_AWRA margin error_prob h,
  sorry

end Rio_Coralio_Length_Estimate_l738_738644


namespace empty_set_S4_l738_738701

def S1 : Set ℝ := {x | x ^ 2 - 4 = 0}
def S2 : Set ℝ := {x | x > 9 ∨ x < 3}
def S3 : Set (ℝ × ℝ) := {(x, y) | x ^ 2 + y ^ 2 = 0}
def S4 : Set ℝ := {_ | _ > 9 ∧ _ < 3}

theorem empty_set_S4 : S4 = ∅ := by
  sorry

end empty_set_S4_l738_738701


namespace coefficient_square_sum_l738_738088

theorem coefficient_square_sum (a b c d e f : ℤ)
  (h : ∀ x : ℤ, 1728 * x ^ 3 + 64 = (a * x ^ 2 + b * x + c) * (d * x ^ 2 + e * x + f)) :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 23456 :=
by
  sorry

end coefficient_square_sum_l738_738088


namespace fraction_irreducible_l738_738882

theorem fraction_irreducible (n : ℕ) (hn : 0 < n) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := 
by sorry

end fraction_irreducible_l738_738882


namespace zeros_in_square_of_nines_l738_738397

def num_zeros (n : ℕ) (m : ℕ) : ℕ :=
  -- Count the number of zeros in the decimal representation of m
sorry

theorem zeros_in_square_of_nines :
  num_zeros 6 ((10^6 - 1)^2) = 5 :=
sorry

end zeros_in_square_of_nines_l738_738397


namespace quadball_playing_time_l738_738302

theorem quadball_playing_time :
  ∀ (total_children : ℕ) (session_duration : ℕ) (children_per_game : ℕ) (total_duration : ℕ),
    total_children = 8 →
    session_duration = 120 →
    children_per_game = 4 →
    total_duration = children_per_game * session_duration →
    total_duration / total_children = 60 :=
by
  intros total_children session_duration children_per_game total_duration h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end quadball_playing_time_l738_738302


namespace simplify_expression_l738_738539

theorem simplify_expression (x : ℝ) : (3 * x + 30) + (150 * x - 45) = 153 * x - 15 :=
by
  sorry

end simplify_expression_l738_738539


namespace bus_accommodates_36_children_l738_738443

theorem bus_accommodates_36_children (rows : ℕ) (children_per_row : ℕ) (h1 : rows = 9) (h2 : children_per_row = 4) : rows * children_per_row = 36 :=
by
  rw [h1, h2]
  rfl

end bus_accommodates_36_children_l738_738443


namespace student_marks_l738_738310

theorem student_marks
(M P C : ℕ) -- the marks of Mathematics, Physics, and Chemistry are natural numbers
(h1 : C = P + 20)  -- Chemistry is 20 marks more than Physics
(h2 : (M + C) / 2 = 30)  -- The average marks in Mathematics and Chemistry is 30
: M + P = 40 := 
sorry

end student_marks_l738_738310


namespace arithmetic_sequence_x1_x3_div2_x2_l738_738415

open Real

/-- Given the curve C: y = 1/x (x > 0), and points A_1(x_1, 0) and A_2(x_2, 0) where x_2 > x_1 > 0,
/  the point A_3(x_3, 0) is where the line through the intersection points of the curve with 
/ perpendiculars from A_1 and A_2 to the x-axis intersects the x-axis again. We need to prove 
/ that x_1, (x_1 + x_2)/2, x_2 form an arithmetic sequence.
/-/
theorem arithmetic_sequence_x1_x3_div2_x2 {x1 x2 x3 : ℝ} (h1 : x1 > 0) (h2 : x2 > x1) (h3 : x3 = x1 + x2) :
  2 * ((x1 + x2) / 2) = x1 + x2 := 
by
  -- Arithmetic sequence condition: the middle term is the average of the two end terms
  rw [← h3]
  ring
  -- Since equality simplifies the average check and the given condition ensures the arithmetic progression.
  sorry

end arithmetic_sequence_x1_x3_div2_x2_l738_738415


namespace max_sum_e3_f3_g3_h3_i3_l738_738502

theorem max_sum_e3_f3_g3_h3_i3 (e f g h i : ℝ) (h_cond : e^4 + f^4 + g^4 + h^4 + i^4 = 5) :
  e^3 + f^3 + g^3 + h^3 + i^3 ≤ 5^(3/4) :=
sorry

end max_sum_e3_f3_g3_h3_i3_l738_738502


namespace quadratic_inequality_solution_l738_738204

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 + x - 12 ≥ 0} = {x : ℝ | x ≤ -4 ∨ x ≥ 3} :=
sorry

end quadratic_inequality_solution_l738_738204


namespace max_value_and_period_tan_4_5_alpha_l738_738495

noncomputable def f (x : ℝ) : ℝ := 6 * (Real.cos x)^2 - sqrt 3 * Real.sin (2 * x)

-- Maximum value and minimum positive period of f(x)
theorem max_value_and_period :
  (∀ x : ℝ, f x ≤ 2 * sqrt 3 + 3) ∧ (∀ x : ℝ, f x = 2 * sqrt 3 + 3 → (x ∈ [nπ + π/6 for n in ℤ])) ∧ (f.period = π) :=
sorry

-- Value of tan(4/5 * α) given f(α) = 3 - 2√3
theorem tan_4_5_alpha (α : ℝ) (h : 0 < α ∧ α < π/2) (h1 : f α = 3 - 2 * sqrt 3) :
  Real.tan (4 / 5 * α) = sqrt 3 :=
sorry

end max_value_and_period_tan_4_5_alpha_l738_738495


namespace arithmetic_sequence_common_difference_l738_738769

noncomputable def common_difference (a : ℕ → ℤ) (d : ℤ) := a 2 = 1 ∧
  (∑ i in finset.range 5, a (i + 1)) = -15 ∧
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference (a : ℕ → ℤ) (d : ℤ) :
  common_difference a d → d = -2 :=
by
  intro h
  sorry

end arithmetic_sequence_common_difference_l738_738769


namespace Winnie_fell_behind_l738_738251

theorem Winnie_fell_behind
    (yesterday_reps today_reps : ℕ)
    (h_yesterday : yesterday_reps = 86) 
    (h_today : today_reps = 73) :
  yesterday_reps - today_reps = 13 :=
by
  -- Given, yesterday_reps = 86 and today_reps = 73, prove 86 - 73 = 13
  rw [h_yesterday, h_today]
  -- calculation
  exact Nat.sub_self 73
  -- additional arithmetic step
  sorry  -- actual calculation should replace this

end Winnie_fell_behind_l738_738251


namespace part1_part2_l738_738427

-- Part 1
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | (x - 2) / (x - 3) < 0}
def B₁ (a : ℝ) : Set ℝ := {x | (x - (1/2)) * (x - (9/4)) < 0}

theorem part1:
  (Set.univ \ B₁ (1/2)) ∪ A = {x | x ≤ (1/2) ∨ x > 2} :=
sorry

-- Part 2
def B₂ (a : ℝ) : Set ℝ := {x | (x - a) * (x - (a^2 + 2)) < 0}
def p (x : ℝ) : Prop := x ∈ A
def q (x : ℝ) : Prop := x ∈ B₂ a

theorem part2 (a : ℝ) : 
  (∀ x, q x → p x) ∧ ¬∀ x, p x → q x ↔ 
  a ∈ Set.Iic (-1) ∪ Set.Icc 1 2 :=
sorry

end part1_part2_l738_738427


namespace find_some_number_l738_738677

theorem find_some_number (some_number : ℕ) : 
  ( ∃ n:ℕ, n = 54 ∧ (n / 18) * (n / some_number) = 1 ) ∧ some_number = 162 :=
by {
  sorry
}

end find_some_number_l738_738677


namespace division_proof_l738_738326

def dividend : ℕ := 144
def inner_divisor_num : ℕ := 12
def inner_divisor_denom : ℕ := 2
def final_divisor : ℕ := inner_divisor_num / inner_divisor_denom
def expected_result : ℕ := 24

theorem division_proof : (dividend / final_divisor) = expected_result := by
  sorry

end division_proof_l738_738326


namespace union_of_sets_l738_738070

theorem union_of_sets (M N : Set ℕ) (hM : M = {1, 2}) (hN : N = {2 * a - 1 | a ∈ M}) :
  M ∪ N = {1, 2, 3} := by
  sorry

end union_of_sets_l738_738070


namespace total_amount_l738_738731

-- Definitions based on the problem conditions
def jack_amount : ℕ := 26
def ben_amount : ℕ := jack_amount - 9
def eric_amount : ℕ := ben_amount - 10

-- Proof statement
theorem total_amount : jack_amount + ben_amount + eric_amount = 50 :=
by
  -- Sorry serves as a placeholder for the actual proof
  sorry

end total_amount_l738_738731


namespace smallest_sum_p_q_r_s_l738_738138

open Matrix

noncomputable def smallest_p_q_r_s_sum : ℕ :=
    let p : ℕ := 3
    let q : ℕ := 2
    let r : ℕ := 12
    let s : ℕ := 7
    p + q + r + s

theorem smallest_sum_p_q_r_s :
  ∃ p q r s : ℕ, (2 * p = 3 * q) ∧ (7 * r = 12 * s) ∧ (p + q + r + s = 24) :=
begin
  use [3, 2, 12, 7],
  split,
  { norm_num, },
  split,
  { norm_num, },
  { norm_num, }
end

end smallest_sum_p_q_r_s_l738_738138


namespace average_screen_time_per_player_l738_738527

def video_point_guard : ℕ := 130
def video_shooting_guard : ℕ := 145
def video_small_forward : ℕ := 85
def video_power_forward : ℕ := 60
def video_center : ℕ := 180
def total_video_time : ℕ := 
  video_point_guard + video_shooting_guard + video_small_forward + video_power_forward + video_center
def total_video_time_minutes : ℕ := total_video_time / 60
def number_of_players : ℕ := 5

theorem average_screen_time_per_player : total_video_time_minutes / number_of_players = 2 :=
  sorry

end average_screen_time_per_player_l738_738527


namespace relationship_abc_l738_738042

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry

def a := f (Real.log (6 * Real.exp 1 / 5))
def b := f (Real.exp 0.2 - 1)
def c := f (2 / 9)

variable (x : ℝ)

axiom derivative_of_f : ∀ x, deriv (f x) = f' x
axiom symmetry_of_f'_shifted : ∀ x, f' (x - 1) = -f' (2 - x)
axiom symmetry_of_f : ∀ x, f x = f (2 - x)
axiom monotonicity_f_on_interval : ∀ x ∈ Set.Icc (-7.0) (-6.0), f  x = sorry


theorem relationship_abc : a < c ∧ c < b :=
by {
  sorry
}

end relationship_abc_l738_738042


namespace find_M_l738_738086

theorem find_M : (5 + 7 + 9) / 3 = (2005 + 2007 + 2009) / 860 :=
by
  -- Rewrite the problem into the form we need to prove
  have h1 : (5 + 7 + 9) / 3 = 7 := by norm_num
  have h2 : (2005 + 2007 + 2009) = 6021 := by norm_num
  rw [← h2, h1]
  norm_num

end find_M_l738_738086


namespace derivative_f_l738_738908

noncomputable def f (x : ℝ) : ℝ := x + (1 / x)

theorem derivative_f (x : ℝ) (hx : x ≠ 0) :
  deriv f x = 1 - (1 / (x ^ 2)) :=
by
  -- The proof goes here
  sorry

end derivative_f_l738_738908


namespace ratio_of_spend_l738_738867

def giftCardValue : ℝ := 200

def remainingAfterMonday (M : ℝ) : ℝ := giftCardValue - M

def spentOnTuesday (M : ℝ) : ℝ := (1 / 4) * remainingAfterMonday M

def remainingAfterTuesday (M : ℝ) : ℝ := remainingAfterMonday M - spentOnTuesday M

theorem ratio_of_spend (M : ℝ) (h : remainingAfterTuesday M = 75) : M / giftCardValue = 1 / 2 :=
by
  have eq1 : remainingAfterTuesday M = remainingAfterMonday M - (1 / 4) * remainingAfterMonday M := rfl
  rw [remainingAfterMonday, spentOnTuesday] at eq1
  rw h at eq1
  have eq2 : 200 - M - (1 / 4) * (200 - M) = 75 := eq1
  linarith

end ratio_of_spend_l738_738867


namespace total_signals_l738_738298

-- Definitions based on conditions:
-- A display with 4 parallel holes each of which can display 0 or 1
-- Displaying 2 holes at a time and no two adjacent holes can be displayed at the same time

def holes (n : Nat) := n = 4
def can_display (displayed_holes : Finset Nat) := 
  displayed_holes.card = 2 ∧ ∀ (x ∈ displayed_holes) (y ∈ displayed_holes), abs (x - y) > 1
def num_signals (num_pairs : Nat) := num_pairs * 4

theorem total_signals : holes 4 → 
  let pairs := (Finset.filter (can_display) 
    (Finset.powersetLen 2 (Finset.range 4))).card 
  in num_signals pairs = 12 := by
  sorry

end total_signals_l738_738298


namespace fraction_of_females_is_correct_l738_738333

noncomputable def number_of_females_last_year (y : ℕ) : Prop :=
(1.15 * (30 + y) = 33 + 1.3 * y)

theorem fraction_of_females_is_correct (y : ℕ) (h1 : number_of_females_last_year y) : 
  (13 / (33 + 13)) = (13 / 46) := by
  sorry

end fraction_of_females_is_correct_l738_738333


namespace part_a_10_months_cost_part_b_36_months_cost_l738_738971

-- Define the conditions
def incandescent_power : ℕ := 60  -- 60 watts
def energy_efficient_power : ℕ := 12  -- 12 watts
def monthly_hours : ℕ := 100  -- 100 hours
def tariff : ℕ := 5  -- 5 rubles per kWh
def energy_efficient_cost : ℕ := 120  -- 120 rubles
def service_company_share : ℕ := 75  -- 75%

-- Part (a): Prove that the total cost of installing the energy-efficient lamp himself over 10 months 
-- is less than the cost of using the energy service company over the same period
theorem part_a_10_months_cost :
  let monthly_cost_incandescent := (incandescent_power * monthly_hours) / 1000 * tariff in
  let total_cost_incandescent_10_months := monthly_cost_incandescent * 10 in
  let monthly_cost_energy_efficient := (energy_efficient_power * monthly_hours) / 1000 * tariff in
  let total_cost_energy_efficient_10_months := energy_efficient_cost + monthly_cost_energy_efficient * 10 in
  let energy_cost_savings := monthly_cost_incandescent - monthly_cost_energy_efficient in
  let service_payment := (service_company_share * energy_cost_savings) / 100 in
  let total_cost_with_company_10_months := (monthly_cost_energy_efficient + service_payment) * 10 in
  total_cost_energy_efficient_10_months < total_cost_with_company_10_months := by
  sorry

-- Part (b): Prove that the total cost of installing the energy-efficient lamp himself over 36 months 
-- is less than the cost of using the energy service company over the same period
theorem part_b_36_months_cost :
  let monthly_cost_energy_efficient := (energy_efficient_power * monthly_hours) / 1000 * tariff in
  let total_cost_energy_efficient_36_months := energy_efficient_cost + monthly_cost_energy_efficient * 36 in
  let energy_cost_savings := (incandescent_power * monthly_hours) / 1000 * tariff - monthly_cost_energy_efficient in
  let service_payment := (service_company_share * energy_cost_savings) / 100 in
  let total_cost_with_company_10_months := (monthly_cost_energy_efficient + service_payment) * 10 in
  let total_cost_with_company_36_months := total_cost_with_company_10_months + monthly_cost_energy_efficient * 26 in
  total_cost_energy_efficient_36_months < total_cost_with_company_36_months := by
  sorry


end part_a_10_months_cost_part_b_36_months_cost_l738_738971


namespace cherries_eaten_l738_738514

-- Define the number of cherries Oliver had initially
def initial_cherries : ℕ := 16

-- Define the number of cherries Oliver had left after eating some
def left_cherries : ℕ := 6

-- Prove that the difference between the initial and left cherries is 10
theorem cherries_eaten : initial_cherries - left_cherries = 10 := by
  sorry

end cherries_eaten_l738_738514


namespace percentage_not_speak_french_l738_738989

open Nat

theorem percentage_not_speak_french (students_surveyed : ℕ)
  (speak_french_and_english : ℕ) (speak_only_french : ℕ) :
  students_surveyed = 200 →
  speak_french_and_english = 25 →
  speak_only_french = 65 →
  ((students_surveyed - (speak_french_and_english + speak_only_french)) * 100 / students_surveyed) = 55 :=
by
  intros h1 h2 h3
  sorry

end percentage_not_speak_french_l738_738989


namespace min_m_plus_n_l738_738033

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log a (x - 1) + 1

theorem min_m_plus_n (a m n : ℝ) 
  (h_a1 : a > 0)
  (h_a2 : a ≠ 1)
  (h_line : (2 / m) + (1 / n) = 1)
  (h_m_pos : m > 0)
  (h_n_pos : n > 0)
  (h_M : f 2 a = 1) :
  m + n ≥ 3 + 2 * sqrt 2 :=
sorry

end min_m_plus_n_l738_738033


namespace fraction_proof_l738_738029

theorem fraction_proof (a b : ℚ) (h : a / b = 3 / 4) : (a + b) / b = 7 / 4 :=
by
  sorry

end fraction_proof_l738_738029


namespace sum_of_cubes_eq_91_l738_738922

theorem sum_of_cubes_eq_91 (a b : ℤ) (h₁ : a^3 + b^3 = 91) (h₂ : a * b = 12) : a^3 + b^3 = 91 :=
by
  exact h₁

end sum_of_cubes_eq_91_l738_738922


namespace B_spends_85_percent_salary_l738_738662

theorem B_spends_85_percent_salary (A_s B_s : ℝ) (A_savings : ℝ) :
  A_s + B_s = 2000 →
  A_s = 1500 →
  A_savings = 0.05 * A_s →
  (B_s - (B_s * (1 - 0.05))) = A_savings →
  (1 - 0.85) * B_s = 0.15 * B_s := 
by
  intros h1 h2 h3 h4
  sorry

end B_spends_85_percent_salary_l738_738662


namespace quadratic_inequality_solution_l738_738351

theorem quadratic_inequality_solution (x : ℝ) : -3 < x ∧ x < 4 → x^2 - x - 12 < 0 := by
  sorry

end quadratic_inequality_solution_l738_738351


namespace quadratic_root_square_of_another_l738_738007

theorem quadratic_root_square_of_another (a : ℚ) :
  (∃ x y : ℚ, x^2 - (15/4) * x + a^3 = 0 ∧ (x = y^2 ∨ y = x^2) ∧ (x*y = a^3)) →
  (a = 3/2 ∨ a = -5/2) :=
sorry

end quadratic_root_square_of_another_l738_738007


namespace total_notebooks_correct_l738_738930

-- Definitions based on conditions
def total_students : ℕ := 28
def half_students : ℕ := total_students / 2
def notebooks_per_student_group1 : ℕ := 5
def notebooks_per_student_group2 : ℕ := 3

-- Total notebooks calculation
def total_notebooks : ℕ :=
  (half_students * notebooks_per_student_group1) + (half_students * notebooks_per_student_group2)

-- Theorem to be proved
theorem total_notebooks_correct : total_notebooks = 112 := by
  sorry

end total_notebooks_correct_l738_738930


namespace henry_added_water_l738_738076

theorem henry_added_water (F : ℕ) (h2 : F = 32) (α β : ℚ) (h3 : α = 3/4) (h4 : β = 7/8) :
  (F * β) - (F * α) = 4 := by
  sorry

end henry_added_water_l738_738076


namespace sqrt_sum_simplification_l738_738172

theorem sqrt_sum_simplification :
  (sqrt 98 + sqrt 32) = 11 * sqrt 2 :=
by
  have h1 : 98 = 2 * 7^2 := by rfl
  have h2 : 32 = 2^5 := by rfl
  sorry

end sqrt_sum_simplification_l738_738172


namespace find_FC_l738_738378

theorem find_FC
  (DC : ℝ) (CB : ℝ) (AD : ℝ)
  (hDC : DC = 9) (hCB : CB = 10)
  (hAB : ∃ (k1 : ℝ), k1 = 1/5 ∧ AB = k1 * AD)
  (hED : ∃ (k2 : ℝ), k2 = 3/4 ∧ ED = k2 * AD) :
  ∃ FC : ℝ, FC = 11.025 :=
by
  sorry

end find_FC_l738_738378


namespace toby_change_is_7_l738_738943

def cheeseburger_cost : ℝ := 3.65
def milkshake_cost : ℝ := 2
def coke_cost : ℝ := 1
def large_fries_cost : ℝ := 4
def cookie_cost : ℝ := 0.5
def tax : ℝ := 0.2
def toby_funds : ℝ := 15

def total_food_cost_before_tax : ℝ := 
  2 * cheeseburger_cost + milkshake_cost + coke_cost + large_fries_cost + 3 * cookie_cost

def total_bill_with_tax : ℝ := total_food_cost_before_tax + tax

def each_person_share : ℝ := total_bill_with_tax / 2

def toby_change : ℝ := toby_funds - each_person_share

theorem toby_change_is_7 : toby_change = 7 := by
  sorry

end toby_change_is_7_l738_738943


namespace incenter_coords_l738_738113

variables {P Q R J : Type} [AddCommGroup P] [AddCommGroup Q] [AddCommGroup R] [AddCommGroup J]
variables {p q r : ℝ}
variables (u v w : ℝ)
variables (P Q R J : AddCommGroup)

noncomputable def incenter (p q r : ℝ) : (ℝ, ℝ, ℝ) :=
  (p / (p + q + r), q / (p + q + r), r / (p + q + r))

theorem incenter_coords (P Q R J : P) (p q r : ℝ) (u v w : ℝ)
  (hp : p = 8) (hq : q = 10) (hr : r = 6) (hsum : u + v + w = 1) :
  incenter p q r = (1 / 3 : ℝ, 5 / 12 : ℝ, 1 / 4 : ℝ) :=
sorry

end incenter_coords_l738_738113


namespace number_of_second_grade_students_l738_738574

theorem number_of_second_grade_students 
    (students_first_grade students_second_grade students_third_grade total_students total_volunteers : ℕ)
    (h_first : students_first_grade = 1200)
    (h_second : students_second_grade = 1000)
    (h_third : students_third_grade = 800)
    (h_total : total_students = students_first_grade + students_second_grade + students_third_grade)
    (h_volunteers : total_volunteers = 30) :
    (total_volunteers * students_second_grade / total_students) = 10 :=
by
  have h_total_eq : total_students = 3000 := by 
    rw [h_total, h_first, h_second, h_third]
    norm_num
  have h_proportion : (total_volunteers * students_second_grade / total_students : ℝ) = 
                      (30 * 1000 / 3000 : ℝ) := by
    rw [h_first, h_second, h_third]
  norm_num at h_proportion
  exact_mod_cast h_proportion

end number_of_second_grade_students_l738_738574


namespace all_items_weight_is_8040_l738_738511

def weight_of_all_items : Real :=
  let num_tables := 15
  let settings_per_table := 8
  let backup_percentage := 0.25

  let weight_fork := 3.5
  let weight_knife := 4.0
  let weight_spoon := 4.5
  let weight_large_plate := 14.0
  let weight_small_plate := 10.0
  let weight_wine_glass := 7.0
  let weight_water_glass := 9.0
  let weight_table_decoration := 16.0

  let total_settings := (num_tables * settings_per_table) * (1 + backup_percentage)
  let weight_per_setting := (weight_fork + weight_knife + weight_spoon) + (weight_large_plate + weight_small_plate) + (weight_wine_glass + weight_water_glass)
  let total_weight_decorations := num_tables * weight_table_decoration

  let total_weight := total_settings * weight_per_setting + total_weight_decorations
  total_weight

theorem all_items_weight_is_8040 :
  weight_of_all_items = 8040 := sorry

end all_items_weight_is_8040_l738_738511


namespace root_intervals_l738_738263

noncomputable def f (a b c x : ℝ) : ℝ := (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a)

theorem root_intervals (a b c : ℝ) (h : a < b ∧ b < c) :
  ∃ r1 r2 : ℝ, (a < r1 ∧ r1 < b ∧ f a b c r1 = 0) ∧ (b < r2 ∧ r2 < c ∧ f a b c r2 = 0) :=
sorry

end root_intervals_l738_738263


namespace smallest_number_l738_738725

theorem smallest_number : Real.min (Real.min (1 / 2) (2^(-1/2))) (log 3 2) = 1 / 2 :=
by sorry

end smallest_number_l738_738725


namespace sum_three_digit_numbers_formed_by_1_2_3_4_l738_738965

variable (S : Finset ℕ) (threeDigitSet : Finset (Finset (ℕ × ℕ × ℕ)))

def isThreeDigitNumberFormed (a b c : ℕ) : Prop :=
  a ∈ S ∧ b ∈ S ∧ c ∈ S

noncomputable def sumThreeDigitNumbers (S : Finset ℕ) : ℕ :=
  threeDigitSet.sum (λ t, 100 * t.1.1 + 10 * t.1.2 + t.1.3)

theorem sum_three_digit_numbers_formed_by_1_2_3_4 :
  sumThreeDigitNumbers {1, 2, 3, 4} = 17760 :=
by sorry

end sum_three_digit_numbers_formed_by_1_2_3_4_l738_738965


namespace square_roots_equal_implication_l738_738096

theorem square_roots_equal_implication (b : ℝ) (h : 5 * b = 3 + 2 * b) : -b = -1 := 
by sorry

end square_roots_equal_implication_l738_738096


namespace angle_conditions_l738_738572

-- Define the points and angle conditions.
variables {A B C A' B' M : Type*}
variables [euclidean_space A B C A' B' M]
variables (AA_prime BB_prime medians : A → B → C → A' → B' → M → Prop)

-- Conditions provided in the problem:
def conditions (A A' B B' C M : A):= 
  -- Medians intersect at the centroid M
  medians A A' B B' C M ∧
  -- Given angle is 120 degrees.
  angle A M B = 120

theorem angle_conditions (A A' B B' C M : A) (h : conditions A A' B B' C M) :
  ¬(acute (angle A B' M) ∧ acute (angle B A' M)) ∧
  ¬(obtuse (angle A B' M) ∧ obtuse (angle B A' M)) := sorry

end angle_conditions_l738_738572


namespace largest_number_in_systematic_sample_l738_738688

theorem largest_number_in_systematic_sample : 
  ∀ (n : ℕ), 0 < n → n ≤ 500 → 
  let interval := 25 in 
  let smallest_1 := 7 in
  let smallest_2 := 32 in
  smallest_2 - smallest_1 = interval → 
  let sample_size := 500 / interval in
  ∃ x, x = smallest_1 + interval * (sample_size - 1) ∧ x = 482 :=
by
  sorry

end largest_number_in_systematic_sample_l738_738688


namespace circumradius_of_triangle_l738_738293

theorem circumradius_of_triangle (a b c : ℕ) (h₁ : a = 8) (h₂ : b = 6) (h₃ : c = 10) 
  (h₄ : a^2 + b^2 = c^2) : 
  (c : ℝ) / 2 = 5 := 
by {
  -- proof goes here
  sorry
}

end circumradius_of_triangle_l738_738293


namespace facu_underlined_numbers_l738_738869

/-- Define the arithmetic sequence a_n = 3n - 2 -/
def arithmetic_sequence (n : ℕ) : ℤ := 3 * n - 2

/-- Define a predicate to check if a number has all digits the same -/
def all_digits_same (x : ℤ) : Prop :=
  let digits := x.to_nat.digits 10 in
  ∃ d, ∀ digit ∈ digits, digit = d

/-- Define the main Lean theorem to prove the numbers Facu underlined -/
theorem facu_underlined_numbers :
  {x : ℤ | 10 < x ∧ x < 100000 ∧ all_digits_same x ∧ ∃ n, arithmetic_sequence n = x} =
  {22, 55, 88, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 
   11111, 22222, 33333, 44444, 55555, 66666, 77777, 88888, 99999}.to_set :=
by
  sorry

end facu_underlined_numbers_l738_738869


namespace least_positive_integer_special_property_l738_738359

theorem least_positive_integer_special_property : ∃ (N : ℕ) (a b c : ℕ), 
  N = 100 * a + 10 * b + c ∧ a ≠ 0 ∧ 10 * b + c = N / 29 ∧ N = 725 :=
by
  sorry

end least_positive_integer_special_property_l738_738359


namespace eighth_graders_count_l738_738176

variables (n : ℕ) (a : ℕ → ℝ) (S : ℝ)
-- Define the conditions
def condition1 (i : ℕ) : Prop := a i > (1/5) * (S - a i)
def condition2 (i : ℕ) : Prop := a i < (1/3) * (S - a i)

-- The main theorem to be proved
theorem eighth_graders_count (h1 : ∀ i, 1 ≤ i ∧ i ≤ n → condition1 a S i)
                           (h2 : ∀ i, 1 ≤ i ∧ i ≤ n → condition2 a S i)
                           (h3 : S = ∑ i in (finset.range n).map (λ x, x+1), a i) :
  n = 5 :=
begin
  sorry
end

end eighth_graders_count_l738_738176


namespace canoe_rental_cost_per_day_l738_738218

-- given conditions and definitions
variables (C K : ℕ)
def kayak_rental_cost_per_day : ℕ := 18
def total_revenue_in_dollars : ℕ := 504
def canoes_more_than_kayaks : ℕ := 7
def kayaks_rented_equation : Prop := (3 / 2 : ℚ) * K = K + canoes_more_than_kayaks

-- Lean statement: prove the daily canoe rental cost
theorem canoe_rental_cost_per_day :
  kayak_rental_cost_per_day * K + C * (K + canoes_more_than_kayaks) = total_revenue_in_dollars ∧ kayaks_rented_equation → C = 12 :=
by
  admit

end canoe_rental_cost_per_day_l738_738218


namespace integer_points_on_circle_l738_738623

-- Define the radius and circle equation
def radius : ℕ := 5
def circle_eq (x y : ℤ) : Prop := x^2 + y^2 = radius^2

-- Define the problem to prove the number of integer points on the circle
theorem integer_points_on_circle : 
  {p : ℤ × ℤ | circle_eq p.1 p.2}.to_finset.card = 12 := 
sorry

end integer_points_on_circle_l738_738623


namespace kelly_gave_away_games_l738_738484

theorem kelly_gave_away_games (initial_games : ℕ) (remaining_games : ℕ) (given_away_games : ℕ) 
  (h1 : initial_games = 183) 
  (h2 : remaining_games = 92) 
  (h3 : given_away_games = initial_games - remaining_games) : 
  given_away_games = 91 := 
by 
  sorry

end kelly_gave_away_games_l738_738484


namespace M_necessary_for_N_l738_738071

open Set

def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

theorem M_necessary_for_N : ∀ a : ℝ, a ∈ N → a ∈ M ∧ ¬(a ∈ M → a ∈ N) :=
by
  sorry

end M_necessary_for_N_l738_738071


namespace centroid_inverse_square_sum_l738_738492

theorem centroid_inverse_square_sum (O : ℝ × ℝ × ℝ) (k : ℝ) (h1 : k > 0)
  (A B C : ℝ × ℝ × ℝ) (hA : A ≠ O) (hB : B ≠ O) (hC : C ≠ O) (h_dist : 1 / real.sqrt ((1 / (A.1 ^ 2)) + (1 / (B.2 ^ 2)) + (1 / (C.3 ^ 2))) = k)
  (p q r : ℝ) (h_centroid : (p, q, r) = ((A.1 / 3), (B.2 / 3), (C.3 / 3))) :
  1 / (p ^ 2) + 1 / (q ^ 2) + 1 / (r ^ 2) = 9 / (k ^ 2) :=
sorry

end centroid_inverse_square_sum_l738_738492


namespace find_r_k_l738_738569

theorem find_r_k :
  ∃ r k : ℚ, (∀ t : ℚ, (∃ x y : ℚ, (x = r + 3 * t ∧ y = 2 + k * t) ∧ y = 5 * x - 7)) ∧ 
            r = 9 / 5 ∧ k = -4 :=
by {
  sorry
}

end find_r_k_l738_738569


namespace sum_of_areas_squares_l738_738693

theorem sum_of_areas_squares (a : ℝ) : 
  (∑' n : ℕ, (a^2 / 4^n)) = (4 * a^2 / 3) :=
by
  sorry

end sum_of_areas_squares_l738_738693


namespace derivative_f_correct_l738_738812

-- Define the function f
def f (x : ℝ) : ℝ := (Real.exp x) / x

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := (x * (Real.exp x) - (Real.exp x)) / (x * x)

-- State the theorem
theorem derivative_f_correct (x : ℝ) (h : x ≠ 0) : (deriv f x) = f' x :=
by
  sorry

end derivative_f_correct_l738_738812


namespace angle_ADB_eq_90_degrees_l738_738666

noncomputable def point := ℝ × ℝ
noncomputable def radius := ℝ
noncomputable def angle := ℝ

axiom center (C : point) : point
axiom radius_C (r : radius) : r = 15
axiom vertex_B (B : point) : point
axiom vertex_A (A : point) : point

axiom AC_eq_BC : dist C A = dist C B
axiom A_on_circle : dist C A = radius_C
axiom D_on_circle : (∃ D : point, line AC D ∧ (dist C D = radius_C))

axiom ACB_eq_80_degrees : ∃ θ : angle, θ = 80

theorem angle_ADB_eq_90_degrees : 
  ∀ (A B C D : point) r, 
    dist C A = r ∧ dist C B = r ∧ r = 15 ∧ (∃ D : point, line AC D ∧ dist C D = r) 
    → ∃ (α : angle), α = 90 := 
by 
  sorry

end angle_ADB_eq_90_degrees_l738_738666


namespace number_of_cows_l738_738626

theorem number_of_cows (C H : ℕ) (L : ℕ) (h1 : L = 4 * C + 2 * H) (h2 : L = 2 * (C + H) + 20) : C = 10 :=
by
  sorry

end number_of_cows_l738_738626


namespace time_brushing_each_cat_l738_738430

theorem time_brushing_each_cat :
  ∀ (t_total_free_time t_vacuum t_dust t_mop t_free_left_after_cleaning t_cats : ℕ),
  t_total_free_time = 3 * 60 →
  t_vacuum = 45 →
  t_dust = 60 →
  t_mop = 30 →
  t_cats = 3 →
  t_free_left_after_cleaning = 30 →
  ((t_total_free_time - t_free_left_after_cleaning) - (t_vacuum + t_dust + t_mop)) / t_cats = 5
 := by
  intros t_total_free_time t_vacuum t_dust t_mop t_free_left_after_cleaning t_cats
  intros h_total_free_time h_vacuum h_dust h_mop h_cats h_free_left
  sorry

end time_brushing_each_cat_l738_738430


namespace find_value_of_a5_l738_738119

variable (a : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a_1 d : ℝ), ∀ n, a n = a_1 + (n - 1) * d

variable (h_arith : is_arithmetic_sequence a)
variable (h : a 2 + a 8 = 12)

theorem find_value_of_a5 : a 5 = 6 :=
by
  sorry

end find_value_of_a5_l738_738119


namespace least_value_f_sidon_2010_l738_738487

noncomputable def least_value_f_sidon (n : ℕ) : ℕ := 
if h : n = 2010 then 4040100 else 0

theorem least_value_f_sidon_2010 : least_value_f_sidon 2010 = 4040100 :=
by
  unfold least_value_f_sidon
  simp [dif_pos]
  sorry

end least_value_f_sidon_2010_l738_738487


namespace muffin_cost_l738_738375

theorem muffin_cost (m : ℝ) :
  let fruit_cup_cost := 3
  let francis_cost := 2 * m + 2 * fruit_cup_cost
  let kiera_cost := 2 * m + 1 * fruit_cup_cost
  let total_cost := 17
  (francis_cost + kiera_cost = total_cost) → m = 2 :=
by
  intro h
  sorry

end muffin_cost_l738_738375


namespace total_peaches_l738_738147

def S : ℝ := 13
def J : ℝ := S - 6
def T : ℝ := S + 5
def J' : ℝ := real.sqrt (1/2) * J
def T' : ℝ := real.cbrt 2 * T

theorem total_peaches : J' + T' = 27.63 := 
by
  -- Specify the expected value
  let expected : ℝ := 27.63
  -- Calculate J
  have hJ : J = 7 := by
    simp [J, S]
  -- Calculate T
  have hT : T = 18 := by
    simp [T, S]
  -- Substitute J and T values into J' and T'
  have hJ' : J' = real.sqrt (0.5) * 7 := by
    simp [J', hJ]
  have hT' : T' = real.cbrt 2 * 18 := by
    simp [T', hT]
  -- Calculate J' and T'
  have hJ'_calc : J' ≈ 4.95 := by
    simp [hJ']
    -- Apply real.sqrt and arithmetic simplification
    norm_num [real.sqrt]
  have hT'_calc : T' ≈ 22.68 := by
    simp [hT']
    -- Apply real.cbrt and arithmetic simplification
    norm_num [real.cbrt]
  -- Calculate the total number of peaches
  have total : J' + T' ≈ 27.63 := by
    rw [hJ'_calc, hT'_calc]
    -- Use rounding and approximation
    norm_num
  exact total

end total_peaches_l738_738147


namespace problem_1_problem_2_problem_3_l738_738055

-- Definition of the functions f and g
def f (x a : ℝ) : ℝ := abs (x - a)
def g (x a : ℝ) : ℝ := a * x
def F (x a : ℝ) : ℝ := (g x a) * (f x a)

-- Proof statement for each question

-- Question 1: If a = 1, solve f(x) = g(x)
theorem problem_1 (x : ℝ) (h : g x 1 = f x 1) : x = 2 :=
by sorry

-- Question 2: If the equation f(x) = g(x) has two solutions, find the range of the real number a
theorem problem_2 (a : ℝ) (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = g x1 a ∧ f x2 a = g x2 a) :
  a ∈ set.Ioo (-1 : ℝ) 0 ∪ set.Ioo 0 1 :=
by sorry

-- Question 3: If a > 0, find the maximum value of y = F(x) in the interval [1, 2]
theorem problem_3 (a : ℝ) (h : a > 0) :
  ∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ 
  F x a = max ((4 * a - 2 * a^2) * indicator (set.Ioo 0 (5 / 3 : ℝ)) a)
             ((a^2 - a) * indicator (set.Icc (5 / 3 : ℝ) 2) a)
             ((3 * a^3 / 4) * indicator (set.Ioo 2 4) a)
             ((2 * a^2 - 4 * a) * indicator (set.Ici 4) a) :=
by sorry

end problem_1_problem_2_problem_3_l738_738055


namespace sum_of_inverses_mod_17_l738_738241

noncomputable def inverse_sum_mod_17 : ℤ :=
  let a1 := Nat.gcdA 3 17 in -- 3^{-1} mod 17
  let a2 := Nat.gcdA (3^2) 17 in -- 3^{-2} mod 17
  let a3 := Nat.gcdA (3^3) 17 in -- 3^{-3} mod 17
  let a4 := Nat.gcdA (3^4) 17 in -- 3^{-4} mod 17
  let a5 := Nat.gcdA (3^5) 17 in -- 3^{-5} mod 17
  let a6 := Nat.gcdA (3^6) 17 in -- 3^{-6} mod 17
  (a1 + a2 + a3 + a4 + a5 + a6) % 17

theorem sum_of_inverses_mod_17 : inverse_sum_mod_17 = 7 := sorry

end sum_of_inverses_mod_17_l738_738241


namespace sum_of_fractions_l738_738609

theorem sum_of_fractions : (1 / 6) + (2 / 9) + (1 / 3) = 13 / 18 := by
  sorry

end sum_of_fractions_l738_738609


namespace trajectory_of_center_slope_of_AB_constant_l738_738798

-- Define the circles C₁ and C₂ as given in the problem
def C₁_eq (x y : ℝ) : Prop := (x + 1)^2 + y = 25
def C₂_eq (x y : ℝ) : Prop := (x - 1)^2 + y = 1

-- Define the trajectory E as an ellipse
def trajectory_eq (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1

-- Theorem 1: Prove the trajectory E of the center of circle C is an ellipse
theorem trajectory_of_center (x y : ℝ) :
  (exists R : ℝ, (C₁_eq (x, y) ∧ |5 - R| ∧ C₂_eq (x, y) ∧ |R - 1| ∧ (|5 - R| + |R - 1| = 4))) ->
  trajectory_eq x y := 
sorry

-- Define the slope of line AB given the coordinates of points A and B
def slope (A B : ℝ × ℝ) : ℝ := (B.2 - A.2) / (B.1 - A.1)

-- Given conditions in part 2
def point_on_trajectory (x y : ℝ) : Prop := trajectory_eq 1 ((3:ℝ)/2)
def negative_reciprocal_slopes (k1 k2 : ℝ) : Prop := k1 * k2 = -1

-- Theorem 2: Prove the slope of AB is constant
theorem slope_of_AB_constant 
  (A B P : ℝ × ℝ) (hP : P = (1, (3:ℝ)/2)) (hAB : negative_reciprocal_slopes (slope P A) (slope P B)) : 
  slope A B = (1:ℝ) / 2 := 
sorry

end trajectory_of_center_slope_of_AB_constant_l738_738798


namespace closest_to_actual_is_140_l738_738346

def actual_value := 3.52 * 7.861 * (6.28 - 1.283)
def approx_value := 3.5 * 8 * 5

theorem closest_to_actual_is_140 :
  abs ((actual_value : ℝ) - 140) < abs ((actual_value - 120)) ∧
  abs ((actual_value : ℝ) - 140) < abs ((actual_value - 160)) ∧
  abs ((actual_value : ℝ) - 140) < abs ((actual_value - 180)) ∧
  abs ((actual_value : ℝ) - 140) < abs ((actual_value - 200)) :=
by
  sorry

end closest_to_actual_is_140_l738_738346


namespace hyperbola_triangle_area_and_perpendicular_l738_738830

section Hyperbola_Proof

variables {x y : ℝ}

/-- Define the hyperbola C1 by the equation 2x^2 - y^2 = 1 -/
def hyperbola_C1 (x y : ℝ) : Prop := 2 * x^2 - y^2 = 1

/-- Define the circle by the equation x^2 + y^2 = 1 -/
def circle (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Define the line through points P and Q with slope 1 -/
def line_slope_1 (x y b : ℝ) : Prop := y = x + b

/-- Given conditions and proving the desired geometrical properties -/
theorem hyperbola_triangle_area_and_perpendicular (x y b : ℝ) (P Q : ℝ × ℝ) 
  (h1 : hyperbola_C1 P.1 P.2) (h2 : hyperbola_C1 Q.1 Q.2) (h3 : line_slope_1 x y b) 
  (tangent : ∀ x y, circle x y → y = x + b)
  (vertex : ∃ x, hyperbola_C1 x 0 ∧ x < 0):
  ∃ area : ℝ, area = 1 / 2 ∧ (0, 0) ≠ P ∧ (0, 0) ≠ Q ∧ (P.1 * Q.1 + P.2 * Q.2 = 0) := 
sorry

end Hyperbola_Proof

end hyperbola_triangle_area_and_perpendicular_l738_738830


namespace special_prime_sum_correct_l738_738364

open Nat

def is_special_prime (p : ℕ) : Prop :=
  Prime p ∧ p % 3 = 1 ∧ p % 5 = 4

def special_prime_sum (n : ℕ) : ℕ :=
  (Finset.filter (λ p, is_special_prime p) (Finset.range n)).sum id

theorem special_prime_sum_correct : special_prime_sum 120 = 207 :=
  by sorry

end special_prime_sum_correct_l738_738364


namespace cosine_double_angle_l738_738760

theorem cosine_double_angle (α : ℝ) (h : Real.sin α = 1 / 3) : Real.cos (2 * α) = 7 / 9 :=
by
  sorry

end cosine_double_angle_l738_738760


namespace sum_of_inverses_mod_17_l738_738238

noncomputable def inverse_sum_mod_17 : ℤ :=
  let a1 := Nat.gcdA 3 17 in -- 3^{-1} mod 17
  let a2 := Nat.gcdA (3^2) 17 in -- 3^{-2} mod 17
  let a3 := Nat.gcdA (3^3) 17 in -- 3^{-3} mod 17
  let a4 := Nat.gcdA (3^4) 17 in -- 3^{-4} mod 17
  let a5 := Nat.gcdA (3^5) 17 in -- 3^{-5} mod 17
  let a6 := Nat.gcdA (3^6) 17 in -- 3^{-6} mod 17
  (a1 + a2 + a3 + a4 + a5 + a6) % 17

theorem sum_of_inverses_mod_17 : inverse_sum_mod_17 = 7 := sorry

end sum_of_inverses_mod_17_l738_738238


namespace quadratic_complete_square_l738_738578

theorem quadratic_complete_square (b c : ℤ) :
  (∀ x : ℝ, x^2 - 16 * x + 15 = (x + b)^2 + c) → b + c = -57 :=
begin
  sorry
end

end quadratic_complete_square_l738_738578


namespace median_length_is_five_l738_738114

-- Formalizing the given conditions
variables {AC BC : ℝ}
variable {C_right : angle B C A = 90}

noncomputable def hypotenuse_length (AC BC : ℝ) : ℝ :=
  real.sqrt (AC^2 + BC^2)

noncomputable def median_to_hypotenuse (hypotenuse_length : ℝ) : ℝ :=
  hypotenuse_length / 2

theorem median_length_is_five (hAC : AC = 6) (hBC : BC = 8) :
  median_to_hypotenuse (hypotenuse_length 6 8) = 5 :=
by
  -- sorry is used to replace the actual proof, which is not required according to instruction
  sorry

end median_length_is_five_l738_738114


namespace energy_comparison_l738_738897

theorem energy_comparison (m : ℝ) : 
  let KE := λ v : ℝ, (1 / 2) * m * v ^ 2 in 
  (KE 4 - KE 2) = 3 * (KE 2 - KE 0) :=
by
  sorry

end energy_comparison_l738_738897


namespace min_handshakes_30_people_3_each_l738_738276

theorem min_handshakes_30_people_3_each : 
  ∃ (H : ℕ), (∀ (n k : ℕ), n = 30 ∧ k = 3 → H = (n * k) / 2) := 
by {
  use 45,
  intros n k h,
  rw [← h.1, ← h.2],
  norm_num,
  sorry
}

end min_handshakes_30_people_3_each_l738_738276


namespace meters_of_cloth_l738_738480

variable (total_cost cost_per_meter : ℝ)
variable (h1 : total_cost = 434.75)
variable (h2 : cost_per_meter = 47)

theorem meters_of_cloth : 
  total_cost / cost_per_meter = 9.25 := 
by
  sorry

end meters_of_cloth_l738_738480


namespace minimum_value_g_l738_738727

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 
  if a > 1 then 
    a * (-1/a) + 1 
  else 
    if 0 < a then 
      a^2 + 1 
    else 
      0  -- adding a default value to make it computable

theorem minimum_value_g (a : ℝ) (m : ℝ) : 0 < a ∧ a < 2 ∧ ∃ x₀, f x₀ a = m → m ≥ 5 / 2 :=
by
  sorry

end minimum_value_g_l738_738727


namespace declaration_of_independence_signed_on_wednesday_l738_738181

/-- 
This theorem proves that the day of the week on which the Declaration 
of Independence was signed in 1776 is Wednesday, given that July 4, 2026, 
which is the 250th anniversary, falls on a Saturday. The leap year rule 
is used in the proof.
-/
theorem declaration_of_independence_signed_on_wednesday :
  (250 : ℕ) % 7 = 3 → (nist_weekday_of_Jul4_2026 Saturday → nist_weekday_of_Jul4_1776 Wednesday) :=
begin
  sorry
end

end declaration_of_independence_signed_on_wednesday_l738_738181


namespace handshakes_minimum_l738_738269

/-- Given 30 people and each person shakes hands with exactly three others,
    the minimum possible number of handshakes is 45. -/
theorem handshakes_minimum (n k : ℕ) (h_n : n = 30) (h_k : k = 3) :
  (n * k) / 2 = 45 :=
by
  sorry

end handshakes_minimum_l738_738269


namespace smallest_divisor_after_323_l738_738510

-- Let n be an even 4-digit number such that 323 is a divisor of n.
def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def is_4_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_divisor (d n : ℕ) : Prop :=
  n % d = 0

theorem smallest_divisor_after_323 (n : ℕ) (h1 : is_even n) (h2 : is_4_digit n) (h3 : is_divisor 323 n) : ∃ k, k > 323 ∧ is_divisor k n ∧ k = 340 :=
by
  sorry

end smallest_divisor_after_323_l738_738510


namespace cube_surface_area_l738_738407

theorem cube_surface_area (edge_length : ℝ) (h : edge_length = 1) : 
  6 * edge_length^2 = 6 :=
by
  rw [h]
  norm_num
  sorry

end cube_surface_area_l738_738407


namespace simplify_fraction_sum_product_l738_738324

theorem simplify_fraction (n : ℕ) (h : 1 ≤ n) : 
  1 / (Real.sqrt (n + 1) + Real.sqrt n) = Real.sqrt (n + 1) - Real.sqrt n :=
sorry

theorem sum_product (sum_eq : ∑ k in Finset.range 2014, 
  (1 / (Real.sqrt (k + 2) + Real.sqrt (k + 1)))) : 
  (sum_eq *(Real.sqrt 2015 + Real.sqrt 2)) = 2013 :=
sorry

end simplify_fraction_sum_product_l738_738324


namespace close_200_cities_keeping_graph_connected_l738_738925

noncomputable def cities := 1998
noncomputable def flights_per_city := 3

-- Condition: A 3-regular, connected graph of 1998 vertices.
-- Definition of this graph as connected and 3-regular behavior
def is_3_regular_connected_graph (G : simple_graph (fin cities)) : Prop :=
  G.is_regular_of_degree flights_per_city ∧ G.connected

-- Lemma: In a connected graph with n vertices and at least (4n / 3) edges, there are two intersecting cycles.
lemma two_intersecting_cycles_in_connected_graph
  {n : ℕ} (h : 4 * n / 3 ≤ simple_graph.edge_count G) :
  ∃ u v, u ≠ v ∧ G.is_cycle (u :: v :: nil)

-- Proof that we can remove 200 cities (vertices) such that the graph remains connected and no two of the removed cities are connected.
theorem close_200_cities_keeping_graph_connected (G : simple_graph (fin cities))
  (h : is_3_regular_connected_graph G) : 
  ∃ closed : fin_set cities, closed.card = 200 ∧
    (∀ (u v : fin cities), u ∈ closed → v ∈ closed → ¬ G.adj u v) ∧
    G.subgraph_induced_by {v | v ∉ closed}.connected :=
sorry

end close_200_cities_keeping_graph_connected_l738_738925


namespace james_total_socks_l738_738123

theorem james_total_socks :
  ∀ (red_socks_pairs : ℕ) (black_socks_ratio red_socks_ratio : ℕ),
    red_socks_pairs = 20 →
    black_socks_ratio = 2 →
    red_socks_ratio = 2 →
    let red_socks := red_socks_pairs * 2 in
    let black_socks := red_socks / black_socks_ratio in
    let red_black_combined := red_socks + black_socks in
    let white_socks := red_black_combined * red_socks_ratio in
    red_socks + black_socks + white_socks = 180 :=
by
  intros red_socks_pairs black_socks_ratio red_socks_ratio
  intro h1 h2 h3
  let red_socks := red_socks_pairs * 2
  let black_socks := red_socks / black_socks_ratio
  let red_black_combined := red_socks + black_socks
  let white_socks := red_black_combined * red_socks_ratio
  have step1 : red_socks = 40 := by rw [h1]; refl
  have step2 : black_socks = 20 := by rw [step1, h2]; refl
  have step3 : red_black_combined = 60 := by rw [step1, step2]; refl
  have step4 : white_socks = 120 := by rw [step3, h3]; refl
  calc 
    red_socks + black_socks + white_socks 
      = 40 + 20 + 120 : by rw [step1, step2, step4]
      ... = 180 : by norm_num

end james_total_socks_l738_738123


namespace inverse_sum_mod_l738_738222

theorem inverse_sum_mod (h1 : ∃ k, 3^6 ≡ 1 [MOD 17])
                        (h2 : ∃ k, 3 * 6 ≡ 1 [MOD 17]) : 
  (6 + 9 + 2 + 1 + 6 + 1) % 17 = 8 :=
by
  cases h1 with k1 h1
  cases h2 with k2 h2
  sorry

end inverse_sum_mod_l738_738222


namespace arrange_numbers_in_ascending_order_l738_738321

theorem arrange_numbers_in_ascending_order :
  ∀ (a b c d : ℕ), 
  a = 440050 → b = 46500 → c = 440500 → d = 439500 →
  list.sort (≤) [a, b, c, d] = [b, d, a, c] :=
by 
  intros a b c d ha hb hc hd
  rw [ha, hb, hc, hd]
  exact list.sort(≤) [440050, 46500, 440500, 439500] = [46500, 439500, 440050, 440500]
  sorry

end arrange_numbers_in_ascending_order_l738_738321


namespace simplify_fraction_l738_738888

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem simplify_fraction :
  (factorial 13) / (factorial 11 + 2 * factorial 10) = 12 := sorry

end simplify_fraction_l738_738888


namespace store_profit_l738_738682

theorem store_profit {C : ℝ} (h₁ : C > 0) : 
  let SP1 := 1.20 * C
  let SP2 := 1.25 * SP1
  let SPF := 0.80 * SP2
  SPF - C = 0.20 * C := 
by 
  let SP1 := 1.20 * C
  let SP2 := 1.25 * SP1
  let SPF := 0.80 * SP2
  sorry

end store_profit_l738_738682


namespace son_l738_738964

theorem son's_age (S M : ℕ) (h₁ : M = S + 25) (h₂ : M + 2 = 2 * (S + 2)) : S = 23 := by
  sorry

end son_l738_738964


namespace perimeter_of_equilateral_triangle_l738_738556

theorem perimeter_of_equilateral_triangle (s : ℝ) 
  (h1 : (s ^ 2 * Real.sqrt 3) / 4 = 2 * s) : 
  3 * s = 8 * Real.sqrt 3 :=
by
  sorry

end perimeter_of_equilateral_triangle_l738_738556


namespace min_handshakes_30_people_3_each_l738_738275

theorem min_handshakes_30_people_3_each : 
  ∃ (H : ℕ), (∀ (n k : ℕ), n = 30 ∧ k = 3 → H = (n * k) / 2) := 
by {
  use 45,
  intros n k h,
  rw [← h.1, ← h.2],
  norm_num,
  sorry
}

end min_handshakes_30_people_3_each_l738_738275


namespace equalizing_table_impossible_l738_738653

theorem equalizing_table_impossible (n : ℕ) (table : matrix (fin n) (fin n) ℕ) 
  (h_ones : ∃! (cells_with_one : fin n × fin n), table cells_with_one = 1) 
  (h_allowed_op : ∀ (r c : fin n), table r c - 1 ∈ ℕ ∧ 
                    (∀ i ∈ fin n, table r i = table r i + 1) ∧ 
                    (∀ j ∈ fin n, table j c = table j c + 1)) :
  ¬ ∃ k : ℕ, ∀ r c : fin n, table r c = k :=
sorry

end equalizing_table_impossible_l738_738653


namespace pauline_shoveling_time_l738_738879

noncomputable def snow_shoveling_time (rate : ℕ → ℝ) (initial_volume : ℝ) : ℕ :=
  let rec snow_remaining (n : ℕ) (volume : ℝ) : ℝ :=
    if n = 0 then volume
    else snow_remaining (n - 1) (volume - rate n);
  nat.find (λ n, snow_remaining n initial_volume ≤ 71)

theorem pauline_shoveling_time :
  let rate (n : ℕ) : ℝ := max 1 (25 - 2 * (n - 1))
  (snow_shoveling_time rate 240) = 13 :=
  sorry

end pauline_shoveling_time_l738_738879


namespace proof_paddle_prices_and_cost_optimization_l738_738287

-- Define the conditions
variables (x y : ℕ)
variables (m : ℕ)
variables (cost : ℕ)

-- The first condition: Total cost for 20 straight + 15 horizontal pairs of paddles with balls is 9000 yuan
def condition1 := 20 * (x + 20 * 2) + 15 * (y + 20 * 2) = 9000

-- The second condition: Cost difference between 10 horizontal and 5 straight pairs is 1600 yuan
def condition2 := 5 * (x + 20 * 2) + 1600 = 10 * (y + 20 * 2)

-- The threshold condition: Number of straight paddles is not more than three times the number of horizontal paddles
def condition3 := m ≤ 3 * (40 - m)

-- The optimization problem: Minimizing total cost for 40 pairs of paddles
def total_cost := (220 + 20 * 2) * m + (260 + 20 * 2) * (40 - m)

theorem proof_paddle_prices_and_cost_optimization 
    (h1 : condition1)
    (h2 : condition2)
    (h3 : condition3)
    (h4 : x = 220)
    (h5 : y = 260) : total_cost = 10000 :=
sorry  -- Proof left as an exercise

end proof_paddle_prices_and_cost_optimization_l738_738287


namespace annie_payment_for_12_kg_l738_738708

-- Given conditions as definitions
def price_per_kg (price : ℝ) (mass : ℝ) : ℝ := price / mass

def total_cost (price_per_kg : ℝ) (mass : ℝ) : ℝ := price_per_kg * mass

-- Mathematical problem rewritten as a Lean theorem statement
theorem annie_payment_for_12_kg (h1 : price_per_kg 6 2 = 3) : total_cost 3 12 = 36 :=
by
  -- Proof will be provided here
  sorry

end annie_payment_for_12_kg_l738_738708


namespace AM_GM_contradiction_l738_738780

open Real

theorem AM_GM_contradiction (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
      ¬ (6 < a + 4 / b ∧ 6 < b + 9 / c ∧ 6 < c + 16 / a) := by
  sorry

end AM_GM_contradiction_l738_738780


namespace cylinder_base_radius_proof_l738_738567

variable (r : ℝ)

def cylinder_height := 3 * r

def sphere_radius := r

noncomputable def cylinder_base_radius : ℝ :=
  r * (4 + 3 * Real.sqrt 2) / 4

theorem cylinder_base_radius_proof :
  (cylinder_height r = 3 * r) ∧
  (sphere_radius r = r) ∧
  (let R := cylinder_base_radius r in
    R = r * (4 + 3 * Real.sqrt 2) / 4) := by
  sorry

end cylinder_base_radius_proof_l738_738567


namespace triangle_is_isosceles_right_triangle_l738_738924

theorem triangle_is_isosceles_right_triangle
    (A B C : ℝ) (h1 : A + B + C = π) 
    (h2 : sin A * sin B + sin A * cos B + cos A * sin B + cos A * cos B = 2) :
  A = B ∧ C = π / 2 :=
by 
  sorry

end triangle_is_isosceles_right_triangle_l738_738924


namespace aaron_walking_speed_l738_738312

-- Definitions of the conditions
def distance_jog : ℝ := 3 -- in miles
def speed_jog : ℝ := 2 -- in miles/hour
def total_time : ℝ := 3 -- in hours

-- The problem statement
theorem aaron_walking_speed :
  ∃ (v : ℝ), v = (distance_jog / (total_time - (distance_jog / speed_jog))) ∧ v = 2 :=
by
  sorry

end aaron_walking_speed_l738_738312


namespace geometric_sequence_ratio_l738_738580

theorem geometric_sequence_ratio (a : ℕ → ℤ) (q : ℤ) (n : ℕ) (i : ℕ → ℕ) (ε : ℕ → ℤ) :
  (∀ (k : ℕ), 1 ≤ k ∧ k ≤ n → a k = a 1 * q ^ (k - 1)) ∧
  (∃ (k : ℕ), 1 ≤ k ∧ k ≤ n → ε k * a (i k) = 0) ∧
  (∀ m, 1 ≤ i m ∧ i m ≤ n) → q = -1 := 
sorry

end geometric_sequence_ratio_l738_738580


namespace matrix_multiplication_result_l738_738592

-- Define the matrix A and B based on the problem's conditions
def A : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![0, 1, -1],
    ![1, 0, 0],
    ![1, 0, 1]
  ]

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![0, 1, 0],
    ![0, 2, 0],
    ![1, 0, 0]
  ]

-- Define the resultant matrix for B * (2A - B)
def resultMatrix : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![2, -2, 0],
    ![0, -4, 0],
    ![1, 1, -2]
  ]

-- Prove that B * (2A - B) equals resultMatrix
theorem matrix_multiplication_result :
  B * (bit0 1 * A - B) = resultMatrix :=
by {
  -- Since this is only the statement, we'll use sorry to skip the proof.
  sorry
}

end matrix_multiplication_result_l738_738592


namespace john_twice_as_old_in_x_years_l738_738009

def frank_is_younger (john_age frank_age : ℕ) : Prop :=
  frank_age = john_age - 15

def frank_future_age (frank_age : ℕ) : ℕ :=
  frank_age + 4

def john_future_age (john_age : ℕ) : ℕ :=
  john_age + 4

theorem john_twice_as_old_in_x_years (john_age frank_age x : ℕ) 
  (h1 : frank_is_younger john_age frank_age)
  (h2 : frank_future_age frank_age = 16)
  (h3 : john_age = frank_age + 15) :
  (john_age + x) = 2 * (frank_age + x) → x = 3 :=
by 
  -- Skip the proof part
  sorry

end john_twice_as_old_in_x_years_l738_738009


namespace quadratic_has_two_distinct_real_roots_l738_738584

def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem quadratic_has_two_distinct_real_roots 
    (a b c : ℝ) (h_eq : a = 1) (h_eq_b : b = -2) (h_eq_c : c = -6) :
  discriminant a b c > 0 :=
by {
  simp [discriminant],
  rw [h_eq, h_eq_b, h_eq_c],
  norm_num,
  -- Calculate discriminant
  exact 4 + 24 > 0,
  simp,
  sorry
}

end quadratic_has_two_distinct_real_roots_l738_738584


namespace cylinder_volume_eq_l738_738304

variable (α β l : ℝ)

theorem cylinder_volume_eq (hα_pos : 0 < α ∧ α < π/2) (hβ_pos : 0 < β ∧ β < π/2) (hl_pos : 0 < l) :
  let V := (π * l^3 * Real.sin (2 * β) * Real.cos β) / (8 * (Real.cos α)^2)
  V = (π * l^3 * Real.sin (2 * β) * Real.cos β) / (8 * (Real.cos α)^2) :=
by 
  sorry

end cylinder_volume_eq_l738_738304


namespace first_investment_value_is_500_l738_738702

variables (x : ℝ)

def combined_return_percent := 0.13
def first_investment_return_percent := 0.07
def second_investment_return_value := 0.15 * 1500
def second_investment_value := 1500

theorem first_investment_value_is_500
    (h1 : combined_return_percent * (x + second_investment_value) 
       = first_investment_return_percent * x + second_investment_return_value) :
  x = 500 :=
sorry

end first_investment_value_is_500_l738_738702


namespace exists_tangency_points_l738_738860

-- Define the structure for a triangle in the Euclidean plane
structure Triangle (P : Type) [AffineSpace P ℝ] :=
  (A B C : P)

-- Definitions for orthocenter, circumcenter, circumradius, and tangency points
variables {P : Type} [AffineSpace P ℝ]
variables (t : Triangle P)

def orthocenter (t : Triangle P) : P := sorry
def circumcenter (t : Triangle P) : P := sorry
def circumradius (t : Triangle P) : ℝ := sorry

/-- Ellipse definition with foci H and O, and major axis R tangent to sides of triangle ABC. -/
noncomputable def ellipse_with_given_properties
  (H O : P) (R : ℝ) (tangent_to : Triangle P → P → Prop) : Prop := 
sorry

/-- Tangency points on the sides of the triangle -/
def tangency_points (H O : P) (R : ℝ) (t : Triangle P) : Prop :=
  ∃ P1 P2 P3 : P,
  (P1 ∈ line_segment t.B t.C) ∧ (P2 ∈ line_segment t.C t.A) ∧ (P3 ∈ line_segment t.A t.B) ∧
  ellipse_with_given_properties H O R (λ t P, P = P1 ∨ P = P2 ∨ P = P3)

/-- Main statement: proving the tangency points exist as per given conditions -/
theorem exists_tangency_points (H O : P) (R : ℝ) (t : Triangle P)
  (H_is_orthocenter : H = orthocenter t)
  (O_is_circumcenter : O = circumcenter t)
  (R_is_circumradius : R = circumradius t) :
  tangency_points H O R t :=
sorry

end exists_tangency_points_l738_738860


namespace propositions_correct_l738_738781

-- Definitions for the problem setup
variable (m n l : Line) (α β γ : Plane)

-- Given Propositions
axiom prop1 : (m ⊆ α) → (n ∥ α) → (m ∥ n)
axiom prop2 : (m ⊆ α) → (n ⊆ β) → (α ⊥ β) → (α ∩ β = l) → (m ⊥ l) → (m ⊥ n)
axiom prop3 : (n ∥ m) → (m ⊆ α) → (n ∥ α)
axiom prop4 : (α ∥ γ) → (β ∥ γ) → (α ∥ β)

-- Theorem to prove that Propositions ② and ④ are correct
theorem propositions_correct :
  (prop2 : (m ⊆ α) → (n ⊆ β) → (α ⊥ β) → (α ∩ β = l) → (m ⊥ l) → (m ⊥ n)) ∧
  (prop4 : (α ∥ γ) → (β ∥ γ) → (α ∥ β)) :=
by
  sorry -- Proof to be provided

end propositions_correct_l738_738781


namespace smallest_natural_number_with_unique_digits_divisible_by_990_l738_738362

noncomputable def is_permutation_of_0_to_9 (n : ℕ) : Prop :=
  let digits := List.ofDigits 10 n
  List.length digits = 10 ∧ 
  List.nodup digits ∧ 
  List.all digits (λ d, d ∈ (List.range' 0 10))

theorem smallest_natural_number_with_unique_digits_divisible_by_990 :
  ∀ n : ℕ, 
  (is_permutation_of_0_to_9 n ∧ n % 990 = 0) → 
  n ≥ 1234758690 :=
by
  sorry

end smallest_natural_number_with_unique_digits_divisible_by_990_l738_738362


namespace select_assistants_l738_738316

open Nat

-- Define a combination function
def combination (n k : ℕ) :=
  nat.factorial n / (nat.factorial k * nat.factorial (n - k))

-- Theorem statement: prove that the number of ways to select 3 individuals ensuring at least one of A or B is 64
theorem select_assistants (n k : ℕ) (A B : bool) :
  combination 10 3 - combination 8 3 = 64 :=
by
  sorry

end select_assistants_l738_738316


namespace concert_tickets_full_price_revenue_l738_738689

theorem concert_tickets_full_price_revenue :
  ∃ (f p d : ℕ), f + d = 200 ∧ f * p + d * (p / 3) = 2688 ∧ f * p = 2128 :=
by
  -- We need to find the solution steps are correct to establish the existence
  sorry

end concert_tickets_full_price_revenue_l738_738689


namespace find_tangent_b_l738_738332

noncomputable theory

def circle_1_center : (ℝ × ℝ) := (1, 3)
def circle_1_radius : ℝ := 3

def circle_2_center : (ℝ × ℝ) := (13, 6)
def circle_2_radius : ℝ := 6

def tangent_line (m b : ℝ) : Prop := ∀ p : ℝ × ℝ, (p.2 = m * p.1 + b) → 
  (real.dist p circle_1_center = circle_1_radius ∨ real.dist p circle_2_center = circle_2_radius)

theorem find_tangent_b (m b : ℝ) (h : m > 0) : 
  (∃ b : ℝ, tangent_line m b) → b = 0.5 := sorry

end find_tangent_b_l738_738332


namespace large_diagonal_proof_l738_738921

variable (a b : ℝ) (α : ℝ)
variable (h₁ : a < b)
variable (h₂ : 1 < a) -- arbitrary positive scalar to make obtuse properties hold

noncomputable def large_diagonal_length : ℝ :=
  Real.sqrt (a^2 + b^2 + 2 * b * (Real.cos α * Real.sqrt (a^2 - b^2 * Real.sin α^2) + b * Real.sin α^2))

theorem large_diagonal_proof
  (h₃ : 90 < α + Real.arcsin (b * Real.sin α / a)) :
  large_diagonal_length a b α = Real.sqrt (a^2 + b^2 + 2 * b * (Real.cos α * Real.sqrt (a^2 - b^2 * Real.sin α^2) + b * Real.sin α^2)) :=
sorry

end large_diagonal_proof_l738_738921


namespace inequality_proof_l738_738949

theorem inequality_proof 
  (x1 x2 y1 y2 z1 z2 : ℝ) 
  (hx1 : 0 < x1) 
  (hx2 : 0 < x2)
  (hxy1 : x1 * y1 > z1 ^ 2)
  (hxy2 : x2 * y2 > z2 ^ 2) :
  8 / ((x1 + x2) * (y1 + y2) - (z1 + z2) ^ 2) ≤
  1 / (x1 * y1 - z1 ^ 2) + 1 / (x2 * y2 - z2 ^ 2) :=
sorry

end inequality_proof_l738_738949


namespace figure_can_form_5x5_square_l738_738617

-- Define the predicate for a shape being able to form a 5x5 square
def can_form_5x5_square : Shape → Prop := λ s, 
  ∃ (parts : list Shape), parts.length = 2 ∧ 
                           (∀ p ∈ parts, is_valid_part p s) ∧ 
                           can_reassemble parts (5, 5)

-- Define the Shapes A, B, C, and D
inductive Shape
| A : Shape
| B : Shape
| C : Shape
| D : Shape

-- Define the properties is_valid_part and can_reassemble
def is_valid_part (p s: Shape) : Prop := sorry
def can_reassemble (parts : list Shape) (dimensions : ℕ × ℕ) : Prop := sorry

-- Statement to prove
theorem figure_can_form_5x5_square :
  can_form_5x5_square Shape.A ∧ 
  can_form_5x5_square Shape.C ∧ 
  can_form_5x5_square Shape.D :=
by sorry

end figure_can_form_5x5_square_l738_738617


namespace range_of_m_l738_738764

theorem range_of_m (x m : ℝ)
  (h1 : (x + 2) / (10 - x) ≥ 0)
  (h2 : x^2 - 2 * x + 1 - m^2 ≤ 0)
  (h3 : m < 0)
  (h4 : ∀ (x : ℝ), (x + 2) / (10 - x) ≥ 0 → (x^2 - 2 * x + 1 - m^2 ≤ 0)) :
  -3 ≤ m ∧ m < 0 :=
sorry

end range_of_m_l738_738764


namespace quadratic_inequality_solution_l738_738353

theorem quadratic_inequality_solution (x : ℝ) : (x^2 - x - 12 < 0) ↔ (-3 < x ∧ x < 4) :=
by sorry

end quadratic_inequality_solution_l738_738353


namespace largest_difference_from_set_l738_738954

theorem largest_difference_from_set : 
  let s := {-10, -3, 1, 5, 7, 15} in 
  (∃ a b ∈ s, (∀ x y ∈ s, a - b ≥ x - y)) 
    ∧ (a = 15) ∧ (b = -10) → a - b = 25 := by
  sorry

end largest_difference_from_set_l738_738954


namespace quadratic_inequality_solution_l738_738354

theorem quadratic_inequality_solution (x : ℝ) : (x^2 - x - 12 < 0) ↔ (-3 < x ∧ x < 4) :=
by sorry

end quadratic_inequality_solution_l738_738354


namespace find_t_find_max_x_find_range_a_l738_738417

-- Definition of the function f(x)
def f (x t : ℝ) : ℝ := 1 / 2 * (t * Real.log (x + 2) - Real.log (x - 2))

-- Condition: f(x) ≥ f(4)
def condition_f_ge_f4 (x t : ℝ) : Prop := f x t ≥ f 4 t

-- Finding the value of t
theorem find_t (t : ℝ) : (∀ x : ℝ, condition_f_ge_f4 x t) → t = 3 := sorry

-- Interval bounds for x
def interval (x : ℝ) : Prop := 3 ≤ x ∧ x ≤ 7

-- Finding the value of x where f(x) reaches its maximum on [3, 7]
theorem find_max_x (t : ℝ) (h_t : t = 3) (h_interval : ∀ x, interval x → f x t) : f x t = f 7 t := sorry

-- Function F(x)
def F (x a t : ℝ) : ℝ := a * Real.log (x - 1) - f x t

-- Monotonicity condition of F(x)
def monotonically_increasing_F (x a t : ℝ) : Prop := 
  ∀ x > 2, (a / (x - 1) - (x - 4) / ((x + 2) * (x - 2))) ≥ 0

-- Finding the range of a
theorem find_range_a (a t : ℝ) (h_t : t = 3) : 
  (∀ x > 2, monotonically_increasing_F x a t) → 1 ≤ a := sorry

end find_t_find_max_x_find_range_a_l738_738417


namespace minimum_handshakes_l738_738271

-- Definitions
def people : ℕ := 30
def handshakes_per_person : ℕ := 3

-- Theorem statement
theorem minimum_handshakes : (people * handshakes_per_person) / 2 = 45 :=
by
  sorry

end minimum_handshakes_l738_738271


namespace smallest_number_of_marbles_l738_738319

theorem smallest_number_of_marbles (r w b g y n : ℕ) :
  (r + w + b + g + y = n) →
  (∑ i in (finset.range n).powerset.filter (λ s, s.card = 4 ∧ s.contains r), i) = (∑ i in (finset.range n).powerset.filter (λ s, s.card = 4 ∧ s.contains (w ∧ r)), i) →
  (∑ i in (finset.range n).powerset.filter (λ s, s.card = 4 ∧ s.contains (w ∧ b ∧ y ∧ r)), i) = (∑ i in (finset.range n).powerset.filter (λ s, s.card = 4 ∧ s.contains (w ∧ b ∧ g ∧ r)), i) →
  n = 11 :=
by
  sorry

end smallest_number_of_marbles_l738_738319


namespace intervals_of_monotonicity_and_extrema_l738_738789

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 - 12 * x + 3

theorem intervals_of_monotonicity_and_extrema :
  (∀ x, f'(x) = 6 * (x + 2) * (x - 1)) →
  (∀ x, x < -2 → f'(x) > 0) →
  (∀ x, x > 1 → f'(x) > 0) →
  (∀ x, -2 < x ∧ x < 1 → f'(x) < 0) →
  (∀ x ∈ Icc (-3 : ℝ) 3, f(x) ≤ 48 ∧ f(x) ≥ -4) :=
by
  intro derivative
  intro increasing1
  intro increasing2
  intro decreasing
  intro boundedness
  sorry

end intervals_of_monotonicity_and_extrema_l738_738789


namespace evaluate_expression_l738_738713

theorem evaluate_expression (x : ℤ) (h : x = 2) : 20 - 2 * (3 * x^2 - 4 * x + 8) = -4 :=
by
  rw [h]
  sorry

end evaluate_expression_l738_738713


namespace probability_even_sum_l738_738599

theorem probability_even_sum :
  let S := {1, 2, 3, 4, 5, 6}
  let P := set.prod S S
  let distinct_pairs := {p : P // p.fst ≠ p.snd}
  let total_pairs := finset.card distinct_pairs
  let even_sum_pairs := {p ∈ distinct_pairs | (p.fst + p.snd) % 2 = 0 }
  let favorable_pairs := finset.card even_sum_pairs
  let probability := (favorable_pairs : ℚ) / (total_pairs : ℚ)
  probability = 2 / 5 :=
sorry

end probability_even_sum_l738_738599


namespace binomial_constant_term_l738_738899

theorem binomial_constant_term :
  (∃ r : ℕ, 8 - 2 * r = 0 ∧ ∑ (k : ℕ) in finset.range (8+1), (binomial 8 k) * (x^(8-k) * (- (1/x))^k))
  = 70 :=
sorry

end binomial_constant_term_l738_738899


namespace perpendicular_lines_b_value_l738_738600

theorem perpendicular_lines_b_value :
  (∀ u v : ℝ, (u * 5 + v * (-3)) = 0 → v = 4 → u = 12 / 5) :=
begin
  intros u v h1 h2,
  rw h2 at h1,
  norm_num at h1,
  linarith,
end

#check perpendicular_lines_b_value

end perpendicular_lines_b_value_l738_738600


namespace hyperbola_eccentricity_l738_738336

theorem hyperbola_eccentricity :
  ∀ (a b c : ℝ) (x y : ℝ), 
  (a > 0) → 
  (b > 0) → 
  (c = real.sqrt (a^2 + b^2)) →
  (∀ (M F₁ F₂ : ℝ × ℝ), 
    ((a : ℝ) * (a : ℝ) * x^2) - ((b : ℝ) * (b : ℝ) * y^2) = (a : ℝ) * (a : ℝ) → 
    (F₁ = (real.sqrt (a^2 + b^2), 0)) → 
    (F₂ = (-real.sqrt (a^2 + b^2), 0)) → 
    (M = (a, (b : ℝ) / real.tan (real.pi / 6))) →
    (dist M F₂ = (b : ℝ) / real.tan (real.pi / 6))) →
    (ℝ.sqrt (c^2 - a^2)) / a = real.sqrt 3 :=
sorry

end hyperbola_eccentricity_l738_738336


namespace tin_amount_in_new_mixture_l738_738659

def mass_alloy_A : ℝ := 160
def mass_alloy_B : ℝ := 210
def mass_alloy_C : ℝ := 120

def ratio_A_lead_tin : ℝ := 2 / 5
def ratio_A_tin : ℝ := 3 / 5

def ratio_B_tin : ℝ := 3 / 7
def ratio_C_tin : ℝ := 2 / 7

def tin_A : ℝ := mass_alloy_A * ratio_A_tin
def tin_B : ℝ := mass_alloy_B * ratio_B_tin
def tin_C : ℝ := mass_alloy_C * ratio_C_tin

def total_tin : ℝ := tin_A + tin_B + tin_C

theorem tin_amount_in_new_mixture : total_tin = 220.29 := by
  sorry

end tin_amount_in_new_mixture_l738_738659


namespace john_treats_patients_per_year_l738_738842

theorem john_treats_patients_per_year :
  (let 
    patients_first_hospital_per_day := 20,
    patients_second_hospital_per_day := patients_first_hospital_per_day + (20 / 100 * patients_first_hospital_per_day),
    days_per_week := 5,
    weeks_per_year := 50,
    patients_first_hospital_per_week := patients_first_hospital_per_day * days_per_week,
    patients_second_hospital_per_week := patients_second_hospital_per_day * days_per_week,
    total_patients_per_week := patients_first_hospital_per_week + patients_second_hospital_per_week,
    total_patients_per_year := total_patients_per_week * weeks_per_year
  in total_patients_per_year = 11000) :=
by 
  sorry

end john_treats_patients_per_year_l738_738842


namespace pradeep_failed_by_25_marks_l738_738530

theorem pradeep_failed_by_25_marks (passing_percentage : ℕ) (pradeep_marks : ℕ) (max_marks : ℕ) :
  passing_percentage = 25 -> pradeep_marks = 185 -> max_marks = 840 -> 
  (max_marks * passing_percentage / 100) - pradeep_marks = 25 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  -- Rest of the proof is omitted
  sorry

end pradeep_failed_by_25_marks_l738_738530


namespace division_quotient_l738_738876

theorem division_quotient (dividend divisor remainder quotient : Nat) 
  (h_dividend : dividend = 109)
  (h_divisor : divisor = 12)
  (h_remainder : remainder = 1)
  (h_division_equation : dividend = divisor * quotient + remainder)
  : quotient = 9 := 
by
  sorry

end division_quotient_l738_738876


namespace rectangle_area_is_correct_l738_738906

noncomputable def area_of_rectangle := 
  let E := 7 in
  let H := 2 in
  let D := 8 in
  let F := H + E in
  let B := H + 14 in
  let I := 2 * H + E in
  let G := 3 * H + E in
  let C := 3 * H + D + E in
  let A := 3 * H + 2 * D + E in
  let width := A in
  let height := H + 14 in
  width * height

theorem rectangle_area_is_correct :
  area_of_rectangle = 464 :=
by
  -- skipping the proof
  sorry

end rectangle_area_is_correct_l738_738906


namespace minimum_value_of_f_l738_738745

noncomputable def f (x : ℝ) : ℝ := x^2 / (x - 5)

theorem minimum_value_of_f : ∃ (x : ℝ), x > 5 ∧ f x = 20 :=
by
  use 10
  sorry

end minimum_value_of_f_l738_738745


namespace perimeter_of_equilateral_triangle_l738_738557

theorem perimeter_of_equilateral_triangle (s : ℝ) 
  (h1 : (s ^ 2 * Real.sqrt 3) / 4 = 2 * s) : 
  3 * s = 8 * Real.sqrt 3 :=
by
  sorry

end perimeter_of_equilateral_triangle_l738_738557


namespace induction_even_l738_738403

theorem induction_even (k : ℕ) (h_k_even : Even k) (h_k_ge_2 : k ≥ 2) 
(h_prop_k : 1 - (1/2) + (1/3) - (1/4) + ... + (1/(k+1)) = 2 * ((1/(k+2)) + (1/(k+4)) + ... + (1/(2*k)))) : 
1 - (1/2) + (1/3) - (1/4) + ... + (1/(k+3)) = 2 * ((1/(k+4)) + (1/(k+6)) + ... + (1/(2*(k+2)))) := 
sorry

end induction_even_l738_738403


namespace quadratic_roots_relationship_l738_738618

noncomputable def quadratic_roots_condition (x1 x2 m k : ℝ) : Prop :=
  x1 + x2 + x1 * x2 = 2 * m + k ∧
  (x1 - 1) * (x2 -1) = m + 1 - k ∧
  x1 - x2 = 1 ∧
  k - m = 1

theorem quadratic_roots_relationship (x1 x2 m k : ℝ) (h : quadratic_roots_condition x1 x2 m k) :
  k^2 = 4 * m + 1 ∧
  (m,x1,x2 = 0,1,0 ∨ m,x1,x2 = 2,2,1) :=
begin
  sorry
end

end quadratic_roots_relationship_l738_738618


namespace part1_part2_l738_738664

-- Definition of initial conditions
def option1_price : ℕ := 30
def option2_price : ℕ := 50
def option3_price : ℕ := 70

def initial_people_option1 : ℕ := 20000
def initial_people_option2 : ℕ := 10000
def initial_people_option3 : ℕ := 10000

def decrease_effect_option1 (x : ℕ) : ℕ := 400 * x
def decrease_effect_option2 (x : ℕ) : ℕ := 600 * x

-- Part 1 proof
theorem part1 (x : ℕ) (h : x = 5) :
  let new_people_option1 := (initial_people_option1 - decrease_effect_option1 x) / 10000 in
  let new_people_option2 := (initial_people_option2 - decrease_effect_option2 x) / 10000 in
  let new_people_option3 := (initial_people_option3 + decrease_effect_option1 x + decrease_effect_option2 x) / 10000 in
  let new_revenue := new_people_option1 * option1_price + new_people_option2 * option2_price + (option3_price - x) * new_people_option3 in
  new_people_option1 = 1.8 ∧ new_people_option2 = 0.7 ∧ new_people_option3 = 1.5 ∧ new_revenue = 186.5 :=
by sorry

-- Part 2 proof
theorem part2 :
  ∃ x w, let revenue := -0.1 * (x - 9)^2 + 188.1 in
  revenue = w ∧ (∀ w', w' < 188.1) ∧ (∀ x', x' = 9 → (70 - x') = 61) :=
by sorry

end part1_part2_l738_738664


namespace unique_strictly_increasing_function_l738_738658

-- Define the strictly increasing function f: ℕ → ℕ
def f (n : ℕ) : ℕ := sorry 

-- The main theorem stating that the only strictly increasing function
-- satisfying the given conditions is f(k) = k.
theorem unique_strictly_increasing_function (f : ℕ → ℕ) (h1 : ∀ m n : ℕ, m.coprime n → f (m * n) = f m * f n)
  (h2 : f 2 = 2) (h3 : ∀ x y : ℕ, x < y → f x < f y) : ∀ k : ℕ, f k = k :=
sorry

end unique_strictly_increasing_function_l738_738658


namespace river_length_GSA_AWRA_l738_738641

-- Define the main problem statement
noncomputable def river_length_estimate (GSA_length AWRA_length GSA_error AWRA_error error_prob : ℝ) : Prop :=
  (GSA_length = 402) ∧ (AWRA_length = 403) ∧ 
  (GSA_error = 0.5) ∧ (AWRA_error = 0.5) ∧ 
  (error_prob = 0.04) ∧ 
  (abs (402.5 - GSA_length) ≤ GSA_error) ∧ 
  (abs (402.5 - AWRA_length) ≤ AWRA_error) ∧ 
  (error_prob = 1 - (2 * 0.02))

-- The main theorem statement
theorem river_length_GSA_AWRA :
  river_length_estimate 402 403 0.5 0.5 0.04 :=
by
  sorry

end river_length_GSA_AWRA_l738_738641


namespace adjacent_sum_eq_98_l738_738576

theorem adjacent_sum_eq_98 :
  let divisors := [2, 4, 5, 7, 8, 10, 14, 20, 28, 35, 40, 56, 70, 140, 280] in
  let valid_adjacents := [7, 28, 70] in
  ∃ (a b : ℕ), a ∈ valid_adjacents ∧ b ∈ valid_adjacents ∧ a ≠ b ∧ gcd a 14 > 1 ∧ gcd b 14 > 1 ∧ a + b = 98 :=
begin
  sorry  
end

end adjacent_sum_eq_98_l738_738576


namespace min_max_value_expression_l738_738501

theorem min_max_value_expression
  (x1 x2 x3 : ℝ) 
  (hx : x1 + x2 + x3 = 1)
  (hx1 : 0 ≤ x1)
  (hx2 : 0 ≤ x2)
  (hx3 : 0 ≤ x3) :
  (x1 + 3 * x2 + 5 * x3) * (x1 + x2 / 3 + x3 / 5) = 1 := 
sorry

end min_max_value_expression_l738_738501


namespace nth_equation_l738_738873

theorem nth_equation (n : ℕ) : 
  1 + 6 * n = (3 * n + 1) ^ 2 - 9 * n ^ 2 := 
by 
  sorry

end nth_equation_l738_738873


namespace price_comparison_l738_738325

def initial_price := 1 -- ruble per kg
def feb_27_price_vintiki := initial_price + 0.5 * initial_price
def feb_27_price_shpuntiki := initial_price - 0.5 * initial_price
def mar_price_vintiki := feb_27_price_vintiki - 0.5 * feb_27_price_vintiki
def mar_price_shpuntiki := feb_27_price_shpuntiki + 0.5 * feb_27_price_shpuntiki
def mar_price_gaecki := initial_price

theorem price_comparison :
  (mar_price_gaecki = 1) ∧ (mar_price_vintiki = 0.75) ∧ (mar_price_shpuntiki = 0.75) :=
by
  sorry

end price_comparison_l738_738325


namespace num_incorrect_propositions_is_four_l738_738800

noncomputable def count_incorrect_propositions (a b : Line) (alpha beta : Plane) : Nat :=
  let prop1 := a.parallel b ∧ a.parallel alpha → b.parallel alpha
  let prop2 := a.perp b ∧ a.perp alpha → b.parallel alpha
  let prop3 := a.parallel alpha ∧ beta.parallel alpha → a.parallel beta
  let prop4 := a.perp alpha ∧ beta.perp alpha → a.parallel beta
  ((¬prop1) + (¬prop2) + (¬prop3) + (¬prop4) : Nat)

theorem num_incorrect_propositions_is_four (a b : Line) (alpha beta : Plane) : count_incorrect_propositions a b alpha beta = 4 :=
  sorry

end num_incorrect_propositions_is_four_l738_738800


namespace solve_for_x_l738_738544

theorem solve_for_x (x : ℝ) (h : (5 * x - 3) / (6 * x - 6) = (4 / 3)) : x = 5 / 3 :=
sorry

end solve_for_x_l738_738544


namespace range_f_eq_real_range_g_ne_real_l738_738163

noncomputable def f (x : ℝ) : ℝ := Real.log 3 (-x)
noncomputable def g (x : ℝ) : ℝ := 3^(-x)

theorem range_f_eq_real : set.range f = set.univ := sorry
theorem range_g_ne_real : set.range g ≠ set.univ := sorry

end range_f_eq_real_range_g_ne_real_l738_738163


namespace range_of_m_l738_738914

theorem range_of_m (m : ℝ) : (-1 : ℝ) ≤ m ∧ m ≤ 3 ∧ ∀ x y : ℝ, x - ((m^2) - 2 * m + 4) * y - 6 > 0 → (x, y) ≠ (-1, -1) := 
by sorry

end range_of_m_l738_738914


namespace inverse_sum_mod_l738_738226

theorem inverse_sum_mod (h1 : ∃ k, 3^6 ≡ 1 [MOD 17])
                        (h2 : ∃ k, 3 * 6 ≡ 1 [MOD 17]) : 
  (6 + 9 + 2 + 1 + 6 + 1) % 17 = 8 :=
by
  cases h1 with k1 h1
  cases h2 with k2 h2
  sorry

end inverse_sum_mod_l738_738226


namespace sasha_took_right_triangle_l738_738210

-- Define types of triangles
inductive Triangle
| acute
| right
| obtuse

open Triangle

-- Define the function that determines if Borya can form a triangle identical to Sasha's
def can_form_identical_triangle (t1 t2 t3: Triangle) : Bool :=
match t1, t2, t3 with
| right, acute, obtuse => true
| _ , _ , _ => false

-- Define the main theorem
theorem sasha_took_right_triangle : 
  ∀ (sasha_takes borya_takes1 borya_takes2 : Triangle),
  (sasha_takes ≠ borya_takes1 ∧ sasha_takes ≠ borya_takes2 ∧ borya_takes1 ≠ borya_takes2) →
  can_form_identical_triangle sasha_takes borya_takes1 borya_takes2 →
  sasha_takes = right :=
by sorry

end sasha_took_right_triangle_l738_738210


namespace find_p_q_sum_l738_738918

theorem find_p_q_sum (p q : ℝ) 
  (sum_condition : p / 3 = 8) 
  (product_condition : q / 3 = 12) : 
  p + q = 60 :=
by
  sorry

end find_p_q_sum_l738_738918


namespace angle_460_in_second_quadrant_l738_738282

theorem angle_460_in_second_quadrant : (460 % 360) ≥ 90 ∧ (460 % 360) < 180 :=
by
  have h : 460 % 360 = 100 := by norm_num
  rw h
  split
  norm_num
  norm_num
  sorry

end angle_460_in_second_quadrant_l738_738282


namespace candle_remaining_height_half_time_l738_738335

noncomputable def total_burn_time : ℕ → ℝ
| 0       := 0
| (n + 1) := total_burn_time n + 10 * (n + 1) * Real.sqrt (n + 1)

noncomputable def remaining_height (n : ℕ) (t : ℝ) : ℕ → ℕ
| 0       := n
| (k + 1) := if total_burn_time (n - (k + 1)) ≤ t then remaining_height (n - 1) t else n

theorem candle_remaining_height_half_time :
  let T := total_burn_time 150 / 2 in
  remaining_height 150 T 150 = 52 :=
by
  sorry

end candle_remaining_height_half_time_l738_738335


namespace find_range_of_values_l738_738025

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

theorem find_range_of_values (f : ℝ → ℝ) (h_even : is_even f)
  (h_increasing : is_increasing_on_nonneg f) (h_f1_zero : f 1 = 0) :
  { x : ℝ | f (Real.log x / Real.log (1/2)) > 0 } = 
  { x : ℝ | 0 < x ∧ x < 1/2 } ∪ { x : ℝ | x > 2 } :=
by 
  sorry

end find_range_of_values_l738_738025


namespace problem_statement_l738_738168

variable {n : ℕ} (x : Fin n.succ → ℝ)
noncomputable def seq_sum (x : Fin n.succ → ℝ) : ℝ := 
  Finset.sum (Finset.range n.succ) (λ i, (x i) ^ 2 / ((x i) ^ 2 + x (i.succ % n.succ) * x (i.succ.succ % n.succ)))

theorem problem_statement (h1 : 2 ≤ n.succ) (h2 : ∀ i, 0 < x i) : 
  seq_sum x ≤ n - 1 := sorry

end problem_statement_l738_738168


namespace hyperbola_eccentricity_l738_738757

theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : let F1 := (c - a, 0) in 
        let F2 := (c + a, 0) in 
        let P := (x, sqrt(1 + (x^2) / (a^2)) - y * (b^2) / (a^2)) in
        ∠ F1 P F2 = 60)
  (h_area : (let F1 := (c - a, 0) in
             let F2 := (c + a, 0) in
             let P := (x, sqrt(1 + (x^2) / (a^2)) - y * (b^2) / (a^2)) in
             abs(1/2 * x * y * sin(60)) = abs(sqrt(3) * a * c)))
  : e = (1 + sqrt(5)) / 2 :=
sorry

end hyperbola_eccentricity_l738_738757


namespace regression_line_eq_l738_738043

/-- Given conditions -/
variables (slope : ℝ) (center_x center_y : ℝ) (intercept : ℝ)

/-- Assumptions -/
axiom slope_eq : slope = 1.23
axiom center_eq : center_x = 4 ∧ center_y = 5
axiom regression_eq : center_y = slope * center_x + intercept

/-- Proof statement -/
theorem regression_line_eq : intercept = 0.08 ∧ ∀ x, 1.23 * x + 0.08 = slope * x + intercept :=
begin
  split,
  {
    rw [slope_eq, center_eq.1, center_eq.2] at regression_eq,
    norm_num at regression_eq,
    exact regression_eq,
  },
  {
    intro x,
    rw slope_eq,
    simp,
  }
end

-- Skip the proof using sorry for parts of complex proofs
-- Please note: The above proof is included for completeness but may utilize simplification steps
-- that the actual problem and conditions imply.


end regression_line_eq_l738_738043


namespace effective_percentage_loss_approx_l738_738690

theorem effective_percentage_loss_approx 
    (total_apples : ℕ)
    (sold_at_20_pct : ℝ)
    (sold_at_30_pct : ℝ)
    (sold_at_40_pct : ℝ)
    (sold_at_35_pct : ℝ)
    (unsold_pct : ℝ)
    (storage_expense : ℝ)
    (packaging_expense : ℝ)
    (transportation_expense : ℝ)
    (cost_price_per_kg : ℝ)
    (effective_loss_percentage : ℝ) :
  total_apples = 150 →
  sold_at_20_pct = 0.30 →
  sold_at_30_pct = 0.25 →
  sold_at_40_pct = 0.15 →
  sold_at_35_pct = 0.10 →
  unsold_pct = 0.20 →
  storage_expense = 15 →
  packaging_expense = 10 →
  transportation_expense = 25 →
  cost_price_per_kg = 1 →
  effective_loss_percentage ≈ -32.83 :=
by 
  -- conditions to establish
  sorry

end effective_percentage_loss_approx_l738_738690


namespace colonization_combinations_l738_738806

-- Definitions of conditions
def num_Earth_like := 6
def num_Mars_like := 6
def Earth_cost := 3
def Mars_cost := 1
def total_resources := 18

-- Required proof: The number of different combinations of planets that can be colonized
theorem colonization_combinations : num_comb := 136 := by
  sorry

end colonization_combinations_l738_738806


namespace abs_diff_inequality_l738_738094

theorem abs_diff_inequality (m : ℝ) : (∃ x : ℝ, |x + 2| - |x + 3| > m) ↔ m < -1 :=
sorry

end abs_diff_inequality_l738_738094


namespace percentage_of_students_with_cars_l738_738591

def seniors : ℕ := 300
def lower_grades : ℕ := 900
def percent_seniors_with_cars : ℝ := 0.50
def percent_lower_grades_with_cars : ℝ := 0.10

def total_students := seniors + lower_grades
def seniors_with_cars := percent_seniors_with_cars * seniors
def lower_grades_with_cars := percent_lower_grades_with_cars * lower_grades
def total_students_with_cars := seniors_with_cars + lower_grades_with_cars
def percent_students_with_cars := (total_students_with_cars / total_students.to_nat) * 100

theorem percentage_of_students_with_cars :
  percent_students_with_cars = 20 :=
by
  sorry

end percentage_of_students_with_cars_l738_738591


namespace max_value_of_f_l738_738004

noncomputable def f (x : ℝ) : ℝ :=
  min (3 * x + 3) (min (-x / 3 + 3) (x / 3 + 9))

theorem max_value_of_f : ∃ x : ℝ, f x = 6 :=
by
  sorry

end max_value_of_f_l738_738004


namespace lean_proof_l738_738260

variables (ABCDEF : Type) [inscribed_in_circle ABCDEF]
variables (A B C D E F : ABCDEF)
variables (AB BC CD DE EF FA : ℝ) (AD : ℝ)
variables (cos_angle_B cos_angle_ADF : ℝ)

-- Conditions
axiom AB_eq_3 : AB = 3
axiom BC_eq_3 : BC = 3
axiom CD_eq_3 : CD = 3
axiom DE_eq_3 : DE = 3
axiom EF_eq_3 : EF = 3
axiom FA_eq_3 : FA = 3
axiom AD_eq_2 : AD = 2

-- Cosine of angles
axiom cos_B : cos_angle_B = 7 / 9
axiom cos_ADF : cos_angle_ADF = 3 / 4

theorem lean_proof : 
  (1 - cos_angle_B) * (1 - cos_angle_ADF) = 1 / 18 :=
by
  sorry

end lean_proof_l738_738260


namespace points_on_plane_l738_738015

theorem points_on_plane (n m : ℕ) (h1 : m ≤ (n - 1) / 2):
  ∃ (V : fin (m + 1) → ℝ × ℝ), ∀ i : fin m, True := sorry

end points_on_plane_l738_738015


namespace determine_number_of_terms_l738_738393

variable (a1 d n : ℕ)

-- Condition 1: Sum of the first 4 terms is 40
def sum_first_4 : Prop := 2 * (2 * a1 + 3 * d) = 40

-- Condition 2: Sum of the last 4 terms is 80
def sum_last_4 : Prop := 2 * (a1 + (n - 4) * d) + 6 * d = 80

-- Condition 3: Sum of all terms is 210
def sum_all_terms : Prop := n / 2 * (2 * a1 + (n - 1) * d) = 210

-- The property to prove: n = 14
theorem determine_number_of_terms (a1 d : ℕ) :
  sum_first_4 a1 d n ∧ sum_last_4 a1 d n ∧ sum_all_terms a1 d n → n = 14 := by
sorry

end determine_number_of_terms_l738_738393


namespace sum_a_n_l738_738002

def a_n (n : ℕ) : ℕ :=
  if n % 90 = 0 then 15
  else if n % 36 = 0 then 16
  else if n % 60 = 0 then 17
  else 0

theorem sum_a_n :
  (∑ n in finset.range 3000, a_n (n + 1)) = 2673 :=
by
  sorry

end sum_a_n_l738_738002


namespace center_on_line_chord_length_condition_find_m_for_chord_length_max_angle_condition_l738_738788

-- Define the center of the circle
def center_of_circle (m : ℝ) : ℝ × ℝ := (m, -m)

-- Define the equation of the circle
def circle_eq (m x y : ℝ) : Prop := x^2 + y^2 - 2*m*x + 2*m*y = 4 - 2*m^2

-- Problem 1: The center lies on the line x + y = 0
theorem center_on_line (m : ℝ) : (center_of_circle m).1 + (center_of_circle m).2 = 0 := by
  sorry

-- Problem 2: The intercepted chord length condition
theorem chord_length_condition (m : ℝ) (h : ∀ x y, circle_eq m x y) (d : ℝ) 
  (chord_length : ℝ) (line_eq : ℝ × ℝ → Prop) : Prop :=
  line_eq (m, -m) ∧ chord_length = 2 * Real.sqrt (4 - (d / Real.sqrt 2)^2)

theorem find_m_for_chord_length (m : ℝ)
  (h : ∀ x y, circle_eq m x y)
  (line_eq : ℝ × ℝ → Prop := λ p, p.1 - p.2 + 4 = 0) :
  chord_length_condition m h 2 2sqrt(2) line_eq → m = -2 + Real.sqrt 2 := by
  sorry

-- Problem 3: Maximum angle condition at 90 degrees
theorem max_angle_condition (m : ℝ) 
  (P : ℝ × ℝ := (Real.sqrt 2, Real.sqrt 2)) 
  (hP : circle_eq m P.1 P.2) : ∃ A B, 
  (P.1, P.2) ∈ circle_eq m A B ∧ (angle <| A P B) = 90° → m = 0 := by
  sorry

end center_on_line_chord_length_condition_find_m_for_chord_length_max_angle_condition_l738_738788


namespace calculate_expr_l738_738714

variable (x y : ℝ)
variable (hx : x ≠ 0)
variable (hy : y ≠ 0)

theorem calculate_expr : ((x^3 * y^2)^2 * (x / y^3)) = x^7 * y :=
by sorry

end calculate_expr_l738_738714


namespace part_I_part_II_l738_738776

variable {a x : ℝ}

-- Conditions
def p := ∀ x : ℝ, MonotonicIncreasing (fun x => log a (x + 1))
def q := ∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x + 1 > 0

-- Prove (I)
theorem part_I (h_q : q) : 2 ≤ a ∧ a < 3 := sorry

-- Prove (II)
theorem part_II (h_p : p) (h_not_q : ¬ q) : (1 < a ∧ a < 2) ∨ (3 ≤ a) := sorry

end part_I_part_II_l738_738776


namespace check_propositions_l738_738752

-- Conditions for the problem
variables {α : Type*} [linear_order α] [decidable_linear_order α] [decidable_eq α] {D : set α} (f : α → α)

-- Domain
def D := set.Icc (-2) 2

-- Proposition definitions
def P1 : Prop := (f (-1) = f 1) ∧ (f (-2) = f 2) → ∀ x ∈ D, f (-x) = f x
def P2 : Prop := (∀ x ∈ D, f (-x) + f x = 0) → ∀ x ∈ D, f (-x) = -f x
def P3 : Prop := (∀ x y ∈ D, x < y → f x ≤ f y ∨ f y ≤ f x) ∧ (f 0 > f 1) → ∀ x y ∈ D, x < y → f x ≥ f y
def P4 : Prop := (f (-1) < f 0) ∧ (f 0 < f 1) ∧ (f 1 < f 2) → ∀ x y ∈ D, x < y → f x ≤ f y

-- Complete the proof
theorem check_propositions : P1 = false ∧ P2 = true ∧ P3 = true ∧ P4 = false :=
by
  have h1 : P1 = false := sorry,
  have h2 : P2 = true := sorry,
  have h3 : P3 = true := sorry,
  have h4 : P4 = false := sorry,
  exact ⟨h1, h2, h3, h4⟩

end check_propositions_l738_738752


namespace returning_players_count_l738_738582

def total_players_in_team (groups : ℕ) (players_per_group : ℕ): ℕ := groups * players_per_group
def returning_players (total_players : ℕ) (new_players : ℕ): ℕ := total_players - new_players

theorem returning_players_count
    (new_players : ℕ)
    (groups : ℕ)
    (players_per_group : ℕ)
    (total_players : ℕ := total_players_in_team groups players_per_group)
    (returning_players_count : ℕ := returning_players total_players new_players):
    new_players = 4 ∧
    groups = 2 ∧
    players_per_group = 5 → 
    returning_players_count = 6 := by
    intros h
    sorry

end returning_players_count_l738_738582


namespace average_player_time_l738_738520

theorem average_player_time:
  let pg := 130
  let sg := 145
  let sf := 85
  let pf := 60
  let c := 180
  let total_secs := pg + sg + sf + pf + c
  let total_mins := total_secs / 60
  let num_players := 5
  let avg_mins_per_player := total_mins / num_players
  avg_mins_per_player = 2 :=
by
  sorry

end average_player_time_l738_738520


namespace min_handshakes_l738_738280

theorem min_handshakes 
  (people : ℕ) 
  (handshakes_per_person : ℕ) 
  (total_people : people = 30) 
  (handshakes_rule : handshakes_per_person = 3) 
  (unique_handshakes : people * handshakes_per_person % 2 = 0) 
  (multiple_people : people > 0):
  (people * handshakes_per_person / 2) = 45 :=
by
  sorry

end min_handshakes_l738_738280


namespace math_problem_l738_738547

variable (a b c : ℝ)

theorem math_problem (h1 : -10 ≤ a ∧ a < 0) (h2 : 0 < a ∧ a < b ∧ b < c) : 
  (a * c < b * c) ∧ (a + c < b + c) ∧ (c / a > 1) :=
by
  sorry

end math_problem_l738_738547


namespace problem_f_neg2_l738_738061

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x ^ 2007 + b * x + 1

theorem problem_f_neg2 (a b : ℝ) (h : f a b 2 = 2) : f a b (-2) = 0 :=
by
  sorry

end problem_f_neg2_l738_738061


namespace sum_of_inverses_mod_17_l738_738229

theorem sum_of_inverses_mod_17 :
  (3⁻¹ + 3⁻² + 3⁻³ + 3⁻⁴ + 3⁻⁵ + 3⁻⁶ : ℤ) % 17 = 7 :=
by {
  sorry
}

end sum_of_inverses_mod_17_l738_738229


namespace not_all_even_numbers_greater_than_1000_representable_l738_738478

theorem not_all_even_numbers_greater_than_1000_representable :
  ∃ E, E > 1000 ∧ even E ∧ ¬ ∃ (m n : ℕ), E = n*(n+1)*(n+2) - m*(m+1) :=
by
  sorry

end not_all_even_numbers_greater_than_1000_representable_l738_738478


namespace find_coprime_pairs_l738_738737

theorem find_coprime_pairs (k n : ℕ) (h1 : n > 0) : 
  n.coprime (k - 1) ∧ n ∣ (k^n - 1) ↔ k = 1 :=
by 
  sorry

end find_coprime_pairs_l738_738737


namespace rearranged_digits_sum_eq_l738_738363

theorem rearranged_digits_sum_eq :
  (∑ n in {x | x ∈ list.permutations [1, 2, 3, 4]}.to_finset, list.foldl (+) 0 (list.zip_with (*) n [1000, 100, 10, 1])) = 66660 :=
sorry

end rearranged_digits_sum_eq_l738_738363


namespace max_value_proof_l738_738880

noncomputable def max_value (x y z : ℝ) : ℝ :=
  1 / x + 2 / y + 3 / z

theorem max_value_proof (x y z : ℝ) (h1 : 2 / 5 ≤ z ∧ z ≤ min x y)
    (h2 : x * z ≥ 4 / 15) (h3 : y * z ≥ 1 / 5) : max_value x y z ≤ 13 := 
by
  sorry

end max_value_proof_l738_738880


namespace light_ray_line_equation_l738_738991

theorem light_ray_line_equation :
  ∃ (k : ℚ), let L := λ x, k * (x + 3) + 3 in
             (∃ (m : ℚ), (m = 1 ∧ dist (2, -2) (x, k * (x + 3) + 3) = 1) ∨ 
             (L = λ x, (-3/4) * (x + 3) + 3) ∨ 
             (L = λ x, (-4/3) * (x + 3) + 3)) →
  (L = λ x, (-(3/4)) * (x + 1) + 1 → 3*x + 4*L(y) - 3 = 0) ∧ 
  (L = λ x, (-(4/3)) * (x + 1) + 1 → 4*x + 3*L(y) + 3 = 0) :=
begin
  sorry
end

end light_ray_line_equation_l738_738991


namespace xiaoming_grandfather_age_l738_738685

def grandfather_age (x xm_diff : ℕ) :=
  xm_diff = 60 ∧ x > 7 * (x - xm_diff) ∧ x < 70

theorem xiaoming_grandfather_age (x : ℕ) (h_cond : grandfather_age x 60) : x = 69 :=
by
  sorry

end xiaoming_grandfather_age_l738_738685


namespace mode_of_List_II_is_3_l738_738866

def median (lst : List ℕ) : ℕ :=
  if lst.length % 2 = 0 then
    (lst.nth! (lst.length / 2 - 1) + lst.nth! (lst.length / 2)) / 2
  else
    lst.nth! (lst.length / 2)

def mode (lst : List ℕ) : ℕ :=
  lst.foldr (λ x count_map,
    count_map.insert x (count_map.find x |>.getD 0 + 1)) Map.empty
  |> λ count_map, count_map.toList.maximumBy (λ pair, pair.2) |>.getD (0, 0) |>.1

theorem mode_of_List_II_is_3 :
  ∀ (ListI ListII : List ℕ) (y m r : ℕ),
  ListI = [y, 2, 4, 7, 10, 11] →
  ListII = [3, 3, 4, 6, 7, 10] →
  y = 9 →
  r = median (ListI.toList.set 0 y).sort →
  r = median ListII.sort + m →
  m = mode ListII →
  m = 3 :=
by
  intros ListI ListII y m r hListI hListII hy hr h_eq h_mode
  sorry

end mode_of_List_II_is_3_l738_738866


namespace Toby_change_l738_738941

def change (orders_cost per_person total_cost given_amount : ℝ) : ℝ :=
  given_amount - per_person

def total_cost (cheeseburgers milkshake coke fries cookies tax : ℝ) : ℝ :=
  cheeseburgers + milkshake + coke + fries + cookies + tax

theorem Toby_change :
  let cheeseburger_cost := 3.65
  let milkshake_cost := 2.0
  let coke_cost := 1.0
  let fries_cost := 4.0
  let cookie_cost := 3 * 0.5 -- Total cost for three cookies
  let tax := 0.2
  let total := total_cost (2 * cheeseburger_cost) milkshake_cost coke_cost fries_cost cookie_cost tax
  let per_person := total / 2
  let toby_arrival := 15.0
  change total per_person total toby_arrival = 7 :=
by
  sorry

end Toby_change_l738_738941


namespace angle_A1_F_B1_90_deg_l738_738992

theorem angle_A1_F_B1_90_deg (p : ℝ) (A B A1 B1 F : ℝ × ℝ) :
  let parabola : set (ℝ × ℝ) := { point | (point.2)^2 = 2 * p * point.1 },
      focus : ℝ × ℝ := (p / 2, 0),
      directrix := { point : (ℝ × ℝ) | point.1 = - p / 2 },
      projection1 : (ℝ × ℝ) := (-(p / 2), A.2),
      projection2 : (ℝ × ℝ) := (-(p / 2), B.2) in
  A ∈ parabola ∧ B ∈ parabola ∧ 
  A1 = projection1 ∧ B1 = projection2 ∧ F = focus →
  ∠ A1 F B1 = 90 :=
begin
  sorry
end

end angle_A1_F_B1_90_deg_l738_738992


namespace max_distance_sum_l738_738342

noncomputable def max_distance_point (A B : Point) (arc : Circle) : Point :=
  midpoint A B arc -- assuming midpoint is a predefined function that can find the midpoint of an arc

theorem max_distance_sum (A B : Point) (arc : Circle) (C : Point) : 
  C ∈ arc → (∀ D ∈ (arc \ {C}), AC + CB ≤ AD + DB) :=
begin
  sorry
end

end max_distance_sum_l738_738342


namespace C1_standard_form_C2_standard_form_max_min_dist_C1_C2_l738_738422

noncomputable def C1_parametric (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, Real.sin θ)

noncomputable def C2_parametric (t : ℝ) : ℝ × ℝ := (-3 + t, (3 + 3 * t) / 4)

theorem C1_standard_form (x y : ℝ) : 
  (∃ θ : ℝ, x = 2 * Real.cos θ ∧ y = Real.sin θ) ↔ (x^2 / 4 + y^2 = 1) := 
sorry

theorem C2_standard_form (x y : ℝ) : 
  (∃ t : ℝ, x = -3 + t ∧ y = (3 + 3 * t) / 4) ↔ (3 * x - 4 * y + 12 = 0) := 
sorry

theorem max_min_dist_C1_C2 :
  (∃ θ : ℝ, ∀ t : ℝ, 
    let d := (-| 6 * Real.cos θ - 4 * Real.sin θ + 12 | / 5)
    in (d = (12 + 2 * Real.sqrt 13) / 5 ∨ d = (12 - 2 * Real.sqrt 13) / 5)) :=
sorry

end C1_standard_form_C2_standard_form_max_min_dist_C1_C2_l738_738422


namespace power_sum_mod_inverse_l738_738232

theorem power_sum_mod_inverse (h : 3^6 ≡ 1 [MOD 17]) : 
  (3^(-1) + 3^(-2) + 3^(-3) + 3^(-4) +  3^(-5) + 3^(-6)) ≡ 1 [MOD 17] := 
by
  sorry

end power_sum_mod_inverse_l738_738232


namespace fibonacci_sum_equals_two_l738_738852

-- First, we define the Fibonacci sequence
def fibonacci : ℕ → ℕ
| 0 := 0
| 1 := 1
| (n + 2) := fibonacci (n + 1) + fibonacci n

-- Then, we define the infinite sum of the sequence
def sum_fib_div_two_pow (n : ℕ) : ℚ :=
  ∑ i in finset.range (n + 1), (fibonacci i) / (2^i : ℚ)

-- Define an assertion about the infinite sum
theorem fibonacci_sum_equals_two : 
  (∑' n, fibonacci n / (2^n : ℚ)) = 2 :=
sorry

end fibonacci_sum_equals_two_l738_738852


namespace find_a_10_l738_738920

-- Definitions and conditions from the problem
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

def S (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (n * (a 1 + a n)) / 2

variable (a : ℕ → ℕ)

-- Conditions given
axiom a_3 : a 3 = 3
axiom S_3 : S a 3 = 6
axiom arithmetic_seq : is_arithmetic_sequence a

-- Proof problem statement
theorem find_a_10 : a 10 = 10 := 
sorry

end find_a_10_l738_738920


namespace problem_part1_problem_part2_l738_738402

variable (a m : ℝ)

def prop_p (a m : ℝ) : Prop := (m - a) * (m - 3 * a) ≤ 0
def prop_q (m : ℝ) : Prop := (m + 2) * (m + 1) < 0

theorem problem_part1 (h₁ : a = -1) (h₂ : prop_p a m ∨ prop_q m) : -3 ≤ m ∧ m ≤ -1 :=
sorry

theorem problem_part2 (h₁ : ∀ m, prop_p a m → ¬prop_q m) :
  -1 / 3 ≤ a ∧ a < 0 ∨ a ≤ -2 :=
sorry

end problem_part1_problem_part2_l738_738402


namespace fuel_consumption_at_40_min_fuel_consumption_l738_738894

-- Define the fuel consumption function, distance, and given speed
def fuel_consumption (x : ℝ) : ℝ :=
  (1 / 128000) * x^3 - (3 / 80) * x + 8

def distance : ℝ := 100
def speed_40 : ℝ := 40

-- Question I: Prove the fuel consumption at 40 km/h over the distance
theorem fuel_consumption_at_40 :
  (fuel_consumption speed_40 * (distance / speed_40)) = 17.5 :=
sorry

-- Define total fuel consumed function
def total_fuel_consumed (x : ℝ) : ℝ :=
  (1 / 1280) * x^2 + (800 / x) - (15 / 4)

-- Question II: Prove the minimum fuel consumption and the corresponding speed
theorem min_fuel_consumption :
  ∃ (x : ℝ), 0 < x ∧ x ≤ 120 ∧
  (total_fuel_consumed x) = 11.25 ∧ x = 80 :=
sorry

end fuel_consumption_at_40_min_fuel_consumption_l738_738894


namespace quadratic_coefficients_l738_738579

theorem quadratic_coefficients : 
  ∀ x : ℝ, 
    let eq := 3 * x^2 + 2 * x - 4 in
    eq = 0 → 
    (let a := 3 in 
     let b := 2 in 
     let c := -4 in 
     a = 3 ∧ b = 2 ∧ c = -4) :=
begin
  sorry
end

end quadratic_coefficients_l738_738579


namespace rectangle_inscribed_circle_radius_l738_738999

theorem rectangle_inscribed_circle_radius (a b R : ℝ) (β : ℝ) 
    (h1 : ∀ A B C D O K P : Type, inscribed_in_circle_rectangle A B C D O K P a b)
    (h2 : segment_viewed_at_angle A B C D O K P β) : 
    R = a / (2 * |Real.cos β|) := 
sorry

end rectangle_inscribed_circle_radius_l738_738999


namespace fixed_point_distance_l738_738470

theorem fixed_point_distance (x y z : ℝ) (hx : |x| = 2) (hy : |y| = 2) (hz : |z| = 2) :
  real.sqrt (x^2 + y^2 + z^2) = 2 * real.sqrt 3 :=
sorry

end fixed_point_distance_l738_738470


namespace x_add_y_bounds_l738_738136

theorem x_add_y_bounds (x y : ℝ) (h1 : y = 3 * (⌊x⌋ : ℝ) + 4)
  (h2 : y = 4 * (⌊x - 3⌋ : ℝ) + 7) (hx : ⌊x⌋ ≠ x) : 
  40 < x + y ∧ x + y < 41 :=
by
  sorry

end x_add_y_bounds_l738_738136


namespace accurate_river_length_l738_738650

-- Define the given conditions
def length_GSA := 402
def length_AWRA := 403
def error_margin := 0.5
def probability_of_error := 0.04

-- State the theorem based on these conditions
theorem accurate_river_length : 
  ∀ Length_GSA Length_AWRA error_margin probability_of_error, 
  Length_GSA = 402 → 
  Length_AWRA = 403 → 
  error_margin = 0.5 → 
  probability_of_error = 0.04 → 
  (this based on independent measurements with above error margins)
  combined_length = 402.5 ∧ combined_probability_of_error = 0.04 :=
by 
  -- Proof to be completed
  sorry

end accurate_river_length_l738_738650


namespace powerjet_pump_water_l738_738552

theorem powerjet_pump_water (r : ℕ) (t : ℚ) (h_r : r = 420) (h_t : t = 1 / 2) : r * t = 210 :=
by
  rw [h_r, h_t]
  norm_num
  sorry

end powerjet_pump_water_l738_738552


namespace min_balls_to_draw_l738_738926

theorem min_balls_to_draw (black white red : ℕ) (h_black : black = 10) (h_white : white = 9) (h_red : red = 8) :
  ∃ n, n = 20 ∧
  ∀ k, (k < 20) → ¬ (∃ b w r, b + w + r = k ∧ b ≤ black ∧ w ≤ white ∧ r ≤ red ∧ r > 0 ∧ w > 0) :=
by {
  sorry
}

end min_balls_to_draw_l738_738926


namespace dog_catches_rabbit_l738_738987

theorem dog_catches_rabbit
  (r v : ℝ) (rabbit_pos : ℝ)
  (h_rabbit_pos : rabbit_pos = 2 * r)  -- Initial position of the rabbit on the perimeter
  (h_speed : ∀ t, dog_speed t = v ∧ rabbit_speed t = v)
  (h_path : ∀ t, dog_path t = line_to_center rabbit_path t) :
  ∃ t, rabbit_path t = rabbit_pos + (1 / 4) * (2 * π * r) → dog_path t = rabbit_path t :=
begin
  sorry
end

-- Definitions of some terms assumed above
def dog_speed (t : ℝ) : ℝ := v
def rabbit_speed (t : ℝ) : ℝ := v
def dog_path (t : ℝ) : ℝ := sorry  -- precise path equation to be defined
def rabbit_path (t : ℝ) : ℝ := sorry  -- precise path equation to be defined
def line_to_center (path : ℝ) (t : ℝ) : ℝ := sorry  -- function defining the line from center to rabbit

end dog_catches_rabbit_l738_738987


namespace sin_squared_plus_one_l738_738400

theorem sin_squared_plus_one (x : ℝ) (hx : Real.tan x = 2) : Real.sin x ^ 2 + 1 = 9 / 5 := 
by 
  sorry

end sin_squared_plus_one_l738_738400


namespace problem_farthest_vertex_l738_738893

noncomputable def farthest_dilated_vertex (origin : ℝ × ℝ) (center_efgh : ℝ × ℝ) (original_area : ℝ) (scale_factor : ℝ) : ℝ × ℝ :=
  let side_length := Real.sqrt original_area
  let half_side := side_length / 2
  let vertices := [
    (center_efgh.1 + half_side, center_efgh.2 + half_side),
    (center_efgh.1 + half_side, center_efgh.2 - half_side),
    (center_efgh.1 - half_side, center_efgh.2 + half_side),
    (center_efgh.1 - half_side, center_efgh.2 - half_side)
  ]
  let dilated_vertices := vertices.map (λ (v : ℝ × ℝ), (v.1 * scale_factor, v.2 * scale_factor))
  dilated_vertices.max_by (λ (v : ℝ × ℝ), Real.sqrt (v.1^2 + v.2^2)).get_or_else origin

theorem problem_farthest_vertex :
  farthest_dilated_vertex (0, 0) (5, -5) 16 3 = (21, -21) :=
by
  sorry

end problem_farthest_vertex_l738_738893


namespace doug_lost_marbles_l738_738729

-- Definitions based on the conditions
variables (D D' : ℕ) -- D is the number of marbles Doug originally had, D' is the number Doug has now

-- Condition 1: Ed had 10 more marbles than Doug originally.
def ed_marble_initial (D : ℕ) : ℕ := D + 10

-- Condition 2: Ed had 45 marbles originally.
axiom ed_initial_marble_count : ed_marble_initial D = 45

-- Solve for D from condition 2
noncomputable def doug_initial_marble_count : ℕ := 45 - 10

-- Condition 3: Ed now has 21 more marbles than Doug.
axiom ed_current_marble_difference : 45 = D' + 21

-- Translate what we need to prove
theorem doug_lost_marbles : (doug_initial_marble_count - D') = 11 :=
by
    -- Insert math proof steps here
    sorry

end doug_lost_marbles_l738_738729


namespace binomial_identity_l738_738144

-- Given:
variables {k n : ℕ}

-- Conditions:
axiom h₁ : 1 < k
axiom h₂ : 1 < n

-- Statement:
theorem binomial_identity (h₁ : 1 < k) (h₂ : 1 < n) : 
  k * Nat.choose n k = n * Nat.choose (n - 1) (k - 1) := 
sorry

end binomial_identity_l738_738144


namespace circles_intersect_l738_738291

noncomputable def circleA : Set (ℝ × ℝ) :=
  { p | let x := p.1; let y := p.2 in x^2 + y^2 + 4*x + 2*y + 1 = 0 }

noncomputable def circleB : Set (ℝ × ℝ) :=
  { p | let x := p.1; let y := p.2 in x^2 + y^2 - 2*x - 6*y + 1 = 0 }

theorem circles_intersect :
  ∃ p : ℝ × ℝ, p ∈ circleA ∧ p ∈ circleB :=
sorry

end circles_intersect_l738_738291


namespace midpoint_AB_l738_738851

noncomputable def s (x t : ℝ) : ℝ := (x + t)^2 + (x - t)^2

noncomputable def CP (x : ℝ) : ℝ := x * Real.sqrt 3 / 2

theorem midpoint_AB (x : ℝ) (P : ℝ) : 
    (s x 0 = 2 * CP x ^ 2) ↔ P = x :=
by
    sorry

end midpoint_AB_l738_738851


namespace vector_magnitude_AD_l738_738802

variables (a b : EuclideanSpace ℝ (Fin 2))
variables (AB AC AD : EuclideanSpace ℝ (Fin 2))
variables (D : EuclideanSpace ℝ (Fin 2))

-- Given conditions
axiom angle_a_b : real.angle a b = π / 6
axiom norm_a : ∥a∥ = real.sqrt 3
axiom norm_b : ∥b∥ = 2
axiom AB_def : AB = 2 • a + 2 • b
axiom AC_def : AC = 2 • a - 6 • b
axiom D_midpoint : D = (1 / 2 : ℝ) • (AB + AC)

-- The theorem we want to prove
theorem vector_magnitude_AD : ∥AD∥ = 2 :=
by
  let a_dot_b := ∥a∥ * ∥b∥ * real.cos (real.angle a b)
  have h1 : a_dot_b = 3, by sorry
  let AD := (1 / 2 : ℝ) • (AB + AC)
  have AD_def : AD = 2 • a - 2 • b, by sorry
  have norm_AD_sq := ∥AD∥^2 = 4 * (∥a∥^2 + ∥b∥^2 - 2 * a_dot_b), by sorry
  have norm_AD := ∥AD∥ = 2, by sorry
  exact norm_AD

end vector_magnitude_AD_l738_738802


namespace min_sticks_to_form_square_12_can_form_square_15_l738_738012

theorem min_sticks_to_form_square_12 : 
  (∀ sticks : List ℕ, (sticks = (List.range 13).tail) → 
  ∃ broken_sticks, broken_sticks.length = 2 ∧ 
  ∃ sides : List ℝ, (sides.sum = 78) ∧ (∀ side ∈ sides, side = 19.5)) :=
begin
  sorry
end

theorem can_form_square_15 :
  (∀ sticks : List ℕ, (sticks = (List.range 16).tail) → 
  ∃ sides : List ℕ, (sides.sum = 120) ∧ (∀ side ∈ sides, side = 30)) :=
begin
  sorry
end

end min_sticks_to_form_square_12_can_form_square_15_l738_738012


namespace average_monthly_growth_rate_l738_738203

theorem average_monthly_growth_rate :
  ∃ x : ℝ, x = 0.2 ∧ (let a₀ := 2; let a₂ := 2.88 in a₀ * (1 + x)^2 = a₂) :=
begin
  sorry
end

end average_monthly_growth_rate_l738_738203


namespace solve_fraction_problem_l738_738101

noncomputable def fraction_problem (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) : Prop :=
  let f := x / y in
  let new_numer := 1.4 * x in
  ∃ k : ℝ, (0 < k) ∧ (new_numer / (k * y) = 2 * f) ∧ (k = 0.7)

theorem solve_fraction_problem (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) : fraction_problem x y h1 h2 :=
by
  sorry

end solve_fraction_problem_l738_738101


namespace average_minutes_per_player_l738_738523

theorem average_minutes_per_player
  (pg sg sf pf c : ℕ)
  (total_players : ℕ)
  (hp_pg : pg = 130)
  (hp_sg : sg = 145)
  (hp_sf : sf = 85)
  (hp_pf : pf = 60)
  (hp_c : c = 180)
  (hp_total_players : total_players = 5) :
  (pg + sg + sf + pf + c) / total_players / 60 = 2 :=
by
  sorry

end average_minutes_per_player_l738_738523


namespace sum_of_digits_of_product_in_base9_l738_738711

def base9_to_decimal (n : ℕ) : ℕ :=
  let d0 := n % 10
  let d1 := (n / 10) % 10
  d1 * 9 + d0

def base10_to_base9 (n : ℕ) : ℕ :=
  let d0 := n % 9
  let d1 := (n / 9) % 9
  let d2 := (n / 81) % 9
  d2 * 100 + d1 * 10 + d0

def sum_of_digits_base9 (n : ℕ) : ℕ :=
  let d0 := n % 10
  let d1 := (n / 10) % 10
  let d2 := (n / 100) % 10
  d2 + d1 + d0

theorem sum_of_digits_of_product_in_base9 :
  let n1 := base9_to_decimal 36
  let n2 := base9_to_decimal 21
  let product := n1 * n2
  let base9_product := base10_to_base9 product
  sum_of_digits_base9 base9_product = 19 :=
by
  sorry

end sum_of_digits_of_product_in_base9_l738_738711


namespace min_denominator_of_sum_600_700_l738_738215

def is_irreducible_fraction (a : ℕ) (b : ℕ) : Prop := 
  Nat.gcd a b = 1

def min_denominator_of_sum (d1 d2 : ℕ) (a b : ℕ) : ℕ :=
  let lcm := Nat.lcm d1 d2
  let sum_numerator := a * (lcm / d1) + b * (lcm / d2)
  Nat.gcd sum_numerator lcm

theorem min_denominator_of_sum_600_700 (a b : ℕ) (h1 : is_irreducible_fraction a 600) (h2 : is_irreducible_fraction b 700) :
  min_denominator_of_sum 600 700 a b = 168 := sorry

end min_denominator_of_sum_600_700_l738_738215


namespace average_score_for_entire_class_l738_738451

theorem average_score_for_entire_class (n x y : ℕ) (a b : ℝ) (hn : n = 100) (hx : x = 70) (hy : y = 30) (ha : a = 0.65) (hb : b = 0.95) :
    ((x * a + y * b) / n) = 0.74 := by
  sorry

end average_score_for_entire_class_l738_738451


namespace money_left_after_shopping_l738_738870

def initial_amount : ℕ := 26
def cost_jumper : ℕ := 9
def cost_tshirt : ℕ := 4
def cost_heels : ℕ := 5

theorem money_left_after_shopping : initial_amount - (cost_jumper + cost_tshirt + cost_heels) = 8 :=
by
  sorry

end money_left_after_shopping_l738_738870


namespace bicycle_parking_income_l738_738697

theorem bicycle_parking_income (x : ℝ) (y : ℝ) 
    (h1 : 0 ≤ x ∧ x ≤ 2000)
    (h2 : y = 0.5 * x + 0.8 * (2000 - x)) : 
    y = -0.3 * x + 1600 := by
  sorry

end bicycle_parking_income_l738_738697


namespace small_triangle_perimeter_l738_738152

theorem small_triangle_perimeter (PΔ: ℕ) (P1 P2 P3: ℕ) (Psmall: ℕ) (a b c: ℕ) :
    PΔ = 11 → 
    P1 = 5 → 
    P2 = 7 → 
    P3 = 9 →
    a + b + c = 11 →
    2 * (a + b + c) + Psmall = P1 + P2 + P3 →
    Psmall = 10 :=
by
  intros hPΔ hP1 hP2 hP3 habc heq
  rw [hPΔ, hP1, hP2, hP3] at heq
  sorry

end small_triangle_perimeter_l738_738152


namespace taxi_fare_miles_l738_738819

theorem taxi_fare_miles :
  let initial_fare : ℝ := 3.00
  let initial_miles : ℝ := 0.75
  let additional_rate : ℝ := 0.25
  let additional_unit : ℝ := 0.1
  let total_money : ℝ := 15
  let tip : ℝ := 3
  let fare_amount : ℝ := total_money - tip
  let total_miles (x : ℝ) : Prop := x = 4.35
  
  ∀ (x : ℝ), 
    3.00 + 0.25 * ((x - 0.75) / 0.1) = fare_amount → total_miles x :=
by
  intro x
  assume h
  sorry

end taxi_fare_miles_l738_738819


namespace angle_ABC_of_isosceles_triangle_l738_738110

theorem angle_ABC_of_isosceles_triangle (t : ℝ):
  ∀ (A B C : Type) (AB BC : A = B) (is_isosceles : AB = BC),
  (angle BAC = t) → (angle ABC = 180 - 2 * t) := by
  intros A B C AB BC is_isosceles h
  sorry

end angle_ABC_of_isosceles_triangle_l738_738110


namespace distinct_digit_sequences_count_l738_738747

theorem distinct_digit_sequences_count (n : ℕ) : 
    ∀ k, 1 ≤ k ∧ k ≤ n → (∑ k in finset.range(n + 1), 2 ^ k - 2 * n - 1) = 2 ^ (n + 1) - 2 * (n + 1) :=
by 
  sorry

end distinct_digit_sequences_count_l738_738747


namespace projection_of_AB_onto_BC_l738_738460

noncomputable def vec_projection (A B C : ℝ) (side_len : ℝ) (AB BC : ℝ) (angle_B : ℝ) : ℝ :=
(AB * BC * -Math.cos (angle_B)) / BC

theorem projection_of_AB_onto_BC :
  let side := 2
  let equilateral_angle := Math.pi / 3
  vec_projection 0 0 0 -- placeholder coordinates for simplicity
    side side -- side lengths AB and BC
    (Math.pi - equilateral_angle) = -1 :=
by
  let side := 2
  let equilateral_angle := Math.pi / 3
  let projection := vec_projection 0 0 0 side side (Math.pi - equilateral_angle)
  show projection = -1
  sorry

end projection_of_AB_onto_BC_l738_738460


namespace isosceles_triangle_CE_sum_square_l738_738105

/-- 
In an isosceles triangle ABC with AB = AC = 10 and BC = 12, 
four distinct points D1, D2, E1, E2 are located such that 
the triangles AD1E1, AD1E2, AD2E1, and AD2E2 are each congruent to triangle ABC, 
with BD1 = BD2 = 5. Prove that ∑(CE_k)^2 = 648.232 for k = 1, 2.
-/
theorem isosceles_triangle_CE_sum_square :
  ∀ {A B C D1 D2 E1 E2 : Type*}
  (h_iso : (AB = 10 ∧ AC = 10 ∧ BC = 12))
  (h_congr : (triangle ABC ≅ triangle AD1E1) ∧ (triangle ABC ≅ triangle AD1E2) ∧
             (triangle ABC ≅ triangle AD2E1) ∧ (triangle ABC ≅ triangle AD2E2))
  (h_dist : BD1 = 5 ∧ BD2 = 5),
    (CE1^2 + CE2^2) = 648.232 := 
sorry

end isosceles_triangle_CE_sum_square_l738_738105


namespace general_term_sum_first_n_terms_l738_738778

-- Define the arithmetic sequence a_n
def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

-- Define the geometric sequence condition
def geometric_sequence_condition (a : ℕ) (d : ℕ) : Prop :=
  a + 2*d ≠ 0 ∧ (a + 8*d) * 1 = (a + 2*d) * (a + 2*d)

-- Statement that the general term of the arithmetic sequence a_n is n
theorem general_term (d : ℕ) (a : ℕ) (h1 : d ≠ 0) (h2 : a = 1) 
  (h3 : geometric_sequence_condition a d) : ∀ n, 
  arithmetic_sequence a d n = n :=
by
  sorry

-- Define the sequence {2^a_n}
def sequence_2_pow (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  2 ^ (arithmetic_sequence a d n)

-- Define the sum of the first n terms of a sequence
def sum_sequence (f : ℕ → ℕ) (n : ℕ) : ℕ :=
  ∑ k in finset.range n, f (k + 1)

-- Statement that the sum of the first n terms of sequence {2^a_n} is 2^(n+1) - 2
theorem sum_first_n_terms (d : ℕ) (a : ℕ) (h1 : d ≠ 0) (h2 : a = 1) 
  (h3 : geometric_sequence_condition a d) : ∀ n,
  sum_sequence (sequence_2_pow a d) n = 2 ^ (n + 1) - 2 :=
by
  sorry

end general_term_sum_first_n_terms_l738_738778


namespace number_of_cows_l738_738625

theorem number_of_cows (C H : ℕ) (hcnd : 4 * C + 2 * H = 2 * (C + H) + 18) :
  C = 9 :=
sorry

end number_of_cows_l738_738625


namespace median_is_106_l738_738466

theorem median_is_106 : 
  let M := 150 in 
  let list := (List.range (M + 1)).flatMap (λ n, List.replicate n n) in 
  let N := list.length in
  let median_index := (N + 1) / 2 in 
  (list.get? (median_index - 1)).get_or_else 0 = 106 := 
by
  let M := 150
  let list := (List.range (M + 1)).flatMap (λ n, List.replicate n n)
  let N := list.length
  let median_index := (N + 1) / 2
  have h : list.nth (median_index - 1) = some 106 := sorry
  rw List.nth_eq_get? at h
  exact Option.eq_some_of_eq_some h

end median_is_106_l738_738466


namespace career_preference_degrees_l738_738627

-- Given Conditions
def ratioMaleFemale (m f : ℕ) : Prop := m = 2 ∧ f = 3

-- Given Function to calculate degrees
def degreesPerCareer (m f : ℕ) (malePreference femalePreference : ℕ) : ℝ :=
  let total_parts := m + f
  let preference_parts := malePreference + femalePreference
  (preference_parts : ℝ) / (total_parts : ℝ) * 360

-- Theorem statement
theorem career_preference_degrees :
  ∀ (m f malePreference femalePreference : ℕ),
    ratioMaleFemale m f →
    malePreference = 1 →
    femalePreference = 1 →
    degreesPerCareer m f malePreference femalePreference = 144 :=
by
  intros m f malePreference femalePreference h_ratio h_malePref h_femalePref
  -- Proof would go here
  sorry

end career_preference_degrees_l738_738627


namespace ellipse_statement_triangle_area_range_statement_l738_738049

noncomputable def ellipse_equation (a b : ℝ) (h : a > b ∧ b > 0) (e : ℝ) (ecc : e = 1 / 2)
  (S : ℝ) (H : S = sqrt(3) / 2) : Prop :=
  (∃ c : ℝ, a = 2*c ∧ a^2 = b^2 + c^2 ∧ b*(a-c) = sqrt(3)) ∧
  (a = 2 ∧ b = sqrt(3)) ∧
  (1 / 4 * x^2 + 1 / 3 * y^2 = 1)

theorem ellipse_statement : 
  ellipse_equation _ _ (by linarith) (1 / 2) (by norm_num) :=
  sorry

noncomputable def triangle_area_range (λ : ℝ) (interval : 2 ≤ λ ∧ λ ≤ 3): Set ℝ :=
{S | λ ∈ [2, 3] → S = sqrt (16 * (λ + 1)^2 / 4λ - 16)}

theorem triangle_area_range_statement (λ : ℝ)
  (h : 2 ≤ λ ∧ λ ≤ 3): triangle_area_range λ h = [sqrt 2, 4 * sqrt 3 / 3] :=
sorry

end ellipse_statement_triangle_area_range_statement_l738_738049


namespace mean_score_l738_738000

theorem mean_score (μ σ : ℝ)
  (h1 : 86 = μ - 7 * σ)
  (h2 : 90 = μ + 3 * σ) : μ = 88.8 := by
  -- Proof steps are not included as per requirements.
  sorry

end mean_score_l738_738000


namespace real_interval_0_1_uncountable_l738_738583

open Set

theorem real_interval_0_1_uncountable : ¬countable {x : ℝ | 0 ≤ x ∧ x ≤ 1} := 
sorry

end real_interval_0_1_uncountable_l738_738583


namespace sum_of_four_terms_is_123_l738_738104

theorem sum_of_four_terms_is_123 
    (a d : ℤ) -- integers instead of ℕ to avoid complications with non-positive terms
    (h_pos : a > 0 ∧ a + d > 0 ∧ a + 2d > 0 ∧ a + 42 > 0)
    (h_arith : a + a + d + a + 2d)
    (h_geom : (a + d) * (a + 42) = (a + 2d)^2)
    (h_diff : a + 42 = a + 42) 
    : 
    (a + (a + d) + (a + 2d) + (a + 42)) = 123 := 
sorry

end sum_of_four_terms_is_123_l738_738104


namespace logan_passengers_count_l738_738624

noncomputable def passengers_used_Kennedy_Airport : ℝ := (1 / 3) * 38.3
noncomputable def passengers_used_Miami_Airport : ℝ := (1 / 2) * passengers_used_Kennedy_Airport
noncomputable def passengers_used_Logan_Airport : ℝ := passengers_used_Miami_Airport / 4

theorem logan_passengers_count : abs (passengers_used_Logan_Airport - 1.6) < 0.01 := by
  sorry

end logan_passengers_count_l738_738624


namespace problem_statement_l738_738053

def f (x : ℝ) (a : ℝ) : ℝ := log (2^x + 1) / log 3 + a / (log (2^x + 1) / log 3)

def p1 (x : ℝ) : Prop :=
  x > 2/3 → f x (-2) = 0 → ∃! x, 2^x + 1 = 3^(real.sqrt 2)

def p2 (x : ℝ) (a : ℝ) : Prop :=
  -2 ≤ a ∧ a ≤ -1/2 → -1/2 ≤ x ∧ x ≤ 3 → 
  abs (f x a) = abs (log (2^x + 1) / log 3 + a / (log (2^x + 1) / log 3)) →
  (∀ x1 x2 : ℝ, x1 < x2 → abs (f x1 a) < abs (f x2 a))

theorem problem_statement : (∃ x, p1 x) ∧ ¬ (∀ x a, p2 x a) :=
sorry

end problem_statement_l738_738053


namespace shaded_region_area_l738_738537

def diameter : ℝ := 3
def radius : ℝ := diameter / 2
def length : ℝ := 18
def num_semicircles : ℕ := (length / diameter).toNat * 2
def area_one_semicircle : ℝ := (real.pi * radius^2) / 2
def total_area : ℝ := num_semicircles * area_one_semicircle

theorem shaded_region_area : total_area = 27 * real.pi :=
by
  sorry

end shaded_region_area_l738_738537


namespace number_of_elective_schemes_l738_738931

def total_courses : ℕ := 10
def conflicting_courses : finset ℕ := {0, 1, 2} -- Assume A, B, C are represented by 0, 1, 2
def choose_k (n k : ℕ) : ℕ := nat.choose n k
def total_elective_courses : ℕ := 3

theorem number_of_elective_schemes :
  choose_k (total_courses - conflicting_courses.card) total_elective_courses +
  conflicting_courses.card * 
  choose_k (total_courses - conflicting_courses.card) (total_elective_courses - 1) = 98 :=
by
  sorry

end number_of_elective_schemes_l738_738931


namespace max_covered_area_l738_738216

theorem max_covered_area (A1 A2 A_overlap: ℝ) (h1: A1 = 1) (h2: A2 = 4) (h3: A_overlap = 0.25) : 
  (A1 + A2 - A_overlap) = 4.75 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end max_covered_area_l738_738216


namespace max_xy_l738_738139

theorem max_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x / 3 + y / 4 = 1) : xy ≤ 3 :=
by {
  -- proof omitted
  sorry
}

end max_xy_l738_738139


namespace number_of_assignments_l738_738317

def is_assignment (stmt : String) : Prop :=
  match stmt with
  | "m = x^3 - x^2" => True
  | "T = T × I" => True
  | "32 = A" => False
  | "A = A + 2" => True
  | "a = b = 4" => False
  | _ => False

def count_assignments (stmts : List String) : Nat :=
  stmts.countp is_assignment

theorem number_of_assignments :
  count_assignments ["m = x^3 - x^2", "T = T × I", "32 = A", "A = A + 2", "a = b = 4"] = 3 :=
  by
  sorry

end number_of_assignments_l738_738317


namespace triangle_ABD_area_l738_738313

theorem triangle_ABD_area (A B C D : Type) 
  (dist_AB : ℝ) (dist_BC : ℝ) (dist_CA : ℝ) (dist_AD : ℝ)
  (AD_bisects_∠BAC : AD ∈ segment A C) :
  dist_AB = 5 → dist_BC = 8 → dist_CA = 11 → 
  let s := (dist_AB + dist_BC + dist_CA) / 2 in
  let area_ABC := Real.sqrt (s * (s - dist_AB) * (s - dist_BC) * (s - dist_CA)) in
  let area_ABD := (5 / (5 + 11)) * area_ABC in
  area_ABD = 5 * Real.sqrt 21 / 4 :=
by
  sorry

end triangle_ABD_area_l738_738313


namespace Rio_Coralio_Length_Estimate_l738_738646

def RioCoralioLength := 402.5
def GSA_length := 402
def AWRA_length := 403
def error_margin := 0.5
def error_probability := 0.04

theorem Rio_Coralio_Length_Estimate :
  ∀ (L_GSA L_AWRA : ℝ) (margin error_prob : ℝ),
  L_GSA = GSA_length ∧ L_AWRA = AWRA_length ∧ 
  margin = error_margin ∧ error_prob = error_probability →
  (RioCoralioLength = 402.5) ∧ (error_probability = 0.04) := 
by 
  intros L_GSA L_AWRA margin error_prob h,
  sorry

end Rio_Coralio_Length_Estimate_l738_738646


namespace problem1_problem2_problem3_l738_738419

section evens_function

/-- Define the function f(x) = (x^2 - 1) / x^2 --/
def f (x : ℝ) : ℝ := (x^2 - 1) / x^2

/-- Problem 1: The function f(x) is even --/
theorem problem1: ∀ x : ℝ, f (-x) = f x :=
by
  intro x
  sorry

end evens_function

section k_range

/-- Define the function f again as it may be needed for consistency --/
def f (x : ℝ) : ℝ := (x^2 - 1) / x^2

/-- Problem 2: The range of k such that k ≤ x f(x) + 1 / x for all x ∈ [1, 3] is k ≤ 1 --/
theorem problem2: ∀ k : ℝ, (∀ x ∈ set.Icc (1 : ℝ) 3, k ≤ x * f x + 1 / x) → k ≤ 1 :=
by
  intro k h
  sorry

end k_range

section t_range

/-- Define the function f once more as consistency is key --/
def f (x : ℝ) : ℝ := (x^2 - 1) / x^2

/-- Define g(x) = t * f(x) + 1 --/
def g (x : ℝ) (t : ℝ) : ℝ := t * f x + 1

/-- Problem 3: The range of t, such that ∀ x ∈ [1/m, 1/n] (m > 0, n > 0), 
the range of g(x) is [2 - 3m, 2 - 3n], is 0 < t < 1 --/
theorem problem3 (m n t : ℝ) (hm : 0 < m) (hn : 0 < n):
  (∀ x ∈ set.Icc (1 / m) (1 / n), set.Icc (2 - 3 * m) (2 - 3 * n) = set.range (λ x, g x t)) → 
  0 < t ∧ t < 1 :=
by
  intros h1 h2 h3
  sorry

end t_range

end problem1_problem2_problem3_l738_738419


namespace problem1_problem2_l738_738792

-- Define the conditions: f is an odd and decreasing function on [-1, 1]
variable (f : ℝ → ℝ)
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_decreasing : ∀ x y, x ≤ y → f y ≤ f x)

-- The domain of interest is [-1, 1]
variable (x1 x2 : ℝ)
variable (h_x1 : x1 ∈ Set.Icc (-1 : ℝ) 1)
variable (h_x2 : x2 ∈ Set.Icc (-1 : ℝ) 1)

-- Proof Problem 1
theorem problem1 : (f x1 + f x2) * (x1 + x2) ≤ 0 := by
  sorry

-- Assume condition for Problem 2
variable (a : ℝ)
variable (h_ineq : f (1 - a) + f (1 - a ^ 2) < 0)
variable (h_dom : ∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → x ∈ Set.Icc (-1 : ℝ) 1)

-- Proof Problem 2
theorem problem2 : 0 < a ∧ a < 1 := by
  sorry

end problem1_problem2_l738_738792


namespace line_eq1_line_eq2_l738_738997

-- Define the line equations
def l1 (x y : ℝ) : Prop := 4 * x + y + 6 = 0
def l2 (x y : ℝ) : Prop := 3 * x - 5 * y - 6 = 0

-- Theorem for when midpoint is at (0, 0)
theorem line_eq1 : ∀ x y : ℝ, (x + 6 * y = 0) ↔
  ∃ (a : ℝ), 
    l1 a (-(a / 6)) ∧
    l2 (-a) ((a / 6)) ∧
    (a + -a = 0) ∧ (-(a / 6) + a / 6 = 0) := 
by 
  sorry

-- Theorem for when midpoint is at (0, 1)
theorem line_eq2 : ∀ x y : ℝ, (x + 2 * y - 2 = 0) ↔
  ∃ (b : ℝ),
    l1 b (-b / 2 + 1) ∧
    l2 (-b) (1 - (-b / 2)) ∧
    (b + -b = 0) ∧ (-b / 2 + 1 + (1 - (-b / 2)) = 2) := 
by 
  sorry

end line_eq1_line_eq2_l738_738997


namespace concave_number_probability_l738_738694

-- Definitions for the conditions
def is_digit (n : ℕ) : Prop := n ∈ {1, 2, 3, 4}
def distinct (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c
def concave_number (a b c : ℕ) : Prop := a > b ∧ b < c

-- The theorem statement
theorem concave_number_probability (a b c : ℕ) (h₁ : is_digit a) (h₂ : is_digit b) (h₃ : is_digit c) (h₄ : distinct a b c) : 
  (Prob (concave_number a b c | distinct a b c) = 1 / 3) :=
sorry

end concave_number_probability_l738_738694


namespace point_Q_exists_bisects_segment_l738_738596

noncomputable def problem_1 (x y: ℝ): Prop :=
  5 * x - 7 * y - 70 = 0

noncomputable def problem_2 (x y: ℝ): Prop :=
  (x^2) / 25 + (y^2) / 9 = 1

theorem point_Q_exists_bisects_segment:
  (∃ Q P M N : ℝ × ℝ,
    l P ∧
    ellipse M ∧
    ellipse N ∧
    chord MN M N P ∧
    (∀ P, l P → passes_through Q (MN P)) ∧
    (∃ Q, bisects Q (MN) ∧ parallel MN l)) :=
  sorry

end point_Q_exists_bisects_segment_l738_738596


namespace Carlos_cookies_per_batch_l738_738008

-- Definitions and conditions
def Anne_area_total : ℝ := 180 -- total dough used by Anne in square inches
def Anne_cookies : ℕ := 15 -- number of cookies made by Anne
def Anne_cookie_area : ℝ := Anne_area_total / Anne_cookies -- single cookie area for Anne

def Carlos_cookie_side : ℝ := 3 -- side length of Carlos' square cookies
def Carlos_cookie_area : ℝ := Carlos_cookie_side ^ 2 -- single cookie area for Carlos

def total_dough : ℝ := Anne_area_total -- total dough used is the same for both Anne and Carlos

-- Theorem statement
theorem Carlos_cookies_per_batch : (total_dough / Carlos_cookie_area) = 20 :=
begin
  sorry
end

end Carlos_cookies_per_batch_l738_738008


namespace garden_walkway_area_l738_738820

theorem garden_walkway_area :
  let width_bed : ℕ := 8,
      height_bed : ℕ := 3,
      walkway_width : ℕ := 2,
      rows : ℕ := 4,
      beds_row1 : ℕ := 3,
      beds_row_others : ℕ := 2 in
  let width_total := 3 * width_bed + 4 * walkway_width,
      height_total := rows * height_bed + 5 * walkway_width,
      total_area := width_total * height_total,
      area_bed := width_bed * height_bed,
      total_beds := beds_row1 + 3 * beds_row_others,
      total_bed_area := total_beds * area_bed,
      walkway_area := total_area - total_bed_area in
  walkway_area = 488 :=
by
  sorry

end garden_walkway_area_l738_738820


namespace min_handshakes_30_people_3_each_l738_738277

theorem min_handshakes_30_people_3_each : 
  ∃ (H : ℕ), (∀ (n k : ℕ), n = 30 ∧ k = 3 → H = (n * k) / 2) := 
by {
  use 45,
  intros n k h,
  rw [← h.1, ← h.2],
  norm_num,
  sorry
}

end min_handshakes_30_people_3_each_l738_738277


namespace probability_Juliet_supporter_in_Capulet_is_correct_l738_738108

noncomputable def probability_Juliet_supporter_in_Capulet (total_population : ℝ) : ℝ :=
  let montague_population := (5 / 8) * total_population
  let capulet_population := (3 / 16) * total_population
  let verona_population := (1 / 8) * total_population
  let mercutio_population := total_population - montague_population - capulet_population - verona_population
  let romeo_supporters_montague := 0.8 * montague_population
  let juliet_supporters_capulet := 0.7 * capulet_population
  let romeo_supporters_verona := 0.65 * verona_population
  let juliet_supporters_mercutio := 0.55 * mercutio_population
  let total_juliet_supporters := juliet_supporters_capulet + juliet_supporters_mercutio
  (juliet_supporters_capulet / total_juliet_supporters) * 100

theorem probability_Juliet_supporter_in_Capulet_is_correct :
  probability_Juliet_supporter_in_Capulet total_population ≈ 66 := sorry

end probability_Juliet_supporter_in_Capulet_is_correct_l738_738108


namespace min_handshakes_l738_738281

theorem min_handshakes 
  (people : ℕ) 
  (handshakes_per_person : ℕ) 
  (total_people : people = 30) 
  (handshakes_rule : handshakes_per_person = 3) 
  (unique_handshakes : people * handshakes_per_person % 2 = 0) 
  (multiple_people : people > 0):
  (people * handshakes_per_person / 2) = 45 :=
by
  sorry

end min_handshakes_l738_738281


namespace determine_m_l738_738795

theorem determine_m (m : ℝ) : (∀ x : ℝ, (m * x = 1 → x = 1 ∨ x = -1)) ↔ (m = 0 ∨ m = 1 ∨ m = -1) :=
by sorry

end determine_m_l738_738795


namespace sequence_term_l738_738390

theorem sequence_term (n : ℕ) (S : ℕ → ℕ) (a : ℕ → ℕ)
  (h1 : ∀ n, log 2 (S n + 1) = n) :
  (∀ n, S n = 2^n - 1) → a n = 2^(n-1) :=
by
  intros hS
  sorry

end sequence_term_l738_738390


namespace expression_value_l738_738608

theorem expression_value : 4 * (8 - 2) ^ 2 - 6 = 138 :=
by
  sorry

end expression_value_l738_738608


namespace total_socks_l738_738121

-- Definitions based on conditions
def red_pairs : ℕ := 20
def red_socks : ℕ := red_pairs * 2
def black_socks : ℕ := red_socks / 2
def white_socks : ℕ := 2 * (red_socks + black_socks)

-- The main theorem we want to prove
theorem total_socks :
  (red_socks + black_socks + white_socks) = 180 := by
  sorry

end total_socks_l738_738121


namespace sum_of_products_lt_one_l738_738590

variable {a : Fin 1959 → ℝ} -- Define 1959 positive numbers

-- Define the sum condition
def sum_condition : Prop := (∑ i, a i) = 1

-- Define the sum of products condition
def sum_of_products (s : Finset (Fin 1959)) (hs : s.card = 1000) : ℝ :=
  ∑ (t ∈ s.powerset) (ht : t.card = 1000), t.prod a

-- The main theorem statement
theorem sum_of_products_lt_one (h : sum_condition) : 
  sum_of_products = (λ s hs, ∑ (t ∈ s.powerset) (ht : t.card = 1000), t.prod a) < 1 :=
sorry

end sum_of_products_lt_one_l738_738590


namespace simultaneous_equations_solution_exists_l738_738755

theorem simultaneous_equations_solution_exists (m : ℝ) : 
  (∃ (x y : ℝ), y = m * x + 6 ∧ y = (2 * m - 3) * x + 9) ↔ m ≠ 3 :=
by
  sorry

end simultaneous_equations_solution_exists_l738_738755


namespace coordinates_of_point_on_x_axis_l738_738039

theorem coordinates_of_point_on_x_axis (m : ℤ) 
  (h : 2 * m + 8 = 0) : (m + 5, 2 * m + 8) = (1, 0) :=
sorry

end coordinates_of_point_on_x_axis_l738_738039


namespace arithmetic_progression_expression_l738_738103

theorem arithmetic_progression_expression :
  let a : ℕ → ℝ := λ n, 150 + (n - 1000) * 0.5 in
  99 * 100 * (∑ k in Finset.range (2020 - 1580 + 1), 1 / (a (1580 + k) * a (1580 + k + 1))) = 15 := 
by
  sorry

end arithmetic_progression_expression_l738_738103


namespace lean_proof_l738_738259

variables (ABCDEF : Type) [inscribed_in_circle ABCDEF]
variables (A B C D E F : ABCDEF)
variables (AB BC CD DE EF FA : ℝ) (AD : ℝ)
variables (cos_angle_B cos_angle_ADF : ℝ)

-- Conditions
axiom AB_eq_3 : AB = 3
axiom BC_eq_3 : BC = 3
axiom CD_eq_3 : CD = 3
axiom DE_eq_3 : DE = 3
axiom EF_eq_3 : EF = 3
axiom FA_eq_3 : FA = 3
axiom AD_eq_2 : AD = 2

-- Cosine of angles
axiom cos_B : cos_angle_B = 7 / 9
axiom cos_ADF : cos_angle_ADF = 3 / 4

theorem lean_proof : 
  (1 - cos_angle_B) * (1 - cos_angle_ADF) = 1 / 18 :=
by
  sorry

end lean_proof_l738_738259


namespace part_a_10_months_cost_part_b_36_months_cost_l738_738970

-- Define the conditions
def incandescent_power : ℕ := 60  -- 60 watts
def energy_efficient_power : ℕ := 12  -- 12 watts
def monthly_hours : ℕ := 100  -- 100 hours
def tariff : ℕ := 5  -- 5 rubles per kWh
def energy_efficient_cost : ℕ := 120  -- 120 rubles
def service_company_share : ℕ := 75  -- 75%

-- Part (a): Prove that the total cost of installing the energy-efficient lamp himself over 10 months 
-- is less than the cost of using the energy service company over the same period
theorem part_a_10_months_cost :
  let monthly_cost_incandescent := (incandescent_power * monthly_hours) / 1000 * tariff in
  let total_cost_incandescent_10_months := monthly_cost_incandescent * 10 in
  let monthly_cost_energy_efficient := (energy_efficient_power * monthly_hours) / 1000 * tariff in
  let total_cost_energy_efficient_10_months := energy_efficient_cost + monthly_cost_energy_efficient * 10 in
  let energy_cost_savings := monthly_cost_incandescent - monthly_cost_energy_efficient in
  let service_payment := (service_company_share * energy_cost_savings) / 100 in
  let total_cost_with_company_10_months := (monthly_cost_energy_efficient + service_payment) * 10 in
  total_cost_energy_efficient_10_months < total_cost_with_company_10_months := by
  sorry

-- Part (b): Prove that the total cost of installing the energy-efficient lamp himself over 36 months 
-- is less than the cost of using the energy service company over the same period
theorem part_b_36_months_cost :
  let monthly_cost_energy_efficient := (energy_efficient_power * monthly_hours) / 1000 * tariff in
  let total_cost_energy_efficient_36_months := energy_efficient_cost + monthly_cost_energy_efficient * 36 in
  let energy_cost_savings := (incandescent_power * monthly_hours) / 1000 * tariff - monthly_cost_energy_efficient in
  let service_payment := (service_company_share * energy_cost_savings) / 100 in
  let total_cost_with_company_10_months := (monthly_cost_energy_efficient + service_payment) * 10 in
  let total_cost_with_company_36_months := total_cost_with_company_10_months + monthly_cost_energy_efficient * 26 in
  total_cost_energy_efficient_36_months < total_cost_with_company_36_months := by
  sorry


end part_a_10_months_cost_part_b_36_months_cost_l738_738970


namespace range_of_a_l738_738396

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (x^2 - x - 2 ≥ 0) ↔ (x ≤ -1 ∨ x ≥ 2)) ∧
  (∀ x : ℝ, (2 * a - 1 ≤ x ∧ x ≤ a + 3)) →
  (-1 ≤ a ∧ a ≤ 0) :=
by
  -- Prove the theorem
  sorry

end range_of_a_l738_738396


namespace fraction_of_subsets_with_odd_smallest_element_l738_738387

theorem fraction_of_subsets_with_odd_smallest_element (n : ℕ) (h : 0 < n) :
  (let N := 2 * n in
   (∑ k in (finset.powerset (finset.range (N + 1))).filter (λ s, s.nonempty),
    if ∃ m ∈ s, m % 2 = 1 then (1 : ℚ) else (0 : ℚ)) /
   (∑ k in (finset.powerset (finset.range (N + 1))).filter (λ s, s.nonempty), (1 : ℚ))) = (2 / 3 : ℚ) :=
sorry

end fraction_of_subsets_with_odd_smallest_element_l738_738387


namespace multiple_of_5_l738_738813

theorem multiple_of_5 (a : ℤ) (h : ¬ (5 ∣ a)) : 5 ∣ (a^12 - 1) :=
by
  sorry

end multiple_of_5_l738_738813


namespace ice_cream_volume_l738_738194

-- Definitions based on Conditions
def radius_cone : Real := 3 -- radius at the opening of the cone
def height_cone : Real := 12 -- height of the cone

-- The proof statement
theorem ice_cream_volume :
  (1 / 3 * Real.pi * radius_cone^2 * height_cone) + (4 / 3 * Real.pi * radius_cone^3) = 72 * Real.pi := by
  sorry

end ice_cream_volume_l738_738194


namespace unattainable_value_of_y_l738_738748

noncomputable def f (x : ℝ) : ℝ := (2 - x) / (3 * x + 4)

theorem unattainable_value_of_y :
  ∃ y : ℝ, y = -(1 / 3) ∧ ∀ x : ℝ, 3 * x + 4 ≠ 0 → f x ≠ y :=
by
  sorry

end unattainable_value_of_y_l738_738748


namespace ryan_correct_percentage_l738_738536

theorem ryan_correct_percentage :
  let problems1 := 25
  let correct1 := 0.8 * problems1
  let problems2 := 40
  let correct2 := 0.9 * problems2
  let problems3 := 10
  let correct3 := 0.7 * problems3
  let total_problems := problems1 + problems2 + problems3
  let total_correct := correct1 + correct2 + correct3
  (total_correct / total_problems) = 0.84 :=
by 
  sorry

end ryan_correct_percentage_l738_738536


namespace ratios_are_constant_l738_738872

theorem ratios_are_constant (n : ℕ) (a b : Fin n → ℝ) 
  (h1 : ∑ i, a i ^ 2 = 2018 ^ 2) 
  (h2 : ∑ i, b i ^ 2 = 2017 ^ 2) 
  (h3 : ∑ i, a i * b i = 2017 * 2018) : 
  ∀ i, a i / b i = 2018 / 2017 := 
by
  sorry

end ratios_are_constant_l738_738872


namespace geometric_seq_sum_first_4_terms_l738_738369

theorem geometric_seq_sum_first_4_terms (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n, a (n + 1) = a n * 2)
  (h3 : ∀ n, S (n + 1) = S n + a (n + 1)) :
  S 4 = 15 :=
by
  -- The actual proof would go here.
  sorry

end geometric_seq_sum_first_4_terms_l738_738369


namespace find_m_l738_738498

variable (m a : ℤ)

def quadratic_eq (x : ℤ) : Prop :=
  (x - 2)^2 + (a - m)^2 = 2 * m * x + a^2 - 2 * a * m

theorem find_m (m a : ℤ) (h1 : 1 ≤ m) (h2 : m ≤ 50) 
  (h3 : ∀ x : ℤ, quadratic_eq m a x → x > 0) :
  ∃ k : ℕ, m = k * k ∧ k ∈ [1, 2, 3, 4, 5, 6, 7]:
sorry

end find_m_l738_738498


namespace range_of_t_l738_738034

noncomputable def f : ℝ → ℝ := sorry

def decreasing (f: ℝ → ℝ) := ∀ x y : ℝ, x < y → f x > f y

def P (t : ℝ) : set ℝ := {x | abs (f (x + t) - 1) < 2}

def Q : set ℝ := {x | f x < -1}

theorem range_of_t (h_decreasing : decreasing f)
  (h_f0 : f 0 = 3)
  (h_f3 : f 3 = -1)
  (h_subset : ∀ t, ∀ x, x ∈ P t → x ∈ Q)
  (h_not_necessary : ∃ t, ∃ x, x ∉ P t ∧ x ∈ Q) :
  t ≤ -3 :=
sorry

end range_of_t_l738_738034


namespace coefficient_x2_in_binomial_expansion_l738_738464

theorem coefficient_x2_in_binomial_expansion :
  (∃ T, binomial_expansion (2 * x - 1 / sqrt(x)) 8 T ∧ coefficient T x^2 = 1120) := sorry

end coefficient_x2_in_binomial_expansion_l738_738464


namespace perpendicular_bisector_eq_l738_738801

-- Definition of points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (3, 2)

-- Theorem stating that the perpendicular bisector has the specified equation
theorem perpendicular_bisector_eq : ∀ (x y : ℝ), (y = -2 * x + 3) ↔ ∃ (a b : ℝ), (a, b) = A ∨ (a, b) = B ∧ (y = -2 * x + 3) :=
by
  sorry

end perpendicular_bisector_eq_l738_738801


namespace total_nuggets_ordered_l738_738315

noncomputable def Alyssa_nuggets : ℕ := 20
noncomputable def Keely_nuggets : ℕ := 2 * Alyssa_nuggets
noncomputable def Kendall_nuggets : ℕ := 2 * Alyssa_nuggets

theorem total_nuggets_ordered : Alyssa_nuggets + Keely_nuggets + Kendall_nuggets = 100 := by
  sorry -- Proof is intentionally omitted

end total_nuggets_ordered_l738_738315


namespace average_screen_time_per_player_l738_738525

def video_point_guard : ℕ := 130
def video_shooting_guard : ℕ := 145
def video_small_forward : ℕ := 85
def video_power_forward : ℕ := 60
def video_center : ℕ := 180
def total_video_time : ℕ := 
  video_point_guard + video_shooting_guard + video_small_forward + video_power_forward + video_center
def total_video_time_minutes : ℕ := total_video_time / 60
def number_of_players : ℕ := 5

theorem average_screen_time_per_player : total_video_time_minutes / number_of_players = 2 :=
  sorry

end average_screen_time_per_player_l738_738525


namespace twin_prime_probability_l738_738098

def is_prime (n : Nat) : Prop := Nat.Prime n

def is_twin_prime (p : Nat) (q : Nat) : Prop := (p < q) ∧ (q = p + 2) ∧ is_prime p ∧ is_prime q

def primes_upto_30 : List Nat := [3, 5, 7, 11, 13, 17, 19, 23, 29]

def twin_prime_pairs (primes : List Nat) : List (Nat × Nat) :=
  (primes.product primes).filter (λ (p, q), is_twin_prime p q)

noncomputable def prob_twin_primes : ℚ :=
  (twin_prime_pairs primes_upto_30).length / (Nat.choose 9 2)

theorem twin_prime_probability :
  prob_twin_primes = 4 / 45 :=
by
  sorry

end twin_prime_probability_l738_738098


namespace min_value_of_expression_l738_738855

theorem min_value_of_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ m, m = a^2 + a*b + b^2 + 1/(a + b)^2 ∧ m = real.sqrt 3 :=
sorry

end min_value_of_expression_l738_738855


namespace new_weights_inequality_l738_738674

theorem new_weights_inequality (W : ℝ) (x y : ℝ) (h_avg_increase : (8 * W - 2 * 68 + x + y) / 8 = W + 5.5)
  (h_sum_new_weights : x + y ≤ 180) : x > W ∧ y > W :=
by {
  sorry
}

end new_weights_inequality_l738_738674


namespace sum_first_n_terms_prove_l738_738392

noncomputable theory

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n, a n = a 0 + n * d

def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
(n * (a 0 + a (n - 1)) / 2)

theorem sum_first_n_terms_prove (a : ℕ → ℤ) (d : ℤ) :
  (a 2 * a 6 = -16) →
  (a 3 + a 5 = 0) →
  (sum_of_first_n_terms a n = n * (n - 9) ∨ sum_of_first_n_terms a n = -n * (n - 9)) :=
by
  sorry

end sum_first_n_terms_prove_l738_738392


namespace sufficient_but_not_necessary_condition_l738_738145

variables (A B C : Prop)

theorem sufficient_but_not_necessary_condition (h1 : B → A) (h2 : C → B) (h3 : ¬(B → C)) : (C → A) ∧ ¬(A → C) :=
by
  sorry

end sufficient_but_not_necessary_condition_l738_738145


namespace power_sum_mod_inverse_l738_738235

theorem power_sum_mod_inverse (h : 3^6 ≡ 1 [MOD 17]) : 
  (3^(-1) + 3^(-2) + 3^(-3) + 3^(-4) +  3^(-5) + 3^(-6)) ≡ 1 [MOD 17] := 
by
  sorry

end power_sum_mod_inverse_l738_738235


namespace no_positive_roots_poly_l738_738489

theorem no_positive_roots_poly {n : ℕ} (a : Fin n → ℕ) (k M : ℕ) :
  (1 < M) →
  (∀ i, 1 ≤ a i) →
  (∑ i, 1 / (a i) = k) →
  (∏ i, a i = M) →
  ∀ x > 0, M * (x + 1) ^ k < (∏ i, (x + a i)) :=
begin
  sorry
end

end no_positive_roots_poly_l738_738489


namespace cos_beta_value_l738_738030

theorem cos_beta_value (α β : ℝ) 
  (h1 : sin α = 4 * real.sqrt 3 / 7) 
  (h2 : cos (α + β) = -11 / 14) 
  (h3 : 0 < α ∧ α < π / 2)
  (h4 : 0 < β ∧ β < π / 2) : 
  cos β = 1 / 2 := 
sory

end cos_beta_value_l738_738030


namespace parking_lot_problem_l738_738211

theorem parking_lot_problem :
  let total_spaces := 50
  let cars := 2
  let total_ways := total_spaces * (total_spaces - 1)
  let adjacent_ways := (total_spaces - 1) * 2
  let valid_ways := total_ways - adjacent_ways
  valid_ways = 2352 :=
by
  sorry

end parking_lot_problem_l738_738211


namespace tan_addition_identity_l738_738014

theorem tan_addition_identity
  (α : ℝ)
  (hα : α ∈ Ioo (π / 2) π)
  (h_sin : Real.sin α = 3 / 5) :
  Real.tan (α + π / 4) = 1 / 7 :=
  sorry

end tan_addition_identity_l738_738014


namespace binary_multiplication_division_l738_738348

theorem binary_multiplication_division :
  let a := 0b1011010
  let b := 0b1010100
  let c := 0b1010
  (a * b) / c = 0b1011100100 :=
by
  let a := 0b1011010
  let b := 0b1010100
  let c := 0b1010
  have div_result : a / c = 0b1001 := sorry -- This is the division result from step 1
  have mul_result : (0b1001 * b) = 0b1011100100 := sorry -- This is the multiplication result from step 2
  exact calc
    (a * b) / c = (0b1001 * b) / 1 : by rw [mul_comm, div_result]
    ... = 0b1011100100 : by rw [mul_result, div_one]

end binary_multiplication_division_l738_738348


namespace original_volume_of_cube_l738_738454

theorem original_volume_of_cube (a V_cube V_new : ℕ) 
  (h1 : (a - 2) ≥ 0)
  (h2 : (a + 2) > 0)
  (hV_cube : V_cube = a^3)
  (hV_new : V_new = (a - 2) * a * (a + 2))
  (hV_diff : V_cube - V_new = 16) :
  V_cube = 8 :=
by
  have h_eq : (a - 2) * a * (a + 2) + 16 = a^3 := by sorry
  have h_a_eq_2 : a = 2 := by sorry
  have h_V_cube : V_cube = 2^3 := by sorry
  rw h_V_cube
  exact rfl

end original_volume_of_cube_l738_738454


namespace true_proposition_l738_738775

variable (a : ℝ) (x : ℝ)

def p : Prop := (∀ x : ℝ, 0 < a → a ≠ 1 → has_deriv_at (fun x => a^x) x (a^x * real.log a)) → ∀ x1 x2, x1 < x2 → a^x1 < a^x2
def q : Prop := ∀ x : ℝ, x ∈ set.Ioo (real.pi / 4) (5 * real.pi / 4) → real.sin x > real.cos x

theorem true_proposition : ¬p ∧ q :=
by
  sorry

end true_proposition_l738_738775


namespace conjunction_used_in_proposition_l738_738834

theorem conjunction_used_in_proposition (x : ℝ) (h : x^2 = 4) :
  (x = 2 ∨ x = -2) :=
sorry

end conjunction_used_in_proposition_l738_738834


namespace product_slope_y_intercept_half_area_line_l738_738597

noncomputable def A : ℝ × ℝ := (0, 6)
noncomputable def B : ℝ × ℝ := (2, 0)
noncomputable def C : ℝ × ℝ := (8, 0)

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

def line_eq (m : ℝ) (p : ℝ × ℝ) : ℝ → ℝ :=
  λ x, m * (x - p.1) + p.2

theorem product_slope_y_intercept_half_area_line : 
  let M := midpoint A C,
      m := slope B M,
      y_intercept := line_eq m B 0 in
  m * y_intercept = -9 / 2 :=
begin
  let M := midpoint A C,
  let m := slope B M,
  let y_intercept := line_eq m B 0,
  show m * y_intercept = -9 / 2,
  sorry
end

end product_slope_y_intercept_half_area_line_l738_738597


namespace sin_cos_roots_l738_738087

theorem sin_cos_roots (α β u v w y : ℝ) 
  (h1 : ∀ x, x^2 - u * x + v = 0 → x = sin α ∨ x = sin β)
  (h2 : ∀ x, x^2 - w * x + y = 0 → x = cos α ∨ x = cos β) :
  w * y = 1 - v :=
sorry

end sin_cos_roots_l738_738087


namespace ratio_S1_S2_l738_738405

variable (r : ℝ)
def S1 : ℝ := (Real.pi * r ^ 2)
def S2 : ℝ := (4 * Real.pi * r ^ 2)

theorem ratio_S1_S2 : S1 r / S2 r = 1 / 4 :=
by
  simp [S1, S2]
  sorry

end ratio_S1_S2_l738_738405


namespace total_sample_size_is_72_l738_738982

-- Definitions based on the given conditions:
def production_A : ℕ := 600
def production_B : ℕ := 1200
def production_C : ℕ := 1800
def total_production : ℕ := production_A + production_B + production_C
def sampled_B : ℕ := 2

-- Main theorem to prove the sample size:
theorem total_sample_size_is_72 : 
  ∃ (n : ℕ), 
    (∃ s_A s_B s_C, 
      s_A = (production_A * sampled_B * total_production) / production_B^2 ∧ 
      s_B = sampled_B ∧ 
      s_C = (production_C * sampled_B * total_production) / production_B^2 ∧
      n = s_A + s_B + s_C) ∧ 
  (n = 72) :=
sorry

end total_sample_size_is_72_l738_738982


namespace maximum_g_value_l738_738127

noncomputable def g (x : ℝ) : ℝ := real.sqrt(x * (100 - x)) + real.sqrt(x * (4 - x))

theorem maximum_g_value : ∃ x1 N, (x1 = 50 / 13) ∧ (N = 4 * real.sqrt(26)) ∧ ∀ x, 0 ≤ x ∧ x ≤ 4 → g(x) ≤ N ∧ g(x1) = N :=
by
  sorry

end maximum_g_value_l738_738127


namespace arithmetic_sequence_a2_value_l738_738413

theorem arithmetic_sequence_a2_value :
  ∃ (a : ℕ) (d : ℕ), (a = 3) ∧ (a + d + (a + 2 * d) = 12) ∧ (a + d = 5) :=
by
  sorry

end arithmetic_sequence_a2_value_l738_738413


namespace dot_product_sum_l738_738073

theorem dot_product_sum (a : ℝ × ℝ × ℝ) (b : ℝ × ℝ × ℝ) (c : ℝ × ℝ × ℝ)
  (h₀ : a = (-3, 2, 5)) 
  (h₁ : b = (1, -3, 0)) 
  (h₂ : c = (7, -2, 1)) :
  (let ab := (a.1 + b.1, a.2 + b.2, a.3 + b.3) in ab.1 * c.1 + ab.2 * c.2 + ab.3 * c.3) = -7 :=
by
  sorry

end dot_product_sum_l738_738073


namespace tina_cleaning_problem_l738_738936

/-- Tina's homework and cleaning problem proof -/
theorem tina_cleaning_problem :
  ∃ (x : ℕ), (14 * x + 10 = 52) ∧ (x = 3) :=
by
  use 3
  split
  · linarith
  · rfl

end tina_cleaning_problem_l738_738936


namespace generating_function_transform_l738_738046

def ogf (a : ℕ → ℂ) : ℂ → ℂ := λ t, ∑' n, a n * t^n

def ogf_transformed (a : ℕ → ℂ) : ℂ → ℂ := λ t, ∑' n, (∑ i in Finset.range (n + 1), a i) * t^n

theorem generating_function_transform (a : ℕ → ℂ) (A : ℂ → ℂ) (hA : A = ogf a) :
    ogf_transformed a = λ t, (1 / (1 - t)) * A t := by
  -- proof goes here
  sorry

end generating_function_transform_l738_738046


namespace sum_of_inverses_mod_17_l738_738240

noncomputable def inverse_sum_mod_17 : ℤ :=
  let a1 := Nat.gcdA 3 17 in -- 3^{-1} mod 17
  let a2 := Nat.gcdA (3^2) 17 in -- 3^{-2} mod 17
  let a3 := Nat.gcdA (3^3) 17 in -- 3^{-3} mod 17
  let a4 := Nat.gcdA (3^4) 17 in -- 3^{-4} mod 17
  let a5 := Nat.gcdA (3^5) 17 in -- 3^{-5} mod 17
  let a6 := Nat.gcdA (3^6) 17 in -- 3^{-6} mod 17
  (a1 + a2 + a3 + a4 + a5 + a6) % 17

theorem sum_of_inverses_mod_17 : inverse_sum_mod_17 = 7 := sorry

end sum_of_inverses_mod_17_l738_738240


namespace floor_abs_S_l738_738550

noncomputable def S : ℝ := ∑ n in Finset.range 10, x n

axiom condition (x : ℕ → ℝ) : 
  ∀ n, 0 ≤ n ∧ n < 10 →
  x n + (n + 2) = S + 20

theorem floor_abs_S (x : ℕ → ℝ) (h : ∀ n, 0 ≤ n ∧ n < 10 → x n + (n + 2) = S + 20) : 
  Int.floor (| ∑ n in Finset.range 10, x n |) = 15 := 
by
  sorry

end floor_abs_S_l738_738550


namespace one_half_of_scientific_notation_l738_738953

theorem one_half_of_scientific_notation :
  (1 / 2) * (1.2 * 10 ^ 30) = 6.0 * 10 ^ 29 :=
by
  sorry

end one_half_of_scientific_notation_l738_738953


namespace sequence_sum_l738_738459

theorem sequence_sum {A B C D E F G H I J : ℤ} (hD : D = 8)
    (h_sum1 : A + B + C + D = 45)
    (h_sum2 : B + C + D + E = 45)
    (h_sum3 : C + D + E + F = 45)
    (h_sum4 : D + E + F + G = 45)
    (h_sum5 : E + F + G + H = 45)
    (h_sum6 : F + G + H + I = 45)
    (h_sum7 : G + H + I + J = 45)
    (h_sum8 : H + I + J + A = 45)
    (h_sum9 : I + J + A + B = 45)
    (h_sum10 : J + A + B + C = 45) :
  A + J = 0 := 
sorry

end sequence_sum_l738_738459


namespace number_of_friendly_polygons_is_28_l738_738306

-- Definition of friendly polygon
def is_friendly_polygon (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 2 ∧ (let exterior_angle := 360 / n in
  (interior_angle := 180 - exterior_angle) ∧ 
  ((∃ m : ℤ, interior_angle = m) ∨ (∃ m : ℕ, interior_angle = m + 0.5)))

-- The main theorem stating the number of different friendly polygons
theorem number_of_friendly_polygons_is_28 :
  (finset.filter is_friendly_polygon (finset.range 721)).card = 28 :=
sorry

end number_of_friendly_polygons_is_28_l738_738306


namespace number_of_polynomials_in_H_l738_738135

theorem number_of_polynomials_in_H :
  let H := {Q : polynomial ℤ // ∃ (n : ℕ) (c : fin n → ℤ), 
    Q = polynomial.monomial n 1 + ∑ i in range n, polynomial.monomial i (c i) ∧ 
    Q.coeff 0 = 75 ∧ 
    ∃ roots : fin (Q.nat_degree) → ℂ, 
    (∀ i, roots i = a + b * complex.I ∧ a ∈ ℤ ∧ b ∈ ℤ) ∧ 
    multiset.card (Q.roots.map complex.of_real)  = Q.nat_degree
  } ∧
  H.to_finset.card = 656
:=
by
  sorry

end number_of_polynomials_in_H_l738_738135


namespace tangent_polar_equation_l738_738066

theorem tangent_polar_equation :
  (∀ t : ℝ, ∃ (x y : ℝ), x = √2 * Real.cos t ∧ y = √2 * Real.sin t) →
  ∃ ρ θ : ℝ, (x = 1) ∧ (y = 1) → 
  ρ * Real.cos θ + ρ * Real.sin θ = 2 := 
by
  sorry

end tangent_polar_equation_l738_738066


namespace Wayne_blocks_count_l738_738217

-- Statement of the proof problem
theorem Wayne_blocks_count (initial_blocks additional_blocks total_blocks : ℕ) 
  (h1 : initial_blocks = 9) 
  (h2 : additional_blocks = 6) 
  (h3 : total_blocks = initial_blocks + additional_blocks) : 
  total_blocks = 15 := 
by 
  -- proof would go here, but we will use sorry for now
  sorry

end Wayne_blocks_count_l738_738217


namespace mutual_exclusivity_l738_738673

def students : Type := { student // student = "male" ∨ student = "female" }
def male_students : Finset students := { ⟨"male", or.inl rfl⟩, ⟨"male", or.inl rfl⟩, ⟨"male", or.inl rfl⟩ }
def female_students : Finset students := { ⟨"female", or.inr rfl⟩, ⟨"female", or.inr rfl⟩ }

def exactly_one_male (s : Finset students) : Prop := (∃ m ∈ s, m = ⟨"male", or.inl rfl⟩) ∧ (∃ f ∈ s, f = ⟨"female", or.inr rfl⟩) ∧ s.card = 2
def at_least_one_male (s : Finset students) : Prop := ∃ m ∈ s, m = ⟨"male", or.inl rfl⟩
def at_least_one_female (s : Finset students) : Prop := ∃ f ∈ s, f = ⟨"female", or.inr rfl⟩
def all_male (s : Finset students) : Prop := ∀ student ∈ s, student = ⟨"male", or.inl rfl⟩
def all_female (s : Finset students) : Prop := ∀ student ∈ s, student = ⟨"female", or.inr rfl⟩

def pair_A_event := λ s, exactly_one_male s ∧ at_least_one_female s
def pair_B_event := λ s, at_least_one_male s ∧ at_least_one_female s
def pair_C_event := λ s, at_least_one_male s ∧ all_male s
def pair_D_event := λ s, at_least_one_male s ∧ all_female s

theorem mutual_exclusivity : 
  (¬ (pair_A_event = pair_B_event)) ∧
  (¬ (pair_C_event = pair_A_event)) ∧
  (¬ (pair_C_event = pair_B_event)) ∧
  (pair_D_event ∧ ∀ s, (at_least_one_female s → ¬ (all_female s)) ∧ (all_female s → ¬ (at_least_one_female s))) :=
by 
  sorry

end mutual_exclusivity_l738_738673


namespace proveEllipseEq_proveConstCrossProduct_l738_738394
noncomputable theory

def isEllipse (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a > b ∧ 
  let c := Real.sqrt (a^2 - b^2) in
  (c / a) = Real.sqrt 2 / 2 ∧ 
  a * b = Real.sqrt 2 ∧ 
  ∃ P : ℝ × ℝ, 
  P.1^2 / a^2 + P.2^2 / b^2 = 1 ∧
  let D := (0, -2 * b) in
  ∃ triangleArea : ℝ,
  triangleArea = (1 / 2) * 3 * b * a ∧
  triangleArea = (3 * Real.sqrt 2) / 2

def ellipseEq (a b : ℝ) : Prop :=
  (∃ (E : ℝ × ℝ → Prop), (∀ p : ℝ × ℝ, E p ↝ p.1^2 / a^2 + p.2^2 / b^2 = 1))

def crossProductConst (a b : ℝ) : Prop :=
  let c := Real.sqrt (a^2 - b^2) in
  c / a = Real.sqrt 2 / 2 →
  a * b = Real.sqrt 2 →
  ∀ (P Q : ℝ × ℝ), 
  let B : ℝ × ℝ := (0, b) in
  let D : ℝ × ℝ := (0, -2 * b) in
  P.1^2 / a^2 + P.2^2 / b^2 = 1 →
  crossLine D P Q →
  crossLine B P Q →
  let M := (P.1 / (1 - P.2), 0) in
  let N := (Q.1 / (1 - Q.2), 0) in
  (|OM| * |ON| = 2 / 3)

theorem proveEllipseEq : ∃ a b : ℝ, isEllipse a b → ellipseEq a b := sorry

theorem proveConstCrossProduct : ∃ a b : ℝ, isEllipse a b → crossProductConst a b := sorry

end proveEllipseEq_proveConstCrossProduct_l738_738394


namespace angle_projections_l738_738995

-- Define the parabola and its associated properties
def parabola (p : ℝ) : set (ℝ × ℝ) := {point | point.snd ^ 2 = 2 * p * point.fst}

-- Define the focus of the parabola
def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)

-- Define the directrix of the parabola
def directrix (p : ℝ) : set (ℝ × ℝ) := {point | point.fst = -p / 2}

-- Define the projections of points A and B onto the directrix
def projection (point : ℝ × ℝ) (p : ℝ) : ℝ × ℝ := (-p / 2, point.snd)

-- The angle ∠A₁FB₁ is 90° given the conditions
theorem angle_projections (p : ℝ) (A B : ℝ × ℝ)
  (hA : A ∈ parabola p) (hB : B ∈ parabola p) 
  (line_through_focus : ∃ m b, A.snd = m * A.fst + b ∧ B.snd = m * B.fst + b ∧ focus p = (m * (p / 2) + b, 0)) :
  ∠ (projection A p) (focus p) (projection B p) = 90 :=
sorry

end angle_projections_l738_738995


namespace find_p_q_sum_l738_738919

theorem find_p_q_sum (p q : ℝ) 
  (sum_condition : p / 3 = 8) 
  (product_condition : q / 3 = 12) : 
  p + q = 60 :=
by
  sorry

end find_p_q_sum_l738_738919


namespace right_triangle_area_and_side_length_l738_738827

theorem right_triangle_area_and_side_length
  (A B C : Type) [EuclideanSpace ℝ ℝ]
  (angle_BC_eq : ∠B = ∠C)
  (AC_eq : dist A C = 8 * Real.sqrt 2)
  (right_angle_A : ∠A = π / 2) :
  (∃ (Area : ℝ), Area = 64) ∧ (∃ (AB_length : ℝ), AB_length = 16) := 
  sorry

end right_triangle_area_and_side_length_l738_738827


namespace num_coefs_divisible_by_144_l738_738465

theorem num_coefs_divisible_by_144 :
  let polynomial := (2 * X + 3 * Y)^20 in
  let expansion := polynomial.expand_in_terms_of_binomial_theorem in
  let coefficients := expansion.coefficients in
  ∃ n : ℕ, n = 15 ∧
  ∀ k : ℕ, 2 ≤ k ∧ k ≤ 16 → coefficient_in_expansion(coefficients, k) % 144 = 0 :=
begin
  sorry
end

end num_coefs_divisible_by_144_l738_738465


namespace roots_squared_sum_l738_738022

theorem roots_squared_sum (x1 x2 : ℝ) (h₁ : x1^2 - 5 * x1 + 3 = 0) (h₂ : x2^2 - 5 * x2 + 3 = 0) :
  x1^2 + x2^2 = 19 :=
by
  sorry

end roots_squared_sum_l738_738022


namespace exists_smallest_k_l738_738863

noncomputable def smallest_k (n : ℕ) (h : Even n ∧ 0 < n) : ℕ :=
  let t := n / (2 ^ (Nat.find (Nat.find_spec (Nat.exists_eq_mod_right (by available_modulo' n 2)))))
  2 ^ t

theorem exists_smallest_k (n : ℕ) (h : Even n ∧ 0 < n) :
  ∃ k : ℕ, (∃ f g : Polynomial ℤ, k = (Polynomial.eval₂ ℤ ℤ (Polynomial.C 1) f) * (x + 1) ^ n + (Polynomial.eval₂ ℤ ℤ (Polynomial.C 1) g) * (x ^ n + 1))
          ∧ k = smallest_k n h :=
begin
  sorry
end

end exists_smallest_k_l738_738863


namespace toby_change_is_7_l738_738944

def cheeseburger_cost : ℝ := 3.65
def milkshake_cost : ℝ := 2
def coke_cost : ℝ := 1
def large_fries_cost : ℝ := 4
def cookie_cost : ℝ := 0.5
def tax : ℝ := 0.2
def toby_funds : ℝ := 15

def total_food_cost_before_tax : ℝ := 
  2 * cheeseburger_cost + milkshake_cost + coke_cost + large_fries_cost + 3 * cookie_cost

def total_bill_with_tax : ℝ := total_food_cost_before_tax + tax

def each_person_share : ℝ := total_bill_with_tax / 2

def toby_change : ℝ := toby_funds - each_person_share

theorem toby_change_is_7 : toby_change = 7 := by
  sorry

end toby_change_is_7_l738_738944


namespace integer_roots_of_poly_l738_738998

-- Define the polynomial
def poly (x : ℤ) (b1 b2 : ℤ) : ℤ :=
  x^3 + b2 * x ^ 2 + b1 * x + 18

-- The list of possible integer roots
def possible_integer_roots := [-18, -9, -6, -3, -2, -1, 1, 2, 3, 6, 9, 18]

-- Statement of the theorem
theorem integer_roots_of_poly (b1 b2 : ℤ) :
  ∀ x : ℤ, poly x b1 b2 = 0 → x ∈ possible_integer_roots :=
sorry

end integer_roots_of_poly_l738_738998


namespace two_points_same_color_at_distance_one_l738_738913

theorem two_points_same_color_at_distance_one (color : ℝ × ℝ → ℕ) (h : ∀p : ℝ × ℝ, color p < 3) :
  ∃ (p q : ℝ × ℝ), dist p q = 1 ∧ color p = color q :=
sorry

end two_points_same_color_at_distance_one_l738_738913


namespace part_a_part_b_l738_738593

-- Definitions based on the conditions
def square_sum : ℕ := 1 + 2 + 3 + 4

-- There are many identical squares.
-- Let n be the number of squares.

-- Part (a)
theorem part_a (n : ℕ) : ¬ (4 * 2004 = n * square_sum) := sorry

-- Part (b)
theorem part_b : ∃ n : ℕ, 4 * 2005 = n * square_sum :=
  by {
    use 802,
    sorry
  }

end part_a_part_b_l738_738593


namespace number_of_oarsmen_l738_738562

-- Define the conditions
variables (n : ℕ)
variables (W : ℕ)
variables (h_avg_increase : (W + 40) / n = W / n + 2)

-- Lean 4 statement without the proof
theorem number_of_oarsmen : n = 20 :=
by
  sorry

end number_of_oarsmen_l738_738562


namespace angle_bisector_length_in_triangle_XYZ_l738_738474

noncomputable def length_of_angle_bisector (XY XZ: ℝ) (cos_angle_X: ℝ) : ℝ :=
  let YZ := Math.sqrt (XY ^ 2 + XZ ^ 2 - 2 * XY * XZ * cos_angle_X)
  let YD := (1 / 3) * YZ
  let term1 := XY ^ 2
  let term2 := (YD ^ 2)
  let term3 := 2 * XY * YD * cos_angle_X / 3 * 1/10
  Math.sqrt (term1 + term2 - term3)

theorem angle_bisector_length_in_triangle_XYZ :
  XY = 4 → XZ = 8 → cos_angle_X = 1/10 → 
  length_of_angle_bisector XY XZ cos_angle_X = 
    Math.sqrt (16 + 73.6 / 9 - 8 / 15 * Math.sqrt 73.6) :=
by
  sorry

end angle_bisector_length_in_triangle_XYZ_l738_738474


namespace octahedron_faces_eq_8_l738_738077

-- Define what an octahedron is in terms of faces
def is_octahedron (P : Type) [polyhedron P] : Prop :=
  ∃ n : ℕ, faces P = n ∧ n = 8

theorem octahedron_faces_eq_8 (P : Type) [polyhedron P] (h : is_octahedron P) : faces P = 8 :=
by sorry

end octahedron_faces_eq_8_l738_738077


namespace identity_function_l738_738857

theorem identity_function (f : ℕ → ℕ) (h : ∀ n : ℕ, f (n + 1) > f (f n)) : ∀ n : ℕ, f n = n :=
by
  sorry

end identity_function_l738_738857


namespace only_pos_integral_values_of_a_l738_738751

theorem only_pos_integral_values_of_a (a : ℕ) :
  (∀ x : ℕ, (3 * x > 4 * x - 4) → (4 * x - a > -8) → x = 3) ↔ (16 ≤ a ∧ a < 20) :=
by
  intros
  sorry

end only_pos_integral_values_of_a_l738_738751


namespace smallest_number_of_cubes_filling_box_l738_738285
open Nat

theorem smallest_number_of_cubes_filling_box (L W D : ℕ) (hL : L = 27) (hW : W = 15) (hD : D = 6) :
  let gcd := 3
  let cubes_along_length := L / gcd
  let cubes_along_width := W / gcd
  let cubes_along_depth := D / gcd
  cubes_along_length * cubes_along_width * cubes_along_depth = 90 :=
by
  sorry

end smallest_number_of_cubes_filling_box_l738_738285


namespace complement_of_M_in_U_l738_738797

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {x | x^2 - 5*x + 6 = 0}
def C_U (M : Set ℕ) (U : Set ℕ) : Set ℕ := U \ M

theorem complement_of_M_in_U : C_U M U = {1, 4} :=
by
  sorry

end complement_of_M_in_U_l738_738797


namespace sum_of_possible_remainders_l738_738900

theorem sum_of_possible_remainders (n : ℕ) (h_even : ∃ k : ℕ, n = 2 * k) : 
  let m := 1000 * (2 * n + 6) + 100 * (2 * n + 4) + 10 * (2 * n + 2) + (2 * n)
  let remainder (k : ℕ) := (1112 * k + 6420) % 29
  23 + 7 + 20 = 50 :=
  by
  sorry

end sum_of_possible_remainders_l738_738900


namespace basketball_scores_l738_738978

theorem basketball_scores : ∃ (scores : Finset ℕ), 
  scores = { x | ∃ a b : ℕ, a + b = 7 ∧ x = 2 * a + 3 * b } ∧ scores.card = 8 :=
by
  sorry

end basketball_scores_l738_738978


namespace arithmetic_sequences_count_l738_738142

theorem arithmetic_sequences_count (n : ℕ) (S : Finset ℕ) (A : Finset ℕ)
  (hS : S = finset.range n + 1)
  (hA : A ⊆ S)
  (hA_arith : ∃ d : ℕ, d > 0 ∧ (∀ x ∈ A, ∃ k : ℕ, x = some (A.min') + k * d)) : 
  A.card ≥ 2 ∧ (∀ x ∈ S, ¬∃ d : ℕ, d > 0 ∧ (∀ y ∈ A ∪ {x}, ∃ k : ℕ, y = some ((A ∪ {x}).min') + k * d)) →
  A.card = ⌊ n^2 / 4 ⌋ :=
sorry

end arithmetic_sequences_count_l738_738142


namespace units_digit_R_54321_l738_738549

noncomputable def c := 4 + Real.sqrt 15
noncomputable def d := 4 - Real.sqrt 15
def R (n : ℕ) : ℝ := 1 / 2 * (c ^ n + d ^ n)

-- Define the recurrence relation
def recurrence_relation (n : ℕ) : Prop :=
  R (n + 1) = 8 * R n - R (n - 1)

theorem units_digit_R_54321 : (R 54321) % 10 = 4 := 
sorry

end units_digit_R_54321_l738_738549


namespace imaginary_part_of_1_div_1_sub_i_eq_one_half_l738_738655

def imaginary_unit_1_div_1_sub_i : ℂ := (1 / (1 - complex.I))

theorem imaginary_part_of_1_div_1_sub_i_eq_one_half : complex.im (imaginary_unit_1_div_1_sub_i) = 1 / 2 :=
sorry

end imaginary_part_of_1_div_1_sub_i_eq_one_half_l738_738655


namespace quadrilateral_area_not_constant_l738_738602

variables (O A B P Q R : Type) 
variables [Point O][Point A][Point B][Point P][Point Q][Point R]
variables (f : Point O → Point) (g : Point O → Point)

-- Problem assumptions
axiom O_circle_radius_one : ∀ O, ∃ OA OB : Point, orthogonal OA OB ∧ radius O OA = 1 ∧ radius O OB = 1
axiom P_circumference : ∀ P, ∃ O, radius O P = 1 ∧ P ≠ OA ∧ P ≠ OB 
axiom Q_intersection : ∀ Q, ∃ O A P, line Q P ∧ line Q OA 
axiom R_intersection : ∀ R, ∃ O B P, line R P ∧ line R OB 

-- Proof problem statement
theorem quadrilateral_area_not_constant (O A B P Q R : Type) (x y : ℝ) :
  (1 + x) * (1 + y) = 2 ∧
  x = cos (angle O P / sin (angle O P) + 1) ∧
  y = sin (angle O P / cos (angle O P) + 1) → 
  ∃ P O, ∀ A B Q R, P_circumference P O ∧ Q_intersection Q OA ∧ R_intersection R OB →
  ¬ constant (area (quadrilateral O A B Q R)).
  sorry

end quadrilateral_area_not_constant_l738_738602


namespace planting_schemes_count_l738_738323

-- Define the conditions
structure PlantingScheme where
  regions : Fin 6 → Fin 4
  adjacent_regions_have_different_plants : ∀ i j : Fin 6, is_adjacent i j → regions i ≠ regions j

def is_adjacent (i j : Fin 6) : Prop :=
  -- Define adjacency based on the regular hexagon
  (i.val = (j.val + 1) % 6) ∨ (i.val = (j.val + 5) % 6)

-- The main theorem statement
theorem planting_schemes_count : 
  ∃ (schemes : Fin 6 → Fin 4 → PlantingScheme), 
  (card (set_of schemes)).to_nat = 732 :=
  sorry

end planting_schemes_count_l738_738323


namespace walnuts_distribution_l738_738367

noncomputable def initial_walnuts : ℕ :=
  15621

theorem walnuts_distribution (x : ℕ) :
  let initial := initial_walnuts in
  let fifth_share := (initial - 1) / 5 in
  let fourth_share := (5 * fifth_share + 1 - 1) / 5 in
  let third_share := (5 * fourth_share + 1 - 1) / 5 in
  let second_share := (5 * third_share + 1 - 1) / 5 in
  let first_share := (5 * second_share + 1 - 1) / 5 in
  (5 * first_share + 1 - 1) / 5 = x →
  1 ≤ fifth_share ∧ 1 ≤ fourth_share ∧ 1 ≤ third_share ∧ 1 ≤ second_share ∧ 1 ≤ first_share ∧
  initial_walnuts = 15621 :=
begin
  sorry
end

end walnuts_distribution_l738_738367


namespace count_perfect_cubes_between_10_and_2000_l738_738079

theorem count_perfect_cubes_between_10_and_2000 : 
  (∃ n_min n_max, n_min^3 ≥ 10 ∧ n_max^3 ≤ 2000 ∧ 
  (n_max - n_min + 1 = 10)) := 
sorry

end count_perfect_cubes_between_10_and_2000_l738_738079


namespace price_of_when_you_rescind_cd_l738_738082

variable (W : ℕ) -- Defining W as a natural number since prices can't be negative

theorem price_of_when_you_rescind_cd
  (price_life_journey : ℕ := 100)
  (price_day_life : ℕ := 50)
  (num_cds_each : ℕ := 3)
  (total_spent : ℕ := 705) :
  3 * price_life_journey + 3 * price_day_life + 3 * W = total_spent → 
  W = 85 :=
by
  intros h
  sorry

end price_of_when_you_rescind_cd_l738_738082


namespace total_numbers_is_eight_l738_738561

theorem total_numbers_is_eight
  (avg_all : ∀ n : ℕ, (total_sum : ℝ) / n = 25)
  (avg_first_two : ∀ a₁ a₂ : ℝ, (a₁ + a₂) / 2 = 20)
  (avg_next_three : ∀ a₃ a₄ a₅ : ℝ, (a₃ + a₄ + a₅) / 3 = 26)
  (h_sixth : ∀ a₆ a₇ a₈ : ℝ, a₆ + 4 = a₇ ∧ a₆ + 6 = a₈)
  (last_num : ∀ a₈ : ℝ, a₈ = 30) :
  ∃ n : ℕ, n = 8 :=
by
  sorry

end total_numbers_is_eight_l738_738561


namespace lines_parallel_if_perpendicular_to_plane_l738_738657

variables {α β γ : Plane} {m n : Line}

-- Define the properties of perpendicular lines to planes and parallel lines
def perpendicular_to (l : Line) (p : Plane) : Prop := 
sorry -- definition skipped

def parallel_to (l1 l2 : Line) : Prop := 
sorry -- definition skipped

-- Theorem Statement (equivalent translation of the given question and its correct answer)
theorem lines_parallel_if_perpendicular_to_plane 
  (h1 : perpendicular_to m α) 
  (h2 : perpendicular_to n α) : parallel_to m n :=
sorry

end lines_parallel_if_perpendicular_to_plane_l738_738657


namespace minimum_banks_needed_l738_738314

-- Condition definitions
def total_amount : ℕ := 10000000
def max_insurance_payout_per_bank : ℕ := 1400000

-- Theorem statement
theorem minimum_banks_needed :
  ∃ n : ℕ, n * max_insurance_payout_per_bank ≥ total_amount ∧ n = 8 :=
sorry

end minimum_banks_needed_l738_738314


namespace paper_cutting_sum_l738_738483

theorem paper_cutting_sum (n : ℕ) (k : ℕ) (a : Fin n → ℝ)
  (h_n : 0 < n) (h_k : k < n) (h_a : ∀ i, 0 ≤ a i ∧ a i ≤ 1) :
  ∃ (m : List (Fin n)), (List.length m = k + 1) ∧ 
  let pieces := (m.zip (m.tail)).map (λ ⟨i,j⟩, (List.rangeBetween i j).sum (λ k, a k)) in
  ∀ i j, i < pieces.length → j < pieces.length → |pieces.nth i - pieces.nth j| ≤ 1 :=
sorry

end paper_cutting_sum_l738_738483


namespace number_of_possible_multisets_l738_738341

theorem number_of_possible_multisets {a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ : ℤ}
  (h₀ : a₁₂ ≠ 0) (h₁ : a₀ ≠ 0) :
  let p(x : ℤ) := a₁₂ * x^12 + a₁₁ * x^11 + a₁₀ * x^10 + a₉ * x^9 + a₈ * x^8 + a₇ * x^7 +
                    a₆ * x^6 + a₅ * x^5 + a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀,
      q(x : ℤ) := a₀ * x^12 + a₁ * x^11 + a₂ * x^10 + a₃ * x^9 + a₄ * x^8 + a₅ * x^7 +
                    a₆ * x^6 + a₇ * x^5 + a₈ * x^4 + a₉ * x^3 + a₁₀ * x^2 + a₁₁ * x + a₁₂ in
  (∃ S : multiset ℤ, S.card = 12 ∧ ∀ r ∈ S, p(r) = 0 ∧ q(r) = 0 ∧ (r = 1 ∨ r = -1)) →
  ∃ (k : ℕ), k = 0 ∧ k ≤ 12 ∧ k + (12 - k) = 12 ∧ ∃ S' : finset ℕ, S'.card = 13 := 
sorry

end number_of_possible_multisets_l738_738341


namespace find_taller_tree_height_l738_738588

-- Define the known variables and conditions
variables (H : ℕ) (ratio : ℚ) (difference : ℕ)

-- Specify the conditions from the problem
def taller_tree_height (H difference : ℕ) := H
def shorter_tree_height (H difference : ℕ) := H - difference
def height_ratio (H : ℕ) (ratio : ℚ) (difference : ℕ) :=
  (shorter_tree_height H difference : ℚ) / (taller_tree_height H difference : ℚ) = ratio

-- Prove the height of the taller tree given the conditions
theorem find_taller_tree_height (H : ℕ) (h_ratio : height_ratio H (2/3) 20) : 
  taller_tree_height H 20 = 60 :=
  sorry

end find_taller_tree_height_l738_738588


namespace second_strategy_more_economical_l738_738613

-- Define the conditions
variables {p1 p2 x y : ℝ}
hypothesis (hp1 : p1 > 0) (hp2 : p2 > 0) (hx : x > 0) (hy : y > 0)

-- Define the average prices for both strategies
def avg_price_first_strategy := (p1 + p2) / 2
def avg_price_second_strategy := (2 * p1 * p2) / (p1 + p2)

-- The theorem to prove
theorem second_strategy_more_economical 
  (hp1 : p1 > 0) (hp2 : p2 > 0) (hx : x > 0) (hy : y > 0) :
  avg_price_second_strategy p1 p2 ≤ avg_price_first_strategy p1 p2 :=
by
  sorry

end second_strategy_more_economical_l738_738613


namespace solve_system_of_equations_l738_738425

theorem solve_system_of_equations :
  ∀ (x y : ℝ),
  (3 * x - 2 * y = 7) →
  (2 * x + 3 * y = 8) →
  x = 37 / 13 :=
by
  intros x y h1 h2
  -- to prove x = 37 / 13 from the given system of equations
  sorry

end solve_system_of_equations_l738_738425


namespace FO_greater_DI_l738_738107

-- The quadrilateral FIDO is assumed to be convex with specified properties
variables {F I D O E : Type*}

variables (length_FI length_DI length_DO length_FO : ℝ)
variables (angle_FIO angle_DIO : ℝ)
variables (E : I)

-- Given conditions
variables (convex_FIDO : Prop) -- FIDO is convex
variables (h1 : length_FI = length_DO)
variables (h2 : length_FI > length_DI)
variables (h3 : angle_FIO = angle_DIO)

-- Use given identity IE = ID
variables (length_IE : ℝ) (h4 : length_IE = length_DI)

theorem FO_greater_DI 
    (length_FI length_DI length_DO length_FO : ℝ)
    (angle_FIO angle_DIO : ℝ)
    (convex_FIDO : Prop)
    (h1 : length_FI = length_DO)
    (h2 : length_FI > length_DI)
    (h3 : angle_FIO = angle_DIO)
    (length_IE : ℝ)
    (h4 : length_IE = length_DI) : 
    length_FO > length_DI :=
sorry

end FO_greater_DI_l738_738107
