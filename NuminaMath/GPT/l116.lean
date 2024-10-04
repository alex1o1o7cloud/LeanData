import Mathlib

namespace max_value_of_sqrt_sum_l116_116245

theorem max_value_of_sqrt_sum (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (ha2 : a ≤ 2) (hb2 : b ≤ 2) (hc2 : c ≤ 2) :
  ∃ x, x = Real.sqrt ((2 - a) * (2 - b) * (2 - c)) + Real.sqrt (a * b * c) ∧ x ≤ 2 :=
begin
  use 2,
  split,
  {
    sorry, -- This is where the proof would go
  },
  {
    sorry, -- This is where the inequality would be shown
  }
end

end max_value_of_sqrt_sum_l116_116245


namespace range_of_m_l116_116128

def vect_a : ℝ × ℝ := (1, 2)
def vect_b (m : ℝ) : ℝ × ℝ := (m - 1, m + 3)

theorem range_of_m (m : ℝ) : (∀ c : ℝ × ℝ, ∃ λ μ : ℝ, c = (λ * vect_a.1 + μ * (vect_b m).1, λ * vect_a.2 + μ * (vect_b m).2)) ↔ m ≠ 5 :=
sorry

end range_of_m_l116_116128


namespace pascal_remaining_miles_l116_116701

section PascalCyclingTrip

variable (D T : ℝ)
variable (V : ℝ := 8)
variable (V_reduced V_increased : ℝ)
variable (time_diff : ℝ := 16)

-- Definitions of reduced and increased speeds
def V_reduced := V - 4
def V_increased := V + 0.5 * V

-- Conditions for the distance
def distance_current := D = V * T
def distance_increased := D = V_increased * (T - time_diff)
def distance_reduced := D = V_reduced * (T + time_diff)

theorem pascal_remaining_miles (h1 : distance_current)
  (h2 : distance_increased)
  (h3 : distance_reduced) : D = 256 := 
sorry

end PascalCyclingTrip

end pascal_remaining_miles_l116_116701


namespace characterize_solution_l116_116260

def is_odd (n : ℤ) := ∃ (k : ℤ), n = 2 * k + 1

def solution (f : ℤ → ℤ) : Prop :=
  (∀ x, is_odd (f x)) ∧
  (∀ x y, f (x + f x + y) + f (x - f x - y) = f (x + y) + f (x - y)) ∧
  (∃ d k : ℤ, d > 0 ∧ is_odd d ∧ k > 0 ∧
  (∀ m : ℤ, ∀ i : ℤ, 0 ≤ i ∧ i < d → 
    ∃ l_i : ℤ, is_odd l_i ∧ f (m * d + i) = 2 * k * m * d + l_i * d))

theorem characterize_solution : ∀ (f : ℤ → ℤ), solution f → 
  ∀ x y, f (x + f x + y) + f (x - f x - y) = f (x + y) + f (x - y) :=
by
  introv Hsol
  rw solution at Hsol
  cases Hsol with Hodd Hrest
  cases Hrest with Hfeq Hstruct
  exact Hfeq

end characterize_solution_l116_116260


namespace count_numbers_containing_2_or_5_l116_116478

theorem count_numbers_containing_2_or_5 : 
  (Π n, 200 ≤ n ∧ n < 500 → (n.to_string.contains '2' ∨ n.to_string.contains '5')) :=
begin 
  sorry
end

def numbers_containing_2_or_5 : ℕ := 
  (Π n, 200 ≤ n ∧ n < 500 → (n.to_string.contains '2' ∨ n.to_string.contains '5')).count

example : numbers_containing_2_or_5 = 230 :=
by sorry

end count_numbers_containing_2_or_5_l116_116478


namespace opposite_of_negative_2023_l116_116793

theorem opposite_of_negative_2023 : (- -2023) = 2023 := 
by {
sorry
}

end opposite_of_negative_2023_l116_116793


namespace find_k_l116_116615

noncomputable theory

def B := (Real.cos (60 * Real.pi / 180), -Real.sqrt 3)
def A := (B.1, -B.2)

theorem find_k : ∃ k, (A.2 = k / A.1) ∧ k = Real.sqrt 3 / 2 :=
by
  use Real.sqrt 3 / 2
  split
  sorry
  sorry

end find_k_l116_116615


namespace domain_of_f_f_strictly_increasing_l116_116559

open Real

noncomputable def f (x : ℝ) : ℝ := log (1 / 2) ((1 / 2)^x - 1)

theorem domain_of_f : ∀ x, x < 0 → ∃ y, y = f x :=
by
  intro x hx
  use f x
  sorry

theorem f_strictly_increasing : ∀ x y, x < 0 → y < 0 → x < y → f x < f y :=
by
  intros x y hx hy hxy
  sorry

end domain_of_f_f_strictly_increasing_l116_116559


namespace prime_condition_l116_116416

theorem prime_condition (p : ℕ) (h_prime : Nat.prime p) (h : Nat.prime (2^(p+1) + p^3 - p^2 - p)) : p = 3 := 
sorry

end prime_condition_l116_116416


namespace opposite_neg_2023_l116_116776

theorem opposite_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_neg_2023_l116_116776


namespace average_of_remaining_two_numbers_l116_116407

theorem average_of_remaining_two_numbers
  (a b c d e f : ℝ)
  (h1 : (a + b + c + d + e + f) / 6 = 3.9)
  (h2 : (a + b) / 2 = 3.4)
  (h3 : (c + d) / 2 = 3.85) :
  (e + f) / 2 = 4.45 :=
sorry

end average_of_remaining_two_numbers_l116_116407


namespace find_d_value_l116_116142

theorem find_d_value (d : ℝ) :
  (∀ x, (8 * x^3 + 27 * x^2 + d * x + 55 = 0) → (2 * x + 5 = 0)) → d = 39.5 :=
by
  sorry

end find_d_value_l116_116142


namespace remaining_distance_proof_l116_116709

-- Define the conditions
def pascal_current_speed : ℝ := 8
def pascal_reduced_speed : ℝ := pascal_current_speed - 4
def pascal_increased_speed : ℝ := pascal_current_speed * 1.5

-- Define the remaining distance in terms of the current speed and time taken
noncomputable def remaining_distance (T : ℝ) : ℝ := pascal_current_speed * T

-- Define the times with the increased and reduced speeds
noncomputable def time_with_increased_speed (T : ℝ) : ℝ := T - 16
noncomputable def time_with_reduced_speed (T : ℝ) : ℝ := T + 16

-- Define the distances using increased and reduced speeds
noncomputable def distance_increased_speed (T : ℝ) : ℝ := pascal_increased_speed * (time_with_increased_speed T)
noncomputable def distance_reduced_speed (T : ℝ) : ℝ := pascal_reduced_speed * (time_with_reduced_speed T)

-- Main theorem stating that the remaining distance is 256 miles
theorem remaining_distance_proof (T : ℝ) (ht_eq: pascal_current_speed * T = 256) : 
  remaining_distance T = 256 := by
  sorry

end remaining_distance_proof_l116_116709


namespace opposite_of_neg2023_l116_116994

theorem opposite_of_neg2023 : ∀ (x : Int), -2023 + x = 0 → x = 2023 :=
by
  intros x h
  sorry

end opposite_of_neg2023_l116_116994


namespace orthocenter_circumradii_equal_l116_116409

-- Define a triangle with its orthocenter and circumradius
variables {A B C H : Point} (R r : ℝ)

-- Assume H is the orthocenter of triangle ABC
def is_orthocenter (H : Point) (A B C : Point) : Prop := 
  sorry -- This should state the definition or properties of an orthocenter

-- Assume the circumradius of triangle ABC is R 
def is_circumradius_ABC (A B C : Point) (R : ℝ) : Prop :=
  sorry -- This should capture the circumradius property

-- Assume circumradius of triangle BHC is r
def is_circumradius_BHC (B H C : Point) (r : ℝ) : Prop :=
  sorry -- This should capture the circumradius property
  
-- Prove that if H is the orthocenter of triangle ABC, the circumradius of ABC is R 
-- and the circumradius of BHC is r, then R = r
theorem orthocenter_circumradii_equal (h_orthocenter : is_orthocenter H A B C) 
  (h_circumradius_ABC : is_circumradius_ABC A B C R)
  (h_circumradius_BHC : is_circumradius_BHC B H C r) : R = r :=
  sorry

end orthocenter_circumradii_equal_l116_116409


namespace angle_B_possibilities_l116_116251

theorem angle_B_possibilities (O H : Point) (A B C : Point) (R b : ℝ)
  (circumcenter : is_circumcenter O A B C)
  (orthocenter : is_orthocenter H A B C)
  (BO_eq_BH : dist B O = dist B H) :
  ∠BAC = 60 ∨ ∠BAC = 120 :=
sorry

end angle_B_possibilities_l116_116251


namespace lattice_points_on_hyperbola_l116_116476

open Real

theorem lattice_points_on_hyperbola : 
  ∃ (S : Finset (ℤ × ℤ)), 
  (∀ (x y : ℤ), ((x, y) ∈ S ↔ x^2 - y^2 = 65)) ∧ S.card = 8 :=
by
  sorry

end lattice_points_on_hyperbola_l116_116476


namespace minimum_value_minimum_value_achieved_l116_116509

theorem minimum_value (x : ℝ) (h : x > 0) : 3 * x + x⁻³ ≥ 4 :=
sorry

theorem minimum_value_achieved (x : ℝ) (h : x = 1) : 3 * x + x⁻³ = 4 :=
sorry

end minimum_value_minimum_value_achieved_l116_116509


namespace tan_angle_addition_l116_116146

theorem tan_angle_addition (x : Real) (h1 : Real.tan x = 3) (h2 : Real.tan (Real.pi / 3) = Real.sqrt 3) : 
  Real.tan (x + Real.pi / 3) = (3 + Real.sqrt 3) / (1 - 3 * Real.sqrt 3) :=
by 
  sorry

end tan_angle_addition_l116_116146


namespace midpoint_hexagon_perimeter_inequality_l116_116636

noncomputable def hexagon := sorry

def is_hexagon_with_equal_angles (hex : hexagon) : Prop := sorry

def midpoints (hex : hexagon) : list (hexagon → (ℝ × ℝ)) := sorry

theorem midpoint_hexagon_perimeter_inequality {hex : hexagon} 
  (h_eq_angles : is_hexagon_with_equal_angles hex)
  (mid_pts : list (hexagon → (ℝ × ℝ)) := midpoints hex) :
  perimeter (midpoint_hexagon mid_pts hex) ≥ 
  (sqrt 3 / 2) * perimeter hex := 
sorry

end midpoint_hexagon_perimeter_inequality_l116_116636


namespace tan_shifted_value_l116_116166

theorem tan_shifted_value (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = -((6 + 5 * Real.sqrt 3) / 13) :=
by
  sorry

end tan_shifted_value_l116_116166


namespace angle_B_possibilities_l116_116253

theorem angle_B_possibilities (O H : Point) (A B C : Point) (R b : ℝ)
  (circumcenter : is_circumcenter O A B C)
  (orthocenter : is_orthocenter H A B C)
  (BO_eq_BH : dist B O = dist B H) :
  ∠BAC = 60 ∨ ∠BAC = 120 :=
sorry

end angle_B_possibilities_l116_116253


namespace divide_square_diagonals_into_four_congruent_triangles_l116_116035

theorem divide_square_diagonals_into_four_congruent_triangles (S : Set (ℝ × ℝ)) (hS : S = {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}) :
  (∀ T : Set (ℝ × ℝ), T ⊆ S → (∃ P : Finset (Set (ℝ × ℝ)), ∀ t ∈ P, t = {p | p.1 + p.2 = 1 ∨ p.1 - p.2 = 0 ∨ p.2 = 0 ∨ p.1 = 0}) →
    (∃ R : Finset (Set (ℝ × ℝ)), ∀ r ∈ R, r ⊆ S ∧ ∃ Q : Finset (Set (ℝ × ℝ)), ∀ q ∈ Q, q = {p | (p.1 ∈ {0, 1} ∧ 0 ≤ p.2 ∧ p.2 ≤ 1) ∨ (p.2 ∈ {0, 1} ∧ 0 ≤ p.1 ∧ p.1 ≤ 1)} ∧ Congruent q r) :=
sorry

end divide_square_diagonals_into_four_congruent_triangles_l116_116035


namespace neither_5_nor_6_nice_1200_l116_116510

def is_k_nice (N k : ℕ) : Prop := N % k = 1

def count_k_nice_up_to (k n : ℕ) : ℕ :=
(n + (k - 1)) / k

def count_neither_5_nor_6_nice_up_to (n : ℕ) : ℕ :=
  let count_5_nice := count_k_nice_up_to 5 n
  let count_6_nice := count_k_nice_up_to 6 n
  let count_5_and_6_nice := count_k_nice_up_to 30 n
  n - (count_5_nice + count_6_nice - count_5_and_6_nice)

theorem neither_5_nor_6_nice_1200 : count_neither_5_nor_6_nice_up_to 1200 = 800 := 
by
  sorry

end neither_5_nor_6_nice_1200_l116_116510


namespace pi_minus_4_greatest_integer_l116_116336

def greatest_integer_function (x : ℝ) : ℤ :=
  int.floor x

theorem pi_minus_4_greatest_integer :
  greatest_integer_function (Real.pi - 4) = -1 :=
by 
  have h1 : 3 < Real.pi := Real.pi_gt_3
  have h2 : Real.pi < 4 := Real.pi_lt_4
  have h : -1 < Real.pi - 4 ∧ Real.pi - 4 < 0 := ⟨by linarith, by linarith⟩
  sorry  -- Here the proof is omitted as per instructions

end pi_minus_4_greatest_integer_l116_116336


namespace construction_doors_needed_l116_116427

theorem construction_doors_needed :
  (let number_of_buildings := 2 in
   let floors_per_building := 12 in
   let apartments_per_floor := 6 in
   let doors_per_apartment := 7 in
   let apartments_per_building := floors_per_building * apartments_per_floor in
   let total_apartments := apartments_per_building * number_of_buildings in
   let total_doors := total_apartments * doors_per_apartment in
   total_doors = 1008) :=
begin
  sorry
end

end construction_doors_needed_l116_116427


namespace tan_angle_addition_l116_116148

theorem tan_angle_addition (x : Real) (h1 : Real.tan x = 3) (h2 : Real.tan (Real.pi / 3) = Real.sqrt 3) : 
  Real.tan (x + Real.pi / 3) = (3 + Real.sqrt 3) / (1 - 3 * Real.sqrt 3) :=
by 
  sorry

end tan_angle_addition_l116_116148


namespace opposite_of_neg2023_l116_116988

theorem opposite_of_neg2023 : ∀ (x : Int), -2023 + x = 0 → x = 2023 :=
by
  intros x h
  sorry

end opposite_of_neg2023_l116_116988


namespace solution_set_inequality_l116_116527

-- Definitions for the problem
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)
variable (e : ∀ x, Math.exp x)

-- Conditions
axiom f_deriv : ∀ x, deriv f x = f' x
axiom f_initial_condition : f 0 = 2
axiom f_inequality : ∀ x, f x + f' x > 1

-- Theorem to prove
theorem solution_set_inequality : { x : ℝ | e x * f x > e x + 1 } = { x : ℝ | x > 0 } := 
by 
  sorry

end solution_set_inequality_l116_116527


namespace area_of_triangle_OAB_is_correct_l116_116095

noncomputable def ellipse_eq := (x^2 / 16) + (y^2 / 4) = 1
def line_eq := y = x + 2
def area_OAB := 16 / 5

theorem area_of_triangle_OAB_is_correct :
  ∀ x y : ℝ,
  ellipse_eq → line_eq →
  (∃ A B : ℝ × ℝ, 
    (ellipse_eq A.1 A.2) ∧ (line_eq A.1 A.2) ∧ (ellipse_eq B.1 B.2) ∧ (line_eq B.1 B.2) ∧
    area_of_triangle O A B = area_OAB) := 
sorry

end area_of_triangle_OAB_is_correct_l116_116095


namespace opposite_of_neg_2023_l116_116890

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116890


namespace find_fraction_inverses_l116_116666

noncomputable def a : ℕ → ℝ
| 0       := -4
| (n + 1) := a n + b n + real.sqrt ((a n)^2 + (b n)^2)

noncomputable def b : ℕ → ℝ
| 0       := 2
| (n + 1) := a n + b n - real.sqrt ((a n)^2 + (b n)^2)

theorem find_fraction_inverses (n : ℕ) : 
  (∑ i in range n, 1 / a i + 1 / b i)  = 1/4 := 
sorry

end find_fraction_inverses_l116_116666


namespace opposite_of_neg_2023_l116_116941

theorem opposite_of_neg_2023 : ∃ x : ℤ, -2023 + x = 0 ∧ x = 2023 := 
by 
  use 2023
  constructor
  · exact rfl
  · exact rfl

end opposite_of_neg_2023_l116_116941


namespace simplify_cube_root_l116_116395

theorem simplify_cube_root :
  ∃ a b : ℕ, a = 5 ∧ b = 36 ∧ a + b = 41 ∧ (∃ c : ℝ, (c ^ 3 = 4500) → c = a * real.cbrt b) :=
by
  use 5
  use 36
  repeat { split }
  sorry

end simplify_cube_root_l116_116395


namespace number_of_boys_l116_116470

theorem number_of_boys
  (x y : ℕ) 
  (h1 : x + y = 43)
  (h2 : 24 * x + 27 * y = 1101) : 
  x = 20 := by
  sorry

end number_of_boys_l116_116470


namespace complex_numbers_not_comparable_l116_116299

-- Definitions based on conditions
def is_real (z : ℂ) : Prop := ∃ r : ℝ, z = r
def is_not_entirely_real (z : ℂ) : Prop := ¬ is_real z

-- Proof problem statement
theorem complex_numbers_not_comparable (z1 z2 : ℂ) (h1 : is_not_entirely_real z1) (h2 : is_not_entirely_real z2) : 
  ¬ (z1.re = z2.re ∧ z1.im = z2.im) :=
sorry

end complex_numbers_not_comparable_l116_116299


namespace find_m_at_min_value_l116_116560

noncomputable def f (x m : ℝ) : ℝ := (x - m)^2 + (Real.log x - 2 * m)^2

theorem find_m_at_min_value : ∃ m : ℝ, (∀ x : ℝ, f x m ≥ f x (\frac {1}{10} - \frac {2}{5} * Real.log 2)) :=
sorry

end find_m_at_min_value_l116_116560


namespace symmetric_line_eq_l116_116503

theorem symmetric_line_eq (x y : ℝ) :
  (y = 2 * x + 3) → (y - 1 = x + 1) → (x - 2 * y = 0) :=
by
  intros h1 h2
  sorry

end symmetric_line_eq_l116_116503


namespace midpoint_sum_four_times_l116_116387

theorem midpoint_sum_four_times (x1 y1 x2 y2 : ℝ) (h1 : x1 = 8) (h2 : y1 = -4) (h3 : x2 = -2) (h4 : y2 = 10) :
  4 * ((x1 + x2) / 2 + (y1 + y2) / 2) = 24 :=
by
  rw [h1, h2, h3, h4]
  -- simplifying to get the desired result
  sorry

end midpoint_sum_four_times_l116_116387


namespace fourth_root_of_sum_of_four_powers_l116_116391

theorem fourth_root_of_sum_of_four_powers (A : ℝ) (h : A = 4^6 + 4^6 + 4^6 + 4^6) :
  Real.root 4 A = 8 * Real.sqrt 2 :=
by
  sorry

end fourth_root_of_sum_of_four_powers_l116_116391


namespace probability_of_9_heads_in_12_flips_l116_116384

theorem probability_of_9_heads_in_12_flips :
  (∃ n : ℕ, n = 12) →
  (1 / (2 ^ 12) * (∑ k in finset.range (12 + 1), if k = 9 then (nat.choose 12 9) else 0)) = (55 / 1024) :=
by
  sorry

end probability_of_9_heads_in_12_flips_l116_116384


namespace opposite_neg_2023_l116_116773

theorem opposite_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_neg_2023_l116_116773


namespace f_positive_if_ineq_holds_l116_116592

theorem f_positive_if_ineq_holds (f : ℝ → ℝ) (h : ∀ x, differentiable_at ℝ f x) 
  (h1 : ∀ x, 2 * f x + x * deriv f x > 0) : ∀ x, f x > 0 := 
by
  sorry

end f_positive_if_ineq_holds_l116_116592


namespace intersection_eq_singleton_l116_116125

-- Defining the sets M and N
def M : Set ℤ := {-1, 1, -2, 2}
def N : Set ℤ := {1, 4}

-- Stating the intersection problem
theorem intersection_eq_singleton :
  M ∩ N = {1} := 
by 
  sorry

end intersection_eq_singleton_l116_116125


namespace opposite_neg_2023_l116_116767

theorem opposite_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_neg_2023_l116_116767


namespace opposite_of_neg_2023_l116_116872

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116872


namespace binomial_expansion_l116_116086

theorem binomial_expansion {a : ℝ} (h : 0 < a) :
  let coefficient := (Nat.choose 6 4) * (a ^ 2) in
  coefficient = 135 → a = 3 :=
by
  intros coefficient hc
  sorry

end binomial_expansion_l116_116086


namespace total_pages_in_book_l116_116033

def pagesReadMonday := 23
def pagesReadTuesday := 38
def pagesReadWednesday := 61
def pagesReadThursday := 12
def pagesReadFriday := 2 * pagesReadThursday

def totalPagesRead := pagesReadMonday + pagesReadTuesday + pagesReadWednesday + pagesReadThursday + pagesReadFriday

theorem total_pages_in_book :
  totalPagesRead = 158 :=
by
  sorry

end total_pages_in_book_l116_116033


namespace greg_and_earl_total_l116_116488

variable (Earl_initial Fred_initial Greg_initial : ℕ)
variable (Earl_to_Fred Fred_to_Greg Greg_to_Earl : ℕ)

def Earl_final := Earl_initial - Earl_to_Fred + Greg_to_Earl
def Fred_final := Fred_initial + Earl_to_Fred - Fred_to_Greg
def Greg_final := Greg_initial + Fred_to_Greg - Greg_to_Earl

theorem greg_and_earl_total :
  Earl_initial = 90 → Fred_initial = 48 → Greg_initial = 36 →
  Earl_to_Fred = 28 → Fred_to_Greg = 32 → Greg_to_Earl = 40 →
  Greg_final + Earl_final = 130 :=
by
  intros h1 h2 h3 h4 h5 h6
  dsimp [Earl_final, Fred_final, Greg_final]
  rw [h1, h2, h3, h4, h5, h6]
  exact sorry

end greg_and_earl_total_l116_116488


namespace opposite_of_neg_2023_l116_116842

def opposite (n : ℤ) : ℤ :=
  -n

theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116842


namespace total_doors_needed_correct_l116_116429

-- Define the conditions
def buildings : ℕ := 2
def floors_per_building : ℕ := 12
def apartments_per_floor : ℕ := 6
def doors_per_apartment : ℕ := 7

-- Define the total number of doors needed
def total_doors_needed : ℕ := buildings * floors_per_building * apartments_per_floor * doors_per_apartment

-- State the theorem to prove the total number of doors needed is 1008
theorem total_doors_needed_correct : total_doors_needed = 1008 := by
  sorry

end total_doors_needed_correct_l116_116429


namespace matrix_product_is_correct_l116_116511

-- Define the matrices A and B
def A : Matrix (Fin 3) (Fin 3) ℤ := ![
  ![3, 1, 1],
  ![2, 1, 2],
  ![1, 2, 3]
]

def B : Matrix (Fin 3) (Fin 3) ℤ := ![
  ![1, 1, -1],
  ![2, -1, 1],
  ![1, 0, 1]
]

-- Define the expected product matrix C
def C : Matrix (Fin 3) (Fin 3) ℤ := ![
  ![6, 2, -1],
  ![6, 1, 1],
  ![8, -1, 4]
]

-- The statement of the problem
theorem matrix_product_is_correct : (A * B) = C := by
  sorry -- Proof is omitted as per instructions

end matrix_product_is_correct_l116_116511


namespace range_of_f_l116_116375

-- Define the function f
def f (x : ℝ) : ℝ := (3 * x^2 - 2) / (x^2 + 1)

-- Define the condition
def condition (x : ℝ) : Prop := x ≥ 0

-- Define the range
def range (y : ℝ) : Prop := -2 ≤ y ∧ y < 3

-- Statement: Prove that the range of the function f for x >= 0 is [-2, 3)
theorem range_of_f (y : ℝ) : (∃ x : ℝ, condition x ∧ f x = y) ↔ range y := 
sorry

end range_of_f_l116_116375


namespace profit_share_of_B_l116_116453

theorem profit_share_of_B (P : ℝ) (A_share B_share C_share : ℝ) :
  let A_initial := 8000
  let B_initial := 10000
  let C_initial := 12000
  let total_capital := A_initial + B_initial + C_initial
  let investment_ratio_A := A_initial / total_capital
  let investment_ratio_B := B_initial / total_capital
  let investment_ratio_C := C_initial / total_capital
  let total_profit := 4200
  let diff_AC := 560
  A_share = (investment_ratio_A * total_profit) →
  B_share = (investment_ratio_B * total_profit) →
  C_share = (investment_ratio_C * total_profit) →
  C_share - A_share = diff_AC →
  B_share = 1400 :=
by
  intros
  sorry

end profit_share_of_B_l116_116453


namespace opposite_of_neg_2023_l116_116809

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by sorry

end opposite_of_neg_2023_l116_116809


namespace find_x3_l116_116368

variable (x1 x2 : ℝ) (hx1 : 0 < x1) (hx2 : x1 < x2)

def y1 : ℝ := Real.log x1
def y2 : ℝ := Real.log x2

def y_c : ℝ := (2 / 3) * y1 + (1 / 3) * y2

theorem find_x3 (hx1 : x1 = 2) (hx2 : x2 = 500) 
: ∃ x3 : ℝ, x3 = 10 * 2^(2/3) * 5^(1/3) := 
sorry

end find_x3_l116_116368


namespace opposite_of_neg_2023_l116_116831

/-- Define the opposite of a number -/
def opposite (x : ℤ) : ℤ := -x

/-- Prove the opposite of -2023 is 2023 -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116831


namespace find_t_l116_116578

def vector (α : Type) := (α × α)

def dot_product {α : Type} [CommRing α] (u v : vector α) : α :=
  u.1 * v.1 + u.2 * v.2

def magnitude {α : Type} [LinearOrderedField α] (u : vector α) : α :=
  real.sqrt (u.1 * u.1 + u.2 * u.2)

def angle_eq_condition (α : Type) [LinearOrderedField α] (a b c : vector α) (t : α) : Prop :=
  (magnitude a) ≠ 0 ∧ (magnitude b) ≠ 0 ∧ (magnitude c) ≠ 0 ∧
  (dot_product a c) / ((magnitude a) * (magnitude c)) = (dot_product b c) / ((magnitude b) * (magnitude c))

theorem find_t (α : Type) [LinearOrderedField α] (t : α) : 
  let a : vector α := (3, 4)
  let b : vector α := (1, 0)
  let c : vector α := (3 + t, 4)
  angle_eq_condition α a b c t → t = 5 :=
  sorry

end find_t_l116_116578


namespace number_of_roots_l116_116313

theorem number_of_roots (f : ℝ → ℝ) (g : ℝ → ℝ) : 
  (∀ x, f x = g x ↔ (x = 9 ∨ x = 0.5)) → 
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = g x₁ ∧ f x₂ = g x₂) :=
by
  let f := λ x, sqrt (9 - x)
  let g := λ x, 2 * x * sqrt (9 - x)
  intro h
  use 9
  use 0.5
  have hx1 : f 9 = g 9 := by calc
    f 9 = sqrt (9 - 9)       : by rfl
    ... = 0                    : by simp
    ... = 2 * 9 * sqrt (9 - 9) : by rw [sqrt_zero, zero_mul]
    ... = g 9                  : by rfl
  have hx2 : f 0.5 = g 0.5 := by calc
    f 0.5 = sqrt (9 - 0.5)        : by rfl
    ... = sqrt(17/2)              : sorry
    ... = 2 * 0.5 * sqrt (9 - 0.5) : by rw [mul_div_cancel' _ two_ne_zero.symm]
    ... = g 0.5                   : by rfl
  split
    exact (ne_of_lt (by norm_num)).symm
    exact ⟨hx1, hx2⟩
  sorry

end number_of_roots_l116_116313


namespace tan_add_pi_div_three_l116_116156

theorem tan_add_pi_div_three (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = -((6 + 5 * Real.sqrt 3) / 13) := 
by 
  sorry

end tan_add_pi_div_three_l116_116156


namespace opposite_of_neg_2023_l116_116861

-- Definition of the problem
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  -- The proof is to be provided
  sorry

end opposite_of_neg_2023_l116_116861


namespace part_I_part_II_l116_116536

noncomputable def ellipse_equation (a b : ℝ) (x y : ℝ) : Prop :=
x^2 / a^2 + y^2 / b^2 = 1

theorem part_I (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
  (eccentricity : ℝ := c / a) (h3 : eccentricity = Real.sqrt 2 / 2) (vertex : ℝ × ℝ := (0, 1)) (h4 : vertex = (0, b)) 
  : ellipse_equation (Real.sqrt 2) 1 (0:ℝ) 1 :=
sorry

theorem part_II (a b k : ℝ) (x y : ℝ) (h1 : a = Real.sqrt 2) (h2 : b = 1)
  (line_eq : ℝ → ℝ := fun x => k * x + 1) 
  (h3 : (1 + 2 * k^2) * x^2 + 4 * k * x = 0) 
  (distance_AB : ℝ := Real.sqrt 2 * 4 / 3) 
  (h4 : Real.sqrt (1 + k^2) * abs ((-4 * k) / (2 * k^2 + 1)) = distance_AB) 
  : (x, y) = (4/3, -1/3) ∨ (x, y) = (-4/3, -1/3) :=
sorry

end part_I_part_II_l116_116536


namespace prob_within_d_units_of_lattice_point_l116_116439

noncomputable def d (π : ℝ) : ℝ := 
  real.sqrt (3 / (4 * π))

theorem prob_within_d_units_of_lattice_point :
  ∀ (π : ℝ), (π > 0) → (∀ x y, 0 ≤ x ∧ x ≤ 2000 ∧ 0 ≤ y ∧ y ≤ 2000 → 
    prob_within_distance (x,y) d(π) = 3 / 4) :=
by
  sorry

end prob_within_d_units_of_lattice_point_l116_116439


namespace trains_crossing_time_l116_116374

def train_length := 150
def faster_train_speed_kmh := 90
def speed_ratio := 2

theorem trains_crossing_time :
  let v := faster_train_speed_kmh / speed_ratio in
  let relative_speed_kmh := v + faster_train_speed_kmh in
  let relative_speed_ms := (relative_speed_kmh : ℝ) * (1000 / 3600) in
  let total_distance := (2 * train_length : ℝ) in
  (total_distance / relative_speed_ms) = 8 :=
by
  sorry

end trains_crossing_time_l116_116374


namespace tan_add_pi_over_3_l116_116173

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = - (6 + 5 * Real.sqrt 3) / 13 :=
sorry

end tan_add_pi_over_3_l116_116173


namespace opposite_of_neg_2023_l116_116877

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116877


namespace find_swap_matrix_l116_116507

variable (a b c d : ℝ)

def swap_matrix_2x2 (M : Matrix (Fin 2) (Fin 2) ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![![(M 1 0), (M 1 1)],
     ![(M 0 0), (M 0 1)]]

theorem find_swap_matrix : 
  ∃ (N : Matrix (Fin 2) (Fin 2) ℝ), 
  N ⬝ !![![a, b], ![c, d]] = !![![c, d], ![a, b]] :=
  ⟨!![![0, 1], ![1, 0]], by simp [Matrix.mul]⟩

end find_swap_matrix_l116_116507


namespace opposite_of_negative_2023_l116_116791

theorem opposite_of_negative_2023 : (- -2023) = 2023 := 
by {
sorry
}

end opposite_of_negative_2023_l116_116791


namespace spherical_distance_A_B_l116_116472

noncomputable def spherical_distance (R : ℝ) (lat1 long1 lat2 long2 : ℝ) : ℝ :=
  R * Real.arccos (sin lat1 * sin lat2 + cos lat1 * cos lat2 * cos (long2 - long1))

theorem spherical_distance_A_B 
  (R : ℝ)
  (latA longA latB longB : ℝ)
  (h_latA : latA = 30 * (Real.pi / 180))
  (h_longA : longA = 20 * (Real.pi / 180))
  (h_latB : latB = 30 * (Real.pi / 180))
  (h_longB : longB = 80 * (Real.pi / 180)) :
  spherical_distance R latA longA latB longB = R * Real.arccos (5 / 8) := sorry

end spherical_distance_A_B_l116_116472


namespace remaining_distance_l116_116711

variable (D : ℝ) -- remaining distance in miles

-- Current speed
def current_speed := 8.0
-- Increased speed by 50%
def increased_speed := 12.0
-- Reduced speed by 4 mph
def reduced_speed := 4.0

-- Trip time relations
def current_time := D / current_speed
def increased_time := D / increased_speed
def reduced_time := D / reduced_speed

-- Time difference condition
axiom condition : reduced_time = increased_time + 16.0

theorem remaining_distance : D = 96.0 :=
by
  -- proof placeholder
  sorry

end remaining_distance_l116_116711


namespace opposite_of_neg_2023_l116_116826

/-- Define the opposite of a number -/
def opposite (x : ℤ) : ℤ := -x

/-- Prove the opposite of -2023 is 2023 -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116826


namespace probability_of_9_heads_in_12_flips_l116_116379

open BigOperators

-- We state the problem in terms of probability theory.
theorem probability_of_9_heads_in_12_flips :
  let p : ℚ := (nat.choose 12 9) / (2 ^ 12)
  p = 55 / 1024 :=
by
  sorry

end probability_of_9_heads_in_12_flips_l116_116379


namespace tangent_equation_l116_116655

noncomputable def f (a x : ℝ) : ℝ := x^3 + a * x^2 + (a - 2) * x

theorem tangent_equation (a : ℝ) (h_even : ∀ x : ℝ, (f a) (-x) = (f a) x) : 
  f' 0 = -2 := by
  sorry

end tangent_equation_l116_116655


namespace perimeter_of_ABFCDE_l116_116008

theorem perimeter_of_ABFCDE (side_length : ℝ) (h1 : side_length = 10) :
  let hypotenuse_length := real.sqrt (side_length ^ 2 + side_length ^ 2)
  in (3 * side_length + hypotenuse_length) = 30 + 10 * real.sqrt 2 := 
by
  sorry

end perimeter_of_ABFCDE_l116_116008


namespace opposite_of_neg_2023_l116_116979

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l116_116979


namespace opposite_of_neg_2023_l116_116865

-- Definition of the problem
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  -- The proof is to be provided
  sorry

end opposite_of_neg_2023_l116_116865


namespace distance_between_x_intercepts_correct_l116_116437

noncomputable def distance_between_x_intercepts : ℝ :=
  let p := (10 : ℝ, 15 : ℝ)
  let m1 := 4
  let m2 := -2
  let x1 := ((15 - 15 + 10 * 4) / 4 : ℝ)
  let x2 := ((15 - 15 - 10 * (-2)) / (-2) : ℝ)
  real.abs (x1 - x2)

theorem distance_between_x_intercepts_correct :
  distance_between_x_intercepts = 11.25 := by
  sorry

end distance_between_x_intercepts_correct_l116_116437


namespace tan_add_pi_over_3_l116_116172

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = - (6 + 5 * Real.sqrt 3) / 13 :=
sorry

end tan_add_pi_over_3_l116_116172


namespace circles_orthogonal_l116_116440

-- Define the point and geometric entities involved.
variable (A B C P O₁ O₂ : Point)
variable (circumcircle_ACP : Circle)
variable (circumcircle_BCP : Circle)

-- Assumptions as per the given conditions.
variable (hABC : ∠ABC = 90)
variable (hP : P ∈ segment AB)
variable (hA_C_P : circumcircle_ACP ⊆ Circle (A, C, P))
variable (hB_C_P : circumcircle_BCP ⊆ Circle (B, C, P))

-- The hypothesized statement to prove.
theorem circles_orthogonal :
  orthogonal circumcircle_ACP circumcircle_BCP :=
sorry

end circles_orthogonal_l116_116440


namespace opposite_of_neg_2023_l116_116920

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by 
  sorry

end opposite_of_neg_2023_l116_116920


namespace multiples_of_12_less_than_60_l116_116254

theorem multiples_of_12_less_than_60 :
  let a := (Finset.filter (λ n, 12 * n < 60) (Finset.range 5)).card
  let b := (Finset.filter (λ n, 12 * n < 60) (Finset.range 5)).card
  (a - b)^2 = 0 := by
sorry

end multiples_of_12_less_than_60_l116_116254


namespace find_value_of_x_l116_116262

theorem find_value_of_x (b : ℕ) (x : ℝ) (h_b_pos : b > 0) (h_x_pos : x > 0) 
  (h_r1 : r = 4 ^ (2 * b)) (h_r2 : r = 2 ^ b * x ^ b) : x = 8 :=
by
  -- Proof omitted for brevity
  sorry

end find_value_of_x_l116_116262


namespace total_weight_remaining_macaroons_l116_116628

theorem total_weight_remaining_macaroons (num_macaroons : ℕ) (weight_per_macaroon : ℕ) (num_bags : ℕ) (num_macaroons_per_bag : ℕ) (weight_steve_eats : ℕ) :
  num_macaroons = 12 → weight_per_macaroon = 5 → num_bags = 4 → num_macaroons_per_bag = num_macaroons / num_bags → weight_steve_eats = num_macaroons_per_bag * weight_per_macaroon → 
  let total_weight_initial := num_macaroons * weight_per_macaroon in
  let total_weight_remaining := total_weight_initial - weight_steve_eats in
  total_weight_remaining = 45 :=
by
  intros h1 h2 h3 h4 h5
  let total_weight_initial := num_macaroons * weight_per_macaroon
  let total_weight_remaining := total_weight_initial - weight_steve_eats
  sorry

end total_weight_remaining_macaroons_l116_116628


namespace opposite_of_neg_2023_l116_116838

def opposite (n : ℤ) : ℤ :=
  -n

theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116838


namespace area_ratio_triangle_CDE_ABE_l116_116616

-- Definitions according to conditions
variables (A B C D E : Point)
hyps : (A B C D E : Point) →
  AB_is_diameter : diameter A B →
  CD_perpendicular_AB : perpendicular CD AB →
  AC_intersects_BD_at_E : intersects AC BD E →
  angle_AEB_alpha : ∠ A E B = α

-- Proof goal: the ratio of the areas of triangles CDE and ABE
theorem area_ratio_triangle_CDE_ABE (A B C D E : Point) 
  (AB_is_diameter : diameter AB)
  (CD_perpendicular_AB : perpendicular CD AB)
  (AC_intersects_BD_at_E : intersects AC BD E)
  (angle_AEB_alpha : ∠ A E B = α) :
  (area (triangle C D E) / area (triangle A B E)) = (Real.sin α)^2 :=
sorry

end area_ratio_triangle_CDE_ABE_l116_116616


namespace equation_of_line_chord_l116_116593

theorem equation_of_line_chord (P : ℝ × ℝ) (A B : ℝ × ℝ) 
    (hP : P = (3, 1)) (h_circle : (∀ (x y : ℝ), ((x - 2)^2 + y^2 = 16) → ((A.1 - 2)^2 + A.2^2 = 16) ∧ ((B.1 - 2)^2 + B.2^2 = 16)))
    (h_midpoint : P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2) :
    (∃ (m b : ℝ), (λ x y, y = m * x + b) (A.1) (A.2) ∧ (λ x y, y = m * x + b) (B.1) (B.2))
    ∧ ∀ (x y : ℝ), y = -x + 4 ↔ y = x + 1 - 3 + 1 + 3 / 2 := 
begin
  sorry
end

end equation_of_line_chord_l116_116593


namespace opposite_of_neg_2023_l116_116864

-- Definition of the problem
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  -- The proof is to be provided
  sorry

end opposite_of_neg_2023_l116_116864


namespace cat_mouse_position_after_299_moves_l116_116208

-- Definitions based on conditions
def cat_position (move : Nat) : Nat :=
  let active_moves := move - (move / 100)
  active_moves % 4

def mouse_position (move : Nat) : Nat :=
  move % 8

-- Main theorem
theorem cat_mouse_position_after_299_moves :
  cat_position 299 = 0 ∧ mouse_position 299 = 3 :=
by
  sorry

end cat_mouse_position_after_299_moves_l116_116208


namespace percentage_j_of_k_theorem_l116_116594

noncomputable def percentage_j_of_k 
  (j k l m : ℝ) (x : ℝ) 
  (h1 : j * (x / 100) = k * (25 / 100))
  (h2 : k * (150 / 100) = l * (50 / 100))
  (h3 : l * (175 / 100) = m * (75 / 100))
  (h4 : m * (20 / 100) = j * (700 / 100)) : Prop :=
  x = 500

theorem percentage_j_of_k_theorem 
  (j k l m : ℝ) (x : ℝ)
  (h1 : j * (x / 100) = k * (25 / 100))
  (h2 : k * (150 / 100) = l * (50 / 100))
  (h3 : l * (175 / 100) = m * (75 / 100))
  (h4 : m * (20 / 100) = j * (700 / 100)) : percentage_j_of_k j k l m x h1 h2 h3 h4 :=
by 
  sorry

end percentage_j_of_k_theorem_l116_116594


namespace initial_investment_l116_116444

theorem initial_investment
  (P r : ℝ)
  (h1 : P + (P * r * 2) / 100 = 600)
  (h2 : P + (P * r * 7) / 100 = 850) :
  P = 500 :=
sorry

end initial_investment_l116_116444


namespace AM_AN_eq_CO_CD_l116_116011

noncomputable def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

-- Condition: CD is the diameter of the ellipse
def C_diameter (a b : ℝ) : ℝ × ℝ := (a, 0)
def D_diameter (a b : ℝ) : ℝ × ℝ := (-a, 0)

-- Condition: A line parallel to CD passes through the left vertex A (-a, 0)
def A_left (a : ℝ) : ℝ × ℝ := (-a, 0)

-- Points of intersection with the ellipse
structure Point (a b : ℝ) :=
  (x : ℝ)
  (y : ℝ)
  (on_ellipse : ellipse a b x y)

def M (a b : ℝ) : Point a b := { x := -a, y := 0, on_ellipse := sorry }

noncomputable def N (a b : ℝ) (θ : ℝ) : Point a b :=
  { x := a * cos θ,
    y := b * sin θ,
    on_ellipse := sorry }

-- The geometric product to be proved equivalently
theorem AM_AN_eq_CO_CD (a b : ℝ) (θ : ℝ) (A M N C D : Point a b) :
  let AM := a / cos θ,
      AN := (2 * a * b^2 * cos θ) / (b^2 * cos θ^2 + a^2 * sin θ^2),
      CO := sqrt ((a^2 * b^2) / (b^2 * cos θ^2 + a^2 * sin θ^2)),
      CD := 2 * CO in
  AM * AN = CO * CD :=
sorry

end AM_AN_eq_CO_CD_l116_116011


namespace no_square_with_odd_last_two_digits_l116_116025

def last_two_digits_odd (n : ℤ) : Prop :=
  (n % 10) % 2 = 1 ∧ ((n / 10) % 10) % 2 = 1

theorem no_square_with_odd_last_two_digits (n : ℤ) (k : ℤ) :
  (k^2 = n) → last_two_digits_odd n → False :=
by
  -- A placeholder for the proof
  sorry

end no_square_with_odd_last_two_digits_l116_116025


namespace day_of_week_N_minus_1_l116_116624

noncomputable def is_leap_year (n : ℕ) : Prop :=
  (n % 4 = 0 ∧ n % 100 ≠ 0) ∨ n % 400 = 0

theorem day_of_week_N_minus_1 :
  ∀ (N : ℕ),
  (250 % 7 = 5) ∧ (160 % 7 = 3) ∧
  (is_leap_year N → (365 - 250) % 7 = 2) ∧ 
  (¬is_leap_year N → (365 - 250) % 7 = 1) →
  (110 % 7 = 4) →
  ((((N : ℕ) % 7 + 4) % 7) = 4) →
  "Thursday" :=
begin
  intros,
  sorry
end

end day_of_week_N_minus_1_l116_116624


namespace opposite_of_neg_2023_l116_116884

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116884


namespace opposite_of_neg_2023_l116_116825

/-- Define the opposite of a number -/
def opposite (x : ℤ) : ℤ := -x

/-- Prove the opposite of -2023 is 2023 -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116825


namespace matrix_satisfies_conditions_l116_116499

theorem matrix_satisfies_conditions :
  ∃ (P : Matrix (Fin 2) (Fin 2) ℚ),
    (P ⬝ (Matrix.vecCons 1 (Matrix.vecCons 2 Matrix.vecEmpty)) = Matrix.vecCons 5 (Matrix.vecCons 11 Matrix.vecEmpty)) ∧
    (P ⬝ (Matrix.vecCons 2 (Matrix.vecCons (-1) Matrix.vecEmpty)) = Matrix.vecCons 0 (Matrix.vecCons 1 Matrix.vecEmpty)) ∧
    P = ![![1, 2],![13 / 5, 21 / 5]] :=
by {
  sorry
}

end matrix_satisfies_conditions_l116_116499


namespace Malou_score_third_quiz_l116_116671

-- Defining the conditions as Lean definitions
def score1 : ℕ := 91
def score2 : ℕ := 92
def average : ℕ := 91
def num_quizzes : ℕ := 3

-- Proving that score3 equals 90
theorem Malou_score_third_quiz :
  ∃ score3 : ℕ, (score1 + score2 + score3) / num_quizzes = average ∧ score3 = 90 :=
by
  use (90 : ℕ)
  sorry

end Malou_score_third_quiz_l116_116671


namespace opposite_neg_2023_l116_116768

theorem opposite_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_neg_2023_l116_116768


namespace find_circle_params_l116_116333

theorem find_circle_params 
(Center IsCenter : ℝ × ℝ)
(Center_eqn : Center = (-2, 3))
(Radius : ℝ)
(Radius_eqn : Radius = 4) :
  ∃ D E F : ℝ, 
    (x y : ℝ) -> x^2 + y^2 + D*x + E*y + F = 0 ∧
    (∀ x := IsCenter.fst, y := IsCenter.snd, x = -2 ∧ y = 3) ∧
    (∀ rad := Radius, rad = 4) 
    ∧ D = 4 ∧ E = -6 ∧ F = -3 :=
  by 
    use [4, -6, -3]
    sorry

end find_circle_params_l116_116333


namespace investment_amount_l116_116438

noncomputable theory

def share_nominal_value : ℝ := 100
def premium_rate : ℝ := 0.25
def dividend_rate : ℝ := 0.05
def total_dividend_received : ℝ := 576

def price_per_share : ℝ := share_nominal_value * (1 + premium_rate)
def dividend_per_share : ℝ := share_nominal_value * dividend_rate
def number_of_shares := total_dividend_received / dividend_per_share
def rounded_number_of_shares : ℕ := number_of_shares.toInt
def investment : ℝ := rounded_number_of_shares * price_per_share

theorem investment_amount : investment = 14375 := 
by 
  sorry

end investment_amount_l116_116438


namespace trigonometric_inequality_l116_116585

theorem trigonometric_inequality (x y : ℝ) (hx1 : 0 < x) (hx2 : x < real.pi / 2)
  (hy1 : 0 < y) (hy2 : y < real.pi / 2) (h : real.sin x = x * real.cos y) :
  x / 2 < y ∧ y < x :=
sorry

end trigonometric_inequality_l116_116585


namespace no_partition_with_max_min_l116_116194

noncomputable def is_partition (M N : set ℚ) : Prop :=
  M ∪ N = set.univ ∧ M ∩ N = ∅ ∧ ∀ m ∈ M, ∀ n ∈ N, m < n

noncomputable def has_max_element (M : set ℚ) : Prop :=
  ∃ max ∈ M, ∀ m ∈ M, m ≤ max

noncomputable def has_min_element (N : set ℚ) : Prop :=
  ∃ min ∈ N, ∀ n ∈ N, min ≤ n

theorem no_partition_with_max_min (M N : set ℚ) (h : is_partition M N) :
  ¬ (has_max_element M ∧ has_min_element N) :=
sorry

end no_partition_with_max_min_l116_116194


namespace opposite_of_negative_2023_l116_116792

theorem opposite_of_negative_2023 : (- -2023) = 2023 := 
by {
sorry
}

end opposite_of_negative_2023_l116_116792


namespace sectors_area_ratio_l116_116361

variable (P Q R O : Point)
variable [h : Circle O]

theorem sectors_area_ratio (h_arc_ratio : arc_length P Q / arc_length Q R = 1/2) 
  (h_arc_ratio2 : arc_length Q R / arc_length R P = 2/3) : 
  area (sector O P Q) / area (sector O Q R) = 1/2 ∧
  area (sector O Q R) / area (sector O R P) = 2/3 := 
by sorry

end sectors_area_ratio_l116_116361


namespace arg_z_range_l116_116526

open Complex

theorem arg_z_range (z : ℂ) (h : abs (arg ((z + 1) / (z + 2))) = π / 6) :
  (arg z) ∈ Set.Ioo (5 * π / 6 - Real.arcsin (Real.sqrt 3 / 3)) π ∪ Set.Ioo π (7 * π / 6 + Real.arcsin (Real.sqrt 3 / 3)) :=
sorry

end arg_z_range_l116_116526


namespace triangle_is_isosceles_l116_116201

theorem triangle_is_isosceles
  {A B C : Type}
  (a b : ℝ)
  (cos : Typ → ℝ → ℝ)
  (triangle : Type)
  [IsTriangle triangle A B C]
  (h : a * cos B = b * cos A) : is_isosceles_triangle triangle A B C :=
sorry

end triangle_is_isosceles_l116_116201


namespace min_value_of_M_l116_116076

theorem min_value_of_M :
  ∀ x y : ℝ, 
    (∃ x y : ℝ, (sqrt (2 * x^2 - 6 * x + 5) + sqrt (y^2 - 4 * y + 5) + sqrt (2 * x^2 - 2 * x * y + y^2) = sqrt 10)) ∧ 
    (∀ x y : ℝ, sqrt (2 * x^2 - 6 * x + 5) + sqrt (y^2 - 4 * y + 5) + sqrt (2 * x^2 - 2 * x * y + y^2) >= sqrt 10) :=
sorry

end min_value_of_M_l116_116076


namespace intersection_complement_M_N_is_0_to_1_l116_116126

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | log (1/2) (x - 1) > -1}
def N : Set ℝ := {x | 1 < 2^x ∧ 2^x < 4 }
def complement_M := { x : ℝ | x ≤ 1 ∨ x ≥ 3 }

theorem intersection_complement_M_N_is_0_to_1 :
  (complement_M ∩ N) = { x : ℝ | 0 < x ∧ x ≤ 1 } :=
by
  sorry

end intersection_complement_M_N_is_0_to_1_l116_116126


namespace pascal_remaining_miles_l116_116703

section PascalCyclingTrip

variable (D T : ℝ)
variable (V : ℝ := 8)
variable (V_reduced V_increased : ℝ)
variable (time_diff : ℝ := 16)

-- Definitions of reduced and increased speeds
def V_reduced := V - 4
def V_increased := V + 0.5 * V

-- Conditions for the distance
def distance_current := D = V * T
def distance_increased := D = V_increased * (T - time_diff)
def distance_reduced := D = V_reduced * (T + time_diff)

theorem pascal_remaining_miles (h1 : distance_current)
  (h2 : distance_increased)
  (h3 : distance_reduced) : D = 256 := 
sorry

end PascalCyclingTrip

end pascal_remaining_miles_l116_116703


namespace interval_of_decrease_l116_116107

def g (x : ℝ) : ℝ := log x / log (1/2)

noncomputable def f (x : ℝ) : ℝ := (1/2 : ℝ) ^ x

theorem interval_of_decrease :
  (∀ x y : ℝ, x < y → f y < f x) ↔ ∀ x : ℝ, -∞ < x := sorry

end interval_of_decrease_l116_116107


namespace triangle_DEC_angles_l116_116096
open EuclideanGeometry

noncomputable def centroid (P M B : Point) : Point := sorry
noncomputable def midpoint (A P : Point) : Point := sorry

theorem triangle_DEC_angles (A B C M P D E : Point) 
(h1 : equilateral_triangle A B C) 
(h2 : parallel AC MP) 
(h3 : intersects MP AB = M) 
(h4 : intersects MP BC = P) 
(h5 : D = centroid P M B) 
(h6 : E = midpoint A P) :
angles_of_triangle D E C = ? := 
sorry

end triangle_DEC_angles_l116_116096


namespace sequence_increasing_l116_116522

theorem sequence_increasing (a : ℕ → ℝ) 
  (h : ∀ n, a (n + 1) = a n + 3) : ∀ n, a (n + 1) > a n := 
by 
  sorry

end sequence_increasing_l116_116522


namespace opposite_of_neg_2023_l116_116801

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by sorry

end opposite_of_neg_2023_l116_116801


namespace minimum_value_proof_l116_116653

noncomputable def minimum_value_condition (a b : ℝ) : Prop :=
a > 0 ∧ b > 0 ∧ a + b = 12

theorem minimum_value_proof : ∀ (a b : ℝ), minimum_value_condition a b → (1 / a + 1 / b ≥ 1 / 3) := 
by
  intros a b h
  sorry

end minimum_value_proof_l116_116653


namespace binom_7_4_plus_5_l116_116029

theorem binom_7_4_plus_5 : ((Nat.choose 7 4) + 5) = 40 := by
  sorry

end binom_7_4_plus_5_l116_116029


namespace opposite_of_neg2023_l116_116996

theorem opposite_of_neg2023 : ∀ (x : Int), -2023 + x = 0 → x = 2023 :=
by
  intros x h
  sorry

end opposite_of_neg2023_l116_116996


namespace transformed_sin_is_cos_l116_116318

theorem transformed_sin_is_cos :
  ∀ x : ℝ, sin ((x - π / 2) / 2) = cos x :=
by
  sorry

end transformed_sin_is_cos_l116_116318


namespace time_to_pass_platform_is_60_seconds_l116_116448

-- Define the given conditions
def length_of_train : ℝ := 720
def speed_of_train_km_hr : ℝ := 90
def length_of_platform : ℝ := 780

-- Conversion factor from km/hr to m/s
def km_per_hr_to_m_per_s (km_per_hr : ℝ) : ℝ :=
  km_per_hr * 1000 / 3600

-- Speed of the train in m/s
def speed_of_train_m_s : ℝ :=
  km_per_hr_to_m_per_s speed_of_train_km_hr

-- Total distance the train needs to cover to pass the platform
def total_distance : ℝ :=
  length_of_train + length_of_platform

-- Time taken to pass the platform
def time_to_pass_platform : ℝ :=
  total_distance / speed_of_train_m_s

-- Theorem to prove the time taken to pass the platform
theorem time_to_pass_platform_is_60_seconds :
  time_to_pass_platform = 60 := by
  sorry

end time_to_pass_platform_is_60_seconds_l116_116448


namespace Smiths_Backery_Pies_l116_116300

theorem Smiths_Backery_Pies : 
  ∀ (p : ℕ), (∃ (q : ℕ), q = 16 ∧ p = 4 * q + 6) → p = 70 :=
by
  intros p h
  cases' h with q hq
  cases' hq with hq1 hq2
  rw [hq1] at hq2
  have h_eq : p = 4 * 16 + 6 := hq2
  norm_num at h_eq
  exact h_eq

end Smiths_Backery_Pies_l116_116300


namespace opposite_of_neg_2023_l116_116818

/-- Define the opposite of a number -/
def opposite (x : ℤ) : ℤ := -x

/-- Prove the opposite of -2023 is 2023 -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116818


namespace opposite_of_neg_2023_l116_116807

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by sorry

end opposite_of_neg_2023_l116_116807


namespace flight_duration_l116_116460

theorem flight_duration (h m : ℕ) (h_pos : 0 < m ∧ m < 60)
  (chi_dep_h : ℕ := 9) (chi_dep_m : ℕ := 17) (nyc_arr_h : ℤ := 14) (nyc_arr_m : ℕ := 53)
  (time_diff : ℤ := 1) :
  h = 5 ∧ m = 36 → (h + m = 41) :=
by
  intro h_m_eq
  cases h_m_eq with h_eq m_eq
  rw [h_eq, m_eq]
  exact rfl

end flight_duration_l116_116460


namespace minimize_distance_l116_116099

structure Point := (x : ℝ) (y : ℝ)

def distance (P Q : Point) : ℝ :=
  real.sqrt ((Q.x - P.x) ^ 2 + (Q.y - P.y) ^ 2)

noncomputable def total_distance_PR_RQ (P Q R : Point) : ℝ :=
  distance P R + distance R Q

theorem minimize_distance :
  ∀ P Q : Point, P.x = -2 → P.y = -3 → Q.x = 5 → Q.y = 3 → ∃ m : ℝ, (R : Point) (h : R.x = 1 ∧ R.y = m) → total_distance_PR_RQ P Q R = total_distance_PR_RQ P Q {x := 1, y := -3 / 7}  :=
by                      -- ∀x ∃y Definitions and the result follows the same problem structure but the order and conditions are being followed for minimal computation error translating into Lean4 and hence correct theorem statement.
  sorry

end minimize_distance_l116_116099


namespace expression_value_l116_116023

theorem expression_value : 200 * (200 - 8) - (200 * 200 + 8) = -1608 := 
by 
  -- We will put the proof here
  sorry

end expression_value_l116_116023


namespace opposite_of_neg_2023_l116_116808

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by sorry

end opposite_of_neg_2023_l116_116808


namespace opposite_of_neg_2023_l116_116912

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by 
  sorry

end opposite_of_neg_2023_l116_116912


namespace reachable_array_has_nonneg_sums_l116_116533

theorem reachable_array_has_nonneg_sums (m n : ℕ) (A : Array (Array ℝ)) :
  (∀ i < m, ∑ j in 0..n, A[i][j] = 0) →
  ∃ B : Array (Array ℝ), 
  (∑ i in 0..m, ∑ j in 0..n, B[i][j] = ∑ i in 0..m, ∑ j in 0..n, A[i][j]) ∧
  (∀ i < m, 0 ≤ ∑ j in 0..n, B[i][j]) ∧
  (∀ j < n, 0 ≤ ∑ i in 0..m, B[i][j]) :=
sorry

end reachable_array_has_nonneg_sums_l116_116533


namespace probability_nine_heads_l116_116382

theorem probability_nine_heads:
  (∀ (n k : ℕ), k ≤ n → (finset.card (finset.filter (λ (s : finset ℕ), s.card = k) (finset.powerset (finset.range n))) = nat.choose n k)) →
  ∃ p : ℚ, p = 55 / 1024 ∧ 
    calc 
      220 / 4096 = p :
      sorry :=
begin
  let n := 12,
  let k := 9,
  have h1: 2^n = 4096 := by norm_num,
  have h2 : nat.choose n k = 220 := by norm_num,
  have h3 : 220 / 4096 = 55 / 1024 := by norm_num,
  existsi (55 / 1024 : ℚ),
  split,
  { refl, },
  { exact h3 }
end

end probability_nine_heads_l116_116382


namespace abs_nested_expr_l116_116183

theorem abs_nested_expr (x : ℝ) (h : x < -4) : |1 - |1 + x + 2|| = -4 - x :=
sorry

end abs_nested_expr_l116_116183


namespace opposite_of_neg2023_l116_116989

theorem opposite_of_neg2023 : ∀ (x : Int), -2023 + x = 0 → x = 2023 :=
by
  intros x h
  sorry

end opposite_of_neg2023_l116_116989


namespace snow_white_problem_l116_116309

theorem snow_white_problem (x y : ℕ) 
  (h1 : ∀ i ∈ {1, 2, 3, 4, 5, 6}, i ∈ finset.univ) 
  (h2 : y ∈ {1, 2, 3, 4, 5, 6}) 
  (h3 : 7 * x - 21 - y = 46) : x = 10 ∧ y = 3 := 
by 
  sorry

end snow_white_problem_l116_116309


namespace opposite_of_neg_2023_l116_116853

-- Definition of the problem
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  -- The proof is to be provided
  sorry

end opposite_of_neg_2023_l116_116853


namespace ways_to_express_2016_as_sum_l116_116581

theorem ways_to_express_2016_as_sum :
  {n : ℕ // ∃ x y : ℕ, 2016 = 2 * x + 3 * y} = 337 :=
sorry

end ways_to_express_2016_as_sum_l116_116581


namespace find_x_l116_116553

-- Definitions from conditions
def passes_through_P (x : ℝ) : Prop := ∃ α : ℝ, ∃ P : ℝ × ℝ, P = (x, -3) ∧ float.cos α = -sqrt 3 / 2

-- Main theorem statement
theorem find_x (x : ℝ) 
  (h1 : passes_through_P x) 
  (h2 : float.cos x = -sqrt 3 / 2) 
  (h3 : float.cos x = x / sqrt (x^2 + 9)) : 
  x = -3 * sqrt 3 :=
sorry

end find_x_l116_116553


namespace opposite_of_neg_2023_l116_116879

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116879


namespace max_circle_sum_l116_116515

noncomputable def circle_sum (a : ℕ → ℝ) (n : ℕ) (h : n ≥ 3) : ℝ :=
  ∑ k in finset.range (n - 1).succ, nat.choose (n - 2) (k.div2 - 1) * a (k + 1)

theorem max_circle_sum {a : ℕ → ℝ} {n : ℕ} (h : n ≥ 3) (ha : ∀ i, a i ≤ a (i + 1)) :
  ∃ S_max, S_max = circle_sum a n h :=
sorry

end max_circle_sum_l116_116515


namespace no_p_cliques_implies_low_degree_l116_116261

noncomputable def graph := Type

theorem no_p_cliques_implies_low_degree (p n : Nat)
  (hp : 1 ≤ p)
  (G : graph)
  (Hn : G.vertices = n)
  (Hp : ∀ S : Finset G.vertices, S.card = p → ¬(S.allAdjacent)) :
  ∃ v : G.vertices, G.degree v ≤ (1 - (1 / (p - 1))) * n := 
sorry

end no_p_cliques_implies_low_degree_l116_116261


namespace opposite_of_neg_2023_l116_116820

/-- Define the opposite of a number -/
def opposite (x : ℤ) : ℤ := -x

/-- Prove the opposite of -2023 is 2023 -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116820


namespace opposite_of_neg_2023_l116_116837

def opposite (n : ℤ) : ℤ :=
  -n

theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116837


namespace opposite_of_neg_2023_l116_116870

-- Definition of the problem
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  -- The proof is to be provided
  sorry

end opposite_of_neg_2023_l116_116870


namespace Pascal_remaining_distance_l116_116699

theorem Pascal_remaining_distance (D T : ℕ) :
  let current_speed := 8
  let reduced_speed := current_speed - 4
  let increased_speed := current_speed + current_speed / 2
  (D = current_speed * T) →
  (D = reduced_speed * (T + 16)) →
  (D = increased_speed * (T - 16)) →
  D = 256 :=
by
  intros
  sorry

end Pascal_remaining_distance_l116_116699


namespace min_value_frac_inv_l116_116646

noncomputable def min_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 12) : ℝ :=
  1 / a + 1 / b

theorem min_value_frac_inv (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 12) :
  min_value a b h1 h2 h3 = 1 / 3 :=
sorry

end min_value_frac_inv_l116_116646


namespace three_a_plus_two_b_l116_116573

-- Define the vectors a and b
def a : ℝ × ℝ := (-1, 2)
def b (y : ℝ) : ℝ × ℝ := (2, y)

-- Define the condition that vectors are parallel
def vectors_parallel (a b : ℝ × ℝ) : Prop :=
  ∃k : ℝ, a = (k * (fst b), k * (snd b))

theorem three_a_plus_two_b (y : ℝ) (h : vectors_parallel (-1, 2) (2, y)) :
  3 • a + 2 • b y = (1, -2) :=
sorry

end three_a_plus_two_b_l116_116573


namespace union_complement_eq_set_l116_116269

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 3, 4}
def N : Set ℕ := {3, 5, 6}
def comp_U_N : Set ℕ := U \ N  -- complement of N with respect to U

theorem union_complement_eq_set :
  M ∪ comp_U_N = {1, 2, 3, 4} :=
by
  simp [U, M, N, comp_U_N]
  sorry

end union_complement_eq_set_l116_116269


namespace opposite_of_neg_2023_l116_116919

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by 
  sorry

end opposite_of_neg_2023_l116_116919


namespace probability_equals_fraction_distribution_and_expectation_correct_l116_116075

noncomputable def probability_same_fee : ℚ :=
  let p := (1 / 4) * (1 / 2) + (1 / 2) * (1 / 4) + (1 / 4) * (1 / 4) in
  p

theorem probability_equals_fraction : probability_same_fee = 5 / 16 :=
  sorry

noncomputable def distribution_and_expectation_xi : Σ (Pxi : ℚ → ℚ), ℚ :=
  let P := λ x, if x = 2 then 5 / 16 else if x = 4 then 5 / 16 else if x = 6 then 3 / 16 else if x = 8 then 1 / 16 else 1 / 8 in
  let E := 5 / 16 * 2 + 5 / 16 * 4 + 3 / 16 * 6 + 1 / 16 * 8 in
  ⟨P, E⟩

theorem distribution_and_expectation_correct :
  distribution_and_expectation_xi = ⟨λ x, if x = 2 then 5 / 16 else if x = 4 then 5 / 16 else if x = 6 then 3 / 16
      else if x = 8 then 1 / 16 else 1 / 8, 7 / 2⟩ :=
  sorry

end probability_equals_fraction_distribution_and_expectation_correct_l116_116075


namespace number_of_people_per_table_l116_116425

theorem number_of_people_per_table (invited : ℕ) (didnt_show_up : ℕ) (tables_needed : ℕ) 
  (total_people_present : invited - didnt_show_up = 10) (total_tables : tables_needed = 5) :
  (10 / 5 = 2) :=
begin
  sorry
end

end number_of_people_per_table_l116_116425


namespace total_time_naomi_30webs_l116_116462

-- Define the constants based on the given conditions
def time_katherine : ℕ := 20
def factor_naomi : ℚ := 5/4
def websites : ℕ := 30

-- Define the time taken by Naomi to build one website based on the conditions
def time_naomi (time_katherine : ℕ) (factor_naomi : ℚ) : ℚ :=
  factor_naomi * time_katherine

-- Define the total time Naomi took to build all websites
def total_time_naomi (time_naomi : ℚ) (websites : ℕ) : ℚ :=
  time_naomi * websites

-- Statement: Proving that the total number of hours Naomi took to create 30 websites is 750
theorem total_time_naomi_30webs : 
  total_time_naomi (time_naomi time_katherine factor_naomi) websites = 750 := 
sorry

end total_time_naomi_30webs_l116_116462


namespace prime_number_five_greater_than_perfect_square_l116_116394

theorem prime_number_five_greater_than_perfect_square 
(p x : ℤ) (h1 : p - 5 = x^2) (h2 : p + 9 = (x + 1)^2) : 
  p = 41 :=
sorry

end prime_number_five_greater_than_perfect_square_l116_116394


namespace probability_of_one_in_pascals_triangle_first_20_rows_l116_116459

noncomputable def number_of_elements (n : ℕ) : ℕ := n * (n + 1) / 2
noncomputable def ones_in_rows (n : ℕ) : ℕ := if n = 0 then 1 else 2 * (n - 1) + 1

theorem probability_of_one_in_pascals_triangle_first_20_rows : 
  let total_elements := number_of_elements 20
  let total_ones := ones_in_rows 20
  in total_ones / total_elements = 13 / 70 :=
by
  sorry

end probability_of_one_in_pascals_triangle_first_20_rows_l116_116459


namespace marble_selection_l116_116632

theorem marble_selection : (∃ num_ways : ℕ, num_ways = 990 ∧ (∃ S : finset ℕ, S.card = 5 ∧ 
  (∃ subset_special : finset ℕ, subset_special.card = 2 ∧ subset_special ⊆ {0, 1, 2, 3} ∧ 
  ∃ subset_rest : finset ℕ, subset_rest.card = 3 ∧ subset_rest ⊆ (finset.range 15 \ {0, 1, 2, 3}) ∧ 
  subset_special ∪ subset_rest = S))) :=
sorry

end marble_selection_l116_116632


namespace arc_length_correct_l116_116322

noncomputable def radius : ℝ :=
  5

noncomputable def area_of_sector : ℝ :=
  8.75

noncomputable def arc_length (θ : ℝ) (r : ℝ) : ℝ :=
  (θ / 360) * 2 * Real.pi * r

theorem arc_length_correct :
  ∃ θ, arc_length θ radius = 3.5 ∧ (θ / 360) * Real.pi * radius^2 = area_of_sector :=
by
  sorry

end arc_length_correct_l116_116322


namespace complex_number_count_l116_116137

open Complex

theorem complex_number_count :
  {z : ℂ | abs z < 50 ∧ exp(z^2) = (z^2 - 1) / (z^2 + 1)}.to_finset.card = 50 :=
by
  sorry

end complex_number_count_l116_116137


namespace permutation_by_transpositions_l116_116290

-- Formalizing the conditions in Lean
section permutations
  variable {n : ℕ}

  -- Define permutations
  def is_permutation (σ : Fin n → Fin n) : Prop :=
    ∃ σ_inv : Fin n → Fin n, 
      (∀ i, σ (σ_inv i) = i) ∧ 
      (∀ i, σ_inv (σ i) = i)

  -- Define transposition
  def transposition (σ : Fin n → Fin n) (i j : Fin n) : Fin n → Fin n :=
    fun x => if x = i then j else if x = j then i else σ x

  -- Main theorem stating that any permutation can be obtained through a series of transpositions
  theorem permutation_by_transpositions (σ : Fin n → Fin n) (h : is_permutation σ) :
    ∃ τ : ℕ → (Fin n → Fin n),
      (∀ i, is_permutation (τ i)) ∧
      (∀ m, ∃ k, τ m = transposition (τ (m - 1)) (⟨ k, sorry ⟩) (σ (⟨ k, sorry⟩))) ∧
      (∃ m, τ m = σ) :=
  sorry
end permutations

end permutation_by_transpositions_l116_116290


namespace gear_ratio_l116_116360

variable (x y z : ℕ) (α β γ : ℝ)

-- Definitions based on problem conditions: product of teeth and angular speed remains constant
axiom gear_A : α * x = α'
axiom gear_B : β * y = β'
axiom gear_C : γ * z = γ'
axiom linear_velocity : α' = β' = γ'

theorem gear_ratio ( h1 : x > 0 ) ( h2 : y > 0 ) ( h3 : z > 0 ) :
  α : β : γ = yz : xz : xy :=
sorry

end gear_ratio_l116_116360


namespace min_students_with_same_score_l116_116206

noncomputable def highest_score : ℕ := 83
noncomputable def lowest_score : ℕ := 30
noncomputable def total_students : ℕ := 8000
noncomputable def range_scores : ℕ := (highest_score - lowest_score + 1)

theorem min_students_with_same_score :
  ∃ k : ℕ, k = Nat.ceil (total_students / range_scores) ∧ k = 149 :=
by
  sorry

end min_students_with_same_score_l116_116206


namespace opposite_of_neg_2023_l116_116904

-- Let x be the opposite of -2023
def is_opposite (x : ℤ) (n : ℤ) : Prop := n + x = 0

theorem opposite_of_neg_2023 : ∃ x : ℤ, is_opposite x (-2023) ∧ x = 2023 :=
by
  exists 2023
  unfold is_opposite
  sorry

end opposite_of_neg_2023_l116_116904


namespace boy_lap_time_l116_116136

noncomputable def muddy_speed : ℝ := 5 * 1000 / 3600
noncomputable def sandy_speed : ℝ := 7 * 1000 / 3600
noncomputable def uphill_speed : ℝ := 4 * 1000 / 3600

noncomputable def muddy_distance : ℝ := 10
noncomputable def sandy_distance : ℝ := 15
noncomputable def uphill_distance : ℝ := 10

noncomputable def time_for_muddy : ℝ := muddy_distance / muddy_speed
noncomputable def time_for_sandy : ℝ := sandy_distance / sandy_speed
noncomputable def time_for_uphill : ℝ := uphill_distance / uphill_speed

noncomputable def total_time_for_one_side : ℝ := time_for_muddy + time_for_sandy + time_for_uphill
noncomputable def total_time_for_lap : ℝ := 4 * total_time_for_one_side

theorem boy_lap_time : total_time_for_lap = 95.656 := by
  sorry

end boy_lap_time_l116_116136


namespace sum_fractions_equals_5_l116_116258

def f (x : ℝ) : ℝ := 4^x / (4^x + 2)

theorem sum_fractions_equals_5 :
    f (1 / 11) + f (2 / 11) + f (3 / 11) + f (4 / 11) + f (5 / 11) + 
    f (6 / 11) + f (7 / 11) + f (8 / 11) + f (9 / 11) + f (10 / 11) = 5 := 
    sorry

end sum_fractions_equals_5_l116_116258


namespace exists_number_divisible_by_5_pow_1000_without_zeros_l116_116292

theorem exists_number_divisible_by_5_pow_1000_without_zeros :
  ∃ N : ℕ, (5 ^ 1000 ∣ N) ∧ (∀ k, k < digit_length N → ¬ (digit N k = 0)) :=
by sorry

end exists_number_divisible_by_5_pow_1000_without_zeros_l116_116292


namespace opposite_of_negative_2023_l116_116780

theorem opposite_of_negative_2023 : (- -2023) = 2023 := 
by {
sorry
}

end opposite_of_negative_2023_l116_116780


namespace remaining_distance_proof_l116_116710

-- Define the conditions
def pascal_current_speed : ℝ := 8
def pascal_reduced_speed : ℝ := pascal_current_speed - 4
def pascal_increased_speed : ℝ := pascal_current_speed * 1.5

-- Define the remaining distance in terms of the current speed and time taken
noncomputable def remaining_distance (T : ℝ) : ℝ := pascal_current_speed * T

-- Define the times with the increased and reduced speeds
noncomputable def time_with_increased_speed (T : ℝ) : ℝ := T - 16
noncomputable def time_with_reduced_speed (T : ℝ) : ℝ := T + 16

-- Define the distances using increased and reduced speeds
noncomputable def distance_increased_speed (T : ℝ) : ℝ := pascal_increased_speed * (time_with_increased_speed T)
noncomputable def distance_reduced_speed (T : ℝ) : ℝ := pascal_reduced_speed * (time_with_reduced_speed T)

-- Main theorem stating that the remaining distance is 256 miles
theorem remaining_distance_proof (T : ℝ) (ht_eq: pascal_current_speed * T = 256) : 
  remaining_distance T = 256 := by
  sorry

end remaining_distance_proof_l116_116710


namespace circle_tangent_line_l116_116351

theorem circle_tangent_line
  (C : ℝ → ℝ → ℝ → Prop)
  (l : ℝ → ℝ → Prop)
  (P : ℝ × ℝ)
  (center_on_x_axis : ∃ a : ℝ, C a 0 0)
  (tangent_at_P : ∀ x y, l x y → P = (x, y) → ∃ r : ℝ, C (fst P) (snd P) r ∧ y = 2 * x + 1) :
  ∃ r : ℝ, C 2 0 r :=
by
  -- The proof steps will go here
  sorry

end circle_tangent_line_l116_116351


namespace general_term_of_sequence_l116_116103

theorem general_term_of_sequence (x : ℕ → ℚ) :
  x 1 = 4 → 
  (∀ n ≥ 2, x n = (5 * n + 2) / (5 * n - 3) * x (n - 1) + 7 * (5 * n + 2)) →
  ∀ n, x n = (49 * n - 45) * (5 * n + 2) / 7 :=
begin
  intros h₁ h₂ n,
  sorry
end

end general_term_of_sequence_l116_116103


namespace arg_z_range_l116_116525

open Complex

theorem arg_z_range (z : ℂ) (h : abs (arg ((z + 1) / (z + 2))) = π / 6) :
  (arg z) ∈ Set.Ioo (5 * π / 6 - Real.arcsin (Real.sqrt 3 / 3)) π ∪ Set.Ioo π (7 * π / 6 + Real.arcsin (Real.sqrt 3 / 3)) :=
sorry

end arg_z_range_l116_116525


namespace opposite_of_neg_2023_l116_116947

theorem opposite_of_neg_2023 : ∃ x : ℤ, -2023 + x = 0 ∧ x = 2023 := 
by 
  use 2023
  constructor
  · exact rfl
  · exact rfl

end opposite_of_neg_2023_l116_116947


namespace king_and_queen_ages_l116_116756

variable (K Q : ℕ)

theorem king_and_queen_ages (h1 : K = 2 * (Q - (K - Q)))
                            (h2 : K + (K + (K - Q)) = 63) :
                            K = 28 ∧ Q = 21 := by
  sorry

end king_and_queen_ages_l116_116756


namespace solution_quadrant_I_l116_116257

theorem solution_quadrant_I (c x y : ℝ) :
  (x - y = 2 ∧ c * x + y = 3 ∧ x > 0 ∧ y > 0) ↔ (-1 < c ∧ c < 3/2) := by
  sorry

end solution_quadrant_I_l116_116257


namespace opposite_of_neg_2023_l116_116886

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116886


namespace opposite_of_neg_2023_l116_116949

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l116_116949


namespace inserted_number_sq_property_l116_116350

noncomputable def inserted_number (n : ℕ) : ℕ :=
  (5 * 10^n - 1) * 10^(n+1) + 1

theorem inserted_number_sq_property (n : ℕ) : (inserted_number n)^2 = (10^(n+1) - 1)^2 :=
by sorry

end inserted_number_sq_property_l116_116350


namespace wise_men_success_l116_116689

-- Define the choices available
inductive Choice
| one : Choice
| two : Choice
| three : Choice

-- Strategy function type: takes previous two choices and returns a new choice
def Strategy : (Choice × Choice) → Choice

-- Simulation of wise men choices
def simulate_choices (strategies : List Strategy) : List Choice :=
  let initial_choices : List Choice := [Choice.three, Choice.two]
  strategies.foldl (fun choices strategy =>
    match List.takeLast 2 choices with
    | [c1, c2] => choices ++ [strategy (c1, c2)]
    | _ => choices
  ) initial_choices

-- Sum of chosen numbers
def sum_choices : List Choice → Nat
| [] => 0
| Choice.one :: xs => 1 + sum_choices xs
| Choice.two :: xs => 2 + sum_choices xs
| Choice.three :: xs => 3 + sum_choices xs

-- Main theorem: Prove that there exists a strategy to guarantee success (sum ≠ 200)
theorem wise_men_success : ∃ strategies : List Strategy, sum_choices (simulate_choices strategies) ≠ 200 :=
sorry

end wise_men_success_l116_116689


namespace value_of_expression_at_x_4_l116_116392

theorem value_of_expression_at_x_4 :
  ∀ (x : ℝ), x = 4 → (x^2 - 2 * x - 8) / (x - 4) = 6 :=
by
  intro x hx
  sorry

end value_of_expression_at_x_4_l116_116392


namespace minimum_value_proof_l116_116652

noncomputable def minimum_value_condition (a b : ℝ) : Prop :=
a > 0 ∧ b > 0 ∧ a + b = 12

theorem minimum_value_proof : ∀ (a b : ℝ), minimum_value_condition a b → (1 / a + 1 / b ≥ 1 / 3) := 
by
  intros a b h
  sorry

end minimum_value_proof_l116_116652


namespace number_of_incorrect_interpretations_l116_116077

-- Define the interpretations as conditions
def opposite_of_neg8 : ℤ := -( -8 )
def product_neg1_neg8 : ℤ := -1 * -8
def abs_value_neg8 : ℤ := | -8 |
def result_of_neg_neg8 : ℤ := 8

-- Define the proof problem
theorem number_of_incorrect_interpretations : 
  opposite_of_neg8 = product_neg1_neg8 ∧ 
  product_neg1_neg8 = abs_value_neg8 ∧ 
  abs_value_neg8 = result_of_neg_neg8 → 
  0 = 0 :=
by
  intros
  sorry

end number_of_incorrect_interpretations_l116_116077


namespace count_three_element_subsets_with_property_P_l116_116660

namespace ThreeElementSubsets

-- Define the set S
def S : Finset ℕ := Finset.range 101 \ Finset.singleton 0

-- Define a subset has property P if a + b = 3c for a subset {a, b, c}
def has_property_P (A : Finset ℕ) : Prop :=
  ∃ a b c, A = {a, b, c} ∧ a + b = 3 * c

-- Define the number of three-element subsets with property P
def count_property_P : ℕ :=
  (S.subsetsOfCard 3).count has_property_P

-- Prove that the count of such subsets is 1600
theorem count_three_element_subsets_with_property_P :
  count_property_P = 1600 :=
sorry

end ThreeElementSubsets

end count_three_element_subsets_with_property_P_l116_116660


namespace solution_set_inequality_l116_116513

theorem solution_set_inequality : {x : ℝ | 3 - 2 * x - x^2 < 0} = set.Iio (-3) ∪ set.Ioi 1 :=
sorry

end solution_set_inequality_l116_116513


namespace opposite_of_neg_2023_l116_116968

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l116_116968


namespace opposite_of_neg_2023_l116_116900

-- Let x be the opposite of -2023
def is_opposite (x : ℤ) (n : ℤ) : Prop := n + x = 0

theorem opposite_of_neg_2023 : ∃ x : ℤ, is_opposite x (-2023) ∧ x = 2023 :=
by
  exists 2023
  unfold is_opposite
  sorry

end opposite_of_neg_2023_l116_116900


namespace find_u_l116_116080

theorem find_u : 
  (∃ x : ℝ, x = ( -15 - sqrt 205 ) / 6 ∧ 3*x^2 + 15*x + (5/3) = 0) →
  (∀ u : ℝ, (∃ x : ℝ, x = ( -15 - sqrt 205 ) / 6 ∧ 3*x^2 + 15*x + u = 0) → u = 5/3) := 
by
  sorry

end find_u_l116_116080


namespace digit_divisible_by_3_l116_116079

theorem digit_divisible_by_3 (d : ℕ) (h : d < 10) : (15780 + d) % 3 = 0 ↔ d = 0 ∨ d = 3 ∨ d = 6 ∨ d = 9 := by
  sorry

end digit_divisible_by_3_l116_116079


namespace shortest_distance_longest_distance_l116_116390

-- Given: a triangle ABC with a point P inside, and D, E, F being the feet 
-- of the perpendiculars from P to sides BC, CA, and AB respectively. 
-- PD, PE, and PF are the perpendicular distances.
-- PA, PB, and PC are the distances from P to vertices A, B, and C respectively.

variables {A B C P D E F : Type*}
variables [IsTriangle A B C] [IsPointInTriangle P A B C]
variables [FootOfPerpendicular D P B C]
variables [FootOfPerpendicular E P C A]
variables [FootOfPerpendicular F P A B]

-- Define the distances
noncomputable def PD : ℝ := distance P D
noncomputable def PE : ℝ := distance P E
noncomputable def PF : ℝ := distance P F
noncomputable def PA : ℝ := distance P A
noncomputable def PB : ℝ := distance P B
noncomputable def PC : ℝ := distance P C

-- Proof that the shortest distance is the minimum of PD, PE, PF
theorem shortest_distance : shortest_distance_in_triangle P A B C = min (PD) (min (PE) (PF)) :=
sorry

-- Proof that the longest distance is the maximum of PA, PB, PC
theorem longest_distance : longest_distance_in_triangle P A B C = max (PA) (max (PB) (PC)) :=
sorry

end shortest_distance_longest_distance_l116_116390


namespace train_length_l116_116001

-- Conditions
def train_speed_kmph := 60
def time_to_pass_seconds := 21.598272138228943
def platform_length_meters := 240
def speed_mps := train_speed_kmph * 1000 / 3600
def total_distance_meters := speed_mps * time_to_pass_seconds

-- The theorem we need to prove
theorem train_length :
  total_distance_meters - platform_length_meters = 120 := by
  sorry

end train_length_l116_116001


namespace number_of_possible_values_of_P_l116_116668

-- Defining the problem parameters
def is_subset_of (s t : set ℕ) : Prop := ∀ x ∈ s, x ∈ t
def is_70_element_subset (C : set ℕ) : Prop := is_subset_of C {n : ℕ | 1 ≤ n ∧ n ≤ 120} ∧ C.card = 70
def sum_of_elements (C : set ℕ) : ℕ := C.to_finset.sum id

-- Stating the theorem
theorem number_of_possible_values_of_P : 
  ∃! (n : ℕ), n = 3501 ∧ 
    ∀ P : ℕ, (∃ C : set ℕ, is_70_element_subset C ∧ P = sum_of_elements C) ↔ (2485 ≤ P ∧ P ≤ 5985) :=
sorry

end number_of_possible_values_of_P_l116_116668


namespace second_dog_average_miles_l116_116014

theorem second_dog_average_miles
  (total_miles_week : ℕ)
  (first_dog_miles_day : ℕ)
  (days_in_week : ℕ)
  (h1 : total_miles_week = 70)
  (h2 : first_dog_miles_day = 2)
  (h3 : days_in_week = 7) :
  (total_miles_week - (first_dog_miles_day * days_in_week)) / days_in_week = 8 := sorry

end second_dog_average_miles_l116_116014


namespace calculate_expression_l116_116024

theorem calculate_expression : (-1^4 + |1 - Real.sqrt 2| - (Real.pi - 3.14)^0) = Real.sqrt 2 - 3 :=
by
  sorry

end calculate_expression_l116_116024


namespace opposite_of_neg_2023_l116_116913

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by 
  sorry

end opposite_of_neg_2023_l116_116913


namespace probability_odd_is_one_half_l116_116328

noncomputable def probability_odd_four_digit_number : ℚ :=
  let digits := {1, 4, 6, 9}
  let odd_digits := {d ∈ digits | d % 2 = 1}
  (odd_digits.card : ℚ) / (digits.card : ℚ)

theorem probability_odd_is_one_half (digits : Finset ℕ) (odd_digits : Finset ℕ) :
  digits = {1, 4, 6, 9} →
  odd_digits = {d ∈ digits | d % 2 = 1} →
  probability_odd_four_digit_number = 1 / 2 :=
by
  intros h1 h2
  sorry

end probability_odd_is_one_half_l116_116328


namespace tan_shifted_value_l116_116169

theorem tan_shifted_value (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = -((6 + 5 * Real.sqrt 3) / 13) :=
by
  sorry

end tan_shifted_value_l116_116169


namespace smiths_bakery_pies_l116_116307

theorem smiths_bakery_pies (m : ℕ) (h : m = 16) : 4 * m + 6 = 70 :=
by
  rw [h]
  norm_num
  sorry

end smiths_bakery_pies_l116_116307


namespace opposite_of_neg_2023_l116_116948

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l116_116948


namespace opposite_of_neg_2023_l116_116841

def opposite (n : ℤ) : ℤ :=
  -n

theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116841


namespace opposite_of_neg_2023_l116_116822

/-- Define the opposite of a number -/
def opposite (x : ℤ) : ℤ := -x

/-- Prove the opposite of -2023 is 2023 -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116822


namespace probability_interval_contains_q_l116_116474

theorem probability_interval_contains_q (P_C P_D : ℝ) (q : ℝ)
    (hC : P_C = 5 / 7) (hD : P_D = 3 / 4) :
    (5 / 28 ≤ q ∧ q ≤ 5 / 7) ↔ (max (P_C + P_D - 1) 0 ≤ q ∧ q ≤ min P_C P_D) :=
by
  sorry

end probability_interval_contains_q_l116_116474


namespace boxes_used_l116_116432

-- Definitions of the conditions
def total_oranges := 2650
def oranges_per_box := 10

-- Statement to prove
theorem boxes_used (total_oranges oranges_per_box : ℕ) : (total_oranges = 2650) → (oranges_per_box = 10) → (total_oranges / oranges_per_box = 265) :=
by
  intros h_total h_per_box
  rw [h_total, h_per_box]
  norm_num

end boxes_used_l116_116432


namespace opposite_of_negative_2023_l116_116784

theorem opposite_of_negative_2023 : (- -2023) = 2023 := 
by {
sorry
}

end opposite_of_negative_2023_l116_116784


namespace volume_of_parallelepiped_with_proximity_l116_116473

/-- 
Given a rectangular parallelepiped measuring 2x3x6 units, we consider the set of points that are 
inside or within one unit of the box. The volume of this set is given as (m + nπ) / p, 
where m, n, and p are positive integers, and n and p are relatively prime. Prove that 
m + n + p = 364.
-/

theorem volume_of_parallelepiped_with_proximity :
  ∃ (m n p : ℕ), (n.gcd p = 1) ∧ ((m : ℝ) + n * Real.pi) / p = 108 + (37 * Real.pi) / 3 ∧ m + n + p = 364 :=
begin
  sorry
end

end volume_of_parallelepiped_with_proximity_l116_116473


namespace problem_statement_l116_116115

variable (x y : ℝ)
variable (h_cond1 : 1 / x + 1 / y = 4)
variable (h_cond2 : x * y - x - y = -7)

theorem problem_statement (h_cond1 : 1 / x + 1 / y = 4) (h_cond2 : x * y - x - y = -7) : 
  x^2 * y + x * y^2 = 196 / 9 := 
sorry

end problem_statement_l116_116115


namespace cos_angle_identity_l116_116577

def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (2, 2)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def cos_angle (v1 v2 : ℝ × ℝ) : ℝ :=
  dot_product v1 v2 / (magnitude v1 * magnitude v2)

theorem cos_angle_identity :
  cos_angle (a.1 + b.1, a.2 + b.2) (a.1 - b.1, a.2 - b.2) = (real.sqrt 17) / 17 :=
by
  finish

end cos_angle_identity_l116_116577


namespace opposite_of_neg_2023_l116_116896

-- Let x be the opposite of -2023
def is_opposite (x : ℤ) (n : ℤ) : Prop := n + x = 0

theorem opposite_of_neg_2023 : ∃ x : ℤ, is_opposite x (-2023) ∧ x = 2023 :=
by
  exists 2023
  unfold is_opposite
  sorry

end opposite_of_neg_2023_l116_116896


namespace find_value_of_expression_l116_116550

theorem find_value_of_expression
  (α : ℝ)
  (h_f : ∀ x ≥ 0, f x = |sin x|)
  (h_intersection : number_of_intersections f (λ x, x) = 3)
  (h_alpha_max : ∀ t, is_intersection t → t ≤ α) :
  (1 + α^2) * sin (2 * α) / α = 2 :=
sorry

end find_value_of_expression_l116_116550


namespace basketball_probability_l116_116571

-- Define the probabilities of A and B making a shot
def prob_A : ℝ := 0.4
def prob_B : ℝ := 0.6

-- Define the probability that both miss their shots in one round
def prob_miss_one_round : ℝ := (1 - prob_A) * (1 - prob_B)

-- Define the probability that A takes k shots to make a basket
noncomputable def P_xi (k : ℕ) : ℝ := (prob_miss_one_round)^(k-1) * prob_A

-- State the theorem
theorem basketball_probability (k : ℕ) : 
  P_xi k = 0.24^(k-1) * 0.4 :=
by
  unfold P_xi
  unfold prob_miss_one_round
  sorry

end basketball_probability_l116_116571


namespace estimated_watched_students_l116_116212

-- Definitions for the problem conditions
def total_students : ℕ := 3600
def surveyed_students : ℕ := 200
def watched_students : ℕ := 160

-- Problem statement (proof not included yet)
theorem estimated_watched_students :
  total_students * (watched_students / surveyed_students : ℝ) = 2880 := by
  -- skipping proof step
  sorry

end estimated_watched_students_l116_116212


namespace find_n_l116_116498

theorem find_n (n : ℕ) :
  (∀ k : ℕ, k > 0 → k^2 + (n / k^2) ≥ 1991) ∧ (∃ k : ℕ, k > 0 ∧ k^2 + (n / k^2) < 1992) ↔ 967 * 1024 ≤ n ∧ n < 968 * 1024 :=
by
  sorry

end find_n_l116_116498


namespace num_seven_digit_palindromes_l116_116068

theorem num_seven_digit_palindromes : 
  let a_choices := 9
  let b_choices := 10
  let c_choices := 10
  let d_choices := 10
  a_choices * b_choices * c_choices * d_choices = 9000 :=
by
  sorry

end num_seven_digit_palindromes_l116_116068


namespace probability_rolls_more_ones_than_eights_l116_116186

noncomputable def probability_more_ones_than_eights (n : ℕ) := 10246 / 32768

theorem probability_rolls_more_ones_than_eights :
  (probability_more_ones_than_eights 5) = 10246 / 32768 :=
by
  sorry

end probability_rolls_more_ones_than_eights_l116_116186


namespace savings_after_purchase_l116_116678

theorem savings_after_purchase :
  let price_sweater := 30
  let price_scarf := 20
  let num_sweaters := 6
  let num_scarves := 6
  let savings := 500
  let total_cost := (num_sweaters * price_sweater) + (num_scarves * price_scarf)
  savings - total_cost = 200 :=
by
  sorry

end savings_after_purchase_l116_116678


namespace equality_of_pow_l116_116180

theorem equality_of_pow (a b : ℝ) (h : {a^2, 0, -1} = {a, b, 0}) : a^2023 + b^2023 = 0 :=
by
  sorry

end equality_of_pow_l116_116180


namespace find_integer_k_l116_116114

noncomputable def circle_center (x : ℝ) : ℝ × ℝ := (1, 0)
noncomputable def circle_radius : ℝ := 1
noncomputable def line_center (k : ℝ) (x : ℝ) : ℝ × ℝ := (x, k * x - 2)
noncomputable def circle_distance (k : ℝ) : ℝ := abs (k - 2) / sqrt (k^2 + 1)
noncomputable def non_intersect_cond (k : ℝ) (l : ℝ) : Prop := circle_distance k > 1 + l

theorem find_integer_k : ∃ k : ℤ, non_intersect_cond k 1 :=
begin
  use (-1),
  sorry
end

end find_integer_k_l116_116114


namespace maria_savings_l116_116676

-- Conditions
def sweater_cost : ℕ := 30
def scarf_cost : ℕ := 20
def num_sweaters : ℕ := 6
def num_scarves : ℕ := 6
def savings : ℕ := 500

-- The proof statement
theorem maria_savings : savings - (num_sweaters * sweater_cost + num_scarves * scarf_cost) = 200 :=
by
  sorry

end maria_savings_l116_116676


namespace acute_triangle_locus_l116_116531

noncomputable def locus_of_acute_triangles (A B : Point) : Set Point :=
  let ω := Circle (segment A B / 2) -- Circle with AB as diameter
  let l_A := PerpendicularLineAt A (segment A B)
  let l_B := PerpendicularLineAt B (segment A B)
  { C | C ∉ ω ∧ C ∉ segment A B ∧ C ∈ region_bounded_by l_A l_B }

theorem acute_triangle_locus (A B : Point) : 
  is_locus_of_acute_triangles (locus_of_acute_triangles A B) :=
  sorry

end acute_triangle_locus_l116_116531


namespace CoinRun_ProcGen_ratio_l116_116134

theorem CoinRun_ProcGen_ratio
  (greg_ppo_reward: ℝ)
  (maximum_procgen_reward: ℝ)
  (ppo_ratio: ℝ)
  (maximum_coinrun_reward: ℝ)
  (coinrun_to_procgen_ratio: ℝ)
  (greg_ppo_reward_eq: greg_ppo_reward = 108)
  (maximum_procgen_reward_eq: maximum_procgen_reward = 240)
  (ppo_ratio_eq: ppo_ratio = 0.90)
  (coinrun_equation: maximum_coinrun_reward = greg_ppo_reward / ppo_ratio)
  (ratio_definition: coinrun_to_procgen_ratio = maximum_coinrun_reward / maximum_procgen_reward) :
  coinrun_to_procgen_ratio = 0.5 :=
sorry

end CoinRun_ProcGen_ratio_l116_116134


namespace john_moves_540kg_l116_116233

-- Conditions
def used_to_back_squat : ℝ := 200
def increased_by : ℝ := 50
def front_squat_ratio : ℝ := 0.8
def triple_ratio : ℝ := 0.9

-- Definitions based on conditions
def current_back_squat : ℝ := used_to_back_squat + increased_by
def current_front_squat : ℝ := front_squat_ratio * current_back_squat
def one_triple : ℝ := triple_ratio * current_front_squat
def three_triples : ℝ := 3 * one_triple

-- The proof statement
theorem john_moves_540kg : three_triples = 540 := by
  sorry

end john_moves_540kg_l116_116233


namespace part1_solution_part2_no_solution_l116_116117

theorem part1_solution (x : ℝ) : 
  (let m := 3 in (3 - 2 * x) / (x - 2) - (m * x - 2) / (2 - x) = -1) ↔ x = 1 / 2 :=
by
  sorry

theorem part2_no_solution (m : ℝ) : 
  (∀ x : ℝ, (3 - 2 * x) / (x - 2) - (m * x - 2) / (2 - x) = -1 → x ≠ 2) ↔ (m = 1 ∨ m = 3 / 2) :=
by
  sorry

end part1_solution_part2_no_solution_l116_116117


namespace opposite_of_neg_2023_l116_116952

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l116_116952


namespace arg_z_range_l116_116524

noncomputable def arg_range (z : ℂ) : Set ℝ :=
  {θ : ℝ | θ ∈ (Set.Ioo (5 * Real.pi / 6 - Real.arcsin (Real.sqrt 3 / 3)) Real.pi ∪ 
                Set.Ioo Real.pi (7 * Real.pi / 6 + Real.arcsin (Real.sqrt 3 / 3)))}

theorem arg_z_range (z : ℂ) (hz : abs (Complex.arg ((z + 1) / (z + 2))) = Real.pi / 6) :
  arg z ∈ arg_range z :=
  sorry

end arg_z_range_l116_116524


namespace geometric_sequence_sum_first_n_terms_l116_116645

noncomputable def a₁ : ℕ := 2
noncomputable def q : ℕ := 2
noncomputable def a (n : ℕ) : ℕ := a₁ * q^(n-1)
noncomputable def b (n : ℕ) : ℕ := 1 + (n-1) * 2
noncomputable def Sn (n : ℕ) : ℕ := finset.range n.sum (λ i, a (i + 1) + b (i + 1))

theorem geometric_sequence (n : ℕ) : a n = 2^n := sorry

theorem sum_first_n_terms (n : ℕ) : Sn n = 2^(n+1) + n^2 - 2 := sorry

end geometric_sequence_sum_first_n_terms_l116_116645


namespace probability_of_9_heads_in_12_flips_l116_116378

open BigOperators

-- We state the problem in terms of probability theory.
theorem probability_of_9_heads_in_12_flips :
  let p : ℚ := (nat.choose 12 9) / (2 ^ 12)
  p = 55 / 1024 :=
by
  sorry

end probability_of_9_heads_in_12_flips_l116_116378


namespace find_f_1_l116_116088

-- Given condition
def f (x : ℝ) : ℝ :=
  if h : ∃ y, x = (y - 3) then 2 * (classical.some h) * (classical.some h) - 3 * (classical.some h) + 1
  else 0 -- This introduces a default case

theorem find_f_1 : f 1 = 21 :=
by
  -- Skipping the proof
  sorry

end find_f_1_l116_116088


namespace opposite_of_negative_2023_l116_116794

theorem opposite_of_negative_2023 : (- -2023) = 2023 := 
by {
sorry
}

end opposite_of_negative_2023_l116_116794


namespace pow_two_add_sq_l116_116028

example : (105: ℕ)^2 = 11025 := by
  calc
    (105: ℕ)^2 = (100 + 5: ℕ)^2       := by rfl
            ... = (100: ℕ)^2 + 2 * 100 * 5 + (5: ℕ)^2 := by rw pow_two_add_sq
            ... = 10000 + 1000 + 25    := by norm_num
            ... = 11025                := by norm_num

-- The statement (pow_two_add_sq) needs to be defined as:
theorem pow_two_add_sq {a b : ℕ} : (a + b)^2 = a^2 + 2 * a * b + b^2 :=
by sorry

end pow_two_add_sq_l116_116028


namespace train_speed_is_correct_l116_116447

-- Define the conditions
def train_length : ℝ := 124
def platform_length : ℝ := 234.9176
def crossing_time : ℝ := 19

-- Define the speed calculation
def speed_in_kmph : ℝ :=
  (train_length + platform_length) / crossing_time * 3.6

-- The theorem statement
theorem train_speed_is_correct : speed_in_kmph = 68.00544 := by
  -- Proof is skipped
  sorry

end train_speed_is_correct_l116_116447


namespace opposite_of_neg_2023_l116_116985

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l116_116985


namespace tiling_equivalence_l116_116280

-- Definitions based on conditions
def Board := Fin 1993 × Fin 1993
def is_edge_square (s : Board) : Prop :=
  s.1 = FZ ∨ s.1 = FZ.pred ∨ s.2 = FZ ∨ s.2 = FZ.pred

def are_separated_by_odd_squares (A B : Board) : Prop :=
  A ≠ B ∧ is_edge_square A ∧ is_edge_square B ∧ (abs (A.1 - B.1) + abs (A.2 - B.2)) % 2 = 1

-- Define the tiling problem without specific solutions
def num_tilings_without (T : fin 1993 × fin 1993) (P : T → Prop) : ℕ := sorry

theorem tiling_equivalence 
  (A B : Board)
  (h_edge_A : is_edge_square A)
  (h_edge_B : is_edge_square B)
  (h_separation : are_separated_by_odd_squares A B) :
  num_tilings_without (⊤ : Fin 1993 × Fin 1993) (λ x, x ≠ A) = 
  num_tilings_without (⊤ : Fin 1993 × Fin 1993) (λ x, x ≠ B) := 
sorry

end tiling_equivalence_l116_116280


namespace log_evaluation_l116_116055

theorem log_evaluation : log 4 (4 * real.sqrt 2) = 5 / 4 := by
  sorry

end log_evaluation_l116_116055


namespace rectangle_area_l116_116213

theorem rectangle_area 
  {ABCD : Type*}
  (Hrect : is_rectangle ABCD)
  (Htrisect : ∃ CF CE : ABCD, is_trisector CF CE ∧ (CF ∩ CE = C))
  (E : ABCD) (H_EonAB : is_on_line E AB)
  (F : ABCD) (H_FonAD : is_on_line F AD)
  (H_BE : length BE = 4)
  (H_AF : length AF = 8) :
  area ABCD = 96 * real.sqrt 3 - 96 :=
by sorry

end rectangle_area_l116_116213


namespace complex_number_solution_l116_116105

noncomputable def z_derivative (z : ℂ → ℂ) : ℂ := sorry

theorem complex_number_solution (z : ℂ → ℂ) (t : ℝ) (C1 C2 : ℝ) :
  (z_derivative z) / (1 - (complex.I)) = complex.I^2017 →
  z(t) = (1 + complex.I) * t + C1 + complex.I * C2 :=
sorry

end complex_number_solution_l116_116105


namespace pizza_slices_left_l116_116005

theorem pizza_slices_left (initial_slices : ℕ) (people : ℕ) (slices_per_person : ℕ) 
  (h1 : initial_slices = 16) (h2 : people = 6) (h3 : slices_per_person = 2) : 
  initial_slices - people * slices_per_person = 4 := 
by
  sorry

end pizza_slices_left_l116_116005


namespace symmetry_about_y_axis_l116_116754

def f (x : ℝ) : ℝ := (4^x + 1) / 2^x

theorem symmetry_about_y_axis : ∀ (x : ℝ), f x = f (-x) :=
by
  sorry

end symmetry_about_y_axis_l116_116754


namespace tan_angle_addition_l116_116147

theorem tan_angle_addition (x : Real) (h1 : Real.tan x = 3) (h2 : Real.tan (Real.pi / 3) = Real.sqrt 3) : 
  Real.tan (x + Real.pi / 3) = (3 + Real.sqrt 3) / (1 - 3 * Real.sqrt 3) :=
by 
  sorry

end tan_angle_addition_l116_116147


namespace unique_strictly_increasing_function_l116_116073

noncomputable def strictly_increasing (f : ℕ+ → ℕ+) : Prop :=
  ∀ {a b : ℕ+}, a < b → f a < f b

theorem unique_strictly_increasing_function
  (f : ℕ+ → ℕ+)
  (h_increasing : strictly_increasing f)
  (h_image : ∀ n : ℕ+, f n ∈ ℕ+)
  (h_f2 : f 2 = 2)
  (h_mul : ∀ n m : ℕ+, f (n * m) = f n * f m) :
  ∀ n : ℕ+, f n = n := 
sorry

end unique_strictly_increasing_function_l116_116073


namespace triangle_largest_angle_l116_116364

theorem triangle_largest_angle
  {A B C : Type}
  (hABC_isosceles_right : is_isosceles_right A B C)
  (hA_30 : angle_measure A = 30) :
  ∃ B, angle_measure B = 120 := by
  sorry

end triangle_largest_angle_l116_116364


namespace minimum_slope_tangent_point_coordinates_l116_116557

theorem minimum_slope_tangent_point_coordinates :
  ∃ a : ℝ, a > 0 ∧ (∀ x : ℝ, (2 * x + a / x ≥ 4) ∧ (2 * x + a / x = 4 ↔ x = 1)) → 
  (1, 1) = (1, 1) := by
sorry

end minimum_slope_tangent_point_coordinates_l116_116557


namespace original_equation_proof_l116_116295

theorem original_equation_proof :
  ∃ (A O H M J : ℕ),
  A ≠ O ∧ A ≠ H ∧ A ≠ M ∧ A ≠ J ∧
  O ≠ H ∧ O ≠ M ∧ O ≠ J ∧
  H ≠ M ∧ H ≠ J ∧
  M ≠ J ∧
  A + 8 * (10 * O + H) = 10 * M + J ∧
  (O = 1) ∧ (H = 2) ∧ (M = 9) ∧ (J = 6) ∧ (A = 0) :=
by
  sorry

end original_equation_proof_l116_116295


namespace tan_angle_addition_l116_116149

theorem tan_angle_addition (x : Real) (h1 : Real.tan x = 3) (h2 : Real.tan (Real.pi / 3) = Real.sqrt 3) : 
  Real.tan (x + Real.pi / 3) = (3 + Real.sqrt 3) / (1 - 3 * Real.sqrt 3) :=
by 
  sorry

end tan_angle_addition_l116_116149


namespace sqrt_product_simplifies_l116_116465

theorem sqrt_product_simplifies (x : ℝ) : sqrt (96 * x^2) * sqrt (50 * x) * sqrt (28 * x^3) = 1260 * x^3 := by
  sorry

end sqrt_product_simplifies_l116_116465


namespace simplify_expression_l116_116735

variable {a : ℝ}

theorem simplify_expression (h1 : a ≠ 2) (h2 : a ≠ -2) :
  ((a^2 + 4*a + 4) / (a^2 - 4) - (a + 3) / (a - 2)) / ((a + 2) / (a - 2)) = -1 / (a + 2) :=
by
  sorry

end simplify_expression_l116_116735


namespace nancy_jason_ratio_l116_116683

noncomputable def ratio_step_feet (N J : ℕ) : Prop :=
N + J = 32 ∧ J = 8 ∧ N / J = 3

theorem nancy_jason_ratio :
  ∃ N J : ℕ, ratio_step_feet N J :=
by {
  existsi 24,
  existsi 8,
  simp [ratio_step_feet],
  sorry
}

end nancy_jason_ratio_l116_116683


namespace orthocentric_tetrahedron_part_a_orthocentric_tetrahedron_part_b_l116_116404

-- Part (a)
theorem orthocentric_tetrahedron_part_a :
  ∀ (AB AC AD BC : ℝ),
  AB = 5 → AC = 7 → AD = 8 → BC = 6 →
  (∃ CD BD : ℝ, CD = 5 * Real.sqrt 3 ∧ BD = Real.sqrt 51) :=
by
  intros AB AC AD BC hAB hAC hAD hBC
  use 5 * Real.sqrt 3, Real.sqrt 51
  simp [hAB, hAC, hAD, hBC]
  sorry

-- Part (b)
theorem orthocentric_tetrahedron_part_b :
  ∀ (AB BC DC : ℝ),
  AB = 8 → BC = 12 → DC = 6 →
  ¬ (∃ AD : ℝ, AB * AB + DC * DC = BC * BC + AD * AD) :=
by
  intros AB BC DC hAB hBC hDC
  intro h
  obtain ⟨AD, hAD⟩ := h
  simp [hAB, hBC, hDC] at hAD
  sorry

end orthocentric_tetrahedron_part_a_orthocentric_tetrahedron_part_b_l116_116404


namespace compare_a_b_c_l116_116520

noncomputable def a : ℝ := (3 / 4) ^ (2 / 3)
noncomputable def b : ℝ := (2 / 3) ^ (3 / 4)
noncomputable def c : ℝ := Real.logBase (2 / 3) (4 / 3)

theorem compare_a_b_c : a > b ∧ b > c :=
by
  sorry

end compare_a_b_c_l116_116520


namespace opposite_of_neg_2023_l116_116966

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l116_116966


namespace distinct_monomial_count_l116_116219

theorem distinct_monomial_count : 
  let f := λ (x y z : ℚ), (x + y + z) ^ 2022 + (x - y - z) ^ 2022 in 
  ∃ (count : ℕ), count = 1024144 ∧ 
                   ∀ (x y z : ℚ), 
                     (f x y z).monomials.size = count := 
begin
  sorry
end

end distinct_monomial_count_l116_116219


namespace solve_for_x_l116_116736

theorem solve_for_x (x : ℝ) : 3^(4 * x) = (81 : ℝ)^(1 / 4) → x = 1 / 4 :=
by
  intros
  sorry

end solve_for_x_l116_116736


namespace lines_parallel_or_not_l116_116287

variables {α : Type*} [Plane α] {a b : Line α} {A B : Point α}

-- Assumptions or conditions
axiom perp_a : Perpendicular a α
axiom perp_b : Perpendicular b α
axiom intersect_a : Intersects a α A
axiom intersect_b : Intersects b α B
axiom ab_in_alpha : In_Plane (Line_through A B) α
axiom perp_ab_a : Perpendicular a (Line_through A B)
axiom perp_ab_b : Perpendicular b (Line_through A B)

-- Statement of the proposition
theorem lines_parallel_or_not :
  (Perpendicular_to_Plane_Prop a α ∧ Perpendicular_to_Plane_Prop b α ∧ In_Plane_Prop (Line_through A B) α ∧ Perpendicular_to_Line_Prop a (Line_through A B) ∧ Perpendicular_to_Line_Prop b (Line_through A B)) → 
  ((p : (Perpendicular a α ∧ Perpendicular b α ∧ In_Plane (Line_through A B) α) → Perpendicular_to_Line_Prop a (Line_through A B) ∧ Perpendicular_to_Line_Prop b (Line_through A B)) ∨ 
  (q : (Perpendicular_to_Line_Prop a (Line_through A B) ∧ Perpendicular_to_Line_Prop b (Line_through A B)) → Parallel a b)) :=
sorry

end lines_parallel_or_not_l116_116287


namespace opposite_of_neg_2023_l116_116932

theorem opposite_of_neg_2023 : ∃ x : ℤ, -2023 + x = 0 ∧ x = 2023 := 
by 
  use 2023
  constructor
  · exact rfl
  · exact rfl

end opposite_of_neg_2023_l116_116932


namespace quadratic_solutions_square_eq_solutions_l116_116315

-- Problem 1: Prove solutions for 2x^2 - 3x + 1 = 0
theorem quadratic_solutions :
  (∀ x, 2 * x^2 - 3 * x + 1 = 0 ↔ x = 1 ∨ x = 1 / 2) :=
by
  sorry

-- Problem 2: Prove solutions for (y - 2)^2 = (2y + 3)^2
theorem square_eq_solutions :
  (∀ y, (y - 2)^2 = (2y + 3)^2 ↔ y = -5 ∨ y = -1 / 3) :=
by
  sorry

end quadratic_solutions_square_eq_solutions_l116_116315


namespace skittles_transfer_l116_116464

-- Define the initial number of Skittles Bridget and Henry have
def bridget_initial_skittles := 4
def henry_initial_skittles := 4

-- The main statement we want to prove
theorem skittles_transfer :
  bridget_initial_skittles + henry_initial_skittles = 8 :=
by
  sorry

end skittles_transfer_l116_116464


namespace opposite_of_neg_2023_l116_116964

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l116_116964


namespace relationship_among_abc_l116_116538

noncomputable def f : ℝ → ℝ := sorry  -- Define f as a general odd increasing function.

def a := -f(log 2 (1 / 5))
def b := f(log 2 4.1)
def c := f(2 ^ 0.8)

theorem relationship_among_abc (h_odd : ∀ x, f (-x) = -f x) (h_inc : ∀ x y, x < y → f x < f y) :
  c < b ∧ b < a :=
by
  have h_a : a = f (log 2 5), from
    have h_log : log 2 (1 / 5) = - log 2 5, by sorry,  -- logarithmic identity
    calc
      a = -f (log 2 (1 / 5)) : rfl
      ... = -f (-log 2 5) : by rw [h_log]
      ... = f (log 2 5) : by rw [h_odd],
  have h_b : b = f (log 2 4.1), from rfl,
  have h_c : c = f (2 ^ 0.8), from rfl,
  have h_order : 1 < 2 ^ 0.8 ∧ 2 ^ 0.8 < log 2 4.1 ∧ log 2 4.1 < log 2 5, by sorry,
  have h_inc_1 : f (2 ^ 0.8) < f (log 2 4.1), from h_inc (2 ^ 0.8) (log 2 4.1) h_order.1,
  have h_inc_2 : f (log 2 4.1) < f (log 2 5), from h_inc (log 2 4.1) (log 2 5) h_order.2.right,
  exact ⟨h_inc_1, h_inc_2⟩

end relationship_among_abc_l116_116538


namespace distance_center_to_point_l116_116640

noncomputable def distance_from_center_to_point (P : ℝ^3) (O : ℝ^3) : ℝ :=
  √3

-- Given conditions
variables (P A B C : ℝ^3)
variables (rays_angle_PA_PB : ℝ)
variables (rays_angle_PB_PC : ℝ)
variables (rays_angle_PC_PA : ℝ)
variables (sphere_radius : ℝ)

-- Conditions, as described in the problem
def non_coplanar_rays : Prop := 
  linearly_independent ℝ ![P - A, P - B, P - C]

def rays_form_angles : Prop :=
  rays_angle_PA_PB = 60 ∧ rays_angle_PB_PC = 60 ∧ rays_angle_PC_PA = 60

def sphere_tangent_to_rays : Prop :=
  sphere_radius = 1

-- Define the main problem to prove the distance
def proof_problem : Prop :=
  non_coplanar_rays P A B C ∧
  rays_form_angles rays_angle_PA_PB rays_angle_PB_PC rays_angle_PC_PA ∧
  sphere_tangent_to_rays sphere_radius →
  distance_from_center_to_point P O = √3

-- Statement to be proved
theorem distance_center_to_point (P A B C : ℝ^3)
  (rays_angle_PA_PB rays_angle_PB_PC rays_angle_PC_PA sphere_radius : ℝ) :
  proof_problem P A B C rays_angle_PA_PB rays_angle_PB_PC rays_angle_PC_PA sphere_radius :=
sorry

end distance_center_to_point_l116_116640


namespace count_seven_digit_palindromes_l116_116071

theorem count_seven_digit_palindromes : 
  let palindromes := { n : ℤ | ∃ (a b c d : ℕ), 
    1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧ 
    n = 1000000 * a + 100000 * b + 10000 * c + 1000 * d + 100 * c + 10 * b + a } in
  ∃ (count : ℕ), count = 9000 ∧ count = palindromes.card :=
by
  -- the proof goes here
  sorry

end count_seven_digit_palindromes_l116_116071


namespace opposite_of_neg2023_l116_116998

theorem opposite_of_neg2023 : ∀ (x : Int), -2023 + x = 0 → x = 2023 :=
by
  intros x h
  sorry

end opposite_of_neg2023_l116_116998


namespace angle_B_possibilities_l116_116248

variables {A B C O H : Type*}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space O] [metric_space H]
variables [affine_space A] [affine_space B] [affine_space C]

def is_circumcenter (O : affine_space) (A B C : affine_space) : Prop := sorry
def is_orthocenter (H : affine_space) (A B C : affine_space) : Prop := sorry

theorem angle_B_possibilities {A B C : Type*} [metric_space A] [metric_space B] [metric_space C]
  [affine_space A] [affine_space B] [affine_space C] {O H : affine_space}
  (h_circumcenter : is_circumcenter O A B C) (h_orthocenter : is_orthocenter H A B C)
  (h_eq : dist B O = dist B H) :
  ∃ (α : ℝ), α = 60 ∨ α = 120 :=
sorry

end angle_B_possibilities_l116_116248


namespace new_person_weight_l116_116324

theorem new_person_weight (n : ℕ) (k : ℝ) (w_old w_new : ℝ) 
  (h_n : n = 6) 
  (h_k : k = 4.5) 
  (h_w_old : w_old = 75) 
  (h_avg_increase : w_new - w_old = n * k) : 
  w_new = 102 := 
sorry

end new_person_weight_l116_116324


namespace simplifyExpressionSimplified_l116_116734

def simplifyExpression (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 0) : Prop :=
  ( ( (x + 1) / (x - 2) - 1 ) / ( (x^2 - 2 * x) / (x^2 - 4 * x + 4) ) = 3 / x

theorem simplifyExpressionSimplified (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 0) :
  simplifyExpression x h1 h2 :=
sorry

end simplifyExpressionSimplified_l116_116734


namespace problem_statement_l116_116475

theorem problem_statement (n : ℕ) (h : ∀ (x : fin n → ℤ), ¬ (n ∣ ∑ i, x i) → 
  ∃ j : fin n, ∀ k : ℕ, 0 < k → k ≤ n → ¬ (n ∣ (∑ i in finset.range k, x ((j + i) % n)))) : 
  (nat.prime n ∨ n = 1) :=
sorry

end problem_statement_l116_116475


namespace opposite_of_neg_2023_l116_116934

theorem opposite_of_neg_2023 : ∃ x : ℤ, -2023 + x = 0 ∧ x = 2023 := 
by 
  use 2023
  constructor
  · exact rfl
  · exact rfl

end opposite_of_neg_2023_l116_116934


namespace greg_and_earl_total_l116_116487

variable (Earl_initial Fred_initial Greg_initial : ℕ)
variable (Earl_to_Fred Fred_to_Greg Greg_to_Earl : ℕ)

def Earl_final := Earl_initial - Earl_to_Fred + Greg_to_Earl
def Fred_final := Fred_initial + Earl_to_Fred - Fred_to_Greg
def Greg_final := Greg_initial + Fred_to_Greg - Greg_to_Earl

theorem greg_and_earl_total :
  Earl_initial = 90 → Fred_initial = 48 → Greg_initial = 36 →
  Earl_to_Fred = 28 → Fred_to_Greg = 32 → Greg_to_Earl = 40 →
  Greg_final + Earl_final = 130 :=
by
  intros h1 h2 h3 h4 h5 h6
  dsimp [Earl_final, Fred_final, Greg_final]
  rw [h1, h2, h3, h4, h5, h6]
  exact sorry

end greg_and_earl_total_l116_116487


namespace min_value_S_invariant_l116_116243

noncomputable def parabola_area_sum_min_value (m : ℤ) (hm : m > 0) : ℤ :=
let S_min_value := m^3 / 3 
in
S_min_value

theorem min_value_S_invariant (m : ℤ) (hm : m > 0) :
  ∀ (z : ℝ), ∃ (t : ℝ), S (m : ℝ) (z : ℝ) (t : ℝ) = m^3 / 3 :=
sorry

end min_value_S_invariant_l116_116243


namespace ratio_of_pages_given_l116_116240

variable (Lana_initial_pages : ℕ) (Duane_initial_pages : ℕ) (Lana_final_pages : ℕ)

theorem ratio_of_pages_given
  (h1 : Lana_initial_pages = 8)
  (h2 : Duane_initial_pages = 42)
  (h3 : Lana_final_pages = 29) :
  (Lana_final_pages - Lana_initial_pages) / Duane_initial_pages = 1 / 2 :=
  by
  -- Placeholder for the proof
  sorry

end ratio_of_pages_given_l116_116240


namespace tan_shifted_value_l116_116171

theorem tan_shifted_value (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = -((6 + 5 * Real.sqrt 3) / 13) :=
by
  sorry

end tan_shifted_value_l116_116171


namespace opposite_of_neg_2023_l116_116889

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116889


namespace mass_percentage_H_in_NH3_is_correct_l116_116506

noncomputable def molar_mass_H : Float := 1.01
noncomputable def molar_mass_N : Float := 14.01
def hydrogen_atoms_in_NH3 : Nat := 3
def nitrogen_atoms_in_NH3 : Nat := 1

def total_mass_H (molar_mass_H : Float) (hydrogen_atoms_in_NH3 : Nat) : Float :=
  hydrogen_atoms_in_NH3 * molar_mass_H

def total_mass_N (molar_mass_N : Float) (nitrogen_atoms_in_NH3 : Nat) : Float :=
  nitrogen_atoms_in_NH3 * molar_mass_N

def molar_mass_NH3 (total_mass_H : Float) (total_mass_N : Float) : Float :=
  total_mass_H + total_mass_N

def mass_percentage_H (total_mass_H : Float) (molar_mass_NH3 : Float) : Float :=
  (total_mass_H / molar_mass_NH3) * 100

theorem mass_percentage_H_in_NH3_is_correct :
  mass_percentage_H (total_mass_H molar_mass_H hydrogen_atoms_in_NH3)
                    (molar_mass_NH3 (total_mass_H molar_mass_H hydrogen_atoms_in_NH3) (total_mass_N molar_mass_N nitrogen_atoms_in_NH3))
  ≈ 17.78 :=
sorry

end mass_percentage_H_in_NH3_is_correct_l116_116506


namespace count_seven_digit_palindromes_l116_116070

theorem count_seven_digit_palindromes : 
  let palindromes := { n : ℤ | ∃ (a b c d : ℕ), 
    1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧ 
    n = 1000000 * a + 100000 * b + 10000 * c + 1000 * d + 100 * c + 10 * b + a } in
  ∃ (count : ℕ), count = 9000 ∧ count = palindromes.card :=
by
  -- the proof goes here
  sorry

end count_seven_digit_palindromes_l116_116070


namespace game_show_possible_guesses_l116_116434

theorem game_show_possible_guesses : 
  (∃ A B C : ℕ, 
    A + B + C = 8 ∧ 
    A > 0 ∧ B > 0 ∧ C > 0 ∧ 
    (A = 1 ∨ A = 4) ∧
    (B = 1 ∨ B = 4) ∧
    (C = 1 ∨ C = 4) ) →
  (number_of_possible_guesses : ℕ) = 210 :=
sorry

end game_show_possible_guesses_l116_116434


namespace distinct_x_intercepts_l116_116138

theorem distinct_x_intercepts : 
  ∃ (s : Finset ℝ), (∀ x ∈ s, (x - 5) * (x ^ 2 + 3 * x + 2) = 0) ∧ s.card = 3 :=
by {
  sorry
}

end distinct_x_intercepts_l116_116138


namespace curve_polar_eqn_max_value_of_OM_l116_116567

noncomputable def curve_parametric_eqn (θ : ℝ) : ℝ × ℝ :=
(-1 + 2 * Real.cos θ, 1 + 2 * Real.sin θ)

def polar_coordinate_eqn (ρ θ : ℝ) : Prop :=
ρ^2 + 2 * ρ * Real.cos θ - 2 * ρ * Real.sin θ - 2 = 0

def max_OM (α : ℝ) : ℝ :=
Real.sqrt 2 * Real.abs (Real.sin (α - Real.pi / 4))

theorem curve_polar_eqn (θ ρ : ℝ) :
  polar_coordinate_eqn ρ θ :=
sorry

theorem max_value_of_OM :
  ∀ α ∈ Icc (0 : ℝ) Real.pi, max_OM α ≤ Real.sqrt 2 ∧ max_OM (3 * Real.pi / 4) = Real.sqrt 2 :=
sorry

end curve_polar_eqn_max_value_of_OM_l116_116567


namespace paul_final_balance_l116_116718

def initial_balance : ℝ := 400
def transfer1 : ℝ := 90
def transfer2 : ℝ := 60
def service_charge_rate : ℝ := 0.02

def service_charge (x : ℝ) : ℝ := service_charge_rate * x

def total_deduction : ℝ := transfer1 + service_charge transfer1 + service_charge transfer2

def final_balance (init_balance : ℝ) (deduction : ℝ) : ℝ := init_balance - deduction

theorem paul_final_balance :
  final_balance initial_balance total_deduction = 307 :=
by
  sorry

end paul_final_balance_l116_116718


namespace Vihaan_more_nephews_than_Alden_l116_116455

theorem Vihaan_more_nephews_than_Alden :
  ∃ (a v : ℕ), (a = 100) ∧ (a + v = 260) ∧ (v - a = 60) := by
  sorry

end Vihaan_more_nephews_than_Alden_l116_116455


namespace opposite_of_neg_2023_l116_116938

theorem opposite_of_neg_2023 : ∃ x : ℤ, -2023 + x = 0 ∧ x = 2023 := 
by 
  use 2023
  constructor
  · exact rfl
  · exact rfl

end opposite_of_neg_2023_l116_116938


namespace simplify_expression_l116_116116

variable (a b : ℤ)

theorem simplify_expression : (a - b) - (3 * (a + b)) - b = a - 8 * b := 
by sorry

end simplify_expression_l116_116116


namespace intersection_A_B_l116_116268

def A := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def B := {x : ℝ | 0 < x ∧ x ≤ 3}

theorem intersection_A_B : (A ∩ B) = {x : ℝ | 0 < x ∧ x ≤ 2} :=
by
  sorry

end intersection_A_B_l116_116268


namespace balloons_ratio_is_two_to_one_l116_116435

variable (total_balloons : ℕ)
variable (blown_first_half_hour : ℕ)
variable (remaining_intact : ℕ)
variable (blown_second_hour : ℕ)

noncomputable def balloons_blown_first_half_hour
: Prop := (blown_first_half_hour = total_balloons / 5)

noncomputable def total_balloons_blown
: Prop := ((total_balloons - remaining_intact) = blown_first_half_hour + blown_second_hour)

noncomputable def ratio
: Prop := (blown_first_half_hour * 2 = blown_second_hour)

theorem balloons_ratio_is_two_to_one
  (h1 : total_balloons = 200)
  (h2 : balloons_blown_first_half_hour = 40)
  (h3 : remaining_intact = 80)
  (h4 : total_balloons_blown)
  (h5 : blown_second_hour = 80)
  :
  ratio :=
by
  sorry

end balloons_ratio_is_two_to_one_l116_116435


namespace Pauline_has_a_winning_strategy_l116_116410

-- Given: p is a prime number and p >= 2
variable (p : ℕ) [Fact (Nat.Prime p)] (hp : p ≥ 2)

-- Indices i from 0 to p - 1
def indices := Finset (Fin p)

-- Values a_i from 0 to 9
variable (a : Fin p → Fin 10)

-- The number M is defined as M = ∑(i=0)^(p-1) a_i * 10^i
def M : ℕ := ∑ i in indices p, (a i).val * 10 ^ (i : ℕ)

-- The goal is to prove that Pauline has a winning strategy to ensure M ≡ 0 [MOD p]
def PaulineWins (p : ℕ) [Fact (Nat.Prime p)] (hp : p ≥ 2) (a : Fin p → Fin 10) : Prop :=
  ∃ Pauline_strategy : Fin p → Fin 10, 
    (∑ i in indices p, (Pauline_strategy i).val * 10 ^ (i : ℕ)) % p = 0

-- Prove the proposition
theorem Pauline_has_a_winning_strategy {p : ℕ} [Fact (Nat.Prime p)] (hp : p ≥ 2) : ∃ (a : Fin p → Fin 10), PaulineWins p hp a :=
sorry

end Pauline_has_a_winning_strategy_l116_116410


namespace opposite_of_neg_2023_l116_116862

-- Definition of the problem
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  -- The proof is to be provided
  sorry

end opposite_of_neg_2023_l116_116862


namespace opposite_of_neg_2023_l116_116917

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by 
  sorry

end opposite_of_neg_2023_l116_116917


namespace problem_1_problem_2_problem_3_problem_4_l116_116622

open Real

-- Problem 1: Prove ∀ ΔABC, A < B → sin A < sin B
theorem problem_1 {A B C a b c : ℝ} (h1 : A+b+C = 180) (h2 : a = 2 * sin A * R) (h3 : b = 2 * sin B * R):
  A < B → sin A < sin B := sorry 

-- Problem 2: Prove that for a triangle with a = 2, and A = 30°, the radius of circumcircle R = 2
theorem problem_2 (a : ℝ) :
  a = 2 → 30 / 180 * π →
  2 / (2 * sin (30 / 180 * π)) = 2 := sorry

-- Problem 3: Prove ∀ ΔABC, a / cos A = b / sin B → A = 45°
theorem problem_3 {A B C a b c : ℝ}  :
  (a / cos A) = (b / sin B) → A = 45 := sorry

-- Problem 4: Prove that for a triangle with A = 30°, a = 4, b = 3, there is one solution for ΔABC
theorem problem_4 (A a b : ℝ) :
  A = 30 → a = 4 → b = 3 →
  ∃! c : ℝ, c > 0 := sorry

end problem_1_problem_2_problem_3_problem_4_l116_116622


namespace angle_A_in_range_l116_116606

variable (a b c : ℝ)
variable (A B C : ℝ)
variable (triangle_ABC : Triangle a b c A B C)

-- Assuming the relevant conditions
axiom longest_side (a > b ∧ a > c)
axiom scalene (a ≠ b ∧ b ≠ c ∧ a ≠ c)
axiom angle_A_range (a^2 < b^2 + c^2 → 60 < A ∧ A < 90)

theorem angle_A_in_range : a^2 < b^2 + c^2 → 60 < A ∧ A < 90 := 
  by
    sorry

end angle_A_in_range_l116_116606


namespace cross_product_correct_l116_116501

noncomputable def vec_u : ℝ × ℝ × ℝ := (3, 4, 2)
noncomputable def vec_v : ℝ × ℝ × ℝ := (1, -2, 5)

theorem cross_product_correct :
  let cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
        (a.2.1 * b.2.2 - a.2.2 * b.2.1,
         a.2.2 * b.1 - a.1 * b.2,
         a.1 * b.2.1 - a.2.1 * b.1)
  in cross_product vec_u vec_v = (24, -13, -10) :=
by
  sorry

end cross_product_correct_l116_116501


namespace Pascal_remaining_distance_l116_116698

theorem Pascal_remaining_distance (D T : ℕ) :
  let current_speed := 8
  let reduced_speed := current_speed - 4
  let increased_speed := current_speed + current_speed / 2
  (D = current_speed * T) →
  (D = reduced_speed * (T + 16)) →
  (D = increased_speed * (T - 16)) →
  D = 256 :=
by
  intros
  sorry

end Pascal_remaining_distance_l116_116698


namespace john_marbles_selection_l116_116629

theorem john_marbles_selection :
  let total_marbles := 15
  let special_colors := 4
  let total_chosen := 5
  let chosen_special_colors := 2
  let remaining_colors := total_marbles - special_colors
  let chosen_normal_colors := total_chosen - chosen_special_colors
  (Nat.choose 4 2) * (Nat.choose 11 3) = 990 :=
by
  sorry

end john_marbles_selection_l116_116629


namespace opposite_neg_2023_l116_116769

theorem opposite_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_neg_2023_l116_116769


namespace alex_needs_additional_coins_l116_116456

theorem alex_needs_additional_coins (friends coins : ℕ) (h_friends : friends = 15) (h_coins : coins = 97) : 
  let required_coins := (friends * (friends + 1)) / 2 in
  required_coins - coins = 23 :=
by
  rw [h_friends, h_coins]
  let required_coins := (15 * 16) / 2
  have h_required_coins : required_coins = 120 := by norm_num
  rw [h_required_coins]
  norm_num
  sorry

end alex_needs_additional_coins_l116_116456


namespace problem_statement_l116_116141

noncomputable def P (x : ℝ) : ℝ := (x^2 + 1) * (x - 2)^9

def polynomial_expansion (x : ℝ) : ℕ → ℝ
| 0     := a₀
| 1     := a₁
| 2     := a₂ * (x - 1)^2
| 3     := a₃ * (x - 1)^3
| 4     := a₄ * (x - 1)^4
| 5     := a₅ * (x - 1)^5
| 6     := a₆ * (x - 1)^6
| 7     := a₇ * (x - 1)^7
| 8     := a₈ * (x - 1)^8
| 9     := a₉ * (x - 1)^9
| 10    := a₁₀ * (x - 1)^10
| 11    := a₁₁ * (x - 1)^11
| _     := 0

theorem problem_statement 
  (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (P 1 = a₀ + a₁ * (1 - 1) + a₂ * (1 - 1)^2 + a₃ * (1 - 1)^3 + a₄ * (1 - 1)^4 + a₅ * (1 - 1)^5 + a₆ * (1 - 1)^6 + a₇ * (1 - 1)^7 + a₈ * (1 - 1)^8 + a₉ * (1 - 1)^9 + a₁₀ * (1 - 1)^10 + a₁₁ * (1 - 1)^11) →
  (P 2 = a₀ + a₁ * (2 - 1) + a₂ * (2 - 1)^2 + a₃ * (2 - 1)^3 + a₄ * (2 - 1)^4 + a₅ * (2 - 1)^5 + a₆ * (2 - 1)^6 + a₇ * (2 - 1)^7 + a₈ * (2 - 1)^8 + a₉ * (2 - 1)^9 + a₁₀ * (2 - 1)^10 + a₁₁ * (2 - 1)^11) →
  (P 0 = a₀ - a₁ * (0 - 1) + a₂ * (0 - 1)^2 - a₃ * (0 - 1)^3 + a₄ * (0 - 1)^4 - a₅ * (0 - 1)^5 + a₆ * (0 - 1)^6 - a₇ * (0 - 1)^7 + a₈ * (0 - 1)^8 - a₉ * (0 - 1)^9 + a₁₀ * (0 - 1)^10 - a₁₁ * (0 - 1)^11) →
  a₀ ≠ 2 ∧
  a₂ + a₄ + a₆ + a₈ + a₁₀ = -254 ∧ 
  a₁₀ ≠ -7 ∧
  a₁ + 2 * a₂ + 3 * a₃ + 4 * a₄ + 5 * a₅ + 6 * a₆ + 7 * a₇ + 8 * a₈ + 9 * a₉ + 10 * a₁₀ + 11 * a₁₁ = 0 :=
sorry

end problem_statement_l116_116141


namespace opposite_of_neg_2023_l116_116874

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116874


namespace tan_add_pi_over_3_l116_116160

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + (Real.pi / 3)) = -(6 * Real.sqrt 3 + 2) / 13 := 
by
  sorry

end tan_add_pi_over_3_l116_116160


namespace vertical_asymptote_l116_116078

theorem vertical_asymptote (x : ℝ) : 
  (∃ x, 4 * x + 5 = 0) → x = -5/4 :=
by 
  sorry

end vertical_asymptote_l116_116078


namespace remaining_distance_l116_116712

variable (D : ℝ) -- remaining distance in miles

-- Current speed
def current_speed := 8.0
-- Increased speed by 50%
def increased_speed := 12.0
-- Reduced speed by 4 mph
def reduced_speed := 4.0

-- Trip time relations
def current_time := D / current_speed
def increased_time := D / increased_speed
def reduced_time := D / reduced_speed

-- Time difference condition
axiom condition : reduced_time = increased_time + 16.0

theorem remaining_distance : D = 96.0 :=
by
  -- proof placeholder
  sorry

end remaining_distance_l116_116712


namespace remaining_savings_after_purchase_l116_116672

-- Definitions of the conditions
def cost_per_sweater : ℕ := 30
def num_sweaters : ℕ := 6
def cost_per_scarf : ℕ := 20
def num_scarves : ℕ := 6
def initial_savings : ℕ := 500

-- Theorem stating the remaining savings
theorem remaining_savings_after_purchase : initial_savings - ((cost_per_sweater * num_sweaters) + (cost_per_scarf * num_scarves)) = 200 :=
by
  -- skipping the proof
  sorry

end remaining_savings_after_purchase_l116_116672


namespace num_mappings_from_A_to_A_l116_116347

open Set Function

-- Define the set A with 2 elements
def A : Set := {0, 1}

-- The theorem stating there are 4 different mappings from set A to set A
theorem num_mappings_from_A_to_A : Fintype.card (A → A) = 4 := sorry

end num_mappings_from_A_to_A_l116_116347


namespace tan_X_l116_116224

noncomputable def triangle_XYZ := 
  {X Y Z : Type} [euclidean_space X] [euclidean_space Y] [euclidean_space Z]
  [XZ_eq_sqrt_41 : XZ = sqrt 41] [YZ_eq_4 : YZ = 4] [angle_Y_90 : angle Y = 90]

theorem tan_X (triangle_XYZ) : 
  ∃ XY, (XY_squared : XY^2 = (XZ^2 - YZ^2)) ∧ (XY = 5) ∧ (tan (angle_const X) = 4/5) :=
by
  sorry

end tan_X_l116_116224


namespace rhombus_area_l116_116325

theorem rhombus_area (d1 d2 : ℝ) (h_d1 : d1 = 11) (h_d2 : d2 = 16) : 
  (d1 * d2) / 2 = 88 :=
by 
  rw [h_d1, h_d2]
  norm_num
  sorry

end rhombus_area_l116_116325


namespace tan_add_pi_over_3_l116_116161

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + (Real.pi / 3)) = -(6 * Real.sqrt 3 + 2) / 13 := 
by
  sorry

end tan_add_pi_over_3_l116_116161


namespace necessary_but_not_sufficient_l116_116547

variables {α β : Type} [Linear Space α]
variable (a : α) -- line a
variable (α₁ : β) -- plane α

-- Let a subset of α₁
axiom line_subset_plane : a ∈ α₁

-- α₁ perpendicular to β is a necessary but not sufficient condition for a perpendicular to β
theorem necessary_but_not_sufficient (h : a ⊥ β) : α₁ ⊥ β := sorry

end necessary_but_not_sufficient_l116_116547


namespace opposite_of_neg_2023_l116_116905

-- Let x be the opposite of -2023
def is_opposite (x : ℤ) (n : ℤ) : Prop := n + x = 0

theorem opposite_of_neg_2023 : ∃ x : ℤ, is_opposite x (-2023) ∧ x = 2023 :=
by
  exists 2023
  unfold is_opposite
  sorry

end opposite_of_neg_2023_l116_116905


namespace smiths_bakery_pies_l116_116304

theorem smiths_bakery_pies (m : ℕ) (h : m = 16) : 
  let s := 4 * m + 6 in 
  s = 70 := 
by
  unfold s
  sorry

end smiths_bakery_pies_l116_116304


namespace radius_solution_l116_116321

theorem radius_solution (n r : ℝ) (h : π * (r + n)^2 = 3 * π * r^2) : r = n * (real.sqrt 3 - 1) / 2 := by
sorry

end radius_solution_l116_116321


namespace increase_in_area_correct_l116_116549

-- Define the initial base and height of the triangle
def initial_base (x : ℝ) : ℝ := 2 * x + 1
def initial_height (x : ℝ) : ℝ := x - 2

-- Define the new base and height after increasing by 5 cm
def new_base (x : ℝ) : ℝ := initial_base x + 5
def new_height (x : ℝ) : ℝ := initial_height x + 5

-- Define the area of a triangle given its base and height
def triangle_area (base height : ℝ) : ℝ := 1 / 2 * base * height

-- Define the initial and new areas of the triangle
def initial_area (x : ℝ) : ℝ := triangle_area (initial_base x) (initial_height x)
def new_area (x : ℝ) : ℝ := triangle_area (new_base x) (new_height x)

-- Define the increase in area
def increase_in_area (x : ℝ) : ℝ := new_area x - initial_area x

-- The main statement that we need to prove
theorem increase_in_area_correct (x : ℝ) : increase_in_area x = 15 / 2 * x + 10 := by
  sorry

end increase_in_area_correct_l116_116549


namespace eccentricity_of_hyperbola_is_two_l116_116566

noncomputable def hyperbola_eccentricity (a b c : ℝ) : Prop :=
  ∀ (P : ℝ × ℝ), 
  (a > 0) ∧ (b = sqrt 3 * a) ∧ 
  (P = (2 * a, 3 * a)) ∧
  (mg_parallel_x_axis : (let M := (P.1, P.2 - 3 * a) in 
                         let G := ((P.1 + F_1.1 + F_2.1) / 3, (P.2 + F_1.2 + F_2.2) / 3) in 
                         M.2 = G.2)) ∧ 
  (|F_1| = (c - a)) ∧ (|F_2| = (c + a))  ∧ 
  (2 * c > 0) →
  let e := sqrt (1 + (b ^ 2 / a ^ 2)) in e = 2

theorem eccentricity_of_hyperbola_is_two (a b c : ℝ) (h : hyperbola_eccentricity a b c) : 
  sqrt (1 + (b ^ 2 / a ^ 2)) = 2 :=
by {
  sorry
}

end eccentricity_of_hyperbola_is_two_l116_116566


namespace opposite_of_neg_2023_l116_116940

theorem opposite_of_neg_2023 : ∃ x : ℤ, -2023 + x = 0 ∧ x = 2023 := 
by 
  use 2023
  constructor
  · exact rfl
  · exact rfl

end opposite_of_neg_2023_l116_116940


namespace swimming_championship_l116_116604

theorem swimming_championship (num_swimmers : ℕ) (lanes : ℕ) (advance : ℕ) (eliminated : ℕ) (total_races : ℕ) : 
  num_swimmers = 300 → 
  lanes = 8 → 
  advance = 2 → 
  eliminated = 6 → 
  total_races = 53 :=
by
  intros
  sorry

end swimming_championship_l116_116604


namespace A_scores_zero_l116_116081

variable (A B C D : Type)
variable (num_points : Type)
variable (competition : Set (A × B))
variable (points : A → num_points)

def chess_competition (A B C D : Type) (points_A points_B points_C points_D : A → num_points) : num_points :=
  let total_points := points_D + points_B + points_C
  num_points - total_points

theorem A_scores_zero (D B C : Type) (points_D : D → nat) (points_B : B → nat) (points_C : C → nat)
  (hD : points_D D = 6) (hB : points_B B = 4) (hC : points_C C = 2) :
  ∑ p in points (A B C D), p = 0 :=
begin
  sorry
end

end A_scores_zero_l116_116081


namespace square_area_l116_116059

theorem square_area (side : ℕ) (h : side = 30) : side * side = 900 := by
  rw [h]
  calc
    30 * 30 = 900 : by norm_num

end square_area_l116_116059


namespace opposite_of_negative_2023_l116_116777

theorem opposite_of_negative_2023 : (- -2023) = 2023 := 
by {
sorry
}

end opposite_of_negative_2023_l116_116777


namespace divide_circle_into_three_equal_parts_l116_116204

-- Define the circle with center O and radius r
structure Circle (O : Type) :=
  (radius : ℝ)
  (center : O)

-- Define points A, B on the circle
structure Point (O : Type) :=
  (x : ℝ)
  (y : ℝ)

-- A line dividing segment AB into three equal parts
structure Line (O : Type) :=
  (A : Point O)
  (B : Point O)
  (divide_into_three : Prop)

-- Define the condition to prove
theorem divide_circle_into_three_equal_parts
  (O : Type)
  (circle : Circle O)
  (A B : Point O)
  (hA : dist circle.center A = circle.radius)
  (hB : dist circle.center B = circle.radius)
  (line : Line O)
  (h_divide : line.divide_into_three) :
  ∃ E F : Point O, 
  dist circle.center E = dist circle.center F ∧
  dist circle.center F = dist circle.center A ∧
  angle_between OE OF = 2π / 3 :=
sorry

end divide_circle_into_three_equal_parts_l116_116204


namespace opposite_of_neg_2023_l116_116944

theorem opposite_of_neg_2023 : ∃ x : ℤ, -2023 + x = 0 ∧ x = 2023 := 
by 
  use 2023
  constructor
  · exact rfl
  · exact rfl

end opposite_of_neg_2023_l116_116944


namespace part1_part2_l116_116561

def f (x : ℝ) : ℝ := 2 * |x - 1| - |x + 2|

theorem part1 (x : ℝ) : f(x) ≤ 6 → x ∈ set.Icc (-2) 10 :=
by
  sorry

theorem part2 (m : ℝ) : (∀ x : ℝ, f(x) ≥ m) → m ≤ -3 :=
by
  sorry

end part1_part2_l116_116561


namespace graph_passes_through_point_l116_116104

theorem graph_passes_through_point (a : ℝ) (h_pos : 0 < a) (h_neq : a ≠ 1) :
  ∃ x y, (x, y) = (0, 3) ∧ (∀ f : ℝ → ℝ, (∀ y, (f y = a ^ y) → (0, f 0 + 2) = (0, 3))) :=
by
  sorry

end graph_passes_through_point_l116_116104


namespace tan_add_pi_div_three_l116_116157

theorem tan_add_pi_div_three (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = -((6 + 5 * Real.sqrt 3) / 13) := 
by 
  sorry

end tan_add_pi_div_three_l116_116157


namespace opposite_of_neg_2023_l116_116860

-- Definition of the problem
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  -- The proof is to be provided
  sorry

end opposite_of_neg_2023_l116_116860


namespace maria_savings_l116_116677

-- Conditions
def sweater_cost : ℕ := 30
def scarf_cost : ℕ := 20
def num_sweaters : ℕ := 6
def num_scarves : ℕ := 6
def savings : ℕ := 500

-- The proof statement
theorem maria_savings : savings - (num_sweaters * sweater_cost + num_scarves * scarf_cost) = 200 :=
by
  sorry

end maria_savings_l116_116677


namespace value_of_a_pow_2023_plus_b_pow_2023_l116_116182

theorem value_of_a_pow_2023_plus_b_pow_2023 (a b : ℤ) (h : {a^2, 0, -1} = {a, b, 0}) : a^2023 + b^2023 = 0 :=
sorry

end value_of_a_pow_2023_plus_b_pow_2023_l116_116182


namespace boxes_used_l116_116430

-- Define the given conditions
def oranges_per_box : ℕ := 10
def total_oranges : ℕ := 2650

-- Define the proof statement
theorem boxes_used : total_oranges / oranges_per_box = 265 :=
by
  -- Proof goes here
  sorry

end boxes_used_l116_116430


namespace find_n_if_roots_opposite_signs_l116_116045

theorem find_n_if_roots_opposite_signs :
  ∃ n : ℝ, (∀ x : ℝ, (x^2 + (n-2)*x) / (2*n*x - 4) = (n+1) / (n-1) → x = -x) →
    (n = (-1 + Real.sqrt 5) / 2 ∨ n = (-1 - Real.sqrt 5) / 2) :=
by
  sorry

end find_n_if_roots_opposite_signs_l116_116045


namespace number_of_children_l116_116484

theorem number_of_children (h1 : ∀ n : ℕ, ∀ k : ℕ, 6 * k = 12 → k = 2) : true :=
by {
  let k := 2,
  have k_def : 6 * k = 12 := by norm_num,
  exact h1 6 k k_def
}

end number_of_children_l116_116484


namespace tan_shifted_value_l116_116167

theorem tan_shifted_value (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = -((6 + 5 * Real.sqrt 3) / 13) :=
by
  sorry

end tan_shifted_value_l116_116167


namespace min_value_of_one_over_a_plus_one_over_b_l116_116649

theorem min_value_of_one_over_a_plus_one_over_b (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 12) :
  (∃ c : ℝ, (c = 1/ a + 1 / b) ∧ c = 1 / 3) :=
sorry

end min_value_of_one_over_a_plus_one_over_b_l116_116649


namespace opposite_of_neg_2023_l116_116894

-- Let x be the opposite of -2023
def is_opposite (x : ℤ) (n : ℤ) : Prop := n + x = 0

theorem opposite_of_neg_2023 : ∃ x : ℤ, is_opposite x (-2023) ∧ x = 2023 :=
by
  exists 2023
  unfold is_opposite
  sorry

end opposite_of_neg_2023_l116_116894


namespace fraction_a_over_b_l116_116586

theorem fraction_a_over_b (a b : ℝ) (h : 2 * a = 3 * b) : a / b = 3 / 2 := by
  sorry

end fraction_a_over_b_l116_116586


namespace no_perfect_squares_l116_116082

theorem no_perfect_squares (x y z t : ℕ) (h1 : xy - zt = k) (h2 : x + y = k) (h3 : z + t = k) :
  ¬ (∃ m n : ℕ, x * y = m^2 ∧ z * t = n^2) := by
  sorry

end no_perfect_squares_l116_116082


namespace john_marbles_selection_l116_116631

theorem john_marbles_selection :
  let total_marbles := 15
  let special_colors := 4
  let total_chosen := 5
  let chosen_special_colors := 2
  let remaining_colors := total_marbles - special_colors
  let chosen_normal_colors := total_chosen - chosen_special_colors
  (Nat.choose 4 2) * (Nat.choose 11 3) = 990 :=
by
  sorry

end john_marbles_selection_l116_116631


namespace angle_B_possibilities_l116_116252

theorem angle_B_possibilities (O H : Point) (A B C : Point) (R b : ℝ)
  (circumcenter : is_circumcenter O A B C)
  (orthocenter : is_orthocenter H A B C)
  (BO_eq_BH : dist B O = dist B H) :
  ∠BAC = 60 ∨ ∠BAC = 120 :=
sorry

end angle_B_possibilities_l116_116252


namespace sum_F_2_n_l116_116413

theorem sum_F_2_n :
  ( ∀ n : ℕ, (n = 0 → F n = 0) ∧ (n = 1 → F n = 3 / 2) ∧ (n ≥ 2 → F n = (5 / 2) * F (n - 1) - F (n - 2)) ) →
  ∑ n : ℕ, 1 / F (2 ^ n) = 1 :=
sorry

end sum_F_2_n_l116_116413


namespace santa_chocolate_candies_l116_116730

theorem santa_chocolate_candies (C M : ℕ) (h₁ : C + M = 2023) (h₂ : C = 3 * M / 4) : C = 867 :=
sorry

end santa_chocolate_candies_l116_116730


namespace math_problem_l116_116102

theorem math_problem (a b : ℝ) (h : (a + 2 * complex.I) / complex.I = b + complex.I) : a + b = 1 :=
sorry

end math_problem_l116_116102


namespace tan_add_pi_over_3_l116_116176

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = - (6 + 5 * Real.sqrt 3) / 13 :=
sorry

end tan_add_pi_over_3_l116_116176


namespace abs_eq_of_fraction_sum_is_int_l116_116626

theorem abs_eq_of_fraction_sum_is_int (a b c : ℤ) 
  (h1 : (a : ℚ) / b + (b : ℚ) / c + (c : ℚ) / a ∈ ℤ)
  (h2 : (a : ℚ) / c + (c : ℚ) / b + (b : ℚ) / a ∈ ℤ) :
  |a| = |b| ∧ |b| = |c| := 
sorry

end abs_eq_of_fraction_sum_is_int_l116_116626


namespace cosine_angle_l116_116575

noncomputable def vec_a : ℝ × ℝ := (3, 1)
noncomputable def vec_b : ℝ × ℝ := (2, 2)

def vec_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def vec_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def norm (u : ℝ × ℝ) : ℝ := Real.sqrt (u.1^2 + u.2^2)

theorem cosine_angle :
  let a := vec_a 
  let b := vec_b 
  let u := vec_add a b
  let v := vec_sub a b
  in Real.cos (dot_product u v) / (norm u * norm v) = Real.sqrt 17 / 17 := by
  sorry

end cosine_angle_l116_116575


namespace compound_statement_logic_l116_116556

variables p q : Prop

theorem compound_statement_logic (h1 : p ∨ q) (h2 : ¬ (p ∧ q)) : 
  (q ↔ ¬ p) ∧ (p ↔ ¬ q) :=
  sorry

end compound_statement_logic_l116_116556


namespace num_seven_digit_palindromes_l116_116065

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def is_seven_digit_palindrome (x : ℕ) : Prop :=
  let a := x / 1000000 % 10 in
  let b := x / 100000 % 10 in
  let c := x / 10000 % 10 in
  let d := x / 1000 % 10 in
  let e := x / 100 % 10 in
  let f := x / 10 % 10 in
  let g := x % 10 in
  a ≠ 0 ∧ a = g ∧ b = f ∧ c = e

theorem num_seven_digit_palindromes : 
  ∃ n : ℕ, (∀ x : ℕ, is_seven_digit_palindrome x → 1 ≤ a (x / 1000000 % 10) ∧ is_digit (b) ∧ is_digit (c) ∧ is_digit (d)) → n = 9000 :=
sorry

end num_seven_digit_palindromes_l116_116065


namespace cosine_theorem_a_cosine_theorem_b_cosine_theorem_c_l116_116034

theorem cosine_theorem_a (a b c A : ℝ) :
  a^2 = b^2 + c^2 - 2 * b * c * Real.cos A := sorry

theorem cosine_theorem_b (a b c B : ℝ) :
  b^2 = a^2 + c^2 - 2 * a * c * Real.cos B := sorry

theorem cosine_theorem_c (a b c C : ℝ) :
  c^2 = a^2 + b^2 - 2 * a * b * Real.cos C := sorry

end cosine_theorem_a_cosine_theorem_b_cosine_theorem_c_l116_116034


namespace opposite_of_negative_2023_l116_116795

theorem opposite_of_negative_2023 : (- -2023) = 2023 := 
by {
sorry
}

end opposite_of_negative_2023_l116_116795


namespace combined_area_of_shapes_eq_5_5_l116_116725

-- Defining the vertices of the quadrilateral and triangle
def quad_vertices : List (ℚ × ℚ) := [(0, 1), (2, 3), (4, 0), (3, 0)]
def tri_vertices : List (ℚ × ℚ) := [(0, 1), (1, 2), (2, 3)]

-- Function to calculate the shoelace formula for the area of a polygon
def shoelace_formula (vertices : List (ℚ × ℚ)) : ℚ :=
  (0.5 : ℚ) * (abs (vertices.zip (vertices.tail ++ [vertices.head])
    |> List.sum (λ ⟨(x1, y1), (x2, y2)⟩ => x1 * y2 - y1 * x2)))

-- Definitions of the areas using the shoelace formula
def quad_area : ℚ := shoelace_formula quad_vertices
def tri_area : ℚ := shoelace_formula tri_vertices
def total_area : ℚ := quad_area + tri_area

theorem combined_area_of_shapes_eq_5_5 :
  total_area = 5.5 := by
  sorry

end combined_area_of_shapes_eq_5_5_l116_116725


namespace probability_of_king_then_queen_l116_116441

noncomputable def probability_top_king_second_queen : ℚ := 4 / 663

theorem probability_of_king_then_queen :
  let deck := (finset.range 52).image (λ i, (i / 13, i % 13)) in
  let shuffled_deck := deck.to_list in
  (∃ shuffled_deck : list (ℕ × ℕ), shuffled_deck = list.permutations deck.to_list) →
  let top_two := shuffled_deck.take 2 in
  (top_two.head = ⟨3, 12⟩ ∧ top_two.tail.head = ⟨2, 11⟩) →
  probability_top_king_second_queen = 4 / 663 :=
begin
  sorry

end probability_of_king_then_queen_l116_116441


namespace inequality_solution_l116_116349

theorem inequality_solution (x : ℝ) : x * |x| ≤ 1 ↔ x ≤ 1 := 
sorry

end inequality_solution_l116_116349


namespace triangle_sine_ratio_l116_116621

-- Define basic facts and assumptions about the triangles and angles
variables {P Q R S : Type}
variables {A : Type} [Real.linearOrderedField A]
variables {sin : A → A}
variables {QR PS QS RS : A}
variables {α β γ : A}

-- Assume angle values and point division
def angle_Q : A := 75 -- degrees
def angle_R : A := 30 -- degrees
def angle_S_division (QR : A) : (A × A) := (2/3 * QR, 1/3 * QR)

-- Use the conditions to state the problem
def problem (sin : A → A) (QR PS : A) : Prop :=
  let (QS, RS) := angle_S_division QR in
  let angle_PAQ := QS * sin 75 / PS in
  let angle_PAR := 2 * RS / PS in
  sin angle_PAQ / sin angle_PAR = (sqrt 6 + sqrt 2) / 4

-- The theorem to prove
theorem triangle_sine_ratio
  (sin : A → A) [is_real_sine : ∀ x : A, sin x = has_sin.sin x]
  (QR PS : A) : problem sin QR PS :=
  sorry

end triangle_sine_ratio_l116_116621


namespace tan_shifted_value_l116_116168

theorem tan_shifted_value (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = -((6 + 5 * Real.sqrt 3) / 13) :=
by
  sorry

end tan_shifted_value_l116_116168


namespace option_d_incorrect_l116_116540

  -- Definitions and conditions
  variable {a d : ℝ} -- a is the first term and d is the common difference
  def S (n : ℕ) : ℝ := n * (2 * a + (n - 1) * d) / 2 -- Sum of the first n terms of an arithmetic sequence

  -- Conditions
  variable {n : ℕ}
  axiom S5_lt_S6 : S 5 < S 6
  axiom S6_eq_S7_gt_S8 : S 6 = S 7 ∧ S 7 > S 8

  -- Translating the problem statement to a theorem in Lean
  theorem option_d_incorrect : S 9 ≤ S 5 :=
  sorry
  
end option_d_incorrect_l116_116540


namespace trains_clear_time_l116_116408

/-- In what time will the two trains be clear of each other given their lengths and speeds? -/
theorem trains_clear_time :
  ∀ (length1 length2 : ℕ) (speed1_kmh speed2_kmh : ℕ),
  length1 = 140 →
  length2 = 280 →
  speed1_kmh = 42 →
  speed2_kmh = 30 →
  let speed1_ms := speed1_kmh * 1000 / 3600,
      speed2_ms := speed2_kmh * 1000 / 3600,
      relative_speed := speed1_ms + speed2_ms,
      combined_length := length1 + length2
  in (combined_length / relative_speed = 21) := by
    intros length1 length2 speed1_kmh speed2_kmh h_length1 h_length2 h_speed1 h_speed2
    let speed1_ms : ℚ := speed1_kmh * 1000 / 3600
    let speed2_ms : ℚ := speed2_kmh * 1000 / 3600
    let relative_speed : ℚ := speed1_ms + speed2_ms
    let combined_length : ℚ := length1 + length2
    sorry

end trains_clear_time_l116_116408


namespace transformed_sample_variance_l116_116197

def sample_variance (s : List ℝ) : ℝ :=
(s.foldl (+) 0) / (s.length : ℝ)

theorem transformed_sample_variance {a1 a2 a3 : ℝ}
  (h : sample_variance [(a1 - (sample_variance [a1, a2, a3]))^2, 
                       (a2 - (sample_variance [a1, a2, a3]))^2, 
                       (a3 - (sample_variance [a1, a2, a3]))^2] = 2) :
  sample_variance [(2 * a1 + 3 - (sample_variance [2 * a1 + 3, 2 * a2 + 3, 2 * a3 + 3]))^2, 
                  (2 * a2 + 3 - (sample_variance [2 * a1 + 3, 2 * a2 + 3, 2 * a3 + 3]))^2, 
                  (2 * a3 + 3 - (sample_variance [2 * a1 + 3, 2 * a2 + 3, 2 * a3 + 3]))^2] = 8 := by
  sorry

end transformed_sample_variance_l116_116197


namespace triangle_problem_DP_equals_l116_116687

-- Define the triangle ABC with specific lengths
structure Triangle :=
(A B C : Point)
(AK BK KC : ℝ)
(D : Point) -- Midpoint of AB
(P : Point) -- Intersection point
(angles_eq : ∠APB = ∠BAC)
(CP_gt_CD : CP > CD)

-- The proof problem to show DP = (-21 + 12 * sqrt 89) / (2 * sqrt 55)
theorem triangle_problem_DP_equals {T : Triangle} :
  CP T > CD T → ∠APB T = ∠BAC T →
  DP T = (-21 + 12 * real.sqrt 89) / (2 * real.sqrt 55) := 
  sorry

end triangle_problem_DP_equals_l116_116687


namespace opposite_of_neg_2023_l116_116942

theorem opposite_of_neg_2023 : ∃ x : ℤ, -2023 + x = 0 ∧ x = 2023 := 
by 
  use 2023
  constructor
  · exact rfl
  · exact rfl

end opposite_of_neg_2023_l116_116942


namespace chessboard_coloring_count_l116_116607

def chessboard_size : ℕ := 8
def number_of_colors : ℕ := 8

/-
Define a function that checks if a given coloring of the chessboard is valid.
The function verifies that neighboring squares have different colors and each row contains all colors.
-/
def is_valid_coloring (coloring : ℕ → ℕ → ℕ) : Prop :=
  (∀ i j, i < chessboard_size ∧ j < chessboard_size →
    (if j + 1 < chessboard_size then coloring i j ≠ coloring i (j + 1) else True) ∧
    (if i + 1 < chessboard_size then coloring i j ≠ coloring (i + 1) j else True)) ∧
  (∀ i, i < chessboard_size → ∃ perm : List ℕ, perm.perm (List.range number_of_colors) ∧
    ∀ j, j < chessboard_size → coloring i j = perm.get j)

theorem chessboard_coloring_count :
  ∃ (count : ℕ), count = factorial 8 * 14833^7 :=
sorry

end chessboard_coloring_count_l116_116607


namespace price_reduction_to_achieve_profit_l116_116422

/-- 
A certain store sells clothing that cost $45$ yuan each to purchase for $65$ yuan each.
On average, they can sell $30$ pieces per day. For each $1$ yuan price reduction, 
an additional $5$ pieces can be sold per day. Given these conditions, 
prove that to achieve a daily profit of $800$ yuan, 
the price must be reduced by $10$ yuan per piece.
-/
theorem price_reduction_to_achieve_profit :
  ∃ x : ℝ, x = 10 ∧
    let original_cost := 45
    let original_price := 65
    let original_pieces_sold := 30
    let additional_pieces_per_yuan := 5
    let target_profit := 800
    let new_profit_per_piece := (original_price - original_cost) - x
    let new_pieces_sold := original_pieces_sold + additional_pieces_per_yuan * x
    new_profit_per_piece * new_pieces_sold = target_profit :=
by {
  sorry
}

end price_reduction_to_achieve_profit_l116_116422


namespace athletes_middle_legs_athletes_adjacent_legs_l116_116731

theorem athletes_middle_legs (A B C D E F : Type) :
  let athletes := [A, B, C, D, E, F]
  let middle_positions := 2
  let remaining_positions := 2
  (∃ legs : list athletes, legs = [A, B, ?x, ?y] ∧
    x ≠ A ∧ x ≠ B ∧ y ≠ A ∧ y ≠ B ∧
    (legs.permutations.count = (math.combinatorics.perm (middle_positions) * math.combinatorics.perm (remaining_positions))) = 24 :=
sorry

theorem athletes_adjacent_legs (A B C D E F : Type) :
  let athletes := [A, B, C, D, E, F]
  let adjacent_positions := 2
  let remaining_positions_count := 2
  let perm_remaining := 3
  (∃ legs : list athletes, legs = [A, B, ?x, ?y] ∧
    x ≠ A ∧ x ≠ B ∧ y ≠ A ∧ y ≠ B ∧
    (legs.permutations.count = (math.combinatorics.perm (adjacent_positions) * (math.combinatorics.choose _ (remaining_positions_count)) * math.combinatorics.perm (perm_remaining))) = 72 :=
sorry

end athletes_middle_legs_athletes_adjacent_legs_l116_116731


namespace largest_binomial_coefficient_in_4th_and_5th_implies_n_eq_7_l116_116745

theorem largest_binomial_coefficient_in_4th_and_5th_implies_n_eq_7
  (n : ℕ) (h : ∀ k, k ≠ 4 → k ≠ 5 → binomial n k < binomial n 4 ∨ binomial n k < binomial n 5) :
  n = 7 :=
sorry

end largest_binomial_coefficient_in_4th_and_5th_implies_n_eq_7_l116_116745


namespace sum_of_digits_at_positions_2022_2023_2024_l116_116026

def seq (n : ℕ) : ℕ := 
  if n % 6 = 0 then 6 else n % 6

def erase_every_second (seq_list : List ℕ) : List ℕ :=
  seq_list.toList.filterWithIndex (λ idx _ => (idx + 1) % 2 ≠ 0)

def erase_every_third (seq_list : List ℕ) : List ℕ :=
  seq_list.toList.filterWithIndex (λ idx _ => (idx + 1) % 3 ≠ 0)

def erase_every_fourth (seq_list : List ℕ) : List ℕ :=
  seq_list.toList.filterWithIndex (λ idx _ => (idx + 1) % 4 ≠ 0)

theorem sum_of_digits_at_positions_2022_2023_2024 :
  let n := 12000
  let initial_seq := List.range n |>.map seq
  let seq_after_first_erasure := erase_every_second initial_seq
  let seq_after_second_erasure := erase_every_third seq_after_first_erasure
  let final_seq := erase_every_fourth seq_after_second_erasure
  final_seq.nthLe (2022 - 1)  sorry + final_seq.nthLe (2023 - 1) sorry + final_seq.nthLe (2024 - 1) sorry = 5 :=
sorry

end sum_of_digits_at_positions_2022_2023_2024_l116_116026


namespace tom_finishes_in_6_years_l116_116363

/-- Combined program years for BS and Ph.D. -/
def BS_years : ℕ := 3
def PhD_years : ℕ := 5

/-- Total combined program time -/
def total_program_years : ℕ := BS_years + PhD_years

/-- Tom's time multiplier -/
def tom_time_multiplier : ℚ := 3 / 4

/-- Tom's total time to finish the program -/
def tom_total_time : ℚ := tom_time_multiplier * total_program_years

theorem tom_finishes_in_6_years : tom_total_time = 6 := 
by 
  -- implementation of the proof is to be filled in here
  sorry

end tom_finishes_in_6_years_l116_116363


namespace find_p_l116_116480

variable (a b c p : ℚ)

theorem find_p (h1 : 5 / (a + b) = p / (a + c)) (h2 : p / (a + c) = 8 / (c - b)) : p = 13 := by
  sorry

end find_p_l116_116480


namespace cost_of_items_l116_116690

theorem cost_of_items {x y z : ℕ} (h1 : x + 3 * y + 2 * z = 98)
                      (h2 : 3 * x + y = 5 * z - 36)
                      (even_x : x % 2 = 0) :
  x = 4 ∧ y = 22 ∧ z = 14 := 
by
  sorry

end cost_of_items_l116_116690


namespace intersection_M_N_l116_116101

def setM : Set ℝ := {x | x^2 - 1 ≤ 0}
def setN : Set ℝ := {x | x^2 - 3 * x > 0}

theorem intersection_M_N :
  {x | -1 ≤ x ∧ x < 0} = setM ∩ setN :=
by
  sorry

end intersection_M_N_l116_116101


namespace cosine_angle_l116_116574

noncomputable def vec_a : ℝ × ℝ := (3, 1)
noncomputable def vec_b : ℝ × ℝ := (2, 2)

def vec_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def vec_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def norm (u : ℝ × ℝ) : ℝ := Real.sqrt (u.1^2 + u.2^2)

theorem cosine_angle :
  let a := vec_a 
  let b := vec_b 
  let u := vec_add a b
  let v := vec_sub a b
  in Real.cos (dot_product u v) / (norm u * norm v) = Real.sqrt 17 / 17 := by
  sorry

end cosine_angle_l116_116574


namespace cost_price_watch_l116_116402

variable (cost_price : ℚ)

-- Conditions
def sold_at_loss (cost_price : ℚ) := 0.90 * cost_price
def sold_at_gain (cost_price : ℚ) := 1.03 * cost_price
def price_difference (cost_price : ℚ) := sold_at_gain cost_price - sold_at_loss cost_price = 140

-- Theorem
theorem cost_price_watch (h : price_difference cost_price) : cost_price = 1076.92 := by
  sorry

end cost_price_watch_l116_116402


namespace sum_gt_four_probability_l116_116605

open Finset

theorem sum_gt_four_probability :
  let S := {1, 2, 3, 4}
  let pairs := (S.powerset.filter (λ s, s.card = 2)).val
  let favorable_pairs := pairs.filter (λ s, s.sum id > 4)
  let probability := favorable_pairs.card / pairs.card
  probability = (2 : ℚ) / 3 :=
sorry

end sum_gt_four_probability_l116_116605


namespace sequence_sum_diff_l116_116466

theorem sequence_sum_diff :
  let seqA := list.range' 2 ((2021 - 2) / 3 + 1) (3 : ℤ)
  let seqB := list.range' 3 ((2022 - 3) / 3 + 1) (3 : ℤ)
  list.sum seqA - list.sum seqB = -544 := by
sorry

end sequence_sum_diff_l116_116466


namespace minimum_dot_product_l116_116211

noncomputable def min_AE_dot_AF : ℝ :=
  let AB : ℝ := 2
  let BC : ℝ := 1
  let AD : ℝ := 1
  let CD : ℝ := 1
  let angle_ABC : ℝ := 60 -- this is 60 degrees, which should be converted to radians if we need to use it
  sorry

theorem minimum_dot_product :
  let AB : ℝ := 2
  let BC : ℝ := 1
  let AD : ℝ := 1
  let CD : ℝ := 1
  let angle_ABC : ℝ := 60
  ∃ (E F : ℝ), (min_AE_dot_AF = 29 / 18) :=
    sorry

end minimum_dot_product_l116_116211


namespace pascal_remaining_miles_l116_116702

section PascalCyclingTrip

variable (D T : ℝ)
variable (V : ℝ := 8)
variable (V_reduced V_increased : ℝ)
variable (time_diff : ℝ := 16)

-- Definitions of reduced and increased speeds
def V_reduced := V - 4
def V_increased := V + 0.5 * V

-- Conditions for the distance
def distance_current := D = V * T
def distance_increased := D = V_increased * (T - time_diff)
def distance_reduced := D = V_reduced * (T + time_diff)

theorem pascal_remaining_miles (h1 : distance_current)
  (h2 : distance_increased)
  (h3 : distance_reduced) : D = 256 := 
sorry

end PascalCyclingTrip

end pascal_remaining_miles_l116_116702


namespace paul_account_balance_l116_116719

variable (initial_balance : ℝ) (transfer1 : ℝ) (transfer2 : ℝ) (service_charge_rate : ℝ)

def final_balance (init_bal transfer1 transfer2 rate : ℝ) : ℝ :=
  let charge1 := transfer1 * rate
  let total_deduction := transfer1 + charge1
  init_bal - total_deduction

theorem paul_account_balance :
  initial_balance = 400 →
  transfer1 = 90 →
  transfer2 = 60 →
  service_charge_rate = 0.02 →
  final_balance 400 90 60 0.02 = 308.2 :=
by
  intros h1 h2 h3 h4
  rw [final_balance, h1, h2, h4]
  norm_num

end paul_account_balance_l116_116719


namespace circumcircle_radius_ABC_l116_116370

noncomputable def radius_of_circumcircle_of_ABC
  (R1 R2 : ℝ)
  (h_sum : R1 + R2 = 11)
  (d : ℝ := 5 * Real.sqrt 17)
  (h_dist : 5 * Real.sqrt 17 = d)
  (A : ℝ := 8)
  (h_tangent : ∀ {R3 : ℝ}, R3 = A → R3 = 8)
  : ℝ :=
2 * Real.sqrt 19

theorem circumcircle_radius_ABC (R1 R2 d A : ℝ)
  (h_sum : R1 + R2 = 11)
  (h_dist : d = 5 * Real.sqrt 17)
  (h_tangent : A = 8)
  : radius_of_circumcircle_of_ABC R1 R2 h_sum d h_dist A h_tangent = 2 * Real.sqrt 19 := by
  sorry

end circumcircle_radius_ABC_l116_116370


namespace opposite_of_neg_2023_l116_116982

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l116_116982


namespace pascal_remaining_miles_l116_116705

section PascalCyclingTrip

variable (D T : ℝ)
variable (V : ℝ := 8)
variable (V_reduced V_increased : ℝ)
variable (time_diff : ℝ := 16)

-- Definitions of reduced and increased speeds
def V_reduced := V - 4
def V_increased := V + 0.5 * V

-- Conditions for the distance
def distance_current := D = V * T
def distance_increased := D = V_increased * (T - time_diff)
def distance_reduced := D = V_reduced * (T + time_diff)

theorem pascal_remaining_miles (h1 : distance_current)
  (h2 : distance_increased)
  (h3 : distance_reduced) : D = 256 := 
sorry

end PascalCyclingTrip

end pascal_remaining_miles_l116_116705


namespace circumscribed_square_ratio_l116_116545

theorem circumscribed_square_ratio (A B C D P : Point) 
  (h_square : is_square A B C D) 
  (h_circumcircle : is_on_circumcircle P A D) 
  : (PA + PC) / PB = Real.sqrt 2 := 
sorry

end circumscribed_square_ratio_l116_116545


namespace find_f_16_l116_116568

variable (α : ℝ)

def f (x : ℝ) : ℝ := x ^ α

theorem find_f_16 (h1 : f α 2 = Real.sqrt 2) : f α 16 = 4 :=
by
  sorry

end find_f_16_l116_116568


namespace opposite_of_neg_2023_l116_116881

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116881


namespace sin_cos_75_eq_quarter_l116_116044

theorem sin_cos_75_eq_quarter : (Real.sin (75 * Real.pi / 180)) * (Real.cos (75 * Real.pi / 180)) = 1 / 4 :=
by
  sorry

end sin_cos_75_eq_quarter_l116_116044


namespace opposite_of_neg2023_l116_116995

theorem opposite_of_neg2023 : ∀ (x : Int), -2023 + x = 0 → x = 2023 :=
by
  intros x h
  sorry

end opposite_of_neg2023_l116_116995


namespace opposite_of_neg_2023_l116_116859

-- Definition of the problem
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  -- The proof is to be provided
  sorry

end opposite_of_neg_2023_l116_116859


namespace Smiths_Backery_Pies_l116_116302

theorem Smiths_Backery_Pies : 
  ∀ (p : ℕ), (∃ (q : ℕ), q = 16 ∧ p = 4 * q + 6) → p = 70 :=
by
  intros p h
  cases' h with q hq
  cases' hq with hq1 hq2
  rw [hq1] at hq2
  have h_eq : p = 4 * 16 + 6 := hq2
  norm_num at h_eq
  exact h_eq

end Smiths_Backery_Pies_l116_116302


namespace trapezoid_concurrency_l116_116286

noncomputable def isosceles_trapezoid (A B C D : Type) := ∃ M : Type, midpoint AD M

noncomputable def bicect_angle_intersects_circumcircle (B D ω K : Type) := ∃ ABC : Type, angle_bisector_intersection B D ABC ω K

noncomputable def meets_circumcircle_again (C M N ω : Type) := ∃ ABCD : Type, meets_again C M ABCD ω N

noncomputable def tangents_drawn_to_circle (B P Q KMN : Type) := ∃ ABCD_MK : Type, tangents_drawn_to_circle B P Q KMN ABCD_MK

noncomputable def are_lines_concurrent (B K M N P Q : Type) := ∃ BP_BQ_KMN : Type, lines_concurrent B K M N P Q BP_BQ_KMN

theorem trapezoid_concurrency (A B C D M ω K P Q N : Type)
  (h1 : isosceles_trapezoid A B C D)
  (h2 : bicect_angle_intersects_circumcircle B D ω K)
  (h3 : meets_circumcircle_again C M N ω)
  (h4 : tangents_drawn_to_circle B P Q KMN) :
  are_lines_concurrent B K M N P Q := sorry

end trapezoid_concurrency_l116_116286


namespace opposite_of_neg_2023_l116_116980

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l116_116980


namespace state_B_more_candidates_l116_116601

theorem state_B_more_candidates (appeared : ℕ) (selected_A_pct selected_B_pct : ℝ)
  (h1 : appeared = 8000)
  (h2 : selected_A_pct = 0.06)
  (h3 : selected_B_pct = 0.07) :
  (selected_B_pct * appeared - selected_A_pct * appeared = 80) :=
by
  sorry

end state_B_more_candidates_l116_116601


namespace opposite_of_neg_2023_l116_116963

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l116_116963


namespace milk_production_l116_116740

theorem milk_production 
  (initial_cows : ℕ)
  (initial_milk : ℕ)
  (initial_days : ℕ)
  (max_milk_per_cow_per_day : ℕ)
  (available_cows : ℕ)
  (days : ℕ)
  (H_initial : initial_cows = 10)
  (H_initial_milk : initial_milk = 40)
  (H_initial_days : initial_days = 5)
  (H_max_milk : max_milk_per_cow_per_day = 2)
  (H_available_cows : available_cows = 15)
  (H_days : days = 8) :
  available_cows * initial_milk / (initial_cows * initial_days) * days = 96 := 
by 
  sorry

end milk_production_l116_116740


namespace range_of_b_l116_116751

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≥ a then x^2 - 2*a*x + 1 else -(x-1)^2

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f(-x) = -f(x)

theorem range_of_b (a : ℝ) (b : ℝ) :
  (∀ x : ℝ, x ≥ a → f a x = x^2 - 2*a*x + 1) →
  is_odd_function (λ x, f a (x^3 + a)) →
  (∀ x : ℝ, x ∈ set.Icc (b-1) (b+2) → f a (b*x) ≥ 4 * f a (x + 1)) →
  b ∈ set.Iic (-real.sqrt 5) ∪ set.Ici ((3 + real.sqrt 5) / 2) :=
sorry

end range_of_b_l116_116751


namespace archibald_percentage_games_won_l116_116009

theorem archibald_percentage_games_won
  (A B F1 F2 : ℝ) -- number of games won by Archibald, his brother, and his two friends
  (total_games : ℝ)
  (A_eq_1_1B : A = 1.1 * B)
  (F_eq_2_1B : F1 + F2 = 2.1 * B)
  (total_games_eq : A + B + F1 + F2 = total_games)
  (total_games_val : total_games = 280) :
  (A / total_games * 100) = 26.19 :=
by
  sorry

end archibald_percentage_games_won_l116_116009


namespace range_of_t_for_point_in_upper_left_side_l116_116620

def point_in_upper_left_side_condition (x y : ℝ) : Prop :=
  x - y + 4 < 0

theorem range_of_t_for_point_in_upper_left_side :
  ∀ t : ℝ, point_in_upper_left_side_condition (-2) t ↔ t > 2 :=
by
  intros t
  unfold point_in_upper_left_side_condition
  simp
  sorry

end range_of_t_for_point_in_upper_left_side_l116_116620


namespace log_expression_simplify_l116_116558

variable (x y z w v t : ℝ)
hypothesis (hx : x > 0)
hypothesis (hy : y > 0)
hypothesis (hz : z > 0)
hypothesis (hw : w > 0)
hypothesis (hv : v > 0)
hypothesis (ht : t > 0)

theorem log_expression_simplify :
  log (x / z) + log (z / y) + log (y / w) - log (x * v / (w * t)) = log (t / v) := 
by sorry

end log_expression_simplify_l116_116558


namespace length_FD_closest_value_l116_116246

variable (A B C D E F : Point)
variable (AB CD : LineSegment)
variable (parallelogram : isParallelogram A B C D)
variable (angle_BAD : ∠ BAD = 150 * degree)
variable (length_AB : length AB = 20)
variable (length_BC : length BC = 12)
variable (equals_AD_BC : length AD = length BC)
variable (line_CD_extended_E : extendsLine CD E)
variable (length_DE : length DE = 6)
variable (intersection_BE_AD : intersectsLine BE AD F)

theorem length_FD_closest_value :
  length FD = 4 := sorry

end length_FD_closest_value_l116_116246


namespace opposite_of_neg_2023_l116_116945

theorem opposite_of_neg_2023 : ∃ x : ℤ, -2023 + x = 0 ∧ x = 2023 := 
by 
  use 2023
  constructor
  · exact rfl
  · exact rfl

end opposite_of_neg_2023_l116_116945


namespace opposite_of_neg_2023_l116_116984

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l116_116984


namespace trajectory_and_chord_l116_116098

noncomputable def trajectory_eq (P : ℝ × ℝ) (F : ℝ × ℝ) (l : ℝ → ℝ) : Prop :=
  let d := |P.1 + 4|
  let PF := real.sqrt ((P.1 + 1)^2 + P.2 ^ 2)
  d = 2 * PF

noncomputable def chord_length (P Q F : ℝ × ℝ) := 
  let x1 := P.1; let y1 := P.2
  let x2 := Q.1; let y2 := Q.2
  ∃ m : ℝ, m ≠ 0 ∧ (x2 + 1, y2) = (-(x1 + 1), -y1) / 2

theorem trajectory_and_chord 
  (P : ℝ × ℝ) (F : ℝ × ℝ) (l : ℝ → ℝ)
  (h1 : P.1 + 4 = 2 * real.sqrt ((P.1 + 1) ^ 2 + P.2 ^ 2))
  (Q : ℝ × ℝ)
  (h2 : ∃ m : ℝ, (x2 + 1, y2) = (-(x1 + 1), -y1) / 2 ∧ m ≠ 0) :
  (∃ x y : ℝ, (x / 4) ^ 2 + (y / 3) ^ 2 = 1) ∧
  (|P.1 - Q.1| * real.sqrt(1 + (4/5)) = 27 / 8) → 
  |9 * real.sqrt 5| / 2 = true :=
sorry

end trajectory_and_chord_l116_116098


namespace value_of_exp_l116_116528

def geom_seq (a : ℕ → ℝ) : Prop :=
  ∃ a₁ r : ℝ, ∀ n, a n = a₁ * r^(n - 1)

variables (a : ℕ → ℝ) (a₁ r : ℝ)

axiom seq_geom : geom_seq a
axiom cond : a 4 + a 8 = -2

theorem value_of_exp : a 6 * (a 2 + 2 * a 6 + a 10) = 4 :=
by
  sorry

end value_of_exp_l116_116528


namespace opposite_of_neg2023_l116_116992

theorem opposite_of_neg2023 : ∀ (x : Int), -2023 + x = 0 → x = 2023 :=
by
  intros x h
  sorry

end opposite_of_neg2023_l116_116992


namespace find_pos_integers_l116_116058

theorem find_pos_integers (x y : ℕ) (h1 : Nat.coprime x (y - 1)) (h2 : x^2 - x + 1 = y^3) :
  (x = 1 ∧ y = 1) ∨ (x = 19 ∧ y = 7) :=
sorry

end find_pos_integers_l116_116058


namespace tan_add_pi_over_3_l116_116178

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = - (6 + 5 * Real.sqrt 3) / 13 :=
sorry

end tan_add_pi_over_3_l116_116178


namespace opposite_of_neg_2023_l116_116885

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116885


namespace opposite_of_neg_2023_l116_116806

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by sorry

end opposite_of_neg_2023_l116_116806


namespace alicia_sequence_fifth_term_l116_116737

def seq_step (m : ℕ) : ℕ :=
  let n := if m % 2 = 0 then m / 2 else m + 1 in
  m + n + 1

noncomputable def alicia_sequence (m : ℕ) : List ℕ :=
  (List.range 5).scanl (λ acc _, seq_step acc) m

theorem alicia_sequence_fifth_term (h : 3) : alicia_sequence 3 ![_, _, _, _, 43] :=
by sorry

end alicia_sequence_fifth_term_l116_116737


namespace opposite_of_neg_2023_l116_116937

theorem opposite_of_neg_2023 : ∃ x : ℤ, -2023 + x = 0 ∧ x = 2023 := 
by 
  use 2023
  constructor
  · exact rfl
  · exact rfl

end opposite_of_neg_2023_l116_116937


namespace find_abc_l116_116411

theorem find_abc (a b c : ℕ) (h_coprime_ab : gcd a b = 1) (h_coprime_ac : gcd a c = 1) 
  (h_coprime_bc : gcd b c = 1) (h1 : ab + bc + ac = 431) (h2 : a + b + c = 39) 
  (h3 : a + b + (ab / c) = 18) : 
  a = 7 ∧ b = 9 ∧ c = 23 := 
sorry

end find_abc_l116_116411


namespace ellipse_equation_length_AB_l116_116551

noncomputable def ellipse_focus (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) : Prop := 
∃ c : ℝ, c = sqrt (a^2 - b^2) ∧ c / a = sqrt 2 / 2 ∧ a - c = sqrt 2 - 1

theorem ellipse_equation (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) (hf : ellipse_focus a b h₁ h₂) :
  ∀ x y : ℝ, (x^2 / 2 + y^2 = 1) ↔ (x^2 / a^2 + y^2 / b^2 = 1) := 
sorry

noncomputable def area_OAB (x1 x2 k : ℝ) : ℝ :=
abs (x1 - x2) * sqrt (1 + k^2) * 2 / (sqrt (1 + k^2))

theorem length_AB (a b : ℝ) (h₁ : a > b) (h₂ : b > 0)
  (hf : ellipse_focus a b h₁ h₂)
  (h_area : ∃ k : ℝ, k^2 > 3 / 2 ∧ area_OAB (-4 * k / (1 + 2 * k^2))
                                            (4 * k / (1 + 2 * k^2)) k = sqrt 2 / 2) :
  ∃ (x1 x2 : ℝ), abs (x1 - x2) * sqrt (1 + (sqrt 14 / 2)^2) = 3 / 2 :=
sorry

end ellipse_equation_length_AB_l116_116551


namespace abs_g_eq_l116_116266

def g (x : ℝ) : ℝ :=
  if -4 ≤ x ∧ x < 0 then
    x + 2
  else if 0 ≤ x ∧ x ≤ 6 then
    Real.sqrt(9 - (x - 3)^2) - 3
  else if 6 < x ∧ x ≤ 7 then
    -x + 6
  else
    0  -- placeholder: outside the given defined range

theorem abs_g_eq (x : ℝ) :
  |g x| = 
    if -4 ≤ x ∧ x < 0 then
      x + 2
    else if 0 ≤ x ∧ x ≤ 6 then
      -Real.sqrt(9 - (x - 3)^2) + 3
    else if 6 < x ∧ x ≤ 7 then
      x - 6
    else
      0 :=
begin
  sorry
end

end abs_g_eq_l116_116266


namespace batsman_highest_score_l116_116419

theorem batsman_highest_score :
  ∃ (H L : ℝ), (H - L = 150) ∧
  (∃ S : ℝ, S / 46 = 58 ∧
   ((S - H - L) / 44 = 58)) ∧
  (H + L = 136) ∧ (H = 143) :=
begin
  sorry
end

end batsman_highest_score_l116_116419


namespace find_number_l116_116405

-- Given conditions and declarations
variable (x : ℕ)
variable (h : x / 3 = x - 42)

-- Proof problem statement
theorem find_number : x = 63 := 
sorry

end find_number_l116_116405


namespace opposite_of_neg_2023_l116_116834

def opposite (n : ℤ) : ℤ :=
  -n

theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116834


namespace exists_subset_A_l116_116100

theorem exists_subset_A (n k : ℕ) (h1 : 2 ≤ n) (h2 : n < 2^k) :
  ∃ A : Finset ℕ, (∀ x y ∈ A, x ≠ y → Nat.choose y x % 2 = 0) ∧
  (A.card ≥ (Nat.choose k (k / 2) / 2^k) * (n + 1)) :=
by
  sorry

end exists_subset_A_l116_116100


namespace longest_side_length_l116_116042

-- Define the conditions as a set of inequalities.
def feasible_region (x y : ℝ) :=
  x + 2 * y ≤ 4 ∧ 3 * x + y ≥ 3 ∧ x ≥ 0 ∧ y ≥ 0

-- Prove that the longest side in the resulting region is 2 * sqrt 5.
theorem longest_side_length :
  ∃ (x1 y1 x2 y2 : ℝ), feasible_region x1 y1 ∧ feasible_region x2 y2 ∧ 
  (x1 ≠ x2 ∨ y1 ≠ y2) ∧
  ∀ (u1 v1 u2 v2 : ℝ), feasible_region u1 v1 ∧ feasible_region u2 v2 ∧ 
  (u1 ≠ u2 ∨ v1 ≠ v2) → 
  real.sqrt (((x2 - x1) ^ 2) + ((y2 - y1) ^ 2)) ≥ real.sqrt (((u2 - u1) ^ 2) + ((v2 - v1) ^ 2)) :=
sorry

end longest_side_length_l116_116042


namespace gcd_1260_1440_l116_116388

open Int

theorem gcd_1260_1440 : gcd 1260 1440 = 180 := by
  have h1 : natPrimeFactors 1260 = {2, 2, 3, 3, 5, 7} := sorry
  have h2 : natPrimeFactors 1440 = {2, 2, 2, 2, 2, 3, 3, 5} := sorry
  have h3 : natPrimeSet := {2, 2, 3, 3, 5} -- Intersection of prime factors with their lowest exponents
  thus gcd 1260 1440 = (2^2) * (3^2) * 5 := sorry 
  sorry


end gcd_1260_1440_l116_116388


namespace intersection_point_on_altitude_l116_116642

-- Given points and conditions
variables (A B C E F Q P K : Point)
variables (hEF_BC : Segment E F ⊆ Segment B C)
variables (hTangent_AB : Tangent (Semicircle E F) (Segment A B) Q)
variables (hTangent_AC : Tangent (Semicircle E F) (Segment A C) P)

-- Defining the question and expressing it in Lean 4
theorem intersection_point_on_altitude :
  (Intersection (Line E P) (Line F Q) = K) → (K ∈ Altitude A (Triangle A B C)) :=
by
  sorry

end intersection_point_on_altitude_l116_116642


namespace change_given_back_l116_116278

theorem change_given_back
  (p s t a : ℕ)
  (hp : p = 140)
  (hs : s = 43)
  (ht : t = 15)
  (ha : a = 200) :
  (a - (p + s + t)) = 2 :=
by
  sorry

end change_given_back_l116_116278


namespace diff_of_powers_of_2_lt_1000_l116_116062

theorem diff_of_powers_of_2_lt_1000 : 
  {n : ℕ | ∃ a b : ℕ, n = 2^a - 2^b ∧ a > b ∧ n < 1000}.to_finset.card = 50 := 
by {
  sorry
}

end diff_of_powers_of_2_lt_1000_l116_116062


namespace opposite_of_negative_2023_l116_116789

theorem opposite_of_negative_2023 : (- -2023) = 2023 := 
by {
sorry
}

end opposite_of_negative_2023_l116_116789


namespace bill_thinks_he_counted_l116_116013

theorem bill_thinks_he_counted (actual_count double_counted missed: ℕ): 
    actual_count = 21 → double_counted = 8 → missed = 3 → 
    (actual_count + double_counted - missed) = 26 := 
by 
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  exact rfl

end bill_thinks_he_counted_l116_116013


namespace factorial_division_l116_116467

theorem factorial_division :
  (fact (fact 4)) / (fact 4) = fact 23 := by
  sorry

end factorial_division_l116_116467


namespace Greg_and_Earl_together_l116_116489

-- Conditions
def Earl_initial : ℕ := 90
def Fred_initial : ℕ := 48
def Greg_initial : ℕ := 36

def Earl_to_Fred : ℕ := 28
def Fred_to_Greg : ℕ := 32
def Greg_to_Earl : ℕ := 40

def Earl_final : ℕ := Earl_initial - Earl_to_Fred + Greg_to_Earl
def Fred_final : ℕ := Fred_initial + Earl_to_Fred - Fred_to_Greg
def Greg_final : ℕ := Greg_initial + Fred_to_Greg - Greg_to_Earl

-- Theorem statement
theorem Greg_and_Earl_together : Greg_final + Earl_final = 130 := by
  sorry

end Greg_and_Earl_together_l116_116489


namespace opposite_neg_2023_l116_116774

theorem opposite_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_neg_2023_l116_116774


namespace min_cost_of_pool_construction_l116_116032

theorem min_cost_of_pool_construction 
  (volume : ℝ) (depth : ℝ) (cost_bottom_per_sqm : ℝ) (cost_walls_per_sqm : ℝ) :
  volume = 18 → depth = 2 → cost_bottom_per_sqm = 200 → cost_walls_per_sqm = 150 →
  ∃ l w : ℝ, lw * depth = volume ∧ l = w → 
  let bottom_area := l * w,
      walls_area := 2 * (l + w) * depth,
      total_cost := bottom_area * cost_bottom_per_sqm + walls_area * cost_walls_per_sqm
  in total_cost = 5400 :=
by
  intro volume depth cost_bottom_per_sqm cost_walls_per_sqm
  intro volume_eq depth_eq cost_bottom_eq cost_walls_eq
  use 3, 3
  split
  sorry -- lw * depth = volume
  split
  sorry -- l = w
  sorry -- total_cost calculation with correct area values showing it equals 5400

end min_cost_of_pool_construction_l116_116032


namespace jogging_track_circumference_l116_116036

-- Define the speeds and time
def deepak_speed : ℝ := 20
def wife_speed : ℝ := 13
def time_minutes : ℝ := 33

-- The time converted to hours
def time_hours : ℝ := time_minutes / 60

-- Calculate the distances each one covers
def deepak_distance : ℝ := deepak_speed * time_hours
def wife_distance : ℝ := wife_speed * time_hours

-- Total distance is the circumference of the track
def circumference : ℝ := deepak_distance + wife_distance

-- The statement to prove that this circumference equals 18.15 km
theorem jogging_track_circumference : circumference = 18.15 := by
  sorry

end jogging_track_circumference_l116_116036


namespace division_result_l116_116072

open Polynomial

noncomputable def dividend := (X ^ 6 - 5 * X ^ 4 + 3 * X ^ 3 - 7 * X ^ 2 + 2 * X - 8 : Polynomial ℤ)
noncomputable def divisor := (X - 3 : Polynomial ℤ)
noncomputable def expected_quotient := (X ^ 5 + 3 * X ^ 4 + 4 * X ^ 3 + 15 * X ^ 2 + 38 * X + 116 : Polynomial ℤ)
noncomputable def expected_remainder := (340 : ℤ)

theorem division_result : (dividend /ₘ divisor) = expected_quotient ∧ (dividend %ₘ divisor) = C expected_remainder := by
  sorry

end division_result_l116_116072


namespace coefficient_x3y3_expansion_l116_116746

theorem coefficient_x3y3_expansion :
  (2 * (nat.choose 5 3) * (2:ℚ) = 20) :=
by sorry

end coefficient_x3y3_expansion_l116_116746


namespace problem1_problem2_l116_116469

theorem problem1 : 6 + (-8) - (-5) = 3 := sorry

theorem problem2 : 18 / (-3) + (-2) * (-4) = 2 := sorry

end problem1_problem2_l116_116469


namespace NV_squared_eq_2_l116_116613

open_locale classical
noncomputable theory

-- Define the square PQRS and the points M, T, N, U, V, W
variables (P Q R S M T N U V W : ℝ²)
variables (PQ PS QR RS TM NV UW : set ℝ²)

-- Assume the necessary geometric conditions are true
variables (h1 : PQRS_is_square P Q R S)
variables (h2 : point_on_segment M P Q)
variables (h3 : point_on_segment T P S)
variables (h4 : PM_eq_PT P M T)
variables (h5 : point_on_segment N Q R)
variables (h6 : point_on_segment U R S)
variables (h7 : point_on_segment V T M)
variables (h8 : point_on_segment W T M)
variables (h9 : NV_perpendicular_TM NV TM)
variables (h10 : UW_perpendicular_TM UW TM)
variables (h11 : area_PMT_eq_1 P M T)
variables (h12 : area_QNVM_eq_1 Q N V M)
variables (h13 : area_STWU_eq_1 S T W U)
variables (h14 : area_RUVWN_eq_1 R U V W N)

-- Prove that NV^2 = 2
theorem NV_squared_eq_2 : NV^2 = 2 :=
sorry

end NV_squared_eq_2_l116_116613


namespace probability_odd_is_one_half_l116_116327

noncomputable def probability_odd_four_digit_number : ℚ :=
  let digits := {1, 4, 6, 9}
  let odd_digits := {d ∈ digits | d % 2 = 1}
  (odd_digits.card : ℚ) / (digits.card : ℚ)

theorem probability_odd_is_one_half (digits : Finset ℕ) (odd_digits : Finset ℕ) :
  digits = {1, 4, 6, 9} →
  odd_digits = {d ∈ digits | d % 2 = 1} →
  probability_odd_four_digit_number = 1 / 2 :=
by
  intros h1 h2
  sorry

end probability_odd_is_one_half_l116_116327


namespace cos_2x_min_val_l116_116482

theorem cos_2x_min_val (x : ℝ) (h : ∀ x, 2 * (Real.sin x)^6 + (Real.cos x)^6 ≤ 2 * (Real.sin x)^6 + (Real.cos x)^6)
  : cos 2 * x = 3 - 2 * Real.sqrt 2 := 
sorry

end cos_2x_min_val_l116_116482


namespace gcd_power_product_bound_l116_116638

theorem gcd_power_product_bound (a : ℕ → ℕ) (n : ℕ) (P : ℕ) :
  (∀ i, 0 < a i) →
  (n % 2 = 1) →
  (∏ i in finset.range n, a i) = P →
  Nat.gcd 
    (finset.fold gcd 0 (finset.range n) (λ i, (a i)^n + P)) 
    ≤ 2 * (finset.fold gcd 0 (finset.range n) a)^n :=
by
  sorry

end gcd_power_product_bound_l116_116638


namespace circle_ratio_l116_116436

theorem circle_ratio (R r : ℝ) (h₁ : R > 0) (h₂ : r > 0) 
                     (h₃ : π * R^2 - π * r^2 = 3 * π * r^2) : R = 2 * r :=
by
  sorry

end circle_ratio_l116_116436


namespace Pascal_remaining_distance_l116_116696

theorem Pascal_remaining_distance (D T : ℕ) :
  let current_speed := 8
  let reduced_speed := current_speed - 4
  let increased_speed := current_speed + current_speed / 2
  (D = current_speed * T) →
  (D = reduced_speed * (T + 16)) →
  (D = increased_speed * (T - 16)) →
  D = 256 :=
by
  intros
  sorry

end Pascal_remaining_distance_l116_116696


namespace probability_of_9_heads_in_12_flips_l116_116386

theorem probability_of_9_heads_in_12_flips :
  (∃ n : ℕ, n = 12) →
  (1 / (2 ^ 12) * (∑ k in finset.range (12 + 1), if k = 9 then (nat.choose 12 9) else 0)) = (55 / 1024) :=
by
  sorry

end probability_of_9_heads_in_12_flips_l116_116386


namespace minimum_value_proof_l116_116654

noncomputable def minimum_value_condition (a b : ℝ) : Prop :=
a > 0 ∧ b > 0 ∧ a + b = 12

theorem minimum_value_proof : ∀ (a b : ℝ), minimum_value_condition a b → (1 / a + 1 / b ≥ 1 / 3) := 
by
  intros a b h
  sorry

end minimum_value_proof_l116_116654


namespace digit_at_2003rd_position_is_4_l116_116317

theorem digit_at_2003rd_position_is_4 :
  let sequence := list.join (list.map (λ n, n.digits 10) (list.range 2004))
  sequence.nth 2002 = some 4 :=
by
  sorry

end digit_at_2003rd_position_is_4_l116_116317


namespace opposite_of_neg_2023_l116_116939

theorem opposite_of_neg_2023 : ∃ x : ℤ, -2023 + x = 0 ∧ x = 2023 := 
by 
  use 2023
  constructor
  · exact rfl
  · exact rfl

end opposite_of_neg_2023_l116_116939


namespace angle_B_possibilities_l116_116249

variables {A B C O H : Type*}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space O] [metric_space H]
variables [affine_space A] [affine_space B] [affine_space C]

def is_circumcenter (O : affine_space) (A B C : affine_space) : Prop := sorry
def is_orthocenter (H : affine_space) (A B C : affine_space) : Prop := sorry

theorem angle_B_possibilities {A B C : Type*} [metric_space A] [metric_space B] [metric_space C]
  [affine_space A] [affine_space B] [affine_space C] {O H : affine_space}
  (h_circumcenter : is_circumcenter O A B C) (h_orthocenter : is_orthocenter H A B C)
  (h_eq : dist B O = dist B H) :
  ∃ (α : ℝ), α = 60 ∨ α = 120 :=
sorry

end angle_B_possibilities_l116_116249


namespace probability_nine_heads_l116_116383

theorem probability_nine_heads:
  (∀ (n k : ℕ), k ≤ n → (finset.card (finset.filter (λ (s : finset ℕ), s.card = k) (finset.powerset (finset.range n))) = nat.choose n k)) →
  ∃ p : ℚ, p = 55 / 1024 ∧ 
    calc 
      220 / 4096 = p :
      sorry :=
begin
  let n := 12,
  let k := 9,
  have h1: 2^n = 4096 := by norm_num,
  have h2 : nat.choose n k = 220 := by norm_num,
  have h3 : 220 / 4096 = 55 / 1024 := by norm_num,
  existsi (55 / 1024 : ℚ),
  split,
  { refl, },
  { exact h3 }
end

end probability_nine_heads_l116_116383


namespace common_tangent_l116_116027

variables {X₁ O Y X₂ : Point}
variables {ω₁ ω₂ : Circle}
variables {A₁ A₂ B₁ B₂ C₁ C₂ : Point}

-- Conditions: Definitions of the problem.
-- Assuming given points and circles with required properties satisfy the conditions
axiom angle_eq : angle X₁ O Y = angle Y O X₂
axiom touch_X₁ : ω₁.touches (line O X₁) at A₁
axiom touch_X₂ : ω₂.touches (line O X₂) at A₂
axiom touch_Y₁ : ω₁.touches (line O Y) at B₁
axiom touch_Y₂ : ω₂.touches (line O Y) at B₂
axiom C₁_def : C₁ = (A₁.project_through B₂ on ω₁).second_intersection
axiom C₂_def : C₂ = (A₂.project_through B₁ on ω₂).second_intersection

-- Theorem to prove
theorem common_tangent : tangent_to_two_circles C₁C₂ ω₁ ω₂ :=
sorry

end common_tangent_l116_116027


namespace correct_conclusion_l116_116727

def prop_p (x : ℝ) : Prop := x ^ 2 = 1 → x = -1
def prop_q (k : ℝ) : Prop :=
  let a := (1, 1, 0)
  let b := (-1, 0, 2)
  let dot (u v : ℝ × ℝ × ℝ) := u.1 * v.1 + u.2 * v.2 + u.3 * v.3
  dot (k * a.1 + b.1, k * a.2 + b.2, k * a.3 + b.3) (2 * a.1 - b.1, 2 * a.2 - b.2, 2 * a.3 - b.3) = 0 ↔ k = 7 / 5

theorem correct_conclusion : ¬prop_p (-1) ∧ prop_q (7 / 5) :=
by {
  sorry
}

end correct_conclusion_l116_116727


namespace opposite_of_neg_2023_l116_116965

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l116_116965


namespace opposite_of_neg_2023_l116_116951

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l116_116951


namespace sum_series_correct_l116_116022

def cyclic_coefficient_sum (n : ℕ) : ℂ := 
  ∑ k in Finset.range n, (k + 1) * (complex.I ^ (k + 1))

theorem sum_series_correct : cyclic_coefficient_sum 2500 = -1252 + 1252 * complex.I :=
by
  sorry

end sum_series_correct_l116_116022


namespace find_smallest_shift_l116_116741

variable {α : Type*} [Add α] [Div α α α] [DecidableEq α]

-- Given conditions
def periodic_function (f : α → α) (p : α) : Prop :=
  ∀ x, f (x - p) = f x

-- Question and the correct result
def smallest_shift (f : α → α) (p : α) (a : α) : Prop :=
  p = 40 ∧ (∀ x, f ((x - a) / 10) = f (x / 10)) ∧ ∀ (a' : α), 0 < a' → (∀ x, f ((x - a') / 10) = f (x / 10)) → a ≤ a'

-- The statement
theorem find_smallest_shift (f : ℝ → ℝ) :
  periodic_function f 40 →
  smallest_shift f 40 400 :=
sorry

end find_smallest_shift_l116_116741


namespace opposite_of_neg_2023_l116_116959

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l116_116959


namespace polyhedron_center_of_symmetry_l116_116728

theorem polyhedron_center_of_symmetry 
  (P : Polyhedron)
  (convex_P : convex P)
  (faces_symm_center : ∀ face ∈ faces P, has_center_of_symmetry face) 
  : has_center_of_symmetry P :=
sorry

end polyhedron_center_of_symmetry_l116_116728


namespace tan_angle_addition_l116_116144

theorem tan_angle_addition (x : Real) (h1 : Real.tan x = 3) (h2 : Real.tan (Real.pi / 3) = Real.sqrt 3) : 
  Real.tan (x + Real.pi / 3) = (3 + Real.sqrt 3) / (1 - 3 * Real.sqrt 3) :=
by 
  sorry

end tan_angle_addition_l116_116144


namespace floor_sqrt_10_eq_3_l116_116053

theorem floor_sqrt_10_eq_3 :
  ∃ x : ℤ, x = 3 ∧ x ≤ Real.sqrt 10 ∧ Real.sqrt 10 < x + 1 :=
by
  use 3
  split
  . rfl
  split
  . apply Real.le_sqrt_of_sq_le
    calc 9 = 3 ^ 2 : by norm_num
           ... ≤ 10 : by norm_num
  . apply Real.sqrt_lt_of_sq_lt
    calc 10 < 16 : by norm_num
           ... = 4 ^ 2 : by norm_num

end floor_sqrt_10_eq_3_l116_116053


namespace range_of_a_l116_116085

-- Definitions
def p (x a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := x^2 + 2 * x - 8 > 0

-- Main theorem to prove
theorem range_of_a (a : ℝ) (h : a < 0)
  (h_necessary : ∀ x, ¬ p x a → ¬ q x) 
  (h_not_sufficient : ∃ x, ¬ p x a ∧ q x): 
  a ≤ -4 :=
sorry

end range_of_a_l116_116085


namespace incident_ray_slope_l116_116113

noncomputable def slope {A B : Type} [LinearOrderedField A] [AddCommGroup B] [Module A B] (p1 p2 : B × B) : A :=
  (p2.2.2 - p1.2.2) / (p2.1 - p1.1)

def P : (ℝ × ℝ) := (-1, -3)
def D : (ℝ × ℝ) := (2, 1)

theorem incident_ray_slope : slope P D = 4 / 3 :=
by
  sorry

end incident_ray_slope_l116_116113


namespace opposite_of_neg_2023_l116_116863

-- Definition of the problem
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  -- The proof is to be provided
  sorry

end opposite_of_neg_2023_l116_116863


namespace log_equation_solution_l116_116312

theorem log_equation_solution (x : ℝ) :
  (log 3 ((4 * x + 10) / (6 * x - 2)) + log 3 ((6 * x - 2) / (2 * x - 3)) = 3) → x = 91 / 50 :=
by
  sorry

end log_equation_solution_l116_116312


namespace sqrt_expr_evaluation_l116_116468

theorem sqrt_expr_evaluation : 
  (Real.sqrt 24) - 3 * (Real.sqrt (1 / 6)) + (Real.sqrt 6) = (5 * Real.sqrt 6) / 2 :=
by
  sorry

end sqrt_expr_evaluation_l116_116468


namespace painting_price_increase_decrease_l116_116345

theorem painting_price_increase_decrease (P : ℝ) :
  let first_year_price := P * 1.25 in
  let second_year_price := first_year_price * 0.85 in
  second_year_price = P * 1.0625 :=
by
  sorry

end painting_price_increase_decrease_l116_116345


namespace sum_of_first_n_terms_l116_116111

noncomputable def a : ℕ → ℕ
| n => 2^n

noncomputable def b : ℕ → ℕ
| n => 2*n - 1

noncomputable def S : ℕ → ℕ
| n => 2 * a n - 2

noncomputable def T : ℕ → ℕ
| 0     => 0
| (n+1) => (2 * (n+1) - 3) * 2^(n+2) + 6

theorem sum_of_first_n_terms (n : ℕ) : 
  (Σ i in finset.range n, a i * b i) = T n := sorry

end sum_of_first_n_terms_l116_116111


namespace remaining_distance_l116_116713

variable (D : ℝ) -- remaining distance in miles

-- Current speed
def current_speed := 8.0
-- Increased speed by 50%
def increased_speed := 12.0
-- Reduced speed by 4 mph
def reduced_speed := 4.0

-- Trip time relations
def current_time := D / current_speed
def increased_time := D / increased_speed
def reduced_time := D / reduced_speed

-- Time difference condition
axiom condition : reduced_time = increased_time + 16.0

theorem remaining_distance : D = 96.0 :=
by
  -- proof placeholder
  sorry

end remaining_distance_l116_116713


namespace find_m_n_equal_probs_l116_116083

open Nat

theorem find_m_n_equal_probs (m n : ℕ) (hmn : 10 ≥ m ∧ m > n ∧ n ≥ 4) :
  (choose m 2 + choose n 2) * choose (m + n) 2 = m * n * choose (m + n) 2 →
  (m, n) = (10, 6) := by
  sorry

end find_m_n_equal_probs_l116_116083


namespace percentage_of_green_leaves_l116_116019

-- Definitions for the conditions in the problem
def leaves_thursday : Nat := 12
def leaves_friday : Nat := 13
def total_leaves : Nat := leaves_thursday + leaves_friday
def brown_leaves_percentage : ℚ := 0.20
def yellow_leaves : Nat := 15
def brown_leaves : Nat := brown_leaves_percentage * total_leaves
def green_leaves : Nat := total_leaves - brown_leaves - yellow_leaves

-- The theorem we want to prove
theorem percentage_of_green_leaves :
  (green_leaves / total_leaves : ℚ) * 100 = 20 :=
by
  sorry

end percentage_of_green_leaves_l116_116019


namespace opposite_of_negative_2023_l116_116788

theorem opposite_of_negative_2023 : (- -2023) = 2023 := 
by {
sorry
}

end opposite_of_negative_2023_l116_116788


namespace tan_X_correct_l116_116226

section
variable {X Y Z : Type} -- Define variables for generic points

-- Define the conditions from the problem
variable (angleY : ∠ Y = 90)
variable (YZ : ℝ) (XZ : ℝ)

-- Specific values given in the problem
variable (h1 : YZ = 4) (h2 : XZ = Real.sqrt 41)

-- Define the resulting tangent function
noncomputable def tan_X (X Y Z : Type) (angleY : ∠ Y = 90) (YZ XZ : ℝ) :=
  YZ / (Real.sqrt (XZ ^ 2 - YZ ^ 2))

-- The proof problem
theorem tan_X_correct : tan_X X Y Z angleY 4 (Real.sqrt 41) = 4 / 5 :=
by
  rw [tan_X]
  rw [h1, h2]
  -- Here we would carry out the calculation indicated in the solution.
  sorry
end

end tan_X_correct_l116_116226


namespace probability_of_non_perimeter_square_l116_116685

-- Defining the total number of squares on a 10x10 board
def total_squares : ℕ := 10 * 10

-- Defining the number of perimeter squares
def perimeter_squares : ℕ := 10 + 10 + (10 - 2) * 2

-- Defining the number of non-perimeter squares
def non_perimeter_squares : ℕ := total_squares - perimeter_squares

-- Defining the probability of selecting a non-perimeter square
def probability_non_perimeter : ℚ := non_perimeter_squares / total_squares

-- The main theorem statement to be proved
theorem probability_of_non_perimeter_square:
  probability_non_perimeter = 16 / 25 := 
sorry

end probability_of_non_perimeter_square_l116_116685


namespace measure_angle_P_l116_116611

theorem measure_angle_P (P Q R S : ℝ) (hP : P = 3 * Q) (hR : 4 * R = P) (hS : 6 * S = P) (sum_angles : P + Q + R + S = 360) :
  P = 206 :=
by
  sorry

end measure_angle_P_l116_116611


namespace opposite_of_neg_2023_l116_116893

-- Let x be the opposite of -2023
def is_opposite (x : ℤ) (n : ℤ) : Prop := n + x = 0

theorem opposite_of_neg_2023 : ∃ x : ℤ, is_opposite x (-2023) ∧ x = 2023 :=
by
  exists 2023
  unfold is_opposite
  sorry

end opposite_of_neg_2023_l116_116893


namespace opposite_of_negative_2023_l116_116785

theorem opposite_of_negative_2023 : (- -2023) = 2023 := 
by {
sorry
}

end opposite_of_negative_2023_l116_116785


namespace opposite_of_neg_2023_l116_116821

/-- Define the opposite of a number -/
def opposite (x : ℤ) : ℤ := -x

/-- Prove the opposite of -2023 is 2023 -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116821


namespace apples_hanging_on_tree_l116_116518

theorem apples_hanging_on_tree (total_apples : ℕ) (fallen_apples : ℕ) (dog_eats : ℕ) (apples_left : ℕ)
                                (h_total : total_apples = apples_left + dog_eats)
                                (h_fallen : fallen_apples = 8)
                                (h_dog : dog_eats = 3)
                                (h_left : apples_left = 10) :
                                total_apples - fallen_apples = 5 :=
by
  rw [h_total, h_dog, h_left]
  simp [h_fallen]
  sorry

end apples_hanging_on_tree_l116_116518


namespace find_x_l116_116495

open Real

theorem find_x (x : ℝ) (h : log 8 (3 * x - 4) = 2) : x = 68 / 3 := by
  sorry

end find_x_l116_116495


namespace normal_distribution_interval_approx_57_of_60_l116_116600

theorem normal_distribution_interval_approx_57_of_60
  (X : ℝ → ProbMeasure ℝ)
  (hX : X = normal 110 5)
  (total_students : ℕ)
  (h_total_students : total_students = 60)
  (approx_students_in_interval : ℕ)
  (h_approx_students_in_interval : approx_students_in_interval = 57) :
  ∃ interval, interval = set.Ioc 100 120 ∧ ∀ s ∈ interval, 
  dist (X 110) < 2 * 5 :=
by
  sorry

end normal_distribution_interval_approx_57_of_60_l116_116600


namespace log8_of_4096_bounds_l116_116352

theorem log8_of_4096_bounds : 
  let a := 4
  let b := 5
  \(\log_8{4096} = 4 \land a + b = 9\) ->
  4 <= \log_8{4096} < 5 ∧ a + b = 9 := 
by 
  sorry 

end log8_of_4096_bounds_l116_116352


namespace ship_speed_one_sail_l116_116020

noncomputable -- Indicate non-computational definitions if necessary

def speed_with_two_sails : ℝ := 50 -- Speed with two sails (in knots)

def hours_with_one_sail : ℝ := 4 -- Time traveled with one sail (in hours)
def hours_with_two_sails : ℝ := 4 -- Time traveled with two sails (in hours)

def total_distance_land_miles : ℝ := 345 -- Total distance traveled (in land miles)
def nautical_mile_to_land_mile : ℝ := 1.15 -- Conversion factor: 1 nautical mile = 1.15 land miles

def total_distance_nautical_miles := total_distance_land_miles / nautical_mile_to_land_mile -- Convert total distance to nautical miles

axiom speed_of_ship_with_one_sail (S : ℝ) : 
  hours_with_one_sail * S + hours_with_two_sails * speed_with_two_sails = total_distance_nautical_miles → 
  S = 25

-- Now, we need the statement that encapsulates this proof problem
theorem ship_speed_one_sail :
  ∃ S : ℝ, (hours_with_one_sail * S + hours_with_two_sails * speed_with_two_sails = total_distance_nautical_miles) ∧ S = 25 :=
begin
  use 25,
  split,
  {
    -- Use the axiom to directly assert the correctness of the distance calculation
    exact speed_of_ship_with_one_sail 25,
  },
  {
    -- Obviously, S = 25 in this context
    refl,
  }
end

end ship_speed_one_sail_l116_116020


namespace maena_wins_if_n_is_odd_gaspard_wins_if_n_is_even_l116_116670

theorem maena_wins_if_n_is_odd (n : ℕ) : (n % 2 = 1) → (maena_has_winning_strategy n) :=
by sorry

theorem gaspard_wins_if_n_is_even (n : ℕ) : (n % 2 = 0) → (gaspard_has_winning_strategy n) :=
by sorry

-- Auxiliary definitions to capture the winning strategies
def maena_has_winning_strategy (n : ℕ) : Prop :=
  -- Maena has a winning strategy starting with n candies
  sorry

def gaspard_has_winning_strategy (n : ℕ) : Prop :=
  -- Gaspard has a winning strategy starting with n candies
  sorry

end maena_wins_if_n_is_odd_gaspard_wins_if_n_is_even_l116_116670


namespace combined_value_l116_116500

noncomputable def π : ℝ := 3.14159

def radius : ℝ := 7

def circumference (r : ℝ) : ℝ := 2 * π * r
def area (r : ℝ) : ℝ := π * r^2

theorem combined_value : circumference radius + area radius = 197.9203 := by
  sorry

end combined_value_l116_116500


namespace closest_integer_to_average_speed_l116_116451

def triathlete_average_speed (v1 v2 v3: ℝ) (L: ℝ) : ℝ :=
  let total_time := (L / v1) + (L / v2) + (L / v3)
  let total_distance := 3 * L
  total_distance / total_time

theorem closest_integer_to_average_speed :
  ∀ (L: ℝ) (L > 0), triathlete_average_speed 2 25 6 L ≈ 4 :=
by
  sorry

end closest_integer_to_average_speed_l116_116451


namespace integer_pairs_eq_1_l116_116582

open Int

theorem integer_pairs_eq_1 :
  (∃! xy : ℤ × ℤ, let x := xy.1, y := xy.2 in sqrt (x - sqrt (x + 23)) = 2 * sqrt 2 - y) :=
sorry

end integer_pairs_eq_1_l116_116582


namespace opposite_of_neg_2023_l116_116840

def opposite (n : ℤ) : ℤ :=
  -n

theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116840


namespace john_marbles_selection_l116_116630

theorem john_marbles_selection :
  let total_marbles := 15
  let special_colors := 4
  let total_chosen := 5
  let chosen_special_colors := 2
  let remaining_colors := total_marbles - special_colors
  let chosen_normal_colors := total_chosen - chosen_special_colors
  (Nat.choose 4 2) * (Nat.choose 11 3) = 990 :=
by
  sorry

end john_marbles_selection_l116_116630


namespace union_complement_eq_set_l116_116270

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 3, 4}
def N : Set ℕ := {3, 5, 6}
def comp_U_N : Set ℕ := U \ N  -- complement of N with respect to U

theorem union_complement_eq_set :
  M ∪ comp_U_N = {1, 2, 3, 4} :=
by
  simp [U, M, N, comp_U_N]
  sorry

end union_complement_eq_set_l116_116270


namespace find_ellipse_eq_dot_product_constant_l116_116537

-- Conditions
variables {a b c t : ℝ}
variables {F1 F2 A1 A B : (ℝ × ℝ)}
variable {M : (ℝ × ℝ)}

-- Given: Ellipse equation and eccentricity
def ellipse (x y : ℝ) : Prop := (x^2 / a^2 + y^2 / b^2 = 1)
def eccentricity (a b c : ℝ) : Prop := (c / a = 1 / 2)

-- Moving point M on the ellipse
def on_ellipse (M : (ℝ × ℝ)) : Prop := ellipse M.1 M.2

-- Maximum area condition
def max_area (F1 M F2 : (ℝ × ℝ)) : Prop := 
  (let area := (1/2) * |F2.1 - F1.1| * |M.2 - F1.2| in area = sqrt 3)

-- Equation of the ellipse
theorem find_ellipse_eq (h1 : a > 0) (h2 : b > 0) (h3 : b < a)
  (h4 : eccentricity a b c) (h5 : a = 2 * t) (h6 : b = sqrt 3 * t)
  (ht : t = 1) : ellipse x y :=
sorry

-- Determine if ∇PF2 ⋅ ∇QF2 is constant
theorem dot_product_constant
  (line_l : ∀ (A B : ℝ × ℝ), x = ty + 1)
  (hAB : ∀ (x y : ℝ), ellipse x y → x = ty + 1 → f A1 A B = P Q)
  : ∇P F2 · ∇Q F2 = 0 :=
sorry


end find_ellipse_eq_dot_product_constant_l116_116537


namespace find_symmetric_point_l116_116089

theorem find_symmetric_point (p : ℝ) (hp : 0 < p) (P Q : ℝ × ℝ) (hP_on_parabola : P.2 ^ 2 = 2 * p * P.1) (hQ : Q = (5,0)) 
  (h_symmetric : ∃ l : ℝ, l = Real.tan (π / 6) ∧ Line_through_and_symmetry (P, Q, (p / 2, 0), l)) : 
  P = (3, Real.sqrt (6)) ∨ P = (3, -Real.sqrt (6)) := 
by
  sorry

-- Additional definition for line-through and symmetry conditions
def Line_through_and_symmetry : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × ℝ → Prop := 
  λ (P Q F l) => 
    let line_through : Prop := (P.2 = l * (P.1 - F.1))
    let symmetric_condition : Prop := (Q.1 + P.1) / 2 = F.1 ∧ l = Real.tan(π / 6)
    line_through ∧ symmetric_condition

end find_symmetric_point_l116_116089


namespace joanne_total_weekly_earnings_l116_116232

-- Define constants for rates and hours
def main_job_hourly_rate := 16.0
def main_job_hours_per_day := 8
def part_time_job_hourly_rate := 13.5
def part_time_job_hours_per_day := 2
def days_per_week := 5

-- Define calculations for daily and weekly earnings
def main_job_daily_earnings := main_job_hourly_rate * main_job_hours_per_day
def main_job_weekly_earnings := main_job_daily_earnings * days_per_week

def part_time_job_daily_earnings := part_time_job_hourly_rate * part_time_job_hours_per_day
def part_time_job_weekly_earnings := part_time_job_daily_earnings * days_per_week

-- Total weekly earnings from both jobs
def total_weekly_earnings := main_job_weekly_earnings + part_time_job_weekly_earnings

-- The main theorem to prove
theorem joanne_total_weekly_earnings : total_weekly_earnings = 775.0 := by
  -- Proof steps would go here
  sorry

end joanne_total_weekly_earnings_l116_116232


namespace math_problem_l116_116264

-- Conditions
def ellipse_eq (a b : ℝ) : Prop := ∀ x y : ℝ, x^2 / (a^2) + y^2 / (b^2) = 1
def eccentricity (a c : ℝ) : Prop := c / a = (Real.sqrt 2) / 2
def major_axis_length (a : ℝ) : Prop := 2 * a = 6 * Real.sqrt 2

-- Equations and properties to be proven
def ellipse_equation : Prop := ∃ a b : ℝ, a = 3 * Real.sqrt 2 ∧ b = 3 ∧ ellipse_eq a b
def length_AB (θ : ℝ) : Prop := ∃ AB : ℝ, AB = (6 * Real.sqrt 2) / (1 + (Real.sin θ)^2)
def min_AB_CD : Prop := ∃ θ : ℝ, (Real.sin (2 * θ) = 1) ∧ (6 * Real.sqrt 2) / (1 + (Real.sin θ)^2) + (6 * Real.sqrt 2) / (1 + (Real.cos θ)^2) = 8 * Real.sqrt 2

-- The complete proof problem
theorem math_problem : ellipse_equation ∧
                       (∀ θ : ℝ, length_AB θ) ∧
                       min_AB_CD := by
  sorry

end math_problem_l116_116264


namespace percent_x_of_w_l116_116198

theorem percent_x_of_w (x y z w : ℝ)
  (h1 : x = 1.2 * y)
  (h2 : y = 0.7 * z)
  (h3 : w = 1.5 * z) : (x / w) * 100 = 56 :=
by
  sorry

end percent_x_of_w_l116_116198


namespace rectangle_areas_and_ratio_l116_116669

-- Define the lengths and widths based on given ratios
def len_large := 40
def wid_large := 20
def len_small := (3 / 5) * len_large
def wid_small := (2 / 3) * wid_large

-- Define the areas of both rectangles
def area_large := len_large * wid_large
def area_small := len_small * wid_small

-- Define the proof statements
theorem rectangle_areas_and_ratio :
  area_large = 800 ∧ area_small = 320 ∧ (area_small / area_large) = (2 / 5) :=
by
  have h1 : area_large = 40 * 20 := rfl
  have h2 : area_small = ((3 / 5) * 40) * ((2 / 3) * 20) := rfl
  have h3 : (area_small / area_large) = (320 / 800) := sorry -- this is a placeholder for exact proof steps
  exact ⟨h1, h2, h3⟩
sorry

end rectangle_areas_and_ratio_l116_116669


namespace range_of_a_l116_116565

def piecewise_f (a : ℝ) (x : ℝ) : ℝ := 
if x ≤ 1 then x^2 - 2 * a * x + 2 
else x + 16 / x - 3 * a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (x ≤ 1 ∧ piecewise_f a x ≥ piecewise_f a 1) ∨ (x > 1 ∧ piecewise_f a x ≥ piecewise_f a 1)) ↔ (1 ≤ a ∧ a ≤ 5) :=
sorry

end range_of_a_l116_116565


namespace min_value_of_one_over_a_plus_one_over_b_l116_116651

theorem min_value_of_one_over_a_plus_one_over_b (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 12) :
  (∃ c : ℝ, (c = 1/ a + 1 / b) ∧ c = 1 / 3) :=
sorry

end min_value_of_one_over_a_plus_one_over_b_l116_116651


namespace hexagon_perimeter_l116_116729
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem hexagon_perimeter {A C : ℝ × ℝ}
  (hA : A = (0, 0)) (hC : C = (5, 6)) :
  let AC := distance A C in
  let s := (2 / 3) * AC in
  6 * s = 4 * Real.sqrt 61 :=
by {
  sorry
}

end hexagon_perimeter_l116_116729


namespace opposite_of_neg_2023_l116_116814

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by sorry

end opposite_of_neg_2023_l116_116814


namespace opposite_neg_2023_l116_116765

theorem opposite_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_neg_2023_l116_116765


namespace opposite_of_neg2023_l116_116993

theorem opposite_of_neg2023 : ∀ (x : Int), -2023 + x = 0 → x = 2023 :=
by
  intros x h
  sorry

end opposite_of_neg2023_l116_116993


namespace opposite_of_neg_2023_l116_116856

-- Definition of the problem
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  -- The proof is to be provided
  sorry

end opposite_of_neg_2023_l116_116856


namespace meera_fraction_4kmh_l116_116276

noncomputable def fraction_of_time_at_4kmh (total_time : ℝ) (x : ℝ) : ℝ :=
  x / total_time

theorem meera_fraction_4kmh (total_time x : ℝ) (h1 : x = total_time / 14) :
  fraction_of_time_at_4kmh total_time x = 1 / 14 :=
by
  sorry

end meera_fraction_4kmh_l116_116276


namespace opposite_of_neg_2023_l116_116812

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by sorry

end opposite_of_neg_2023_l116_116812


namespace part_I_part_II_l116_116084

noncomputable def n (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, Real.sqrt 3 * Real.sin x)
noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.cos x, 2 * Real.cos x)
noncomputable def f (x a : ℝ) : ℝ := (n x).1 * (m x).1 + (n x).2 * (m x).2 + a

theorem part_I (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  (∃ (x_max : ℝ), f x_max 1 = 4 ∧ 0 ≤ x_max ∧ x_max ≤ Real.pi / 2 ∧ x_max = Real.pi / 6) ∧
  (∃ (x_min : ℝ), f x_min 1 = 1 ∧ 0 ≤ x_min ∧ x_min ≤ Real.pi / 2 ∧ x_min = Real.pi / 2) := sorry

theorem part_II (x1 x2 : ℝ) (hx1x2 : 0 ≤ x1 ∧ x1 ≤ Real.pi ∧ 0 ≤ x2 ∧ x2 ≤ Real.pi) :
  ((f x1 (-1) = b ∧ f x2 (-1) = b ∧ x1 ≠ x2) → b ∈ (-2,1) ∨ b ∈ (1,2)) ∧ 
  (∃ (x1 x2 : ℝ), f x1 (-1) = b ∧ f x2 (-1) = b ∧ x1 ≠ x2 → (x1 + x2 = Real.pi / 3 ∨ x1 + x2 = 4 * Real.pi / 3)) := sorry

end part_I_part_II_l116_116084


namespace opposite_of_neg_2023_l116_116799

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by sorry

end opposite_of_neg_2023_l116_116799


namespace opposite_of_neg_2023_l116_116846

def opposite (n : ℤ) : ℤ :=
  -n

theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116846


namespace solution_l116_116046

noncomputable def problem (f : ℝ → ℝ) : Prop :=
  (∀ x y, f(x + y) * f(x - y) = f(x)^2 + f(y)^2 - 1) →
  (∃ k : ℝ, ∀ x, f''(x) = k^2 * f(x) ∨ f''(x) = - k^2 * f(x))

theorem solution (f : ℝ → ℝ) (h : ∀ x y, f(x + y) * f(x - y) = f(x)^2 + f(y)^2 - 1)
    (h_diff : differentiable ℝ f) (h_diff2 : differentiable ℝ (λ x, deriv (deriv f x))) :
    ∃ k : ℝ, (∀ x, f x = cos (k * x) ∨ f x = cosh (k * x) ∨ f x = - cos (k * x) ∨ f x = - cosh (k * x)) :=
sorry

end solution_l116_116046


namespace property_a_property_b_property_c_property_d_property_e_l116_116293

def h (m : ℕ) (x : ℝ) : ℝ :=
if m = 0 then 1 else (finset.range m).prod (λ i, 1 - x^(i + 1))

def g (k l : ℕ) (x : ℝ) : ℝ :=
h (k + l) x / (h k x * h l x)

theorem property_a (k l : ℕ) (x : ℝ) : 
  g k l x = (h (k + l) x) / (h k x * h l x) :=
sorry

theorem property_b (k l : ℕ) (x : ℝ) : 
  g k 1 x = g l k x :=
sorry

theorem property_c (k l : ℕ) (x : ℝ) : 
  g k l x = g (k - 1) l x + x^k * g k (l - 1) x ∧ 
  g k l x = g k (l - 1) x + x^l * g (k - 1) l x :=
sorry

theorem property_d (k l : ℕ) (x : ℝ) : 
  g k (l + 1) x = finset.range (k + 1).sum (λ i, x^i * g i l x) :=
sorry

theorem property_e (k l : ℕ) (x : ℝ) : 
  polynomial.degree (g k l x) = k * l :=
sorry

end property_a_property_b_property_c_property_d_property_e_l116_116293


namespace measure_angle_P_l116_116612

theorem measure_angle_P (P Q R S : ℝ) (hP : P = 3 * Q) (hR : 4 * R = P) (hS : 6 * S = P) (sum_angles : P + Q + R + S = 360) :
  P = 206 :=
by
  sorry

end measure_angle_P_l116_116612


namespace train_length_correct_l116_116000

-- Definitions for conditions
def time_to_cross_pole : ℝ := 4.99960003199744
def speed_kmh : ℝ := 144
def speed_ms : ℝ := speed_kmh * (1000 / 1) * (1 / 3600)  -- Speed converted from km/hr to m/s

-- Definition for the correct answer (length of the train)
def length_of_train : ℝ := speed_ms * time_to_cross_pole

-- Proof statement
theorem train_length_correct : length_of_train = 199.9840012798976 :=
by
  sorry

end train_length_correct_l116_116000


namespace convex_polyhedron_max_intersected_edges_non_convex_polyhedron_possible_intersected_edges_polyhedron_impossible_full_edge_intersection_l116_116342

-- Statement for part (a)
theorem convex_polyhedron_max_intersected_edges (polyhedron : Type) [convex_polyhedron polyhedron] (E : ℕ)
(eh : E = 100) : 
  ∃ (max_intersected_edges : ℕ), max_intersected_edges = 66 :=
by sorry

-- Statement for part (b)
theorem non_convex_polyhedron_possible_intersected_edges (polyhedron : Type) [non_convex_polyhedron polyhedron] (E : ℕ)
(eh : E = 100) : 
  ∃ (possible_intersected_edges : ℕ), possible_intersected_edges = 96 :=
by sorry

-- Statement for part (c)
theorem polyhedron_impossible_full_edge_intersection (polyhedron : Type) (E : ℕ)
(eh : E = 100) : 
  ¬ (∃ (intersected_edges : ℕ), intersected_edges = E) :=
by sorry

end convex_polyhedron_max_intersected_edges_non_convex_polyhedron_possible_intersected_edges_polyhedron_impossible_full_edge_intersection_l116_116342


namespace num_chords_and_triangles_l116_116743

theorem num_chords_and_triangles (n : ℕ) (h : n = 10) :
    (nat.choose n 2 = 45) ∧ (nat.choose n 3 = 120) :=
by
  sorry

end num_chords_and_triangles_l116_116743


namespace solve_arccos_cos_2x_l116_116311

theorem solve_arccos_cos_2x (x : ℝ) (k : ℤ) : 
  (arccos (cos (2 * x)) = x) →
  (x = 2 * k * π) ∨ (x = (2 * k * π) + (2 * π / 3)) ∨ (x = (2 * k * π) - (2 * π / 3)) := by
  sorry

end solve_arccos_cos_2x_l116_116311


namespace opposite_of_negative_2023_l116_116790

theorem opposite_of_negative_2023 : (- -2023) = 2023 := 
by {
sorry
}

end opposite_of_negative_2023_l116_116790


namespace incorrect_analogical_reasoning_l116_116397

-- Define the associative law for multiplication of real numbers
def real_mul_associative (a b c : ℝ) : (a * b) * c = a * (b * c) := sorry

-- Define the scalar product operation and its property
def scalar_product (a b c : ℝ) : Prop := 
  ((a * b) * c ≠ a * (b * c))

-- Define perpendicular lines in a plane and their parallelism
def perpendicular_plane (l1 l2 l3 : ℝ) : Prop :=
  (l1 ⊥ l3) ∧ (l2 ⊥ l3) → (l1 ∥ l2)

-- Define perpendicular lines in space and their non-parallelism
def perpendicular_space (l1 l2 l3 : ℝ) : Prop :=
  (l1 ⊥ l3) ∧ (l2 ⊥ l3) → ¬ (l1 ∥ l2)

-- Theorem to prove that options ① and ③ are incorrect
theorem incorrect_analogical_reasoning : 
  (scalar_product 1 2 3) ∧ (perpendicular_space 1 2 3) := sorry

end incorrect_analogical_reasoning_l116_116397


namespace Euler_identity_correct_l116_116320

theorem Euler_identity_correct (x : ℝ) : 
  (cos x + complex.I * sin x) ^ 2 = cos (2 * x) + complex.I * sin (2 * x) := 
by 
  sorry

end Euler_identity_correct_l116_116320


namespace opposite_of_neg2023_l116_116987

theorem opposite_of_neg2023 : ∀ (x : Int), -2023 + x = 0 → x = 2023 :=
by
  intros x h
  sorry

end opposite_of_neg2023_l116_116987


namespace tetrahedron_triangle_sides_l116_116275
-- Import needed mathematical library

-- Definition of vertices and edges
variables (a b c d e f : ℝ)

-- Conditions with labels for each vertex
def vertex_sum_v1 : ℝ := a + b + c
def vertex_sum_v2 : ℝ := b + d + f
def vertex_sum_v3 : ℝ := c + d + e
def vertex_sum_v4 : ℝ := a + e + f

-- Main statement asserting the equality of all vertex sums and proving triangle inequality for each vertex
theorem tetrahedron_triangle_sides :
  vertex_sum_v1 a b c = vertex_sum_v2 b d f ∧
  vertex_sum_v1 a b c = vertex_sum_v3 c d e ∧
  vertex_sum_v1 a b c = vertex_sum_v4 a e f →
  (a + b > c ∧ b + c > a ∧ a + c > b) ∧
  (b + d > f ∧ d + f > b ∧ b + f > d) ∧
  (c + d > e ∧ d + e > c ∧ c + e > d) ∧
  (a + e > f ∧ e + f > a ∧ a + f > e) :=
begin
  -- to be proved
  sorry
end

end tetrahedron_triangle_sides_l116_116275


namespace arithmetic_mean_difference_l116_116190

theorem arithmetic_mean_difference (p q r : ℝ)
  (h1 : (p + q) / 2 = 10)
  (h2 : (q + r) / 2 = 20) : 
  r - p = 20 := 
by sorry

end arithmetic_mean_difference_l116_116190


namespace circle_circumference_l116_116007

theorem circle_circumference (π : Real) (diameter : Real) (hπ : π ≈ 3.14159) (hdiameter : diameter = 2) :
  π * diameter ≈ 6.28318 :=
by
  rw [hdiameter]
  sorry

end circle_circumference_l116_116007


namespace opposite_of_neg_2023_l116_116888

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116888


namespace opposite_of_neg_2023_l116_116953

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l116_116953


namespace A_share_in_profit_l116_116003

-- Define the investments of A, B and C
def investment_A : ℕ := 6300
def investment_B : ℕ := 4200
def investment_C : ℕ := 10500

-- Define the total profit
def total_profit : ℕ := 12100

-- Define the total investment
def total_investment := investment_A + investment_B + investment_C

-- Define A's ratio of the total investment
def ratio_A := investment_A.to_rat / total_investment.to_rat

-- Define A's share in the profit
def share_A := (total_profit.to_rat * ratio_A).to_nat

-- The main statement to prove
theorem A_share_in_profit : share_A = 3630 :=
by
  sorry

end A_share_in_profit_l116_116003


namespace opposite_of_neg_2023_l116_116915

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by 
  sorry

end opposite_of_neg_2023_l116_116915


namespace ratio_area_smaller_sector_eof_to_circle_l116_116686

-- Definition of the given conditions
variables (O E F A B : Type) [circle O]
variables (E_on_circle : on_circle O E) (F_on_circle : on_circle O F)
variables (A_on_diameter : on_diameter O A) (B_on_diameter : on_diameter O B)
variables (AE_BF_same_side : same_side O E F A B)
variables (angle_AOE : angle O A E = 60)
variables (angle_FOB : angle O F B = 90)

-- Definition of the theorem to be proved
theorem ratio_area_smaller_sector_eof_to_circle :
  ratio_area EOF O = 1 / 12 :=
sorry

end ratio_area_smaller_sector_eof_to_circle_l116_116686


namespace order_of_x_y_z_l116_116544

noncomputable def x : ℝ := Real.log 3 / Real.log 2 - Real.log (Real.sqrt 3) / Real.log 2
noncomputable def y : ℝ := Real.log π / Real.log 0.5
noncomputable def z : ℝ := 0.9 ^ (-1.1)

theorem order_of_x_y_z : y < x ∧ x < z :=
by
  have x_simplified : x = Real.log (Real.sqrt 3) / Real.log 2 := by
    sorry  -- simplification steps
  have y_simplified : y = -Real.log π / Real.log 2 := by
    sorry  -- simplification steps
  have x_range : 0 < x ∧ x < 1 := by
    sorry  -- proof that 0 < x < 1
  have y_negative : y < 0 := by
    sorry  -- proof that y < 0
  have z_greater_1 : z > 1 := by
    sorry  -- proof that z > 1
  exact And.intro y_negative (And.intro x_range.right z_greater_1)

end order_of_x_y_z_l116_116544


namespace tan_add_pi_over_3_l116_116162

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + (Real.pi / 3)) = -(6 * Real.sqrt 3 + 2) / 13 := 
by
  sorry

end tan_add_pi_over_3_l116_116162


namespace opposite_neg_2023_l116_116759

theorem opposite_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_neg_2023_l116_116759


namespace line_through_point_with_opposite_intercepts_l116_116504

theorem line_through_point_with_opposite_intercepts :
  (∃ m : ℝ, (∀ x y : ℝ, y = m * x → (2,3) = (x, y)) ∧ ((∀ a : ℝ, a ≠ 0 → (x / a + y / (-a) = 1) → (2 - 3 = a ∧ a = -1)))) →
  ((∀ x y : ℝ, 3 * x - 2 * y = 0) ∨ (∀ x y : ℝ, x - y + 1 = 0)) :=
by
  sorry

end line_through_point_with_opposite_intercepts_l116_116504


namespace find_x_l116_116659

noncomputable def A (S : ℝ × list ℝ) : list ℝ :=
(list.zipWith (λ a b, (a + b) / 2) S.1 (S.2 ++ [S.1])).drop 1

noncomputable def A_power (S : list ℝ) (m : ℕ) : list ℝ :=
if m = 0 then S else A (S.head, A_power S (m - 1))

theorem find_x {x : ℝ} (hx : x > 0)
  (hS : ∃ S : list ℝ, S = (1 :: list.range 150₀).map (λ n, x^n))
  (hA100 : A_power (1 :: (list.range 150₀).map (λ n, x^n)) 100 = [1 / 2^75]) :
  x = 2^(25 / 51) - 1 :=
sorry

end find_x_l116_116659


namespace opposite_neg_2023_l116_116772

theorem opposite_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_neg_2023_l116_116772


namespace determinant_of_tangent_triangle_is_neg_one_l116_116665

noncomputable def determinant_tangent_triangle : Real :=
  Matrix.det !![!![Real.tan (Real.pi / 4), 1, 1], !![1, Real.tan (75 * Real.pi / 180), 1], !![1, 1, Real.tan (Real.pi / 3)]]

theorem determinant_of_tangent_triangle_is_neg_one :
  determinant_tangent_triangle = -1 :=
sorry

end determinant_of_tangent_triangle_is_neg_one_l116_116665


namespace opposite_neg_2023_l116_116775

theorem opposite_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_neg_2023_l116_116775


namespace pascal_remaining_distance_l116_116695

variable (v : ℕ) (v_red : ℕ) (v_inc : ℕ) (t : ℕ) (d : ℕ)

def remaining_distance_proof : Prop :=
  v = 8 ∧
  v_red = 4 ∧
  v_inc = 12 ∧
  d = v * t ∧ 
  d = v_red * (t + 16) ∧ 
  d = v_inc * (t - 16) ∧ 
  d = 256

theorem pascal_remaining_distance (v : ℕ) (v_red : ℕ) (v_inc : ℕ) (t : ℕ) (d : ℕ) 
  (h1 : v = 8)
  (h2 : v_red = 4)
  (h3 : v_inc = 12)
  (h4 : d = v * t)
  (h5 : d = v_red * (t + 16))
  (h6 : d = v_inc * (t - 16)) : 
  d = 256 :=
by {
  -- Initial setup
  have h7 : d = 4 * (t + 16) := h5,
  have h8 : d = 12 * (t - 16) := h6,
  have h9 : 4 * (t + 16) = 12 * (t - 16), from sorry,
  -- Solve the equation 
  have h10 : 4 * t + 64 = 12 * t - 192, from sorry,
  have h11 : 4 * t + 256 = 12 * t, from sorry,
  have h12 : 8 * t = 256, from sorry,
  have ht : t = 32, from sorry,
  -- Compute d
  have hd : d = 8 * t, from h4,
  rw ht at hd,
  exact hd.symm,
}

end pascal_remaining_distance_l116_116695


namespace opposite_of_neg_2023_l116_116833

/-- Define the opposite of a number -/
def opposite (x : ℤ) : ℤ := -x

/-- Prove the opposite of -2023 is 2023 -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116833


namespace equilateral_triangle_AXZ_l116_116750

theorem equilateral_triangle_AXZ (A B C W X Y Z : Type) 
  [equilateral_triangle A B C] 
  [square A W X B] 
  [square A Y Z C] : 
  equilateral_triangle A X Z := 
sorry

end equilateral_triangle_AXZ_l116_116750


namespace number_of_boys_l116_116603

theorem number_of_boys (n_total n_girls n_boys : ℕ) (h_total : n_total = 160) (h_ratio : n_girls = n_total / 4) (h_boys : n_boys = n_total - n_girls) : n_boys = 120 :=
by
  rw [h_total, h_ratio, h_boys]
  norm_num
  sorry

end number_of_boys_l116_116603


namespace determine_range_of_a_l116_116120

def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log (x + 1) - (1 / 2) * x^2

theorem determine_range_of_a (a : ℝ) (p q : ℝ) 
  (hpq : 0 < p ∧ p < 1 ∧ 0 < q ∧ q < 1 ∧ p ≠ q)
  (hineq : (f a (p + 1) - f a (q + 1)) / (p - q) > 3) : 
  a ≥ 15 :=
sorry

end determine_range_of_a_l116_116120


namespace nested_sqrt_inequality_l116_116627

theorem nested_sqrt_inequality (n : ℕ) (h : n ≥ 2) : 
  sqrt (2 * sqrt (3 * sqrt (4 * ... sqrt n))) < 3 := 
sorry

end nested_sqrt_inequality_l116_116627


namespace planting_trees_equation_l116_116414

noncomputable theory
open_locale classical

-- Defining the conditions from part a)
def total_students : ℕ := 20
def total_seedlings : ℕ := 52
def seedlings_per_male : ℕ := 3
def seedlings_per_female : ℕ := 2
variable (x : ℕ) -- number of male students

-- The theorem states that the equation represents the conditions
theorem planting_trees_equation :
  seedlings_per_male * x + seedlings_per_female * (total_students - x) = total_seedlings :=
sorry

end planting_trees_equation_l116_116414


namespace log_simplification_l116_116043

theorem log_simplification :
  (log 10 8 + 3 * log 10 2 - 2 * log 5 25 + log 2 16 = log 10 64) :=
by
  sorry

end log_simplification_l116_116043


namespace opposite_of_neg_2023_l116_116805

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by sorry

end opposite_of_neg_2023_l116_116805


namespace original_area_of_triangle_l116_116331

theorem original_area_of_triangle (A : ℝ) (h1 : 4 * A * 16 = 64) : A = 4 :=
by
  sorry

end original_area_of_triangle_l116_116331


namespace inscribed_triangle_hexagon_area_l116_116296

theorem inscribed_triangle_hexagon_area (r : ℝ) :
  let triangle_area := (sqrt 3 / 4) * r^2 in
  let hexagon_area := 6 * ((sqrt 3 / 4) * r^2) in
  triangle_area = hexagon_area / 2 :=
by
  sorry

end inscribed_triangle_hexagon_area_l116_116296


namespace f_strictly_increasing_l116_116133

def integer_part (x : ℝ) : ℤ := int.floor x

def fractional_part (x : ℝ) : ℝ := x - (integer_part x).toReal

noncomputable def f (x : ℝ) : ℝ := (integer_part x).toReal + real.sqrt (fractional_part x)

theorem f_strictly_increasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂ :=
by
  intro x1 x2 h
  sorry

end f_strictly_increasing_l116_116133


namespace opposite_of_neg_2023_l116_116958

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l116_116958


namespace product_of_7_and_sum_l116_116417

theorem product_of_7_and_sum (x : ℕ) (h : 27 - 7 = x * 5) : 7 * (x + 5) = 63 :=
by {
  have hx : x = 4,
  { 
    nlinarith, -- Light automation to find x = 4 from the equation 20 = x * 5.
  },
  -- Substitution of x = 4 into 7 * (x + 5)
  rw hx,
  norm_num, -- Simplifying to get 63
  sorry -- skipping the actual simplification line for brevity
}

end product_of_7_and_sum_l116_116417


namespace find_b_l116_116530

def vector_diff (p1 p2 : ℝ × ℝ) : ℝ × ℝ := (p2.1 - p1.1, p2.2 - p1.2)

theorem find_b :
  ∀ (p1 p2 : ℝ × ℝ) (b : ℝ),
  p1 = (-3, 4) →
  p2 = (2, -1) →
  vector_diff p1 p2 = (b, -1) →
  b = 1 :=
by
  intros p1 p2 b hp1 hp2 hv
  have h1 := congr_arg (λ t : ℝ × ℝ, t.1) hv
  simp at h1
  have h2 := congr_arg (λ t : ℝ × ℝ, t.2) hv
  simp at h2
  assumption

end find_b_l116_116530


namespace opposite_of_neg_2023_l116_116873

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116873


namespace remaining_distance_proof_l116_116708

-- Define the conditions
def pascal_current_speed : ℝ := 8
def pascal_reduced_speed : ℝ := pascal_current_speed - 4
def pascal_increased_speed : ℝ := pascal_current_speed * 1.5

-- Define the remaining distance in terms of the current speed and time taken
noncomputable def remaining_distance (T : ℝ) : ℝ := pascal_current_speed * T

-- Define the times with the increased and reduced speeds
noncomputable def time_with_increased_speed (T : ℝ) : ℝ := T - 16
noncomputable def time_with_reduced_speed (T : ℝ) : ℝ := T + 16

-- Define the distances using increased and reduced speeds
noncomputable def distance_increased_speed (T : ℝ) : ℝ := pascal_increased_speed * (time_with_increased_speed T)
noncomputable def distance_reduced_speed (T : ℝ) : ℝ := pascal_reduced_speed * (time_with_reduced_speed T)

-- Main theorem stating that the remaining distance is 256 miles
theorem remaining_distance_proof (T : ℝ) (ht_eq: pascal_current_speed * T = 256) : 
  remaining_distance T = 256 := by
  sorry

end remaining_distance_proof_l116_116708


namespace max_airline_companies_l116_116334

theorem max_airline_companies (n : ℕ) (h : n = 127) :
  let k := n * (n - 1) / 2 in k / n/ 2 = 63 :=
by
  have h2 : n * (n - 1) = 127 * 126 := sorry
  have h3 : 127*126/2 = 63 := sorry
  rw [h3]

end max_airline_companies_l116_116334


namespace circle_line_distance_properties_l116_116548

theorem circle_line_distance_properties :
  ∀ (P : ℝ × ℝ), (P.1 - 3)^2 + (P.2 - 3)^2 = 4 →
  let C := ((3, 3), 2) in 
  let AB := (1, 1, -2) in
  ∃ (d : ℝ), 
    d > 2 ∧ -- Line AB is disjoint from circle C
    d - 2 > 1/2 ∧ -- The distance from point P to line AB is greater than 1/2
    d + 2 < 5     -- The distance from point P to line AB is less than 5
  :=
begin
  intros P hP,
  let C := ((3: ℝ, 3: ℝ), 2: ℝ),
  let AB := (1: ℝ, 1: ℝ, -2: ℝ),
  use dist_from_point_to_line (3, 3) AB,
  split,
  { sorry }, -- proof that line AB is disjoint from circle C
  split,
  { sorry }, -- proof that the distance from P to line AB is greater than 1/2
  { sorry }  -- proof that the distance from P to line AB is less than 5
end

end circle_line_distance_properties_l116_116548


namespace find_m_l116_116529

def line_eq (x y : ℝ) : Prop := x + 2 * y - 3 = 0

def circle_eq (x y m : ℝ) : Prop := x * x + y * y + x - 6 * y + m = 0

def perpendicular_vectors (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

theorem find_m (m : ℝ) :
  (∃ (x y : ℝ), line_eq x y ∧ line_eq (3 - 2 * y) y ∧ circle_eq x y m ∧ circle_eq (3 - 2 * y) y m) ∧
  (∃ (x1 y1 x2 y2 : ℝ), line_eq x1 y1 ∧ line_eq x2 y2 ∧ perpendicular_vectors x1 y1 x2 y2) → m = 3 :=
sorry

end find_m_l116_116529


namespace find_x_l116_116187

theorem find_x (x y : ℝ) (h1 : x ≠ 0) (h2 : x / 3 = y^2) (h3 : x / 5 = 5 * y + 2) :
  x = (685 + 25 * Real.sqrt 745) / 6 :=
by
  sorry

end find_x_l116_116187


namespace opposite_of_negative_2023_l116_116782

theorem opposite_of_negative_2023 : (- -2023) = 2023 := 
by {
sorry
}

end opposite_of_negative_2023_l116_116782


namespace opposite_of_neg_2023_l116_116970

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l116_116970


namespace opposite_of_neg_2023_l116_116936

theorem opposite_of_neg_2023 : ∃ x : ℤ, -2023 + x = 0 ∧ x = 2023 := 
by 
  use 2023
  constructor
  · exact rfl
  · exact rfl

end opposite_of_neg_2023_l116_116936


namespace simplify_expression_l116_116056

theorem simplify_expression :
  (3 * (Real.sqrt 3 + Real.sqrt 5)) / (4 * Real.sqrt (3 + Real.sqrt 4))
  = (3 * Real.sqrt 15 + 3 * Real.sqrt 5) / 20 :=
by
  sorry

end simplify_expression_l116_116056


namespace find_value_of_abc_cubed_l116_116661

-- Variables and conditions
variables {a b c : ℝ}
variables (h₁ : a + b + c = 0) (h₂ : a^2 + b^2 + c^2 = a^4 + b^4 + c^4)

-- The statement
theorem find_value_of_abc_cubed (ha : a ≠ 0) (hb: b ≠ 0) (hc: c ≠ 0) :
  a^3 + b^3 + c^3 = -3 * a * b * (a + b) :=
by
  sorry

end find_value_of_abc_cubed_l116_116661


namespace shift_graph_sin2x_sqrt3cos2x_l116_116335

theorem shift_graph_sin2x_sqrt3cos2x :
  ∀ x : ℝ, (sin (2 * x) - sqrt 3 * cos (2 * x)) = 2 * sin (2 * (x - π / 6)) := 
sorry

end shift_graph_sin2x_sqrt3cos2x_l116_116335


namespace question1_question2_l116_116564

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * Real.log x - Real.exp 1

theorem question1 (a : ℝ) :
  (∀ x : ℝ, x > 0 → f(x, a).deriv = 0 → x = 1) → a = Real.exp 1 ∧ 
  (∀ x : ℝ, 0 < x < 1 → f(x, a).deriv < 0) ∧ 
  (∀ x : ℝ, 1 < x → f(x, a).deriv > 0) :=
by
  sorry

theorem question2 (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x → f(x, a) ≥ 0) → a ≤ Real.exp 1 :=
by
  sorry

end question1_question2_l116_116564


namespace opposite_of_neg_2023_l116_116847

def opposite (n : ℤ) : ℤ :=
  -n

theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116847


namespace tan_X_l116_116225

noncomputable def triangle_XYZ := 
  {X Y Z : Type} [euclidean_space X] [euclidean_space Y] [euclidean_space Z]
  [XZ_eq_sqrt_41 : XZ = sqrt 41] [YZ_eq_4 : YZ = 4] [angle_Y_90 : angle Y = 90]

theorem tan_X (triangle_XYZ) : 
  ∃ XY, (XY_squared : XY^2 = (XZ^2 - YZ^2)) ∧ (XY = 5) ∧ (tan (angle_const X) = 4/5) :=
by
  sorry

end tan_X_l116_116225


namespace opposite_neg_2023_l116_116762

theorem opposite_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_neg_2023_l116_116762


namespace monotonically_increasing_interval_l116_116119

noncomputable def increasing_interval (ω : ℝ) (φ : ℝ) := 
  {x : ℝ | -π/8 ≤ x ∧ x ≤ 3*π/8}

theorem monotonically_increasing_interval
  (ω φ : ℝ) (hω_pos : ω > 0)
  (h_symm_axes : ω = 2)
  (h_condition : cos (ω * x + φ) ≤ cos (ω * (-7*π/8) + φ))
  : increasing_interval ω φ = {x | -π/8 ≤ x ∧ x ≤ 3*π/8} :=
sorry

end monotonically_increasing_interval_l116_116119


namespace opposite_neg_2023_l116_116770

theorem opposite_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_neg_2023_l116_116770


namespace non_decreasing_paths_count_l116_116377

noncomputable def Catalan : ℕ → ℚ :=
  λ n, 1 / (n + 1) * (Nat.choose (2 * n) n)

theorem non_decreasing_paths_count (n : ℕ) : 
  (number_of_paths_from (0, 0) to (n, n) remaining_below_diagonal_y_eq_x) = Catalan (n + 1) := 
sorry

end non_decreasing_paths_count_l116_116377


namespace opposite_of_neg2023_l116_116997

theorem opposite_of_neg2023 : ∀ (x : Int), -2023 + x = 0 → x = 2023 :=
by
  intros x h
  sorry

end opposite_of_neg2023_l116_116997


namespace travel_time_comparison_l116_116399

-- Definitions for given conditions
variables {a b S : ℝ}

-- Condition variables
-- Speed of the boat in still water is greater than the speed of the river current.
axiom H1 : 0 < b
axiom H2 : b < a

-- Time taken for the round trip on the river
def T_river (a b S : ℝ) : ℝ := (2 * a * S) / (a^2 - b^2)

-- Time taken for the round trip on the lake in still water
def T_lake (a S : ℝ) : ℝ := (2 * S) / a

-- Prove that the time on the river is greater than the time on the lake
theorem travel_time_comparison (H1 : 0 < b) (H2 : b < a) (S : ℝ): T_river a b S > T_lake a S := by
  sorry

end travel_time_comparison_l116_116399


namespace divisor_of_a_l116_116255

namespace MathProofProblem

-- Define the given problem
variable (a b c d : ℕ) -- Variables representing positive integers

-- Given conditions
variables (h_gcd_ab : Nat.gcd a b = 30)
variables (h_gcd_bc : Nat.gcd b c = 42)
variables (h_gcd_cd : Nat.gcd c d = 66)
variables (h_lcm_cd : Nat.lcm c d = 2772)
variables (h_gcd_da : 100 < Nat.gcd d a ∧ Nat.gcd d a < 150)

-- Target statement to prove
theorem divisor_of_a : 13 ∣ a :=
by
  sorry

end MathProofProblem

end divisor_of_a_l116_116255


namespace opposite_of_neg_2023_l116_116983

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l116_116983


namespace get_6_on_10th_roll_is_random_event_l116_116442

-- Define the conditions
def die_rolls (n : ℕ) : Prop := ∀ i, (i < n) → (roll_die i ≠ 6)

-- The main theorem statement
theorem get_6_on_10th_roll_is_random_event : 
  die_rolls 9 → event_is_random (roll_die 10 = 6) :=
by
  sorry

end get_6_on_10th_roll_is_random_event_l116_116442


namespace interest_percent_is_correct_l116_116400

-- We first define the given conditions
def encyclopedia_cost : ℝ := 750
def down_payment : ℝ := 300
def monthly_payment : ℝ := 57
def final_payment : ℝ := 21
def number_of_payments : ℕ := 9

-- Calculate intermediary values based on the conditions
def total_paid_in_installments : ℝ :=
  (number_of_payments * monthly_payment) + final_payment

def total_paid : ℝ := down_payment + total_paid_in_installments

def amount_borrowed : ℝ := encyclopedia_cost - down_payment

def interest_paid : ℝ := total_paid - amount_borrowed

def interest_percent : ℝ := (interest_paid / amount_borrowed) * 100

-- Finally, we state the proof problem:
theorem interest_percent_is_correct : 
  interest_percent = 85.33 := by
  sorry

end interest_percent_is_correct_l116_116400


namespace solution_set_inequality_l116_116037

noncomputable def f : ℝ → ℝ := sorry

axiom f_condition1 : f 1 = 1
axiom f_condition2 : ∀ x : ℝ, deriv f x < 1 / 2

theorem solution_set_inequality : {x : ℝ | 0 < x ∧ x < 2} = {x : ℝ | f (Real.log x / Real.log 2) > (Real.log x / Real.log 2 + 1) / 2} :=
by
  sorry

end solution_set_inequality_l116_116037


namespace cos_double_angle_zero_l116_116542

theorem cos_double_angle_zero 
  (α β : ℝ) 
  (h1 : sin α = sqrt 5 / 5)
  (h2 : sin (α - β) = -sqrt 10 / 10)
  (h3 : 0 < α ∧ α < π / 2)
  (h4 : 0 < β ∧ β < π / 2) : 
  cos (2 * β) = 0 := 
by 
  sorry

end cos_double_angle_zero_l116_116542


namespace opposite_of_neg_2023_l116_116855

-- Definition of the problem
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  -- The proof is to be provided
  sorry

end opposite_of_neg_2023_l116_116855


namespace diminished_gcd_equals_100_l116_116041

theorem diminished_gcd_equals_100 : Nat.gcd 7800 360 - 20 = 100 := by
  sorry

end diminished_gcd_equals_100_l116_116041


namespace smallest_positive_sum_of_products_l116_116486

theorem smallest_positive_sum_of_products :
  ∀ (b : Fin 83 → ℤ), (∀ i, b i = 1 ∨ b i = -1) →
  let s := (∑ i in Finset.univ, b i) in
  let T := (∑ i in Finset.range 83, ∑ j in Finset.Ico i (83) if i ≠ j then b i * b j else 0) in
  ∃ (T : ℤ), T = 19 :=
by {
  intro b h,
  let s := (∑ i in Finset.univ, b i),
  let T := (∑ i in Finset.range 83, ∑ j in Finset.Ico i (83) if i ≠ j then b i * b j else 0),
  use 19,
  sorry
}

end smallest_positive_sum_of_products_l116_116486


namespace maria_savings_l116_116675

-- Conditions
def sweater_cost : ℕ := 30
def scarf_cost : ℕ := 20
def num_sweaters : ℕ := 6
def num_scarves : ℕ := 6
def savings : ℕ := 500

-- The proof statement
theorem maria_savings : savings - (num_sweaters * sweater_cost + num_scarves * scarf_cost) = 200 :=
by
  sorry

end maria_savings_l116_116675


namespace area_triangle_KDC_l116_116218

theorem area_triangle_KDC
    (r : ℝ) (KA : ℝ) (CD : ℝ) (length_KA : KA = 20)
    (radius_O : r = 10) (chord_CD : CD = 12)
    (collinear_KAOB : collinear({k, a, o, b} : Set ℝ)) 
    (O_KA_CD_parallel : parallel(k, b, c, d)) :
    ∃ area : ℝ, area = 48 :=
by
  sorry

end area_triangle_KDC_l116_116218


namespace opposite_of_neg_2023_l116_116823

/-- Define the opposite of a number -/
def opposite (x : ℤ) : ℤ := -x

/-- Prove the opposite of -2023 is 2023 -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116823


namespace Q_one_over_neg_one_l116_116344

noncomputable def g : Polynomial ℤ := Polynomial.monomial 2009 1 + Polynomial.monomial 2008 19 + Polynomial.C 1

variables {s : Fin 2009 → ℂ} (hs : ∀ i j : Fin 2009, i ≠ j → s i ≠ s j) 
variable (Q : Polynomial ℂ)

theorem Q_one_over_neg_one :
  (∀ i, Q (s i + 1 / s i) = 0) →
  Q.eval 1 / Q.eval (-1) = 361 / 331 :=
by
  intro hQ
  sorry

end Q_one_over_neg_one_l116_116344


namespace circumscribed_circle_radius_squared_l116_116127

-- Given a triangle ABC with AB = 12, BC = 10, and ∠ABC = 120°, prove that the square of the radius 
-- of the smallest circumscribing circle is 91.

theorem circumscribed_circle_radius_squared
  (A B C : Type) [InnerProductSpace ℝ A]
  (h_ab : dist A B = 12) 
  (h_bc : dist B C = 10)
  (h_angle_abc : angle A B C = 2 * Real.pi / 3) :
  let R := (dist A C) / 2 in R ^ 2 = 91 :=
by
  sorry

end circumscribed_circle_radius_squared_l116_116127


namespace paul_account_balance_after_transactions_l116_116723

theorem paul_account_balance_after_transactions :
  let transfer1 := 90
  let transfer2 := 60
  let service_charge_percent := 2 / 100
  let initial_balance := 400
  let service_charge1 := service_charge_percent * transfer1
  let service_charge2 := service_charge_percent * transfer2
  let total_deduction1 := transfer1 + service_charge1
  let total_deduction2 := transfer2 + service_charge2
  let reversed_transfer2 := total_deduction2 - transfer2
  let balance_after_transfer1 := initial_balance - total_deduction1
  let final_balance := balance_after_transfer1 - reversed_transfer2
  (final_balance = 307) :=
by
  sorry

end paul_account_balance_after_transactions_l116_116723


namespace general_formula_range_of_a_infinite_geometric_sequences_l116_116090

-- Definitions for the sequence and its initial conditions
def sequence_a (a : ℝ) (n : ℕ) : ℝ :=
  if n = 1 then a else if n % 2 = 1 then 6 * (n + 1) / 2 + a - 6 else 6 * n - 1

-- Definition of the sum \(S_n\) of the first \(n\) terms
def sum_S (a : ℝ) (n : ℕ) : ℝ :=
  (∑ i in range (n + 1), sequence_a a i)

-- Conditions and the problem statements
theorem general_formula (a : ℝ) (n : ℕ) (hn : n > 0) :
  (sequence_a a) n = 
    if n % 2 = 1 then 6 * (n + 1) / 2 + a - 6 else 6 * n - 1 :=
sorry

theorem range_of_a (a : ℝ) (hn : 0 < n) :
  (sum_S a) n ≤ n * (3 * n + 1) → 0 < a ∧ a ≤ 4 :=
sorry

theorem infinite_geometric_sequences (a : ℝ) (h : a = 2) (k : ℕ) :
  ∃ q (hn : q > 1), infinite (filter_seq k (sequence_a a)) ∧ is_geometric_seq (filter_seq k (sequence_a a)) q :=
sorry

end general_formula_range_of_a_infinite_geometric_sequences_l116_116090


namespace pascal_sum_difference_l116_116569

noncomputable def binomial : ℕ → ℕ → ℝ
| n, k => (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k)))

noncomputable def a_i (i : ℕ) : ℝ := binomial 101 i

noncomputable def b_i (i : ℕ) : ℝ := binomial 102 i

noncomputable def c_i (i : ℕ) : ℝ := binomial 103 i

theorem pascal_sum_difference : 
  (∑ i in finset.range 102, b_i i / c_i i) - (∑ i in finset.range 101, a_i i / b_i i) = 0.7491 := 
sorry

end pascal_sum_difference_l116_116569


namespace pascal_remaining_miles_l116_116704

section PascalCyclingTrip

variable (D T : ℝ)
variable (V : ℝ := 8)
variable (V_reduced V_increased : ℝ)
variable (time_diff : ℝ := 16)

-- Definitions of reduced and increased speeds
def V_reduced := V - 4
def V_increased := V + 0.5 * V

-- Conditions for the distance
def distance_current := D = V * T
def distance_increased := D = V_increased * (T - time_diff)
def distance_reduced := D = V_reduced * (T + time_diff)

theorem pascal_remaining_miles (h1 : distance_current)
  (h2 : distance_increased)
  (h3 : distance_reduced) : D = 256 := 
sorry

end PascalCyclingTrip

end pascal_remaining_miles_l116_116704


namespace john_moves_540kg_l116_116234

-- Conditions
def used_to_back_squat : ℝ := 200
def increased_by : ℝ := 50
def front_squat_ratio : ℝ := 0.8
def triple_ratio : ℝ := 0.9

-- Definitions based on conditions
def current_back_squat : ℝ := used_to_back_squat + increased_by
def current_front_squat : ℝ := front_squat_ratio * current_back_squat
def one_triple : ℝ := triple_ratio * current_front_squat
def three_triples : ℝ := 3 * one_triple

-- The proof statement
theorem john_moves_540kg : three_triples = 540 := by
  sorry

end john_moves_540kg_l116_116234


namespace smiths_bakery_pies_l116_116303

theorem smiths_bakery_pies (m : ℕ) (h : m = 16) : 
  let s := 4 * m + 6 in 
  s = 70 := 
by
  unfold s
  sorry

end smiths_bakery_pies_l116_116303


namespace opposite_of_neg_2023_l116_116852

def opposite (n : ℤ) : ℤ :=
  -n

theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116852


namespace smiths_bakery_pies_l116_116308

theorem smiths_bakery_pies (m : ℕ) (h : m = 16) : 4 * m + 6 = 70 :=
by
  rw [h]
  norm_num
  sorry

end smiths_bakery_pies_l116_116308


namespace opposite_of_neg_2023_l116_116797

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by sorry

end opposite_of_neg_2023_l116_116797


namespace opposite_of_neg_2023_l116_116976

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l116_116976


namespace brazil_championship_probability_l116_116202

noncomputable theory
open_locale classical

-- Define the conditions: probabilities, points system, number of matches, advancing criteria, penalties
def probability_win_brazil := (1 : ℝ) / 2
def probability_draw_brazil := (1 : ℝ) / 3
def probability_lose_brazil := (1 : ℝ) / 6
def points_for_win := 3
def points_for_draw := 1
def points_for_loss := 0
def num_group_stage_matches := 3
def min_points_to_advance := 4
def probability_win_penalty := (3 : ℝ) / 5

-- Define the proposition: the probability of Brazil winning the championship with exactly one match going to penalties
theorem brazil_championship_probability :
  let ξ := 4 +
            -- Group stage points calculation would go here
          in
  -- Probability calculations for advancing and winning the championship with penalties would go here
  P(win_championship_with_one_penalty ξ) = (1 : ℝ) / 12 := sorry

end brazil_championship_probability_l116_116202


namespace min_max_sum_l116_116508

theorem min_max_sum (a b c d e f g : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) 
  (h4 : d ≥ 0) (h5 : e ≥ 0) (h6 : f ≥ 0) (h7 : g ≥ 0) 
  (h_sum : a + b + c + d + e + f + g = 1) : 
  ∃ (p : ℝ), p = max (a + b + c) (max (b + c + d) (max (c + d + e) (max (d + e + f) (e + f + g)))) ∧ p = 1 / 3 :=
  sorry

end min_max_sum_l116_116508


namespace opposite_of_neg_2023_l116_116962

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l116_116962


namespace opposite_of_neg_2023_l116_116957

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l116_116957


namespace tan_A_in_right_triangle_l116_116494

theorem tan_A_in_right_triangle
  (A B C : Type)
  [triangle : geometry.Triangle A B C]
  (rt : geometry.RightTriangle A B C)
  (AB BC : ℝ) (AB_eq : AB = 40) (BC_eq : BC = 41) :
  geometry.tan A B C = 9 / 40 :=
sorry

end tan_A_in_right_triangle_l116_116494


namespace pascal_remaining_distance_l116_116692

variable (v : ℕ) (v_red : ℕ) (v_inc : ℕ) (t : ℕ) (d : ℕ)

def remaining_distance_proof : Prop :=
  v = 8 ∧
  v_red = 4 ∧
  v_inc = 12 ∧
  d = v * t ∧ 
  d = v_red * (t + 16) ∧ 
  d = v_inc * (t - 16) ∧ 
  d = 256

theorem pascal_remaining_distance (v : ℕ) (v_red : ℕ) (v_inc : ℕ) (t : ℕ) (d : ℕ) 
  (h1 : v = 8)
  (h2 : v_red = 4)
  (h3 : v_inc = 12)
  (h4 : d = v * t)
  (h5 : d = v_red * (t + 16))
  (h6 : d = v_inc * (t - 16)) : 
  d = 256 :=
by {
  -- Initial setup
  have h7 : d = 4 * (t + 16) := h5,
  have h8 : d = 12 * (t - 16) := h6,
  have h9 : 4 * (t + 16) = 12 * (t - 16), from sorry,
  -- Solve the equation 
  have h10 : 4 * t + 64 = 12 * t - 192, from sorry,
  have h11 : 4 * t + 256 = 12 * t, from sorry,
  have h12 : 8 * t = 256, from sorry,
  have ht : t = 32, from sorry,
  -- Compute d
  have hd : d = 8 * t, from h4,
  rw ht at hd,
  exact hd.symm,
}

end pascal_remaining_distance_l116_116692


namespace opposite_of_neg_2023_l116_116924

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by 
  sorry

end opposite_of_neg_2023_l116_116924


namespace reciprocal_of_point_b_l116_116281

theorem reciprocal_of_point_b : 
  ∀ (A B : ℤ), A = -3 → B = A + 4 → (1 : ℝ) / (B : ℝ) = 1 :=
by
  intros A B hA hB
  rw [hA, hB]
  sorry

end reciprocal_of_point_b_l116_116281


namespace opposite_of_neg_2023_l116_116918

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by 
  sorry

end opposite_of_neg_2023_l116_116918


namespace boxes_with_neither_l116_116635

-- Definitions translating the conditions from the problem
def total_boxes : Nat := 15
def boxes_with_markers : Nat := 8
def boxes_with_crayons : Nat := 4
def boxes_with_both : Nat := 3

-- The theorem statement to prove
theorem boxes_with_neither : total_boxes - (boxes_with_markers + boxes_with_crayons - boxes_with_both) = 6 := by
  -- Proof will go here
  sorry

end boxes_with_neither_l116_116635


namespace arrangement_ways_13_books_arrangement_ways_13_books_with_4_arithmetic_together_arrangement_ways_13_books_with_4_arithmetic_6_algebra_together_arrangement_ways_13_books_with_4_arithmetic_6_algebra_3_geometry_together_l116_116584

-- Statement for Question 1
theorem arrangement_ways_13_books : 
  (Nat.factorial 13) = 6227020800 := 
sorry

-- Statement for Question 2
theorem arrangement_ways_13_books_with_4_arithmetic_together :
  (Nat.factorial 10) * (Nat.factorial 4) = 87091200 := 
sorry

-- Statement for Question 3
theorem arrangement_ways_13_books_with_4_arithmetic_6_algebra_together :
  (Nat.factorial 5) * (Nat.factorial 4) * (Nat.factorial 6) = 2073600 := 
sorry

-- Statement for Question 4
theorem arrangement_ways_13_books_with_4_arithmetic_6_algebra_3_geometry_together :
  (Nat.factorial 3) * (Nat.factorial 4) * (Nat.factorial 6) * (Nat.factorial 3) = 622080 := 
sorry

end arrangement_ways_13_books_arrangement_ways_13_books_with_4_arithmetic_together_arrangement_ways_13_books_with_4_arithmetic_6_algebra_together_arrangement_ways_13_books_with_4_arithmetic_6_algebra_3_geometry_together_l116_116584


namespace largest_divisor_l116_116667

/-- Given an integer n > 3, let S be a set of n integers. Prove that the largest integer d such 
that there are four distinct nonempty subsets of S whose sums are divisible by d is n-2. -/
theorem largest_divisor (n : ℕ) (S : Finset ℤ) (h : ∀ s, s ∈ S → s ≤ n) (h_n : 3 < n) : 
  ∃ d, ∀ {S : Finset ℤ} (hcard : S.card = n), ( ∀ T₁ T₂ T₃ T₄ : Finset ℤ, 
      T₁ ≠ ∅ → T₂ ≠ ∅ → T₃ ≠ ∅ → T₄ ≠ ∅ → 
      T₁ ≠ T₂ → T₁ ≠ T₃ → T₁ ≠ T₄ → T₂ ≠ T₃ → T₂ ≠ T₄ → T₃ ≠ T₄ → 
      T₁.sum % d = 0 → T₂.sum % d = 0 → T₃.sum % d = 0 → T₄.sum % d = 0) → 
  d = n - 2 := 
by
  sorry

end largest_divisor_l116_116667


namespace opposite_of_neg_2023_l116_116880

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116880


namespace total_doors_needed_correct_l116_116428

-- Define the conditions
def buildings : ℕ := 2
def floors_per_building : ℕ := 12
def apartments_per_floor : ℕ := 6
def doors_per_apartment : ℕ := 7

-- Define the total number of doors needed
def total_doors_needed : ℕ := buildings * floors_per_building * apartments_per_floor * doors_per_apartment

-- State the theorem to prove the total number of doors needed is 1008
theorem total_doors_needed_correct : total_doors_needed = 1008 := by
  sorry

end total_doors_needed_correct_l116_116428


namespace evaluate_expression_l116_116491

theorem evaluate_expression (d : ℕ) (h : d = 4) :
  (d^d - d * (d - 2)^d + (d - 1)!)^2 = 39204  :=
by
  sorry

end evaluate_expression_l116_116491


namespace num_seven_digit_palindromes_l116_116064

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def is_seven_digit_palindrome (x : ℕ) : Prop :=
  let a := x / 1000000 % 10 in
  let b := x / 100000 % 10 in
  let c := x / 10000 % 10 in
  let d := x / 1000 % 10 in
  let e := x / 100 % 10 in
  let f := x / 10 % 10 in
  let g := x % 10 in
  a ≠ 0 ∧ a = g ∧ b = f ∧ c = e

theorem num_seven_digit_palindromes : 
  ∃ n : ℕ, (∀ x : ℕ, is_seven_digit_palindrome x → 1 ≤ a (x / 1000000 % 10) ∧ is_digit (b) ∧ is_digit (c) ∧ is_digit (d)) → n = 9000 :=
sorry

end num_seven_digit_palindromes_l116_116064


namespace tan_add_pi_over_3_l116_116177

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = - (6 + 5 * Real.sqrt 3) / 13 :=
sorry

end tan_add_pi_over_3_l116_116177


namespace eccentricity_of_hyperbola_l116_116122

theorem eccentricity_of_hyperbola (a b c : ℝ) (h_a : a > 0) (h_b : b > 0)
  (h_c : c = Real.sqrt (a^2 + b^2))
  (F1 : ℝ × ℝ := (-c, 0))
  (A B : ℝ × ℝ)
  (slope_of_AB : ∀ (x y : ℝ), y = x + c)
  (asymptotes_eqn : ∀ (x : ℝ), x = a ∨ x = -a)
  (intersections : A = (-(a * c / (a - b)), -(b * c / (a - b))) ∧ B = (-(a * c / (a + b)), (b * c / (a + b))))
  (AB_eq_2BF1 : 2 * (F1 - B) = A - B) :
  Real.sqrt (1 + (b / a)^2) = Real.sqrt 5 :=
sorry

end eccentricity_of_hyperbola_l116_116122


namespace complex_point_coords_l116_116216

theorem complex_point_coords :
  ∀ z : ℂ, z = (1 - 2 * complex.I) / (2 + complex.I) → z = 0 - complex.I :=
begin
  assume z,
  sorry
end

end complex_point_coords_l116_116216


namespace greatest_whole_number_ineq_solution_greatest_whole_number_result_greatest_whole_number_l116_116060

theorem greatest_whole_number_ineq (x : ℤ) (h : 5 * x - 4 < 3 - 2 * x) : x < 1 := 
by sorry

theorem solution_greatest_whole_number : ℤ := 
0

theorem result_greatest_whole_number : (h : 5 * solution_greatest_whole_number - 4 < 3 - 2 * solution_greatest_whole_number) := 
by 
  simp [solution_greatest_whole_number]
  norm_num

end greatest_whole_number_ineq_solution_greatest_whole_number_result_greatest_whole_number_l116_116060


namespace opposite_of_neg_2023_l116_116813

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by sorry

end opposite_of_neg_2023_l116_116813


namespace correct_number_of_propositions_l116_116097

noncomputable def proposition1 (m : Set Point) (n : Set Point) (α : Set (Set Point)) [Parallel m α] [Parallel n α] : Parallel m n := sorry

noncomputable def proposition2 (m : Set Point) (n : Set Point) (α : Set (Set Point)) [Parallel m α] [Perpendicular n α] : Perpendicular n m := sorry

noncomputable def proposition3 (m : Set Point) (α : Set (Set Point)) (β : Set (Set Point)) [Perpendicular m α] [Parallel m β] : Perpendicular α β := sorry

theorem correct_number_of_propositions (m n : Set Point) (α β : Set (Set Point)) :
  (∃ p1 : Parallel m α, ∃ p2 : Parallel n α, ¬Parallel m n → False) +
  (∃ p1 : Parallel m α, ∃ p2 : Perpendicular n α, Perpendicular n m → True) +
  (∃ p1 : Perpendicular m α, ∃ p2 : Parallel m β, Perpendicular α β → True) = 2 := sorry

end correct_number_of_propositions_l116_116097


namespace derivative_of_f_l116_116749

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) / x

theorem derivative_of_f :
  ∀ x ≠ 0, deriv f x = ((-x * Real.sin x - Real.cos x) / (x^2)) := sorry

end derivative_of_f_l116_116749


namespace number_of_ordered_pairs_l116_116477

noncomputable def harmonic_mean (x y : ℕ) : ℝ :=
  (2 * x * y) / (x + y)

theorem number_of_ordered_pairs :
  {n : ℕ // n = 20} = {n : ℕ // ∃ (x y : ℕ), 
  0 < x ∧ 0 < y ∧ x < y ∧ harmonic_mean x y = (5 : ℝ)^20}.card :=
by
  sorry

end number_of_ordered_pairs_l116_116477


namespace opposite_of_neg_2023_l116_116819

/-- Define the opposite of a number -/
def opposite (x : ℤ) : ℤ := -x

/-- Prove the opposite of -2023 is 2023 -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116819


namespace project_completion_time_l116_116229

theorem project_completion_time : 
  ∀ (d₁ d₂ : ℝ) (w₁ w₂ : ℝ) (eff_increase : ℝ)
  (total_work : ℝ),
  d₁ = 6 →
  w₁ = 120 →
  total_work = w₁ * d₁ →
  eff_increase = 0.2 →
  w₂ = 80 →
  d₂ = total_work / ((1 + eff_increase) * w₂) →
  d₂ = 7.5 :=
by
  intros d₁ d₂ w₁ w₂ eff_increase total_work h_d1 h_w1 h_total_work h_eff_increase h_w2 h_d2
  sorry

end project_completion_time_l116_116229


namespace tan_theta_plus_pi_over_eight_sub_inv_l116_116587

/-- Given the trigonometric identity, we can prove the tangent calculation -/
theorem tan_theta_plus_pi_over_eight_sub_inv (θ : ℝ)
  (h : 3 * Real.sin θ + Real.cos θ = Real.sqrt 10) :
  Real.tan (θ + Real.pi / 8) - 1 / Real.tan (θ + Real.pi / 8) = -14 := 
sorry

end tan_theta_plus_pi_over_eight_sub_inv_l116_116587


namespace profit_per_item_is_10_percent_of_P_l116_116485

-- Define the price of each item P as a positive real number
variable (P : ℝ) (P_pos : 0 < P)

-- Define the profit per item without discount
def profit_per_item_without_discount : ℝ := 0.10 * P

-- Define the number of items sold without discount and the total profit without discount
def items_sold_without_discount : ℕ := 100
def total_profit_without_discount : ℝ := items_sold_without_discount * profit_per_item_without_discount P

-- Define the profit per item with a 5% discount
def profit_per_item_with_discount : ℝ := 0.05 * P

-- Define the number of items sold with discount and total profit with discount
def items_sold_with_discount : ℕ := 222.22.toNat
def total_profit_with_discount : ℝ := items_sold_with_discount * profit_per_item_with_discount P

-- Theorem statement: profit per item is 10% of P given the conditions
theorem profit_per_item_is_10_percent_of_P :
  profit_per_item_without_discount P = 0.10 * P :=
by 
  sorry

end profit_per_item_is_10_percent_of_P_l116_116485


namespace find_x_l116_116496

open Real

theorem find_x (x : ℝ) (h : log 8 (3 * x - 4) = 2) : x = 68 / 3 := by
  sorry

end find_x_l116_116496


namespace circle_radius_of_tangency_l116_116366

-- Setting up the problem
def circle_eq (r : ℝ) (x y : ℝ): Prop :=
  (x - r)^2 + y^2 = r^2

def ellipse_eq (x y : ℝ) : Prop :=
  x^2 + 4y^2 = 5

theorem circle_radius_of_tangency :
  ∃ r : ℝ, 
  (∀ x y : ℝ, ellipse_eq x y → (circle_eq r x y) → x^2 + 4y^2 = 5) 
  → r = (Real.sqrt 15) / 4 :=
by
  /- Proof is omitted. -/
  sorry

end circle_radius_of_tangency_l116_116366


namespace opposite_of_neg_2023_l116_116911

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by 
  sorry

end opposite_of_neg_2023_l116_116911


namespace opposite_of_neg_2023_l116_116832

/-- Define the opposite of a number -/
def opposite (x : ℤ) : ℤ := -x

/-- Prove the opposite of -2023 is 2023 -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116832


namespace calculate_expression_l116_116247

def N (x : ℝ) : ℝ := 3 * real.sqrt x
def O (x : ℝ) : ℝ := x^2 + 1

theorem calculate_expression : 
  N(O(N(O(N(O(2)))))) = 3 * real.sqrt 415 := 
by
  sorry

end calculate_expression_l116_116247


namespace find_largest_n_l116_116310

theorem find_largest_n : ∃ n : ℕ, (∀ m : ℕ, ((2 * m : ℕ) = (m : ℕ) in base 7) → m ≤ n) ∧ n = 156 := 
by 
  sorry

end find_largest_n_l116_116310


namespace consecutive_odd_split_l116_116517

theorem consecutive_odd_split (m : ℕ) (hm : m > 1) : (∃ n : ℕ, n = 2015 ∧ n < ((m + 2) * (m - 1)) / 2) → m = 45 :=
by
  sorry

end consecutive_odd_split_l116_116517


namespace radius_circumcircle_l116_116372

variables (R1 R2 R3 : ℝ)
variables (d : ℝ)
variables (R : ℝ)

noncomputable def sum_radii := R1 + R2 = 11
noncomputable def distance_centers := d = 5 * Real.sqrt 17
noncomputable def radius_third_sphere := R3 = 8
noncomputable def touching := R1 + R2 + 2 * R3 = d

theorem radius_circumcircle :
  R = 5 * Real.sqrt 17 / 2 :=
  by
  -- Use conditions here if necessary
  sorry

end radius_circumcircle_l116_116372


namespace negation_relation_l116_116657

def p (x : ℝ) : Prop := x < -1 ∨ x > 1
def q (x : ℝ) : Prop := x < -2 ∨ x > 1

def not_p (x : ℝ) : Prop := x ≥ -1 ∧ x ≤ 1
def not_q (x : ℝ) : Prop := x ≥ -2 ∧ x ≤ 1

theorem negation_relation : (∀ x, not_p x → not_q x) ∧ ¬ (∀ x, not_q x → not_p x) :=
by 
  sorry

end negation_relation_l116_116657


namespace opposite_of_neg2023_l116_116999

theorem opposite_of_neg2023 : ∀ (x : Int), -2023 + x = 0 → x = 2023 :=
by
  intros x h
  sorry

end opposite_of_neg2023_l116_116999


namespace opposite_of_neg_2023_l116_116878

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116878


namespace tan_add_pi_over_3_l116_116164

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + (Real.pi / 3)) = -(6 * Real.sqrt 3 + 2) / 13 := 
by
  sorry

end tan_add_pi_over_3_l116_116164


namespace equilateral_triangle_partition_l116_116006

/-- An equilateral triangle partitioned into smaller equilateral triangular pieces
    must contain at least two pieces of the same size. -/
theorem equilateral_triangle_partition (T : Type) [equilateral_triangle T] 
  (partition : T → set T) : ∃ (t1 t2 : T), t1 ≠ t2 ∧ size t1 = size t2 :=
sorry

end equilateral_triangle_partition_l116_116006


namespace ratio_eight_to_sixteen_rounded_l116_116283

theorem ratio_eight_to_sixteen_rounded :
  (Real.floor ((8 / 16) * 10 + 0.5) / 10) = 0.5 :=
by
  sorry

end ratio_eight_to_sixteen_rounded_l116_116283


namespace subset_intersection_exists_l116_116214

theorem subset_intersection_exists (U : Finset ℕ) (hU : U.card = 11)
    (A : Finset (Finset ℕ)) (hA : A.card = 1024) :
  (∀ A_i ∈ A, A_i ⊆ U) ∧ 
  (∀ A_i A_j ∈ A, A_i ≠ A_j → (A_i ∩ A_j).nonempty) ∧
  (∀ B ⊆ U, B ∉ A → ∃ A_i ∈ A, (A_i ∩ B) = ∅) :=
sorry

end subset_intersection_exists_l116_116214


namespace dislikeBothRadioAndMusic_count_l116_116285

def totalPeople : ℕ := 1500
def dislikeRadioPercentage : ℚ := 0.25
def dislikeBothFromDislikeRadioPercentage : ℚ := 0.15

theorem dislikeBothRadioAndMusic_count :
  let dislikeRadioCount := dislikeRadioPercentage * totalPeople
  let dislikeBothCount := dislikeBothFromDislikeRadioPercentage * dislikeRadioCount
  (dislikeBothCount : ℚ).round = 56 := 
by
  sorry

end dislikeBothRadioAndMusic_count_l116_116285


namespace find_length_RT_l116_116223

variables {P Q R T U : Type} [inner_product_space ℝ P] [inner_product_space ℝ Q]
  [inner_product_space ℝ R] [inner_product_space ℝ T] [inner_product_space ℝ U]

-- Given conditions
def angle_R_90 (P Q R : Type) [inner_product_space ℝ P] [inner_product_space ℝ Q] [inner_product_space ℝ R] : Prop :=
  ∠QPR = 90

def length_PR : ℝ := 9
def length_QR : ℝ := 12

def on_PQ (T : Type) (P Q : Type) [inner_product_space ℝ T] [inner_product_space ℝ P] [inner_product_space ℝ Q] : Prop := true
def on_QR (U : Type) (Q R : Type) [inner_product_space ℝ U] [inner_product_space ℝ Q] [inner_product_space ℝ R] : Prop := true

def angle_TUR_90 : ∀ {T U R : Type}, T → U → R → Prop := 
  λ _ _ _ T U R, ∠TUR = 90

def length_TU : ℝ := 6

variable {length_RT : ℝ}

-- The proof goal
theorem find_length_RT (P Q R T U : Type)
  [inner_product_space ℝ P] [inner_product_space ℝ Q] [inner_product_space ℝ R]
  [inner_product_space ℝ T] [inner_product_space ℝ U]
  (hR_90 : angle_R_90 P Q R) 
  (h_on_PQ : on_PQ T P Q)
  (h_on_QR : on_QR U Q R)
  (h_angle_TUR : angle_TUR_90 T U R)
  (h_len_PR : ∥P - R∥ = length_PR)
  (h_len_QR : ∥Q - R∥ = length_QR)
  (h_len_TU : ∥T - U∥ = length_TU) :
  ∥R - T∥ = 10 := sorry

end find_length_RT_l116_116223


namespace paul_final_balance_l116_116717

def initial_balance : ℝ := 400
def transfer1 : ℝ := 90
def transfer2 : ℝ := 60
def service_charge_rate : ℝ := 0.02

def service_charge (x : ℝ) : ℝ := service_charge_rate * x

def total_deduction : ℝ := transfer1 + service_charge transfer1 + service_charge transfer2

def final_balance (init_balance : ℝ) (deduction : ℝ) : ℝ := init_balance - deduction

theorem paul_final_balance :
  final_balance initial_balance total_deduction = 307 :=
by
  sorry

end paul_final_balance_l116_116717


namespace number_of_street_trees_l116_116319

theorem number_of_street_trees (road_length interval_length : ℕ) (h1 : road_length = 2575) (h2 : interval_length = 25) : 
  (road_length / interval_length) + 1 = 104 :=
by
  rw [h1, h2]
  sorry

end number_of_street_trees_l116_116319


namespace opposite_of_neg_2023_l116_116978

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l116_116978


namespace concyclic_points_l116_116242

variable (Point : Type) [Inhabited Point] [AddGroup Point]

noncomputable def midpoint (A B : Point) : Point := (A + B) / 2

variables (A B C D A1 B1 A2 B2 : Point)
variables (M N : Point) -- midpoints
variables (circumcircle : Point → Point → Point → Point → Prop)
variables (cyclic_quadrilateral : Point → Point → Point → Point → Prop)

-- Conditions
def is_trapezium (P Q R S : Point) : Prop := Q - P = S - R ∨ R - P = Q - S
def symmetric_about (P Q : Point) : Point := 2 * P - Q

-- Hypotheses
hypothesis h_trapezium : is_trapezium A B C D
hypothesis h_circumcircle : circumcircle A B C D
hypothesis h_cyclic_quadrilateral : cyclic_quadrilateral D A1 B1 C

def problem : Prop :=
  let M := midpoint A C in
  let N := midpoint B C in
  let A2 := symmetric_about M A1 in
  let B2 := symmetric_about N B1 in
  cyclic_quadrilateral A B A2 B2

theorem concyclic_points : problem A B C D A1 B1 := 
by
sorry

end concyclic_points_l116_116242


namespace road_trip_split_costs_l116_116483

theorem road_trip_split_costs (a b : ℕ) : 
  let alice_paid := 90
  let bob_paid := 150
  let carlos_paid := 210
  let total_cost := alice_paid + bob_paid + carlos_paid
  let each_should_pay := total_cost / 3
  let a := if alice_paid < each_should_pay then each_should_pay - alice_paid else 0
  let b := if bob_paid < each_should_pay then each_should_pay - bob_paid else 0
  a - b = 60 :=
by {
  let alice_paid := 90,
  let bob_paid := 150,
  let carlos_paid := 210,
  let total_cost := alice_paid + bob_paid + carlos_paid,
  let each_should_pay := total_cost / 3,
  let a := if alice_paid < each_should_pay then each_should_pay - alice_paid else 0,
  let b := if bob_paid < each_should_pay then each_should_pay - bob_paid else 0,
  exact sorry,
}

end road_trip_split_costs_l116_116483


namespace opposite_of_neg_2023_l116_116891

-- Let x be the opposite of -2023
def is_opposite (x : ℤ) (n : ℤ) : Prop := n + x = 0

theorem opposite_of_neg_2023 : ∃ x : ℤ, is_opposite x (-2023) ∧ x = 2023 :=
by
  exists 2023
  unfold is_opposite
  sorry

end opposite_of_neg_2023_l116_116891


namespace tan_alpha_value_l116_116546

variable {α : ℝ} (h_quadrant: π/2 < α ∧ α < π) (h_sin_sum: sin (α + π/4) = sqrt 2 / 10)

theorem tan_alpha_value :
  tan α = -4/3 :=
by sorry

end tan_alpha_value_l116_116546


namespace vectors_parallel_l116_116129

theorem vectors_parallel (m n : ℝ) (k : ℝ) (h1 : 2 = k * 1) (h2 : -1 = k * m) (h3 : 2 = k * n) : 
  m + n = 1 / 2 := 
by
  sorry

end vectors_parallel_l116_116129


namespace tom_bike_rental_hours_calculation_l116_116684

variable (h : ℕ)
variable (base_cost : ℕ := 17)
variable (hourly_rate : ℕ := 7)
variable (total_paid : ℕ := 80)

theorem tom_bike_rental_hours_calculation (h : ℕ) 
  (base_cost : ℕ := 17) (hourly_rate : ℕ := 7) (total_paid : ℕ := 80) 
  (hours_eq : total_paid = base_cost + hourly_rate * h) : 
  h = 9 := 
by
  -- The proof is omitted.
  sorry

end tom_bike_rental_hours_calculation_l116_116684


namespace prove_a_is_perfect_square_l116_116637

-- Definition of a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Main theorem statement
theorem prove_a_is_perfect_square 
  (a b : ℕ) 
  (hb_odd : b % 2 = 1) 
  (h_integer : ∃ k : ℕ, ((a + b) * (a + b) + 4 * a) = k * a * b) :
  is_perfect_square a :=
sorry

end prove_a_is_perfect_square_l116_116637


namespace opposite_neg_2023_l116_116761

theorem opposite_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_neg_2023_l116_116761


namespace number_with_divisor_product_is_18_l116_116403

theorem number_with_divisor_product_is_18 :
  ∃ A : ℕ, ∏ (d : ℕ) in divisors A, d = 5832 → A = 18 := 
by
  sorry

end number_with_divisor_product_is_18_l116_116403


namespace sixty_eighth_term_in_Q_is_464_l116_116265

def P : Set ℕ := {0, 2, 4, 6, 8}
def Q : Set ℕ := {m | ∃ a1 a2 a3 ∈ P, m = 100 * a1 + 10 * a2 + a3}

theorem sixty_eighth_term_in_Q_is_464 :
  ∃ S : List ℕ, S.sorted (≤) ∧ S.length ≥ 68 ∧ S.get? (68-1) = some 464 :=
by
  sorry

end sixty_eighth_term_in_Q_is_464_l116_116265


namespace opposite_of_neg_2023_l116_116902

-- Let x be the opposite of -2023
def is_opposite (x : ℤ) (n : ℤ) : Prop := n + x = 0

theorem opposite_of_neg_2023 : ∃ x : ℤ, is_opposite x (-2023) ∧ x = 2023 :=
by
  exists 2023
  unfold is_opposite
  sorry

end opposite_of_neg_2023_l116_116902


namespace tangent_line_at_point_l116_116753

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

def point : ℝ × ℝ := (1, Real.exp 1)

theorem tangent_line_at_point :
  let m := deriv f (point.1)
  let b := point.2 - m * point.1
  (∀ x : ℝ, (λ y : ℝ, y = m * x + b) x = (2 * Real.exp x - Real.exp 1)) :=
by
  sorry

end tangent_line_at_point_l116_116753


namespace extremum_of_g_range_of_a_l116_116521

-- Condition definitions
def f (a : ℝ) (x : ℝ) := Real.exp x - a * x^2
def g (a : ℝ) (x : ℝ) := deriv (f a) x
def h (a : ℝ) (x : ℝ) := Real.exp x - a * x - 1

-- I) Proving the extremum of g(x)
theorem extremum_of_g (a : ℝ) (ha : a > 0) : ∃ x : ℝ, g a x = 2 * a - 2 * a * Real.log (2 * a) :=
sorry

-- II) Range of a such that f(x) ≥ x + (1 - x) * exp(x) for x ≥ 0
theorem range_of_a (a : ℝ) (H : ∀ x ≥ 0, f a x ≥ x + (1 - x) * Real.exp x) : a ≤ 1 :=
sorry

end extremum_of_g_range_of_a_l116_116521


namespace total_weight_moved_l116_116236

-- Given conditions as definitions
def initial_back_squat : ℝ := 200
def back_squat_increase : ℝ := 50
def front_squat_ratio : ℝ := 0.8
def triple_ratio : ℝ := 0.9

-- Proving the total weight moved for three triples
theorem total_weight_moved : 
  let new_back_squat := initial_back_squat + back_squat_increase in
  let front_squat := new_back_squat * front_squat_ratio in
  let weight_per_triple := front_squat * triple_ratio in
  3 * weight_per_triple = 540 :=
by
  sorry

end total_weight_moved_l116_116236


namespace last_digit_of_large_exponentiation_l116_116389

theorem last_digit_of_large_exponentiation :
  let n1 := 99^9,
      n2 := 999^n1,
      n3 := 9999^n2,
      N := 99999^n3
  in N % 10 = 9 :=
by
  sorry

end last_digit_of_large_exponentiation_l116_116389


namespace opposite_of_neg_2023_l116_116844

def opposite (n : ℤ) : ℤ :=
  -n

theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116844


namespace min_value_of_f_l116_116087

def f (a1 a2 a3 a4 : ℝ) : ℝ :=
  a1^2 + a2^2 + a3^2 + a4^2 + a1*a3 + a2*a4

theorem min_value_of_f :
  ∀ (a1 a2 a3 a4 : ℝ), (a1 * a4 - a2 * a3 = 1) →
  f a1 a2 a3 a4 ≥ √3 :=
by
  sorry

end min_value_of_f_l116_116087


namespace simplify_expression_l116_116732

theorem simplify_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a^(2/3) * b^(1/2)) * (-3 * a^(1/2) * b^(1/3)) / (1/3 * a^(1/6) * b^(5/6)) = -9 * a :=
by sorry

end simplify_expression_l116_116732


namespace opposite_of_neg_2023_l116_116897

-- Let x be the opposite of -2023
def is_opposite (x : ℤ) (n : ℤ) : Prop := n + x = 0

theorem opposite_of_neg_2023 : ∃ x : ℤ, is_opposite x (-2023) ∧ x = 2023 :=
by
  exists 2023
  unfold is_opposite
  sorry

end opposite_of_neg_2023_l116_116897


namespace angle_bisectors_form_right_triangle_l116_116282

theorem angle_bisectors_form_right_triangle (A B C : Point) 
  (h : ∠ B = 120°) : 
  ∃ D E M, is_angle_bisector A B D ∧ 
           is_angle_bisector B C E ∧ 
           is_angle_bisector C A M ∧ 
           ∠ MDE = 90° :=
sorry

end angle_bisectors_form_right_triangle_l116_116282


namespace jenny_graduating_class_sum_x_l116_116231

theorem jenny_graduating_class_sum_x :
  let total_students := 360
  let valid_xs := {x | ∃ y, x * y = total_students ∧ x ≥ 18 ∧ y ≥ 12}
  (∑ x in valid_xs, x) = 92 :=
sorry

end jenny_graduating_class_sum_x_l116_116231


namespace opposite_of_neg_2023_l116_116927

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by 
  sorry

end opposite_of_neg_2023_l116_116927


namespace dress_cost_l116_116273

theorem dress_cost (q_original q_left : ℕ) (quarter_value : ℚ) (cost : ℚ) 
  (h1 : q_original = 160) 
  (h2 : q_left = 20) 
  (h3 : quarter_value = 0.25) : 
  cost = (q_original - q_left) * quarter_value := 
by
  sorry

example :
  ∃ cost : ℚ,
    let q_original := 160 in
    let q_left := 20 in
    let quarter_value := 0.25 in
    cost = (q_original - q_left) * quarter_value := 
by
  use 35
  sorry

end dress_cost_l116_116273


namespace opposite_of_neg_2023_l116_116906

-- Let x be the opposite of -2023
def is_opposite (x : ℤ) (n : ℤ) : Prop := n + x = 0

theorem opposite_of_neg_2023 : ∃ x : ℤ, is_opposite x (-2023) ∧ x = 2023 :=
by
  exists 2023
  unfold is_opposite
  sorry

end opposite_of_neg_2023_l116_116906


namespace max_sum_of_arithmetic_sequence_l116_116210

-- Let \( a \) be the arithmetic sequence with first term \( a_1 \) and common difference \( d \).
variable (a : ℕ → ℤ)
variable (S : ℕ → ℤ)

-- Define the conditions
def a_1 := 50
def S (n : ℕ) := n * (2 * a_1 + (n - 1) * d) / 2

-- Condition from problem statement
def condition1 : a 1 = 50 := by sorry
def condition2 : S 9 = S 17 := by sorry

-- Maximum value of sum of first \( n \) terms
theorem max_sum_of_arithmetic_sequence : 
  ∃ n : ℕ, S n = 91 := by sorry

end max_sum_of_arithmetic_sequence_l116_116210


namespace floor_sqrt_10_eq_3_l116_116054

theorem floor_sqrt_10_eq_3 :
  ∃ x : ℤ, x = 3 ∧ x ≤ Real.sqrt 10 ∧ Real.sqrt 10 < x + 1 :=
by
  use 3
  split
  . rfl
  split
  . apply Real.le_sqrt_of_sq_le
    calc 9 = 3 ^ 2 : by norm_num
           ... ≤ 10 : by norm_num
  . apply Real.sqrt_lt_of_sq_lt
    calc 10 < 16 : by norm_num
           ... = 4 ^ 2 : by norm_num

end floor_sqrt_10_eq_3_l116_116054


namespace train_speed_l116_116401

theorem train_speed (length : ℕ) (time : ℕ) (h_length : length = 700) (h_time : time = 20) : (length / time) = 35 :=
by
  rw [h_length, h_time]
  norm_num
  sorry

end train_speed_l116_116401


namespace pascal_remaining_distance_l116_116694

variable (v : ℕ) (v_red : ℕ) (v_inc : ℕ) (t : ℕ) (d : ℕ)

def remaining_distance_proof : Prop :=
  v = 8 ∧
  v_red = 4 ∧
  v_inc = 12 ∧
  d = v * t ∧ 
  d = v_red * (t + 16) ∧ 
  d = v_inc * (t - 16) ∧ 
  d = 256

theorem pascal_remaining_distance (v : ℕ) (v_red : ℕ) (v_inc : ℕ) (t : ℕ) (d : ℕ) 
  (h1 : v = 8)
  (h2 : v_red = 4)
  (h3 : v_inc = 12)
  (h4 : d = v * t)
  (h5 : d = v_red * (t + 16))
  (h6 : d = v_inc * (t - 16)) : 
  d = 256 :=
by {
  -- Initial setup
  have h7 : d = 4 * (t + 16) := h5,
  have h8 : d = 12 * (t - 16) := h6,
  have h9 : 4 * (t + 16) = 12 * (t - 16), from sorry,
  -- Solve the equation 
  have h10 : 4 * t + 64 = 12 * t - 192, from sorry,
  have h11 : 4 * t + 256 = 12 * t, from sorry,
  have h12 : 8 * t = 256, from sorry,
  have ht : t = 32, from sorry,
  -- Compute d
  have hd : d = 8 * t, from h4,
  rw ht at hd,
  exact hd.symm,
}

end pascal_remaining_distance_l116_116694


namespace opposite_of_negative_2023_l116_116787

theorem opposite_of_negative_2023 : (- -2023) = 2023 := 
by {
sorry
}

end opposite_of_negative_2023_l116_116787


namespace simplify_expr_l116_116733

theorem simplify_expr : 
  (real.sqrt 448 / real.sqrt 128) + (real.sqrt 98 / real.sqrt 49) = (real.sqrt 14 + 2 * real.sqrt 2) / 2 :=
by 
  sorry

end simplify_expr_l116_116733


namespace parallelogram_if_equal_segments_l116_116217

variables {A B C D M M' N N' : Type*} [AffineSpace ℝ Type*]

def midpoint {A B : Type*} [AffineSpace ℝ Type*] (a b : A) : A := (a + b) / 2

theorem parallelogram_if_equal_segments
  (ABCD : ∀ {A B C D : Type*} [convex_space ℝ A] [convex_space ℝ B] [convex_space ℝ C] [convex_space ℝ D], Prop)
  (M : Midpoint A C)
  (N : Midpoint B D)
  (intersects : l ? (A B) ∩ l (C D) = {M' N'}) 
  (h : MM' = NN') :
  parallel (l ? (B C)) (l (A D)) := 
sorry

end parallelogram_if_equal_segments_l116_116217


namespace find_line_l2_calculate_triangle_area_l116_116123

-- Necessary conditions of the problems
variable {x y : ℝ}

def line_l1 : Prop := 3 * x + 4 * y - 1 = 0
def point_A : Prop := (3, 0)

-- Proving the questions as stated
theorem find_line_l2 (x y : ℝ) (h1 : line_l1) (h2 : point_A) :
  ∃ x y : ℝ, 4 * x - 3 * y - 12 = 0 := sorry

theorem calculate_triangle_area (x1 x2 y1 y2 : ℝ) (h1 : 4 * x1 - 3 * y1 - 12 = 0)
  (h2 : 4 * x2 - 3 * y2 - 12 = 0) (h3: x1 = 3 ∧ y1 = 0) (h4: x2 = 0 ∧ y2 = -4) :
  1 / 2 * abs (x1 * y2 - x2 * y1) = 6 := sorry

end find_line_l2_calculate_triangle_area_l116_116123


namespace complex_expression_l116_116555

noncomputable 
def complex_number := (1 - complex.I) / (1 + complex.I)

theorem complex_expression (a b : ℝ) (h : complex_number = a + b * complex.I) : a^2 - b^2 = -1 :=
by
  sorry

end complex_expression_l116_116555


namespace vector_projection_is_three_l116_116108

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

def magnitude (v : V) := ∥v∥

axiom mag_a : magnitude a = 4
axiom mag_b : magnitude b = 1
axiom dot_ab : ⟪a, b⟫ = 2

theorem vector_projection_is_three : ∥b∥ ≠ 0 → (2 • a - b) ⬝ b / ∥b∥ = 3 := by
  intros
  sorry

end vector_projection_is_three_l116_116108


namespace opposite_of_neg_2023_l116_116802

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by sorry

end opposite_of_neg_2023_l116_116802


namespace opposite_of_neg_2023_l116_116969

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l116_116969


namespace ratio_of_gold_and_copper_l116_116132

theorem ratio_of_gold_and_copper
  (G C : ℝ)
  (hG : G = 11)
  (hC : C = 5)
  (hA : (11 * G + 5 * C) / (G + C) = 8) : G = C :=
by
  sorry

end ratio_of_gold_and_copper_l116_116132


namespace opposite_of_neg_2023_l116_116875

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116875


namespace opposite_of_neg_2023_l116_116829

/-- Define the opposite of a number -/
def opposite (x : ℤ) : ℤ := -x

/-- Prove the opposite of -2023 is 2023 -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116829


namespace opposite_of_neg_2023_l116_116816

/-- Define the opposite of a number -/
def opposite (x : ℤ) : ℤ := -x

/-- Prove the opposite of -2023 is 2023 -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116816


namespace opposite_of_neg_2023_l116_116869

-- Definition of the problem
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  -- The proof is to be provided
  sorry

end opposite_of_neg_2023_l116_116869


namespace acme_vowel_soup_word_count_l116_116004

noncomputable def acme_extended_vowel_soup_five_letter_words : Nat :=
  let num_vowels := 3 + 4 + 2 + 5 + 3 + 3 in
  num_vowels ^ 5

theorem acme_vowel_soup_word_count :
  acme_extended_vowel_soup_five_letter_words = 3200000 :=
by
  unfold acme_extended_vowel_soup_five_letter_words
  sorry

end acme_vowel_soup_word_count_l116_116004


namespace opposite_of_neg2023_l116_116990

theorem opposite_of_neg2023 : ∀ (x : Int), -2023 + x = 0 → x = 2023 :=
by
  intros x h
  sorry

end opposite_of_neg2023_l116_116990


namespace opposite_of_neg_2023_l116_116946

theorem opposite_of_neg_2023 : ∃ x : ℤ, -2023 + x = 0 ∧ x = 2023 := 
by 
  use 2023
  constructor
  · exact rfl
  · exact rfl

end opposite_of_neg_2023_l116_116946


namespace tank_capacity_l116_116445

theorem tank_capacity :
  let rateA := 40  -- Pipe A fills at 40 liters per minute
  let rateB := 30  -- Pipe B fills at 30 liters per minute
  let rateC := -20  -- Pipe C (drains) at 20 liters per minute, thus negative contribution
  let cycle_duration := 3  -- The cycle duration is 3 minutes
  let total_duration := 51  -- The tank gets full in 51 minutes
  let net_per_cycle := rateA + rateB + rateC  -- Net fill per cycle of 3 minutes
  let num_cycles := total_duration / cycle_duration  -- Number of complete cycles
  let tank_capacity := net_per_cycle * num_cycles  -- Tank capacity in liters
  tank_capacity = 850  -- Assertion that needs to be proven
:= by
  let rateA := 40
  let rateB := 30
  let rateC := -20
  let cycle_duration := 3
  let total_duration := 51
  let net_per_cycle := rateA + rateB + rateC
  let num_cycles := total_duration / cycle_duration
  let tank_capacity := net_per_cycle * num_cycles
  have : tank_capacity = 850 := by
    sorry
  assumption

end tank_capacity_l116_116445


namespace professor_son_age_l116_116412

noncomputable def polynomial_with_integer_coefficients := Π x : ℤ, ℤ

theorem professor_son_age
  (f : polynomial_with_integer_coefficients)
  (A : ℤ)
  (P : ℤ)
  (h1 : f(A) = A)  -- Condition: f(A) = A
  (h2: f(0) = P)  -- Condition: f(0) = P
  (h3: P.prime)   -- Condition: P is a prime number
  (h4: P > A)     -- Condition: P > A
  : A = 1 := sorry

end professor_son_age_l116_116412


namespace distance_focus_directrix_eq_l116_116332

-- Define the equation of the parabola
def parabola (x : ℝ) : ℝ := 4 * x^2

-- Define the focus, directrix, and distance calculation
def focus : ℝ × ℝ := (0, 1 / 16)
def directrix : ℝ := -1 / 16
def distance (a b : ℝ) : ℝ := abs (a - b)

-- The theorem that states the problem
theorem distance_focus_directrix_eq : 
  distance focus.snd directrix = 1 / 8 := by sorry

end distance_focus_directrix_eq_l116_116332


namespace second_dog_miles_per_day_l116_116017

-- Definitions describing conditions
section DogWalk
variable (total_miles_week : ℕ)
variable (first_dog_miles_day : ℕ)
variable (days_in_week : ℕ)

-- Assert conditions given in the problem
def condition1 := total_miles_week = 70
def condition2 := first_dog_miles_day = 2
def condition3 := days_in_week = 7

-- The theorem to prove
theorem second_dog_miles_per_day
  (h1 : condition1 total_miles_week)
  (h2 : condition2 first_dog_miles_day)
  (h3 : condition3 days_in_week) :
  (total_miles_week - days_in_week * first_dog_miles_day) / days_in_week = 8 :=
sorry
end DogWalk

end second_dog_miles_per_day_l116_116017


namespace range_of_n_over_m_l116_116590

noncomputable theory

open Complex

def A (n m : ℝ) := {z : ℂ | abs (z + (n * I)) + abs (z - (m * I)) = n}
def B (n m : ℝ) := {z : ℂ | abs (z + (n * I)) - abs (z - (m * I)) = -m}

theorem range_of_n_over_m (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) 
  (hA_nonempty : ∃ z : ℂ, z ∈ A n m) (hB_nonempty : ∃ z : ℂ, z ∈ B n m) :
  n / m ∈ Icc (-∞) (-2) :=
sorry

end range_of_n_over_m_l116_116590


namespace opposite_of_neg_2023_l116_116828

/-- Define the opposite of a number -/
def opposite (x : ℤ) : ℤ := -x

/-- Prove the opposite of -2023 is 2023 -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116828


namespace golden_ratio_positive_root_ab_value_pq_minus_n_l116_116346

-- Definition for golden ratio question
def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2

-- Question 1: Prove x^2 + x - 1 = 0 has the positive root (1 + sqrt(5)) / 2
theorem golden_ratio_positive_root :
  (golden_ratio ^ 2 + golden_ratio - 1 = 0) :=
by
  sorry

-- Question 2: Given the conditions, prove ab = 2
variables (a b m : ℝ)

theorem ab_value (h1 : a^2 + m*a = 1) (h2 : b^2 - 2*m*b = 4) (h3 : b ≠ -2*a) :
  a * b = 2 :=
by
  sorry

-- Question 3: Given the conditions, prove pq - n = 0
variables (n p q : ℝ)

theorem pq_minus_n (hpq1 : p ≠ q) (hpq2 : p^2 + n*p - 1 = q) (hpq3 : q^2 + n*q - 1 = p) :
  p * q - n = 0 :=
by
  sorry

end golden_ratio_positive_root_ab_value_pq_minus_n_l116_116346


namespace opposite_of_neg_2023_l116_116975

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l116_116975


namespace opposite_of_neg_2023_l116_116830

/-- Define the opposite of a number -/
def opposite (x : ℤ) : ℤ := -x

/-- Prove the opposite of -2023 is 2023 -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116830


namespace r_needs_35_days_l116_116406

def work_rate (P Q R: ℚ) : Prop :=
  (P = Q + R) ∧ (P + Q = 1/10) ∧ (Q = 1/28)

theorem r_needs_35_days (P Q R: ℚ) (h: work_rate P Q R) : 1 / R = 35 :=
by 
  sorry

end r_needs_35_days_l116_116406


namespace boxes_used_l116_116433

-- Definitions of the conditions
def total_oranges := 2650
def oranges_per_box := 10

-- Statement to prove
theorem boxes_used (total_oranges oranges_per_box : ℕ) : (total_oranges = 2650) → (oranges_per_box = 10) → (total_oranges / oranges_per_box = 265) :=
by
  intros h_total h_per_box
  rw [h_total, h_per_box]
  norm_num

end boxes_used_l116_116433


namespace min_value_frac_inv_l116_116648

noncomputable def min_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 12) : ℝ :=
  1 / a + 1 / b

theorem min_value_frac_inv (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 12) :
  min_value a b h1 h2 h3 = 1 / 3 :=
sorry

end min_value_frac_inv_l116_116648


namespace sum_of_first_9_terms_l116_116093

variable (a : ℕ → ℤ)
variable (S : ℕ → ℤ)
variable (a1 : ℤ)
variable (d : ℤ)

-- Given is that the sequence is arithmetic.
-- Given a1 is the first term, and d is the common difference, we can define properties based on the conditions.
def is_arithmetic_sequence (a : ℕ → ℤ) (a1 d : ℤ) : Prop :=
  ∀ n : ℕ, n > 0 → a n = a1 + (n - 1) * d

def sum_first_n_terms (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

-- Given condition: 2a_1 + a_13 = -9.
def given_condition (a : ℕ → ℤ) (a1 : ℤ) (d : ℤ) : Prop :=
  2 * a1 + (a1 + 12 * d) = -9

theorem sum_of_first_9_terms (a : ℕ → ℤ) (S : ℕ → ℤ) (a1 d : ℤ)
  (h_arith : is_arithmetic_sequence a a1 d)
  (h_sum : sum_first_n_terms S a)
  (h_cond : given_condition a a1 d) :
  S 9 = -27 :=
sorry

end sum_of_first_9_terms_l116_116093


namespace opposite_of_neg_2023_l116_116810

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by sorry

end opposite_of_neg_2023_l116_116810


namespace correct_answer_is_B_l116_116239

/-
Kolya participates in the TV game "Become a Millionaire". There are 4 answer choices for the question: A, B, C, D. Kolya gets 4 hints:

- The correct answer is A or B.
- The correct answer is C or D.
- The correct answer is B.
- Answer D is incorrect.

It is known that three of the hints are wrong and only one is correct. What is the correct answer?
-/

theorem correct_answer_is_B
  (A B C D : Prop)
  (h1 : A ∨ B)
  (h2 : C ∨ D)
  (h3 : B)
  (h4 : ¬ D)
  (H : (¬ h1 ∧ ¬ h2 ∧ h3 ∧ ¬ h4) ∨ 
        (¬ h1 ∧ ¬ h2 ∧ ¬ h3 ∧ h4) ∨ 
        (¬ h1 ∧ h2 ∧ ¬ h3 ∧ ¬ h4) ∨ 
        (h1 ∧ ¬ h2 ∧ ¬ h3 ∧ ¬ h4)) :
  B := by
  -- The proof is not provided as per instruction.
  sorry

end correct_answer_is_B_l116_116239


namespace min_value_of_geometric_sequence_l116_116552

noncomputable def geometric_sequence_min_value (a : ℕ → ℝ) (k : ℕ) : Prop :=
  (∀ n : ℕ, n ≥ 1 → a n > 0) ∧
  (∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n+1) = a n * r) ∧
  (a (2 * k) + a (2 * k - 1) + ... + a (k + 1) - (a k + a (k - 1) + ... + a 1) = 8) ∧
  (k > 0) →
  a (2 * k + 1) + a (2 * k + 2) + ... + a (3 * k) = 32

theorem min_value_of_geometric_sequence (a : ℕ → ℝ) (k : ℕ) : 
  geometric_sequence_min_value a k :=
sorry

end min_value_of_geometric_sequence_l116_116552


namespace average_speed_of_trip_l116_116421

noncomputable def total_distance (d1 d2 : ℝ) : ℝ :=
  d1 + d2

noncomputable def travel_time (distance speed : ℝ) : ℝ :=
  distance / speed

noncomputable def average_speed (total_distance total_time : ℝ) : ℝ :=
  total_distance / total_time

theorem average_speed_of_trip :
  let d1 := 60
  let s1 := 20
  let d2 := 120
  let s2 := 60
  let total_d := total_distance d1 d2
  let time1 := travel_time d1 s1
  let time2 := travel_time d2 s2
  let total_t := time1 + time2
  average_speed total_d total_t = 36 :=
by
  sorry

end average_speed_of_trip_l116_116421


namespace temperature_decrease_denotation_l116_116618

noncomputable def temperature_sign_opposite (a : ℝ) (b : ℝ) : Prop :=
  if a > 0 then b = -a else if a < 0 then b = -a else b = 0

theorem temperature_decrease_denotation 
  (decrease : ℝ) (denotation : ℝ) 
  (h1 : decrease = 2) 
  (h2 : temperature_sign_opposite 3 -3 ) :
  denotation = -2 :=
sorry

end temperature_decrease_denotation_l116_116618


namespace opposite_of_neg_2023_l116_116974

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l116_116974


namespace symmetry_wrt_z_axis_l116_116222

structure Point3D :=
  (x : Int)
  (y : Int)
  (z : Int)

def symmetric_z_axis (p : Point3D) : Point3D :=
  {x := -p.x, y := -p.y, z := p.z}

theorem symmetry_wrt_z_axis :
  (M N : Point3D) (hM : M = ⟨1, -2, 3⟩) (hN : N = ⟨-1, 2, 3⟩) :
  N = symmetric_z_axis M :=
  sorry

end symmetry_wrt_z_axis_l116_116222


namespace remaining_distance_l116_116714

variable (D : ℝ) -- remaining distance in miles

-- Current speed
def current_speed := 8.0
-- Increased speed by 50%
def increased_speed := 12.0
-- Reduced speed by 4 mph
def reduced_speed := 4.0

-- Trip time relations
def current_time := D / current_speed
def increased_time := D / increased_speed
def reduced_time := D / reduced_speed

-- Time difference condition
axiom condition : reduced_time = increased_time + 16.0

theorem remaining_distance : D = 96.0 :=
by
  -- proof placeholder
  sorry

end remaining_distance_l116_116714


namespace find_x_l116_116131

-- Define the vectors a and b
def vec_a : ℝ × ℝ := (3, 4)
def vec_b (x : ℝ) : ℝ × ℝ := (x, 1)

-- Define the condition that (a - b) is perpendicular to a
def perp_condition (x : ℝ) : Prop := 
  let a_minus_b := (vec_a.1 - x, vec_a.2 - 1)
  in (a_minus_b.1 * vec_a.1 + a_minus_b.2 * vec_a.2) = 0

-- State the theorem
theorem find_x : ∃ x : ℝ, perp_condition x ∧ x = 7 := 
by
  sorry

end find_x_l116_116131


namespace opposite_of_neg_2023_l116_116977

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l116_116977


namespace shaded_region_area_l116_116348

theorem shaded_region_area (d : ℝ) (n : ℕ) (side : ℝ) (area_small : ℝ) (area_total : ℝ) :
  d = 10 →
  n = 25 →
  side = d / Real.sqrt 2 →
  area_small = side^2 / n →
  area_total = area_small * n →
  area_total = 50 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end shaded_region_area_l116_116348


namespace opposite_of_neg_2023_l116_116892

-- Let x be the opposite of -2023
def is_opposite (x : ℤ) (n : ℤ) : Prop := n + x = 0

theorem opposite_of_neg_2023 : ∃ x : ℤ, is_opposite x (-2023) ∧ x = 2023 :=
by
  exists 2023
  unfold is_opposite
  sorry

end opposite_of_neg_2023_l116_116892


namespace distance_to_second_picture_edge_l116_116452

/-- Given a wall of width 25 feet, with a first picture 5 feet wide centered on the wall,
and a second picture 3 feet wide centered in the remaining space, the distance 
from the nearest edge of the second picture to the end of the wall is 13.5 feet. -/
theorem distance_to_second_picture_edge :
  let wall_width := 25
  let first_picture_width := 5
  let second_picture_width := 3
  let side_space := (wall_width - first_picture_width) / 2
  let remaining_space := side_space
  let second_picture_side_space := (remaining_space - second_picture_width) / 2
  10 + 3.5 = 13.5 :=
by
  sorry

end distance_to_second_picture_edge_l116_116452


namespace problem1_problem2a_problem2b_l116_116572

variables {a b : ℝ × ℝ} -- Representing vectors as pairs of real numbers
variable {θ : ℝ} -- Representing the angle between a and b as a real number
variable {x : ℝ} -- Representing x as a real number

-- Condition definitions
def non_collinear (a b : ℝ × ℝ) : Prop := a ≠ (0,0) ∧ b ≠ (0,0) ∧ a.1 * b.2 ≠ a.2 * b.1
def included_angle (a b : ℝ × ℝ) (θ : ℝ) : Prop := (a.1 * b.1 + a.2 * b.2) = |a| * |b| * (Real.cos θ)
def magnitude (a : ℝ × ℝ) : ℝ := Real.sqrt (a.1 ^ 2 + a.2 ^ 2)
def is_perpendicular (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

-- Problem 1: Prove that tan θ = sqrt(35)
theorem problem1 (h₁ : non_collinear a b)
  (h₂ : magnitude a = 3)
  (h₃ : magnitude b = 1)
  (h₄ : is_perpendicular (a.1 + 2 * b.1, a.2 + 2 * b.2) (a.1 - 4 * b.1, a.2 - 4 * b.2)) :
  Real.tan θ = Real.sqrt 35 :=
sorry

-- Problem 2a: Prove the minimum value of |x * a - b| is 1/2 at x = sqrt(3)/6
theorem problem2a (h₁ : non_collinear a b)
  (h₂ : magnitude a = 3)
  (h₃ : magnitude b = 1)
  (h₄ : θ = Real.pi / 6) :
  ∃ x, x = Real.sqrt 3 / 6 ∧ magnitude (x * a.1 - b.1, x * a.2 - b.2) = 1 / 2 :=
sorry

-- Problem 2b: Prove that a is perpendicular to x * a - b given x = sqrt(3)/6
theorem problem2b (h₁ : non_collinear a b)
  (h₂ : magnitude a = 3)
  (h₃ : magnitude b = 1)
  (h₄ : θ = Real.pi / 6)
  (h₅ : x = Real.sqrt 3 / 6) :
  is_perpendicular a (x * a.1 - b.1, x * a.2 - b.2) :=
sorry

end problem1_problem2a_problem2b_l116_116572


namespace opposite_neg_2023_l116_116766

theorem opposite_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_neg_2023_l116_116766


namespace opposite_of_neg2023_l116_116986

theorem opposite_of_neg2023 : ∀ (x : Int), -2023 + x = 0 → x = 2023 :=
by
  intros x h
  sorry

end opposite_of_neg2023_l116_116986


namespace probability_of_odd_number_l116_116330

theorem probability_of_odd_number :
  let digits := [1, 4, 6, 9] in
  let is_odd (n : Nat) := n % 2 = 1 in
  let units_choices := [1, 9] in
  probability (random_permutation digits) (λ n, is_odd n.units_digit) = 1 / 2 :=
sorry

end probability_of_odd_number_l116_116330


namespace monotonicity_of_f_range_of_a_sum_of_x1_x2_l116_116118

noncomputable def f (a x : ℝ) : ℝ := ln (a*x + 1) - a*x - ln a
noncomputable def h (a x : ℝ) : ℝ := a*x - f a x

open Set

-- Statement for the monotonicity of f(x)
theorem monotonicity_of_f (a : ℝ) (ha : 0 < a) :
  (∀ x ∈ Ioo (-1/a) 0, 0 < derivative (f a) x) ∧ (∀ x ∈ Ioi 0, derivative (f a) x < 0) :=
sorry

-- Statement for the range of a
theorem range_of_a (a : ℝ) (hpos : ∀ x : ℝ, h a x > 0) : (e / 2) < a :=
sorry

-- Statement for x₁ + x₂ > 0
theorem sum_of_x1_x2 (a x₁ x₂ : ℝ) (hx₁ : -1 / a < x₁) (hx₁0 : x₁ < 0) (hx₂ : 0 < x₂) 
  (hf_x₁ : f a x₁ = 0) (hf_x₂ : f a x₂ = 0) : 
  0 < x₁ + x₂ :=
sorry

end monotonicity_of_f_range_of_a_sum_of_x1_x2_l116_116118


namespace Hazel_missed_days_l116_116355

theorem Hazel_missed_days (total_days : ℕ) (percent_allowable_missed : ℝ) (remaining_days : ℕ) : 
    total_days = 180 →
    percent_allowable_missed = 0.05 →
    remaining_days = 3 →
    ∃ (days_missed : ℕ), days_missed = 6 :=
by
  intro h1 h2 h3
  have allowable_missed_days := nat.floor ((total_days : ℝ) * percent_allowable_missed)  -- Calculating allowable missed days
  have already_missed_days := allowable_missed_days - remaining_days  -- Determining already missed days
  use already_missed_days
  rw [h1, h2, h3]
  sorry  -- This will be replaced with the actual calculations

end Hazel_missed_days_l116_116355


namespace opposite_of_neg_2023_l116_116876

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116876


namespace tan_add_pi_div_three_l116_116153

theorem tan_add_pi_div_three (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = -((6 + 5 * Real.sqrt 3) / 13) := 
by 
  sorry

end tan_add_pi_div_three_l116_116153


namespace collinear_points_sum_l116_116597

theorem collinear_points_sum (a b : ℝ) 
  (h_collin: ∃ k : ℝ, 
    (1 - a) / (a - a) = k * (a - b) / (b - b) ∧
    (a - a) / (2 - b) = k * (2 - 3) / (3 - 3) ∧
    (a - b) / (3 - 3) = k * (a - a) / (3 - b) ) : 
  a + b = 4 :=
by
  sorry

end collinear_points_sum_l116_116597


namespace problem1_problem2_l116_116415

/-- Problem 1: Given that α is an angle in the third quadrant, and tan α = 1 / 3, 
    find the values of sin α and cos α. --/
theorem problem1 (α : ℝ) (hα : α ∈ Icc π (3 * π / 2)) (h_tan : Real.tan α = 1 / 3) :
    Real.sin α = -√10 / 10 ∧ Real.cos α = -3 * √10 / 10 :=
sorry

/-- Problem 2: Given that the terminal side of angle α passes through a point P with 
    coordinates (3a, 4a), where a ≠ 0, find sin α, cos α, and tan α. --/
theorem problem2 (α : ℝ) (a : ℝ) (hα : ∃ r, (3 * a, 4 * a) = (r * cos α, r * sin α)) (h_neq : a ≠ 0) :
    (a > 0 → Real.sin α = 4 / 5 ∧ Real.cos α = 3 / 5 ∧ Real.tan α = 4 / 3) ∧
    (a < 0 → Real.sin α = -4 / 5 ∧ Real.cos α = -3 / 5 ∧ Real.tan α = 4 / 3) :=
sorry

end problem1_problem2_l116_116415


namespace average_x_correct_average_y_correct_correlation_coefficient_correct_estimated_total_volume_correct_l116_116454

namespace ForestEstimation

-- Constants given in the problem
def x_vals : List ℝ := [0.04, 0.06, 0.04, 0.08, 0.08, 0.05, 0.05, 0.07, 0.07, 0.06]
def y_vals : List ℝ := [0.25, 0.40, 0.22, 0.54, 0.51, 0.34, 0.36, 0.46, 0.42, 0.40]

def sum_x_vals : ℝ := 0.6
def sum_y_vals : ℝ := 3.9
def sum_x_vals_sq : ℝ := 0.038
def sum_y_vals_sq : ℝ := 1.6158
def sum_x_y_vals : ℝ := 0.2474
def total_cross_sectional_area : ℝ := 186

-- Calculations derived in the solution
def average_x : ℝ := sum_x_vals / 10
def average_y : ℝ := sum_y_vals / 10

def correlation_coefficient : ℝ :=
  let x_bar := average_x
  let y_bar := average_y
  (sum_x_y_vals - 10 * x_bar * y_bar) /
  (Math.sqrt ((sum_x_vals_sq - 10 * x_bar^2) * (sum_y_vals_sq - 10 * y_bar^2)))

def estimated_total_volume : ℝ :=
  (average_y / average_x) * total_cross_sectional_area

-- Proof statements
theorem average_x_correct : 
  average_x = 0.06 := by
  sorry

theorem average_y_correct : 
  average_y = 0.39 := by
  sorry

theorem correlation_coefficient_correct : 
  correlation_coefficient ≈ 0.97 := by
  sorry

theorem estimated_total_volume_correct : 
  estimated_total_volume = 1209 := by
  sorry

end ForestEstimation

end average_x_correct_average_y_correct_correlation_coefficient_correct_estimated_total_volume_correct_l116_116454


namespace remaining_distance_l116_116715

variable (D : ℝ) -- remaining distance in miles

-- Current speed
def current_speed := 8.0
-- Increased speed by 50%
def increased_speed := 12.0
-- Reduced speed by 4 mph
def reduced_speed := 4.0

-- Trip time relations
def current_time := D / current_speed
def increased_time := D / increased_speed
def reduced_time := D / reduced_speed

-- Time difference condition
axiom condition : reduced_time = increased_time + 16.0

theorem remaining_distance : D = 96.0 :=
by
  -- proof placeholder
  sorry

end remaining_distance_l116_116715


namespace equality_of_pow_l116_116179

theorem equality_of_pow (a b : ℝ) (h : {a^2, 0, -1} = {a, b, 0}) : a^2023 + b^2023 = 0 :=
by
  sorry

end equality_of_pow_l116_116179


namespace opposite_of_negative_2023_l116_116783

theorem opposite_of_negative_2023 : (- -2023) = 2023 := 
by {
sorry
}

end opposite_of_negative_2023_l116_116783


namespace opposite_of_neg_2023_l116_116866

-- Definition of the problem
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  -- The proof is to be provided
  sorry

end opposite_of_neg_2023_l116_116866


namespace paul_account_balance_after_transactions_l116_116724

theorem paul_account_balance_after_transactions :
  let transfer1 := 90
  let transfer2 := 60
  let service_charge_percent := 2 / 100
  let initial_balance := 400
  let service_charge1 := service_charge_percent * transfer1
  let service_charge2 := service_charge_percent * transfer2
  let total_deduction1 := transfer1 + service_charge1
  let total_deduction2 := transfer2 + service_charge2
  let reversed_transfer2 := total_deduction2 - transfer2
  let balance_after_transfer1 := initial_balance - total_deduction1
  let final_balance := balance_after_transfer1 - reversed_transfer2
  (final_balance = 307) :=
by
  sorry

end paul_account_balance_after_transactions_l116_116724


namespace smiths_bakery_pies_l116_116305

theorem smiths_bakery_pies (m : ℕ) (h : m = 16) : 
  let s := 4 * m + 6 in 
  s = 70 := 
by
  unfold s
  sorry

end smiths_bakery_pies_l116_116305


namespace opposite_of_neg_2023_l116_116929

theorem opposite_of_neg_2023 : ∃ x : ℤ, -2023 + x = 0 ∧ x = 2023 := 
by 
  use 2023
  constructor
  · exact rfl
  · exact rfl

end opposite_of_neg_2023_l116_116929


namespace opposite_of_neg_2023_l116_116898

-- Let x be the opposite of -2023
def is_opposite (x : ℤ) (n : ℤ) : Prop := n + x = 0

theorem opposite_of_neg_2023 : ∃ x : ℤ, is_opposite x (-2023) ∧ x = 2023 :=
by
  exists 2023
  unfold is_opposite
  sorry

end opposite_of_neg_2023_l116_116898


namespace opposite_of_negative_2023_l116_116778

theorem opposite_of_negative_2023 : (- -2023) = 2023 := 
by {
sorry
}

end opposite_of_negative_2023_l116_116778


namespace sequence_formula_l116_116221

theorem sequence_formula (a : ℕ → ℤ) (h₁ : a 1 = 1) (h₂ : ∀ n : ℕ, a n - a (n + 1) + 2 = 0) :
  ∀ n : ℕ, a n = 2 * n - 1 := 
sorry

end sequence_formula_l116_116221


namespace purple_shoes_count_l116_116354

-- Define the conditions
def total_shoes : ℕ := 1250
def blue_shoes : ℕ := 540
def remaining_shoes : ℕ := total_shoes - blue_shoes
def green_shoes := remaining_shoes / 2
def purple_shoes := green_shoes

-- State the theorem to be proven
theorem purple_shoes_count : purple_shoes = 355 := 
by
-- Proof can be filled in here (not needed for the task)
sorry

end purple_shoes_count_l116_116354


namespace savings_after_purchase_l116_116679

theorem savings_after_purchase :
  let price_sweater := 30
  let price_scarf := 20
  let num_sweaters := 6
  let num_scarves := 6
  let savings := 500
  let total_cost := (num_sweaters * price_sweater) + (num_scarves * price_scarf)
  savings - total_cost = 200 :=
by
  sorry

end savings_after_purchase_l116_116679


namespace opposite_of_neg_2023_l116_116901

-- Let x be the opposite of -2023
def is_opposite (x : ℤ) (n : ℤ) : Prop := n + x = 0

theorem opposite_of_neg_2023 : ∃ x : ℤ, is_opposite x (-2023) ∧ x = 2023 :=
by
  exists 2023
  unfold is_opposite
  sorry

end opposite_of_neg_2023_l116_116901


namespace slope_of_line_m_is_one_l116_116274

/-
 Line m lies in the xy-plane.
 The y-intercept of line m is -2.
 Line m passes through the midpoint of the line segment whose endpoints are (2, 8) and (6, -4).
 We want to prove that the slope of line m is 1.
-/

noncomputable def midpoint (x1 y1 x2 y2 : ℝ) : ℝ × ℝ :=
((x1 + x2) / 2, (y1 + y2) / 2)

noncomputable def slope (x1 y1 x2 y2 : ℝ) : ℝ :=
(y2 - y1) / (x2 - x1)

theorem slope_of_line_m_is_one :
  let midpoint_m := midpoint 2 8 6 (-4) in
  let y_intercept_m := -2 in
  let point_m := (4, 2) in
  let point_y_intercept := (0, y_intercept_m) in
  line_in_xy_plane ∧
  line_passes_through point_m ∧
  line_passes_through point_y_intercept →
  slope (fst point_y_intercept) (snd point_y_intercept) (fst point_m) (snd point_m) = 1 :=
by
  let midpoint_m := midpoint 2 8 6 (-4)
  let y_intercept_m := -2
  let point_m := (fst midpoint_m, snd midpoint_m)
  let point_y_intercept := (0, y_intercept_m)
  sorry

end slope_of_line_m_is_one_l116_116274


namespace opposite_of_neg_2023_l116_116815

/-- Define the opposite of a number -/
def opposite (x : ℤ) : ℤ := -x

/-- Prove the opposite of -2023 is 2023 -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116815


namespace intervals_where_increasing_l116_116505

noncomputable def is_monotonically_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, x < y → f x < f y

def cos_intervals (k : ℤ) : set ℝ :=
  {x | k * π + π / 8 ≤ x ∧ x ≤ k * π + 5 * π / 8}

theorem intervals_where_increasing :
  ∀ k : ℤ, is_monotonically_increasing (λ x, -⅓ * Real.cos (2 * x - π / 4)) (cos_intervals k) :=
begin
  sorry
end

end intervals_where_increasing_l116_116505


namespace opposite_of_neg_2023_l116_116926

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by 
  sorry

end opposite_of_neg_2023_l116_116926


namespace find_k_l116_116617

variable (a : ℕ → ℤ)
variable (d : ℤ)
variable (a1 : ℤ)

def arithmetic_sequence (n : ℕ) := a1 + (n - 1) * d

axiom a2_eq_3 : arithmetic_sequence a 2 = 3
axiom a4_eq_7 : arithmetic_sequence a 4 = 7
axiom ak_eq_15 : ∃ k : ℕ, arithmetic_sequence a k = 15

theorem find_k : ∃ k : ℕ, arithmetic_sequence a k = 15 → k = 8 :=
begin
  sorry
end

end find_k_l116_116617


namespace max_value_of_MN_l116_116193

def f (x : ℝ) : ℝ := Real.sin x
def g (x : ℝ) : ℝ := 2 * (Real.cos x)^2 - 1
def h (x : ℝ) : ℝ := f x - g x

theorem max_value_of_MN : 
  ∃ a : ℝ, (∃ M N : ℝ, f M = Real.sin a ∧ g N = 2 * (Real.cos a)^2 - 1 ∧ ∀ x, |h x| ≤ 2) :=
by
  sorry

end max_value_of_MN_l116_116193


namespace condition1_find_a_n_find_T_n_l116_116110

open Nat

-- Definition of sequences and sums
def a_seq (n : ℕ) : ℝ := (1 / 2) ^ n

def b_seq (n : ℕ) : ℝ := n * 2^n

def sum_seq (s : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in range n, s (i + 1)

-- Conditions
theorem condition1 {n : ℕ} (hn_pos : n > 0) : ∑ i in range n, a_seq (i + 1) = 1 - a_seq n :=
sorry

-- Proofs to be completed
theorem find_a_n (n : ℕ) (hn_pos : n > 0) : a_seq n = (1 / 2) ^ n :=
sorry

theorem find_T_n (n : ℕ) (hn_pos : n > 0) : sum_seq b_seq n = (n - 1) * 2^(n + 1) + 2 :=
sorry

end condition1_find_a_n_find_T_n_l116_116110


namespace length_of_PR_l116_116619

def right_angled_triangle_cutoff (s : ℝ) : Prop :=
  let area_triangle := (1 / 18) * s^2 in
  let total_area_cutoff := 4 * area_triangle in
  total_area_cutoff = 288

def length_PR (s : ℝ) : ℝ :=
  (2 / 3) * s

theorem length_of_PR (s : ℝ) (h1 : right_angled_triangle_cutoff s) : length_PR s = 24 :=
  sorry

end length_of_PR_l116_116619


namespace remaining_area_after_cut_l116_116519

theorem remaining_area_after_cut
  (cell_side_length : ℝ)
  (grid_side_length : ℕ)
  (total_area : ℝ)
  (removed_area : ℝ)
  (hyp1 : cell_side_length = 1)
  (hyp2 : grid_side_length = 6)
  (hyp3 : total_area = (grid_side_length * grid_side_length) * cell_side_length * cell_side_length) 
  (hyp4 : removed_area = 9) :
  total_area - removed_area = 27 := by
  sorry

end remaining_area_after_cut_l116_116519


namespace necessary_condition_l116_116288

theorem necessary_condition (A B C D : Prop) (h1 : A > B → C < D) : A > B → C < D := by
  exact h1 -- This is just a placeholder for the actual hypothesis, a required assumption in our initial problem statement

end necessary_condition_l116_116288


namespace opposite_of_negative_2023_l116_116779

theorem opposite_of_negative_2023 : (- -2023) = 2023 := 
by {
sorry
}

end opposite_of_negative_2023_l116_116779


namespace opposite_of_neg_2023_l116_116903

-- Let x be the opposite of -2023
def is_opposite (x : ℤ) (n : ℤ) : Prop := n + x = 0

theorem opposite_of_neg_2023 : ∃ x : ℤ, is_opposite x (-2023) ∧ x = 2023 :=
by
  exists 2023
  unfold is_opposite
  sorry

end opposite_of_neg_2023_l116_116903


namespace sin_C_l116_116543

variable {a b c A B C : ℝ}
variable (h1 : a = 1) (h2 : b = real.sqrt 3) (h3 : A + C = 2 * B) (triangle : a * sin A = b * sin B)

theorem sin_C (h1 : a = 1) (h2 : b = real.sqrt 3) (h3 : A + C = 2 * B) (triangle : a * sin A = b * sin B) : sin C = 1 := by
  sorry

end sin_C_l116_116543


namespace number_of_ways_is_25_l116_116356

-- Define the number of books
def number_of_books : ℕ := 5

-- Define the function to calculate the number of ways
def number_of_ways_to_buy_books : ℕ :=
  number_of_books * number_of_books

-- Define the theorem to be proved
theorem number_of_ways_is_25 : 
  number_of_ways_to_buy_books = 25 :=
by
  sorry

end number_of_ways_is_25_l116_116356


namespace tan_add_pi_over_3_l116_116159

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + (Real.pi / 3)) = -(6 * Real.sqrt 3 + 2) / 13 := 
by
  sorry

end tan_add_pi_over_3_l116_116159


namespace pyramid_volume_l116_116074

noncomputable def volume_of_pyramid (r β : ℝ) : ℝ :=
  3 * r^3 * (Real.cot (β / 2))^3 * Real.cos (β / 2) * Real.sqrt 3

theorem pyramid_volume (r β : ℝ) :
  volume_of_pyramid r β = 3 * r^3 * (Real.cot (β / 2))^3 * Real.cos (β / 2) * Real.sqrt 3 :=
sorry

end pyramid_volume_l116_116074


namespace floor_sqrt_10_l116_116051

theorem floor_sqrt_10 : (Real.floor (Real.sqrt 10) = 3) :=
by
  have h1 : Real.sqrt 9 = 3 := by norm_num
  have h2 : Real.sqrt 16 = 4 := by norm_num
  have h3 : 3 ≤ Real.sqrt 10 ∧ Real.sqrt 10 < 4 := ⟨by linarith [h1], by linarith [h2]⟩
  sorry

end floor_sqrt_10_l116_116051


namespace opposite_of_neg_2023_l116_116960

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l116_116960


namespace joyce_gave_apples_l116_116237

theorem joyce_gave_apples : 
  ∀ (initial_apples final_apples given_apples : ℕ), (initial_apples = 75) ∧ (final_apples = 23) → (given_apples = initial_apples - final_apples) → (given_apples = 52) :=
by
  intros
  sorry

end joyce_gave_apples_l116_116237


namespace opposite_of_neg_2023_l116_116972

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l116_116972


namespace children_tickets_l116_116461

-- Definition of the problem
variables (A C t : ℕ) (h_eq_people : A + C = t) (h_eq_money : 9 * A + 5 * C = 190)

-- The main statement we need to prove
theorem children_tickets (h_t : t = 30) : C = 20 :=
by {
  -- Proof will go here eventually
  sorry
}

end children_tickets_l116_116461


namespace opposite_of_neg_2023_l116_116928

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by 
  sorry

end opposite_of_neg_2023_l116_116928


namespace pascal_remaining_distance_l116_116693

variable (v : ℕ) (v_red : ℕ) (v_inc : ℕ) (t : ℕ) (d : ℕ)

def remaining_distance_proof : Prop :=
  v = 8 ∧
  v_red = 4 ∧
  v_inc = 12 ∧
  d = v * t ∧ 
  d = v_red * (t + 16) ∧ 
  d = v_inc * (t - 16) ∧ 
  d = 256

theorem pascal_remaining_distance (v : ℕ) (v_red : ℕ) (v_inc : ℕ) (t : ℕ) (d : ℕ) 
  (h1 : v = 8)
  (h2 : v_red = 4)
  (h3 : v_inc = 12)
  (h4 : d = v * t)
  (h5 : d = v_red * (t + 16))
  (h6 : d = v_inc * (t - 16)) : 
  d = 256 :=
by {
  -- Initial setup
  have h7 : d = 4 * (t + 16) := h5,
  have h8 : d = 12 * (t - 16) := h6,
  have h9 : 4 * (t + 16) = 12 * (t - 16), from sorry,
  -- Solve the equation 
  have h10 : 4 * t + 64 = 12 * t - 192, from sorry,
  have h11 : 4 * t + 256 = 12 * t, from sorry,
  have h12 : 8 * t = 256, from sorry,
  have ht : t = 32, from sorry,
  -- Compute d
  have hd : d = 8 * t, from h4,
  rw ht at hd,
  exact hd.symm,
}

end pascal_remaining_distance_l116_116693


namespace cos_angle_identity_l116_116576

def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (2, 2)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def cos_angle (v1 v2 : ℝ × ℝ) : ℝ :=
  dot_product v1 v2 / (magnitude v1 * magnitude v2)

theorem cos_angle_identity :
  cos_angle (a.1 + b.1, a.2 + b.2) (a.1 - b.1, a.2 - b.2) = (real.sqrt 17) / 17 :=
by
  finish

end cos_angle_identity_l116_116576


namespace remaining_distance_proof_l116_116707

-- Define the conditions
def pascal_current_speed : ℝ := 8
def pascal_reduced_speed : ℝ := pascal_current_speed - 4
def pascal_increased_speed : ℝ := pascal_current_speed * 1.5

-- Define the remaining distance in terms of the current speed and time taken
noncomputable def remaining_distance (T : ℝ) : ℝ := pascal_current_speed * T

-- Define the times with the increased and reduced speeds
noncomputable def time_with_increased_speed (T : ℝ) : ℝ := T - 16
noncomputable def time_with_reduced_speed (T : ℝ) : ℝ := T + 16

-- Define the distances using increased and reduced speeds
noncomputable def distance_increased_speed (T : ℝ) : ℝ := pascal_increased_speed * (time_with_increased_speed T)
noncomputable def distance_reduced_speed (T : ℝ) : ℝ := pascal_reduced_speed * (time_with_reduced_speed T)

-- Main theorem stating that the remaining distance is 256 miles
theorem remaining_distance_proof (T : ℝ) (ht_eq: pascal_current_speed * T = 256) : 
  remaining_distance T = 256 := by
  sorry

end remaining_distance_proof_l116_116707


namespace angle_P_in_quadrilateral_l116_116610

theorem angle_P_in_quadrilateral : 
  ∀ (P Q R S : ℝ), (P = 3 * Q) → (P = 4 * R) → (P = 6 * S) → (P + Q + R + S = 360) → P = 206 := 
by
  intros P Q R S hP1 hP2 hP3 hSum
  sorry

end angle_P_in_quadrilateral_l116_116610


namespace solve_trig_eq_l116_116314

theorem solve_trig_eq (k : ℤ) : ∃ x : ℝ, 
  8.438 * cos (x - π / 4) * (1 - 4 * cos (2 * x) ^ 2) - 2 * cos (4 * x) = 3 ∧
  x = π / 4 * (8 * k + 1) :=
begin
  sorry
end

end solve_trig_eq_l116_116314


namespace num_subsets_l116_116040

theorem num_subsets (A : Set ℕ) :
  (∀ a, a ∈ A → a ∈ {1, 2, 3, 4}) ∧ (∃ s, {2, 3} ⊆ s ∧ s ⊆ A) →
  ∃ n, n = 4 :=
by
  sorry

end num_subsets_l116_116040


namespace house_number_count_l116_116048

noncomputable def count_valid_house_numbers : Nat :=
  let two_digit_primes := [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
  let valid_combinations := two_digit_primes.product two_digit_primes |>.filter (λ (WX, YZ) => WX ≠ YZ)
  valid_combinations.length

theorem house_number_count : count_valid_house_numbers = 110 :=
  by
    sorry

end house_number_count_l116_116048


namespace five_to_one_ratio_to_eleven_is_fifty_five_l116_116393

theorem five_to_one_ratio_to_eleven_is_fifty_five (y : ℚ) (h : 5 / 1 = y / 11) : y = 55 :=
by
  sorry

end five_to_one_ratio_to_eleven_is_fifty_five_l116_116393


namespace intersection_equality_l116_116112

def A : Set (ℝ × ℝ) := 
  { p | let x := p.1 in let y := p.2 in x^2 - y^2 = x / (x^2 + y^2) }

def B : Set (ℝ × ℝ) := 
  { p | let x := p.1 in let y := p.2 in 2 * x * y + y / (x^2 + y^2) = 3 }

def C : Set (ℝ × ℝ) := 
  { p | let x := p.1 in let y := p.2 in x^3 - 3 * x * y^2 + 3 * y = 1 }

def D : Set (ℝ × ℝ) := 
  { p | let x := p.1 in let y := p.2 in 3 * x^2 * y - 3 * x - y^3 = 0 }

theorem intersection_equality : A ∩ B = C ∩ D := sorry

end intersection_equality_l116_116112


namespace opposite_of_neg_2023_l116_116850

def opposite (n : ℤ) : ℤ :=
  -n

theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116850


namespace tan_angle_addition_l116_116150

theorem tan_angle_addition (x : Real) (h1 : Real.tan x = 3) (h2 : Real.tan (Real.pi / 3) = Real.sqrt 3) : 
  Real.tan (x + Real.pi / 3) = (3 + Real.sqrt 3) / (1 - 3 * Real.sqrt 3) :=
by 
  sorry

end tan_angle_addition_l116_116150


namespace magnitude_of_z_l116_116663

open Complex

theorem magnitude_of_z (z : ℂ) (h : z^2 + Complex.normSq z = 4 - 7 * Complex.I) : 
  Complex.normSq z = 65 / 8 := 
by
  sorry

end magnitude_of_z_l116_116663


namespace opposite_neg_2023_l116_116758

theorem opposite_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_neg_2023_l116_116758


namespace opposite_of_neg_2023_l116_116935

theorem opposite_of_neg_2023 : ∃ x : ℤ, -2023 + x = 0 ∧ x = 2023 := 
by 
  use 2023
  constructor
  · exact rfl
  · exact rfl

end opposite_of_neg_2023_l116_116935


namespace num_seven_digit_palindromes_l116_116067

theorem num_seven_digit_palindromes : 
  let a_choices := 9
  let b_choices := 10
  let c_choices := 10
  let d_choices := 10
  a_choices * b_choices * c_choices * d_choices = 9000 :=
by
  sorry

end num_seven_digit_palindromes_l116_116067


namespace paul_final_balance_l116_116716

def initial_balance : ℝ := 400
def transfer1 : ℝ := 90
def transfer2 : ℝ := 60
def service_charge_rate : ℝ := 0.02

def service_charge (x : ℝ) : ℝ := service_charge_rate * x

def total_deduction : ℝ := transfer1 + service_charge transfer1 + service_charge transfer2

def final_balance (init_balance : ℝ) (deduction : ℝ) : ℝ := init_balance - deduction

theorem paul_final_balance :
  final_balance initial_balance total_deduction = 307 :=
by
  sorry

end paul_final_balance_l116_116716


namespace tan_add_pi_div_three_l116_116154

theorem tan_add_pi_div_three (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = -((6 + 5 * Real.sqrt 3) / 13) := 
by 
  sorry

end tan_add_pi_div_three_l116_116154


namespace not_perfect_square_l116_116294

theorem not_perfect_square (n : ℕ) : ¬ ∃ k : ℕ, (3^n + 2 * 17^n) = k^2 :=
by
  sorry

end not_perfect_square_l116_116294


namespace minimize_elements_in_A_n_l116_116092

-- Given conditions
def isRegularPrism (prism : Type) (edgeLength : ℝ) (vertices : ℕ) : Prop :=
  -- Conditions about regular prism with edge length 1 and 2n vertices
  edgeLength = 1 ∧ ∃ n, vertices = 2 * n

def vectorBetweenVertices (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v2.1 - v1.1, v2.2 - v1.2, v2.3 - v1.3)

def edgeVectorAB (A B : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  vectorBetweenVertices A B

def dotProduct (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def setA_n (vectors : List (ℝ × ℝ × ℝ)) (AB : ℝ × ℝ × ℝ) : Set ℝ :=
  { x | ∃ v ∈ vectors, x = dotProduct v AB }

-- Statement to prove
theorem minimize_elements_in_A_n (n : ℕ) (v1 v2 A B : ℝ × ℝ × ℝ)
  (prism : Type) (vectors : List (ℝ × ℝ × ℝ))
  (h_prism : isRegularPrism prism 1 (2 * n))
  (h_edge : AB = edgeVectorAB A B)
  (h_vectors : ∀ v ∈ vectors, v = vectorBetweenVertices v1 v2) :
  n = 4 ↔ (∀ m < n, |setA_n vectors AB| ≥ |setA_n (vectors.filter (≠ AB)) AB|) ∧
    |setA_n vectors AB| < (∀ m > n, |setA_n vectors AB|) :=
by
  sorry

end minimize_elements_in_A_n_l116_116092


namespace opposite_of_neg_2023_l116_116981

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l116_116981


namespace probability_sum_odd_of_three_selected_primes_l116_116365

theorem probability_sum_odd_of_three_selected_primes :
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
  in 12 ∈ primes →
     (∃ selected_set : Finset ℕ, selected_set.card = 12 ∧ selected_set ⊆ primes.to_finset) →
     (∃ selected_three : Finset ℕ, selected_three.card = 3 ∧ selected_three ⊆ primes.to_finset) →
     ∃ selected : Finset ℕ, selected.card = 3 ∧ (∑ x in selected, x % 2 = 1) ∧ 
     ((∃ selected_with_2 : Finset ℕ, (2 ∈ selected_with_2 ∧ selected_with_2.card = 3))
      ∨ 
      (∀ x ∈ selected_with_2, x ≠ 2 ∧ selected_with_2.card = 3)) →
     (91 / 455 = (1 / 5) : ℚ) := 
  by
    sorry

end probability_sum_odd_of_three_selected_primes_l116_116365


namespace advertising_department_size_l116_116424

-- Define the conditions provided in the problem.
def total_employees : Nat := 1000
def sample_size : Nat := 80
def advertising_sample_size : Nat := 4

-- Define the main theorem to prove the given problem.
theorem advertising_department_size :
  ∃ n : Nat, (advertising_sample_size : ℚ) / n = (sample_size : ℚ) / total_employees ∧ n = 50 :=
by
  sorry

end advertising_department_size_l116_116424


namespace ratio_of_tuna_to_salmon_l116_116230

theorem ratio_of_tuna_to_salmon 
  (trout_weight : ℕ) (salmon_weight : ℕ) (total_weight : ℕ) (tuna_weight : ℕ) 
  (h1 : trout_weight = 200)
  (h2 : salmon_weight = trout_weight + (0.5 * trout_weight)) 
  (h3 : total_weight = 1100)
  (h4 : tuna_weight = total_weight - trout_weight - salmon_weight) :
  (tuna_weight / salmon_weight) = 2 := 
sorry

end ratio_of_tuna_to_salmon_l116_116230


namespace opposite_of_neg_2023_l116_116883

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116883


namespace smallest_s_for_triangle_7_5_s_13_l116_116450

theorem smallest_s_for_triangle_7_5_s_13 :
  ∃ s : ℕ, (7.5 + s > 13) ∧ (7.5 + 13 > s) ∧ (s + 13 > 7.5) ∧ (s = 6) :=
by
  sorry

end smallest_s_for_triangle_7_5_s_13_l116_116450


namespace opposite_of_neg_2023_l116_116871

-- Definition of the problem
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  -- The proof is to be provided
  sorry

end opposite_of_neg_2023_l116_116871


namespace opposite_of_neg_2023_l116_116824

/-- Define the opposite of a number -/
def opposite (x : ℤ) : ℤ := -x

/-- Prove the opposite of -2023 is 2023 -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116824


namespace age_difference_l116_116200

-- Defining the necessary variables and their types
variables (A B : ℕ)

-- Given conditions: 
axiom B_current_age : B = 38
axiom future_age_relationship : A + 10 = 2 * (B - 10)

-- Proof goal statement
theorem age_difference : A - B = 8 :=
by
  sorry

end age_difference_l116_116200


namespace floor_sqrt_10_l116_116052

theorem floor_sqrt_10 : (Real.floor (Real.sqrt 10) = 3) :=
by
  have h1 : Real.sqrt 9 = 3 := by norm_num
  have h2 : Real.sqrt 16 = 4 := by norm_num
  have h3 : 3 ≤ Real.sqrt 10 ∧ Real.sqrt 10 < 4 := ⟨by linarith [h1], by linarith [h2]⟩
  sorry

end floor_sqrt_10_l116_116052


namespace min_major_axis_ellipse_exists_l116_116339

def inclination_angle : ℝ := Real.pi / 4
def y_intercept : ℝ := 3
def hyperbola_eq (x y : ℝ) : Prop := 12 * x^2 - 4 * y^2 = 3

theorem min_major_axis_ellipse_exists : 
  ∃ a b : ℝ, 
    (a^2 = 5) ∧ 
    (b^2 = 4) ∧ 
    ∀ x y : ℝ, 
      (hyperbola_eq x y) → 
      (∃ x' y', 
        (x = x') ∧ 
        (y = y' + 3) ∧ 
        ((x' / a)^2 + (y' / b)^2 = 1)) :=
sorry

end min_major_axis_ellipse_exists_l116_116339


namespace perpendicular_AR_AD_l116_116010

variables {A B C D P Q R H : Point}

-- We assume the necessary geometric conditions
axiom angle_bisector (ABC : Triangle) : bisector A D ∈ ABC

axiom parallels (BP CQ AD : Line) : BP ∥ CQ ∧ CQ ∥ AD

axiom product_condition (BP CQ AB AC : Segment) : BP.length * CQ.length = AB.length * AC.length

axiom orthocenter (PQR : Triangle) : orthocenter H P Q R

-- Final goal: AR ⊥ AD
theorem perpendicular_AR_AD :
  angle_bisector (triangle A B C) →
  parallels (line B P) (line C Q) (line A D) →
  product_condition (segment B P) (segment C Q) (segment A B) (segment A C) →
  orthocenter (triangle P Q R) H →
  perp (line A R) (line A D) :=
by
sory

end perpendicular_AR_AD_l116_116010


namespace min_value_frac_inv_l116_116647

noncomputable def min_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 12) : ℝ :=
  1 / a + 1 / b

theorem min_value_frac_inv (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 12) :
  min_value a b h1 h2 h3 = 1 / 3 :=
sorry

end min_value_frac_inv_l116_116647


namespace ways_to_place_balls_into_boxes_l116_116139

theorem ways_to_place_balls_into_boxes :
  let n := 5 in
  let k := 3 in
  let ways := Nat.choose (n + k - 1) (k - 1) in
  ways = 21 :=
by
  let n := 5
  let k := 3
  let ways := Nat.choose (n + k - 1) (k - 1)
  exact (by simp [ways])

end ways_to_place_balls_into_boxes_l116_116139


namespace choose_officers_count_l116_116284

variable (members : Finset ℕ) (seniorMembers : Finset ℕ)
variable (club_members : members.card = 12)
variable (senior_members_limit : seniorMembers.card = 4)

theorem choose_officers_count :
  (4 * 11 * 10 * 9 * 8 = 31680) 
  → 
  ∃ president vice_president secretary treasurer morale_officer, 
  president ∈ seniorMembers ∧ 
  (∀ x, x ∈ {vice_president, secretary, treasurer, morale_officer} → x ∈ (members \ {president}) ∧
     x ≠ president ∧ 
     vice_president ≠ secretary ∧
     vice_president ≠ treasurer ∧
     vice_president ≠ morale_officer ∧
     secretary ≠ treasurer ∧
     secretary ≠ morale_officer ∧
     treasurer ≠ morale_officer) := sorry

end choose_officers_count_l116_116284


namespace find_f_at_2_l116_116106

-- Definition of the function f and its inverse
variable (f : ℝ → ℝ)
variable (f_inv : ℝ → ℝ)

-- Given conditions: 
-- f_inv is the inverse function of f, defined as f_inv(x) = sqrt(x - 1)
def f_inv_def (x : ℝ) := f_inv x = Real.sqrt (x - 1)

-- The function f has an inverse, so we have f(f_inv(x)) = x
def f_left_inverse (x : ℝ) := f(f_inv x) = x

-- Prove that f(2) = 5 given the above conditions
theorem find_f_at_2 : f_inv 5 = 2 → f 2 = 5 :=
by 
  assume h : f_inv 5 = 2
  -- Proof is omitted
  sorry

end find_f_at_2_l116_116106


namespace opposite_of_neg_2023_l116_116914

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by 
  sorry

end opposite_of_neg_2023_l116_116914


namespace smiths_bakery_pies_l116_116306

theorem smiths_bakery_pies (m : ℕ) (h : m = 16) : 4 * m + 6 = 70 :=
by
  rw [h]
  norm_num
  sorry

end smiths_bakery_pies_l116_116306


namespace tan_shifted_value_l116_116165

theorem tan_shifted_value (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = -((6 + 5 * Real.sqrt 3) / 13) :=
by
  sorry

end tan_shifted_value_l116_116165


namespace number_of_finishing_orders_l116_116135

theorem number_of_finishing_orders (p1 p2 p3 p4 : Prop) : finset.card 
  (finset.univ.permutations : finset (list Prop)).card = 24 := 
by
  sorry

end number_of_finishing_orders_l116_116135


namespace leading_coefficient_polynomial_l116_116061

def polynomial : Polynomial ℤ :=
  -5 * (Polynomial.monomial 5 1 - Polynomial.monomial 4 1 + Polynomial.monomial 1 2)
  + 9 * (Polynomial.monomial 5 1 + Polynomial.C 3)
  - 6 * (Polynomial.monomial 5 3 + Polynomial.monomial 3 1 + Polynomial.C 2)

theorem leading_coefficient_polynomial :
  Polynomial.leadingCoeff (polynomial) = -14 :=
by
  sorry

end leading_coefficient_polynomial_l116_116061


namespace salary_increase_exceeds_100_percent_l116_116682

theorem salary_increase_exceeds_100_percent (S : ℝ) (h : 0 < S) :
  let factor := 1.15 in
  let final_salary := S * factor^5 in
  final_salary > S * 2 :=
by sorry

end salary_increase_exceeds_100_percent_l116_116682


namespace relationship_between_a_b_l116_116739

theorem relationship_between_a_b (a b x : ℝ) 
  (h₁ : x = (a + b) / 2)
  (h₂ : x^2 = (a^2 - b^2) / 2):
  a = -b ∨ a = 3 * b :=
sorry

end relationship_between_a_b_l116_116739


namespace westgate_high_school_chemistry_l116_116012

theorem westgate_high_school_chemistry :
  ∀ (total_players physics_both physics : ℕ),
    total_players = 15 →
    physics_both = 3 →
    physics = 8 →
    (total_players - (physics - physics_both)) - physics_both = 10 := by
  intros total_players physics_both physics h1 h2 h3
  sorry

end westgate_high_school_chemistry_l116_116012


namespace determinant_roots_cubic_eqn_zero_l116_116656

variable {α : Type*} [Field α]

theorem determinant_roots_cubic_eqn_zero {a b c p q r : α}
  (h_eqn : ∀ x : α, x^3 - p * x^2 + q * x - r = 0) :
  Matrix.det (Matrix.vec3 (a + b) (b + c) (c + a) (b + c) (c + a) (a + b) (c + a) (a + b) (b + c)) = 0 := 
sorry

end determinant_roots_cubic_eqn_zero_l116_116656


namespace opposite_of_negative_2023_l116_116786

theorem opposite_of_negative_2023 : (- -2023) = 2023 := 
by {
sorry
}

end opposite_of_negative_2023_l116_116786


namespace num_8_digit_integers_l116_116580

theorem num_8_digit_integers : 
  let number_of_first_digits := 9,
      number_of_middle_digits := 10,
      number_of_last_digits := 8,
      total_number := number_of_first_digits * (number_of_middle_digits ^ 6) * number_of_last_digits
  in total_number = 72000000 := by
  sorry

end num_8_digit_integers_l116_116580


namespace opposite_of_neg_2023_l116_116931

theorem opposite_of_neg_2023 : ∃ x : ℤ, -2023 + x = 0 ∧ x = 2023 := 
by 
  use 2023
  constructor
  · exact rfl
  · exact rfl

end opposite_of_neg_2023_l116_116931


namespace arithmetic_general_term_sum_b_terms_l116_116535

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ a1 d, ∀ n, a n = a1 + (n - 1) * d

def sum_arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

def b_sequence (b : ℕ → ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ n, b n = 2^(a n) + 2 * n

noncomputable def sum_b_sequence (b : ℕ → ℤ) (T : ℕ → ℤ) : Prop :=
  ∀ n, T n = (2 * 4^n + 3 * n^2 + 3 * n - 2) / 3

theorem arithmetic_general_term :
  ∀ (a : ℕ → ℤ) (S : ℕ → ℤ),
  (arithmetic_sequence a) →
  (sum_arithmetic_sequence a S) →
  a 3 = 5 →
  S 15 = 225 →
  ∀ n, a n = 2 * n - 1 :=
by
  intros a S ha hS ha3 hS15 n
  sorry

theorem sum_b_terms :
  ∀ (a b : ℕ → ℤ) (S T : ℕ → ℤ),
  (arithmetic_sequence a) →
  (sum_arithmetic_sequence a S) →
  a 3 = 5 →
  S 15 = 225 →
  (b_sequence b a) →
  (sum_b_sequence b T) →
  ∀ n, T n = (2 * 4^n + 3 * n^2 + 3 * n - 2) / 3 :=
by
  intros a b S T ha hS ha3 hS15 hb hT n
  sorry

end arithmetic_general_term_sum_b_terms_l116_116535


namespace tan_add_pi_over_3_l116_116175

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = - (6 + 5 * Real.sqrt 3) / 13 :=
sorry

end tan_add_pi_over_3_l116_116175


namespace total_weight_moved_l116_116235

-- Given conditions as definitions
def initial_back_squat : ℝ := 200
def back_squat_increase : ℝ := 50
def front_squat_ratio : ℝ := 0.8
def triple_ratio : ℝ := 0.9

-- Proving the total weight moved for three triples
theorem total_weight_moved : 
  let new_back_squat := initial_back_squat + back_squat_increase in
  let front_squat := new_back_squat * front_squat_ratio in
  let weight_per_triple := front_squat * triple_ratio in
  3 * weight_per_triple = 540 :=
by
  sorry

end total_weight_moved_l116_116235


namespace opposite_of_neg_2023_l116_116907

-- Let x be the opposite of -2023
def is_opposite (x : ℤ) (n : ℤ) : Prop := n + x = 0

theorem opposite_of_neg_2023 : ∃ x : ℤ, is_opposite x (-2023) ∧ x = 2023 :=
by
  exists 2023
  unfold is_opposite
  sorry

end opposite_of_neg_2023_l116_116907


namespace repeating_decimal_sum_l116_116057

theorem repeating_decimal_sum :
  (sum_repeating_decimal 0.2 0.04) = (200 / 769) :=
sorry

end repeating_decimal_sum_l116_116057


namespace problem_statement_l116_116658

noncomputable def s : ℝ := Classical.choose (Exists.choose_spec (real.exists_pos_of_has_deriv_at_true (λ x, x^3 + (1/4)*x - 1) (by continuity)))

lemma series_sum (T : ℝ) (h : s^3 + (1/4)*s - 1 = 0) : T = s^3 + 2*s^7 + 3*s^11 + 4*s^15 + ∑' n, (n+1) * s^(4*n+3) :=
begin
  sorry
end

theorem problem_statement (h : s^3 + (1/4) * s - 1 = 0) : s^3 + 2*s^7 + 3*s^11 + 4*s^15 + ∑' n, (n+1) * s^(4*n+3) = 16*s :=
begin
  sorry
end

end problem_statement_l116_116658


namespace arithmetic_geometric_sequence_l116_116323

theorem arithmetic_geometric_sequence (a1 d : ℝ) (h1 : a1 = 1) (h2 : d ≠ 0) (h_geom : (a1 + d) ^ 2 = a1 * (a1 + 4 * d)) :
  d = 2 :=
by
  sorry

end arithmetic_geometric_sequence_l116_116323


namespace quadratic_function_expression_value_at_8_existence_of_x_for_given_y_l116_116091

def f (x : ℝ) : ℝ := x^2 + 2 * x + 2

theorem quadratic_function_expression :
  ∃ (a b c : ℝ), a ≠ 0 ∧ b = 2 ∧ c = 2 ∧ ∀ x, f x = a * x^2 + b * x + c :=
begin
  use [1, 2, 2],
  simp,
  split,
  { exact one_ne_zero, },
  split,
  { refl, },
  { refl, },
  { intros, simp [f], },
end

theorem value_at_8 :
  f 8 = 82 :=
by norm_num [f]

theorem existence_of_x_for_given_y (y : ℝ) :
  (∃ x : ℝ, y = f x) ↔ y ≥ 1 :=
begin
  split,
  { intro h,
    rcases h with ⟨x, hx⟩,
    simp [f] at hx,
    have : (x + 1)^2 ≥ 0 := pow_two_nonneg (x + 1),
    linarith, },
  { intro hy,
    use (real.sqrt (y - 1) - 1),
    simp [f],
    field_simp [(show (real.sqrt (y - 1)) ^ 2 = y - 1, by simp [sq_sqrt (sub_nonneg.2 hy)]), hy] },
end

end quadratic_function_expression_value_at_8_existence_of_x_for_given_y_l116_116091


namespace exists_coloring_no_monochromatic_seq_l116_116050

def A := finset.Icc 1 2017

def is_coloring (f : ℕ → bool) : Prop := ∀ x ∈ A, f x = tt ∨ f x = ff

noncomputable def no_monochromatic_seq (f : ℕ → bool) (n : ℕ) : Prop :=
  ∀ a r, (∀ i, a + i * r ∈ A) → r ≠ 0 →
    (∃ i1 i2, i1 ≠ i2 ∧ f (a + i1 * r) ≠ f (a + i2 * r))

theorem exists_coloring_no_monochromatic_seq (n : ℕ) (hn : n ≥ 18) :
  ∃ f : ℕ → bool, is_coloring f ∧ no_monochromatic_seq f n := by
  sorry

end exists_coloring_no_monochromatic_seq_l116_116050


namespace point_P_not_on_line_l_difference_max_min_distance_Q_line_l_l116_116608

noncomputable def parametric_line_l (t : ℝ) : ℝ × ℝ :=
  (t / 2, (sqrt 3) / 2 * t + 1)

noncomputable def parametric_curve_C (θ : ℝ) : ℝ × ℝ :=
  (2 + cos θ, sin θ)

noncomputable def polar_to_cartesian (r θ : ℝ) : ℝ × ℝ :=
  (r * cos θ, r * sin θ)

theorem point_P_not_on_line_l :
  let P := polar_to_cartesian 4 (π / 3) in
  ¬ ∃ t, parametric_line_l t = P :=
sorry

theorem difference_max_min_distance_Q_line_l :
  let C_center := (2, 0) in
  let d := abs ((2 * sqrt 3 * 0 + 1) / (sqrt (3 ^ 2 + 1))) in
  let r := 1 in
  let max_d := d + r in
  let min_d := d - r in
  max_d - min_d = 2 :=
sorry

end point_P_not_on_line_l_difference_max_min_distance_Q_line_l_l116_116608


namespace prove_center_of_symmetry_l116_116563

noncomputable def center_of_symmetry_of_sine_function (ω : ℝ) (φ : ℝ) : Prop :=
  ∀ (x : ℝ), 
    (ω > 0) ∧ 
    (-real.pi / 2 < φ) ∧ (φ < real.pi / 2) ∧ 
    (simple_period : ℝ) =
    (2 * real.pi / ω = real.pi) ∧
    (symmetry_line : ℝ) =
    (2 * (2 * real.pi / 3) + φ = real.pi / 2 + k * real.pi) →
    φ = k * real.pi - 5 * real.pi / 6 →
    f(x) = real.sin (2 * x + φ) →
    (center : ℝ) = (-real.pi / 12, 0)

theorem prove_center_of_symmetry 
   (ω : ℝ) (φ : ℝ) (k : ℤ) (f : ℝ → ℝ) :
  (ω > 0) → 
  (-real.pi / 2 < φ) → (φ < real.pi / 2) → 
  (2 * real.pi / ω = real.pi) → 
  (2 * (2 * real.pi / 3) + φ = real.pi / 2 + k * real.pi) →
  φ = k * real.pi - 5 * real.pi / 6 →
  f = λ x, real.sin (2 * x + φ) →
  (-real.pi / 12, 0) := 
sorry

end prove_center_of_symmetry_l116_116563


namespace integral_equality_a_eq_2_l116_116140

theorem integral_equality_a_eq_2 (a : ℝ) :
  (∫ x in 1..2, (x - a)) = (∫ x in 0..(3 * Real.pi / 4), Real.cos (2 * x)) → a = 2 :=
by
  sorry

end integral_equality_a_eq_2_l116_116140


namespace compute_expression_l116_116030

/-- Definitions of parts of the expression --/
def expr1 := 6 ^ 2
def expr2 := 4 * 5
def expr3 := 2 ^ 3
def expr4 := 4 ^ 2 / 2

/-- Main statement to prove --/
theorem compute_expression : expr1 + expr2 - expr3 + expr4 = 56 := 
by
  sorry

end compute_expression_l116_116030


namespace opposite_of_neg_2023_l116_116895

-- Let x be the opposite of -2023
def is_opposite (x : ℤ) (n : ℤ) : Prop := n + x = 0

theorem opposite_of_neg_2023 : ∃ x : ℤ, is_opposite x (-2023) ∧ x = 2023 :=
by
  exists 2023
  unfold is_opposite
  sorry

end opposite_of_neg_2023_l116_116895


namespace angle_B_possibilities_l116_116250

variables {A B C O H : Type*}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space O] [metric_space H]
variables [affine_space A] [affine_space B] [affine_space C]

def is_circumcenter (O : affine_space) (A B C : affine_space) : Prop := sorry
def is_orthocenter (H : affine_space) (A B C : affine_space) : Prop := sorry

theorem angle_B_possibilities {A B C : Type*} [metric_space A] [metric_space B] [metric_space C]
  [affine_space A] [affine_space B] [affine_space C] {O H : affine_space}
  (h_circumcenter : is_circumcenter O A B C) (h_orthocenter : is_orthocenter H A B C)
  (h_eq : dist B O = dist B H) :
  ∃ (α : ℝ), α = 60 ∨ α = 120 :=
sorry

end angle_B_possibilities_l116_116250


namespace limit_f_1_minus_h_l116_116562

open Real

def f (x : ℝ) : ℝ := 1 / x

theorem limit_f_1_minus_h (f : ℝ → ℝ) (h : ℝ) (h_ne: h ≠ 0) :
  (f = λ x, 1 / x) → (tendsto (λ h, (f(1 - h) - f(1)) / h) (nhds 0) (nhds 1)) :=
by
  intro hf
  rw [hf]
  sorry

end limit_f_1_minus_h_l116_116562


namespace opposite_of_neg_2023_l116_116950

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l116_116950


namespace special_discount_percentage_l116_116463

theorem special_discount_percentage (original_price discounted_price : ℝ) (h₀ : original_price = 80) (h₁ : discounted_price = 68) : 
  ((original_price - discounted_price) / original_price) * 100 = 15 :=
by 
  sorry

end special_discount_percentage_l116_116463


namespace paul_account_balance_after_transactions_l116_116722

theorem paul_account_balance_after_transactions :
  let transfer1 := 90
  let transfer2 := 60
  let service_charge_percent := 2 / 100
  let initial_balance := 400
  let service_charge1 := service_charge_percent * transfer1
  let service_charge2 := service_charge_percent * transfer2
  let total_deduction1 := transfer1 + service_charge1
  let total_deduction2 := transfer2 + service_charge2
  let reversed_transfer2 := total_deduction2 - transfer2
  let balance_after_transfer1 := initial_balance - total_deduction1
  let final_balance := balance_after_transfer1 - reversed_transfer2
  (final_balance = 307) :=
by
  sorry

end paul_account_balance_after_transactions_l116_116722


namespace infinite_unsum_power_subset_l116_116047

-- Definitions and properties needed from the problem statement
def is_power (n : ℕ) : Prop := 
  ∃ a b : ℕ, a ≥ 2 ∧ b ≥ 2 ∧ n = a^b

def special_subset (A : Set ℕ) : Prop :=
  ∃ (q : ℕ → ℕ), (∀ i, q i ∈ A) ∧ (∀ (I : Finset ℕ), ¬ is_power (∑ i in I, q i))

-- Main statement
theorem infinite_unsum_power_subset : ∃ A : Set ℕ, Set.Infinite A ∧ special_subset A :=
sorry

end infinite_unsum_power_subset_l116_116047


namespace opposite_of_neg2023_l116_116991

theorem opposite_of_neg2023 : ∀ (x : Int), -2023 + x = 0 → x = 2023 :=
by
  intros x h
  sorry

end opposite_of_neg2023_l116_116991


namespace base_for_256_l116_116479

theorem base_for_256 := 
  ∃ b : ℕ, (b^3 ≤ 256 ∧ 256 < b^4) ∧ b = 6 := sorry

end base_for_256_l116_116479


namespace total_fiscal_revenue_scientific_notation_l116_116599

theorem total_fiscal_revenue_scientific_notation : 
  ∃ a n, (1073 * 10^8 : ℝ) = a * 10^n ∧ (1 ≤ |a| ∧ |a| < 10) ∧ a = 1.07 ∧ n = 11 :=
by
  use 1.07, 11
  simp
  sorry

end total_fiscal_revenue_scientific_notation_l116_116599


namespace opposite_of_neg_2023_l116_116916

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by 
  sorry

end opposite_of_neg_2023_l116_116916


namespace Pascal_remaining_distance_l116_116697

theorem Pascal_remaining_distance (D T : ℕ) :
  let current_speed := 8
  let reduced_speed := current_speed - 4
  let increased_speed := current_speed + current_speed / 2
  (D = current_speed * T) →
  (D = reduced_speed * (T + 16)) →
  (D = increased_speed * (T - 16)) →
  D = 256 :=
by
  intros
  sorry

end Pascal_remaining_distance_l116_116697


namespace opposite_of_neg_2023_l116_116839

def opposite (n : ℤ) : ℤ :=
  -n

theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116839


namespace arg_z_range_l116_116523

noncomputable def arg_range (z : ℂ) : Set ℝ :=
  {θ : ℝ | θ ∈ (Set.Ioo (5 * Real.pi / 6 - Real.arcsin (Real.sqrt 3 / 3)) Real.pi ∪ 
                Set.Ioo Real.pi (7 * Real.pi / 6 + Real.arcsin (Real.sqrt 3 / 3)))}

theorem arg_z_range (z : ℂ) (hz : abs (Complex.arg ((z + 1) / (z + 2))) = Real.pi / 6) :
  arg z ∈ arg_range z :=
  sorry

end arg_z_range_l116_116523


namespace part_I_general_term_seq_S_n_general_term_seq_a_part_II_l116_116641

noncomputable def seq_S (n : ℕ) : ℝ := sorry  -- Place the definition of S_n here
def seq_a (n : ℕ) : ℝ :=
if n = 1 then 1
else -2 / (2 * n - 1) / (2 * n - 3)

def seq_b (n : ℕ) : ℝ := (seq_S n) / (2 * n + 1)

theorem part_I : ∀ n : ℕ, n > 0 → 
  (1 / (seq_S (n + 1)) - 1 / (seq_S n) = 2) :=
sorry  -- Proof that {1 / S_n} is an arithmetic sequence with common difference 2.

theorem general_term_seq_S_n : ∀ n : ℕ, n > 0 → 
  seq_S n = 1 / (2 * n - 1) :=
sorry  -- Proof that S_n = 1 / (2n - 1).

theorem general_term_seq_a : ∀ n : ℕ, n > 0 → 
  seq_a n = if n = 1 then 1 else -2 / (2 * n - 1) / (2 * n - 3) :=
sorry  -- Proof of the general term for a_n.

theorem part_II : ∀ n : ℕ, n > 0 → 
  let T_n := ∑ k in finset.range n, seq_b (k + 1) in T_n = n / (2 * n + 1) :=
sorry  -- Proof that T_n = n / (2n + 1).

end part_I_general_term_seq_S_n_general_term_seq_a_part_II_l116_116641


namespace boxes_used_l116_116431

-- Define the given conditions
def oranges_per_box : ℕ := 10
def total_oranges : ℕ := 2650

-- Define the proof statement
theorem boxes_used : total_oranges / oranges_per_box = 265 :=
by
  -- Proof goes here
  sorry

end boxes_used_l116_116431


namespace complex_quadrant_l116_116748

theorem complex_quadrant (z : ℂ) (h : (1 + complex.I)^2 * z = -1 + complex.I) : 
  ∃ x y : ℝ, z = x + y * complex.I ∧ 0 < x ∧ 0 < y := 
sorry

end complex_quadrant_l116_116748


namespace third_person_time_l116_116367

theorem third_person_time
  (combined_time : ℝ)
  (lowest_fraction : ℝ)
  (time_A : ℝ)
  (time_B : ℝ)
  (h1 : combined_time = 1)
  (h2 : lowest_fraction = 0.29166666666666663)
  (h3 : time_A = 4)
  (h4 : time_B = 6) :
  let rate_A := 1 / time_A
  let rate_B := 1 / time_B
  let combined_rate := rate_A + rate_B
  let rate_C := lowest_fraction
  let time_C := 1 / rate_C in 
  time_C ≈ 3.43 :=
by
  sorry

end third_person_time_l116_116367


namespace solve_triangle_l116_116598

open Real

noncomputable def triangle_sides_angles (a b c A B C : ℝ) : Prop :=
  b^2 - (2 * (sqrt 3 / 3) * b * c * sin A) + c^2 = a^2

theorem solve_triangle 
  (b c : ℝ) (hb : b = 2) (hc : c = 3)
  (h : triangle_sides_angles a b c A B C) : 
  (A = π / 3) ∧ 
  (a = sqrt 7) ∧ 
  (sin (2 * B - A) = 3 * sqrt 3 / 14) := 
by
  sorry

end solve_triangle_l116_116598


namespace tan_add_pi_div_three_l116_116155

theorem tan_add_pi_div_three (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = -((6 + 5 * Real.sqrt 3) / 13) := 
by 
  sorry

end tan_add_pi_div_three_l116_116155


namespace total_number_of_coins_l116_116625

theorem total_number_of_coins {N B : ℕ} 
    (h1 : B - 2 = Nat.floor (N / 9))
    (h2 : N - 6 * (B - 3) = 3) 
    : N = 45 :=
by
  sorry

end total_number_of_coins_l116_116625


namespace min_value_of_one_over_a_plus_one_over_b_l116_116650

theorem min_value_of_one_over_a_plus_one_over_b (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 12) :
  (∃ c : ℝ, (c = 1/ a + 1 / b) ∧ c = 1 / 3) :=
sorry

end min_value_of_one_over_a_plus_one_over_b_l116_116650


namespace polygon_six_sides_l116_116195

theorem polygon_six_sides (n : ℕ) (h1 : (n - 2) * 180 = 2 * 360) : n = 6 := by
  sorry

end polygon_six_sides_l116_116195


namespace opposite_of_neg_2023_l116_116882

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116882


namespace tabby_swimming_speed_l116_116742

theorem tabby_swimming_speed :
  ∃ (S : ℝ), S = 4.125 ∧ (∀ (D : ℝ), 6 = (2 * D) / ((D / S) + (D / 11))) :=
by {
 sorry
}

end tabby_swimming_speed_l116_116742


namespace probability_identifiers_l116_116205

noncomputable def number_of_students_earning_B : ℝ :=
  let x := 45 / (0.5 + 1 + 1.6) in x

theorem probability_identifiers (students : ℝ) (prob_A : ℝ) (prob_B : ℝ) (prob_C : ℝ)
    (h1 : prob_A = 0.5 * prob_B) (h2 : prob_C = 1.6 * prob_B) (h3 : students = 45)
    : students * (prob_A + prob_B + prob_C) / (prob_A + prob_B + prob_C) = 15 :=
by
  rw [h1, h2, h3]
  suffices prob_A + prob_B + prob_C = 3.1 * prob_B by
  simp [number_of_students_earning_B, prob_A, prob_B, prob_C]
  sorry

end probability_identifiers_l116_116205


namespace proposition_A_iff_proposition_B_l116_116215

noncomputable def is_constant (z : Complex) (c : ℝ) : Prop :=
  |z - 3| + |z + 3| = c

def is_ellipse (z : Complex) : Prop :=
  ∃ c > 6, |z - 3| + |z + 3| = c

theorem proposition_A_iff_proposition_B (z : Complex) (c : ℝ) :
  (is_constant z c) ↔ (is_ellipse z) :=
sorry

end proposition_A_iff_proposition_B_l116_116215


namespace PQ_tangent_to_circle_diameter_AB_l116_116579

-- Given data and conditions
variables {O1 O2 : Type} [Circle O1] [Circle O2]
variables {A B C D P Q : Point}
variable {line_through_A : Line A}
variable {tangent_at_C : Line C} 
variable {tangent_at_D : Line D} 
variable {circle_with_diameter_AB : Circle (diameter A B)}

-- Definitions of conditions
variable {circle_intersect : Circle_Intersection O1 O2 A B}
variable {line_A_intersect_C : Line_Circle_Intersection line_through_A O1 A C}
variable {line_A_intersect_D : Line_Circle_Intersection line_through_A O2 A D}
variable {B_proj_P : Projection B tangent_at_C P}
variable {B_proj_Q : Projection B tangent_at_D Q}

-- Theorem statement
theorem PQ_tangent_to_circle_diameter_AB (h : circle_intersect → line_A_intersect_C → line_A_intersect_D → B_proj_P → B_proj_Q) : 
  Tangent(PQ, circle_with_diameter_AB) :=
sorry

end PQ_tangent_to_circle_diameter_AB_l116_116579


namespace paul_account_balance_l116_116720

variable (initial_balance : ℝ) (transfer1 : ℝ) (transfer2 : ℝ) (service_charge_rate : ℝ)

def final_balance (init_bal transfer1 transfer2 rate : ℝ) : ℝ :=
  let charge1 := transfer1 * rate
  let total_deduction := transfer1 + charge1
  init_bal - total_deduction

theorem paul_account_balance :
  initial_balance = 400 →
  transfer1 = 90 →
  transfer2 = 60 →
  service_charge_rate = 0.02 →
  final_balance 400 90 60 0.02 = 308.2 :=
by
  intros h1 h2 h3 h4
  rw [final_balance, h1, h2, h4]
  norm_num

end paul_account_balance_l116_116720


namespace inequality_satisfied_for_all_x_l116_116192

variable (a x : ℝ)

def quadratic_inequality (a x : ℝ) : Prop :=
 (a - 3) * x^2 + 2 * (a - 3) * x - 4 < 0 

theorem inequality_satisfied_for_all_x :
  (∀ x : ℝ, quadratic_inequality a x) ↔ a ∈ set.Ioc (-1: ℝ) 3 :=
sorry

end inequality_satisfied_for_all_x_l116_116192


namespace opposite_neg_2023_l116_116763

theorem opposite_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_neg_2023_l116_116763


namespace opposite_of_neg_2023_l116_116967

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l116_116967


namespace least_positive_k_l116_116241

noncomputable
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

noncomputable
def A : set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}

noncomputable
def exists_prime_sum_in_k_subset (k : ℕ) : Prop :=
  ∀ S ⊆ A, S.card = k → ∃ a b ∈ S, a ≠ b ∧ is_prime (a^2 + b^2)

theorem least_positive_k : exists_prime_sum_in_k_subset 9 :=
sorry

end least_positive_k_l116_116241


namespace magnitude_of_a_l116_116130

variables (a : ℝ × ℝ) (b : ℝ × ℝ) (x : ℝ)

-- Define the vectors a and b
def vec_a : ℝ × ℝ := (1, -x)
def vec_b : ℝ × ℝ := (3 * x - 1, 2)

-- Define orthogonality condition
def orthogonal (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

-- Define magnitude of a vector
def magnitude (a : ℝ × ℝ) : ℝ := Real.sqrt (a.1 ^ 2 + a.2 ^ 2)

-- The Proposition we need to prove
theorem magnitude_of_a :
  orthogonal (vec_a x) (vec_b x) → magnitude (vec_a x) = Real.sqrt 2 :=
by
  sorry

end magnitude_of_a_l116_116130


namespace tan_add_pi_over_3_l116_116163

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + (Real.pi / 3)) = -(6 * Real.sqrt 3 + 2) / 13 := 
by
  sorry

end tan_add_pi_over_3_l116_116163


namespace opposite_of_neg_2023_l116_116854

-- Definition of the problem
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  -- The proof is to be provided
  sorry

end opposite_of_neg_2023_l116_116854


namespace Donggil_cleaning_time_l116_116396

-- Define the total area of the school as A.
variable (A : ℝ)

-- Define the cleaning rates of Daehyeon (D) and Donggil (G).
variable (D G : ℝ)

-- Conditions given in the problem
def condition1 : Prop := (D + G) * 8 = (7 / 12) * A
def condition2 : Prop := D * 10 = (5 / 12) * A

-- The goal is to prove that Donggil can clean the entire area alone in 32 days.
theorem Donggil_cleaning_time : condition1 A D G ∧ condition2 A D → 32 * G = A :=
by
  sorry

end Donggil_cleaning_time_l116_116396


namespace central_angle_correct_l116_116595

-- Define arc length, radius, and central angle
variables (l r α : ℝ)

-- Given conditions
def arc_length := 3
def radius := 2

-- Theorem to prove
theorem central_angle_correct : (l = arc_length) → (r = radius) → (l = r * α) → α = 3 / 2 :=
by
  intros h1 h2 h3
  sorry

end central_angle_correct_l116_116595


namespace magnitude_projection_l116_116643

-- Define the magnitude function
def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Define the dot product of two vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the projection of v onto w
def proj (v w : ℝ × ℝ) : ℝ × ℝ :=
let scaler := (dot_product v w) / (dot_product w w) in (scaler * w.1, scaler * w.2)

-- Problem statement
theorem magnitude_projection (v w : ℝ × ℝ) (h₁ : dot_product v w = 6) (h₂ : magnitude w = 3) :
  magnitude (proj v w) = 2 ∧ ¬ (dot_product v w = 0) :=
by
  sorry

end magnitude_projection_l116_116643


namespace opposite_of_neg_2023_l116_116887

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116887


namespace exists_circle_intersect_B_R_l116_116639

open Set

-- Define the problem conditions and question in Lean
variable (n : ℕ) (hn : 0 < n)
variable (lines : Fin (2 * n) → AffineSubspace ℝ ℝ) 
variable (blue_indices : Fin n → Fin (2 * n))
variable (red_indices : Fin n → Fin (2 * n))
-- Ensure no two lines are parallel and all lines are distinct
variable (distinct_lines :  Function.Injective lines)
variable (no_parallel : ∀ i j, i ≠ j → ¬Parallel (lines i) (lines j))

-- Defining sets of points based on the lines
def blue_points := ⋃ i, ↑(lines (blue_indices i))
def red_points := ⋃ i, ↑(lines (red_indices i))

-- The statement of the theorem
theorem exists_circle_intersect_B_R (n : ℕ) (hn : 0 < n) 
  (lines : Fin (2 * n) → AffineSubspace ℝ ℝ)
  (blue_indices : Fin n → Fin (2 * n))
  (red_indices : Fin n → Fin (2 * n))
  (distinct_lines : Function.Injective lines)
  (no_parallel : ∀ i j, i ≠ j → ¬Parallel (lines i) (lines j)) :
  ∃ (C : ℝ → ℝ → Prop), 
    (|{ p : ℝ × ℝ | C p.1 p.2 ∧ p ∈ blue_points lines blue_indices}| = 2 * n - 1) ∧ 
    (|{ p : ℝ×ℝ | C p.1 p.2 ∧ p ∈ red_points lines red_indices}| = 2 * n - 1) :=
by
  -- Proof goes here
  sorry

end exists_circle_intersect_B_R_l116_116639


namespace cos_alpha_value_l116_116143

open Real

theorem cos_alpha_value (α : ℝ) : 
  (sin (α - (π / 3)) = 1 / 5) ∧ (0 < α) ∧ (α < π / 2) → 
  (cos α = (2 * sqrt 6 - sqrt 3) / 10) := 
by
  intros h
  sorry

end cos_alpha_value_l116_116143


namespace total_chapters_l116_116420

theorem total_chapters (chapters_read : ℕ) (time_read : ℕ) (remaining_time : ℕ) (total_time : ℕ)
  (reading_rate : chapters_read = 2 ∧ time_read = 3 ∧ remaining_time = 9 ∧ total_time = 8) :
  chapters_read + (2 * remaining_time / time_read) = total_time :=
by 
  cases reading_rate
  repeat {rwa chapters_read}
  repeat {rwa time_read}
  repeat {rwa remaining_time}
  repeat {rwa total_time}
  sorry

end total_chapters_l116_116420


namespace opposite_of_neg_2023_l116_116943

theorem opposite_of_neg_2023 : ∃ x : ℤ, -2023 + x = 0 ∧ x = 2023 := 
by 
  use 2023
  constructor
  · exact rfl
  · exact rfl

end opposite_of_neg_2023_l116_116943


namespace opposite_neg_2023_l116_116771

theorem opposite_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_neg_2023_l116_116771


namespace opposite_of_neg_2023_l116_116817

/-- Define the opposite of a number -/
def opposite (x : ℤ) : ℤ := -x

/-- Prove the opposite of -2023 is 2023 -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116817


namespace opposite_of_neg_2023_l116_116955

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l116_116955


namespace ratio_height_to_width_equals_4_l116_116337

variable {w : ℝ} {h : ℝ} {l : ℝ} {V : ℝ}
variable {k : ℝ}

-- Conditions
axiom height_is_multiple_of_width : h = k * w
axiom length_is_triple_of_height : l = 3 * h
axiom volume_is_10368 : V = 10368
axiom width_is_approximately_6 : w ≈ 6

-- Theorem: The ratio of the height to the width of the wall is 4:1
theorem ratio_height_to_width_equals_4 : k = 4 :=
by
  -- proof steps would go here, but we can use sorry to indicate the proof is omitted
  sorry

end ratio_height_to_width_equals_4_l116_116337


namespace function_zero_solution_l116_116497

-- Define the statement of the problem
theorem function_zero_solution (f : ℝ → ℝ) :
  (∀ x : ℝ, x ≠ 0 → ∀ y : ℝ, f (x ^ 2 + y) ≥ (1 / x + 1) * f y) →
  (∀ x : ℝ, f x = 0) :=
by
  -- The proof of this theorem will be inserted here.
  sorry

end function_zero_solution_l116_116497


namespace product_max_min_two_digit_numbers_l116_116481

theorem product_max_min_two_digit_numbers {a b : ℕ} (h₁ : a = 99) (h₂ : b = 10) : a * b = 990 :=
by
  rw [h₁, h₂]
  rfl

end product_max_min_two_digit_numbers_l116_116481


namespace circumcircle_radius_ABC_l116_116371

noncomputable def radius_of_circumcircle_of_ABC
  (R1 R2 : ℝ)
  (h_sum : R1 + R2 = 11)
  (d : ℝ := 5 * Real.sqrt 17)
  (h_dist : 5 * Real.sqrt 17 = d)
  (A : ℝ := 8)
  (h_tangent : ∀ {R3 : ℝ}, R3 = A → R3 = 8)
  : ℝ :=
2 * Real.sqrt 19

theorem circumcircle_radius_ABC (R1 R2 d A : ℝ)
  (h_sum : R1 + R2 = 11)
  (h_dist : d = 5 * Real.sqrt 17)
  (h_tangent : A = 8)
  : radius_of_circumcircle_of_ABC R1 R2 h_sum d h_dist A h_tangent = 2 * Real.sqrt 19 := by
  sorry

end circumcircle_radius_ABC_l116_116371


namespace problem_inversely_directly_proportional_l116_116738

theorem problem_inversely_directly_proportional (x y k c z : ℝ) (z_constant : ∀ t₁ t₂, z t₁ = z t₂) (h1 : x * y = k) (h2 : y = c * z)
  (hx : x' = 1.20 * x) :
  ∃ y₁ y₂, (y₁ = c * z) ∧ (y₂ = c * z) ∧ (z_constant y₁ y₂) ∧ (-100 * (y₂ - y₁) / y₁ = 16.67) := by
  sorry

end problem_inversely_directly_proportional_l116_116738


namespace tangent_line_parallel_to_given_line_l116_116591

theorem tangent_line_parallel_to_given_line 
  (x : ℝ) (y : ℝ) (tangent_line : ℝ → ℝ) :
  (tangent_line y = x^2 - 1) → 
  (tangent_line = 4) → 
  (4 * x - y - 5 = 0) :=
by 
  sorry

end tangent_line_parallel_to_given_line_l116_116591


namespace money_spent_on_mower_blades_l116_116681

variable (money_made money_left money_spent price_per_game games_bought : ℕ)

-- Conditions
axiom h1 : money_made = 69
axiom h2 : games_bought = 9
axiom h3 : price_per_game = 5
axiom h4 : money_left = games_bought * price_per_game
axiom h5 : money_spent = money_made - money_left

theorem money_spent_on_mower_blades :
  money_spent = 24 :=
by
  simp [h1, h2, h3, h4, h5]
  sorry

end money_spent_on_mower_blades_l116_116681


namespace Smiths_Backery_Pies_l116_116301

theorem Smiths_Backery_Pies : 
  ∀ (p : ℕ), (∃ (q : ℕ), q = 16 ∧ p = 4 * q + 6) → p = 70 :=
by
  intros p h
  cases' h with q hq
  cases' hq with hq1 hq2
  rw [hq1] at hq2
  have h_eq : p = 4 * 16 + 6 := hq2
  norm_num at h_eq
  exact h_eq

end Smiths_Backery_Pies_l116_116301


namespace modulus_of_z_l116_116340

def z : ℂ := 2 / (1 - I)

theorem modulus_of_z : complex.abs z = real.sqrt 2 := 
by
  sorry

end modulus_of_z_l116_116340


namespace pascal_remaining_distance_l116_116691

variable (v : ℕ) (v_red : ℕ) (v_inc : ℕ) (t : ℕ) (d : ℕ)

def remaining_distance_proof : Prop :=
  v = 8 ∧
  v_red = 4 ∧
  v_inc = 12 ∧
  d = v * t ∧ 
  d = v_red * (t + 16) ∧ 
  d = v_inc * (t - 16) ∧ 
  d = 256

theorem pascal_remaining_distance (v : ℕ) (v_red : ℕ) (v_inc : ℕ) (t : ℕ) (d : ℕ) 
  (h1 : v = 8)
  (h2 : v_red = 4)
  (h3 : v_inc = 12)
  (h4 : d = v * t)
  (h5 : d = v_red * (t + 16))
  (h6 : d = v_inc * (t - 16)) : 
  d = 256 :=
by {
  -- Initial setup
  have h7 : d = 4 * (t + 16) := h5,
  have h8 : d = 12 * (t - 16) := h6,
  have h9 : 4 * (t + 16) = 12 * (t - 16), from sorry,
  -- Solve the equation 
  have h10 : 4 * t + 64 = 12 * t - 192, from sorry,
  have h11 : 4 * t + 256 = 12 * t, from sorry,
  have h12 : 8 * t = 256, from sorry,
  have ht : t = 32, from sorry,
  -- Compute d
  have hd : d = 8 * t, from h4,
  rw ht at hd,
  exact hd.symm,
}

end pascal_remaining_distance_l116_116691


namespace opposite_of_neg_2023_l116_116868

-- Definition of the problem
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  -- The proof is to be provided
  sorry

end opposite_of_neg_2023_l116_116868


namespace opposite_of_neg_2023_l116_116845

def opposite (n : ℤ) : ℤ :=
  -n

theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116845


namespace pineapples_sold_l116_116357

/-- 
There were initially 86 pineapples in the store. After selling some pineapples,
9 of the remaining pineapples were rotten and were discarded. Given that there 
are 29 fresh pineapples left, prove that the number of pineapples sold is 48.
-/
theorem pineapples_sold (initial_pineapples : ℕ) (rotten_pineapples : ℕ) (remaining_fresh_pineapples : ℕ)
  (h_init : initial_pineapples = 86)
  (h_rotten : rotten_pineapples = 9)
  (h_fresh : remaining_fresh_pineapples = 29) :
  initial_pineapples - (remaining_fresh_pineapples + rotten_pineapples) = 48 :=
sorry

end pineapples_sold_l116_116357


namespace opposite_of_neg_2023_l116_116933

theorem opposite_of_neg_2023 : ∃ x : ℤ, -2023 + x = 0 ∧ x = 2023 := 
by 
  use 2023
  constructor
  · exact rfl
  · exact rfl

end opposite_of_neg_2023_l116_116933


namespace midpoint_on_y_axis_ellipse_l116_116031

/-- Let the ellipse be given by the equation x²/12 + y²/3 = 1. 
If the midpoint M of the line segment PF₁ (where F₁ is the left focus of the ellipse) lies on the y-axis,
then the y-coordinate of the point P is ±√(3) / 2. --/
theorem midpoint_on_y_axis_ellipse (x y: ℝ) 
  (h₁: (x^2) / 12 + (y^2) / 3 = 1)
  (h₂: x ≠ 0) 
  (h₃: y ≠ 0)
  (h₄: (2*x) = 0):
  y = ± (√3 / 2) :=
sorry

end midpoint_on_y_axis_ellipse_l116_116031


namespace pentagon_angle_l116_116207

noncomputable def angle_CDG (ABCDE : Type) [regular_pentagon ABCDE]
  (F G : ABCDE) (A B C D E: ABCDE)
  (on_AB : F ∈ segment A B) (on_BC : G ∈ segment B C) 
  (FG_eq_GD : dist F G = dist G D) (angle_FDE_60 : ∠ F D E = 60) : Prop :=
  ∠ C D G = 6

theorem pentagon_angle
  (ABCDE : Type) [regular_pentagon ABCDE]
  (F G : ABCDE) (A B C D E: ABCDE)
  (on_AB : F ∈ segment A B) (on_BC : G ∈ segment B C) 
  (FG_eq_GD : dist F G = dist G D) (angle_FDE_60 : ∠ F D E = 60) :
  angle_CDG ABCDE F G A B C D E on_AB on_BC FG_eq_GD angle_FDE_60 :=
begin
  -- Proof would be here
  sorry
end

end pentagon_angle_l116_116207


namespace opposite_of_neg_2023_l116_116858

-- Definition of the problem
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  -- The proof is to be provided
  sorry

end opposite_of_neg_2023_l116_116858


namespace opposite_neg_2023_l116_116764

theorem opposite_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_neg_2023_l116_116764


namespace find_XY_squared_l116_116644

open EuclideanGeometry
open Real

-- Define the setup for the problem
variable (A B C O T X Y : Point)
variable (ω : Circline)
variable [h₁ : IsScalene B C A]
variable [h₂ : IsAcute A B C]
variable [h₃ : OnCircumcircle ω A B C O]
variable [h₄ : TangentPoints ω B C T]
variable (BT_eq_CT : dist B T = 20)
variable (BC_eq_30 : dist B C = 30)
variable (X_is_proj : IsProjection T X (line A B))
variable (Y_is_proj : IsProjection T Y (line A C))
variable (sum_eq : dist T X ^ 2 + dist T Y ^ 2 + dist X Y ^ 2 = 2020)

-- The final theorem statement
theorem find_XY_squared : dist X Y ^ 2 = XY_squared :=
by
  sorry

end find_XY_squared_l116_116644


namespace problem200_squared_minus_399_composite_l116_116228

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  ¬ is_prime n

theorem problem200_squared_minus_399_composite : is_composite (200^2 - 399) :=
sorry

end problem200_squared_minus_399_composite_l116_116228


namespace part1_part2_l116_116094

noncomputable def arithmetic_seq := ℕ → ℤ 
noncomputable def geom_seq := ℕ → (ℕ → ℤ)

variables {a : arithmetic_seq}
variables {c : geom_seq}

-- Conditions
axiom h1 : a 7 = 9
axiom h2 : ∑ i in finset.range 7, a (i + 1) = 42

-- Prove a_{15} = 17 and S_{20} = 250
theorem part1 : a 15 = 17 ∧ (∑ i in finset.range 20, a (i + 1)) = 250 :=
by
  sorry

-- Conditions for sequence ∃c_n = 2^n * a_n
axiom h3 : ∀ n, c n = λ i, ((2^i) * a i)

-- Prove T_n = (n+1) * 2^(n+1) - 2
theorem part2 (n : ℕ) : (∑ i in finset.range n, c n (i+1)) = (n+1) * 2^(n+1) - 2 :=
by
  sorry

end part1_part2_l116_116094


namespace blithe_initial_toys_l116_116018

-- Define the conditions as given in the problem
def lost_toys : ℤ := 6
def found_toys : ℤ := 9
def final_toys : ℤ := 43

-- Define the problem statement to prove the initial number of toys
theorem blithe_initial_toys (T : ℤ) (h : T - lost_toys + found_toys = final_toys) : T = 40 :=
sorry

end blithe_initial_toys_l116_116018


namespace right_triangle_area_integer_l116_116291

theorem right_triangle_area_integer (a b c : ℤ) (h : a * a + b * b = c * c) : ∃ (n : ℤ), (1 / 2 : ℚ) * a * b = ↑n := 
sorry

end right_triangle_area_integer_l116_116291


namespace ep_eq_eq_l116_116263

/-- Let Ω₁ and Ω₂ be two circles intersecting at points M and N. Let Δ be the common tangent to the two circles, closer to M than to N.
    Δ is tangent to Ω₁ at A and to Ω₂ at B. The line passing through M and parallel to Δ intersects Ω₁ at C and Ω₂ at D. 
    Let E be the intersection of lines CA and BD, P the intersection point of lines AN and CD, and Q the intersection of lines BN and CD. 
    Show that EP = EQ. --/
theorem ep_eq_eq
  (Ω₁ Ω₂ : Type) [Circle Ω₁] [Circle Ω₂]
  (M N A B C D E P Q : Type)
  (h1 : M ∈ Ω₁ ∧ M ∈ Ω₂)
  (h2 : N ∈ Ω₁ ∧ N ∈ Ω₂)
  (h3 : M ≠ N)
  (h4 : Tangent Ω₁ Omega₂ at A)
  (h5 : Tangent Ω₂ Omega₁ at B)
  (h6 : LineThrough M ∥ TangentLine Δ)
  (h7 : intersects Ω₁ C)
  (h8 : intersects Ω₂ D)
  (h9 : intersects (LineThrough CA) E)
  (h10 : intersects (LineThrough BD) E)
  (h11 : intersects (LineThrough AN) P)
  (h12 : intersects (LineThrough CD) P)
  (h13 : intersects (LineThrough BN) Q)
  (h14 : intersects (LineThrough CD) Q) 
: EP = EQ := 
sorry

end ep_eq_eq_l116_116263


namespace Tim_kittens_initial_count_l116_116362

theorem Tim_kittens_initial_count:
  (given_to_jessica given_to_sara now_left initially_had : ℕ)
  (h_jessica : given_to_jessica = 3)
  (h_sara : given_to_sara = 6)
  (h_left : now_left = 9)
  (h_given_away : given_to_jessica + given_to_sara = 9)
  (h_total : initially_had = given_to_jessica + given_to_sara + now_left):
  initially_had = 18 := 
by
  sorry

end Tim_kittens_initial_count_l116_116362


namespace complement_intersection_example_l116_116570

open Set

variable (U A B : Set ℕ)

def C_U (A : Set ℕ) (U : Set ℕ) : Set ℕ := U \ A

theorem complement_intersection_example 
  (hU : U = {0, 1, 2, 3})
  (hA : A = {0, 1})
  (hB : B = {1, 2, 3}) :
  (C_U A U) ∩ B = {2, 3} :=
by
  sorry

end complement_intersection_example_l116_116570


namespace remaining_distance_proof_l116_116706

-- Define the conditions
def pascal_current_speed : ℝ := 8
def pascal_reduced_speed : ℝ := pascal_current_speed - 4
def pascal_increased_speed : ℝ := pascal_current_speed * 1.5

-- Define the remaining distance in terms of the current speed and time taken
noncomputable def remaining_distance (T : ℝ) : ℝ := pascal_current_speed * T

-- Define the times with the increased and reduced speeds
noncomputable def time_with_increased_speed (T : ℝ) : ℝ := T - 16
noncomputable def time_with_reduced_speed (T : ℝ) : ℝ := T + 16

-- Define the distances using increased and reduced speeds
noncomputable def distance_increased_speed (T : ℝ) : ℝ := pascal_increased_speed * (time_with_increased_speed T)
noncomputable def distance_reduced_speed (T : ℝ) : ℝ := pascal_reduced_speed * (time_with_reduced_speed T)

-- Main theorem stating that the remaining distance is 256 miles
theorem remaining_distance_proof (T : ℝ) (ht_eq: pascal_current_speed * T = 256) : 
  remaining_distance T = 256 := by
  sorry

end remaining_distance_proof_l116_116706


namespace trader_profit_l116_116446

theorem trader_profit (P : ℝ) (hP : P > 0) : 
  let first_discount := 0.80 * P,
      second_discount := 0.72 * P,
      first_increase := 1.08 * P,
      final_price := 1.35 * P,
      profit := final_price - P in
  (profit / P) * 100 = 35 := 
by sorry

end trader_profit_l116_116446


namespace inscribed_sphere_radius_l116_116532

noncomputable def radius_inscribed_sphere (S1 S2 S3 S4 V : ℝ) : ℝ :=
  3 * V / (S1 + S2 + S3 + S4)

theorem inscribed_sphere_radius (S1 S2 S3 S4 V R : ℝ) :
  R = radius_inscribed_sphere S1 S2 S3 S4 V :=
by
  sorry

end inscribed_sphere_radius_l116_116532


namespace largest_A_is_correct_l116_116124

open Set

noncomputable def largest_possible_A_size (m n : ℕ) (h1 : 2 ≤ m) (h2 : 3 ≤ n) : ℕ := 2 * m + n - 2

theorem largest_A_is_correct (m n : ℕ) (h1 : 2 ≤ m) (h2 : 3 ≤ n) (A : Set (ℕ × ℕ)) (hA : A ⊆ {p | 1 ≤ p.1 ∧ p.1 ≤ m ∧ 1 ≤ p.2 ∧ p.2 ≤ n})
  (hCond : ¬∃ x1 x2 y1 y2 y3, x1 < x2 ∧ y1 < y2 ∧ y2 < y3 ∧ (x1, y1) ∈ A ∧ (x1, y2) ∈ A ∧ (x1, y3) ∈ A ∧ (x2, y2) ∈ A) :
  A.card ≤ largest_possible_A_size m n h1 h2 :=
sorry

end largest_A_is_correct_l116_116124


namespace total_time_spent_l116_116271

variables (x y z w : ℝ)

theorem total_time_spent : 
  let active_time := x + y * x + z * (y * x) + w * (z * (y * x)) in
  let writing_email_time := (1/2) * active_time in
  let full_time := active_time + writing_email_time in
  full_time = (3/2) * x * (1 + y + z * y + w * z * y) := 
sorry

end total_time_spent_l116_116271


namespace product_of_divisors_of_30_l116_116512

open Nat

def divisors_of_30 : List ℕ := [1, 2, 3, 5, 6, 10, 15, 30]

theorem product_of_divisors_of_30 :
  (divisors_of_30.foldr (· * ·) 1) = 810000 := by
  sorry

end product_of_divisors_of_30_l116_116512


namespace opposite_of_neg_2023_l116_116899

-- Let x be the opposite of -2023
def is_opposite (x : ℤ) (n : ℤ) : Prop := n + x = 0

theorem opposite_of_neg_2023 : ∃ x : ℤ, is_opposite x (-2023) ∧ x = 2023 :=
by
  exists 2023
  unfold is_opposite
  sorry

end opposite_of_neg_2023_l116_116899


namespace num_seven_digit_palindromes_l116_116066

theorem num_seven_digit_palindromes : 
  let a_choices := 9
  let b_choices := 10
  let c_choices := 10
  let d_choices := 10
  a_choices * b_choices * c_choices * d_choices = 9000 :=
by
  sorry

end num_seven_digit_palindromes_l116_116066


namespace tan_X_correct_l116_116227

section
variable {X Y Z : Type} -- Define variables for generic points

-- Define the conditions from the problem
variable (angleY : ∠ Y = 90)
variable (YZ : ℝ) (XZ : ℝ)

-- Specific values given in the problem
variable (h1 : YZ = 4) (h2 : XZ = Real.sqrt 41)

-- Define the resulting tangent function
noncomputable def tan_X (X Y Z : Type) (angleY : ∠ Y = 90) (YZ XZ : ℝ) :=
  YZ / (Real.sqrt (XZ ^ 2 - YZ ^ 2))

-- The proof problem
theorem tan_X_correct : tan_X X Y Z angleY 4 (Real.sqrt 41) = 4 / 5 :=
by
  rw [tan_X]
  rw [h1, h2]
  -- Here we would carry out the calculation indicated in the solution.
  sorry
end

end tan_X_correct_l116_116227


namespace cyclic_quadrilateral_iff_eq_dist_l116_116602

variables {A B C D P : Type} [Geometry ℝ A B C D P]

-- Given conditions
def convex_quadrilateral (ABCD : Quadrilateral A B C D) : Prop :=
convex ABCD

def not_angle_bisector_BD (BD : LineSegment (B, D) (D, B)) : Prop :=
¬ (angleBisector BD (∠ ABC) ∧ angleBisector BD (∠ CDA))

variables (P : Point) (interior_P : interior P ABCD : Prop)

def angle_conditions (P B C D A : Point) : Prop :=
∠ PBC = ∠ DBA ∧ ∠ PDC = ∠ BDA

-- The theorem to prove
theorem cyclic_quadrilateral_iff_eq_dist {A B C D P : Point}
  (h1 : convex_quadrilateral ABCD) 
  (h2 : not_angle_bisector_BD BD)
  (h3 : interior P ABCD)
  (h4 : angle_conditions P B C D A) : 
  (is_cyclic_quadrilateral ABCD ↔ dist A P = dist C P) := sorry

end cyclic_quadrilateral_iff_eq_dist_l116_116602


namespace opposite_of_neg_2023_l116_116857

-- Definition of the problem
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  -- The proof is to be provided
  sorry

end opposite_of_neg_2023_l116_116857


namespace angle_sum_B_D_l116_116423

/--
Given a quadrilateral \(ABCD\) with a circle passing through \(A\) and \(C\), intersecting the sides \(AB\), \(BC\), \(CD\), and \(AD\) at \(M\), \(N\), \(P\), and \(Q\) respectively. 
Also given that \(BM = BN = DP = DQ = R\), where \(R\) is the radius.
Prove that the sum of the angles \(B\) and \(D\) of the quadrilateral is \(120^\circ\).
-/
theorem angle_sum_B_D (ABCD : Type) (A B C D M N P Q : ABCD) (R : ℝ) 
  (cir : Circle ABCD) (intersect: Intersect cir A C) 
  (BM BN DP DQ : ℝ) 
  (hB: BM = R) (hN: BN = R) (hD: DP = R) (hQ: DQ = R) :
  ∠B + ∠D = 120 :=
  sorry

end angle_sum_B_D_l116_116423


namespace number_of_tiles_forming_equilateral_triangle_l116_116326

-- Define that we have tiles made up of smaller equilateral triangles, with specific values of k.
def tiles : List ℕ := [1, 2, 3, 12]

-- Define the predicate to check if 3k is a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Define the condition under which placing three identical tiles forms an equilateral triangle
def forms_equilateral_triangle (k : ℕ) : Prop :=
  is_perfect_square (3 * k)

-- State the theorem to be proven
theorem number_of_tiles_forming_equilateral_triangle : 
  (tiles.filter forms_equilateral_triangle).length = 2 := 
sorry

end number_of_tiles_forming_equilateral_triangle_l116_116326


namespace opposite_of_negative_2023_l116_116781

theorem opposite_of_negative_2023 : (- -2023) = 2023 := 
by {
sorry
}

end opposite_of_negative_2023_l116_116781


namespace opposite_of_neg_2023_l116_116848

def opposite (n : ℤ) : ℤ :=
  -n

theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116848


namespace opposite_of_neg_2023_l116_116803

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by sorry

end opposite_of_neg_2023_l116_116803


namespace average_k_of_polynomial_l116_116109

noncomputable def roots := {r : ℕ × ℕ | r.fst * r.snd = 18}
noncomputable def k_values := { k : ℕ | ∃ r ∈ roots, k = r.fst + r.snd }
noncomputable def average_k := ∑ k in k_values.to_finset, k / k_values.to_finset.card

theorem average_k_of_polynomial : average_k = 13 := 
by sorry

end average_k_of_polynomial_l116_116109


namespace remaining_savings_after_purchase_l116_116673

-- Definitions of the conditions
def cost_per_sweater : ℕ := 30
def num_sweaters : ℕ := 6
def cost_per_scarf : ℕ := 20
def num_scarves : ℕ := 6
def initial_savings : ℕ := 500

-- Theorem stating the remaining savings
theorem remaining_savings_after_purchase : initial_savings - ((cost_per_sweater * num_sweaters) + (cost_per_scarf * num_scarves)) = 200 :=
by
  -- skipping the proof
  sorry

end remaining_savings_after_purchase_l116_116673


namespace difference_of_x_y_l116_116554

theorem difference_of_x_y :
  ∀ (x y : ℤ), x + y = 10 → x = 14 → x - y = 18 :=
by
  intros x y h1 h2
  sorry

end difference_of_x_y_l116_116554


namespace opposite_of_neg_2023_l116_116923

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by 
  sorry

end opposite_of_neg_2023_l116_116923


namespace probability_of_odd_number_l116_116329

theorem probability_of_odd_number :
  let digits := [1, 4, 6, 9] in
  let is_odd (n : Nat) := n % 2 = 1 in
  let units_choices := [1, 9] in
  probability (random_permutation digits) (λ n, is_odd n.units_digit) = 1 / 2 :=
sorry

end probability_of_odd_number_l116_116329


namespace volleyball_lineup_ways_l116_116002

def num_ways_lineup (team_size : ℕ) (positions : ℕ) : ℕ :=
  if positions ≤ team_size then
    Nat.descFactorial team_size positions
  else
    0

theorem volleyball_lineup_ways :
  num_ways_lineup 10 5 = 30240 :=
by
  rfl

end volleyball_lineup_ways_l116_116002


namespace opposite_of_neg_2023_l116_116925

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by 
  sorry

end opposite_of_neg_2023_l116_116925


namespace opposite_neg_2023_l116_116760

theorem opposite_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_neg_2023_l116_116760


namespace sum_of_coordinates_l116_116596

theorem sum_of_coordinates (g h : ℕ → ℕ) (g_cond : g 4 = 5) (h_def : ∀ x, h x = (g x) ^ 3) : (4 + h 4) = 129 := 
by 
  have h_4_value : h 4 = 125 := by 
    rw [h_def, g_cond]
    norm_num
  rw [h_4_value]
  norm_num

end sum_of_coordinates_l116_116596


namespace expression_eq_answer_l116_116471

noncomputable def math_expression_eval : ℚ :=
  (2^(Real.log 1/4) - (8/27)^(2/3) + (Real.log10 1/100) + ((Real.sqrt 2 - 1)^(Real.log10 1)))

theorem expression_eq_answer : math_expression_eval = -43/36 := by
  sorry

end expression_eq_answer_l116_116471


namespace common_chord_l116_116502

-- Define the first circle
def Circle1 (x y : ℝ) : Prop :=
  x^2 + y^2 - 25 = 0

-- Define the second circle
def Circle2 (x y : ℝ) : Prop :=
  x^2 + y^2 - 4x + 3y - 10 = 0

-- Statement of the common chord of the two circles
theorem common_chord (x y : ℝ) (h1 : Circle1 x y) (h2 : Circle2 x y) :
  4 * x - 3 * y - 15 = 0 := 
  sorry

end common_chord_l116_116502


namespace max_sqrt_sum_l116_116256

noncomputable def maxValue (a b c : ℝ) : ℝ :=
  sqrt (4 * a + 2) + sqrt (4 * b + 8) + sqrt (4 * c + 10)

theorem max_sqrt_sum (a b c : ℝ) (h1 : a + b + c = 3)
  (h2 : a ≥ -1/2) (h3 : b ≥ -2) (h4 : c ≥ -7/3) :
  maxValue a b c ≤ 4 * sqrt 6 :=
sorry

end max_sqrt_sum_l116_116256


namespace tan_add_pi_over_3_l116_116158

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + (Real.pi / 3)) = -(6 * Real.sqrt 3 + 2) / 13 := 
by
  sorry

end tan_add_pi_over_3_l116_116158


namespace opposite_of_neg_2023_l116_116798

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by sorry

end opposite_of_neg_2023_l116_116798


namespace probability_rolls_more_ones_than_eights_l116_116185

noncomputable def probability_more_ones_than_eights (n : ℕ) := 10246 / 32768

theorem probability_rolls_more_ones_than_eights :
  (probability_more_ones_than_eights 5) = 10246 / 32768 :=
by
  sorry

end probability_rolls_more_ones_than_eights_l116_116185


namespace cot_225_l116_116493

theorem cot_225 (h1 : Real.cot 225 = 1 / Real.tan 225) 
    (h2 : Real.tan 225 = Real.tan (180 + 45)) 
    (h3 : Real.tan 45 = 1) : 
    Real.cot 225 = 1 := 
by
  sorry

end cot_225_l116_116493


namespace line_DE_passes_through_incenter_l116_116726

theorem line_DE_passes_through_incenter 
  {A B C D E D' I : Type*} [Isometric (triangle A B C)]
  (h1 : isosceles_triangle A B C)
  (h2 : on_lateral_side D A B)
  (h3 : on_lateral_side E B C)
  (h4 : ∠ B E D = 3 * ∠ B D E)
  (h5 : reflection D' D AC)
  (h6 : incenter I A B C)
  : collinear D' E I :=
sorry

end line_DE_passes_through_incenter_l116_116726


namespace opposite_of_neg_2023_l116_116956

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l116_116956


namespace opposite_of_neg_2023_l116_116827

/-- Define the opposite of a number -/
def opposite (x : ℤ) : ℤ := -x

/-- Prove the opposite of -2023 is 2023 -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116827


namespace opposite_of_neg_2023_l116_116796

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by sorry

end opposite_of_neg_2023_l116_116796


namespace opposite_of_neg_2023_l116_116973

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l116_116973


namespace probability_nine_heads_l116_116381

theorem probability_nine_heads:
  (∀ (n k : ℕ), k ≤ n → (finset.card (finset.filter (λ (s : finset ℕ), s.card = k) (finset.powerset (finset.range n))) = nat.choose n k)) →
  ∃ p : ℚ, p = 55 / 1024 ∧ 
    calc 
      220 / 4096 = p :
      sorry :=
begin
  let n := 12,
  let k := 9,
  have h1: 2^n = 4096 := by norm_num,
  have h2 : nat.choose n k = 220 := by norm_num,
  have h3 : 220 / 4096 = 55 / 1024 := by norm_num,
  existsi (55 / 1024 : ℚ),
  split,
  { refl, },
  { exact h3 }
end

end probability_nine_heads_l116_116381


namespace proof_problem_l116_116121

variable {R : Type} [Field R]
variable (a b : R)
variable (f : R → R) (df : R → R) (c : R)
variable (h_tangent : ∀ x, f x = a * x^2 + b)
variable (h_derivative : ∀ x, df x = (deriv (λ x, a * x^2 + b)) x)
variable (point : R × R) (slope : R)

theorem proof_problem (h_tangent : f = λ x, a * x^2 + b)
    (h_tangent_slope : df 1 = 2)
    (point_def : point = (1, 3))
    (tangent_condition : f 1 = 3)
    (deriv_correct : df = λ x, 2 * a * x) :
  b / a = 2 := 
  sorry  -- Proof is omitted.

end proof_problem_l116_116121


namespace sqrt_multiplication_l116_116021

theorem sqrt_multiplication (p : ℝ) : 
  (sqrt (15 * p^3) * sqrt (25 * p^2) * sqrt (2 * p^5) = 25 * p^5 * sqrt 6) := 
by
  sorry

end sqrt_multiplication_l116_116021


namespace opposite_of_neg_2023_l116_116930

theorem opposite_of_neg_2023 : ∃ x : ℤ, -2023 + x = 0 ∧ x = 2023 := 
by 
  use 2023
  constructor
  · exact rfl
  · exact rfl

end opposite_of_neg_2023_l116_116930


namespace unique_T_n_l116_116516

variable (a d : ℝ)
variable (n : ℕ)
variable (S : ℕ → ℝ)
variable (T : ℕ → ℝ)

-- Conditions from the problem
def arithmetic_sequence_S (n : ℕ) : ℝ := (2 * a + (n - 1) * d) * n / 2
def T_n (n : ℕ) : ℝ := ∑ k in Finset.range (n + 1), arithmetic_sequence_S a d k

-- Given S_{2023} uniquely determines T_n for some n
axiom S_2023 : S 2023 = 2023 * (a + 1011 * d)

theorem unique_T_n : T (3034 : ℕ) = T_n a d 3034 :=
by
  sorry

end unique_T_n_l116_116516


namespace students_scoring_above_90_l116_116203

theorem students_scoring_above_90
    (total_students : ℕ)
    (mean : ℝ)
    (variance : ℝ)
    (students_80_to_90 : ℕ)
    (scores_distribution : ∀ x : ℝ, ℙ((X x) = normalPdf mean variance))
    (total_students_eq : total_students = 48)
    (mean_eq : mean = 80)
    (students_80_to_90_eq : students_80_to_90 = 16) :
    (students_scoring_above_90 : ℕ) :=
    students_scoring_above_90 = 8 :=
begin
    sorry
end

end students_scoring_above_90_l116_116203


namespace second_dog_average_miles_l116_116015

theorem second_dog_average_miles
  (total_miles_week : ℕ)
  (first_dog_miles_day : ℕ)
  (days_in_week : ℕ)
  (h1 : total_miles_week = 70)
  (h2 : first_dog_miles_day = 2)
  (h3 : days_in_week = 7) :
  (total_miles_week - (first_dog_miles_day * days_in_week)) / days_in_week = 8 := sorry

end second_dog_average_miles_l116_116015


namespace range_of_k_l116_116188

theorem range_of_k (k : ℝ) :
  (∃ (x : ℝ), 2 < x ∧ x < 3 ∧ x^2 + (1 - k) * x - 2 * (k + 1) = 0) →
  1 < k ∧ k < 2 :=
by
  sorry

end range_of_k_l116_116188


namespace tetrahedron_pairs_of_edges_determine_plane_l116_116583

theorem tetrahedron_pairs_of_edges_determine_plane :
  let num_edges := 6
  in finset.card (finset.powerset_len 2 (finset.range num_edges)) = 15 :=
by
  let num_edges := 6
  have h : finset.card (finset.powerset_len 2 (finset.range num_edges)) = 15 := by sorry
  exact h

end tetrahedron_pairs_of_edges_determine_plane_l116_116583


namespace find_BD_l116_116199

theorem find_BD (A B C D : Type) (AB AC BC CD : ℝ) (H_AC : AC = 8) (H_BC : BC = 8) (H_AB : AB = 2) (H_CD : CD = 10) (H_D : ∃ (BD : ℝ), BD = sqrt 37 - 1) :
  ∃ BD, BD = sqrt 37 - 1 :=
by
  use sqrt 37 - 1
  exact H_D

end find_BD_l116_116199


namespace find_x3_l116_116369

variable (x1 x2 : ℝ) (hx1 : 0 < x1) (hx2 : x1 < x2)

def y1 : ℝ := Real.log x1
def y2 : ℝ := Real.log x2

def y_c : ℝ := (2 / 3) * y1 + (1 / 3) * y2

theorem find_x3 (hx1 : x1 = 2) (hx2 : x2 = 500) 
: ∃ x3 : ℝ, x3 = 10 * 2^(2/3) * 5^(1/3) := 
sorry

end find_x3_l116_116369


namespace max_value_of_g_l116_116376

def g (x : ℝ) : ℝ := 3 * Real.sin x + 4 * Real.cos x

theorem max_value_of_g :
  ∃ x, g x = 5 ∧ Real.tan x = 3 / 4 :=
sorry

end max_value_of_g_l116_116376


namespace cornbread_pieces_count_l116_116238

def cornbread_pieces (pan_length pan_width piece_length piece_width : ℕ) : ℕ := 
  (pan_length * pan_width) / (piece_length * piece_width)

theorem cornbread_pieces_count :
  cornbread_pieces 24 20 3 3 = 53 :=
by
  -- The definitions and the equivalence transformation tell us that this is true
  sorry

end cornbread_pieces_count_l116_116238


namespace triangle_angle_contradiction_l116_116289

-- Define the condition: all internal angles of the triangle are less than 60 degrees.
def condition (α β γ : ℝ) (h: α + β + γ = 180): Prop :=
  α < 60 ∧ β < 60 ∧ γ < 60

-- The proof statement
theorem triangle_angle_contradiction (α β γ : ℝ) (h_sum : α + β + γ = 180) (h: condition α β γ h_sum) : false :=
sorry

end triangle_angle_contradiction_l116_116289


namespace opposite_of_neg_2023_l116_116971

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l116_116971


namespace factorization_of_polynomial_l116_116492

theorem factorization_of_polynomial :
  ∀ x : ℝ, (x^4 - 4*x^3 + 6*x^2 - 4*x + 1) = (x - 1)^4 :=
by
  intro x
  sorry

end factorization_of_polynomial_l116_116492


namespace probability_plane_intersects_interior_of_cube_l116_116359

def cube_vertices : ℕ := 8
def combinations (n k : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k)))

theorem probability_plane_intersects_interior_of_cube :
  let total_ways_to_choose_vertices := combinations cube_vertices 3,
      ways_to_choose_vertices_on_one_face := combinations 4 3,
      total_ways_to_choose_vertices_on_faces := 6 * ways_to_choose_vertices_on_one_face in
  (1 - (total_ways_to_choose_vertices_on_faces / total_ways_to_choose_vertices : ℚ)) = (4 / 7 : ℚ) :=
by
  sorry

end probability_plane_intersects_interior_of_cube_l116_116359


namespace common_ratio_of_arithmetic_seq_l116_116534

theorem common_ratio_of_arithmetic_seq (a_1 q : ℝ) 
  (h1 : a_1 + a_1 * q^2 = 10) 
  (h2 : a_1 * q^3 + a_1 * q^5 = 5 / 4) : 
  q = 1 / 2 := 
by 
  sorry

end common_ratio_of_arithmetic_seq_l116_116534


namespace problem1_problem2_problem3_l116_116267

-- Given conditions as Lean definitions
variable (f : ℝ → ℝ) (f' : ℝ → ℝ) [Differentiable ℝ f]
variable [Inc : ∀ x y, x < y → f'(x) ≤ f'(y)]
variable [Pos : ∀ x, 0 < f'(x)]
variable (p : ℝ) (g : ℝ → ℝ) (h_tangent : ∀ x, g(x) = f'(p) * (x - p) + f(p))

-- Problem 1: Prove f(x) ≥ g(x) and equality holds iff x = p
theorem problem1 : ∀ x, f(x) ≥ g(x) ∧ (f(x) = g(x) ↔ x = p) :=
sorry

-- Given additional condition for Problem 2
variable (a x₀ : ℝ) (hx : g(a) = f(x₀))

-- Problem 2: Prove x₀ ≤ a
theorem problem2 : x₀ ≤ a :=
sorry

-- Given conditions for Problem 3
variable (m : ℝ)

-- Assumption: ∀ x > -m, e^x > ln(x + m)
axiom exp_ln_ineq (x : ℝ) (h : x > -m) : exp x > log (x + m)

-- Problem 3: Prove m < 5/2
theorem problem3 : m < 5 / 2 :=
sorry

end problem1_problem2_problem3_l116_116267


namespace opposite_of_neg_2023_l116_116843

def opposite (n : ℤ) : ℤ :=
  -n

theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116843


namespace opposite_of_neg_2023_l116_116961

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l116_116961


namespace domain_of_f_equals_l116_116039

noncomputable def domain_of_function := {x : ℝ | x > -1 ∧ -(x+4) * (x-1) > 0}

theorem domain_of_f_equals : domain_of_function = { x : ℝ | -1 < x ∧ x < 1 } :=
by
  sorry

end domain_of_f_equals_l116_116039


namespace proposition_C_valid_l116_116458

-- Definitions and assumptions
variables (m n : Line) (α : Plane)

-- Conditions
def line_in_plane (l : Line) (p : Plane) : Prop := l ⊆ p
def line_parallel_plane (l : Line) (p : Plane) : Prop := ∀ p₁ p₂ ∈ p, l ∩ p = ∅
def coplanar (l₁ l₂ : Line) : Prop := ∃ p : Plane, l₁ ⊆ p ∧ l₂ ⊆ p
def parallel (l₁ l₂ : Line) : Prop := ∀ p₁ p₂ ∈ l₁, ∀ q₁ q₂ ∈ l₂, p₂ - p₁ = q₂ - q₁

-- Proof problem statement
theorem proposition_C_valid :
  (line_in_plane m α) →
  (line_parallel_plane n α) →
  (coplanar m n) →
  (parallel m n) :=
by
  sorry

end proposition_C_valid_l116_116458


namespace variance_invariant_under_translation_l116_116049

variable {α : Type*} [TopologicalSpace α] [MetricSpace α]

-- Define original and new data sets
variable (original_data new_data : List ℝ)

-- Condition: Each number in original_data is subtracted by 80 to obtain new_data
def data_transformation (original_data : List ℝ) (new_data : List ℝ) : Prop :=
  ∀ x ∈ original_data, ∃ y ∈ new_data, y = x - 80

-- The average of the new data is 1.2 
def average_of_new_data (new_data : List ℝ) : Prop :=
  (new_data.sum / new_data.length.toReal) = 1.2

-- The variance of the new data is 4.4
def variance_of_new_data (new_data : List ℝ) : Prop :=
  let mean := new_data.sum / new_data.length.toReal in
  (new_data.map (λ x, (x - mean)^2)).sum / new_data.length.toReal = 4.4

-- The variance of the original data should be proven to be 4.4
theorem variance_invariant_under_translation 
  (original_data new_data : List ℝ)
  (h1 : data_transformation original_data new_data)
  (h2 : average_of_new_data new_data)
  (h3 : variance_of_new_data new_data) :
  let mean := original_data.sum / original_data.length.toReal in
  (original_data.map (λ x, (x - mean)^2)).sum / original_data.length.toReal = 4.4 := by
  sorry

end variance_invariant_under_translation_l116_116049


namespace sufficient_but_not_necessary_condition_l116_116184

variable (x : ℝ)

theorem sufficient_but_not_necessary_condition :
  (∀ x : ℝ, |2*x - 1| ≤ x → x^2 + x - 2 ≤ 0) ∧ 
  ¬(∀ x : ℝ, x^2 + x - 2 ≤ 0 → |2 * x - 1| ≤ x) := sorry

end sufficient_but_not_necessary_condition_l116_116184


namespace opposite_of_neg_2023_l116_116849

def opposite (n : ℤ) : ℤ :=
  -n

theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116849


namespace sequence_eventually_one_l116_116038

noncomputable def sequence (p q : ℕ) : ℕ → ℕ 
| 1 := p
| 2 := q
| (n + 3) := if ∃ m : ℕ, (sequence (n + 1)) + (sequence (n + 2)) = 2^m 
             then 1 
             else smallest_odd_prime_factor ((sequence (n + 1)) + (sequence (n + 2)))
             
theorem sequence_eventually_one (p q : ℕ) (hp : prime p) (hq : prime q) (h : p < q) : 
  ∃ M : ℕ, ∀ n > M, sequence p q n = 1 := 
sorry

end sequence_eventually_one_l116_116038


namespace monomial_properties_l116_116747

-- Define the monomial
def monomial := -(3 : ℚ) / 5 * a^2 * b * c

-- Define the coefficient
def coefficient := -(3 : ℚ) / 5

-- Define the degree calculation for the given monomial's variables
def degree := 2 + 1 + 1

-- Theorem stating the coefficient and the degree of the monomial
theorem monomial_properties : 
  (coefficient, degree) = (-(3 : ℚ) / 5, 4) := 
  by
    -- Proof is to be provided here
    sorry

end monomial_properties_l116_116747


namespace tan_angle_addition_l116_116145

theorem tan_angle_addition (x : Real) (h1 : Real.tan x = 3) (h2 : Real.tan (Real.pi / 3) = Real.sqrt 3) : 
  Real.tan (x + Real.pi / 3) = (3 + Real.sqrt 3) / (1 - 3 * Real.sqrt 3) :=
by 
  sorry

end tan_angle_addition_l116_116145


namespace incenter_coincides_with_point_N_l116_116209

theorem incenter_coincides_with_point_N
  {A B C N : Point}
  (hBAC : Angle BAC)
  (h_circle : InscribedCircle (segment A B C) (angle_bisector BAC) N):
  IsIncenter N (triangle ABC) :=
by
  sorry

end incenter_coincides_with_point_N_l116_116209


namespace area_ratio_equilateral_triangl_l116_116664

theorem area_ratio_equilateral_triangl (x : ℝ) :
  let sA : ℝ := x 
  let sB : ℝ := 3 * sA
  let sC : ℝ := 5 * sA
  let sD : ℝ := 4 * sA
  let area_ABC := (Real.sqrt 3 / 4) * (sA ^ 2)
  let s := (sB + sC + sD) / 2
  let area_A'B'C' := Real.sqrt (s * (s - sB) * (s - sC) * (s - sD))
  (area_A'B'C' / area_ABC) = 8 * Real.sqrt 3 := by
  sorry

end area_ratio_equilateral_triangl_l116_116664


namespace angle_P_in_quadrilateral_l116_116609

theorem angle_P_in_quadrilateral : 
  ∀ (P Q R S : ℝ), (P = 3 * Q) → (P = 4 * R) → (P = 6 * S) → (P + Q + R + S = 360) → P = 206 := 
by
  intros P Q R S hP1 hP2 hP3 hSum
  sorry

end angle_P_in_quadrilateral_l116_116609


namespace arithmetic_sequence_difference_l116_116272

-- Given the sequences and their ranges
def first_sequence : List ℕ := List.range' 2001 100
def second_sequence : List ℕ := List.range' 301 100

-- Assertion to prove the difference of the sums of these sequences equals 170000
theorem arithmetic_sequence_difference :
  first_sequence.sum - second_sequence.sum = 170000 := by
  sorry

end arithmetic_sequence_difference_l116_116272


namespace find_ordered_triple_l116_116259

noncomputable def solution_triple : ℝ × ℝ × ℝ :=
  let (x, y, z) := (13:ℝ, 11:ℝ, 6:ℝ)
  in (x, y, z)

theorem find_ordered_triple (x y z : ℝ) (h1 : x > 2) (h2 : y > 2) (h3 : z > 2)
  (h4 : (x + 3)^2 / (y + z - 3) + (y + 5)^2 / (z + x - 5) + (z + 7)^2 / (x + y - 7) = 45) :
  (x, y, z) = solution_triple :=
by
  sorry

end find_ordered_triple_l116_116259


namespace construction_doors_needed_l116_116426

theorem construction_doors_needed :
  (let number_of_buildings := 2 in
   let floors_per_building := 12 in
   let apartments_per_floor := 6 in
   let doors_per_apartment := 7 in
   let apartments_per_building := floors_per_building * apartments_per_floor in
   let total_apartments := apartments_per_building * number_of_buildings in
   let total_doors := total_apartments * doors_per_apartment in
   total_doors = 1008) :=
begin
  sorry
end

end construction_doors_needed_l116_116426


namespace number_of_distinct_colorings_l116_116220

def disks := Fin 8

inductive Color
| Blue
| Red
| Green

def coloring := disks → Color

def is_symmetry (c : coloring): Prop :=
  ∃ f : disks → disks, ∃ g: coloring, (∀ x, g (f x) = c x ∧
    ( (equiv.rotate 8).symm f = f ∨ (equiv.reflect.disk 8).symm f = f))

theorem number_of_distinct_colorings : 
  {p : ∀ c : coloring, is_symmetry c, (finset.image c p).size} = 38 := 
sorry

end number_of_distinct_colorings_l116_116220


namespace problem_statement_l116_116662

noncomputable def t (x : ℝ) : ℝ := real.sqrt (5 * x + 2)
noncomputable def f (x : ℝ) : ℝ := 7 - t x

theorem problem_statement : t (f 3) = real.sqrt (37 - 5 * real.sqrt 17) :=
by
  -- statement to guide the proof, if needed.
  sorry

end problem_statement_l116_116662


namespace largest_store_visitation_l116_116353

theorem largest_store_visitation (stores total_visitors total_shoppers : ℕ) 
  (visits_two visits_three remaining_people remaining_visit : ℕ) 
  (h_stores : stores = 12)
  (h_total_visitors : total_visitors = 36)
  (h_total_shoppers : total_shoppers = 18) 
  (h_visits_two : visits_two = 10 * 2)
  (h_visits_three : visits_three = 5 * 3)
  (h_remaining_people : remaining_people = total_shoppers - 15) 
  (h_remaining_visit : remaining_visit = total_visitors - (visits_two + visits_three))
  (h_cond_remaining_people : remaining_people = 3)
  (h_cond_remaining_visit : remaining_visit = 1)
  : ∃ (max_stores : ℕ), max_stores = 1 :=
by
  use 1
  sorry

end largest_store_visitation_l116_116353


namespace tan_add_pi_over_3_l116_116174

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = - (6 + 5 * Real.sqrt 3) / 13 :=
sorry

end tan_add_pi_over_3_l116_116174


namespace opposite_of_neg_2023_l116_116804

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by sorry

end opposite_of_neg_2023_l116_116804


namespace five_digit_palindromes_count_l116_116418

def is_palindrome (n : ℕ) : Prop :=
  let s := n.repr
  s = s.reverse

def is_five_digit (n : ℕ) : Prop := 
  10000 ≤ n ∧ n < 100000

noncomputable def count_five_digit_palindromes : ℕ :=
  (list.range 100000).filter (λ n, is_five_digit n ∧ is_palindrome n).length

theorem five_digit_palindromes_count : count_five_digit_palindromes = 900 :=
sorry

end five_digit_palindromes_count_l116_116418


namespace probability_prime_multiple_of_11_l116_116688

theorem probability_prime_multiple_of_11 (cards : Finset ℕ) (h_range : ∀ n ∈ cards, 1 ≤ n ∧ n ≤ 100)
    (h_card : cards.card = 100) :
    let prime11 := 11 in
    let primes := {n ∈ cards | Nat.Prime n} in
    let multiples_of_11 := {n ∈ cards | 11 ∣ n} in
    (primes ∩ multiples_of_11).card.toReal / cards.card.toReal = 1 / 100 :=
by
  sorry

end probability_prime_multiple_of_11_l116_116688


namespace opposite_of_neg_2023_l116_116921

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by 
  sorry

end opposite_of_neg_2023_l116_116921


namespace sum_even_coeffs_eq_l116_116244

theorem sum_even_coeffs_eq {n : ℕ} :
  let g (x : ℝ) := (1 - x + x^2) ^ n,
      b_ks := (λ k, g x.coeff k) in
  (b_ks 0 + b_ks 2 + b_ks 4 + ... + b_ks (2*n)) = (3^n + 1) / 2 :=
by sorry

end sum_even_coeffs_eq_l116_116244


namespace simplify_expr1_simplify_expr2_simplify_expr3_l116_116298

-- 1. Proving (1)(2x^{2})^{3}-x^{2}·x^{4} = 7x^{6}
theorem simplify_expr1 (x : ℝ) : (1 : ℝ) * (2 * x^2)^3 - x^2 * x^4 = 7 * x^6 := 
by 
  sorry

-- 2. Proving (a+b)^{2}-b(2a+b) = a^{2}
theorem simplify_expr2 (a b : ℝ) : (a + b)^2 - b * (2 * a + b) = a^2 := 
by 
  sorry

-- 3. Proving (x+1)(x-1)-x^{2} = -1
theorem simplify_expr3 (x : ℝ) : (x + 1) * (x - 1) - x^2 = -1 :=
by 
  sorry

end simplify_expr1_simplify_expr2_simplify_expr3_l116_116298


namespace average_is_75x_of_sequence_plus_x_l116_116744

theorem average_is_75x_of_sequence_plus_x (x : ℚ) :
  let S := (150 * 151) / 2 ∧
  let n := 151 ∧
  (S + x) / n = 75 * x →
  x = 11325 / 11324 :=
by
  intro S n hAvg
  have hS : S = 11325 := sorry 
  have hn : n = 151 := sorry
  rw [hS, hn] at hAvg
  sorry

end average_is_75x_of_sequence_plus_x_l116_116744


namespace minimum_value_of_f_l116_116757

def f (x : ℝ) : ℝ := (x^2 - 2*x + 6) / (x + 1)

theorem minimum_value_of_f : ∀ x : ℝ, x > -1 → (∀ y : ℝ, y = f x ↔ y ≥ 2 ∧ (y = 2 → x = 2)) :=
by 
  sorry

end minimum_value_of_f_l116_116757


namespace alice_bob_meet_after_five_turns_l116_116457

theorem alice_bob_meet_after_five_turns :
  ∃ k : ℕ, k = 5 ∧ 
  ∀ (circumference : ℕ) (alice_move : ℕ) (bob_move : ℕ) (start : ℕ), 
  circumference = 15 → alice_move = 7 → bob_move = 11 → start = 15 →
  ((alice_move * k - ((circumference - bob_move) * k)) % circumference = 0) := 
by {
  let k := 5,
  use k,
  split,
  { refl },
  { intros circumference alice_move bob_move start h1 h2 h3 h4,
    have h5: (alice_move * k - ((circumference - bob_move) * k)) % circumference = 0,
    { sorry },
    exact h5
  }
}

end alice_bob_meet_after_five_turns_l116_116457


namespace opposite_of_neg_2023_l116_116908

-- Let x be the opposite of -2023
def is_opposite (x : ℤ) (n : ℤ) : Prop := n + x = 0

theorem opposite_of_neg_2023 : ∃ x : ℤ, is_opposite x (-2023) ∧ x = 2023 :=
by
  exists 2023
  unfold is_opposite
  sorry

end opposite_of_neg_2023_l116_116908


namespace opposite_of_neg_2023_l116_116922

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by 
  sorry

end opposite_of_neg_2023_l116_116922


namespace tan_shifted_value_l116_116170

theorem tan_shifted_value (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = -((6 + 5 * Real.sqrt 3) / 13) :=
by
  sorry

end tan_shifted_value_l116_116170


namespace train_cross_platform_time_l116_116449

/-- A train traveling at 72 kmph crosses a platform in some time and a man standing 
on the platform in 18 seconds. The length of the platform is 300 meters. 
Prove the train takes 33 seconds to cross the platform. -/
theorem train_cross_platform_time {speed_kmph : ℕ} {cross_time_man : ℕ} {platform_length : ℕ}
  (h1 : speed_kmph = 72)
  (h2 : cross_time_man = 18)
  (h3 : platform_length = 300) :
  let speed_mps := (speed_kmph * 1000) / 3600,
      train_length := speed_mps * cross_time_man,
      total_distance := train_length + platform_length in
  total_distance / speed_mps = 33 :=
by
  admit

end train_cross_platform_time_l116_116449


namespace tan_add_pi_div_three_l116_116151

theorem tan_add_pi_div_three (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = -((6 + 5 * Real.sqrt 3) / 13) := 
by 
  sorry

end tan_add_pi_div_three_l116_116151


namespace equation_of_line_passing_through_and_parallel_l116_116752

theorem equation_of_line_passing_through_and_parallel :
  ∀ (x y : ℝ), (x = -3 ∧ y = -1) → (∃ (C : ℝ), x - 2 * y + C = 0) → C = 1 :=
by
  intros x y h₁ h₂
  sorry

end equation_of_line_passing_through_and_parallel_l116_116752


namespace marble_selection_l116_116634

theorem marble_selection : (∃ num_ways : ℕ, num_ways = 990 ∧ (∃ S : finset ℕ, S.card = 5 ∧ 
  (∃ subset_special : finset ℕ, subset_special.card = 2 ∧ subset_special ⊆ {0, 1, 2, 3} ∧ 
  ∃ subset_rest : finset ℕ, subset_rest.card = 3 ∧ subset_rest ⊆ (finset.range 15 \ {0, 1, 2, 3}) ∧ 
  subset_special ∪ subset_rest = S))) :=
sorry

end marble_selection_l116_116634


namespace value_of_2m_plus_3n_l116_116539

theorem value_of_2m_plus_3n (m n : ℝ) (h : (m^2 + 4 * m + 5) * (n^2 - 2 * n + 6) = 5) : 2 * m + 3 * n = -1 :=
by
  sorry

end value_of_2m_plus_3n_l116_116539


namespace number_value_proof_l116_116589

theorem number_value_proof (x y : ℝ) (h1 : 0.5 * x = y + 20) (h2 : x - 2 * y = 40) : x = 40 := 
by
  sorry

end number_value_proof_l116_116589


namespace interest_rate_l116_116443

theorem interest_rate (SI P T R : ℝ) (h1 : SI = 100) (h2 : P = 500) (h3 : T = 4) (h4 : SI = (P * R * T) / 100) :
  R = 5 :=
by
  sorry

end interest_rate_l116_116443


namespace radius_circumcircle_l116_116373

variables (R1 R2 R3 : ℝ)
variables (d : ℝ)
variables (R : ℝ)

noncomputable def sum_radii := R1 + R2 = 11
noncomputable def distance_centers := d = 5 * Real.sqrt 17
noncomputable def radius_third_sphere := R3 = 8
noncomputable def touching := R1 + R2 + 2 * R3 = d

theorem radius_circumcircle :
  R = 5 * Real.sqrt 17 / 2 :=
  by
  -- Use conditions here if necessary
  sorry

end radius_circumcircle_l116_116373


namespace mike_total_payment_l116_116277

-- Definitions for the conditions
def total_car_price : ℝ := 35000
def loan_amount : ℝ := 20000
def interest_rate : ℝ := 0.15
def time : ℝ := 1

-- Definition for the problem statement
theorem mike_total_payment : 
  let interest := loan_amount * interest_rate * time in
  let total_loan_repayment := loan_amount + interest in
  let amount_without_loan := total_car_price - loan_amount in
  total_loan_repayment + amount_without_loan = 38000 :=
by
  sorry

end mike_total_payment_l116_116277


namespace count_seven_digit_palindromes_l116_116069

theorem count_seven_digit_palindromes : 
  let palindromes := { n : ℤ | ∃ (a b c d : ℕ), 
    1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧ 
    n = 1000000 * a + 100000 * b + 10000 * c + 1000 * d + 100 * c + 10 * b + a } in
  ∃ (count : ℕ), count = 9000 ∧ count = palindromes.card :=
by
  -- the proof goes here
  sorry

end count_seven_digit_palindromes_l116_116069


namespace opposite_of_neg_2023_l116_116836

def opposite (n : ℤ) : ℤ :=
  -n

theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116836


namespace Greg_and_Earl_together_l116_116490

-- Conditions
def Earl_initial : ℕ := 90
def Fred_initial : ℕ := 48
def Greg_initial : ℕ := 36

def Earl_to_Fred : ℕ := 28
def Fred_to_Greg : ℕ := 32
def Greg_to_Earl : ℕ := 40

def Earl_final : ℕ := Earl_initial - Earl_to_Fred + Greg_to_Earl
def Fred_final : ℕ := Fred_initial + Earl_to_Fred - Fred_to_Greg
def Greg_final : ℕ := Greg_initial + Fred_to_Greg - Greg_to_Earl

-- Theorem statement
theorem Greg_and_Earl_together : Greg_final + Earl_final = 130 := by
  sorry

end Greg_and_Earl_together_l116_116490


namespace inequality_solution_l116_116316

theorem inequality_solution (x : ℝ) :
  (6*x^2 + 24*x - 63) / ((3*x - 4)*(x + 5)) < 4 ↔ x ∈ Set.Ioo (-(5:ℝ)) (4 / 3) ∪ Set.Iio (5) ∪ Set.Ioi (4 / 3) := by
  sorry

end inequality_solution_l116_116316


namespace savings_after_purchase_l116_116680

theorem savings_after_purchase :
  let price_sweater := 30
  let price_scarf := 20
  let num_sweaters := 6
  let num_scarves := 6
  let savings := 500
  let total_cost := (num_sweaters * price_sweater) + (num_scarves * price_scarf)
  savings - total_cost = 200 :=
by
  sorry

end savings_after_purchase_l116_116680


namespace seahawks_final_score_l116_116358

def num_touchdowns : ℕ := 4
def num_field_goals : ℕ := 3
def points_per_touchdown : ℕ := 7
def points_per_fieldgoal : ℕ := 3

theorem seahawks_final_score : (num_touchdowns * points_per_touchdown) + (num_field_goals * points_per_fieldgoal) = 37 := by
  sorry

end seahawks_final_score_l116_116358


namespace solve_for_x_l116_116588

theorem solve_for_x (x : Real) :
  sin (4 * x) * sin (5 * x) = cos (4 * x) * cos (5 * x) → 
  x = π / 9 :=
sorry

end solve_for_x_l116_116588


namespace inclination_angle_of_line_l116_116755

theorem inclination_angle_of_line :
  ∃ θ ∈ set.Ico 0 180, θ = 120 ∧ 
  ∃ (m : ℝ), m = -√3 ∧ ∀ (x y : ℝ), y = -√3 * x + 3 → θ = Real.arctan m :=
by
  sorry

end inclination_angle_of_line_l116_116755


namespace opposite_of_neg_2023_l116_116851

def opposite (n : ℤ) : ℤ :=
  -n

theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116851


namespace circle_repr_eq_l116_116191

theorem circle_repr_eq (a : ℝ) : (∃ x y : ℝ, x^2 + y^2 - 2*x + 4*y + 1 + a = 0) ↔ a < 4 :=
by
  sorry

end circle_repr_eq_l116_116191


namespace sin_2x_increasing_intervals_l116_116341

theorem sin_2x_increasing_intervals (k : ℤ) :
  ∀ x, (kπ - π / 4 ≤ x ∧ x ≤ kπ + π / 4) ↔ (∃ y, y = sin (2*x)) :=
sorry

end sin_2x_increasing_intervals_l116_116341


namespace opposite_of_neg_2023_l116_116910

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by 
  sorry

end opposite_of_neg_2023_l116_116910


namespace opposite_of_neg_2023_l116_116867

-- Definition of the problem
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  -- The proof is to be provided
  sorry

end opposite_of_neg_2023_l116_116867


namespace marble_selection_l116_116633

theorem marble_selection : (∃ num_ways : ℕ, num_ways = 990 ∧ (∃ S : finset ℕ, S.card = 5 ∧ 
  (∃ subset_special : finset ℕ, subset_special.card = 2 ∧ subset_special ⊆ {0, 1, 2, 3} ∧ 
  ∃ subset_rest : finset ℕ, subset_rest.card = 3 ∧ subset_rest ⊆ (finset.range 15 \ {0, 1, 2, 3}) ∧ 
  subset_special ∪ subset_rest = S))) :=
sorry

end marble_selection_l116_116633


namespace replace_some_expression_with_x_minus_2_l116_116189

theorem replace_some_expression_with_x_minus_2 :
  ∀ (q : ℝ) (x : ℝ), 
    (∃ E : ℝ, q = E^2 + (x + 1)^2 - 6)
    → y_is_least_when_x_is_2 (λ x, (x - 2)^2 + (x + 1)^2 - 6)
    → ∀ E, E = (x - 2) := 
by 
  intros q x h_y_is_min E_condition 
  sorry

end replace_some_expression_with_x_minus_2_l116_116189


namespace find_m_l116_116541

-- Define the vectors a and b
def a : ℝ × ℝ × ℝ := (-3, 2, 5)
def b (m : ℝ) : ℝ × ℝ × ℝ := (1, m, 3)

-- Define a function to compute the dot product of two 3D vectors
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Statement of the problem in Lean 4
theorem find_m (m : ℝ) (h : dot_product a (b m) = 0) : m = -6 :=
sorry

end find_m_l116_116541


namespace opposite_of_neg_2023_l116_116800

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by sorry

end opposite_of_neg_2023_l116_116800


namespace verify_conditions_l116_116398

-- Define the conditions as expressions
def condition_A (a : ℝ) : Prop := 2 * a * 3 * a = 6 * a
def condition_B (a b : ℝ) : Prop := 3 * a^2 * b - 3 * a * b^2 = 0
def condition_C (a : ℝ) : Prop := 6 * a / (2 * a) = 3
def condition_D (a : ℝ) : Prop := (-2 * a) ^ 3 = -6 * a^3

-- Prove which condition is correct
theorem verify_conditions (a b : ℝ) (h : a ≠ 0) : 
  ¬ condition_A a ∧ ¬ condition_B a b ∧ condition_C a ∧ ¬ condition_D a :=
by 
  sorry

end verify_conditions_l116_116398


namespace probability_of_9_heads_in_12_flips_l116_116385

theorem probability_of_9_heads_in_12_flips :
  (∃ n : ℕ, n = 12) →
  (1 / (2 ^ 12) * (∑ k in finset.range (12 + 1), if k = 9 then (nat.choose 12 9) else 0)) = (55 / 1024) :=
by
  sorry

end probability_of_9_heads_in_12_flips_l116_116385


namespace solve_train_bridge_problem_l116_116338

noncomputable def train_bridge_problem : Prop :=
  ∃ (length_of_bridge : ℕ),
    let train_length := 120 in
    let train_speed_km_per_hr := 45 in
    let time_seconds := 30 in
    let train_speed_m_per_s := train_speed_km_per_hr * 1000 / 3600 in
    let total_distance := train_speed_m_per_s * time_seconds in
    length_of_bridge = total_distance - train_length

theorem solve_train_bridge_problem : train_bridge_problem :=
  sorry

end solve_train_bridge_problem_l116_116338


namespace simplify_and_evaluate_l116_116297

theorem simplify_and_evaluate (a : ℕ) (h : a = 2023) : (a + 1) / a / (a - 1 / a) = 1 / 2022 :=
by
  sorry

end simplify_and_evaluate_l116_116297


namespace lights_at_top_layer_l116_116614

theorem lights_at_top_layer (a : ℕ) (h1 : ∑ i in finset.range 7, a * (1 / 2) ^ i = 381) :
  ∀ n, n = 6 → a * (1 / 2) ^ n = 3 :=
by
  intros n hn
  rw hn
  exact sorry

end lights_at_top_layer_l116_116614


namespace find_line_equation_l116_116343

-- Defining the points O and A
structure Point where
  x : ℝ
  y : ℝ

def O : Point := { x := 0, y := 0 }
def A : Point := { x := -4, y := 2 }

-- Defining the concept of symmetry about a line
def isPerpendicularBisector (l : ℝ → ℝ → Prop) (P Q : Point) : Prop :=
  let midpoint := { x := (P.x + Q.x) / 2, y := (P.y + Q.y) / 2 }
  ∃ m b, l = fun x y => y = m*x + b ∧ (Q.y - P.y) / (Q.x - P.x) * m = -1

-- The line l in standard form ax + by + c = 0 with a, b, c as given
def line_eq (a b c : ℝ) : ℝ → ℝ → Prop :=
  fun x y => a * x + b * y + c = 0

-- Stating the theorem to be proved
theorem find_line_equation : 
  ∃ l : ℝ → ℝ → Prop, isPerpendicularBisector l O A ∧ l = line_eq 2 (-1) 5 :=
by
  sorry

end find_line_equation_l116_116343


namespace sum_of_coefficients_l116_116514

theorem sum_of_coefficients (a : Fin 10 → ℤ) (x : ℤ) :
  (2 * x - 3) ^ 9 = ∑ i, a i * (x - 1) ^ i →
  ∑ i in Finset.range 9.succ \ {0}, a i = 2 :=
by
  intro h
  sorry

end sum_of_coefficients_l116_116514


namespace second_dog_miles_per_day_l116_116016

-- Definitions describing conditions
section DogWalk
variable (total_miles_week : ℕ)
variable (first_dog_miles_day : ℕ)
variable (days_in_week : ℕ)

-- Assert conditions given in the problem
def condition1 := total_miles_week = 70
def condition2 := first_dog_miles_day = 2
def condition3 := days_in_week = 7

-- The theorem to prove
theorem second_dog_miles_per_day
  (h1 : condition1 total_miles_week)
  (h2 : condition2 first_dog_miles_day)
  (h3 : condition3 days_in_week) :
  (total_miles_week - days_in_week * first_dog_miles_day) / days_in_week = 8 :=
sorry
end DogWalk

end second_dog_miles_per_day_l116_116016


namespace paul_account_balance_l116_116721

variable (initial_balance : ℝ) (transfer1 : ℝ) (transfer2 : ℝ) (service_charge_rate : ℝ)

def final_balance (init_bal transfer1 transfer2 rate : ℝ) : ℝ :=
  let charge1 := transfer1 * rate
  let total_deduction := transfer1 + charge1
  init_bal - total_deduction

theorem paul_account_balance :
  initial_balance = 400 →
  transfer1 = 90 →
  transfer2 = 60 →
  service_charge_rate = 0.02 →
  final_balance 400 90 60 0.02 = 308.2 :=
by
  intros h1 h2 h3 h4
  rw [final_balance, h1, h2, h4]
  norm_num

end paul_account_balance_l116_116721


namespace num_seven_digit_palindromes_l116_116063

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def is_seven_digit_palindrome (x : ℕ) : Prop :=
  let a := x / 1000000 % 10 in
  let b := x / 100000 % 10 in
  let c := x / 10000 % 10 in
  let d := x / 1000 % 10 in
  let e := x / 100 % 10 in
  let f := x / 10 % 10 in
  let g := x % 10 in
  a ≠ 0 ∧ a = g ∧ b = f ∧ c = e

theorem num_seven_digit_palindromes : 
  ∃ n : ℕ, (∀ x : ℕ, is_seven_digit_palindrome x → 1 ≤ a (x / 1000000 % 10) ∧ is_digit (b) ∧ is_digit (c) ∧ is_digit (d)) → n = 9000 :=
sorry

end num_seven_digit_palindromes_l116_116063


namespace opposite_of_neg_2023_l116_116811

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by sorry

end opposite_of_neg_2023_l116_116811


namespace remaining_savings_after_purchase_l116_116674

-- Definitions of the conditions
def cost_per_sweater : ℕ := 30
def num_sweaters : ℕ := 6
def cost_per_scarf : ℕ := 20
def num_scarves : ℕ := 6
def initial_savings : ℕ := 500

-- Theorem stating the remaining savings
theorem remaining_savings_after_purchase : initial_savings - ((cost_per_sweater * num_sweaters) + (cost_per_scarf * num_scarves)) = 200 :=
by
  -- skipping the proof
  sorry

end remaining_savings_after_purchase_l116_116674


namespace Pascal_remaining_distance_l116_116700

theorem Pascal_remaining_distance (D T : ℕ) :
  let current_speed := 8
  let reduced_speed := current_speed - 4
  let increased_speed := current_speed + current_speed / 2
  (D = current_speed * T) →
  (D = reduced_speed * (T + 16)) →
  (D = increased_speed * (T - 16)) →
  D = 256 :=
by
  intros
  sorry

end Pascal_remaining_distance_l116_116700


namespace triangles_equal_in_area_l116_116623

noncomputable theory
open_locale classical

-- Definitions
variables {A B C G M N O : Type} 
[inhabited A] [inhabited B] [inhabited C] [inhabited G] [inhabited M] [inhabited N] [inhabited O]

-- Assume triangle ABC with centroid G and M, N, O are midpoints of sides BC, CA, AB respectively
def triangle (A B C : Type) : Prop := true -- Abstract definition of a triangle
def is_centroid (G : Type) (A B C : Type) : Prop := true -- G is centroid of triangle ABC
def is_midpoint (M : Type) (B C : Type) : Prop := true -- M is midpoint of BC
def is_midpoint (N : Type) (C A : Type) : Prop := true -- N is midpoint of CA
def is_midpoint (O : Type) (A B : Type) : Prop := true -- O is midpoint of AB

-- Statement to prove
theorem triangles_equal_in_area :
  triangle A B C →
  is_centroid G A B C →
  is_midpoint M B C →
  is_midpoint N C A →
  is_midpoint O A B →
  (area (triangle A G B) = area (triangle B G C)) ∧
  (area (triangle B G C) = area (triangle C G A)) ∧
  (area (triangle A G M) = area (triangle B G N)) ∧
  (area (triangle B G N) = area (triangle C G O)) :=
begin
  sorry
end

end triangles_equal_in_area_l116_116623


namespace symmetric_angles_y_axis_l116_116196

theorem symmetric_angles_y_axis (α β : ℝ) (k : ℤ)
  (h : ∃ k : ℤ, β = 2 * k * π + (π - α)) :
  α + β = (2 * k + 1) * π ∨ α = -β + (2 * k + 1) * π :=
by sorry

end symmetric_angles_y_axis_l116_116196


namespace opposite_of_neg_2023_l116_116909

-- Let x be the opposite of -2023
def is_opposite (x : ℤ) (n : ℤ) : Prop := n + x = 0

theorem opposite_of_neg_2023 : ∃ x : ℤ, is_opposite x (-2023) ∧ x = 2023 :=
by
  exists 2023
  unfold is_opposite
  sorry

end opposite_of_neg_2023_l116_116909


namespace value_of_a_pow_2023_plus_b_pow_2023_l116_116181

theorem value_of_a_pow_2023_plus_b_pow_2023 (a b : ℤ) (h : {a^2, 0, -1} = {a, b, 0}) : a^2023 + b^2023 = 0 :=
sorry

end value_of_a_pow_2023_plus_b_pow_2023_l116_116181


namespace probability_of_9_heads_in_12_flips_l116_116380

open BigOperators

-- We state the problem in terms of probability theory.
theorem probability_of_9_heads_in_12_flips :
  let p : ℚ := (nat.choose 12 9) / (2 ^ 12)
  p = 55 / 1024 :=
by
  sorry

end probability_of_9_heads_in_12_flips_l116_116380


namespace tan_add_pi_div_three_l116_116152

theorem tan_add_pi_div_three (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = -((6 + 5 * Real.sqrt 3) / 13) := 
by 
  sorry

end tan_add_pi_div_three_l116_116152


namespace opposite_of_neg_2023_l116_116835

def opposite (n : ℤ) : ℤ :=
  -n

theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l116_116835


namespace opposite_of_neg_2023_l116_116954

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l116_116954


namespace meet_probability_zero_l116_116279

/-- Define the setup conditions for objects A and B. -/
def movement_condition (A B : ℕ × ℕ) : Prop :=
  (A = (0, 0) ∧ B = (6, 8) ∧ 
  (∀ n, A.1 + A.2 = 4 → B.1 + B.2 = 4) ∧ 
  (A.1 = 6 - B.1) ∧ (A.2 = 8 - B.2))

/-- Define the function to calculate the probability. -/
def meeting_probability (steps : ℕ) : ℚ :=
  if ∃ (A B : ℕ × ℕ), movement_condition A B then 1 / (2 ^ (2 * steps)) else 0

/-- The probability that A and B meet after moving exactly four steps is zero. -/
theorem meet_probability_zero : meeting_probability 4 = 0 :=
sorry

end meet_probability_zero_l116_116279
