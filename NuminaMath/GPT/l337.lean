import Mathlib

namespace no_real_roots_sqrt_eqn_l337_337605

noncomputable def has_no_real_roots (f : ℝ → ℝ) : Prop :=
  ¬ ∃ x : ℝ, f x = 0

theorem no_real_roots_sqrt_eqn :
  has_no_real_roots (λ x : ℝ, sqrt (x + 7) - sqrt (x - 5) + 2) :=
sorry

end no_real_roots_sqrt_eqn_l337_337605


namespace sasha_min_places_l337_337625

theorem sasha_min_places 
  (hemisphere : ℝ³ → Prop) 
  (not_on_boundary : ∀ (x : ℝ³), ¬ boundary x → hemisphere x) 
  (celebrations : set ℝ³) 
  (h_celebrations : finite celebrations) 
  (h_on_sphere : ∀ (x : ℝ³), x ∈ celebrations → sphere x) 
  (h_no_boundary : ∀ (x : ℝ³), x ∈ celebrations → ¬ boundary x) : 
  ∃ (n : ℕ), n ≥ 4 ∧ ∀ (H : ℝ³ → Prop), hemisphere H → ∃ (x : ℝ³), x ∈ celebrations ∧ ¬ H x := 
sorry

end sasha_min_places_l337_337625


namespace sum_natural_numbers_perfect_square_l337_337422

theorem sum_natural_numbers_perfect_square (K : ℕ) (N : ℕ) (h : N < 50) :
  ∃ (K : ℕ), (1 + 2 + 3 + ... + K = N^2) ↔ (K = 1 ∨ K = 8 ∨ K = 49) :=
by
  sorry

end sum_natural_numbers_perfect_square_l337_337422


namespace exists_bound_for_expression_l337_337141

theorem exists_bound_for_expression :
  ∃ (C : ℝ), (∀ (k : ℤ), abs ((k^8 - 2*k + 1 : ℤ) / (k^4 - 3 : ℤ)) < C) := 
sorry

end exists_bound_for_expression_l337_337141


namespace area_of_triangles_eq_area_of_quadrilateral_l337_337543

noncomputable def area {α : Type*} [linear_ordered_field α] (A B C : Point α) : α := sorry

variables {α : Type*} [linear_ordered_field α]
variables (A B C D K L E F : Point α)
variables (is_trapezoid : trapezoid A B C D)
variables (K_on_AB : lies_on_segment K A B)
variables (L_on_CD : lies_on_segment L C D)
variables (E_on_AL_DK : intersect_point E (segment A L) (segment D K))
variables (F_on_BL_CK : intersect_point F (segment B L) (segment C K))

theorem area_of_triangles_eq_area_of_quadrilateral :
  area A D E + area B C F = area E K F L :=
sorry

end area_of_triangles_eq_area_of_quadrilateral_l337_337543


namespace total_cost_of_path_l337_337649

variable (length_field : ℝ) (width_field : ℝ)
variable (width_path_longer : ℝ) (width_path_shorter1 : ℝ) (width_path_shorter2 : ℝ)
variable (cost_longer : ℝ) (cost_shorter1 : ℝ) (cost_shorter2 : ℝ)

def area_path_longer_sides (length_field width_path_longer : ℝ) : ℝ :=
  2 * (length_field * width_path_longer)

def area_path_shorter_side1 (width_field width_path_shorter1 : ℝ) : ℝ :=
  width_field * width_path_shorter1

def area_path_shorter_side2 (width_field width_path_shorter2 : ℝ) : ℝ :=
  width_field * width_path_shorter2

def cost_path_longer (length_field width_path_longer cost_longer : ℝ) : ℝ :=
  area_path_longer_sides length_field width_path_longer * cost_longer

def cost_path_shorter1 (width_field width_path_shorter1 cost_shorter1 : ℝ) : ℝ :=
  area_path_shorter_side1 width_field width_path_shorter1 * cost_shorter1

def cost_path_shorter2 (width_field width_path_shorter2 cost_shorter2 : ℝ) : ℝ :=
  area_path_shorter_side2 width_field width_path_shorter2 * cost_shorter2

theorem total_cost_of_path :
  let length_field := 75
  let width_field := 55
  let width_path_longer := 2.5
  let width_path_shorter1 := 3
  let width_path_shorter2 := 4
  let cost_longer := 7
  let cost_shorter1 := 9
  let cost_shorter2 := 12 in
  cost_path_longer length_field width_path_longer cost_longer +
    cost_path_shorter1 width_field width_path_shorter1 cost_shorter1 +
    cost_path_shorter2 width_field width_path_shorter2 cost_shorter2 = 6750 :=
by
  sorry

end total_cost_of_path_l337_337649


namespace length_of_square_side_l337_337308

noncomputable def speed_km_per_hr_to_m_per_s (speed : ℝ) : ℝ :=
  speed * 1000 / 3600

noncomputable def perimeter_of_square (side_length : ℝ) : ℝ :=
  4 * side_length

theorem length_of_square_side
  (time_seconds : ℝ)
  (speed_km_per_hr : ℝ)
  (distance_m : ℝ)
  (side_length : ℝ)
  (h1 : time_seconds = 72)
  (h2 : speed_km_per_hr = 10)
  (h3 : distance_m = speed_km_per_hr_to_m_per_s speed_km_per_hr * time_seconds)
  (h4 : distance_m = perimeter_of_square side_length) :
  side_length = 50 :=
sorry

end length_of_square_side_l337_337308


namespace find_f_expression_l337_337570

-- Definition of the given function g(x)
def g (x : ℝ) (hx : 0 < x) : ℝ := log x / log 2

-- The main theorem stating the problem
theorem find_f_expression (f : ℝ → ℝ) 
    (h : ∀ x > 0, f (-x) = - g x (by linarith)) :
    ∀ x < 0, f x = - log (-x) / log 2 :=
sorry

end find_f_expression_l337_337570


namespace min_value_of_function_l337_337749

theorem min_value_of_function (x : ℝ) (h1 : -3 < x) (h2 : x < 0) : 
  ∃ (c : ℝ), c = -9 / 2 ∧ ∀ y : ℝ, y = x * real.sqrt (9 - x^2) → y ≥ c :=
sorry

end min_value_of_function_l337_337749


namespace total_visible_surface_area_l337_337004

-- Define the cubes by their volumes
def volumes : List ℝ := [1, 8, 27, 125, 216, 343, 512, 729]

-- Define the arrangement information as specified
def arrangement_conditions : Prop :=
  ∃ (s8 s7 s6 s5 s4 s3 s2 s1 : ℝ),
    s8^3 = 729 ∧ s7^3 = 512 ∧ s6^3 = 343 ∧ s5^3 = 216 ∧
    s4^3 = 125 ∧ s3^3 = 27 ∧ s2^3 = 8 ∧ s1^3 = 1 ∧
    5 * s8^2 + (5 * s7^2 + 4 * s6^2 + 4 * s5^2) + 
    (5 * s4^2 + 4 * s3^2 + 5 * s2^2 + 4 * s1^2) = 1250

-- The proof statement
theorem total_visible_surface_area : arrangement_conditions → 1250 = 1250 := by
  intro _ -- this stands for not proving the condition, taking it as assumption
  exact rfl


end total_visible_surface_area_l337_337004


namespace circle_problems_l337_337474

noncomputable def polar_equation_of_circle (ρ θ : ℝ) : Prop :=
  ∃ (r : ℝ), θ = 2 * Real.cos θ

noncomputable def length_of_chord_cut_by_line (C : ℝ × ℝ) (r d : ℝ) : Prop :=
  2 * Real.sqrt (r^2 - d^2) = 1

theorem circle_problems {ρ θ : ℝ} :
  (point_P_polar_coordinate ρ θ) ∧ (center_of_circle_is_intersection_point (ρ  θ)) →
  (polar_equation_of_circle ρ θ) ∧ (length_of_chord_cut_by_line (1, 0) 1 (√3 / 2)) :=
by
  sorry

end circle_problems_l337_337474


namespace number_of_complex_numbers_l337_337380

-- Definitions: defining the conditions
def complex_has_magnitude_two (z : ℂ) : Prop :=
  complex.abs z = 2

def complex_expression_abs_eq_two (z : ℂ) : Prop :=
  abs ((z^2 / (conj z)^2) + ((conj z)^2 / z^2)) = 2

-- The theorem: there are exactly 8 such complex numbers
theorem number_of_complex_numbers (S : set ℂ) :
  (∀ z : ℂ, z ∈ S ↔ complex_has_magnitude_two z ∧ complex_expression_abs_eq_two z) →
  S.card = 8 :=
by
  sorry

end number_of_complex_numbers_l337_337380


namespace valentine_day_spending_l337_337533

structure DogTreatsConfig where
  heart_biscuits_count_A : Nat
  puppy_boots_count_A : Nat
  small_toy_count_A : Nat
  heart_biscuits_count_B : Nat
  puppy_boots_count_B : Nat
  large_toy_count_B : Nat
  heart_biscuit_price : Nat
  puppy_boots_price : Nat
  small_toy_price : Nat
  large_toy_price : Nat
  heart_biscuits_discount : Float
  large_toy_discount : Float

def treats_config : DogTreatsConfig :=
  { heart_biscuits_count_A := 5
    puppy_boots_count_A := 1
    small_toy_count_A := 1
    heart_biscuits_count_B := 7
    puppy_boots_count_B := 2
    large_toy_count_B := 1
    heart_biscuit_price := 2
    puppy_boots_price := 15
    small_toy_price := 10
    large_toy_price := 20
    heart_biscuits_discount := 0.20
    large_toy_discount := 0.15 }

def total_discounted_amount_spent (cfg : DogTreatsConfig) : Float :=
  let heart_biscuits_total_cost := (cfg.heart_biscuits_count_A + cfg.heart_biscuits_count_B) * cfg.heart_biscuit_price
  let puppy_boots_total_cost := (cfg.puppy_boots_count_A * cfg.puppy_boots_price) + (cfg.puppy_boots_count_B * cfg.puppy_boots_price)
  let small_toy_total_cost := cfg.small_toy_count_A * cfg.small_toy_price
  let large_toy_total_cost := cfg.large_toy_count_B * cfg.large_toy_price
  let total_cost_without_discount := Float.ofNat (heart_biscuits_total_cost + puppy_boots_total_cost + small_toy_total_cost + large_toy_total_cost)
  let heart_biscuits_discount_amount := cfg.heart_biscuits_discount * Float.ofNat heart_biscuits_total_cost
  let large_toy_discount_amount := cfg.large_toy_discount * Float.ofNat large_toy_total_cost
  let total_discount_amount := heart_biscuits_discount_amount + large_toy_discount_amount
  total_cost_without_discount - total_discount_amount

theorem valentine_day_spending : total_discounted_amount_spent treats_config = 91.20 := by
  sorry

end valentine_day_spending_l337_337533


namespace find_angle_CME_l337_337137

variable (A B C M D E F : Point)
variable (AM BC : Line)
variable (omega : Circle)
variable (angle_BFE angle_ABC : Real)
variable (angle_DEF : Real := angle_ABC)
variable (angle_EAM angle_CME : Real)

-- Conditions
variable (triangle_ABC : Triangle A B C)
variable (is_median_AM : is_median triangle_ABC A M BC)
variable (circle_omega : Circle ω)
variable (passes_through_A : ω.passes_through A)
variable (tangent_BC : ω.tangent BC M)
variable (intersects_AB : ω.intersects AB D)
variable (intersects_AC : ω.intersects AC E)
variable (angle_condition : angle B F E = 72)
variable (angle_EQ: angle D E F = angle_ABC)

-- Proof goal
theorem find_angle_CME : angle CME = 36 := sorry

end find_angle_CME_l337_337137


namespace simplify_and_evaluate_expr_find_ab_l337_337007

theorem simplify_and_evaluate_expr (x y : ℝ) (hx : x = 0.5) (hy : y = -1) :
  (x - 5 * y) * (-x - 5 * y) - (-x + 5 * y)^2 = -5.5 :=
by
  rw [hx, hy]
  sorry

theorem find_ab (a b : ℝ) (h : a^2 - 2 * a + b^2 + 4 * b + 5 = 0) :
  (a + b) ^ 2013 = -1 :=
by
  sorry

end simplify_and_evaluate_expr_find_ab_l337_337007


namespace unique_six_digit_numbers_l337_337057

/-
Prove the number of unique six-digit numbers greater than 300,000 that can be formed
where the digit in the thousand's place is less than 3 is equal to 216.
-/
theorem unique_six_digit_numbers (digits : Finset ℕ) (h_digits : digits = {0, 1, 2, 3, 4, 5}) :
  ∃ (n : ℕ), n = 216 ∧ 
  (∀ d1 d2 d3 d4 d5 d6 : ℕ, d1 * 100000 + d2 * 10000 + d3 * 1000 + d4 * 100 + d5 * 10 + d6 > 300000 ∧ 
  (d1 ∈ digits ∧ d2 ∈ digits ∧ d3 ∈ digits ∧ d4 ∈ digits ∧ d5 ∈ digits ∧ d6 ∈ digits) ∧
  (∃ d1 d2 d3 d4 d5 d6 : ℕ, d1 ∈ {3, 4, 5} ∧ d4 < 3 ∧
    {d1, d2, d3, d4, d5, d6}.card = 6 ∧ 
    {d1, d2, d3, d4, d5, d6} ⊆ digits ↔ {d1, d2, d3, d4, d5, d6} = 216)) := 
sorry

end unique_six_digit_numbers_l337_337057


namespace quadratic_solution_average_l337_337324

-- Definitions of the quadratic equation and conditions.
def quadratic_eq (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0
def has_real_and_distinct_solutions (a b c : ℝ) : Prop := b^2 - 4 * a * c > 0
def average_of_solutions (a b : ℝ) : ℝ := -b / (2 * a)

-- Main statement.
theorem quadratic_solution_average (a b c : ℝ) (h1 : a = 3) (h2 : b = -9) (h3 : c < 6.75) :
  average_of_solutions a b = 3 / 2 :=
by
  sorry

end quadratic_solution_average_l337_337324


namespace cube_root_less_than_5_l337_337091

theorem cube_root_less_than_5 :
  {n : ℕ | n > 0 ∧ (∃ m : ℝ, m^3 = n ∧ m < 5)}.finite.card = 124 :=
by
  sorry

end cube_root_less_than_5_l337_337091


namespace probability_of_same_color_l337_337304

-- Definitions for the conditions
def red_marbles := 6
def white_marbles := 8
def blue_marbles := 9
def total_marbles := red_marbles + white_marbles + blue_marbles

def total_draws := 4

-- Definition capturing the probability calculations
def probability_same_color := 
  ((6 * 5 * 4 / (23 * 22 * 21)) + 
   (8 * 7 * 6 / (23 * 22 * 21)) + 
   (9 * 8 * 7 / (23 * 22 * 21)))

-- Translate the problem statement into a Lean 4 statement
theorem probability_of_same_color:
  probability_same_color = (160 / 1771) := sorry

end probability_of_same_color_l337_337304


namespace ratio_of_saramago_readers_l337_337116

theorem ratio_of_saramago_readers 
  (W : ℕ) (S K B N : ℕ)
  (h1 : W = 42)
  (h2 : K = W / 6)
  (h3 : B = 3)
  (h4 : N = (S - B) - 1)
  (h5 : W = (S - B) + (K - B) + B + N) :
  S / W = 1 / 2 :=
by
  sorry

end ratio_of_saramago_readers_l337_337116


namespace number_of_integer_values_l337_337898

def Q (x : ℤ) : ℤ := x^4 + 4 * x^3 + 9 * x^2 + 2 * x + 17

theorem number_of_integer_values :
  (∃ xs : List ℤ, xs.length = 4 ∧ ∀ x ∈ xs, Nat.Prime (Int.natAbs (Q x))) :=
by
  sorry

end number_of_integer_values_l337_337898


namespace Kyle_is_25_l337_337493

-- Definitions based on the conditions
def Tyson_age : Nat := 20
def Frederick_age : Nat := 2 * Tyson_age
def Julian_age : Nat := Frederick_age - 20
def Kyle_age : Nat := Julian_age + 5

-- The theorem to prove
theorem Kyle_is_25 : Kyle_age = 25 := by
  sorry

end Kyle_is_25_l337_337493


namespace medians_sine_ratio_l337_337210

theorem medians_sine_ratio (A B C S : Point) (k_a k_b k_c : Length) 
(h_median_ratio : ∀ (M : Line), divides S M 2 1) 
(h_area : Δ ABS = Δ ASC = Δ BSC = 1/3 * Δ ABC)
(h_sine : ∀ (M1 M2 : Line), trigonometric_relation M1 M2 S) :
  (k_a : k_b : k_c) = (sin(∠ BSC) : sin(∠ ASC) : sin(∠ ASB)) := 
begin
  sorry
end

end medians_sine_ratio_l337_337210


namespace product_less_than_one_tenth_l337_337548

theorem product_less_than_one_tenth : 
  let prod := (List.range 50).map (λ k, (2 * k + 1 : ℕ) / (2 * (k + 1) : ℕ)).prod in
  prod < 1 / 10 := 
by
  let prod := (List.range 50).map (λ k, (2 * k + 1 : ℕ) / (2 * (k + 1) : ℕ)).prod
  sorry

end product_less_than_one_tenth_l337_337548


namespace perimeter_of_octagon_l337_337188

theorem perimeter_of_octagon :
  let base := 10
  let left_side := 9
  let right_side := 11
  let top_left_diagonal := 6
  let top_right_diagonal := 7
  let small_side1 := 2
  let small_side2 := 3
  let small_side3 := 4
  base + left_side + right_side + top_left_diagonal + top_right_diagonal + small_side1 + small_side2 + small_side3 = 52 :=
by
  -- This automatically assumes all the definitions and shows the equation
  sorry

end perimeter_of_octagon_l337_337188


namespace minimize_max_value_F_l337_337953

noncomputable def F (x A B : ℝ) : ℝ :=
| (Real.cos x)^2 + 2 * (Real.sin x) * (Real.cos x) - (Real.sin x)^2 + A * x + B |

theorem minimize_max_value_F :
  (∀ x ∈ set.Icc 0 (3 * Real.pi / 2), abs ((cos x)^2 + 2 * sin x * cos x - (sin x)^2) ≤ sqrt 2) →
  (∀ A B, (A ≠ 0 ∨ B ≠ 0) → ∃ x ∈ set.Icc 0 (3 * Real.pi / 2), F x A B > sqrt 2) :=
begin
  sorry
end

end minimize_max_value_F_l337_337953


namespace complex_number_quadrant_l337_337411

theorem complex_number_quadrant (a : ℝ) (h : (a^2 - 3 * a - 4 = 0) → (a - 4 ≠ 0)) :
  (a = -1) → (Complex.re (a - Complex.i * a) < 0 ∧ Complex.im (a - Complex.i * a) > 0) :=
by
  intro ha
  rw [ha]
  simp
  exact ⟨by linarith, by linarith⟩

end complex_number_quadrant_l337_337411


namespace pascal_triangle_contains_53_only_once_l337_337799

theorem pascal_triangle_contains_53_only_once (n : ℕ) (k : ℕ) (h_prime : Nat.prime 53) :
  (n = 53 ∧ (k = 1 ∨ k = 52) ∨ 
   ∀ m < 53, Π l, Nat.binomial m l ≠ 53) ∧ 
  (n > 53 → (k = 0 ∨ k = n ∨ Π a b, a * 53 ≠ b * Nat.factorial (n - k + 1))) :=
sorry

end pascal_triangle_contains_53_only_once_l337_337799


namespace polygon_sides_l337_337416

theorem polygon_sides (n : ℕ) (h : n - 3 = 5) : n = 8 :=
by {
  sorry
}

end polygon_sides_l337_337416


namespace pascal_triangle_contains_53_l337_337820

theorem pascal_triangle_contains_53 (n : ℕ) :
  (∃ k, binomial n k = 53) ↔ n = 53 := 
sorry

end pascal_triangle_contains_53_l337_337820


namespace problem1_problem2_l337_337942

-- Definitions based on conditions
def distance : ℝ := 360
def speed_express : ℝ := 72
def speed_slow : ℝ := 48
def early_departure : ℝ := 25 / 60

-- Problem 1
theorem problem1 : ∃ (x : ℝ), (speed_express * x + speed_slow * x = distance ∧ x = 3) :=
by
  -- We prove the theorem with Lean's tactics
  sorry

-- Problem 2
theorem problem2 : ∃ (y : ℝ), (speed_slow * y + speed_express * (y + early_departure) = distance ∧ y = 11 / 4) :=
by
  -- We prove the theorem with Lean's tactics
  sorry

end problem1_problem2_l337_337942


namespace number_of_rows_containing_53_l337_337817

theorem number_of_rows_containing_53 (h_prime_53 : Nat.Prime 53) : 
  ∃! n, (n = 53 ∧ ∃ k, k ≥ 0 ∧ k ≤ n ∧ Nat.choose n k = 53) :=
by 
  sorry

end number_of_rows_containing_53_l337_337817


namespace find_whole_number_l337_337384

theorem find_whole_number (N : ℕ) (h1 : 5.5 < N / 4) (h2 : N / 4 < 6) : N = 23 := by
  -- The proof is omitted
  sorry

end find_whole_number_l337_337384


namespace verify_solution_l337_337462

-- Definition of the condition about Pascal's Triangle rows.
def pascals_triangle_contains_odd (n : ℕ) : ℕ → Prop
| 0 => true
| k => ∀ (m : ℕ) (hm : m ≤ k), ((n.choose m) % 2) = 1

-- Define the proposition we wish to prove.
def problem_statement_as_proof : Prop :=
  (Finset.filter (λ n, pascals_triangle_contains_odd n n)
                 (Finset.range 20)).card = 1

-- Create the main theorem to verify our claim.
theorem verify_solution : problem_statement_as_proof := by
  sorry

end verify_solution_l337_337462


namespace min_value_ineq_l337_337893

theorem min_value_ineq (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h_sum : x + y + z = 9) :
  (∑ c in [{x, y, z]].toFinset.powerset.filter (λ s, s.card = 2), let ⟨a, b⟩ := (s : finset ℝ).val.toSeq in a^2 + b^2) / (∑ c in [{x, y, z]].toFinset.powerset.filter (λ s, s.card = 2), let ⟨a, b⟩ := (s : finset ℝ).val.toSeq in a + b) = 9 :=
  sorry

end min_value_ineq_l337_337893


namespace A_can_finish_remaining_work_in_4_days_l337_337309

theorem A_can_finish_remaining_work_in_4_days
  (A_days : ℕ) (B_days : ℕ) (B_worked_days : ℕ) : 
  A_days = 12 → B_days = 15 → B_worked_days = 10 → 
  (4 * (1 / A_days) = 1 / 3 - B_worked_days * (1 / B_days)) :=
by
  intros hA hB hBwork
  sorry

end A_can_finish_remaining_work_in_4_days_l337_337309


namespace height_not_always_inside_l337_337954

-- Define what an obtuse triangle is
def obtuse_triangle (α β γ : ℝ) : Prop :=
  α + β + γ = 180 ∧ (α > 90 ∨ β > 90 ∨ γ > 90)

-- Main theorem
theorem height_not_always_inside (α β γ : ℝ) (h_obtuse : obtuse_triangle α β γ) :
  ¬∀ (h : ℝ), (h > 0) → height_inside_triangle α β γ h := sorry

end height_not_always_inside_l337_337954


namespace time_to_fill_pool_l337_337526

theorem time_to_fill_pool :
  ∀ (total_volume : ℝ) (filling_rate : ℝ) (leaking_rate : ℝ),
  total_volume = 60 →
  filling_rate = 1.6 →
  leaking_rate = 0.1 →
  (total_volume / (filling_rate - leaking_rate)) = 40 :=
by
  intros total_volume filling_rate leaking_rate hv hf hl
  rw [hv, hf, hl]
  sorry

end time_to_fill_pool_l337_337526


namespace least_integer_value_l337_337985

theorem least_integer_value :
  ∃ x : ℤ, (∀ x' : ℤ, (|3 * x' + 4| <= 18) → (x' >= x)) ∧ (|3 * x + 4| <= 18) ∧ x = -7 := 
sorry

end least_integer_value_l337_337985


namespace rhombus_area_600_l337_337711

noncomputable def area_of_rhombus (x y : ℝ) : ℝ := (x * y) * 2

theorem rhombus_area_600 (x y : ℝ) (qx qy : ℝ)
  (hx : x = 15) (hy : y = 20)
  (hr1 : qx = 15) (hr2 : qy = 20)
  (h_ratio : qy / qx = 4 / 3) :
  area_of_rhombus (2 * (x + y - 2)) (x + y) = 600 :=
by
  rw [hx, hy]
  sorry

end rhombus_area_600_l337_337711


namespace yogurt_calories_per_ounce_l337_337616

variable (calories_strawberries_per_unit : ℕ)
variable (calories_yogurt_total : ℕ)
variable (calories_total : ℕ)
variable (strawberries_count : ℕ)
variable (yogurt_ounces_count : ℕ)

theorem yogurt_calories_per_ounce (h1: strawberries_count = 12)
                                   (h2: yogurt_ounces_count = 6)
                                   (h3: calories_strawberries_per_unit = 4)
                                   (h4: calories_total = 150)
                                   (h5: calories_yogurt_total = calories_total - strawberries_count * calories_strawberries_per_unit):
                                   calories_yogurt_total / yogurt_ounces_count = 17 :=
by
  -- We conjecture that this is correct based on given conditions.
  sorry

end yogurt_calories_per_ounce_l337_337616


namespace overlapping_area_l337_337598

-- Define the basic properties of a 30-60-90 triangle
structure Triangle306090 where
  hypotenuse : ℝ
  short_leg : ℝ
  long_leg : ℝ
  hypotenuse_eq_2_short_leg : hypotenuse = 2 * short_leg
  long_leg_eq_short_leg_sqrt3 : long_leg = short_leg * Real.sqrt 3

-- Define two such triangles with hypotenuse of 10 cm
def triangle1 : Triangle306090 := {
  hypotenuse := 10,
  short_leg := 5,
  long_leg := 5 * Real.sqrt 3,
  hypotenuse_eq_2_short_leg := by linarith,
  long_leg_eq_short_leg_sqrt3 := by linarith
}

def triangle2 : Triangle306090 := {
  hypotenuse := 10,
  short_leg := 5,
  long_leg := 5 * Real.sqrt 3,
  hypotenuse_eq_2_short_leg := by linarith,
  long_leg_eq_short_leg_sqrt3 := by linarith
}

-- Define the configuration condition where the triangles are placed to form an overlapping region
structure OverlappingRegion where
  overlapping_triangle : Triangle306090
  hypotenuse_overlap_half_long_leg : overlapping_triangle.hypotenuse = triangle1.long_leg / 2

-- Assume the overlapping region forms another 30-60-90 triangle
def overlapping_region : OverlappingRegion := {
  overlapping_triangle := {
    hypotenuse := 5 * Real.sqrt 3 / 2,
    short_leg := 5 * Real.sqrt 3 / 4,
    long_leg := (5 * Real.sqrt 3 / 4) * Real.sqrt 3,
    hypotenuse_eq_2_short_leg := by linarith,
    long_leg_eq_short_leg_sqrt3 := by linarith
  },
  hypotenuse_overlap_half_long_leg := by linarith
}

-- Define the proof problem (statement only, no proof required)
theorem overlapping_area : overlapping_region.overlapping_triangle.short_leg * overlapping_region.overlapping_triangle.long_leg / 2 = 75 / 8 := by
  sorry

end overlapping_area_l337_337598


namespace ellipse_properties_l337_337746

theorem ellipse_properties :
  (∃ a b : ℝ, 
    a > b ∧ b > 0 ∧ 
    c = sqrt(a^2 - b^2) ∧ 
    c / a = sqrt(2) / 2 ∧ 
    2 * c * sin(real.pi / 4) = sqrt(2) ∧ 
    4 * a * b = 2 ∧
    (∀ k : ℝ, k < -sqrt(6)/2 ∨ k > sqrt(6)/2 →
      let x1 := ... -- solve for x1 from quadratic equation
      let x2 := ... -- solve for x2 from quadratic equation
      (∃ m : ℝ, 
        m = 1/2 ∧ 
        ((kx + 2 - m) / x1 + (kx + 2 - m) / x2) = 0))
    ∧ Equation_C : ∀ x y : ℝ, x^2 / 2 + y^2 = 1)

end ellipse_properties_l337_337746


namespace find_m_l337_337774

def a := (1 : ℕ, 1 : ℕ)
def b (m : ℕ) := (3, m)
def a_plus_b (m : ℕ) := (a.1 + b(m).1, a.2 + b(m).2)

def parallel (u v : ℕ × ℕ) : Prop :=
  u.1 * v.2 = u.2 * v.1

theorem find_m (m : ℕ) (h : parallel a (a_plus_b m)) : m = 3 :=
by
  sorry

end find_m_l337_337774


namespace fifteenth_entry_is_21_l337_337020

def r_9 (n : ℕ) : ℕ := n % 9

def condition (n : ℕ) : Prop := (7 * n) % 9 ≤ 5

def sequence_elements (k : ℕ) : ℕ := 
  if k = 0 then 0
  else if k = 1 then 2
  else if k = 2 then 3
  else if k = 3 then 4
  else if k = 4 then 7
  else if k = 5 then 8
  else if k = 6 then 9
  else if k = 7 then 11
  else if k = 8 then 12
  else if k = 9 then 13
  else if k = 10 then 16
  else if k = 11 then 17
  else if k = 12 then 18
  else if k = 13 then 20
  else if k = 14 then 21
  else 0 -- for the sake of ensuring completeness

theorem fifteenth_entry_is_21 : sequence_elements 14 = 21 :=
by
  -- Mathematical proof omitted.
  sorry

end fifteenth_entry_is_21_l337_337020


namespace five_points_plane_distance_gt3_five_points_space_not_necessarily_gt3_l337_337406

/-
Problem (a): Given five points on a plane, where the distance between any two points is greater than 2. 
             Prove that there exists a distance between some two of them that is greater than 3.
-/
theorem five_points_plane_distance_gt3 (P : Fin 5 → ℝ × ℝ) 
    (h : ∀ i j : Fin 5, i ≠ j → dist (P i) (P j) > 2) : 
    ∃ i j : Fin 5, i ≠ j ∧ dist (P i) (P j) > 3 :=
sorry

/-
Problem (b): Given five points in space, where the distance between any two points is greater than 2. 
             Prove that it is not necessarily true that there exists a distance between some two of them that is greater than 3.
-/
theorem five_points_space_not_necessarily_gt3 (P : Fin 5 → ℝ × ℝ × ℝ) 
    (h : ∀ i j : Fin 5, i ≠ j → dist (P i) (P j) > 2) : 
    ¬ ∃ i j : Fin 5, i ≠ j ∧ dist (P i) (P j) > 3 :=
sorry

end five_points_plane_distance_gt3_five_points_space_not_necessarily_gt3_l337_337406


namespace pizza_cut_possible_l337_337536
-- Step 1: Import necessary library

-- Step 2: Define the problem statement
theorem pizza_cut_possible (N : ℕ) (h₁ : N = 201 ∨ N = 400) : 
  ∃ (cuts : ℕ), cuts ≤ 100 ∧ 
  ∀ (parts : set ℝ), parts.card = N ∧ (∀ part ∈ parts, part.area = 1) :=
by
  sorry

end pizza_cut_possible_l337_337536


namespace right_angled_triangle_with_inscribed_circle_isosceles_triangle_with_inscribed_circle_l337_337266

structure Triangle :=
(base : ℝ)
(height : ℝ)
(hypotenuse : ℝ)

structure IsoscelesTriangle :=
(side : ℝ)
(base : ℝ)

def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

noncomputable def perimeter_triangle (t : Triangle) : ℝ :=
t.base + t.height + t.hypotenuse

noncomputable def perimeter_isosceles_triangle (it : IsoscelesTriangle) : ℝ :=
2 * it.side + it.base

def angles_triangle := (α β : ℝ) -> Triangle -> α + β = 90

def angles_isosceles_triangle := (α β : ℝ) -> IsoscelesTriangle -> α = β

theorem right_angled_triangle_with_inscribed_circle (r : ℝ)
    (P : ℝ := 2 * circumference r) 
    (t : Triangle) 
    (α β : ℝ) 
    (h_perimeter : perimeter_triangle t = P)
    (h_angles : angles_triangle α β t) :
    (α ≈ 58 ∧ β ≈ 32) :=
sorry

theorem isosceles_triangle_with_inscribed_circle (r : ℝ)
    (P : ℝ := 2 * circumference r)
    (it : IsoscelesTriangle)
    (α β : ℝ)
    (h_perimeter : perimeter_isosceles_triangle it = P)
    (h_angles : angles_isosceles_triangle α β it) :
    (α ≈ 75 ∧ β ≈ 30) :=
sorry

end right_angled_triangle_with_inscribed_circle_isosceles_triangle_with_inscribed_circle_l337_337266


namespace prove_expression_l337_337159

theorem prove_expression (a b : ℕ) 
  (h1 : 180 % 2^a = 0 ∧ 180 % 2^(a+1) ≠ 0)
  (h2 : 180 % 3^b = 0 ∧ 180 % 3^(b+1) ≠ 0) :
  (1 / 4 : ℚ)^(b - a) = 1 := 
sorry

end prove_expression_l337_337159


namespace arithmetic_sequence_sum_l337_337464

theorem arithmetic_sequence_sum (S : ℕ → ℝ) (h_arith : ∀ k, S (k + 1) - S k = S 1 - S 0)
  (h_S5 : S 5 = 10) (h_S10 : S 10 = 18) : S 15 = 26 :=
by
  -- Rest of the proof goes here
  sorry

end arithmetic_sequence_sum_l337_337464


namespace sum_of_fractions_inequality_l337_337052

theorem sum_of_fractions_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + (1 / b) + b + (1 / c) + c + (1 / a) ≥ 6 →
  ∃ x ∈ {a + (1 / b), b + (1 / c), c + (1 / a)}, x ≥ 2 :=
by
  sorry

end sum_of_fractions_inequality_l337_337052


namespace volunteer_arrangements_l337_337000

noncomputable def arrangements (volunteers : ℕ) (exits : ℕ) : ℕ :=
  if h : volunteers ≥ exits then
    4 ^ (volunteers - exits) * nat.choose (volunteers - 1) (exits - 1)
  else 0

theorem volunteer_arrangements :
  arrangements 5 4 = 240 :=
by
  sorry

end volunteer_arrangements_l337_337000


namespace find_a_l337_337838

theorem find_a (a : ℝ) (x : ℝ) :
  (∃ b : ℝ, (9 * x^2 - 18 * x + a) = (3 * x + b) ^ 2) → a = 9 := by
  sorry

end find_a_l337_337838


namespace determine_power_function_l337_337230

noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x ^ α

theorem determine_power_function (α : ℝ) :
  f α 3 = real.sqrt 3 → α = 1 / 2 :=
by
  intro h
  sorry

end determine_power_function_l337_337230


namespace max_perfect_squares_sequence_l337_337401

def seq (a₀ : ℕ) : ℕ → ℕ
| 0     => a₀
| (n+1) => (seq n) ^ 5 + 487

theorem max_perfect_squares_sequence (m : ℕ) : m = 9 → 
(∀ n : ℕ, (∃ k : ℕ, seq m n = k^2) → n ≤ 1) := 
sorry

end max_perfect_squares_sequence_l337_337401


namespace combined_area_of_triangles_l337_337590

theorem combined_area_of_triangles :
  let line1 := fun (x : ℝ) => x
  let line2 := fun (x : ℝ) => -6
  let line3 := fun (x : ℝ) => -2 * x + 4
  let triangle_area := fun (base height : ℝ) => 1 / 2 * base * height
  let area_triangle1 := triangle_area 6 6
  let area_triangle2 := triangle_area (4 / 3) (4 / 3)
  let area_triangle3 := triangle_area 6 16
  let combined_area := area_triangle1 + area_triangle2 + area_triangle3
  combined_area = 66 + 8 / 9 :=
by
  let line1 := fun (x : ℝ) => x
  let line2 := fun (x : ℝ) => -6
  let line3 := fun (x : ℝ) => -2 * x + 4
  let triangle_area := fun (base height : ℝ) => 1 / 2 * base * height
  let area_triangle1 := triangle_area 6 6
  let area_triangle2 := triangle_area (4 / 3) (4 / 3)
  let area_triangle3 := triangle_area 6 16
  let combined_area := area_triangle1 + area_triangle2 + area_triangle3
  have h1 : area_triangle1 = 18 := by sorry
  have h2 : area_triangle2 = 8 / 9 := by sorry
  have h3 : area_triangle3 = 48 := by sorry
  show combined_area = 66 + 8 / 9 from by
    rw [←h1, ←h2, ←h3]
    norm_num

end combined_area_of_triangles_l337_337590


namespace find_f_neg_8_l337_337426

def f : ℝ → ℝ 
| x => if x > 0 then Real.log x / Real.log 2 else f (x + 6)

theorem find_f_neg_8 : f (-8) = 2 :=
sorry

end find_f_neg_8_l337_337426


namespace calculate_drift_l337_337998

theorem calculate_drift (w v t : ℝ) (hw : w = 400) (hv : v = 10) (ht : t = 50) : v * t - w = 100 :=
by
  sorry

end calculate_drift_l337_337998


namespace factorization_sum_l337_337948

theorem factorization_sum (a b c : ℤ) 
  (h1 : ∀ x : ℝ, (x + a) * (x + b) = x^2 + 13 * x + 40)
  (h2 : ∀ x : ℝ, (x - b) * (x - c) = x^2 - 19 * x + 88) :
  a + b + c = 24 := 
sorry

end factorization_sum_l337_337948


namespace max_distance_from_center_to_line_max_distance_from_center_to_line_exact_l337_337034

/-- Define the distance from a point (x0, y0) to the line Ax + By + C = 0 -/
def distance_from_point_to_line (x0 y0 A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / (sqrt (A^2 + B^2))

/-- Define the conditions for the problem -/
def circle_with_point (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 3)^2 = 1

/-- Define the question as a theorem statement -/
theorem max_distance_from_center_to_line : ∀ (x y : ℝ),
  circle_with_point x y → 
  distance_from_point_to_line x y 3 (-4) (-4) ≤ 3 :=
by
  sorry

theorem max_distance_from_center_to_line_exact : 
  ∃ (x y : ℝ), circle_with_point x y ∧ distance_from_point_to_line x y 3 (-4) (-4) = 3 :=
by
  sorry

end max_distance_from_center_to_line_max_distance_from_center_to_line_exact_l337_337034


namespace pascal_triangle_contains_53_l337_337811

theorem pascal_triangle_contains_53:
  ∃! n, ∃ k, (n ≥ 0) ∧ (k ≥ 0) ∧ (binom n k = 53) := 
sorry

end pascal_triangle_contains_53_l337_337811


namespace traci_flour_brought_l337_337595

-- Definitions based on the conditions
def harris_flour : ℕ := 400
def flour_per_cake : ℕ := 100
def cakes_each : ℕ := 9

-- Proving the amount of flour Traci brought
theorem traci_flour_brought :
  (cakes_each * flour_per_cake) - harris_flour = 500 :=
by
  sorry

end traci_flour_brought_l337_337595


namespace james_total_distance_l337_337145

structure Segment where
  speed : ℝ -- speed in mph
  time : ℝ -- time in hours

def totalDistance (segments : List Segment) : ℝ :=
  segments.foldr (λ seg acc => seg.speed * seg.time + acc) 0

theorem james_total_distance :
  let segments := [
    Segment.mk 30 0.5,
    Segment.mk 60 0.75,
    Segment.mk 75 1.5,
    Segment.mk 60 2
  ]
  totalDistance segments = 292.5 :=
by
  sorry

end james_total_distance_l337_337145


namespace probability_le_zero_l337_337421

variable {σ : ℝ} (X : ℝ → ℝ)
variable [NormalDist X 1 σ]  -- X ~ N(1, σ^2)

def probability_le_two (X : ℝ → ℝ) : Prop :=
  ∃ p : ℝ, P X ≤ 2 = 0.72

theorem probability_le_zero (σ : ℝ) (X : ℝ → ℝ) 
    [NormalDist X 1 σ] 
    (h : probability_le_two X) : 
    P X ≤ 0 = 0.28 :=
sorry

end probability_le_zero_l337_337421


namespace combination_eq_permutation_div_factorial_l337_337635

-- Step d): Lean 4 Statement

variables (n k : ℕ)

-- Define combination C_n^k is any k-element subset of an n-element set
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Define permutation A_n^k is the number of ways to arrange k elements out of n elements
def permutation (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

-- Statement to prove: C_n^k = A_n^k / k!
theorem combination_eq_permutation_div_factorial :
  combination n k = permutation n k / (Nat.factorial k) :=
by
  sorry

end combination_eq_permutation_div_factorial_l337_337635


namespace perpendicular_condition_sufficient_but_not_necessary_l337_337739

theorem perpendicular_condition_sufficient_but_not_necessary (m : ℝ) (h : m = -1) :
  (∀ x y : ℝ, mx + (2 * m - 1) * y + 1 = 0 ∧ 3 * x + m * y + 2 = 0) → (m = 0 ∨ m = -1) → (m = 0 ∨ m = -1) :=
by
  intro h1 h2
  sorry

end perpendicular_condition_sufficient_but_not_necessary_l337_337739


namespace pythagorean_inequality_l337_337901

variables (a b c : ℝ) (n : ℕ)

theorem pythagorean_inequality (h₀ : a > b) (h₁ : b > c) (h₂ : a^2 = b^2 + c^2) (h₃ : n > 2) : a^n > b^n + c^n :=
sorry

end pythagorean_inequality_l337_337901


namespace new_variance_l337_337398

variable (n : ℕ) (x : ℕ → ℝ) (s : ℝ)

-- The original variance definition
def variance (n : ℕ) (x : ℕ → ℝ) (s : ℝ) : Prop :=
  s ^ 2 = (1 / n) * (Finset.sum (Finset.range n) (λ i, (x i - (Finset.sum (Finset.range n) x / n)) ^ 2))

-- The proof goal
theorem new_variance (n : ℕ) (x : ℕ → ℝ) (s : ℝ) (h : variance n x s) :
  variance n (λ i, 10 * x i) (10 * s) :=
sorry

end new_variance_l337_337398


namespace perpendicular_bisector_equation_equal_distances_imply_m_values_l337_337127

namespace Geometry

-- Define the coordinates of points A and B
def A : ℝ × ℝ := (-3, -4)
def B : ℝ × ℝ := (6, 3)

-- Define the line equation l: x + my + 1 = 0
variable (m : ℝ)

-- Define the proof problem for part 1
theorem perpendicular_bisector_equation :
  let midpoint := (A.1 + B.1) / 2, (A.2 + B.2) / 2
  let slope_AB := (B.2 - A.2) / (B.1 - A.1)
  let perpendicular_slope := -1 / slope_AB
  let bisector := (midpoint.2 + perpendicular_slope * (x - midpoint.1))
  9 * x + 7 * y - 10 = 0 :=
sorry

-- Define the proof problem for part 2
theorem equal_distances_imply_m_values (equal_distances : 
  abs (4 * m + 2) / sqrt (1 + m ^ 2) =
  abs (3 * m + 7) / sqrt (1 + m ^ 2)) :
  m = 5 ∨ m = -9 / 7 :=
sorry

end Geometry

end perpendicular_bisector_equation_equal_distances_imply_m_values_l337_337127


namespace total_price_correct_l337_337576

-- Define the initial price, reduction, and the number of boxes
def initial_price : ℝ := 104
def price_reduction : ℝ := 24
def number_of_boxes : ℕ := 20

-- Define the new price as initial price minus the reduction
def new_price := initial_price - price_reduction

-- Define the total price as the new price times the number of boxes
def total_price := (number_of_boxes : ℝ) * new_price

-- The goal is to prove the total price equals 1600
theorem total_price_correct : total_price = 1600 := by
  sorry

end total_price_correct_l337_337576


namespace line_parallel_to_plane_l337_337364

variable {Point : Type} [MetricSpace Point]

structure Quadrilateral (Point : Type) :=
(A B C D : Point)

def is_midpoint (P Q R : Point) : Prop :=
dist P R = dist P Q + dist Q R / 2

def is_parallel (l : Set Point) (plane : Set Point) : Prop := sorry -- Definition for line-plane parallelism

def midpoints {Point : Type} [MetricSpace Point] (quad : Quadrilateral Point) : Point × Point × Point × Point :=
  let e : Point := sorry -- use midpoint property for AB
  let f : Point := sorry -- use midpoint property for BC
  let g : Point := sorry -- use midpoint property for CD
  let h : Point := sorry -- use midpoint property for DA
  (e, f, g, h)

theorem line_parallel_to_plane {Point : Type} [MetricSpace Point] (quad : Quadrilateral Point) :
  let (E, F, G, H) := midpoints quad in
  is_parallel ({A, C} : Set Point) ({E, F, G, H} : Set Point) :=
sorry -- Proof goes here

end line_parallel_to_plane_l337_337364


namespace triangle_tangent_area_max_l337_337476

-- Definitions of the problem
variables {A B C : Type} [inner_product_space ℝ A]

def sides (a b c : ℝ) (α β γ : ℝ) :=
  a^2 + b^2 = a * b + c^2

def tan_subtract (C : ℝ) := 
  tan (C - (pi / 4)) = 2 - sqrt 3

def area_max (a b S : ℝ) :=
  S ≤ (3 * sqrt 3) / 4
  
-- Statement of the problem
theorem triangle_tangent_area_max 
  (a b c : ℝ) (C : ℝ)
  (hc : c = sqrt 3)
  (h_sides : a^2 + b^2 = a * b + c^2) :
  (tan_subtract C) ∧ (area_max a b ((sqrt 3) / 4 * a * b)) := 
sorry

end triangle_tangent_area_max_l337_337476


namespace matrix_inverse_or_zero_l337_337013

def matrix_invertibility (A : Matrix (Fin 2) (Fin 2) ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let det := A.det
  if det = 0 then Matrix.zero else A⁻¹

theorem matrix_inverse_or_zero :
  let A : Matrix (Fin 2) (Fin 2) ℝ :=
    ![![5, 10],
      ![-15, -30]]
  let zero_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
    ![![0, 0],
      ![0, 0]]
  matrix_invertibility A = zero_matrix := by
  sorry

end matrix_inverse_or_zero_l337_337013


namespace sin_squared_not_periodic_l337_337479

noncomputable def sin_squared (x : ℝ) : ℝ := Real.sin (x^2)

theorem sin_squared_not_periodic : 
  ¬ (∃ T > 0, ∀ x ∈ Set.univ, sin_squared (x + T) = sin_squared x) := 
sorry

end sin_squared_not_periodic_l337_337479


namespace curve_tangent_line_condition_l337_337418

theorem curve_tangent_line_condition (a b : ℝ) :
  (∀ x : ℝ, y = x^2 + a * x + b)
  → (let y_tangent := x - y + 1 in
     y_tangent = 0 → x = 1 ∧ y = b)
  → a = -1 ∧ b = 2 :=
by
  sorry

end curve_tangent_line_condition_l337_337418


namespace value_of_expression_l337_337027

theorem value_of_expression (a : ℝ) (h : a^2 + a = 0) : 4*a^2 + 4*a + 2011 = 2011 :=
by
  sorry

end value_of_expression_l337_337027


namespace MollyPresentAge_l337_337186

theorem MollyPresentAge : 
    ∃ M : ℕ, M + 18 = 5 * (M - 6) ∧ M = 12 :=
by
  use 12
  split
  · linarith
  · rfl

end MollyPresentAge_l337_337186


namespace minimize_surface_area_l337_337742

theorem minimize_surface_area (V r h : ℝ) (hV : V = π * r^2 * h) (hA : 2 * π * r^2 + 2 * π * r * h = 2 * π * r^2 + 2 * π * r * h) : 
  (h / r) = 2 := 
by
  sorry

end minimize_surface_area_l337_337742


namespace equilateral_triangles_l337_337667

-- Definitions for given conditions
variables {A B C D P Q P' Q' : Type}
variables [triangle : triangle A P Q]
variables [rect : rectangle A B C D]
variables [on_PBC : point_on_side P B C]
variables [on_QCD : point_on_side Q C D]
variables [midpoint_AP : midpoint P' A P]
variables [midpoint_AQ : midpoint Q' A Q]

-- Theorem statement
theorem equilateral_triangles (circumscribed : circumscribed_rectangle rect triangle) : 
  equilateral_triangle B Q' C ∧ equilateral_triangle C P' D := 
sorry

end equilateral_triangles_l337_337667


namespace number_of_rows_containing_53_l337_337814

theorem number_of_rows_containing_53 (h_prime_53 : Nat.Prime 53) : 
  ∃! n, (n = 53 ∧ ∃ k, k ≥ 0 ∧ k ≤ n ∧ Nat.choose n k = 53) :=
by 
  sorry

end number_of_rows_containing_53_l337_337814


namespace pascal_triangle_contains_53_l337_337805

theorem pascal_triangle_contains_53:
  ∃! n, ∃ k, (n ≥ 0) ∧ (k ≥ 0) ∧ (binom n k = 53) := 
sorry

end pascal_triangle_contains_53_l337_337805


namespace series_sum_eq_l337_337349

-- Define the series
def series_sum (a : ℕ → ℝ) (b : ℝ) := ∑' n, a n / (3^n)

-- Define the specific series in the problem
def a (n : ℕ) : ℝ := 4 * n - 3

-- Theorem statement
theorem series_sum_eq : series_sum a (3/2) :=
  sorry

end series_sum_eq_l337_337349


namespace at_least_one_student_mistaken_l337_337852

theorem at_least_one_student_mistaken
  (n : ℕ) (total_players : ℕ) (total_matches : ℕ) (players_claims : ℕ → ℕ)
  (h1 : total_players = 18)
  (h2 : total_matches = 17)
  (h3 : ∀ i, i < 6 → players_claims i = 4) :
  ∃ i, i < 6 ∧ players_claims i ≠ 4 :=
by
  -- let us consider the total number of matches claimed by the 6 students
  let claimed_matches_total := (∑ i in finset.range 6, players_claims i)
  -- according to their claims, the total matches would be 24
  have h4 : claimed_matches_total = 24 := by simp [h3]
  -- but since each match involves two players
  let actual_matches := claimed_matches_total / 2
  -- total matches would be 12 for these 6 students
  have h5 : actual_matches = 12 := by simp [h4]
  -- total matches played in the tournament is 17
  have h6 : total_remaining_matches := 17 - actual_matches
  -- remainig matches should be 5 for the 12 other players
  have h7 : total_remaining_matches = 5 := by simp [h6]
  -- but each match eliminates one player so its irrational lets return contradiction
  contradiction

end at_least_one_student_mistaken_l337_337852


namespace angle_turned_by_minute_hand_l337_337989

-- Define the inputs and required calculation
def time_hours := 2
def time_minutes := 40

-- Calculations for angle
def angle_per_hour := 360
def angle_per_minute := 360 / 60

-- This is the Lean statement that represents the given proof problem
theorem angle_turned_by_minute_hand :
  let total_angle := -(time_hours * angle_per_hour) - (time_minutes * angle_per_minute)
  total_angle = -960 :=
begin
  sorry -- Proof goes here
end

end angle_turned_by_minute_hand_l337_337989


namespace find_n_l337_337032

theorem find_n 
  (n : ℕ)
  (a : ℕ → ℕ) 
  (h : ∑ i in Finset.range (n + 1), a i = 62) 
  (h_expansion : ∑ k in Finset.range n, (1 + 1)^k = ∑ k in Finset.range (n + 1), a k) 
  : n = 5 := sorry

end find_n_l337_337032


namespace max_sum_is_1717_l337_337846

noncomputable def max_arithmetic_sum (a d : ℤ) : ℤ :=
  let n := 34
  let S : ℤ := n * (2*a + (n - 1)*d) / 2
  S

theorem max_sum_is_1717 (a d : ℤ) (h1 : a + 16 * d = 52) (h2 : a + 29 * d = 13) (hd : d = -3) (ha : a = 100) :
  max_arithmetic_sum a d = 1717 :=
by
  unfold max_arithmetic_sum
  rw [hd, ha]
  -- Add the necessary steps to prove max_arithmetic_sum 100 (-3) = 1717
  -- Sorry ensures the theorem can be checked syntactically without proof
  sorry

end max_sum_is_1717_l337_337846


namespace problem1_correct_problem2_correct_l337_337879

noncomputable def problem1 (A C : ℝ) (h : Real.tan A + Real.tan C = Real.sqrt 3 * (Real.tan A * Real.tan C - 1)) : Prop :=
  ∃ B : ℝ,
  B = Real.pi / 3 ∧ 
  A + B + C = Real.pi

noncomputable def problem2 (a c : ℝ) (h1 : b = 2) (h2 : a^2 + c^2 = a * c + 4) 
  (h3 : a * c ≤ 4) : Prop :=
  (∃ A C : ℝ, maxArea : ℝ, 
  maxArea = (1 / 2) * a * c * Real.sin (Real.pi / 3) ∧ 
  maxArea = Real.sqrt 3)

-- Theorem statements without proofs
theorem problem1_correct : ∀ (A C : ℝ), (t A C -> problem1 A C) :=
  begin
    sorry
  end

theorem problem2_correct : ∀ (a c : ℝ), (h1 : b = 2) -> (h2 : a^2 + c^2 = a * c + 4) -> (h3 : a * c ≤ 4) -> problem2 a c :=
  begin
    sorry
  end

end problem1_correct_problem2_correct_l337_337879


namespace determine_x_when_a_greater_b_l337_337507

theorem determine_x_when_a_greater_b (x : ℝ) (h : (∛x + ∛(30 - x) = 3) ∧ (∛x > ∛(30 - x))) : 
  x = (9 + Real.sqrt 93) / 6 ^ 3 :=
sorry

end determine_x_when_a_greater_b_l337_337507


namespace range_of_a_l337_337438

variable (a : ℝ)

def A (a : ℝ) : Set ℝ := {x : ℝ | (a * x - 1) / (x - a) < 0}

theorem range_of_a (h1 : 2 ∈ A a) (h2 : 3 ∉ A a) : (1 / 3 : ℝ) ≤ a ∧ a < 1 / 2 ∨ 2 < a ∧ a ≤ 3 :=
by
  sorry

end range_of_a_l337_337438


namespace unit_cube_400_points_l337_337138

theorem unit_cube_400_points (cube : set (ℝ × ℝ × ℝ))
  (points : finite (set (ℝ × ℝ × ℝ))) (h_cube : cube = set.Icc (0, 0, 0) (1, 1, 1))
  (h_points : points.card = 400) (h_inside : ∀ p ∈ points, p ∈ cube) :
  ∃ (p1 p2 p3 p4 : ℝ × ℝ × ℝ), 
    p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ p4 ∈ points ∧
    dist p1 p2 ≤ 8/23 ∧ dist p1 p3 ≤ 8/23 ∧ dist p1 p4 ≤ 8/23 ∧ dist p2 p3 ≤ 8/23 ∧ dist p2 p4 ≤ 8/23 ∧ dist p3 p4 ≤ 8/23 := 
sorry

end unit_cube_400_points_l337_337138


namespace sine_angle_VC_plane_MBC_l337_337475

open_locale real

variables {V A B C D M : Point}
-- Given Conditions
axiom base_square_pyramid : base_square_pyramid V A B C D
axiom VA_perpendicular_base : VA.is_perpendicular (plane A B C D)
axiom VA_eq_AB : VA.length = AB.length
axiom M_midpoint_VA : M.is_midpoint V A

-- Question: To prove the sine of the angle formed by line VC and plane MBC
theorem sine_angle_VC_plane_MBC : 
  sin (angle_between_line_and_plane V C M B C) = some_value := 
sorry

end sine_angle_VC_plane_MBC_l337_337475


namespace expected_value_sum_even_marble_l337_337093

theorem expected_value_sum_even_marble :
  let marbles := {1, 2, 3, 4, 5, 6, 7, 8}
  ∃ values_sm (draw : ({ x : ℕ | x ∈ marbles } × { y : ℕ | y ∈ marbles }) -> ℚ),
  (∀ x y, x ≠ y ∧ (x % 2 = 0 ∨ y % 2 = 0) -> values_sm (⟨x, _⟩, ⟨y, _⟩) = (x + y)) → 
  (let favorable_pairs := ∑ x y in marbles, if x ≠ y ∧ (x % 2 = 0 ∨ y % 2 = 0) then 1 else 0,
       total_sum := ∑ x y in marbles, if x ≠ y ∧ (x % 2 = 0 ∨ y % 2 = 0) then x + y else 0
  in total_sum / favorable_pairs = 96 / 11) :=
sorry

end expected_value_sum_even_marble_l337_337093


namespace stride_vs_leap_difference_l337_337910

theorem stride_vs_leap_difference :
  ∀ (strides := 54) (leaps := 16) (poles := 81) (distance := 10560), 
    let n_gaps := poles - 1 in
    let elmer_strides := strides * n_gaps in
    let oscar_leaps := leaps * n_gaps in
    let elmer_stride_length := distance / elmer_strides in
    let oscar_leap_length := distance / oscar_leaps in
    (oscar_leap_length - elmer_stride_length) = 5.8 :=
by
  intros
  let n_gaps := poles - 1
  let elmer_strides := strides * n_gaps
  let oscar_leaps := leaps * n_gaps
  let elmer_stride_length := distance / elmer_strides
  let oscar_leap_length := distance / oscar_leaps
  have h_distance_positive : distance > 0 := sorry
  have h_strides_positive : strides > 0 := sorry
  have h_leaps_positive : leaps > 0 := sorry
  have h_n_gaps_nonnegative : n_gaps ≥ 0 := by
    linarith
  have h_elmer_strides_positive : elmer_strides > 0 := by
    apply mul_pos
    exact h_strides_positive
    linarith
  have h_oscar_leaps_positive : oscar_leaps > 0 := by
    apply mul_pos
    exact h_leaps_positive
    linarith
  have h_elmer_stride_length_approx : elmer_stride_length ≈ 2.44 := by
    sorry
  have h_oscar_leap_length_approx : oscar_leap_length ≈ 8.25 := by
    sorry
  have h_difference_approx : (oscar_leap_length - elmer_stride_length) ≈ 5.81 := by
    sorry
  exact (oscar_leap_length - elmer_stride_length) = 5.8

end stride_vs_leap_difference_l337_337910


namespace problem_solution_l337_337287

theorem problem_solution : (3.242 * 14) / 100 = 0.45388 := by
  sorry

end problem_solution_l337_337287


namespace part_one_solution_set_part_two_m_range_l337_337062

def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - (m + 1) * x + 1

theorem part_one_solution_set (m : ℝ) (h : m > 0) : 
  (if m > 1 then {x : ℝ | x ∈ Ioo (1/m) 1}
  else if 0 < m ∧ m < 1 then {x : ℝ | x ∈ Ioo 1 (1/m)}
  else if m = 1 then ∅
  else ∅) = {x : ℝ | f m x < 0} :=
sorry

theorem part_two_m_range {m : ℝ} : 
  (∀ x ∈ Icc 1 2, f m x ≤ 2) → m ∈ Iio (3/2) :=
sorry

end part_one_solution_set_part_two_m_range_l337_337062


namespace ratio_violin_piano_l337_337691

variables (daily_piano : ℕ) (weekly_days : ℕ) (monthly_total : ℕ) (weeks_in_month : ℕ)

def weekly_piano_time : ℕ := daily_piano * weekly_days

def monthly_piano_time : ℕ := weekly_piano_time daily_piano weekly_days * weeks_in_month

def monthly_violin_time : ℕ := monthly_total - monthly_piano_time daily_piano weekly_days weeks_in_month

def daily_violin_time : ℕ := monthly_violin_time daily_piano weekly_days weeks_in_month / (weekly_days * weeks_in_month)

theorem ratio_violin_piano (h1 : daily_piano = 20) 
                            (h2 : weekly_days = 6)
                            (h3 : monthly_total = 1920)
                            (h4 : weeks_in_month = 4) :
  daily_violin_time daily_piano weekly_days monthly_total weeks_in_month / daily_piano = 3 := 
by 
  sorry

end ratio_violin_piano_l337_337691


namespace min_value_cos_sin_half_l337_337377

theorem min_value_cos_sin_half (A : ℝ) (h1 : (∃ C : ℝ, sin(C) = 1/2 ∧ cos(C) = real.sqrt 3 / 2))
  (h2 : A = 60 ∨ A = -180 ∨ A = 30 ∨ A = 90 ∨ ¬(A = 60 ∨ A = -180 ∨ A = 30 ∨ A = 90)) 
  : min ((cos (A / 2) + (real.sqrt 3) * sin (A / 2))) ≤ -2 := 
  sorry

end min_value_cos_sin_half_l337_337377


namespace range_of_a1_l337_337040

noncomputable def sequence_a (n : ℕ) : ℤ := sorry
noncomputable def sum_S (n : ℕ) : ℤ := sorry

theorem range_of_a1 :
  (∀ n : ℕ, n > 0 → sum_S n + sum_S (n+1) = 2 * n^2 + n) ∧
  (∀ n : ℕ, n > 0 → sequence_a n < sequence_a (n+1)) →
  -1/4 < sequence_a 1 ∧ sequence_a 1 < 3/4 := sorry

end range_of_a1_l337_337040


namespace pqrsum_eq_neg209_l337_337504

noncomputable def Q (z : ℂ) (p q r : ℝ) : ℂ :=
  z^3 + (p : ℂ) * z^2 + (q : ℂ) * z + (r : ℂ)

theorem pqrsum_eq_neg209 (u p q r : ℂ) (i : ℂ) (hu : u.im ≠ 0) (huj : i^2 = -1)
  (hroots : (Q (u + 2 * i) p q r) = 0 ∧ (Q (u + 7 * i) p q r) = 0 ∧ (Q (2 * u - 5) p q r) = 0)
  (hreals : p.im = 0 ∧ q.im = 0 ∧ r.im = 0) :
  p + q + r = -209 :=
sorry

end pqrsum_eq_neg209_l337_337504


namespace triangle_angle_conditions_l337_337478

theorem triangle_angle_conditions (A B C : Point) (angle_A : ℝ) (is_median_equal_altitude : length (median B) = length (altitude C))
  (h1 : ∠ A B C = 30) : ∠ B A C = 90 ∧ ∠ C A B = 60 :=
sorry

end triangle_angle_conditions_l337_337478


namespace students_neither_play_football_nor_cricket_l337_337545

theorem students_neither_play_football_nor_cricket
  (total_students football_players cricket_players both_players : ℕ)
  (h_total : total_students = 470)
  (h_football : football_players = 325)
  (h_cricket : cricket_players = 175)
  (h_both : both_players = 80) :
  (total_students - (football_players + cricket_players - both_players)) = 50 :=
by
  sorry

end students_neither_play_football_nor_cricket_l337_337545


namespace matrix_self_inverse_pairs_l337_337701

theorem matrix_self_inverse_pairs :
  ∃ p : Finset (ℝ × ℝ), (∀ a d, (a, d) ∈ p ↔ (∃ (m : Matrix (Fin 2) (Fin 2) ℝ), 
    m = !![a, 4; -9, d] ∧ m * m = 1)) ∧ p.card = 2 :=
by {
  sorry
}

end matrix_self_inverse_pairs_l337_337701


namespace part1_part2_l337_337737

-- Define the vectors m and n
def m (x : ℝ) : ℝ × ℝ :=
  (2 * Real.sin (x + Real.pi / 6), 1)

def n (x : ℝ) : ℝ × ℝ :=
  (2 * Real.cos x, -1)

-- Define the function f as the dot product of m and n
def f (x : ℝ) : ℝ :=
  (m x).fst * (n x).fst + (m x).snd * (n x).snd

-- Prove that if m is parallel to n and x ∈ [0, π], then x = 2π/3
theorem part1 (h_parallel : ∀ x : ℝ, (m x).fst * (n x).snd = (m x).snd * (n x).fst) (h_domain : ∀ x, x ∈ Set.Icc 0 Real.pi) : 
  ∀ x : ℝ, (m x) = (n x) → x = 2 * Real.pi / 3 :=
sorry

-- Prove the monotonic intervals of f
theorem part2 : 
  (∀ x : ℝ, f x = 2 * Real.sqrt 3 * Real.sin (2 * x + Real.pi / 6)) → 
  (∀ k : ℤ, is_increasing (f : ℝ → ℝ) [k * Real.pi - Real.pi / 3, k * Real.pi + Real.pi / 6]) ∧ 
  (∀ k : ℤ, is_decreasing (f : ℝ → ℝ) [k * Real.pi + Real.pi / 6, k * Real.pi + 2 * Real.pi / 3]) :=
sorry

end part1_part2_l337_337737


namespace pq_sum_l337_337054

theorem pq_sum {p q : ℤ}
  (h : ∀ x : ℤ, 36 * x^2 - 4 * (p^2 + 11) * x + 135 * (p + q) + 576 = 0) :
  p + q = 20 :=
sorry

end pq_sum_l337_337054


namespace share_price_increase_l337_337670

theorem share_price_increase
  (P : ℝ)
  -- At the end of the first quarter, the share price was 20% higher than at the beginning of the year.
  (end_of_first_quarter : ℝ := 1.20 * P)
  -- The percent increase from the end of the first quarter to the end of the second quarter was 25%.
  (percent_increase_second_quarter : ℝ := 0.25)
  -- At the end of the second quarter, the share price
  (end_of_second_quarter : ℝ := end_of_first_quarter + percent_increase_second_quarter * end_of_first_quarter) :
  -- What is the percent increase in share price at the end of the second quarter compared to the beginning of the year?
  end_of_second_quarter = 1.50 * P :=
by
  sorry

end share_price_increase_l337_337670


namespace new_team_average_weight_l337_337623

theorem new_team_average_weight :
  let original_team_weight := 7 * 94
  let new_players_weight := 110 + 60
  let new_total_weight := original_team_weight + new_players_weight
  let new_player_count := 9
  (new_total_weight / new_player_count) = 92 :=
by
  let original_team_weight := 7 * 94
  let new_players_weight := 110 + 60
  let new_total_weight := original_team_weight + new_players_weight
  let new_player_count := 9
  sorry

end new_team_average_weight_l337_337623


namespace inequality_on_unit_circle_l337_337518

theorem inequality_on_unit_circle
  (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h : a * b + c * d = 1)
  (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ)
  (h₁ : x₁^2 + y₁^2 = 1) (h₂ : x₂^2 + y₂^2 = 1) 
  (h₃ : x₃^2 + y₃^2 = 1) (h₄ : x₄^2 + y₄^2 = 1) :
  (a * y₁ + b * y₂ + c * y₃ + d * y₄)^2 + (a * x₁ + b * x₃ + c * x₂ + d * x₄)^2
  ≤ 2 * ((a^2 + b^2) / (a * b) + (c^2 + d^2) / (c * d)) :=
begin
  sorry
end

end inequality_on_unit_circle_l337_337518


namespace no_solution_abs_val_l337_337279

theorem no_solution_abs_val (x : ℝ) : ¬(∃ x : ℝ, |5 * x| + 7 = 0) :=
sorry

end no_solution_abs_val_l337_337279


namespace solve_inequality_l337_337430

noncomputable def f (x : ℝ) : ℝ := -x^2 + 2 * x

theorem solve_inequality {x : ℝ} (hx : 0 < x) : 
  f (Real.log x / Real.log 2) < f 2 ↔ (0 < x ∧ x < 1) ∨ (4 < x) :=
by
sorry

end solve_inequality_l337_337430


namespace Genevieve_cherry_weight_l337_337731

theorem Genevieve_cherry_weight
  (cost_per_kg : ℕ) (short_of_total : ℕ) (amount_owned : ℕ) (total_kg : ℕ) :
  cost_per_kg = 8 →
  short_of_total = 400 →
  amount_owned = 1600 →
  total_kg = 250 :=
by
  intros h_cost_per_kg h_short_of_total h_amount_owned
  have h_equation : 8 * total_kg = 1600 + 400 := by
    rw [h_cost_per_kg, h_short_of_total, h_amount_owned]
    apply sorry -- This is where the exact proof mechanism would go
  sorry -- Skipping the remainder of the proof

end Genevieve_cherry_weight_l337_337731


namespace calculate_amount_l337_337292

-- Definitions
def principal_amount : ℝ := 51200
def rate_of_increase : ℝ := 1 / 8
def years : ℕ := 2

-- Theorem statement
theorem calculate_amount (principal_amount : ℝ) (rate_of_increase : ℝ) (years : ℕ) :
  principal_amount = 51200 → rate_of_increase = 1 / 8 → years = 2 →
  principal_amount * (1 + rate_of_increase) ^ years = 64800 :=
by
  intros h₁ h₂ h₃
  sorry

end calculate_amount_l337_337292


namespace days_to_complete_work_l337_337843

theorem days_to_complete_work :
  ∀ (M B: ℝ) (D: ℝ),
    (M = 2 * B)
    → (13 * M + 24 * B) * 4 = (12 * M + 16 * B) * D
    → D = 5 :=
by
  intros M B D h1 h2
  sorry

end days_to_complete_work_l337_337843


namespace slope_angle_of_line_l337_337583

theorem slope_angle_of_line (θ : ℝ) (hθ_range : 0 ≤ θ ∧ θ ≤ 180) : θ = 60 :=
by {
  -- The line equation y = √3x implies a slope (m) of √3
  have hm : tan θ = √3,
  -- Evaluate the angle that satisfies tan θ = √3 within the given range
  simp [tan_60, tan],
  exact 60,
  sorry
}

end slope_angle_of_line_l337_337583


namespace find_a12_a12_value_l337_337404

variable (a : ℕ → ℝ)

-- Given conditions
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

axiom h1 : a 6 + a 10 = 16
axiom h2 : a 4 = 1

-- Theorem to prove
theorem find_a12 : a 6 + a 10 = a 4 + a 12 := by
  -- Place for the proof
  sorry

theorem a12_value : (∃ a12, a 6 + a 10 = 16 ∧ a 4 = 1 ∧ a 6 + a 10 = a 4 + a12) → a 12 = 15 :=
by
  -- Place for the proof
  sorry

end find_a12_a12_value_l337_337404


namespace balls_combination_count_l337_337854

theorem balls_combination_count :
  let n := 10
  let total_balls := 20
  let white_balls := 9
  let red_balls := 5
  let black_balls := 6
  let condition (x y z : ℕ) := x + y + z = n ∧ 2 ≤ x ∧ x ≤ 5 ∧ 0 ≤ y ∧ y ≤ 3 ∧ 2 ≤ z ∧ z ≤ 8
  (finset.range (total_balls + 1)).sum (λ i, if condition (i % 5 + 2) (i % 4) (i % 8 + 2) then 1 else 0) = 16 :=
by
  sorry

end balls_combination_count_l337_337854


namespace percentage_reduction_l337_337650

noncomputable def reduced_price_kg : ℝ := 34.2
def total_money : ℝ := 684
def extra_kg : ℝ := 4
def original_price_kg (P : ℝ) : Prop := (total_money / P) + extra_kg = (total_money / reduced_price_kg)

theorem percentage_reduction (P : ℝ) (h : original_price_kg P) :
  (P - reduced_price_kg) / P * 100 ≈ 20.09 :=
sorry

end percentage_reduction_l337_337650


namespace concyclic_points_l337_337172

variables {A B C D E F O H X : ℝ} [Nonempty ℝ]

def is_orthocenter (H A B C : ℝ) : Prop := sorry -- Definition of orthocenter
def is_circumcenter (O A B C : ℝ) : Prop := sorry -- Definition of circumcenter
def is_foot_of_altitude (P A B C : ℝ) : Prop := sorry -- Definition of foot of the perpendicular

def is_reflection (X A E F : ℝ) : Prop := sorry -- Definition of reflection across a line

theorem concyclic_points :
  ∀ (A B C D E F O H X : ℝ),
    is_circumcenter O A B C →
    is_orthocenter H A B C →
    is_foot_of_altitude D A B C →
    is_foot_of_altitude E B A C →
    is_foot_of_altitude F C A B →
    is_reflection X A E F →
    ∃ k : ℝ, circle_eq (mk_circle O k) = {O, H, D, X} :=
begin
  sorry
end

end concyclic_points_l337_337172


namespace determine_values_of_abc_l337_337769

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c
noncomputable def f_inv (a b c : ℝ) (x : ℝ) : ℝ := c * x^2 + b * x + a

theorem determine_values_of_abc 
  (a b c : ℝ) 
  (h_f : ∀ x : ℝ, f a b c (f_inv a b c x) = x)
  (h_f_inv : ∀ x : ℝ, f_inv a b c (f a b c x) = x) : 
  a = -1 ∧ b = 1 ∧ c = 0 :=
by
  sorry

end determine_values_of_abc_l337_337769


namespace max_tied_teams_for_most_wins_l337_337858

theorem max_tied_teams_for_most_wins (n : ℕ) (h : n = 8) :
  ∃ k : ℕ, k = 7 ∧ (∀ (a b : ℕ), a ≠ b → valid_tournament_schedule n k a b ∧ all_games_played n a b) := 
sorry

/-
Definitions to be useful in the theorem
valid_tournament_schedule n k a b would check:
 - there are n teams
 - there are k teams tied for the most wins
 - each pairwise team plays exactly once

all_games_played n a b :
 - each game results in a winner and a loser
 - a valid win-loss record considering the overall number of games.
-/

end max_tied_teams_for_most_wins_l337_337858


namespace no_real_solutions_log_eq_l337_337457

noncomputable def log_defined (x : ℝ) : Prop :=
  (x + 3 > 0) ∧ (x - 1 > 0) ∧ (x^2 - 2 * x - 3 > 0)

theorem no_real_solutions_log_eq (x : ℝ) (h : log_defined x) : ¬(log (x + 3) + log (x - 1) = log (x^2 - 2 * x - 3)) :=
sorry

end no_real_solutions_log_eq_l337_337457


namespace min_area_ABCD_l337_337669

section Quadrilateral

variables {S1 S2 S3 S4 : ℝ}

-- Define the areas of the triangles
def area_APB := S1
def area_BPC := S2
def area_CPD := S3
def area_DPA := S4

-- Condition: Product of the areas of ΔAPB and ΔCPD is 36
axiom prod_APB_CPD : S1 * S3 = 36

-- We need to prove that the minimum area of the quadrilateral ABCD is 24
theorem min_area_ABCD : S1 + S2 + S3 + S4 ≥ 24 :=
by
  sorry

end Quadrilateral

end min_area_ABCD_l337_337669


namespace pipe_a_filling_time_l337_337970

theorem pipe_a_filling_time
  (pipeA_fill_time : ℝ)
  (pipeB_fill_time : ℝ)
  (both_pipes_open : Bool)
  (pipeB_shutoff_time : ℝ)
  (overflow_time : ℝ)
  (pipeB_rate : ℝ)
  (combined_rate : ℝ)
  (a_filling_time : ℝ) :
  pipeA_fill_time = 1 / 2 :=
by
  -- Definitions directly from conditions in a)
  let pipeA_fill_time := a_filling_time
  let pipeB_fill_time := 1  -- Pipe B fills in 1 hour
  let both_pipes_open := True
  let pipeB_shutoff_time := 0.5 -- Pipe B shuts 30 minutes before overflow
  let overflow_time := 0.5  -- Tank overflows in 30 minutes
  let pipeB_rate := 1 / pipeB_fill_time
  
  -- Goal to prove
  sorry

end pipe_a_filling_time_l337_337970


namespace find_median_l337_337986

-- Define the conditions
def consecutive_integers (a n : ℤ) := ∀ (i : ℤ), 0 ≤ i ∧ i < n → (a + i)

def sum_condition (a n : ℤ) := 2 * a + n - 1 = 120

-- State the main theorem
theorem find_median (a n : ℤ) (h_consecutive : consecutive_integers a n) (h_sum : sum_condition a n) :
  (2 * a + (n - 1)) / 2 = 60 := by
  sorry

end find_median_l337_337986


namespace max_value_achieved_at_a_n_a_n_le_sum_S_n_lt_l337_337904

noncomputable def f_n (n : ℕ) (x : ℝ) : ℝ := x^n * (1 - x)^2

noncomputable def a_n : ℕ → ℝ 
| 1 => 1 / 8
| n + 2 => 4 * (n + 2) ^ (n + 2) / (n + 4) ^ (n + 4)

noncomputable def S_n (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ k => a_n (k + 1))

theorem max_value_achieved_at_a_n (n : ℕ) : 
  let aₙ := if n = 1 then 1 / 8 else 4 * (n ^ n) / ((n + 2) ^ (n + 2))
  f_n n aₙ = max (f_n n 1) (f_n n (n / (n + 2))) :=
sorry

theorem a_n_le (n : ℕ) (h : n ≥ 2) : a_n n ≤ 1 / (n + 2) ^ 2 :=
sorry

theorem sum_S_n_lt (n : ℕ) : S_n n < 7 / 16 :=
sorry

end max_value_achieved_at_a_n_a_n_le_sum_S_n_lt_l337_337904


namespace minimum_spend_l337_337277

def box_length : ℝ := 20
def box_width : ℝ := 20
def box_height : ℝ := 15
def box_cost : ℝ := 0.90
def total_volume : ℝ := 3_060_000
def packing_efficiency : ℝ := 0.80

noncomputable def box_volume : ℝ := box_length * box_width * box_height
noncomputable def effective_box_volume : ℝ := box_volume * packing_efficiency
noncomputable def boxes_needed : ℝ := total_volume / effective_box_volume
noncomputable def boxes_needed_rounded : ℝ := boxes_needed.ceil
noncomputable def minimum_cost : ℝ := boxes_needed_rounded * box_cost

theorem minimum_spend (ans : ℝ) (h : ans = 574.20) : minimum_cost = ans := by
  sorry

end minimum_spend_l337_337277


namespace census_entirety_is_population_l337_337565

-- Define the options as a type
inductive CensusOptions
| Part
| Whole
| Individual
| Population

-- Define the condition: the entire object under investigation in a census
def entirety_of_objects_under_investigation : CensusOptions := CensusOptions.Population

-- Prove that the entirety of objects under investigation in a census is called Population
theorem census_entirety_is_population :
  entirety_of_objects_under_investigation = CensusOptions.Population :=
sorry

end census_entirety_is_population_l337_337565


namespace part_a_part_b_l337_337881

variable (a : ℝ) (h : a + 1 / a ∈ ℤ)

theorem part_a : a^2 + 1 / a^2 ∈ ℤ :=
by sorry

theorem part_b (n : ℕ) : a^n + 1 / a^n ∈ ℤ :=
by sorry

end part_a_part_b_l337_337881


namespace genevieve_cherries_purchase_l337_337733

theorem genevieve_cherries_purchase (cherries_cost_per_kg: ℝ) (genevieve_money: ℝ) (extra_money_needed: ℝ) (total_kg: ℝ) : 
  cherries_cost_per_kg = 8 → 
  genevieve_money = 1600 →
  extra_money_needed = 400 →
  total_kg = (genevieve_money + extra_money_needed) / cherries_cost_per_kg →
  total_kg = 250 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end genevieve_cherries_purchase_l337_337733


namespace min_a_sqrt_two_l337_337100

noncomputable def min_value_of_a (x y a : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ a > 0 ∧ (sqrt x + sqrt y ≤ a * sqrt (x + y))

theorem min_a_sqrt_two (x y a : ℝ) (h : min_value_of_a x y a) : a ≥ sqrt 2 :=
  sorry

end min_a_sqrt_two_l337_337100


namespace num_pos_whole_numbers_with_cube_roots_less_than_five_l337_337086

theorem num_pos_whole_numbers_with_cube_roots_less_than_five : 
  {n : ℕ | ∃ k : ℕ, k < 5 ∧ k^3 = n}.card = 124 :=
sorry

end num_pos_whole_numbers_with_cube_roots_less_than_five_l337_337086


namespace triangle_JKL_tangent_sine_cosine_l337_337120

variable {J K L : Type} [MetricSpace J] [MetricSpace K] [MetricSpace L]

def right_triangle (a b c : ℝ) := a^2 + b^2 = c^2

theorem triangle_JKL_tangent_sine_cosine:
  ∀ (JK JL : ℝ), right_triangle JK JL (sqrt (JK^2 + JL^2)) →
  JK = 12 →
  JL = 13 →
  tan K = JL / JK ∧ sin K * cos K ≠ 1/2 * tan K :=
by
  intros
  sorry

end triangle_JKL_tangent_sine_cosine_l337_337120


namespace range_of_b_l337_337397

open Real

theorem range_of_b (b : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 4 → abs (y - (x + b)) = 1) ↔ -sqrt 2 < b ∧ b < sqrt 2 := 
by sorry

end range_of_b_l337_337397


namespace no_two_girls_next_to_each_other_l337_337541

structure Child where
  is_boy : Bool
  initial_flowers : ℕ
  final_flowers : ℕ

noncomputable def children_list : List Child := sorry

theorem no_two_girls_next_to_each_other
  (children : List Child)
  (num_boys : ∃ n : ℕ, n = 101)
  (num_girls : ∃ n : ℕ, n = 3)
  (initial_boys_flowers : ∀ child, child ∈ children → child.is_boy = true → child.initial_flowers = 50)
  (final_boys_flowers : ∀ child, child ∈ children → child.is_boy = true → child.final_flowers = 49)
  (carnations_passing : ∀ child, child ∈ children → fres pass_criterion at signal time so each child pass flowers)
  : ∀ child1 child2, child1 ∈ children → child2 ∈ children → (child1 ≠ child2) → ¬(child1.is_boy = false ∧ child2.is_boy = false ∧ (index of child1 and child2 in children list are consecutive)) :=
sorry

end no_two_girls_next_to_each_other_l337_337541


namespace tangency_points_concurrent_l337_337163

def Circle : Type := sorry
def Point : Type := sorry

def TangentTo (c1 c2 : Circle) : Prop := sorry
def TangencyPoint (ci c : Circle) : Point := sorry
def Concurrent (A B C : Point) : Prop := sorry
def LineThrough (A B : Point) : Line := sorry

noncomputable theory

-- Given conditions
variables (ω Ω : Circle) (ω_i : Fin 8 → Circle)
  (H1 : ∀ i, TangentTo (ω_i i) ω)
  (H2 : ∀ i, TangentTo (ω_i i) Ω)
  (H3 : ∀ i, TangentTo (ω_i i) (ω_i ((i + 1) % 8)))
  (T : Fin 8 → Point)
  (H4 : ∀ i, T i = TangencyPoint (ω_i i) ω)

-- Statement to prove
theorem tangency_points_concurrent :
  Concurrent (LineThrough (T 0) (T 4)) (LineThrough (T 1) (T 5)) (LineThrough (T 2) (T 6)) (LineThrough (T 3) (T 7)) :=
sorry

end tangency_points_concurrent_l337_337163


namespace minimum_value_of_reciprocal_sums_l337_337077

theorem minimum_value_of_reciprocal_sums (a b : ℝ) (hab : a ≠ 0 ∧ b ≠ 0)
  (hC1 : ∀ x y, x^2 + y^2 + 2*a*x + a^2 - 4 = 0)
  (hC2 : ∀ x y, x^2 + y^2 - 2*b*y + b^2 - 1 = 0)
  (h_tangent : ∀ a b, sqrt(a^2 + 4*b^2) = 3) :
  (1 / a^2 + 1 / b^2) = 9 :=
sorry

end minimum_value_of_reciprocal_sums_l337_337077


namespace point_B_in_quad_IV_l337_337104

def A : ℝ × ℝ := (2, m)
def B (m : ℝ) : ℝ × ℝ := (m + 1, m - 1)

theorem point_B_in_quad_IV (m : ℝ) (h : A.snd = 0) : B(m).1 > 0 ∧ B(m).2 < 0 :=
by
  have hm : m = 0 := h
  rw [hm]
  simp only [B]
  split
  · exact zero_lt_one
  · exact neg_neg_of_pos zero_lt_one
  sorry

end point_B_in_quad_IV_l337_337104


namespace find_d_l337_337875

theorem find_d (d : ℝ) (h : ∃ (x y : ℝ), 3 * x + 5 * y + d = 0 ∧ x = -d / 3 ∧ y = -d / 5 ∧ -d / 3 + (-d / 5) = 15) : d = -225 / 8 :=
by 
  sorry

end find_d_l337_337875


namespace inequality_solution_l337_337247

theorem inequality_solution (x : ℝ) (h₁ : 1 - x < 0) (h₂ : x - 3 ≤ 0) : 1 < x ∧ x ≤ 3 :=
by
  sorry

end inequality_solution_l337_337247


namespace pascal_triangle_contains_53_l337_337810

theorem pascal_triangle_contains_53:
  ∃! n, ∃ k, (n ≥ 0) ∧ (k ≥ 0) ∧ (binom n k = 53) := 
sorry

end pascal_triangle_contains_53_l337_337810


namespace distance_from_focus_to_asymptote_l337_337714

theorem distance_from_focus_to_asymptote :
  let focus := (1, 0 : ℝ)
  let parabola := ∀ (y : ℝ), y^2 = 4 * focus.1
  let hyperbola := ∀ (x y : ℝ), x^2 - y^2 = 2
  let asymptote := ∀ (x y : ℝ), y = x
  ∃ d : ℝ, d = (1 - 0) / real.sqrt(1^2 + (-1)^2) := 
  sorry

end distance_from_focus_to_asymptote_l337_337714


namespace sequence_inequality_l337_337580

open Real

def seq (F : ℕ → ℝ) : Prop :=
  F 1 = 1 ∧ F 2 = 2 ∧ ∀ n ≥ 1, F (n + 2) = F (n + 1) + F n

theorem sequence_inequality (F : ℕ → ℝ) (h : seq F) (n : ℕ) : 
  sqrt (F (n+1))^(1/(n:ℝ)) ≥ 1 + 1 / sqrt (F n)^(1/(n:ℝ)) :=
sorry

end sequence_inequality_l337_337580


namespace cos_alpha_plus_pi_div_4_value_l337_337412

noncomputable def cos_alpha_plus_pi_div_4 (α : ℝ) (h1 : π / 2 < α ∧ α < π) (h2 : Real.sin (α - 3 * π / 4) = 3 / 5) : Real :=
  Real.cos (α + π / 4)

theorem cos_alpha_plus_pi_div_4_value (α : ℝ) (h1 : π / 2 < α ∧ α < π) (h2 : Real.sin (α - 3 * π / 4) = 3 / 5) :
  cos_alpha_plus_pi_div_4 α h1 h2 = -4 / 5 :=
sorry

end cos_alpha_plus_pi_div_4_value_l337_337412


namespace simplify_expression_l337_337928

variable (x y : ℝ) -- Define x and y as real numbers
variable (i : ℂ) [norm_num_class i] -- Define i as an imaginary unit in complex numbers

-- Define the expression and the target simplified form
def expression := (2*x + 3*i*y) * (2*x - 3*i*y) + 2*x
def target := 4*x^2 + 2*x - 9*y^2

-- statement to prove equivalence
theorem simplify_expression : expression x y = target x y :=
by
  sorry -- Proof is omitted as per instructions

end simplify_expression_l337_337928


namespace calculate_expression_l337_337680

theorem calculate_expression : (-1 : ℝ) ^ 2 + (1 / 3 : ℝ) ^ 0 = 2 := 
by
  sorry

end calculate_expression_l337_337680


namespace hexagon_area_division_l337_337325

-- Define the given conditions as a set of hypotheses and the theorem statement
theorem hexagon_area_division :
  ∀ (area_hexagon : ℝ) (n_parts : ℕ), 
    area_hexagon = 54.3 → n_parts = 6 → 
    (area_hexagon / n_parts) = 9.05 := by 
  intros area_hexagon n_parts h_area h_parts
  rw [h_area, h_parts]
  norm_num
  exact True.intro -- This line is just to satisfy Lean to compile successfully

end hexagon_area_division_l337_337325


namespace find_x_l337_337080

def vec_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v1.2, v2.1 + v2.2)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem find_x (x : ℝ) (a b c : ℝ × ℝ) (h1 : a = (1, 1))
                               (h2 : b = (-1, 3))
                               (h3 : c = (2, x))
                               (h4 : dot_product (vec_add (vec_add a a) a, b) c = 10) :
  x = 1 :=
sorry

end find_x_l337_337080


namespace exists_N_minimal_l337_337924

-- Assuming m and n are positive and coprime
variables (m n : ℕ)
variables (h_pos_m : 0 < m) (h_pos_n : 0 < n)
variables (h_coprime : Nat.gcd m n = 1)

-- Statement of the mathematical problem
theorem exists_N_minimal :
  ∃ N : ℕ, (∀ k : ℕ, k ≥ N → ∃ a b : ℕ, k = a * m + b * n) ∧
           (N = m * n - m - n + 1) := 
  sorry

end exists_N_minimal_l337_337924


namespace bob_raise_per_hour_l337_337674

theorem bob_raise_per_hour
  (hours_per_week : ℕ := 40)
  (monthly_housing_reduction : ℤ := 60)
  (weekly_earnings_increase : ℤ := 5)
  (weeks_per_month : ℕ := 4) :
  ∃ (R : ℚ), 40 * R - (monthly_housing_reduction / weeks_per_month) + weekly_earnings_increase = 0 ∧
              R = 0.25 := 
by
  sorry

end bob_raise_per_hour_l337_337674


namespace number_of_rows_containing_53_l337_337818

theorem number_of_rows_containing_53 (h_prime_53 : Nat.Prime 53) : 
  ∃! n, (n = 53 ∧ ∃ k, k ≥ 0 ∧ k ≤ n ∧ Nat.choose n k = 53) :=
by 
  sorry

end number_of_rows_containing_53_l337_337818


namespace no_counterexample_exists_l337_337170

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

theorem no_counterexample_exists :
  ∀ n ∈ [45, 54, 63, 81],
  (is_divisible_by_9 (sum_of_digits n) → is_divisible_by_9 n) :=
by
  intros n hn
  have : n = 45 ∨ n = 54 ∨ n = 63 ∨ n = 81 := by simp [hn]
  cases this <;> {
    simp [sum_of_digits, is_divisible_by_9, this]
    apply exists.intro 45; norm_num <|> apply exists.intro 54; norm_num <|> apply exists.intro 63; norm_num <|> apply exists.intro 81; norm_num
  }

end no_counterexample_exists_l337_337170


namespace find_smallest_c_l337_337342

theorem find_smallest_c (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
    (graph_eq : ∀ x, (a * Real.sin (b * x + c) + d) = 5 → x = (π / 6))
    (amplitude_eq : a = 3) : c = π / 2 :=
sorry

end find_smallest_c_l337_337342


namespace kabulek_four_digits_l337_337218

def isKabulekNumber (N: ℕ) : Prop :=
  let a := N / 100
  let b := N % 100
  (a + b) ^ 2 = N

theorem kabulek_four_digits :
  {N : ℕ | 1000 ≤ N ∧ N < 10000 ∧ isKabulekNumber N} = {2025, 3025, 9801} :=
by sorry

end kabulek_four_digits_l337_337218


namespace rectangle_area_kd2_l337_337961

theorem rectangle_area_kd2 (x d : ℝ) (h_ratio : 5 * x / (2 * x) = 5 / 2) (h_diag : d^2 = (5 * x)^2 + (2 * x)^2) :
  ∃ (k : ℝ), (5 * x) * (2 * x) = k * d^2 ∧ k = 10 / 29 :=
by
  have h_length_width : 5 * x = 5 * x :=
    by
      sorry
  have h_width_length : 2 * x = 2 * x :=
    by
      sorry
  have h_area : (5 * x) * (2 * x) = 10 * x ^ 2 :=
    by
      sorry
  have h_pyth : d^2 = 25 * x^2 + 4 * x^2 :=
    by
      sorry 
  rcases (eq_of_sq_eq_sq h_diag h_pyth) with ⟨k, hk⟩,
  use k,
  split,
  {
    rw hk,
    exact h_area,
  },
  {
    exact h_diag,
   }

end rectangle_area_kd2_l337_337961


namespace line_through_point_parallel_l337_337566

open Real

theorem line_through_point_parallel
  (A : Point := (2, 1))
  (l₁ l₂ : Line) 
  (h₁ : passes_through l₁ A)
  (h₂ : is_parallel l₁ l₂)
  (h₃ : l₂ = { a := 2, b := -3, c := 1 }) :
  l₁ = { a := 2, b := -3, c := -1 } :=
sorry

end line_through_point_parallel_l337_337566


namespace exists_k_for_abs_diff_le_one_l337_337748

theorem exists_k_for_abs_diff_le_one 
  (n : ℕ) (x : ℕ → ℝ) 
  (h_n : n > 2)
  (h_sum_abs_gt_one : |∑ i in finset.range n, x i| > 1)
  (h_abs_le_one : ∀ i, i < n → |x i| ≤ 1) :
  ∃ k : ℕ, 1 ≤ k ∧ k < n ∧ |∑ i in finset.range k, x i - ∑ i in finset.Ico k n, x i| ≤ 1 := 
sorry

end exists_k_for_abs_diff_le_one_l337_337748


namespace function_a_increasing_function_c_increasing_function_b_not_increasing_function_d_not_increasing_l337_337280

-- Definitions of the functions
def f_a (x : ℝ) := x^3
def f_b (x : ℝ) := x^2
def f_c (x : ℝ) := x^(1/2)
def f_d (x : ℝ) := -x^(-1)

-- Definitions of increasing on a domain
def is_increasing_on (f : ℝ → ℝ) (s : set ℝ) := ∀ ⦃x y⦄, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

-- Domains of the functions
def domain_a : set ℝ := set.univ
def domain_b : set ℝ := set.univ
def domain_c : set ℝ := { x | 0 ≤ x }
def domain_d : set ℝ := { x | x ≠ 0 }

-- Proof statements
theorem function_a_increasing : is_increasing_on f_a domain_a := sorry
theorem function_c_increasing : is_increasing_on f_c domain_c := sorry

-- Non-increasing functions
theorem function_b_not_increasing : ¬ is_increasing_on f_b domain_b := sorry
theorem function_d_not_increasing : ¬ is_increasing_on f_d domain_d := sorry

end function_a_increasing_function_c_increasing_function_b_not_increasing_function_d_not_increasing_l337_337280


namespace min_chord_length_l337_337014

open Real 

def circle_center : ℝ × ℝ := (2, -3)
def circle_radius : ℝ := 1
def line (x : ℝ) : ℝ := 2 * x + 3

theorem min_chord_length :
  ∃ (P : ℝ × ℝ) (T : ℝ × ℝ), 
    (fst P ≃ fst T) ∧ 
    (snd P ≃ line (fst P)) ∧
    dist P circle_center = 2 * sqrt 5 ∧ 
    dist T circle_center = 1 →
    dist P T = sqrt 19 :=
sorry

end min_chord_length_l337_337014


namespace volunteer_arrangements_l337_337001

noncomputable def arrangements (volunteers : ℕ) (exits : ℕ) : ℕ :=
  if h : volunteers ≥ exits then
    4 ^ (volunteers - exits) * nat.choose (volunteers - 1) (exits - 1)
  else 0

theorem volunteer_arrangements :
  arrangements 5 4 = 240 :=
by
  sorry

end volunteer_arrangements_l337_337001


namespace range_of_m_l337_337132

theorem range_of_m (m : ℝ) :
  (∀ (λ μ : ℝ), ∃ c : ℝ × ℝ, c = (λ * m + μ, λ * (3 * m - 4) + 2 * μ)) ↔ (m ≠ 4) := by
  sorry

end range_of_m_l337_337132


namespace correct_calculation_is_c_l337_337990

theorem correct_calculation_is_c (a b : ℕ) :
  (2 * a ^ 2 * b) ^ 3 = 8 * a ^ 6 * b ^ 3 := 
sorry

end correct_calculation_is_c_l337_337990


namespace possible_values_reciprocals_l337_337511

variable {x y : ℝ}
variables (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 2)

theorem possible_values_reciprocals : 
  ∃ S : set ℝ, S = {z : ℝ | z ≥ 2} ∧ (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 2 ∧ z = (1/x + 1/y)) :=
sorry

end possible_values_reciprocals_l337_337511


namespace intersection_of_A_and_B_l337_337070

-- Define set A
def A : Set ℤ := {-1, 0, 1, 2, 3, 4, 5}

-- Define set B
def B : Set ℤ := {2, 4, 6, 8}

-- Prove that the intersection of set A and set B is {2, 4}.
theorem intersection_of_A_and_B : A ∩ B = {2, 4} :=
by
  sorry

end intersection_of_A_and_B_l337_337070


namespace solution_inequality_l337_337182

noncomputable def f : ℝ → ℝ :=
λ x, if x >= 0 then x^2 - 4*x + 6 else x + 6

theorem solution_inequality (x : ℝ) : 
  (f x > f 1) ↔ (-3 < x ∧ x < 1) ∨ (x > 3) := by
  sorry

end solution_inequality_l337_337182


namespace smallest_z_is_14_l337_337471

-- Define the consecutive even integers and the equation.
def w (k : ℕ) := 2 * k
def x (k : ℕ) := 2 * k + 2
def y (k : ℕ) := 2 * k + 4
def z (k : ℕ) := 2 * k + 6

theorem smallest_z_is_14 : ∃ k : ℕ, z k = 14 ∧ w k ^ 3 + x k ^ 3 + y k ^ 3 = z k ^ 3 :=
by sorry

end smallest_z_is_14_l337_337471


namespace greatest_visible_unit_cubes_from_one_point_12_l337_337631

def num_unit_cubes (n : ℕ) : ℕ := n * n * n

def face_count (n : ℕ) : ℕ := n * n

def edge_count (n : ℕ) : ℕ := n

def visible_unit_cubes_from_one_point (n : ℕ) : ℕ :=
  let faces := 3 * face_count n
  let edges := 3 * (edge_count n - 1)
  let corner := 1
  faces - edges + corner

theorem greatest_visible_unit_cubes_from_one_point_12 :
  visible_unit_cubes_from_one_point 12 = 400 :=
  by
  sorry

end greatest_visible_unit_cubes_from_one_point_12_l337_337631


namespace fraction_tabs_closed_l337_337147

theorem fraction_tabs_closed (x : ℝ) (h₁ : 400 * (1 - x) * (3/5) * (1/2) = 90) : 
  x = 1 / 4 :=
by
  have := h₁
  sorry

end fraction_tabs_closed_l337_337147


namespace inequality_tangents_l337_337428

def f (x : ℝ) (a b : ℝ) : ℝ := x^3 - a * x - b

theorem inequality_tangents (a b : ℝ) (h1 : 0 < a)
  (h2 : ∃ x0 : ℝ, 2 * x0^3 - 3 * a * x0^2 + a^2 + 2 * b = 0): 
  -a^2 / 2 < b ∧ b < f a a b :=
by
  sorry

end inequality_tangents_l337_337428


namespace range_of_m_l337_337740

open Real

variable {m : ℝ}

def p : Prop := ∃ a b : Real, a ≠ b ∧ (a^2 + m*a + 1 = 0) ∧ (b^2 + m*b + 1 = 0)

def q : Prop := ∀ x : Real, 4*x^2 + 4*(m-2)*x + 1 > 0

theorem range_of_m (h : p ∧ ¬q) : m ∈ Iio (-2) ∪ Ici 3 := sorry

end range_of_m_l337_337740


namespace solve_system_l337_337251

theorem solve_system (x y : ℝ) (h1 : 2 * x - y = 0) (h2 : x + 2 * y = 1) : 
  x = 1 / 5 ∧ y = 2 / 5 :=
by
  sorry

end solve_system_l337_337251


namespace find_f1_value_l337_337419

variable {ℝ : Type*} [LinearOrderedField ℝ]

def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ {a b}, 0 < a → 0 < b → a < b → f(a) ≤ f(b)

theorem find_f1_value (f : ℝ → ℝ) (h_mono : is_monotonically_increasing f)
  (h_fun : ∀ x : ℝ, 0 < x → f(f(x) + 2 / x) = -1) : f(1) = -1 :=
sorry

end find_f1_value_l337_337419


namespace impossible_odd_n_m_l337_337454

theorem impossible_odd_n_m (n m : ℤ) (h : Even (n^2 + m + n * m)) : ¬ (Odd n ∧ Odd m) :=
by
  intro h1
  sorry

end impossible_odd_n_m_l337_337454


namespace tori_current_height_l337_337974

theorem tori_current_height :
  let original_height := 4.4
  let growth := 2.86
  original_height + growth = 7.26 := 
by
  sorry

end tori_current_height_l337_337974


namespace hyperbola_eccentricity_is_three_over_two_l337_337434

noncomputable def hyperbola_eccentricity (a b : ℝ) (h0 : a > 0) (h1 : b > 0)
  (h2 : ∃ (F1 F2 P : ℝ×ℝ), |F1.1 - F2.1| = 12 ∧ sqrt ((F2.1 - P.1)^2 + (F2.2 - P.2)^2) = 5 ∧ F1 ≠ F2 ∧ P ≠ F2 ∧ P.2 ≠ F2.2 ∧ F1.2 = 0 ∧ F2.2 = 0)
  : ℝ :=
(|12| / |5| : ℝ)

-- Proposition: The eccentricity of the given hyperbola is 3/2
theorem hyperbola_eccentricity_is_three_over_two 
  (a b : ℝ) (h0 : a > 0) (h1 : b > 0)
  (h2 : ∃ (F1 F2 P : ℝ×ℝ), |F1.1 - F2.1| = 12 ∧ sqrt ((F2.1 - P.1)^2 + (F2.2 - P.2)^2) = 5 ∧ F1 ≠ F2 ∧ P ≠ F2 ∧ P.2 ≠ F2.2 ∧ F1.2 = 0 ∧ F2.2 = 0) :
  hyperbola_eccentricity a b h0 h1 h2 = 3 / 2 :=
sorry

end hyperbola_eccentricity_is_three_over_two_l337_337434


namespace product_multiple_of_3_probability_l337_337589

theorem product_multiple_of_3_probability :
  let s : Finset ℕ := {1, 2, 3, 4, 5, 6}
  let total_choices := (Finset.card s).choose 3
  let non_multiples_of_3 := {1, 2, 4, 5}
  let non_multiples_choices := (Finset.card non_multiples_of_3).choose 3
  (1 - non_multiples_choices / total_choices) = 4 / 5 :=
by
  let s : Finset ℕ := {1, 2, 3, 4, 5, 6}
  let total_choices := (Finset.card s).choose 3
  let non_multiples_of_3 := {1, 2, 4, 5}
  let non_multiples_choices := (Finset.card non_multiples_of_3).choose 3
  have h : (1 - non_multiples_choices / total_choices) = 4 / 5 := sorry
  exact h

end product_multiple_of_3_probability_l337_337589


namespace monotonicity_of_f_range_of_a_l337_337761

noncomputable def f (x a : ℝ) : ℝ :=
  Real.exp x * (Real.exp x - a) - a^2 * x

theorem monotonicity_of_f (a : ℝ) :
  (a = 0 → ∀ x y, x < y → f x a < f y a) ∧
  (a > 0 → ∀ x, x < Real.log a → f' x a < 0 ∧ f' x a > 0 → f x a < f (Real.log a) a ∧ 
                    x > Real.log a → f' x a > 0) ∧
  (a < 0 → ∀ x, x < Real.log (-a / 2) → f' x a < 0 ∧ f' x a > 0 → f x a < f (Real.log (-a / 2)) a ∧ 
                    x > Real.log (-a / 2) → f' x a > 0) := 
sorry

theorem range_of_a (a : ℝ) :
  (∀ x, f x a ≥ 0) ↔ a ∈ Set.Icc (-2 * Real.exp (3 / 4)) 1 := 
sorry

end monotonicity_of_f_range_of_a_l337_337761


namespace difference_mean_median_l337_337119

-- Define the percentage of students scoring each score
def perc_75 : ℝ := 0.05
def perc_85 : ℝ := 0.35
def perc_90 : ℝ := 0.25
def perc_100 : ℝ := 0.10
def perc_110 : ℝ := 1.0 - (perc_75 + perc_85 + perc_90 + perc_100)

-- Define the number of students as a simple number (e.g., 40 for simplicity)
def total_students : ℕ := 40

-- Calculate the number of students scoring each score
def num_75 : ℕ := (perc_75 * total_students).toNat
def num_85 : ℕ := (perc_85 * total_students).toNat
def num_90 : ℕ := (perc_90 * total_students).toNat
def num_100 : ℕ := (perc_100 * total_students).toNat
def num_110 : ℕ := (perc_110 * total_students).toNat

-- Define the scores
def scores : List (ℕ × ℕ) := [(75, num_75), (85, num_85), (90, num_90), (100, num_100), (110, num_110)]

-- Statement to prove
theorem difference_mean_median : 
  let total_score := scores.foldl (λ acc (s, n) => acc + s * n) 0
  let mean := total_score / total_students
  let median := 90 -- as determined by the problem solution steps 
  (abs (mean - median : ℝ)) = 3.5 := by
  sorry

end difference_mean_median_l337_337119


namespace max_lines_proof_l337_337540

-- Definitions and conditions
def distinct_lines (L : list (affine.line ℝ)) : Prop :=
  ∀ i j, i < j → L.nth i ≠ L.nth j

def all_intersect (L : list (affine.line ℝ)) : Prop :=
  ∀ i j, i < j → ∃ p, affine.line.contains (L.nth i) p ∧ affine.line.contains (L.nth j) p

def angle_60_among_any_15 (L : list (affine.line ℝ)) : Prop :=
  ∀ S, S ⊆ L ∧ S.length = 15 →
       ∃ i j, i < j ∧ ∃ θ, affine.line.angle (S.nth i) (S.nth j) = real.pi / 3

-- Lean 4 statement to prove
theorem max_lines_proof (N : ℕ) (L : list (affine.line ℝ)) :
  distinct_lines L →
  all_intersect L →
  angle_60_among_any_15 L →
  L.length ≤ 42 :=
sorry

end max_lines_proof_l337_337540


namespace max_distance_from_center_to_line_max_distance_from_center_to_line_exact_l337_337033

/-- Define the distance from a point (x0, y0) to the line Ax + By + C = 0 -/
def distance_from_point_to_line (x0 y0 A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / (sqrt (A^2 + B^2))

/-- Define the conditions for the problem -/
def circle_with_point (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 3)^2 = 1

/-- Define the question as a theorem statement -/
theorem max_distance_from_center_to_line : ∀ (x y : ℝ),
  circle_with_point x y → 
  distance_from_point_to_line x y 3 (-4) (-4) ≤ 3 :=
by
  sorry

theorem max_distance_from_center_to_line_exact : 
  ∃ (x y : ℝ), circle_with_point x y ∧ distance_from_point_to_line x y 3 (-4) (-4) = 3 :=
by
  sorry

end max_distance_from_center_to_line_max_distance_from_center_to_line_exact_l337_337033


namespace ball_distribution_l337_337452

theorem ball_distribution :
  let n := 6 in
  let b := 3 in
  let ways := (choose 6 6) + (choose 6 5) + (choose 6 4 * choose 2 2) + (choose 6 3 * choose 3 2 * choose 1 1) in
  ways = 82 :=
by
  sorry

end ball_distribution_l337_337452


namespace range_of_a_l337_337760

theorem range_of_a (a : ℝ) :
  (∃ x, 0 < x ∧ x < 1 ∧ (a^2 * x - 2 * a + 1 = 0)) ↔ (a > 1/2 ∧ a ≠ 1) :=
by
  sorry

end range_of_a_l337_337760


namespace F_is_even_l337_337736

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

noncomputable def F (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  (x^3 - 2*x) * f x

theorem F_is_even (f : ℝ → ℝ) (h_odd : is_odd_function f) (h_nonzero : f 1 ≠ 0) :
  is_even_function (F f) :=
sorry

end F_is_even_l337_337736


namespace cube_root_less_than_five_count_l337_337085

theorem cube_root_less_than_five_count :
  (∃ n : ℕ, n = 124 ∧ ∀ x : ℕ, 1 ≤ x → x < 5^3 → x < 125) := 
sorry

end cube_root_less_than_five_count_l337_337085


namespace no_positive_integer_satisfies_inequality_l337_337451

theorem no_positive_integer_satisfies_inequality :
  ∀ x : ℕ, 0 < x → ¬ (15 < -3 * (x : ℤ) + 18) := by
  sorry

end no_positive_integer_satisfies_inequality_l337_337451


namespace product_of_differences_divisible_by_6_to_7_l337_337577

theorem product_of_differences_divisible_by_6_to_7 (a : Fin 8 → ℤ) (h : (∑ i, a i) = 28) :
  ∃ k ≥ 7, (∏ i j in Finset.univ.filter (λ p, p.1 < p.2), (a i - a j)) % (6 ^ k) = 0 := 
sorry

end product_of_differences_divisible_by_6_to_7_l337_337577


namespace max_pairs_1607_l337_337024

noncomputable def max_possible_pairs : ℕ :=
  let S (k : ℕ) : ℕ := k * (2 * k + 1)
  let T (k : ℕ) : ℕ := (8039 - k) * k / 2
  if h : 2 * (S 1607) ≤ T 1607 then 1607 else sorry

theorem max_pairs_1607 :
  ∃ k : ℕ, k ≤ 1607 ∧ 
  (∀ a b : ℕ, a < b ∧ a + b ≤ 4019 → ∃ i j : ℕ, a = i ∧ b = j ∧ i < j) ∧ 
  (∀ i j : ℕ, i ≠ j → (∀ b a, a < b ∧ a + b ≤ 4019 → (a_i + b_i) ≠ (a_j + b_j)))
:=
sorry

end max_pairs_1607_l337_337024


namespace cats_on_ship_l337_337289

theorem cats_on_ship :
  ∃ (C S : ℕ), 
  (C + S + 1 + 1 = 16) ∧
  (4 * C + 2 * S + 2 * 1 + 1 * 1 = 41) ∧ 
  C = 5 :=
by
  sorry

end cats_on_ship_l337_337289


namespace pascal_triangle_contains_53_once_l337_337829

theorem pascal_triangle_contains_53_once
  (h_prime : Nat.Prime 53) :
  ∃! n, ∃ k, n ≥ k ∧ n > 0 ∧ k > 0 ∧ Nat.choose n k = 53 := by
  sorry

end pascal_triangle_contains_53_once_l337_337829


namespace f_at_2015_l337_337523

noncomputable def f : ℤ → ℤ := sorry

axiom f_odd : ∀ x : ℤ, f(-x) = -f(x)
axiom f_period : ∀ x : ℤ, f(x + 3) = f(x)
axiom f_at_1 : f(1) = -1

theorem f_at_2015 : f(2015) = 1 := by
  sorry

end f_at_2015_l337_337523


namespace pascal_contains_53_l337_337794

theorem pascal_contains_53 (n : ℕ) (h1 : Nat.Prime 53) (h2 : ∃ k, 1 ≤ k ∧ k ≤ 52 ∧ nat.choose 53 k = 53) (h3 : ∀ m < 53, ¬ (∃ k, 1 ≤ k ∧ k ≤ m - 1 ∧ nat.choose m k = 53)) (h4 : ∀ m > 53, ¬ (∃ k, 1 ≤ k ∧ k ≤ m - 1 ∧ nat.choose m k = 53)) : 
  (n = 53) → (n = 1) := 
by
  intros
  sorry

end pascal_contains_53_l337_337794


namespace quadrilateral_area_is_171_l337_337668

-- Define the problem statement
def quadrilateral_area (a b c d : ℝ) (PE PF PM PN: ℝ) : ℝ :=
  if (PE = PF) ∧ (PF = PM) ∧ (PM = PN) ∧ (PE = 6) ∧ (a + b + c + d = 57)
  then 0.5 * 6 * (a + b + c + d)
  else 0

-- The theorem to be proven
theorem quadrilateral_area_is_171 {a b c d : ℝ} :
  quadrilateral_area a b c d 6 6 6 6 = 171 :=
by {
  sorry -- Proof needed
}

end quadrilateral_area_is_171_l337_337668


namespace iris_to_tulip_ratio_l337_337480

theorem iris_to_tulip_ratio (earnings_per_bulb : ℚ)
  (tulip_bulbs daffodil_bulbs crocus_ratio total_earnings : ℕ)
  (iris_bulbs : ℕ) (h0 : earnings_per_bulb = 0.50)
  (h1 : tulip_bulbs = 20) (h2 : daffodil_bulbs = 30)
  (h3 : crocus_ratio = 3) (h4 : total_earnings = 75)
  (h5 : total_earnings = earnings_per_bulb * (tulip_bulbs + iris_bulbs + daffodil_bulbs + crocus_ratio * daffodil_bulbs))
  : iris_bulbs = 10 → tulip_bulbs = 20 → (iris_bulbs : ℚ) / (tulip_bulbs : ℚ) = 1 / 2 :=
by {
  intros; sorry
}

end iris_to_tulip_ratio_l337_337480


namespace sqrt_360000_eq_600_l337_337925

theorem sqrt_360000_eq_600 : Real.sqrt 360000 = 600 := 
sorry

end sqrt_360000_eq_600_l337_337925


namespace hexagon_area_half_triangle_area_l337_337123

-- Given an acute-angled triangle ABC and perpendiculars are drawn from the midpoint of each side to the two other sides
variable (A B C : Point)
variable (h_acute : AcuteTriangle A B C)
variable (M_AB : Line)
variable (M_BC : Line)
variable (M_CA : Line)
variable (M1 : Midpoint A B)
variable (M2 : Midpoint B C)
variable (M3 : Midpoint C A)
variable (h1 : Perpendicular M1 M2)
variable (h2 : Perpendicular M2 M3)
variable (h3 : Perpendicular M3 M1)

-- Prove that the area of the hexagon enclosed by these perpendiculars is equal to half the area of the triangle
theorem hexagon_area_half_triangle_area :
    ∀ (hex_area : ℝ) (tri_area : ℝ), 
    hex_area = 1 / 2 * tri_area 
    --> ∃ (M1 M2 M3 : Point), 
    midpoint M1 A B ∧ midpoint M2 B C ∧ midpoint M3 C A ∧
    perpendicular M1 A B ∧ perpendicular M2 B C ∧ perpendicular M3 C A := 
by
  sorry

end hexagon_area_half_triangle_area_l337_337123


namespace window_savings_l337_337651

theorem window_savings (price_per_window : ℕ) (offer : ℕ → ℕ) (dave_wants : ℕ) (doug_wants : ℕ) :
  price_per_window = 100 →
  (∀ n, offer n = n - n / 3) →
  dave_wants = 10 →
  doug_wants = 12 →
  let total_cost_separate := (dave_wants - dave_wants / 3) * price_per_window + 
                             (doug_wants - doug_wants / 3) * price_per_window in
  let total_cost_joint := (dave_wants + doug_wants - (dave_wants + doug_wants) / 3) * price_per_window in
  total_cost_separate - total_cost_joint = 0 :=
by
  intros
  let dave_cost := (dave_wants - dave_wants / 3) * price_per_window
  let doug_cost := (doug_wants - doug_wants / 3) * price_per_window
  let total_cost_separate := dave_cost + doug_cost
  let total_cost_joint := (dave_wants + doug_wants - (dave_wants + doug_wants) / 3) * price_per_window
  have dave_correct_cost : dave_cost = 700 := by sorry
  have doug_correct_cost : doug_cost = 800 := by sorry
  have separate_correct_cost : total_cost_separate = 1500 := by sorry
  have joint_correct_cost : total_cost_joint = 1500 := by sorry
  show total_cost_separate - total_cost_joint = 0 from by
    rw [separate_correct_cost, joint_correct_cost]
    exact Nat.sub_self 1500

end window_savings_l337_337651


namespace inequality_proof_l337_337919

variable (a b c : ℝ)

theorem inequality_proof :
  1 < (a / Real.sqrt (a^2 + b^2) + b / Real.sqrt (b^2 + c^2) + c / Real.sqrt (c^2 + a^2)) ∧
  (a / Real.sqrt (a^2 + b^2) + b / Real.sqrt (b^2 + c^2) + c / Real.sqrt (c^2 + a^2)) ≤ 3 * Real.sqrt 2 / 2 :=
sorry

end inequality_proof_l337_337919


namespace problem_statement_l337_337757

theorem problem_statement 
  (a b c : ℤ)
  (h1 : (5 * a + 2) ^ (1/3) = 3)
  (h2 : (3 * a + b - 1) ^ (1/2) = 4)
  (h3 : c = Int.floor (Real.sqrt 13))
  : a = 5 ∧ b = 2 ∧ c = 3 ∧ Real.sqrt (3 * a - b + c) = 4 := 
by 
  sorry

end problem_statement_l337_337757


namespace rectangle_dimension_correct_l337_337957

-- Definition of the Width and Length based on given conditions
def width := 3 / 2
def length := 3

-- Perimeter and Area conditions
def perimeter_condition (w l : ℝ) := 2 * (w + l) = 2 * (w * l)
def length_condition (w l : ℝ) := l = 2 * w

-- Main theorem statement
theorem rectangle_dimension_correct :
  ∃ (w l : ℝ), perimeter_condition w l ∧ length_condition w l ∧ w = width ∧ l = length :=
by {
  -- add sorry to skip the proof
  sorry
}

end rectangle_dimension_correct_l337_337957


namespace number_of_three_leaf_clovers_l337_337496

theorem number_of_three_leaf_clovers (total_leaves : ℕ) (three_leaf_clover : ℕ) (four_leaf_clover : ℕ) (n : ℕ)
  (h1 : total_leaves = 40) (h2 : three_leaf_clover = 3) (h3 : four_leaf_clover = 4) (h4: total_leaves = 3 * n + 4) :
  n = 12 :=
by
  sorry

end number_of_three_leaf_clovers_l337_337496


namespace min_perimeter_triangle_l337_337850

theorem min_perimeter_triangle (
  (D E F : ℝ) 
  (d e f : ℕ)
  (h_cosD : real.cos D = 8 / 17)
  (h_cosE : real.cos E = 15 / 17)
  (h_cosF : real.cos F = -5 / 13)
  (h_angle_sum : D + E + F = real.pi)
  (h_triangle_ineq1 : d + e > f)
  (h_triangle_ineq2 : d + f > e)
  (h_triangle_ineq3 : e + f > d)
  (h_law_of_sines : d / real.sin D = e / real.sin E)
  (h_law_of_sines' : d / real.sin D = f / real.sin F)
  (h_nonneg_sides : 0 < d ∧ 0 < e ∧ 0 < f)
) : d + e + f = 503 :=
sorry

end min_perimeter_triangle_l337_337850


namespace real_roots_of_equation_l337_337578

noncomputable def greatest_integer (x : ℝ) : ℤ := floor x

theorem real_roots_of_equation (x : ℝ) (h : x^2 - 8 * (greatest_integer x) + 7 = 0) : 
  x = 1 ∨ x = Real.sqrt 33 ∨ x = Real.sqrt 41 ∨ x = 7 :=
by
  sorry

end real_roots_of_equation_l337_337578


namespace find_modulus_of_z1_at_neg2_find_a_when_z1_bar_plus_z2_is_real_l337_337520

def z1 (a : ℝ) : Complex := Complex.mk (a + 5) (10 - a^2)
def z2 (a : ℝ) : Complex := Complex.mk (1 - 2 * a) (2 * a - 5)

theorem find_modulus_of_z1_at_neg2 :
  Complex.abs (z1 (-2)) = 3 * Real.sqrt 5 := by
  sorry

theorem find_a_when_z1_bar_plus_z2_is_real :
  ∃ (a : ℝ), (Complex.re ((Complex.conj (z1 a)) + (z2 a)) = (Complex.conj (z1 a) + z2 a) ∧ 
                 Complex.im ((Complex.conj (z1 a)) + (z2 a)) = 0) :=
  (a = -5 ∨ a = 3) := by
  sorry

end find_modulus_of_z1_at_neg2_find_a_when_z1_bar_plus_z2_is_real_l337_337520


namespace range_of_m_l337_337738

def f (x : ℝ) : ℝ := Real.log (x^2 + 1)
def g (x m : ℝ) : ℝ := (1/2)^x - m

theorem range_of_m (m : ℝ) : 
  (∀ x1 ∈ Icc 0 3, ∃ x2 ∈ Icc 1 2, f x1 ≥ g x2 m) ↔ m ≥ 1/4 :=
sorry

end range_of_m_l337_337738


namespace part1_part2_l337_337181

noncomputable def z (θ : ℝ) := -3 * Real.cos θ + Complex.i * Real.sin θ
noncomputable def z1 (θ : ℝ) := Real.cos θ - Complex.i * Real.sin θ

theorem part1 : z (4 * Real.pi / 3).abs = Real.sqrt 3 := 
by 
  sorry

theorem part2 (θ : ℝ) (hθ : θ ∈ Set.Icc (Real.pi / 2) Real.pi) : 
  z1 θ * z θ ∈ {i * x | x : ℝ} → θ = 2 * Real.pi / 3 :=
by 
  sorry

end part1_part2_l337_337181


namespace find_g_5_l337_337226

noncomputable def g : ℝ → ℝ := sorry

axiom g_property (x y : ℝ) : g (x - y) = g x * g y
axiom g_nonzero (x : ℝ) : g x ≠ 0

theorem find_g_5 : g 5 = 1 :=
by
  sorry

end find_g_5_l337_337226


namespace find_angle_four_l337_337652

theorem find_angle_four (angle1 angle2 angle3 angle4 : ℝ)
  (h1 : angle1 + angle2 = 180)
  (h2 : angle1 + angle3 + 60 = 180)
  (h3 : angle3 = angle4) :
  angle4 = 60 :=
by sorry

end find_angle_four_l337_337652


namespace area_of_quadrilateral_ABCD_l337_337869

-- Define points and assumptions
variables (A B C D E : Type) [MetricSpace E]

-- Define the given conditions
axiom right_angle_AEB : ∀ x y z : E, (∠AEB = 90)
axiom right_angle_BEC : ∀ x y z : E, (∠BEC = 90)
axiom right_angle_CDE : ∀ x y z : E, (∠CDE = 90)
axiom angle_AEB_45 : ∀ x y z : E, (∠AEB = 45)
axiom angle_BEC_45 : ∀ x y z : E, (∠BEC = 45)
axiom angle_CED_45 : ∀ x y z : E, (∠CED = 45)
axiom AE_length : ∀ x y z : E, (AE = 18)

-- Define the problem statement and conclude the proof
theorem area_of_quadrilateral_ABCD : ∀ (A B C D E : E) [MetricSpace E], 
  (right_angle_AEB ∧ right_angle_BEC ∧ right_angle_CDE ∧ angle_AEB_45 ∧ angle_BEC_45 ∧ angle_CED_45 ∧ AE_length) →
  area_ABCD = 202.5 :=
by sorry

end area_of_quadrilateral_ABCD_l337_337869


namespace riley_pawns_lost_l337_337489

theorem riley_pawns_lost (initial_pawns : ℕ) (kennedy_lost : ℕ) (total_pawns_left : ℕ)
  (kennedy_initial_pawns : ℕ) (riley_initial_pawns : ℕ) : 
  kennedy_initial_pawns = initial_pawns ∧
  riley_initial_pawns = initial_pawns ∧
  kennedy_lost = 4 ∧
  total_pawns_left = 11 →
  riley_initial_pawns - (total_pawns_left - (kennedy_initial_pawns - kennedy_lost)) = 1 :=
by
  sorry

end riley_pawns_lost_l337_337489


namespace ellipse_hyperbola_eccentricity_range_l337_337044

theorem ellipse_hyperbola_eccentricity_range :
  let F1 := (0, 0), F2 := (0, 0), P := (0, 0)
  let m := 10
  let c : ℝ := 0
  let a1 := 5 + c
  let a2 := 5 - c
  let e1 := c / a1
  let e2 := c / a2
  \text{
    (1 < 25 / c^2 < 4) implies (1 / (25 / c^2 - 1) > 1 / 3)
  } :=
sorry

end ellipse_hyperbola_eccentricity_range_l337_337044


namespace math_problem_l337_337688

theorem math_problem : (-1: ℝ)^2 + (1/3: ℝ)^0 = 2 := by
  sorry

end math_problem_l337_337688


namespace problem_solution_l337_337693

noncomputable def solve_problem (c : ℝ) : Prop :=
  ∃ x y : ℝ, sqrt (x * y) = c^c ∧ log c (x^log c y) + log c (y^log c x) = 5 * c^5

theorem problem_solution (c : ℝ) : solve_problem c ↔ c ∈ set.Ioc 0 ((2 / 5)^(1 / 3)) := sorry

end problem_solution_l337_337693


namespace smallest_n_geq_10_l337_337382

theorem smallest_n_geq_10 (n : ℕ) (h : n ≥ 10) :
  (∑ i in range (n + 1), i^2) * (∑ i in range (2*n + 1, 3*n + 1), i^2) = m^2 → n = 71 :=
by sorry

end smallest_n_geq_10_l337_337382


namespace pascal_triangle_contains_53_l337_337806

theorem pascal_triangle_contains_53:
  ∃! n, ∃ k, (n ≥ 0) ∧ (k ≥ 0) ∧ (binom n k = 53) := 
sorry

end pascal_triangle_contains_53_l337_337806


namespace base7_product_digit_sum_l337_337959

noncomputable def base7_to_base10 (n : Nat) : Nat :=
  match n with
  | 350 => 3 * 7 + 5
  | 217 => 2 * 7 + 1
  | _ => 0

noncomputable def base10_to_base7 (n : Nat) : Nat := 
  if n = 390 then 1065 else 0

noncomputable def digit_sum_in_base7 (n : Nat) : Nat :=
  if n = 1065 then 1 + 0 + 6 + 5 else 0

noncomputable def sum_to_base7 (n : Nat) : Nat :=
  if n = 12 then 15 else 0

theorem base7_product_digit_sum :
  digit_sum_in_base7 (base10_to_base7 (base7_to_base10 350 * base7_to_base10 217)) = 15 :=
by
  sorry

end base7_product_digit_sum_l337_337959


namespace find_r_l337_337179

theorem find_r (r : ℝ) :
  (f : ℝ → ℝ) = (λ x, 3*x^4 + 2*x^3 - x^2 - 4*x + r) →
  f 3 = 0 ↔ r = -276 :=
by sorry

end find_r_l337_337179


namespace min_queries_to_determine_parity_l337_337117

-- Define the set of bags and the querying method
noncomputable def bags : Finset ℕ := (Finset.range 100).map ⟨(λ n, n + 1), fun _ _ => by simp⟩

-- Define the querying function
def query (S : Finset ℕ) (p : ℕ → bool) : bool := Finset.card (S.filter (λ x => p x))

-- Define the main theorem
theorem min_queries_to_determine_parity : (∀ (p : ℕ → bool), ∃ (S : Finset (Finset ℕ)), 
  (∀s ∈ S, s.card = 15) ∧ 
  (S.card < 3 → (p 1 = (query s p for some s in S))) = false)
  :=
sorry

end min_queries_to_determine_parity_l337_337117


namespace lines_perpendicular_l337_337550

-- Define the conditions: lines not parallel to the coordinate planes 
-- (which translates to k_1 and k_2 not being infinite, but we can code it directly as a statement on the product being -1)
variable {k1 k2 l1 l2 : ℝ} 

-- Define the theorem statement 
theorem lines_perpendicular (hk : k1 * k2 = -1) : 
  ∀ (x : ℝ), (k1 ≠ 0) ∧ (k2 ≠ 0) → 
  (∀ (y1 y2 : ℝ), y1 = k1 * x + l1 → y2 = k2 * x + l2 → 
  (k1 * k2 = -1)) :=
sorry

end lines_perpendicular_l337_337550


namespace correct_inequalities_count_l337_337409

theorem correct_inequalities_count
    (a b : ℝ)
    (h : 1 / a < 1 / b ∧ 1 / b < 0) :
    (∃ P : ℝ → ℝ → Prop, P a b ∧ (P = λ a b, a + b < a * b) ∨
     P = λ a b, |a| > |b| ∨
     P = λ a b, a < b ∨
     P = λ a b, b / a + a / b > 2) →
    ∃ f : ℝ → ℝ → ℕ, f a b = 2 := by
  sorry

end correct_inequalities_count_l337_337409


namespace find_x_values_l337_337891

noncomputable def g1 (x : ℚ) : ℚ :=
  (3/4 : ℚ) - (4 / (4 * x + 1))

noncomputable def g : ℕ → ℚ → ℚ
| 1 x := g1 x
| (n + 1) x := g1 (g n x)

theorem find_x_values (x : ℚ) : g 1002 x = x - 4 ↔ (x = 543 / 51 ∨ x = 5 / 32) :=
by {
  sorry
}

end find_x_values_l337_337891


namespace set_eq_inter_of_union_eq_and_inter_eq_l337_337396

variables {α : Type*} (A B X : set α)

theorem set_eq_inter_of_union_eq_and_inter_eq 
  (h1 : A ∪ B ∪ X = A ∪ B)
  (h2 : A ∩ X = A ∩ B)
  (h3 : B ∩ X = A ∩ B) :
  X = A ∩ B := 
sorry

end set_eq_inter_of_union_eq_and_inter_eq_l337_337396


namespace altitudes_of_triangle_roots_of_poly_degree_six_with_rational_coefficients_l337_337582

theorem altitudes_of_triangle_roots_of_poly_degree_six_with_rational_coefficients
  (a b c d : ℚ)
  (h_poly : ∀ x, (a * x^3 + b * x^2 + c * x + d = 0) → true)
  (r1 r2 r3 : ℝ)
  (h_roots : Polynomial.roots (Polynomial.C a * Polynomial.X ^ 3 + Polynomial.C b * Polynomial.X ^ 2 + Polynomial.C c * Polynomial.X + Polynomial.C d) = {r1, r2, r3}) :
  ∃ (p : Polynomial ℚ), p.degree = 6 ∧ ∀ (h : ℝ), h = 2 * sqrt (abs (s (s - r1) (s - r2) (s - r3))) / r1 ∨
                                             h = 2 * sqrt (abs (s (s - r1) (s - r2) (s - r3))) / r2 ∨
                                             h = 2 * sqrt (abs (s (s - r1) (s - r2) (s - r3))) / r3 → 
                                             p.eval h = 0 :=
begin
    sorry -- Proof to be filled in
end

end altitudes_of_triangle_roots_of_poly_degree_six_with_rational_coefficients_l337_337582


namespace megan_total_songs_l337_337530

theorem megan_total_songs : 
  ∀ (initial_albums removed_albums songs_per_album : ℕ),
  initial_albums = 8 → 
  removed_albums = 2 →
  songs_per_album = 7 →
  (initial_albums - removed_albums) * songs_per_album = 42 :=
by
  intros initial_albums removed_albums songs_per_album
  intro h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry

end megan_total_songs_l337_337530


namespace right_triangle_proportion_l337_337500

variables {α : Type*} [linear_ordered_field α]

structure Triangle :=
(A B C H : α)
(angle_A : α)
(foot_H : α)

def is_right_triangle (T : Triangle) : Prop :=
T.angle_A = 90

def is_foot_of_altitude_from_A (T : Triangle) : Prop :=
T.foot_H = T.H

theorem right_triangle_proportion (T : Triangle) 
    (h_right : is_right_triangle T)
    (h_altitude : is_foot_of_altitude_from_A T) :
  T.AH^2 = T.HB * T.HC ∧ 
  T.AB^2 = T.BH * T.BC ∧
  T.AC^2 = T.CH * T.BC := 
sorry

end right_triangle_proportion_l337_337500


namespace days_to_complete_work_l337_337842

theorem days_to_complete_work :
  ∀ (M B: ℝ) (D: ℝ),
    (M = 2 * B)
    → (13 * M + 24 * B) * 4 = (12 * M + 16 * B) * D
    → D = 5 :=
by
  intros M B D h1 h2
  sorry

end days_to_complete_work_l337_337842


namespace pump_without_leak_time_l337_337323

variables (P : ℝ) (effective_rate_with_leak : ℝ) (leak_rate : ℝ)
variable (pump_filling_time : ℝ)

-- Define the conditions
def conditions :=
  effective_rate_with_leak = 3/7 ∧
  leak_rate = 1/14 ∧
  pump_filling_time = P

-- Define the theorem
theorem pump_without_leak_time (h : conditions P effective_rate_with_leak leak_rate pump_filling_time) : 
  P = 2 :=
sorry

end pump_without_leak_time_l337_337323


namespace slope_of_line_l337_337609

noncomputable def slope {x1 y1 x2 y2 : ℝ} (A B : ℝ × ℝ) : ℝ :=
  (B.snd - A.snd) / (B.fst - A.fst)

theorem slope_of_line (A B : ℝ × ℝ)
  (hA : A = (-4, 7)) (hB : B = (3, -4)) : slope A B = -11 / 7 :=
by
  rw [hA, hB]
  simp [slope]
  sorry

end slope_of_line_l337_337609


namespace interest_after_4_years_l337_337661
-- Importing the necessary library

-- Definitions based on the conditions
def initial_amount : ℝ := 1500
def annual_interest_rate : ℝ := 0.12
def number_of_years : ℕ := 4

-- Calculating the total amount after 4 years using compound interest formula
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

-- Calculating the interest earned
def interest_earned : ℝ :=
  compound_interest initial_amount annual_interest_rate number_of_years - initial_amount

-- The Lean statement to prove the interest earned is $859.25
theorem interest_after_4_years : interest_earned = 859.25 :=
by
  sorry

end interest_after_4_years_l337_337661


namespace parametric_equation_proof_minimized_point_proof_l337_337865

-- Define the polar equation condition
def polar_equation (theta : ℝ) : ℝ := 2 * sqrt 2 * sin theta

-- Convert the polar equation to Cartesian coordinates
def cartesian_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2 * sqrt 2 * y = 0

-- Parametric equations for curve C
def parametric_equation (alpha : ℝ) (x y : ℝ) : Prop :=
  x = sqrt 2 * cos alpha ∧ y = sqrt 2 + sqrt 2 * sin alpha

-- Line l in Cartesian coordinates
def line_l (x y : ℝ) : Prop :=
  y = -x + 6

-- Finding point M for minimized distance |MQ|
noncomputable def minimized_distance (x y : ℝ) : Prop :=
  (∃ (alpha : ℝ), parametric_equation alpha x y) ∧ 
  line_l x y ∧ 
  ∀ (alpha : ℝ), 
    let M := (sqrt 2 * cos alpha, sqrt 2 + sqrt 2 * sin alpha) in
    dist M (x, y) = min (dist (sqrt 2 * cos alpha, sqrt 2 + sqrt 2 * sin alpha) (x, y))

theorem parametric_equation_proof : 
  ∀ (α : ℝ), ∃ (x y : ℝ), parametric_equation α x y := 
by
  sorry

theorem minimized_point_proof :
  ∃ (x y : ℝ), minimized_distance x y ∧ x = 1 ∧ y = sqrt 2 + 1 :=
by
  sorry

end parametric_equation_proof_minimized_point_proof_l337_337865


namespace smallest_class_size_l337_337124

-- Let n be the total number of students in the class.
variable (n : ℕ)

-- Define the conditions
def six_students_score_100 : Prop := 
  exists (p : Fin 6 → ℕ), (∀ i, p i = 100) ∧ (∑ i, p i = 600)

def at_least_70_points (s : Fin n → ℕ) : Prop := 
  ∀ i, s i ≥ 70

def average_score (s : Fin n → ℕ) : Prop := 
  (∑ i, s i) / n = 82

-- Define a predicate to match n to 15, which is our correct answer
def smallest_possible_number_of_students (n : ℕ) : Prop := 
  n = 15

-- The main statement combining all conditions to achieve the result
theorem smallest_class_size 
  (s : Fin n → ℕ)
  (H1 : six_students_score_100) 
  (H2 : at_least_70_points s)
  (H3 : average_score s) : 
  smallest_possible_number_of_students n := 
sorry

end smallest_class_size_l337_337124


namespace coins_on_board_l337_337193

theorem coins_on_board:
  let board := (2, 100)
  let coins := 99
  (∀ (cell₁ cell₂ : ℕ × ℕ), (cell₁.1, cell₁.2) ≠ (cell₂.1, cell₂.2) ∧ (abs (cell₁.1 - cell₂.1) + abs (cell₁.2 - cell₂.2) ≠ 1)) →
  fintype.card {configuration : finset (ℕ × ℕ) // finset.card configuration = 99 ∧ ∀ (cell₁ cell₂ ∈ configuration), abs (cell₁.1 - cell₂.1) + abs (cell₂.1 - cell₂.1) ≠ 1} = 396 :=
by
  sorry

end coins_on_board_l337_337193


namespace general_formula_sum_first_10_terms_l337_337770

noncomputable def a : ℕ → ℚ
| 0 => 0
| 1 => 1
| (n+1) =>  (a n) * (n+2)/(n+1)

theorem general_formula (n: ℕ) (h : n ≥ 1) : 
  a n = (n+1) / 2 :=
sorry

theorem sum_first_10_terms : 
  ∑ k in Finset.range 11, (1 / (a k * a (k+1))) = 5 / 3 :=
sorry

end general_formula_sum_first_10_terms_l337_337770


namespace A_lt_B_l337_337281

def A (n : ℕ) : ℕ := Nat.pow 2 (Nat.pow 2 (Nat.pow 2 (Nat.pow 2 n)))
def B (m : ℕ) : ℕ := Nat.pow 3 (Nat.pow 3 (Nat.pow 3 m))

theorem A_lt_B : A 1001 < B 1000 := 
by
  simp [A, B]
  sorry

end A_lt_B_l337_337281


namespace sara_height_correct_l337_337553

variable (Roy_height : ℕ)
variable (Joe_height : ℕ)
variable (Sara_height : ℕ)

def problem_conditions (Roy_height Joe_height Sara_height : ℕ) : Prop :=
  Roy_height = 36 ∧
  Joe_height = Roy_height + 3 ∧
  Sara_height = Joe_height + 6

theorem sara_height_correct (Roy_height Joe_height Sara_height : ℕ) :
  problem_conditions Roy_height Joe_height Sara_height → Sara_height = 45 := by
  sorry

end sara_height_correct_l337_337553


namespace rectangle_sides_l337_337646

theorem rectangle_sides :
  ∀ (x : ℝ), 
    (3 * x = 8) ∧ (8 / 3 * 3 = 8) →
    ((2 * (3 * x + x) = 3 * x^2) ∧ (2 * (3 * (8 / 3) + (8 / 3)) = 3 * (8 / 3)^2) →
    x = 8 / 3
      ∧ 3 * x = 8) := 
by
  sorry

end rectangle_sides_l337_337646


namespace combined_cost_price_l337_337607

theorem combined_cost_price :
  let stock1_price := 100
  let stock1_discount := 5 / 100
  let stock1_brokerage := 1.5 / 100
  let stock2_price := 200
  let stock2_discount := 7 / 100
  let stock2_brokerage := 0.75 / 100
  let stock3_price := 300
  let stock3_discount := 3 / 100
  let stock3_brokerage := 1 / 100

  -- Calculated values
  let stock1_discounted_price := stock1_price * (1 - stock1_discount)
  let stock1_total_price := stock1_discounted_price * (1 + stock1_brokerage)
  
  let stock2_discounted_price := stock2_price * (1 - stock2_discount)
  let stock2_total_price := stock2_discounted_price * (1 + stock2_brokerage)
  
  let stock3_discounted_price := stock3_price * (1 - stock3_discount)
  let stock3_total_price := stock3_discounted_price * (1 + stock3_brokerage)
  
  let combined_cost := stock1_total_price + stock2_total_price + stock3_total_price
  combined_cost = 577.73 := sorry

end combined_cost_price_l337_337607


namespace angle_A_is_ninety_l337_337887

-- Definitions pertaining to the problem.
variables {A B C : Point}
def centroid (A B C : Point) : Point := 
  (A + B + C) / 3

def orthocenter (A B C : Point) : Point :=
  -- Lean code to calculate the orthocenter would go here.
  sorry

def distance (P Q : Point) : Real :=
  -- Lean code to calculate distance between points would go here
  sorry

-- Given conditions
axiom centroid_eq_of_triang ΔG : G = centroid A B C
axiom orthocenter_eq_of_triang ΔH : H = orthocenter A B C
axiom equal_dist : distance A G = distance A H

-- The proof statement for the problem
theorem angle_A_is_ninety (A B C : Point) (G H : Point) : 
  centroid_eq_of_triang ΔG → 
  orthocenter_eq_of_triang ΔH → 
  equal_dist → 
  ∠ A = 90 :=
  by
  sorry

end angle_A_is_ninety_l337_337887


namespace smallest_b_l337_337166

noncomputable def Q (b : ℤ) (x : ℤ) : ℤ := sorry -- Q is a polynomial, will be defined in proof

theorem smallest_b (b : ℤ) 
  (h1 : b > 0) 
  (h2 : ∀ x, x = 2 ∨ x = 4 ∨ x = 6 ∨ x = 8 → Q b x = b) 
  (h3 : ∀ x, x = 1 ∨ x = 3 ∨ x = 5 ∨ x = 7 → Q b x = -b) 
  : b = 315 := sorry

end smallest_b_l337_337166


namespace problem_x_pow_n_exp_le_factorial_l337_337497

theorem problem_x_pow_n_exp_le_factorial (n : ℕ) (x : ℝ) (h1 : 0 ≤ x) (h2 : 0 < n) : x^n * real.exp (1 - x) ≤ nat.factorial n := 
sorry

end problem_x_pow_n_exp_le_factorial_l337_337497


namespace calculate_expression_l337_337683

theorem calculate_expression : (-1:ℝ)^2 + (1/3:ℝ)^0 = 2 := by
  sorry

end calculate_expression_l337_337683


namespace solve_complex_eq_l337_337700

theorem solve_complex_eq (a b : ℝ) (z : ℂ) (h1 : z = a + b * complex.I) 
  (h2 : 3 * z - 4 * complex.I * (complex.conj z) = -5 + 4 * complex.I) :
    z = -1/7 - (8/7) * complex.I :=
by
  sorry

end solve_complex_eq_l337_337700


namespace smallest_sphere_radius_polyhedron_volume_l337_337314

variable (c : ℝ)
axiom convex_polyhedron : Prop
axiom congruent_faces : Prop
axiom edge_relationships : Prop

theorem smallest_sphere_radius :
  convex_polyhedron ∧ congruent_faces ∧ edge_relationships →
  (radius_enclosing_sphere : ℝ) = c * (sqrt (1 + sqrt 2)) :=
sorry

theorem polyhedron_volume :
  convex_polyhedron ∧ congruent_faces ∧ edge_relationships →
  (volume_polyhedron : ℝ) = c^3 * (2 * real.sqrt (real.sqrt 8)) / 3 * (1 + 5 * real.sqrt 2) :=
sorry

end smallest_sphere_radius_polyhedron_volume_l337_337314


namespace area_of_isosceles_right_triangle_l337_337372

def is_isosceles_right_triangle (a b c : ℝ) : Prop :=
  (a = b) ∧ (a^2 + b^2 = c^2)

theorem area_of_isosceles_right_triangle (a : ℝ) (hypotenuse : ℝ) (h_isosceles : is_isosceles_right_triangle a a hypotenuse) (h_hypotenuse : hypotenuse = 6) :
  (1 / 2) * a * a = 9 :=
by
  sorry

end area_of_isosceles_right_triangle_l337_337372


namespace equilateral_triangle_l337_337081

-- We define the sides and vectors as given
variables {a b c : ℝ} {A B C : Type}

-- The three conditions from the problem
variables {a_vector b_vector c_vector : Type}
variables (side_a side_b side_c : Type)
variables (BC CA AB : Type)
variables [dot_prod1 : inner_product_space ℝ BC] [dot_prod2 : inner_product_space ℝ CA] [dot_prod3 : inner_product_space ℝ AB]

-- The given conditions
variables (h1 : side_a = side_b) (h2 : side_b = side_c) (h3 : side_c = side_a)
variables (h4 : BC = a_vector) (h5 : CA = b_vector) (h6 : AB = c_vector)
variables (h7 : (a_vector • b_vector) = (b_vector • c_vector))
variables (h8 : (b_vector • c_vector) = (c_vector • a_vector))
variables (h9 : (c_vector • a_vector) = (a_vector • b_vector))

-- The proof problem to check if the triangle is equilateral
theorem equilateral_triangle (A B C : Type) 
  [dot_prod1 : inner_product_space ℝ BC]
  [dot_prod2 : inner_product_space ℝ CA]
  [dot_prod3 : inner_product_space ℝ AB]
  (h1 : side_a = side_b) (h2 : side_b = side_c) (h3 : side_c = side_a)
  (h4 : BC = a_vector) (h5 : CA = b_vector) (h6 : AB = c_vector)
  (h7 : (a_vector • b_vector) = (b_vector • c_vector))
  (h8 : (b_vector • c_vector) = (c_vector • a_vector))
  (h9 : (c_vector • a_vector) = (a_vector • b_vector)) :
  side_a = side_b → side_b = side_c → side_c = side_a :=
by sorry

end equilateral_triangle_l337_337081


namespace teacher_li_is_male_l337_337561

def gender_of_id_card (id_card : String) : String :=
  if id_card.get 16 % 2 = 1 then "Male"
  else "Female"

theorem teacher_li_is_male : gender_of_id_card "530322197303160019" = "Male" :=
by
  sorry

end teacher_li_is_male_l337_337561


namespace pascal_triangle_contains_53_only_once_l337_337804

theorem pascal_triangle_contains_53_only_once (n : ℕ) (k : ℕ) (h_prime : Nat.prime 53) :
  (n = 53 ∧ (k = 1 ∨ k = 52) ∨ 
   ∀ m < 53, Π l, Nat.binomial m l ≠ 53) ∧ 
  (n > 53 → (k = 0 ∨ k = n ∨ Π a b, a * 53 ≠ b * Nat.factorial (n - k + 1))) :=
sorry

end pascal_triangle_contains_53_only_once_l337_337804


namespace length_of_equal_side_none_of_the_above_l337_337219

noncomputable def isosceles_triangle_bisector (triangle : Type) :=
  ∃ (x : ℝ) (y z : ℝ), 
  -- Conditions
  2 * x + y + z = 98 ∧
  (y = (5 / 9) * x) ∧
  (z = (9 / 5) * y) ∧
  -- None of the given answers match
  x ≠ 26.4 ∧ 
  x ≠ 33 ∧ 
  x ≠ 38.5

theorem length_of_equal_side_none_of_the_above :
  isosceles_triangle_bisector (ℝ) :=
begin
  sorry
end

end length_of_equal_side_none_of_the_above_l337_337219


namespace probability_three_draws_l337_337633

theorem probability_three_draws (r w : ℕ) (p_r : ℚ) (p_w : ℚ) :
  r = 3 → w = 2 →
  p_r = (3 / 5 : ℚ) → p_w = (2 / 5 : ℚ) →
  let P := (p_r * p_w * p_r) + (p_w * p_r * p_r) in
  P = 36 / 125 :=
by
  intros hr hw hp_r hp_w
  let P1 := p_r * p_w * p_r
  let P2 := p_w * p_r * p_r
  let P := P1 + P2
  have : P1 = 18 / 125 := by sorry
  have : P2 = 18 / 125 := by sorry
  have : P = 36 / 125 := by sorry
  exact this

end probability_three_draws_l337_337633


namespace sum_of_real_solutions_l337_337016

theorem sum_of_real_solutions (x : ℝ) (h : ∃ x : ℝ, (sqrt x + sqrt (9 / x) + sqrt (x + 9 / x) = 8)) : 
  (∑ x in {x : ℝ | sqrt x + sqrt (9 / x) + sqrt (x + 9 / x) = 8}, x) = 3025 / 256 :=
sorry

end sum_of_real_solutions_l337_337016


namespace pascals_triangle_53_rows_l337_337784

theorem pascals_triangle_53_rows : 
  ∃! row, (∃ k, 1 ≤ k ∧ k ≤ row ∧ 53 = Nat.choose row k) ∧ 
          (∀ k, 1 ≤ k ∧ k ≤ row → 53 = Nat.choose row k → row = 53) :=
sorry

end pascals_triangle_53_rows_l337_337784


namespace smallest_non_10_multiple_abundant_l337_337358

def is_abundant (n : ℕ) : Prop :=
  (∑ d in Nat.properDivisors n, d) > n

def not_multiple_of_10 (n : ℕ) : Prop :=
  ¬ (10 ∣ n)

theorem smallest_non_10_multiple_abundant :
  ∀ n : ℕ, is_abundant n ∧ not_multiple_of_10 n → n = 12 :=
begin
  assume n,
  intros h,
  sorry
end

end smallest_non_10_multiple_abundant_l337_337358


namespace decreasing_interval_l337_337941

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x - Real.pi / 4)

theorem decreasing_interval :
  ∀ x ∈ Set.Icc 0 (2 * Real.pi), x ∈ Set.Icc (3 * Real.pi / 4) (2 * Real.pi) ↔ (∀ ε > 0, f x > f (x + ε)) := 
sorry

end decreasing_interval_l337_337941


namespace calculate_expression_l337_337684

theorem calculate_expression : (-1:ℝ)^2 + (1/3:ℝ)^0 = 2 := by
  sorry

end calculate_expression_l337_337684


namespace math_problem_l337_337689

theorem math_problem : (-1: ℝ)^2 + (1/3: ℝ)^0 = 2 := by
  sorry

end math_problem_l337_337689


namespace pipe_filling_time_l337_337194

-- Definitions for the conditions
variables (A : ℝ) (h : 1 / A - 1 / 24 = 1 / 12)

-- The statement of the problem
theorem pipe_filling_time : A = 8 :=
by
  sorry

end pipe_filling_time_l337_337194


namespace parabola_vertex_y_coord_l337_337966

theorem parabola_vertex_y_coord : 
  ∀ (x : ℝ), let y := -3 * x^2 - 30 * x - 81 in ∃ n : ℝ, n = -6 :=
by
  intros
  use -6
  sorry

end parabola_vertex_y_coord_l337_337966


namespace calculate_expression_l337_337681

theorem calculate_expression : (-1 : ℝ) ^ 2 + (1 / 3 : ℝ) ^ 0 = 2 := 
by
  sorry

end calculate_expression_l337_337681


namespace pascal_triangle_contains_53_once_l337_337832

theorem pascal_triangle_contains_53_once
  (h_prime : Nat.Prime 53) :
  ∃! n, ∃ k, n ≥ k ∧ n > 0 ∧ k > 0 ∧ Nat.choose n k = 53 := by
  sorry

end pascal_triangle_contains_53_once_l337_337832


namespace arithmetic_seq_sum_div_fifth_term_l337_337581

open Int

/-- The sequence {a_n} is an arithmetic sequence with a non-zero common difference,
    given that a₂ + a₆ = a₈, prove that S₅ / a₅ = 3. -/
theorem arithmetic_seq_sum_div_fifth_term
  (a : ℕ → ℤ)
  (d : ℤ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_nonzero : d ≠ 0)
  (h_condition : a 2 + a 6 = a 8) :
  ((5 * a 1 + 10 * d) / (a 1 + 4 * d) : ℚ) = 3 := 
by
  sorry

end arithmetic_seq_sum_div_fifth_term_l337_337581


namespace hexagonal_tile_problem_l337_337544

theorem hexagonal_tile_problem
  (initial_red_tiles : ℕ)
  (initial_yellow_tiles : ℕ)
  (tiles_per_side_of_hexagon : ℕ)
  (sides_of_hexagon : ℕ)
  (new_yellow_tiles : ℕ := sides_of_hexagon * tiles_per_side_of_hexagon)
  (total_yellow_tiles : ℕ := initial_yellow_tiles + new_yellow_tiles)
  (difference : ℕ := total_yellow_tiles - initial_red_tiles)
  (h_initial_red : initial_red_tiles = 12)
  (h_initial_yellow : initial_yellow_tiles = 8)
  (h_tiles_per_side : tiles_per_side_of_hexagon = 4)
  (h_sides_of_hexagon : sides_of_hexagon = 6) :
  difference = 20 :=
by
  rw [h_initial_red, h_initial_yellow, h_tiles_per_side, h_sides_of_hexagon]
  unfold new_yellow_tiles total_yellow_tiles difference
  rw [Nat.mul_comm]
  norm_num

-- Placeholder for the actual proof.
#check sorry

end hexagonal_tile_problem_l337_337544


namespace equal_roots_m_eq_nine_over_four_l337_337111

theorem equal_roots_m_eq_nine_over_four (m : ℝ) : 
  (∀ x : ℝ, (x^2 - 3*x + m = 0) → (x = (3 - sqrt(9 - 4*m))/2) ∨ (x = (3 + sqrt(9 - 4*m))/2)) → 
  (9 - 4*m = 0) → 
  m = 9 / 4 := 
by
  sorry

end equal_roots_m_eq_nine_over_four_l337_337111


namespace problem_solution_l337_337061

-- Definitions
def f (a b : ℝ) (x : ℝ) : ℝ := a * x * Real.log x + b
def g (a b : ℝ) (x : ℝ) : ℝ := (f a b x + 1) / x

noncomputable def monotonic_intervals (a : ℝ) : set (set ℝ) :=
  if a = 1 then
    {I | I = (set.Ioo 0 (Real.exp (-1))) ∨ I = (set.Ioi (Real.exp (-1)))}
  else
    ∅

-- The main theorem
theorem problem_solution :
  (f 1 0 x = x * Real.log x) ∧ (monotonic_intervals 1 = {set.Ioo 0 (Real.exp (-1)), set.Ioi (Real.exp (-1))}) ∧
  (∀ x1 x2 : ℝ, x1 < x2 → g 1 0 x1 = g 1 0 x2 → x1 + x2 > 2) := sorry

end problem_solution_l337_337061


namespace segments_before_returning_to_A_l337_337934

noncomputable def segments (angle : ℝ) : ℕ :=
  (360 / (2 * angle)).to_nat

theorem segments_before_returning_to_A :
  let angle := 60
  segments angle = 3 := by
  sorry

end segments_before_returning_to_A_l337_337934


namespace probability_two_queens_or_at_least_two_jacks_l337_337101

/-- There are 4 Queens and 4 Jacks in a standard deck of 52 cards. -/
variables (total_cards : ℕ := 52) (total_queens : ℕ := 4) (total_jacks : ℕ := 4)

/-- The probability of drawing either two queens or at least two jacks when 3 cards are selected randomly. -/
theorem probability_two_queens_or_at_least_two_jacks :
  probability (selecting_either_two_queens_or_at_least_two_jacks total_cards total_queens total_jacks) = 74 / 850 := sorry

end probability_two_queens_or_at_least_two_jacks_l337_337101


namespace equivalent_increase_l337_337293

variable (P : ℝ) (X : ℝ)
def final_price (P X : ℝ) : ℝ := (1.06) ^ 2 * P * (1 + X / 100)

theorem equivalent_increase (P X : ℝ) : 
  ∃ Y : ℝ, final_price P X = P * (1 + Y / 100) ∧ Y = 12.36 + 1.1236 * X :=
by
  use 12.36 + 1.1236 * X
  split
  · unfold final_price
    ring
  · sorry

end equivalent_increase_l337_337293


namespace distinct_values_count_sum_of_extremes_l337_337245

def x_values : Set ℤ := {-21, -20, -19, ..., 17, 18}
def y_values : Set ℤ := {-3, -4, ..., -13, -14}

theorem distinct_values_count : 
  (∃ x_vals y_vals, x_vals = {-21, -20, -19, ..., 17, 18} ∧ y_vals = {-3, -4, ..., -13, -14} ∧ x + y ∈ (x_vals × y_vals) → (51)) := sorry

theorem sum_of_extremes : 
  (∃ x_vals y_vals, x_vals = {-21, -20, -19, ..., 17, 18} ∧ y_vals = {-3, -4, ..., -13, -14} ∧ x + y ∈ (x_vals × y_vals) → (-20)) := sorry

end distinct_values_count_sum_of_extremes_l337_337245


namespace average_speed_is_50_l337_337231

-- Defining the conditions
def totalDistance : ℕ := 250
def totalTime : ℕ := 5

-- Defining the average speed
def averageSpeed := totalDistance / totalTime

-- The theorem statement
theorem average_speed_is_50 : averageSpeed = 50 := sorry

end average_speed_is_50_l337_337231


namespace cube_root_less_than_five_count_l337_337083

theorem cube_root_less_than_five_count :
  (∃ n : ℕ, n = 124 ∧ ∀ x : ℕ, 1 ≤ x → x < 5^3 → x < 125) := 
sorry

end cube_root_less_than_five_count_l337_337083


namespace convex_quadrilateral_angle_equality_l337_337037

/-- Given a convex quadrilateral KLMN where ∠NKL = 90°, P is the midpoint of LM, 
    and ∠KNL = ∠MKP, prove that ∠KNM = ∠LKP. -/
theorem convex_quadrilateral_angle_equality
  {K L M N P : Type}
  (convex_KLMN : convex_quadrilateral K L M N)
  (angle_NKL_90 : ∠ NKL = 90°)
  (P_midpoint_LM : midpoint P L M)
  (angle_KNL_eq_MKP : ∠ KNL = ∠ MKP) : ∠ KNM = ∠ LKP := 
sorry

end convex_quadrilateral_angle_equality_l337_337037


namespace vector_statement_D_incorrect_l337_337775

variables {α : Type*} [inner_product_space ℝ α]
variables (a b : α) [nonzero_vector a] [nonzero_vector b]

noncomputable theory

def statement_D_is_incorrect : Prop :=
  ∃ (λ : ℝ), (b = λ • a) → ∥a + b∥ ≠ ∥a∥ - ∥b∥

theorem vector_statement_D_incorrect (a b : α) [h₁ : nonzero_vector a] [h₂ : nonzero_vector b] : statement_D_is_incorrect a b :=
sorry

end vector_statement_D_incorrect_l337_337775


namespace square_of_binomial_l337_337840

theorem square_of_binomial (a : ℝ) :
  (∃ b : ℝ, (3 * x + b) ^ 2 = 9 * x^2 - 18 * x + a) ↔ a = 9 :=
by
  sorry

end square_of_binomial_l337_337840


namespace find_expression_for_f_sum_of_reciprocals_l337_337762

variables {A : ℝ} {φ : ℝ} (x : ℝ) (n : ℕ)
noncomputable def f : ℝ → ℝ := λ x, A * Real.sin (2 * x + φ)

axiom pos_A : A > 0
axiom range_φ : 0 < φ ∧ φ < π
axiom min_value : f (-π/3) = -4

theorem find_expression_for_f : f x = 4 * Real.sin (2 * x + π / 6) :=
sorry

noncomputable def S (n : ℕ) : ℝ := n * (n + 1) / 2
noncomputable def T (n : ℕ) : ℝ := Σ' k in Finset.range n, 1 / S (k + 1)

theorem sum_of_reciprocals (n : ℕ) : T n = 2 * n / (n + 1) :=
sorry

end find_expression_for_f_sum_of_reciprocals_l337_337762


namespace base_b_representation_1987_l337_337413

theorem base_b_representation_1987 (x y z b : ℕ) (h1 : x + y + z = 25) (h2 : x ≥ 1)
  (h3 : 1987 = x * b^2 + y * b + z) (h4 : 12 < b) (h5 : b < 45) :
  x = 5 ∧ y = 9 ∧ z = 11 ∧ b = 19 :=
sorry

end base_b_representation_1987_l337_337413


namespace cube_partition_into_tetrahedrons_l337_337139

theorem cube_partition_into_tetrahedrons :
  ∃ (n : ℕ), n = 5 ∧ 
    (∃ t : fin n → set (ℝ × ℝ × ℝ), 
      (∀ (i : fin n), ∃ (tetra : set (ℝ × ℝ × ℝ)), t i = tetra ∧ 
        ∃ (vertices : fin 4 → (ℝ × ℝ × ℝ)),
          (∀ j k : fin 4, j ≠ k → vertices j ≠ vertices k) ∧ 
            convex_hull (set.range vertices) = tetra) ∧
        (∀ i j : fin n, i ≠ j → (t i ∩ t j).nonempty → false) ∧
        (⋃ (i : fin n), convex_hull (set.range (λ x, t i))) = 
          convex_hull (set.univ : set (ℝ × ℝ × ℝ)))) :=
sorry

end cube_partition_into_tetrahedrons_l337_337139


namespace coefficient_of_x3_in_expansion_l337_337938

theorem coefficient_of_x3_in_expansion :
  let f := (1 + x - x^2)
  let c := (10 : ℕ)
  (binomial_theorem_expansion f c).coefficient(3) = 30 := by
  sorry

end coefficient_of_x3_in_expansion_l337_337938


namespace total_water_heaters_l337_337268

-- Define the conditions
variables (W C : ℕ) -- W: capacity of Wallace's water heater, C: capacity of Catherine's water heater
variable (wallace_3over4_full : W = 40 ∧ W * 3 / 4 ∧ C = W / 2 ∧ C * 3 / 4)

-- The proof problem
theorem total_water_heaters (wallace_3over4_full : W = 40 ∧ (W * 3 / 4 = 30) ∧ C = W / 2 ∧ (C * 3 / 4 = 15)) : W * 3 / 4 + C * 3 / 4 = 45 :=
sorry

end total_water_heaters_l337_337268


namespace binomial_expansion_integral_eval_l337_337472

theorem binomial_expansion_integral_eval :
  (let a := -C(5, 3) in ∫ x in a..-1, 2 * x dx = -99) :=
by
  -- Let a be the coefficient of the term with x in the expansion.
  let a := -C(5, 3)
  -- Calculate the integral.
  have : a = -10 := sorry
  calc
    ∫ x in a..-1, 2 * x dx = (x^2) ∣_{a..-1} : sorry
                         ... = (-1)^2 - (-10)^2 : sorry
                         ... = 1 - 100 : sorry
                         ... = -99 : sorry

end binomial_expansion_integral_eval_l337_337472


namespace math_problem_l337_337686

theorem math_problem : (-1: ℝ)^2 + (1/3: ℝ)^0 = 2 := by
  sorry

end math_problem_l337_337686


namespace surface_area_of_sphere_l337_337249

-- Definitions based on the problem statement
def cube_surface_area (a : ℝ) : Prop := 6 * (a / √6)^2 = a^2
def sphere_diameter (a : ℝ) : ℝ := (a / √6) * √3
def sphere_radius (a : ℝ) : ℝ := sphere_diameter(a) / 2
def sphere_surface_area (a : ℝ) : ℝ := 4 * π * (sphere_radius(a))^2

-- The statement to prove
theorem surface_area_of_sphere (a : ℝ) (h : cube_surface_area a) : sphere_surface_area a = 2 * π * a^2 :=
by {
  sorry
}

end surface_area_of_sphere_l337_337249


namespace equation_of_ellipse_equation_of_line_MN_l337_337042

-- Define the problem conditions
def ellipse (a b : ℝ) : (ℝ × ℝ) → Prop := λ p, let (x, y) := p in (x^2 / a^2) + (y^2 / b^2) = 1

def eccentricity (a c : ℝ) : ℝ := c / a

def distance (p q : ℝ × ℝ) : ℝ := sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

def point_on_ellipse (a b : ℝ) (A : ℝ × ℝ) : Prop := ellipse a b A

def angle (A B C : (ℝ × ℝ)) : ℝ := 
  let dAB := distance A B in
  let dBC := distance B C in
  let dAC := distance A C in
  acos ((dAB^2 + dBC^2 - dAC^2) / (2 * dAB * dBC))

def line_through (l : ℝ × ℝ → Prop) (F2 : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, ∃ b : ℝ, ∀ p : ℝ × ℝ, l p ↔ p.2 = k * (p.1 - F2.1) + b

def perpendicular (p q r : (ℝ × ℝ)) : Prop :=
  (q.2 - p.2) * (r.2 - p.2) + (q.1 - p.1) * (r.1 - p.1) = 0

-- Define the proof statements
theorem equation_of_ellipse
  (a b c : ℝ)
  (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
  (h4 : eccentricity a c = 1 / 2)
  (h5 : ∃ A : ℝ × ℝ, point_on_ellipse a b A ∧ distance A ⟨-c, 0⟩ = 2 ∧ angle ⟨-c, 0⟩ A ⟨c, 0⟩ = π / 3) :
  ellipse 2 sqrt(3) :=
sorry

theorem equation_of_line_MN
  (a b c : ℝ)
  (A : ℝ × ℝ)
  (l : ℝ × ℝ → Prop)
  (P Q : ℝ × ℝ)
  (M : ℝ × ℝ := (0, 1/8))
  (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
  (h4 : eccentricity a c = 1 / 2)
  (h5 : point_on_ellipse a b A ∧ distance A ⟨-c, 0⟩ = 2 ∧ angle ⟨-c, 0⟩ A ⟨c, 0⟩ = π / 3)
  (h6 : line_through l ⟨c, 0⟩)
  (h7 : ∃7, l P ∧ l Q)
  (h8 : ∃ N : ℝ × ℝ, (P + Q) / 2 = N ∧ perpendicular M N P) :
  ∃ k1 k2 b1 b2 : ℝ, k1 ≠ 0 ∧ k2 ≠ 0 ∧ 
  (∀ x y : ℝ, M = (x, y) → (x ≠ 0 → (16 * x + 8 * y - 1 = 0 ∨ 16 * x + 24 * y - 3 = 0))) :=
sorry

end equation_of_ellipse_equation_of_line_MN_l337_337042


namespace carton_eggs_to_milk_ratio_l337_337390

-- Step-by-step Lean translation of the problem
variables (price_bun : ℝ) (num_buns : ℕ) (price_milk_bottle : ℝ) (num_milk_bottles : ℕ)
variables (total_cost : ℝ) (cost_carton_eggs : ℝ)

-- Definitions derived from conditions in a)
def total_cost_buns := num_buns * price_bun
def total_cost_milk := num_milk_bottles * price_milk_bottle
def total_cost_buns_and_milk := total_cost_buns + total_cost_milk
def cost_carton_eggs := total_cost - total_cost_buns_and_milk

-- Question to be proven as a Lean statement
theorem carton_eggs_to_milk_ratio (h1 : price_bun = 0.1) (h2 : num_buns = 10) 
  (h3 : price_milk_bottle = 2) (h4 : num_milk_bottles = 2) (h5 : total_cost = 11) :
  cost_carton_eggs / price_milk_bottle = 3 :=
by {
  -- Using the given conditions to compute the costs
  unfold total_cost_buns total_cost_milk total_cost_buns_and_milk,
  -- Proving the ratio
  sorry
}

end carton_eggs_to_milk_ratio_l337_337390


namespace intersectionAandB_l337_337072

def setA (x : ℝ) : Prop := abs (x + 3) + abs (x - 4) ≤ 9
def setB (x : ℝ) : Prop := ∃ t : ℝ, 0 < t ∧ x = 4 * t + 1 / t - 6

theorem intersectionAandB : {x : ℝ | setA x} ∩ {x : ℝ | setB x} = {x : ℝ | -2 ≤ x ∧ x ≤ 5} := 
by 
  sorry

end intersectionAandB_l337_337072


namespace fathers_age_l337_337317

variable (S F : ℕ)
variable (h1 : F = 3 * S)
variable (h2 : F + 15 = 2 * (S + 15))

theorem fathers_age : F = 45 :=
by
  -- the proof steps would go here
  sorry

end fathers_age_l337_337317


namespace investment_in_stocks_is_l337_337334

-- Definitions based on the problem conditions
variables (total investment fixedDeposit : ℝ)
variables (investmentInBonds investmentInStocks : ℝ)

-- Given conditions
def totalInvestment : Prop := total = 200000
def fixedDepositAmt : Prop := fixedDeposit = 40000
def stockToBondRatio : Prop := investmentInStocks = 3.5 * investmentInBonds
def bondStockSum : Prop := investmentInBonds + investmentInStocks = total - fixedDeposit

-- Theorem to prove the amount invested in stocks
theorem investment_in_stocks_is :
  totalInvestment → fixedDepositAmt → stockToBondRatio → bondStockSum → investmentInStocks = 124444.44 :=
by
  intro h1 h2 h3 h4,
  sorry

end investment_in_stocks_is_l337_337334


namespace ternary_to_decimal_l337_337586

theorem ternary_to_decimal : ∀ n : ℕ, (n = 121) → (1 * 3^2 + 2 * 3^1 + 1 * 3^0 : ℕ) = 16 :=
by intros n h; rw h; sorry

end ternary_to_decimal_l337_337586


namespace smallest_degree_of_f_l337_337516

theorem smallest_degree_of_f (f : ℤ[X])
  (gcd_coeffs : f.coeffs.gcd = 1)
  (h : ∀ n : ℕ, 85 ∣ f.coeff n) : f.degree = 17 :=
sorry

end smallest_degree_of_f_l337_337516


namespace exists_divisible_by_13_in_79_consecutive_l337_337198

def sum_of_digits (n : ℕ) : ℕ :=
  (n.to_string.foldl (λ acc c, acc + c.to_nat - '0'.to_nat) 0)

theorem exists_divisible_by_13_in_79_consecutive :
  ∀n, ∃k, n ≤ k ∧ k < n + 79 ∧ (sum_of_digits k) % 13 = 0 :=
by
  intro n
  sorry

end exists_divisible_by_13_in_79_consecutive_l337_337198


namespace probability_D_told_truth_l337_337143

-- Define probability that a person tells the truth (1/3) and lies (2/3)
def prob_truth := 1 / 3
def prob_lie := 2 / 3

-- Define the events based on the problem's conditions
def A_claims_B_denies_C_asserts_D_lied (D_truth : Bool) : Bool :=
  if D_truth then 
    !C_asserts_D_lied false -- if D told the truth, check if C asserts D lied
  else 
    !C_asserts_D_lied true  -- if D lied, check if C asserts D lied

def B_denies (C_asserts: Bool) : Bool :=
  !C_asserts

def C_asserts_D_lied (D_truth: Bool) : Bool :=
  !D_truth

-- The main theorem to show the probability calculation
theorem probability_D_told_truth :
  P(D_told_truth | A_claims_B_denies_C_asserts_D_lied) = 13 / 41 :=
  sorry

end probability_D_told_truth_l337_337143


namespace area_of_triangle_MEF_l337_337128

theorem area_of_triangle_MEF
  (O : Point)
  (M A B E F : Point)
  (r : ℝ)
  (hO_rad : r = 10)
  (hEF_len : dist E F = 12)
  (hEF_parallel_MB : parallel E F M B)
  (hMA_len : dist M A = 24)
  (h_collinear : collinear M A O ∧ collinear A B O) :
  area_triangle M E F = 48 := 
sorry

end area_of_triangle_MEF_l337_337128


namespace max_distance_from_circle_center_to_line_l337_337035

theorem max_distance_from_circle_center_to_line :
  ∀ (center : ℝ × ℝ) (line : ℝ × ℝ × ℝ), 
  (dist center (2, 3) = 1) → (line = (3, -4, -4)) →
  let d := (|3 * 2 - 4 * 3 - 4| / real.sqrt (3^2 + 4^2)) in
  (d + 1 = 3) :=
begin
  intros center line h_radius h_line,
  let d := (|3 * 2 - 4 * 3 - 4| / real.sqrt (3^2 + 4^2)),
  have hd : d = 2,
  {
    -- Detailed proof of d calculation (not necessary here, just assuming it's calculated properly)
    sorry
  },
  have result : d + 1 = 3,
  {
    rw hd,
    norm_num,
  },
  exact result,
end

end max_distance_from_circle_center_to_line_l337_337035


namespace number_of_elements_l337_337257

theorem number_of_elements (n : ℕ) (S : ℕ) (sum_first_six : ℕ) (sum_last_six : ℕ) (sixth_number : ℕ)
    (h1 : S = 22 * n) 
    (h2 : sum_first_six = 6 * 19) 
    (h3 : sum_last_six = 6 * 27) 
    (h4 : sixth_number = 34) 
    (h5 : S = sum_first_six + sum_last_six - sixth_number) : 
    n = 11 := 
by
  sorry

end number_of_elements_l337_337257


namespace largest_int_less_than_100_with_remainder_2_div_6_l337_337715

theorem largest_int_less_than_100_with_remainder_2_div_6 : 
  ∃ x : ℤ, (x < 100 ∧ (x % 6 = 2) ∧ ∀ y : ℤ, y < 100 → (y % 6 = 2) → y ≤ x) := 
begin
  use 98,
  split,
  {
    linarith,
  },
  split,
  {
    norm_num,
  },
  {
    intros y hy hmod,
    have h' : y < 98 ∨ y = 98 ∨ y > 98, by linarith,
    cases h',
    {
      exfalso,
      have : (y % 6 < 2) ∨ (y % 6 > 2), 
      { 
        have h₁ : y % 6 < 2, from by linarith [int.mod_lt y (by norm_num)],
        have h₂ : y % 6 > 2, from by linarith [int.mod_nonneg y (by norm_num)],
        exact or.inl h₁,
      },
      cases this,
      {
        exfalso,
        exact absurd hmod this,
      },
      {
        exfalso,
        exact absurd hmod this,
      }
    },
    {
      exact le_of_eq h',
    },
    {
      exfalso,
      have h'' : x = 98, from by linarith,
      exact absurd h'' h',
    }
  }
end

end largest_int_less_than_100_with_remainder_2_div_6_l337_337715


namespace mixed_fractions_calculation_l337_337690

theorem mixed_fractions_calculation :
  2017 + (2016 / 2017) / (2019 + (1 / 2016)) + (1 / 2017) = 1 :=
by
  sorry

end mixed_fractions_calculation_l337_337690


namespace M_inter_N_eq_I_l337_337847

-- Definitions
def M : Set ℝ := { x | x < 2 }
def N : Set ℝ := { x | x^2 - x ≤ 0 }
def I : Set ℝ := Icc 0 1

-- Statement
theorem M_inter_N_eq_I : M ∩ N = I := by
  sorry

end M_inter_N_eq_I_l337_337847


namespace Maria_waist_size_correct_l337_337386

noncomputable def waist_size_mm (waist_size_in : ℕ) (mm_per_ft : ℝ) (in_per_ft : ℕ) : ℝ :=
  (waist_size_in : ℝ) / (in_per_ft : ℝ) * mm_per_ft

theorem Maria_waist_size_correct :
  let waist_size_in := 27
  let mm_per_ft := 305
  let in_per_ft := 12
  waist_size_mm waist_size_in mm_per_ft in_per_ft = 686.3 :=
by
  sorry

end Maria_waist_size_correct_l337_337386


namespace apple_and_pear_costs_l337_337914

theorem apple_and_pear_costs (x y : ℝ) (h1 : x + 2 * y = 194) (h2 : 2 * x + 5 * y = 458) : 
  y = 70 ∧ x = 54 := 
by 
  sorry

end apple_and_pear_costs_l337_337914


namespace prod_expression_eq_l337_337897

def f (x : ℕ) : ℕ := x^2 + 3*x + 2

theorem prod_expression_eq :
  (∏ n in Finset.range 2019, 1 - (2 / (f (n + 1) : ℝ))) = (337 : ℝ) / 1010 :=
by
  sorry

end prod_expression_eq_l337_337897


namespace distance_between_parallel_planes_l337_337713

noncomputable def distance_between_planes : ℝ :=
  let normal_vector := (3, -1, 2)
  let plane1 := (3:ℝ, -1:ℝ, 2:ℝ, -3:ℝ)
  let plane2 := (6:ℝ, -2:ℝ, 4:ℝ, 4:ℝ)
  let point_on_plane1 := (1:ℝ, 0:ℝ, 0:ℝ)
  let numerator := abs (6 * 1 - 2 * 0 + 4 * 0 + 4)
  let denominator := real.sqrt (6^2 + (-2)^2 + 4^2)
  let distance := numerator / denominator
  distance

theorem distance_between_parallel_planes :
  distance_between_planes = 5 * real.sqrt 14 / 14 :=
by
  sorry

end distance_between_parallel_planes_l337_337713


namespace toys_per_hour_computation_l337_337639

noncomputable def total_toys : ℕ := 20500
noncomputable def monday_hours : ℕ := 8
noncomputable def tuesday_hours : ℕ := 7
noncomputable def wednesday_hours : ℕ := 9
noncomputable def thursday_hours : ℕ := 6

noncomputable def total_hours_worked : ℕ := monday_hours + tuesday_hours + wednesday_hours + thursday_hours
noncomputable def toys_produced_each_hour : ℚ := total_toys / total_hours_worked

theorem toys_per_hour_computation :
  toys_produced_each_hour = 20500 / (8 + 7 + 9 + 6) :=
by
  -- Proof goes here
  sorry

end toys_per_hour_computation_l337_337639


namespace table_runners_coverage_l337_337592

theorem table_runners_coverage :
  let A_r := 212
  let A_t := 175
  let A_2 := 24
  let A_3 := 24
  let A_covered := A_r - (2 * A_2) - (3 * A_3)
  let percentage_covered := (A_covered / A_t) * 100
  percentage_covered ≈ 52.57 :=
by
  sorry

end table_runners_coverage_l337_337592


namespace pascals_triangle_53_rows_l337_337786

theorem pascals_triangle_53_rows : 
  ∃! row, (∃ k, 1 ≤ k ∧ k ≤ row ∧ 53 = Nat.choose row k) ∧ 
          (∀ k, 1 ≤ k ∧ k ≤ row → 53 = Nat.choose row k → row = 53) :=
sorry

end pascals_triangle_53_rows_l337_337786


namespace find_x_l337_337385

noncomputable section

variable (x : ℝ)
def vector_v : ℝ × ℝ := (x, 4)
def vector_w : ℝ × ℝ := (5, 2)
def projection (v w : ℝ × ℝ) : ℝ × ℝ :=
  let num := (v.1 * w.1 + v.2 * w.2)
  let den := (w.1 * w.1 + w.2 * w.2)
  (num / den * w.1, num / den * w.2)

theorem find_x (h : projection (vector_v x) (vector_w) = (3, 1.2)) : 
  x = 47 / 25 :=
by
  sorry

end find_x_l337_337385


namespace obtuse_angle_sum_l337_337466

theorem obtuse_angle_sum (A B C P: Type) [RightTriangle A B C]
    (angle_A_45 : angle A = 45)
    (angle_B_45 : angle B = 45)
    (angle_bisector_AP : IsAngleBisector AP A)
    (angle_bisector_BP : IsAngleBisector BP B)
    (P_intersection : Intersection AP BP P) :
    measure(angle APB) = 135 :=
    sorry

end obtuse_angle_sum_l337_337466


namespace Kyle_is_25_l337_337490

variable (Tyson_age : ℕ := 20)
variable (Frederick_age : ℕ := 2 * Tyson_age)
variable (Julian_age : ℕ := Frederick_age - 20)
variable (Kyle_age : ℕ := Julian_age + 5)

theorem Kyle_is_25 : Kyle_age = 25 := by
  sorry

end Kyle_is_25_l337_337490


namespace even_function_a_eq_neg_one_l337_337109

-- Definitions for the function f and the condition for it being an even function
def f (x a : ℝ) := (x - 1) * (x - a)

-- The theorem stating that if f is an even function, then a = -1
theorem even_function_a_eq_neg_one (a : ℝ) (h : ∀ x : ℝ, f x a = f (-x) a) : a = -1 :=
by
  sorry

end even_function_a_eq_neg_one_l337_337109


namespace genevieve_cherries_purchase_l337_337734

theorem genevieve_cherries_purchase (cherries_cost_per_kg: ℝ) (genevieve_money: ℝ) (extra_money_needed: ℝ) (total_kg: ℝ) : 
  cherries_cost_per_kg = 8 → 
  genevieve_money = 1600 →
  extra_money_needed = 400 →
  total_kg = (genevieve_money + extra_money_needed) / cherries_cost_per_kg →
  total_kg = 250 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end genevieve_cherries_purchase_l337_337734


namespace rs_value_l337_337168

theorem rs_value (r s : ℝ) (hr : 0 < r) (hs: 0 < s) (h1 : r^2 + s^2 = 1) (h2 : r^4 + s^4 = 3 / 4) :
  r * s = Real.sqrt 2 / 4 :=
sorry

end rs_value_l337_337168


namespace number_of_students_l337_337134

-- Define the sample size and probability
def sample_size := 50
def selection_probability := 0.1

-- State the theorem to prove the number of students
theorem number_of_students :
  sample_size / selection_probability = 500 :=
sorry

end number_of_students_l337_337134


namespace min_value_of_fraction_l337_337395

theorem min_value_of_fraction (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 1) : 
  ∃ m, m = 4 ∧ (∀ z, z = (y / x + 1 / y) → m ≤ z) :=
begin
  sorry
end

end min_value_of_fraction_l337_337395


namespace days_to_complete_work_l337_337845

theorem days_to_complete_work
  (M B W : ℝ)  -- Define variables for daily work done by a man, a boy, and the total work
  (hM : M = 2 * B)  -- Condition: daily work done by a man is twice that of a boy
  (hW : (13 * M + 24 * B) * 4 = W)  -- Condition: 13 men and 24 boys complete work in 4 days
  (H : 12 * M + 16 * B) -- Help Lean infer the first group's total work per day
  (hW2 : (12 * M + 16 * B) * 5 = W)  -- Condition: first group must complete work in same time (5 days, to be proven)
  : (12 * M + 16 * B) * 5 = W := -- Prove equivalence
sorry

end days_to_complete_work_l337_337845


namespace pascal_triangle_contains_53_once_l337_337831

theorem pascal_triangle_contains_53_once
  (h_prime : Nat.Prime 53) :
  ∃! n, ∃ k, n ≥ k ∧ n > 0 ∧ k > 0 ∧ Nat.choose n k = 53 := by
  sorry

end pascal_triangle_contains_53_once_l337_337831


namespace frog_never_stops_90_degrees_frog_stops_72_degrees_integer_θ_values_l337_337978

-- Problem (a)
theorem frog_never_stops_90_degrees (Q : Point) (θ : ℝ) (hθ : θ = 90) (F: Point) (hF : F ≠ Q) :
  ¬stops (frog Q θ F) :=
sorry

-- Problem (b)
theorem frog_stops_72_degrees (Q : Point) (θ : ℝ) (hθ : θ = 72) (F: Point) (hF : F ≠ Q) :
  stops (frog Q θ F) :=
sorry

-- Problem (c)
theorem integer_θ_values (Q: Point) (θ: ℕ) (hθ: 0 < θ ∧ θ < 180) :
  (number_of_θ_values (frog Q θ)) = 22 :=
sorry

end frog_never_stops_90_degrees_frog_stops_72_degrees_integer_θ_values_l337_337978


namespace parametric_curve_length_l337_337374

noncomputable def length_parametric_curve : ℝ :=
  ∫ t in 0..(2 * Real.pi), sqrt ((3 * Real.cos t)^2 + (-3 * Real.sin t)^2)

theorem parametric_curve_length :
  length_parametric_curve = 6 * Real.pi :=
sorry

end parametric_curve_length_l337_337374


namespace midpoint_perpendicular_bisects_l337_337956

-- Definitions for F being the midpoint, M on the arc, and AM > MB
variables {A B F M T : Type} [LinearOrderedField ℝ]
variables (AM MB AT TM : ℝ)

def midpoint (F : ℝ) (A B : ℝ) := F = (A + B) / 2
def point_on_arc (M : ℝ) (arc : ℝ) := M < arc
def arc_condition (AM MB : ℝ) := AM > MB

-- Main theorem statement
theorem midpoint_perpendicular_bisects {A B F M T : ℝ} :
  midpoint F A B ∧ point_on_arc M (A + B) ∧ AM > MB →
  AT = (TM + MB) :=
by
  sorry

end midpoint_perpendicular_bisects_l337_337956


namespace find_x_minus_2y_l337_337417

theorem find_x_minus_2y (x y : ℝ) (h1 : real.cbrt (x - 1) = 2) (h2 : real.sqrt (y + 2) = 3) : x - 2 * y = -5 :=
by
  sorry

end find_x_minus_2y_l337_337417


namespace ff3_eq_5_f_decreasing_on_interval_l337_337063

def f : ℝ → ℝ :=
  λ x, if x > 1 then log x / log (1/3) else -x^2 - 2*x + 4

theorem ff3_eq_5 : f (f 3) = 5 :=
  sorry

theorem f_decreasing_on_interval : ∀ x y, x ∈ Set.Icc (-1 : ℝ) (y : ℝ) → f x ≥ f y :=
  sorry

end ff3_eq_5_f_decreasing_on_interval_l337_337063


namespace find_monic_cubic_polynomial_with_integer_coefficients_l337_337370

noncomputable def Q (x : ℝ) : ℝ := x^3 - 6*x^2 + 12*x - 11

theorem find_monic_cubic_polynomial_with_integer_coefficients 
  (y : ℝ) (h : y = (3 : ℝ)^(1/3) + 2) :
  Q(y) = 0 ∧ (∀ (a b c d : ℤ), Q(x) = x^3 + a * x^2 + b * x + c → d = 1) := 
sorry

end find_monic_cubic_polynomial_with_integer_coefficients_l337_337370


namespace neg_univ_implies_exist_l337_337067

theorem neg_univ_implies_exist (p : Prop) :
  (∀ x : ℝ, 2^x = 5) ↔ (∃ x : ℝ, 2^x ≠ 5) :=
by sorry

end neg_univ_implies_exist_l337_337067


namespace inequality_inequal_pos_numbers_l337_337197

theorem inequality_inequal_pos_numbers {a b : ℝ} (h : a ≠ b) (ha : 0 < a) (hb : 0 < b) : 
  (2 / (1 / a + 1 / b)) < real.sqrt (a * b) ∧ real.sqrt (a * b) < (a + b) / 2 :=
by
  sorry

end inequality_inequal_pos_numbers_l337_337197


namespace interval_between_births_is_2_l337_337584

-- Given conditions
def sum_of_ages (I : ℕ) : ℕ :=
  let Y := 7
  Y + (Y + I) + (Y + 2 * I) + (Y + 3 * I) + (Y + 4 * I)

-- The condition that the sum of ages is 55 years
axiom sum_ages_is_55 (I : ℕ) : sum_of_ages I = 55

-- Prove the interval between births is 2 years
theorem interval_between_births_is_2 : ∃ I : ℕ, sum_of_ages I = 55 ∧ I = 2 :=
by
  exists 2
  split
  apply sum_ages_is_55
  sorry

end interval_between_births_is_2_l337_337584


namespace circle_area_triple_l337_337220

theorem circle_area_triple (r n : ℝ) (h : π * (r + n)^2 = 3 * π * r^2) : r = (n * (Real.sqrt 3 - 1)) / 2 :=
sorry

end circle_area_triple_l337_337220


namespace find_monic_cubic_polynomial_with_integer_coefficients_l337_337369

noncomputable def Q (x : ℝ) : ℝ := x^3 - 6*x^2 + 12*x - 11

theorem find_monic_cubic_polynomial_with_integer_coefficients 
  (y : ℝ) (h : y = (3 : ℝ)^(1/3) + 2) :
  Q(y) = 0 ∧ (∀ (a b c d : ℤ), Q(x) = x^3 + a * x^2 + b * x + c → d = 1) := 
sorry

end find_monic_cubic_polynomial_with_integer_coefficients_l337_337369


namespace AQ_eq_D2P_l337_337176

variables {A B C D1 D2 E1 E2 P Q : ℝ} -- Coordinates or lengths in ℝ-space

-- Definitions of the conditions
def triangle_ABC (A B C : ℝ) : Prop := A ≠ B ∧ B ≠ C ∧ A ≠ C
def incircle_tangent_BC (ω : ℝ) (BC : ℝ) (D1 : ℝ) : Prop := true -- To represent tangency
def incircle_tangent_AC (ω : ℝ) (AC : ℝ) (E1 : ℝ) : Prop := true -- To represent tangency
def equal_segment_CD2_BD1 (C D2 B D1 : ℝ) : Prop := C = D2 ∧ B = D1
def equal_segment_CE2_AE1 (C E2 A E1 : ℝ) : Prop := C = E2 ∧ A = E1
def intersection_AD2_BE2 (AD2 BE2 : ℝ) (P : ℝ) : Prop := true -- To represent intersection

-- Theorem statement
theorem AQ_eq_D2P
  (A B C ω D1 E1 D2 E2 P Q : ℝ)
  (h1 : triangle_ABC A B C)
  (h2 : incircle_tangent_BC ω B D1)
  (h3 : incircle_tangent_AC ω A E1)
  (h4 : equal_segment_CD2_BD1 C D2 B D1)
  (h5 : equal_segment_CE2_AE1 C E2 A E1)
  (h6 : intersection_AD2_BE2 D2 E2 P)
  (h7 : ω = ω ∧ ω = ω ∧ A ≠ P ∧ Q ≠ D2) :
  (A - Q) = (D2 - P) :=
  
  begin
  sorry,
  end

end AQ_eq_D2P_l337_337176


namespace upper_seat_ticket_price_l337_337937

variable (U : ℝ) 

-- Conditions
def lower_seat_price : ℝ := 30
def total_tickets_sold : ℝ := 80
def total_revenue : ℝ := 2100
def lower_tickets_sold : ℝ := 50

theorem upper_seat_ticket_price :
  (lower_seat_price * lower_tickets_sold + (total_tickets_sold - lower_tickets_sold) * U = total_revenue) →
  U = 20 := by
  sorry

end upper_seat_ticket_price_l337_337937


namespace correct_propositions_l337_337892

variables {line : Type} [has_perp line] [has_parallel line]
variables {plane : Type} [has_perp plane] [has_parallel plane]

variables (m n : line) (α β γ : plane)

theorem correct_propositions :
(m ⊥ α ∧ n || α → m ⊥ n) ∧
(α ⊥ γ ∧ β ⊥ γ → ¬ (α || β)) ∧
(m || α ∧ n || α → ¬ (m || n)) ∧
(α || β ∧ β || γ ∧ m ⊥ α → m ⊥ γ) :=
begin
  split,
  { intros h,
    cases h,
    exact sorry, },
  split,
  { intros h,
    cases h,
    exact sorry, },
  split,
  { intros h,
    cases h,
    exact sorry, },
  { intros h,
    cases h,
    exact sorry, },
end

end correct_propositions_l337_337892


namespace intersection_M_N_l337_337439

def M := {x : ℤ | 1 < x ∧ x < 4}
def N := {1, 2, 3, 4, 5}

theorem intersection_M_N : M ∩ N = {2, 3} :=
by
  sorry

end intersection_M_N_l337_337439


namespace asymptotic_lines_necc_suff_l337_337221

theorem asymptotic_lines_necc_suff (a b c e: ℝ) (ha: a ≠ 0) (hb: b ≠ 0) (hc: c = sqrt (a^2 + b^2))
  (hyp1 : a / b = sqrt 2 → e = c / a → e ≠ sqrt 3)
  (hyp2 : b / a = sqrt 2 → e = c / a → e ≠ sqrt 3) :
  ¬ ((∀ a b c, a / b = sqrt 2 ∧ e = c / a → e = sqrt 3) ∧ 
     (∀ a b c, e = sqrt 3 → a / b = sqrt 2)) :=
sorry

end asymptotic_lines_necc_suff_l337_337221


namespace pascal_triangle_contains_53_once_l337_337828

theorem pascal_triangle_contains_53_once
  (h_prime : Nat.Prime 53) :
  ∃! n, ∃ k, n ≥ k ∧ n > 0 ∧ k > 0 ∧ Nat.choose n k = 53 := by
  sorry

end pascal_triangle_contains_53_once_l337_337828


namespace polygons_overlap_area_at_least_one_l337_337291

theorem polygons_overlap_area_at_least_one :
  ∀ (square : Set ℝ) (polygons : Set (Set ℝ)),
    measure square = 6 →
    (∀ p ∈ polygons, measure p = 3) →
    (Set.card polygons = 3) →
    ∃ (p1 p2 : Set ℝ), p1 ∈ polygons ∧ p2 ∈ polygons ∧ p1 ≠ p2 ∧ (measure (p1 ∩ p2) ≥ 1) :=
 by sorry

end polygons_overlap_area_at_least_one_l337_337291


namespace traci_flour_l337_337594

variable (HarrisFlour : ℕ) (cakeFlour : ℕ) (cakesEach : ℕ)

theorem traci_flour (HarrisFlour := 400) (cakeFlour := 100) (cakesEach := 9) :
  ∃ (TraciFlour : ℕ), 
  (cakesEach * 2 * cakeFlour) - HarrisFlour = TraciFlour ∧ 
  TraciFlour = 1400 :=
by
  have totalCakes : ℕ := cakesEach * 2
  have totalFlourNeeded : ℕ := totalCakes * cakeFlour
  have TraciFlour := totalFlourNeeded - HarrisFlour
  exact ⟨TraciFlour, rfl, rfl⟩

end traci_flour_l337_337594


namespace reciprocal_of_8_l337_337962

theorem reciprocal_of_8:
  (1 : ℝ) / 8 = (1 / 8 : ℝ) := by
  sorry

end reciprocal_of_8_l337_337962


namespace brand_z_percentage_correct_l337_337335

noncomputable def percentage_of_brand_z (capacity : ℝ := 1) (brand_z1 : ℝ := 1) (brand_x1 : ℝ := 0) 
(brand_z2 : ℝ := 1/4) (brand_x2 : ℝ := 3/4) (brand_z3 : ℝ := 5/8) (brand_x3 : ℝ := 3/8) 
(brand_z4 : ℝ := 5/16) (brand_x4 : ℝ := 11/16) : ℝ :=
    (brand_z4 / (brand_z4 + brand_x4)) * 100

theorem brand_z_percentage_correct : percentage_of_brand_z = 31.25 := by
  sorry

end brand_z_percentage_correct_l337_337335


namespace no_arithmetic_or_geometric_l337_337440

noncomputable def arithmetic_sequence (a d : ℝ) := (a, a + d, a + 2 * d)

noncomputable def harmonic_sequence (a b c : ℝ) := (1/a, 1/b, 1/c)

theorem no_arithmetic_or_geometric (a d : ℝ) (h : d ≠ 0) :
  ∀ (seq : ℝ × ℝ × ℝ),
    seq = harmonic_sequence a (a + d) (a + 2 * d) →
    ¬ ((∃ a_diff : ℝ, seq.2 - seq.1 = a_diff ∧ seq.3 - seq.2 = a_diff) ∨
       (∃ r : ℝ, seq.2 = seq.1 * r ∧ seq.3 = seq.2 * r)) :=
by sorry

end no_arithmetic_or_geometric_l337_337440


namespace pascal_triangle_contains_53_once_l337_337827

theorem pascal_triangle_contains_53_once
  (h_prime : Nat.Prime 53) :
  ∃! n, ∃ k, n ≥ k ∧ n > 0 ∧ k > 0 ∧ Nat.choose n k = 53 := by
  sorry

end pascal_triangle_contains_53_once_l337_337827


namespace equal_faces_are_rhombuses_l337_337918

theorem equal_faces_are_rhombuses
    (P : Type) [parallelepiped P]
    (faces_equal : ∀ (f1 f2 : parallelogram), f1 ∈ faces P → f2 ∈ faces P → f1 = f2)
    (opposite_faces_equal : ∀ (f1 f2 : parallelogram), f1 ∈ faces P → f2 ∈ faces P → opposite_face f1 = f2)
    (edges_equal : ∀ (e1 e2 e3 : edge), e1 ∈ edges P → e2 ∈ edges P → e3 ∈ edges P → e1 = e2 ∧ e2 = e3) :
  ∀ (f : parallelogram), f ∈ faces P → is_rhombus f :=
by
  apply sorry

end equal_faces_are_rhombuses_l337_337918


namespace probability_two_red_buttons_l337_337882

noncomputable def jar_probability : ℚ :=
let total_initial_buttons : ℚ := 12
    total_final_buttons : ℚ := (2 / 3) * total_initial_buttons
    total_buttons_removed : ℚ := total_initial_buttons - total_final_buttons
    red_buttons_removed : ℚ := total_buttons_removed / 2
    blue_buttons_removed : ℚ := total_buttons_removed / 2
    red_buttons_jar_a : ℚ := 4 - red_buttons_removed
    blue_buttons_jar_a : ℚ := 8 - blue_buttons_removed
    total_buttons_jar_a : ℚ := red_buttons_jar_a + blue_buttons_jar_a
    red_buttons_jar_b : ℚ := red_buttons_removed
    blue_buttons_jar_b : ℚ := blue_buttons_removed
    total_buttons_jar_b : ℚ := red_buttons_jar_b + blue_buttons_jar_b
    prob_red_jar_a : ℚ := red_buttons_jar_a / total_buttons_jar_a
    prob_red_jar_b : ℚ := red_buttons_jar_b / total_buttons_jar_b
in prob_red_jar_a * prob_red_jar_b

theorem probability_two_red_buttons :
  jar_probability = 1 / 8 :=
by sorry

end probability_two_red_buttons_l337_337882


namespace value_of_f_8_l337_337098

variable (f : ℝ → ℝ)

-- Define the conditions as assumptions
axiom f_condition : ∀ (x y : ℝ), f(x + y) = f(x) * f(y)
axiom f_at_2 : f 2 = 3

-- State the theorem to prove the correct answer
theorem value_of_f_8 : f 8 = 81 :=
sorry

end value_of_f_8_l337_337098


namespace route_B_is_faster_by_7_5_minutes_l337_337187

def distance_A := 10  -- miles
def normal_speed_A := 30  -- mph
def construction_distance_A := 2  -- miles
def construction_speed_A := 15  -- mph
def distance_B := 8  -- miles
def normal_speed_B := 40  -- mph
def school_zone_distance_B := 1  -- miles
def school_zone_speed_B := 10  -- mph

noncomputable def time_for_normal_speed_A : ℝ := (distance_A - construction_distance_A) / normal_speed_A * 60  -- minutes
noncomputable def time_for_construction_A : ℝ := construction_distance_A / construction_speed_A * 60  -- minutes
noncomputable def total_time_A : ℝ := time_for_normal_speed_A + time_for_construction_A

noncomputable def time_for_normal_speed_B : ℝ := (distance_B - school_zone_distance_B) / normal_speed_B * 60  -- minutes
noncomputable def time_for_school_zone_B : ℝ := school_zone_distance_B / school_zone_speed_B * 60  -- minutes
noncomputable def total_time_B : ℝ := time_for_normal_speed_B + time_for_school_zone_B

theorem route_B_is_faster_by_7_5_minutes : total_time_B + 7.5 = total_time_A := by
  sorry

end route_B_is_faster_by_7_5_minutes_l337_337187


namespace find_k_l337_337837

variable {α k : ℝ}

theorem find_k
  (h1 : ∀ α : ℝ, ∃ k : ℝ, is_root (2 * x^2 - 4 * k * x - 3 * k) (real.sin α) ∧ 
                             is_root (2 * x^2 - 4 * k * x - 3 * k) (real.cos α)) :
  k = 1 / 4 :=
sorry

end find_k_l337_337837


namespace pascals_triangle_53_rows_l337_337785

theorem pascals_triangle_53_rows : 
  ∃! row, (∃ k, 1 ≤ k ∧ k ≤ row ∧ 53 = Nat.choose row k) ∧ 
          (∀ k, 1 ≤ k ∧ k ≤ row → 53 = Nat.choose row k → row = 53) :=
sorry

end pascals_triangle_53_rows_l337_337785


namespace ant_food_cost_l337_337538

-- Definitions for the conditions
def number_of_ants : ℕ := 400
def food_per_ant : ℕ := 2
def job_charge : ℕ := 5
def leaf_charge : ℕ := 1 / 100 -- 1 penny is 1 cent which is 0.01 dollars
def leaves_raked : ℕ := 6000
def jobs_completed : ℕ := 4

-- Compute the total money earned from jobs
def money_from_jobs : ℕ := jobs_completed * job_charge

-- Compute the total money earned from raking leaves
def money_from_leaves : ℕ := leaves_raked * leaf_charge

-- Compute the total money earned
def total_money_earned : ℕ := money_from_jobs + money_from_leaves

-- Compute the total ounces of food needed
def total_food_needed : ℕ := number_of_ants * food_per_ant

-- Calculate the cost per ounce of food
def cost_per_ounce : ℕ := total_money_earned / total_food_needed

theorem ant_food_cost :
  cost_per_ounce = 1 / 10 := sorry

end ant_food_cost_l337_337538


namespace find_value_of_a_l337_337755

variable (a : ℝ)

def f (x : ℝ) := x^2 + 4
def g (x : ℝ) := x^2 - 2

theorem find_value_of_a (h_pos : a > 0) (h_eq : f (g a) = 12) : a = Real.sqrt (2 * (Real.sqrt 2 + 1)) := 
by
  sorry

end find_value_of_a_l337_337755


namespace exp_inequality_l337_337028

theorem exp_inequality (a b : ℝ) (h : a < b) :
  let f := λ x : ℝ, Real.exp x
  let A := f b - f a
  let B := 1/2 * (b - a) * (f a + f b)
  A < B :=
by 
  -- Definitions
  let f := λ x : ℝ, Real.exp x
  let A := f b - f a
  let B := 1/2 * (b - a) * (f a + f b)
  sorry

end exp_inequality_l337_337028


namespace smallest_possible_domain_length_l337_337604

def f (x : ℕ) : ℕ :=
  if x % 2 = 1 then 3 * x + 1 else x / 2

theorem smallest_possible_domain_length :
  ∀ (f: ℕ → ℕ), (f 13 = 33) ∧ (∀ a b, f a = b → (f b = 3 * b + 1 ∨ f b = b / 2)) →
  set.finite {x | ∃ y, f y = x} ∧ (set.finite (set_of (λ x, ∃ y, f y = x))).to_finset.card = 18 :=
begin
  sorry
end

end smallest_possible_domain_length_l337_337604


namespace log_equation_solutions_l337_337930

noncomputable def solutions_to_logarithmic_equation : Set ℝ :=
  { x | log (2 * x) (4 * x) + log (4 * x) (16 * x) = 4 }

theorem log_equation_solutions (x : ℝ) :
  x > 0 → 2 * x ≠ 1 → 4 * x ≠ 1 →
  (x = 1 ∨ x = 1 / (2 * Real.sqrt 2)) ↔ x ∈ solutions_to_logarithmic_equation :=
by
  intro hx1 hx2 hx3
  split
  { intro h
    cases h
    { rw [h]
      sorry }
    { rw [h]
      sorry } }
  { intro h
    sorry }

end log_equation_solutions_l337_337930


namespace last_digit_base_5_119_l337_337697

def convert_to_base_5 (n : ℕ) : list ℕ :=
  if n < 5 then [n] else convert_to_base_5 (n / 5) ++ [n % 5]

theorem last_digit_base_5_119 : (convert_to_base_5 119).last = some 4 :=
by
  sorry

end last_digit_base_5_119_l337_337697


namespace invoice_mistyped_correct_amount_l337_337336

theorem invoice_mistyped_correct_amount (x y : ℕ) (hx : 10 ≤ x ∧ x ≤ 99) (hy : 10 ≤ y ∧ y ≤ 99) (h_exceeds : 100 * x + y - (100 * y + x) = 3654) : 
x = 63 ∧ y = 26 :=
by 
  have h : x - y = 37 := calc
    100 * x + y - (100 * y + x) = 3654 : h_exceeds
    99 * x - 99 * y = 3654 : by ring
    x - y = 37 : by norm_num
  have h_bound : x = 63 ∧ y = 26 
  sorry

end invoice_mistyped_correct_amount_l337_337336


namespace range_of_g_l337_337704

open Real

def g (t : ℝ) : ℝ := (t^2 + t) / (t^2 + 1)

theorem range_of_g :
  ∀ y, (∃ t : ℝ, g t = y) ↔ y = 1 / 2 :=
by sorry

end range_of_g_l337_337704


namespace max_value_quadratic_l337_337055

theorem max_value_quadratic (m a b : ℝ)
  (h_roots : ∀ x, x^2 + 2 * m * x + m = (x - a) * (x - b))
  (h_ab_sum : 4 ≤ a + b ∧ a + b ≤ 6) :
  ∃ H, (∀ x, 1 ≤ x ∧ x ≤ 3 → y x = x^2 + 2 * m * x + m) ∧ H = 3 * m + 1 :=
by
    sorry

end max_value_quadratic_l337_337055


namespace pascals_triangle_53_rows_l337_337790

theorem pascals_triangle_53_rows : 
  ∃! row, (∃ k, 1 ≤ k ∧ k ≤ row ∧ 53 = Nat.choose row k) ∧ 
          (∀ k, 1 ≤ k ∧ k ≤ row → 53 = Nat.choose row k → row = 53) :=
sorry

end pascals_triangle_53_rows_l337_337790


namespace lena_more_than_nicole_l337_337149

theorem lena_more_than_nicole :
  ∀ (L K N : ℝ),
    L = 37.5 →
    (L + 9.5) = 5 * K →
    K = N - 8.5 →
    (L - N) = 19.6 :=
by
  intros L K N hL hLK hK
  sorry

end lena_more_than_nicole_l337_337149


namespace order_of_f_values_l337_337029

noncomputable def f (x : ℝ) : ℝ := Real.log x - Real.exp (-x)

noncomputable def a : ℝ := Real.exp 1 ^ Real.exp 1 -- 2^e
noncomputable def b : ℝ := Real.log 2 -- ln 2
noncomputable def c : ℝ := Real.log 2 -- log_2 e = ln(e)/ln(2) = 1

theorem order_of_f_values : f b < f c < f a :=
by sorry

end order_of_f_values_l337_337029


namespace shaded_area_l337_337467

-- Define the problem conditions
def radius : ℝ := 6 -- Radius of the circle is 6
def angle_sector : ℝ := (60 : ℝ) -- Angle subtended by each sector is 60 degrees
def pi : ℝ := Real.pi -- Define π

-- Define the lean statement
theorem shaded_area (radius : ℝ) (angle_sector : ℝ) (pi : ℝ) : 
  let area_shaded := 2 * (1/2 * radius * radius) + 2 * (1/6 * pi * radius^2)
  in area_shaded = 36 + 12 * pi :=
by
  sorry -- Proof placeholder

end shaded_area_l337_337467


namespace limit_sequence_sqrt_l337_337294

theorem limit_sequence_sqrt:
  tendsto (λ n : ℕ, (sqrt ((n^3 + 1) * (n^2 + 3)) - sqrt (n * (n^4 + 2))) / (2 * sqrt n))
    at_top (𝓝 (3 / 4)) :=
sorry

end limit_sequence_sqrt_l337_337294


namespace sum_of_3digit_numbers_remainder_2_l337_337275

-- Define the smallest and largest three-digit numbers leaving remainder 2 when divided by 5
def smallest : ℕ := 102
def largest  : ℕ := 997
def common_diff : ℕ := 5

-- Define the arithmetic sequence
def seq_length : ℕ := ((largest - smallest) / common_diff) + 1
def sequence_sum : ℕ := seq_length * (smallest + largest) / 2

-- The theorem to be proven
theorem sum_of_3digit_numbers_remainder_2 : sequence_sum = 98910 :=
by
  sorry

end sum_of_3digit_numbers_remainder_2_l337_337275


namespace find_k_values_l337_337217

noncomputable def k_values (a b c d k : ℂ) : Prop :=
  (a ≠ 0) ∧ (b ≠ 0) ∧ (c ≠ 0) ∧ (d ≠ 0) ∧ 
  (a * k^4 + b * k^3 + c * k^2 + d * k + a = 0) ∧
  (a * k^3 + b * k^2 + c * k + d = 0)

theorem find_k_values (a b c d : ℂ) :
  (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) →
  ∃ k, k_values a b c d k ∧ 
    (k = complex.exp (complex.I * π / 4) ∨ 
     k = -complex.exp (complex.I * π / 4) ∨ 
     k = complex.exp (3 * complex.I * π / 4) ∨ 
     k = -complex.exp (3 * complex.I * π / 4)) :=
by
  sorry

end find_k_values_l337_337217


namespace marcie_average_speed_l337_337528

variable (O1 O2 : ℕ) -- Odometer readings
variable (T1 T2 : ℕ) -- Time driven in hours 

def average_speed (distance : ℕ) (time : ℝ) : ℝ :=
  (distance : ℝ) / time

theorem marcie_average_speed :
  O1 = 25652 → O2 = 25852 → T1 = 5 → T2 = 7 →
  average_speed (O2 - O1) (T1 + T2) = 16.67 :=
by
  intros h1 h2 h3 h4
  sorry

end marcie_average_speed_l337_337528


namespace transformed_set_stats_l337_337041

variables (a : ℕ → ℝ) (n : ℕ)
noncomputable def average (a : ℕ → ℝ) (n : ℕ) : ℝ := (1 / n) * (Finset.range n).sum (λ i, a i)
noncomputable def stddev (a : ℕ → ℝ) (n : ℕ) : ℝ := 
  real.sqrt ((1 / n) * (Finset.range n).sum (λ i, (a i - average a n) ^ 2))

theorem transformed_set_stats (a : ℕ → ℝ) (n : ℕ) :
  let avg := average a n in
  let s := stddev a n in
  average (λ i, -2 * a i + 3) n = -2 * avg + 3 ∧ stddev (λ i, -2 * a i + 3) n = 2 * s := 
by
  sorry

end transformed_set_stats_l337_337041


namespace lim_T_n_div_n_sq_l337_337509

noncomputable def T_n (n : ℕ) : ℕ :=
  ∑ i in Finset.range (n + 1), i + 1

theorem lim_T_n_div_n_sq :
  (real.lim ((λ n : ℕ, (T_n n : ℝ) / (n : ℝ)^2) : ℕ → ℝ)) = 1 / 2 := by
sorry

end lim_T_n_div_n_sq_l337_337509


namespace transform_sin_to_cos_l337_337973

theorem transform_sin_to_cos (x : ℝ) :
  (∃ f : ℝ → ℝ, (∀ x, f x = sin (x + π / 4) + 1) ∧ (∀ x, f x = (sqrt 2 / 2) * sin x + (sqrt 2 / 2) * cos x + 1)) →
  (∃ shift : ℝ, ∃ shift_up : ℝ, (shift = -π / 4) ∧ (shift_up = 1)) :=
by
  sorry

end transform_sin_to_cos_l337_337973


namespace range_of_a_l337_337047

variable (a : ℝ)
def p : Prop := a > 1/4
def q : Prop := a ≤ -1 ∨ a ≥ 1

theorem range_of_a :
  ((p a ∧ ¬ (q a)) ∨ (q a ∧ ¬ (p a))) ↔ (a > 1/4 ∧ a < 1) ∨ (a ≤ -1) :=
by
  sorry

end range_of_a_l337_337047


namespace volleyball_uniform_probability_l337_337488

theorem volleyball_uniform_probability :
  let shorts_colors := {black, white, gold}
  let jersey_colors := {black, white, gold, blue}
  let total_configurations := shorts_colors.card * jersey_colors.card
  let non_matching_configurations := shorts_colors.card * (jersey_colors.card - 1)
  (non_matching_configurations : ℚ) / total_configurations = 3 / 4 :=
by
  sorry

end volleyball_uniform_probability_l337_337488


namespace line_intersects_circle_l337_337237

theorem line_intersects_circle (a: ℝ) :
  let l := (a-1)*x - y + a = 1
  let c := x^2 + y^2 + 2*x + 4*y - 20 = 0
  ∃ p : ℝ × ℝ, l p.1 p.2 ∧ c p.1 p.2 :=
by
  sorry

end line_intersects_circle_l337_337237


namespace num_pairs_satisfying_inequality_l337_337781

theorem num_pairs_satisfying_inequality : 
  ∃ (s : Nat), s = 204 ∧ ∀ (m n : ℕ), m > 0 → n > 0 → m^2 + n < 50 → s = 204 :=
by
  sorry

end num_pairs_satisfying_inequality_l337_337781


namespace parabola_vertex_n_l337_337968

theorem parabola_vertex_n : 
  (∃ m : ℝ, ∃ n : ℝ, (∀ x : ℝ, -3 * x^2 - 30 * x - 81 = -3 * (x + m) ^ 2 + n) ∧ n = -6) :=
begin
  sorry
end

end parabola_vertex_n_l337_337968


namespace pascal_triangle_contains_53_l337_337823

theorem pascal_triangle_contains_53 (n : ℕ) :
  (∃ k, binomial n k = 53) ↔ n = 53 := 
sorry

end pascal_triangle_contains_53_l337_337823


namespace feb_10_nine_day_period_l337_337238

def days_in_month (month : ℕ) (year : ℕ) : ℕ := 
  if month = 12 then 31
  else if month = 1 then 31
  else if month = 2 then 28 -- February in non-leap year simplification
  else 0

def days_between (start : ℕ × ℕ × ℕ) (end : ℕ × ℕ × ℕ) : ℕ :=
  let (start_year, start_month, start_day) := start
  let (end_year, end_month, end_day) := end
  let days_in_dec := (days_in_month 12 start_year) - start_day + 1
  let days_in_jan := days_in_month 1 end_year
  let days_in_feb := end_day
  days_in_dec + days_in_jan + days_in_feb

def winter_solstice_to_feb_10 : ℕ := days_between (2012, 12, 21) (2013, 2, 10)

def number_nine (total_days : ℕ) : ℕ := total_days / 9 + 1
def day_in_nine (total_days : ℕ) : ℕ := total_days % 9

theorem feb_10_nine_day_period :
  number_nine winter_solstice_to_feb_10 = 6 ∧ day_in_nine winter_solstice_to_feb_10 = 7 := by
  sorry

end feb_10_nine_day_period_l337_337238


namespace find_a_l337_337849

theorem find_a (a : ℝ) : (∀ x : ℝ, |x - a| ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) → a = 2 :=
by
  intro h
  have h1 : |(-1 : ℝ) - a| = 3 := sorry
  have h2 : |(5 : ℝ) - a| = 3 := sorry
  sorry

end find_a_l337_337849


namespace S4_div_S12_eq_neg1_div_12_l337_337423

-- Define the sum of the first n terms of an arithmetic sequence
def Sn (a : ℕ → ℝ) (n : ℕ) : ℝ := (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))

-- Define the properties given
axiom S8_eq_neg3S4 (S : ℕ → ℝ) : S 8 = -3 * S 4

-- Define the condition that S8 is not 0
axiom S8_ne_zero (S : ℕ → ℝ) : S 8 ≠ 0

-- Prove the main statement
theorem S4_div_S12_eq_neg1_div_12 (S : ℕ → ℝ) [forall n, Sum_Arith_Sequence (S n)] : S 4 / S 12 = -1 / 12 :=
by
  sorry

end S4_div_S12_eq_neg1_div_12_l337_337423


namespace total_valid_votes_l337_337465

theorem total_valid_votes (V : ℝ) (h1 : 0.70 * V - 0.30 * V = 176) : V = 440 :=
by sorry

end total_valid_votes_l337_337465


namespace part_one_part_two_l337_337677

theorem part_one : 
  (2 * log10 2 + log10 3) / (1 + 1/2 * log10 0.36 + 1/3 * log10 8) = 1 := 
by
  sorry

theorem part_two : 
  3 * (-4) ^ 3 - (1/2)^0 + 0.25^(1/2) * (-1 / real.sqrt 2)^(-4) = -191 := 
by
  sorry

end part_one_part_two_l337_337677


namespace square_of_binomial_l337_337841

theorem square_of_binomial (a : ℝ) :
  (∃ b : ℝ, (3 * x + b) ^ 2 = 9 * x^2 - 18 * x + a) ↔ a = 9 :=
by
  sorry

end square_of_binomial_l337_337841


namespace general_equation_of_curve_l337_337227

variable (θ x y : ℝ)

theorem general_equation_of_curve
  (h1 : x = Real.cos θ - 1)
  (h2 : y = Real.sin θ + 1) :
  (x + 1)^2 + (y - 1)^2 = 1 := sorry

end general_equation_of_curve_l337_337227


namespace count_possible_s_values_l337_337234

noncomputable def pqrs_is_digit (n : ℕ) : Prop := n < 10

theorem count_possible_s_values :
  let lower_bound : ℚ := 1592 / 10000
  let upper_bound : ℚ := 3030 / 10000
  ∀ (s : ℚ),  lower_bound ≤ s ∧ s < upper_bound ∧ 
              ∃ p q r s,
                (0 <= p) ∧ (p < 10) ∧
                (0 <= q) ∧ (q < 10) ∧
                (0 <= r) ∧ (r < 10) ∧
                (0 <= s) ∧ (s < 10) ∧
                s = p / 10 + q / 100 + r / 1000 + s / 10000
  count_possible_s_values :=
by
  sorry

end count_possible_s_values_l337_337234


namespace triangle_DEF_area_l337_337981

theorem triangle_DEF_area (
  (D E F L : Type) 
  [EuclideanGeometry Triangle DEF] 
  (h1 : Altitude D L (segment E F)) 
  (h2 : Length (segment D E) = 12) 
  (h3 : Length (segment E L) = 9) 
  (h4 : Length (segment E F) = 15)
) : 
  Area DEF = (45 * Real.sqrt 7) / 2 := 
sorry

end triangle_DEF_area_l337_337981


namespace transistors_in_2002_transistors_in_2010_l337_337332

-- Definitions based on the conditions
def mooresLawDoubling (initial_transistors : ℕ) (years : ℕ) : ℕ :=
  initial_transistors * 2^(years / 2)

-- Conditions
def initial_transistors := 2000000
def year_1992 := 1992
def year_2002 := 2002
def year_2010 := 2010

-- Questions translated into proof targets
theorem transistors_in_2002 : mooresLawDoubling initial_transistors (year_2002 - year_1992) = 64000000 := by
  sorry

theorem transistors_in_2010 : mooresLawDoubling (mooresLawDoubling initial_transistors (year_2002 - year_1992)) (year_2010 - year_2002) = 1024000000 := by
  sorry

end transistors_in_2002_transistors_in_2010_l337_337332


namespace range_of_a_l337_337763

-- Define the function
def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then log x / log a else a * x - 2

-- Prove the range of a such that for any x1 ≠ x2, (f(a, x1) - f(a, x2)) / (x1 - x2) > 0
theorem range_of_a :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) > 0) ↔ (1 < a ∧ a ≤ 2) :=
sorry

end range_of_a_l337_337763


namespace length_XY_l337_337921

-- Definitions
variables {PQ QR PR XQ YR XY : ℝ}
variables {P Q R S X Y : Type*}
variables [Nonempty X] [Nonempty Y] [Nonempty P] [Nonempty Q] [Nonempty R] [Nonempty S]

-- Given conditions
def isRectangle (PQ QR : ℝ) := PQ = 5 ∧ QR = 12

def perpendicular (XY QR : ℝ) (Q : Type*) := ∃ XY_perpendicular_to_QR, XY_perpendicular_to_QR

def point_on_segment (P XQ : Type*) := ∃ P_on_XQ, P_on_XQ

def point_on_segment (S YR : Type*) := ∃ S_on_YR, S_on_YR

-- Statement to prove
theorem length_XY (h_rectangle : isRectangle PQ QR)
                  (h_perpendicular : perpendicular XY QR Q)
                  (h_point_on_XQ : point_on_segment P XQ)
                  (h_point_on_YR : point_on_segment S YR) :
                  XY = 36.61667 :=
begin
  sorry
end

end length_XY_l337_337921


namespace false_proposition_l337_337435

-- Definitions based on conditions
def p := ∃ (x0 : ℝ), sin x0 ≥ 1
def q := ∀ (a b : ℝ), (ln a > ln b) ↔ (a > b ∧ b > 0)

-- The formal statement to prove that p ∧ q is false
theorem false_proposition : ¬ (p ∧ q) :=
sorry

end false_proposition_l337_337435


namespace pipe_fills_tank_without_leak_in_6_hours_l337_337643

theorem pipe_fills_tank_without_leak_in_6_hours
  (leak_drains_in_12_hours : True) -- The leak alone can empty the full tank in 12 hours
  (pipe_and_leak_fill_in_12_hours : True) -- With the leak, it takes 12 hours for the pipe to fill the tank
  : ∃ T : ℝ, T = 6 :=
by
  use 6
  sorry

end pipe_fills_tank_without_leak_in_6_hours_l337_337643


namespace subtract_base3_sum_eq_result_l337_337556

theorem subtract_base3_sum_eq_result :
  let a := 10 -- interpreted as 10_3
  let b := 1101 -- interpreted as 1101_3
  let c := 2102 -- interpreted as 2102_3
  let d := 212 -- interpreted as 212_3
  let sum := 1210 -- interpreted as the base 3 sum of a + b + c
  let result := 1101 -- interpreted as the final base 3 result
  sum - d = result :=
by sorry

end subtract_base3_sum_eq_result_l337_337556


namespace find_acute_angle_x_l337_337506

def a_parallel_b (x : ℝ) : Prop :=
  let a := (Real.sin x, 3 / 4)
  let b := (1 / 3, 1 / 2 * Real.cos x)
  b.1 * a.2 = a.1 * b.2

theorem find_acute_angle_x (x : ℝ) (h : a_parallel_b x) : x = Real.pi / 4 :=
by
  sorry

end find_acute_angle_x_l337_337506


namespace smallest_value_zero_l337_337453

open RealInnerProductSpace

-- Given conditions
variables (a b c : ℝ^3)
axiom unit_a : ∥a∥ = 1
axiom unit_b : ∥b∥ = 1
axiom unit_c : ∥c∥ = 1

-- Prove the question equals the answer
theorem smallest_value_zero : ∥a - b∥^2 + ∥b - c∥^2 + ∥c - a∥^2 = 0 := by
  sorry

end smallest_value_zero_l337_337453


namespace extreme_values_l337_337764

-- Define the function f(x) with symbolic constants a and b
def f (x a b : ℝ) : ℝ := x^3 - a * x^2 - b * x

-- Given conditions
def intersects_at_1_0 (a b : ℝ) : Prop := (f 1 a b = 0)
def derivative_at_1_0 (a b : ℝ) : Prop := (3 - 2 * a - b = 0)

-- Main theorem statement
theorem extreme_values (a b : ℝ) (h1 : intersects_at_1_0 a b) (h2 : derivative_at_1_0 a b) :
  (∀ x, f x a b ≤ 4 / 27) ∧ (∀ x, 0 ≤ f x a b) :=
sorry

end extreme_values_l337_337764


namespace compare_P_Q_l337_337750

noncomputable def P : ℝ := Real.sqrt 7 - 1
noncomputable def Q : ℝ := Real.sqrt 11 - Real.sqrt 5

theorem compare_P_Q : P > Q :=
sorry

end compare_P_Q_l337_337750


namespace necessary_and_sufficient_condition_l337_337162

open Real

variables {a b : Real} -- Assuming vectors in ℝ^n case
variables [Nonzero a] [Nonzero b]

-- Assumptions
def not_equals_neg_vector (a b : Real) : Prop := ¬ (a = b ∨ a = -b)

-- The statement to prove
theorem necessary_and_sufficient_condition
  (h₀ : not_equals_neg_vector a b) :
  (|a| = |b|) ↔ (a + b) * (a - b) = 0 :=
sorry

end necessary_and_sufficient_condition_l337_337162


namespace function_symmetry_about_half_l337_337696

notation "⌊" n "⌋" => Int.floor n

def g (x : ℝ) : ℝ :=
  |⌊2 * x⌋| - |⌊2 - 2 * x⌋|

theorem function_symmetry_about_half :
  ∀ x : ℝ, g (1 - x) = -g x :=
by
  intro x
  sorry

end function_symmetry_about_half_l337_337696


namespace tire_cost_l337_337213

theorem tire_cost (total_cost : ℕ) (number_of_tires : ℕ) (cost_per_tire : ℕ) 
    (h1 : total_cost = 240) 
    (h2 : number_of_tires = 4)
    (h3 : cost_per_tire = total_cost / number_of_tires) : 
    cost_per_tire = 60 :=
sorry

end tire_cost_l337_337213


namespace constant_is_5_variables_are_n_and_S_l337_337857

-- Define the conditions
def cost_per_box : ℕ := 5
def total_cost (n : ℕ) : ℕ := n * cost_per_box

-- Define the statement to be proved
-- constant is 5
theorem constant_is_5 : cost_per_box = 5 := 
by sorry

-- variables are n and S, where S is total_cost n
theorem variables_are_n_and_S (n : ℕ) : 
    ∃ S : ℕ, S = total_cost n :=
by sorry

end constant_is_5_variables_are_n_and_S_l337_337857


namespace max_possible_piles_l337_337253

-- Definitions
def is_stone_weight (w : ℕ) : Prop := 1 ≤ w ∧ w ≤ 25

def pile (p : List ℕ) : Prop := p.length = 2018 ∧ ∀ w ∈ p, is_stone_weight w

def total_weight (p : List ℕ) : ℕ := p.sum

def all_piles_diff_weights (piles : List (List ℕ)) : Prop := 
  ∀ (i j : ℕ), i < piles.length → j < piles.length → i ≠ j → 
  total_weight (piles.get i) ≠ total_weight (piles.get j)

def condition_removal (p1 p2 : List ℕ) : Prop :=
  let p1' := (p1.erase (p1.maximum!?)).erase (p1.minimum!?)
  let p2' := (p2.erase (p2.maximum!?)).erase (p2.minimum!?)
  in total_weight p1 < total_weight p2 → total_weight p1' > total_weight p2'

-- Problem statement
theorem max_possible_piles : ∃ (piles : List (List ℕ)), 
  piles.length = 12 ∧
  (∀ p ∈ piles, pile p) ∧
  all_piles_diff_weights piles ∧
  ∀ (i j : ℕ), i < piles.length → j < piles.length → i ≠ j → 
  condition_removal (piles.get i) (piles.get j) :=
sorry

end max_possible_piles_l337_337253


namespace find_height_of_frustum_l337_337328

theorem find_height_of_frustum (h hC h2 : ℝ) (r : ℝ) (ratio : ℝ) (HC : h = 6) (HhC: hC = 2) (Hr : r = 5) (HRatio : ratio = 1/2) : 
    ∃ h_f2 : ℝ, h_f2 ≈ 2.3 :=
by 
  sorry

end find_height_of_frustum_l337_337328


namespace percentage_of_number_l337_337192

theorem percentage_of_number (n : ℝ) (h : (1 / 4) * (1 / 3) * (2 / 5) * n = 16) : 0.4 * n = 192 :=
by 
  sorry

end percentage_of_number_l337_337192


namespace forty_percent_of_N_l337_337915

variable (N : ℕ)

theorem forty_percent_of_N (h : (1/4 : ℝ) * (1/3) * (2/5) * N = 35) : 0.40 * N = 420 := 
by
  -- The proof goes here
  sorry

end forty_percent_of_N_l337_337915


namespace pizza_cut_possible_l337_337537
-- Step 1: Import necessary library

-- Step 2: Define the problem statement
theorem pizza_cut_possible (N : ℕ) (h₁ : N = 201 ∨ N = 400) : 
  ∃ (cuts : ℕ), cuts ≤ 100 ∧ 
  ∀ (parts : set ℝ), parts.card = N ∧ (∀ part ∈ parts, part.area = 1) :=
by
  sorry

end pizza_cut_possible_l337_337537


namespace maximize_product_intersecting_circles_l337_337078

theorem maximize_product_intersecting_circles
  (circle1 circle2 : Circle)
  (M N : Point)
  (h1 : circle1.IntersectsAt M N)
  (h2 : circle2.IntersectsAt M N)
  (A B : Point)
  (h3 : LineSegment A B).Contains M
  (h4 : A ∈ circle1)
  (h5 : B ∈ circle2) :
  ∃ (α : ℝ), α = π - Angle(c1.center, M, A) - Angle(c2.center, M, B) ∧
  let θ := α / 2 in
  let segment := LineSegment.construct_by_angle N θ in
  AM * MB is maximized at angle θ := sorry

end maximize_product_intersecting_circles_l337_337078


namespace digit_of_fraction_l337_337983

theorem digit_of_fraction (n : ℕ) : (15 / 37 : ℝ) = 0.405 ∧ 415 % 3 = 1 → ∃ d : ℕ, d = 4 :=
by
  sorry

end digit_of_fraction_l337_337983


namespace exists_point_X_l337_337152

variables {O1 O2 A B P C D E L F X : Point}
variables (C1 C2 : Circle)

def congruent_circles (O1 O2 : Point) : Prop := ∃ r, C1 = Circle.mk O1 r ∧ C2 = Circle.mk O2 r
def midpoint (P E : Point) : Point := Point.mk ((P.x + E.x) / 2) ((P.y + E.y) / 2)
def symmetric_point (D mid_PE : Point) : Point := Point.mk (2 * mid_PE.x - D.x) (2 * mid_PE.y - D.y)

theorem exists_point_X (h1 : congruent_circles O1 O2)
                      (h2 : C1 ∩ C2 = {A, B})
                      (h3 : P ∈ arc A B C2)
                      (h4 : A ∈ P)
                      (h5 : C ∈ P)
                      (h6 : D ∈ CB ∩ C2)
                      (h7 : angle_bisector (∠CAD) ∩ C1 = {E})
                      (h8 : angle_bisector (∠CAD) ∩ C2 = {L})
                      (h9 : F = symmetric_point D (midpoint P E)) :
    ∃ X, ∠ X F L = 30 ∧ ∠ X D C = 30 ∧ dist C X = dist O1 O2 := 
sorry

end exists_point_X_l337_337152


namespace triangle_area_LMN_is_4_l337_337984

def point := ℝ × ℝ

-- Define the points L, M, N
def L : point := (2, 3)
def M : point := (5, 1)
def N : point := (3, 5)

-- Define the function for the area of a triangle using the Shoelace formula.
def triangle_area (A B C : point) : ℝ :=
  |(A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2|

-- The theorem to prove the area of the triangle using the defined points.
theorem triangle_area_LMN_is_4 : triangle_area L M N = 4 :=
by 
  sorry

end triangle_area_LMN_is_4_l337_337984


namespace cistern_fill_time_l337_337288

theorem cistern_fill_time (F E : ℝ) (hF : F = 1/3) (hE : E = 1/6) : (1 / (F - E)) = 6 :=
by sorry

end cistern_fill_time_l337_337288


namespace part1_part2_l337_337524

noncomputable def S_n (n : ℕ) : ℕ := sorry
noncomputable def T_n (n : ℕ) : ℕ := sorry

axiom a1 (n : ℕ) : S_n 1 = 1
axiom cond1 (n : ℕ) : let a := (S_n (n + 1) - 2 * S_n n, S_n n) in
                      let b := (2, n) in
                      a.1 * b.2 = a.2 * b.1

theorem part1 (n : ℕ) : ∀ n, ∃ r, ∀ n, S_n n / n = r ^ (n - 1) := sorry

theorem part2 (n : ℕ) : T_n n = (n - 1) * 2 ^ n + 1 := sorry

end part1_part2_l337_337524


namespace initial_person_count_l337_337222

theorem initial_person_count
  (avg_weight_increase : ℝ)
  (weight_old_person : ℝ)
  (weight_new_person : ℝ)
  (h1 : avg_weight_increase = 4.2)
  (h2 : weight_old_person = 65)
  (h3 : weight_new_person = 98.6) :
  ∃ n : ℕ, weight_new_person - weight_old_person = avg_weight_increase * n ∧ n = 8 := 
by
  sorry

end initial_person_count_l337_337222


namespace exist_congruent_triangle_with_same_color_points_l337_337236

theorem exist_congruent_triangle_with_same_color_points (T : Triangle) (colors : Finset Color) (h_colors : colors.card = 1993) 
  (coloring : Plane → Color) (h_coloring : ∀ c ∈ colors, ∃ p : Plane, coloring p = c) :
  ∃ T' : Triangle, T' ≅ T ∧ (∀ i : Fin 3, ∃ p : Plane, p ∈ T'.side i ∧ ∃ c : Color, c ∈ colors ∧ coloring p = c) :=
sorry

end exist_congruent_triangle_with_same_color_points_l337_337236


namespace quadratic_real_roots_quadratic_product_of_roots_l337_337573

theorem quadratic_real_roots (m : ℝ) :
  (∃ x : ℝ, x^2 - 2 * m * x + m^2 + m - 3 = 0) ↔ m ≤ 3 := by
{
  sorry
}

theorem quadratic_product_of_roots (m : ℝ) (α β : ℝ) :
  α * β = 17 ∧ α^2 - 2 * m * α + m^2 + m - 3 = 0 ∧ β^2 - 2 * m * β + m^2 + m - 3 = 0 →
  m = -5 := by
{
  sorry
}

end quadratic_real_roots_quadratic_product_of_roots_l337_337573


namespace number_of_remaining_wolves_l337_337909

def prime_count_lt (n : ℕ) : ℕ :=
  (Finset.filter Nat.prime (Finset.range n)).card

def can_eat_sheep (i : ℕ) : Prop :=
  ∃ j : ℕ, j < 7 ∧ prime_count_lt i % 7 = j

def turns_into_sheep (i : ℕ) : Prop :=
  can_eat_sheep i

def wolves_that_remain : Finset ℕ :=
  (Finset.range 2017).filter (λ i, ¬ turns_into_sheep i)

theorem number_of_remaining_wolves : wolves_that_remain.card = 288 :=
  sorry

end number_of_remaining_wolves_l337_337909


namespace jose_completion_time_l337_337995

noncomputable def rate_jose : ℚ := 1 / 30
noncomputable def rate_jane : ℚ := 1 / 6

theorem jose_completion_time :
  ∀ (J A : ℚ), 
    (J + A = 1 / 5) ∧ (J = rate_jose) ∧ (A = rate_jane) → 
    (1 / J = 30) :=
by
  intros J A h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end jose_completion_time_l337_337995


namespace genevieve_cherries_purchase_l337_337735

theorem genevieve_cherries_purchase (cherries_cost_per_kg: ℝ) (genevieve_money: ℝ) (extra_money_needed: ℝ) (total_kg: ℝ) : 
  cherries_cost_per_kg = 8 → 
  genevieve_money = 1600 →
  extra_money_needed = 400 →
  total_kg = (genevieve_money + extra_money_needed) / cherries_cost_per_kg →
  total_kg = 250 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end genevieve_cherries_purchase_l337_337735


namespace problem_1_problem_2_l337_337460

theorem problem_1 (A B C : ℝ) (h_cond : (abs (B - A)) * (abs (C - A)) * (Real.cos A) = 3 * (abs (A - B)) * (abs (C - B)) * (Real.cos B)) : 
  (Real.tan B = 3 * Real.tan A) := 
sorry

theorem problem_2 (A B C : ℝ) (h_cosC : Real.cos C = Real.sqrt 5 / 5) (h_tanB : Real.tan B = 3 * Real.tan A) : 
  (A = Real.pi / 4) := 
sorry

end problem_1_problem_2_l337_337460


namespace perimeter_of_equilateral_triangle_l337_337871

theorem perimeter_of_equilateral_triangle (a : ℕ) (h1 : a = 12) (h2 : ∀ sides, sides = 3) : 
  3 * a = 36 := 
by
  sorry

end perimeter_of_equilateral_triangle_l337_337871


namespace largest_prime_factor_of_sum_l337_337327
open Nat

theorem largest_prime_factor_of_sum (seq : ℕ → ℕ) (n : ℕ) (digits_match : ∀ i, (seq i / 1000 % 10 = seq (i + 1) / 100 % 10) ∧ (seq i / 100 % 10 = seq (i + 1) / 10 % 10) ∧ (seq i / 10 % 10 = seq (i + 1) % 10) ∧ (seq i % 10 = seq (i + 1 + n) / 1000 % 10)) :
  101 ∣ ∑ i in range n, seq i → 101
:= sorry

end largest_prime_factor_of_sum_l337_337327


namespace root_in_interval_l337_337963

noncomputable def f (x : ℝ) := log x / log 3 + x - 3

theorem root_in_interval : ∃ x ∈ Ioo (2 : ℝ) 3, f x = 0 := by
  -- Proof is required here, omitted with sorry
  sorry

end root_in_interval_l337_337963


namespace pond_length_l337_337571

noncomputable def length_of_pond (L : ℝ) (W : ℝ) (PondArea : ℝ) : ℝ :=
  ℝ.sqrt PondArea

theorem pond_length
  (L W : ℝ)
  (h1 : L = 2 * W)
  (h2 : L = 96)
  (h3 : PondArea = 1 / 72 * (L * W)) : 
  length_of_pond L W PondArea = 8 :=
by
  sorry

end pond_length_l337_337571


namespace complex_sum_omega_l337_337164

noncomputable def ω : ℂ := sorry

axiom ω_pow_8_eq_1 : ω^8 = 1
axiom ω_ne_one : ω ≠ 1

theorem complex_sum_omega (ω : ℂ) (h1 : ω^8 = 1) (h2 : ω ≠ 1) :
  (ω^17 + ω^20 + ω^23 + ... + ω^65) = ω :=
sorry

end complex_sum_omega_l337_337164


namespace cube_root_less_than_5_l337_337089

theorem cube_root_less_than_5 :
  {n : ℕ | n > 0 ∧ (∃ m : ℝ, m^3 = n ∧ m < 5)}.finite.card = 124 :=
by
  sorry

end cube_root_less_than_5_l337_337089


namespace rotation_maps_triangles_l337_337977

/-- A proof that $n + p + q = 102$ if a $n$ degrees rotation clockwise around point $(p, q)$,
where $0 < n < 180$, maps $\triangle ABC$ with vertices $(0,0)$, $(4,0)$, $(0,3)$ onto 
$\triangle DEF$ with vertices $(8,10)$, $(8,6)$, $(11,10)$. -/
theorem rotation_maps_triangles (n p q : ℝ) 
  (h₀ : 0 < n) (h₁ : n < 180)
  (hpq_A : (p - q * real.tan (n * (real.pi / 180)) = 8) ∧ (q + p * real.tan (n * (real.pi / 180)) = 10))
  (hpq_B : (p + (0 - p) * real.cos (n * (real.pi / 180)) - (4 - q) * real.sin (n * (real.pi / 180)) = 8) ∧ 
           (q + (4 - q) * real.cos (n * (real.pi / 180)) + (0 - p) * real.sin (n * (real.pi / 180)) = 6)) :
  n + p + q = 102 :=
sorry

end rotation_maps_triangles_l337_337977


namespace train_cross_pole_time_l337_337621

variable (L : Real) (V : Real)

theorem train_cross_pole_time (hL : L = 110) (hV : V = 144) : 
  (110 / (144 * 1000 / 3600) = 2.75) := 
by
  sorry

end train_cross_pole_time_l337_337621


namespace problem_statement_l337_337297

theorem problem_statement : (-0.125 ^ 2006) * (8 ^ 2005) = -0.125 := by
  sorry

end problem_statement_l337_337297


namespace pascal_triangle_contains_53_only_once_l337_337802

theorem pascal_triangle_contains_53_only_once (n : ℕ) (k : ℕ) (h_prime : Nat.prime 53) :
  (n = 53 ∧ (k = 1 ∨ k = 52) ∨ 
   ∀ m < 53, Π l, Nat.binomial m l ≠ 53) ∧ 
  (n > 53 → (k = 0 ∨ k = n ∨ Π a b, a * 53 ≠ b * Nat.factorial (n - k + 1))) :=
sorry

end pascal_triangle_contains_53_only_once_l337_337802


namespace harmonic_mean_2_3_6_l337_337675

theorem harmonic_mean_2_3_6 : harmonic_mean [2, 3, 6] = 3 := by
  sorry

def harmonic_mean (lst : List ℕ) : ℚ :=
  let n := lst.length
  n / (lst.map (λ x => (1 : ℚ)/x)).sum

#eval harmonic_mean [2, 3, 6]

end harmonic_mean_2_3_6_l337_337675


namespace quadratic_equation_roots_l337_337105

-- Define the two numbers α and β such that their arithmetic and geometric means are given.
variables (α β : ℝ)

-- Arithmetic mean condition
def arithmetic_mean_condition : Prop := (α + β = 16)

-- Geometric mean condition
def geometric_mean_condition : Prop := (α * β = 225)

-- The quadratic equation with roots α and β
def quadratic_equation (x : ℝ) : ℝ := x^2 - 16 * x + 225

-- The proof statement
theorem quadratic_equation_roots (α β : ℝ) (h1 : arithmetic_mean_condition α β) (h2 : geometric_mean_condition α β) :
  ∃ x : ℝ, quadratic_equation x = 0 :=
sorry

end quadratic_equation_roots_l337_337105


namespace license_plate_palindrome_prob_l337_337184

/-!
  Many states use a sequence of three letters followed by a sequence of four digits 
  as their standard license-plate pattern. Given that each three-letter four-digit 
  arrangement is equally likely,
  the probability that such a license plate will contain at least one palindrome 
  (a three-letter arrangement or a four-digit arrangement that reads the same left-to-right 
  as it does right-to-left) is $\dfrac{25}{520}$.

  We need to prove that $m+n=545$ where $\frac{m}{n}$ is the simplified form of the probability.
-/

theorem license_plate_palindrome_prob :
  let n := 520
  let m := 25
  m + n = 545 :=
by
  -- Define the probability form and prove the sum m+n
  let prob := (25 : ℚ) / 520
  have key : m + n = 545 := rfl
  sorry

end license_plate_palindrome_prob_l337_337184


namespace pascal_contains_53_l337_337792

theorem pascal_contains_53 (n : ℕ) (h1 : Nat.Prime 53) (h2 : ∃ k, 1 ≤ k ∧ k ≤ 52 ∧ nat.choose 53 k = 53) (h3 : ∀ m < 53, ¬ (∃ k, 1 ≤ k ∧ k ≤ m - 1 ∧ nat.choose m k = 53)) (h4 : ∀ m > 53, ¬ (∃ k, 1 ≤ k ∧ k ≤ m - 1 ∧ nat.choose m k = 53)) : 
  (n = 53) → (n = 1) := 
by
  intros
  sorry

end pascal_contains_53_l337_337792


namespace subset_sum_not_11_l337_337555

theorem subset_sum_not_11 :
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  let pairs := [(1, 10), (2, 9), (3, 8), (4, 7), (5, 6)]
  ∃ T ⊆ S, 
    (∀ x y ∈ T, x ≠ y → x + y ≠ 11) ∧ 
    (T.card = 5) ∧ 
    (T.card = 32) :=
sorry

end subset_sum_not_11_l337_337555


namespace prime_divisor_property_l337_337905

-- Given conditions
variable (p k : ℕ)
variable (prime_p : Nat.Prime p)
variable (divisor_p : p ∣ (2 ^ (2 ^ k)) + 1)

-- The theorem we need to prove
theorem prime_divisor_property (p k : ℕ) (prime_p : Nat.Prime p) (divisor_p : p ∣ (2 ^ (2 ^ k)) + 1) : (2 ^ (k + 1)) ∣ (p - 1) := 
by 
  sorry

end prime_divisor_property_l337_337905


namespace sum_of_first_eight_terms_arithmetic_sequence_l337_337165

theorem sum_of_first_eight_terms_arithmetic_sequence 
  (a : ℕ → ℝ)
  (h1 : ∀ n, a (n + 1) = a n + d) 
  (h_condition1 : a 1 + a 3 + a 5 = 9) 
  (h_condition2 : a 6 = 9) : 
  ∑ n in range 8, a n = 48 := sorry

end sum_of_first_eight_terms_arithmetic_sequence_l337_337165


namespace problem_statement_l337_337069

theorem problem_statement (x : Fin 2019 → ℝ) 
    (h1 : ∀ n : Fin 2017, x (Fin.ofNat (n + 1))^2 ≤ x n * x (Fin.ofNat (n + 2))) 
    (h2 : (∏ n in Finset.range 2018, x (Fin.ofNat n)) = 1): 
    x (Fin.ofNat 1009) * x (Fin.ofNat 1010) ≤ 1 := 
sorry

end problem_statement_l337_337069


namespace general_solution_l337_337012

def is_general_solution (x y : ℝ) (C : ℝ) : Prop :=
  x^2 + 2 * x * y - y^2 = C^2

theorem general_solution (x y : ℝ) (C : ℝ) :
  (x + y) * (differential x) + (x - y) * (differential y) = 0 →
  is_general_solution x y C :=
sorry

end general_solution_l337_337012


namespace complement_of_M_in_U_l337_337183

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}

theorem complement_of_M_in_U : U \ M = {2, 3, 5} := by
  sorry

end complement_of_M_in_U_l337_337183


namespace father_ate_six_cookies_l337_337532

theorem father_ate_six_cookies :
  ∃ F : ℝ, (F + F / 2 + (F / 2 + 2) = 22) ∧ (F = 6) :=
begin
  -- skipping the proof
  sorry

end father_ate_six_cookies_l337_337532


namespace pizza_division_l337_337534

theorem pizza_division (N : ℕ) (hN : N = 201 ∨ N = 400) : 
  ∃ cuts: ℕ, cuts ≤ 100 ∧ is_possible_division N cuts := 
sorry

-- Assume the necessary definitions of is_possible_division
-- This means is_possible_division N cuts denotes whether the pizza can be divided into N equal parts using "cuts" cuts.

-- You can define is_possible_division mathematically later if needed
def is_possible_division (N : ℕ) (cuts : ℕ) : Prop := sorry

end pizza_division_l337_337534


namespace cover_triangle_with_isosceles_l337_337549

theorem cover_triangle_with_isosceles {A B C : ℝ} (area_ABC : ℝ) : 
  area_ABC = 1 → ∃ area_iso : ℝ, area_iso < Real.sqrt 2 ∧ 
  ∃ AC BC H : ℝ, is_isosceles_triangle AC BC H ∧ covers (Δ ABC) (Δ AC H) :=
by sorry

end cover_triangle_with_isosceles_l337_337549


namespace pascal_triangle_contains_53_only_once_l337_337803

theorem pascal_triangle_contains_53_only_once (n : ℕ) (k : ℕ) (h_prime : Nat.prime 53) :
  (n = 53 ∧ (k = 1 ∨ k = 52) ∨ 
   ∀ m < 53, Π l, Nat.binomial m l ≠ 53) ∧ 
  (n > 53 → (k = 0 ∨ k = n ∨ Π a b, a * 53 ≠ b * Nat.factorial (n - k + 1))) :=
sorry

end pascal_triangle_contains_53_only_once_l337_337803


namespace step_count_initial_l337_337285

theorem step_count_initial :
  ∃ (x y : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧ (11 * y - x = 64) ∧ (10 * x + y = 26) :=
by
  sorry

end step_count_initial_l337_337285


namespace puppy_ratios_l337_337316

theorem puppy_ratios :
  ∀(total_puppies : ℕ)(golden_retriever_females golden_retriever_males : ℕ)
   (labrador_females labrador_males : ℕ)(poodle_females poodle_males : ℕ)
   (beagle_females beagle_males : ℕ),
  total_puppies = golden_retriever_females + golden_retriever_males +
                  labrador_females + labrador_males +
                  poodle_females + poodle_males +
                  beagle_females + beagle_males →
  golden_retriever_females = 2 →
  golden_retriever_males = 4 →
  labrador_females = 1 →
  labrador_males = 3 →
  poodle_females = 3 →
  poodle_males = 2 →
  beagle_females = 1 →
  beagle_males = 2 →
  (golden_retriever_females / golden_retriever_males = 1 / 2) ∧
  (labrador_females / labrador_males = 1 / 3) ∧
  (poodle_females / poodle_males = 3 / 2) ∧
  (beagle_females / beagle_males = 1 / 2) ∧
  (7 / 11 = (golden_retriever_females + labrador_females + poodle_females + beagle_females) / 
            (golden_retriever_males + labrador_males + poodle_males + beagle_males)) :=
by intros;
   sorry

end puppy_ratios_l337_337316


namespace car_overtakes_buses_l337_337345

/-- 
  Buses leave the airport every 3 minutes. 
  A bus takes 60 minutes to travel from the airport to the city center. 
  A car takes 35 minutes to travel from the airport to the city center. 
  Prove that the car overtakes 8 buses on its way to the city center excluding the bus it left with.
--/
theorem car_overtakes_buses (arr_bus : ℕ) (arr_car : ℕ) (interval : ℕ) (diff : ℕ) : 
  interval = 3 → arr_bus = 60 → arr_car = 35 → diff = arr_bus - arr_car →
  ∃ n : ℕ, n = diff / interval ∧ n = 8 := by
  sorry

end car_overtakes_buses_l337_337345


namespace calculate_expression_l337_337678

theorem calculate_expression : (-1 : ℝ) ^ 2 + (1 / 3 : ℝ) ^ 0 = 2 := 
by
  sorry

end calculate_expression_l337_337678


namespace thirteenth_result_is_78_l337_337936

theorem thirteenth_result_is_78 
  (avg_25 : ℕ → ℕ)
  (Σ_25 : (∑ i in Finset.range 25, avg_25 i) = 25 * 18)
  (Σ_first_12 : ∑ i in Finset.range 12, avg_25 i = 12 * 14)
  (Σ_last_12 : ∑ i in Finset.range 12, avg_25 (i + 13) = 12 * 17) :
  avg_25 12 = 78 :=
by {
  -- Sum of all 25 results is 450
  have h1 : (12 * 14) + avg_25 12 + (12 * 17) = 25 * 18, by {
    rw [Σ_25, Σ_first_12, Σ_last_12],
  },
  -- Simplify the equation to find the 13th result
  have h2 : 168 + avg_25 12 + 204 = 450, from h1,
  have h3 : 372 + avg_25 12 = 450, by linarith,
  have h4 : avg_25 12 = 450 - 372, by linarith,
  exact h4
}

end thirteenth_result_is_78_l337_337936


namespace sufficient_but_not_necessary_condition_l337_337437

def line1 (a : ℝ) : ℝ × ℝ → Prop := λ p, a * p.1 + p.2 + 1 = 0
def line2 (a : ℝ) : ℝ × ℝ → Prop := λ p, 2 * p.1 + (a + 1) * p.2 + 3 = 0

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a = 1 → ∀ p, line1 a p → ∀ q, line2 a q → p.2 = -a * p.1 - 1 → q.2 = -2 / (a + 1) * q.1 - 3 / (a + 1)) ∧
  (∃ b, b ≠ 1 ∧ ∀ p, line1 b p → ∀ q, line2 b q → p.2 = -b * p.1 - 1 → q.2 = -2 / (b + 1) * q.1 - 3 / (b + 1)) := sorry

end sufficient_but_not_necessary_condition_l337_337437


namespace stratified_sampling_males_l337_337878

theorem stratified_sampling_males 
    (total_male : ℕ) 
    (total_female : ℕ) 
    (sample_size : ℕ) 
    (total_students : total_students = total_male + total_female) 
    (prob : ℚ := (sample_size : ℚ) / total_students) : 
    (male_sample : ℕ := total_male * prob.toNat) 
    (sample_size = 280) 
    (total_male = 560) 
    (total_female = 420) 
    : male_sample = 160 := 
by 
    sorry

end stratified_sampling_males_l337_337878


namespace pascal_triangle_contains_53_l337_337819

theorem pascal_triangle_contains_53 (n : ℕ) :
  (∃ k, binomial n k = 53) ↔ n = 53 := 
sorry

end pascal_triangle_contains_53_l337_337819


namespace possible_values_of_a_eq_1_and_6_l337_337071

noncomputable def A : set ℚ := {1/2, 3}
noncomputable def B (a : ℚ) : set ℚ := {x : ℚ | 2*x = a}

theorem possible_values_of_a_eq_1_and_6 : 
  (λ a, B a ⊆ A) = (λ a, a = 1 ∨ a = 6) :=
by
  sorry

end possible_values_of_a_eq_1_and_6_l337_337071


namespace points_earned_l337_337082

-- Definition of the conditions explicitly stated in the problem
def points_per_bag := 8
def total_bags := 4
def bags_not_recycled := 2

-- Calculation of bags recycled
def bags_recycled := total_bags - bags_not_recycled

-- The main theorem stating the proof equivalent
theorem points_earned : points_per_bag * bags_recycled = 16 := 
by
  sorry

end points_earned_l337_337082


namespace angle_between_AB1C_base_l337_337706
noncomputable def angle_between_planes (α β : ℝ) : ℝ :=
  real.arccos (real.sqrt (1 / real.tan α * 1 / real.tan β))

theorem angle_between_AB1C_base (α β : ℝ) :
  angle_between_planes α β = real.arccos (real.sqrt (real.cot α * real.cot β)) :=
sorry

end angle_between_AB1C_base_l337_337706


namespace reconstruct_triangles_diagonals_l337_337923

noncomputable def reconstruct_diagonals (n : ℕ) : Prop :=
∀ (polygon : ConvexPolygon n) (adjacent_triangles : Fin n → ℕ), 
  can_reconstruct_diagonals polygon adjacent_triangles

theorem reconstruct_triangles_diagonals (n : ℕ) (polygon : ConvexPolygon n) (adjacent_triangles : Fin n → ℕ) :
  ∀ n ≥ 3, reconstruct_diagonals n :=
by sorry

end reconstruct_triangles_diagonals_l337_337923


namespace Kyle_is_25_l337_337492

-- Definitions based on the conditions
def Tyson_age : Nat := 20
def Frederick_age : Nat := 2 * Tyson_age
def Julian_age : Nat := Frederick_age - 20
def Kyle_age : Nat := Julian_age + 5

-- The theorem to prove
theorem Kyle_is_25 : Kyle_age = 25 := by
  sorry

end Kyle_is_25_l337_337492


namespace sufficient_condition_not_necessary_condition_l337_337095

/--
\(a > 1\) is a sufficient but not necessary condition for \(\frac{1}{a} < 1\).
-/
theorem sufficient_condition (a : ℝ) (h : a > 1) : 1 / a < 1 :=
by
  sorry

theorem not_necessary_condition (a : ℝ) (h : 1 / a < 1) : a > 1 ∨ a < 0 :=
by
  sorry

end sufficient_condition_not_necessary_condition_l337_337095


namespace fractions_sum_multiplied_by_2_l337_337676

-- Define the given fractions
def a := 2 / 20
def b := 3 / 30
def c := 4 / 40

-- State that the sum of the fractions multiplied by 2 equals 0.6
theorem fractions_sum_multiplied_by_2 : (a + b + c) * 2 = 0.6 := 
sorry

end fractions_sum_multiplied_by_2_l337_337676


namespace locus_of_point_C_l337_337862

structure Point :=
  (x : ℝ)
  (y : ℝ)

def is_isosceles_triangle (A B C : Point) : Prop := 
  let AB := (A.x - B.x)^2 + (A.y - B.y)^2
  let AC := (A.x - C.x)^2 + (A.y - C.y)^2
  AB = AC

def circle_eqn (C : Point) : Prop :=
  C.x^2 + C.y^2 - 3 * C.x + C.y = 2

def not_points (C : Point) : Prop :=
  (C ≠ {x := 3, y := -2}) ∧ (C ≠ {x := 0, y := 1})

theorem locus_of_point_C :
  ∀ (A B C : Point),
    A = {x := 3, y := -2} →
    B = {x := 0, y := 1} →
    is_isosceles_triangle A B C →
    circle_eqn C ∧ not_points C :=
by
  intros A B C hA hB hIso
  sorry

end locus_of_point_C_l337_337862


namespace least_number_of_cans_l337_337618

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

theorem least_number_of_cans (Maaza Pepsi Sprite : ℕ) (hMaaza : Maaza = 20) (hPepsi : Pepsi = 144) (hSprite : Sprite = 368) :
  let g := gcd (gcd Maaza Pepsi) Sprite in
  ((Maaza / g) + (Pepsi / g) + (Sprite / g)) = 133 :=
by
  rw [hMaaza, hPepsi, hSprite]
  let g := gcd (gcd 20 144) 368
  have hg : g = 4 := by
    sorry
  have h1 : 20 / g = 5 := by
    sorry
  have h2 : 144 / g = 36 := by
    sorry
  have h3 : 368 / g = 92 := by
    sorry
  show 
    ((20 / g) + (144 / g) + (368 / g)) = 133
  rw [h1, h2, h3]
  exact Nat.add_assoc 5 36 92
  end

end least_number_of_cans_l337_337618


namespace asymptote_of_rational_function_l337_337357

noncomputable def horizontal_asymptote :=
  ∀ (x : ℝ), x → ∞ → (6 * x^2 - 2 * x - 7) / (4 * x^2 + 3 * x + 1) = 3 / 2

theorem asymptote_of_rational_function : horizontal_asymptote := by
  sorry

end asymptote_of_rational_function_l337_337357


namespace bus_speed_including_stoppages_l337_337367

theorem bus_speed_including_stoppages
  (speed_without_stoppages : ℝ)
  (stoppage_time : ℝ)
  (remaining_time_ratio : ℝ)
  (h1 : speed_without_stoppages = 12)
  (h2 : stoppage_time = 0.5)
  (h3 : remaining_time_ratio = 1 - stoppage_time) :
  (speed_without_stoppages * remaining_time_ratio) = 6 := 
by
  sorry

end bus_speed_including_stoppages_l337_337367


namespace number_of_sets_B_l337_337408

theorem number_of_sets_B (A : Set ℕ) (hA : A = {1, 2}) :
    ∃ (n : ℕ), n = 4 ∧ (∀ B : Set ℕ, A ∪ B = {1, 2} → B ⊆ A) := sorry

end number_of_sets_B_l337_337408


namespace same_biking_time_l337_337202

noncomputable def biking_time : ℝ := 620

variable (r : ℝ) (tr tj : ℝ)

-- Conditions
def rudolph_rate (r t : ℝ) := r = 50 / (t - 245)
def jennifer_rate (r t : ℝ) := (3 / 4) * r = 50 / (t - 120)

-- Prove that both have the same biking time
theorem same_biking_time (r : ℝ) :
    (∃ t : ℝ, rudolph_rate r t ∧ jennifer_rate r t) → tr = 620 ∧ tj = 620 := by
  intro ⟨t, hr, hj⟩
  specialize hr biking_time
  specialize hj biking_time
  -- proof exactly here
  sorry

end same_biking_time_l337_337202


namespace relationship_between_sums_l337_337407

-- Conditions: four distinct positive integers
variables {a b c d : ℕ}
-- additional conditions: positive integers
variables (a_pos : a > 0) (b_pos : b > 0) (c_pos : c > 0) (d_pos : d > 0)

-- Condition: a is the largest and d is the smallest
variables (a_largest : a > b ∧ a > c ∧ a > d)
variables (d_smallest : d < b ∧ d < c ∧ d < a)

-- Condition: a / b = c / d
variables (ratio_condition : a * d = b * c)

theorem relationship_between_sums :
  a + d > b + c :=
sorry

end relationship_between_sums_l337_337407


namespace smallest_period_monotone_decreasing_intervals_find_a_l337_337060

noncomputable def f (x a : ℝ) : ℝ := 
  sin (2 * x - π / 6) + sin (2 * x + π / 6) + 2 * cos x ^ 2 + a - 1

theorem smallest_period (a : ℝ) : 
  (∀ x, f (x + π) a = f x a) ∧ (∀ T > 0, (∀ x, f (x + T) a = f x a) → T ≥ π) :=
sorry

theorem monotone_decreasing_intervals (a : ℝ) : 
  ∀ k : ℤ, 
  ∀ x, (k * π + π / 6 ≤ x ∧ x ≤ k * π + 2 * π / 3) → 
  ∀ y ∈ Set.Icc (x - 1) x, f y a ≤ f x a :=
sorry

theorem find_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 (π / 2), f x a ≥ -2) ∧ 
  (∃ x ∈ Set.Icc 0 (π / 2), f x a = -2) ↔ a = -1 :=
sorry

end smallest_period_monotone_decreasing_intervals_find_a_l337_337060


namespace sqrt_inequality_l337_337752

theorem sqrt_inequality (a₁ a₂ b₁ b₂ : ℝ) : 
  sqrt (a₁^2 + a₂^2) + sqrt (b₁^2 + b₂^2) ≥ sqrt ((a₁ - b₁)^2 + (a₂ - b₂)^2) :=
by
  sorry

end sqrt_inequality_l337_337752


namespace multiple_of_puppies_l337_337254

theorem multiple_of_puppies:
  ∀ (x : ℤ), 
  ∃ k : ℤ, 
  (k = 32) ∧ (78 = x * k + 14) → x = 2 :=
by
  intros x h
  rcases h with ⟨k, hk, h_eq⟩
  specialize hk
  sorry

end multiple_of_puppies_l337_337254


namespace B_is_more_stable_l337_337303

-- Define the scores of A
def A_scores : List ℝ := [8, 7, 9, 7, 9]

-- Function to calculate the mean of a list of real numbers
def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

-- Function to calculate the variance of a list of real numbers
def variance (l : List ℝ) : ℝ :=
  let m := mean l
  l.map (λ x => (x - m) ^ 2).sum / l.length

-- Definitions of average and variance for B
def B_mean : ℝ := 8
def B_variance : ℝ := 0.4

-- Mean and variance for A
def A_mean := mean A_scores
def A_variance := variance A_scores

-- Lean 4 theorem statement proving the stability of B's performance
theorem B_is_more_stable : B_variance < A_variance := by
  sorry

end B_is_more_stable_l337_337303


namespace max_d_is_9_l337_337010

-- Define the 6-digit number of the form 8d8, 45e
def num (d e : ℕ) : ℕ :=
  800000 + 10000 * d + 800 + 450 + e

-- Define the conditions: the number is a multiple of 45, 0 ≤ d, e ≤ 9
def conditions (d e : ℕ) : Prop :=
  0 ≤ d ∧ d ≤ 9 ∧ 0 ≤ e ∧ e ≤ 9 ∧
  (num d e) % 45 = 0

-- Define the maximum value of d
noncomputable def max_d : ℕ :=
  9

-- The theorem statement to be proved
theorem max_d_is_9 :
  ∀ (d e : ℕ), conditions d e → d ≤ max_d :=
by
  sorry

end max_d_is_9_l337_337010


namespace first_part_amount_l337_337552

-- Given Definitions
def total_amount : ℝ := 3200
def interest_rate_part1 : ℝ := 0.03
def interest_rate_part2 : ℝ := 0.05
def total_interest : ℝ := 144

-- The problem to be proven
theorem first_part_amount : 
  ∃ (x : ℝ), 0.03 * x + 0.05 * (3200 - x) = 144 ∧ x = 800 :=
by
  sorry

end first_part_amount_l337_337552


namespace B_plus_amount_is_5_l337_337546

-- Define the conditions
def B_plus_reward (x : ℝ) : ℝ := x
def A_reward (x : ℝ) : ℝ := 2 * x
def A_plus_reward : ℝ := 15  -- Flate rate $15 for each A+
def max_reward (num_B_plus num_A num_A_plus : ℕ) (x : ℝ) : ℝ := 
  if num_A_plus ≥ 2 then 2 * (num_B_plus * B_plus_reward x + num_A * A_reward x) + num_A_plus * A_plus_reward
  else num_B_plus * B_plus_reward x + num_A * A_reward x + num_A_plus * A_plus_reward

-- The proof goal
theorem B_plus_amount_is_5 : 
  ∃ x, let num_A_plus := 2 in let num_A := 8 in let num_B_plus := 0 in 
  max_reward num_B_plus num_A num_A_plus x = 190 ∧ x = 5 :=
by 
  sorry

end B_plus_amount_is_5_l337_337546


namespace like_terms_exponent_l337_337836

theorem like_terms_exponent (a : ℝ) : (2 * a = a + 3) → a = 3 := 
by
  intros h
  -- Proof here
  sorry

end like_terms_exponent_l337_337836


namespace smallest_odd_prime_factor_l337_337718

theorem smallest_odd_prime_factor (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  (2023 ^ 8 + 1) % p = 0 ↔ p = 17 := 
by
  sorry

end smallest_odd_prime_factor_l337_337718


namespace perpendicular_line_eq_l337_337944

theorem perpendicular_line_eq :
  ∃ (m b : ℝ), (point : ℝ × ℝ) (line : ℝ × ℝ → ℝ) 
  (perpendicular : ℝ × ℝ → ℝ) (eq : ℝ × ℝ → ℝ),
  point = (0,1) ∧
  line (x,y) = 2*x - y + 1 ∧
  perpendicular (x,y) = y + (1/2)*x - b ∧
  eq = x + 2*y - 2 = 0,
  ∀ x y, line (x,y) = 0 → ∃ x' y', perpendicular (x',y') = 0 := 
sorry

end perpendicular_line_eq_l337_337944


namespace necessary_condition_l337_337097

theorem necessary_condition (a b : ℝ) : a > b → 2^(a + 1) > 2^b := 
by sorry

end necessary_condition_l337_337097


namespace inequality_must_hold_l337_337393

theorem inequality_must_hold (a b c : ℝ) (h : (a / c^2) > (b / c^2)) (hc : c ≠ 0) : a^2 > b^2 :=
sorry

end inequality_must_hold_l337_337393


namespace pascal_triangle_contains_53_once_l337_337830

theorem pascal_triangle_contains_53_once
  (h_prime : Nat.Prime 53) :
  ∃! n, ∃ k, n ≥ k ∧ n > 0 ∧ k > 0 ∧ Nat.choose n k = 53 := by
  sorry

end pascal_triangle_contains_53_once_l337_337830


namespace small_town_population_l337_337575

theorem small_town_population (total_population : ℕ) (equal_parts : ℕ) (male_parts : ℕ)
  (h1 : total_population = 450)
  (h2 : equal_parts = 4)
  (h3 : male_parts = 2) : 
  (male_parts * (total_population / equal_parts)) = 225 :=
by {
  have h4 : total_population / equal_parts = 112.5,
  { rw [h1, h2], norm_num },
  have h5 : male_parts * (total_population / equal_parts) = 2 * 112.5,
  { rw [h3, h4] },
  norm_num at h5,
  exact h5,
}

end small_town_population_l337_337575


namespace backward_clock_correct_times_per_day_l337_337659
-- Import full Mathlib libraries as suggested

-- State the main theorem to be proven
theorem backward_clock_correct_times_per_day
    (initial_time_synchronized : Prop)
    (backward_clock : Prop)
    (hour_hand_rotation : ∀ t : ℝ, t / 12 ∈ ℤ)
    (minute_hand_rotation : ∀ t : ℝ, t / 1 ∈ ℤ) :
    ∃ (correct_times : ℕ), correct_times = 4 :=
by
  -- Proof logic will go here
  sorry

end backward_clock_correct_times_per_day_l337_337659


namespace train_speed_is_72_l337_337331

def distance : ℕ := 24
def time_minutes : ℕ := 20
def time_hours : ℚ := time_minutes / 60
def speed := distance / time_hours

theorem train_speed_is_72 :
  speed = 72 := by
  sorry

end train_speed_is_72_l337_337331


namespace points_with_irrational_distances_and_rational_areas_l337_337510

theorem points_with_irrational_distances_and_rational_areas (n : ℕ) (hn : n ≥ 3) :
  ∃ (points : Fin n → ℚ × ℚ), 
    (∀ i j : Fin n, i ≠ j → (points i).1 ≠ (points j).1) ∧
    (∀ i j : Fin n, i ≠ j → ∃ d : ℚ, (d = (Real.sqrt (((points i).1 - (points j).1)^2 + ((points i).2 - (points j).2)^2))) ∧ irrational d) ∧
    (∀ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k →
      ∃ area : ℚ, 
        (area = 
          (1 / 2 : ℚ) *
          |(points i).1 * ((points j).2 - (points k).2) +
           (points j).1 * ((points k).2 - (points i).2) +
           (points k).1 * ((points i).2 - (points j).2)|) ∧
        area ∈ ℚ) :=
sorry

end points_with_irrational_distances_and_rational_areas_l337_337510


namespace balloons_remaining_proof_l337_337341

-- The initial number of balloons the clown has
def initial_balloons : ℕ := 3 * 12

-- The number of boys who buy balloons
def boys : ℕ := 3

-- The number of girls who buy balloons
def girls : ℕ := 12

-- The total number of children buying balloons
def total_children : ℕ := boys + girls

-- The remaining number of balloons after sales
def remaining_balloons (initial : ℕ) (sold : ℕ) : ℕ := initial - sold

-- Problem statement: Proof that the remaining balloons are 21 given the conditions
theorem balloons_remaining_proof : remaining_balloons initial_balloons total_children = 21 := sorry

end balloons_remaining_proof_l337_337341


namespace part_a_first_player_wins_part_b_first_player_wins_l337_337619

/-- Define the initial state of the game -/
structure GameState :=
(pile1 : Nat) (pile2 : Nat)

/-- Define the moves allowed in Part a) -/
inductive MoveA
| take_from_pile1 : MoveA
| take_from_pile2 : MoveA
| take_from_both  : MoveA

/-- Define the moves allowed in Part b) -/
inductive MoveB
| take_from_pile1 : MoveB
| take_from_pile2 : MoveB
| take_from_both  : MoveB
| transfer_to_pile2 : MoveB

/-- Define what it means for the first player to have a winning strategy in part a) -/
def first_player_wins_a (initial_state : GameState) : Prop := sorry

/-- Define what it means for the first player to have a winning strategy in part b) -/
def first_player_wins_b (initial_state : GameState) : Prop := sorry

/-- Theorem statement for part a) -/
theorem part_a_first_player_wins :
  first_player_wins_a ⟨7, 7⟩ :=
sorry

/-- Theorem statement for part b) -/
theorem part_b_first_player_wins :
  first_player_wins_b ⟨7, 7⟩ :=
sorry

end part_a_first_player_wins_part_b_first_player_wins_l337_337619


namespace correct_statements_l337_337402

-- Conditions
variable {a : ℕ → ℝ} -- Sequence of positive terms {a_n}
variable {S : ℕ → ℝ} -- Sum of the first n terms S_n
variable {m : ℝ}     -- Initial term a_1 = m

-- Given Recurrence Relation and Initial Value
axiom recurrence_relation (n : ℕ) : (a (n + 1) + 1) / 6 = (S n + n) / (S (n + 1) - S n + 1)
axiom initial_value : a 1 = m

-- To Prove:
theorem correct_statements :
  (a 2 = 5) ∧
  (∀ n, nat.odd n → a n = 3 * n + m - 3) ∧
  (∀ n, a 2 + (∑ k in range n, a (2 * k + 2)) = 3 * n ^ 2 + 2 * n) := 
sorry

end correct_statements_l337_337402


namespace log_sum_of_zeros_l337_337952

def f (x : ℝ) : ℝ :=
if x ≠ 3 then log (abs (x - 3)) / log 10 else 3

def F (x : ℝ) (b c : ℝ) : ℝ :=
(f x) ^ 2 + b * (f x) + c

theorem log_sum_of_zeros (b c x1 x2 x3 : ℝ) (h : F x1 b c = 0 ∧ F x2 b c = 0 ∧ F x3 b c = 0) :
  ln (x1 + x2 + x3) = 2 * ln 3 := by
  sorry

end log_sum_of_zeros_l337_337952


namespace binary_trailing_zeros_l337_337777

theorem binary_trailing_zeros (n : ℕ) :
  (n * 1024 + (4 * 64 + 2)).binary_repr.count_trailing_zeros >= 9 :=
by
  sorry

end binary_trailing_zeros_l337_337777


namespace perpendicular_lines_with_foot_l337_337955

theorem perpendicular_lines_with_foot (n : ℝ) : 
  (∀ x y, 10 * x + 4 * y - 2 = 0 ↔ 2 * x - 5 * y + n = 0) ∧
  (2 * 1 - 5 * (-2) + n = 0) → n = -12 := 
by sorry

end perpendicular_lines_with_foot_l337_337955


namespace shortest_time_to_equal_digits_l337_337250

-- Define the time representation and conditions
def all_digits_equal (hours : ℕ) (minutes : ℕ) : Prop :=
  let h1 := hours / 10 in
  let h2 := hours % 10 in
  let m1 := minutes / 10 in
  let m2 := minutes % 10 in
  h1 = h2 ∧ h2 = m1 ∧ m1 = m2

def starting_time := (3, 33)

-- The target time (hours and minutes) when all digits are equal
def target_time := (4, 44)

-- Calculate the time difference in minutes
def time_difference (start : ℕ × ℕ) (target : ℕ × ℕ) : ℕ :=
  let (sh, sm) := start in
  let (th, tm) := target in
  (th - sh) * 60 + (tm - sm)

-- The proof statement
theorem shortest_time_to_equal_digits :
  time_difference starting_time target_time = 71 :=
by
  -- Define the start and target times explicitly
  let start := starting_time
  let target := target_time
  -- Calculate the time difference and show it's 71 minutes
  show time_difference start target = 71
  sorry

end shortest_time_to_equal_digits_l337_337250


namespace cross_section_area_of_pyramid_l337_337039

def regular_pyramid_base_area (side_length : ℝ) : ℝ := 
  (sqrt 3) / 4 * side_length^2

def cross_section_area_through_P (height : ℝ) (side_length : ℝ) (AP_ratio : ℝ): ℝ :=
  (sqrt 3) / 9  -- because ratio results in scaling down area by (AP_ratio / (1 + AP_ratio))²

theorem cross_section_area_of_pyramid (h : ℝ) (a : ℝ) (r : ℝ) (P : Point ℝ) :
  h = 3 ∧ a = 6 ∧ r = 8 →
  cross_section_area_through_P h a r = (sqrt 3) / 9 :=
by
  sorry

end cross_section_area_of_pyramid_l337_337039


namespace minimum_value_exists_l337_337229

theorem minimum_value_exists (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : 2 * m + n = 1) : 
  ∃ (min_val : ℝ), min_val = (3 + 2 * Real.sqrt 2) ∧ (1 / m + 1 / n ≥ min_val) :=
by {
  -- Proof will be provided here.
  sorry
}

end minimum_value_exists_l337_337229


namespace neither_odd_nor_even_l337_337333

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f (x)

def is_neither_odd_nor_even (f : ℝ → ℝ) : Prop :=
  ¬ (is_odd f) ∧ ¬ (is_even f)

theorem neither_odd_nor_even (h1: is_neither_odd_nor_even (λ x : ℝ, x^2 + sin x))
                             (h2: ¬ is_neither_odd_nor_even (λ x : ℝ, x^2 - cos x))
                             (h3: ¬ is_neither_odd_nor_even (λ x : ℝ, 2^x + 1/(2^x)))
                             (h4: ¬ is_neither_odd_nor_even (λ x : ℝ, x + sin (2 * x))) : 
  (λ x : ℝ, x^2 + sin x) = (λ x : ℝ, x^2 + sin x) :=
by
  sorry

end neither_odd_nor_even_l337_337333


namespace complement_intersection_l337_337075

open Set

variable (U A B : Set ℕ)
variable (hU : U = {0, 1, 2, 3, 4, 5, 6})
variable (hA : A = {1, 2})
variable (hB : B = {0, 2, 5})

theorem complement_intersection : compl A ∩ B = {0, 5} :=
by
  rw [compl_set_of]
  have h1 : compl A = {0, 3, 4, 5, 6}, from sorry
  rw [h1, hB]
  exact sorry

end complement_intersection_l337_337075


namespace floor_of_neg_sqrt_5_minus_1_l337_337982

def floor_largest_int_le (x : ℝ) := ⌊x⌋

theorem floor_of_neg_sqrt_5_minus_1 : floor_largest_int_le (-sqrt 5 - 1) = -4 :=
by
  -- The proof will go here.
  sorry

end floor_of_neg_sqrt_5_minus_1_l337_337982


namespace binomial_distributions_l337_337991

-- Define the conditions for each random variable
variables (n N M : ℕ) (p : ℝ)
variables (hMN : M < N) (hp : 0 ≤ p ∧ p ≤ 1)

-- Definitions of the random variables
def xi_A : ℕ → ProbabilityTheory.ProbabilitySpace ℕ := sorry -- Throws of a die
def xi_B : ℕ → ProbabilityTheory.ProbabilitySpace ℕ := sorry -- Shots to hit target
def xi_C : ℕ → ProbabilityTheory.ProbabilitySpace ℕ := sorry -- With replacement
def xi_D : ℕ → ProbabilityTheory.ProbabilitySpace ℕ := sorry -- Without replacement

-- Define the binomial distribution property
def is_binomial (xi : ℕ → ProbabilityTheory.ProbabilitySpace ℕ) (k : ℕ) (p : ℝ) :=
  ∃ n : ℕ, xi n = ProbabilityTheory.binomial n p

-- Theorem stating which variables follow the binomial distribution
theorem binomial_distributions :
  is_binomial xi_A n (1/3) ∧ is_binomial xi_C n (M/N) :=
by sorry

end binomial_distributions_l337_337991


namespace determine_k_l337_337356

theorem determine_k (k : ℤ) : (∀ n : ℤ, gcd (4 * n + 1) (k * n + 1) = 1) ↔ 
  (∃ m : ℕ, k = 4 + 2 ^ m ∨ k = 4 - 2 ^ m) :=
by
  sorry

end determine_k_l337_337356


namespace parametric_curve_length_l337_337375

noncomputable def curve_length : ℝ :=
  ∫ t in 0..(2 * Real.pi), Real.sqrt ((3 * Real.cos t) ^ 2 + (-3 * Real.sin t) ^ 2)

theorem parametric_curve_length :
  curve_length = 6 * Real.pi :=
by
  rw [curve_length]
  -- further steps of the formal proof would go here
  sorry

end parametric_curve_length_l337_337375


namespace smallest_n_value_l337_337215

theorem smallest_n_value (a b c m n : ℕ) (h1 : a + b + c = 2010) (h2 : c % 2 = 0) (h3 : factorial a * factorial b * factorial c = m * 10 ^ n) (h4 : ¬ (10 ∣ m)) : n = 501 :=
by
  sorry

end smallest_n_value_l337_337215


namespace find_9_athletes_l337_337860

-- Let's define the labels for the cities and events
def cities : Type := fin 5 -- Cities A1, A2, A3, A4, A5
def events : Type := fin 49 -- Events B1, B2, ..., B49
def gender : Type := bool -- True for Male, False for Female

-- Define participants as a function that gives us the gender of the athlete from city i in event j
def participants (c : cities) (e : events) : gender := sorry

theorem find_9_athletes : 
  ∃ (cities' : fin 3 → cities) (events' : fin 3 → events) (g : gender),
    ∀ (i j : fin 3), participants (cities' i) (events' j) = g := 
by 
  sorry

end find_9_athletes_l337_337860


namespace largest_possible_difference_l337_337666

def est_Anita : ℕ := 40000
def est_Bob : ℕ := 50000

def range_Albuquerque : set ℕ := { n | 32000 ≤ n ∧ n ≤ 48000 }
def range_Buffalo : set ℕ := { n | 41667 ≤ n ∧ n ≤ 62500 }

theorem largest_possible_difference :
  ∀ (A ∈ range_Albuquerque) (B ∈ range_Buffalo), 
  max_diff A B = 31000 :=
sorry

end largest_possible_difference_l337_337666


namespace triangle_proof_l337_337461

-- Defining the problem in the context of a triangle ABC with specific side and angle relations
variables {A B C : ℝ} {a b c : ℝ}
variables (D : ℝ) (AB AC : ℝ) (AD : ℝ)

-- Defining conditions in the context of the problem
def condition1 (A B C a b c : ℝ) : Prop := sin A + cos^2 ((B + C) / 2) = 1
def condition2 (a : ℝ) := a = 4 * sqrt 2
def condition3 (b : ℝ) := b = 5
def condition4 (cosA : ℝ) := cosA = 3/5
def condition5 (AD : ℝ) := AD = (1 / 4) * AB + (3 / 4) * AC

theorem triangle_proof 
  (h1 : condition1 A B C a b c) 
  (h2 : condition2 a)
  (h3 : condition3 b)
  (h4 : condition4 (cos A))
  (h5 : condition5 AD):
  sin A = 4 / 5 ∧ AD = 5 :=
by {
  sorry
}

end triangle_proof_l337_337461


namespace bike_growth_equation_l337_337206

-- Declare the parameters
variables (b1 b3 : ℕ) (x : ℝ)
-- Define the conditions
def condition1 : b1 = 1000 := sorry
def condition2 : b3 = b1 + 440 := sorry

-- Define the proposition to be proved
theorem bike_growth_equation (cond1 : b1 = 1000) (cond2 : b3 = b1 + 440) :
  b1 * (1 + x)^2 = b3 :=
sorry

end bike_growth_equation_l337_337206


namespace sum_of_palindromic_primes_less_than_60_l337_337347

/-- Definition of a prime number -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

/-- Definition of a palindromic number -/
def is_palindromic (n : ℕ) : Prop :=
  let s := n.to_string in s = s.reverse

/-- Sum of all palindromic primes less than 60 -/
theorem sum_of_palindromic_primes_less_than_60 :
  (∑ p in finset.filter (λ n, is_prime n ∧ is_palindromic n) (finset.range 60), p) = 55 :=
by sorry

end sum_of_palindromic_primes_less_than_60_l337_337347


namespace valid_passwords_count_l337_337391

def total_passwords : Nat := 10 ^ 5
def restricted_passwords : Nat := 10

theorem valid_passwords_count : total_passwords - restricted_passwords = 99990 := by
  sorry

end valid_passwords_count_l337_337391


namespace find_range_of_x_l337_337753

variable (f : ℝ → ℝ) (x : ℝ)

-- Assume f is an increasing function on [-1, 1]
def is_increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ b ∧ a ≤ y ∧ y ≤ b ∧ x ≤ y → f x ≤ f y

-- Main theorem statement based on the problem
theorem find_range_of_x (h_increasing : is_increasing_on_interval f (-1) 1)
                        (h_condition : f (x - 1) < f (1 - 3 * x)) :
  0 ≤ x ∧ x < (1 / 2) :=
sorry

end find_range_of_x_l337_337753


namespace quadratic_roots_inequality_solution_set_l337_337298

-- Problem 1 statement
theorem quadratic_roots : 
  (∀ x : ℝ, x^2 - 4 * x + 1 = 0 ↔ x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3) := 
by
  sorry

-- Problem 2 statement
theorem inequality_solution_set :
  (∀ x : ℝ, (x - 2 * (x - 1) ≤ 1 ∧ (1 + x) / 3 > x - 1) ↔ -1 ≤ x ∧ x < 2) :=
by
  sorry

end quadratic_roots_inequality_solution_set_l337_337298


namespace simplify_expression_l337_337211

-- Define the given expression
def given_expr (x y : ℝ) := 3 * x + 4 * y + 5 * x^2 + 2 - (8 - 5 * x - 3 * y - 2 * x^2)

-- Define the expected simplified expression
def simplified_expr (x y : ℝ) := 7 * x^2 + 8 * x + 7 * y - 6

-- Theorem statement to prove the equivalence of the expressions
theorem simplify_expression (x y : ℝ) : 
  given_expr x y = simplified_expr x y := sorry

end simplify_expression_l337_337211


namespace prove_f_x1_minus_f_x2_lt_zero_l337_337420

variable {f : ℝ → ℝ}

-- Define even function
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Specify that f is decreasing for x < 0
def decreasing_on_negative (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < 0 → y < 0 → x < y → f x > f y

theorem prove_f_x1_minus_f_x2_lt_zero (hx1x2 : |x1| < |x2|)
  (h_even : even_function f)
  (h_decreasing : decreasing_on_negative f) :
  f x1 - f x2 < 0 :=
sorry

end prove_f_x1_minus_f_x2_lt_zero_l337_337420


namespace min_score_needed_l337_337710

theorem min_score_needed 
  (s1 s2 s3 s4 s5 : ℕ)
  (next_test_goal_increment : ℕ)
  (current_scores_sum : ℕ)
  (desired_average : ℕ)
  (total_tests : ℕ)
  (required_total_sum : ℕ)
  (required_next_score : ℕ)
  (current_scores : s1 = 88 ∧ s2 = 92 ∧ s3 = 75 ∧ s4 = 85 ∧ s5 = 80)
  (increment_eq : next_test_goal_increment = 5)
  (current_sum_eq : current_scores_sum = s1 + s2 + s3 + s4 + s5)
  (desired_average_eq : desired_average = (current_scores_sum / 5) + next_test_goal_increment)
  (total_tests_eq : total_tests = 6)
  (required_total_sum_eq : required_total_sum = desired_average * total_tests)
  (required_next_score_eq : required_next_score = required_total_sum - current_scores_sum) :
  required_next_score = 114 := by
    sorry

end min_score_needed_l337_337710


namespace winnie_keeps_10_lollipops_l337_337613

def winnie_keep_lollipops : Prop :=
  let cherry := 72
  let wintergreen := 89
  let grape := 23
  let shrimp_cocktail := 316
  let total_lollipops := cherry + wintergreen + grape + shrimp_cocktail
  let friends := 14
  let lollipops_per_friend := total_lollipops / friends
  let winnie_keeps := total_lollipops % friends
  winnie_keeps = 10

theorem winnie_keeps_10_lollipops : winnie_keep_lollipops := by
  sorry

end winnie_keeps_10_lollipops_l337_337613


namespace reflection_coordinates_l337_337940

-- Define the original coordinates of point M
def original_point : (ℝ × ℝ) := (3, -4)

-- Define the function to reflect a point across the x-axis
def reflect_across_x_axis (p: ℝ × ℝ) : (ℝ × ℝ) :=
  (p.1, -p.2)

-- State the theorem to prove the coordinates after reflection
theorem reflection_coordinates :
  reflect_across_x_axis original_point = (3, 4) :=
by
  sorry

end reflection_coordinates_l337_337940


namespace wire_length_and_square_side_length_l337_337597

theorem wire_length_and_square_side_length (l w : ℕ) (hl : l = 12) (hw : w = 8) :
    let wire_length := (l + w) * 2 in
    wire_length = 40 ∧ wire_length / 4 = 10 :=
by
  -- Definitions of the wire length and square side length
  let wire_length := (l + w) * 2
  have h1 : wire_length = 40 := by
    rw [hl, hw]
    calc
      (12 + 8) * 2 = 20 * 2 := by norm_num
                   ... = 40   := by norm_num
    done
  have h2 : wire_length / 4 = 10 := by
    rw [h1]
    calc
      40 / 4 = 10 := by norm_num
    done
  exact ⟨h1, h2⟩


end wire_length_and_square_side_length_l337_337597


namespace first_term_geometric_sequence_l337_337719

theorem first_term_geometric_sequence (a r : ℕ) (h1 : r = 3) (h2 : a * r^4 = 81) : a = 1 :=
by
  sorry

end first_term_geometric_sequence_l337_337719


namespace math_problem_l337_337687

theorem math_problem : (-1: ℝ)^2 + (1/3: ℝ)^0 = 2 := by
  sorry

end math_problem_l337_337687


namespace pascal_triangle_contains_53_l337_337822

theorem pascal_triangle_contains_53 (n : ℕ) :
  (∃ k, binomial n k = 53) ↔ n = 53 := 
sorry

end pascal_triangle_contains_53_l337_337822


namespace logarithmic_inequality_l337_337026

noncomputable def a : ℝ := 2 -- placeholder value, since we assume a > 1 and monotonicity property
def x (a : ℝ) := log a (sqrt 2) + 1/2 * log a 3
def y (a : ℝ) := 1/2 * log a 5
def z (a : ℝ) := log a (sqrt 21) - log a (sqrt 3)

theorem logarithmic_inequality (a_gt_one : a > 1) : z a > x a > y a :=
by
  sorry

end logarithmic_inequality_l337_337026


namespace graph_symmetric_y_axis_f_increasing_on_nonnegative_f_range_l337_337432

noncomputable def f (x : ℝ) : ℝ := Real.logBase 4 (1 + 4^x) - 0.5 * x

theorem graph_symmetric_y_axis : ∀ x : ℝ, f (-x) = f x :=
by
  sorry

theorem f_increasing_on_nonnegative : ∀ x₁ x₂ : ℝ, 0 ≤ x₁ → x₁ < x₂ → f x₁ < f x₂ :=
by
  sorry

theorem f_range : ∀ y : ℝ, ∃ x : ℝ, f x = y ↔ y ≥ 0.5 :=
by
  sorry

end graph_symmetric_y_axis_f_increasing_on_nonnegative_f_range_l337_337432


namespace clock_strikes_total_l337_337833

theorem clock_strikes_total {n : ℕ} (h1 : n = 12) :
  let hourly_strikes := (n * (n + 1)) / 2 in
  let daily_hourly_strikes := 2 * hourly_strikes in
  let half_hourly_strikes := 24 in
  daily_hourly_strikes + half_hourly_strikes = 180 :=
by
  let hourly_strikes := (n * (n + 1)) / 2
  let daily_hourly_strikes := 2 * hourly_strikes
  let half_hourly_strikes := 24
  have h : 2 * (12 * (12 + 1)) / 2 + 24 = 180,
  { simp },
  show daily_hourly_strikes + half_hourly_strikes = 180, from h

#eval clock_strikes_total rfl

end clock_strikes_total_l337_337833


namespace min_value_of_f_range_of_a_range_of_x_l337_337951

open Real

def f (x a : ℝ) := cos x ^ 2 + a * sin x + a + 1

def g (a : ℝ) : ℝ :=
  if a ≥ 0 then 1 else 2 * a + 1

theorem min_value_of_f (a : ℝ) : ∃ m, ∀ x, f x a ≥ m ∧ g a = m :=
sorry

theorem range_of_a : ∃ a, (∀ x : ℝ, f x a ≥ 0) ↔ a ≥ (-1 / 2) :=
sorry

theorem range_of_x : ∃ x, (∀ a ∈ Icc (-2 : ℝ) 0, f x a ≥ 0) ↔
                      ∃ k : ℤ, 2 * k * π - π ≤ x ∧ x ≤ 2 * k * π :=
sorry

end min_value_of_f_range_of_a_range_of_x_l337_337951


namespace pascal_triangle_contains_53_only_once_l337_337798

theorem pascal_triangle_contains_53_only_once (n : ℕ) (k : ℕ) (h_prime : Nat.prime 53) :
  (n = 53 ∧ (k = 1 ∨ k = 52) ∨ 
   ∀ m < 53, Π l, Nat.binomial m l ≠ 53) ∧ 
  (n > 53 → (k = 0 ∨ k = n ∨ Π a b, a * 53 ≠ b * Nat.factorial (n - k + 1))) :=
sorry

end pascal_triangle_contains_53_only_once_l337_337798


namespace sine_of_angle_l337_337442

variable {V : Type} [InnerProductSpace ℝ V] (a b : V)

def angle_sine (θ : ℝ) : Prop :=
  sin θ = sqrt 3 / 2

theorem sine_of_angle (h1 : ⟪a, a⟫ + ⟪a, b⟫ = 3)
  (h2 : ∥a∥ = 2)
  (h3 : ∥b∥ = 1) :
  ∃ θ : ℝ, angle_sine θ :=
begin
  sorry
end

end sine_of_angle_l337_337442


namespace sum_binomials_6_l337_337252

theorem sum_binomials_6 :
  ∑ k in finset.range 6, nat.choose 6 k - nat.choose 6 0 - nat.choose 6 6 = 62 := by
sorry

end sum_binomials_6_l337_337252


namespace balance_balls_l337_337191

-- Define the weights of the balls as variables
variables (B R O S : ℝ)

-- Given conditions
axiom h1 : R = 2 * B
axiom h2 : O = (7 / 3) * B
axiom h3 : S = (5 / 3) * B

-- Statement to prove
theorem balance_balls :
  (5 * R + 3 * O + 4 * S) = (71 / 3) * B :=
by {
  -- The proof is omitted
  sorry
}

end balance_balls_l337_337191


namespace find_p_l337_337873

-- Define the coordinates of the points
structure Point where
  x : Real
  y : Real

def Q := Point.mk 0 15
def A := Point.mk 3 15
def B := Point.mk 15 0
def O := Point.mk 0 0
def C (p : Real) := Point.mk 0 p

-- Given the area of triangle ABC and the coordinates of Q, A, B, O, and C, prove that p = 12.75
theorem find_p (p : Real) (h_area_ABC : 36 = 36) (h_Q : Q = Point.mk 0 15)
                (h_A : A = Point.mk 3 15) (h_B : B = Point.mk 15 0) 
                (h_O : O = Point.mk 0 0) : p = 12.75 := 
sorry

end find_p_l337_337873


namespace angle_in_third_quadrant_l337_337025

theorem angle_in_third_quadrant (α : ℝ) (h1 : Real.sin α < 0) (h2 : Real.cos α < 0) : 
  ∃ k : ℤ, α = (2 * k + 1) * Real.pi + β ∧ β ∈ Set.Ioo (0 : ℝ) Real.pi :=
by
  sorry

end angle_in_third_quadrant_l337_337025


namespace candles_to_new_five_oz_l337_337445

theorem candles_to_new_five_oz 
  (h_wax_percent: ℝ)
  (h_candles_20oz_count: ℕ) 
  (h_candles_5oz_count: ℕ) 
  (h_candles_1oz_count: ℕ) 
  (h_candles_20oz_wax: ℝ) 
  (h_candles_5oz_wax: ℝ)
  (h_candles_1oz_wax: ℝ):
  h_wax_percent = 0.10 →
  h_candles_20oz_count = 5 →
  h_candles_5oz_count = 5 → 
  h_candles_1oz_count = 25 →
  h_candles_20oz_wax = 20 →
  h_candles_5oz_wax = 5 →
  h_candles_1oz_wax = 1 →
  (h_wax_percent * h_candles_20oz_wax * h_candles_20oz_count + 
   h_wax_percent * h_candles_5oz_wax * h_candles_5oz_count + 
   h_wax_percent * h_candles_1oz_wax * h_candles_1oz_count) / 5 = 3 :=
by
  sorry

end candles_to_new_five_oz_l337_337445


namespace number_of_tangents_l337_337641

noncomputable def point_on_parabola (x y : ℝ) : Prop :=
  y^2 = -8 * x

noncomputable def is_tangent (l : ℝ → ℝ) : Prop :=
  ∀ x1 x2 y1 y2, point_on_parabola x1 y1 → point_on_parabola x2 y2 → 
  l(x1) = y1 → l(x2) = y2 → (x1 = x2 ∧ y1 = y2)

theorem number_of_tangents : ∀ (l : ℝ → ℝ), 
  (l (-2) = -4) → 
  ∃ n : ℕ, n = 2 ∧ 
  (∃ x1 y1, point_on_parabola x1 y1 ∧ l(x1) = y1) :=
sorry

end number_of_tangents_l337_337641


namespace form_triangle_condition_right_angled_triangle_condition_l337_337079

def vector (α : Type*) := α × α
noncomputable def oa : vector ℝ := ⟨2, -1⟩
noncomputable def ob : vector ℝ := ⟨3, 2⟩
noncomputable def oc (m : ℝ) : vector ℝ := ⟨m, 2 * m + 1⟩

def vector_sub (v1 v2 : vector ℝ) : vector ℝ := ⟨v1.1 - v2.1, v1.2 - v2.2⟩
def vector_dot (v1 v2 : vector ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem form_triangle_condition (m : ℝ) : 
  ¬ ((vector_sub ob oa).1 * (vector_sub (oc m) oa).2 = (vector_sub ob oa).2 * (vector_sub (oc m) oa).1) ↔ m ≠ 8 :=
sorry

theorem right_angled_triangle_condition (m : ℝ) : 
  (vector_dot (vector_sub ob oa) (vector_sub (oc m) oa) = 0 ∨ 
   vector_dot (vector_sub ob oa) (vector_sub (oc m) ob) = 0 ∨ 
   vector_dot (vector_sub (oc m) oa) (vector_sub (oc m) ob) = 0) ↔ 
  (m = -4/7 ∨ m = 6/7) :=
sorry

end form_triangle_condition_right_angled_triangle_condition_l337_337079


namespace vikki_tax_deduction_percentage_l337_337267

/- Definitions and conditions -/
def hours_worked : ℝ := 42
def hourly_pay_rate : ℝ := 10
def insurance_rate : ℝ := 0.05
def union_dues : ℝ := 5
def take_home_pay : ℝ := 310
def gross_earnings := hours_worked * hourly_pay_rate
def insurance_cover_deduction := insurance_rate * gross_earnings
def total_deductions_excl_tax := insurance_cover_deduction + union_dues
def total_deductions_incl_tax := gross_earnings - take_home_pay
def tax_deduction := total_deductions_incl_tax - total_deductions_excl_tax

/- The proof statement, asserting the main theorem -/
theorem vikki_tax_deduction_percentage :
  (tax_deduction / gross_earnings) * 100 = 20 :=
by
  sorry

end vikki_tax_deduction_percentage_l337_337267


namespace total_water_in_heaters_l337_337270

theorem total_water_in_heaters (wallace_capacity : ℕ) (catherine_capacity : ℕ) 
(wallace_water : ℕ) (catherine_water : ℕ) :
  wallace_capacity = 40 →
  (wallace_water = (3 * wallace_capacity) / 4) →
  wallace_capacity = 2 * catherine_capacity →
  (catherine_water = (3 * catherine_capacity) / 4) →
  wallace_water + catherine_water = 45 :=
by
  sorry

end total_water_in_heaters_l337_337270


namespace knight_coloring_proof_l337_337273

def knight_moves (x y : ℕ) : set (ℕ × ℕ) :=
{ (x + 2, y + 1), (x + 2, y - 1), (x - 2, y + 1), (x - 2, y - 1),
  (x + 1, y + 2), (x + 1, y - 2), (x - 1, y + 2), (x - 1, y - 2) }

def knight_unreachable (grid_size m : ℕ) : Prop :=
∃ (coloring : ℕ × ℕ → bool),
  (∀ (x y : ℕ), x < grid_size → y < grid_size → ¬coloring (x, y) → 
    (∀ (u v : ℕ), (u, v) ∈ knight_moves x y → u < grid_size → v < grid_size → coloring (u, v))
  ) ∧
  (coloring.fst.count (λ p, coloring p = tt) ≥ m)

theorem knight_coloring_proof :
  knight_unreachable 65 2112 := sorry

end knight_coloring_proof_l337_337273


namespace taco_castle_trucks_l337_337256

theorem taco_castle_trucks :
  ∀ (ford_trucks toyota_trucks vw_bugs dodge_trucks : ℕ),
    (ford_trucks = 2 * toyota_trucks) →
    (vw_bugs = toyota_trucks / 2) →
    (vw_bugs = 5) →
    (dodge_trucks = 60) →
    ford_trucks / dodge_trucks = 1 / 3 := 
by
  intros ford_trucks toyota_trucks vw_bugs dodge_trucks
  intros h1 h2 h3 h4
  sorry

end taco_castle_trucks_l337_337256


namespace odd_prime_div_product_plus_one_iff_div_square_minus_5_l337_337628

theorem odd_prime_div_product_plus_one_iff_div_square_minus_5 (p : ℕ) 
  (hp : Nat.Prime p) (oddp : p % 2 = 1) : 
  (∃ n : ℤ, p ∣ n*(n+1)*(n+2)*(n+3) + 1) ↔ (∃ m : ℤ, p ∣ m^2 - 5) := 
by 
  sorry

end odd_prime_div_product_plus_one_iff_div_square_minus_5_l337_337628


namespace area_enclosed_by_f_and_g_l337_337261

noncomputable def f (x : ℝ) : ℝ := Real.sin x
noncomputable def g (x : ℝ) : ℝ := Real.sin (x + π / 3)

open IntervalIntegral
open Real

theorem area_enclosed_by_f_and_g :
  ∫ x in (π / 3)..(4 * π / 3), (f x - g x) = 2 := by
  sorry

end area_enclosed_by_f_and_g_l337_337261


namespace john_total_distance_l337_337485

-- Define the conditions
def john_speed_alone : ℝ := 4 -- miles per hour
def john_speed_with_dog : ℝ := 6 -- miles per hour
def time_with_dog : ℝ := 0.5 -- hours
def time_alone : ℝ := 0.5 -- hours

-- Calculate distances based on conditions and prove the total distance
theorem john_total_distance : 
  john_speed_with_dog * time_with_dog + john_speed_alone * time_alone = 5 := 
by 
  calc
    john_speed_with_dog * time_with_dog + john_speed_alone * time_alone
    = 6 * 0.5 + 4 * 0.5 : by sorry
    ... = 3 + 2 : by sorry
    ... = 5 : by sorry

end john_total_distance_l337_337485


namespace sum_areas_frequency_distribution_histogram_l337_337855

theorem sum_areas_frequency_distribution_histogram :
  ∀ (rectangles : List ℝ), (∀ r ∈ rectangles, 0 ≤ r ∧ r ≤ 1) → rectangles.sum = 1 := 
  by
    intro rectangles h
    sorry

end sum_areas_frequency_distribution_histogram_l337_337855


namespace virus_diameter_scientific_notation_l337_337224

theorem virus_diameter_scientific_notation : 
    (0.0000012 : ℝ) = 1.2 * 10 ^ (-6) :=
by
  sorry

end virus_diameter_scientific_notation_l337_337224


namespace infinitely_many_n_with_Qn_gt_Qn1_l337_337021

open Nat

-- Define the main function Q(n) representing the LCM of n, n+1, ..., n+k
def Q (n k : ℕ) : ℕ := lcm.list (list.range (n + k + 1)).drop n

theorem infinitely_many_n_with_Qn_gt_Qn1 (k : ℕ) (hk : k > 1) :
  ∃ᶠ n in at_top, Q n k > Q (n + 1) k :=
begin
  sorry
end

end infinitely_many_n_with_Qn_gt_Qn1_l337_337021


namespace common_tangent_and_inequality_l337_337066

noncomputable def f (x : ℝ) := Real.log (1 + x)
noncomputable def g (x : ℝ) := x - (1 / 2) * x^2 + (1 / 3) * x^3

theorem common_tangent_and_inequality :
  -- Condition: common tangent at (0, 0)
  (∀ x, deriv f x = deriv g x) →
  -- Condition: values of a and b found to be 0 and 1 respectively
  (∀ x, f x ≤ g x) :=
by
  intro h
  sorry

end common_tangent_and_inequality_l337_337066


namespace sara_height_correct_l337_337554

variable (Roy_height : ℕ)
variable (Joe_height : ℕ)
variable (Sara_height : ℕ)

def problem_conditions (Roy_height Joe_height Sara_height : ℕ) : Prop :=
  Roy_height = 36 ∧
  Joe_height = Roy_height + 3 ∧
  Sara_height = Joe_height + 6

theorem sara_height_correct (Roy_height Joe_height Sara_height : ℕ) :
  problem_conditions Roy_height Joe_height Sara_height → Sara_height = 45 := by
  sorry

end sara_height_correct_l337_337554


namespace find_m_value_l337_337773

theorem find_m_value (m : ℝ) (A B : set ℝ) (hA : A = {m, 1}) (hB : B = {m^2, -1}) (hAB : A = B) : m = -1 :=
by
  sorry

end find_m_value_l337_337773


namespace expected_inspection_fee_l337_337754
-- Import the necessary Lean libraries

-- Define the conditions and theorem statement
theorem expected_inspection_fee :
  (let prob_2000 := (2 / 5) * (1 / 4) in
   let prob_3000 := ((2 / 5) * (3 / 4) * (1 / 3)) + ((3 / 5) * (2 / 4) * (1 / 3)) + ((3 / 5) * (2 / 4) * (1 / 3)) in
   let prob_4000 := 1 - prob_2000 - prob_3000 in
   let expected_fee := 2000 * prob_2000 + 3000 * prob_3000 + 4000 * prob_4000 in
   expected_fee = 3500)
:= 
begin
  sorry
end

end expected_inspection_fee_l337_337754


namespace simplify_and_evaluate_l337_337927

noncomputable def a := 2 * Real.sqrt 3 + 3
noncomputable def expr := (1 - 1 / (a - 2)) / ((a ^ 2 - 6 * a + 9) / (2 * a - 4))

theorem simplify_and_evaluate : expr = Real.sqrt 3 / 3 := by
  sorry

end simplify_and_evaluate_l337_337927


namespace coefficient_of_x8y2_l337_337872

theorem coefficient_of_x8y2 :
  let term1 := (1 / x^2)
  let term2 := (3 / y)
  let expansion := (x^2 - y)^7
  let coeff1 := 21 * (x ^ 10) * (y ^ 2) * (-1)
  let coeff2 := 35 * (3 / y) * (x ^ 8) * (y ^ 3)
  let comb := coeff1 + coeff2
  comb = -84 * x ^ 8 * y ^ 2 := by
  sorry

end coefficient_of_x8y2_l337_337872


namespace determine_number_of_quarters_l337_337653

def number_of_coins (Q D : ℕ) : Prop := Q + D = 23

def total_value (Q D : ℕ) : Prop := 25 * Q + 10 * D = 335

theorem determine_number_of_quarters (Q D : ℕ) 
  (h1 : number_of_coins Q D) 
  (h2 : total_value Q D) : 
  Q = 7 :=
by
  -- Equating and simplifying using h2, we find 15Q = 105, hence Q = 7
  sorry

end determine_number_of_quarters_l337_337653


namespace paint_first_level_paint_entire_tower_l337_337997

/-- Problem Part (a) - Painting the first level --/
theorem paint_first_level : 
  let colors := {Blue, Yellow, Brown}
  ∃ (front_colors back_colors left_colors right_colors : colors),
    (front_colors ≠ back_colors) ∧
    (left_colors.1 ≠ front_colors) ∧ (left_colors.2 ≠ left_colors.1) ∧ (left_colors.2 ≠ back_colors) ∧
    (right_colors.1 ≠ front_colors) ∧ (right_colors.2 ≠ right_colors.1) ∧ (right_colors.2 ≠ back_colors) ∧
    ((3 : ℕ) * (2 : ℕ) * (4 : ℕ) * (4 : ℕ) = 96) :=
sorry

/-- Problem Part (b) - Painting the entire tower --/
theorem paint_entire_tower :
  let colors := {Blue, Yellow, Brown}
  let first_level_ways := (3 : ℕ) * (2 : ℕ) * (4 : ℕ) * (4 : ℕ)
  first_level_ways = 96 →
  let second_level_ways := first_level_ways
  let top_level_ways := (3 : ℕ)
  (top_level_ways * second_level_ways * first_level_ways) = 27648 :=
sorry

end paint_first_level_paint_entire_tower_l337_337997


namespace probability_red_or_white_is_11_over_13_l337_337632

-- Given data
def total_marbles : ℕ := 60
def blue_marbles : ℕ := 5
def red_marbles : ℕ := 9
def white_marbles : ℕ := total_marbles - blue_marbles - red_marbles

def blue_size : ℕ := 2
def red_size : ℕ := 1
def white_size : ℕ := 1

-- Total size value of all marbles
def total_size_value : ℕ := (blue_size * blue_marbles) + (red_size * red_marbles) + (white_size * white_marbles)

-- Probability of selecting a red or white marble
def probability_red_or_white : ℚ := (red_size * red_marbles + white_size * white_marbles) / total_size_value

-- Theorem to prove
theorem probability_red_or_white_is_11_over_13 : probability_red_or_white = 11 / 13 :=
by sorry

end probability_red_or_white_is_11_over_13_l337_337632


namespace a_10_value_l337_337243

def sequence (a : ℕ → ℕ) :=
  ∀ n : ℕ, n > 0 → a (n + 1) = a n + a 2

def initial_value (a : ℕ → ℕ) :=
  a 3 = 6

theorem a_10_value
  {a : ℕ → ℕ}
  (h_seq : sequence a)
  (h_init : initial_value a) :
  a 10 = 27 :=
sorry

end a_10_value_l337_337243


namespace probability_not_next_to_each_other_l337_337121

-- Define the context and conditions.
variable (chairs : Fin 10) 
variable (seats : Finset (Fin 10))

-- Define the chairs Mary and James are not allowed to sit in.
def not_allowed_chairs : Finset (Fin 10) := {0, 9}

-- Define the set of available chairs for Mary and James.
def available_chairs : Finset (Fin 10) := Finset.erase (Finset.erase (Finset.univ) 0) 9

-- Define pairs that result in Mary and James sitting next to each other.
def adjacent_pairs : Finset (Fin (10 × 10)) :=
  (Finset.filter (λ p : Fin (10 × 10), p.1.val.succ = p.2.val + 1 ∨ p.1.val = p.2.val.succ)
    (Finset.product available_chairs available_chairs))

-- Define the total ways to choose 2 chairs from the available chairs.
def total_ways : ℕ := Nat.choose 8 2

-- Define the number of ways they can sit next to each other.
def ways_next_to : ℕ := adjacent_pairs.card

-- Define the probability they sit next to each other.
def prob_next_to : ℚ := ways_next_to / total_ways

-- Define the probability they do not sit next to each other.
def prob_not_next_to : ℚ := 1 - prob_next_to

-- The theorem we want to prove.
theorem probability_not_next_to_each_other : prob_not_next_to = 3 / 4 := sorry

end probability_not_next_to_each_other_l337_337121


namespace find_x_l337_337776

def vector_a (x : ℝ) : ℝ × ℝ × ℝ := (x, 2, -2)
def vector_b : ℝ × ℝ × ℝ := (3, -4, 2)

def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

theorem find_x (x : ℝ) (h : dot_product (vector_a x) vector_b = 0) : x = 4 :=
by
  sorry

end find_x_l337_337776


namespace solve_system_l337_337931

theorem solve_system:
  ∀ (x y z : ℝ),
  sin x + 2 * sin (x + y + z) = 0 →
  sin y + 3 * sin (x + y + z) = 0 →
  sin z + 4 * sin (x + y + z) = 0 →
  ∃ (n l m : ℤ), x = n * Real.pi ∧ y = l * Real.pi ∧ z = m * Real.pi :=
by
  intros x y z h1 h2 h3
  sorry

end solve_system_l337_337931


namespace number_of_elements_not_dividing_quotient_l337_337885

theorem number_of_elements_not_dividing_quotient (S : Set ℕ) (hS : ∀ n, n ∈ S ↔ (1 ≤ n ∧ n ≤ 2017)) 
  (L : ℕ) (hL : L = nat.lcm (set.to_finset S).val) : 
  (S.filter (λ n, ¬ n ∣ L / 2016)).card = 44 := 
sorry

end number_of_elements_not_dividing_quotient_l337_337885


namespace rounding_estimate_l337_337856

variables (a b c d : ℕ) [fact (a > 0)] [fact (b > 0)] [fact (c > 0)] [fact (d > 0)]

theorem rounding_estimate:
  let exact_val := (d * a / b) + c in
  let estimated_val := (d * (a - 1) / (b + 1)) + (c - 1) in
  estimated_val < exact_val := 
sorry

end rounding_estimate_l337_337856


namespace discontinuous_points_countable_l337_337338

variable {X : Type} [MetricSpace X] [SeparableSpace X]

def lim_exists_at (f : X → ℝ) : Prop :=
  ∀ a : ℝ, ∃ l : ℝ, Tendsto f (𝓝 (a : X)) (𝓝 l)

theorem discontinuous_points_countable
  (f : X → ℝ) (h : lim_exists_at f) :
  {x | ¬ ContinuousAt f x}.Countable :=
begin
  sorry
end

end discontinuous_points_countable_l337_337338


namespace largest_value_of_b_l337_337508

theorem largest_value_of_b (b : ℚ) (h : (2 * b + 5) * (b - 1) = 6 * b) : b = 5 / 2 :=
by
  sorry

end largest_value_of_b_l337_337508


namespace expected_coins_100_rounds_l337_337996

noncomputable def expectedCoinsAfterGame (rounds : ℕ) (initialCoins : ℕ) : ℝ :=
  initialCoins * (101 / 100) ^ rounds

theorem expected_coins_100_rounds :
  expectedCoinsAfterGame 100 1 = (101 / 100 : ℝ) ^ 100 :=
by
  sorry

end expected_coins_100_rounds_l337_337996


namespace combined_total_meows_l337_337559

-- Define the meowing frequencies per minute
def meows_per_minute_cat1 := 3
def meows_per_minute_cat2 := 2 * meows_per_minute_cat1
def meows_per_minute_cat3 := (1 / 3) * meows_per_minute_cat2
def meows_per_minute_cat4 := 4
def meows_per_minute_cat5 := (60 / 45)
def meows_per_minute_cat6 := (5 / 2)
def meows_per_minute_cat7 := (1 / 2) * meows_per_minute_cat2
def meows_per_minute_cat8 := (6 / 3)
def meows_per_minute_cat9 := (2.5 / (90 / 60))
def meows_per_minute_cat10 := (3.5 / (75 / 60))

-- Calculate the total meows per minute
def total_meows_per_minute :=
  meows_per_minute_cat1 +
  meows_per_minute_cat2 +
  meows_per_minute_cat3 +
  meows_per_minute_cat4 +
  meows_per_minute_cat5 +
  meows_per_minute_cat6 +
  meows_per_minute_cat7 +
  meows_per_minute_cat8 +
  meows_per_minute_cat9 +
  meows_per_minute_cat10

-- Calculate the total meows in 7.5 minutes
def total_meows_in_7_5_minutes := total_meows_per_minute * 7.5

theorem combined_total_meows :
  total_meows_in_7_5_minutes.floor = 212 := by
  -- Here, we will put the proof steps to verify the theorem
  sorry

end combined_total_meows_l337_337559


namespace num_factorial_square_pairs_l337_337702

def is_sum_factorial_square (n m : ℕ) : Prop :=
  (Finset.range (n + 1)).sum (λ k => (Nat.factorial k)) = m ^ 2

theorem num_factorial_square_pairs : 
  (Finset.card 
    (Finset.filter 
      (λ (n_m : ℕ × ℕ), is_sum_factorial_square n_m.1 n_m.2) 
      (Finset.product 
        (Finset.range 4) -- since we only consider n < 4
        (Finset.range (4 ^ 2 + 1))))) = 2 :=
by
  sorry

end num_factorial_square_pairs_l337_337702


namespace xiao_liang_reaches_museum_l337_337284

noncomputable def xiao_liang_distance_to_museum : ℝ :=
  let science_museum := (200 * Real.sqrt 2, 200 * Real.sqrt 2)
  let initial_mistake := (-300 * Real.sqrt 2, 300 * Real.sqrt 2)
  let to_supermarket := (-100 * Real.sqrt 2, 500 * Real.sqrt 2)
  Real.sqrt ((science_museum.1 - to_supermarket.1)^2 + (science_museum.2 - to_supermarket.2)^2)

theorem xiao_liang_reaches_museum :
  xiao_liang_distance_to_museum = 600 :=
sorry

end xiao_liang_reaches_museum_l337_337284


namespace part1_part2a_part2b_l337_337064

-- Part 1
theorem part1 (x : ℝ) (h : 0 < x) : f 1 = (1 / 2 * (x^2 - 1) - log x) : f x :=
by sorry

-- Part 2
theorem part2a (x : ℝ) (y : ℝ) (h : 0 < x) : f' x = f' y -> x = y := 
by sorry

theorem part2b (m : ℝ) : 
  let f := λ x, (1 / 2) * m * (x^2 - 1) - log x in 
  m ≤ 0 ∧ (∀ x > 0, 0 < f x) ∧ 
  m > 0 -> 
  ∃ x, x = (1 / (sqrt m)) -> f x = (1 / 2) * log m + (1 / 2) - (1 / 2) * m := 
by sorry

end part1_part2a_part2b_l337_337064


namespace inscribed_rectangle_sides_l337_337122

theorem inscribed_rectangle_sides {a b c : ℕ} (h₀ : a = 3) (h₁ : b = 4) (h₂ : c = 5) (ratio : ℚ) (h_ratio : ratio = 1 / 3) :
  ∃ (x y : ℚ), x = 20 / 29 ∧ y = 60 / 29 ∧ x = ratio * y :=
by
  sorry

end inscribed_rectangle_sides_l337_337122


namespace simplify_fraction_l337_337926

theorem simplify_fraction : 
  (2 * Real.sqrt 6) / (Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5) = Real.sqrt 2 + Real.sqrt 3 - Real.sqrt 5 := 
by
  sorry

end simplify_fraction_l337_337926


namespace limit_is_one_over_2_pi_l337_337624

theorem limit_is_one_over_2_pi :
  (filter.tendsto (λ x : ℝ, (sqrt (1 + x) - 1) / (sin (π * (x + 2))) ) (nhds 0) (nhds (1 / (2 * π)))) :=
by sorry

end limit_is_one_over_2_pi_l337_337624


namespace math_problem_l337_337459

theorem math_problem (x : ℝ) (h : x + (1 / x) = 5) : x^2 + (1 / x^2) = 23 :=
sorry

end math_problem_l337_337459


namespace license_plate_count_l337_337463

theorem license_plate_count :
  let second_char_choices := 3 in
  let first_char_choices := 5 in
  let remaining_char_pos := 3 in
  let remaining_char_choices := 4 in
  first_char_choices * second_char_choices * (remaining_char_choices^remaining_char_pos) = 960 :=
by
  let second_char_choices := 3
  let first_char_choices := 5
  let remaining_char_pos := 3
  let remaining_char_choices := 4
  show first_char_choices * second_char_choices * (remaining_char_choices ^ remaining_char_pos) = 960 from sorry

end license_plate_count_l337_337463


namespace parametric_curve_length_l337_337373

noncomputable def length_parametric_curve : ℝ :=
  ∫ t in 0..(2 * Real.pi), sqrt ((3 * Real.cos t)^2 + (-3 * Real.sin t)^2)

theorem parametric_curve_length :
  length_parametric_curve = 6 * Real.pi :=
sorry

end parametric_curve_length_l337_337373


namespace quadratic_neq_l337_337458

theorem quadratic_neq (m : ℝ) : (m-2) ≠ 0 ↔ m ≠ 2 :=
sorry

end quadratic_neq_l337_337458


namespace largest_common_term_l337_337562

theorem largest_common_term (n m : ℕ) (k : ℕ) (a : ℕ) 
  (h1 : a = 7 + 7 * n) 
  (h2 : a = 8 + 12 * m) 
  (h3 : 56 + 84 * k < 500) : a = 476 :=
  sorry

end largest_common_term_l337_337562


namespace term_containing_x_squared_l337_337050

theorem term_containing_x_squared (n r : ℕ) (x : ℝ) (h₁ : nat.choose n 6 = nat.choose n 4) :
  ∃ (r : ℕ), r = 2 ∧ ∃ (m : ℕ), (∑ (k : ℕ) in finset.range (m + 1), (nat.choose n k) * (sqrt x)^(n - k) * ((-1/(3 * x))^k)) = (nat.choose n 2) * (sqrt x)^(n - 2) * (-1/(3 * x))^2 :=
by {
  sorry
}

end term_containing_x_squared_l337_337050


namespace count_triangles_l337_337834

-- Assuming the conditions are already defined and given as parameters  
-- Let's define a proposition to prove the solution

noncomputable def total_triangles_in_figure : ℕ := 68

-- Create the theorem statement:
theorem count_triangles : total_triangles_in_figure = 68 := 
by
  sorry

end count_triangles_l337_337834


namespace slant_asymptote_sum_l337_337724

theorem slant_asymptote_sum (x : ℝ) (hx : x ≠ 5) :
  (5 : ℝ) + (21 : ℝ) = 26 :=
by
  sorry

end slant_asymptote_sum_l337_337724


namespace bike_growth_equation_l337_337207

-- Declare the parameters
variables (b1 b3 : ℕ) (x : ℝ)
-- Define the conditions
def condition1 : b1 = 1000 := sorry
def condition2 : b3 = b1 + 440 := sorry

-- Define the proposition to be proved
theorem bike_growth_equation (cond1 : b1 = 1000) (cond2 : b3 = b1 + 440) :
  b1 * (1 + x)^2 = b3 :=
sorry

end bike_growth_equation_l337_337207


namespace minimum_disks_to_guarantee_fifteen_same_label_l337_337494

theorem minimum_disks_to_guarantee_fifteen_same_label :
  ∀ (disks : list ℕ) (n : ℕ),
    (∀ (lbl : ℕ), lbl ∈ disks → lbl ≤ 60) →
    (list.sum disks = 1830) →
    (∀ (lbl : ℕ), ((disks.count lbl) < 15)) →
    (list.length disks > 749) →
    ∃ (lbl : ℕ), (disks.count lbl) ≥ 15 :=
by { sorry }

end minimum_disks_to_guarantee_fifteen_same_label_l337_337494


namespace function_relationship_l337_337399

variable {A B : Type} [Nonempty A] [Nonempty B]
variable (f : A → B) 

def domain (f : A → B) : Set A := {a | ∃ b, f a = b}
def range (f : A → B) : Set B := {b | ∃ a, f a = b}

theorem function_relationship (M : Set A) (N : Set B) (hM : M = Set.univ)
                              (hN : N = range f) : M = Set.univ ∧ N ⊆ Set.univ :=
  sorry

end function_relationship_l337_337399


namespace least_a_divisible_by_10_l337_337456

theorem least_a_divisible_by_10 : ∃ a : ℕ, (a = 2) ∧ ((1995^a + 1996^a + 1997^a) % 10 = 0) := by
  exists 2
  split
  exact rfl
  sorry

end least_a_divisible_by_10_l337_337456


namespace least_months_to_owe_triple_l337_337916

theorem least_months_to_owe_triple (t : ℕ) (h : 1.06^t > 3) : t = 19 :=
by {
  -- We will skip the proof, as per instructions
  sorry
}

end least_months_to_owe_triple_l337_337916


namespace term_150_is_98_l337_337883

noncomputable def sequence_step (n : ℕ) : ℕ :=
if n < 15 then n * 7
else if n % 2 = 0 then n / 2
else n - 7

noncomputable def sequence : ℕ → ℕ
| 0 := 63
| (n + 1) := sequence_step (sequence n)

theorem term_150_is_98 : sequence 150 = 98 :=
sorry

end term_150_is_98_l337_337883


namespace g_3_2_plus_g_3_4_l337_337151

-- Define the piecewise function g(x, y)
def g (x y : ℝ) : ℝ :=
  if x + y ≤ 5 then (3 * x * y - x + 3) / (3 * x)
  else (x * y - y - 3) / (-3 * y)

-- Statement of the theorem to be proved
theorem g_3_2_plus_g_3_4 : g 3 2 + g 3 4 = 19 / 12 :=
by
  simp only [g]
  -- These steps will break the proof based on the piecewise definition
  have h₁ : g 3 2 = 2, by { simp [g], norm_num }
  have h₂ : g 3 4 = -5 / 12, by { simp [g], norm_num }
  rw [h₁, h₂]
  norm_num
  sorry

end g_3_2_plus_g_3_4_l337_337151


namespace bicycle_travel_distance_l337_337306

theorem bicycle_travel_distance 
  (front_tire_lifespan : ℕ)
  (rear_tire_lifespan : ℕ)
  (front_tire_lifespan = 5000)
  (rear_tire_lifespan = 3000)
  (tire_swap_condition : ℕ)
  (tire_swap_condition := front_tire_lifespan * rear_tire_lifespan / Nat.gcd front_tire_lifespan rear_tire_lifespan / 8) :
  tire_swap_condition = 3750 := 
sorry

end bicycle_travel_distance_l337_337306


namespace train_length_l337_337655

theorem train_length (speed_km_hr : ℝ) (time_sec : ℝ) (length_m : ℝ) 
  (h1 : speed_km_hr = 52) (h2 : time_sec = 9) (h3 : length_m = 129.96) : 
  length_m = (speed_km_hr * 1000 / 3600) * time_sec := 
sorry

end train_length_l337_337655


namespace even_function_a_is_neg_one_l337_337106

-- Define f and the condition that it is an even function
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define the given function f(x) = (x-1)*(x-a)
def f (a x : ℝ) : ℝ := (x - 1) * (x - a)

-- Statement to prove that if f is an even function, then a must be -1
theorem even_function_a_is_neg_one (a : ℝ) :
  is_even (f a) → a = -1 :=
by 
  intro h,
  sorry

end even_function_a_is_neg_one_l337_337106


namespace pascal_contains_53_l337_337795

theorem pascal_contains_53 (n : ℕ) (h1 : Nat.Prime 53) (h2 : ∃ k, 1 ≤ k ∧ k ≤ 52 ∧ nat.choose 53 k = 53) (h3 : ∀ m < 53, ¬ (∃ k, 1 ≤ k ∧ k ≤ m - 1 ∧ nat.choose m k = 53)) (h4 : ∀ m > 53, ¬ (∃ k, 1 ≤ k ∧ k ≤ m - 1 ∧ nat.choose m k = 53)) : 
  (n = 53) → (n = 1) := 
by
  intros
  sorry

end pascal_contains_53_l337_337795


namespace inverse_of_118_mod_119_l337_337368

theorem inverse_of_118_mod_119 : ∃ x : ℤ, (118 * x ≡ 1 [MOD 119]) ∧ (x ≡ 118 [MOD 119]) := 
by
  have h1 : 118 ≡ -1 [MOD 119] := by sorry
  have h2 : 118 * 118 ≡ 1 [MOD 119] := by
    calc
    118 * 118 ≡ (-1) * (-1) [MOD 119] : by rw [← h1]
    ... ≡ 1 [MOD 119] : by norm_num
  use 118
  constructor
  · exact h2
  · refl

end inverse_of_118_mod_119_l337_337368


namespace f_four_eq_zero_l337_337521

variable (f : ℝ → ℝ)

-- Assume the conditions of the function
def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g(x) = -g(-x)

def is_even_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g(x) = g(-x)

axiom f_odd : is_odd_function (λ x, f(2*x + 2))
axiom f_even : is_even_function (λ x, f(x + 1))

-- Statement to prove
theorem f_four_eq_zero : f 4 = 0 :=
sorry

end f_four_eq_zero_l337_337521


namespace four_points_on_circle_general_position_lines_l337_337180

-- Define the general setup of intersection points on lines and circles.
structure IntersectionOnCircle (lines : List Line) (circle : Circle) : Prop :=
  (n : ℕ)
  (h_lines : lines.length = n)
  (h_points_on_circle : ∀ (i j : ℕ) (h_i : i < n) (h_j : j < n), PointOnCircle (intersection (lines[i]) (lines[j])) circle)

-- Define the structural property for n lines given the general position and points on a circle.
theorem four_points_on_circle (l1 l2 l3 l4 : Line)
  (circle : Circle)
  (h : IntersectionOnCircle [l1, l2, l3, l4] circle) :
  ∀ (i j k : ℕ) (h_i : i < 4) (h_j : j < 4) (h_k : k < 4) (hne1 : i ≠ j) (hne2 : j ≠ k) (hne3 : i ≠ k),
  PointsLieOnCircle {p : Point | ∃ l ∈ [l1, l2, l3, l4].erase (l1,l2,l3,l4)[i], contactonCircle(p)} circle := sorry

-- General proof for odd and even number of lines.
theorem general_position_lines (lines : List Line) (circle : Circle)
  (h : IntersectionOnCircle lines circle) :
  (lines.length % 2 = 1 → ∃ point : Point, ∀ subset : List Line, subset.length = lines.length - 1 →
  ∃ subset_circle : Circle, PointsCoCircle subset circle point subset_circle) ∧
  (lines.length % 2 = 0 → ∃ circle' : Circle, ∀ subset : List Line, subset.length = lines.length - 1 →
  ∃ p : Point, PointsLieOnCircle subset circle') := sorry

end four_points_on_circle_general_position_lines_l337_337180


namespace cantor_bernstein_theorem_l337_337658

-- Let E and F be sets
variables {E F : Type} 

-- There exists an injection from E to F
variables (f : E → F) (hf : function.injective f)

-- There exists an injection from F to E
variables (g : F → E) (hg : function.injective g)

-- The Cantor-Bernstein theorem states that if there are injections from E to F and F to E, there is a bijection from E to F
theorem cantor_bernstein_theorem :
  (∃ (b : E → F), function.bijective b) :=
begin
  sorry
end

end cantor_bernstein_theorem_l337_337658


namespace area_of_quadrilateral_eq_sum_of_two_triangles_l337_337913

open EuclideanGeometry
open Polygon

theorem area_of_quadrilateral_eq_sum_of_two_triangles
  (A B C D M N : Point)
  (hquad: ConvexQuadrilateral A B C D)
  (hM: M ∈ Segment A B)
  (hN: N ∈ Segment C D)
  (hratio: SegmentRatio A M B = SegmentRatio C N D) :
  area_of_quadrilateral A B N D = area_of_triangle B M C + area_of_triangle A M D :=
by
  sorry

end area_of_quadrilateral_eq_sum_of_two_triangles_l337_337913


namespace solve_system_of_equations_l337_337932

theorem solve_system_of_equations :
  ∃ x y : ℝ, (2 * x - 5 * y = -1) ∧ (-4 * x + y = -7) ∧ (x = 2) ∧ (y = 1) :=
by
  -- proof omitted
  sorry

end solve_system_of_equations_l337_337932


namespace kanul_initial_amount_l337_337487

-- Definition based on the problem conditions
def spent_on_raw_materials : ℝ := 3000
def spent_on_machinery : ℝ := 2000
def spent_on_labor : ℝ := 1000
def percent_spent : ℝ := 0.15

-- Definition of the total amount initially had by Kanul
def total_amount_initial (X : ℝ) : Prop :=
  spent_on_raw_materials + spent_on_machinery + percent_spent * X + spent_on_labor = X

-- Theorem stating the conclusion based on the given conditions
theorem kanul_initial_amount : ∃ X : ℝ, total_amount_initial X ∧ X = 7058.82 :=
by {
  sorry
}

end kanul_initial_amount_l337_337487


namespace combined_variance_is_178_l337_337958

noncomputable def average_weight_A := 60
noncomputable def variance_A := 100
noncomputable def average_weight_B := 64
noncomputable def variance_B := 200
noncomputable def ratio_A_B := (1, 3)

theorem combined_variance_is_178 :
  let nA := ratio_A_B.1
  let nB := ratio_A_B.2
  let avg_comb := (nA * average_weight_A + nB * average_weight_B) / (nA + nB)
  let var_comb := (nA * (variance_A + (average_weight_A - avg_comb)^2) + 
                   nB * (variance_B + (average_weight_B - avg_comb)^2)) / 
                   (nA + nB)
  var_comb = 178 := 
by
  sorry

end combined_variance_is_178_l337_337958


namespace pascal_contains_53_l337_337797

theorem pascal_contains_53 (n : ℕ) (h1 : Nat.Prime 53) (h2 : ∃ k, 1 ≤ k ∧ k ≤ 52 ∧ nat.choose 53 k = 53) (h3 : ∀ m < 53, ¬ (∃ k, 1 ≤ k ∧ k ≤ m - 1 ∧ nat.choose m k = 53)) (h4 : ∀ m > 53, ¬ (∃ k, 1 ≤ k ∧ k ≤ m - 1 ∧ nat.choose m k = 53)) : 
  (n = 53) → (n = 1) := 
by
  intros
  sorry

end pascal_contains_53_l337_337797


namespace pascals_triangle_53_rows_l337_337787

theorem pascals_triangle_53_rows : 
  ∃! row, (∃ k, 1 ≤ k ∧ k ≤ row ∧ 53 = Nat.choose row k) ∧ 
          (∀ k, 1 ≤ k ∧ k ≤ row → 53 = Nat.choose row k → row = 53) :=
sorry

end pascals_triangle_53_rows_l337_337787


namespace four_digit_number_l337_337656

-- Definitions of the conditions
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

-- Statement of the theorem
theorem four_digit_number (x y : ℕ) (hx : is_two_digit x) (hy : is_two_digit y) :
    (100 * x + y) = 1000 * x + y := sorry

end four_digit_number_l337_337656


namespace distance_from_home_to_school_is_correct_l337_337286

-- Definitions of the given conditions
def number_of_pedals : ℕ := 1500
def wheel_diameter_in_inches : ℝ := 26
def inch_to_meter : ℝ := 0.0254
def chainring_teeth : ℕ := 48
def freewheel_teeth : ℕ := 16
def pi_approx : ℝ := 3.14

-- Conversion of inches to meters for the wheel diameter
def wheel_diameter_in_meters : ℝ := wheel_diameter_in_inches * inch_to_meter

-- The distance Xiaoqiang covers from his home to school
def distance_home_to_school : ℝ :=
  let pedal_revolutions := number_of_pedals / 2
  let wheel_revolutions := pedal_revolutions * (chainring_teeth / freewheel_teeth)
  let wheel_circumference := pi_approx * wheel_diameter_in_meters
  wheel_revolutions * wheel_circumference

theorem distance_from_home_to_school_is_correct :
  distance_home_to_school ≈ 4665 :=
by
  sorry

end distance_from_home_to_school_is_correct_l337_337286


namespace box_internal_volume_l337_337483

def length_in_inches := 26
def width_in_inches := 26
def height_in_inches := 14
def wall_thickness_in_inches := 1

def internal_length_in_inches := length_in_inches - 2 * wall_thickness_in_inches
def internal_width_in_inches := width_in_inches - 2 * wall_thickness_in_inches
def internal_height_in_inches := height_in_inches - 2 * wall_thickness_in_inches

def length_in_feet := internal_length_in_inches / 12
def width_in_feet := internal_width_in_inches / 12
def height_in_feet := internal_height_in_inches / 12

def internal_volume_in_cubic_feet := length_in_feet * width_in_feet * height_in_feet

theorem box_internal_volume:
  internal_volume_in_cubic_feet = 4 :=
begin
  -- Proof steps will go here
  sorry
end

end box_internal_volume_l337_337483


namespace number_of_rows_containing_53_l337_337816

theorem number_of_rows_containing_53 (h_prime_53 : Nat.Prime 53) : 
  ∃! n, (n = 53 ∧ ∃ k, k ≥ 0 ∧ k ≤ n ∧ Nat.choose n k = 53) :=
by 
  sorry

end number_of_rows_containing_53_l337_337816


namespace pascal_triangle_contains_53_only_once_l337_337800

theorem pascal_triangle_contains_53_only_once (n : ℕ) (k : ℕ) (h_prime : Nat.prime 53) :
  (n = 53 ∧ (k = 1 ∨ k = 52) ∨ 
   ∀ m < 53, Π l, Nat.binomial m l ≠ 53) ∧ 
  (n > 53 → (k = 0 ∨ k = n ∨ Π a b, a * 53 ≠ b * Nat.factorial (n - k + 1))) :=
sorry

end pascal_triangle_contains_53_only_once_l337_337800


namespace similar_triangles_with_smallest_angle_45_l337_337387

/-- For a triangle \( T = ABC \), we take the point \( X \) on the side \( AB \) such that \( \frac{AX}{XB} = \frac{4}{5} \),
the point \( Y \) on the segment \( CX \) such that \( CY = 2YX \), and the point \( Z \) on the ray \( CA \) such 
that \( \angle CXZ = 180^\circ - \angle ABC \). If \( \angle XYZ = 45^\circ \),
then all triangles in this configuration are similar and the smallest angle is \( 45^\circ \). -/
theorem similar_triangles_with_smallest_angle_45
  (T : Triangle)
  (A B C : Point)
  (X : Point)
  (Y : Point)
  (Z : Point)
  (hX : X ∈ segment A B ∧ (length (segment A X) / length (segment X B) = 4 / 5))
  (hY : Y ∈ segment C X ∧ (length (segment C Y) = 2 * length (segment Y X)))
  (hZ : Z ∈ ray C A ∧ ∠ C X Z = 180 - ∠ A B C)
  (hXYZ : ∠ X Y Z = 45) :
  ∀ T₁ T₂ ∈ Σ, similar T₁ T₂ ∧ ∀ t ∈ Σ, angle t = 45 :=
sorry

end similar_triangles_with_smallest_angle_45_l337_337387


namespace sum_of_possible_values_l337_337890

theorem sum_of_possible_values (a b c d : ℝ) (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 7) : 
  (a - c) * (b - d) / ((c - d) * (d - a)) = 1 :=
by
  -- Solution omitted
  sorry

end sum_of_possible_values_l337_337890


namespace capabilities_of_digital_cities_l337_337362

-- Definitions from the problem conditions
def digital_city := 
1. digitization of city facilities ∧
2. networking of the city (internet, cable TV, mobile networks, fiber optics, wide area networks, local area networks) ∧
3. intelligence of the city which includes online commerce, online finance, online education, online hospitals, and online government affairs

-- Capabilities defined in the problem
def travel_around_the_world := false
def receive_distance_education := true
def shop_online := true
def seek_medical_advice_online := true

-- Prove the equivalent statement according to the conditions
theorem capabilities_of_digital_cities :
  (digital_city → (receive_distance_education ∧ shop_online ∧ seek_medical_advice_online)) := 
by 
  sorry

end capabilities_of_digital_cities_l337_337362


namespace number_of_subsets_of_M_l337_337772

-- Define the set M
def M := {1, 2} : Set ℕ

-- Define the proof statement: the number of subsets of set M is 4
theorem number_of_subsets_of_M: ∃ n : ℕ, n = 4 ∧ n = 2 ^ 2 := by
  sorry

end number_of_subsets_of_M_l337_337772


namespace triangle_AD_length_l337_337603

theorem triangle_AD_length (A B C D : Type) [is_triangle A B C]
  (on_AC : D ∈ AC) (AB_eq_one : AB = 1) (DC_eq_one : DC = 1)
  (angle_DBC : ∠DBC = 30) (angle_ABD : ∠ABD = 90) :
  D.length AD = real.cbrt 2 :=
by sorry

end triangle_AD_length_l337_337603


namespace find_a_l337_337383

theorem find_a (a : ℝ) : 
  --- Conditions:
  (∃ a, (dist_point_line (a, 6) (3, -4, 2) = 4)) ↔ (a = 14 ∨ a = 2/3) :=
begin
  sorry
end

-- You might need to implement the distance function
-- Use appropriate definition for dist_point_line if not already defined
def dist_point_line (P : ℝ × ℝ) (l : ℝ × ℝ × ℝ) : ℝ :=
  |l.1 * P.1 + l.2 * P.2 + l.3| / (real.sqrt (l.1^2 + l.2^2))

end find_a_l337_337383


namespace billy_win_probability_l337_337660

-- Definitions of states and transition probabilities
def alice_step_prob_pos : ℚ := 1 / 2
def alice_step_prob_neg : ℚ := 1 / 2
def billy_step_prob_pos : ℚ := 2 / 3
def billy_step_prob_neg : ℚ := 1 / 3

-- Definitions of states in the Markov chain
inductive State
| S0 | S1 | Sm1 | S2 | Sm2 -- Alice's states
| T0 | T1 | Tm1 | T2 | Tm2 -- Billy's states

open State

-- The theorem statement: the probability that Billy wins the game
theorem billy_win_probability : 
  ∃ (P : State → ℚ), 
  P S0 = 11 / 19 ∧ P T0 = 14 / 19 ∧ 
  P S1 = 1 / 2 * P T0 ∧
  P Sm1 = 1 / 2 * P S0 + 1 / 2 ∧
  P T0 = 2 / 3 * P T1 + 1 / 3 * P Tm1 ∧
  P T1 = 2 / 3 + 1 / 3 * P S0 ∧
  P Tm1 = 2 / 3 * P T0 ∧
  P S2 = 0 ∧ P Sm2 = 1 ∧ P T2 = 1 ∧ P Tm2 = 0 := 
by 
  sorry

end billy_win_probability_l337_337660


namespace num_pos_whole_numbers_with_cube_roots_less_than_five_l337_337088

theorem num_pos_whole_numbers_with_cube_roots_less_than_five : 
  {n : ℕ | ∃ k : ℕ, k < 5 ∧ k^3 = n}.card = 124 :=
sorry

end num_pos_whole_numbers_with_cube_roots_less_than_five_l337_337088


namespace abc_inequality_l337_337888

theorem abc_inequality (a b c : ℝ) : a^2 + b^2 + c^2 ≥ ab + ac + bc :=
by
  sorry

end abc_inequality_l337_337888


namespace right_triangle_ratios_is_right_triangle_l337_337241

theorem right_triangle_ratios (x : ℝ) (h_pos : x > 0) : 
  let a := x
  let b := x * real.sqrt 3
  let c := 2 * x
  a^2 + b^2 = c^2 := 
by 
  let a := x
  let b := x * real.sqrt 3
  let c := 2 * x
  calc
    a^2 + b^2 = x^2 + (x * real.sqrt 3)^2 : by sorry
           ... = x^2 + 3 * x^2 : by sorry
           ... = 4 * x^2 : by sorry
           ... = c^2 : by sorry

-- Prove the triangle is a right triangle based on the Pythagorean theorem.
theorem is_right_triangle (a b c : ℝ)
  (h : a^2 + b^2 = c^2) : 
  a = x ∧ b = x * real.sqrt 3 ∧ c = 2 * x → 
  a^2 + b^2 = c^2 :=
by
  intro h_ratios
  exact h

end right_triangle_ratios_is_right_triangle_l337_337241


namespace common_ratio_neg_two_l337_337228

theorem common_ratio_neg_two (a : ℕ → ℝ) (q : ℝ) 
  (h : ∀ n, a (n + 1) = a n * q)
  (H : 8 * a 2 + a 5 = 0) : 
  q = -2 :=
sorry

end common_ratio_neg_two_l337_337228


namespace cost_of_five_dozen_l337_337482

noncomputable def price_per_dozen (total_cost : ℝ) (num_dozen : ℕ) : ℝ :=
  total_cost / num_dozen

noncomputable def total_cost (price_per_dozen : ℝ) (num_dozen : ℕ) : ℝ :=
  price_per_dozen * num_dozen

theorem cost_of_five_dozen (total_cost_threedozens : ℝ := 28.20) (num_threedozens : ℕ := 3) (num_fivedozens : ℕ := 5) :
  total_cost (price_per_dozen total_cost_threedozens num_threedozens) num_fivedozens = 47.00 :=
  by sorry

end cost_of_five_dozen_l337_337482


namespace cylinder_longest_segment_cylinder_volume_l337_337637

-- Define the cylinder properties
def cylinder_radius : ℝ := 5
def cylinder_height : ℝ := 12

-- Calculate the diameter from the radius
def cylinder_diameter : ℝ := 2 * cylinder_radius

-- Proof for the longest segment (diagonal of the rectangular cross-section)
theorem cylinder_longest_segment :
  let hypotenuse := Real.sqrt (cylinder_height^2 + cylinder_diameter^2)
  hypotenuse = 2 * (Real.sqrt 61) :=
by
  let h := Real.sqrt (cylinder_height^2 + cylinder_diameter^2)
  have h_eq : h = 2 * (Real.sqrt 61), from sorry
  exact h_eq

-- Proof for the volume of the cylinder
theorem cylinder_volume :
  let volume := Real.pi * cylinder_radius^2 * cylinder_height
  volume = 300 * Real.pi :=
by
  let V := Real.pi * cylinder_radius^2 * cylinder_height
  have V_eq : V = 300 * Real.pi, from sorry
  exact V_eq

end cylinder_longest_segment_cylinder_volume_l337_337637


namespace number_of_textbooks_l337_337481

theorem number_of_textbooks (h_capacity : 80)
  (h_hardcover_count : 70) (h_hardcover_weight : 0.5)
  (h_textbook_weight : 2) (h_knick_knack_count : 3) 
  (h_knick_knack_weight : 6) (h_over_limit : 33) : 
  let total_hardcover_weight := h_hardcover_count * h_hardcover_weight,
      total_knick_knack_weight := h_knick_knack_count * h_knick_knack_weight,
      total_weight := h_capacity + h_over_limit,
      weight_of_textbooks := total_weight - (total_hardcover_weight + total_knick_knack_weight),
      number_of_textbooks := weight_of_textbooks / h_textbook_weight
  in number_of_textbooks = 30 :=
by
  sorry

end number_of_textbooks_l337_337481


namespace three_times_sum_of_midpoint_l337_337276

theorem three_times_sum_of_midpoint (a1 b1 a2 b2 : ℝ) (h₁ : a1 = 10) (h₂ : b1 = 3) (h₃ : a2 = 4) (h₄ : b2 = -5) :
  3 * ((a1 + a2) / 2 + (b1 + b2) / 2) = 18 :=
by
  rw [h₁, h₂, h₃, h₄]
  norm_num
  sorry

end three_times_sum_of_midpoint_l337_337276


namespace edward_money_left_l337_337709

noncomputable def toy_cost : ℝ := 0.95

noncomputable def toy_quantity : ℕ := 4

noncomputable def toy_discount : ℝ := 0.15

noncomputable def race_track_cost : ℝ := 6.00

noncomputable def race_track_tax : ℝ := 0.08

noncomputable def initial_amount : ℝ := 17.80

noncomputable def total_toy_cost_before_discount : ℝ := toy_quantity * toy_cost

noncomputable def discount_amount : ℝ := toy_discount * total_toy_cost_before_discount

noncomputable def total_toy_cost_after_discount : ℝ := total_toy_cost_before_discount - discount_amount

noncomputable def race_track_tax_amount : ℝ := race_track_tax * race_track_cost

noncomputable def total_race_track_cost_after_tax : ℝ := race_track_cost + race_track_tax_amount

noncomputable def total_amount_spent : ℝ := total_toy_cost_after_discount + total_race_track_cost_after_tax

noncomputable def money_left : ℝ := initial_amount - total_amount_spent

theorem edward_money_left : money_left = 8.09 := by
  -- proof goes here
  sorry

end edward_money_left_l337_337709


namespace minimum_barge_capacity_l337_337671

theorem minimum_barge_capacity :
  ∃ (k : ℝ), k = 180 / 19 ∧ 
  (∀ {cargo_mass : ℕ → ℝ} 
    (Htruck1 : ∀ i, 0 ≤ cargo_mass i)
    (Htruck2 : ∀ i, cargo_mass i ≤ 10)
    (Htot : ∑ i in range 9, cargo_mass i ≤ 90)
    (Hitem : ∀ i, cargo_mass i ≤ 1)
    (barges : ℕ), barges = 10 →
    ∑ i in range 9, cargo_mass i ≤ barges * k) :=
sorry

end minimum_barge_capacity_l337_337671


namespace find_loan_amount_l337_337201

-- Define the conditions
def rate_of_interest : ℝ := 0.06
def time_period : ℝ := 6
def interest_paid : ℝ := 432

-- Define the simple interest formula
def simple_interest (P r t : ℝ) : ℝ := P * r * t

-- State the theorem to prove the loan amount
theorem find_loan_amount (P : ℝ) (h1 : rate_of_interest = 0.06) (h2 : time_period = 6) (h3 : interest_paid = 432) (h4 : simple_interest P rate_of_interest time_period = interest_paid) : P = 1200 :=
by
  -- Here should be the proof, but it's omitted for now
  sorry

end find_loan_amount_l337_337201


namespace num_pos_whole_numbers_with_cube_roots_less_than_five_l337_337087

theorem num_pos_whole_numbers_with_cube_roots_less_than_five : 
  {n : ℕ | ∃ k : ℕ, k < 5 ∧ k^3 = n}.card = 124 :=
sorry

end num_pos_whole_numbers_with_cube_roots_less_than_five_l337_337087


namespace find_unknown_rate_l337_337320

-- Define the known quantities
def num_blankets1 := 4
def price1 := 100

def num_blankets2 := 5
def price2 := 150

def num_blankets3 := 3
def price3 := 200

def num_blankets4 := 6
def price4 := 75

def num_blankets_unknown := 2

def avg_price := 150
def total_blankets := num_blankets1 + num_blankets2 + num_blankets3 + num_blankets4 + num_blankets_unknown -- 20 blankets in total

-- Hypotheses
def total_known_cost := num_blankets1 * price1 + num_blankets2 * price2 + num_blankets3 * price3 + num_blankets4 * price4
-- 2200 Rs.

def total_cost := total_blankets * avg_price -- 3000 Rs.

theorem find_unknown_rate :
  (total_cost - total_known_cost) / num_blankets_unknown = 400 :=
by sorry

end find_unknown_rate_l337_337320


namespace solve_for_x_l337_337019

theorem solve_for_x (x : ℝ) (h : 5^x * 125^(3 * x) = 625^7) : x = 2.8 :=
by
  sorry

end solve_for_x_l337_337019


namespace arctan_tan_expression_l337_337692

theorem arctan_tan_expression :
  ∀ (x : ℝ), x = 75 → arctan (2 * tan (75 * pi / 180) - 3 * tan (15 * pi / 180)) = 30 * pi / 180 :=
by
  sorry

end arctan_tan_expression_l337_337692


namespace atomic_weight_of_calcium_l337_337634

theorem atomic_weight_of_calcium (molecular_weight_CaO : ℕ) (atomic_weight_O : ℕ) 
  (h1 : molecular_weight_CaO = 56) (h2 : atomic_weight_O = 16) :
  56 - 16 = 40 :=
by
  rw [h1, h2]
  exact rfl

end atomic_weight_of_calcium_l337_337634


namespace coordinates_sum_l337_337339

theorem coordinates_sum (g : ℝ → ℝ) 
  (h₁ : g 8 = 5) : 
  let x := 3 in 
  let y := (14 / 9) in 
  x + y = 41 / 9 := 
by
  sorry

end coordinates_sum_l337_337339


namespace simplified_polynomial_l337_337672

theorem simplified_polynomial : ∀ (x : ℝ), (3 * x + 2) * (3 * x - 2) - (3 * x - 1) ^ 2 = 6 * x - 5 := by
  sorry

end simplified_polynomial_l337_337672


namespace triangle_side_relation_l337_337880

theorem triangle_side_relation 
  (a b c : ℝ) 
  (A : ℝ) 
  (h : b^2 + c^2 = a * ((√3 / 3) * b * c + a)) : 
  a = 2 * √3 * Real.cos A := 
sorry

end triangle_side_relation_l337_337880


namespace no_real_or_imaginary_values_of_t_l337_337947

theorem no_real_or_imaginary_values_of_t (t : ℂ) : sqrt(16 - t^2) + 3 = 0 → false :=
by
  sorry

end no_real_or_imaginary_values_of_t_l337_337947


namespace proportion_sweets_not_overweight_l337_337118

variables {Ω : Type} [ProbabilitySpace Ω]
variables (A B : Event Ω)
variables (P_A_given_B : Probability (B | A) = 0.80)
variables (P_B_given_A : Probability (A | B) = 0.70)

theorem proportion_sweets_not_overweight (A B : Event Ω)
  (P_A_given_B : Probability (B | A) = 0.80)
  (P_B_given_A : Probability (A | B) = 0.70) :
  Probability (B ∩ Aᶜ) = (12 / 47) * Probability (B) :=
by
  sorry

end proportion_sweets_not_overweight_l337_337118


namespace max_min_product_f_l337_337030

-- Definitions of complex numbers based on the problem conditions
def z1 (x y : ℝ) : ℂ := x + real.sqrt 5 + y * complex.I
def z2 (x y : ℝ) : ℂ := x - real.sqrt 5 + y * complex.I

-- Definition of the function f(x, y)
def f (x y : ℝ) : ℝ := abs (2 * x - 3 * y - 12)

-- The proof problem statement
theorem max_min_product_f {x y : ℝ} 
(hz : abs (z1 x y) + abs (z2 x y) = 6) : 
let f_max := max (f 3 2) (f (-3) 2) in
let f_min := min (f 3 2) (f (-3) 2) in
f_max * f_min = 72 :=
sorry

end max_min_product_f_l337_337030


namespace red_light_probability_l337_337330

theorem red_light_probability:
  ∃ (R Y G T: ℕ), 
  R = 30 ∧ Y = 5 ∧ G = 40 ∧ T = R + Y + G ∧ (R / T.toRat = 2 / 5) :=
by
  -- Define the durations
  let R := 30
  let Y := 5
  let G := 40
  let T := 30 + 5 + 40
  exists R, Y, G, T
  split
  exact rfl
  split
  exact rfl
  split
  exact rfl
  split
  exact rfl
  -- Simplify the probability calculation
  have H1: (R : ℚ) / T.toRat = 2 / 5, by
    have H2: R = 30 := rfl
    have H3: T = 75 := rfl
    rw [H2, H3]
    norm_num
  exact H1
  sorry

end red_light_probability_l337_337330


namespace ellipse_equation_and_line_existence_l337_337043

theorem ellipse_equation_and_line_existence :
  ∃ (a b : ℝ), 
    (a > b) ∧ (b > 0) ∧ 
    (∃ c : ℝ, b = Real.sqrt 3 * c ∧ (1/(a^2) + 9/(4 * b^2) = 1) ∧ (a^2 = b^2 + c^2)) ∧
    (1, 3/2) ∈ {p | ∃ x y : ℝ, p = (x,y) ∧ (x^2/4 + y^2/3 = 1)} ∧
    (∃ k m : ℝ,
      (k = 1/2 ∨ k = -1/2) ∧ (m = sqrt 21 / 7 ∨ m = -sqrt 21 / 7) ∧
      (∀ (l : ℝ → ℝ),
        (l = (λ x, k*x + m)) ∧
        (∃ A P N M : ℝ × ℝ,
          -- Point definitions
          ∀ x, (l x = 0 → N = (x, 0)) ∧
          (k*x + 3m = 0 → P = (x, 2*m)) ∧
          (∀ M, M = (0, m)) ∧
          abs (dist P M) = abs (dist N M) ∧
          -- Reflection and intersection conditions
          -- Here you would expand on further conditions that match the geometric interpretations
          true ))
sorry

end ellipse_equation_and_line_existence_l337_337043


namespace find_remainder_2_l337_337099

noncomputable def remainder_1 (x : ℝ) : ℝ :=
  (x + 1 / 2) % x^8

noncomputable def q_1 (x : ℝ) : ℝ :=
  x^7 - (1 / 2) * x^6 + (1 / 4) * x^5 - (1 / 8) * x^4 + (1 / 16) * x^3 - (1 / 32) * x^2 + (1 / 64) * x - (1 / 128)

theorem find_remainder_2 : 
  (let r_1 := ((-1 / 2)^8) in
   r_1 = 1 / 256 ∧
   q_1(-1 / 2) = -1 / 16) →
  ∃ r_2 : ℝ, r_2 = -1 / 16 :=
by
  sorry

end find_remainder_2_l337_337099


namespace leading_coeff_divisible_by_p_l337_337886

variables {p k : ℕ} {Q : ℕ → ℤ} (hp : Nat.Prime p) (hk1 : k > 1) (hk2 : k ∣ (p - 1))
  (hdeg : ∀ n m : ℕ, n > k → Q n ≠ 0)
  (hcoeffs : ∀ n : ℕ, Q n ∈ Set.Univ)
  (hvals : ∃ f : ℕ → ℕ, ∀ x, 0 ≤ x < p → Q(x) ≡ f x [MOD p]) :

theorem leading_coeff_divisible_by_p :
  (leading_coeff Q) ∣ p :=
sorry

end leading_coeff_divisible_by_p_l337_337886


namespace chess_moves_l337_337917

theorem chess_moves (p_move_time p_time_each_move p_total_moves : ℕ)
    (q_move_time q_time_each_move q_total_moves : ℕ)
    (p_move_time_eq : p_move_time = 28)
    (p_total_moves_eq : p_total_moves = 17 * 60)
    (q_move_time_eq : q_move_time = 40)
    : p_total_moves + q_total_moves = 30 :=
by
  have total_time : p_move_time * p_time_each_move + q_move_time * q_time_each_move = 17 * 60 :=
    by rw [p_move_time_eq,p_total_moves_eq,q_move_time_eq]
  sorry

end chess_moves_l337_337917


namespace parabola_vertex_n_l337_337969

theorem parabola_vertex_n : 
  (∃ m : ℝ, ∃ n : ℝ, (∀ x : ℝ, -3 * x^2 - 30 * x - 81 = -3 * (x + m) ^ 2 + n) ∧ n = -6) :=
begin
  sorry
end

end parabola_vertex_n_l337_337969


namespace element_correspondence_l337_337876

def A : Type := { p : ℝ × ℝ // true }
def B : Type := { p : ℝ × ℝ // true }

def f (p : A) : B :=
  let (x, y) := p.1
  ⟨(x - y, x + y), trivial⟩

theorem element_correspondence :
  ∃ (p : A), f p = ⟨(-1, 1), trivial⟩ ∧ p.1 = (0, 1) := by
  sorry

end element_correspondence_l337_337876


namespace find_smallest_k_l337_337295

def x_table (i j : ℕ) : ℝ → Prop := i ≤ 100 ∧ j ≤ 25 ∧ x i j ≥ 0

def sorted_x'_table (i j : ℕ) (x x' : ℕ → ℕ → ℝ) : Prop :=
  ∀ j, ∀ i₁ i₂, 1 ≤ i₁ → i₁ ≤ 100 → 1 ≤ i₂ → i₂ ≤ 100 → 
  (i₁ ≤ i₂ → x' i₁ j ≥ x' i₂ j)

def row_sum_le (x : ℕ → ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i, 1 ≤ i → i ≤ n → ∑ j in finset.range 25, x i j ≤ 1

theorem find_smallest_k (x x' : ℕ → ℕ → ℝ) :
  (∀ i, 1 ≤ i → i ≤ 100 → (∑ j in finset.range 25, x i j) ≤ 1) →
  sorted_x'_table 100 25 x x' →
  ∃ k, k = 97 ∧ (∀ i, i ≥ k → (∑ j in finset.range 25, x' i j) ≤ 1) :=
begin
  sorry
end

end find_smallest_k_l337_337295


namespace count_valid_c_l337_337389

-- Define the interval and the equation
def interval := set.Icc 0 1000
def equation (x : ℝ) (c : ℝ) : Prop := 9 * ⌊x⌋ + 3 * ⌈x⌉ + 5 * (x - ⌊x⌋) = c

-- Define the proposition to be proved
theorem count_valid_c : 
  ∃ (s : finset ℝ), (∀ c ∈ s, c ∈ interval ∧ ∃ x, equation x c) ∧ 
  s.card = 84 :=
sorry

end count_valid_c_l337_337389


namespace mark_reading_time_l337_337529

variable (x y : ℕ)

theorem mark_reading_time (x y : ℕ) : 
  7 * x + y = 7 * x + y :=
by
  sorry

end mark_reading_time_l337_337529


namespace pascal_contains_53_l337_337796

theorem pascal_contains_53 (n : ℕ) (h1 : Nat.Prime 53) (h2 : ∃ k, 1 ≤ k ∧ k ≤ 52 ∧ nat.choose 53 k = 53) (h3 : ∀ m < 53, ¬ (∃ k, 1 ≤ k ∧ k ≤ m - 1 ∧ nat.choose m k = 53)) (h4 : ∀ m > 53, ¬ (∃ k, 1 ≤ k ∧ k ≤ m - 1 ∧ nat.choose m k = 53)) : 
  (n = 53) → (n = 1) := 
by
  intros
  sorry

end pascal_contains_53_l337_337796


namespace isosceles_triangles_not_necessarily_congruent_l337_337140

-- Define the isosceles triangle with equal sides and an incircle radius
structure IsoscelesTriangle :=
  (a : ℝ) -- side length of equal sides
  (r : ℝ) -- radius of the inscribed circle
  (h : ℝ) -- height of the triangle

-- Conditions for defining the radius of an incircle
def incircle_radius_condition (Δ : IsoscelesTriangle) : Prop :=
  ∃ b : ℝ, let h := Δ.h in
  let A := (1/2) * b * h in
  let s := (2 * Δ.a + b) / 2 in
  Δ.r = A / s

-- The main theorem statement
theorem isosceles_triangles_not_necessarily_congruent (Δ1 Δ2 : IsoscelesTriangle)
  (cond1 : incircle_radius_condition Δ1)
  (cond2 : incircle_radius_condition Δ2) :
  Δ1.a = Δ2.a ∧ Δ1.r = Δ2.r →
  ¬ (Δ1 = Δ2) :=
by
  sorry

end isosceles_triangles_not_necessarily_congruent_l337_337140


namespace big_eighteen_basketball_conference_games_l337_337560

theorem big_eighteen_basketball_conference_games :
  let divisions := 3
  let teams_per_division := 6
  let intra_division_games := divisions * (teams_per_division * (teams_per_division - 1) / 2 * 3)
  let inter_division_opponents_per_team := (divisions - 1) * teams_per_division
  let inter_division_games_per_team := inter_division_opponents_per_team * 2
  let total_inter_division_games := inter_division_games_per_team * (divisions * teams_per_division) / 2
  intra_division_games + total_inter_division_games = 351 :=
by
  let divisions := 3
  let teams_per_division := 6
  let intra_division_games := divisions * (teams_per_division * (teams_per_division - 1) / 2 * 3)
  let inter_division_opponents_per_team := (divisions - 1) * teams_per_division
  let inter_division_games_per_team := inter_division_opponents_per_team * 2
  let total_inter_division_games := inter_division_games_per_team * (divisions * teams_per_division) / 2
  show intra_division_games + total_inter_division_games = 351 from
    sorry

end big_eighteen_basketball_conference_games_l337_337560


namespace unique_affine_transformation_point_basis_unique_affine_transformation_triangles_unique_affine_transformation_parallelograms_l337_337627

-- Define affine transformations, points, and vectors
section affine_transformations

-- 1. Statement for affine transformation mapping of point and basis vectors
theorem unique_affine_transformation_point_basis 
  (O O' : Point) 
  (e1 e2 e1' e2' : Vector) :
  ∃! (L : AffineTransformation), L.map_point O = O' ∧ L.map_vector e1 = e1' ∧ L.map_vector e2 = e2' := sorry

-- 2. Statement for affine transformation between triangles
theorem unique_affine_transformation_triangles
  (A B C A1 B1 C1 : Point) :
  ∃! (L : AffineTransformation), L.map_point A = A1 ∧ L.map_point B = B1 ∧ L.map_point C = C1 := sorry

-- 3. Statement for affine transformation between parallelograms
theorem unique_affine_transformation_parallelograms 
  (p1 p2 p3 p4 q1 q2 q3 q4 : Point)
  (H1 : is_parallelogram p1 p2 p3 p4)
  (H2 : is_parallelogram q1 q2 q3 q4) :
  ∃! (L : AffineTransformation), 
    L.map_point p1 = q1 ∧
    L.map_point p2 = q2 ∧
    L.map_point p3 = q3 ∧
    L.map_point p4 = q4 := sorry

end affine_transformations

end unique_affine_transformation_point_basis_unique_affine_transformation_triangles_unique_affine_transformation_parallelograms_l337_337627


namespace largest_integer_divides_product_l337_337175

theorem largest_integer_divides_product (n : ℕ) : 
  ∃ m, ∀ k : ℕ, k = (2*n-1)*(2*n)*(2*n+2) → m ≥ 1 ∧ m = 8 ∧ m ∣ k :=
by
  sorry

end largest_integer_divides_product_l337_337175


namespace find_theta_l337_337751

-- Given definitions
variables (e1 e2 : EuclideanSpace ℝ (Fin 2))
variable (θ : ℝ)
variables (a : EuclideanSpace ℝ (Fin 2))
hypothesis norm_e1 : ∥e1∥ = 1
hypothesis norm_e2 : ∥e2∥ = 1
hypothesis angle : real.angle (e1) (e2) = θ
hypothesis def_a : a = 2 • e1 + 3 • e2
hypothesis norm_a : ∥a∥ = 1

-- The problem statement
theorem find_theta : θ = real.pi := 
by 
  sorry

end find_theta_l337_337751


namespace cherries_purchase_l337_337729

theorem cherries_purchase (total_money : ℝ) (price_per_kg : ℝ) 
  (genevieve_money : ℝ) (shortage : ℝ) (clarice_money : ℝ) :
  genevieve_money = 1600 → shortage = 400 → clarice_money = 400 → price_per_kg = 8 →
  total_money = genevieve_money + shortage + clarice_money →
  total_money / price_per_kg = 250 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end cherries_purchase_l337_337729


namespace vector_identity_l337_337125

variables {V : Type*} [AddCommGroup V]

theorem vector_identity
  (OA OB AC BC : V)
  (h1 : BC = OA - OB + AC) :
  OA - OB + AC = BC :=
by
  rw <- h1
  sorry

end vector_identity_l337_337125


namespace ellipse_perimeter_20_l337_337943

noncomputable def ellipse_perimeter_of_triangle (x y : ℝ) (F₁ F₂ A B : (ℝ × ℝ)) : ℝ :=
  4 * real.sqrt 25

theorem ellipse_perimeter_20 :
  let C := λ x y : ℝ, (x^2) / 25 + (y^2) / 16 = 1 in
  let F₁ := (-5, 0) in
  let F₂ := (5, 0) in
  let A := sorry in -- Definitions for A and B that satisfy the conditions of the problem
  let B := sorry in
  let A_and_B_on_C := C (A.1) (A.2) ∧ C (B.1) (B.2) in
  (C (A.1) (A.2)) ∧ (C (B.1) (B.2)) ∧ (A.1 - F₂.1 = 0) ∧ (B.1 - F₂.1 = 0) →
  ellipse_perimeter_of_triangle A.1 A.2 F₁ F₂ A B = 20 := 
by
  intros
  simp
  sorry

end ellipse_perimeter_20_l337_337943


namespace surface_area_of_sphere_l337_337415

theorem surface_area_of_sphere (s : ℝ) (r : ℝ) (S : ℝ) :
  s = 1 → r = (real.sqrt 3) / 2 → S = 4 * real.pi * r^2 → S = 3 * real.pi :=
begin
  sorry
end

end surface_area_of_sphere_l337_337415


namespace fill_tub_in_seconds_l337_337726

theorem fill_tub_in_seconds 
  (faucet_rate : ℚ)
  (four_faucet_rate : ℚ := 4 * faucet_rate)
  (three_faucet_rate : ℚ := 3 * faucet_rate)
  (time_for_100_gallons_in_minutes : ℚ := 6)
  (time_for_100_gallons_in_seconds : ℚ := time_for_100_gallons_in_minutes * 60)
  (volume_100_gallons : ℚ := 100)
  (rate_per_three_faucets_in_gallons_per_second : ℚ := volume_100_gallons / time_for_100_gallons_in_seconds)
  (rate_per_faucet : ℚ := rate_per_three_faucets_in_gallons_per_second / 3)
  (rate_per_four_faucets : ℚ := 4 * rate_per_faucet)
  (volume_50_gallons : ℚ := 50)
  (expected_time_seconds : ℚ := volume_50_gallons / rate_per_four_faucets) :
  expected_time_seconds = 135 :=
sorry

end fill_tub_in_seconds_l337_337726


namespace whole_numbers_between_sqrt10_sqrt90_l337_337092

theorem whole_numbers_between_sqrt10_sqrt90 : 
  let n := λ (x : ℝ), x ∈ set.Icc (⌈real.sqrt 10⌉) (⌊real.sqrt 90⌋) in
  (⨆ n, finset.card (finset.filter n (finset.range 10))) = 6 := by
sorry

end whole_numbers_between_sqrt10_sqrt90_l337_337092


namespace exists_nat_number_gt_1000_l337_337626

noncomputable def sum_of_digits (n : ℕ) : ℕ := sorry

theorem exists_nat_number_gt_1000 (S : ℕ → ℕ) :
  (∀ n : ℕ, S (2^n) = sum_of_digits (2^n)) →
  ∃ n : ℕ, n > 1000 ∧ S (2^n) > S (2^(n + 1)) :=
by sorry

end exists_nat_number_gt_1000_l337_337626


namespace pascal_triangle_contains_53_l337_337824

theorem pascal_triangle_contains_53 (n : ℕ) :
  (∃ k, binomial n k = 53) ↔ n = 53 := 
sorry

end pascal_triangle_contains_53_l337_337824


namespace compare_games_l337_337305
open ProbabilityTheory

noncomputable def biasedCoin : Probability ℚ :=
  ProbabilityTheory.Event 0.25

noncomputable def gameC (coin : Probability ℚ) : Probability ℚ :=
  (coin * (1 - coin) + (1 - coin) * coin)

noncomputable def gameD (coin : Probability ℚ) : Probability ℚ :=
  ((coin ^ 2 * (1 - coin)) + (coin * (1 - coin) ^ 2) + ((1 - coin) * coin ^ 2) + (coin ^ 3))

theorem compare_games
  (coin : Probability ℚ := biasedCoin) :
  (gameD coin - gameC coin = 15 / 32) :=
begin
  sorry
end

end compare_games_l337_337305


namespace abs_diff_is_13_l337_337114

-- Definition for the sum and product of three numbers
def sum (a b c : ℕ) : Prop := a + b + c = 78
def product (a b c : ℕ) : Prop := a * b * c = 9240

-- Definition for finding the absolute difference between the largest and smallest number
def abs_diff (a b c : ℕ) : ℕ := |max a b c - min a b c|

-- The statement we need to prove
theorem abs_diff_is_13 {a b c : ℕ} (h_sum : sum a b c) (h_product : product a b c) : abs_diff a b c = 13 := 
sorry

end abs_diff_is_13_l337_337114


namespace reflection_line_equation_l337_337263

theorem reflection_line_equation :
  ∀ (D E F D' E' F' : ℝ × ℝ),
  D = (1, 2) → E = (6, 3) → F = (-3, 4) →
  D' = (1, -2) → E' = (6, -3) → F' = (-3, -4) →
  ∃ M : ℝ, (∀ P : ℝ × ℝ, P ∈ [D, E, F] → ∃ P' : ℝ × ℝ, P' ∈ [D', E', F'] ∧ P reflects to P' over (M, M)) ∧ M = 0 :=
by
  sorry

end reflection_line_equation_l337_337263


namespace conj_z_in_first_quadrant_l337_337759

-- Define the complex number z
def z : ℂ := (2 - I) / (1 + I)

-- Define the conjugate of z
def conj_z : ℂ := conj z

-- Define the quadrant_check function to determine the quadrant
def quadrant_check (z : ℂ) : String :=
  if z.re > 0 ∧ z.im > 0 then "First Quadrant"
  else if z.re < 0 ∧ z.im > 0 then "Second Quadrant"
  else if z.re < 0 ∧ z.im < 0 then "Third Quadrant"
  else "Fourth Quadrant"

-- Prove that the conjugate of z lies in the first quadrant
theorem conj_z_in_first_quadrant : quadrant_check conj_z = "First Quadrant" :=
by sorry

end conj_z_in_first_quadrant_l337_337759


namespace log_equation_solution_l337_337008

theorem log_equation_solution (x : ℝ) (h : log 4 (3 * x - 4) = 2) : x = 20 / 3 :=
sorry

end log_equation_solution_l337_337008


namespace gen_term_a_gen_term_b_sum_b_l337_337771

variable {n : ℕ} (hn : n ≥ 2) (h_nat : n ∈ ℕ)

-- Definitions based on conditions
def a : ℕ → ℕ
| 1     := 1
| (n+1) := 2 * (n + 1) - a n + 1

noncomputable def b (n : ℕ) : ℚ := 1 / (4 * (a n) - 1)

noncomputable def T (n : ℕ) : ℚ := (Finset.range n).sum (λ k, b (k + 1))

-- The proof goal for part (I)
theorem gen_term_a (n : ℕ) (hn : n ≥ 1) (h_nat : n ∈ ℕ) : a n = n^2 := sorry

-- The proof goal for part (II) - general term for b_n
theorem gen_term_b (n : ℕ) (hn : n ≥ 1) (h_nat : n ∈ ℕ) : b n = 1 / (4 * (n^2) - 1) := sorry

-- The proof goal for part (II) - sum of the first n terms T_n
theorem sum_b (n : ℕ) (hn : n ≥ 1) (h_nat : n ∈ ℕ) : T n = ↑n / (2 * ↑n + 1) := sorry

end gen_term_a_gen_term_b_sum_b_l337_337771


namespace hamburger_cost_is_4_l337_337337

-- Given conditions
def initial_money : ℝ := 132
def remaining_money : ℝ := 70
def milkshake_cost : ℝ := 5
def hamburgers_bought : ℕ := 8
def milkshakes_bought : ℕ := 6

-- Hamburger cost to be proved
def hamburger_cost : ℝ := (initial_money - remaining_money - milkshake_cost * milkshakes_bought) / hamburgers_bought

theorem hamburger_cost_is_4 : hamburger_cost = 4 := by
  sorry

end hamburger_cost_is_4_l337_337337


namespace smallest_total_students_l337_337859

theorem smallest_total_students :
  (∃ (n : ℕ), 4 * n + (n + 2) > 50 ∧ ∀ m, 4 * m + (m + 2) > 50 → m ≥ n) → 4 * 10 + (10 + 2) = 52 :=
by
  sorry

end smallest_total_students_l337_337859


namespace sequence_an_eq_l337_337242

def sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, (finset.range n).sum (λ i, 3^i * (a (i + 1))) = n / 2

theorem sequence_an_eq (a : ℕ → ℝ)
  (h : sequence a) :
  ∀ n : ℕ, a n = 1 / (2 * 3^(n - 1)) :=
by sorry

end sequence_an_eq_l337_337242


namespace cos_7theta_l337_337094

theorem cos_7theta (θ : ℝ) (h : Real.cos θ = 1/3) : Real.cos (7 * θ) = 49 / 2187 := 
  sorry

end cos_7theta_l337_337094


namespace find_k_and_b_l337_337073

noncomputable def setA := {p : ℝ × ℝ | p.2^2 - p.1 - 1 = 0}
noncomputable def setB := {p : ℝ × ℝ | 4 * p.1^2 + 2 * p.1 - 2 * p.2 + 5 = 0}
noncomputable def setC (k b : ℝ) := {p : ℝ × ℝ | p.2 = k * p.1 + b}

theorem find_k_and_b (k b : ℕ) : 
  (setA ∪ setB) ∩ setC k b = ∅ ↔ (k = 1 ∧ b = 2) := 
sorry

end find_k_and_b_l337_337073


namespace distinct_permutations_without_palindromic_substrings_l337_337778

def is_palindromic {α : Type*} [DecidableEq α] (s : List α) : Prop :=
  s = s.reverse

def has_palindromic_substring_of_length_two_or_more (s : String) : Prop :=
  ∃ t : String, t ∈ s.substrings ∧ t.length ≥ 2 ∧ is_palindromic (t.to_list)

def valid_permutations_count (word : String) : ℕ :=
  let perms := word.to_list.permutations.map (λ l, ⟨String.mk l, by simp⟩: Subtype (λ s, s.to_list.perm word.to_list))
  (perms.filter (λ s, ¬ has_palindromic_substring_of_length_two_or_more s.val)).length

theorem distinct_permutations_without_palindromic_substrings (word : String) (h : word = "REDDER") :
  valid_permutations_count word = 1 :=
by { sorry }

end distinct_permutations_without_palindromic_substrings_l337_337778


namespace number_of_rocks_tossed_l337_337262

-- Conditions
def pebbles : ℕ := 6
def rocks : ℕ := 3
def boulders : ℕ := 2
def pebble_splash : ℚ := 1 / 4
def rock_splash : ℚ := 1 / 2
def boulder_splash : ℚ := 2

-- Total width of the splashes
def total_splash (R : ℕ) : ℚ := 
  pebbles * pebble_splash + R * rock_splash + boulders * boulder_splash

-- Given condition
def total_splash_condition : ℚ := 7

theorem number_of_rocks_tossed : 
  total_splash rocks = total_splash_condition → rocks = 3 :=
by
  intro h
  sorry

end number_of_rocks_tossed_l337_337262


namespace daily_wage_c_l337_337999

variable (a_work_days b_work_days c_work_days d_work_days : ℕ)
variable (a_ratio b_ratio c_ratio d_ratio : ℕ)
variable (total_earning : ℝ)

-- Definitions from the problem
def wages (x : ℝ) (ratio : ℕ) : ℝ := x * ratio

theorem daily_wage_c (x : ℝ) (h : total_earning = (a_work_days * wages x a_ratio) +
                                             (b_work_days * wages x b_ratio) +
                                             (c_work_days * wages x c_ratio) +
                                             (d_work_days * wages x d_ratio)) :
    wages (total_earning / (a_work_days * a_ratio + b_work_days * b_ratio + c_work_days * c_ratio + d_work_days * d_ratio)) c_ratio = 5 * (total_earning / (a_work_days * a_ratio + b_work_days * b_ratio + c_work_days * c_ratio + d_work_days * d_ratio)) :=
by
    sorry

-- Example instance constants
def a_work_days : ℕ := 6
def b_work_days : ℕ := 9
def c_work_days : ℕ := 4
def d_work_days : ℕ := 7

def a_ratio : ℕ := 3
def b_ratio : ℕ := 4
def c_ratio : ℕ := 5
def d_ratio : ℕ := 6

def total_earning : ℝ := 2806

-- Now, we enforce the theorem on these constants
def example := 
daily_wage_c a_work_days b_work_days c_work_days d_work_days a_ratio b_ratio c_ratio d_ratio total_earning (x : ℝ) sorry

end daily_wage_c_l337_337999


namespace sum_of_dot_products_eq_l337_337894

-- Define the conditions for sets A and B.
variables {A B : Set ℝ}
variable {m n : ℕ}
variable {a b : ℝ}

-- Condition: the sum of elements in A is a and the sum of elements in B is b.
def sum_of_elements_condition_A : Σ (A : Set ℝ), (∑ s in A, id s) = a ∧ A.card = m := sorry
def sum_of_elements_condition_B : Σ (B : Set ℝ), (∑ t in B, id t) = b ∧ B.card = n := sorry

-- The set C as described in the problem.
def C : Set (ℝ × ℝ) := {p | ∃ s ∈ A, ∃ t ∈ B, p = (s, t)}

-- Main statement to verify in Lean 4.
theorem sum_of_dot_products_eq :
  (∑ x in C, ∑ y in C, x.1 * y.1 + x.2 * y.2) = n^2 * a^2 + m^2 * b^2 :=
sorry

end sum_of_dot_products_eq_l337_337894


namespace factorization_sum_l337_337567

theorem factorization_sum (a b c : ℤ) 
  (h1 : ∀ x : ℤ, x^2 + 9 * x + 20 = (x + a) * (x + b))
  (h2 : ∀ x : ℤ, x^2 + 7 * x - 60 = (x + b) * (x - c)) :
  a + b + c = 21 :=
by
  sorry

end factorization_sum_l337_337567


namespace sin_sol_l337_337410

theorem sin_sol (x : ℝ) (h : tan x = sin (x + π / 2)) : sin x = (sqrt 5 - 1) / 2 :=
sorry

end sin_sol_l337_337410


namespace quadratic_graph_above_x_axis_l337_337299

theorem quadratic_graph_above_x_axis (a b c : ℝ) :
  ¬ ((b^2 - 4*a*c < 0) ↔ ∀ x : ℝ, a*x^2 + b*x + c > 0) :=
sorry

end quadratic_graph_above_x_axis_l337_337299


namespace cube_plane_intersection_distance_l337_337350

noncomputable def point : Type := (ℝ × ℝ × ℝ)

def dist (p1 p2 : point) : ℝ :=
  ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2).sqrt

def plane_eq (p q r : point) (x y z : ℝ) : Prop :=
  let n := cross_product (p - q) (p - r);
  n.1 * x + n.2 * y + n.3 * z = n.1 * p.1 + n.2 * p.2 + n.3 * p.3

theorem cube_plane_intersection_distance :
  let P : point := (0, 3, 0);
  let Q : point := (2, 0, 0);
  let R : point := (2, 6, 6);
  let S : point := (6, 0, 6);
  let T : point := (0, 6, 3);
  dist S T = 3 * (10).sqrt :=
by
  sorry

end cube_plane_intersection_distance_l337_337350


namespace sum_f_eq_two_pow_n_minus_one_l337_337498

open BigOperators

def partitions (n : ℕ) := {s : Multiset ℕ // s.sum = n ∧ (∀ x ∈ s, x ≥ 1)}

noncomputable def f (x : Multiset ℕ) := if h : x.card > 1 then
  ∏ i in Finset.range (x.card - 1), Nat.choose (x.nth_le (i + 1) h) (x.nth_le i h)
else 1

theorem sum_f_eq_two_pow_n_minus_one (n : ℕ) (hn : 0 < n) :
    ∑ x in (partitions n).val, f x = 2^(n-1) :=
sorry

end sum_f_eq_two_pow_n_minus_one_l337_337498


namespace calculation_l337_337705

theorem calculation (a b : ℕ) (h1 : a = 7) (h2 : b = 5) : (a^2 - b^2) ^ 2 = 576 :=
by
  sorry

end calculation_l337_337705


namespace exists_real_q_l337_337723

noncomputable def f (n : ℕ) := min ((λ m : ℤ, abs (sqrt 2 - m / n)))

theorem exists_real_q 
  (C : ℝ)
  (hC : ∀ i : ℕ, f (n_i i) < C / (n_i i) ^ 2) :
  ∃ q : ℝ, q > 1 ∧ (∀ i : ℕ, n_i i ≥ q ^ (i - 1)) :=
sorry

end exists_real_q_l337_337723


namespace spanning_trees_equiv_crossing_edges_l337_337721

open Classical

universe u

variable {V : Type u} [Fintype V] [DecidableEq V] (G : SimpleGraph V)

noncomputable def edge_disjoint_spanning_trees (k : ℕ) : Prop :=
  ∃ (T : Fin k → Set (Sym2 V)), (∀ i, IsSpanningTree (G.subgraph (T i))) ∧ Pairwise Disjoint T

theorem spanning_trees_equiv_crossing_edges (k : ℕ) (G : SimpleGraph V) :
    (edge_disjoint_spanning_trees G k) ↔ 
      (∀ (P : Finset (Finset V)), let ℓ := P.card in 1 < ℓ → G.edgeCrossings P ≥ k * (ℓ - 1)) :=
sorry

end spanning_trees_equiv_crossing_edges_l337_337721


namespace num_valid_lists_len_5_eq_105_l337_337449

def valid_list_condition (l : List ℕ) : Prop := 
  ∀ i ∈ l, 2 ≤ i ∧ i ≤ 5 → (i + 1 ∈ l.take (l.indexOf i) ∨ i - 1 ∈ l.take (l.indexOf i))

def count_valid_lists_of_length_five : ℕ :=
  List.length { l : List ℕ // l.length = 5 ∧ l.perm (List.range 1 6) ∧ valid_list_condition l }

theorem num_valid_lists_len_5_eq_105 : count_valid_lists_of_length_five = 105 :=
sorry

end num_valid_lists_len_5_eq_105_l337_337449


namespace probability_C_l337_337307

variable (P_A P_B P_D P_C : ℚ)

def total_probability := 1
def prob_A := P_A = 1 / 4
def prob_B := P_B = 1 / 3
def prob_D := P_D = 1 / 6
def prob_C := P_C = 1 / 4

theorem probability_C (hA : prob_A) (hB : prob_B) (hD : prob_D) : P_C = 1 - (P_A + P_B + P_D) :=
by
  sorry

end probability_C_l337_337307


namespace min_students_l337_337851

noncomputable def smallest_possible_number_of_students (b g : ℕ) : ℕ :=
if 3 * (3 * b) = 5 * (4 * g) then b + g else 0

theorem min_students (b g : ℕ) (h1 : 0 < b) (h2 : 0 < g) (h3 : 3 * (3 * b) = 5 * (4 * g)) :
  smallest_possible_number_of_students b g = 29 := sorry

end min_students_l337_337851


namespace max_value_abc_eq_l337_337889

noncomputable def max_abc (a b c : ℝ) (h : a + b + c = 2) (ha : 0 < a) (hb : 0 < b) (hc: 0 < c) : ℝ :=
  a^2 * b^3 * c^2

theorem max_value_abc_eq : 
  ∃ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 2 ∧ max_abc a b c (by assumption) (by assumption) (by assumption) = 128 / 2187 :=
sorry

end max_value_abc_eq_l337_337889


namespace calculate_expression_l337_337682

theorem calculate_expression : (-1:ℝ)^2 + (1/3:ℝ)^0 = 2 := by
  sorry

end calculate_expression_l337_337682


namespace athletes_numbers_l337_337301

theorem athletes_numbers (n : ℕ) (cond : ∀ (s : Finset ℕ), s.card = 12 → ∃ (x y : ℕ), x ∈ s ∧ y ∈ s ∧ knows x y) :
  ∃ (x y : ℕ), x ≠ y ∧ knows x y ∧ (x.leading_digit = y.leading_digit) := sorry

def leading_digit (x : ℕ) : ℕ := x / 10^(nat.log10 x)

def knows (a b : ℕ) : Prop := -- assuming non-negation of "they know each other"

end athletes_numbers_l337_337301


namespace sum_of_two_smallest_prime_factors_of_120_is_5_l337_337611

theorem sum_of_two_smallest_prime_factors_of_120_is_5 : 
  (∃ (p q : ℕ), p ∈ Multiset.filter Nat.Prime (120.factorization.toMultiset) ∧ q ∈ Multiset.filter Nat.Prime (120.factorization.toMultiset) ∧ p < q ∧ (p + q = 5)) := 
by 
  sorry

end sum_of_two_smallest_prime_factors_of_120_is_5_l337_337611


namespace sum_of_a2_and_a3_l337_337056

theorem sum_of_a2_and_a3 (S : ℕ → ℕ) (hS : ∀ n, S n = 3^n + 1) :
  S 3 - S 1 = 24 :=
by
  sorry

end sum_of_a2_and_a3_l337_337056


namespace max_triangles_in_4_free_graph_l337_337173

theorem max_triangles_in_4_free_graph (G : Type) [graph G] (h4_free : ∀ {K_4 : subgraph G}, false)
  (vertices_count : ∃ k : ℕ, card (vertices G) = 3 * k) : 
  ∃ k : ℕ, k > 0 → card (triangles G) ≤ k^3 :=
sorry

end max_triangles_in_4_free_graph_l337_337173


namespace unique_two_digit_solution_l337_337258

theorem unique_two_digit_solution :
  ∃! (u : ℕ), 9 < u ∧ u < 100 ∧ 13 * u % 100 = 52 := 
sorry

end unique_two_digit_solution_l337_337258


namespace Kyle_is_25_l337_337491

variable (Tyson_age : ℕ := 20)
variable (Frederick_age : ℕ := 2 * Tyson_age)
variable (Julian_age : ℕ := Frederick_age - 20)
variable (Kyle_age : ℕ := Julian_age + 5)

theorem Kyle_is_25 : Kyle_age = 25 := by
  sorry

end Kyle_is_25_l337_337491


namespace S_card_leq_six_l337_337156

variables {x y z a b c : ℝ} {p q r : ℂ}

noncomputable def S := 
  { (x, y, z) : ℂ × ℂ × ℂ | 
    a * x + b * y + c * z = p ∧
    a * x^2 + b * y^2 + c * z^2 = q ∧
    a * x^3 + b * y^3 + c * z^3 = r }

theorem S_card_leq_six 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) 
  (hp: p ∈ ℂ) 
  (hq: q ∈ ℂ) 
  (hr: r ∈ ℂ) : 
  fintype.card S ≤ 6 :=
sorry

end S_card_leq_six_l337_337156


namespace henry_oscillation_distance_l337_337446

noncomputable def distance_from_home_to_gym := 3 -- distance from home to the gym in km
noncomputable def ratio := 2 / 3 -- the ratio distance Henry walks each time

-- Define stable points for the oscillation
axiom stable_A : ℝ
axiom stable_B : ℝ

-- Given conditions and compute the distance between the stable points A and B
theorem henry_oscillation_distance
    (distance_home_gym : distance_from_home_to_gym = 3)
    (walk_ratio : ratio = 2 / 3)
    (stable_eq1 : stable_A = 1 / 3 * stable_B + 2)
    (stable_eq2 : stable_B = 1 / 3 * stable_A + 2) : 
    | stable_A - stable_B | = 1.20 := 
sorry

end henry_oscillation_distance_l337_337446


namespace map_distance_l337_337911

variable (map_distance_km : ℚ) (map_distance_inches : ℚ) (actual_distance_km: ℚ)

theorem map_distance (h1 : actual_distance_km = 136)
                     (h2 : map_distance_inches = 42)
                     (h3 : map_distance_km = 18.307692307692307) :
  (actual_distance_km * map_distance_inches / map_distance_km = 312) :=
by sorry

end map_distance_l337_337911


namespace parametric_curve_length_l337_337376

noncomputable def curve_length : ℝ :=
  ∫ t in 0..(2 * Real.pi), Real.sqrt ((3 * Real.cos t) ^ 2 + (-3 * Real.sin t) ^ 2)

theorem parametric_curve_length :
  curve_length = 6 * Real.pi :=
by
  rw [curve_length]
  -- further steps of the formal proof would go here
  sorry

end parametric_curve_length_l337_337376


namespace volunteers_distribution_l337_337003

theorem volunteers_distribution : 
  ∃ (arrangements : ℕ), arrangements = 240 
  ∧ ∀ (volunteers : ℕ) (exits : ℕ), volunteers = 5 ∧ exits = 4 
  → arrangements = 240 :=
by {
  exist 240,
  intros volunteers exits,
  intro h,
  apply and.intro,
  {
    exact 240,
  },
  {
    rintro ⟨hv, he⟩,
    cases hv,
    cases he,
    refl,
  },
}

end volunteers_distribution_l337_337003


namespace min_value_sin_cos_l337_337378

theorem min_value_sin_cos :
  ∀ x : ℝ, (sin x)^6 + 2 * (cos x)^6 ≥ (2 / 3) :=
by
  sorry

end min_value_sin_cos_l337_337378


namespace calculate_expression_l337_337679

theorem calculate_expression : (-1 : ℝ) ^ 2 + (1 / 3 : ℝ) ^ 0 = 2 := 
by
  sorry

end calculate_expression_l337_337679


namespace max_price_per_binder_l337_337698

def price_per_binder (total_money : ℝ) (entrance_fee : ℝ) (sales_tax_rate : ℝ) (num_binders : ℕ) : ℝ :=
  let max_money_available := total_money - entrance_fee
  let max_total_cost_before_tax := max_money_available / (1 + sales_tax_rate)
  real.floor (max_total_cost_before_tax / num_binders)

theorem max_price_per_binder : price_per_binder 160 5 0.08 18 = 7 := by
  sorry

end max_price_per_binder_l337_337698


namespace divide_square_50_unique_cuts_l337_337142

-- Conditions
def condition1 (square : Type) (cuts : ℕ) : Prop :=
cuts = 4 → (∃ squares : ℕ, squares = 5)

def condition2 (square : Type) (cuts : ℕ) : Prop :=
cuts = 6 → (∃ squares : ℕ, squares = 10)

-- Theorem statement
theorem divide_square_50_unique_cuts (square : Type) :
    (condition1 square 4) →
    (condition2 square 6) →
    (∃ cuts : ℕ, cuts = 10 ∧ (∃ congruentSquares : ℕ, congruentSquares = 50)) :=
begin
    intros h1 h2,
    -- sorry is a placeholder for the actual proof
    sorry
end

end divide_square_50_unique_cuts_l337_337142


namespace smallest_third_altitude_exists_l337_337591

theorem smallest_third_altitude_exists (h_P h_Q : ℕ) (h_P_eq_9 : h_P = 9) (h_Q_eq_3 : h_Q = 3) :
  ∃ h_R : ℕ, (PR \cdot h_R = 9 \cdot PQ) ∧ h_R = 3 :=
by
  sorry

end smallest_third_altitude_exists_l337_337591


namespace Genevieve_cherry_weight_l337_337730

theorem Genevieve_cherry_weight
  (cost_per_kg : ℕ) (short_of_total : ℕ) (amount_owned : ℕ) (total_kg : ℕ) :
  cost_per_kg = 8 →
  short_of_total = 400 →
  amount_owned = 1600 →
  total_kg = 250 :=
by
  intros h_cost_per_kg h_short_of_total h_amount_owned
  have h_equation : 8 * total_kg = 1600 + 400 := by
    rw [h_cost_per_kg, h_short_of_total, h_amount_owned]
    apply sorry -- This is where the exact proof mechanism would go
  sorry -- Skipping the remainder of the proof

end Genevieve_cherry_weight_l337_337730


namespace second_player_wins_optimal_play_l337_337971

theorem second_player_wins_optimal_play :
  ∀ (starting_pile : ℕ),
    starting_pile = 31 →
    (∃ optimal_strategy_p1 optimal_strategy_p2 : ℕ → ℕ × ℕ,
      (∀ n, n > 1 → let (a, b) := optimal_strategy_p1 n in a + b = n ∧ a > 0 ∧ b > 0) ∧
      (∀ n, n > 1 → let (a, b) := optimal_strategy_p2 n in a + b = n ∧ a > 0 ∧ b > 0) ∧
      (∃ n, n < 2 * starting_pile - 1 ∧
        (∀ k < n, (k % 2 = 0 → fst (optimal_strategy_p2 k) = 1 ∧ snd (optimal_strategy_p2 k) = 1) ∧
        (k % 2 = 1 → fst (optimal_strategy_p1 k) = 1 ∧ snd (optimal_strategy_p1 k) = 1)) →
        n % 2 = 0)) :=
begin
  sorry
end

end second_player_wins_optimal_play_l337_337971


namespace trajectory_parabola_or_line_l337_337912

theorem trajectory_parabola_or_line (fixed_point : Point) (fixed_line : Line)
(moving_point : Point) (d1 d2 : ℝ)
(h1 : d1 = distance moving_point fixed_point)
(h2 : d2 = distance moving_point fixed_line)
(h3 : d1 / d2 = 1) :
trajectory moving_point fixed_point fixed_line = (parabola fixed_point fixed_line) ∨ 
trajectory moving_point fixed_point fixed_line = (line_through fixed_point fixed_line) :=
sorry

end trajectory_parabola_or_line_l337_337912


namespace plane_part_covered_by_squares_l337_337992

structure Square :=
(center : ℝ × ℝ)
(side_length : ℝ)

def diagonal_rotation_scaling_center (sq : Square) : Square :=
{
  center := sq.center,
  side_length := sq.side_length * real.sqrt 2
}

-- Assume there is a given square N
variable (N : Square)

-- Define what N_1 means under the transformation described
def N_1 := diagonal_rotation_scaling_center N

-- Problem statement in Lean
theorem plane_part_covered_by_squares (sq : Square) :
  (∃ sq_diag, 
    sq_diag ∈ {d | ∃ P Q, sq.center = (P + Q) / 2 ∧ sq_diag = P - Q ∧ d • (1, -1),
                d • (1, 1) ∈ N.center }) → 
  sq ∈ N_1 :=
begin
  sorry
end

end plane_part_covered_by_squares_l337_337992


namespace equilateral_triangle_problem_l337_337363

-- Define the equilateral triangle ABC
structure Triangle :=
  (A B C : Point)
  (side_length : ℝ)
  (is_equilateral : (dist A B = side_length) ∧ (dist B C = side_length) ∧ (dist C A = side_length))

-- Define a node structure to hold positions and values
structure Node :=
  (position : Point)
  (value : ℝ)

-- Function for dividing a side into n parts
def divide_side (A B : Point) (n : ℕ) : List Point := sorry

-- Define the condition for the rhombus
def rhombus_condition (n1 n2 n3 n4 : Node) : Prop :=
  n1.value + n3.value = n2.value + n4.value

-- Define the total sum S of the numbers at all nodes
def total_sum (nodes : List Node) : ℝ :=
  nodes.foldr (λ n acc => n.value + acc) 0

noncomputable def shortest_distance (nodes : List Node) : ℝ := sorry

theorem equilateral_triangle_problem (ABC : Triangle) (n : ℕ) (a b c : ℝ)
  (nodes : List Node)
  (A_node : Node := {position := ABC.A, value := a})
  (B_node : Node := {position := ABC.B, value := b})
  (C_node : Node := {position := ABC.C, value := c})
  (h_nodes : ∀ (A B : Point) (n₁ n₂ : Node),
    (A ≠ B) → (divide_side A B n).map (λ p => ∃ (n : Node), n.position = p ∧ nodes.contains n))
  (h_rhombus : ∀ (n₁ n₂ n₃ n₄ : Node), rhombus_condition n₁ n₂ n₃ n₄)
  : total_sum nodes = (1 / 6) * (n + 1) * (n + 2) * (a + b + c) ∧
    match n % 2 with
    | 0 => shortest_distance nodes = (1 / 2) * Real.sqrt 3
    | _ => shortest_distance nodes = (1 / 2) * Real.sqrt (3 + 1 / n^2) :=
sorry

end equilateral_triangle_problem_l337_337363


namespace simplified_value_of_sum_l337_337274

theorem simplified_value_of_sum :
  (-1)^(2004) + (-1)^(2005) + 1^(2006) - 1^(2007) = -2 := by
  sorry

end simplified_value_of_sum_l337_337274


namespace geometry_problem_l337_337864

open EuclideanGeometry

def rectangle (A B C D : Point) : Prop :=
  parallel A D B C ∧ parallel A B C D

def tangent (P : Point) (ω : Circle) : Line :=
  let t := (tangent_line P ω) in t -- Assuming tangent_line returns the tangent at point P

theorem geometry_problem (A B C D P K : Point) (ω : Circle)
  (h1 : rectangle A B C D)
  (h2 : diameter ω A B)
  (h3 : line_intersects_circle_two_points AC ω P)
  (h4 : tangent P ω D K C)
  (h5 : KD = 36) :
  AD = 24 :=
  sorry

end geometry_problem_l337_337864


namespace evaluate_expression_l337_337455

-- Defining the primary condition
def condition (x : ℝ) : Prop := x > 3

-- Definition of the expression we need to evaluate
def expression (x : ℝ) : ℝ := abs (1 - abs (x - 3))

-- Stating the theorem
theorem evaluate_expression (x : ℝ) (h : condition x) : expression x = abs (4 - x) := 
by 
  -- Since the problem only asks for the statement, the proof is left as sorry.
  sorry

end evaluate_expression_l337_337455


namespace find_arithmetic_sequence_numbers_l337_337260

theorem find_arithmetic_sequence_numbers 
  (a d : ℝ) 
  (h1: (a - d) + a + (a + d) = 15) 
  (h2: (a + 3)^2 = (a - d + 1) * (a + d + 9)) :
  {x // x > 0 ∧ x ∈ {a - d, a, a + d}} = {1, 5, 9} :=
by
  sorry

end find_arithmetic_sequence_numbers_l337_337260


namespace factorization_of_P_l337_337767

open Polynomial

noncomputable def P : Polynomial ℤ := X^4 + 3*X^3 - 15*X^2 - 19*X + 30

theorem factorization_of_P :
  ∃ (a b c d : ℤ), P = (X - C a) * (X - C b) * (X - C c) * (X - C d) ∧ a = -2 ∧ b = -5 ∧ c = 1 ∧ d = 3 :=
by
  use [-2, -5, 1, 3]
  split
  · exact Polynomial.ext (λ n, sorry)
  · split; norm_num

end factorization_of_P_l337_337767


namespace proof_problem_l337_337189

theorem proof_problem (x : ℝ) 
    (h1 : (x - 1) * (x + 1) = x^2 - 1)
    (h2 : (x - 1) * (x^2 + x + 1) = x^3 - 1)
    (h3 : (x - 1) * (x^3 + x^2 + x + 1) = x^4 - 1)
    (h4 : (x - 1) * (x^4 + x^3 + x^2 + x + 1) = -2) :
    x^2023 = -1 := 
by 
  sorry -- Proof is omitted

end proof_problem_l337_337189


namespace Genevieve_cherry_weight_l337_337732

theorem Genevieve_cherry_weight
  (cost_per_kg : ℕ) (short_of_total : ℕ) (amount_owned : ℕ) (total_kg : ℕ) :
  cost_per_kg = 8 →
  short_of_total = 400 →
  amount_owned = 1600 →
  total_kg = 250 :=
by
  intros h_cost_per_kg h_short_of_total h_amount_owned
  have h_equation : 8 * total_kg = 1600 + 400 := by
    rw [h_cost_per_kg, h_short_of_total, h_amount_owned]
    apply sorry -- This is where the exact proof mechanism would go
  sorry -- Skipping the remainder of the proof

end Genevieve_cherry_weight_l337_337732


namespace max_distance_AB_is_2_l337_337126

-- Definitions for the parametric and Cartesian equations
def C1_param (t : ℝ) : ℝ × ℝ :=
  (Real.cos t, 1 + Real.sin t)

def C1_cart (x y : ℝ) : Prop :=
  x^2 + (y - 1)^2 = 1

def C2_cart (x y : ℝ) : Prop :=
  x^2 + (y - 2)^2 = 4

-- Polar equations
def polar_C1 (θ : ℝ) : ℝ :=
  2 * Real.sin θ

def polar_C2 (θ : ℝ) : ℝ :=
  4 * Real.sin θ

def ray_l (α : ℝ) : Prop :=
  0 < α ∧ α < π 

-- Points A and B as intersections in polar coordinates
def point_A (α : ℝ) : ℝ × ℝ :=
  (2 * Real.sin α, α)

def point_B (α : ℝ) : ℝ × ℝ :=
  (4 * Real.sin α, α)

-- Distance between the points A and B
def distance_AB (α : ℝ) : ℝ :=
  |(4 * Real.sin α) - (2 * Real.sin α)|

-- Proof that the maximum distance is 2
theorem max_distance_AB_is_2 : ∀ (α : ℝ), ray_l α → distance_AB α = 2 :=
by 
  sorry

end max_distance_AB_is_2_l337_337126


namespace ratio_of_segments_l337_337045

theorem ratio_of_segments (a b : ℝ) (h : 2 * a = 3 * b) : a / b = 3 / 2 :=
sorry

end ratio_of_segments_l337_337045


namespace possible_remainders_l337_337157

theorem possible_remainders (p : ℕ) (r : ℕ) (x : ℕ → ℤ)
  (hp : Nat.Prime p) (hp3 : 3 ≤ p) (hr : 0 ≤ r) (hrp : r ≤ p - 3)
  (h_sum : ∀ k, 1 ≤ k → k ≤ p - 2 → (∑ i in Finset.range (p - 1 + r), x i ^ k) % p = r) :
  ∃ s : Finset ℤ, 
    s = (Finset.range r).image (λ _, 1) ∪ (Finset.range (p-1)).image (λ _, 0) ∨
    s = (Finset.range r).image (λ _, 1) ∪ (Finset.range (p-1)).image (λ i, i+1 % p) :=
sorry

end possible_remainders_l337_337157


namespace find_largest_expression_l337_337352

theorem find_largest_expression :
  let a := real.sqrt (real.cbrt (7 * 8))
  let b := real.sqrt (8 * real.cbrt 7)
  let c := real.sqrt (7 * real.cbrt 8)
  let d := real.cbrt (7 * real.sqrt 8)
  let e := real.cbrt (8 * real.sqrt 7)
  b > a ∧ b > c ∧ b > d ∧ b > e :=
by
  sorry

end find_largest_expression_l337_337352


namespace cyclic_quadrilateral_angles_cyclic_quadrilateral_lengths_l337_337870

theorem cyclic_quadrilateral_angles (ABCD : Quadrilateral)
  (E : Point) (F : Point) (M : Point) (N : Point)
  (h_cyclic : ABCD.cyclic)
  (h_diag_inter : ∃E, E ∈ AC ∧ E ∈ BD)
  (h_perp_diags : AC ⊥ BD)
  (h_eq_sides : AB = AC ∧ AC = BD)
  (h_DF : DF ⊥ BD ∧ F ∈ (BA)⁺)
  (h_bisect : angle_bisector (∠BFD) M AD ∧ angle_bisector (∠BFD) N BD) :
  ∠BAD = 3 * ∠DAC := 
sorry

theorem cyclic_quadrilateral_lengths (ABCD : Quadrilateral)
  (E : Point) (F : Point) (M : Point) (N : Point)
  (h_cyclic : ABCD.cyclic)
  (h_diag_inter : ∃E, E ∈ AC ∧ E ∈ BD)
  (h_perp_diags : AC ⊥ BD)
  (h_eq_sides : AB = AC ∧ AC = BD)
  (h_DF : DF ⊥ BD ∧ F ∈ (BA)⁺)
  (h_bisect : angle_bisector (∠BFD) M AD ∧ angle_bisector (∠BFD) N BD)
  (h_eq_len : MN = MD) :
  BF = CD + DF := 
sorry

end cyclic_quadrilateral_angles_cyclic_quadrilateral_lengths_l337_337870


namespace base3_vs_base8_digits_l337_337779

theorem base3_vs_base8_digits (n : ℕ) (h : n = 987) :
  (nat.digits 3 n).length - (nat.digits 8 n).length = 3 :=
by
  rw h
  sorry

end base3_vs_base8_digits_l337_337779


namespace decreasing_function_range_l337_337568

open Real

noncomputable def g (a x : ℝ) : ℝ := a*x^3 - x

theorem decreasing_function_range (a : ℝ) :
  (∀ x : ℝ, (3 * a * x^2 - 1) ≤ 0) → (a ∈ set.Iic 0) := 
sorry

end decreasing_function_range_l337_337568


namespace valid_shirt_tie_combinations_l337_337216

theorem valid_shirt_tie_combinations
  (num_shirts : ℕ)
  (num_ties : ℕ)
  (restricted_shirts : ℕ)
  (restricted_ties : ℕ)
  (h_shirts : num_shirts = 8)
  (h_ties : num_ties = 7)
  (h_restricted_shirts : restricted_shirts = 3)
  (h_restricted_ties : restricted_ties = 2) :
  num_shirts * num_ties - restricted_shirts * restricted_ties = 50 := by
  sorry

end valid_shirt_tie_combinations_l337_337216


namespace max_sum_small_numbers_l337_337572

-- Definition: A number is either "big" or "small"
def is_big_or_small (n : ℕ) (neighbors : List ℕ) :=
  (n > neighbors.head! ∧ n > neighbors.tail.head!) ∨ (n < neighbors.head! ∧ n < neighbors.tail.head!)

-- Given the conditions
def conditions (circle: List ℕ) : Prop :=
  circle.length = 8 ∧ 
  ∀ i, is_big_or_small (circle.nth_le i sorry) [circle.nth_le ((i-1) % 8) sorry, circle.nth_le ((i+1) % 8) sorry]

-- Proof statement: The maximum possible sum of the small numbers is 13
theorem max_sum_small_numbers : ∀ circle: List ℕ, conditions circle → ∑ i in (Finset.filter (λ n, n = circle.nth_le i sorry ∧ n < circle.nth_le ((i-1) % 8) sorry ∧ n < circle.nth_le ((i+1) % 8) sorry) (Finset.range 8)), circle.nth_le i sorry = 13 :=
by
  -- Here you would generally prove the theorem
  sorry

end max_sum_small_numbers_l337_337572


namespace number_of_ways_to_select_planning_committee_l337_337329

-- Define the conditions
def num_ways_to_select_two_person_committee (n : ℕ) : ℕ := n * (n - 1) / 2

def num_ways_to_select_four_person_committee (n : ℕ) : ℕ := n * (n - 1) * (n - 2) * (n - 3) / 24

-- The proof problem
theorem number_of_ways_to_select_planning_committee :
  (∃ n : ℕ, num_ways_to_select_two_person_committee n = 15) →
  (num_ways_to_select_four_person_committee 6 = 15) :=
by
  intro h,
  have h_n : num_ways_to_select_two_person_committee 6 = 15 := by
    sorry,
  rw h_n,
  sorry

end number_of_ways_to_select_planning_committee_l337_337329


namespace det_matrix_A_l337_337296

noncomputable def matrix_A {n : ℕ} (α β : Fin n → Fin n) [Fact (2 ≤ n)] : Matrix (Fin n) (Fin n) ℕ :=
  λ i j => (1 + (α i).val * (β j).val) ^ (n - 1)

theorem det_matrix_A {n : ℕ} (α β : Fin n → Fin n) [Fact (2 ≤ n)] (hα : ∀ i, ∃ j, α j = i) (hβ : ∀ i, ∃ j, β j = i) :
  ∃ (s : ℤ), det (matrix_A α β) = s * (n - 1)! ^ n := by
  sorry

end det_matrix_A_l337_337296


namespace cyclist_speed_ratio_l337_337601

variables (k r t v1 v2 : ℝ)
variable (h1 : v1 = 2 * v2) -- Condition 5

-- When traveling in the same direction, relative speed is v1 - v2 and they cover 2k miles in 3r hours
variable (h2 : 2 * k = (v1 - v2) * 3 * r)

-- When traveling in opposite directions, relative speed is v1 + v2 and they pass each other in 2t hours
variable (h3 : 2 * k = (v1 + v2) * 2 * t)

theorem cyclist_speed_ratio (h1 : v1 = 2 * v2) (h2 : 2 * k = (v1 - v2) * 3 * r) (h3 : 2 * k = (v1 + v2) * 2 * t) :
  v1 / v2 = 2 :=
sorry

end cyclist_speed_ratio_l337_337601


namespace pascal_triangle_contains_53_l337_337825

theorem pascal_triangle_contains_53 (n : ℕ) :
  (∃ k, binomial n k = 53) ↔ n = 53 := 
sorry

end pascal_triangle_contains_53_l337_337825


namespace num_students_scoring_high_l337_337102
noncomputable def normal_distribution_students_scoring_high (μ σ n : ℕ) :=
  let distribution := Normal μ σ
  let probability := 0.5 * (1 - 0.6826)
  let expected_students := probability * n
  expected_students

theorem num_students_scoring_high {μ σ : ℕ} {n : ℕ} (hμ : μ = 120) (hσ : σ = 10) (hn : n = 40) :
  normal_distribution_students_scoring_high μ σ n ≈ 6 :=
by
  rw [hμ, hσ, hn]
  dsimp [normal_distribution_students_scoring_high]
  norm_num
  sorry

end num_students_scoring_high_l337_337102


namespace max_angle_between_BD1_BDC1_l337_337563

noncomputable def maximum_angle : ℝ :=
  Real.arcsin (1 / 3)

theorem max_angle_between_BD1_BDC1 (x : ℝ) (h : x > 0) :
  ∃ φ : ℝ, φ = Real.arcsin (1 / 3) ∧ 
    φ = arcsin ((x) / real.sqrt ((2 + x^2) * (1 + 2 * x^2))) :=
begin
  use maximum_angle,
  split,
  { refl }, -- show φ = maximum_angle
  { sorry } -- prove φ = arcsin ((x) / real.sqrt ((2 + x^2) * (1 + 2 * x^2)))
end

end max_angle_between_BD1_BDC1_l337_337563


namespace range_of_p_l337_337758

def sequence_sum (n : ℕ) : ℚ := (-1) ^ (n + 1) * (1 / 2 ^ n)

def a_n (n : ℕ) : ℚ :=
  if h : n = 0 then sequence_sum 1 else
  sequence_sum n - sequence_sum (n - 1)

theorem range_of_p (p : ℚ) : 
  (∃ n : ℕ, 0 < n ∧ (p - a_n n) * (p - a_n (n + 1)) < 0) ↔ 
  - 3 / 4 < p ∧ p < 1 / 2 :=
sorry

end range_of_p_l337_337758


namespace bricks_needed_for_wall_l337_337447

theorem bricks_needed_for_wall :
  let brick_length := 25 
  let brick_width := 11.25 
  let brick_height := 6 
  let wall_length := 800 
  let wall_height := 600 
  let wall_thickness := 22.5 
  let volume_wall := wall_length * wall_height * wall_thickness
  let volume_brick := brick_length * brick_width * brick_height 
  volume_wall / volume_brick ≈ 6400 :=
by 
  sorry

end bricks_needed_for_wall_l337_337447


namespace possible_integral_values_BC_l337_337171

noncomputable def number_of_possible_BC (ABC : Triangle) (AB : ℝ) (bisector_D : Point) (E F : Point) : ℕ :=
  if h : (AB = 7) ∧ (bisector_D ∈ lineThrough ABC B) ∧ 
            (E ∈ lineThrough ABC A) ∧ (F ∈ lineThrough ABC B) ∧ 
            parallel (lineThrough ABC A) (lineThrough E F) ∧ 
            dividesTriangleIntoEqualAreas ABC E F 3 
  then 13
  else 0

theorem possible_integral_values_BC (ABC : Triangle) (AB : ℝ) (bisector_D : Point) 
  (E F : Point) (BC : ℝ) : 
  (AB = 7) ∧ (bisector_D ∈ lineThrough ABC B) ∧ 
  (E ∈ lineThrough ABC A) ∧ (F ∈ lineThrough ABC B) ∧ 
  parallel (lineThrough ABC A) (lineThrough E F) ∧ 
  dividesTriangleIntoEqualAreas ABC E F 3 ∧ 
  int.cast_trunc BC ∈ set.Icc 8 20 ↔ (number_of_possible_BC ABC AB bisector_D E F) = 13 := 
by {
  sorry
}

end possible_integral_values_BC_l337_337171


namespace quartic_trinomial_l337_337400

theorem quartic_trinomial (m : ℤ) :
  (∃ (x y : ℤ), x ≠ 0 ∧ y ≠ 0 ∧ (m + 1) * x^3 * y^(abs m) + x * y^2 + 8^5 = 0) →
  (m = 1 ∨ m = -1) :=
by
  sorry

end quartic_trinomial_l337_337400


namespace construct_rectangle_l337_337353

-- Define the essential properties of the rectangles
structure Rectangle where
  length : ℕ
  width : ℕ 

-- Define the given rectangles
def r1 : Rectangle := ⟨7, 1⟩
def r2 : Rectangle := ⟨6, 1⟩
def r3 : Rectangle := ⟨5, 1⟩
def r4 : Rectangle := ⟨4, 1⟩
def r5 : Rectangle := ⟨3, 1⟩
def r6 : Rectangle := ⟨2, 1⟩
def s  : Rectangle := ⟨1, 1⟩

-- Hypothesis for condition that length of each side of resulting rectangle should be > 1
def validSide (rect : Rectangle) : Prop :=
  rect.length > 1 ∧ rect.width > 1

-- The proof statement
theorem construct_rectangle : 
  (∃ rect1 rect2 rect3 rect4 : Rectangle, 
      rect1 = ⟨7, 1⟩ ∧ rect2 = ⟨6, 1⟩ ∧ rect3 = ⟨5, 1⟩ ∧ rect4 = ⟨4, 1⟩) →
  (∃ rect5 rect6 : Rectangle, 
      rect5 = ⟨3, 1⟩ ∧ rect6 = ⟨2, 1⟩) →
  (∃ square : Rectangle, 
      square = ⟨1, 1⟩) →
  (∃ compositeRect : Rectangle, 
      compositeRect.length = 7 ∧ 
      compositeRect.width = 4 ∧ 
      validSide compositeRect) :=
sorry

end construct_rectangle_l337_337353


namespace train_cross_bridge_time_l337_337654

noncomputable def speed_ms : Float := (60 * 1000) / 3600
noncomputable def total_distance : Float := 450 + 200
noncomputable def time_to_cross_bridge : Float := total_distance / speed_ms

theorem train_cross_bridge_time :
  time_to_cross_bridge ≈ 38.99 :=
by
  sorry

end train_cross_bridge_time_l337_337654


namespace sum_y_sequence_l337_337167

theorem sum_y_sequence (n : ℕ) (h : n > 0) : 
  let y : ℕ → ℕ := 
    λ k, nat.rec_on k 0 (λ k' y_k, nat.cases_on k' 1 
    (λ k'' ih ih', ((n-1) * ih + (n + k'') * ih') / (k'' + 1))) 
  in 
  ∑ k in finset.range (2*n + 1), y k = 1 + (n-1)*(2*n^2 + 3*n + 1) :=
sorry

end sum_y_sequence_l337_337167


namespace range_of_f_l337_337240

variable (x : ℝ)

def f (x : ℝ) := 2 * x - x^2

theorem range_of_f : 
  Set.range (λ x, f x) = Set.Icc (-3 : ℝ) (1 : ℝ) :=
by
  sorry

end range_of_f_l337_337240


namespace triangle_perimeter_l337_337115

theorem triangle_perimeter (A B C X Y W Z : ℝ) 
(AB_eq : AB = 10) 
(angle_C_90 : ∠C = 90) 
(points_on_circle : X, Y, W are on a circle) 
(Z_not_on_circle : Z is not on the circle) 
(construction : squares ABXY and CBWZ are constructed outside of the triangle):
  perimeter_of_triangle ABC = 10 + 10*sqrt(2) :=
sorry

end triangle_perimeter_l337_337115


namespace modulus_eq_sqrt2_square_eq_2i_conjugate_not_minus1_plus_i_root_of_eqn_l337_337282

noncomputable def z : ℂ := 2 / (1 - I)

theorem modulus_eq_sqrt2 : complex.abs z = real.sqrt 2 :=
by
  sorry

theorem square_eq_2i : z^2 = (2 : ℂ) * I :=
by
  sorry

theorem conjugate_not_minus1_plus_i : complex.conj z ≠ -1 + I :=
by
  sorry

theorem root_of_eqn : z^2 - 2 * z + 2 = 0 :=
by
  sorry

end modulus_eq_sqrt2_square_eq_2i_conjugate_not_minus1_plus_i_root_of_eqn_l337_337282


namespace election_proof_l337_337259

noncomputable def total_votes : ℕ := 6480

lemma percentage_third_candidate :
  let total_votes := 6480 in
  100 - 42 - 37 = 21 := by
  simp

lemma votes_winner : 
  0.42 * total_votes = 2722 := by
  simp [total_votes]

lemma votes_second_candidate : 
  0.37 * total_votes = 2398 := by
  simp [total_votes]

lemma votes_third_candidate : 
  total_votes - 2722 - 2398 = 1360 := by
  simp [total_votes]

-- Combined statement proving all the points
theorem election_proof :
  let total_votes := 6480 in
  let winner_votes := 0.42 * total_votes in
  let second_votes := 0.37 * total_votes in
  let third_votes := total_votes - winner_votes - second_votes in
  100 - 42 - 37 = 21 ∧
  winner_votes = 2722 ∧
  second_votes = 2398 ∧
  third_votes = 1360 := by
  simp [total_votes]
  repeat { sorry } -- Proof parts to be completed

end election_proof_l337_337259


namespace min_socks_removal_l337_337638

-- Define the math problem in terms of Lean
theorem min_socks_removal (n : ℕ) (h : n = 2019) : 
  ∃ k : ℕ, k ≥ 2019 + 1 ∧ 
           (∀ l : ℕ, l < k → 
             ∀ (socks : Fin l → Fin (2 * n)), 
               (∀ i j, socks i ≠ socks j) → 
               ∃ i j, socks i = socks j) :=
by {
  use 2020,
  have h1 : 2020 ≥ 2019 + 1, by sorry,
  split,
  exact h1,
  intros l hl socks hdistinct,
  sorry
}

end min_socks_removal_l337_337638


namespace minimum_stamps_satisfying_congruences_l337_337203

theorem minimum_stamps_satisfying_congruences (n : ℕ) :
  (n % 4 = 3) ∧ (n % 5 = 2) ∧ (n % 7 = 1) → n = 107 :=
by
  sorry

end minimum_stamps_satisfying_congruences_l337_337203


namespace inequality_solution_l337_337113

theorem inequality_solution (a : ℝ) (h : ∀ x : ℝ, (a + 1) * x > a + 1 ↔ x < 1) : a < -1 :=
sorry

end inequality_solution_l337_337113


namespace find_a_l337_337096

noncomputable def a_value (a b : ℝ) : Prop := a ^ b = b ^ a ∧ b = 4 * a

theorem find_a (a : ℝ) (b : ℝ) (h1 : 0 < a) (h2 : a_value a b) : a = real.cbrt 4 :=
by
  sorry

end find_a_l337_337096


namespace house_numbers_count_l337_337707

def is_prime (n : ℕ) : Prop := 
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the conditions
def two_digit_primes : list ℕ := [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

def three_digit_primes : list ℕ := [101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 
  193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 
  313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 
  443, 449, 457, 461, 463, 467, 479, 487, 491, 499]

-- Define the proof problem
theorem house_numbers_count : 
  (list.length two_digit_primes) * (list.length three_digit_primes) = 1302 := 
by
  sorry

end house_numbers_count_l337_337707


namespace find_polynomials_l337_337519

variables {K : Type*} [Field K]

noncomputable def Lagrange (a : K) (n : ℕ) (a_vals b_vals : Fin n → K) : K :=
  ∑ i : Fin n, b_vals i * ∏ j in Finset.univ.filter (λ j, j ≠ i), (a - a_vals j) / (a_vals i - a_vals j)

noncomputable def PolySolution (n : ℕ) (a_vals b_vals : Fin n → K) (c : K) : K[X] :=
  (λ X, (Lagrange X n a_vals b_vals) + c * ∏ i, (X - polynomial.C (a_vals i)))

theorem find_polynomials (a_vals b_vals : Fin n → K) :
  ∃ c : K, ∀ P : K[X], (∀ i : Fin n, P.eval (a_vals i) = b_vals i) ↔
    ∃ c : K, P = (PolySolution n a_vals b_vals c) :=
begin
  sorry
end

end find_polynomials_l337_337519


namespace remainder_mod_68_l337_337716

theorem remainder_mod_68 (n : ℕ) (h : 67^67 + 67 ≡ 66 [MOD n]) : n = 68 := 
by 
  sorry

end remainder_mod_68_l337_337716


namespace roots_of_equation_l337_337265

theorem roots_of_equation (
  x y: ℝ
) (h1: x + y = 10) (h2: |x - y| = 12):
  (x = 11 ∧ y = -1) ∨ (x = -1 ∧ y = 11) ↔ ∃ (a b: ℝ), a = 11 ∧ b = -1 ∨ a = -1 ∧ b = 11 ∧ a^2 - 10*a - 22 = 0 ∧ b^2 - 10*b - 22 = 0 := 
by sorry

end roots_of_equation_l337_337265


namespace john_total_distance_l337_337486

-- Define the conditions
def john_speed_alone : ℝ := 4 -- miles per hour
def john_speed_with_dog : ℝ := 6 -- miles per hour
def time_with_dog : ℝ := 0.5 -- hours
def time_alone : ℝ := 0.5 -- hours

-- Calculate distances based on conditions and prove the total distance
theorem john_total_distance : 
  john_speed_with_dog * time_with_dog + john_speed_alone * time_alone = 5 := 
by 
  calc
    john_speed_with_dog * time_with_dog + john_speed_alone * time_alone
    = 6 * 0.5 + 4 * 0.5 : by sorry
    ... = 3 + 2 : by sorry
    ... = 5 : by sorry

end john_total_distance_l337_337486


namespace find_original_price_l337_337933

-- Given conditions:
-- 1. 10% cashback
-- 2. $25 mail-in rebate
-- 3. Final cost is $110

def original_price (P : ℝ) (cashback : ℝ) (rebate : ℝ) (final_cost : ℝ) :=
  final_cost = P - (cashback * P + rebate)

theorem find_original_price :
  ∀ (P : ℝ), original_price P 0.10 25 110 → P = 150 :=
by
  sorry

end find_original_price_l337_337933


namespace Marcella_shoes_l337_337527

theorem Marcella_shoes (pairs_before : ℕ) (shoes_lost : ℕ) : 
  pairs_before = 24 → shoes_lost = 9 → 
  ∃ pairs_left : ℕ, pairs_left = 15 :=
by
  intros h_pairs h_lost
  use 15
  sorry

end Marcella_shoes_l337_337527


namespace coefficient_a_neg1_l337_337011

noncomputable def binom (n k : ℕ) : ℕ := nat.choose n k

theorem coefficient_a_neg1 : 
  let a := λ (x : ℝ), x in
  let y := λ (x : ℝ), -x⁻¹⁄² in
  ∑ k in finset.range 9, (binom 8 k * (a 1)^(8 - k) * (y 1)^k) = 28 :=
by
  sorry

end coefficient_a_neg1_l337_337011


namespace seven_digit_number_insertion_l337_337622

theorem seven_digit_number_insertion (num : ℕ) (h : num = 52115) : (∃ (count : ℕ), count = 21) :=
by 
  sorry

end seven_digit_number_insertion_l337_337622


namespace Alyssa_has_37_balloons_l337_337662

variable (Sandy_balloons : ℕ) (Sally_balloons : ℕ) (Total_balloons : ℕ)

-- Conditions
axiom Sandy_Condition : Sandy_balloons = 28
axiom Sally_Condition : Sally_balloons = 39
axiom Total_Condition : Total_balloons = 104

-- Definition of Alyssa's balloons
def Alyssa_balloons : ℕ := Total_balloons - (Sandy_balloons + Sally_balloons)

-- The proof statement 
theorem Alyssa_has_37_balloons 
: Alyssa_balloons Sandy_balloons Sally_balloons Total_balloons = 37 :=
by
  -- The proof body will be placed here, but we will leave it as a placeholder for now
  sorry

end Alyssa_has_37_balloons_l337_337662


namespace unique_geometric_four_digit_number_l337_337371

theorem unique_geometric_four_digit_number :
  (∃! abcd : ℕ,
    ∃ a b c d : ℕ,
      1 ≤ a ∧ a ≤ 9 ∧
      0 ≤ b ∧ b ≤ 9 ∧
      0 ≤ c ∧ c ≤ 9 ∧
      0 ≤ d ∧ d ≤ 9 ∧
      abcd = 1000 * a + 100 * b + 10 * c + d ∧
      let ab := 10 * a + b in
      let bc := 10 * b + c in
      let cd := 10 * c + d in
      ∃ r : ℚ, 
        bc = r * ab ∧ 
        cd = r * bc) :=
sorry

end unique_geometric_four_digit_number_l337_337371


namespace cube_root_less_than_5_l337_337090

theorem cube_root_less_than_5 :
  {n : ℕ | n > 0 ∧ (∃ m : ℝ, m^3 = n ∧ m < 5)}.finite.card = 124 :=
by
  sorry

end cube_root_less_than_5_l337_337090


namespace proof_equation_approximation_l337_337272

theorem proof_equation_approximation :
  (0.47 * 1602 / 2 - 0.36 * 1513 * 3 + (3^5 - 88)) + 63 * (sqrt 25) - (97 / 3)^2 = -1832.22 :=
by
  sorry

end proof_equation_approximation_l337_337272


namespace equilateral_triangle_fold_3_layers_thirty_sixty_ninety_triangle_cannot_fold_3_layers_every_triangle_fold_2020_layers_l337_337630

-- (a) Equilateral triangle can be folded into a uniform thickness of 3 layers.
theorem equilateral_triangle_fold_3_layers :
  ∀ (A B C : Point), is_equilateral_triangle A B C → can_be_folded_into_uniform_thickness A B C 3 :=
by
  sorry

-- (b) 30-60-90 triangle cannot be folded into a uniform thickness of 3 layers.
theorem thirty_sixty_ninety_triangle_cannot_fold_3_layers :
  ∀ (D E F : Point), is_30_60_90_triangle D E F → ¬can_be_folded_into_uniform_thickness D E F 3 :=
by
  sorry

-- (c) Every triangle can be folded into a uniform thickness of 2020 layers.
theorem every_triangle_fold_2020_layers :
  ∀ (G H I : Point), is_triangle G H I → can_be_folded_into_uniform_thickness G H I 2020 :=
by
  sorry

end equilateral_triangle_fold_3_layers_thirty_sixty_ninety_triangle_cannot_fold_3_layers_every_triangle_fold_2020_layers_l337_337630


namespace max_min_sum_l337_337920

theorem max_min_sum {x y z : ℝ} (h : 5 * (x + y + z) = x^2 + y^2 + z^2) :
  let M := (81 / 2 : ℝ)
  let m := (-5 / 2 : ℝ)
  M + 10 * m = 31 :=
by
  let M := 81 / 2
  let m := -5 / 2
  show M + 10 * m = 31
  sorry

end max_min_sum_l337_337920


namespace equilateral_triangle_not_always_implies_l337_337503

-- Definitions of key points
variable (A B C O G H G1 G2 G3 : Type)

-- Conditions
axiom circumcenter : ∀ (A B C : Type), O 
axiom centroid : ∀ (A B C : Type), G 
axiom orthocenter : ∀ (A B C : Type), H 

axiom centroid_OBC : ∀ (O B C : Type), G1 
axiom centroid_GCA : ∀ (G C A : Type), G2 
axiom centroid_HAB : ∀ (H A B : Type), G3 

-- Proof problem
theorem equilateral_triangle_not_always_implies : 
  (equilateral (triangle G1 G2 G3)) → ¬(equilateral (triangle A B C)) := 
by 
  sorry

end equilateral_triangle_not_always_implies_l337_337503


namespace range_of_m_min_value_of_2a_plus_3_over_2_b_l337_337766

-- Part (I): The range of m
theorem range_of_m (x m : ℝ) : (∀ x, |x+2| + |6-x| - m ≥ 0) → m ≤ 8 :=
by
  intro h1
  have h2 : ∀ x, |x + 2| + |6 - x| ≥ 8 := by
    intro x
    sorry
  sorry

-- Part (II): The minimum value of 2a + (3/2)b
theorem min_value_of_2a_plus_3_over_2_b {a b : ℝ} (h0 : 0 < a) (h1 : 0 < b) (n : ℝ) (h2 : n = 8) 
  (h3 : (8 / (3 * a + b)) + (2 / (a + 2 * b)) = n) : 2 * a + (3 / 2) * b ≥ 9 / 8 :=
by
  sorry

end range_of_m_min_value_of_2a_plus_3_over_2_b_l337_337766


namespace chimney_bricks_l337_337344

theorem chimney_bricks (x : ℕ) 
  (h1 : Brenda_rate = x / 8) 
  (h2 : Brandon_rate = x / 12) 
  (h3 : Brian_rate = x / 16) 
  (h4 : effective_combined_rate = (Brenda_rate + Brandon_rate + Brian_rate) - 15) 
  (h5 : total_time = 4) :
  (4 * effective_combined_rate) = x := 
  sorry

end chimney_bricks_l337_337344


namespace frogs_meet_time_proven_l337_337232

-- Define the problem
def frogs_will_meet_at_time : Prop :=
  ∃ (meet_time : Nat),
    let initial_time := 12 * 60 -- 12:00 PM in minutes
    let initial_distance := 2015
    let green_frog_jump := 9
    let blue_frog_jump := 8 
    let combined_reduction := green_frog_jump + blue_frog_jump
    initial_distance % combined_reduction = 0 ∧
    meet_time == initial_time + (2 * (initial_distance / combined_reduction))

theorem frogs_meet_time_proven (h : frogs_will_meet_at_time) : meet_time = 15 * 60 + 56 :=
sorry

end frogs_meet_time_proven_l337_337232


namespace circles_intersect_and_inequality_l337_337599

variable {R r d : ℝ}

theorem circles_intersect_and_inequality (hR : R > r) (h_intersect: R - r < d ∧ d < R + r) : R - r < d ∧ d < R + r :=
by
  exact h_intersect

end circles_intersect_and_inequality_l337_337599


namespace diff_local_face_value_7_in_657_90385_diff_local_face_value_3_in_3578_90365_l337_337608

theorem diff_local_face_value_7_in_657_90385 :
  let face_value_7 := 7
  ∧ let local_value_7 := 7 * 10
  ∧ let diff_7 := local_value_7 - face_value_7
  in diff_7 = 63 :=
by
  let face_value_7 := 7
  let local_value_7 := 7 * 10
  let diff_7 := local_value_7 - face_value_7
  show diff_7 = 63, from sorry

theorem diff_local_face_value_3_in_3578_90365 :
  let face_value_3 := 3
  ∧ let local_value_3 := 3 * 1000
  ∧ let diff_3 := local_value_3 - face_value_3
  in diff_3 = 2997 :=
by
  let face_value_3 := 3
  let local_value_3 := 3 * 1000
  let diff_3 := local_value_3 - face_value_3
  show diff_3 = 2997, from sorry

end diff_local_face_value_7_in_657_90385_diff_local_face_value_3_in_3578_90365_l337_337608


namespace fair_split_adjustment_l337_337495

theorem fair_split_adjustment
    (A B : ℝ)
    (h : A < B)
    (d1 d2 d3 : ℝ)
    (h1 : d1 = 120)
    (h2 : d2 = 150)
    (h3 : d3 = 180)
    (bernardo_pays_twice : ∀ D, (2 : ℝ) * D = d1 + d2 + d3) :
    (B - A) / 2 - 75 = ((d1 + d2 + d3) - 450) / 2 - (A - (d1 + d2 + d3) / 3) :=
by
  sorry

end fair_split_adjustment_l337_337495


namespace spider_leg_pressure_l337_337640

def weight_previous_spider := 6.4 
def weight_multiplier := 2.5
def cross_sectional_area := 0.5
def number_of_legs := 8

theorem spider_leg_pressure :
  let weight_new_spider := weight_previous_spider * weight_multiplier in
  let weight_per_leg := weight_new_spider / number_of_legs in
  let pressure_per_leg := weight_per_leg / cross_sectional_area in
  pressure_per_leg = 4 :=
by
  let weight_new_spider := 6.4 * 2.5
  let weight_per_leg := weight_new_spider / 8
  let pressure_per_leg := weight_per_leg / 0.5
  show pressure_per_leg = 4
  sorry

end spider_leg_pressure_l337_337640


namespace smallest_n_for_positive_reals_l337_337359

theorem smallest_n_for_positive_reals (a b : ℝ) (n : ℕ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) 
  (h_eq : (complex.mk a b) ^ (n + 1) = (complex.conj (complex.mk a b)) ^ (n + 1)) : 
  (n = 3) ∧ (b / a = 1) := 
by 
  sorry

end smallest_n_for_positive_reals_l337_337359


namespace number_of_m_values_l337_337388

theorem number_of_m_values : 
  (∃ m_list : List ℕ, 
  ∀ m ∈ m_list, 
    let k := m^2 - 4 
    in k > 0 ∧ 2310 % k = 0 ∧ 
       k ∈ [1, 2, 3, 5, 6, 7, 10, 11, 14, 15, 21, 22, 30, 33, 35, 42, 55, 66, 70, 77, 105, 110, 154, 210, 231, 330, 385, 462, 770, 1155, 2310] ∧ 
       (∃ n, n^2 = k)) ∧
  m_list.length = 4 :=
by
  sorry

end number_of_m_values_l337_337388


namespace initial_time_is_11_55_l337_337313

-- Definitions for the conditions
variable (X : ℕ) (Y : ℕ)

def initial_time_shown_by_clock (X Y : ℕ) : Prop :=
  (5 * (18 - X) = 35) ∧ (Y = 60 - 5)

theorem initial_time_is_11_55 (h : initial_time_shown_by_clock X Y) : (X = 11) ∧ (Y = 55) :=
sorry

end initial_time_is_11_55_l337_337313


namespace no_valid_integer_pair_exists_l337_337361

theorem no_valid_integer_pair_exists :
  ¬ ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ (real.sqrt a + real.sqrt b = 10) ∧ (real.sqrt a * real.sqrt b = 18) :=
by
  sorry

end no_valid_integer_pair_exists_l337_337361


namespace angle_SPR_value_l337_337868

theorem angle_SPR_value
  (PQ_parallel_RS : PQ ∥ RS)
  (PRT_straight : angle PRT = 180)
  (angle_PSR : angle PSR = 130)
  (angle_PQR : angle PQR = 85)
  (angle_PRS : angle PRS = 120) :
  angle SPR = 85 := 
sorry

end angle_SPR_value_l337_337868


namespace find_sides_of_rectangle_l337_337647

-- Define the conditions
def isRectangle (w l : ℝ) : Prop :=
  l = 3 * w ∧ 2 * l + 2 * w = l * w

-- Main theorem statement
theorem find_sides_of_rectangle (w l : ℝ) :
  isRectangle w l → w = 8 / 3 ∧ l = 8 :=
by
  sorry

end find_sides_of_rectangle_l337_337647


namespace sum_real_solutions_sqrt_l337_337017

theorem sum_real_solutions_sqrt (x : ℝ) (h : x > 0) :
    (\sqrt{x} + \sqrt{9 / x} + \sqrt{x + 9 / x} = 8) → 
    ∑ (x : ℝ) in (λ (x : ℝ), \sqrt{x} + \sqrt{9 / x} + \sqrt{x + 9 / x} = 8), x = 3025 / 256 := 
sorry

end sum_real_solutions_sqrt_l337_337017


namespace ordered_pairs_polynomial_l337_337381

theorem ordered_pairs_polynomial :
  ∃ n : ℕ, n = ∑ a in (Finset.range 199).filter (λ a, a ≥ 2), 
  ∑ b in (Finset.range (a + 1)), 
  ∃ r s : ℤ, r + s = -a ∧ r * s = b ∧ |r - s| ≥ 2 :=
sorry

end ordered_pairs_polynomial_l337_337381


namespace drive_time_from_city_B_to_city_A_l337_337615

theorem drive_time_from_city_B_to_city_A
  (t : ℝ)
  (round_trip_distance : ℝ := 360)
  (saved_time_per_trip : ℝ := 0.5)
  (average_speed : ℝ := 80) :
  (80 * ((3 + t) - 2 * 0.5)) = 360 → t = 2.5 :=
by
  intro h
  sorry

end drive_time_from_city_B_to_city_A_l337_337615


namespace monthly_incomes_l337_337588

theorem monthly_incomes (a b c d e : ℕ) : 
  a + b = 8100 ∧ 
  b + c = 10500 ∧ 
  a + c = 8400 ∧
  (a + b + d) / 3 = 4800 ∧
  (c + d + e) / 3 = 6000 ∧
  (b + a + e) / 3 = 4500 → 
  (a = 3000 ∧ b = 5100 ∧ c = 5400 ∧ d = 6300 ∧ e = 5400) :=
by sorry

end monthly_incomes_l337_337588


namespace max_isosceles_right_triangle_volume_l337_337993

-- Constants and assumptions
def equilateral_triangle (a : ℝ) := sorry
def isosceles_right_triangle (a : ℝ) := sorry
def tetrahedron (faces : List ℝ) := sorry

-- Main proof problem statement
theorem max_isosceles_right_triangle_volume :
  ∃ (V : ℝ), V = 0.262 ∧
  ∀ (T : tetrahedron) (units : ℝ), 
  (∀ face, face ∈ T.faces → face = equilateral_triangle units ∨ face = isosceles_right_triangle units) →
  (∀ (f1 f2 : ℝ), f1 ∈ T.faces → f2 ∈ T.faces → f1 = isosceles_right_triangle units → f2 = isosceles_right_triangle units → f1 = f2) →
  volume T = V :=
sorry

end max_isosceles_right_triangle_volume_l337_337993


namespace birds_in_marsh_end_of_day_l337_337130

def geese_initial : Nat := 58
def ducks : Nat := 37
def geese_flew_away : Nat := 15
def swans : Nat := 22
def herons : Nat := 2

theorem birds_in_marsh_end_of_day : 
  58 - 15 + 37 + 22 + 2 = 104 := by
  sorry

end birds_in_marsh_end_of_day_l337_337130


namespace angle_between_vectors_eq_pi_over_3_l337_337443

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

-- Condition 1: a is perpendicular to a - 2b
axiom a_perp_a_minus_2b : ⟪a, a - 2 • b⟫ = 0

-- Condition 2: (a + b) is perpendicular to (a - b)
axiom a_plus_b_perp_a_minus_b : ⟪a + b, a - b⟫ = 0

-- Goal: The angle between vectors a and b is π/3
theorem angle_between_vectors_eq_pi_over_3 :
  real.angle (inner_product_geometry.direction a) (inner_product_geometry.direction b) = real.pi / 3 :=
sorry

end angle_between_vectors_eq_pi_over_3_l337_337443


namespace number_of_valid_pairs_l337_337780

theorem number_of_valid_pairs : 
  {p : ℕ × ℕ // p.1 > 0 ∧ p.2 > 0 ∧ p.1 ≥ p.2 ∧ p.1^2 - p.2^2 = 72}.card = 3 := 
by
  sorry

end number_of_valid_pairs_l337_337780


namespace tree_height_fraction_l337_337278

theorem tree_height_fraction :
  ∀ (initial_height growth_per_year : ℝ),
  initial_height = 4 ∧ growth_per_year = 0.5 →
  ((initial_height + 6 * growth_per_year) - (initial_height + 4 * growth_per_year)) / (initial_height + 4 * growth_per_year) = 1 / 6 :=
by
  intros initial_height growth_per_year h
  rcases h with ⟨h1, h2⟩
  sorry

end tree_height_fraction_l337_337278


namespace domain_of_f_f_is_monotonically_increasing_on_domain_l337_337429

def f (x : ℝ) : ℝ := real.sqrt (x - 1)

theorem domain_of_f :
  set_of (λ x : ℝ, 1 ≤ x) = set.Ici 1 :=
by
  ext x
  simp

theorem f_is_monotonically_increasing_on_domain :
  ∀ x1 x2 ∈ set.Ici (1 : ℝ), x1 < x2 → f x1 < f x2 :=
by
  intros x1 x2 hx1 hx2 h
  simp [f, real.sqrt_lt_iff]
  split
  . exact h
  . exact real.lt_of_sub_pos (by linarith)
  . linarith
  . exact real.sqrt_nonneg _
  . exact real.sqrt_nonneg _

end domain_of_f_f_is_monotonically_increasing_on_domain_l337_337429


namespace volunteers_distribution_l337_337002

theorem volunteers_distribution : 
  ∃ (arrangements : ℕ), arrangements = 240 
  ∧ ∀ (volunteers : ℕ) (exits : ℕ), volunteers = 5 ∧ exits = 4 
  → arrangements = 240 :=
by {
  exist 240,
  intros volunteers exits,
  intro h,
  apply and.intro,
  {
    exact 240,
  },
  {
    rintro ⟨hv, he⟩,
    cases hv,
    cases he,
    refl,
  },
}

end volunteers_distribution_l337_337002


namespace remaining_time_to_sleep_l337_337365

-- Conditions
def flight_time_la_to_hawaii : ℕ := 360 -- in minutes

def time_watching_documentary : ℕ := 90 -- minutes
def time_eating_meal : ℕ := 40 -- minutes
def time_playing_video_game : ℕ := 120 -- minutes

-- Problem Statement
theorem remaining_time_to_sleep :
  let total_time_spent := time_watching_documentary + time_eating_meal + time_playing_video_game in
  flight_time_la_to_hawaii - total_time_spent = 110 :=
by
  sorry

end remaining_time_to_sleep_l337_337365


namespace parabola_vertex_y_coord_l337_337967

theorem parabola_vertex_y_coord : 
  ∀ (x : ℝ), let y := -3 * x^2 - 30 * x - 81 in ∃ n : ℝ, n = -6 :=
by
  intros
  use -6
  sorry

end parabola_vertex_y_coord_l337_337967


namespace share_of_A_in_profit_l337_337302

def initial_investment_A : ℝ := 3000
def initial_investment_B : ℝ := 4000
def profit_at_end : ℝ := 630
def duration_first_part : ℝ := 8 / 12  -- 8 months of the year
def duration_second_part : ℝ := 4 / 12 -- 4 months of the year
def withdrawal_A : ℝ := 1000
def advancement_B : ℝ := 1000

theorem share_of_A_in_profit :
  let inv_A_part1 := initial_investment_A * duration_first_part
  let inv_A_part2 := (initial_investment_A - withdrawal_A) * duration_second_part
  let total_inv_A := inv_A_part1 + inv_A_part2

  let inv_B_part1 := initial_investment_B * duration_first_part
  let inv_B_part2 := (initial_investment_B + advancement_B) * duration_second_part
  let total_inv_B := inv_B_part1 + inv_B_part2

  let ratio_A_B := total_inv_A / total_inv_B

  (ratio_A_B * profit_at_end) = 240 :=
by
  let inv_A_part1 := initial_investment_A * duration_first_part
  let inv_A_part2 := (initial_investment_A - withdrawal_A) * duration_second_part
  let total_inv_A := inv_A_part1 + inv_A_part2

  let inv_B_part1 := initial_investment_B * duration_first_part
  let inv_B_part2 := (initial_investment_B + advancement_B) * duration_second_part
  let total_inv_B := inv_B_part1 + inv_B_part2

  let ratio_A_B := total_inv_A / total_inv_B
  
  have h : (8/21) * profit_at_end = 240 := sorry
  
  exact h

end share_of_A_in_profit_l337_337302


namespace pascal_triangle_contains_53_l337_337808

theorem pascal_triangle_contains_53:
  ∃! n, ∃ k, (n ≥ 0) ∧ (k ≥ 0) ∧ (binom n k = 53) := 
sorry

end pascal_triangle_contains_53_l337_337808


namespace mark_trees_final_count_l337_337906

-- Condition: Mark has 120 trees currently
def initial_trees : ℕ := 120

-- Condition: Mark plans to plant additional 5.5% of his current number of trees
def additional_percentage : ℝ := 5.5 / 100

-- Calculate the additional number of trees (ignoring the fractional part as trees must be whole)
def additional_trees : ℕ := ⌊(additional_percentage * initial_trees : ℝ)⌋₊

-- Theorem stating that the final number of trees will be 126
theorem mark_trees_final_count : 
  (initial_trees + additional_trees) = 126 :=
by 
  -- Declare approximation and use it for the proof. Note that this part includes conversion steps which are beyond Lean code requirements here
  let fractional_part := additional_percentage * initial_trees
  have h1 : fractional_part = 6.6 := by sorry
  have h2 : additional_trees = 6 := by sorry
  exact sorry

end mark_trees_final_count_l337_337906


namespace cost_of_fencing_l337_337712

theorem cost_of_fencing
  (d : ℝ) (rate : ℝ) (π : ℝ)
  (diameter_eq : d = 18.5)
  (rate_eq : rate = 2.75)
  (pi_approx : π ≈ 3.14159):
  let C := π * d in
  let cost := C * rate in
  cost ≈ 159.80 :=
by
  sorry

end cost_of_fencing_l337_337712


namespace find_length_of_first_train_l337_337979

def speed_of_first_train := 60 -- in kmph
def speed_of_second_train := 40 -- in kmph
def time_to_cross := 10.799136069114471 -- in seconds
def length_of_second_train := 160 -- in meters

noncomputable def length_of_first_train : ℝ :=
  let relative_speed := (speed_of_first_train + speed_of_second_train : ℝ) * 1000 / 3600 in
  let total_distance := relative_speed * time_to_cross in
  total_distance - length_of_second_train

theorem find_length_of_first_train :
  length_of_first_train = 140 :=
by
  sorry

end find_length_of_first_train_l337_337979


namespace traci_flour_l337_337593

variable (HarrisFlour : ℕ) (cakeFlour : ℕ) (cakesEach : ℕ)

theorem traci_flour (HarrisFlour := 400) (cakeFlour := 100) (cakesEach := 9) :
  ∃ (TraciFlour : ℕ), 
  (cakesEach * 2 * cakeFlour) - HarrisFlour = TraciFlour ∧ 
  TraciFlour = 1400 :=
by
  have totalCakes : ℕ := cakesEach * 2
  have totalFlourNeeded : ℕ := totalCakes * cakeFlour
  have TraciFlour := totalFlourNeeded - HarrisFlour
  exact ⟨TraciFlour, rfl, rfl⟩

end traci_flour_l337_337593


namespace find_a_l337_337839

theorem find_a (a : ℝ) (x : ℝ) :
  (∃ b : ℝ, (9 * x^2 - 18 * x + a) = (3 * x + b) ^ 2) → a = 9 := by
  sorry

end find_a_l337_337839


namespace triangle_proof_l337_337477

noncomputable theory

open Real

namespace TriangleProof

variables {a b c C : ℝ}

-- Conditions for the problem
axiom condition1 : a^2 - c^2 + b^2 = ab

-- Question 1: Proving the Measure of Angle C
def measure_of_angle_C (a b c : ℝ) (h : a^2 - c^2 + b^2 = ab) : Prop :=
  C = real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))

-- Question 2: Proving the area of the triangle when a = b = 3 and angle C = π/3
def area_of_triangle (a b : ℝ) (C : ℝ) (h : a = b) (hC : C = π / 3) : Prop :=
  abs ((1 / 2) * a * b * sin C) = (9 * sqrt 3) / 4

-- Theorem combining the conditions and questions
theorem triangle_proof :
  measure_of_angle_C a b c condition1 ∧ area_of_triangle 3 3 (π / 3) rfl rfl :=
sorry

end TriangleProof

end triangle_proof_l337_337477


namespace sum_tens_ones_digits_of_6_pow_15_l337_337610

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
def ones_digit (n : ℕ) : ℕ := n % 10

theorem sum_tens_ones_digits_of_6_pow_15 :
  tens_digit (6^15) + ones_digit (6^15) = 13 :=
by
  -- we simplify 6^15 mod 100
  have h : (6^15 : Zmod 100) = 76 := sorry
  
  -- tens digit of 76 is 7
  have tens : tens_digit (6^15) = 7 := by
    rw h
    exact rfl
  
  -- ones digit of 76 is 6
  have ones : ones_digit (6^15) = 6 := by
    rw h
    exact rfl
  
  -- sum of tens and ones digits is 13
  rw [tens, ones]
  exact rfl

sorry

end sum_tens_ones_digits_of_6_pow_15_l337_337610


namespace number_of_terms_with_rational_coefficients_l337_337987

noncomputable def number_of_rational_terms : ℕ :=
  let n := 500 in
  let k_values := {k | k % 4 = 0 ∧ (n - k) % 2 = 0} in
  k_values.card

theorem number_of_terms_with_rational_coefficients :
  number_of_rational_terms = 126 :=
by
  sorry

end number_of_terms_with_rational_coefficients_l337_337987


namespace num_points_on_ellipse_l337_337756

theorem num_points_on_ellipse (x y : ℝ) (h1 : x^2 / 4 + y^2 = 1)
  (h2 : ∃ P : ℝ × ℝ, P = (x, y) ∧ area_triangle_with_foci_is_sqrt_3 P F1 F2 = √3) :
  ∃ n : ℕ, n = 2 :=
sorry

def area_triangle_with_foci_is_sqrt_3 (P F1 F2 : ℝ × ℝ) : ℝ := 
sorry

def F1 : ℝ × ℝ := (−√3, 0)
def F2 : ℝ × ℝ := (√3, 0)

end num_points_on_ellipse_l337_337756


namespace max_distance_from_circle_center_to_line_l337_337036

theorem max_distance_from_circle_center_to_line :
  ∀ (center : ℝ × ℝ) (line : ℝ × ℝ × ℝ), 
  (dist center (2, 3) = 1) → (line = (3, -4, -4)) →
  let d := (|3 * 2 - 4 * 3 - 4| / real.sqrt (3^2 + 4^2)) in
  (d + 1 = 3) :=
begin
  intros center line h_radius h_line,
  let d := (|3 * 2 - 4 * 3 - 4| / real.sqrt (3^2 + 4^2)),
  have hd : d = 2,
  {
    -- Detailed proof of d calculation (not necessary here, just assuming it's calculated properly)
    sorry
  },
  have result : d + 1 = 3,
  {
    rw hd,
    norm_num,
  },
  exact result,
end

end max_distance_from_circle_center_to_line_l337_337036


namespace total_water_heaters_l337_337269

-- Define the conditions
variables (W C : ℕ) -- W: capacity of Wallace's water heater, C: capacity of Catherine's water heater
variable (wallace_3over4_full : W = 40 ∧ W * 3 / 4 ∧ C = W / 2 ∧ C * 3 / 4)

-- The proof problem
theorem total_water_heaters (wallace_3over4_full : W = 40 ∧ (W * 3 / 4 = 30) ∧ C = W / 2 ∧ (C * 3 / 4 = 15)) : W * 3 / 4 + C * 3 / 4 = 45 :=
sorry

end total_water_heaters_l337_337269


namespace traci_flour_brought_l337_337596

-- Definitions based on the conditions
def harris_flour : ℕ := 400
def flour_per_cake : ℕ := 100
def cakes_each : ℕ := 9

-- Proving the amount of flour Traci brought
theorem traci_flour_brought :
  (cakes_each * flour_per_cake) - harris_flour = 500 :=
by
  sorry

end traci_flour_brought_l337_337596


namespace triangle_ratio_l337_337051

variables (A B C : ℝ) (a b c : ℝ)

theorem triangle_ratio (h_cosB : Real.cos B = 4/5)
    (h_a : a = 5)
    (h_area : 1/2 * a * c * Real.sin B = 12) :
    (a + c) / (Real.sin A + Real.sin C) = 25 / 3 :=
sorry

end triangle_ratio_l337_337051


namespace robin_total_bottles_l337_337551

theorem robin_total_bottles (m a e n : ℕ) 
  (h_m : m = 7)
  (h_a : a = 9)
  (h_e : e = 5)
  (h_n : n = 3) : 
  m + a + e + n = 24 := 
by 
  rw [h_m, h_a, h_e, h_n]; 
  exact Nat.add_assoc (7 + 9) 5 3 ▸ Nat.add_comm 5 3 ▸ Nat.add_assoc 7 9 (5 + 3);
  sorry

end robin_total_bottles_l337_337551


namespace volume_ratio_l337_337239

-- Variable definitions
variable (r : ℝ) -- radius of the smaller sphere
variable (R : ℝ) -- radius of the larger sphere
variable (volume : ℝ → ℝ) -- volume function

-- Constraints and definitions
def R_is_4r (r R : ℝ) : Prop := R = 4 * r
def volume_sphere (r : ℝ) := (4 / 3) * Real.pi * r^3

-- Prove the volume ratio
theorem volume_ratio (r R : ℝ) (h : R_is_4r r R) : volume_sphere R = 64 * volume_sphere r :=
by
  -- Note: the proof is omitted; sorry is used here
  sorry

end volume_ratio_l337_337239


namespace product_zero_of_special_set_l337_337403

theorem product_zero_of_special_set (a : Fin 1997 → ℝ) (h : ∀ i, a i = ∑ j in Finset.univ.erase i, a j) :
  (∑ i, a i = 0) → (∏ i, a i = 0) :=
begin
  sorry
end

end product_zero_of_special_set_l337_337403


namespace range_of_m_l337_337522

theorem range_of_m (α β m : ℝ) (hαβ : 0 < α ∧ α < 1 ∧ 1 < β ∧ β < 2)
  (h_eq : ∀ x, x^2 - 2*(m-1)*x + (m-1) = 0 ↔ (x = α ∨ x = β)) :
  2 < m ∧ m < 7 / 3 := by
  sorry

end range_of_m_l337_337522


namespace pascal_contains_53_l337_337791

theorem pascal_contains_53 (n : ℕ) (h1 : Nat.Prime 53) (h2 : ∃ k, 1 ≤ k ∧ k ≤ 52 ∧ nat.choose 53 k = 53) (h3 : ∀ m < 53, ¬ (∃ k, 1 ≤ k ∧ k ≤ m - 1 ∧ nat.choose m k = 53)) (h4 : ∀ m > 53, ¬ (∃ k, 1 ≤ k ∧ k ≤ m - 1 ∧ nat.choose m k = 53)) : 
  (n = 53) → (n = 1) := 
by
  intros
  sorry

end pascal_contains_53_l337_337791


namespace converse_log_decreasing_l337_337939

theorem converse_log_decreasing (a : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) :
  log a 2 ≥ 0 → ¬ (∀ x y : ℝ, x > 0 → y > 0 → x < y → log a x > log a y) :=
by
  sorry

end converse_log_decreasing_l337_337939


namespace number_of_rows_containing_53_l337_337815

theorem number_of_rows_containing_53 (h_prime_53 : Nat.Prime 53) : 
  ∃! n, (n = 53 ∧ ∃ k, k ≥ 0 ∧ k ≤ n ∧ Nat.choose n k = 53) :=
by 
  sorry

end number_of_rows_containing_53_l337_337815


namespace intersect_line_circle_MA_MB_l337_337131

-- Definitions of points and equations
def M : ℝ × ℝ := (1, 2)
def theta : ℝ := π / 3
def slope := Math.tan theta

-- Parametric equations of line l
def line_l (t : ℝ) : ℝ × ℝ := (1 + t / 2, 2 + (sqrt 3) / 2 * t)

-- Circle C defined in polar coordinates
def polar_to_cartesian (rho θ : ℝ) : ℝ × ℝ := (rho * Math.cos θ, rho * Math.sin θ)
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

theorem intersect_line_circle_MA_MB :
  ∃ (t1 t2 : ℝ), 
    let A := line_l t1,
    let B := line_l t2,
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧ 1 = abs ((t1) * (t2)) :=
sorry

end intersect_line_circle_MA_MB_l337_337131


namespace sum_leq_2k_l337_337517

theorem sum_leq_2k {k : ℕ} (h_k : k ≥ 2) (a : ℕ → ℕ) 
  (h_pos : ∀ i, 1 ≤ i → i ≤ k → a i ≥ 1)
  (h_sorted : ∀ i j, 1 ≤ i → i ≤ j → j ≤ k → a i ≤ a j) 
  (h_sum_eq_prod : ∑ i in finset.range k, a (i+1) = ∏ i in finset.range k, a (i+1)) :
  ∑ i in finset.range k, a (i+1) ≤ 2 * k :=
by
  sorry

end sum_leq_2k_l337_337517


namespace pascal_triangle_contains_53_l337_337807

theorem pascal_triangle_contains_53:
  ∃! n, ∃ k, (n ≥ 0) ∧ (k ≥ 0) ∧ (binom n k = 53) := 
sorry

end pascal_triangle_contains_53_l337_337807


namespace probability_of_selecting_copresidents_vp_in_clubs_l337_337255

theorem probability_of_selecting_copresidents_vp_in_clubs :
  let total_prob := (1/3) * ((3 / 15) + (5 / 70) + (6 / 126)) in
  total_prob = (67 / 630) :=
by
  sorry

end probability_of_selecting_copresidents_vp_in_clubs_l337_337255


namespace shortest_distance_to_line_l337_337717

noncomputable def curve (x : ℝ) : ℝ := x^2 - Real.log x
def line (x y : ℝ) : Prop := x - y - 2 = 0

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem shortest_distance_to_line :
  ∃ (x : ℝ), line x (curve x) →
  (∀ (x1 y1 : ℝ), distance x (curve x) x1 y1 ≥ sqrt 2) := by
  sorry

end shortest_distance_to_line_l337_337717


namespace largest_root_of_polynomial_correct_l337_337895

noncomputable def largest_root_of_polynomial (P : Polynomial ℤ) : Prop :=
  degree P = 2008 ∧ leading_coeff P = 1 ∧
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 2007 → P.eval ↑k = 0) ∧
  P.eval 0 = (2009 : ℤ).fact → largest_root 2009 2008 4034072

theorem largest_root_of_polynomial_correct (P : Polynomial ℤ) :
  largest_root_of_polynomial P :=
  sorry

end largest_root_of_polynomial_correct_l337_337895


namespace ellipse_problem_example_l337_337405

noncomputable def ellipse_equation (m n : ℝ) (h : m > 0 ∧ n > 0) 
  (parabola_focus : ℝ × ℝ) (h_parabola_focus : parabola_focus = (2, 0)) 
  (e : ℝ) (h_e : e = 1 / 2) : Prop :=
  ∃ (m n : ℝ), (m = 4 ∧ n = 2√3) ∧ (h ∧ h_parabola_focus ∧ h_e) ∧ 
  (∀ x y : ℝ, (x^2 / m^2 + y^2 / n^2 = x^2 / 16 + y^2 / 12) := 1)

theorem ellipse_problem_example {m n : ℝ} (h : m > 0 ∧ n > 0) (parabola_focus : ℝ × ℝ)
  (h_parabola_focus : parabola_focus = (2, 0)) (e : ℝ) (h_e : e = 1 / 2) :
  ellipse_equation m n h parabola_focus h_parabola_focus e h_e :=
sorry

end ellipse_problem_example_l337_337405


namespace pq_false_l337_337068

-- Definitions of propositions p and q
def p (x : ℝ) : Prop := x > 3 ↔ x^2 > 9
def q (a b : ℝ) : Prop := a^2 > b^2 ↔ a > b

-- Theorem to prove that p ∨ q is false given the conditions
theorem pq_false (x a b : ℝ) (hp : ¬ p x) (hq : ¬ q a b) : ¬ (p x ∨ q a b) :=
by
  sorry

end pq_false_l337_337068


namespace polynomial_solution_l337_337703

noncomputable def q (x : ℝ) := 4 * sqrt 3 * x^4

theorem polynomial_solution (q : ℝ → ℝ) (h : ∀ x : ℝ, q (x^4) - q (x^4 - 3) = q x ^ 3 + 18) : 
  ∀ x : ℝ, q x = 4 * sqrt 3 * x^4 :=
begin
  sorry
end

end polynomial_solution_l337_337703


namespace arithmetic_seq_a12_l337_337866

variable {a : ℕ → ℝ}
variable {d : ℝ}

noncomputable def a_seq (n : ℕ) : ℝ := a 1 + (n - 1) * d

theorem arithmetic_seq_a12 :
  (a_seq 3 + a_seq 4 + a_seq 5 = 3) →
  (a_seq 8 = 8) →
  (a_seq 12 = 15) :=
by
  sorry

end arithmetic_seq_a12_l337_337866


namespace sonika_years_in_bank_l337_337212

variable (P A1 A2 : ℚ)
variables (r t : ℚ)

def simple_interest (P r t : ℚ) : ℚ := P * r * t / 100
def amount_with_interest (P r t : ℚ) : ℚ := P + simple_interest P r t

theorem sonika_years_in_bank :
  P = 9000 → A1 = 10200 → A2 = 10740 →
  amount_with_interest P r t = A1 →
  amount_with_interest P (r + 2) t = A2 →
  t = 3 :=
by
  intros hP hA1 hA2 hA1_eq hA2_eq
  sorry

end sonika_years_in_bank_l337_337212


namespace find_z_l337_337525

theorem find_z (x y z w : ℚ) (h1 : x > y) (h2 : y > z) 
    (h3 : x - 1 = y) (h4 : y - 1 = z) 
    (h5 : w = 5 * x / 3) (h6 : w^2 = x * z) 
    (h7 : 2 * x + 3 * y + 3 * z = 5 * y + 11) : 
    z = 3 :=
by
  -- proof will be inserted here
  sorry

end find_z_l337_337525


namespace calculate_expression_l337_337685

theorem calculate_expression : (-1:ℝ)^2 + (1/3:ℝ)^0 = 2 := by
  sorry

end calculate_expression_l337_337685


namespace new_arithmetic_mean_l337_337935

theorem new_arithmetic_mean (mean_original : ℝ) (num_elements : ℕ) (discarded : ℕ → ℝ) :
  mean_original = 42 →
  num_elements = 60 →
  discarded 0 = 48 →
  discarded 1 = 52 →
  discarded 2 = 56 →
  (sum_discarded : ℝ) (H_sum_discarded : sum_discarded = discarded 0 + discarded 1 + discarded 2) →
  mean_new : ℝ → mean_new = 41.47 :=
by
  -- sum of original set
  let sum_original := mean_original * (num_elements : ℝ)
  -- sum of new set
  let sum_new := sum_original - sum_discarded
  -- number of new elements
  let num_new := num_elements - 3
  -- calculate mean of new set
  let mean_new := sum_new / (num_new : ℝ)
  sorry

end new_arithmetic_mean_l337_337935


namespace digit_sum_decrease_l337_337642

theorem digit_sum_decrease (n : ℕ) (h1 : digit_sum n = 2013) (h2 : digit_sum (n + 1) < 2013) (h3 : ¬ (n + 1) % 4 = 0) :
  digit_sum (n + 1) = 2005 :=
  sorry

end digit_sum_decrease_l337_337642


namespace original_square_area_l337_337414

-- Definitions based on the given problem conditions
variable (s : ℝ) (A : ℝ)
def is_square (s : ℝ) : Prop := s > 0
def oblique_projection (s : ℝ) (A : ℝ) : Prop :=
  (A = s^2 ∨ A = 4^2) ∧ s = 4

-- The theorem statement based on the problem question and correct answer
theorem original_square_area :
  is_square s →
  oblique_projection s A →
  ∃ A, A = 16 ∨ A = 64 := 
sorry

end original_square_area_l337_337414


namespace range_of_x_l337_337499

noncomputable def f (a x : ℝ) : ℝ := real.log a (a^(2 * x) - 2 * a^x - 2)

theorem range_of_x (a : ℝ) (x : ℝ) (h : 0 < a ∧ a < 1) : f a x < 0 ↔ x < real.log a 3 :=
by
  sorry

end range_of_x_l337_337499


namespace solve_for_y_l337_337929

theorem solve_for_y (y : ℚ) : 27^(3*y - 5) = 9^(y + 4) → y = 23 / 7 := 
by 
  sorry

end solve_for_y_l337_337929


namespace problem_I_problem_II_problem_III_l337_337065

-- Problem (I)
noncomputable def f (x a : ℝ) := Real.log x - a * (x - 1)
noncomputable def tangent_line (x a : ℝ) := (1 - a) * (x - 1)

theorem problem_I (a : ℝ) :
  ∃ y, tangent_line y a = f 1 a / (1 : ℝ) :=
sorry

-- Problem (II)
theorem problem_II (a : ℝ) (h : a ≥ 1 / 2) :
  ∀ x ≥ 1, f x a ≤ Real.log x / (x + 1) :=
sorry

-- Problem (III)
theorem problem_III (a : ℝ) :
  ∀ x ≥ 1, Real.exp (x - 1) - a * (x ^ 2 - x) ≥ x * f x a + 1 :=
sorry

end problem_I_problem_II_problem_III_l337_337065


namespace car_speed_l337_337617

/-- Given a car covers a distance of 624 km in 2 3/5 hours,
    prove that the speed of the car is 240 km/h. -/
theorem car_speed (distance : ℝ) (time : ℝ)
  (h_distance : distance = 624)
  (h_time : time = 13 / 5) :
  (distance / time) = 240 :=
by
  sorry

end car_speed_l337_337617


namespace xiaohongs_mother_deposit_l337_337614

theorem xiaohongs_mother_deposit (x : ℝ) :
  x + x * 3.69 / 100 * 3 * (1 - 20 / 100) = 5442.8 :=
by
  sorry

end xiaohongs_mother_deposit_l337_337614


namespace ice_cream_bar_price_l337_337310

theorem ice_cream_bar_price 
  (num_bars num_sundaes : ℕ)
  (total_cost : ℝ)
  (sundae_price ice_cream_bar_price : ℝ)
  (h1 : num_bars = 125)
  (h2 : num_sundaes = 125)
  (h3 : total_cost = 250.00)
  (h4 : sundae_price = 1.40)
  (total_price_condition : num_bars * ice_cream_bar_price + num_sundaes * sundae_price = total_cost) :
  ice_cream_bar_price = 0.60 :=
sorry

end ice_cream_bar_price_l337_337310


namespace find_f_f_f_3_l337_337900

def f(x : ℝ) : ℝ :=
if x > 5 then real.sqrt (x + 1) else x^2 + 1

theorem find_f_f_f_3 : f(f(f(3))) = real.sqrt (real.sqrt 11 + 1) := by
  sorry

end find_f_f_f_3_l337_337900


namespace cosine_sum_formula_example_l337_337424

noncomputable def cos_theta_plus_pi_over_4 (x y : ℝ) (h : x^2 + y^2 ≠ 0) : ℝ :=
let r := real.sqrt (x^2 + y^2) in
let cos_theta := x / r in
let sin_theta := y / r in
cos_theta * real.sqrt(2) / 2 - sin_theta * real.sqrt(2) / 2

theorem cosine_sum_formula_example (h : 3^2 + (-4)^2 ≠ 0) :
  cos_theta_plus_pi_over_4 3 (-4) h = (7 * real.sqrt 2) / 10 := by
  sorry

end cosine_sum_formula_example_l337_337424


namespace tetrahedron_orthocentric_iff_midpoints_equal_l337_337199

def tetrahedron := Type -- represent a tetrahedron as a type

structure OrthocentricTetrahedron (T : tetrahedron) : Prop :=
  (altitudes_intersect : ∃ P : T, P is the intersection point of the altitudes)

structure EdgeMidpoints (T : tetrahedron) : Prop :=
  (midpoints_equal : ∀ e1 e2 e3 e4, midpoints(e1) = midpoints(e2) → midpoints(e3) = midpoints(e4))

theorem tetrahedron_orthocentric_iff_midpoints_equal (T : tetrahedron) :
  OrthocentricTetrahedron T ↔ EdgeMidpoints T :=
sorry

end tetrahedron_orthocentric_iff_midpoints_equal_l337_337199


namespace number_of_digits_in_product_l337_337346

open Nat

noncomputable def num_digits (n : ℕ) : ℕ :=
if n = 0 then 1 else Nat.log 10 n + 1

def compute_product : ℕ := 234567 * 123^3

theorem number_of_digits_in_product : num_digits compute_product = 13 := by 
  sorry

end number_of_digits_in_product_l337_337346


namespace parallel_projection_two_lines_l337_337574

variables {P : Type} [Plane P] (l1 l2 : Line P) (proj_plane : Plane P)

-- Definitions of the conditions
def intersect (l1 l2 : Line P) : Prop := ∃ (p : P), p ∈ l1 ∧ p ∈ l2

def is_perpendicular (plane1 plane2 : Plane P) : Prop := sorry
-- assuming we have some definition for perpendicular planes

-- Theorem statement
theorem parallel_projection_two_lines (h1 : intersect l1 l2) (h2 : is_perpendicular (plane l1 ∪ plane l2) proj_plane ∨ ¬ is_perpendicular (plane l1 ∪ plane l2) proj_plane) :
  parallel_projection l1 proj_plane ∪ parallel_projection l2 proj_plane = l1 ∪ l2
:= sorry

end parallel_projection_two_lines_l337_337574


namespace total_water_in_heaters_l337_337271

theorem total_water_in_heaters (wallace_capacity : ℕ) (catherine_capacity : ℕ) 
(wallace_water : ℕ) (catherine_water : ℕ) :
  wallace_capacity = 40 →
  (wallace_water = (3 * wallace_capacity) / 4) →
  wallace_capacity = 2 * catherine_capacity →
  (catherine_water = (3 * catherine_capacity) / 4) →
  wallace_water + catherine_water = 45 :=
by
  sorry

end total_water_in_heaters_l337_337271


namespace circle_diameter_l337_337023

-- The problem statement in Lean 4

theorem circle_diameter
  (d α β : ℝ) :
  ∃ r: ℝ,
  r * 2 = d * (Real.sin α) * (Real.sin β) / (Real.cos ((α + β) / 2) * (Real.sin ((α - β) / 2))) :=
sorry

end circle_diameter_l337_337023


namespace sum_real_solutions_sqrt_l337_337018

theorem sum_real_solutions_sqrt (x : ℝ) (h : x > 0) :
    (\sqrt{x} + \sqrt{9 / x} + \sqrt{x + 9 / x} = 8) → 
    ∑ (x : ℝ) in (λ (x : ℝ), \sqrt{x} + \sqrt{9 / x} + \sqrt{x + 9 / x} = 8), x = 3025 / 256 := 
sorry

end sum_real_solutions_sqrt_l337_337018


namespace planar_graph_edge_bound_l337_337196

structure Graph :=
  (V E : ℕ) -- vertices and edges

def planar_connected (G : Graph) : Prop := 
  sorry -- Planarity and connectivity conditions are complex to formalize

def num_faces (G : Graph) : ℕ :=
  sorry -- Number of faces based on V, E and planarity

theorem planar_graph_edge_bound (G : Graph) (h_planar : planar_connected G) 
  (euler : G.V - G.E + num_faces G = 2) 
  (face_bound : 2 * G.E ≥ 3 * num_faces G) : 
  G.E ≤ 3 * G.V - 6 :=
sorry

end planar_graph_edge_bound_l337_337196


namespace count_of_numbers_with_two_distinct_primes_l337_337450

def is_prime (n : ℕ) : Prop := nat.prime n

def is_product_of_two_distinct_primes (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p ≠ q ∧ p * q = n

noncomputable def count_numbers_less_than_1000000_with_two_distinct_primes : ℕ :=
  finset.card (finset.filter is_product_of_two_distinct_primes (finset.range 1000000))

theorem count_of_numbers_with_two_distinct_primes :
  count_numbers_less_than_1000000_with_two_distinct_primes = 209867 :=
sorry

end count_of_numbers_with_two_distinct_primes_l337_337450


namespace m2_sequence_proof_min_S_when_m3_proof_min_n_when_m1000_proof_l337_337204

-- Given conditions for the sequence A_n
structure SeqProps (n : ℕ) (a : Fin n → ℕ) (m : ℕ) :=
(a1 : a 0 = 1)
(an : a ⟨n-1, by simp [gt, Nat.ne_of_gt, ge, Nat.lt_trans, Nat.zero_lt_succ]⟩ = m)
(diff_cond : ∀ k : Fin (n-1), a ⟨k.1 + 1, by simp [lt_iff_le_and_ne, zero_le, Nat.lt_trans]; exact k.2⟩ - a k ∈ {0, 1})
(sum_distinct : ∀ i j s t : Fin n, i ≠ j → i ≠ s → i ≠ t → j ≠ s → j ≠ t → s ≠ t → a i + a j = a s + a t)

-- When m = 2, check if sequences meet the condition
def m2_sequence_check : Prop :=
  ∀ (seq : Fin 6 → ℕ), 
  (seq = ⟨[1,1,1,2,2,2], by simp [Nat.zero_lt_succ, zero_le]⟩ →
   ∃ (a : Fin 6 → ℕ), SeqProps 6 a 2 → False) ∧
  (seq = ⟨[1,1,1,1,2,2,2,2], by simp [Nat.zero_lt_succ, zero_le]⟩ →
   ∃ (a : Fin 8 → ℕ), SeqProps 8 a 2) ∧
  (seq = ⟨[1,1,1,1,1,2,2,2,2,2], by simp [Nat.zero_lt_succ, zero_le]⟩ →
   ∃ (a : Fin 10 → ℕ), SeqProps 10 a 2)

-- Prove the minimum S for m = 3
def min_S_when_m3 : Prop :=
  ∀ (n : ℕ) (a : Fin n → ℕ), SeqProps n a 3 → (n ≥ 6 → ∃ k : Fin n, ∑ i : Fin n, a i ≥ 20)

-- Prove the minimum n for m = 1000
def min_n_when_m1000 : Prop :=
  ∀ (n : ℕ) (a : Fin n → ℕ), SeqProps n a 1000 → (n ≥ 1008 → ∑ i : Fin 1008, a i )}

-- Skip the proofs for now
theorem m2_sequence_proof : m2_sequence_check :=
by sorry

theorem min_S_when_m3_proof : min_S_when_m3 :=
by sorry

theorem min_n_when_m1000_proof : min_n_when_m1000 :=
by sorry

end m2_sequence_proof_min_S_when_m3_proof_min_n_when_m1000_proof_l337_337204


namespace intersection_of_lines_l337_337074

theorem intersection_of_lines :
  let l1 := λ x : ℝ, x + 5
  let l2 := λ x : ℝ, -0.5 * x - 1
  ∃ x y : ℝ, l1 x = y ∧ l2 x = y ∧ (x = -4 ∧ y = 1) :=
by
  let x1 : ℝ := -4
  let y1 : ℝ := 1
  let h1 : x1 - y1 = -5 := by norm_num
  let h2 : x1 + 2 * y1 = -2 := by norm_num
  have : l1 x1 = y1 := by norm_num
  have : l2 x1 = y1 := by norm_num
  use x1, y1
  simp
  exact ⟨this, ⟨this, ⟨rfl, rfl⟩⟩⟩

end intersection_of_lines_l337_337074


namespace unique_n_with_conditions_l337_337699

theorem unique_n_with_conditions :
  ∃ n : ℕ, n > 0 ∧ 
    (∃ (a b : Fin n → ℤ), (∀ i, a i ≠ b i) ∧ pairwise (≠) (Array.toList a) ∧ pairwise (≠) (Array.toList b) ∧
    (∀ k, (∏ i, (a k)^2 + a i * a k + b i = 0) ∧ 
    (∏ i, b k^2 + a i * b k + b i = 0))) ↔ n = 2 := 
sorry

end unique_n_with_conditions_l337_337699


namespace a_n_upper_bound_l337_337579

-- Define the sequence a_n by given conditions
def seq_a : ℕ → ℕ 
| 0       := 1
| (n + 1) := if seq_a n = n then seq_a n + 3 else seq_a n + 2

-- Define the property to be proved
theorem a_n_upper_bound :
  ∀ n : ℕ, seq_a n < (1 + Real.sqrt 2) * n :=
by
  -- Skipping the proof part
  sorry

end a_n_upper_bound_l337_337579


namespace sqrt_0_arith_sqrt_9_cube_root_sqrt_64_l337_337248

-- Definition for square root
def square_root (x : ℕ) (y : ℕ) : Prop := y * y = x

-- Definition for arithmetic square root (non-negative)
def arith_square_root (x : ℕ) (y : ℕ) : Prop := y * y = x ∧ y ≥ 0

-- Definition for cube root
def cube_root (x : ℕ) (y : ℕ) : Prop := y * y * y = x

-- Lean statements for the proof problems
theorem sqrt_0 : ∃ y, square_root 0 y ∧ y = 0 := 
by {
  use 0,
  split,
  { unfold square_root, exact zero_mul 0 },
  { refl },
  sorry
}

theorem arith_sqrt_9 : ∃ y, arith_square_root 9 y ∧ y = 3 := 
by {
  use 3,
  split,
  { unfold arith_square_root, split,
    { norm_num },
    { norm_num, }
    },
  { refl },
  sorry
}

theorem cube_root_sqrt_64 : ∃ y, ∃ z, square_root 64 z ∧ cube_root z y ∧ y = 2 := 
by {
  use 2,
  use 8,
  split,
  { unfold square_root, norm_num },
  split,
  { unfold cube_root, norm_num },
  { refl },
  sorry
}

end sqrt_0_arith_sqrt_9_cube_root_sqrt_64_l337_337248


namespace tangent_line_equation_l337_337946

noncomputable def f (x : ℝ) : ℝ := (x^3 - 1) / x

theorem tangent_line_equation :
  let x₀ := 1
  let y₀ := f x₀
  let m := deriv f x₀
  y₀ = 0 →
  m = 3 →
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (y = 3 * x - 3) :=
by
  intros x₀ y₀ m h₀ hm x y
  sorry

end tangent_line_equation_l337_337946


namespace find_ABC_l337_337558

-- Define the conditions
variables (A B C : ℕ)

-- A, B, C are non-zero digits less than 7
def valid_digits : Prop := A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ A < 7 ∧ B < 7 ∧ C < 7

-- Conversion of the numbers
def AB_base7 := A * 7 + B
def C_base7 := C
def C0_base7 := C * 7
def BA_base7 := B * 7 + A
def CC_base7 := C * 7 + C

-- The conditions in the problem
def condition1 : Prop := AB_base7 + C_base7 = C0_base7
def condition2 : Prop := AB_base7 + BA_base7 = CC_base7

-- The proof problem
theorem find_ABC (h_valid : valid_digits A B C) 
  (h1 : condition1 A B C) 
  (h2 : condition2 A B C) : 
  100 * A + 10 * B + C = 325 :=
sorry

end find_ABC_l337_337558


namespace range_of_k_l337_337425

theorem range_of_k (k : ℝ) :
  (∃ x : ℝ, x ≠ 0 ∧ (frac (abs x) (x - 2)) = k * x) ∧
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    (\frac{|x₁|}{x₁ - 2} = k * x₁) ∧
    (\frac{|x₂|}{x₂ - 2} = k * x₂) ∧
    (\frac{|x₃|}{x₃ - 2} = k * x₃)) ↔
  (0 < k ∧ k < 1 / 2) :=
sorry

end range_of_k_l337_337425


namespace rectangle_ratio_width_length_l337_337470

variable (w : ℝ)

theorem rectangle_ratio_width_length (h1 : w + 8 + w + 8 = 24) : 
  w / 8 = 1 / 2 :=
by
  sorry

end rectangle_ratio_width_length_l337_337470


namespace definite_integral_value_l337_337360

noncomputable def integral_value : ℝ :=
  ∫ x in 1..2, (2 * x + 1 / x)

theorem definite_integral_value :
  integral_value = 3 + Real.log 2 :=
by
  sorry

end definite_integral_value_l337_337360


namespace polar_to_rectangular_l337_337877

theorem polar_to_rectangular : 
  ∀ (r θ : ℝ), r = 2 ∧ θ = 2 * Real.pi / 3 → 
  (r * Real.cos θ, r * Real.sin θ) = (-1, Real.sqrt 3) := by
  sorry

end polar_to_rectangular_l337_337877


namespace areas_equal_l337_337513

variable {A B C O D E : Type}
variable [EuclideanGeometry A B C O D E]
variable {triangle_ABC_is_acute : triangle_is_acute A B C}
variable {O_is_circumcenter : is_circumcenter O A B C}
variable {D_is_intersection : ∃ D, is_perpendicular_from_vertex A B C ∧ is_on_circumcircle D A B C ∧ D ≠ A}
variable {E_is_intersection : ∃ E, is_on_line B O ∧ is_on_circumcircle E A B C ∧ E ≠ B}

theorem areas_equal (h1 : triangle_ABC_is_acute) (h2 : O_is_circumcenter) 
                    (h3 : D_is_intersection) (h4 : E_is_intersection) :
  area_triangle A B C = area_triangle B D C := 
sorry

end areas_equal_l337_337513


namespace correct_growth_rate_equation_l337_337208

noncomputable def numberOfBikesFirstMonth : ℕ := 1000
noncomputable def additionalBikesThirdMonth : ℕ := 440
noncomputable def monthlyGrowthRate (x : ℝ) : Prop :=
  numberOfBikesFirstMonth * (1 + x)^2 = numberOfBikesFirstMonth + additionalBikesThirdMonth

theorem correct_growth_rate_equation (x : ℝ) : monthlyGrowthRate x :=
by
  sorry

end correct_growth_rate_equation_l337_337208


namespace count_R_locations_l337_337547

-- Define point P and Q such that PQ = 10
def P : (ℝ × ℝ) := (-5, 0)
def Q : (ℝ × ℝ) := (5, 0)

-- Define a predicate to check if a point is equidistant from P and Q
def is_equidistant (R : ℝ × ℝ) : Prop :=
  (Real.sqrt ((R.1 - P.1) ^ 2 + (R.2 - P.2) ^ 2)) = (Real.sqrt ((R.1 - Q.1) ^ 2 + (R.2 - Q.2) ^ 2))

-- Define a predicate for the area of triangle PQR to be 20 square units when it is a right triangle
def area_is_20 (R : ℝ × ℝ) : Prop :=
  let base := Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2) in  -- PQ's length
  let height := Real.abs R.2 in  -- Since R lies on y-axis
  (1 / 2) * base * height = 20

-- Define the final proof goal: there are exactly 2 such points R satisfying all conditions
theorem count_R_locations : 
  ∃ (R1 R2 : ℝ × ℝ),
  R1 ≠ R2 ∧ is_equidistant R1 ∧ area_is_20 R1 ∧ is_equidistant R2 ∧ area_is_20 R2 ∧ 
  (∀ R : ℝ × ℝ, is_equidistant R ∧ area_is_20 R → (R = R1 ∨ R = R2)) :=
sorry  -- Proof goes here (currently omitted for brevity)

end count_R_locations_l337_337547


namespace pet_shop_grooming_time_l337_337235

theorem pet_shop_grooming_time:
  (time_to_groom_poodle time_to_groom_terrier total_time : ℕ)
  (h1 : time_to_groom_poodle = 30)
  (h2 : time_to_groom_terrier = time_to_groom_poodle / 2)
  (h3 : total_time = 3 * time_to_groom_poodle + 8 * time_to_groom_terrier) :
  total_time = 210 :=
by
  sorry

end pet_shop_grooming_time_l337_337235


namespace solution_set_inequality_l337_337053

variables {f : ℝ → ℝ}
variables {A B : ℝ × ℝ}
variables [decidable_pred (λ x, x ∈ set.Ioo (1/e) (exp 2))]

noncomputable def is_decreasing (f : ℝ → ℝ) := ∀ x y : ℝ, x < y → f x > f y

theorem solution_set_inequality 
  (decreasing_f : is_decreasing f) 
  (A : A = (3, -1)) 
  (B : B = (0, 1)) :
  { x : ℝ | |f (1 + real.log x)| < 1 } = set.Ioo (1/real.exp 1) (real.exp 2) :=
sorry

end solution_set_inequality_l337_337053


namespace find_number_l337_337290

-- Define the problem conditions as hypotheses
variables (N : ℝ)
hypothesis h1 : ((5 / 4) * N = (4 / 5) * N + 45)

-- The goal is to prove that N = 100
theorem find_number : N = 100 :=
by 
  -- We would proceed with the solving steps here; for now, we leave it as sorry
  sorry

end find_number_l337_337290


namespace find_a_x_l337_337322

theorem find_a_x (x : ℝ) (a : ℝ) (h1 : x > 0) (h2 : -a + 2 = Real.sqrt x) (h3 : 2a - 1 = Real.sqrt x) : 
  a = -1 ∧ x = 9 :=
by
  sorry

end find_a_x_l337_337322


namespace part1_part2_part3_l337_337765

-- Statement (1)
theorem part1 (x : ℝ) : (x > 2) → log (x + 2) / log 2 < 2 * log x / log 2 :=
begin
  sorry
end

-- Statement (2)
theorem part2 (a : ℝ) : (¬ ∃ a : ℝ, ∀ x ∈ Icc (-1:ℝ) 2, abs (log (x + a) / log 2) ≤ log 3 / log 2) :=
begin
  sorry
end

-- Statement (3)
theorem part3 (a x : ℝ) : (0 < a) → (x ∈ Ioo 0 2) →
  log (x + a) / log 2 < 1 / 2 * (log (4 * x + a) / log 2) ↔ (0 < a ∧ a ≤ 1) :=
begin
  sorry
end

end part1_part2_part3_l337_337765


namespace john_receives_amount_l337_337205

noncomputable def ratio_parts (a b c d e : ℕ) : ℕ :=
  a + b + c + d + e

noncomputable def one_part_value (total parts : ℕ) : ℝ :=
  total / parts

noncomputable def john_share (total parts : ℝ) (john_ratio : ℕ) : ℝ :=
  john_ratio * parts

theorem john_receives_amount (total : ℕ) (john_ratio jose_ratio binoy_ratio maria_ratio steve_ratio : ℕ) :
  ratio_parts john_ratio jose_ratio binoy_ratio maria_ratio steve_ratio = 21 →
  (one_part_value total 21) * 3 = 1234 := by
  intros
  unfold ratio_parts at *
  unfold one_part_value
  unfold john_share
  sorry

end john_receives_amount_l337_337205


namespace ellipse_eq_proof_fixed_point_proof_l337_337005

-- Define the parameters and conditions given in the problem
def a : ℝ := 2
def b : ℝ := √3
def c : ℝ := 1
def A : ℝ × ℝ := (1, 3/2)

def ellipse_eq (x y a b : ℝ) : Prop :=
  (x^2 / a^2 + y^2 / b^2 = 1)

def foci_condition (f : ℝ × ℝ) : Prop :=
  f = (1, 0)

def point_on_ellipse (P : ℝ × ℝ) (x y a b : ℝ) : Prop :=
  ellipse_eq P.1 P.2 a b

-- Proof statement for part (I)
theorem ellipse_eq_proof :
  foci_condition (1, 0) ∧ point_on_ellipse (1, 3/2) 1 0 a b →
  ellipse_eq 1 (3/2) a b :=
sorry

-- Define the parameters and conditions for part (II)
def M (k : ℝ) : Prop := sorry
def N (k : ℝ) : Prop := sorry

def line_intersects_ellipse (k : ℝ) : Prop :=
  ∃ M N : ℝ × ℝ, M k ∧ N k ∧ ellipse_eq M.1 M.2 a b ∧ ellipse_eq N.1 N.2 a b

def perpendicular_bisector_intersects_x_axis (M N : ℝ × ℝ) : Prop :=
  let midpoint := ((M.1 + N.1) / 2, (M.2 + N.2) / 2) in
  ∃ P : ℝ × ℝ, P = (midpoint.1, 0)

def fixed_point_condition (P Q : ℝ × ℝ) (MN : ℝ) : Prop :=
  ∃ Q : ℝ × ℝ, Q = (1, 0) ∧ abs (P.1 - Q.1) / MN = 1 / 4

-- Proof statement for part (II)
theorem fixed_point_proof (k : ℝ) (MN : ℝ) :
  line_intersects_ellipse k ∧ perpendicular_bisector_intersects_x_axis (1,0) →
  fixed_point_condition (1,0) (1,0) MN :=
sorry

end ellipse_eq_proof_fixed_point_proof_l337_337005


namespace max_sum_of_distances_l337_337048

theorem max_sum_of_distances (x1 x2 y1 y2 : ℝ)
  (h1 : x1^2 + y1^2 = 1)
  (h2 : x2^2 + y2^2 = 1)
  (h3 : x1 * x2 + y1 * y2 = 1 / 2) :
  (|x1 + y1 - 1| / Real.sqrt 2 + |x2 + y2 - 1| / Real.sqrt 2) ≤ Real.sqrt 2 + Real.sqrt 3 :=
sorry

end max_sum_of_distances_l337_337048


namespace even_function_a_eq_neg_one_l337_337108

-- Definitions for the function f and the condition for it being an even function
def f (x a : ℝ) := (x - 1) * (x - a)

-- The theorem stating that if f is an even function, then a = -1
theorem even_function_a_eq_neg_one (a : ℝ) (h : ∀ x : ℝ, f x a = f (-x) a) : a = -1 :=
by
  sorry

end even_function_a_eq_neg_one_l337_337108


namespace find_a_b_max_min_values_l337_337431

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  (1/3) * x^3 + a * x^2 + b * x

noncomputable def f' (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  x^2 + 2 * a * x + b

theorem find_a_b (a b : ℝ) :
  f' (-3) a b = 0 ∧ f (-3) a b = 9 → a = 1 ∧ b = -3 :=
  by sorry

theorem max_min_values (a b : ℝ) (h₁ : a = 1) (h₂ : b = -3):
  ∀ x ∈ Set.Icc (-3 : ℝ) 3, f x a b ≥ -5 / 3 ∧ f x a b ≤ 9 :=
  by sorry

end find_a_b_max_min_values_l337_337431


namespace part1_part2_l337_337136

variables (a b c α β γ : ℝ)

-- Part (1)
theorem part1 (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c)
    (hα : 0 ≤ α) (hβ : 0 ≤ β) (hγ : 0 ≤ γ) :
  a * β * γ + b * γ * α + c * α * β ≥ a * b * c :=
sorry

-- Part (2)
theorem part2 (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c)
    (hα : 0 ≤ α) (hβ : 0 ≤ β) (hγ : 0 ≤ γ) :
  α * b * c + β * c * a + γ * a * b ≥ (real.sqrt 3) * a * b * c :=
sorry

end part1_part2_l337_337136


namespace pool_cleaning_percentage_l337_337146

theorem pool_cleaning_percentage (total_capa_ml : ℕ) (splash_per_jump_ml : ℕ) (jump_count : ℕ) :
  total_capa_ml = 2000000 →
  splash_per_jump_ml = 400 →
  jump_count = 1000 →
  let remaining_water_ml := total_capa_ml - (splash_per_jump_ml * jump_count)
  in 100 * remaining_water_ml / total_capa_ml = 80 := by
  intros h_total h_splash h_jump
  rw [h_total, h_splash, h_jump]
  let remaining_water_ml := 2_000_000 - (400 * 1000)
  have h_remaining : remaining_water_ml = 1_600_000 := by norm_num
  rw h_remaining
  norm_num
  sorry

end pool_cleaning_percentage_l337_337146


namespace coloring_ways_l337_337539

def color : Type := {c : Type // c = Red ∨ c = Yellow ∨ c = Blue}
def dominated_by (sq : Square) (c : color) : Prop :=
  ∃ a b, a ≠ b ∧ color_edge a ≠ color_edge b ∧ 
  dominates_color sq c ∧ (dominates_color sq ≠ c)

theorem coloring_ways :
  ∃ c : color, ∃ sq1 sq2 sq3,
    dominated_by sq1 c.Red ∧
    dominated_by sq2 c.Yellow ∧
    dominated_by sq3 c.Blue ∧
    ∀ sq ∈ {sq1, sq2, sq3}, 
      length (edges sq) = 10 ∧ unique_colors sq ∧ 
      segments_coloring sq 1026 := sorry

end coloring_ways_l337_337539


namespace find_beta_l337_337031

theorem find_beta (α β : ℝ) 
  (h₁ : sin α + sin (α + β) + cos (α + β) = sqrt 3)
  (h₂ : β ∈ set.Icc (π / 4) π) : 
  β = π / 4 :=
by
  sorry

end find_beta_l337_337031


namespace area_quadrilateral_ABCDE_correct_l337_337468

noncomputable def area_quadrilateral_ABCDE (AM NM AN BN BO OC CP CD EP DE : ℝ) : ℝ :=
  (0.5 * AM * NM * Real.sqrt 2) + (0.5 * BN * BO) + (0.5 * OC * CP * Real.sqrt 2) - (0.5 * DE * EP)

theorem area_quadrilateral_ABCDE_correct :
  ∀ (AM NM AN BN BO OC CP CD EP DE : ℝ),
    DE = 12 ∧ 
    AM = 36 ∧ 
    NM = 36 ∧ 
    AN = 36 * Real.sqrt 2 ∧
    BN = 36 * Real.sqrt 2 - 36 ∧
    BO = 36 ∧
    OC = 36 ∧
    CP = 36 * Real.sqrt 2 ∧
    CD = 24 ∧
    EP = 24
    → area_quadrilateral_ABCDE AM NM AN BN BO OC CP CD EP DE = 2311.2 * Real.sqrt 2 + 504 :=
by intro AM NM AN BN BO OC CP CD EP DE h;
   cases h;
   sorry

end area_quadrilateral_ABCDE_correct_l337_337468


namespace pow_sum_ge_mul_l337_337200

theorem pow_sum_ge_mul (m n : ℕ) : 2^(m + n - 2) ≥ m * n := 
sorry

end pow_sum_ge_mul_l337_337200


namespace john_ratio_amounts_l337_337484

/-- John gets $30 from his grandpa and some multiple of that amount from his grandma. 
He got $120 from the two grandparents. What is the ratio of the amount he got from 
his grandma to the amount he got from his grandpa? --/
theorem john_ratio_amounts (amount_grandpa amount_total : ℝ) (multiple : ℝ) :
  amount_grandpa = 30 → amount_total = 120 →
  amount_total = amount_grandpa + multiple * amount_grandpa →
  multiple = 3 :=
by
  intros h1 h2 h3
  sorry

end john_ratio_amounts_l337_337484


namespace max_pM_incenter_max_muABC_equilateral_l337_337154

/-- Given a triangle ABC, and a point M in its interior,
define A', B', and C' such that MA' ⊥ BC, MB' ⊥ CA, and MC' ⊥ AB. --/
variables {A B C M A' B' C' : Type*}
          [point A] [point B] [point C] [point M] [point A'] [point B'] [point C']
          (hA' : ∃ x, mkSegment M x ⊥ mkSegment B C)
          (hB' : ∃ x, mkSegment M x ⊥ mkSegment C A)
          (hC' : ∃ x, mkSegment M x ⊥ mkSegment A B)

/-- Define the function p(M) --/
def pM (MA' MB' MC' MA MB MC : ℝ) : ℝ :=
  (MA' * MB' * MC') / (MA * MB * MC)

/-- The point M that maximizes p(M) is the incenter of the triangle ABC --/
theorem max_pM_incenter {M_max : Type*} [point M_max]
  (incenter_condition : is_incenter M_max A B C):
  ∃ M, pM (MA' M) (MB' M) (MC' M) (MA M) (MB M) (MC M) = pM (MA' M_max) (MB' M_max) (MC' M_max) (MA M_max) (MB M_max) (MC M_max) :=
sorry

/-- The maximum value of p(M) is attained when the triangle ABC is equilateral --/
theorem max_muABC_equilateral {max_M : ℝ}
  (equilateral_condition : is_equilateral A B C):
  ∃ M, pM (MA' M) (MB' M) (MC' M) (MA M) (MB M) (MC M) = max_M :=
sorry

end max_pM_incenter_max_muABC_equilateral_l337_337154


namespace drone_highest_effective_average_speed_l337_337665

def distance_and_wind : List (ℝ × ℝ) :=
[(50, 5), (70, 3), (80, 2), (60, -1), (55, -2), (65, 1), (75, 4)]

def effective_speed (d w : ℝ) : ℝ := d + w

def highest_effective_average_speed_interval (speeds : List ℝ) : ℕ :=
speeds.index_of (speeds.foldr max (speeds.head? .get!))

theorem drone_highest_effective_average_speed :
highest_effective_average_speed_interval (List.map (λ p, effective_speed p.1 p.2) distance_and_wind) = 2 :=
by
  sorry

end drone_highest_effective_average_speed_l337_337665


namespace jim_anne_mary_paul_report_time_l337_337600

def typing_rate_jim := 1 / 12
def typing_rate_anne := 1 / 20
def combined_typing_rate := typing_rate_jim + typing_rate_anne
def typing_time := 1 / combined_typing_rate

def editing_rate_mary := 1 / 30
def editing_rate_paul := 1 / 10
def combined_editing_rate := editing_rate_mary + editing_rate_paul
def editing_time := 1 / combined_editing_rate

theorem jim_anne_mary_paul_report_time : 
  typing_time + editing_time = 15 := by
  sorry

end jim_anne_mary_paul_report_time_l337_337600


namespace range_of_a_monotonic_increasing_interval_l337_337110

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x^2 - 2

theorem range_of_a_monotonic_increasing_interval :
  ∃ a : ℝ, (a > -2) ∧ (∀ x ∈ set.Ioo (1 / 2) 2, 0 < (derivative (f a) x)) := sorry

end range_of_a_monotonic_increasing_interval_l337_337110


namespace profit_percentage_l337_337620

theorem profit_percentage (CP SP : ℝ) (h : 18 * CP = 16 * SP) : 
  (SP - CP) / CP * 100 = 12.5 := by
sorry

end profit_percentage_l337_337620


namespace first_month_sale_l337_337318

theorem first_month_sale (sales_2 : ℕ) (sales_3 : ℕ) (sales_4 : ℕ) (sales_5 : ℕ) (sales_6 : ℕ) (average_sale : ℕ) (total_months : ℕ)
  (H_sales_2 : sales_2 = 6927)
  (H_sales_3 : sales_3 = 6855)
  (H_sales_4 : sales_4 = 7230)
  (H_sales_5 : sales_5 = 6562)
  (H_sales_6 : sales_6 = 5591)
  (H_average_sale : average_sale = 6600)
  (H_total_months : total_months = 6) :
  ∃ (sale_1 : ℕ), sale_1 = 6435 :=
by
  -- placeholder for the proof
  sorry

end first_month_sale_l337_337318


namespace math_problem_solution_l337_337158

noncomputable def a_n (n : ℕ) : ℝ := (Real.sqrt 5)^n * Real.cos (n * Real.arctan 2)
noncomputable def b_n (n : ℕ) : ℝ := (Real.sqrt 5)^n * Real.sin (n * Real.arctan 2)

theorem math_problem_solution :
  ∑ n in Finset.range 8, a_n n * b_n n / 10^n = 7 / 16 :=
by
  sorry

end math_problem_solution_l337_337158


namespace circle_reflection_l337_337223

theorem circle_reflection (x y : ℤ) : 
    x = 7 → y = -3 → 
    let (x', y') := (y, x) in
    let (x'', y'') := (-x', y') in
    (x'', y'') = (3, 7) :=
by
  assume h1 h2
  let (x', y') := (y, x)
  let (x'', y'') := (-x', y')
  sorry

end circle_reflection_l337_337223


namespace max_pairs_correct_l337_337174

def max_pairs (n : ℕ) : ℕ :=
  if h : n > 1 then (n * n) / 4 else 0

theorem max_pairs_correct (n : ℕ) (h : n ≥ 2) :
  (max_pairs n = (n * n) / 4) :=
by sorry

end max_pairs_correct_l337_337174


namespace triangle_inequality_1_triangle_inequality_2_l337_337161

variable (a b c : ℝ)

theorem triangle_inequality_1 (h1 : a + b + c = 2) (h2 : 0 ≤ a) (h3 : 0 ≤ b) (h4 : 0 ≤ c) (h5 : a ≤ 1) (h6 : b ≤ 1) (h7 : c ≤ 1) : 
  a * b * c + 28 / 27 ≥ a * b + b * c + c * a :=
by
  sorry

theorem triangle_inequality_2 (h1 : a + b + c = 2) (h2 : 0 ≤ a) (h3 : 0 ≤ b) (h4 : 0 ≤ c) (h5 : a ≤ 1) (h6 : b ≤ 1) (h7 : c ≤ 1) : 
  a * b + b * c + c * a ≥ a * b * c + 1 :=
by
  sorry

end triangle_inequality_1_triangle_inequality_2_l337_337161


namespace pascal_triangle_contains_53_l337_337821

theorem pascal_triangle_contains_53 (n : ℕ) :
  (∃ k, binomial n k = 53) ↔ n = 53 := 
sorry

end pascal_triangle_contains_53_l337_337821


namespace rectangle_area_correct_l337_337606

-- Definitions of side lengths
def sideOne : ℝ := 5.9
def sideTwo : ℝ := 3

-- Definition of the area calculation for a rectangle
def rectangleArea (a b : ℝ) : ℝ :=
  a * b

-- The main theorem stating the area is as calculated
theorem rectangle_area_correct :
  rectangleArea sideOne sideTwo = 17.7 := by
  sorry

end rectangle_area_correct_l337_337606


namespace intersecting_lines_product_distances_l337_337747

-- Definitions for the problem conditions and questions
def circle (x y : ℝ) := x^2 + y^2 - 6*x - 4*y + 10
def line1 (k : ℝ) (x y : ℝ) := y = k*x
def line2 (x y : ℝ) := 3*x + 2*y + 10 = 0

theorem intersecting_lines (k : ℝ) (x y : ℝ) :
  (∃ (x y : ℝ), circle x y = 0 ∧ line1 k x y) ↔ (k > (6 - sqrt 30) / 6 ∧ k < (6 + sqrt 30) / 6) :=
sorry

theorem product_distances (k x y : ℝ) 
  (P Q : ℝ × ℝ) (O : ℝ × ℝ) :
  P = ((2*k + 3) / sqrt (k^2+1), 0) ∧ Q = (-10 / (3 + 2*k), -10*k / (3+2*k)) →
  |OP| = (sqrt (13 - ((3*k - 2)^2 / (k^2 + 1)))) →
  |OQ| = (10 / (3 + 2*k) * sqrt (1 + k^2)) →
  |OP| * |OQ| = 10 :=
sorry

end intersecting_lines_product_distances_l337_337747


namespace min_value_of_f_l337_337379

noncomputable def f (x : ℝ) := real.sqrt (x^2 - 2*x) + 2^real.sqrt (x^2 - 5*x + 4)

theorem min_value_of_f : 
  (∀ x, (x^2 - 2*x ≥ 0) → (x^2 - 5*x + 4 ≥ 0) → (x ≥ 4 ∨ x ≤ 0) → f x ≥ 2*real.sqrt(2) + 1) ∧
  (f 4 = 2*real.sqrt(2) + 1) :=
by
  sorry

end min_value_of_f_l337_337379


namespace minimum_positive_period_π_mono_increasing_interval_range_f_interval_l337_337059

noncomputable def f (x : ℝ) : ℝ :=
  cos x * sin (x + (Real.pi / 3)) - sqrt 3 * (cos x)^2 + (sqrt 3) / 4

-- The minimum positive period of the function f(x) is π.
theorem minimum_positive_period_π : Periodic f Real.pi := sorry

-- The monotonically increasing interval of the function f(x)
theorem mono_increasing_interval (k : ℤ) :
  ∀ x, -Real.pi / 12 + k * Real.pi ≤ x ∧ x ≤ k * Real.pi + 5 * Real.pi / 12 → deriv f x > 0 := sorry

-- The range of the function f(x) on [-π/4, π/3]
theorem range_f_interval : ∀ x, 
  -Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 3 → 
  -1 / 2 ≤ f x ∧ f x ≤ sqrt 3 / 4 := sorry

end minimum_positive_period_π_mono_increasing_interval_range_f_interval_l337_337059


namespace nth_inequality_l337_337190

noncomputable theory 
open_locale big_operators

def sum_of_reciprocals_odd (n : ℕ) : ℝ :=
  ∑ i in finset.range n, 1 / (2 * i + 1)

def sum_of_reciprocals_even (n : ℕ) : ℝ :=
  ∑ i in finset.range n, 1 / (2 * (i + 1))

theorem nth_inequality (n : ℕ) (h : 0 < n): 
  (1 / (n + 1) * sum_of_reciprocals_odd n) 
  ≥ (1 / n * sum_of_reciprocals_even n) := 
begin
  sorry
end

end nth_inequality_l337_337190


namespace quadratic_equation_roots_l337_337768

theorem quadratic_equation_roots (a b c : ℝ) : 
  (b ^ 6 > 4 * (a ^ 3) * (c ^ 3)) → (b ^ 10 > 4 * (a ^ 5) * (c ^ 5)) :=
by
  sorry

end quadratic_equation_roots_l337_337768


namespace area_T_l337_337160

-- Define the matrix
def matrix : Matrix (Fin 2) (Fin 2) ℝ := !![ 3, 4; 1, 6 ]

-- Define region T with area 7
def area_T : ℝ := 7

-- The desired theorem statement
theorem area_T'_transformation :
  let determinant := (matrix.det) in
  let area_T' := determinant * area_T in
  determinant = 14 ∧ area_T' = 98 :=
begin
  let determinant := (matrix.det),
  have h1 : determinant = 14,
    { sorry }, -- Determinant calculation step
  have h2 : determinant * area_T = 98,
    { sorry }, -- Area transformation step
  exact ⟨h1, h2⟩,
end

end area_T_l337_337160


namespace domain_is_correct_l337_337564

-- Given a function f(x)
def f (x : ℝ) := real.sqrt (1 - 2 * real.log x / real.log 6)

-- Conditions for the domain of the function
def conditions (x : ℝ) : Prop := 1 - 2 * real.log x / real.log 6 ≥ 0 ∧ x > 0

-- The domain of f 
def domain_of_f := {x : ℝ | conditions x}

-- Prove that the domain is (0, √6]
theorem domain_is_correct : domain_of_f = set.Ioc 0 (real.sqrt 6) :=
by
  sorry

end domain_is_correct_l337_337564


namespace solution_set_of_inequality_l337_337392

def f (x : ℝ) : ℝ :=
  if x > 0 then 1 else -1

theorem solution_set_of_inequality :
  {x : ℝ | x + (x + 2) * f (x + 2) ≤ 5} = {x : ℝ | x ≤ 3 / 2} :=
by
  sorry

end solution_set_of_inequality_l337_337392


namespace correct_growth_rate_equation_l337_337209

noncomputable def numberOfBikesFirstMonth : ℕ := 1000
noncomputable def additionalBikesThirdMonth : ℕ := 440
noncomputable def monthlyGrowthRate (x : ℝ) : Prop :=
  numberOfBikesFirstMonth * (1 + x)^2 = numberOfBikesFirstMonth + additionalBikesThirdMonth

theorem correct_growth_rate_equation (x : ℝ) : monthlyGrowthRate x :=
by
  sorry

end correct_growth_rate_equation_l337_337209


namespace geometric_progressions_sum_eq_l337_337585

variable {a q b : ℝ}
variable {n : ℕ}
variable (h1 : q ≠ 1)

/-- The given statement in Lean 4 -/
theorem geometric_progressions_sum_eq (h : a * (q^(3*n) - 1) / (q - 1) = b * (q^(3*n) - 1) / (q^3 - 1)) : 
  b = a * (1 + q + q^2) := 
by
  sorry

end geometric_progressions_sum_eq_l337_337585


namespace number_of_rows_containing_53_l337_337813

theorem number_of_rows_containing_53 (h_prime_53 : Nat.Prime 53) : 
  ∃! n, (n = 53 ∧ ∃ k, k ≥ 0 ∧ k ≤ n ∧ Nat.choose n k = 53) :=
by 
  sorry

end number_of_rows_containing_53_l337_337813


namespace number_of_unit_fraction_pairs_l337_337225

/-- 
 The number of ways that 1/2007 can be expressed as the sum of two distinct positive unit fractions is 7.
-/
theorem number_of_unit_fraction_pairs : 
  ∃ (pairs : Finset (ℕ × ℕ)), 
    (∀ p ∈ pairs, p.1 ≠ p.2 ∧ (1 : ℚ) / 2007 = 1 / ↑p.1 + 1 / ↑p.2) ∧ 
    pairs.card = 7 :=
sorry

end number_of_unit_fraction_pairs_l337_337225


namespace cubes_difference_l337_337720

theorem cubes_difference :
  let a := 642
  let b := 641
  a^3 - b^3 = 1234567 :=
by
  let a := 642
  let b := 641
  have h : a^3 - b^3 = 264609288 - 263374721 := sorry
  have h_correct : 264609288 - 263374721 = 1234567 := sorry
  exact Eq.trans h h_correct

end cubes_difference_l337_337720


namespace evaluate_expression_l337_337988

theorem evaluate_expression : 3 - (-3) ^ (-2 : ℤ) = 26 / 9 := by
  sorry

end evaluate_expression_l337_337988


namespace will_jogged_for_30_minutes_l337_337283

theorem will_jogged_for_30_minutes 
  (calories_before : ℕ)
  (calories_per_minute : ℕ)
  (net_calories_after : ℕ)
  (h1 : calories_before = 900)
  (h2 : calories_per_minute = 10)
  (h3 : net_calories_after = 600) :
  let calories_burned := calories_before - net_calories_after
  let jogging_time := calories_burned / calories_per_minute
  jogging_time = 30 := by
  sorry

end will_jogged_for_30_minutes_l337_337283


namespace evaluate_powers_of_i_l337_337006

theorem evaluate_powers_of_i (i : ℂ) (h : i^4 = 1) : i^5 + i^{17} + i^{-15} + i^{23} = 0 :=
by
  sorry

end evaluate_powers_of_i_l337_337006


namespace max_imaginary_part_theta_l337_337663

theorem max_imaginary_part_theta :
  ∃ θ : ℝ, -90 ≤ θ ∧ θ ≤ 90 ∧ θ = 45 ∧ 
      ∀ z : ℂ, z ^ 6 - z ^ 4 + z ^ 2 - 1 = 0 → 
      ∃ θ_ : ℝ, -90 ≤ θ_ ∧ θ_ ≤ 90 ∧ HasSin.sin θ_ = z.im ∧ θ_ ≤ θ :=
begin
  sorry
end

end max_imaginary_part_theta_l337_337663


namespace possible_values_reciprocals_l337_337512

variable {x y : ℝ}
variables (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 2)

theorem possible_values_reciprocals : 
  ∃ S : set ℝ, S = {z : ℝ | z ≥ 2} ∧ (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 2 ∧ z = (1/x + 1/y)) :=
sorry

end possible_values_reciprocals_l337_337512


namespace pd_eq_abs_pe_pf_l337_337994

variables {O A B C D P E F : Type}
variables [circle O] [equilateral_triangle (triangle A B C)]
variables (diameter AD : Π (x : O), x ≠ A)
variables (P_on_arc : P ∈ (arc B C) \ {A})
variables [incenter E (triangle P A B)] [incenter F (triangle P A C)]

theorem pd_eq_abs_pe_pf
  (circ_O : circle O)
  (eq_tri : equilateral_triangle (triangle A B C))
  (diam_AD : diameter AD)
  (arc_BC_P : P_on_arc P (arc B C))
  (incenter_PAB : incenter E (triangle P A B))
  (incenter_PAC : incenter F (triangle P A C)) :
  dist P D = |dist P E - dist P F| :=
sorry

end pd_eq_abs_pe_pf_l337_337994


namespace parallelogram_bf_length_l337_337863

theorem parallelogram_bf_length (A B C D E F : Type)
  [AddCommGroup A] [Module ℝ A]
  (AB BC CD DA BE EC : ℝ)
  (h_AB_420 : AB = 420)
  (h_ratio_BE_EC : BE/EC = 5/7)
  (h_AB_CD : AB = CD)
  {BF : ℝ}
  (h_BE_EC : BC = BE + EC)
  (h_DE_intersects_AB_ext : true) : BF = 300 := 
begin
  sorry
end

end parallelogram_bf_length_l337_337863


namespace sum_of_real_solutions_l337_337015

theorem sum_of_real_solutions (x : ℝ) (h : ∃ x : ℝ, (sqrt x + sqrt (9 / x) + sqrt (x + 9 / x) = 8)) : 
  (∑ x in {x : ℝ | sqrt x + sqrt (9 / x) + sqrt (x + 9 / x) = 8}, x) = 3025 / 256 :=
sorry

end sum_of_real_solutions_l337_337015


namespace find_a_l337_337743

theorem find_a {l : Line} {x y a : ℝ} (P : Point)
  (h1 : P = (0, 2)) 
  (h2 : is_tangent l ((λx y, (x - 1)^2 + y^2 = 5) : Point → ℝ → Bool)) 
  (h3 : is_perpendicular l (λx y, a * x - 2 * y + 1 = 0)) :
  a = -4 :=
sorry

end find_a_l337_337743


namespace find_ratio_l337_337569

noncomputable def p (x : ℝ) : ℝ := 3 * x * (x - 5)
noncomputable def q (x : ℝ) : ℝ := (x + 2) * (x - 5)

theorem find_ratio : (p 3) / (q 3) = 9 / 5 := by
  sorry

end find_ratio_l337_337569


namespace pascal_triangle_contains_53_l337_337809

theorem pascal_triangle_contains_53:
  ∃! n, ∃ k, (n ≥ 0) ∧ (k ≥ 0) ∧ (binom n k = 53) := 
sorry

end pascal_triangle_contains_53_l337_337809


namespace number_of_SYTs_l337_337673

theorem number_of_SYTs (shape: List ℕ): shape = [5,4,3,2,1] →
  StandardYoungTableaux.count shape = 292864 :=
begin
  intros h,
  rw h,
  sorry,
end

end number_of_SYTs_l337_337673


namespace length_of_PF_l337_337469

-- Variables and initial conditions
variables {P Q R L M F : Type} 
variables {PQ PR : ℝ}
variables [triangle P Q R]
variables [right_angled_at P Q R P]
variables [altitude P L]
variables [median R M]
variables [midpoint M P Q]
variables [intersection F (altitude P L) (median R M)]

-- Given lengths
def PQ : ℝ := 2
def PR : ℝ := 2 * Real.sqrt 3

-- The goal is to calculate PF
def PF := Real.sqrt 3 - (3 * Real.sqrt 3) / 7

theorem length_of_PF : PF = (4 * Real.sqrt 3) / 7 :=
sorry

end length_of_PF_l337_337469


namespace problem1_problem2_l337_337629

theorem problem1 : 0.008 ^ (-1 / 3 : ℝ) + 81 ^ (1 / 2 : ℝ) + real.log (sqrt (1 / 16 : ℝ)) / real.log (sqrt (2 : ℝ)) = 10 := 
by
  sorry

theorem problem2 (x : ℝ) : real.log x * real.log (x / 100) = 3 ↔ (x = 1000 ∨ x = 1 / 10) := 
by
  sorry

end problem1_problem2_l337_337629


namespace smallest_top_block_number_is_150_l337_337694

-- We define the structure of our pyramid and the requirement that the numbers in subsequent layers 
-- are sums of the numbers of the blocks below them.

inductive PyramidLayer
| BottomLayer (blocks: Fin 15) -- 15 blocks in the bottom layer
| SecondLayer (blocks: Fin 10) -- 10 blocks in the second layer
| ThirdLayer (blocks: Fin 6) -- 6 blocks in the third layer
| FourthLayer (blocks: Fin 3) -- 3 blocks in the fourth layer
| TopLayer (block: Fin 1) -- 1 block in the top layer

-- We specify a function that calculates the number for each block in the pyramid based on the 
-- numbers in the lower layer.
def blockNumber : PyramidLayer → Nat
| PyramidLayer.BottomLayer n := n + 1 -- blocks in bottom layer numbered 1 to 15
| PyramidLayer.SecondLayer n := 
  blockNumber (PyramidLayer.BottomLayer (2 * n)) + 
  blockNumber (PyramidLayer.BottomLayer (2 * n + 1)) + 
  blockNumber (PyramidLayer.BottomLayer (2 * n + 2))
| PyramidLayer.ThirdLayer n := 
  blockNumber (PyramidLayer.SecondLayer (2 * n)) + 
  blockNumber (PyramidLayer.SecondLayer (2 * n + 1)) + 
  blockNumber (PyramidLayer.SecondLayer (2 * n + 2))
| PyramidLayer.FourthLayer n := 
  blockNumber (PyramidLayer.ThirdLayer (2 * n)) + 
  blockNumber (PyramidLayer.ThirdLayer (2 * n + 1)) + 
  blockNumber (PyramidLayer.ThirdLayer (2 * n + 2))
| PyramidLayer.TopLayer n := 
  blockNumber (PyramidLayer.FourthLayer (2 * n)) + 
  blockNumber (PyramidLayer.FourthLayer (2 * n + 1)) + 
  blockNumber (PyramidLayer.FourthLayer (2 * n + 2))

-- The main theorem to be proved
theorem smallest_top_block_number_is_150 : blockNumber (PyramidLayer.TopLayer 0) = 150 := 
by sorry

end smallest_top_block_number_is_150_l337_337694


namespace min_val_f_l337_337233

def f (x : ℝ) : ℝ := x^3 + 3 / x

theorem min_val_f : ∃ x ∈ (set.Ioi (0 : ℝ)), f x = 4 := by
  sorry

end min_val_f_l337_337233


namespace total_rowing_campers_l337_337300

theorem total_rowing_campers (morning_rowing afternoon_rowing : ℕ) : 
  morning_rowing = 13 -> 
  afternoon_rowing = 21 -> 
  morning_rowing + afternoon_rowing = 34 :=
by
  sorry

end total_rowing_campers_l337_337300


namespace polynomial_has_exactly_one_real_root_l337_337922

theorem polynomial_has_exactly_one_real_root :
  ∀ (x : ℝ), (2007 * x^3 + 2006 * x^2 + 2005 * x = 0) → x = 0 :=
by
  sorry

end polynomial_has_exactly_one_real_root_l337_337922


namespace complex_arithmetic_l337_337903

variable (z : ℂ)
hypothesis h1 : z = 1 + complex.i

theorem complex_arithmetic :
  (2 / z + (conj z) ^ 2) = 1 - 3 * complex.i :=
by
  sorry

end complex_arithmetic_l337_337903


namespace rectangle_sides_l337_337645

theorem rectangle_sides :
  ∀ (x : ℝ), 
    (3 * x = 8) ∧ (8 / 3 * 3 = 8) →
    ((2 * (3 * x + x) = 3 * x^2) ∧ (2 * (3 * (8 / 3) + (8 / 3)) = 3 * (8 / 3)^2) →
    x = 8 / 3
      ∧ 3 * x = 8) := 
by
  sorry

end rectangle_sides_l337_337645


namespace factorization_sum_l337_337949

theorem factorization_sum :
  ∃ a b c : ℤ, (∀ x : ℝ, (x^2 + 20 * x + 96 = (x + a) * (x + b)) ∧
                      (x^2 + 18 * x + 81 = (x - b) * (x + c))) →
              (a + b + c = 30) :=
by
  sorry

end factorization_sum_l337_337949


namespace minimize_expression_at_c_l337_337725

theorem minimize_expression_at_c (c : ℝ) : (c = 7 / 4) → (∀ x : ℝ, 2 * c^2 - 7 * c + 4 ≤ 2 * x^2 - 7 * x + 4) :=
sorry

end minimize_expression_at_c_l337_337725


namespace euler_line_of_fourth_triangle_l337_337214

noncomputable theory

-- Define the points A, B, C, D as variables
variables (A B C D : Type) [metric_space A]

-- We need the points to form triangles and describe their Euler lines
axiom non_obtuse_triangles : 
  ∀ (P Q R : A), 
  ¬((angle P Q R > pi / 2) ∨ (angle Q R P > pi / 2) ∨ (angle R P Q > pi / 2))

-- Define Euler lines for each triangle
def euler_line (P Q R : A) : Type := sorry

-- Assume the Euler lines of three triangles are concurrent
axiom euler_lines_concurrent :
  ∃ (P : A),
  P ∈ euler_line A B D ∧
  P ∈ euler_line B C D ∧
  P ∈ euler_line C A D

-- Prove that the Euler line of the fourth triangle is also concurrent at the same point
theorem euler_line_of_fourth_triangle :
  ∃ (P : A),
  P ∈ euler_line A B C :=
sorry

end euler_line_of_fourth_triangle_l337_337214


namespace find_m_value_l337_337046

theorem find_m_value (m : ℝ) (P : Point) (l1 l2 l3 : Line) (d : ℝ)
  (hP : P = ⟨-1, 4⟩)
  (hl2_eq1 : l2 = ⟨2, -1, 5⟩ ∨ l2 = ⟨1, 2, -2⟩ ∨ (l2.slope = 1 ∧ l1.slope = 2 * 1))
  (hl1_eq : l1.contains P)
  (hl3_eq : l3 = ⟨4, -2, m⟩)
  (h_distance : distance_lines l1 l3 = 2 * (Real.sqrt 5)) :
  m = 32 ∨ m = -8 := sorry

end find_m_value_l337_337046


namespace same_circumcenter_of_triangles_l337_337515

open EuclideanGeometry

variables {A B C I O_A O_B O_C : Point}

/-- Given a triangle ABC with incenter I and the circumcenters of 
    triangles IBC, ICA, and IAB being O_A, O_B, and O_C respectively, 
    prove that the circumcenter of triangle O_A O_B O_C is the same 
    as the circumcenter of triangle ABC. -/
theorem same_circumcenter_of_triangles
  (hI : incenter I A B C)
  (hO_A : circumcenter O_A I B C)
  (hO_B : circumcenter O_B I C A)
  (hO_C : circumcenter O_C I A B) : 
  circumcenter A B C = circumcenter O_A O_B O_C := 
sorry

end same_circumcenter_of_triangles_l337_337515


namespace four_digit_numbers_count_even_four_digit_numbers_count_eighty_fifth_four_digit_number_l337_337980

/-- Given digits 0, 1, 2, 3, 4, 5 to form a four-digit number without repeating any digit,
the number of different four-digit numbers that can be formed is 300. -/
theorem four_digit_numbers_count : 
  let digits := [0, 1, 2, 3, 4, 5] in
  let count := list.permutations digits |>.filter (λ l, l.length = 4 ∧ l.head ≠ 0) |>.length in
  count = 300 := sorry

/-- Given digits 0, 1, 2, 3, 4, 5 to form a four-digit number without repeating any digit,
the number of even four-digit numbers is 156. -/
theorem even_four_digit_numbers_count : 
  let digits := [0, 1, 2, 3, 4, 5] in
  let even_count := list.permutations digits |>.filter (λ l, l.length = 4 ∧ l.head ≠ 0 ∧ l.ilast ∈ [0, 2, 4]) |>.length in
  even_count = 156 := sorry

/-- Given digits 0, 1, 2, 3, 4, 5 to form a four-digit number without repeating any digit,
ordered in ascending order,  the 85th number is 2301. -/
theorem eighty_fifth_four_digit_number : 
  let digits := [0, 1, 2, 3, 4, 5] in
  let four_digit_numbers := list.filter (λ l, l.length = 4 ∧ l.head ≠ 0) (list.permutations digits) in
  let sorted_numbers := list.sort four_digit_numbers in
  nth sorted_numbers 84 = [2, 3, 0, 1] := sorry

end four_digit_numbers_count_even_four_digit_numbers_count_eighty_fifth_four_digit_number_l337_337980


namespace find_multiple_of_q_l337_337103

variable (p q m : ℚ)

theorem find_multiple_of_q (h1 : p / q = 3 / 4) (h2 : 3 * p + m * q = 6.25) :
  m = 4 :=
sorry

end find_multiple_of_q_l337_337103


namespace resulting_polygon_sides_rational_l337_337636

-- Define a structure for Rational Polygon
structure ConvexPolygon (n : ℕ) :=
(vertices : Fin n → (ℚ × ℚ))  -- List of vertices with rational coordinates
(is_convex : ∀ (i j k : Fin n), i ≠ j → i ≠ k → j ≠ k → 
  -- Convexity condition: turn left at each vertex
  ((vertices j).1 - (vertices i).1) * ((vertices k).2 - (vertices j).2) > 
  ((vertices k).1 - (vertices j).1) * ((vertices j).2 - (vertices i).2))

-- Function to retrieve side lengths, assuming side lengths are rational
def side_lengths (P : ConvexPolygon n) : Fin n → ℚ :=
  λ i, (((P.vertices i).1 - (P.vertices (i + 1) % n).1)^2 + 
        ((P.vertices i).2 - (P.vertices (i + 1) % n).2)^2).sqrt

-- Define condition that all sides and diagonals are rational
axiom all_sides_rational (P : ConvexPolygon n) : 
  ∀ i, ((P.vertices i).1 ≠ (P.vertices (i + 1) % n).1) → 
  (((P.vertices i).1 - (P.vertices (i + 1) % n).1)^2 + 
   ((P.vertices i).2 - (P.vertices (i + 1) % n).2)^2).sqrt ∈ ℚ

-- Define condition that all diagonals are rational
axiom all_diagonals_rational (P : ConvexPolygon n) : 
  ∀ i j, i ≠ j → 
  (((P.vertices i).1 - (P.vertices j).1)^2 + 
   ((P.vertices i).2 - (P.vertices j).2)^2).sqrt ∈ ℚ

-- Theorem to prove the side lengths of the resulting smaller polygons are rational.
theorem resulting_polygon_sides_rational (P : ConvexPolygon n) :
  all_sides_rational P →
  all_diagonals_rational P →
  ∀ (i j k : Fin n), i ≠ j → i ≠ k → j ≠ k → 
  ∀ (P' : ConvexPolygon 3), 
  P'.vertices = ![(P.vertices i), (P.vertices j), (P.vertices k)] →
  (all_sides_rational P' ∧ all_diagonals_rational P') :=
by sorry

end resulting_polygon_sides_rational_l337_337636


namespace sqrt_equality_l337_337612

theorem sqrt_equality (hA : Real.sqrt 8 = 2 * Real.sqrt 2)
                      (hB : Real.sqrt 9 = 3)
                      (hC : Real.sqrt 12 = 2 * Real.sqrt 3)
                      (hD : Real.sqrt 18 = 3 * Real.sqrt 2) :
  (3 * Real.sqrt 2) = Real.sqrt 18 :=
by
  exact hD

end sqrt_equality_l337_337612


namespace minimum_m_value_l337_337744

theorem minimum_m_value 
  (n : ℕ) (h_n : n ≥ 2) 
  (s : Finset ℕ) (h_s : s.card = 2 * n + 2) (h_range : ∀ x ∈ s, x ∈ Finset.range (3 * n + 1)) :
  ∃ a b c d ∈ s, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ a = b + c + d :=
sorry

end minimum_m_value_l337_337744


namespace sequence_a_100_l337_337745

noncomputable def a : ℕ → ℕ
| 1       := 2
| (n + 1) := a n + 2 * n

theorem sequence_a_100 :
  a 100 = 9902 :=
sorry

end sequence_a_100_l337_337745


namespace find_sides_of_rectangle_l337_337648

-- Define the conditions
def isRectangle (w l : ℝ) : Prop :=
  l = 3 * w ∧ 2 * l + 2 * w = l * w

-- Main theorem statement
theorem find_sides_of_rectangle (w l : ℝ) :
  isRectangle w l → w = 8 / 3 ∧ l = 8 :=
by
  sorry

end find_sides_of_rectangle_l337_337648


namespace sum_of_consecutive_odds_l337_337965

theorem sum_of_consecutive_odds (N1 N2 N3 : ℕ) (h1 : N1 % 2 = 1) (h2 : N2 % 2 = 1) (h3 : N3 % 2 = 1)
  (h_consec1 : N2 = N1 + 2) (h_consec2 : N3 = N2 + 2) (h_max : N3 = 27) : 
  N1 + N2 + N3 = 75 := by
  sorry

end sum_of_consecutive_odds_l337_337965


namespace biking_ratio_l337_337531

variable (S : ℕ)
variable (MondayDistance : ℕ := 6)
variable (WednesdayDistance : ℕ := 12)
variable (TotalDistance : ℕ := 30)

theorem biking_ratio :
  6 + 12 + S = 30 → S = 12 → (S : ℕ) / MondayDistance = 2 :=
by
  intros h1 h2
  rw [h2]
  exact by norm_num

#eval biking_ratio -- The Lean statement should build successfully.

end biking_ratio_l337_337531


namespace distance_walked_west_is_10_l337_337321

-- Definitions from conditions
def distance_north := 10
def distance_from_start := 14.142135623730951

-- The problem statement: Prove that the man walked 10 miles west
theorem distance_walked_west_is_10 :
  ∃ (distance_west : ℝ), 
  (distance_west^2 + distance_north^2 = distance_from_start^2) ∧ 
  (distance_west = 10) :=
by {
  -- Since we are only writing the statement
  sorry
}

end distance_walked_west_is_10_l337_337321


namespace invisible_square_exists_l337_337351

theorem invisible_square_exists (L : ℕ) :
  ∃ (a b : ℤ), ∀ (i j : ℕ), (0 ≤ i ∧ i ≤ L) ∧ (0 ≤ j ∧ j ≤ L) → 
  ∃ p : ℕ, nat.prime p ∧ p > 1 ∧ p ∣ (a + i) ∧ p ∣ (b + j) :=
by
  sorry

end invisible_square_exists_l337_337351


namespace conditional_probability_example_l337_337311

theorem conditional_probability_example (P : Set (Set ℝ) → ℝ)
  {A B : Set ℝ}
  (hP_A : P A = 0.8)
  (hP_B : P B = 0.5)
  (hP_A_and_B : P (A ∩ B) = 0.5) :
  P(B | A) = 5 / 8 :=
by
  have hP_A_ne_zero : P A ≠ 0 := by linarith [hP_A]
  have h_cond := @real.cond_prob_def (Set ℝ) P _ A B hP_A_ne_zero
  rw [hP_A_and_B, hP_A] at h_cond
  exact h_cond

end conditional_probability_example_l337_337311


namespace impossible_equal_sums_3x3_l337_337867

theorem impossible_equal_sums_3x3 (a b c d e f g h i : ℕ) :
  a + b + c = 13 ∨ a + b + c = 14 ∨ a + b + c = 15 ∨ a + b + c = 16 ∨ a + b + c = 17 ∨ a + b + c = 18 ∨ a + b + c = 19 ∨ a + b + c = 20 →
  (a + d + g) = 13 ∨ (a + d + g) = 14 ∨ (a + d + g) = 15 ∨ (a + d + g) = 16 ∨ (a + d + g) = 17 ∨ (a + d + g) = 18 ∨ (a + d + g) = 19 ∨ (a + d + g) = 20 →
  (a + e + i) = 13 ∨ (a + e + i) = 14 ∨ (a + e + i) = 15 ∨ (a + e + i) = 16 ∨ (a + e + i) = 17 ∨ (a + e + i) = 18 ∨ (a + e + i) = 19 ∨ (a + e + i) = 20 →
  1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 1 ≤ d ∧ d ≤ 9 ∧ 1 ≤ e ∧ e ≤ 9 ∧ 1 ≤ f ∧ f ≤ 9 ∧ 1 ≤ g ∧ g ≤ 9 ∧ 1 ≤ h ∧ h ≤ 9 ∧ 1 ≤ i ∧ i ≤ 9 →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ g ≠ h ∧ g ≠ i ∧ h ≠ i →
  false :=
sorry

end impossible_equal_sums_3x3_l337_337867


namespace flight_time_estimate_l337_337960

noncomputable def estimate_flight_time (radius : ℝ) (speed : ℝ) : ℝ :=
  (2 * Real.pi * radius) / speed

theorem flight_time_estimate
  (radius : ℝ)
  (speed : ℝ)
  (hr : radius = 5000)
  (hs : speed = 600) :
  estimate_flight_time radius speed ≈ 50 :=
by
  simp [estimate_flight_time, hr, hs]
  sorry

end flight_time_estimate_l337_337960


namespace radius_of_third_circle_equals_28_l337_337264

theorem radius_of_third_circle_equals_28 
  (r1 r2 : ℝ) (h1 : r1 = 21) (h2 : r2 = 35) : 
  ∃ (r3 : ℝ), r3 = 28 ∧ π * r3 ^ 2 = π * (r2 ^ 2 - r1 ^ 2) :=
by 
  -- Apply the given conditions
  rw [h1, h2]
  have h_shaded : π * (35^2 - 21^2) = 784π := sorry
  existsi 28
  split
  -- Show that r3 = 28
  exact rfl
  -- Show that the area of the third circle equals the shaded area
  rw [← h_shaded, mul_comm, mul_assoc]
  sorry -- Calculations to complete the proof

end radius_of_third_circle_equals_28_l337_337264


namespace probability_outside_unit_circle_l337_337129

-- Probability measure over the interval [-1, 1].
def square_integral := (set.Icc (-1 : ℝ) 1).prod (set.Icc (-1 : ℝ) 1)

noncomputable def area_of_unit_circle : ℝ := pi * 1 * 1

noncomputable def area_of_square_minus_unit_circle : ℝ := 4 - pi

noncomputable def probability_x2_y2_geq_1 : ℝ :=
  (area_of_square_minus_unit_circle / 4) * 4

theorem probability_outside_unit_circle :
  probability_x2_y2_geq_1 = 1 - (pi / 8) :=
sorry

end probability_outside_unit_circle_l337_337129


namespace pascal_triangle_contains_53_only_once_l337_337801

theorem pascal_triangle_contains_53_only_once (n : ℕ) (k : ℕ) (h_prime : Nat.prime 53) :
  (n = 53 ∧ (k = 1 ∨ k = 52) ∨ 
   ∀ m < 53, Π l, Nat.binomial m l ≠ 53) ∧ 
  (n > 53 → (k = 0 ∨ k = n ∨ Π a b, a * 53 ≠ b * Nat.factorial (n - k + 1))) :=
sorry

end pascal_triangle_contains_53_only_once_l337_337801


namespace days_to_complete_work_l337_337844

theorem days_to_complete_work
  (M B W : ℝ)  -- Define variables for daily work done by a man, a boy, and the total work
  (hM : M = 2 * B)  -- Condition: daily work done by a man is twice that of a boy
  (hW : (13 * M + 24 * B) * 4 = W)  -- Condition: 13 men and 24 boys complete work in 4 days
  (H : 12 * M + 16 * B) -- Help Lean infer the first group's total work per day
  (hW2 : (12 * M + 16 * B) * 5 = W)  -- Condition: first group must complete work in same time (5 days, to be proven)
  : (12 * M + 16 * B) * 5 = W := -- Prove equivalence
sorry

end days_to_complete_work_l337_337844


namespace pascal_contains_53_l337_337793

theorem pascal_contains_53 (n : ℕ) (h1 : Nat.Prime 53) (h2 : ∃ k, 1 ≤ k ∧ k ≤ 52 ∧ nat.choose 53 k = 53) (h3 : ∀ m < 53, ¬ (∃ k, 1 ≤ k ∧ k ≤ m - 1 ∧ nat.choose m k = 53)) (h4 : ∀ m > 53, ¬ (∃ k, 1 ≤ k ∧ k ≤ m - 1 ∧ nat.choose m k = 53)) : 
  (n = 53) → (n = 1) := 
by
  intros
  sorry

end pascal_contains_53_l337_337793


namespace second_route_time_l337_337319

-- Defining time for the first route with all green lights
def R_green : ℕ := 10

-- Defining the additional time added by each red light
def per_red_light : ℕ := 3

-- Defining total time for the first route with all red lights
def R_red : ℕ := R_green + 3 * per_red_light

-- Defining the second route time plus the difference
def S : ℕ := R_red - 5

theorem second_route_time : S = 14 := by
  sorry

end second_route_time_l337_337319


namespace eval_abs_7_minus_sqrt_53_l337_337366

theorem eval_abs_7_minus_sqrt_53 : ∀ (a b : ℝ), a = 7 → b = real.sqrt 53 → a < b → |a - b| = b - a := by
  intros a b ha hb hab
  rw [ha, hb]
  sorry

end eval_abs_7_minus_sqrt_53_l337_337366


namespace find_xy_plus_yz_plus_xz_l337_337058

theorem find_xy_plus_yz_plus_xz
  (x y z : ℝ)
  (h₁ : x > 0)
  (h₂ : y > 0)
  (h₃ : z > 0)
  (eq1 : x^2 + x * y + y^2 = 75)
  (eq2 : y^2 + y * z + z^2 = 64)
  (eq3 : z^2 + z * x + x^2 = 139) :
  x * y + y * z + z * x = 80 :=
by
  sorry

end find_xy_plus_yz_plus_xz_l337_337058


namespace pascals_triangle_53_rows_l337_337789

theorem pascals_triangle_53_rows : 
  ∃! row, (∃ k, 1 ≤ k ∧ k ≤ row ∧ 53 = Nat.choose row k) ∧ 
          (∀ k, 1 ≤ k ∧ k ≤ row → 53 = Nat.choose row k → row = 53) :=
sorry

end pascals_triangle_53_rows_l337_337789


namespace axis_of_symmetry_l337_337112

theorem axis_of_symmetry {a b c : ℝ} (h1 : (2 : ℝ) * (a * 2 + b) + c = 5) (h2 : (4 : ℝ) * (a * 4 + b) + c = 5) : 
  (2 + 4) / 2 = 3 := 
by 
  sorry

end axis_of_symmetry_l337_337112


namespace find_a_sol_l337_337741

noncomputable def find_a (a : ℝ) : Prop :=
  let circle_eq : ℝ × ℝ → Prop := λ coordinates, (coordinates.fst + a)^2 + (coordinates.snd)^2 = 4
  let line_eq : ℝ × ℝ → Prop := λ coordinates, coordinates.fst - coordinates.snd - 4 = 0
  let distance_from_center_to_line := abs (-a - 4) / real.sqrt 2 = real.sqrt 2 -- distance formula
  let chord_length_condition := real.sqrt (2 * 2 - 2) = real.sqrt 2 -- chord length condition
  distance_from_center_to_line ∧ chord_length_condition

theorem find_a_sol (a : ℝ) :
  find_a a ↔ (a = -2 ∨ a = -6) :=
sorry

end find_a_sol_l337_337741


namespace geometric_sequence_product_l337_337874

theorem geometric_sequence_product (a : ℕ → ℝ) (r : ℝ) (h_geom : ∀ n, a (n+1) = r * a n) (h_cond : a 7 * a 12 = 5) :
  a 8 * a 9 * a 10 * a 11 = 25 :=
by 
  sorry

end geometric_sequence_product_l337_337874


namespace find_a_l337_337049

theorem find_a (A B : Set ℝ) (a : ℝ)
  (hA : A = {1, 2})
  (hB : B = {a, a^2 + 1})
  (hUnion : A ∪ B = {0, 1, 2}) :
  a = 0 :=
sorry

end find_a_l337_337049


namespace nice_triples_l337_337155

theorem nice_triples (a b : ℚ) (c : ℕ) :
  (∀ n : ℕ, S a n = (S b n) ^ c) → ((a = b ∧ c = 1) ∨ (a = 3 ∧ b = 1 ∧ c = 2)) :=
sorry

def S (r : ℚ) (n : ℕ) : ℚ := ∑ i in finset.range (n + 1), (i : ℚ) ^ r

end nice_triples_l337_337155


namespace identify_even_and_decreasing_function_l337_337664

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_monotonically_decreasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f x ≥ f y

theorem identify_even_and_decreasing_function :
  ∃! f : ℝ → ℝ, 
  (
    (f = fun x => -x^2 + 1) ∧ 
    is_even f ∧
    is_monotonically_decreasing f (Set.Ioi 0) ∧
    (f ≠ fun x => real.log (abs x)) ∧ 
    (f ≠ fun x => x⁻¹) ∧ 
    (f ≠ fun x => real.cos x)
  ) :=
by
  sorry

end identify_even_and_decreasing_function_l337_337664


namespace proof_n_properties_l337_337009

def is_possible_grouping (n : ℕ) : Prop :=
  ∃ (groups : list (ℕ × ℕ)), 
    list.length groups = 500 ∧
    list.sum (list.map (λ p, p.1 * p.2) groups) ∣ n

def correct_n : ℕ :=
  1111... (expand the number fully in digits as given in the solution)

theorem proof_n_properties : 
  (digits_of correct_n).length = 1000 ∧ 
  (∀ d, d ∈ digits_of correct_n → d ≠ 0) ∧ 
  is_possible_grouping correct_n :=
begin
  sorry
end

end proof_n_properties_l337_337009


namespace radian_to_degree_conversion_l337_337354

theorem radian_to_degree_conversion (h : real.pi = 180) : -8 * real.pi / 3 = -480 :=
by
  sorry

end radian_to_degree_conversion_l337_337354


namespace circumcenter_on_circumcircle_l337_337514

variables {A B C H D P Q : Type*}
variables [AcuteTriangle A B C]
variable [Orthocenter A B C H]
variable [Circumcircle (Triangle H A B) ω]

-- Define the conditions
variable (D : intersect (Circumcircle H A B) (line B C) ≠ B)
variable (P : Intersection (line D H) (segment A C))
variable (Q : Circumcenter (Triangle A D P))

-- Main theorem statement
theorem circumcenter_on_circumcircle (center_ω : center (Circumcircle (Triangle H A B))) :
  lies_on (center_ω) (Circumcircle (Triangle B D Q)) :=
sorry

end circumcenter_on_circumcircle_l337_337514


namespace Kayla_and_Kimiko_age_ratio_l337_337148

theorem Kayla_and_Kimiko_age_ratio :
  ∀ (K_age K_min_drive K_wait Kimiko_age: ℤ) (K_wait_pos: K_wait > 0),
    Kimiko_age = 26 →
    K_min_drive = 18 →
    K_wait = 5 →
    K_age = K_min_drive - K_wait →
  (K_age / Kimiko_age : ℚ) = 1 / 2 :=
by {
  intros K_age K_min_drive K_wait Kimiko_age K_wait_pos h1 h2 h3 h4,
  sorry
}

end Kayla_and_Kimiko_age_ratio_l337_337148


namespace John_study_time_second_exam_l337_337884

variable (StudyTime Score : ℝ)
variable (k : ℝ) (h1 : k = Score / StudyTime)
variable (study_first : ℝ := 3) (score_first : ℝ := 60)
variable (avg_target : ℝ := 75)
variable (total_tests : ℕ := 2)

theorem John_study_time_second_exam :
  (avg_target * total_tests - score_first) / (score_first / study_first) = 4.5 :=
by
  sorry

end John_study_time_second_exam_l337_337884


namespace billy_scores_two_points_each_round_l337_337343

def billy_old_score := 725
def billy_rounds := 363
def billy_target_score := billy_old_score + 1
def billy_points_per_round := billy_target_score / billy_rounds

theorem billy_scores_two_points_each_round :
  billy_points_per_round = 2 := by
  sorry

end billy_scores_two_points_each_round_l337_337343


namespace unique_solution_in_interval_maximum_value_m_l337_337433

-- Problem 1
theorem unique_solution_in_interval (f : ℝ → ℝ) (g : ℝ → ℝ) (a m x : ℝ) (h1 : ∀ x, f(x) = exp x) :
  (∃! x, 2 < x ∧ x < 3 ∧ (f (-x) = 6 - 2 * x)) :=
sorry

-- Problem 2
theorem maximum_value_m (f : ℝ → ℝ) (g : ℝ → ℝ) (a m x : ℝ) (h1 : ∀ x, f(x) = exp x)
  (h2 : ∀ x, g(x) = x^3 - 6*x^2 + 3*x + a) (h3 : a ≥ 0) (h4 : m > 1) :
  (∀ x ∈ set.Icc (1:ℝ) m, f(x) * g(x) ≤ x) → m ≤ 5 :=
sorry

end unique_solution_in_interval_maximum_value_m_l337_337433


namespace minimum_value_of_xy_l337_337394

theorem minimum_value_of_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y - x * y = 0) : xy = 64 :=
sorry

end minimum_value_of_xy_l337_337394


namespace train_speeds_l337_337542

noncomputable def c1 : ℝ := sorry  -- speed of the passenger train in km/min
noncomputable def c2 : ℝ := sorry  -- speed of the freight train in km/min
noncomputable def c3 : ℝ := sorry  -- speed of the express train in km/min

def conditions : Prop :=
  (5 / c1 + 5 / c2 = 15) ∧
  (5 / c2 + 5 / c3 = 11) ∧
  (c2 ≤ c1) ∧
  (c3 ≤ 2.5)

-- The theorem to be proved
theorem train_speeds :
  conditions →
  (40 / 60 ≤ c1 ∧ c1 ≤ 50 / 60) ∧ 
  (100 / 3 / 60 ≤ c2 ∧ c2 ≤ 40 / 60) ∧ 
  (600 / 7 / 60 ≤ c3 ∧ c3 ≤ 150 / 60) :=
sorry

end train_speeds_l337_337542


namespace sin_add_cos_l337_337587

theorem sin_add_cos (s72 c18 c72 s18 : ℝ) (h1 : s72 = Real.sin (72 * Real.pi / 180)) (h2 : c18 = Real.cos (18 * Real.pi / 180)) (h3 : c72 = Real.cos (72 * Real.pi / 180)) (h4 : s18 = Real.sin (18 * Real.pi / 180)) :
  s72 * c18 + c72 * s18 = 1 :=
by 
  sorry

end sin_add_cos_l337_337587


namespace min_k_minus_s_l337_337853

noncomputable def smallest_difference (k r s : ℕ) (table_width table_height : ℕ) : ℕ :=
  let total_cells := table_width * table_height
  total_cells - s

theorem min_k_minus_s (k r s : ℕ) (table_width : ℕ) (table_height : ℕ) :
  k + r + s = table_width * table_height →
  k ≥ r →
  r ≥ s →
  (∀ i j, (i = 0 ∨ i = table_height - 1 ∨ j = 0 ∨ j = table_width - 1) →
          at_least_two_neighbors_same_color i j k r s) →
  (∀ i j, (i ≠ 0 ∧ i ≠ table_height - 1 ∧ j ≠ 0 ∧ j ≠ table_width - 1) →
          at_least_three_neighbors_same_color i j k r s) →
  smallest_difference k r s table_width table_height = 28 := 
sorry

def at_least_two_neighbors_same_color (i j k r s : ℕ) : Prop := sorry
def at_least_three_neighbors_same_color (i j k r s : ℕ) : Prop := sorry

end min_k_minus_s_l337_337853


namespace functional_eq_solutions_l337_337355

noncomputable def functional_equation_solution (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, f(x * f(y) + y) = f(x * y) + f(y)

theorem functional_eq_solutions (f : ℝ → ℝ) :
  functional_equation_solution f → (∀ x : ℝ, f(x) = 0 ∨ f(x) = x) :=
by
  intro h
  sorry

end functional_eq_solutions_l337_337355


namespace minimal_p0_of_conditions_l337_337557

noncomputable def random_variable_X : Type := Nat → ℝ
def p_0 (X : random_variable_X) : ℝ := X 0
def EX (X : random_variable_X) : ℝ := ∑ i, i * X i
def EX2 (X : random_variable_X) : ℝ := ∑ i, (i^2) * X i
def EX3 (X : random_variable_X) : ℝ := ∑ i, (i^3) * X i

theorem minimal_p0_of_conditions (X : random_variable_X) (h1 : EX X = 1) (h2 : EX2 X = 2) (h3 : EX3 X = 5) :
  (∀ i, 0 ≤ X i) ∧ (∑ i, X i = 1) → p_0 X = 1 / 3 :=
by
  sorry

end minimal_p0_of_conditions_l337_337557


namespace pizza_division_l337_337535

theorem pizza_division (N : ℕ) (hN : N = 201 ∨ N = 400) : 
  ∃ cuts: ℕ, cuts ≤ 100 ∧ is_possible_division N cuts := 
sorry

-- Assume the necessary definitions of is_possible_division
-- This means is_possible_division N cuts denotes whether the pizza can be divided into N equal parts using "cuts" cuts.

-- You can define is_possible_division mathematically later if needed
def is_possible_division (N : ℕ) (cuts : ℕ) : Prop := sorry

end pizza_division_l337_337535


namespace max_value_proof_l337_337169

noncomputable def maximum_value (x y z : ℝ) : ℝ :=
  x + y^3 + z^4

theorem max_value_proof
  (x y z : ℝ)
  (hx : 0 ≤ x)
  (hy : 0 ≤ y)
  (hz : 0 ≤ z)
  (h1 : x + y + z = 1)
  (h2 : x^2 + y^2 + z^2 = 1) :
  maximum_value x y z ≤ 1 :=
sorry

end max_value_proof_l337_337169


namespace divisibility_if_and_only_if_l337_337177

variables {a b m n : ℕ}

def coprime (a b : ℕ) := nat.gcd a b = 1

theorem divisibility_if_and_only_if
  (hnat : ∀ n, 0 < n)
  (hcoprime : coprime a b)
  (hgreater : a > 1)
: (∃ k : ℕ, k % 2 = 1 ∧ m = k * n) ↔ (a ^ m + b ^ m) % (a ^ n + b ^ n) = 0 :=
sorry

end divisibility_if_and_only_if_l337_337177


namespace geometric_sequence_sum_l337_337964

noncomputable def geometric_sequence (a : ℕ → ℝ) (r: ℝ): Prop :=
  ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) (r: ℝ)
  (h_geometric : geometric_sequence a r)
  (h_ratio : r = 2)
  (h_sum_condition : a 1 + a 4 + a 7 = 10) :
  a 3 + a 6 + a 9 = 20 := 
sorry

end geometric_sequence_sum_l337_337964


namespace num_tangent_circles_tangent_to_line_l337_337501

open EuclideanGeometry -- open Euclidean geometry namespace to use its definitions and theorems more conveniently

variable {plane : Type*} [metric_space plane] [normed_group plane] 

def num_tangent_circles_of_radius_four 
  (C₁ C₂ : Circle plane) (L : Line plane)
  (hC₁ : C₁.radius = 2) (hC₂ : C₂.radius = 2)
  (h_tangent_C₁_C₂ : C₁ ∂ C₂) (hL : L ∂ C₁ ∧ L ∂ C₂) 
  : ℕ := sorry

-- Here is the statement
theorem num_tangent_circles_tangent_to_line
  {C₁ C₂ : Circle plane} {L : Line plane} 
  (hC₁ : C₁.radius = 2) (hC₂ : C₂.radius = 2)
  (h_tangent_C₁_C₂ : C₁ ∂ C₂) (hL_tangent : L ∂ C₁ ∧ L ∂ C₂) 
  : num_tangent_circles_of_radius_four C₁ C₂ L hC₁ hC₂ h_tangent_C₁_C₂ hL_tangent = 2 := sorry

end num_tangent_circles_tangent_to_line_l337_337501


namespace cherries_purchase_l337_337727

theorem cherries_purchase (total_money : ℝ) (price_per_kg : ℝ) 
  (genevieve_money : ℝ) (shortage : ℝ) (clarice_money : ℝ) :
  genevieve_money = 1600 → shortage = 400 → clarice_money = 400 → price_per_kg = 8 →
  total_money = genevieve_money + shortage + clarice_money →
  total_money / price_per_kg = 250 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end cherries_purchase_l337_337727


namespace log2_ratio_squared_l337_337195

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log2_ratio_squared :
  ∀ (x y : ℝ), x ≠ 1 → y ≠ 1 → log_base 2 x = log_base y 25 → x * y = 81
  → (log_base 2 (x / y))^2 = 5.11 :=
by
  intros x y hx hy hlog hxy
  sorry

end log2_ratio_squared_l337_337195


namespace pencil_cost_l337_337972

theorem pencil_cost 
  (x y : ℚ)
  (h1 : 3 * x + 2 * y = 165)
  (h2 : 4 * x + 7 * y = 303) :
  y = 19.155 := 
by
  sorry

end pencil_cost_l337_337972


namespace farmer_land_l337_337908

noncomputable def farmer_land_example (A : ℝ) : Prop :=
  let cleared_land := 0.90 * A
  let barley_land := 0.70 * cleared_land
  let potatoes_land := 0.10 * cleared_land
  let corn_land := 0.10 * cleared_land
  let tomatoes_bell_peppers_land := 0.10 * cleared_land
  tomatoes_bell_peppers_land = 90 → A = 1000

theorem farmer_land (A : ℝ) (h_cleared_land : 0.90 * A = cleared_land)
  (h_barley_land : 0.70 * cleared_land = barley_land)
  (h_potatoes_land : 0.10 * cleared_land = potatoes_land)
  (h_corn_land : 0.10 * cleared_land = corn_land)
  (h_tomatoes_bell_peppers_land : 0.10 * cleared_land = 90) :
  A = 1000 :=
by
  sorry

end farmer_land_l337_337908


namespace angle_between_tangents_l337_337441

open real

/-- 
Given two concentric circles with radii 1 and 3 and a common center O, 
and a third circle that touches both of them, the angle between the tangents 
to the third circle drawn from point O is 60 degrees.
-/
theorem angle_between_tangents (O O1 : Point) (r1 r2 : ℝ) (h1 : r1 = 1) (h2 : r2 = 3)
  (h3 : dist O O1 = 2) (h4 : tangent_to_circle O O1 r1) (h5 : tangent_to_circle O O1 r2) :
  ∠(tangent O O1 r1) (tangent O O1 r2) = 60 :=
sorry

end angle_between_tangents_l337_337441


namespace hyperbola_m_value_l337_337848

theorem hyperbola_m_value (m k : ℝ) (h₀ : k > 0) (h₁ : 0 < -m) 
  (h₂ : 2 * k = Real.sqrt (1 + m)) : 
  m = -3 := 
by {
  sorry
}

end hyperbola_m_value_l337_337848


namespace sum_distances_eq_segment_l337_337244

theorem sum_distances_eq_segment (A B P : Point) :
    PA + PB = AB ↔ P ∈ segment A B := 
sorry

end sum_distances_eq_segment_l337_337244


namespace minimum_filtrations_l337_337644

noncomputable def ref_lg2 : Real := 0.301

theorem minimum_filtrations (n : ℕ) (h1 : 0.0 < (1 - 0.2 : Real))
  (h2 : \lg (2 : Real) = ref_lg2) : (1 - 0.2)^n < 0.05 → n ≥ 14 :=
by
  sorry

end minimum_filtrations_l337_337644


namespace b_bounded_l337_337153

open Real

-- Define sequences of real numbers
def a : ℕ → ℝ := sorry
def b : ℕ → ℝ := sorry

-- Define initial conditions and properties
axiom a0_gt_half : a 0 > 1/2
axiom a_non_decreasing : ∀ n : ℕ, a (n + 1) ≥ a n
axiom b_recursive : ∀ n : ℕ, b (n + 1) = a n * (b n + b (n + 2))

-- Prove the sequence (b_n) is bounded
theorem b_bounded : ∃ M : ℝ, ∀ n : ℕ, b n ≤ M :=
by
  sorry

end b_bounded_l337_337153


namespace grace_walks_distance_l337_337444

theorem grace_walks_distance
  (south_blocks west_blocks : ℕ)
  (block_length_in_miles : ℚ)
  (h_south_blocks : south_blocks = 4)
  (h_west_blocks : west_blocks = 8)
  (h_block_length : block_length_in_miles = 1 / 4)
  : ((south_blocks + west_blocks) * block_length_in_miles = 3) :=
by 
  sorry

end grace_walks_distance_l337_337444


namespace x_cubed_plus_recip_l337_337835

theorem x_cubed_plus_recip (x : ℝ) (hx : 53 = x^6 + x^(-6)) : 
  x^3 + x^(-3) = real.sqrt 55 ∨ x^3 + x^(-3) = -real.sqrt 55 := 
by 
  sorry

end x_cubed_plus_recip_l337_337835


namespace impossible_to_all_minus_l337_337695

def initial_grid : List (List Int) :=
  [[1, 1, -1, 1], 
   [-1, -1, 1, 1], 
   [1, 1, 1, 1], 
   [1, -1, 1, -1]]

-- Define the operation of flipping a row
def flip_row (grid : List (List Int)) (r : Nat) : List (List Int) :=
  grid.mapIdx (fun i row => if i == r then row.map (fun x => -x) else row)

-- Define the operation of flipping a column
def flip_col (grid : List (List Int)) (c : Nat) : List (List Int) :=
  grid.map (fun row => row.mapIdx (fun j x => if j == c then -x else x))

-- Predicate to check if all elements in the grid are -1
def all_minus (grid : List (List Int)) : Prop :=
  grid.all (fun row => row.all (fun x => x = -1))

-- The main theorem
theorem impossible_to_all_minus (init : List (List Int)) (hf1 : init = initial_grid) :
  ∀ grid, (grid = init ∨ ∃ r, grid = flip_row grid r ∨ ∃ c, grid = flip_col grid c) →
  ¬ all_minus grid := by
    sorry

end impossible_to_all_minus_l337_337695


namespace general_term_a_sequence_1_exists_c_a_sequence_2_l337_337178

-- Definition for the sequence with b = 1
def a_sequence_1 (n : ℕ) : ℝ :=
  if n = 0 then 1 else real.sqrt ((a_sequence_1 (n - 1)) ^ 2 - 2 * (a_sequence_1 (n - 1)) + 2) + 1

-- Proof Problem 1 in Lean 4: Prove that for all n in ℕ, a_sequence_1 follows the general term formula
theorem general_term_a_sequence_1 (n : ℕ) (h : n ≠ 0) : 
  a_sequence_1 n = real.sqrt (n - 1) + 1 :=
sorry

-- Definition for the sequence with b = -1
def a_sequence_2 (n : ℕ) : ℝ :=
  if n = 0 then 1 else real.sqrt ((a_sequence_2 (n - 1)) ^ 2 - 2 * (a_sequence_2 (n - 1)) + 2) - 1

-- Proof Problem 2 in Lean 4: Prove that there exists a real number c such that a_{2n} < c < a_{2n+1} for all n
theorem exists_c_a_sequence_2 : 
  ∃ c : ℝ, c = 1 / 4 ∧ ∀ (n : ℕ), n ≠ 0 → a_sequence_2 (2 * n) < c ∧ c < a_sequence_2 (2 * n + 1) :=
sorry

end general_term_a_sequence_1_exists_c_a_sequence_2_l337_337178


namespace find_b_continuity_l337_337150

def f (x : ℝ) (b : ℝ) :=
  if x ≤ 4 then 5 * x^2 + 4
  else b * x + 2

theorem find_b_continuity (b : ℝ) : 
  (∀ x ∈ {4}, f x b = 84) -> b = 20.5 := by 
  sorry

end find_b_continuity_l337_337150


namespace range_of_a_l337_337436

theorem range_of_a (a : ℝ) (p q : Prop) :
  (p ↔ (1 < a ∧ a < 2)) →
  (q ↔ (0 < a ∧ a ≤ sqrt 3)) →
  (¬ (p ∧ q)) →
  (p ∨ q) →
  (a ∈ Set.Ioc sqrt 3 2 ∪ Set.Icc 0 1) :=
by intros h₁ h₂ h₃ h₄
   sorry

end range_of_a_l337_337436


namespace cyclist_time_no_wind_l337_337315

theorem cyclist_time_no_wind (v w : ℝ) 
    (h1 : v + w = 1 / 3) 
    (h2 : v - w = 1 / 4) : 
    1 / v = 24 / 7 := 
by
  sorry

end cyclist_time_no_wind_l337_337315


namespace determine_a_l337_337896

theorem determine_a (a : ℝ) (ha : a ≠ 0) (hfocal : ∃ (x y : ℝ), x^2 + a * y^2 + a^2 = 0 ∧ 2 * (sqrt (a^2 - a)) = 4) :
  a = (1 - sqrt 17) / 2 :=
by sorry

end determine_a_l337_337896


namespace balloons_remaining_proof_l337_337340

-- The initial number of balloons the clown has
def initial_balloons : ℕ := 3 * 12

-- The number of boys who buy balloons
def boys : ℕ := 3

-- The number of girls who buy balloons
def girls : ℕ := 12

-- The total number of children buying balloons
def total_children : ℕ := boys + girls

-- The remaining number of balloons after sales
def remaining_balloons (initial : ℕ) (sold : ℕ) : ℕ := initial - sold

-- Problem statement: Proof that the remaining balloons are 21 given the conditions
theorem balloons_remaining_proof : remaining_balloons initial_balloons total_children = 21 := sorry

end balloons_remaining_proof_l337_337340


namespace scale_circle_to_ellipse_l337_337133

-- Definitions based on conditions
def circle (x y : ℝ) : Prop := x^2 + y^2 = 1

def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

-- Scaling transformation
def scaling_transformation (x y : ℝ) : ℝ × ℝ := (3 * x, 2 * y)

-- Theorem stating that the transformation converts the circle to the ellipse
theorem scale_circle_to_ellipse (x y : ℝ) (hx : circle x y) : ellipse (3 * x) (2 * y) :=
by
  -- Proof omitted
  sorry

end scale_circle_to_ellipse_l337_337133


namespace maximum_notebooks_maria_can_buy_l337_337185

def price_single : ℕ := 1
def price_pack_4 : ℕ := 3
def price_pack_7 : ℕ := 5
def total_budget : ℕ := 10

def max_notebooks (budget : ℕ) : ℕ :=
  if budget < price_single then 0
  else if budget < price_pack_4 then budget / price_single
  else if budget < price_pack_7 then max (budget / price_single) (4 * (budget / price_pack_4))
  else max (budget / price_single) (7 * (budget / price_pack_7))

theorem maximum_notebooks_maria_can_buy :
  max_notebooks total_budget = 14 := by
  sorry

end maximum_notebooks_maria_can_buy_l337_337185


namespace valid_rearrangements_count_l337_337783

theorem valid_rearrangements_count : 
  let s := "abcde".to_list in
  let valid_permutation (perm : List Char) := 
    ∀ (i : Nat), i < perm.length - 1 → 
    |perm.get i, perm.get (i + 1)| ∉ [|('a','b'),('b','c'),('c','d'),('d','e'),('b','a'),('c','b'),('d','c'),('e','d')|] in
  (List.permutations s).count valid_permutation = 4 := 
by
  sorry

end valid_rearrangements_count_l337_337783


namespace prime_p_function_exists_eq_two_l337_337022

theorem prime_p_function_exists_eq_two (p : ℕ) (hp : Nat.Prime p) :
  (∃ f : Fin (p) → Fin (p), ∀ n : Fin (p), (n.val * f n.val * f (f n.val) - 1) % p = 0 ) → p = 2 :=
by sorry

end prime_p_function_exists_eq_two_l337_337022


namespace number_of_bugs_seen_l337_337907

-- Defining the conditions
def flowers_per_bug : ℕ := 2
def total_flowers_eaten : ℕ := 6

-- The statement to prove
theorem number_of_bugs_seen : total_flowers_eaten / flowers_per_bug = 3 :=
by
  sorry

end number_of_bugs_seen_l337_337907


namespace inequality_rel_neither_sufficient_nor_necessary_l337_337473

theorem inequality_rel_neither_sufficient_nor_necessary (a b : ℝ) (h1 : 2^a > 2^b) : 
  ¬((\frac{1}{a} < \frac{1}{b}) ↔ (2^a > 2^b)) :=
by sorry

end inequality_rel_neither_sufficient_nor_necessary_l337_337473


namespace number_count_two_digit_property_l337_337657

open Nat

theorem number_count_two_digit_property : 
  (∃ (n : Finset ℕ), (∀ (x : ℕ), x ∈ n ↔ ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 11 * a + 2 * b ≡ 7 [MOD 10] ∧ x = 10 * a + b) ∧ n.card = 5) :=
by
  sorry

end number_count_two_digit_property_l337_337657


namespace village_population_rate_decrease_l337_337602

theorem village_population_rate_decrease :
  ∃ R : ℕ, 
    let population_X := 70000
        population_Y := 42000
        increase_Y := 800
        years := 14
    in population_X - years * R = population_Y + years * increase_Y ∧ R = 1200 :=
  by
    sorry

end village_population_rate_decrease_l337_337602


namespace count_multiples_of_7_ending_in_6_below_500_l337_337782

theorem count_multiples_of_7_ending_in_6_below_500 : 
  {n : ℕ | n < 500 ∧ ∃ k, n = 7 * (10 * k + 4)}.card = 7 :=
sorry

end count_multiples_of_7_ending_in_6_below_500_l337_337782


namespace cherries_purchase_l337_337728

theorem cherries_purchase (total_money : ℝ) (price_per_kg : ℝ) 
  (genevieve_money : ℝ) (shortage : ℝ) (clarice_money : ℝ) :
  genevieve_money = 1600 → shortage = 400 → clarice_money = 400 → price_per_kg = 8 →
  total_money = genevieve_money + shortage + clarice_money →
  total_money / price_per_kg = 250 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end cherries_purchase_l337_337728


namespace line_eq_l337_337945

theorem line_eq {x y : ℝ} :
  (∃ A : ℝ × ℝ, A = (-1, 4)) ∧ (∃ B : ℝ × ℝ, B = (3, 0)) →
  (∀ A B : ℝ × ℝ, A = (-1, 4) ∧ B = (3, 0) → (∀ x y, x + y - 3 = 0)) :=
begin
  sorry
end

end line_eq_l337_337945


namespace sum_of_squares_base_6_l337_337348

def to_base (n b : ℕ) : ℕ := sorry

theorem sum_of_squares_base_6 :
  let squares := (List.range 12).map (λ x => x.succ ^ 2);
  let squares_base6 := squares.map (λ x => to_base x 6);
  (squares_base6.sum) = to_base 10515 6 :=
by sorry

end sum_of_squares_base_6_l337_337348


namespace even_function_a_is_neg_one_l337_337107

-- Define f and the condition that it is an even function
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define the given function f(x) = (x-1)*(x-a)
def f (a x : ℝ) : ℝ := (x - 1) * (x - a)

-- Statement to prove that if f is an even function, then a must be -1
theorem even_function_a_is_neg_one (a : ℝ) :
  is_even (f a) → a = -1 :=
by 
  intro h,
  sorry

end even_function_a_is_neg_one_l337_337107


namespace people_in_park_at_11am_l337_337708

-- Define sequences for entering and leaving the park
def entering (n : ℕ) : ℕ := 2 ^ n
def leaving (n : ℕ) : ℕ := n - 1

-- Define sequence for number of people in the park after each interval
def netEntering (n : ℕ) : ℕ := entering (n + 1) - leaving (n + 1)

-- Define the sum of the first n terms of the sequence
def S (n : ℕ) : ℕ := ∑ i in range n, netEntering i

-- The main theorem to prove
theorem people_in_park_at_11am : S 10 = 2 ^ 11 - 47 := by
  sorry

end people_in_park_at_11am_l337_337708


namespace max_non_overlapping_areas_l337_337312

theorem max_non_overlapping_areas (n : ℕ) (h : 0 < n) :
  ∃ k : ℕ, k = 4 * n + 1 :=
sorry

end max_non_overlapping_areas_l337_337312


namespace triangle_area_l337_337135

noncomputable def area_of_triangle (a b c : ℝ) (θ : ℝ) :=
  (1 / 2) * a * b * Real.sin θ

theorem triangle_area :
  ∀ (A B C : Type) [euclidean_space A] [euclidean_space B]
      (AC BC : ℝ) (angle_B : ℝ),
  AC = Real.sqrt 7 →
  BC = 2 →
  angle_B = Real.pi / 3 →
  area_of_triangle 3 2 (Real.sqrt 7) (Real.pi / 3) = (3 * Real.sqrt 3) / 2 :=
by
  sorry

end triangle_area_l337_337135


namespace g_properties_l337_337976

-- Define the original function f
def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 2)

-- Define the transformation to the right by π/4 units
def g (x : ℝ) : ℝ := f (x - Real.pi / 4)

-- Define the properties to prove
theorem g_properties : 
  (∀ x : ℝ, 0 < x ∧ x < Real.pi / 4 → g x < g (x - ε)) ∧ 
  (∀ x : ℝ, g (-x) = -g x) :=
sorry

end g_properties_l337_337976


namespace train_b_speed_l337_337975

theorem train_b_speed (v : ℝ) (t : ℝ) (d : ℝ) (sA : ℝ := 30) (start_time_diff : ℝ := 2) :
  (d = 180) -> (60 + sA*t = d) -> (v * t = d) -> v = 45 := by 
  sorry

end train_b_speed_l337_337975


namespace cube_root_less_than_five_count_l337_337084

theorem cube_root_less_than_five_count :
  (∃ n : ℕ, n = 124 ∧ ∀ x : ℕ, 1 ≤ x → x < 5^3 → x < 125) := 
sorry

end cube_root_less_than_five_count_l337_337084


namespace equal_area_locus_l337_337076

variables {A B C P : Type} [AffineSpace Type A B C P]
noncomputable def E : A := midpoint A C

theorem equal_area_locus :
  ∀ (P : A), (area P A B = area P B C) ↔
    (P ∈ line_through B E ∨ P ∈ parallel_through B (line_through A C)) :=
by
  sorry

end equal_area_locus_l337_337076


namespace number_of_rows_containing_53_l337_337812

theorem number_of_rows_containing_53 (h_prime_53 : Nat.Prime 53) : 
  ∃! n, (n = 53 ∧ ∃ k, k ≥ 0 ∧ k ≤ n ∧ Nat.choose n k = 53) :=
by 
  sorry

end number_of_rows_containing_53_l337_337812


namespace ellipse_sum_l337_337502

noncomputable def h : ℝ := 3
noncomputable def k : ℝ := 0
noncomputable def a : ℝ := 5
noncomputable def b : ℝ := Real.sqrt 21
noncomputable def F_1 : (ℝ × ℝ) := (1, 0)
noncomputable def F_2 : (ℝ × ℝ) := (5, 0)

theorem ellipse_sum :
  (F_1 = (1, 0)) → 
  (F_2 = (5, 0)) →
  (∀ P : (ℝ × ℝ), (Real.sqrt ((P.1 - F_1.1)^2 + (P.2 - F_1.2)^2) + Real.sqrt ((P.1 - F_2.1)^2 + (P.2 - F_2.2)^2) = 10)) →
  (h + k + a + b = 8 + Real.sqrt 21) :=
by
  intros
  sorry

end ellipse_sum_l337_337502


namespace hyperbola_eccentricity_l337_337038

theorem hyperbola_eccentricity (a b c e : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : ∀ x y, (x^2 / a^2) - (y^2 / b^2) = 1 → (x, y) ≠ (3, -4))
  (h2 : b / a = 4 / 3)
  (h3 : b^2 = c^2 - a^2)
  (h4 : c / a = e):
  e = 5 / 3 :=
by
  sorry

end hyperbola_eccentricity_l337_337038


namespace pascals_triangle_53_rows_l337_337788

theorem pascals_triangle_53_rows : 
  ∃! row, (∃ k, 1 ≤ k ∧ k ≤ row ∧ 53 = Nat.choose row k) ∧ 
          (∀ k, 1 ≤ k ∧ k ≤ row → 53 = Nat.choose row k → row = 53) :=
sorry

end pascals_triangle_53_rows_l337_337788


namespace count_equilateral_triangles_in_T_l337_337505

def point3d := (ℕ × ℕ × ℕ)

def T : set point3d := {p | ∃ x y z, p = (x, y, z) ∧ x ∈ {0, 1, 2, 3} ∧ y ∈ {0, 1, 2, 3} ∧ z ∈ {0, 1, 2, 3}}

def distance (p1 p2 : point3d) :=
  let (x1, y1, z1) := p1 in
  let (x2, y2, z2) := p2 in
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

def is_equilateral (p1 p2 p3 : point3d) :=
  distance p1 p2 = distance p2 p3 ∧ distance p2 p3 = distance p3 p1

def equilateral_triangles_count (T : set point3d) : ℕ :=
  -- Count all unique equilateral triangles from set T
  sorry

theorem count_equilateral_triangles_in_T : equilateral_triangles_count T = 115 :=
  sorry

end count_equilateral_triangles_in_T_l337_337505


namespace find_a_l337_337427

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1 / 3) * x ^ 2 + a * Real.log x

theorem find_a (b : ℝ) (h_tangent : ∀ x y, y = (x - b) → x - y + b = 0) :
  (∃ (a : ℝ), ∀ x, x = 2 → 
    (f' := (λ x, (2 / 3) * x + a / x))
    (f' 2 = 1) → a = -2 / 3) :=
sorry

end find_a_l337_337427


namespace jill_draws_spade_probability_l337_337144

noncomputable def probability_jill_draws_spade : ℚ :=
  ∑' (k : ℕ), ((3 / 4) * (3 / 4))^k * ((3 / 4) * (1 / 4))

theorem jill_draws_spade_probability : probability_jill_draws_spade = 3 / 7 :=
sorry

end jill_draws_spade_probability_l337_337144


namespace number_of_segments_l337_337326

-- Definitions based on the conditions
def rope_length : ℝ := 3       -- The rope is 3 meters long
def segment_length : ℝ := 1/4  -- Each segment is 1/4 meter long

-- The theorem that needs to be proved
theorem number_of_segments :
  rope_length / segment_length = 12 := 
by
  sorry

end number_of_segments_l337_337326


namespace integer_values_less_than_4sqrt3_l337_337448

theorem integer_values_less_than_4sqrt3 : 
  {x : ℤ | |(x : ℝ)| < 4 * Real.sqrt 3}.finite.to_finset.card = 13 := 
sorry

end integer_values_less_than_4sqrt3_l337_337448


namespace union_M_N_l337_337902

def M : Set ℝ := { x | x^2 + 2 * x = 0 }
def N : Set ℝ := { x | x^2 - 2 * x = 0 }

theorem union_M_N : M ∪ N = {0, -2, 2} := by
  sorry

end union_M_N_l337_337902


namespace solve_inequality_l337_337246

theorem solve_inequality :
  ∀ x : ℝ, (3 * x^2 - 4 * x - 7 < 0) ↔ (-1 < x ∧ x < 7 / 3) :=
by
  sorry

end solve_inequality_l337_337246


namespace find_j_in_polynomial_l337_337950

noncomputable def arithmeticProgressionPoly (a d : ℝ) : Polynomial ℝ :=
  Polynomial.C (a * (a + d) * (a + 2 * d) * (a + 3 * d))

theorem find_j_in_polynomial (j k : ℝ) (hp : ∀ a (r : List ℝ), 
  r = [a, a + (-2/3 * a), a + (2/3 * a), a + (-(4/3) * a)] → 
  r.prod ((-) (Polynomial.C 0)) = 225 * Polynomial.C 1) :
  j = -50 := by
    sorry

end find_j_in_polynomial_l337_337950


namespace base_n_number_modular_count_valid_n_integers_l337_337722

def base_n_number := λ (n : ℕ), 2 * n^5 + 2 * n^4 + n^3 + 3 * n^2 + n + 4

theorem base_n_number_modular {n : ℕ} (h : 2 ≤ n ∧ n ≤ 100) :
  (212314_n % 5 = 0) ↔ n ≡ 1 [MOD 5] :=
sorry

theorem count_valid_n_integers : finset.filter (λ n, (212314_n % 5 = 0)) (finset.range 101) = 19 :=
sorry

end base_n_number_modular_count_valid_n_integers_l337_337722


namespace ivan_paid_l337_337861

noncomputable theory

/-- Each Uno Giant Family Card costs $15 -/
def card_price : ℚ := 15

/-- Ivan purchased 30 pieces -/
def quantity : ℕ := 30

/-- Tiered discount system for bulk purchasing -/
def tiered_discount (q : ℕ) : ℚ :=
  if q ≥ 35 then 0.20
  else if q ≥ 25 then 0.15
  else if q ≥ 15 then 0.10
  else 0

/-- Additional store membership discount -/
def membership_discount : ℚ := 0.05

/-- Calculate the discounted price per card given the tiered discount -/
def discounted_price_per_card (price : ℚ) (discount : ℚ) : ℚ :=
  price * (1 - discount)

/-- Calculate the total cost for given quantity and discounted price per card -/
def total_cost (quantity : ℕ) (discounted_price : ℚ): ℚ :=
  quantity * discounted_price

/-- Calculate the final cost after membership discount -/
def final_price (total : ℚ) (membership_discount : ℚ) : ℚ :=
  total * (1 - membership_discount)

theorem ivan_paid (card_price : ℚ) (quantity : ℕ) (membership_discount : ℚ)
  (tiered_discount : ℕ → ℚ) :
  let discount := tiered_discount quantity in
  let discounted_price := discounted_price_per_card card_price discount in
  let total := total_cost quantity discounted_price in
  let final := final_price total membership_discount in
  final = 363.38 :=
by sorry

end ivan_paid_l337_337861


namespace pascal_triangle_contains_53_once_l337_337826

theorem pascal_triangle_contains_53_once
  (h_prime : Nat.Prime 53) :
  ∃! n, ∃ k, n ≥ k ∧ n > 0 ∧ k > 0 ∧ Nat.choose n k = 53 := by
  sorry

end pascal_triangle_contains_53_once_l337_337826


namespace N_inverse_is_linear_combination_l337_337899

open Matrix -- Opens matrix notation
open Scalar -- Opens scalar functionality

noncomputable def N : Matrix (Fin 2) (Fin 2) ℚ := !![3, 1; 4, -2]

noncomputable def N_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  Matrix.inv N

theorem N_inverse_is_linear_combination :
  ∃ (c d : ℚ),
    N_inv =
    c • N + d • (1 : Matrix (Fin 2) (Fin 2) ℚ) :=
begin
  use [1 / 10, -1 / 10],
  sorry
end

end N_inverse_is_linear_combination_l337_337899
