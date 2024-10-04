import Mathlib

namespace distance_between_stripes_l267_267123

/-- Define the parallel curbs and stripes -/
structure Crosswalk where
  distance_between_curbs : ℝ
  curb_distance_between_stripes : ℝ
  stripe_length : ℝ
  stripe_cross_distance : ℝ
  
open Crosswalk

/-- Conditions given in the problem -/
def crosswalk : Crosswalk where
  distance_between_curbs := 60 -- feet
  curb_distance_between_stripes := 20 -- feet
  stripe_length := 50 -- feet
  stripe_cross_distance := 50 -- feet

/-- Theorem to prove the distance between stripes -/
theorem distance_between_stripes (cw : Crosswalk) :
  2 * (cw.curb_distance_between_stripes * cw.distance_between_curbs) / cw.stripe_length = 24 := sorry

end distance_between_stripes_l267_267123


namespace cubic_root_sum_eq_constant_term_divided_l267_267653

theorem cubic_root_sum_eq_constant_term_divided 
  (a b c : ℝ) 
  (h_roots : (24 * a^3 - 36 * a^2 + 14 * a - 1 = 0) 
           ∧ (24 * b^3 - 36 * b^2 + 14 * b - 1 = 0) 
           ∧ (24 * c^3 - 36 * c^2 + 14 * c - 1 = 0))
  (h_bounds : 0 < a ∧ a < 1 ∧ 0 < b ∧ b < 1 ∧ 0 < c ∧ c < 1) 
  : (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c)) = (158 / 73) := 
sorry

end cubic_root_sum_eq_constant_term_divided_l267_267653


namespace product_of_five_consecutive_is_divisible_by_sixty_l267_267518

theorem product_of_five_consecutive_is_divisible_by_sixty (n : ℤ) :
  60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end product_of_five_consecutive_is_divisible_by_sixty_l267_267518


namespace min_g_is_three_l267_267692

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

noncomputable def g (x y z : ℝ) : ℝ :=
  max (distance (x, y) (z, z))
      (max (distance (x, y) (6 - z, z - 6))
           (distance (x, y) (0, 0)))

theorem min_g_is_three : ∃ x y z : ℝ, z ≠ 0 ∧ z ≠ 6 ∧ ∀ x' y' z', z' ≠ 0 ∧ z' ≠ 6 → g x y z ≤ g x' y' z' := by
  sorry

end min_g_is_three_l267_267692


namespace total_surface_area_excluding_bases_l267_267094

def lower_base_radius : ℝ := 8
def upper_base_radius : ℝ := 5
def frustum_height : ℝ := 6
def cylinder_section_height : ℝ := 2
def cylinder_section_radius : ℝ := 5

theorem total_surface_area_excluding_bases :
  let l := Real.sqrt (frustum_height ^ 2 + (lower_base_radius - upper_base_radius) ^ 2)
  let lateral_surface_area_frustum := π * (lower_base_radius + upper_base_radius) * l
  let lateral_surface_area_cylinder := 2 * π * cylinder_section_radius * cylinder_section_height
  lateral_surface_area_frustum + lateral_surface_area_cylinder = 39 * π * Real.sqrt 5 + 20 * π :=
by
  sorry

end total_surface_area_excluding_bases_l267_267094


namespace unique_solution_is_candidate_l267_267906

open Real

def positiveReals := {x : ℝ // x > 0}

def satisfies_condition (f: positiveReals → positiveReals) : Prop :=
  ∀ (x : positiveReals), ∃! (y : positiveReals), x.val * f y + y.val * f x ≤ 2

noncomputable def candidate_function : positiveReals → positiveReals :=
λ x, ⟨1 / x.val, by exact one_div_pos.mpr x.property⟩

theorem unique_solution_is_candidate :
  ∀ (f : positiveReals → positiveReals), satisfies_condition f → f = candidate_function :=
by sorry

end unique_solution_is_candidate_l267_267906


namespace option_d_satisfies_equation_l267_267555

theorem option_d_satisfies_equation (x y z : ℤ) (h1 : x = z) (h2 : y = x + 1) : x * (x - y) + y * (y - z) + z * (z - x) = 2 :=
by
  sorry

end option_d_satisfies_equation_l267_267555


namespace function_fixed_point_l267_267389

theorem function_fixed_point (a : ℝ) : (∃ y, y = a ^ (3 - 3) + 3 ∧ y = 4) :=
by
  use 4
  rw [pow_zero]
  norm_num
  split; refl
  sorry

end function_fixed_point_l267_267389


namespace max_knights_is_six_l267_267287

-- Definitions used in the Lean 4 statement should appear:
-- Room can be occupied by either a knight (True for knight) or a liar (False for liar)
inductive Person : Type
| knight : Person
| liar : Person

open Person

-- Define the 3x3 grid
noncomputable def grid : Type := Matrix (Fin 3) (Fin 3) Person

-- Define neighbors
def neighbors (i j : Fin 3) : List (Fin 3 × Fin 3) :=
  [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)].filter (λ ⟨i', j'⟩, i'.val < 3 ∧ j'.val < 3)

-- Define the maximum number of knights can be in grid
def maxKnights (g : grid) : ℕ :=
  (Fin 3).fold (λ acc i, (Fin 3).fold (λ acc' j, if g i j = knight then acc' + 1 else acc') acc) 0

-- Main theorem statement
theorem max_knights_is_six {g : grid}
  (h : ∀ i j, 
       (g i j = knight →
        ∃ (ni nj : Fin 3), 
          (ni, nj) ∈ neighbors i j ∧ g ni nj = liar)) : 
  maxKnights g ≤ 6 :=
sorry

end max_knights_is_six_l267_267287


namespace S₈_proof_l267_267855

-- Definitions
variables {a₁ q : ℝ} {S : ℕ → ℝ}

-- Condition for the sum of the first n terms of the geometric sequence
def geom_sum (n : ℕ) : ℝ :=
  a₁ * (1 - q ^ n) / (1 - q)

-- Conditions from the problem
def S₄ := geom_sum 4 = -5
def S₆ := geom_sum 6 = 21 * geom_sum 2

-- Theorem to prove
theorem S₈_proof (h₁ : S₄) (h₂ : S₆) : geom_sum 8 = -85 :=
by
  sorry

end S₈_proof_l267_267855


namespace value_of_k_l267_267276

open Real

theorem value_of_k {k : ℝ} : 
  (∃ x : ℝ, k * x ^ 2 - 2 * k * x + 4 = 0 ∧ (∀ y : ℝ, k * y ^ 2 - 2 * k * y + 4 = 0 → x = y)) → k = 4 := 
by
  intros h
  sorry

end value_of_k_l267_267276


namespace product_of_five_consecutive_integers_divisible_by_120_l267_267453

theorem product_of_five_consecutive_integers_divisible_by_120 (n : ℤ) : 
  120 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end product_of_five_consecutive_integers_divisible_by_120_l267_267453


namespace statement_correct_D_l267_267063

theorem statement_correct_D :
  ∃ (coeff1 coeff2 coeff3 : ℤ) (a b : ℕ → ℕ) (degree1 degree2 degree3 : ℕ),
  coeff1 = 5 → coeff2 = -4 → coeff3 = 1 →
  a 0 = 2 → b 0 = 3 → a 1 = 2 → b 1 = 1 →
  degree1 = a 0 + b 0 → degree2 = a 1 + b 1 → 
  degree3 = 0 →
  list.max [degree1, degree2, degree3] = 5 ∧ list.length [coeff1, coeff2, coeff3] = 3 := 
by
  sorry

end statement_correct_D_l267_267063


namespace max_perimeter_of_right_angled_quadrilateral_is_4rsqrt2_l267_267553

noncomputable def max_perimeter_of_right_angled_quadrilateral (r : ℝ) : ℝ :=
  4 * r * Real.sqrt 2

theorem max_perimeter_of_right_angled_quadrilateral_is_4rsqrt2
  (r : ℝ) :
  ∃ (k : ℝ), 
  (∀ (x y : ℝ), x^2 + y^2 = 4 * r^2 → 2 * (x + y) ≤ max_perimeter_of_right_angled_quadrilateral r)
  ∧ (k = max_perimeter_of_right_angled_quadrilateral r) :=
sorry

end max_perimeter_of_right_angled_quadrilateral_is_4rsqrt2_l267_267553


namespace fair_special_savings_l267_267136

-- Define the conditions
def regular_price (price : ℕ) (h : price = 40) := price
def second_hat_discount := 0.3
def third_hat_discount := 0.6
def num_hats (n : ℕ) := n = 3

-- Formulate the problem
theorem fair_special_savings (price : ℕ) (h : price = 40) :
  let total_regular_price := 3 * price in
  let second_hat_price := price * (1 - second_hat_discount) in
  let third_hat_price := price * (1 - third_hat_discount) in
  let discounted_total := price + second_hat_price + third_hat_price in
  let savings := total_regular_price - discounted_total in
  let percentage_saved := (savings / total_regular_price) * 100 in
  percentage_saved = 30 :=
by
  sorry

end fair_special_savings_l267_267136


namespace pages_copied_l267_267311

theorem pages_copied (cents_per_page : ℚ) (pages_copied_per_cents : ℚ) (total_cents : ℚ) :
  (cents_per_page = 5) → (pages_copied_per_cents = 3) → (total_cents = 1500) → 
  (total_cents * (pages_copied_per_cents / cents_per_page) = 900) :=
by
  intros h_cp h_pc h_tc
  rw [h_cp, h_pc, h_tc]
  simp
  sorry

end pages_copied_l267_267311


namespace polar_to_rectangular_conversion_l267_267169

theorem polar_to_rectangular_conversion :
  ∀ (r θ : ℝ), r = 12 → θ = 11 * Real.pi / 6 →
  (r * Real.cos θ, r * Real.sin θ) = (6 * Real.sqrt 3, -6) :=
by
  intros r θ hr hθ
  rw [hr, hθ]
  rw [Real.cos_periodic_sub, Real.sin_periodic_sub, Real.cos_pi_div_six, Real.sin_pi_div_six]
  simp
  ring
  sorry

end polar_to_rectangular_conversion_l267_267169


namespace isosceles_triangle_incenter_length_ID_l267_267896

/-- Given an isosceles triangle ABC with AB = AC, base BC = 40, 
    the incenter I such that IC = 25, and D the midpoint of BC, 
    prove that the length of the segment ID is 15. -/
theorem isosceles_triangle_incenter_length_ID :
  ∀ (A B C I D : Type) [IsoscelesTriangle A B C] [BC_eq_40 : BC = 40]
    [Icenter : I = incenter A B C] [IC_eq_25 : IC = 25]
    [D_midpoint : D = midpoint B C], 
    length (ID) = 15 := 
sorry

end isosceles_triangle_incenter_length_ID_l267_267896


namespace sum_of_factorials_l267_267923

theorem sum_of_factorials (n : ℕ) (h : 0 < n) : (∑ i in Finset.range n.succ, (i + 1) * (i + 1)!) = (n + 2)! - 1 :=
  by sorry

end sum_of_factorials_l267_267923


namespace eight_term_sum_l267_267818

variable {α : Type*} [Field α]
variable (a q : α)

-- Define the n-th sum of the geometric sequence
def S_n (n : ℕ) : α := a * (1 - q ^ n) / (1 - q)

-- Given conditions
def S4 : α := S_n 4 = -5
def S6 : α := S_n 6 = 21 * S_n 2

-- Prove the target statement
theorem eight_term_sum : S_n 8 = -85 :=
  sorry

end eight_term_sum_l267_267818


namespace largest_divisor_of_product_of_five_consecutive_integers_l267_267495

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∀ (n : ℤ), ∃ k : ℤ, k = 60 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intro n
  use 60
  split
  { refl }
  { sorry }

end largest_divisor_of_product_of_five_consecutive_integers_l267_267495


namespace max_acute_triangles_l267_267322

theorem max_acute_triangles (n : ℕ) (hn : n ≥ 3) :
  (∃ k, k = if n % 2 = 0 then (n * (n-2) * (n+2)) / 24 else (n * (n-1) * (n+1)) / 24) :=
by 
  sorry

end max_acute_triangles_l267_267322


namespace domain_of_sqrt_log_function_l267_267383

def domain_of_function (x : ℝ) : Prop :=
  (1 ≤ x ∧ x < 2) ∨ (2 < x ∧ x < 3)

theorem domain_of_sqrt_log_function :
  ∀ x : ℝ, (x - 1 ≥ 0) → (x - 2 ≠ 0) → (-x^2 + 2 * x + 3 > 0) →
    domain_of_function x :=
by
  intros x h1 h2 h3
  unfold domain_of_function
  sorry

end domain_of_sqrt_log_function_l267_267383


namespace triangle_equilateral_l267_267308

theorem triangle_equilateral 
  (A B C D E F : Type)
  (AD : is_altitude A D B C)
  (BE : is_angle_bisector B E C A)
  (CF : is_median C F A B)
  (concurrent : concurrent AD BE CF) :
  is_equilateral_triangle A B C :=
sorry

end triangle_equilateral_l267_267308


namespace dice_faces_l267_267198

theorem dice_faces (n : ℕ) (h : (1 / (n : ℝ)) ^ 5 = 0.0007716049382716049) : n = 10 := sorry

end dice_faces_l267_267198


namespace ram_leela_piggy_bank_l267_267927

theorem ram_leela_piggy_bank (final_amount future_deposits weeks: ℕ) 
  (initial_deposit common_diff: ℕ) (total_deposits : ℕ) 
  (h_total : total_deposits = (weeks * (initial_deposit + (initial_deposit + (weeks - 1) * common_diff)) / 2)) 
  (h_final : final_amount = 1478) 
  (h_weeks : weeks = 52) 
  (h_future_deposits : future_deposits = total_deposits) 
  (h_initial_deposit : initial_deposit = 1) 
  (h_common_diff : common_diff = 1) 
  : final_amount - future_deposits = 100 :=
sorry

end ram_leela_piggy_bank_l267_267927


namespace area_of_triangle_sum_l267_267042

theorem area_of_triangle_sum (r : ℝ) (a b : ℕ) (h₁ : ∀ (ω₁ ω₂ ω₃ : ℝ), ω₁ = ω₂ ∧ ω₂ = ω₃ ∧ ω₁ = r) 
  (h₂ : ∀ (P₁ P₂ P₃ : ℝ), dist P₁ P₂ = dist P₂ P₃ ∧ dist P₂ P₃ = dist P₃ P₁) 
  (h₃ : P₁ P₂ P₃ : ℝ, dist P₁ P₂ = sqrt (r + r) ^ 2 - r^2) : a + b = 3024 :=
sorry

end area_of_triangle_sum_l267_267042


namespace dot_product_correct_l267_267259

-- Define the vectors as given conditions
def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (1, -2)

-- State the theorem to prove the dot product
theorem dot_product_correct : a.1 * b.1 + a.2 * b.2 = -4 := by
  -- Proof steps go here
  sorry

end dot_product_correct_l267_267259


namespace product_of_5_consecutive_integers_divisible_by_60_l267_267539

theorem product_of_5_consecutive_integers_divisible_by_60 :
  ∀a : ℤ, 60 ∣ (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
by
  sorry

end product_of_5_consecutive_integers_divisible_by_60_l267_267539


namespace find_total_salary_l267_267589

noncomputable def total_salary (salary_left : ℕ) : ℚ :=
  salary_left * (120 / 19)

theorem find_total_salary
  (food : ℚ) (house_rent : ℚ) (clothes : ℚ) (transport : ℚ) (remaining : ℕ) :
  food = 1 / 4 →
  house_rent = 1 / 8 →
  clothes = 3 / 10 →
  transport = 1 / 6 →
  remaining = 35000 →
  total_salary remaining = 210552.63 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end find_total_salary_l267_267589


namespace classify_conic_l267_267924

noncomputable def classify_conic_section (a b c d e f : ℝ) : Prop :=
let Q (x y : ℝ) := a * x ^ 2 + 2 * b * x * y + c * y ^ 2 in
ac - b^2 ≠ 0 → 
(∃ α β : ℝ, (α ≠ 0 ∧ β ≠ 0 ∧ 
((Q(x, y) + 2 * d * x + 2 * e * y = f) = (x^2 / α^2 + y^2 / β^2 = 1) ∨ 
(Q(x, y) + 2 * d * x + 2 * e * y = f) = (x^2 / α^2 - y^2 / β^2 = 1) ∨
(Q(x, y) + 2 * d * x + 2 * e * y = f) = (x^2 / α^2 = y^2 / β^2) ∨
(Q(x, y) + 2 * d * x + 2 * e * y = f) = single_point_or_empty_set)))

theorem classify_conic (a b c d e f : ℝ) (h : ac - b^2 ≠ 0) : 
classify_conic_section a b c d e f :=
sorry

end classify_conic_l267_267924


namespace largest_divisor_of_product_of_five_consecutive_integers_l267_267472

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∀ (n : ℤ), ∃ (d : ℤ), d = 60 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end largest_divisor_of_product_of_five_consecutive_integers_l267_267472


namespace geometric_sequence_sum_eight_l267_267833

noncomputable def sum_geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ :=
  a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_eight {a1 q : ℝ} (hq : q ≠ 1) 
  (h4 : sum_geometric_sequence a1 q 4 = -5) 
  (h6 : sum_geometric_sequence a1 q 6 = 21 * sum_geometric_sequence a1 q 2) : 
  sum_geometric_sequence a1 q 8 = -85 :=
sorry

end geometric_sequence_sum_eight_l267_267833


namespace measure_of_angle_GDA_l267_267779

-- Definitions for the angles
def angle_in_square : ℝ := 90
def angle_in_pentagon : ℝ := 108

-- Definition of the problem
theorem measure_of_angle_GDA : 
  let ADC := angle_in_square in
  let CDE := angle_in_pentagon in
  let FDG := angle_in_square in
  ADC + CDE + FDG = 288 →
  360 - (ADC + CDE + FDG) = 72 :=
begin
  intros ADC CDE FDG H,
  sorry
end

end measure_of_angle_GDA_l267_267779


namespace meeting_point_163_l267_267626

noncomputable def lamp_meeting_point (n : ℕ) (A_start : ℕ) (B_start : ℕ) (A_pos : ℕ) (B_pos : ℕ) : ℕ :=
  let A_distance := A_pos - A_start
  let B_distance := B_start - B_pos
  let total_intervals := n - 1
  let combined_distance := A_distance + B_distance
  if combined_distance = total_intervals / 3 then 
    A_start + 3 * A_distance
  else 
    0 -- We place 0 as a placeholder for an invalid state which is not supposed to occur in this scenario.

theorem meeting_point_163 :
  lamp_meeting_point 400 1 400 55 321 = 163 :=
by 
  simp [lamp_meeting_point]
  sorry

end meeting_point_163_l267_267626


namespace domain_of_h_l267_267669

noncomputable def h (x : ℝ) : ℝ :=
  (x^2 - 9) / (abs (x - 4) + x^2 - 1)

theorem domain_of_h :
  ∀ (x : ℝ), x ≠ (1 + Real.sqrt 13) / 2 → (abs (x - 4) + x^2 - 1) ≠ 0 :=
sorry

end domain_of_h_l267_267669


namespace interval_of_monotonic_increase_l267_267370

open Real

theorem interval_of_monotonic_increase
  (f : ℝ → ℝ) (ω : ℝ) (ϕ : ℝ) (k : ℤ)
  (hω : 0 < ω)
  (hϕ : -π / 2 ≤ ϕ ∧ ϕ < π / 2)
  (hx : f = fun x => sin(ω * x + ϕ))
  (stretch_factor : ℝ := 2)
  (shift_units : ℝ := 5 * π / 6) :
  ∃ (I : set ℝ), I = set.Icc (k * π - π / 12) (k * π + 5 * π / 12) ∧ 
                 ∀ x ∈ I, f x = cos x := 
begin
  sorry
end

end interval_of_monotonic_increase_l267_267370


namespace max_divisor_of_five_consecutive_integers_l267_267508

theorem max_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, 60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intros n
  sorry

end max_divisor_of_five_consecutive_integers_l267_267508


namespace problem_statement_l267_267905

open EuclideanGeometry

noncomputable def point (α : Type*) := point α
variables {V : Type*} [inner_product_space ℝ V] [euclidean_space V]

namespace my_problem

variables {A B C D M O K L N : point V}

def is_center_of_tetrahedron (O A B C D : point V) : Prop := sorry
def on_tetrahedron_face (M : point V) (A B C : point V) : Prop := sorry
def is_perpendicular_foot (M N : point V) (P Q R : point V) : Prop := sorry
def centroid (A B C : point V) : point V := sorry

theorem problem_statement (O A B C D M K L N : point V)
  (h1 : is_center_of_tetrahedron O A B C D)
  (h2 : on_tetrahedron_face M A B C ∨ on_tetrahedron_face M A B D ∨ on_tetrahedron_face M A C D ∨ on_tetrahedron_face M B C D)
  (h3 : is_perpendicular_foot M K A B C)
  (h4 : is_perpendicular_foot M L A C D)
  (h5 : is_perpendicular_foot M N B C D) :
  line_through O M ∩ plane K L N = centroid K L N :=
sorry

end my_problem

end problem_statement_l267_267905


namespace time_after_1450_minutes_l267_267428

theorem time_after_1450_minutes (initial_time_in_minutes : ℕ := 360) (minutes_to_add : ℕ := 1450) : 
  (initial_time_in_minutes + minutes_to_add) % (24 * 60) = 370 :=
by
  -- Given (initial_time_in_minutes = 360 which is 6:00 a.m., minutes_to_add = 1450)
  -- Compute the time in minutes after 1450 minutes
  -- 24 hours = 1440 minutes, so (360 + 1450) % 1440 should equal 370
  sorry

end time_after_1450_minutes_l267_267428


namespace exists_divisible_pair_l267_267902

theorem exists_divisible_pair (n : ℕ) (h : n ≥ 1) (S : Finset ℕ) (hS : S.card = n + 1) (h_range : ∀ x ∈ S, 1 ≤ x ∧ x ≤ 2 * n) :
  ∃ a b ∈ S, a ∣ b ∧ a ≠ b :=
by
  sorry

end exists_divisible_pair_l267_267902


namespace largest_divisor_of_five_consecutive_integers_l267_267455

open Nat

theorem largest_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, k ∈ {n, n+1, n+2, n+3, n+4} ∧
    ∀ m ∈ {2, 3, 4, 5}, m ∣ k → 60 ∣ (n * (n+1) * (n+2) * (n+3) * (n+4)) := 
sorry

end largest_divisor_of_five_consecutive_integers_l267_267455


namespace five_consecutive_product_div_24_l267_267494

theorem five_consecutive_product_div_24 (n : ℤ) : 
  24 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) := 
sorry

end five_consecutive_product_div_24_l267_267494


namespace five_consecutive_product_div_24_l267_267488

theorem five_consecutive_product_div_24 (n : ℤ) : 
  24 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) := 
sorry

end five_consecutive_product_div_24_l267_267488


namespace sum_of_eight_l267_267876

variable (a₁ : ℕ) (q : ℕ)
variable (S : ℕ → ℕ) -- Assume S is a function from natural numbers to natural numbers representing S_n

-- Condition 1: Sum of first 4 terms equals -5
axiom h1 : S 4 = -5

-- Condition 2: Sum of the first 6 terms is 21 times the sum of the first 2 terms
axiom h2 : S 6 = 21 * S 2

-- The formula for the sum of the first n terms of a geometric sequence
def sum_of_first_n_terms (n : ℕ) := a₁ * (1 - q^n) / (1 - q)

-- Statement to be proved
theorem sum_of_eight : S 8 = -85 := sorry

end sum_of_eight_l267_267876


namespace highest_power_of_8_dividing_15_factorial_l267_267743

/-- The highest power of 8 that divides 15! is 5. --/
theorem highest_power_of_8_dividing_15_factorial : 
  let n := 15! in ∃ k, n = 8^k * 15! ∧ k = 5 := 
by
  sorry

end highest_power_of_8_dividing_15_factorial_l267_267743


namespace range_of_a_l267_267200

noncomputable def h (a x : ℝ) := 2^(3*x) - (log a x + 1)

theorem range_of_a
  (a : ℝ)
  (cond : ∀ x : ℝ, 0 < x ∧ x < 1/3 → h(a, x) ≤ 0) :
  1/3 ≤ a ∧ a < 1 :=
sorry

end range_of_a_l267_267200


namespace cost_difference_proof_l267_267201

-- Define the cost per copy at print shop X
def cost_per_copy_X : ℝ := 1.25

-- Define the cost per copy at print shop Y
def cost_per_copy_Y : ℝ := 2.75

-- Define the number of copies
def number_of_copies : ℝ := 60

-- Define the total cost at print shop X
def total_cost_X : ℝ := cost_per_copy_X * number_of_copies

-- Define the total cost at print shop Y
def total_cost_Y : ℝ := cost_per_copy_Y * number_of_copies

-- Define the difference in cost between print shop Y and print shop X
def cost_difference : ℝ := total_cost_Y - total_cost_X

-- The theorem statement proving the cost difference is $90
theorem cost_difference_proof : cost_difference = 90 := by
  sorry

end cost_difference_proof_l267_267201


namespace curlers_count_l267_267178

theorem curlers_count (T P B G : ℕ) 
  (hT : T = 16)
  (hP : P = T / 4)
  (hB : B = 2 * P)
  (hG : G = T - (P + B)) : 
  G = 4 :=
by
  sorry

end curlers_count_l267_267178


namespace decimal_ternary_divisibility_l267_267269

theorem decimal_ternary_divisibility
  (n : ℕ)
  (a b l : ℕ)
  (digits_dec : List ℕ) (digits_tern : List ℕ)
  (Hdec : digits_dec = [a, b, ..., l])
  (Htern : digits_tern = [a, b, ..., l])
  (Hlen : digits_dec.length = n)
  (Hdiv_ternary : (List.sum (List.map (λ (i : ℕ) => digits_tern.get! i * 3^(n-1-i)) (List.range n))) % 7 = 0) :
  (List.sum (List.map (λ (i : ℕ) => digits_dec.get! i * 10^(n-1-i)) (List.range n))) % 7 = 0 := by
  sorry

end decimal_ternary_divisibility_l267_267269


namespace equilateral_triangles_congruent_l267_267557

theorem equilateral_triangles_congruent
  (T1 T2 : Triangle) 
  (h1 : T1.is_equilateral) 
  (h2 : T2.is_equilateral)
  (h3 : T1.side_lengths = T2.side_lengths) :
  T1 ≅ T2 :=
sorry

end equilateral_triangles_congruent_l267_267557


namespace dave_files_count_l267_267657

theorem dave_files_count :
  ∀ (a f a' F : ℕ), a = 15 ∧ f = 24 ∧ a' = 21 ∧ a' = F + 17 → F = 4 :=
by
  intros a f a' F
  intro h
  cases h with h1 h
  cases h with h2 h
  cases h with h3 h4
  unfold at h1 h2 h3 h4
  sorry

end dave_files_count_l267_267657


namespace largest_divisor_of_5_consecutive_integers_l267_267484

theorem largest_divisor_of_5_consecutive_integers :
  ∀ (a b c d e : ℤ), 
    a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e →
    (∃ k : ℤ, k ∣ (a * b * c * d * e) ∧ k = 60) :=
by 
  intro a b c d e h
  sorry

end largest_divisor_of_5_consecutive_integers_l267_267484


namespace parisians_hair_count_l267_267021

theorem parisians_hair_count :
  ∀ (n : ℕ), n > 2000000 →
  ∀ (m : ℕ), m = 150000 →
  ∃ k : ℕ, k ≥ 14 ∧ ∃ (f : fin n → fin (m + 1)), 
    ∃ a b : fin n, a ≠ b ∧ f a = f b := 
by
  intros n hn m hm
  have : 2000000 > 150000,
    by norm_num,
  sorry

end parisians_hair_count_l267_267021


namespace new_ticket_price_l267_267987

theorem new_ticket_price (a : ℕ) (x : ℝ) (initial_price : ℝ) (revenue_increase : ℝ) (spectator_increase : ℝ)
  (h₀ : initial_price = 25)
  (h₁ : spectator_increase = 1.5)
  (h₂ : revenue_increase = 1.14)
  (h₃ : x = 0.76):
  initial_price * x = 19 :=
by
  sorry

end new_ticket_price_l267_267987


namespace pyramid_base_is_octagon_l267_267381
-- Import necessary library

-- Declare the problem
theorem pyramid_base_is_octagon (A : Nat) (h : A = 8) : A = 8 :=
by
  -- Proof goes here
  sorry

end pyramid_base_is_octagon_l267_267381


namespace remaining_quadrilateral_perimeter_l267_267113

theorem remaining_quadrilateral_perimeter :
  ∀ (ABC : Type) (DBE : Type)
  [is_right_angled_isosceles_triangle DBE 1]
  [is_equilateral_triangle ABC 4],
  ∃ (D E A C : Point) [In ΔDBE D] [In ΔDBE E] [In ΔABC A] [In ΔABC C],
  D ≠ E ∧ A ≠ C ∧ D ≠ A ∧ E ≠ C →
  (side_length AC = 4 ∧ side_length CE = 3 ∧ side_length ED = √2 ∧ side_length DA = 3)
  → perimeter A C E D = 10 + √2 :=
sorry

end remaining_quadrilateral_perimeter_l267_267113


namespace unique_function_satisfies_equation_l267_267684

theorem unique_function_satisfies_equation :
  (∃! (f : ℝ → ℝ), ∀ x y : ℝ, f(x + f(y) + 1) = x + y + 1) :=
sorry

end unique_function_satisfies_equation_l267_267684


namespace scientific_notation_of_448000_l267_267980

theorem scientific_notation_of_448000 :
  448000 = 4.48 * 10^5 :=
by 
  sorry

end scientific_notation_of_448000_l267_267980


namespace part_a_part_b_l267_267808

noncomputable def f (A : set ℝ) (x : ℝ) (hx : x ∈ A) : ℝ := sorry

def A : set ℝ := {x : ℝ | 0 ≤ x ∧ x < 1}

lemma f_properties (x : ℝ) (hx : x ∈ A) :
  f A x hx = 2 * f A (x / 2) (by linarith) :=
sorry

lemma f_properties_half (x : ℝ) (hx : x ∈ A) (h : 1 / 2 ≤ x) :
  f A x hx = 1 - f A (x - 1 / 2) (by linarith) :=
sorry

theorem part_a (x : ℝ) (hx1 : x ∈ A) (hx2 : x ∈ ℚ) (hx3 : 0 < x) :
  f A x hx1 + f A (1 - x) (by linarith) ≥ 2 / 3 :=
sorry

theorem part_b : ∃ᶠ (q : ℕ) (hq : odd q), f A (1 / q) (by linarith) + f A (1 - 1 / q) (by linarith) = 2 / 3 :=
sorry

end part_a_part_b_l267_267808


namespace phase_shift_sin_cos_eq_l267_267046

noncomputable def phase_shift (x : ℝ) : ℝ := x - (π / 12)

theorem phase_shift_sin_cos_eq :
  ∀ x, cos (9 * phase_shift x) = sin (9 * x - (π / 4)) :=
by
  intro x
  sorry

end phase_shift_sin_cos_eq_l267_267046


namespace geometric_sequence_S8_l267_267872

theorem geometric_sequence_S8 
  (a1 q : ℝ) (S : ℕ → ℝ)
  (h1 : S 4 = a1 * (1 - q^4) / (1 - q) = -5)
  (h2 : S 6 = 21 * S 2)
  (geom_sum : ∀ n, S n = a1 * (1 - q^n) / (1 - q))
  (h3 : q ≠ 1) : S 8 = -85 := 
begin
  sorry
end

end geometric_sequence_S8_l267_267872


namespace number_of_birds_is_400_l267_267401

-- Definitions of the problem
def num_stones : ℕ := 40
def num_trees : ℕ := 3 * num_stones + num_stones
def combined_trees_stones : ℕ := num_trees + num_stones
def num_birds : ℕ := 2 * combined_trees_stones

-- Statement to prove
theorem number_of_birds_is_400 : num_birds = 400 := by
  sorry

end number_of_birds_is_400_l267_267401


namespace toothpaste_length_l267_267581

/-- The length of toothpaste squeezed out from a tube, given that it forms a cylinder with 
a diameter of 6 mm and a volume of 75 ml, is approximately 2.653 meters. -/
theorem toothpaste_length (V : ℝ) (d : ℝ) (h : ℝ) (π : ℝ) 
  (V_eq : V = 75) (d_eq : d = 0.6) (pi_def : π = Real.pi) : 
  h = 75 / (π * (d / 2)^2) :=
by
  -- Given conditions
  have r : ℝ := d / 2 
  have V_cm3 : ℝ := V  -- since 1 ml = 1 cm^3
  -- Volume of cylinder formula
  have calc_height : h = V / (π * r^2) := by 
    rw [← V_eq, ← d_eq, ← pi_def]
    sorry 

lemma toothpaste_length_in_meters : 
  (toothpaste_length 75 0.6 (75 / (Real.pi * (0.6 / 2)^2)) Real.pi / 100) ≈ 2.653 :=
by simp ; sorry

end toothpaste_length_l267_267581


namespace geometric_sequence_sum_eight_l267_267830

noncomputable def sum_geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ :=
  a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_eight {a1 q : ℝ} (hq : q ≠ 1) 
  (h4 : sum_geometric_sequence a1 q 4 = -5) 
  (h6 : sum_geometric_sequence a1 q 6 = 21 * sum_geometric_sequence a1 q 2) : 
  sum_geometric_sequence a1 q 8 = -85 :=
sorry

end geometric_sequence_sum_eight_l267_267830


namespace product_of_5_consecutive_integers_divisible_by_60_l267_267543

theorem product_of_5_consecutive_integers_divisible_by_60 :
  ∀a : ℤ, 60 ∣ (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
by
  sorry

end product_of_5_consecutive_integers_divisible_by_60_l267_267543


namespace polynomial_identity_l267_267265

theorem polynomial_identity :
  (∀ x : ℝ, (2*x + real.sqrt 3)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 :=
by {
  sorry
}

end polynomial_identity_l267_267265


namespace pentagon_AE_length_l267_267300

/-- 
In pentagon ABCDE, BC = CD = DE = 1 unit.
Angle E is a right angle.
m∠B = m∠D = 120°, m∠C = 120°.
Compute the length of segment AE, given it can be expressed in simplest form as a + sqrt(b) units.
Prove that a + b = 4.
--/
theorem pentagon_AE_length (BC CD DE : ℝ) (angleE : ℝ) (angleB : ℝ) (angleD : ℝ) (angleC : ℝ) 
    (hBC : BC = 1) (hCD : CD = 1) (hDE : DE = 1) (hangleE : angleE = π / 2) (hangleB : angleB = 2 * π / 3) 
    (hangleD : angleD = 2 * π / 3) (hangleC : angleC = 2 * π / 3) : 
    ∃ a b : ℝ, (AE = a + sqrt b) ∧ (a + b = 4) :=
begin
  sorry
end

end pentagon_AE_length_l267_267300


namespace largest_divisor_of_5_consecutive_integers_l267_267526

theorem largest_divisor_of_5_consecutive_integers :
  ∀ (n : ℤ), ∃ d, d = 120 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intro n
  use 120
  split
  exact rfl
  sorry

end largest_divisor_of_5_consecutive_integers_l267_267526


namespace number_of_toys_sold_l267_267103

theorem number_of_toys_sold (n : ℕ) 
  (sell_price : ℕ) (gain_price : ℕ) (cost_price_per_toy : ℕ) :
  sell_price = 27300 → 
  gain_price = 3 * cost_price_per_toy → 
  cost_price_per_toy = 1300 →
  n * cost_price_per_toy + gain_price = sell_price → 
  n = 18 :=
by sorry

end number_of_toys_sold_l267_267103


namespace raisins_not_all_eaten_l267_267738

-- Define the problem in Lean 4 statements

theorem raisins_not_all_eaten (n : ℕ) (a : Fin n → ℤ)
  (h_odd : 2011 % 2 = 1)
  (h_total : finset.univ.sum a = 2011)
  (h_property : ∀ i : Fin n, (a i = 2 * a ((i + 1) % n)) ∨ (a i = a ((i + 1) % n) - 6)) :
  False :=
by sorry

end raisins_not_all_eaten_l267_267738


namespace rectangle_perimeter_from_square_l267_267915

theorem rectangle_perimeter_from_square (d : ℝ)
  (h : d = 6) :
  ∃ (p : ℝ), p = 12 :=
by
  sorry

end rectangle_perimeter_from_square_l267_267915


namespace Jolene_raised_total_money_l267_267805

-- Definitions for the conditions
def babysits_earning_per_family : ℤ := 30
def number_of_families : ℤ := 4
def cars_earning_per_car : ℤ := 12
def number_of_cars : ℤ := 5

-- Calculation of total earnings
def babysitting_earnings : ℤ := babysits_earning_per_family * number_of_families
def car_washing_earnings : ℤ := cars_earning_per_car * number_of_cars
def total_earnings : ℤ := babysitting_earnings + car_washing_earnings

-- The proof statement
theorem Jolene_raised_total_money : total_earnings = 180 := by
  sorry

end Jolene_raised_total_money_l267_267805


namespace max_log_sum_l267_267267

noncomputable def log (x : ℝ) : ℝ := Real.log x

theorem max_log_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 4) : 
  ∃ L, (∀ x y, x > 0 → y > 0 → x + y = 4 → log x + log y ≤ L) ∧ L = log 4 :=
by
  sorry

end max_log_sum_l267_267267


namespace sqrt_of_2x_y_z_l267_267719

variables (x y z : ℝ)
-- Definitions from the conditions
def condition1 := sqrt (2 * x + 1) = 0
def condition2 := sqrt y = 4
def condition3 := z = -3

-- Statement to be proved
theorem sqrt_of_2x_y_z (h1 : condition1 x) (h2 : condition2 y) (h3 : condition3 z) :
  sqrt (2 * x + y + z) = 2 * sqrt 3 ∨ sqrt (2 * x + y + z) = -2 * sqrt 3 :=
by
  sorry

end sqrt_of_2x_y_z_l267_267719


namespace total_gain_correct_l267_267806

noncomputable def total_gain (X T : ℝ) (krishan_investment_ratio nandan_time_ratio : ℝ) (nandan_gain : ℝ) : ℝ :=
let krishan_investment := krishan_investment_ratio * X in
let krishan_time := nandan_time_ratio * T in
let nandan_gain_proportional := X * T in
let krishan_gain_proportional := krishan_investment * krishan_time in
let total_gain_proportional := nandan_gain_proportional + krishan_gain_proportional in
(nandan_gain / nandan_gain_proportional) * total_gain_proportional

theorem total_gain_correct
  (X T : ℝ)
  (krishan_investment_ratio : ℝ := 6)
  (nandan_time_ratio : ℝ := 2)
  (nandan_gain : ℝ := 6000) :
  total_gain X T krishan_investment_ratio nandan_time_ratio nandan_gain = 78000 :=
by
  sorry

end total_gain_correct_l267_267806


namespace cylinder_new_volume_l267_267407

-- Definitions based on conditions
def original_volume_r_h (π R H : ℝ) : ℝ := π * R^2 * H

def new_volume (π R H : ℝ) : ℝ := π * (3 * R)^2 * (2 * H)

theorem cylinder_new_volume (π R H : ℝ) (h_original_volume : original_volume_r_h π R H = 15) :
  new_volume π R H = 270 :=
by sorry

end cylinder_new_volume_l267_267407


namespace cost_of_fence_l267_267566

theorem cost_of_fence (area : ℝ) (price_per_foot : ℝ) (s : ℝ) (perimeter : ℝ) (cost : ℝ) 
(h1 : area = 289)
(h2 : price_per_foot = 54)
(h3 : s = real.sqrt area)
(h4 : perimeter = 4 * s)
(h5 : cost = perimeter * price_per_foot) : 
  cost = 3672 :=
by
  sorry

end cost_of_fence_l267_267566


namespace max_knights_in_grid_l267_267289

inductive Person
| knight
| liar

def adjacent (i j : ℕ × ℕ) : ℕ × ℕ → Prop
| (i, j) (i, j') := j ≠ j' ∧ j = j'
| (i, j) (i', j) := i ≠ i' ∧ i = i'
| _ _            := false

def valid_configuration (grid : ℕ → ℕ → Person) : Prop :=
  ∀ i j, (0 ≤ i ∧ i < 3) ∧ (0 ≤ j ∧ j < 3) →
  (grid i j = (Person.knight)) →
  (∃ n, adjacent (i, j) n ∧ grid n = (Person.liar))

theorem max_knights_in_grid : 
  ∃ (grid : ℕ → ℕ → Person), valid_configuration grid ∧ 
  (Σ' (i j : ℕ), grid i j = Person.knight).card = 6 :=
sorry

end max_knights_in_grid_l267_267289


namespace ratio_of_earnings_l267_267796

theorem ratio_of_earnings (jacob_hourly: ℕ) (jake_total: ℕ) (days: ℕ) (hours_per_day: ℕ) (jake_hourly: ℕ) (ratio: ℕ) 
  (h_jacob: jacob_hourly = 6)
  (h_jake_total: jake_total = 720)
  (h_days: days = 5)
  (h_hours_per_day: hours_per_day = 8)
  (h_jake_hourly: jake_hourly = jake_total / (days * hours_per_day))
  (h_ratio: ratio = jake_hourly / jacob_hourly) :
  ratio = 3 := 
sorry

end ratio_of_earnings_l267_267796


namespace conjugate_of_square_complex_l267_267958

theorem conjugate_of_square_complex (z : ℂ) (hz : z = 1 + 2 * complex.I) : complex.conj (z ^ 2) = -3 - 4 * complex.I :=
by
  rw [hz, ←complex.mul_re, complex.conj, complex.add_re, complex.mul_I_re, complex.pow_two_real, complex.mul_re, complex.I_mul_I]
  sorry

end conjugate_of_square_complex_l267_267958


namespace factorization_of_x10_minus_1024_l267_267155

theorem factorization_of_x10_minus_1024 (x : ℝ) :
  x^10 - 1024 = (x^5 + 32) * (x - 2) * (x^4 + 2 * x^3 + 4 * x^2 + 8 * x + 16) :=
by sorry

end factorization_of_x10_minus_1024_l267_267155


namespace sum_of_series_l267_267898

noncomputable def geometric_series_sum : ℝ := ∑' n, (2 / 5) ^ n

theorem sum_of_series : geometric_series_sum = 5 / 3 := by
  -- We define our series as an infinite sum
  have h1 : geometric_series_sum = ∑' n, (2 / 5) ^ n := rfl
  
  -- By the formula for the sum of an infinite geometric series with |r| < 1
  have h2 : ∑' n, (2 / 5) ^ n = 1 / (1 - (2 / 5)) := by
    apply tsum_geometric_of_norm_lt_1
    norm_num

  -- Simplifying the sum
  have h3 : 1 / (1 - (2 / 5)) = 5 / 3 := by
    field_simp [ne_of_gt (by norm_num : (3:ℝ) > 0)]
    norm_num

  -- Combining the steps
  rw [h1, h2, h3]
  rfl

end sum_of_series_l267_267898


namespace eight_term_sum_l267_267820

variable {α : Type*} [Field α]
variable (a q : α)

-- Define the n-th sum of the geometric sequence
def S_n (n : ℕ) : α := a * (1 - q ^ n) / (1 - q)

-- Given conditions
def S4 : α := S_n 4 = -5
def S6 : α := S_n 6 = 21 * S_n 2

-- Prove the target statement
theorem eight_term_sum : S_n 8 = -85 :=
  sorry

end eight_term_sum_l267_267820


namespace line_and_x_intercept_l267_267258

noncomputable def T : Type := tuple real

def A : T := (-2, -3)
def B : T := (3, 0)

def midpoint (P Q : T) : T :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

def slope (P Q : T) : real :=
  (Q.2 - P.2) / (Q.1 - P.1)

def eqn_line (C : T) (k : real) (x y : real) : Prop :=
  y = k * (x - C.1) + C.2

def perp_slope (k : real) : real :=
  -1 / k

theorem line_and_x_intercept :
  let C := midpoint A B,
      k_AB := slope A B,
      k_l := perp_slope k_AB,
      line_eq := eqn_line C k_l,
      x_int := - (2 / 5)
  in (∃ l : real × real → Prop, ∀ (x y : real), l (x, y) ↔ 5 * x + 3 * y + 2 = 0) ∧ 
      (∃ x, 5 * x + 2 = 0 ∧ x = - (2 / 5)) :=
begin
  sorry
end

end line_and_x_intercept_l267_267258


namespace solution_l267_267151

noncomputable def expr := (sqrt 4 + cbrt (-8) - (sqrt 6 - sqrt 24) / sqrt 2)
#reduce expr

theorem solution : expr = sqrt 3 := by {
  have h1 : sqrt 4 = 2 := by norm_num,
  have h2 : cbrt (-8) = -2 := by norm_num,
  have h3 : sqrt 24 = 2 * sqrt 6 := by {
    rw [sqrt_mul, sqrt_eq_rfl], norm_num, simp,
  },
  have h4 : (sqrt 6 - sqrt 24) / sqrt 2 = - sqrt 3 := by {
    rw [h3, sub_div, div_sqrt, sub_sqrt_self], norm_num,
  },
  calc
    expr = (2 + (-2) - (- sqrt 3)) : by rw [h1, h2, h4]
    ... = sqrt 3 : by norm_num,
}

end solution_l267_267151


namespace product_of_five_consecutive_integers_divisible_by_120_l267_267445

theorem product_of_five_consecutive_integers_divisible_by_120 (n : ℤ) : 
  120 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end product_of_five_consecutive_integers_divisible_by_120_l267_267445


namespace integer_values_in_range_l267_267413

theorem integer_values_in_range :
  {x : ℤ | 5 < real.sqrt x ∧ real.sqrt x < 6}.finite.to_finset.card = 10 :=
by
  sorry

end integer_values_in_range_l267_267413


namespace books_problem_l267_267443

theorem books_problem
  (M H : ℕ)
  (h1 : M + H = 80)
  (h2 : 4 * M + 5 * H = 390) :
  M = 10 :=
by
  sorry

end books_problem_l267_267443


namespace reflection_of_K_lies_on_BC_l267_267320

open EuclideanGeometry

-- Define the main theorem
theorem reflection_of_K_lies_on_BC
    (ABC : Triangle)
    (acute : ABC.is_acute)
    (non_isosceles : ¬ ABC.is_isosceles)
    (H : Point)
    (H_is_orthocenter : ABC.is_orthocenter H)
    (O : Point)
    (O_is_circumcenter : ABC.is_circumcenter O)
    (K : Point)
    (K_is_circumcenter_AHO : ABC.is_circumcenter_of_AHO K) :
    ∃ K', K'.is_reflection_wrt_OH K ∧ K'.lies_on_BC :=
by
  sorry

end reflection_of_K_lies_on_BC_l267_267320


namespace sphere_surface_area_l267_267340

-- Define scope of problem
variables (l : ℝ) (O : Type) (P : O) [has_dist O]

-- Conditions
def common_point (P : O) : Prop := dist l P = 0 

def semiplane_alpha : Prop := ∃ α, α.dist O = 1
def semiplane_beta : Prop := ∃ β, β.dist O = sqrt(3)

def dihedral_angle : Prop := ∃ θ, θ = (5 * π) / 6

-- Define radius and surface area of sphere O
noncomputable def radius_square (R : ℝ) : Prop := R^2 = 28
noncomputable def surface_area (R : ℝ) : ℝ := 4 * π * R^2

-- Concluding the proof
theorem sphere_surface_area (R : ℝ) : common_point P ∧ semiplane_alpha ∧ semiplane_beta ∧ dihedral_angle → surface_area R = 112 * π :=
sorry

end sphere_surface_area_l267_267340


namespace largest_divisor_of_product_of_five_consecutive_integers_l267_267471

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∀ (n : ℤ), ∃ (d : ℤ), d = 60 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end largest_divisor_of_product_of_five_consecutive_integers_l267_267471


namespace barrels_of_pitch_needed_l267_267116

-- Define the basic properties and conditions
def total_length_road := 16
def truckloads_per_mile := 3
def bags_of_gravel_per_truckload := 2
def gravel_to_pitch_ratio := 5
def miles_paved_first_day := 4
def miles_paved_second_day := 2 * miles_paved_first_day - 1
def miles_already_paved := miles_paved_first_day + miles_paved_second_day
def remaining_miles := total_length_road - miles_already_paved
def total_truckloads := truckloads_per_mile * remaining_miles
def total_bags_of_gravel := bags_of_gravel_per_truckload * total_truckloads
def barrels_of_pitch := total_bags_of_gravel / gravel_to_pitch_ratio

-- State the theorem to prove the number of barrels of pitch needed
theorem barrels_of_pitch_needed :
    barrels_of_pitch = 6 :=
by
    sorry

end barrels_of_pitch_needed_l267_267116


namespace trapezoid_area_l267_267404

-- Definitions based on the problem conditions
def Vertex := (Real × Real)

structure Triangle :=
(A : Vertex)
(B : Vertex)
(C : Vertex)
(area : Real)

structure Trapezoid :=
(AB : Real)
(CD : Real)
(M : Vertex)
(area_triangle_ABM : Real)
(area_triangle_CDM : Real)

-- The main theorem we want to prove
theorem trapezoid_area (T : Trapezoid)
  (parallel_sides : T.AB < T.CD)
  (intersect_at_M : ∃ M : Vertex, M = T.M)
  (area_ABM : T.area_triangle_ABM = 2)
  (area_CDM : T.area_triangle_CDM = 8) :
  T.AB * T.CD / (T.CD - T.AB) + T.CD * T.AB / (T.CD - T.AB) = 18 :=
sorry

end trapezoid_area_l267_267404


namespace max_divisor_of_five_consecutive_integers_l267_267513

theorem max_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, 60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intros n
  sorry

end max_divisor_of_five_consecutive_integers_l267_267513


namespace complex_division_proof_l267_267651

def question_and_conditions : Prop :=
  let i := Complex.I in
  (1 - i) / (1 + i) = -i

theorem complex_division_proof : question_and_conditions :=
by
  -- Proof will be here
  sorry

end complex_division_proof_l267_267651


namespace number_of_senior_managers_selected_l267_267090

-- Define the given conditions
def total_senior_managers : ℕ := 10
def total_people : ℕ := 200
def selected_people : ℕ := 40

-- Define the function for stratified sampling
def sampling_fraction : ℚ := selected_people / total_people
def selected_senior_managers : ℚ := sampling_fraction * total_senior_managers

-- The theorem to prove
theorem number_of_senior_managers_selected :
  selected_senior_managers = 2 :=
begin
  -- Proof will be added here.
  sorry
end

end number_of_senior_managers_selected_l267_267090


namespace completing_square_solution_l267_267011

theorem completing_square_solution (x : ℝ) : x^2 + 4 * x - 1 = 0 → (x + 2)^2 = 5 :=
by
  sorry

end completing_square_solution_l267_267011


namespace sequence_general_term_l267_267782

theorem sequence_general_term (a : ℕ → ℝ) (h₁ : a 1 = 10) (h₂ : ∀ n, a (n + 1) = 2 * real.sqrt (a n)) :
  ∀ n, a n = 4 * (5 / 2) ^ (2 ^ (1 - n)) :=
by
  sorry

end sequence_general_term_l267_267782


namespace max_value_of_function_cos_sin_l267_267965

noncomputable def max_value_function (x : ℝ) : ℝ := 
  (Real.cos x)^3 + (Real.sin x)^2 - Real.cos x

theorem max_value_of_function_cos_sin : 
  ∃ x ∈ (Set.univ : Set ℝ), max_value_function x = (32 / 27) := 
sorry

end max_value_of_function_cos_sin_l267_267965


namespace event_occurs_eventually_l267_267752

open ProbabilityTheory

theorem event_occurs_eventually (ε : ℝ) (hε_positive : 0 < ε) (hε_less_one : ε < 1) : 
  tendsto (λ n, 1 - (1 - ε)^n) atTop (𝓝 1) := 
begin
  sorry,
end

end event_occurs_eventually_l267_267752


namespace l_shaped_structure_surface_area_l267_267680

-- Definitions derived from the conditions
def unit_cube := ℕ -- Representing the concept of unit cubes in natural number space

def bottom_layer_surface (n : unit_cube) : ℕ :=
  let side_length := 3 in -- Given conditions: 3x3 square
  2 * (side_length * side_length) -- Top and sides, bottom is not counted

def vertical_stack_surface (n : unit_cube) : ℕ :=
  let height := 6 in -- Given conditions: vertical stack of 6 cubes
  let sidelength := 1 in
  4 * (sidelength * height) -- Four sides of the vertical stack

def total_surface_area : ℕ :=
  bottom_layer_surface 9 + vertical_stack_surface 6 - 2 * 1 -- Subtract the area of the shared face

-- The proof problem statement
theorem l_shaped_structure_surface_area : total_surface_area = 29 := 
  sorry

end l_shaped_structure_surface_area_l267_267680


namespace value_of_x_l267_267948

theorem value_of_x : 
  let x := (sqrt (7^2 + 24^2)) / (sqrt (49 + 16)) 
  in x = 25 * sqrt 65 / 65 
  := 
  sorry

end value_of_x_l267_267948


namespace problem1_problem2_l267_267152

theorem problem1 : 1 - 2 + 3 + (-4) = -2 :=
sorry

theorem problem2 : (-6) / 3 - (-10) - abs (-8) = 0 :=
sorry

end problem1_problem2_l267_267152


namespace constant_term_expansion_l267_267204

-- Define the binomial expression
def binomial_exp : ℕ → (ℕ → Int) :=
  λ n k, (-1) ^ k * (Nat.choose n k)

-- Define the function to find the term in binomial expansion
def binomial_term (n k : ℕ) : Int × ℤ :=
  (binomial_exp 6 k, 6 - 2 * k)

-- Define the theorem to be proved
theorem constant_term_expansion (k : ℕ) :
  (λ (exp_val : Int × ℤ), 
     if exp_val.2 = 0 then exp_val.1 else 0) (binomial_term 6 k) = -20 :=
sorry

end constant_term_expansion_l267_267204


namespace smallest_n_square_area_l267_267624

theorem smallest_n_square_area (n : ℕ) (n_positive : 0 < n) : ∃ k : ℕ, 14 * n = k^2 ↔ n = 14 := 
sorry

end smallest_n_square_area_l267_267624


namespace custom_dollar_five_neg3_l267_267658

-- Define the custom operation
def custom_dollar (a b : Int) : Int :=
  a * (b - 1) + a * b

-- State the theorem
theorem custom_dollar_five_neg3 : custom_dollar 5 (-3) = -35 := by
  sorry

end custom_dollar_five_neg3_l267_267658


namespace intersection_inverse_dist_sum_l267_267302

noncomputable def inclination_angle : ℝ := Real.pi / 4

noncomputable def M : ℝ × ℝ := (2, 1)

noncomputable def polar_eq_circle (θ : ℝ) : ℝ := 4 * Real.sqrt 2 * Real.sin(θ + Real.pi / 4)

noncomputable def cartesian_eq_line (t : ℝ) : ℝ × ℝ :=
  (2 + Real.sqrt 2 / 2 * t, 1 + Real.sqrt 2 / 2 * t)

noncomputable def cartesian_eq_circle (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y = 0

theorem intersection_inverse_dist_sum :
  ∃ A B : ℝ × ℝ, cartesian_eq_circle A.fst A.snd ∧ cartesian_eq_circle B.fst B.snd ∧
  let MA := Real.sqrt ((A.fst - M.fst)^2 + (A.snd - M.snd)^2) in
  let MB := Real.sqrt ((B.fst - M.fst)^2 + (B.snd - M.snd)^2) in
  MA ≠ 0 ∧ MB ≠ 0 ∧ (1 / MA + 1 / MB) = Real.sqrt 30 / 7 :=
sorry

end intersection_inverse_dist_sum_l267_267302


namespace danica_car_arrangement_l267_267170

theorem danica_car_arrangement :
  ∃ (k : ℕ), 37 + k ≡ 0 [MOD 9] ∧ ∀ m : ℕ, (37 + m ≡ 0 [MOD 9] → k ≤ m) :=
begin
  use 8,
  split,
  { norm_num, },
  { intros m hm,
    norm_num at hm,
    exact le_antisymm (nat.le_of_dvd (nat.sub_pos_of_lt (by norm_num)) hm) (by norm_num) },
end

end danica_car_arrangement_l267_267170


namespace largest_less_than_07_l267_267128

theorem largest_less_than_07 :
  ∀ (a b c d : ℝ), a = 0.8 → b = 1/2 → c = 0.9 → d = 1/3 →
    (∀ x, (x = a ∨ x = b ∨ x = c ∨ x = d) → x ≤ 0.7 → x ≤ b) :=
by
  intros a b c d ha hb hc hd h x h₁ h₂
  rw [ha, hb, hc, hd] at *
  dsimp at h₂ ⊢
  by_cases h : x ≤ 0.7;
  { exact le_trans h h₂
  }; exact ⟨⟩

end largest_less_than_07_l267_267128


namespace solve_for_x_l267_267003

theorem solve_for_x : ∃ x : ℚ, (2/3 - 1/4) = 1/x ∧ x = 12/5 :=
by
  use 12/5
  split
  · norm_num
  · norm_num
  · sorry

end solve_for_x_l267_267003


namespace pickle_to_tomato_ratio_l267_267137

theorem pickle_to_tomato_ratio 
  (mushrooms : ℕ) 
  (cherry_tomatoes : ℕ) 
  (pickles : ℕ) 
  (bacon_bits : ℕ) 
  (red_bacon_bits : ℕ) 
  (h1 : mushrooms = 3) 
  (h2 : cherry_tomatoes = 2 * mushrooms)
  (h3 : red_bacon_bits = 32)
  (h4 : bacon_bits = 3 * red_bacon_bits)
  (h5 : bacon_bits = 4 * pickles) : 
  pickles/cherry_tomatoes = 4 :=
by
  sorry

end pickle_to_tomato_ratio_l267_267137


namespace area_triangle_ABC_l267_267760

variables (A B C D E F : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F]
variables (triangle_ABC : Triangle A B C) (triangle_DEF : Triangle D E F)

noncomputable def is_midpoint (D : Type) (B C : Type) : Prop :=
  (distance D B) = (distance D C)

noncomputable def ratio (P Q : Type) (r : ℚ) : Prop :=
  P / Q = r

noncomputable def area_of_triangle (triangle : Triangle A B C) : ℝ :=
  sorry

noncomputable def area (T : Triangle) : ℝ :=
  sorry

theorem area_triangle_ABC :
  is_midpoint D B C ∧ 
  ratio (distance A E) (distance E C) = 2 / 1 ∧
  ratio (distance A F) (distance F D) = 1 / 3 ∧ 
  area triangle_DEF = 23 →
  area triangle_ABC = 92 := 
sorry

end area_triangle_ABC_l267_267760


namespace cylinder_volume_ratio_l267_267573

noncomputable def volume_ratio (h1 h2 : ℝ) (c1 c2 : ℝ) : ℝ :=
  let r1 := c1 / (2 * Real.pi)
  let r2 := c2 / (2 * Real.pi)
  let V1 := Real.pi * r1^2 * h1
  let V2 := Real.pi * r2^2 * h2
  if V1 > V2 then V1 / V2 else V2 / V1

theorem cylinder_volume_ratio :
  volume_ratio 7 6 6 7 = 7 / 4 :=
by
  sorry

end cylinder_volume_ratio_l267_267573


namespace solve_for_x_l267_267009

theorem solve_for_x:
  ∃ x : ℚ, (2 / 3 - 1 / 4 = 1 / x) ∧ (x = 12 / 5) := by
  sorry

end solve_for_x_l267_267009


namespace find_y_of_x_pow_l267_267268

theorem find_y_of_x_pow (x y : ℝ) (h1 : x = 2) (h2 : x^(3*y - 1) = 8) : y = 4 / 3 :=
by
  -- skipping proof
  sorry

end find_y_of_x_pow_l267_267268


namespace geometric_seq_sum_l267_267840

-- Definitions of the conditions
variables {a₁ q : ℚ}
def S (n : ℕ) := a₁ * (1 - q^n) / (1 - q)

-- Hypotheses from the conditions
theorem geometric_seq_sum :
  (S 4 = -5) →
  (S 6 = 21 * S 2) →
  (S 8 = -85) :=
by
  -- Assume the given conditions
  intros h1 h2
  -- The actual proof will be inserted here
  sorry

end geometric_seq_sum_l267_267840


namespace solve_for_x_l267_267005

theorem solve_for_x (x : ℚ) : (2/3 : ℚ) - 1/4 = 1/x → x = 12/5 := 
by
  sorry

end solve_for_x_l267_267005


namespace sum_of_eight_l267_267883

variable (a₁ : ℕ) (q : ℕ)
variable (S : ℕ → ℕ) -- Assume S is a function from natural numbers to natural numbers representing S_n

-- Condition 1: Sum of first 4 terms equals -5
axiom h1 : S 4 = -5

-- Condition 2: Sum of the first 6 terms is 21 times the sum of the first 2 terms
axiom h2 : S 6 = 21 * S 2

-- The formula for the sum of the first n terms of a geometric sequence
def sum_of_first_n_terms (n : ℕ) := a₁ * (1 - q^n) / (1 - q)

-- Statement to be proved
theorem sum_of_eight : S 8 = -85 := sorry

end sum_of_eight_l267_267883


namespace ellipse_standard_equation_range_of_eccentricity_l267_267770

noncomputable def ellipse (x y : ℝ) (a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

noncomputable def point_ellipse (a b : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ ellipse (sqrt(6) / 2) 1 a b

noncomputable def focal_length_condition (a b : ℝ) : Prop :=
  a^2 - b^2 = 1

noncomputable def point_relationship (P : ℝ × ℝ) (A : ℝ × ℝ) (F : ℝ × ℝ) : Prop :=
  let (x, y) := P in
  let (a1, a2) := A in
  let (f1, f2) := F in
  sqrt((x - a1)^2 + y^2) = sqrt(2) * sqrt((x - f1)^2 + y^2)

theorem ellipse_standard_equation :
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧
  focal_length_condition a b ∧
  point_ellipse a b ∧
  (∀ (x y : ℝ), ellipse x y a b ↔ (x^2 / 3 + y^2 / 2 = 1)) :=
sorry

theorem range_of_eccentricity :
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧
  focal_length_condition a b ∧
  eclipse a b ∧
  ∃ (P : ℝ × ℝ), point_relationship P (-2, 0) (-1, 0) ∧
  √2 ≤ a ∧ a ≤ √3 ∧ (1/a) * √(a^2 - (a^2 - b^2)) ∈ [√(3)/3, √(2)/2] :=
sorry 

end ellipse_standard_equation_range_of_eccentricity_l267_267770


namespace derivative_problem_l267_267745

variable {α : Type} [LinearOrder α] [TopologicalSpace α] [OrderTopology α] [Zero α] [Add α] [Neg α] [HasSub α]

open Filter TopologicalSpace

noncomputable def f' (f : α → α) (x : α) := 
  lim (tendsto_nhds (λ h, (f (x + h) - f x) / h))

theorem derivative_problem 
  {f : ℝ → ℝ} {x₀ : ℝ}
  (h : lim (λ h, ((f x₀) - (f (x₀ + 3 * h))) / (2 * h)) (nhds (0 : ℝ)) = 1) : 
  f' f x₀ = -2 / 3 :=
sorry

end derivative_problem_l267_267745


namespace triangle_FG_value_l267_267292

theorem triangle_FG_value (c k d : ℝ) (h_triangle : ∀{E F G H : Type}, -- triangle EFG is right-angled
  (EG = c) ∧ (GH / HF = 4 / 5) ∧ (∠GEH = ∠FEH) ∧ (c = 1) 
  → d = FG ): 
(GH = 4 * k) ∧ (HF = 5 * k) ∧ (k = 1 / 12) 
 → d = 3 / 4 :=
by sorry

end triangle_FG_value_l267_267292


namespace geometric_sequence_sum_S8_l267_267852

noncomputable def sum_geometric_seq (a q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_S8 (a q : ℝ) (h1 : q ≠ 1)
  (h2 : sum_geometric_seq a q 4 = -5)
  (h3 : sum_geometric_seq a q 6 = 21 * sum_geometric_seq a q 2) :
  sum_geometric_seq a q 8 = -85 :=
sorry

end geometric_sequence_sum_S8_l267_267852


namespace diane_postage_problem_l267_267672

/-
  Diane has one 1-cent stamp, two identical 2-cent stamps, up to fifteen identical 15-cent stamps.
  We are to find the number of distinct arrangements that yield exactly 15 cents, considering
  that arrangements which are a rotation, inversion, or swapping equivalent stamps are the same.
-/
def number_of_ways : Nat := sorry

theorem diane_postage_problem :
  ∃ n : Nat, 
  n = number_of_ways 
  :=
begin
  sorry
end

end diane_postage_problem_l267_267672


namespace distance_between_max_min_cos_l267_267729

theorem distance_between_max_min_cos (x : ℝ) :
  let y := cos (x + 1)
  sqrt (Real.pi ^ 2 + 4) = sqrt ((2 * Real.pi / 2) ^ 2 + 2 ^ 2) :=
by
  sorry

end distance_between_max_min_cos_l267_267729


namespace alice_saves_four_dimes_in_seven_days_l267_267127

def amount_saved (n : ℕ) : ℚ :=
  0.01 * 2^(n - 1)

/--
Given that Alice starts saving with a penny on the first day and doubles the amount saved each subsequent day, prove that it takes her exactly 7 days to save at least 4 dimes (0.40 dollars).
-/
theorem alice_saves_four_dimes_in_seven_days :
  ∃ n, amount_saved (7) ≥ 0.40 :=
by
  sorry

end alice_saves_four_dimes_in_seven_days_l267_267127


namespace reroll_probability_l267_267797

/-- 
Given:
1. Jason rolls three fair standard six-sided dice.
2. After seeing the roll, he may choose to reroll a subset of the dice (possibly empty, possibly all three dice).
3. He wins if the sum of the numbers face up on the three dice after rerolling is exactly 8.
4. Any rerolled dice showing a "1" result in losing.
5. To win, rerolled even-numbered dice must also show even numbers.

Prove that the probability that Jason chooses to reroll exactly two of the dice is 1/144.
-/
theorem reroll_probability (d1 d2 d3 : ℕ) (rolls: set ℕ) :
  (∑ (x:ℕ) in rolls, x) = 8 →
  ∀ x ∈ rolls, x ∉ {1} → 
  ∀ x ∈ rolls, x % 2 = 0 →
  probability_reroll_two = 1 / 144 :=
sorry

end reroll_probability_l267_267797


namespace chrystal_pass_mountain_in_six_hours_l267_267647

-- Define the conditions as constants.
def initial_speed : ℝ := 30  -- miles per hour
def speed_factor_ascend : ℝ := 0.5
def speed_factor_descend : ℝ := 1.2
def distance_ascend : ℝ := 60  -- miles
def distance_descend : ℝ := 72  -- miles

-- Define the total time to pass the mountain
def total_time_to_pass_mountain : ℝ :=
  (distance_ascend / (initial_speed * speed_factor_ascend)) +
  (distance_descend / (initial_speed * speed_factor_descend))

-- State the theorem to be proved
theorem chrystal_pass_mountain_in_six_hours :
  total_time_to_pass_mountain = 6 := by
  sorry -- Proof omitted

end chrystal_pass_mountain_in_six_hours_l267_267647


namespace solve_for_x_l267_267942

theorem solve_for_x : 
  let x := (√(7^2 + 24^2)) / (√(49 + 16)) in 
  x = 25 * √65 / 65 := 
by
  -- Step 1: expand the terms inside the square roots
  let a := 7^2 + 24^2 
  let b := 49 + 16

  have a_eq : a = 625 := by
    calc
      a = 7^2 + 24^2 : rfl
      ... = 49 + 576 : rfl
      ... = 625 : rfl

  have b_eq : b = 65 := by
    calc
      b = 49 + 16 : rfl
      ... = 65 : rfl

  -- Step 2: Simplify the square roots
  let sqrt_a := √a
  have sqrt_a_eq : sqrt_a = 25 := by
    rw [a_eq]
    norm_num

  let sqrt_b := √b
  have sqrt_b_eq : sqrt_b = √65 := by
    rw [b_eq]

  -- Step 3: Simplify x
  let x := sqrt_a / sqrt_b

  show x = 25 * √65 / 65
  rw [sqrt_a_eq, sqrt_b_eq]
  field_simp
  norm_num
  rw [mul_div_cancel_left 25 (sqrt_ne_zero.2 (ne_of_gt (by norm_num : √65 ≠ 0))) ]
  sorry

end solve_for_x_l267_267942


namespace number_of_children_l267_267291

theorem number_of_children (C : ℝ) 
  (h1 : 0.30 * C >= 0)
  (h2 : 0.20 * C >= 0)
  (h3 : 0.50 * C >= 0)
  (h4 : 0.70 * C = 42) : 
  C = 60 := by
  sorry

end number_of_children_l267_267291


namespace sum_of_sequence_l267_267220

section
-- Definitions
def S (n : ℕ) : ℚ := (n / 2) * (2 * a₁ + (n - 1) * d)  -- Sum of the first n terms of the arithmetic sequence
def a₁ := 0  -- From the solution's system of linear equations solving
def d := -1  -- From the solution's system of linear equations solving
def a (n : ℕ) : ℚ := a₁ + (n - 1) * d  -- General term of the arithmetic sequence
def b (n : ℕ) : ℚ := -a n + 1

-- The theorem statement
theorem sum_of_sequence (n : ℕ) :
  S 3 = -3 ∧ S 7 = -21 →
  (∀ n, a n = -n + 1) ∧ (T n = (3 / 4) - ((2 * n + 3) / (2 * (n + 1) * (n + 2)))) :=
sorry
  
end

end sum_of_sequence_l267_267220


namespace rain_probability_l267_267406

theorem rain_probability:
  (P_sun : ℝ) (P_sat : ℝ) (P_mon : ℝ)
  (independent_days: ∀ A B C : Prop, A ∧ B → C ∧ (A → ¬ C) ∧ (B → ¬ C) ) :
  P_sat = 0.60 → P_sun = 0.40 → (∀ i, P_mon i = 0.30 → (
    let both_sat_sun := P_sat * P_sun,
        only_sat := P_sat * (1 - P_sun),
        only_sun := (1 - P_sat) * P_sun in
    both_sat_sun + only_sat + only_sun = 0.52) :=
by
  intros P_sun P_sat P_mon independent_days hSAT hSUN hMON
  sorry

end rain_probability_l267_267406


namespace time_to_read_18_pages_l267_267628

-- Definitions based on the conditions
def reading_rate : ℚ := 2 / 4 -- Amalia reads 4 pages in 2 minutes
def pages_to_read : ℕ := 18 -- Number of pages Amalia needs to read

-- Goal: Total time required to read 18 pages
theorem time_to_read_18_pages (r : ℚ := reading_rate) (p : ℕ := pages_to_read) :
  p * r = 9 := by
  sorry

end time_to_read_18_pages_l267_267628


namespace geometric_sequence_sum_l267_267892

variable (a1 q : ℝ) -- Define the first term and common ratio as real numbers

-- Define the sum of the first n terms of a geometric sequence
def S (n : ℕ) : ℝ := a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum :
  (S 4 a1 q = -5) → (S 6 a1 q = 21 * S 2 a1 q) → (S 8 a1 q = -85) :=
by
  intro h1 h2
  sorry

end geometric_sequence_sum_l267_267892


namespace solve_for_x_l267_267945

noncomputable def n : ℝ := Real.sqrt (7^2 + 24^2)
noncomputable def d : ℝ := Real.sqrt (49 + 16)
noncomputable def x : ℝ := n / d

theorem solve_for_x : x = 5 * Real.sqrt 65 / 13 := by
  sorry

end solve_for_x_l267_267945


namespace football_team_goal_l267_267093

-- Definitions of the conditions
def L1 : ℤ := -5
def G2 : ℤ := 13
def L3 : ℤ := -(L1 ^ 2)
def G4 : ℚ := - (L3 : ℚ) / 2

def total_yardage : ℚ := L1 + G2 + L3 + G4

-- The statement to be proved
theorem football_team_goal : total_yardage < 30 := by
  -- sorry for now since no proof is needed
  sorry

end football_team_goal_l267_267093


namespace regular_polygon_sides_l267_267603

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ n → ∃ p, ∠(polygon_interior_angle p) = 144) : n = 10 := 
by
  sorry

end regular_polygon_sides_l267_267603


namespace sum_of_squares_l267_267426

variable (a b c : ℝ)
variable (S : ℝ)

theorem sum_of_squares (h1 : ab + bc + ac = 131)
                       (h2 : a + b + c = 22) :
  a^2 + b^2 + c^2 = 222 :=
by
  -- Proof would be placed here
  sorry

end sum_of_squares_l267_267426


namespace square_field_area_l267_267076

theorem square_field_area (d : ℝ) (h : d = 26): (d^2 / 2) = 338 :=
by
  rw h
  sorry

end square_field_area_l267_267076


namespace five_consecutive_product_div_24_l267_267485

theorem five_consecutive_product_div_24 (n : ℤ) : 
  24 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) := 
sorry

end five_consecutive_product_div_24_l267_267485


namespace monthly_average_growth_rate_additional_staff_needed_l267_267430

-- Define the constants and conditions
def initial_deliveries_january : ℕ := 100000
def deliveries_march : ℕ := 121000
def current_staff : ℕ := 21
def per_person_capacity : ℕ := 600 -- converting 0.6 thousand to actual value

-- Problem 1: Prove the monthly average growth rate
theorem monthly_average_growth_rate (x : ℝ) : 
  (initial_deliveries_january * (1 + x)^2 = deliveries_march) → (x = 0.1) :=
by
  sorry

-- Problem 2: Prove at least 35 additional staff members are needed
theorem additional_staff_needed (m : ℕ) : 
  (deliveries_march * 1.1 > current_staff * per_person_capacity) →
  (m * per_person_capacity + current_staff * per_person_capacity ≥ deliveries_march * 1.1) → 
  m ≥ 35 :=
by
  sorry

end monthly_average_growth_rate_additional_staff_needed_l267_267430


namespace max_knights_in_grid_l267_267288

inductive Person
| knight
| liar

def adjacent (i j : ℕ × ℕ) : ℕ × ℕ → Prop
| (i, j) (i, j') := j ≠ j' ∧ j = j'
| (i, j) (i', j) := i ≠ i' ∧ i = i'
| _ _            := false

def valid_configuration (grid : ℕ → ℕ → Person) : Prop :=
  ∀ i j, (0 ≤ i ∧ i < 3) ∧ (0 ≤ j ∧ j < 3) →
  (grid i j = (Person.knight)) →
  (∃ n, adjacent (i, j) n ∧ grid n = (Person.liar))

theorem max_knights_in_grid : 
  ∃ (grid : ℕ → ℕ → Person), valid_configuration grid ∧ 
  (Σ' (i j : ℕ), grid i j = Person.knight).card = 6 :=
sorry

end max_knights_in_grid_l267_267288


namespace geometric_seq_sum_l267_267839

-- Definitions of the conditions
variables {a₁ q : ℚ}
def S (n : ℕ) := a₁ * (1 - q^n) / (1 - q)

-- Hypotheses from the conditions
theorem geometric_seq_sum :
  (S 4 = -5) →
  (S 6 = 21 * S 2) →
  (S 8 = -85) :=
by
  -- Assume the given conditions
  intros h1 h2
  -- The actual proof will be inserted here
  sorry

end geometric_seq_sum_l267_267839


namespace trip_assistant_cost_l267_267049

theorem trip_assistant_cost :
  (let 
    hours_one_way := 4
    hours_round_trip := hours_one_way * 2
    cost_per_hour := 10
    total_cost := hours_round_trip * cost_per_hour
  in 
    total_cost = 80) :=
by
  simp only []
  sorry

end trip_assistant_cost_l267_267049


namespace square_floor_tile_count_l267_267069

/-
A square floor is tiled with congruent square tiles.
The tiles on the two diagonals of the floor are black.
If there are 101 black tiles, then the total number of tiles is 2601.
-/
theorem square_floor_tile_count  
  (s : ℕ) 
  (hs_odd : s % 2 = 1)  -- s is odd
  (h_black_tile_count : 2 * s - 1 = 101) 
  : s^2 = 2601 := 
by 
  sorry

end square_floor_tile_count_l267_267069


namespace sqrt_x_plus_1_defined_range_l267_267045

theorem sqrt_x_plus_1_defined_range (x : ℝ) : (∃ y : ℝ, y = sqrt (x + 1)) ↔ x ≥ -1 :=
by
  sorry

end sqrt_x_plus_1_defined_range_l267_267045


namespace option_a_solution_l267_267350

theorem option_a_solution (x y : ℕ) (h₁: x = 2) (h₂: y = 2) : 2 * x + y = 6 := by
sorry

end option_a_solution_l267_267350


namespace ratio_correct_l267_267636

noncomputable def ratio_female_to_male_members (f m : ℕ) (hf : 0 < f) (hm : 0 < m)
  (avg_female_age : ℕ := 45) (avg_male_age : ℕ := 20) (avg_member_age : ℕ := 28) 
  (h1 : ¬(f = 0 ∨ m = 0)) (hf_avg : avg_female_age * f) 
  (hm_avg : avg_male_age * m) (h_avg : (45 * f + 20 * m) / (f + m) = 28) : ℚ :=
(f : ℚ) / (m : ℚ)

theorem ratio_correct (f m : ℕ) (h1 : ¬(f = 0 ∨ m = 0)) 
  (hf_avg : 45 * f) (hm_avg : 20 * m) 
  (h_avg : (45 * f + 20 * m) / (f + m) = 28) : (f : ℚ) / (m : ℚ) = 8 / 17 :=
by
  sorry

end ratio_correct_l267_267636


namespace largest_divisor_of_product_of_five_consecutive_integers_l267_267465

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∀ (n : ℤ), ∃ (d : ℤ), d = 60 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end largest_divisor_of_product_of_five_consecutive_integers_l267_267465


namespace b_minus_a_less_zero_l267_267266

-- Given conditions
variables {a b : ℝ}

-- Define the condition
def a_greater_b (a b : ℝ) : Prop := a > b

-- Lean 4 proof problem statement
theorem b_minus_a_less_zero (a b : ℝ) (h : a_greater_b a b) : b - a < 0 := 
sorry

end b_minus_a_less_zero_l267_267266


namespace largest_divisor_of_5_consecutive_integers_l267_267478

theorem largest_divisor_of_5_consecutive_integers :
  ∀ (a b c d e : ℤ), 
    a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e →
    (∃ k : ℤ, k ∣ (a * b * c * d * e) ∧ k = 60) :=
by 
  intro a b c d e h
  sorry

end largest_divisor_of_5_consecutive_integers_l267_267478


namespace largest_divisor_of_five_consecutive_integers_l267_267460

open Nat

theorem largest_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, k ∈ {n, n+1, n+2, n+3, n+4} ∧
    ∀ m ∈ {2, 3, 4, 5}, m ∣ k → 60 ∣ (n * (n+1) * (n+2) * (n+3) * (n+4)) := 
sorry

end largest_divisor_of_five_consecutive_integers_l267_267460


namespace geometric_seq_sum_l267_267834

-- Definitions of the conditions
variables {a₁ q : ℚ}
def S (n : ℕ) := a₁ * (1 - q^n) / (1 - q)

-- Hypotheses from the conditions
theorem geometric_seq_sum :
  (S 4 = -5) →
  (S 6 = 21 * S 2) →
  (S 8 = -85) :=
by
  -- Assume the given conditions
  intros h1 h2
  -- The actual proof will be inserted here
  sorry

end geometric_seq_sum_l267_267834


namespace eight_term_sum_l267_267822

variable {α : Type*} [Field α]
variable (a q : α)

-- Define the n-th sum of the geometric sequence
def S_n (n : ℕ) : α := a * (1 - q ^ n) / (1 - q)

-- Given conditions
def S4 : α := S_n 4 = -5
def S6 : α := S_n 6 = 21 * S_n 2

-- Prove the target statement
theorem eight_term_sum : S_n 8 = -85 :=
  sorry

end eight_term_sum_l267_267822


namespace no_common_parallel_plane_l267_267561

noncomputable def skew_lines (a b : Line) : Prop :=
  ∃ (P : Point), P ∉ a ∧ P ∉ b ∧ ¬∃ (α : Plane), a ⊆ α ∧ b ⊆ α

theorem no_common_parallel_plane (a b : Line) (A : Point) (ha : skew_lines a b) (hA : A ∉ a ∧ A ∉ b) :
  ∃ (α : Plane), α ∃ (a' b' : Line), a' ∥ a ∧ b' ∥ b ∧ A ∈ a' ∧ A ∈ b' ∧ α ∋ a' ∧ α ∋ b' ∧ (¬(a ⊆ α ∨ b ⊆ α)) :=
sorry

end no_common_parallel_plane_l267_267561


namespace smaller_cube_volume_l267_267988

variable (e_s e_l : ℝ)

theorem smaller_cube_volume (h1 : e_l ^ 3 / e_s ^ 3 = 125) 
  (h2 : e_l / e_s ≈ 5) 
  (h3 : e_l ^ 3 = 125) : 
  e_s ^ 3 = 1 := 
sorry

end smaller_cube_volume_l267_267988


namespace area_calculation_l267_267148

noncomputable def area_bounded (x y : ℝ → ℝ) (t : ℝ) : ℝ :=
  ∫ t in -π/3..π/3, x t * (deriv y t) dt

theorem area_calculation :
  let x (t : ℝ) := 32 * (Real.cos t) ^ 3,
      y (t : ℝ) := (Real.sin t) ^ 3,
      S := area_bounded x y in
  S = 4 * Real.pi + 3 * Real.sqrt 3 :=
sorry

end area_calculation_l267_267148


namespace largest_divisor_of_product_of_five_consecutive_integers_l267_267497

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∀ (n : ℤ), ∃ k : ℤ, k = 60 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intro n
  use 60
  split
  { refl }
  { sorry }

end largest_divisor_of_product_of_five_consecutive_integers_l267_267497


namespace set_difference_equals_six_l267_267171

-- Set Operations definitions used
def set_difference (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

-- Define sets M and N
def M : Set ℕ := {1, 2, 3, 4, 5}
def N : Set ℕ := {2, 3, 6}

-- Problem statement to prove
theorem set_difference_equals_six : set_difference N M = {6} :=
  sorry

end set_difference_equals_six_l267_267171


namespace arithmetic_sequence_sum_l267_267221

variable {a : ℕ → ℝ}

noncomputable def sum_of_first_ten_terms (a : ℕ → ℝ) : ℝ :=
  (10 / 2) * (a 1 + a 10)

theorem arithmetic_sequence_sum (h : a 5 + a 6 = 28) :
  sum_of_first_ten_terms a = 140 :=
by
  sorry

end arithmetic_sequence_sum_l267_267221


namespace regular_polygon_sides_l267_267613

theorem regular_polygon_sides (n : ℕ) (h : n ≥ 3) (interior_angle : ℝ) : 
  interior_angle = 144 → n = 10 :=
by
  intro h1
  sorry

end regular_polygon_sides_l267_267613


namespace S₈_proof_l267_267860

-- Definitions
variables {a₁ q : ℝ} {S : ℕ → ℝ}

-- Condition for the sum of the first n terms of the geometric sequence
def geom_sum (n : ℕ) : ℝ :=
  a₁ * (1 - q ^ n) / (1 - q)

-- Conditions from the problem
def S₄ := geom_sum 4 = -5
def S₆ := geom_sum 6 = 21 * geom_sum 2

-- Theorem to prove
theorem S₈_proof (h₁ : S₄) (h₂ : S₆) : geom_sum 8 = -85 :=
by
  sorry

end S₈_proof_l267_267860


namespace solution_m_le_9_l267_267731

theorem solution_m_le_9 (x : ℝ) (m : ℝ) :
  (2^(x^2 - 4 * x + 3) < 1) ∧ (2 / (4 - x) ≥ 1) → (2x^2 - 9 * x + m < 0) :=
by {
  intro h,
  cases h with h1 h2,
  have x_range : 2 ≤ x ∧ x < 3, {
    sorry, -- Using the conditions 2^(x^2 - 4*x + 3) < 1 and 2/(4-x) ≥ 1,
           -- derive the range [2, 3).
  },
  show 2x^2 - 9 * x + m < 0,
  sorry, -- Prove that for all x in [2, 3), 2x^2 - 9*x + m < 0 implies m ≤ 9
}.

end solution_m_le_9_l267_267731


namespace geometric_sequence_S8_l267_267870

theorem geometric_sequence_S8 
  (a1 q : ℝ) (S : ℕ → ℝ)
  (h1 : S 4 = a1 * (1 - q^4) / (1 - q) = -5)
  (h2 : S 6 = 21 * S 2)
  (geom_sum : ∀ n, S n = a1 * (1 - q^n) / (1 - q))
  (h3 : q ≠ 1) : S 8 = -85 := 
begin
  sorry
end

end geometric_sequence_S8_l267_267870


namespace exactly_one_wins_probability_l267_267077

theorem exactly_one_wins_probability :
  let P_A := (2 : ℚ) / 3
  let P_B := (3 : ℚ) / 4
  P_A * (1 - P_B) + P_B * (1 - P_A) = (5 : ℚ) / 12 := by
  let P_A := (2 : ℚ) / 3
  let P_B := (3 : ℚ) / 4
  change P_A * (1 - P_B) + P_B * (1 - P_A) = (5 : ℚ) / 12
  sorry

end exactly_one_wins_probability_l267_267077


namespace arc_length_l267_267020

def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

theorem arc_length (r : ℝ) (c: ℝ) (theta : ℝ) (h1 : c = 72) (h2 : theta = 45) :
  (theta / 360) * c = 9 :=
by
  sorry

end arc_length_l267_267020


namespace monotonic_decreasing_interval_l267_267966

noncomputable def f : ℝ → ℝ := λ x, x^3 - 3*x^2 + 1

-- Define the derivative of the function
noncomputable def f_prime : ℝ → ℝ := λ x, 3*x^2 - 6*x

-- Statement to prove: f is monotonic decreasing in (0, 2)
theorem monotonic_decreasing_interval : ∀ x, 0 < x ∧ x < 2 → f_prime x < 0 :=
by
  sorry

end monotonic_decreasing_interval_l267_267966


namespace distance_between_stripes_l267_267122

/-- Define the parallel curbs and stripes -/
structure Crosswalk where
  distance_between_curbs : ℝ
  curb_distance_between_stripes : ℝ
  stripe_length : ℝ
  stripe_cross_distance : ℝ
  
open Crosswalk

/-- Conditions given in the problem -/
def crosswalk : Crosswalk where
  distance_between_curbs := 60 -- feet
  curb_distance_between_stripes := 20 -- feet
  stripe_length := 50 -- feet
  stripe_cross_distance := 50 -- feet

/-- Theorem to prove the distance between stripes -/
theorem distance_between_stripes (cw : Crosswalk) :
  2 * (cw.curb_distance_between_stripes * cw.distance_between_curbs) / cw.stripe_length = 24 := sorry

end distance_between_stripes_l267_267122


namespace circle_radius_increase_l267_267954

-- Defining the problem conditions and the resulting proof
theorem circle_radius_increase (r n : ℝ) (h : π * (r + n)^2 = 3 * π * r^2) : r = n * (Real.sqrt 3 - 1) / 2 :=
sorry  -- Proof is left as an exercise

end circle_radius_increase_l267_267954


namespace product_of_five_consecutive_is_divisible_by_sixty_l267_267519

theorem product_of_five_consecutive_is_divisible_by_sixty (n : ℤ) :
  60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end product_of_five_consecutive_is_divisible_by_sixty_l267_267519


namespace sum_of_pairwise_distinct_positive_numbers_l267_267925

theorem sum_of_pairwise_distinct_positive_numbers
  (x y z t : ℝ)
  (h_distinct : x ≠ y ∧ x ≠ z ∧ x ≠ t ∧ y ≠ z ∧ y ≠ t ∧ z ≠ t)
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0)
  (h_sum : x + y + z + t = 1) :
  (sqrt x + sqrt y > 1) ∨ (sqrt x + sqrt z > 1) ∨ (sqrt x + t > 1) ∨
  (sqrt y + sqrt z > 1) ∨ (sqrt y + sqrt t > 1) ∨ (sqrt z + sqrt t > 1) := 
sorry

end sum_of_pairwise_distinct_positive_numbers_l267_267925


namespace sqrt_2_cos_pi_over_4_minus_alpha_l267_267231

theorem sqrt_2_cos_pi_over_4_minus_alpha (α : ℝ) 
  (h1 : sin (2 * α) = 24 / 25) 
  (h2 : 0 < α ∧ α < π / 2) :
  sqrt 2 * cos (π / 4 - α) = 7 / 5 :=
by
  sorry

end sqrt_2_cos_pi_over_4_minus_alpha_l267_267231


namespace line_equation_l267_267385

theorem line_equation (m : ℝ) (x1 y1 : ℝ) (b : ℝ) :
  m = -3 → x1 = -2 → y1 = 0 → 
  (∀ x y, y - y1 = m * (x - x1) ↔ 3 * x + y + 6 = 0) :=
sorry

end line_equation_l267_267385


namespace rectangular_plot_area_l267_267393

-- Define the conditions
def breadth := 11  -- breadth in meters
def length := 3 * breadth  -- length is thrice the breadth

-- Define the function to calculate area
def area (length breadth : ℕ) := length * breadth

-- The theorem to prove
theorem rectangular_plot_area : area length breadth = 363 := by
  sorry

end rectangular_plot_area_l267_267393


namespace sum_of_perimeters_l267_267079

-- Define the concept of a triangle and its perimeter
structure Triangle :=
  (A B C : Point)

def perimeter (t : Triangle) : ℝ :=
  dist t.A t.B + dist t.B t.C + dist t.C t.A

-- Define the concept of a point and distance
structure Point := 
  (x y : ℝ)

def dist (p1 p2 : Point) : ℝ :=
  real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define the condition of inscribed circle center O and parallel lines cutting triangles
structure InscribedCircleCenter (t : Triangle) :=
  (O : Point)

structure ParallelLinesThroughO (t : Triangle) (i : InscribedCircleCenter t) :=
  (line1 line2 line3 : Line)
  (parallel_line1 : ∃ b1 e1, b1 ≠ e1 ∧ Line.through b1 e1 = line1 ∧ Line.parallel t.AB line1)
  (parallel_line2 : ∃ b2 e2, b2 ≠ e2 ∧ Line.through b2 e2 = line2 ∧ Line.parallel t.BC line1)
  (parallel_line3 : ∃ b3 e3, b3 ≠ e3 ∧ Line.through b3 e3 = line3 ∧ Line.parallel t.CA line1)
  (through_O_line1 : line1.contains i.O)
  (through_O_line2 : line2.contains i.O)
  (through_O_line3 : line3.contains i.O)

-- Define the concept of line, the 'through' relation, and the 'parallel' relation
structure Line := 
  (through : Point → Point)

def Line.parallel (l1 l2 : Line) : Prop :=
  ∃ θ, ∀ p1 p2, l1.through p1 → l2.through p2 → ∃ k, k ≠ 0 ∧ (p2.y - p1.y) = k * (p2.x - p1.x)

-- Define statement of the problem
theorem sum_of_perimeters (t : Triangle) (i : InscribedCircleCenter t) (pl : ParallelLinesThroughO t i) :
  let cut_off_triangles := [ (t.A, pl.line1.through t.B), (t.B, pl.line2.through t.C), (t.C, pl.line3.through t.A) ]
  2 * perimeter t = list.sum (list.map perimeter cut_off_triangles) :=
sorry

end sum_of_perimeters_l267_267079


namespace S₈_proof_l267_267854

-- Definitions
variables {a₁ q : ℝ} {S : ℕ → ℝ}

-- Condition for the sum of the first n terms of the geometric sequence
def geom_sum (n : ℕ) : ℝ :=
  a₁ * (1 - q ^ n) / (1 - q)

-- Conditions from the problem
def S₄ := geom_sum 4 = -5
def S₆ := geom_sum 6 = 21 * geom_sum 2

-- Theorem to prove
theorem S₈_proof (h₁ : S₄) (h₂ : S₆) : geom_sum 8 = -85 :=
by
  sorry

end S₈_proof_l267_267854


namespace complement_of_M_with_respect_to_U_l267_267429

namespace Complements

open Set

def U : Set Int := {1, -2, 3, -4, 5, -6}
def M : Set Int := {1, -2, 3, -4}

theorem complement_of_M_with_respect_to_U :
  U \ M = {5, -6} :=
by
  sorry

end Complements

end complement_of_M_with_respect_to_U_l267_267429


namespace geometric_seq_sum_l267_267838

-- Definitions of the conditions
variables {a₁ q : ℚ}
def S (n : ℕ) := a₁ * (1 - q^n) / (1 - q)

-- Hypotheses from the conditions
theorem geometric_seq_sum :
  (S 4 = -5) →
  (S 6 = 21 * S 2) →
  (S 8 = -85) :=
by
  -- Assume the given conditions
  intros h1 h2
  -- The actual proof will be inserted here
  sorry

end geometric_seq_sum_l267_267838


namespace distance_between_neg5_and_neg1_l267_267961

theorem distance_between_neg5_and_neg1 : 
  dist (-5 : ℝ) (-1) = 4 := by
sorry

end distance_between_neg5_and_neg1_l267_267961


namespace chessboard_property_exists_l267_267652

theorem chessboard_property_exists (n : ℕ) (x : Fin n → Fin n → ℝ) 
  (h : ∀ i j k : Fin n, x i j + x j k + x k i = 0) :
  ∃ (t : Fin n → ℝ), ∀ i j, x i j = t i - t j := 
sorry

end chessboard_property_exists_l267_267652


namespace measure_of_angle_F_l267_267309

-- Definitions of the conditions
variable (x : ℝ) -- angle D
variable (angleD : ℝ)
variable (angleE : ℝ)
variable (angleF : ℝ)

def condition1 := angleD = x
def condition2 := angleE = 2 * x
def condition3 := angleF = x + 40
def condition4 := angleD + angleE + angleF = 180

-- Theorem statement
theorem measure_of_angle_F : (condition1 x angleD) → (condition2 x angleE) → (condition3 x angleF) → (condition4 x angleD angleE angleF) → angleF = 75 := by
  sorry

end measure_of_angle_F_l267_267309


namespace percent_less_than_l267_267983

theorem percent_less_than (A B : ℕ) (h : A / B = 3 / 4) : (B - A) / B = 0.25 :=
by
  sorry

end percent_less_than_l267_267983


namespace distance_sum_find_m_alpha_l267_267254

noncomputable def parametric_curve (m α t : ℝ) : ℝ × ℝ :=
  (m + t * Real.cos α, t * Real.sin α)

noncomputable def polar_curve (θ : ℝ) : ℝ :=
  4 * Real.cos θ

theorem distance_sum (ϕ : ℝ) :
  let OA := 4 * Real.cos ϕ,
      OB := 4 * Real.cos (ϕ + Real.pi / 4),
      OC := 4 * Real.cos (ϕ - Real.pi / 4) in
  OB + OC = Real.sqrt 2 * OA :=
by
  -- Proof steps would go here...
  sorry

theorem find_m_alpha (m α : ℝ) (t₁ t₂ : ℝ) :
  let ϕ := Real.pi / 12
  let B := (1.0, Real.sqrt 3)
  let C := (3.0, -Real.sqrt 3) in
  B = parametric_curve m α t₁ ∧ C = parametric_curve m α t₂ → 
  m = 2 ∧ α = 2 * Real.pi / 3 :=
by
  -- Proof steps would go here...
  sorry

end distance_sum_find_m_alpha_l267_267254


namespace count_valid_numbers_l267_267740

theorem count_valid_numbers : 
  (∃ (N : ℕ) (a b c d : ℕ), 3000 ≤ N ∧ N < 5000 ∧ 
  N % 4 = 0 ∧ 2 ≤ b ∧ b < c ∧ c ≤ 7 ∧ 
  N = 1000 * a + 100 * b + 10 * c + d) → 90 :=
sorry

end count_valid_numbers_l267_267740


namespace largest_divisor_of_5_consecutive_integers_l267_267531

theorem largest_divisor_of_5_consecutive_integers :
  ∀ (n : ℤ), ∃ d, d = 120 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intro n
  use 120
  split
  exact rfl
  sorry

end largest_divisor_of_5_consecutive_integers_l267_267531


namespace highest_probability_of_red_ball_l267_267298

theorem highest_probability_of_red_ball (red yellow white blue : ℕ) (H1 : red = 5) (H2 : yellow = 4) (H3 : white = 1) (H4 : blue = 3) :
  (red : ℚ) / (red + yellow + white + blue) > (yellow : ℚ) / (red + yellow + white + blue) ∧
  (red : ℚ) / (red + yellow + white + blue) > (white : ℚ) / (red + yellow + white + blue) ∧
  (red : ℚ) / (red + yellow + white + blue) > (blue : ℚ) / (red + yellow + white + blue) := 
by {
  sorry
}

end highest_probability_of_red_ball_l267_267298


namespace smallest_prime_after_four_consecutive_nonprime_l267_267552

def is_prime (n: ℕ) := nat.prime n
def is_nonprime (n: ℕ) := ¬ nat.prime n

theorem smallest_prime_after_four_consecutive_nonprime :
  ∃ p, is_prime p ∧ 
      (∃ n, is_nonprime n ∧ is_nonprime (n + 1) ∧ is_nonprime (n + 2) ∧ is_nonprime (n + 3) ∧
           p > n + 3 ∧ (∀ m, m > n + 3 ∧ m < p → is_nonprime m)) ∧ p = 29 :=
by sorry

end smallest_prime_after_four_consecutive_nonprime_l267_267552


namespace largest_divisor_of_product_of_five_consecutive_integers_l267_267504

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∀ (n : ℤ), ∃ k : ℤ, k = 60 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intro n
  use 60
  split
  { refl }
  { sorry }

end largest_divisor_of_product_of_five_consecutive_integers_l267_267504


namespace trip_assistant_cost_l267_267050

theorem trip_assistant_cost :
  (let 
    hours_one_way := 4
    hours_round_trip := hours_one_way * 2
    cost_per_hour := 10
    total_cost := hours_round_trip * cost_per_hour
  in 
    total_cost = 80) :=
by
  simp only []
  sorry

end trip_assistant_cost_l267_267050


namespace true_compound_proposition_l267_267226

variables (p q : Prop)

-- Assume proposition p is false
axiom h1 : p = False

-- Assume proposition q is true
axiom h2 : q = True

-- We need to prove p ∨ q
theorem true_compound_proposition : p ∨ q :=
by
  -- Given p is false and q is true, it follows p ∨ q is true
  rw [h1, h2]
  exact or.inr trivial

-- Provide a placeholder proof
sorry

end true_compound_proposition_l267_267226


namespace book_arrangements_l267_267264

theorem book_arrangements : 
  let total_books := 6
  let identical_copies_1 := 3
  let identical_copies_2 := 2
  let unique_books := total_books - identical_copies_1 - identical_copies_2
  (total_books)! / ((identical_copies_1)! * (identical_copies_2)!) = 60 :=
by
  sorry

end book_arrangements_l267_267264


namespace regular_polygon_sides_l267_267608

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, interior_angle n i = 144) : n = 10 :=
sorry

end regular_polygon_sides_l267_267608


namespace five_consecutive_product_div_24_l267_267487

theorem five_consecutive_product_div_24 (n : ℤ) : 
  24 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) := 
sorry

end five_consecutive_product_div_24_l267_267487


namespace mod_sum_inverse_eq_zero_l267_267060

theorem mod_sum_inverse_eq_zero :
  (3⁻¹ + 3⁻² + 3⁻³ + 3⁻⁴ + 3⁻⁵ + 3⁻⁶ : ℤ) % 7 = 0 := 
by
  have h1 : (3^6 : ℤ) % 7 = 1 := by sorry
  have h2 : (3⁻¹ : ℤ) % 7 = 5 := by sorry
  have h3 : (3⁻² : ℤ) % 7 = 4 := by sorry
  have h4 : (3⁻³ : ℤ) % 7 = 6 := by sorry
  have h5 : (3⁻⁴ : ℤ) % 7 = 2 := by sorry
  have h6 : (3⁻⁵ : ℤ) % 7 = 3 := by sorry
  have h7 : (3⁻⁶ : ℤ) % 7 = 1 := by sorry
  have h_sum : (5 + 4 + 6 + 2 + 3 + 1 : ℤ) % 7 = 0 := 
  by rw [←h1, add_mod, add_mod, add_mod, add_mod, add_mod]; norm_num
  exact h_sum

end mod_sum_inverse_eq_zero_l267_267060


namespace geometric_sequence_sum_l267_267884

variable (a1 q : ℝ) -- Define the first term and common ratio as real numbers

-- Define the sum of the first n terms of a geometric sequence
def S (n : ℕ) : ℝ := a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum :
  (S 4 a1 q = -5) → (S 6 a1 q = 21 * S 2 a1 q) → (S 8 a1 q = -85) :=
by
  intro h1 h2
  sorry

end geometric_sequence_sum_l267_267884


namespace minimum_distance_proof_l267_267295

noncomputable def minimum_distance (A B : Point) (wall : Wall) (dA : ℝ) (dB : ℝ) : ℝ :=
  let B' := reflect_across_wall B wall
  let distance := (A.distance_to B').round  -- where round function rounds to the nearest meter
  distance

theorem minimum_distance_proof : 
  ∀ (A B : Point) (wall : Wall)
  (dA : ℝ) (dB : ℝ)
  (h_distA: dA = 600)
  (h_distB: dB = 800)
  (h_wall_length : wall.length = 1600)
  (A_to_wall : A.vertical_distance_to wall.closest_point = dA)
  (wall_to_B : wall.closest_point.vertical_distance_to B = dB),

  minimum_distance A B wall dA dB = 2127 :=
by
  sorry  -- Proof will be done here

end minimum_distance_proof_l267_267295


namespace probability_interval_l267_267780

open Real

/-- 
  In the interval [0, π], a number θ is randomly selected.
  We need to prove that the probability that √2 ≤ √2 * cos θ + √2 * sin θ ≤ 2
  holds true is 1 / 2.
--/
theorem probability_interval :
  (Probability (λ θ, (θ ∈ Icc 0 π) ∧ (√2 ≤ √2 * cos θ + √2 * sin θ ∧ √2 * cos θ + √2 * sin θ ≤ 2)) {θ : ℝ | θ ∈ Icc 0 π}) = 1 / 2 := 
sorry

end probability_interval_l267_267780


namespace packets_for_dollars_l267_267015

variable (P R C : ℕ)

theorem packets_for_dollars :
  let dimes := 10 * C
  let taxable_dimes := 9 * C
  ∃ x, x = taxable_dimes * P / R :=
sorry

end packets_for_dollars_l267_267015


namespace shaded_area_of_triangle_l267_267110

theorem shaded_area_of_triangle 
  (A B C D E F : Point) 
  (hA : A = (0, 12)) (hB : B = (0, 0)) (hC : C = (8, 0)) (hD : D = (8, 12)) 
  (hE : E = (16, 0)) (hF : F = (8, 12)) :
  area (triangle C E F) - area (triangle D G C) = 24 :=
sorry

end shaded_area_of_triangle_l267_267110


namespace find_second_period_interest_rate_l267_267131

-- defining the given conditions
def initial_investment : ℝ := 10000
def first_period_interest_rate : ℝ := 12 / 100
def total_value_after_one_year : ℝ := 11130

-- the main statement to prove the annual interest rate of the second certificate
theorem find_second_period_interest_rate (r : ℝ) :
  let value_after_first_six_months := initial_investment * (1 + first_period_interest_rate / 2)
  in value_after_first_six_months * (1 + r / 200) = total_value_after_one_year → 
  r = 10 := 
by
  sorry

end find_second_period_interest_rate_l267_267131


namespace sales_in_july_l267_267577

def is_multiple_of (n k : ℕ) : Prop := ∃ m : ℕ, n = k * m

lemma bookstore_sales (day : ℕ) : Prop := is_multiple_of day 4
lemma sportswear_sales (day : ℕ) : Prop := ∃ n : ℕ, day = 5 + 7 * n

/-- The number of days in July when both the bookstore and the sportswear store have sales -/
theorem sales_in_july : ∃! day : ℕ, day ∈ {1, 2, ..., 31} ∧ bookstore_sales day ∧ sportswear_sales day :=
by
  sorry -- This is where the proof would go, but it is not required for this task.

end sales_in_july_l267_267577


namespace cylinder_volume_l267_267754

theorem cylinder_volume (A : ℝ) (h r : ℝ) (π : ℝ) (hA : A = 4)
  (hc : r = 1) (hh : h = 2) (hπ : π = real.pi) :
  volume = π * r^2 * h :=
begin
  sorry
end

end cylinder_volume_l267_267754


namespace calculate_triple_hash_40_l267_267662

noncomputable def hash_number (N : ℝ) : ℝ :=
0.3 * N + 2

theorem calculate_triple_hash_40 : hash_number (hash_number (hash_number 40)) = 3.86 :=
by
  have h1 : hash_number 40 = 0.3 * 40 + 2 := rfl
  have h2 : hash_number 40 = 14 := by norm_num [h1]
  have h3 : hash_number 14 = 0.3 * 14 + 2 := rfl
  have h4 : hash_number 14 = 6.2 := by norm_num [h3]
  have h5 : hash_number 6.2 = 0.3 * 6.2 + 2 := rfl
  have h6 : hash_number 6.2 = 3.86 := by norm_num [h5]
  rw [←h6, ←h4, ←h2]
  exact rfl

end calculate_triple_hash_40_l267_267662


namespace crate_problem_answer_l267_267376

noncomputable def crate_problem_sol : ℕ := 
  let a := 8
  let b := 1
  let c := 1
  let total_crates := 10
  let total_height := 43
  let possible_height_combinations := (total_crates)! / (a! * b! * c!)
  let total_possible_orientations := 3 ^ total_crates
  let probability := (90 : ℚ) / (total_possible_orientations : ℚ)
  let reduced_fraction := probability.num / probability.denom
  (reduced_fraction / reduced_fraction.gcd).num_nat

theorem crate_problem_answer : crate_problem_sol = 10 :=
  by
  sorry

end crate_problem_answer_l267_267376


namespace set_intersection_complement_l267_267570

open Set

theorem set_intersection_complement (U A B : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hA : A = {1, 2}) (hB : B = {2, 3}) :
  (U \ A) ∩ B = {3} :=
by
  sorry

end set_intersection_complement_l267_267570


namespace number_of_hexagonal_tiles_correct_l267_267087

noncomputable def number_of_hexagonal_tiles : ℕ :=
  let p := 30 - 30 in
  let h := 30 in 
  h

theorem number_of_hexagonal_tiles_correct :
  ∃ (p h : ℕ), 
    p + h = 30 ∧
    5 * p + 6 * h = 120 ∧
    h = 30 :=
by
  use 0
  use number_of_hexagonal_tiles
  split
  . exact (rfl : 0 + 30 = 30)
  split
  . exact (rfl : 5 * 0 + 6 * 30 = 120)
  . exact (rfl : 30 = 30)

end number_of_hexagonal_tiles_correct_l267_267087


namespace evaluate_nested_square_root_l267_267186

-- Define the condition
def pos_real_solution (x : ℝ) : Prop := x = Real.sqrt (18 + x)

-- State the theorem
theorem evaluate_nested_square_root :
  ∃ (x : ℝ), pos_real_solution x ∧ x = (1 + Real.sqrt 73) / 2 :=
sorry

end evaluate_nested_square_root_l267_267186


namespace num_even_tens_digit_l267_267203

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def is_tens_digit_even (n : ℕ) : Prop := (tens_digit (n^2)) % 2 = 0

theorem num_even_tens_digit :
  (finset.card (finset.filter is_tens_digit_even (finset.Icc 1 150))) = 120 :=
sorry

end num_even_tens_digit_l267_267203


namespace barrels_of_pitch_needed_l267_267117

-- Define the basic properties and conditions
def total_length_road := 16
def truckloads_per_mile := 3
def bags_of_gravel_per_truckload := 2
def gravel_to_pitch_ratio := 5
def miles_paved_first_day := 4
def miles_paved_second_day := 2 * miles_paved_first_day - 1
def miles_already_paved := miles_paved_first_day + miles_paved_second_day
def remaining_miles := total_length_road - miles_already_paved
def total_truckloads := truckloads_per_mile * remaining_miles
def total_bags_of_gravel := bags_of_gravel_per_truckload * total_truckloads
def barrels_of_pitch := total_bags_of_gravel / gravel_to_pitch_ratio

-- State the theorem to prove the number of barrels of pitch needed
theorem barrels_of_pitch_needed :
    barrels_of_pitch = 6 :=
by
    sorry

end barrels_of_pitch_needed_l267_267117


namespace octagon_product_is_65535_l267_267133

noncomputable def complex_product_of_octagon_vertices : ℂ :=
  ((2 + 0 * complex.i) * (x2 + y2 * complex.i) * (x3 + y3 * complex.i) * 
   (x4 + y4 * complex.i) * (6 + 0 * complex.i) * 
   (x6 + y6 * complex.i) * (x7 + y7 * complex.i) * (x8 + y8 * complex.i))

theorem octagon_product_is_65535
  (h : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 8 → ∃ x y : ℝ, Q_k = (x, y) ∧ 
                complex.mk x y ∈ {z : ℂ | (z - 4) ^ 8 = 1}) :
  complex_product_of_octagon_vertices = 65535 :=
by
  sorry

end octagon_product_is_65535_l267_267133


namespace KC_eq_BC_l267_267044

theorem KC_eq_BC
  {O K A M P B C : Type*}
  [circle : metric_space O]
  (hK_on_circle : circle.contains K)
  (hKA_chord : arc_length (segment K A) > π / 2)
  (hMP_tangent : is_tangent M P K)
  (hB_on_AK : intersects (perpendicular (line_through O (radius O)) (segment O A)) (segment A K) B)
  (hC_on_MP : intersects (perpendicular (line_through O (radius O)) (segment O A)) (tangent M P) C) :
  distance K C = distance B C :=
sorry

end KC_eq_BC_l267_267044


namespace largest_divisor_of_5_consecutive_integers_l267_267534

theorem largest_divisor_of_5_consecutive_integers :
  ∀ (n : ℤ), ∃ d, d = 120 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intro n
  use 120
  split
  exact rfl
  sorry

end largest_divisor_of_5_consecutive_integers_l267_267534


namespace smallest_b_factors_to_binomials_l267_267197

theorem smallest_b_factors_to_binomials :
  ∃ (b : ℤ), (∀ r s : ℤ, r * s = 1890 → r + s = b) ∧ b = 141 :=
begin
  use 141,
  split,
  { intros r s h,
    sorry },
  refl,
end

end smallest_b_factors_to_binomials_l267_267197


namespace geometric_sequence_sum_S8_l267_267844

noncomputable def sum_geometric_seq (a q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_S8 (a q : ℝ) (h1 : q ≠ 1)
  (h2 : sum_geometric_seq a q 4 = -5)
  (h3 : sum_geometric_seq a q 6 = 21 * sum_geometric_seq a q 2) :
  sum_geometric_seq a q 8 = -85 :=
sorry

end geometric_sequence_sum_S8_l267_267844


namespace geometric_sequence_sum_S8_l267_267850

noncomputable def sum_geometric_seq (a q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_S8 (a q : ℝ) (h1 : q ≠ 1)
  (h2 : sum_geometric_seq a q 4 = -5)
  (h3 : sum_geometric_seq a q 6 = 21 * sum_geometric_seq a q 2) :
  sum_geometric_seq a q 8 = -85 :=
sorry

end geometric_sequence_sum_S8_l267_267850


namespace age_of_older_teenager_l267_267427

theorem age_of_older_teenager
  (a b : ℕ) 
  (h1 : a^2 - b^2 = 4 * (a + b)) 
  (h2 : a + b = 8 * (a - b)) 
  (h3 : a > b) : 
  a = 18 :=
sorry

end age_of_older_teenager_l267_267427


namespace custom_op_value_l267_267660

-- Define the custom operation (a \$ b)
def custom_op (a b : Int) : Int := a * (b - 1) + a * b

-- Main theorem to prove the equivalence
theorem custom_op_value : custom_op 5 (-3) = -35 := by
  sorry

end custom_op_value_l267_267660


namespace Jolene_total_raised_l267_267803

theorem Jolene_total_raised :
  let babysitting_earnings := 4 * 30
  let car_washing_earnings := 5 * 12
  babysitting_earnings + car_washing_earnings = 180 :=
by
  let babysitting_earnings := 4 * 30
  let car_washing_earnings := 5 * 12
  calc
    babysitting_earnings + car_washing_earnings = 120 + 60 : by rfl
    ... = 180 : by rfl

end Jolene_total_raised_l267_267803


namespace relatively_prime_exists_l267_267224

open Nat

theorem relatively_prime_exists (m n k : ℕ) (d : ℕ) (h1 : d = gcd m n) :
  ∃ (r s : ℕ), gcd r s = 1 ∧ (r * m + s * n) % k = 0 := 
sorry

end relatively_prime_exists_l267_267224


namespace calc_two_power_neg_one_plus_three_power_zero_l267_267644

theorem calc_two_power_neg_one_plus_three_power_zero : 2⁻¹ + 3⁰ = 3 / 2 := 
  sorry

end calc_two_power_neg_one_plus_three_power_zero_l267_267644


namespace geometric_seq_sum_l267_267841

-- Definitions of the conditions
variables {a₁ q : ℚ}
def S (n : ℕ) := a₁ * (1 - q^n) / (1 - q)

-- Hypotheses from the conditions
theorem geometric_seq_sum :
  (S 4 = -5) →
  (S 6 = 21 * S 2) →
  (S 8 = -85) :=
by
  -- Assume the given conditions
  intros h1 h2
  -- The actual proof will be inserted here
  sorry

end geometric_seq_sum_l267_267841


namespace value_of_x_l267_267946

theorem value_of_x : 
  let x := (sqrt (7^2 + 24^2)) / (sqrt (49 + 16)) 
  in x = 25 * sqrt 65 / 65 
  := 
  sorry

end value_of_x_l267_267946


namespace sum_and_product_of_roots_l267_267062

theorem sum_and_product_of_roots :
  let a := 1
  let b := -7
  let c := 12
  (∀ x: ℝ, x^2 - 7*x + 12 = 0 → (x = 3 ∨ x = 4)) →
  (-b/a = 7) ∧ (c/a = 12) := 
by
  sorry

end sum_and_product_of_roots_l267_267062


namespace sum_of_eight_l267_267881

variable (a₁ : ℕ) (q : ℕ)
variable (S : ℕ → ℕ) -- Assume S is a function from natural numbers to natural numbers representing S_n

-- Condition 1: Sum of first 4 terms equals -5
axiom h1 : S 4 = -5

-- Condition 2: Sum of the first 6 terms is 21 times the sum of the first 2 terms
axiom h2 : S 6 = 21 * S 2

-- The formula for the sum of the first n terms of a geometric sequence
def sum_of_first_n_terms (n : ℕ) := a₁ * (1 - q^n) / (1 - q)

-- Statement to be proved
theorem sum_of_eight : S 8 = -85 := sorry

end sum_of_eight_l267_267881


namespace jim_fraction_of_tank_left_l267_267799

def jim_fraction_left (tank_capacity : ℕ) (distance_to_work : ℕ) (miles_per_gallon : ℕ) : ℚ :=
  let total_distance := 2 * distance_to_work
  let gallons_used := total_distance / miles_per_gallon
  let fraction_used := (gallons_used : ℚ) / tank_capacity
  1 - fraction_used

theorem jim_fraction_of_tank_left :
  jim_fraction_left 12 10 5 = 2 / 3 :=
by
  unfold jim_fraction_left
  norm_num
  sorry

end jim_fraction_of_tank_left_l267_267799


namespace solve_inequality_l267_267665

def otimes (x y : ℝ) : ℝ := x * (1 - y)

theorem solve_inequality (x : ℝ) : (otimes (x-2) (x+2) < 2) ↔ x ∈ Set.Iio 0 ∪ Set.Ioi 1 :=
by
  sorry

end solve_inequality_l267_267665


namespace fibonacci_remainders_l267_267057

theorem fibonacci_remainders (p : ℕ) [Fact p.Prime] :
  let F := λ n : ℕ, n.fibonacci in
   (p = 2 → (F(p) ≡ 1 [MOD p] ∧ F(p + 1) ≡ 0 [MOD p])) ∧ 
   (p = 5 → (F(p) ≡ 0 [MOD p] ∧ F(p + 1) ≡ 3 [MOD p])) ∧ 
   (p % 10 = 1 ∨ p % 10 = 9 → (F(p) ≡ 1 [MOD p] ∧ F(p + 1) ≡ 1 [MOD p])) ∧ 
   (p % 10 = 3 ∨ p % 10 = 7 → (F(p) ≡ p - 1 [MOD p] ∧ F(p + 1) ≡ 0 [MOD p])) :=
by sorry

end fibonacci_remainders_l267_267057


namespace Xiaohua_floors_when_Xiaoli_25th_l267_267064

theorem Xiaohua_floors_when_Xiaoli_25th :
  ∀ (x_li x_hua initial_li initial_hua target_li : ℕ),
    initial_li = 5 → initial_hua = 3 → x_li = 25 → 
    x_hua = initial_hua + ((x_li - initial_li) * (initial_hua - 1) / (initial_li - 1)) →
    x_hua = 13 :=
by intros x_li x_hua initial_li initial_hua target_li h1 h2 h3 h4; exact h4

end Xiaohua_floors_when_Xiaoli_25th_l267_267064


namespace ratio_limit_l267_267762

variables {a b d h : ℝ}
variables {x : ℝ} (hx : x ≤ a) (ha : a > 0) (hb : b = a + x - h)
variables (r := a + x)

noncomputable def K (x : ℝ) := (2 * r + d) * h / 2
noncomputable def R (x : ℝ) := d * h / 2

theorem ratio_limit :
  (limit (λ x, (K x) / (R x)) at_top = (2 * a + d) / d) :=
sorry

end ratio_limit_l267_267762


namespace part1_part2_l267_267227

def A (x : ℝ) : Prop := x < -3 ∨ x > 7
def B (m x : ℝ) : Prop := m + 1 ≤ x ∧ x ≤ 2 * m - 1
def complement_R_A (x : ℝ) : Prop := -3 ≤ x ∧ x ≤ 7

theorem part1 (m : ℝ) :
  (∀ x, complement_R_A x ∨ B m x → complement_R_A x) →
  m ≤ 4 :=
by
  sorry

theorem part2 (m : ℝ) (a b : ℝ) :
  (∀ x, complement_R_A x ∧ B m x ↔ (a ≤ x ∧ x ≤ b)) ∧ (b - a ≥ 1) →
  3 ≤ m ∧ m ≤ 5 :=
by
  sorry

end part1_part2_l267_267227


namespace geometric_seq_sum_l267_267837

-- Definitions of the conditions
variables {a₁ q : ℚ}
def S (n : ℕ) := a₁ * (1 - q^n) / (1 - q)

-- Hypotheses from the conditions
theorem geometric_seq_sum :
  (S 4 = -5) →
  (S 6 = 21 * S 2) →
  (S 8 = -85) :=
by
  -- Assume the given conditions
  intros h1 h2
  -- The actual proof will be inserted here
  sorry

end geometric_seq_sum_l267_267837


namespace new_area_rhombus_l267_267274

theorem new_area_rhombus (d1 d2 : ℝ) (h : (d1 * d2) / 2 = 3) : 
  ((5 * d1) * (5 * d2)) / 2 = 75 := 
by
  sorry

end new_area_rhombus_l267_267274


namespace select_1996_sets_l267_267352

theorem select_1996_sets (k : ℕ) (sets : Finset (Finset ℕ)) (h : k > 1993006) (h_sets : sets.card = k) :
  ∃ (selected_sets : Finset (Finset ℕ)), selected_sets.card = 1996 ∧
  ∀ (x y z : Finset ℕ), x ∈ selected_sets → y ∈ selected_sets → z ∈ selected_sets → z = x ∪ y → false :=
sorry

end select_1996_sets_l267_267352


namespace custom_dollar_five_neg3_l267_267659

-- Define the custom operation
def custom_dollar (a b : Int) : Int :=
  a * (b - 1) + a * b

-- State the theorem
theorem custom_dollar_five_neg3 : custom_dollar 5 (-3) = -35 := by
  sorry

end custom_dollar_five_neg3_l267_267659


namespace sum_of_eight_l267_267882

variable (a₁ : ℕ) (q : ℕ)
variable (S : ℕ → ℕ) -- Assume S is a function from natural numbers to natural numbers representing S_n

-- Condition 1: Sum of first 4 terms equals -5
axiom h1 : S 4 = -5

-- Condition 2: Sum of the first 6 terms is 21 times the sum of the first 2 terms
axiom h2 : S 6 = 21 * S 2

-- The formula for the sum of the first n terms of a geometric sequence
def sum_of_first_n_terms (n : ℕ) := a₁ * (1 - q^n) / (1 - q)

-- Statement to be proved
theorem sum_of_eight : S 8 = -85 := sorry

end sum_of_eight_l267_267882


namespace integer_values_satisfying_sqrt_condition_l267_267419

theorem integer_values_satisfying_sqrt_condition :
  {x : ℤ | 5 < Real.sqrt x ∧ Real.sqrt x < 6}.card = 10 :=
by
  sorry

end integer_values_satisfying_sqrt_condition_l267_267419


namespace percent_of_a_is_b_percent_of_d_is_c_percent_of_d_is_e_l267_267271

variables (a b c d e : ℝ)

-- Conditions
def condition1 : Prop := c = 0.25 * a
def condition2 : Prop := c = 0.50 * b
def condition3 : Prop := d = 0.40 * a
def condition4 : Prop := d = 0.20 * b
def condition5 : Prop := e = 0.35 * d
def condition6 : Prop := e = 0.15 * c

-- Proof Problem Statements
theorem percent_of_a_is_b (h1 : condition1 a c) (h2 : condition2 c b) : b = 0.5 * a := sorry

theorem percent_of_d_is_c (h1 : condition1 a c) (h3 : condition3 a d) : c = 0.625 * d := sorry

theorem percent_of_d_is_e (h5 : condition5 e d) : e = 0.35 * d := sorry

end percent_of_a_is_b_percent_of_d_is_c_percent_of_d_is_e_l267_267271


namespace round_to_nearest_hundredth_l267_267931

theorem round_to_nearest_hundredth (x : ℝ) (h : x = 24.58673) : Real.round (x * 100) / 100 = 24.59 :=
by
  have : 24.58673 * 100 = 2458.673 := by norm_num
  rw [h, this]
  norm_num
  sorry

end round_to_nearest_hundredth_l267_267931


namespace good_trapezoid_specification_l267_267270

theorem good_trapezoid_specification {A B C D S E F : Point} 
  (h1 : is_trapezoid ABCD) 
  (h2 : is_circumscribed ABCD)
  (h3 : AB ∥ CD)
  (h4 : CD < AB)
  (h5 : BS ∥ AD) 
  (h6 : tangent_from S E (circumcircle ABCD))
  (h7 : tangent_from S F (circumcircle ABCD))
  (h8 : E.on_same_side_of_line_as A CD)
  (h9 : angle_eq BSE FSC) :
  (angle_eq BAD 60° ∨ AB = AD) :=
sorry

end good_trapezoid_specification_l267_267270


namespace geometric_sequence_sum_l267_267893

variable (a1 q : ℝ) -- Define the first term and common ratio as real numbers

-- Define the sum of the first n terms of a geometric sequence
def S (n : ℕ) : ℝ := a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum :
  (S 4 a1 q = -5) → (S 6 a1 q = 21 * S 2 a1 q) → (S 8 a1 q = -85) :=
by
  intro h1 h2
  sorry

end geometric_sequence_sum_l267_267893


namespace overall_profit_percentage_l267_267591

variables {totalQuantity quantitySold18 : ℕ}
variables {profit8 profit18 : ℕ}

-- Definitions based on conditions
def totalQuantity := 1000
def quantitySold18 := 600

-- The per unit profit percentages
def profit8 := 8 -- in percent
def profit18 := 18 -- in percent

-- Equations derived from the problem conditions and solution
theorem overall_profit_percentage :
  ∀ (totalQuantity quantitySold18 : ℕ)
  (profit8 profit18 : ℕ),
  totalQuantity = 1000 →
  quantitySold18 = 600 →
  profit8 = 8 →
  profit18 = 18 →
  let quantitySold8 := totalQuantity - quantitySold18 in
  let totalProfit := (profit8 * quantitySold8 + profit18 * quantitySold18) / 100 in
  (totalProfit.toFloat / totalQuantity.toFloat * 100).round = 14 :=
by
  intros totalQuantity quantitySold18 profit8 profit18 h1 h2 h3 h4
  sorry

end overall_profit_percentage_l267_267591


namespace cyclic_points_O_B_K_F_A_l267_267326

-- Given Definitions and Hypotheses
noncomputable def inscribed_triangle (A B C O : Point) (C : Circle) :=
  C.contains A ∧ C.contains B ∧ C.contains C ∧ C.center = O

noncomputable def point_on_segment (F A B : Point) :=
  F ∈ segment A B

-- Main Proof Statement
theorem cyclic_points_O_B_K_F_A'
  (A B C O F K A' : Point)
  (C : Circle)
  (h_ins : inscribed_triangle A B C O C)
  (hFseg : point_on_segment F A B)
  (hFcond : dist A F ≤ dist A B / 2)
  (hKF : circle F A ∩ C = {K})
  (hA'A : circle F A ∩ line (A, O) = {A'}) :
  cyclic O B K F A' :=
sorry

end cyclic_points_O_B_K_F_A_l267_267326


namespace geometric_sequence_sum_eight_l267_267827

noncomputable def sum_geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ :=
  a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_eight {a1 q : ℝ} (hq : q ≠ 1) 
  (h4 : sum_geometric_sequence a1 q 4 = -5) 
  (h6 : sum_geometric_sequence a1 q 6 = 21 * sum_geometric_sequence a1 q 2) : 
  sum_geometric_sequence a1 q 8 = -85 :=
sorry

end geometric_sequence_sum_eight_l267_267827


namespace distance_between_stripes_l267_267121

/-- Given a crosswalk parallelogram with curbs 60 feet apart, a base of 20 feet, 
and each stripe of length 50 feet, show that the distance between the stripes is 24 feet. -/
theorem distance_between_stripes (h : Real) (b : Real) (s : Real) : h = 60 ∧ b = 20 ∧ s = 50 → (b * h) / s = 24 :=
by
  sorry

end distance_between_stripes_l267_267121


namespace temperature_conversion_correct_l267_267058

noncomputable def f_to_c (T : ℝ) : ℝ := (T - 32) * (5 / 9)

theorem temperature_conversion_correct :
  f_to_c 104 = 40 :=
by
  sorry

end temperature_conversion_correct_l267_267058


namespace length_BY_addition_l267_267301

open Classical

theorem length_BY_addition (ABCD : Type)
  [square : Square ABCD]
  (A B C D P Q X Y : Point ABCD)
  (DP : ℝ) (BQ : ℝ) 
  (circumcircle_APX : ∀ P Q X, Circle (Triangle A P X))
  (P_on_AD_extended : ray A D P)
  (PQ_intersect_AB : ∀ P Q, Intersect (line P Q) (line A B) Q)
  (X_perpendicular_BQ : ∀ B Q, PerpendicularFrom B (line D Q) X)
  (circumcircle : ∀ A P X, circle (Triangle A P X) Y)
  : DP = 16 / 3 → BQ = 27 → 
    ∃ p q : ℕ, BY = p / q ∧ Nat.coprime p q ∧ p + q = 65 :=
begin
  sorry
end

end length_BY_addition_l267_267301


namespace sum_good_permutations_l267_267106

theorem sum_good_permutations (n : ℕ) (h : 2 ≤ n) (p : ℕ → ℕ)
  (hp : ∀ k, 2 ≤ k → p k = 2^k - k - 1) :
  (∑ i in Finset.range n \ Finset.range 2, p i / 2 ^ i) = n - 3 + (n + 3) / 2^n :=
by
  sorry

end sum_good_permutations_l267_267106


namespace product_of_five_consecutive_is_divisible_by_sixty_l267_267520

theorem product_of_five_consecutive_is_divisible_by_sixty (n : ℤ) :
  60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end product_of_five_consecutive_is_divisible_by_sixty_l267_267520


namespace find_k_l267_267230

variables {E : Type*} [inner_product_space ℝ E]

def unit_vectors (e1 e2 : E) : Prop :=
  ∥e1∥ = 1 ∧ ∥e2∥ = 1 ∧ inner e1 e2 = -1 / 2

def a (e1 e2 : E) : E :=
  e1 - (2 : ℝ) • e2

def b (e1 e2 : E) (k : ℝ) : E :=
  k • e1 + e2

theorem find_k (e1 e2 : E) (k : ℝ) (h : unit_vectors e1 e2) :
  inner (a e1 e2) (b e1 e2 k) = 0 → k = 5 / 4 :=
sorry

end find_k_l267_267230


namespace sequence_value_2023_l267_267707

theorem sequence_value_2023 (a : ℕ → ℕ) (h₁ : a 1 = 3)
  (h₂ : ∀ m n : ℕ, a (m + n) = a m + a n) : a 2023 = 6069 := by
  sorry

end sequence_value_2023_l267_267707


namespace possible_n_values_l267_267192

theorem possible_n_values (n : ℕ) : 
  (∃ f : ℝ[X], (∀ k : ℤ, (⌊f.eval k⌋ : ℤ) = if k % n = 0 then ⌊f.eval k⌋ else 0) ∧ f.degree < n) ↔
  (n = 1 ∨ ∃ p : ℕ, p.prime ∧ ∃ α : ℕ, n = p^α) :=
by
  sorry

end possible_n_values_l267_267192


namespace output_in_scientific_notation_l267_267973

def output_kilowatt_hours : ℝ := 448000
def scientific_notation (n : ℝ) : Prop := n = 4.48 * 10^5

theorem output_in_scientific_notation : scientific_notation output_kilowatt_hours :=
by
  -- Proof steps are not required
  sorry

end output_in_scientific_notation_l267_267973


namespace sum_of_six_distinct_real_roots_l267_267388

theorem sum_of_six_distinct_real_roots (g : ℝ → ℝ) 
  (h_symm : ∀ x : ℝ, g(3 + x) = g(3 - x)) 
  (h_roots : {x : ℝ | g x = 0}.finite ∧ {x : ℝ | g x = 0}.to_finset.card = 6) : 
  ∑ r in ({x : ℝ | g x = 0}.to_finset : finset ℝ), r = 18 := 
  sorry

end sum_of_six_distinct_real_roots_l267_267388


namespace bench_cost_l267_267095

theorem bench_cost (B : ℕ) (h : B + 2 * B = 450) : B = 150 :=
by {
  sorry
}

end bench_cost_l267_267095


namespace sequence_bound_l267_267411

noncomputable def condition1 (a : ℕ → ℝ) (k : ℕ) : Prop :=
  a k - 2 * a (k + 1) + a (k + 2) ≥ 0

noncomputable def condition2 (a : ℕ → ℝ) (k : ℕ) : Prop :=
  ∑ i in finset.range k.succ, a i ≤ 1

theorem sequence_bound (a : ℕ → ℝ) (h1 : ∀ k : ℕ, condition1 a k) (h2 : ∀ k : ℕ, condition2 a k) (k : ℕ) :
  0 ≤ a k - a (k + 1) ∧ a k - a (k + 1) < 2 / (k : ℝ)^2 :=
sorry

end sequence_bound_l267_267411


namespace flowchart_result_l267_267548

def runFlowchart : ℕ := 10

theorem flowchart_result : 
  ∃ result : ℕ, 
  (result = 20 ∨ result = 6 ∨ result = 10 ∨ result = 15) ∧ 
  (runFlowchart = 10) := 
by
  use 10
  split
  · right; right; left; refl
  · refl

end flowchart_result_l267_267548


namespace starting_lineup_combinations_l267_267918

/-- We have 18 players and need to choose a starting lineup of 8, with one specific point guard. 
   Prove that the number of ways to do this is exactly 349,864. --/
theorem starting_lineup_combinations :
  let team_size := 18
  let lineup_size := 8
  let point_guard := 1
  let remaining_players := team_size - point_guard in
  let choose_point_guard := Nat.choose team_size point_guard in
  let choose_remaining := Nat.choose remaining_players 7 in
  let total_combinations := choose_point_guard * choose_remaining in
  total_combinations = 349864 := 
by
  sorry

end starting_lineup_combinations_l267_267918


namespace max_blocks_fit_in_box_l267_267545

theorem max_blocks_fit_in_box :
  let block_dims := (1 : ℕ, 1 : ℕ, 2 : ℕ)
  let box_dims := (4 : ℕ, 3 : ℕ, 2 : ℕ)
  ∀ (n : ℕ), n ≤ (box_dims.1 * box_dims.2 * box_dims.3) / (block_dims.1 * block_dims.2 * block_dims.3)
  → n = 12 :=
sorry

end max_blocks_fit_in_box_l267_267545


namespace Jolene_total_raised_l267_267802

theorem Jolene_total_raised :
  let babysitting_earnings := 4 * 30
  let car_washing_earnings := 5 * 12
  babysitting_earnings + car_washing_earnings = 180 :=
by
  let babysitting_earnings := 4 * 30
  let car_washing_earnings := 5 * 12
  calc
    babysitting_earnings + car_washing_earnings = 120 + 60 : by rfl
    ... = 180 : by rfl

end Jolene_total_raised_l267_267802


namespace hyperbola_eccentricity_is_sqrt_five_l267_267272

noncomputable def hyperbola_eccentricity (a b : ℝ) (h0 : a > 0) (h1 : b > 0) (h2 : (|a * b| / (sqrt (a^2 + b^2))) = 2 * a) : ℝ :=
  sqrt (5)

theorem hyperbola_eccentricity_is_sqrt_five {a b : ℝ} (h0 : a > 0) (h1 : b > 0) (h2 : abs (a * b) / real.sqrt (a^2 + b^2) = 2 * a) :
  hyperbola_eccentricity a b h0 h1 h2 = sqrt 5 :=
sorry

end hyperbola_eccentricity_is_sqrt_five_l267_267272


namespace cos_five_alpha_sin_five_alpha_cos_n_alpha_sin_n_alpha_l267_267071

theorem cos_five_alpha (α : ℝ) :
  cos (5 * α) = (cos α)^5 - 10 * (cos α)^3 * (sin α)^2 + 5 * cos α * (sin α)^4 :=
  sorry

theorem sin_five_alpha (α : ℝ) :
  sin (5 * α) = 5 * (cos α)^4 * sin α - 10 * (cos α)^2 * (sin α)^3 + (sin α)^5 :=
  sorry

theorem cos_n_alpha (n : ℤ) (α : ℝ) :
  cos (n * α) = ∑ k in finset.range (n+1) \ k % 2 = 0, (nat.choose n k) * (cos α)^(n-k) * (-1)^(k/2) * (sin α)^k :=
  sorry

theorem sin_n_alpha (n : ℤ) (α : ℝ) :
  sin (n * α) = ∑ k in finset.range (n+1) \ k % 2 = 1, (nat.choose n k) * (cos α)^(n-k) * (-1)^((k-1)/2) * (sin α)^k :=
  sorry

end cos_five_alpha_sin_five_alpha_cos_n_alpha_sin_n_alpha_l267_267071


namespace incorrect_equation_l267_267125

-- Define the given conditions
def speed_truck : ℝ := 40
def speed_bus : ℝ := 2.5 * speed_truck
def travel_time : ℝ := 5

-- Define the correct total distance equation
def correct_distance : ℝ := (speed_truck * travel_time) + (speed_bus * travel_time)

-- Proving which equation is incorrect
theorem incorrect_equation : 
  ∃ (eq : ℝ), eq = (speed_truck * travel_time + speed_truck * 2.5) ∧ (eq ≠ correct_distance) :=
by {
  -- Define the incorrect equation
  let incorrect_eq := speed_truck * travel_time + speed_truck * 2.5,
  -- We need to show that this is not equal to the actual distance
  use incorrect_eq,
  split,
  {
    -- Prove that the incorrect equation is as given
    refl,
  },
  {
    -- Prove that the incorrect equation is not equal to the correct equation
    intro h,
    -- Using numerical values to show they are not equal
    have h1 : incorrect_eq = 40 * 5 + 40 * 2.5, by simp [incorrect_eq, speed_truck, travel_time],
    have h2 : correct_distance = 40 * 5 + 40 * 2.5 * 5, by simp [correct_distance, speed_truck, speed_bus, travel_time],
    rw h1 at h, rw h2 at h,
    linarith, -- This would reveal a contradiction
  },
  exact sorry -- Leaving some space for the proof which will be filled by the user
}

end incorrect_equation_l267_267125


namespace B_should_be_paid_2307_69_l267_267574

noncomputable def A_work_per_day : ℚ := 1 / 15
noncomputable def B_work_per_day : ℚ := 1 / 10
noncomputable def C_work_per_day : ℚ := 1 / 20
noncomputable def combined_work_per_day : ℚ := A_work_per_day + B_work_per_day + C_work_per_day
noncomputable def total_work : ℚ := 1
noncomputable def total_wages : ℚ := 5000
noncomputable def time_taken : ℚ := total_work / combined_work_per_day
noncomputable def B_share_of_work : ℚ := B_work_per_day / combined_work_per_day
noncomputable def B_share_of_wages : ℚ := B_share_of_work * total_wages

theorem B_should_be_paid_2307_69 : B_share_of_wages = 2307.69 := by
  sorry

end B_should_be_paid_2307_69_l267_267574


namespace largest_divisor_of_product_of_five_consecutive_integers_l267_267467

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∀ (n : ℤ), ∃ (d : ℤ), d = 60 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end largest_divisor_of_product_of_five_consecutive_integers_l267_267467


namespace curlers_count_l267_267179

theorem curlers_count (T P B G : ℕ) 
  (hT : T = 16)
  (hP : P = T / 4)
  (hB : B = 2 * P)
  (hG : G = T - (P + B)) : 
  G = 4 :=
by
  sorry

end curlers_count_l267_267179


namespace factorization_l267_267157

theorem factorization (x : ℝ) : x^10 - 1024 = (x^5 + 32) * (x^5 - 32) := 
by 
  sorry

end factorization_l267_267157


namespace geometric_sequence_S8_l267_267871

theorem geometric_sequence_S8 
  (a1 q : ℝ) (S : ℕ → ℝ)
  (h1 : S 4 = a1 * (1 - q^4) / (1 - q) = -5)
  (h2 : S 6 = 21 * S 2)
  (geom_sum : ∀ n, S n = a1 * (1 - q^n) / (1 - q))
  (h3 : q ≠ 1) : S 8 = -85 := 
begin
  sorry
end

end geometric_sequence_S8_l267_267871


namespace ratio_of_divisors_l267_267336

-- Define the value of N
def N : ℕ := 46 * 46 * 81 * 450

-- (Prime factorization is not directly stated but implied in solving conditions)
noncomputable def sum_of_divisors (n : ℕ) : ℕ :=
sorry -- Implementation omitted

noncomputable def sum_of_odd_divisors (n : ℕ) : ℕ :=
sorry -- Implementation omitted

noncomputable def sum_of_even_divisors (n : ℕ) : ℕ :=
sorry -- Implementation omitted

theorem ratio_of_divisors (n : ℕ) (hn : n = N) :
  (sum_of_odd_divisors n : ℚ) / sum_of_even_divisors n = 1 / 14 :=
by
  rw hn
  sorry -- Detailed proof steps omitted.

end ratio_of_divisors_l267_267336


namespace prob_memes_given_m_l267_267176

open Probability Theory

theorem prob_memes_given_m :
  let P_word : ProbabilitySpace (Word → ℝ) := sorry
  (P_word.event {w | w = "MATHEMATICS"} = 1 / 2) →
  (P_word.event {w | w = "MEMES"} = 1 / 2) →
  let P_M_math : ProbabilitySpace (Letter → ℝ) := sorry
  (P_M_math.event {l | l = "M"} = 2 / 11) →
  let P_M_memes : ProbabilitySpace (Letter → ℝ) := sorry
  (P_M_memes.event {l | l = "M"} = 2 / 5) →
  conditional_probability (P_word) ("MEMES") (P_M_memes)
    (⋃ word ≃ "MEMES" , (word.weight "M")) = 11 / 16 :=
begin
  sorry
end

end prob_memes_given_m_l267_267176


namespace y_intercept_of_line_l267_267345

open Real

noncomputable def midpoint (p q : ℝ × ℝ) : ℝ × ℝ :=
    ((p.1 + q.1) / 2, (p.2 + q.2) / 2)

theorem y_intercept_of_line 
    (slope : ℝ) 
    (midpoint : ℝ × ℝ) 
    (y_intercept : ℝ) 
    (h_slope : slope = 1)
    (h_midpoint : midpoint = ((2 + 8) / 2, (8 + -2) / 2))
    (h_line : ∀ x y : ℝ, y = slope * x + y_intercept → 
                     x - midpoint.1 = y - midpoint.2 / slope) :
  y_intercept = -2 :=
by
  have h_midpoint_calc : midpoint = (5, 3) := by sorry
  have h_line_eq : ∀ x y, y = slope * x + y_intercept ↔ y = x - 2 := by sorry
  sorry

end y_intercept_of_line_l267_267345


namespace medians_perpendicular_area_l267_267912

variable {ABC : Type*} [real_inner_product_space ℝ ABC]

noncomputable def area_of_triangle (A B C : ABC) : ℝ :=
  1 / 2 * ∥(B - A) × (C - A)∥

theorem medians_perpendicular_area {A B C D E : ABC}
  (hD : is_median A B C D) (hE : is_median B A C E)
  (h_perp : ⟪D - A, E - B⟫ = 0)
  (h_AD : ∥D - A∥ = 15)
  (h_BE : ∥E - B∥ = 20) :
  area_of_triangle A B C = 200 := by
  sorry

end medians_perpendicular_area_l267_267912


namespace tan_alpha_plus_pi_over_4_sin_2alpha_expr_l267_267212

open Real

theorem tan_alpha_plus_pi_over_4 (α : ℝ) (h : tan α = 2) : tan (α + π / 4) = -3 :=
by
  sorry

theorem sin_2alpha_expr (α : ℝ) (h : tan α = 2) :
  (sin (2 * α)) / (sin (α) ^ 2 + sin (α) * cos (α)) = 2 / 3 :=
by
  sorry

end tan_alpha_plus_pi_over_4_sin_2alpha_expr_l267_267212


namespace regular_polygon_sides_l267_267685

theorem regular_polygon_sides (n : ℕ) (h : 2 ≤ n) (h_angle : 180 * (n - 2) / n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l267_267685


namespace arithmetic_sequence_problem_l267_267725

theorem arithmetic_sequence_problem
  (a : ℕ → ℝ)
  (h1 : ∀ n m, a n = a m + (n - m) * (a 1 - a 0))
  (h2 : a 5 + a 7 = ∫ x in 0..Real.pi, Real.sin x) :
  a 4 + 2 * a 6 + a 8 = 4 :=
by
  have I : ∫ x in 0..Real.pi, Real.sin x = 2 := sorry
  rw [h3] at h2
  sorry

end arithmetic_sequence_problem_l267_267725


namespace product_of_five_consecutive_integers_divisible_by_120_l267_267447

theorem product_of_five_consecutive_integers_divisible_by_120 (n : ℤ) : 
  120 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end product_of_five_consecutive_integers_divisible_by_120_l267_267447


namespace geometric_sequence_sum_S8_l267_267846

noncomputable def sum_geometric_seq (a q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_S8 (a q : ℝ) (h1 : q ≠ 1)
  (h2 : sum_geometric_seq a q 4 = -5)
  (h3 : sum_geometric_seq a q 6 = 21 * sum_geometric_seq a q 2) :
  sum_geometric_seq a q 8 = -85 :=
sorry

end geometric_sequence_sum_S8_l267_267846


namespace shingle_area_l267_267600

-- Definitions from conditions
def length := 10 -- uncut side length in inches
def width := 7   -- uncut side width in inches
def trapezoid_base1 := 6 -- base of the trapezoid in inches
def trapezoid_height := 2 -- height of the trapezoid in inches

-- Definition derived from conditions
def trapezoid_base2 := length - trapezoid_base1 -- the second base of the trapezoid

-- Required proof in Lean
theorem shingle_area : (length * width - (1/2 * (trapezoid_base1 + trapezoid_base2) * trapezoid_height)) = 60 := 
by
  sorry

end shingle_area_l267_267600


namespace geometric_seq_sum_l267_267842

-- Definitions of the conditions
variables {a₁ q : ℚ}
def S (n : ℕ) := a₁ * (1 - q^n) / (1 - q)

-- Hypotheses from the conditions
theorem geometric_seq_sum :
  (S 4 = -5) →
  (S 6 = 21 * S 2) →
  (S 8 = -85) :=
by
  -- Assume the given conditions
  intros h1 h2
  -- The actual proof will be inserted here
  sorry

end geometric_seq_sum_l267_267842


namespace handshakes_among_ten_women_l267_267951

theorem handshakes_among_ten_women : ∀ (w : Fin 10 → ℕ), (∀ i j, i < j → w i ≠ w j) → 
  (∑ i in Finset.range 10, i) = 45 :=
by
  intro w h_diff
  sorry

end handshakes_among_ten_women_l267_267951


namespace jinho_initial_money_l267_267318

variable (M : ℝ)

theorem jinho_initial_money :
  (M / 2 + 300) + (((M / 2 - 300) / 2) + 400) = M :=
by
  -- This proof is yet to be completed.
  sorry

end jinho_initial_money_l267_267318


namespace max_knights_is_six_l267_267286

-- Definitions used in the Lean 4 statement should appear:
-- Room can be occupied by either a knight (True for knight) or a liar (False for liar)
inductive Person : Type
| knight : Person
| liar : Person

open Person

-- Define the 3x3 grid
noncomputable def grid : Type := Matrix (Fin 3) (Fin 3) Person

-- Define neighbors
def neighbors (i j : Fin 3) : List (Fin 3 × Fin 3) :=
  [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)].filter (λ ⟨i', j'⟩, i'.val < 3 ∧ j'.val < 3)

-- Define the maximum number of knights can be in grid
def maxKnights (g : grid) : ℕ :=
  (Fin 3).fold (λ acc i, (Fin 3).fold (λ acc' j, if g i j = knight then acc' + 1 else acc') acc) 0

-- Main theorem statement
theorem max_knights_is_six {g : grid}
  (h : ∀ i j, 
       (g i j = knight →
        ∃ (ni nj : Fin 3), 
          (ni, nj) ∈ neighbors i j ∧ g ni nj = liar)) : 
  maxKnights g ≤ 6 :=
sorry

end max_knights_is_six_l267_267286


namespace geometric_sequence_sum_S8_l267_267851

noncomputable def sum_geometric_seq (a q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_S8 (a q : ℝ) (h1 : q ≠ 1)
  (h2 : sum_geometric_seq a q 4 = -5)
  (h3 : sum_geometric_seq a q 6 = 21 * sum_geometric_seq a q 2) :
  sum_geometric_seq a q 8 = -85 :=
sorry

end geometric_sequence_sum_S8_l267_267851


namespace S₈_proof_l267_267862

-- Definitions
variables {a₁ q : ℝ} {S : ℕ → ℝ}

-- Condition for the sum of the first n terms of the geometric sequence
def geom_sum (n : ℕ) : ℝ :=
  a₁ * (1 - q ^ n) / (1 - q)

-- Conditions from the problem
def S₄ := geom_sum 4 = -5
def S₆ := geom_sum 6 = 21 * geom_sum 2

-- Theorem to prove
theorem S₈_proof (h₁ : S₄) (h₂ : S₆) : geom_sum 8 = -85 :=
by
  sorry

end S₈_proof_l267_267862


namespace complex_inequality_l267_267717

open Complex

noncomputable def problem_statement : Prop :=
∀ (a b c : ℂ),
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ (a / Complex.abs a ≠ b / Complex.abs b) ⟹
    max (Complex.abs (a * c + b)) (Complex.abs (b * c + a)) ≥
    (1 / 2) * Complex.abs (a + b) * Complex.abs ((a / Complex.abs a) - (b / Complex.abs b))

theorem complex_inequality : problem_statement :=
by
  sorry

end complex_inequality_l267_267717


namespace total_students_l267_267037

theorem total_students (boys_2nd:int) (girls_2nd:int) (students_2nd:int) (students_3rd:int) (total_students:int):
  (boys_2nd = 20) ->
  (girls_2nd = 11) ->
  (students_2nd = boys_2nd + girls_2nd) ->
  (students_3rd = 2 * students_2nd) ->
  (total_students = students_2nd + students_3rd) ->
  total_students = 93 := 
begin
  intros h1 h2 h3 h4 h5,
  rw [h1, h2] at h3,
  rw h3 at h4,
  rw [h3, h4] at h5,
  exact h5,
end

end total_students_l267_267037


namespace functions_satisfying_equation_are_constants_l267_267190

theorem functions_satisfying_equation_are_constants (f g : ℝ → ℝ) :
  (∀ x y : ℝ, f (f (x + y)) = x * f y + g x) → ∃ k : ℝ, (∀ x : ℝ, f x = k) ∧ (∀ x : ℝ, g x = k * (1 - x)) :=
by
  sorry

end functions_satisfying_equation_are_constants_l267_267190


namespace count_integer_values_l267_267422

theorem count_integer_values (x : ℕ) : 5 < Real.sqrt x ∧ Real.sqrt x < 6 → 
  (finset.card (finset.filter (λ n, 5 < Real.sqrt n ∧ Real.sqrt n < 6) (finset.range 37))) = 10 :=
by
  sorry

end count_integer_values_l267_267422


namespace part_1_part_2_l267_267699

noncomputable def f (a x : ℝ) : ℝ := a - 1 / (1 + 2^x)

theorem part_1 (a : ℝ) (h1 : f a 1 + f a (-1) = 0) : a = 1 / 2 :=
by sorry

theorem part_2 : ∃ a : ℝ, ∀ x : ℝ, f a (-x) + f a x = 0 :=
by sorry

end part_1_part_2_l267_267699


namespace juice_bar_group_total_l267_267096

theorem juice_bar_group_total (total_spent : ℕ) (mango_cost : ℕ) (pineapple_cost : ℕ) 
  (spent_on_pineapple : ℕ) (num_people_total : ℕ) :
  total_spent = 94 →
  mango_cost = 5 →
  pineapple_cost = 6 →
  spent_on_pineapple = 54 →
  num_people_total = (40 / 5) + (54 / 6) →
  num_people_total = 17 :=
by {
  intros h_total h_mango h_pineapple h_pineapple_spent h_calc,
  sorry
}

end juice_bar_group_total_l267_267096


namespace geometric_sequence_S8_l267_267873

theorem geometric_sequence_S8 
  (a1 q : ℝ) (S : ℕ → ℝ)
  (h1 : S 4 = a1 * (1 - q^4) / (1 - q) = -5)
  (h2 : S 6 = 21 * S 2)
  (geom_sum : ∀ n, S n = a1 * (1 - q^n) / (1 - q))
  (h3 : q ≠ 1) : S 8 = -85 := 
begin
  sorry
end

end geometric_sequence_S8_l267_267873


namespace largest_divisor_of_product_of_five_consecutive_integers_l267_267501

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∀ (n : ℤ), ∃ k : ℤ, k = 60 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intro n
  use 60
  split
  { refl }
  { sorry }

end largest_divisor_of_product_of_five_consecutive_integers_l267_267501


namespace largest_divisor_of_5_consecutive_integers_l267_267480

theorem largest_divisor_of_5_consecutive_integers :
  ∀ (a b c d e : ℤ), 
    a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e →
    (∃ k : ℤ, k ∣ (a * b * c * d * e) ∧ k = 60) :=
by 
  intro a b c d e h
  sorry

end largest_divisor_of_5_consecutive_integers_l267_267480


namespace subcommittees_with_at_least_one_coach_l267_267410

-- Definitions based on conditions
def total_members : ℕ := 12
def total_coaches : ℕ := 5
def subcommittee_size : ℕ := 5

-- Lean statement of the problem
theorem subcommittees_with_at_least_one_coach :
  (Nat.choose total_members subcommittee_size) - (Nat.choose (total_members - total_coaches) subcommittee_size) = 771 := by
  sorry

end subcommittees_with_at_least_one_coach_l267_267410


namespace triangle_reflection_ratio_l267_267621

theorem triangle_reflection_ratio (A B C P : Type) [normed_space ℝ A] [normed_space ℝ B] [normed_space ℝ C] [normed_space ℝ P] 
  (AB BC CA : ℝ) (h_AB : AB = 13) (h_BC : BC = 16) (h_CA : CA = 9)
  (M : midpoint B C) (h_M : is_midpoint M B C)
  (h_P : reflect_point A M BC P) :
  ∃ (m n : ℕ), gcd m n = 1 ∧ 100 * m + n = 2716 := 
sorry

end triangle_reflection_ratio_l267_267621


namespace quadratic_form_rewrite_l267_267361

theorem quadratic_form_rewrite (x : ℝ) : 2 * x ^ 2 + 7 = 4 * x → 2 * x ^ 2 - 4 * x + 7 = 0 :=
by
    intro h
    linarith

end quadratic_form_rewrite_l267_267361


namespace curlers_count_l267_267177

theorem curlers_count (T P B G : ℕ) 
  (hT : T = 16)
  (hP : P = T / 4)
  (hB : B = 2 * P)
  (hG : G = T - (P + B)) : 
  G = 4 :=
by
  sorry

end curlers_count_l267_267177


namespace magnitude_of_c_is_sqrt_10_l267_267691

noncomputable def complex_magnitude (c : ℂ) : ℝ := complex.abs c

theorem magnitude_of_c_is_sqrt_10 (c : ℂ) (P : polynomial ℂ)
  (h1 : P = (X^2 - C 2 * X + C 2) * (X^2 - C c * X + C 4) * (X^2 - C 4 * X + C 8))
  (h2 : ∀ z, P.eval z = 0 → z ∈ {1 - complex.I, 1 + complex.I, 2 - 2 * complex.I, 2 + 2 * complex.I}) :
  complex_magnitude c = real.sqrt 10 :=
sorry

end magnitude_of_c_is_sqrt_10_l267_267691


namespace ferry_tourists_total_l267_267580

/-- The number of tourists a ferry takes to the island given the ferry schedule and tourist decrement. -/
theorem ferry_tourists_total :
  let trips_per_day := 15 in
  let initial_tourists := 100 in
  let decrement_per_trip := 2 in
  let total_tourists :=
    trips_per_day * initial_tourists - decrement_per_trip * (trips_per_day * (trips_per_day - 1)) / 2
  in
  total_tourists = 1290 := 
by
  sorry

end ferry_tourists_total_l267_267580


namespace large_green_curlers_l267_267182

-- Define the number of total curlers
def total_curlers : ℕ := 16

-- Define the fraction for pink curlers
def pink_fraction : ℕ := 1 / 4

-- Define the number of pink curlers
def pink_curlers : ℕ := pink_fraction * total_curlers

-- Define the number of blue curlers
def blue_curlers : ℕ := 2 * pink_curlers

-- Define the total number of pink and blue curlers
def pink_and_blue_curlers : ℕ := pink_curlers + blue_curlers

-- Define the number of green curlers
def green_curlers : ℕ := total_curlers - pink_and_blue_curlers

-- Theorem stating the number of green curlers is 4
theorem large_green_curlers : green_curlers = 4 := by
  -- Proof would go here
  sorry

end large_green_curlers_l267_267182


namespace trig_identity_l267_267676

theorem trig_identity : cos (75 * π / 180) * cos (15 * π / 180) - sin (75 * π / 180) * sin (15 * π / 180) = 0 := by
  sorry

end trig_identity_l267_267676


namespace area_triangle_constant_l267_267771

open Real

theorem area_triangle_constant (x y : ℝ) (hx : x > 0) (hy : y = 1 / x) :
  let P := (x, y)
      tangent_line := λ x1, y - (1 / x) = -(1 / x^2) * (x1 - x)
      A := (0, 2 / x)
      B := (2 * x, 0)
      O := (0, 0) in
  let area_OAB := 1 / 2 * 2 / x * 2 * x in
  area_OAB = 2 := by
  sorry

end area_triangle_constant_l267_267771


namespace initial_number_of_balls_l267_267432

theorem initial_number_of_balls (T B : ℕ) (P : ℚ) (after3_blue : ℕ) (prob : ℚ) :
  B = 7 → after3_blue = B - 3 → prob = after3_blue / T → prob = 1/3 → T = 15 :=
by
  sorry

end initial_number_of_balls_l267_267432


namespace Jolene_raised_total_money_l267_267804

-- Definitions for the conditions
def babysits_earning_per_family : ℤ := 30
def number_of_families : ℤ := 4
def cars_earning_per_car : ℤ := 12
def number_of_cars : ℤ := 5

-- Calculation of total earnings
def babysitting_earnings : ℤ := babysits_earning_per_family * number_of_families
def car_washing_earnings : ℤ := cars_earning_per_car * number_of_cars
def total_earnings : ℤ := babysitting_earnings + car_washing_earnings

-- The proof statement
theorem Jolene_raised_total_money : total_earnings = 180 := by
  sorry

end Jolene_raised_total_money_l267_267804


namespace pizza_combinations_l267_267108

/-- The number of unique pizzas that can be made with exactly 5 toppings from a selection of 8 is 56. -/
theorem pizza_combinations : (Nat.choose 8 5) = 56 := by
  sorry

end pizza_combinations_l267_267108


namespace ab_cd_is_1_or_minus_1_l267_267794

theorem ab_cd_is_1_or_minus_1 (a b c d : ℤ) (h1 : ∃ k₁ : ℤ, a = k₁ * (a * b - c * d))
  (h2 : ∃ k₂ : ℤ, b = k₂ * (a * b - c * d)) (h3 : ∃ k₃ : ℤ, c = k₃ * (a * b - c * d))
  (h4 : ∃ k₄ : ℤ, d = k₄ * (a * b - c * d)) :
  a * b - c * d = 1 ∨ a * b - c * d = -1 := 
sorry

end ab_cd_is_1_or_minus_1_l267_267794


namespace regular_polygon_sides_l267_267609

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, interior_angle n i = 144) : n = 10 :=
sorry

end regular_polygon_sides_l267_267609


namespace rate_grapes_l267_267146

/-- Given that Bruce purchased 8 kg of grapes at a rate G per kg, 8 kg of mangoes at the rate of 55 per kg, 
and paid a total of 1000 to the shopkeeper, prove that the rate per kg for the grapes (G) is 70. -/
theorem rate_grapes (G : ℝ) (h1 : 8 * G + 8 * 55 = 1000) : G = 70 :=
by 
  sorry

end rate_grapes_l267_267146


namespace randy_spent_fraction_on_ice_cream_l267_267359

theorem randy_spent_fraction_on_ice_cream :
  ∀ (initial_money money_spent_on_lunch cost_of_ice_cream : ℝ),
    initial_money = 30 →
    money_spent_on_lunch = 10 →
    cost_of_ice_cream = 5 →
    (cost_of_ice_cream / (initial_money - money_spent_on_lunch) = 1/4) :=
by
  intros initial_money money_spent_on_lunch cost_of_ice_cream h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end randy_spent_fraction_on_ice_cream_l267_267359


namespace brandon_businesses_l267_267142

theorem brandon_businesses (total_businesses: ℕ) (fire_fraction: ℚ) (quit_fraction: ℚ) 
  (h_total: total_businesses = 72) 
  (h_fire_fraction: fire_fraction = 1/2) 
  (h_quit_fraction: quit_fraction = 1/3) : 
  total_businesses - (total_businesses * fire_fraction + total_businesses * quit_fraction) = 12 :=
by 
  sorry

end brandon_businesses_l267_267142


namespace average_class_size_l267_267034

theorem average_class_size 
  (three_year_olds : ℕ := 13)
  (four_year_olds : ℕ := 20)
  (five_year_olds : ℕ := 15)
  (six_year_olds : ℕ := 22) : 
  ((three_year_olds + four_year_olds + five_year_olds + six_year_olds) / 2) = 35 := 
by
  sorry

end average_class_size_l267_267034


namespace geometric_sequence_sum_l267_267891

variable (a1 q : ℝ) -- Define the first term and common ratio as real numbers

-- Define the sum of the first n terms of a geometric sequence
def S (n : ℕ) : ℝ := a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum :
  (S 4 a1 q = -5) → (S 6 a1 q = 21 * S 2 a1 q) → (S 8 a1 q = -85) :=
by
  intro h1 h2
  sorry

end geometric_sequence_sum_l267_267891


namespace largest_divisor_of_product_of_five_consecutive_integers_l267_267473

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∀ (n : ℤ), ∃ (d : ℤ), d = 60 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end largest_divisor_of_product_of_five_consecutive_integers_l267_267473


namespace juice_bar_group_total_l267_267098

theorem juice_bar_group_total (total_spent : ℕ) (mango_cost : ℕ) (pineapple_cost : ℕ) 
  (spent_on_pineapple : ℕ) (num_people_total : ℕ) :
  total_spent = 94 →
  mango_cost = 5 →
  pineapple_cost = 6 →
  spent_on_pineapple = 54 →
  num_people_total = (40 / 5) + (54 / 6) →
  num_people_total = 17 :=
by {
  intros h_total h_mango h_pineapple h_pineapple_spent h_calc,
  sorry
}

end juice_bar_group_total_l267_267098


namespace imaginary_part_of_complex_division_l267_267964

theorem imaginary_part_of_complex_division :
  let i := complex.I in
  complex.im (i^3 / (2 * i - 1)) = 1 / 5 :=
by sorry

end imaginary_part_of_complex_division_l267_267964


namespace geometric_sequence_sum_S8_l267_267847

noncomputable def sum_geometric_seq (a q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_S8 (a q : ℝ) (h1 : q ≠ 1)
  (h2 : sum_geometric_seq a q 4 = -5)
  (h3 : sum_geometric_seq a q 6 = 21 * sum_geometric_seq a q 2) :
  sum_geometric_seq a q 8 = -85 :=
sorry

end geometric_sequence_sum_S8_l267_267847


namespace money_original_l267_267349

-- Definitions based on conditions
def john_initial := 2.50
def richard_initial := 1.50

def john_final := 3.50
def richard_final := 3.00

def transaction (j : ℝ) (r : ℝ) : Prop :=
  (j - (r + j) = john_final) ∧ (2 * r + 2 * j = 6.0)

theorem money_original (J R : ℝ) :
  transaction J R →
  J = john_initial ∧ R = richard_initial :=
by
  intro h
  sorry

end money_original_l267_267349


namespace output_in_scientific_notation_l267_267974

def output_kilowatt_hours : ℝ := 448000
def scientific_notation (n : ℝ) : Prop := n = 4.48 * 10^5

theorem output_in_scientific_notation : scientific_notation output_kilowatt_hours :=
by
  -- Proof steps are not required
  sorry

end output_in_scientific_notation_l267_267974


namespace product_of_five_consecutive_is_divisible_by_sixty_l267_267523

theorem product_of_five_consecutive_is_divisible_by_sixty (n : ℤ) :
  60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end product_of_five_consecutive_is_divisible_by_sixty_l267_267523


namespace value_of_k_l267_267390

theorem value_of_k :
  ∃ k : ℝ, (∀ x y : ℝ, 2*x - y = k ↔ (4, 6) ∈ set_of (λ p : ℝ × ℝ, 2 * p.1 - p.2 = k)) ∧
            (∀ x y : ℝ, (x, y) = (2, 4) ∨ (x, y) = (6, 8) ∨ (x, y) = (4, 6) →
                         2*x - y = k ∧ (x - 4) * (4 - x) + (y - 6) * (6 - y) ≤ 0) :=
begin
  use 2,
  split,
  { intros x y,
    split,
    { intro h1, rw h1 at h1, exact ⟨4, 6⟩ },
    { intro h2, exact h2 } },
  { intros x y h3,
    cases h3,
    { cases h3,
      exact ⟨(4, 6), by refl⟩ } },
  sorry
end

end value_of_k_l267_267390


namespace infinite_product_equals_root_l267_267650

theorem infinite_product_equals_root :
  (∏ n in (finset.range 1).filter (λ n, n ≠ 0), (3 ^ n) ^ (1 / n ^ 2)) = real.sqrt (real.sqrt 27) :=
sorry

end infinite_product_equals_root_l267_267650


namespace solve_for_x_l267_267002

theorem solve_for_x : ∃ x : ℚ, (2/3 - 1/4) = 1/x ∧ x = 12/5 :=
by
  use 12/5
  split
  · norm_num
  · norm_num
  · sorry

end solve_for_x_l267_267002


namespace product_of_five_consecutive_integers_divisible_by_120_l267_267449

theorem product_of_five_consecutive_integers_divisible_by_120 (n : ℤ) : 
  120 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end product_of_five_consecutive_integers_divisible_by_120_l267_267449


namespace product_of_5_consecutive_integers_divisible_by_60_l267_267541

theorem product_of_5_consecutive_integers_divisible_by_60 :
  ∀a : ℤ, 60 ∣ (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
by
  sorry

end product_of_5_consecutive_integers_divisible_by_60_l267_267541


namespace geometric_sequence_sum_S8_l267_267853

noncomputable def sum_geometric_seq (a q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_S8 (a q : ℝ) (h1 : q ≠ 1)
  (h2 : sum_geometric_seq a q 4 = -5)
  (h3 : sum_geometric_seq a q 6 = 21 * sum_geometric_seq a q 2) :
  sum_geometric_seq a q 8 = -85 :=
sorry

end geometric_sequence_sum_S8_l267_267853


namespace tan_sec_inequality_l267_267786

variable (A B C : ℝ)
variable (x y z : ℝ)

-- Assuming the internal angles A, B, C of a triangle
axiom angle_sum : A + B + C = π

-- Definitions as per the problem
def cos_half_angle_a : ℝ := cos (A / 2)
def cos_half_angle_b : ℝ := cos (B / 2)
def cos_half_angle_c : ℝ := cos (C / 2)

-- Assertions corresponding to conditions
axiom angle_a_half : x = cos_half_angle_a A
axiom angle_b_half : y = cos_half_angle_b B
axiom angle_c_half : z = cos_half_angle_c C

theorem tan_sec_inequality :
  (tan (A / 2) + tan (B / 2) + tan (C / 2)) >= 
  (1 / 2) * (sec (A / 2) + sec (B / 2) + sec (C / 2)) := 
sorry

end tan_sec_inequality_l267_267786


namespace third_frog_jump_length_l267_267436

-- Definitions based on conditions
def FrogJump (a b c : ℝ) : ℝ :=
  let jump_length_b := 60
  let distance_ac := 40
  let distance_ba := distance_ac / 2
  distance_ba + distance_ac / 2

-- Statement of the problem
theorem third_frog_jump_length : FrogJump 0 60 40 = 30 :=
  sorry

end third_frog_jump_length_l267_267436


namespace product_is_zero_l267_267187

theorem product_is_zero : 
  (∏ n in Finset.range 351, (n^3 - (350 - n))) = 0 :=
by
  sorry

end product_is_zero_l267_267187


namespace largest_divisor_of_product_of_five_consecutive_integers_l267_267503

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∀ (n : ℤ), ∃ k : ℤ, k = 60 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intro n
  use 60
  split
  { refl }
  { sorry }

end largest_divisor_of_product_of_five_consecutive_integers_l267_267503


namespace angle_P_of_extended_sides_l267_267365

theorem angle_P_of_extended_sides (ABCDEFGH : Type) [RegularOctagon ABCDEFGH]
  (A B G H : Point)
  (AB GH : Line)
  (P : Point)
  (ext_AB : extended AB P)
  (ext_GH : extended GH P) :
  measure_angle P = 90 :=
sorry

end angle_P_of_extended_sides_l267_267365


namespace convert_to_scientific_notation_l267_267976

theorem convert_to_scientific_notation :
  (448000 : ℝ) = 4.48 * 10^5 :=
by
  sorry

end convert_to_scientific_notation_l267_267976


namespace product_of_5_consecutive_integers_divisible_by_60_l267_267540

theorem product_of_5_consecutive_integers_divisible_by_60 :
  ∀a : ℤ, 60 ∣ (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
by
  sorry

end product_of_5_consecutive_integers_divisible_by_60_l267_267540


namespace find_radius_of_cylinder_l267_267617

noncomputable def radius_of_cylinder : ℚ :=
  let d_cylinder := 2 * 2  -- Diameter of the cylinder is equal to the height: 2r where r = 2 * r
  let r_cone := 4  -- Radius of the cone
  let h_cone := 10  -- Height of the cone
  let cone_ratio := λ r (h : ℚ) => (h_cone - h) / r -- Ratio from the geometry
  r_cone / cone_ratio 2 (2 * r_cone / cone_ratio 2 r_cone * / 2) -- Simplified using similar triangles

theorem find_radius_of_cylinder :
  radius_of_cylinder = (20 / 9 : ℚ) :=
sorry

end find_radius_of_cylinder_l267_267617


namespace solve_determinant_l267_267701

theorem solve_determinant (a x : ℝ) (h : a ≠ 0) :
    det ![![2*x + a, x, x], ![x, x + a, x], ![x, x, x + a]] = 0 →
    (x = -a / 2 ∨ x = a / Real.sqrt 2 ∨ x = -a / Real.sqrt 2) := by
  sorry

end solve_determinant_l267_267701


namespace largest_divisor_of_five_consecutive_integers_l267_267464

open Nat

theorem largest_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, k ∈ {n, n+1, n+2, n+3, n+4} ∧
    ∀ m ∈ {2, 3, 4, 5}, m ∣ k → 60 ∣ (n * (n+1) * (n+2) * (n+3) * (n+4)) := 
sorry

end largest_divisor_of_five_consecutive_integers_l267_267464


namespace distinct_arrangements_SUCCESS_l267_267263

theorem distinct_arrangements_SUCCESS : 
  let n := 7
  let n_S := 3
  let n_C := 2
  let n_U := 1
  let n_E := 1
  (fact n) / (fact n_S * fact n_C * fact n_U * fact n_E) = 420 :=
by
  sorry

end distinct_arrangements_SUCCESS_l267_267263


namespace range_sinx_pow6_cosx_pow4_l267_267134

open Real

-- Define the function f(x) = sin^6(x) + cos^4(x)
noncomputable def f (x : ℝ) : ℝ := (sin x) ^ 6 + (cos x) ^ 4

-- Prove that the range of f(x) is [0, 1]
theorem range_sinx_pow6_cosx_pow4 : ∀ x : ℝ, 0 ≤ f x ∧ f x ≤ 1 :=
by
  intro x
  -- The actual proof will be here
  sorry

end range_sinx_pow6_cosx_pow4_l267_267134


namespace area_of_ABCD_l267_267781

variable {A B C D E F : Type}
variable {area : Π {X Y Z : Type}, Prop}
variable {m n : ℝ}

-- Definition of the areas
definition area_ADF : Prop := area A D F 
definition area_AECF : Prop := area A E C ∧ area F E C

-- Given conditions
axiom h1 : ∀ {X Y Z : Type}, area (X Y Z) = true
axiom ratio_DF_FC : DF = FC
axiom ratio_CE_EB : CE = 2 * EB
axiom area_ADF_is_m : area A D F = m
axiom area_AECF_is_n : area A E C ∧ area F E C = n

theorem area_of_ABCD (h : n > m) : 
  area A B C D = (3/2) * n + (1/2) * m :=
by 
  sorry

end area_of_ABCD_l267_267781


namespace find_w_l267_267027

noncomputable def proj (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let dp := (a.1 * b.1 + a.2 * b.2 + a.3 * b.3)
  let b_sq := (b.1 ^ 2 + b.2 ^ 2 + b.3 ^ 2)
  (dp / b_sq) • b

theorem find_w
  (w : ℝ)
  (proj_cond : proj (1, 4, w) (2, -3, 6) = (5 / 49) • (2, -3, 6)) :
  w = 5 / 2 :=
by
  sorry

end find_w_l267_267027


namespace bela_always_wins_l267_267140

theorem bela_always_wins (n : ℕ) (hn : n > 4) : 
  ∃ strategy : ∀ (turn : ℕ) (choices : set ℝ), Real, 
    (∀ (turn : ℕ) (choices : set ℝ), 
      (strategy turn choices ∈ set.Icc (0 : ℝ) n) 
      ∧ 
      ∀ (c : ℝ), c ∈ choices → abs (strategy turn choices - c) > 1) 
    ∧ 
    (∀ (turn : ℕ), 
      (turn % 2 = 0 → ∃ choice, choice ∈ choices ∧ choice = strategy turn choices) 
      → bela_wins (strategy turn choices)) :=
sorry

end bela_always_wins_l267_267140


namespace tangent_slope_at_one_one_l267_267028

noncomputable def curve (x : ℝ) : ℝ := x * Real.exp (x - 1)

theorem tangent_slope_at_one_one : (deriv curve 1) = 2 := 
sorry

end tangent_slope_at_one_one_l267_267028


namespace sqrt_sequence_solution_l267_267183

theorem sqrt_sequence_solution : 
  (∃ x : ℝ, x = sqrt (18 + x) ∧ x > 0) → (∃ x : ℝ, x = 6) :=
by
  assume h,
  sorry

end sqrt_sequence_solution_l267_267183


namespace simplify_expression_l267_267000

noncomputable def simplify_fraction (x : ℝ) (h : x ≠ 2) : ℝ :=
  (1 + (1 / (x - 2))) / ((x - x^2) / (x - 2))

theorem simplify_expression (x : ℝ) (h : x ≠ 2) : simplify_fraction x h = -(x - 1) / x :=
  sorry

end simplify_expression_l267_267000


namespace sum_of_3digit_numbers_remainder_2_div_3_l267_267550

theorem sum_of_3digit_numbers_remainder_2_div_3 : 
  (∑ k in finset.Icc 102 998, if (k % 3 = 2) then k else 0) = 164450 := 
by sorry

end sum_of_3digit_numbers_remainder_2_div_3_l267_267550


namespace largest_divisor_of_product_of_five_consecutive_integers_l267_267498

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∀ (n : ℤ), ∃ k : ℤ, k = 60 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intro n
  use 60
  split
  { refl }
  { sorry }

end largest_divisor_of_product_of_five_consecutive_integers_l267_267498


namespace shop_b_tvs_l267_267041

noncomputable def shop_a : ℕ := 20
noncomputable def shop_c : ℕ := 60
noncomputable def shop_d : ℕ := 80
noncomputable def shop_e : ℕ := 50
noncomputable def total_shops : ℕ := 5
noncomputable def average_tvs : ℕ := 48
noncomputable def total_tvs : ℕ := average_tvs * total_shops

theorem shop_b_tvs : 
  shop_a + shop_c + shop_d + shop_e = 210 →
  total_tvs - (shop_a + shop_c + shop_d + shop_e) = 30 :=
by 
  intros h
  rw [←h, total_tvs]
  sorry

end shop_b_tvs_l267_267041


namespace problem_probability_l267_267761

noncomputable def probability_no_one_gets_their_ball : ℚ :=
  let total_ways := 4! in
  let favorable_ways := 9 in
  favorable_ways / total_ways

theorem problem_probability :
  probability_no_one_gets_their_ball = 3 / 8 :=
by
  sorry

end problem_probability_l267_267761


namespace S₈_proof_l267_267863

-- Definitions
variables {a₁ q : ℝ} {S : ℕ → ℝ}

-- Condition for the sum of the first n terms of the geometric sequence
def geom_sum (n : ℕ) : ℝ :=
  a₁ * (1 - q ^ n) / (1 - q)

-- Conditions from the problem
def S₄ := geom_sum 4 = -5
def S₆ := geom_sum 6 = 21 * geom_sum 2

-- Theorem to prove
theorem S₈_proof (h₁ : S₄) (h₂ : S₆) : geom_sum 8 = -85 :=
by
  sorry

end S₈_proof_l267_267863


namespace james_profit_l267_267312

/--
  Prove that James's profit from buying 200 lotto tickets at $2 each, given the 
  conditions about winning tickets, is $4,830.
-/
theorem james_profit 
  (total_tickets : ℕ := 200)
  (cost_per_ticket : ℕ := 2)
  (winner_percentage : ℝ := 0.2)
  (five_dollar_win_pct : ℝ := 0.8)
  (grand_prize : ℝ := 5000)
  (average_other_wins : ℝ := 10) :
  let total_cost := total_tickets * cost_per_ticket 
  let total_winners := winner_percentage * total_tickets
  let five_dollar_winners := five_dollar_win_pct * total_winners
  let total_five_dollar := five_dollar_winners * 5
  let remaining_winners := total_winners - 1 - five_dollar_winners
  let total_remaining_winners := remaining_winners * average_other_wins
  let total_winnings := total_five_dollar + grand_prize + total_remaining_winners
  let profit := total_winnings - total_cost
  profit = 4830 :=
by
  sorry

end james_profit_l267_267312


namespace problem_statement_l267_267714

theorem problem_statement (a b c : ℝ) (h1 : 0 < a) (h2 : a < 2)
    (h3 : 0 < b) (h4 : b < 2) (h5 : 0 < c) (h6 : c < 2) :
    ¬ ((2 - a) * b > 1 ∧ (2 - b) * c > 1 ∧ (2 - c) * a > 1) :=
by
  sorry

end problem_statement_l267_267714


namespace average_class_size_l267_267032

theorem average_class_size 
  (num_3_year_olds : ℕ) 
  (num_4_year_olds : ℕ) 
  (num_5_year_olds : ℕ) 
  (num_6_year_olds : ℕ) 
  (class_size_3_and_4 : num_3_year_olds = 13 ∧ num_4_year_olds = 20) 
  (class_size_5_and_6 : num_5_year_olds = 15 ∧ num_6_year_olds = 22) :
  (num_3_year_olds + num_4_year_olds + num_5_year_olds + num_6_year_olds) / 2 = 35 :=
by
  sorry

end average_class_size_l267_267032


namespace variance_comparison_l267_267620

-- Define the scores of Player A and Player B
def scores_A : List ℕ := [5, 10, 9, 3, 8]
def scores_B : List ℕ := [8, 6, 8, 6, 7]

/-- Variance of a list of natural numbers -/
noncomputable def variance (l : List ℕ) : ℝ := 
  let n := l.length
  let mean := (l.map (λ x => (x : ℝ))).sum / n
  let sq_diff := l.map (λ x => ((x : ℝ) - mean) ^ 2)
  (sq_diff.sum) / n

def S_A2 := variance scores_A
def S_B2 := variance scores_B

-- The theorem statement
theorem variance_comparison : S_A2 > S_B2 := 
  sorry

end variance_comparison_l267_267620


namespace inequality_condition_l267_267697

theorem inequality_condition (a b x : ℝ) (h1 : a < 0) (h2 : b > 0) :
  (|a - |x - 1|| + | |x - 1| - b| ≥ |a - b|) ↔ (1 - b ≤ x ∧ x ≤ 1 + b) :=
  sorry

end inequality_condition_l267_267697


namespace total_degree_difference_l267_267149

-- Definitions based on conditions
def timeStart : ℕ := 12 * 60  -- noon in minutes
def timeEnd : ℕ := 14 * 60 + 30  -- 2:30 PM in minutes
def numTimeZones : ℕ := 3  -- Three time zones
def degreesInCircle : ℕ := 360  -- Degrees in a full circle

-- Calculate degrees moved by each hand
def degreesMovedByHourHand : ℚ := (timeEnd - timeStart) / (12 * 60) * degreesInCircle
def degreesMovedByMinuteHand : ℚ := (timeEnd - timeStart) % 60 * (degreesInCircle / 60)
def degreesMovedBySecondHand : ℕ := 0  -- At 2:30 PM, second hand is at initial position

-- Calculate total degree difference for all three hands and time zones
def totalDegrees : ℚ := 
  (degreesMovedByHourHand + degreesMovedByMinuteHand + degreesMovedBySecondHand) * numTimeZones

-- Theorem statement to prove
theorem total_degree_difference :
  totalDegrees = 765 := by
  sorry

end total_degree_difference_l267_267149


namespace regular_polygon_sides_l267_267615

theorem regular_polygon_sides (n : ℕ) (h : n ≥ 3) (interior_angle : ℝ) : 
  interior_angle = 144 → n = 10 :=
by
  intro h1
  sorry

end regular_polygon_sides_l267_267615


namespace number_of_birds_is_400_l267_267402

-- Definitions of the problem
def num_stones : ℕ := 40
def num_trees : ℕ := 3 * num_stones + num_stones
def combined_trees_stones : ℕ := num_trees + num_stones
def num_birds : ℕ := 2 * combined_trees_stones

-- Statement to prove
theorem number_of_birds_is_400 : num_birds = 400 := by
  sorry

end number_of_birds_is_400_l267_267402


namespace find_c_l267_267721

variable (a b c x : ℝ)

-- Given the conditions:
def cond1 := x = 2 → x - b = 0
def cond2 := x = 0.5 → (2 * x + a) / (x - b) = 0
def cond3 := (2 * x + a) / (x - b) = 3 → x = c

-- We need to prove that c = 5.
theorem find_c (h1 : cond1) (h2 : cond2) (h3 : cond3) : c = 5 := by
  sorry

end find_c_l267_267721


namespace total_number_of_students_l267_267986

theorem total_number_of_students (T G : ℕ) (h1 : 50 + G = T) (h2 : G = 50 * T / 100) : T = 100 :=
  sorry

end total_number_of_students_l267_267986


namespace range_of_2x_minus_y_l267_267209

variable {x y : ℝ}

theorem range_of_2x_minus_y (h1 : 2 < x) (h2 : x < 4) (h3 : -1 < y) (h4 : y < 3) :
  ∃ (a b : ℝ), (1 < a) ∧ (a < 2 * x - y) ∧ (2 * x - y < b) ∧ (b < 9) :=
by
  sorry

end range_of_2x_minus_y_l267_267209


namespace baron_munchausen_claim_l267_267637

-- Given conditions and question:
def weight_partition_problem (weights : Finset ℕ) (h_card : weights.card = 50) (h_distinct : ∀ w ∈ weights,  1 ≤ w ∧ w ≤ 100) (h_sum_even : weights.sum id % 2 = 0) : Prop :=
  ¬(∃ (s1 s2 : Finset ℕ), s1 ∪ s2 = weights ∧ s1 ∩ s2 = ∅ ∧ s1.sum id = s2.sum id)

-- We need to prove that the above statement is true.
theorem baron_munchausen_claim :
  ∀ (weights : Finset ℕ), weights.card = 50 ∧ (∀ w ∈ weights, 1 ≤ w ∧ w ≤ 100) ∧ weights.sum id % 2 = 0 → weight_partition_problem weights (by sorry) (by sorry) (by sorry) :=
sorry

end baron_munchausen_claim_l267_267637


namespace completing_square_solution_l267_267010

theorem completing_square_solution (x : ℝ) : x^2 + 4 * x - 1 = 0 → (x + 2)^2 = 5 :=
by
  sorry

end completing_square_solution_l267_267010


namespace ring_coverage_l267_267998

noncomputable def radar_placement (r θ : ℝ) : ℝ := r / θ.sin

noncomputable def ring_area (r θ : ℝ) : ℝ := 2 * r * 11 * θ.tan / θ.sin

theorem ring_coverage (r : ℝ) (θ : ℝ) (h1 : r = 61)
  (h2 :  2 * r / θ.sin = 2 * r * 11 * θ.tan / θ.sin)
  : radar_placement 60 (20 * π / 180) = 60 / (sin (20 * π / 180)) ∧
    ring_area (60) (20 * π / 180) = 2640 * π / (tan (20 * π / 180)) :=
by
  sorry

end ring_coverage_l267_267998


namespace log_expression_l267_267715

theorem log_expression (a : ℝ) (h : 2^a = 3) : a = Real.log 3 / Real.log 2 ∧ Real.log 12 / Real.log 3 - Real.log 6 / Real.log 3 = 1 / a :=
by
  split
  { -- Prove a = Real.log 3 / Real.log 2
    sorry
  }
  { -- Prove Real.log 12 / Real.log 3 - Real.log 6 / Real.log 3 = 1 / a
    sorry
  }

end log_expression_l267_267715


namespace combined_salaries_of_A_B_D_E_l267_267984

theorem combined_salaries_of_A_B_D_E 
  (C_salary : ℕ) 
  (avg_salary : ℕ) 
  (total_individuals : ℕ) 
  (h1 : C_salary = 11000)
  (h2 : avg_salary = 8400)
  (h3 : total_individuals = 5) : 
  ∃ (combined_ABDE_salary : ℕ), combined_ABDE_salary = 31000 := 
by
  -- Total salary calculation
  have total_salary : ℕ := avg_salary * total_individuals,
  -- Combined salaries of A, B, D, and E calculation
  have combined_ABDE_salary := total_salary - C_salary,
  use combined_ABDE_salary,
  -- Proof step
  sorry

end combined_salaries_of_A_B_D_E_l267_267984


namespace minimum_n_l267_267689

theorem minimum_n (n : ℕ) (x : ℕ → ℝ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ n → |x i| < 1)
  (h2 : ∑ i in finset.range n, |x i| = 2005 + |∑ i in finset.range n, x i|) : n ≥ 2006 :=
sorry

end minimum_n_l267_267689


namespace xiaoming_journey_to_school_l267_267065

-- Definitions based on conditions
def total_time := 20 -- total time in minutes
def bike_speed := 200 -- speed on bike in meters per minute
def walk_speed := 70 -- walking speed in meters per minute
def total_distance := 3350 -- total distance in meters

def x : ℝ -- time spent riding bike in minutes
def y : ℝ -- time spent walking in minutes

-- The system of equations representing Xiao Ming's journey
theorem xiaoming_journey_to_school : 
  (x + y = total_time) ∧ (bike_speed * x + walk_speed * y = total_distance) := 
  sorry

end xiaoming_journey_to_school_l267_267065


namespace geometric_sequence_sum_l267_267886

variable (a1 q : ℝ) -- Define the first term and common ratio as real numbers

-- Define the sum of the first n terms of a geometric sequence
def S (n : ℕ) : ℝ := a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum :
  (S 4 a1 q = -5) → (S 6 a1 q = 21 * S 2 a1 q) → (S 8 a1 q = -85) :=
by
  intro h1 h2
  sorry

end geometric_sequence_sum_l267_267886


namespace total_boys_in_school_l267_267297

theorem total_boys_in_school (B : ℝ) (Muslims Hindus Sikhs Other : ℝ) (h1 : Muslims/B = 0.44) 
  (h2 : Hindus/B = 0.14) (h3 : Sikhs/B = 0.10) (h4 : Other = 272) : B = 850 :=
by 
  have h5 : (Muslims + Hindus + Sikhs + Other = B),
  -- Use the fact that the total percentage for Muslims, Hindus and Sikhs is 0.68
  have h6 : 0.32 * B = Other,
  -- Converting the percentage relationship to the total number B
  sorry

end total_boys_in_school_l267_267297


namespace largest_divisor_of_5_consecutive_integers_l267_267477

theorem largest_divisor_of_5_consecutive_integers :
  ∀ (a b c d e : ℤ), 
    a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e →
    (∃ k : ℤ, k ∣ (a * b * c * d * e) ∧ k = 60) :=
by 
  intro a b c d e h
  sorry

end largest_divisor_of_5_consecutive_integers_l267_267477


namespace yanna_baked_butter_cookies_in_morning_l267_267559

-- Define the conditions
def biscuits_morning : ℕ := 40
def biscuits_afternoon : ℕ := 20
def cookies_afternoon : ℕ := 10
def total_more_biscuits : ℕ := 30

-- Define the statement to be proved
theorem yanna_baked_butter_cookies_in_morning (B : ℕ) : 
  (biscuits_morning + biscuits_afternoon = (B + cookies_afternoon) + total_more_biscuits) → B = 20 :=
by
  sorry

end yanna_baked_butter_cookies_in_morning_l267_267559


namespace five_consecutive_product_div_24_l267_267493

theorem five_consecutive_product_div_24 (n : ℤ) : 
  24 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) := 
sorry

end five_consecutive_product_div_24_l267_267493


namespace value_of_x_l267_267947

theorem value_of_x : 
  let x := (sqrt (7^2 + 24^2)) / (sqrt (49 + 16)) 
  in x = 25 * sqrt 65 / 65 
  := 
  sorry

end value_of_x_l267_267947


namespace max_ships_on_grid_l267_267994

/-- There are 1x2 ships that can be placed on a 10x10 grid without touching each other
(even at a single point). Prove that the maximum number of such ships is 13. -/
theorem max_ships_on_grid : 
  ∃ k : ℕ, k = 13 ∧ (∀ placement : matrix (fin 10) (fin 10) (option (fin 2)),
    (∀ i j, placement i j → placement i (j + 1) = placement i j ∨ placement i j = none) ∧ 
    (∀ i j, placement i j → i + 1 < 10 → placement (i + 1) j = none) ∧
    (∀ i j, placement i j → j + 1 < 10 → placement i (j + 2) = none) ∧
    (∀ i j, placement i i → i ≠ j -> ∀ n, placement i j → placement i n = none ∧ placement i (n + 1) = none) 
    → placement i j = none) :=
begin
  sorry
end

end max_ships_on_grid_l267_267994


namespace probability_above_parabola_l267_267371

def is_above_parabola (a b : ℕ) : Prop :=
  ∀ x : ℝ, b > a * x ^ 2 + b * x

theorem probability_above_parabola :
  let points : ℕ × ℕ := { (a, b) | a ∈ finset.range 1 10 ∧ b ∈ finset.range 1 10 }
  let valid_points : ℕ × ℕ := { p ∈ points | is_above_parabola p.1 p.2 }
  (finset.card valid_points / finset.card points : ℚ) = 8 / 9 := sorry

end probability_above_parabola_l267_267371


namespace Q_equals_10_04_l267_267444
-- Import Mathlib for mathematical operations and equivalence checking

-- Define the given conditions
def a := 6
def b := 3
def c := 2

-- Define the expression to be evaluated
def Q : ℚ := (a^3 + b^3 + c^3) / (a^2 - a*b + b^2 - b*c + c^2)

-- Prove that the expression equals 10.04
theorem Q_equals_10_04 : Q = 10.04 := by
  -- Proof goes here
  sorry

end Q_equals_10_04_l267_267444


namespace sin_2_angle_BPC_l267_267922

variables {A B C D P : Type}
variables (a b c d : Float) (alpha beta gamma : Float)

-- Conditions
def equally_spaced (A B C D : Float) : Prop := (B - A = 1) ∧ (C - B = 1) ∧ (D - C = 1)
def cos_APC (P A C : Float) : Prop := cos (angle P A C) = 3 / 5
def cos_BPD (P B D : Float) : Prop := cos (angle P B D) = 1 / 5

-- Math proof problem statement
theorem sin_2_angle_BPC : 
  equally_spaced A B C D →
  cos_APC P A C →
  cos_BPD P B D →
  sin (2 * angle P B C) = 2 * sqrt(6) / 25 :=
by
  sorry

end sin_2_angle_BPC_l267_267922


namespace find_B_l267_267070

variable (A B : ℝ)

def condition1 : Prop := A + B = 1210
def condition2 : Prop := (4 / 15) * A = (2 / 5) * B

theorem find_B (h1 : condition1 A B) (h2 : condition2 A B) : B = 484 :=
sorry

end find_B_l267_267070


namespace geometric_series_solution_l267_267029

noncomputable def geometric_series_sums (b1 q : ℝ) : Prop :=
  (b1 / (1 - q) = 16) ∧ (b1^2 / (1 - q^2) = 153.6) ∧ (|q| < 1)

theorem geometric_series_solution (b1 q : ℝ) (h : geometric_series_sums b1 q) :
  q = 2 / 3 ∧ b1 * q^3 = 32 / 9 :=
by
  sorry

end geometric_series_solution_l267_267029


namespace smallest_positive_period_of_f_max_min_values_of_f_l267_267250

noncomputable def f (x : ℝ) : ℝ := sin (x / 2) ^ 2 + sqrt 3 * sin (x / 2) * cos (x / 2)

theorem smallest_positive_period_of_f : ∀ (x : ℝ), f (x + 2 * π) = f x := by
  sorry

theorem max_min_values_of_f : 
  ∀ (x : ℝ), (π / 2 ≤ x ∧ x ≤ π) → (1 ≤ f x ∧ f x ≤ 3 / 2) := by
  sorry

end smallest_positive_period_of_f_max_min_values_of_f_l267_267250


namespace largest_divisor_of_five_consecutive_integers_l267_267462

open Nat

theorem largest_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, k ∈ {n, n+1, n+2, n+3, n+4} ∧
    ∀ m ∈ {2, 3, 4, 5}, m ∣ k → 60 ∣ (n * (n+1) * (n+2) * (n+3) * (n+4)) := 
sorry

end largest_divisor_of_five_consecutive_integers_l267_267462


namespace S₈_proof_l267_267858

-- Definitions
variables {a₁ q : ℝ} {S : ℕ → ℝ}

-- Condition for the sum of the first n terms of the geometric sequence
def geom_sum (n : ℕ) : ℝ :=
  a₁ * (1 - q ^ n) / (1 - q)

-- Conditions from the problem
def S₄ := geom_sum 4 = -5
def S₆ := geom_sum 6 = 21 * geom_sum 2

-- Theorem to prove
theorem S₈_proof (h₁ : S₄) (h₂ : S₆) : geom_sum 8 = -85 :=
by
  sorry

end S₈_proof_l267_267858


namespace solve_for_x_l267_267001

theorem solve_for_x : ∃ x : ℚ, (2/3 - 1/4) = 1/x ∧ x = 12/5 :=
by
  use 12/5
  split
  · norm_num
  · norm_num
  · sorry

end solve_for_x_l267_267001


namespace lambda_parallel_lambda_perpendicular_l267_267736

-- Given vectors
def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 2)
def c : ℝ × ℝ := (4, 1)

-- Vector operations
def vec_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def scalar_mul (λ : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (λ * v.1, λ * v.2)

-- Part 1: Prove λ = -16/13
theorem lambda_parallel (λ : ℝ) : vec_add a (scalar_mul λ c) = scalar_mul (-5 / (3 + 4*λ)) (2 * vec_add b (-a)) → λ = -16 / 13 :=
by 
  sorry

-- Part 2: Prove λ = -11/18
theorem lambda_perpendicular (λ : ℝ) : vec_add a (scalar_mul λ c) = (λ * (4, 1)) → (-5 * (3 + 4*λ)) + (2 * (2 + λ)) = 0 → λ = -11 / 18 :=
by
  sorry

end lambda_parallel_lambda_perpendicular_l267_267736


namespace regular_polygon_sides_l267_267611

theorem regular_polygon_sides (n : ℕ) (h : n ≥ 3) (interior_angle : ℝ) : 
  interior_angle = 144 → n = 10 :=
by
  intro h1
  sorry

end regular_polygon_sides_l267_267611


namespace integer_values_in_range_l267_267416

theorem integer_values_in_range :
  {x : ℤ | 5 < real.sqrt x ∧ real.sqrt x < 6}.finite.to_finset.card = 10 :=
by
  sorry

end integer_values_in_range_l267_267416


namespace integer_values_satisfying_sqrt_condition_l267_267418

theorem integer_values_satisfying_sqrt_condition :
  {x : ℤ | 5 < Real.sqrt x ∧ Real.sqrt x < 6}.card = 10 :=
by
  sorry

end integer_values_satisfying_sqrt_condition_l267_267418


namespace sum_of_eight_l267_267880

variable (a₁ : ℕ) (q : ℕ)
variable (S : ℕ → ℕ) -- Assume S is a function from natural numbers to natural numbers representing S_n

-- Condition 1: Sum of first 4 terms equals -5
axiom h1 : S 4 = -5

-- Condition 2: Sum of the first 6 terms is 21 times the sum of the first 2 terms
axiom h2 : S 6 = 21 * S 2

-- The formula for the sum of the first n terms of a geometric sequence
def sum_of_first_n_terms (n : ℕ) := a₁ * (1 - q^n) / (1 - q)

-- Statement to be proved
theorem sum_of_eight : S 8 = -85 := sorry

end sum_of_eight_l267_267880


namespace ellipse_major_minor_axis_l267_267632

theorem ellipse_major_minor_axis (k : ℝ) : 
  (∀ x y : ℝ, x^2 + k * y^2 = 1) ∧
  (∀ a b : ℝ, a = sqrt (1 / k) ∧ b = 1) ∧
  (2 * sqrt (1 / k) = 2 * 2) ∧
  (1 / k > 1) 
  → k = 1 / 4 := 
by
  sorry

end ellipse_major_minor_axis_l267_267632


namespace circle_line_distance_l267_267382

-- Define the problem statement
theorem circle_line_distance (c : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = 4 ∧ dist (x, y) (12, -5, c) = 1) ↔ (-13 < c ∧ c < 13) := 
sorry

end circle_line_distance_l267_267382


namespace trigonometric_expression_value_l267_267643

theorem trigonometric_expression_value :
  4 * Real.cos (15 * Real.pi / 180) * Real.cos (75 * Real.pi / 180) -
  Real.sin (15 * Real.pi / 180) * Real.sin (75 * Real.pi / 180) = 3 / 4 := sorry

end trigonometric_expression_value_l267_267643


namespace probability_neither_prime_nor_composite_l267_267275

def is_prime (n : ℕ) : Prop := n ≥ 2 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop := n > 1 ∧ ¬is_prime n

def count_special_numbers (range : list ℕ) : nat :=
  range.countp (λ n, ¬is_prime n ∧ ¬is_composite n)

def total_count := 98

def special_probability : ℚ := count_special_numbers (list.range 99) / total_count

theorem probability_neither_prime_nor_composite : special_probability = 1 / 98 := sorry

end probability_neither_prime_nor_composite_l267_267275


namespace value_of_q_when_p_is_smallest_l267_267440

-- Definitions of primality
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m > 1, m < n → ¬ (n % m = 0)

-- smallest prime number
def smallest_prime : ℕ := 2

-- Given conditions
def p : ℕ := 3
def q : ℕ := 2 + 13 * p

-- The theorem to prove
theorem value_of_q_when_p_is_smallest :
  is_prime smallest_prime →
  is_prime q →
  smallest_prime = 2 →
  p = 3 →
  q = 41 :=
by sorry

end value_of_q_when_p_is_smallest_l267_267440


namespace problem_l267_267343

noncomputable def a : ℕ → Int
| 0     := 0
| (n+1) := a (n / 2) + (-1)^(n+1)

theorem problem (n : ℕ) (h : n ≤ 1996) :
  (∀ n ≤ 1996, ∃ max a(n) = 9) ∧ (∀ n ≤ 1996, ∃ min a(n) = -10) ∧ (number_of_zeros (a n 1996) = 346) :=
by
  sorry

-- We define a function to calculate the number of zeros in the sequence till n.
noncomputable def number_of_zeros : (ℕ → ℤ) → ℕ → ℕ
| a, n => ((List.range (n + 1)).filter (λ k => a k = 0)).length

end problem_l267_267343


namespace problem_solution_l267_267962

theorem problem_solution (a b : ℕ) (x : ℝ) (h1 : x^2 + 14 * x = 24) (h2 : x = Real.sqrt a - b) (h3 : a > 0) (h4 : b > 0) :
  a + b = 80 := 
sorry

end problem_solution_l267_267962


namespace num_4_digit_odd_distinct_l267_267913

theorem num_4_digit_odd_distinct : 
  ∃ n : ℕ, n = 5 * 4 * 3 * 2 :=
sorry

end num_4_digit_odd_distinct_l267_267913


namespace n_plus_one_integers_properties_l267_267163

open Nat

theorem n_plus_one_integers_properties (n : ℕ) (h1 : n > 0) (s : Finset ℕ) (h2 : s.card = n + 1) (h3 : ∀ x ∈ s, x ∈ Finset.range (2 * n + 1)) :
  (∃ a b ∈ s, a ≠ b ∧ gcd a b = 1) ∧ (∃ a b ∈ s, a ≠ b ∧ (a ∣ b ∨ b ∣ a)) :=
sorry

end n_plus_one_integers_properties_l267_267163


namespace inflated_cost_per_person_l267_267373

def estimated_cost : ℝ := 30e9
def people_sharing : ℝ := 200e6
def inflation_rate : ℝ := 0.05

theorem inflated_cost_per_person :
  (estimated_cost * (1 + inflation_rate)) / people_sharing = 157.5 := by
  sorry

end inflated_cost_per_person_l267_267373


namespace distinct_ordered_triple_solutions_count_l267_267246

theorem distinct_ordered_triple_solutions_count :
  ∃ (n : ℕ), n = 1176 ∧ (∀ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 50) → true := 
by {
  exact ⟨1176, rfl, sorry⟩
}

end distinct_ordered_triple_solutions_count_l267_267246


namespace missing_side_length_of_pan_l267_267575

-- Definition of the given problem's conditions
def pan_side_length := 29
def total_fudge_pieces := 522
def fudge_piece_area := 1

-- Proof statement in Lean 4
theorem missing_side_length_of_pan : 
  (total_fudge_pieces * fudge_piece_area) = (pan_side_length * 18) :=
by
  sorry

end missing_side_length_of_pan_l267_267575


namespace max_divisor_of_five_consecutive_integers_l267_267512

theorem max_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, 60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intros n
  sorry

end max_divisor_of_five_consecutive_integers_l267_267512


namespace convert_to_scientific_notation_l267_267977

theorem convert_to_scientific_notation :
  (448000 : ℝ) = 4.48 * 10^5 :=
by
  sorry

end convert_to_scientific_notation_l267_267977


namespace correct_calculation_b_l267_267554

theorem correct_calculation_b (a : ℝ) : a^3 * a^4 = a^7 :=
sorry

end correct_calculation_b_l267_267554


namespace largest_divisor_of_5_consecutive_integers_l267_267529

theorem largest_divisor_of_5_consecutive_integers :
  ∀ (n : ℤ), ∃ d, d = 120 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intro n
  use 120
  split
  exact rfl
  sorry

end largest_divisor_of_5_consecutive_integers_l267_267529


namespace regular_polygon_sides_l267_267604

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ n → ∃ p, ∠(polygon_interior_angle p) = 144) : n = 10 := 
by
  sorry

end regular_polygon_sides_l267_267604


namespace relationship_between_a_b_c_l267_267241

noncomputable def f (x : ℝ) : ℝ := sin x - x

def a := f (-1/2)
def b := f 3
def c := f 0

theorem relationship_between_a_b_c :
  (b < a) ∧ (a < c) :=
by
  sorry

end relationship_between_a_b_c_l267_267241


namespace intersection_with_y_axis_is_correct_l267_267775

theorem intersection_with_y_axis_is_correct (x y : ℝ) (h : y = 5 * x + 1) (hx : x = 0) : y = 1 :=
by
  sorry

end intersection_with_y_axis_is_correct_l267_267775


namespace solve_for_x_l267_267940

theorem solve_for_x : 
  let x := (√(7^2 + 24^2)) / (√(49 + 16)) in 
  x = 25 * √65 / 65 := 
by
  -- Step 1: expand the terms inside the square roots
  let a := 7^2 + 24^2 
  let b := 49 + 16

  have a_eq : a = 625 := by
    calc
      a = 7^2 + 24^2 : rfl
      ... = 49 + 576 : rfl
      ... = 625 : rfl

  have b_eq : b = 65 := by
    calc
      b = 49 + 16 : rfl
      ... = 65 : rfl

  -- Step 2: Simplify the square roots
  let sqrt_a := √a
  have sqrt_a_eq : sqrt_a = 25 := by
    rw [a_eq]
    norm_num

  let sqrt_b := √b
  have sqrt_b_eq : sqrt_b = √65 := by
    rw [b_eq]

  -- Step 3: Simplify x
  let x := sqrt_a / sqrt_b

  show x = 25 * √65 / 65
  rw [sqrt_a_eq, sqrt_b_eq]
  field_simp
  norm_num
  rw [mul_div_cancel_left 25 (sqrt_ne_zero.2 (ne_of_gt (by norm_num : √65 ≠ 0))) ]
  sorry

end solve_for_x_l267_267940


namespace problem_area_of_shaded_region_l267_267648

noncomputable def area_of_shaded_region (r : ℝ) (d : ℝ) : ℝ :=
  let rectangle_area := 3 * (2 * r * d)
  let triangle_area := 2 * (1 / 2 * 3 * (r * sqrt 2))
  let sector_area := 2 * (1 / 8 * pi * 3^2)
  rectangle_area - triangle_area - sector_area

theorem problem_area_of_shaded_region :
  area_of_shaded_region (3) (sqrt 3) = 18 * sqrt 3 - 9 * sqrt 2 - (9 * pi / 4) :=
by
  sorry

end problem_area_of_shaded_region_l267_267648


namespace hypotenuse_square_l267_267296

-- Define the right triangle property and the consecutive integer property
variables (a b c : ℤ)

-- Noncomputable definition will be used as we are proving a property related to integers
noncomputable def consecutive_integers (a b : ℤ) : Prop := b = a + 1

-- Define the statement to prove
theorem hypotenuse_square (h_consec : consecutive_integers a b) (h_right_triangle : a * a + b * b = c * c) : 
  c * c = 2 * a * a + 2 * a + 1 :=
by {
  -- We only need to state the theorem
  sorry
}

end hypotenuse_square_l267_267296


namespace four_digit_numbers_count_l267_267261

noncomputable def numberOfValidFourDigitNumbers : Nat :=
  264

theorem four_digit_numbers_count :
  ∃ (numbers : Finset (Fin 10000)), 
    (∀ n ∈ numbers, 
      let digits := [n / 1000, (n % 1000) / 100, (n % 100) / 10, n % 10] in
      ∀ (i j k l : Nat), 
        (i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l) ∧ 
        (some digit is average of the other two) ∧
        (one of these two is the average of the first and last digit)
    ) ∧
    numbers.card = numberOfValidFourDigitNumbers :=
sorry

end four_digit_numbers_count_l267_267261


namespace inradius_of_triangle_l267_267025

theorem inradius_of_triangle :
  ∀ (P A r : ℝ), P = 48 ∧ A = 60 ∧ A = r * (P / 2) → r = 2.5 :=
by 
  intros P A r h,
  cases h with hP hA,
  cases hA with hA hR,
  sorry

end inradius_of_triangle_l267_267025


namespace friend_spent_more_l267_267066

theorem friend_spent_more (total_spent friend_spent: ℝ) (h_total: total_spent = 15) (h_friend: friend_spent = 10) :
  friend_spent - (total_spent - friend_spent) = 5 :=
by
  sorry

end friend_spent_more_l267_267066


namespace average_class_size_l267_267033

theorem average_class_size 
  (three_year_olds : ℕ := 13)
  (four_year_olds : ℕ := 20)
  (five_year_olds : ℕ := 15)
  (six_year_olds : ℕ := 22) : 
  ((three_year_olds + four_year_olds + five_year_olds + six_year_olds) / 2) = 35 := 
by
  sorry

end average_class_size_l267_267033


namespace number_of_people_in_group_l267_267100

-- Definitions and conditions
def total_cost : ℕ := 94
def mango_juice_cost : ℕ := 5
def pineapple_juice_cost : ℕ := 6
def pineapple_cost_total : ℕ := 54

-- Theorem statement to prove
theorem number_of_people_in_group : 
  ∃ M P : ℕ, 
    mango_juice_cost * M + pineapple_juice_cost * P = total_cost ∧ 
    pineapple_juice_cost * P = pineapple_cost_total ∧ 
    M + P = 17 := 
by 
  sorry

end number_of_people_in_group_l267_267100


namespace factorization_l267_267160

theorem factorization (x : ℝ) : x^10 - 1024 = (x^5 + 32) * (x^5 - 32) := 
by 
  sorry

end factorization_l267_267160


namespace no_three_digit_number_l267_267262

theorem no_three_digit_number (N : ℕ) : 
  (100 ≤ N ∧ N < 1000 ∧ 
   (∀ k, k ∈ [1,2,3] → 5 < (N / 10^(k - 1) % 10)) ∧ 
   (N % 6 = 0) ∧ (N % 5 = 0)) → 
  false :=
by
sorry

end no_three_digit_number_l267_267262


namespace geometric_sequence_sum_S8_l267_267845

noncomputable def sum_geometric_seq (a q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_S8 (a q : ℝ) (h1 : q ≠ 1)
  (h2 : sum_geometric_seq a q 4 = -5)
  (h3 : sum_geometric_seq a q 6 = 21 * sum_geometric_seq a q 2) :
  sum_geometric_seq a q 8 = -85 :=
sorry

end geometric_sequence_sum_S8_l267_267845


namespace minimum_value_proof_l267_267387

noncomputable def minimum_value_condition (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : (deriv (λ x : ℝ, a * x^2 + b * x) 1 = 2)) : ℝ :=
  (8 * a + b) / (a * b)

theorem minimum_value_proof (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : (deriv (λ x : ℝ, a * x^2 + b * x) 1 = 2)) :
  minimum_value_condition a b h1 h2 h3 = 9 :=
by
  sorry

end minimum_value_proof_l267_267387


namespace solve_for_x_l267_267943

noncomputable def n : ℝ := Real.sqrt (7^2 + 24^2)
noncomputable def d : ℝ := Real.sqrt (49 + 16)
noncomputable def x : ℝ := n / d

theorem solve_for_x : x = 5 * Real.sqrt 65 / 13 := by
  sorry

end solve_for_x_l267_267943


namespace largest_divisor_of_5_consecutive_integers_l267_267525

theorem largest_divisor_of_5_consecutive_integers :
  ∀ (n : ℤ), ∃ d, d = 120 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intro n
  use 120
  split
  exact rfl
  sorry

end largest_divisor_of_5_consecutive_integers_l267_267525


namespace floor_square_difference_l267_267674

noncomputable def x := 13.3
noncomputable def floor (y : ℝ) : ℤ := ⌊y⌋

theorem floor_square_difference :
  floor (x * x) - (floor x * floor x) = 7 := 
by
  sorry

end floor_square_difference_l267_267674


namespace elevator_problem_cos_B_triangle_dot_product_quadrilateral_ellipse_statements_l267_267083

-- Problem 1
theorem elevator_problem (elevators : Fin 4) (people : Fin 3) (two_same : ℕ) :
  ∃ n, n = 36 :=
by
  -- Placeholder for the proof
  sorry

-- Problem 2
theorem cos_B_triangle (A B : ℝ) (a b : ℝ) (h1 : A = 2 * B) (h2 : 2 * a = 3 * b) :
  ∃ cosB, cosB = 3 / 4 :=
by
  -- Placeholder for the proof
  sorry

-- Problem 3
theorem dot_product_quadrilateral (AB DA CA CB AC DC : ℝ → ℝ)
  (AB_eq : AB = 2) (DA_eq : DA = (1/2) * (CA + CB)) :
  AB • DC = 2 :=
by
  -- Placeholder for the proof
  sorry

-- Problem 4
theorem ellipse_statements (x y b : ℝ) (h0 : 0 < b) (h1 : b < sqrt 6)
  (condition : |PB1| + |PB2| = |PF1| + |PF2|) :
  (trajectory_symmetric : ∃ i, i = true) ∧
  (only_two_points : ∃ ii, ii = false) ∧
  (min_OP : ∃ iii, iii = true) :=
by
  -- Placeholder for the proof
  sorry

end elevator_problem_cos_B_triangle_dot_product_quadrilateral_ellipse_statements_l267_267083


namespace angle_BAC_is_75_l267_267783

theorem angle_BAC_is_75
  (ABC : Triangle)
  (CBA_eq_45 : ABC.angle C B A = 45)
  (P_on_BC : ∃ P, ABC.lineSegment B C ∋ P ∧ (ABC.dist B P) / (ABC.dist P C) = 1 / 2)
  (CPA_eq_60 : ∃ P, P ∈ ABC.lineSegment B C ∧ ABC.angle C P A = 60) :
  ABC.angle B A C = 75 := sorry

end angle_BAC_is_75_l267_267783


namespace volume_of_Q4_l267_267706

theorem volume_of_Q4 :
  let Q0 := 1,
  let delta_volume n := match n with
    | 0 => 0
    | 1 => 4 / 27
    | n + 1 => (2 / 9) * delta_volume n,
  let total_volume n := Q0 + (Finset.range n).sum delta_volume,
  total_volume 5 = 67191 / 19683 :=
by
  sorry

end volume_of_Q4_l267_267706


namespace cover_all_points_inside_triangle_l267_267059

variables (ABC : Triangle) (R1 R2 R3 : Set Point)
variables (parallel_to_perpendicular_directions : (Set Point) → Prop)

def covers_sides (R1 R2 R3 : Set Point) (ABC : Triangle) :=
  (∀ p, p ∈ ABC.side_1 → p ∈ R1 ∪ R2 ∪ R3) ∧
  (∀ p, p ∈ ABC.side_2 → p ∈ R1 ∪ R2 ∪ R3) ∧
  (∀ p, p ∈ ABC.side_3 → p ∈ R1 ∪ R2 ∪ R3)

def sides_parallel (R : Set Point) :=
  parallel_to_perpendicular_directions R

theorem cover_all_points_inside_triangle (ABC : Triangle) (R1 R2 R3 : Set Point)
  (h_sides_parallel_R1 : sides_parallel R1)
  (h_sides_parallel_R2 : sides_parallel R2)
  (h_sides_parallel_R3 : sides_parallel R3)
  (h_cover_sides : covers_sides R1 R2 R3 ABC) :
  ∀ p, p ∈ ABC → p ∈ R1 ∪ R2 ∪ R3 :=
sorry

end cover_all_points_inside_triangle_l267_267059


namespace integer_values_in_range_l267_267414

theorem integer_values_in_range :
  {x : ℤ | 5 < real.sqrt x ∧ real.sqrt x < 6}.finite.to_finset.card = 10 :=
by
  sorry

end integer_values_in_range_l267_267414


namespace find_a5_a6_l267_267304

-- Define the arithmetic sequence properties
variables (a : ℕ → ℤ) (d : ℤ)

-- Define the conditions in the problem
def condition1 := a 1 + a 2 = 2
def condition2 := a 3 + a 4 = 10

-- Define the relationship of the sequence based on the conditions
def seq_relation : Prop := 
  ∀ n : ℕ, a (n + 2) = a n + 2 * d

-- The theorem to prove
theorem find_a5_a6 (h1 : condition1) (h2 : condition2) (h_seq : seq_relation a d) : 
  a 5 + a 6 = 18 :=
sorry

end find_a5_a6_l267_267304


namespace jimmy_can_lose_5_more_points_l267_267315

theorem jimmy_can_lose_5_more_points (min_points_to_pass : ℕ) (points_per_exam : ℕ) (number_of_exams : ℕ) (points_lost : ℕ) : 
  min_points_to_pass = 50 → 
  points_per_exam = 20 → 
  number_of_exams = 3 → 
  points_lost = 5 → 
  (points_per_exam * number_of_exams - points_lost - 5) = min_points_to_pass :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end jimmy_can_lose_5_more_points_l267_267315


namespace midpoint_polygon_area_half_l267_267355

noncomputable def polygon_area (n : ℕ) (vertices : List (ℝ × ℝ)) : ℝ := sorry

theorem midpoint_polygon_area_half
  (n : ℕ)
  (hn : n ≥ 4)
  (P : Fin n → (ℝ × ℝ))  -- Original n-gon as a function from Fin n to points
  (convex : Convex ℝ (ConvexHull ℝ (Set.range P))) : -- Convexity condition
  let midpoints := λ i : Fin n, ((P i.fst + P i.snd) / 2) -- Midpoints of sides
  polygon_area n midpoints ≥ (polygon_area n P) / 2 :=
sorry

end midpoint_polygon_area_half_l267_267355


namespace find_all_functions_l267_267191

def satisfies_conditions (f : ℕ+ → ℝ) : Prop :=
  (∀ n : ℕ+, f(n + 1) ≥ f(n)) ∧ (∀ m n : ℕ+, m.gcd n = 1 → f(m * n) = f(m) * f(n))

theorem find_all_functions (f : ℕ+ → ℝ) (hf : satisfies_conditions f) :
  (∀ n : ℕ+, f(n) = 0) ∨ (∃ a : ℝ, ∀ n : ℕ+, f(n) = n^a ∧ a ≥ 0) :=
sorry

end find_all_functions_l267_267191


namespace construct_pentagon_l267_267168

-- Define a structure for vector spaces with the required properties
structure Vector :=
  (x : ℝ)
  (y : ℝ)

-- Define the reflection function
def reflection (a b : Vector) : Vector :=
  { x := 2 * b.x - a.x,
    y := 2 * b.y - a.y }

-- Define the conditions
variables (b1 b2 b3 b4 b5 : Vector)

-- Define the proof problem statement
theorem construct_pentagon :
  ∃ (a1 a2 a3 a4 a5 : Vector),
    a1 = {
      x := (b1.x + 2 * b2.x + 4 * b3.x + 8 * b4.x + 16 * b5.x) / 31,
      y := (b1.y + 2 * b2.y + 4 * b3.y + 8 * b4.y + 16 * b5.y) / 31
    } ∧
    a2 = { x := (a1.x + b1.x) / 2, y := (a1.y + b1.y) / 2 } ∧
    a3 = { x := (a2.x + b2.x) / 2, y := (a2.y + b2.y) / 2 } ∧
    a4 = { x := (a3.x + b3.x) / 2, y := (a3.y + b3.y) / 2 } ∧
    a5 = { x := (a4.x + b4.x) / 2, y := (a4.y + b4.y) / 2 } ∧
    a1 = { x := (a5.x + b5.x) / 2, y := (a5.y + b5.y) / 2 } := by
  sorry

end construct_pentagon_l267_267168


namespace jimmy_max_loss_l267_267317

-- Definition of the conditions
def exam_points : ℕ := 20
def number_of_exams : ℕ := 3
def points_lost_for_behavior : ℕ := 5
def passing_score : ℕ := 50

-- Total points Jimmy has earned and lost
def total_points : ℕ := (number_of_exams * exam_points) - points_lost_for_behavior

-- The maximum points Jimmy can lose and still pass
def max_points_jimmy_can_lose : ℕ := total_points - passing_score

-- Statement to prove
theorem jimmy_max_loss : max_points_jimmy_can_lose = 5 := 
by
  sorry

end jimmy_max_loss_l267_267317


namespace number_of_proper_subsets_l267_267911

open Set

theorem number_of_proper_subsets 
  (M : Set ℤ) (hM : M = {x | x^2 - 2*x - 3 < 0}) :
  card (powerset M) - 1 = 7 := 
by
  sorry

end number_of_proper_subsets_l267_267911


namespace maximize_f_l267_267217

noncomputable def f (x y z : ℝ) := x * y^2 * z^3

theorem maximize_f :
  ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → x + y + z = 1 →
  f x y z ≤ 1 / 432 ∧ (f x y z = 1 / 432 → x = 1/6 ∧ y = 1/3 ∧ z = 1/2) :=
by
  sorry

end maximize_f_l267_267217


namespace sum_of_arith_prog_l267_267690

def arithmetic_sum (p : ℕ) (n : ℕ) :=
  n / 2 * (2 * p + (2 * p + 1) * (n - 1))

def Tp (p : ℕ) :=
  arithmetic_sum (2 * p) 50

def series_sum :=
  ∑ p in range 1 11, Tp p

theorem sum_of_arith_prog :
  series_sum = 152500 :=
sorry

end sum_of_arith_prog_l267_267690


namespace rationalize_denominator_l267_267928

theorem rationalize_denominator (a b : ℝ) (h : a = 45 * real.sqrt 3) (h₁ : b = real.sqrt 45) :
  a / b = 3 * real.sqrt 15 := sorry

end rationalize_denominator_l267_267928


namespace simplify_expression_l267_267939

theorem simplify_expression : ( (2^8 + 4^5) * (2^3 - (-2)^3) ^ 8 ) = 0 := 
by sorry

end simplify_expression_l267_267939


namespace product_of_five_consecutive_integers_divisible_by_120_l267_267448

theorem product_of_five_consecutive_integers_divisible_by_120 (n : ℤ) : 
  120 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end product_of_five_consecutive_integers_divisible_by_120_l267_267448


namespace no_real_or_purely_imaginary_roots_l267_267901

variable {a b c : ℝ}

def f (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Given conditions
axiom purely_imaginary_roots (h : f 0 = 0) : ∃ m : ℝ, m ≠ 0 ∧ f (m * complex.I) = 0

-- Question
theorem no_real_or_purely_imaginary_roots (h : ∃ m : ℝ, m ≠ 0 ∧ f (m * complex.I) = 0) :
  ∀ x : ℝ, f (f x) ≠ 0 :=
sorry

end no_real_or_purely_imaginary_roots_l267_267901


namespace brandon_businesses_l267_267143

theorem brandon_businesses (total_businesses: ℕ) (fire_fraction: ℚ) (quit_fraction: ℚ) 
  (h_total: total_businesses = 72) 
  (h_fire_fraction: fire_fraction = 1/2) 
  (h_quit_fraction: quit_fraction = 1/3) : 
  total_businesses - (total_businesses * fire_fraction + total_businesses * quit_fraction) = 12 :=
by 
  sorry

end brandon_businesses_l267_267143


namespace find_line_and_area_l267_267710

-- Definitions and Conditions
def point := ℝ × ℝ
def P : point := (2, 0)
def circle (center : point) (radius : ℝ) : set point := {p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}
def circle_origin_radius2 := circle (0, 0) 2
def line_l (p1 p2 : point) : set point := {p | (p2.2 - p1.2) * (p.1 - p1.1) = (p.2 - p1.2) * (p2.1 - p1.1)}

-- Given
def A : point := sorry -- Point of intersection 1
def B : point := sorry -- Point of intersection 2
def M : point := ((A.1 + B.1) / 2, (A.2 + B.2) / 2) -- Midpoint M
def O : point := (0, 0) -- Origin
def circle_C := circle (0, 4) 4 -- Given circle C

-- Proving part
theorem find_line_and_area : 
  (∃ l: set point, l = line_l P M ∧ (∀ q ∈ circle_C, q ∈ l → q = A ∨ q = B) ∧ |(O -| M)| = |(O -| P)|) 
  → 
  (∃ eq_l : set point, eq_l = {p | p.1 + 2 * p.2 - 2 = 0} ∧ 
   ∃ area_ABC : ℝ, area_ABC = (12 * real.sqrt 11) / 5) :=
by
  sorry

end find_line_and_area_l267_267710


namespace geometric_seq_sum_l267_267835

-- Definitions of the conditions
variables {a₁ q : ℚ}
def S (n : ℕ) := a₁ * (1 - q^n) / (1 - q)

-- Hypotheses from the conditions
theorem geometric_seq_sum :
  (S 4 = -5) →
  (S 6 = 21 * S 2) →
  (S 8 = -85) :=
by
  -- Assume the given conditions
  intros h1 h2
  -- The actual proof will be inserted here
  sorry

end geometric_seq_sum_l267_267835


namespace largest_divisor_of_five_consecutive_integers_l267_267463

open Nat

theorem largest_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, k ∈ {n, n+1, n+2, n+3, n+4} ∧
    ∀ m ∈ {2, 3, 4, 5}, m ∣ k → 60 ∣ (n * (n+1) * (n+2) * (n+3) * (n+4)) := 
sorry

end largest_divisor_of_five_consecutive_integers_l267_267463


namespace number_of_elements_in_M_l267_267396

theorem number_of_elements_in_M :
  (∃! (M : Finset ℕ), M = {m | ∃ (n : ℕ), n > 0 ∧ m = 2*n - 1 ∧ m < 60 } ∧ M.card = 30) :=
sorry

end number_of_elements_in_M_l267_267396


namespace males_in_band_not_in_orchestra_or_choir_l267_267378

theorem males_in_band_not_in_orchestra_or_choir :
  let females_in_band := 50
  let males_in_band := 40
  let females_in_orchestra := 40
  let males_in_orchestra := 50
  let females_in_choir := 30
  let males_in_choir := 45
  let females_in_band_orchestra := 20
  let females_in_band_choir := 15
  let females_in_orchestra_choir := 10
  let females_all_three := 5
  let total_students := 120
  let total_females := females_in_band + females_in_orchestra + females_in_choir
                      - females_in_band_orchestra - females_in_band_choir
                      - females_in_orchestra_choir + females_all_three
  let total_males := total_students - total_females
  let males_in_band_only := males_in_band - (total_males - males_in_band)
  (total_males = 40) → (total_females = 80) → (males_in_band_not_in_orchestra_or_choir = 30) :=
by
  intros
  sorry

end males_in_band_not_in_orchestra_or_choir_l267_267378


namespace area_of_right_triangle_ABC_l267_267778

variable {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]

noncomputable def area_triangle_ABC (AB BC : ℝ) (angleB : ℕ) (hangle : angleB = 90) (hAB : AB = 30) (hBC : BC = 40) : ℝ :=
  1 / 2 * AB * BC

theorem area_of_right_triangle_ABC (AB BC : ℝ) (angleB : ℕ) (hangle : angleB = 90) 
  (hAB : AB = 30) (hBC : BC = 40) : 
  area_triangle_ABC AB BC angleB hangle hAB hBC = 600 :=
by
  sorry

end area_of_right_triangle_ABC_l267_267778


namespace fraction_S5_S3_l267_267210

theorem fraction_S5_S3 
  (S : ℕ → ℤ)
  (a : ℕ → ℤ)
  (h_sum : ∀ n, S n = (finset.range n).sum a)
  (h_perp : ∀ n, 4 * (a n - 1) - 2 * S n = 0) :
  S 5 / S 3 = 31 / 7 :=
sorry

end fraction_S5_S3_l267_267210


namespace tammy_climb_total_distance_l267_267950

theorem tammy_climb_total_distance :
  ∀ (h : ℕ), 
    (h + (h - 2) = 14) →
    (∀ h1 h2 : ℕ, h = h1 + 2 * h2) →
    let speed1 : ℝ := 3.5, speed2 : ℝ := 4.0
    let time1 : ℝ := h, time2 : ℝ := h - 2
    (time2 * speed2) + (time1 * speed1) = 52 :=
by
  sorry

end tammy_climb_total_distance_l267_267950


namespace arithmetic_seq_sum_l267_267222

variable {a_n : ℕ → ℕ}
variable (S_n : ℕ → ℕ)
variable (q : ℕ)
variable (a_1 : ℕ)

axiom h1 : a_n 2 = 2
axiom h2 : a_n 6 = 32
axiom h3 : ∀ n, S_n n = a_1 * (1 - q ^ n) / (1 - q)

theorem arithmetic_seq_sum : S_n 100 = 2^100 - 1 :=
by
  sorry

end arithmetic_seq_sum_l267_267222


namespace largest_divisor_of_product_of_five_consecutive_integers_l267_267468

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∀ (n : ℤ), ∃ (d : ℤ), d = 60 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end largest_divisor_of_product_of_five_consecutive_integers_l267_267468


namespace correct_range_of_k_l267_267234

noncomputable def k_valid_range (f : ℝ → ℝ) (f' : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f'(x) = real.exp x * (2 * x + 3) + f(x)) ∧
  f(0) = 1 ∧
  ∃ k : ℝ, f(-1) = -real.exp(-1) ∧ 
           f(-2) = -real.exp(-2) ∧ 
           (f(x) - k < 0 → set.countable { x : ℝ | f(x) - k < 0 } ∧ 
             { x : ℝ | f(x) - k < 0 }.to_finset.card = 2) ∧
           k ∈ (-real.exp (-2), 0]

theorem correct_range_of_k (f : ℝ → ℝ) (f' : ℝ → ℝ) :
  k_valid_range f f' :=
sorry

end correct_range_of_k_l267_267234


namespace angle_ABM_l267_267790

-- Define the conditions and proven statement in Lean.
theorem angle_ABM (α : ℝ) (A B C D M : ℝ) (square : ∀ (A B C D : ℝ), (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ D) ∧ (D ≠ A)) 
  (inside_square : ∀ (M : ℝ), (M ≠ A) ∧ (M ≠ B) ∧ (M ≠ C) ∧ (M ≠ D)) 
  (angles : ∠ M A C = α ∧ ∠ M C D = α) : ∠ A B M = 90 - 2 * α := 
by
  sorry

end angle_ABM_l267_267790


namespace students_more_than_guinea_pigs_l267_267673

-- Definitions based on the problem's conditions
def students_per_classroom : Nat := 22
def guinea_pigs_per_classroom : Nat := 3
def classrooms : Nat := 5

-- The proof statement
theorem students_more_than_guinea_pigs :
  (students_per_classroom * classrooms) - (guinea_pigs_per_classroom * classrooms) = 95 :=
by
  sorry

end students_more_than_guinea_pigs_l267_267673


namespace sandy_phone_bill_expense_l267_267932

def sandy_age_now (kim_age : ℕ) : ℕ := 3 * (kim_age + 2) - 2

def sandy_phone_bill (sandy_age : ℕ) : ℕ := 10 * sandy_age

theorem sandy_phone_bill_expense
  (kim_age : ℕ)
  (kim_age_condition : kim_age = 10)
  : sandy_phone_bill (sandy_age_now kim_age) = 340 := by
  sorry

end sandy_phone_bill_expense_l267_267932


namespace intersection_with_y_axis_is_correct_l267_267774

theorem intersection_with_y_axis_is_correct (x y : ℝ) (h : y = 5 * x + 1) (hx : x = 0) : y = 1 :=
by
  sorry

end intersection_with_y_axis_is_correct_l267_267774


namespace quadratic_root_count_in_interval_l267_267666

theorem quadratic_root_count_in_interval (p : ℝ) :
  (if p ≥ 1/3 ∨ p ≤ -1/17 then ∃! x ∈ Ioo (-1: ℝ) 1, 2 * x^2 - 10 * p * x + 7 * p - 1 = 0
  else ∃ x1 x2 ∈ Ioo (-1: ℝ) 1, x1 ≠ x2 ∧ 2 * x1^2 - 10 * p * x1 + 7 * p - 1 = 0 ∧ 2 * x2^2 - 10 * p * x2 + 7 * p - 1 = 0) :=
sorry

end quadratic_root_count_in_interval_l267_267666


namespace geometric_sequence_sum_S8_l267_267849

noncomputable def sum_geometric_seq (a q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_S8 (a q : ℝ) (h1 : q ≠ 1)
  (h2 : sum_geometric_seq a q 4 = -5)
  (h3 : sum_geometric_seq a q 6 = 21 * sum_geometric_seq a q 2) :
  sum_geometric_seq a q 8 = -85 :=
sorry

end geometric_sequence_sum_S8_l267_267849


namespace conic_section_hyperbola_l267_267670

theorem conic_section_hyperbola (x y : ℝ) : 
  (2 * x - 7)^2 - 4 * (y + 3)^2 = 169 → 
  -- Explain that this equation is of a hyperbola
  true := 
sorry

end conic_section_hyperbola_l267_267670


namespace express_y_in_terms_of_x_l267_267727

theorem express_y_in_terms_of_x (x y : ℝ) (h : 3 * x - y = 9) : y = 3 * x - 9 := 
by
  sorry

end express_y_in_terms_of_x_l267_267727


namespace restore_triangle_l267_267930

structure Triangle (P : Type) :=
  (A B C: P)

variables {P : Type} [EuclideanGeometry P]

def isCircumcenter (O : P) (T : Triangle P) : Prop := 
  ∃ r, ∀ (p ∈ [T.A, T.B, T.C]), dist O p = r

def isLemoinePoint (L : P) (T : Triangle P) : Prop := 
  -- Definition of the Lemoine point in terms of the triangle's properties
  sorry

theorem restore_triangle
  (A L_e O : P)
  (hC : isCircumcenter O (Triangle.mk A _ _))
  (hL : isLemoinePoint L_e (Triangle.mk A _ _))
  : ∃ (B C : P), Triangle.mk A B C := 
sorry

end restore_triangle_l267_267930


namespace animal_eyes_count_l267_267294

noncomputable def total_animal_eyes (frogs : ℕ) (crocodiles : ℕ) (eyes_per_frog : ℕ) (eyes_per_crocodile : ℕ) : ℕ :=
frogs * eyes_per_frog + crocodiles * eyes_per_crocodile

theorem animal_eyes_count (frogs : ℕ) (crocodiles : ℕ) (eyes_per_frog : ℕ) (eyes_per_crocodile : ℕ):
  frogs = 20 → crocodiles = 10 → eyes_per_frog = 2 → eyes_per_crocodile = 2 → total_animal_eyes frogs crocodiles eyes_per_frog eyes_per_crocodile = 60 :=
by
  sorry

end animal_eyes_count_l267_267294


namespace problem_part_1_problem_part_2i_problem_part_2ii_problem_part_2iii_l267_267730

-- Definitions given in the problem
def f (x : ℝ) (a : ℝ) : ℝ := real.log x - a * x^2
def g (x : ℝ) (a : ℝ) : ℝ := x * real.exp x + a * x - 3/2
def g' (x : ℝ) (a : ℝ) : ℝ := (x + 1) * real.exp x + a
def k (a : ℝ) : ℝ := 2 * real.exp 1 + a

-- Given conditions 
axiom slope_condition (a : ℝ) : a * k(a) = 3 * (real.exp 2)
axiom monotonic_intervals (h : ∀ x : ℝ, (a = 1 → monotonic_interval f(1) = (0, real.sqrt(2) / 2) ∧ ¬monotonic_interval f(1) = (real.sqrt(2) / 2, ∞)))
axiom function_inequality (h : a = 1 → ∀ x : ℝ, f(x) < g(x))

-- Problem statement
theorem problem_part_1 (a : ℝ) : (a = real.exp 1 ∨ a = -3 * real.exp 1) :=
sorry

theorem problem_part_2i {a : ℝ} : a = 1 → ∀ x : ℝ, x ∈ (0, real.sqrt 2 / 2) → f(x) ≤ f(x) :=
sorry

theorem problem_part_2ii {a : ℝ} : a = 1 → ∀ x : ℝ, x ∈ (real.sqrt 2 / 2, ∞) → f(x) ≤ f(x) :=
sorry

theorem problem_part_2iii {a : ℝ} : a = 1 → ∀ x : ℝ, f(x) < g(x) :=
sorry

end problem_part_1_problem_part_2i_problem_part_2ii_problem_part_2iii_l267_267730


namespace geometric_seq_sum_l267_267843

-- Definitions of the conditions
variables {a₁ q : ℚ}
def S (n : ℕ) := a₁ * (1 - q^n) / (1 - q)

-- Hypotheses from the conditions
theorem geometric_seq_sum :
  (S 4 = -5) →
  (S 6 = 21 * S 2) →
  (S 8 = -85) :=
by
  -- Assume the given conditions
  intros h1 h2
  -- The actual proof will be inserted here
  sorry

end geometric_seq_sum_l267_267843


namespace largest_divisor_of_5_consecutive_integers_l267_267479

theorem largest_divisor_of_5_consecutive_integers :
  ∀ (a b c d e : ℤ), 
    a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e →
    (∃ k : ℤ, k ∣ (a * b * c * d * e) ∧ k = 60) :=
by 
  intro a b c d e h
  sorry

end largest_divisor_of_5_consecutive_integers_l267_267479


namespace number_divisible_by_two_power_digits_1_and_2_l267_267926

theorem number_divisible_by_two_power_digits_1_and_2 :
  ∀ n : ℕ, ∃ N : ℕ, 
  (∀ d : ℕ, d ∈ digits 10 N → d = 1 ∨ d = 2) ∧ 
  (N.digits 10).length = n ∧ 2^n ∣ N := 
by sorry

end number_divisible_by_two_power_digits_1_and_2_l267_267926


namespace product_of_five_consecutive_integers_divisible_by_120_l267_267451

theorem product_of_five_consecutive_integers_divisible_by_120 (n : ℤ) : 
  120 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end product_of_five_consecutive_integers_divisible_by_120_l267_267451


namespace find_radius_of_cylinder_l267_267616

noncomputable def radius_of_cylinder : ℚ :=
  let d_cylinder := 2 * 2  -- Diameter of the cylinder is equal to the height: 2r where r = 2 * r
  let r_cone := 4  -- Radius of the cone
  let h_cone := 10  -- Height of the cone
  let cone_ratio := λ r (h : ℚ) => (h_cone - h) / r -- Ratio from the geometry
  r_cone / cone_ratio 2 (2 * r_cone / cone_ratio 2 r_cone * / 2) -- Simplified using similar triangles

theorem find_radius_of_cylinder :
  radius_of_cylinder = (20 / 9 : ℚ) :=
sorry

end find_radius_of_cylinder_l267_267616


namespace perp_OE_CD_l267_267633

-- Define the points O, A, B, C, D, E in the real vector space.
variables (O A B C D E : ℝ → ℝ → ℝ → ℝ)

-- Define conditions
-- 1. O is the circumcenter of triangle ABC.
def is_circumcenter (O A B C : ℝ → ℝ) : Prop :=
  dist O A = dist O B ∧ dist O B = dist O C

-- 2. D is the midpoint of side AB.
def is_midpoint (A B D : ℝ → ℝ) : Prop :=
  D = (A + B) / 2

-- 3. E is the centroid of triangle ACD.
def is_centroid_ACD (A C D E : ℝ → ℝ) : Prop :=
  E = (A + C + D) / 3

-- 4. AB = AC
def is_isosceles (A B C : ℝ → ℝ) : Prop :=
  dist A B = dist A C

-- The proof statement: OE ⊥ CD given the above conditions.
theorem perp_OE_CD
  (O A B C D E : ℝ → ℝ)
  (h1 : is_circumcenter O A B C)
  (h2 : is_midpoint A B D)
  (h3 : is_centroid_ACD A C D E)
  (h4 : is_isosceles A B C) :
  (O E) ⊥ (C D) :=
sorry

end perp_OE_CD_l267_267633


namespace product_divisible_by_3_or_5_l267_267968

theorem product_divisible_by_3_or_5 {a b c d : ℕ} (h : Nat.lcm (Nat.lcm (Nat.lcm a b) c) d = a + b + c + d) :
  (a * b * c * d) % 3 = 0 ∨ (a * b * c * d) % 5 = 0 :=
by
  sorry

end product_divisible_by_3_or_5_l267_267968


namespace dodecahedron_society_proof_l267_267078

open Fin

-- Definitions based on conditions provided
def individuals : Fin 12 := default

def adjacency (i j : Fin 12) : Prop :=
  -- Define adjacency based on adjacency of faces in a dodecahedron.
  sorry

def acquaintances : Fin 12 → Finset (Fin 12) :=
  λ i, {j | adjacency i j}

-- Conditions
def condition_a : Prop :=
  ∀ i : Fin 12, (acquaintances i).card = 5  -- known to exactly 6 people

def condition_b : Prop :=
  ∀ i : Fin 12, ∃ j k : Fin 12, i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ adjacency i j ∧ adjacency i k ∧ adjacency j k  -- trio of mutually acquainted people

def condition_c : Prop :=
  ∀ a b c d : Fin 12, not (adjacency a b ∧ adjacency a c ∧ adjacency a d ∧ adjacency b c ∧ adjacency b d ∧ adjacency c d)

def condition_d : Prop :=
  ∀ a b c d : Fin 12, not (not (adjacency a b) ∧ not (adjacency a c) ∧ not (adjacency a d) ∧ not (adjacency b c) ∧ not (adjacency b d) ∧ not (adjacency c d))

def condition_e : Prop :=
  ∀ i : Fin 12, ∃ j k : Fin 12, i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ not (adjacency i j) ∧ not (adjacency i k) ∧ not (adjacency j k) -- trio of mutually unacquainted people

def condition_f : Prop :=
  ∀ i : Fin 12, ∃ j : Fin 12, not (adjacency i j) ∧ ∀ k : Fin 12, not (adjacency j k) -- someone who has no mutual acquaintances

-- Theorem to be proven equivalent to the given problem
theorem dodecahedron_society_proof :
  condition_a ∧ condition_b ∧ condition_c ∧ condition_d ∧ condition_e ∧ condition_f :=
sorry

end dodecahedron_society_proof_l267_267078


namespace cost_of_tea_l267_267793

theorem cost_of_tea (x : ℕ) (h1 : 9 * x < 1000) (h2 : 10 * x > 1100) : x = 111 :=
by
  sorry

end cost_of_tea_l267_267793


namespace total_laptops_l267_267995

theorem total_laptops (rows : ℕ) (laptops_per_row : ℕ) (h_rows : rows = 5) (h_laptops_per_row : laptops_per_row = 8) : 
  rows * laptops_per_row = 40 :=
by
  rw [h_rows, h_laptops_per_row]
  norm_num

end total_laptops_l267_267995


namespace proof_l267_267569

-- Define the universal set U.
def U : Set ℕ := {x | x > 0 ∧ x < 9}

-- Define set M.
def M : Set ℕ := {1, 2, 3}

-- Define set N.
def N : Set ℕ := {3, 4, 5, 6}

-- The complement of M with respect to U.
def compl_U_M : Set ℕ := {x ∈ U | x ∉ M}

-- The intersection of complement of M and N.
def result : Set ℕ := compl_U_M ∩ N

-- The theorem to be proven.
theorem proof : result = {4, 5, 6} := by
  -- This is where the proof would go.
  sorry

end proof_l267_267569


namespace maximum_free_cells_l267_267348

def grid_size : ℕ := 100
def triangle_leg_length : ℕ := 1

theorem maximum_free_cells 
  (grid : fin grid_size × fin grid_size → bool)
  (triangle : fin grid_size × fin grid_size → fin 4 → Prop)
  (h1 : ∀ x y, triangle (x, y) 0 ∨ triangle (x, y) 1 ∨ triangle (x, y) 2 ∨ triangle (x, y) 3)
  (h2 : ∀ x y, grid (x, y) = ∃ t, triangle t 0 → ¬ grid (x, y)) : 
  ∃ free_cells, free_cells = 2450 :=
sorry

end maximum_free_cells_l267_267348


namespace inverse_of_zero_is_one_l267_267900

def f (x : ℝ) : ℝ := 4^x - 2^(x + 1)

theorem inverse_of_zero_is_one : f 1 = 0 ↔ ∃ x : ℝ, f x = 0 ∧ x = 1 :=
by
  sorry

end inverse_of_zero_is_one_l267_267900


namespace regular_polygon_sides_l267_267612

theorem regular_polygon_sides (n : ℕ) (h : n ≥ 3) (interior_angle : ℝ) : 
  interior_angle = 144 → n = 10 :=
by
  intro h1
  sorry

end regular_polygon_sides_l267_267612


namespace number_of_ten_dollar_bills_l267_267997

theorem number_of_ten_dollar_bills 
  (x : ℤ) 
  (− needs_to_pay: 128 ℤ)
  (− has_five_dollar_bills: 11 ℤ)
  (− has_one_dollar_bills: 17 ℤ)
  (− uses_at_least_sixteen_bills: 16 ℤ)
  :
  5 * has_five_dollar_bills + 1 * has_one_dollar_bills + 10 * x = needs_to_pay ∧
  has_five_dollar_bills + has_one_dollar_bills + x ≥ uses_at_least_sixteen_bills  → x = 6 :=
by
  sorry

end number_of_ten_dollar_bills_l267_267997


namespace triangle_third_side_l267_267055

theorem triangle_third_side (a b : ℕ) (h₁ : a = 7) (h₂ : b = 11) : ∃ c : ℕ, c < a + b ∧ c > |a - b| ∧ c = 17 := 
by {
  existsi 17,
  split,
  linarith,
  split,
  linarith,
  refl,
}

end triangle_third_side_l267_267055


namespace correct_number_of_statements_l267_267970

noncomputable def number_of_correct_statements := 1

def statement_1 : Prop := false -- Equal angles are not preserved
def statement_2 : Prop := false -- Equal lengths are not preserved
def statement_3 : Prop := false -- The longest segment feature is not preserved
def statement_4 : Prop := true  -- The midpoint feature is preserved

theorem correct_number_of_statements :
  (statement_1 ∧ statement_2 ∧ statement_3 ∧ statement_4) = true →
  number_of_correct_statements = 1 :=
by
  sorry

end correct_number_of_statements_l267_267970


namespace triangle_shape_l267_267785

variable (A B C : ℝ) (AB BC AC : ℝ)

def sin_squared (x : ℝ) : ℝ := real.sin x * real.sin x

theorem triangle_shape (h1 : sin_squared A = sin_squared C - sin_squared B)
                       (h2 : AB = 2 * (BC - AC * real.cos C)) :
  (A + B + C = π) ∧ (C = π / 2) ∧ (B = π / 3) := 
  sorry

end triangle_shape_l267_267785


namespace bela_always_wins_l267_267141

theorem bela_always_wins (n : ℕ) (hn : n > 4) : 
  ∃ strategy : ∀ (turn : ℕ) (choices : set ℝ), Real, 
    (∀ (turn : ℕ) (choices : set ℝ), 
      (strategy turn choices ∈ set.Icc (0 : ℝ) n) 
      ∧ 
      ∀ (c : ℝ), c ∈ choices → abs (strategy turn choices - c) > 1) 
    ∧ 
    (∀ (turn : ℕ), 
      (turn % 2 = 0 → ∃ choice, choice ∈ choices ∧ choice = strategy turn choices) 
      → bela_wins (strategy turn choices)) :=
sorry

end bela_always_wins_l267_267141


namespace remainder_calculation_l267_267547

theorem remainder_calculation 
  (dividend divisor quotient : ℕ)
  (h1 : dividend = 140)
  (h2 : divisor = 15)
  (h3 : quotient = 9) :
  dividend = (divisor * quotient) + (dividend - (divisor * quotient)) := by
sorry

end remainder_calculation_l267_267547


namespace crayons_per_row_correct_l267_267679

-- Declare the given conditions
def total_crayons : ℕ := 210
def num_rows : ℕ := 7

-- Define the expected number of crayons per row
def crayons_per_row : ℕ := 30

-- The desired proof statement: Prove that dividing total crayons by the number of rows yields the expected crayons per row.
theorem crayons_per_row_correct : total_crayons / num_rows = crayons_per_row :=
by sorry

end crayons_per_row_correct_l267_267679


namespace largest_divisor_of_5_consecutive_integers_l267_267528

theorem largest_divisor_of_5_consecutive_integers :
  ∀ (n : ℤ), ∃ d, d = 120 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intro n
  use 120
  split
  exact rfl
  sorry

end largest_divisor_of_5_consecutive_integers_l267_267528


namespace number_of_people_in_group_l267_267101

-- Definitions and conditions
def total_cost : ℕ := 94
def mango_juice_cost : ℕ := 5
def pineapple_juice_cost : ℕ := 6
def pineapple_cost_total : ℕ := 54

-- Theorem statement to prove
theorem number_of_people_in_group : 
  ∃ M P : ℕ, 
    mango_juice_cost * M + pineapple_juice_cost * P = total_cost ∧ 
    pineapple_juice_cost * P = pineapple_cost_total ∧ 
    M + P = 17 := 
by 
  sorry

end number_of_people_in_group_l267_267101


namespace sphere_intersection_proof_l267_267119

theorem sphere_intersection_proof :
  ∀ (c₁ c₂ : ℝ × ℝ × ℝ) (r₁ r₂ : ℝ),
  c₁ = (1, 3, 0) →
  r₁ = 2 →
  c₂ = (0, 3, -8) →
  let center := (1, 3, -8)
  let sphere_radius := 2 * Real.sqrt 17
  let circle_radius_yz := Real.sqrt 67
  dist c₁ center = r₁ →
  dist c₂ center = circle_radius_yz
  :=
begin
  intros,
  sorry
end

end sphere_intersection_proof_l267_267119


namespace sum_of_filled_grid_is_150_l267_267769

def initial_grid : list (list (option ℕ)) :=
  [[some 1, none, none, none, none, none, some 2],
   [some 5, some 3, none, none, some 3, some 4, none],
   [none, none, none, some 2, none, none, none],
   [none, none, none, none, none, none, none],
   [none, none, none, none, some 1, none, none],
   [none, none, some 5, none, none, none, none],
   [some 4, none, none, none, none, none, none]]

def is_contiguous (grid : list (list (option ℕ))) : Prop :=
  -- This predicate should be defined based on the problem constraints
  sorry

theorem sum_of_filled_grid_is_150 (grid : list (list (option ℕ))) :
  is_contiguous grid →
  -- grid should be filled correctly with the given problem constraints
  List.sum (List.map (λ row, List.sum (List.map (λ x, x.getD 0) row)) grid) = 150 := sorry

end sum_of_filled_grid_is_150_l267_267769


namespace regular_polygon_sides_l267_267601

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ n → ∃ p, ∠(polygon_interior_angle p) = 144) : n = 10 := 
by
  sorry

end regular_polygon_sides_l267_267601


namespace number_of_special_numbers_l267_267403

def has_five_digits (n : Nat) : Prop := 10000 ≤ n ∧ n < 100000

def starts_with_two (n : Nat) : Prop := (n / 10000) = 2

def has_three_identical_digits (n : Nat) : Prop :=
  let digits := [n / 10000 % 10, n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10]
  digits.erase_dup.length = 3 ∧ 3 ∈ digits.erase_dup.map (λ d, digits.count d)

theorem number_of_special_numbers : 
  Nat.card ({n : Nat // has_five_digits n ∧ starts_with_two n ∧ has_three_identical_digits n}) = 792 :=
sorry

end number_of_special_numbers_l267_267403


namespace find_number_of_members_of_set_A_l267_267431

theorem find_number_of_members_of_set_A
  (U : Finset α)
  (A B Neither : Finset α) :
  (|U| = 193) →
  (|B| = 41) →
  (|Neither| = 59) →
  (|A ∩ B| = 23) →
  (|A| = 116) :=
by 
  sorry

end find_number_of_members_of_set_A_l267_267431


namespace option_C_option_D_l267_267240

-- Define the function f(x)
variable (f : ℝ → ℝ)

-- Define the conditions
axiom h1 : ∀ x, f(-x) = -f(x) -- f is odd
axiom h2 : ∀ x < 0, x * deriv f x - f(x) > 0 -- xf'(x) - f(x) > 0 for x < 0
axiom h3 : f(2) = 0 -- f(2) = 0

-- State the correctness of option C
theorem option_C : 4 * f (-3) + 3 * f (4) < 0 := 
sorry

-- State the correctness of option D
theorem option_D : ∀ x, f x > 0 ↔ (x ∈ set.Ioo (-∞) (-2) ∪ set.Ioo 0 2) :=
sorry

end option_C_option_D_l267_267240


namespace tan_sec_inequality_l267_267358

open Real

theorem tan_sec_inequality (x y : ℝ) (h₁ : x ≤ y) : 
  tan x - tan y ≤ sec x - sec y := 
sorry

end tan_sec_inequality_l267_267358


namespace bela_always_wins_l267_267138

noncomputable def optimal_strategy_winner (n : ℕ) (h : n > 4) : String := 
  let optimal_strategy := 
    sorry -- the implementation of the strategy is omitted
  "Bela"

theorem bela_always_wins (n : ℕ) (h : n > 4) : optimal_strategy_winner n h = "Bela" := 
  sorry -- the proof is omitted

end bela_always_wins_l267_267138


namespace jimmy_max_loss_l267_267316

-- Definition of the conditions
def exam_points : ℕ := 20
def number_of_exams : ℕ := 3
def points_lost_for_behavior : ℕ := 5
def passing_score : ℕ := 50

-- Total points Jimmy has earned and lost
def total_points : ℕ := (number_of_exams * exam_points) - points_lost_for_behavior

-- The maximum points Jimmy can lose and still pass
def max_points_jimmy_can_lose : ℕ := total_points - passing_score

-- Statement to prove
theorem jimmy_max_loss : max_points_jimmy_can_lose = 5 := 
by
  sorry

end jimmy_max_loss_l267_267316


namespace last_letter_150th_permutation_l267_267972

theorem last_letter_150th_permutation (letters : List Char) (h : letters = ['P', 'H', 'R', 'A', 'S', 'E']) :
  (letters.permutations.nth 149).get_last = 'P' :=
sorry

end last_letter_150th_permutation_l267_267972


namespace solve_equation_l267_267367

theorem solve_equation {n k l m : ℕ} (h_l : l > 1) :
  (1 + n^k)^l = 1 + n^m ↔ (n = 2 ∧ k = 1 ∧ l = 2 ∧ m = 3) :=
sorry

end solve_equation_l267_267367


namespace combination_multiplication_and_addition_l267_267649

theorem combination_multiplication_and_addition :
  (Nat.choose 10 3) * (Nat.choose 8 3) + (Nat.choose 5 2) = 6730 :=
by
  sorry

end combination_multiplication_and_addition_l267_267649


namespace total_cost_for_round_trip_l267_267047

def time_to_cross_one_way : ℕ := 4 -- time in hours to cross the lake one way
def cost_per_hour : ℕ := 10 -- cost in dollars per hour

def total_time := time_to_cross_one_way * 2 -- total time in hours for a round trip
def total_cost := total_time * cost_per_hour -- total cost in dollars for the assistant

theorem total_cost_for_round_trip : total_cost = 80 := by
  repeat {sorry} -- Leaving the proof for now

end total_cost_for_round_trip_l267_267047


namespace largest_divisor_of_five_consecutive_integers_l267_267457

open Nat

theorem largest_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, k ∈ {n, n+1, n+2, n+3, n+4} ∧
    ∀ m ∈ {2, 3, 4, 5}, m ∣ k → 60 ∣ (n * (n+1) * (n+2) * (n+3) * (n+4)) := 
sorry

end largest_divisor_of_five_consecutive_integers_l267_267457


namespace S₈_proof_l267_267859

-- Definitions
variables {a₁ q : ℝ} {S : ℕ → ℝ}

-- Condition for the sum of the first n terms of the geometric sequence
def geom_sum (n : ℕ) : ℝ :=
  a₁ * (1 - q ^ n) / (1 - q)

-- Conditions from the problem
def S₄ := geom_sum 4 = -5
def S₆ := geom_sum 6 = 21 * geom_sum 2

-- Theorem to prove
theorem S₈_proof (h₁ : S₄) (h₂ : S₆) : geom_sum 8 = -85 :=
by
  sorry

end S₈_proof_l267_267859


namespace polynomial_strictly_monotone_l267_267405

def strictly_monotone (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

theorem polynomial_strictly_monotone
  (P : ℝ → ℝ)
  (H1 : strictly_monotone (P ∘ P))
  (H2 : strictly_monotone (P ∘ P ∘ P)) :
  strictly_monotone P :=
sorry

end polynomial_strictly_monotone_l267_267405


namespace prob_X_less_than_5_l267_267082

variables (X : ℕ) (n : ℕ)
def Px_eq (k : ℕ) : ℝ := if h : k ∈ (finset.range n).image (+1) then 1 / n else 0

theorem prob_X_less_than_5 (h : finset.sum (finset.range 4) Px_eq = 0.2) : n = 20 := 
sorry

end prob_X_less_than_5_l267_267082


namespace product_of_five_consecutive_is_divisible_by_sixty_l267_267521

theorem product_of_five_consecutive_is_divisible_by_sixty (n : ℤ) :
  60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end product_of_five_consecutive_is_divisible_by_sixty_l267_267521


namespace quadratic_inequality_aux_l267_267810

theorem quadratic_inequality_aux (a b c x y z : ℝ) 
  (h1 : x + y + z = 0)
  (h2 : a + b + c ≥ 0)
  (h3 : ab + bc + ca ≥ 0) :
  ax^2 + by^2 + cz^2 ≥ 0 := 
  sorry

end quadratic_inequality_aux_l267_267810


namespace distance_from_M_to_line_NF_l267_267102

-- Conditions
def parabola : Prop := ∀ x y : ℝ, y^2 = 4 * x
def focus_F : Prop := (1 : ℝ, 0 : ℝ)
def line_through_focus_with_slope := ∀ F : ℝ × ℝ, F = (1, 0) → ∃ M : ℝ × ℝ, M.y = √3 * (M.x - 1) ∧ parabola = M
def directrix (C : Prop) := ∀ x y : ℝ, C → y = 1/4
def perpendicular_condition (N : ℝ × ℝ) := N.y = 1/4

-- Problem statement
theorem distance_from_M_to_line_NF :
  let M := (3 : ℝ, 2 * √3 : ℝ) in
  let N := (-1 : ℝ, 2 * √3 : ℝ) in
  let NF := (y = -√3 * (x - 1) : Prop) in
  let distance_from_M_to_NF := ∀ M : ℝ × ℝ, M = (3, 2 * √3) → 
                                ∀ NF : ℝ × ℝ → Prop, NF y = -√3 * (NF.x - 1) →
                                abs(3√3 + 2√3 - √3 / sqrt(3 + 1)) = 2√3 in
  distance_from_M_to_NF := 2*√3
:= sorry

end distance_from_M_to_line_NF_l267_267102


namespace nail_salon_revenue_l267_267105

theorem nail_salon_revenue :
  ∀ (manicure_cost : ℕ) (total_fingers : ℕ) (fingers_per_person : ℕ) (non_clients : ℕ),
    manicure_cost = 20 → total_fingers = 210 → fingers_per_person = 10 → non_clients = 11 →
    (total_fingers / fingers_per_person - non_clients) * manicure_cost = 200 :=
by
  intros manicure_cost total_fingers fingers_per_person non_clients
  intros h_manicure_cost h_total_fingers h_fingers_per_person h_non_clients
  rw [h_manicure_cost, h_total_fingers, h_fingers_per_person, h_non_clients]
  sorry

end nail_salon_revenue_l267_267105


namespace prob_even_heads_40_l267_267576

noncomputable def probability_even_heads (n : ℕ) : ℚ :=
  if n = 0 then 1 else
  (1/2) * (1 + (2/5) ^ n)

theorem prob_even_heads_40 :
  probability_even_heads 40 = 1/2 * (1 + (2/5) ^ 40) :=
by {
  sorry
}

end prob_even_heads_40_l267_267576


namespace k_term_weighted_jensen_l267_267354

-- Definition of the function satisfying the two-term symmetric Jensen inequality
def sym_jensen_inequality (f : ℝ → ℝ) :=
  ∀ a b λ, λ ∈ (0, 1) → f(λ * a + (1 - λ) * b) ≤ λ * f(a) + (1 - λ) * f(b)

-- Statement of the k-term weighted Jensen inequality for rational weights
theorem k_term_weighted_jensen (f : ℝ → ℝ) (x : ℕ → ℝ) (q : ℕ → ℚ) (k : ℕ) :
  sym_jensen_inequality f →
  (∀ i, i < k → q i > 0) →
  (∑ i in finset.range k, (q i : ℝ)) = 1 →
  f(∑ i in finset.range k, (q i : ℝ) * x i) ≤ ∑ i in finset.range k, (q i : ℝ) * f(x i) := 
sorry

end k_term_weighted_jensen_l267_267354


namespace f_f_2_eq_0_l267_267341

def f (x : ℝ) : ℝ :=
  if x < 1 then 2 * x - 1 else 1 / x

theorem f_f_2_eq_0 : f (f 2) = 0 :=
by
  sorry

end f_f_2_eq_0_l267_267341


namespace sum_of_products_of_roots_l267_267903

theorem sum_of_products_of_roots (p q r : ℂ) (h : 4 * (p^3) - 2 * (p^2) + 13 * p - 9 = 0 ∧ 4 * (q^3) - 2 * (q^2) + 13 * q - 9 = 0 ∧ 4 * (r^3) - 2 * (r^2) + 13 * r - 9 = 0) :
  p*q + p*r + q*r = 13 / 4 :=
  sorry

end sum_of_products_of_roots_l267_267903


namespace product_of_five_consecutive_is_divisible_by_sixty_l267_267516

theorem product_of_five_consecutive_is_divisible_by_sixty (n : ℤ) :
  60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end product_of_five_consecutive_is_divisible_by_sixty_l267_267516


namespace gasoline_reduction_l267_267757

theorem gasoline_reduction (P Q : ℝ) : 
  let new_price := 1.20 * P,
      original_spending := P * Q,
      new_spending := 1.14 * original_spending,
      Q' := new_spending / new_price in
  Q' / Q = 0.95 → (1 - Q' / Q) * 100 = 5 :=
by
  intros h1
  unfold let_let
  rw [← mul_div_cancel_left (1.14 * P * Q) (1.20 * P * H)], 
  sorry

end gasoline_reduction_l267_267757


namespace sale_price_with_different_discount_l267_267982

noncomputable def original_price (sale_price : ℝ) (discount : ℝ) : ℝ :=
  sale_price / (1 - discount)

theorem sale_price_with_different_discount
  (sale_price : ℝ) (discount1 discount2 : ℝ) :
  original_price sale_price discount1 * (1 - discount2) = 98 :=
by
  -- Given conditions
  assume h₁ : sale_price = 112
  assume h₂ : discount1 = 0.2
  assume h₃ : discount2 = 0.3
  
  -- Proof will be here
  sorry

end sale_price_with_different_discount_l267_267982


namespace equal_partition_of_cake_l267_267188

-- Definitions of points
structure Point (α : Type) :=
(x y : α)

-- Quadrilateral vertices
variables (A B C D M N P Q O : Point ℝ)

-- Midpoints
def midpoint (a b : Point ℝ) : Point ℝ :=
{ x := (a.x + b.x) / 2,
  y := (a.y + b.y) / 2 }

-- Definitions of midpoints
axiom M_def : M = midpoint A B
axiom N_def : N = midpoint B C
axiom P_def : P = midpoint C D
axiom Q_def : Q = midpoint D A

-- Intersection point of diagonals NQ and MP
axiom O_def : ∃ (t₁ t₂ : ℝ), O = { x := N.x + t₁ * (Q.x - N.x), y := N.y + t₁ * (Q.y - N.y) }
                        ∧ O = { x := M.x + t₂ * (P.x - M.x), y := M.y + t₂ * (P.y - M.y) }

-- Area definitions using parallelogram law
axiom area_AMQO : ℝ
axiom area_MBNO : ℝ
axiom area_NCPQ : ℝ
axiom area_QPDO : ℝ

-- Condition for equal areas
axiom equal_areas : 
  area_AMQO = area_MBNO ∧ area_NCPQ = area_QPDO

-- Proving that the sum of areas is equal
theorem equal_partition_of_cake :
  area_AMQO + area_NCPQ = area_MBNO + area_QPDO :=
begin
  rw equal_areas,
  sorry
end

end equal_partition_of_cake_l267_267188


namespace geometric_sequence_S8_l267_267867

theorem geometric_sequence_S8 
  (a1 q : ℝ) (S : ℕ → ℝ)
  (h1 : S 4 = a1 * (1 - q^4) / (1 - q) = -5)
  (h2 : S 6 = 21 * S 2)
  (geom_sum : ∀ n, S n = a1 * (1 - q^n) / (1 - q))
  (h3 : q ≠ 1) : S 8 = -85 := 
begin
  sorry
end

end geometric_sequence_S8_l267_267867


namespace volume_of_square_pyramid_is_correct_l267_267732

-- Define the base side length and height of the pyramid
def side_length := 20  -- cm
def height := 20  -- cm

-- Function to compute the volume of a square pyramid given side length and height
noncomputable def volume_square_pyramid (side_length height : ℝ) : ℝ :=
  (1 / 3) * (side_length ^ 2) * height

-- Theorem stating that the volume of the given square pyramid is as calculated
theorem volume_of_square_pyramid_is_correct :
  volume_square_pyramid 20 20 = 8000 / 3 := by
  -- Placeholder for the proof
  sorry

end volume_of_square_pyramid_is_correct_l267_267732


namespace percentage_paid_to_X_l267_267053

theorem percentage_paid_to_X (X Y : ℝ) (h1 : X + Y = 880) (h2 : Y = 400) : 
  ((X / Y) * 100) = 120 :=
by
  sorry

end percentage_paid_to_X_l267_267053


namespace output_in_scientific_notation_l267_267975

def output_kilowatt_hours : ℝ := 448000
def scientific_notation (n : ℝ) : Prop := n = 4.48 * 10^5

theorem output_in_scientific_notation : scientific_notation output_kilowatt_hours :=
by
  -- Proof steps are not required
  sorry

end output_in_scientific_notation_l267_267975


namespace geometric_sequence_S8_l267_267865

theorem geometric_sequence_S8 
  (a1 q : ℝ) (S : ℕ → ℝ)
  (h1 : S 4 = a1 * (1 - q^4) / (1 - q) = -5)
  (h2 : S 6 = 21 * S 2)
  (geom_sum : ∀ n, S n = a1 * (1 - q^n) / (1 - q))
  (h3 : q ≠ 1) : S 8 = -85 := 
begin
  sorry
end

end geometric_sequence_S8_l267_267865


namespace num_elements_in_T_l267_267895

theorem num_elements_in_T : ∃ T : Set ℕ, 
  (∀ n, T n ↔ (n > 1 ∧ (∃ m : ℕ, ∀ i : ℕ, (10^i / n) % 10 = (10^(i+15) / n) % 10))) ∧ 
  (T.finite.card = 7) :=
by
  let div_count := 8 -- since 10^15 - 1 has 8 divisors, including 1
  let pos_int_count := div_count - 1 -- excluding 1
  use {n | n > 1 ∧ (∃ m : ℕ, ∀ i : ℕ, (10^i / n) % 10 = (10^(i+15) / n) % 10)}
  exact ⟨by sorry, by sorry⟩

end num_elements_in_T_l267_267895


namespace line_equation_passing_through_point_and_equal_intercepts_l267_267235

theorem line_equation_passing_through_point_and_equal_intercepts :
    (∃ k: ℝ, ∀ x y: ℝ, (2, 5) = (x, k * x) ∨ x + y = 7) :=
by
  sorry

end line_equation_passing_through_point_and_equal_intercepts_l267_267235


namespace geometric_sequence_sum_eight_l267_267831

noncomputable def sum_geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ :=
  a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_eight {a1 q : ℝ} (hq : q ≠ 1) 
  (h4 : sum_geometric_sequence a1 q 4 = -5) 
  (h6 : sum_geometric_sequence a1 q 6 = 21 * sum_geometric_sequence a1 q 2) : 
  sum_geometric_sequence a1 q 8 = -85 :=
sorry

end geometric_sequence_sum_eight_l267_267831


namespace sin_sub_cos_value_tan_value_l267_267208

variable (x : ℝ)
variable h1 : -π/2 < x ∧ x < 0
variable h2 : sin x + cos x = 1/5

theorem sin_sub_cos_value : sin x - cos x = -7/5 :=
by
  sorry

theorem tan_value : tan x = -3/4 :=
by
  sorry

end sin_sub_cos_value_tan_value_l267_267208


namespace nectar_water_percentage_l267_267551

-- Definitions as per conditions
def nectar_weight : ℝ := 1.2
def honey_weight : ℝ := 1
def honey_water_ratio : ℝ := 0.4

-- Final statement to prove
theorem nectar_water_percentage : (honey_weight * honey_water_ratio + (nectar_weight - honey_weight)) / nectar_weight = 0.5 := by
  sorry

end nectar_water_percentage_l267_267551


namespace number_of_people_in_group_l267_267099

-- Definitions and conditions
def total_cost : ℕ := 94
def mango_juice_cost : ℕ := 5
def pineapple_juice_cost : ℕ := 6
def pineapple_cost_total : ℕ := 54

-- Theorem statement to prove
theorem number_of_people_in_group : 
  ∃ M P : ℕ, 
    mango_juice_cost * M + pineapple_juice_cost * P = total_cost ∧ 
    pineapple_juice_cost * P = pineapple_cost_total ∧ 
    M + P = 17 := 
by 
  sorry

end number_of_people_in_group_l267_267099


namespace loci_is_perpendicular_lines_l267_267356

structure Point :=
  (x : ℝ) (y : ℝ)

structure Rectangle :=
  (A B C D : Point)
  (O : Point) -- Intersection of diagonals
  (midpoints : Point × Point)

-- Define the distance function
def distance (P Q : Point) : ℝ :=
  Real.sqrt ((Q.x - P.x)^2 + (Q.y - P.y)^2)

-- Condition: Given distances sum condition
def distances_sum_condition (M : Point) (rect : Rectangle) :=
  distance M rect.A + distance M rect.C = distance M rect.B + distance M rect.D

-- Theorem: proving the loci is a pair of perpendicular lines
theorem loci_is_perpendicular_lines (rect : Rectangle) :
  ∀ M : Point, distances_sum_condition M rect →
    M ∈ (line_through (rect.midpoints.1) (line_perpendicular (rect.A) (rect.midpoints.1))) ∨ 
    M ∈ (line_through (rect.midpoints.2) (line_perpendicular (rect.B) (rect.midpoints.2))) :=
sorry

end loci_is_perpendicular_lines_l267_267356


namespace strawberries_taken_out_l267_267646

theorem strawberries_taken_out : 
  ∀ (initial_total_strawberries buckets strawberries_left_per_bucket : ℕ),
  initial_total_strawberries = 300 → 
  buckets = 5 → 
  strawberries_left_per_bucket = 40 → 
  (initial_total_strawberries / buckets - strawberries_left_per_bucket = 20) :=
by
  intros initial_total_strawberries buckets strawberries_left_per_bucket h1 h2 h3
  sorry

end strawberries_taken_out_l267_267646


namespace incircle_shared_l267_267441

noncomputable def common_incircle (ABC DEF : Triangle) : Circumcircle :=
sorry

theorem incircle_shared {A B C D E F : Point} 
  (h1 : have_common_incircle (triangle A B C) (triangle D E F)) 
  (h2 : lies_on_circumcircle A B C D E) : 
  lies_on_circumcircle A B C D E F :=
sorry

end incircle_shared_l267_267441


namespace xiao_wang_ways_to_make_8_cents_l267_267558

theorem xiao_wang_ways_to_make_8_cents :
  let one_cent_coins := 8
  let two_cent_coins := 4
  let five_cent_coin := 1
  ∃ ways, ways = 7 ∧ (
       (ways = 8 ∧ one_cent_coins >= 8) ∨
       (ways = 4 ∧ two_cent_coins >= 4) ∨
       (ways = 2 ∧ one_cent_coins >= 2 ∧ two_cent_coins >= 3) ∨
       (ways = 4 ∧ one_cent_coins >= 4 ∧ two_cent_coins >= 2) ∨
       (ways = 6 ∧ one_cent_coins >= 6 ∧ two_cent_coins >= 1) ∨
       (ways = 3 ∧ one_cent_coins >= 3 ∧ five_cent_coin >= 1) ∨
       (ways = 1 ∧ one_cent_coins >= 1 ∧ two_cent_coins >= 1 ∧ five_cent_coin >= 1)
   ) :=
  sorry

end xiao_wang_ways_to_make_8_cents_l267_267558


namespace custom_op_value_l267_267661

-- Define the custom operation (a \$ b)
def custom_op (a b : Int) : Int := a * (b - 1) + a * b

-- Main theorem to prove the equivalence
theorem custom_op_value : custom_op 5 (-3) = -35 := by
  sorry

end custom_op_value_l267_267661


namespace no_integer_solutions_m2n_eq_2mn_minus_3_l267_267667

theorem no_integer_solutions_m2n_eq_2mn_minus_3 :
  ∀ (m n : ℤ), m + 2 * n ≠ 2 * m * n - 3 := 
sorry

end no_integer_solutions_m2n_eq_2mn_minus_3_l267_267667


namespace total_cats_and_kittens_received_l267_267807

theorem total_cats_and_kittens_received
  (adult_cats : ℕ)
  (fraction_females : ℚ)
  (fraction_litters : ℚ)
  (average_kittens_per_litter : ℕ)
  (h1 : adult_cats = 120)
  (h2 : fraction_females = 2 / 3)
  (h3 : fraction_litters = 2 / 5)
  (h4 : average_kittens_per_litter = 5) :
  adult_cats + (fraction_litters * fraction_females * adult_cats * average_kittens_per_litter).toNat = 280 :=
by 
  sorry

end total_cats_and_kittens_received_l267_267807


namespace vector_norm_solution_l267_267211

-- Definitions for vectors a and b, and various vector operations
def vec2 : Type := (ℝ × ℝ)

def a : vec2 := (-1, 3)
def b (t : ℝ) : vec2 := (1, t)

def dot (u v : vec2) : ℝ := u.1 * v.1 + u.2 * v.2

def perpendicular (u v : vec2) : Prop := dot u v = 0

def norm (v : vec2) : ℝ := Real.sqrt (dot v v)

-- The statement to be proved
theorem vector_norm_solution :
  ∀ t : ℝ, perpendicular (a.1 - 2 * (b t).1, a.2 - 2 * (b t).2) a →
    norm (b 2) = Real.sqrt 5 :=
by
  sorry

end vector_norm_solution_l267_267211


namespace gcd_linear_combination_l267_267189

theorem gcd_linear_combination (a b : ℤ) (h : Int.gcd a b = 1) : 
    Int.gcd (11 * a + 2 * b) (18 * a + 5 * b) = 1 := 
by
  sorry

end gcd_linear_combination_l267_267189


namespace product_of_sides_le_four_l267_267353

theorem product_of_sides_le_four 
  (Q : Type*) [convex Q] [covered_by_unit_disk Q] :
  sides_product Q ≤ 4 :=
sorry

end product_of_sides_le_four_l267_267353


namespace product_of_five_consecutive_integers_divisible_by_120_l267_267450

theorem product_of_five_consecutive_integers_divisible_by_120 (n : ℤ) : 
  120 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end product_of_five_consecutive_integers_divisible_by_120_l267_267450


namespace complex_number_forms_l267_267677

theorem complex_number_forms :
  let z := 3 - 3 * Complex.i * Real.sqrt 3 in
  z = 6 * (Complex.cos (5 * Real.pi / 3) + Complex.i * Complex.sin (5 * Real.pi / 3))
  ∧ z = 6 * Complex.exp (Complex.i * (5 * Real.pi / 3)) :=
by
  let z := 3 - 3 * Complex.i * Real.sqrt 3
  -- missing proof
  sorry

end complex_number_forms_l267_267677


namespace sum_of_largest_smallest_angles_l267_267243

noncomputable section

def sides_ratio (a b c : ℝ) : Prop := a / 5 = b / 7 ∧ b / 7 = c / 8

theorem sum_of_largest_smallest_angles (a b c : ℝ) (θA θB θC : ℝ) 
  (h1 : sides_ratio a b c) 
  (h2 : a^2 + b^2 - c^2 = 2 * a * b * Real.cos θC)
  (h3 : b^2 + c^2 - a^2 = 2 * b * c * Real.cos θA)
  (h4 : c^2 + a^2 - b^2 = 2 * c * a * Real.cos θB)
  (h5 : θA + θB + θC = 180) :
  θA + θC = 120 :=
sorry

end sum_of_largest_smallest_angles_l267_267243


namespace large_green_curlers_l267_267181

-- Define the number of total curlers
def total_curlers : ℕ := 16

-- Define the fraction for pink curlers
def pink_fraction : ℕ := 1 / 4

-- Define the number of pink curlers
def pink_curlers : ℕ := pink_fraction * total_curlers

-- Define the number of blue curlers
def blue_curlers : ℕ := 2 * pink_curlers

-- Define the total number of pink and blue curlers
def pink_and_blue_curlers : ℕ := pink_curlers + blue_curlers

-- Define the number of green curlers
def green_curlers : ℕ := total_curlers - pink_and_blue_curlers

-- Theorem stating the number of green curlers is 4
theorem large_green_curlers : green_curlers = 4 := by
  -- Proof would go here
  sorry

end large_green_curlers_l267_267181


namespace product_of_five_consecutive_integers_divisible_by_120_l267_267452

theorem product_of_five_consecutive_integers_divisible_by_120 (n : ℤ) : 
  120 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end product_of_five_consecutive_integers_divisible_by_120_l267_267452


namespace angle_AGF_measure_l267_267765

-- Define the setting of a regular hexagon and the diagonals
variables (A B C D E F G : Type) 
variable [shape : regular_hexagon A B C D E F]
variable (h_intersect : intersect AC DF G)
variable (h_int_angle : ∀ (P : Type), interior_angle P (120 : ℤ))

-- State the problem
theorem angle_AGF_measure : measure_angle A G F = 120 :=
by
  sorry

end angle_AGF_measure_l267_267765


namespace sum_of_eight_l267_267877

variable (a₁ : ℕ) (q : ℕ)
variable (S : ℕ → ℕ) -- Assume S is a function from natural numbers to natural numbers representing S_n

-- Condition 1: Sum of first 4 terms equals -5
axiom h1 : S 4 = -5

-- Condition 2: Sum of the first 6 terms is 21 times the sum of the first 2 terms
axiom h2 : S 6 = 21 * S 2

-- The formula for the sum of the first n terms of a geometric sequence
def sum_of_first_n_terms (n : ℕ) := a₁ * (1 - q^n) / (1 - q)

-- Statement to be proved
theorem sum_of_eight : S 8 = -85 := sorry

end sum_of_eight_l267_267877


namespace EQ_perpendicular_AB_l267_267787

variables {A B C I E F P Q : Type}
variables (AI AC AB : A)
variables (incircle : I)
variables (touches_AC_E : touches incircle AC E)
variables (touches_AB_F : touches incircle AB F)
variables (perpendicular_bisector_AI : perpendicular_bisector (seg AI) AC P)
variables (QI_perpendicular_FP : QI ⊥ seg FP)
variables (QI_on_AB : Q ∈ AB)

theorem EQ_perpendicular_AB 
  (h1 : touches incircle AC E)
  (h2 : touches incircle AB F)
  (h3 : perpendicular_bisector (seg AI) AC P)
  (h4 : QI ⊥ seg FP)
  (h5 : Q ∈ AB) :
  EQ ⊥ AB :=
by sorry

end EQ_perpendicular_AB_l267_267787


namespace eight_term_sum_l267_267816

variable {α : Type*} [Field α]
variable (a q : α)

-- Define the n-th sum of the geometric sequence
def S_n (n : ℕ) : α := a * (1 - q ^ n) / (1 - q)

-- Given conditions
def S4 : α := S_n 4 = -5
def S6 : α := S_n 6 = 21 * S_n 2

-- Prove the target statement
theorem eight_term_sum : S_n 8 = -85 :=
  sorry

end eight_term_sum_l267_267816


namespace regular_polygon_sides_l267_267610

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, interior_angle n i = 144) : n = 10 :=
sorry

end regular_polygon_sides_l267_267610


namespace union_A_B_complement_intersection_A_B_l267_267344

open Set

variable (𝕌 : Type) (A B : Set ℝ)

def universal_set : Set ℝ := univ

def set_A : Set ℝ := { x | -2 ≤ x ∧ x < 4 }

def set_B : Set ℝ := { x | x ≥ 3 }

theorem union_A_B : A ∪ B = { x | x ≥ -2 } :=
begin
  sorry
end

theorem complement_intersection_A_B : (universal_set \ (A ∩ B)) = { x | x < 3 ∨ x ≥ 4 } :=
begin
  sorry
end

end union_A_B_complement_intersection_A_B_l267_267344


namespace total_cost_for_round_trip_l267_267048

def time_to_cross_one_way : ℕ := 4 -- time in hours to cross the lake one way
def cost_per_hour : ℕ := 10 -- cost in dollars per hour

def total_time := time_to_cross_one_way * 2 -- total time in hours for a round trip
def total_cost := total_time * cost_per_hour -- total cost in dollars for the assistant

theorem total_cost_for_round_trip : total_cost = 80 := by
  repeat {sorry} -- Leaving the proof for now

end total_cost_for_round_trip_l267_267048


namespace solution1_solution2_solution3_solution4_l267_267368

-- 1st problem translation
def problem1 (x : ℚ) : Prop := 
  (9 / 10) * x = (3 / 4) * 2

-- 2nd problem translation
def problem2 (x : ℚ) : Prop := 
  (0.75 * x) = 0.5 * 6

-- 3rd problem translation
def problem3 (x : ℚ) : Prop : = 
  5 * x = 2 * 20

-- 4th problem translation
def problem4 (x : ℚ) : Prop := 
  (1 / 20) * x = 2 / 3

theorem solution1 : ∃ x : ℚ, problem1 x ∧ x = 5 / 3 :=
by
  use 5 / 3
  split
  exact sorry
  exact sorry

theorem solution2 : ∃ x : ℚ, problem2 x ∧ x = 4 :=
by
  use 4
  split
  exact sorry
  exact sorry

theorem solution3 : ∃ x : ℚ, problem3 x ∧ x = 8 :=
by
  use 8
  split
  exact sorry
  exact sorry

theorem solution4 : ∃ x : ℚ, problem4 x ∧ x = 40 / 3 :=
by
  use 40 / 3
  split
  exact sorry
  exact sorry


end solution1_solution2_solution3_solution4_l267_267368


namespace volume_ratio_l267_267742

-- Define the edge of the cube and calculate its volume
def cube_edge := 1 -- in meters
def cube_volume := cube_edge ^ 3

-- Define the dimensions of the cuboid in meters and calculate its volume
def cuboid_width := 0.5 -- in meters
def cuboid_length := 0.5 -- in meters
def cuboid_height := 0.2 -- in meters
def cuboid_volume := cuboid_width * cuboid_length * cuboid_height

-- Define the number of times the cube's volume is the cuboid's volume
def number_of_times := cube_volume / cuboid_volume

theorem volume_ratio : number_of_times = 20 :=
by
  -- The proof is not required
  sorry

end volume_ratio_l267_267742


namespace distinct_remainders_mod_mk_same_remainders_mod_mk_l267_267572

-- Part 1: For (m, k) = 1
theorem distinct_remainders_mod_mk {m k : ℕ} (hmk : Nat.gcd m k = 1) :
  ∃ (a : Fin m → ℕ) (b : Fin k → ℕ),
  ∀ (i : Fin m) (j : Fin k), (∀ (i' : Fin m) (j' : Fin k), (i, j) ≠ (i', j') → (a i * b j) % (m * k) ≠ (a i' * b j') % (m * k)) :=
sorry

-- Part 2: For (m, k) > 1
theorem same_remainders_mod_mk {m k : ℕ} (hmk : Nat.gcd m k > 1) :
  ∀ (a : Fin m → ℕ) (b : Fin k → ℕ),
  ∃ (i j : Fin m) (s t : Fin k), (i, j) ≠ (s, t) ∧ (a i * b j) % (m * k) = (a s * b t) % (m * k) :=
sorry

end distinct_remainders_mod_mk_same_remainders_mod_mk_l267_267572


namespace bela_always_wins_l267_267139

noncomputable def optimal_strategy_winner (n : ℕ) (h : n > 4) : String := 
  let optimal_strategy := 
    sorry -- the implementation of the strategy is omitted
  "Bela"

theorem bela_always_wins (n : ℕ) (h : n > 4) : optimal_strategy_winner n h = "Bela" := 
  sorry -- the proof is omitted

end bela_always_wins_l267_267139


namespace sandyPhoneBill_is_340_l267_267935

namespace SandyPhoneBill

variable (sandyAgeNow : ℕ) (kimAgeNow : ℕ) (sandyPhoneBill : ℕ)

-- Conditions
def kimCurrentAge := kimAgeNow = 10
def sandyFutureAge := sandyAgeNow + 2 = 3 * (kimAgeNow + 2)
def sandyPhoneBillDefinition := sandyPhoneBill = 10 * sandyAgeNow

-- Target proof
theorem sandyPhoneBill_is_340 
  (h1 : kimCurrentAge)
  (h2 : sandyFutureAge)
  (h3 : sandyPhoneBillDefinition) :
  sandyPhoneBill = 340 :=
sorry

end SandyPhoneBill

end sandyPhoneBill_is_340_l267_267935


namespace art_class_students_not_in_science_l267_267634

theorem art_class_students_not_in_science (n S A S_inter_A_only_A : ℕ) 
  (h_n : n = 120) 
  (h_S : S = 85) 
  (h_A : A = 65) 
  (h_union: n = S + A - S_inter_A_only_A) : 
  S_inter_A_only_A = 30 → 
  A - S_inter_A_only_A = 35 :=
by
  intros h
  rw [h]
  sorry

end art_class_students_not_in_science_l267_267634


namespace intersection_x_coordinate_l267_267172

-- Definitions based on conditions
def line1 (x : ℝ) : ℝ := 3 * x + 5
def line2 (x : ℝ) : ℝ := 35 - 5 * x

-- Proof statement
theorem intersection_x_coordinate : ∃ x : ℝ, line1 x = line2 x ∧ x = 15 / 4 :=
by
  use 15 / 4
  sorry

end intersection_x_coordinate_l267_267172


namespace arc_length_TQ_l267_267763

-- Definitions from the conditions
def center (O : Type) : Prop := true
def inscribedAngle (T I Q : Type) (angle : ℝ) := angle = 45
def radius (T : Type) (len : ℝ) := len = 12

-- Theorem to be proved
theorem arc_length_TQ (O T I Q : Type) (r : ℝ) (angle : ℝ) 
  (h_center : center O) 
  (h_angle : inscribedAngle T I Q angle)
  (h_radius : radius T r) :
  ∃ l : ℝ, l = 6 * Real.pi := 
sorry

end arc_length_TQ_l267_267763


namespace extreme_points_of_f_l267_267166

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := ln x + (1 / 2) * x^2 + a * x
noncomputable def g (x : ℝ) : ℝ := exp x + (3 / 2) * x^2

theorem extreme_points_of_f (a : ℝ) : (∀ x > 0, ln x + (1 / 2) * x^2 + a * x ≤ exp x + (3 / 2) * x^2 → a ≤ Real.exp 1 + 1) ∧ 
  (if a ∈ set.Icc (-2 : ℝ) (Real.top) then ∀ x > 0, ¬ ∃ x1 x2, f a x1 = f a x2 else 
   if a ∈ set.Iio (-2 : ℝ) then ∃ x1 x2, f a x1 = f a x2 else false) := 
by sorry

end extreme_points_of_f_l267_267166


namespace seating_arrangements_l267_267992

theorem seating_arrangements : ∃ n : ℕ, n = 100 ∧
  ∀ (seating : Fin 7 → Option Char),
  (∃ (A_pos : Fin 7) (B_pos : Fin 7) (C_pos : Fin 7),
    seating A_pos = some 'A' ∧ seating B_pos = some 'B' ∧ seating C_pos = some 'C' ∧
    (abs (A_pos.val - B_pos.val) > 1) ∧ (abs (A_pos.val - C_pos.val) > 1)) :=
sorry

end seating_arrangements_l267_267992


namespace enclosed_area_f_g_l267_267307

def f (a x : ℝ) : ℝ := a * sin (a * x) + cos (a * x)
def g (a : ℝ) : ℝ := sqrt (a^2 + 1)

theorem enclosed_area_f_g (a : ℝ) (h : a > 0) :
  ∫ x in 0..((2 * π) / a), |f a x - g a| = (2 * π * sqrt (a^2 + 1)) / a :=
by
  sorry

end enclosed_area_f_g_l267_267307


namespace completing_square_result_l267_267013

theorem completing_square_result:
  ∀ x : ℝ, (x^2 + 4 * x - 1 = 0) → (x + 2)^2 = 5 :=
by
  sorry

end completing_square_result_l267_267013


namespace five_consecutive_product_div_24_l267_267491

theorem five_consecutive_product_div_24 (n : ℤ) : 
  24 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) := 
sorry

end five_consecutive_product_div_24_l267_267491


namespace largest_divisor_of_5_consecutive_integers_l267_267532

theorem largest_divisor_of_5_consecutive_integers :
  ∀ (n : ℤ), ∃ d, d = 120 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intro n
  use 120
  split
  exact rfl
  sorry

end largest_divisor_of_5_consecutive_integers_l267_267532


namespace incenter_circumcircles_intersection_l267_267758

variables {A B C D E P I : Type*} 
variables [triangle_ABC : Triangle A B C]
variables [shortest_side_BC : Side BC]
variables [on_side_AB : OnSide D AB]
variables [on_side_AC : OnSide E AC]
variables [equal_sides1 : EqualSides DB BC]
variables [equal_sides2 : EqualSides BC CE]
variables [intersection_BE_CD : Intersection P BE CD]
variables [incenter : Incenter I A B C]

theorem incenter_circumcircles_intersection :
  OnCircumcircle I B D P ∧ OnCircumcircle I C E P := 
sorry

end incenter_circumcircles_intersection_l267_267758


namespace square_perimeter_sum_area_l267_267074

theorem square_perimeter_sum_area (P₁ P₂ : ℝ) (hP₁ : P₁ = 24) (hP₂ : P₂ = 32) : 
  let s₁ := P₁ / 4 in
  let s₂ := P₂ / 4 in
  let A₁ := s₁ ^ 2 in
  let A₂ := s₂ ^ 2 in
  let A₃ := A₁ + A₂ in
  let s₃ := Real.sqrt A₃ in 
  4 * s₃ = 40 :=
by
  sorry

end square_perimeter_sum_area_l267_267074


namespace sum_of_eight_l267_267875

variable (a₁ : ℕ) (q : ℕ)
variable (S : ℕ → ℕ) -- Assume S is a function from natural numbers to natural numbers representing S_n

-- Condition 1: Sum of first 4 terms equals -5
axiom h1 : S 4 = -5

-- Condition 2: Sum of the first 6 terms is 21 times the sum of the first 2 terms
axiom h2 : S 6 = 21 * S 2

-- The formula for the sum of the first n terms of a geometric sequence
def sum_of_first_n_terms (n : ℕ) := a₁ * (1 - q^n) / (1 - q)

-- Statement to be proved
theorem sum_of_eight : S 8 = -85 := sorry

end sum_of_eight_l267_267875


namespace det_D_eq_l267_267809

noncomputable def matrix_D (n : ℕ) : Matrix (Fin n) (Fin n) ℝ :=
  Matrix.of (λ i j => | (i : ℕ) - (j : ℕ) | )

theorem det_D_eq (n : ℕ) : 
  Matrix.det (matrix_D n) = (-1)^(n-1) * (n-1) * 2^(n-2) := 
sorry

end det_D_eq_l267_267809


namespace center_of_circle_l267_267195

theorem center_of_circle : ∀ (x y : ℝ), x^2 + y^2 = 4 * x - 6 * y + 9 → (x, y) = (2, -3) :=
by
sorry

end center_of_circle_l267_267195


namespace sally_four_digit_numbers_count_l267_267362

/-- Sally is thinking of a positive four-digit integer. When she divides it by any one-digit integer
greater than 1, the remainder is 1. How many possible values are there for Sally's four-digit number? -/
theorem sally_four_digit_numbers_count :
  let N_set := {N : ℕ | 1000 ≤ N ∧ N < 10000 ∧ ∀ d ∈ {2, 3, 4, 5, 6, 7, 8, 9}, N % d = 1} in
  N_set.card = 3 :=
by
  sorry

end sally_four_digit_numbers_count_l267_267362


namespace twenty_four_game_solution_l267_267993

def rational_numbers : List ℚ := [3, 4, -6, 10]

def valid_methods (num_list : List ℚ) : Prop :=
  (num_list.head! + num_list.tail! 1 + num_list.tail! 2) * num_list.head! = 24 ∧
  (num_list.head! - num_list.tail! 1) * num_list.tail! 2 - (- num_list.tail! 3) = 24

theorem twenty_four_game_solution : valid_methods rational_numbers :=
by
  sorry

end twenty_four_game_solution_l267_267993


namespace largest_divisor_of_product_of_five_consecutive_integers_l267_267499

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∀ (n : ℤ), ∃ k : ℤ, k = 60 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intro n
  use 60
  split
  { refl }
  { sorry }

end largest_divisor_of_product_of_five_consecutive_integers_l267_267499


namespace find_angles_of_triangle_l267_267953

-- Defining the geometric setup and conditions
variables {A B C O M H : Type} -- Points in the triangle and intersection

-- Conditions
variables [is_triangle A B C] -- Triangle ABC
variables [angle_bisectors A M] [angle_bisectors B H] -- AM and BH are angle bisectors
variables (O : Type) -- Intersection point of bisectors AM and BH
variables (h1 : AO / MO = sqrt (3)) -- Condition 1
variables (h2 : HO / BO = sqrt (3) - 1) -- Condition 2

-- Main theorem
theorem find_angles_of_triangle :
  ∠ABC = 30 ∧ ∠BCA = 60 ∧ ∠CAB = 90 :=
begin
  sorry, -- Proof required
end

end find_angles_of_triangle_l267_267953


namespace milk_production_l267_267016

variables (x α y z w β v : ℝ)

theorem milk_production :
  (w * v * β * y) / (α^2 * x * z) = β * y * w * v / (α^2 * x * z) := 
by
  sorry

end milk_production_l267_267016


namespace convert_to_scientific_notation_l267_267978

theorem convert_to_scientific_notation :
  (448000 : ℝ) = 4.48 * 10^5 :=
by
  sorry

end convert_to_scientific_notation_l267_267978


namespace divides_127_l267_267619

noncomputable def a (n : ℕ) : ℕ :=
if n = 1 then 1 else if n = 2 then 0 else 
let S := λ (k : ℕ), ∑ i in finset.range k, a i in
(S (n-1) + 1) * S (n-2)

def S (k : ℕ) : ℕ := ∑ i in finset.range k, a i

theorem divides_127 (m : ℕ) (h1 : m > 2) : 127 ∣ a m → m = 16 := sorry

end divides_127_l267_267619


namespace complex_number_quadrant_l267_267700

theorem complex_number_quadrant (m : ℝ) (h : m < 1) : 
  (1 - m + complex.i).re > 0 ∧ (1 - m + complex.i).im > 0 :=
by
  sorry

end complex_number_quadrant_l267_267700


namespace five_consecutive_product_div_24_l267_267492

theorem five_consecutive_product_div_24 (n : ℤ) : 
  24 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) := 
sorry

end five_consecutive_product_div_24_l267_267492


namespace find_a_plus_b_l267_267253

def M (a b : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := ![![1, a], ![-1, b]]

def α : Matrix (Fin 2) (Fin 1) ℝ := ![![2], ![1]]

def λ : ℝ := 2

theorem find_a_plus_b (a b : ℝ) (h : M a b ⬝ α = λ • α) : a + b = 6 :=
by sorry

end find_a_plus_b_l267_267253


namespace value_that_number_exceeds_l267_267592

theorem value_that_number_exceeds (V : ℤ) (h : 69 = V + 3 * (86 - 69)) : V = 18 :=
by
  sorry

end value_that_number_exceeds_l267_267592


namespace problem_sum_of_coefficients_l267_267325

theorem problem_sum_of_coefficients {n : ℕ} : 
  let g (x : ℝ) := (1 + x + x^3)^n,
      coeffs := (0 .. 3 * n + 1).map (λ k, ((one_plus_x_plus_x_cubed_pow_n n).coeff k)),
      t := coeffs.filter (λ i, i % 3 = 0).sum in
  t = 2^n :=
by
  sorry

end problem_sum_of_coefficients_l267_267325


namespace pet_purchase_ways_l267_267595

theorem pet_purchase_ways :
  let num_puppies := 10 in
  let num_kittens := 6 in
  let num_hamsters := 8 in
  ∃ (ways : ℕ),
    (ways = (num_puppies * num_kittens * num_hamsters) + (num_kittens * num_puppies * num_hamsters)) ∧
    (ways = 960) :=
by
  let num_puppies := 10;
  let num_kittens := 6;
  let num_hamsters := 8;
  use (num_puppies * num_kittens * num_hamsters) + (num_kittens * num_puppies * num_hamsters);
  split;
  {
    -- Prove the calculation matches 960
    calc
      (num_puppies * num_kittens * num_hamsters) + (num_kittens * num_puppies * num_hamsters)
        = (10 * 6 * 8) + (6 * 10 * 8) : by rfl
    ... = 480 + 480 : by sorry -- Perform the intermediate multiplications
    ... = 960 : by sorry -- Perform the final addition
  },
  {
    -- Prove it the answer matches 960
    rfl
  }

end pet_purchase_ways_l267_267595


namespace product_of_b_eq_neg_40_over_3_l267_267664

def f (b x : ℝ) := b / (3 * x - 4)

theorem product_of_b_eq_neg_40_over_3 :
  (forall b : ℝ, f b 3 = f b⁻¹ (b + 2)) →
  (b ≠ 20 / 3) →
  (∑ b in roots_of_quadratic (3 : ℝ) (-19) (-40), b) = (-40 / 3 : ℝ) :=
sorry

end product_of_b_eq_neg_40_over_3_l267_267664


namespace regular_polygon_sides_l267_267607

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, interior_angle n i = 144) : n = 10 :=
sorry

end regular_polygon_sides_l267_267607


namespace three_digit_number_is_495_l267_267435

/-- A three-digit number whose digits are not all the same is represented as xyz.
    The maximum number formed by rearrangement of its digits is abc, and the minimum is cba.
    If the difference between the obtained maximum and minimum number is the original
    three-digit number, prove that this original number is 495. --/
theorem three_digit_number_is_495 (x y z a b c : ℕ) :
  let n := 100 * x + 10 * y + z,
      max_num := 100 * a + 10 * b + c,
      min_num := 100 * c + 10 * b + a in
  (a ≥ b ∧ b ≥ c) →
  (c < a) →
  max_num - min_num = n →
  n = 495 :=
by
  intros
  sorry

end three_digit_number_is_495_l267_267435


namespace find_c_d_l267_267749

theorem find_c_d (y c d : ℕ) (H1 : y = c + Real.sqrt d) (H2 : y^2 + 4 * y + 4 / y + 1 / (y^2) = 30) :
  c + d = 5 :=
sorry

end find_c_d_l267_267749


namespace Brandon_can_still_apply_l267_267145

-- Definitions based on the given conditions
def total_businesses : ℕ := 72
def fired_businesses : ℕ := total_businesses / 2
def quit_businesses : ℕ := total_businesses / 3
def businesses_restricted : ℕ := fired_businesses + quit_businesses

-- The final proof statement
theorem Brandon_can_still_apply : total_businesses - businesses_restricted = 12 :=
by
  -- Note: Proof is omitted; replace sorry with detailed proof in practice.
  sorry

end Brandon_can_still_apply_l267_267145


namespace tetrahedron_areas_l267_267043

-- Define the conditions
def OA := 7
def OB := 2
def OC := 6
def ∠AOB := 90
def ∠AOC := 90
def ∠BOC := 90

-- Define the areas of the triangles
def area_triangle_OAB := (1 / 2) * OA * OB
def area_triangle_OAC := (1 / 2) * OA * OC
def area_triangle_OBC := (1 / 2) * OB * OC
def area_triangle_ABC := Math.sqrt((area_triangle_OAC)^2 + (area_triangle_OBC)^2 - (area_triangle_OAB)^2)

-- Define the sum of squares
def value := (area_triangle_OAB)^2 + (area_triangle_OAC)^2 + (area_triangle_OBC)^2 + (area_triangle_ABC)^2

-- The statement to prove
theorem tetrahedron_areas : value = 1052 := by sorry

end tetrahedron_areas_l267_267043


namespace count_integer_values_l267_267423

theorem count_integer_values (x : ℕ) : 5 < Real.sqrt x ∧ Real.sqrt x < 6 → 
  (finset.card (finset.filter (λ n, 5 < Real.sqrt n ∧ Real.sqrt n < 6) (finset.range 37))) = 10 :=
by
  sorry

end count_integer_values_l267_267423


namespace geometric_sequence_S8_l267_267868

theorem geometric_sequence_S8 
  (a1 q : ℝ) (S : ℕ → ℝ)
  (h1 : S 4 = a1 * (1 - q^4) / (1 - q) = -5)
  (h2 : S 6 = 21 * S 2)
  (geom_sum : ∀ n, S n = a1 * (1 - q^n) / (1 - q))
  (h3 : q ≠ 1) : S 8 = -85 := 
begin
  sorry
end

end geometric_sequence_S8_l267_267868


namespace inequality_abc_l267_267709

theorem inequality_abc (a b c : ℝ) 
  (habc : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ ab + bc + ca = 1) :
  (1 / (a + b) + 1 / (b + c) + 1 / (c + a) ≥ 5 / 2) :=
sorry

end inequality_abc_l267_267709


namespace large_circle_radius_l267_267052

noncomputable def radius_of_large_circle (R : ℝ) : Prop :=
  ∃ r : ℝ, (r = 2) ∧
           (R = r + r) ∧
           (r = 2) ∧
           (R - r = 2) ∧
           (R = 4)

theorem large_circle_radius :
  radius_of_large_circle 4 :=
by
  sorry

end large_circle_radius_l267_267052


namespace system_of_equations_solution_l267_267014

theorem system_of_equations_solution (x y z : ℝ) 
  (h1 : x + 2 * y = 4)
  (h2 : 2 * x + 5 * y - 2 * z = 11)
  (h3 : 3 * x - 5 * y + 2 * z = -1) : 
  x = 2 ∧ y = 1 ∧ z = -1 :=
by {
  sorry
}

end system_of_equations_solution_l267_267014


namespace intersection_of_A_and_B_l267_267255

def U := Set ℝ
def A := {x : ℝ | -2 ≤ x ∧ x ≤ 3 }
def B := {x : ℝ | x < -1}
def C := {x : ℝ | -2 ≤ x ∧ x < -1}

theorem intersection_of_A_and_B : A ∩ B = C :=
by sorry

end intersection_of_A_and_B_l267_267255


namespace eval_g_4_times_l267_267339

def g (x : ℕ) : ℕ :=
  if x % 3 == 0 then x / 3 else 2 * x + 1

theorem eval_g_4_times (h : g(g(g(g(6)))) = 23) : g(g(g(g(6)))) = 23 := 
by
  sorry

end eval_g_4_times_l267_267339


namespace maximal_separated_sequences_l267_267323

theorem maximal_separated_sequences (n : ℕ) (h_n : n ≥ 2) (X : Finset (Fin (n+1))) :
  ∃ S : Finset (Vector (Fin (n+1)) n), 
    (∀ (a b ∈ S), a ≠ b → ∃ (i j : ℕ), i ≠ j ∧ a.nth i = b.nth j) ∧
    S.card = (n+1)! / 2 :=
sorry

end maximal_separated_sequences_l267_267323


namespace part1_part2_l267_267929

theorem part1 :
  let a := (Real.sqrt 13) - 3
  let b := 5
  a + b - (Real.sqrt 13) = 2 := 
by {
  let a := (Real.sqrt 13) - 3
  let b := 5
  calc 
    a + b - Real.sqrt 13 
        = (Real.sqrt 13 - 3) + 5 - Real.sqrt 13 : by rfl
    ... = 2 : by linarith
}

theorem part2 :
  let x := 13
  let y := Real.sqrt 3 - 1
  x - y = 14 - Real.sqrt 3 -> 
  - (x - y) = Real.sqrt 3 - 14 := 
by {
  intro h
  have : x - y = 14 - Real.sqrt 3 := h
  simp [*, sub_eq_add_neg, add_comm] at *,
}

end part1_part2_l267_267929


namespace carla_coin_die_probability_l267_267645

theorem carla_coin_die_probability :
  let age := 11
  let coin_faces := {5, 15}
  let die_faces := {1, 2, 3, 4, 5, 6}
  let coin_probability := 1 / 2
  let die_probability := 1 / 6
  let target_sum := age
  let valid_combination_probability := coin_probability * die_probability
  target_sum ∈ {c + d | c ∈ coin_faces, d ∈ die_faces} →
  target_sum = age →
  target_sum = 11 →
  valid_combination_probability = 1 / 12
:= by
  sorry

end carla_coin_die_probability_l267_267645


namespace count_integer_values_l267_267421

theorem count_integer_values (x : ℕ) : 5 < Real.sqrt x ∧ Real.sqrt x < 6 → 
  (finset.card (finset.filter (λ n, 5 < Real.sqrt n ∧ Real.sqrt n < 6) (finset.range 37))) = 10 :=
by
  sorry

end count_integer_values_l267_267421


namespace angle_EDC_is_180_l267_267281

-- Definitions based on conditions
variables {A B C D E : Point}
variables (triangle_ABC : Triangle A B C)
variables (AD_bisects_BAC : bisects (angle_of_triangle triangle_ABC A B C) A D)
variables (BC_extended_to_E : extends B C E)
variable (angle_ABE_90 : angle B A E = 90)

-- The theorem we need to prove
theorem angle_EDC_is_180 (h1 : bisects (angle_of_triangle triangle_ABC A B C) A D)
                         (h2 : extends B C E)
                         (h3 : angle B A E = 90) :
                         angle E D C = 180 :=
by 
  sorry

end angle_EDC_is_180_l267_267281


namespace large_green_curlers_l267_267180

-- Define the number of total curlers
def total_curlers : ℕ := 16

-- Define the fraction for pink curlers
def pink_fraction : ℕ := 1 / 4

-- Define the number of pink curlers
def pink_curlers : ℕ := pink_fraction * total_curlers

-- Define the number of blue curlers
def blue_curlers : ℕ := 2 * pink_curlers

-- Define the total number of pink and blue curlers
def pink_and_blue_curlers : ℕ := pink_curlers + blue_curlers

-- Define the number of green curlers
def green_curlers : ℕ := total_curlers - pink_and_blue_curlers

-- Theorem stating the number of green curlers is 4
theorem large_green_curlers : green_curlers = 4 := by
  -- Proof would go here
  sorry

end large_green_curlers_l267_267180


namespace find_approximation_of_x_l267_267957

-- Define the given condition and solve for x
theorem find_approximation_of_x :
  ∃ x : ℝ, (69.28 * 0.004) / x = 9.237333333333334 → x ≈ 0.03 :=
by
  sorry

end find_approximation_of_x_l267_267957


namespace find_pens_given_to_sharon_l267_267067

/-
This definition introduces the initial number of pens you start with.
-/
def initial_pens : ℕ := 5

/-
This definition introduces the number of additional pens given by Mike.
-/
def pens_from_mike : ℕ := 20

/-
This definition introduces the final number of pens after Cindy doubles and pens are given to Sharon.
-/
def final_pens : ℕ := 40

/-
This proposition formulates the problem of finding out the number of pens given to Sharon.
-/
def pens_given_to_sharon (initial_pens : ℕ) (pens_from_mike : ℕ) (final_pens : ℕ) : ℕ :=
  let doubled_pens := (initial_pens + pens_from_mike) * 2 in
  doubled_pens - final_pens

/-
We state the theorem that expresses that the number of pens given to Sharon is 10.
-/
theorem find_pens_given_to_sharon : pens_given_to_sharon initial_pens pens_from_mike final_pens = 10 :=
sorry

end find_pens_given_to_sharon_l267_267067


namespace cost_of_natural_seedless_raisins_l267_267741

theorem cost_of_natural_seedless_raisins
  (cost_golden: ℝ) (n_golden: ℕ) (n_natural: ℕ) (cost_mixture: ℝ) (cost_per_natural: ℝ) :
  cost_golden = 2.55 ∧ n_golden = 20 ∧ n_natural = 20 ∧ cost_mixture = 3
  → cost_per_natural = 3.45 :=
by
  sorry

end cost_of_natural_seedless_raisins_l267_267741


namespace number_of_values_f3_sum_of_values_f3_product_of_n_and_s_l267_267813

def S := { x : ℝ // x ≠ 0 }

def f (x : S) : S := sorry

lemma functional_equation (x y : S) (h : (x.val + y.val) ≠ 0) :
  (f x).val + (f y).val = (f ⟨(x.val * y.val) / (x.val + y.val) * (f ⟨x.val + y.val, sorry⟩).val, sorry⟩).val := sorry

-- Prove that the number of possible values of f(3) is 1

theorem number_of_values_f3 : ∃ n : ℕ, n = 1 := sorry

-- Prove that the sum of all possible values of f(3) is 1/3

theorem sum_of_values_f3 : ∃ s : ℚ, s = 1/3 := sorry

-- Prove that n * s = 1/3

theorem product_of_n_and_s (n : ℕ) (s : ℚ) (hn : n = 1) (hs : s = 1/3) : n * s = 1/3 := by
  rw [hn, hs]
  norm_num

end number_of_values_f3_sum_of_values_f3_product_of_n_and_s_l267_267813


namespace algebraic_expression_transformation_l267_267753

theorem algebraic_expression_transformation (a b : ℝ) :
  (∀ x : ℝ, x^2 + 4 * x + 3 = (x - 1)^2 + a * (x - 1) + b) → (a + b = 14) :=
by
  intros h
  sorry

end algebraic_expression_transformation_l267_267753


namespace minimum_distance_proof_l267_267720

variables (P Q α β : Type) [metric_space α] [metric_space β]

noncomputable def dihedral_angle (α : Type) (l : Type) (β : Type) : ℝ := 60
noncomputable def distance_point_to_plane (P : Type) (β : Type) : ℝ := sqrt 3
noncomputable def distance_point_to_other_plane (Q : Type) (α : Type) : ℝ := 2 * sqrt 3
noncomputable def minimum_distance_between_points (P Q : Type) : ℝ := 2 * sqrt 3

theorem minimum_distance_proof (P Q α β : Type) [metric_space α] [metric_space β]
  (h1 : dihedral_angle α α β = 60)
  (h2 : distance_point_to_plane P β = sqrt 3)
  (h3 : distance_point_to_other_plane Q α = 2 * sqrt 3) :
  minimum_distance_between_points P Q = 2 * sqrt 3 :=
sorry

end minimum_distance_proof_l267_267720


namespace angle_ABM_l267_267789

-- Define the conditions and proven statement in Lean.
theorem angle_ABM (α : ℝ) (A B C D M : ℝ) (square : ∀ (A B C D : ℝ), (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ D) ∧ (D ≠ A)) 
  (inside_square : ∀ (M : ℝ), (M ≠ A) ∧ (M ≠ B) ∧ (M ≠ C) ∧ (M ≠ D)) 
  (angles : ∠ M A C = α ∧ ∠ M C D = α) : ∠ A B M = 90 - 2 * α := 
by
  sorry

end angle_ABM_l267_267789


namespace regular_polygon_sides_l267_267606

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, interior_angle n i = 144) : n = 10 :=
sorry

end regular_polygon_sides_l267_267606


namespace find_three_fifths_of_neg_twelve_sevenths_l267_267688

def a : ℚ := -12 / 7
def b : ℚ := 3 / 5
def c : ℚ := -36 / 35

theorem find_three_fifths_of_neg_twelve_sevenths : b * a = c := by 
  -- sorry is a placeholder for the actual proof
  sorry

end find_three_fifths_of_neg_twelve_sevenths_l267_267688


namespace values_expression_locus_midpoint_l267_267167

variable {R r : ℝ} (hRr : R > r)

-- Condition Definitions
variables (c1 c2 : ℂ)
variables (P : ℂ) (hP : abs P = r)
variables (B : ℂ) (hB : abs B = R)
variables (C : ℂ) (hC : abs C = R)
variables (A : ℂ) (hA : abs A = r)
variables (BP_PC : ((B - P) * (P - A)).im = 0)  -- BP ⊥ PA

-- Question 1 Statement
theorem values_expression (hBP_intersects_C : C = (1 : ℂ)):
  abs (C - B)^2 + abs (A - C)^2 + abs (A - B)^2 = 6 * R^2 + 2 * r^2 :=
sorry

-- Question 2 Statement
theorem locus_midpoint :
  let Q := (A + B) / 2 in
  abs (Q - (P / 2)) = R / 2 :=
sorry

end values_expression_locus_midpoint_l267_267167


namespace Brandon_can_still_apply_l267_267144

-- Definitions based on the given conditions
def total_businesses : ℕ := 72
def fired_businesses : ℕ := total_businesses / 2
def quit_businesses : ℕ := total_businesses / 3
def businesses_restricted : ℕ := fired_businesses + quit_businesses

-- The final proof statement
theorem Brandon_can_still_apply : total_businesses - businesses_restricted = 12 :=
by
  -- Note: Proof is omitted; replace sorry with detailed proof in practice.
  sorry

end Brandon_can_still_apply_l267_267144


namespace range_of_a_l267_267249

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x > 1 then x else a * x^2 + 2 * x

theorem range_of_a (R : Set ℝ) :
  (∀ x : ℝ, f x a ∈ R) → (a ∈ Set.Icc (-1 : ℝ) 0) :=
sorry

end range_of_a_l267_267249


namespace daughter_l267_267386

variable (D F : ℕ)
hypothesis (h1 : F = 3 * D)
hypothesis (h2 : F + 12 = 2 * (D + 12))

theorem daughter's_age : D = 12 := by
  sorry

end daughter_l267_267386


namespace no_real_roots_sqrt_eq_l267_267174

theorem no_real_roots_sqrt_eq (x : ℝ) : 
  (sqrt (x + 5) - sqrt (x - 2) + 2 = 0) → 
  (¬ ∃ (x : ℝ), sqrt (x + 5) - sqrt (x - 2) + 2 = 0) :=
by
  intro h
  sorry

end no_real_roots_sqrt_eq_l267_267174


namespace factorization_of_x10_minus_1024_l267_267156

theorem factorization_of_x10_minus_1024 (x : ℝ) :
  x^10 - 1024 = (x^5 + 32) * (x - 2) * (x^4 + 2 * x^3 + 4 * x^2 + 8 * x + 16) :=
by sorry

end factorization_of_x10_minus_1024_l267_267156


namespace max_行_value_l267_267225

noncomputable def max_行 : ℕ := 8

theorem max_行_value (a b c d e : ℕ) 
  (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d) (h4 : a ≠ e)
  (h5 : b ≠ c) (h6 : b ≠ d) (h7 : b ≠ e)
  (h8 : c ≠ d) (h9 : c ≠ e)
  (h10 : d ≠ e)
  (hx : a ≠ 0) (hy : b ≠ 0) (hz : c ≠ 0) (ht : d ≠ 0) (hu : e ≠ 0)
  (hxi : a + b + c + d + e = 21)
  (hsum : (a + b + c + d + e) + (a + b + c + d + e)
         = 84) :
  e ≤ max_行 :=
sorry

end max_行_value_l267_267225


namespace least_element_in_T_l267_267894

variable (S : Finset ℕ)
variable (T : Finset ℕ)
variable (hS : S = Finset.range 16 \ {0})
variable (hT : T.card = 5)
variable (hTsubS : T ⊆ S)
variable (hCond : ∀ x y, x ∈ T → y ∈ T → x < y → ¬ (y % x = 0))

theorem least_element_in_T (S T : Finset ℕ) (hT : T.card = 5) (hTsubS : T ⊆ S)
  (hCond : ∀ x y, x ∈ T → y ∈ T → x < y → ¬ (y % x = 0)) : 
  ∃ m ∈ T, m = 5 :=
by
  sorry

end least_element_in_T_l267_267894


namespace barrels_of_pitch_needed_on_third_day_l267_267115

def road_length : ℕ := 16
def truckloads_per_mile : ℕ := 3
def bags_of_gravel_per_truckload : ℕ := 2
def gravel_bags_to_pitch_ratio : ℕ := 5
def paved_distance_day1 : ℕ := 4
def paved_distance_day2 : ℕ := 2 * paved_distance_day1 - 1

theorem barrels_of_pitch_needed_on_third_day :
  let paved_distance_first_two_days := paved_distance_day1 + paved_distance_day2 in
  let remaining_distance := road_length - paved_distance_first_two_days in
  let truckloads_needed := remaining_distance * truckloads_per_mile in
  let barrels_per_truckload := (bags_of_gravel_per_truckload : ℚ) / gravel_bags_to_pitch_ratio in
  let total_barrels_needed := truckloads_needed * barrels_per_truckload in
  total_barrels_needed = 6 := 
by
  sorry

end barrels_of_pitch_needed_on_third_day_l267_267115


namespace third_number_from_left_l267_267776

theorem third_number_from_left (n : ℕ) (h : n ≥ 3) : 
  let first_number := 1 + (n - 1) * n
  in first_number + 4 = n^2 - n + 5 := 
by
  sorry

end third_number_from_left_l267_267776


namespace regular_polygon_sides_l267_267614

theorem regular_polygon_sides (n : ℕ) (h : n ≥ 3) (interior_angle : ℝ) : 
  interior_angle = 144 → n = 10 :=
by
  intro h1
  sorry

end regular_polygon_sides_l267_267614


namespace company_members_and_days_l267_267579

theorem company_members_and_days {t n : ℕ} (h : t = 6) :
    n = (t * (t - 1)) / 2 → n = 15 :=
by
  intro hn
  rw [h] at hn
  simp at hn
  exact hn

end company_members_and_days_l267_267579


namespace average_visitors_proof_statement_l267_267584
-- Import requisite libraries

-- Define the conditions
def average_visitors_on_sunday : ℕ := 510
def average_visitors_per_day : ℕ := 285
def days_in_month : ℕ := 30
def sundays_in_month : ℕ := 5

-- Define the target variable for average visitors on other days
def average_visitors_other_days : ℕ := 240

-- Construct the Lean statement to prove the relation
theorem average_visitors_proof_statement :
  let total_visitors_in_month := average_visitors_per_day * days_in_month in
  let total_visitors_on_sundays := sundays_in_month * average_visitors_on_sunday in
  (total_visitors_on_sundays + (days_in_month - sundays_in_month) * average_visitors_other_days) = total_visitors_in_month :=
by
  sorry

end average_visitors_proof_statement_l267_267584


namespace barrels_of_pitch_needed_on_third_day_l267_267114

def road_length : ℕ := 16
def truckloads_per_mile : ℕ := 3
def bags_of_gravel_per_truckload : ℕ := 2
def gravel_bags_to_pitch_ratio : ℕ := 5
def paved_distance_day1 : ℕ := 4
def paved_distance_day2 : ℕ := 2 * paved_distance_day1 - 1

theorem barrels_of_pitch_needed_on_third_day :
  let paved_distance_first_two_days := paved_distance_day1 + paved_distance_day2 in
  let remaining_distance := road_length - paved_distance_first_two_days in
  let truckloads_needed := remaining_distance * truckloads_per_mile in
  let barrels_per_truckload := (bags_of_gravel_per_truckload : ℚ) / gravel_bags_to_pitch_ratio in
  let total_barrels_needed := truckloads_needed * barrels_per_truckload in
  total_barrels_needed = 6 := 
by
  sorry

end barrels_of_pitch_needed_on_third_day_l267_267114


namespace find_vanilla_cookies_l267_267068

variable (V : ℕ)

def num_vanilla_cookies_sold (choc_cookies: ℕ) (vanilla_cookies: ℕ) (total_revenue: ℕ) : Prop :=
  choc_cookies * 1 + vanilla_cookies * 2 = total_revenue

theorem find_vanilla_cookies (h : num_vanilla_cookies_sold 220 V 360) : V = 70 :=
by
  sorry

end find_vanilla_cookies_l267_267068


namespace lace_maker_combined_time_l267_267054

  theorem lace_maker_combined_time (t1 t2 : ℕ) (h1 : t1 = 8) (h2 : t2 = 13) : 
    let combined_time := 104 / 21 in 
    combined_time = real.of_rat (104 / 21 : ℚ) := 
  by
    sorry
  
end lace_maker_combined_time_l267_267054


namespace jared_popcorn_l267_267040

-- Define the given conditions
def pieces_per_serving := 30
def number_of_friends := 3
def pieces_per_friend := 60
def servings_ordered := 9

-- Define the total pieces of popcorn
def total_pieces := servings_ordered * pieces_per_serving

-- Define the total pieces of popcorn eaten by Jared's friends
def friends_total_pieces := number_of_friends * pieces_per_friend

-- State the theorem
theorem jared_popcorn : total_pieces - friends_total_pieces = 90 :=
by 
  -- The detailed proof would go here.
  sorry

end jared_popcorn_l267_267040


namespace no_such_convex_polyhedron_exists_l267_267357

-- Definitions of convex polyhedron and the properties related to its faces and vertices.
structure ConvexPolyhedron where
  V : ℕ  -- Number of vertices
  E : ℕ  -- Number of edges
  F : ℕ  -- Number of faces
  -- Additional properties and constraints can be added if necessary

-- Definition that captures the condition where each face has more than 5 sides.
def each_face_has_more_than_five_sides (P : ConvexPolyhedron) : Prop :=
  ∀ f, f > 5 -- Simplified assumption

-- Definition that captures the condition where more than five edges meet at each vertex.
def more_than_five_edges_meet_each_vertex (P : ConvexPolyhedron) : Prop :=
  ∀ v, v > 5 -- Simplified assumption

-- The statement to be proven
theorem no_such_convex_polyhedron_exists :
  ¬ ∃ (P : ConvexPolyhedron), (each_face_has_more_than_five_sides P) ∨ (more_than_five_edges_meet_each_vertex P) := by
  -- Proof of this theorem is omitted with "sorry"
  sorry

end no_such_convex_polyhedron_exists_l267_267357


namespace transformed_function_l267_267251

theorem transformed_function :
  ∀ x : ℝ, (let f := (λ x, Real.cos (2 * x + 4 * Real.pi / 5)) in
    let t1 := (λ x, x - Real.pi / 2) in
    let t2 := (λ x, 2 * x) in
    let t3 := (λ y, 4 * y) in
    t3 (f (t2 (t1 x))) = 4 * Real.cos (4 * x - Real.pi / 5)) :=
by
  intro x
  have h1 : 2 * (x - Real.pi / 2) = 2 * x - Real.pi := sorry
  have h2 : Real.cos (2 * x - Real.pi + 4 * Real.pi / 5) = Real.cos (2 * x - Real.pi + 4 * Real.pi / 5) := sorry
  rw [←h1] at h2
  have h3 : Real.cos (2 * x - Real.pi + 4 * Real.pi / 5) = Real.cos (2 * x - Real.pi + 4 * Real.pi / 5) := sorry
  rw [h2, h3]
  have h4 : Real.cos (4 * x - Real.pi / 5) = Real.cos (4 * x - Real.pi / 5) := sorry
  rw [h4]
  rfl

end transformed_function_l267_267251


namespace sum_of_eight_l267_267878

variable (a₁ : ℕ) (q : ℕ)
variable (S : ℕ → ℕ) -- Assume S is a function from natural numbers to natural numbers representing S_n

-- Condition 1: Sum of first 4 terms equals -5
axiom h1 : S 4 = -5

-- Condition 2: Sum of the first 6 terms is 21 times the sum of the first 2 terms
axiom h2 : S 6 = 21 * S 2

-- The formula for the sum of the first n terms of a geometric sequence
def sum_of_first_n_terms (n : ℕ) := a₁ * (1 - q^n) / (1 - q)

-- Statement to be proved
theorem sum_of_eight : S 8 = -85 := sorry

end sum_of_eight_l267_267878


namespace integral_sin4_cos4_l267_267682

theorem integral_sin4_cos4 (C : ℝ) :
  ∫ (x : ℝ) in _, (sin x) ^ 4 * (cos x) ^ 4 =
    (1 / 128 : ℝ) * (3 * x - 4 * sin (4 * x) + (1 / 8) * sin (8 * x)) + C :=
  sorry

end integral_sin4_cos4_l267_267682


namespace percent_nonunion_women_l267_267303

variable (E : ℝ) -- Total number of employees

-- Definitions derived from the problem conditions
def menPercent : ℝ := 0.46
def unionPercent : ℝ := 0.60
def nonUnionPercent : ℝ := 1 - unionPercent
def nonUnionWomenPercent : ℝ := 0.90

theorem percent_nonunion_women :
  nonUnionWomenPercent = 0.90 :=
by
  sorry

end percent_nonunion_women_l267_267303


namespace cube_surface_area_l267_267073

theorem cube_surface_area (V : ℝ) (s : ℝ) (A : ℝ) :
  V = 729 ∧ V = s^3 ∧ A = 6 * s^2 → A = 486 := by
  sorry

end cube_surface_area_l267_267073


namespace geometric_sequence_sum_l267_267887

variable (a1 q : ℝ) -- Define the first term and common ratio as real numbers

-- Define the sum of the first n terms of a geometric sequence
def S (n : ℕ) : ℝ := a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum :
  (S 4 a1 q = -5) → (S 6 a1 q = 21 * S 2 a1 q) → (S 8 a1 q = -85) :=
by
  intro h1 h2
  sorry

end geometric_sequence_sum_l267_267887


namespace length_of_P1P2_segment_l267_267582

theorem length_of_P1P2_segment (x : ℝ) (h₀ : 0 < x ∧ x < π / 2) (h₁ : 6 * Real.cos x = 9 * Real.tan x) :
  Real.sin x = 1 / 2 :=
by
  sorry

end length_of_P1P2_segment_l267_267582


namespace geometric_sequence_S8_l267_267866

theorem geometric_sequence_S8 
  (a1 q : ℝ) (S : ℕ → ℝ)
  (h1 : S 4 = a1 * (1 - q^4) / (1 - q) = -5)
  (h2 : S 6 = 21 * S 2)
  (geom_sum : ∀ n, S n = a1 * (1 - q^n) / (1 - q))
  (h3 : q ≠ 1) : S 8 = -85 := 
begin
  sorry
end

end geometric_sequence_S8_l267_267866


namespace semicircle_to_quarter_circle_area_ratio_l267_267788

theorem semicircle_to_quarter_circle_area_ratio (R : ℝ) (hR : 0 < R) : 
  let area_quarter := (1 / 4) * Real.pi * R^2,
      area_semi := (1 / 2) * Real.pi * (R / 2)^2
  in (area_semi / area_quarter) = (1 / 2) :=
by 
  sorry

end semicircle_to_quarter_circle_area_ratio_l267_267788


namespace point_R_locus_l267_267245

open Set

-- Defining the ellipse
def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  let ⟨x, y⟩ := P in (x^2 / 4) + (y^2 / 2) = 1

-- Fixed points A, B, C
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)
def C : ℝ × ℝ := (3, 0)

-- Intersection points P and Q of a line through C and the ellipse
def is_on_line_through_C (P Q : ℝ × ℝ) : Prop :=
  let ⟨x₁, y₁⟩ := P in
  let ⟨x₂, y₂⟩ := Q in
  y₁ ≠ y₂ ∧ ∃ m b : ℝ, ∀ x : ℝ, (x, m * (x - 3) + b) = P ∨ (x, m * (x - 3) + b) = Q

-- Line AP and BQ intersection point R
def intersection_point (A B P Q : ℝ × ℝ) : ℝ × ℝ :=
  let ⟨xa, ya⟩ := A in
  let ⟨xb, yb⟩ := B in
  let ⟨xp, yp⟩ := P in
  let ⟨xq, yq⟩ := Q in
  let m1 := (yp - ya) / (xp - xa) in
  let m2 := (yq - yb) / (xq - xb) in
  let x := (yb - ya - m1 * xa + m2 * xb) / (m2 - m1) in
  let y := m1 * (x - xa) + ya in
  (x, y)

-- Mathematical statement to be proven
theorem point_R_locus :
  ∀ P Q R : ℝ × ℝ, is_on_ellipse P →
  is_on_ellipse Q →
  is_on_line_through_C P Q →
  intersection_point A B P Q = R →
  (45 * (R.1)^2 - 108 * (R.2)^2 = 20 ∧ R.1 ∈ Icc (2/3 : ℝ) (4/3 : ℝ))
:= by
  intros P Q R hP hQ hLine hIntersection
  sorry

end point_R_locus_l267_267245


namespace perimeter_of_garden_l267_267955

-- Define the area of the square garden
def area_square_garden : ℕ := 49

-- Define the relationship between q and p
def q_equals_p_plus_21 (q p : ℕ) : Prop := q = p + 21

-- Define the length of the side of the square garden
def side_length (area : ℕ) : ℕ := Nat.sqrt area

-- Define the perimeter of the square garden
def perimeter (side_length : ℕ) : ℕ := 4 * side_length

-- Define the perimeter of the square garden as a specific perimeter
def specific_perimeter (side_length : ℕ) : ℕ := perimeter side_length

-- Statement of the theorem
theorem perimeter_of_garden (q p : ℕ) (h1 : q = 49) (h2 : q_equals_p_plus_21 q p) : 
  specific_perimeter (side_length 49) = 28 := by
  sorry

end perimeter_of_garden_l267_267955


namespace citric_acid_molecular_weight_l267_267640

noncomputable def molecularWeightOfCitricAcid : ℝ :=
  let weight_C := 12.01
  let weight_H := 1.008
  let weight_O := 16.00
  let num_C := 6
  let num_H := 8
  let num_O := 7
  (num_C * weight_C) + (num_H * weight_H) + (num_O * weight_O)

theorem citric_acid_molecular_weight :
  molecularWeightOfCitricAcid = 192.124 :=
by
  -- the step-by-step proof will go here
  sorry

end citric_acid_molecular_weight_l267_267640


namespace geometric_sequence_sum_eight_l267_267824

noncomputable def sum_geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ :=
  a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_eight {a1 q : ℝ} (hq : q ≠ 1) 
  (h4 : sum_geometric_sequence a1 q 4 = -5) 
  (h6 : sum_geometric_sequence a1 q 6 = 21 * sum_geometric_sequence a1 q 2) : 
  sum_geometric_sequence a1 q 8 = -85 :=
sorry

end geometric_sequence_sum_eight_l267_267824


namespace mechanic_worked_hours_l267_267346

-- Definitions based on conditions
def total_amount_spent : ℕ := 220
def cost_per_part : ℕ := 20
def number_of_parts : ℕ := 2
def labor_cost_per_minute : ℝ := 0.5

-- Total cost of parts
def total_parts_cost : ℕ := number_of_parts * cost_per_part

-- Total labor cost
def total_labor_cost : ℕ := total_amount_spent - total_parts_cost

-- Total labor time in minutes
def labor_time_minutes : ℝ := total_labor_cost / labor_cost_per_minute

-- Convert labor time to hours
def labor_time_hours : ℝ := labor_time_minutes / 60

theorem mechanic_worked_hours : labor_time_hours = 6 :=
by
  -- Proof will be inserted here
  sorry

end mechanic_worked_hours_l267_267346


namespace joan_missed_games_l267_267800

-- Define the number of total games and games attended as constants
def total_games : ℕ := 864
def games_attended : ℕ := 395

-- The theorem statement: the number of missed games is equal to 469
theorem joan_missed_games : total_games - games_attended = 469 :=
by
  -- Proof goes here
  sorry

end joan_missed_games_l267_267800


namespace correct_option_is_A_l267_267556

variable (a b : ℝ)

theorem correct_option_is_A :
  ((-a^2)^3 = -a^6) ∧ ((-b)^5 / b^3 = -b^2) ∧ (-12 / 3 * (-2 / 3) = 8 / 3) ∧ (real.cbrt 8 + real.sqrt 6 * (-real.sqrt (3 / 2)) = -1) →
  (∃ x, x = A :=
  (-a^2)^3 = -a^6 ∧ ∀ x, length[A] = 42)

end correct_option_is_A_l267_267556


namespace polygon_perimeter_l267_267596

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

def perimeter (points : list (ℝ × ℝ)) : ℝ :=
  points.zip (points.tail ++ [points.head]).map (λ ⟨p1, p2⟩, distance p1 p2).sum

theorem polygon_perimeter :
  let points := [(0, 0), (1, 1), (3, 1), (3, 2), (2, 2), (2, 0), (0, 0)]
  let a_b_c := 8 + real.sqrt(2) + 0 * real.sqrt(3)
  in a_b_c = 8 + 1 + 0 := 9 :=
by
  sorry

end polygon_perimeter_l267_267596


namespace number_of_elements_in_M_l267_267395

theorem number_of_elements_in_M :
  (∃! (M : Finset ℕ), M = {m | ∃ (n : ℕ), n > 0 ∧ m = 2*n - 1 ∧ m < 60 } ∧ M.card = 30) :=
sorry

end number_of_elements_in_M_l267_267395


namespace eval_piecewise_function_l267_267247

def piecewise_function_f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 2*x + 2 else -x^2

theorem eval_piecewise_function :
  piecewise_function_f (piecewise_function_f 1) = 1 :=
by
  sorry

end eval_piecewise_function_l267_267247


namespace solve_for_x_l267_267006

theorem solve_for_x (x : ℚ) : (2/3 : ℚ) - 1/4 = 1/x → x = 12/5 := 
by
  sorry

end solve_for_x_l267_267006


namespace max_divisor_of_five_consecutive_integers_l267_267510

theorem max_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, 60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intros n
  sorry

end max_divisor_of_five_consecutive_integers_l267_267510


namespace penguins_remaining_to_get_fish_l267_267989

def total_penguins : Nat := 36
def fed_penguins : Nat := 19

theorem penguins_remaining_to_get_fish : (total_penguins - fed_penguins = 17) :=
by
  sorry

end penguins_remaining_to_get_fish_l267_267989


namespace large_circle_radius_l267_267206

noncomputable def radius_of_large_circle : ℝ :=
  let r_small := 1
  let side_length := 2 * r_small
  let diagonal_length := Real.sqrt (side_length ^ 2 + side_length ^ 2)
  let radius_large := (diagonal_length / 2) + r_small
  radius_large + r_small

theorem large_circle_radius :
  radius_of_large_circle = Real.sqrt 2 + 2 :=
by
  sorry

end large_circle_radius_l267_267206


namespace three_digit_numbers_sum_26_l267_267694

theorem three_digit_numbers_sum_26 : 
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ (∃ (a b c : ℕ),
    n = a * 100 + b * 10 + c ∧
    a + b + c = 26 ∧ 
    1 ≤ a ∧ a ≤ 9 ∧
    1 ≤ b ∧ b ≤ 9 ∧
    1 ≤ c ∧ c ≤ 9)}.finite.card = 2 := sorry

end three_digit_numbers_sum_26_l267_267694


namespace proof_problem_l267_267744

noncomputable def x := real

theorem proof_problem (x : ℝ) (h : x + sqrt (x^2 - 4) + 1 / (x - sqrt (x^2 - 4)) = 24) : 
  x^2 + sqrt (x^4 - 4) + 1 / (x^2 + sqrt (x^4 - 4)) = 685 / 9 :=
by
  sorry

end proof_problem_l267_267744


namespace cone_base_radius_l267_267656

noncomputable def sector_radius : ℝ := 9
noncomputable def central_angle_deg : ℝ := 240

theorem cone_base_radius :
  let arc_length := (central_angle_deg * Real.pi * sector_radius) / 180
  let base_circumference := arc_length
  let base_radius := base_circumference / (2 * Real.pi)
  base_radius = 6 :=
by
  sorry

end cone_base_radius_l267_267656


namespace hats_distribution_l267_267919

theorem hats_distribution (Paityn_red Paityn_blue : ℕ) (Zola_red Zola_blue : ℕ)
  (h1 : Paityn_red = 20)
  (h2 : Paityn_blue = 24)
  (h3 : Zola_red = (4 / 5) * Paityn_red)
  (h4 : Zola_blue = 2 * Paityn_blue) :
  let total_hats := (Paityn_red + Zola_red) + (Paityn_blue + Zola_blue)
  in total_hats / 2 = 54 := 
by
  sorry

end hats_distribution_l267_267919


namespace necessary_and_sufficient_condition_l267_267716

noncomputable def f (x : ℝ) : ℝ := sorry

-- Conditions
axiom even_function : ∀ x : ℝ, f x = f (-x)
axiom periodic_function : ∀ x : ℝ, f (x + 2) = f x
axiom increasing_on_01 : ∀ x y : ℝ, 0 ≤ x → x < y → y ≤ 1 → f x < f y

-- Required to prove
theorem necessary_and_sufficient_condition :
  (∀ x y : ℝ, 3 ≤ x → x < y → y ≤ 4 → f x > f y) ↔ (increasing_on_01) :=
by
  sorry

end necessary_and_sufficient_condition_l267_267716


namespace students_neither_correct_l267_267914

-- Define the total number of students and the numbers for chemistry, biology, and both
def total_students := 75
def chemistry_students := 42
def biology_students := 33
def both_subject_students := 18

-- Define a function to calculate the number of students taking neither chemistry nor biology
def students_neither : ℕ :=
  total_students - ((chemistry_students - both_subject_students) 
                    + (biology_students - both_subject_students) 
                    + both_subject_students)

-- Theorem stating that the number of students taking neither chemistry nor biology is as expected
theorem students_neither_correct : students_neither = 18 :=
  sorry

end students_neither_correct_l267_267914


namespace largest_divisor_of_5_consecutive_integers_l267_267533

theorem largest_divisor_of_5_consecutive_integers :
  ∀ (n : ℤ), ∃ d, d = 120 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intro n
  use 120
  split
  exact rfl
  sorry

end largest_divisor_of_5_consecutive_integers_l267_267533


namespace largest_divisor_of_5_consecutive_integers_l267_267475

theorem largest_divisor_of_5_consecutive_integers :
  ∀ (a b c d e : ℤ), 
    a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e →
    (∃ k : ℤ, k ∣ (a * b * c * d * e) ∧ k = 60) :=
by 
  intro a b c d e h
  sorry

end largest_divisor_of_5_consecutive_integers_l267_267475


namespace volume_of_prism_l267_267956

theorem volume_of_prism (a b c : ℝ)
  (h_ab : a * b = 36)
  (h_ac : a * c = 54)
  (h_bc : b * c = 72) :
  a * b * c = 648 :=
by
  sorry

end volume_of_prism_l267_267956


namespace solve_for_x_l267_267944

noncomputable def n : ℝ := Real.sqrt (7^2 + 24^2)
noncomputable def d : ℝ := Real.sqrt (49 + 16)
noncomputable def x : ℝ := n / d

theorem solve_for_x : x = 5 * Real.sqrt 65 / 13 := by
  sorry

end solve_for_x_l267_267944


namespace inequality_8xyz_leq_1_equality_cases_8xyz_eq_1_l267_267713

theorem inequality_8xyz_leq_1 (x y z : ℝ) (h : x^2 + y^2 + z^2 + 2 * x * y * z = 1) :
  8 * x * y * z ≤ 1 :=
sorry

theorem equality_cases_8xyz_eq_1 (x y z : ℝ) (h1 : x^2 + y^2 + z^2 + 2 * x * y * z = 1) :
  8 * x * y * z = 1 ↔ 
  (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) ∨ 
  (x = -1/2 ∧ y = -1/2 ∧ z = 1/2) ∨ 
  (x = -1/2 ∧ y = 1/2 ∧ z = -1/2) ∨ 
  (x = 1/2 ∧ y = -1/2 ∧ z = -1/2) :=
sorry

end inequality_8xyz_leq_1_equality_cases_8xyz_eq_1_l267_267713


namespace function_property_l267_267223

noncomputable def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem function_property
  (f : ℝ → ℝ)
  (h_odd : odd_function f)
  (h_property : ∀ (x1 x2 : ℝ), 0 < x1 → 0 < x2 → x1 ≠ x2 → (x1 - x2) * (f x1 - f x2) > 0)
  : f (-4) > f (-6) :=
sorry

end function_property_l267_267223


namespace five_consecutive_product_div_24_l267_267489

theorem five_consecutive_product_div_24 (n : ℤ) : 
  24 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) := 
sorry

end five_consecutive_product_div_24_l267_267489


namespace geometric_sequence_sum_eight_l267_267826

noncomputable def sum_geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ :=
  a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_eight {a1 q : ℝ} (hq : q ≠ 1) 
  (h4 : sum_geometric_sequence a1 q 4 = -5) 
  (h6 : sum_geometric_sequence a1 q 6 = 21 * sum_geometric_sequence a1 q 2) : 
  sum_geometric_sequence a1 q 8 = -85 :=
sorry

end geometric_sequence_sum_eight_l267_267826


namespace faye_age_l267_267671

variable (C D E F : ℕ)

-- Conditions
axiom h1 : D = 16
axiom h2 : D = E - 4
axiom h3 : E = C + 5
axiom h4 : F = C + 2

-- Goal: Prove that F = 17
theorem faye_age : F = 17 :=
by
  sorry

end faye_age_l267_267671


namespace sum_first_100_terms_l267_267963

-- Define the sequence a_n
def a (n : ℕ) : ℝ := (-1 : ℝ) ^ n * n * Real.sin (n * Real.pi / 2) + 1

-- Define the sum of the first n terms of the sequence
def S (n : ℕ) : ℝ := ∑ i in Finset.range n, a i

-- Problem statement: Prove that the sum of the first 100 terms equals 150
theorem sum_first_100_terms : S 100 = 150 := by
  sorry

end sum_first_100_terms_l267_267963


namespace solution_set_l267_267193

noncomputable def inequality (x : ℝ) : Prop :=
  real.sqrt ((1 / (2 - x) + 1) ^ 2) ≥ 2

theorem solution_set (x : ℝ) :
  inequality x ↔ ((1 ≤ x ∧ x < 2) ∨ (2 < x ∧ x ≤ (7 / 3))) := sorry

end solution_set_l267_267193


namespace odd_squares_diff_divisible_by_8_l267_267960

-- Definition of the problem
def odd_difference_div_by_eight (a b : ℤ) (h : a > b) : Prop :=
  (2 * a + 1) ^ 2 - (2 * b + 1) ^ 2 ≡ 0 [MOD 8]

-- The statement to be proved
theorem odd_squares_diff_divisible_by_8 (a b : ℤ) (h : a > b) : odd_difference_div_by_eight a b h :=
begin
  sorry
end

end odd_squares_diff_divisible_by_8_l267_267960


namespace largest_divisor_of_five_consecutive_integers_l267_267461

open Nat

theorem largest_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, k ∈ {n, n+1, n+2, n+3, n+4} ∧
    ∀ m ∈ {2, 3, 4, 5}, m ∣ k → 60 ∣ (n * (n+1) * (n+2) * (n+3) * (n+4)) := 
sorry

end largest_divisor_of_five_consecutive_integers_l267_267461


namespace equilateral_triangle_perimeter_l267_267546

theorem equilateral_triangle_perimeter (a : ℕ) (a = 5) : 3 * a = 15 := by
  sorry

end equilateral_triangle_perimeter_l267_267546


namespace no_nonzero_integers_satisfy_conditions_l267_267364

theorem no_nonzero_integers_satisfy_conditions :
  ¬ ∃ a b x y : ℤ, (a ≠ 0 ∧ b ≠ 0 ∧ x ≠ 0 ∧ y ≠ 0) ∧ (a * x - b * y = 16) ∧ (a * y + b * x = 1) :=
by
  sorry

end no_nonzero_integers_satisfy_conditions_l267_267364


namespace inverse_variation_example_l267_267372

theorem inverse_variation_example
  (k : ℝ)
  (h1 : ∀ (c d : ℝ), (c^2) * (d^4) = k)
  (h2 : ∃ (c : ℝ), c = 8 ∧ (∀ (d : ℝ), d = 2 → (c^2) * (d^4) = k)) : 
  (∀ (d : ℝ), d = 4 → (∃ (c : ℝ), (c^2) = 4)) := 
by 
  sorry

end inverse_variation_example_l267_267372


namespace solve_equation_l267_267985

theorem solve_equation :
  ∀ x : ℝ, (4^x - 6 * 2^x + 8 = 0) ↔ (x = 1 ∨ x = 2) :=
by
  intro x
  split
  sorry

end solve_equation_l267_267985


namespace surface_area_proof_l267_267642

-- Definitions based on conditions
def radius : ℝ := 10
def height : ℝ := radius / 2

def curved_surface_area_hemisphere : ℝ := 2 * (π * radius^2)
def base_surface_area_hemisphere : ℝ := π * radius^2
def lateral_surface_area_cylinder : ℝ := 2 * π * radius * height

-- Total surface area computation
def total_surface_area : ℝ :=
  curved_surface_area_hemisphere + base_surface_area_hemisphere + lateral_surface_area_cylinder

-- Proof statement
theorem surface_area_proof : total_surface_area = 400 * π := by
  sorry

end surface_area_proof_l267_267642


namespace max_parts_three_planes_l267_267996

-- Define the condition for three non-overlapping planes
def three_non_overlapping_planes (P1 P2 P3 : Plane) : Prop :=
  ∀ x ∈ (P1 ∪ P2 ∪ P3), ¬ (x ∈ P1 ∩ P2 ∩ P3)

-- Define the maximum number of parts for three non-overlapping planes
def max_parts_divided_by_planes (n : ℕ) : Prop :=
  n = 8

-- The proof problem statement
theorem max_parts_three_planes (P1 P2 P3 : Plane) (h: three_non_overlapping_planes P1 P2 P3) :
  ∃ n, max_parts_divided_by_planes n :=
begin
  use 8,
  sorry
end

end max_parts_three_planes_l267_267996


namespace bc_over_a_sq_plus_ac_over_b_sq_plus_ab_over_c_sq_eq_3_l267_267085

variables {a b c : ℝ}
-- Given conditions from Vieta's formulas for the polynomial x^3 - 20x^2 + 22
axiom vieta1 : a + b + c = 20
axiom vieta2 : a * b + b * c + c * a = 0
axiom vieta3 : a * b * c = -22

theorem bc_over_a_sq_plus_ac_over_b_sq_plus_ab_over_c_sq_eq_3 (a b c : ℝ)
  (h1 : a + b + c = 20)
  (h2 : a * b + b * c + c * a = 0)
  (h3 : a * b * c = -22) :
  (b * c / a^2) + (a * c / b^2) + (a * b / c^2) = 3 := 
  sorry

end bc_over_a_sq_plus_ac_over_b_sq_plus_ab_over_c_sq_eq_3_l267_267085


namespace odd_function_periodic_value_l267_267332

noncomputable def f : ℝ → ℝ := sorry  -- Define f

theorem odd_function_periodic_value:
  (∀ x, f (-x) = - f x) →  -- f is odd
  (∀ x, f (x + 3) = f x) → -- f has period 3
  f 1 = 2014 →            -- given f(1) = 2014
  f 2013 + f 2014 + f 2015 = 0 := by
  intros h_odd h_period h_f1
  sorry

end odd_function_periodic_value_l267_267332


namespace number_of_elements_in_M_l267_267397

def positive_nats : Set ℕ := {n | n > 0}
def M : Set ℕ := {m | ∃ n ∈ positive_nats, m = 2 * n - 1 ∧ m < 60}

theorem number_of_elements_in_M : ∃ s : Finset ℕ, (∀ x, x ∈ s ↔ x ∈ M) ∧ s.card = 30 := 
by
  sorry

end number_of_elements_in_M_l267_267397


namespace shortest_distance_between_stations_l267_267425

/-- 
Given two vehicles A and B shuttling between two locations,
with Vehicle A stopping every 0.5 kilometers and Vehicle B stopping every 0.8 kilometers,
prove that the shortest distance between two stations where Vehicles A and B do not stop at the same place is 0.1 kilometers.
-/
theorem shortest_distance_between_stations :
  ∀ (dA dB : ℝ), (dA = 0.5) → (dB = 0.8) → ∃ δ : ℝ, (δ = 0.1) ∧ (∀ n m : ℕ, dA * n ≠ dB * m → abs ((dA * n) - (dB * m)) = δ) :=
by
  intros dA dB hA hB
  use 0.1
  sorry

end shortest_distance_between_stations_l267_267425


namespace probability_fail_then_succeed_l267_267594

theorem probability_fail_then_succeed
  (P_fail_first : ℚ := 9 / 10)
  (P_succeed_second : ℚ := 1 / 9) :
  P_fail_first * P_succeed_second = 1 / 10 :=
by
  sorry

end probability_fail_then_succeed_l267_267594


namespace max_divisor_of_five_consecutive_integers_l267_267507

theorem max_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, 60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intros n
  sorry

end max_divisor_of_five_consecutive_integers_l267_267507


namespace dr_mary_donation_l267_267377

theorem dr_mary_donation :
  let total_donations := 15000 in
  let first_home := 3500 in
  let second_home := 2750 in
  let third_home := 3870 in
  let fourth_home := 2475 in
  let already_donated := first_home + second_home + third_home + fourth_home in
  let remaining_amount := total_donations - already_donated in
  let amount_per_home := remaining_amount / 4 in
  amount_per_home = 601.25 :=
by
  sorry

end dr_mary_donation_l267_267377


namespace evaluate_nested_square_root_l267_267185

-- Define the condition
def pos_real_solution (x : ℝ) : Prop := x = Real.sqrt (18 + x)

-- State the theorem
theorem evaluate_nested_square_root :
  ∃ (x : ℝ), pos_real_solution x ∧ x = (1 + Real.sqrt 73) / 2 :=
sorry

end evaluate_nested_square_root_l267_267185


namespace janet_lives_lost_l267_267313

def janet_initial_lives : ℕ := 47
def janet_lives_after_gain : ℕ := 70
def lives_gained : ℕ := 46

theorem janet_lives_lost : ∃ L : ℕ, janet_initial_lives - L + lives_gained = janet_lives_after_gain ∧ L = 23 := by
  use 23
  split
  {
    -- Prove the equation 47 - 23 + 46 = 70
    norm_num
  }
  {
    -- Prove L = 23
    refl
  }
  sorry -- include this to skip the detailed steps

end janet_lives_lost_l267_267313


namespace log_eq_div_l267_267675

theorem log_eq_div (log2_50 : ℝ) (h1 : 8 = 2 ^ 3):
  log 8 50 = log2_50 / 3 := 
sorry

end log_eq_div_l267_267675


namespace number_of_elements_in_M_l267_267398

def positive_nats : Set ℕ := {n | n > 0}
def M : Set ℕ := {m | ∃ n ∈ positive_nats, m = 2 * n - 1 ∧ m < 60}

theorem number_of_elements_in_M : ∃ s : Finset ℕ, (∀ x, x ∈ s ↔ x ∈ M) ∧ s.card = 30 := 
by
  sorry

end number_of_elements_in_M_l267_267398


namespace convex_polygon_diagonal_intersections_l267_267238

theorem convex_polygon_diagonal_intersections (n : ℕ) (h1 : n > 3)
  (h_no3 : ∀ (polygon : Type) [convex_polygon polygon] (v : polygon → ℝ × ℝ), 
    ∀ p1 p2 p3 : polygon, ¬(diagonal_intersection v p1 p2 p3)) :
  number_of_diagonal_intersections n = (n * (n - 1) * (n - 2) * (n - 3)) / 24 := 
sorry

end convex_polygon_diagonal_intersections_l267_267238


namespace flippy_numbers_divisible_by_15_are_4_l267_267593

def is_flippy (n : ℕ) : Prop :=
  let digits := (n % 10, (n / 10) % 10, (n / 100) % 10, (n / 1000) % 10, (n / 10000) % 10) in
  digits.1 ≠ digits.2 ∧ digits.2 = digits.3 ∧ digits.3 ≠ digits.4 ∧ digits.4 = digits.5

def is_divisible_by_15 (n : ℕ) : Prop :=
  n % 15 = 0

def is_valid_flippy_number (n : ℕ) : Prop :=
  (10000 ≤ n ∧ n < 100000) ∧ is_flippy n ∧ is_divisible_by_15 n

def count_valid_flippy_numbers : ℕ :=
  finset.card (finset.filter is_valid_flippy_number (finset.range 100000))

theorem flippy_numbers_divisible_by_15_are_4 : count_valid_flippy_numbers = 4 :=
by sorry

end flippy_numbers_divisible_by_15_are_4_l267_267593


namespace min_value_condition_l267_267899

theorem min_value_condition
  (a b c d e f g h : ℝ)
  (h1 : a * b * c * d = 16)
  (h2 : e * f * g * h = 36) :
  ∃ x : ℝ, x = (ae)^2 + (bf)^2 + (cg)^2 + (dh)^2 ∧ x ≥ 576 := sorry

end min_value_condition_l267_267899


namespace solve_for_x_l267_267941

theorem solve_for_x : 
  let x := (√(7^2 + 24^2)) / (√(49 + 16)) in 
  x = 25 * √65 / 65 := 
by
  -- Step 1: expand the terms inside the square roots
  let a := 7^2 + 24^2 
  let b := 49 + 16

  have a_eq : a = 625 := by
    calc
      a = 7^2 + 24^2 : rfl
      ... = 49 + 576 : rfl
      ... = 625 : rfl

  have b_eq : b = 65 := by
    calc
      b = 49 + 16 : rfl
      ... = 65 : rfl

  -- Step 2: Simplify the square roots
  let sqrt_a := √a
  have sqrt_a_eq : sqrt_a = 25 := by
    rw [a_eq]
    norm_num

  let sqrt_b := √b
  have sqrt_b_eq : sqrt_b = √65 := by
    rw [b_eq]

  -- Step 3: Simplify x
  let x := sqrt_a / sqrt_b

  show x = 25 * √65 / 65
  rw [sqrt_a_eq, sqrt_b_eq]
  field_simp
  norm_num
  rw [mul_div_cancel_left 25 (sqrt_ne_zero.2 (ne_of_gt (by norm_num : √65 ≠ 0))) ]
  sorry

end solve_for_x_l267_267941


namespace students_in_grades_2_and_3_l267_267035

theorem students_in_grades_2_and_3 (boys_2nd : ℕ) (girls_2nd : ℕ) (third_grade_factor : ℕ) 
  (h_boys_2nd : boys_2nd = 20) (h_girls_2nd : girls_2nd = 11) (h_third_grade_factor : third_grade_factor = 2) :
  (boys_2nd + girls_2nd) + ((boys_2nd + girls_2nd) * third_grade_factor) = 93 := by
  sorry

end students_in_grades_2_and_3_l267_267035


namespace area_of_triangle_ABC_l267_267310

theorem area_of_triangle_ABC (T1 T2 T3 : ℝ) (T : ℝ) (h₁ : T1 = 4) (h₂ : T2 = 9) (h₃ : T3 = 49) 
  (h₄ : T = ((real.sqrt T1) + (real.sqrt T2) + (real.sqrt T3))^2) : T = 144 :=
by 
  sorry

end area_of_triangle_ABC_l267_267310


namespace ratio_CD_BD_l267_267282

variables (A B C D E T : Type)
variables (Triangle : A → B → C → Type) (AD : A → D) (BE : B → E)
variable (BC : B → C) (AC : A → C)
variables (AT : A → T) (DT : D → T) (BT : B → T) (ET : E → T)

-- Define the necessary conditions as assumptions
axiom AT_DT_ratio (h1 : ∀ (A T D : Type), (AT A T) / (DT D T) = 2)
axiom BT_ET_ratio (h2 : ∀ (B T E : Type), (BT B T) / (ET E T) = 3)

-- Prove the required ratio CD / BD
theorem ratio_CD_BD (A B C D E T : Type) (h1 : AT_DT_ratio) (h2 : BT_ET_ratio) : 
  (CD / BD) = 1 := sorry

end ratio_CD_BD_l267_267282


namespace problem1_problem2_l267_267219

/-
Definition of the sequence a_n
-/
def a : ℕ → ℝ
| 0       => 1/2
| (n + 1) => (1/3) * (a n)^3 + (2/3) * (a n)

/-
Problem 1: Prove that for all n, a_n is bounded by given expressions
-/
theorem problem1 (n : ℕ) (hn : n > 0) :
  1/2 * (2/3)^(n-1) ≤ a n ∧ a n ≤ 1/2 * (3/4)^(n-1) :=
sorry

/-
Problem 2: Prove the summation inequality
-/
theorem problem2 (n : ℕ) (hn : n > 0) :
  (∑ k in Finset.range n, (1 - a (k + 1)) / (1 - a k)) 
  ≥ (∑ k in Finset.range n, a (k + 1) / a k) + 6 * (1 - ((11/12)^n)) :=
sorry

end problem1_problem2_l267_267219


namespace siblings_age_problem_l267_267379

variable {x y z : ℕ}

theorem siblings_age_problem
  (h1 : x - y = 3)
  (h2 : z - 1 = 2 * (x + y))
  (h3 : z + 20 = x + y + 40) :
  x = 11 ∧ y = 8 ∧ z = 39 :=
by
  sorry

end siblings_age_problem_l267_267379


namespace prove_f_g_neg3_l267_267755

def f (x : ℝ) : ℝ :=
if x > 0 then real.log x / real.log 3 - 2 else g(x)

axiom odd_function (f : ℝ → ℝ) : ( ∀ x : ℝ, f (-x) = -f(x) )

axiom g_def : g(-3) = -f(3)

theorem prove_f_g_neg3 :
  f (g (-3)) = -2 :=
by
  have h := odd_function f -- f is an odd function
  rw g_def -- Substitute the definition of g():
  calc
    f (g(-3)) = f (-f(3)) : by rw g_def
            ... = -f (f(3)) : by rw h
            ... = -f (1) : by sorry -- Substitute f(3) = 1 from given conditions
            ... = - (0 - 2) : by sorry -- Substitute f(1) = 0 - 2
            ... = -2 : by sorry

end prove_f_g_neg3_l267_267755


namespace max_selected_numbers_l267_267129

theorem max_selected_numbers {n : ℕ} (h : n > 1999) :
  ∃ S : set ℕ, (∀ a b ∈ S, (a % 100) + (b % 100) = 0) ∧ S.card = 20 :=
sorry

end max_selected_numbers_l267_267129


namespace laptop_full_price_l267_267936

theorem laptop_full_price (p : ℝ) (deposit : ℝ) (h1 : deposit = 0.25 * p) (h2 : deposit = 400) : p = 1600 :=
by
  sorry

end laptop_full_price_l267_267936


namespace distance_between_stripes_l267_267120

/-- Given a crosswalk parallelogram with curbs 60 feet apart, a base of 20 feet, 
and each stripe of length 50 feet, show that the distance between the stripes is 24 feet. -/
theorem distance_between_stripes (h : Real) (b : Real) (s : Real) : h = 60 ∧ b = 20 ∧ s = 50 → (b * h) / s = 24 :=
by
  sorry

end distance_between_stripes_l267_267120


namespace geometric_sequence_of_c_n_l267_267704

open_locale big_operators

variables {α : Type*} [comm_ring α] (a : ℕ → α) (q : α) (m : ℕ) (n : ℕ)
hypothesis (hq : ∀ (k : ℕ), a (k+1) = q * a k)
hypothesis (hm_pos : 0 < m)
hypothesis (hn_pos : 0 < n)

def c_n (n : ℕ) : α :=
∏ i in finset.range m, a (m * (n - 1) + i + 1)

theorem geometric_sequence_of_c_n :
  ∀ (n : ℕ), c_n q m (n + 1) = q^m * c_n q m n :=
sorry

end geometric_sequence_of_c_n_l267_267704


namespace number_of_boys_l267_267086

def totalEarnings : ℕ := 150
def menWages : ℕ := 10
def numberOfMen : ℕ := 5

def women (W : ℕ) : Prop := numberOfMen = W
def boys (B : ℕ) : Prop := numberOfMen = B
def earningsByMen : ℕ := numberOfMen * menWages
def remainingEarnings : ℕ := totalEarnings - earningsByMen

theorem number_of_boys (W B : ℕ) (hW : women W) (hB : boys B) (wageW_wageB : W = B) : B = 10 :=
by
  have wages_eq : 5 * W + 5 * B = remainingEarnings := sorry
  have same_wage : W = 10 := sorry
  exact nat.cast_inj.mpr same_wage

end number_of_boys_l267_267086


namespace max_divisor_of_five_consecutive_integers_l267_267511

theorem max_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, 60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intros n
  sorry

end max_divisor_of_five_consecutive_integers_l267_267511


namespace find_speed_l267_267921

noncomputable def distance : ℝ := 600
noncomputable def speed1 : ℝ := 50
noncomputable def meeting_distance : ℝ := distance / 2
noncomputable def departure_time1 : ℝ := 7
noncomputable def departure_time2 : ℝ := 8
noncomputable def meeting_time : ℝ := 13

theorem find_speed (x : ℝ) : 
  (meeting_distance / speed1 = meeting_time - departure_time1) ∧
  (meeting_distance / x = meeting_time - departure_time2) → 
  x = 60 :=
by
  sorry

end find_speed_l267_267921


namespace juice_bar_group_total_l267_267097

theorem juice_bar_group_total (total_spent : ℕ) (mango_cost : ℕ) (pineapple_cost : ℕ) 
  (spent_on_pineapple : ℕ) (num_people_total : ℕ) :
  total_spent = 94 →
  mango_cost = 5 →
  pineapple_cost = 6 →
  spent_on_pineapple = 54 →
  num_people_total = (40 / 5) + (54 / 6) →
  num_people_total = 17 :=
by {
  intros h_total h_mango h_pineapple h_pineapple_spent h_calc,
  sorry
}

end juice_bar_group_total_l267_267097


namespace geometric_seq_sum_l267_267836

-- Definitions of the conditions
variables {a₁ q : ℚ}
def S (n : ℕ) := a₁ * (1 - q^n) / (1 - q)

-- Hypotheses from the conditions
theorem geometric_seq_sum :
  (S 4 = -5) →
  (S 6 = 21 * S 2) →
  (S 8 = -85) :=
by
  -- Assume the given conditions
  intros h1 h2
  -- The actual proof will be inserted here
  sorry

end geometric_seq_sum_l267_267836


namespace five_consecutive_product_div_24_l267_267490

theorem five_consecutive_product_div_24 (n : ℤ) : 
  24 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) := 
sorry

end five_consecutive_product_div_24_l267_267490


namespace jimmy_can_lose_5_more_points_l267_267314

theorem jimmy_can_lose_5_more_points (min_points_to_pass : ℕ) (points_per_exam : ℕ) (number_of_exams : ℕ) (points_lost : ℕ) : 
  min_points_to_pass = 50 → 
  points_per_exam = 20 → 
  number_of_exams = 3 → 
  points_lost = 5 → 
  (points_per_exam * number_of_exams - points_lost - 5) = min_points_to_pass :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end jimmy_can_lose_5_more_points_l267_267314


namespace vertex_of_parabola_l267_267959

theorem vertex_of_parabola (x : ℝ) : 
  ∀ x y : ℝ, (y = x^2 - 6 * x + 1) → (∃ h k : ℝ, y = (x - h)^2 + k ∧ h = 3 ∧ k = -8) :=
by
  -- This is to state that given the parabola equation x^2 - 6x + 1, its vertex coordinates are (3, -8).
  sorry

end vertex_of_parabola_l267_267959


namespace average_visitors_on_other_days_l267_267586

theorem average_visitors_on_other_days
  (avg_sunday_visitors : ℕ) (avg_monthly_visitors : ℕ) (num_sundays : ℕ) (num_days_in_month : ℕ)
  (total_days_in_month : num_days_in_month = 30) (first_day_sunday : Prop)
  (num_other_days : num_days_in_month - num_sundays = 25) :
  avg_sunday_visitors = 510 →
  avg_monthly_visitors = 285 →
  ∃ (V : ℕ), 
    let total_sunday_visitors := num_sundays * avg_sunday_visitors,
        total_visitors_in_month := avg_monthly_visitors * num_days_in_month,
        total_other_day_visitors := num_other_days * V
    in 
    total_sunday_visitors + total_other_day_visitors = total_visitors_in_month ∧ V = 240 := 
by
  sorry

end average_visitors_on_other_days_l267_267586


namespace integer_values_in_range_l267_267415

theorem integer_values_in_range :
  {x : ℤ | 5 < real.sqrt x ∧ real.sqrt x < 6}.finite.to_finset.card = 10 :=
by
  sorry

end integer_values_in_range_l267_267415


namespace part_1_part_2_l267_267907

def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem part_1 (x : ℝ) : f x ≤ 4 ↔ x ∈ Set.Icc (-2 : ℝ) 2 :=
by sorry

theorem part_2 (b : ℝ) (h₁ : b ≠ 0) (x : ℝ) (h₂ : f x ≥ (|2 * b + 1| + |1 - b|) / |b|) : x ≤ -1.5 :=
by sorry

end part_1_part_2_l267_267907


namespace largest_divisor_of_five_consecutive_integers_l267_267456

open Nat

theorem largest_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, k ∈ {n, n+1, n+2, n+3, n+4} ∧
    ∀ m ∈ {2, 3, 4, 5}, m ∣ k → 60 ∣ (n * (n+1) * (n+2) * (n+3) * (n+4)) := 
sorry

end largest_divisor_of_five_consecutive_integers_l267_267456


namespace count_distinct_z_values_l267_267724

noncomputable def distinct_z_values : set ℤ := { z | ∃ x y : ℤ,
  1000 ≤ x ∧ x ≤ 9999 ∧
  1000 ≤ y ∧ y ≤ 9999 ∧
  y = (1000 * (x % 10) + 100 * ((x / 10) % 10) + 10 * ((x / 100) % 10) + (x / 1000)) ∧
  z = |x - y| }

theorem count_distinct_z_values : ∃ s : set ℤ, s = distinct_z_values :=
begin
  sorry
end

end count_distinct_z_values_l267_267724


namespace prime_pattern_l267_267812

theorem prime_pattern (n x : ℕ) (h1 : x = (10^n - 1) / 9) (h2 : Prime x) : Prime n :=
sorry

end prime_pattern_l267_267812


namespace four_digit_integers_count_l267_267260

theorem four_digit_integers_count :
  let S := { n : ℕ | n % 7 = 3 ∧ n % 8 = 4 ∧ n % 10 = 6 ∧ 1000 ≤ n ∧ n < 10000 }
  in S.card = 161 :=
by
  sorry

end four_digit_integers_count_l267_267260


namespace max_divisor_of_five_consecutive_integers_l267_267514

theorem max_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, 60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intros n
  sorry

end max_divisor_of_five_consecutive_integers_l267_267514


namespace mrs_snyder_income_l267_267347

theorem mrs_snyder_income :
  ∃ I : ℝ, 
    (0.45 * I = 0.30 * (I + 850) ∧ I = 1700) :=
by
  use 1700
  split
  . simp
  . rfl

end mrs_snyder_income_l267_267347


namespace product_of_five_consecutive_integers_divisible_by_120_l267_267446

theorem product_of_five_consecutive_integers_divisible_by_120 (n : ℤ) : 
  120 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end product_of_five_consecutive_integers_divisible_by_120_l267_267446


namespace adjacent_abby_bridget_probability_l267_267126
open Nat

-- Define the conditions
def total_kids := 6
def grid_rows := 3
def grid_cols := 2
def middle_row := 2
def abby_and_bridget := 2

-- Define the probability calculation
theorem adjacent_abby_bridget_probability :
  let total_arrangements := 6!
  let num_ways_adjacent :=
    (2 * abby_and_bridget) * (total_kids - abby_and_bridget)!
  let total_outcomes := total_arrangements
  (num_ways_adjacent / total_outcomes : ℚ) = 4 / 15
:= sorry

end adjacent_abby_bridget_probability_l267_267126


namespace difference_shares_l267_267630

-- Given conditions in the problem
variable (V : ℕ) (F R : ℕ)
variable (hV : V = 1500)
variable (hRatioF : F = 3 * (V / 5))
variable (hRatioR : R = 11 * (V / 5))

-- The statement we need to prove
theorem difference_shares : R - F = 2400 :=
by
  -- Using the conditions to derive the result.
  sorry

end difference_shares_l267_267630


namespace product_of_5_consecutive_integers_divisible_by_60_l267_267537

theorem product_of_5_consecutive_integers_divisible_by_60 :
  ∀a : ℤ, 60 ∣ (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
by
  sorry

end product_of_5_consecutive_integers_divisible_by_60_l267_267537


namespace hospitals_in_pittsburgh_l267_267795

theorem hospitals_in_pittsburgh :
  let s := 2000
  let sc := 200
  let ps := 20
  let total_new_city := 2175
  ∃ H : ℕ, (s / 2) + (2 * H) + (sc - 50) + (ps + 5) = total_new_city ∧ H = 500 :=
by
  let s := 2000
  let sc := 200
  let ps := 20
  let total_new_city := 2175
  use 500
  split
  { show (s / 2) + (2 * 500) + (sc - 50) + (ps + 5) = total_new_city
    calc
      1000 + (2 * 500) + (sc - 50) + (ps + 5)
      = 1000 + 1000 + 150 + 25
        : by simp [sc, ps]
      ... = 2175 : by norm_num
  }
  { show 500 = 500
    refl
  }

end hospitals_in_pittsburgh_l267_267795


namespace largest_divisor_of_product_of_five_consecutive_integers_l267_267502

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∀ (n : ℤ), ∃ k : ℤ, k = 60 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intro n
  use 60
  split
  { refl }
  { sorry }

end largest_divisor_of_product_of_five_consecutive_integers_l267_267502


namespace eight_term_sum_l267_267819

variable {α : Type*} [Field α]
variable (a q : α)

-- Define the n-th sum of the geometric sequence
def S_n (n : ℕ) : α := a * (1 - q ^ n) / (1 - q)

-- Given conditions
def S4 : α := S_n 4 = -5
def S6 : α := S_n 6 = 21 * S_n 2

-- Prove the target statement
theorem eight_term_sum : S_n 8 = -85 :=
  sorry

end eight_term_sum_l267_267819


namespace hyperbola_quadrilateral_area_l267_267722

theorem hyperbola_quadrilateral_area (a : ℝ) (e : ℝ) (y0 : ℝ) (F1 F2 P Q : ℝ × ℝ)
  (h1 : a > 0)
  (h2 : e = Real.sqrt 5 / 2)
  (h3 : F1 = (-2 * a * e, 0))
  (h4 : F2 = (2 * a * e, 0))
  (h5 : P = (5, y0))
  (h6 : Q = (-5, -y0))
  (h7 : (5^2)/(a^2) - (y0^2)/4 = 1)
  (h8 : Q = (-5, -y0))
  :
  let area := 2 * (1/2) * abs (4 * Real.sqrt 5 * y0 / 2) in
  area = 6 * Real.sqrt 5 :=
  sorry

end hyperbola_quadrilateral_area_l267_267722


namespace product_factorial_integer_l267_267351

theorem product_factorial_integer (k : ℕ) (hk : 0 < k) :
  (k^2)! * ∏ j in Finset.range k, (j! / (j + k)!) ∈ Int :=
sorry

end product_factorial_integer_l267_267351


namespace problem_l267_267408

def a : ℝ := (-2)^2002
def b : ℝ := (-2)^2003

theorem problem : a + b = -2^2002 := by
  sorry

end problem_l267_267408


namespace power_sum_calculation_l267_267147

theorem power_sum_calculation : (-1)^53 + 3^(2^3 + 5^2 - 4!) = 19682 := 
by
  sorry

end power_sum_calculation_l267_267147


namespace find_difference_l267_267638

noncomputable def g : ℝ → ℝ := sorry    -- Definition of the function g (since it's graph-based and specific)

-- Given conditions
variables (c d : ℝ)
axiom h1 : Function.Injective g          -- g is an invertible function (injective functions have inverses)
axiom h2 : g c = d
axiom h3 : g d = 6

-- Theorem to prove
theorem find_difference : c - d = -2 :=
by {
  -- sorry is needed since the exact proof steps are not provided
  sorry
}

end find_difference_l267_267638


namespace students_in_grades_2_and_3_l267_267036

theorem students_in_grades_2_and_3 (boys_2nd : ℕ) (girls_2nd : ℕ) (third_grade_factor : ℕ) 
  (h_boys_2nd : boys_2nd = 20) (h_girls_2nd : girls_2nd = 11) (h_third_grade_factor : third_grade_factor = 2) :
  (boys_2nd + girls_2nd) + ((boys_2nd + girls_2nd) * third_grade_factor) = 93 := by
  sorry

end students_in_grades_2_and_3_l267_267036


namespace comparison_of_logs_l267_267233

noncomputable def a : ℝ := Real.log 6 / Real.log 3
noncomputable def b : ℝ := Real.log 12 / Real.log 6
noncomputable def c : ℝ := Real.log 16 / Real.log 8

theorem comparison_of_logs : a > b ∧ b > c :=
by
  sorry

end comparison_of_logs_l267_267233


namespace sum_of_eight_l267_267874

variable (a₁ : ℕ) (q : ℕ)
variable (S : ℕ → ℕ) -- Assume S is a function from natural numbers to natural numbers representing S_n

-- Condition 1: Sum of first 4 terms equals -5
axiom h1 : S 4 = -5

-- Condition 2: Sum of the first 6 terms is 21 times the sum of the first 2 terms
axiom h2 : S 6 = 21 * S 2

-- The formula for the sum of the first n terms of a geometric sequence
def sum_of_first_n_terms (n : ℕ) := a₁ * (1 - q^n) / (1 - q)

-- Statement to be proved
theorem sum_of_eight : S 8 = -85 := sorry

end sum_of_eight_l267_267874


namespace intersection_at_y_axis_l267_267772

theorem intersection_at_y_axis : ∃ y, (y = 5 * 0 + 1) ∧ (0, y) = (0, 1) :=
begin
  use 1,
  split,
  { norm_num, },
  { refl, },
end

end intersection_at_y_axis_l267_267772


namespace number_of_ways_to_fill_grid_l267_267681

-- Conditions as Definitions
def grid_size : ℕ := 3
def numbers : List ℕ := List.range' 1 (grid_size * grid_size + 1)
def fixed_center : ℕ := 4

-- Statement of the problem
theorem number_of_ways_to_fill_grid : 
  ∃ arrangements : Finset (Matrix (Fin grid_size) (Fin grid_size) ℕ),
  (∀ A ∈ arrangements, 
    (∀ i j, A i j ∈ numbers) ∧ 
    (∀ j, strict_mono (λ i, A i j)) ∧ 
    (∀ i, strict_mono (λ j, A i j)) ∧ 
    (A (Fin.ofNat 1) (Fin.ofNat 1) = fixed_center)) ∧ 
  arrangements.card = 12 :=
by
  sorry

end number_of_ways_to_fill_grid_l267_267681


namespace students_per_group_l267_267991

theorem students_per_group :
  ∀ (total_students not_picked groups : ℕ),
  total_students = 35 →
  not_picked = 11 →
  groups = 4 →
  (total_students - not_picked) / groups = 6 :=
begin
  intros total_students not_picked groups h1 h2 h3,
  rw [h1, h2, h3],
  norm_num,
end

end students_per_group_l267_267991


namespace max_list_elem_l267_267588

theorem max_list_elem : ∀ (L : List ℕ), 
  (L.length = 4) →
  ((List.nthLe L 1 sorry + List.nthLe L 2 sorry) / 2 = 4) →
  (List.sum L / L.length = 10) →
  ∃ x, L.max = x ∧ x = 28 :=
by
  sorry

end max_list_elem_l267_267588


namespace sequence_solution_l267_267654

/-
Let {s : ℕ → ℚ} be a sequence defined by:
1. s 1 = 2
2. ∀ n > 1, if n % 3 = 0 then s n = 2 + s (n / 3)
3. ∀ n > 1, if n % 3 ≠ 0 then s n = 2 / s (n - 1)

Prove that if s n = 13 / 29 then n = 154305.
-/

noncomputable def s : ℕ → ℚ
| 1       => 2
| (n + 1) => if (n + 1) % 3 = 0 then 2 + s ((n + 1) / 3) else 2 / s n

theorem sequence_solution (n : ℕ) (h : s n = 13 / 29) : n = 154305 :=
by
  sorry

end sequence_solution_l267_267654


namespace equilateral_triangle_area_l267_267380

theorem equilateral_triangle_area (altitude : ℝ) (h : altitude = 3 * Real.sqrt 3) : 
  ∃ (area : ℝ), area = 9 * Real.sqrt 3 :=
by
  use 9 * Real.sqrt 3
  rw h
  sorry

end equilateral_triangle_area_l267_267380


namespace ashok_average_marks_l267_267135

theorem ashok_average_marks (avg_6 : ℝ) (marks_6 : ℝ) (total_sub : ℕ) (sub_6 : ℕ)
  (h1 : avg_6 = 75) (h2 : marks_6 = 80) (h3 : total_sub = 6) (h4 : sub_6 = 5) :
  (avg_6 * total_sub - marks_6) / sub_6 = 74 :=
by
  sorry

end ashok_average_marks_l267_267135


namespace regular_polygon_sides_l267_267602

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ n → ∃ p, ∠(polygon_interior_angle p) = 144) : n = 10 := 
by
  sorry

end regular_polygon_sides_l267_267602


namespace sum_recurrence_relation_l267_267215

-- Definitions based on the conditions
def a : ℕ → ℚ
| 0 := 1
| 1 := 2
| (n + 2) := (n * (n - 1) * a (n + 1) - (n - 2) * a n) / (n * (n + 1))

-- Theorem statement to prove
theorem sum_recurrence_relation :
  (∑ n in Finset.range 51, a n / a (n + 1)) = 1327 :=
by
  sorry

end sum_recurrence_relation_l267_267215


namespace f_is_odd_l267_267024

def f (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by
  intros x
  -- proof goes here
  sorry

end f_is_odd_l267_267024


namespace find_f_2_find_f_neg2_l267_267663

noncomputable def f : ℝ → ℝ := sorry -- This is left to be defined as a function on ℝ

axiom f_property : ∀ x y : ℝ, f (x + y) = f x + f y + 2 * x * y
axiom f_at_1 : f 1 = 2

theorem find_f_2 : f 2 = 6 := by
  sorry

theorem find_f_neg2 : f (-2) = 2 := by
  sorry

end find_f_2_find_f_neg2_l267_267663


namespace largest_divisor_of_product_of_five_consecutive_integers_l267_267470

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∀ (n : ℤ), ∃ (d : ℤ), d = 60 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end largest_divisor_of_product_of_five_consecutive_integers_l267_267470


namespace S₈_proof_l267_267857

-- Definitions
variables {a₁ q : ℝ} {S : ℕ → ℝ}

-- Condition for the sum of the first n terms of the geometric sequence
def geom_sum (n : ℕ) : ℝ :=
  a₁ * (1 - q ^ n) / (1 - q)

-- Conditions from the problem
def S₄ := geom_sum 4 = -5
def S₆ := geom_sum 6 = 21 * geom_sum 2

-- Theorem to prove
theorem S₈_proof (h₁ : S₄) (h₂ : S₆) : geom_sum 8 = -85 :=
by
  sorry

end S₈_proof_l267_267857


namespace p_divisible_by_prime_l267_267338

theorem p_divisible_by_prime {k p q : ℕ} (h_eq : (p : ℚ) / q = ∑ i in finset.range (4*k-1), (-1)^(i+1) / (i+1)) 
  (h_prime : prime (6*k - 1)) : 
  6*k - 1 ∣ p :=
sorry

end p_divisible_by_prime_l267_267338


namespace birds_in_trees_l267_267399

def number_of_stones := 40
def number_of_trees := number_of_stones + 3 * number_of_stones
def combined_number := number_of_trees + number_of_stones
def number_of_birds := 2 * combined_number

theorem birds_in_trees : number_of_birds = 400 := by
  sorry

end birds_in_trees_l267_267399


namespace point_A_circle_l267_267089

-- Define the geometry, including the circle C with center P and radius r.
-- B is a point inside the circle C on the line passing through P, such that the distance PB = d where d < r.

variables {P B A : Point}
variables {r d : ℝ}
variables [metric_space Point]

-- Some required conditions
def circle_equation (P : Point) (r : ℝ) (C : set Point) := ∀ x, x ∈ C ↔ dist x P = r

axiom B_inside_circle (h : B ∈ C) : dist B P = d ∧ d < r
axiom A_condition (h1 : ∀ x ∈ C, dist A B < dist A x) : A ∈ set.univ

-- Theorem statement expressing that the set of points A forms a circle centered at B with radius r - d
theorem point_A_circle
  (hC : circle_equation P r C)
  (hB : B_inside_circle B)
  (hA : A_condition A)
  : ∀ y : Point, A ∈ set.univ ↔ dist A B < r - d := 
sorry

end point_A_circle_l267_267089


namespace geometric_sequence_sum_l267_267888

variable (a1 q : ℝ) -- Define the first term and common ratio as real numbers

-- Define the sum of the first n terms of a geometric sequence
def S (n : ℕ) : ℝ := a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum :
  (S 4 a1 q = -5) → (S 6 a1 q = 21 * S 2 a1 q) → (S 8 a1 q = -85) :=
by
  intro h1 h2
  sorry

end geometric_sequence_sum_l267_267888


namespace polynomial_at_3_l267_267337

theorem polynomial_at_3 (P : ℝ → ℝ)
  (h1 : ∀ x : ℝ, P(x) = P(0) + P(1) * x + P(2) * x^2)
  (h2 : P(-1) = 1) :
  P(3) = 5 :=
sorry

end polynomial_at_3_l267_267337


namespace ajay_avg_monthly_income_l267_267625

theorem ajay_avg_monthly_income 
  (saving_percents : List ℚ)
  (total_savings : ℚ)
  (months : ℕ)
  (total_income : ℚ) :
  saving_percents = [10, 15, 8, 12, 7, 18] →
  months = 6 →
  total_savings = 58500 →
  total_income / months ≈ 13928.57 :=
by
  have h_total_percent := saving_percents.sum * months ≈ 70
  have h_income := h_total_percent * total_income ≈ total_savings
  sorry

end ajay_avg_monthly_income_l267_267625


namespace even_monotonic_inequality_l267_267571

-- Definitions of the conditions
variable {α : Type*} [LinearOrder α] [Preorder α]
variable (f : α → ℝ)

-- Conditions
-- f is an even function
def even_function (f : α → ℝ) (a : α) (ha : a ∈ set.Icc (-5 : α) (5 : α)) : Prop :=
  ∀ x ∈ set.Icc (-5 : α) (5 : α), f (-x) = f x

-- f is monotonic on [0, 5]
def monotonic_on (f : α → ℝ) (a b : α) : Prop :=
  ∀ x y ∈ set.Icc a b, x ≤ y → f x ≤ f y ∨ f x ≥ f y

-- Given inequality
def given_inequality (f : α → ℝ) : Prop :=
  f (-4 : α) < f (-2 : α)

-- Proof problem to show "f(0) > f(1)" given the above conditions
theorem even_monotonic_inequality (hf_even : even_function f 5 (by norm_num))
  (hf_monotonic : monotonic_on f 0 5) (h_ineq : given_inequality f) : f 0 > f 1 := 
sorry


end even_monotonic_inequality_l267_267571


namespace max_divisor_of_five_consecutive_integers_l267_267505

theorem max_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, 60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intros n
  sorry

end max_divisor_of_five_consecutive_integers_l267_267505


namespace product_of_5_consecutive_integers_divisible_by_60_l267_267535

theorem product_of_5_consecutive_integers_divisible_by_60 :
  ∀a : ℤ, 60 ∣ (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
by
  sorry

end product_of_5_consecutive_integers_divisible_by_60_l267_267535


namespace part1_part2_part3_l267_267199

open Real

-- Definition of "$k$-derived point"
def k_derived_point (P : ℝ × ℝ) (k : ℝ) : ℝ × ℝ := (P.1 + k * P.2, k * P.1 + P.2)

-- Problem statements to prove
theorem part1 :
  k_derived_point (-2, 3) 2 = (4, -1) :=
sorry

theorem part2 (P : ℝ × ℝ) (h : k_derived_point P 3 = (9, 11)) :
  P = (3, 2) :=
sorry

theorem part3 (b k : ℝ) (h1 : b > 0) (h2 : |k * b| ≥ 5 * b) :
  k ≥ 5 ∨ k ≤ -5 :=
sorry

end part1_part2_part3_l267_267199


namespace polygon_is_decagon_l267_267723

-- Definitions based on conditions
def exterior_angles_sum (x : ℕ) : ℝ := 360

def interior_angles_sum (x : ℕ) : ℝ := 4 * exterior_angles_sum x

def interior_sum_formula (n : ℕ) : ℝ := (n - 2) * 180

-- Mathematically equivalent proof problem
theorem polygon_is_decagon (n : ℕ) (h1 : exterior_angles_sum n = 360)
  (h2 : interior_angles_sum n = 4 * exterior_angles_sum n)
  (h3 : interior_sum_formula n = interior_angles_sum n) : n = 10 :=
sorry

end polygon_is_decagon_l267_267723


namespace nesbitts_inequality_l267_267702

variable (a b c : ℝ)

theorem nesbitts_inequality (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (b + c) + b / (c + a) + c / (a + b)) >= 3 / 2 := 
sorry

end nesbitts_inequality_l267_267702


namespace largest_divisor_of_5_consecutive_integers_l267_267527

theorem largest_divisor_of_5_consecutive_integers :
  ∀ (n : ℤ), ∃ d, d = 120 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intro n
  use 120
  split
  exact rfl
  sorry

end largest_divisor_of_5_consecutive_integers_l267_267527


namespace circles_touching_line_l267_267562

theorem circles_touching_line (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : three_circles_touch_each_other_and_line a b c) : 
  1 / Real.sqrt c = 1 / Real.sqrt a + 1 / Real.sqrt b := 
sorry

end circles_touching_line_l267_267562


namespace cos_B_of_triangle_ABC_l267_267237

theorem cos_B_of_triangle_ABC (A B C : Type)
  [inner_product_space ℝ A]
  [triangle A B C]
  (BC : ℝ) (AC : ℝ) (angle_A : ℝ)
  (hBC : BC = 15)
  (hAC : AC = 10)
  (hA : angle_A = 60 * (π / 180)) :
  let B := ∠B A C in
  cos B = √6 / 3 := 
by
  sorry

end cos_B_of_triangle_ABC_l267_267237


namespace probability_interval_l267_267705

open MeasureTheory ProbabilityTheory

theorem probability_interval (X : ℝ → ℝ) (μ σ : ℝ) :
  (∀ x, X x ∈ Normal μ (σ^2)) →
  (P {x | μ - 2 * σ < X x ∧ X x < μ + 2 * σ} = 0.9544) →
  (P {x | μ - σ < X x ∧ X x < μ + σ} = 0.6826) →
  μ = 4 →
  σ = 1 →
  P {x | 5 < X x ∧ X x < 6} = 0.1359 :=
by
  intros
  sorry

end probability_interval_l267_267705


namespace annual_rent_per_square_foot_correct_l267_267967

noncomputable def shop_dimensions : ℝ × ℝ := (20, 18)
noncomputable def monthly_rent : ℝ := 1440
noncomputable def annual_rent_per_square_foot : ℝ := 48

theorem annual_rent_per_square_foot_correct : 
  let area := shop_dimensions.1 * shop_dimensions.2 in
  let annual_rent := monthly_rent * 12 in
  annual_rent / area = annual_rent_per_square_foot := 
by 
  -- Proof would be here
  sorry

end annual_rent_per_square_foot_correct_l267_267967


namespace product_of_5_consecutive_integers_divisible_by_60_l267_267536

theorem product_of_5_consecutive_integers_divisible_by_60 :
  ∀a : ℤ, 60 ∣ (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
by
  sorry

end product_of_5_consecutive_integers_divisible_by_60_l267_267536


namespace number_of_panes_l267_267622

theorem number_of_panes (length width total_area : ℕ) (h_length : length = 12) (h_width : width = 8) (h_total_area : total_area = 768) :
  total_area / (length * width) = 8 :=
by
  sorry

end number_of_panes_l267_267622


namespace correct_propositions_l267_267256

variables (m n l : Type)
variables (α β : Type)
variables (parallel : ∀ {x y : Type}, Prop)
variables (perpendicular : ∀ {x y : Type}, Prop)
variables (subset : ∀ {x y : Type}, Prop)
variables (intersection : ∀ {x y : Type}, Type)

open Function 

axiom m_parallel_n : parallel m n
axiom n_in_alpha : subset n α
axiom l_perp_alpha : perpendicular l α
axiom m_perp_beta : perpendicular m β
axiom l_parallel_m : parallel l m
axiom m_in_alpha : subset m α
axiom n_in_alpha_2 : subset n α
axiom m_parallel_beta : parallel m β
axiom n_parallel_beta : parallel n β
axiom alpha_perp_beta : perpendicular α β
axiom alpha_cap_beta : intersection α β = m
axiom n_in_beta : subset n β
axiom n_perp_m : perpendicular n m

def number_of_correct_propositions : ℕ := 2

theorem correct_propositions :
  (¬ (m_parallel_n → (parallel m α ∨ subset m α))) ∧
  (l_perp_alpha ∧ m_perp_beta ∧ l_parallel_m → parallel α β) ∧
  (¬ (m_in_alpha ∧ n_in_alpha_2 ∧ m_parallel_beta ∧ n_parallel_beta → parallel α β)) ∧
  (alpha_perp_beta ∧ alpha_cap_beta ∧ n_in_beta ∧ n_perp_m → perpendicular n α) →
  number_of_correct_propositions = 2 :=
sorry

end correct_propositions_l267_267256


namespace eight_term_sum_l267_267821

variable {α : Type*} [Field α]
variable (a q : α)

-- Define the n-th sum of the geometric sequence
def S_n (n : ℕ) : α := a * (1 - q ^ n) / (1 - q)

-- Given conditions
def S4 : α := S_n 4 = -5
def S6 : α := S_n 6 = 21 * S_n 2

-- Prove the target statement
theorem eight_term_sum : S_n 8 = -85 :=
  sorry

end eight_term_sum_l267_267821


namespace circle_equation_tangent_lines_l267_267703

-- Definition of the problem conditions
def center : ℝ × ℝ := (1, 2)
def line (x y : ℝ) : Prop := 2 * x - y - 5 = 0
def chord_length : ℝ := 4 * real.sqrt 5
def point_P : ℝ × ℝ := (-4, -13)

-- Statement of the proof problem
theorem circle_equation 
    (h1 : ∀ x y, line x y → (2 * x - y - 5 = 0))
    (h2 : chord_length = 4 * real.sqrt 5) :
    ∃ R, (R = 5) ∧ ((x - 1)^2 + (y - 2)^2 = R^2) := 
sorry

theorem tangent_lines 
    (h1 : ∃ P, point_P = (-4, -13)) 
    (h2 : ∀ x y, line x y → (2 * x - y - 5 = 0)) :
    ∃ k, k = 4 / 3 ∨ k = -1 :=
sorry

end circle_equation_tangent_lines_l267_267703


namespace product_of_5_consecutive_integers_divisible_by_60_l267_267544

theorem product_of_5_consecutive_integers_divisible_by_60 :
  ∀a : ℤ, 60 ∣ (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
by
  sorry

end product_of_5_consecutive_integers_divisible_by_60_l267_267544


namespace determine_var_phi_l267_267175

open Real

theorem determine_var_phi (φ : ℝ) (h₀ : 0 ≤ φ ∧ φ ≤ 2 * π) :
  (∀ x, sin (x + φ) = sin (x - π / 6)) → φ = 11 * π / 6 :=
by
  sorry

end determine_var_phi_l267_267175


namespace line_passing_through_points_l267_267384

theorem line_passing_through_points (m b : ℝ) (h₁ : ∃ m b, ∀ x y, (x, y) = (1, -2) ∨ (x, y) = (-2, 7) → y = m * x + b) :
  m + b = -2 :=
by
  -- Assuming given points and conditions
  have hline : ∀ x y, (x, y) = (1, -2) ∨ (x, y) = (-2, 7) → y = m * x + b, from h₁,
  sorry

end line_passing_through_points_l267_267384


namespace painting_arrangement_count_l267_267938

def painting_area (w h : ℕ): ℕ := w * h

def paintings := [(2, 1), (1, 1), (1, 1), (1, 2), (1, 2), (2, 2), (2, 2),
                  (4, 3), (4, 3), (4, 4), (4, 4)]

def total_painting_area : ℕ :=
  paintings.foldl (λ acc (dim : ℕ × ℕ), acc + painting_area dim.fst dim.snd) 0

def wall_area : ℕ := painting_area 12 6

theorem painting_arrangement_count :
  total_painting_area = wall_area →
  -- If the total painting area matches the wall area
  -- then the number of ways to arrange the paintings is 16896.
  ∃ n : ℕ, n = 16896 :=
begin
  intro h,
  -- Add detailed breakdown if required.
  sorry,
end

#eval painting_arrangement_count

end painting_arrangement_count_l267_267938


namespace trig_identity_l267_267560

theorem trig_identity (x y : ℝ) : cos(x)^4 + sin(y)^2 + (1 / 4) * sin(2 * x)^2 - 1 = sin(y + x) * sin(y - x) :=
sorry

end trig_identity_l267_267560


namespace max_red_dominated_rows_plus_blue_dominated_columns_l267_267811

-- Definitions of the problem conditions and statement
theorem max_red_dominated_rows_plus_blue_dominated_columns (m n : ℕ)
  (h1 : Odd m) (h2 : Odd n) (h3 : 0 < m ∧ 0 < n) :
  ∃ A : Finset (Fin m) × Finset (Fin n),
  (A.1.card + A.2.card = m + n - 2) :=
sorry

end max_red_dominated_rows_plus_blue_dominated_columns_l267_267811


namespace normal_median_eq_mean_l267_267342
noncomputable theory
open ProbabilityTheory

variable {Ω : Type*} [MeasureSpace Ω]

def std_normal (μ σ : ℝ) : MeasureTheory.ProbMeasure ℝ :=
  MeasureTheory.MeasureSpace.NormalDistribution μ σ

theorem normal_median_eq_mean (X : Ω → ℝ) (hX : ∀ω, X ω = MeasureTheory.sample (std_normal 2 (3^2)) ω)
  (c : ℝ) (hP : P(X ≤ c) = P(X > c)) : c = 2 :=
by sorry

end normal_median_eq_mean_l267_267342


namespace range_of_m_l267_267273

variables (λ : ℝ) (m : ℝ)

theorem range_of_m (h : ∀ x : ℝ, |x + 1| - |x - 3| ≤ 2^λ + m / 2^λ) (h₁ : m > 0) : m ≥ 4 :=
sorry

end range_of_m_l267_267273


namespace Kims_final_score_l267_267023

def easy_points : ℕ := 2
def average_points : ℕ := 3
def hard_points : ℕ := 5
def expert_points : ℕ := 7

def easy_correct : ℕ := 6
def average_correct : ℕ := 2
def hard_correct : ℕ := 4
def expert_correct : ℕ := 3

def complex_problems_bonus : ℕ := 1
def complex_problems_solved : ℕ := 2

def penalty_per_incorrect : ℕ := 1
def easy_incorrect : ℕ := 1
def average_incorrect : ℕ := 2
def hard_incorrect : ℕ := 2
def expert_incorrect : ℕ := 3

theorem Kims_final_score : 
  (easy_correct * easy_points + 
   average_correct * average_points + 
   hard_correct * hard_points + 
   expert_correct * expert_points + 
   complex_problems_solved * complex_problems_bonus) - 
   (easy_incorrect * penalty_per_incorrect + 
    average_incorrect * penalty_per_incorrect + 
    hard_incorrect * penalty_per_incorrect + 
    expert_incorrect * penalty_per_incorrect) = 53 :=
by 
  sorry

end Kims_final_score_l267_267023


namespace first_5_serial_numbers_correct_l267_267999

noncomputable def first_5_serial_numbers (random_row : list ℕ) : list ℕ :=
  [785, 567, 199, 507, 175]

theorem first_5_serial_numbers_correct : 
  let random_row := [63, 01, 63, 78, 59, 16, 95, 55, 67, 19, 98, 10, 50, 71, 75, 12, 86, 73, 58, 07, 44, 39, 52, 38, 79] in
  first_5_serial_numbers random_row = [785, 567, 199, 507, 175] :=
  by {
    unfold first_5_serial_numbers,
    sorry 
  }

end first_5_serial_numbers_correct_l267_267999


namespace sequence_formula_l267_267734

noncomputable def a_seq (n : ℕ) : ℂ :=
  let C1 := ↑(-3 : ℤ) / 8 + (7 : ℤ) / 8 * complex.I
  let C2 := ↑(-3 : ℤ) / 8 - (7 : ℤ) / 8 * complex.I
  let C3 := (3 : ℂ) / 4
  let C4 := (-(19 : ℂ)) / 4
  let C5 := (27 : ℂ) / 4
  C1 * (complex.I ^ n) + C2 * ((-complex.I) ^ n) + C3 * n^2 + C4 * n + C5

theorem sequence_formula
  (a : ℕ → ℂ)
  (h_initial : a 1 = 1 ∧ a 2 = 1 ∧ a 3 = 1 ∧ a 4 = -1 ∧ a 5 = 0)
  (h_recurrence : ∀ n, a (n + 5) = 3 * a (n + 4) - 4 * a (n + 3) + 4 * a (n + 2) - 3 * a (n + 1) + a n) :
  ∀ n, a n = a_seq n :=
by
  sorry

end sequence_formula_l267_267734


namespace ratio_of_AM_to_AD_l267_267920

theorem ratio_of_AM_to_AD (A B C D K M P Q : Point) (AD BK AC QP AB CD : Line)
  (λ : ℝ) (hABCD_trapezoid : trapezoid A B C D) 
  (hK_on_AD : K ∈ AD) 
  (hAK_ratio : |AK| = λ * |AD|) 
  (hP_intersection : P = line_intersection BK AC) 
  (hQ_intersection : Q = line_intersection AB CD)
  (hM_intersection : M = line_intersection QP AD) :
  |AM| / |AD| = λ / (λ + 1) := 
sorry

end ratio_of_AM_to_AD_l267_267920


namespace area_covered_three_layers_l267_267409

noncomputable def auditorium_width : ℕ := 10
noncomputable def auditorium_height : ℕ := 10

noncomputable def first_rug_width : ℕ := 6
noncomputable def first_rug_height : ℕ := 8
noncomputable def second_rug_width : ℕ := 6
noncomputable def second_rug_height : ℕ := 6
noncomputable def third_rug_width : ℕ := 5
noncomputable def third_rug_height : ℕ := 7

-- Prove that the area of part of the auditorium covered with rugs in three layers is 6 square meters.
theorem area_covered_three_layers : 
  let horizontal_overlap_second_third := 5
  let vertical_overlap_second_third := 3
  let area_overlap_second_third := horizontal_overlap_second_third * vertical_overlap_second_third
  let horizontal_overlap_all := 3
  let vertical_overlap_all := 2
  let area_overlap_all := horizontal_overlap_all * vertical_overlap_all
  area_overlap_all = 6 := 
by
  sorry

end area_covered_three_layers_l267_267409


namespace time_to_read_18_pages_l267_267627

-- Definitions based on the conditions
def reading_rate : ℚ := 2 / 4 -- Amalia reads 4 pages in 2 minutes
def pages_to_read : ℕ := 18 -- Number of pages Amalia needs to read

-- Goal: Total time required to read 18 pages
theorem time_to_read_18_pages (r : ℚ := reading_rate) (p : ℕ := pages_to_read) :
  p * r = 9 := by
  sorry

end time_to_read_18_pages_l267_267627


namespace largest_divisor_of_product_of_five_consecutive_integers_l267_267474

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∀ (n : ℤ), ∃ (d : ℤ), d = 60 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end largest_divisor_of_product_of_five_consecutive_integers_l267_267474


namespace tan_A_div_tan_B_l267_267759

-- Define the given conditions
variables {V : Type*} [inner_product_space ℝ V] 
variables {A B C : V}
def condition_1 (CA CB AB : V) : Prop := 
  (CA + CB) ⬝ AB = (3 / 5) * ∥AB∥^2

-- Define the main theorem statement
theorem tan_A_div_tan_B (CA CB AB : V) (h : condition_1 CA CB AB) : 
  tan (angle A B C) / tan (angle B A C) = 4 := 
sorry

end tan_A_div_tan_B_l267_267759


namespace shortest_path_length_l267_267777

theorem shortest_path_length :
  let p1 := (0: ℝ, 1: ℝ, 2: ℝ),
      p2 := (22: ℝ, 4: ℝ, 2: ℝ),
      x_bound := 22: ℝ,
      y_bound := 5: ℝ,
      z_bound := 4: ℝ,
      coord_plane1 := (x: ℝ) = 0,
      coord_plane2 := (y: ℝ) = 0,
      coord_plane3 := (z: ℝ) = 0
  in 
  ∃ path_length: ℝ, path_length = Real.sqrt(657: ℝ) := 
by
  sorry

end shortest_path_length_l267_267777


namespace largest_divisor_of_5_consecutive_integers_l267_267482

theorem largest_divisor_of_5_consecutive_integers :
  ∀ (a b c d e : ℤ), 
    a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e →
    (∃ k : ℤ, k ∣ (a * b * c * d * e) ∧ k = 60) :=
by 
  intro a b c d e h
  sorry

end largest_divisor_of_5_consecutive_integers_l267_267482


namespace log_uncomputable_without_tables_l267_267280

theorem log_uncomputable_without_tables:
  (log 8 ≈ 0.9031) → (log 9 ≈ 0.9542) → 
  (¬ ∃ c : ℝ, log 17 = c) :=
by 
  intros h1 h2
  sorry

end log_uncomputable_without_tables_l267_267280


namespace weight_of_person_replaced_is_65_l267_267293

noncomputable def weight_of_person_replaced (W : ℝ) : ℝ :=
  let avg_old := W / 9
  let avg_new := avg_old + 2.5
  let new_person_weight := 87.5
  let total_weight_new := W - weight_of_person_replaced W + new_person_weight
  let avg_new_calc := total_weight_new / 9 
  weight_of_person_replaced W = new_person_weight - 22.5

theorem weight_of_person_replaced_is_65 (W : ℝ) :
  let weight_replaced := 65 in
  (let avg_old := W / 9 in
  let avg_new := avg_old + 2.5 in
  let new_person_weight := 87.5 in
  let total_weight_new := W - weight_replaced + new_person_weight in
  let avg_new_calc := total_weight_new / 9 in
  avg_new_calc = avg_new) → weight_replaced = 65 :=
by
  sorry

end weight_of_person_replaced_is_65_l267_267293


namespace solve_for_x_l267_267007

theorem solve_for_x:
  ∃ x : ℚ, (2 / 3 - 1 / 4 = 1 / x) ∧ (x = 12 / 5) := by
  sorry

end solve_for_x_l267_267007


namespace find_possible_a_l267_267655

def A : Set ℝ := {x | x^2 - 2*x - 8 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + a*x + a^2 - 12 = 0}

theorem find_possible_a :
  (∀ a, (B a ⊆ A) → (a ∈ (-∞, -4) ∪ {-2} ∪ [4, ∞)))

end find_possible_a_l267_267655


namespace rectangle_length_DG_l267_267360

theorem rectangle_length_DG (a b : ℕ) (S : ℕ) (h1 : S = 29 * (a + b)) (h2 : S % a = 0) (h3 : S % b = 0) : 
  let k := S / a in 
  let l := S / b in 
  k = 870 :=
by
  sorry

end rectangle_length_DG_l267_267360


namespace largest_divisor_of_5_consecutive_integers_l267_267530

theorem largest_divisor_of_5_consecutive_integers :
  ∀ (n : ℤ), ∃ d, d = 120 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intro n
  use 120
  split
  exact rfl
  sorry

end largest_divisor_of_5_consecutive_integers_l267_267530


namespace eight_term_sum_l267_267814

variable {α : Type*} [Field α]
variable (a q : α)

-- Define the n-th sum of the geometric sequence
def S_n (n : ℕ) : α := a * (1 - q ^ n) / (1 - q)

-- Given conditions
def S4 : α := S_n 4 = -5
def S6 : α := S_n 6 = 21 * S_n 2

-- Prove the target statement
theorem eight_term_sum : S_n 8 = -85 :=
  sorry

end eight_term_sum_l267_267814


namespace num_pos_pairs_7_to_x_minus_3_times_2_to_y_eq_1_l267_267971

/-- 
  The number of positive integer pairs (x, y) that satisfy the equation 
  7^x - 3 * 2^y = 1 is exactly 2. 
--/

theorem num_pos_pairs_7_to_x_minus_3_times_2_to_y_eq_1 :
  {p : ℕ × ℕ // 0 < p.1 ∧ 0 < p.2 ∧ 7^p.1 - 3 * 2^p.2 = 1}.card = 2 :=
sorry

end num_pos_pairs_7_to_x_minus_3_times_2_to_y_eq_1_l267_267971


namespace circumference_of_tank_A_l267_267375

noncomputable def circumference_tank_A (r_A : ℝ) : ℝ := 2 * real.pi * r_A
noncomputable def radius_tank_B (C_B : ℝ) : ℝ := C_B / (2 * real.pi)
noncomputable def radius_tank_A (r_B : ℝ) : ℝ := real.sqrt(0.6400000000000001) * r_B

theorem circumference_of_tank_A :
  ∀ (h_A h_B : ℝ) (C_B : ℝ), 
    h_A = 8 → 
    h_B = 8 → 
    C_B = 10 → 
    circumference_tank_A (radius_tank_A (radius_tank_B C_B)) = 8 :=
  by
    intros
    unfold circumference_tank_A radius_tank_A radius_tank_B
    sorry

end circumference_of_tank_A_l267_267375


namespace average_visitors_proof_statement_l267_267585
-- Import requisite libraries

-- Define the conditions
def average_visitors_on_sunday : ℕ := 510
def average_visitors_per_day : ℕ := 285
def days_in_month : ℕ := 30
def sundays_in_month : ℕ := 5

-- Define the target variable for average visitors on other days
def average_visitors_other_days : ℕ := 240

-- Construct the Lean statement to prove the relation
theorem average_visitors_proof_statement :
  let total_visitors_in_month := average_visitors_per_day * days_in_month in
  let total_visitors_on_sundays := sundays_in_month * average_visitors_on_sunday in
  (total_visitors_on_sundays + (days_in_month - sundays_in_month) * average_visitors_other_days) = total_visitors_in_month :=
by
  sorry

end average_visitors_proof_statement_l267_267585


namespace closest_point_proof_l267_267686

noncomputable def closest_point_on_line_to_point 
  (a b : ℝ) (line_slope line_intercept : ℝ) (given_x given_y closest_x closest_y : ℝ) : Prop :=
  let line := λ x, line_slope * x + line_intercept
  ∧ given_y = line given_x
  ∧ closest_y = line closest_x
  ∧ closest_x = (given_x + line_slope * (given_y - line_intercept)) / (1 + (line_slope ^ 2))
  ∧ closest_y = line (closest_x)

theorem closest_point_proof : 
  closest_point_on_line_to_point 1 4 3 (-1) 1 4 (-3/5) (-4/5) :=
by
  sorry

end closest_point_proof_l267_267686


namespace product_of_five_consecutive_integers_divisible_by_120_l267_267454

theorem product_of_five_consecutive_integers_divisible_by_120 (n : ℤ) : 
  120 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end product_of_five_consecutive_integers_divisible_by_120_l267_267454


namespace quadrilateral_ABCD_x_y_sum_l267_267768

noncomputable def x_y_sum_in_kite (BC CD AD : ℝ) (angle_A angle_B : ℝ) (x y : ℕ) : ℕ :=
  if BC = 10 ∧ CD = 14 ∧ AD = 12 ∧ angle_A = 60 ∧ angle_B = 60 ∧
    let AB := x + real.sqrt y in
    x = 2 ∧ y = 109
  then x + y
  else 0

theorem quadrilateral_ABCD_x_y_sum :
  x_y_sum_in_kite 10 14 12 60 60 2 109 = 111 :=
by
  sorry

end quadrilateral_ABCD_x_y_sum_l267_267768


namespace arithmetic_sequence_term_sum_l267_267244

noncomputable def a (n : ℕ) (a₁ d : ℝ) : ℝ := a₁ + (n - 1) * d

noncomputable def S (n : ℕ) (a₁ d : ℝ) : ℝ := n * a₁ + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_term_sum 
  (a₁ d : ℝ)
  (h1 : a 4 a₁ d = 4)
  (h2 : S 5 a₁ d = 15) :
  (∑ n in finset.range 2018, 1 / (a n a₁ d * a (n + 1) a₁ d)) = 2018 / 2019 :=
begin
  sorry
end

end arithmetic_sequence_term_sum_l267_267244


namespace range_of_m_l267_267257

theorem range_of_m (m : ℝ) : 
  (∀ P : ℝ × ℝ, P ∈ {p | ∃ x y: ℝ, p = (x, y) ∧ 3*x - 4*y + m = 0} →
    let M := (-1,0 : ℝ × ℝ) 
        N := (1, 0 : ℝ × ℝ) 
        PM := (P.1 + 1, P.2 : ℝ × ℝ)
        PN := (P.1 - 1, P.2 : ℝ × ℝ)
    in PM.1 * PN.1 + PM.2 * PN.2 = 0) →
  -5 ≤ m ∧ m ≤ 5 :=
by
  intro h
  let dist := |m| / (Real.sqrt (3^2 + 4^2))
  have key_ineq : dist ≤ 1, from sorry
  have key_ineq_solved : |m| / 5 ≤ 1, by simp [Real.sqrt_add];
  have key_bound := abs_le.mp key_ineq_solved
  exact ⟨key_bound.1, key_bound.2⟩

end range_of_m_l267_267257


namespace largest_divisor_of_5_consecutive_integers_l267_267476

theorem largest_divisor_of_5_consecutive_integers :
  ∀ (a b c d e : ℤ), 
    a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e →
    (∃ k : ℤ, k ∣ (a * b * c * d * e) ∧ k = 60) :=
by 
  intro a b c d e h
  sorry

end largest_divisor_of_5_consecutive_integers_l267_267476


namespace log_equivalence_l267_267746

theorem log_equivalence (y : ℝ) (h : log 8 (5 * y) = 3) : log y 64 = 6 / (log 2 102.4) :=
by
  sorry

end log_equivalence_l267_267746


namespace g_range_l267_267335

noncomputable def g (x y θ : ℝ) : ℝ :=
  x / (x + y) + y / (y + Real.cos θ ^ 2) + Real.cos θ ^ 2 / (Real.cos θ ^ 2 + x)

theorem g_range (x y θ : ℝ) (h₁ : x > 0) (h₂ : y > 0) : 
  Set.Ioo (0 : ℝ) 3 = { v | ∃ x y θ, v = g x y θ ∧ x > 0 ∧ y > 0 ∧ (Real.cos θ)^2 ∈ (0, 1] } := 
sorry

end g_range_l267_267335


namespace geometric_sequence_sum_eight_l267_267828

noncomputable def sum_geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ :=
  a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_eight {a1 q : ℝ} (hq : q ≠ 1) 
  (h4 : sum_geometric_sequence a1 q 4 = -5) 
  (h6 : sum_geometric_sequence a1 q 6 = 21 * sum_geometric_sequence a1 q 2) : 
  sum_geometric_sequence a1 q 8 = -85 :=
sorry

end geometric_sequence_sum_eight_l267_267828


namespace common_area_of_rectangle_and_circle_l267_267130

theorem common_area_of_rectangle_and_circle (r : ℝ) (a b : ℝ) (h_center : r = 5) (h_dim : a = 10 ∧ b = 4) :
  let sector_area := (25 * Real.pi) / 2 
  let triangle_area := 4 * Real.sqrt 21 
  let result := sector_area + triangle_area 
  result = (25 * Real.pi) / 2 + 4 * Real.sqrt 21 := 
by
  sorry

end common_area_of_rectangle_and_circle_l267_267130


namespace evaluate_expression_l267_267118

def a (i : ℕ) : ℕ :=
  if h : 1 ≤ i ∧ i ≤ 7 then 2 * i
  else a 1 * a 2 * a 3 * a 4 * a 5 * a 6 * a 7 * (a(l) - 1) for all i > 7 .

theorem evaluate_expression : a 1 * a 2 * a 3 * a 4 * a 5 * a 6 * a 7 * (∏ i, a i) - ∑ i in finset.range 1008, a i ^ 2 = 643560 :=
sorry

end evaluate_expression_l267_267118


namespace intersection_points_1_or_0_l267_267252

noncomputable def intersection_points (f : ℝ → ℝ) (a b : ℝ) :=
  {p : ℝ × ℝ | p.1 = 2 ∧ p.2 = f p.1 ∧ p.1 ∈ set.Icc a b}.finite ∧ 
  (finite.to_finset {p : ℝ × ℝ | p.1 = 2 ∧ p.2 = f p.1 ∧ p.1 ∈ set.Icc a b}).card ≤ 1

theorem intersection_points_1_or_0 (f : ℝ → ℝ) (a b : ℝ) :
  (a ≤ 2 ∧ 2 ≤ b → (finite.to_finset {p : ℝ × ℝ | p.1 = 2 ∧ p.2 = f p.1 ∧ p.1 ∈ set.Icc a b}).card = 1) ∧
  ((2 < a ∨ b < 2) → (finite.to_finset {p : ℝ × ℝ | p.1 = 2 ∧ p.2 = f p.1 ∧ p.1 ∈ set.Icc a b}).card = 0) :=
by sorry

end intersection_points_1_or_0_l267_267252


namespace largest_among_a_b_c_d_l267_267331

noncomputable def a : ℝ := Real.sin (Real.cos (2015 * Real.pi / 180))
noncomputable def b : ℝ := Real.sin (Real.sin (2015 * Real.pi / 180))
noncomputable def c : ℝ := Real.cos (Real.sin (2015 * Real.pi / 180))
noncomputable def d : ℝ := Real.cos (Real.cos (2015 * Real.pi / 180))

theorem largest_among_a_b_c_d : c = max a (max b (max c d)) := by
  sorry

end largest_among_a_b_c_d_l267_267331


namespace solve_system_of_equations_l267_267949

theorem solve_system_of_equations (x y : ℝ) :
  (1 / 2 * x - 3 / 2 * y = -1) ∧ (2 * x + y = 3) → 
  (x = 1) ∧ (y = 1) :=
by
  sorry

end solve_system_of_equations_l267_267949


namespace prove_ax5_by5_l267_267747

variables {a b x y : ℝ}

theorem prove_ax5_by5 (h1 : a * x + b * y = 5)
                      (h2 : a * x^2 + b * y^2 = 11)
                      (h3 : a * x^3 + b * y^3 = 30)
                      (h4 : a * x^4 + b * y^4 = 85) :
  a * x^5 + b * y^5 = 7025 / 29 :=
sorry

end prove_ax5_by5_l267_267747


namespace carpet_needed_l267_267111

/-- A rectangular room with dimensions 15 feet by 9 feet has a non-carpeted area occupied by 
a table with dimensions 3 feet by 2 feet. We want to prove that the number of square yards 
of carpet needed to cover the rest of the floor is 15. -/
theorem carpet_needed
  (room_length : ℝ) (room_width : ℝ) (table_length : ℝ) (table_width : ℝ)
  (h_room : room_length = 15) (h_room_width : room_width = 9)
  (h_table : table_length = 3) (h_table_width : table_width = 2) : 
  (⌈(((room_length * room_width) - (table_length * table_width)) / 9 : ℝ)⌉ = 15) := 
by
  sorry

end carpet_needed_l267_267111


namespace find_diagonal_AC_l267_267412

variables (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables (AB BC CD DA AC : ℝ)
variables (is_integer : ∀ x : ℝ, ∃ k : ℤ, x = k)

-- Conditions from the problem statement
def quadrilateral_sides : Prop :=
  AB = 9 ∧ BC = 2 ∧ CD = 14 ∧ DA = 5

-- Desired property to prove
def diagonal_AC_is_ten : Prop :=
  AC = 10

-- Final statement
theorem find_diagonal_AC (h : quadrilateral_sides)
  (hinteger : is_integer AC) : diagonal_AC_is_ten :=
sorry

end find_diagonal_AC_l267_267412


namespace count_three_digit_concave_numbers_l267_267597

def is_concave_number (a b c : ℕ) : Prop :=
  a > b ∧ c > b

theorem count_three_digit_concave_numbers : 
  (∃! n : ℕ, n = 240) := by
  sorry

end count_three_digit_concave_numbers_l267_267597


namespace rectangle_measurement_error_l267_267767

theorem rectangle_measurement_error 
  (L W : ℝ)
  (measured_length : ℝ := 1.05 * L)
  (measured_width : ℝ := 0.96 * W)
  (actual_area : ℝ := L * W)
  (calculated_area : ℝ := measured_length * measured_width)
  (error : ℝ := calculated_area - actual_area) :
  ((error / actual_area) * 100) = 0.8 :=
sorry

end rectangle_measurement_error_l267_267767


namespace largest_divisor_of_product_of_five_consecutive_integers_l267_267500

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∀ (n : ℤ), ∃ k : ℤ, k = 60 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intro n
  use 60
  split
  { refl }
  { sorry }

end largest_divisor_of_product_of_five_consecutive_integers_l267_267500


namespace vector_addition_l267_267737

def vector : Type := (ℝ × ℝ)

def a : vector := (1, -1)
def b : vector := (-1, 1)
def c : vector := (5, 1)

theorem vector_addition :
  c + (a + b) = c := 
  sorry

end vector_addition_l267_267737


namespace tom_and_jerry_same_speed_l267_267438

theorem tom_and_jerry_same_speed (x : ℝ) (h1 : speed_tom = x^2 - 14x - 48)
                                 (h2 : speed_jerry = (x^2 - 5x - 84) / (x + 8))
                                 (h3 : speed_tom = speed_jerry) :
                                 speed_tom = 6 :=
by
  sorry

end tom_and_jerry_same_speed_l267_267438


namespace solve_for_x_l267_267008

theorem solve_for_x:
  ∃ x : ℚ, (2 / 3 - 1 / 4 = 1 / x) ∧ (x = 12 / 5) := by
  sorry

end solve_for_x_l267_267008


namespace geometric_sequence_S8_l267_267869

theorem geometric_sequence_S8 
  (a1 q : ℝ) (S : ℕ → ℝ)
  (h1 : S 4 = a1 * (1 - q^4) / (1 - q) = -5)
  (h2 : S 6 = 21 * S 2)
  (geom_sum : ∀ n, S n = a1 * (1 - q^n) / (1 - q))
  (h3 : q ≠ 1) : S 8 = -85 := 
begin
  sorry
end

end geometric_sequence_S8_l267_267869


namespace propositions_validity_l267_267728

-- Definitions of the propositions
def prop1 : Prop := ∀ (L1 L2 : Type) (pl : Type) [HasParallel L1 pl] [HasParallel L2 pl], is_parallel L1 L2
def prop2 : Prop := ∀ (L1 L2 : Type) (pl : Type) [HasPerpendicular L1 pl] [HasPerpendicular L2 pl], is_parallel L1 L2
def prop3 : Prop := ∀ (L : Type) (pl : Type) [HasParallel L pl], ∀ (L2 : Type) [HasInPlane L2 pl], is_parallel L L2
def prop4 : Prop := ∀ (L : Type) (pl : Type) [HasPerpendicular L pl], ∀ (L2 : Type) [HasInPlane L2 pl], is_perpendicular L L2

-- The actual proof problem
theorem propositions_validity : prop1 = false ∧ prop2 = true ∧ prop3 = false ∧ prop4 = true :=
by
  sorry

end propositions_validity_l267_267728


namespace geometric_sequence_sum_eight_l267_267825

noncomputable def sum_geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ :=
  a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_eight {a1 q : ℝ} (hq : q ≠ 1) 
  (h4 : sum_geometric_sequence a1 q 4 = -5) 
  (h6 : sum_geometric_sequence a1 q 6 = 21 * sum_geometric_sequence a1 q 2) : 
  sum_geometric_sequence a1 q 8 = -85 :=
sorry

end geometric_sequence_sum_eight_l267_267825


namespace factorization_of_x10_minus_1024_l267_267154

theorem factorization_of_x10_minus_1024 (x : ℝ) :
  x^10 - 1024 = (x^5 + 32) * (x - 2) * (x^4 + 2 * x^3 + 4 * x^2 + 8 * x + 16) :=
by sorry

end factorization_of_x10_minus_1024_l267_267154


namespace cyclic_quadrilateral_area_roots_l267_267084

noncomputable def cyclic_quadrilateral_area (a b c d : ℝ) :=
  let s := (a + b + c + d) / 2
  in Real.sqrt ((s - a) * (s - b) * (s - c) * (s - d))

theorem cyclic_quadrilateral_area_roots :
  let f := λ x : ℝ, x^4 - 10*x^3 + 34*x^2 - 45*x + 19
  ∃ a b c d : ℝ, 
    (f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0) ∧ 
    (a + b + c + d = 10) ∧ 
    cyclic_quadrilateral_area a b c d = Real.sqrt 19 := by
sorry

end cyclic_quadrilateral_area_roots_l267_267084


namespace sum_of_eight_l267_267879

variable (a₁ : ℕ) (q : ℕ)
variable (S : ℕ → ℕ) -- Assume S is a function from natural numbers to natural numbers representing S_n

-- Condition 1: Sum of first 4 terms equals -5
axiom h1 : S 4 = -5

-- Condition 2: Sum of the first 6 terms is 21 times the sum of the first 2 terms
axiom h2 : S 6 = 21 * S 2

-- The formula for the sum of the first n terms of a geometric sequence
def sum_of_first_n_terms (n : ℕ) := a₁ * (1 - q^n) / (1 - q)

-- Statement to be proved
theorem sum_of_eight : S 8 = -85 := sorry

end sum_of_eight_l267_267879


namespace angle_equivalence_l267_267631

theorem angle_equivalence (k : ℤ) : ∃ k : ℤ, ∀ θ : ℤ, θ = -463 → θ + 720 * k = 257 + 360 * k ∧ θ ≡ 257 [MOD 360] :=
by {
  intro θ,
  intro h,
  use (k),
  split,
  {
    rw h,
    ring,
  },
  sorry
}

end angle_equivalence_l267_267631


namespace jenny_kenny_see_each_other_l267_267798

-- Definitions of conditions
def kenny_speed : ℝ := 4
def jenny_speed : ℝ := 2
def paths_distance : ℝ := 300
def radius_building : ℝ := 75
def start_distance : ℝ := 300

-- Theorem statement
theorem jenny_kenny_see_each_other : ∃ t : ℝ, (t = 120) :=
by
  sorry

end jenny_kenny_see_each_other_l267_267798


namespace sufficient_but_not_necessary_condition_l267_267080

-- Define the power function with the given exponent
def power_function (x : ℝ) (m : ℝ) : ℝ := x^(m^2 - 2*m - 1)

-- Define the condition for the function to be monotonically decreasing on (0, +∞)
def monotonically_decreasing_on_pos (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 < x → x < y → f y ≤ f x

-- Define the main theorem
theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (m = 1 → monotonically_decreasing_on_pos (λ x, power_function x m)) ∧
  ¬(monotonically_decreasing_on_pos (λ x, power_function x m) → m = 1) :=
sorry

end sufficient_but_not_necessary_condition_l267_267080


namespace quadratic_real_roots_l267_267205

theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, (k + 1) * x^2 - 2 * x + 1 = 0) → (k ≤ 0 ∧ k ≠ -1) :=
by
  sorry

end quadratic_real_roots_l267_267205


namespace geometric_sequence_sum_eight_l267_267832

noncomputable def sum_geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ :=
  a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_eight {a1 q : ℝ} (hq : q ≠ 1) 
  (h4 : sum_geometric_sequence a1 q 4 = -5) 
  (h6 : sum_geometric_sequence a1 q 6 = 21 * sum_geometric_sequence a1 q 2) : 
  sum_geometric_sequence a1 q 8 = -85 :=
sorry

end geometric_sequence_sum_eight_l267_267832


namespace count_integer_values_l267_267424

theorem count_integer_values (x : ℕ) : 5 < Real.sqrt x ∧ Real.sqrt x < 6 → 
  (finset.card (finset.filter (λ n, 5 < Real.sqrt n ∧ Real.sqrt n < 6) (finset.range 37))) = 10 :=
by
  sorry

end count_integer_values_l267_267424


namespace factorization_l267_267158

theorem factorization (x : ℝ) : x^10 - 1024 = (x^5 + 32) * (x^5 - 32) := 
by 
  sorry

end factorization_l267_267158


namespace find_m_l267_267229

variables {α : Type*} [linear_ordered_field α]

-- Given conditions
variables (a : α) (cos_alpha : α) (sin_alpha : α) (tan_alpha : α)
hypothesis h_cos_alpha_neq_zero : cos_alpha ≠ 0
hypothesis h_tan_eq_sin_div_cos : tan_alpha = sin_alpha / cos_alpha
hypothesis h_roots : (1 / cos_alpha) + tan_alpha = -3

-- To prove
theorem find_m (h1 : 1 / cos_alpha = a)
               (h2 : a + tan_alpha = -3)
               (h3 : cos_alpha ≠ 0) 
               (h4 : m = (sin_alpha / (1 - sin_alpha^2))) : m = 20 / 9 := 
sorry

end find_m_l267_267229


namespace triangle_side_AC_value_l267_267766

theorem triangle_side_AC_value
  (AB BC : ℝ) (AC : ℕ)
  (hAB : AB = 1)
  (hBC : BC = 2007)
  (hAC_int : ∃ (n : ℕ), AC = n) :
  AC = 2007 :=
by
  sorry

end triangle_side_AC_value_l267_267766


namespace largest_divisor_of_5_consecutive_integers_l267_267481

theorem largest_divisor_of_5_consecutive_integers :
  ∀ (a b c d e : ℤ), 
    a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e →
    (∃ k : ℤ, k ∣ (a * b * c * d * e) ∧ k = 60) :=
by 
  intro a b c d e h
  sorry

end largest_divisor_of_5_consecutive_integers_l267_267481


namespace right_triangle_hypotenuse_l267_267112

noncomputable def triangle_hypotenuse (a b c : ℝ) : Prop :=
(a + b + c = 40) ∧
(a * b = 48) ∧
(a^2 + b^2 = c^2) ∧
(c = 18.8)

theorem right_triangle_hypotenuse :
  ∃ (a b c : ℝ), triangle_hypotenuse a b c :=
by
  sorry

end right_triangle_hypotenuse_l267_267112


namespace sum_lengths_DE_EF_equals_9_l267_267018

variable (AB BC FA : ℝ)
variable (area_ABCDEF : ℝ)
variable (DE EF : ℝ)

theorem sum_lengths_DE_EF_equals_9 (h1 : area_ABCDEF = 52) (h2 : AB = 8) (h3 : BC = 9) (h4 : FA = 5)
  (h5 : AB * BC - area_ABCDEF = DE * EF) (h6 : BC - FA = DE) : DE + EF = 9 := 
by 
  sorry

end sum_lengths_DE_EF_equals_9_l267_267018


namespace scientific_notation_of_448000_l267_267979

theorem scientific_notation_of_448000 :
  448000 = 4.48 * 10^5 :=
by 
  sorry

end scientific_notation_of_448000_l267_267979


namespace geometric_sequence_sum_l267_267885

variable (a1 q : ℝ) -- Define the first term and common ratio as real numbers

-- Define the sum of the first n terms of a geometric sequence
def S (n : ℕ) : ℝ := a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum :
  (S 4 a1 q = -5) → (S 6 a1 q = 21 * S 2 a1 q) → (S 8 a1 q = -85) :=
by
  intro h1 h2
  sorry

end geometric_sequence_sum_l267_267885


namespace find_f_5_l267_267696

def f : ℤ → ℤ :=
  λ x, if x ≥ 6 then x - 5 else f (x + 2)

theorem find_f_5 : f 5 = 2 :=
by
  sorry

end find_f_5_l267_267696


namespace simplify_polynomial_l267_267366

def p (x : ℝ) : ℝ := 3 * x^5 - x^4 + 2 * x^3 + 5 * x^2 - 3 * x + 7
def q (x : ℝ) : ℝ := -x^5 + 4 * x^4 + x^3 - 6 * x^2 + 5 * x - 4
def r (x : ℝ) : ℝ := 2 * x^5 - 3 * x^4 + 4 * x^3 - x^2 - x + 2

theorem simplify_polynomial (x : ℝ) :
  (p x) + (q x) - (r x) = 6 * x^4 - x^3 + 3 * x + 1 :=
by sorry

end simplify_polynomial_l267_267366


namespace triangle_area_l267_267194

def area (A B C : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (1 / 2) * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

theorem triangle_area :
  let A := (-2, 3)
  let B := (8, -1)
  let C := (10, 6)
  area A B C = 39 :=
by
  sorry

end triangle_area_l267_267194


namespace min_y_value_l267_267173

open Real

noncomputable def f (x : ℝ) : ℝ := (x^2 + 7*x + 10) / (x + 1)

theorem min_y_value : ∀ x > -1, f x ≥ 9 :=
by sorry

end min_y_value_l267_267173


namespace sufficient_but_not_necessary_l267_267698

-- Definitions and conditions
variable {a b : ℝ}
variable (h : a < b)

def f (x : ℝ) : ℝ := Real.sin x
def g (x : ℝ) : ℝ := Real.cos x

-- Propositions p and q
def p : Prop := f a * f b < 0
def q : Prop := ∃ c ∈ Ioo a b, IsLocalExtremum (λ x, g x) c

-- The statement to be proved
theorem sufficient_but_not_necessary (h : a < b) (hp : p) : q ∧ ¬(q → p) :=
by 
    sorry

end sufficient_but_not_necessary_l267_267698


namespace product_of_5_consecutive_integers_divisible_by_60_l267_267538

theorem product_of_5_consecutive_integers_divisible_by_60 :
  ∀a : ℤ, 60 ∣ (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
by
  sorry

end product_of_5_consecutive_integers_divisible_by_60_l267_267538


namespace geometric_sequence_sum_eight_l267_267829

noncomputable def sum_geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ :=
  a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_eight {a1 q : ℝ} (hq : q ≠ 1) 
  (h4 : sum_geometric_sequence a1 q 4 = -5) 
  (h6 : sum_geometric_sequence a1 q 6 = 21 * sum_geometric_sequence a1 q 2) : 
  sum_geometric_sequence a1 q 8 = -85 :=
sorry

end geometric_sequence_sum_eight_l267_267829


namespace regular_polygon_sides_l267_267605

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ n → ∃ p, ∠(polygon_interior_angle p) = 144) : n = 10 := 
by
  sorry

end regular_polygon_sides_l267_267605


namespace simplify_expression_l267_267165

variable (x : ℝ)

theorem simplify_expression (x : ℝ) : 
  (3 * x - 1 - 5 * x) / 3 = -(2 / 3) * x - (1 / 3) := 
by
  sorry

end simplify_expression_l267_267165


namespace john_running_speed_rounded_l267_267801

noncomputable def John_running_speed (x : ℝ) : ℝ :=
  1.2 * x

theorem john_running_speed_rounded (x : ℝ) (hx : 0 < x) :
  (∀ x, (30 / (3 * x + 2) + 10 / (1.2 * x) = 17 / 6)) →
  Real.round (John_running_speed x * 100) / 100 = 9.73 := sorry

end john_running_speed_rounded_l267_267801


namespace factorization_of_x10_minus_1024_l267_267153

theorem factorization_of_x10_minus_1024 (x : ℝ) :
  x^10 - 1024 = (x^5 + 32) * (x - 2) * (x^4 + 2 * x^3 + 4 * x^2 + 8 * x + 16) :=
by sorry

end factorization_of_x10_minus_1024_l267_267153


namespace cylinder_volume_l267_267751

theorem cylinder_volume :
  ∀ (r : ℝ),
  (h = 2 * r) →
  (2 * π * r * h = π) →
  (π * r^2 * h = π / 4) :=
begin
  intros r h_eq diameter_eq surface_area_eq,
  sorry
end

end cylinder_volume_l267_267751


namespace prove_a2_a3_a4_sum_l267_267213

theorem prove_a2_a3_a4_sum (a1 a2 a3 a4 a5 : ℝ) (h : ∀ x : ℝ, a1 * (x-1)^4 + a2 * (x-1)^3 + a3 * (x-1)^2 + a4 * (x-1) + a5 = x^4) :
  a2 + a3 + a4 = 14 :=
sorry

end prove_a2_a3_a4_sum_l267_267213


namespace five_consecutive_product_div_24_l267_267486

theorem five_consecutive_product_div_24 (n : ℤ) : 
  24 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) := 
sorry

end five_consecutive_product_div_24_l267_267486


namespace hall_length_width_difference_l267_267564

theorem hall_length_width_difference : 
  ∃ L W : ℝ, W = (1 / 2) * L ∧ L * W = 450 ∧ L - W = 15 :=
sorry

end hall_length_width_difference_l267_267564


namespace find_a_for_parallel_and_perpendicular_l267_267708

-- Define the equations of the lines
def line1 (a x y : ℝ) : Prop := (a + 1) * x + 3 * y + 2 = 0
def line2 (x y : ℝ) : Prop := x + 2 * y + 1 = 0

-- Define a function to find the slope of a line given in the form ax + by + c = 0
noncomputable def slope (a b c : ℝ) : ℝ := -a / b

-- Theorem stating the conditions and results
theorem find_a_for_parallel_and_perpendicular (a : ℝ) :
  (∀ x y : ℝ, line1 a x y → line2 x y → slope (a+1) 3 2 = slope 1 2 1 → a = 1 / 2) ∧
  (∀ x y : ℝ, line1 a x y → line2 x y → slope (a+1) 3 2 * slope 1 2 1 = -1 → a = -7) :=
by
  sorry

end find_a_for_parallel_and_perpendicular_l267_267708


namespace product_of_five_consecutive_is_divisible_by_sixty_l267_267522

theorem product_of_five_consecutive_is_divisible_by_sixty (n : ℤ) :
  60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end product_of_five_consecutive_is_divisible_by_sixty_l267_267522


namespace bisecting_lines_upper_bound_l267_267236

variable (n : ℕ) (P : Type) [ConvexPolygon P] (O : Point)

theorem bisecting_lines_upper_bound (h : ConvexPolygon.sides P = n) :
  ∀ k > n, ¬(∃ L : Finset (Line P),
    (∀ l ∈ L, BisectsArea l P O) ∧ 
    L.card = k) := by sorry

end bisecting_lines_upper_bound_l267_267236


namespace sandyPhoneBill_is_340_l267_267934

namespace SandyPhoneBill

variable (sandyAgeNow : ℕ) (kimAgeNow : ℕ) (sandyPhoneBill : ℕ)

-- Conditions
def kimCurrentAge := kimAgeNow = 10
def sandyFutureAge := sandyAgeNow + 2 = 3 * (kimAgeNow + 2)
def sandyPhoneBillDefinition := sandyPhoneBill = 10 * sandyAgeNow

-- Target proof
theorem sandyPhoneBill_is_340 
  (h1 : kimCurrentAge)
  (h2 : sandyFutureAge)
  (h3 : sandyPhoneBillDefinition) :
  sandyPhoneBill = 340 :=
sorry

end SandyPhoneBill

end sandyPhoneBill_is_340_l267_267934


namespace overlap_area_l267_267056

def strip_width := 2
def angle : ℝ := 60 * (Real.pi / 180)
def rect_width := 4
def rect_height := 3

theorem overlap_area (w : ℝ) (a : ℝ) (rw : ℝ) (rh : ℝ) : a = 60 * (Real.pi / 180) → w = 2 → rw = 4 → rh = 3 → 
  (1 / 2) * w * (w / Real.sin a) = (4 * Real.sqrt 3) / 3 := 
by
  rfl -- placeholder for the proof
sorry

end overlap_area_l267_267056


namespace cost_price_equivalence_l267_267107

theorem cost_price_equivalence (list_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) (cost_price : ℝ) :
  list_price = 132 → discount_rate = 0.1 → profit_rate = 0.1 → 
  (list_price * (1 - discount_rate)) = cost_price * (1 + profit_rate) →
  cost_price = 108 :=
by
  intros h1 h2 h3 h4
  sorry

end cost_price_equivalence_l267_267107


namespace infinitely_many_divisible_by_sum_of_digits_l267_267132

def sum_of_digits (n : ℕ) : ℕ :=
  nat.digits 10 n |>.sum

def divisible_by_sum_of_digits (n : ℕ) : Prop :=
  sum_of_digits n > 0 ∧ n % sum_of_digits n = 0

theorem infinitely_many_divisible_by_sum_of_digits :
  ∀ (n : ℕ), ∃ m, m > n ∧ divisible_by_sum_of_digits m :=
begin
  sorry
end

end infinitely_many_divisible_by_sum_of_digits_l267_267132


namespace product_of_5_consecutive_integers_divisible_by_60_l267_267542

theorem product_of_5_consecutive_integers_divisible_by_60 :
  ∀a : ℤ, 60 ∣ (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
by
  sorry

end product_of_5_consecutive_integers_divisible_by_60_l267_267542


namespace geometric_series_solution_l267_267030

noncomputable def geometric_series_sums (b1 q : ℝ) : Prop :=
  (b1 / (1 - q) = 16) ∧ (b1^2 / (1 - q^2) = 153.6) ∧ (|q| < 1)

theorem geometric_series_solution (b1 q : ℝ) (h : geometric_series_sums b1 q) :
  q = 2 / 3 ∧ b1 * q^3 = 32 / 9 :=
by
  sorry

end geometric_series_solution_l267_267030


namespace prime_1002_n_count_l267_267693

theorem prime_1002_n_count :
  ∃! n : ℕ, n ≥ 2 ∧ Prime (n^3 + 2) :=
by
  sorry

end prime_1002_n_count_l267_267693


namespace student_C_has_sweetest_water_l267_267437

-- Define concentrations for each student
def concentration_A : ℚ := 35 / 175 * 100
def concentration_B : ℚ := 45 / 175 * 100
def concentration_C : ℚ := 65 / 225 * 100

-- Prove that Student C has the highest concentration
theorem student_C_has_sweetest_water :
  concentration_C > concentration_B ∧ concentration_C > concentration_A :=
by
  -- By direct calculation from the provided conditions
  sorry

end student_C_has_sweetest_water_l267_267437


namespace problem_l267_267216

theorem problem (f : ℝ → ℝ) (h : ∀ x, f (Real.cos x) = Real.cos (17 * x)) (x : ℝ) :
  f (Real.cos x) ^ 2 + f (Real.sin x) ^ 2 = 1 :=
sorry

end problem_l267_267216


namespace smallest_prime_factor_in_C_l267_267937

def C : Set ℕ := {73, 75, 76, 77, 79}

theorem smallest_prime_factor_in_C : 76 ∈ C ∧ ∀ n ∈ C, (n = 76 ∨ (∀ p : ℕ, prime p → p ∣ n → p ≥ 2)) :=
by
  sorry

end smallest_prime_factor_in_C_l267_267937


namespace clothes_spending_ratio_l267_267374

def initial_earnings : ℝ := 600
def remaining_after_books (c : ℝ) : ℝ := (initial_earnings - c) / 2
def final_amount : ℝ := 150

theorem clothes_spending_ratio (c : ℝ) (h : remaining_after_books c = final_amount) :
  (c / initial_earnings) = 1 / 2 :=
sorry

end clothes_spending_ratio_l267_267374


namespace smallest_solution_of_equation_l267_267061

theorem smallest_solution_of_equation :
  ∃ x : ℝ, (9 * x^2 - 45 * x + 50 = 0) ∧ (∀ y : ℝ, 9 * y^2 - 45 * y + 50 = 0 → x ≤ y) :=
sorry

end smallest_solution_of_equation_l267_267061


namespace sandy_phone_bill_expense_l267_267933

def sandy_age_now (kim_age : ℕ) : ℕ := 3 * (kim_age + 2) - 2

def sandy_phone_bill (sandy_age : ℕ) : ℕ := 10 * sandy_age

theorem sandy_phone_bill_expense
  (kim_age : ℕ)
  (kim_age_condition : kim_age = 10)
  : sandy_phone_bill (sandy_age_now kim_age) = 340 := by
  sorry

end sandy_phone_bill_expense_l267_267933


namespace shorter_road_network_exists_l267_267207

-- Definitions based on conditions
def village_positions : list (ℝ × ℝ) := 
  [(0, 0), (1, 0), (1, 1), (0, 1)]

def diagonal_road_length : ℝ := 2 * real.sqrt 2

-- Lean statement of the equivalent proof problem
theorem shorter_road_network_exists :
  ∃ (network : set (ℝ × ℝ)), 
    (∀ (v ∈ village_positions) (w ∈ village_positions), v ≠ w → (v, w) ∈ network) ∧
    (∑ (d ∈ network), real.sqrt (d.2.1 - d.1.1)^2 + (d.2.2 - d.1.2)^2) < diagonal_road_length :=
sorry

end shorter_road_network_exists_l267_267207


namespace largest_divisor_of_product_of_five_consecutive_integers_l267_267496

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∀ (n : ℤ), ∃ k : ℤ, k = 60 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intro n
  use 60
  split
  { refl }
  { sorry }

end largest_divisor_of_product_of_five_consecutive_integers_l267_267496


namespace percentage_answered_second_question_correctly_l267_267750

open Finset

-- Define our universal set of total students
variable (U : Type) [Fintype U]

-- Define sets A and B
variable (A B : Finset U)

-- Define the conditions
axiom condition1 : (A.card : ℝ) / (card U) = 0.75
axiom condition2 : (A ∩ B).card / (card U) = 0.25
axiom condition3 : (U \ (A ∪ B)).card / (card U) = 0.20

-- Define the theorem to prove the percentage who answered the second question correctly
theorem percentage_answered_second_question_correctly : 
  (B.card : ℝ) / (card U) = 0.30 :=
sorry

end percentage_answered_second_question_correctly_l267_267750


namespace total_students_l267_267038

theorem total_students (boys_2nd:int) (girls_2nd:int) (students_2nd:int) (students_3rd:int) (total_students:int):
  (boys_2nd = 20) ->
  (girls_2nd = 11) ->
  (students_2nd = boys_2nd + girls_2nd) ->
  (students_3rd = 2 * students_2nd) ->
  (total_students = students_2nd + students_3rd) ->
  total_students = 93 := 
begin
  intros h1 h2 h3 h4 h5,
  rw [h1, h2] at h3,
  rw h3 at h4,
  rw [h3, h4] at h5,
  exact h5,
end

end total_students_l267_267038


namespace double_angle_trig_values_l267_267232

variable (α : ℝ)
variable (h : sin α = 3 / 5)
variable (q : π / 2 < α ∧ α < π)

theorem double_angle_trig_values :
  sin 2 * α = -24 / 25 ∧
  cos 2 * α = 7 / 25 ∧
  tan 2 * α = -24 / 7 :=
by
  sorry

end double_angle_trig_values_l267_267232


namespace intersection_at_y_axis_l267_267773

theorem intersection_at_y_axis : ∃ y, (y = 5 * 0 + 1) ∧ (0, y) = (0, 1) :=
begin
  use 1,
  split,
  { norm_num, },
  { refl, },
end

end intersection_at_y_axis_l267_267773


namespace triangle_OPQ_right_isosceles_l267_267568

/--
  Given:
  1. \( O \) is the origin (0,0)
  2. Line \( l \) is perpendicular to line \( OP \)
  3. Point \( P \) has coordinates \((2,1)\)
  4. Line \( l \) intersects the parabola \( y^2 = 2px \) (with \( p > 0 \)) at points \( A \) and \( B \)
  5. \( Q \) is the midpoint of segment \( AB \)
  
  Prove that:
  \( \triangle OPQ \) being a right isosceles triangle results in \( p = 2 \).
-/
theorem triangle_OPQ_right_isosceles (p : ℝ) (h : p > 0) :
  let O := (0 : ℝ, 0 : ℝ),
      P := (2 : ℝ, 1 : ℝ),
      -- l is perpendicular to OP
      slope_OP := (P.2 - O.2) / (P.1 - O.1),
      slope_l := (-1) / slope_OP,
      l : ℝ × ℝ → Prop := λ Q, Q.2 = slope_l * (Q.1 - P.1) + P.2,
      -- Parabola
      parabola : ℝ × ℝ → Prop := λ Q, Q.2^2 = 2 * p * Q.1,
      -- Intersection points A and B
      intersects_parabola_l : ∃ (A B : ℝ × ℝ), l A ∧ parabola A ∧ l B ∧ parabola B ∧ A ≠ B,
      -- Midpoint Q of segment AB
      Q := λ A B, ((A.1 + B.1) / 2, (A.2 + B.2) / 2),
      -- |PQ| = |OP|
      OPQ_right_isosceles := ∀ A B, l A → parabola A → l B → parabola B → A ≠ B → 
                                let Q_mid := Q A B in 
                                (Q_mid.1 - O.1)^2 + (Q_mid.2 - O.2)^2 = (2 - 0)^2 + (1 - 0)^2 :=
  p = 2 := by
  sorry

end triangle_OPQ_right_isosceles_l267_267568


namespace distance_between_parallel_lines_l267_267242

theorem distance_between_parallel_lines :
  ∀ (a b c₁ c₂ : ℝ),
  (a ≠ 0 ∨ b ≠ 0) →
  3 * 3 + 4 * 4 = 3 * 3 + 4 * 4 →
  (a = 3) → (b = 4) → (c₁ = -5) → (c₂ = 7) →
  (abs (c₂ - c₁) / real.sqrt (a * a + b * b) = 2 / 5) :=
begin
  intros a b c1 c2 h1 h2 ha hb hc1 hc2,
  sorry, -- Proof is not required
end

end distance_between_parallel_lines_l267_267242


namespace sum_first_2010_terms_zero_l267_267735

def seq (n : ℕ) : ℤ :=
  if n % 6 = 0 then 2004 else
  if n % 6 = 1 then 2005 else
  if n % 6 = 2 then 1 else
  if n % 6 = 3 then -2004 else
  if n % 6 = 4 then -2005 else -1

theorem sum_first_2010_terms_zero : (∑ i in finset.range 2010, seq i) = 0 :=
  sorry

end sum_first_2010_terms_zero_l267_267735


namespace increasing_function_d_l267_267629

-- Define the given functions
def fA (x : ℝ) : ℝ := 4 - 2 * x
def fB (x : ℝ) : ℝ := 1 / (x - 2)
def fC (x : ℝ) : ℝ := x^2 - 2 * x - 2
def fD (x : ℝ) : ℝ := -|x|

-- Define the condition for the functions to be increasing on (-∞, 0)
def is_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 ∈ I → x2 ∈ I → x1 < x2 → f(x1) < f(x2)

-- The proof statement
theorem increasing_function_d : 
  is_increasing_on fD {x : ℝ | x < 0} ∧ 
  ¬ is_increasing_on fA {x : ℝ | x < 0} ∧
  ¬ is_increasing_on fB {x : ℝ | x < 0} ∧
  ¬ is_increasing_on fC {x : ℝ | x < 0} := by
  sorry

end increasing_function_d_l267_267629


namespace prism_diagonal_and_surface_area_l267_267599

/-- 
  A rectangular prism has dimensions of 12 inches, 16 inches, and 21 inches.
  Prove that the length of the diagonal is 29 inches, 
  and the total surface area of the prism is 1560 square inches.
-/
theorem prism_diagonal_and_surface_area :
  let a := 12
  let b := 16
  let c := 21
  let d := Real.sqrt (a^2 + b^2 + c^2)
  let S := 2 * (a * b + b * c + c * a)
  d = 29 ∧ S = 1560 := by
  let a := 12
  let b := 16
  let c := 21
  let d := Real.sqrt (a^2 + b^2 + c^2)
  let S := 2 * (a * b + b * c + c * a)
  sorry

end prism_diagonal_and_surface_area_l267_267599


namespace complement_of_M_wrt_U_l267_267909

-- Definitions of the sets U and M as given in the problem
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 3, 5}

-- The goal is to show the complement of M w.r.t. U is {2, 4, 6}
theorem complement_of_M_wrt_U :
  (U \ M) = {2, 4, 6} := 
by
  sorry

end complement_of_M_wrt_U_l267_267909


namespace length_CD_constant_l267_267321

variables {O O' A B C D : Point}
variables {Γ Γ' : Circle}

-- Define congruence of the circles and their centers
axiom congruent_circles : Γ.center = O ∧ Γ'.center = O' ∧ (Γ.radius = Γ'.radius)

-- Define the intersection and point properties
axiom intersection_of_circles : 
  ∃ A₁ A₂ : Point, A₁ ≠ A₂ ∧ A₁ ∈ Γ ∧ A₁ ∈ Γ' ∧ A₂ ∈ Γ ∧ A₂ ∈ Γ'

-- Define point B on circle Γ
axiom B_on_Γ : B ∈ Γ

-- Define the intersection of line AB with circle Γ'
axiom intersection_AB_Γ' : ∃ C : Point, C ≠ A ∧ C ∈ Line(A, B) ∧ C ∈ Γ'

-- Define the parallelogram OBDO'
axiom parallelogram_obdo' : is_parallelogram O B D O'

-- Goal: Show that the length of CD does not depend on the position of B
theorem length_CD_constant (congruent_circles : Γ.center = O ∧ Γ'.center = O' ∧ (Γ.radius = Γ'.radius)) 
 (intersection_of_circles : ∃ A₁ A₂ : Point, A₁ ≠ A₂ ∧ A₁ ∈ Γ ∧ A₁ ∈ Γ' ∧ A₂ ∈ Γ ∧ A₂ ∈ Γ') 
 (B_on_Γ : B ∈ Γ) (intersection_AB_Γ' : ∃ C : Point, C ≠ A ∧ C ∈ Line(A, B) ∧ C ∈ Γ') 
 (parallelogram_obdo' : is_parallelogram O B D O'): 
  ∃ d : ℝ, ∀ B₁ : Point, B₁ ∈ Γ → (distance C D = d) :=
sorry

end length_CD_constant_l267_267321


namespace angle_ABM_l267_267792

theorem angle_ABM (A B C D M : Point) (α : ℝ)
  (sqABCD : is_square A B C D)
  (inside_sq : M ∈ interior ABCD)
  (angle_MAC : angle M A C = α)
  (angle_MCD : angle M C D = α)
  : angle A B M = 90 - 2 * α := 
sorry

end angle_ABM_l267_267792


namespace total_canoes_built_l267_267639

-- Definition of the conditions as suggested by the problem
def num_canoes_in_february : Nat := 5
def growth_rate : Nat := 3
def number_of_months : Nat := 5

-- Final statement to prove
theorem total_canoes_built : (num_canoes_in_february * (growth_rate^number_of_months - 1)) / (growth_rate - 1) = 605 := 
by sorry

end total_canoes_built_l267_267639


namespace train_speed_approximation_l267_267124
noncomputable def speed_of_train (length_in_meters : ℝ) (time_in_seconds : ℝ) : ℝ :=
  (length_in_meters / time_in_seconds) * 3.6

theorem train_speed_approximation :
  let length := 100.0
  let time := 4.99960003199744 in
  abs (speed_of_train length time - 72.01) < 0.01 :=
by
  sorry

end train_speed_approximation_l267_267124


namespace series_sum_eq_negative_one_third_l267_267161

noncomputable def series_sum : ℝ :=
  ∑' n, (2 * n + 1) / (n * (n + 1) * (n + 2) * (n + 3))

theorem series_sum_eq_negative_one_third : series_sum = -1 / 3 := sorry

end series_sum_eq_negative_one_third_l267_267161


namespace area_of_enclosing_rectangle_l267_267897

theorem area_of_enclosing_rectangle (a : ℝ) (ha : 0 < a) :
  let X := (0, 0)
      Y := (a, 0)
      Z := (a, a),
      right_triangle := (Y.1 - X.1)^2 + (Z.2 - X.2)^2 = (Z.1 - X.1)^2,
      hypotenuse := real.sqrt((Y.1 - X.1)^2 + (Z.2 - X.2)^2) = 2 * a
  in
    right_triangle ∧ hypotenuse →
    let length := 2 * a,
        width := 2 * a,
        area := length * width
    in
      area = 4 * a^2 :=
by
  sorry

end area_of_enclosing_rectangle_l267_267897


namespace integer_values_satisfying_sqrt_condition_l267_267420

theorem integer_values_satisfying_sqrt_condition :
  {x : ℤ | 5 < Real.sqrt x ∧ Real.sqrt x < 6}.card = 10 :=
by
  sorry

end integer_values_satisfying_sqrt_condition_l267_267420


namespace sum_of_four_digits_l267_267739

theorem sum_of_four_digits (EH OY AY OH : ℕ) (h1 : EH = 4 * OY) (h2 : AY = 4 * OH) : EH + OY + AY + OH = 150 :=
sorry

end sum_of_four_digits_l267_267739


namespace trapezium_area_l267_267683

theorem trapezium_area (a b area h : ℝ) (h1 : a = 20) (h2 : b = 15) (h3 : area = 245) :
  area = 1 / 2 * (a + b) * h → h = 14 :=
by
  sorry

end trapezium_area_l267_267683


namespace largest_divisor_of_product_of_five_consecutive_integers_l267_267469

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∀ (n : ℤ), ∃ (d : ℤ), d = 60 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end largest_divisor_of_product_of_five_consecutive_integers_l267_267469


namespace cube_edge_length_is_3_l267_267164

-- Define the problem conditions as constants or premises
constant base_side_length : ℝ
constant pyramid_height : ℝ
constant centroid_to_apex_length : ℝ
constant cube_edge_length : ℝ

-- Set the specific values for the problem
axiom base_side_length_is_3 : base_side_length = 3
axiom pyramid_height_is_9 : pyramid_height = 9
axiom centroid_to_apex_is_diagonal : centroid_to_apex_length = 9
axiom diagonal_relation : centroid_to_apex_length = cube_edge_length * Real.sqrt 3

-- The theorem stating the conclusion we want to prove
theorem cube_edge_length_is_3 :
  base_side_length = 3 →
  pyramid_height = 9 →
  centroid_to_apex_length = cube_edge_length * Real.sqrt 3 →
  cube_edge_length = 3 :=
by
  sorry

end cube_edge_length_is_3_l267_267164


namespace geometric_sequence_sum_l267_267890

variable (a1 q : ℝ) -- Define the first term and common ratio as real numbers

-- Define the sum of the first n terms of a geometric sequence
def S (n : ℕ) : ℝ := a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum :
  (S 4 a1 q = -5) → (S 6 a1 q = 21 * S 2 a1 q) → (S 8 a1 q = -85) :=
by
  intro h1 h2
  sorry

end geometric_sequence_sum_l267_267890


namespace concurrency_of_AP_BI1_CI2_l267_267328

variables {A B C P I1 I2 : Type*}
variables [EuclideanGeometry A B C P I1 I2]

-- Given conditions
def condition1 (A B C P : Point) : Prop := 
  ∠APB - ∠ACB = ∠APC - ∠ABC

def incenter (X Y Z P : Point) : Point :=
  sorry -- Definition of incenter, which might rely on external geometry constructs

-- Main statement to prove
theorem concurrency_of_AP_BI1_CI2
  (h₁ : condition1 A B C P)
  (I1_def : I1 = incenter A P B)
  (I2_def : I2 = incenter A P C) :
  concurrent AP BI1 CI2 :=
sorry

end concurrency_of_AP_BI1_CI2_l267_267328


namespace completing_square_result_l267_267012

theorem completing_square_result:
  ∀ x : ℝ, (x^2 + 4 * x - 1 = 0) → (x + 2)^2 = 5 :=
by
  sorry

end completing_square_result_l267_267012


namespace matches_divisible_by_2_l267_267990

def tournament_matches := 128

theorem matches_divisible_by_2 :
  let matches := tournament_matches - 1 in -- 127 normal losses + 1 wild card match
  matches = 128 :=
by
  -- This is where we would provide the proof steps, but we'll replace that with "sorry" for now
  sorry

end matches_divisible_by_2_l267_267990


namespace questions_for_second_project_l267_267104

open Nat

theorem questions_for_second_project (days_per_week : ℕ) (first_project_q : ℕ) (questions_per_day : ℕ) 
  (total_questions : ℕ) (second_project_q : ℕ) 
  (h1 : days_per_week = 7)
  (h2 : first_project_q = 518)
  (h3 : questions_per_day = 142)
  (h4 : total_questions = days_per_week * questions_per_day)
  (h5 : second_project_q = total_questions - first_project_q) :
  second_project_q = 476 :=
by
  -- we assume the solution steps as correct
  sorry

end questions_for_second_project_l267_267104


namespace product_of_five_consecutive_is_divisible_by_sixty_l267_267517

theorem product_of_five_consecutive_is_divisible_by_sixty (n : ℤ) :
  60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end product_of_five_consecutive_is_divisible_by_sixty_l267_267517


namespace geometric_sequence_sum_S8_l267_267848

noncomputable def sum_geometric_seq (a q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_S8 (a q : ℝ) (h1 : q ≠ 1)
  (h2 : sum_geometric_seq a q 4 = -5)
  (h3 : sum_geometric_seq a q 6 = 21 * sum_geometric_seq a q 2) :
  sum_geometric_seq a q 8 = -85 :=
sorry

end geometric_sequence_sum_S8_l267_267848


namespace correct_statements_l267_267239

-- Define the function f and its properties
noncomputable def f : ℝ → ℝ := sorry

-- Given conditions
axiom even_f : ∀ x : ℝ, f(x) = f(-x)
axiom symmetry_f : ∀ x : ℝ, f(2 + x) = f(2 - x)
axiom definition_f : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → f(x) = x^2 - x

-- Statements to prove
theorem correct_statements :
  (∀ x, f x ≥ -1/4 ∧ f x ≤ 2) ∧
  (∀ x, f (x + 4) = f x) ∧
  (f 2023 = 0) ∧
  (∑ i in finset.range 2024 \ {0}, f i = 1012)
:= sorry

end correct_statements_l267_267239


namespace geometric_sequence_S8_l267_267864

theorem geometric_sequence_S8 
  (a1 q : ℝ) (S : ℕ → ℝ)
  (h1 : S 4 = a1 * (1 - q^4) / (1 - q) = -5)
  (h2 : S 6 = 21 * S 2)
  (geom_sum : ∀ n, S n = a1 * (1 - q^n) / (1 - q))
  (h3 : q ≠ 1) : S 8 = -85 := 
begin
  sorry
end

end geometric_sequence_S8_l267_267864


namespace ratio_of_blue_marbles_l267_267039

theorem ratio_of_blue_marbles {total_marbles red_marbles orange_marbles blue_marbles : ℕ} 
  (h_total : total_marbles = 24)
  (h_red : red_marbles = 6)
  (h_orange : orange_marbles = 6)
  (h_blue : blue_marbles = total_marbles - red_marbles - orange_marbles) : 
  (blue_marbles : ℚ) / (total_marbles : ℚ) = 1 / 2 := 
by
  sorry -- the proof is omitted as per instructions

end ratio_of_blue_marbles_l267_267039


namespace total_pencils_l267_267072

theorem total_pencils (pencils_per_child : ℕ) (children : ℕ) (h1 : pencils_per_child = 2) (h2 : children = 9) : pencils_per_child * children = 18 :=
sorry

end total_pencils_l267_267072


namespace cost_of_seven_CDs_l267_267439

theorem cost_of_seven_CDs :
  let cost_per_CD := 24 / 2 in
  7 * cost_per_CD = 84 :=
by
  sorry

end cost_of_seven_CDs_l267_267439


namespace max_acute_angles_convex_polygon_2000_sides_l267_267091

theorem max_acute_angles_convex_polygon_2000_sides :
  ∀ (P : Type) [polygon P] (h_convex : convex P) (h_sides : num_sides P = 2000), 
  ∃ (n : ℕ), n ≤ 3 ∧ acute_angles P = n := 
sorry

end max_acute_angles_convex_polygon_2000_sides_l267_267091


namespace opposite_numbers_l267_267279

theorem opposite_numbers (a b : ℝ) (h : a = -b) : b = -a := 
by 
  sorry

end opposite_numbers_l267_267279


namespace birds_in_trees_l267_267400

def number_of_stones := 40
def number_of_trees := number_of_stones + 3 * number_of_stones
def combined_number := number_of_trees + number_of_stones
def number_of_birds := 2 * combined_number

theorem birds_in_trees : number_of_birds = 400 := by
  sorry

end birds_in_trees_l267_267400


namespace log_expression_evaluation_l267_267081

theorem log_expression_evaluation :
  (log 10 25 + (log 10 2)^2 + log 10 2 * log 10 50) = 2 :=
by
  sorry

end log_expression_evaluation_l267_267081


namespace regular_polygon_interior_angle_integer_l267_267334

theorem regular_polygon_interior_angle_integer :
  ∃ l : List ℕ, l.length = 9 ∧ ∀ n ∈ l, 3 ≤ n ∧ n ≤ 15 ∧ (180 * (n - 2)) % n = 0 :=
by
  sorry

end regular_polygon_interior_angle_integer_l267_267334


namespace perfect_square_in_product_l267_267026

/-- 
Given 1986 natural numbers such that their product has exactly 1985 distinct prime divisors, 
prove that either one of these numbers, or the product of several of them, is a perfect square.
-/
theorem perfect_square_in_product
  (a : Fin 1986 → ℕ)
  (h : ∃ p : Fin 1985 → ℕ, (∀ i j, i ≠ j → p i ≠ p j) ∧ Multiset.toFinset (Multiset.bind (Finset.univ.val.map a).map prime_factors) = Multiset.toFinset (Multiset.bind (Finset.range 1985).map p)) :
  ∃ s : Finset (Fin 1986), (∏ i in s, a i) ∣ (∏ i in s, a i)^2 := 
sorry

end perfect_square_in_product_l267_267026


namespace find_CD_l267_267329

-- Define the main condition stated in the problem
variables (A B C D E : Type) [MetricSpace A] [MetricSpace B]

-- Assume necessary geometric properties based on the problem statement
def is_right_triangle (ABC : Triangle A B C) : Prop := 
  ∃ (B : Point), isRightAngle ABC.∠B

def circle_diameter_BC_intersects_AC_at_D (ABC : Triangle A B C) (D : Point) : Prop := 
  ∃ (c : Circle), c.diameter = dist B C ∧ c.intersects_segment (Segment A C)

def distances (A B C D : Point) : Prop :=
  dist A D = 3 ∧ dist B D = 6 ∧
  (∀ E : Point, on_segment A C E ∧ E ≠ D → dist A E = 2)

-- The main theorem we need to prove
theorem find_CD (A B C D : Type) [MetricSpace A] [MetricSpace B] 
  (h1 : is_right_triangle (Triangle A B C))
  (h2 : circle_diameter_BC_intersects_AC_at_D (Triangle A B C) D)
  (h3 : distances A B C D) :
  dist C D = 12 :=
sorry

end find_CD_l267_267329


namespace count_valid_permutations_l267_267299

def is_increasing_triplet (a b c : ℕ) : Prop := a < b ∧ b < c
def is_decreasing_triplet (a b c : ℕ) : Prop := a > b ∧ b > c

def valid_permutation (seq : List ℕ) : Prop :=
  seq.length = 5 ∧ 
  seq.nodup ∧
  ∃ (_, _, _).all (fun i => i + 2 < seq.length ->
    ¬is_increasing_triplet (seq.nth i).getOrElse 0 (seq.nth (i + 1)).getOrElse 0 (seq.nth (i + 2)).getOrElse 0 ∧
    ¬is_decreasing_triplet (seq.nth i).getOrElse 0 (seq.nth (i + 1)).getOrElse 0 (seq.nth (i + 2)).getOrElse 0)

theorem count_valid_permutations : 
  ∃! seqs : Finset (List ℕ), 
    (∀ seq ∈ seqs, valid_permutation seq) ∧ seqs.card = 32 :=
sorry

end count_valid_permutations_l267_267299


namespace number_of_pentagonal_faces_l267_267092

theorem number_of_pentagonal_faces :
  ∀ (n k : ℕ), 
  (n + k : ℕ) ∈ ℕ → 
  (5 * n + 6 * k) % 3 = 0 → 
  (5 * n + 6 * k) % 2 = 0 →
  (3 * n = 6 * k - 6) :=
sorry

end number_of_pentagonal_faces_l267_267092


namespace greatest_value_x_l267_267196

theorem greatest_value_x (x : ℝ) : 
  (x ≠ 9) → 
  (x^2 - 5 * x - 84) / (x - 9) = 4 / (x + 6) →
  x ≤ -2 :=
by
  sorry

end greatest_value_x_l267_267196


namespace scientific_notation_of_448000_l267_267981

theorem scientific_notation_of_448000 :
  448000 = 4.48 * 10^5 :=
by 
  sorry

end scientific_notation_of_448000_l267_267981


namespace cube_shadow_problem_l267_267623

noncomputable def greatest_integer_le_1000x (x : ℝ) : ℤ :=
  if h : ∀ y : ℤ, y < 1000 * x + 1 ∧ y ≥ 1000 * x then y
  else 0 -- placeholder

theorem cube_shadow_problem:
  ∃ (x : ℝ), 
  (∃ (a : ℝ), a = 50) ∧  -- condition: additional shadow area
  (∃ (b : ℝ), b = 2) ∧  -- condition: cube's edge length
  (greatest_integer_le_1000x x = 5347) := 
sorry

end cube_shadow_problem_l267_267623


namespace range_of_f_on_0_3_l267_267333

noncomputable def g (x : ℝ) : ℝ := sorry
def f (x : ℝ) : ℝ := x + g x

theorem range_of_f_on_0_3 :
  (∀ x : ℝ, g (x + 1) = g x) →
  (set.range (λ x, f x) ∩ set.Icc 0 1 = set.Icc (-2 : ℝ) 5) →
  (set.range (λ x, f x) ∩ set.Icc 0 3 = set.Icc (-2 : ℝ) 7) :=
by
  intros h_period h_range_0_1
  sorry

end range_of_f_on_0_3_l267_267333


namespace range_of_a_l267_267277

noncomputable def quadratic (a x : ℝ) : ℝ := a * x^2 - 2 * x + a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, quadratic a x > 0) ↔ 0 < a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l267_267277


namespace largest_divisor_of_5_consecutive_integers_l267_267483

theorem largest_divisor_of_5_consecutive_integers :
  ∀ (a b c d e : ℤ), 
    a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e →
    (∃ k : ℤ, k ∣ (a * b * c * d * e) ∧ k = 60) :=
by 
  intro a b c d e h
  sorry

end largest_divisor_of_5_consecutive_integers_l267_267483


namespace measure_SIDE_XY_l267_267565

noncomputable def triangle_measure (a : ℝ) (b : ℝ) :=
  a = b * ℝ.sqrt(2)

theorem measure_SIDE_XY
  (a b : ℝ)
  (h1 : a > b)
  (h2 : (1 / 2) * b * b = 49)
  : triangle_measure a b → a = 14 :=
by
  sorry

end measure_SIDE_XY_l267_267565


namespace largest_divisor_of_five_consecutive_integers_l267_267459

open Nat

theorem largest_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, k ∈ {n, n+1, n+2, n+3, n+4} ∧
    ∀ m ∈ {2, 3, 4, 5}, m ∣ k → 60 ∣ (n * (n+1) * (n+2) * (n+3) * (n+4)) := 
sorry

end largest_divisor_of_five_consecutive_integers_l267_267459


namespace left_jazz_lovers_count_l267_267764

-- Definitions based on the given conditions
variables (club_size left_hand_count jazz_lovers right_dislike_jazz left_jazz_lovers : ℕ)

-- Given conditions
axiom club_size_def : club_size = 25
axiom left_hand_count_def : left_hand_count = 12
axiom jazz_lovers_def : jazz_lovers = 18
axiom right_dislike_jazz_def : right_dislike_jazz = 3
axiom total_people_classification : ∀ (left_jazz_lovers : ℕ), 
  left_jazz_lovers + (left_hand_count - left_jazz_lovers) + (jazz_lovers - left_jazz_lovers) + right_dislike_jazz = club_size

-- Goal: Find the number of left-handed jazz lovers
theorem left_jazz_lovers_count : ∃ left_jazz_lovers, 
  left_jazz_lovers + (left_hand_count - left_jazz_lovers) + (jazz_lovers - left_jazz_lovers) + right_dislike_jazz = club_size ∧ 
  left_jazz_lovers = 8 :=
begin
  use 8,
  split,
  { apply total_people_classification },
  { refl },
  sorry -- Placeholder for the correctness of classification axiom application
end

end left_jazz_lovers_count_l267_267764


namespace k_eq_1_t_range_a_range_l267_267248

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := (x + 2 - k * x^2) / x^2

-- Part 1: Prove k = 1
theorem k_eq_1 (k : ℝ) :
  (∀ x, x ∈ ((-1 : ℝ), 0) ∪ (0, 2) → f x k > 0) ↔ k = 1 :=
by sorry

-- Part 2: Find the range of t
theorem t_range (t : ℝ) (x : ℝ) :
  (∀ x, x ∈ ((1 / 2 : ℝ), 1) → t - 1 < f x 1) ∧ (∃ x0, x0 ∈ ((-5 : ℝ), 0) ∧ t - 1 < f x0 1) ↔
  (-3/25 : ℝ) ≤ t ∧ t ≤ 3 :=
by sorry

-- Part 3: Find the range of a
theorem a_range (a : ℝ) :
  (∃ x, (0 < x ∧ x < 2) ∧ ln (f x 1) + 2 * ln x = ln (3 - a * x)) ↔
  a = 1 ∨ (a ≥ 3/2) :=
by sorry

end k_eq_1_t_range_a_range_l267_267248


namespace factorization_l267_267159

theorem factorization (x : ℝ) : x^10 - 1024 = (x^5 + 32) * (x^5 - 32) := 
by 
  sorry

end factorization_l267_267159


namespace most_stable_performance_l267_267695

theorem most_stable_performance 
    (S_A S_B S_C S_D : ℝ)
    (h_A : S_A = 0.54) 
    (h_B : S_B = 0.61) 
    (h_C : S_C = 0.7) 
    (h_D : S_D = 0.63) :
    S_A <= S_B ∧ S_A <= S_C ∧ S_A <= S_D :=
by {
  sorry
}

end most_stable_performance_l267_267695


namespace eight_term_sum_l267_267817

variable {α : Type*} [Field α]
variable (a q : α)

-- Define the n-th sum of the geometric sequence
def S_n (n : ℕ) : α := a * (1 - q ^ n) / (1 - q)

-- Given conditions
def S4 : α := S_n 4 = -5
def S6 : α := S_n 6 = 21 * S_n 2

-- Prove the target statement
theorem eight_term_sum : S_n 8 = -85 :=
  sorry

end eight_term_sum_l267_267817


namespace angle_ABM_l267_267791

theorem angle_ABM (A B C D M : Point) (α : ℝ)
  (sqABCD : is_square A B C D)
  (inside_sq : M ∈ interior ABCD)
  (angle_MAC : angle M A C = α)
  (angle_MCD : angle M C D = α)
  : angle A B M = 90 - 2 * α := 
sorry

end angle_ABM_l267_267791


namespace point_Q_fixed_line_KQ_eq_QH_l267_267952

-- Define the geometric setup of a non-isosceles triangle and its altitudes
variables {A B C : Type} [NonIsoscelesTriangle A B C]
variables {H : Point} (h_orthocenter : Orthocenter H A B C)
variables (P : Point) (on_side_bc : OnSegment P B C)
variables {X Y : Point} (proj_X_AB : Projection X P A B) (proj_Y_AC : Projection Y P A C)
variables {Q : Point} (tangents_meet_Q : TangentsIntersect Q (Circumcircle X B H) (Circumcircle H C Y))
variables {R T K : Point} (rt_meets_bc_at_K : IntersectsAt K (Line R T) (Line B C))

-- Part (a): Prove Q lies on a fixed line
theorem point_Q_fixed_line (b c : Real) (b_c_nonzero : b ≠ 0 ∧ c ≠ 0) :
  ∃ (line_Q : Line), ∀ P : Point,
  (on_side_bc P) → (tangents_meet_Q Q (Circumcircle X B H) (Circumcircle H C Y)) → (Q ∈ line_Q) :=
sorry

-- Part (b): Prove KQ = QH
theorem KQ_eq_QH :
  dist K Q = dist Q H :=
sorry

end point_Q_fixed_line_KQ_eq_QH_l267_267952


namespace cylinder_radius_height_diagonal_l267_267017

theorem cylinder_radius_height_diagonal : 
  ∀ (r h d : ℕ), r = 5 → h = 12 → d = 13 → (r^2 + h^2 = d^2) :=
by
  intros r h d r_eq h_eq d_eq
  rw [r_eq, h_eq, d_eq]
  norm_num
  sorry

end cylinder_radius_height_diagonal_l267_267017


namespace cos_pi_minus_2theta_l267_267718

theorem cos_pi_minus_2theta :
  ∃ θ : ℝ, (3, -4) = (3 * cos θ, -4 * sin θ) → cos(π - 2 * θ) = 7 / 25 :=
by
  let θ := real.arctan (4 / 3)
  refine' ⟨θ, _⟩
  intro H
  -- Further work to be done to actually link θ with the angle and the trigonometric identities.
  sorry

end cos_pi_minus_2theta_l267_267718


namespace sum_inverse_n_n_plus_3_l267_267162

theorem sum_inverse_n_n_plus_3 :
  (\sum' n : ℕ, if n = 0 then 0 else 1 / (n * (n + 3))) = 11 / 18 :=
by
  sorry

end sum_inverse_n_n_plus_3_l267_267162


namespace product_of_five_consecutive_is_divisible_by_sixty_l267_267515

theorem product_of_five_consecutive_is_divisible_by_sixty (n : ℤ) :
  60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end product_of_five_consecutive_is_divisible_by_sixty_l267_267515


namespace earnings_difference_l267_267618

def commission_old (sales : ℝ) : ℝ :=
  0.05 * sales

def commission_new (sales : ℝ) : ℝ :=
  let excess_sales := max (sales - 4000) 0
  1300 + 0.025 * excess_sales

theorem earnings_difference (sales : ℝ) (h_sales : sales = 12000) : 
  commission_new sales - commission_old sales = 900 :=
by
  sorry

end earnings_difference_l267_267618


namespace trains_will_clear_in_18point46_seconds_l267_267075

noncomputable def time_to_clear_each_other (length1 length2 : ℝ) (speed1_kmph speed2_kmph : ℝ) : ℝ := 
  let relative_speed := (speed1_kmph + speed2_kmph) * (1000 / 3600)
  let total_distance := length1 + length2
  total_distance / relative_speed

theorem trains_will_clear_in_18point46_seconds : time_to_clear_each_other 120 280 42 36 ≈ 18.46 :=
  by
  sorry

end trains_will_clear_in_18point46_seconds_l267_267075


namespace frog_jump_distance_l267_267391

variable (grasshopper_jump : ℕ) (frog_jump : ℕ) (mouse_jump : ℕ)
hypothesis (h1 : grasshopper_jump = 19)
hypothesis (h2 : frog_jump = grasshopper_jump + 39)

theorem frog_jump_distance : frog_jump = 58 := by
  -- The proof will go here, but it is skipped with sorry
  sorry

end frog_jump_distance_l267_267391


namespace appropriate_sampling_methods_l267_267578

/-- Define the regions with corresponding sales points -/
def region_A : ℕ := 150
def region_B : ℕ := 120
def region_C : ℕ := 180
def region_D : ℕ := 150
def total_sales_points : ℕ := region_A + region_B + region_C + region_D

/-- Investigation (1) sample size -/
def investigation_1_sample_size : ℕ := 100

/-- In region C, specific details for investigation (2) -/
def large_sales_points_in_C : ℕ := 20
def investigation_2_sample_size : ℕ := 7

/-- The appropriate sampling methods are Stratified sampling and Simple random sampling -/
theorem appropriate_sampling_methods:
  ∃ method1 method2, method1 = "Stratified sampling" ∧ method2 = "Simple random sampling" :=
by
  exists "Stratified sampling"
  exists "Simple random sampling"
  split
  . exact rfl
  . exact rfl

end appropriate_sampling_methods_l267_267578


namespace max_divisor_of_five_consecutive_integers_l267_267506

theorem max_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, 60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intros n
  sorry

end max_divisor_of_five_consecutive_integers_l267_267506


namespace find_fa_fb_max_perimeter_l267_267733

-- Definition of the curve C
def curve_C (x y : ℝ) : Prop := x^2 + 3 * y^2 = 12

-- Parametric equation of a line
def line_l (m t : ℝ) := (x = m + (Real.sqrt 2) / 2 * t) ∧ (y = (Real.sqrt 2) / 2 * t)

-- The focus F of the curve C
def focus_F : ℝ × ℝ := (-2 * Real.sqrt 2, 0)

-- Verify if the focus F lies on the line l
def focus_on_line (m: ℝ) : Prop := 
  ∃ (t : ℝ), line_l (-2 * Real.sqrt 2) t

-- Statement 1: Find the value of |FA|·|FB|
theorem find_fa_fb (m: ℝ) :  
  focus_on_line m → (∃ t1 t2 : ℝ, t1 ≠ t2 ∧ curve_C (m + (Real.sqrt 2) / 2 * t1) ((Real.sqrt 2) / 2 * t1) 
  ∧ curve_C (m + (Real.sqrt 2) / 2 * t2) ((Real.sqrt 2) / 2 * t2) ∧ |t1 - t2| = 2) :=
sorry

-- Statement 2: Find the maximum perimeter of the inscribed rectangle
theorem max_perimeter : 
(∃ (x y: ℝ), 0 < x ∧ x < 2 * Real.sqrt 3 ∧ 0 < y ∧ y < 2 ∧ curve_C x y ∧ (4 * x + 4 * y) ≤ 16 ) :=
sorry

end find_fa_fb_max_perimeter_l267_267733


namespace largest_divisor_of_product_of_five_consecutive_integers_l267_267466

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∀ (n : ℤ), ∃ (d : ℤ), d = 60 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end largest_divisor_of_product_of_five_consecutive_integers_l267_267466


namespace smallest_k_for_permutations_l267_267549

theorem smallest_k_for_permutations (n : ℕ) : 
  ∃ k, (∀ p : Equiv.Perm (Fin n), (p ^ k) = Equiv.refl (Fin n)) ∧ 
       (∀ m, (∀ p : Equiv.Perm (Fin n), (p ^ m) = Equiv.refl (Fin n)) → k ≤ m) ↔ 
       k = Nat.lcm (Finset.range (n+1)).val :=
begin
  sorry
end

end smallest_k_for_permutations_l267_267549


namespace max_height_projectile_l267_267598

def height (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 50

theorem max_height_projectile : ∃ t : ℝ, height t = 175 :=
by
  sorry

end max_height_projectile_l267_267598


namespace S₈_proof_l267_267861

-- Definitions
variables {a₁ q : ℝ} {S : ℕ → ℝ}

-- Condition for the sum of the first n terms of the geometric sequence
def geom_sum (n : ℕ) : ℝ :=
  a₁ * (1 - q ^ n) / (1 - q)

-- Conditions from the problem
def S₄ := geom_sum 4 = -5
def S₆ := geom_sum 6 = 21 * geom_sum 2

-- Theorem to prove
theorem S₈_proof (h₁ : S₄) (h₂ : S₆) : geom_sum 8 = -85 :=
by
  sorry

end S₈_proof_l267_267861


namespace largest_divisor_of_five_consecutive_integers_l267_267458

open Nat

theorem largest_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, k ∈ {n, n+1, n+2, n+3, n+4} ∧
    ∀ m ∈ {2, 3, 4, 5}, m ∣ k → 60 ∣ (n * (n+1) * (n+2) * (n+3) * (n+4)) := 
sorry

end largest_divisor_of_five_consecutive_integers_l267_267458


namespace remainder_of_polynomial_l267_267687

theorem remainder_of_polynomial :
  let p := 5*x^5 - 12*x^4 + 3*x^3 - 7*x^2 + x - 30
  let x_val := (3 : ℝ)
  p.eval x_val = 234 := by
sorry

end remainder_of_polynomial_l267_267687


namespace solve_for_x_l267_267004

theorem solve_for_x (x : ℚ) : (2/3 : ℚ) - 1/4 = 1/x → x = 12/5 := 
by
  sorry

end solve_for_x_l267_267004


namespace triangle_max_area_l267_267283

-- definitions of geometric entities and properties
variables (A B C : Type) [normed_group A] [inner_product_space ℝ A]
variables (a b c : ℝ) (vCA vCB : A)

-- conditions
axioms
  (opposite_sides : ∀ (A B C : A), (a b c : ℝ) )
  (sides_eq : opposite_sides a b c)
  (cosine_eq : 4 * real.cos c + real.cos (2 * c) = 4 * real.cos c * real.cos (c / 2) ^ 2)
  (magnitude_eq : abs (vCA - 1/2 • vCB) = 2)

-- Theorem statement
theorem triangle_max_area (A B C : A) (a b c : ℝ) 
  (vCA vCB : A) : 
  4 * real.cos c + real.cos (2 * c) = 4 * real.cos c * real.cos (c / 2) ^ 2 → 
  abs (vCA - 1 / 2 • vCB) = 2 → 
  c = real.pi / 3 ∧ (1 / 2 * a * b * real.sin (real.pi / 3) ≤ 2 * real.sqrt 3) := 
sorry

end triangle_max_area_l267_267283


namespace negate_existence_l267_267969

theorem negate_existence (x ∈ ℝ) : ¬∃ x : ℝ, x^2 - x - 1 > 0 → ∀ x : ℝ, x^2 - x - 1 ≤ 0 :=
by
  sorry

end negate_existence_l267_267969


namespace average_class_size_l267_267031

theorem average_class_size 
  (num_3_year_olds : ℕ) 
  (num_4_year_olds : ℕ) 
  (num_5_year_olds : ℕ) 
  (num_6_year_olds : ℕ) 
  (class_size_3_and_4 : num_3_year_olds = 13 ∧ num_4_year_olds = 20) 
  (class_size_5_and_6 : num_5_year_olds = 15 ∧ num_6_year_olds = 22) :
  (num_3_year_olds + num_4_year_olds + num_5_year_olds + num_6_year_olds) / 2 = 35 :=
by
  sorry

end average_class_size_l267_267031


namespace product_of_five_consecutive_is_divisible_by_sixty_l267_267524

theorem product_of_five_consecutive_is_divisible_by_sixty (n : ℤ) :
  60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end product_of_five_consecutive_is_divisible_by_sixty_l267_267524


namespace calc_expression_eq_l267_267150

theorem calc_expression_eq : 
  (sqrt 8 - abs (-2) + (1 / 3)^(-1) - 2 * real.cos (real.pi / 4)) = sqrt 2 + 1 :=
by sorry

end calc_expression_eq_l267_267150


namespace max_divisor_of_five_consecutive_integers_l267_267509

theorem max_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, 60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intros n
  sorry

end max_divisor_of_five_consecutive_integers_l267_267509


namespace min_n_div_2_3_abs_diff_pow_l267_267109

theorem min_n_div_2_3_abs_diff_pow (n : ℕ) :
  n ≠ 0 ∧
  (∀ a b : ℕ, abs (2^a - 3^b) ≠ n) ∧
  ¬ (∃ m : ℕ, m > 0 ∧ m < n ∧ m % 2 ≠ 0 ∧ m % 3 ≠ 0 ∧ ∀ a b : ℕ, abs (2^a - 3^b) ≠ m) :=
n = 35
  sorry

end min_n_div_2_3_abs_diff_pow_l267_267109


namespace tony_water_intake_l267_267051

-- Define the constants and conditions
def water_yesterday : ℝ := 48
def percentage_less_yesterday : ℝ := 0.04
def percentage_more_day_before_yesterday : ℝ := 0.05

-- Define the key quantity to find
noncomputable def water_two_days_ago : ℝ := water_yesterday / (1.05 * (1 - percentage_less_yesterday))

-- The proof statement
theorem tony_water_intake :
  water_two_days_ago = 47.62 :=
by
  sorry

end tony_water_intake_l267_267051


namespace equalSidesFromConditions_l267_267917

open Real Angle

noncomputable def cyclicQuadrilateralABCD (A B C D : Point) : Prop :=
  inscribedQuadrilateral A B C D

noncomputable def extensionWithProperty (A D E: Point) (C : Point) : Prop :=
  (segmentLength A C = segmentLength C E) ∧
  (angleBetween B D C = angleBetween D E C)

theorem equalSidesFromConditions
  (A B C D E : Point)
  (h1 : cyclicQuadrilateralABCD A B C D)
  (h2 : extensionWithProperty A D E C):
  segmentLength A B = segmentLength D E := 
sorry

end equalSidesFromConditions_l267_267917


namespace average_visitors_on_other_days_l267_267587

theorem average_visitors_on_other_days
  (avg_sunday_visitors : ℕ) (avg_monthly_visitors : ℕ) (num_sundays : ℕ) (num_days_in_month : ℕ)
  (total_days_in_month : num_days_in_month = 30) (first_day_sunday : Prop)
  (num_other_days : num_days_in_month - num_sundays = 25) :
  avg_sunday_visitors = 510 →
  avg_monthly_visitors = 285 →
  ∃ (V : ℕ), 
    let total_sunday_visitors := num_sundays * avg_sunday_visitors,
        total_visitors_in_month := avg_monthly_visitors * num_days_in_month,
        total_other_day_visitors := num_other_days * V
    in 
    total_sunday_visitors + total_other_day_visitors = total_visitors_in_month ∧ V = 240 := 
by
  sorry

end average_visitors_on_other_days_l267_267587


namespace lines_parallel_to_plane_ABP_l267_267228

-- Define the cube with vertices labeled as ABCD A_1 B_1 C_1 D_1
structure Cube :=
  (A B C D A1 B1 C1 D1 : Point)
  (AB : Line A B)
  (BC : Line B C)
  (CD : Line C D)
  (DA : Line D A)
  (AA1 : Line A A1)
  (BB1 : Line B B1)
  (CC1 : Line C C1)
  (DD1 : Line D D1)
  (A1B1 : Line A1 B1)
  (B1C1 : Line B1 C1)
  (C1D1 : Line C1 D1)
  (D1A1 : Line D1 A1)
  -- Additional constraints for edges and right angles to form a cube could be added here if necessary

-- Define points, lines, and plane based on the provided condition
variable (cube : Cube)
variable (P : Point)
variable (on_edge_P : P ∈ cube.DD1)

-- Define the plane ABP
def plane_ABP : Plane := PlaneSpan cube.A cube.B P

-- Define the lines specified in the solution
def line_DC : Line := cube.CD
def line_D1C1 : Line := cube.C1D1
def line_A1B1 : Line := cube.A1B1

-- The theorem to be proven
theorem lines_parallel_to_plane_ABP :
  LineParallel plane_ABP line_DC ∧
  LineParallel plane_ABP line_D1C1 ∧
  LineParallel plane_ABP line_A1B1 :=
sorry

end lines_parallel_to_plane_ABP_l267_267228


namespace katie_candy_l267_267319

/--
Katie and her sister combined the candy they received for Halloween.
Katie had 8 pieces of candy while her sister had 23. They ate some pieces the first night and had 23 pieces left.
Prove that they ate 8 pieces the first night.
-/
theorem katie_candy : 
  let initial_katie := 8 in
  let initial_sister := 23 in
  let total_initial := initial_katie + initial_sister in
  let pieces_left := 23 in
  total_initial - pieces_left = 8 := 
by
  sorry

end katie_candy_l267_267319


namespace image_of_rectangle_is_correct_l267_267327

-- Definitions of the vertices
structure Point :=
  (x : ℝ)
  (y : ℝ)

def O : Point := ⟨0, 0⟩
def P : Point := ⟨2, 0⟩
def Q : Point := ⟨2, 2⟩
def R : Point := ⟨0, 2⟩

-- Transfer function
def transform (pt : Point) : Point :=
  ⟨pt.x^2 - pt.y^2, 2 * pt.x * pt.y⟩

-- Proving the image of the rectangle
theorem image_of_rectangle_is_correct :
  {transform O, transform P, transform Q, transform R} = {⟨0,0⟩, ⟨4,0⟩, ⟨0,8⟩, ⟨-4,0⟩} :=
  by sorry

end image_of_rectangle_is_correct_l267_267327


namespace smallest_c_l267_267330

theorem smallest_c {a b c : ℤ} (h1 : a < b) (h2 : b < c) 
  (h3 : 2 * b = a + c)
  (h4 : a^2 = c * b) : c = 4 :=
by
  -- We state the theorem here without proof. 
  -- The actual proof steps are omitted and replaced by sorry.
  sorry

end smallest_c_l267_267330


namespace remainder_2010_mod_1000_l267_267904

definition q (x : ℤ) : ℤ := x^2010 + x^2009 + x^2008 + x^2007 + x^2006 + x^2005 + x^2004 + x^2003 + x^2002 + x^2001 + x^2000 + x^1999 + x^1998 + x^1997 + x^1996 + 
  x^1995 + x^1994 + x^1993 + x^1992 + x^1991 + x^1990 + x^1989 + x^1988 + x^1987 + x^1986 + x^1985 + x^1984 + x^1983 + x^1982 + x^1981 + x^1980 + x^1979 + x^1978 +
  x^1977 + x^1976 + x^1975 + x^1974 + x^1973 + x^1972 + x^1971 + x^1970 + x^1969 + x^1968 + x^1967 + x^1966 + x^1965 + x^1964 + x^1963 + x^1962 + x^1961 + x^1960 +
  x^1959 + x^1958 + x^1957 + x^1956 + x^1955 + x^1954 + x^1953 + x^1952 + x^1951 + x^1950 + x^1949 + x^1948 + x^1947 + x^1946 + x^1945 + x^1944 + x^1943 + x^1942 +
  x^1941 + x^1940 + x^1939 + x^1938 + x^1937 + x^1936 + x^1935 + x^1934 + x^1933 + x^1932 + x^1931 + x^1930 + x^1929 + x^1928 + x^1927 + x^1926 + x^1925 + ... + x^1 + 1

definition divisor (x : ℤ) : ℤ := x^4 - x^3 + 2x^2 - x + 1

theorem remainder_2010_mod_1000 : 
  let s (x : ℤ) := (q(x)) % (divisor(x)) in (abs(s(2010)) % 1000) = 111 :=
sorry

end remainder_2010_mod_1000_l267_267904


namespace forest_total_trees_l267_267290

theorem forest_total_trees (T spruce pine oak birch: ℝ)
  (h1: spruce = 0.10 * T)
  (h2: pine = 0.13 * T)
  (h3: oak = spruce + pine)
  (h4: birch = 2160)
  (h5: T = spruce + pine + oak + birch) : 
  T = 4000 :=
by 
  have hs : spruce = 0.10 * T := h1
  have hp : pine = 0.13 * T := h2
  have ho : oak = spruce + pine := h3
  rw [hs, hp, ->] at ho
  have hb : birch = 2160 := h4
  rw [hs, hp, ho, hb] at h5
  simp only [h1, h2, h3, h4] at h5
  rw [add_assoc, <- add_assoc (0.10 * T), h1] at h5
  rw [<- add_assoc, add_assoc (0.13 * T)] at h5
  simp at h5
  sorry -- proof will go here

end forest_total_trees_l267_267290


namespace min_value_expression_l267_267748

theorem min_value_expression (x : ℝ) (h : x > 2) : 
  ∃ y, y = x + 1 / (x - 2) ∧ y = 4 := 
sorry

end min_value_expression_l267_267748


namespace find_phi_l267_267022

theorem find_phi {φ : ℝ} (h1 : 0 ≤ φ ∧ φ ≤ π) (hf : ∀ x, cos (3 * x + φ) = -cos (3 * -x + φ)) : 
  φ = π / 2 :=
by
  sorry

end find_phi_l267_267022


namespace find_number_l267_267369

theorem find_number (x : ℕ) (hx : (x / 100) * 100 = 20) : x = 20 :=
sorry

end find_number_l267_267369


namespace min_operations_solution_l267_267434

noncomputable def min_operations (c_initial : ℝ) (c_target : ℝ) (operation_effect : ℝ) : ℕ :=
  let x := (Real.log c_target) / (Real.log operation_effect)
  let min_x := x.ceil.to_nat
  min_x

theorem min_operations_solution : min_operations 0.9 0.1 0.9 = 21 := 
by {
  -- We can prove this but for now we skip actual proof steps
  sorry
}

end min_operations_solution_l267_267434


namespace number_of_people_on_boats_l267_267567

def boats := 5
def people_per_boat := 3

theorem number_of_people_on_boats : boats * people_per_boat = 15 :=
by
  sorry

end number_of_people_on_boats_l267_267567


namespace eight_term_sum_l267_267815

variable {α : Type*} [Field α]
variable (a q : α)

-- Define the n-th sum of the geometric sequence
def S_n (n : ℕ) : α := a * (1 - q ^ n) / (1 - q)

-- Given conditions
def S4 : α := S_n 4 = -5
def S6 : α := S_n 6 = 21 * S_n 2

-- Prove the target statement
theorem eight_term_sum : S_n 8 = -85 :=
  sorry

end eight_term_sum_l267_267815


namespace certain_number_problem_l267_267088

theorem certain_number_problem :
  ∃ x : ℕ, 40 * x = 173 * 240 ∧ x = 1038 :=
by
  use 1038
  split
  · calc
    40 * 1038 = 41520 : by norm_num
    ... = 173 * 240 : by norm_num
  · rfl


end certain_number_problem_l267_267088


namespace probability_same_color_ball_draw_l267_267285

theorem probability_same_color_ball_draw (red white : ℕ) 
    (h_red : red = 2) (h_white : white = 2) : 
    let total_outcomes := (red + white) * (red + white)
    let same_color_outcomes := 2 * (red * red + white * white)
    same_color_outcomes / total_outcomes = 1 / 2 :=
by
  sorry

end probability_same_color_ball_draw_l267_267285


namespace eight_term_sum_l267_267823

variable {α : Type*} [Field α]
variable (a q : α)

-- Define the n-th sum of the geometric sequence
def S_n (n : ℕ) : α := a * (1 - q ^ n) / (1 - q)

-- Given conditions
def S4 : α := S_n 4 = -5
def S6 : α := S_n 6 = 21 * S_n 2

-- Prove the target statement
theorem eight_term_sum : S_n 8 = -85 :=
  sorry

end eight_term_sum_l267_267823


namespace negation_example_l267_267394

theorem negation_example :
  (¬ (∀ x : ℝ, x^2 - 2 * x + 1 > 0)) ↔ (∃ x : ℝ, x^2 - 2 * x + 1 ≤ 0) :=
sorry

end negation_example_l267_267394


namespace imaginary_part_of_complex_l267_267392

open Complex -- Opens the complex numbers namespace

theorem imaginary_part_of_complex:
  ∀ (a b c d : ℂ), (a = (2 + I) / (1 - I) - (2 - I) / (1 + I)) → (a.im = 3) :=
by
  sorry

end imaginary_part_of_complex_l267_267392


namespace fewer_bands_l267_267305

theorem fewer_bands (J B Y : ℕ) (h1 : J = B + 10) (h2 : B - 4 = 8) (h3 : Y = 24) :
  Y - J = 2 :=
sorry

end fewer_bands_l267_267305


namespace undefined_values_of_expression_l267_267668

def num_undefined_values (f : ℝ → ℝ) : ℕ :=
  {x | ∃ y, f y = x}.to_finset.card

theorem undefined_values_of_expression :
  num_undefined_values (λ x, (x^2 + 4 * x - 5) * (x + 4)) = 3 := 
sorry

end undefined_values_of_expression_l267_267668


namespace geometric_sequence_sum_l267_267889

variable (a1 q : ℝ) -- Define the first term and common ratio as real numbers

-- Define the sum of the first n terms of a geometric sequence
def S (n : ℕ) : ℝ := a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum :
  (S 4 a1 q = -5) → (S 6 a1 q = 21 * S 2 a1 q) → (S 8 a1 q = -85) :=
by
  intro h1 h2
  sorry

end geometric_sequence_sum_l267_267889


namespace clerical_percentage_after_reduction_l267_267563

-- Define the initial conditions
def total_employees : ℕ := 3600
def clerical_fraction : ℚ := 1/4
def reduction_fraction : ℚ := 1/4

-- Define the intermediate calculations
def initial_clerical_employees : ℚ := clerical_fraction * total_employees
def clerical_reduction : ℚ := reduction_fraction * initial_clerical_employees
def new_clerical_employees : ℚ := initial_clerical_employees - clerical_reduction
def total_employees_after_reduction : ℚ := total_employees - clerical_reduction

-- State the theorem
theorem clerical_percentage_after_reduction :
  (new_clerical_employees / total_employees_after_reduction) * 100 = 20 :=
sorry

end clerical_percentage_after_reduction_l267_267563


namespace minimum_a_l267_267910

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a * real.log x - (a - 2) * x

def has_two_zeros (a : ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ f(a, x₁) = 0 ∧ f(a, x₂) = 0

theorem minimum_a (a : ℝ) :
  has_two_zeros a → ∃ n : ℕ, n = 3 ∧ (n : ℝ) <= a := by
  sorry

end minimum_a_l267_267910


namespace least_xy_l267_267711

theorem least_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 1 / (3 * y) = 1 / 9) : x * y = 108 :=
by
  sorry

end least_xy_l267_267711


namespace crayons_per_row_l267_267678

theorem crayons_per_row:
  ∀ (c : ℕ), 
  (11 * 31) + (11 * c) = 638 → 
  c = 27 :=
by
  intros c h
  have h_pencil : 11 * 31 = 341 := by norm_num
  have h_total : 638 = 341 + (11 * c) := by rw [h_pencil, add_comm] at h; exact h
  suffices h_crayons: 297 = 11 * c by
    exact eq_of_mul_eq_mul_left' (11 : ℕ) (by norm_num) h_crayons
  exact eq_of_sub_eq_zero (by linarith)
  sorry

end crayons_per_row_l267_267678


namespace triangle_area_is_approx_175_95_l267_267278

def s (a b c : ℝ) : ℝ := (a + b + c) / 2
def heron_area (a b c : ℝ) : ℝ :=
  let semi_perimeter := s a b c in
  real.sqrt (semi_perimeter * (semi_perimeter - a) * (semi_perimeter - b) * (semi_perimeter - c))

theorem triangle_area_is_approx_175_95 :
  heron_area 26 24 15 ≈ 175.95 :=
by
  sorry

end triangle_area_is_approx_175_95_l267_267278


namespace cost_of_one_each_l267_267433

theorem cost_of_one_each (x y z : ℝ) (h1 : 3 * x + 7 * y + z = 24) (h2 : 4 * x + 10 * y + z = 33) :
  x + y + z = 6 :=
sorry

end cost_of_one_each_l267_267433


namespace integer_values_satisfying_sqrt_condition_l267_267417

theorem integer_values_satisfying_sqrt_condition :
  {x : ℤ | 5 < Real.sqrt x ∧ Real.sqrt x < 6}.card = 10 :=
by
  sorry

end integer_values_satisfying_sqrt_condition_l267_267417


namespace union_dues_proof_l267_267442

noncomputable def h : ℕ := 42
noncomputable def r : ℕ := 10
noncomputable def tax_rate : ℝ := 0.20
noncomputable def insurance_rate : ℝ := 0.05
noncomputable def take_home_pay : ℝ := 310

noncomputable def gross_earnings : ℝ := h * r
noncomputable def tax_deduction : ℝ := tax_rate * gross_earnings
noncomputable def insurance_deduction : ℝ := insurance_rate * gross_earnings
noncomputable def total_deductions : ℝ := tax_deduction + insurance_deduction
noncomputable def net_earnings_before_union_dues : ℝ := gross_earnings - total_deductions
noncomputable def union_dues_deduction : ℝ := net_earnings_before_union_dues - take_home_pay

theorem union_dues_proof : union_dues_deduction = 5 := 
by sorry

end union_dues_proof_l267_267442


namespace cost_of_lollipop_l267_267635

theorem cost_of_lollipop (n : ℕ) : 
  (∀ m, m = 2 → cost m = 2.40 → cost 1 = 1.20) ∧ 
  (∀ k, k = 6 → cost k = 7.20 → cost 1 = 1.20) -> 
  cost n = n * 1.20 :=
by sorry

end cost_of_lollipop_l267_267635


namespace angle_EDC_60_angle_AEC_90_l267_267306

-- Definitions of the geometric conditions
variables {Point : Type} [affine_space Point]
variables (A E D B C : Point)
variables (circle_c : circle Point) (tangent_c_A : tangent circle_c A E)
variables [perpendicular AD BC] [parallel DE AB]
variables (angle_EAC : angle E A C = 60)

-- Goals to Prove
theorem angle_EDC_60 : angle E D C = 60 := sorry

theorem angle_AEC_90 : angle A E C = 90 := sorry

end angle_EDC_60_angle_AEC_90_l267_267306


namespace ellipse_focal_length_m_l267_267726

theorem ellipse_focal_length_m (m : ℝ) (h1 : 0 < m)
  (h2 : ∃ (x y : ℝ), (x / 5)^2 + (y / m)^2 = 1 ∧ 2 * real.sqrt (25 - m^2) = 8 ∨ 2 * real.sqrt (m^2 - 25) = 8) :
  m = 3 ∨ m = real.sqrt 41 :=
by
  sorry

end ellipse_focal_length_m_l267_267726


namespace lineUpCorrect_roundTableCorrect_committeeCorrect_l267_267583

namespace HospitalEmployees

def employees : List String :=
  ["Sara Dores da Costa", "Iná Lemos", "Ester Elisa", "Ema Thomas", "Ana Lisa", "Inácio Filho"]

def numWaysToLineUp (ls : List α) : Nat :=
  List.permutations ls |>.length

def numWaysToRoundTable (ls : List α) : Nat :=
  if h : 0 < ls.length then
    (List.length ls - 1)!.toNat
  else 0

def numWaysToFormCommittee (ls : List α) : Nat :=
  if h : 3 ≤ ls.length then
    ls.length * (ls.length - 1) * (ls.length - 2)
  else 0

#eval numWaysToLineUp employees -- expected 720
#eval numWaysToRoundTable employees -- expected 120
#eval numWaysToFormCommittee employees -- expected 120

theorem lineUpCorrect : numWaysToLineUp employees = 720 := by
  sorry

theorem roundTableCorrect : numWaysToRoundTable employees = 120 := by
  sorry

theorem committeeCorrect : numWaysToFormCommittee employees = 120 := by
  sorry

end HospitalEmployees

end lineUpCorrect_roundTableCorrect_committeeCorrect_l267_267583


namespace range_of_a_l267_267908

-- Define the set M
def M : Set ℝ := { x | -1 ≤ x ∧ x < 2 }

-- Define the set N
def N (a : ℝ) : Set ℝ := { x | x ≤ a }

-- The theorem to be proved
theorem range_of_a (a : ℝ) (h : (M ∩ N a).Nonempty) : a ≥ -1 := sorry

end range_of_a_l267_267908


namespace seed_mixture_percentage_l267_267363

theorem seed_mixture_percentage (x y : ℝ) 
  (hx : 0.4 * x + 0.25 * y = 30)
  (hxy : x + y = 100) :
  x / 100 = 0.3333 :=
by 
  sorry

end seed_mixture_percentage_l267_267363


namespace trapezoid_proof_l267_267916

variables {Point : Type} [MetricSpace Point]

-- Definitions of the points and segments as given conditions.
variables (A B C D E : Point)

-- Definitions representing the trapezoid and point E's property.
def is_trapezoid (ABCD : (A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A)) : Prop :=
  (A ≠ B) ∧ (C ≠ D)

def on_segment (E : Point) (A D : Point) : Prop :=
  -- This definition will encompass the fact that E is on segment AD.
  -- Representing the notion that E lies between A and D.
  dist A E + dist E D = dist A D

def equal_perimeters (E : Point) (A B C D : Point) : Prop :=
  let p1 := (dist A B + dist B E + dist E A)
  let p2 := (dist B C + dist C E + dist E B)
  let p3 := (dist C D + dist D E + dist E C)
  p1 = p2 ∧ p2 = p3

-- The theorem we need to prove.
theorem trapezoid_proof (ABCD : A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A) (onSeg : on_segment E A D) (eqPerim : equal_perimeters E A B C D) : 
  dist B C = dist A D / 2 :=
sorry

end trapezoid_proof_l267_267916


namespace complex_number_identity_l267_267756

theorem complex_number_identity : |-i| + i^2018 = 0 := by
  sorry

end complex_number_identity_l267_267756


namespace locus_intersection_point_l267_267284

variables {R : Type*} [Field R]

def A (a : R) := (a, 0)
def B (b : R) := (b, 0)
def C (c : R) := (0, c)

theorem locus_intersection_point (a b c x y : R) (h1 : a ≠ 0) (h2 : b ≠ 0) (h_C : C c = (0, c)) 
(h_l : ∃ λ, ∀ x y, y = λ * x) (h_AC : ∃ λ, ∀ x y, y = λ * x + c):
(1 + (b * y^2)) = x * (b + c * (b * x)) - (a * x * y - a * x^2 * (a / c)) - 1 → 
  (frac ((x - (b / 2))^2) ((b^2) / 4)) + (frac (y^2) ((a * b) / 4)) = 1 :=
by sorry

end locus_intersection_point_l267_267284


namespace total_surface_area_l267_267019

theorem total_surface_area (r : ℝ) :
  (π * r^2 = 144 * π) →
  (radius_disk : ℝ) (radius_disk = 5) →
  (total_area : ℝ) (total_area = 2 * π * r^2 + π * radius_disk^2) →
  total_area = 313 * π :=
by
  sorry

end total_surface_area_l267_267019


namespace number_of_dissimilar_terms_l267_267641

theorem number_of_dissimilar_terms :
  let n := 7;
  let k := 4;
  let number_of_terms := Nat.choose (n + k - 1) (k - 1);
  number_of_terms = 120 :=
by
  sorry

end number_of_dissimilar_terms_l267_267641


namespace eccentricity_of_hyperbola_l267_267218

-- Define the parameters and given conditions
variables (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop :=
  y^2 = 4 * c * x

-- Define the focal distance condition
def focal_distance : Prop :=
  ∃ F1 F2 : ℝ × ℝ, (2 * c = |F1.1 - F2.1|) ∧ (|F1.1 - M.1| = 4 * c)

-- Define the condition given by the relationship c^2 = a^2 + b^2
def focal_relation : Prop :=
  c^2 = a^2 + b^2

-- Statement in Lean to prove the eccentricity
theorem eccentricity_of_hyperbola :
  hyperbola a b c →
  parabola a b c →
  focal_distance a b c →
  focal_relation a b c →
  e = 1 + (√6 / 3) :=
by
  sorry

end eccentricity_of_hyperbola_l267_267218


namespace solve_n_l267_267324

theorem solve_n (p : ℕ) (h_prime : Nat.Prime p) (h_odd : p % 2 = 1) (n : ℕ) (h_pos : n > 0)
  (h_int : ∃ k : ℕ, sqrt (n^2 - n * p) = k) :
  n = (p + 1) * (p + 1) / 4 :=
by
  sorry

end solve_n_l267_267324


namespace number_of_integer_solutions_l267_267202

noncomputable def f (t : ℝ) : ℝ := sorry

axiom f_add_prop : ∀ x y : ℝ, f (x + y) = f x + f y + x * y + 1
axiom f_neg_two : f (-2) = -2

theorem number_of_integer_solutions (hf : ∀ x y : ℝ, f (x + y) = f x + f y + x * y + 1)
                                   (hf_neg_two : f (-2) = -2) :
  {a : ℤ | f a = a}.to_finset.card = 2 :=
begin
  sorry
end

end number_of_integer_solutions_l267_267202


namespace sqrt_sequence_solution_l267_267184

theorem sqrt_sequence_solution : 
  (∃ x : ℝ, x = sqrt (18 + x) ∧ x > 0) → (∃ x : ℝ, x = 6) :=
by
  assume h,
  sorry

end sqrt_sequence_solution_l267_267184


namespace walking_speed_l267_267590

-- Define the constants and variables
def speed_there := 25 -- speed from village to post-office in kmph
def total_time := 5.8 -- total round trip time in hours
def distance := 20.0 -- distance to the post-office in km
 
-- Define the theorem that needs to be proved
theorem walking_speed :
  ∃ (speed_back : ℝ), speed_back = 4 := 
by
  sorry

end walking_speed_l267_267590


namespace area_ratio_triangle_l267_267784

noncomputable def length_CA : ℝ := 40
noncomputable def length_CB : ℝ := 45
noncomputable def length_AB : ℝ := 55
def point_D_bisects_angle_ACB (A B C D : Type) : Prop := sorry -- Define the angle bisector property.

theorem area_ratio_triangle (A B C D : Type)
  (hCA : length_CA = 40)
  (hCB : length_CB = 45)
  (hAB : length_AB = 55)
  (hD : point_D_bisects_angle_ACB A B C D) :
  ratio_of_area (triangle B C D) (triangle A C D) = 9 / 8 :=
sorry -- Proof to be provided.

end area_ratio_triangle_l267_267784


namespace S₈_proof_l267_267856

-- Definitions
variables {a₁ q : ℝ} {S : ℕ → ℝ}

-- Condition for the sum of the first n terms of the geometric sequence
def geom_sum (n : ℕ) : ℝ :=
  a₁ * (1 - q ^ n) / (1 - q)

-- Conditions from the problem
def S₄ := geom_sum 4 = -5
def S₆ := geom_sum 6 = 21 * geom_sum 2

-- Theorem to prove
theorem S₈_proof (h₁ : S₄) (h₂ : S₆) : geom_sum 8 = -85 :=
by
  sorry

end S₈_proof_l267_267856


namespace correct_proposition_l267_267712

open Real

-- Define proposition p
def p : Prop := ∀ x : ℝ, 2^x > 1

-- Define f and its derivative
def f (x : ℝ) := x - sin x
def f_prime (x : ℝ) := 1 - cos x

-- Define proposition q
def q : Prop := ∀ x : ℝ, 0 ≤ f_prime x

-- The statement of the proof problem
theorem correct_proposition : (¬ p) ∧ q := by
  -- Proof goes here
  sorry

end correct_proposition_l267_267712


namespace a_2016_is_neg1_l267_267214

noncomputable def a : ℕ → ℤ
| 0     => 0 -- Arbitrary value for n = 0 since sequences generally start from 1 in Lean
| 1     => 1
| 2     => 2
| n + 1 => a n - a (n - 1)

theorem a_2016_is_neg1 : a 2016 = -1 := sorry

end a_2016_is_neg1_l267_267214
