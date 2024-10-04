import Mathlib

namespace square_area_from_diagonal_l769_769538

theorem square_area_from_diagonal (d : ℝ) (h : d = 8 * Real.sqrt 2) : 
  ∃ (A : ℝ), A = 64 :=
begin
  use 64,
  sorry
end

end square_area_from_diagonal_l769_769538


namespace standard_spherical_coordinates_l769_769207

theorem standard_spherical_coordinates :
  ∀ (ρ θ φ : ℝ), 
  ρ = 5 → θ = 3 * Real.pi / 4 → φ = 9 * Real.pi / 5 →
  (ρ > 0) →
  (0 ≤ θ ∧ θ < 2 * Real.pi) →
  (0 ≤ φ ∧ φ ≤ Real.pi) →
  (ρ, θ, φ) = (5, 7 * Real.pi / 4, Real.pi / 5) :=
by sorry

end standard_spherical_coordinates_l769_769207


namespace A_share_is_correct_l769_769561

variables (x : ℝ) (total_profit : ℝ) (A_share : ℝ)

-- Define the investment amounts and periods
def A_investment_amount := x
def B_investment_amount := 2 * x
def C_investment_amount := 3 * x

def A_investment_period := 12
def B_investment_period := 6
def C_investment_period := 4

-- Define the total annual gain
def total_annual_gain := 18600

-- Calculate the share ratios
def A_share_ratio := A_investment_amount * A_investment_period
def B_share_ratio := B_investment_amount * B_investment_period
def C_share_ratio := C_investment_amount * C_investment_period

-- Compute the total ratio
def total_ratio := A_share_ratio + B_share_ratio + C_share_ratio

-- Calculate A's share of the profit
def calculated_A_share := (A_share_ratio / total_ratio) * total_profit

-- The proof problem statement
theorem A_share_is_correct : 
calculated_A_share = 6200 :=
by 
    sorry

end A_share_is_correct_l769_769561


namespace Okeydokey_investment_l769_769011

-- Definitions and conditions
variables (Okeydokey_apples Artichokey_apples : ℕ)
axiom Artichokey_paid : Artichokey_apples = 7
axiom total_earthworms : ℕ := 60
axiom Okeydokey_earthworms : ℕ := 25
axiom proportional_payout : ∀ (x y : ℕ), (Okeydokey_earthworms * (x + Artichokey_apples) = total_earthworms * x)

-- Theorem statement
theorem Okeydokey_investment : Okeydokey_apples = 5 :=
by {
  sorry
  -- Proof omitted
}

end Okeydokey_investment_l769_769011


namespace rod_length_l769_769517

-- Definitions for the problem's conditions
def length1 : ℝ := 13
def weight1 : ℝ := 13.4
def weight2 : ℝ := 6.184615384615385

-- The target length of the second rod
def length2 : ℝ := (weight2 * length1) / weight1 -- Calculation based on the given problem

-- The theorem stating the equivalence to be proved
theorem rod_length (h_length1 : length1 = 13) (h_weight1 : weight1 = 13.4) (h_weight2 : weight2 = 6.184615384615385) :
    length2 = 6 := by
  -- The proof is omitted
  sorry

end rod_length_l769_769517


namespace find_alpha_intersection_l769_769646

theorem find_alpha_intersection :
  ∀ (l : ℝ → ℝ) 
    (h_l : l 0 = 0) 
    (h_f : ∀ x : ℝ, 0 ≤ x → l x = |sin x| → ∃! α : ℝ, 0 ≤ α ∧ l α = |sin α| ∧ ∀ β : ℝ, β ≠ α → l β ≠ |sin β|),
  ∀ α : ℝ, (α = max (set_of (λ x : ℝ, l x = |sin x|)) ∧ α > 0) →
  (1 + α^2) * sin (2 * α) / (2 * α) = 1 :=
by
  intros l h_l h_f α h_max,
  sorry

end find_alpha_intersection_l769_769646


namespace no_valid_triangle_exists_l769_769596

-- Variables representing the sides and altitudes of the triangle
variables (a b c h_a h_b h_c : ℕ)

-- Definition of the perimeter condition
def perimeter_condition : Prop := a + b + c = 1995

-- Definition of integer altitudes condition (simplified)
def integer_altitudes_condition : Prop := 
  ∃ (h_a h_b h_c : ℕ), (h_a * 4 * a ^ 2 = 2 * a ^ 2 * b ^ 2 + 2 * a ^ 2 * c ^ 2 + 2 * c ^ 2 * b ^ 2 - a ^ 4 - b ^ 4 - c ^ 4)

-- The main theorem to prove no valid triangle exists
theorem no_valid_triangle_exists : ¬ (∃ (a b c : ℕ), perimeter_condition a b c ∧ integer_altitudes_condition a b c) :=
sorry

end no_valid_triangle_exists_l769_769596


namespace separable_iff_dual_l769_769364

variable (n t : ℕ)

-- Define X to be the set {1, 2, ..., n}
def X : Finset ℕ := Finset.range (n + 1)

-- Define A to be a family of sets indexed from {0, 1, ..., t-1}
variable (A : Fin t → Finset ℕ)

-- Define completely separable for a family of sets
def completely_separable (A : Fin t → Finset ℕ) : Prop :=
  ∀ (i j : ℕ), 1 ≤ i → i < j → j ≤ n → ∃ k h, (i ∈ A k) ∧ (i ∉ A h) ∧ (j ∈ A h) ∧ (j ∉ A k)

-- Define B_i for a given i
def B (i : ℕ) : Finset (Fin t) := {k ∈ Finset.fin_range t | i ∈ A k}

-- Define the family of sets B_i
def B_family : Fin n → Finset (Fin t) := λ i, B A i

-- Define the dual of S
def dual_of_S (B : Fin n → Finset (Fin t)) : Prop :=
  ∀ (i j : ℕ), 1 ≤ i → i < j → j ≤ n → ∃ k h, (k ∈ B i) ∧ (k ∉ B j) ∧ (h ∈ B j) ∧ (h ∉ B i)

-- Main theorem
theorem separable_iff_dual (A : Fin t → Finset ℕ) :
  completely_separable A ↔ dual_of_S (B_family A) :=
sorry

end separable_iff_dual_l769_769364


namespace power_of_negative_base_l769_769213

theorem power_of_negative_base : (-64 : ℤ)^(7 / 6) = -128 := by
  sorry

end power_of_negative_base_l769_769213


namespace power_of_negative_base_l769_769212

theorem power_of_negative_base : (-64 : ℤ)^(7 / 6) = -128 := by
  sorry

end power_of_negative_base_l769_769212


namespace player_b_wins_if_a_goes_first_l769_769942

theorem player_b_wins_if_a_goes_first (n : ℕ) (h : 2 ≤ n) :
  ∃ f : (fin (2 * n - 1) → ℝ), ∀ g : (fin (2 * n - 1) → ℝ), ∃ x : ℝ, sum (λ i, (f i) * (x ^ (2 * n - 1 - i))) + x ^ (2 * n) + 1 = 0 :=
sorry

end player_b_wins_if_a_goes_first_l769_769942


namespace tangent_line_y_intercept_l769_769688

def f (x : ℝ) (a : ℝ) : ℝ := a * Real.exp x - 3 * x + 1

theorem tangent_line_y_intercept :
  ∃ b : ℝ, (∀ a : ℝ, 
    (∀ (x : ℝ), deriv (λ x, f x a) x = a * Real.exp x - 3) →
    (deriv (λ x, f x a) 0 = 1) →
    (f 0 4 = 5) →
    (0, 5) ∈ (set_of (λ (p : ℝ × ℝ), p.2 = p.1 + b))) ∧ b = 5 :=
begin
  sorry
end

end tangent_line_y_intercept_l769_769688


namespace value_of_a2018_l769_769697

noncomputable def a : ℕ → ℝ
| 0       => 2
| (n + 1) => (1 + a n) / (1 - a n)

theorem value_of_a2018 : a 2017 = -3 := sorry

end value_of_a2018_l769_769697


namespace percentage_change_area_l769_769423

theorem percentage_change_area (L B : ℝ) :
  let A_original := L * B
  let A_new := (L / 2) * (3 * B)
  (A_new - A_original) / A_original * 100 = 50 := by
  sorry

end percentage_change_area_l769_769423


namespace Vulcan_can_enclose_flooded_cells_l769_769105

theorem Vulcan_can_enclose_flooded_cells (grid : Type) [fintype grid]
    (initial_flooded : set grid)
    (add_walls_per_turn : ℕ)
    (flood_spread : grid → set grid → set grid)
    : ∃ (levee : set (grid × grid)), 
      (∀ turn : ℕ, levee_properties levee turn initial_flooded add_walls_per_turn flood_spread) :=
sorry

end Vulcan_can_enclose_flooded_cells_l769_769105


namespace count_N_lt_500_solution_exists_l769_769288

theorem count_N_lt_500_solution_exists:
  ∃ (N : ℕ), N < 500 ∧ (∃ (x : ℝ), x^floor x = N) = 287 :=
sorry

end count_N_lt_500_solution_exists_l769_769288


namespace scheduling_methods_correct_l769_769461

-- define the problem conditions
def volunteers := {A, B, C}
def days := {mon, tue, wed, thu, fri}
def valid_schedule (schedule : volunteers → days) : Prop :=
  ∀ (x y : volunteers), schedule x ≠ schedule y ∧
  (schedule A < schedule B ∧ schedule A < schedule C)

noncomputable def total_scheduling_methods : ℕ :=
  -- Calculation to determine the total number of valid schedules
  sorry

theorem scheduling_methods_correct : total_scheduling_methods = 20 := 
sorry

end scheduling_methods_correct_l769_769461


namespace second_discount_percentage_l769_769077

-- Defining the variables
variables (P S : ℝ) (d1 d2 : ℝ)

-- Given conditions
def original_price : P = 200 := by sorry
def sale_price_after_initial_discount : S = 171 := by sorry
def first_discount_rate : d1 = 0.10 := by sorry

-- Required to prove
theorem second_discount_percentage :
  ∃ d2, (d2 = 0.05) :=
sorry

end second_discount_percentage_l769_769077


namespace simplify_complex_expr_l769_769403

theorem simplify_complex_expr :
  let i : ℂ := complex.I in
  3 * (2 + i) - i * (3 - i) + 2 * (1 - 2 * i) = 7 - 4 * i :=
by
  sorry

end simplify_complex_expr_l769_769403


namespace range_of_a_for_decreasing_function_l769_769274

variable (a : ℝ)

noncomputable def f (x : ℝ) : ℝ := -x^2 - 2 * (a - 1) * x + 5

noncomputable def f' (x : ℝ) : ℝ := -2 * x - 2 * (a - 1)

theorem range_of_a_for_decreasing_function :
  (∀ x : ℝ, -1 ≤ x → f' a x ≤ 0) → 2 ≤ a := sorry

end range_of_a_for_decreasing_function_l769_769274


namespace smallest_palindrome_prop_l769_769237

def is_palindrome (n : ℕ) : Prop :=
  let digits := Int.to_digits n in
  digits = List.reverse digits

def is_three_digit_palindrome (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000 ∧ is_palindrome n

noncomputable def smallest_palindrome_condition : ℕ :=
  606

theorem smallest_palindrome_prop :
  ∃ n, is_three_digit_palindrome n ∧ (103 * n - 300) ≥ 10000 ∧ (103 * n - 300) < 100000 
    ∧ ¬ is_palindrome (103 * n - 300) ∧ n = smallest_palindrome_condition :=
by
  use 606
  sorry

end smallest_palindrome_prop_l769_769237


namespace num_factors_of_180_multiple_of_15_l769_769735

def is_factor (n m : Nat) : Prop := m % n = 0
def is_multiple_of_15 (n : Nat) : Prop := is_factor 15 n

theorem num_factors_of_180_multiple_of_15 : 
  (Nat.filter (fun n => is_factor n 180 ∧ is_multiple_of_15 n) (Nat.range (180 + 1))).length = 6 :=
sorry

end num_factors_of_180_multiple_of_15_l769_769735


namespace hyperbola_asymptotes_l769_769590

theorem hyperbola_asymptotes (x y : ℝ) :
  (x^2 / 16 - y^2 / 9 = 1) → (y = 3/4 * x ∨ y = -3/4 * x) :=
by
  sorry

end hyperbola_asymptotes_l769_769590


namespace factors_of_180_multiple_of_15_l769_769753

/-- A factor of a number n is a positive integer which divides n without leaving a remainder. -/
def is_factor (n k : ℕ) : Prop :=
k > 0 ∧ n % k = 0

/-- A multiple of a number m is an integer which is divisible by m without leaving a remainder. -/
def is_multiple (k m : ℕ) : Prop :=
k % m = 0

/-- There are 6 factors of 180 that are also multiples of 15. -/
theorem factors_of_180_multiple_of_15 : 
  (Finset.filter (λ k, is_multiple k 15) { x | is_factor 180 x }.to_finset).card = 6 :=
by 
  sorry

end factors_of_180_multiple_of_15_l769_769753


namespace cds_total_l769_769351

theorem cds_total (dawn_cds : ℕ) (h1 : dawn_cds = 10) (h2 : ∀ kristine_cds : ℕ, kristine_cds = dawn_cds + 7) :
  dawn_cds + (dawn_cds + 7) = 27 :=
by
  sorry

end cds_total_l769_769351


namespace total_oranges_is_correct_l769_769528

-- Definitions based on the problem's conditions
def layer_count : ℕ := 6
def base_length : ℕ := 9
def base_width : ℕ := 6

-- Function to compute the number of oranges in a layer given the current dimensions
def oranges_in_layer (length width : ℕ) : ℕ :=
  length * width

-- Function to compute the total number of oranges in the stack
def total_oranges_in_stack (base_length base_width : ℕ) : ℕ :=
  oranges_in_layer base_length base_width +
  oranges_in_layer (base_length - 1) (base_width - 1) +
  oranges_in_layer (base_length - 2) (base_width - 2) +
  oranges_in_layer (base_length - 3) (base_width - 3) +
  oranges_in_layer (base_length - 4) (base_width - 4) +
  oranges_in_layer (base_length - 5) (base_width - 5)

-- The theorem to be proved
theorem total_oranges_is_correct : total_oranges_in_stack 9 6 = 154 := by
  sorry

end total_oranges_is_correct_l769_769528


namespace max_f_equals_5_div_4_l769_769268

-- Define the function f
def f (x : ℝ) : ℝ := (Real.cos (π / 2 + x)) + (Real.sin (π / 2 + x))^2

-- State the theorem
theorem max_f_equals_5_div_4 : ∃ x : ℝ, f x = 5 / 4 :=
by sorry

end max_f_equals_5_div_4_l769_769268


namespace wrapping_paper_area_correct_l769_769150

-- Conditions as given in the problem
variables (l w h : ℝ)
variable (hlw : l > w)

-- Definition of the area of the wrapping paper
def wrapping_paper_area (l w h : ℝ) : ℝ :=
  (l + 2 * h) * (w + 2 * h)

-- Proof statement
theorem wrapping_paper_area_correct (hlw : l > w) : 
  wrapping_paper_area l w h = l * w + 2 * l * h + 2 * w * h + 4 * h^2 :=
by
  sorry

end wrapping_paper_area_correct_l769_769150


namespace arithmetic_sequence_sum_l769_769917

theorem arithmetic_sequence_sum
  (a_n : ℕ → ℤ)
  (S_n : ℕ → ℤ)
  (n : ℕ)
  (a1 : ℤ)
  (d : ℤ)
  (h1 : a1 = 2)
  (h2 : a_n 5 = a_n 1 + 4 * d)
  (h3 : a_n 3 = a_n 1 + 2 * d)
  (h4 : a_n 5 = 3 * a_n 3) :
  S_n 9 = -54 := 
by  
  sorry

end arithmetic_sequence_sum_l769_769917


namespace molecular_weight_of_6_moles_l769_769107

-- Define the molecular weight of the compound
def molecular_weight : ℕ := 1404

-- Define the number of moles
def number_of_moles : ℕ := 6

-- The hypothesis would be the molecular weight condition
theorem molecular_weight_of_6_moles : number_of_moles * molecular_weight = 8424 :=
by sorry

end molecular_weight_of_6_moles_l769_769107


namespace determine_ratio_if_points_coincide_l769_769353

/--
Let  \(\triangle ABC\)  be an equilateral triangle and let  P  be a point on  \([AB]\).
 Q  is the point on  BC  such that  PQ  is perpendicular to  AB .
 R  is the point on  AC  such that  QR  is perpendicular to  BC .
 And  S  is the point on  AB  such that  RS  is perpendicular to  AC .
 Q'  is the point on  BC  such that  PQ'  is perpendicular to  BC .
 R'  is the point on  AC  such that  Q'R'  is perpendicular to  AC .
 And  S'  is the point on  AB  such that  R'S'  is perpendicular to  AB .
Determine  \(\frac{|PB|}{|AB|}\)  if  \(S = S'\).
-/
theorem determine_ratio_if_points_coincide (a b c P Q R S Q' R' S' : ℝ) (x PB AB : ℝ) :
  triangle_equi a b c ∧
  point_on_line_segment P a b ∧
  perp PQ AB ∧ point_on_line_segment Q b c ∧
  perp QR BC ∧ point_on_line_segment R a c ∧
  perp RS AC ∧ point_on_line_segment S a b ∧
  perp PQ' BC ∧ point_on_line_segment Q' b c ∧
  perp Q'R' AC ∧ point_on_line_segment R' a c ∧
  perp R'S' AB ∧ point_on_line_segment S' a b ∧
  S = S' →
  PB / AB = 1 := 
sorry

end determine_ratio_if_points_coincide_l769_769353


namespace no_positive_integer_m_such_that_S_m_plus_one_eq_4_S_m_l769_769209

def S (n : ℕ) : ℕ := nat.lcm_list (list.range (n+1))

theorem no_positive_integer_m_such_that_S_m_plus_one_eq_4_S_m :
  ¬ ∃ m : ℕ, 0 < m ∧ S (m + 1) = 4 * S m :=
by
  sorry

end no_positive_integer_m_such_that_S_m_plus_one_eq_4_S_m_l769_769209


namespace count_valid_N_under_500_l769_769299

def hasSolution (N : ℕ) (x : ℝ) : Prop :=
  N = x ^ (Real.floor x)

def validN (N : ℕ) : Prop :=
  ∃ x : ℝ, hasSolution N x

theorem count_valid_N_under_500 : 
  let N_set := {N : ℕ | N < 500 ∧ validN N}
  N_set.card = 287 := sorry

end count_valid_N_under_500_l769_769299


namespace maximize_volume_cone_height_l769_769156

-- Definitions of the given problem conditions
def slant_height : Real := 30
def volume_cone (r h : Real) : Real := (1 / 3) * Real.pi * r^2 * h

-- Maximizing the volume given the condition
theorem maximize_volume_cone_height :
  ∃ h : Real, (0 < h ∧ h < 30) ∧ (∀ h' : Real, 0 < h' ∧ h' < 30 → 
  volume_cone (Real.sqrt (slant_height^2 - h^2)) h ≤ volume_cone (Real.sqrt (slant_height^2 - h'^2)) h') ∧
  h = 10 * Real.sqrt 3 :=
sorry

end maximize_volume_cone_height_l769_769156


namespace size_of_sixth_doll_l769_769376

def nth_doll_size (n : ℕ) : ℝ :=
  243 * (2 / 3) ^ n

theorem size_of_sixth_doll : nth_doll_size 5 = 32 := by
  sorry

end size_of_sixth_doll_l769_769376


namespace count_valid_N_l769_769308

-- Definitions based on identified mathematical conditions
def valid_N (N : ℕ) : Prop :=
  (∃ x : ℚ, 0 ≤ floor x ∧ floor x < 5 ∧ x ^ (floor x).natAbs = N) ∧ N < 500

theorem count_valid_N : finset.card (finset.filter valid_N (finset.range 500)) = 287 :=
by sorry

end count_valid_N_l769_769308


namespace possible_n_values_l769_769366

def equation (x y z n : ℕ) : Prop := 2 * x + 2 * y + z = n

theorem possible_n_values (n : ℕ) (h₀ : n > 0) 
  (h₁ : ∃ T : Finset (ℕ × ℕ × ℕ), (∀ t ∈ T, let (x, y, z) := t in equation x y z n ∧ x > 0 ∧ y > 0 ∧ z > 0) ∧ T.card = 28) :
  n = 17 ∨ n = 18 := sorry

end possible_n_values_l769_769366


namespace pencils_placed_by_Joan_l769_769083

variable (initial_pencils : ℕ)
variable (total_pencils : ℕ)

theorem pencils_placed_by_Joan 
  (h1 : initial_pencils = 33) 
  (h2 : total_pencils = 60)
  : total_pencils - initial_pencils = 27 := 
by
  sorry

end pencils_placed_by_Joan_l769_769083


namespace log_solution_l769_769592

theorem log_solution (a : ℝ) (h : log 10 (a ^ 2 - 20 * a) = 2) : a = 10 + 10 * real.sqrt 2 ∨ a = 10 - 10 * real.sqrt 2 :=
by sorry

end log_solution_l769_769592


namespace scientific_notation_of_coronavirus_size_l769_769081

theorem scientific_notation_of_coronavirus_size :
  (0.000000125 : ℝ) = 1.25 * 10^(-7) :=
sorry

end scientific_notation_of_coronavirus_size_l769_769081


namespace inverse_of_inverse_at_9_l769_769057

noncomputable def f (x : ℝ) : ℝ := 4 * x + 5

noncomputable def f_inv (x : ℝ) : ℝ := (x - 5) / 4

theorem inverse_of_inverse_at_9 : f_inv (f_inv 9) = -1 :=
by
  sorry

end inverse_of_inverse_at_9_l769_769057


namespace max_trinomials_without_roots_l769_769938

-- Definitions of arithmetic progressions
def is_arith_progression (a : List ℝ) : Prop :=
  ∃ d : ℝ, ∀ i, 1 ≤ i ∧ i < a.length → a.get i + d = a.get (i + 1)

-- Definition of quadratic trinomial
def quadratic_trinomial (a b : ℝ) (x : ℝ) : ℝ :=
  x^2 + a * x + b

-- Formalization of the problem
theorem max_trinomials_without_roots :
  ∀ (a b : List ℝ), 
    a.length = 9 → 
    b.length = 9 → 
    is_arith_progression a → 
    is_arith_progression b → 
    (∃ x, ∑ i in List.range 9, quadratic_trinomial (a.get i) (b.get i) x = 0) → 
    ∃ n ≤ 4, ∀ (i : ℕ), i < 9 → (∃ x, quadratic_trinomial (a.get i) (b.get i) x ≠ 0) := 
sorry

end max_trinomials_without_roots_l769_769938


namespace gum_division_l769_769350

variable (John Cole Aubrey Maria : ℕ)
variable (Liam Emma : Unit)
variable (total_people : ℕ)

noncomputable def gum_shared (total_gum : ℕ) (persons : ℕ) : ℕ := total_gum / persons

theorem gum_division :
  John = 54 →
  Cole = 45 →
  Aubrey = 37 →
  Maria = 70 →
  total_people = 6 →
  gum_shared (John + Cole + Aubrey + Maria) total_people = 34 :=
by
  intros hJohn hCole hAubrey hMaria hTotalPeople
  dsimp [gum_shared]
  rw [hJohn, hCole, hAubrey, hMaria, hTotalPeople]
  norm_num -- evaluates the arithmetic directly to verify the division
  sorry

end gum_division_l769_769350


namespace count_N_lt_500_solution_exists_l769_769289

theorem count_N_lt_500_solution_exists:
  ∃ (N : ℕ), N < 500 ∧ (∃ (x : ℝ), x^floor x = N) = 287 :=
sorry

end count_N_lt_500_solution_exists_l769_769289


namespace kaleb_toy_count_l769_769889

theorem kaleb_toy_count:
  let saved_money := 39
  let toy_price := 8
  let discount := 0.20
  let allowance := 25
  let tax := 0.10
  let discounted_price := toy_price * (1 - discount)
  let final_price := discounted_price * (1 + tax)
  let total_money := saved_money + allowance
  let toy_count := total_money / final_price
  ⌊toy_count⌋ = 9 := by
  let saved_money := 39
  let toy_price := 8
  let discount := 0.20
  let allowance := 25
  let tax := 0.10
  let discounted_price := toy_price * (1 - discount)
  let final_price := discounted_price * (1 + tax)
  let total_money := saved_money + allowance
  let toy_count := total_money / final_price
  have h : ⌊toy_count⌋ = 9 := sorry
  exact h

end kaleb_toy_count_l769_769889


namespace angle_equality_l769_769916

-- declaring the types (triangle, point, angle) and useful methods
universe u
variables (α : Type u) [geometry α]

open_locale absolute_value

/-- Mathematically equivalent proof problem. -/
theorem angle_equality 
  (A B C H O D E F : α) 
  (triangle_ABC : is_triangle A B C) 
  (orthocenter_H : is_orthocenter H A B C) 
  (circumcenter_O : is_circumcenter O A B C) 
  (foot_D : is_foot D A B C) 
  (foot_E : is_foot E B A C) 
  (foot_F : is_foot F C A B) 
  (concyclic_AEHF : is_cyclic A E H F) :
  ∠ B A O = ∠ C A H := 
sorry

end angle_equality_l769_769916


namespace count_factors_of_180_multiple_of_15_l769_769716

def is_factor (x n : ℕ) := n % x = 0 -- Definition of a factor
def is_multiple (x k : ℕ) := x % k = 0 -- Definition of a multiple

theorem count_factors_of_180_multiple_of_15 :
  (finset.filter (λ x, is_multiple x 15) (finset.filter (λ x, is_factor x 180) (finset.range (180 + 1)))).card = 6 := by
  sorry

end count_factors_of_180_multiple_of_15_l769_769716


namespace relationship_of_areas_l769_769145

theorem relationship_of_areas (a b c : ℝ) (X Y Z : ℝ) (r : ℝ) :
  a = 13 ∧ b = 14 ∧ c = 15 ∧
  ∀ (s K : ℝ), s = (a + b + c) / 2 ∧ K = real.sqrt (s * (s - a) * (s - b) * (s - c)) ∧
  r = (a * b * c) / (4 * K) ∧ 
  X + Y + Z = real.pi * r ^ 2 - K ∧
  r = 8.125 ∧ K = 84 ∧
  Z = real.pi * (8.125) ^ 2 - 84 
  → X + Y = Z - 84 :=
by
  sorry

end relationship_of_areas_l769_769145


namespace sum_of_cubes_of_roots_l769_769586

def alpha := real.cbrt 17
def beta := real.cbrt 37
def gamma := real.cbrt 57

theorem sum_of_cubes_of_roots :
  r^3 + s^3 + t^3 = 107.5
  where
    card r s t : real,
    r + s + t = alpha + beta + gamma,
    rs + rt + st = alpha * beta + alpha * gamma + beta * gamma,
    rst = alpha * beta * gamma - 0.5
:= sorry

end sum_of_cubes_of_roots_l769_769586


namespace count_solutions_eq_287_l769_769295

noncomputable def count_solutions : ℕ :=
  (({n | ∃ x : ℝ, n < 500 ∧ (⌊x⌋.to_nat ≥ 0) ∧ x^(⌊x⌋.to_nat) = n} : set ℕ).to_finset.card)

theorem count_solutions_eq_287 : count_solutions = 287 :=
  sorry

end count_solutions_eq_287_l769_769295


namespace exists_a_b_l769_769893

theorem exists_a_b (n : ℕ) (h : 0 < n) : 
  ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ a = n^2 ∧ b = n - 1 ∧ (a^2 + a + 1) / (b^2 + b + 1) = n^2 + n + 1 := 
by
    let a := n^2
    let b := n - 1
    have ha : 0 < a := by sorry
    have hb : 0 < b := by sorry
    use [a, b]
    split
    · exact ha
    split
    · exact hb
    split
    · rfl
    split
    · rfl
    · sorry

end exists_a_b_l769_769893


namespace inequality_solution_set_l769_769454

theorem inequality_solution_set :
  {x : ℝ | (0 ≤ x ∧ x < 2) ∨ (x = 0)} = {x : ℝ | 0 ≤ x ∧ x < 2} :=
by
  sorry

end inequality_solution_set_l769_769454


namespace identify_genuine_coins_l769_769992

section IdentifyGenuineCoins

variables (coins : Fin 25 → ℝ) 
          (is_genuine : Fin 25 → Prop) 
          (is_counterfeit : Fin 25 → Prop)

-- Conditions
axiom coin_total : ∀ i, is_genuine i ∨ is_counterfeit i
axiom genuine_count : ∃ s : Finset (Fin 25), s.card = 22 ∧ ∀ i ∈ s, is_genuine i
axiom counterfeit_count : ∃ t : Finset (Fin 25), t.card = 3 ∧ ∀ i ∈ t, is_counterfeit i
axiom genuine_weight : ∃ w : ℝ, ∀ i, is_genuine i → coins i = w
axiom counterfeit_weight : ∃ c : ℝ, ∀ i, is_counterfeit i → coins i = c
axiom counterfeit_lighter : ∀ (w c : ℝ), (∃ i, is_genuine i → coins i = w) ∧ (∃ j, is_counterfeit j → coins j = c) → c < w

-- Theorem: Identifying 6 genuine coins using two weighings
theorem identify_genuine_coins : ∃ s : Finset (Fin 25), s.card = 6 ∧ ∀ i ∈ s, is_genuine i :=
sorry

end IdentifyGenuineCoins

end identify_genuine_coins_l769_769992


namespace percentage_change_area_l769_769443

theorem percentage_change_area (L B : ℝ) (hL : L > 0) (hB : B > 0) :
    let A := L * B
    let L' := L / 2
    let B' := 3 * B
    let A' := (L / 2) * (3 * B)
    (A' - A) / A * 100 = 50 :=
by
  let A := L * B
  let A' := (L / 2) * (3 * B)
  have h : (A' - A) / A * 100 = (1 / 2) * 100 := sorry
  exact h

end percentage_change_area_l769_769443


namespace correct_option_is_C_l769_769111

-- Definitions based on the problem conditions
def option_A : Prop := (-3 + (-3)) = 0
def option_B : Prop := (-3 - abs (-3)) = 0
def option_C (a b : ℝ) : Prop := (3 * a^2 * b - 4 * b * a^2) = - a^2 * b
def option_D (x : ℝ) : Prop := (-(5 * x - 2)) = -5 * x - 2

-- The theorem to be proved that option C is the correct calculation
theorem correct_option_is_C (a b : ℝ) : option_C a b :=
sorry

end correct_option_is_C_l769_769111


namespace annual_interest_rate_l769_769622

theorem annual_interest_rate
  (P : ℝ) (T : ℝ) (SI : ℝ) (R : ℝ)
  (hP : P = 10000)
  (hT : T = 1)
  (hSI : SI = 900)
  (hSI_eq : SI = P * R * T) :
  R = 0.09 :=
by {
  rw [hP, hT, hSI] at hSI_eq,
  sorry
}

end annual_interest_rate_l769_769622


namespace correct_operation_l769_769499

theorem correct_operation (a b : ℝ) :
  ¬ (a^2 + a^3 = a^6) ∧
  ¬ ((a*b)^2 = a*(b^2)) ∧
  ¬ ((a + b)^2 = a^2 + b^2) ∧
  ((a + b)*(a - b) = a^2 - b^2) := 
by
  sorry

end correct_operation_l769_769499


namespace sum_of_swapped_numbers_l769_769465

def swap_digits (n : ℕ) : ℕ :=
  match n / 100, (n / 10) % 10, n % 10 with
  | a, b, c => 100 * c + 10 * b + a

theorem sum_of_swapped_numbers (a b c : ℕ) 
  (h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h2 : set.of_list (digits 10 a ++ digits 10 b ++ digits 10 c) = {1, 2, 3, 4, 5, 6, 7, 8, 9})
  (h3 : a + b + c = 1665) :
  (swap_digits a) + (swap_digits b) + (swap_digits c) = 1665 :=
sorry

end sum_of_swapped_numbers_l769_769465


namespace total_skips_l769_769605

theorem total_skips (fifth throw : ℕ) (fourth throw : ℕ) (third throw : ℕ) (second throw : ℕ) (first throw : ℕ) :
  fifth throw = 8 →
  fourth throw = fifth throw - 1 →
  third throw = fourth throw + 3 →
  second throw = third throw / 2 →
  first throw = second throw - 2 →
  first throw + second throw + third throw + fourth throw + fifth throw = 33 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  rw [h₁, h₂, h₃, h₄, h₅]
  sorry

end total_skips_l769_769605


namespace sum_of_indices_l769_769997

theorem sum_of_indices (s : ℕ) (m : fin s → ℕ) (b : fin s → ℤ)
  (h1 : ∀ i, b i = 1 ∨ b i = -1)
  (h2 : ∀ i j, i ≠ j → m i ≠ m j)
  (h3 : ∀ i j, i < j → m i > m j)
  (h4 : ∑ i, b i * 3^(m i) = 2023) :
  ∑ i, m i = 8 :=
by sorry

end sum_of_indices_l769_769997


namespace correct_operation_l769_769492

variable {a b : ℝ}

def conditionA : Prop := a^2 + a^3 = a^6
def conditionB : Prop := (a * b)^2 = a * b^2
def conditionC : Prop := (a + b)^2 = a^2 + b^2
def conditionD : Prop := (a + b) * (a - b) = a^2 - b^2

theorem correct_operation : conditionD ∧ ¬conditionA ∧ ¬conditionB ∧ ¬conditionC := by
  sorry

end correct_operation_l769_769492


namespace size_of_sixth_doll_l769_769375

def nth_doll_size (n : ℕ) : ℝ :=
  243 * (2 / 3) ^ n

theorem size_of_sixth_doll : nth_doll_size 5 = 32 := by
  sorry

end size_of_sixth_doll_l769_769375


namespace _l769_769941

/-
Pat and Mat went on a trip. They set off after eight in the morning, at a time when the hour and minute hands on Pat's watch were pointing in exactly opposite directions. When they returned before noon, the hands on Pat's watch were again pointing in exactly opposite directions. Mat measured the duration of the trip with a stopwatch. Determine, with an accuracy of seconds, how long the trip lasted. Assume that both Pat's watch and Mat's stopwatch were accurate.
-/

noncomputable def hour_hand_speed := 0.5 -- degrees per minute
noncomputable def minute_hand_speed := 6 -- degrees per minute

-- Initial positions at 8:00
noncomputable def initial_hour_hand_position := 240.0 -- degrees
noncomputable def initial_minute_hand_position := 0.0 -- degrees

-- Calculate when the hands are opposite again after 8:00
noncomputable def time_after_eight : ℚ :=
  ((240.0 + 180) - initial_minute_hand_position) / (minute_hand_speed - hour_hand_speed)

-- Calculate when the hands are opposite again after 11:00
noncomputable def final_hour_hand_position := 330.0 -- degrees at 11:00
noncomputable def time_after_eleven : ℚ :=
  ((final_hour_hand_position + 180) - initial_minute_hand_position) / (minute_hand_speed - hour_hand_speed)

-- Duration of the trip
noncomputable def trip_duration_hours : ℚ :=
  (11 - 8) + (time_after_eleven - time_after_eight)

-- Lean Definition encapsulating the entire problem
def trip_duration : Prop :=
  trip_duration_hours = 3 + 16 / 60 + 22 / 3600

-- Lean theorem statement
example : trip_duration := 
by
  -- Definitions and calculations would lead to this verification
  sorry

end _l769_769941


namespace almost_involution_on_partitions_l769_769959

noncomputable def is_almost_involution {α : Type*} (f : α → α) (P : α → Prop) :=
  ∀ x, P x → ((f (f x) = x) ∨ (P x ∧ (x.sized = (n : ℕ) (λ k, k * (n * 2*n-1)/2) ∨
  (x.sized = (n : ℕ) (λ k, k * (n+1+2n)/2)))))

theorem almost_involution_on_partitions {
  α : Type* 
  f : α → α 
  h : Π (λ : ℕ), ∃ (x : α), x.size = (λ(n : ℕ), n*(2*n-1)/2) 
         ∨ ∃ (x : α), x.size = (λ(n : ℕ), n*(n+1+2n)/2)) 
  P : α → Prop,
  ∀ x, P x ↔ (P x ∧ (f (f x) = x) 
             ∨ (x.sized = (n : ℕ) (λ (k : ℕ), (k*(2*k-1))/2)) 
             ∨ (x.sized = (n : ℕ) (λ (k : ℕ), (k*(k + 1 + 2*k))/2))) :=
sorry

end almost_involution_on_partitions_l769_769959


namespace consistent_rectangle_sum_l769_769335

theorem consistent_rectangle_sum {a : ℕ × ℕ → ℕ} (h : ∀ (s : ℕ) (R₁ R₂ : Finset (ℕ × ℕ)), 
  R₁.card = s → R₂.card = s → (R₁ ∪ R₂).card ≤ s * 1000 → 
  R₁.sum (λ (p : ℕ × ℕ), a p) = R₂.sum (λ (p : ℕ × ℕ), a p)) : 
  ∀ s : ℕ, (∀ (R : Finset (ℕ × ℕ)), R.card = s → 
    (∃ k, ∀ p ∈ R, a p = k)) ↔ s = 1 :=
sorry

end consistent_rectangle_sum_l769_769335


namespace find_first_day_income_l769_769141

def income_4 (i2 i3 i4 i5 : ℕ) : ℕ := i2 + i3 + i4 + i5

def total_income_5 (average_income : ℕ) : ℕ := 5 * average_income

def income_1 (total : ℕ) (known : ℕ) : ℕ := total - known

theorem find_first_day_income (i2 i3 i4 i5 a income5 : ℕ) (h1 : income_4 i2 i3 i4 i5 = 1800)
  (h2 : a = 440)
  (h3 : total_income_5 a = income5)
  : income_1 income5 (income_4 i2 i3 i4 i5) = 400 := 
sorry

end find_first_day_income_l769_769141


namespace count_positive_factors_of_180_multiple_of_15_l769_769725

statement: ℕ
statement := sorry

theorem count_positive_factors_of_180_multiple_of_15 :
  let factors_of_180 := {d ∈ (finset.range 181).filter (λ x, 180 % x = 0)}
  let multiples_of_15 := (finset.range 181).filter (λ x, x % 15 = 0)
  let desired_factors := factors_of_180 ∩ multiples_of_15
  desired_factors.card = 6 :=
by
  sorry

end count_positive_factors_of_180_multiple_of_15_l769_769725


namespace quadrilateral_area_ratio_l769_769948

theorem quadrilateral_area_ratio (c : ℝ) (r : ℝ) 
  (h : QuadrilateralInscribed c) 
  (EG_is_diameter : is_diameter c r)
  (m_angle_FEG : measure_angle F E G = 15)
  (m_angle_GEF : measure_angle G E F = 60)
  (desired_expression : area_ratio (EFGH_area E F G H) (circle_area c) = (d + sqrt e) / (f * π)) :
  d + e + f = 7 :=
sorry

end quadrilateral_area_ratio_l769_769948


namespace find_angle_of_inclination_l769_769972

noncomputable def is_tangent_angle (α : Real) : Prop :=
  let x := λ t : Real => t * Real.cos α
  let y := λ t : Real => t * Real.sin α
  let x_circle := λ ϕ : Real => 4 + 2 * Real.cos ϕ
  let y_circle := λ ϕ : Real => 2 * Real.sin ϕ
  let dist := |4 * Real.tan α| / Real.sqrt (1 + (Real.tan α) ^ 2)
  α > π / 2 ∧ dist = 2

theorem find_angle_of_inclination :
  ∃ α : Real, is_tangent_angle α ∧ α = 5 * π / 6 :=
sorry

end find_angle_of_inclination_l769_769972


namespace probability_AM_less_than_AC_l769_769334

theorem probability_AM_less_than_AC (ABC : Type) [isosceles_right_triangle ABC] (A B C M : point ABC) 
(h : distance A C = 1)
(AB_hypotenuse : hypotenuse A B = B) : 
probability (M ∈ segment AB ∧ distance A M < distance A C) = real.sqrt 2 / 2 := 
sorry

end probability_AM_less_than_AC_l769_769334


namespace angle_between_c_and_c_plus_3d_is_90_l769_769902

variable {ℝ : Type*}
variables (c d : ℝ → ℝ)

def norm (v : ℝ → ℝ) : ℝ := real.sqrt (v (v 0) * v (v 0) + v (v 1) * v (v 1) + v (v 2) * v (v 2))

theorem angle_between_c_and_c_plus_3d_is_90
  (h : norm (fun t => c t + 3 * d t) = norm d) :
  vector_angle (fun t => c t) (fun t => c t + 3 * d t) = real.pi / 2 :=
begin
  sorry
end

end angle_between_c_and_c_plus_3d_is_90_l769_769902


namespace not_possible_perimeter_l769_769329

theorem not_possible_perimeter :
  ∀ (x : ℝ), 7 < x ∧ x < 43 → (18 + 25 + x ≠ 88) :=
by {
  intros x hx,
  sorry
}

end not_possible_perimeter_l769_769329


namespace problem_statement_l769_769302

noncomputable def countNs : Nat :=
  let N_values := {N : ℕ | ∃ x : ℝ, 0 < N ∧ N < 500 ∧ (x ^ Nat.floor x = N)}
  N_values.toFinset.card

theorem problem_statement :
  countNs = 287 := by
  sorry

end problem_statement_l769_769302


namespace total_skips_l769_769599

-- Definitions based on conditions
def fifth_throw := 8
def fourth_throw := fifth_throw - 1
def third_throw := fourth_throw + 3
def second_throw := third_throw / 2
def first_throw := second_throw - 2

-- Statement of the proof problem
theorem total_skips : first_throw + second_throw + third_throw + fourth_throw + fifth_throw = 33 := by
  sorry

end total_skips_l769_769599


namespace a4_coefficient_l769_769895

-- Define the expansion of (2x + 1)^5 in terms of (x+1)
def polynomial_expansion (x : ℝ) : ℝ :=
  (2*x + 1)^5

-- Define the generic expansion coefficients
def expansion_coefficients (x : ℝ) : ℝ :=
  ∑ i in range 6, (polynomial_expansion x) 

-- Define the specific value a_4
def value_a4 : ℝ :=
  -80

-- Statement of the proof problem
theorem a4_coefficient :
  ∀ (x : ℝ), expansion_coefficients x →
    let a4 := 5.choose(1) * 2^4 * (-1)^1
    in a4 = value_a4 :=
by sorry

end a4_coefficient_l769_769895


namespace area_of_triangle_BDF_l769_769891

def Midpoint (p1 p2 : Point) : Point :=
  ⟨(p1.x + p2.x) / 2, (p1.y + p2.y) / 2⟩

noncomputable def AreaTriangle (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

structure Point where
  x : ℝ
  y : ℝ

theorem area_of_triangle_BDF :
  let A : Point := ⟨18, 0⟩
  let C : Point := ⟨0, 0⟩
  let E : Point := ⟨0, 18⟩
  let B : Point := Midpoint A C
  let D : Point := Midpoint C E
  let F : Point := ⟨6, 6⟩ -- This is derived manually but not part of conditions, used as a definition for F.
  AreaTriangle B D F = 13.5 :=
by 
  -- Proof steps would go here
  sorry

end area_of_triangle_BDF_l769_769891


namespace proof_problem_l769_769572

variables (total_books : ℕ) (hist_percent sci_percent bios_percent myst_percent : ℕ)
          (hist_new_percent sci_new_percent bios_new_percent myst_new_percent : ℕ)
          (discount_percent : ℕ)

noncomputable def hist_books := total_books * hist_percent / 100
noncomputable def sci_books := total_books * sci_percent / 100
noncomputable def bios_books := total_books * bios_percent / 100
noncomputable def myst_books := total_books * myst_percent / 100

noncomputable def hist_new_books := hist_books * hist_new_percent / 100
noncomputable def sci_new_books := sci_books * sci_new_percent / 100
noncomputable def bios_new_books := bios_books * bios_new_percent / 100
noncomputable def myst_new_books := myst_books * myst_new_percent / 100

noncomputable def total_new_books := hist_new_books + sci_new_books + bios_new_books + myst_new_books
noncomputable def hist_new_fraction := hist_new_books / total_new_books

noncomputable def discounted_hist_new_books := hist_new_books * discount_percent / 100

theorem proof_problem :
  total_books = 2000 →
  hist_percent = 40 →
  sci_percent = 25 →
  bios_percent = 15 →
  myst_percent = 20 →
  hist_new_percent = 45 →
  sci_new_percent = 30 →
  bios_new_percent = 50 →
  myst_new_percent = 35 →
  discount_percent = 10 →
  hist_new_fraction = 9 / 20 ∧
  discounted_hist_new_books = 36 :=
by {
  intro h1 h2 h3 h4 h5 h6 h7 h8 h9 h10,
  sorry
}

end proof_problem_l769_769572


namespace hades_prevents_topmost_step_l769_769128

theorem hades_prevents_topmost_step (steps : ℕ) (stones : ℕ) 
  (initial_placement : ∀ i : ℕ, i < 500 → ∃ j, j < 500 ∧ (i = j))
  (sisyphus_moves : ∀ (n : ℕ), n < steps → ∃ m, m > n ∧ is_free m)
  (hades_moves : ∀ (n : ℕ), n > 0 → ∃ m, m < n ∧ is_free m) :
  ∀ n, n = 1001 → ¬ (∃ n, n = 1001 ∧ occupied n) :=
by sorry

end hades_prevents_topmost_step_l769_769128


namespace annual_interest_rate_l769_769311

-- Define the given values
def P1 := 5000
def I1 := 250
def P2 := 20000
def I2 := 1000
def T := 1

-- State the problem: proving the annual interest rate is 0.05
theorem annual_interest_rate :
  (I1 = P1 * R * T) →
  (I2 = P2 * R * T) →
  R = 0.05 :=
by
  intros h1 h2
  sorry -- Proof is omitted as per instructions.

end annual_interest_rate_l769_769311


namespace function_continuity_l769_769689

variable (f : ℝ → ℝ)

theorem function_continuity {f : ℝ → ℝ} 
  (h1 : ∀ a > 1, continuous (λ x, f x + f (a * x))) : 
  continuous f :=
by
  sorry

end function_continuity_l769_769689


namespace percentage_change_in_area_of_rectangle_l769_769440

theorem percentage_change_in_area_of_rectangle
  (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  let A1 := L * B
  let A2 := (L / 2) * (3 * B)
  let percentage_change := ((A2 - A1) / A1) * 100
  percentage_change = 50 := by
  let A1 := L * B
  let A2 := (L / 2) * (3 * B)
  let percentage_change := ((A2 - A1) / A1) * 100
  calc percentage_change
      = ((A2 - A1) / A1) * 100 : rfl
  ... = ((((L / 2) * (3 * B)) - (L * B)) / (L * B)) * 100 : by rw [A1, A2]
  ... = ((3 / 2 * A1 - A1) / A1) * 100 : by rw [A2, ←mul_assoc, mul_div_cancel' _ (two_ne_zero : 2 ≠ 0)]
  ... = ((3 / 2 - 1) * A1 / A1) * 100 : by simp only [mul_sub, mul_div_cancel' _ (ne_of_gt hL)]
  ... = (1 / 2) * 100 : by rw [div_mul_cancel' (ne_of_gt (lt_of_lt_of_le zero_lt_one (le_of_lt ((mul_lt_iff_lt_one_left (lt_of_lt_of_le zero_lt_one hL)).mpr (half_pos (lt_of_le_of_ne (le_of_lt (lt_of_lt_of_le (lt_one_mul_self (lt_of_le_of_lt (zero_le_two) (lt_add_one 1))) (lt_of_lt_of_le (two_le_one) (lt_one 2))))))))) so))
  ... = 50 : rfl

end percentage_change_in_area_of_rectangle_l769_769440


namespace four_digit_numbers_count_l769_769103

theorem four_digit_numbers_count : 
  let digits := [1, 2, 3, 4] in
  let is_valid_unit_digit := λ d, d ≠ 2 in
  let possibilities := 
    λ (units_digit : Nat), 
    let remaining_digits := (digits.erase units_digit) in
      remaining_digits.length * 
      (remaining_digits.erase remaining_digits[0]!).length * 
      (remaining_digits.erase remaining_digits[1]!).length 
  in
  (digits.filter is_valid_unit_digit).map possibilities |> List.sum = 18 := 
by
  sorry

end four_digit_numbers_count_l769_769103


namespace find_x_l769_769281

variable (x : ℝ)

def a : ℝ × ℝ × ℝ := (-1, 2, 1 / 2)
def b (x : ℝ) : ℝ × ℝ × ℝ := (-3, x, 2)

def dot_product (v w : ℝ × ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2 + v.3 * w.3

theorem find_x (h : dot_product (a) (b x) = 0) : x = -2 :=
by
  sorry

end find_x_l769_769281


namespace total_skips_l769_769606

theorem total_skips (fifth throw : ℕ) (fourth throw : ℕ) (third throw : ℕ) (second throw : ℕ) (first throw : ℕ) :
  fifth throw = 8 →
  fourth throw = fifth throw - 1 →
  third throw = fourth throw + 3 →
  second throw = third throw / 2 →
  first throw = second throw - 2 →
  first throw + second throw + third throw + fourth throw + fifth throw = 33 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  rw [h₁, h₂, h₃, h₄, h₅]
  sorry

end total_skips_l769_769606


namespace range_of_k_if_f_has_two_zeros_l769_769685

noncomputable def f (k x : ℝ) : ℝ := (Real.log x) / x - k * x

def g (x : ℝ) : ℝ := (Real.log x) / (x * x)

theorem range_of_k_if_f_has_two_zeros :
  (∃ a b ∈ Icc (1 / Real.exp 1) (Real.exp 2), f k a = 0 ∧ f k b = 0 ∧ a ≠ b) →
  ∃ k : ℝ, ∀ k, (2 / Real.exp 4) ≤ k ∧ k < (1 / (2 * Real.exp 1)) :=
sorry

end range_of_k_if_f_has_two_zeros_l769_769685


namespace find_a_7_l769_769078

def seq_a (a : ℕ → ℝ) : Prop :=
  (a 0 = 1) ∧ ∀ n, a (n + 1) = (9 / 4) * (a n) + (3 / 4) * sqrt (9^n - (a n)^2)

noncomputable def a_7_val : ℝ := (39402 + 10935 * sqrt 7) / 64

theorem find_a_7 (a : ℕ → ℝ) (h : seq_a a) : 
  a 7 = a_7_val := sorry

end find_a_7_l769_769078


namespace MrsHiltCanTakeFriendsToMovies_l769_769010

def TotalFriends : ℕ := 15
def FriendsCantGo : ℕ := 7
def FriendsCanGo : ℕ := 8

theorem MrsHiltCanTakeFriendsToMovies : TotalFriends - FriendsCantGo = FriendsCanGo := by
  -- The proof will show that 15 - 7 = 8.
  sorry

end MrsHiltCanTakeFriendsToMovies_l769_769010


namespace unique_f_exists_l769_769239

def d (n : ℕ) : ℕ := finset.card (finset.filter (λ x, n % x = 0) (finset.range (n + 1)))

noncomputable def f : ℕ → ℕ := sorry

theorem unique_f_exists (f : ℕ → ℕ) :
  (∀ x : ℕ, d (f x) = x) ∧
  (∀ x y : ℕ, f (x * y) ∣ (x - 1) * y^(x * y - 1) * f x) ∧
  f 1 = 1 ∧
  (∀ p m : ℕ, p.prime → f (p^m) = p^(p^m - 1)) ∧
  (∀ x y : ℕ, nat.coprime x y → f (x*y) = f x * f y) := sorry

end unique_f_exists_l769_769239


namespace product_of_a_n_l769_769625

def a_n (n : ℕ) : ℚ :=
  if h : n ≥ 4 then (↑((n + 1)^3 - 1) / (n * (n^3 - 1))) else 0

theorem product_of_a_n :
  ∏ (n : ℕ) in (Finset.range 96).filter (λ k, k ≥ 4).map (λ k, k + 4), a_n n = 962 / (Nat.factorial 98) :=
by
  sorry

end product_of_a_n_l769_769625


namespace max_angle_line_perpendicular_l769_769866

open RealInnerProductSpace

variables {α β : Plane}
variables {A : Point}
variables (h_intersection : intersects α β)
variables (h_A_on_intersection : on_intersection_line A α β h_intersection)
variables {l : Line}
variables (h_l_in_alpha : lies_in_plane l α)
variables (h_l_through_A : passes_through l A)

theorem max_angle_line_perpendicular (h_l_in_alpha : lies_in_plane l α)
  (h_l_through_A : passes_through l A) : 
  ∃ l_perp, perpendicular (intersection_line α β h_intersection) l_perp ∧
  forms_largest_angle_with_plane l_perp β :=
sorry

end max_angle_line_perpendicular_l769_769866


namespace percentage_change_area_l769_769420

theorem percentage_change_area (L B : ℝ) :
  let A_original := L * B
  let A_new := (L / 2) * (3 * B)
  (A_new - A_original) / A_original * 100 = 50 := by
  sorry

end percentage_change_area_l769_769420


namespace factors_of_180_multiple_of_15_count_l769_769805

theorem factors_of_180_multiple_of_15_count :
  ∃ n : Nat, n = 6 ∧ ∀ x : Nat, x ∣ 180 → 15 ∣ x → x > 0 :=
by
  sorry

end factors_of_180_multiple_of_15_count_l769_769805


namespace annual_sparkling_water_cost_l769_769384

theorem annual_sparkling_water_cost :
  (let cost_per_bottle := 2.00
       nights_per_year := 365
       fraction_bottle_per_night := 1 / 5
       bottles_per_year := nights_per_year * fraction_bottle_per_night in
   bottles_per_year * cost_per_bottle = 146.00) :=
by
  -- This is where the actual proof would go.
  sorry

end annual_sparkling_water_cost_l769_769384


namespace maximal_possible_edges_l769_769328

noncomputable def graph_problem : Prop :=
  ∀ (G : SimpleGraph (Fin 300)), (∀ v w : Fin 300, v ≠ w → G.Adj v w → G.degree v ≠ G.degree w) →
  G.edgeFinset.card ≤ 42550

theorem maximal_possible_edges : graph_problem :=
sorry

end maximal_possible_edges_l769_769328


namespace volume_of_given_tetrahedron_l769_769127

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def volume_of_tetrahedron (A1 A2 A3 A4 : Point3D) : ℝ :=
  let v1 := (A2.x - A1.x, A2.y - A1.y, A2.z - A1.z)
  let v2 := (A3.x - A1.x, A3.y - A1.y, A3.z - A1.z)
  let v3 := (A4.x - A1.x, A4.y - A1.y, A4.z - A1.z)
  let scalar_triple_product :=
    v1.1 * (v2.2 * v3.3 - v2.3 * v3.2) -
    v1.2 * (v2.1 * v3.3 - v2.3 * v3.1) +
    v1.3 * (v2.1 * v3.2 - v2.2 * v3.1)
  (1 / 6) * |scalar_triple_product|

theorem volume_of_given_tetrahedron :
  let A1 := Point3D.mk (-3) 4 (-7)
  let A2 := Point3D.mk 1 5 (-4)
  let A3 := Point3D.mk (-5) (-2) 0
  let A4 := Point3D.mk 2 5 4
  volume_of_tetrahedron A1 A2 A3 A4 = 151 / 6 :=
by
  sorry

end volume_of_given_tetrahedron_l769_769127


namespace fraction_rounded_to_three_decimal_places_l769_769042

theorem fraction_rounded_to_three_decimal_places :
  Real.Round (5 / 11) 3 = 0.455 :=
sorry

end fraction_rounded_to_three_decimal_places_l769_769042


namespace midtown_academy_absent_students_l769_769007

theorem midtown_academy_absent_students 
  (total_students : ℕ)
  (boys : ℕ)
  (girls : ℕ)
  (absent_boys_ratio : ℚ)
  (absent_girls_ratio : ℚ)
  (h1 : total_students = 150)
  (h2 : boys = 90)
  (h3 : girls = 60)
  (h4 : absent_boys_ratio = 1 / 6)
  (h5 : absent_girls_ratio = 1 / 4) :
  let absent_boys := absent_boys_ratio * boys in
  let absent_girls := absent_girls_ratio * girls in
  let total_absent := absent_boys + absent_girls in
  let percent_absent := total_absent / total_students * 100 in
  percent_absent = 20 :=
by
  sorry

end midtown_academy_absent_students_l769_769007


namespace factors_of_180_l769_769768

-- Define a predicate to check if a number is a factor of another number
def is_factor (m n : ℕ) : Prop := n % m = 0

-- Define a predicate to check if a number is a multiple of another number
def is_multiple (m n : ℕ) : Prop := m % n = 0

-- Define the problem
theorem factors_of_180 : {n : ℕ // is_factor n 180 ∧ is_multiple n 15}.card = 6 :=
by sorry

end factors_of_180_l769_769768


namespace max_slope_proof_l769_769253

-- Define the conditions for x and y
variables (x y : ℝ)
def satisfies_equation : Prop := x^2 + y^2 - 6*x - 6*y + 12 = 0

-- Define the maximum slope function
noncomputable def max_slope (x y : ℝ) (h : satisfies_equation x y) : ℝ :=
  3 + 2 * Real.sqrt 2

-- Prove the statement
theorem max_slope_proof :
  ∀ (x y : ℝ), satisfies_equation x y → (max_slope x y (by assumption) = 3 + 2 * Real.sqrt 2) :=
by
  auto
  sorry

end max_slope_proof_l769_769253


namespace marilyn_bottle_caps_start_l769_769926

-- Definitions based on the conditions
def initial_bottle_caps (X : ℕ) := X  -- Number of bottle caps Marilyn started with
def shared_bottle_caps := 36           -- Number of bottle caps shared with Nancy
def remaining_bottle_caps := 15        -- Number of bottle caps left after sharing

-- Theorem statement: Given the conditions, show that Marilyn started with 51 bottle caps
theorem marilyn_bottle_caps_start (X : ℕ) 
  (h1 : initial_bottle_caps X - shared_bottle_caps = remaining_bottle_caps) : 
  X = 51 := 
sorry  -- Proof omitted

end marilyn_bottle_caps_start_l769_769926


namespace how_many_kids_joined_l769_769093

theorem how_many_kids_joined (original_kids : ℕ) (new_kids : ℕ) (h : original_kids = 14) (h1 : new_kids = 36) :
  new_kids - original_kids = 22 :=
by
  sorry

end how_many_kids_joined_l769_769093


namespace joanne_part_time_job_hourly_wage_l769_769349

-- Definitions based on conditions
def main_job_hourly_wage : ℝ := 16.00
def main_job_hours_per_day : ℕ := 8
def part_time_job_hours_per_day : ℕ := 2
def total_weekly_earnings : ℝ := 775.00
def work_days_per_week : ℕ := 5
def expected_part_time_job_hourly_wage : ℝ := 13.50

-- Problem statement
theorem joanne_part_time_job_hourly_wage :
  let main_job_weekly_earnings := main_job_hourly_wage * main_job_hours_per_day * (work_days_per_week : ℝ),
      part_time_job_weekly_earnings := total_weekly_earnings - main_job_weekly_earnings,
      part_time_job_total_hours := part_time_job_hours_per_day * (work_days_per_week : ℕ),
      part_time_job_hourly_wage := part_time_job_weekly_earnings / (part_time_job_total_hours : ℝ)
  in part_time_job_hourly_wage = expected_part_time_job_hourly_wage :=
  by
  sorry

end joanne_part_time_job_hourly_wage_l769_769349


namespace sufficient_condition_A_B_disjoint_A_B_l769_769255

def set_A : set ℝ := { x : ℝ | x^2 - x - 2 < 0 }
def set_B (a : ℝ) : set ℝ := { x : ℝ | x^2 - (2*a + 6)*x + a^2 + 6*a <= 0 }

-- Part 1
theorem sufficient_condition_A_B (a : ℝ) : 
  (∀ x, x ∈ set_A → x ∈ set_B a) ∧ (∃ x, x ∉ set_A ∧ x ∈ set_B a) → 
  -4 ≤ a ∧ a ≤ -1 := 
sorry

-- Part 2
theorem disjoint_A_B (a : ℝ) : 
  (∀ x, x ∈ set_A → x ∉ set_B a) → 
  a ≤ -7 ∨ a ≥ 2 := 
sorry

end sufficient_condition_A_B_disjoint_A_B_l769_769255


namespace factory_car_production_l769_769550

theorem factory_car_production :
  let cars_yesterday := 60
  let cars_today := 2 * cars_yesterday
  let total_cars := cars_yesterday + cars_today
  total_cars = 180 :=
by
  sorry

end factory_car_production_l769_769550


namespace bisection_of_triangle_l769_769971

theorem bisection_of_triangle : 
  ∃! (L : StraightLine) (a : ℝ), (a > 0) ∧ 
    (∃ (AD AE : ℝ), 
      (AD = a) ∧ 
      (let AB BC AC := (8 : ℝ), 6, 10 in
       let b := AC - AE in
       let EC := AE in
       let d := AB - AD in
       (a + b = EC + d + 6) ∧ 
       (a * (d - a + 6) = 10 * d))) :=
begin
  sorry
end

end bisection_of_triangle_l769_769971


namespace product_result_eq_v_l769_769191

theorem product_result_eq_v : 
  ∃ v : ℚ, (∏ n in finset.range 10 + 1, (n * (n + 3)) / (n + 5)^2) = v := by
  -- We indicate the starting of the proof here
  sorry  -- The proof steps are omitted as per the instructions.

end product_result_eq_v_l769_769191


namespace taco_salad_cost_correct_l769_769238

variable (cost_hamburger cost_french_fries cost_lemonade platters_hamburger platters_french_fries platters_lemonade payments_friends total_friends : ℕ)
variable [decidable_eq ℚ]

def total_known_cost : ℚ :=
  platters_hamburger * cost_hamburger + platters_french_fries * cost_french_fries + platters_lemonade * cost_lemonade

def total_amount_collected : ℚ :=
  total_friends * payments_friends

def taco_salad_cost : ℚ :=
  total_amount_collected - total_known_cost

theorem taco_salad_cost_correct
  (cost_hamburger = 5) (cost_french_fries = 2.5) (cost_lemonade = 2)
  (platters_hamburger = 5) (platters_french_fries = 4) (platters_lemonade = 5)
  (total_friends = 5) (payments_friends = 11) :
  taco_salad_cost cost_hamburger cost_french_fries cost_lemonade platters_hamburger platters_french_fries platters_lemonade payments_friends total_friends = 10 :=
by
  sorry

end taco_salad_cost_correct_l769_769238


namespace lisa_walks_distance_per_minute_l769_769377

-- Variables and conditions
variable (d : ℤ) -- distance that Lisa walks each minute (what we're solving for)
variable (daily_distance : ℤ) -- distance that Lisa walks each hour
variable (total_distance_in_two_days : ℤ := 1200) -- total distance in two days
variable (hours_per_day : ℤ := 1) -- one hour per day

-- Given conditions
axiom walks_for_an_hour_each_day : ∀ (d: ℤ), daily_distance = d * 60
axiom walks_1200_meters_in_two_days : ∀ (d: ℤ), total_distance_in_two_days = 2 * daily_distance

-- The theorem we want to prove
theorem lisa_walks_distance_per_minute : (d = 10) :=
by
  -- TODO: complete the proof
  sorry

end lisa_walks_distance_per_minute_l769_769377


namespace total_cars_made_in_two_days_l769_769556

-- Define the number of cars made yesterday.
def cars_yesterday : ℕ := 60

-- Define the number of cars made today, which is twice the number of cars made yesterday.
def cars_today : ℕ := 2 * cars_yesterday

-- Define the total number of cars made over the two days.
def total_cars : ℕ := cars_yesterday + cars_today

-- Proof statement: Prove that the total number of cars made in these two days is 180.
theorem total_cars_made_in_two_days : total_cars = 180 := by
  -- Here, we would provide the proof, but we'll use sorry to skip it.
  sorry

end total_cars_made_in_two_days_l769_769556


namespace jon_q_public_inheritance_l769_769009

theorem jon_q_public_inheritance
  (x : ℝ)
  (donation : ℝ := 0.05 * x)
  (after_donation : ℝ := x - donation)
  (federal_tax : ℝ := 0.25 * after_donation)
  (after_federal : ℝ := after_donation - federal_tax)
  (state_tax : ℝ := 0.12 * after_federal)
  (total_taxes : ℝ := federal_tax + state_tax)
  (total_paid : ℝ := 15000)
  (tax_equation : total_taxes = total_paid) :
  x ≈ 46400 := by
  sorry

end jon_q_public_inheritance_l769_769009


namespace factors_of_180_multiples_of_15_l769_769795

theorem factors_of_180_multiples_of_15 :
  (finset.filter (λ x : ℕ, x % 15 = 0) (finset.filter (λ x : ℕ, 180 % x = 0) (finset.range (180 + 1)))).card = 6 :=
sorry

end factors_of_180_multiples_of_15_l769_769795


namespace sum_le_n_div_3_l769_769400

theorem sum_le_n_div_3 {n : ℕ} {x : Fin n → ℝ} 
  (hx1 : ∀ i, x i ∈ Set.Icc (-1 : ℝ) 1) 
  (hx2 : ∑ i, (x i)^3 = 0) : 
  ∑ i, x i ≤ n / 3 := 
sorry

end sum_le_n_div_3_l769_769400


namespace spider_footwear_order_l769_769166

theorem spider_footwear_order :
  ∃ n : ℕ, n = 81729648000 ∧
  ∀ (legs : fin 8), (8.shocks + 8.shoes).perm (16.choose 8) = (16! / 2^8) := sorry

end spider_footwear_order_l769_769166


namespace spheres_in_base_l769_769933

theorem spheres_in_base (n : ℕ) (T_n : ℕ) (total_spheres : ℕ) :
  (total_spheres = 165) →
  (total_spheres = (1 / 6 : ℚ) * ↑n * ↑(n + 1) * ↑(n + 2)) →
  (T_n = n * (n + 1) / 2) →
  n = 9 →
  T_n = 45 :=
by
  intros _ _ _ _
  sorry

end spheres_in_base_l769_769933


namespace square_area_from_diagonal_l769_769541

theorem square_area_from_diagonal (d : ℝ) (h : d = 8 * Real.sqrt 2) : (d / Real.sqrt 2) ^ 2 = 64 :=
by
  sorry

end square_area_from_diagonal_l769_769541


namespace symmetric_point_proof_l769_769063

def symmetric_point (P : ℝ × ℝ) (line : ℝ → ℝ) : ℝ × ℝ := sorry

theorem symmetric_point_proof :
  symmetric_point (2, 5) (λ x => 1 - x) = (-4, -1) := sorry

end symmetric_point_proof_l769_769063


namespace joining_fee_per_person_l769_769888

variables (F : ℝ)
variables (family_members : ℕ) (monthly_cost_per_person : ℝ) (john_yearly_payment : ℝ)

def total_cost (F : ℝ) (family_members : ℕ) (monthly_cost_per_person : ℝ) : ℝ :=
  family_members * (F + 12 * monthly_cost_per_person)

theorem joining_fee_per_person :
  (family_members = 4) →
  (monthly_cost_per_person = 1000) →
  (john_yearly_payment = 32000) →
  john_yearly_payment = 0.5 * total_cost F family_members monthly_cost_per_person →
  F = 4000 :=
by
  intros h_family h_monthly_cost h_yearly_payment h_eq
  sorry

end joining_fee_per_person_l769_769888


namespace original_cube_volume_l769_769525

theorem original_cube_volume (V₂ : ℝ) (s : ℝ) (h₀ : V₂ = 216) (h₁ : (2 * s) ^ 3 = V₂) : s ^ 3 = 27 := by
  sorry

end original_cube_volume_l769_769525


namespace time_to_cross_bridge_l769_769174

-- Given conditions
def train_length : ℕ := 150
def bridge_length : ℕ := 225
def train_speed_kmh : ℕ := 45

-- Proving the time taken for the train to cross the bridge
theorem time_to_cross_bridge : 
  let total_distance := train_length + bridge_length in
  let speed_mps := (train_speed_kmh * 1000) / 3600 in
  (total_distance / speed_mps) = 30 := 
by
  sorry

end time_to_cross_bridge_l769_769174


namespace power_equality_l769_769946

noncomputable def log_a (a x : ℝ) : ℝ := log x / log a

theorem power_equality (a x y z : ℝ) 
  (h1 : a > 0) (h2 : a ≠ 1) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hx1 : x ≠ 1) (hy1 : y ≠ 1) (hz1 : z ≠ 1)
  (h : (x * (y + z - x)) / (log_a a x) = (y * (x + z - y)) / (log_a a y) 
       ∧ (y * (x + z - y)) / (log_a a y) = (z * (x + y - z)) / (log_a a z)) : 
  (x^y * y^x = z^x * x^z ∧ z^x * x^z = y^z * z^y) := 
sorry

end power_equality_l769_769946


namespace problem1_problem2_l769_769194

-- Problem 1: Prove that √18 - √8 + √2 = 2√2
theorem problem1 : sqrt 18 - sqrt 8 + sqrt 2 = 2 * sqrt 2 := 
  sorry

-- Problem 2: Prove that (√48 - √12) ÷ √3 = 2
theorem problem2 : (sqrt 48 - sqrt 12) / sqrt 3 = 2 :=
  sorry

end problem1_problem2_l769_769194


namespace boxes_containing_neither_l769_769149

theorem boxes_containing_neither
  (total_boxes : ℕ := 15)
  (boxes_with_markers : ℕ := 9)
  (boxes_with_crayons : ℕ := 5)
  (boxes_with_both : ℕ := 4) :
  (total_boxes - ((boxes_with_markers - boxes_with_both) + (boxes_with_crayons - boxes_with_both) + boxes_with_both)) = 5 := by
  sorry

end boxes_containing_neither_l769_769149


namespace probability_of_x_squared_less_than_y_in_rectangle_l769_769161

noncomputable def probability_x_squared_less_than_y : ℝ := (3 - Real.sqrt 3) / 12

theorem probability_of_x_squared_less_than_y_in_rectangle :
  let rect_area := 4 * 3 in
  ∃ (x y : ℝ), 0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 3 ∧ (∫ 0 (Real.sqrt 3), x^2) / rect_area = probability_x_squared_less_than_y :=
sorry

end probability_of_x_squared_less_than_y_in_rectangle_l769_769161


namespace irreducible_fraction_l769_769401

theorem irreducible_fraction {n : ℕ} : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 :=
by
  sorry

end irreducible_fraction_l769_769401


namespace right_triangle_area_l769_769863

theorem right_triangle_area :
  ∀ (a b : ℕ), 
  a = 60 * 12 → b = 80 * 12 → 
  ∃ (area : ℕ), area = (1/2 : ℚ) * a * b ∧ area = 345600 := 
by
  intros a b h1 h2
  use (1/2 : ℚ) * a * b
  rw [h1, h2]
  norm_num
  sorry

end right_triangle_area_l769_769863


namespace no_20_odd_rows_15_odd_columns_l769_769663

theorem no_20_odd_rows_15_odd_columns (n : ℕ) (table : ℕ → ℕ → bool) (cross_count 
  : ℕ) 
  (odd_rows : ℕ → bool) 
  (odd_columns : ℕ → bool) :
  (∀ i, i < n → (odd_rows i = true ↔ ∃ j, j < n ∧ table i j = true ∧ cross_count = 20))
  → (∀ j, j < n → (odd_columns j = true ↔ ∃ i, i < n ∧ table i j = true ∧ cross_count = 15))
  → false := 
sorry

end no_20_odd_rows_15_odd_columns_l769_769663


namespace sum_of_reciprocals_l769_769458

theorem sum_of_reciprocals (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x + y = 5 * x * y) : (1 / x) + (1 / y) = 5 :=
by
  sorry

end sum_of_reciprocals_l769_769458


namespace area_of_overlapping_squares_l769_769629

theorem area_of_overlapping_squares :
  ∀ (s : ℝ), (s = 3) →
  let area_of_one_square := s
  let total_area_without_overlap := 4 * area_of_one_square
  let overlapping_area := 6 * (s / 16)
  let net_area := total_area_without_overlap - overlapping_area in
  net_area = 10.875 := by
  intros s hs
  let area_of_one_square := s
  let total_area_without_overlap := 4 * area_of_one_square
  let overlapping_area := 6 * (s / 16)
  let net_area := total_area_without_overlap - overlapping_area
  sorry

end area_of_overlapping_squares_l769_769629


namespace find_BD_l769_769023

-- Defining the variables and constants used in the problem
variable (a : ℝ) (BC : ℝ := 3) (AD : ℝ := 1)

-- Condition: Right triangle ABC with sides AC = a, BC = 3
def AB := Real.sqrt (a^2 + BC^2)

-- Condition: Right triangle ABD with sides AD = 1 and hypotenuse AB
variable (BD : ℝ)
def AB' := Real.sqrt (AD^2 + BD^2)

theorem find_BD : BD = Real.sqrt (a^2 + 8) :=
by
  -- Using the equality of the hypotenuse squared from both triangles
  have h_eq : AB^2 = AB'^2 :=
    by
      simp [AB, AB', AD, BC]
      sorry -- Steps involve solving the equality

  sorry -- This is where the actual proof steps would go.

end find_BD_l769_769023


namespace proof_problem_l769_769973

def line_intersection_x_axis (a : ℝ) (ha : a > 0) : Point := (1 / a, 0)
def line_intersection_y_axis (a : ℝ) (ha : a > 0) : Point := (0, a)

def triangle_area (a : ℝ) (ha : a > 0) : ℝ :=
  1 / 2 * a * (1 / a)

def distance_AB (a : ℝ) (ha : a > 0) : ℝ :=
  Real.sqrt (a^2 + (1 / a)^2)

def chord_length_CD (a : ℝ) (ha : a > 0) : ℝ :=
  2 * Real.sqrt (1 - (1 / (a^2 + (1 / a)^2)))

theorem proof_problem (a : ℝ) (ha : a > 0) :
  (triangle_area a ha = 1 / 2) ∧ (¬ ∃ b > 0, distance_AB b (by linarith) < chord_length_CD b (by linarith)) :=
by
  sorry

end proof_problem_l769_769973


namespace num_two_digit_numbers_l769_769284

-- Define the set of given digits
def digits : Finset ℕ := {0, 2, 5}

-- Define the function that counts the number of valid two-digit numbers
def count_two_digit_numbers (d : Finset ℕ) : ℕ :=
  (d.erase 0).card * (d.card - 1)

theorem num_two_digit_numbers : count_two_digit_numbers digits = 4 :=
by {
  -- sorry placeholder for the proof
  sorry
}

end num_two_digit_numbers_l769_769284


namespace total_turnips_grown_l769_769006

theorem total_turnips_grown 
  (melanie_turnips : ℕ) 
  (benny_turnips : ℕ) 
  (jack_turnips : ℕ) 
  (lynn_turnips : ℕ) : 
  melanie_turnips = 1395 ∧
  benny_turnips = 11380 ∧
  jack_turnips = 15825 ∧
  lynn_turnips = 23500 → 
  melanie_turnips + benny_turnips + jack_turnips + lynn_turnips = 52100 :=
by
  intros h
  rcases h with ⟨hm, hb, hj, hl⟩
  sorry

end total_turnips_grown_l769_769006


namespace nearly_uniform_coloring_exists_unique_l769_769135

theorem nearly_uniform_coloring_exists_unique {n k : ℕ} (h : k ≤ n) :
  ∃ f : Fin n → Prop, (∀ m : ℕ, ∀ M1 M2 : Finset (Fin n), M1.card = m → M2.card = m → (M1.filter f).card ≤ (M2.filter f).card + 1 ∧ (M2.filter f).card ≤ (M1.filter f).card + 1) ∧ (∀ g : Fin n → Prop, (∀ m : ℕ, ∀ M1 M2 : Finset (Fin n), M1.card = m → M2.card = m → (M1.filter g).card ≤ (M2.filter g).card + 1 ∧ (M2.filter g).card ≤ (M1.filter g).card + 1) → (∃ i : ℕ, ∀ j : ℕ, f ((i + j) % n) = g ((i + j) % n))) :=
sorry

end nearly_uniform_coloring_exists_unique_l769_769135


namespace factors_of_180_multiples_of_15_l769_769793

theorem factors_of_180_multiples_of_15 :
  (finset.filter (λ x : ℕ, x % 15 = 0) (finset.filter (λ x : ℕ, 180 % x = 0) (finset.range (180 + 1)))).card = 6 :=
sorry

end factors_of_180_multiples_of_15_l769_769793


namespace cuboid_volume_l769_769076

/-- Define the ratio condition for the dimensions of the cuboid. -/
def ratio (l w h : ℕ) : Prop :=
  (∃ x : ℕ, l = 2*x ∧ w = x ∧ h = 3*x)

/-- Define the total surface area condition for the cuboid. -/
def surface_area (l w h sa : ℕ) : Prop :=
  2*(l*w + l*h + w*h) = sa

/-- Volume of the cuboid given the ratio and surface area conditions. -/
theorem cuboid_volume (l w h : ℕ) (sa : ℕ) (h_ratio : ratio l w h) (h_surface : surface_area l w h sa) :
  ∃ v : ℕ, v = l * w * h ∧ v = 48 :=
by
  sorry

end cuboid_volume_l769_769076


namespace count_solutions_eq_287_l769_769292

noncomputable def count_solutions : ℕ :=
  (({n | ∃ x : ℝ, n < 500 ∧ (⌊x⌋.to_nat ≥ 0) ∧ x^(⌊x⌋.to_nat) = n} : set ℕ).to_finset.card)

theorem count_solutions_eq_287 : count_solutions = 287 :=
  sorry

end count_solutions_eq_287_l769_769292


namespace symmetry_center_l769_769100

noncomputable def translated_function (x : ℝ) : ℝ :=
  3 * Real.sin (2 * x - Real.pi / 6)

theorem symmetry_center : ∃ k : ℤ, (translated_function (k * Real.pi / 2 + Real.pi / 12) = 0) := 
begin
  use 1,
  simp [translated_function, Real.sin],
  sorry
end

end symmetry_center_l769_769100


namespace find_alpha_polar_equation_l769_769632

noncomputable def alpha := (3 * Real.pi) / 4

theorem find_alpha (P : ℝ × ℝ) (l : ℝ → ℝ × ℝ) (A B : ℝ × ℝ)
  (hP : P = (2, 1))
  (hl_def : ∀ t, l t = (2 + t * Real.cos alpha, 1 + t * Real.sin alpha))
  (hA : ∃ t, l t = (A.1, 0) ∧ A.1 > 0)
  (hB : ∃ t, l t = (0, B.2) ∧ B.2 > 0)
  (h_cond : dist P A * dist P B = 4) : alpha = (3 * Real.pi) / 4 :=
sorry

theorem polar_equation (l : ℝ → ℝ × ℝ)
  (hl_def : ∀ t, l t = (2 + t * Real.cos alpha, 1 + t * Real.sin alpha))
  (h_alpha : alpha = (3 * Real.pi) / 4)
  (h_polar : ∀ ρ θ, l ρ = (ρ * Real.cos θ, ρ * Real.sin θ))
  : ∀ ρ θ, ρ * (Real.cos θ + Real.sin θ) = 3 :=
sorry

end find_alpha_polar_equation_l769_769632


namespace evaluate_complex_pow_l769_769215

open Complex

noncomputable def calc : ℂ := (-64 : ℂ) ^ (7 / 6)

theorem evaluate_complex_pow : calc = 128 * Complex.I := by 
  -- Recognize that (-64) = (-4)^3
  -- Apply exponent rules: ((-4)^3)^(7/6) = (-4)^(3 * 7/6) = (-4)^(7/2)
  -- Simplify (-4)^(7/2) = √((-4)^7) = √(-16384)
  -- Calculation (-4)^7 = -16384
  -- Simplify √(-16384) = 128i
  sorry

end evaluate_complex_pow_l769_769215


namespace ratio_of_areas_l769_769899

noncomputable theory
open_locale classical

variables
  (A B C D E : Type)
  [incidence_geometry A]

variables 
  [is_parallel : is_parallel' A B, is_parallel' C E]
  [is_parallel' B C, is_parallel' A D]
  [is_parallel' A C, is_parallel' D E]
  [AB_length : distance A B = 4]
  [BC_length : distance B C = 6]
  [DE_length : distance D E = 24]
  [angle_ABC : angle A B C = 150]

-- the main theorem statement
theorem ratio_of_areas (A B C D E F : Type)
  [incidence_geometry A]
  [is_parallel' A B C E D]
  [angle_ABC A B C = 150]
  [distance A B = 4]
  [distance B C = 6]
  [distance D E = 24] : p + q = 289 :=
sorry

end ratio_of_areas_l769_769899


namespace minimum_A_plus_B_l769_769669

noncomputable def min_value_of_A_plus_B (A B : Set ℕ) (hA : A.Finite) (hB : B.Finite)
    (hA_card : A.toFinset.card = 20) (hB_card : B.toFinset.card = 16)
    (hA_condition : ∀ (a b m n : ℕ), a ∈ A → b ∈ A → m ∈ A → n ∈ A → (a + b = m + n) → ({a, b} = {m, n})) :
    ℕ :=
  |A + B|

theorem minimum_A_plus_B : ∀ (A B : Set ℕ), 
    A.Finite → B.Finite → 
    A.toFinset.card = 20 → B.toFinset.card = 16 → 
    (∀ a b m n, a ∈ A → b ∈ A → m ∈ A → n ∈ A → (a + b = m + n) → ({a, b} = {m, n})) → 
    min_value_of_A_plus_B A B = 200 := 
by
  intros A B hA hB hA_card hB_card hA_condition
  unfold min_value_of_A_plus_B
  sorry

end minimum_A_plus_B_l769_769669


namespace kids_joined_in_l769_769090

-- Define the given conditions
def original : ℕ := 14
def current : ℕ := 36

-- State the goal
theorem kids_joined_in : (current - original = 22) :=
by
  sorry

end kids_joined_in_l769_769090


namespace num_of_distinct_squares_in_set_I_l769_769338

def Point := (ℕ × ℕ)

def valid_point (p : Point) : Prop :=
  let (x, y) := p in 0 ≤ x ∧ x ≤ 5 ∧ 0 ≤ y ∧ y ≤ 5

def I : set Point :=
  { p | valid_point p }

theorem num_of_distinct_squares_in_set_I : 
  (∃ n, n = 105 ∧ ∀ s, s ∈ I → (square_formed_from_I s)) :=
sorry

end num_of_distinct_squares_in_set_I_l769_769338


namespace sides_of_triangle_l769_769391

theorem sides_of_triangle (n : ℕ) (t : Fin n → ℝ) (h : n^2 + 1 > (∑ i, t i) * (∑ i, t i⁻¹)) 
  (i j k : Fin n) (hijk : 1 ≤ i < j < k ≤ n) : t i + t j > t k := by
  sorry

end sides_of_triangle_l769_769391


namespace tommys_books_l769_769098

-- Define the cost of each book
def book_cost : ℕ := 5

-- Define the amount Tommy already has
def tommy_money : ℕ := 13

-- Define the amount Tommy needs to save up
def tommy_goal : ℕ := 27

-- Prove the number of books Tommy wants to buy
theorem tommys_books : tommy_goal + tommy_money = 40 ∧ (tommy_goal + tommy_money) / book_cost = 8 :=
by
  sorry

end tommys_books_l769_769098


namespace min_sequence_length_l769_769363

def is_subsequence {α : Type*} (S : list α) (B : list α) : Prop :=
  ∃ l₁ l₂, l₁ ++ B ++ l₂ = S

noncomputable def sequence_property (S : list ℕ) (n : ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ (B : list ℕ), B ≠ [] → B ⊆ S → ∃ (i : ℕ), is_subsequence (list.map a (list.range n).drop i) B

theorem min_sequence_length : 
  ∃ n, ∀ a : ℕ → ℕ,
    sequence_property [1, 2, 3, 4] n a :=
  ∃ n, n = 8 ∧ ∀ a : ℕ → ℕ, sequence_property [1, 2, 3, 4] n a :=
sorry

end min_sequence_length_l769_769363


namespace plane_equation_l769_769531

noncomputable def param_plane_eq (s t : ℝ) : ℝ × ℝ × ℝ :=
  (2 + 2 * s - 3 * t, 1 + s, 4 - 3 * s + t)

theorem plane_equation :
  ∃ A B C D : ℤ, A > 0 ∧ Int.gcd (Int.gcd (A.natAbs) (B.natAbs)) (Int.gcd (C.natAbs) (D.natAbs)) = 1 ∧
  (∀ v: ℝ × ℝ × ℝ, (∃ s t : ℝ, v = param_plane_eq s t) ↔ (A * v.1 + B * v.2 + C * v.3 + D = 0)) :=
  sorry

end plane_equation_l769_769531


namespace range_of_x_pow_k_l769_769670

theorem range_of_x_pow_k (k : ℝ) : set.range (λ x : ℝ, x ∈ set.Icc 0 1 → x^k) = set.Ioi 0 ∪ {+∞} :=
sorry

end range_of_x_pow_k_l769_769670


namespace fraction_of_cream_in_cup1_after_operations_l769_769179

/-
We consider two cups of liquids with the following contents initially:
Cup 1 has 6 ounces of coffee.
Cup 2 has 2 ounces of coffee and 4 ounces of cream.
After pouring half of Cup 1's content into Cup 2, stirring, and then pouring half of Cup 2's new content back into Cup 1, we need to show that 
the fraction of the liquid in Cup 1 that is now cream is 4/15.
-/

theorem fraction_of_cream_in_cup1_after_operations :
  let cup1_initial_coffee := 6
  let cup2_initial_coffee := 2
  let cup2_initial_cream := 4
  let cup2_initial_liquid := cup2_initial_coffee + cup2_initial_cream
  let cup1_to_cup2_coffee := cup1_initial_coffee / 2
  let cup1_final_coffee := cup1_initial_coffee - cup1_to_cup2_coffee
  let cup2_final_coffee := cup2_initial_coffee + cup1_to_cup2_coffee
  let cup2_final_liquid := cup2_final_coffee + cup2_initial_cream
  let cup2_to_cup1_liquid := cup2_final_liquid / 2
  let cup2_coffee_fraction := cup2_final_coffee / cup2_final_liquid
  let cup2_cream_fraction := cup2_initial_cream / cup2_final_liquid
  let cup2_to_cup1_coffee := cup2_to_cup1_liquid * cup2_coffee_fraction
  let cup2_to_cup1_cream := cup2_to_cup1_liquid * cup2_cream_fraction
  let cup1_final_liquid_coffee := cup1_final_coffee + cup2_to_cup1_coffee
  let cup1_final_liquid_cream := cup2_to_cup1_cream
  let cup1_final_liquid := cup1_final_liquid_coffee + cup1_final_liquid_cream
  (cup1_final_liquid_cream / cup1_final_liquid) = 4 / 15 :=
by
  sorry

end fraction_of_cream_in_cup1_after_operations_l769_769179


namespace BP_parallel_OQ_l769_769062

-- Define the basic structures and properties
variables {a b : ℝ} (h_ab : a > 0 ∧ b > 0)

-- Definitions of the geometric points
structure Point :=
  (x : ℝ) (y : ℝ)

-- Points
def O : Point := ⟨0, 0⟩
def A : Point := ⟨a, 0⟩
def B : Point := ⟨-a, 0⟩
variable {P : Point}
variable {Q : Point}

-- Condition for point P lying on the ellipse
def on_ellipse (P : Point) : Prop := 
  P.x^2 / a^2 + P.y^2 / b^2 = 1

-- Condition for point Q being the intersection of tangents at A and P
def Q_is_intersection (Q P : Point) : Prop := 
  Q.x = a ∧ Q.y = (b^2 * (a - P.x)) / (a * P.y)

-- Definition of slope between two points
def slope (P₁ P₂ : Point) : ℝ := 
  (P₂.y - P₁.y) / (P₂.x - P₁.x)

-- Proposition: Prove BP ∥ OQ
theorem BP_parallel_OQ (hP_on_ellipse : on_ellipse P) (hQ_intersection : Q_is_intersection Q P) :
  slope B P = slope O Q :=
sorry

end BP_parallel_OQ_l769_769062


namespace last_digit_of_power_sum_l769_769070

theorem last_digit_of_power_sum (m : ℕ) (hm : 0 < m) : (2^(m + 2006) + 2^m) % 10 = 0 := 
sorry

end last_digit_of_power_sum_l769_769070


namespace find_functions_satisfying_conditions_l769_769610

theorem find_functions_satisfying_conditions :
  (∀ {f : ℕ → ℕ}, (∀ n, 1 ≤ n → (0 < f n)) ∧
  (∀ n m : ℕ, 1 ≤ n → 1 ≤ m → f (n + m) = f n * f m) ∧
  (∃ n₀ : ℕ, (1 ≤ n₀ ∧ (f (f n₀) = f n₀ * f n₀))) →
  (∀ n, 1 ≤ n → (f n = 1 ∨ ∀ n, 1 ≤ n → f n = 2^n))) :=
begin
  sorry
end

end find_functions_satisfying_conditions_l769_769610


namespace length_of_first_train_l769_769175

-- Given conditions
def speed_first_train_kmph : ℝ := 120
def speed_second_train_kmph : ℝ := 80
def time_seconds : ℝ := 9
def length_second_train_meters : ℝ := 140.04

-- Conversion factor from kmph to m/s
def conversion_factor : ℝ := 5 / 18

-- Compute relative speed in m/s
def relative_speed_ms := (speed_first_train_kmph + speed_second_train_kmph) * conversion_factor

-- Compute combined length of both trains
def combined_length_meters := relative_speed_ms * time_seconds

-- Statement to prove the length of the first train
theorem length_of_first_train : combined_length_meters - length_second_train_meters = 360 := by
  sorry

end length_of_first_train_l769_769175


namespace f_is_xi_function_l769_769589

-- Define the properties of a ξ function
def xi_function (f : ℝ → ℝ) : Prop :=
  ∀ T : ℝ, T > 0 → ∃ m ≠ 0, ∀ x : ℝ, f(x + T) = m * f(x)

-- Define the specific function to test
noncomputable def f (x : ℝ) : ℝ := (1 / 2)^(3 * x - 2)

-- State the theorem to prove that f(x) is a ξ function
theorem f_is_xi_function : xi_function f :=
by
  sorry -- The proof is omitted as per instructions

end f_is_xi_function_l769_769589


namespace prime_triplet_geometric_progression_l769_769232

-- Define a predicate for a prime number
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the main proof objective including conditions
theorem prime_triplet_geometric_progression :
  ∃ (a b c : ℕ), a < b ∧ b < c ∧ c < 100 ∧ is_prime a ∧ is_prime b ∧ is_prime c ∧ 
  let a' := a + 1, b' := b + 1, c' := c + 1 in 
  b' * b' = a' * c' ∧ 
  ((a, b, c) = (2, 5, 11) ∨ (a, b, c) = (5, 11, 23) ∨ (a, b, c) = (7, 11, 17) ∨ 
  (a, b, c) = (11, 23, 47) ∨ (a, b, c) = (17, 41, 97)) := 
sorry

end prime_triplet_geometric_progression_l769_769232


namespace problem_l769_769830

-- Define the variable
variable (x : ℝ)

-- Define the condition
def condition := 3 * x - 1 = 8

-- Define the statement to be proven
theorem problem (h : condition x) : 150 * (1 / x) + 2 = 52 :=
  sorry

end problem_l769_769830


namespace prefer_scheme1_representation_beneficial_l769_769058

-- Definitions pertaining to the problem
def total_cost (L : ℝ) : ℝ := (1 / 3200) * L^2
def production (L : ℝ) : ℝ := 16 * real.sqrt L
def market_price : ℝ := 1

-- Scheme 1: You set the wage rate w first.
def scheme1_profit (w L : ℝ) : ℝ := 
  0.01 * w * L - total_cost L

-- Scheme 2: The enterprise sets the wage rate w first.
def scheme2_profit (w L : ℝ) : ℝ :=
  0.01 * w * L - total_cost L

-- Prove that scheme 1 is more preferable and the corresponding profit for the agency.
theorem prefer_scheme1 :
  ∀ w L : ℝ, 
    w = 2 → L = 64 / w^2 → (scheme1_profit w L = 0.24) →
    ∀ w' L' : ℝ, 
      w' = 1 → L' = 16 * w' → scheme1_profit w' L' < 0.24 := sorry

-- Prove that representation by the staffing agency is beneficial for the townspeople.
theorem representation_beneficial :
  ∀ w L : ℝ, 
    w = 2 → L = 64 / w^2 → 
    (scheme1_profit w L > 0) ∧
    (scheme2_profit 1 (16 * 1) > 0) := sorry

end prefer_scheme1_representation_beneficial_l769_769058


namespace basketball_free_throws_l769_769449

-- Define the given conditions as assumptions
variables {a b x : ℝ}
variables (h1 : 3 * b = 2 * a)
variables (h2 : x = 2 * a - 2)
variables (h3 : 2 * a + 3 * b + x = 78)

-- State the theorem to be proven
theorem basketball_free_throws : x = 74 / 3 :=
by {
  -- We will provide the proof later
  sorry
}

end basketball_free_throws_l769_769449


namespace piggy_bank_leftover_l769_769347

theorem piggy_bank_leftover :
  let initial_amount := 204
  let spent_on_toy := 0.60 * initial_amount
  let remaining_after_toy := initial_amount - spent_on_toy
  let spent_on_book := 0.50 * remaining_after_toy
  let remaining_after_book := remaining_after_toy - spent_on_book
  let spent_on_gift := 0.35 * remaining_after_book
  let remaining_after_gift := remaining_after_book - spent_on_gift
  in remaining_after_gift = 26.52 :=
by {
  sorry
}

end piggy_bank_leftover_l769_769347


namespace count_valid_N_l769_769307

-- Definitions based on identified mathematical conditions
def valid_N (N : ℕ) : Prop :=
  (∃ x : ℚ, 0 ≤ floor x ∧ floor x < 5 ∧ x ^ (floor x).natAbs = N) ∧ N < 500

theorem count_valid_N : finset.card (finset.filter valid_N (finset.range 500)) = 287 :=
by sorry

end count_valid_N_l769_769307


namespace range_of_a_l769_769102

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, a * x ^ 2 + a * x + 1 > 0) ∨ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 * x1 - x1 + a = 0 ∧ x2 * x2 - x2 + a = 0) ∧
  ¬((∀ x : ℝ, a * x ^ 2 + a * x + 1 > 0) ∧ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 * x1 - x1 + a = 0 ∧ x2 * x2 - x2 + a = 0)) →
  a ∈ set.Ioo (-(⊤ : ℝ)) 0 ∪ set.Ico (1/4) 4 :=
by 
  sorry

end range_of_a_l769_769102


namespace count_factors_of_180_multiple_of_15_l769_769711

def is_factor (x n : ℕ) := n % x = 0 -- Definition of a factor
def is_multiple (x k : ℕ) := x % k = 0 -- Definition of a multiple

theorem count_factors_of_180_multiple_of_15 :
  (finset.filter (λ x, is_multiple x 15) (finset.filter (λ x, is_factor x 180) (finset.range (180 + 1)))).card = 6 := by
  sorry

end count_factors_of_180_multiple_of_15_l769_769711


namespace range_of_m_l769_769319

theorem range_of_m (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 3) :
  ∀ m : ℝ, (∀ x y > 0, x + y = 3 → (4 / (x + 1) + 16 / y > m^2 - 3 * m + 5)) ↔ -1 < m ∧ m < 4 := 
sorry

end range_of_m_l769_769319


namespace simplify_expression_l769_769587

theorem simplify_expression (n : ℤ) :
  (2 : ℝ) ^ (-(3 * n + 1)) + (2 : ℝ) ^ (-(3 * n - 2)) - 3 * (2 : ℝ) ^ (-3 * n) = (3 / 2) * (2 : ℝ) ^ (-3 * n) :=
by
  sorry

end simplify_expression_l769_769587


namespace factors_of_180_multiple_of_15_l769_769747

/-- A factor of a number n is a positive integer which divides n without leaving a remainder. -/
def is_factor (n k : ℕ) : Prop :=
k > 0 ∧ n % k = 0

/-- A multiple of a number m is an integer which is divisible by m without leaving a remainder. -/
def is_multiple (k m : ℕ) : Prop :=
k % m = 0

/-- There are 6 factors of 180 that are also multiples of 15. -/
theorem factors_of_180_multiple_of_15 : 
  (Finset.filter (λ k, is_multiple k 15) { x | is_factor 180 x }.to_finset).card = 6 :=
by 
  sorry

end factors_of_180_multiple_of_15_l769_769747


namespace solve_system_l769_769405

def solution_set : set (ℝ × ℝ) :=
  { (x, y) | 
    x^2 + y^2 ≤ 1 ∧ 16 * x^4 - 8 * x^2 * y^2 + y^4 - 40 * x^2 - 10 * y^2 + 25 = 0 
  }

theorem solve_system :
  solution_set = { (-2 / real.sqrt 5, 1 / real.sqrt 5),
                   (-2 / real.sqrt 5, -1 / real.sqrt 5),
                   (2 / real.sqrt 5, -1 / real.sqrt 5),
                   (2 / real.sqrt 5, 1 / real.sqrt 5) } :=
by
  -- proof to be filled in later
  sorry

end solve_system_l769_769405


namespace ratio_of_area_of_smaller_circle_to_larger_rectangle_l769_769147

noncomputable def ratio_areas (w : ℝ) : ℝ :=
  (3.25 * Real.pi * w^2 / 4) / (1.5 * w^2)

theorem ratio_of_area_of_smaller_circle_to_larger_rectangle (w : ℝ) : 
  ratio_areas w = 13 * Real.pi / 24 := 
by 
  sorry

end ratio_of_area_of_smaller_circle_to_larger_rectangle_l769_769147


namespace positive_factors_of_180_multiple_of_15_count_l769_769763

theorem positive_factors_of_180_multiple_of_15_count : 
  let factorization_180 := 2^2 * 3^2 * 5;
  let is_multiple_of_15 (n : ℕ) := ∃ k1 k2 : ℕ, k1 * 3 = n ∧ k2 * 5 = n;
  let factors_180 := { n : ℕ | n ∣ factorization_180 };
  { n : ℕ | n ∈ factors_180 ∧ is_multiple_of_15 n }.card = 6 :=
by
  sorry

end positive_factors_of_180_multiple_of_15_count_l769_769763


namespace sufficient_condition_l769_769894

variables {a b : ℝ^3}

def parallel (a b: ℝ^3) : Prop :=
  ∃ k: ℝ, a = k • b

def same_direction (a b: ℝ^3) : Prop :=
  (parallel a b) ∧ (∃ k > 0, a = k • b)

theorem sufficient_condition
  (non_zero_1 : a ≠ 0)
  (non_zero_2 : b ≠ 0)
  (par_and_eq_mag : parallel a b ∧ (|a| = |b|)) :
  (a / |a| = b / |b|) :=
begin
  sorry
end

end sufficient_condition_l769_769894


namespace eccentricity_of_hyperbola_l769_769690

variable {a b : ℝ}
axiom h1 : a > 0
axiom h2 : b > 0
axiom h3 : (y = (1/2) * x ∨ y = -(1/2) * x) → (x^2 / a^2 - y^2 / b^2 = 1)

def hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : (y = (1/2) * x ∨ y = -(1/2) * x) → (x^2 / a^2 - y^2 / b^2 = 1)) : ℝ :=
  sqrt (a^2 + b^2) / a

theorem eccentricity_of_hyperbola : hyperbola_eccentricity a b h1 h2 h3 = sqrt 5 / 2 :=
sorry

end eccentricity_of_hyperbola_l769_769690


namespace problem_statement_l769_769365

noncomputable def a (n : ℕ) : ℝ := (n^2 + 1) / Real.sqrt (n^4 + 4)
noncomputable def b (n : ℕ) : ℝ := (List.range n).map (fun i => a (i + 1)).prod

theorem problem_statement (n : ℕ) (hn : 0 < n) :
  (b n) / Real.sqrt 2 = Real.sqrt (n^2 + 1) / Real.sqrt (n^2 + 2 * n + 2)
  ∧
  (1 / (n+1)^3 < ((b n) / Real.sqrt 2 - n / (n+1)) ∧ ((b n) / Real.sqrt 2 - n / (n+1)) < 1 / (n^3)) :=
by
  sorry

end problem_statement_l769_769365


namespace proof_a3b_m2_l769_769914

def a : ℚ := 4 / 7
def b : ℚ := 5 / 6

theorem proof_a3b_m2 : a^3 * b^(-2) = 2304 / 8575 := by
  sorry

end proof_a3b_m2_l769_769914


namespace ball_falls_total_distance_l769_769518

noncomputable def total_distance : ℕ → ℤ → ℤ → ℤ
| 0, a, _ => 0
| (n+1), a, d => a + total_distance n (a + d) d

theorem ball_falls_total_distance :
  total_distance 5 30 (-6) = 90 :=
by
  sorry

end ball_falls_total_distance_l769_769518


namespace Sine_Theorem_Trihedral_Angle_l769_769962

theorem Sine_Theorem_Trihedral_Angle
  (α β γ A B C : ℝ)
  (hα : 0 < α ∧ α < π)
  (hβ : 0 < β ∧ β < π)
  (hγ : 0 < γ ∧ γ < π)
  (hA : 0 < A ∧ A < π)
  (hB : 0 < B ∧ B < π)
  (hC : 0 < C ∧ C < π)
  (hcos_α : cos α = cos β * cos γ + sin β * sin γ * cos A)
  (hcos_β : cos β = cos α * cos γ + sin α * sin γ * cos B)
  (hcos_γ : cos γ = cos α * cos β + sin α * sin β * cos C) :
  sin A / sin α = sin B / sin β ∧ sin B / sin β = sin C / sin γ := by
  sorry

end Sine_Theorem_Trihedral_Angle_l769_769962


namespace correct_operation_l769_769501

theorem correct_operation (a b : ℝ) : (a + b) * (a - b) = a^2 - b^2 := by
  sorry

end correct_operation_l769_769501


namespace remainder_when_divided_by_9_l769_769487

theorem remainder_when_divided_by_9 (x : ℕ) (h1 : x > 0) (h2 : (5 * x) % 9 = 7) : x % 9 = 5 :=
sorry

end remainder_when_divided_by_9_l769_769487


namespace simplify_expression_l769_769047

variable (x : ℝ)

theorem simplify_expression : 
  2 * x^3 - (7 * x^2 - 9 * x) - 2 * (x^3 - 3 * x^2 + 4 * x) = -x^2 + x := 
by
  sorry

end simplify_expression_l769_769047


namespace max_band_members_l769_769407

theorem max_band_members (n : ℤ) (h1 : 22 * n % 24 = 2) (h2 : 22 * n < 1000) : 22 * n = 770 :=
  sorry

end max_band_members_l769_769407


namespace tank_length_l769_769478

variable (rate : ℝ)
variable (time : ℝ)
variable (width : ℝ)
variable (depth : ℝ)
variable (volume : ℝ)
variable (length : ℝ)

-- Given conditions
axiom rate_cond : rate = 5 -- cubic feet per hour
axiom time_cond : time = 60 -- hours
axiom width_cond : width = 6 -- feet
axiom depth_cond : depth = 5 -- feet

-- Derived volume from the rate and time
axiom volume_cond : volume = rate * time

-- Definition of length from volume, width, and depth
axiom length_def : length = volume / (width * depth)

-- The proof problem to show
theorem tank_length : length = 10 := by
  -- conditions provided and we expect the length to be computed
  sorry

end tank_length_l769_769478


namespace Sandy_total_marks_l769_769043

theorem Sandy_total_marks
  (correct_marks_per_sum : ℤ)
  (incorrect_marks_per_sum : ℤ)
  (total_sums : ℕ)
  (correct_sums : ℕ)
  (incorrect_sums : ℕ)
  (total_marks : ℤ) :
  correct_marks_per_sum = 3 →
  incorrect_marks_per_sum = -2 →
  total_sums = 30 →
  correct_sums = 24 →
  incorrect_sums = total_sums - correct_sums →
  total_marks = correct_marks_per_sum * correct_sums + incorrect_marks_per_sum * incorrect_sums →
  total_marks = 60 :=
by
  sorry

end Sandy_total_marks_l769_769043


namespace count_positive_factors_of_180_multiple_of_15_l769_769730

statement: ℕ
statement := sorry

theorem count_positive_factors_of_180_multiple_of_15 :
  let factors_of_180 := {d ∈ (finset.range 181).filter (λ x, 180 % x = 0)}
  let multiples_of_15 := (finset.range 181).filter (λ x, x % 15 = 0)
  let desired_factors := factors_of_180 ∩ multiples_of_15
  desired_factors.card = 6 :=
by
  sorry

end count_positive_factors_of_180_multiple_of_15_l769_769730


namespace combine_exponent_remains_unchanged_l769_769486

-- Define combining like terms condition
def combining_like_terms (terms : List (ℕ × String)) : List (ℕ × String) := sorry

-- Define the problem statement
theorem combine_exponent_remains_unchanged (terms : List (ℕ × String)) : 
  (combining_like_terms terms).map Prod.snd = terms.map Prod.snd :=
sorry

end combine_exponent_remains_unchanged_l769_769486


namespace correct_equation_l769_769494

theorem correct_equation (a b : ℝ) : 
  (¬ (a^2 + a^3 = a^6)) ∧ (¬ ((ab)^2 = ab^2)) ∧ (¬ ((a+b)^2 = a^2 + b^2)) ∧ ((a+b)*(a-b) = a^2 - b^2) :=
by {
  sorry
}

end correct_equation_l769_769494


namespace concyclic_points_l769_769915

noncomputable theory

open EuclideanGeometry

variables {A B C A₁ B₁ P Q P₁ Q₁ : Point}

-- Given:
variables (hABC : Triangle A B C)
variables (hA1 : A₁ ∈ Segment B C)
variables (hB1 : B₁ ∈ Segment C A)
variables (hP : P ∈ Segment A A₁)
variables (hQ : Q ∈ Segment B B₁)
variables (hPQ_parallel_AB : Parallel (Line PQ) (Line AB))
variables (hP1_on_PB1 : P₁ ∈ Line P B₁)
variables (hB1_between_P_and_P1 : B₁ ∈ Segment P P₁)
variables (h_angle_PP1C_eq_angle_BAC : Angle P P₁ C = Angle B A C)
variables (hQ1_on_QC1 : Q₁ ∈ Line Q C₁)
variables (hC1_between_Q_and_Q1 : C₁ ∈ Segment Q Q₁)
variables (h_angle_QQ1C_eq_angle_ABC : Angle Q Q₁ C = Angle A B C)

-- To Prove:
theorem concyclic_points (hABC : Triangle A B C)
  (hA1 : A₁ ∈ Segment B C)
  (hB1 : B₁ ∈ Segment C A)
  (hP : P ∈ Segment A A₁)
  (hQ : Q ∈ Segment B B₁)
  (hPQ_parallel_AB : Parallel (Line PQ) (Line AB))
  (hP1_on_PB1 : P₁ ∈ Line P B₁)
  (hB1_between_P_and_P1 : B₁ ∈ Segment P P₁)
  (h_angle_PP1C_eq_angle_BAC : Angle P P₁ C = Angle B A C)
  (hQ1_on_QC1 : Q₁ ∈ Line Q C₁)
  (hC1_between_Q_and_Q1 : C₁ ∈ Segment Q Q₁)
  (h_angle_QQ1C_eq_angle_ABC : Angle Q Q₁ C = Angle A B C) :
  Cyclic (Set.FromList [P, Q, P₁, Q₁]) :=
sorry

end concyclic_points_l769_769915


namespace factors_of_180_multiple_of_15_count_l769_769804

theorem factors_of_180_multiple_of_15_count :
  ∃ n : Nat, n = 6 ∧ ∀ x : Nat, x ∣ 180 → 15 ∣ x → x > 0 :=
by
  sorry

end factors_of_180_multiple_of_15_count_l769_769804


namespace find_vector_difference_norm_l769_769672

variables (a b : ℝ)
variables (vec_a vec_b : EuclideanSpace ℝ (Fin 2)) 

def norm (v : EuclideanSpace ℝ (Fin 2)) := Real.sqrt (euclid_normSq v)

theorem find_vector_difference_norm
  (h1 : norm vec_a = 6)
  (h2 : norm vec_b = 8)
  (h3 : norm (vec_a + vec_b) = norm (vec_a - vec_b)) :
  norm (vec_a - vec_b) = 10 :=
sorry

end find_vector_difference_norm_l769_769672


namespace find_ff_neg3_l769_769968

noncomputable def f : ℝ → ℝ :=
λ x, if x > 0 then 3 - Real.log x / Real.log 2 else x^2 - 1

theorem find_ff_neg3 :
  f (f (-3)) = 0 :=
by
  sorry

end find_ff_neg3_l769_769968


namespace retailer_percentage_increase_l769_769526

variable (C : ℝ)

def customer_price (C : ℝ) : ℝ := 1.54 * C

def retailer_price (C : ℝ) : ℝ := customer_price C / 1.10

def percentage_increase (cost retail : ℝ) : ℝ := ((retail - cost) / cost) * 100

theorem retailer_percentage_increase (C : ℝ) :
  percentage_increase C (retailer_price C) = 40 := by
  sorry

end retailer_percentage_increase_l769_769526


namespace y1_gt_y2_l769_769315

theorem y1_gt_y2 (y : ℤ → ℤ) (h_eq : ∀ x, y x = 8 * x - 1)
  (y1 y2 : ℤ) (h_y1 : y 3 = y1) (h_y2 : y 2 = y2) : y1 > y2 :=
by
  -- proof
  sorry

end y1_gt_y2_l769_769315


namespace real_part_of_complex_num_is_real_l769_769370

-- Define the complex number and its conditions
def complex_num (a : ℝ) : ℂ := (a + 2 * Complex.I) / (1 + Complex.I)

-- Theorem stating that the value of a if the given complex number is real
theorem real_part_of_complex_num_is_real (a : ℝ) (h : complex_num a ∈ ℝ) : a = 2 :=
by
  sorry

end real_part_of_complex_num_is_real_l769_769370


namespace count_valid_N_l769_769306

-- Definitions based on identified mathematical conditions
def valid_N (N : ℕ) : Prop :=
  (∃ x : ℚ, 0 ≤ floor x ∧ floor x < 5 ∧ x ^ (floor x).natAbs = N) ∧ N < 500

theorem count_valid_N : finset.card (finset.filter valid_N (finset.range 500)) = 287 :=
by sorry

end count_valid_N_l769_769306


namespace range_of_a_for_tangents_l769_769263

-- Definitions for the problem conditions
def point_outside_circle (p : ℝ × ℝ) (c : ℝ × ℝ) (r : ℝ) : Prop :=
  let d := (p.1 - c.1)^2 + (p.2 - c.2)^2
  d > r^2

-- Lean proof statement
theorem range_of_a_for_tangents (a : ℝ) :
  point_outside_circle (-2, 3) (a, 2) 3 →
  (a < -2 - 2 * Real.sqrt 2 ∨ a > -2 + 2 * Real.sqrt 2) :=
by {
  assume h : point_outside_circle (-2, 3) (a, 2) 3,
  sorry
}

end range_of_a_for_tangents_l769_769263


namespace intersection_of_M_and_N_l769_769278

def M : Set ℝ := {x | x < 1 }
def N : Set ℝ := {x | 2^x > 1}

theorem intersection_of_M_and_N : M ∩ N = {x | 0 < x ∧ x < 1} :=
by sorry

end intersection_of_M_and_N_l769_769278


namespace sqrt_x_plus_one_over_sqrt_x_eq_ten_l769_769911

noncomputable theory

open Real

theorem sqrt_x_plus_one_over_sqrt_x_eq_ten (x : ℝ) (hx1 : 0 < x) (hx2 : x + 1/x = 98) :
  sqrt x + 1 / sqrt x = 10 :=
by
  sorry

end sqrt_x_plus_one_over_sqrt_x_eq_ten_l769_769911


namespace simplify_expression_l769_769402

theorem simplify_expression (s : ℤ) : 120 * s - 32 * s = 88 * s := by
  sorry

end simplify_expression_l769_769402


namespace arithmetic_progression_roots_l769_769611

theorem arithmetic_progression_roots (b : ℝ) : 
  (∃ (s : ℝ) (h : ℂ), h.im ≠ 0 ∧ (s-h, s, s+h) ∈ Multiset.roots (x^3 - 9*x^2 + 39*x + b)) → b = -36 := 
sorry

end arithmetic_progression_roots_l769_769611


namespace dihedral_angle_of_regular_pyramid_l769_769862

noncomputable def dihedral_angle_range (n : ℕ) : Prop :=
  ∀ (θ : ℝ), (0 < θ) ∧ (θ < π)

theorem dihedral_angle_of_regular_pyramid (n : ℕ) :
  dihedral_angle_range n :=
sorry

end dihedral_angle_of_regular_pyramid_l769_769862


namespace sum_possible_amounts_l769_769000

theorem sum_possible_amounts : 
  let valid_amounts := {x | x < 100 ∧ x % 5 = 4 ∧ x % 10 = 7}
  (Finset.sum (Finset.filter valid_amounts (Finset.range 100))) = 520 :=
by
  sorry

end sum_possible_amounts_l769_769000


namespace probability_of_triangle_with_C_l769_769876

-- Definitions for the points and triangles
def Point : Type := ℤ

noncomputable def A : Point := 0
noncomputable def B : Point := 1
noncomputable def C : Point := 2 -- Vertex with a dot
noncomputable def D : Point := 3
noncomputable def E : Point := 4

def triangles : list (Point × Point × Point) :=
  [(A, C, D), (B, C, D), (A, B, C), (A, B, E), (A, D, E), (B, D, E)]

def triangles_with_C := [ (A, C, D), (B, C, D), (A, B, C) ]

theorem probability_of_triangle_with_C :
  ↑(triangles_with_C.length) / ↑(triangles.length) = 1 / 2 :=
  by
    sorry

end probability_of_triangle_with_C_l769_769876


namespace true_props_are_123_l769_769682

-- Proposition definitions
def prop1 : Prop := ∀ (a b c : ℝ), a ≠ 0 → (b^2 - 4 * a * c < 0 → ¬has_real_roots a b c)
def prop2 : Prop := ∀ (A B C : Type), is_equilateral_triangle A B C → sides_are_equal A B C
def prop3 : Prop := ∀ (a b : ℝ), (a > b ∧ b > 0) → (3 * a > 3 * b ∧ 3 * b > 0)
def prop4 : Prop := ∀ (m : ℝ), (m > 1 → solution_set_is_real m)

-- Negations, converses, and contrapositives
def neg_prop1 : Prop := ∀ (a b c : ℝ), a ≠ 0 → (b^2 - 4 * a * c ≥ 0 → has_real_roots a b c)
def conv_prop2 : Prop := ∀ (A B C : Type), is_equilateral_triangle A B C ↔ sides_are_equal A B C
def contrap_prop3 : Prop := ∀ (a b : ℝ), (¬(3 * a > 3 * b ∧ 3 * b > 0) → ¬(a > b ∧ b > 0))
def conv_prop4 : Prop := ∀ (m : ℝ), solution_set_is_real m → m > 1

-- Proof of truth values
theorem true_props_are_123 :
  (neg_prop1 ∧ conv_prop2 ∧ contrap_prop3) ∧ ¬conv_prop4 := 
begin
  sorry, -- The proof is not required as per instructions
end

end true_props_are_123_l769_769682


namespace probability_two_green_apples_l769_769885

noncomputable def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_two_green_apples :
  ∀ (total_apples green_apples choose_apples : ℕ),
    total_apples = 7 →
    green_apples = 3 →
    choose_apples = 2 →
    (binom green_apples choose_apples : ℝ) / binom total_apples choose_apples = 1 / 7 :=
by
  intro total_apples green_apples choose_apples
  intro h_total h_green h_choose
  rw [h_total, h_green, h_choose]
  -- The proof would go here
  sorry

end probability_two_green_apples_l769_769885


namespace perpendicular_segments_l769_769146

theorem perpendicular_segments
  (circle : Set ℝ) (A B D C E M : EuclideanSpace ℝ 2)
  (arcAB : Arc circle A B) (arcAD : Arc circle A D)
  (midC : IsMidpoint C arcAB) (midE : IsMidpoint E arcAD) (midM : IsMidpoint M (Segment B D)) :
  ∠ C M E = 90 :=
sorry

end perpendicular_segments_l769_769146


namespace parabola_circle_intersection_sum_of_distances_l769_769904

noncomputable def sum_of_distances : ℝ :=
  let focus := (0 : ℝ, 1 / 4 : ℝ)
  let points := [(1 : ℝ, 1 : ℝ), (4 : ℝ, (4^2 : ℝ)), (-9 : ℝ, (-9)^2 : ℝ), (4 : ℝ, (4^2 : ℝ))]
  points.sum (λ p, real.sqrt ((p.1 - focus.1)^2 + (p.2 - focus.2)^2))

theorem parabola_circle_intersection_sum_of_distances :
  sum_of_distances = 114.737 := 
  sorry

end parabola_circle_intersection_sum_of_distances_l769_769904


namespace num_factors_of_180_multiple_of_15_l769_769734

def is_factor (n m : Nat) : Prop := m % n = 0
def is_multiple_of_15 (n : Nat) : Prop := is_factor 15 n

theorem num_factors_of_180_multiple_of_15 : 
  (Nat.filter (fun n => is_factor n 180 ∧ is_multiple_of_15 n) (Nat.range (180 + 1))).length = 6 :=
sorry

end num_factors_of_180_multiple_of_15_l769_769734


namespace collinearity_of_AKM_l769_769069

variable {A B C D E F K M I : Type}
variable [Incircle : IncircleType A B C D E F I] -- Assume a custom type class to represent the incircle properties

-- Definitions corresponding to the conditions
def incircle_touches_D (I D : Type) (tr : IncircleType A B C D E F I) : Prop := touches I D (side BC)
def incircle_touches_E (I E : Type) (tr : IncircleType A B C D E F I) : Prop := touches I E (side CA)
def incircle_touches_F (I F : Type) (tr : IncircleType A B C D E F I) : Prop := touches I F (side AB)
def midpoint_of_BC {M : Point} (BC : Line) : Prop := midpoint M B C

-- Given the conditions and required to prove collinearity
theorem collinearity_of_AKM {I : Type} [Incircle : IncircleType A B C D E F I] (I D E F K M : Type) :
  incircle_touches_D I D (Incircle) ∧
  incircle_touches_E I E (Incircle) ∧
  incircle_touches_F I F (Incircle) ∧
  ∃ K, intersect (line_from_to ID) (segment EF) K ∧
  midpoint_of_BC M (line BC)
  →
  collinear {A, K, M} :=
by {
  sorry
}

end collinearity_of_AKM_l769_769069


namespace annual_sparkling_water_cost_l769_769385

theorem annual_sparkling_water_cost :
  (let cost_per_bottle := 2.00
       nights_per_year := 365
       fraction_bottle_per_night := 1 / 5
       bottles_per_year := nights_per_year * fraction_bottle_per_night in
   bottles_per_year * cost_per_bottle = 146.00) :=
by
  -- This is where the actual proof would go.
  sorry

end annual_sparkling_water_cost_l769_769385


namespace total_cars_made_in_two_days_l769_769555

-- Define the number of cars made yesterday.
def cars_yesterday : ℕ := 60

-- Define the number of cars made today, which is twice the number of cars made yesterday.
def cars_today : ℕ := 2 * cars_yesterday

-- Define the total number of cars made over the two days.
def total_cars : ℕ := cars_yesterday + cars_today

-- Proof statement: Prove that the total number of cars made in these two days is 180.
theorem total_cars_made_in_two_days : total_cars = 180 := by
  -- Here, we would provide the proof, but we'll use sorry to skip it.
  sorry

end total_cars_made_in_two_days_l769_769555


namespace avg_A_lt_avg_B_combined_avg_eq_6_6_l769_769473

-- Define the scores for A and B
def scores_A := [4, 5, 5, 6, 6, 7, 7, 8, 8, 9]
def scores_B := [2, 5, 6, 6, 7, 7, 7, 8, 9, 10]

-- Define the average score function
def average (scores : List ℚ) : ℚ := (scores.sum : ℚ) / scores.length

-- Define the mean for the combined data
def combined_average : ℚ :=
  (average scores_A * scores_A.length + average scores_B * scores_B.length) / 
  (scores_A.length + scores_B.length)

-- Specify the variances given in the problem
def variance_A := 2.25
def variance_B := 4.41

-- Claim the average score of A is smaller than the average score of B
theorem avg_A_lt_avg_B : average scores_A < average scores_B := by sorry

-- Claim the average score of these 20 data points is 6.6
theorem combined_avg_eq_6_6 : combined_average = 6.6 := by sorry

end avg_A_lt_avg_B_combined_avg_eq_6_6_l769_769473


namespace problem_statement_l769_769853

theorem problem_statement (x y : ℝ) (hx : x - y = 3) (hxy : x = 4 ∧ y = 1) : 2 * (x - y) = 6 * y :=
by
  rcases hxy with ⟨hx', hy'⟩
  rw [hx', hy']
  sorry

end problem_statement_l769_769853


namespace arithmetic_mean_y_value_l769_769061

theorem arithmetic_mean_y_value (y : ℝ) (h : (y + 10 + 20 + 3y + 18 + 3y + 6 + 12) / 6 = 30) : y = 114 / 7 :=
by
  sorry

end arithmetic_mean_y_value_l769_769061


namespace range_of_a_l769_769691

def set_A (a : ℝ) : set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p in ax - 2y + 8 ≥ 0 ∧ x - y - 1 ≤ 0 ∧ 2x + ay - 2 ≤ 0}

def z (p : ℝ × ℝ) : ℝ :=
  let (x, y) := p in y - x

theorem range_of_a (a : ℝ) (nonempty_A : (set_A a).nonempty) :
  ∃ max_z min_z : ℝ, ∀ p ∈ set_A a, min_z ≤ z p ∧ z p ≤ max_z ↔ a ≥ 2 :=
sorry

end range_of_a_l769_769691


namespace toy_car_production_l769_769553

theorem toy_car_production (yesterday today total : ℕ) 
  (hy : yesterday = 60)
  (ht : today = 2 * yesterday) :
  total = yesterday + today :=
by
  sorry

end toy_car_production_l769_769553


namespace count_factors_of_180_multiple_of_15_l769_769718

def is_factor (x n : ℕ) := n % x = 0 -- Definition of a factor
def is_multiple (x k : ℕ) := x % k = 0 -- Definition of a multiple

theorem count_factors_of_180_multiple_of_15 :
  (finset.filter (λ x, is_multiple x 15) (finset.filter (λ x, is_factor x 180) (finset.range (180 + 1)))).card = 6 := by
  sorry

end count_factors_of_180_multiple_of_15_l769_769718


namespace find_c8_l769_769674

-- Definitions of arithmetic sequences and their products
def arithmetic_seq (a d : ℤ) (n : ℕ) := a + n * d

def c_n (a d1 b d2 : ℤ) (n : ℕ) := arithmetic_seq a d1 n * arithmetic_seq b d2 n

-- Given conditions
variables (a1 d1 a2 d2 : ℤ)
variables (c1 c2 c3 : ℤ)
variables (h1 : c_n a1 d1 a2 d2 1 = 1440)
variables (h2 : c_n a1 d1 a2 d2 2 = 1716)
variables (h3 : c_n a1 d1 a2 d2 3 = 1848)

-- The goal is to prove c_8 = 348
theorem find_c8 : c_n a1 d1 a2 d2 8 = 348 :=
sorry

end find_c8_l769_769674


namespace factors_of_180_multiple_of_15_l769_769746

/-- A factor of a number n is a positive integer which divides n without leaving a remainder. -/
def is_factor (n k : ℕ) : Prop :=
k > 0 ∧ n % k = 0

/-- A multiple of a number m is an integer which is divisible by m without leaving a remainder. -/
def is_multiple (k m : ℕ) : Prop :=
k % m = 0

/-- There are 6 factors of 180 that are also multiples of 15. -/
theorem factors_of_180_multiple_of_15 : 
  (Finset.filter (λ k, is_multiple k 15) { x | is_factor 180 x }.to_finset).card = 6 :=
by 
  sorry

end factors_of_180_multiple_of_15_l769_769746


namespace monge_point_intersection_altitude_foot_circumcircle_l769_769119

-- Define a tetrahedron in 3D space
noncomputable def Tetrahedron (A B C D : EuclideanSpace ℝ (Fin 3)) : Prop := sorry

-- Define the concept of midpoints and perpendicular planes
noncomputable def Midpoint (a b : EuclideanSpace ℝ (Fin 3)) := (1/2) • (a + b)
axiom Perpendicular (a b c : EuclideanSpace ℝ (Fin 3)) : Prop

-- Define the Monge point
noncomputable def MongePoint (A B C D : EuclideanSpace ℝ (Fin 3)) : EuclideanSpace ℝ (Fin 3) := sorry

-- Define the circumscribed circle and the foot of the altitude
axiom CircumscribedCircle (A B C : EuclideanSpace ℝ (Fin 3)) : Set (EuclideanSpace ℝ (Fin 3))
axiom FootOfAltitude (D A B C : EuclideanSpace ℝ (Fin 3)) : EuclideanSpace ℝ (Fin 3)

-- Define the conditions as properties
axiom MidpointPerpendicularPlanes (A B C D : EuclideanSpace ℝ (Fin 3)) :
  (∀ a b, (a, b) ∈ {(A, B), (A, C), (A, D), (B, C), (B, D), (C, D)} → 
  ∃ p : EuclideanSpace ℝ (Fin 3), Perpendicular p (Midpoint a b) (opposite_edge a b) ∧ 
  ∀ q : EuclideanSpace ℝ (Fin 3), Perpendicular q (Midpoint a b) (opposite_edge a b) → q = p ) ∧
  ∃ M : EuclideanSpace ℝ (Fin 3), ∀ p : EuclideanSpace ℝ (Fin 3), MongePoint A B C D = p

-- Prove that all midpoints planes intersect at the Monge point
theorem monge_point_intersection (A B C D : EuclideanSpace ℝ (Fin 3)) :
  (Tetrahedron A B C D) → 
  (∀ e1 e2 : EuclideanSpace ℝ (Fin 3), MidpointPerpendicularPlanes e1 e2 → 
   ∀ p : EuclideanSpace ℝ (Fin 3), Perpendicular p (Midpoint e1 e2) e1 e2) →
  ∃ O : EuclideanSpace ℝ (Fin 3), MongePoint A B C D = O := 
  sorry

-- Prove the property about the foot of the altitude and the circumcircle
theorem altitude_foot_circumcircle (A B C D : EuclideanSpace ℝ (Fin 3)) :
  (Tetrahedron A B C D) → 
  (MongePoint A B C D ∈ Plane (A, B, C)) →
  FootOfAltitude D A B C ∈ CircumscribedCircle A B C :=
  sorry

end monge_point_intersection_altitude_foot_circumcircle_l769_769119


namespace B_share_in_profit_l769_769177

theorem B_share_in_profit (A B C : ℝ) (total_profit : ℝ) 
    (h1 : A = 3 * B)
    (h2 : B = (2/3) * C)
    (h3 : total_profit = 6600) :
    (B / (A + B + C)) * total_profit = 1200 := 
by
  sorry

end B_share_in_profit_l769_769177


namespace tom_catches_jerry_in_6_2_seconds_l769_769950

-- Geometry and basic facts about triangles
noncomputable def time_to_catch (t : ℝ) : Prop :=
  let distance_JC := 3 * t
  let distance_TC := 5 * t
  let distance_TJ := 15.0
  distance_TC^2 = distance_TJ^2 + distance_JC^2 - 2 * distance_TJ * distance_JC * (real.cos (135 * real.pi / 180))

theorem tom_catches_jerry_in_6_2_seconds :
  ∃ t : ℝ, time_to_catch t ∧ abs (t - 6.2) < 0.01 :=
sorry

end tom_catches_jerry_in_6_2_seconds_l769_769950


namespace clive_can_correct_time_l769_769054

def can_show_correct_time (hour_hand_angle minute_hand_angle : ℝ) :=
  ∃ θ : ℝ, θ ∈ [0, 360] ∧ hour_hand_angle + θ % 360 = minute_hand_angle + θ % 360

theorem clive_can_correct_time (hour_hand_angle minute_hand_angle : ℝ) :
  can_show_correct_time hour_hand_angle minute_hand_angle :=
sorry

end clive_can_correct_time_l769_769054


namespace circles_internally_tangent_l769_769978

theorem circles_internally_tangent :
  let C1 := (3, -2)
  let r1 := 1
  let C2 := (7, 1)
  let r2 := 6
  let d := Real.sqrt (((7 - 3)^2 + (1 - (-2))^2) : ℝ)
  d = r2 - r1 :=
by
  sorry

end circles_internally_tangent_l769_769978


namespace polyhedron_volume_l769_769877

-- Define the properties of the polygons
def isosceles_right_triangle (a : ℝ) := a ≠ 0 ∧ ∀ (x y : ℝ), x = y

def square (side : ℝ) := side = 2

def equilateral_triangle (side : ℝ) := side = 2 * Real.sqrt 2

-- Define the conditions
def condition_AE : Prop := isosceles_right_triangle 2
def condition_B : Prop := square 2
def condition_C : Prop := square 2
def condition_D : Prop := square 2
def condition_G : Prop := equilateral_triangle (2 * Real.sqrt 2)

-- Define the polyhedron volume calculation problem
theorem polyhedron_volume (hA : condition_AE) (hE : condition_AE) (hF : condition_AE) (hB : condition_B) (hC : condition_C) (hD : condition_D) (hG : condition_G) : 
  ∃ V : ℝ, V = 16 := 
sorry

end polyhedron_volume_l769_769877


namespace eq_ellipse_determine_k_fixed_point_l769_769666

-- Problem 1: Prove the equation of the ellipse
theorem eq_ellipse (a b : ℝ) (h1 : 0 < b) (h2 : b < a) (h3 : 2 * sqrt 2 = 2 * b * sqrt (a^2 / b^2 - 1)) :
  (∃ h : a = sqrt (b^2 + (sqrt 2)^2), (∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2) / 4 + (y^2) / 2 = 1)) :=
sorry

-- Problem 2(i): Determine the slope k
theorem determine_k (k : ℝ) (h1 : ∃ m : ℝ, 3 * m + 3 * (2 / 3 - m) = 2)
  (h2 : 2 * m / 3 * (k - 1) = - 1 / k) (h3 : 7 * k^2 - 18 * k + 8 = 0) :
  (k = 2 ∨ k = 4 / 7) :=
sorry

-- Problem 2(ii): Prove point G is fixed
theorem fixed_point (k : ℝ) (t : ℝ) (h1 : k * (2 - 4 * k^2 / (1 + 2 * k^2) - 2 - 4 * k^2 / (1 + 2 * k^2))
  * k / (2 - t) = - 1) :
  (t = 0) :=
sorry

end eq_ellipse_determine_k_fixed_point_l769_769666


namespace max_neg_expr_l769_769257

theorem max_neg_expr (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  (- (1 / (2 * a)) - (2 / b)) ≤ - (9 / 2) :=
sorry

end max_neg_expr_l769_769257


namespace factors_of_180_l769_769766

-- Define a predicate to check if a number is a factor of another number
def is_factor (m n : ℕ) : Prop := n % m = 0

-- Define a predicate to check if a number is a multiple of another number
def is_multiple (m n : ℕ) : Prop := m % n = 0

-- Define the problem
theorem factors_of_180 : {n : ℕ // is_factor n 180 ∧ is_multiple n 15}.card = 6 :=
by sorry

end factors_of_180_l769_769766


namespace toy_car_production_l769_769552

theorem toy_car_production (yesterday today total : ℕ) 
  (hy : yesterday = 60)
  (ht : today = 2 * yesterday) :
  total = yesterday + today :=
by
  sorry

end toy_car_production_l769_769552


namespace correct_equation_l769_769493

theorem correct_equation (a b : ℝ) : 
  (¬ (a^2 + a^3 = a^6)) ∧ (¬ ((ab)^2 = ab^2)) ∧ (¬ ((a+b)^2 = a^2 + b^2)) ∧ ((a+b)*(a-b) = a^2 - b^2) :=
by {
  sorry
}

end correct_equation_l769_769493


namespace no_20_odd_rows_15_odd_columns_l769_769662

theorem no_20_odd_rows_15_odd_columns (n : ℕ) (table : ℕ → ℕ → bool) (cross_count 
  : ℕ) 
  (odd_rows : ℕ → bool) 
  (odd_columns : ℕ → bool) :
  (∀ i, i < n → (odd_rows i = true ↔ ∃ j, j < n ∧ table i j = true ∧ cross_count = 20))
  → (∀ j, j < n → (odd_columns j = true ↔ ∃ i, i < n ∧ table i j = true ∧ cross_count = 15))
  → false := 
sorry

end no_20_odd_rows_15_odd_columns_l769_769662


namespace marilyn_initial_bottle_caps_l769_769924

theorem marilyn_initial_bottle_caps (x : ℕ) (h : x - 36 = 15) : x = 51 :=
sorry

end marilyn_initial_bottle_caps_l769_769924


namespace calculate_final_amount_l769_769162

noncomputable def compound_interest (P r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem calculate_final_amount :
  compound_interest 8000 0.07 2 5 ≈ 11284.8 :=
by
  sorry

end calculate_final_amount_l769_769162


namespace matthew_total_time_l769_769005

def assemble_time : ℝ := 1
def bake_time_normal : ℝ := 1.5
def decorate_time : ℝ := 1
def bake_time_double : ℝ := bake_time_normal * 2

theorem matthew_total_time :
  assemble_time + bake_time_double + decorate_time = 5 := 
by 
  -- The proof will be filled in here
  sorry

end matthew_total_time_l769_769005


namespace area_between_circles_of_octagon_l769_769580

-- Define some necessary geometric terms and functions
noncomputable def cot (θ : ℝ) : ℝ := 1 / Real.tan θ
noncomputable def csc (θ : ℝ) : ℝ := 1 / Real.sin θ

/-- The main theorem stating the area between the inscribed and circumscribed circles of a regular octagon is π. -/
theorem area_between_circles_of_octagon :
  let side_length := 2
  let θ := Real.pi / 8 -- 22.5 degrees in radians
  let apothem := cot θ
  let circum_radius := csc θ
  let area_between_circles := π * (circum_radius^2 - apothem^2)
  area_between_circles = π :=
by
  sorry

end area_between_circles_of_octagon_l769_769580


namespace Petya_lives_in_6th_entrance_l769_769027

-- Definitions follows conditions given in problem.

def VasyaEntrance : ℕ := 4

axiom shortest_path_invariant_for_Petya :
  ∀ x y : ℕ, (shortest_path_around_house x = shortest_path_around_house y)

def PetyaEntrance : ℕ := 6

theorem Petya_lives_in_6th_entrance :
  ∃ (PetyaEntrance : ℕ), PetyaEntrance = 6 :=
by
  use 6
  sorry

end Petya_lives_in_6th_entrance_l769_769027


namespace even_product_probability_l769_769597

/-
Each of two boxes contains four chips numbered 1, 2, 4, and 5.
A chip is drawn randomly from each box, and the numbers on these two chips are multiplied.
We want to prove the probability that their product is even.
-/

def chips_box : List ℕ := [1, 2, 4, 5]

/-- A pair of numbers is drawn, one from each box --/
def pair_draw := (chips_box.product chips_box)

-- Counting the total number of pairs
def total_outcomes := pair_draw.length

/-- Filtering pairs where the product is even --/
def favorable_pairs : List (ℕ × ℕ) := pair_draw.filter (λ pair, ((pair.1 * pair.2) % 2 = 0))

-- Counting the number of favorable pairs
def favorable_outcomes := favorable_pairs.length

-- Computing the probability
def probability_even_product := (favorable_outcomes : ℚ) / total_outcomes

theorem even_product_probability :
  probability_even_product = (3 / 4) :=
sorry

end even_product_probability_l769_769597


namespace problem_statement_l769_769696

noncomputable def seq : ℕ → ℝ
| 1       := 0
| (n + 1) := (seq n - real.sqrt 3) / (real.sqrt 3 * seq n + 1)

theorem problem_statement : seq 2016 = real.sqrt 3 :=
  by
  sorry

end problem_statement_l769_769696


namespace factors_of_180_multiples_of_15_l769_769792

theorem factors_of_180_multiples_of_15 :
  (finset.filter (λ x : ℕ, x % 15 = 0) (finset.filter (λ x : ℕ, 180 % x = 0) (finset.range (180 + 1)))).card = 6 :=
sorry

end factors_of_180_multiples_of_15_l769_769792


namespace alissa_earrings_l769_769187

theorem alissa_earrings :
  ∀ (bought_pairs given_earrings : ℕ), 
    bought_pairs = 12 →
    (∀ pairs_per_earring, pairs_per_earring = 2 → 
      (bought_pairs * pairs_per_earring) / 2 = given_earrings →
      given_earrings * 3 = 36) :=
begin
  intros bought_pairs given_earrings bought_pairs_eq pairs_per_earring_eq division_eq,
  unfold pairs_per_earring at *,
  rw pairs_per_earring_eq at *,
  rw bought_pairs_eq at *,
  rw mul_comm at *,
  have bought_earrings : 12 * 2 = 24 := by norm_num,
  rw ← bought_earrings at *,
  have given_earrings_calc : 24 / 2 = 12 := by norm_num,
  rw ← given_earrings_calc at division_eq,
  sorry
end

end alissa_earrings_l769_769187


namespace sum_of_distances_equals_side_lengths_l769_769512

variables {P S R F Q: Type} {a b: ℝ}
variables [metric_space P] [metric_space S] [metric_space R] [metric_space F] [metric_space Q]
variables [metric_space (rectangle : Type)] [metric_space (diagonal : Type)]
variables {ABCD : rectangle} {AF PQ : diagonal}

// Define a rectangle ABCD with given side lengths
def is_rectangle (ABCD : rectangle) : Prop :=
  ∃ (A B C D : P), dist A B = a ∧ dist B C = b ∧ dist C D = a ∧ dist D A = b ∧
  ∃ (diagonal_AC diagonal_BD : diagonal),
    dist A diagonal_BD = dist C diagonal_BD ∧
    dist B diagonal_AC = dist D diagonal_AC

-- Define perpendicular relations
def is_perpendicular (PS PR AF : diagonal) : Prop :=
  dist P S ⊥ dist P diagonal_BD ∧
  dist P R ⊥ dist P diagonal_AC ∧
  dist A F ⊥ dist A diagonal_BD ∧
  dist P Q ⊥ dist P AF

-- Define distances along diagonals and sum relations
def distance_sum (PR PS : ℝ) : Prop :=
  PR + PS = a + b

-- Main theorem to prove: Given the conditions, PR + PS = a + b
theorem sum_of_distances_equals_side_lengths
  (h1 : is_rectangle ABCD)
  (h2 : is_perpendicular PS PR AF)
  (h3 : distance_sum PR PS) : PR + PS = a + b :=
sorry

end sum_of_distances_equals_side_lengths_l769_769512


namespace adjusted_average_and_variance_l769_769859

-- Define constants
def n : ℕ := 35
def mu_original : ℝ := 185
def h_wrong : ℝ := 166
def h_correct : ℝ := 106

-- Define the function to calculate the corrected total height
def total_height_corrected (n : ℕ) (mu_original : ℝ) (h_wrong : ℝ) (h_correct : ℝ) : ℝ :=
  n * mu_original - (h_wrong - h_correct)

-- Define the function to calculate the adjusted average height
def adjusted_average_height (total_height_corrected : ℝ) (n : ℕ) : ℝ :=
  total_height_corrected / n

-- Define the change in variance calculation
def change_in_variance (h_wrong : ℝ) (mu_original : ℝ) (h_correct : ℝ) (mu_adjusted : ℝ) : ℝ :=
  (((h_wrong - mu_original) ^ 2) - ((h_correct - mu_adjusted) ^ 2)) / n

-- Statement to be proved
theorem adjusted_average_and_variance :
  adjusted_average_height (total_height_corrected n mu_original h_wrong h_correct) n = 183.29 ∧
  change_in_variance h_wrong mu_original h_correct 183.29 = -160.41326 :=
by
  sorry

end adjusted_average_and_variance_l769_769859


namespace pythagorean_triples_seventh_group_pythagorean_triples_general_l769_769936

theorem pythagorean_triples_seventh_group :
    (2 * 7 + 1 = 15) ∧ (2 * 7 * (7 + 1) = 112) ∧ (2 * 7 * (7 + 1) + 1 = 113) := 
by
  sorry

theorem pythagorean_triples_general (n : ℕ) :
    ∃ a b c : ℕ, a = 2 * n + 1 ∧ b = 2 * n * (n + 1) ∧ c = 2 * n * (n + 1) + 1 := 
by
  use [2 * n + 1, 2 * n * (n + 1), 2 * n * (n + 1) + 1]
  sorry

end pythagorean_triples_seventh_group_pythagorean_triples_general_l769_769936


namespace ratio_S1_S2_l769_769235

-- Definitions of S1 and S2
def S1 : ℝ := ∑ k in (range 18).map (λ k, (-1)^(k + 1) * (1 / 2^k))
def S2 : ℝ := ∑ k in (range 18).map (λ k, (-1)^k * (1 / 2^(k+1)))

-- Main theorem where we prove the ratio S1/S2
theorem ratio_S1_S2 : (S1 / S2) = 1 := by sorry

end ratio_S1_S2_l769_769235


namespace mulch_price_per_pound_l769_769516

noncomputable def price_per_pound (total_cost : ℝ) (total_tons : ℝ) (pounds_per_ton : ℝ) : ℝ :=
  total_cost / (total_tons * pounds_per_ton)

theorem mulch_price_per_pound :
  price_per_pound 15000 3 2000 = 2.5 :=
by
  sorry

end mulch_price_per_pound_l769_769516


namespace part1_proof_part2_proof_l769_769243

section

-- Definitions for the given problem.
def maintaining_value_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (a < b) ∧ (∀ x y : ℝ, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y) ∧ (set.range (λ x, f x) = set.Icc a b)

-- Part 1: y = 2x^2 with maintaining value interval [0, 1/2]
def function1 : ℝ → ℝ := λ x, 2 * x ^ 2

theorem part1_proof : maintaining_value_interval function1 0 (1/2) :=
sorry

-- Part 2: f(x) = mx^2 - 3x + 4 with maintaining value interval condition on m
def function2 (m : ℝ) : ℝ → ℝ := λ x, m * x ^ 2 - 3 * x + 4

theorem part2_proof (m : ℝ) (h : 0 < m) : maintaining_value_interval (function2 m) a b → (11/16 ≤ m ∧ m < 3/4) ∨ (15/16 ≤ m ∧ m < 1) :=
sorry

end

end part1_proof_part2_proof_l769_769243


namespace invariant_PQ_length_l769_769630

-- Defining a triangle and the properties of altitudes and perpendiculars [omitted precise content for brevity]
variable {A B C P Q : Type}
variables [is_acute_triangle A B C] [altitude_foot D from A to BC]
variables [perpendicular_from D to AB is P] [perpendicular_from D to AC is Q]

theorem invariant_PQ_length : ∀ (A B C : Type) [is_acute_triangle A B C] (P Q D : Type), 
  (altitude_foot D from A to BC) →
  (perpendicular_from D to AB is P) →
  (perpendicular_from D to AC is Q) →
  length_PQ A B C P Q D = length_PQ A B C P Q (foot_of_altitude_from B to AC) := sorry

end invariant_PQ_length_l769_769630


namespace find_a_b_l769_769455

theorem find_a_b (a b : ℝ) :
  (∀ x : ℝ, (x < -2 ∨ x > 1) → (x^2 + a * x + b > 0)) →
  (a = 1 ∧ b = -2) :=
by
  sorry

end find_a_b_l769_769455


namespace sparkling_water_cost_l769_769383

theorem sparkling_water_cost
  (drinks_per_day : ℚ := 1 / 5)
  (bottle_cost : ℝ := 2.00)
  (days_in_year : ℤ := 365) :
  (drinks_per_day * days_in_year) * bottle_cost = 146 :=
by
  sorry

end sparkling_water_cost_l769_769383


namespace ctg_product_three_halves_l769_769072

noncomputable def ctg (x : ℝ) : ℝ := cos x / sin x

theorem ctg_product_three_halves
  (x y : ℝ)
  (h1 : ctg x + ctg y = 3)
  (h2 : 2 * sin (2 * (x + y)) = sin (2 * x) * sin (2 * y)) :
  ctg x * ctg y = 3 / 2 :=
by
  sorry

end ctg_product_three_halves_l769_769072


namespace bisect_MK_l769_769129

variable {A B C A1 B1 C1 O M K : Type}
variables [EuclideanGeometry O M K A B C A1 B1 C1]

-- Conditions
def is_circumcenter (O : Type) (△: triangle A B C) : Prop := sorry  -- Definition of circumcenter
def is_orthocenter (M : Type) (△: triangle A B C) : Prop := sorry  -- Definition of orthocenter

def reflection (P : Type) (bisector : Type) : Type := sorry -- Reflection definition

def is_incenter (K : Type) (△: triangle A1 B1 C1) : Prop := sorry -- Definition of incenter

-- Problem Statement
theorem bisect_MK (O M K A B C A1 B1 C1: Type)
  (h1 : is_circumcenter O (triangle.mk A B C))
  (h2 : is_orthocenter M (triangle.mk A B C))
  (h3 : A1 = reflection A (perpendicular_bisector (segment.mk B C)))
  (h4 : B1 = reflection B (perpendicular_bisector (segment.mk C A)))
  (h5 : C1 = reflection C (perpendicular_bisector (segment.mk A B)))
  (h6 : is_incenter K (triangle.mk A1 B1 C1)) : 
  midpoint O M K :=
sorry

end bisect_MK_l769_769129


namespace factors_of_180_multiple_of_15_count_l769_769806

theorem factors_of_180_multiple_of_15_count :
  ∃ n : Nat, n = 6 ∧ ∀ x : Nat, x ∣ 180 → 15 ∣ x → x > 0 :=
by
  sorry

end factors_of_180_multiple_of_15_count_l769_769806


namespace principal_amount_l769_769507

theorem principal_amount (P R : ℝ) : 
  (P + P * R * 2 / 100 = 850) ∧ (P + P * R * 7 / 100 = 1020) → P = 782 :=
by
  sorry

end principal_amount_l769_769507


namespace red_apples_ordered_l769_769979

variable (R : ℕ)

theorem red_apples_ordered (h : R + 32 = 2 + 73) : R = 43 := by
  sorry

end red_apples_ordered_l769_769979


namespace percentage_change_area_l769_769431

theorem percentage_change_area (L B : ℝ) (h : L > 0 ∧ B > 0) :
  let A1 := L * B
  let L2 := L / 2
  let B2 := 3 * B
  let A2 := L2 * B2
  let percentage_change := ((A2 - A1) / A1) * 100
  percentage_change = 50 := by
  have A1_def := calc
    A1 = L * B : rfl
  have A2_def := calc
    A2 = (L / 2) * (3 * B) : rfl
    ... = (3 / 2) * (L * B) : by ring
  have pc_def := calc
    percentage_change = ((A2 - A1) / A1) * 100 : rfl
    ... = (( (3 / 2) * (L * B) - (L * B) ) / (L * B) ) * 100 : by rw [A1_def, A2_def]
    ... = (1 / 2) * 100 : by ring
  exact pc_def

end percentage_change_area_l769_769431


namespace ratio_S₁_S₂_l769_769234

noncomputable def S₁ : ℝ := (List.sum $ List.map (λ k, (-1)^k * (1 / (2^k : ℝ))) (List.range' 1 18)).toReal
noncomputable def S₂ : ℝ := (List.sum $ List.map (λ k, (-1)^(k+1) * (1 / (2^k : ℝ))) (List.range' 1 18)).toReal

theorem ratio_S₁_S₂ : S₁ / S₂ = -2 :=
sorry

end ratio_S₁_S₂_l769_769234


namespace technicians_count_l769_769330

theorem technicians_count 
  (T R : ℕ) 
  (h1 : T + R = 14) 
  (h2 : 12000 * T + 6000 * R = 9000 * 14) : 
  T = 7 :=
by
  sorry

end technicians_count_l769_769330


namespace count_factors_of_180_multiple_of_15_l769_769779

def is_factor (n k : ℕ) : Prop := n % k = 0

def is_multiple_of (k m : ℕ) : Prop := k % m = 0

theorem count_factors_of_180_multiple_of_15 : 
  (finset.filter (λ f, is_factor 180 f ∧ is_multiple_of f 15) (finset.range (181))).card = 6 := 
by 
sorry

end count_factors_of_180_multiple_of_15_l769_769779


namespace probability_divisible_by_4_l769_769631

/-- Problem Statement: From the set of numbers {1, 2, 3, 4, 5}, if three numbers are randomly selected,
    what is the probability that the sum of the selected numbers can be divisible by 4?
-/
noncomputable def num_set : set ℕ := {1, 2, 3, 4, 5}

noncomputable def chosen_subsets : finset (finset ℕ) :=
  {s : finset ℕ | s ∈ num_set.powerset.filter (λ x, x.card = 3)}

noncomputable def favorable_subsets : finset (finset ℕ) :=
  {s : finset ℕ | s ∈ chosen_subsets ∧ s.sum % 4 = 0}

theorem probability_divisible_by_4 : 
  (favorable_subsets.card : ℚ) / (chosen_subsets.card : ℚ) = 3 / 10 := 
by
  sorry

end probability_divisible_by_4_l769_769631


namespace div_equiv_l769_769576

theorem div_equiv (a b c d : ℚ) (h1 : a = 120) (h2 : b = 6) (h3 : c = 2) (h4 : d = 4) :
  a / (b / c + d) = 17 + 1 / 7 :=
by
  have h5 : b / c = 3 := sorry
  have h6 : b / c + d = 7 := sorry
  have h7 : a / 7 = 17 + 1 / 7 := sorry
  exact h7

end div_equiv_l769_769576


namespace collinearity_of_intersection_points_l769_769898

variable {A B C H M N P Q : Point}

-- Definitions and conditions
-- Assuming we have types and instance definitions as per the context

def is_triangle (A B C : Point) : Prop := -- predicate stating A, B, C form a triangle
  -- Additional conditions specific to triangle formation could go here
  sorry

def orthocenter (H A B C : Point) : Prop := -- predicate stating H is the orthocenter of triangle ABC
  sorry

def on_segment (X A B : Point) : Prop := -- predicate stating X is on the segment AB
  sorry

def diameter_circle_intersections (BN CM : Circle) (P Q : Point) : Prop := -- predicate stating P, Q are intersections of circles with diameters BN and CM
  sorry

def collinear (X Y Z : Point) : Prop := -- predicate stating X, Y, Z are collinear
  sorry

-- Mathematical equivalence proof hypothesis and conclusion
theorem collinearity_of_intersection_points {A B C H M N P Q : Point}
    (h_triangle : is_triangle A B C)
    (h_orthocenter : orthocenter H A B C)
    (h_on_segment_M : on_segment M A B)
    (h_on_segment_N : on_segment N A C)
    (h_diameter_intersections : diameter_circle_intersections ⟨B, N⟩ ⟨C, M⟩ P Q) :
  collinear P Q H :=
begin
  sorry
end

end collinearity_of_intersection_points_l769_769898


namespace sequence_a4_l769_769980

def sequence (n : ℕ) : ℚ :=
match n with
| 0 => 0  -- since sequences are generally 1-indexed, n = 0 case is set to 0
| 1 => 1
| (n+1) => 1 / sequence n + 1

theorem sequence_a4 : sequence 4 = 5 / 3 :=
by sorry

end sequence_a4_l769_769980


namespace calculate_total_payment_l769_769176

theorem calculate_total_payment
(adult_price : ℕ := 30)
(teen_price : ℕ := 20)
(child_price : ℕ := 15)
(num_adults : ℕ := 4)
(num_teenagers : ℕ := 4)
(num_children : ℕ := 2)
(num_activities : ℕ := 5)
(has_coupon : Bool := true)
(soda_price : ℕ := 5)
(num_sodas : ℕ := 5)

(total_admission_before_discount : ℕ := 
  num_adults * adult_price + num_teenagers * teen_price + num_children * child_price)
(discount_on_activities : ℕ := if num_activities >= 7 then 15 else if num_activities >= 5 then 10 else if num_activities >= 3 then 5 else 0)
(admission_after_activity_discount : ℕ := 
  total_admission_before_discount - total_admission_before_discount * discount_on_activities / 100)
(additional_discount : ℕ := if has_coupon then 5 else 0)
(admission_after_all_discounts : ℕ := 
  admission_after_activity_discount - admission_after_activity_discount * additional_discount / 100)

(total_cost : ℕ := admission_after_all_discounts + num_sodas * soda_price) :
total_cost = 22165 := 
sorry

end calculate_total_payment_l769_769176


namespace quadratic_function_properties_l769_769906

theorem quadratic_function_properties 
  (a b c : ℝ)
  (h0 : a ≠ 0)
  (h1 : ∀ x : ℝ, f x = a * x^2 + b * x + c)
  (h2 : f 0 = 2)
  (h3 : ∀ x : ℝ, f (x + 1) - f x = -2 * x - 1) :
  (∀ x : ℝ, f x = -x^2 + 2) ∧
  (∀ t : ℝ, 
    (t < -2 -> g t = -t^2 - 4 * t - 2) ∧ 
    (-2 ≤ t ∧ t ≤ 0 -> g t = 2) ∧ 
    (t > 0 -> g t = -t^2 + 2)) :=
by
  sorry

end quadratic_function_properties_l769_769906


namespace ratio_S1_S2_l769_769236

-- Definitions of S1 and S2
def S1 : ℝ := ∑ k in (range 18).map (λ k, (-1)^(k + 1) * (1 / 2^k))
def S2 : ℝ := ∑ k in (range 18).map (λ k, (-1)^k * (1 / 2^(k+1)))

-- Main theorem where we prove the ratio S1/S2
theorem ratio_S1_S2 : (S1 / S2) = 1 := by sorry

end ratio_S1_S2_l769_769236


namespace positive_factors_of_180_multiple_of_15_count_l769_769762

theorem positive_factors_of_180_multiple_of_15_count : 
  let factorization_180 := 2^2 * 3^2 * 5;
  let is_multiple_of_15 (n : ℕ) := ∃ k1 k2 : ℕ, k1 * 3 = n ∧ k2 * 5 = n;
  let factors_180 := { n : ℕ | n ∣ factorization_180 };
  { n : ℕ | n ∈ factors_180 ∧ is_multiple_of_15 n }.card = 6 :=
by
  sorry

end positive_factors_of_180_multiple_of_15_count_l769_769762


namespace factors_of_180_multiples_of_15_l769_769790

theorem factors_of_180_multiples_of_15 :
  (finset.filter (λ x : ℕ, x % 15 = 0) (finset.filter (λ x : ℕ, 180 % x = 0) (finset.range (180 + 1)))).card = 6 :=
sorry

end factors_of_180_multiples_of_15_l769_769790


namespace distance_from_center_of_circle_to_line_l769_769336

open Real

theorem distance_from_center_of_circle_to_line : 
  let l := (λ (t : ℝ), (t, t + 1)) in
  let C := (λ (θ : ℝ), (cos θ + 1, sin θ)) in
  let line_equation := (λ (x y : ℝ), x - y + 1) in
  let center_of_circle := (1 : ℝ, 0 : ℝ) in
  |line_equation center_of_circle.1 center_of_circle.2| / sqrt (1 ^ 2 + (-1) ^ 2) = sqrt 2 :=
by 
  sorry

end distance_from_center_of_circle_to_line_l769_769336


namespace range_of_a_l769_769270

noncomputable def piecewise_function (x k a : ℝ) : ℝ :=
if h : x ∈ [-1, k] then 
  real.logb (1/2) (-x+1) - 1
else 
  -2 * |x - 1|

theorem range_of_a (k a : ℝ) :
  (∃ k : ℝ, ∀ x ∈ ([−1, k] ∪ (k, a)), piecewise_function x k a ∈ [-2, 0]) →
  (1 / 2 < a ∧ a ≤ 2) :=
sorry

end range_of_a_l769_769270


namespace count_solutions_eq_287_l769_769294

noncomputable def count_solutions : ℕ :=
  (({n | ∃ x : ℝ, n < 500 ∧ (⌊x⌋.to_nat ≥ 0) ∧ x^(⌊x⌋.to_nat) = n} : set ℕ).to_finset.card)

theorem count_solutions_eq_287 : count_solutions = 287 :=
  sorry

end count_solutions_eq_287_l769_769294


namespace part_a_impossible_part_b_possible_l769_769651

-- Statement for part (a)
theorem part_a_impossible (table : ℕ → ℕ → bool) (n : ℕ) 
    (odd_row_count : fin n → bool) (odd_col_count : fin n → bool)
    (odd_count_r : ℕ) (odd_count_c : ℕ) (cross_in_cell : fin n → fin n → bool) :
    (∀ r : fin n, odd_row_count r = odd (fin n) (\sum c : fin n, cross_in_cell r c)) →
    (∀ c : fin n, odd_col_count c = odd (fin n) (\sum r : fin n, cross_in_cell r c)) →
    odd_count_r = 20 → odd_count_c = 15 → False :=
sorry

-- Statement for part (b)
theorem part_b_possible (table : ℕ → ℕ → bool) 
    (n : ℕ) (cross_count : ℕ) (row_count : fin n → ℕ) (col_count : fin n → ℕ)
    (cross_in_cell : fin n → fin n → bool) :
    n = 16 → cross_count = 126 →
    (∀ r : fin n, odd (row_count r)) →
    (∀ c : fin n, odd (col_count c)) →
    (∃ table, (∀ r c, cross_in_cell r c = (table r c)) ∧ (\sum r, row_count r = 126) ∧ (\sum c, col_count c = 126)) :=
sorry

end part_a_impossible_part_b_possible_l769_769651


namespace real_solution_to_y_abs_y_eq_neg_3y_plus_5_l769_769620

theorem real_solution_to_y_abs_y_eq_neg_3y_plus_5 :
  let y := (-3 + Real.sqrt 29) / 2
  in y * |y| = -3 * y + 5 := 
by
  sorry

end real_solution_to_y_abs_y_eq_neg_3y_plus_5_l769_769620


namespace roots_of_equation_l769_769621

noncomputable def equation (z : ℂ) : Prop := z^2 + 2*z = 7 + 2*complex.i

theorem roots_of_equation : equation (2 + (1 / 3) * complex.i) ∧ equation (-4 - (1 / 3) * complex.i) :=
by
  sorry

end roots_of_equation_l769_769621


namespace percentage_change_in_area_halved_length_tripled_breadth_l769_769430

def original_area (L B : ℝ) : ℝ := L * B
def new_area (L B : ℝ) : ℝ := (L / 2) * (3 * B)

def percentage_change (A1 A2 : ℝ) : ℝ :=
  ((A2 - A1) / A1) * 100

theorem percentage_change_in_area_halved_length_tripled_breadth
  (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  percentage_change (original_area L B) (new_area L B) = 50 :=
begin
  sorry
end

end percentage_change_in_area_halved_length_tripled_breadth_l769_769430


namespace percentage_change_in_area_of_rectangle_l769_769441

theorem percentage_change_in_area_of_rectangle
  (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  let A1 := L * B
  let A2 := (L / 2) * (3 * B)
  let percentage_change := ((A2 - A1) / A1) * 100
  percentage_change = 50 := by
  let A1 := L * B
  let A2 := (L / 2) * (3 * B)
  let percentage_change := ((A2 - A1) / A1) * 100
  calc percentage_change
      = ((A2 - A1) / A1) * 100 : rfl
  ... = ((((L / 2) * (3 * B)) - (L * B)) / (L * B)) * 100 : by rw [A1, A2]
  ... = ((3 / 2 * A1 - A1) / A1) * 100 : by rw [A2, ←mul_assoc, mul_div_cancel' _ (two_ne_zero : 2 ≠ 0)]
  ... = ((3 / 2 - 1) * A1 / A1) * 100 : by simp only [mul_sub, mul_div_cancel' _ (ne_of_gt hL)]
  ... = (1 / 2) * 100 : by rw [div_mul_cancel' (ne_of_gt (lt_of_lt_of_le zero_lt_one (le_of_lt ((mul_lt_iff_lt_one_left (lt_of_lt_of_le zero_lt_one hL)).mpr (half_pos (lt_of_le_of_ne (le_of_lt (lt_of_lt_of_le (lt_one_mul_self (lt_of_le_of_lt (zero_le_two) (lt_add_one 1))) (lt_of_lt_of_le (two_le_one) (lt_one 2))))))))) so))
  ... = 50 : rfl

end percentage_change_in_area_of_rectangle_l769_769441


namespace trapezium_distance_l769_769613

variable (a b h : ℝ)

theorem trapezium_distance (h_pos : 0 < h) (a_pos : 0 < a) (b_pos : 0 < b)
  (area_eq : 270 = 1/2 * (a + b) * h) (a_eq : a = 20) (b_eq : b = 16) : h = 15 :=
by {
  sorry
}

end trapezium_distance_l769_769613


namespace avg_A_less_avg_B_avg_20_points_is_6_6_l769_769471

noncomputable def scores_A : List ℕ := [4, 5, 5, 6, 6, 7, 7, 8, 8, 9]
noncomputable def scores_B : List ℕ := [2, 5, 6, 6, 7, 7, 7, 8, 9, 10]
noncomputable def variance_A : ℝ := 2.25
noncomputable def variance_B : ℝ := 4.41

theorem avg_A_less_avg_B :
  let avg_A := (scores_A.sum.to_float / scores_A.length) 
  let avg_B := (scores_B.sum.to_float / scores_B.length) 
  avg_A < avg_B := 
by
  sorry

theorem avg_20_points_is_6_6 :
  let avg_A := (scores_A.sum.to_float / scores_A.length) 
  let avg_B := (scores_B.sum.to_float / scores_B.length) 
  let avg_20 := ((avg_A * scores_A.length + avg_B * scores_B.length) / (scores_A.length + scores_B.length))
  avg_20 = 6.6 := 
by
  sorry

end avg_A_less_avg_B_avg_20_points_is_6_6_l769_769471


namespace percentage_change_area_l769_769444

theorem percentage_change_area (L B : ℝ) (hL : L > 0) (hB : B > 0) :
    let A := L * B
    let L' := L / 2
    let B' := 3 * B
    let A' := (L / 2) * (3 * B)
    (A' - A) / A * 100 = 50 :=
by
  let A := L * B
  let A' := (L / 2) * (3 * B)
  have h : (A' - A) / A * 100 = (1 / 2) * 100 := sorry
  exact h

end percentage_change_area_l769_769444


namespace factors_of_180_multiple_of_15_count_l769_769800

theorem factors_of_180_multiple_of_15_count :
  ∃ n : Nat, n = 6 ∧ ∀ x : Nat, x ∣ 180 → 15 ∣ x → x > 0 :=
by
  sorry

end factors_of_180_multiple_of_15_count_l769_769800


namespace parallel_segments_l769_769034

structure Point2D where
  x : Int
  y : Int

def vector (P Q : Point2D) : Point2D :=
  { x := Q.x - P.x, y := Q.y - P.y }

def is_parallel (v1 v2 : Point2D) : Prop :=
  ∃ k : Int, v2.x = k * v1.x ∧ v2.y = k * v1.y 

theorem parallel_segments :
  let A := { x := 1, y := 3 }
  let B := { x := 2, y := -1 }
  let C := { x := 0, y := 4 }
  let D := { x := 2, y := -4 }
  is_parallel (vector A B) (vector C D) := 
  sorry

end parallel_segments_l769_769034


namespace number_of_green_balls_l769_769137

-- Define the problem statement and conditions
def total_balls : ℕ := 12
def probability_both_green (g : ℕ) : ℚ := (g / 12) * ((g - 1) / 11)

-- The main theorem statement
theorem number_of_green_balls (g : ℕ) (h : probability_both_green g = 1 / 22) : g = 3 :=
sorry

end number_of_green_balls_l769_769137


namespace petya_can_force_difference_2014_l769_769017

theorem petya_can_force_difference_2014 :
  ∀ (p q r : ℚ), ∃ (a b c : ℚ), ∀ (x : ℝ), (x^3 + a * x^2 + b * x + c = 0) → 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 - x2 = 2014 :=
by sorry

end petya_can_force_difference_2014_l769_769017


namespace tangents_circle_ratio_invariance_l769_769919

variables {R1 R2 : ℝ} {P Q X Y: ℝ}

theorem tangents_circle_ratio_invariance 
  (h_touch1: P = 2 * R1 * sin (π * R1))
  (h_touch2: Q = 2 * R2 * sin (π * R2)) 
  (h_parallel: X * P = Y * Q):
  XP / YQ = real.sqrt (R1 / R2) :=
sorry

end tangents_circle_ratio_invariance_l769_769919


namespace percentage_change_area_l769_769434

theorem percentage_change_area (L B : ℝ) (h : L > 0 ∧ B > 0) :
  let A1 := L * B
  let L2 := L / 2
  let B2 := 3 * B
  let A2 := L2 * B2
  let percentage_change := ((A2 - A1) / A1) * 100
  percentage_change = 50 := by
  have A1_def := calc
    A1 = L * B : rfl
  have A2_def := calc
    A2 = (L / 2) * (3 * B) : rfl
    ... = (3 / 2) * (L * B) : by ring
  have pc_def := calc
    percentage_change = ((A2 - A1) / A1) * 100 : rfl
    ... = (( (3 / 2) * (L * B) - (L * B) ) / (L * B) ) * 100 : by rw [A1_def, A2_def]
    ... = (1 / 2) * 100 : by ring
  exact pc_def

end percentage_change_area_l769_769434


namespace factors_of_180_multiple_of_15_count_l769_769802

theorem factors_of_180_multiple_of_15_count :
  ∃ n : Nat, n = 6 ∧ ∀ x : Nat, x ∣ 180 → 15 ∣ x → x > 0 :=
by
  sorry

end factors_of_180_multiple_of_15_count_l769_769802


namespace find_xy_l769_769231

theorem find_xy : ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ x^4 = y^2 + 71 ∧ x = 6 ∧ y = 35 :=
by
  sorry

end find_xy_l769_769231


namespace smallest_integer_in_ratio_l769_769463

theorem smallest_integer_in_ratio (a b c : ℕ) 
    (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
    (h_sum : a + b + c = 100) 
    (h_ratio : c = 5 * a / 2 ∧ b = 3 * a / 2) : 
    a = 20 := 
by
  sorry

end smallest_integer_in_ratio_l769_769463


namespace not_possible_20_odd_rows_15_odd_columns_l769_769656

def odd (n : ℕ) : Prop :=
  n % 2 = 1

def even (n : ℕ) : Prop :=
  n % 2 = 0

theorem not_possible_20_odd_rows_15_odd_columns
  (table : ℕ → ℕ → Prop) -- table representing the presence of crosses
  (n : ℕ) -- number of rows and columns in the square table
  (h_square_table: ∀ i j, table i j → i < n ∧ j < n)
  (odd_rows : ℕ)
  (odd_columns : ℕ)
  (h_odd_rows : odd_rows = 20)
  (h_odd_columns : odd_columns = 15)
  (h_def_odd_row: ∀ r, (∃ m, m < n ∧ odd (finset.card {c | c < n ∧ table r c})) ↔ r < odd_rows)
  (h_def_odd_column: ∀ c, (∃ m, m < n ∧ odd (finset.card {r | r < n ∧ table r c})) ↔ c < odd_columns)
  : false :=
by
  sorry

end not_possible_20_odd_rows_15_odd_columns_l769_769656


namespace shaded_region_area_eq_one_l769_769222

-- Definition of point and line equations
structure Point where
  x : ℝ
  y : ℝ

def line1 (x : ℝ) : ℝ := -1/2 * x + 5
def line2 (x : ℝ) : ℝ := -x + 6

-- Define the problem statement
theorem shaded_region_area_eq_one :
  let y_intersect (x : ℝ) := line2 x - line1 x,
  let integral_value := ∫ x in 0..2, y_intersect x
  integral_value = 1 := by
  sorry

end shaded_region_area_eq_one_l769_769222


namespace houses_with_neither_l769_769123

theorem houses_with_neither (T G P GP N : ℕ) (hT : T = 65) (hG : G = 50) (hP : P = 40) (hGP : GP = 35) (hN : N = T - (G + P - GP)) :
  N = 10 :=
by
  rw [hT, hG, hP, hGP] at hN
  exact hN

-- Proof is not required, just the statement is enough.

end houses_with_neither_l769_769123


namespace correct_operation_l769_769491

variable {a b : ℝ}

def conditionA : Prop := a^2 + a^3 = a^6
def conditionB : Prop := (a * b)^2 = a * b^2
def conditionC : Prop := (a + b)^2 = a^2 + b^2
def conditionD : Prop := (a + b) * (a - b) = a^2 - b^2

theorem correct_operation : conditionD ∧ ¬conditionA ∧ ¬conditionB ∧ ¬conditionC := by
  sorry

end correct_operation_l769_769491


namespace factors_of_180_multiples_of_15_l769_769789

theorem factors_of_180_multiples_of_15 :
  (finset.filter (λ x : ℕ, x % 15 = 0) (finset.filter (λ x : ℕ, 180 % x = 0) (finset.range (180 + 1)))).card = 6 :=
sorry

end factors_of_180_multiples_of_15_l769_769789


namespace sinA_value_find_b_c_l769_769259

-- Define the conditions
def triangle (A B C : Type) (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

variable {A B C : Type} (a b c : ℝ)
variable {S_triangle_ABC : ℝ}
variable {cosB : ℝ}

-- Given conditions
axiom cosB_val : cosB = 3 / 5
axiom a_val : a = 2

-- Problem 1: Prove sinA = 2/5 given additional condition b = 4
axiom b_val : b = 4

theorem sinA_value (h_triangle : triangle A B C a b c) (h_cosB : cosB = 3/5) (h_a : a = 2) (h_b : b = 4) : 
  ∃ sinA : ℝ, sinA = 2 / 5 :=
sorry

-- Problem 2: Prove b = sqrt(17) and c = 5 given the area
axiom area_val : S_triangle_ABC = 4

theorem find_b_c (h_triangle : triangle A B C a b c) (h_cosB : cosB = 3/5) (h_a : a = 2) (h_area : S_triangle_ABC = 4) : 
  ∃ b c : ℝ, b = Real.sqrt 17 ∧ c = 5 :=
sorry

end sinA_value_find_b_c_l769_769259


namespace percentage_change_area_l769_769447

theorem percentage_change_area (L B : ℝ) (hL : L > 0) (hB : B > 0) :
    let A := L * B
    let L' := L / 2
    let B' := 3 * B
    let A' := (L / 2) * (3 * B)
    (A' - A) / A * 100 = 50 :=
by
  let A := L * B
  let A' := (L / 2) * (3 * B)
  have h : (A' - A) / A * 100 = (1 / 2) * 100 := sorry
  exact h

end percentage_change_area_l769_769447


namespace correct_operation_l769_769112

theorem correct_operation (a b : ℝ) :
  (3 * a^2 - a^2 ≠ 3) ∧
  ((a + b)^2 ≠ a^2 + b^2) ∧
  ((-3 * a * b^2)^2 ≠ -6 * a^2 * b^4) →
  a^3 / a^2 = a :=
by
sorry

end correct_operation_l769_769112


namespace num_factors_of_180_multiple_of_15_l769_769739

def is_factor (n m : Nat) : Prop := m % n = 0
def is_multiple_of_15 (n : Nat) : Prop := is_factor 15 n

theorem num_factors_of_180_multiple_of_15 : 
  (Nat.filter (fun n => is_factor n 180 ∧ is_multiple_of_15 n) (Nat.range (180 + 1))).length = 6 :=
sorry

end num_factors_of_180_multiple_of_15_l769_769739


namespace sum_of_an_bn_div_5n_l769_769896

theorem sum_of_an_bn_div_5n {a_n b_n : ℕ → ℝ} (h : ∀ n : ℕ, (1 + 2 * complex.I)^n = a_n n + b_n n * complex.I) :
  ∑' n, (a_n n * b_n n) / 5^n = 0 := 
sorry

end sum_of_an_bn_div_5n_l769_769896


namespace problem_statement_l769_769301

noncomputable def countNs : Nat :=
  let N_values := {N : ℕ | ∃ x : ℝ, 0 < N ∧ N < 500 ∧ (x ^ Nat.floor x = N)}
  N_values.toFinset.card

theorem problem_statement :
  countNs = 287 := by
  sorry

end problem_statement_l769_769301


namespace smallest_p_l769_769367

theorem smallest_p (p q : ℕ) (h1 : p + q = 2005) (h2 : (5:ℚ)/8 < p / q) (h3 : p / q < (7:ℚ)/8) : p = 772 :=
sorry

end smallest_p_l769_769367


namespace Adam_spent_21_dollars_l769_769178

-- Define the conditions as given in the problem
def initial_money : ℕ := 91
def spent_money (x : ℕ) : Prop := (initial_money - x) * 3 = 10 * x

-- The theorem we want to prove: Adam spent 21 dollars on new books
theorem Adam_spent_21_dollars : spent_money 21 :=
by sorry

end Adam_spent_21_dollars_l769_769178


namespace count_factors_of_180_multiple_of_15_l769_769713

def is_factor (x n : ℕ) := n % x = 0 -- Definition of a factor
def is_multiple (x k : ℕ) := x % k = 0 -- Definition of a multiple

theorem count_factors_of_180_multiple_of_15 :
  (finset.filter (λ x, is_multiple x 15) (finset.filter (λ x, is_factor x 180) (finset.range (180 + 1)))).card = 6 := by
  sorry

end count_factors_of_180_multiple_of_15_l769_769713


namespace sequence_sum_l769_769882

theorem sequence_sum (P Q R S T U V : ℕ) (h1 : S = 7)
  (h2 : P + Q + R = 21) (h3 : Q + R + S = 21)
  (h4 : R + S + T = 21) (h5 : S + T + U = 21)
  (h6 : T + U + V = 21) : P + V = 14 :=
by
  sorry

end sequence_sum_l769_769882


namespace determinant_power_l769_769827

theorem determinant_power (A : Matrix n n ℝ) (hA : det A = -2) : det (A^4) = 16 := 
by 
  sorry

end determinant_power_l769_769827


namespace sum_of_even_numbers_l769_769456

theorem sum_of_even_numbers (sum_eq : ∑ k in (finset.range n).filter (λ x, x % 2 = 0) = 89 * 90) (odd_n : n % 2 = 1) : n = 179 := 
  sorry

end sum_of_even_numbers_l769_769456


namespace woman_work_rate_l769_769118

theorem woman_work_rate (W : ℝ) :
  (1 / 6) + W + (1 / 9) = (1 / 3) → W = (1 / 18) :=
by
  intro h
  sorry

end woman_work_rate_l769_769118


namespace ratio_S₁_S₂_l769_769233

noncomputable def S₁ : ℝ := (List.sum $ List.map (λ k, (-1)^k * (1 / (2^k : ℝ))) (List.range' 1 18)).toReal
noncomputable def S₂ : ℝ := (List.sum $ List.map (λ k, (-1)^(k+1) * (1 / (2^k : ℝ))) (List.range' 1 18)).toReal

theorem ratio_S₁_S₂ : S₁ / S₂ = -2 :=
sorry

end ratio_S₁_S₂_l769_769233


namespace ratio_of_b_age_to_c_age_l769_769117

variable (A B C : ℕ)
variable (h1 : A = B + 2) (h2 : A + B + C = 42) (h3 : B = 16)

theorem ratio_of_b_age_to_c_age : B / C = 2 / 1 :=
by 
  have hA : A = B + 2, from h1
  have hSum : A + B + C = 42, from h2
  have hB : B = 16, from h3
  have hA_sub : 16 + 2 = 18,
  {
    calc
    16 + 2 = B + 2 : by rw hB
      ... = A : by rw h1
  }
  have hSum_sub : 18 + 16 + C = 42,
  {
    calc
    18 + 16 + C = (B + 2) + 16 + C : by rw hA
          ... = A + B + C : by rw [h1, h3]
          ... = 42 : by rw hSum
  }
  have hC : C = 42 - 34,
  {
    calc
    C = 42 - (18 + 16) : by rw [← hA_sub, hSum_sub]
      ... = 42 - 34 : by sorry
  }
  have hC_eval : C = 8, by sorry
  have hRatio : B / C = 16 / 8, by sorry
  rw hC_eval at hRatio,
  norm_num at hRatio,
  exact hRatio

end ratio_of_b_age_to_c_age_l769_769117


namespace number_of_solutions_l769_769583

open Real

def fractional_part (t : ℝ) : ℝ :=
  t - floor t

noncomputable def f (t : ℝ) : ℝ :=
  50 * sin (fractional_part t)

theorem number_of_solutions :
  { t : ℝ | 0 ≤ t ∧ t < 50 ∧ t = f t }.finite.card = 50 :=
sorry

end number_of_solutions_l769_769583


namespace ratio_length_width_l769_769071

theorem ratio_length_width (A L W : ℕ) (hA : A = 432) (hW : W = 12) (hArea : A = L * W) : L / W = 3 := 
by
  -- Placeholders for the actual mathematical proof
  sorry

end ratio_length_width_l769_769071


namespace beads_counter_representation_l769_769939

-- Given conditions
variable (a : ℕ) -- a is a natural number representing the beads in the tens place.
variable (h : a ≥ 0) -- Ensure a is non-negative since the number of beads cannot be negative.

-- The main statement to prove
theorem beads_counter_representation (a : ℕ) (h : a ≥ 0) : 10 * a + 4 = (10 * a) + 4 :=
by sorry

end beads_counter_representation_l769_769939


namespace marilyn_initial_bottle_caps_l769_769923

theorem marilyn_initial_bottle_caps (x : ℕ) (h : x - 36 = 15) : x = 51 :=
sorry

end marilyn_initial_bottle_caps_l769_769923


namespace no_20_odd_rows_15_odd_columns_l769_769665

theorem no_20_odd_rows_15_odd_columns (n : ℕ) (table : ℕ → ℕ → bool) (cross_count 
  : ℕ) 
  (odd_rows : ℕ → bool) 
  (odd_columns : ℕ → bool) :
  (∀ i, i < n → (odd_rows i = true ↔ ∃ j, j < n ∧ table i j = true ∧ cross_count = 20))
  → (∀ j, j < n → (odd_columns j = true ↔ ∃ i, i < n ∧ table i j = true ∧ cross_count = 15))
  → false := 
sorry

end no_20_odd_rows_15_odd_columns_l769_769665


namespace count_valid_N_under_500_l769_769298

def hasSolution (N : ℕ) (x : ℝ) : Prop :=
  N = x ^ (Real.floor x)

def validN (N : ℕ) : Prop :=
  ∃ x : ℝ, hasSolution N x

theorem count_valid_N_under_500 : 
  let N_set := {N : ℕ | N < 500 ∧ validN N}
  N_set.card = 287 := sorry

end count_valid_N_under_500_l769_769298


namespace annual_increase_fraction_l769_769609

theorem annual_increase_fraction (InitAmt FinalAmt : ℝ) (f : ℝ) :
  InitAmt = 51200 ∧ FinalAmt = 64800 ∧ FinalAmt = InitAmt * (1 + f)^2 →
  f = 0.125 :=
by
  intros h
  sorry

end annual_increase_fraction_l769_769609


namespace sum_even_minus_odd_from_1_to_100_l769_769244

noncomputable def sum_even_numbers : Nat :=
  (List.range' 2 99 2).sum

noncomputable def sum_odd_numbers : Nat :=
  (List.range' 1 100 2).sum

theorem sum_even_minus_odd_from_1_to_100 :
  sum_even_numbers - sum_odd_numbers = 50 :=
by
  sorry

end sum_even_minus_odd_from_1_to_100_l769_769244


namespace F_is_commutative_field_l769_769359

noncomputable def is_commutative_field (F : Type*) [field F] : Prop :=
∀ (a b : F), a * b = b * a

variables (p : ℕ) [fact (nat.prime p)]
variables (H : polynomial (zmod p))

def field_mod_H : Type* := polynomial (zmod p) ⧸ ideal.span {H}

instance : field (field_mod_H p H) :=
begin
  sorry
end

theorem F_is_commutative_field : is_commutative_field (field_mod_H p H) :=
begin
  sorry
end

end F_is_commutative_field_l769_769359


namespace smallest_total_bananas_l769_769094

theorem smallest_total_bananas :
  ∃ (a b c : ℕ), 
    (∃ (x : ℕ), 
    let t := 30 * x in 
    a = 6 * x ∧ b = 12 * x ∧ c = 12 * x ∧ 
    (t = a + b + c) ∧ 
    (2/3 * a + 1/3 * b + 5/12 * c = 4 * x) ∧ 
    (1/6 * a + 1/3 * b + 5/12 * c = 3 * x) ∧ 
    (1/6 * a + 1/3 * b + 1/6 * c = 2 * x)) ∧ 
    t = 30 :=
by {
  sorry
}

end smallest_total_bananas_l769_769094


namespace verify_true_proposition_l769_769180

-- Definitions based on proposition statements
def prop_A (a : ℝ) : Prop := a^2 ≥ 0 → a ≥ 0
def angle_supplement_greater (A : ℝ) : Prop := 180 - A > A
def corresponding_angles_equal (l₁ l₂ : ℝ) : Prop := l₁ = l₂ → l₁ = l₂  -- This is a dummy definition for the premise
def exterior_angle_triangle (∠BAC ∠ABC ∠BCA ∠BCD : ℝ) : Prop :=
  ∠BCD = ∠BAC + ∠BCA

-- Formal statement to be proven in Lean 4
theorem verify_true_proposition {a : ℝ} {∠A ∠B ∠C ∠D ∠E ∠F ∠BAC ∠ABC ∠BCA ∠BCD : ℝ} :
  ¬ prop_A a ∧ ¬ angle_supplement_greater ∠A ∧ ¬ corresponding_angles_equal ∠C ∠F ∧ exterior_angle_triangle ∠BAC ∠ABC ∠BCA ∠BCD := sorry

end verify_true_proposition_l769_769180


namespace perpendicular_given_conditions_l769_769132

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)
 
def equilateral_triangle (A B C : V) (d : ℝ) := 
  dist A B = d ∧ dist B C = d ∧ dist C A = d

theorem perpendicular_given_conditions (A B C : V) (h₁ : equilateral_triangle A B C 2)
  (h₂ : B - A = 2 • a) (h₃ : C - A = 2 • a + b) :
  inner (4 • a + b) (C - B) = 0 :=
by sorry

end perpendicular_given_conditions_l769_769132


namespace sqrt_diff_rounded_to_two_decimals_l769_769578

def approx_sqrt11 : Real := 3.31662
def approx_sqrt6 : Real := 2.44948
def expected_result : Real := 0.87

theorem sqrt_diff_rounded_to_two_decimals :
  Real.round (approx_sqrt11 - approx_sqrt6) 2 = expected_result :=
by
  sorry

end sqrt_diff_rounded_to_two_decimals_l769_769578


namespace smallest_M_value_l769_769907

theorem smallest_M_value 
  (a b c d e : ℕ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : 0 < e) 
  (h_sum : a + b + c + d + e = 2010) : 
  (∃ M, M = max (a+b) (max (b+c) (max (c+d) (d+e))) ∧ M = 671) :=
by
  sorry

end smallest_M_value_l769_769907


namespace sum_of_solutions_mod25_l769_769114

def P (x : ℤ) : ℤ := x^3 + 3 * x^2 - 2 * x + 4

theorem sum_of_solutions_mod25 :
  let solutions := (List.filter (λ x, P x % 25 = 0) (List.range 25))
  in (solutions.sum % 25 = 6) := by
  sorry

end sum_of_solutions_mod25_l769_769114


namespace percentage_change_in_area_of_rectangle_l769_769439

theorem percentage_change_in_area_of_rectangle
  (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  let A1 := L * B
  let A2 := (L / 2) * (3 * B)
  let percentage_change := ((A2 - A1) / A1) * 100
  percentage_change = 50 := by
  let A1 := L * B
  let A2 := (L / 2) * (3 * B)
  let percentage_change := ((A2 - A1) / A1) * 100
  calc percentage_change
      = ((A2 - A1) / A1) * 100 : rfl
  ... = ((((L / 2) * (3 * B)) - (L * B)) / (L * B)) * 100 : by rw [A1, A2]
  ... = ((3 / 2 * A1 - A1) / A1) * 100 : by rw [A2, ←mul_assoc, mul_div_cancel' _ (two_ne_zero : 2 ≠ 0)]
  ... = ((3 / 2 - 1) * A1 / A1) * 100 : by simp only [mul_sub, mul_div_cancel' _ (ne_of_gt hL)]
  ... = (1 / 2) * 100 : by rw [div_mul_cancel' (ne_of_gt (lt_of_lt_of_le zero_lt_one (le_of_lt ((mul_lt_iff_lt_one_left (lt_of_lt_of_le zero_lt_one hL)).mpr (half_pos (lt_of_le_of_ne (le_of_lt (lt_of_lt_of_le (lt_one_mul_self (lt_of_le_of_lt (zero_le_two) (lt_add_one 1))) (lt_of_lt_of_le (two_le_one) (lt_one 2))))))))) so))
  ... = 50 : rfl

end percentage_change_in_area_of_rectangle_l769_769439


namespace find_point_distances_to_intersection_l769_769368

noncomputable def point_distances_to_intersection : ℝ := 
  let x := Classical.some (Exists.intro 0.7569 (by sorry))
  let y := Classical.some (Exists.intro 2.9236 (by sorry))
  x + y + 30

theorem find_point_distances_to_intersection (x y : ℝ) 
  (hx : x * (x + 30) = 75) 
  (hy : y * (y + 30) = 140) : 
  point_distances_to_intersection = 33.6805 := 
by
  sorry

end find_point_distances_to_intersection_l769_769368


namespace factors_of_180_multiple_of_15_count_l769_769798

theorem factors_of_180_multiple_of_15_count :
  ∃ n : Nat, n = 6 ∧ ∀ x : Nat, x ∣ 180 → 15 ∣ x → x > 0 :=
by
  sorry

end factors_of_180_multiple_of_15_count_l769_769798


namespace sequence_values_induction_proof_l769_769695

def seq (a : ℕ → ℤ) := a 1 = 3 ∧ ∀ n : ℕ, a (n + 1) = a n ^ 2 - 2 * n * a n + 2

theorem sequence_values (a : ℕ → ℤ) (h : seq a) :
  a 2 = 5 ∧ a 3 = 7 ∧ a 4 = 9 :=
sorry

theorem induction_proof (a : ℕ → ℤ) (h : seq a) :
  ∀ n : ℕ, a n = 2 * n + 1 :=
sorry

end sequence_values_induction_proof_l769_769695


namespace students_participating_in_both_l769_769325

theorem students_participating_in_both (total_students : ℕ)
  (min_percent_one : ℝ) (max_percent_one : ℝ)
  (min_percent_two : ℝ) (max_percent_two : ℝ)
  (h_min_valid : 50 ≤ min_percent_one ∧ min_percent_one ≤ 65)
  (h_max_valid : 50 ≤ max_percent_two ∧ max_percent_two ≤ 55)
  (h_total : total_students = 1200) :
  180 ∈ set.Icc (1200 * (min_percent_one + min_percent_two - 100) / 100) 
                     (1200 * (max_percent_one + max_percent_two - 100) / 100) :=
by {
  sorry
}

#eval students_participating_in_both 1200 60 65 50 55

end students_participating_in_both_l769_769325


namespace conference_center_people_count_l769_769153

-- Definition of the conditions
def rooms : ℕ := 6
def capacity_per_room : ℕ := 80
def fraction_full : ℚ := 2/3

-- Total capacity of the conference center
def total_capacity := rooms * capacity_per_room

-- Number of people in the conference center when 2/3 full
def num_people := fraction_full * total_capacity

-- The theorem stating the problem
theorem conference_center_people_count :
  num_people = 320 := 
by
  -- This is a placeholder for the proof
  sorry

end conference_center_people_count_l769_769153


namespace rotation_value_l769_769389

theorem rotation_value (P Q R : Type) (h1 : 735 % 360 = 15) (h2 : 0 < 345 ∧ 345 < 360) :
  ∃ y : ℕ, y = 345 ∧ y < 360 :=
by
  use 345
  split
  · refl
  · exact h2.2

end rotation_value_l769_769389


namespace right_triangle_angle_bisector_eq_l769_769373

theorem right_triangle_angle_bisector_eq (A B C D : Point)
  (h_triangle : right_triangle A B C)
  (h_angle_bisector : angle_bisector B D (ray A C))
  (h_AB_pos : 0 < AB)
  (h_BC : BC = 1 + sqrt 5)
  (h_BD : BD = sqrt 5 - 1) :
  (BC - BD = 2 * AB) ↔ (1 / BD - 1 / BC = 1 / (2 * AB)) :=
sorry

end right_triangle_angle_bisector_eq_l769_769373


namespace GP_passes_through_midpoint_of_XY_l769_769892

theorem GP_passes_through_midpoint_of_XY 
  (A B C G X Y E F P : Point)
  (hG : is_centroid A B C G)
  (hE : is_on_line_segment B C E)
  (hF : is_on_line_segment B C F)
  (hEF : dist B E = dist E F ∧ dist E F = dist F C)
  (hX : is_on_line A B X)
  (hY : is_on_line A C Y)
  (h_not_collinear : ¬collinear {X, Y, G})
  (h_parallel_E : is_parallel (line_through E (parallel_line_through G X)) (line_through G X))
  (h_parallel_F : is_parallel (line_through F (parallel_line_through G Y)) (line_through G Y))
  (hP_intersection : P ≠ G ∧ intersection (line_through E (parallel_line_through G X)) (line_through F (parallel_line_through G Y)) = P)
  : passes_through_midpoint G P X Y :=
sorry

end GP_passes_through_midpoint_of_XY_l769_769892


namespace no_charming_two_digit_numbers_l769_769559

/-
Define what it means for a two-digit number to be charming.
-/
def is_charming (a b : ℕ) : Prop :=
  10 * a + b = a^2 + a * b

/-
Prove that there are no two-digit numbers that are charming.
-/
theorem no_charming_two_digit_numbers : 
  ∀ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 → ¬ is_charming a b :=
by {
  intros a b h,
  sorry
}

end no_charming_two_digit_numbers_l769_769559


namespace no_prime_numbers_divisible_by_91_l769_769821

-- Define the concept of a prime number.
def is_prime (n : ℕ) : Prop :=
  1 < n ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- Define the factors of 91.
def factors_of_91 (n : ℕ) : Prop :=
  n = 7 ∨ n = 13

-- State the problem formally: there are no prime numbers divisible by 91.
theorem no_prime_numbers_divisible_by_91 :
  ∀ p : ℕ, is_prime p → ¬ (91 ∣ p) :=
by
  intros p prime_p div91
  sorry

end no_prime_numbers_divisible_by_91_l769_769821


namespace total_pieces_correct_l769_769961

theorem total_pieces_correct :
  let bell_peppers := 10
  let onions := 7
  let zucchinis := 15
  let bell_peppers_slices := (2 * 20)  -- 25% of 10 bell peppers sliced into 20 slices each
  let bell_peppers_large_pieces := (7 * 10)  -- Remaining 75% cut into 10 pieces each
  let bell_peppers_smaller_pieces := (35 * 3)  -- Half of large pieces cut into 3 pieces each
  let onions_slices := (3 * 18)  -- 50% of onions sliced into 18 slices each
  let onions_pieces := (4 * 8)  -- Remaining 50% cut into 8 pieces each
  let zucchinis_slices := (4 * 15)  -- 30% of zucchinis sliced into 15 pieces each
  let zucchinis_pieces := (10 * 8)  -- Remaining 70% cut into 8 pieces each
  let total_slices := bell_peppers_slices + onions_slices + zucchinis_slices
  let total_pieces := bell_peppers_large_pieces + bell_peppers_smaller_pieces + onions_pieces + zucchinis_pieces
  total_slices + total_pieces = 441 :=
by
  sorry

end total_pieces_correct_l769_769961


namespace f_2_eq_9_l769_769675

def f : ℝ → ℝ :=
  λ x, if x < 0 then x^3 - 1 else -(((-x)^(3 : ℕ) - 1) * 1)

theorem f_2_eq_9 (x : ℝ) (h_dom : x ∈ set.univ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_neg_def : ∀ x, x < 0 → f x = x^3 - 1) : f 2 = 9 :=
by
  -- Here would be the proof, which we are skipping as per instructions.
  sorry

end f_2_eq_9_l769_769675


namespace log2_50_between_and_sum_eq_11_l769_769988

theorem log2_50_between_and_sum_eq_11 :
  ∃ c d : ℤ, c < d ∧ (c ≤ Real.log 50 / Real.log 2) ∧ (Real.log 50 / Real.log 2 < d) ∧ (c + d = 11) :=
by {
  let c := 5,
  let d := 6,
  use [c, d],
  split,
  { exact int.lt_succ_self c },
  split,
  { exact Real.log_le_log_of_log_le (by norm_num) (by norm_num) (by norm_num1: 32 < 50) },
  split,
  { exact Real.log_le_log_of_log_le (by norm_num) (by norm_num) (by norm_num2: 50 < 64).trans (by norm_num) },
  { exact add_self_eq_zero.mp (by norm_num) }
}

end log2_50_between_and_sum_eq_11_l769_769988


namespace domain_of_f_l769_769413

def f (x : ℝ) : ℝ := sqrt (2 - x) / x + (x - 1) ^ 0

theorem domain_of_f :
  (∀ x, x ∈ (Set.Ioo 0 1) ∪ (Set.Ioc 1 2) ↔ (x > 0 ∧ x < 1) ∨ (x > 1 ∧ x ≤ 2)) :=
begin
  sorry
end

end domain_of_f_l769_769413


namespace mechanic_worked_hours_l769_769929

theorem mechanic_worked_hours 
  (total_cost : ℝ) 
  (part_cost : ℝ) 
  (part_count : ℕ) 
  (labor_cost_per_minute : ℝ) 
  (parts_total_cost : ℝ) 
  (labor_total_cost : ℝ) 
  (hours_worked : ℝ) 
  (h1 : total_cost = 220) 
  (h2 : part_count = 2) 
  (h3 : part_cost = 20) 
  (h4 : labor_cost_per_minute = 0.5) 
  (h5 : parts_total_cost = part_count * part_cost) 
  (h6 : labor_total_cost = total_cost - parts_total_cost) 
  (labor_cost_per_hour := labor_cost_per_minute * 60) 
  (h7 : hours_worked = labor_total_cost / labor_cost_per_hour) : 
  hours_worked = 6 := 
sorry

end mechanic_worked_hours_l769_769929


namespace volume_of_pyramid_SPQR_l769_769029

-- Definitions of the points and conditions.
variables {P Q R S : Type}
variables (SP SQ SR : ℝ)
variables (h_perpendicular_1 : SP ⊥ SQ)
variables (h_perpendicular_2 : SQ ⊥ SR)
variables (h_perpendicular_3 : SR ⊥ SP)
variables (h_SP : SP = 12)
variables (h_SQ : SQ = 12)
variables (h_SR : SR = 8)

-- The theorem statement
theorem volume_of_pyramid_SPQR : 
  (∃ (vol : ℝ), vol = 1/3 * (1/2 * SP * SQ) * SR) -> 
  ∃ (vol : ℝ), vol = 192 :=
by
  sorry

end volume_of_pyramid_SPQR_l769_769029


namespace lucy_fish_l769_769378

theorem lucy_fish (current_fish total_fish : ℕ) (h_current : current_fish = 212) (h_total : total_fish = 280) : 
  total_fish - current_fish = 68 :=
by
  rw [h_current, h_total]
  norm_num

end lucy_fish_l769_769378


namespace Maria_telephone_numbers_l769_769568

def num_distinct_telephone_numbers : ℕ :=
  Nat.choose 7 5

theorem Maria_telephone_numbers :
  num_distinct_telephone_numbers = 21 := by
  sorry

end Maria_telephone_numbers_l769_769568


namespace toy_car_production_l769_769551

theorem toy_car_production (yesterday today total : ℕ) 
  (hy : yesterday = 60)
  (ht : today = 2 * yesterday) :
  total = yesterday + today :=
by
  sorry

end toy_car_production_l769_769551


namespace tiling_rectangle_if_5n_covered_then_n_even_tiling_5x2n_at_least_3_exp_n_minus_one_over_two_ways_l769_769547

theorem tiling_rectangle_if_5n_covered_then_n_even (n : ℕ) (can_cover : ∃ tiles : list (tile), cover_5n_with_tiles n tiles) : n % 2 = 0 :=
sorry

theorem tiling_5x2n_at_least_3_exp_n_minus_one_over_two_ways (n : ℕ) (can_tile : ∃ tiles : list (tile), cover_5x2n_with_tiles 2*n tiles) : num_ways_to_tile_5x2n n ≥ 3^(n - 1) / 2 :=
sorry

-- Definitions and additional structures needed for the above theorems
structure tile := (turn_over : bool) (rotate : ℕ)

def cover_5n_with_tiles (n : ℕ) (tiles : list tile) : Prop := sorry
def cover_5x2n_with_tiles (n : ℕ) (tiles : list tile) : Prop := sorry
def num_ways_to_tile_5x2n (n : ℕ) : ℕ := sorry

end tiling_rectangle_if_5n_covered_then_n_even_tiling_5x2n_at_least_3_exp_n_minus_one_over_two_ways_l769_769547


namespace repeating_decimal_addition_l769_769219

noncomputable def repeating_decimal_to_fraction := 
  λ (r : ℚ), (0.\overline{r} : ℚ) 

theorem repeating_decimal_addition :
  repeating_decimal_to_fraction 3 / 9 + repeating_decimal_to_fraction 6 / 99 = 13 / 33 :=
by sorry

end repeating_decimal_addition_l769_769219


namespace quadratic_eq_m_neg1_l769_769843

theorem quadratic_eq_m_neg1 (m : ℝ) (h1 : (m - 3) ≠ 0) (h2 : m^2 - 2*m - 3 = 0) : m = -1 :=
sorry

end quadratic_eq_m_neg1_l769_769843


namespace pilot_speed_return_flight_l769_769530

noncomputable def speed_out := 300 -- mph
noncomputable def distance := 1500 -- miles
noncomputable def total_time := 8  -- hours

theorem pilot_speed_return_flight : 
  let time_out := distance / speed_out in
  let time_return := total_time - time_out in
  let speed_return := distance / time_return in
  speed_return = 500 :=
by
  sorry

end pilot_speed_return_flight_l769_769530


namespace equation_of_circle_l769_769616

theorem equation_of_circle :
  ∀ (x y : ℝ), (∀ (c : ℝ), c = (x + 1) * (x + 1) + y * y → sqrt 3 = c ^ (1 / 2)) →
  (x + 1) * (x + 1) + y * y = 3 :=
by
  intros x y h
  sorry

end equation_of_circle_l769_769616


namespace angle_A_area_of_triangle_l769_769680

open Real

theorem angle_A (a : ℝ) (A B C : ℝ) 
  (h_a : a = 2 * sqrt 3)
  (h_condition1 : 4 * cos A ^ 2 + 4 * cos B * cos C + 1 = 4 * sin B * sin C) :
  A = π / 3 := 
sorry

theorem area_of_triangle (a b c A : ℝ) 
  (h_A : A = π / 3)
  (h_a : a = 2 * sqrt 3)
  (h_b : b = 3 * c) :
  (1 / 2) * b * c * sin A = 9 * sqrt 3 / 7 := 
sorry

end angle_A_area_of_triangle_l769_769680


namespace num_of_n_with_condition_l769_769241
noncomputable def proof_problem : Prop :=
  ∀ (n : ℕ), (1 ≤ n ∧ n ≤ 500) → (∀ (t : ℝ), (cos t - complex.i * sin t) ^ n = (cos (n * t) - complex.i * sin (n * t)))

theorem num_of_n_with_condition : finset.card (finset.filter (λ n, ∀ t,
  (cos t - complex.i * sin t) ^ n = (cos (n * t) - complex.i * sin (n * t))) (finset.range 501)) = 500 :=
sorry

end num_of_n_with_condition_l769_769241


namespace sum_of_cosines_of_dihedral_angles_of_tetrahedron_equals_two_sum_of_cosines_of_dihedral_angles_of_trihedral_equals_one_l769_769508

-- Problem (a): Prove that the sum of the cosines of the dihedral angles of a regular tetrahedron equals 2

theorem sum_of_cosines_of_dihedral_angles_of_tetrahedron_equals_two
  (e1 e2 e3 e4 : EuclideanSpace ℝ (Fin 3))
  (h1 : ∥e1∥ = 1 ∧ ∥e2∥ = 1 ∧ ∥e3∥ = 1 ∧ ∥e4∥ = 1)
  (h2 : e1 + e2 + e3 + e4 = 0) :
  ∑ i : Fin 4, ∑ j in Icc (i + 1) 3, Real.cos (angle e1 e2) = 2 := sorry

-- Problem (b): Prove that the sum of the cosines of the dihedral angles of a trihedral angle is 1

theorem sum_of_cosines_of_dihedral_angles_of_trihedral_equals_one
  (α β γ : ℝ)
  (h : α + β + γ = π) :
  Real.cos (π - α) + Real.cos (π - β) + Real.cos (π - γ) = 1 := sorry

end sum_of_cosines_of_dihedral_angles_of_tetrahedron_equals_two_sum_of_cosines_of_dihedral_angles_of_trihedral_equals_one_l769_769508


namespace range_of_k_l769_769372

noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / x
noncomputable def g (x : ℝ) : ℝ := x / Real.exp x

theorem range_of_k (k : ℝ) (x1 x2 : ℝ) (hx1 : 0 < x1) (hx2 : 0 < x2) :
  (g x1 / k ≤ f x2 / (k + 1)) ↔ k ≥ 1 / (2 * Real.exp 1 - 1) := by
  sorry

end range_of_k_l769_769372


namespace percentage_change_area_l769_769445

theorem percentage_change_area (L B : ℝ) (hL : L > 0) (hB : B > 0) :
    let A := L * B
    let L' := L / 2
    let B' := 3 * B
    let A' := (L / 2) * (3 * B)
    (A' - A) / A * 100 = 50 :=
by
  let A := L * B
  let A' := (L / 2) * (3 * B)
  have h : (A' - A) / A * 100 = (1 / 2) * 100 := sorry
  exact h

end percentage_change_area_l769_769445


namespace probability_x_gt_7y_l769_769943

theorem probability_x_gt_7y (x y : ℝ) (h1 : 0 ≤ x ∧ x ≤ 2009) (h2 : 0 ≤ y ∧ y ≤ 2010) :
  ∃ (p : ℚ), p = 287 / 4020 ∧ 
  let P := (x, y) in P ∈ setOf (λ P : ℝ × ℝ, P.1 > 7 * P.2) →
  p = Probability (P) := by
  sorry

end probability_x_gt_7y_l769_769943


namespace percentage_change_area_l769_769433

theorem percentage_change_area (L B : ℝ) (h : L > 0 ∧ B > 0) :
  let A1 := L * B
  let L2 := L / 2
  let B2 := 3 * B
  let A2 := L2 * B2
  let percentage_change := ((A2 - A1) / A1) * 100
  percentage_change = 50 := by
  have A1_def := calc
    A1 = L * B : rfl
  have A2_def := calc
    A2 = (L / 2) * (3 * B) : rfl
    ... = (3 / 2) * (L * B) : by ring
  have pc_def := calc
    percentage_change = ((A2 - A1) / A1) * 100 : rfl
    ... = (( (3 / 2) * (L * B) - (L * B) ) / (L * B) ) * 100 : by rw [A1_def, A2_def]
    ... = (1 / 2) * 100 : by ring
  exact pc_def

end percentage_change_area_l769_769433


namespace projectile_reaches_35_at_1p57_seconds_l769_769416

theorem projectile_reaches_35_at_1p57_seconds :
  ∀ (t : ℝ), (y : ℝ) (h_eq : y = -4.9 * t^2 + 30 * t)
  (h_initial_velocity : true)  -- Given that the projectile is launched from the ground, we assume this as a given
  (h_conditions : y = 35),
  t = 1.57 :=
by
  sorry

end projectile_reaches_35_at_1p57_seconds_l769_769416


namespace find_slope_angle_l769_769412

theorem find_slope_angle 
  (k : ℝ)
  (h_line : ∀ x y : ℝ, y = k * x + 3)
  (h_circle : ∀ x y : ℝ, (x - 2)^2 + (y - 3)^2 = 4)
  (h_chord : 2 * sqrt 3 = 2 * sqrt (4 - (4 * k^2) / (k^2 + 1))) : 
  ∃ θ : ℝ, θ = π / 6 ∨ θ = 5 * π / 6 := 
  sorry

end find_slope_angle_l769_769412


namespace count_factors_of_180_multiple_of_15_l769_769710

def is_factor (x n : ℕ) := n % x = 0 -- Definition of a factor
def is_multiple (x k : ℕ) := x % k = 0 -- Definition of a multiple

theorem count_factors_of_180_multiple_of_15 :
  (finset.filter (λ x, is_multiple x 15) (finset.filter (λ x, is_factor x 180) (finset.range (180 + 1)))).card = 6 := by
  sorry

end count_factors_of_180_multiple_of_15_l769_769710


namespace count_factors_of_180_multiple_of_15_l769_769781

def is_factor (n k : ℕ) : Prop := n % k = 0

def is_multiple_of (k m : ℕ) : Prop := k % m = 0

theorem count_factors_of_180_multiple_of_15 : 
  (finset.filter (λ f, is_factor 180 f ∧ is_multiple_of f 15) (finset.range (181))).card = 6 := 
by 
sorry

end count_factors_of_180_multiple_of_15_l769_769781


namespace count_factors_of_180_multiple_of_15_l769_769719

def is_factor (x n : ℕ) := n % x = 0 -- Definition of a factor
def is_multiple (x k : ℕ) := x % k = 0 -- Definition of a multiple

theorem count_factors_of_180_multiple_of_15 :
  (finset.filter (λ x, is_multiple x 15) (finset.filter (λ x, is_factor x 180) (finset.range (180 + 1)))).card = 6 := by
  sorry

end count_factors_of_180_multiple_of_15_l769_769719


namespace adapted_bowling_ball_volume_l769_769139

noncomputable def volume_adapted_bowling_ball : ℝ :=
  let volume_sphere := (4/3) * Real.pi * (20 ^ 3)
  let volume_hole1 := Real.pi * (1 ^ 2) * 10
  let volume_hole2 := Real.pi * (1.5 ^ 2) * 10
  let volume_hole3 := Real.pi * (2 ^ 2) * 10
  volume_sphere - (volume_hole1 + volume_hole2 + volume_hole3)

theorem adapted_bowling_ball_volume :
  volume_adapted_bowling_ball = 10594.17 * Real.pi :=
sorry

end adapted_bowling_ball_volume_l769_769139


namespace tan_overlap_right_shift_l769_769687

theorem tan_overlap_right_shift (ω : ℝ) (hω : 2 < ω ∧ ω < 10) :
  (∀ x, tan (ω * x - (ω * π / 6) + (π / 3)) = tan (ω * x + (π / 3))) → ω = 6 := by
  sorry

end tan_overlap_right_shift_l769_769687


namespace correct_operation_l769_769489

variable {a b : ℝ}

def conditionA : Prop := a^2 + a^3 = a^6
def conditionB : Prop := (a * b)^2 = a * b^2
def conditionC : Prop := (a + b)^2 = a^2 + b^2
def conditionD : Prop := (a + b) * (a - b) = a^2 - b^2

theorem correct_operation : conditionD ∧ ¬conditionA ∧ ¬conditionB ∧ ¬conditionC := by
  sorry

end correct_operation_l769_769489


namespace day_365_is_Tuesday_l769_769839

def day_of_week : Type := ℕ

def Day.January_15_2005_is_Tuesday (day_of_week_n : day_of_week) : Prop :=
  day_of_week_n ≡ 2 [MOD 7]

def day_365_is_same_day_of_week (day_of_week_n day_after_n_days : day_of_week) (days_between : ℕ) : Prop :=
  (day_of_week_n + days_between) % 7 = day_after_n_days % 7

theorem day_365_is_Tuesday (day_15 : day_of_week) :
  (Day.January_15_2005_is_Tuesday day_15) →
  day_365_is_same_day_of_week day_15 day_15 350 →
  (day_15 % 7 = 2 % 7) :=
by
  intros h1 h2
  sorry

end day_365_is_Tuesday_l769_769839


namespace total_gulbis_is_correct_l769_769459

-- Definitions based on given conditions
def num_dureums : ℕ := 156
def num_gulbis_in_one_dureum : ℕ := 20

-- Definition of total gulbis calculated
def total_gulbis : ℕ := num_dureums * num_gulbis_in_one_dureum

-- Statement to prove
theorem total_gulbis_is_correct : total_gulbis = 3120 := by
  -- The actual proof would go here
  sorry

end total_gulbis_is_correct_l769_769459


namespace cos_squared_half_angle_sin_squared_half_angle_l769_769945

variable (a b c p : ℝ)
variable (α : ℝ)
variable (triangle : Prop)

def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

axiom angle_cosine_rule :
  ∀ (a b c α : ℝ), 
    triangle → 
    α = 2 * Math.Sin(α / 2) * Math.Cos(α / 2) → Math.Cos(α) = (b^2 + c^2 - a^2) / (2 * b * c)

theorem cos_squared_half_angle :
  ∀ (a b c p α : ℝ), 
    triangle → 
    p = semiperimeter a b c →
    α = 2 * Math.Sin(α / 2) * Math.Cos(α / 2) →
    Math.Cos(α / 2) * Math.Cos(α / 2) = p * (p - a) / (b * c)
:= by
  intros
  sorry

theorem sin_squared_half_angle :
  ∀ (a b c p α : ℝ), 
    triangle → 
    p = semiperimeter a b c →
    α = 2 * Math.Sin(α / 2) * Math.Cos(α / 2) →
    Math.Sin(α / 2) * Math.Sin(α / 2) = (p - b) * (p - c) / (b * c)
:= by
  intros
  sorry

end cos_squared_half_angle_sin_squared_half_angle_l769_769945


namespace maximum_distance_between_balls_l769_769332

theorem maximum_distance_between_balls 
  (a b c : ℝ) 
  (aluminum_ball_heavier : true) -- Implicitly understood property rather than used in calculation directly
  (wood_ball_lighter : true) -- Implicitly understood property rather than used in calculation directly
  : ∃ d : ℝ, d = Real.sqrt (a^2 + b^2 + c^2) → d = Real.sqrt (3^2 + 4^2 + 2^2) := 
by
  use Real.sqrt (3^2 + 4^2 + 2^2)
  sorry

end maximum_distance_between_balls_l769_769332


namespace smallest_a_value_l769_769905

noncomputable def P (x : ℤ) : ℤ := sorry  -- Definition of the polynomial P(x)

theorem smallest_a_value :
  ∃ (a : ℤ), a > 0 ∧ P 1 = a ∧ P 4 = a ∧ P 7 = a ∧
  P 3 = -a ∧ P 5 = -a ∧ P 6 = -a ∧ P 8 = -a ∧
  (∀ b > 0, (P 1 = b ∧ P 4 = b ∧ P 7 = b ∧
             P 3 = -b ∧ P 5 = -b ∧ P 6 = -b ∧ P 8 = -b →
             b ≥ a)) :=
begin
  use 84,
  -- This part contains the proof steps which are skipped with sorry.
  -- The steps would involve verifying the conditions specified in the problem.
  repeat { sorry },
end

end smallest_a_value_l769_769905


namespace time_to_cross_bridge_l769_769172

-- Conditions
def length_of_train : ℝ := 150 -- in meters
def speed_of_train : ℝ := 45 * 1000 / 3600 -- converted to m/s
def length_of_bridge : ℝ := 225 -- in meters

-- Problem Statement
theorem time_to_cross_bridge : (length_of_train + length_of_bridge) / speed_of_train = 30 := by
  sorry

end time_to_cross_bridge_l769_769172


namespace conference_center_people_count_l769_769152

-- Definition of the conditions
def rooms : ℕ := 6
def capacity_per_room : ℕ := 80
def fraction_full : ℚ := 2/3

-- Total capacity of the conference center
def total_capacity := rooms * capacity_per_room

-- Number of people in the conference center when 2/3 full
def num_people := fraction_full * total_capacity

-- The theorem stating the problem
theorem conference_center_people_count :
  num_people = 320 := 
by
  -- This is a placeholder for the proof
  sorry

end conference_center_people_count_l769_769152


namespace correct_operation_l769_769498

theorem correct_operation (a b : ℝ) :
  ¬ (a^2 + a^3 = a^6) ∧
  ¬ ((a*b)^2 = a*(b^2)) ∧
  ¬ ((a + b)^2 = a^2 + b^2) ∧
  ((a + b)*(a - b) = a^2 - b^2) := 
by
  sorry

end correct_operation_l769_769498


namespace arrangements_not_together_l769_769994

-- Define the 6 people standing in a row
inductive Person
| A | B | C | D | E | F
deriving DecidableEq

-- Define the condition that there are 6 people standing in a row
def people_standing_in_a_row : Finset (List Person) :=
  { list.permutations [Person.A, Person.B, Person.C, Person.D, Person.E, Person.F] }

-- Define the concept of A, B, and C standing together
def are_together (l : List Person) : Bool :=
  let idxA := l.indexOf Person.A
  let idxB := l.indexOf Person.B
  let idxC := l.indexOf Person.C
  (idxA.succ == idxB ∧ idxB.succ == idxC) ∨ (idxB.succ == idxA ∧ idxA.succ == idxC) ∨ (idxA.succ == idxC ∧ idxC.succ == idxB)

-- Define the number of arrangements where A, B, and C are together
def together_arrangements : Finset (List Person) :=
  people_standing_in_a_row.filter are_together

-- Define the number of arrangements where A, B, and C are not together
def not_together_arrangements : ℕ :=
  people_standing_in_a_row.card - together_arrangements.card

-- Statement we need to prove
theorem arrangements_not_together : not_together_arrangements = 576 :=
by
  sorry

end arrangements_not_together_l769_769994


namespace day_365_in_2005_is_tuesday_l769_769835

theorem day_365_in_2005_is_tuesday
  (day15_is_tuesday : (15 % 7 = 1) ∧ (365 % 7 = 1)) 
  : true := 
by
  have day_of_week_of_15 := "Tuesday"
  have day_of_week_of_365 := "Tuesday"
  exact trivial

end day_365_in_2005_is_tuesday_l769_769835


namespace count_factors_180_multiple_of_15_is_6_l769_769817

-- We define a function to check if 'n' is a factor of 'm'
def is_factor (m n : ℕ) : Prop := n ∣ m

-- We define a function to check if 'n' is a multiple of 'd'
def is_multiple (n d : ℕ) : Prop := d ∣ n

-- We define a function to collect all factors of a number 'n'
def factors (n : ℕ) : list ℕ := 
  -- We use list.filter to keep elements that satisfy is_factor n
  list.filter (λ x, is_factor n x) (list.range (n + 1))

-- We count the number of elements in a list that satisfy a predicate
def count_satisfying {α : Type*} (pred : α → Prop) [decidable_pred pred] (l : list α) : ℕ :=
  list.length (list.filter pred l)

-- Noncomputable definition to specify that counting number
noncomputable def count_positive_factors_of_180_multiple_of_15 : ℕ :=
  count_satisfying (λ x, is_multiple x 15) (factors 180)

-- We state the theorem equivalence to the proof problem
theorem count_factors_180_multiple_of_15_is_6 : count_positive_factors_of_180_multiple_of_15 = 6 :=
  sorry


end count_factors_180_multiple_of_15_is_6_l769_769817


namespace cyclic_and_area_ratio_of_perpendicular_bisector_quadrilateral_l769_769534

theorem cyclic_and_area_ratio_of_perpendicular_bisector_quadrilateral
  (A B C D E M N P Q : Point)
  (O O' : Circle)
  (hO : is_cyclic_quadrilateral A B C D O)
  (hD1: is_perpendicular A C B D)
  (hD2: diagonals_intersect_at A C B D E)
  (hM : is_perpendicular E M A B)
  (hN : is_perpendicular E N B C)
  (hP : is_perpendicular E P C D)
  (hQ : is_perpendicular E Q D A)
  (hO' : can_be_inscribed M N Q P O')
  (area_of_ABCD : ℝ)
  (area_of_MNQP : ℝ)
  (radius_O : ℝ)
  (radius_O' : ℝ) :
  is_cyclic_quadrilateral M N Q P O' ∧
  (area_of_MNQP / area_of_ABCD = radius_O' / radius_O) :=
sorry

end cyclic_and_area_ratio_of_perpendicular_bisector_quadrilateral_l769_769534


namespace count_positive_factors_of_180_multiple_of_15_l769_769726

statement: ℕ
statement := sorry

theorem count_positive_factors_of_180_multiple_of_15 :
  let factors_of_180 := {d ∈ (finset.range 181).filter (λ x, 180 % x = 0)}
  let multiples_of_15 := (finset.range 181).filter (λ x, x % 15 = 0)
  let desired_factors := factors_of_180 ∩ multiples_of_15
  desired_factors.card = 6 :=
by
  sorry

end count_positive_factors_of_180_multiple_of_15_l769_769726


namespace count_factors_of_180_multiple_of_15_l769_769778

def is_factor (n k : ℕ) : Prop := n % k = 0

def is_multiple_of (k m : ℕ) : Prop := k % m = 0

theorem count_factors_of_180_multiple_of_15 : 
  (finset.filter (λ f, is_factor 180 f ∧ is_multiple_of f 15) (finset.range (181))).card = 6 := 
by 
sorry

end count_factors_of_180_multiple_of_15_l769_769778


namespace union_M_N_is_real_l769_769317

open Set

noncomputable theory

-- Define sets M and N based on given conditions
def M : Set ℝ := {y | ∃ x : ℝ, y = (2:ℝ)^x }
def N : Set ℝ := {y | ∃ x : ℝ, y = Real.logBase 0.5 (Real.sqrt (x^2 + 1)) }

-- Define the union of sets M and N
def union_M_N := M ∪ N

-- Theorem statement: The union of M and N is the set of all real numbers
theorem union_M_N_is_real : union_M_N = univ :=
by sorry

end union_M_N_is_real_l769_769317


namespace find_b_l769_769684

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ := -x^3 + a * x^2 + b * x + c

theorem find_b (a b c : ℝ) :
  (∀ x : ℝ, x < 0 → (f x a b c < f 0 a b c)) →
  (∀ x : ℝ, 0 < x ∧ x < 1 → (f 0 a b c < f x a b c)) →
  b = 0 :=
by
  intro h1 h2,
  have : (f 0 a b c = c) := by simp [f],
  sorry

end find_b_l769_769684


namespace graph_passes_through_fixed_point_l769_769418

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a^(x-1) + 3

-- The theorem to prove the graph of f always passes through the point (1, 4)
theorem graph_passes_through_fixed_point (a : ℝ) : f a 1 = 4 :=
by
  -- Proof: To be filled in
  sorry

end graph_passes_through_fixed_point_l769_769418


namespace seq_a_2014_l769_769880

theorem seq_a_2014 {a : ℕ → ℕ}
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, 0 < n → n * a (n + 1) = (n + 1) * a n) :
  a 2014 = 2014 :=
sorry

end seq_a_2014_l769_769880


namespace meaningful_fraction_l769_769844

theorem meaningful_fraction (x : ℝ) : (1 / (x + 1) ∈ ℚ) ↔ x ≠ -1 :=
by
  sorry

end meaningful_fraction_l769_769844


namespace count_N_lt_500_solution_exists_l769_769287

theorem count_N_lt_500_solution_exists:
  ∃ (N : ℕ), N < 500 ∧ (∃ (x : ℝ), x^floor x = N) = 287 :=
sorry

end count_N_lt_500_solution_exists_l769_769287


namespace day_crew_fraction_l769_769574

-- Definitions of number of boxes per worker for day crew, and workers for day crew
variables (D : ℕ) (W : ℕ)

-- Definitions of night crew loading rate and worker ratio based on given conditions
def night_boxes_per_worker := (3 / 4 : ℚ) * D
def night_workers := (2 / 3 : ℚ) * W

-- Definition of total boxes loaded by each crew
def day_crew_total := D * W
def night_crew_total := night_boxes_per_worker D * night_workers W

-- The proof problem shows fraction loaded by day crew equals 2/3
theorem day_crew_fraction : (day_crew_total D W) / (day_crew_total D W + night_crew_total D W) = (2 / 3 : ℚ) := by
  sorry

end day_crew_fraction_l769_769574


namespace total_skips_l769_769601

-- Definitions based on conditions
def fifth_throw := 8
def fourth_throw := fifth_throw - 1
def third_throw := fourth_throw + 3
def second_throw := third_throw / 2
def first_throw := second_throw - 2

-- Statement of the proof problem
theorem total_skips : first_throw + second_throw + third_throw + fourth_throw + fifth_throw = 33 := by
  sorry

end total_skips_l769_769601


namespace sum_of_squares_CE_l769_769667

theorem sum_of_squares_CE : 
  let (ABC : Triangle) := equilateral_triangle (sqrt 144) in
  let O := circumcenter ABC in
  let D1 := point_on_circumcircle O ABC 36 45 in
  let D2 := point_on_circumcircle O ABC 36 45 in
  let E1 := point_on_line_segments AD1 in
  let E2 := point_on_line_segments AD1 in
  let E3 := point_on_line_segments AD2 in
  let E4 := point_on_line_segments AD2 in
  (CE1 ^ 2 + CE2 ^ 2 + CE3 ^ 2 + CE4 ^ 2) = 576 :=
by
  sorry

end sum_of_squares_CE_l769_769667


namespace integral_sqrt2_sin_l769_769608

theorem integral_sqrt2_sin (I : ℝ) :
  (I = ∫ x in 0..(Real.pi / 2), (sqrt 2) * Real.sin (x + Real.pi / 4)) → I = 2 :=
by
  intro h
  rw [h]
  sorry

end integral_sqrt2_sin_l769_769608


namespace overlapping_area_arrows_in_grid_l769_769987

-- Define the grid dimensions (4 cm x 4 cm)
def grid_dim : ℕ := 4

-- Define the condition: The overlapping area of two specific arrows in a same grid
-- We assume that the arrows are already given, hence defined by grid cells overlapping.

-- Proof problem: Prove the overlapping area of the two arrows.
theorem overlapping_area_arrows_in_grid (g : fin grid_dim × fin grid_dim → bool) (arrow1 arrow2 : fin grid_dim × fin grid_dim → bool) :
  (∀ x, (arrow1 x) ∧ (arrow2 x)) = 6 :=
sorry

end overlapping_area_arrows_in_grid_l769_769987


namespace scientific_notation_6500_l769_769048

theorem scientific_notation_6500 : (6500 : ℝ) = 6.5 * 10^3 := 
by 
  sorry

end scientific_notation_6500_l769_769048


namespace num_factors_of_180_multiple_of_15_l769_769738

def is_factor (n m : Nat) : Prop := m % n = 0
def is_multiple_of_15 (n : Nat) : Prop := is_factor 15 n

theorem num_factors_of_180_multiple_of_15 : 
  (Nat.filter (fun n => is_factor n 180 ∧ is_multiple_of_15 n) (Nat.range (180 + 1))).length = 6 :=
sorry

end num_factors_of_180_multiple_of_15_l769_769738


namespace triangle_perimeter_l769_769250

variable {A B C : Type} [EuclideanGeometry.Triangle A B C]
variable (area_ABC : Real := (3 * Real.sqrt 15) / 4)
variable (cos_B : Real := -1 / 4)
variable (AC : Real := 4)

theorem triangle_perimeter 
  (h_area : EuclideanGeometry.area A B C = area_ABC)
  (h_cos : EuclideanGeometry.cos B = cos_B)
  (h_AC : EuclideanGeometry.length A C = AC) : 
  EuclideanGeometry.perimeter A B C = 9 :=
sorry

end triangle_perimeter_l769_769250


namespace overlapping_area_of_triangles_is_1_l769_769977

def point := (ℝ × ℝ)

def vertices_triangle1 : set point := {(0, 0), (2, 1), (1, 2)}
def vertices_triangle2 : set point := {(2, 2), (0, 1), (1, 0)}

def grid_points : set point := {(x, y) | (x = 0 ∨ x = 1 ∨ x = 2) ∧ (y = 0 ∨ y = 1 ∨ y = 2)}

def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

lemma distance_between_adjacent_points :
  ∀ p1 p2 ∈ grid_points, (dist p1 p2 = 1) → (abs (p1.1 - p2.1) + abs (p1.2 - p2.2) = 1) :=
sorry -- Skip proof

theorem overlapping_area_of_triangles_is_1 :
  let hexagon_area := 1 in
  hexagon_area = 1 :=
sorry -- Skip proof

end overlapping_area_of_triangles_is_1_l769_769977


namespace num_factors_of_180_multiple_of_15_l769_769737

def is_factor (n m : Nat) : Prop := m % n = 0
def is_multiple_of_15 (n : Nat) : Prop := is_factor 15 n

theorem num_factors_of_180_multiple_of_15 : 
  (Nat.filter (fun n => is_factor n 180 ∧ is_multiple_of_15 n) (Nat.range (180 + 1))).length = 6 :=
sorry

end num_factors_of_180_multiple_of_15_l769_769737


namespace min_reciprocal_sum_l769_769908

theorem min_reciprocal_sum (a : Fin 15 → ℝ) (h_pos : ∀ i, 0 < a i) (h_sum : (∑ i, a i) = 1) : 
  (∑ i, (1 / a i)) = 225 :=
sorry

end min_reciprocal_sum_l769_769908


namespace transform_grades_leq_4_l769_769097

variables (a b c d e : ℕ) (n : ℕ)
hypothesis h1 : n = a + b + c + d + e
hypothesis h2 : (a + 2 * b + 3 * c + 4 * d + 5 * e) / n ≤ 3

theorem transform_grades_leq_4 : (3 * a + 2 * b + 3 * c + 4 * d + 5 * e) / n ≤ 4 :=
by
  sorry

end transform_grades_leq_4_l769_769097


namespace day_365_in_2005_is_tuesday_l769_769834

theorem day_365_in_2005_is_tuesday
  (day15_is_tuesday : (15 % 7 = 1) ∧ (365 % 7 = 1)) 
  : true := 
by
  have day_of_week_of_15 := "Tuesday"
  have day_of_week_of_365 := "Tuesday"
  exact trivial

end day_365_in_2005_is_tuesday_l769_769834


namespace minimum_b_l769_769949

theorem minimum_b (k a b : ℝ) (h1 : 1 < k) (h2 : k < a) (h3 : a < b)
  (h4 : ¬(k + a > b)) (h5 : ¬(1/a + 1/b > 1/k)) :
  2 * k ≤ b :=
by
  sorry

end minimum_b_l769_769949


namespace petya_can_force_difference_2014_l769_769016

theorem petya_can_force_difference_2014 :
  ∀ (p q r : ℚ), ∃ (a b c : ℚ), ∀ (x : ℝ), (x^3 + a * x^2 + b * x + c = 0) → 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 - x2 = 2014 :=
by sorry

end petya_can_force_difference_2014_l769_769016


namespace cos_pi_plus_2alpha_l769_769636

-- Define the main theorem using the given condition and the result to be proven
theorem cos_pi_plus_2alpha (α : ℝ) (h : Real.sin (π / 2 - α) = 1 / 3) : Real.cos (π + 2 * α) = 7 / 9 :=
sorry

end cos_pi_plus_2alpha_l769_769636


namespace at_least_one_corner_square_selected_l769_769096

theorem at_least_one_corner_square_selected :
  let total_squares := 16
  let total_corners := 4
  let total_non_corners := 12
  let ways_to_select_3_from_total := Nat.choose total_squares 3
  let ways_to_select_3_from_non_corners := Nat.choose total_non_corners 3
  let probability_no_corners := (ways_to_select_3_from_non_corners : ℚ) / ways_to_select_3_from_total
  let probability_at_least_one_corner := 1 - probability_no_corners
  probability_at_least_one_corner = (17 / 28 : ℚ) :=
by
  sorry

end at_least_one_corner_square_selected_l769_769096


namespace equilateral_triangle_iff_l769_769398

theorem equilateral_triangle_iff (a b c : ℝ) :
  a^2 + b^2 + c^2 = a*b + b*c + c*a ↔ a = b ∧ b = c :=
sorry

end equilateral_triangle_iff_l769_769398


namespace quadratic_inequality_solution_l769_769075

theorem quadratic_inequality_solution (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 2 * x + a ≥ 0) → a ≥ 1 :=
by
  sorry

end quadratic_inequality_solution_l769_769075


namespace distinct_book_arrangements_l769_769824

def num_books := 7
def num_identical_books := 3
def num_unique_books := num_books - num_identical_books

theorem distinct_book_arrangements :
  (Nat.factorial num_books) / (Nat.factorial num_identical_books) = 840 := 
  by 
  sorry

end distinct_book_arrangements_l769_769824


namespace verify_solution_l769_769230

noncomputable def particular_solution : ℝ → ℝ := λ x, (x^5)/3 + x^2

theorem verify_solution :
  ∃ f : ℝ → ℝ,
  (∀ x, deriv f x = (2 / x) * f x + x^4) ∧
  f 1 = 4 / 3 ∧
  ∀ x, f x = particular_solution x :=
by
  sorry

end verify_solution_l769_769230


namespace positive_factors_of_180_multiple_of_15_count_l769_769757

theorem positive_factors_of_180_multiple_of_15_count : 
  let factorization_180 := 2^2 * 3^2 * 5;
  let is_multiple_of_15 (n : ℕ) := ∃ k1 k2 : ℕ, k1 * 3 = n ∧ k2 * 5 = n;
  let factors_180 := { n : ℕ | n ∣ factorization_180 };
  { n : ℕ | n ∈ factors_180 ∧ is_multiple_of_15 n }.card = 6 :=
by
  sorry

end positive_factors_of_180_multiple_of_15_count_l769_769757


namespace distinct_arrangements_l769_769087

-- Defining the conditions as constants
def num_women : ℕ := 9
def num_men : ℕ := 3
def total_slots : ℕ := num_women + num_men

-- Using the combination formula directly as part of the statement
theorem distinct_arrangements : Nat.choose total_slots num_men = 220 := by
  sorry

end distinct_arrangements_l769_769087


namespace least_positive_integer_divisible_by_first_four_primes_l769_769106

theorem least_positive_integer_divisible_by_first_four_primes :
  ∃ n : ℕ, n > 0 ∧ (∀ p ∈ {2, 3, 5, 7}, p ∣ n) ∧ n = 210 :=
by
  sorry

end least_positive_integer_divisible_by_first_four_primes_l769_769106


namespace factors_of_180_multiple_of_15_l769_769751

/-- A factor of a number n is a positive integer which divides n without leaving a remainder. -/
def is_factor (n k : ℕ) : Prop :=
k > 0 ∧ n % k = 0

/-- A multiple of a number m is an integer which is divisible by m without leaving a remainder. -/
def is_multiple (k m : ℕ) : Prop :=
k % m = 0

/-- There are 6 factors of 180 that are also multiples of 15. -/
theorem factors_of_180_multiple_of_15 : 
  (Finset.filter (λ k, is_multiple k 15) { x | is_factor 180 x }.to_finset).card = 6 :=
by 
  sorry

end factors_of_180_multiple_of_15_l769_769751


namespace percentage_change_area_l769_769419

theorem percentage_change_area (L B : ℝ) :
  let A_original := L * B
  let A_new := (L / 2) * (3 * B)
  (A_new - A_original) / A_original * 100 = 50 := by
  sorry

end percentage_change_area_l769_769419


namespace correct_statements_l769_769039

def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x + Real.pi / 3)

theorem correct_statements : 
  (∀ x : ℝ, f x = 4 * Real.cos (2 * x - Real.pi / 6)) ∧ 
  (f (-Real.pi / 6) = 0) :=
by
  sorry

end correct_statements_l769_769039


namespace sin_A_is_eight_ninths_l769_769842

variable (AB AC : ℝ) (A : ℝ)

-- Given conditions
def area_triangle := 1 / 2 * AB * AC * Real.sin A = 100
def geometric_mean := Real.sqrt (AB * AC) = 15

-- Proof statement
theorem sin_A_is_eight_ninths (h1 : area_triangle AB AC A) (h2 : geometric_mean AB AC) :
  Real.sin A = 8 / 9 := sorry

end sin_A_is_eight_ninths_l769_769842


namespace total_skips_l769_769600

-- Definitions based on conditions
def fifth_throw := 8
def fourth_throw := fifth_throw - 1
def third_throw := fourth_throw + 3
def second_throw := third_throw / 2
def first_throw := second_throw - 2

-- Statement of the proof problem
theorem total_skips : first_throw + second_throw + third_throw + fourth_throw + fifth_throw = 33 := by
  sorry

end total_skips_l769_769600


namespace no_prime_divisible_by_91_l769_769823

theorem no_prime_divisible_by_91 : ¬ ∃ p : ℕ, p > 1 ∧ Prime p ∧ 91 ∣ p :=
by
  sorry

end no_prime_divisible_by_91_l769_769823


namespace ellipse_equation_exists_line_pq_fixed_point_l769_769252

noncomputable def c := sqrt 3 / 2
noncomputable def a := 2
noncomputable def b := 1
noncomputable def B := (0, 1)
noncomputable def fixed_point := (0, -3 / 5)

def ellipse := ∀ x y : ℝ, (x^2 / 4) + y^2 = 1

theorem ellipse_equation_exists :
  (e = sqrt 3 / 2) ∧ (b = 1) ∧ (c = e * a) ∧ (a^2 = 4) →
  ellipse :=
by 
  sorry

theorem line_pq_fixed_point (P Q : ℝ × ℝ) :
  ellipse P.1 P.2 ∧ ellipse Q.1 Q.2 ∧ (P ≠ B) ∧ (Q ≠ B) ∧ (BP ⊥ BQ) → 
  (line PQ passes through fixed_point) :=
by 
  sorry

end ellipse_equation_exists_line_pq_fixed_point_l769_769252


namespace smallest_pos_four_digit_div_by_11_has_3odd1even_l769_769108

def is_four_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def has_three_odd_one_even_digits (n : ℕ) : Prop :=
  let digits := List.ofDigits 10 ((Nat.digits 10 n).toList) in
  List.filter Nat.odd digits |>.length = 3 ∧
  List.filter (λ x => ¬ Nat.odd x) digits |>.length = 1

def divisible_by_11 (n : ℕ) : Prop :=
  let digits := List.ofDigits 10 ((Nat.digits 10 n).toList) in
  let odd_positions := List.filteri (λ i x => i % 2 = 0) digits in
  let even_positions := List.filteri (λ i x => i % 2 = 1) digits in
  (List.sum odd_positions - List.sum even_positions) % 11 = 0

theorem smallest_pos_four_digit_div_by_11_has_3odd1even : ∃ n : ℕ,
  is_four_digit_number n ∧ has_three_odd_one_even_digits n ∧ divisible_by_11 n ∧
  (∀ m : ℕ, is_four_digit_number m ∧ has_three_odd_one_even_digits m ∧ divisible_by_11 m → n ≤ m) :=
sorry

end smallest_pos_four_digit_div_by_11_has_3odd1even_l769_769108


namespace max_sin_expr_l769_769642

theorem max_sin_expr (α β : ℝ) (hα : 0 ≤ α ∧ α ≤ π / 4) (hβ : 0 ≤ β ∧ β ≤ π / 4) :
  ∃ (x : ℝ), x = \sqrt{5} ∧ (∀ a b, (0 ≤ a ∧ a ≤ π / 4) → (0 ≤ b ∧ b ≤ π / 4) → \sin (a - b) + 2 * \sin (a + b) ≤ x) :=
sorry

end max_sin_expr_l769_769642


namespace lambda_range_l769_769279

theorem lambda_range (M N: set ℂ) (λ: ℝ):
  (M = {z | ∃ α: ℝ, z = complex.mk (real.cos α) (4 - (real.cos α)^2)}) →
  (N = {z | ∃ β: ℝ, z = complex.mk (real.cos β) (λ + real.sin β)}) →
  (M ∩ N ≠ ∅) →
  λ ∈ set.Icc (11/4: ℝ) 5 :=
by
  intros hM hN hMN
  sorry

end lambda_range_l769_769279


namespace ratio_third_number_l769_769452

theorem ratio_third_number (x : ℚ) (h : 215 / 474 = x / 26) : x ≈ 11.79 :=
by
  -- We need to show that x is approximately 11.79. 
  -- The steps to prove this involve solving the equation given by the proportion.
  sorry

end ratio_third_number_l769_769452


namespace count_valid_N_l769_769310

-- Definitions based on identified mathematical conditions
def valid_N (N : ℕ) : Prop :=
  (∃ x : ℚ, 0 ≤ floor x ∧ floor x < 5 ∧ x ^ (floor x).natAbs = N) ∧ N < 500

theorem count_valid_N : finset.card (finset.filter valid_N (finset.range 500)) = 287 :=
by sorry

end count_valid_N_l769_769310


namespace book_read_stats_l769_769012

def book_readings : List ℕ := [6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 12, 12, 12, 12, 12, 12, 12]

def mode (lst : List ℕ) : ℕ :=
(lst.groupBy id).map (λ l, (l.head!, l.length)).maxBy (λ p, p.snd)).fst

def median (lst : List ℕ) : ℕ :=
let sorted := lst.sort
in if h : lst.length % 2 = 0 then
  let mid_hi := sorted.nth_le (lst.length / 2) (by linarith)
  let mid_lo := sorted.nth_le (lst.length / 2 - 1) (by linarith)
  (mid_hi + mid_lo) / 2
else
  sorted.nth_le (lst.length / 2) (by linarith)

theorem book_read_stats : median book_readings = 9 ∧ mode book_readings = 9 :=
by
  sorry

end book_read_stats_l769_769012


namespace find_b_l769_769056

-- Define the number 1234567 in base 36
def numBase36 : ℤ := 1 * 36^6 + 2 * 36^5 + 3 * 36^4 + 4 * 36^3 + 5 * 36^2 + 6 * 36^1 + 7 * 36^0

-- Prove that for b being an integer such that 0 ≤ b ≤ 10,
-- and given (numBase36 - b) is a multiple of 17, b must be 0
theorem find_b (b : ℤ) (h1 : 0 ≤ b) (h2 : b ≤ 10) (h3 : (numBase36 - b) % 17 = 0) : b = 0 :=
by
  sorry

end find_b_l769_769056


namespace winner_won_by_l769_769869

theorem winner_won_by (V : ℝ) (h₁ : 0.62 * V = 806) : 806 - 0.38 * V = 312 :=
by
  sorry

end winner_won_by_l769_769869


namespace part_a_impossibility_l769_769660

-- Define the number of rows and columns
def num_rows : ℕ := 20
def num_columns : ℕ := 15

-- Define a function that checks if the sum of the counts in rows and columns match the conditions
def is_possible_configuration : Prop :=
  (num_rows % 2 = 0) ∧ (num_columns % 2 = 1)

theorem part_a_impossibility : ¬ is_possible_configuration :=
by
  -- The proof for the contradiction will go here
  sorry

end part_a_impossibility_l769_769660


namespace kids_joined_in_l769_769091

-- Define the given conditions
def original : ℕ := 14
def current : ℕ := 36

-- State the goal
theorem kids_joined_in : (current - original = 22) :=
by
  sorry

end kids_joined_in_l769_769091


namespace Corey_goal_reachable_l769_769204

theorem Corey_goal_reachable :
  ∀ (goal balls_found_saturday balls_found_sunday additional_balls : ℕ),
    goal = 48 →
    balls_found_saturday = 16 →
    balls_found_sunday = 18 →
    additional_balls = goal - (balls_found_saturday + balls_found_sunday) →
    additional_balls = 14 :=
by
  intros goal balls_found_saturday balls_found_sunday additional_balls
  intro goal_eq
  intro saturday_eq
  intro sunday_eq
  intro additional_eq
  sorry

end Corey_goal_reachable_l769_769204


namespace molecular_weight_of_barium_iodide_l769_769482

-- Define the atomic weights
def atomic_weight_of_ba : ℝ := 137.33
def atomic_weight_of_i : ℝ := 126.90

-- Define the molecular weight calculation for Barium iodide
def molecular_weight_of_bai2 : ℝ := atomic_weight_of_ba + 2 * atomic_weight_of_i

-- The main theorem to prove
theorem molecular_weight_of_barium_iodide : molecular_weight_of_bai2 = 391.13 := by
  -- we are given that atomic_weight_of_ba = 137.33 and atomic_weight_of_i = 126.90
  -- hence, molecular_weight_of_bai2 = 137.33 + 2 * 126.90
  -- simplifying this, we get
  -- molecular_weight_of_bai2 = 137.33 + 253.80 = 391.13
  sorry

end molecular_weight_of_barium_iodide_l769_769482


namespace inequality_solution_l769_769113

theorem inequality_solution (x : ℚ) : (3 * x - 5 ≥ 9 - 2 * x) → (x ≥ 14 / 5) :=
by
  sorry

end inequality_solution_l769_769113


namespace virginia_initial_eggs_l769_769104

theorem virginia_initial_eggs (final_eggs : ℕ) (taken_eggs : ℕ) (H : final_eggs = 93) (G : taken_eggs = 3) : final_eggs + taken_eggs = 96 := 
by
  -- proof part could go here
  sorry

end virginia_initial_eggs_l769_769104


namespace scientific_notation_3080000_l769_769985

theorem scientific_notation_3080000 : (∃ (a : ℝ) (b : ℤ), 1 ≤ a ∧ a < 10 ∧ (3080000 : ℝ) = a * 10^b) ∧ (3080000 : ℝ) = 3.08 * 10^6 :=
by
  sorry

end scientific_notation_3080000_l769_769985


namespace count_valid_N_l769_769309

-- Definitions based on identified mathematical conditions
def valid_N (N : ℕ) : Prop :=
  (∃ x : ℚ, 0 ≤ floor x ∧ floor x < 5 ∧ x ^ (floor x).natAbs = N) ∧ N < 500

theorem count_valid_N : finset.card (finset.filter valid_N (finset.range 500)) = 287 :=
by sorry

end count_valid_N_l769_769309


namespace find_e_l769_769921

theorem find_e 
  (a b c d e : ℕ) 
  (h1 : a = 16)
  (h2 : b = 2)
  (h3 : c = 3)
  (h4 : d = 12)
  (h5 : 32 / e = 288 / e) 
  : e = 9 := 
by
  sorry

end find_e_l769_769921


namespace count_factors_of_180_multiple_of_15_l769_769782

def is_factor (n k : ℕ) : Prop := n % k = 0

def is_multiple_of (k m : ℕ) : Prop := k % m = 0

theorem count_factors_of_180_multiple_of_15 : 
  (finset.filter (λ f, is_factor 180 f ∧ is_multiple_of f 15) (finset.range (181))).card = 6 := 
by 
sorry

end count_factors_of_180_multiple_of_15_l769_769782


namespace spheres_in_base_l769_769934

theorem spheres_in_base (n : ℕ) (T_n : ℕ) (total_spheres : ℕ) :
  (total_spheres = 165) →
  (total_spheres = (1 / 6 : ℚ) * ↑n * ↑(n + 1) * ↑(n + 2)) →
  (T_n = n * (n + 1) / 2) →
  n = 9 →
  T_n = 45 :=
by
  intros _ _ _ _
  sorry

end spheres_in_base_l769_769934


namespace count_N_lt_500_solution_exists_l769_769290

theorem count_N_lt_500_solution_exists:
  ∃ (N : ℕ), N < 500 ∧ (∃ (x : ℝ), x^floor x = N) = 287 :=
sorry

end count_N_lt_500_solution_exists_l769_769290


namespace count_solutions_eq_287_l769_769293

noncomputable def count_solutions : ℕ :=
  (({n | ∃ x : ℝ, n < 500 ∧ (⌊x⌋.to_nat ≥ 0) ∧ x^(⌊x⌋.to_nat) = n} : set ℕ).to_finset.card)

theorem count_solutions_eq_287 : count_solutions = 287 :=
  sorry

end count_solutions_eq_287_l769_769293


namespace largest_domain_of_g_l769_769371

noncomputable def g (x : ℝ) : ℝ := sorry

theorem largest_domain_of_g :
  (∀ x : ℝ, x ∈ domain g → x^2 ∈ domain g ∧ g(x) + g(x^2) = x^3) →
  {x : ℝ | x ∈ domain g} = {x : ℝ | x ≠ -1} :=
by
  intros h
  sorry

end largest_domain_of_g_l769_769371


namespace part1_part2_l769_769909

-- Function definition f for Part (1) and Part (2)
def f (x k : ℝ) : ℝ := (x^2)/2 + (1 - k) * x - k * real.log x

-- Define part1 for the tangent line equation given k=1
theorem part1 : (f 1 1 = 1/2) := 
by
  sorry

-- Define part2 for the inequality proof
theorem part2 (k : ℝ) (x : ℝ) (hk : k > 0) : 
  f x k + (3/2) * k^2 - 2 * k ≥ 0 := 
by
  sorry

end part1_part2_l769_769909


namespace increasing_intervals_sin_A_div_sin_C_l769_769273

noncomputable def f (x : ℝ) : ℝ := sin x * (cos x - sqrt 3 * sin x)

theorem increasing_intervals : 
  ∀ x ∈ set.Icc (0 : ℝ) π, 
  (0 ≤ x ∧ x ≤ π / 12) ∨ (7 * π / 12 ≤ x ∧ x ≤ π) → 
  ∃ δ > 0, ∀ ε > 0, ε < δ → f (x + ε) > f x :=
sorry

theorem sin_A_div_sin_C (A B C a b c : ℝ) 
  (hABC : A + B + C = π)
  (h_fB : f B = 0)
  (h_arith : (a + b) / 2 = b / 2 + sqrt 3 * c / 2)
  (h_positive : b > a ∧ sqrt 3 * c > a) :
  (sin A / sin C = sqrt 3 * (3 - 2 * sqrt 2) / 3) :=
sorry

end increasing_intervals_sin_A_div_sin_C_l769_769273


namespace derivative_y_l769_769614

-- Define the function y
def y (x : ℝ) : ℝ := (cos (arccot 3) * (cos (14 * x))^2) / (28 * sin (28 * x))

-- Goal is to prove the derivative of y is equal to the given expression
theorem derivative_y : 
  ∀ x : ℝ, 
  (deriv y x) = - (cos (arccot 3)) / (4 * (sin (14 * x))^2) :=
by
  sorry

end derivative_y_l769_769614


namespace apples_left_proof_l769_769008

def apples_left (mike_apples : Float) (nancy_apples : Float) (keith_apples_eaten : Float): Float :=
  mike_apples + nancy_apples - keith_apples_eaten

theorem apples_left_proof :
  apples_left 7.0 3.0 6.0 = 4.0 :=
by
  unfold apples_left
  norm_num
  sorry

end apples_left_proof_l769_769008


namespace sin_cos_identity_l769_769196

theorem sin_cos_identity : sin 43 * cos 13 + sin 47 * cos 103 = 1 / 2 :=
by
  sorry

end sin_cos_identity_l769_769196


namespace minimal_set_A_l769_769079

def is_sum_of_elements (A : Set ℤ) (n : ℤ) : Prop :=
  ∃ (x y : ℤ), x ∈ A ∧ y ∈ A ∧ n = x + y

theorem minimal_set_A :
  ∀ (A : Set ℤ),
    (1 ∈ A) → 
    (100 ∈ A) → 
    (∀ (a : ℤ), a ∈ A → 
      a = 1 ∨ is_sum_of_elements A a) →
    (∃ (S : Set ℤ), 
      (S = {1, 2, 4, 8, 16, 32, 64}) ∧ ∀ (B : Set ℤ),
      (1 ∈ B) →
      (100 ∈ B) →
      (∀ (b : ℤ), b ∈ B → 
        b = 1 ∨ is_sum_of_elements B b) →
      (S ⊆ B) → (S.card ≤ B.card)) :=
sorry

end minimal_set_A_l769_769079


namespace geography_class_grade_distribution_l769_769326

theorem geography_class_grade_distribution (total_students : ℕ) (n_b : ℚ) :
  total_students = 50 →
  let n_a := 0.8 * n_b in
  let n_c := 1.2 * n_b in
  n_a + n_b + n_c = total_students →
  n_b = 50 / 3 :=
by {
  intros,
  sorry
}

end geography_class_grade_distribution_l769_769326


namespace prove_inequality_l769_769258

-- Defining properties of f
variable {α : Type*} [LinearOrderedField α] (f : α → α)

-- Condition 1: f is even function
def is_even_function (f : α → α) : Prop := ∀ x : α, f (-x) = f x

-- Condition 2: f is monotonically increasing on (0, ∞)
def is_monotonically_increasing_on_positive (f : α → α) : Prop := ∀ ⦃x y : α⦄, 0 < x → 0 < y → x < y → f x < f y

-- Define the main theorem we need to prove:
theorem prove_inequality (h1 : is_even_function f) (h2 : is_monotonically_increasing_on_positive f) : 
  f (-1) < f 2 ∧ f 2 < f (-3) :=
by
  sorry

end prove_inequality_l769_769258


namespace brandon_initial_skittles_l769_769575

theorem brandon_initial_skittles (initial_skittles : ℕ) (loss : ℕ) (final_skittles : ℕ) (h1 : final_skittles = 87) (h2 : loss = 9) (h3 : final_skittles = initial_skittles - loss) : initial_skittles = 96 :=
sorry

end brandon_initial_skittles_l769_769575


namespace trig_identity_l769_769947

theorem trig_identity (α β : ℝ)
  (hα1 : 0 < α)
  (hα2 : α < (π / 2))
  (hβ1 : 0 < β)
  (hβ2 : β < (π / 2))
  (hcosα : Real.cos α = 7 / Real.sqrt 50)
  (htgβ : Real.tan β = 1 / 3) :
  α + 2 * β = π / 4 := 
sorry

end trig_identity_l769_769947


namespace equation_of_line_with_slope_angle_45_and_y_intercept_neg1_l769_769966

theorem equation_of_line_with_slope_angle_45_and_y_intercept_neg1 : 
  ∃ (x y : ℝ), ∃ (c : ℝ), (c = -1) → (tan 45 = 1) → (y = x - 1) ∧ (x - y - c = 0) :=
by
  sorry

end equation_of_line_with_slope_angle_45_and_y_intercept_neg1_l769_769966


namespace part_a_impossibility_l769_769659

-- Define the number of rows and columns
def num_rows : ℕ := 20
def num_columns : ℕ := 15

-- Define a function that checks if the sum of the counts in rows and columns match the conditions
def is_possible_configuration : Prop :=
  (num_rows % 2 = 0) ∧ (num_columns % 2 = 1)

theorem part_a_impossibility : ¬ is_possible_configuration :=
by
  -- The proof for the contradiction will go here
  sorry

end part_a_impossibility_l769_769659


namespace line_slope_of_circle_intersection_l769_769343

/-
Problem setup:
 1. Circle equation: (x+6)^2 + y^2 = 25
 2. Parametric line equations: x = t * cos α, y = t * sin α
 3. Distance between intersection points A and B is √10

Goal: Prove that the slope of the line l is either sqrt(15) / 3 or - sqrt(15) / 3.
-/

-- Define the circle equation
def is_on_circle (x y : ℝ) : Prop := (x + 6)^2 + y^2 = 25

-- Parametric equations of the line
def parametric_line (t α : ℝ) (x y : ℝ) : Prop := x = t * Real.cos α ∧ y = t * Real.sin α

-- Distance between points A and B
def distance_ab (ρ1 ρ2 : ℝ) : ℝ := Real.sqrt ((ρ1 + ρ2)^2 - 4 * ρ1 * ρ2)

-- Slope of the line
def slope_of_line (α : ℝ) : ℝ := Real.tan α

theorem line_slope_of_circle_intersection 
  (x y t α ρ1 ρ2 : ℝ) 
  (h_circle : is_on_circle x y) 
  (h_line : parametric_line t α x y) 
  (h_distance : distance_ab ρ1 ρ2 = Real.sqrt 10) : 
  slope_of_line α = Real.sqrt 15 / 3 ∨ slope_of_line α = - Real.sqrt 15 / 3 :=
sorry

end line_slope_of_circle_intersection_l769_769343


namespace max_an_div_an_minus1_l769_769649

theorem max_an_div_an_minus1 : 
  ∀ {a_n S_n : ℕ → ℝ}, 
   (∀ n, a_n n = 3 * S_n n / (n + 2)) → 
   (∀ n > 1, S_n n = (n + 2) / 3 * a_n n) →
    ∃ n₀, (∀ n ≥ n₀, a_n n / a_n (n - 1) ≤ 5 / 3) ∧ 
          (a_n 2 / a_n 1 = 5 / 3) :=
begin
  sorry
end

end max_an_div_an_minus1_l769_769649


namespace max_horizontal_distance_domino_l769_769544

theorem max_horizontal_distance_domino (n : ℕ) : 
    (n > 0) → ∃ d, d = 2 * Real.log n := 
by {
    sorry
}

end max_horizontal_distance_domino_l769_769544


namespace tan_double_angle_l769_769245

variable {α : ℝ}

-- Conditions
def in_third_quadrant (α : ℝ) : Prop := π < α ∧ α < 3 * π / 2
def cos_pi_minus_alpha (α : ℝ) : Prop := cos (π - α) = 4 / 5

-- Question to Prove
theorem tan_double_angle (h1 : in_third_quadrant α) (h2 : cos_pi_minus_alpha α) : 
  tan (2 * α) = 24 / 7 := 
by
  sorry

end tan_double_angle_l769_769245


namespace projectile_height_reaches_35_l769_769415

theorem projectile_height_reaches_35 
  (t : ℝ)
  (h_eq : -4.9 * t^2 + 30 * t = 35) :
  t = 2 ∨ t = 50 / 7 ∧ t = min (2 : ℝ) (50 / 7) :=
by
  sorry

end projectile_height_reaches_35_l769_769415


namespace mechanic_worked_hours_l769_769930

theorem mechanic_worked_hours 
  (total_cost : ℝ) 
  (part_cost : ℝ) 
  (part_count : ℕ) 
  (labor_cost_per_minute : ℝ) 
  (parts_total_cost : ℝ) 
  (labor_total_cost : ℝ) 
  (hours_worked : ℝ) 
  (h1 : total_cost = 220) 
  (h2 : part_count = 2) 
  (h3 : part_cost = 20) 
  (h4 : labor_cost_per_minute = 0.5) 
  (h5 : parts_total_cost = part_count * part_cost) 
  (h6 : labor_total_cost = total_cost - parts_total_cost) 
  (labor_cost_per_hour := labor_cost_per_minute * 60) 
  (h7 : hours_worked = labor_total_cost / labor_cost_per_hour) : 
  hours_worked = 6 := 
sorry

end mechanic_worked_hours_l769_769930


namespace min_value_M_l769_769673

theorem min_value_M (a b : ℕ) (ha: 0 < a) (hb: 0 < b) : ∃ a b, M = 3 * a^2 - a * b^2 - 2 * b - 4 ∧ M = 2 := sorry

end min_value_M_l769_769673


namespace count_factors_180_multiple_of_15_is_6_l769_769819

-- We define a function to check if 'n' is a factor of 'm'
def is_factor (m n : ℕ) : Prop := n ∣ m

-- We define a function to check if 'n' is a multiple of 'd'
def is_multiple (n d : ℕ) : Prop := d ∣ n

-- We define a function to collect all factors of a number 'n'
def factors (n : ℕ) : list ℕ := 
  -- We use list.filter to keep elements that satisfy is_factor n
  list.filter (λ x, is_factor n x) (list.range (n + 1))

-- We count the number of elements in a list that satisfy a predicate
def count_satisfying {α : Type*} (pred : α → Prop) [decidable_pred pred] (l : list α) : ℕ :=
  list.length (list.filter pred l)

-- Noncomputable definition to specify that counting number
noncomputable def count_positive_factors_of_180_multiple_of_15 : ℕ :=
  count_satisfying (λ x, is_multiple x 15) (factors 180)

-- We state the theorem equivalence to the proof problem
theorem count_factors_180_multiple_of_15_is_6 : count_positive_factors_of_180_multiple_of_15 = 6 :=
  sorry


end count_factors_180_multiple_of_15_is_6_l769_769819


namespace entire_function_with_properties_exists_l769_769210

noncomputable def exists_entire_function_with_properties : Prop :=
  ∃ (F : ℂ → ℂ), 
    (∀ z : ℂ, F z ≠ 0) ∧ 
    (∀ z : ℂ, |F z| ≤ exp (|z|)) ∧ 
    (∀ y : ℝ, |F (complex.I * y)| ≤ 1) ∧ 
    (∃ (roots : ℕ → ℝ), ∀ n : ℕ, F (roots n) = 0)
    
theorem entire_function_with_properties_exists : exists_entire_function_with_properties :=
sorry

end entire_function_with_properties_exists_l769_769210


namespace count_solutions_eq_287_l769_769291

noncomputable def count_solutions : ℕ :=
  (({n | ∃ x : ℝ, n < 500 ∧ (⌊x⌋.to_nat ≥ 0) ∧ x^(⌊x⌋.to_nat) = n} : set ℕ).to_finset.card)

theorem count_solutions_eq_287 : count_solutions = 287 :=
  sorry

end count_solutions_eq_287_l769_769291


namespace factors_of_180_l769_769773

-- Define a predicate to check if a number is a factor of another number
def is_factor (m n : ℕ) : Prop := n % m = 0

-- Define a predicate to check if a number is a multiple of another number
def is_multiple (m n : ℕ) : Prop := m % n = 0

-- Define the problem
theorem factors_of_180 : {n : ℕ // is_factor n 180 ∧ is_multiple n 15}.card = 6 :=
by sorry

end factors_of_180_l769_769773


namespace cos2alpha_necessary_but_not_sufficient_l769_769511

theorem cos2alpha_necessary_but_not_sufficient (α : ℝ) :
  (cos(2 * α) = 0 → sin(α) + cos(α) = 0) ∧ ¬(sin(α) + cos(α) = 0 → cos(2 * α) = 0) :=
  sorry

end cos2alpha_necessary_but_not_sufficient_l769_769511


namespace solution_set_of_inequality_l769_769134

theorem solution_set_of_inequality (x : ℝ) : (|x + 1| - |x - 3| ≥ 0) ↔ (1 ≤ x) := 
sorry

end solution_set_of_inequality_l769_769134


namespace chord_lengths_l769_769858

/-- Given a circle with radius 7 units, perpendicular diameters CD and AB,
    if chord CH intersects AB at point K such that AK = 3 units
    and CH = 12 units, then the lengths of segments AK and KB
    are 3 units and 11 units, respectively. -/
theorem chord_lengths (r : ℝ) (ch : ℝ) (ak : ℝ) (ab : ℝ) (cd : ℝ) (kb : ℝ) :
  r = 7 ∧ ch = 12 ∧ ak = 3 ∧ ab = 2 * r ∧ ab ⊥ cd → kb = ab - ak :=
by sorry

end chord_lengths_l769_769858


namespace fraction_s_over_r_quadratic_completion_l769_769951

theorem fraction_s_over_r_quadratic_completion 
  (k : ℝ) (d r s : ℝ)
  (h : 5 * k^2 - 6 * k + 15 = d * (k + r)^2 + s)
  (hd : d = 5)
  (hr : r = -3/5)
  (hs : s = 66/5) :
  s / r = -22 := by
sory

end fraction_s_over_r_quadratic_completion_l769_769951


namespace abs_inequality_l769_769313

theorem abs_inequality (a b c : ℝ) (h : |a - c| < |b|) : |a| < |b| + |c| :=
sorry

end abs_inequality_l769_769313


namespace angle_sum_180_l769_769856

open EuclideanGeometry

variables {A B C P D M : Point}
variables (h : ∠ACB = 90) (M_midpoint : Midpoint M B C)
variables (D_on_BC : orthogonal_projection A D BC) (P_on_AM : P ∈ Segment A M) (PD_eq_AD : dist P D = dist A D)

theorem angle_sum_180 :
  ∠BAC + ∠BPC = 180 :=
begin
  sorry
end

end angle_sum_180_l769_769856


namespace bushes_needed_l769_769566

theorem bushes_needed
  (num_sides : ℕ) (side_length : ℝ) (bush_fill : ℝ) (total_length : ℝ) (num_bushes : ℕ) :
  num_sides = 3 ∧ side_length = 16 ∧ bush_fill = 4 ∧ total_length = num_sides * side_length ∧ num_bushes = total_length / bush_fill →
  num_bushes = 12 := by
  sorry

end bushes_needed_l769_769566


namespace find_starting_number_of_range_l769_769085

theorem find_starting_number_of_range :
  ∃ n : ℕ, ∀ k : ℕ, k < 7 → (n + k * 9) ∣ 9 ∧ (n + k * 9) ≤ 97 ∧ (∀ m < k, (n + m * 9) < n + (m + 1) * 9) := 
sorry

end find_starting_number_of_range_l769_769085


namespace find_largest_n_l769_769225

theorem find_largest_n :
  ∃ n : ℕ, n < 50000 ∧ 9 * (n-1)^3 - 3 * n^3 + 19 * n + 27 % 3 = 0 ∧ ∀ m : ℕ, m < 50000 ∧ 9 * (m-1)^3 - 3 * m^3 + 19 * m + 27 % 3 = 0 → m ≤ n :=
  by
    -- existence of n
    let n := 49998
    use n
    split
    -- condition 1 
    exact Nat.lt_succ_self 49999
    split
    -- condition 2
    sorry -- proof of 49998 being multiple of 3
    -- condition 3
    sorry -- proof of 49998 being the largest less than 50000

end find_largest_n_l769_769225


namespace sequence_sum_l769_769249

-- Given conditions
def a_n (n : ℕ) : ℤ := -4 * n + 5
def q (n : ℕ) (hn : n ≥ 2) : ℤ := a_n n - a_n (n - 1)
def b_1 : ℤ := 2
def b_n (n : ℕ) : ℤ := b_1 * (-4)^(n - 1)

-- Given question to prove the sum of |b_1| + |b_2| + ... + |b_n| = 4^n - 1
theorem sequence_sum (n : ℕ) :
  (finset.range (n + 1)).sum (λ i, |b_n i|) = 4^n - 1 := 
sorry

end sequence_sum_l769_769249


namespace count_factors_180_multiple_of_15_is_6_l769_769812

-- We define a function to check if 'n' is a factor of 'm'
def is_factor (m n : ℕ) : Prop := n ∣ m

-- We define a function to check if 'n' is a multiple of 'd'
def is_multiple (n d : ℕ) : Prop := d ∣ n

-- We define a function to collect all factors of a number 'n'
def factors (n : ℕ) : list ℕ := 
  -- We use list.filter to keep elements that satisfy is_factor n
  list.filter (λ x, is_factor n x) (list.range (n + 1))

-- We count the number of elements in a list that satisfy a predicate
def count_satisfying {α : Type*} (pred : α → Prop) [decidable_pred pred] (l : list α) : ℕ :=
  list.length (list.filter pred l)

-- Noncomputable definition to specify that counting number
noncomputable def count_positive_factors_of_180_multiple_of_15 : ℕ :=
  count_satisfying (λ x, is_multiple x 15) (factors 180)

-- We state the theorem equivalence to the proof problem
theorem count_factors_180_multiple_of_15_is_6 : count_positive_factors_of_180_multiple_of_15 = 6 :=
  sorry


end count_factors_180_multiple_of_15_is_6_l769_769812


namespace member_number_property_l769_769870

theorem member_number_property :
  ∃ (country : Fin 6) (member_number : Fin 1978),
    (∀ (i j : Fin 1978), i ≠ j → member_number ≠ i + j) ∨
    (∀ (k : Fin 1978), member_number ≠ 2 * k) :=
by
  sorry

end member_number_property_l769_769870


namespace minimum_value_a_l769_769271

def f (x : ℝ) : ℝ := 1 / (1 + 2^x)

theorem minimum_value_a (a : ℝ) : 
  (∀ x > 0, f (a * Real.exp x) ≤ 1 - f (Real.log a - Real.log x)) → a ≥ Real.exp (-1) :=
sorry

end minimum_value_a_l769_769271


namespace area_of_triangle_l769_769223

-- Define the lines using equations
def line1 (x : ℝ) : Prop := ∃ y : ℝ, y - 4 * x = -2
def line2 (x : ℝ) : Prop := ∃ y : ℝ, 2 * y + x = 12

-- Define the vertices: find y-intercepts and intersection point
def vertex1 : ℝ × ℝ := (0, -2)
def vertex2 : ℝ × ℝ := (0, 6)
def vertex3 : ℝ × ℝ := (16 / 9, 46 / 9)

-- Define the area of the triangle formed by these vertices
def triangle_area : ℝ := (1 / 2) * 8 * (16 / 9)

theorem area_of_triangle : triangle_area = 64 / 9 := by
  -- Here you'd normally provide the proof, but we'll leave it as sorry.
  sorry

end area_of_triangle_l769_769223


namespace positive_factors_of_180_multiple_of_15_count_l769_769755

theorem positive_factors_of_180_multiple_of_15_count : 
  let factorization_180 := 2^2 * 3^2 * 5;
  let is_multiple_of_15 (n : ℕ) := ∃ k1 k2 : ℕ, k1 * 3 = n ∧ k2 * 5 = n;
  let factors_180 := { n : ℕ | n ∣ factorization_180 };
  { n : ℕ | n ∈ factors_180 ∧ is_multiple_of_15 n }.card = 6 :=
by
  sorry

end positive_factors_of_180_multiple_of_15_count_l769_769755


namespace smallest_integer_eight_divisors_l769_769483

theorem smallest_integer_eight_divisors : ∃ (n : ℕ), (∀ m : ℕ, (∀ d ∈ divisors m, d ∈ divisors n) → (∃! m, m = 24 ∧ (divisors m).card = 8)) := sorry

end smallest_integer_eight_divisors_l769_769483


namespace problem1_problem2_l769_769679

theorem problem1 (a b : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = a * x^3 + b)
                 (h2 : deriv f 1 = 3) :
  f = λ x, x^3 + 1 := by
sorry

theorem problem2 (f : ℝ → ℝ) (h1 : ∀ x, f x = x^3 + 1)
                 (h2 : deriv f (-1) = 3) (h3 : deriv f (1/2) = 3 / 4)
                 (h4 : f (-1) = 0)  :
  ∃ m n : ℝ, (m ≠ -1 → m ≠ 1/2 → (deriv f m = (n / (m + 1)) ∧ f m = n))
  ∧ (y = 3 * x + 3 ∨ y = 3 / 4 * x + 3 / 4) := by
sorry

end problem1_problem2_l769_769679


namespace line_equation_through_two_points_l769_769397

noncomputable def LineEquation (x0 y0 x1 y1 x y : ℝ) : Prop :=
  (x1 ≠ x0) → (y1 ≠ y0) → 
  (y - y0) / (y1 - y0) = (x - x0) / (x1 - x0)

theorem line_equation_through_two_points 
  (x0 y0 x1 y1 : ℝ) 
  (h₁ : x1 ≠ x0) 
  (h₂ : y1 ≠ y0) : 
  ∀ (x y : ℝ), LineEquation x0 y0 x1 y1 x y :=  
by
  sorry

end line_equation_through_two_points_l769_769397


namespace volume_tetrahedron_eq_ten_l769_769208

noncomputable def tetrahedron_volume
  (PQ PR PS QR QS RS : ℝ)
  (h_PQ : PQ = 3)
  (h_PR : PR = 4)
  (h_PS : PS = 5)
  (h_QR : QR = 5)
  (h_QS : QS = sqrt 34)
  (h_RS : RS = sqrt 41) : ℝ :=
  let volume_PQRS := 10 in
  volume_PQRS

theorem volume_tetrahedron_eq_ten
  (PQ PR PS QR QS RS : ℝ)
  (h_PQ : PQ = 3)
  (h_PR : PR = 4)
  (h_PS : PS = 5)
  (h_QR : QR = 5)
  (h_QS : QS = sqrt 34)
  (h_RS : RS = sqrt 41) :
  tetrahedron_volume PQ PR PS QR QS RS h_PQ h_PR h_PS h_QR h_QS h_RS = 10 :=
begin
  sorry
end

end volume_tetrahedron_eq_ten_l769_769208


namespace remainder_approx_l769_769110

def x : ℝ := 74.99999999999716 * 96
def y : ℝ := 74.99999999999716
def quotient : ℝ := 96
def expected_remainder : ℝ := 0.4096

theorem remainder_approx (x y : ℝ) (quotient : ℝ) (h1 : y = 74.99999999999716)
  (h2 : quotient = 96) (h3 : x = y * quotient) :
  x - y * quotient = expected_remainder :=
by
  sorry

end remainder_approx_l769_769110


namespace miles_per_tank_l769_769064

def cost_per_fill := 45
def distance_to_grammys := 2000
def food_ratio := 3 / 5
def total_expense := 288

theorem miles_per_tank :
  let T := total_expense / (1 + food_ratio) in
  let num_fills := T / cost_per_fill in
  distance_to_grammys / num_fills = 500 := 
by sorry

end miles_per_tank_l769_769064


namespace no_prime_divisible_by_91_l769_769822

theorem no_prime_divisible_by_91 : ¬ ∃ p : ℕ, p > 1 ∧ Prime p ∧ 91 ∣ p :=
by
  sorry

end no_prime_divisible_by_91_l769_769822


namespace rate_of_initial_investment_l769_769158

def initial_investment : ℝ := 8000
def additional_investment : ℝ := 4000
def additional_rate : ℝ := 8
def total_amount : ℝ := 12000
def total_rate : ℝ := 6
def interest_initial (R : ℝ) : ℝ := (initial_investment * R) / 100
def interest_additional : ℝ := (additional_investment * additional_rate) / 100
def total_interest : ℝ := (total_amount * total_rate) / 100

theorem rate_of_initial_investment : ∃ R, interest_initial R + interest_additional = total_interest ∧ R = 5 :=
by 
  sorry

end rate_of_initial_investment_l769_769158


namespace log2_50_between_and_sum_eq_11_l769_769989

theorem log2_50_between_and_sum_eq_11 :
  ∃ c d : ℤ, c < d ∧ (c ≤ Real.log 50 / Real.log 2) ∧ (Real.log 50 / Real.log 2 < d) ∧ (c + d = 11) :=
by {
  let c := 5,
  let d := 6,
  use [c, d],
  split,
  { exact int.lt_succ_self c },
  split,
  { exact Real.log_le_log_of_log_le (by norm_num) (by norm_num) (by norm_num1: 32 < 50) },
  split,
  { exact Real.log_le_log_of_log_le (by norm_num) (by norm_num) (by norm_num2: 50 < 64).trans (by norm_num) },
  { exact add_self_eq_zero.mp (by norm_num) }
}

end log2_50_between_and_sum_eq_11_l769_769989


namespace solve_system_l769_769983

theorem solve_system :
  ∃ x y : ℝ, (x + y = 5) ∧ (x + 2 * y = 8) ∧ (x = 2) ∧ (y = 3) :=
by
  sorry

end solve_system_l769_769983


namespace sum_of_squares_of_ages_l769_769466

theorem sum_of_squares_of_ages 
  (d t h : ℕ) 
  (cond1 : 3 * d + t = 2 * h)
  (cond2 : 2 * h ^ 3 = 3 * d ^ 3 + t ^ 3)
  (rel_prime : Nat.gcd d (Nat.gcd t h) = 1) :
  d ^ 2 + t ^ 2 + h ^ 2 = 42 :=
sorry

end sum_of_squares_of_ages_l769_769466


namespace non_juniors_play_instrument_l769_769864

theorem non_juniors_play_instrument (total_students juniors non_juniors play_instrument_juniors play_instrument_non_juniors total_do_not_play : ℝ) :
  total_students = 600 →
  play_instrument_juniors = 0.3 * juniors →
  play_instrument_non_juniors = 0.65 * non_juniors →
  total_do_not_play = 0.4 * total_students →
  0.7 * juniors + 0.35 * non_juniors = total_do_not_play →
  juniors + non_juniors = total_students →
  non_juniors * 0.65 = 334 :=
by
  sorry

end non_juniors_play_instrument_l769_769864


namespace pumps_fill_time_l769_769536

-- Definitions for the rates and the time calculation
def small_pump_rate : ℚ := 1 / 3
def large_pump_rate : ℚ := 4
def third_pump_rate : ℚ := 1 / 2

def total_pump_rate : ℚ := small_pump_rate + large_pump_rate + third_pump_rate

theorem pumps_fill_time :
  1 / total_pump_rate = 6 / 29 :=
by
  -- Definition of the rates has already been given.
  -- Here we specify the calculation for the combined rate and filling time.
  sorry

end pumps_fill_time_l769_769536


namespace count_positive_factors_of_180_multiple_of_15_l769_769731

statement: ℕ
statement := sorry

theorem count_positive_factors_of_180_multiple_of_15 :
  let factors_of_180 := {d ∈ (finset.range 181).filter (λ x, 180 % x = 0)}
  let multiples_of_15 := (finset.range 181).filter (λ x, x % 15 = 0)
  let desired_factors := factors_of_180 ∩ multiples_of_15
  desired_factors.card = 6 :=
by
  sorry

end count_positive_factors_of_180_multiple_of_15_l769_769731


namespace quadrilateral_inequality_l769_769033

theorem quadrilateral_inequality
  (a b c d x1 x2 : ℝ)
  (α1 α2: ℝ)
  (h_a_gt_0 : a > 0)
  (h_b_gt_0 : b > 0)
  (h_c_gt_0 : c > 0)
  (h_d_gt_0 : d > 0)
  (h_x1_gt_0 : x1 > 0)
  (h_x2_gt_0 : x2 > 0)
  (h_alpha1_lt_halfpi : α1 < real.pi / 2)
  (h_alpha2_gt_halfpi : α2 > real.pi / 2)
  (h_cos_alpha_eq_ratio : real.cos α1 / real.cos (real.pi - α2) = x1 / x2)
  : (a / x1 + d / x2 < b / x1 + c / x2) := by
  sorry

end quadrilateral_inequality_l769_769033


namespace milo_eggs_weight_l769_769388

def weight_of_one_egg : ℚ := 1/16
def eggs_per_dozen : ℕ := 12
def dozens_needed : ℕ := 8

theorem milo_eggs_weight :
  (dozens_needed * eggs_per_dozen : ℚ) * weight_of_one_egg = 6 := by sorry

end milo_eggs_weight_l769_769388


namespace annual_income_before_tax_l769_769852

variable (I : ℝ) -- Define I as the annual income before tax

-- Conditions
def original_tax (I : ℝ) : ℝ := 0.42 * I
def new_tax (I : ℝ) : ℝ := 0.32 * I
def differential_savings (I : ℝ) : ℝ := original_tax I - new_tax I

-- Theorem: Given the conditions, the taxpayer's annual income before tax is $42,400
theorem annual_income_before_tax : differential_savings I = 4240 → I = 42400 := by
  sorry

end annual_income_before_tax_l769_769852


namespace count_factors_180_multiple_of_15_is_6_l769_769816

-- We define a function to check if 'n' is a factor of 'm'
def is_factor (m n : ℕ) : Prop := n ∣ m

-- We define a function to check if 'n' is a multiple of 'd'
def is_multiple (n d : ℕ) : Prop := d ∣ n

-- We define a function to collect all factors of a number 'n'
def factors (n : ℕ) : list ℕ := 
  -- We use list.filter to keep elements that satisfy is_factor n
  list.filter (λ x, is_factor n x) (list.range (n + 1))

-- We count the number of elements in a list that satisfy a predicate
def count_satisfying {α : Type*} (pred : α → Prop) [decidable_pred pred] (l : list α) : ℕ :=
  list.length (list.filter pred l)

-- Noncomputable definition to specify that counting number
noncomputable def count_positive_factors_of_180_multiple_of_15 : ℕ :=
  count_satisfying (λ x, is_multiple x 15) (factors 180)

-- We state the theorem equivalence to the proof problem
theorem count_factors_180_multiple_of_15_is_6 : count_positive_factors_of_180_multiple_of_15 = 6 :=
  sorry


end count_factors_180_multiple_of_15_is_6_l769_769816


namespace calculate_expression_l769_769577

theorem calculate_expression : 
  10 * 9 * 8 + 7 * 6 * 5 + 6 * 5 * 4 + 3 * 2 * 1 - 9 * 8 * 7 - 8 * 7 * 6 - 5 * 4 * 3 - 4 * 3 * 2 = 132 :=
by
  sorry

end calculate_expression_l769_769577


namespace gcd_72_108_150_l769_769480

theorem gcd_72_108_150 : Nat.gcd (Nat.gcd 72 108) 150 = 6 := by
  sorry

end gcd_72_108_150_l769_769480


namespace log_of_50_between_consecutive_integers_l769_769991

theorem log_of_50_between_consecutive_integers : 
  (∃ c d : ℕ, 4 < log 2 50 ∧ log 2 50 < 6 ∧ c = 5 ∧ d = 6 ∧ c + d = 11) :=
by
  -- Conditions
  have h1 : log 2 16 = 4 := by norm_num,
  have h2 : log 2 64 = 6 := by norm_num,
  have h3 : 16 < 50 := by norm_num,
  have h4 : 50 < 64 := by norm_num,
  sorry

end log_of_50_between_consecutive_integers_l769_769991


namespace gum_total_l769_769040

theorem gum_total (initial_gum : ℝ) (additional_gum : ℝ) : initial_gum = 18.5 → additional_gum = 44.25 → initial_gum + additional_gum = 62.75 :=
by
  intros
  sorry

end gum_total_l769_769040


namespace total_skips_is_33_l769_769603

theorem total_skips_is_33 {
  let skips_5 := 8,
  ∃ skips_4 skips_3 skips_2 skips_1 : ℕ,
  (skips_5 = skips_4 + 1) ∧
  (skips_4 = skips_3 - 3) ∧
  (skips_3 = skips_2 * 2) ∧
  (skips_2 = skips_1 + 2) ∧
  (skips_1 + skips_2 + skips_3 + skips_4 + skips_5 = 33) 
} sorry

end total_skips_is_33_l769_769603


namespace factors_of_180_multiple_of_15_count_l769_769803

theorem factors_of_180_multiple_of_15_count :
  ∃ n : Nat, n = 6 ∧ ∀ x : Nat, x ∣ 180 → 15 ∣ x → x > 0 :=
by
  sorry

end factors_of_180_multiple_of_15_count_l769_769803


namespace stock_investor_loss_l769_769116

theorem stock_investor_loss (a : ℝ) (n : ℕ) :
  let final_price := a * (0.99) ^ n
  in final_price < a :=
by
  intros
  unfold final_price
  sorry

end stock_investor_loss_l769_769116


namespace arithmetic_sequences_count_l769_769913

def S (n : ℕ) : Set ℕ := {i | 1 ≤ i ∧ i ≤ n}

def is_arithmetic_sequence (A : List ℕ) : Prop :=
  ∃ d > 0, ∀ i j, i < j → j < A.length →
  A.nth i + d = A.nth j 

def valid_sequences_count (n : ℕ) : ℕ :=
  ⌊n^2 / 4⌋

theorem arithmetic_sequences_count (n : ℕ) :
  ∃ A : List ℕ, (∀ i, A.nth i ∈ S n) ∧ is_arithmetic_sequence A ∧ 
  (∀ B, (∀ i, B.nth i ∈ S n) ∧ is_arithmetic_sequence B → B = A) →
  valid_sequences_count n = ⌊n^2 / 4⌋ :=
sorry

end arithmetic_sequences_count_l769_769913


namespace percentage_change_area_l769_769435

theorem percentage_change_area (L B : ℝ) (h : L > 0 ∧ B > 0) :
  let A1 := L * B
  let L2 := L / 2
  let B2 := 3 * B
  let A2 := L2 * B2
  let percentage_change := ((A2 - A1) / A1) * 100
  percentage_change = 50 := by
  have A1_def := calc
    A1 = L * B : rfl
  have A2_def := calc
    A2 = (L / 2) * (3 * B) : rfl
    ... = (3 / 2) * (L * B) : by ring
  have pc_def := calc
    percentage_change = ((A2 - A1) / A1) * 100 : rfl
    ... = (( (3 / 2) * (L * B) - (L * B) ) / (L * B) ) * 100 : by rw [A1_def, A2_def]
    ... = (1 / 2) * 100 : by ring
  exact pc_def

end percentage_change_area_l769_769435


namespace sum_inverse_diff_correct_l769_769694

-- Define the sequence a_n
def a_n (n : ℕ) : ℕ := 2^n + 1

-- Define the sum to be proved
def sum_inverse_diff (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ k, 1 / (a_n (k + 1) - a_n k))

-- Define the expected value of the sum
def expected_value (n : ℕ) : ℝ := 1 - (1 / 2^n)

-- The theorem to prove
theorem sum_inverse_diff_correct (n : ℕ) : sum_inverse_diff n = expected_value n :=
by
  sorry

end sum_inverse_diff_correct_l769_769694


namespace distance_from_plate_to_bottom_edge_l769_769940

theorem distance_from_plate_to_bottom_edge :
    ∀ (d : ℕ), 10 + 63 = 20 + d → d = 53 :=
by
  intros d h
  sorry

end distance_from_plate_to_bottom_edge_l769_769940


namespace correct_operation_l769_769504

theorem correct_operation (a b : ℝ) : (a + b) * (a - b) = a^2 - b^2 := by
  sorry

end correct_operation_l769_769504


namespace sequence_behavior_l769_769247

noncomputable def seq (a : ℝ) : ℕ → ℝ
| 0 => a
| (n+1) => a ^ (seq n)

theorem sequence_behavior (a : ℝ) (n : ℕ) (h : 0 < a ∧ a < 1) :
  (∀ k, seq a (2 * k + 1) < seq a (2 * k + 3)) ∧ (∀ k, seq a (2 * k) > seq a (2 * k + 2)) :=
sorry

end sequence_behavior_l769_769247


namespace prob_of_perfect_square_divisor_m_plus_n_l769_769533

theorem prob_of_perfect_square_divisor :
  let n := 15.factorial
  let divisors := finset.range (n + 1).filter (λ d, (n % d) = 0)
  let perfect_squares := divisors.filter (λ d, (sqrt d) ^ 2 = d)
  let p := (perfect_squares.card : ℚ) / (divisors.card : ℚ)
  p = 1 / 84 :=
begin
  sorry
end

theorem m_plus_n :
  let m := 1
  let n := 84
  m + n = 85 :=
begin
  sorry
end

end prob_of_perfect_square_divisor_m_plus_n_l769_769533


namespace fixed_point_l769_769851

section parabola

def parabola (p x : ℝ) : ℝ := 2 * x ^ 2 - p * x + 4 * p + 1

theorem fixed_point : ∀ (p : ℝ), parabola p 4 = 33 :=
by
  intro p
  have h : parabola p 4 = 2 * 4 ^ 2 - p * 4 + 4 * p + 1 := rfl
  rw [h]
  norm_num
  exact add_eq_of_eq_sub' rfl

end parabola

end fixed_point_l769_769851


namespace divergence_of_a_l769_769615

noncomputable def divergence (a : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ → ℝ :=
  sorry -- divergence calculation formula goes here

/-- 
Given \(\mathbf{a} = \frac{\varphi(r)}{r} \mathbf{r}\), where \( r = \sqrt{x^2 + y^2 + z^2} \),
prove that the divergence of \(\mathbf{a}\) is \(2 \frac{\varphi(r)}{r} + \varphi'(r)\).
-/
theorem divergence_of_a (φ : ℝ → ℝ) (r : ℝ := λ r, sqrt (r.1^2 + r.2^2 + r.3^2))
  (a : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ := λ r, (φ (sqrt (r.1^2 + r.2^2 + r.3^2)) / sqrt (r.1^2 + r.2^2 + r.3^2)) • r) :
  divergence a = λ r, 2 * (φ (sqrt (r.1^2 + r.2^2 + r.3^2)) / sqrt (r.1^2 + r.2^2 + r.3^2)) + φ' (sqrt (r.1^2 + r.2^2 + r.3^2)) := sorry

end divergence_of_a_l769_769615


namespace correct_statements_l769_769242

def f (x : ℝ) : ℝ := 2^x

theorem correct_statements (x₁ x₂ : ℝ) (h1 : f(x₁ + x₂) = f(x₁) * f(x₂))
                        (h2 : f(x₁ * x₂) = f(x₁) + f(x₂))
                        (h3 : ∀ x, f(x) > 0)
                        (h4 : ∀ x, f'(x) > 0) :
  f(x₁ + x₂) = f(x₁) * f(x₂) ∧
  f(x₁ * x₂) ≠ f(x₁) + f(x₂) ∧
  ∀ x, f(x) > 0 ∧
  ∀ x, f'(x) > 0 := sorry

end correct_statements_l769_769242


namespace factors_of_180_multiple_of_15_l769_769743

/-- A factor of a number n is a positive integer which divides n without leaving a remainder. -/
def is_factor (n k : ℕ) : Prop :=
k > 0 ∧ n % k = 0

/-- A multiple of a number m is an integer which is divisible by m without leaving a remainder. -/
def is_multiple (k m : ℕ) : Prop :=
k % m = 0

/-- There are 6 factors of 180 that are also multiples of 15. -/
theorem factors_of_180_multiple_of_15 : 
  (Finset.filter (λ k, is_multiple k 15) { x | is_factor 180 x }.to_finset).card = 6 :=
by 
  sorry

end factors_of_180_multiple_of_15_l769_769743


namespace micks_equal_to_macks_l769_769831

noncomputable def mick_to_mock : ℚ := 3 / 8
noncomputable def mock_to_mack : ℚ := 6 / 5

theorem micks_equal_to_macks (micks mocks macks : ℚ) :
  (8 * micks = 3 * mocks) →
  (5 * mocks = 6 * macks) →
  (micks * 30 = 200 / 3) :=
by
  intros h1 h2
  have h3 : mocks = (30 * micks * 5) / 6 := calc
    mocks = (30 * micks * 5) / 6 : sorry
  have h4 : micks = (200 / 3) := calc
    micks = (200 / 3) : sorry
  exact h4

end micks_equal_to_macks_l769_769831


namespace correct_equation_l769_769496

theorem correct_equation (a b : ℝ) : 
  (¬ (a^2 + a^3 = a^6)) ∧ (¬ ((ab)^2 = ab^2)) ∧ (¬ ((a+b)^2 = a^2 + b^2)) ∧ ((a+b)*(a-b) = a^2 - b^2) :=
by {
  sorry
}

end correct_equation_l769_769496


namespace sequence_S5_l769_769197

-- Definitions based on the given conditions
def a₁ : ℕ := 1
def S : ℕ → ℕ
| 0     := 0
| (n+1) := S n + a₁
def a (n : ℕ) : ℕ := if n = 0 then a₁ else 2 * S n + 3

-- Main theorem to prove
theorem sequence_S5 : S 5 = 201 :=
sorry

end sequence_S5_l769_769197


namespace find_value_of_a_l769_769260

theorem find_value_of_a (a : ℝ) :
  (∃ (a : ℝ), (4 : ℝ) = - 4 / a) → a = -1 :=
by
  intro h
  cases h with a ha
  sorry

end find_value_of_a_l769_769260


namespace factors_of_180_l769_769772

-- Define a predicate to check if a number is a factor of another number
def is_factor (m n : ℕ) : Prop := n % m = 0

-- Define a predicate to check if a number is a multiple of another number
def is_multiple (m n : ℕ) : Prop := m % n = 0

-- Define the problem
theorem factors_of_180 : {n : ℕ // is_factor n 180 ∧ is_multiple n 15}.card = 6 :=
by sorry

end factors_of_180_l769_769772


namespace number_of_students_with_type_B_l769_769515

theorem number_of_students_with_type_B
  (total_students : ℕ)
  (students_with_type_A : total_students ≠ 0 ∧ total_students ≠ 0 → 2 * total_students = 90)
  (students_with_type_B : 2 * total_students = 90) :
  2/5 * total_students = 18 :=
by
  sorry

end number_of_students_with_type_B_l769_769515


namespace not_possible_20_odd_rows_15_odd_columns_l769_769654

def odd (n : ℕ) : Prop :=
  n % 2 = 1

def even (n : ℕ) : Prop :=
  n % 2 = 0

theorem not_possible_20_odd_rows_15_odd_columns
  (table : ℕ → ℕ → Prop) -- table representing the presence of crosses
  (n : ℕ) -- number of rows and columns in the square table
  (h_square_table: ∀ i j, table i j → i < n ∧ j < n)
  (odd_rows : ℕ)
  (odd_columns : ℕ)
  (h_odd_rows : odd_rows = 20)
  (h_odd_columns : odd_columns = 15)
  (h_def_odd_row: ∀ r, (∃ m, m < n ∧ odd (finset.card {c | c < n ∧ table r c})) ↔ r < odd_rows)
  (h_def_odd_column: ∀ c, (∃ m, m < n ∧ odd (finset.card {r | r < n ∧ table r c})) ↔ c < odd_columns)
  : false :=
by
  sorry

end not_possible_20_odd_rows_15_odd_columns_l769_769654


namespace vector_dot_product_l769_769903

variables {V : Type*} [inner_product_space ℝ V]
variables (u v w : V)

-- Conditions
def condition1 : Prop := ∥u∥ = 1
def condition2 : Prop := ∥v∥ = 1
def condition3 : Prop := ∥u + v∥ = 2
def condition4 : Prop := w - u - 3 • v = 2 • (u ×ₗ v)

-- Theorem
theorem vector_dot_product (h1 : condition1 u) (h2 : condition2 v) (h3 : condition3 u v) (h4 : condition4 u v w) :
  v ⬝ w = 4 :=
sorry

end vector_dot_product_l769_769903


namespace train_path_alternation_possible_l769_769408

theorem train_path_alternation_possible (curve : ℝ → ℝ × ℝ) (h_curve_closed : curve 0 = curve 1) :
  let regions := {r : set (ℝ × ℝ) | ∃ p ∈ r, ∀ q ∈ r, curve p = curve q → p = q}
  in ∃ (coloring : (ℝ × ℝ) → bool),
    (∀ (r1 r2 : set (ℝ × ℝ)), r1 ∈ regions → r2 ∈ regions → (∃ p ∈ r1, ∃ q ∈ r2, p = q ∧ q = p ∧ r1 ≠ r2) →
      coloring r1 ≠ coloring r2) ∧
    (∀ (p : ℝ), (curve p).fst ∈ regions ∧ (curve p).snd ∈ regions → coloring ((curve p).fst) ≠ coloring ((curve p).snd)) :=
sorry

end train_path_alternation_possible_l769_769408


namespace masha_spheres_base_l769_769932

theorem masha_spheres_base (n T n9 : ℕ) (h1 : T = (n * (n + 1)) / 2)
                           (h2 : 1 / 6 * n * (n + 1) * (n + 2) = 165)
                           (h3 : n = 9)
                           (h4 : T = 45) : n9 = 45 :=
by {
  have h5 : n * (n + 1) * (n + 2) = 990, from sorry,
  have h6 : 45 = (n * (n + 1)) / 2, from sorry,
  exact h6
}

end masha_spheres_base_l769_769932


namespace weight_of_b_l769_769125

theorem weight_of_b (A B C : ℝ)
  (h1 : A + B + C = 129)
  (h2 : A + B = 96)
  (h3 : B + C = 84) : B = 51 := 
by
  sorry

end weight_of_b_l769_769125


namespace sinA_max_value_l769_769321

theorem sinA_max_value (A B C : ℝ) (h : ∠A + ∠B + ∠C = π) 
  (tan_cond : tan A * tan C + tan A * tan B = 5 * tan B * tan C) :
  sin A ≤ (3 * real.sqrt 5) / 7 :=
sorry

end sinA_max_value_l769_769321


namespace campsite_coloring_minimum_colors_l769_769984

-- Define the graph structure and the chromatic number.
theorem campsite_coloring_minimum_colors {G : SimpleGraph (Fin 9)} 
  (h_triangle : ∃ (a b c : Fin 9), G.adj a b ∧ G.adj b c ∧ G.adj c a) : 
  G.chromaticNumber = 3 :=
sorry

end campsite_coloring_minimum_colors_l769_769984


namespace pyramid_volume_theorem_l769_769080

noncomputable def volume_of_regular_square_pyramid : ℝ := 
  let side_edge_length := 2 * Real.sqrt 3
  let angle := Real.pi / 3 -- 60 degrees in radians
  let height := side_edge_length * Real.sin angle
  let base_area := 2 * (1 / 2) * side_edge_length * Real.sqrt 3
  (1 / 3) * base_area * height

theorem pyramid_volume_theorem :
  let side_edge_length := 2 * Real.sqrt 3
  let angle := Real.pi / 3 -- 60 degrees in radians
  let height := side_edge_length * Real.sin angle
  let base_area := 2 * (1 / 2) * (side_edge_length * Real.sqrt 3)
  (1 / 3) * base_area * height = 6 := 
by
  sorry

end pyramid_volume_theorem_l769_769080


namespace PetyaCanAlwaysForceDifferenceOfRoots2014_l769_769014

noncomputable def canPetyaForceDifferenceOfRoots2014 : Prop :=
  ∀ (v1 v2 : ℚ) (vasyachooses : (ℚ → ℚ) → Prop), (∃ p q : ℚ, vasyachooses (λ _: ℚ, _) ∧ vasyachooses (λ _: ℚ, _)) →
  ∃ (a b c : ℚ), 
    (vasyachooses (λ _: ℚ, a) ∧ vasyachooses (λ _: ℚ, c)) ∨
    (vasyachooses (λ _: ℚ, b) ∧ vasyachooses (λ _: ℚ, c)) ∧
    (∀ x y : ℚ, (x^3 + a*x^2 + b*x + c = 0 → y^3 + a*y^2 + b*y + c = 0 → abs(x - y) = 2014))

theorem PetyaCanAlwaysForceDifferenceOfRoots2014 : canPetyaForceDifferenceOfRoots2014 :=
sorry

end PetyaCanAlwaysForceDifferenceOfRoots2014_l769_769014


namespace bottles_per_case_l769_769151

theorem bottles_per_case (total_bottles : ℕ) (total_cases : ℕ)
  (h1 : total_bottles = 72000) (h2 : total_cases = 8000) :
  total_bottles / total_cases = 9 :=
by
  rw [h1, h2]
  norm_num
  sorry

end bottles_per_case_l769_769151


namespace convention_handshakes_l769_769993

-- Introducing the conditions
def companies : ℕ := 5
def reps_per_company : ℕ := 4
def total_reps : ℕ := companies * reps_per_company
def shakes_per_rep : ℕ := total_reps - 1 - (reps_per_company - 1)
def handshakes : ℕ := (total_reps * shakes_per_rep) / 2

-- Statement of the proof
theorem convention_handshakes : handshakes = 160 :=
by
  sorry  -- Proof is not required in this task.

end convention_handshakes_l769_769993


namespace avg_A_less_avg_B_avg_20_points_is_6_6_l769_769472

noncomputable def scores_A : List ℕ := [4, 5, 5, 6, 6, 7, 7, 8, 8, 9]
noncomputable def scores_B : List ℕ := [2, 5, 6, 6, 7, 7, 7, 8, 9, 10]
noncomputable def variance_A : ℝ := 2.25
noncomputable def variance_B : ℝ := 4.41

theorem avg_A_less_avg_B :
  let avg_A := (scores_A.sum.to_float / scores_A.length) 
  let avg_B := (scores_B.sum.to_float / scores_B.length) 
  avg_A < avg_B := 
by
  sorry

theorem avg_20_points_is_6_6 :
  let avg_A := (scores_A.sum.to_float / scores_A.length) 
  let avg_B := (scores_B.sum.to_float / scores_B.length) 
  let avg_20 := ((avg_A * scores_A.length + avg_B * scores_B.length) / (scores_A.length + scores_B.length))
  avg_20 = 6.6 := 
by
  sorry

end avg_A_less_avg_B_avg_20_points_is_6_6_l769_769472


namespace exterior_angle_theorem_l769_769874

noncomputable def angleABD : ℝ := 154
noncomputable def angleBDC : ℝ := 58

theorem exterior_angle_theorem 
  (ABC_straight_line : ∃ A B C : Type, is_collinear A B C)
  (angleABD_eq : ∠ABD = angleABD)
  (angleBDC_eq : ∠BDC = angleBDC) :
  ∠BCD = 96 :=
by
  sorry

end exterior_angle_theorem_l769_769874


namespace value_of_a_l769_769318

theorem value_of_a (a : ℝ) :
  (∀ x : ℝ, x ∈ Ioo (-1 : ℝ) (2 : ℝ) ↔ |a * x + 2| < 6) → a = -4 :=
by
  intro h
  sorry

end value_of_a_l769_769318


namespace possible_values_of_a_l769_769638

variable (a : ℝ)
def A : Set ℝ := { x | x^2 ≠ 1 }
def B (a : ℝ) : Set ℝ := { x | a * x = 1 }

theorem possible_values_of_a (h : (A ∪ B a) = A) : a = 1 ∨ a = -1 ∨ a = 0 :=
by
  sorry

end possible_values_of_a_l769_769638


namespace first_motorcyclist_laps_per_hour_l769_769095

noncomputable def motorcyclist_laps (x y z : ℝ) (P1 : 0 < x - y) (P2 : 0 < x - z) (P3 : 0 < y - z) : Prop :=
  (4.5 / (x - y) = 4.5) ∧ (4.5 / (x - z) = 4.5 - 0.5) ∧ (3 / (y - z) = 3) → x = 3

theorem first_motorcyclist_laps_per_hour (x y z : ℝ) (P1: 0 < x - y) (P2: 0 < x - z) (P3: 0 < y - z) :
  motorcyclist_laps x y z P1 P2 P3 →
  x = 3 :=
sorry

end first_motorcyclist_laps_per_hour_l769_769095


namespace problem_l769_769339

-- Define the complex number z
def z : ℂ := (2 * complex.I) / (1 + complex.I)

-- Define the conjugate of z
def z_conj : ℂ := complex.conj z

-- Define point A associated with z_conj
def A : ℂ := z_conj

-- Define the conditions for the proof
def fourth_quadrant (a : ℂ) : Prop := a.re > 0 ∧ a.im < 0

def B : ℂ := -A

-- Prove the problem statements based on the conditions
theorem problem (hz : z = (2 * complex.I) / (1 + complex.I)) :
  fourth_quadrant A ∧ B = (-1:ℂ) + complex.I :=
by
  -- Specific details of theorems about conversion from solution steps are omitted.
  sorry

end problem_l769_769339


namespace count_factors_180_multiple_of_15_is_6_l769_769813

-- We define a function to check if 'n' is a factor of 'm'
def is_factor (m n : ℕ) : Prop := n ∣ m

-- We define a function to check if 'n' is a multiple of 'd'
def is_multiple (n d : ℕ) : Prop := d ∣ n

-- We define a function to collect all factors of a number 'n'
def factors (n : ℕ) : list ℕ := 
  -- We use list.filter to keep elements that satisfy is_factor n
  list.filter (λ x, is_factor n x) (list.range (n + 1))

-- We count the number of elements in a list that satisfy a predicate
def count_satisfying {α : Type*} (pred : α → Prop) [decidable_pred pred] (l : list α) : ℕ :=
  list.length (list.filter pred l)

-- Noncomputable definition to specify that counting number
noncomputable def count_positive_factors_of_180_multiple_of_15 : ℕ :=
  count_satisfying (λ x, is_multiple x 15) (factors 180)

-- We state the theorem equivalence to the proof problem
theorem count_factors_180_multiple_of_15_is_6 : count_positive_factors_of_180_multiple_of_15 = 6 :=
  sorry


end count_factors_180_multiple_of_15_is_6_l769_769813


namespace min_positive_period_of_tan_l769_769976

def f (x : ℝ) : ℝ := Real.tan (3 * x + Real.pi / 4)

theorem min_positive_period_of_tan (x : ℝ) : 
  ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ ∀ T' > 0, (∀ x : ℝ, f (x + T') = f x) → T <= T' := 
  ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ (∀ T' ∈ set.Ioo 0 T, ∃ x : ℝ, f (x + T') ≠ f x) := 
-- T = π / 3
sorry

end min_positive_period_of_tan_l769_769976


namespace Michael_needs_more_money_l769_769386

def money_Michael_has : ℕ := 50
def cake_cost : ℕ := 20
def bouquet_cost : ℕ := 36
def balloons_cost : ℕ := 5

def total_cost : ℕ := cake_cost + bouquet_cost + balloons_cost
def money_needed : ℕ := total_cost - money_Michael_has

theorem Michael_needs_more_money : money_needed = 11 :=
by
  sorry

end Michael_needs_more_money_l769_769386


namespace pairs_mod_p_identical_set_l769_769352

theorem pairs_mod_p_identical_set (p : ℕ) (prime_p : nat.prime p) (a b : fin (p^2) → ℤ)
  (h : ∀ (ξ η : ℤ), (¬ (ξ % p = 0 ∧ η % p = 0)) →
       (∃ (s : fin (p^2) → ℤ), function.surjective (λ k, (a k * ξ + b k * eta) % p) ∧
       (∀ m : fin (p), ∃ n, multiset.count n (fin (p^2).val.map (λ k, (a k * ξ + b k * eta) % p) = p)) : 
       (fin (p^2) → ℤ) → Prop) :
  { k : fin (p^2) | (a k % p, b k % p) } = { z | ∃ (u v : fin p), z = (u, v) } :=
sorry

end pairs_mod_p_identical_set_l769_769352


namespace fraction_of_trumpets_in_band_l769_769002

theorem fraction_of_trumpets_in_band
  (total_flutes : ℕ := 20)
  (flute_fraction_in_band : ℝ := 0.80)
  (total_clarinets : ℕ := 30)
  (clarinet_fraction_in_band : ℝ := 0.5)
  (total_pianists : ℕ := 20)
  (pianist_fraction_in_band : ℝ := 0.10)
  (total_band_members : ℕ := 53)
  (total_trumpets : ℕ := 60) :
  let num_flutes := flute_fraction_in_band * total_flutes,
      num_clarinets := clarinet_fraction_in_band * total_clarinets,
      num_pianists := pianist_fraction_in_band * total_pianists,
      num_non_trumpet_players := num_flutes + num_clarinets + num_pianists,
      num_trumpets_in_band := total_band_members - num_non_trumpet_players,
      fraction_trumpets_in_band := num_trumpets_in_band / total_trumpets
  in fraction_trumpets_in_band = 1 / 3 :=
by
  sorry

end fraction_of_trumpets_in_band_l769_769002


namespace smallest_coins_l769_769140

theorem smallest_coins (n : ℕ) (n_min : ℕ) (h1 : ∃ n, n % 8 = 5 ∧ n % 7 = 4 ∧ n = 53) (h2 : n_min = n):
  (n_min ≡ 5 [MOD 8]) ∧ (n_min ≡ 4 [MOD 7]) ∧ (n_min = 53) ∧ (53 % 9 = 8) :=
by
  sorry

end smallest_coins_l769_769140


namespace crosswalk_distance_l769_769546

noncomputable def distance_between_stripes (area : ℝ) (side : ℝ) (angle : ℝ) : ℝ :=
  (2 * area) / (side * Real.cos angle)

theorem crosswalk_distance
  (curb_distance : ℝ) (crosswalk_angle_deg : ℝ) (curb_length : ℝ) (stripe_length : ℝ) 
  (h₁ : curb_distance = 50)
  (h₂ : crosswalk_angle_deg = 30)
  (h₃ : curb_length = 20)
  (h₄ : stripe_length = 60) :
  abs (distance_between_stripes (curb_length * curb_distance) stripe_length (Real.pi * crosswalk_angle_deg / 180) - 19.24) < 0.01 := 
by
  sorry

end crosswalk_distance_l769_769546


namespace count_factors_180_multiple_of_15_is_6_l769_769809

-- We define a function to check if 'n' is a factor of 'm'
def is_factor (m n : ℕ) : Prop := n ∣ m

-- We define a function to check if 'n' is a multiple of 'd'
def is_multiple (n d : ℕ) : Prop := d ∣ n

-- We define a function to collect all factors of a number 'n'
def factors (n : ℕ) : list ℕ := 
  -- We use list.filter to keep elements that satisfy is_factor n
  list.filter (λ x, is_factor n x) (list.range (n + 1))

-- We count the number of elements in a list that satisfy a predicate
def count_satisfying {α : Type*} (pred : α → Prop) [decidable_pred pred] (l : list α) : ℕ :=
  list.length (list.filter pred l)

-- Noncomputable definition to specify that counting number
noncomputable def count_positive_factors_of_180_multiple_of_15 : ℕ :=
  count_satisfying (λ x, is_multiple x 15) (factors 180)

-- We state the theorem equivalence to the proof problem
theorem count_factors_180_multiple_of_15_is_6 : count_positive_factors_of_180_multiple_of_15 = 6 :=
  sorry


end count_factors_180_multiple_of_15_is_6_l769_769809


namespace infinitely_matching_fractions_l769_769032

theorem infinitely_matching_fractions :
  ∃ (d_i d_j : ℕ → ℕ) (i j: ℕ), 
  i ≠ j ∧ d_i ≠ d_j ∧ (∀ k, ∃ inf_set : set ℕ, {n | d_i n = d_j n} = inf_set ∧ inf_set.infinite ) :=
by 
  -- Assume d_i and d_j are sequences representing the decimal digits of two fractions 
  -- out of 11 different infinite decimal fractions where i and j are indices.
  sorry

end infinitely_matching_fractions_l769_769032


namespace y_squared_plus_three_y_is_perfect_square_l769_769627

theorem y_squared_plus_three_y_is_perfect_square (y : ℕ) :
  (∃ x : ℕ, y^2 + 3^y = x^2) ↔ y = 1 ∨ y = 3 := 
by
  sorry

end y_squared_plus_three_y_is_perfect_square_l769_769627


namespace factors_of_180_multiples_of_15_l769_769787

theorem factors_of_180_multiples_of_15 :
  (finset.filter (λ x : ℕ, x % 15 = 0) (finset.filter (λ x : ℕ, 180 % x = 0) (finset.range (180 + 1)))).card = 6 :=
sorry

end factors_of_180_multiples_of_15_l769_769787


namespace sum_first_10_terms_arithmetic_geometric_sequence_l769_769676

theorem sum_first_10_terms_arithmetic_geometric_sequence :
  ∃ (a b : ℕ → ℕ), 
  (a 1 = 1) ∧
  (c 1 = 3 ∧ c 2 = 6 ∧ c 3 = 11) ∧ 
  (∀ n, c n = a n + b n) ∧ 
  (∀ n, a n = n ∧ b n = 2^n) →
  (∑ i in range 10, c (i + 1)) = 2099 := 
by
  sorry

end sum_first_10_terms_arithmetic_geometric_sequence_l769_769676


namespace least_x_divisibility_l769_769228

theorem least_x_divisibility :
  ∃ x : ℕ, (x > 0) ∧ ((x^2 + 164) % 3 = 0) ∧ ((x^2 + 164) % 4 = 0) ∧ ((x^2 + 164) % 5 = 0) ∧
  ((x^2 + 164) % 6 = 0) ∧ ((x^2 + 164) % 7 = 0) ∧ ((x^2 + 164) % 8 = 0) ∧ 
  ((x^2 + 164) % 9 = 0) ∧ ((x^2 + 164) % 10 = 0) ∧ ((x^2 + 164) % 11 = 0) ∧ x = 166 → 
  3 = 3 :=
by
  sorry

end least_x_divisibility_l769_769228


namespace count_factors_of_180_multiple_of_15_l769_769720

def is_factor (x n : ℕ) := n % x = 0 -- Definition of a factor
def is_multiple (x k : ℕ) := x % k = 0 -- Definition of a multiple

theorem count_factors_of_180_multiple_of_15 :
  (finset.filter (λ x, is_multiple x 15) (finset.filter (λ x, is_factor x 180) (finset.range (180 + 1)))).card = 6 := by
  sorry

end count_factors_of_180_multiple_of_15_l769_769720


namespace rental_fee_expression_minimum_rental_cost_l769_769144

variable (x : Nat)

def buses_condition (x : Nat) : Prop :=
  0 < x ∧ x < 6

theorem rental_fee_expression (hx : buses_condition x) : 
  ∃ y, y = 2700 - 150 * x :=
by 
  use (2700 - 150 * x)
  exact rfl

theorem minimum_rental_cost : ∃ x y, buses_condition x ∧ x = 2 ∧ y = 2400 ∧ y = 2700 - 150 * x :=
by
  use 2
  use 2400
  have hx : buses_condition 2 := by 
    dsimp [buses_condition]
    exact And.intro (by norm_num) (by norm_num)
  exists hx
  split
  rfl
  exact rfl
  done

end rental_fee_expression_minimum_rental_cost_l769_769144


namespace find_normal_monthly_charge_l769_769380

-- Define the conditions
def normal_monthly_charge (x : ℕ) : Prop :=
  let first_month_charge := x / 3
  let fourth_month_charge := x + 15
  let other_months_charge := 4 * x
  (first_month_charge + fourth_month_charge + other_months_charge = 175)

-- The statement to prove
theorem find_normal_monthly_charge : ∃ x : ℕ, normal_monthly_charge x ∧ x = 30 := by
  sorry

end find_normal_monthly_charge_l769_769380


namespace CardTransformationImpossible_l769_769996

theorem CardTransformationImpossible (a b c d: ℤ)(h1 : a - b = 5 - 19) (h2: c - d = 1 - 1988) :
a ≡ b [MOD 7] → a ≡ b [MOD 7] → a - b ≠ c - d:=
by
  intro h1 h2
  sorry

end CardTransformationImpossible_l769_769996


namespace winning_candidate_votes_l769_769999

theorem winning_candidate_votes  (V W : ℝ) (hW : W = 0.5666666666666664 * V) (hV : V = W + 7636 + 11628) : 
  W = 25216 := 
by 
  sorry

end winning_candidate_votes_l769_769999


namespace factors_of_180_l769_769765

-- Define a predicate to check if a number is a factor of another number
def is_factor (m n : ℕ) : Prop := n % m = 0

-- Define a predicate to check if a number is a multiple of another number
def is_multiple (m n : ℕ) : Prop := m % n = 0

-- Define the problem
theorem factors_of_180 : {n : ℕ // is_factor n 180 ∧ is_multiple n 15}.card = 6 :=
by sorry

end factors_of_180_l769_769765


namespace coloring_impossible_l769_769195

-- Definitions for the problem conditions
def board := fin 5 → fin 5 → fin 4

def is_valid_color (board : board) : Prop :=
  ∀ (r1 r2 c1 c2 : fin 5), r1 ≠ r2 → c1 ≠ c2 →
  (board r1 c1 ≠ board r1 c2 ∧ board r1 c1 ≠ board r2 c1 ∧ board r1 c1 ≠ board r2 c2) ∨
  (board r1 c2 ≠ board r2 c2 ∧ board r2 c1 ≠ board r2 c2 ∧ board r1 c2 ≠ board r2 c1)

-- Formal problem statement in Lean 4
theorem coloring_impossible : ¬ ∃ (b : board), is_valid_color b :=
by {
  sorry,
}

end coloring_impossible_l769_769195


namespace positional_relationship_l769_769692

noncomputable def circle_polar : ℝ → ℝ → Prop :=
  λ ρ θ, sqrt 2 * ρ = 4 * sin (θ + π / 4)
  
def line_parametric : ℝ → ℝ × ℝ :=
  λ t, (3 + t, 1 - 2 * t)

def circle_cartesian : ℝ × ℝ → Prop :=
  λ p, (p.1 - 1) ^ 2 + (p.2 - 1) ^ 2 = 2

def line_standard : ℝ × ℝ → Prop :=
  λ p, 2 * p.1 + p.2 - 7 = 0

def distance_from_center_to_line : ℝ :=
  4 / sqrt 5

def separate : Prop :=
  distance_from_center_to_line > sqrt 2

theorem positional_relationship:
  ∀ (ρ θ t : ℝ),
  (circle_polar ρ θ → circle_cartesian (ρ * cos θ, ρ * sin θ)) ∧
  (line_parametric t = (ρ * cos θ, ρ * sin θ) → line_standard (3 + t, 1 - 2 * t)) ∧
  separate :=
by
  intros ρ θ t
  split
  {
    sorry
  }
  split
  {
    sorry
  }
  {
    sorry
  }

end positional_relationship_l769_769692


namespace solve_equations_l769_769681

theorem solve_equations (x : ℝ) :
  (3 * x^2 = 27 → x = 3 ∨ x = -3) ∧
  (2 * x^2 + x = 55 → x = 5 ∨ x = -5.5) ∧
  (2 * x^2 + 18 = 15 * x → x = 6 ∨ x = 1.5) :=
by
  sorry

end solve_equations_l769_769681


namespace count_factors_180_multiple_of_15_is_6_l769_769810

-- We define a function to check if 'n' is a factor of 'm'
def is_factor (m n : ℕ) : Prop := n ∣ m

-- We define a function to check if 'n' is a multiple of 'd'
def is_multiple (n d : ℕ) : Prop := d ∣ n

-- We define a function to collect all factors of a number 'n'
def factors (n : ℕ) : list ℕ := 
  -- We use list.filter to keep elements that satisfy is_factor n
  list.filter (λ x, is_factor n x) (list.range (n + 1))

-- We count the number of elements in a list that satisfy a predicate
def count_satisfying {α : Type*} (pred : α → Prop) [decidable_pred pred] (l : list α) : ℕ :=
  list.length (list.filter pred l)

-- Noncomputable definition to specify that counting number
noncomputable def count_positive_factors_of_180_multiple_of_15 : ℕ :=
  count_satisfying (λ x, is_multiple x 15) (factors 180)

-- We state the theorem equivalence to the proof problem
theorem count_factors_180_multiple_of_15_is_6 : count_positive_factors_of_180_multiple_of_15 = 6 :=
  sorry


end count_factors_180_multiple_of_15_is_6_l769_769810


namespace marilyn_bottle_caps_start_l769_769925

-- Definitions based on the conditions
def initial_bottle_caps (X : ℕ) := X  -- Number of bottle caps Marilyn started with
def shared_bottle_caps := 36           -- Number of bottle caps shared with Nancy
def remaining_bottle_caps := 15        -- Number of bottle caps left after sharing

-- Theorem statement: Given the conditions, show that Marilyn started with 51 bottle caps
theorem marilyn_bottle_caps_start (X : ℕ) 
  (h1 : initial_bottle_caps X - shared_bottle_caps = remaining_bottle_caps) : 
  X = 51 := 
sorry  -- Proof omitted

end marilyn_bottle_caps_start_l769_769925


namespace masha_spheres_base_l769_769931

theorem masha_spheres_base (n T n9 : ℕ) (h1 : T = (n * (n + 1)) / 2)
                           (h2 : 1 / 6 * n * (n + 1) * (n + 2) = 165)
                           (h3 : n = 9)
                           (h4 : T = 45) : n9 = 45 :=
by {
  have h5 : n * (n + 1) * (n + 2) = 990, from sorry,
  have h6 : 45 = (n * (n + 1)) / 2, from sorry,
  exact h6
}

end masha_spheres_base_l769_769931


namespace prime_sum_42_l769_769832

theorem prime_sum_42 (p q r s t u : ℕ)
  (hp : nat.prime p) (hq : nat.prime q) (hr : nat.prime r)
  (h_eq : 1998 = p^s * q^t * r^u) :
  p + q + r = 42 :=
sorry

end prime_sum_42_l769_769832


namespace shaded_area_l769_769875

/-- Calculate total shaded area that is exactly covered by three of intersecting six circles -/
theorem shaded_area (r : ℝ) (h_r : r = 5) :
  let sector_area := (1/4) * Real.pi * r^2 -- area of quarter-circle
  let triangle_area := (sqrt 3 / 4) * r^2 -- area of equilateral triangle
  let region_area := (1/3 * triangle_area) - sector_area -- net area of one region
  6 * region_area = 37.5 * Real.pi - 25 * (Real.sqrt 3) / 2 := 
by
  sorry

end shaded_area_l769_769875


namespace ralph_did_not_hit_110_balls_l769_769038

def tennis_problem : Prop :=
  ∀ (total_balls first_batch second_batch hit_first hit_second not_hit_first not_hit_second not_hit_total : ℕ),
  total_balls = 175 →
  first_batch = 100 →
  second_batch = 75 →
  hit_first = 2/5 * first_batch →
  hit_second = 1/3 * second_batch →
  not_hit_first = first_batch - hit_first →
  not_hit_second = second_batch - hit_second →
  not_hit_total = not_hit_first + not_hit_second →
  not_hit_total = 110

theorem ralph_did_not_hit_110_balls : tennis_problem := by
  unfold tennis_problem
  intros
  sorry

end ralph_did_not_hit_110_balls_l769_769038


namespace sales_volume_decrease_may_sales_prediction_l769_769323

theorem sales_volume_decrease 
  (jan_sales mar_sales : ℝ) 
  (sqrt_approx : ∀ (y : ℝ), y = 0.9 → real.sqrt y ≈ 0.95)
  (percentage_decrease : ℝ)
  (sales_feb_eq : jan_sales * (1 - percentage_decrease) = 5400)
  (sales_feb_approx : (1 - percentage_decrease) = 0.95) :
  percentage_decrease = 0.05 :=
sorry

theorem may_sales_prediction
  (jan_sales mar_sales : ℝ)
  (sqrt_approx : ∀ (y : ℝ), y = 0.9 → real.sqrt y ≈ 0.95)
  (percentage_decrease : ℝ)
  (sales_feb_eq : jan_sales * (1 - percentage_decrease) = 5400)
  (sales_feb_approx : (1 - percentage_decrease) = 0.95) 
  (apr_sales : mar_sales * 0.95 = 5130) 
  (may_sales : mar_sales * 0.95 ^ 2 = 4873.5) :
  4873.5 ≥ 4500 :=
sorry

end sales_volume_decrease_may_sales_prediction_l769_769323


namespace total_balls_estimation_l769_769136

theorem total_balls_estimation
  (n : ℕ)  -- Let n be the total number of balls in the bag
  (yellow_balls : ℕ)  -- Let yellow_balls be the number of yellow balls
  (frequency : ℝ)  -- Let frequency be the stabilized frequency of drawing a yellow ball
  (h1 : yellow_balls = 6)
  (h2 : frequency = 0.3)
  (h3 : (yellow_balls : ℝ) / (n : ℝ) = frequency) :
  n = 20 :=
by
  sorry

end total_balls_estimation_l769_769136


namespace sum_of_fourth_powers_is_three_times_square_l769_769958

theorem sum_of_fourth_powers_is_three_times_square (n : ℤ) (h : n ≠ 0) :
  (n - 1)^4 + n^4 + (n + 1)^4 + 10 = 3 * (n^2 + 2)^2 :=
by
  sorry

end sum_of_fourth_powers_is_three_times_square_l769_769958


namespace geom_seq_inequality_l769_769901

-- Define S_n as a.sum of the first n terms of a geometric sequence with ratio q and first term a_1
noncomputable def S (a_1 : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
if h : q = 1 then (n + 1) * a_1 else a_1 * (1 - q ^ (n + 1)) / (1 - q)

-- Define a_n for geometric sequence
noncomputable def a_seq (a_1 : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
a_1 * q ^ n

-- The main theorem to prove
theorem geom_seq_inequality (a_1 : ℝ) (q : ℝ) (n : ℕ) (hq_pos : 0 < q) :
  S a_1 q (n + 1) * a_seq a_1 q n > S a_1 q n * a_seq a_1 q (n + 1) :=
by {
  sorry -- Placeholder for actual proof
}

end geom_seq_inequality_l769_769901


namespace count_positive_factors_of_180_multiple_of_15_l769_769728

statement: ℕ
statement := sorry

theorem count_positive_factors_of_180_multiple_of_15 :
  let factors_of_180 := {d ∈ (finset.range 181).filter (λ x, 180 % x = 0)}
  let multiples_of_15 := (finset.range 181).filter (λ x, x % 15 = 0)
  let desired_factors := factors_of_180 ∩ multiples_of_15
  desired_factors.card = 6 :=
by
  sorry

end count_positive_factors_of_180_multiple_of_15_l769_769728


namespace count_factors_180_multiple_of_15_is_6_l769_769815

-- We define a function to check if 'n' is a factor of 'm'
def is_factor (m n : ℕ) : Prop := n ∣ m

-- We define a function to check if 'n' is a multiple of 'd'
def is_multiple (n d : ℕ) : Prop := d ∣ n

-- We define a function to collect all factors of a number 'n'
def factors (n : ℕ) : list ℕ := 
  -- We use list.filter to keep elements that satisfy is_factor n
  list.filter (λ x, is_factor n x) (list.range (n + 1))

-- We count the number of elements in a list that satisfy a predicate
def count_satisfying {α : Type*} (pred : α → Prop) [decidable_pred pred] (l : list α) : ℕ :=
  list.length (list.filter pred l)

-- Noncomputable definition to specify that counting number
noncomputable def count_positive_factors_of_180_multiple_of_15 : ℕ :=
  count_satisfying (λ x, is_multiple x 15) (factors 180)

-- We state the theorem equivalence to the proof problem
theorem count_factors_180_multiple_of_15_is_6 : count_positive_factors_of_180_multiple_of_15 = 6 :=
  sorry


end count_factors_180_multiple_of_15_is_6_l769_769815


namespace square_area_PS_l769_769470

noncomputable def area_of_square_on_PS : ℕ :=
  sorry

theorem square_area_PS (PQ QR RS PR PS : ℝ)
  (h1 : PQ ^ 2 = 25)
  (h2 : QR ^ 2 = 49)
  (h3 : RS ^ 2 = 64)
  (h4 : PQ^2 + QR^2 = PR^2)
  (h5 : PR^2 + RS^2 = PS^2) :
  PS^2 = 138 :=
by
  -- proof skipping
  sorry


end square_area_PS_l769_769470


namespace train_speed_correct_l769_769557

noncomputable def train_speed (train_length : ℝ) (bridge_length : ℝ) (time_to_cross : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length in
  let speed_m_per_s := total_distance / time_to_cross in
  speed_m_per_s * 3.6

theorem train_speed_correct (train_length : ℝ) (bridge_length : ℝ) (time_to_cross : ℝ) :
  train_length = 100 ∧ bridge_length = 80 ∧ time_to_cross = 10.799136069114471 →
  train_speed train_length bridge_length time_to_cross = 60.0048 :=
by
  intro h
  cases h with h1 h'
  cases h' with h2 h3
  -- We have the conditions in h1, h2, and h3
  have h_total_distance : total_distance = 180 := by sorry
  have h_speed_m_per_s : speed_m_per_s = 16.668 := by sorry
  have h_speed_km_per_h : 16.668 * 3.6 = 60.0048 := by sorry
  rw [h1, h2, h3, h_total_distance, h_speed_m_per_s, h_speed_km_per_h]
  rfl
  sorry

end train_speed_correct_l769_769557


namespace count_valid_N_under_500_l769_769297

def hasSolution (N : ℕ) (x : ℝ) : Prop :=
  N = x ^ (Real.floor x)

def validN (N : ℕ) : Prop :=
  ∃ x : ℝ, hasSolution N x

theorem count_valid_N_under_500 : 
  let N_set := {N : ℕ | N < 500 ∧ validN N}
  N_set.card = 287 := sorry

end count_valid_N_under_500_l769_769297


namespace sphere_radius_l769_769451

theorem sphere_radius (r_circle d_sphere_plane : ℝ) 
  (h1 : r_circle = sqrt 2) 
  (h2 : d_sphere_plane = 1) : 
  sqrt (r_circle^2 + d_sphere_plane^2) = sqrt 3 :=
by 
  rw [h1, h2]
  simp [pow_two]
  norm_num

end sphere_radius_l769_769451


namespace students_married_is_30_percent_l769_769861
open Real

def total_students : ℝ := sorry
def percent_male : ℝ := 0.70
def percent_female : ℝ := 0.30
def fraction_married_male : ℝ := 1 / 7
def fraction_single_female : ℝ := 0.3333333333333333
def fraction_married_female : ℝ := 2 / 3

theorem students_married_is_30_percent :
  (fraction_married_male * (percent_male * total_students) + fraction_married_female * (percent_female * total_students)) / total_students = 0.30 :=
by simp [fraction_married_male, percent_male, fraction_married_female, percent_female]; sorry

end students_married_is_30_percent_l769_769861


namespace mary_groceries_fitting_l769_769003

theorem mary_groceries_fitting :
  (∀ bags wt_green wt_milk wt_carrots wt_apples wt_bread wt_rice,
    bags = 2 →
    wt_green = 4 →
    wt_milk = 6 →
    wt_carrots = 2 * wt_green →
    wt_apples = 3 →
    wt_bread = 1 →
    wt_rice = 5 →
    (wt_green + wt_milk + wt_carrots + wt_apples + wt_bread + wt_rice = 27) →
    (∀ b, b < 20 →
      (b = 6 + 5 ∨ b = 22 - 11) →
      (20 - b = 9))) :=
by
  intros bags wt_green wt_milk wt_carrots wt_apples wt_bread wt_rice h_bags h_green h_milk h_carrots h_apples h_bread h_rice h_total h_b
  sorry

end mary_groceries_fitting_l769_769003


namespace trapezoid_area_correct_l769_769190

-- Define the given dimensions of the trapezoid
def a : ℝ := 16  -- length of one parallel side in cm
def b : ℝ := 44  -- length of the other parallel side in cm
def c : ℝ := 17  -- length of one non-parallel side in cm
def d : ℝ := 25  -- length of the other non-parallel side in cm

-- Define the expected area of the trapezoid
def expectedArea : ℝ := 450  -- expected area in cm²

-- The theorem to prove that the area calculation is correct
theorem trapezoid_area_correct : 
  let h := Real.sqrt (d ^ 2 - (let y := (c^2 - (Real.sqrt (b^2 - ((a+c-b)^2) * d))) / 2) ^ 2) 
  in ((a + b) / 2) * h = expectedArea := sorry

end trapezoid_area_correct_l769_769190


namespace no_carry_pairs_count_l769_769240

-- Define the digit extraction function for an integer
def digits (n : ℕ) : ℕ × ℕ × ℕ × ℕ :=
  (n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10)

-- Define a predicate to check if no carrying is required for addition and subtraction
def no_carry_pair (n1 n2 : ℕ) : Prop :=
  let ⟨d3, d2, d1, d0⟩ := digits n1 in
  let ⟨_, _, _, d0'⟩ := digits n2 in
  d0 ≠ 9 ∧ d0 ≠ 0 ∧ d0' = d0 + 1

-- Define the range of focus
def in_range (n : ℕ) : Prop := 500 ≤ n ∧ n ≤ 999

-- Define the count of pairs
def count_no_carry_pairs : ℕ :=
  ((500 to 999).filter (λ n, in_range n ∧ no_carry_pair n (n + 1))).length

-- The theorem statement
theorem no_carry_pairs_count :
  count_no_carry_pairs = 8000 := by
  sorry

end no_carry_pairs_count_l769_769240


namespace parabola_sum_coefficients_l769_769159

theorem parabola_sum_coefficients :
  ∃ (a b c : ℤ), 
    (∀ x : ℝ, (x = 0 → a * (x^2) + b * x + c = 1)) ∧
    (∀ x : ℝ, (x = 2 → a * (x^2) + b * x + c = 9)) ∧
    (a * (1^2) + b * 1 + c = 4)
  → a + b + c = 4 :=
by sorry

end parabola_sum_coefficients_l769_769159


namespace value_of_k_l769_769211

theorem value_of_k (x y : ℝ) (t : ℝ) (k : ℝ) : 
  (x + t * y + 8 = 0) ∧ (5 * x - t * y + 4 = 0) ∧ (3 * x - k * y + 1 = 0) → k = 5 :=
by
  sorry

end value_of_k_l769_769211


namespace sum_of_solutions_l769_769109

-- Define the condition that checks the integer values satisfying the given inequality.
def satisfies_condition (x : ℤ) : Prop :=
  4 < (x - 2) ^ 2 ∧ (x - 2) ^ 2 < 36

-- State the main theorem
theorem sum_of_solutions : 
  ∑ x in Finset.filter satisfies_condition (Finset.Icc (-100) 100), x = 16 := 
sorry


end sum_of_solutions_l769_769109


namespace sum_of_angles_in_triangle_sum_of_angles_in_polygon_exponential_equation_logarithmic_equation_l769_769878

-- 1. Sum of the interior angles in a triangle is 180 degrees.
theorem sum_of_angles_in_triangle : ∀ a : ℕ, (∀ x y z : ℕ, x + y + z = 180) → a = 180 := by
  intros a h
  have : a = 180 := sorry
  exact this

-- 2. Sum of interior angles of a regular b-sided polygon is 1080 degrees.
theorem sum_of_angles_in_polygon : ∀ b : ℕ, ((b - 2) * 180 = 1080) → b = 8 := by
  intros b h
  have : b = 8 := sorry
  exact this

-- 3. Exponential equation involving b.
theorem exponential_equation : ∀ p b : ℕ, (8 ^ b = p ^ 21) ∧ (b = 8) → p = 2 := by
  intros p b h
  have : p = 2 := sorry
  exact this

-- 4. Logarithmic equation involving p.
theorem logarithmic_equation : ∀ q p : ℕ, (p = Real.log 81 / Real.log q) ∧ (p = 2) → q = 9 := by
  intros q p h
  have : q = 9 := sorry
  exact this

end sum_of_angles_in_triangle_sum_of_angles_in_polygon_exponential_equation_logarithmic_equation_l769_769878


namespace b_range_monotonic_increasing_l769_769849

theorem b_range_monotonic_increasing (b : ℝ) :
  (∀ x ∈ set.Icc 1 real.exp, deriv (λ x, (x - b) * real.log x) x ≥ 0) ↔ b ∈ set.Iic 1 :=
by
  sorry

end b_range_monotonic_increasing_l769_769849


namespace expected_students_playing_games_l769_769148

theorem expected_students_playing_games (n m : ℕ) (h_n : n = 5) (h_m : m = 6) :
  (∑ i in finset.range m, if i = 0 then (4/5 : ℚ) else if i = 1 then (3/4 : ℚ) else if i = 2 then (2/3 : ℚ) else if i = 3 then (1/2 : ℚ) else 0)
  * (n : ℚ) = 163 / 10 := sorry

end expected_students_playing_games_l769_769148


namespace sum_of_range_and_even_count_eq_641_l769_769320

theorem sum_of_range_and_even_count_eq_641 (n m : ℤ) :
  let x := (n + m) * (m - n + 1) / 2 in
  let y := if (n % 2 = 0 ∧ m % 2 = 0) ∨ (n % 2 ≠ 0 ∧ m % 2 ≠ 0)
           then (m - n) / 2 + 1
           else (m - n + 1) / 2 in
  x + y = 641 :=
sorry

end sum_of_range_and_even_count_eq_641_l769_769320


namespace num_factors_of_180_multiple_of_15_l769_769733

def is_factor (n m : Nat) : Prop := m % n = 0
def is_multiple_of_15 (n : Nat) : Prop := is_factor 15 n

theorem num_factors_of_180_multiple_of_15 : 
  (Nat.filter (fun n => is_factor n 180 ∧ is_multiple_of_15 n) (Nat.range (180 + 1))).length = 6 :=
sorry

end num_factors_of_180_multiple_of_15_l769_769733


namespace partition_possible_l769_769396

-- Define a polyhedron with integer edge lengths as a structure
structure Polyhedron where
  edges : ℕ  -- The edge length of the polyhedron.

-- Define the partitioning property and the condition of not having ten polyhedra with the same edge lengths
def partition_with_no_ten_identical (P : Type) [Polyhedron P] : Prop :=
  -- Placeholder for the property that ensures space can be partitioned
  -- into regular octahedrons and tetrahedrons with integer edge lengths,
  -- and not having ten polyhedra with the same edge lengths.
  sorry

-- The main theorem statement
theorem partition_possible : ∃ P : Type, [Polyhedron P] → partition_with_no_ten_identical P :=
sorry

end partition_possible_l769_769396


namespace cost_per_can_l769_769049

/-- 
Soft drinks are on sale at the grocery store for 2.99 dollars for a 12 pack. 
If a pack contains 12 cans, then what is the cost per can of soft drink?
-/
theorem cost_per_can (total_cost : ℝ) (num_cans : ℕ) (h : total_cost = 2.99) (h2 : num_cans = 12) : 
  (total_cost / num_cans).round(2) = 0.25 :=
by
  sorry

end cost_per_can_l769_769049


namespace handshakes_at_gathering_l769_769185

-- Define a function that calculates the total number of handshakes
def total_handshakes (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2

-- Prove that the total handshakes with the conditions provided is 405
theorem handshakes_at_gathering (n : ℕ) (couples : ℕ) (h1 : n = 30) (h2 : couples = 15) :
  total_handshakes n - couples = 405 :=
by {
  -- Given conditions
  rw [h1, h2],
  -- Simplify the total handshakes: total_handshakes 30 - 15
  unfold total_handshakes,
  norm_num,
  -- 420 - 15 = 405
  exact rfl
}

end handshakes_at_gathering_l769_769185


namespace Sasha_can_write_2011_l769_769044

theorem Sasha_can_write_2011 (N : ℕ) (hN : N > 1) : 
    ∃ (s : ℕ → ℕ), (s 0 = N) ∧ (∃ n, s n = 2011) ∧ 
    (∀ k, ∃ d, d > 1 ∧ (s (k + 1) = s k + d ∨ s (k + 1) = s k - d)) :=
sorry

end Sasha_can_write_2011_l769_769044


namespace three_digit_N_with_perfect_square_difference_l769_769612

theorem three_digit_N_with_perfect_square_difference :
  ∃ (Nset : Finset ℕ), 
  (∀ N ∈ Nset, (∃ a b c : ℕ, N = 100 * a + 10 * b + c ∧ a ≠ 0 ∧ 100 * a + 10 * b + c < 1000)) ∧ 
  (∀ N ∈ Nset, ∃ k : ℕ, ∃ a c : ℕ, k^2 = 99 * (abs (a - c) ∧ 
    N = 100 * a + 10 * b + c ∧ k^2 = 99 * (abs (a - c)))) ∧ 
  Nset.card = 28 :=
begin
  sorry
end

end three_digit_N_with_perfect_square_difference_l769_769612


namespace circumscribed_sphere_radius_l769_769981

theorem circumscribed_sphere_radius (a : ℝ) (h : real) 
  (h_proof : h = 1/2) :
  radius = 2 * a * real.sqrt 2 / real.sqrt 7 := 
sorry

end circumscribed_sphere_radius_l769_769981


namespace count_factors_of_180_multiple_of_15_l769_769784

def is_factor (n k : ℕ) : Prop := n % k = 0

def is_multiple_of (k m : ℕ) : Prop := k % m = 0

theorem count_factors_of_180_multiple_of_15 : 
  (finset.filter (λ f, is_factor 180 f ∧ is_multiple_of f 15) (finset.range (181))).card = 6 := 
by 
sorry

end count_factors_of_180_multiple_of_15_l769_769784


namespace necessary_but_not_sufficient_condition_l769_769130

theorem necessary_but_not_sufficient_condition : 
  ∀ x : ℝ, (sin x = sqrt 3 / 2 → x = π / 3) ∧ ¬(x = π / 3 → sin x = sqrt 3 / 2) :=
by 
  sorry

end necessary_but_not_sufficient_condition_l769_769130


namespace pizza_eaten_after_six_trips_l769_769115

theorem pizza_eaten_after_six_trips:
  (∑ i in Finset.range 6, (1/3) * (2/3)^i) = 665/729 :=
by
  sorry

end pizza_eaten_after_six_trips_l769_769115


namespace total_skips_is_33_l769_769604

theorem total_skips_is_33 {
  let skips_5 := 8,
  ∃ skips_4 skips_3 skips_2 skips_1 : ℕ,
  (skips_5 = skips_4 + 1) ∧
  (skips_4 = skips_3 - 3) ∧
  (skips_3 = skips_2 * 2) ∧
  (skips_2 = skips_1 + 2) ∧
  (skips_1 + skips_2 + skips_3 + skips_4 + skips_5 = 33) 
} sorry

end total_skips_is_33_l769_769604


namespace largest_x_solution_l769_769224

noncomputable def fractional_part (x : ℝ) : ℝ := x - floor(x)

theorem largest_x_solution : ∃ x : ℝ, ⌊x⌋ = 7 + 50 * fractional_part x ∧ 0 ≤ fractional_part x ∧ fractional_part x < 1 ∧ x = 56.98 :=
by
  sorry

end largest_x_solution_l769_769224


namespace starting_lineups_count_l769_769581

-- Definitions from the conditions
open_locale big_operators

def total_players : ℕ := 12
def starting_lineup_size : ℕ := 5
def guaranteed_players : ℕ := 2
def remaining_players : ℕ := total_players - guaranteed_players 
def spots_to_fill : ℕ := starting_lineup_size - guaranteed_players

-- The main proof problem
theorem starting_lineups_count : ∃ k: ℕ, k = (nat.choose remaining_players spots_to_fill) := 120 := 
by
  sorry

end starting_lineups_count_l769_769581


namespace average_salary_l769_769410

theorem average_salary (R S T : ℝ) 
  (h1 : (R + S) / 2 = 4000) 
  (h2 : T = 7000) : 
  (R + S + T) / 3 = 5000 :=
by
  sorry

end average_salary_l769_769410


namespace avg_A_lt_avg_B_combined_avg_eq_6_6_l769_769474

-- Define the scores for A and B
def scores_A := [4, 5, 5, 6, 6, 7, 7, 8, 8, 9]
def scores_B := [2, 5, 6, 6, 7, 7, 7, 8, 9, 10]

-- Define the average score function
def average (scores : List ℚ) : ℚ := (scores.sum : ℚ) / scores.length

-- Define the mean for the combined data
def combined_average : ℚ :=
  (average scores_A * scores_A.length + average scores_B * scores_B.length) / 
  (scores_A.length + scores_B.length)

-- Specify the variances given in the problem
def variance_A := 2.25
def variance_B := 4.41

-- Claim the average score of A is smaller than the average score of B
theorem avg_A_lt_avg_B : average scores_A < average scores_B := by sorry

-- Claim the average score of these 20 data points is 6.6
theorem combined_avg_eq_6_6 : combined_average = 6.6 := by sorry

end avg_A_lt_avg_B_combined_avg_eq_6_6_l769_769474


namespace scientific_notation_conversion_l769_769974

theorem scientific_notation_conversion :
  0.000037 = 3.7 * 10^(-5) :=
by
  sorry

end scientific_notation_conversion_l769_769974


namespace union_A_B_is_R_l769_769699

noncomputable def A : set ℝ := {x : ℝ | x^2 - 16 < 0}
noncomputable def B : set ℝ := {x : ℝ | x^2 - 4x + 3 > 0}

theorem union_A_B_is_R : A ∪ B = set.univ :=
by {
  sorry
}

end union_A_B_is_R_l769_769699


namespace factors_of_180_multiple_of_15_l769_769745

/-- A factor of a number n is a positive integer which divides n without leaving a remainder. -/
def is_factor (n k : ℕ) : Prop :=
k > 0 ∧ n % k = 0

/-- A multiple of a number m is an integer which is divisible by m without leaving a remainder. -/
def is_multiple (k m : ℕ) : Prop :=
k % m = 0

/-- There are 6 factors of 180 that are also multiples of 15. -/
theorem factors_of_180_multiple_of_15 : 
  (Finset.filter (λ k, is_multiple k 15) { x | is_factor 180 x }.to_finset).card = 6 :=
by 
  sorry

end factors_of_180_multiple_of_15_l769_769745


namespace Jack_marbles_after_sharing_l769_769348

def Jack_starting_marbles : ℕ := 1000

def sharing_ratio : ℕ × ℕ := (7, 3)

def percentage_kept (share : ℕ × ℕ) : ℝ :=
  let total_parts := share.1 + share.2
  (share.2 : ℝ) / (total_parts : ℝ)

def marbles_remaining (start : ℕ) (percentage_kept : ℝ) : ℕ :=
  (percentage_kept * (start : ℝ)).toInt1mp

theorem Jack_marbles_after_sharing :
  marbles_remaining Jack_starting_marbles (percentage_kept sharing_ratio) = 300 :=
sorry

end Jack_marbles_after_sharing_l769_769348


namespace fraction_B_approx_one_fourth_l769_769857

noncomputable def frac_B (T : ℕ) : ℚ :=
  3 / 10 - 25 / T

theorem fraction_B_approx_one_fourth (T : ℕ) (hT : T = 500) :
  1 / 5 * T + frac_B T * T + 1 / 2 * T + 25 = T → frac_B T ≈ 1/4 :=
by
  sorry

end fraction_B_approx_one_fourth_l769_769857


namespace count_valid_N_under_500_l769_769296

def hasSolution (N : ℕ) (x : ℝ) : Prop :=
  N = x ^ (Real.floor x)

def validN (N : ℕ) : Prop :=
  ∃ x : ℝ, hasSolution N x

theorem count_valid_N_under_500 : 
  let N_set := {N : ℕ | N < 500 ∧ validN N}
  N_set.card = 287 := sorry

end count_valid_N_under_500_l769_769296


namespace toms_total_out_of_pocket_is_680_l769_769468

namespace HealthCosts

def doctor_visit_cost : ℝ := 300
def cast_cost : ℝ := 200
def initial_insurance_coverage : ℝ := 0.60
def therapy_session_cost : ℝ := 100
def number_of_sessions : ℕ := 8
def therapy_insurance_coverage : ℝ := 0.40

def total_initial_cost : ℝ :=
  doctor_visit_cost + cast_cost

def initial_out_of_pocket : ℝ :=
  total_initial_cost * (1 - initial_insurance_coverage)

def total_therapy_cost : ℝ :=
  therapy_session_cost * number_of_sessions

def therapy_out_of_pocket : ℝ :=
  total_therapy_cost * (1 - therapy_insurance_coverage)

def total_out_of_pocket : ℝ :=
  initial_out_of_pocket + therapy_out_of_pocket

theorem toms_total_out_of_pocket_is_680 :
  total_out_of_pocket = 680 := by
  sorry

end HealthCosts

end toms_total_out_of_pocket_is_680_l769_769468


namespace find_b_l769_769256

theorem find_b (b : ℝ) (x : ℝ) (hx : x^2 + b * x - 45 = 0) (h_root : x = -5) : b = -4 :=
by
  sorry

end find_b_l769_769256


namespace projectile_reaches_35_at_1p57_seconds_l769_769417

theorem projectile_reaches_35_at_1p57_seconds :
  ∀ (t : ℝ), (y : ℝ) (h_eq : y = -4.9 * t^2 + 30 * t)
  (h_initial_velocity : true)  -- Given that the projectile is launched from the ground, we assume this as a given
  (h_conditions : y = 35),
  t = 1.57 :=
by
  sorry

end projectile_reaches_35_at_1p57_seconds_l769_769417


namespace domain_of_y_l769_769067

noncomputable def domain_of_function : Set ℝ := {x : ℝ | -3 < x ∧ x < 2 ∧ x ≠ 1}

theorem domain_of_y :
  ∀ x : ℝ, (2 - x > 0) ∧ (12 + x - x^2 > 0) ∧ (x ≠ 1) ↔ x ∈ domain_of_function :=
begin
  sorry
end

end domain_of_y_l769_769067


namespace f_log2_6_eq_12_l769_769686

-- Define the function f as given
noncomputable def f : ℝ → ℝ
| x => if x >= 3 then 2^x else f (x + 1)

-- The theorem statement we need to prove
theorem f_log2_6_eq_12 : f (Real.log 6 / Real.log 2) = 12 :=
by
  -- Skip the proof
  sorry

end f_log2_6_eq_12_l769_769686


namespace num_factors_of_180_multiple_of_15_l769_769732

def is_factor (n m : Nat) : Prop := m % n = 0
def is_multiple_of_15 (n : Nat) : Prop := is_factor 15 n

theorem num_factors_of_180_multiple_of_15 : 
  (Nat.filter (fun n => is_factor n 180 ∧ is_multiple_of_15 n) (Nat.range (180 + 1))).length = 6 :=
sorry

end num_factors_of_180_multiple_of_15_l769_769732


namespace math_problem_l769_769464

theorem math_problem
    (p q s : ℕ)
    (prime_p : Nat.Prime p)
    (prime_q : Nat.Prime q)
    (prime_s : Nat.Prime s)
    (h1 : p * q = s + 6)
    (h2 : 3 < p)
    (h3 : p < q) :
    p = 5 :=
    sorry

end math_problem_l769_769464


namespace distinct_sums_count_l769_769585

def is_sum_of_three_distinct_members (s : Set ℤ) (n : ℤ) : Prop :=
  ∃ x y z ∈ s, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ n = x + y + z

theorem distinct_sums_count :
  let s := {2, 5, 8, 11, 14, 17, 20} : Set ℤ
  card { n | is_sum_of_three_distinct_members s n } = 13 :=
by
  sorry

end distinct_sums_count_l769_769585


namespace LittleRed_system_of_eqns_l769_769920

theorem LittleRed_system_of_eqns :
  ∃ (x y : ℝ), (2/60) * x + (3/60) * y = 1.5 ∧ x + y = 18 :=
sorry

end LittleRed_system_of_eqns_l769_769920


namespace equation_II_consecutive_integers_l769_769198

theorem equation_II_consecutive_integers :
  ∃ x y z w : ℕ, x + y + z + w = 46 ∧ [x, x+1, x+2, x+3] = [x, y, z, w] :=
by
  sorry

end equation_II_consecutive_integers_l769_769198


namespace factors_of_180_multiples_of_15_l769_769788

theorem factors_of_180_multiples_of_15 :
  (finset.filter (λ x : ℕ, x % 15 = 0) (finset.filter (λ x : ℕ, 180 % x = 0) (finset.range (180 + 1)))).card = 6 :=
sorry

end factors_of_180_multiples_of_15_l769_769788


namespace solve_inequality1_solve_inequality_system_l769_769050

-- Define the first condition inequality
def inequality1 (x : ℝ) : Prop := 
  (2 * x - 1) / 3 - (9 * x + 2) / 6 ≤ 1

-- Theorem for the first inequality proving x >= -2
theorem solve_inequality1 {x : ℝ} (h : inequality1 x) : x ≥ -2 := 
sorry

-- Define the first condition for the system of inequalities
def inequality2 (x : ℝ) : Prop := 
  x - 3 * (x - 2) ≥ 4

-- Define the second condition for the system of inequalities
def inequality3 (x : ℝ) : Prop := 
  (2 * x - 1) / 5 < (x + 1) / 2

-- Theorem for the system of inequalities proving -7 < x ≤ 1
theorem solve_inequality_system {x : ℝ} (h1 : inequality2 x) (h2 : inequality3 x) : -7 < x ∧ x ≤ 1 := 
sorry

end solve_inequality1_solve_inequality_system_l769_769050


namespace positive_factors_of_180_multiple_of_15_count_l769_769758

theorem positive_factors_of_180_multiple_of_15_count : 
  let factorization_180 := 2^2 * 3^2 * 5;
  let is_multiple_of_15 (n : ℕ) := ∃ k1 k2 : ℕ, k1 * 3 = n ∧ k2 * 5 = n;
  let factors_180 := { n : ℕ | n ∣ factorization_180 };
  { n : ℕ | n ∈ factors_180 ∧ is_multiple_of_15 n }.card = 6 :=
by
  sorry

end positive_factors_of_180_multiple_of_15_count_l769_769758


namespace find_angle_Q_l769_769855
-- Import the necessary Lean library

-- Define the conditions and the problem
theorem find_angle_Q (AB AC P Q : ℝ) (h_isosceles : AB = AC) (h_parallel : ∀ (x y : ℝ), x = AB → y = ED → x ∥ y) (h_angle_ABC : ∠ ABC = P) (h_angle_ADE : ∠ ADE = Q) :
  Q = 180 - 2 * P :=
sorry

end find_angle_Q_l769_769855


namespace moles_of_magnesium_l769_769709

-- Assuming the given conditions as hypotheses
variables (Mg CO₂ MgO C : ℕ)

-- Theorem statement
theorem moles_of_magnesium (h1 : 2 * Mg + CO₂ = 2 * MgO + C) 
                           (h2 : MgO = Mg) 
                           (h3 : CO₂ = 1) 
                           : Mg = 2 :=
by sorry  -- Proof to be provided

end moles_of_magnesium_l769_769709


namespace count_factors_of_180_multiple_of_15_l769_769717

def is_factor (x n : ℕ) := n % x = 0 -- Definition of a factor
def is_multiple (x k : ℕ) := x % k = 0 -- Definition of a multiple

theorem count_factors_of_180_multiple_of_15 :
  (finset.filter (λ x, is_multiple x 15) (finset.filter (λ x, is_factor x 180) (finset.range (180 + 1)))).card = 6 := by
  sorry

end count_factors_of_180_multiple_of_15_l769_769717


namespace wieners_age_l769_769004

theorem wieners_age :
  ∃ a : ℕ, (1000 ≤ a^3 ∧ a^3 ≤ 9999) ∧ (100000 ≤ a^4 ∧ a^4 ≤ 999999) ∧ 
           (Multiset.of_list (a^3.digits 10 ++ a^4.digits 10) = Multiset.range 10) ∧ a = 18 :=
by
  sorry

end wieners_age_l769_769004


namespace triangle_inequality_inequality_l769_769354

theorem triangle_inequality_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  real.cbrt ((a^2 + b * c) * (b^2 + c * a) * (c^2 + a * b)) > (a^2 + b^2 + c^2) / 2 := 
sorry

end triangle_inequality_inequality_l769_769354


namespace probability_no_prize_A_probability_700_yuan_given_passed_level_1_l769_769142

section TVStation

variable {Ω : Type} [MeasurableSpace Ω] (P : ProbabilityTheory.ProbabilityMeasure Ω)

-- Pass rates for the levels
def pass_rate_level_1 : ℝ := 0.6
def pass_rate_level_2 : ℝ := 0.5
def pass_rate_level_3 : ℝ := 0.4

-- Event of passing a level
variable (pass_level_1_A pass_level_2_A pass_level_3_A pass_level_1_B pass_level_2_B pass_level_3_B : Set Ω)

-- Independent events conditions
axiom indep_pass_levels_A : Pairwise (ProbabilityTheory.Indep P) [pass_level_1_A, pass_level_2_A, pass_level_3_A]
axiom indep_pass_levels_B : Pairwise (ProbabilityTheory.Indep P) [pass_level_1_B, pass_level_2_B, pass_level_3_B]

-- Pass rates conditions
axiom pass_rate_1_A : P pass_level_1_A = pass_rate_level_1
axiom pass_rate_2_A : P pass_level_2_A = pass_rate_level_2
axiom pass_rate_3_A : P pass_level_3_A = pass_rate_level_3

axiom pass_rate_1_B : P pass_level_1_B = pass_rate_level_1
axiom pass_rate_2_B : P pass_level_2_B = pass_rate_level_2
axiom pass_rate_3_B : P pass_level_3_B = pass_rate_level_3

-- Part 1: Probability that contestant A does not win any prize.
theorem probability_no_prize_A : P ((pass_level_1_Aᶜ ∪ (pass_level_1_A ∩ pass_level_2_Aᶜ))) = 0.7 := by sorry

-- Part 2: Given both contestants passed the first level, probability that their combined prize is 700 yuan.
variable both_passed_level_1 : Set Ω

axiom both_passed_level1_eq : both_passed_level_1 = pass_level_1_A ∩ pass_level_1_B

theorem probability_700_yuan_given_passed_level_1 :
  P[both_passed_level_1, (pass_level_2_A ∩ pass_level_3_A ∩ pass_level_2_B ∪ pass_level_2_B ∩ pass_level_3_B ∩ pass_level_2_A)] = 0.12 := by sorry

end TVStation

end probability_no_prize_A_probability_700_yuan_given_passed_level_1_l769_769142


namespace percentage_change_in_area_of_rectangle_l769_769437

theorem percentage_change_in_area_of_rectangle
  (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  let A1 := L * B
  let A2 := (L / 2) * (3 * B)
  let percentage_change := ((A2 - A1) / A1) * 100
  percentage_change = 50 := by
  let A1 := L * B
  let A2 := (L / 2) * (3 * B)
  let percentage_change := ((A2 - A1) / A1) * 100
  calc percentage_change
      = ((A2 - A1) / A1) * 100 : rfl
  ... = ((((L / 2) * (3 * B)) - (L * B)) / (L * B)) * 100 : by rw [A1, A2]
  ... = ((3 / 2 * A1 - A1) / A1) * 100 : by rw [A2, ←mul_assoc, mul_div_cancel' _ (two_ne_zero : 2 ≠ 0)]
  ... = ((3 / 2 - 1) * A1 / A1) * 100 : by simp only [mul_sub, mul_div_cancel' _ (ne_of_gt hL)]
  ... = (1 / 2) * 100 : by rw [div_mul_cancel' (ne_of_gt (lt_of_lt_of_le zero_lt_one (le_of_lt ((mul_lt_iff_lt_one_left (lt_of_lt_of_le zero_lt_one hL)).mpr (half_pos (lt_of_le_of_ne (le_of_lt (lt_of_lt_of_le (lt_one_mul_self (lt_of_le_of_lt (zero_le_two) (lt_add_one 1))) (lt_of_lt_of_le (two_le_one) (lt_one 2))))))))) so))
  ... = 50 : rfl

end percentage_change_in_area_of_rectangle_l769_769437


namespace hexagon_side_length_l769_769871

theorem hexagon_side_length (a b AF : ℝ) 
  (h1 : AB = 2 ∧ BC = 2 ∧ CD = 2 ∧ DE = 2 ∧ EF = 2)
  (h2 : ∀ (A B C D E : ℝ), angle A 135 ∧ angle B 135 ∧ angle C 135 ∧ angle D 135 ∧ angle E 135) 
  (h3 : angle F = 90)
  (h4 : AF = 4 * sqrt 2)
  (h5 : 0 + 3 * sqrt 2 = AF) :
  a + b = 2 := 
sorry

end hexagon_side_length_l769_769871


namespace radius_of_circle_l769_769337

-- Definitions based on conditions
def center_in_first_quadrant (C : ℝ × ℝ) : Prop :=
  C.1 > 0 ∧ C.2 > 0

def intersects_x_axis (C : ℝ × ℝ) (r : ℝ) : Prop :=
  r = Real.sqrt ((C.1 - 1)^2 + (C.2)^2) ∧ r = Real.sqrt ((C.1 - 3)^2 + (C.2)^2)

def tangent_to_line (C : ℝ × ℝ) (r : ℝ) : Prop :=
  r = abs (C.1 - C.2 + 1) / Real.sqrt 2

-- Main statement
theorem radius_of_circle (C : ℝ × ℝ) (r : ℝ) 
  (h1 : center_in_first_quadrant C)
  (h2 : intersects_x_axis C r)
  (h3 : tangent_to_line C r) : 
  r = Real.sqrt 2 := 
sorry

end radius_of_circle_l769_769337


namespace positive_factors_of_180_multiple_of_15_count_l769_769756

theorem positive_factors_of_180_multiple_of_15_count : 
  let factorization_180 := 2^2 * 3^2 * 5;
  let is_multiple_of_15 (n : ℕ) := ∃ k1 k2 : ℕ, k1 * 3 = n ∧ k2 * 5 = n;
  let factors_180 := { n : ℕ | n ∣ factorization_180 };
  { n : ℕ | n ∈ factors_180 ∧ is_multiple_of_15 n }.card = 6 :=
by
  sorry

end positive_factors_of_180_multiple_of_15_count_l769_769756


namespace num_solutions_abs_sum_eq_20_l769_769937

theorem num_solutions_abs_sum_eq_20 : ∀ (n : ℕ), n > 0 → (∑ x, ∑ y, (|x| + |y| = n) = 4 * n) → (∑ x, ∑ y, (|x| + |y| = 20) = 80) :=
by
  sorry

end num_solutions_abs_sum_eq_20_l769_769937


namespace num_factors_of_180_multiple_of_15_l769_769742

def is_factor (n m : Nat) : Prop := m % n = 0
def is_multiple_of_15 (n : Nat) : Prop := is_factor 15 n

theorem num_factors_of_180_multiple_of_15 : 
  (Nat.filter (fun n => is_factor n 180 ∧ is_multiple_of_15 n) (Nat.range (180 + 1))).length = 6 :=
sorry

end num_factors_of_180_multiple_of_15_l769_769742


namespace exists_sphere_tangent_to_lines_l769_769024

variables
  (A B C D K L M N : Point)
  (AB BC CD DA : Line)
  (sphere : Sphere)

-- Given conditions
def AN_eq_AK : AN = AK := sorry
def BK_eq_BL : BK = BL := sorry
def CL_eq_CM : CL = CM := sorry
def DM_eq_DN : DM = DN := sorry
def sphere_tangent (s : Sphere) (l : Line) : Prop := sorry -- define tangency condition

-- Problem statement
theorem exists_sphere_tangent_to_lines :
  ∃ S : Sphere, 
    sphere_tangent S AB ∧
    sphere_tangent S BC ∧
    sphere_tangent S CD ∧
    sphere_tangent S DA := sorry

end exists_sphere_tangent_to_lines_l769_769024


namespace probability_of_rolling_prime_l769_769406

-- Define the problem conditions
def prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def num_favorable_outcomes : ℕ := { n | prime n ∧ n ≥ 1 ∧ n ≤ 8 }.to_finset.card
def total_possible_outcomes : ℕ := 8

-- Prove the probability of rolling a prime number
theorem probability_of_rolling_prime : num_favorable_outcomes.to_rat / total_possible_outcomes.to_rat = 1 / 2 := 
by
  sorry

end probability_of_rolling_prime_l769_769406


namespace books_sold_on_friday_l769_769887

theorem books_sold_on_friday :
  let initial_stock := 1100 in
  let books_sold_monday := 75 in
  let books_sold_tuesday := 50 in
  let books_sold_wednesday := 64 in
  let books_sold_thursday := 78 in
  let percentage_not_sold := 0.6345 in
  let books_not_sold := (percentage_not_sold * initial_stock).floor in
  let books_sold_monday_to_thursday := books_sold_monday + books_sold_tuesday + books_sold_wednesday + books_sold_thursday in
  let books_sold_friday := initial_stock - (books_not_sold + books_sold_monday_to_thursday) in
  books_sold_friday = 136 :=
by
  sorry

end books_sold_on_friday_l769_769887


namespace mechanic_worked_hours_l769_769928

theorem mechanic_worked_hours (total_spent : ℕ) (cost_per_part : ℕ) (labor_cost_per_minute : ℚ) (parts_needed : ℕ) :
  total_spent = 220 → cost_per_part = 20 → labor_cost_per_minute = 0.5 → parts_needed = 2 →
  (total_spent - cost_per_part * parts_needed) / labor_cost_per_minute / 60 = 6 := by
  -- Proof will be inserted here
  sorry

end mechanic_worked_hours_l769_769928


namespace purely_imaginary_condition_l769_769910

theorem purely_imaginary_condition (x : ℝ) :
  (z : ℂ) → (z = (x^2 - 1 : ℂ) + (x + 1 : ℂ) * Complex.I) →
  (x = 1 ↔ (∃ y : ℂ, z = y * Complex.I)) :=
by
  sorry

end purely_imaginary_condition_l769_769910


namespace count_integers_in_range_l769_769708

theorem count_integers_in_range : 
  ∃ n : ℕ, n = 6 ∧ ∀ x : ℤ, (-7 ≤ 3 * x + 2 ∧ 3 * x + 2 ≤ 8) ↔ (-3 ≤ x ∧ x ≤ 2) :=
by
  sorry

end count_integers_in_range_l769_769708


namespace total_skips_is_33_l769_769602

theorem total_skips_is_33 {
  let skips_5 := 8,
  ∃ skips_4 skips_3 skips_2 skips_1 : ℕ,
  (skips_5 = skips_4 + 1) ∧
  (skips_4 = skips_3 - 3) ∧
  (skips_3 = skips_2 * 2) ∧
  (skips_2 = skips_1 + 2) ∧
  (skips_1 + skips_2 + skips_3 + skips_4 + skips_5 = 33) 
} sorry

end total_skips_is_33_l769_769602


namespace number_of_male_students_in_third_year_l769_769143

-- Conditions
def total_students_third_year : ℕ := 800
def sample_size : ℕ := 40
def female_students_in_sample : ℕ := 11
def male_students_in_sample : ℕ := sample_size - female_students_in_sample

-- Proof problem: Prove the number of male students in the third year is 580 given the conditions.
theorem number_of_male_students_in_third_year :
  let total_male_students := 580 in
  (male_students_in_sample / sample_size : ℚ) = (total_male_students / total_students_third_year) :=
by
  sorry

end number_of_male_students_in_third_year_l769_769143


namespace sqrt_sum_equality_l769_769918

-- Define the conditions
def a : ℤ := 3
def b : ℤ := 2
def n : ℤ := 7

-- State the main equality to be proved under the given conditions
theorem sqrt_sum_equality (h : a^2 - 2 * b^2 = n) : sqrt (↑a + sqrt 2) + sqrt (↑a - sqrt 2) = sqrt (6 + 2 * sqrt 7) := by
  sorry

end sqrt_sum_equality_l769_769918


namespace range_tan4_add_cot2_l769_769183

theorem range_tan4_add_cot2 : 
  ∀ x : ℝ, ∃ y ∈ Set.Ici (2 : ℝ), y = (Real.tan x)^4 + (Real.cot x)^2 := sorry

end range_tan4_add_cot2_l769_769183


namespace circumscribed_circle_ratio_l769_769522

theorem circumscribed_circle_ratio 
  (a b c : ℝ) 
  (h_triangle_sides : a = 8 ∧ b = 15 ∧ c = 17)
  (h_triangle_right : a^2 + b^2 = c^2)
  (h_circumcircle_diameter : let d := c in True)
  (h_triangle_area : let area_triangle := (1 / 2) * a * b in True)
  (h_circle_area : let r := c / 2 in let area_circle := π * r^2 in True)
  (h_remaining_areas : let Z := (area_circle / 2) in let X_plus_Y := Z - area_triangle in True)
  : (X_plus_Y / Z) = 0.471 :=
sorry

end circumscribed_circle_ratio_l769_769522


namespace factors_of_180_multiple_of_15_l769_769752

/-- A factor of a number n is a positive integer which divides n without leaving a remainder. -/
def is_factor (n k : ℕ) : Prop :=
k > 0 ∧ n % k = 0

/-- A multiple of a number m is an integer which is divisible by m without leaving a remainder. -/
def is_multiple (k m : ℕ) : Prop :=
k % m = 0

/-- There are 6 factors of 180 that are also multiples of 15. -/
theorem factors_of_180_multiple_of_15 : 
  (Finset.filter (λ k, is_multiple k 15) { x | is_factor 180 x }.to_finset).card = 6 :=
by 
  sorry

end factors_of_180_multiple_of_15_l769_769752


namespace sum_of_BCD_l769_769825

-- Definition of initial conditions
def div_by_4 (n: Nat) : Prop := n % 4 = 0
def is_prime (n: Nat) : Prop := Nat.Prime n

-- Defining B, C, D and the number itself
variable (B C D: Nat)
variable hDiv : div_by_4 (10 * D + 2)
variable hPrime : is_prime (B + C + D)

-- Stating the theorem: the sum of all possible values of B, C, and D is 61
theorem sum_of_BCD: B + C + D = 61 := by
  sorry

end sum_of_BCD_l769_769825


namespace phone_cost_l769_769935

theorem phone_cost (C : ℝ) (h1 : 0.40 * C + 780 = C) : C = 1300 := by
  sorry

end phone_cost_l769_769935


namespace square_area_from_diagonal_l769_769539

theorem square_area_from_diagonal (d : ℝ) (h : d = 8 * Real.sqrt 2) : 
  ∃ (A : ℝ), A = 64 :=
begin
  use 64,
  sorry
end

end square_area_from_diagonal_l769_769539


namespace correct_operation_l769_769503

theorem correct_operation (a b : ℝ) : (a + b) * (a - b) = a^2 - b^2 := by
  sorry

end correct_operation_l769_769503


namespace hexagon_inequality_l769_769900

variables {Point : Type} [MetricSpace Point]

-- Definitions of points and distances
variables (A B C D E F G H : Point) 
variables (dist : Point → Point → ℝ)
variables (angle : Point → Point → Point → ℝ)

-- Conditions
variables (hABCDEF : ConvexHexagon A B C D E F)
variables (hAB_BC_CD : dist A B = dist B C ∧ dist B C = dist C D)
variables (hDE_EF_FA : dist D E = dist E F ∧ dist E F = dist F A)
variables (hBCD_60 : angle B C D = 60)
variables (hEFA_60 : angle E F A = 60)
variables (hAGB_120 : angle A G B = 120)
variables (hDHE_120 : angle D H E = 120)

-- Objective statement
theorem hexagon_inequality : 
  dist A G + dist G B + dist G H + dist D H + dist H E ≥ dist C F :=
sorry

end hexagon_inequality_l769_769900


namespace find_coords_of_A_l769_769867

-- Define the coordinates of the points
def pointA (z : ℝ) := (0, 0, z)
def pointB := (2, 1, -3)

-- Define the distance formula in 3D
def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  Math.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2)

-- Given condition for point A and B's distance
def point_A_on_z_axis (z : ℝ) : Prop :=
  distance (pointA z) pointB = 3

-- The proof statement
theorem find_coords_of_A (z : ℝ) (h : point_A_on_z_axis z) :
  z = -1 ∨ z = -5 :=
sorry

end find_coords_of_A_l769_769867


namespace count_factors_of_180_multiple_of_15_l769_769786

def is_factor (n k : ℕ) : Prop := n % k = 0

def is_multiple_of (k m : ℕ) : Prop := k % m = 0

theorem count_factors_of_180_multiple_of_15 : 
  (finset.filter (λ f, is_factor 180 f ∧ is_multiple_of f 15) (finset.range (181))).card = 6 := 
by 
sorry

end count_factors_of_180_multiple_of_15_l769_769786


namespace find_quadratic_polynomial_l769_769068

theorem find_quadratic_polynomial (q : ℝ → ℝ) 
  (h1 : q 2 = 12) 
  (h2 : q 3 = 0) : 
  q = λ x, -4 * x^2 + 8 * x + 12 :=
sorry

end find_quadratic_polynomial_l769_769068


namespace factors_of_180_l769_769769

-- Define a predicate to check if a number is a factor of another number
def is_factor (m n : ℕ) : Prop := n % m = 0

-- Define a predicate to check if a number is a multiple of another number
def is_multiple (m n : ℕ) : Prop := m % n = 0

-- Define the problem
theorem factors_of_180 : {n : ℕ // is_factor n 180 ∧ is_multiple n 15}.card = 6 :=
by sorry

end factors_of_180_l769_769769


namespace factors_of_180_multiple_of_15_count_l769_769801

theorem factors_of_180_multiple_of_15_count :
  ∃ n : Nat, n = 6 ∧ ∀ x : Nat, x ∣ 180 → 15 ∣ x → x > 0 :=
by
  sorry

end factors_of_180_multiple_of_15_count_l769_769801


namespace karlson_candy_consumption_l769_769514

noncomputable def boy_candy_count : ℕ := 300
noncomputable def boy_daily_candy_consumption : ℕ := 1
noncomputable def karlson_weekly_candy_gift : ℕ := 2
noncomputable def january_first_2013_weekday : String := "Tuesday"
noncomputable def total_candies_karlson_consumed : ℕ := 66

theorem karlson_candy_consumption :
  let days_in_week := 7 in
  let karlson_weekly_visits_consumption := 2 in
  let total_candies := 300 in
  let weeks := total_candies / (days_in_week + karlson_weekly_visits_consumption) in
  let remaining_candies := total_candies % (days_in_week + karlson_weekly_visits_consumption) in
  let karlson_consumption := weeks * karlson_weekly_visits_consumption in
  karlson_consumption = total_candies_karlson_consumed :=
by
  sorry

end karlson_candy_consumption_l769_769514


namespace pyramid_top_value_l769_769267

theorem pyramid_top_value 
  (p : ℕ) (q : ℕ) (z : ℕ) (m : ℕ) (n : ℕ) (left_mid : ℕ) (right_mid : ℕ) 
  (left_upper : ℕ) (right_upper : ℕ) (x_pre : ℕ) (x : ℕ) : 
  p = 20 → 
  q = 6 → 
  z = 44 → 
  m = p + 34 → 
  n = q + z → 
  left_mid = 17 + 29 → 
  right_mid = m + n → 
  left_upper = 36 + left_mid → 
  right_upper = right_mid + 42 → 
  x_pre = left_upper + 78 → 
  x = 2 * x_pre → 
  x = 320 :=
by
  intros
  sorry

end pyramid_top_value_l769_769267


namespace meaningful_fraction_l769_769845

theorem meaningful_fraction (x : ℝ) : (1 / (x + 1) ∈ ℚ) ↔ x ≠ -1 :=
by
  sorry

end meaningful_fraction_l769_769845


namespace intersection_is_correct_l769_769254

-- Defining sets A and B
def setA : Set ℝ := {x : ℝ | x^2 - 2 * x ≤ 0}
def setB : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}

-- Target intersection set
def setIntersection : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 1}

-- Theorem to be proved
theorem intersection_is_correct : (setA ∩ setB) = setIntersection :=
by
  -- Proof steps will go here
  sorry

end intersection_is_correct_l769_769254


namespace certain_event_l769_769488

def event_A := (rngtoss : Bool → Prop) -- Successful ring toss defined as a random event
def event_B := (fthrows : Bool → Prop) -- Successful free throw defined as a random event
def event_C := (redballs : Bool := true) -- Drawing a red ball, which is a certain event
def event_D := (dice10 : Bool := false) -- Rolling dice to get a 10, an impossible event

theorem certain_event : event_C = true :=
by sorry

end certain_event_l769_769488


namespace centroid_property_parallelogram_diagonal_property_l769_769133

-- Problem (1)
theorem centroid_property (G A B C P : Point) (hG : is_centroid G A B C) :
  ∀ (P : Point), vector_eq (PG P G) (scalar_mul (1/3) (vector_sum [PA P A, PB P B, PC P C])) :=
sorry

-- Problem (2)
theorem parallelogram_diagonal_property (G A B C D P : Point) 
  (hG1 : is_midpoint G A C) (hG2 : is_midpoint G B D) :
  ∀ (P : Point), vector_eq (PG P G) (scalar_mul (1/4) (vector_sum [PA P A, PB P B, PC P C, PD P D])) :=
sorry

end centroid_property_parallelogram_diagonal_property_l769_769133


namespace ratio_of_largest_element_to_sum_other_elements_l769_769588

theorem ratio_of_largest_element_to_sum_other_elements :
  let largest_element := 2^20
  let geometric_series_sum := (2^20) - 1
  (largest_element : ℝ) / (geometric_series_sum : ℝ) ≈ 1 :=
by
  let largest_element := 2^20
  let geometric_series_sum := 2^20 - 1
  -- Ratio is approximately 1
  have : (largest_element : ℝ) / (geometric_series_sum : ℝ) ≈ 1 := sorry
  exact this

end ratio_of_largest_element_to_sum_other_elements_l769_769588


namespace count_factors_of_180_multiple_of_15_l769_769780

def is_factor (n k : ℕ) : Prop := n % k = 0

def is_multiple_of (k m : ℕ) : Prop := k % m = 0

theorem count_factors_of_180_multiple_of_15 : 
  (finset.filter (λ f, is_factor 180 f ∧ is_multiple_of f 15) (finset.range (181))).card = 6 := 
by 
sorry

end count_factors_of_180_multiple_of_15_l769_769780


namespace problem_statement_l769_769304

noncomputable def countNs : Nat :=
  let N_values := {N : ℕ | ∃ x : ℝ, 0 < N ∧ N < 500 ∧ (x ^ Nat.floor x = N)}
  N_values.toFinset.card

theorem problem_statement :
  countNs = 287 := by
  sorry

end problem_statement_l769_769304


namespace area_of_region_B_correct_l769_769201

noncomputable def area_of_region_B : ℝ :=
  let B_conditions (z : ℂ) : Prop := 
    let z_re := z.re
    and z_im := z.im
    in (0 ≤ z_re / 50) ∧ (z_re / 50 ≤ 1) ∧ (0 ≤ z_im / 50) ∧ (z_im / 50 ≤ 1) ∧
       (0 ≤ 50 * z_re / (z_re^2 + z_im^2)) ∧ (50 * z_re / (z_re^2 + z_im^2) ≤ 1) ∧
       (0 ≤ 50 * z_im / (z_re^2 + z_im^2)) ∧ (50 * z_im / (z_re^2 + z_im^2) ≤ 1)
  in if ∃ z : ℂ, B_conditions z then
          1875 - 312.5 * Real.pi
     else
          0

theorem area_of_region_B_correct : 
  ∃ z : ℂ, let B_conditions (z : ℂ) : Prop := 
              let z_re := z.re
              and z_im := z.im
              in (0 ≤ z_re / 50) ∧ (z_re / 50 ≤ 1) ∧ (0 ≤ z_im / 50) ∧ (z_im / 50 ≤ 1) ∧
                 (0 ≤ 50 * z_re / (z_re^2 + z_im^2)) ∧ (50 * z_re / (z_re^2 + z_im^2) ≤ 1) ∧
                 (0 ≤ 50 * z_im / (z_re^2 + z_im^2)) ∧ (50 * z_im / (z_re^2 + z_im^2) ≤ 1)
           in B_conditions z
  -> area_of_region_B = 1875 - 312.5 * Real.pi :=
by
  sorry

end area_of_region_B_correct_l769_769201


namespace find_m_value_l769_769897

theorem find_m_value (m a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℝ)
  (h1 : (x + m)^9 = a_0 + a_1 * (x + 1) + a_2 * (x + 1)^2 + a_3 * (x + 1)^3 + 
  a_4 * (x + 1)^4 + a_5 * (x + 1)^5 + a_6 * (x + 1)^6 + a_7 * (x + 1)^7 + 
  a_8 * (x + 1)^8 + a_9 * (x + 1)^9)
  (h2 : a_0 - a_1 + a_2 - a_3 + a_4 - a_5 + a_6 - a_7 + a_8 - a_9 = 3^9) :
  m = 4 :=
by
  sorry

end find_m_value_l769_769897


namespace percentage_change_area_l769_769432

theorem percentage_change_area (L B : ℝ) (h : L > 0 ∧ B > 0) :
  let A1 := L * B
  let L2 := L / 2
  let B2 := 3 * B
  let A2 := L2 * B2
  let percentage_change := ((A2 - A1) / A1) * 100
  percentage_change = 50 := by
  have A1_def := calc
    A1 = L * B : rfl
  have A2_def := calc
    A2 = (L / 2) * (3 * B) : rfl
    ... = (3 / 2) * (L * B) : by ring
  have pc_def := calc
    percentage_change = ((A2 - A1) / A1) * 100 : rfl
    ... = (( (3 / 2) * (L * B) - (L * B) ) / (L * B) ) * 100 : by rw [A1_def, A2_def]
    ... = (1 / 2) * 100 : by ring
  exact pc_def

end percentage_change_area_l769_769432


namespace sum_reciprocal_of_roots_l769_769584

theorem sum_reciprocal_of_roots :
  let p q : ℝ := roots_of_quadratic 6 5 7 in
  1/p + 1/q = -5/7 :=
by
  let p q : ℝ := roots_of_quadratic 6 5 7
  have h_sum : p + q = -5/6 := Vieta_sum 6 5 7
  have h_prod : p q = 7/6 := Vieta_product 6 5 7
  show 1/p + 1/q = -5/7 from
  calc
    1/p + 1/q = (p + q) / (p q) : by sorry
             ... = -5/7 : by sorry

end sum_reciprocal_of_roots_l769_769584


namespace glass_coloring_possible_l769_769160

-- Define the constant that represents the area of the glass
def glass_area : ℝ := 1

-- Define the total number of regions (countries) on each side of the glass
def num_regions : ℕ := 5

-- Define a matrix to represent the areas of regions colored the same on both sides
noncomputable def area_matrix (m n : ℕ) : ℝ :=
if 1 ≤ m ∧ m ≤ num_regions ∧ 1 + num_regions ≤ n ∧ n ≤ 2 * num_regions then
    -- Hypothetical area value (exact value to be filled in with real calculation in the proof)
    sorry
else
    0

-- Formulate the theorem to be proven
theorem glass_coloring_possible : ∃ (colors : Fin num_regions → Fin num_regions),
  ∑ i : Fin num_regions, area_matrix i (colors i + num_regions) ≥ (glass_area / num_regions) :=
by
  -- Existence of such a coloring is guaranteed by provided proof steps
  sorry

end glass_coloring_possible_l769_769160


namespace alice_catch_up_time_l769_769564

def alice_speed : ℝ := 45
def tom_speed : ℝ := 15
def initial_distance : ℝ := 4
def minutes_per_hour : ℝ := 60

theorem alice_catch_up_time :
  (initial_distance / (alice_speed - tom_speed)) * minutes_per_hour = 8 :=
by
  sorry

end alice_catch_up_time_l769_769564


namespace no_20_odd_rows_15_odd_columns_l769_769664

theorem no_20_odd_rows_15_odd_columns (n : ℕ) (table : ℕ → ℕ → bool) (cross_count 
  : ℕ) 
  (odd_rows : ℕ → bool) 
  (odd_columns : ℕ → bool) :
  (∀ i, i < n → (odd_rows i = true ↔ ∃ j, j < n ∧ table i j = true ∧ cross_count = 20))
  → (∀ j, j < n → (odd_columns j = true ↔ ∃ i, i < n ∧ table i j = true ∧ cross_count = 15))
  → false := 
sorry

end no_20_odd_rows_15_odd_columns_l769_769664


namespace area_of_circle_given_circumference_l769_769060

theorem area_of_circle_given_circumference (C : ℝ) (hC : C = 18 * Real.pi) (k : ℝ) :
  ∃ r : ℝ, C = 2 * Real.pi * r ∧ k * Real.pi = Real.pi * r^2 → k = 81 :=
by
  sorry

end area_of_circle_given_circumference_l769_769060


namespace correct_operation_l769_769502

theorem correct_operation (a b : ℝ) : (a + b) * (a - b) = a^2 - b^2 := by
  sorry

end correct_operation_l769_769502


namespace find_y_satisfies_eq_l769_769623

theorem find_y_satisfies_eq (y : ℚ) (h : (sqrt (8 * y)) / (sqrt (2 * (y - 2))) = 3) : y = 18 / 5 :=
sorry

end find_y_satisfies_eq_l769_769623


namespace parabolas_intersect_l769_769101

theorem parabolas_intersect :
  ∃ (x1 y1 x2 y2 : ℝ),
    (y1 = 4 * x1 ^ 2 + 3 * x1 - 4) ∧ (y1 = 2 * x1 ^ 2 + 15) ∧
    (y2 = 4 * x2 ^ 2 + 3 * x2 - 4) ∧ (y2 = 2 * x2 ^ 2 + 15) ∧
    ((x1, y1) = (-19/2:ℝ, 195.5) ∨ (x1, y1) = (5/2:ℝ, 27.5)) ∧
    ((x2, y2) = (-19/2:ℝ, 195.5) ∨ (x2, y2) = (5/2:ℝ, 27.5)) ∧
    x1 ≠ x2 := sorry

end parabolas_intersect_l769_769101


namespace jake_comic_books_l769_769884

variables (J : ℕ)

def brother_comic_books := J + 15
def total_comic_books := J + brother_comic_books

theorem jake_comic_books : total_comic_books = 87 → J = 36 :=
by
  sorry

end jake_comic_books_l769_769884


namespace S10_value_l769_769453

variable (a_n : ℕ → ℝ) (S : ℕ → ℝ)
variable (d a₁ : ℝ)
variable (n : ℕ)

-- Assuming the sequence is arithmetic
def is_arithmetic_sequence (a_n : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a_n (n + 1) = a_n n + d

-- Sum of first n terms of the arithmetic sequence
def sum_of_terms (a_n : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ), S n = (n * (a₁ + a₁ + (n - 1) * d)) / 2

-- Given conditions
axiom a7_condition : a_n 7 = 5
axiom S7_condition : S 7 = 21

-- To prove
theorem S10_value : (a_n 1 = 1) → (d = 2/3) → S 10 = 40 :=
by
  intro ha1 hd
  rw [sum_of_terms a_n S 10]
  sorry

end S10_value_l769_769453


namespace two_powers_permutation_implies_equality_l769_769960

def is_permutation_of_digits (a b : ℕ) : Prop :=
  (list.permutations (a.digits 10)).contains (b.digits 10)

theorem two_powers_permutation_implies_equality (r s : ℕ) (hr : 0 < r) (hs : 0 < s)
  (h_permute : is_permutation_of_digits (2^r) (2^s)) : r = s :=
by
  sorry

end two_powers_permutation_implies_equality_l769_769960


namespace max_value_2a_minus_b_l769_769705

noncomputable def a (θ : ℝ) : ℝ × ℝ × ℝ := (Real.cos θ, Real.sin θ, 1)
def b : ℝ × ℝ × ℝ := (1, -1, 2)

def vector2a_minus_b (θ : ℝ) : ℝ × ℝ × ℝ :=
  let (a1, a2, a3) := a θ
  (2 * a1 - b.1, 2 * a2 - b.snd, 2 * a3 - b.2)

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  let (x, y, z) := v
  Real.sqrt (x^2 + y^2 + z^2)

theorem max_value_2a_minus_b : ∃ θ : ℝ, magnitude (vector2a_minus_b θ) = 4 := by
  sorry

end max_value_2a_minus_b_l769_769705


namespace correct_equation_l769_769495

theorem correct_equation (a b : ℝ) : 
  (¬ (a^2 + a^3 = a^6)) ∧ (¬ ((ab)^2 = ab^2)) ∧ (¬ ((a+b)^2 = a^2 + b^2)) ∧ ((a+b)*(a-b) = a^2 - b^2) :=
by {
  sorry
}

end correct_equation_l769_769495


namespace area_of_shape_tangent_and_curve_l769_769964

theorem area_of_shape_tangent_and_curve :
  let f := λ x : ℝ, x^3 - x^2 + 1,
      g := λ x : ℝ, -x^2,
      point := (1 : ℝ, 1 : ℝ),
      tangent := λ x : ℝ, x,
      area := ∫ x in -1..0, -(x^2 + x)
  in area = 1 / 6 := by
sorry

end area_of_shape_tangent_and_curve_l769_769964


namespace arithmetic_sequence_proof_l769_769872

variable {α : Type*}

structure ArithmeticSeq (a : ℕ → α) where
  d : α
  aₙ : ∀ n, a n = a 0 + n * d

-- Define the arithmetic sequence {a_n}
variables (a : ℕ → ℝ) [ArithmeticSeq a]

-- Define given conditions
def given_condition_1 := a 1 + 3 * a 6 + a 11 = 100

-- Translate the main question to a proof goal
theorem arithmetic_sequence_proof (h1 : given_condition_1) : 2 * a 7 - a 8 = 20 := by
  sorry

end arithmetic_sequence_proof_l769_769872


namespace find_acute_angle_l769_769264

noncomputable def vector_a (α : ℝ) : ℝ × ℝ := (3 * Real.cos α, 2)
noncomputable def vector_b (α : ℝ) : ℝ × ℝ := (3, 4 * Real.sin α)
def are_parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 = a.2 * b.1

theorem find_acute_angle (α : ℝ) (h : are_parallel (vector_a α) (vector_b α)) (h_acute : 0 < α ∧ α < π / 2) : 
  α = π / 4 :=
by
  sorry

end find_acute_angle_l769_769264


namespace log_of_50_between_consecutive_integers_l769_769990

theorem log_of_50_between_consecutive_integers : 
  (∃ c d : ℕ, 4 < log 2 50 ∧ log 2 50 < 6 ∧ c = 5 ∧ d = 6 ∧ c + d = 11) :=
by
  -- Conditions
  have h1 : log 2 16 = 4 := by norm_num,
  have h2 : log 2 64 = 6 := by norm_num,
  have h3 : 16 < 50 := by norm_num,
  have h4 : 50 < 64 := by norm_num,
  sorry

end log_of_50_between_consecutive_integers_l769_769990


namespace smallest_period_of_f_interval_of_monotonic_increase_l769_769704

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (Real.cos x + Real.sin x, 2 * Real.sin x)
noncomputable def vec_b (x : ℝ) : ℝ × ℝ := (Real.cos x - Real.sin x, Real.cos x)
noncomputable def f (x : ℝ) : ℝ := vec_a(x).1 * vec_b(x).1 + vec_a(x).2 * vec_b(x).2

theorem smallest_period_of_f : ∃ T > 0, ∀ x, f(x + T) = f(x) ∧ T = Real.pi :=
by
  sorry

theorem interval_of_monotonic_increase : 
  ∀ x, x ∈ Set.Icc (Real.pi / 4) (3 * Real.pi / 4) → 
  (f(x + Real.pi / 8) - f(x)) ≥ 0 → 
  x ∈ Set.Icc (5 * Real.pi / 8) (3 * Real.pi / 4) :=
by
  sorry

end smallest_period_of_f_interval_of_monotonic_increase_l769_769704


namespace petya_can_force_difference_2014_l769_769018

theorem petya_can_force_difference_2014 :
  ∀ (p q r : ℚ), ∃ (a b c : ℚ), ∀ (x : ℝ), (x^3 + a * x^2 + b * x + c = 0) → 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 - x2 = 2014 :=
by sorry

end petya_can_force_difference_2014_l769_769018


namespace prize_interval_l769_769333

theorem prize_interval (prize1 prize2 prize3 prize4 prize5 interval : ℝ) (h1 : prize1 = 5000) 
  (h2 : prize2 = 5000 - interval) (h3 : prize3 = 5000 - 2 * interval) 
  (h4 : prize4 = 5000 - 3 * interval) (h5 : prize5 = 5000 - 4 * interval) 
  (h_total : prize1 + prize2 + prize3 + prize4 + prize5 = 15000) : 
  interval = 1000 := 
by
  sorry

end prize_interval_l769_769333


namespace problem1_problem2_l769_769693

-- Definitions

def f (x q : ℝ) := x^2 - 16 * x + q + 3

-- Problem 1: Given minimum value of -60
theorem problem1 (q : ℝ) (hx : ∀ x, f x q ≥ -60) : q = 1 := sorry

-- Problem 2: Given function has a root in the interval [-1, 1]
theorem problem2 (q : ℝ) (hx : ∃ x : ℝ, x ∈ set.Icc (-1) 1 ∧ f x q = 0) : -20 ≤ q ∧ q ≤ 12 := sorry

end problem1_problem2_l769_769693


namespace day_365_is_Tuesday_l769_769841

def day_of_week : Type := ℕ

def Day.January_15_2005_is_Tuesday (day_of_week_n : day_of_week) : Prop :=
  day_of_week_n ≡ 2 [MOD 7]

def day_365_is_same_day_of_week (day_of_week_n day_after_n_days : day_of_week) (days_between : ℕ) : Prop :=
  (day_of_week_n + days_between) % 7 = day_after_n_days % 7

theorem day_365_is_Tuesday (day_15 : day_of_week) :
  (Day.January_15_2005_is_Tuesday day_15) →
  day_365_is_same_day_of_week day_15 day_15 350 →
  (day_15 % 7 = 2 % 7) :=
by
  intros h1 h2
  sorry

end day_365_is_Tuesday_l769_769841


namespace inequality_e_f1_gt_f2_l769_769645

variable {f : ℝ → ℝ}

theorem inequality_e_f1_gt_f2 (h : ∀ x : ℝ, f x > f'' x) : e * f 1 > f 2 :=
sorry

end inequality_e_f1_gt_f2_l769_769645


namespace PetyaCanAlwaysForceDifferenceOfRoots2014_l769_769013

noncomputable def canPetyaForceDifferenceOfRoots2014 : Prop :=
  ∀ (v1 v2 : ℚ) (vasyachooses : (ℚ → ℚ) → Prop), (∃ p q : ℚ, vasyachooses (λ _: ℚ, _) ∧ vasyachooses (λ _: ℚ, _)) →
  ∃ (a b c : ℚ), 
    (vasyachooses (λ _: ℚ, a) ∧ vasyachooses (λ _: ℚ, c)) ∨
    (vasyachooses (λ _: ℚ, b) ∧ vasyachooses (λ _: ℚ, c)) ∧
    (∀ x y : ℚ, (x^3 + a*x^2 + b*x + c = 0 → y^3 + a*y^2 + b*y + c = 0 → abs(x - y) = 2014))

theorem PetyaCanAlwaysForceDifferenceOfRoots2014 : canPetyaForceDifferenceOfRoots2014 :=
sorry

end PetyaCanAlwaysForceDifferenceOfRoots2014_l769_769013


namespace positive_factors_of_180_multiple_of_15_count_l769_769759

theorem positive_factors_of_180_multiple_of_15_count : 
  let factorization_180 := 2^2 * 3^2 * 5;
  let is_multiple_of_15 (n : ℕ) := ∃ k1 k2 : ℕ, k1 * 3 = n ∧ k2 * 5 = n;
  let factors_180 := { n : ℕ | n ∣ factorization_180 };
  { n : ℕ | n ∈ factors_180 ∧ is_multiple_of_15 n }.card = 6 :=
by
  sorry

end positive_factors_of_180_multiple_of_15_count_l769_769759


namespace median_of_set_is_90_l769_769975

namespace MedianProblem

-- Conditions
def known_nums : List ℕ := [92, 88, 86, 90, 91]
def y : ℕ := 90
def full_set : List ℕ := known_nums ++ [y]
def mean := float_of_list full_set / 6

-- The goal is to prove that the median of the set is 90
theorem median_of_set_is_90 :
  (mean = 89.5) →
  median full_set = 90 :=
by
  sorry

-- Helper function to compute mean of a list of natural numbers (for completeness)
def float_of_list : List ℕ → ℝ
| [] => 0
| (h :: t) => h + float_of_list t

-- Helper function to compute the median of a list of natural numbers
def median (lst : List ℕ) : ℕ :=
let sorted := lst.qsort (· ≤ ·)
let mid := sorted.length / 2
if sorted.length % 2 = 0 then (sorted.get! (mid - 1) + sorted.get! mid) / 2
else sorted.get! mid

end MedianProblem

end median_of_set_is_90_l769_769975


namespace count_positive_factors_of_180_multiple_of_15_l769_769729

statement: ℕ
statement := sorry

theorem count_positive_factors_of_180_multiple_of_15 :
  let factors_of_180 := {d ∈ (finset.range 181).filter (λ x, 180 % x = 0)}
  let multiples_of_15 := (finset.range 181).filter (λ x, x % 15 = 0)
  let desired_factors := factors_of_180 ∩ multiples_of_15
  desired_factors.card = 6 :=
by
  sorry

end count_positive_factors_of_180_multiple_of_15_l769_769729


namespace sum_of_AB_divisible_by_9_l769_769826

theorem sum_of_AB_divisible_by_9 (A B : ℕ) (hA : 0 ≤ A ∧ A ≤ 9) (hB : 0 ≤ B ∧ B ≤ 9) 
  (divisibility_condition : ((25 + A + B) % 9 = 0)) 
  : (A + B = 2 ∨ A + B = 11) → ((A + B = 2) + (A + B = 11) = 13) :=
by
  sorry

end sum_of_AB_divisible_by_9_l769_769826


namespace find_factor_l769_769168

theorem find_factor (n f : ℤ) (h₁ : n = 124) (h₂ : n * f - 138 = 110) : f = 2 := by
  sorry

end find_factor_l769_769168


namespace leak_drain_time_l769_769506

-- Define the conditions
def pump_fill_rate : ℚ := 1 / 2
def combined_fill_rate : ℚ := 7 / 15

-- Define the question and the required answer as a theorem
theorem leak_drain_time : 
  ∀ (P L : ℚ), 
    P = pump_fill_rate → 
    P - L = combined_fill_rate → 
    (1 / L) = 30 := 
begin
  intros P L hP hPL,
  sorry
end

end leak_drain_time_l769_769506


namespace pencils_per_associate_professor_l769_769573

theorem pencils_per_associate_professor
    (A B P : ℕ) -- the number of associate professors, assistant professors, and pencils per associate professor respectively
    (h1 : A + B = 6) -- there are a total of 6 people
    (h2 : A * P + B = 7) -- total number of pencils is 7
    (h3 : A + 2 * B = 11) -- total number of charts is 11
    : P = 2 :=
by
  -- Placeholder for the proof
  sorry

end pencils_per_associate_professor_l769_769573


namespace percentage_change_area_l769_769446

theorem percentage_change_area (L B : ℝ) (hL : L > 0) (hB : B > 0) :
    let A := L * B
    let L' := L / 2
    let B' := 3 * B
    let A' := (L / 2) * (3 * B)
    (A' - A) / A * 100 = 50 :=
by
  let A := L * B
  let A' := (L / 2) * (3 * B)
  have h : (A' - A) / A * 100 = (1 / 2) * 100 := sorry
  exact h

end percentage_change_area_l769_769446


namespace remainder_ab_divided_by_n_l769_769360

variables {n a b : ℕ}

/-- Main theorem statement -/
theorem remainder_ab_divided_by_n 
  (hn : 0 < n)
  (ha_inv : ∃ a_inv, (a * a_inv) % n = 1)
  (hb_inv : ∃ b_inv, (b * b_inv) % n = 1)
  (hab : (a % n) = (2 * (nat.gcd_a b n) % n)) :
  (a * b) % n = 2 :=
by
  sorry

end remainder_ab_divided_by_n_l769_769360


namespace circle_area_l769_769221

theorem circle_area : 
  let eq := (x^2 + y^2 - 4*x + 10*y + 20 = 0) in
  let circle_eq := (x - 2)^2 + (y + 5)^2 = 9 in
  (∃ x y, circle_eq) → 
  Area(circle_eq) = 9 * π :=
by
  sorry

end circle_area_l769_769221


namespace max_blue_points_l769_769322

-- We define the number of spheres and the categorization of red and green spheres
def number_of_spheres : ℕ := 2016

-- Definition of the number of red spheres
def red_spheres (r : ℕ) : Prop := r <= number_of_spheres

-- Definition of the number of green spheres as the complement of red spheres
def green_spheres (r : ℕ) : ℕ := number_of_spheres - r

-- Definition of the number of blue points as the intersection of red and green spheres
def blue_points (r : ℕ) : ℕ := r * green_spheres r

-- Theorem: Given the conditions, the maximum number of blue points is 1016064
theorem max_blue_points : ∃ r : ℕ, red_spheres r ∧ blue_points r = 1016064 := by
  sorry

end max_blue_points_l769_769322


namespace total_skips_l769_769607

theorem total_skips (fifth throw : ℕ) (fourth throw : ℕ) (third throw : ℕ) (second throw : ℕ) (first throw : ℕ) :
  fifth throw = 8 →
  fourth throw = fifth throw - 1 →
  third throw = fourth throw + 3 →
  second throw = third throw / 2 →
  first throw = second throw - 2 →
  first throw + second throw + third throw + fourth throw + fifth throw = 33 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  rw [h₁, h₂, h₃, h₄, h₅]
  sorry

end total_skips_l769_769607


namespace total_cookies_baked_l769_769283

-- Definitions based on conditions
def pans : ℕ := 5
def cookies_per_pan : ℕ := 8

-- Statement of the theorem to be proven
theorem total_cookies_baked :
  pans * cookies_per_pan = 40 := by
  sorry

end total_cookies_baked_l769_769283


namespace count_factors_of_180_multiple_of_15_l769_769785

def is_factor (n k : ℕ) : Prop := n % k = 0

def is_multiple_of (k m : ℕ) : Prop := k % m = 0

theorem count_factors_of_180_multiple_of_15 : 
  (finset.filter (λ f, is_factor 180 f ∧ is_multiple_of f 15) (finset.range (181))).card = 6 := 
by 
sorry

end count_factors_of_180_multiple_of_15_l769_769785


namespace count_factors_of_180_multiple_of_15_l769_769712

def is_factor (x n : ℕ) := n % x = 0 -- Definition of a factor
def is_multiple (x k : ℕ) := x % k = 0 -- Definition of a multiple

theorem count_factors_of_180_multiple_of_15 :
  (finset.filter (λ x, is_multiple x 15) (finset.filter (λ x, is_factor x 180) (finset.range (180 + 1)))).card = 6 := by
  sorry

end count_factors_of_180_multiple_of_15_l769_769712


namespace factors_of_180_multiple_of_15_l769_769749

/-- A factor of a number n is a positive integer which divides n without leaving a remainder. -/
def is_factor (n k : ℕ) : Prop :=
k > 0 ∧ n % k = 0

/-- A multiple of a number m is an integer which is divisible by m without leaving a remainder. -/
def is_multiple (k m : ℕ) : Prop :=
k % m = 0

/-- There are 6 factors of 180 that are also multiples of 15. -/
theorem factors_of_180_multiple_of_15 : 
  (Finset.filter (λ k, is_multiple k 15) { x | is_factor 180 x }.to_finset).card = 6 :=
by 
  sorry

end factors_of_180_multiple_of_15_l769_769749


namespace count_factors_180_multiple_of_15_is_6_l769_769814

-- We define a function to check if 'n' is a factor of 'm'
def is_factor (m n : ℕ) : Prop := n ∣ m

-- We define a function to check if 'n' is a multiple of 'd'
def is_multiple (n d : ℕ) : Prop := d ∣ n

-- We define a function to collect all factors of a number 'n'
def factors (n : ℕ) : list ℕ := 
  -- We use list.filter to keep elements that satisfy is_factor n
  list.filter (λ x, is_factor n x) (list.range (n + 1))

-- We count the number of elements in a list that satisfy a predicate
def count_satisfying {α : Type*} (pred : α → Prop) [decidable_pred pred] (l : list α) : ℕ :=
  list.length (list.filter pred l)

-- Noncomputable definition to specify that counting number
noncomputable def count_positive_factors_of_180_multiple_of_15 : ℕ :=
  count_satisfying (λ x, is_multiple x 15) (factors 180)

-- We state the theorem equivalence to the proof problem
theorem count_factors_180_multiple_of_15_is_6 : count_positive_factors_of_180_multiple_of_15 = 6 :=
  sorry


end count_factors_180_multiple_of_15_is_6_l769_769814


namespace average_of_first_5_subjects_l769_769184

theorem average_of_first_5_subjects (avg_6_subjects : ℚ) (marks_6th_subject : ℚ) (total_subjects : ℕ) (total_marks_6_subjects : ℚ) (total_marks_5_subjects : ℚ) (avg_5_subjects : ℚ) :
  avg_6_subjects = 77 ∧ marks_6th_subject = 92 ∧ total_subjects = 6 ∧ total_marks_6_subjects = avg_6_subjects * total_subjects ∧ total_marks_5_subjects = total_marks_6_subjects - marks_6th_subject ∧ avg_5_subjects = total_marks_5_subjects / 5
  → avg_5_subjects = 74 := by
  sorry

end average_of_first_5_subjects_l769_769184


namespace evaluate_complex_pow_l769_769217

open Complex

noncomputable def calc : ℂ := (-64 : ℂ) ^ (7 / 6)

theorem evaluate_complex_pow : calc = 128 * Complex.I := by 
  -- Recognize that (-64) = (-4)^3
  -- Apply exponent rules: ((-4)^3)^(7/6) = (-4)^(3 * 7/6) = (-4)^(7/2)
  -- Simplify (-4)^(7/2) = √((-4)^7) = √(-16384)
  -- Calculation (-4)^7 = -16384
  -- Simplify √(-16384) = 128i
  sorry

end evaluate_complex_pow_l769_769217


namespace transform_sin_function_l769_769099

theorem transform_sin_function :
  ∀ x ∈ ℝ, 
  (λ x : ℝ, sin x) (2 * (x + π / 6)) = sin(2 * x + π / 3) :=
by sorry

end transform_sin_function_l769_769099


namespace total_avg_donation_per_person_l769_769324

-- Definition of variables and conditions
variables (avgA avgB : ℝ) (numA numB : ℕ)
variables (h1 : avgB = avgA - 100)
variables (h2 : 2 * numA * avgA = 4 * numB * (avgA - 100))
variables (h3 : numA = numB / 4)

-- Lean 4 statement to prove the total average donation per person is 120
theorem total_avg_donation_per_person (h1 :  avgB = avgA - 100)
    (h2 : 2 * numA * avgA = 4 * numB * (avgA - 100))
    (h3 : numA = numB / 4) : 
    ( (numA * avgA + numB * avgB) / (numA + numB) ) = 120 :=
sorry

end total_avg_donation_per_person_l769_769324


namespace cost_of_blue_socks_l769_769379

theorem cost_of_blue_socks
    (pairs_of_red_socks : ℕ)
    (pairs_of_blue_socks : ℕ)
    (total_cost: ℕ)
    (cost_per_red_pair: ℕ)
    (H1 : pairs_of_red_socks = 4)
    (H2 : pairs_of_blue_socks = 6)
    (H3 : total_cost = 42)
    (H4 : cost_per_red_pair = 3) :
    total_cost - (pairs_of_red_socks * cost_per_red_pair) = pairs_of_blue_socks * 5 := 
begin
    sorry
end

end cost_of_blue_socks_l769_769379


namespace marie_tasks_finish_time_l769_769922

noncomputable def total_time (times : List ℕ) : ℕ :=
  times.foldr (· + ·) 0

theorem marie_tasks_finish_time :
  let task_times := [30, 40, 50, 60]
  let start_time := 8 * 60 -- Start time in minutes (8:00 AM)
  let end_time := start_time + total_time task_times
  end_time = 11 * 60 := -- 11:00 AM in minutes
by
  -- Add a placeholder for the proof
  sorry

end marie_tasks_finish_time_l769_769922


namespace sum_of_square_roots_l769_769192

theorem sum_of_square_roots :
  (√1 + √(1 + 3) + √(1 + 3 + 5) + √(1 + 3 + 5 + 7) + √(1 + 3 + 5 + 7 + 9)) = 15 :=
by
  sorry

end sum_of_square_roots_l769_769192


namespace original_quadratic_equation_l769_769998

theorem original_quadratic_equation {p q : ℤ} (h1 : p ≠ 0) (h2 : q ≠ 0) (hq : q = -2 * p)
  (hq_pq : p = - (p + q))
  (original_eq_formed_eq : x^2 + p * x + q = x^2 - (p + q) * x + p * q) : ∃ eq : ℤ → ℤ, eq = λ x, x^2 + x - 2 :=
by
  sorry

end original_quadratic_equation_l769_769998


namespace remainder_x_plus_13_div_41_l769_769121

theorem remainder_x_plus_13_div_41 (x : ℤ) (h : x % 82 = 5) : (x + 13) % 41 = 18 := by
  sorry

end remainder_x_plus_13_div_41_l769_769121


namespace alissa_earrings_l769_769186

theorem alissa_earrings :
  ∀ (bought_pairs given_earrings : ℕ), 
    bought_pairs = 12 →
    (∀ pairs_per_earring, pairs_per_earring = 2 → 
      (bought_pairs * pairs_per_earring) / 2 = given_earrings →
      given_earrings * 3 = 36) :=
begin
  intros bought_pairs given_earrings bought_pairs_eq pairs_per_earring_eq division_eq,
  unfold pairs_per_earring at *,
  rw pairs_per_earring_eq at *,
  rw bought_pairs_eq at *,
  rw mul_comm at *,
  have bought_earrings : 12 * 2 = 24 := by norm_num,
  rw ← bought_earrings at *,
  have given_earrings_calc : 24 / 2 = 12 := by norm_num,
  rw ← given_earrings_calc at division_eq,
  sorry
end

end alissa_earrings_l769_769186


namespace percentage_change_in_area_halved_length_tripled_breadth_l769_769427

def original_area (L B : ℝ) : ℝ := L * B
def new_area (L B : ℝ) : ℝ := (L / 2) * (3 * B)

def percentage_change (A1 A2 : ℝ) : ℝ :=
  ((A2 - A1) / A1) * 100

theorem percentage_change_in_area_halved_length_tripled_breadth
  (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  percentage_change (original_area L B) (new_area L B) = 50 :=
begin
  sorry
end

end percentage_change_in_area_halved_length_tripled_breadth_l769_769427


namespace time_to_cross_bridge_l769_769171

-- Conditions
def length_of_train : ℝ := 150 -- in meters
def speed_of_train : ℝ := 45 * 1000 / 3600 -- converted to m/s
def length_of_bridge : ℝ := 225 -- in meters

-- Problem Statement
theorem time_to_cross_bridge : (length_of_train + length_of_bridge) / speed_of_train = 30 := by
  sorry

end time_to_cross_bridge_l769_769171


namespace negation_of_exists_l769_769450

-- Lean definition of the proposition P
def P (a : ℝ) : Prop :=
  ∃ x0 : ℝ, x0 > 0 ∧ 2^x0 * (x0 - a) > 1

-- The negation of the proposition P
def neg_P (a : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → 2^x * (x - a) ≤ 1

-- Theorem stating that the negation of P is neg_P
theorem negation_of_exists (a : ℝ) : ¬ P a ↔ neg_P a :=
by
  -- (Proof to be provided)
  sorry

end negation_of_exists_l769_769450


namespace find_min_value_of_lambda_plus_2mu_l769_769854

-- We define the points and vectors based on given conditions
variables {A B C P M N : Type} [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] 
          [Module ℝ A] [Module ℝ B] [Module ℝ C]

-- We define the variables λ and μ
variables (λ μ : ℝ) (hλ : λ > 0) (hμ : μ > 0)

-- Given conditions in terms of vector operations
variables (h1 : B -ᵥ P = (1 / 2 : ℝ) • (C -ᵥ P))
variables (h2 : M -ᵥ A = λ • (B -ᵥ A))
variables (h3 : N -ᵥ A = μ • (C -ᵥ A))

-- The goal to prove
theorem find_min_value_of_lambda_plus_2mu (h1 : B -ᵥ P = (1 / 2 : ℝ) • (C -ᵥ P))
                                          (h2 : M -ᵥ A = λ • (B -ᵥ A))
                                          (h3 : N -ᵥ A = μ • (C -ᵥ A)) 
                                          (hλ : λ > 0) (hμ : μ > 0) :
  λ + 2 * μ = 8 / 3 :=
sorry

end find_min_value_of_lambda_plus_2mu_l769_769854


namespace distance_in_part_exists_l769_769543

open set

-- Define the square of area 1 km²
def square := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Given a division of the square into three parts A, B, and C
variables (A B C : set (ℝ × ℝ))

-- Prove that there's always a pair of points within the same part with distance at least √(65 / 64)
theorem distance_in_part_exists (hA : A ⊆ square) (hB : B ⊆ square) (hC : C ⊆ square) (h_partition : square = A ∪ B ∪ C) (h_disjoint : A ∩ B = ∅ ∧ B ∩ C = ∅ ∧ A ∩ C = ∅) :
  ∃ (P Q : ℝ × ℝ) (part : set (ℝ × ℝ)), part ∈ {A, B, C} ∧ P ∈ part ∧ Q ∈ part ∧ dist P Q ≥ real.sqrt (65 / 64) :=
sorry

end distance_in_part_exists_l769_769543


namespace find_binomial_params_l769_769648

noncomputable def binomial_params (n p : ℝ) := 2.4 = n * p ∧ 1.44 = n * p * (1 - p)

theorem find_binomial_params (n p : ℝ) (h : binomial_params n p) : n = 6 ∧ p = 0.4 :=
by
  sorry

end find_binomial_params_l769_769648


namespace jia_passes_l769_769169

/-- 10 possible questions, Jia can answer 5 correctly, selects 3 at random, needs 2 correct to pass -/
def jia_passing_probability : Prop :=
  let correct_questions := 5
  let total_questions := 10
  let selected_questions := 3
  let min_correct_to_pass := 2
  let total_combinations := Nat.choose total_questions selected_questions
  let combinations_for_two_correct := (Nat.choose correct_questions 2) * (Nat.choose (total_questions - correct_questions) 1)
  let combinations_for_three_correct := Nat.choose correct_questions 3
  (combinations_for_two_correct + combinations_for_three_correct) / total_combinations = 1 / 2

theorem jia_passes :
  jia_passing_probability :=
begin
  sorry
end

end jia_passes_l769_769169


namespace count_N_lt_500_solution_exists_l769_769286

theorem count_N_lt_500_solution_exists:
  ∃ (N : ℕ), N < 500 ∧ (∃ (x : ℝ), x^floor x = N) = 287 :=
sorry

end count_N_lt_500_solution_exists_l769_769286


namespace count_positive_factors_of_180_multiple_of_15_l769_769722

statement: ℕ
statement := sorry

theorem count_positive_factors_of_180_multiple_of_15 :
  let factors_of_180 := {d ∈ (finset.range 181).filter (λ x, 180 % x = 0)}
  let multiples_of_15 := (finset.range 181).filter (λ x, x % 15 = 0)
  let desired_factors := factors_of_180 ∩ multiples_of_15
  desired_factors.card = 6 :=
by
  sorry

end count_positive_factors_of_180_multiple_of_15_l769_769722


namespace correct_operation_l769_769490

variable {a b : ℝ}

def conditionA : Prop := a^2 + a^3 = a^6
def conditionB : Prop := (a * b)^2 = a * b^2
def conditionC : Prop := (a + b)^2 = a^2 + b^2
def conditionD : Prop := (a + b) * (a - b) = a^2 - b^2

theorem correct_operation : conditionD ∧ ¬conditionA ∧ ¬conditionB ∧ ¬conditionC := by
  sorry

end correct_operation_l769_769490


namespace lines_intersection_l769_769617

theorem lines_intersection : 
  (∃ x y : ℝ, 3 * y = -2 * x + 6 ∧ 2 * y = -6 * x + 4 ∧ x = 0 ∧ y = 2) :=
begin
  sorry
end

end lines_intersection_l769_769617


namespace expand_expression_l769_769218

theorem expand_expression :
  (3 * t^2 - 2 * t + 3) * (-2 * t^2 + 3 * t - 4) = -6 * t^4 + 13 * t^3 - 24 * t^2 + 17 * t - 12 :=
by sorry

end expand_expression_l769_769218


namespace fraction_defined_iff_l769_769846

theorem fraction_defined_iff (x : ℝ) : (1 / (x + 1) ≠ ⊥) ↔ x ≠ -1 := 
by
  sorry

end fraction_defined_iff_l769_769846


namespace factory_car_production_l769_769549

theorem factory_car_production :
  let cars_yesterday := 60
  let cars_today := 2 * cars_yesterday
  let total_cars := cars_yesterday + cars_today
  total_cars = 180 :=
by
  sorry

end factory_car_production_l769_769549


namespace solve_complex_addition_l769_769312

def complex_addition_problem : Prop :=
  let B := Complex.mk 3 (-2)
  let Q := Complex.mk (-5) 1
  let R := Complex.mk 1 (-2)
  let T := Complex.mk 4 3
  B - Q + R + T = Complex.mk 13 (-2)

theorem solve_complex_addition : complex_addition_problem := by
  sorry

end solve_complex_addition_l769_769312


namespace count_valid_N_under_500_l769_769300

def hasSolution (N : ℕ) (x : ℝ) : Prop :=
  N = x ^ (Real.floor x)

def validN (N : ℕ) : Prop :=
  ∃ x : ℝ, hasSolution N x

theorem count_valid_N_under_500 : 
  let N_set := {N : ℕ | N < 500 ∧ validN N}
  N_set.card = 287 := sorry

end count_valid_N_under_500_l769_769300


namespace min_red_vertex_proof_l769_769598

-- Define a Cube with vertices colored either red or blue
structure Cube (α : Type) [DecidableEq α] :=
(colored_vertices : α → Fin 8 → Prop)

-- Define the condition that each face of the cube must have at least one red vertex
def face_has_red_vertex {α : Type} [DecidableEq α] (cube : Cube α) : Prop :=
∀ (f : Fin 6), ∃ (v : Fin 4), cube.colored_vertices v f.val

-- Define the function to count red vertices
def count_red_vertices {α : Type} [DecidableEq α] (cube : Cube α) : ℕ :=
Finset.card {v : Fin 8 | cube.colored_vertices v 0 }

-- Define the minimum red vertex condition
def min_red_vertices_condition (cube : Cube ℕ) : Prop :=
count_red_vertices cube = 2

-- The theorem stating what we need to prove
theorem min_red_vertex_proof (cube : Cube ℕ) :
  face_has_red_vertex cube → min_red_vertices_condition cube :=
sorry

end min_red_vertex_proof_l769_769598


namespace factors_of_180_multiple_of_15_count_l769_769807

theorem factors_of_180_multiple_of_15_count :
  ∃ n : Nat, n = 6 ∧ ∀ x : Nat, x ∣ 180 → 15 ∣ x → x > 0 :=
by
  sorry

end factors_of_180_multiple_of_15_count_l769_769807


namespace greatest_number_dividing_4351_and_5161_with_remainders_8_and_10_l769_769120

theorem greatest_number_dividing_4351_and_5161_with_remainders_8_and_10 :
  ∃ d : ℕ, (d ∣ (4351 - 8)) ∧ (d ∣ (5161 - 10)) ∧
  ∀ k : ℕ, (k ∣ (4351 - 8)) ∧ (k ∣ (5161 - 10)) → k ≤ 1 :=
begin
  sorry
end

end greatest_number_dividing_4351_and_5161_with_remainders_8_and_10_l769_769120


namespace inequality_proof_l769_769637

theorem inequality_proof (a b m n p : ℝ) (h1 : a > b) (h2 : m > n) (h3 : p > 0) : n - a * p < m - b * p :=
sorry

end inequality_proof_l769_769637


namespace highest_price_minimum_sales_volume_l769_769521

section problem1

variable (x : ℝ) (sales_volume : ℝ)

-- Conditions from the problem
def initial_price := 25
def initial_sales_volume := 80000
def price_increase_impact := 2000

-- Highest price per unit without reducing total sales revenue
def new_sales_volume := initial_sales_volume - price_increase_impact * (x - initial_price)
def total_sales_revenue := new_sales_volume * x
def original_revenue := initial_price * initial_sales_volume

theorem highest_price (hx : 25 ≤ x ∧ x ≤ 40) : original_revenue ≤ total_sales_revenue := sorry

end problem1


section problem2

variable (x : ℝ) (a : ℝ)

-- Conditions from the problem
def original_price := 25
def original_sales_volume := 80 -- in ten thousand units
def original_income := original_price * original_sales_volume

def technical_reform_fees := (1/6) * (x^2 - 600)
def fixed_promotion_fees := 50
def variable_promotion_fees := (1/5) * x
def total_investment := technical_reform_fees + fixed_promotion_fees + variable_promotion_fees

def next_years_revenue := a * x

theorem minimum_sales_volume (hx : x > 25 ∧ x = 30) (ha : a ≥ 10.2):
  next_years_revenue ≥ original_income + total_investment := sorry

end problem2

end highest_price_minimum_sales_volume_l769_769521


namespace factors_of_180_multiple_of_15_l769_769750

/-- A factor of a number n is a positive integer which divides n without leaving a remainder. -/
def is_factor (n k : ℕ) : Prop :=
k > 0 ∧ n % k = 0

/-- A multiple of a number m is an integer which is divisible by m without leaving a remainder. -/
def is_multiple (k m : ℕ) : Prop :=
k % m = 0

/-- There are 6 factors of 180 that are also multiples of 15. -/
theorem factors_of_180_multiple_of_15 : 
  (Finset.filter (λ k, is_multiple k 15) { x | is_factor 180 x }.to_finset).card = 6 :=
by 
  sorry

end factors_of_180_multiple_of_15_l769_769750


namespace length_of_floor_l769_769126

-- Definitions
def rate_per_sq_meter : ℝ := 3
def total_cost : ℝ := 400
def area : ℝ := total_cost / rate_per_sq_meter
def breadth : ℝ := real.sqrt (total_cost / (3 * rate_per_sq_meter))
def length : ℝ := 3 * breadth

-- Theorem statement
theorem length_of_floor : length = 20 :=
by
  sorry

end length_of_floor_l769_769126


namespace angle_AEB_is_90_degrees_l769_769523

open EuclideanGeometry

-- Conditions: A square ABCD, a circle γ with radius 6, center at C, and passes through B and D. AC is extended to E.
structure Square (A B C D : Point) : Prop :=
(square : is_square A B C D)

structure Circle (C : Point) (r : ℝ) (B D : Point) : Prop :=
(circle : is_circle C r)
(on_circle_B : on_circle C r B)
(on_circle_D : on_circle C r D)

variable (A B C D E : Point)

-- Conjecture to be proved:
theorem angle_AEB_is_90_degrees 
  (hSquare : Square A B C D)
  (hCircle : Circle C 6 B D)
  (hLineSegment : collinear {A, C, E}) :
  angle A E B = 90 :=
sorry

end angle_AEB_is_90_degrees_l769_769523


namespace intersection_M_N_l769_769700

noncomputable def M : set ℝ := { x | -2 < x ∧ x < 3 }
noncomputable def N : set ℝ := { x | Real.log (x + 2) ≥ 0 }

theorem intersection_M_N :
  M ∩ N = { x | -1 ≤ x ∧ x < 3 } := sorry

end intersection_M_N_l769_769700


namespace circle_people_count_l769_769124

def num_people (n : ℕ) (a b : ℕ) : Prop :=
  a = 7 ∧ b = 18 ∧ (b = a + (n / 2))

theorem circle_people_count (n : ℕ) (a b : ℕ) (h : num_people n a b) : n = 24 :=
by
  sorry

end circle_people_count_l769_769124


namespace peter_has_4_finches_l769_769026

variable (parakeet_eats_per_day : ℕ) (parrot_eats_per_day : ℕ) (finch_eats_per_day : ℕ)
variable (num_parakeets : ℕ) (num_parrots : ℕ) (num_finches : ℕ)
variable (total_birdseed : ℕ)

theorem peter_has_4_finches
    (h1 : parakeet_eats_per_day = 2)
    (h2 : parrot_eats_per_day = 14)
    (h3 : finch_eats_per_day = 1)
    (h4 : num_parakeets = 3)
    (h5 : num_parrots = 2)
    (h6 : total_birdseed = 266)
    (h7 : total_birdseed = (num_parakeets * parakeet_eats_per_day + num_parrots * parrot_eats_per_day) * 7 + num_finches * finch_eats_per_day * 7) :
    num_finches = 4 :=
by
  sorry

end peter_has_4_finches_l769_769026


namespace no_palindrome_year_product_l769_769205

def is_palindrome (n : ℕ) : Prop :=
  n.toString = n.toString.reverse

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def is_palindromic_prime (n : ℕ) : Prop :=
  is_palindrome n ∧ is_prime n

def in_range (y : ℕ) : Prop :=
  2000 ≤ y ∧ y < 3000

theorem no_palindrome_year_product :
  ∀ y, in_range y → is_palindrome y →
  ¬ ∃ (p q : ℕ), is_palindromic_prime p ∧ is_palindromic_prime q ∧ y = p * q :=
by sorry

end no_palindrome_year_product_l769_769205


namespace equilateral_triangle_on_same_branch_impossible_find_coordinates_of_equilateral_vertices_on_different_branches_l769_769200

-- Define the hyperbola, branches, and conditions for equilateral triangle vertices
def hyperbola (x y : ℝ) : Prop := x * y = 1
def branch1 (P : ℝ × ℝ) : Prop := P.1 > 0 ∧ P.2 > 0 ∧ hyperbola P.1 P.2
def branch2 (P : ℝ × ℝ) : Prop := P.1 < 0 ∧ P.2 < 0 ∧ hyperbola P.1 P.2

-- Formalization of the proof that an equilateral triangle cannot have all vertices on the same branch
theorem equilateral_triangle_on_same_branch_impossible (P Q R : ℝ × ℝ) :
  branch1 P ∧ branch1 Q ∧ branch1 R ∧
  (dist P Q = dist Q R ∧ dist Q R = dist R P ∧ dist R P = dist P Q) → false :=
sorry

-- Formalization of finding specific coordinates of Q and R given P on the opposite branch
theorem find_coordinates_of_equilateral_vertices_on_different_branches (Q R : ℝ × ℝ) :
  let P : ℝ × ℝ := (-1, -1)
  in branch2 P ∧ branch1 Q ∧ branch1 R ∧
     (dist P Q = dist Q R ∧ dist Q R = dist R P ∧ dist R P = dist P Q) → 
     Q = (2 - Real.sqrt 3, 2 + Real.sqrt 3) ∧ R = (2 + Real.sqrt 3, 2 - Real.sqrt 3) :=
sorry

end equilateral_triangle_on_same_branch_impossible_find_coordinates_of_equilateral_vertices_on_different_branches_l769_769200


namespace positive_factors_of_180_multiple_of_15_count_l769_769754

theorem positive_factors_of_180_multiple_of_15_count : 
  let factorization_180 := 2^2 * 3^2 * 5;
  let is_multiple_of_15 (n : ℕ) := ∃ k1 k2 : ℕ, k1 * 3 = n ∧ k2 * 5 = n;
  let factors_180 := { n : ℕ | n ∣ factorization_180 };
  { n : ℕ | n ∈ factors_180 ∧ is_multiple_of_15 n }.card = 6 :=
by
  sorry

end positive_factors_of_180_multiple_of_15_count_l769_769754


namespace tiger_catch_distance_correct_l769_769170

noncomputable def tiger_catch_distance (tiger_leaps_behind : ℕ) (tiger_leaps_per_minute : ℕ) (deer_leaps_per_minute : ℕ) (tiger_m_per_leap : ℕ) (deer_m_per_leap : ℕ) : ℕ :=
  let initial_distance := tiger_leaps_behind * tiger_m_per_leap
  let tiger_per_minute := tiger_leaps_per_minute * tiger_m_per_leap
  let deer_per_minute := deer_leaps_per_minute * deer_m_per_leap
  let gain_per_minute := tiger_per_minute - deer_per_minute
  let time_to_catch := initial_distance / gain_per_minute
  time_to_catch * tiger_per_minute

theorem tiger_catch_distance_correct :
  tiger_catch_distance 50 5 4 8 5 = 800 :=
by
  -- This is the placeholder for the proof.
  sorry

end tiger_catch_distance_correct_l769_769170


namespace count_positive_factors_of_180_multiple_of_15_l769_769724

statement: ℕ
statement := sorry

theorem count_positive_factors_of_180_multiple_of_15 :
  let factors_of_180 := {d ∈ (finset.range 181).filter (λ x, 180 % x = 0)}
  let multiples_of_15 := (finset.range 181).filter (λ x, x % 15 = 0)
  let desired_factors := factors_of_180 ∩ multiples_of_15
  desired_factors.card = 6 :=
by
  sorry

end count_positive_factors_of_180_multiple_of_15_l769_769724


namespace percentage_change_in_area_halved_length_tripled_breadth_l769_769428

def original_area (L B : ℝ) : ℝ := L * B
def new_area (L B : ℝ) : ℝ := (L / 2) * (3 * B)

def percentage_change (A1 A2 : ℝ) : ℝ :=
  ((A2 - A1) / A1) * 100

theorem percentage_change_in_area_halved_length_tripled_breadth
  (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  percentage_change (original_area L B) (new_area L B) = 50 :=
begin
  sorry
end

end percentage_change_in_area_halved_length_tripled_breadth_l769_769428


namespace part_a_impossible_part_b_possible_l769_769653

-- Statement for part (a)
theorem part_a_impossible (table : ℕ → ℕ → bool) (n : ℕ) 
    (odd_row_count : fin n → bool) (odd_col_count : fin n → bool)
    (odd_count_r : ℕ) (odd_count_c : ℕ) (cross_in_cell : fin n → fin n → bool) :
    (∀ r : fin n, odd_row_count r = odd (fin n) (\sum c : fin n, cross_in_cell r c)) →
    (∀ c : fin n, odd_col_count c = odd (fin n) (\sum r : fin n, cross_in_cell r c)) →
    odd_count_r = 20 → odd_count_c = 15 → False :=
sorry

-- Statement for part (b)
theorem part_b_possible (table : ℕ → ℕ → bool) 
    (n : ℕ) (cross_count : ℕ) (row_count : fin n → ℕ) (col_count : fin n → ℕ)
    (cross_in_cell : fin n → fin n → bool) :
    n = 16 → cross_count = 126 →
    (∀ r : fin n, odd (row_count r)) →
    (∀ c : fin n, odd (col_count c)) →
    (∃ table, (∀ r c, cross_in_cell r c = (table r c)) ∧ (\sum r, row_count r = 126) ∧ (\sum c, col_count c = 126)) :=
sorry

end part_a_impossible_part_b_possible_l769_769653


namespace find_alpha_polar_equation_l769_769633

noncomputable def alpha := (3 * Real.pi) / 4

theorem find_alpha (P : ℝ × ℝ) (l : ℝ → ℝ × ℝ) (A B : ℝ × ℝ)
  (hP : P = (2, 1))
  (hl_def : ∀ t, l t = (2 + t * Real.cos alpha, 1 + t * Real.sin alpha))
  (hA : ∃ t, l t = (A.1, 0) ∧ A.1 > 0)
  (hB : ∃ t, l t = (0, B.2) ∧ B.2 > 0)
  (h_cond : dist P A * dist P B = 4) : alpha = (3 * Real.pi) / 4 :=
sorry

theorem polar_equation (l : ℝ → ℝ × ℝ)
  (hl_def : ∀ t, l t = (2 + t * Real.cos alpha, 1 + t * Real.sin alpha))
  (h_alpha : alpha = (3 * Real.pi) / 4)
  (h_polar : ∀ ρ θ, l ρ = (ρ * Real.cos θ, ρ * Real.sin θ))
  : ∀ ρ θ, ρ * (Real.cos θ + Real.sin θ) = 3 :=
sorry

end find_alpha_polar_equation_l769_769633


namespace part_a_impossible_part_b_possible_l769_769650

-- Statement for part (a)
theorem part_a_impossible (table : ℕ → ℕ → bool) (n : ℕ) 
    (odd_row_count : fin n → bool) (odd_col_count : fin n → bool)
    (odd_count_r : ℕ) (odd_count_c : ℕ) (cross_in_cell : fin n → fin n → bool) :
    (∀ r : fin n, odd_row_count r = odd (fin n) (\sum c : fin n, cross_in_cell r c)) →
    (∀ c : fin n, odd_col_count c = odd (fin n) (\sum r : fin n, cross_in_cell r c)) →
    odd_count_r = 20 → odd_count_c = 15 → False :=
sorry

-- Statement for part (b)
theorem part_b_possible (table : ℕ → ℕ → bool) 
    (n : ℕ) (cross_count : ℕ) (row_count : fin n → ℕ) (col_count : fin n → ℕ)
    (cross_in_cell : fin n → fin n → bool) :
    n = 16 → cross_count = 126 →
    (∀ r : fin n, odd (row_count r)) →
    (∀ c : fin n, odd (col_count c)) →
    (∃ table, (∀ r c, cross_in_cell r c = (table r c)) ∧ (\sum r, row_count r = 126) ∧ (\sum c, col_count c = 126)) :=
sorry

end part_a_impossible_part_b_possible_l769_769650


namespace transformed_mean_variance_l769_769850

variables {n : ℕ} {x : ℕ → ℝ}
variable (mean_var : (ℝ × ℝ))
noncomputable def mean (x : ℕ → ℝ) (n : ℕ) : ℝ :=
  (∑ i in Finset.range n, x i) / n

noncomputable def variance (x : ℕ → ℝ) (mean : ℝ) (n : ℕ) : ℝ :=
  (∑ i in Finset.range n, (x i - mean) ^ 2) / n

theorem transformed_mean_variance (mean_var) :
  let (mean_x, var_x) := mean_var in
  mean x n = mean_x →
  variance x mean_x n = var_x →
  mean (λ k, 2 * x k + 3) n = 2 * mean_x + 3 ∧
  variance (λ k, 2 * x k + 3) (2 * mean_x + 3) n = 4 * var_x :=
by
  intros mean_x var_x h_mean h_var
  sorry

end transformed_mean_variance_l769_769850


namespace prod_geq_three_pow_n_l769_769641

theorem prod_geq_three_pow_n
  (n : ℕ)
  (a : Fin n → ℝ)
  (h_pos : ∀ i, 0 < a i)
  (h_prod : (∏ i, a i) = 1) :
  (∏ i, 2 + a i) ≥ 3 ^ n :=
by
  sorry

end prod_geq_three_pow_n_l769_769641


namespace robinson_determines_day_within_4days_l769_769041

-- Define the days of the week
inductive DayOfWeek
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

-- Define the behavior of Friday
def tells_truth_on_friday (d : DayOfWeek) : Prop :=
  if d = DayOfWeek.Friday then True else False

-- Define the behavior of Friday answering questions (truth on Friday, lies on other days)
def friday_answer (d today : DayOfWeek) : Prop :=
  if tells_truth_on_friday(today) then d = today else d ≠ today

-- The main theorem to prove that Robinson can determine the day of the week within 4 days
theorem robinson_determines_day_within_4days :
  ∀ (start_day : DayOfWeek), ∃ days n, n ≤ 4 ∧ ∀ today, (today = start_day + n) → 
  (∃ d, friday_answer d today) -> d = today :=
sorry

end robinson_determines_day_within_4days_l769_769041


namespace smallest_positive_period_and_range_sin_2x0_if_zero_of_f_l769_769683

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sin x)^2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x - (1 / 2) * Real.cos (2 * x)

theorem smallest_positive_period_and_range :
  (∀ x, f (x + Real.pi) = f x) ∧ (Set.range f = Set.Icc (-3 / 2) (5 / 2)) :=
by
  sorry

theorem sin_2x0_if_zero_of_f (x0 : ℝ) (hx0 : 0 ≤ x0 ∧ x0 ≤ Real.pi / 2)
  (hf : f x0 = 0) : Real.sin (2 * x0) = (Real.sqrt 15 - Real.sqrt 3) / 8 :=
by
  sorry

end smallest_positive_period_and_range_sin_2x0_if_zero_of_f_l769_769683


namespace area_of_square_l769_769392

theorem area_of_square (x1 y1 x2 y2 : ℝ)
  (h1 : (x1, y1) = (1, -2))
  (h2 : (x2, y2) = (4, 1))
  (adjacent : dist (x1, y1) (x2, y2) = 3 * real.sqrt 2) :
  (3 * real.sqrt 2) ^ 2 = 18 := by
  sorry

end area_of_square_l769_769392


namespace count_factors_of_180_multiple_of_15_l769_769777

def is_factor (n k : ℕ) : Prop := n % k = 0

def is_multiple_of (k m : ℕ) : Prop := k % m = 0

theorem count_factors_of_180_multiple_of_15 : 
  (finset.filter (λ f, is_factor 180 f ∧ is_multiple_of f 15) (finset.range (181))).card = 6 := 
by 
sorry

end count_factors_of_180_multiple_of_15_l769_769777


namespace positive_factors_of_180_multiple_of_15_count_l769_769760

theorem positive_factors_of_180_multiple_of_15_count : 
  let factorization_180 := 2^2 * 3^2 * 5;
  let is_multiple_of_15 (n : ℕ) := ∃ k1 k2 : ℕ, k1 * 3 = n ∧ k2 * 5 = n;
  let factors_180 := { n : ℕ | n ∣ factorization_180 };
  { n : ℕ | n ∈ factors_180 ∧ is_multiple_of_15 n }.card = 6 :=
by
  sorry

end positive_factors_of_180_multiple_of_15_count_l769_769760


namespace arrange_snow_leopards_l769_769381

theorem arrange_snow_leopards :
  let n := 9 -- number of leopards
  let factorial x := (Nat.factorial x) -- definition for factorial
  let tall_short_perm := 2 -- there are 2 ways to arrange the tallest and shortest leopards at the ends
  tall_short_perm * factorial (n - 2) = 10080 := by sorry

end arrange_snow_leopards_l769_769381


namespace part_a_impossibility_l769_769661

-- Define the number of rows and columns
def num_rows : ℕ := 20
def num_columns : ℕ := 15

-- Define a function that checks if the sum of the counts in rows and columns match the conditions
def is_possible_configuration : Prop :=
  (num_rows % 2 = 0) ∧ (num_columns % 2 = 1)

theorem part_a_impossibility : ¬ is_possible_configuration :=
by
  -- The proof for the contradiction will go here
  sorry

end part_a_impossibility_l769_769661


namespace Vec_dot_product_eq_sqrt2_l769_769912

variables {ℝ : Type*} [field ℝ] [metric_space ℝ]

def vec_norm (v : ℝ → ℝ) : ℝ := real.sqrt (v 1 ^ 2 + v 2 ^ 2)
def vec_dot (a b : ℝ → ℝ) : ℝ := (a 1 * b 1) + (a 2 * b 2)
def f (a b : ℝ → ℝ) (x : ℝ) : ℝ := vec_norm (λ i, a i + x * b i)

noncomputable def vec_a := (λ i : ℝ, if i = 1 then 2 else 2)
noncomputable def vec_b := (λ i : ℝ, if i = 1 then 2 else 2)

theorem Vec_dot_product_eq_sqrt2 (h1: vec_norm vec_a = real.sqrt 2) 
    (h2 : vec_norm vec_b = real.sqrt 2) 
    (h3 : ∃ x: ℝ, f vec_a vec_b x = 1) : vec_dot vec_a vec_b = real.sqrt 2 ∨ vec_dot vec_a vec_b = -real.sqrt 2 :=
sorry

end Vec_dot_product_eq_sqrt2_l769_769912


namespace price_per_liter_after_discount_l769_769956

-- Define the initial conditions
def num_bottles : ℕ := 6
def liters_per_bottle : ℝ := 2
def original_total_cost : ℝ := 15
def discounted_total_cost : ℝ := 12

-- Calculate the total number of liters
def total_liters : ℝ := num_bottles * liters_per_bottle

-- Define the expected price per liter after discount
def expected_price_per_liter : ℝ := 1

-- Lean query to verify the expected price per liter
theorem price_per_liter_after_discount : (discounted_total_cost / total_liters) = expected_price_per_liter := by
  sorry

end price_per_liter_after_discount_l769_769956


namespace ratio_of_ages_l769_769476

variable (V B : ℕ)
variable (hSum : V + B = 32)
variable (hB : B = 10)
variable (M Y : ℕ)
variable (hViggoAgeEquation : 14 = 2 * M + Y)

theorem ratio_of_ages (hV := hSum ▸ hB ▸ nat.sub_add_cancel 10 (hSum ▸ hB ▸ nat.le_add_left 10 V)) :
  (14 / 2) = 7 :=
by
  have hV_curr := hSum ▸ hB
  have hV_value : V = 22 := by
    rw [hB, add_comm] at hV_curr
    exact nat.add_right_cancel hV_curr

  have hDiff : V - B = 12 := by
    rw [hV_value, hB]
    exact rfl

  have hV_past : V - 10 + 2 = 14 := by
    rw [hV_value]
    exact rfl

  have hEquation : 2 * 7 + 0 = 14 := by rfl

  exact nat.div_eq_of_eq_mul_left (nat.pos_of_gt (nat.zero_lt_bit0 (nat.zero_lt_succ 1))) (by linarith)

end ratio_of_ages_l769_769476


namespace solve_for_x_and_calculate_l769_769829

theorem solve_for_x_and_calculate (x y : ℚ) 
  (h1 : 102 * x - 5 * y = 25) 
  (h2 : 3 * y - x = 10) : 
  10 - x = 2885 / 301 :=
by 
  -- These proof steps would solve the problem and validate the theorem
  sorry

end solve_for_x_and_calculate_l769_769829


namespace alissa_earrings_count_l769_769188

theorem alissa_earrings_count:
  let total_pairs := 12 in
  let individual_earrings := total_pairs * 2 in
  let given_earrings := individual_earrings / 2 in
  3 * given_earrings = 36 :=
by
  sorry

end alissa_earrings_count_l769_769188


namespace part_a_part_b_l769_769399

-- Definitions for part (a)
def log2_pi_a (a : ℝ) : ℝ := 2^a - Real.pi

def log5_pi_b (b : ℝ) : ℝ := 5^b - Real.pi

-- Proof statement for part (a)
theorem part_a (a b : ℝ) : log2_pi_a a = 0 → log5_pi_b b = 0 → (1 / a + 1 / b > 2) := by
  sorry

-- Definitions for part (b)
def log2_pi_a (a : ℝ) : ℝ := 2^a - Real.pi

def logpi2_b (b : ℝ) : ℝ := Real.pi^b - 2

-- Proof statement for part (b)
theorem part_b (a b : ℝ) : log2_pi_a a = 0 → logpi2_b b = 0 → (1 / a + 1 / b > 2) := by
  sorry

end part_a_part_b_l769_769399


namespace circles_5_and_8_same_color_l769_769030

-- Define the circles and colors
inductive Color
  | red
  | yellow
  | blue

def circles : Nat := 8

-- Define the adjacency relationship (i.e., directly connected)
-- This is a placeholder. In practice, this would be defined based on the problem's diagram.
def directly_connected (c1 c2 : Nat) : Prop := sorry

-- Simulate painting circles with given constraints
def painted (c : Nat) : Color := sorry

-- Define the conditions
axiom paint_condition (c1 c2 : Nat) (h : directly_connected c1 c2) : painted c1 ≠ painted c2

-- The proof problem: show that circles 5 and 8 must be painted the same color
theorem circles_5_and_8_same_color : painted 5 = painted 8 := 
sorry

end circles_5_and_8_same_color_l769_769030


namespace percentage_change_area_l769_769436

theorem percentage_change_area (L B : ℝ) (h : L > 0 ∧ B > 0) :
  let A1 := L * B
  let L2 := L / 2
  let B2 := 3 * B
  let A2 := L2 * B2
  let percentage_change := ((A2 - A1) / A1) * 100
  percentage_change = 50 := by
  have A1_def := calc
    A1 = L * B : rfl
  have A2_def := calc
    A2 = (L / 2) * (3 * B) : rfl
    ... = (3 / 2) * (L * B) : by ring
  have pc_def := calc
    percentage_change = ((A2 - A1) / A1) * 100 : rfl
    ... = (( (3 / 2) * (L * B) - (L * B) ) / (L * B) ) * 100 : by rw [A1_def, A2_def]
    ... = (1 / 2) * 100 : by ring
  exact pc_def

end percentage_change_area_l769_769436


namespace Hector_gumballs_l769_769282

theorem Hector_gumballs :
  ∃ (total_gumballs : ℕ)
  (gumballs_Todd : ℕ) (gumballs_Alisha : ℕ) (gumballs_Bobby : ℕ) (gumballs_remaining : ℕ),
  gumballs_Todd = 4 ∧
  gumballs_Alisha = 2 * gumballs_Todd ∧
  gumballs_Bobby = 4 * gumballs_Alisha - 5 ∧
  gumballs_remaining = 6 ∧
  total_gumballs = gumballs_Todd + gumballs_Alisha + gumballs_Bobby + gumballs_remaining ∧
  total_gumballs = 45 :=
by
  sorry

end Hector_gumballs_l769_769282


namespace fourth_card_value_l769_769995

noncomputable def card_numbers : List ℚ :=
  [17 / 10, 1 / 5, 1 / 5, 1, 3 / 5, 3 / 8, 14 / 10]

def fourth_card (cards : List ℚ) : ℚ :=
  (cards.sort (· > ·)).getD 3 0

theorem fourth_card_value :
  fourth_card card_numbers = 3 / 5 :=
by
  sorry

end fourth_card_value_l769_769995


namespace length_of_platform_l769_769558

variables (a : ℝ) (L_p : ℝ)
-- Initial speed of the train in m/s
def initial_speed : ℝ := 15
-- Time to pass the man in seconds
def t1 : ℝ := 20
-- Time to pass the platform in seconds
def t2 : ℝ := 22

-- Equation for the length of the train
def L_t : ℝ := initial_speed * t1 + 0.5 * a * t1^2

-- Equation for the length of the train plus the platform
def L_t_plus_L_p : ℝ := initial_speed * t2 + 0.5 * a * t2^2

-- Expression for the length of the platform in terms of a
def L_p_formula : ℝ := 30 + 42 * a

-- Final theorem statement
theorem length_of_platform (h : L_t + L_p = L_t_plus_L_p) : L_p = L_p_formula :=
sorry

end length_of_platform_l769_769558


namespace projection_of_vector_2a_b_eq_7_l769_769265

noncomputable def vec_projection (a b : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  let angle_ab := Real.cos (Real.pi / 3)
  let dot_ab := ∥a∥ * ∥b∥ * angle_ab
  let dot_proj := 2 * ∥a∥^2 + dot_ab
  dot_proj / ∥a∥

theorem projection_of_vector_2a_b_eq_7 (a b : EuclideanSpace ℝ (Fin 3))
  (ha : ∥a∥ = 2) (hb : ∥b∥ = 6) (angle_ab : Real.angle a b = Real.pi / 3) :
  vec_projection a b = 7 :=
by
  sorry

end projection_of_vector_2a_b_eq_7_l769_769265


namespace count_factors_of_180_multiple_of_15_l769_769715

def is_factor (x n : ℕ) := n % x = 0 -- Definition of a factor
def is_multiple (x k : ℕ) := x % k = 0 -- Definition of a multiple

theorem count_factors_of_180_multiple_of_15 :
  (finset.filter (λ x, is_multiple x 15) (finset.filter (λ x, is_factor x 180) (finset.range (180 + 1)))).card = 6 := by
  sorry

end count_factors_of_180_multiple_of_15_l769_769715


namespace factors_of_180_multiples_of_15_l769_769797

theorem factors_of_180_multiples_of_15 :
  (finset.filter (λ x : ℕ, x % 15 = 0) (finset.filter (λ x : ℕ, 180 % x = 0) (finset.range (180 + 1)))).card = 6 :=
sorry

end factors_of_180_multiples_of_15_l769_769797


namespace bowling_ball_weight_l769_769059

theorem bowling_ball_weight :
  (∃ b k : ℝ, 10 * b = 6 * k ∧ 3 * k = 90) → (∃ b : ℝ, b = 18) :=
by
  intro h
  obtain ⟨b, k, h1, h2⟩ := h
  have hk : k = 30 := by
    linarith
  have hb : b = 18 := by
    rw [hk] at h1
    linarith
  use b
  exact hb

end bowling_ball_weight_l769_769059


namespace lambda_le_zero_l769_769643

theorem lambda_le_zero (λ : ℝ) (h : ∀ n : ℕ, λ < Real.sin (Real.sqrt (4 * n^2 + 1) * Real.pi)) : λ ≤ 0 := 
sorry

end lambda_le_zero_l769_769643


namespace smallest_odd_prime_factor_2021_8_plus_1_l769_769227

noncomputable def least_odd_prime_factor (n : ℕ) : ℕ :=
  if 2021^8 + 1 = 0 then 2021^8 + 1 else sorry 

theorem smallest_odd_prime_factor_2021_8_plus_1 :
  least_odd_prime_factor (2021^8 + 1) = 97 :=
  by
    sorry

end smallest_odd_prime_factor_2021_8_plus_1_l769_769227


namespace distance_QR_eq_15_l769_769952

open EuclideanGeometry

variables {DEF : Triangle} (DE EF DF : ℝ) (Q R : Point)
variable [RightTriangle DEF DE EF DF]

-- Conditions
def is_right_triangle (DEF : Triangle) (DE EF DF : ℝ) : Prop :=
  right_triangle DEF DE EF DF ∧ DE = 9 ∧ EF = 12 ∧ DF = 15

def circle_tangent_to_EF_at_E (Q : Point) (DE EF DF : ℝ) : Prop :=
  ∀ (E D : Point), collinear E F Q ∧ Q = midpoint D F ∧ distance Q E = radius EF Q

def circle_tangent_to_DE_at_D (R : Point) (DE EF DF : ℝ) : Prop :=
  ∀ (D F : Point), collinear D R E ∧ R = midpoint D F ∧ distance R D = radius DE R

-- Goal
theorem distance_QR_eq_15 (h_tr : is_right_triangle DEF DE EF DF)
  (h_Q : circle_tangent_to_EF_at_E Q DE EF DF)
  (h_R : circle_tangent_to_DE_at_D R DE EF DF) :
  distance Q R = 15 :=
sorry

end distance_QR_eq_15_l769_769952


namespace lateral_surface_area_correct_l769_769226

noncomputable def lateral_surface_area_hexagonal_pyramid (h l : ℝ) : ℝ :=
  (3 / 2) * sqrt ((l^2 - h^2) * (3 * l^2 + h^2))

theorem lateral_surface_area_correct (h l : ℝ) :
  lateral_surface_area_hexagonal_pyramid h l = (3 / 2) * sqrt ((l^2 - h^2) * (3 * l^2 + h^2)) := 
sorry

end lateral_surface_area_correct_l769_769226


namespace factors_of_180_multiple_of_15_count_l769_769808

theorem factors_of_180_multiple_of_15_count :
  ∃ n : Nat, n = 6 ∧ ∀ x : Nat, x ∣ 180 → 15 ∣ x → x > 0 :=
by
  sorry

end factors_of_180_multiple_of_15_count_l769_769808


namespace correct_statement_l769_769181

-- Definitions for conditions
def Condition_A : Prop :=
  ∀ (l₁ l₂ l₃ : ℝ → ℝ → ℝ), (∃ p : ℝ × ℝ, p ∈ l₁ ∧ p ∈ l₂) ∧ 
    (∃ q : ℝ × ℝ, q ∈ l₂ ∧ q ∈ l₃) ∧ 
    (∃ r : ℝ × ℝ, r ∈ l₃ ∧ r ∈ l₁) → 
    ∃ (plane : ℝ × ℝ × ℝ → ℝ), ∀ (l : ℝ → ℝ → ℝ), l = l₁ ∨ l = l₂ ∨ l = l₃ → l ⊆ plane

def Condition_B : Prop :=
  ∀ (m : ℝ → ℝ → ℝ) (α : ℝ × ℝ → ℝ), 
    (∃ l : ℝ → ℝ → ℝ, l ⊆ α ∧ m ∥ l) →
    m ∥ α

def Condition_C : Prop :=
  ∀ (α β : ℝ × ℝ × ℝ → ℝ) (l : ℝ → ℝ → ℝ) (P : ℝ × ℝ),
    α ⊥ β ∧ (∀ p, p ∈ α ∧ p ∈ β → p ∈ l) →
    (P ∈ α ∧ P ∉ l) →
    ∃ q, q ∈ α ∧ q ∉ l ∧ q ⊥ β
  
def Condition_D : Prop :=
  ∀ (a b l : ℝ → ℝ → ℝ),
    a ∥ b ∧ l ⊥ a → l ⊥ b

-- Statement to prove
theorem correct_statement : Condition_A → Condition_B → Condition_C → Condition_D → 
  (∃ (correct_option : ℕ), correct_option = 4) := 
by 
  intros _ _ _ hD
  use 4
  exact hD

end correct_statement_l769_769181


namespace find_a7_l769_769277

def sequence (n : ℕ) : ℚ :=
  if n = 0 then -4/3
  else if h : 1 ≤ n then 
    let rec : ℕ → ℚ 
      | 0 => -4/3
      | 1 => -4/3
      | k + 2 => 1 / (rec k + 1)
    rec (n - 1)
  else 0

theorem find_a7 : sequence 6 = 2 :=
sorry

end find_a7_l769_769277


namespace find_lambda_l769_769280

variables (a b : ℝ → ℝ → ℝ) (λ : ℝ)
variables [normed_group (ℝ → ℝ → ℝ)] [normed_space ℝ (ℝ → ℝ → ℝ)]

def orthogonal (v w : ℝ → ℝ → ℝ) : Prop := inner v w = 0

theorem find_lambda
  (ha : ∥a∥ = 2)
  (hb : ∥b∥ = 1)
  (hab_orthogonal : orthogonal a b)
  (hab_lambda_orthogonal : orthogonal (a + b) (a - λ • b))
  : λ = 4 := sorry

end find_lambda_l769_769280


namespace day_365_in_2005_is_tuesday_l769_769833

theorem day_365_in_2005_is_tuesday
  (day15_is_tuesday : (15 % 7 = 1) ∧ (365 % 7 = 1)) 
  : true := 
by
  have day_of_week_of_15 := "Tuesday"
  have day_of_week_of_365 := "Tuesday"
  exact trivial

end day_365_in_2005_is_tuesday_l769_769833


namespace a_2022_eq_674_l769_769479

noncomputable def a : ℕ → ℕ
| 0 := 1
| 1 := 1
| 2 := 1
| (n + 3) := (n + 3) / (a (n + 2) * a (n + 1) * a n)

theorem a_2022_eq_674 : a 2022 = 674 :=
by
  sorry

end a_2022_eq_674_l769_769479


namespace parabola_coefficients_sum_l769_769969

theorem parabola_coefficients_sum (a b c : ℝ) :
  (∀ x y : ℝ, y = a * x^2 + b * x + c ↔ y = a * (x + 3)^2 + 2) ∧
  (-6 = a * (1 + 3)^2 + 2) →
  a + b + c = -11/2 :=
by
  sorry

end parabola_coefficients_sum_l769_769969


namespace factory_car_production_l769_769548

theorem factory_car_production :
  let cars_yesterday := 60
  let cars_today := 2 * cars_yesterday
  let total_cars := cars_yesterday + cars_today
  total_cars = 180 :=
by
  sorry

end factory_car_production_l769_769548


namespace spam_ratio_l769_769571

theorem spam_ratio (total_emails important_emails promotional_fraction promotional_emails spam_emails : ℕ) 
  (h1 : total_emails = 400) 
  (h2 : important_emails = 180) 
  (h3 : promotional_fraction = 2/5) 
  (h4 : total_emails - important_emails = spam_emails + promotional_emails) 
  (h5 : promotional_emails = promotional_fraction * (total_emails - important_emails)) 
  : spam_emails / total_emails = 33 / 100 := 
by {
  sorry
}

end spam_ratio_l769_769571


namespace bushes_needed_l769_769565

theorem bushes_needed
  (num_sides : ℕ) (side_length : ℝ) (bush_fill : ℝ) (total_length : ℝ) (num_bushes : ℕ) :
  num_sides = 3 ∧ side_length = 16 ∧ bush_fill = 4 ∧ total_length = num_sides * side_length ∧ num_bushes = total_length / bush_fill →
  num_bushes = 12 := by
  sorry

end bushes_needed_l769_769565


namespace prob_of_three_successes_correct_l769_769986

noncomputable def prob_of_three_successes (p : ℝ) : ℝ :=
  (Nat.choose 10 3) * (p^3) * (1-p)^7

theorem prob_of_three_successes_correct (p : ℝ) :
  prob_of_three_successes p = (Nat.choose 10 3 : ℝ) * (p^3) * (1-p)^7 :=
by
  sorry

end prob_of_three_successes_correct_l769_769986


namespace minimize_f_l769_769485

noncomputable def f : ℝ → ℝ := λ x => (3/2) * x^2 - 9 * x + 7

theorem minimize_f : ∀ x, f x ≥ f 3 :=
by 
  intro x
  sorry

end minimize_f_l769_769485


namespace sparkling_water_cost_l769_769382

theorem sparkling_water_cost
  (drinks_per_day : ℚ := 1 / 5)
  (bottle_cost : ℝ := 2.00)
  (days_in_year : ℤ := 365) :
  (drinks_per_day * days_in_year) * bottle_cost = 146 :=
by
  sorry

end sparkling_water_cost_l769_769382


namespace sin_2gamma_proof_l769_769393

-- Assume necessary definitions and conditions
variables {A B C D P : Type}
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited P]
variables (a b c d: ℝ)
variables (α β γ: ℝ)

-- Assume points A, B, C, D, P lie on a circle in that order and AB = BC = CD
axiom points_on_circle : a = b ∧ b = c ∧ c = d
axiom cos_apc : Real.cos α = 3/5
axiom cos_bpd : Real.cos β = 1/5

noncomputable def sin_2gamma : ℝ :=
  2 * Real.sin γ * Real.cos γ

-- Statement to prove sin(2 * γ) given the conditions
theorem sin_2gamma_proof : sin_2gamma γ = 8 * Real.sqrt 5 / 25 :=
sorry

end sin_2gamma_proof_l769_769393


namespace hexagon_side_lengths_l769_769203

theorem hexagon_side_lengths (a b c d e f : ℕ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e ∧ e ≠ f)
(h1: a = 7 ∧ b = 5 ∧ (a + b + c + d + e + f = 38)) : 
(a + b + c + d + e + f = 38 ∧ a + b + c + d + e + f = 7 + 7 + 7 + 7 + 5 + 5) → 
(a + b + c + d + e + f = (4 * 7) + (2 * 5)) :=
sorry

end hexagon_side_lengths_l769_769203


namespace max_and_min_values_l769_769229

open Set Function

noncomputable def f : ℝ → ℝ := fun x => -x^2 + 4 * x - 2

theorem max_and_min_values : 
  (sup (f '' (Icc 0 3)) = 2) ∧ (inf (f '' (Icc 0 3)) = -2) :=
by
  sorry

end max_and_min_values_l769_769229


namespace factors_of_180_l769_769770

-- Define a predicate to check if a number is a factor of another number
def is_factor (m n : ℕ) : Prop := n % m = 0

-- Define a predicate to check if a number is a multiple of another number
def is_multiple (m n : ℕ) : Prop := m % n = 0

-- Define the problem
theorem factors_of_180 : {n : ℕ // is_factor n 180 ∧ is_multiple n 15}.card = 6 :=
by sorry

end factors_of_180_l769_769770


namespace magician_tricks_base10_l769_769529

-- Definitions related to problem conditions
def base9Number : List ℕ := [5, 4, 3, 2]
def base9ToBase10 (digs : List ℕ) : ℕ :=
  List.foldr (λ (d : ℕ) (acc_pow_val : ℕ × ℕ),
    let (acc, pow) := acc_pow_val in
    (acc + d * pow, pow * 9)) (0, 1) digs |>.1

-- The proof problem statement
theorem magician_tricks_base10 : base9ToBase10 base9Number = 3998 := by
  sorry

end magician_tricks_base10_l769_769529


namespace childrens_ticket_cost_l769_769477

def cost_of_children (family_pass : ℕ) (adult_ticket : ℕ) (num_children : ℕ) (total_separate : ℕ) : ℕ :=
  (total_separate - adult_ticket) / num_children

theorem childrens_ticket_cost (family_pass : ℕ) (adult_ticket : ℕ) (num_children : ℕ) (total_separate : ℕ) :
  family_pass = 120 → adult_ticket = 35 → num_children = 6 → total_separate = 155 → cost_of_children family_pass adult_ticket num_children total_separate = 20 :=
by
  intros h1 h2 h3 h4
  simp [cost_of_children, h2, h3, h4]
  norm_num


end childrens_ticket_cost_l769_769477


namespace max_volume_pyramid_MNKP_l769_769944

-- Definitions based on the problem conditions
variables {A A1 B B1 C C1 M N K P : Type*}
variable volumeOfPrism : ℝ

-- Assume initial volumes and ratios given in the problem
def ratio_AM_AA1 : ℝ := 1 / 2
def ratio_BN_BB1 : ℝ := 1 / 3
def ratio_CK_CC1 : ℝ := 1 / 4
def volume_prism := 16

-- The main theorem stating the maximum volume of pyramid MNKP
theorem max_volume_pyramid_MNKP
  (h1 : volume_prism = 16)
  (h2 : ratio_AM_AA1 = 1 / 2)
  (h3 : ratio_BN_BB1 = 1 / 3)
  (h4 : ratio_CK_CC1 = 1 / 4) :
  ∃ (V_mnkP : ℝ), V_mnkP = 4 :=
by
  sorry

end max_volume_pyramid_MNKP_l769_769944


namespace calculate_2m_minus_b_l769_769965

/-- Define points A and B -/
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (4, 15)

/-- Define the slope calculation -/
def m : ℝ := (B.snd - A.snd) / (B.fst - A.fst)

/-- Define the y-intercept b calculation -/
def b : ℝ := A.snd - m * A.fst

/-- Define the final calculation and the proof statement -/
theorem calculate_2m_minus_b : 2 * m - b = 9 := by
  -- proof details are omitted
  sorry

end calculate_2m_minus_b_l769_769965


namespace trig_identity_proof_l769_769193

theorem trig_identity_proof :
  (Real.cos (10 * Real.pi / 180) * Real.sin (70 * Real.pi / 180) - Real.cos (80 * Real.pi / 180) * Real.sin (20 * Real.pi / 180)) = (Real.sqrt 3 / 2) :=
by
  sorry

end trig_identity_proof_l769_769193


namespace quadrilateral_inequality_l769_769593

theorem quadrilateral_inequality 
  {A B C D : Type*}
  [IsConvexQuadrilateral A B C D]
  (AB AD AC BC DC : ℝ)
  (hAC : divides AC into equal areas of AB and AD)
  (hAB_gt_AD : AB > AD) :
  BC < DC := sorry

end quadrilateral_inequality_l769_769593


namespace positive_factors_of_180_multiple_of_15_count_l769_769764

theorem positive_factors_of_180_multiple_of_15_count : 
  let factorization_180 := 2^2 * 3^2 * 5;
  let is_multiple_of_15 (n : ℕ) := ∃ k1 k2 : ℕ, k1 * 3 = n ∧ k2 * 5 = n;
  let factors_180 := { n : ℕ | n ∣ factorization_180 };
  { n : ℕ | n ∈ factors_180 ∧ is_multiple_of_15 n }.card = 6 :=
by
  sorry

end positive_factors_of_180_multiple_of_15_count_l769_769764


namespace minimum_grid_n_l769_769331

theorem minimum_grid_n (n : ℕ) (h : ∀ grid : Array (Array ℕ), (∀ i j, grid[i][j] ≠ grid[(i+1) % n][j] ∧ grid[i][j] ≠ grid[i][(j+1) % n]) → ∃ i j, |grid[i][j] - grid[i+1][j]| ≥ 1011 ∨ |grid[i][j] - grid[i][j+1]| ≥ 1011) : n ≥ 2020 := 
sorry

end minimum_grid_n_l769_769331


namespace divides_three_uv_l769_769036

theorem divides_three_uv (u v : ℤ) (h : 9 ∣ u^2 + u * v + v^2) : 3 ∣ u ∧ 3 ∣ v :=
by
  -- Introspect the variables and hypothesis
  intros u v h
  
  -- Add the proof here
  sorry

end divides_three_uv_l769_769036


namespace soccer_team_lineups_l769_769165

open Finset

theorem soccer_team_lineups : 
  let players := range 12
  let quadruplets := {0, 1, 2, 3}  -- Assume Ben, Bob, Bill, Bert are represented by indices 0, 1, 2, and 3
  let others := players \ quadruplets
  let choose_quadruplets := (quadruplets.card.choose 2)
  let choose_others := (others.card.choose 3)
  (choose_quadruplets * choose_others) = 336 :=
by 
  -- Definitions
  let players := range 12
  let quadruplets := {0, 1, 2, 3}
  let others := players \ quadruplets
  let choose_quadruplets := (quadruplets.card.choose 2)
  let choose_others := (others.card.choose 3)
  
  -- Assertions
  have h1 : quadruplets.card = 4 := rfl
  have h2 : others.card = 8 := rfl
  have h3 : choose_quadruplets = 6 := by { rw h1, exact choose_symm _ _ }
  have h4 : choose_others = 56 := by { rw h2, exact choose_symm _ _ }
  
  -- Final proof
  unfold choose_quadruplets choose_others,
  rw [h3, h4],
  norm_num

end soccer_team_lineups_l769_769165


namespace percentage_change_in_area_halved_length_tripled_breadth_l769_769425

def original_area (L B : ℝ) : ℝ := L * B
def new_area (L B : ℝ) : ℝ := (L / 2) * (3 * B)

def percentage_change (A1 A2 : ℝ) : ℝ :=
  ((A2 - A1) / A1) * 100

theorem percentage_change_in_area_halved_length_tripled_breadth
  (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  percentage_change (original_area L B) (new_area L B) = 50 :=
begin
  sorry
end

end percentage_change_in_area_halved_length_tripled_breadth_l769_769425


namespace segment_construction_l769_769563

variable {α : Type*} [Plane α]

-- Define the triangle ABC and points X and Y
variables (A B C X Y : α)
-- Define segments and parallel condition
variable (XY BC AX AY : ℝ)
variable (parallel_bc : parallel XY BC)

/-- In a triangle ABC, with a segment XY parallel to BC intersecting the extended sides at points X and Y, we have BC + XY = AX + AY -/
theorem segment_construction (h1 : ∀ {A B C X Y : α} (ABC : triangle A B C) (parallel_bc : parallel XY BC),
  XY + BC = AX + AY) : Prop := 
    sorry

end segment_construction_l769_769563


namespace day_of_week_365th_day_l769_769838

theorem day_of_week_365th_day (day15_is_tuesday : 15 % 7 = 2) :
    365 % 7 = 2 :=
by
  -- Given that the 15th day falls on a Tuesday, we have 15 ≡ 2 (mod 7)
  have h1 : 15 % 7 = 2 := day15_is_tuesday
  -- Calculating the day of the week for the 365th day
  have h2 : 365 % 7 = (15 + 350) % 7 := by norm_num
  rw [h1, ← nat.add_mod],
  -- Since 350 % 7 = 0, we get the same day as the 15th day
  norm_num

end day_of_week_365th_day_l769_769838


namespace no_prime_numbers_divisible_by_91_l769_769820

-- Define the concept of a prime number.
def is_prime (n : ℕ) : Prop :=
  1 < n ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- Define the factors of 91.
def factors_of_91 (n : ℕ) : Prop :=
  n = 7 ∨ n = 13

-- State the problem formally: there are no prime numbers divisible by 91.
theorem no_prime_numbers_divisible_by_91 :
  ∀ p : ℕ, is_prime p → ¬ (91 ∣ p) :=
by
  intros p prime_p div91
  sorry

end no_prime_numbers_divisible_by_91_l769_769820


namespace area_of_ABCD_l769_769873

noncomputable def area_of_rectangle (length width : ℝ) : ℝ := length * width

theorem area_of_ABCD 
  (A B C D E F : ℝ × ℝ)
  (h1 : similar_rectangles (A, B, C, D) (F, G, E, J))
  (h2 : similar_rectangles (A, B, C, D) (I, G, C, H))
  (dist_AE : dist A E = 3)
  (dist_DF : dist D F = 3)
  (dist_BE : dist B E = 12)
  : area_of_rectangle 15 (5 * √6) = 75 * √6 :=
by 
  sorry

end area_of_ABCD_l769_769873


namespace total_short_trees_l769_769084

def short_trees_initial := 41
def short_trees_planted := 57

theorem total_short_trees : short_trees_initial + short_trees_planted = 98 := by
  sorry

end total_short_trees_l769_769084


namespace projectile_height_reaches_35_l769_769414

theorem projectile_height_reaches_35 
  (t : ℝ)
  (h_eq : -4.9 * t^2 + 30 * t = 35) :
  t = 2 ∨ t = 50 / 7 ∧ t = min (2 : ℝ) (50 / 7) :=
by
  sorry

end projectile_height_reaches_35_l769_769414


namespace min_value_xy_expression_l769_769481

theorem min_value_xy_expression : ∃ x y : ℝ, (xy - 2)^2 + (x^2 + y^2) = 4 :=
by
  sorry

end min_value_xy_expression_l769_769481


namespace part_a_impossibility_l769_769658

-- Define the number of rows and columns
def num_rows : ℕ := 20
def num_columns : ℕ := 15

-- Define a function that checks if the sum of the counts in rows and columns match the conditions
def is_possible_configuration : Prop :=
  (num_rows % 2 = 0) ∧ (num_columns % 2 = 1)

theorem part_a_impossibility : ¬ is_possible_configuration :=
by
  -- The proof for the contradiction will go here
  sorry

end part_a_impossibility_l769_769658


namespace regular_permutation_l769_769647

def is_legal_transportation (a : List ℕ) (i j : ℕ) : Prop :=
  a.get? i = some 0 ∧ i > 0 ∧ a.get? (i-1) = some (a.get! j - 1)

def is_regular (a : List ℕ) (n : ℕ) : Prop :=
  ∃ a' : List ℕ, (∀ i j, is_legal_transportation a i j → is_legal_transportation a' i j) ∧ a' = List.range (n+1) ++ [0]

theorem regular_permutation (n : ℕ) :
  is_regular ([1, n] ++ List.range (n-1).reverse ++ [0]) n ↔ n = 2 ∨ ∃ j : ℕ, n = 2^j - 1 :=
by
  sorry

end regular_permutation_l769_769647


namespace real_x_values_satisfy_equation_l769_769199

theorem real_x_values_satisfy_equation (x : ℝ) (h : x > 0) :
  (x ^ log10 x = x^5 / 10000) ↔ (x = 10 ∨ x = 10000) :=
by
  sorry

end real_x_values_satisfy_equation_l769_769199


namespace count_positive_factors_of_180_multiple_of_15_l769_769723

statement: ℕ
statement := sorry

theorem count_positive_factors_of_180_multiple_of_15 :
  let factors_of_180 := {d ∈ (finset.range 181).filter (λ x, 180 % x = 0)}
  let multiples_of_15 := (finset.range 181).filter (λ x, x % 15 = 0)
  let desired_factors := factors_of_180 ∩ multiples_of_15
  desired_factors.card = 6 :=
by
  sorry

end count_positive_factors_of_180_multiple_of_15_l769_769723


namespace not_possible_20_odd_rows_15_odd_columns_l769_769655

def odd (n : ℕ) : Prop :=
  n % 2 = 1

def even (n : ℕ) : Prop :=
  n % 2 = 0

theorem not_possible_20_odd_rows_15_odd_columns
  (table : ℕ → ℕ → Prop) -- table representing the presence of crosses
  (n : ℕ) -- number of rows and columns in the square table
  (h_square_table: ∀ i j, table i j → i < n ∧ j < n)
  (odd_rows : ℕ)
  (odd_columns : ℕ)
  (h_odd_rows : odd_rows = 20)
  (h_odd_columns : odd_columns = 15)
  (h_def_odd_row: ∀ r, (∃ m, m < n ∧ odd (finset.card {c | c < n ∧ table r c})) ↔ r < odd_rows)
  (h_def_odd_column: ∀ c, (∃ m, m < n ∧ odd (finset.card {r | r < n ∧ table r c})) ↔ c < odd_columns)
  : false :=
by
  sorry

end not_possible_20_odd_rows_15_odd_columns_l769_769655


namespace num_factors_of_180_multiple_of_15_l769_769740

def is_factor (n m : Nat) : Prop := m % n = 0
def is_multiple_of_15 (n : Nat) : Prop := is_factor 15 n

theorem num_factors_of_180_multiple_of_15 : 
  (Nat.filter (fun n => is_factor n 180 ∧ is_multiple_of_15 n) (Nat.range (180 + 1))).length = 6 :=
sorry

end num_factors_of_180_multiple_of_15_l769_769740


namespace square_area_from_diagonal_l769_769537

theorem square_area_from_diagonal (d : ℝ) (h : d = 8 * Real.sqrt 2) : 
  ∃ (A : ℝ), A = 64 :=
begin
  use 64,
  sorry
end

end square_area_from_diagonal_l769_769537


namespace factors_of_180_multiple_of_15_l769_769748

/-- A factor of a number n is a positive integer which divides n without leaving a remainder. -/
def is_factor (n k : ℕ) : Prop :=
k > 0 ∧ n % k = 0

/-- A multiple of a number m is an integer which is divisible by m without leaving a remainder. -/
def is_multiple (k m : ℕ) : Prop :=
k % m = 0

/-- There are 6 factors of 180 that are also multiples of 15. -/
theorem factors_of_180_multiple_of_15 : 
  (Finset.filter (λ k, is_multiple k 15) { x | is_factor 180 x }.to_finset).card = 6 :=
by 
  sorry

end factors_of_180_multiple_of_15_l769_769748


namespace range_f_symmetric_f_zeros_g_l769_769272

noncomputable def f (x : ℝ) : ℝ := 4 / (Real.exp x + 1)
noncomputable def g (x : ℝ) : ℝ := f x - Real.abs x

-- Theorem statements based on the correct conclusions

theorem range_f (x : ℝ) : 0 < f x ∧ f x < 4 := by
  sorry

theorem symmetric_f : ∀ x : ℝ, f x + f (-x) = 4 := by
  sorry

theorem zeros_g : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ g x1 = 0 ∧ g x2 = 0 := by
  sorry

end range_f_symmetric_f_zeros_g_l769_769272


namespace fraction_of_top_10_lists_l769_769527

theorem fraction_of_top_10_lists (total_members : ℝ) (min_top_10_lists : ℝ) (fraction : ℝ) 
  (h1 : total_members = 765) (h2 : min_top_10_lists = 191.25) : 
    min_top_10_lists / total_members = fraction := by
  have h3 : fraction = 0.25 := by sorry
  rw [h1, h2, h3]
  sorry

end fraction_of_top_10_lists_l769_769527


namespace positive_factors_of_180_multiple_of_15_count_l769_769761

theorem positive_factors_of_180_multiple_of_15_count : 
  let factorization_180 := 2^2 * 3^2 * 5;
  let is_multiple_of_15 (n : ℕ) := ∃ k1 k2 : ℕ, k1 * 3 = n ∧ k2 * 5 = n;
  let factors_180 := { n : ℕ | n ∣ factorization_180 };
  { n : ℕ | n ∈ factors_180 ∧ is_multiple_of_15 n }.card = 6 :=
by
  sorry

end positive_factors_of_180_multiple_of_15_count_l769_769761


namespace percentage_change_area_l769_769422

theorem percentage_change_area (L B : ℝ) :
  let A_original := L * B
  let A_new := (L / 2) * (3 * B)
  (A_new - A_original) / A_original * 100 = 50 := by
  sorry

end percentage_change_area_l769_769422


namespace part1_part2_l769_769266

-- Part 1: Prove that if a^2 + b^2 - 2a + 6b + 10 = 0, then a + b = -2
theorem part1 (a b : ℝ) (h : a^2 + b^2 - 2a + 6b + 10 = 0) : a + b = -2 :=
sorry

-- Part 2: Prove that the minimum value of a^2 + 5b^2 + c^2 - 2ab - 4b + 6c + 15 is 5 for correct values.
theorem part2 (a b c : ℝ) (h₀ : a = 1/2) (h₁ : b = 1/2) (h₂ : c = -3) : 
  a^2 + 5*b^2 + c^2 - 2*a*b - 4*b + 6*c + 15 = 5 :=
sorry

end part1_part2_l769_769266


namespace alice_should_give_rattle_to_first_brother_l769_769567

def is_sunday (d : Day) : Prop := sorry
def tells_truth (brother : Brother) : Day → Prop := sorry
def lies (brother : Brother) : Day → Prop := sorry

-- The brothers type with instances Tweedledee and Tweedledum
inductive Brother 
| Tweedledum 
| Tweedledee

def first_brother := Brother.Tweedledum
def second_brother := Brother.Tweedledee

-- Ownership of rattle assertion by the first brother
def rattle_belongs_to (brother : Brother) : Prop :=
  brother = Brother.Tweedledee

-- Identity assertion by the second brother
def second_brother_identity (brother : Brother) : Prop :=
  brother = Brother.Tweedledum

-- Final condition stated in the problem
def not_sunday (today : Day) : Prop := ¬ is_sunday today
def one_tells_truth_one_lies (b1 b2 : Brother) : Day → Prop := 
  ∀ d, not_sunday d → ((tells_truth b1 d ∧ lies b2 d) ∨ (tells_truth b2 d ∧ lies b1 d))

-- The statement we wish to prove
theorem alice_should_give_rattle_to_first_brother (today : Day) : 
  not_sunday today → one_tells_truth_one_lies first_brother second_brother today → 
  (if rattle_belongs_to first_brother then rattle_belongs_to first_brother else rattle_belongs_to second_brother) →
  first_brother = Brother.Tweedledum :=
sorry

end alice_should_give_rattle_to_first_brother_l769_769567


namespace product_of_odd_and_even_is_odd_l769_769848

-- Definitions of odd and even functions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_even_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

-- The final Lean statement
theorem product_of_odd_and_even_is_odd (f g : ℝ → ℝ)
  (hf : is_odd_function f) (hg : is_even_function g) :
  is_odd_function (λ x, f x * g x) :=
by
  -- proof will go here
  sorry

end product_of_odd_and_even_is_odd_l769_769848


namespace alissa_earrings_count_l769_769189

theorem alissa_earrings_count:
  let total_pairs := 12 in
  let individual_earrings := total_pairs * 2 in
  let given_earrings := individual_earrings / 2 in
  3 * given_earrings = 36 :=
by
  sorry

end alissa_earrings_count_l769_769189


namespace how_many_kids_joined_l769_769092

theorem how_many_kids_joined (original_kids : ℕ) (new_kids : ℕ) (h : original_kids = 14) (h1 : new_kids = 36) :
  new_kids - original_kids = 22 :=
by
  sorry

end how_many_kids_joined_l769_769092


namespace incorrect_rational_number_statement_l769_769505

theorem incorrect_rational_number_statement :
  ¬ (∀ x : ℚ, x > 0 ∨ x < 0) := by
sorry

end incorrect_rational_number_statement_l769_769505


namespace expected_variance_X_expected_200_variance_360_l769_769624

noncomputable def problem_conditions (n : ℕ) (p : ℝ) : Prop :=
  n = 1000 ∧ p = 0.1

noncomputable def define_X (ξ : ℕ → ℝ) : (ℕ → ℝ) :=
  λ n, 2 * ξ n

noncomputable def expected_value_of_X (Eξ : ℕ → ℝ) : ℝ :=
  2 * Eξ 1000

noncomputable def variance_of_X (n : ℕ) (p : ℝ) (Dξ : ℕ → ℝ) : ℝ :=
  4 * n * p * (1 - p)

theorem expected_variance_X_expected_200_variance_360 {n : ℕ} {p : ℝ} (h : problem_conditions n p) :
  let ξ : ℕ → ℝ := fun n => n.to_real * p,
      Eξ : ℕ → ℝ := λ n, n.to_real * p,
      Dξ : ℕ → ℝ := λ n, n.to_real * p * (1 - p),
      X : (ℕ → ℝ) := define_X ξ
  in expected_value_of_X Eξ = 200 ∧ variance_of_X n p Dξ = 360 := by
  obtain ⟨h₁, h₂⟩ := h
  unfold ξ Eξ Dξ at *
  rw [h₁, h₂]
  have hE : expected_value_of_X (λ n, n.to_real * 0.1) = 200, by
    simp [expected_value_of_X, mul_assoc, eq_comm]
  have hV : variance_of_X 1000 0.1 (λ n, n.to_real * 0.1 * 0.9) = 360, by
    simp [variance_of_X, mul_assoc, mul_comm, eq_comm, sub_self]
  exact ⟨hE, hV⟩

end expected_variance_X_expected_200_variance_360_l769_769624


namespace no_nat_fun_satisfying_property_l769_769591

theorem no_nat_fun_satisfying_property :
  ¬ ∃ f : ℕ → ℕ, ∀ x : ℕ, f (f x) = x + 1 :=
by
  sorry

end no_nat_fun_satisfying_property_l769_769591


namespace max_lambda_l769_769262

noncomputable def a (n : ℕ) : ℝ := 2 * n - 1
def S (n : ℕ) : ℝ := finset.sum (finset.range n) (λ k, a (k + 1))

theorem max_lambda (λ : ℝ) :
  (∀ n : ℕ, 0 < n →
    (a (n+1) ≠ 0 ∧ a n ^ 2 = S (2*n-1)) ∧
    (λ / a (n+1) ≤ (n + 8 * (-1)^n) / 2*n)) →
  λ ≤ -21 / 2 :=
by
  sorry

end max_lambda_l769_769262


namespace count_permutations_product_l769_769890

noncomputable def c (n m : ℕ) : ℕ := sorry

theorem count_permutations_product (t : ℝ) :
  ∀ (n : ℕ), (∑ m in (Finset.range (n + 1)), (c n m) * t ^ m) =
  ∏ i in (Finset.range (n + 1)), (∑ j in (Finset.range i), t ^ j) :=
begin
  sorry
end

end count_permutations_product_l769_769890


namespace find_y_when_x_eq_4_l769_769967

theorem find_y_when_x_eq_4 (x y : ℝ) (k : ℝ) :
  (8 * y = k / x^3) →
  (y = 25) →
  (x = 2) →
  (exists y', x = 4 → y' = 25/8) :=
by
  sorry

end find_y_when_x_eq_4_l769_769967


namespace range_of_a_l769_769269

theorem range_of_a (a : ℝ) :
    (∀ x : ℝ, -3 * x^2 + a * x - 1 ≤ 0) ↔ (-real.sqrt 3 ≤ a ∧ a ≤ real.sqrt 3) :=
by
  sorry

end range_of_a_l769_769269


namespace rotation_implies_equilateral_l769_769361

noncomputable def is_equilateral (A1 A2 A3 : Point) : Prop :=
  distance A1 A2 = distance A2 A3 ∧
  distance A2 A3 = distance A3 A1 ∧
  angle A1 A2 A3 = π / 3

theorem rotation_implies_equilateral
  (triangle : Triangle)
  (P0 : Point)
  (A : ℕ → Point)
  (P: ℕ → Point)
  (hA : ∀ s, s ≥ 4 → A s = A (s-3))
  (hP : ∀ k, P (k+1) = rotate_pt (A (k+1)) (2 * π / 3) (P k))
  (hP2007 : P 2007 = P0) :
  is_equilateral (A 1) (A 2) (A 3) :=
sorry

end rotation_implies_equilateral_l769_769361


namespace XF_mul_XG_eq_17_l769_769037

theorem XF_mul_XG_eq_17 (O : Type*) [field O]
  (A B C D X Y E F G : O) (h : ∀ {P Q R S : O}, P + Q = R + S) :
  intersect O A X Y F 
  → parallel O Y A D 
  → parallel O E C F 
  →  G ∈ circle_points O
  → line_contains O G C
  → side_length O A B 3
  → side_length O B C 2
  → side_length O C D 6
  → side_length O D A 8
  → ratio_eq O D X B D (1/4)
  → ratio_eq O B Y B D (11/36)
  → calculate_XF_XG O X F G 17 :=
sorry

end XF_mul_XG_eq_17_l769_769037


namespace day_of_week_365th_day_l769_769836

theorem day_of_week_365th_day (day15_is_tuesday : 15 % 7 = 2) :
    365 % 7 = 2 :=
by
  -- Given that the 15th day falls on a Tuesday, we have 15 ≡ 2 (mod 7)
  have h1 : 15 % 7 = 2 := day15_is_tuesday
  -- Calculating the day of the week for the 365th day
  have h2 : 365 % 7 = (15 + 350) % 7 := by norm_num
  rw [h1, ← nat.add_mod],
  -- Since 350 % 7 = 0, we get the same day as the 15th day
  norm_num

end day_of_week_365th_day_l769_769836


namespace max_tiles_l769_769509

open Nat

theorem max_tiles (tile_width tile_height floor_width floor_height : ℕ) 
  (h_tile_dims : tile_width = 45 ∧ tile_height = 50) 
  (h_floor_dims : floor_width = 250 ∧ floor_height = 180) : 
  (max 
      ((floor (floor_width / tile_width.toRat)).toNat * (floor (floor_height / tile_height.toRat)).toNat)
      ((floor (floor_width / tile_height.toRat)).toNat * (floor (floor_height / tile_width.toRat)).toNat)) = 20 := 
by
  sorry

end max_tiles_l769_769509


namespace find_alpha_polar_eqn_of_line_l769_769634

open Real

noncomputable def P : Point := ⟨2, 1⟩

def line_eqs (alpha : ℝ) (t : ℝ) : Point :=
  ⟨2 + t * cos(alpha), 1 + t * sin(alpha)⟩

def PA (alpha : ℝ) : ℝ :=
  dist P ⟨2 + (-1 / sin(alpha)) * cos(alpha), 0⟩

def PB (alpha : ℝ) : ℝ :=
  dist P ⟨0, 1 + (-2 / cos(alpha)) * sin(alpha)⟩

theorem find_alpha (alpha : ℝ) (h : PA(alpha) * PB(alpha) = 4) : alpha = 3 * π / 4 :=
sorry

theorem polar_eqn_of_line (alpha : ℝ) (h : PA(alpha) * PB(alpha) = 4) : 
  (∃ ρ θ, ρ * (cos(θ) + sin(θ)) = 3) :=
sorry

end find_alpha_polar_eqn_of_line_l769_769634


namespace perpendicular_planes_lines_l769_769702

/-- Given two mutually perpendicular planes α and β that intersect at line l, 
    and two lines m and n, with m being parallel to α and n being perpendicular to β,
    we want to prove that n is perpendicular to l. -/
theorem perpendicular_planes_lines
  (α β : Plane) (l m n : Line)
  (h1 : α ⊥ β)
  (h2 : l ∈ α ∩ β)
  (h3 : m ∥ α)
  (h4 : n ⊥ β) :
  n ⊥ l :=
sorry

end perpendicular_planes_lines_l769_769702


namespace compare_magnitudes_l769_769639

noncomputable def a : ℝ := Real.sin (46 * Real.pi / 180)
noncomputable def b : ℝ := Real.cos (46 * Real.pi / 180)
noncomputable def c : ℝ := Real.cos (36 * Real.pi / 180)

theorem compare_magnitudes : c > a ∧ a > b :=
by
  have ha : a = Real.cos (44 * Real.pi / 180), from sorry,
  exact And.intro sorry sorry

end compare_magnitudes_l769_769639


namespace collinear_reflected_lines_l769_769356

-- Given definitions
variables {A B C P : Type*} [plane_geometry A B C P]
variables {γ : line} (PA PB PC : line)

-- Reflected points
def A' : point := reflection P γ PA
def B' : point := reflection P γ PB
def C' : point := reflection P γ PC

-- Lean statement for the collinearity of points A', B', and C'
theorem collinear_reflected_lines :
  collinear [A', B', C'] :=
sorry

end collinear_reflected_lines_l769_769356


namespace largest_divisor_of_a5_minus_a_l769_769618

theorem largest_divisor_of_a5_minus_a :
  ∃ n, (∀ a : ℤ, n ∣ a^5 - a) ∧ (∀ m > n, ∃ a : ℤ, ¬ m ∣ a^5 - a) :=
begin
  use 30,
  split,
  {
    intro a,
    sorry,  -- This part would contain the proof that 30 divides a^5 - a for any integer a.
  },
  {
    intros m hm,
    sorry,  -- This part would contain the proof that no number greater than 30 can always divide a^5 - a for any integer a.
  }
end

end largest_divisor_of_a5_minus_a_l769_769618


namespace unshaded_area_correct_l769_769167

noncomputable def squareSide : ℝ := 12
noncomputable def radius : ℝ := squareSide / 4

-- Define the areas
noncomputable def areaSquare : ℝ := squareSide ^ 2
noncomputable def areaQuarterCircle : ℝ := (3 / 4) * (π * radius ^ 2)
noncomputable def totalQuarterCircleArea : ℝ := 4 * areaQuarterCircle
noncomputable def unshadedRegion : ℝ := areaSquare - totalQuarterCircleArea

theorem unshaded_area_correct : 
  unshadedRegion = 144 - 27 * π := by
  sorry

end unshaded_area_correct_l769_769167


namespace divide_cookie_into_equal_parts_l769_769594

-- Define the structure and properties of the cookie
structure Cookie :=
  (num_squares : ℕ)
  (num_semicircles : ℕ)
  (center : Prop)
  (axes_of_symmetry : ℕ)
  (equal_parts : ℕ)

-- Define the conditions
def cookie_conditions (c : Cookie) : Prop :=
  c.num_squares = 64 ∧
  c.num_semicircles = 16 ∧
  c.axes_of_symmetry = 2 ∧
  c.equal_parts = 16

-- Define the property that each part should ideally have one semicircle and four squares
def part_properties (c : Cookie) : Prop :=
  ∀ i : fin c.equal_parts,
  (1 = 1 ∧ 4 = 4) -- Placeholder for rigorous geometric condition of each part having exact one semicircle and four squares

-- Define the theorem to prove the possibility of dividing the cookie into equal parts
theorem divide_cookie_into_equal_parts :
  ∀ (c : Cookie), cookie_conditions c → part_properties c :=
by
  intros c hc
  sorry

end divide_cookie_into_equal_parts_l769_769594


namespace day_of_week_365th_day_l769_769837

theorem day_of_week_365th_day (day15_is_tuesday : 15 % 7 = 2) :
    365 % 7 = 2 :=
by
  -- Given that the 15th day falls on a Tuesday, we have 15 ≡ 2 (mod 7)
  have h1 : 15 % 7 = 2 := day15_is_tuesday
  -- Calculating the day of the week for the 365th day
  have h2 : 365 % 7 = (15 + 350) % 7 := by norm_num
  rw [h1, ← nat.add_mod],
  -- Since 350 % 7 = 0, we get the same day as the 15th day
  norm_num

end day_of_week_365th_day_l769_769837


namespace find_alpha_polar_eqn_of_line_l769_769635

open Real

noncomputable def P : Point := ⟨2, 1⟩

def line_eqs (alpha : ℝ) (t : ℝ) : Point :=
  ⟨2 + t * cos(alpha), 1 + t * sin(alpha)⟩

def PA (alpha : ℝ) : ℝ :=
  dist P ⟨2 + (-1 / sin(alpha)) * cos(alpha), 0⟩

def PB (alpha : ℝ) : ℝ :=
  dist P ⟨0, 1 + (-2 / cos(alpha)) * sin(alpha)⟩

theorem find_alpha (alpha : ℝ) (h : PA(alpha) * PB(alpha) = 4) : alpha = 3 * π / 4 :=
sorry

theorem polar_eqn_of_line (alpha : ℝ) (h : PA(alpha) * PB(alpha) = 4) : 
  (∃ ρ θ, ρ * (cos(θ) + sin(θ)) = 3) :=
sorry

end find_alpha_polar_eqn_of_line_l769_769635


namespace longest_side_in_ratio_5_6_7_l769_769982

theorem longest_side_in_ratio_5_6_7 (x : ℕ) (h : 5 * x + 6 * x + 7 * x = 720) : 7 * x = 280 := 
by
  sorry

end longest_side_in_ratio_5_6_7_l769_769982


namespace least_vertices_l769_769052

variables (G : Type) [Graph G] (n : ℕ)

def degree (v : G) := Graph.degree G v
def connected : Prop := Graph.connected G
def hamiltonian_path_free (G : Type) [Graph G] : Prop := ¬∃ (l : List G), List.Nodup l ∧ Graph.walk_of_list G l

axiom HamiltonianPathFreeCondition
  (hG_conn : connected G) 
  (hG_deg : ∀ v : G, degree v ≥ n) 
  (hn_pos : n ≥ 3) :
  hamiltonian_path_free G

/-- The least possible number of vertices in a connected graph with minimum degree n and no Hamiltonian path is 2n + 2 -/
theorem least_vertices 
  (hG_conn : connected G) 
  (hG_deg : ∀ v : G, degree v ≥ n) 
  (hn_pos : n ≥ 3) :
  ∃ (m : ℕ), m = 2 * n + 2 ∧ hamiltonian_path_free G :=
sorry

end least_vertices_l769_769052


namespace xiaoming_age_l769_769462

theorem xiaoming_age
  (x x' : ℕ) 
  (h₁ : ∃ f : ℕ, f = 4 * x) 
  (h₂ : (x + 25) + (4 * x + 25) = 100) : 
  x = 10 :=
by
  obtain ⟨f, hf⟩ := h₁
  sorry

end xiaoming_age_l769_769462


namespace num_diagonals_tetragon_l769_769706

def num_diagonals_in_polygon (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem num_diagonals_tetragon : num_diagonals_in_polygon 4 = 2 := by
  sorry

end num_diagonals_tetragon_l769_769706


namespace exists_multiple_with_sum_of_digits_eq_l769_769395

theorem exists_multiple_with_sum_of_digits_eq (n : ℕ) (hn : 0 < n) :
  ∃ m : ℕ, (m % n = 0) ∧ (sum_of_digits m = n) :=
sorry

end exists_multiple_with_sum_of_digits_eq_l769_769395


namespace at_least_one_not_greater_one_third_sqrt_inequality_l769_769513

-- Problem 1: Prove at least one of a, b, c is not greater than 1/3 given a + b + c = 1
theorem at_least_one_not_greater_one_third (a b c : ℝ) (h : a + b + c = 1) : a ≤ 1/3 ∨ b ≤ 1/3 ∨ c ≤ 1/3 :=
sorry

-- Problem 2: Prove sqrt(6) + sqrt(7) > 2sqrt(2) + sqrt(5)
theorem sqrt_inequality : √6 + √7 > 2 * √2 + √5 :=
sorry

end at_least_one_not_greater_one_third_sqrt_inequality_l769_769513


namespace quadratic_equation_unique_solution_l769_769055

theorem quadratic_equation_unique_solution (a b x k : ℝ) (h : a = 8) (h₁ : b = 36) (h₂ : k = 40.5) : 
  (8*x^2 + 36*x + 40.5 = 0) ∧ x = -2.25 :=
by {
  sorry
}

end quadratic_equation_unique_solution_l769_769055


namespace option_a_option_b_l769_769545

noncomputable def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def height (V g : ℝ) (t : ℝ) : ℝ := V * t - (g * t^2) / 2

theorem option_a (V : ℝ) (hV : 10 * Real.sqrt 2 ≤ V ∧ V < 15) :
  ∃ t1 t2 : ℝ, 1 < t1 ∧ t1 < t2 ∧ t2 < 2 ∧ height V 10 t1 = 10 ∧ height V 10 t2 = 10 :=
sorry

theorem option_b (V : ℝ) : ¬(∃ t1 t2 : ℝ, 2 < t1 ∧ t1 < t2 ∧ t2 < 4 ∧ height V 10 t1 = 10 ∧ height V 10 t2 = 10) :=
sorry

end option_a_option_b_l769_769545


namespace unique_solution_m_l769_769698

theorem unique_solution_m (m : ℝ) :
  (∃! x : ℝ, m * x^2 + 2 * x - 1 = 0) ↔ (m = 0 ∨ m = -1) :=
by
  sorry

end unique_solution_m_l769_769698


namespace factors_of_180_l769_769771

-- Define a predicate to check if a number is a factor of another number
def is_factor (m n : ℕ) : Prop := n % m = 0

-- Define a predicate to check if a number is a multiple of another number
def is_multiple (m n : ℕ) : Prop := m % n = 0

-- Define the problem
theorem factors_of_180 : {n : ℕ // is_factor n 180 ∧ is_multiple n 15}.card = 6 :=
by sorry

end factors_of_180_l769_769771


namespace sum_of_x_values_eq_five_l769_769484

theorem sum_of_x_values_eq_five : 
  (∑ x in {x : ℝ | 10 = (x*x*x - 5*x*x - 10*x) / (x + 2) ∧ x ≠ -2}, x) = 5 :=
by
  sorry

end sum_of_x_values_eq_five_l769_769484


namespace number_of_true_propositions_l769_769358

-- Definitions for distinct planes and lines
variables {α β γ : Type} -- Representing planes
variables {l m n : Type} -- Representing lines

-- Propositions as functions
def prop_1 (α β γ : Type) : Prop := (α ≠ γ) ∧ (β ≠ γ) → (α ≠ β)
def prop_2 (α β : Type) (m n : Type) : Prop := (m ⊆ α) ∧ (n ⊆ α) ∧ (m ∥ β) ∧ (n ∥ β) → (α ≠ β)
def prop_3 (α β : Type) (l : Type) : Prop := (α ∥ β) ∧ (l ⊆ α) → (l ∥ β)
def prop_4 (α β γ : Type) (l m n : Type) : Prop := (α ∩ β = l) ∧ (β ∩ γ = m) ∧ (γ ∩ α = n) ∧ (l ∥ γ) → (m ∥ n)

-- Correct answer (number of true propositions)
def correct_answer : Nat := 2

-- Hypothesis for distinctness of planes and lines
variables (h1 : α ≠ β ∧ β ≠ γ ∧ γ ≠ α)
variables (h2 : l ≠ m ∧ m ≠ n ∧ n ≠ l)

theorem number_of_true_propositions (α β γ : Type) (l m n : Type)
    (h1 : α ≠ β ∧ β ≠ γ ∧ γ ≠ α)
    (h2 : l ≠ m ∧ m ≠ n ∧ n ≠ l)
    : ((prop_1 α β γ ∨ ¬ prop_1 α β γ) + 
       (prop_2 α β m n ∨ ¬ prop_2 α β m n) + 
       (prop_3 α β l ∨ ¬ prop_3 α β l) + 
       (prop_4 α β γ l m n ∨ ¬ prop_4 α β γ l m n)) = correct_answer :=
sorry

end number_of_true_propositions_l769_769358


namespace quadratic_equation_new_roots_l769_769671

variables {α : Type*} [CommRing α] 

theorem quadratic_equation_new_roots (a b r1 r2 : α) 
(h1 : r1 + r2 = a) (h2 : r1 * r2 = b) :
  (Polynomial.X ^ 2 - (Polynomial.C (a^2 + a - 2*b)) * Polynomial.X + Polynomial.C (a^3 - a * b)).roots =
    (r1^2 + r2, r1 + r2^2) :=
sorry

end quadratic_equation_new_roots_l769_769671


namespace solve_system_solve_system_neg_l769_769051

/-- Problem: Solve the system of equations
  ∀ i ∈ {1, 2, ..., n},
  ∏ j from {1, 2, ..., n} \ i / x_i = a_i,
Prove the solution for x_i in the cases a_i > 0 and a_i < 0:
-/

theorem solve_system (n : ℕ) (a : Fin n → ℝ) (x : Fin n → ℝ)
  (h : ∀ i : Fin n, (∏ j in Finset.univ \ {i}, x j) / x i = a i)
  (pos_cond : ∀ i, 0 < a i) :
  ∀ i, x i = sqrt ((∏ i, a i)^(1 / (n - 2)) / a i) ∨ x i = -sqrt ((∏ i, a i)^(1 / (n - 2)) / a i) :=
sorry

theorem solve_system_neg (n : ℕ) (a : Fin n → ℝ) (x : Fin n → ℝ)
  (h : ∀ i : Fin n, (∏ j in Finset.univ \ {i}, x j) / x i = a i)
  (neg_cond : ∀ i, a i < 0) :
  ∀ i, i = ⟨0, (by simp [lt_of_lt_of_le (zero_lt_one) (nat.succ_le_of_lt (show 1 < n, by sorry))])⟩ → x i = -sqrt ((∏ i, -a i)^(1 / (n - 2)) / -a i) ∨
          ∀ i, x i = sqrt ((∏ i, -a i)^(1 / (n - 2)) / -a i) :=
sorry

end solve_system_solve_system_neg_l769_769051


namespace tangent_perpendicular_point_l769_769316

theorem tangent_perpendicular_point :
  ∃ P : ℝ × ℝ, P = (1, 0) ∧ 
    (∃ (f : ℝ → ℝ) (f' : ℝ → ℝ), f x = x^4 - x ∧ (∀ x, f' x = 4*x^3 - 1) ∧ (∃ p, p ∈ P ∧ f' p = 3)) :=
by
  sorry

end tangent_perpendicular_point_l769_769316


namespace tangent_line_min_slope_l769_769275

noncomputable def tangent_line_equation : Prop :=
  ∃ k b, (∀ x y, y = x^3 + 3 * x - 1 → y = k * x + b) ∧
         k = 3 * x^2 + 3 ∧ 
         (∀ x, k ≥ 3) ∧ 
         k = 3 ∧ b = -1 ∧ 
         3 * x - y = 1

theorem tangent_line_min_slope : tangent_line_equation := sorry

end tangent_line_min_slope_l769_769275


namespace factors_of_180_multiples_of_15_l769_769796

theorem factors_of_180_multiples_of_15 :
  (finset.filter (λ x : ℕ, x % 15 = 0) (finset.filter (λ x : ℕ, 180 % x = 0) (finset.range (180 + 1)))).card = 6 :=
sorry

end factors_of_180_multiples_of_15_l769_769796


namespace increase_in_average_weight_l769_769411

theorem increase_in_average_weight 
  (initial_weight new_weight : ℕ)
  (number_of_oarsmen : ℕ)
  (initial_weight = 53) 
  (new_weight = 71) 
  (number_of_oarsmen = 10) : 
  (new_weight - initial_weight) / number_of_oarsmen = 1.8 := 
by
  sorry

end increase_in_average_weight_l769_769411


namespace num_factors_of_180_multiple_of_15_l769_769741

def is_factor (n m : Nat) : Prop := m % n = 0
def is_multiple_of_15 (n : Nat) : Prop := is_factor 15 n

theorem num_factors_of_180_multiple_of_15 : 
  (Nat.filter (fun n => is_factor n 180 ∧ is_multiple_of_15 n) (Nat.range (180 + 1))).length = 6 :=
sorry

end num_factors_of_180_multiple_of_15_l769_769741


namespace abs_sum_plus_two_eq_sum_abs_l769_769369

theorem abs_sum_plus_two_eq_sum_abs {a b c : ℤ} (h : |a + b + c| + 2 = |a| + |b| + |c|) :
  a^2 = 1 ∨ b^2 = 1 ∨ c^2 = 1 :=
sorry

end abs_sum_plus_two_eq_sum_abs_l769_769369


namespace square_area_combined_circle_area_l769_769628

-- Define the problem conditions
def radius : ℝ := 6
def diameter : ℝ := 2 * radius
def side_length_square : ℝ := 2 * diameter

-- Statement for area of the square
theorem square_area : side_length_square ^ 2 = 576 := by
  sorry

-- Statement for the combined area of the circles
theorem combined_circle_area : 4 * (Real.pi * radius ^ 2) ≈ 452.388 := by
  sorry

end square_area_combined_circle_area_l769_769628


namespace triangle_sum_of_sides_l769_769868

/-- Given a triangle with angles 45 degrees and 60 degrees, and the side opposite the 45-degree angle is 8 units, 
prove the sum of the lengths of the other two sides is approximately 19.3 units. -/
theorem triangle_sum_of_sides (A B C : Type) [EuclideanGeometry A] [Triangle A B C] 
  (angle_A : ∠ A = 45) (angle_B : ∠ B = 60) (side_opposite45 : dist B C = 8) : 
  dist A B + dist A C ≈ 19.3 :=
sorry

end triangle_sum_of_sides_l769_769868


namespace sum_excluding_multiples_l769_769344

theorem sum_excluding_multiples (S_total S_2 S_3 S_6 : ℕ) 
  (hS_total : S_total = (100 * (1 + 100)) / 2) 
  (hS_2 : S_2 = (50 * (2 + 100)) / 2) 
  (hS_3 : S_3 = (33 * (3 + 99)) / 2) 
  (hS_6 : S_6 = (16 * (6 + 96)) / 2) :
  S_total - S_2 - S_3 + S_6 = 1633 :=
by
  sorry

end sum_excluding_multiples_l769_769344


namespace family_size_is_four_l769_769088

-- Let us declare the given conditions as variables and constants.
variable (n : ℕ)
variable (current_average_age : ℝ := 20)
variable (youngest_age : ℝ := 10)
variable (birth_average_age : ℝ := 12.5)

-- The theorem to prove.
theorem family_size_is_four :
  let total_age_at_birth := birth_average_age * n
      total_age_now := total_age_at_birth + (youngest_age * (n - 1))
  in current_average_age * n = total_age_now → n = 4 := 
by
  intros h
  sorry

end family_size_is_four_l769_769088


namespace youngest_child_age_l769_769460

theorem youngest_child_age (x y z : ℕ) 
  (h1 : 3 * x + 6 = 48) 
  (h2 : 3 * y + 9 = 60) 
  (h3 : 2 * z + 4 = 30) : 
  z = 13 := 
sorry

end youngest_child_age_l769_769460


namespace square_area_from_diagonal_l769_769540

theorem square_area_from_diagonal (d : ℝ) (h : d = 8 * Real.sqrt 2) : (d / Real.sqrt 2) ^ 2 = 64 :=
by
  sorry

end square_area_from_diagonal_l769_769540


namespace sum_of_first_nine_is_153_point_on_line_condition_general_term_is_correct_sum_of_new_sequence_is_correct_l769_769677

def sequence_a (n : ℕ) : ℕ := 3 * n + 2

def sum_first_nine_terms (sum9 : ℕ) : Prop :=
  sum9 = (9 * (sequence_a 1 + sequence_a 9)) / 2

def point_condition (n : ℕ) : Prop :=
  sequence_a (n + 1) = sequence_a n + 3

def general_term_proof : Prop :=
  ∀ n : ℕ, sequence_a n = 3 * n + 2

def sequence_b (n : ℕ) : ℕ :=
  sequence_a (n * 2^n)

def sum_first_n_terms_b (S_n : ℕ) (n : ℕ) : Prop :=
  S_n = ∑ i in range n, sequence_b i -- Sum of the first 'n' terms of sequence b

def sum_formula (S_n : ℕ) (n : ℕ) : Prop :=
  S_n = 3 * (n - 1) * 2^(n + 1) + 2 * n + 6

theorem sum_of_first_nine_is_153 : sum_first_nine_terms 153 :=
sorry

theorem point_on_line_condition : ∀ n, point_condition n :=
sorry

theorem general_term_is_correct : general_term_proof :=
sorry

theorem sum_of_new_sequence_is_correct (n : ℕ) : sum_first_n_terms_b (3 * (n - 1) * 2^(n + 1) + 2 * n + 6) n ↔ sum_formula (3 * (n - 1) * 2^(n + 1) + 2 * n + 6) n :=
sorry

end sum_of_first_nine_is_153_point_on_line_condition_general_term_is_correct_sum_of_new_sequence_is_correct_l769_769677


namespace sum_of_reciprocals_l769_769457

theorem sum_of_reciprocals (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x + y = 5 * x * y) : (1 / x) + (1 / y) = 5 :=
by
  sorry

end sum_of_reciprocals_l769_769457


namespace max_value_f_l769_769626

def f (x : ℝ) : ℝ := min (4 * x + 1) (min (x + 2) (-2 * x + 4))

theorem max_value_f : ∃ x : ℝ, f(x) = 8 / 3 := sorry

end max_value_f_l769_769626


namespace correct_operation_l769_769500

theorem correct_operation (a b : ℝ) :
  ¬ (a^2 + a^3 = a^6) ∧
  ¬ ((a*b)^2 = a*(b^2)) ∧
  ¬ ((a + b)^2 = a^2 + b^2) ∧
  ((a + b)*(a - b) = a^2 - b^2) := 
by
  sorry

end correct_operation_l769_769500


namespace chess_tournament_games_l769_769510

/--
There are 4 chess amateurs playing in a tournament. Each amateur plays against every other amateur exactly once.
Prove that the total number of unique chess games possible to be played in the tournament is 6.
-/
theorem chess_tournament_games 
  (num_players : ℕ)
  (h : num_players = 4)
  : nat.choose 4 2 = 6 := 
by sorry

end chess_tournament_games_l769_769510


namespace not_possible_20_odd_rows_15_odd_columns_l769_769657

def odd (n : ℕ) : Prop :=
  n % 2 = 1

def even (n : ℕ) : Prop :=
  n % 2 = 0

theorem not_possible_20_odd_rows_15_odd_columns
  (table : ℕ → ℕ → Prop) -- table representing the presence of crosses
  (n : ℕ) -- number of rows and columns in the square table
  (h_square_table: ∀ i j, table i j → i < n ∧ j < n)
  (odd_rows : ℕ)
  (odd_columns : ℕ)
  (h_odd_rows : odd_rows = 20)
  (h_odd_columns : odd_columns = 15)
  (h_def_odd_row: ∀ r, (∃ m, m < n ∧ odd (finset.card {c | c < n ∧ table r c})) ↔ r < odd_rows)
  (h_def_odd_column: ∀ c, (∃ m, m < n ∧ odd (finset.card {r | r < n ∧ table r c})) ↔ c < odd_columns)
  : false :=
by
  sorry

end not_possible_20_odd_rows_15_odd_columns_l769_769657


namespace percentage_change_in_area_of_rectangle_l769_769438

theorem percentage_change_in_area_of_rectangle
  (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  let A1 := L * B
  let A2 := (L / 2) * (3 * B)
  let percentage_change := ((A2 - A1) / A1) * 100
  percentage_change = 50 := by
  let A1 := L * B
  let A2 := (L / 2) * (3 * B)
  let percentage_change := ((A2 - A1) / A1) * 100
  calc percentage_change
      = ((A2 - A1) / A1) * 100 : rfl
  ... = ((((L / 2) * (3 * B)) - (L * B)) / (L * B)) * 100 : by rw [A1, A2]
  ... = ((3 / 2 * A1 - A1) / A1) * 100 : by rw [A2, ←mul_assoc, mul_div_cancel' _ (two_ne_zero : 2 ≠ 0)]
  ... = ((3 / 2 - 1) * A1 / A1) * 100 : by simp only [mul_sub, mul_div_cancel' _ (ne_of_gt hL)]
  ... = (1 / 2) * 100 : by rw [div_mul_cancel' (ne_of_gt (lt_of_lt_of_le zero_lt_one (le_of_lt ((mul_lt_iff_lt_one_left (lt_of_lt_of_le zero_lt_one hL)).mpr (half_pos (lt_of_le_of_ne (le_of_lt (lt_of_lt_of_le (lt_one_mul_self (lt_of_le_of_lt (zero_le_two) (lt_add_one 1))) (lt_of_lt_of_le (two_le_one) (lt_one 2))))))))) so))
  ... = 50 : rfl

end percentage_change_in_area_of_rectangle_l769_769438


namespace trapezoid_base_difference_is_10_l769_769963

noncomputable def trapezoid_base_difference (AD BC AB : ℝ) (angle_BAD angle_ADC : ℝ) : ℝ :=
if angle_BAD = 60 ∧ angle_ADC = 30 ∧ AB = 5 then AD - BC else 0

theorem trapezoid_base_difference_is_10 (AD BC : ℝ) (angle_BAD angle_ADC : ℝ) (h_BAD : angle_BAD = 60)
(h_ADC : angle_ADC = 30) (h_AB : AB = 5) : trapezoid_base_difference AD BC AB angle_BAD angle_ADC = 10 :=
sorry

end trapezoid_base_difference_is_10_l769_769963


namespace problem_statement_l769_769305

noncomputable def countNs : Nat :=
  let N_values := {N : ℕ | ∃ x : ℝ, 0 < N ∧ N < 500 ∧ (x ^ Nat.floor x = N)}
  N_values.toFinset.card

theorem problem_statement :
  countNs = 287 := by
  sorry

end problem_statement_l769_769305


namespace complex_number_solution_l769_769678

theorem complex_number_solution (z : ℂ) (i : ℂ) (h : i = complex.I) (h_condition : z * (1 - i) = 2) : z = 1 + i := 
by 
  sorry

end complex_number_solution_l769_769678


namespace total_cars_made_in_two_days_l769_769554

-- Define the number of cars made yesterday.
def cars_yesterday : ℕ := 60

-- Define the number of cars made today, which is twice the number of cars made yesterday.
def cars_today : ℕ := 2 * cars_yesterday

-- Define the total number of cars made over the two days.
def total_cars : ℕ := cars_yesterday + cars_today

-- Proof statement: Prove that the total number of cars made in these two days is 180.
theorem total_cars_made_in_two_days : total_cars = 180 := by
  -- Here, we would provide the proof, but we'll use sorry to skip it.
  sorry

end total_cars_made_in_two_days_l769_769554


namespace P_neither_sufficient_nor_necessary_Q_l769_769276

-- Definitions
def P (pyramid : Type) : Prop := 
  ∀ (faces : Set (Set pyramid)), (∀ (f1 f2 : Set pyramid), f1 ≠ f2 → (is_dihedral_angle_equal f1 f2))

def Q (pyramid : Type) : Prop := 
  ∀ (edges : Set (Set pyramid)), (∀ (e1 e2 : Set pyramid), e1 ≠ e2 → (is_spatial_angle_equal e1 e2))

-- Theorem
theorem P_neither_sufficient_nor_necessary_Q (pyramid : Type) : ¬ (P pyramid → Q pyramid) ∧ ¬ (Q pyramid → P pyramid) :=
by
  -- We would need to show instances or counterexamples to prove this. In this statement, we're asserting the theorem.
  sorry

end P_neither_sufficient_nor_necessary_Q_l769_769276


namespace count_factors_of_180_multiple_of_15_l769_769714

def is_factor (x n : ℕ) := n % x = 0 -- Definition of a factor
def is_multiple (x k : ℕ) := x % k = 0 -- Definition of a multiple

theorem count_factors_of_180_multiple_of_15 :
  (finset.filter (λ x, is_multiple x 15) (finset.filter (λ x, is_factor x 180) (finset.range (180 + 1)))).card = 6 := by
  sorry

end count_factors_of_180_multiple_of_15_l769_769714


namespace tan_C_value_b_value_l769_769346

-- Define variables and conditions
variable (A B C a b c : ℝ)
variable (A_eq : A = Real.pi / 4)
variable (cond : b^2 - a^2 = 1 / 4 * c^2)
variable (area_eq : 1 / 2 * b * c * Real.sin A = 5 / 2)

-- First part: Prove tan(C) = 4 given the conditions
theorem tan_C_value : A = Real.pi / 4 ∧ b^2 - a^2 = 1 / 4 * c^2 → Real.tan C = 4 := by
  intro h
  sorry

-- Second part: Prove b = 5 / 2 given the area condition
theorem b_value : (1 / 2 * b * c * Real.sin (Real.pi / 4) = 5 / 2) → b = 5 / 2 := by
  intro h
  sorry

end tan_C_value_b_value_l769_769346


namespace sides_of_all_squares_given_conditions_l769_769163

theorem sides_of_all_squares_given_conditions : 
  ∀ (s1 s2 s3 s4 s5 s6 : ℕ), 
    (s1 = 18) → 
    (s2 = 3) → 
    (s3 = s1 - s2) → 
    (s4 = s3 - s2) → 
    (s5 = s4) → 
    (s6 = s1 + s2) → 
    (s3 = 15 ∧ s4 = 12 ∧ s5 = 12 ∧ s6 = 21) :=
begin
  intros s1 s2 s3 s4 s5 s6 h1 h2 h3 h4 h5 h6,
  rw [h1, h2] at *,
  simp at *,
  sorry

end sides_of_all_squares_given_conditions_l769_769163


namespace factors_of_180_multiples_of_15_l769_769794

theorem factors_of_180_multiples_of_15 :
  (finset.filter (λ x : ℕ, x % 15 = 0) (finset.filter (λ x : ℕ, 180 % x = 0) (finset.range (180 + 1)))).card = 6 :=
sorry

end factors_of_180_multiples_of_15_l769_769794


namespace sum_of_yellow_and_blue_is_red_l769_769074

theorem sum_of_yellow_and_blue_is_red (a b : ℕ) : ∃ k : ℕ, (4 * a + 3) + (4 * b + 2) = 4 * k + 1 :=
by sorry

end sum_of_yellow_and_blue_is_red_l769_769074


namespace typeA_cloth_typeB_cloth_typeC_cloth_l769_769535

section ClothPrices

variables (CPA CPB CPC : ℝ)

theorem typeA_cloth :
  (300 * CPA * 0.90 = 9000) → CPA = 33.33 :=
by
  intro hCPA
  sorry

theorem typeB_cloth :
  (250 * CPB * 1.05 = 7000) → CPB = 26.67 :=
by
  intro hCPB
  sorry

theorem typeC_cloth :
  (400 * (CPC + 8) = 12000) → CPC = 22 :=
by
  intro hCPC
  sorry

end ClothPrices

end typeA_cloth_typeB_cloth_typeC_cloth_l769_769535


namespace smaller_side_of_new_rectangle_is_10_l769_769569

/-- We have a 10x25 rectangle that is divided into two congruent polygons and rearranged 
to form another rectangle. We need to prove that the length of the smaller side of the 
resulting rectangle is 10. -/
theorem smaller_side_of_new_rectangle_is_10 :
  ∃ (y x : ℕ), (y * x = 10 * 25) ∧ (y ≤ x) ∧ y = 10 := 
sorry

end smaller_side_of_new_rectangle_is_10_l769_769569


namespace lowest_exam_score_l769_769374

theorem lowest_exam_score 
  (first_exam_score : ℕ := 90) 
  (second_exam_score : ℕ := 108) 
  (third_exam_score : ℕ := 102) 
  (max_score_per_exam : ℕ := 120) 
  (desired_average : ℕ := 100) 
  (total_exams : ℕ := 5) 
  (total_score_needed : ℕ := desired_average * total_exams) : 
  ∃ (lowest_score : ℕ), lowest_score = 80 :=
by
  sorry

end lowest_exam_score_l769_769374


namespace Petya_can_ensure_root_difference_of_2014_l769_769019

theorem Petya_can_ensure_root_difference_of_2014 :
  ∀ a1 a2 : ℚ, ∃ a3 : ℚ, ∀ (r1 r2 r3 : ℚ),
    (r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3) ∧
    (r1, r2, r3 are roots of (λ x : ℚ, x^3 + a1 * x^2 + a2 * x + a3)) →
    (r1 - r2 = 2014 ∨ r1 - r2 = -2014 ∨
     r1 - r3 = 2014 ∨ r1 - r3 = -2014 ∨
     r2 - r3 = 2014 ∨ r2 - r3 = -2014) :=
by
  assume a1 a2 : ℚ
  have h : ∃ a3 : ℚ, ∀ (p : polynomial ℚ),
    (roots_of p = {0, 2014, r3}) ∨ (roots_of p = {0, r2, 2014})
  existsi a3
  sorry

end Petya_can_ensure_root_difference_of_2014_l769_769019


namespace revenue_decrease_10_percent_l769_769082

theorem revenue_decrease_10_percent (T C : ℝ) (hT : T > 0) (hC : C > 0) :
  let new_tax := 0.75 * T,
      new_consumption := 1.20 * C,
      original_revenue := T * C,
      new_revenue := new_tax * new_consumption in
  new_revenue = 0.90 * original_revenue :=
by
  sorry

end revenue_decrease_10_percent_l769_769082


namespace simplify_expression_l769_769955

theorem simplify_expression :
  (64^(1/3) - 216^(1/3) = -2) :=
by
  have h1 : 64 = 4^3 := by norm_num
  have h2 : 216 = 6^3 := by norm_num
  sorry

end simplify_expression_l769_769955


namespace probability_computation_l769_769261

noncomputable def P_xi_ge_minus_1 (xi : ℝ → ℝ) (μ : ℝ) : Prop :=
  ∀ x : ℝ, xi(μ) - 1 ≤ x * (1 - x)

noncomputable def normal_distribution (μ σ : ℝ) (x : ℝ) : ℝ :=
  (1 / (σ * sqrt (2 * π))) * exp (-(x - μ)^2 / (2*σ^2))

noncomputable def integral (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  real.Integral.integral (λ x, Ioc (f x) a b)

theorem probability_computation :
  (∀ xi : ℝ → ℝ, P_xi_ge_minus_1 xi (-1) → (integral (normal_distribution (-1) 1) 1 2 = 0.0214)) →
  true :=
by
  intro h
  sorry

end probability_computation_l769_769261


namespace smallest_c_for_poly_roots_l769_769073

theorem smallest_c_for_poly_roots (r s t : ℕ) (hposr : r > 0) (hposs : s > 0) (hpost : t > 0) (hpoly : r * s * t = 3080) (heq_poly : polynomial.X ^ 3 - polynomial.C (r + s + t) * polynomial.X ^ 2 + polynomial.C _ * polynomial.X - polynomial.C 3080 = 0) : r + s + t = 34 :=
sorry

end smallest_c_for_poly_roots_l769_769073


namespace find_a3_l769_769341

noncomputable def geometric_seq (a : ℕ → ℕ) (q : ℕ) : Prop :=
∀ n, a (n+1) = a n * q

theorem find_a3 (a : ℕ → ℕ) (q : ℕ) (h_geom : geometric_seq a q) (hq : q > 1)
  (h1 : a 4 - a 0 = 15) (h2 : a 3 - a 1 = 6) :
  a 2 = 4 :=
by
  sorry

end find_a3_l769_769341


namespace find_length_ED_l769_769345

variable {Point : Type}
variable [metric_space Point]

noncomputable def segment_length (A B : Point): length := sorry

def triangle := {A B C : Point}
def angle_bisector (A B C L : Point) := sorry
def collinear (A B C : Point) := sorry
def parallel (A B C D : Point) := sorry

theorem find_length_ED 
  (A B C L E D : Point) 
  (AE := 15) 
  (AC := 12) 
  (h1 : triangle A B C) 
  (h2 : angle_bisector A B C L)
  (h3 : E ∈ segment A B)
  (h4 : D ∈ segment B L)
  (h5 : length D L = length L C)
  (h6 : parallel E D A C): 
  segment_length E D = 3 :=
sorry

end find_length_ED_l769_769345


namespace find_lambda_l769_769703

-- Given conditions
variables (m n : ℝ^3) (λ : ℝ)
variables (h_nonzero_m : \| m \| ≠ 0 ∧ \| n \| ≠ 0)  -- Non-zero vectors
variables (h_angle : ∃ θ, θ = π / 3 ∧ θ = real.angle m n)  -- Angle π/3 between vectors
variables (h_lambda : ∥n∥ = λ * ∥m∥ ∧ λ > 0)  -- Magnitude condition
variables (h_min : min_possible_value (x1.dot y1 + x2.dot y2 + x3.dot y3) = 4 * (m.dot m))
where
  x1, x2, x3 : ℝ^3 -- Given x vectors
  y1, y2, y3 : ℝ^3 -- Given y vectors
  
-- Lean theorem statement
theorem find_lambda (m n : ℝ^3) (λ : ℝ)
  (h_nonzero_m : \|m\| ≠ 0 ∧ \|n\| ≠ 0)
  (h_angle : ∃ θ, θ = π / 3 ∧ θ = real.angle m n)
  (h_lambda : ∥n∥ = λ * ∥m∥ ∧ λ > 0)
  (h_min : min_possible_value (x1.dot y1 + x2.dot y2 + x3.dot y3) = 4 * (m.dot m)) :
  λ = 8 / 3 :=
sorry

end find_lambda_l769_769703


namespace solve_system_l769_769404

noncomputable theory

def system_of_eqs (x y z : ℝ) :=
  (log z / log (2 * x) = 3) ∧
  (log z / log (5 * y) = 6) ∧
  (log z / log (x * y) = 2/3)

theorem solve_system : 
  ∃ x y z : ℝ, system_of_eqs x y z ∧ 
    x = 1 / (2 * real.cbrt 10) ∧ 
    y = 1 / (5 * real.root 6 10) ∧ 
    z = 1 / 10 :=
by
  sorry

end solve_system_l769_769404


namespace part_one_part_two_part_three_l769_769640

section BinomialProof

variable (f : ℕ → ℝ → ℝ) (a : ℕ → ℝ) (n : ℕ)

-- Define f(x) = (1 + 2x)^n
def f (x : ℝ) : ℝ := (1 + 2 * x) ^ n

-- Assume the sum of the binomial coefficients is 64
axiom sum_binomial_coeffs : (f 1 = 64)

-- Questions:
-- 1. Find the value of a_2
def a₂ := a 2

-- 2. Find the term with the largest binomial coefficient
def largest_binomial_coeff_term := (∃ (k : ℕ), k * (binom n k) = 160)

-- 3. Find the value of a₁ + 2a₂ + 3a₃ + ... + naₙ
def sum_weighted_coeffs (n: ℕ) := ∑ k in range (n+1), k * a k

-- Proof problems:
theorem part_one : a₂ = 60 := sorry
theorem part_two : largest_binomial_coeff_term = (160 * x^3) := sorry
theorem part_three : sum_weighted_coeffs n = 2916 := sorry

end BinomialProof

end part_one_part_two_part_three_l769_769640


namespace total_number_of_distribution_methods_l769_769028

def cardDistributeWays : ℕ :=
  let cards := {1, 2, 3, 4, 5, 6}
  let envelopes : Finset (Finset ℕ) := 
    ({ {1, 2}, {3, 4}, {5, 6} } : Finset (Finset ℕ))
  let validDistributions := envelopes.filter (λ env, {1, 2} ∈ env)
  validDistributions.card

theorem total_number_of_distribution_methods : cardDistributeWays = 36 :=
by 
  sorry

end total_number_of_distribution_methods_l769_769028


namespace conference_center_people_l769_769154

theorem conference_center_people (rooms : ℕ) (capacity_per_room : ℕ) (occupancy_fraction : ℚ) (total_people : ℕ) :
  rooms = 6 →
  capacity_per_room = 80 →
  occupancy_fraction = 2/3 →
  total_people = rooms * capacity_per_room * occupancy_fraction →
  total_people = 320 := 
by
  intros h_rooms h_capacity h_fraction h_total
  rw [h_rooms, h_capacity, h_fraction] at h_total
  norm_num at h_total
  exact h_total

end conference_center_people_l769_769154


namespace min_disks_required_l769_769045

-- Define the initial conditions
def num_files : ℕ := 40
def disk_capacity : ℕ := 2 -- capacity in MB
def num_files_1MB : ℕ := 5
def num_files_0_8MB : ℕ := 15
def num_files_0_5MB : ℕ := 20
def size_1MB : ℕ := 1
def size_0_8MB : ℕ := 8/10 -- 0.8 MB
def size_0_5MB : ℕ := 1/2 -- 0.5 MB

-- Define the mathematical problem
theorem min_disks_required :
  (num_files_1MB * size_1MB + num_files_0_8MB * size_0_8MB + num_files_0_5MB * size_0_5MB) / disk_capacity ≤ 15 := by
  sorry

end min_disks_required_l769_769045


namespace inequality_holds_l769_769954

theorem inequality_holds (x a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : (a < c ∧ c < b) ∨ (b < c ∧ c < a)) 
  (h5 : (x - a) * (x - b) * (x - c) > 0) :
  (1 / (x - a)) + (1 / (x - b)) > 1 / (x - c) := 
by sorry

end inequality_holds_l769_769954


namespace count_valid_integers_l769_769707

theorem count_valid_integers : 
  {n : ℕ // 2000 < n ∧ n < 3000 ∧ ∃ b c d : ℕ, n = 3000 + b * 100 + c * 10 + d ∧ d = 3 + b + c ∧ b % 2 = 0 ∧ b < 10 ∧ c < 10 ∧ d < 10}.to_finset.card = 16 :=
by sorry

end count_valid_integers_l769_769707


namespace female_computer_literacy_l769_769122

variable (E F C M CM CF : ℕ)

theorem female_computer_literacy (hE : E = 1200) 
                                (hF : F = 720) 
                                (hC : C = 744) 
                                (hM : M = 480) 
                                (hCM : CM = 240) 
                                (hCF : CF = C - CM) : 
                                CF = 504 :=
by {
  sorry
}

end female_computer_literacy_l769_769122


namespace time_to_cross_bridge_l769_769173

-- Given conditions
def train_length : ℕ := 150
def bridge_length : ℕ := 225
def train_speed_kmh : ℕ := 45

-- Proving the time taken for the train to cross the bridge
theorem time_to_cross_bridge : 
  let total_distance := train_length + bridge_length in
  let speed_mps := (train_speed_kmh * 1000) / 3600 in
  (total_distance / speed_mps) = 30 := 
by
  sorry

end time_to_cross_bridge_l769_769173


namespace car_wait_time_is_3_point_8_l769_769157

-- Definitions
def initial_speed_cyclist : ℝ := 15 -- mph
def speed_car : ℝ := 60 -- mph
def distance_uphill : ℝ := 1 -- mile
def final_speed_cyclist : ℝ := 10 -- mph

-- Average speed of the cyclist
def average_speed_cyclist : ℝ := (initial_speed_cyclist + final_speed_cyclist) / 2

-- Time taken by the car to travel the distance
def time_car : ℚ := (distance_uphill / speed_car) * 60 -- converting from hours to minutes

-- Time taken by the cyclist to travel the distance
def time_cyclist : ℚ := (distance_uphill / average_speed_cyclist) * 60 -- converting from hours to minutes

-- Wait time for the car
def wait_time : ℚ := time_cyclist - time_car

-- The proof statement
theorem car_wait_time_is_3_point_8 : wait_time = 3.8 := 
by 
  -- Placeholder for actual proof
  sorry

end car_wait_time_is_3_point_8_l769_769157


namespace work_done_l769_769138

-- Define the piecewise force function F(x)
def F (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 2 then 10
  else if 2 < x then 3 * x + 4
  else 0

-- The work done by force F(x) over the interval [0, 4]
theorem work_done : ∫ x in 0..4, F x = 46 :=
by
  sorry

end work_done_l769_769138


namespace exists_min_a_l769_769220

open Real

theorem exists_min_a (x y z : ℝ) : 
  (∃ x y z : ℝ, (sqrt (x - 1) + sqrt (y - 1) + sqrt (z - 1) = (11/2 - 1)) ∧ 
  (sqrt (x + 1) + sqrt (y + 1) + sqrt (z + 1) = (11/2 + 1))) :=
sorry

end exists_min_a_l769_769220


namespace increasing_digits_count_l769_769285

theorem increasing_digits_count : 
  ∃ n, n = 120 ∧ ∀ x : ℕ, x ≤ 1000 → (∀ i j : ℕ, i < j → ((x / 10^i % 10) < (x / 10^j % 10)) → 
  x ≤ 1000 ∧ (x / 10^i % 10) ≠ (x / 10^j % 10)) :=
sorry

end increasing_digits_count_l769_769285


namespace equal_angles_in_right_angle_quadrilateral_l769_769342

theorem equal_angles_in_right_angle_quadrilateral
  (A B C D : Type*)
  [EuclideanGeometry A B C D]
  (hA : ∠ A = 90°)
  (hC : ∠ C = 90°) :
  ∠ CBD = ∠ CAD :=
by
  sorry

end equal_angles_in_right_angle_quadrilateral_l769_769342


namespace factors_of_180_l769_769775

-- Define a predicate to check if a number is a factor of another number
def is_factor (m n : ℕ) : Prop := n % m = 0

-- Define a predicate to check if a number is a multiple of another number
def is_multiple (m n : ℕ) : Prop := m % n = 0

-- Define the problem
theorem factors_of_180 : {n : ℕ // is_factor n 180 ∧ is_multiple n 15}.card = 6 :=
by sorry

end factors_of_180_l769_769775


namespace sum_of_variables_l769_769828

theorem sum_of_variables (x y z : ℝ) (h₁ : x + y = 1) (h₂ : y + z = 1) (h₃ : z + x = 1) : x + y + z = 3 / 2 := 
sorry

end sum_of_variables_l769_769828


namespace exterior_angle_of_regular_pentagon_l769_769182

theorem exterior_angle_of_regular_pentagon : 
  (360 / 5) = 72 := by
  sorry

end exterior_angle_of_regular_pentagon_l769_769182


namespace square_area_from_diagonal_l769_769542

theorem square_area_from_diagonal (d : ℝ) (h : d = 8 * Real.sqrt 2) : (d / Real.sqrt 2) ^ 2 = 64 :=
by
  sorry

end square_area_from_diagonal_l769_769542


namespace min_distance_line_l769_769246

theorem min_distance_line
  (x y : ℝ)
  (h : x + y + 1 = 0) :
  sqrt ((x + 2)^2 + (y + 3)^2) = 2 * sqrt 2 :=
sorry

end min_distance_line_l769_769246


namespace factors_of_180_multiple_of_15_count_l769_769799

theorem factors_of_180_multiple_of_15_count :
  ∃ n : Nat, n = 6 ∧ ∀ x : Nat, x ∣ 180 → 15 ∣ x → x > 0 :=
by
  sorry

end factors_of_180_multiple_of_15_count_l769_769799


namespace heights_form_rectangle_l769_769879

variables (A B C D M N P Q : Type)
variables [parallelogram A B C D]
variables (h1 : is_height B M A D) (h2 : is_height B N D C)
variables (h3 : is_height D P A B) (h4 : is_height D Q B C)

theorem heights_form_rectangle : is_rectangle M N P Q :=
by sorry

end heights_form_rectangle_l769_769879


namespace alt_sum_seq_zero_l769_769131

-- Define the sequence of integers from 1 to 1996
def seq : List ℤ := List.range' 1 1996

-- Define the alternating sum pattern
def alt_sum (l : List ℤ) : ℤ :=
  l.enum.foldl (λ acc ⟨i, x⟩, acc + (-1) ^ i * x) 0

-- Claim that the alternating sum of the sequence is 0
theorem alt_sum_seq_zero : alt_sum seq = 0 := 
by
  sorry

end alt_sum_seq_zero_l769_769131


namespace correct_operation_l769_769497

theorem correct_operation (a b : ℝ) :
  ¬ (a^2 + a^3 = a^6) ∧
  ¬ ((a*b)^2 = a*(b^2)) ∧
  ¬ ((a + b)^2 = a^2 + b^2) ∧
  ((a + b)*(a - b) = a^2 - b^2) := 
by
  sorry

end correct_operation_l769_769497


namespace exists_k_with_n_prime_divisors_l769_769031

theorem exists_k_with_n_prime_divisors (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  ∃ k : ℕ, 0 < k ∧ (∃ p : Finset ℕ, (p.card = n ∧ ∀ q ∈ p, prime q ∧ q ∣ (2^k - m))) :=
sorry

end exists_k_with_n_prime_divisors_l769_769031


namespace hydrogen_atoms_in_compound_l769_769524

theorem hydrogen_atoms_in_compound :
  ∀ (n : ℕ), 98 = 14 + n + 80 → n = 4 :=
by intro n h_eq
   sorry

end hydrogen_atoms_in_compound_l769_769524


namespace count_factors_of_180_multiple_of_15_l769_769776

def is_factor (n k : ℕ) : Prop := n % k = 0

def is_multiple_of (k m : ℕ) : Prop := k % m = 0

theorem count_factors_of_180_multiple_of_15 : 
  (finset.filter (λ f, is_factor 180 f ∧ is_multiple_of f 15) (finset.range (181))).card = 6 := 
by 
sorry

end count_factors_of_180_multiple_of_15_l769_769776


namespace Michael_needs_more_money_l769_769387

def money_Michael_has : ℕ := 50
def cake_cost : ℕ := 20
def bouquet_cost : ℕ := 36
def balloons_cost : ℕ := 5

def total_cost : ℕ := cake_cost + bouquet_cost + balloons_cost
def money_needed : ℕ := total_cost - money_Michael_has

theorem Michael_needs_more_money : money_needed = 11 :=
by
  sorry

end Michael_needs_more_money_l769_769387


namespace solve_system_l769_769957

theorem solve_system :
  ∃ x y z : ℝ, 
    (x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) ∧
     (xy / (5x + 4y) = 6) ∧ 
     (xz / (3x + 2z) = 8) ∧ 
     (yz / (3y + 5z) = 6) ∧ 
     (x = 48 ∧ y = 16 ∧ z = 12) :=
by
  sorry

end solve_system_l769_769957


namespace fraction_defined_iff_l769_769847

theorem fraction_defined_iff (x : ℝ) : (1 / (x + 1) ≠ ⊥) ↔ x ≠ -1 := 
by
  sorry

end fraction_defined_iff_l769_769847


namespace factors_of_180_multiple_of_15_l769_769744

/-- A factor of a number n is a positive integer which divides n without leaving a remainder. -/
def is_factor (n k : ℕ) : Prop :=
k > 0 ∧ n % k = 0

/-- A multiple of a number m is an integer which is divisible by m without leaving a remainder. -/
def is_multiple (k m : ℕ) : Prop :=
k % m = 0

/-- There are 6 factors of 180 that are also multiples of 15. -/
theorem factors_of_180_multiple_of_15 : 
  (Finset.filter (λ k, is_multiple k 15) { x | is_factor 180 x }.to_finset).card = 6 :=
by 
  sorry

end factors_of_180_multiple_of_15_l769_769744


namespace percentage_change_area_l769_769424

theorem percentage_change_area (L B : ℝ) :
  let A_original := L * B
  let A_new := (L / 2) * (3 * B)
  (A_new - A_original) / A_original * 100 = 50 := by
  sorry

end percentage_change_area_l769_769424


namespace num_factors_of_180_multiple_of_15_l769_769736

def is_factor (n m : Nat) : Prop := m % n = 0
def is_multiple_of_15 (n : Nat) : Prop := is_factor 15 n

theorem num_factors_of_180_multiple_of_15 : 
  (Nat.filter (fun n => is_factor n 180 ∧ is_multiple_of_15 n) (Nat.range (180 + 1))).length = 6 :=
sorry

end num_factors_of_180_multiple_of_15_l769_769736


namespace convert_base5_to_base7_is_correct_l769_769202

-- Definitions of base conversion 
def convert_base5_to_decimal (n : ℕ) : ℕ :=
  let digits := [2, 1, 4] in -- for 412 base 5
  digits[0] * 5^0 + digits[1] * 5^1 + digits[2] * 5^2

def convert_decimal_to_base7 (n : ℕ) : list ℕ := 
  [n % 7, (n / 7) % 7, (n / (7 * 7)) % 7]

-- Problem statement:
theorem convert_base5_to_base7_is_correct :
  convert_base5_to_decimal 412 = 107 → convert_decimal_to_base7 107 = [2, 1, 2] :=
by
  sorry

end convert_base5_to_base7_is_correct_l769_769202


namespace peggy_dolls_ratio_l769_769025

noncomputable def peggy_dolls_original := 6
noncomputable def peggy_dolls_from_grandmother := 30
noncomputable def peggy_dolls_total := 51

theorem peggy_dolls_ratio :
  ∃ x, peggy_dolls_original + peggy_dolls_from_grandmother + x = peggy_dolls_total ∧ x / peggy_dolls_from_grandmother = 1 / 2 :=
by {
  sorry
}

end peggy_dolls_ratio_l769_769025


namespace count_factors_180_multiple_of_15_is_6_l769_769811

-- We define a function to check if 'n' is a factor of 'm'
def is_factor (m n : ℕ) : Prop := n ∣ m

-- We define a function to check if 'n' is a multiple of 'd'
def is_multiple (n d : ℕ) : Prop := d ∣ n

-- We define a function to collect all factors of a number 'n'
def factors (n : ℕ) : list ℕ := 
  -- We use list.filter to keep elements that satisfy is_factor n
  list.filter (λ x, is_factor n x) (list.range (n + 1))

-- We count the number of elements in a list that satisfy a predicate
def count_satisfying {α : Type*} (pred : α → Prop) [decidable_pred pred] (l : list α) : ℕ :=
  list.length (list.filter pred l)

-- Noncomputable definition to specify that counting number
noncomputable def count_positive_factors_of_180_multiple_of_15 : ℕ :=
  count_satisfying (λ x, is_multiple x 15) (factors 180)

-- We state the theorem equivalence to the proof problem
theorem count_factors_180_multiple_of_15_is_6 : count_positive_factors_of_180_multiple_of_15 = 6 :=
  sorry


end count_factors_180_multiple_of_15_is_6_l769_769811


namespace sum_of_digits_congruence_l769_769355

-- Definitions and conditions
variables (a b b' c m q : ℕ) (M : ℕ)
variable (S_q : ℕ → ℕ)
variables (h0 : 0 < a) (h1 : 0 < b) (h2 : 0 < b') 
          (h3 : 0 < c) (h4 : 1 < m) (h5 : 1 < q)
          (h6 : |b - b'| ≥ a)
          (h7 : ∀ n, n ≥ M → S_q (a * n + b) ≡ S_q (a * n + b') + c [MOD m])

-- Theorem statement
theorem sum_of_digits_congruence : ∀ n, 0 < n → S_q (a * n + b) ≡ S_q (a * n + b') + c [MOD m] :=
by intros n hn; sorry

end sum_of_digits_congruence_l769_769355


namespace sum_of_first_5_b_n_l769_769251

theorem sum_of_first_5_b_n
  (a : ℕ → ℕ)
  (h1 : a 2 = 6)
  (h2 : a 5 = 15)
  (b : ℕ → ℕ := λ n, a (2 * n))
  : (Finset.range 5).sum (λ n, b (n + 1)) = 90 :=
sorry

end sum_of_first_5_b_n_l769_769251


namespace factors_of_180_l769_769767

-- Define a predicate to check if a number is a factor of another number
def is_factor (m n : ℕ) : Prop := n % m = 0

-- Define a predicate to check if a number is a multiple of another number
def is_multiple (m n : ℕ) : Prop := m % n = 0

-- Define the problem
theorem factors_of_180 : {n : ℕ // is_factor n 180 ∧ is_multiple n 15}.card = 6 :=
by sorry

end factors_of_180_l769_769767


namespace count_positive_factors_of_180_multiple_of_15_l769_769721

statement: ℕ
statement := sorry

theorem count_positive_factors_of_180_multiple_of_15 :
  let factors_of_180 := {d ∈ (finset.range 181).filter (λ x, 180 % x = 0)}
  let multiples_of_15 := (finset.range 181).filter (λ x, x % 15 = 0)
  let desired_factors := factors_of_180 ∩ multiples_of_15
  desired_factors.card = 6 :=
by
  sorry

end count_positive_factors_of_180_multiple_of_15_l769_769721


namespace boat_downstream_distance_l769_769520

theorem boat_downstream_distance (V_b V_s : ℝ) (t_downstream t_upstream : ℝ) (d_upstream : ℝ) 
  (h1 : t_downstream = 8) (h2 : t_upstream = 15) (h3 : d_upstream = 75) (h4 : V_s = 3.75) 
  (h5 : V_b - V_s = (d_upstream / t_upstream)) : (V_b + V_s) * t_downstream = 100 :=
by
  sorry

end boat_downstream_distance_l769_769520


namespace travel_time_seattle_to_lasvegas_l769_769582

def distance_seattle_boise : ℝ := 640
def distance_boise_saltlakecity : ℝ := 400
def distance_saltlakecity_phoenix : ℝ := 750
def distance_phoenix_lasvegas : ℝ := 300

def speed_highway_seattle_boise : ℝ := 80
def speed_city_seattle_boise : ℝ := 35

def speed_highway_boise_saltlakecity : ℝ := 65
def speed_city_boise_saltlakecity : ℝ := 25

def speed_highway_saltlakecity_denver : ℝ := 75
def speed_city_saltlakecity_denver : ℝ := 30

def speed_highway_denver_phoenix : ℝ := 70
def speed_city_denver_phoenix : ℝ := 20

def speed_highway_phoenix_lasvegas : ℝ := 50
def speed_city_phoenix_lasvegas : ℝ := 30

def city_distance_estimate : ℝ := 10

noncomputable def total_time : ℝ :=
  let time_seattle_boise := ((distance_seattle_boise - city_distance_estimate) / speed_highway_seattle_boise) + (city_distance_estimate / speed_city_seattle_boise)
  let time_boise_saltlakecity := ((distance_boise_saltlakecity - city_distance_estimate) / speed_highway_boise_saltlakecity) + (city_distance_estimate / speed_city_boise_saltlakecity)
  let time_saltlakecity_phoenix := ((distance_saltlakecity_phoenix - city_distance_estimate) / speed_highway_saltlakecity_denver) + (city_distance_estimate / speed_city_saltlakecity_denver)
  let time_phoenix_lasvegas := ((distance_phoenix_lasvegas - city_distance_estimate) / speed_highway_phoenix_lasvegas) + (city_distance_estimate / speed_city_phoenix_lasvegas)
  time_seattle_boise + time_boise_saltlakecity + time_saltlakecity_phoenix + time_phoenix_lasvegas

theorem travel_time_seattle_to_lasvegas :
  total_time = 30.89 :=
sorry

end travel_time_seattle_to_lasvegas_l769_769582


namespace simplify_fraction_l769_769046

theorem simplify_fraction :
  (4^5 + 4^3) / (4^4 - 4^2 + 2) = 544 / 121 := 
by sorry

end simplify_fraction_l769_769046


namespace pizza_combinations_l769_769390

theorem pizza_combinations (n r k : ℕ) (h_n : n = 9) (h_r : r = 4) (h_k : k = 2) :
  Nat.choose n r - Nat.choose (n - k) r = 91 :=
by
  have h1 : Nat.choose 9 4 = 126 := by sorry
  have h2 : Nat.choose 7 4 = 35 := by sorry
  rw [h_n, h_r, h_k]
  calc
    Nat.choose 9 4 - Nat.choose (9 - 2) 4 = 126 - 35 := by rw [h1, h2]
    ... = 91 := by sorry

end pizza_combinations_l769_769390


namespace distance_between_2x_plus_y_minus_3_and_4x_plus_2y_minus_1_l769_769065

noncomputable def distance_between_parallel_lines (A B c1 c2 : ℝ) : ℝ :=
  |c2 - c1| / real.sqrt (A^2 + B^2)

theorem distance_between_2x_plus_y_minus_3_and_4x_plus_2y_minus_1 :
  distance_between_parallel_lines 4 2 (-6) (-1) = real.sqrt 5 / 2 :=
by
  sorry

end distance_between_2x_plus_y_minus_3_and_4x_plus_2y_minus_1_l769_769065


namespace trapezoid_induction_proof_l769_769362

/-- Let E be the intersection point of the lateral sides AD and BC of the trapezoid ABCD.
Let B_(n+1) be the intersection point of the lines A_n C and BD (A_0 = A),
and A_(n+1) be the intersection point of the lines EB_(n+1) and AB. -/
variables (A B C D E : Point) (A_n B_n : ℕ → Point)
variables (n : ℕ)

-- Hypothesize the conditions
hypotheses (h1 : intersect AD BC = E)
hypotheses (h2 : ∀ n, intersect (A_n n) C BD = B_n (n+1))
hypotheses (h3 : ∀ n, intersect E (B_n (n+1)) AB = A_n (n+1))
hypotheses (A_0 : A_n 0 = A)

-- Define the proof problem
theorem trapezoid_induction_proof : ∀ n : ℕ, distance (A_n n) B = distance A B / (n+1) :=
by
  sorry

end trapezoid_induction_proof_l769_769362


namespace percentage_change_in_area_of_rectangle_l769_769442

theorem percentage_change_in_area_of_rectangle
  (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  let A1 := L * B
  let A2 := (L / 2) * (3 * B)
  let percentage_change := ((A2 - A1) / A1) * 100
  percentage_change = 50 := by
  let A1 := L * B
  let A2 := (L / 2) * (3 * B)
  let percentage_change := ((A2 - A1) / A1) * 100
  calc percentage_change
      = ((A2 - A1) / A1) * 100 : rfl
  ... = ((((L / 2) * (3 * B)) - (L * B)) / (L * B)) * 100 : by rw [A1, A2]
  ... = ((3 / 2 * A1 - A1) / A1) * 100 : by rw [A2, ←mul_assoc, mul_div_cancel' _ (two_ne_zero : 2 ≠ 0)]
  ... = ((3 / 2 - 1) * A1 / A1) * 100 : by simp only [mul_sub, mul_div_cancel' _ (ne_of_gt hL)]
  ... = (1 / 2) * 100 : by rw [div_mul_cancel' (ne_of_gt (lt_of_lt_of_le zero_lt_one (le_of_lt ((mul_lt_iff_lt_one_left (lt_of_lt_of_le zero_lt_one hL)).mpr (half_pos (lt_of_le_of_ne (le_of_lt (lt_of_lt_of_le (lt_one_mul_self (lt_of_le_of_lt (zero_le_two) (lt_add_one 1))) (lt_of_lt_of_le (two_le_one) (lt_one 2))))))))) so))
  ... = 50 : rfl

end percentage_change_in_area_of_rectangle_l769_769442


namespace no_distinct_roots_exist_l769_769595

theorem no_distinct_roots_exist :
  ¬ ∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
  (a^2 - 2 * b * a + c^2 = 0) ∧
  (b^2 - 2 * c * b + a^2 = 0) ∧ 
  (c^2 - 2 * a * c + b^2 = 0) := 
sorry

end no_distinct_roots_exist_l769_769595


namespace percentage_change_in_area_halved_length_tripled_breadth_l769_769429

def original_area (L B : ℝ) : ℝ := L * B
def new_area (L B : ℝ) : ℝ := (L / 2) * (3 * B)

def percentage_change (A1 A2 : ℝ) : ℝ :=
  ((A2 - A1) / A1) * 100

theorem percentage_change_in_area_halved_length_tripled_breadth
  (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  percentage_change (original_area L B) (new_area L B) = 50 :=
begin
  sorry
end

end percentage_change_in_area_halved_length_tripled_breadth_l769_769429


namespace average_of_three_quantities_l769_769409

theorem average_of_three_quantities 
  (five_avg : ℚ) (three_avg : ℚ) (two_avg : ℚ) 
  (h_five_avg : five_avg = 10) 
  (h_two_avg : two_avg = 19) : 
  three_avg = 4 := 
by 
  let sum_5 := 5 * 10
  let sum_2 := 2 * 19
  let sum_3 := sum_5 - sum_2
  let three_avg := sum_3 / 3
  sorry

end average_of_three_quantities_l769_769409


namespace triangle_is_obtuse_l769_769644

theorem triangle_is_obtuse
  (α : ℝ)
  (h1 : α > 0 ∧ α < π)
  (h2 : Real.sin α + Real.cos α = 2 / 3) :
  ∃ β γ, β > 0 ∧ β < π ∧ γ > 0 ∧ γ < π ∧ β + γ + α = π ∧ (α > π / 2 ∨ β > π / 2 ∨ γ > π / 2) :=
sorry

end triangle_is_obtuse_l769_769644


namespace PetyaCanAlwaysForceDifferenceOfRoots2014_l769_769015

noncomputable def canPetyaForceDifferenceOfRoots2014 : Prop :=
  ∀ (v1 v2 : ℚ) (vasyachooses : (ℚ → ℚ) → Prop), (∃ p q : ℚ, vasyachooses (λ _: ℚ, _) ∧ vasyachooses (λ _: ℚ, _)) →
  ∃ (a b c : ℚ), 
    (vasyachooses (λ _: ℚ, a) ∧ vasyachooses (λ _: ℚ, c)) ∨
    (vasyachooses (λ _: ℚ, b) ∧ vasyachooses (λ _: ℚ, c)) ∧
    (∀ x y : ℚ, (x^3 + a*x^2 + b*x + c = 0 → y^3 + a*y^2 + b*y + c = 0 → abs(x - y) = 2014))

theorem PetyaCanAlwaysForceDifferenceOfRoots2014 : canPetyaForceDifferenceOfRoots2014 :=
sorry

end PetyaCanAlwaysForceDifferenceOfRoots2014_l769_769015


namespace cube_painting_l769_769560

theorem cube_painting (n : ℕ) (h1 : n > 3) 
  (h2 : 2 * (n-2) * (n-2) = 4 * (n-2)) :
  n = 4 :=
sorry

end cube_painting_l769_769560


namespace count_factors_of_180_multiple_of_15_l769_769783

def is_factor (n k : ℕ) : Prop := n % k = 0

def is_multiple_of (k m : ℕ) : Prop := k % m = 0

theorem count_factors_of_180_multiple_of_15 : 
  (finset.filter (λ f, is_factor 180 f ∧ is_multiple_of f 15) (finset.range (181))).card = 6 := 
by 
sorry

end count_factors_of_180_multiple_of_15_l769_769783


namespace dice_even_sum_probability_l769_769469

theorem dice_even_sum_probability :
  (let outcomes_die1 := {1, 2, 3, 4}.card; 
       outcomes_die2 := {1, 2, 3, 4, 5, 6, 7, 8}.card in 
   ∑ (x ∈ {1, 2, 3, 4}), ∑ (y ∈ {1, 2, 3, 4, 5, 6, 7, 8}), if (x + y) % 2 = 0 then 1 else 0) 
 = 1 / 2 := 
by 
  let outcomes_die1 := {1, 2, 3, 4}.card
  let outcomes_die2 := {1, 2, 3, 4, 5, 6, 7, 8}.card
  have h1: outcomes_die1 = 4 := rfl
  have h2: outcomes_die2 = 8 := rfl
  sorry

end dice_even_sum_probability_l769_769469


namespace max_value_y_is_4_l769_769619

noncomputable def max_value_y : ℝ :=
  let y := λ x : ℝ, sqrt (2 * x - 3) + sqrt (2 * x) + sqrt (7 - 3 * x)
  in Sup (set.range y)

theorem max_value_y_is_4 :
  max_value_y = 4 := sorry

end max_value_y_is_4_l769_769619


namespace percentage_change_in_area_halved_length_tripled_breadth_l769_769426

def original_area (L B : ℝ) : ℝ := L * B
def new_area (L B : ℝ) : ℝ := (L / 2) * (3 * B)

def percentage_change (A1 A2 : ℝ) : ℝ :=
  ((A2 - A1) / A1) * 100

theorem percentage_change_in_area_halved_length_tripled_breadth
  (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  percentage_change (original_area L B) (new_area L B) = 50 :=
begin
  sorry
end

end percentage_change_in_area_halved_length_tripled_breadth_l769_769426


namespace car_fuel_tanks_l769_769053

theorem car_fuel_tanks {x X p : ℝ}
  (h1 : x + X = 70)            -- Condition: total capacity is 70 liters
  (h2 : x * p = 45)            -- Condition: cost to fill small car's tank
  (h3 : X * (p + 0.29) = 68)   -- Condition: cost to fill large car's tank
  : x = 30 ∧ X = 40            -- Conclusion: capacities of the tanks
  :=
by {
  sorry
}

end car_fuel_tanks_l769_769053


namespace janine_test_score_l769_769022

theorem janine_test_score :
  let num_mc := 10
  let p_mc := 0.80
  let num_sa := 30
  let p_sa := 0.70
  let total_questions := 40
  let correct_mc := p_mc * num_mc
  let correct_sa := p_sa * num_sa
  let total_correct := correct_mc + correct_sa
  (total_correct / total_questions) * 100 = 72.5 := 
by
  sorry

end janine_test_score_l769_769022


namespace evaluate_complex_pow_l769_769216

open Complex

noncomputable def calc : ℂ := (-64 : ℂ) ^ (7 / 6)

theorem evaluate_complex_pow : calc = 128 * Complex.I := by 
  -- Recognize that (-64) = (-4)^3
  -- Apply exponent rules: ((-4)^3)^(7/6) = (-4)^(3 * 7/6) = (-4)^(7/2)
  -- Simplify (-4)^(7/2) = √((-4)^7) = √(-16384)
  -- Calculation (-4)^7 = -16384
  -- Simplify √(-16384) = 128i
  sorry

end evaluate_complex_pow_l769_769216


namespace brenda_age_l769_769562

theorem brenda_age (A B J : ℕ) (h1 : A = 3 * B) (h2 : J = B + 10) (h3 : A = J) : B = 5 :=
sorry

end brenda_age_l769_769562


namespace regression_line_estimate_l769_769467

section
variables (x y : ℕ → ℝ)
noncomputable def sum_x  := ∑ i in finset.range 10, x i
noncomputable def sum_y  := ∑ i in finset.range 10, y i
noncomputable def sum_x2 := ∑ i in finset.range 10, (x i) ^ 2
noncomputable def sum_xy := ∑ i in finset.range 10, (x i - (1/10 : ℝ) * sum_x) * (y i - (1/10 : ℝ) * sum_y)

theorem regression_line_estimate :
  sum_x = 30 →
  sum_y = 600 →
  sum_x2 = 250 →
  sum_xy = 400 →
  ∃ a b : ℝ, a = 52.5 ∧ b = 2.5 ∧ (∀ x, y = b * x + a) ∧ (y 5 = 65) :=
begin
  intros h_sum_x h_sum_y h_sum_x2 h_sum_xy,
  sorry -- Proof to be done
end

end

end regression_line_estimate_l769_769467


namespace Petya_can_ensure_root_difference_of_2014_l769_769021

theorem Petya_can_ensure_root_difference_of_2014 :
  ∀ a1 a2 : ℚ, ∃ a3 : ℚ, ∀ (r1 r2 r3 : ℚ),
    (r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3) ∧
    (r1, r2, r3 are roots of (λ x : ℚ, x^3 + a1 * x^2 + a2 * x + a3)) →
    (r1 - r2 = 2014 ∨ r1 - r2 = -2014 ∨
     r1 - r3 = 2014 ∨ r1 - r3 = -2014 ∨
     r2 - r3 = 2014 ∨ r2 - r3 = -2014) :=
by
  assume a1 a2 : ℚ
  have h : ∃ a3 : ℚ, ∀ (p : polynomial ℚ),
    (roots_of p = {0, 2014, r3}) ∨ (roots_of p = {0, r2, 2014})
  existsi a3
  sorry

end Petya_can_ensure_root_difference_of_2014_l769_769021


namespace probability_b_l769_769668

theorem probability_b (p : Set ℝ → ℝ) (a b : Set ℝ)  
  (h1 : p a = 6 / 17) 
  (h2 : p (a ∪ b) = 4 / 17) 
  (h3 : p (b ∩ a) / p a = 2 / 3) : 
  p b = 2 / 17 :=
by sorry

end probability_b_l769_769668


namespace factors_of_180_multiples_of_15_l769_769791

theorem factors_of_180_multiples_of_15 :
  (finset.filter (λ x : ℕ, x % 15 = 0) (finset.filter (λ x : ℕ, 180 % x = 0) (finset.range (180 + 1)))).card = 6 :=
sorry

end factors_of_180_multiples_of_15_l769_769791


namespace scaling_transformation_cosine_l769_769881

theorem scaling_transformation_cosine (x y : ℝ) (h : y = (1 / 3) * cos (2 * x)) 
  (x' : ℝ) (hx' : x' = 2 * x) (hy' : y' = 3 * y) : y' = cos x' :=
sorry

end scaling_transformation_cosine_l769_769881


namespace power_of_negative_base_l769_769214

theorem power_of_negative_base : (-64 : ℤ)^(7 / 6) = -128 := by
  sorry

end power_of_negative_base_l769_769214


namespace light_distance_200_years_l769_769066

-- Define the distance light travels in one year.
def distance_one_year := 5870000000000

-- Define the scientific notation representation for distance in one year
def distance_one_year_sci := 587 * 10^10

-- Define the distance light travels in 200 years.
def distance_200_years := distance_one_year * 200

-- Define the expected distance in scientific notation for 200 years.
def expected_distance := 1174 * 10^12

-- The theorem stating the given condition and the conclusion to prove
theorem light_distance_200_years : distance_200_years = expected_distance :=
by
  -- skipping the proof
  sorry

end light_distance_200_years_l769_769066


namespace Petya_can_ensure_root_difference_of_2014_l769_769020

theorem Petya_can_ensure_root_difference_of_2014 :
  ∀ a1 a2 : ℚ, ∃ a3 : ℚ, ∀ (r1 r2 r3 : ℚ),
    (r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3) ∧
    (r1, r2, r3 are roots of (λ x : ℚ, x^3 + a1 * x^2 + a2 * x + a3)) →
    (r1 - r2 = 2014 ∨ r1 - r2 = -2014 ∨
     r1 - r3 = 2014 ∨ r1 - r3 = -2014 ∨
     r2 - r3 = 2014 ∨ r2 - r3 = -2014) :=
by
  assume a1 a2 : ℚ
  have h : ∃ a3 : ℚ, ∀ (p : polynomial ℚ),
    (roots_of p = {0, 2014, r3}) ∨ (roots_of p = {0, r2, 2014})
  existsi a3
  sorry

end Petya_can_ensure_root_difference_of_2014_l769_769020


namespace chocolates_total_l769_769086

theorem chocolates_total (x : ℕ)
  (h1 : x - 12 + x - 18 + x - 20 = 2 * x) :
  x = 50 :=
  sorry

end chocolates_total_l769_769086


namespace intersect_complement_l769_769701

open Set

def U := {x : ℤ | -1 ≤ x ∧ x ≤ 5}
def A := {1, 2, 5} : Set ℤ
def B := {x : ℕ | -1 < x ∧ x < 4}

def C_U (A : Set ℤ) := U \ A

theorem intersect_complement (U : Set ℤ) (A : Set ℤ) (B : Set ℕ) :
  B ∩ (U \ A) = {0, 3} :=
by sorry

end intersect_complement_l769_769701


namespace largest_d_l769_769206

theorem largest_d (x : Fin 51 → ℝ) (M : ℝ)
  (h1 : ∑ i, x i = 100)
  (h2 : (∃ i, x 25 = i ∧ M = x 25) ∧ sorted (≤) x) :
  ∑ i, (x i) ^ 2 ≥ (104 / 25) * M ^ 2 :=
sorry

end largest_d_l769_769206


namespace kite_area_l769_769089

theorem kite_area (d1 d2 : ℝ) (h1 : d1 = 9) (h2 : d2 = 14) : 
  (d1 * d2) / 2 = 63 :=
by
  rw [h1, h2]
  norm_num
  exact sorry

end kite_area_l769_769089


namespace wilsons_theorem_l769_769953

theorem wilsons_theorem (p : ℕ) [hp : Fact p.Prime] : (p - 1)! ≡ -1 [MOD p] := 
sorry

end wilsons_theorem_l769_769953


namespace count_positive_factors_of_180_multiple_of_15_l769_769727

statement: ℕ
statement := sorry

theorem count_positive_factors_of_180_multiple_of_15 :
  let factors_of_180 := {d ∈ (finset.range 181).filter (λ x, 180 % x = 0)}
  let multiples_of_15 := (finset.range 181).filter (λ x, x % 15 = 0)
  let desired_factors := factors_of_180 ∩ multiples_of_15
  desired_factors.card = 6 :=
by
  sorry

end count_positive_factors_of_180_multiple_of_15_l769_769727


namespace chord_length_eq_4_l769_769970

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2 * x - 4 * y = 0
def line_eq (x y : ℝ) : Prop := x + 2 * y - 5 + Real.sqrt 5 = 0

theorem chord_length_eq_4 : 
  (∀ x y : ℝ, circle_eq x y ∧ line_eq x y) →
    (line_chord_length : ∀ (x y : ℝ), circle_eq x y ∧ line_eq x y → ℝ) = 4 :=
by
  sorry

end chord_length_eq_4_l769_769970


namespace mechanic_worked_hours_l769_769927

theorem mechanic_worked_hours (total_spent : ℕ) (cost_per_part : ℕ) (labor_cost_per_minute : ℚ) (parts_needed : ℕ) :
  total_spent = 220 → cost_per_part = 20 → labor_cost_per_minute = 0.5 → parts_needed = 2 →
  (total_spent - cost_per_part * parts_needed) / labor_cost_per_minute / 60 = 6 := by
  -- Proof will be inserted here
  sorry

end mechanic_worked_hours_l769_769927


namespace acute_angle_solution_l769_769340

-- Define the acute angle in the problem
def α := 40 * Real.pi / 180

-- Define the given equation
def eqn (θ : ℝ) : Prop :=
  Real.cos θ * (1 + Real.sqrt 3 * Real.tan (10 * Real.pi / 180)) = 1

-- The problem requires to prove that α satisfies the equation
theorem acute_angle_solution : eqn α :=
sorry

end acute_angle_solution_l769_769340


namespace evaluate_F_of_4_and_f_of_5_l769_769314

def f (a : ℤ) : ℤ := 2 * a - 2
def F (a b : ℤ) : ℤ := b^2 + a + 1

theorem evaluate_F_of_4_and_f_of_5 : F 4 (f 5) = 69 := by
  -- Definitions and intermediate steps are not included in the statement, proof is omitted.
  sorry

end evaluate_F_of_4_and_f_of_5_l769_769314


namespace find_regular_working_hours_l769_769327

noncomputable def regular_working_hours_per_day : ℕ :=
  let weekly_working_days := 5
  let hourly_wage_regular := 2.40
  let hourly_wage_overtime := 3.20
  let total_earnings := 432
  let total_hours_worked := 175
  find H such that
    20 * H * hourly_wage_regular + (total_hours_worked - 20 * H) * hourly_wage_overtime = total_earnings

theorem find_regular_working_hours : ∃ H : ℕ, 5 * 4 * H = 175 ∧ 20 * H * 2.40 + (175 - 20 * H) * 3.20 = 432 :=
sorry

end find_regular_working_hours_l769_769327


namespace a_c_sum_l769_769883

theorem a_c_sum (a b c d : ℕ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c) (h4 : d = a * b * c) (h5 : 233 % d = 79) : a + c = 13 :=
sorry

end a_c_sum_l769_769883


namespace students_less_in_each_subsequent_class_l769_769865

theorem students_less_in_each_subsequent_class
  (classes : ℕ) (total_students : ℕ) (largest_class : ℕ)
  (h_classes : classes = 5)
  (h_total_students : total_students = 140)
  (h_largest_class : largest_class = 32)
  (h_less_each_class : ∀ i : ℕ, i < classes → ∃ x : ℕ, ∀ j : ℕ, j > 0 → j < classes → x < largest_class ∧
    (largest_class - j * x : ∀ k : ℕ, k ≥ 0 → k < classes → largest_class - k * x)) :
  ∃ x, x = 2 := by
  sorry

end students_less_in_each_subsequent_class_l769_769865


namespace registration_results_count_l769_769475

-- Definitions based on the conditions in a)
def Student := ℕ
def University := ℕ
def universities : List University := [0, 1, 2] -- Representing Zhejiang, Fudan, and Shanghai Jiao Tong Universities

def C (n k : ℕ) : ℕ := Nat.choose n k

-- The main theorem based on the question and correct answer
theorem registration_results_count : 
  let C31 := C 3 1;
  let C32 := C 3 2;
  let case1 := C31 * C31;
  let case2 := C32 * C32;
  let case3 := 2 * C31 * C32;
  case1 + case2 + case3 = 36 := by
  sorry

end registration_results_count_l769_769475


namespace percentage_change_area_l769_769421

theorem percentage_change_area (L B : ℝ) :
  let A_original := L * B
  let A_new := (L / 2) * (3 * B)
  (A_new - A_original) / A_original * 100 = 50 := by
  sorry

end percentage_change_area_l769_769421


namespace part_a_impossible_part_b_possible_l769_769652

-- Statement for part (a)
theorem part_a_impossible (table : ℕ → ℕ → bool) (n : ℕ) 
    (odd_row_count : fin n → bool) (odd_col_count : fin n → bool)
    (odd_count_r : ℕ) (odd_count_c : ℕ) (cross_in_cell : fin n → fin n → bool) :
    (∀ r : fin n, odd_row_count r = odd (fin n) (\sum c : fin n, cross_in_cell r c)) →
    (∀ c : fin n, odd_col_count c = odd (fin n) (\sum r : fin n, cross_in_cell r c)) →
    odd_count_r = 20 → odd_count_c = 15 → False :=
sorry

-- Statement for part (b)
theorem part_b_possible (table : ℕ → ℕ → bool) 
    (n : ℕ) (cross_count : ℕ) (row_count : fin n → ℕ) (col_count : fin n → ℕ)
    (cross_in_cell : fin n → fin n → bool) :
    n = 16 → cross_count = 126 →
    (∀ r : fin n, odd (row_count r)) →
    (∀ c : fin n, odd (col_count c)) →
    (∃ table, (∀ r c, cross_in_cell r c = (table r c)) ∧ (\sum r, row_count r = 126) ∧ (\sum c, col_count c = 126)) :=
sorry

end part_a_impossible_part_b_possible_l769_769652


namespace count_factors_180_multiple_of_15_is_6_l769_769818

-- We define a function to check if 'n' is a factor of 'm'
def is_factor (m n : ℕ) : Prop := n ∣ m

-- We define a function to check if 'n' is a multiple of 'd'
def is_multiple (n d : ℕ) : Prop := d ∣ n

-- We define a function to collect all factors of a number 'n'
def factors (n : ℕ) : list ℕ := 
  -- We use list.filter to keep elements that satisfy is_factor n
  list.filter (λ x, is_factor n x) (list.range (n + 1))

-- We count the number of elements in a list that satisfy a predicate
def count_satisfying {α : Type*} (pred : α → Prop) [decidable_pred pred] (l : list α) : ℕ :=
  list.length (list.filter pred l)

-- Noncomputable definition to specify that counting number
noncomputable def count_positive_factors_of_180_multiple_of_15 : ℕ :=
  count_satisfying (λ x, is_multiple x 15) (factors 180)

-- We state the theorem equivalence to the proof problem
theorem count_factors_180_multiple_of_15_is_6 : count_positive_factors_of_180_multiple_of_15 = 6 :=
  sorry


end count_factors_180_multiple_of_15_is_6_l769_769818


namespace solve_E_l769_769164

-- Definitions based on the conditions provided
variables {A H S M C O E : ℕ}

-- Given conditions
def algebra_books := A
def geometry_books := H
def history_books := C
def S_algebra_books := S
def M_geometry_books := M
def O_history_books := O
def E_algebra_books := E

-- Prove that E = (AM + AO - SH - SC) / (M + O - H - C) given the conditions
theorem solve_E (h1: A ≠ H) (h2: A ≠ S) (h3: A ≠ M) (h4: A ≠ C) (h5: A ≠ O) (h6: A ≠ E)
                (h7: H ≠ S) (h8: H ≠ M) (h9: H ≠ C) (h10: H ≠ O) (h11: H ≠ E)
                (h12: S ≠ M) (h13: S ≠ C) (h14: S ≠ O) (h15: S ≠ E)
                (h16: M ≠ C) (h17: M ≠ O) (h18: M ≠ E)
                (h19: C ≠ O) (h20: C ≠ E)
                (h21: O ≠ E)
                (pos1: 0 < A) (pos2: 0 < H) (pos3: 0 < S) (pos4: 0 < M) (pos5: 0 < C)
                (pos6: 0 < O) (pos7: 0 < E) :
  E = (A * M + A * O - S * H - S * C) / (M + O - H - C) :=
sorry

end solve_E_l769_769164


namespace find_general_term_sum_first_n_terms_l769_769248

-- Given definition: sequence {a_n} with positive terms and sum S_n of its first n terms.
axiom pos_sequence (a : ℕ → ℕ) : ∀ n, 0 < a n
def Sn (a : ℕ → ℕ) (n : ℕ) : ℕ := (finset.range n).sum a
axiom arithmetic_sequence_property (a : ℕ → ℕ) (n : ℕ) : 2 * Sn a n = a n + (a n) ^ 2

-- Proof goals
theorem find_general_term (a : ℕ → ℕ) (Sn : ℕ → ℕ) (h1 : ∀ n, 0 < a n)
  (h2 : ∀ n, 2 * Sn n = a n + (a n) ^ 2) :
  ∀ n, a n = n :=
by
  sorry

theorem sum_first_n_terms (a : ℕ → ℕ) (Sn : ℕ → ℕ) (h1 : ∀ n, 0 < a n)
  (h2 : ∀ n, 2 * Sn n = a n + (a n) ^ 2) :
  ∑ k in finset.range n, (2 : ℚ) / (a k * a (k + 1)) = (2 * n) / (n + 1) :=
by
  sorry

end find_general_term_sum_first_n_terms_l769_769248


namespace problem_statement_l769_769303

noncomputable def countNs : Nat :=
  let N_values := {N : ℕ | ∃ x : ℝ, 0 < N ∧ N < 500 ∧ (x ^ Nat.floor x = N)}
  N_values.toFinset.card

theorem problem_statement :
  countNs = 287 := by
  sorry

end problem_statement_l769_769303


namespace increase_in_average_l769_769519

theorem increase_in_average (A : ℤ) (avg_after_12 : ℤ) (score_12th_inning : ℤ) (A : ℤ) : 
  score_12th_inning = 75 → avg_after_12 = 64 → (11 * A + score_12th_inning = 768) → (avg_after_12 - A = 1) :=
by
  intros h_score h_avg h_total
  sorry

end increase_in_average_l769_769519


namespace problem_solution_l769_769570

-- Let's define the probability of heads and the number of tosses
def p_heads : ℝ := 3 / 5
def tosses : ℕ := 40

-- Let P(n) be the probability of getting an even number of heads after n tosses
def even_heads_prob (n : ℕ) : ℝ :=
  if n = 0 then 1
  else 2 / 5 + 1 / 5 * even_heads_prob (n - 1)

-- Define Q(n) as the probability of getting an odd number of tails after n tosses
def odd_tails_prob (n : ℕ) : ℝ :=
  1 - even_heads_prob n

-- Define the combined probability
def combined_prob (n : ℕ) : ℝ :=
  even_heads_prob n * odd_tails_prob n

theorem problem_solution : combined_prob 40 = 1 / 4 * (1 - 1 / (5 ^ 80)) :=
  sorry

end problem_solution_l769_769570


namespace roots_of_P_on_unit_circle_l769_769394

noncomputable def polynomial_P (n : ℕ) (h : n > 0) : polynomial ℂ :=
∑ i in finset.range (2 * n + 1),
  if h' : i ≤ 2 * n then ((2 * n : ℂ) - i) * polynomial.X ^ i else 0

theorem roots_of_P_on_unit_circle (n : ℕ) (h : n > 0) :
  ∀ r : ℂ, r ∈ (polynomial_P n h).roots → |r| = 1 := 
by
  sorry

end roots_of_P_on_unit_circle_l769_769394


namespace MakarlaMeetingTimePercentage_l769_769001

def MakarlaWorkDayPercentage : ℕ → ℕ → ℕ → (ℕ → Prop) :=
  let total_work_minutes := 10 * 60
  let first_meeting_minutes := 60
  let second_meeting_minutes := 2 * first_meeting_minutes
  let third_meeting_minutes := second_meeting_minutes / 2
  let total_meeting_minutes := first_meeting_minutes + second_meeting_minutes + third_meeting_minutes
  fun p => p = (total_meeting_minutes * 100) / total_work_minutes

theorem MakarlaMeetingTimePercentage : MakarlaWorkDayPercentage 600 60 240 40 :=
  by
  sorry

end MakarlaMeetingTimePercentage_l769_769001


namespace data_grouping_l769_769860

theorem data_grouping (max_value min_value interval : ℕ) (h_max : max_value = 145) (h_min : min_value = 50) (h_interval : interval = 10) :
  let range := max_value - min_value,
      number_of_groups := (range + interval - 1) / interval
  in number_of_groups = 10 :=
by
  rw [h_max, h_min, h_interval]
  let range := 145 - 50
  let number_of_groups := (range + 10 - 1) / 10
  have range_calc : range = 95 := by norm_num
  have number_calc : number_of_groups = (95 + 10 - 1) / 10 := by rw range_calc
  have number_final : number_of_groups = 10 := by norm_num at number_calc
  exact number_final

end data_grouping_l769_769860


namespace percentage_change_area_l769_769448

theorem percentage_change_area (L B : ℝ) (hL : L > 0) (hB : B > 0) :
    let A := L * B
    let L' := L / 2
    let B' := 3 * B
    let A' := (L / 2) * (3 * B)
    (A' - A) / A * 100 = 50 :=
by
  let A := L * B
  let A' := (L / 2) * (3 * B)
  have h : (A' - A) / A * 100 = (1 / 2) * 100 := sorry
  exact h

end percentage_change_area_l769_769448


namespace parametric_to_plane_l769_769532

/-- Definition of the point and direction vectors in the parametric form -/
structure ParametricPlane where
  s t: ℝ → ℝ
  v : ℝ × ℝ × ℝ
  param_eq : v = ⟨2 + 2 * s - 3 * t, 1 - 2 * s, 4 + 3 * s + 4 * t⟩

/-- Normal vector formed from the cross product of direction vectors -/
def normal_vector : ℝ × ℝ × ℝ :=
  let v1 := (2, -2, 3)
  let v2 := (-3, 0, 4)
  cross_product v1 v2

/-- Equation of the plane satisfying: Ax + By + Cz + D = 0 -/
def plane_equation (x y z: ℝ) : ℝ :=
  8 * x + 17 * y + 6 * z - 57

/-- Statement that the parametric form defines the same plane equation -/
theorem parametric_to_plane (s t : ℝ) 
 (p : ParametricPlane) : 
  plane_equation p.v.1 p.v.2 p.v.3 = 0 := sorry

end parametric_to_plane_l769_769532


namespace sum_of_digit_permutations_1234_l769_769579

-- Define the function sum_of_all_numbers_using_digits_once
def sum_of_all_numbers_using_digits_once (d1 d2 d3 d4 : ℕ) : ℕ :=
  let digits := [d1, d2, d3, d4]
  let perms := digits.permutations
  perms.foldl (· + sum_of_digits.to_nat) 0

-- lean theorem statement
theorem sum_of_digit_permutations_1234 : 
  sum_of_all_numbers_using_digits_once 1 2 3 4 = 66660 := 
sorry

end sum_of_digit_permutations_1234_l769_769579


namespace day_365_is_Tuesday_l769_769840

def day_of_week : Type := ℕ

def Day.January_15_2005_is_Tuesday (day_of_week_n : day_of_week) : Prop :=
  day_of_week_n ≡ 2 [MOD 7]

def day_365_is_same_day_of_week (day_of_week_n day_after_n_days : day_of_week) (days_between : ℕ) : Prop :=
  (day_of_week_n + days_between) % 7 = day_after_n_days % 7

theorem day_365_is_Tuesday (day_15 : day_of_week) :
  (Day.January_15_2005_is_Tuesday day_15) →
  day_365_is_same_day_of_week day_15 day_15 350 →
  (day_15 % 7 = 2 % 7) :=
by
  intros h1 h2
  sorry

end day_365_is_Tuesday_l769_769840


namespace exists_convex_1990_gon_l769_769035

structure Polygon (n : ℕ) :=
  (angles : Fin n → ℝ)
  (side_lengths : Fin n → ℝ)

def is_equal_angles {n : ℕ} (polygon : Polygon n) : Prop :=
  ∀ (i : Fin n), polygon.angles i = (n - 2) * 180 / n

def has_side_lengths {n : ℕ} (polygon : Polygon n) (lengths : Finset ℝ) : Prop :=
  (Finset.univ.image polygon.side_lengths) = lengths

def is_convex {n : ℕ} (polygon : Polygon n) : Prop :=
  ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k → 
  polygon.angles i + polygon.angles j + polygon.angles k < 540

theorem exists_convex_1990_gon : 
  ∃ (polygon : Polygon 1990), 
    is_convex polygon ∧ 
    is_equal_angles polygon ∧ 
    has_side_lengths polygon (Finset.image (λ x: Fin 1990, x.val^2) Finset.univ) :=
sorry

end exists_convex_1990_gon_l769_769035


namespace Sn_formula_l769_769357

def pos_integers := {n : ℕ // n > 0}

def X_n (n : ℕ) : set ℕ := {i | 1 ≤ i ∧ i ≤ n}

def f (A : set ℕ) : ℕ := if h : A.nonempty then A.max' h else 0

def S_n (n : ℕ) : ℕ := ∑ A in (set.powerset (X_n n)).filter (λ A, A.nonempty), f A

theorem Sn_formula (n : pos_integers) : S_n n = (n - 1) * 2 ^ n + 1 := 
  sorry

end Sn_formula_l769_769357


namespace conference_center_people_l769_769155

theorem conference_center_people (rooms : ℕ) (capacity_per_room : ℕ) (occupancy_fraction : ℚ) (total_people : ℕ) :
  rooms = 6 →
  capacity_per_room = 80 →
  occupancy_fraction = 2/3 →
  total_people = rooms * capacity_per_room * occupancy_fraction →
  total_people = 320 := 
by
  intros h_rooms h_capacity h_fraction h_total
  rw [h_rooms, h_capacity, h_fraction] at h_total
  norm_num at h_total
  exact h_total

end conference_center_people_l769_769155


namespace factors_of_180_l769_769774

-- Define a predicate to check if a number is a factor of another number
def is_factor (m n : ℕ) : Prop := n % m = 0

-- Define a predicate to check if a number is a multiple of another number
def is_multiple (m n : ℕ) : Prop := m % n = 0

-- Define the problem
theorem factors_of_180 : {n : ℕ // is_factor n 180 ∧ is_multiple n 15}.card = 6 :=
by sorry

end factors_of_180_l769_769774


namespace probability_of_correct_dialing_l769_769886

-- Given conditions
def first_three_digits : Finset ℕ := {307, 308, 312}
def last_four_digits_permutations : Finset (List ℕ) :=
  {list.permutations [0, 2, 6, 8]}

-- Total number of possible combinations
def total_combinations : ℕ := first_three_digits.card * last_four_digits_permutations.card

-- Correct probability
def correct_probability : ℚ := 1 / total_combinations

-- Theorem to be proved
theorem probability_of_correct_dialing :
  correct_probability = 1 / 72 := by {
  sorry
}

end probability_of_correct_dialing_l769_769886
